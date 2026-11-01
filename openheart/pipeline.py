"""Pipelines: an ordered list of steps run over one case or a whole cohort.

Running over a cohort with an output directory checkpoints every case after every
step (the case folder plus its ``case.json``, which records the signatures of the
steps applied). Re-running the same command resumes each case from its first
unfinished step; changing a step's parameters changes its signature, so that step
and everything after it are recomputed. One failing case is recorded and skipped,
never fatal to the run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

from .case import Case, Cohort
from .layouts import CONTAINERS, RECORD, get_layout

log = logging.getLogger(__name__)


class Step:
    """One stage of a pipeline: takes a :class:`Case` and returns it with more filled in.

    Subclasses implement :meth:`process`. Expensive resources (models) are created
    in :meth:`setup`, which runs once per process before the first case and is never
    pickled. Public attributes are the step's parameters: they appear in its
    ``signature`` (used for checkpointing) except those listed in ``volatile``.
    ``writes`` names the parts of a case the step changes; resuming uses it to undo
    the work of steps that are no longer in the pipeline or whose parameters changed.
    """

    name = "step"
    volatile: tuple = ()  # parameters that do not change results (e.g. device)
    writes: tuple = CONTAINERS  # conservative default for custom steps

    def setup(self) -> None:
        """Load models or other heavy resources. Called lazily, once per process."""

    def process(self, case: Case) -> Case:
        raise NotImplementedError

    def params(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    @property
    def signature(self) -> str:
        stable = {k: v for k, v in self.params().items() if k not in self.volatile}
        digest = hashlib.sha1(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()
        return f"{self.name}:{digest[:8]}"

    def __call__(self, case: Case) -> Case:
        if not getattr(self, "_ready", False):
            self.setup()
            self._ready = True
        return self.process(case)

    def __getstate__(self):
        return self.params()  # drop loaded models; the receiving process re-runs setup

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({args})"


@dataclass
class CaseResult:
    """Outcome of one case in a cohort run."""

    id: str
    status: str  # "ok" | "failed"
    error: str | None = None
    metrics: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    seconds: float = 0.0
    resumed_from: int = 0  # number of steps restored from a checkpoint

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class CohortResult(list):
    """A list of :class:`CaseResult`, one per case, with table helpers."""

    @property
    def failed(self) -> list:
        return [r for r in self if not r.ok]

    def rows(self) -> list:
        return [{"case_id": r.id, "status": r.status, "error": r.error or "", **r.metrics} for r in self]

    def to_dataframe(self):
        """One row per case: ``case_id``, ``status``, ``error``, then every metric."""
        import pandas as pd

        return pd.DataFrame(self.rows())

    def to_csv(self, path) -> None:
        rows = self.rows()
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def __repr__(self) -> str:
        return f"CohortResult({len(self)} cases, {len(self) - len(self.failed)} ok, {len(self.failed)} failed)"


class Pipeline:
    """An ordered list of steps. Call it on a :class:`Case`, or :meth:`run` it on a cohort."""

    def __init__(self, steps):
        self.steps = list(steps)

    def __call__(self, case: Case) -> Case:
        for step in self.steps:
            case = apply_step(step, case)
        return case

    def __repr__(self) -> str:
        return "Pipeline([\n" + "".join(f"    {s!r},\n" for s in self.steps) + "])"

    def resume_point(self, history: list) -> int:
        """How many leading steps a saved ``history`` already covers."""
        n = 0
        for step, done in zip(self.steps, history):
            if step.signature != done["step"]:
                break
            n += 1
        return n

    def state_at(self, k: int, in_layout, folder, case_id, saved: Case | None = None):
        """The case as it is after this pipeline's first ``k`` steps, and whether it came
        from the checkpoint.

        Uses the checkpoint ``saved`` when its history covers those steps and the work
        of any later (now obsolete) steps can be undone exactly; otherwise replays
        the steps from the source folder.
        """
        if saved is not None and self.resume_point(saved.history) >= k:
            case = rollback(saved, k)
            if case is not None:
                return case, True
        case = in_layout.read(folder, case_id=case_id)
        # A source folder is fresh input, even if it is the output of an earlier run.
        case.history, case.metrics, case.features = [], {}, {}
        case.origin = [c for c in CONTAINERS if getattr(case, c)]
        for step in self.steps[:k]:
            case = apply_step(step, case)
        return case, False

    def run(self, cohort, out_dir=None, *, layout="openheart", jobs: int = 1,
            resume: bool = True, scheduler: str = "local") -> CohortResult:
        """Run every case of ``cohort`` (a :class:`Cohort` or a folder path).

        ``out_dir``: write each case to ``out_dir/<case_id>`` in ``layout`` (also the
        checkpoint), plus ``out_dir/metrics.csv``. Without it nothing is written and
        nothing can be resumed. ``jobs``: worker processes. ``scheduler="luigi"`` (or
        ``"luigi-central"``) hands the same work to Luigi (see :mod:`openheart.luigi_tasks`).
        """
        if not isinstance(cohort, Cohort):
            cohort = Cohort.load(cohort)
        out_dir = Path(out_dir) if out_dir is not None else None
        out_layout = get_layout(layout)
        if scheduler in ("luigi", "luigi-central"):
            if out_dir is None:
                raise ValueError("the luigi scheduler needs out_dir (its checkpoints live there)")
            from .luigi_tasks import run_with_luigi

            results = run_with_luigi(self, cohort, out_dir, out_layout, jobs=jobs, resume=resume,
                                     central=scheduler == "luigi-central")
        elif scheduler == "local":
            results = self._run_local(cohort, out_dir, out_layout, jobs, resume)
        else:
            raise ValueError(f"unknown scheduler {scheduler!r}; use 'local', 'luigi' or 'luigi-central'")
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            results.to_csv(out_dir / "metrics.csv")
        log.info("%r", results)
        return results

    def _run_local(self, cohort, out_dir, out_layout, jobs, resume) -> CohortResult:
        tasks = [(cohort.case_id(f), f) for f in cohort.folders]
        results = CohortResult()
        total = len(tasks)

        def report(res):
            results.append(res)
            extra = f" (resumed after {res.resumed_from} steps)" if res.resumed_from else ""
            msg = f"[{len(results)}/{total}] {res.id}: {res.status}{extra} {res.seconds:.1f}s"
            (log.info if res.ok else log.warning)(msg + (f" — {res.error}" if res.error else ""))

        if jobs <= 1:
            for case_id, folder in tasks:
                report(run_case(self, cohort.layout, folder, case_id, out_dir, out_layout, resume))
        else:
            with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn"),
                                     initializer=_init_worker, initargs=(self,)) as pool:
                futures = [pool.submit(_worker_run_case, cohort.layout, folder, case_id,
                                       out_dir, out_layout, resume) for case_id, folder in tasks]
                for future in as_completed(futures):
                    report(future.result())
        order = {case_id: i for i, (case_id, _) in enumerate(tasks)}
        results.sort(key=lambda r: order[r.id])
        return results


def run_case(pipeline: Pipeline, in_layout, folder, case_id, out_dir, out_layout, resume) -> CaseResult:
    """Run (or resume) one case and checkpoint it after every step. Never raises."""
    t0 = time.perf_counter()
    case_dir = out_dir / case_id if out_dir is not None else None
    case, start = None, 0
    try:
        saved = None
        if resume and case_dir is not None and (case_dir / RECORD).exists():
            saved = out_layout.read(case_dir, case_id=case_id)
            start = pipeline.resume_point(saved.history)
            if (start == len(pipeline.steps) == len(saved.history)
                    and _record(case_dir).get("status") == "ok"):
                return CaseResult(case_id, "ok", metrics=saved.metrics, info=saved.info,
                                  seconds=time.perf_counter() - t0, resumed_from=start)
        case, restored = pipeline.state_at(start, in_layout, folder, case_id, saved)
        start = len(case.history) if restored else 0  # steps restored, for reporting
        for step in pipeline.steps[len(case.history):]:
            case = apply_step(step, case)
            if case_dir is not None:
                case.save(case_dir, out_layout, status="running")
        if case_dir is not None:
            case.save(case_dir, out_layout, status="ok")
        return CaseResult(case_id, "ok", metrics=case.metrics, info=case.info,
                          seconds=time.perf_counter() - t0, resumed_from=start)
    except Exception as e:  # one bad case must not stop the cohort
        error = f"{type(e).__name__}: {e}"
        log.debug("case %s failed:\n%s", case_id, traceback.format_exc())
        if case_dir is not None and case is not None:
            try:
                case.save(case_dir, out_layout, status="failed", error=error)
            except Exception:
                log.debug("could not save failed case %s", case_id, exc_info=True)
        return CaseResult(case_id, "failed", error=error,
                          metrics=case.metrics if case is not None else {},
                          seconds=time.perf_counter() - t0, resumed_from=start)


def apply_step(step: Step, case: Case) -> Case:
    """Run one step and append it (signature, parts written, metrics added) to the history."""
    before = set(case.metrics)
    case = step(case)
    case.history.append({"step": step.signature, "writes": list(step.writes),
                         "metrics": sorted(set(case.metrics) - before)})
    return case


def rollback(case: Case, k: int):
    """Undo the history entries after the first ``k``; ``None`` if that cannot be exact.

    Metrics are removed by name. Any other part written by an undone step is
    cleared, which is exact only when neither the source folder nor a kept step
    also provided it; otherwise there is no record of the earlier version.
    """
    kept, undone = case.history[:k], case.history[k:]
    if not undone:
        return case
    provided = set(case.origin).union(*(h["writes"] for h in kept))
    cleared = set().union(*(h["writes"] for h in undone)) - {"metrics"}
    if cleared & provided:
        return None
    for h in undone:
        for key in h["metrics"]:
            case.metrics.pop(key, None)
    for part in cleared:
        setattr(case, part, {})
    case.history = kept
    return case


def _record(case_dir: Path) -> dict:
    try:
        return json.loads((case_dir / RECORD).read_text())
    except (OSError, ValueError):
        return {}


_WORKER_PIPELINE = None


def _init_worker(pipeline):
    global _WORKER_PIPELINE
    logging.basicConfig(level=logging.WARNING)
    _WORKER_PIPELINE = pipeline


def _worker_run_case(in_layout, folder, case_id, out_dir, out_layout, resume):
    return run_case(_WORKER_PIPELINE, in_layout, folder, case_id, out_dir, out_layout, resume)
