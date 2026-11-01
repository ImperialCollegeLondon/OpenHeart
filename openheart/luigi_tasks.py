"""Run a pipeline under the Luigi scheduler (``pip install openheart[luigi]``).

Each (case, step) pair becomes one ``openheart.Step`` task that requires the same
case's previous step. A task is complete when the case's ``case.json`` records
that step's signature, the same checkpoint the built-in runner uses, so Luigi
and the local runner can resume each other's work.

With ``jobs > 1`` Luigi runs every task in a fresh worker process, so models are
reloaded per task; ``jobs=1`` keeps one process and loads each model once.
``scheduler="luigi-central"`` uses the central ``luigid`` set in luigi.cfg (web
UI, cross-machine locking) instead of a local scheduler.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import luigi

from .layouts import RECORD
from .pipeline import CaseResult, CohortResult, _record, apply_step

log = logging.getLogger(__name__)

_CONTEXT: dict = {}  # set by run_with_luigi; forked Luigi workers inherit it


class Step(luigi.Task):
    task_namespace = "openheart"
    case_id = luigi.Parameter()
    index = luigi.IntParameter()

    def requires(self):
        return [Step(case_id=self.case_id, index=self.index - 1)] if self.index > 0 else []

    def output(self):
        return luigi.LocalTarget(str(_CONTEXT["out_dir"] / self.case_id / RECORD))

    def complete(self):
        ctx = _CONTEXT
        rec = _record(ctx["out_dir"] / self.case_id)
        if not ctx["resume"] and rec.get("run_id") != ctx["run_id"]:
            return False
        n = self.index + 1
        if [h["step"] for h in rec.get("history", [])[:n]] != ctx["signatures"][:n]:
            return False
        return n < len(ctx["signatures"]) or rec.get("status") == "ok"

    def run(self):
        ctx = _CONTEXT
        case_dir = ctx["out_dir"] / self.case_id
        pipeline = ctx["pipeline"]
        saved = (ctx["out_layout"].read(case_dir, case_id=self.case_id)
                 if self.index > 0 and (case_dir / RECORD).exists() else None)
        case, _ = pipeline.state_at(self.index, ctx["in_layout"], ctx["folders"][self.case_id],
                                 self.case_id, saved)
        try:
            case = apply_step(pipeline.steps[self.index], case)
        except Exception as e:
            case.save(case_dir, ctx["out_layout"], status="failed",
                      error=f"{type(e).__name__}: {e}", run_id=ctx["run_id"])
            raise
        last = self.index == len(ctx["signatures"]) - 1
        case.save(case_dir, ctx["out_layout"], status="ok" if last else "running", run_id=ctx["run_id"])


def run_with_luigi(pipeline, cohort, out_dir: Path, out_layout, jobs=1, resume=True,
                   central=False) -> CohortResult:
    folders = {cohort.case_id(f): f for f in cohort.folders}
    _CONTEXT.update(pipeline=pipeline, out_dir=Path(out_dir), in_layout=cohort.layout,
                    out_layout=out_layout, folders=folders, resume=resume,
                    signatures=[s.signature for s in pipeline.steps], run_id=uuid.uuid4().hex)
    last = len(pipeline.steps) - 1
    luigi.build([Step(case_id=cid, index=last) for cid in folders], workers=jobs,
                local_scheduler=not central, no_lock=True, log_level="WARNING")
    results = CohortResult()
    for cid in folders:
        rec = _record(Path(out_dir) / cid)
        done = ([h["step"] for h in rec.get("history", [])] == _CONTEXT["signatures"]
                and rec.get("status") == "ok")
        results.append(CaseResult(cid, "ok" if done else "failed",
                                  error=None if done else rec.get("error", "not completed"),
                                  metrics=rec.get("metrics", {}), info=rec.get("info", {})))
    return results
