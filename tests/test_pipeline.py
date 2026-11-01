"""Pipeline runner: history, checkpoint/resume, rollback, failure isolation, parallelism."""

import csv
import json

import pytest

from openheart import Cohort, Measure, Pipeline, Step


class Count(Step):
    """Adds one metric; counts how often it really ran (single-process tests only)."""

    name = "count"
    writes = ("metrics",)
    calls = 0

    def __init__(self, key="n", value=1):
        self.key, self.value = key, value

    def process(self, case):
        Count.calls += 1
        case.metrics[self.key] = self.value
        return case


class Explode(Step):
    name = "explode"
    writes = ("metrics",)

    def __init__(self, bad_id):
        self.bad_id = bad_id

    def process(self, case):
        if case.id == self.bad_id:
            raise RuntimeError("boom")
        return case


@pytest.fixture(autouse=True)
def reset_counter():
    Count.calls = 0


def rows(out):
    return {r["case_id"]: r for r in csv.DictReader(open(out / "metrics.csv"))}


def test_signature_tracks_params_but_not_volatile():
    from openheart import FitMesh

    assert Count("a").signature != Count("b").signature
    assert FitMesh(device="cpu").signature == FitMesh(device="cuda").signature


def test_call_records_history(case):
    case = Pipeline([Count("a"), Count("b", 2)])(case)
    assert [h["step"] for h in case.history] == [Count("a").signature, Count("b", 2).signature]
    assert case.history[1]["metrics"] == ["b"]


def test_run_writes_cases_and_table(cohort_dir, tmp_path):
    out = tmp_path / "out"
    result = Pipeline([Measure()]).run(Cohort.load(cohort_dir), out)
    assert len(result) == 3 and not result.failed
    assert set(rows(out)) == {"site_a/p1", "site_a/p2", "site_b/p3"}
    assert json.loads((out / "site_b/p3/case.json").read_text())["status"] == "ok"
    assert list(result.to_dataframe().columns[:3]) == ["case_id", "status", "error"]


def test_identical_rerun_is_skipped(cohort_dir, tmp_path):
    pipe = Pipeline([Count("a"), Count("b")])
    pipe.run(cohort_dir, tmp_path / "out")
    assert Count.calls == 6
    result = pipe.run(cohort_dir, tmp_path / "out")
    assert Count.calls == 6 and all(r.resumed_from == 2 for r in result)


def test_changed_step_reruns_from_there_and_drops_stale_metrics(cohort_dir, tmp_path):
    out = tmp_path / "out"
    Pipeline([Count("a"), Count("old")]).run(cohort_dir, out)
    Count.calls = 0
    result = Pipeline([Count("a"), Count("new")]).run(cohort_dir, out)
    assert Count.calls == 3  # only the changed second step, once per case
    assert all(r.resumed_from == 1 for r in result)
    table = rows(out)["site_a/p1"]
    assert "new" in table and "old" not in table


def test_no_resume_recomputes(cohort_dir, tmp_path):
    pipe = Pipeline([Count("a")])
    pipe.run(cohort_dir, tmp_path / "out")
    pipe.run(cohort_dir, tmp_path / "out", resume=False)
    assert Count.calls == 6


def test_one_failing_case_does_not_stop_the_run(cohort_dir, tmp_path):
    out = tmp_path / "out"
    result = Pipeline([Count("a"), Explode("site_a/p2")]).run(cohort_dir, out)
    assert [r.id for r in result.failed] == ["site_a/p2"]
    assert "boom" in result.failed[0].error
    assert json.loads((out / "site_a/p2/case.json").read_text())["status"] == "failed"
    Count.calls = 0
    result = Pipeline([Count("a"), Explode("nobody")]).run(cohort_dir, out)
    assert not result.failed and Count.calls == 0  # step 1 restored from the checkpoint


def test_parallel_workers_match_serial(cohort_dir, tmp_path):
    serial = Pipeline([Measure()]).run(cohort_dir, tmp_path / "a")
    parallel = Pipeline([Measure()]).run(cohort_dir, tmp_path / "b", jobs=2)
    assert [r.metrics for r in serial] == [r.metrics for r in parallel]


def test_luigi_scheduler_shares_checkpoints_with_local_runner(cohort_dir, tmp_path):
    pytest.importorskip("luigi")
    out = tmp_path / "out"
    result = Pipeline([Measure()]).run(cohort_dir, out, scheduler="luigi")
    assert len(result) == 3 and not result.failed
    again = Pipeline([Measure()]).run(cohort_dir, out)  # local runner picks up luigi's work
    assert all(r.resumed_from == 1 for r in again)
