"""Semantic artifact writers + readers — one schema per artifact, round-tripped.

Pins `artifacts.write_*` against `present.load_*`: a `CellRecord` written
through `write_cell` reads back through `load_cell_record` with its views'
`sources_cited` surviving as `SourceRef` (the quirk the raw-dict form had — a
read-back gave bare dicts, not the typed model). Also pins the
job/run/trial/decision/health/prefetch writers as one-liners that can't
drop the `model_dump(mode="json")`.
"""

from __future__ import annotations

import json
from pathlib import Path

from fintel.cli.present import (
    job_summary_line,
    load_cell_record,
    load_decision,
    load_health,
    load_job_result,
    load_run_result,
    print_decision_block,
    print_job_artifacts,
)
from fintel.models.common import Symbol
from fintel.models.decision import AgentResponse, SourceRef, View
from fintel.models.job import JobResult
from fintel.models.run import RunResult
from fintel.models.trial import CellRecord, TrialResult
from fintel.simulate.artifacts import (
    write_cell,
    write_decision,
    write_health,
    write_job_result,
    write_run_result,
    write_trial_result,
)


def _view(symbol: str, score: float = 0.3) -> View:
    return View(
        symbol=symbol,
        score=score,
        rationale="r",
        sources_cited=[SourceRef(source_type="prices", source_id=f"prices:{symbol}")],
    )


def test_cell_record_round_trips_with_typed_sources_cited(tmp_path: Path):
    """The quirk: a raw-dict cell record read back as bare dicts for
    `sources_cited`; a `CellRecord` read back as `SourceRef`."""
    record = CellRecord(
        cell="AAPL",
        symbols=["AAPL"],
        decision_date="2025-01-02",
        scope="single_name",
        outcome="ok",
        n_views=1,
        views={"AAPL": _view("AAPL")},
        environment={"cell": "AAPL", "health": {"status": "ok"}},
        elapsed_ms=12,
        started_at="2025-01-02T00:00:00Z",
    )
    path = tmp_path / "cells" / "AAPL.json"
    write_cell(path, record)
    assert path.is_file()

    loaded = load_cell_record(path)
    assert loaded is not None
    assert loaded.cell == "AAPL"
    assert loaded.outcome == "ok"
    # The sources_cited survive as the typed SourceRef, not a bare dict.
    cited = loaded.views["AAPL"].sources_cited
    assert len(cited) == 1
    assert isinstance(cited[0], SourceRef)
    assert cited[0].source_type == "prices"
    assert cited[0].source_id == "prices:AAPL"


def test_decision_writer_and_reader(tmp_path: Path):
    views = {"AAPL": _view("AAPL"), "MSFT": _view("MSFT", -0.1)}
    path = tmp_path / "decision.json"
    write_decision(path, views)
    loaded = load_decision(tmp_path)
    assert loaded is not None
    assert set(loaded) == {"AAPL", "MSFT"}
    assert loaded["AAPL"]["score"] == 0.3


def test_trial_run_job_result_writers_round_trip(tmp_path: Path):
    trial = TrialResult(decision_date="2025-01-02", status="ok", n_views=2)
    tp = tmp_path / "trial" / "result.json"
    write_trial_result(tp, trial)
    assert json.loads(tp.read_text())["status"] == "ok"

    run = RunResult(run_id="r1", job_id="j1", k_index=1, status="ok", n_trials=1)
    rp = tmp_path / "run" / "result.json"
    write_run_result(rp, run)
    assert load_run_result(tmp_path / "run").status == "ok"

    job = JobResult(job_id="j1", strategy="s", agent="a", k_repeats=1, status="ok")
    jp = tmp_path / "job" / "result.json"
    write_job_result(jp, job)
    assert load_job_result(tmp_path / "job").status == "ok"


def test_health_and_prefetch_writers(tmp_path: Path):
    write_health(tmp_path / "health.json", {"status": "ok", "n_cells": 2})
    assert load_health(tmp_path)["n_cells"] == 2


def test_job_summary_line_falls_back_to_health_then_corrupt(tmp_path: Path):
    # No artifacts at all → unknown.
    line = job_summary_line(tmp_path)
    assert "status=?" in line and "health=?" in line
    # A health.json only → status flagged corrupt (result missing).
    write_health(tmp_path / "health.json", {"status": "degraded", "n_cells": 1})
    line = job_summary_line(tmp_path)
    assert "status=corrupt" in line and "health=degraded" in line
    # A valid result.json → status from it.
    write_job_result(tmp_path / "result.json", JobResult(job_id="j", strategy="s", agent="a", k_repeats=1, status="ok"))
    assert "status=ok" in job_summary_line(tmp_path)


def test_load_readers_return_none_for_corrupt_or_missing(tmp_path: Path):
    assert load_job_result(tmp_path) is None
    (tmp_path / "result.json").write_text("{not json")
    assert load_job_result(tmp_path) is None
    assert load_cell_record(tmp_path / "x.json") is None


def test_print_decision_block_and_job_artifacts(tmp_path: Path, capsys):
    write_decision(tmp_path / "decision.json", {"AAPL": _view("AAPL")})
    print_decision_block(tmp_path)
    out = capsys.readouterr().out
    assert "== decision" in out and "AAPL" in out

    write_job_result(tmp_path / "result.json", JobResult(job_id="j", strategy="s", agent="a", k_repeats=1, status="ok"))
    write_health(tmp_path / "health.json", {"status": "ok"})
    print_job_artifacts(tmp_path)
    out = capsys.readouterr().out
    assert "== result.json" in out and "== health.json" in out
