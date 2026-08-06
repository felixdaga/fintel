"""Tests for the live nerve dashboard (fintel.cli.watch)."""

from __future__ import annotations

from pathlib import Path

from fintel.cli.watch import _apply, _Run, resolve_paths


def _feed(run: _Run, events: list[dict]) -> None:
    import time

    now = time.monotonic()
    for e in events:
        _apply(run, e, now)


def test_apply_tracks_cell_stage_and_tool_calls():
    run = _Run(tag="jan", path=Path("x"))
    _feed(
        run,
        [
            {"event": "run_start"},
            {"event": "probe_kind", "kind": "prices", "result": "ok"},
            {"event": "cell_start", "cell": "NVDA", "decision_date": "2026-01-02"},
            {
                "event": "agent_stage",
                "cell": "NVDA",
                "decision_date": "2026-01-02",
                "stage": "reasoning",
                "round": 1,
                "text": "thinking",
            },
            {
                "event": "agent_tool_call",
                "cell": "NVDA",
                "decision_date": "2026-01-02",
                "tool": "fintel__get_prices",
            },
            {
                "event": "agent_tool_result",
                "cell": "NVDA",
                "decision_date": "2026-01-02",
                "tool": "fintel__get_prices",
                "ok": True,
            },
            {
                "event": "agent_tool_call",
                "cell": "NVDA",
                "decision_date": "2026-01-02",
                "tool": "fintel__get_news",
            },
            {
                "event": "agent_tool_result",
                "cell": "NVDA",
                "decision_date": "2026-01-02",
                "tool": "fintel__get_news",
                "ok": False,
            },
            {
                "event": "cell_done",
                "cell": "NVDA",
                "decision_date": "2026-01-02",
                "outcome": "ok",
                "health": "ok",
                "n_reads": 2,
                "elapsed_ms": 10,
            },
            {"event": "run_done", "status": "ok"},
        ],
    )
    assert run.status == "ok" and run.done
    assert run.probes == {"prices": "ok"}
    c = run.cells["NVDA@2026-01-02"]
    assert c.stage == "done" and c.outcome == "ok" and c.reads == 2
    assert c.calls == 2 and c.errors == 1
    assert c.last_tool == "fintel__get_news"


def test_render_produces_a_frame_with_run_and_cell():
    run = _Run(tag="jan", path=Path("x"))
    _feed(
        run,
        [
            {"event": "run_start"},
            {"event": "cell_start", "cell": "NVDA", "decision_date": "2026-01-02"},
            {
                "event": "agent_tool_call",
                "cell": "NVDA",
                "decision_date": "2026-01-02",
                "tool": "fintel__get_prices",
            },
            {
                "event": "cell_done",
                "cell": "NVDA",
                "decision_date": "2026-01-02",
                "outcome": "ok",
                "health": "ok",
                "n_reads": 1,
                "elapsed_ms": 5,
            },
            {"event": "run_done", "status": "ok"},
        ],
    )
    import time

    from fintel.cli.watch import _render_lines

    frame = "\n".join(_render_lines([run], time.monotonic(), collapse_done=False))
    assert "fintel nerve" in frame
    assert "jan" in frame and "NVDA" in frame
    assert "2026-01-02" in frame
    assert "done:ok" in frame
    assert "press q" in frame and "to quit" in frame


def test_apply_disambiguates_same_symbol_across_dates():
    """A multi-date run has NVDA@Jan and NVDA@Apr as two distinct cells — they
    must not collide in the dashboard (the reason decision_date is on every
    per-cell event)."""
    run = _Run(tag="multi", path=Path("x"))
    _feed(
        run,
        [
            {"event": "run_start"},
            {"event": "cell_start", "cell": "NVDA", "decision_date": "2026-01-02"},
            {
                "event": "agent_tool_call",
                "cell": "NVDA",
                "decision_date": "2026-01-02",
                "tool": "fintel__get_prices",
            },
            {
                "event": "cell_done",
                "cell": "NVDA",
                "decision_date": "2026-01-02",
                "outcome": "ok",
                "health": "ok",
                "n_reads": 1,
                "elapsed_ms": 5,
            },
            {"event": "cell_start", "cell": "NVDA", "decision_date": "2026-04-01"},
            {
                "event": "agent_stage",
                "cell": "NVDA",
                "decision_date": "2026-04-01",
                "stage": "reasoning",
                "round": 1,
                "text": "apr thinking",
            },
            {"event": "run_done", "status": "ok"},
        ],
    )
    assert set(run.cells) == {"NVDA@2026-01-02", "NVDA@2026-04-01"}
    jan = run.cells["NVDA@2026-01-02"]
    apr = run.cells["NVDA@2026-04-01"]
    assert jan.stage == "done" and jan.calls == 1
    assert apr.stage == "reasoning" and apr.round == 1 and apr.calls == 0


def test_apply_shows_adapter_defined_stage_label():
    """TUI displays whatever stage string the adapter emitted — no remapping."""
    run = _Run(tag="opt", path=Path("x"))
    _feed(
        run,
        [
            {"event": "run_start"},
            {"event": "cell_start", "cell": "AAPL", "decision_date": "2025-01-01"},
            {
                "event": "agent_stage",
                "cell": "AAPL",
                "decision_date": "2025-01-01",
                "stage": "quantitative_specialist",
            },
            {
                "event": "agent_stage",
                "cell": "AAPL",
                "decision_date": "2025-01-01",
                "stage": "independent_verifier",
                "text": "checks ok",
            },
        ],
    )
    c = run.cells["AAPL@2025-01-01"]
    assert c.stage == "independent_verifier"
    assert c.last_reason == "checks ok"


def test_resolve_paths_handles_job_ids_and_run_log(tmp_path: Path):
    job = tmp_path / "jobA"
    (job / "r1").mkdir(parents=True)
    (job / "r1" / "run.log").write_text("{}\n")
    paths = resolve_paths(["jobA"], tmp_path)
    assert paths == [job / "r1" / "run.log"]
    # literal run.log path passes through
    direct = resolve_paths([str(job / "r1" / "run.log")], tmp_path)
    assert direct == [job / "r1" / "run.log"]
