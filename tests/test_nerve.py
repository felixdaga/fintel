"""`Nerve` — the single emit surface. One schema, two sinks (terminal + run.log).

Covers the consolidation: every event goes to both sinks, the log file is
durable JSONL, and `verbose=False` silences the terminal line without losing
the log. Also pins the renderer for the new probe events.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from fintel.environment.nerve import Nerve


def _read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_emit_writes_jsonl_to_run_log(tmp_path: Path):
    nerve = Nerve(run_root=tmp_path, stream=io.StringIO(), verbose=False)
    nerve.emit("job_start", job_id="j1", strategy="s", agent="a", k_repeats=1)
    nerve.emit("probe_ok", n_ok=2, n_kinds=2)
    log = _read_log(tmp_path / "run.log")
    assert [e["event"] for e in log] == ["job_start", "probe_ok"]
    assert log[0]["job_id"] == "j1"
    assert log[1]["n_ok"] == 2


def test_emit_prints_terminal_line_when_verbose(tmp_path: Path):
    out = io.StringIO()
    nerve = Nerve(run_root=tmp_path, stream=out, verbose=True)
    nerve.emit("job_start", job_id="j1", strategy="s", agent="a", k_repeats=1)
    line = out.getvalue().strip()
    assert "job j1" in line and "strategy=s" in line


def test_emit_silent_when_not_verbose_but_log_still_written(tmp_path: Path):
    out = io.StringIO()
    nerve = Nerve(run_root=tmp_path, stream=out, verbose=False)
    nerve.emit("cell_start", cell="AAPL")
    assert out.getvalue() == ""  # no terminal line
    log = _read_log(tmp_path / "run.log")
    assert log[0]["event"] == "cell_start"  # but the log recorded it


def test_probe_event_rendering(tmp_path: Path):
    out = io.StringIO()
    nerve = Nerve(run_root=tmp_path, stream=out, verbose=True)
    nerve.emit("probe_start", kinds=["news", "prices"], symbol="AAPL", timeout_s=15)
    nerve.emit(
        "probe_kind", kind="news", source="massive_news", status="ok", latency_ms=12.3, n=366
    )
    nerve.emit(
        "probe_kind",
        kind="prices",
        source="massive_prices",
        status="failed",
        latency_ms=5.0,
        n=None,
    )
    nerve.emit("probe_failed", n_failed=1, n_kinds=2, failed_kinds=["prices"])
    lines = out.getvalue().splitlines()
    assert any("probe" in l and "kinds=" in l for l in lines)
    assert any("probe news" in l and "[ok]" in l for l in lines)
    assert any("probe prices" in l and "[FAIL]" in l for l in lines)
    assert any("probe FAILED" in l and "prices" in l for l in lines)


def test_agent_stage_event_rendering(tmp_path: Path):
    out = io.StringIO()
    nerve = Nerve(run_root=tmp_path, stream=out, verbose=True)
    nerve.emit(
        "agent_stage",
        cell="AAPL",
        stage="thinking",
        round=3,
        text="comparing P/E ratios across peers",
    )
    line = out.getvalue().strip()
    assert "AAPL" in line and "thinking" in line and "r3" in line
    assert "comparing P/E" in line  # text snippet rendered


def test_unknown_event_falls_back_to_generic_line(tmp_path: Path):
    # A new emit site must not silently vanish from the terminal — it shows
    # plain until given a dedicated format.
    out = io.StringIO()
    nerve = Nerve(run_root=tmp_path, stream=out, verbose=True)
    nerve.emit("some_new_event", cell="MSFT", foo="bar")
    line = out.getvalue().strip()
    assert "some_new_event" in line and "foo=bar" in line


def test_nerve_emits_run_grid_and_agent_stalled_from_staging_events(tmp_path: Path):
    """Feeding staging events through the nerve produces a throttled `run_grid`
    snapshot in run.log, and a cell that goes quiet past the stall threshold
    produces an `agent_stalled` event — the holistic run-level view."""
    out = io.StringIO()
    nerve = Nerve(
        run_root=tmp_path,
        stream=out,
        verbose=False,
        grid_interval_s=0.0,  # emit a grid on every staging event
        stall_threshold_s=60.0,
    )
    nerve.emit("cell_start", cell="AAPL")
    nerve.emit("agent_stage", cell="AAPL", stage="reasoning", round=1)
    nerve.emit("agent_tool_call", cell="AAPL", tool="get_prices")
    nerve.emit("cell_start", cell="MSFT")
    nerve.emit("agent_stage", cell="MSFT", stage="reasoning", round=1)
    # MSFT goes silent; force its last-event time into the past.
    nerve.tracker._state["MSFT"].last_event_mono -= 100.0
    # Any later staging event triggers a stall sweep.
    nerve.emit("agent_tool_result", cell="AAPL", tool="get_prices", ok=True)

    log = [json.loads(l) for l in (tmp_path / "run.log").read_text().splitlines() if l.strip()]
    events = [e["event"] for e in log]
    assert "run_grid" in events
    assert "agent_stalled" in events
    stalled = [e for e in log if e["event"] == "agent_stalled"][0]
    assert stalled["cell"] == "MSFT"
    grid = [e for e in log if e["event"] == "run_grid"][-1]
    assert "AAPL" in grid["grid"] and "MSFT" in grid["grid"]
