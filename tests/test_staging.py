"""Live agent staging — the run-level view of every cell's stage.

Pins the `StageTracker`: it folds staging events into per-cell state, flags
cells that go quiet too long (stuck heuristic), flags cells making too many
tool calls (the 'browsing forever' signal), and renders a one-line-per-cell
grid. The nerve owns emission; this module owns the accumulated state.
"""

from __future__ import annotations

from fintel.environment.staging import StageTracker


def test_tracker_folds_staging_events_into_per_cell_state():
    t = StageTracker()
    t.update("cell_start", {"cell": "AAPL"})
    t.update("agent_stage", {"cell": "AAPL", "stage": "reasoning", "round": 1})
    t.update("agent_tool_call", {"cell": "AAPL", "tool": "get_prices", "args": "{}"})
    t.update("agent_tool_result", {"cell": "AAPL", "tool": "get_prices", "ok": True})
    t.update("agent_tool_call", {"cell": "AAPL", "tool": "get_news", "args": "{}"})
    t.update("agent_tool_result", {"cell": "AAPL", "tool": "get_news", "ok": False})
    t.update("cell_done", {"cell": "AAPL", "outcome": "ok"})

    st = t._state["AAPL"]
    assert st.done is True
    assert st.outcome == "ok"
    assert st.n_tool_calls == 2
    assert st.n_tool_errors == 1
    assert st.round == 1


def test_tracker_ignores_non_staging_events_so_derived_events_dont_recurse():
    t = StageTracker()
    t.update("run_grid", {"grid": "..."})  # no cell → ignored
    t.update("agent_stalled", {"cell": "AAPL", "reason": "x"})  # not staging → ignored
    assert t._state == {}


def test_stalled_flags_quiet_unfinished_cells_once():
    t = StageTracker()
    t.update("cell_start", {"cell": "AAPL"})
    t.update("cell_start", {"cell": "MSFT"})
    # AAPL started reasoning, then went quiet; MSFT keeps emitting.
    t.update("agent_stage", {"cell": "AAPL", "stage": "reasoning"})
    t.update("agent_stage", {"cell": "MSFT", "stage": "reasoning"})
    base = t._state["AAPL"].last_event_mono
    # Pretend AAPL's last event was long ago.
    t._state["AAPL"].last_event_mono = base - 100.0
    stalled = t.stalled(threshold_s=60.0, now=base)
    assert stalled == ["AAPL"]
    # A second check does not re-announce the same stall.
    assert t.stalled(threshold_s=60.0, now=base) == []


def test_stalled_ignores_cells_in_cold_start():
    """A cell that has only seen cell_start (no staging event yet) is waiting
    to start — MCP attach, profile fork — not stalled. The subprocess timeout
    is the backstop for a cell that never starts."""
    t = StageTracker()
    t.update("cell_start", {"cell": "AAPL"})
    base = t._state["AAPL"].last_event_mono
    t._state["AAPL"].last_event_mono = base - 1000.0
    assert t.stalled(threshold_s=60.0, now=base) == []


def test_stalled_excludes_finished_cells():
    t = StageTracker()
    t.update("cell_start", {"cell": "AAPL"})
    t.update("cell_done", {"cell": "AAPL", "outcome": "ok"})
    base = t._state["AAPL"].last_event_mono
    t._state["AAPL"].last_event_mono = base - 1000.0
    assert t.stalled(threshold_s=60.0, now=base) == []


def test_extra_calls_flags_browsing_forever_cells():
    t = StageTracker()
    t.update("cell_start", {"cell": "AAPL"})
    t.update("cell_start", {"cell": "MSFT"})
    for _ in range(8):
        t.update("agent_tool_call", {"cell": "AAPL", "tool": "get_prices"})
    t.update("agent_tool_call", {"cell": "MSFT", "tool": "get_prices"})
    assert t.extra_calls(threshold=5) == ["AAPL"]
    # Finished cells drop out of the 'browsing' signal.
    t.update("cell_done", {"cell": "AAPL", "outcome": "ok"})
    assert t.extra_calls(threshold=5) == []


def test_grid_renders_one_line_per_cell_with_stage_and_counts():
    t = StageTracker()
    t.update("cell_start", {"cell": "AAPL"})
    t.update("agent_stage", {"cell": "AAPL", "stage": "reasoning", "round": 2})
    t.update("agent_tool_call", {"cell": "AAPL", "tool": "get_prices"})
    t.update("agent_tool_result", {"cell": "AAPL", "tool": "get_prices", "ok": False})
    t.update("cell_start", {"cell": "MSFT"})
    t.update("cell_done", {"cell": "MSFT", "outcome": "ok"})
    grid = t.grid()
    assert "== cells" in grid
    assert "AAPL" in grid and "MSFT" in grid
    assert "calls=1" in grid
    assert "err=1" in grid
    assert "r2" in grid
    # MSFT is done.
    msft_line = [ln for ln in grid.splitlines() if "MSFT" in ln][0]
    assert "✓" in msft_line


def test_grid_empty_when_no_cells():
    t = StageTracker()
    assert t.grid() == ""
