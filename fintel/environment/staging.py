"""Live agent staging: per-cell stage state, stuck heuristics, a run grid.

This is the 'see the agent thinking' half of the nervous system. The nerve
streams one event at a time (a tool call, a reasoning turn); this module holds
the *run-level* view that those events build up: which cell is on which round,
how many tool calls each has made, which one has gone quiet too long, which one
is burning calls. A reviewer watching a run sees a holistic grid refresh, not
just a tail of single events.

Two concerns, kept separate:

  * `StageTracker` accumulates state from staging events. It does not emit —
    the nerve owns emission. The nerve updates the tracker on each staging
    event and, throttled, asks it for a grid + a stall check, then emits those
    as derived `run_grid` / `agent_stalled` events.
  * Stuck detection is heuristic and event-driven: when any cell emits, every
    not-yet-finished cell whose last event is older than the stall threshold is
    flagged. A cell that emits nothing at all is caught by the existing
    subprocess timeout; this catches the cell that started, then went silent
    while others kept working.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Events that carry per-cell staging state. Mirrors the nerve's set; kept here
# so the tracker is self-contained and testable without the nerve.
_STAGING_EVENTS = frozenset(
    {"cell_start", "agent_stage", "agent_tool_call", "agent_tool_result", "cell_done"}
)


@dataclass
class _CellState:
    cell: str
    stage: str = "pending"
    round: int = 0
    n_tool_calls: int = 0
    n_tool_errors: int = 0
    last_event_mono: float = 0.0
    done: bool = False
    outcome: str = ""


@dataclass
class StageTracker:
    """Per-cell stage state, accumulated from staging events.

    Times are monotonic-relative (the tracker is created when the run starts),
    so stall thresholds are wall-clock-ish seconds regardless of test speed.
    """

    cells: list[str] = field(default_factory=list)
    _state: dict[str, _CellState] = field(default_factory=dict)
    _born: float = field(default_factory=time.monotonic)
    _stalled_flagged: set[str] = field(default_factory=set)

    def register(self, cell: str) -> None:
        if cell not in self._state:
            self._state[cell] = _CellState(cell=cell, last_event_mono=time.monotonic())
            self.cells.append(cell)

    def update(self, event: str, fields: dict) -> None:
        """Fold one staging event into the tracker. Non-staging events are
        ignored (so derived `run_grid`/`agent_stalled` don't recurse)."""
        if event not in _STAGING_EVENTS:
            return
        cell = fields.get("cell")
        if not cell:
            return
        self.register(cell)
        st = self._state[cell]
        st.last_event_mono = time.monotonic()
        if event == "cell_start":
            st.stage = "started"
        elif event == "agent_stage":
            st.stage = str(fields.get("stage", "reasoning"))
            r = fields.get("round")
            if isinstance(r, int) and r > st.round:
                st.round = r
        elif event == "agent_tool_call":
            st.stage = "tool_call"
            st.n_tool_calls += 1
        elif event == "agent_tool_result":
            st.stage = "tool_result"
            if fields.get("ok") is False:
                st.n_tool_errors += 1
        elif event == "cell_done":
            st.done = True
            st.stage = "done"
            st.outcome = str(fields.get("outcome", ""))

    def stalled(self, *, threshold_s: float, now: float | None = None) -> list[str]:
        """Cells not finished and silent longer than `threshold_s`. Each cell
        is flagged once per stall (so the same stall isn't re-announced)."""
        now = now if now is not None else time.monotonic()
        out: list[str] = []
        for cell, st in self._state.items():
            if st.done or cell in self._stalled_flagged:
                continue
            if (now - st.last_event_mono) >= threshold_s:
                out.append(cell)
                self._stalled_flagged.add(cell)
        return out

    def extra_calls(self, *, threshold: int) -> list[str]:
        """Cells making more tool calls than `threshold` — the 'this agent is
        browsing forever' signal. Finished cells are excluded."""
        return [
            st.cell
            for st in self._state.values()
            if not st.done and st.n_tool_calls > threshold
        ]

    def grid(self, *, now: float | None = None) -> str:
        """One line per cell: stage, round, tool calls/errors, elapsed. The
        holistic run-level view that refreshes as the run proceeds."""
        now = now if now is not None else time.monotonic()
        if not self._state:
            return ""
        lines = ["== cells"]
        for cell in self.cells:
            st = self._state[cell]
            elapsed = int(now - st.last_event_mono)
            tag = "✓" if st.done else ("…" if st.stage != "done" else "…")
            errs = f" err={st.n_tool_errors}" if st.n_tool_errors else ""
            r = f" r{st.round}" if st.round else ""
            lines.append(
                f"   {tag} {cell:8} {st.stage:12}{r}  calls={st.n_tool_calls}{errs}"
                f"  idle={elapsed}s"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        self._state.clear()
        self.cells.clear()
        self._stalled_flagged.clear()
        self._born = time.monotonic()
