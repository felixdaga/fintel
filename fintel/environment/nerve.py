"""The central nervous system: one emit surface for a run.

`Nerve` is the single place a run's events flow through. It implements the
`Progress` protocol (so it threads through the existing `progress=` plumbing)
and fans every event to two sinks that share one schema:

  * a live terminal line (stderr), so an operator watching a run sees stages;
  * a durable `run.log` JSONL under the run folder, so a finished run is
    replayable and post-mortem-able without re-running.

This consolidates what was previously two parallel streams — the orchestration
`progress` events (live) and the cell `access.jsonl` events (audit) — onto one
schema owned by the environment module. The per-cell `AccessLog` stays (it is
the PIT audit of *data reads*, a different concern) and continues to write
`access.jsonl`; `Nerve` owns the *run* stream (orchestration + agent staging).

Producers (simulate/job, run, trial, cell; agents/catcher) call
`nerve.emit(event, **fields)`. They do not decide where events land — `Nerve`
does. That is the consolidation: one owner, one schema, two sinks.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from fintel.environment.progress import Progress
from fintel.environment.staging import StageTracker


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# Events that carry per-cell staging state and so feed the StageTracker.
# Derived events (run_grid, agent_stalled) are deliberately NOT here, so they
# don't recurse through the tracker.
_STAGING_EVENTS = frozenset(
    {"cell_start", "agent_stage", "agent_tool_call", "agent_tool_result", "cell_done"}
)


@dataclass
class Nerve(Progress):
    """One run's emit surface. Implements `Progress`, so it threads through
    the existing `progress=` plumbing unchanged.

    `run_root` is the run folder (`runs/<job_id>/r<k>`); the live log is written
    to `<run_root>/run.log` as JSONL. `stream` defaults to stderr; `verbose=False`
    silences the terminal line (the log still records everything), for tests
    and non-interactive runs.

    A `StageTracker` accumulates per-cell state from staging events and, on a
    throttle, the nerve emits a `run_grid` snapshot and any newly-detected
    `agent_stalled` cells — the holistic run-level view an operator watches.
    """

    run_root: Path
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    verbose: bool = True
    log_path: Path | None = None
    stall_threshold_s: float = 60.0
    grid_interval_s: float = 2.0
    tracker: StageTracker = field(default_factory=StageTracker)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last_grid_mono: float = 0.0

    def __post_init__(self) -> None:
        if self.log_path is None:
            self.log_path = self.run_root / "run.log"

    def emit(self, event: str, **fields: Any) -> None:
        record = {"ts": _now(), "event": event, **fields}
        line = _format(event, fields)
        with self._lock:
            if self.verbose and line:
                self.stream.write(line + "\n")
                self.stream.flush()
            if self.log_path is not None:
                try:
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.log_path.open("a") as handle:
                        handle.write(json.dumps(record, default=str) + "\n")
                except OSError:
                    # Losing the live log must not lose the run.
                    pass
        # Derived run-level events are emitted outside the lock (the lock is
        # non-reentrant) and only for staging events, so they can't recurse.
        if event in _STAGING_EVENTS:
            self.tracker.update(event, fields)
            self._maybe_derived()

    def _maybe_derived(self) -> None:
        now = time.monotonic()
        if now - self._last_grid_mono >= self.grid_interval_s:
            self._last_grid_mono = now
            grid = self.tracker.grid(now=now)
            if grid:
                self.emit("run_grid", grid=grid)
        for cell in self.tracker.stalled(threshold_s=self.stall_threshold_s, now=now):
            idle = int(now - self.tracker._state[cell].last_event_mono)
            self.emit(
                "agent_stalled",
                cell=cell,
                reason="no staging event for %.0fs" % self.stall_threshold_s,
                since_ms=idle * 1000,
            )


# ── rendering ─────────────────────────────────────────────────────────────────
# Moved here from simulate/progress.py when ConsoleProgress was folded into
# Nerve. One renderer, one place: every event the run emits is formatted here.


def _format(event: str, fields: dict[str, Any]) -> str:
    """Compact one-line messages for the common events.

    Unknown events fall back to a generic ``event  k=v k=v`` line, so a new
    emit site never silently disappears from the terminal — it just looks
    plain until someone gives it a dedicated format.
    """
    if event == "job_start":
        return (
            f"== job {fields.get('job_id')}  strategy={fields.get('strategy')}  "
            f"agent={fields.get('agent')}  k={fields.get('k_repeats')}"
        )
    if event == "preflight_ok":
        return (
            f"   preflight ok  dates={fields.get('n_dates')}  "
            f"universe≈{fields.get('n_symbols')}"
        )
    if event == "probe_start":
        return (
            f"   probe  kinds={fields.get('kinds')}  symbol={fields.get('symbol')}  "
            f"timeout={fields.get('timeout_s')}s"
        )
    if event == "probe_kind":
        status = fields.get("status", "?")
        tag = "ok" if status in ("ok", "empty") else "FAIL"
        n = fields.get("n")
        n_str = f"  n={n}" if n is not None else ""
        return (
            f"     probe {fields.get('kind'):14} [{tag}]  "
            f"{fields.get('latency_ms', 0)}ms{n_str}"
        )
    if event == "probe_ok":
        return f"   probe ok  {fields.get('n_ok')}/{fields.get('n_kinds')} kinds reachable"
    if event == "probe_failed":
        return (
            f"   probe FAILED  {fields.get('n_failed')}/{fields.get('n_kinds')} kinds "
            f"unreachable: {fields.get('failed_kinds')}"
        )
    if event == "preflight_start":  # prefetch warm
        return (
            f"   prefetch  symbols={fields.get('n_symbols')}  "
            f"kinds={fields.get('kinds')}  window={fields.get('from_date')}…{fields.get('through_date')}"
        )
    if event == "preflight_done":
        status = "ok" if not fields.get("n_failed") else f"failed={fields.get('n_failed')}"
        return (
            f"   prefetch done  warmed={fields.get('n_warmed')}  {status}  "
            f"{fields.get('elapsed_ms', 0)}ms"
        )
    if event == "run_start":
        return f"-- run {fields.get('run_id')} ({fields.get('k_index')}/{fields.get('k_repeats')})"
    if event == "run_echo":
        # The echo is a pre-rendered multi-line block; print it verbatim.
        return str(fields.get("echo") or "")
    if event == "run_grid":
        # The grid is a pre-rendered multi-line snapshot of all cells.
        return str(fields.get("grid") or "")
    if event == "trial_start":
        return (
            f"   trial {fields.get('decision_date')}  "
            f"symbols={fields.get('n_symbols')}  cells={fields.get('n_cells')}"
        )
    if event == "cell_start":
        return f"     cell {fields.get('cell')} …"
    if event == "agent_stage":
        stage = fields.get("stage", "?")
        round_n = fields.get("round")
        round_str = f" r{round_n}" if round_n is not None else ""
        text = str(fields.get("text") or "")[:80]
        text_str = f"  {text}" if text else ""
        return f"       {fields.get('cell')} {stage}{round_str}{text_str}"
    if event == "agent_tool_call":
        return f"     → {fields.get('tool')}  {fields.get('args', '')}"
    if event == "agent_tool_result":
        ok = fields.get("ok", True)
        tag = "ok" if ok else "ERROR"
        text = f"  {fields.get('text', '')[:80]}" if not ok else ""
        return f"     ← {fields.get('tool')}  [{tag}]{text}"
    if event == "agent_stalled":
        return (
            f"     !! {fields.get('cell')} stalled: {fields.get('reason')} "
            f"({fields.get('since_ms', 0)}ms)"
        )
    if event == "cell_done":
        health = fields.get("health", "?")
        outcome = fields.get("outcome", "?")
        detail = fields.get("detail") or ""
        extra = f"  {detail[:80]}" if detail and health != "ok" else ""
        return (
            f"     cell {fields.get('cell')}  outcome={outcome}  "
            f"health={health}  reads={fields.get('n_reads', 0)}  "
            f"{fields.get('elapsed_ms', 0)}ms{extra}"
        )
    if event == "trial_done":
        return (
            f"   trial {fields.get('decision_date')} done  "
            f"status={fields.get('status')}  views={fields.get('n_views')}"
        )
    if event == "run_done":
        return (
            f"-- run {fields.get('run_id')} done  status={fields.get('status')}  "
            f"views={fields.get('n_views')}"
        )
    if event == "job_done":
        return (
            f"== job {fields.get('job_id')} done  status={fields.get('status')}  "
            f"health={fields.get('health', '?')}"
        )
    if event == "job_health":
        issues = fields.get("issues") or []
        head = f"   health={fields.get('status')}  cells={fields.get('n_cells')}"
        if issues:
            return head + "  " + "; ".join(str(i) for i in issues[:3])
        return head
    # Fallback for anything else — visible, just plain.
    bits = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    return f"   {event}  {bits}".rstrip()
