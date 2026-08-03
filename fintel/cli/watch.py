"""``fintel runs watch`` — a live, in-place terminal dashboard for nerve runs.

The nerve already records every run event to ``run.log`` (JSONL). This module
tails one or more of those logs and renders a single, non-scrolling screen that
updates in place: per-run blocks side by side, each with a preflight line and a
per-cell grid showing stage / round / tool calls / errors / idle / last tool /
last reasoning snippet. No third-party deps — pure stdlib ANSI.

Usage:
    fintel runs watch <job_id>...            # watches runs/<job>/r*/run.log
    fintel runs watch <job_id>... --output-root runs

Quit with ``q`` (or Ctrl-C). The dashboard uses the alternate screen buffer,
so the prior terminal contents are restored on exit.
"""

from __future__ import annotations

import json
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path

# ANSI escapes.
_ALT_ON = "\033[?1049h"
_ALT_OFF = "\033[?1049l"
_HOME = "\033[H"
_CLEAR_LINE = "\033[K"
_CLEAR_SCREEN = "\033[2J"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"

# Colors.
_C = {"g": "\033[32m", "r": "\033[31m", "y": "\033[33m", "c": "\033[36m", "d": "\033[2m", "b": "\033[1m", "x": "\033[0m"}


def _c(code: str, s: str) -> str:
    return f"{_C[code]}{s}{_C['x']}"


def _col(code: str, s: str, width: int = 0, align: str = "left") -> str:
    """Format a plain string to a fixed width (so ANSI color codes added after
    don't count toward the field width and break alignment), then color it."""
    if width:
        s = s[:width]
        s = s.rjust(width) if align == "right" else s.ljust(width)
    return _c(code, s)


@dataclass
class _Cell:
    name: str
    date: str = ""
    stage: str = "pending"  # pending | reasoning | tool | done | stalled
    round: int = 0
    calls: int = 0
    errors: int = 0
    last_mono: float = 0.0
    last_tool: str = ""
    last_reason: str = ""
    outcome: str = ""
    health: str = ""
    reads: int = 0
    elapsed_ms: float = 0.0


@dataclass
class _Preflight:
    """Job-level preflight/probe state, drained from job.log. Rendered as a
    single header line above the per-run tracks (probe is job-level, not per-run)."""
    probes: dict[str, str] = field(default_factory=dict)
    status: str = "starting"  # starting | ok | failed

    def apply(self, ev: dict) -> None:
        name = ev.get("event")
        if name == "probe_kind":
            self.probes[ev.get("kind", "?")] = ev.get("result") or ev.get("status") or "ok"
        elif name == "probe_ok":
            self.status = "ok"
        elif name == "preflight_done":
            self.status = "ok"
        elif name == "job_start":
            self.status = "running"


@dataclass
class _Run:
    tag: str
    path: Path
    status: str = "starting"  # starting | running | done | failed
    probes: dict[str, str] = field(default_factory=dict)  # kind -> ok|empty|failed
    cells: dict[str, _Cell] = field(default_factory=dict)  # key = "{cell}@{date}"
    started_mono: float = 0.0
    done: bool = False
    _offset: int = 0

    def cell(self, name: str, date: str) -> _Cell:
        key = f"{name}@{date}"
        if key not in self.cells:
            c = _Cell(name=name, date=date)
            c.last_mono = time.monotonic()
            self.cells[key] = c
        return self.cells[key]


def _apply(run: _Run, ev: dict, now: float) -> None:
    name = ev.get("event")
    # Every per-cell event carries decision_date (cell_start/cell_done and the
    # staging events from catcher/llm_agent/scripted). Older logs without it
    # fall back to "" — fine for single-date runs.
    cell = ev.get("cell", "?")
    date = ev.get("decision_date", "")
    if name == "run_start":
        run.status = "running"
        run.started_mono = now
    elif name == "run_done":
        run.done = True
        run.status = ev.get("status", "done") or "done"
    elif name == "probe_kind":
        kind = ev.get("kind", "?")
        run.probes[kind] = ev.get("result") or ev.get("status") or "ok"
    elif name == "probe_ok":
        for k, v in (ev.get("kinds") or {}).items():
            run.probes[k] = v
    elif name == "cell_start":
        c = run.cell(cell, date)
        c.stage = "running"
        c.last_mono = now
    elif name == "agent_stage":
        c = run.cell(cell, date)
        # Adapter-defined label — TUI displays whatever was emitted.
        c.stage = str(ev.get("stage") or "stage")
        c.round = ev.get("round", c.round)
        c.last_reason = str(ev.get("text", ""))[:60]
        c.last_mono = now
    elif name == "agent_tool_call":
        c = run.cell(cell, date)
        c.stage = "tool"
        c.calls += 1
        c.last_tool = ev.get("tool", "")
        c.last_mono = now
    elif name == "agent_tool_result":
        c = run.cell(cell, date)
        if not ev.get("ok", True):
            c.errors += 1
        c.last_mono = now
    elif name == "agent_stalled":
        c = run.cell(cell, date)
        c.stage = "stalled"
    elif name == "cell_done":
        c = run.cell(cell, date)
        c.stage = "done"
        c.outcome = ev.get("outcome", "")
        c.health = ev.get("health", "")
        c.reads = ev.get("n_reads", 0) or 0
        c.elapsed_ms = ev.get("elapsed_ms", 0.0) or 0.0
        c.last_mono = now


def _drain(run: _Run) -> None:
    """Read any new lines appended to run.path since the last read."""
    try:
        size = run.path.stat().st_size
    except OSError:
        return
    if size < run._offset:
        run._offset = 0  # file was truncated/rotated
    if size == run._offset:
        return
    try:
        with run.path.open("r") as fh:
            fh.seek(run._offset)
            chunk = fh.read()
            run._offset = fh.tell()
    except OSError:
        return
    now = time.monotonic()
    for line in chunk.splitlines():
        if not line.strip():
            continue
        try:
            _apply(run, json.loads(line), now)
        except json.JSONDecodeError:
            continue


def _drain_preflight(pf: _Preflight, path: Path, offset: list[int]) -> None:
    """Read new lines from job.log into the shared preflight state."""
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size < offset[0]:
        offset[0] = 0
    if size == offset[0]:
        return
    try:
        with path.open("r") as fh:
            fh.seek(offset[0])
            chunk = fh.read()
            offset[0] = fh.tell()
    except OSError:
        return
    for line in chunk.splitlines():
        if not line.strip():
            continue
        try:
            pf.apply(json.loads(line))
        except json.JSONDecodeError:
            continue


def _stage_color(stage: str) -> str:
    if stage.startswith("done"):
        return "g"
    return {
        "done": "g",
        "stalled": "r",
        "tool": "y",
        "running": "y",
        "pending": "d",
    }.get(stage, "c")  # any adapter-defined stage → active


def _render(runs: list[_Run], now: float, preflight: _Preflight | None = None) -> str:
    lines: list[str] = []
    lines.append(_c("b", "fintel nerve") + _c("d", "  live dashboard  ") + _c("d", time.strftime("%H:%M:%S")))
    # Shared job-level preflight header (probe is run once, before any track).
    if preflight is not None and preflight.probes:
        bits = []
        for kind, res in preflight.probes.items():
            rc = {"ok": "g", "empty": "y", "failed": "r"}.get(res, "d")
            bits.append(f"{kind}:{_c(rc, res or '?')}")
        stc = {"ok": "g", "failed": "r", "running": "c", "starting": "d"}.get(preflight.status, "d")
        lines.append(f"{_c('b', 'preflight')}  {_c(stc, preflight.status)}  " + _c("d", "probe[") + " ".join(bits) + _c("d", "]"))
    lines.append(_c("d", "─" * 78))
    for run in runs:
        elapsed = (now - run.started_mono) if run.started_mono and not run.done else 0.0
        st = run.status
        stc = {"done": "g", "failed": "r", "running": "c", "starting": "d"}.get(st, "d")
        head = f"{_c('b', run.tag)}  {_c(stc, st)}  {_c('d', f'{elapsed:5.1f}s')}"
        # preflight / probes line
        if run.probes:
            bits = []
            for kind, res in run.probes.items():
                rc = {"ok": "g", "empty": "y", "failed": "r"}.get(res, "d")
                bits.append(f"{kind}:{_c(rc, res or '?')}")
            head += "   " + _c("d", "probe[") + " ".join(bits) + _c("d", "]")
        lines.append(head)
        if not run.cells:
            lines.append(_c("d", "   (no cells yet)"))
        else:
            lines.append(
                _c("d", "   cell       date        stage                    round calls err idle    last tool / text")
            )
            for c in run.cells.values():
                idle = 0.0 if c.stage == "done" else now - c.last_mono
                sc = _stage_color(c.stage)
                stage = c.stage
                if c.stage == "done":
                    stage = f"done:{c.outcome}"
                tail = c.last_tool or c.last_reason
                tail = _c("y", tail) if c.last_tool else _c("d", tail)
                row = (
                    f"   {_col('b', c.name, 9)} {_col('c', c.date, 10)} {_col(sc, stage, 24)} "
                    f"r{c.round:<2} {_col('c', str(c.calls), 4, 'right')} "
                    f"{_col('r' if c.errors else 'd', str(c.errors), 3, 'right')} "
                    f"{idle:4.0f}s  {tail}"
                )
                lines.append(row + _CLEAR_LINE)
        lines.append(_c("d", "─" * 78))
    lines.append(_c("d", "press q to quit"))
    # Home + clear, then each line with a trailing clear-to-EOL, then clear to EOS.
    return _HOME + _CLEAR_SCREEN + "\n".join(l + _CLEAR_LINE for l in lines) + "\n" + "\033[J"


def watch_run_logs(
    paths: list[Path],
    tags: list[str] | None = None,
    poll_s: float = 0.3,
    job_log: Path | None = None,
) -> int:
    runs: list[_Run] = []
    for i, p in enumerate(paths):
        tag = (tags[i] if tags and i < len(tags) else p.parent.parent.parent.name)
        runs.append(_Run(tag=tag, path=p))
    preflight = _Preflight()
    pf_offset = [0]
    interactive = sys.stdout.isatty() and sys.stdin.isatty()
    sys.stdout.write((_ALT_ON + _HIDE_CURSOR) if interactive else "")
    sys.stdout.flush()
    old = None
    try:
        if interactive:
            try:
                old = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
            except (termios.error, ValueError):
                old = None
        while True:
            for run in runs:
                _drain(run)
            if job_log is not None:
                _drain_preflight(preflight, job_log, pf_offset)
            frame = _render(runs, time.monotonic(), preflight=preflight)
            if interactive:
                sys.stdout.write(frame)
            else:
                # Plain frames for piped/captured output: clear screen each frame.
                sys.stdout.write(_CLEAR_SCREEN + _HOME + frame)
            sys.stdout.flush()
            if interactive and old is not None:
                r, _, _ = select.select([sys.stdin], [], [], poll_s)
                if r and sys.stdin.read(1) in ("q", "\x03"):
                    break
            else:
                time.sleep(poll_s)
            if all(r.done for r in runs):
                time.sleep(0.5)
                break
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write((_SHOW_CURSOR + _ALT_OFF) if interactive else "")
        sys.stdout.flush()
        if old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
    return 0


def resolve_paths(job_ids: list[str], output_root: Path) -> list[Path]:
    paths: list[Path] = []
    for jid in job_ids:
        job = output_root / jid
        if not job.is_dir():
            # maybe it's a literal run.log path
            p = Path(jid)
            if p.is_file() and p.name == "run.log":
                paths.append(p)
            continue
        for r in sorted(job.glob("r*/run.log")):
            paths.append(r)
        if not any(p.parent.parent.parent == job for p in paths):
            # keep only those under this job
            paths = [p for p in paths if p.parent.parent.parent == job] or paths
    return paths


def run_watch(args) -> int:
    root = Path(getattr(args, "output_root", "runs"))
    job_ids = list(args.job_ids)
    paths = resolve_paths(job_ids, root)
    if not paths:
        print(f"no run.log found for {job_ids} under {root}", file=sys.stderr)
        return 1
    tags = [p.parent.parent.parent.name for p in paths]
    return watch_run_logs(paths, tags=tags)
