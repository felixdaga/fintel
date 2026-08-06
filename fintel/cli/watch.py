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
import os
import re
import select
import shutil
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
# Cursor up N lines (used by stream mode to rewrite in place without alt screen).
_CUU = lambda n: f"\033[{n}A" if n > 0 else ""
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

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
    # Running usage total across completed cells (live cost). cost_usd stays
    # None until a cell reports a real charge; basis tracks provenance.
    n_llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = None
    cost_basis: str = "unknown"
    n_cells_done: int = 0

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
    if name in ("run_start", "backfill_start"):
        run.status = "running"
        run.started_mono = now
    elif name in ("run_done", "backfill_done"):
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
        # Live cost rollup: add this cell's usage to the running total.
        run.n_cells_done += 1
        run.n_llm_calls += int(ev.get("n_llm_calls", 0) or 0)
        run.tokens_in += int(ev.get("tokens_in", 0) or 0)
        run.tokens_out += int(ev.get("tokens_out", 0) or 0)
        cell_cost = ev.get("cost_usd")
        if cell_cost is not None:
            run.cost_usd = (run.cost_usd or 0.0) + float(cell_cost)
        # basis: reported wins; once unknown stays unknown unless a real cost lands.
        cb = ev.get("cost_basis", "unknown")
        if cb == "reported" or (run.cost_usd is not None and cb in ("reported", "estimated")):
            run.cost_basis = cb if run.cost_basis != "reported" else "reported"
        elif run.cost_basis != "reported":
            run.cost_basis = cb


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


def _fmt_tok(n: int) -> str:
    """Compact token count: 1234 -> 1.2k, 1234567 -> 1.2M."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _cost_line(run: _Run) -> str:
    """One-line live usage rollup for a run, shown at the top of its block.

    `cost_usd` stays None until a cell reports a real charge; `basis` travels
    with the number so an estimate never reads as a measurement.
    """
    if run.cost_usd is not None:
        cost = f"${run.cost_usd:.4f}"
    else:
        cost = _c("d", "n/a")
    basis = run.cost_basis
    bc = {"reported": "g", "estimated": "y", "mixed": "y", "unknown": "d"}.get(basis, "d")
    return (
        _c("d", "cost ") + _c(bc, cost)
        + _c("d", f"  ({basis})")
        + _c("d", "  tok ") + _fmt_tok(run.tokens_in) + _c("d", "→") + _fmt_tok(run.tokens_out)
        + _c("d", f"  {run.n_llm_calls} calls  {run.n_cells_done} cells")
    )


def _visible_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def _truncate_ansi(s: str, width: int) -> str:
    """Truncate a possibly-colored string to ``width`` visible columns."""
    if width <= 0 or _visible_len(s) <= width:
        return s
    out: list[str] = []
    visible = 0
    i = 0
    while i < len(s) and visible < width - 1:
        if s.startswith("\033[", i):
            m = _ANSI_RE.match(s, i)
            if m:
                out.append(m.group(0))
                i = m.end()
                continue
        out.append(s[i])
        visible += 1
        i += 1
    out.append(_C["x"])
    return "".join(out)


def _render_lines(
    runs: list[_Run],
    now: float,
    preflight: _Preflight | None = None,
    *,
    collapse_done: bool = True,
) -> list[str]:
    """The dashboard body as a list of plain lines (no cursor/positioning escapes).

    Callers add their own positioning (alt-screen home+clear, or stream cursor-up)
    so the same body serves both render modes.

    ``collapse_done`` (default True) keeps only active cells in the grid and
    summarises finished ones — otherwise a 55-cell backfill makes the frame
    grow every tick and stream-mode cursor-up drifts.
    """
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
        # Live cost rollup (total across completed cells), shown at the top.
        head += "   " + _cost_line(run)
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
            active = [c for c in run.cells.values() if c.stage != "done"]
            done = [c for c in run.cells.values() if c.stage == "done"]
            show = active if collapse_done else list(run.cells.values())
            if collapse_done and done:
                ok_n = sum(1 for c in done if c.outcome == "ok")
                fail_n = len(done) - ok_n
                lines.append(
                    _c("d", f"   done {len(done)}/{len(run.cells)}")
                    + _c("g", f"  ok={ok_n}")
                    + (_c("r", f"  fail={fail_n}") if fail_n else "")
                    + _c("d", f"  active={len(active)}")
                )
            if show:
                lines.append(
                    _c("d", "   cell       date        stage                    round calls err idle    last tool / text")
                )
                for c in show:
                    idle = 0.0 if c.stage == "done" else now - c.last_mono
                    sc = _stage_color(c.stage)
                    stage = c.stage
                    if c.stage == "done":
                        stage = f"done:{c.outcome}"
                    # Cap the free-text tail so rows don't wrap (wrap breaks stream CUU).
                    raw_tail = (c.last_tool or c.last_reason or "")[:40]
                    tail = _c("y", raw_tail) if c.last_tool else _c("d", raw_tail)
                    row = (
                        f"   {_col('b', c.name, 9)} {_col('c', c.date, 10)} {_col(sc, stage, 24)} "
                        f"r{c.round:<2} {_col('c', str(c.calls), 4, 'right')} "
                        f"{_col('r' if c.errors else 'd', str(c.errors), 3, 'right')} "
                        f"{idle:4.0f}s  {tail}"
                    )
                    lines.append(row)
            elif not done:
                lines.append(_c("d", "   (no cells yet)"))
        lines.append(_c("d", "─" * 78))
    lines.append(_c("d", "press q (or Ctrl-C) to quit"))
    return lines


def _render(runs: list[_Run], now: float, preflight: _Preflight | None = None) -> str:
    # Home + clear, then each line with a trailing clear-to-EOL, then clear to EOS.
    body = "\n".join(l + _CLEAR_LINE for l in _render_lines(runs, now, preflight))
    return _HOME + _CLEAR_SCREEN + body + "\n" + "\033[J"


def _auto_mode() -> str:
    """Pick a render mode from the environment.

    Cursor/VS Code's integrated terminal (xterm.js) handles the alternate-screen
    buffer poorly — the dashboard can render blank or fail to restore on exit —
    so default to stream mode there. A real terminal gets the full alt-screen app.
    """
    tp = (os.environ.get("TERM_PROGRAM") or "").lower()
    if tp in ("cursor", "vscode", "vsce"):
        return "stream"
    return "alt"


def watch_run_logs(
    paths: list[Path],
    tags: list[str] | None = None,
    poll_s: float = 0.3,
    job_log: Path | None = None,
    mode: str = "auto",
) -> int:
    if mode == "auto":
        mode = _auto_mode()
    runs: list[_Run] = []
    for i, p in enumerate(paths):
        tag = (tags[i] if tags and i < len(tags) else _log_tag(p))
        runs.append(_Run(tag=tag, path=p))
    preflight = _Preflight()
    pf_offset = [0]
    interactive = sys.stdout.isatty()
    use_alt = interactive and mode == "alt"
    use_stream = interactive and mode == "stream"

    sys.stdout.write((_ALT_ON + _HIDE_CURSOR) if use_alt else "")
    sys.stdout.flush()
    old = None
    if use_alt:
        try:
            old = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        except (termios.error, ValueError):
            old = None
    prev_lines = 0
    try:
        while True:
            for run in runs:
                _drain(run)
            if job_log is not None:
                _drain_preflight(preflight, job_log, pf_offset)
            # Stream mode collapses done cells so the frame height stays roughly
            # stable (active pool only). Alt mode has room for the full grid.
            lines = _render_lines(
                runs, time.monotonic(), preflight=preflight,
                collapse_done=use_stream or not use_alt,
            )
            if use_alt:
                body = "\n".join(l + _CLEAR_LINE for l in lines) + "\n"
                sys.stdout.write(_HOME + _CLEAR_SCREEN + body + "\033[J")
            elif use_stream:
                # Rewrite in place: CUU by the *previous content line count*
                # (not +1 — the trailing newline already parks the cursor one
                # row below the frame). Truncate to terminal width so wrapped
                # lines can't desync the cursor. Pad shorter frames so leftover
                # rows from a taller previous frame are blanked.
                cols = shutil.get_terminal_size((80, 24)).columns
                lines = [_truncate_ansi(l, max(20, cols - 1)) for l in lines]
                while len(lines) < prev_lines:
                    lines.append("")
                body = "\n".join(l + _CLEAR_LINE for l in lines)
                if prev_lines:
                    sys.stdout.write(_CUU(prev_lines))
                sys.stdout.write(body + "\n")
                prev_lines = len(lines)
            else:
                body = "\n".join(l + _CLEAR_LINE for l in lines) + "\n"
                # Non-interactive / captured: clear-and-home each frame.
                sys.stdout.write(_CLEAR_SCREEN + _HOME + body)
            sys.stdout.flush()
            if use_alt and old is not None:
                r, _, _ = select.select([sys.stdin], [], [], poll_s)
                if r and sys.stdin.read(1) in ("q", "\x03"):
                    break
            else:
                # Stream mode: no raw stdin — quit via Ctrl-C (caught below) or
                # auto-exit when all runs report done.
                time.sleep(poll_s)
            if all(r.done for r in runs):
                time.sleep(0.5)
                break
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write((_SHOW_CURSOR + _ALT_OFF) if use_alt else "")
        sys.stdout.flush()
        if old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
    return 0


def resolve_paths(job_ids: list[str], output_root: Path) -> list[Path]:
    """Resolve job ids / log paths into nerve JSONL files to watch.

    Accepts:
      · a job id under ``output_root`` — prefers ``backfill.log`` when present
        (active backfill), otherwise ``r*/run.log``
      · a literal path to any ``*.log`` nerve file (``run.log``, ``backfill.log``, …)
    """
    paths: list[Path] = []
    for jid in job_ids:
        job = output_root / jid
        if not job.is_dir():
            p = Path(jid)
            if p.is_file() and p.suffix == ".log":
                paths.append(p)
            continue
        # Prefer an in-progress backfill over the finished run logs.
        backfill = job / "backfill.log"
        if backfill.is_file():
            paths.append(backfill)
            continue
        for r in sorted(job.glob("r*/run.log")):
            paths.append(r)
    return paths


def _log_tag(path: Path) -> str:
    """Human label for a nerve log: job id, or ``job/backfill`` / ``job/r1``."""
    if path.name == "backfill.log":
        return f"{path.parent.name}/backfill"
    if path.name == "run.log" and path.parent.name.startswith("r"):
        return f"{path.parent.parent.name}/{path.parent.name}"
    return path.parent.name


def run_watch(args) -> int:
    root = Path(getattr(args, "output_root", "runs"))
    job_ids = list(args.job_ids)
    mode = getattr(args, "watch_mode", "auto")
    wait_s = int(getattr(args, "wait", 0) or 0)
    paths = resolve_paths(job_ids, root)
    if not paths and wait_s > 0:
        import time as _time
        print(
            f"waiting for {job_ids} under {root} (up to {wait_s}s)...",
            file=sys.stderr, flush=True,
        )
        deadline = _time.monotonic() + wait_s
        while not paths and _time.monotonic() < deadline:
            _time.sleep(1.0)
            paths = resolve_paths(job_ids, root)
    if not paths:
        print(f"no nerve log found for {job_ids} under {root}", file=sys.stderr)
        return 1
    tags = [_log_tag(p) for p in paths]
    return watch_run_logs(paths, tags=tags, mode=mode)
