"""Post-run (and mid-run) health from the access trace.

Agent outcome and harness health are different. A cell can finish with
``outcome=ok`` / ``status=ok`` while every tool call was denied or failed —
exactly the openclaw ``requires ['symbol']; got ['kwargs']`` case. This module
reads ``access.jsonl`` (+ optional result/detail) and grades the *environment*.

Statuses:
  ok       — reads happened as expected; no failed/denied; no PIT suspects
  degraded — empty-only data, or minor denials, but not a hard harness break
  broken   — failed reads, schema denials, tool-error abstain, PIT leak, or
             a tool-calling cell with zero successful reads
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path
from typing import Any, Literal

from fintel.environment.session import RESULT_FILE, TRACE_FILE
from fintel.environment.trace import load as load_trace

HealthStatus = Literal["ok", "degraded", "broken"]

# Past-learning patterns: schema routing bugs, TypeErrors surfaced to the agent.
_HARNESS_ERROR = re.compile(
    r"(requires \[|got \['kwargs'\]|TypeError|unexpected keyword|"
    r"missing.*(argument|required)|no tool named|does not accept)",
    re.IGNORECASE,
)

# Query keys that, if present and on/after the decision date, look like a PIT leak.
_DATE_KEYS = ("as_of", "end", "end_date", "until", "to", "filing_date", "published")


@dataclass(frozen=True)
class CellHealth:
    """What the access log says about one cell's environment."""

    status: HealthStatus
    issues: list[str] = field(default_factory=list)
    n_reads: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    kinds_used: list[str] = field(default_factory=list)
    denied: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    empty: list[dict] = field(default_factory=list)
    pit_suspects: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "issues": list(self.issues),
            "n_reads": self.n_reads,
            "by_status": dict(self.by_status),
            "kinds_used": list(self.kinds_used),
            "denied": list(self.denied),
            "failed": list(self.failed),
            "empty": list(self.empty),
            "pit_suspects": list(self.pit_suspects),
        }


def audit_events(
    events: list[dict],
    *,
    decision_date: Date | str | None = None,
    expect_tools: bool = True,
    outcome: str | None = None,
    detail: str | None = None,
    abstain_reason: str | None = None,
) -> CellHealth:
    """Grade a cell from in-memory or loaded access events."""
    if isinstance(decision_date, str):
        decision_date = Date.fromisoformat(decision_date)

    reads = [e for e in events if e.get("event") == "read"]
    by_status: dict[str, int] = {}
    for read in reads:
        status = str(read.get("status", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1

    kinds_used = sorted({r["kind"] for r in reads if r.get("kind")})
    denied = [r for r in reads if r.get("status") == "denied"]
    failed = [r for r in reads if r.get("status") == "failed"]
    empty = [r for r in reads if r.get("status") == "empty"]
    ok_n = by_status.get("ok", 0)

    pit_suspects: list[dict] = []
    if decision_date is not None:
        for read in reads:
            for hit in _pit_hits(read.get("query") or {}, decision_date):
                pit_suspects.append({"kind": read.get("kind"), **hit})

    issues: list[str] = []
    status: HealthStatus = "ok"

    # Tool errors captured from the agent's own transcript (the catcher).
    tool_error_events = [e for e in events if e.get("event") == "tool_errors"]
    if tool_error_events:
        status = "broken"
        n_errs = sum(e.get("n", 0) for e in tool_error_events)
        issues.append(f"{n_errs} tool error(s) the agent saw (MCP timeout/connection)")
        for ev in tool_error_events:
            for err in (ev.get("errors") or [])[:5]:
                issues.append(f"agent saw {err.get('tool')}: {err.get('error', '')[:120]}")

    if failed:
        status = "broken"
        issues.append(f"{len(failed)} failed read(s)")
        for read in failed[:5]:
            if read.get("detail"):
                issues.append(f"failed {read.get('kind')}: {read['detail'][:160]}")

    harness_denials = [
        r for r in denied if _HARNESS_ERROR.search(str(r.get("detail") or ""))
    ]
    if harness_denials:
        status = "broken"
        issues.append(f"{len(harness_denials)} schema/tool denial(s)")
        for read in harness_denials[:5]:
            issues.append(f"denied {read.get('kind')}: {str(read.get('detail', ''))[:160]}")
    elif denied:
        if status == "ok":
            status = "degraded"
        issues.append(f"{len(denied)} denied read(s)")

    if pit_suspects:
        status = "broken"
        issues.append(f"{len(pit_suspects)} PIT-suspect query field(s)")

    reason = abstain_reason or detail or ""
    if outcome == "abstained" and _HARNESS_ERROR.search(reason):
        status = "broken"
        issues.append(f"abstain looks like harness error: {reason[:160]}")

    if expect_tools and not reads:
        # Subprocess agents that never reached MCP leave an empty log — broken
        # when we expected tools. In-process pack agents may truly have zero reads.
        status = "broken"
        issues.append("expected tool reads but access log has none")
    elif ok_n == 0 and empty and not failed and not denied:
        # Genuine absence (quiet ticker / empty news) is degraded, not a harness bug.
        if status == "ok":
            status = "degraded"
        issues.append(f"all {len(empty)} read(s) empty — data gap, not necessarily a bug")
    elif expect_tools and ok_n == 0 and reads:
        status = "broken"
        issues.append(
            f"tool-calling cell produced {len(reads)} read(s) but zero ok "
            f"(by_status={by_status})"
        )

    return CellHealth(
        status=status,
        issues=issues,
        n_reads=len(reads),
        by_status=by_status,
        kinds_used=kinds_used,
        denied=[{"kind": r.get("kind"), "detail": r.get("detail", "")} for r in denied],
        failed=[{"kind": r.get("kind"), "detail": r.get("detail", "")} for r in failed],
        empty=[{"kind": r.get("kind"), "query": r.get("query", {})} for r in empty],
        pit_suspects=pit_suspects,
    )


def audit_session(
    session_path: str | Path,
    *,
    expect_tools: bool = True,
    decision_date: Date | str | None = None,
) -> CellHealth:
    """Audit one cell session directory (``access.jsonl`` + optional ``result.json``)."""
    path = Path(session_path)
    events = load_trace(path / TRACE_FILE) if (path / TRACE_FILE).is_file() else []
    if decision_date is None:
        for event in events:
            if event.get("event") == "cell_opened" and event.get("decision_date"):
                decision_date = event["decision_date"]
                break

    outcome = detail = abstain_reason = None
    result_path = path / RESULT_FILE
    if result_path.is_file():
        import json

        try:
            payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            if payload.get("abstain"):
                outcome = "abstained"
                abstain_reason = str(payload.get("abstain_reason") or "")
            elif payload.get("views"):
                outcome = "ok"
            detail = abstain_reason

    return audit_events(
        events,
        decision_date=decision_date,
        expect_tools=expect_tools,
        outcome=outcome,
        detail=detail,
        abstain_reason=abstain_reason,
    )


def audit_job(job_root: str | Path, *, expect_tools: bool = True) -> dict[str, Any]:
    """Walk a finished job's sessions and roll up cell health."""
    root = Path(job_root)
    cells: list[dict] = []
    by_status: dict[str, int] = {"ok": 0, "degraded": 0, "broken": 0}

    for access_path in sorted(root.glob("r*/sessions/*/*/*/access.jsonl")):
        session = access_path.parent
        health = audit_session(session, expect_tools=expect_tools)
        by_status[health.status] = by_status.get(health.status, 0) + 1
        cells.append({"session": str(session.relative_to(root)), **health.to_dict()})

    issues: list[str] = []
    if by_status.get("broken"):
        issues.append(f"{by_status['broken']} cell(s) broken")
    if by_status.get("degraded"):
        issues.append(f"{by_status['degraded']} cell(s) degraded")

    if by_status.get("broken"):
        status: HealthStatus = "broken"
    elif by_status.get("degraded"):
        status = "degraded"
    elif cells:
        status = "ok"
    elif (root / "result.json").is_file():
        status = "broken"
        issues.append("job has result.json but no session access traces")
    else:
        status = "ok"

    return {
        "status": status,
        "issues": issues,
        "by_status": by_status,
        "n_cells": len(cells),
        "cells": cells,
    }


def _pit_hits(query: dict, decision_date: Date) -> list[dict]:
    hits = []
    for key in _DATE_KEYS:
        if key not in query:
            continue
        raw = query[key]
        if raw is None:
            continue
        try:
            value = Date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        if value >= decision_date:
            hits.append({"field": key, "value": value.isoformat()})
    return hits


def worst(*statuses: HealthStatus) -> HealthStatus:
    order = {"ok": 0, "degraded": 1, "broken": 2}
    if not statuses:
        return "ok"
    return max(statuses, key=lambda s: order.get(s, 0))
