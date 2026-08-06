"""Environment health: catch harness bugs that used to look like clean runs."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fintel.environment.access import DataAccess
from fintel.environment.cell import Cell
from fintel.environment.health import audit_events, audit_job, audit_session, worst
from fintel.environment.policy import AccessPolicy
from fintel.environment.tools import ToolSurface
from fintel.environment.trace import AccessLog, load
from fintel.market import catalog

DAY = date(2025, 1, 2)


class Spy:
    name = "spy"
    kinds = ("prices",)

    def fetch(self, query, cutoff):
        return [{"date": "2024-12-01", "close": 1.0}]


def _access() -> DataAccess:
    cell = Cell(run_id="r1", decision_date=DAY, symbols=("AAPL",))
    policy = AccessPolicy(
        kinds=frozenset({"prices"}), decidable=frozenset({"AAPL"}), peers=frozenset()
    )
    return DataAccess(cell=cell, sources={"prices": Spy()}, policy=policy)


def test_tool_schema_denial_is_recorded():
    """The kwargs bug: denied before fetch must still hit the access log."""
    access = _access()
    log = AccessLog(cell=access.cell)
    access.on_read = log.record
    name = (
        "synthetic_prices"
        if catalog.has_source("synthetic_prices")
        else next(s.name for s in catalog.sources() if s.kind == "prices")
    )
    tools = ToolSurface(access=access, bound={"prices": name})

    payload = tools.call("get_prices", {"kwargs": {"symbol": "AAPL"}})
    assert payload["status"] == "denied"
    assert any(e.get("status") == "denied" for e in log.reads)
    health = audit_events(log.events, decision_date=DAY, expect_tools=True)
    assert health.status == "broken"
    assert any("schema" in i or "denied" in i for i in health.issues)


def test_zero_ok_reads_is_broken_for_tool_agents():
    events = [
        {
            "event": "read",
            "kind": "prices",
            "status": "failed",
            "detail": "TypeError: boom",
            "query": {"symbol": "AAPL"},
        }
    ]
    health = audit_events(events, decision_date=DAY, expect_tools=True, outcome="abstained")
    assert health.status == "broken"


def test_tool_errors_from_transcript_flip_health_to_broken():
    """The -32001 errors the agent saw (but our server executed fine) must
    flip the cell to broken — that's the whole point of the catcher."""
    events = [
        {
            "event": "read",
            "kind": "fundamentals",
            "status": "ok",
            "query": {"symbol": "AAPL"},
            "n": 8,
        },
        {
            "event": "tool_errors",
            "n": 3,
            "errors": [
                {
                    "tool": "fintel__get_fundamentals",
                    "error": "MCP error -32001: Request timed out",
                },
                {"tool": "fintel__get_prices", "error": "MCP error -32001: Request timed out"},
                {"tool": "fintel__get_news", "error": "MCP error -32000: Connection closed"},
            ],
        },
    ]
    health = audit_events(events, decision_date=DAY, expect_tools=True, outcome="abstained")
    assert health.status == "broken"
    assert any("tool error" in i.lower() for i in health.issues)
    assert any("-32001" in i for i in health.issues)


def test_empty_only_is_degraded_not_broken():
    events = [
        {
            "event": "read",
            "kind": "news",
            "status": "empty",
            "query": {"symbol": "AAPL"},
        }
    ]
    health = audit_events(events, decision_date=DAY, expect_tools=True, outcome="ok")
    assert health.status == "degraded"


def test_pit_suspect_on_or_after_decision_date():
    events = [
        {
            "event": "read",
            "kind": "prices",
            "status": "ok",
            "query": {"symbol": "AAPL", "end": "2025-01-02"},
        }
    ]
    health = audit_events(events, decision_date=DAY, expect_tools=True)
    assert health.status == "broken"
    assert health.pit_suspects


def test_abstain_reason_with_kwargs_pattern_is_broken():
    health = audit_events(
        [],
        decision_date=DAY,
        expect_tools=True,
        outcome="abstained",
        abstain_reason="tools require ['symbol']; got ['kwargs']",
    )
    assert health.status == "broken"


def test_pack_agent_may_have_zero_reads():
    health = audit_events([], decision_date=DAY, expect_tools=False, outcome="ok")
    assert health.status == "ok"


def test_worst_rollup():
    assert worst("ok", "degraded", "broken") == "broken"
    assert worst("ok", "degraded") == "degraded"
    assert worst() == "ok"


def test_audit_session_and_job(tmp_path: Path):
    session = tmp_path / "job" / "r1" / "sessions" / "job-r1" / "2025-01-02" / "AAPL"
    session.mkdir(parents=True)
    cell = Cell(run_id="job-r1", decision_date=DAY, symbols=("AAPL",))
    log = AccessLog(cell=cell, path=session / "access.jsonl")
    from fintel.environment.access import Reading

    log.record(
        Reading(
            kind="prices",
            query={"symbol": "AAPL"},
            status="denied",
            detail="get_prices requires ['symbol']; got ['kwargs']",
        )
    )
    (session / "result.json").write_text(
        '{"views": [], "abstain": true, "abstain_reason": "requires [\'symbol\']"}'
    )

    cell_health = audit_session(session, expect_tools=True)
    assert cell_health.status == "broken"

    job_report = audit_job(tmp_path / "job", expect_tools=True)
    assert job_report["status"] == "broken"
    assert job_report["n_cells"] == 1


def test_mcp_attach_appends_without_second_cell_opened(tmp_path: Path):
    cell = Cell(run_id="r1", decision_date=DAY, symbols=("AAPL",))
    path = tmp_path / "access.jsonl"
    parent = AccessLog(cell=cell, path=path)
    mcp = AccessLog(cell=cell, path=path, attach=True)
    mcp.append("read", kind="prices", status="ok", query={"symbol": "AAPL"})
    events = load(path)
    assert sum(1 for e in events if e["event"] == "cell_opened") == 1
    assert any(e["event"] == "mcp_attached" for e in events)
    assert any(e.get("status") == "ok" for e in events if e["event"] == "read")
    assert parent.events  # parent still has its own memory
