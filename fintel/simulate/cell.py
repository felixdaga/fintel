"""Run one cell: build its environment, invoke the agent, record the answer.

The leaf of the fan-out. Everything above (trial, run, job) is orchestration;
this is where the agent is actually called, through `agents.invoke`, which
never raises — a failed cell is a recorded outcome, not a lost run. The old
platform let an adapter exception abort the whole date.

The agent is built fresh per cell. For a scripted agent this is free; for an
LLM agent it constructs an HTTP client, which is cheap next to the call itself.
The reason is correctness under parallel cells: `LLMAgent` accumulates trace
steps on `self`, so two threads sharing one instance would corrupt each other's
trace. Building per cell makes the agent re-entrant by construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from fintel.agents import factory as agent_factory
from fintel.agents.run import invoke
from fintel.environment.cell import Cell
from fintel.environment.factory import RuntimeConfig, build_environment
from fintel.environment.health import audit_events
from fintel.environment.trace import load
from fintel.market.data.base import DataSource
from fintel.market.settings import MarketConfig
from fintel.models.agent import AgentSpec
from fintel.models.common import Symbol
from fintel.models.decision import AgentResponse
from fintel.models.trial import CellResult, CellRecord
from fintel.environment.progress import NullProgress, Progress
from fintel.simulate.artifacts import write_cell


@dataclass(frozen=True)
class CellOutcome:
    """What `run_cell` returns: the result model plus the raw response.

    The raw response is kept so the trial coordinator can reduce views into a
    decision without re-reading the cell file from disk.
    """

    result: CellResult
    response: AgentResponse


def build_agent(spec: AgentSpec, *, mission_text: str = "", output_schema_text: str = ""):
    """Resolve an `AgentSpec` into a live agent.

    The model pin and the opaque `options` are the platform's usual hand-off;
    `mission_text`/`output_schema_text` are the strategy pack's mission.md and
    output_schema.json, offered to every agent uniformly (an explicit `options`
    entry still wins via `setdefault`). Every builtin declares both, so it
    always gets them; a custom `module:Class` adapter that doesn't is built
    without them rather than failing the cell — optional platform context, not
    a contract a one-off adapter is forced into.
    """
    params: dict = dict(spec.options)
    if spec.model.id:
        params.setdefault("model", spec.model.id)
    with_context = dict(params)
    with_context.setdefault("mission_text", mission_text)
    with_context.setdefault("output_schema_text", output_schema_text)
    try:
        return agent_factory.build(spec.name, **with_context)
    except TypeError:
        return agent_factory.build(spec.name, **params)


def expect_tools(spec: AgentSpec) -> bool:
    """Whether this agent is expected to leave tool reads in the access log.

    Pack-channel / constant agents can produce a real decision with zero reads;
    CLI + tools-channel hosts cannot — empty logs mean the harness failed.
    """
    name = spec.name
    channel = str(spec.options.get("channel", ""))
    if name in ("openclaw", "claude-code"):
        return True
    if name == "llm" and channel != "pack":
        return True
    return False


def run_cell(
    *,
    cell: Cell,
    sources: dict[str, DataSource],
    universe: list[Symbol],
    agent_spec: AgentSpec,
    runtime: RuntimeConfig,
    cell_path: Path,
    mission_text: str = "",
    output_schema_text: str = "",
    market_config: MarketConfig | None = None,
    progress: Progress | None = None,
) -> CellOutcome:
    """Execute one cell end to end. Never raises; a failure is a CellResult.

    After the agent returns, health is graded from the on-disk access log (so
    MCP subprocess reads count) plus the agent outcome/detail. Broken health
    downgrades the cell status even when the agent stuffed a view.
    """
    progress = progress or NullProgress()
    started = time.perf_counter()
    started_at = _now_iso()
    progress.emit("cell_start", cell=cell.name, decision_date=cell.decision_date.isoformat())

    env = build_environment(
        cell=cell,
        sources=sources,
        universe=universe,
        kinds=tuple(sources),
        runtime=runtime,
        market_config=market_config,
        nerve=progress,
    )

    agent = build_agent(
        agent_spec, mission_text=mission_text, output_schema_text=output_schema_text
    )
    response = invoke(agent, env)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    summary = env.close()

    events = (
        load(env.session.trace)
        if env.session is not None and env.session.trace.is_file()
        else list(env.log.events)
    )
    health = audit_events(
        events,
        decision_date=cell.decision_date,
        expect_tools=expect_tools(agent_spec),
        outcome=response.outcome,
        detail=response.detail,
        abstain_reason=response.detail if response.outcome == "abstained" else None,
    )
    env.log.append("health", **health.to_dict())

    cell_record = CellRecord(
        cell=cell.name,
        symbols=list(cell.symbols),
        decision_date=cell.decision_date.isoformat(),
        scope=cell.scope,
        outcome=response.outcome,
        detail=response.detail,
        n_views=len(response.views),
        usage=response.usage,
        views=dict(response.views),
        environment={**summary, "health": health.to_dict()},
        elapsed_ms=elapsed_ms,
        started_at=started_at,
    )
    write_cell(cell_path, cell_record)

    status = _cell_status(response, health.status)
    result = CellResult(
        cell=cell.name,
        symbols=list(cell.symbols),
        status=status,
        n_views=len(response.views),
        attempts=1,
        duration_ms=elapsed_ms,
        usage=response.usage,
        error=_cell_error(response, health),
        health=health.status,
        health_issues=list(health.issues),
    )
    progress.emit(
        "cell_done",
        cell=cell.name,
        decision_date=cell.decision_date.isoformat(),
        outcome=response.outcome,
        health=health.status,
        n_reads=health.n_reads,
        elapsed_ms=elapsed_ms,
        n_llm_calls=response.usage.n_llm_calls,
        tokens_in=response.usage.tokens_in,
        tokens_out=response.usage.tokens_out,
        cost_usd=response.usage.cost_usd,
        cost_basis=response.usage.basis,
        detail="; ".join(health.issues[:2]) if health.issues else response.detail,
    )
    return CellOutcome(result=result, response=response)


def _cell_status(response: AgentResponse, health: str) -> str:
    if health == "broken":
        return "failed"
    if response.outcome == "ok":
        return "ok"
    if response.outcome == "abstained":
        return "skipped"
    return "failed"


def _cell_error(response: AgentResponse, health) -> str | None:
    if health.status == "broken":
        return "; ".join(health.issues) if health.issues else "environment health broken"
    if response.outcome not in ("ok", "abstained"):
        return response.detail
    return None


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")
