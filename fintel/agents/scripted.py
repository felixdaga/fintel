"""Agents with no model behind them.

`ConstantAgent` is a real baseline: a strategy whose scores carry no information
is the floor any agent has to beat, and a backtest without one has no scale.

`ScriptedAgent` is the test double. Every failure an adapter can have is
reachable from it, so the platform's error handling is exercised on every commit
instead of only when a provider happens to rate-limit us. It also reads through
whichever channel it is told to, which is how the conformance suite checks that
tools, pack and direct access really are one path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fintel.agents.base import AgentError
from fintel.environment import Environment
from fintel.environment.tools import tool_name
from fintel.models.common import Outcome, Symbol
from fintel.models.decision import AgentResponse, SourceRef, View
from fintel.models.trace import CostBasis, Usage


@dataclass
class ConstantAgent:
    """One fixed score for every symbol it may decide. Reads nothing."""

    score: float = 0.0
    conviction: float = 0.5
    rationale: str = "constant baseline"
    name: str = "constant"
    version: str = "1"

    def decide(self, env: Environment) -> AgentResponse:
        views = {
            symbol: View(
                symbol=symbol,
                score=self.score,
                conviction=self.conviction,
                rationale=self.rationale,
            )
            for symbol in sorted(env.policy.decidable)
        }
        outcome: Outcome = "ok" if views else "empty"
        return AgentResponse(views=views, outcome=outcome)


@dataclass
class ScriptedAgent:
    """Deterministic agent whose behaviour is entirely declared up front."""

    score: float = 0.5
    channel: str = "direct"  # direct | tools | pack
    reads: tuple[str, ...] = ()  # kinds to read before deciding
    raises: BaseException | None = None
    outcome: Outcome = "ok"
    usage: Usage | None = None
    cost_basis: CostBasis = "unknown"
    name: str = "scripted"
    version: str = "1"
    seen: dict[str, Any] = field(default_factory=dict, init=False)

    def decide(self, env: Environment) -> AgentResponse:
        if self.raises is not None:
            raise self.raises
        for kind in self.reads or env.kinds:
            self.seen[kind] = self._read(env, kind)

        if self.outcome != "ok":
            # Declared non-ok outcomes carry no views, which is what the
            # response model requires anyway.
            return AgentResponse(views={}, outcome=self.outcome, usage=self._usage())

        views = {
            symbol: View(
                symbol=symbol,
                score=self.score,
                rationale=f"scripted via {self.channel}",
                sources_cited=[
                    SourceRef(source_type=kind, source_id=f"{kind}:{symbol}")
                    for kind in sorted(self.seen)
                ],
            )
            for symbol in sorted(env.policy.decidable)
        }
        outcome: Outcome = "ok" if views else "empty"
        return AgentResponse(views=views, outcome=outcome, usage=self._usage())

    def _read(self, env: Environment, kind: str) -> Any:
        subject: Symbol | None = next(iter(sorted(env.policy.decidable)), None)
        if self.channel == "pack":
            return env.evidence(symbol=subject)
        if self.channel == "tools":
            return env.tools.call(tool_name(kind), {"symbol": subject})
        if self.channel == "direct":
            return env.access.read(kind, symbol=subject)
        raise AgentError(f"unknown channel {self.channel!r}")

    def _usage(self) -> Usage:
        if self.usage is not None:
            return self.usage
        return Usage(
            n_llm_calls=1,
            tokens_in=100,
            tokens_out=10,
            cost_usd=0.001 if self.cost_basis != "unknown" else None,
            basis=self.cost_basis,
        )
