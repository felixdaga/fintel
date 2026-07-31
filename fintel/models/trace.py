from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from functools import reduce
from typing import Literal

from pydantic import BaseModel, Field

# `reported` = the provider told us the charge. `estimated` = we derived it from
# token counts and a rate card. `mixed` = a sum of both, so not comparable.
CostBasis = Literal["reported", "estimated", "mixed", "unknown"]


class TraceStep(BaseModel):
    step_id: str
    kind: str  # llm_call | tool_call | subagent | thought | ...
    started_at: datetime
    duration_ms: int = 0
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    cache_hit: bool = False
    parent_step_id: str | None = None
    payload: dict = Field(default_factory=dict)


class ReasoningTrace(BaseModel):
    final_explanation: str = ""
    steps: list[TraceStep] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class Usage(BaseModel):
    """Token/cost rollup. `cost_usd` stays None rather than a misleading 0.

    `basis` is load-bearing for the platform's whole purpose. A subprocess CLI
    usually reports tokens but no price, so its cost has to be derived from a
    published rate card, while an HTTP agent returns the charge the provider
    actually made. Adding those two produces a number that is not comparable to
    either, so a mixed rollup says so instead of picking a label. The old
    pipeline stamped a total `openrouter_authoritative` whenever *any* leg
    reported a cost, which made an estimate look like a measurement.
    """

    n_llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    basis: CostBasis = "unknown"

    @property
    def empty(self) -> bool:
        """Contributed nothing, so it can't dilute a rollup's basis."""
        return self.n_llm_calls == 0 and self.cost_usd is None

    @property
    def comparable(self) -> bool:
        """Whether this cost may be used to rank agents against each other."""
        return self.cost_usd is not None and self.basis in ("reported", "estimated")

    def merge(self, other: Usage) -> Usage:
        if self.empty:
            return other
        if other.empty:
            return self
        cost: float | None
        if self.cost_usd is None and other.cost_usd is None:
            cost = None
        else:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            n_llm_calls=self.n_llm_calls + other.n_llm_calls,
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost_usd=cost,
            basis=self.basis if self.basis == other.basis else "mixed",
        )


def total(usages: Iterable[Usage]) -> Usage:
    return reduce(lambda a, b: a.merge(b), usages, Usage())
