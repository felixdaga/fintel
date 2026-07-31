from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


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
    """Token/cost rollup. `cost_usd` stays None rather than a misleading 0."""

    n_llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    cost_source: str | None = None

    def merge(self, other: Usage) -> Usage:
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
            cost_source=self.cost_source or other.cost_source,
        )
