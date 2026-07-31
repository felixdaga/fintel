"""What the agent says, and what gets persisted per decision date.

The agent emits opinions, not orders. How opinions become a quality number is
the strategy's KPI.
"""

from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fintel.models.common import Symbol, TimeHorizon
from fintel.models.trace import ReasoningTrace


class SourceRef(BaseModel):
    """A pointer to the datum that moved a view. `source_type` is open — a
    package registering a novel data kind must be able to cite it."""

    source_type: str
    source_id: str
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    excerpt: str | None = None


class View(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: Symbol
    score: float = Field(..., ge=-1.0, le=1.0)
    conviction: float = Field(default=0.5, ge=0.0, le=1.0)
    time_horizon: TimeHorizon = "quarter"
    rationale: str = ""
    key_factors: list[str] = Field(default_factory=list)
    sources_cited: list[SourceRef] = Field(default_factory=list)


class AgentResponse(BaseModel):
    views: dict[Symbol, View]
    trace: ReasoningTrace = Field(default_factory=ReasoningTrace)

    @field_validator("views")
    @classmethod
    def _keys_match(cls, v: dict[Symbol, View]) -> dict[Symbol, View]:
        for sym, view in v.items():
            if view.symbol != sym:
                raise ValueError(f"view key {sym!r} != view.symbol {view.symbol!r}")
        return v


class Decision(BaseModel):
    """One decision date's record: `runs/<job>/rK/trials/<date>/decision.json`."""

    run_id: str
    decision_date: Date
    universe: list[Symbol]
    agent_response: AgentResponse
    context_hash: str
