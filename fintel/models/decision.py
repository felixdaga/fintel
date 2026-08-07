"""What the agent says, and what gets persisted per decision date.

The agent emits opinions, not orders. How opinions become a quality number is
the strategy's KPI.
"""

from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fintel.models.common import RETRYABLE, Outcome, Symbol, TimeHorizon
from fintel.models.trace import ReasoningTrace, Usage


class SourceRef(BaseModel):
    """A pointer to the datum that moved a view. `source_type` is open — a
    package registering a novel data kind must be able to cite it."""

    source_type: str
    source_id: str
    # Optional: only persist when the strategy schema / submitter provides it.
    relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    excerpt: str | None = None


class View(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: Symbol
    score: float = Field(..., ge=-1.0, le=1.0)
    # Optional platform fields — omit unless the submitter (or package schema)
    # actually provides them. Emit must not invent defaults for fields the
    # strategy output contract does not ask for.
    conviction: float | None = Field(default=None, ge=0.0, le=1.0)
    time_horizon: TimeHorizon | None = None
    rationale: str = ""
    key_factors: list[str] = Field(default_factory=list)
    sources_cited: list[SourceRef] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """One agent invocation's result, including how it ended.

    Partial views alongside a failed `outcome` are allowed and wanted: an agent
    that timed out after covering three of five names should keep those three.
    """

    views: dict[Symbol, View]
    outcome: Outcome = "ok"
    detail: str = ""
    usage: Usage = Field(default_factory=Usage)
    trace: ReasoningTrace = Field(default_factory=ReasoningTrace)

    @field_validator("views")
    @classmethod
    def _keys_match(cls, v: dict[Symbol, View]) -> dict[Symbol, View]:
        for sym, view in v.items():
            if view.symbol != sym:
                raise ValueError(f"view key {sym!r} != view.symbol {view.symbol!r}")
        return v

    @model_validator(mode="after")
    def _outcome_matches_views(self) -> AgentResponse:
        # Makes the ambiguity unrepresentable: a caller holding an empty
        # response is forced to have said why it is empty.
        if self.outcome == "ok" and not self.views:
            raise ValueError(
                "outcome 'ok' with no views — say why it is empty: "
                "'abstained' if the agent declined, 'empty' if it went quiet, "
                "or the specific failure"
            )
        return self

    @property
    def retryable(self) -> bool:
        return self.outcome in RETRYABLE


class Decision(BaseModel):
    """One decision date's record: `runs/<job>/rK/trials/<date>/decision.json`."""

    run_id: str
    decision_date: Date
    universe: list[Symbol]
    agent_response: AgentResponse
    context_hash: str
