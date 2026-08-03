"""One decision date (Trial), and one agent invocation within it (Cell)."""

from __future__ import annotations

from datetime import date as Date
from functools import reduce

from pydantic import BaseModel, ConfigDict, Field

from fintel.models.common import (
    PORTFOLIO_CELL,
    DecisionScope,
    HealthStatus,
    Outcome,
    Status,
    Symbol,
)
from fintel.models.decision import View
from fintel.models.trace import Usage


class CellConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell: str
    symbols: list[Symbol]

    @property
    def is_portfolio(self) -> bool:
        return self.cell == PORTFOLIO_CELL


class CellResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell: str
    symbols: list[Symbol]
    status: Status
    n_views: int = 0
    attempts: int = 1
    duration_ms: int = 0
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None
    health: HealthStatus = "ok"
    health_issues: list[str] = Field(default_factory=list)


class CellRecord(BaseModel):
    """The rich on-disk record for one cell: `runs/<job>/rK/trials/<date>/cells/<cell>.json`.

    `CellResult` is the reduced form the trial/run/job roll up; `CellRecord` is
    the full per-cell artifact a reviewer reads — the outcome, the views (with
    their cited sources), the usage, the environment summary and health, and
    timing. Promoting it from a raw dict to a model gives a typed round-trip
    (the views' `sources_cited` survive a read-back as `SourceRef`, not bare
    dicts) and one schema to point the CLI's reader at.
    """

    cell: str
    symbols: list[Symbol]
    decision_date: str
    scope: DecisionScope
    outcome: Outcome
    detail: str = ""
    n_views: int = 0
    usage: Usage = Field(default_factory=Usage)
    views: dict[Symbol, View] = Field(default_factory=dict)
    environment: dict = Field(default_factory=dict)
    elapsed_ms: int = 0
    started_at: str = ""


class TrialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    decision_date: Date
    universe: list[Symbol]
    scope: DecisionScope
    next_decision_date: Date | None = None

    def cells(self) -> list[CellConfig]:
        if self.scope == "portfolio":
            return [CellConfig(cell=PORTFOLIO_CELL, symbols=list(self.universe))]
        return [CellConfig(cell=s, symbols=[s]) for s in self.universe]


class TrialResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_date: Date
    status: Status
    cells: list[CellResult] = Field(default_factory=list)
    n_views: int = 0
    missing: list[Symbol] = Field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: int = 0
    error: str | None = None
    health: HealthStatus = "ok"

    @property
    def usage(self) -> Usage:
        return reduce(lambda a, c: a.merge(c.usage), self.cells, Usage())
