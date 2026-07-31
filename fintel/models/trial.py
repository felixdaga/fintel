"""One decision date (Trial), and one agent invocation within it (Cell)."""

from __future__ import annotations

from datetime import date as Date
from functools import reduce

from pydantic import BaseModel, ConfigDict, Field

from fintel.models.common import PORTFOLIO_CELL, DecisionScope, Status, Symbol
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

    @property
    def usage(self) -> Usage:
        return reduce(lambda a, c: a.merge(c.usage), self.cells, Usage())
