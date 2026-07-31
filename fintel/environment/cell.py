"""The identity of one agent invocation. Immutable, explicit, and self-contained.

In the old code a running MCP server discovered which cell it was serving by
walking process ancestry to find the CLI's pid and reading a file in a directory
named after it. Under a gateway that spawns the server itself, the ancestry walk
finds nothing, every concurrent call falls back to the same shared directory, and
cells contaminate each other. The server also cached the first cell it ever
loaded, so a reused process served a stale decision date.

The fix is to stop making cell identity ambient. A `Cell` is constructed by the
caller, passed explicitly, frozen, and carries its own `Cutoff` — so there is no
step where a component has to work out what it's serving.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from fintel.models import ids
from fintel.models.common import PORTFOLIO_CELL, DecisionScope, Symbol
from fintel.pit import Cutoff


@dataclass(frozen=True)
class Cell:
    """One agent invocation: who decides what, on which date, seeing what.

    `symbols` is what the agent decides *on*. What it may *read* is the policy's
    business — a single-name cell often needs peer data to form a view.
    """

    run_id: str
    decision_date: date
    symbols: tuple[Symbol, ...]
    scope: DecisionScope = "single_name"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("cell needs a run_id")
        if not self.symbols:
            raise ValueError("cell needs at least one symbol")
        if self.scope == "single_name" and len(self.symbols) != 1:
            raise ValueError(
                f"single_name cell decides on exactly one symbol, got {list(self.symbols)}; "
                f"use scope='portfolio' to decide across a universe"
            )
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError(f"duplicate symbols in cell: {list(self.symbols)}")

    @property
    def cutoff(self) -> Cutoff:
        """The point-in-time boundary. Derived, so it can't disagree with the date."""
        return Cutoff(self.decision_date)

    @property
    def trial(self) -> str:
        return ids.trial_id(self.decision_date)

    @property
    def name(self) -> str:
        """The cell id: the symbol, or the portfolio sentinel."""
        return ids.cell_id(self.symbols[0] if self.scope == "single_name" else None)

    @property
    def is_portfolio(self) -> bool:
        return self.scope == "portfolio"

    @property
    def key(self) -> str:
        """Unique across a job. Used for session dirs and artifact names."""
        return f"{self.run_id}/{self.trial}/{self.name}"

    def describe(self) -> str:
        subject = self.symbols[0] if self.scope == "single_name" else PORTFOLIO_CELL
        return f"{subject} @ {self.decision_date.isoformat()}"
