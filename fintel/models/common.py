from __future__ import annotations

from typing import Literal

Symbol = str

# The agent's thesis horizon, decoupled from the platform's rebalance cadence.
# Left open: a package's output_schema.json owns its own vocabulary.
TimeHorizon = str

DecisionScope = Literal["single_name", "portfolio"]
Status = Literal["ok", "partial", "empty", "failed", "timeout", "skipped"]

PORTFOLIO_CELL = "__portfolio__"
