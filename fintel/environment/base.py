"""The environment: strategy, market and runtime brought together for one cell.

One object per agent invocation, holding everything that invocation may see and
everything it may not. Built by `environment/factory.py`; consumed by an agent
adapter, which decides whether to hand the agent tools or a rendered pack.
"""

from __future__ import annotations

from dataclasses import dataclass

from fintel.environment.access import DataAccess
from fintel.environment.cell import Cell
from fintel.environment.policy import AccessPolicy
from fintel.environment.session import SessionDir
from fintel.environment.tools import ToolSurface
from fintel.environment.trace import AccessLog


@dataclass
class Environment:
    """What one agent invocation is allowed to know.

    `tools` and `evidence` are two presentations of the same `access`, not two
    data paths — so an agent given tools and an agent given text are looking at
    the same PIT-clamped, policy-checked world.
    """

    cell: Cell
    access: DataAccess
    policy: AccessPolicy
    log: AccessLog
    tools: ToolSurface
    session: SessionDir | None = None

    @property
    def kinds(self) -> tuple[str, ...]:
        return self.access.kinds

    def evidence(self, *, symbol: str | None = None) -> str:
        from fintel.environment import evidence as render

        return render.build(self.access, symbol=symbol)

    def summary(self) -> dict:
        """For the cell record: what was asked, and whether it was answered."""
        return {"cell": self.cell.name, "decision_date": self.cell.decision_date.isoformat(),
                "kinds": list(self.kinds), **self.log.summary()}

    def close(self) -> dict:
        self.log.append("cell_closed", **self.log.summary())
        return self.summary()
