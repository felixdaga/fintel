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
from fintel.environment.progress import Progress
from fintel.environment.session import SessionDir
from fintel.environment.tools import ToolSurface
from fintel.environment.trace import AccessLog
from fintel.market.settings import MarketConfig


@dataclass
class Environment:
    """What one agent invocation is allowed to know.

    `tools` and `evidence` are two presentations of the same `access`, not two
    data paths — so an agent given tools and an agent given text are looking at
    the same PIT-clamped, policy-checked world.

    `nerve` is the run's emit surface (the `Nerve`), threaded onto the
    environment so agents can emit *live staging* events (a reasoning turn, a
    tool call, a stall) without each adapter re-deriving where events go. It
    is the run stream; the per-cell `log` (AccessLog) is the PIT audit stream —
    two concerns, two sinks, one owner (the environment module).
    """

    cell: Cell
    access: DataAccess
    policy: AccessPolicy
    log: AccessLog
    tools: ToolSurface
    session: SessionDir | None = None
    # Persisted into bindings.json so a subprocess MCP server rebuilds against
    # the same cache (and offline flag) the orchestrator used. Keys are not
    # stored here — they ride in the MCP server's env block.
    market_config: MarketConfig | None = None
    nerve: Progress | None = None

    @property
    def kinds(self) -> tuple[str, ...]:
        return self.access.kinds

    def evidence(self, *, symbol: str | None = None) -> str:
        from fintel.environment import evidence as render

        return render.build(self.access, symbol=symbol)

    def summary(self) -> dict:
        """For the cell record: what was asked, and whether it was answered."""
        return {
            "cell": self.cell.name,
            "decision_date": self.cell.decision_date.isoformat(),
            "kinds": list(self.kinds),
            **self._trace_summary(),
        }

    def close(self) -> dict:
        """Close the cell. Prefer the on-disk trace so MCP reads are counted."""
        fields = self._trace_summary()
        self.log.append("cell_closed", **fields)
        return {
            "cell": self.cell.name,
            "decision_date": self.cell.decision_date.isoformat(),
            "kinds": list(self.kinds),
            **fields,
        }

    def _trace_summary(self) -> dict:
        """In-memory events plus anything a subprocess already flushed to disk."""
        from fintel.environment.trace import load, summary_from_events

        if self.session is not None and self.session.trace.is_file():
            return summary_from_events(load(self.session.trace))
        return self.log.summary()
