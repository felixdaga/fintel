"""Where strategy, market and runtime meet.

  strategy → which kinds, with which params, and what one cell decides on
  market   → the universe at the decision date, and the bound PIT-clamped sources
  runtime  → where a cell may write, and how much it may ask for

None of these is allowed to imply the others. The universe is resolved *at the
decision date* rather than taken from the manifest, because an index's membership
changes and a cell must be judged against the world as it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from fintel.environment.access import DataAccess
from fintel.environment.base import Environment
from fintel.environment.cell import Cell
from fintel.environment.policy import AccessPolicy, PolicyBuilder
from fintel.environment.session import SessionDir
from fintel.environment.tools import ToolSurface
from fintel.environment.trace import AccessLog
from fintel.market.data.base import DataSource
from fintel.models.common import DecisionScope, Symbol


@dataclass
class RuntimeConfig:
    """The runtime half: limits and where a cell may write.

    Kept separate from the strategy so the same package can run under different
    budgets without editing its manifest.
    """

    session_root: Path | None = None
    trace: bool = True
    reset_sessions: bool = False
    limits: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.session_root, str):
            self.session_root = Path(self.session_root)


def cells_for(
    *, run_id: str, decision_date: date, symbols: list[Symbol], scope: DecisionScope
) -> list[Cell]:
    """The fan-out for one decision date. This is where scope becomes work.

    A portfolio-scope strategy gets one cell that sees the whole universe; a
    single-name strategy gets one cell per symbol, each isolated from the others.
    """
    if not symbols:
        return []
    if scope == "portfolio":
        return [
            Cell(
                run_id=run_id,
                decision_date=decision_date,
                symbols=tuple(symbols),
                scope="portfolio",
            )
        ]
    return [
        Cell(run_id=run_id, decision_date=decision_date, symbols=(symbol,), scope="single_name")
        for symbol in symbols
    ]


def build_policy(
    *,
    cell: Cell,
    kinds: tuple[str, ...],
    universe: list[Symbol],
    peers: bool = False,
    limits: dict | None = None,
) -> AccessPolicy:
    """What this cell may read and decide.

    A single-name cell decides on its own symbol only. Whether it may *read* the
    rest of the universe is the strategy's call: comparing against peers is
    ordinary analysis, but it is also a wider information set, so it's explicit.
    """
    return PolicyBuilder(
        kinds=kinds,
        decidable=cell.symbols,
        peers=tuple(universe) if peers else (),
        limits=limits or {},
    ).build()


def build_environment(
    *,
    cell: Cell,
    sources: dict[str, DataSource],
    universe: list[Symbol],
    kinds: tuple[str, ...] | None = None,
    peers: bool = False,
    runtime: RuntimeConfig | None = None,
) -> Environment:
    """Assemble one cell's environment.

    `kinds` defaults to whatever is bound, so a caller can't accidentally grant a
    kind the strategy didn't declare — the policy is built from the declaration,
    and `DataAccess.kinds` is the intersection with what's actually bound.
    """
    runtime = runtime or RuntimeConfig()
    policy = build_policy(
        cell=cell,
        kinds=kinds if kinds is not None else tuple(sources),
        universe=universe,
        peers=peers,
        limits=runtime.limits,
    )

    session = None
    trace_path = None
    if runtime.session_root is not None:
        session = SessionDir(root=runtime.session_root, cell=cell)
        session.create(reset=runtime.reset_sessions)
        trace_path = session.trace if runtime.trace else None

    log = AccessLog(cell=cell, path=trace_path)
    access = DataAccess(cell=cell, sources=sources, policy=policy, on_read=log.record)
    return Environment(
        cell=cell,
        access=access,
        policy=policy,
        log=log,
        tools=ToolSurface(
            access=access,
            bound={kind: getattr(src, "name", "") for kind, src in sources.items()},
        ),
        session=session,
    )
