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
from fintel.environment.progress import Progress
from fintel.environment.session import SessionDir
from fintel.environment.tools import ToolSurface
from fintel.environment.trace import AccessLog
from fintel.market.data.base import DataSource
from fintel.market.settings import MarketConfig
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
    lookback_caps: dict[str, int] | None = None,
    render_caps: dict[str, dict[str, int]] | None = None,
    limits: dict | None = None,
) -> AccessPolicy:
    """What this cell may read and decide.

    A single-name cell decides on its own symbol only. Whether it may *read* the
    rest of the universe is the strategy's call: comparing against peers is
    ordinary analysis, but it is also a wider information set, so it's explicit.

    `lookback_caps` is per-kind, read from each bound source's `lookback_days`
    (which the factory baked from the strategy binding). A caller may request
    less history, never more than the strategy declared.
    """
    return PolicyBuilder(
        kinds=kinds,
        decidable=cell.symbols,
        peers=tuple(universe) if peers else (),
        lookback_caps=dict(lookback_caps or {}),
        render_caps=dict(render_caps or {}),
        limits=limits or {},
    ).build()


def _lookback_caps_from(sources: dict[str, DataSource]) -> dict[str, int]:
    """Per-kind lookback cap from each source's resolved `lookback_days`."""
    caps: dict[str, int] = {}
    for kind, src in sources.items():
        lb = getattr(src, "lookback_days", None)
        if lb is None:
            spec = getattr(src, "spec", None)
            if spec is not None and hasattr(spec, "lookback_days"):
                lb = spec.lookback_days
        if lb is not None:
            caps[kind] = int(lb)
    return caps


def _render_caps_from(sources: dict[str, DataSource]) -> dict[str, dict[str, int]]:
    """Per-kind render caps carried by each source (e.g. snippet_max_chars for
    web_search, summary_max_chars for news). These are evidence-rendering
    knobs, not fetch knobs; the source acts as a carrier for the resolved
    binding/catalog default so the policy can surface them to the renderer.
    """
    out: dict[str, dict[str, int]] = {}
    for kind, src in sources.items():
        rc = getattr(src, "render_caps", None)
        if rc:
            out[kind] = dict(rc)
        smc = getattr(src, "snippet_max_chars", None)
        if smc is not None and kind == "web_search":
            out.setdefault(kind, {})["snippet_max_chars"] = int(smc)
    return out


def build_environment(
    *,
    cell: Cell,
    sources: dict[str, DataSource],
    universe: list[Symbol],
    kinds: tuple[str, ...] | None = None,
    peers: bool = False,
    runtime: RuntimeConfig | None = None,
    market_config: MarketConfig | None = None,
    nerve: Progress | None = None,
) -> Environment:
    """Assemble one cell's environment.

    `kinds` defaults to whatever is bound, so a caller can't accidentally grant a
    kind the strategy didn't declare — the policy is built from the declaration,
    and `DataAccess.kinds` is the intersection with what's actually bound.

    `market_config` is carried so a subprocess MCP server can rebuild against
    the same cache the orchestrator used (written into bindings.json).
    """
    runtime = runtime or RuntimeConfig()
    lookback_caps = _lookback_caps_from(sources)
    render_caps = _render_caps_from(sources)
    policy = build_policy(
        cell=cell,
        kinds=kinds if kinds is not None else tuple(sources),
        universe=universe,
        peers=peers,
        lookback_caps=lookback_caps,
        render_caps=render_caps,
        limits=runtime.limits,
    )

    session = None
    trace_path = None
    if runtime.session_root is not None:
        session = SessionDir(root=runtime.session_root, cell=cell)
        session.create(reset=runtime.reset_sessions)
        trace_path = session.trace if runtime.trace else None

    log = AccessLog(cell=cell, path=trace_path, nerve=nerve)
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
        market_config=market_config,
        nerve=nerve,
    )
