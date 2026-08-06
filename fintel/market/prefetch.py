"""Warm the on-disk cache for a whole run before any agent runs.

Why this exists: OpenClaw fires MCP tool calls in parallel over one stdio
server, and the host's per-request timeout is ~60s. A cold Massive fill
(even one that ultimately succeeds in-process) can blow past that, so the
agent sees ``-32001`` and abstains — a harness failure that looks like a
quiet market. Delorean avoided it in practice by prefetching the whole
universe into a warm cache before the agent loop, so every tool call is a
cache hit in milliseconds.

This module does the same for fintel: one pass over the union of symbols
the run will touch, parallel across symbols, calling each bound source's
``_ensure`` so coverage gaps are filled once. Skip-if-warm is free —
``_ensure`` returns immediately when coverage already spans the request.

Runtime-only kinds (``web_search``) are never prefetched: their query text
isn't known until the agent asks, and the cache is keyed by query, not symbol.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from typing import Any

from fintel.market.data.base import DataSource
from fintel.market.data.filings import MassiveFilingText
from fintel.market.data.massive import MassivePrices, MassiveRecords
from fintel.models.common import Symbol

logger = logging.getLogger(__name__)

# Kinds whose data is determined by the run (symbol + date window) and so
# can be prefetched. web_search is keyed by free-text query → runtime only.
PREFETCHABLE = ("prices", "fundamentals", "news", "filing_text")
RUNTIME_ONLY = ("web_search",)


@dataclass
class PrefetchResult:
    """What happened during the warm-up. Failures don't abort the run."""

    symbols: list[Symbol] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    from_date: Date | None = None
    through_date: Date | None = None
    warmed: list[str] = field(default_factory=list)  # "SYMBOL:kind"
    failed: dict[str, str] = field(default_factory=dict)  # "SYMBOL:kind" -> error
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "symbols": list(self.symbols),
            "kinds": list(self.kinds),
            "from_date": self.from_date.isoformat() if self.from_date else None,
            "through_date": self.through_date.isoformat() if self.through_date else None,
            "n_warmed": len(self.warmed),
            "n_failed": len(self.failed),
            "failed": dict(self.failed),
            "elapsed_ms": self.elapsed_ms,
        }


def prefetch_window(
    decision_dates: list[Date],
    sources: dict[str, DataSource],
    bindings: list[Any] | None = None,
) -> tuple[Date, Date]:
    """The [from, through] window a prefetch must cover.

    ``through`` is the last decision date minus one day (PIT: an agent at the
    last date may see up to the prior day). ``from`` is the earliest decision
    date minus the largest lookback any bound kind declares, so a cell at the
    first date asking for 730 days of fundamentals is still a cache hit.

    This is the *widest* window — used for the prefetch progress report and as
    a safety net. Per-kind warming (:func:`prefetch`) uses each kind's own
    lookback so a high-volume kind like news doesn't fetch two years of articles
    just because fundamentals declares a 730-day lookback.
    """
    if not decision_dates:
        return Date.today(), Date.today()
    through = max(decision_dates) - timedelta(days=1)

    lookbacks: list[int] = []
    for kind, src in sources.items():
        lb = _lookback_days(kind, src, bindings)
        if lb:
            lookbacks.append(lb)
    # Also honour a source's own history_start (prices carry one).
    starts: list[Date] = []
    for src in sources.values():
        hs = getattr(src, "history_start", None)
        if hs is not None:
            starts.append(hs)

    max_lookback = max(lookbacks) if lookbacks else 365
    floor = min(decision_dates) - timedelta(days=max_lookback)
    if starts:
        floor = min(floor, min(starts))
    return floor, through


def _kind_from(sources: dict[str, DataSource], kind: str, decision_dates: list[Date]) -> Date:
    """The floor for warming one kind: its own lookback, not the global max.

    News declares 90 days; fundamentals 730. Warming news from the global
    730-day floor fetches ~2 years of articles for popular tickers — minutes
    of sequential cursoring for what an agent will only ever read 90 days of.
    Each kind is warmed from its own declared lookback behind the earliest
    decision date.
    """
    src = sources.get(kind)
    if src is None:
        return min(decision_dates)
    lb = _lookback_days(kind, src, None)
    if lb is None:
        return min(decision_dates)
    # A source's own history_start (prices) is an absolute floor we still honour.
    hs = getattr(src, "history_start", None)
    floor = min(decision_dates) - timedelta(days=lb)
    if hs is not None:
        floor = min(floor, hs)
    return floor


def _lookback_days(kind: str, src: DataSource, bindings: list[Any] | None) -> int | None:
    # The source instance carries the resolved lookback (binding → catalog
    # default), baked in by the factory. `bindings` is kept for signature
    # stability only; the instance is the single source of truth.
    lb = getattr(src, "lookback_days", None)
    if lb is not None:
        return int(lb)
    spec = getattr(src, "spec", None)
    if spec is not None and hasattr(spec, "lookback_days"):
        return int(spec.lookback_days)
    return None


def prefetch(
    *,
    symbols: list[Symbol],
    sources: dict[str, DataSource],
    from_date: Date,
    through_date: Date,
    workers: int = 8,
    decision_dates: list[Date] | None = None,
) -> PrefetchResult:
    """Warm the cache for every (symbol, kind) the run will touch.

    Parallel across symbols; kinds within a symbol run sequentially (they
    share one HTTP client and one cache lock per symbol). A single symbol
    failing is recorded, not raised — a prefetch gap becomes a cold fill
    at cell time, which the access log already grades as `failed`.

    When ``decision_dates`` is provided, each kind is warmed from its *own*
    declared lookback behind the earliest decision date, not the global max
    lookback. News declares 90 days; warming it from the global 730-day
    fundamentals floor fetches ~2 years of articles for popular tickers —
    minutes of work for 90 days of need. Without ``decision_dates`` the
    global ``from_date`` is used for every kind (backward compatible).
    """
    import time

    started = time.perf_counter()
    kinds = sorted(k for k in sources if k in PREFETCHABLE)
    result = PrefetchResult(
        symbols=list(symbols),
        kinds=kinds,
        from_date=from_date,
        through_date=through_date,
    )

    if not symbols or not kinds:
        return result

    def _warm_one(symbol: Symbol) -> list[tuple[str, str | None]]:
        out: list[tuple[str, str | None]] = []
        for kind in kinds:
            label = f"{symbol}:{kind}"
            try:
                if decision_dates:
                    kind_from = _kind_from(sources, kind, decision_dates)
                else:
                    kind_from = from_date
                _warm_kind(sources[kind], kind, symbol, kind_from, through_date)
                out.append((label, None))
            except Exception as exc:  # noqa: BLE001 — one symbol's gap must not abort the warm-up
                logger.warning("prefetch %s failed: %s", label, exc)
                out.append((label, f"{type(exc).__name__}: {exc}"))
        return out

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_warm_one, s): s for s in symbols}
        for fut in as_completed(futures):
            for label, err in fut.result():
                if err is None:
                    result.warmed.append(label)
                else:
                    result.failed[label] = err

    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result


def _warm_kind(
    src: DataSource, kind: str, symbol: Symbol, from_date: Date, through_date: Date
) -> None:
    """Call the source's gap-fill once. Skip-if-warm is handled inside `_ensure`."""
    if isinstance(src, MassivePrices):
        src._ensure(symbol, through_date)  # noqa: SLF001 — gap-fill is the warm path
        return
    if isinstance(src, MassiveRecords):
        # Records clamp on availability (filing_date / published_at). Warm the
        # full window the agent might ask for; `_ensure` splits out gaps.
        src._ensure(symbol, from_date, through_date)  # noqa: SLF001
        return
    if isinstance(src, MassiveFilingText):
        src._ensure(symbol, from_date, through_date)  # noqa: SLF001
        return
    # Unknown source type: nothing to warm (e.g. computed ratios pull from
    # upstream at fetch time, and web_search is runtime-only). Skip silently.
    logger.debug("prefetch: no warm path for %s (%s), skipping", kind, type(src).__name__)


def is_prefetchable(kind: str) -> bool:
    return kind in PREFETCHABLE


def is_runtime_only(kind: str) -> bool:
    return kind in RUNTIME_ONLY
