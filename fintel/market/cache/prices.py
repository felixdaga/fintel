"""Gap-aware cache-first fills for daily price bars (parquet / PriceStore)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime as DateTime

import pandas as pd

from fintel.market.calendar import TradingCalendar
from fintel.market.data import coverage as cov
from fintel.market.data.base import DataError, EntitlementError
from fintel.market.data.store import PriceStore

logger = logging.getLogger(__name__)

FetchBars = Callable[[Date, Date], pd.DataFrame | None]


def _as_date(raw: object) -> Date | None:
    if raw is None:
        return None
    if isinstance(raw, DateTime):
        return raw.date()
    if isinstance(raw, Date):
        return raw


def interior_session_holes(
    df: pd.DataFrame | None,
    need_from: Date,
    through: Date,
    *,
    cal: TradingCalendar | None = None,
) -> list[tuple[Date, Date]]:
    """NYSE sessions between the first and last cached bar that have no row.

    Coverage sidecars record the *fetched window*, including weekends and
    empty vendor responses. A sparse parquet (bars on 08-06 and 08-13, nothing
    in between) still looks fully covered, so ``cov.missing`` will not refetch.
    Interior holes are the opposite: we already have bars on both sides, so
    the missing sessions are a truncated fill, not an empty market.
    """
    if df is None or df.empty or "date" not in df.columns:
        return []
    have: set[Date] = set()
    for raw in df["date"]:
        d = _as_date(raw)
        if d is not None:
            have.add(d)
    in_window = [d for d in have if need_from <= d <= through]
    if len(in_window) < 2:
        return []
    cal = cal or TradingCalendar()
    lo, hi = min(in_window), max(in_window)
    missing = [d for d in cal.days(lo, hi) if d not in have]
    return cov.coalesce([(d, d) for d in missing])


def ensure_prices(
    store: PriceStore,
    symbol: str,
    need_from: Date,
    through: Date,
    *,
    fetch_bars: FetchBars,
    online: bool,
    source_name: str,
) -> pd.DataFrame | None:
    """Return bars covering ``[need_from, through]``, filling gaps when online.

    Empty network spans are recorded so a quiet symbol is not re-fetched forever.
    Interior session holes (sparse parquet inside a covered window) are
    refetched but not recorded empty — a truncated vendor response should
    retry on the next ensure.

    ``EntitlementError`` on a gap (plan doesn't cover that window) is treated
    like an empty fetch: the span is recorded so we stop retrying it, and any
    already-cached bars are still returned. Without this, a single unentitled
    early gap (e.g. delisted WBA before the plan's history floor) aborts the
    whole read even when later bars are already on disk.
    """
    if through < need_from:
        return store.read(symbol)
    cached = store.read(symbol)
    gaps = cov.missing(store.coverage(symbol), need_from, through)
    holes = interior_session_holes(cached, need_from, through)
    to_fetch = cov.coalesce([*gaps, *holes])
    if not to_fetch:
        return cached
    if not online:
        if cached is None:
            raise DataError(
                f"{source_name}: no cached prices for {symbol} covering "
                f"[{need_from}, {through}] and no network access configured"
            )
        logger.warning(
            "%s: %s cache is short of [%s, %s] by %d span(s); serving what is cached",
            source_name,
            symbol,
            need_from,
            through,
            len(to_fetch),
        )
        return cached
    coverage_gaps = set(gaps)
    for lo, hi in to_fetch:
        try:
            fresh = fetch_bars(lo, hi)
        except EntitlementError as exc:
            logger.warning(
                "%s: %s entitlement blocked [%s, %s] — recording empty span, "
                "serving cached bars if any (%s)",
                source_name,
                symbol,
                lo,
                hi,
                exc,
            )
            store.record_empty_span(symbol, (lo, hi))
            continue
        if fresh is not None and not fresh.empty:
            store.merge(symbol, fresh, (lo, hi))
        elif (lo, hi) in coverage_gaps:
            store.record_empty_span(symbol, (lo, hi))
        else:
            logger.warning(
                "%s: %s interior hole [%s, %s] still empty after fetch",
                source_name,
                symbol,
                lo,
                hi,
            )
    return store.read(symbol)


@dataclass
class CachedPricesFeed:
    store: PriceStore
    fetch_bars: Callable[[str, Date, Date], pd.DataFrame | None]
    history_start: Date
    online: bool
    source_name: str

    def ensure(self, key: str, since: Date, through: Date) -> pd.DataFrame | None:
        # Prices warm from history_start (not the request since) so a backtest
        # that jumps dates never leaves early history as a permanent gap.
        need_from = self.history_start
        return ensure_prices(
            self.store,
            key,
            need_from,
            through,
            fetch_bars=lambda lo, hi: self.fetch_bars(key, lo, hi),
            online=self.online,
            source_name=self.source_name,
        )
