"""Gap-aware cache-first fills for daily price bars (parquet / PriceStore)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date as Date

import pandas as pd

from fintel.market.data import coverage as cov
from fintel.market.data.base import DataError
from fintel.market.data.store import PriceStore

logger = logging.getLogger(__name__)

FetchBars = Callable[[Date, Date], pd.DataFrame | None]


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
    """
    if through < need_from:
        return store.read(symbol)
    gaps = cov.missing(store.coverage(symbol), need_from, through)
    if not gaps:
        return store.read(symbol)
    if not online:
        cached = store.read(symbol)
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
            len(gaps),
        )
        return cached
    for lo, hi in gaps:
        fresh = fetch_bars(lo, hi)
        if fresh is not None and not fresh.empty:
            store.merge(symbol, fresh, (lo, hi))
        else:
            store.record_empty_span(symbol, (lo, hi))
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
