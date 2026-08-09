"""Realized prices for scoring — deliberately NOT point-in-time clamped.

A KPI must see what happened *after* the decision date; that's the measurement.
This is the one place in the platform allowed to read the future, so it is a
separate module with a separate name from the agent-facing source. In the old
code both lived on one class as `fetch` and `price_at`, one typo apart.

Named `realized`, not `valuation`: agent-facing valuation *ratios* are a
computed PIT-clamped kind, and two modules called valuation where only one may
read the future is a trap.

Nothing under `environment/` or `agents/` may import this. There's a test.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date as Date
from math import ceil

from fintel.market.calendar import TradingCalendar
from fintel.market.data.store import PriceStore
from fintel.models.common import Symbol

# Entry and exit both use the open, so a return spans decision-to-decision on
# the same field. Simple returns, no dividends — total return would need a
# distribution series this cache doesn't carry.
PRICE_FIELD = "open"


@dataclass(frozen=True)
class PriceLookup:
    store: PriceStore
    price_field: str = PRICE_FIELD
    calendar: TradingCalendar = field(default_factory=TradingCalendar)

    def price_at(self, symbol: Symbol, on: Date) -> float | None:
        """Price on `on`, else the most recent prior session.

        The fallback is why a decision date that isn't a trading day still
        prices: it silently uses the prior session's open. Half the reference
        package's grid relies on this, which is why preflight reports
        non-trading decision dates rather than letting it pass unnoticed.
        """
        bars = self.store.bars_on_or_before(symbol, on)
        if bars is None or bars.empty or self.price_field not in bars.columns:
            return None
        value = bars[self.price_field].iloc[-1]
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        return price if price > 0 else None

    def forward_return(self, symbol: Symbol, start: Date, end: Date) -> float | None:
        p0 = self.price_at(symbol, start)
        p1 = self.price_at(symbol, end)
        if p0 is None or p1 is None or p0 <= 0:
            return None
        return p1 / p0 - 1.0

    def forward_returns(self, symbols: list[Symbol], start: Date, end: Date) -> dict[Symbol, float]:
        """Cross-section of realized returns. Names without prices are absent
        rather than zero — a missing return is not a flat one."""
        out: dict[Symbol, float] = {}
        for sym in symbols:
            r = self.forward_return(sym, start, end)
            if r is not None:
                out[sym] = r
        return out

    def prices_at(self, symbols: list[Symbol], on: Date) -> dict[Symbol, float]:
        out: dict[Symbol, float] = {}
        for sym in symbols:
            p = self.price_at(sym, on)
            if p is not None:
                out[sym] = p
        return out

    def latest_bar_date(
        self,
        symbols: Iterable[Symbol] | None = None,
        *,
        min_coverage: float = 1.0,
    ) -> Date | None:
        """Latest mark date from the price cache for ``symbols``.

        Per-symbol we take the max bar date; the result is the latest date ``d``
        such that at least ``min_coverage`` of the (non-empty) symbols have a bar
        on or after ``d``. Default ``min_coverage=1.0`` is the intersection of
        maxima — every name must be current.

        Empty symbols are skipped (they do not force ``None``). A single stale
        cache file can still pin the date when ``min_coverage`` is 1.0; callers
        that mark a held book should pass that book's names (not the full
        historical universe) and/or warm the cache first.
        """
        if not 0.0 < min_coverage <= 1.0:
            raise ValueError(f"min_coverage must be in (0, 1], got {min_coverage}")
        syms = list(symbols) if symbols is not None else self.store.symbols()
        if not syms:
            return None
        maxima: list[Date] = []
        for sym in syms:
            df = self.store.read(sym)
            if df is None or df.empty:
                continue
            maxima.append(df["date"].iloc[-1])
        if not maxima:
            return None
        n = len(maxima)
        need = max(1, ceil(n * min_coverage))
        for d in sorted(set(maxima), reverse=True):
            if sum(1 for m in maxima if m >= d) >= need:
                return d
        return min(maxima)
