"""Daily mark-to-market NAV for the public strategy page.

Holdings still change only on decision dates. The published return series is
every trading day: the last book's forward return between consecutive sessions,
with turnover cost only on rebalance days (the interval that starts on a
decision date). Cadence does not change the NAV frequency.
"""

from __future__ import annotations

from datetime import date as Date
from math import ceil
from typing import Callable, Protocol

from fintel.market.calendar import TradingCalendar


class Prices(Protocol):
    def forward_return(self, symbol: str, start: Date, end: Date) -> float | None: ...
    def price_at(self, symbol: str, on: Date) -> float | None: ...


def last_on_or_before(dates: list[Date], on: Date) -> Date | None:
    eligible = [d for d in dates if d <= on]
    return max(eligible) if eligible else None


def weighted_return(
    weights: dict[str, float], start: Date, end: Date, prices: Prices
) -> float:
    if not weights:
        return 0.0
    fwd = {
        s: r
        for s in weights
        if (r := prices.forward_return(s, start, end)) is not None
    }
    if not fwd:
        return 0.0
    mass = sum(weights[s] for s in fwd) or 1.0
    return sum((weights[s] / mass) * fwd[s] for s in fwd)


def price_weighted(symbols: list[str], on: Date, prices: Prices) -> dict[str, float]:
    px: dict[str, float] = {}
    for s in symbols:
        p = prices.price_at(s, on)
        if p is not None and p > 0:
            px[s] = float(p)
    total = sum(px.values())
    return {s: p / total for s, p in px.items()} if total > 0 else {}


def daily_grid(start: Date, end: Date, cal: TradingCalendar | None = None) -> list[Date]:
    """Trading days in [start, end], snapping start forward if it is a holiday."""
    cal = cal or TradingCalendar()
    start = cal.snap_forward(start)
    if end < start:
        return []
    return cal.days(start, end)


def observed_price_days(
    prices: Prices,
    symbols: list[str],
    start: Date,
    end: Date,
    *,
    min_coverage: float = 0.5,
    cal: TradingCalendar | None = None,
) -> list[Date]:
    """Trading days in [start, end] that actually have bars in the price cache.

    ``PriceLookup.price_at`` forward-fills, so a calendar grid over a cache gap
    prints a fake zero return. Prefer dates present on at least
    ``min_coverage`` of ``symbols``. No store (tests / stubs) falls back to
    the NYSE calendar.
    """
    cal = cal or TradingCalendar()
    start = cal.snap_forward(start)
    if end < start:
        return []
    store = getattr(prices, "store", None)
    read = getattr(store, "read", None) if store is not None else None
    if read is None:
        return cal.days(start, end)

    n = 0
    counts: dict[Date, int] = {}
    for s in symbols:
        df = read(s)
        if df is None or getattr(df, "empty", True) or "date" not in getattr(df, "columns", []):
            continue
        n += 1
        for raw in df["date"]:
            d = raw.date() if hasattr(raw, "date") and not isinstance(raw, Date) else raw
            if isinstance(d, Date) and start <= d <= end and cal.is_trading_day(d):
                counts[d] = counts.get(d, 0) + 1
    if not n:
        return cal.days(start, end)
    need = max(1, ceil(n * min_coverage))
    return sorted(d for d, c in counts.items() if c >= need)


def walk_nav(
    *,
    daily: list[Date],
    decision_dates: list[Date],
    book_on: Callable[[Date], dict[str, float]],
    prices: Prices,
    cost_bps: float,
    first_rebalance_free: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Gross and net NAV points on ``daily``.

    Rebalance at the close of a decision date: the interval *starting* that
    day uses the new book. Turnover cost hits that first step (except the
    inception rebalance, which is free). Intra-horizon days are a hold.
    """
    if not daily:
        return [], []
    decision_set = set(decision_dates)
    nav_g = nav_n = 1.0
    prev_w: dict[str, float] = {}
    gross = [{"date": daily[0].isoformat(), "nav": 1.0}]
    net = [{"date": daily[0].isoformat(), "nav": 1.0}]
    for i in range(len(daily) - 1):
        prev, cur = daily[i], daily[i + 1]
        weights = book_on(prev)
        r = weighted_return(weights, prev, cur, prices)
        turnover = sum(
            abs(weights.get(s, 0.0) - prev_w.get(s, 0.0))
            for s in set(weights) | set(prev_w)
        )
        is_rebalance = prev in decision_set and not (first_rebalance_free and i == 0)
        cost = turnover * cost_bps / 10000.0 if is_rebalance else 0.0
        nav_g *= 1.0 + r
        nav_n *= 1.0 + r - cost
        gross.append({"date": cur.isoformat(), "nav": round(nav_g, 6)})
        net.append({"date": cur.isoformat(), "nav": round(nav_n, 6)})
        prev_w = dict(weights)
    return gross, net


def walk_benchmark(
    daily: list[Date], universe: list[str], prices: Prices
) -> list[dict]:
    """Price-weighted universe, marked daily on the same grid as F1."""
    if not daily:
        return []
    nav = 1.0
    points = [{"date": daily[0].isoformat(), "nav": 1.0}]
    for i in range(len(daily) - 1):
        prev, cur = daily[i], daily[i + 1]
        weights = price_weighted(universe, prev, prices)
        nav *= 1.0 + weighted_return(weights, prev, cur, prices)
        points.append({"date": cur.isoformat(), "nav": round(nav, 6)})
    return points
