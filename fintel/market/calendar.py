"""US equity trading calendar.

Load-bearing, not a nicety: the reference package's decision grid includes
2023-01-01 (a Sunday) and three New Year holidays. A weekday-only calendar
calls 2024-01-01 a trading day, so it cannot tell you your decision date has no
prices. Both the schedule grid and the scoring price lookup depend on getting
this right.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from functools import lru_cache

# Days the NYSE closed outside the standard holiday rules. Only affects
# daily-frequency work, but a missing entry looks exactly like missing data.
SPECIAL_CLOSURES: frozenset[Date] = frozenset(
    {
        Date(2001, 9, 11),
        Date(2001, 9, 12),
        Date(2001, 9, 13),
        Date(2001, 9, 14),
        Date(2004, 6, 11),  # Reagan state funeral
        Date(2007, 1, 2),  # Ford state funeral
        Date(2012, 10, 29),  # Hurricane Sandy
        Date(2012, 10, 30),
        Date(2018, 12, 5),  # Bush state funeral
        Date(2025, 1, 9),  # Carter state funeral
    }
)


def easter(year: int) -> Date:
    """Gregorian Easter Sunday (Meeus/Jones/Butcher). Good Friday hangs off it."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lam = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lam) // 451
    month, day = divmod(h + lam - 7 * m + 114, 31)
    return Date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> Date:
    """`n`-th `weekday` of a month; n=-1 means the last one."""
    if n > 0:
        first = Date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    nxt = Date(year + 1, 1, 1) if month == 12 else Date(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: Date, *, shift_saturday: bool = True) -> Date | None:
    """NYSE observance: Sunday rolls forward, Saturday rolls back.

    New Year's Day is the exception — the exchange does not close on Dec 31 for
    a Saturday Jan 1, so it passes `shift_saturday=False` and drops out.
    """
    if d.weekday() == 6:
        return d + timedelta(days=1)
    if d.weekday() == 5:
        return d - timedelta(days=1) if shift_saturday else None
    return d


@lru_cache(maxsize=256)
def market_holidays(year: int) -> frozenset[Date]:
    """Observed NYSE holidays for one calendar year."""
    out: set[Date] = set()

    def add(d: Date | None) -> None:
        if d is not None and d.year == year:
            out.add(d)

    add(_observed(Date(year, 1, 1), shift_saturday=False))
    if year >= 1998:
        out.add(_nth_weekday(year, 1, 0, 3))  # MLK Day
    out.add(_nth_weekday(year, 2, 0, 3))  # Washington's Birthday
    out.add(easter(year) - timedelta(days=2))  # Good Friday
    out.add(_nth_weekday(year, 5, 0, -1))  # Memorial Day
    if year >= 2022:
        add(_observed(Date(year, 6, 19)))  # Juneteenth
    add(_observed(Date(year, 7, 4)))
    out.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
    out.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    add(_observed(Date(year, 12, 25)))
    return frozenset(out)


@dataclass(frozen=True)
class TradingCalendar:
    name: str = "nyse"
    extra_closures: frozenset[Date] = field(default=SPECIAL_CLOSURES)

    def is_trading_day(self, d: Date) -> bool:
        return d.weekday() < 5 and d not in market_holidays(d.year) and d not in self.extra_closures

    def next(self, d: Date, n: int = 1) -> Date:
        """`n` trading days forward; negative goes back. n=0 returns `d` as given."""
        if n == 0:
            return d
        step = timedelta(days=1 if n > 0 else -1)
        remaining = abs(n)
        out = d
        while remaining:
            out += step
            if self.is_trading_day(out):
                remaining -= 1
        return out

    def prev(self, d: Date, n: int = 1) -> Date:
        return self.next(d, -n)

    def snap_forward(self, d: Date) -> Date:
        """`d` if it trades, else the next trading day."""
        return d if self.is_trading_day(d) else self.next(d)

    def snap_back(self, d: Date) -> Date:
        return d if self.is_trading_day(d) else self.prev(d)

    def days(self, start: Date, end: Date) -> list[Date]:
        """Trading days in [start, end]."""
        out: list[Date] = []
        d = start
        while d <= end:
            if self.is_trading_day(d):
                out.append(d)
            d += timedelta(days=1)
        return out

    def non_trading(self, dates: list[Date]) -> list[Date]:
        """The audit direction — which of these dates have no session."""
        return [d for d in dates if not self.is_trading_day(d)]

    def interior_missing_sessions(
        self, have: Iterable[Date], start: Date, end: Date
    ) -> list[Date]:
        """NYSE sessions between the first and last date in ``have`` ∩ [start, end]
        that are absent from ``have``.

        Days before the first bar (IPO / entitlement floor) and after the last
        bar (today's session not published yet) are not holes — those are
        empty-span / not-yet-fetched, not a truncated fill.
        """
        in_window = [d for d in have if start <= d <= end]
        if len(in_window) < 2:
            return []
        have_set = set(have)
        return [d for d in self.days(min(in_window), max(in_window)) if d not in have_set]
