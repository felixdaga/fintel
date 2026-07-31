"""When decisions happen. A schedule is a date generator and nothing more.

Sizing and signal rules are the strategy's; the platform only needs the grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import ClassVar, Protocol, runtime_checkable

from fintel.market.calendar import TradingCalendar

QUARTER_STARTS = ((1, 1), (4, 1), (7, 1), (10, 1))


@runtime_checkable
class Schedule(Protocol):
    kind: str

    def dates(self, start: Date | None = None, end: Date | None = None) -> list[Date]: ...


def _window(
    own: tuple[Date | None, Date | None], asked: tuple[Date | None, Date | None]
) -> tuple[Date | None, Date | None]:
    """Intersect the schedule's own window with the caller's."""
    lo = max((d for d in (own[0], asked[0]) if d is not None), default=None)
    hi = min((d for d in (own[1], asked[1]) if d is not None), default=None)
    return lo, hi


def _within(d: Date, lo: Date | None, hi: Date | None) -> bool:
    return (lo is None or d >= lo) and (hi is None or d <= hi)


@dataclass(frozen=True)
class SinglePoint:
    """One decision. The smoke-test and single-shot-evaluation shape."""

    on: Date
    kind: ClassVar[str] = "single_point"

    def dates(self, start: Date | None = None, end: Date | None = None) -> list[Date]:
        return [self.on] if _within(self.on, start, end) else []


@dataclass(frozen=True)
class CustomDates:
    """An explicit grid.

    Unlike the old implementation this honours the window rather than ignoring
    it, so `--start/--end` can narrow a package's grid instead of being silently
    dropped. Dates are returned exactly as declared — a date that isn't a
    trading day is a preflight finding, not something to quietly move, because
    moving it would change the entry price and therefore the result.
    """

    dates_: tuple[Date, ...]
    start: Date | None = None
    end: Date | None = None
    kind: ClassVar[str] = "custom_dates"

    def dates(self, start: Date | None = None, end: Date | None = None) -> list[Date]:
        lo, hi = _window((self.start, self.end), (start, end))
        return sorted({d for d in self.dates_ if _within(d, lo, hi)})


@dataclass(frozen=True)
class Quarterly:
    """First trading day of each calendar quarter.

    Holiday-aware, which the old `pd.date_range(freq="BQS")` was not: BQS
    returns 2024-01-01, a Monday the NYSE was closed.
    """

    start: Date | None = None
    end: Date | None = None
    calendar: TradingCalendar = field(default_factory=TradingCalendar)
    kind: ClassVar[str] = "quarterly"

    def dates(self, start: Date | None = None, end: Date | None = None) -> list[Date]:
        lo, hi = _window((self.start, self.end), (start, end))
        if lo is None or hi is None:
            raise ValueError("quarterly schedule needs a bounded window (start and end)")
        out: list[Date] = []
        for year in range(lo.year, hi.year + 1):
            for month, day in QUARTER_STARTS:
                d = self.calendar.snap_forward(Date(year, month, day))
                if _within(d, lo, hi):
                    out.append(d)
        return sorted(out)
