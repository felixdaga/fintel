"""Daily site NAV is independent of decision cadence."""

from datetime import date as Date

import pytest

from fintel.deploy.site_nav import (
    daily_grid,
    last_on_or_before,
    observed_price_days,
    walk_benchmark,
    walk_nav,
    weighted_return,
)


class FakePrices:
    def __init__(self, px: dict[str, dict[Date, float]]):
        self.px = px

    def price_at(self, symbol: str, on: Date) -> float | None:
        return self.px.get(symbol, {}).get(on)

    def forward_return(self, symbol: str, start: Date, end: Date) -> float | None:
        p0 = self.price_at(symbol, start)
        p1 = self.price_at(symbol, end)
        if p0 is None or p1 is None or p0 <= 0:
            return None
        return p1 / p0 - 1.0


def test_daily_grid_skips_weekend():
    days = daily_grid(Date(2026, 4, 24), Date(2026, 4, 27))
    assert days == [Date(2026, 4, 24), Date(2026, 4, 27)]  # Fri, Mon


def test_last_on_or_before_holds_between_decisions():
    dates = [Date(2026, 4, 24), Date(2026, 5, 8)]
    assert last_on_or_before(dates, Date(2026, 4, 24)) == Date(2026, 4, 24)
    assert last_on_or_before(dates, Date(2026, 5, 1)) == Date(2026, 4, 24)
    assert last_on_or_before(dates, Date(2026, 5, 8)) == Date(2026, 5, 8)
    assert last_on_or_before(dates, Date(2026, 4, 1)) is None


def test_walk_nav_marks_every_session_with_last_book():
    # Fri / Mon / Tue. Decision only on Friday; 10% then 10% on A.
    daily = [Date(2026, 4, 24), Date(2026, 4, 27), Date(2026, 4, 28)]
    prices = FakePrices(
        {"A": {daily[0]: 100.0, daily[1]: 110.0, daily[2]: 121.0}}
    )
    books = {daily[0]: {"A": 1.0}}

    def book_on(d: Date) -> dict[str, float]:
        dec = last_on_or_before(list(books), d)
        return books[dec] if dec else {}

    gross, net = walk_nav(
        daily=daily,
        decision_dates=[daily[0]],
        book_on=book_on,
        prices=prices,
        cost_bps=5.0,
    )
    assert [p["date"] for p in gross] == [d.isoformat() for d in daily]
    assert gross[-1]["nav"] == 1.21
    assert net[-1]["nav"] == 1.21  # first rebalance free, then a hold


def test_turnover_cost_only_on_rebalance_day():
    daily = [Date(2026, 4, 24), Date(2026, 4, 27), Date(2026, 4, 28)]
    prices = FakePrices(
        {
            "A": {d: 100.0 for d in daily},
            "B": {d: 100.0 for d in daily},
        }
    )
    books = {daily[0]: {"A": 1.0}, daily[1]: {"B": 1.0}}

    def book_on(d: Date) -> dict[str, float]:
        dec = last_on_or_before([daily[0], daily[1]], d)
        return books[dec] if dec else {}

    gross, net = walk_nav(
        daily=daily,
        decision_dates=[daily[0], daily[1]],
        book_on=book_on,
        prices=prices,
        cost_bps=5.0,
    )
    assert gross[-1]["nav"] == 1.0
    # Mon→Tue starts on a decision date, two-way turnover 2.0 × 5bps
    assert net[-1]["nav"] == round(1.0 * (1.0 - 2.0 * 5.0 / 10000.0), 6)


def test_benchmark_is_daily_price_weighted():
    daily = [Date(2026, 4, 24), Date(2026, 4, 27)]
    prices = FakePrices(
        {
            "A": {daily[0]: 100.0, daily[1]: 110.0},
            "B": {daily[0]: 200.0, daily[1]: 200.0},
        }
    )
    points = walk_benchmark(daily, ["A", "B"], prices)
    # weights 1/3 A, 2/3 B → r = (1/3)*0.10 + (2/3)*0 = 1/30
    assert points[-1]["nav"] == round(1.0 + 1.0 / 30.0, 6)


def test_weighted_return_renormalizes_missing_names():
    prices = FakePrices({"A": {Date(2026, 4, 24): 100.0, Date(2026, 4, 27): 110.0}})
    r = weighted_return(
        {"A": 0.5, "MISSING": 0.5}, Date(2026, 4, 24), Date(2026, 4, 27), prices
    )
    assert r == pytest.approx(0.10)


class _FakeStore:
    def __init__(self, frames: dict[str, object]):
        self.frames = frames

    def read(self, symbol: str):
        return self.frames.get(symbol)


def test_observed_price_days_skips_cache_gaps():
    import pandas as pd

    class P:
        store = _FakeStore(
            {
                "A": pd.DataFrame(
                    {"date": [Date(2026, 4, 24), Date(2026, 4, 28)]}
                ),
                "B": pd.DataFrame(
                    {"date": [Date(2026, 4, 24), Date(2026, 4, 28)]}
                ),
            }
        )

    days = observed_price_days(P(), ["A", "B"], Date(2026, 4, 24), Date(2026, 4, 28))
    assert Date(2026, 4, 27) not in days  # Monday missing from cache
    assert days == [Date(2026, 4, 24), Date(2026, 4, 28)]


def test_observed_price_days_without_store_uses_calendar():
    days = observed_price_days(FakePrices({}), ["A"], Date(2026, 4, 24), Date(2026, 4, 27))
    assert days == [Date(2026, 4, 24), Date(2026, 4, 27)]
