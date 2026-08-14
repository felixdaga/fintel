from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from fintel.market.calendar import TradingCalendar, easter, market_holidays
from fintel.market.constituents import HistoricalUniverse, normalise
from fintel.market.factory import as_date, build_schedule, build_universe
from fintel.market.schedule import BiweeklyFridays, CustomDates, Quarterly, Schedule, SinglePoint
from fintel.market.settings import MarketConfig
from fintel.market.universe import STATIC_PRESETS, StaticUniverse, Universe, static_preset
from fintel.models.market import ScheduleRef, UniverseRef

CAL = TradingCalendar()

# The reference package's grid. Four of these are not trading days, which is the
# whole reason this layer needs a real calendar.
REFERENCE_GRID = [
    date(2022, 7, 1), date(2022, 10, 3), date(2023, 1, 3), date(2023, 4, 3),
    date(2023, 7, 3), date(2023, 10, 2), date(2024, 1, 2), date(2024, 4, 1),
    date(2024, 7, 1), date(2024, 10, 1), date(2025, 1, 2), date(2025, 4, 1),
    date(2025, 7, 1), date(2025, 10, 1), date(2026, 1, 2), date(2026, 4, 1),
]  # fmt: skip


# ── calendar ─────────────────────────────────────────────────────────────────


def test_easter_matches_known_years():
    assert easter(2022) == date(2022, 4, 17)
    assert easter(2023) == date(2023, 4, 9)
    assert easter(2024) == date(2024, 3, 31)
    assert easter(2025) == date(2025, 4, 20)


def test_good_friday_is_closed():
    assert not CAL.is_trading_day(date(2024, 3, 29))
    assert not CAL.is_trading_day(date(2025, 4, 18))


def test_weekends_are_closed():
    assert not CAL.is_trading_day(date(2023, 1, 1))  # Sunday
    assert not CAL.is_trading_day(date(2022, 7, 2))  # Saturday


def test_fixed_holidays_and_observance():
    assert not CAL.is_trading_day(date(2024, 1, 1))  # New Year, Monday
    assert not CAL.is_trading_day(date(2024, 7, 4))
    # 2021-07-04 was a Sunday, observed Monday the 5th.
    assert not CAL.is_trading_day(date(2021, 7, 5))
    # 2020-07-04 was a Saturday, observed Friday the 3rd.
    assert not CAL.is_trading_day(date(2020, 7, 3))


def test_new_year_on_saturday_does_not_close_dec_31():
    """The NYSE exception: a Saturday Jan 1 is not observed on Dec 31."""
    assert date(2022, 1, 1).weekday() == 5
    assert CAL.is_trading_day(date(2021, 12, 31))


def test_floating_holidays():
    assert not CAL.is_trading_day(date(2024, 1, 15))  # MLK, 3rd Mon Jan
    assert not CAL.is_trading_day(date(2024, 2, 19))  # Washington, 3rd Mon Feb
    assert not CAL.is_trading_day(date(2024, 5, 27))  # Memorial, last Mon May
    assert not CAL.is_trading_day(date(2024, 9, 2))  # Labor, 1st Mon Sep
    assert not CAL.is_trading_day(date(2024, 11, 28))  # Thanksgiving, 4th Thu Nov


def test_juneteenth_only_from_2022():
    assert CAL.is_trading_day(date(2021, 6, 18))
    assert not CAL.is_trading_day(date(2024, 6, 19))


def test_mlk_only_from_1998():
    assert 1997 not in {d.year for d in market_holidays(1997) if d.month == 1 and d.day > 2}


def test_special_closures():
    assert not CAL.is_trading_day(date(2012, 10, 30))  # Hurricane Sandy
    assert not CAL.is_trading_day(date(2018, 12, 5))  # Bush funeral
    assert not CAL.is_trading_day(date(2025, 1, 9))  # Carter funeral


def test_holidays_stay_in_their_year():
    for year in (2020, 2021, 2022, 2023, 2024, 2025):
        assert all(d.year == year for d in market_holidays(year))


def test_next_and_prev_skip_closures():
    # 2024-01-01 is a Monday holiday, so the year opens on the 2nd.
    assert CAL.next(date(2023, 12, 29)) == date(2024, 1, 2)
    assert CAL.prev(date(2024, 1, 2)) == date(2023, 12, 29)
    assert CAL.next(date(2024, 7, 3)) == date(2024, 7, 5)  # over July 4


def test_next_zero_is_identity_even_on_a_holiday():
    assert CAL.next(date(2024, 1, 1), 0) == date(2024, 1, 1)


def test_snap_forward_and_back():
    assert CAL.snap_forward(date(2024, 1, 1)) == date(2024, 1, 2)
    assert CAL.snap_back(date(2024, 1, 1)) == date(2023, 12, 29)
    assert CAL.snap_forward(date(2024, 1, 2)) == date(2024, 1, 2)


def test_days_counts_a_known_week():
    # Week of July 4 2024 (Thursday): Mon, Tue, Wed, Fri.
    assert CAL.days(date(2024, 7, 1), date(2024, 7, 7)) == [
        date(2024, 7, 1),
        date(2024, 7, 2),
        date(2024, 7, 3),
        date(2024, 7, 5),
    ]


def test_non_trading_flags_the_old_reference_grid():
    """The 16 dates the reference package actually shipped. Half have no market
    session, absorbed silently by a price fallback to the prior open. A
    weekday-only calendar would have missed the three New Year holidays."""
    declared = [
        date(2022, 7, 1), date(2022, 10, 1), date(2023, 1, 1), date(2023, 4, 1),
        date(2023, 7, 1), date(2023, 10, 1), date(2024, 1, 1), date(2024, 4, 1),
        date(2024, 7, 1), date(2024, 10, 1), date(2025, 1, 1), date(2025, 4, 1),
        date(2025, 7, 1), date(2025, 10, 1), date(2026, 1, 1), date(2026, 4, 1),
    ]  # fmt: skip
    assert CAL.non_trading(declared) == [
        date(2022, 10, 1),  # Saturday
        date(2023, 1, 1),  # Sunday
        date(2023, 4, 1),  # Saturday
        date(2023, 7, 1),  # Saturday
        date(2023, 10, 1),  # Sunday
        date(2024, 1, 1),  # New Year
        date(2025, 1, 1),  # New Year
        date(2026, 1, 1),  # New Year
    ]


def test_reference_grid_is_all_trading_days():
    assert CAL.non_trading(REFERENCE_GRID) == []


# ── schedules ────────────────────────────────────────────────────────────────


def test_schedules_satisfy_the_protocol():
    assert isinstance(SinglePoint(on=date(2024, 1, 2)), Schedule)
    assert isinstance(CustomDates(dates_=(date(2024, 1, 2),)), Schedule)
    assert isinstance(Quarterly(), Schedule)
    assert isinstance(BiweeklyFridays(start=date(2025, 6, 6)), Schedule)


def test_single_point():
    s = SinglePoint(on=date(2024, 3, 1))
    assert s.dates() == [date(2024, 3, 1)]
    assert s.dates(date(2024, 4, 1), date(2024, 6, 1)) == []


def test_custom_dates_sorts_and_dedupes():
    s = CustomDates(dates_=(date(2024, 3, 1), date(2024, 1, 2), date(2024, 1, 2)))
    assert s.dates() == [date(2024, 1, 2), date(2024, 3, 1)]


def test_custom_dates_honours_the_window():
    """The old implementation ignored start/end, so --start silently did nothing."""
    s = CustomDates(dates_=(date(2024, 1, 2), date(2024, 6, 3), date(2025, 1, 2)))
    assert s.dates(start=date(2024, 6, 1)) == [date(2024, 6, 3), date(2025, 1, 2)]
    assert s.dates(end=date(2024, 6, 1)) == [date(2024, 1, 2)]


def test_custom_dates_intersects_own_window_with_the_callers():
    s = CustomDates(
        dates_=(date(2024, 1, 2), date(2024, 6, 3), date(2025, 1, 2)),
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
    )
    assert s.dates(start=date(2024, 5, 1)) == [date(2024, 6, 3)]
    assert date(2025, 1, 2) not in s.dates()


def test_quarterly_is_holiday_aware():
    """pandas BQS returns 2024-01-01 — a Monday the exchange was closed."""
    got = Quarterly(start=date(2024, 1, 1), end=date(2024, 12, 31)).dates()
    assert got == [date(2024, 1, 2), date(2024, 4, 1), date(2024, 7, 1), date(2024, 10, 1)]
    assert pd.date_range("2024-01-01", "2024-12-31", freq="BQS")[0].date() == date(2024, 1, 1)


def test_quarterly_spans_years_and_needs_a_window():
    got = Quarterly(start=date(2022, 7, 1), end=date(2023, 6, 30)).dates()
    assert got == [date(2022, 7, 1), date(2022, 10, 3), date(2023, 1, 3), date(2023, 4, 3)]
    with pytest.raises(ValueError, match="bounded window"):
        Quarterly().dates()


def test_biweekly_fridays_phase_from_anchor():
    s = BiweeklyFridays(
        start=date(2026, 8, 1),
        end=date(2026, 9, 15),
        anchor=date(2026, 8, 14),
    )
    assert s.dates() == [date(2026, 8, 14), date(2026, 8, 28), date(2026, 9, 11)]


def test_biweekly_fridays_snaps_holiday_friday():
    # 2025-07-04 is Friday Independence Day → snap to 2025-07-07
    s = BiweeklyFridays(
        start=date(2025, 6, 6),
        end=date(2025, 7, 20),
        anchor=date(2025, 6, 6),
    )
    assert date(2025, 6, 6) in s.dates()
    assert date(2025, 6, 20) in s.dates()
    assert date(2025, 7, 7) in s.dates()
    assert date(2025, 7, 4) not in s.dates()


def test_build_biweekly_fridays_from_ref():
    ref = ScheduleRef(
        kind="biweekly_fridays",
        start="2026-08-01",
        end="2026-09-15",
        anchor="2026-08-14",
    )
    assert build_schedule(ref).dates() == [
        date(2026, 8, 14),
        date(2026, 8, 28),
        date(2026, 9, 11),
    ]


# ── universes ────────────────────────────────────────────────────────────────


def test_static_universe_ignores_the_date():
    u = StaticUniverse(symbols=("AAPL", "MSFT"))
    assert isinstance(u, Universe)
    assert u.active_at(date(1999, 1, 1)) == u.active_at(date(2030, 1, 1)) == ["AAPL", "MSFT"]


@pytest.mark.parametrize("name", sorted(STATIC_PRESETS))
def test_static_presets_are_deduped_and_nonempty(name: str):
    u = static_preset(name)
    assert len(u.symbols) == len(set(u.symbols))
    assert u.symbols


def test_djia_snapshots_are_30_names_and_differ_by_the_known_swap():
    a = set(static_preset("djia_2024_11_08").symbols)
    b = set(static_preset("djia_2026_06_29").symbols)
    assert len(a) == len(b) == 30
    assert a - b == {"VZ"}
    assert b - a == {"GOOGL"}


def test_unknown_static_preset_lists_what_exists():
    with pytest.raises(ValueError, match="available"):
        static_preset("nope")


# ── point-in-time membership ─────────────────────────────────────────────────

HISTORY = pd.DataFrame(
    {
        "symbol": ["OLD", "NEW", "FOREVER"],
        "opt-in": ["2000-01-01", "2024-06-01", "1990-01-01"],
        "opt-out": ["2024-06-01", None, None],
    }
)


def _universe() -> HistoricalUniverse:
    return HistoricalUniverse(name="test", index_key="test", history=normalise(HISTORY))


def test_membership_edges():
    u = _universe()
    # Member on the opt-in date, not on the opt-out date.
    assert u.active_at(date(2024, 5, 31)) == ["FOREVER", "OLD"]
    assert u.active_at(date(2024, 6, 1)) == ["FOREVER", "NEW"]


def test_blank_opt_out_means_still_a_member():
    assert "FOREVER" in _universe().active_at(date(2030, 1, 1))


def test_names_are_absent_before_they_joined():
    assert "NEW" not in _universe().active_at(date(2020, 1, 1))


def test_union_over_covers_every_date_for_prefetch():
    u = _universe()
    assert u.union_over([date(2020, 1, 1), date(2025, 1, 1)]) == ["FOREVER", "NEW", "OLD"]


def test_verify_reports_rebalance_boundaries():
    report = _universe().verify([date(2024, 5, 31), date(2024, 6, 1)])
    assert report.per_date == {"2024-05-31": 2, "2024-06-01": 2}
    assert report.n_symbols == 3
    assert report.weights_available is False


def test_verify_raises_on_an_empty_date():
    """A stale cache or a pre-index date used to produce a silent empty run."""
    with pytest.raises(ValueError, match="ZERO members"):
        _universe().verify([date(1980, 1, 1)])


def test_dates_before_the_index_existed_report_the_earliest_opt_in():
    """A pre-index date is a zero-member date, so it raises rather than warning.
    The earliest opt-in has to be in that message or the cause is invisible."""
    with pytest.raises(ValueError, match="Earliest opt-in in the table is 1990-01-01"):
        _universe().verify([date(1980, 1, 1)])


def test_normalise_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        normalise(pd.DataFrame({"symbol": ["A"]}))


def test_normalise_rejects_inverted_dates():
    with pytest.raises(ValueError, match="opt-in after opt-out"):
        normalise(
            pd.DataFrame({"symbol": ["A"], "opt-in": ["2024-01-01"], "opt-out": ["2020-01-01"]})
        )


def test_normalise_rejects_unparseable_dates():
    with pytest.raises(ValueError, match="unparseable"):
        normalise(pd.DataFrame({"symbol": ["A"], "opt-in": ["garbage"], "opt-out": [None]}))


# ── factory ──────────────────────────────────────────────────────────────────


def test_as_date_accepts_toml_shapes():
    assert as_date("2024-01-02") == date(2024, 1, 2)
    assert as_date(date(2024, 1, 2)) == date(2024, 1, 2)
    with pytest.raises(TypeError):
        as_date(20240102)


def test_build_schedule_from_a_manifest_ref():
    ref = ScheduleRef(kind="custom_dates", dates=["2024-01-02", "2024-04-01"], start="2024-01-01")
    assert build_schedule(ref).dates() == [date(2024, 1, 2), date(2024, 4, 1)]


def test_build_quarterly_and_single_point():
    q = build_schedule(ScheduleRef(kind="quarterly", start="2024-01-01", end="2024-06-30"))
    assert q.dates() == [date(2024, 1, 2), date(2024, 4, 1)]
    s = build_schedule(ScheduleRef(kind="single_point", on="2024-03-01"))
    assert s.dates() == [date(2024, 3, 1)]


def test_build_schedule_errors_are_actionable():
    with pytest.raises(ValueError, match="available"):
        build_schedule(ScheduleRef(kind="monthly"))
    with pytest.raises(ValueError, match="non-empty"):
        build_schedule(ScheduleRef(kind="custom_dates"))
    with pytest.raises(ValueError, match="needs `on`"):
        build_schedule(ScheduleRef(kind="single_point"))


def test_build_schedule_via_import_path():
    ref = ScheduleRef(kind="fintel.market.schedule:SinglePoint", on=date(2024, 3, 1))
    assert build_schedule(ref).dates() == [date(2024, 3, 1)]


def test_build_universe_from_symbols(tmp_path):
    cfg = MarketConfig(cache_root=tmp_path)
    u = build_universe(UniverseRef(symbols=["MSFT", "AAPL", "MSFT"]), config=cfg)
    assert u.active_at(date(2024, 1, 2)) == ["MSFT", "AAPL"]


def test_build_universe_from_static_preset(tmp_path):
    u = build_universe(UniverseRef(preset="sector35"), config=MarketConfig(cache_root=tmp_path))
    assert len(u.active_at(date(2024, 1, 2))) == 35


def test_build_universe_unknown_preset_lists_both_registries(tmp_path):
    with pytest.raises(ValueError, match="index presets"):
        build_universe(UniverseRef(preset="dow31"), config=MarketConfig(cache_root=tmp_path))


def test_index_preset_offline_without_cache_is_a_clear_error(tmp_path):
    cfg = MarketConfig(cache_root=tmp_path, offline=True)
    with pytest.raises(FileNotFoundError, match="offline"):
        build_universe(UniverseRef(preset="dow30"), config=cfg)


def test_custom_universe_source_without_config(tmp_path):
    ref = UniverseRef(source="tests.fixtures:tiny_universe")
    u = build_universe(ref, config=MarketConfig(cache_root=tmp_path))
    assert u.active_at(date(2024, 1, 2)) == ["XYZ"]


def test_custom_universe_source_receives_config_when_it_asks(tmp_path):
    ref = UniverseRef(source="tests.fixtures:cache_aware_universe", tag="t")
    u = build_universe(ref, config=MarketConfig(cache_root=tmp_path / "mycache"))
    assert u.active_at(date(2024, 1, 2)) == ["mycache", "t"]


# ── settings ─────────────────────────────────────────────────────────────────


def test_config_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FINTEL_CACHE", str(tmp_path))
    monkeypatch.setenv("FINTEL_OFFLINE", "1")
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    cfg = MarketConfig.from_env()
    assert cfg.cache_root == tmp_path
    assert cfg.offline is True
    assert cfg.massive_api_key is None


def test_explicit_cache_root_beats_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FINTEL_CACHE", "/should/not/be/used")
    assert MarketConfig.from_env(tmp_path).cache_root == tmp_path


def test_cache_root_accepts_a_string():
    assert MarketConfig(cache_root="/tmp/x").cache_root == Path("/tmp/x")


def test_dir_creates_subdirectories(tmp_path):
    d = MarketConfig(cache_root=tmp_path).dir("prices")
    assert d.is_dir() and d.name == "prices"


def test_require_key_explains_the_fix(tmp_path):
    cfg = MarketConfig(cache_root=tmp_path)
    with pytest.raises(RuntimeError, match="FINTEL_OFFLINE"):
        cfg.require_key("MASSIVE_API_KEY", None)
    assert cfg.require_key("MASSIVE_API_KEY", "abc") == "abc"
