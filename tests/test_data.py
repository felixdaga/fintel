from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from fintel.market import catalog
from fintel.market.data import coverage as cov
from fintel.market.data.base import DataError, DataSource
from fintel.market.data.massive import (
    FUNDAMENTALS,
    NEWS,
    MassivePrices,
    MassiveRecords,
    normalise_article,
    normalise_financial,
)
from fintel.market.data.store import PriceStore, RecordCache
from fintel.market.data.synthetic import SyntheticPrices
from fintel.market.factory import build_data_source, build_data_sources
from fintel.market.realized import PriceLookup
from fintel.market.settings import MarketConfig
from fintel.models.market import DataBinding
from fintel.pit import Cutoff
from tests import fixtures

CUT = Cutoff(date(2024, 6, 3))


# ── coverage span algebra ────────────────────────────────────────────────────


def test_coalesce_merges_overlapping_and_adjacent():
    spans = [(date(2024, 1, 1), date(2024, 1, 10)), (date(2024, 1, 11), date(2024, 1, 20))]
    assert cov.coalesce(spans) == [(date(2024, 1, 1), date(2024, 1, 20))]


def test_coalesce_drops_inverted_spans():
    assert cov.coalesce([(date(2024, 2, 1), date(2024, 1, 1))]) == []


def test_missing_finds_interior_gaps():
    coverage = [(date(2024, 1, 1), date(2024, 1, 10)), (date(2024, 2, 1), date(2024, 2, 10))]
    assert cov.missing(coverage, date(2024, 1, 1), date(2024, 2, 10)) == [
        (date(2024, 1, 11), date(2024, 1, 31))
    ]


def test_missing_finds_leading_and_trailing_gaps():
    coverage = [(date(2024, 1, 10), date(2024, 1, 20))]
    assert cov.missing(coverage, date(2024, 1, 1), date(2024, 1, 31)) == [
        (date(2024, 1, 1), date(2024, 1, 9)),
        (date(2024, 1, 21), date(2024, 1, 31)),
    ]


def test_backward_jump_is_a_miss_not_a_hit():
    """The bug a single `fetched_through` bound causes: a cache warmed through
    2026 claiming to cover a 2024 request it holds nothing for."""
    coverage = [(date(2026, 1, 1), date(2026, 6, 1))]
    assert not cov.covers(coverage, date(2024, 1, 1), date(2024, 3, 1))
    assert cov.missing(coverage, date(2024, 1, 1), date(2024, 3, 1)) == [
        (date(2024, 1, 1), date(2024, 3, 1))
    ]


def test_covers_when_fully_contained():
    assert cov.covers([(date(2024, 1, 1), date(2024, 12, 31))], date(2024, 3, 1), date(2024, 4, 1))


def test_span_json_round_trip():
    spans = [(date(2024, 1, 1), date(2024, 1, 10))]
    assert cov.from_json(cov.to_json(spans)) == spans
    assert cov.from_json([["bad", "worse"], None]) == []


# ── stores ───────────────────────────────────────────────────────────────────


def _bars(start: date, n: int, price: float = 100.0) -> pd.DataFrame:
    days = pd.bdate_range(start=start, periods=n)
    return pd.DataFrame(
        {
            "date": [d.date() for d in days],
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 1000,
        }
    )


def test_price_store_round_trip_and_dedupe(tmp_path):
    store = PriceStore(root=tmp_path)
    df = _bars(date(2024, 1, 2), 5)
    store.write("AAPL", pd.concat([df, df]), [(date(2024, 1, 2), date(2024, 1, 8))])
    got = store.read("AAPL")
    assert len(got) == 5
    assert got["date"].is_monotonic_increasing
    assert store.symbols() == ["AAPL"]


def test_price_store_coverage_prefers_the_sidecar(tmp_path):
    store = PriceStore(root=tmp_path)
    store.write("AAPL", _bars(date(2024, 1, 2), 5), [(date(2023, 1, 1), date(2024, 1, 8))])
    # Sidecar records the whole fetched span, wider than the bars present.
    assert store.coverage("AAPL") == [(date(2023, 1, 1), date(2024, 1, 8))]


def test_price_store_infers_coverage_without_a_sidecar(tmp_path):
    """Caches written by the previous implementation have no sidecar."""
    store = PriceStore(root=tmp_path)
    store.write("AAPL", _bars(date(2024, 1, 2), 5), [(date(2024, 1, 2), date(2024, 1, 8))])
    store._sidecar("AAPL").unlink()
    assert store.coverage("AAPL") == [(date(2024, 1, 2), date(2024, 1, 8))]


def test_price_store_missing_symbol_is_none(tmp_path):
    assert PriceStore(root=tmp_path).read("NOPE") is None
    assert PriceStore(root=tmp_path).coverage("NOPE") == []


def test_record_cache_round_trip(tmp_path):
    cache = RecordCache(root=tmp_path, kind="news")
    spans = [(date(2024, 1, 1), date(2024, 3, 1))]
    cache.write("AAPL", spans, [{"id": "a", "published_at": "2024-02-01"}])
    coverage, records = cache.read("AAPL")
    assert coverage == spans
    assert records[0]["id"] == "a"


def test_record_cache_treats_corruption_as_empty(tmp_path):
    cache = RecordCache(root=tmp_path, kind="news")
    path = cache.path("AAPL")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert cache.read("AAPL") == ([], [])


# ── price source ─────────────────────────────────────────────────────────────


def test_prices_are_clamped_and_windowed(tmp_path):
    store = PriceStore(root=tmp_path)
    store.write("AAPL", _bars(date(2024, 1, 2), 200), [(date(2020, 1, 1), date(2024, 10, 1))])
    src = MassivePrices(store=store, client=None, history_start=date(2020, 1, 1))
    out = src.fetch({"symbol": "AAPL", "lookback_days": 30}, CUT)
    assert out["date"].max() < CUT.decision_date
    assert out["date"].min() >= CUT.decision_date - timedelta(days=30)


def test_prices_field_projection(tmp_path):
    store = PriceStore(root=tmp_path)
    store.write("AAPL", _bars(date(2024, 1, 2), 100), [(date(2020, 1, 1), date(2024, 10, 1))])
    src = MassivePrices(store=store, client=None, history_start=date(2020, 1, 1))
    out = src.fetch({"symbol": "AAPL", "fields": ["close"]}, CUT)
    assert list(out.columns) == ["date", "close"]


def test_prices_offline_with_no_cache_raises(tmp_path):
    """Coverage is what makes 'never fetched' distinguishable from 'no data'."""
    src = MassivePrices(store=PriceStore(root=tmp_path), client=None)
    with pytest.raises(DataError, match="no cached prices"):
        src.fetch({"symbol": "AAPL"}, CUT)


def test_prices_offline_with_partial_cache_serves_what_exists(tmp_path, caplog):
    store = PriceStore(root=tmp_path)
    store.write("AAPL", _bars(date(2024, 1, 2), 20), [(date(2024, 1, 2), date(2024, 1, 30))])
    src = MassivePrices(store=store, client=None, history_start=date(2010, 1, 1))
    out = src.fetch({"symbol": "AAPL", "lookback_days": 3650}, CUT)
    assert not out.empty
    assert "short of" in caplog.text


def test_source_missing_symbol_key_is_explicit(tmp_path):
    src = MassivePrices(store=PriceStore(root=tmp_path), client=None)
    with pytest.raises(DataError, match="missing required key 'symbol'"):
        src.fetch({}, CUT)


def test_sources_satisfy_the_protocol(tmp_path):
    assert isinstance(MassivePrices(store=PriceStore(root=tmp_path)), DataSource)
    assert isinstance(SyntheticPrices(), DataSource)
    assert isinstance(
        MassiveRecords(spec=NEWS, cache=RecordCache(root=tmp_path, kind="news")), DataSource
    )


# ── record sources ───────────────────────────────────────────────────────────


def test_records_are_clamped_on_availability(tmp_path):
    cache = RecordCache(root=tmp_path, kind="news")
    cache.write(
        "AAPL",
        [(date(2024, 1, 1), date(2024, 12, 31))],
        [
            {"id": "past", "published_at": "2024-05-01"},
            {"id": "same_day", "published_at": "2024-06-03"},
            {"id": "future", "published_at": "2024-07-01"},
        ],
    )
    src = MassiveRecords(spec=NEWS, cache=cache, client=None, name="massive_news")
    got = src.fetch({"symbol": "AAPL", "lookback_days": 90}, CUT)
    assert [r["id"] for r in got] == ["past"]


def test_fundamentals_reject_a_period_that_has_not_ended(tmp_path):
    """A filing stamped before the decision date whose period ends after it
    cannot be knowable, whatever the vendor says."""
    cache = RecordCache(root=tmp_path, kind="fundamentals")
    cache.write(
        "AAPL",
        [(date(2023, 1, 1), date(2024, 12, 31))],
        [
            {"filing_date": "2024-05-01", "period_end": "2024-03-31", "eps_diluted": 1.0},
            {"filing_date": "2024-05-02", "period_end": "2024-09-30", "eps_diluted": 9.9},
        ],
    )
    src = MassiveRecords(spec=FUNDAMENTALS, cache=cache, client=None, name="massive_fundamentals")
    got = src.fetch({"symbol": "AAPL"}, CUT)
    assert [r["period_end"] for r in got] == ["2024-03-31"]


def test_records_respect_the_lookback_floor(tmp_path):
    cache = RecordCache(root=tmp_path, kind="news")
    cache.write(
        "AAPL",
        [(date(2020, 1, 1), date(2024, 12, 31))],
        [
            {"id": "old", "published_at": "2021-01-01"},
            {"id": "recent", "published_at": "2024-05-20"},
        ],
    )
    src = MassiveRecords(spec=NEWS, cache=cache, client=None)
    assert [r["id"] for r in src.fetch({"symbol": "AAPL", "lookback_days": 30}, CUT)] == ["recent"]


def test_records_offline_with_nothing_cached_raises(tmp_path):
    src = MassiveRecords(spec=NEWS, cache=RecordCache(root=tmp_path, kind="news"), client=None)
    with pytest.raises(DataError, match="nothing cached"):
        src.fetch({"symbol": "AAPL"}, CUT)


# ── vendor normalisation ─────────────────────────────────────────────────────


def test_normalise_financial_flattens_statements():
    got = normalise_financial(
        {
            "filing_date": "2024-05-01",
            "end_date": "2024-03-31",
            "timeframe": "quarterly",
            "financials": {
                "income_statement": {
                    "revenues": {"value": 1000.0},
                    "diluted_earnings_per_share": {"value": 2.5},
                },
                "balance_sheet": {"assets": {"value": 5000.0}},
                "cash_flow_statement": {
                    "net_cash_flow_from_operating_activities": {"value": 800.0},
                    "payments_for_property_plant_and_equipment": {"value": -200.0},
                },
            },
        }
    )
    assert got["revenue"] == 1000.0
    assert got["eps_diluted"] == 2.5
    assert got["total_assets"] == 5000.0
    assert got["free_cash_flow"] == 600.0  # capex is negative as filed


def test_free_cash_flow_stays_none_without_capex():
    """Approximating FCF from total investing cash flow is nonsense in any
    quarter with material M&A, so absence is reported as absence."""
    got = normalise_financial(
        {"financials": {"cash_flow_statement": {"net_cash_flow_from_operating_activities": 800.0}}}
    )
    assert got["operating_cash_flow"] == 800.0
    assert got["free_cash_flow"] is None
    assert got["capex"] is None


def test_normalise_article_takes_the_date_from_the_timestamp():
    got = normalise_article(
        {
            "id": "x",
            "title": "T",
            "published_utc": "2024-05-01T13:45:00Z",
            "publisher": {"name": "P"},
        }
    )
    assert got["published_at"] == "2024-05-01"
    assert got["publisher"] == "P"


# ── synthetic ────────────────────────────────────────────────────────────────


def test_synthetic_is_deterministic_and_symbol_specific():
    a = SyntheticPrices().fetch({"symbol": "AAA"}, CUT)
    b = SyntheticPrices().fetch({"symbol": "AAA"}, CUT)
    c = SyntheticPrices().fetch({"symbol": "BBB"}, CUT)
    assert a["close"].tolist() == b["close"].tolist()
    assert a["close"].tolist() != c["close"].tolist()


def test_synthetic_lands_only_on_trading_days_and_respects_the_cutoff():
    out = SyntheticPrices().fetch({"symbol": "AAA"}, Cutoff(date(2024, 1, 3)))
    assert out["date"].max() == date(2024, 1, 2)  # Jan 1 is a holiday
    assert all(d.weekday() < 5 for d in out["date"])


# ── realized prices (the unclamped scoring path) ─────────────────────────────


def test_price_lookup_reads_the_open(tmp_path):
    store = PriceStore(root=tmp_path)
    df = _bars(date(2024, 1, 2), 10)
    df.loc[0, "open"] = 111.0
    store.write("AAPL", df, [(date(2024, 1, 2), date(2024, 1, 15))])
    assert PriceLookup(store=store).price_at("AAPL", date(2024, 1, 2)) == 111.0


def test_price_lookup_sees_the_future_on_purpose(tmp_path):
    """The measurement path must read past the decision date."""
    store = PriceStore(root=tmp_path)
    store.write("AAPL", _bars(date(2024, 1, 2), 200), [(date(2024, 1, 2), date(2024, 10, 1))])
    lookup = PriceLookup(store=store)
    assert lookup.price_at("AAPL", date(2024, 8, 1)) is not None
    assert lookup.forward_return("AAPL", date(2024, 1, 2), date(2024, 8, 1)) is not None


def test_non_trading_date_falls_back_to_the_prior_session(tmp_path):
    store = PriceStore(root=tmp_path)
    store.write("AAPL", _bars(date(2023, 12, 26), 10), [(date(2023, 12, 26), date(2024, 1, 10))])
    lookup = PriceLookup(store=store)
    # 2024-01-01 is a holiday; the last bar on or before it is 2023-12-29.
    assert lookup.price_at("AAPL", date(2024, 1, 1)) == lookup.price_at("AAPL", date(2023, 12, 29))


def test_forward_return_arithmetic(tmp_path):
    store = PriceStore(root=tmp_path)
    df = _bars(date(2024, 1, 2), 5)
    df.loc[0, "open"] = 100.0
    df.loc[4, "open"] = 110.0
    store.write("AAPL", df, [(date(2024, 1, 2), date(2024, 1, 8))])
    got = PriceLookup(store=store).forward_return("AAPL", date(2024, 1, 2), df["date"].iloc[4])
    assert got == pytest.approx(0.10)


def test_missing_prices_are_absent_not_zero(tmp_path):
    lookup = PriceLookup(store=PriceStore(root=tmp_path))
    assert lookup.price_at("NOPE", date(2024, 1, 2)) is None
    assert lookup.forward_return("NOPE", date(2024, 1, 2), date(2024, 2, 1)) is None
    assert lookup.forward_returns(["NOPE"], date(2024, 1, 2), date(2024, 2, 1)) == {}


def test_latest_bar_date_is_intersection_of_maxima(tmp_path):
    store = PriceStore(root=tmp_path)
    store.write("AAA", _bars(date(2024, 1, 2), 5), [(date(2024, 1, 2), date(2024, 1, 8))])
    store.write("BBB", _bars(date(2024, 1, 2), 10), [(date(2024, 1, 2), date(2024, 1, 15))])
    lookup = PriceLookup(store=store)
    # AAA ends earlier → intersection pinned to AAA's last bar.
    assert lookup.latest_bar_date(["AAA", "BBB"]) == store.read("AAA")["date"].iloc[-1]


def test_latest_bar_date_skips_empty_and_honors_min_coverage(tmp_path):
    store = PriceStore(root=tmp_path)
    store.write("AAA", _bars(date(2024, 1, 2), 10), [(date(2024, 1, 2), date(2024, 1, 15))])
    store.write("BBB", _bars(date(2024, 1, 2), 5), [(date(2024, 1, 2), date(2024, 1, 8))])
    lookup = PriceLookup(store=store)
    assert lookup.latest_bar_date(["AAA", "MISSING"]) == store.read("AAA")["date"].iloc[-1]
    # With 50% coverage, the fresher AAA date is allowed.
    assert lookup.latest_bar_date(["AAA", "BBB"], min_coverage=0.5) == store.read("AAA")[
        "date"
    ].iloc[-1]


# ── the catalog: pick from, or add to ────────────────────────────────────────


def test_builtin_sources_are_browsable_by_kind():
    catalog.register_builtins()
    assert {s.name for s in catalog.sources(kind="prices")} >= {
        "massive_prices",
        "synthetic_prices",
    }
    assert "fundamentals" in catalog.kinds()
    assert "massive" in catalog.providers()


def test_every_source_declares_fields_and_a_resolvable_target():
    from fintel.utils.import_path import resolve

    catalog.register_builtins()
    for info in catalog.sources():
        assert info.fields, f"{info.name} declares no fields"
        assert resolve(info.target) is not None


def test_a_source_must_declare_fields_to_be_registered():
    from fintel.market.catalog import SourceInfo

    with pytest.raises(ValueError, match="must declare its fields"):
        catalog.register_source(SourceInfo(name="opaque", kind="x", provider="p", target="a:b"))


def test_duplicate_registration_is_rejected_unless_replacing():
    catalog.register_builtins()
    info = catalog.source("massive_prices")
    with pytest.raises(ValueError, match="already registered"):
        catalog.register_source(info)
    assert catalog.register_source(info, replace=True) is info


def test_universes_are_browsable_and_flag_survivorship():
    catalog.register_builtins()
    pit = {u.name for u in catalog.universes(point_in_time=True)}
    snapshots = {u.name for u in catalog.universes(point_in_time=False)}
    assert "dow30" in pit
    assert "djia_2024_11_08" in snapshots
    assert catalog.universe("djia_2024_11_08").n_symbols == 30
    assert catalog.universe("djia_2024_11_08").as_of == "2024-11-08"


def test_field_roster_is_queryable():
    catalog.register_builtins()
    names = [f.name for f in catalog.fields_for("massive_prices")]
    assert names == ["date", "open", "high", "low", "close", "volume"]
    assert catalog.source("massive_fundamentals").field_names[:2] == ("form", "filing_date")


def test_unknown_names_list_what_exists():
    with pytest.raises(KeyError, match="registered"):
        catalog.source("nope")
    with pytest.raises(KeyError, match="registered"):
        catalog.universe("nope")


# ── swapping and composing sources ───────────────────────────────────────────


def test_a_third_party_source_serves_an_existing_kind(tmp_path):
    """Standing in for yfinance: same kind, different vendor, no platform edit."""
    fixtures.register_all()
    assert {s.name for s in catalog.sources(kind="prices")} >= {"massive_prices", "flat_prices"}
    src = build_data_source(
        DataBinding(kind="prices", source="flat_prices"), config=MarketConfig(cache_root=tmp_path)
    )
    out = src.fetch({"symbol": "AAPL"}, CUT)
    assert out["close"].iloc[-1] == 50.0
    assert out["date"].max() < CUT.decision_date


def test_unregistered_import_path_still_works(tmp_path):
    binding = DataBinding(kind="prices", source="tests.fixtures:flat_prices", price=7.0)
    src = build_data_source(binding, config=MarketConfig(cache_root=tmp_path))
    assert src.fetch({"symbol": "AAPL"}, CUT)["close"].iloc[-1] == 7.0


def test_binding_a_source_to_the_wrong_kind_fails_fast(tmp_path):
    fixtures.register_all()
    with pytest.raises(ValueError, match="serves kind 'prices'"):
        build_data_source(
            DataBinding(kind="news", source="flat_prices"), config=MarketConfig(cache_root=tmp_path)
        )


def test_unknown_param_is_rejected_against_the_catalog(tmp_path):
    catalog.register_builtins()
    with pytest.raises(ValueError, match="does not accept"):
        build_data_source(
            DataBinding(kind="prices", source="synthetic_prices", lookbcak_days=30),
            config=MarketConfig(cache_root=tmp_path),
        )


def test_computed_source_gets_its_upstream_kinds_injected(tmp_path):
    fixtures.register_all()
    bindings = [
        DataBinding(kind="prices", source="flat_prices", price=100.0),
        DataBinding(kind="fundamentals", source="stub_fundamentals", eps=5.0),
        DataBinding(kind="ratios", source="simple_ratios"),
    ]
    built = build_data_sources(bindings, config=MarketConfig(cache_root=tmp_path))
    assert set(built) == {"prices", "fundamentals", "ratios"}
    assert built["ratios"].fetch({"symbol": "AAPL"}, CUT)["pe"] == pytest.approx(20.0)


def test_computed_source_swaps_provider_without_touching_the_computation(tmp_path):
    fixtures.register_all()
    bindings = [
        DataBinding(kind="prices", source="flat_prices", price=200.0),
        DataBinding(kind="fundamentals", source="stub_fundamentals", eps=5.0),
        DataBinding(kind="ratios", source="simple_ratios"),
    ]
    built = build_data_sources(bindings, config=MarketConfig(cache_root=tmp_path))
    assert built["ratios"].fetch({"symbol": "AAPL"}, CUT)["pe"] == pytest.approx(40.0)


def test_computed_source_names_the_missing_upstream_binding(tmp_path):
    fixtures.register_all()
    bindings = [
        DataBinding(kind="prices", source="flat_prices"),
        DataBinding(kind="ratios", source="simple_ratios"),
    ]
    with pytest.raises(ValueError, match=r"add a \[\[data\]\] block for \['fundamentals'\]"):
        build_data_sources(bindings, config=MarketConfig(cache_root=tmp_path))


def test_computed_source_reports_a_missing_input_as_none(tmp_path):
    fixtures.register_all()
    bindings = [
        DataBinding(kind="prices", source="flat_prices"),
        DataBinding(kind="fundamentals", source="stub_fundamentals", eps=None),
        DataBinding(kind="ratios", source="simple_ratios"),
    ]
    built = build_data_sources(bindings, config=MarketConfig(cache_root=tmp_path))
    assert built["ratios"].fetch({"symbol": "AAPL"}, CUT)["pe"] is None
