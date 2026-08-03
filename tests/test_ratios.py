"""The trailing-window math and the ratios computed from it.

These pin the behaviours the original was careful about: refusing to mis-pair a
quarter, surviving stock splits, and reporting absence as absence.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from fintel.market.data.ratios import RATIO_FIELDS, ValuationRatios, compute_ratios
from fintel.market.data.ttm import (
    PRIOR_MATCH_TOLERANCE_DAYS,
    build_trailing_filing,
    build_trailing_filing_carryforward,
)


def q(period_end: str, filing_date: str, *, timeframe: str = "quarterly", **fields) -> dict:
    return {"period_end": period_end, "filing_date": filing_date, "timeframe": timeframe, **fields}


# ── trailing window assembly ─────────────────────────────────────────────────


def test_empty_or_undated_input_yields_nothing():
    assert build_trailing_filing(None) is None
    assert build_trailing_filing([]) is None
    # No filing_date: the vendor has been seen to emit these.
    assert build_trailing_filing([q("2024-03-31", "")]) is None
    assert build_trailing_filing([q("garbage", "2024-05-01")]) is None


def test_latest_annual_is_used_directly():
    got = build_trailing_filing([q("2023-12-31", "2024-02-01", timeframe="annual", revenue=400.0)])
    assert got["ttm_method"] == "annual_only"
    assert got["revenue"] == 400.0


def test_no_annual_anchor_refuses():
    """Quarters with no annual can't be anchored by this formula."""
    filings = [q("2023-06-30", "2023-08-01"), q("2023-09-30", "2023-11-01")]
    assert build_trailing_filing(filings) is None


def test_annual_plus_delta_is_the_analyst_formula():
    """Trailing = annual + new periods - matching prior-year periods."""
    filings = [
        q("2022-12-31", "2023-02-01", timeframe="annual", revenue=1000.0, net_income=100.0),
        q("2022-03-31", "2022-05-01", revenue=200.0, net_income=20.0),
        q("2022-06-30", "2022-08-01", revenue=210.0, net_income=21.0),
        q("2023-03-31", "2023-05-01", revenue=250.0, net_income=25.0),
        q("2023-06-30", "2023-08-01", revenue=260.0, net_income=26.0),
    ]
    got = build_trailing_filing(filings)
    assert got["ttm_method"] == "annual_plus_delta"
    # revenue:    1000 + (250 + 260) - (200 + 210)
    # net income:  100 + ( 25 +  26) - ( 20 +  21)
    assert got["revenue"] == pytest.approx(1100.0)
    assert got["net_income"] == pytest.approx(110.0)
    assert got["timeframe"] == "ttm"
    assert got["ttm_annual_period_end"] == "2022-12-31"
    assert sorted(got["ttm_newer_period_ends"]) == ["2023-03-31", "2023-06-30"]


def test_mid_year_fiscal_calendar_works_without_reading_labels():
    """MSFT-shaped: FY ends June. Matching is day-based, never on 'Q1'/'Q2'."""
    filings = [
        q("2023-06-30", "2023-07-27", timeframe="annual", revenue=2110.0),
        q("2022-09-30", "2022-10-25", revenue=501.0),
        q("2022-12-31", "2023-01-24", revenue=527.0),
        q("2023-09-30", "2023-10-24", revenue=565.0),
        q("2023-12-31", "2024-01-30", revenue=620.0),
    ]
    got = build_trailing_filing(filings)
    assert got["revenue"] == pytest.approx(2110.0 + (565.0 + 620.0) - (501.0 + 527.0))
    assert got["ttm_anchor_period_end"] == "2023-12-31"


def test_retail_calendar_drift_is_tolerated():
    """13-week calendars shift a few days year over year; +/-20d absorbs that."""
    filings = [
        q("2023-01-28", "2023-03-01", timeframe="annual", revenue=1000.0),
        q("2022-04-30", "2022-06-01", revenue=200.0),
        q("2023-04-29", "2023-06-01", revenue=240.0),  # 364 days later
    ]
    got = build_trailing_filing(filings)
    assert got["revenue"] == pytest.approx(1040.0)


def test_a_missing_prior_year_match_refuses_rather_than_mis_pairing():
    """Pairing a December quarter with a September one overlaps three months and
    silently double-counts. A plausible wrong number is worse than None."""
    filings = [
        q("2022-12-31", "2023-02-01", timeframe="annual", revenue=1000.0),
        q("2022-06-30", "2022-08-01", revenue=210.0),  # ~6 months off the target
        q("2023-03-31", "2023-05-01", revenue=250.0),
    ]
    assert build_trailing_filing(filings) is None


def test_match_tolerance_boundary():
    inside = q("2023-03-31", "2023-05-01", revenue=250.0)
    prior_ok = q("2022-04-15", "2022-06-01", revenue=200.0)  # 15 days off
    prior_bad = q("2022-05-10", "2022-07-01", revenue=200.0)  # 40 days off
    annual = q("2022-12-31", "2023-02-01", timeframe="annual", revenue=1000.0)
    assert PRIOR_MATCH_TOLERANCE_DAYS == 20
    assert build_trailing_filing([annual, prior_ok, inside]) is not None
    assert build_trailing_filing([annual, prior_bad, inside]) is None


def test_semi_annual_filers_are_not_paired_with_quarterlies():
    filings = [
        q("2022-12-31", "2023-02-01", timeframe="annual", revenue=1000.0),
        q("2022-06-30", "2022-08-01", timeframe="semiannual", revenue=480.0),
        q("2023-06-30", "2023-08-01", timeframe="semiannual", revenue=520.0),
    ]
    got = build_trailing_filing(filings)
    assert got["revenue"] == pytest.approx(1040.0)


def test_balance_sheet_items_come_from_the_latest_filing_not_summed():
    filings = [
        q("2022-12-31", "2023-02-01", timeframe="annual", revenue=1000.0, total_assets=5000.0),
        q("2022-03-31", "2022-05-01", revenue=200.0, total_assets=4000.0),
        q("2023-03-31", "2023-05-01", revenue=250.0, total_assets=5500.0),
    ]
    got = build_trailing_filing(filings)
    assert got["total_assets"] == 5500.0


def test_eps_is_derived_from_net_income_to_survive_splits():
    """Summing pre- and post-split EPS turned NVDA's trailing EPS negative."""
    filings = [
        q(
            "2022-12-31", "2023-02-01", timeframe="annual",
            net_income=1000.0, eps_diluted=10.0, shares_diluted=100.0,
        ),
        q("2022-03-31", "2022-05-01", net_income=200.0, eps_diluted=2.0, shares_diluted=100.0),
        # 10:1 split: shares jump, EPS collapses, net income is unaffected.
        q("2023-03-31", "2023-05-01", net_income=300.0, eps_diluted=0.3, shares_diluted=1000.0),
    ]
    got = build_trailing_filing(filings)
    assert got["net_income"] == pytest.approx(1100.0)
    assert got["shares_diluted"] == 1000.0
    assert got["eps_diluted"] == pytest.approx(1.1)  # not a sum of 10.0 + 0.3 - 2.0


def test_flows_are_none_when_any_component_is_missing():
    filings = [
        q("2022-12-31", "2023-02-01", timeframe="annual", revenue=1000.0),
        q("2022-03-31", "2022-05-01", revenue=200.0),
        q("2023-03-31", "2023-05-01", revenue=None),
    ]
    assert build_trailing_filing(filings)["revenue"] is None


def test_no_new_filings_since_the_annual_uses_the_annual():
    filings = [
        q("2022-12-31", "2023-02-01", timeframe="annual", revenue=1000.0),
        q("2022-09-30", "2022-11-01", revenue=250.0),
    ]
    got = build_trailing_filing(filings)
    assert got["ttm_method"] == "annual_only"
    assert got["revenue"] == 1000.0


# ── carryforward ─────────────────────────────────────────────────────────────


def test_carryforward_walks_back_to_the_last_clean_anchor():
    """A provider gap at the newest quarter shouldn't null the whole ratio."""
    filings = [
        q("2022-12-31", "2023-02-01", timeframe="annual", revenue=1000.0),
        q("2022-03-31", "2022-05-01", revenue=200.0),
        q("2023-03-31", "2023-05-01", revenue=250.0),
        # Newest quarter has no prior-year counterpart in the data.
        q("2023-09-30", "2023-11-01", revenue=280.0),
    ]
    assert build_trailing_filing(filings) is None
    got = build_trailing_filing_carryforward(filings)
    assert got is not None
    assert got["carried_forward"] is True
    assert got["carried_forward_anchor_period_end"] == "2023-03-31"
    assert got["carried_forward_latest_period_end"] == "2023-09-30"
    assert got["revenue"] == pytest.approx(1050.0)


def test_carryforward_is_unstamped_when_the_newest_anchor_works():
    filings = [
        q("2022-12-31", "2023-02-01", timeframe="annual", revenue=1000.0),
        q("2022-03-31", "2022-05-01", revenue=200.0),
        q("2023-03-31", "2023-05-01", revenue=250.0),
    ]
    assert "carried_forward" not in build_trailing_filing_carryforward(filings)


def test_carryforward_never_reaches_forward():
    """Only the newest periods are dropped, so the result stays PIT-safe."""
    filings = [
        q("2022-12-31", "2023-02-01", timeframe="annual", revenue=1000.0),
        q("2022-03-31", "2022-05-01", revenue=200.0),
        q("2023-03-31", "2023-05-01", revenue=250.0),
        q("2023-09-30", "2023-11-01", revenue=280.0),
    ]
    got = build_trailing_filing_carryforward(filings)
    assert got["filing_date"] <= "2023-05-01"


def test_carryforward_with_annual_only_history():
    filings = [q("2023-12-31", "2024-02-01", timeframe="annual", revenue=400.0)]
    assert build_trailing_filing_carryforward(filings)["revenue"] == 400.0


def test_carryforward_returns_none_when_nothing_is_clean():
    assert build_trailing_filing_carryforward([q("2023-03-31", "2023-05-01")]) is None


# ── ratios ───────────────────────────────────────────────────────────────────

FILING = {
    "filing_date": "2024-05-01",
    "period_end": "2024-03-31",
    "revenue": 1000.0,
    "gross_profit": 400.0,
    "operating_income": 200.0,
    "net_income": 150.0,
    "eps_diluted": 1.5,
    "eps_basic": 1.6,
    "shares_diluted": 100.0,
    "total_assets": 5000.0,
    "total_equity": 2000.0,
    "total_debt": 800.0,
    "cash": 300.0,
    "operating_cash_flow": 250.0,
    "free_cash_flow": 180.0,
}


def test_every_declared_field_is_always_present():
    out = compute_ratios(filing=FILING, price=30.0, as_of=date(2024, 6, 3))
    assert set(out) == set(RATIO_FIELDS)


def test_core_ratio_arithmetic():
    out = compute_ratios(filing=FILING, price=30.0, as_of=date(2024, 6, 3))
    assert out["market_cap"] == pytest.approx(3000.0)
    assert out["net_debt"] == pytest.approx(500.0)
    assert out["enterprise_value"] == pytest.approx(3500.0)
    assert out["book_value_per_share"] == pytest.approx(20.0)
    assert out["pe_diluted"] == pytest.approx(20.0)
    assert out["earnings_yield"] == pytest.approx(0.05)
    assert out["p_b"] == pytest.approx(1.5)
    assert out["p_s"] == pytest.approx(3.0)
    assert out["p_fcf"] == pytest.approx(3000.0 / 180.0)
    assert out["ev_to_ebit"] == pytest.approx(3500.0 / 200.0)
    assert out["gross_margin"] == pytest.approx(0.4)
    assert out["roe"] == pytest.approx(0.075)
    assert out["roa"] == pytest.approx(0.03)
    assert out["debt_to_equity"] == pytest.approx(0.4)


def test_no_filing_is_reported_not_guessed():
    out = compute_ratios(filing=None, price=30.0, as_of=date(2024, 6, 3))
    assert out["notes"] == ["no_filing"]
    assert out["pe_diluted"] is None


def test_negative_eps_omits_pe_and_says_so():
    out = compute_ratios(filing={**FILING, "eps_diluted": -2.0}, price=30.0, as_of=date(2024, 6, 3))
    assert out["pe_diluted"] is None
    assert "negative_eps_pe_omitted" in out["notes"]


def test_missing_price_nulls_price_ratios():
    out = compute_ratios(filing=FILING, price=None, as_of=date(2024, 6, 3))
    assert out["market_cap"] is None
    assert out["pe_diluted"] is None
    assert "no_price" in out["notes"]
    # Margins need no price.
    assert out["gross_margin"] == pytest.approx(0.4)


def test_missing_cash_or_debt_nulls_ev_and_names_what_was_missing():
    out = compute_ratios(filing={**FILING, "cash": None}, price=30.0, as_of=date(2024, 6, 3))
    assert out["enterprise_value"] is None
    assert out["ev_to_sales"] is None
    assert "ev_unavailable_missing_cash" in out["notes"]


def test_absent_fcf_is_flagged_rather_than_approximated():
    no_fcf = {**FILING, "free_cash_flow": None}
    out = compute_ratios(filing=no_fcf, price=30.0, as_of=date(2024, 6, 3))
    assert out["p_fcf"] is None
    assert out["fcf_yield"] is None
    assert "fcf_unavailable_use_ocf" in out["notes"]
    assert out["p_ocf"] is not None


def test_ebit_proxy_is_always_disclosed():
    out = compute_ratios(filing=FILING, price=30.0, as_of=date(2024, 6, 3))
    assert "ev_to_ebit_used_no_d_and_a" in out["notes"]


def test_zero_equity_does_not_divide():
    out = compute_ratios(filing={**FILING, "total_equity": 0.0}, price=30.0, as_of=date(2024, 6, 3))
    assert out["roe"] is None
    assert out["debt_to_equity"] is None
    assert out["p_b"] is None


def test_carried_forward_provenance_reaches_the_notes():
    filing = {
        **FILING,
        "carried_forward": True,
        "carried_forward_anchor_period_end": "2023-12-31",
        "carried_forward_latest_period_end": "2024-03-31",
    }
    out = compute_ratios(filing=filing, price=30.0, as_of=date(2024, 6, 3))
    assert any("ttm_carried_forward" in n for n in out["notes"])


def test_garbage_values_become_none_not_exceptions():
    bad = {**FILING, "revenue": "n/a", "total_assets": None}
    out = compute_ratios(filing=bad, price=30.0, as_of=date(2024, 6, 3))
    assert out["p_s"] is None
    assert out["roa"] is None
    assert "no_revenue" in out["notes"]


# ── as a computed source ─────────────────────────────────────────────────────


def test_ratios_source_composes_upstream_kinds(tmp_path):
    from fintel.market.factory import build_data_sources
    from fintel.market.settings import MarketConfig
    from fintel.models.market import DataBinding
    from fintel.pit import Cutoff
    from tests import fixtures

    fixtures.register_all()
    built = build_data_sources(
        [
            DataBinding(kind="prices", source="flat_prices", price=30.0),
            DataBinding(kind="fundamentals", source="annual_fundamentals"),
            DataBinding(kind="ratios", source="valuation_ratios"),
        ],
        config=MarketConfig(cache_root=tmp_path),
    )
    out = built["ratios"].fetch({"symbol": "AAPL"}, Cutoff(date(2024, 6, 3)))
    assert isinstance(built["ratios"], ValuationRatios)
    assert out["price"] == pytest.approx(30.0)
    assert out["pe_diluted"] is not None
    assert set(RATIO_FIELDS) <= set(out)
    assert "entries" in out
    assert isinstance(out["entries"], list)
    assert len(out["entries"]) >= 1
    assert out["date"] == out["entries"][-1]["date"]
    # Daily series is oldest→newest and strictly before the decision date.
    dates = [e["date"] for e in out["entries"]]
    assert dates == sorted(dates)
    assert all(d < "2024-06-03" for d in dates)


def test_daily_ratio_series_matches_delorean_shape():
    """One entry per trading day; TTM advances only when a filing becomes public."""
    from fintel.market.data.ratios import build_daily_ratio_series

    filings = [
        {
            "filing_date": "2024-01-15",
            "period_end": "2023-12-31",
            "timeframe": "annual",
            "revenue": 1000.0,
            "gross_profit": 400.0,
            "operating_income": 200.0,
            "net_income": 150.0,
            "eps_diluted": 1.5,
            "eps_basic": 1.6,
            "shares_diluted": 100.0,
            "total_assets": 5000.0,
            "total_equity": 2000.0,
            "total_debt": 800.0,
            "cash": 300.0,
            "operating_cash_flow": 250.0,
            "free_cash_flow": 180.0,
        },
        {
            "filing_date": "2024-05-01",
            "period_end": "2024-03-31",
            "timeframe": "quarterly",
            "fiscal_period": "Q1",
            "revenue": 280.0,
            "gross_profit": 110.0,
            "operating_income": 55.0,
            "net_income": 40.0,
            "eps_diluted": 0.4,
            "eps_basic": 0.42,
            "shares_diluted": 100.0,
            "total_assets": 5100.0,
            "total_equity": 2050.0,
            "total_debt": 790.0,
            "cash": 310.0,
            "operating_cash_flow": 70.0,
            "free_cash_flow": 50.0,
        },
    ]
    days = pd.bdate_range("2024-01-10", "2024-05-10")
    prices = pd.DataFrame({
        "date": [d.date() for d in days],
        "close": [30.0 + i * 0.1 for i in range(len(days))],
    })
    entries = build_daily_ratio_series(filings=filings, prices=prices)
    assert entries
    assert all("date" in e and "pe_diluted" in e for e in entries)
    # No entry before the first public filing.
    assert all(e["date"] >= "2024-01-15" for e in entries)
    # Price tracks the bar for that day.
    by_date = {e["date"]: e for e in entries}
    mid = next(d for d in by_date if "2024-02" in d)
    assert by_date[mid]["price"] == pytest.approx(
        float(prices.loc[prices["date"].astype(str).str[:10] == mid, "close"].iloc[0])
    )
