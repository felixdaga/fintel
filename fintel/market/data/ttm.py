"""Trailing-window aggregation of filings. Pure math, no I/O, no PIT opinion.

A quarterly 10-Q reports three months, so a P/E built from one quarter's EPS is
4x too high — and P/S, ROE and EV/EBIT are wrong by the same factor. The naive
fix, "sum the last four quarterlies", is also wrong for most US filers: they
file Q1+Q2+Q3+10-K, not Q1+Q2+Q3+Q4, so the naive pick reaches into the prior
fiscal year and double-counts a period.

The analyst formula instead:

    Trailing = LatestAnnual + sum(periods after it) - sum(matching periods a year earlier)

Matching is purely day-based: for each new period, look for a prior period of the
same timeframe whose period_end is within +/-20 days of `new.period_end - 365`.
Fiscal labels like "Q1" are never read, so this handles mid-year fiscal years
(MSFT, AAPL, WMT), 13-week retail calendars that drift a few days annually,
semi-annual filers, and companies that change their fiscal year.

Ported from delorean's valuation/ratios.py with the algorithm unchanged.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime, timedelta
from typing import Any

# Flows accumulate over a period, so they sum across the trailing window.
FLOW_FIELDS: tuple[str, ...] = (
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "operating_expenses",
    "rd_expense",
    "operating_income",
    "net_income",
    # EPS is deliberately excluded: it's per-share, so after a split the
    # denominator changes while net income does not. Summing pre- and
    # post-split EPS through the formula above goes badly wrong (NVDA's 10:1
    # in June 2024 turns trailing EPS negative). Derived below instead.
    "operating_cash_flow",
    "capex",
    "free_cash_flow",
)

# Balance-sheet items are snapshots; summing them is meaningless.
STOCK_FIELDS: tuple[str, ...] = (
    "total_assets",
    "total_liabilities",
    "total_equity",
    "cash",
    "total_debt",
)

# Weighted averages from the most recent filing. Marginally inconsistent with a
# trailing window, but it's the vendor convention and inside rounding noise.
SHARES_FIELDS: tuple[str, ...] = ("shares_basic", "shares_diluted")

# +/-20 days absorbs fiscal calendars that shift year-over-year. Tighter would
# reject good matches; looser risks matching across a fiscal-year reset.
PRIOR_MATCH_TOLERANCE_DAYS = 20

DEFAULT_WINDOW_DAYS = 365


def parse_iso(value: Any) -> Date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, Date):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _dated(filings: list[dict]) -> list[tuple[Date, dict]]:
    """Filings with a parseable period_end, newest first.

    Records with an empty filing_date are dropped: the upstream PIT filter
    should have excluded them, but the vendor has been seen to emit "future"
    records with no filing date at all.
    """
    out = [
        (pe, f)
        for f in filings
        if f.get("filing_date") and f.get("timeframe") and (pe := parse_iso(f.get("period_end")))
    ]
    out.sort(key=lambda pair: (pair[0], str(pair[1].get("filing_date", ""))), reverse=True)
    return out


def _annual_only(filing: dict, window_days: int) -> dict:
    out = dict(filing)
    out["ttm_method"] = "annual_only"
    out["ttm_window_days"] = window_days
    out["ttm_anchor_period_end"] = filing.get("period_end")
    return out


def build_trailing_filing(
    filings: list[dict] | None, *, window_days: int = DEFAULT_WINDOW_DAYS
) -> dict | None:
    """A synthetic trailing-window filing, anchored at the newest period.

    Returns None when no clean aggregate exists — no annual anchor, a missing
    prior-year match, or missing flow fields. A refusal beats publishing a
    number that can't be defended: mis-pairing a December quarter with a
    September one overlaps three months and silently double-counts, which looks
    entirely plausible in a report.

    PIT is the caller's job; pass only filings with `filing_date < as_of`.
    """
    dated = _dated(filings or [])
    if not dated:
        return None

    latest = dated[0][1]
    annuals = [(pe, f) for pe, f in dated if f.get("timeframe") == "annual"]
    non_annuals = [(pe, f) for pe, f in dated if f.get("timeframe") != "annual"]

    # The newest filing is itself an annual, so it already spans ~365 days.
    if latest.get("timeframe") == "annual":
        return _annual_only(latest, window_days)
    if not annuals:
        return None

    annual_end, latest_annual = annuals[0]
    newer = [(pe, f) for pe, f in non_annuals if pe > annual_end]
    if not newer:
        return _annual_only(latest_annual, window_days)

    pairs: list[tuple[dict, dict]] = []
    for nf_end, nf in newer:
        target = nf_end - timedelta(days=365)
        candidates = [
            (abs((pe - target).days), f)
            for pe, f in non_annuals
            if f.get("timeframe") == nf.get("timeframe") and pe <= annual_end
        ]
        within = sorted(
            (c for c in candidates if c[0] <= PRIOR_MATCH_TOLERANCE_DAYS), key=lambda c: c[0]
        )
        if not within:
            return None
        pairs.append((nf, within[0][1]))

    out: dict[str, Any] = {
        "form": latest.get("form", ""),
        "filing_date": latest.get("filing_date", ""),
        "period_end": latest.get("period_end", ""),
        "timeframe": "ttm",
        "fiscal_period": "TTM",
        "fiscal_year": latest.get("fiscal_year", ""),
        "source_url": latest.get("source_url", ""),
        # Provenance, so a reviewer can reconstruct any number by hand.
        "ttm_method": "annual_plus_delta",
        "ttm_window_days": window_days,
        "ttm_anchor_period_end": latest.get("period_end"),
        "ttm_annual_period_end": latest_annual.get("period_end"),
        "ttm_newer_period_ends": [nf.get("period_end") for nf, _ in pairs],
        "ttm_prior_period_ends": [pf.get("period_end") for _, pf in pairs],
    }

    for fld in FLOW_FIELDS:
        annual_value = latest_annual.get(fld)
        new_values = [nf.get(fld) for nf, _ in pairs]
        old_values = [pf.get(fld) for _, pf in pairs]
        numbers = [annual_value, *new_values, *old_values]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in numbers):
            out[fld] = float(annual_value) + sum(new_values) - sum(old_values)
        else:
            out[fld] = None

    for fld in (*STOCK_FIELDS, *SHARES_FIELDS):
        out[fld] = latest.get(fld)

    # Split-safe EPS: trailing net income over the latest share count.
    net_income = out.get("net_income")
    if isinstance(net_income, (int, float)):
        for eps_field, shares_field in (
            ("eps_diluted", "shares_diluted"),
            ("eps_basic", "shares_basic"),
        ):
            shares = out.get(shares_field)
            if isinstance(shares, (int, float)) and shares > 0:
                out[eps_field] = net_income / shares
    return out


def build_trailing_filing_carryforward(
    filings: list[dict] | None, *, window_days: int = DEFAULT_WINDOW_DAYS
) -> dict | None:
    """Gap-resilient: anchor at the newest period that yields a clean window.

    The strict builder returns None when the provider is missing a prior-year
    quarter for the newest anchor. This walks *back* to the next-newest anchor
    and retries. It only ever drops the newest periods and never reaches
    forward, so the result stays PIT-safe — it's just slightly stale, and says
    so via `carried_forward`.
    """
    dated = _dated(filings or [])
    if not dated:
        return None

    non_annual_ends = sorted(
        {pe for pe, f in dated if f.get("timeframe") != "annual"}, reverse=True
    )
    if not non_annual_ends:
        return build_trailing_filing(filings, window_days=window_days)

    latest_end = non_annual_ends[0]
    for anchor_end in non_annual_ends:
        subset = [f for pe, f in dated if pe <= anchor_end]
        trailing = build_trailing_filing(subset, window_days=window_days)
        if trailing is not None:
            if anchor_end != latest_end:
                trailing["carried_forward"] = True
                trailing["carried_forward_anchor_period_end"] = anchor_end.isoformat()
                trailing["carried_forward_latest_period_end"] = latest_end.isoformat()
            return trailing
    return None
