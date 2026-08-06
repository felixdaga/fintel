"""Valuation ratios — a computed kind, derived from prices + fundamentals.

Why the platform computes these rather than letting the agent divide: one
chokepoint for the trailing-window math, and the agent reads a finished number
instead of doing arithmetic that invites "a P/E of 30 looks rich" from training
priors rather than from this filing.

Any ratio whose denominator is missing, zero, or non-positive where positivity
is required comes back None — never a silent zero, never a NaN that propagates
through an agent's reasoning.

History shape matches Delorean's ``cache/ratios/SYMBOL.json``: one entry per
trading day, using that day's close and a TTM built from filings with
``filing_date <= day``. Computed on demand from already-clamped upstream data,
so PIT is enforced once in the upstream sources; the daily march never sees a
bar or filing at or after the decision date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any

import pandas as pd

from fintel.market.data.base import DataSource, require
from fintel.market.data.ttm import build_trailing_filing_carryforward
from fintel.pit import Cutoff

# The published roster for one day's snapshot. Order is the reading order
# surfaced to an agent. ``date`` / ``entries`` are history wrappers added by
# ``ValuationRatios.fetch``, not by ``compute_ratios``.
RATIO_FIELDS: tuple[str, ...] = (
    # anchors
    "as_of",
    "price",
    "filing_date",
    "period_end",
    "shares_diluted",
    "market_cap",
    "enterprise_value",
    "net_debt",
    "book_value_per_share",
    "ebit",
    # valuation
    "pe_diluted",
    "pe_basic",
    "p_b",
    "p_s",
    "p_fcf",
    "p_ocf",
    "ev_to_sales",
    "ev_to_ebit",
    "earnings_yield",
    "fcf_yield",
    # profitability
    "gross_margin",
    "operating_margin",
    "net_margin",
    "fcf_margin",
    "roe",
    "roa",
    # leverage
    "debt_to_equity",
    "debt_to_assets",
    # provenance
    "notes",
)

DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_FILINGS_LOOKBACK_DAYS = 1460


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_ratios(*, filing: dict | None, price: float | None, as_of: Date) -> dict:
    """Every key in RATIO_FIELDS, with None wherever the input didn't support it.

    Never raises on bad input; bad input becomes None fields plus a `notes`
    entry explaining which one was missing.
    """
    out: dict[str, Any] = dict.fromkeys(RATIO_FIELDS)
    out["as_of"] = as_of.isoformat()
    notes: list[str] = []

    if not filing:
        out["notes"] = ["no_filing"]
        return out

    out["filing_date"] = filing.get("filing_date") or None
    out["period_end"] = filing.get("period_end") or None

    revenue = as_float(filing.get("revenue"))
    gross_profit = as_float(filing.get("gross_profit"))
    op_income = as_float(filing.get("operating_income"))
    net_income = as_float(filing.get("net_income"))
    eps_basic = as_float(filing.get("eps_basic"))
    eps_diluted = as_float(filing.get("eps_diluted"))
    shares_diluted = as_float(filing.get("shares_diluted"))
    total_assets = as_float(filing.get("total_assets"))
    total_equity = as_float(filing.get("total_equity"))
    total_debt = as_float(filing.get("total_debt"))
    cash = as_float(filing.get("cash"))
    ocf = as_float(filing.get("operating_cash_flow"))
    fcf = as_float(filing.get("free_cash_flow"))

    out["shares_diluted"] = shares_diluted
    # EBIT stands in for EBITDA: the source carries no D&A line.
    out["ebit"] = op_income
    if price is not None:
        out["price"] = float(price)

    if shares_diluted and shares_diluted > 0:
        if price is not None:
            out["market_cap"] = float(price) * shares_diluted
        if total_equity is not None:
            out["book_value_per_share"] = total_equity / shares_diluted

    if total_debt is not None and cash is not None:
        out["net_debt"] = total_debt - cash

    mcap = out["market_cap"]
    if mcap is not None and total_debt is not None and cash is not None:
        out["enterprise_value"] = mcap + total_debt - cash

    # Valuation. P/E only when EPS is positive — a negative "multiple" is not a
    # multiple, and None is the honest answer.
    if price is not None:
        if eps_diluted is not None and eps_diluted > 0:
            out["pe_diluted"] = price / eps_diluted
            out["earnings_yield"] = eps_diluted / price
        if eps_basic is not None and eps_basic > 0:
            out["pe_basic"] = price / eps_basic
        bvps = out["book_value_per_share"]
        if bvps is not None and bvps > 0:
            out["p_b"] = price / bvps

    if mcap is not None and mcap > 0:
        if revenue is not None and revenue > 0:
            out["p_s"] = mcap / revenue
        if fcf is not None and fcf > 0:
            out["p_fcf"] = mcap / fcf
            out["fcf_yield"] = fcf / mcap
        if ocf is not None and ocf > 0:
            out["p_ocf"] = mcap / ocf

    ev = out["enterprise_value"]
    if ev is not None and ev > 0:
        if revenue is not None and revenue > 0:
            out["ev_to_sales"] = ev / revenue
        if op_income is not None and op_income > 0:
            out["ev_to_ebit"] = ev / op_income

    if revenue is not None and revenue > 0:
        if gross_profit is not None:
            out["gross_margin"] = gross_profit / revenue
        if op_income is not None:
            out["operating_margin"] = op_income / revenue
        if net_income is not None:
            out["net_margin"] = net_income / revenue
        if fcf is not None:
            out["fcf_margin"] = fcf / revenue

    if total_equity is not None and total_equity > 0 and net_income is not None:
        out["roe"] = net_income / total_equity
    if total_assets is not None and total_assets > 0 and net_income is not None:
        out["roa"] = net_income / total_assets

    if total_debt is not None and total_equity is not None and total_equity > 0:
        out["debt_to_equity"] = total_debt / total_equity
    if total_debt is not None and total_assets is not None and total_assets > 0:
        out["debt_to_assets"] = total_debt / total_assets

    if price is None:
        notes.append("no_price")
    if not shares_diluted or shares_diluted <= 0:
        notes.append("no_diluted_shares")
    if revenue is None or revenue <= 0:
        notes.append("no_revenue")
    if eps_diluted is not None and eps_diluted <= 0:
        notes.append("negative_eps_pe_omitted")
    if fcf is None:
        # A source that rolls all investing activity into one line can't give a
        # clean capex split. Manufacturing an FCF that mixes in M&A and
        # securities would be worse than leaving the derived ratios null.
        notes.append("fcf_unavailable_use_ocf")
    if out["enterprise_value"] is None:
        missing = [n for n, v in (("debt", total_debt), ("cash", cash)) if v is None]
        if missing:
            notes.append(f"ev_unavailable_missing_{'_'.join(missing)}")
    # Always true for this source; say it so nobody reads ev_to_ebit as EV/EBITDA.
    notes.append("ev_to_ebit_used_no_d_and_a")
    if filing.get("carried_forward"):
        notes.append(
            f"ttm_carried_forward: source gap near "
            f"{filing.get('carried_forward_latest_period_end')}; anchored at last clean "
            f"period {filing.get('carried_forward_anchor_period_end')}"
        )

    out["notes"] = notes
    return out


def build_daily_ratio_series(
    *,
    filings: list[dict],
    prices: pd.DataFrame | None,
    window_days: int = 365,
) -> list[dict]:
    """One ratio entry per trading day — Delorean ``compute_ratios_cache`` math.

    Each day's ratios use:
      • price = that day's closing price
      • filing = TTM from all filings whose ``filing_date <= day`` (PIT)

    Returns entries oldest→newest. Empty when there are no bars or no public
    filing yet for any day in the window.
    """
    if prices is None or not len(prices) or "close" not in prices.columns:
        return []
    if "date" not in prices.columns:
        return []

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame = frame.sort_values("date").reset_index(drop=True)

    dated_filings = sorted(
        [f for f in filings if f.get("filing_date")],
        key=lambda f: str(f["filing_date"])[:10],
    )

    entries: list[dict[str, Any]] = []
    active_filings: list[dict] = []
    filing_idx = 0
    ttm_cache: dict | None = None

    for _, row in frame.iterrows():
        day: Date = row["date"]
        day_str = day.isoformat()
        close = as_float(row["close"])
        if close is None:
            continue

        advanced = False
        while filing_idx < len(dated_filings):
            fd = str(dated_filings[filing_idx].get("filing_date", ""))[:10]
            if fd and fd <= day_str:
                active_filings.append(dated_filings[filing_idx])
                filing_idx += 1
                advanced = True
            else:
                break

        if not active_filings:
            continue

        if advanced or ttm_cache is None:
            ttm_cache = build_trailing_filing_carryforward(active_filings, window_days=window_days)

        ratio_dict = compute_ratios(filing=ttm_cache, price=close, as_of=day)
        entry: dict[str, Any] = {"date": day_str}
        entry.update(ratio_dict)
        entries.append(entry)

    return entries


@dataclass
class ValuationRatios:
    """Daily trailing-ratio history from whichever price/fundamentals are bound.

    ``fetch`` returns the latest snapshot fields (same roster as before) plus
    ``date`` and ``entries`` — the Delorean-shaped daily series inside the
    requested lookback, all strictly before the decision date.
    """

    upstream: dict[str, DataSource] = field(default_factory=dict)
    window_days: int = 365
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    filings_lookback_days: int = DEFAULT_FILINGS_LOOKBACK_DAYS
    name: str = "valuation_ratios"
    kinds: tuple[str, ...] = ("ratios",)

    def fetch(self, query: dict, cutoff: Cutoff) -> dict:
        symbol = require(query, "symbol", self.name)
        lookback = int(query.get("lookback_days", self.lookback_days))
        filings_lookback = int(query.get("filings_lookback_days", self.filings_lookback_days))
        # Cover the price window plus enough history for a clean TTM at the
        # start of that window (annual+delta needs ~2y of filings).
        fund_lookback = max(filings_lookback, lookback + self.window_days + 400)

        filings = self.upstream["fundamentals"].fetch(
            {"symbol": symbol, "lookback_days": fund_lookback}, cutoff
        )
        bars = self.upstream["prices"].fetch({"symbol": symbol, "lookback_days": lookback}, cutoff)

        entries = build_daily_ratio_series(
            filings=list(filings or []),
            prices=bars,
            window_days=self.window_days,
        )
        # Belt-and-suspenders: never return a bar on/after the decision date.
        ceil = cutoff.decision_date.isoformat()
        entries = [e for e in entries if (e.get("date") or "") < ceil]

        if not entries:
            empty = compute_ratios(filing=None, price=None, as_of=cutoff.decision_date)
            empty["date"] = None
            empty["entries"] = []
            return empty

        latest = entries[-1]
        out = {key: latest.get(key) for key in RATIO_FIELDS}
        out["date"] = latest.get("date")
        out["entries"] = entries
        return out
