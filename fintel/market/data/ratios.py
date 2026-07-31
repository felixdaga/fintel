"""Valuation ratios — a computed kind, derived from prices + fundamentals.

Why the platform computes these rather than letting the agent divide: one
chokepoint for the trailing-window math, and the agent reads a finished number
instead of doing arithmetic that invites "a P/E of 30 looks rich" from training
priors rather than from this filing.

Any ratio whose denominator is missing, zero, or non-positive where positivity
is required comes back None — never a silent zero, never a NaN that propagates
through an agent's reasoning.

Computed on demand from already-clamped upstream data, so PIT is enforced once
in the upstream sources. The old code precomputed a daily `cache/ratios/` series
from the full cache and re-filtered per day, which meant a second PIT
implementation to keep correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any

from fintel.market.data.base import DataSource, require
from fintel.market.data.ttm import build_trailing_filing_carryforward
from fintel.pit import Cutoff

# The published roster. Order is the reading order surfaced to an agent.
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


@dataclass
class ValuationRatios:
    """Trailing ratios from whichever price and fundamentals sources are bound."""

    upstream: dict[str, DataSource] = field(default_factory=dict)
    window_days: int = 365
    name: str = "valuation_ratios"
    kinds: tuple[str, ...] = ("ratios",)

    def fetch(self, query: dict, cutoff: Cutoff) -> dict:
        symbol = require(query, "symbol", self.name)
        filings = self.upstream["fundamentals"].fetch(
            {"symbol": symbol, "lookback_days": query.get("filings_lookback_days", 1460)}, cutoff
        )
        bars = self.upstream["prices"].fetch({"symbol": symbol, "lookback_days": 30}, cutoff)

        price = None
        if bars is not None and len(bars) and "close" in bars.columns:
            price = as_float(bars["close"].iloc[-1])

        # Newest-first is what the trailing builder documents as its input.
        trailing = build_trailing_filing_carryforward(
            list(reversed(filings or [])), window_days=self.window_days
        )
        return compute_ratios(filing=trailing, price=price, as_of=cutoff.decision_date)
