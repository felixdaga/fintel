"""Massive.com sources: prices, fundamentals, news.

Cache-first. A miss fetches only the spans coverage says are absent, so
arbitrary date jumps across a backtest don't refetch the world.

Offline with nothing cached raises rather than returning empty — coverage is
what makes "we never fetched this" distinguishable from "there genuinely is no
data here", and conflating them is how a run reports zero news for a live name.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from typing import Any

import pandas as pd

from fintel.market.data import coverage as cov
from fintel.market.data.base import DataError, require
from fintel.market.data.http import MassiveClient
from fintel.market.data.store import PriceStore, RecordCache
from fintel.models.common import Symbol
from fintel.pit import Cutoff

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_YEARS = 5


def default_history_start(today: Date | None = None) -> Date:
    d = today or Date.today()
    return d.replace(year=d.year - DEFAULT_HISTORY_YEARS, month=1, day=1)


def _num(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


# ── prices ───────────────────────────────────────────────────────────────────


@dataclass
class MassivePrices:
    """Split/dividend-adjusted daily bars.

    `lookback_days` is calendar days. The previous implementation named the
    param the same but applied it as a row count, so `365` meant 365 *sessions*
    (~17 months); a re-run under this source sees the year it asked for.
    """

    store: PriceStore
    client: MassiveClient | None = None
    history_start: Date = field(default_factory=default_history_start)
    adjusted: bool = True
    name: str = "massive_prices"
    kinds: tuple[str, ...] = ("prices",)

    def fetch(self, query: dict, cutoff: Cutoff) -> pd.DataFrame:
        symbol = require(query, "symbol", self.name)
        lookback = int(query.get("lookback_days", 365))
        df = self._ensure(symbol, cutoff.decision_date - timedelta(days=1))
        if df is None:
            return pd.DataFrame(columns=list(query.get("fields") or []))
        out = cutoff.clamp_frame(df, "date")
        floor = cutoff.decision_date - timedelta(days=lookback)
        out = out[out["date"] >= floor]
        fields = query.get("fields")
        if fields:
            keep = [c for c in out.columns if c == "date" or c in set(fields)]
            out = out[keep]
        return out.reset_index(drop=True)

    def _ensure(self, symbol: Symbol, through: Date) -> pd.DataFrame | None:
        need_from = self.history_start
        if through < need_from:
            return self.store.read(symbol)
        gaps = cov.missing(self.store.coverage(symbol), need_from, through)
        if not gaps:
            return self.store.read(symbol)
        if self.client is None:
            cached = self.store.read(symbol)
            if cached is None:
                raise DataError(
                    f"{self.name}: no cached prices for {symbol} covering "
                    f"[{need_from}, {through}] and no network access configured"
                )
            logger.warning(
                "%s: %s cache is short of [%s, %s] by %d span(s); serving what is cached",
                self.name,
                symbol,
                need_from,
                through,
                len(gaps),
            )
            return cached
        for lo, hi in gaps:
            fresh = self._fetch_bars(symbol, lo, hi)
            if fresh is not None and not fresh.empty:
                self.store.merge(symbol, fresh, (lo, hi))
            else:
                # Record the attempt so a genuinely dataless span isn't re-fetched.
                self.store.write(
                    symbol,
                    self.store.read(symbol) or _empty_bars(),
                    [*self.store.coverage(symbol), (lo, hi)],
                )
        return self.store.read(symbol)

    def _fetch_bars(self, symbol: Symbol, start: Date, end: Date) -> pd.DataFrame | None:
        assert self.client is not None
        path = f"/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        results = self.client.paginate(
            path, {"adjusted": "true" if self.adjusted else "false", "sort": "asc", "limit": 50000}
        )
        if not results:
            return None
        rows = [
            {
                "date": pd.Timestamp(r["t"], unit="ms").date(),
                "open": r.get("o"),
                "high": r.get("h"),
                "low": r.get("l"),
                "close": r.get("c"),
                "volume": r.get("v"),
            }
            for r in results
            if r.get("t") is not None
        ]
        return pd.DataFrame(rows) if rows else None


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.Series(dtype="object")})


# ── record sources (fundamentals, news) ──────────────────────────────────────


def normalise_financial(item: dict) -> dict:
    """Flatten a /vX/reference/financials result."""
    fin = item.get("financials") or {}
    inc = fin.get("income_statement") or {}
    bal = fin.get("balance_sheet") or {}
    cf = fin.get("cash_flow_statement") or {}

    def val(statement: dict, *keys: str) -> float | None:
        for k in keys:
            got = _num(statement.get(k))
            if got is not None:
                return got
        return None

    ocf = val(cf, "net_cash_flow_from_operating_activities")
    capex = val(cf, "payments_for_property_plant_and_equipment", "capital_expenditures")
    # capex is negative in filings, so addition gives FCF. When capex isn't
    # reported separately, FCF stays None: approximating it from total investing
    # cash flow is nonsense in any quarter with material M&A.
    fcf = ocf + capex if ocf is not None and capex is not None else None

    return {
        "form": item.get("source_filing_type", "10-Q"),
        "filing_date": item.get("filing_date", ""),
        "period_end": item.get("end_date") or item.get("period_of_report_date", ""),
        "timeframe": item.get("timeframe", "quarterly"),
        "fiscal_period": item.get("fiscal_period", ""),
        "fiscal_year": item.get("fiscal_year", ""),
        "source_url": item.get("source_filing_url", ""),
        "revenue": val(
            inc,
            "revenues",
            "net_revenues",
            "revenue_from_contract_with_customer_excluding_assessed_tax",
        ),
        "cost_of_revenue": val(inc, "cost_of_revenue"),
        "gross_profit": val(inc, "gross_profit"),
        "operating_expenses": val(inc, "operating_expenses"),
        "rd_expense": val(inc, "research_and_development"),
        "operating_income": val(inc, "operating_income_loss"),
        "net_income": val(inc, "net_income_loss"),
        "eps_basic": val(inc, "basic_earnings_per_share"),
        "eps_diluted": val(inc, "diluted_earnings_per_share"),
        # Weighted averages over the period — the EPS convention.
        "shares_basic": val(inc, "basic_average_shares"),
        "shares_diluted": val(inc, "diluted_average_shares"),
        "total_assets": val(bal, "assets"),
        "total_liabilities": val(bal, "liabilities"),
        "total_equity": val(bal, "equity"),
        "cash": val(
            bal,
            "cash",
            "cash_and_cash_equivalents_including_restricted_cash",
            "cash_and_cash_equivalents_at_carrying_value",
        ),
        "total_debt": val(bal, "long_term_debt_and_capital_lease_obligations", "long_term_debt"),
        "operating_cash_flow": ocf,
        "capex": capex,
        "free_cash_flow": fcf,
    }


def financial_identity(r: dict) -> tuple:
    """period_end keeps an amendment from masquerading as a new record, and
    keeps two timeframes sharing a filing_date distinct."""
    return (
        str(r.get("filing_date", "")),
        str(r.get("period_end", "")),
        str(r.get("timeframe", "")),
    )


def normalise_article(item: dict) -> dict:
    published = str(item.get("published_utc") or "")
    return {
        "id": item.get("id") or "",
        "title": item.get("title") or "",
        "published_at": published[:10],
        "published_utc": published,
        "publisher": (item.get("publisher") or {}).get("name", ""),
        "summary": item.get("description") or "",
        "url": item.get("article_url") or "",
        "tickers": item.get("tickers") or [],
        "insights": item.get("insights") or [],
    }


@dataclass(frozen=True)
class RecordSpec:
    """How one dated-record kind is fetched, clamped and de-duplicated."""

    kind: str
    endpoint: str
    cutoff_field: str  # the availability stamp — what PIT clamps on
    date_param: str  # provider's range parameter
    normalise: Callable[[dict], dict]
    identity: Callable[[dict], Any]
    lookback_days: int
    extra_filter: Callable[[dict, Date], bool] | None = None


FUNDAMENTALS = RecordSpec(
    kind="fundamentals",
    endpoint="/vX/reference/financials",
    cutoff_field="filing_date",
    date_param="filing_date",
    normalise=normalise_financial,
    identity=financial_identity,
    lookback_days=730,
    # A statement whose period hasn't ended yet is not knowable, even if the
    # vendor stamped it with an early filing date.
    extra_filter=lambda r, on: str(r.get("period_end", "")) <= on.isoformat(),
)

NEWS = RecordSpec(
    kind="news",
    endpoint="/v2/reference/news",
    cutoff_field="published_at",
    date_param="published_utc",
    normalise=normalise_article,
    identity=lambda r: r.get("id") or (r.get("title"), r.get("published_at")),
    lookback_days=90,
)


@dataclass
class MassiveRecords:
    """Dated records with interval-coverage caching, clamped on availability."""

    spec: RecordSpec
    cache: RecordCache
    client: MassiveClient | None = None
    name: str = "massive_records"

    @property
    def kinds(self) -> tuple[str, ...]:
        return (self.spec.kind,)

    def fetch(self, query: dict, cutoff: Cutoff) -> list[dict]:
        symbol = require(query, "symbol", self.name)
        lookback = int(query.get("lookback_days", self.spec.lookback_days))
        through = cutoff.decision_date - timedelta(days=1)
        since = cutoff.decision_date - timedelta(days=lookback)
        records = self._ensure(symbol, since, through)
        out = cutoff.clamp_records(records, self.spec.cutoff_field)
        out = [r for r in out if str(r.get(self.spec.cutoff_field, "")) >= since.isoformat()]
        if self.spec.extra_filter:
            out = [r for r in out if self.spec.extra_filter(r, cutoff.decision_date)]
        limit = query.get("limit")
        return out[-int(limit) :] if limit else out

    def _ensure(self, symbol: Symbol, since: Date, through: Date) -> list[dict]:
        coverage, records = self.cache.read(symbol)
        gaps = cov.missing(coverage, since, through)
        if not gaps:
            return records
        if self.client is None:
            if not coverage:
                raise DataError(
                    f"{self.name}: nothing cached for {symbol} {self.spec.kind} covering "
                    f"[{since}, {through}] and no network access configured"
                )
            logger.warning(
                "%s: %s %s cache is short of [%s, %s]; serving what is cached",
                self.name,
                symbol,
                self.spec.kind,
                since,
                through,
            )
            return records
        merged = {self.spec.identity(r): r for r in records}
        for lo, hi in gaps:
            for item in self._fetch_span(symbol, lo, hi):
                rec = self.spec.normalise(item)
                merged[self.spec.identity(rec)] = rec
            coverage = cov.coalesce([*coverage, (lo, hi)])
        out = sorted(merged.values(), key=lambda r: str(r.get(self.spec.cutoff_field, "")))
        self.cache.write(symbol, coverage, out)
        return out

    def _fetch_span(self, symbol: Symbol, lo: Date, hi: Date) -> list[dict]:
        assert self.client is not None
        return self.client.paginate(
            self.spec.endpoint,
            {
                "ticker": symbol,
                f"{self.spec.date_param}.gte": lo.isoformat(),
                f"{self.spec.date_param}.lte": hi.isoformat(),
                "limit": 100,
                "sort": self.spec.date_param,
            },
        )
