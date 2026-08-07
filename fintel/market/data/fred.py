"""FRED macroeconomic time series — free, PIT-safe.

Vendor: St. Louis Fed's free API (https://fred.stlouisfed.org/docs/api/). A
free ``FRED_API_KEY`` is required for network fills. Each observation carries a
date, so PIT is enforced for real: we ask FRED for the needed span and then
drop any observation whose date is not strictly before the decision date.

Cache-first policy lives in :mod:`fintel.market.cache` (``ensure_records``).
On disk: ``{cache_root}/macro/{SERIES_ID}.json`` + sibling ``.meta.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx

from fintel.market.cache import ensure_records
from fintel.market.data.base import DataError
from fintel.market.data.store import RecordCache, atomic_write
from fintel.pit import Cutoff

logger = logging.getLogger(__name__)

FRED_API_BASE = "https://api.stlouisfed.org/fred"
REQUEST_TIMEOUT = 30.0
DEFAULT_LOOKBACK_DAYS = 365

# Curated human-friendly aliases -> FRED series IDs. Anything not listed is
# used verbatim as a raw FRED series ID, so power users are never limited to
# this set. Lifted from TradingAgents/dataflows/fred.py.
MACRO_SERIES: dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",
    "federal_funds_rate": "FEDFUNDS",
    "fed_funds": "FEDFUNDS",
    "3m_treasury": "DGS3MO",
    "2y_treasury": "DGS2",
    "10y_treasury": "DGS10",
    "30y_treasury": "DGS30",
    "10y_2y_spread": "T10Y2Y",
    "yield_curve": "T10Y2Y",
    "10y_3m_spread": "T10Y3M",
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "pce": "PCEPI",
    "core_pce": "PCEPILFE",
    "inflation_expectations": "T10YIE",
    "real_gdp": "GDPC1",
    "gdp": "GDP",
    "industrial_production": "INDPRO",
    "unemployment_rate": "UNRATE",
    "unemployment": "UNRATE",
    "nonfarm_payrolls": "PAYEMS",
    "payrolls": "PAYEMS",
    "initial_claims": "ICSA",
    "m2": "M2SL",
    "money_supply": "M2SL",
    "vix": "VIXCLS",
    "dollar_index": "DTWEXBGS",
    "consumer_sentiment": "UMCSENT",
    "housing_starts": "HOUST",
    "retail_sales": "RSAFS",
    "wti_oil": "DCOILWTICO",
    "crude_oil": "DCOILWTICO",
    "aaa_yield": "DAAA",
    "baa_yield": "DBAA",
}

# Default bundle for a *weekly* DJIA-30 single-name rating: the series that
# actually move on a weekly horizon and differentially drive cross-sectional
# DJIA-30 returns. Leads with daily movers (rates, curve, vol, FX, breakeven,
# oil, credit) plus the one weekly-frequency series (initial claims). Slow
# monthly/quarterly series (CPI, UNRATE, GDP, sentiment, housing, retail
# sales, IP) stay available via explicit `indicator=` reads but are not in
# the default block — they'd repeat a stale value across ~4-13 weekly
# decisions.
DEFAULT_BUNDLE: tuple[str, ...] = (
    "fed_funds_rate",
    "10y_treasury",
    "2y_treasury",
    "3m_treasury",
    "10y_2y_spread",
    "10y_3m_spread",
    "vix",
    "dollar_index",
    "inflation_expectations",
    "wti_oil",
    "baa_yield",
    "initial_claims",
)


def _resolve_series_id(indicator: str) -> str:
    """Map a friendly alias to a FRED series ID, or pass a raw ID through."""
    key = indicator.strip().lower().replace(" ", "_").replace("-", "_")
    if key in MACRO_SERIES:
        return MACRO_SERIES[key]
    candidate = indicator.strip().upper()
    # FRED series IDs never contain whitespace and are short; reject anything
    # else (a descriptive phrase) rather than 400ing the API.
    if not candidate or len(candidate) > 30 or any(c.isspace() for c in candidate):
        raise ValueError(
            f"{indicator!r} is not a known macro alias or a valid FRED series ID. "
            f"Use an alias (e.g. 'cpi', 'unemployment', '10y_treasury') or a raw "
            f"FRED series ID (e.g. 'CPIAUCSL')."
        )
    return candidate


@dataclass
class FredMacro:
    """FRED macroeconomic series, PIT-clamped on observation date, cache-first."""

    cache: RecordCache
    api_key: str | None = None
    base_url: str = FRED_API_BASE
    timeout: float = REQUEST_TIMEOUT
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    indicators: tuple[str, ...] = DEFAULT_BUNDLE
    name: str = "fred_macro"
    kinds: tuple[str, ...] = ("macro",)
    _client: httpx.Client | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.indicators, list):
            self.indicators = tuple(self.indicators)
        if self.api_key:
            self._client = httpx.Client(timeout=self.timeout)

    def fetch(self, query: dict, cutoff: Cutoff) -> dict[str, Any]:
        decision_date = cutoff.decision_date
        lookback = int(query.get("lookback_days", self.lookback_days))
        # PIT: agent may see through the prior day only.
        through = decision_date - timedelta(days=1)
        since = decision_date - timedelta(days=lookback)
        # An explicit indicator yields a single-series read; omitting it returns
        # the curated bundle, so an evidence pack renders a macro block in one call.
        requested = (query.get("indicator"),) if query.get("indicator") else self.indicators

        series_out: list[dict[str, Any]] = []
        for ind in requested:
            try:
                series_id = _resolve_series_id(str(ind))
            except ValueError as exc:
                # An unknown alias is a soft failure for one series, not a whole read.
                logger.warning("fred: %s", exc)
                continue
            entry = self._assemble_series(series_id, since, through, decision_date)
            if entry is not None:
                series_out.append(entry)

        return {
            "as_of": decision_date.isoformat(),
            "lookback_days": lookback,
            "series": series_out,
        }

    def warm(self, since: Date, through: Date) -> None:
        """Gap-fill the default (or configured) bundle for a prefetch window."""
        for ind in self.indicators:
            try:
                series_id = _resolve_series_id(str(ind))
            except ValueError as exc:
                logger.warning("fred warm: %s", exc)
                continue
            self._ensure(series_id, since, through)

    def _assemble_series(
        self, series_id: str, since: Date, through: Date, decision_date: Date
    ) -> dict[str, Any] | None:
        records = self._ensure(series_id, since, through)
        meta = self._read_meta(series_id) or {
            "series_id": series_id,
            "title": series_id,
            "units": "",
            "frequency": "",
            "seasonal_adjustment": "",
        }
        # Window + PIT: keep observations in [since, through] and strictly
        # before the decision date (defense in depth if coverage is loose).
        points = [
            (str(r["date"]), str(r["value"]))
            for r in records
            if r.get("date")
            and r.get("value") not in (".", None, "")
            and since.isoformat() <= str(r["date"]) <= through.isoformat()
            and str(r["date"]) < decision_date.isoformat()
        ]
        window = f"{since.isoformat()} to {decision_date.isoformat()}"
        base = {
            "series_id": series_id,
            "title": meta.get("title", series_id),
            "units": meta.get("units", ""),
            "frequency": meta.get("frequency", ""),
            "seasonal_adjustment": meta.get("seasonal_adjustment", ""),
            "window": window,
            "observations": [{"date": d, "value": v} for d, v in points],
        }
        if not points:
            return base

        first_date, first_val = points[0]
        last_date, last_val = points[-1]
        change: float | None = None
        change_pct: float | None = None
        try:
            change = float(last_val) - float(first_val)
            base_v = float(first_val)
            change_pct = (change / base_v * 100.0) if base_v != 0 else None
        except ValueError:
            pass

        return {
            **base,
            "latest_date": last_date,
            "latest_value": last_val,
            "change": change,
            "change_pct": change_pct,
        }

    def ensure(self, key: str, since: Date, through: Date) -> list[dict]:
        """Public warm path — cache policy lives in market.cache."""
        return ensure_records(
            self.cache,
            key,
            since,
            through,
            fetch_span=lambda lo, hi: self._fetch_span(key, lo, hi),
            identity=lambda r: str(r.get("date", "")),
            sort=lambda r: str(r.get("date", "")),
            online=bool(self._client is not None and self.api_key),
            source_name=self.name,
            kind_label="macro",
        )

    def _ensure(self, series_id: str, since: Date, through: Date) -> list[dict]:
        return self.ensure(series_id, since, through)

    def _fetch_span(self, series_id: str, lo: Date, hi: Date) -> list[dict]:
        """Vendor I/O only: fetch one span and persist series meta."""
        assert self._client is not None
        meta_raw = self._get("series", {"series_id": series_id})
        info = (meta_raw.get("seriess") or [{}])[0]
        meta = {
            "series_id": series_id,
            "title": info.get("title", series_id),
            "units": info.get("units_short") or info.get("units", ""),
            "frequency": info.get("frequency", ""),
            "seasonal_adjustment": info.get("seasonal_adjustment_short", ""),
        }
        self._write_meta(series_id, meta)
        obs = (
            self._get(
                "series/observations",
                {
                    "series_id": series_id,
                    "observation_start": lo.isoformat(),
                    "observation_end": hi.isoformat(),
                    "sort_order": "asc",
                },
            ).get("observations")
            or []
        )
        return [
            {"date": o["date"], "value": o["value"]}
            for o in obs
            if o.get("value") not in (".", None, "")
            and o.get("date")
            and lo.isoformat() <= o["date"] <= hi.isoformat()
        ]

    def _meta_path(self, series_id: str) -> Path:
        return self.cache.path(series_id).with_name(f"{series_id}.meta.json")

    def _read_meta(self, series_id: str) -> dict[str, Any] | None:
        path = self._meta_path(series_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("corrupt macro meta %s: %s", path, exc)
            return None

    def _write_meta(self, series_id: str, meta: dict[str, Any]) -> None:
        atomic_write(self._meta_path(series_id), json.dumps(meta, default=str))

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self.api_key is not None
        p = {**params, "api_key": self.api_key, "file_type": "json"}
        url = f"{self.base_url}/{path}"
        try:
            resp = self._client.get(url, params=p)  # type: ignore[union-attr]
        except httpx.RequestError as exc:
            raise DataError(f"fred network error for {path}: {exc}") from exc
        if resp.status_code == 400:
            try:
                message = resp.json().get("error_message", resp.text)
            except ValueError:
                message = resp.text
            raise DataError(f"fred request failed: {message}")
        if resp.status_code == 401 or resp.status_code == 403:
            raise DataError(f"fred rejected the key ({resp.status_code}); check FRED_API_KEY")
        if resp.status_code >= 400:
            raise DataError(f"fred HTTP {resp.status_code} for {path}: {resp.text[:200]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise DataError(f"fred non-JSON response for {path}: {exc}") from exc

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
