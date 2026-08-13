"""Country health — curated FRED bundle for both parties in a trade dispute.

A geopol event package needs macro indicators for the involved countries. This
source wraps :class:`fintel.market.data.fred.FredMacro` with a curated bundle of
FRED series relevant to a tariff/trade dispute, organised by country. Both
party cells see the same data (symmetric information access — no home-team
advantage).

The bundle covers:

- **USA**: rates, dollar, sentiment, IP, CPI, credit, oil, vol, plus bilateral
  trade with China (IMPCH / EXPCH / BOPGTB) and CNY/USD.
- **CHN**: industrial production, CPI, discount rate, reserves, merchandise
  trade, CNY/USD.

Missing / retired FRED series soft-fail (logged + skipped) so one bad ID cannot
take down the whole tool. The package may also ship a
``country_overrides.json`` for non-FRED indicators.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from pathlib import Path
from typing import Any

from fintel.market.data.base import DataError
from fintel.market.data.fred import FredMacro, _resolve_series_id
from fintel.market.data.store import RecordCache
from fintel.pit import Cutoff

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 180

# Friendly aliases (via fred.MACRO_SERIES) + raw FRED IDs for bilateral trade.
USA_SERIES: tuple[str, ...] = (
    "fed_funds_rate",
    "10y_treasury",
    "2y_treasury",
    "dollar_index",
    "consumer_sentiment",
    "industrial_production",
    "cpi",
    "baa_yield",
    "wti_oil",
    "vix",
    "IMPCH",  # US imports from China
    "EXPCH",  # US exports to China
    "BOPGTB",  # US goods trade balance
    "DEXCHUS",  # CNY per USD (daily)
)

# Verified FRED series IDs for China-side indicators (as of 2026).
CHN_SERIES: tuple[str, ...] = (
    "CHNPRINTO01IXPYM",  # Industrial production excl. construction
    "CHNCPIALLMINMEI",  # CPI
    "INTDSRCNM193N",  # Discount rate
    "TRESEGCNM052N",  # Total reserves excl. gold
    "XTEXVA01CNM667S",  # Merchandise exports
    "XTIMVA01CNM667S",  # Merchandise imports
    "EXCHUS",  # CNY per USD (monthly)
)


@dataclass
class CountryHealth:
    """Curated FRED bundle for both parties, symmetric access.

    Delegates to :class:`FredMacro` for the actual FRED fetch + cache. The
    ``country_overrides_file`` is an optional JSON file with extra indicators
    not in FRED (e.g. PBOC data), keyed by country.
    """

    cache: RecordCache
    api_key: str | None = None
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    country_overrides_file: str | None = None
    name: str = "country_health"
    kinds: tuple[str, ...] = ("country_health",)
    _fred: FredMacro | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._fred = FredMacro(
            cache=self.cache,
            api_key=self.api_key,
            lookback_days=self.lookback_days,
            indicators=(),  # we fetch per-series, not the default bundle
        )

    def fetch(self, query: dict, cutoff: Cutoff) -> dict[str, Any]:
        decision_date = cutoff.decision_date
        lookback = int(query.get("lookback_days", self.lookback_days))
        since = decision_date - timedelta(days=lookback)
        through = decision_date - timedelta(days=1)

        countries: dict[str, list[dict]] = {
            "USA": self._fetch_series(USA_SERIES, "USA", since, through, decision_date),
            "CHN": self._fetch_series(CHN_SERIES, "CHN", since, through, decision_date),
        }

        overrides = self._load_overrides()
        for country, series_list in overrides.items():
            if country not in countries:
                countries[country] = []
            countries[country].extend(series_list)

        return {
            "as_of": decision_date.isoformat(),
            "lookback_days": lookback,
            "countries": countries,
        }

    def _fetch_series(
        self,
        series_ids: tuple[str, ...],
        country: str,
        since: Date,
        through: Date,
        decision_date: Date,
    ) -> list[dict]:
        out: list[dict] = []
        for raw in series_ids:
            try:
                series_id = _resolve_series_id(str(raw))
            except ValueError as exc:
                logger.warning("country_health: %s", exc)
                continue
            try:
                entry = self._fred._assemble_series(series_id, since, through, decision_date)
            except DataError as exc:
                # One missing/retired FRED ID must not fail the whole tool.
                logger.warning("country_health: skip %s: %s", series_id, exc)
                continue
            if entry is not None:
                entry["country"] = country
                out.append(entry)
        return out

    def _load_overrides(self) -> dict[str, list[dict]]:
        if not self.country_overrides_file:
            return {}
        path = Path(self.country_overrides_file)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("country_health: overrides file %s: %s", path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def warm(self, since: Date, through: Date) -> None:
        for raw in (*USA_SERIES, *CHN_SERIES):
            try:
                series_id = _resolve_series_id(str(raw))
            except ValueError:
                continue
            try:
                self._fred._ensure(series_id, since, through)
            except DataError as exc:
                logger.warning("country_health warm: skip %s: %s", series_id, exc)
