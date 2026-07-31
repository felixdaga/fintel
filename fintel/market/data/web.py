"""Web search, PIT-controlled.

The one kind whose query text is not known in advance, so the cache is keyed by
`(to_date, from_date, query)` rather than by symbol. `to_date` is
`decision_date - 1`: the provider's freshness window is what enforces PIT here,
because there's no per-result date to clamp on afterwards.

An exact key match is a hit. That makes a replay deterministic and keeps a run
honest about when it left the frozen cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from fintel.market.data.base import DataError, require
from fintel.market.data.store import atomic_write
from fintel.pit import Cutoff

logger = logging.getLogger(__name__)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/llm/context"
DEFAULT_LOOKBACK_DAYS = 30


def query_hash(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:12]


@dataclass
class WebSearch:
    """Freshness-windowed search. `api_key=None` means cache-only."""

    cache_root: Path
    api_key: str | None = None
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    max_results: int = 10
    name: str = "web_search"
    kinds: tuple[str, ...] = ("web_search",)

    def path(self, query: str, since: Date, through: Date) -> Path:
        slug = f"{through.isoformat()}_{since.isoformat()}_{query_hash(query)}"
        return self.cache_root / "web_search" / f"{slug}.json"

    def window(self, cutoff: Cutoff, lookback_days: int) -> tuple[Date, Date]:
        through = cutoff.decision_date - timedelta(days=1)
        return through - timedelta(days=lookback_days), through

    def fetch(self, query: dict, cutoff: Cutoff) -> dict:
        text = require(query, "query", self.name)
        lookback = int(query.get("lookback_days", self.lookback_days))
        since, through = self.window(cutoff, lookback)
        path = self.path(text, since, through)

        if path.exists():
            try:
                blob = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("corrupt web_search cache %s: %s", path, exc)
            else:
                return blob

        if self.api_key is None:
            raise DataError(
                f"{self.name}: no cached result for {text!r} in "
                f"[{since}, {through}] and no API key configured"
            )
        blob = self._search(text, since, through)
        atomic_write(path, json.dumps(blob))
        return blob

    def _search(self, text: str, since: Date, through: Date) -> dict:
        import httpx

        freshness = f"{since.isoformat()}to{through.isoformat()}"
        try:
            resp = httpx.get(
                BRAVE_ENDPOINT,
                params={"q": text, "freshness": freshness, "count": self.max_results},
                headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            raise DataError(f"{self.name}: network error for {text!r}: {exc}") from exc
        if resp.status_code >= 400:
            raise DataError(f"{self.name}: HTTP {resp.status_code} for {text!r}: {resp.text[:200]}")
        return {
            "query": text,
            "search_window": {"from": since.isoformat(), "to": through.isoformat()},
            "sources": resp.json() if resp.content else [],
        }
