"""Web search, PIT-controlled.

The one kind whose query text is not known in advance, so the cache is keyed by
`(to_date, from_date, query)` rather than by symbol. `to_date` is
`decision_date - 1`. The provider's freshness window is the first PIT control;
when ``clamp_by_age`` is on (catalog default), results are post-filtered using
Brave's ``sources[url].age`` so soft freshness leaks do not reach the agent.

An exact key match is a hit. That makes a replay deterministic and keeps a run
honest about when it left the frozen cache. Age clamp is applied in memory on
every read (cache or live) and does not rewrite the cached payload — turning
the knob off still sees the full stored response.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta
from pathlib import Path
from typing import Any

from fintel.market.data.base import DataError, require
from fintel.market.data.store import atomic_write
from fintel.pit import Cutoff

logger = logging.getLogger(__name__)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/llm/context"
DEFAULT_LOOKBACK_DAYS = 30


def query_hash(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:12]


def parse_brave_age(age: Any) -> Date | None:
    """Brave ``sources[url].age`` → calendar date, or None when undated.

    Fixed positions (Brave LLM Context): ``[human, YYYY-MM-DD, relative, ISO8601]``.
    Prefer index 1; fall back to the date portion of index 3. Never use the
    relative string (index 2) — it is wall-clock relative and wrong for
    backtests.
    """
    if not isinstance(age, (list, tuple)):
        return None
    for idx in (1, 3):
        if len(age) <= idx:
            continue
        raw = age[idx]
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            return Date.fromisoformat(raw.strip()[:10])
        except ValueError:
            continue
    return None


def clamp_brave_by_age(
    payload: Any, since: Date, through: Date
) -> Any:
    """Keep only ``grounding.generic`` items whose Brave age is in ``[since, through]``.

    Undated items (empty / missing ``age``) are kept — we cannot prove a leak.
    The parallel ``sources`` map is pruned to the kept URLs. Non-dict payloads
    (empty / unexpected) pass through unchanged.
    """
    if not isinstance(payload, dict):
        return payload
    meta = payload.get("sources")
    grounding = payload.get("grounding")
    if not isinstance(meta, dict) or not isinstance(grounding, dict):
        return payload
    generic = grounding.get("generic")
    if not isinstance(generic, list):
        return payload

    kept: list[dict] = []
    for item in generic:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        entry = meta.get(url) if isinstance(url, str) else None
        age = entry.get("age") if isinstance(entry, dict) else None
        ymd = parse_brave_age(age)
        if ymd is None or since <= ymd <= through:
            kept.append(item)

    kept_urls = {g.get("url") for g in kept if isinstance(g.get("url"), str)}
    pruned_meta = {u: meta[u] for u in kept_urls if u in meta}
    out = dict(payload)
    out["grounding"] = {**grounding, "generic": kept}
    out["sources"] = pruned_meta
    return out


@dataclass
class WebSearch:
    """Freshness-windowed search. `api_key=None` means cache-only."""

    cache_root: Path
    api_key: str | None = None
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    max_results: int = 10
    # Render cap carried for the evidence pack (not used for fetching). The
    # factory bakes the strategy binding / catalog default here; the policy
    # surfaces it to the renderer. See catalog.py `web_search` params.
    snippet_max_chars: int = 640
    # Post-fetch PIT: filter Brave results using sources[url].age against the
    # search window. Catalog default True; strategies override via [[data]].
    clamp_by_age: bool = True
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

        blob: dict | None = None
        if path.exists():
            try:
                blob = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("corrupt web_search cache %s: %s", path, exc)

        if blob is None:
            if self.api_key is None:
                raise DataError(
                    f"{self.name}: no cached result for {text!r} in "
                    f"[{since}, {through}] and no API key configured"
                )
            blob = self._search(text, since, through)
            atomic_write(path, json.dumps(blob))

        return self._maybe_clamp(blob, since, through)

    def _maybe_clamp(self, blob: dict, since: Date, through: Date) -> dict:
        if not self.clamp_by_age:
            return blob
        sources = blob.get("sources")
        clamped = clamp_brave_by_age(sources, since, through)
        if clamped is sources:
            return blob
        out = dict(blob)
        out["sources"] = clamped
        return out

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
