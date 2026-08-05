"""Alpha Vantage News & Sentiment — free, PIT-safe, per-ticker.

Vendor: Alpha Vantage's ``NEWS_SENTIMENT`` endpoint
(https://www.alphavantage.co/documentation/#news-sentiment). A free
``ALPHA_VANTAGE_API_KEY`` is required. Unlike the social feeds, this endpoint
honours historical ``time_from``/``time_to`` windows, so it is PIT-safe for a
historical backtest: we ask for articles in
``[decision_date - lookback, decision_date)`` and re-drop anything stamped on
or after the decision date.

The value over the existing ``massive_news`` kind is the per-article sentiment
score and the per-ticker relevance score — a quantitative sentiment read
alongside the qualitative headlines, which is what a weekly rater needs and
what ``massive_news``'s ``insights`` only sometimes carry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from typing import Any

import httpx

from fintel.market.data.base import DataError, require
from fintel.pit import Cutoff

logger = logging.getLogger(__name__)

API_BASE_URL = "https://www.alphavantage.co/query"
REQUEST_TIMEOUT = 30.0
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_LIMIT = 50


def _fmt_av_date(d: Date) -> str:
    """Alpha Vantage's ``YYYYMMDDTHHMM`` time format (midnight UTC)."""
    return d.strftime("%Y%m%d") + "T0000"


def _parse_av_date(s: str) -> str | None:
    """Convert AV's ``YYYYMMDDTHHMM`` (or an ISO string) to a ``YYYY-MM-DD`` date.

    Returns ``None`` when the value is empty or unparseable. AV stamps articles
    with ``YYYYMMDDTHHMM``; we normalise to ISO so the PIT compare and the
    ``published_at`` field are honest.
    """
    if not s:
        return None
    s = s.strip()
    # AV format: 20260728T1300 → 2026-07-28
    if len(s) >= 8 and s[:8].isdigit() and s[4].isdigit() and s[6].isdigit():
        try:
            return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
        except (ValueError, IndexError):
            pass
    # Already ISO-ish: 2026-07-28[ T...]
    try:
        return Date.fromisoformat(s[:10]).isoformat()
    except ValueError:
        return None


@dataclass
class AlphaVantageNews:
    """Per-ticker news + sentiment from Alpha Vantage, PIT-clamped on publish date."""

    api_key: str
    base_url: str = API_BASE_URL
    timeout: float = REQUEST_TIMEOUT
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    limit: int = DEFAULT_LIMIT
    name: str = "alphavantage_news"
    kinds: tuple[str, ...] = ("av_news",)
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.Client(timeout=self.timeout)

    def fetch(self, query: dict, cutoff: Cutoff) -> list[dict[str, Any]]:
        symbol = require(query, "symbol", self.name)
        decision_date = cutoff.decision_date
        lookback = int(query.get("lookback_days", self.lookback_days))
        limit = int(query.get("limit", self.limit))

        # PIT: half-open window ending strictly before the decision date.
        time_from = _fmt_av_date(decision_date - timedelta(days=lookback))
        time_to = _fmt_av_date(decision_date - timedelta(days=1))

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "time_from": time_from,
            "time_to": time_to,
            "limit": str(limit),
            "apikey": self.api_key,
            "source": "fintel",
        }
        data = self._request(params)
        feed = data.get("feed") or []

        ceil_iso = decision_date.isoformat()
        out: list[dict[str, Any]] = []
        for item in feed:
            pub = _parse_av_date(item.get("time_published") or "")
            # Belt-and-suspenders PIT: AV's window is inclusive of time_to, so
            # drop anything stamped on/after the decision date.
            if pub is None or pub >= ceil_iso:
                continue
            out.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "published_at": pub,
                    "source": item.get("source") or item.get("source_domain", ""),
                    "overall_sentiment_score": _float(item.get("overall_sentiment_score")),
                    "overall_sentiment_label": item.get("overall_sentiment_label"),
                    "ticker_sentiment": [
                        {
                            "ticker": ts.get("ticker"),
                            "relevance_score": _float(ts.get("relevance_score")),
                            "ticker_sentiment_score": _float(ts.get("ticker_sentiment_score")),
                            "ticker_sentiment_label": ts.get("ticker_sentiment_label"),
                        }
                        for ts in (item.get("ticker_sentiment") or [])
                        if isinstance(ts, dict)
                    ],
                }
            )
        return out

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._client.get(self.base_url, params=params)
        except httpx.RequestError as exc:
            raise DataError(f"alphavantage network error: {exc}") from exc
        if resp.status_code >= 400:
            raise DataError(f"alphavantage HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise DataError(f"alphavantage non-JSON response: {exc}") from exc

        # AV reports problems via "Information"/"Note" instead of HTTP status.
        notice = body.get("Information") or body.get("Note")
        if notice:
            low = notice.lower()
            if any(m in low for m in ("rate limit", "requests per day", "call frequency", "premium")):
                raise DataError(f"alphavantage rate limit: {notice}")
            if "api key" in low or "apikey" in low:
                raise DataError(f"alphavantage key invalid/missing: {notice}")
        return body

    def close(self) -> None:
        self._client.close()


def _float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
