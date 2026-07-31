"""News sentiment — a computed kind, derived from the news kind.

Aggregates the provider's per-article insights into a daily net score in
[-1, +1]. Days with no sentiment data are omitted rather than filled with zero:
"no coverage" and "balanced coverage" are different facts and a zero would make
them indistinguishable to a signal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from fintel.market.data.base import DataSource, require
from fintel.pit import Cutoff


@dataclass
class NewsSentiment:
    upstream: dict[str, DataSource] = field(default_factory=dict)
    name: str = "news_sentiment"
    kinds: tuple[str, ...] = ("news_sentiment",)

    def fetch(self, query: dict, cutoff: Cutoff) -> dict:
        symbol = require(query, "symbol", self.name)
        articles = self.upstream["news"].fetch(
            {"symbol": symbol, "lookback_days": query.get("lookback_days", 90)}, cutoff
        )

        daily: dict[str, dict[str, int]] = defaultdict(lambda: {"pos": 0, "neg": 0, "neu": 0})
        for article in articles or []:
            day = str(article.get("published_at") or article.get("published_utc") or "")[:10]
            if not day:
                continue
            for insight in article.get("insights") or []:
                ticker = str(insight.get("ticker") or "")
                mood = str(insight.get("sentiment") or "").lower()
                if not ticker or not mood or ticker.upper() != symbol.upper():
                    continue
                bucket = {"positive": "pos", "negative": "neg"}.get(mood, "neu")
                daily[day][bucket] += 1

        series = []
        for day in sorted(daily):
            counts = daily[day]
            total = counts["pos"] + counts["neg"] + counts["neu"]
            if total:
                score = round((counts["pos"] - counts["neg"]) / total, 3)
                series.append({"date": day, "score": score, "n": total})

        return {
            "as_of": cutoff.decision_date.isoformat(),
            "series": series,
            "n_articles": len(articles or []),
            "n_scored": sum(entry["n"] for entry in series),
            "mean_score": (
                round(sum(e["score"] for e in series) / len(series), 3) if series else None
            ),
        }
