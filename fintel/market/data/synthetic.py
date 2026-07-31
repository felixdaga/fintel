"""Deterministic offline prices. Smoke runs and tests, no network, no key."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta

import pandas as pd

from fintel.market.calendar import TradingCalendar
from fintel.market.data.base import require
from fintel.models.common import Symbol
from fintel.pit import Cutoff


def _seed(symbol: str) -> int:
    return int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)


@dataclass
class SyntheticPrices:
    """A seeded random walk on real trading days, stable per symbol."""

    start: Date = Date(2015, 1, 2)
    end: Date = Date(2030, 12, 31)
    base_price: float = 100.0
    daily_vol: float = 0.015
    calendar: TradingCalendar = field(default_factory=TradingCalendar)
    name: str = "synthetic_prices"
    kinds: tuple[str, ...] = ("prices",)

    def bars(self, symbol: Symbol) -> pd.DataFrame:
        import numpy as np

        days = self.calendar.days(self.start, self.end)
        rng = np.random.default_rng(_seed(symbol))
        steps = rng.normal(0.0003, self.daily_vol, size=len(days))
        closes = self.base_price * np.exp(np.cumsum(steps))
        opens = closes * (1.0 + rng.normal(0, 0.002, size=len(days)))
        return pd.DataFrame(
            {
                "date": days,
                "open": opens,
                "high": np.maximum(opens, closes) * 1.004,
                "low": np.minimum(opens, closes) * 0.996,
                "close": closes,
                "volume": rng.integers(1_000_000, 9_000_000, size=len(days)),
            }
        )

    def fetch(self, query: dict, cutoff: Cutoff) -> pd.DataFrame:
        symbol = require(query, "symbol", self.name)
        lookback = int(query.get("lookback_days", 365))
        df = cutoff.clamp_frame(self.bars(symbol), "date")
        floor = cutoff.decision_date - timedelta(days=lookback)
        return df[df["date"] >= floor].reset_index(drop=True)
