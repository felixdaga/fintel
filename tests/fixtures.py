"""Stand-ins for third-party extensions, so the `module:Callable` seam is
exercised by something outside the platform rather than assumed to work."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from fintel.market.catalog import Field, Param, SourceInfo
from fintel.market.data.base import DataSource, require
from fintel.market.settings import MarketConfig
from fintel.market.universe import StaticUniverse
from fintel.pit import Cutoff


def tiny_universe(**_: object) -> StaticUniverse:
    """A custom universe needing no platform services stays trivial."""
    return StaticUniverse(symbols=("XYZ",), name="tiny")


def cache_aware_universe(*, config: MarketConfig, tag: str = "") -> StaticUniverse:
    """One that needs the cache root declares `config` and is handed it."""
    return StaticUniverse(symbols=(config.cache_root.name, tag), name="cache_aware")


# ── a third-party provider for an existing kind ──────────────────────────────
# Stands in for yfinance: same `prices` kind, different vendor, no platform edit.


@dataclass
class FlatPrices:
    price: float = 50.0
    name: str = "flat_prices"
    kinds: tuple[str, ...] = ("prices",)

    def fetch(self, query: dict, cutoff: Cutoff) -> pd.DataFrame:
        require(query, "symbol", self.name)
        days = pd.bdate_range(end=cutoff.decision_date, periods=30, inclusive="left")
        frame = pd.DataFrame({"date": [d.date() for d in days], "close": self.price})
        return cutoff.clamp_frame(frame, "date").reset_index(drop=True)


def flat_prices(**params: object) -> FlatPrices:
    return FlatPrices(**{k: v for k, v in params.items() if k == "price"})


FLAT_PRICES_INFO = SourceInfo(
    name="flat_prices",
    kind="prices",
    provider="testing",
    target="tests.fixtures:flat_prices",
    fields=(Field("date", "date"), Field("close", "number", unit="usd")),
    params=(Param("price", "number", 50.0),),
    description="Constant close. Stands in for a second price vendor.",
)


# ── a computed kind, derived from two upstream kinds ─────────────────────────


@dataclass
class SimpleRatios:
    """Trailing P/E from whatever price and fundamentals sources are bound.

    The point is the wiring: a computed source names upstream *kinds*, not
    vendors, so swapping the price provider changes nothing here.
    """

    upstream: dict[str, DataSource] = field(default_factory=dict)
    name: str = "simple_ratios"
    kinds: tuple[str, ...] = ("ratios",)

    def fetch(self, query: dict, cutoff: Cutoff) -> dict:
        symbol = require(query, "symbol", self.name)
        bars = self.upstream["prices"].fetch({"symbol": symbol}, cutoff)
        filings = self.upstream["fundamentals"].fetch({"symbol": symbol}, cutoff)
        price = float(bars["close"].iloc[-1]) if len(bars) else None
        eps = filings[-1].get("eps_diluted") if filings else None
        # A missing input yields None, never a placeholder number.
        pe = price / eps if price and eps and eps > 0 else None
        return {"as_of": cutoff.decision_date.isoformat(), "price": price, "pe": pe}


def simple_ratios(*, upstream: dict[str, DataSource], **_: object) -> SimpleRatios:
    return SimpleRatios(upstream=upstream)


SIMPLE_RATIOS_INFO = SourceInfo(
    name="simple_ratios",
    kind="ratios",
    provider="computed",
    target="tests.fixtures:simple_ratios",
    fields=(
        Field("as_of", "date"),
        Field("price", "number", unit="usd"),
        Field("pe", "number", "Trailing price / diluted EPS", "ratio"),
    ),
    derives_from=("prices", "fundamentals"),
    description="Computed kind. Declares upstream kinds instead of a vendor.",
)


@dataclass
class StubFundamentals:
    eps: float | None = 5.0
    name: str = "stub_fundamentals"
    kinds: tuple[str, ...] = ("fundamentals",)

    def fetch(self, query: dict, cutoff: Cutoff) -> list[dict]:
        require(query, "symbol", self.name)
        filed = cutoff.decision_date - timedelta(days=7)
        return [{"filing_date": filed.isoformat(), "eps_diluted": self.eps}]


def stub_fundamentals(**params: object) -> StubFundamentals:
    return StubFundamentals(**{k: v for k, v in params.items() if k == "eps"})


STUB_FUNDAMENTALS_INFO = SourceInfo(
    name="stub_fundamentals",
    kind="fundamentals",
    provider="testing",
    target="tests.fixtures:stub_fundamentals",
    fields=(Field("filing_date", "date"), Field("eps_diluted", "number", unit="usd")),
    params=(Param("eps", "number", 5.0),),
)


# ── fundamentals shaped so a real trailing window can be assembled ───────────
# Annual anchor, one new quarter, and its prior-year counterpart.

_ANNUAL_HISTORY: tuple[dict, ...] = (
    {
        "filing_date": "2022-05-01", "period_end": "2022-03-31", "timeframe": "quarterly",
        "revenue": 200.0, "net_income": 20.0, "shares_diluted": 100.0,
    },
    {
        "filing_date": "2023-02-01", "period_end": "2022-12-31", "timeframe": "annual",
        "revenue": 1000.0, "net_income": 150.0, "gross_profit": 400.0,
        "operating_income": 200.0, "shares_diluted": 100.0,
    },
    {
        "filing_date": "2023-05-01", "period_end": "2023-03-31", "timeframe": "quarterly",
        "revenue": 250.0, "net_income": 30.0, "shares_diluted": 100.0,
        "total_assets": 5000.0, "total_equity": 2000.0, "total_debt": 800.0, "cash": 300.0,
    },
)  # fmt: skip


@dataclass
class AnnualFundamentals:
    name: str = "annual_fundamentals"
    kinds: tuple[str, ...] = ("fundamentals",)

    def fetch(self, query: dict, cutoff: Cutoff) -> list[dict]:
        require(query, "symbol", self.name)
        # Ascending by availability, like every real record source.
        return cutoff.clamp_records(list(_ANNUAL_HISTORY), "filing_date")


def annual_fundamentals(**_: object) -> AnnualFundamentals:
    return AnnualFundamentals()


ANNUAL_FUNDAMENTALS_INFO = SourceInfo(
    name="annual_fundamentals",
    kind="fundamentals",
    provider="testing",
    target="tests.fixtures:annual_fundamentals",
    fields=(
        Field("filing_date", "date"),
        Field("period_end", "date"),
        Field("timeframe", "text"),
        Field("revenue", "number", unit="usd"),
        Field("net_income", "number", unit="usd"),
    ),
)


def register_all() -> None:
    """What a third party ships: register, and it's pickable by name."""
    from fintel.market import catalog

    for info in (
        FLAT_PRICES_INFO,
        SIMPLE_RATIOS_INFO,
        STUB_FUNDAMENTALS_INFO,
        ANNUAL_FUNDAMENTALS_INFO,
    ):
        catalog.register_source(info, replace=True)
