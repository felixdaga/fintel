"""The library: what universes and data fields exist, and where they come from.

A strategy picks by name. This module is what makes picking possible — without a
browsable catalog, "extensible" only means "editable by whoever wrote it".

Three things are separable on purpose:

  kind      what the agent asks for      "prices"
  source    who answers                  "massive_prices" | "yfinance_prices"
  fields    what comes back              open, high, low, close, volume

So swapping a provider is a one-line manifest change, and a computed kind
(`derives_from`) is a source like any other — it just declares the upstream
kinds it needs instead of a vendor.

Register a source to add to the library; nothing here is a closed set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

DType = Literal["number", "text", "date", "bool", "list"]
UniverseKind = Literal["index_history", "snapshot", "custom"]


@dataclass(frozen=True)
class Field:
    name: str
    dtype: DType
    description: str = ""
    unit: str | None = None  # usd, shares, ratio, bps, count


@dataclass(frozen=True)
class Param:
    """A query knob a strategy may set in its `[[data]]` block."""

    name: str
    dtype: DType
    default: Any = None
    description: str = ""


@dataclass(frozen=True)
class SourceInfo:
    name: str
    kind: str
    provider: str  # massive | yfinance | computed | synthetic | <third party>
    target: str  # module:Callable
    fields: tuple[Field, ...] = ()
    params: tuple[Param, ...] = ()
    requires_env: tuple[str, ...] = ()
    derives_from: tuple[str, ...] = ()  # upstream kinds, for computed sources
    description: str = ""

    @property
    def is_computed(self) -> bool:
        return bool(self.derives_from)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)


@dataclass(frozen=True)
class UniverseInfo:
    name: str
    label: str
    kind: UniverseKind
    point_in_time: bool
    target: str
    n_symbols: int | None = None
    as_of: str | None = None  # meaningful for snapshots only
    description: str = ""


_SOURCES: dict[str, SourceInfo] = {}
_UNIVERSES: dict[str, UniverseInfo] = {}


# ── registration ─────────────────────────────────────────────────────────────


def register_source(info: SourceInfo, *, replace: bool = False) -> SourceInfo:
    if info.name in _SOURCES and not replace:
        raise ValueError(f"data source {info.name!r} is already registered")
    if not info.fields:
        raise ValueError(
            f"data source {info.name!r} must declare its fields; an undocumented "
            f"source cannot be picked from a library"
        )
    _SOURCES[info.name] = info
    return info


def register_universe(info: UniverseInfo, *, replace: bool = False) -> UniverseInfo:
    if info.name in _UNIVERSES and not replace:
        raise ValueError(f"universe {info.name!r} is already registered")
    _UNIVERSES[info.name] = info
    return info


# ── browsing ─────────────────────────────────────────────────────────────────


def sources(*, kind: str | None = None, provider: str | None = None) -> list[SourceInfo]:
    out = list(_SOURCES.values())
    if kind is not None:
        out = [s for s in out if s.kind == kind]
    if provider is not None:
        out = [s for s in out if s.provider == provider]
    return sorted(out, key=lambda s: (s.kind, s.name))


def source(name: str) -> SourceInfo:
    if name not in _SOURCES:
        raise KeyError(f"unknown data source {name!r}; registered: {sorted(_SOURCES)}")
    return _SOURCES[name]


def has_source(name: str) -> bool:
    return name in _SOURCES


def kinds() -> list[str]:
    return sorted({s.kind for s in _SOURCES.values()})


def providers() -> list[str]:
    return sorted({s.provider for s in _SOURCES.values()})


def fields_for(name: str) -> tuple[Field, ...]:
    return source(name).fields


def universes(*, point_in_time: bool | None = None) -> list[UniverseInfo]:
    out = list(_UNIVERSES.values())
    if point_in_time is not None:
        out = [u for u in out if u.point_in_time is point_in_time]
    return sorted(out, key=lambda u: u.name)


def universe(name: str) -> UniverseInfo:
    if name not in _UNIVERSES:
        raise KeyError(f"unknown universe {name!r}; registered: {sorted(_UNIVERSES)}")
    return _UNIVERSES[name]


def has_universe(name: str) -> bool:
    return name in _UNIVERSES


# ── builtin field rosters ────────────────────────────────────────────────────

PRICE_FIELDS: tuple[Field, ...] = (
    Field("date", "date", "Session date"),
    Field("open", "number", "Split/dividend-adjusted open", "usd"),
    Field("high", "number", "Adjusted high", "usd"),
    Field("low", "number", "Adjusted low", "usd"),
    Field("close", "number", "Adjusted close", "usd"),
    Field("volume", "number", "Shares traded", "shares"),
)

FUNDAMENTAL_FIELDS: tuple[Field, ...] = (
    Field("form", "text", "Filing form type, e.g. 10-Q"),
    Field("filing_date", "date", "When the filing became public — the PIT stamp"),
    Field("period_end", "date", "Last day of the reported period"),
    Field("timeframe", "text", "quarterly | annual"),
    Field("fiscal_period", "text", "Q1..Q4 / FY"),
    Field("fiscal_year", "text", "Issuer's fiscal year label"),
    Field("source_url", "text", "Link to the filing"),
    Field("revenue", "number", "Total revenue", "usd"),
    Field("cost_of_revenue", "number", "Cost of revenue", "usd"),
    Field("gross_profit", "number", "Gross profit", "usd"),
    Field("operating_expenses", "number", "Operating expenses", "usd"),
    Field("rd_expense", "number", "Research and development", "usd"),
    Field("operating_income", "number", "Operating income", "usd"),
    Field("net_income", "number", "Net income", "usd"),
    Field("eps_basic", "number", "Basic EPS", "usd"),
    Field("eps_diluted", "number", "Diluted EPS", "usd"),
    Field("shares_basic", "number", "Weighted-average basic shares", "shares"),
    Field("shares_diluted", "number", "Weighted-average diluted shares", "shares"),
    Field("total_assets", "number", "Total assets", "usd"),
    Field("total_liabilities", "number", "Total liabilities", "usd"),
    Field("total_equity", "number", "Total equity", "usd"),
    Field("cash", "number", "Cash and equivalents", "usd"),
    Field("total_debt", "number", "Long-term debt incl. capital leases", "usd"),
    Field("operating_cash_flow", "number", "Net cash from operations", "usd"),
    Field("capex", "number", "Capital expenditure, negative as filed", "usd"),
    Field("free_cash_flow", "number", "OCF + capex; None when capex isn't broken out", "usd"),
)

NEWS_FIELDS: tuple[Field, ...] = (
    Field("id", "text", "Provider article id"),
    Field("title", "text", "Headline"),
    Field("published_at", "date", "Publication date — the PIT stamp"),
    Field("published_utc", "text", "Full publication timestamp"),
    Field("publisher", "text", "Publisher name"),
    Field("summary", "text", "Provider description"),
    Field("url", "text", "Article link"),
    Field("tickers", "list", "Symbols the provider tagged"),
    Field("insights", "list", "Provider sentiment annotations"),
)

LOOKBACK = Param("lookback_days", "number", 365, "Calendar days of history to serve")


def register_builtins() -> None:
    """Idempotent so importing twice (or a reload in tests) is harmless."""
    register_source(
        SourceInfo(
            name="massive_prices",
            kind="prices",
            provider="massive",
            target="fintel.market.factory:massive_prices",
            fields=PRICE_FIELDS,
            params=(LOOKBACK, Param("fields", "list", None, "Subset of columns to return")),
            requires_env=("MASSIVE_API_KEY",),
            description="Adjusted daily bars, cached per symbol as parquet.",
        ),
        replace=True,
    )
    register_source(
        SourceInfo(
            name="massive_fundamentals",
            kind="fundamentals",
            provider="massive",
            target="fintel.market.factory:massive_fundamentals",
            fields=FUNDAMENTAL_FIELDS,
            params=(Param("lookback_days", "number", 730), Param("limit", "number")),
            requires_env=("MASSIVE_API_KEY",),
            description="Normalised financial statements, clamped on filing_date.",
        ),
        replace=True,
    )
    register_source(
        SourceInfo(
            name="massive_news",
            kind="news",
            provider="massive",
            target="fintel.market.factory:massive_news",
            fields=NEWS_FIELDS,
            params=(Param("lookback_days", "number", 90), Param("limit", "number")),
            requires_env=("MASSIVE_API_KEY",),
            description="Per-ticker articles, clamped on published_at.",
        ),
        replace=True,
    )
    register_source(
        SourceInfo(
            name="synthetic_prices",
            kind="prices",
            provider="synthetic",
            target="fintel.market.factory:synthetic_prices",
            fields=PRICE_FIELDS,
            params=(
                LOOKBACK,
                Param("base_price", "number", 100.0),
                Param("daily_vol", "number", 0.015),
            ),
            description="Deterministic seeded walk on real trading days. No network or key.",
        ),
        replace=True,
    )

    for name, (_key, label) in _index_presets().items():
        register_universe(
            UniverseInfo(
                name=name,
                label=label,
                kind="index_history",
                point_in_time=True,
                target="fintel.market.constituents:historical_universe",
                description=f"Opt-in/opt-out membership for {label}. Survivorship-clean.",
            ),
            replace=True,
        )
    for name, (symbols, as_of) in _static_presets().items():
        register_universe(
            UniverseInfo(
                name=name,
                label=name,
                kind="snapshot",
                point_in_time=False,
                target="fintel.market.universe:static_preset",
                n_symbols=len(symbols),
                as_of=as_of.isoformat() if as_of else None,
                description="Frozen list; carries survivorship bias before the snapshot date.",
            ),
            replace=True,
        )


def _index_presets() -> dict[str, tuple[str, str]]:
    from fintel.market.constituents import INDEX_PRESETS

    return INDEX_PRESETS


def _static_presets() -> dict:
    from fintel.market.universe import STATIC_PRESETS

    return STATIC_PRESETS
