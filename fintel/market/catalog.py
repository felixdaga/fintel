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


# ── validation: a strategy's data list must match the catalog ────────────────


def check_bindings(bindings: list) -> list[str]:
    """Every problem with a manifest's `[[data]]` list, as readable messages.

    Returns all findings rather than raising on the first, so one preflight run
    tells you everything to fix. An unregistered `module:Callable` is allowed —
    that's how a package ships its own source — but a bare name that isn't in
    the library is a typo, and a param the source doesn't accept is a setting
    that would otherwise be silently ignored.
    """
    problems: list[str] = []
    seen: dict[str, str] = {}

    for binding in bindings:
        kind, name = binding.kind, binding.source
        if kind in seen:
            problems.append(
                f"kind {kind!r} is bound twice, to {seen[kind]!r} and {name!r}; "
                f"one source per kind"
            )
        seen[kind] = name

        if not has_source(name):
            if ":" not in name:
                by_kind = [s.name for s in sources(kind=kind)]
                hint = f"sources for {kind!r}: {by_kind}" if by_kind else f"kinds: {kinds()}"
                problems.append(f"unknown source {name!r}; {hint}")
            continue

        info = source(name)
        if info.kind != kind:
            problems.append(
                f"source {name!r} serves kind {info.kind!r}, but it's bound to {kind!r}"
            )
        accepted = {p.name for p in info.params}
        for unknown in sorted(set(binding.params) - accepted):
            problems.append(
                f"source {name!r} does not accept param {unknown!r}; accepted: {sorted(accepted)}"
            )

    # Computed kinds need their upstream kinds bound in the same manifest.
    for binding in bindings:
        if not has_source(binding.source):
            continue
        for upstream_kind in source(binding.source).derives_from:
            if upstream_kind not in seen:
                problems.append(
                    f"source {binding.source!r} is computed from {upstream_kind!r}; "
                    f"add a [[data]] block binding kind = {upstream_kind!r}"
                )
    return problems


def required_env(bindings: list) -> list[str]:
    """Credentials the declared bindings need, so preflight can say so once."""
    out: set[str] = set()
    for binding in bindings:
        if has_source(binding.source):
            out.update(source(binding.source).requires_env)
    return sorted(out)


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

FILING_TEXT_FIELDS: tuple[Field, ...] = (
    Field("id", "text", "Accession number, or form:date:section"),
    Field("ticker", "text", "Subject symbol"),
    Field("form_type", "text", "8-K | 10-K"),
    Field("filing_date", "date", "When the filing became public — the PIT stamp"),
    Field("period_end", "date", "Period covered, for 10-K sections"),
    Field("section", "text", "business | risk_factors | 8-K"),
    Field("text", "text", "Extracted filing text"),
)

RATIO_FIELD_DESCRIPTIONS: dict[str, tuple[str, str | None]] = {
    "as_of": ("Decision date the ratios were computed for", None),
    "price": ("Last close strictly before the decision date", "usd"),
    "filing_date": ("Filing date of the trailing anchor", None),
    "period_end": ("Period end of the trailing anchor", None),
    "shares_diluted": ("Weighted-average diluted shares", "shares"),
    "market_cap": ("price x diluted shares", "usd"),
    "enterprise_value": ("Market cap + debt - cash", "usd"),
    "net_debt": ("Total debt - cash", "usd"),
    "book_value_per_share": ("Total equity / diluted shares", "usd"),
    "ebit": ("Operating income; a proxy for EBITDA, no D&A in source", "usd"),
    "pe_diluted": ("Price / trailing diluted EPS, only when EPS > 0", "ratio"),
    "pe_basic": ("Price / trailing basic EPS, only when EPS > 0", "ratio"),
    "p_b": ("Price / book value per share", "ratio"),
    "p_s": ("Market cap / trailing revenue", "ratio"),
    "p_fcf": ("Market cap / trailing free cash flow", "ratio"),
    "p_ocf": ("Market cap / trailing operating cash flow", "ratio"),
    "ev_to_sales": ("Enterprise value / trailing revenue", "ratio"),
    "ev_to_ebit": ("Enterprise value / EBIT — not EV/EBITDA", "ratio"),
    "earnings_yield": ("Trailing diluted EPS / price", "ratio"),
    "fcf_yield": ("Trailing free cash flow / market cap", "ratio"),
    "gross_margin": ("Gross profit / revenue", "ratio"),
    "operating_margin": ("Operating income / revenue", "ratio"),
    "net_margin": ("Net income / revenue", "ratio"),
    "fcf_margin": ("Free cash flow / revenue", "ratio"),
    "roe": ("Trailing net income / total equity", "ratio"),
    "roa": ("Trailing net income / total assets", "ratio"),
    "debt_to_equity": ("Total debt / total equity", "ratio"),
    "debt_to_assets": ("Total debt / total assets", "ratio"),
    "notes": ("Why any field is None, and trailing-window provenance", None),
}

SENTIMENT_FIELDS: tuple[Field, ...] = (
    Field("as_of", "date", "Decision date"),
    Field("series", "list", "Daily {date, score in [-1,1], n}; silent days omitted"),
    Field("n_articles", "number", "Articles in the window", "count"),
    Field("n_scored", "number", "Article-insights carrying a sentiment", "count"),
    Field("mean_score", "number", "Unweighted mean of daily scores, None when empty", "ratio"),
)

WEB_SEARCH_FIELDS: tuple[Field, ...] = (
    Field("query", "text", "Query as sent to the provider"),
    Field("search_window", "text", "{from, to} freshness bounds enforcing PIT"),
    Field("sources", "list", "Provider results"),
)


def _ratio_fields() -> tuple[Field, ...]:
    from fintel.market.data.ratios import RATIO_FIELDS

    out = []
    for name in RATIO_FIELDS:
        description, unit = RATIO_FIELD_DESCRIPTIONS.get(name, ("", None))
        dtype: DType = "list" if name == "notes" else ("date" if name == "as_of" else "number")
        if name in {"filing_date", "period_end"}:
            dtype = "date"
        out.append(Field(name, dtype, description, unit))
    return tuple(out)


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
            name="massive_filing_text",
            kind="filing_text",
            provider="massive",
            target="fintel.market.factory:massive_filing_text",
            fields=FILING_TEXT_FIELDS,
            params=(
                Param("lookback_days", "number", 730),
                Param("forms", "list", None, "Subset of 8-K / 10-K"),
                Param("sections", "list", None, "10-K sections: business, risk_factors"),
                Param("max_chars", "number", None, "Truncate each text to this many chars"),
            ),
            requires_env=("MASSIVE_API_KEY",),
            description="8-K item text and 10-K sections, clamped on filing_date.",
        ),
        replace=True,
    )
    register_source(
        SourceInfo(
            name="valuation_ratios",
            kind="ratios",
            provider="computed",
            target="fintel.market.factory:valuation_ratios",
            fields=_ratio_fields(),
            params=(
                Param("window_days", "number", 365, "Trailing window length"),
                Param("filings_lookback_days", "number", 1460),
            ),
            derives_from=("prices", "fundamentals"),
            description=(
                "Trailing valuation, profitability and leverage ratios. Uses the "
                "annual+delta trailing formula, not a naive 4-quarter sum."
            ),
        ),
        replace=True,
    )
    register_source(
        SourceInfo(
            name="news_sentiment",
            kind="news_sentiment",
            provider="computed",
            target="fintel.market.factory:news_sentiment",
            fields=SENTIMENT_FIELDS,
            params=(Param("lookback_days", "number", 90),),
            derives_from=("news",),
            description="Daily net sentiment from provider insights. Silent days omitted.",
        ),
        replace=True,
    )
    register_source(
        SourceInfo(
            name="web_search",
            kind="web_search",
            provider="brave",
            target="fintel.market.factory:web_search",
            fields=WEB_SEARCH_FIELDS,
            params=(
                Param("lookback_days", "number", 30),
                Param("max_results", "number", 10),
            ),
            requires_env=("BRAVE_API_KEY",),
            description=(
                "Freshness-windowed search. PIT rests on the provider window, since "
                "results carry no date to clamp on afterwards."
            ),
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
