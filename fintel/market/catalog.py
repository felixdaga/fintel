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
    """A query knob a strategy may set in its `[[data]]` block.

    `per_call` is whether an *agent* may also override it mid-run. Most knobs are
    just "how much history", which is fair game. Some define what the number
    means — the trailing window behind a P/E — and letting an agent change those
    would make two readings in one run incomparable.
    """

    name: str
    dtype: DType
    default: Any = None
    description: str = ""
    per_call: bool = True


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
    # What identifies one fetch. The environment turns this into the required
    # argument of the generated tool, so the tool surface follows the catalog
    # instead of being a hand-maintained list that drifts from it.
    subject: Literal["symbol", "query", "none"] = "symbol"

    @property
    def is_computed(self) -> bool:
        return bool(self.derives_from)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    @property
    def call_params(self) -> tuple[Param, ...]:
        return tuple(p for p in self.params if p.per_call)


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
                f"kind {kind!r} is bound twice, to {seen[kind]!r} and {name!r}; one source per kind"
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


def resolve_lookback(binding) -> int | None:
    """The one lookback value for a kind.

    Strategy ``[[data]].lookback_days`` wins; the catalog ``Param.default`` is
    the fallback when the strategy omits it. Returns ``None`` when neither
    declares one — a custom ``module:Callable`` source with no catalog entry and
    no strategy value; callers supply a fallback.

    This is the single source of truth. The factory bakes the resolved value
    into each source instance, so prefetch, probe, tool schemas, the access
    cap, and evidence packs all read the same number from one place.
    """
    v = (binding.params or {}).get("lookback_days")
    if v is not None:
        return int(v)
    if has_source(binding.source):
        for p in source(binding.source).params:
            if p.name == "lookback_days" and p.default is not None:
                return int(p.default)
    return None


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


# ── external macro & news (free, PIT-safe) ───────────────────────────────────
# FRED macroeconomic series (https://fred.stlouisfed.org). True time series, so
# PIT is enforced on the observation date — safe for historical backtests.
MACRO_FIELDS: tuple[Field, ...] = (
    Field("series_id", "text", "FRED series ID"),
    Field("title", "text", "Series title"),
    Field("units", "text", "Units of measurement"),
    Field("frequency", "text", "Reporting frequency"),
    Field("observations", "list", "[{date, value}] oldest→newest, PIT < decision_date"),
    Field("latest_date", "date", "Most recent observation date in window"),
    Field("latest_value", "number", "Most recent observation value"),
    Field("change", "number", "latest - first in window"),
    Field("change_pct", "number", "(latest/first - 1) * 100"),
)

# Alpha Vantage News & Sentiment (https://www.alphavantage.co). Honours
# historical time_from/time_to windows, so PIT-safe for backtests. The value
# over `news` is the per-article sentiment score and per-ticker relevance.
AV_NEWS_FIELDS: tuple[Field, ...] = (
    Field("title", "text", "Headline"),
    Field("url", "text", "Article link"),
    Field("published_at", "date", "Publication date — the PIT stamp"),
    Field("source", "text", "Publisher / source domain"),
    Field("overall_sentiment_score", "number", "Article sentiment score [-1, 1]", "ratio"),
    Field("overall_sentiment_label", "text", "Bearish/Neutral/Bullish"),
    Field(
        "ticker_sentiment",
        "list",
        "[{ticker, relevance_score, ticker_sentiment_score, ticker_sentiment_label}]",
    ),
)

# ── geopol event timeline (curated, PIT-clamped on entry date) ────────────────
EVENT_TIMELINE_FIELDS: tuple[Field, ...] = (
    Field("as_of", "date", "Decision date"),
    Field("lookback_days", "number", "Calendar days of timeline served", "count"),
    Field("n_entries_total", "number", "Total entries in the curated file", "count"),
    Field("n_entries_visible", "number", "Entries within the PIT window", "count"),
    Field("entries", "list", "[{entry_date, body}] oldest→newest, PIT < decision_date"),
)

# ── country health (curated FRED bundle, symmetric across parties) ───────────
COUNTRY_HEALTH_FIELDS: tuple[Field, ...] = (
    Field("as_of", "date", "Decision date"),
    Field("lookback_days", "number", "Calendar days of macro history served", "count"),
    Field(
        "countries",
        "list",
        "[{country, series_id, title, units, observations, latest_date, latest_value, change, change_pct}]",
    ),
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
    out.append(Field("date", "date", "Trading day of the latest snapshot in entries"))
    out.append(
        Field(
            "entries",
            "list",
            "Daily ratio snapshots (oldest→newest), one per trading day in the lookback",
        )
    )
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
            params=(
                Param("lookback_days", "number", 90),
                Param("limit", "number"),
                # Render cap: how much of each news summary the evidence
                # pack shows the agent. Default owned here (markets module);
                # a strategy may override via `[[data]].params`.
                Param(
                    "summary_max_chars",
                    "number",
                    640,
                    "Max chars of each news summary rendered to the agent",
                    per_call=False,
                ),
            ),
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
                # The strategy owns the window: two readings in one run must mean
                # the same thing.
                Param("window_days", "number", 365, "Trailing window length", per_call=False),
                Param(
                    "lookback_days",
                    "number",
                    365,
                    "Calendar days of daily ratio history to serve",
                ),
            ),
            derives_from=("prices", "fundamentals"),
            description=(
                "Daily trailing valuation, profitability and leverage ratios "
                "(Delorean-shaped history: one entry per trading day). Uses the "
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
                # Render cap: how much of each web snippet the evidence pack
                # shows the agent. A fetch-time `max_results` bounds the
                # *number* of results; this bounds the *text* of each. The
                # default lives here (markets module) alongside other data
                # defaults; a strategy may override it via `[[data]].params`
                # like `lookback_days`.
                Param(
                    "snippet_max_chars",
                    "number",
                    640,
                    "Max chars of each web snippet rendered to the agent",
                    per_call=False,
                ),
                # Post-fetch PIT: Brave's freshness window is soft; the response
                # carries sources[url].age = [human, YYYY-MM-DD, relative, ISO].
                # When on, drop results whose age date falls outside the search
                # window. Undated results are kept. Strategies may disable.
                Param(
                    "clamp_by_age",
                    "bool",
                    True,
                    "Post-filter Brave results using sources[url].age vs the "
                    "search window (undated kept)",
                    per_call=False,
                ),
            ),
            requires_env=("BRAVE_API_KEY",),
            subject="query",
            description=(
                "Freshness-windowed web search. Requires `query` (search text). "
                "Use to dig deeper on a specific development, actor, market move, "
                "or claim after reading primary evidence. PIT: provider window "
                "ends decision_date-1, then optional post-clamp on Brave age dates."
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
    # ── external free, PIT-safe vendors (TradingAgents dataflows ports) ──────
    register_source(
        SourceInfo(
            name="fred_macro",
            kind="macro",
            provider="fred",
            target="fintel.market.factory:fred_macro",
            fields=MACRO_FIELDS,
            params=(
                Param("lookback_days", "number", 365, "Calendar days of history to serve"),
                Param(
                    "indicator",
                    "text",
                    None,
                    "Friendly alias (cpi, unemployment, 10y_treasury) or raw FRED series ID",
                ),
            ),
            requires_env=("FRED_API_KEY",),
            subject="none",
            description=(
                "FRED macro time series (rates, yields, inflation, labour, growth). "
                "True dated observations, PIT-clamped on observation date. "
                "Omit `indicator` for the default bundle."
            ),
        ),
        replace=True,
    )
    register_source(
        SourceInfo(
            name="alphavantage_news",
            kind="av_news",
            provider="alpha_vantage",
            target="fintel.market.factory:alphavantage_news",
            fields=AV_NEWS_FIELDS,
            params=(
                Param("lookback_days", "number", 30),
                Param("limit", "number", 50),
            ),
            requires_env=("ALPHA_VANTAGE_API_KEY",),
            subject="symbol",
            description=(
                "Alpha Vantage News & Sentiment: per-ticker articles with per-article "
                "sentiment score and per-ticker relevance. Honours historical windows, "
                "so PIT-safe for backtests."
            ),
        ),
        replace=True,
    )

    # ── geopol event timeline (curated, PIT-clamped, one-off DB) ──────────────
    register_source(
        SourceInfo(
            name="event_timeline",
            kind="event_timeline",
            provider="computed",
            target="fintel.market.factory:event_timeline",
            fields=EVENT_TIMELINE_FIELDS,
            params=(
                Param("lookback_days", "number", 365, "Calendar days of timeline to serve"),
                Param(
                    "event_file",
                    "text",
                    "event.md",
                    "Path to the curated event chronology file (strategy-owned)",
                    per_call=False,
                ),
            ),
            subject="none",
            description=(
                "Chronology of the geopolitical dispute. Call this FIRST "
                "to ground the decision date in what has already happened. No "
                "symbol/party argument — the timeline is shared. Returns dated "
                "entries (entry_date, body) strictly before the decision date. "
                "Read every visible entry before scoring threat or choosing an action."
            ),
        ),
        replace=True,
    )
    # ── country health (curated FRED bundle, symmetric across parties) ───────
    register_source(
        SourceInfo(
            name="country_health",
            kind="country_health",
            provider="fred",
            target="fintel.market.factory:country_health",
            fields=COUNTRY_HEALTH_FIELDS,
            params=(
                Param("lookback_days", "number", 180, "Calendar days of macro history to serve"),
                Param(
                    "country_overrides_file",
                    "text",
                    None,
                    "Optional JSON with extra non-FRED indicators (strategy-owned)",
                    per_call=False,
                ),
            ),
            requires_env=("FRED_API_KEY",),
            subject="none",
            description=(
                "Macroeconomic health for BOTH parties in the dispute (symmetric "
                "access — every cell sees USA and CHN). No symbol/party argument. "
                "Call after the event timeline to check whether the dispute is "
                "already biting the real economy. USA block: rates, dollar, "
                "sentiment, IP, CPI, credit, oil, vol, plus bilateral trade "
                "(IMPCH/EXPCH/BOPGTB) and CNY/USD. CHN block: industrial "
                "production, CPI, discount rate, reserves, merchandise trade, "
                "CNY/USD. Returns countries.{USA,CHN} series with latest_value, "
                "change, and observations."
            ),
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
