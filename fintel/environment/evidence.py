"""The same data, pre-rendered, for agents that cannot call tools.

Both paths read through `DataAccess`, so the evidence pack and the tool surface
cannot disagree about PIT or about what a cell may see. In the old code these
were separate implementations — `builder.py` froze a JSON bundle and
`dossier/sections.py` rendered text, each with its own filters and lookbacks.

The rendering rule that matters: missing data is stated, never implied. The old
dossier dropped empty sections silently, so an agent could not distinguish "no
news this quarter" from "the news source was never configured", and its sentiment
renderer defaulted a missing score to `0.0`, which reads as genuine neutrality.
"""

from __future__ import annotations

from typing import Any

from fintel.environment.access import DataAccess, Reading

MISSING = "n/a"


def number(value: Any, *, digits: int = 2) -> str:
    """Compact magnitudes, and an explicit marker for absence.

    `None` renders as `n/a` rather than `0`, which would be indistinguishable
    from a real zero in a ratio or a margin.
    """
    if value is None or isinstance(value, bool):
        return MISSING
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f or f in (float("inf"), float("-inf")):
        return MISSING
    for cut, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(f) >= cut:
            return f"{f / cut:.2f}{suffix}"
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    return f"{f:.{digits}f}".rstrip("0").rstrip(".") or "0"


def _status_line(reading: Reading) -> str:
    """What to say when there's nothing to show. Failure is not absence."""
    if reading.status == "failed":
        return f"UNAVAILABLE — this lookup failed: {reading.detail}. Absence here means nothing."
    if reading.status == "denied":
        return f"NOT PERMITTED — {reading.detail}"
    return "None found in the requested window."


def render_prices(reading: Reading, *, sample: int = 30) -> str:
    bars = reading.data
    if bars is None or not len(bars):
        return _status_line(reading)
    tail = bars.tail(sample)
    lines = [
        f"{str(row.date)[:10]}: {number(row.close)}"
        for row in tail.itertuples()
        if hasattr(row, "close")
    ]
    closes = [c for c in bars["close"].tolist() if c is not None]
    if len(closes) >= 2:
        change = (closes[-1] / closes[0] - 1) * 100 if closes[0] else None
        span = f"last={number(closes[-1])}  over {len(closes)} sessions: "
        span += f"{change:+.1f}%" if change is not None else MISSING
        lines.append(span)
    return "\n".join(lines)


def render_records(reading: Reading, *, limit: int, keys: tuple[str, ...]) -> str:
    records = reading.data
    if not records:
        return _status_line(reading)
    lines = []
    for record in list(reversed(records))[:limit]:
        parts = [str(record.get(k)) for k in keys if record.get(k)]
        lines.append(" — ".join(parts) if parts else str(record)[:200])
    if len(records) > limit:
        lines.append(f"({len(records) - limit} older items not shown)")
    return "\n".join(lines)


def render_ratios(reading: Reading, *, history_points: int = 12) -> str:
    """Latest snapshot plus sparse path — Delorean Toolkit valuation shape."""
    data = reading.data
    if not isinstance(data, dict) or reading.status != "ok":
        return _status_line(reading)
    entries = data.get("entries") if isinstance(data.get("entries"), list) else []
    latest = entries[-1] if entries else data
    keys = (
        "pe_diluted",
        "ev_to_ebit",
        "fcf_yield",
        "p_b",
        "p_s",
        "earnings_yield",
        "net_margin",
        "roe",
        "debt_to_equity",
        "gross_margin",
        "operating_margin",
    )
    kv = "  ".join(f"{k}={number(latest.get(k))}" for k in keys if latest.get(k) is not None)
    lines = [
        f"latest {latest.get('date') or data.get('date') or data.get('as_of')}: "
        + (kv or "No values could be computed.")
    ]
    if len(entries) >= 2:
        n_pts = min(len(entries), history_points)
        if n_pts <= 1:
            sampled = [entries[-1]]
        else:
            idxs = [round(i * (len(entries) - 1) / (n_pts - 1)) for i in range(n_pts)]
            seen: set[int] = set()
            sampled = []
            for i in idxs:
                if i in seen:
                    continue
                seen.add(i)
                sampled.append(entries[i])
        hist_keys = ("pe_diluted", "ev_to_ebit", "fcf_yield", "p_b", "net_margin", "roe")
        lines.append(f"sparse history ({len(sampled)} pts, oldest→newest):")
        for e in sampled:
            parts = " ".join(f"{k}={number(e.get(k))}" for k in hist_keys if e.get(k) is not None)
            lines.append(f"  {e.get('date')}: {parts}" if parts else f"  {e.get('date')}: (n/a)")
    notes = latest.get("notes") or data.get("notes")
    if notes:
        lines.append("notes: " + "; ".join(str(n) for n in notes))
    return "\n".join(lines)


def render_mapping(reading: Reading) -> str:
    data = reading.data
    if not isinstance(data, dict) or reading.status != "ok":
        return _status_line(reading)
    notes = data.get("notes")
    body = [
        f"{k}={number(v)}"
        for k, v in data.items()
        if k not in ("notes", "as_of", "series", "entries", "date") and v is not None
    ]
    out = ["  ".join(body) if body else "No values could be computed."]
    series = data.get("series")
    if isinstance(series, list) and series:
        out.append(
            "series: "
            + ", ".join(f"{e.get('date')}={number(e.get('score'))}" for e in series[-14:])
        )
    if notes:
        out.append("notes: " + "; ".join(str(n) for n in notes))
    return "\n".join(out)


# How each kind is rendered, and what to ask for. A kind with no renderer here
# still appears, rendered generically — better than being dropped.
RENDERERS: dict[str, dict] = {
    "prices": {"title": "Prices (daily close, oldest first)", "render": render_prices},
    "fundamentals": {
        "title": "Fundamentals (most recent filings first)",
        "render": lambda r: render_records(
            r, limit=6, keys=("filing_date", "period_end", "timeframe")
        ),
    },
    "ratios": {"title": "Valuation ratios (trailing)", "render": render_ratios},
    "news": {
        "title": "News (newest first)",
        "render": lambda r: render_records(r, limit=12, keys=("published_at", "title")),
    },
    "news_sentiment": {"title": "News sentiment (daily net score)", "render": render_mapping},
    "filing_text": {
        "title": "Filing text",
        "render": lambda r: render_records(
            r, limit=4, keys=("filing_date", "form_type", "section")
        ),
    },
}


# Reading order: the hard numbers before the commentary, so an agent forms a
# view from the filing and the price before it meets a headline about them.
# Kinds not listed here follow, in declaration order.
READING_ORDER: tuple[str, ...] = (
    "prices",
    "fundamentals",
    "ratios",
    "filing_text",
    "news",
    "news_sentiment",
    "web_search",
)


def ordered(kinds: tuple[str, ...]) -> tuple[str, ...]:
    rank = {kind: i for i, kind in enumerate(READING_ORDER)}
    return tuple(sorted(kinds, key=lambda k: (rank.get(k, len(rank)), k)))


def render_generic(reading: Reading) -> str:
    data = reading.data
    if isinstance(data, dict):
        return render_mapping(reading)
    if isinstance(data, list):
        return render_records(reading, limit=10, keys=())
    return _status_line(reading) if data is None else str(data)[:2000]


def build(access: DataAccess, *, symbol: str | None = None, kinds: tuple[str, ...] = ()) -> str:
    """One text pack for a cell.

    Every declared kind gets a section, including the ones that came back empty
    or failed — a silently omitted section is indistinguishable from a kind the
    strategy never asked for.
    """
    cell = access.cell
    subject = symbol or (cell.symbols[0] if not cell.is_portfolio else None)
    wanted = ordered(kinds or access.kinds)

    header = [
        f"# Evidence — {subject or 'portfolio'} as of {cell.decision_date.isoformat()}",
        "",
        "Everything below was publicly available strictly before the date above. "
        "Nothing dated on or after it is included.",
    ]
    sections: list[str] = []
    for kind in wanted:
        # web_search needs a query, so it can't be pre-rendered; it is a tool or
        # nothing, and saying so beats an empty section.
        if kind == "web_search":
            sections.append(
                "## Web search\nAvailable on request only — it needs a query. Not pre-rendered."
            )
            continue
        query: dict = {"symbol": subject} if subject else {}
        reading = access.read(kind, **query)
        spec = RENDERERS.get(kind)
        title = spec["title"] if spec else kind.replace("_", " ").title()
        body = spec["render"](reading) if spec else render_generic(reading)
        sections.append(f"## {title}\n{body}")

    return "\n".join(header) + "\n\n" + "\n\n".join(sections) + "\n"
