"""How an agent hands back an answer. One schema, one parser.

The old repo coerced model output into `View` in three places — the MCP server,
the HTTP agent, and each LangGraph desk — with different field names, different
clamping and different silent drops. Two agents could therefore disagree because
their parsers disagreed. This is the only parser.

The schema gives the model an explicit way to decline. Without one, "no views"
means both "I looked and had no opinion" and "something went wrong", which is the
distinction the platform exists to keep.
"""

from __future__ import annotations

from typing import Any

from fintel.models.common import Symbol
from fintel.models.decision import SourceRef, View

SUBMIT_TOOL = "submit_views"

# Tool-facing view fields. Platform ``View`` still carries conviction /
# time_horizon with defaults on parse — they are not solicited from the model
# (unused by single_name signal; wastes reasoning).
VIEW_PROPERTIES: dict[str, Any] = {
    "symbol": {"type": "string", "description": "Ticker this view is about."},
    "score": {
        "type": "number",
        "minimum": -1,
        "maximum": 1,
        "description": "Direction on [-1, +1]: -1 most negative, +1 most positive.",
    },
    "rationale": {"type": "string", "description": "Why, in a few sentences."},
    "key_factors": {
        "type": "array",
        "items": {"type": "string"},
        "description": "The specific drivers behind the score.",
    },
    "sources_cited": {
        "type": "array",
        "description": (
            "The specific data points that moved this view. Each entry is an "
            "object with source_type (the data kind, e.g. prices, fundamentals, "
            "ratios, news, web_search, macro, news_sentiment), source_id (the "
            "date or identifier of that point), and excerpt (the exact value or "
            "short verbatim quote). A bare string is accepted for back-compat and "
            "is treated as a source_type only."
        ),
        "items": {
            "type": "object",
            "properties": {
                "source_type": {"type": "string"},
                "source_id": {"type": "string"},
                "excerpt": {"type": "string"},
            },
        },
    },
}


def submit_schema(symbols: tuple[Symbol, ...]) -> dict:
    listed = ", ".join(symbols) if symbols else "the assigned symbols"
    return {
        "type": "object",
        "properties": {
            "views": {
                "type": "array",
                "description": f"One entry per symbol you are deciding on: {listed}.",
                "items": {
                    "type": "object",
                    "properties": VIEW_PROPERTIES,
                    "required": ["symbol", "score", "rationale"],
                },
            },
            "abstain": {
                "type": "boolean",
                "description": (
                    "True if you are deliberately declining to take a position. "
                    "Declining is a legitimate answer and is recorded as one — "
                    "prefer it to inventing a score you do not believe."
                ),
            },
            "abstain_reason": {"type": "string", "description": "Why you are declining."},
        },
        "required": ["views"],
    }


def submit_description(symbols: tuple[Symbol, ...]) -> str:
    return (
        "Submit your final answer. Call this exactly once, when you are done "
        f"gathering evidence. You are deciding on: {', '.join(symbols) or 'the assigned symbols'}. "
        "If you have no view, set abstain=true with a reason rather than "
        "submitting a score you don't believe."
    )


def clamp(value: Any, low: float, high: float, default: float) -> tuple[float, bool]:
    """Coerce a number into range. Returns (value, was_adjusted).

    Models overshoot — a 1.4 means "very positive", so clamping keeps the
    information that rejecting would throw away. The adjustment is reported so it
    lands in the record rather than passing unnoticed.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default, True
    if number != number:  # NaN
        return default, True
    if number < low:
        return low, True
    if number > high:
        return high, True
    return number, False


def parse_views(
    payload: dict, *, decidable: frozenset[Symbol] | set[Symbol]
) -> tuple[dict[Symbol, View], list[str]]:
    """Turn a `submit_views` payload into views, with notes on what was adjusted.

    Never raises: a malformed entry is dropped with a note, because losing one
    view should not lose the other four.
    """
    notes: list[str] = []
    views: dict[Symbol, View] = {}
    entries = payload.get("views")
    if not isinstance(entries, list):
        return {}, [f"'views' was {type(entries).__name__}, expected a list"]

    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            notes.append(f"view[{index}] was not an object")
            continue
        symbol = str(raw.get("symbol") or "").strip().upper()
        if not symbol:
            notes.append(f"view[{index}] had no symbol")
            continue
        if symbol not in decidable:
            notes.append(f"{symbol} is not decidable by this cell")
            continue
        if symbol in views:
            notes.append(f"{symbol} appeared twice; kept the first")
            continue

        score, score_adjusted = clamp(raw.get("score"), -1.0, 1.0, 0.0)
        if score_adjusted:
            notes.append(f"{symbol}: score {raw.get('score')!r} coerced to {score}")
        conviction, conviction_adjusted = clamp(raw.get("conviction", 0.5), 0.0, 1.0, 0.5)
        if conviction_adjusted:
            notes.append(f"{symbol}: conviction {raw.get('conviction')!r} coerced to {conviction}")

        views[symbol] = View(
            symbol=symbol,
            score=score,
            conviction=conviction,
            time_horizon=str(raw.get("time_horizon") or "quarter"),
            rationale=str(raw.get("rationale") or ""),
            key_factors=[str(f) for f in raw.get("key_factors") or [] if str(f).strip()],
            sources_cited=_parse_sources(raw.get("sources_cited")),
        )
    return views, notes


def _parse_sources(raw: Any) -> list[SourceRef]:
    """Turn a sources_cited entry list into SourceRef objects.

    Accepts both the provenance form (objects with source_type / source_id /
    excerpt) and the legacy form (bare strings, treated as a source_type only).
    Drops anything unusable rather than failing the whole view.
    """
    out: list[SourceRef] = []
    if not isinstance(raw, list):
        return out
    for src in raw:
        if isinstance(src, dict):
            st = str(src.get("source_type") or "").strip()
            sid = str(src.get("source_id") or "").strip()
            if not st:
                continue
            out.append(
                SourceRef(
                    source_type=st,
                    source_id=sid or st,
                    excerpt=(str(src.get("excerpt")).strip() or None)
                    if src.get("excerpt") is not None
                    else None,
                )
            )
        elif isinstance(src, str) and src.strip():
            s = src.strip()
            out.append(SourceRef(source_type=s, source_id=s))
    return out


def abstained(payload: dict) -> str | None:
    """The reason if the model declined, else None."""
    if not payload.get("abstain"):
        return None
    return str(payload.get("abstain_reason") or "").strip() or "no reason given"
