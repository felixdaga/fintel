"""How an agent hands back an answer. One parser.

The old repo coerced model output into `View` in three places — the MCP server,
the HTTP agent, and each LangGraph desk — with different field names, different
clamping and different silent drops. Two agents could therefore disagree because
their parsers disagreed. This is the only parser.

The ``submit_views`` **schema** + runtime validation live in
``fintel.environment.submit_schema`` (environment layer) so the MCP server —
the transport — can build the same pack-aware tool schema the agents
advertise, without reaching up into agents (which would invert the layer
ladder). This module re-exports the schema surface for back-compat with
existing ``from fintel.agents import emit`` call sites, and owns the
**parser** (``parse_views``) that turns a submit payload into ``View`` objects.
"""

from __future__ import annotations

from typing import Any

from fintel.environment.submit_schema import (
    SUBMIT_TOOL,
    VIEW_PROPERTIES,
    item_schema_from_text,
    submit_description,
    submit_schema,
    validate_submit,
)
from fintel.models.common import Symbol
from fintel.models.decision import SourceRef, View

# Platform View fields handled explicitly by parse_views. Anything else in
# the raw payload is a pack-declared native field (e.g. geopol's threat_score /
# action_score / action_level) — passed through so decision.json follows the
# pack output_schema, not just the platform View keys.
_PLATFORM_VIEW_KEYS = frozenset(
    {"symbol", "score", "conviction", "time_horizon", "rationale", "key_factors", "sources_cited"}
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

        raw_score = raw.get("score")
        if raw_score is None:
            # Pack omitted `score` from its output contract — leave None so the
            # signal layer surfaces NaN, not a silent 0.0 neutral reading.
            score: float | None = None
            score_adjusted = False
        else:
            score, score_adjusted = clamp(raw_score, -1.0, 1.0, 0.0)
            if score_adjusted:
                notes.append(f"{symbol}: score {raw_score!r} coerced to {score}")

        conviction: float | None = None
        if "conviction" in raw and raw.get("conviction") is not None:
            conviction, conviction_adjusted = clamp(raw.get("conviction"), 0.0, 1.0, 0.5)
            if conviction_adjusted:
                notes.append(
                    f"{symbol}: conviction {raw.get('conviction')!r} coerced to {conviction}"
                )

        time_horizon: str | None = None
        if raw.get("time_horizon") not in (None, ""):
            time_horizon = str(raw.get("time_horizon"))

        views[symbol] = View(
            symbol=symbol,
            score=score,
            conviction=conviction,
            time_horizon=time_horizon,
            rationale=str(raw.get("rationale") or ""),
            key_factors=[str(f) for f in raw.get("key_factors") or [] if str(f).strip()],
            sources_cited=_parse_sources(raw.get("sources_cited")),
            **_pack_extras(raw),
        )
    return views, notes


def _pack_extras(raw: dict) -> dict:
    return {
        k: v
        for k, v in raw.items()
        if k not in _PLATFORM_VIEW_KEYS and v is not None
    }


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
            relevance = None
            if "relevance" in src and src.get("relevance") is not None:
                relevance, _ = clamp(src.get("relevance"), 0.0, 1.0, 1.0)
            out.append(
                SourceRef(
                    source_type=st,
                    source_id=sid or st,
                    relevance=relevance,
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


__all__ = [
    "SUBMIT_TOOL",
    "VIEW_PROPERTIES",
    "abstained",
    "clamp",
    "item_schema_from_text",
    "parse_views",
    "submit_description",
    "submit_schema",
    "validate_submit",
]
