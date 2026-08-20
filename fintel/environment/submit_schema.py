"""The ``submit_views`` tool JSON Schema + runtime validation.

Lives in the environment layer (not agents) so the MCP server — which is the
transport — can build the same pack-aware tool schema the agents advertise,
without reaching up into agents (which would invert the layer ladder).

Strategy packs may ship ``output_schema.json`` describing **one view item**.
``submit_schema`` wraps that item (or the platform default) as the
``views[]`` element schema for every delivery — MCP tool parameters, LLM tool
specs, and runtime validation — so pack semantics are enforced, not only
pasted into the prompt.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from fintel.models.common import Symbol

SUBMIT_TOOL = "submit_views"

# Tool-facing view fields. Matches the strategy-pack output contract surface
# (symbol / score / rationale / key_factors / sources_cited). Platform ``View``
# may carry optional conviction / time_horizon, but emit never invents them.
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

_ABSTAIN = {
    "type": "boolean",
    "description": (
        "True if you are deliberately declining to take a position. "
        "Declining is a legitimate answer and is recorded as one — "
        "prefer it to inventing a score you do not believe."
    ),
}
_ABSTAIN_REASON = {"type": "string", "description": "Why you are declining."}


def _strip_internal_notes(node: Any) -> Any:
    """Recursively drop ``$comment`` keys so pack-author notes never reach the agent.

    A pack's ``output_schema.json`` may carry ``$comment`` for human authors
    (architecture, field-mapping rationale). Those notes are internal and must
    not leak into the agent-visible tool schema or prompt text.
    """
    if isinstance(node, dict):
        return {k: _strip_internal_notes(v) for k, v in node.items() if k != "$comment"}
    if isinstance(node, list):
        return [_strip_internal_notes(v) for v in node]
    return node


def for_agent_text(text: str | None) -> str:
    """Render a pack schema body for the agent: parsed, internals stripped, re-dumped.

    Drops ``$comment`` and numeric ``minimum``/``maximum`` (see ``submit_schema``).
    Falls back to the raw text if it is not valid JSON, so a malformed schema
    still reaches the agent verbatim rather than vanishing.
    """
    if not text or not str(text).strip():
        return ""
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text.strip()
    if not isinstance(data, dict):
        return text.strip()
    cleaned = _schema_without_numeric_bounds(_strip_internal_notes(data))
    return json.dumps(cleaned, ensure_ascii=False, indent=2)


def item_schema_from_text(text: str | None) -> dict[str, Any] | None:
    """Parse a pack ``output_schema.json`` body into a view-item (or submit) schema.

    Returns None when missing/unusable so callers fall back to the platform
    default item shape. Does not raise — a bad pack schema must not crash a run.
    ``$comment`` keys are stripped so pack-author notes never reach the agent.
    """
    if not text or not str(text).strip():
        return None
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return _strip_internal_notes(data)


def submit_schema(
    symbols: tuple[Symbol, ...],
    *,
    item_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """JSON Schema for the ``submit_views`` tool arguments.

    ``item_schema`` is the optional pack ``output_schema.json`` object:
      * normal case — describes **one view** (``properties`` / ``required`` /
        ``$defs``); wrapped as ``views.items`` with ``$defs`` hoisted to the root
        so ``$ref: #/$defs/...`` keeps resolving;
      * rare case — already submit-shaped (has ``properties.views``); used as the
        base schema with platform abstain fields filled in if missing;
      * ``None`` / unusable — platform default item (``symbol``/``score``/``rationale``).

    Numeric ``minimum``/``maximum`` are stripped before this schema is advertised
    as a tool. Provider constrained-decoding (OpenRouter ``tool_choice`` on
    mimo-v2.5-pro) treats a bounded ``number`` as an integer range: fractional
    positives still come through, every negative collapses to ``-1``. Same pack
    and model via OpenClaw (no provider tool grammar) emit a continuous short
    side. Pack ``output_schema.json`` may still declare bounds for humans;
    ``parse_views`` keeps clamping overshoots.
    """
    listed = ", ".join(symbols) if symbols else "the assigned symbols"

    if item_schema and isinstance(item_schema.get("properties"), dict):
        if "views" in item_schema["properties"]:
            schema = copy.deepcopy(item_schema)
            props = schema.setdefault("properties", {})
            props.setdefault("abstain", copy.deepcopy(_ABSTAIN))
            props.setdefault("abstain_reason", copy.deepcopy(_ABSTAIN_REASON))
            schema.setdefault("required", ["views"])
            schema.setdefault("type", "object")
            views = props.get("views")
            if isinstance(views, dict):
                views.setdefault(
                    "description",
                    f"One entry per symbol you are deciding on: {listed}.",
                )
            return _schema_without_numeric_bounds(schema)

    items, defs = _items_from_pack(item_schema)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "views": {
                "type": "array",
                "description": f"One entry per symbol you are deciding on: {listed}.",
                "items": items,
            },
            "abstain": copy.deepcopy(_ABSTAIN),
            "abstain_reason": copy.deepcopy(_ABSTAIN_REASON),
        },
        "required": ["views"],
    }
    if defs:
        schema["$defs"] = defs
    return _schema_without_numeric_bounds(schema)


def _items_from_pack(
    item_schema: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not item_schema or not isinstance(item_schema.get("properties"), dict):
        return (
            {
                "type": "object",
                "properties": copy.deepcopy(VIEW_PROPERTIES),
                "required": ["symbol", "score", "rationale"],
            },
            None,
        )
    item = {
        k: copy.deepcopy(v)
        for k, v in item_schema.items()
        if k not in ("$schema", "$comment", "$id")
    }
    defs = item.pop("$defs", None) or item.pop("definitions", None)
    if "type" not in item:
        item["type"] = "object"
    return item, defs if isinstance(defs, dict) else None


def validate_submit(payload: Any, schema: dict[str, Any]) -> list[str]:
    """Return human-readable schema violations (empty list = ok).

    Explicit ``abstain=true`` skips item checks — declining is a first-class
    answer. Used by the MCP ``submit_views`` tool (reject before writing
    ``result.json``) and by in-process agents that can re-prompt.

    Numeric ``minimum``/``maximum`` on the advertised tool schema are guidance
    for the model; runtime validation strips them so ``parse_views`` can keep
    clamping overshoots (a 1.4 still means "very positive") instead of hard-
    rejecting a recoverable answer.
    """
    if not isinstance(payload, dict):
        return ["payload must be an object"]
    if payload.get("abstain"):
        return []
    check_schema = _schema_without_numeric_bounds(schema)
    try:
        import jsonschema
        from jsonschema.exceptions import SchemaError
    except ImportError:
        return _validate_submit_lite(payload, check_schema)

    try:
        validator_cls = getattr(jsonschema, "Draft202012Validator", jsonschema.Draft7Validator)
        validator = validator_cls(check_schema)
        errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
    except SchemaError as exc:
        return [f"output schema is invalid: {exc.message}"]
    return [_format_schema_error(err) for err in errors[:8]]


def _schema_without_numeric_bounds(schema: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy ``schema`` with ``minimum``/``maximum`` keys removed."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items() if k not in ("minimum", "maximum")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def _format_schema_error(err: Any) -> str:
    path = ".".join(str(p) for p in err.absolute_path) or "(root)"
    return f"{path}: {err.message}"


def _validate_submit_lite(payload: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Minimal required-field check when jsonschema is not installed."""
    views = payload.get("views")
    if not isinstance(views, list):
        return ["views: must be an array"]
    items = (schema.get("properties") or {}).get("views", {}).get("items") or {}
    required = list(items.get("required") or ["symbol", "score", "rationale"])
    notes: list[str] = []
    for i, raw in enumerate(views):
        if not isinstance(raw, dict):
            notes.append(f"views.{i}: must be an object")
            continue
        for key in required:
            if key not in raw or raw.get(key) in (None, ""):
                notes.append(f"views.{i}: missing required property {key!r}")
    return notes


def submit_description(symbols: tuple[Symbol, ...]) -> str:
    return (
        "Submit your final answer. Call this exactly once, when you are done "
        f"gathering evidence. You are deciding on: {', '.join(symbols) or 'the assigned symbols'}. "
        "Arguments must match this tool's JSON schema (the output schema shown "
        "to you, when present). If rejected, fix the payload and call again. "
        "If you have no view, set abstain=true with a reason rather than "
        "submitting a score you don't believe."
    )


__all__ = [
    "SUBMIT_TOOL",
    "VIEW_PROPERTIES",
    "item_schema_from_text",
    "submit_schema",
    "submit_description",
    "validate_submit",
]
