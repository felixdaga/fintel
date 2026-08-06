"""Char accounting for evidence, by source.

Two numbers per read, both in chars (no tokenizer — see the assessment in the
chat transcript; chars are exact and sufficient for now):

* ``fetched_chars``  — exact chars of the raw payload returned by the source.
  This is the supply-side figure the environment can measure alone, for any
  data tool and any agent, because every read flows through ``DataAccess``.

* ``capped_chars``   — *predicted* chars of the rendered evidence after the
  per-kind render caps (``snippet_max_chars`` for web_search, ``summary_max_chars``
  for news) are applied. It is a prediction, not a measurement: the actual
  rendered string is assembled in the agents module (``FintelEvidence``), which
  the environment cannot see. The prediction mirrors the cap application and
  the framing overhead, so it tracks the real rendered size within a few
  percent; for kinds with no render caps it equals ``fetched_chars``.

The cap *defaults* live in ``catalog.py``; the cap *application shape* lives
here, next to them, so adding a new capped kind is one predictor entry — the
environment stays generic and never hardcodes a source's payload shape.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .catalog import source as _catalog_source  # noqa: F401  (ensures registry ready)


def fetched_chars(data: Any) -> int:
    """Exact chars of the raw fetched payload (JSON, default=str)."""
    if data is None:
        return 0
    try:
        return len(json.dumps(data, default=str))
    except (TypeError, ValueError):
        return len(str(data))


# chars-per-token for English prose / mixed evidence payloads. The standard
# rule of thumb (~4) holds for GPT/Claude/Llama families on prose; JSON/code
# runs closer to 3.5. We use 4 for display/estimation and expose the divisor
# so a caller can pick a more conservative bound (3.5) for overflow guards.
CHARS_PER_TOKEN = 4.0


def estimate_tokens(chars: int, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Approximate tokens from chars. Model-specific in truth (and unknown for
    models with no public tokenizer, e.g. mimo-v2.5-pro via OpenRouter); this
    is an estimate for tracking, not a billing figure."""
    if chars <= 0:
        return 0
    return int(round(chars / chars_per_token))


def predict_capped_chars(kind: str, data: Any, render_caps: dict[str, int]) -> int:
    """Predicted rendered chars after per-kind caps are applied.

    Falls back to ``fetched_chars`` for kinds with no registered predictor or
    no relevant cap declared. ``render_caps`` is the per-kind map from
    ``policy.render_cap_map`` (e.g. ``{"snippet_max_chars": 640}``).
    """
    if data is None:
        return 0
    predictor = _PREDICTORS.get(kind)
    if predictor is None or not render_caps:
        return fetched_chars(data)
    try:
        return int(predictor(data, render_caps))
    except Exception:  # noqa: BLE001 — a predictor bug must not lose the read
        return fetched_chars(data)


# ── per-kind predictors ────────────────────────────────────────────────────
# Each mirrors the cap application in ``fintel.agents.evidence``: find the
# capped text fields in the payload, bound each at its cap, add the small
# framing overhead (title/host/publisher + separators) the renderer adds.

_FRAME_OVERHEAD = 8  # chars of separators/labels per item, approx


def _web_search_results(data: Any) -> list[dict]:
    """Best-effort extraction of the generic result list across cache shapes."""
    if isinstance(data, dict):
        srcs = data.get("sources") or data.get("results") or data
        if isinstance(srcs, dict):
            grd = srcs.get("grounding") or srcs
            if isinstance(grd, dict):
                gen = grd.get("generic") or grd.get("results") or grd.get("web")
                if isinstance(gen, list):
                    return [g for g in gen if isinstance(g, dict)]
        if isinstance(srcs, list):
            return [g for g in srcs if isinstance(g, dict)]
    if isinstance(data, list):
        return [g for g in data if isinstance(g, dict)]
    return []


def _predict_web_search(data: Any, caps: dict[str, int]) -> int:
    cap = int(caps.get("snippet_max_chars", 640))
    total = 0
    for res in _web_search_results(data):
        total += len(str(res.get("title") or ""))
        total += _FRAME_OVERHEAD
        snips = res.get("snippets") or ([res.get("snippet")] if res.get("snippet") else [])
        for sn in snips:
            if sn:
                total += min(len(str(sn)), cap)
    return total


def _news_articles(data: Any) -> list[dict]:
    if isinstance(data, dict):
        for key in ("articles", "results", "data", "items"):
            v = data.get(key)
            if isinstance(v, list):
                return [art for art in v if isinstance(art, dict)]
        if isinstance(data.get("title"), str) and data.get("published_at"):
            return [data]
    if isinstance(data, list):
        return [art for art in data if isinstance(art, dict)]
    return []


def _predict_news(data: Any, caps: dict[str, int]) -> int:
    cap = int(caps.get("summary_max_chars", 640))
    total = 0
    for art in _news_articles(data):
        total += len(str(art.get("title") or ""))
        total += len(str(art.get("publisher") or ""))
        total += _FRAME_OVERHEAD
        summary = art.get("summary") or ""
        if summary:
            total += min(len(str(summary)), cap)
    return total


_PREDICTORS: dict[str, Callable[[Any, dict[str, int]], int]] = {
    "web_search": _predict_web_search,
    "news": _predict_news,
}
