"""Pluggable holdings rules: turn scores → weights.

Each rule takes a signal dict ``{symbol: score}`` and returns a weight dict
``{symbol: weight}`` (summing to 1, or empty if no names qualify). The deploy
config selects which rule to use via ``[holdings].rule``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fintel.models.common import Symbol

WeightFn = Callable[[dict[Symbol, float], dict[str, Any]], dict[Symbol, float]]

_REGISTRY: dict[str, WeightFn] = {}


def register(name: str) -> Callable[[WeightFn], WeightFn]:
    def decorator(fn: WeightFn) -> WeightFn:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_rule(name: str) -> WeightFn:
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown holdings rule {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


@register("ew_long_threshold")
def ew_long_threshold(
    signal: dict[Symbol, float], params: dict[str, Any]
) -> dict[Symbol, float]:
    """Equal-weight across names scoring above a threshold."""
    threshold = float(params.get("threshold", 0.3))
    longs = {s: v for s, v in signal.items() if v > threshold}
    if not longs:
        return {}
    w = 1.0 / len(longs)
    return {s: w for s in longs}


@register("score_weighted_long")
def score_weighted_long(
    signal: dict[Symbol, float], params: dict[str, Any]
) -> dict[Symbol, float]:
    """Weight proportional to score, across names above a threshold."""
    threshold = float(params.get("threshold", 0.3))
    longs = {s: v for s, v in signal.items() if v > threshold}
    if not longs:
        return {}
    total = sum(longs.values())
    return {s: v / total for s, v in longs.items()} if total > 0 else {s: 1.0 / len(longs) for s in longs}


@register("naive_tilt")
def naive_tilt(
    signal: dict[Symbol, float], params: dict[str, Any]
) -> dict[Symbol, float]:
    """Long-only tilt around equal-weight: max(0, 1/N + budget·s/Σ|s|), renorm."""
    budget = float(params.get("active_budget", 0.5))
    n = len(signal)
    if n == 0:
        return {}
    bmark = 1.0 / n
    total_abs = sum(abs(v) for v in signal.values()) or 1.0
    raw = {s: max(0.0, bmark + budget * v / total_abs) for s, v in signal.items()}
    total_w = sum(raw.values()) or 1.0
    return {s: w / total_w for s, w in raw.items()}


@register("price_weighted")
def price_weighted(
    signal: dict[Symbol, float], params: dict[str, Any]
) -> dict[Symbol, float]:
    """Price-weighted (like DJIA): weight ∝ price. Uses prices from params."""
    prices: dict[Symbol, float] = params.get("prices", {})
    px = {s: p for s, p in prices.items() if s in signal and p and p > 0}
    total = sum(px.values())
    return {s: p / total for s, p in px.items()} if total > 0 else {}
