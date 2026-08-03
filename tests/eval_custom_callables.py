"""Custom signal + KPI callables for the evaluation extension-seam test.

These are NOT builtins — they live in an importable module and are referenced
from a manifest as `module:Callable`, exactly like a package shipping its own
data source. They demonstrate that a strategy can define *arbitrary* signal
construction (here: clamp + cross-sectional normalize) and *arbitrary* KPI
math (here: mean absolute score), with the platform making no assumption about
how either is computed.
"""

from __future__ import annotations

from datetime import date as Date

from fintel.market.realized import PriceLookup
from fintel.models.common import Symbol
from fintel.models.decision import View


def truncate_normalize_signal(views: dict[Symbol, View]) -> dict[Symbol, float]:
    """A custom signal: clamp scores to [-0.5, +0.5], then cross-sectionally
    normalize to unit sum-of-abs. Whatever the strategy decides 'the signal'
    means; the platform just calls it."""
    out: dict[Symbol, float] = {}
    for sym, v in views.items():
        out[sym] = max(-0.5, min(0.5, float(v.score)))
    total = sum(abs(x) for x in out.values()) or 1.0
    return {s: x / total for s, x in out.items()}


def mean_abs_score_kpi(
    signal_by_date: dict[Date, dict[Symbol, float]],
    prices: PriceLookup,
    horizons: list[int],
    params: dict,
) -> dict:
    """A custom KPI: the mean absolute signal across all (date, symbol) cells.
    Ignores prices entirely — a non-IR KPI, to show the protocol carries any
    metric, not just IC."""
    vals = [abs(v) for sig in signal_by_date.values() for v in sig.values()]
    return {
        "kpi": "mean_abs_score",
        "metric_key": params.get("metric_key", "mean_abs"),
        "mean_abs": round(sum(vals) / len(vals), 6) if vals else None,
        "n_cells": len(vals),
    }
