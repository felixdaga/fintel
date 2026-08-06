"""Signal transforms — the shape applied to a signal after the strategy's
`signal_fn` and before ensemble/holdings.

Builtin transforms are registered by name in `evaluate/signals.py`. A custom
transform ships as `module:Callable`. These are pure functions on a signal
dict; none of them know what the signal *means*.
"""

from __future__ import annotations

from fintel.models.common import Symbol


def identity(signal: dict[Symbol, float]) -> dict[Symbol, float]:
    """No-op — the signal is used as-is (the single-name default)."""
    return dict(signal)


def rank_range(signal: dict[Symbol, float]) -> dict[Symbol, float]:
    """Map ranks to [-0.5, +0.5] linearly. Ties share the average rank.

    The cross-sectional ranking that makes a portfolio-scope signal comparable
    across names of very different score scales.
    """
    if not signal:
        return {}
    items = sorted(signal.items(), key=lambda kv: kv[1])
    n = len(items)
    # average ranks for ties
    ranks: dict[Symbol, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        avg = (i + j) / 2.0  # 0-indexed average rank
        for k in range(i, j + 1):
            ranks[items[k][0]] = avg
        i = j + 1
    # map rank 0..n-1 to -0.5..+0.5
    return {s: (r / (n - 1) - 0.5) if n > 1 else 0.0 for s, r in ranks.items()}


def zscore(signal: dict[Symbol, float]) -> dict[Symbol, float]:
    """Cross-sectional z-score (mean 0, std 1). Degenerate spread -> zeros."""
    if not signal:
        return {}
    vals = list(signal.values())
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    std = var**0.5
    if std == 0:
        return {s: 0.0 for s in signal}
    return {s: (v - mu) / std for s, v in signal.items()}
