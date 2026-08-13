"""The signal protocol + transforms + ensemble — platform mechanics over a
strategy-defined signal.

The strategy owns **what the signal is** (`ScoringSpec.signal` -> a callable
`signal_fn(views) -> {symbol: float}` per date). This module owns the
mechanics around it: resolving the callable, applying the declared transform,
and ensembling across K repeats by cell-mean.

It never inspects what's inside `signal_fn`. A builtin (`single_name` = the
view's score) and a custom `module:Callable` are the same shape to this code.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date as Date

from fintel.models.common import Symbol
from fintel.models.decision import View
from fintel.models.evaluate import RunData, Signals
from fintel.utils.import_path import resolve  # noqa: F401  (re-exported below)

# A signal function: one date's views -> one date's signal {symbol: value}.
SignalFn = Callable[[dict[Symbol, View]], dict[Symbol, float]]

# A transform: signal dict -> signal dict (e.g. rank, zscore). Applied after the
# signal function, before ensemble/holdings.
Transform = Callable[[dict[Symbol, float]], dict[Symbol, float]]

# Builtin signal functions, resolved by name from `ScoringSpec.signal`.
SIGNAL_BUILTINS: dict[str, str] = {
    "single_name": "fintel.evaluate.signals:single_name_signal",
}

# Builtin transforms, resolved by name from `ScoringSpec.transform`.
TRANSFORM_BUILTINS: dict[str, str] = {
    "single_name": "fintel.evaluate.transforms:identity",
    "identity": "fintel.evaluate.transforms:identity",
    "rank_range": "fintel.evaluate.transforms:rank_range",
    "zscore": "fintel.evaluate.transforms:zscore",
}


def single_name_signal(views: dict[Symbol, View]) -> dict[Symbol, float]:
    """The view's score is THE signal. The default for single-name strategies.

    A missing score (the pack omitted ``score`` from its output contract) maps
    to NaN, not 0.0 — a missing reading is not a neutral one, and the
    evaluation must see the gap.
    """
    return {
        sym: float(v.score) if v.score is not None else float("nan")
        for sym, v in views.items()
    }


def resolve_signal(name: str) -> SignalFn:
    """Resolve a builtin signal name or a `module:Callable` into a callable.

    A signal callable built from `module:Callable` must accept
    `(views: dict[Symbol, View])` and return `dict[Symbol, float]`.
    """
    spec = SIGNAL_BUILTINS.get(name, name)
    if ":" not in spec:
        raise ValueError(
            f"unknown signal {name!r}; expected one of {sorted(SIGNAL_BUILTINS)} "
            "or 'module:Callable'"
        )
    return resolve(spec)  # type: ignore[return-value]


def resolve_transform(name: str) -> Transform:
    """Resolve a builtin transform name or a `module:Callable`."""
    spec = TRANSFORM_BUILTINS.get(name, name)
    if ":" not in spec:
        raise ValueError(
            f"unknown transform {name!r}; expected one of {sorted(TRANSFORM_BUILTINS)} "
            "or 'module:Callable'"
        )
    return resolve(spec)  # type: ignore[return-value]


def _signal_for_run(
    run: RunData, *, signal_fn: SignalFn, transform: Transform
) -> dict[Date, dict[Symbol, float]]:
    """Build one run's signal series: views -> signal_fn -> transform, per date."""
    out: dict[Date, dict[Symbol, float]] = {}
    for d in run.decision_dates:
        views = run.views_by_date.get(d, {})
        if not views:
            continue
        raw = signal_fn(views)
        out[d] = transform(raw)
    return out


def ensemble_signal(
    per_run: list[dict[Date, dict[Symbol, float]]],
) -> dict[Date, dict[Symbol, float]]:
    """Cell-mean across repeats. The ensemble rule is mechanical (mean after the
    strategy's signal+transform), so it is platform-owned, not strategy-owned.

    A symbol absent from one run is excluded from that run's mean rather than
    treated as zero — a missing signal is not a neutral one.
    """
    if not per_run:
        return {}
    dates = sorted({d for run_sig in per_run for d in run_sig})
    out: dict[Date, dict[Symbol, float]] = {}
    for d in dates:
        # symbols present on this date across runs
        per_run_syms = [run_sig.get(d, {}) for run_sig in per_run]
        syms = sorted({s for sig in per_run_syms for s in sig})
        cell: dict[Symbol, float] = {}
        for s in syms:
            vals = [sig[s] for sig in per_run_syms if s in sig]
            if vals:
                cell[s] = sum(vals) / len(vals)
        if cell:
            out[d] = cell
    return out


def build_signals(runs: list[RunData], *, signal: str, transform: str) -> Signals:
    """The full signal build: resolve the strategy's signal + transform, build
    each run's series, then ensemble. Returns `Signals` for the downstream KPI /
    holdings layers."""
    signal_fn = resolve_signal(signal)
    transform = resolve_transform(transform)
    per_run = [_signal_for_run(r, signal_fn=signal_fn, transform=transform) for r in runs]
    ensemble = ensemble_signal(per_run)
    # Universe + dates from the first run that has any (they should all agree).
    universe: list[Symbol] = []
    dates: list[Date] = []
    for r in runs:
        if r.universe:
            universe = r.universe
            dates = r.decision_dates
            break
    return Signals(
        per_run=per_run,
        ensemble=ensemble,
        decision_dates=dates,
        universe=universe,
    )
