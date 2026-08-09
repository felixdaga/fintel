"""Default opt-in holdings + returns — platform mechanics over the signal.

The strategy owns *what the signal is*; this module owns the *mechanics* of
turning a signal into holdings and returns. It is opt-in via
`ScoringSpec.params["holdings"] = true`. The weight rule is the naive long-only
tilt around equal-weight (delorean's `_active_weights`), and the return is the
weighted forward return with a two-way turnover cost (delorean's
`_cost_adjusted_returns`). No MVO, no factor model — that's post-MVP.

The NAV uses horizon-1 forward returns on the decision-date grid (decision-to-
decision), the same grid the KPI's IC is computed on.
"""

from __future__ import annotations

from datetime import date as Date
from typing import Any

from fintel.market.realized import PriceLookup
from fintel.models.common import Symbol
from fintel.models.evaluate import Signals

DEFAULT_ACTIVE_BUDGET = 0.5
DEFAULT_COST_BPS = 5.0


def active_weights(
    signal: dict[Symbol, float], *, active_budget: float = DEFAULT_ACTIVE_BUDGET
) -> dict[Symbol, float]:
    """Long-only tilt around equal-weight: `w_i = max(0, 1/N + budget·s_i/Σ|s|)`,
    renormalized. The mechanical default; the strategy only tunes the budget."""
    n = len(signal)
    if n == 0:
        return {}
    bmark = 1.0 / n
    total_abs = sum(abs(v) for v in signal.values()) or 1.0
    raw = {s: max(0.0, bmark + active_budget * v / total_abs) for s, v in signal.items()}
    total_w = sum(raw.values()) or 1.0
    return {s: w / total_w for s, w in raw.items()}


def turnover(prev: dict[Symbol, float], curr: dict[Symbol, float]) -> float:
    """Two-way turnover: `Σ|w_i,t − w_i,t−1|` over the union of symbols."""
    keys = set(prev) | set(curr)
    return sum(abs(curr.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)


def _nav_series(
    signal_by_date: dict[Date, dict[Symbol, float]],
    prices: PriceLookup,
    *,
    cost_bps: float,
    active_budget: float,
    as_of: Date | None = None,
) -> dict[str, Any]:
    """Gross + net cumulative NAV for one signal series. Horizon-1 forward
    returns on the decision-date grid; the first rebalance is free.

    When ``as_of`` is after the last decision, the last book is held and marked
    through that date (no terminal turnover cost).
    """
    dates = sorted(signal_by_date)
    grid = list(dates)
    if as_of is not None and dates and as_of > dates[-1]:
        grid = grid + [as_of]
    gross_nav = 1.0
    net_nav = 1.0
    prev_w: dict[Symbol, float] = {}
    gross_series: list[dict] = [{"date": grid[0].isoformat() if grid else None, "nav": 1.0}]
    net_series: list[dict] = [{"date": grid[0].isoformat() if grid else None, "nav": 1.0}]
    turnover_total = 0.0
    for i, d in enumerate(grid[:-1]):
        end = grid[i + 1]
        decision = d if d in signal_by_date else max((x for x in dates if x <= d), default=d)
        w = active_weights(signal_by_date[decision], active_budget=active_budget)
        fwd = {s: r for s, r in ((s, prices.forward_return(s, d, end)) for s in w) if r is not None}
        if not fwd:
            prev_w = w
            continue
        # renormalize weights over names with a realized return
        w_common = {s: w[s] for s in fwd}
        w_sum = sum(w_common.values()) or 1.0
        w_norm = {s: v / w_sum for s, v in w_common.items()}
        r_gross = sum(w_norm[s] * fwd[s] for s in fwd)
        # cost: two-way turnover × bps; skip first rebalance and the hold-to-mark tail
        is_hold_tail = as_of is not None and end == as_of and end not in signal_by_date
        if i > 0 and not is_hold_tail:
            t = turnover(prev_w, w)
            turnover_total += t
            cost = t * (cost_bps / 10000.0)
        else:
            cost = 0.0
        gross_nav *= 1.0 + r_gross
        net_nav *= 1.0 + r_gross - cost
        gross_series.append({"date": end.isoformat(), "nav": round(gross_nav, 6)})
        net_series.append({"date": end.isoformat(), "nav": round(net_nav, 6)})
        prev_w = w
    return {
        "gross": gross_series,
        "net": net_series,
        "turnover_total": round(turnover_total, 4),
        "cost_bps": cost_bps,
        "as_of": as_of.isoformat() if as_of is not None else None,
    }


def build(signals: Signals, prices: PriceLookup, *, params: dict) -> dict[str, Any] | None:
    """Opt-in holdings + returns. Returns None when `params["holdings"]` is not
    truthy, so the report layer can skip the section entirely."""
    if not params.get("holdings"):
        return None
    active_budget = float(params.get("active_budget", DEFAULT_ACTIVE_BUDGET))
    cost_bps = float(params.get("cost_bps", DEFAULT_COST_BPS))
    dates = sorted(signals.ensemble)
    hold_syms = sorted(signals.ensemble[dates[-1]]) if dates else list(signals.universe)
    as_of = prices.latest_bar_date(hold_syms)
    ensemble = _nav_series(
        signals.ensemble, prices, cost_bps=cost_bps, active_budget=active_budget, as_of=as_of
    )
    per_run = [
        _nav_series(sig, prices, cost_bps=cost_bps, active_budget=active_budget, as_of=as_of)
        for sig in signals.per_run
    ]
    return {
        "active_budget": active_budget,
        "cost_bps": cost_bps,
        "as_of": as_of.isoformat() if as_of is not None else None,
        "ensemble": ensemble,
        "per_run": per_run,
    }
