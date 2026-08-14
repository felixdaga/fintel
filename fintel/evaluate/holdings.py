"""Opt-in holdings + returns — platform mechanics over the signal.

The strategy owns *what the signal is*; this module owns the *mechanics* of
turning a signal into holdings and returns. Opt-in via
`ScoringSpec.params["holdings"] = true`.

Default books (when holdings is on):

- ``pw`` — price-weighted (DJIA-style) benchmark
- ``sw_long`` / ``ew_long`` at each ``long_thresholds`` entry (empty → cash)
- ``naive_tilt`` — long-only tilt around equal-weight (the original MVP book)
- ``mvo`` — mean-variance overlay around PW (cvxpy; falls back to naive tilt)

NAV uses horizon-1 forward returns on the decision-date grid (decision-to-
decision), the same grid the KPI's IC is computed on. Sharpe / IR / vol are
annualized with ``params["ppy"]``; total return and max DD stay sample.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date as Date
from typing import Any

from fintel.market.realized import PriceLookup
from fintel.models.common import Symbol
from fintel.models.evaluate import Signals

DEFAULT_ACTIVE_BUDGET = 0.5
DEFAULT_COST_BPS = 5.0
DEFAULT_LONG_THRESHOLDS = (0.0, 0.3)
DEFAULT_BOOKS = ("pw", "sw_long", "ew_long", "naive_tilt", "mvo")
DEFAULT_RISK_AVERSION = 2.0
DEFAULT_COV_LOOKBACK = 60

WeightsFn = Callable[[Date], dict[Symbol, float]]


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


def _thr_key(prefix: str, threshold: float) -> str:
    return f"{prefix}_{threshold:.1f}"


def _price_weighted(signal: dict[Symbol, float], prices: PriceLookup, d: Date) -> dict[Symbol, float]:
    price_at = getattr(prices, "price_at", None)
    if price_at is None:
        n = len(signal)
        return {s: 1.0 / n for s in signal} if n else {}
    px: dict[Symbol, float] = {}
    for s in signal:
        p = price_at(s, d)
        if p is not None and p > 0:
            px[s] = float(p)
    total = sum(px.values())
    return {s: p / total for s, p in px.items()} if total > 0 else {}


def _longs(signal: dict[Symbol, float], threshold: float) -> dict[Symbol, float]:
    return {s: v for s, v in signal.items() if v > threshold}


def _score_weighted_long(signal: dict[Symbol, float], threshold: float) -> dict[Symbol, float]:
    longs = _longs(signal, threshold)
    if not longs:
        return {}
    total = sum(longs.values())
    if total > 0:
        return {s: v / total for s, v in longs.items()}
    w = 1.0 / len(longs)
    return {s: w for s in longs}


def _equal_weight_long(signal: dict[Symbol, float], threshold: float) -> dict[Symbol, float]:
    longs = _longs(signal, threshold)
    if not longs:
        return {}
    w = 1.0 / len(longs)
    return {s: w for s in longs}


def _cov_matrix(
    prices: PriceLookup, universe: list[Symbol], d: Date, *, lookback: int
):
    """Daily-return covariance annualized ×252 from cached bars on or before `d`."""
    store = getattr(prices, "store", None)
    bars_fn = getattr(store, "bars_on_or_before", None) if store is not None else None
    if bars_fn is None:
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    field = getattr(prices, "price_field", "open")
    cols: dict[Symbol, Any] = {}
    for s in universe:
        bars = bars_fn(s, d)
        if bars is None or getattr(bars, "empty", True) or field not in bars.columns:
            continue
        tail = bars.tail(lookback + 1)
        if "date" in tail.columns:
            ser = tail.set_index("date")[field]
        else:
            ser = tail[field]
        if len(ser) >= 20:
            cols[s] = ser
    if len(cols) < 2:
        return None
    pdf = pd.DataFrame(cols).ffill().dropna(axis=1, how="all")
    rets = pdf.pct_change().dropna()
    if len(rets) < 5 or rets.shape[1] < 2:
        return None
    return rets.cov() * 252.0


def _mvo_weights(
    signal: dict[Symbol, float],
    prices: PriceLookup,
    d: Date,
    *,
    active_budget: float,
    risk_aversion: float,
    lookback: int,
    fallback: dict[Symbol, float],
) -> tuple[dict[Symbol, float], str]:
    """PW-centered mean-variance overlay. Returns (weights, status)."""
    try:
        import cvxpy as cp
        import numpy as np
    except ImportError:
        return fallback, "fallback:no_cvxpy"

    wbm = _price_weighted(signal, prices, d)
    universe = sorted(signal)
    sigma = _cov_matrix(prices, universe, d, lookback=lookback)
    if sigma is None:
        return fallback, "fallback:no_cov"
    common = sorted(set(signal) & set(sigma.index) & set(wbm))
    if len(common) < 2:
        return wbm or fallback, "fallback:no_overlap"
    mu = np.array([signal[s] for s in common], dtype=float)
    mu = mu - mu.mean()
    wbm_v = np.array([wbm[s] for s in common], dtype=float)
    sig = sigma.loc[common, common].values.astype(float)
    n = len(common)
    sig = sig + 1e-8 * np.eye(n)
    cap = max(active_budget / n, 0.01)
    w = cp.Variable(n)
    active = w - wbm_v
    obj = cp.Maximize(mu @ w - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(sig)))
    cons = [
        cp.sum(w) == float(wbm_v.sum()),
        cp.sum(active) == 0,
        cp.abs(active) <= cap,
        cp.norm1(active) <= 2.0 * active_budget,
    ]
    prob = cp.Problem(obj, cons)
    status = "solver"
    for solver in ("OSQP", None):
        try:
            if solver is None:
                prob.solve(verbose=False)
            else:
                prob.solve(solver=solver, verbose=False)
        except Exception:  # noqa: BLE001 — any solver failure is a fallback
            continue
        if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
            status = str(prob.status)
            break
    else:
        return fallback, f"fallback:solver:{getattr(prob, 'status', 'unknown')}"

    w_full = np.where(np.abs(w.value) < 1e-9, 0.0, np.asarray(w.value).flatten())
    s = float(w_full.sum())
    target = float(wbm_v.sum())
    if abs(s) > 1e-9 and abs(s - target) > 1e-9:
        w_full = w_full * (target / s)
    out = dict(wbm)
    for i, sym in enumerate(common):
        out[sym] = float(w_full[i])
    return out, status


def _nav_series(
    weights_fn: WeightsFn,
    dates: list[Date],
    prices: PriceLookup,
    *,
    cost_bps: float,
    as_of: Date | None = None,
) -> dict[str, Any]:
    """Gross + net cumulative NAV. Horizon-1 forward returns on the grid;
    the first rebalance is free. Empty weights are cash (r=0).

    When ``as_of`` is after the last decision, the last book is held and marked
    through that date (no terminal turnover cost).
    """
    grid = list(dates)
    if as_of is not None and dates and as_of > dates[-1]:
        grid = grid + [as_of]
    date_set = set(dates)
    gross_nav = 1.0
    net_nav = 1.0
    prev_w: dict[Symbol, float] = {}
    start = grid[0].isoformat() if grid else None
    gross_series: list[dict] = [{"date": start, "nav": 1.0}]
    net_series: list[dict] = [{"date": start, "nav": 1.0}]
    turnover_total = 0.0
    n_rebalances = 0
    for i, d in enumerate(grid[:-1]):
        end = grid[i + 1]
        decision = d if d in date_set else max((x for x in dates if x <= d), default=d)
        w = weights_fn(decision) or {}
        if w:
            fwd = {
                s: r
                for s, r in ((s, prices.forward_return(s, d, end)) for s in w)
                if r is not None
            }
            if fwd:
                w_common = {s: w[s] for s in fwd}
                w_sum = sum(w_common.values()) or 1.0
                r_gross = sum((w_common[s] / w_sum) * fwd[s] for s in fwd)
            else:
                r_gross = 0.0
        else:
            r_gross = 0.0  # empty long book → cash
        is_hold_tail = as_of is not None and end == as_of and end not in dates
        if i > 0 and not is_hold_tail:
            t = turnover(prev_w, w)
            turnover_total += t
            n_rebalances += 1
            cost = t * (cost_bps / 10000.0)
        else:
            cost = 0.0
        gross_nav *= 1.0 + r_gross
        net_nav *= 1.0 + r_gross - cost
        gross_series.append({"date": end.isoformat(), "nav": round(gross_nav, 6)})
        net_series.append({"date": end.isoformat(), "nav": round(net_nav, 6)})
        prev_w = w
    avg_turnover = (turnover_total / n_rebalances) if n_rebalances else 0.0
    return {
        "gross": gross_series,
        "net": net_series,
        "turnover_total": round(turnover_total, 4),
        "avg_turnover": round(avg_turnover, 6),
        "cost_bps": cost_bps,
        "as_of": as_of.isoformat() if as_of is not None else None,
    }


def _period_returns(series: list[dict]) -> list[float]:
    out: list[float] = []
    for i in range(1, len(series)):
        prev, cur = series[i - 1]["nav"], series[i]["nav"]
        out.append((cur / prev - 1.0) if prev else 0.0)
    return out


def _max_drawdown(navs: list[float]) -> float:
    peak = -1e300
    mdd = 0.0
    for v in navs:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1.0)
    return mdd


def _metrics(
    book: dict[str, Any],
    *,
    bench_rets: list[float] | None,
    ppy: float,
    years: float,
    is_bench: bool,
) -> dict[str, Any]:
    net = book.get("net") or []
    gross = book.get("gross") or []
    navs = [p["nav"] for p in net]
    rets = _period_returns(net)
    gnav = gross[-1]["nav"] if gross else 1.0
    nnav = navs[-1] if navs else 1.0
    total = nnav - 1.0
    n = len(rets)
    vol = (sum((r - sum(rets) / n) ** 2 for r in rets) / (n - 1)) ** 0.5 if n >= 2 else 0.0
    mean_r = (sum(rets) / n) if n else 0.0
    sharpe = (mean_r / vol) if vol > 1e-12 else None
    ir = None
    if not is_bench and bench_rets is not None and n >= 2:
        k = min(n, len(bench_rets))
        active = [rets[i] - bench_rets[i] for i in range(k)]
        te = (
            (sum((a - sum(active) / k) ** 2 for a in active) / (k - 1)) ** 0.5 if k >= 2 else 0.0
        )
        ir = (sum(active) / k / te) if te > 1e-12 else None
    sqrt_ppy = ppy**0.5
    ann_ret = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 and years > 0 else None
    ann_cost = (gnav / nnav) ** (1.0 / years) - 1.0 if nnav > 0 and years > 0 else None
    avg_to = float(book.get("avg_turnover") or 0.0)
    return {
        "total": round(total, 6),
        "ann_ret": round(ann_ret, 6) if ann_ret is not None else None,
        "ann_vol": round(vol * sqrt_ppy, 6),
        "max_dd": round(_max_drawdown(navs), 6),
        "ann_sharpe": round(sharpe * sqrt_ppy, 6) if sharpe is not None else None,
        "ann_ir": round(ir * sqrt_ppy, 6) if ir is not None else None,
        "ann_turn": round(avg_to * ppy, 6),
        "ann_cost": round(ann_cost, 6) if ann_cost is not None else None,
        "n_periods": n,
        "gross_nav": round(gnav, 6),
        "net_nav": round(nnav, 6),
    }


def _parse_thresholds(params: dict) -> list[float]:
    raw = params.get("long_thresholds", DEFAULT_LONG_THRESHOLDS)
    try:
        return [float(t) for t in raw]
    except (TypeError, ValueError):
        return list(DEFAULT_LONG_THRESHOLDS)


def _parse_books(params: dict) -> list[str]:
    raw = params.get("books", DEFAULT_BOOKS)
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    for b in raw:
        b = str(b).strip()
        if b and b not in out:
            out.append(b)
    return out or list(DEFAULT_BOOKS)


def _l1(a: dict[Symbol, float], b: dict[Symbol, float]) -> float:
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def build(signals: Signals, prices: PriceLookup, *, params: dict) -> dict[str, Any] | None:
    """Opt-in holdings + returns. Returns None when `params["holdings"]` is not
    truthy, so the report layer can skip the section entirely.

    ``ensemble`` remains the naive-tilt NAV (backward compatible). ``strategies``
    holds every requested book plus annualized metrics.
    """
    if not params.get("holdings"):
        return None
    active_budget = float(params.get("active_budget", DEFAULT_ACTIVE_BUDGET))
    cost_bps = float(params.get("cost_bps", DEFAULT_COST_BPS))
    risk_aversion = float(params.get("risk_aversion", DEFAULT_RISK_AVERSION))
    lookback = int(params.get("cov_lookback", DEFAULT_COV_LOOKBACK))
    thresholds = _parse_thresholds(params)
    books = _parse_books(params)
    ppy = float(params.get("ppy") or 12.0)
    cadence = params.get("cadence") or {}
    strategy_name = str(params.get("strategy_name") or "")
    dates = sorted(signals.ensemble)
    hold_syms = sorted(signals.ensemble[dates[-1]]) if dates else list(signals.universe)
    as_of = prices.latest_bar_date(hold_syms)
    ensemble_sig = signals.ensemble

    def _sig(d: Date) -> dict[Symbol, float]:
        return ensemble_sig.get(d, {})

    naive_cache: dict[Date, dict[Symbol, float]] = {}

    def naive_fn(d: Date) -> dict[Symbol, float]:
        if d not in naive_cache:
            naive_cache[d] = active_weights(_sig(d), active_budget=active_budget)
        return naive_cache[d]

    mvo_status: dict[str, Any] = {
        "available": True,
        "n_dates": 0,
        "n_solved": 0,
        "n_fallback": 0,
        "reasons": {},
        "distinct_from_naive": None,
    }
    mvo_cache: dict[Date, dict[Symbol, float]] = {}

    def mvo_fn(d: Date) -> dict[Symbol, float]:
        if d not in mvo_cache:
            fb = naive_fn(d)
            w, st = _mvo_weights(
                _sig(d),
                prices,
                d,
                active_budget=active_budget,
                risk_aversion=risk_aversion,
                lookback=lookback,
                fallback=fb,
            )
            mvo_cache[d] = w
            mvo_status["n_dates"] += 1
            reasons: dict[str, int] = mvo_status["reasons"]
            reasons[st] = reasons.get(st, 0) + 1
            if st.startswith("fallback"):
                mvo_status["n_fallback"] += 1
            else:
                mvo_status["n_solved"] += 1
            if mvo_status["distinct_from_naive"] is not True:
                mvo_status["distinct_from_naive"] = _l1(w, fb) > 1e-6
        return mvo_cache[d]

    weight_fns: dict[str, WeightsFn] = {}
    labels: dict[str, str] = {}
    pw_label = "DJIA PW" if "djia" in strategy_name.lower() else "Price-weighted"
    if "pw" in books:
        weight_fns["pw"] = lambda d: _price_weighted(_sig(d), prices, d)
        labels["pw"] = pw_label
    if "sw_long" in books:
        for t in thresholds:
            key = _thr_key("sw", t)
            weight_fns[key] = lambda d, t=t: _score_weighted_long(_sig(d), t)
            labels[key] = f"SW Long >{t:.1f}"
    if "ew_long" in books:
        for t in thresholds:
            key = _thr_key("ew", t)
            weight_fns[key] = lambda d, t=t: _equal_weight_long(_sig(d), t)
            labels[key] = f"EW Long >{t:.1f}"
    if "naive_tilt" in books:
        weight_fns["naive_tilt"] = naive_fn
        labels["naive_tilt"] = "Naive tilt"
    if "mvo" in books:
        weight_fns["mvo"] = mvo_fn
        labels["mvo"] = "MVO"

    # Always compute naive tilt for the backward-compat `ensemble` key.
    ensemble = _nav_series(naive_fn, dates, prices, cost_bps=cost_bps, as_of=as_of)
    strategies: dict[str, Any] = {}
    for key, fn in weight_fns.items():
        strategies[key] = _nav_series(fn, dates, prices, cost_bps=cost_bps, as_of=as_of)
        strategies[key]["label"] = labels.get(key, key)

    nav_dates = [p["date"] for p in ensemble.get("net") or [] if p.get("date")]
    years = 0.0
    if len(nav_dates) >= 2:
        first, last = Date.fromisoformat(nav_dates[0]), Date.fromisoformat(nav_dates[-1])
        years = max((last - first).days / 365.25, 1e-9)
    bench_rets = _period_returns((strategies.get("pw") or {}).get("net") or [])
    metrics: dict[str, Any] = {}
    for key, book in strategies.items():
        metrics[key] = _metrics(
            book,
            bench_rets=bench_rets or None,
            ppy=ppy,
            years=years or 1e-9,
            is_bench=(key == "pw"),
        )
        metrics[key]["label"] = book.get("label", key)

    if "mvo" not in books:
        mvo_status = {"available": False, "note": "not requested"}
    elif mvo_status["n_dates"] == 0:
        mvo_status["available"] = False
        mvo_status["note"] = "no decision dates"

    per_run = [
        _nav_series(
            lambda d, sig=sig: active_weights(sig.get(d, {}), active_budget=active_budget),
            sorted(sig),
            prices,
            cost_bps=cost_bps,
            as_of=as_of,
        )
        for sig in signals.per_run
    ]
    return {
        "active_budget": active_budget,
        "cost_bps": cost_bps,
        "as_of": as_of.isoformat() if as_of is not None else None,
        "ppy": ppy,
        "cadence": cadence.get("cadence") if isinstance(cadence, dict) else cadence,
        "years": round(years, 6) if years else None,
        "long_thresholds": thresholds,
        "books": books,
        "labels": labels,
        "ensemble": ensemble,
        "strategies": strategies,
        "metrics": metrics,
        "mvo": mvo_status,
        "per_run": per_run,
    }
