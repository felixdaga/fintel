"""OLS, IC, NW t, and NAV computation — part 2 of challenge_scoring."""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
from scipy import stats

from challenge_scoring import (
    ACTIVE_BUDGET,
    CACHE,
    GICS_SECTOR_DJIA30,
    HORIZONS,
    IC_METHOD,
    djia_members,
)


def zscore_valid(col: list[float | None]) -> np.ndarray:
    arr = np.array([np.nan if x is None else x for x in col], dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) < 2:
        return np.zeros(len(col))
    mu, sd = valid.mean(), valid.std(ddof=1)
    if sd == 0:
        return np.zeros(len(col))
    z = (arr - mu) / sd
    z[np.isnan(z)] = 0.0
    return z


def cross_sectional_regression(
    agent_signal: dict[str, float],
    factor_scores: dict[str, dict[str, float]],
    sector_of: dict[str, str] | None,
) -> dict[str, Any]:
    common = sorted(agent_signal.keys())
    n = len(common)
    if n < 3:
        return {"residuals": {}, "fitted": {}, "r_squared": None, "betas": {}, "n": n}

    factor_names = sorted(factor_scores.keys())
    cols = [
        zscore_valid([factor_scores[fn].get(s) for s in common])
        for fn in factor_names
    ]
    X = np.column_stack(cols) if cols else np.empty((n, 0))

    n_sector = 0
    if sector_of:
        sectors = sorted({sector_of[s] for s in common if s in sector_of})
        if len(sectors) >= 2:
            for sec in sectors[1:]:
                dummy = np.array(
                    [1.0 if sector_of.get(s) == sec else 0.0 for s in common]
                )
                X = np.column_stack([X, dummy]) if X.size else dummy.reshape(-1, 1)
                n_sector += 1

    X = np.column_stack([np.ones(n), X]) if X.size else np.ones((n, 1))
    y = np.array([agent_signal[s] for s in common], dtype=float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    resid = y - y_pred
    thresh = 1e-9 * max(np.max(np.abs(y)), 1.0)
    resid[np.abs(resid) < thresh] = 0.0

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return {
        "residuals": {common[i]: float(resid[i]) for i in range(n)},
        "fitted": {common[i]: float(y_pred[i]) for i in range(n)},
        "r_squared": r_sq,
        "betas": {fn: float(beta[1 + j]) for j, fn in enumerate(factor_names)},
        "n": n,
        "n_sector_controls": n_sector,
    }


def _ic(signal: dict[str, float], fwd: dict[str, float]) -> float | None:
    common = sorted(set(signal) & set(fwd))
    if len(common) < 3:
        return None
    s = [signal[c] for c in common]
    f = [fwd[c] for c in common]
    if IC_METHOD == "pearson":
        r, _ = stats.pearsonr(s, f)
    else:
        r, _ = stats.spearmanr(s, f)
    return float(r) if np.isfinite(r) else None


def newey_west_mean(ics: list[float], lag: int = 0) -> dict[str, Any]:
    n = len(ics)
    if n == 0:
        return {"mean": None, "se": None, "t": None, "n": 0, "lag": 0}
    arr = np.array(ics, dtype=float)
    mean = float(arr.mean())
    L = min(lag, n - 1)
    dev = arr - mean
    gamma0 = float(np.sum(dev ** 2) / n)
    lr = gamma0
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1)
        lr += 2 * w * float(np.sum(dev[l:] * dev[:-l]) / n)
    se = float(np.sqrt(lr / n)) if lr > 0 else 0.0
    return {
        "mean": round(mean, 4),
        "se": round(se, 4),
        "t": round(float(mean / se) if se > 0 else 0.0, 4),
        "n": n,
        "lag": L,
    }


def ic_summary(ics: list[float]) -> dict[str, Any]:
    if not ics:
        return {}
    arr = np.array(ics, dtype=float)
    n = len(arr)
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    return {
        "mean_ic": round(float(arr.mean()), 4),
        "median_ic": round(float(np.median(arr)), 4),
        "ic_std": round(sd, 4),
        "n_periods": n,
        "n_positive": int(np.sum(arr > 0)),
        "positive_fraction": round(int(np.sum(arr > 0)) / n, 2),
        "raw_icir": round(float(arr.mean() / sd) if sd > 0 else 0.0, 4),
    }


def naive_tilt(
    signal: dict[str, float],
    benchmark: dict[str, float],
    active_budget: float,
) -> dict[str, float] | None:
    if not signal:
        return None
    arr = {s: float(v) for s, v in signal.items()}
    dm_m = float(np.mean(list(arr.values())))
    dm = {s: v - dm_m for s, v in arr.items()}
    pos = {s: v for s, v in dm.items() if v > 0}
    neg = {s: v for s, v in dm.items() if v < 0}
    if not pos or not neg:
        return None
    sp, sn = sum(pos.values()), sum(-v for v in neg.values())
    if sp == 0 or sn == 0:
        return None
    active: dict[str, float] = {}
    for s, v in pos.items():
        active[s] = active_budget * v / sp
    for s, v in neg.items():
        active[s] = -active_budget * (-v) / sn
    full = dict(benchmark)
    for s, a in active.items():
        full[s] = full.get(s, 0.0) + a
    return full


def _closest_close(parquet_path: Path, target: date, window: int = 7) -> float | None:
    """Closest available close price within ±window calendar days."""
    if not parquet_path.exists():
        return None
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    if "close" not in df.columns or "date" not in df.columns:
        return None
    best, best_d = None, window + 1
    for _, row in df.iterrows():
        d = row["date"]
        if hasattr(d, "date"):
            d = d.date()
        elif isinstance(d, str):
            from datetime import datetime
            d = datetime.strptime(d[:10], "%Y-%m-%d").date()
        gap = abs((d - target).days)
        if gap <= window and gap < best_d:
            v = float(row["close"])
            if v > 0:
                best, best_d = v, gap
    return best


def benchmark_weights(pl, members: list[str], d: date) -> dict[str, float]:
    """DJIA price-weighted benchmark using close prices (matches Delorean).

    Uses closest close within ±7 calendar days, not open-at-or-before.
    """
    prices_dir = CACHE / "prices"
    prices = {}
    for s in members:
        p = _closest_close(prices_dir / f"{s}.parquet", d)
        if p and p > 0:
            prices[s] = p
    total = sum(prices.values())
    return {s: p / total for s, p in prices.items()} if total > 0 else {}


def compute_nav(
    dates: list[date],
    residual_by_date: dict[date, dict[str, float]],
    pl,
    active_budget: float,
) -> dict[str, Any]:
    nav_d = [dates[0]]
    nav_r = [1.0]
    nav_b = [1.0]
    for i, d in enumerate(dates[:-1]):
        nd = dates[i + 1]
        members = djia_members(d)
        bmark = benchmark_weights(pl, members, d)
        fwd: dict[str, float] = {}
        for s in bmark:
            p0 = pl.price_at(s, d)
            p1 = pl.price_at(s, nd)
            fwd[s] = float(p1) / float(p0) - 1.0 if (p0 and p1 and p0 > 0) else 0.0
        sig = residual_by_date.get(d, {})
        w = naive_tilt(sig, bmark, active_budget) if sig else None
        if w:
            r_r = sum(w.get(s, 0) * fwd.get(s, 0) for s in set(w) | set(fwd))
        else:
            r_r = sum(bmark.get(s, 0) * fwd.get(s, 0) for s in bmark)
        r_b = sum(bmark.get(s, 0) * fwd.get(s, 0) for s in bmark)
        nav_r.append(nav_r[-1] * (1 + r_r))
        nav_b.append(nav_b[-1] * (1 + r_b))
        nav_d.append(nd)
    return {"dates": nav_d, "naive_sandbox_residual": nav_r, "benchmark": nav_b}
