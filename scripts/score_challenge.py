#!/usr/bin/env python3
"""Score a fintel run on the_challenge benchmark.

Computes factor-neutralized residual IC, Newey-West t, and naive
residual-tilt cumulative NAV -- the leaderboard metrics.

Usage:
  uv run python scripts/score_challenge.py runs/<job>/r1 [runs/<job>/r2 ...]
"""
from __future__ import annotations

import json
import sys
from datetime import date as _date_type
from pathlib import Path

import numpy as np

# scripts/ is not a package; add to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from challenge_scoring import (
    FACTORS,
    FACTOR_RUNS,
    GICS_SECTOR_DJIA30,
    HORIZONS,
    djia_members,
    ensemble,
    load_factor_scores,
    load_run_scores,
    price_lookup,
    _zscore_transform,
)
from challenge_metrics import (
    cross_sectional_regression,
    compute_nav,
    ic_summary,
    newey_west_mean,
    _ic,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: score_challenge.py <run_dir> [run_dir ...]")
        sys.exit(1)

    run_dirs = [Path(a) for a in sys.argv[1:]]
    print(f"scoring {len(run_dirs)} run(s)")

    # Delorean: transform each run (zscore), then ensemble-average
    runs = [{d: _zscore_transform(s) for d, s in load_run_scores(d).items()}
            for d in run_dirs]
    agent_scores = ensemble(runs) if len(runs) > 1 else runs[0]
    dates = sorted(agent_scores.keys())
    print(f"  {len(dates)} dates: {dates[0]} -> {dates[-1]}")

    factor_scores = {}
    for fn, stem in FACTORS:
        fdir = FACTOR_RUNS / stem
        if not fdir.is_dir():
            print(f"  WARNING: {stem} missing")
            continue
        factor_scores[fn] = load_factor_scores(fdir)
        print(f"  loaded {fn}: {len(factor_scores[fn])} dates")

    residual_by_date = {}
    agent_by_date = {}
    per_date_reg = {}
    r2s = []
    for d in dates:
        sig = agent_scores[d]
        fac = {fn: factor_scores[fn].get(d, {}) for fn in factor_scores}
        reg = cross_sectional_regression(sig, fac, GICS_SECTOR_DJIA30)
        per_date_reg[d.isoformat()] = reg
        if reg["residuals"]:
            residual_by_date[d] = reg["residuals"]
            agent_by_date[d] = {s: sig[s] for s in reg["residuals"]}
            if reg["r_squared"] is not None:
                r2s.append(reg["r_squared"])
    print(f"  mean R2 = {np.mean(r2s):.4f}" if r2s else "  no R2")

    pl = price_lookup()
    by_horizon = {}
    for h in HORIZONS:
        fwd_all = {}
        for i, d in enumerate(dates):
            if i + h >= len(dates):
                break
            fut = dates[i + h]
            fr = {}
            for s in djia_members(d):
                p0 = pl.price_at(s, d)
                p1 = pl.price_at(s, fut)
                if p0 and p1 and p0 > 0:
                    fr[s] = float(p1) / float(p0) - 1.0
            if fr:
                fwd_all[d] = fr
        scored = [d for d in dates if d in fwd_all and d in residual_by_date]
        ri = [x for x in (_ic(residual_by_date[d], fwd_all[d]) for d in scored) if x is not None]
        ai = [x for x in (_ic(agent_by_date[d], fwd_all[d]) for d in scored) if x is not None]
        nw = newey_west_mean(ri, lag=max(0, h - 1))
        by_horizon[str(h)] = {
            "residual_ic": round(float(np.mean(ri)), 4) if ri else None,
            "residual_mean_ic_nw": nw,
            "residual_ic_summary": ic_summary(ri),
            "agent_ic": round(float(np.mean(ai)), 4) if ai else None,
        }
        print(f"  h={h}: res IC={np.mean(ri):.4f} NWt={nw['t']} (n={len(ri)})")

    nav = compute_nav(dates, residual_by_date, pl, 0.5)
    result = {
        "run_dirs": [str(d) for d in run_dirs],
        "n_runs": len(run_dirs),
        "n_dates": len(dates),
        "dates": [d.isoformat() for d in dates],
        "mean_r_squared": round(float(np.mean(r2s)), 4) if r2s else None,
        "factor_attribution": {
            "control_factors": [fn for fn, _ in FACTORS],
            "sector_controls": True,
            "per_date_regression": per_date_reg,
            "by_horizon": by_horizon,
        },
        "portfolio_construction": {"nav": nav},
    }
    out_path = run_dirs[0].parent / "challenge_score.json"

    def _json_default(o):
        if isinstance(o, _date_type):
            return o.isoformat()
        raise TypeError(f"not serializable: {type(o)}")

    out_path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
    print(f"\nwrote {out_path}")
    h1 = by_horizon.get("1", {})
    print(f"  residual IC (h=1): {h1.get('residual_ic')}")
    print(f"  NW t (h=1):        {h1.get('residual_mean_ic_nw', {}).get('t')}")
    print(f"  agent IC (h=1):    {h1.get('agent_ic')}")
    print(f"  final resid NAV:   {nav['naive_sandbox_residual'][-1]:.4f}")
    print(f"  final bench NAV:   {nav['benchmark'][-1]:.4f}")


if __name__ == "__main__":
    main()
