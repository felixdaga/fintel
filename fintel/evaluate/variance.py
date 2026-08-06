"""Layer 2 — agent output (decision) stochasticity across K repeats.

Always on. Measures dispersion of the agent's *decisions* (scores, signs,
ranks) across the K repeats, independent of any price/return. This is the
"did the agent say the same thing twice?" layer — distinct from L1 (did it
*do* the same thing) and from the KPI (was what it said *right*).

Operates on the per-run signal series (post-transform), so it sees what the
holdings/KPI layers see. No prices.
"""

from __future__ import annotations

from datetime import date as Date
from statistics import mean, pstdev
from typing import Any

from fintel.models.common import Symbol


def _stdev(vals: list[float]) -> float:
    return pstdev(vals) if len(vals) >= 2 else 0.0


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation of two equal-length series. Pure python."""
    n = len(a)
    if n < 2:
        return None

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx = ranks(a)
    ry = ranks(b)
    mx = sum(rx) / n
    my = sum(ry) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    sxx = sum((x - mx) ** 2 for x in rx)
    syy = sum((y - my) ** 2 for y in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def analyse(per_run: list[dict[Date, dict[Symbol, float]]]) -> dict[str, Any]:
    """Per-cell score/sign/rank dispersion across K repeats + a summary.

    `per_run` is the post-transform signal series (one entry per repeat), the
    same object the ensemble/KPI layers consume.
    """
    if len(per_run) < 2:
        return {
            "available": False,
            "per_cell": {},
            "per_date": {},
            "summary": {"note": "need >= 2 runs for cross-run variance"},
        }

    # (date, symbol) -> list of signal values across runs
    cells: dict[tuple[Date, Symbol], list[float]] = {}
    for run_sig in per_run:
        for d, sig in run_sig.items():
            for sym, val in sig.items():
                cells.setdefault((d, sym), []).append(val)

    per_cell: dict[str, dict] = {}
    for (d, sym), vals in sorted(cells.items()):
        if len(vals) < 2:
            continue
        scores = [float(v) for v in vals]
        signs = [_sign(v) for v in scores]
        sign_agreement = len(set(signs)) == 1
        per_cell[f"{d.isoformat()}|{sym}"] = {
            "n_runs": len(scores),
            "score_mean": round(mean(scores), 4),
            "score_std": round(_stdev(scores), 4),
            "score_range": (round(min(scores), 4), round(max(scores), 4)),
            "sign_agreement": sign_agreement,
            "signs": signs,
        }

    # per-date mean pairwise rank correlation across runs
    per_date: dict[str, Any] = {}
    dates = sorted({d for run_sig in per_run for d in run_sig})
    for d in dates:
        per_run_syms = [run_sig.get(d, {}) for run_sig in per_run]
        # pairwise spearman over the common symbols
        corrs: list[float] = []
        for i in range(len(per_run_syms)):
            for j in range(i + 1, len(per_run_syms)):
                common = sorted(set(per_run_syms[i]) & set(per_run_syms[j]))
                if len(common) < 2:
                    continue
                c = _spearman(
                    [per_run_syms[i][s] for s in common],
                    [per_run_syms[j][s] for s in common],
                )
                if c is not None:
                    corrs.append(c)
        if corrs:
            per_date[d.isoformat()] = {
                "mean_rank_corr": round(mean(corrs), 4),
                "n_pairs": len(corrs),
            }

    if per_cell:
        score_stds = [v["score_std"] for v in per_cell.values()]
        agreements = [v["sign_agreement"] for v in per_cell.values()]
        summary = {
            "available": True,
            "n_cells": len(per_cell),
            "mean_score_std": round(mean(score_stds), 4) if score_stds else 0.0,
            "min_sign_agreement": all(agreements),
            "mean_rank_corr": (
                round(mean([v["mean_rank_corr"] for v in per_date.values()]), 4)
                if per_date
                else None
            ),
        }
    else:
        summary = {"available": True, "n_cells": 0, "mean_score_std": 0.0}
    return {
        "available": True,
        "per_cell": per_cell,
        "per_date": per_date,
        "summary": summary,
    }
