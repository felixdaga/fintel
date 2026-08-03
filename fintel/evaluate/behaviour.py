"""Layer 1 — agent behavioural stochasticity across K repeats.

Always on, but **no-ops for agents with no tool-call trace** (scripted /
constant baselines): a `CellBehaviour` with `has_trace=False` is excluded, so
the layer reports `n/a: no tool calls` rather than zeros that would read as
"perfectly stable". This is the only behaviour-aware seam: an agent that never
called tools has no behaviour to be stochastic about.

Reads `RunData.behaviour_by_date` (per-cell process stats) and the per-run
signals. It never touches prices — behaviour is process, not outcome.
"""

from __future__ import annotations

from datetime import date as Date
from statistics import mean, pstdev
from typing import Any

from fintel.models.common import Symbol
from fintel.models.evaluate import RunData


def _cells_with_traces(runs: list[RunData]) -> set[tuple[Date, Symbol]]:
    """The (date, cell) pairs that recorded tool activity in any run."""
    out: set[tuple[Date, Symbol]] = set()
    for r in runs:
        for d, cells in r.behaviour_by_date.items():
            for sym, b in cells.items():
                if b.has_trace:
                    out.add((d, sym))
    return out


def _stdev(vals: list[float]) -> float:
    """Population std; 0 for a single value (no dispersion to report)."""
    return pstdev(vals) if len(vals) >= 2 else 0.0


def analyse(runs: list[RunData]) -> dict[str, Any]:
    """Per-cell behavioural dispersion across K repeats + a summary.

    Returns `{per_cell: {...}, summary: {...}}`. When no cell has a trace the
    summary carries `available=False` so the report can show "n/a".
    """
    cells = _cells_with_traces(runs)
    if not cells:
        return {
            "available": False,
            "per_cell": {},
            "summary": {"note": "no tool-call traces (agent has no behaviour to disperse)"},
        }

    per_cell: dict[str, dict] = {}
    for d, sym in sorted(cells):
        # gather this cell's behaviour across the runs that have it
        records = []
        for r in runs:
            b = r.behaviour_by_date.get(d, {}).get(sym)
            if b and b.has_trace:
                records.append(b)
        if len(records) < 2:
            continue  # a single run has no cross-run dispersion
        calls = [b.n_tool_calls for b in records]
        errors = [b.n_tool_errors for b in records]
        reads = [b.n_reads for b in records]
        per_cell[f"{d.isoformat()}|{sym}"] = {
            "n_runs": len(records),
            "n_tool_calls_mean": round(mean(calls), 2),
            "n_tool_calls_std": round(_stdev([float(c) for c in calls]), 2),
            "n_tool_calls_range": (min(calls), max(calls)),
            "n_tool_errors_mean": round(mean(errors), 2),
            "n_reads_mean": round(mean(reads), 2),
        }

    # summary: mean across cells of the per-cell stds
    if per_cell:
        call_stds = [v["n_tool_calls_std"] for v in per_cell.values()]
        summary = {
            "available": True,
            "n_cells": len(per_cell),
            "mean_call_count_std": round(mean(call_stds), 3) if call_stds else 0.0,
        }
    else:
        summary = {"available": True, "n_cells": 0, "mean_call_count_std": 0.0}
    return {"available": True, "per_cell": per_cell, "summary": summary}
