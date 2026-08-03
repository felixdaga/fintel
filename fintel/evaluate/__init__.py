"""The evaluation layer — read-only analytics over finished runs.

Sits above `simulate/` (L7) and imports it never. The strategy defines **the
signal** and **the KPI**; this layer owns the mechanics: ensemble, holdings,
returns, stochasticity, rendering.

Entry points:
- `read.load_job(job_dir)` -> list[RunData]
- `signals.build_signals(runs, signal=..., transform=...)` -> Signals
- `kpi.compute(signals, prices, kpi=..., horizons=..., params=...)` -> dict
- `behaviour.analyse(runs)` / `variance.analyse(per_run)` -> dict
- `holdings.build(signals, prices, params=...)` -> dict | None (opt-in)
- `report.report(job_dir, scoring=...)` -> ReportPayload  (full pipeline)
- `report.write_report(payload, job_dir)` -> writes report.json + report.md
"""

from __future__ import annotations

from fintel.evaluate.behaviour import analyse as analyse_behaviour
from fintel.evaluate.holdings import build as build_holdings
from fintel.evaluate.kpi import compute as compute_kpi
from fintel.evaluate.prices import price_lookup_for
from fintel.evaluate.read import load_job, load_run
from fintel.evaluate.report import report, render_markdown, write_report
from fintel.evaluate.signals import build_signals
from fintel.evaluate.variance import analyse as analyse_variance

__all__ = [
    "load_job",
    "load_run",
    "build_signals",
    "compute_kpi",
    "analyse_behaviour",
    "analyse_variance",
    "build_holdings",
    "price_lookup_for",
    "report",
    "render_markdown",
    "write_report",
]
