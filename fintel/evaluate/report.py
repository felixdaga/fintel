"""The evaluation pipeline orchestration + report writer.

`report(job_dir, scoring=...)` runs the full pipeline — read -> signal ->
stochasticity + KPI + holdings -> `ReportPayload` — and writes `report.json`
and `report.md` under `<job_dir>/report/`. This is the one entry point the CLI
calls; it knows the order of the layers and nothing about how any of them
compute their results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fintel.evaluate.behaviour import analyse as analyse_behaviour
from fintel.evaluate.holdings import build as build_holdings
from fintel.evaluate.kpi import compute as compute_kpi
from fintel.evaluate.prices import price_lookup_for
from fintel.evaluate.read import load_job
from fintel.evaluate.signals import build_signals
from fintel.evaluate.variance import analyse as analyse_variance
from fintel.market.realized import PriceLookup
from fintel.models.evaluate import ReportPayload
from fintel.models.paths import JobPaths
from fintel.models.strategy import ScoringSpec


def report(
    job_dir: Path,
    *,
    scoring: ScoringSpec,
    cache_root: str | Path | None = None,
    prices: PriceLookup | None = None,
) -> ReportPayload:
    """Run the full evaluation over a finished job and return the payload.

    `prices` may be injected (for tests); by default it is built from the job's
    cache via `price_lookup_for`.
    """
    runs = load_job(job_dir)
    signals = build_signals(runs, signal=scoring.signal, transform=scoring.transform)
    if prices is None:
        prices = price_lookup_for(job_dir, cache_root=cache_root)

    kpi_result = compute_kpi(
        signals,
        prices,
        kpi=scoring.kpi,
        horizons=scoring.horizons,
        params={**scoring.params, "metric_key": scoring.metric_key},
    )
    behaviour = analyse_behaviour(runs)
    variance = analyse_variance(signals.per_run)
    holdings = build_holdings(signals, prices, params=scoring.params)

    job_id = job_dir.name
    # strategy name + k_repeats from the first run's config
    strategy_name = ""
    k_repeats = len(runs)
    config_path = JobPaths(root=job_dir).config
    if config_path.is_file():
        try:
            cfg = json.loads(config_path.read_text())
            strat = cfg.get("strategy", "")
            if isinstance(strat, dict):
                strategy_name = strat.get("name", "")
            elif isinstance(strat, str):
                strategy_name = strat
        except json.JSONDecodeError:
            pass

    return ReportPayload(
        job_id=job_id,
        k_repeats=k_repeats,
        strategy=strategy_name,
        signal=scoring.signal,
        transform=scoring.transform,
        kpi=scoring.kpi,
        metric_key=scoring.metric_key,
        horizons=list(scoring.horizons),
        decision_dates=signals.decision_dates,
        universe=signals.universe,
        kpi_result=kpi_result,
        behaviour=behaviour,
        variance=variance,
        holdings=holdings,
        meta={"n_runs": len(runs)},
    )


def write_report(payload: ReportPayload, job_dir: Path) -> dict[str, Path]:
    """Write `report.json` + `report.md` under `<job_dir>/report/`. Returns the
    paths written."""
    report_dir = JobPaths(root=job_dir).report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "report.json"
    md_path = report_dir / "report.md"
    json_path.write_text(payload.model_dump_json(indent=2))
    md_path.write_text(render_markdown(payload))
    return {"json": json_path, "markdown": md_path}


def render_markdown(p: ReportPayload) -> str:
    """A compact, human-readable rendering of the payload."""
    lines: list[str] = []
    lines.append(f"# fintel report — {p.job_id}")
    lines.append("")
    lines.append(
        f"strategy: `{p.strategy}`  agent repeats: {p.k_repeats}  "
        f"signal: `{p.signal}`  transform: `{p.transform}`  kpi: `{p.kpi}`"
    )
    lines.append(
        f"dates: {len(p.decision_dates)}  universe: {len(p.universe)} ({', '.join(p.universe[:8])}{'…' if len(p.universe) > 8 else ''})"
    )
    lines.append("")

    # KPI
    lines.append("## KPI")
    ens = p.kpi_result.get("ensemble", {})
    per_horizon = ens.get("per_horizon", {})
    if per_horizon:
        lines.append(
            f"ensemble ({ens.get('kpi', p.kpi)}, metric_key={ens.get('metric_key', p.metric_key)}):"
        )
        lines.append("")
        lines.append("| horizon | mean_ic | raw_icir | n_periods |")
        lines.append("|---|---|---|---|")
        for h in p.horizons:
            row = per_horizon.get(str(h)) or per_horizon.get(h) or {}
            mi = row.get("mean_ic")
            ir = row.get("raw_icir")
            n = row.get("n_periods", 0)
            lines.append(f"| {h} | {_fmt(mi)} | {_fmt(ir)} | {n} |")
        # per-run dispersion
        per_run = p.kpi_result.get("per_run", [])
        if len(per_run) >= 2:
            irs = []
            for r in per_run:
                rh = (
                    (r.get("per_horizon", {}) or {}).get(str(p.horizons[0])) if p.horizons else None
                )
                if rh and rh.get("raw_icir") is not None:
                    irs.append(rh["raw_icir"])
            if irs:
                lines.append("")
                lines.append(
                    f"per-run {p.horizons[0]}-horizon raw_icir: "
                    f"mean={round(sum(irs) / len(irs), 4)}  min={min(irs)}  max={max(irs)}"
                )
    else:
        lines.append("_no forward periods (need >= 2 decision dates)_")
    lines.append("")

    # Variance (L2)
    v = p.variance
    if v.get("available"):
        s = v.get("summary", {})
        lines.append("## Output variance (L2)")
        lines.append(
            f"cells: {s.get('n_cells', 0)}  mean_score_std: {s.get('mean_score_std', 0)}  "
            f"all_signs_agree: {s.get('min_sign_agreement')}  "
            f"mean_rank_corr: {s.get('mean_rank_corr')}"
        )
    else:
        lines.append("## Output variance (L2)")
        lines.append(f"_{v.get('summary', {}).get('note', 'n/a')}_")
    lines.append("")

    # Behaviour (L1)
    b = p.behaviour
    if b.get("available"):
        s = b.get("summary", {})
        lines.append("## Behaviour (L1)")
        lines.append(
            f"cells: {s.get('n_cells', 0)}  mean_call_count_std: {s.get('mean_call_count_std', 0)}"
        )
    else:
        lines.append("## Behaviour (L1)")
        lines.append(f"_{b.get('summary', {}).get('note', 'n/a')}_")
    lines.append("")

    # Holdings (opt-in)
    if p.holdings is not None:
        ens = p.holdings.get("ensemble", {})
        lines.append("## Holdings & returns (opt-in)")
        lines.append(
            f"active_budget: {p.holdings.get('active_budget')}  cost_bps: {p.holdings.get('cost_bps')}  "
            f"turnover_total: {ens.get('turnover_total')}"
        )
        gross = ens.get("gross", [])
        net = ens.get("net", [])
        if gross and net:
            lines.append(
                f"ensemble NAV: gross {gross[-1].get('nav', 1.0)}  net {net[-1].get('nav', 1.0)}"
            )
    lines.append("")
    return "\n".join(lines)


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)
