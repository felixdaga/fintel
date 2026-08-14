"""The evaluation pipeline orchestration + report writer.

`report(job_dir, scoring=...)` runs the full pipeline — read -> signal ->
stochasticity + KPI + holdings -> `ReportPayload` — and writes `report.json`
and `report.md` under `<job_dir>/report/`. This is the one entry point the CLI
calls; it knows the order of the layers and nothing about how any of them
compute their results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fintel.evaluate.agent_eval import evaluate as run_agent_eval
from fintel.evaluate.behaviour import analyse as analyse_behaviour
from fintel.evaluate.cadence import detect_cadence
from fintel.evaluate.holdings import build as build_holdings
from fintel.evaluate.kpi import compute as compute_kpi
from fintel.evaluate.prices import price_lookup_for
from fintel.evaluate.read import load_job
from fintel.evaluate.signals import build_signals
from fintel.evaluate.variance import analyse as analyse_variance
from fintel.market.realized import PriceLookup
from fintel.models.evaluate import ReportPayload
from fintel.models.paths import JobPaths
from fintel.models.strategy import EvalSpec, ScoringSpec

logger = logging.getLogger(__name__)


def report(
    job_dir: Path,
    *,
    scoring: ScoringSpec,
    cache_root: str | Path | None = None,
    prices: PriceLookup | None = None,
    eval_spec: EvalSpec | None = None,
    strategy_root: Path | None = None,
    shared_concurrency: int | None = None,
) -> ReportPayload:
    """Run the full evaluation over a finished job and return the payload.

    `prices` may be injected (for tests); by default it is built from the job's
    cache via `price_lookup_for`.

    The pack's ``scoring.py`` is imported by module path (e.g.
    ``geopol_trade_war_2018.scoring:geopol_signal``), which requires the pack's
    parent directory on ``sys.path``. We insert ``strategy_root.parent`` here so
    the signal/KPI resolution can import it — simulate never imports scoring
    (KPI is computed here, at report time), so nothing else puts the packages
    dir on the path.
    """
    import sys

    if strategy_root is not None:
        parent = str(Path(strategy_root).parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

    runs = load_job(job_dir)
    signals = build_signals(runs, signal=scoring.signal, transform=scoring.transform)
    if prices is None:
        prices = price_lookup_for(job_dir, cache_root=cache_root)

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
    cadence = detect_cadence(
        name=f"{strategy_name} {job_id}", dates=list(signals.decision_dates)
    )
    eval_params = {
        **scoring.params,
        "metric_key": scoring.metric_key,
        "ppy": cadence["ppy"],
        "cadence": cadence,
        "strategy_name": strategy_name or job_id,
    }

    kpi_result = compute_kpi(
        signals,
        prices,
        kpi=scoring.kpi,
        horizons=scoring.horizons,
        params=eval_params,
    )
    behaviour = analyse_behaviour(runs)
    variance = analyse_variance(signals.per_run)
    holdings = build_holdings(signals, prices, params=eval_params)

    # Agent-on-agent evaluation (opt-in via [eval] in strategy.toml).
    agent_eval_result: dict | None = None
    if eval_spec is not None and strategy_root is not None:
        try:
            agent_eval_result = run_agent_eval(
                job_dir,
                eval_spec=eval_spec,
                strategy_root=strategy_root,
                cache_root=cache_root,
                shared_concurrency=shared_concurrency,
            )
        except Exception:
            logger.warning("agent_eval failed", exc_info=True)
            agent_eval_result = {"available": False, "summary": {"note": "eval raised"}}

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
        agent_eval=agent_eval_result,
        meta={"n_runs": len(runs), "cadence": cadence},
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
    cad = (p.meta or {}).get("cadence") or {}
    if cad:
        lines.append(
            f"cadence: {cad.get('cadence', '?')}  ppy: {cad.get('ppy', '?')}  "
            f"median_gap: {cad.get('median_gap_days')}d  "
            f"dates: {len(p.decision_dates)}  universe: {len(p.universe)} "
            f"({', '.join(p.universe[:8])}{'…' if len(p.universe) > 8 else ''})"
        )
        lines.append(
            "Annualized: Sharpe, IR, vol, ICIR (×√ppy); turnover ×ppy; "
            "cost (gross/net)^(1/years)−1. Not annualized: total return, max DD, mean IC."
        )
    else:
        lines.append(
            f"dates: {len(p.decision_dates)}  universe: {len(p.universe)} "
            f"({', '.join(p.universe[:8])}{'…' if len(p.universe) > 8 else ''})"
        )
    lines.append("")

    # KPI
    lines.append("## KPI")
    ens = p.kpi_result.get("ensemble", {})
    per_horizon = ens.get("per_horizon", {})
    if per_horizon:
        ppy = ens.get("ppy") or cad.get("ppy")
        lines.append(
            f"ensemble ({ens.get('kpi', p.kpi)}, metric_key={ens.get('metric_key', p.metric_key)}"
            f"{f', ppy={ppy:g}' if ppy else ''}):"
        )
        lines.append("")
        lines.append(
            "| h | sp_mean | sp_t | sp_icir_ann | pe_mean | pe_t | pe_icir_ann | n |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for h in p.horizons:
            row = per_horizon.get(str(h)) or per_horizon.get(h) or {}
            lines.append(
                "| {h} | {sp_mean} | {sp_t} | {sp_ann} | {pe_mean} | {pe_t} | {pe_ann} | {n} |".format(
                    h=h,
                    sp_mean=_fmt(row.get("mean_ic")),
                    sp_t=_fmt(row.get("t_stat")),
                    sp_ann=_fmt(row.get("icir_ann")),
                    pe_mean=_fmt(row.get("pearson_mean")),
                    pe_t=_fmt(row.get("pearson_t")),
                    pe_ann=_fmt(row.get("pearson_icir_ann")),
                    n=row.get("n_periods", 0),
                )
            )
        lines.append("")
        lines.append(
            "Mean IC is a rank/linear correlation (not annualized). "
            "t = ICIR_raw × √n (mean IC vs 0). ICIR_ann = ICIR_raw × √ppy. "
            "`h` is decision-grid steps."
        )
        # per-run dispersion
        per_run = p.kpi_result.get("per_run", [])
        if len(per_run) >= 2:
            irs = []
            for r in per_run:
                rh = (
                    (r.get("per_horizon", {}) or {}).get(str(p.horizons[0])) if p.horizons else None
                )
                if rh is None and p.horizons:
                    rh = (r.get("per_horizon", {}) or {}).get(p.horizons[0])
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
        lines.append("## Holdings & returns")
        h = p.holdings
        cad_h = h.get("cadence") or cad.get("cadence")
        ppy_h = h.get("ppy") or cad.get("ppy")
        years = h.get("years")
        lines.append(
            f"cadence: {cad_h}  ppy: {ppy_h}  years: {_fmt(years)}  "
            f"active_budget: {h.get('active_budget')}  cost_bps: {h.get('cost_bps')}  "
            f"as_of: {h.get('as_of')}"
        )
        mvo = h.get("mvo") or {}
        if mvo:
            if mvo.get("n_dates"):
                lines.append(
                    f"MVO: solved {mvo.get('n_solved', 0)}/{mvo.get('n_dates', 0)} dates  "
                    f"fallback {mvo.get('n_fallback', 0)}  "
                    f"distinct_from_naive: {mvo.get('distinct_from_naive')}  "
                    f"status: {mvo.get('reasons')}"
                )
            elif mvo.get("note"):
                lines.append(f"MVO: {mvo.get('note')}")
        metrics = h.get("metrics") or {}
        if metrics:
            lines.append("")
            lines.append(
                "| book | total | ann ret | ann vol | max DD | ann Sharpe | ann IR | "
                "ann turn | ann cost | n |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            order = list(h.get("books") or [])
            # expand sw_long/ew_long into per-threshold keys present in metrics
            keys: list[str] = []
            seen: set[str] = set()
            for b in order:
                if b in ("sw_long", "ew_long"):
                    prefix = "sw_" if b == "sw_long" else "ew_"
                    for k in metrics:
                        if k.startswith(prefix) and k not in seen:
                            keys.append(k)
                            seen.add(k)
                elif b in metrics and b not in seen:
                    keys.append(b)
                    seen.add(b)
            for k in metrics:
                if k not in seen:
                    keys.append(k)
            for k in keys:
                row = metrics[k]
                label = row.get("label") or k
                lines.append(
                    "| {lab} | {tot} | {ar} | {av} | {dd} | {sh} | {ir} | {tn} | {cs} | {n} |".format(
                        lab=label,
                        tot=_fmt_pct(row.get("total")),
                        ar=_fmt_pct(row.get("ann_ret")),
                        av=_fmt_pct(row.get("ann_vol")),
                        dd=_fmt_pct(row.get("max_dd")),
                        sh=_fmt(row.get("ann_sharpe")),
                        ir=_fmt(row.get("ann_ir")),
                        tn=_fmt(row.get("ann_turn")),
                        cs=_fmt_pct(row.get("ann_cost")),
                        n=row.get("n_periods", 0),
                    )
                )
        else:
            ens_h = h.get("ensemble", {})
            lines.append(
                f"active_budget: {h.get('active_budget')}  cost_bps: {h.get('cost_bps')}  "
                f"turnover_total: {ens_h.get('turnover_total')}"
            )
            gross = ens_h.get("gross", [])
            net = ens_h.get("net", [])
            if gross and net:
                lines.append(
                    f"ensemble NAV: gross {gross[-1].get('nav', 1.0)}  "
                    f"net {net[-1].get('nav', 1.0)}"
                )
    lines.append("")

    # Agent-on-agent evaluation (opt-in)
    ae = p.agent_eval
    if ae is not None:
        lines.append("## Agent-on-agent evaluation")
        if ae.get("available"):
            s = ae.get("summary", {})
            lines.append(
                f"cells rated: {s.get('n_rated', 0)}  failed: {s.get('n_failed', 0)}  "
                f"run: {s.get('rating_run', '?')}  agent: {s.get('rating_agent', '?')}  "
                f"elapsed: {s.get('elapsed_ms', 0)}ms"
            )
            per_cell = ae.get("per_cell", {})
            if per_cell:
                lines.append("")
                lines.append(
                    "| date | cell | loyalty | bias | aggression | rec_rating | bias_flags |"
                )
                lines.append("|---|---|---|---|---|---|---|")
                for key, rating in sorted(per_cell.items()):
                    date_str, cell_name = key.split("|", 1)
                    loyalty = _fmt_score(rating.get("loyalty_score"))
                    bias = _fmt_score(rating.get("bias_score"))
                    aggression = _fmt_score(rating.get("aggression_score"))
                    rec = rating.get("recommendation_rating", "—")
                    flags = rating.get("bias_flags", [])
                    flags_str = ", ".join(flags) if flags else "—"
                    lines.append(
                        f"| {date_str} | {cell_name} | {loyalty} | {bias} | {aggression} | {rec} | {flags_str} |"
                    )
        else:
            lines.append(f"_{ae.get('summary', {}).get('note', 'n/a')}_")
    lines.append("")
    return "\n".join(lines)


def _fmt_score(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, (int, float)):
        return f"{x:+.1f}" if x != 0 else " 0.0"
    return str(x)


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _fmt_pct(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, (int, float)):
        return f"{100.0 * float(x):.2f}%"
    return str(x)
