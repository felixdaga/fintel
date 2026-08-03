"""The reader adapter — fintel run artifacts into the evaluation models.

This is the one fintel-specific seam in the evaluation layer. Everything
upstream of it (`signals`, `kpi`, `holdings`, `behaviour`, `variance`) is
portable math operating on `RunData` / `Signals`; this module is what knows
about the on-disk layout (`runs/<job>/rK/trials/<date>/...`).

Read-only. It never writes and never imports `simulate/` — the layer guard in
`tests/test_architecture.py` forbids it.
"""

from __future__ import annotations

import json
from datetime import date as Date
from pathlib import Path

from fintel.models.common import Symbol
from fintel.models.decision import View
from fintel.models.evaluate import CellBehaviour, RunData
from fintel.models.paths import JobPaths
from fintel.models.trial import CellRecord


def _load_decision(trial_dir: Path, decision_date: Date) -> dict[Symbol, View]:
    """`decision.json` is keyed by symbol -> View. Missing file = no decision."""
    path = trial_dir / "decision.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    out: dict[Symbol, View] = {}
    for sym, view in raw.items():
        try:
            out[Symbol(sym)] = View.model_validate(view)
        except Exception:
            # A corrupt view shouldn't sink the whole evaluation; skip it.
            continue
    return out


def _load_behaviour(
    trial_dir: Path, decision_date: Date
) -> dict[Symbol, CellBehaviour]:
    """One `CellRecord` per cell -> `CellBehaviour`. `has_trace` is set when the
    cell recorded any tool activity, so the behaviour layer can no-op on agents
    that never call tools."""
    cells_dir = trial_dir / "cells"
    if not cells_dir.is_dir():
        return {}
    out: dict[Symbol, CellBehaviour] = {}
    for cell_path in sorted(cells_dir.glob("*.json")):
        try:
            rec = CellRecord.model_validate_json(cell_path.read_text())
        except Exception:
            continue
        env = rec.environment or {}
        n_reads = int(env.get("n_reads", 0) or 0)
        # A cell has a "trace" if it recorded reads or views came from tools.
        has_trace = n_reads > 0 or bool(rec.views)
        out[Symbol(rec.cell)] = CellBehaviour(
            cell=rec.cell,
            decision_date=decision_date.isoformat(),
            n_tool_calls=int(env.get("n_tool_calls", n_reads) or 0),
            n_tool_errors=int(env.get("n_tool_errors", 0) or 0),
            n_reads=n_reads,
            outcome=rec.outcome,
            has_trace=has_trace,
        )
    return out


def load_run(run_dir: Path) -> RunData:
    """Load one repeat (`runs/<job>/rK/`) into `RunData`."""
    run_paths = JobPaths(root=run_dir.parent).run(int(run_dir.name[1:]))
    config_path = run_paths.config
    run_id = run_dir.name
    k_index = int(run_dir.name[1:]) if run_dir.name[1:].isdigit() else 0

    decision_dates: list[Date] = []
    universe: list[Symbol] = []
    views_by_date: dict[Date, dict[Symbol, View]] = {}
    behaviour_by_date: dict[Date, dict[Symbol, CellBehaviour]] = {}

    # Resolve the schedule + universe from the run config if present.
    config: dict = {}
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            config = {}
    for d in config.get("schedule_dates") or []:
        try:
            decision_dates.append(Date.fromisoformat(d))
        except ValueError:
            continue
    for s in config.get("universe_symbols") or []:
        universe.append(Symbol(s))

    # Walk the trials on disk — the source of truth is the artifacts, not the
    # config, so a partial run still loads what it produced.
    trials_dir = run_dir / "trials"
    if trials_dir.is_dir():
        for trial_dir in sorted(p for p in trials_dir.iterdir() if p.is_dir()):
            try:
                d = Date.fromisoformat(trial_dir.name)
            except ValueError:
                continue
            if d not in decision_dates:
                decision_dates.append(d)
            views = _load_decision(trial_dir, d)
            behaviour = _load_behaviour(trial_dir, d)
            if views:
                views_by_date[d] = views
                for sym in views:
                    if sym not in universe:
                        universe.append(sym)
            if behaviour:
                behaviour_by_date[d] = behaviour

    decision_dates.sort()
    universe.sort()
    return RunData(
        run_id=run_id,
        k_index=k_index,
        decision_dates=decision_dates,
        universe=universe,
        views_by_date=views_by_date,
        behaviour_by_date=behaviour_by_date,
    )


def load_job(job_dir: Path) -> list[RunData]:
    """Load all repeats of a job (`runs/<job>/`) — one `RunData` per `rK`."""
    paths = JobPaths(root=job_dir)
    runs: list[RunData] = []
    for run_dir in paths.run_dirs():
        runs.append(load_run(run_dir))
    return runs
