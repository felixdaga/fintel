"""Backfill: rerun just the error cells from a finished run.

A completed run may have cells that failed (agent error, parse error, rate
limit, etc.) while the rest succeeded.  Re-running the whole job wastes the
money already spent on the good cells.  Backfill isolates the error cells,
re-executes only those, and re-reduces the affected trials and run/job
summaries — so the on-disk artifacts look exactly as if those cells had
succeeded the first time.

The frozen ``rK/config.json`` (``RunConfig``) is the authoritative source
for rebuilding the execution context: universe, data sources, agent spec,
strategy pack path.  No strategy package re-load or re-preflight is needed
— the cache should be warm and the config is self-describing.

Concurrency is independent of the original job: backfill always uses a flat
pool (like ``shared_concurrency``) across all error cells, because the
cells are independent by definition (memory/feedback are off — they were
required for the original run to have produced error cells that can be
safely re-run in isolation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

from fintel.environment.access import dedup_sources
from fintel.environment.cell import Cell
from fintel.environment.factory import RuntimeConfig
from fintel.environment.health import audit_job
from fintel.environment.progress import NullProgress, Progress
from fintel.market.factory import build_data_sources, build_universe
from fintel.market.settings import MarketConfig
from fintel.models.common import Symbol
from fintel.models.decision import AgentResponse
from fintel.models.job import JobResult
from fintel.models.paths import JobPaths, RunPaths
from fintel.models.run import RunConfig, RunResult
from fintel.models.trial import CellRecord, CellResult, TrialResult
from fintel.simulate.artifacts import write_health, write_job_result, write_run_result
from fintel.simulate.cell import CellOutcome, expect_tools, run_cell
from fintel.simulate.queue import map_parallel
from fintel.simulate.record import reduce_job, reduce_run
from fintel.simulate.store import read_json
from fintel.simulate.trial import finalize_trial
from fintel.strategy import load
from fintel.strategy.views import load_pack_context


@dataclass(frozen=True)
class BackfillReport:
    """Summary of what backfill did, for the CLI and tests."""

    job_id: str
    run_index: int
    n_total_cells: int
    n_error_cells: int
    n_reran: int
    n_fixed: int
    n_still_failed: int
    affected_dates: list[str]
    elapsed_ms: int


def run_backfill(
    *,
    job_id: str,
    run_index: int = 1,
    cell_concurrency: int = 1,
    output_root: str | Path = "runs",
    market_config: MarketConfig | None = None,
    progress: Progress | None = None,
) -> BackfillReport:
    """Rerun error cells from a finished run and update on-disk artifacts.

    Parameters
    ----------
    job_id
        The job directory name under ``output_root``.
    run_index
        Which repeat (``rK``) to backfill.  Defaults to 1.
    cell_concurrency
        Flat pool size for rerunning error cells.  Defaults to 1 (sequential).
    output_root
        Root directory containing job folders.  Defaults to ``"runs"``.
    market_config
        Market config (cache root, keys).  Defaults to ``MarketConfig.from_env``
        with ``<output_root>/cache``.
    progress
        Progress sink for live events.  Defaults to ``NullProgress``.

    Returns
    -------
    BackfillReport
        Counts of cells reran, fixed, still failing, and affected dates.
    """
    progress = progress or NullProgress()
    started = time.perf_counter()

    job_paths = JobPaths.under(output_root, job_id)
    run_paths = job_paths.run(run_index)

    run_config = _load_run_config(run_paths)
    run_result = _load_run_result(run_paths)

    if market_config is None:
        market_config = MarketConfig.from_env(cache_root=Path(output_root) / "cache")

    # 1. Identify error cells
    error_cells = _find_error_cells(run_result, run_paths)
    if not error_cells:
        progress.emit(
            "backfill_done", job_id=job_id, run_index=run_index, n_error_cells=0, n_reran=0
        )
        return BackfillReport(
            job_id=job_id,
            run_index=run_index,
            n_total_cells=sum(len(t.cells) for t in run_result.trials),
            n_error_cells=0,
            n_reran=0,
            n_fixed=0,
            n_still_failed=0,
            affected_dates=[],
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    affected_dates = sorted({e.decision_date for e in error_cells})
    progress.emit(
        "backfill_start",
        job_id=job_id,
        run_index=run_index,
        n_error_cells=len(error_cells),
        n_affected_dates=len(affected_dates),
    )

    # 2. Rebuild execution context from frozen config
    universe = build_universe(run_config.universe, config=market_config)
    sources = build_data_sources(run_config.data, config=market_config)
    sources = dedup_sources(sources)
    pack = _load_strategy_context(run_config)
    mission_text = pack.mission_text
    output_schema_text = pack.output_schema_text
    company_names = pack.company_names
    alpha_views = pack.alpha_views
    runtime = RuntimeConfig(
        session_root=run_paths.root / "sessions",
        trace=True,
        reset_sessions=True,
    )
    agent_spec = run_config.agent

    # 3. Rerun error cells in a flat pool
    work_items = _build_work_items(error_cells, run_config, universe, run_paths)

    def _rerun_one(item: _RerunWork) -> CellOutcome | None:
        try:
            return run_cell(
                cell=item.cell,
                sources=sources,
                universe=item.active_universe,
                agent_spec=agent_spec,
                runtime=runtime,
                cell_path=item.cell_path,
                mission_text=mission_text,
                output_schema_text=output_schema_text,
                company_names=company_names,
                alpha_views=alpha_views,
                market_config=market_config,
                progress=progress,
            )
        except Exception:
            return CellOutcome(
                result=CellResult(
                    cell=item.cell.name,
                    symbols=list(item.cell.symbols),
                    status="failed",
                    error="backfill cell executor raised",
                    health="broken",
                    health_issues=["backfill cell executor raised"],
                ),
                response=AgentResponse(
                    views={},
                    outcome="crashed",
                    detail="backfill cell executor raised",
                ),
            )

    outcomes = map_parallel(_rerun_one, work_items, bound=cell_concurrency)
    reran_results: dict[tuple[Date, str], CellOutcome | None] = {}
    for item, outcome in zip(work_items, outcomes):
        reran_results[(item.decision_date, item.cell.name)] = outcome

    # 4. Re-finalize affected trials
    updated_trials: dict[Date, TrialResult] = {}
    for trial in run_result.trials:
        if trial.decision_date not in affected_dates:
            continue
        trial_paths = run_paths.trial(trial.decision_date)
        cells, cell_outcomes = _assemble_trial_outcomes(
            trial=trial,
            trial_paths=trial_paths,
            reran_results=reran_results,
            run_id=run_config.run_id,
        )
        updated = finalize_trial(
            decision_date=trial.decision_date,
            cells=cells,
            outcomes=cell_outcomes,
            paths=trial_paths,
            started_at=trial.started_at or _now_iso(),
            progress=progress,
            duration_ms=trial.duration_ms,
        )
        updated_trials[trial.decision_date] = updated

    # 5. Re-reduce run + job
    new_trials = [updated_trials.get(t.decision_date, t) for t in run_result.trials]
    new_run_result = reduce_run(
        run_id=run_config.run_id,
        job_id=run_config.job_id,
        k_index=run_config.k_index,
        trials=new_trials,
        started_at=run_result.started_at,
        finished_at=_now_iso(),
        duration_ms=run_result.duration_ms,
    )
    write_run_result(run_paths.result, new_run_result)
    _update_job_result(
        job_paths=job_paths,
        run_config=run_config,
        new_run_result=new_run_result,
        agent_spec=agent_spec,
    )

    # 6. Report
    n_reran = len(work_items)
    n_fixed = sum(1 for o in outcomes if o is not None and o.result.status == "ok")
    n_still_failed = n_reran - n_fixed
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    report = BackfillReport(
        job_id=job_id,
        run_index=run_index,
        n_total_cells=sum(len(t.cells) for t in run_result.trials),
        n_error_cells=len(error_cells),
        n_reran=n_reran,
        n_fixed=n_fixed,
        n_still_failed=n_still_failed,
        affected_dates=[d.isoformat() for d in affected_dates],
        elapsed_ms=elapsed_ms,
    )
    progress.emit(
        "backfill_done",
        job_id=job_id,
        run_index=run_index,
        n_error_cells=report.n_error_cells,
        n_reran=n_reran,
        n_fixed=n_fixed,
        n_still_failed=n_still_failed,
        elapsed_ms=elapsed_ms,
    )
    return report


# -- Helpers -----------------------------------------------------------------


@dataclass(frozen=True)
class _ErrorCell:
    decision_date: Date
    cell_name: str
    symbols: list[Symbol]


@dataclass(frozen=True)
class _RerunWork:
    decision_date: Date
    cell: Cell
    active_universe: list[Symbol]
    cell_path: Path


def _load_run_config(run_paths: RunPaths) -> RunConfig:
    path = run_paths.config
    if not path.is_file():
        raise FileNotFoundError(f"Run config not found: {path}. Is this a completed run?")
    return RunConfig.model_validate(read_json(path))


def _load_run_result(run_paths: RunPaths) -> RunResult:
    path = run_paths.result
    if not path.is_file():
        raise FileNotFoundError(f"Run result not found: {path}. Did the run complete?")
    return RunResult.model_validate(read_json(path))


def _find_error_cells(run_result: RunResult, run_paths: RunPaths) -> list[_ErrorCell]:
    """Cells with status not in ('ok', 'skipped') — the ones worth rerunning.

    This includes cells where the agent produced views but the environment
    was broken (e.g. FRED 429): a tool/environment error is still a failure
    worth retrying, since the data source may have recovered and the agent
    can now produce a view with complete data.
    """
    errors: list[_ErrorCell] = []
    for trial in run_result.trials:
        for cell in trial.cells:
            if cell.status not in ("ok", "skipped"):
                errors.append(
                    _ErrorCell(
                        decision_date=trial.decision_date,
                        cell_name=cell.cell,
                        symbols=list(cell.symbols),
                    )
                )
    return errors


def _load_strategy_context(run_config: RunConfig):
    """Re-read pack prompts from the strategy path frozen in RunConfig."""
    return load_pack_context(load(run_config.strategy.path))


def _build_work_items(
    error_cells: list[_ErrorCell],
    run_config: RunConfig,
    universe,
    run_paths: RunPaths,
) -> list[_RerunWork]:
    """Build rerun work items, resolving the active universe per date."""
    items: list[_RerunWork] = []
    for ec in error_cells:
        active = list(universe.active_at(ec.decision_date))
        trial_paths = run_paths.trial(ec.decision_date)
        cell = _rebuild_cell(ec, run_config)
        items.append(
            _RerunWork(
                decision_date=ec.decision_date,
                cell=cell,
                active_universe=active,
                cell_path=trial_paths.cell(ec.cell_name),
            )
        )
    return items


def _rebuild_cell(ec: _ErrorCell, run_config: RunConfig) -> Cell:
    """Reconstruct a Cell from the error-cell metadata."""
    if run_config.scope == "portfolio":
        return Cell(
            run_id=run_config.run_id,
            decision_date=ec.decision_date,
            symbols=tuple(ec.symbols),
            scope="portfolio",
        )
    return Cell(
        run_id=run_config.run_id,
        decision_date=ec.decision_date,
        symbols=(ec.symbols[0],) if ec.symbols else (),
        scope="single_name",
    )


def _assemble_trial_outcomes(
    *,
    trial: TrialResult,
    trial_paths: RunPaths,
    reran_results: dict[tuple[Date, str], CellOutcome | None],
    run_id: str = "",
) -> tuple[list[Cell], list[CellOutcome | None]]:
    """Gather all cells for a trial: reran cells get fresh outcomes, kept
    cells get reconstructed outcomes from their on-disk CellRecord."""
    cells: list[Cell] = []
    outcomes: list[CellOutcome | None] = []

    for cr in trial.cells:
        if len(cr.symbols) == 1:
            cell = Cell(
                run_id=run_id,
                decision_date=trial.decision_date,
                symbols=(cr.symbols[0],),
                scope="single_name",
            )
        else:
            cell = Cell(
                run_id=run_id,
                decision_date=trial.decision_date,
                symbols=tuple(cr.symbols),
                scope="portfolio",
            )
        cells.append(cell)

        key = (trial.decision_date, cr.cell)
        if key in reran_results:
            outcomes.append(reran_results[key])
        else:
            outcomes.append(_reconstruct_outcome(cr, trial_paths))

    return cells, outcomes


def _reconstruct_outcome(cr: CellResult, trial_paths: RunPaths) -> CellOutcome | None:
    """Rebuild a CellOutcome from a kept cell's on-disk CellRecord."""
    cell_path = trial_paths.cell(cr.cell)
    record = _load_cell_record(cell_path)
    if record is None:
        return None
    response = AgentResponse(
        views=dict(record.views),
        outcome=record.outcome,
        detail=record.detail,
        usage=record.usage,
    )
    return CellOutcome(result=cr, response=response)


def _load_cell_record(cell_path: Path) -> CellRecord | None:
    if not cell_path.is_file():
        return None
    try:
        return CellRecord.model_validate(read_json(cell_path))
    except Exception:
        return None


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _update_job_result(
    *,
    job_paths: JobPaths,
    run_config: RunConfig,
    new_run_result: RunResult,
    agent_spec,
) -> None:
    """Re-reduce the job result with the updated run, re-audit health."""
    old_job = _load_job_result(job_paths.root)
    if old_job is None:
        result = reduce_job(
            job_id=run_config.job_id,
            strategy=run_config.strategy.name,
            agent=agent_spec.name,
            k_repeats=run_config.k_repeats,
            runs=[new_run_result],
        )
    else:
        # Load each run's full result.json from disk, replacing the
        # backfilled one with the fresh result.
        runs: list[RunResult] = []
        for summary in old_job.runs:
            if summary.k_index == run_config.k_index:
                runs.append(new_run_result)
            else:
                rr = _load_run_result_from_disk(job_paths.run(summary.k_index))
                if rr is not None:
                    runs.append(rr)
        result = reduce_job(
            job_id=old_job.job_id,
            strategy=old_job.strategy,
            agent=old_job.agent,
            k_repeats=old_job.k_repeats,
            runs=runs,
            started_at=old_job.started_at,
            finished_at=_now_iso(),
        )

    job_health = audit_job(job_paths.root, expect_tools=expect_tools(agent_spec))
    write_health(job_paths.root / "health.json", job_health)
    if job_health["status"] != "ok":
        result = result.model_copy(
            update={
                "health": job_health["status"],
                "health_issues": list(job_health.get("issues") or []) + list(result.health_issues),
                "status": (
                    "failed"
                    if job_health["status"] == "broken" and result.status == "ok"
                    else result.status
                ),
            }
        )
    write_job_result(job_paths.result, result)


def _load_job_result(job_root: Path) -> JobResult | None:
    path = job_root / "result.json"
    if not path.is_file():
        return None
    try:
        return JobResult.model_validate(read_json(path))
    except Exception:
        return None


def _load_run_result_from_disk(run_paths: RunPaths) -> RunResult | None:
    path = run_paths.result
    if not path.is_file():
        return None
    try:
        return RunResult.model_validate(read_json(path))
    except Exception:
        return None
