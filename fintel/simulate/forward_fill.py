"""Forward-fill: add new decision dates to a finished run.

Like backfill, but for *new* dates instead of error cells. The frozen
``rK/config.json`` is the authoritative source for rebuilding the execution
context — universe, data sources, agent spec, strategy pack. New dates are
taken from the schedule (or an explicit list), cells are run via the same
``run_cell`` primitive, and the new trials are merged into the existing run +
job results on disk.

This is what powers a deployed strategy's weekly refresh: one job grows over
time as each new decision date is forward-filled in, without re-running any
prior date.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

from fintel.environment.access import dedup_sources
from fintel.environment.cell import Cell
from fintel.environment.factory import RuntimeConfig, cells_for
from fintel.environment.health import audit_job
from fintel.environment.progress import NullProgress, Progress
from fintel.market.calendar import TradingCalendar
from fintel.market.factory import build_data_sources, build_schedule, build_universe
from fintel.market.settings import MarketConfig
from fintel.models.common import Symbol
from fintel.models.decision import AgentResponse
from fintel.models.job import JobResult
from fintel.models.market import ScheduleRef
from fintel.models.paths import JobPaths, RunPaths
from fintel.models.run import RunConfig, RunResult
from fintel.models.trial import CellResult, TrialResult
from fintel.simulate.artifacts import write_health, write_job_result, write_run_result
from fintel.simulate.cell import CellOutcome, expect_tools, run_cell
from fintel.simulate.queue import map_parallel
from fintel.simulate.record import reduce_job, reduce_run
from fintel.simulate.store import read_json
from fintel.simulate.trial import finalize_trial
from fintel.strategy import load
from fintel.strategy.views import load_pack_context


@dataclass(frozen=True)
class ForwardFillReport:
    """Summary of what forward-fill did, for the CLI and tests."""

    job_id: str
    run_index: int
    n_existing_dates: int
    n_new_dates: int
    n_cells_run: int
    n_ok: int
    n_failed: int
    new_dates: list[str]
    elapsed_ms: int


def run_forward_fill(
    *,
    job_id: str,
    run_index: int = 1,
    through: Date | None = None,
    dates: list[Date] | None = None,
    cell_concurrency: int = 1,
    output_root: str | Path = "runs",
    market_config: MarketConfig | None = None,
    schedule_override: ScheduleRef | None = None,
    progress: Progress | None = None,
) -> ForwardFillReport:
    """Add new decision dates to a finished run and merge results in-place.

    Parameters
    ----------
    job_id
        The job directory name under ``output_root``.
    run_index
        Which repeat (``rK``) to forward-fill.  Defaults to 1.
    through
        Run all scheduled dates up to and including this date that aren't
        already in the run.  Defaults to today.
    dates
        Explicit list of dates to add (overrides the schedule).  Dates already
        in the run are skipped.
    cell_concurrency
        Flat pool size for running cells.  Defaults to 1 (sequential).
    output_root
        Root directory containing job folders.  Defaults to ``"runs"``.
    market_config
        Market config (cache root, keys).  Defaults to ``MarketConfig.from_env``
        with ``<output_root>/cache``.
    schedule_override
        Replace the frozen run config's schedule with this one.  Used by
        deployed strategies whose original run used ``custom_dates`` but
        need an open-ended weekly cadence going forward.
    progress
        Progress sink for live events.  Defaults to ``NullProgress``.

    Returns
    -------
    ForwardFillReport
        Counts of dates added, cells run, and pass/fail breakdown.
    """
    progress = progress or NullProgress()
    started = time.perf_counter()
    through = through or Date.today()

    job_paths = JobPaths.under(output_root, job_id)
    run_paths = job_paths.run(run_index)

    run_config = _load_run_config(run_paths)
    run_result = _load_run_result(run_paths)

    if market_config is None:
        market_config = MarketConfig.from_env(cache_root=Path(output_root) / "cache")

    # 1. Identify new dates
    existing_dates = {t.decision_date for t in run_result.trials}
    new_dates = _resolve_new_dates(run_config, existing_dates, through, dates, schedule_override)
    if not new_dates:
        progress.emit(
            "forward_fill_done",
            job_id=job_id,
            run_index=run_index,
            n_new_dates=0,
        )
        return ForwardFillReport(
            job_id=job_id,
            run_index=run_index,
            n_existing_dates=len(existing_dates),
            n_new_dates=0,
            n_cells_run=0,
            n_ok=0,
            n_failed=0,
            new_dates=[],
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    progress.emit(
        "forward_fill_start",
        job_id=job_id,
        run_index=run_index,
        n_new_dates=len(new_dates),
        new_dates=[d.isoformat() for d in new_dates],
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

    # 3. Build work items: cells for each new date × universe
    work_items = _build_work_items(new_dates, run_config, universe, run_paths)

    # 4. Run cells in a flat pool
    def _run_one(item: _FillWork) -> CellOutcome | None:
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
                    error="forward-fill cell executor raised",
                    health="broken",
                    health_issues=["forward-fill cell executor raised"],
                ),
                response=AgentResponse(
                    views={},
                    outcome="crashed",
                    detail="forward-fill cell executor raised",
                ),
            )

    outcomes = map_parallel(_run_one, work_items, bound=cell_concurrency)

    # 5. Group outcomes by date, finalize each new trial
    outcomes_by_date: dict[Date, list[tuple[Cell, CellOutcome | None]]] = {}
    for item, outcome in zip(work_items, outcomes):
        outcomes_by_date.setdefault(item.decision_date, []).append((item.cell, outcome))

    new_trials: list[TrialResult] = []
    for d in new_dates:
        cell_outcomes = outcomes_by_date.get(d, [])
        cells = [c for c, _ in cell_outcomes]
        outs = [o for _, o in cell_outcomes]
        trial_paths = run_paths.trial(d)
        trial = finalize_trial(
            decision_date=d,
            cells=cells,
            outcomes=outs,
            paths=trial_paths,
            started_at=_now_iso(),
            progress=progress,
        )
        new_trials.append(trial)

    # 6. Merge new trials into existing run, re-reduce
    all_trials = list(run_result.trials) + new_trials
    all_trials.sort(key=lambda t: t.decision_date)
    new_run_result = reduce_run(
        run_id=run_config.run_id,
        job_id=run_config.job_id,
        k_index=run_config.k_index,
        trials=all_trials,
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

    # 7. Report
    n_ok = sum(1 for o in outcomes if o is not None and o.result.status == "ok")
    n_failed = len(outcomes) - n_ok
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    report = ForwardFillReport(
        job_id=job_id,
        run_index=run_index,
        n_existing_dates=len(existing_dates),
        n_new_dates=len(new_dates),
        n_cells_run=len(work_items),
        n_ok=n_ok,
        n_failed=n_failed,
        new_dates=[d.isoformat() for d in new_dates],
        elapsed_ms=elapsed_ms,
    )
    progress.emit(
        "forward_fill_done",
        job_id=job_id,
        run_index=run_index,
        n_new_dates=len(new_dates),
        n_cells=len(work_items),
        n_ok=n_ok,
        n_failed=n_failed,
        elapsed_ms=elapsed_ms,
    )
    return report


# -- Helpers -----------------------------------------------------------------


@dataclass(frozen=True)
class _FillWork:
    decision_date: Date
    cell: Cell
    active_universe: list[Symbol]
    cell_path: Path


def _resolve_new_dates(
    run_config: RunConfig,
    existing: set[Date],
    through: Date,
    explicit: list[Date] | None,
    schedule_override: ScheduleRef | None = None,
) -> list[Date]:
    """Dates to add: from the schedule or explicit list, minus existing, ≤ through."""
    if explicit:
        candidates = sorted(explicit)
    else:
        calendar = TradingCalendar()
        ref = schedule_override or run_config.schedule
        schedule = build_schedule(ref, calendar=calendar)
        candidates = sorted(d for d in schedule.dates(end=through) if d <= through)
    return [d for d in candidates if d not in existing]


def _build_work_items(
    new_dates: list[Date],
    run_config: RunConfig,
    universe,
    run_paths: RunPaths,
) -> list[_FillWork]:
    """Build cells for each new date × universe (same fan-out as a fresh run)."""
    items: list[_FillWork] = []
    for d in new_dates:
        active = list(universe.active_at(d))
        cells = cells_for(
            run_id=run_config.run_id,
            decision_date=d,
            symbols=active,
            scope=run_config.scope,
        )
        for cell in cells:
            trial_paths = run_paths.trial(d)
            items.append(
                _FillWork(
                    decision_date=d,
                    cell=cell,
                    active_universe=active,
                    cell_path=trial_paths.cell(cell.name),
                )
            )
    return items


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


def _load_strategy_context(run_config: RunConfig):
    """Re-read pack prompts from the strategy path frozen in RunConfig."""
    return load_pack_context(load(run_config.strategy.path))


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
