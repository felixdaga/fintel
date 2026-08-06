"""One of K repeats: the frozen effective config, fanned out across dates.

A run takes the `RunConfig` (the effective world after package defaults + job
overrides — self-describing, so a finished run is reproducible from this file
alone) and walks the schedule. Each decision date is one trial. The universe is
resolved *at the decision date*, because an index's membership changes and a
cell must be judged against the world as it was.

Data sources are built once per run and shared across its trials: the cache is
the expensive part, and a source that has fetched a symbol's price history for
one date should not re-fetch it for the next.

Concurrency modes:

  · Nested (default): `trial_concurrency` dates × `cell_concurrency` tickers.
  · Flat: `shared_concurrency=N` keeps N cells in flight across all dates —
    a finished cell rolls to the next (date, ticker). Requires independent
    dates (memory and feedback off).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date as Date

from fintel.environment.access import dedup_sources
from fintel.environment.cell import Cell
from fintel.environment.factory import RuntimeConfig, cells_for
from fintel.environment.progress import Progress
from fintel.market.calendar import TradingCalendar
from fintel.market.factory import (
    build_data_sources,
    build_schedule,
    build_universe,
)
from fintel.market.settings import MarketConfig
from fintel.models.common import Symbol
from fintel.models.decision import AgentResponse
from fintel.models.paths import RunPaths, TrialPaths
from fintel.models.run import RunConfig, RunResult
from fintel.models.trial import CellResult, TrialConfig, TrialResult
from fintel.simulate.artifacts import (
    write_run_config,
    write_run_result,
    write_trial_result,
)
from fintel.simulate.cell import CellOutcome, run_cell
from fintel.simulate.queue import map_parallel
from fintel.simulate.record import reduce_run, reduce_trial
from fintel.simulate.trial import finalize_trial, run_trial

_log = logging.getLogger(__name__)


def run_run(
    *,
    run_config: RunConfig,
    market_config: MarketConfig,
    paths: RunPaths,
    cell_concurrency: int | None = None,
    trial_concurrency: int = 1,
    shared_concurrency: int | None = None,
    memory_on: bool = False,
    feedback_on: bool = False,
    mission_text: str = "",
    output_schema_text: str = "",
    company_names: dict[str, str] | None = None,
    strategy_description: str = "",
    progress: Progress | None = None,
    quiet: bool = True,
) -> RunResult:
    """Execute one repeat: build the world, fan out trials, reduce.

    `cell_concurrency=None` means auto: resolve to the universe size at each
    date, so all tickers decide at once (the "10 concurrent" case). An explicit
    int caps it.

    `trial_concurrency` defaults to 1 — dates run sequentially because a date's
    session carries the prior date's portfolio + memory. The `memory_on` guard
    forces this to 1 when memory is on, so a misconfigured job that asked for
    parallel dates can't race on shared state. This is delorean's
    `concurrent_dates` safety check, carried forward.

    `shared_concurrency` is a flat pool of N cells across all dates. When set,
    it replaces the nested cell × trial fan-out. Blocked when memory or
    feedback couples dates.

    `mission_text`/`output_schema_text` are the strategy pack's mission.md and
    output_schema.json (read once by `run_job`, passed down here rather than
    re-read per trial). They flow to every cell's agent unchanged; a fingerprint
    of them (plus the agent config) is sealed into `config.json` before any
    trial runs, so a crash still leaves a readable identity for the run.
    """
    # The run nerve: the live emit surface for this run, writing `<run>/run.log`
    # and streaming to the terminal. Built here, not in run_job, so each of K
    # repeats gets its own run.log in its own run folder. A caller-supplied
    # `progress` (tests, custom sink) overrides it.
    if progress is None:
        from fintel.environment.nerve import Nerve

        progress = Nerve(run_root=paths.root, verbose=not quiet)
    started = time.perf_counter()
    started_at = _now_iso()
    progress.emit(
        "run_start",
        run_id=run_config.run_id,
        k_index=run_config.k_index,
        k_repeats=run_config.k_repeats,
    )

    dates_coupled = memory_on or feedback_on
    if shared_concurrency is not None and dates_coupled:
        raise ValueError(
            "shared_concurrency requires independent dates "
            "(memory and feedback must be off); "
            f"got memory_on={memory_on}, feedback_on={feedback_on}"
        )

    if memory_on and trial_concurrency > 1:
        _log.warning(
            "trial_concurrency=%d ignored — memory is on, so dates must be "
            "sequential (each carries the prior date's state); forcing 1.",
            trial_concurrency,
        )
        trial_concurrency = 1

    if shared_concurrency is not None and (trial_concurrency > 1 or cell_concurrency is not None):
        _log.info(
            "shared_concurrency=%d active — nested cell/trial concurrency ignored.",
            shared_concurrency,
        )

    calendar = TradingCalendar()
    schedule = build_schedule(run_config.schedule, calendar=calendar)
    decision_dates = schedule.dates()

    universe = build_universe(run_config.universe, config=market_config)
    sources = build_data_sources(run_config.data, config=market_config)
    # Single-flight the shared sources so a thundering herd of concurrent cells
    # issuing the same (query, cutoff) fetch — classically the symbol-independent
    # macro bundle, fetched once per cell — collapses to one network call. Applied
    # to every kind/source here so all cell reads dedup against one in-flight table.
    sources = dedup_sources(sources)

    fingerprint = _build_fingerprint(run_config, sources, mission_text)
    # Freeze identity once sources + fingerprint are known: config carries the
    # digest (no sibling fingerprint.json / hollow lock.json).
    write_run_config(
        paths.config,
        run_config.model_copy(update={"fingerprint": fingerprint.to_dict()}),
    )

    # Run echo: every input gathered before any cell runs. Emitted to the nerve
    # (terminal + run.log) — not persisted as a sibling echo.json. Full prompt
    # text still lives in the strategy pack; the log keeps the scannable summary.
    from fintel.environment.echo import build_echo, render_echo

    echo = build_echo(
        run_config=run_config,
        strategy_description=strategy_description,
        sources=sources,
        mission_text=mission_text,
        output_schema_text=output_schema_text,
        fingerprint=fingerprint.to_dict(),
    )
    progress.emit("run_echo", echo=render_echo(echo))

    runtime = RuntimeConfig(
        session_root=paths.root / "sessions",
        trace=True,
        reset_sessions=True,
    )

    agent_spec = _agent_spec(run_config)

    if shared_concurrency is not None:
        trials = _run_shared(
            run_config=run_config,
            decision_dates=decision_dates,
            universe=universe,
            sources=sources,
            agent_spec=agent_spec,
            runtime=runtime,
            paths=paths,
            shared_concurrency=shared_concurrency,
            mission_text=mission_text,
            output_schema_text=output_schema_text,
            company_names=company_names or {},
            market_config=market_config,
            progress=progress,
        )
    else:
        trials = _run_nested(
            run_config=run_config,
            decision_dates=decision_dates,
            universe=universe,
            sources=sources,
            agent_spec=agent_spec,
            runtime=runtime,
            paths=paths,
            cell_concurrency=cell_concurrency,
            trial_concurrency=trial_concurrency,
            mission_text=mission_text,
            output_schema_text=output_schema_text,
            company_names=company_names or {},
            market_config=market_config,
            progress=progress,
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    result = reduce_run(
        run_id=run_config.run_id,
        job_id=run_config.job_id,
        k_index=run_config.k_index,
        trials=trials,
        started_at=started_at,
        finished_at=_now_iso(),
        duration_ms=elapsed_ms,
    )
    write_run_result(paths.result, result)
    progress.emit(
        "run_done",
        run_id=run_config.run_id,
        status=result.status,
        n_views=result.n_views,
        health=result.health,
    )
    return result


def _run_nested(
    *,
    run_config: RunConfig,
    decision_dates: list[Date],
    universe,
    sources: dict,
    agent_spec,
    runtime: RuntimeConfig,
    paths: RunPaths,
    cell_concurrency: int | None,
    trial_concurrency: int,
    mission_text: str,
    output_schema_text: str,
    company_names: dict[str, str],
    market_config: MarketConfig,
    progress: Progress,
) -> list[TrialResult]:
    """Classic date × ticker fan-out (trial_concurrency × cell_concurrency)."""

    def _run_one_trial(decision_date: Date) -> TrialResult | None:
        try:
            active = universe.active_at(decision_date)
            trial_config = TrialConfig(
                run_id=run_config.run_id,
                decision_date=decision_date,
                universe=active,
                scope=run_config.scope,
            )
            cell_bound = max(1, len(active)) if cell_concurrency is None else cell_concurrency
            return run_trial(
                trial_config=trial_config,
                sources=sources,
                agent_spec=agent_spec,
                runtime=runtime,
                paths=paths.trial(decision_date),
                cell_concurrency=cell_bound,
                mission_text=mission_text,
                output_schema_text=output_schema_text,
                company_names=company_names,
                market_config=market_config,
                progress=progress,
            )
        except Exception as exc:
            return TrialResult(
                decision_date=decision_date,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    trial_results = map_parallel(_run_one_trial, decision_dates, bound=trial_concurrency)
    return [t for t in trial_results if t is not None]


@dataclass(frozen=True)
class _SharedWork:
    decision_date: Date
    cell: Cell
    universe: list[Symbol]
    trial_paths: TrialPaths


def _run_shared(
    *,
    run_config: RunConfig,
    decision_dates: list[Date],
    universe,
    sources: dict,
    agent_spec,
    runtime: RuntimeConfig,
    paths: RunPaths,
    shared_concurrency: int,
    mission_text: str,
    output_schema_text: str,
    company_names: dict[str, str],
    market_config: MarketConfig,
    progress: Progress,
) -> list[TrialResult]:
    """Flat pool: keep `shared_concurrency` cells in flight across all dates."""
    work: list[_SharedWork] = []
    # Per-date bookkeeping so we can finalize in schedule order after the pool.
    cells_by_date: dict[Date, list[Cell]] = {}
    started_at_by_date: dict[Date, str] = {}
    paths_by_date: dict[Date, TrialPaths] = {}
    early: dict[Date, TrialResult] = {}

    for decision_date in decision_dates:
        started_at = _now_iso()
        try:
            active = list(universe.active_at(decision_date))
            trial_paths = paths.trial(decision_date)
            cells = cells_for(
                run_id=run_config.run_id,
                decision_date=decision_date,
                symbols=active,
                scope=run_config.scope,
            )
            progress.emit(
                "trial_start",
                decision_date=decision_date.isoformat(),
                n_symbols=len(active),
                n_cells=len(cells),
            )
            if not cells:
                result = reduce_trial(
                    decision_date, [], started_at=started_at, finished_at=started_at
                )
                write_trial_result(trial_paths.result, result)
                progress.emit(
                    "trial_done",
                    decision_date=decision_date.isoformat(),
                    status=result.status,
                    n_views=0,
                )
                early[decision_date] = result
                continue
            cells_by_date[decision_date] = cells
            started_at_by_date[decision_date] = started_at
            paths_by_date[decision_date] = trial_paths
            for cell in cells:
                work.append(
                    _SharedWork(
                        decision_date=decision_date,
                        cell=cell,
                        universe=active,
                        trial_paths=trial_paths,
                    )
                )
        except Exception as exc:
            early[decision_date] = TrialResult(
                decision_date=decision_date,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _run_one(item: _SharedWork) -> CellOutcome | None:
        try:
            return run_cell(
                cell=item.cell,
                sources=sources,
                universe=item.universe,
                agent_spec=agent_spec,
                runtime=runtime,
                cell_path=item.trial_paths.cell(item.cell.name),
                mission_text=mission_text,
                output_schema_text=output_schema_text,
                company_names=company_names,
                market_config=market_config,
                progress=progress,
            )
        except Exception:
            return CellOutcome(
                result=CellResult(
                    cell=item.cell.name,
                    symbols=list(item.cell.symbols),
                    status="failed",
                    error="cell executor raised",
                    health="broken",
                    health_issues=["cell executor raised"],
                ),
                response=AgentResponse(views={}, outcome="crashed", detail="cell executor raised"),
            )

    outcomes = map_parallel(_run_one, work, bound=shared_concurrency) if work else []

    # Group outcomes back onto their dates (work order preserved by map_parallel).
    outcomes_by_date: dict[Date, list[CellOutcome | None]] = {d: [] for d in cells_by_date}
    for item, outcome in zip(work, outcomes):
        outcomes_by_date[item.decision_date].append(outcome)

    trials: list[TrialResult] = []
    for decision_date in decision_dates:
        if decision_date in early:
            trials.append(early[decision_date])
            continue
        cells = cells_by_date[decision_date]
        trials.append(
            finalize_trial(
                decision_date=decision_date,
                cells=cells,
                outcomes=outcomes_by_date[decision_date],
                paths=paths_by_date[decision_date],
                started_at=started_at_by_date[decision_date],
                progress=progress,
            )
        )
    return trials


def _agent_spec(run_config: RunConfig):
    """The agent spec is carried on the run config. Resolved here so the trial
    layer doesn't import `models.run` for it."""
    return run_config.agent


def _build_fingerprint(run_config: RunConfig, sources: dict, mission_text: str):
    """The run's reproducibility digest: agent identity, model, channel, prompt
    hash, data kinds and adapter params — everything that should make two runs
    identical. Built from the declared config, not a live agent instance, so
    fingerprinting never has the side effects (HTTP client setup, profile
    reads) that building the real adapter does.

    PIT enforcement mode (and the deny list for cli_deny adapters) is part of
    the digest: a run that still had web/fs is not comparable to one that didn't.
    """
    from fintel.agents.factory import AGENTS
    from fintel.agents.fingerprint import fingerprint as build_fingerprint
    from fintel.agents.pit_policy import CLAUDE_CODE_DENY, OPENCLAW_DENY
    from fintel.utils.import_path import resolve

    agent = run_config.agent
    params = dict(agent.options)
    channel = str(params.pop("channel", ""))
    version = str(params.pop("version", "1"))

    target = AGENTS.get(agent.name, agent.name if ":" in agent.name else None)
    pit_enforcement = "unknown"
    pit_deny: list[str] = []
    if target is not None:
        cls = resolve(target)
        pit_enforcement = str(getattr(cls, "pit_enforcement", "unknown"))
        if pit_enforcement == "cli_deny":
            if agent.name == "openclaw":
                pit_deny = list(OPENCLAW_DENY)
            elif agent.name == "claude-code":
                pit_deny = list(CLAUDE_CODE_DENY)

    params = {
        **params,
        "pit_enforcement": pit_enforcement,
        "pit_deny": pit_deny,
    }
    return build_fingerprint(
        agent_name=agent.name,
        agent_version=version,
        model=agent.model.id or "",
        channel=channel,
        prompt=mission_text,
        data_kinds=tuple(sorted(sources)),
        adapter_params=params,
    )


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")
