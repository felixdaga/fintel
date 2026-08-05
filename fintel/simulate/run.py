"""One of K repeats: the frozen effective config, fanned out across dates.

A run takes the `RunConfig` (the effective world after package defaults + job
overrides — self-describing, so a finished run is reproducible from this file
alone) and walks the schedule. Each decision date is one trial. The universe is
resolved *at the decision date*, because an index's membership changes and a
cell must be judged against the world as it was.

Data sources are built once per run and shared across its trials: the cache is
the expensive part, and a source that has fetched a symbol's price history for
one date should not re-fetch it for the next.
"""

from __future__ import annotations

import time
from datetime import date as Date

from fintel.environment.factory import RuntimeConfig
from fintel.environment.access import dedup_sources
from fintel.market.calendar import TradingCalendar
from fintel.market.factory import (
    build_data_sources,
    build_schedule,
    build_universe,
)
from fintel.market.settings import MarketConfig
from fintel.models.common import Symbol
from fintel.models.paths import RunPaths
from fintel.models.run import RunConfig, RunResult
from fintel.models.trial import TrialConfig
from fintel.environment.progress import Progress
from fintel.simulate.artifacts import (
    write_run_config,
    write_run_result,
)
from fintel.simulate.queue import map_parallel
from fintel.simulate.record import reduce_run
from fintel.simulate.trial import run_trial


def run_run(
    *,
    run_config: RunConfig,
    market_config: MarketConfig,
    paths: RunPaths,
    cell_concurrency: int | None = None,
    trial_concurrency: int = 1,
    memory_on: bool = False,
    mission_text: str = "",
    output_schema_text: str = "",
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

    if memory_on and trial_concurrency > 1:
        import logging

        logging.getLogger(__name__).warning(
            "trial_concurrency=%d ignored — memory is on, so dates must be "
            "sequential (each carries the prior date's state); forcing 1.",
            trial_concurrency,
        )
        trial_concurrency = 1

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

    def _run_one_trial(decision_date: Date) -> RunResult | None:
        try:
            active = universe.active_at(decision_date)
            trial_config = TrialConfig(
                run_id=run_config.run_id,
                decision_date=decision_date,
                universe=active,
                scope=run_config.scope,
            )
            # Resolve auto cell concurrency to this date's universe size, so
            # "all tickers at once" tracks a changing universe without a
            # static knob.
            cell_bound = (
                max(1, len(active)) if cell_concurrency is None else cell_concurrency
            )
            return run_trial(
                trial_config=trial_config,
                sources=sources,
                agent_spec=_agent_spec(run_config),
                runtime=runtime,
                paths=paths.trial(decision_date),
                cell_concurrency=cell_bound,
                mission_text=mission_text,
                output_schema_text=output_schema_text,
                market_config=market_config,
                progress=progress,
            )
        except Exception as exc:
            # A trial that can't even start (bad universe resolution) is a
            # failed trial, not a failed run. The run continues with the rest.
            from fintel.models.trial import TrialResult

            return TrialResult(
                decision_date=decision_date,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    trial_results = map_parallel(_run_one_trial, decision_dates, bound=trial_concurrency)
    trials = [t for t in trial_results if t is not None]

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
