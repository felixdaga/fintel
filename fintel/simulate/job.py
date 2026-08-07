"""One `fintel simulation` invocation: package × agent × market × K repeats.

The top of the fan-out. It loads and preflights the strategy package once,
freezes a `RunConfig` per repeat (package defaults + job overrides, so each run
is self-describing and reproducible from its config alone), and fans the K runs
out with bounded concurrency. A failed run is a failed run — it does not abort
the job, because the whole point of K repeats is to see the distribution, and
losing nine because one crashed defeats that.
"""

from __future__ import annotations

import logging
import time
from datetime import date as Date
from pathlib import Path

from fintel.agents.factory import preflight as agent_preflight
from fintel.environment.health import audit_job
from fintel.environment.progress import Progress
from fintel.market.calendar import TradingCalendar
from fintel.market.factory import build_data_sources, build_schedule, build_universe
from fintel.market.prefetch import prefetch as warm_cache
from fintel.market.prefetch import prefetch_window
from fintel.market.probe import probe as probe_sources
from fintel.market.settings import MarketConfig
from fintel.models.common import Symbol
from fintel.models.ids import new_job_id, run_id
from fintel.models.job import JobConfig, JobResult
from fintel.models.market import UniverseRef
from fintel.models.paths import JobPaths
from fintel.models.run import RunConfig, StrategyRef
from fintel.models.strategy import StrategyManifest
from fintel.simulate.artifacts import (
    write_health,
    write_job_config,
    write_job_result,
    write_prefetch,
)
from fintel.simulate.queue import map_parallel
from fintel.simulate.record import reduce_job
from fintel.simulate.run import run_run
from fintel.strategy import build_lock, load, preflight
from fintel.strategy.preflight import PreflightError

logger = logging.getLogger(__name__)


def run_job(
    job_config: JobConfig,
    *,
    market_config: MarketConfig | None = None,
    progress: Progress | None = None,
    quiet: bool = False,
) -> JobResult:
    """Load the package, preflight the *effective* config, fan out K runs, reduce.

    Preflight runs on the effective config (package defaults + job overrides),
    not the bare package: a job that overrides `data` to a no-key source should
    not fail preflight because the package's *default* data needs a key. The
    lock, by contrast, freezes the package's own identity (manifest text,
    mission, schema), which is what makes two runs of the same package the same
    package regardless of overrides.

    If `progress` is None a `Nerve` is constructed against the job folder, so
    every run has a live `run.log` + terminal stream by default. Pass a `Progress`
    (e.g. `NullProgress`) to override, e.g. in tests. `quiet` only affects the
    default Nerve's terminal line, not the log file.
    """
    started = time.perf_counter()
    started_at = _now_iso()
    market_config = market_config or MarketConfig.from_env(
        cache_root=Path(job_config.output_root) / "cache"
    )

    # Load the package, then preflight the effective config (after overrides).
    paths = load(job_config.strategy)
    manifest = paths.manifest
    effective = _effective_config(job_config, manifest)
    effective_manifest = manifest.model_copy(
        update={
            "data": effective["data"],
            "universe": effective["universe"],
            "decision": manifest.decision.model_copy(update={"schedule": effective["schedule"]}),
        }
    )
    from fintel.models.strategy import StrategyPaths

    preflight_result = preflight(StrategyPaths(root=paths.root, manifest=effective_manifest))
    preflight_result.raise_if_not_ok()

    # The agent side of preflight: can this adapter actually run (API key set,
    # CLI on PATH, profile configured)? Same fail-closed shape as the strategy
    # check above — one command should surface every reason a run can't start.
    agent_problems = agent_preflight(job_config.agent.name, **job_config.agent.options)
    if agent_problems:
        raise PreflightError(agent_problems)

    # The mission and output schema the strategy hands every agent — read once
    # here, not per cell. Preflight above already guarantees the mission file
    # exists; the schema is optional (a package may rely on submit_views' own
    # JSON schema alone).
    mission_text = paths.mission.read_text() if paths.mission.is_file() else ""
    output_schema_text = paths.output_schema.read_text() if paths.output_schema.is_file() else ""
    company_names: dict[str, str] = {}
    if paths.company_names.is_file():
        import json

        try:
            company_names = json.loads(paths.company_names.read_text())
        except (json.JSONDecodeError, ValueError):
            logger.warning("company_names.json is not valid JSON; ignored")

    # Lock the *package* identity (not the effective config).
    from fintel.market import catalog

    strategy_lock = build_lock(
        paths,
        catalog_sources=tuple(sorted(s.name for s in catalog.sources())),
        catalog_kinds=tuple(sorted(catalog.kinds())),
    )
    strategy_lock.write(paths.lock)

    job_id = job_config.job_id or new_job_id(
        strategy=manifest.name,
        agent=job_config.agent.name,
        model=job_config.agent.model.id or None,
        k_repeats=job_config.k_repeats,
    )
    job_paths = JobPaths.under(job_config.output_root, job_id)

    # The central nervous system, owned by the environment module. Two nerves
    # by default, two logs, one terminal stream:
    #   · a job nerve  → `<job>/job.log`   for job-level events (preflight, probe,
    #     prefetch, job_done) that happen before/after any run.
    #   · a run nerve  → `<run>/run.log`   for run-level events (run echo, trial,
    #     cell, live agent staging) — the live log the operator watches. The run
    #     echo is emitted here (not as a sibling echo.json).
    # If the caller passed a Progress (tests, or a custom sink) it receives
    # everything — both levels funnel into the one sink.
    if progress is None:
        from fintel.environment.nerve import Nerve

        job_progress = Nerve(
            run_root=job_paths.root,
            log_path=job_paths.root / "job.log",
            verbose=not quiet,
        )
        run_progress: Progress | None = None  # run_run builds its own run nerve
    else:
        job_progress = progress
        run_progress = progress
    write_job_config(job_paths.config, job_config)

    calendar = TradingCalendar()
    schedule = build_schedule(effective["schedule"], calendar=calendar)
    schedule_dates = [d.isoformat() for d in schedule.dates()]

    universe_snapshot = _universe_snapshot(effective["universe"], market_config, schedule_dates)

    job_progress.emit(
        "job_start",
        job_id=job_id,
        strategy=manifest.name,
        agent=job_config.agent.name,
        k_repeats=job_config.k_repeats,
    )
    job_progress.emit(
        "preflight_ok",
        n_dates=len(schedule_dates),
        n_symbols=len(universe_snapshot),
    )

    # Live reachability probe — once, run-level, before any cell. Preflight
    # above proved the declared world is resolvable (cheap, no fetch). This
    # proves it is *reachable*: a source can be declared and keyed yet 401 on
    # every call. Reuses the real read path (DataAccess.read with a synthetic
    # probe cell), so there is no second fetch implementation to drift. A
    # `failed` kind here stops the run at the gate — not three cells in.
    if job_config.prefetch and schedule_dates:
        probe_sources_for_probe = build_data_sources(effective["data"], config=market_config)
        probe_symbol = universe_snapshot[0] if universe_snapshot else None
        job_progress.emit(
            "probe_start",
            kinds=sorted(k for k in probe_sources_for_probe),
            symbol=probe_symbol or "AAPL",
            timeout_s=15,
        )
        probe_result = probe_sources(
            sources=probe_sources_for_probe,
            symbol=probe_symbol,
            cutoff=Date.fromisoformat(schedule_dates[0]),
        )
        for kp in probe_result.kinds:
            job_progress.emit(
                "probe_kind",
                kind=kp.kind,
                source=kp.source,
                status=kp.status,
                latency_ms=round(kp.latency_ms, 1),
                n=kp.n,
            )
        if probe_result.ok:
            job_progress.emit(
                "probe_ok",
                n_ok=len(probe_result.kinds),
                n_kinds=len(probe_result.kinds),
            )
        else:
            failed = [k.kind for k in probe_result.failed_kinds]
            job_progress.emit(
                "probe_failed",
                n_failed=len(failed),
                n_kinds=len(probe_result.kinds),
                failed_kinds=failed,
            )
            raise PreflightError(
                "data not extractable — probe failed for kind(s) "
                f"{failed}: {probe_result.to_dict()}"
            )

    # Warm the cache once for the whole job before K fan-out. The union over
    # all decision dates is the symbol set; the window covers the widest
    # lookback any bound kind declares. Without this, OpenClaw's parallel MCP
    # tool calls hit cold Massive fills and time out at the host's ~60s
    # per-request ceiling — the failure mode that made 0005 abstain.
    if job_config.prefetch and schedule_dates:
        try:
            decision_dates = [Date.fromisoformat(d) for d in schedule_dates]
            universe_obj = build_universe(effective["universe"], config=market_config)
            # HistoricalUniverse has union_over; StaticUniverse just has symbols.
            if hasattr(universe_obj, "union_over"):
                prefetch_symbols = list(universe_obj.union_over(decision_dates))
            else:
                prefetch_symbols = list(universe_obj.active_at(decision_dates[0]))
            prefetch_sources = build_data_sources(effective["data"], config=market_config)
            pfrom, pthrough = prefetch_window(decision_dates, prefetch_sources, effective["data"])
            job_progress.emit(
                "preflight_start",
                n_symbols=len(prefetch_symbols),
                kinds=sorted(k for k in prefetch_sources),
                from_date=pfrom.isoformat(),
                through_date=pthrough.isoformat(),
            )
            presult = warm_cache(
                symbols=prefetch_symbols,
                sources=prefetch_sources,
                from_date=pfrom,
                through_date=pthrough,
                workers=job_config.prefetch_workers,
                decision_dates=decision_dates,
            )
            write_prefetch(job_paths.root / "prefetch.json", presult)
            job_progress.emit(
                "preflight_done",
                n_warmed=len(presult.warmed),
                n_failed=len(presult.failed),
                elapsed_ms=presult.elapsed_ms,
            )
        except Exception as exc:
            # Prefetch is an optimisation, not a gate. A failure here (e.g. a
            # universe that can't be resolved offline) must not abort the job
            # — cells will cold-fill and the access log will grade the result.
            logger.warning("prefetch failed: %s", exc)
            job_progress.emit("preflight_done", n_warmed=0, n_failed=1, elapsed_ms=0)

    strategy_ref = StrategyRef(
        name=manifest.name,
        path=str(paths.root),
        digest=strategy_lock.strategy_digest,
    )

    def _run_one_repeat(k: int) -> JobResult | None:
        try:
            run_config = RunConfig(
                run_id=run_id(job_id, k),
                job_id=job_id,
                k_index=k,
                k_repeats=job_config.k_repeats,
                created_at=_now_iso(),
                strategy=strategy_ref,
                agent=job_config.agent,
                scope=manifest.decision.scope,
                universe=effective["universe"],
                universe_symbols=universe_snapshot,
                schedule=effective["schedule"],
                schedule_dates=schedule_dates,
                data=effective["data"],
                scoring=manifest.scoring,
            )
            return run_run(
                run_config=run_config,
                market_config=market_config,
                paths=job_paths.run(k),
                cell_concurrency=job_config.cell_concurrency,
                trial_concurrency=job_config.trial_concurrency,
                shared_concurrency=job_config.shared_concurrency,
                memory_on=_memory_on(manifest),
                feedback_on=_feedback_on(manifest),
                mission_text=mission_text,
                output_schema_text=output_schema_text,
                company_names=company_names,
                strategy_description=manifest.description,
                progress=run_progress,
                quiet=quiet,
            )
        except Exception as exc:
            # A run that can't start is a failed run, recorded but not fatal.
            from fintel.models.run import RunResult

            return RunResult(
                run_id=run_id(job_id, k),
                job_id=job_id,
                k_index=k,
                status="failed",
                started_at=started_at,
                finished_at=_now_iso(),
                error=f"{type(exc).__name__}: {exc}",
            )

    run_results = map_parallel(
        _run_one_repeat,
        list(range(1, job_config.k_repeats + 1)),
        bound=job_config.resolve_run_concurrency(),
    )
    runs = [r for r in run_results if r is not None]

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    result = reduce_job(
        job_id=job_id,
        strategy=manifest.name,
        agent=job_config.agent.name,
        k_repeats=job_config.k_repeats,
        runs=runs,
        started_at=started_at,
        finished_at=_now_iso(),
    )
    # Full-job re-audit from disk (MCP + parent events) — the authoritative
    # health rollup for the CLI and for `fintel health <job>`.
    from fintel.simulate.cell import expect_tools

    job_health = audit_job(job_paths.root, expect_tools=expect_tools(job_config.agent))
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
    job_progress.emit("job_health", **{k: job_health[k] for k in ("status", "n_cells", "issues")})
    write_job_result(job_paths.result, result)
    job_progress.emit(
        "job_done",
        job_id=job_id,
        status=result.status,
        health=result.health,
        elapsed_ms=elapsed_ms,
    )
    return result


def _effective_config(job: JobConfig, manifest: StrategyManifest) -> dict[str, object]:
    """Package defaults, overridden where the job says otherwise."""
    universe = job.universe or manifest.universe
    schedule = job.schedule or manifest.decision.schedule
    data = job.data if job.data is not None else manifest.data
    return {"universe": universe, "schedule": schedule, "data": data}


def _memory_on(manifest: StrategyManifest) -> bool:
    """Whether this run carries portfolio/memory state forward across dates.

    Memory isn't wired yet, so this is False — which means dates *could* run
    concurrently. We keep the default `trial_concurrency=1` anyway (sequential
    is the safe default), and this guard is the hook that will force sequential
    the moment a strategy declares memory. The shape mirrors delorean's
    `concurrent_dates` safety check: parallel dates are only legal when dates
    are truly independent, i.e. memory off.
    """
    # When memory lands, this becomes something like:
    #   return getattr(manifest, "memory_level", "off") != "off"
    return False


def _feedback_on(manifest: StrategyManifest) -> bool:
    """Whether this run feeds prior-date outcomes into later dates.

    Same independence rule as memory: shared_concurrency / parallel dates are
    only legal when feedback is off. Not wired yet — stub returns False.
    """
    # When feedback lands, this becomes something like:
    #   return getattr(manifest, "feedback", "off") != "off"
    return False


def _universe_snapshot(
    universe: UniverseRef, config: MarketConfig, schedule_dates: list[str]
) -> list[Symbol]:
    """A reproducible snapshot of the universe for the run config.

    For a historical universe the active set changes per date; the snapshot is
    taken at the first decision date (or today if no schedule), and per-date
    resolution still happens in `run_run`. The snapshot makes the run config
    self-describing without freezing a misleading single list for a dynamic
    universe — it's labelled as a snapshot, not the truth.
    """
    ref_date = Date.fromisoformat(schedule_dates[0]) if schedule_dates else Date.today()
    try:
        universe_obj = build_universe(universe, config=config)
        return list(universe_obj.active_at(ref_date))
    except Exception:
        return []


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")
