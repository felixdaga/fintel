"""``fintel simulation`` — build a JobConfig and run it with live progress.

One command, all features: preflight + reachability probe, run echo, live
per-cell staging (reasoning turns / tool calls), per-run isolation, and an
in-place dashboard. Pass ``--no-watch`` (or run non-interactively) for the old
synchronous verbose-line behavior.
"""

from __future__ import annotations

import sys
import threading
from argparse import Namespace
from pathlib import Path

from fintel.models.agent import AgentSpec, ModelSpec
from fintel.models.job import JobConfig
from fintel.models.market import ScheduleRef, UniverseRef


def _parse_opts(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--agent-opt must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def run_simulation(args: Namespace) -> int:
    package = Path(args.package).resolve()
    if not package.is_dir():
        raise SystemExit(f"package not found: {package}")

    if not args.no_bootstrap:
        from fintel.utils.secrets import bootstrap_env

        bootstrap_env()

    options = _parse_opts(args.agent_opt)
    agent = AgentSpec(
        name=args.agent,
        model=ModelSpec(id=args.model) if args.model else ModelSpec(),
        options=options,
    )

    universe = None
    if args.universe:
        symbols = [s.strip() for s in args.universe.split(",") if s.strip()]
        universe = UniverseRef(symbols=symbols)

    schedule = None
    if args.dates:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
        schedule = ScheduleRef(kind="custom_dates", dates=dates)

    job = JobConfig(
        job_id=args.job_id or "",  # filled below if empty
        strategy=str(package),
        agent=agent,
        k_repeats=args.k_repeats,
        max_concurrent=args.max_concurrent,
        cell_concurrency=args.cell_concurrency,
        trial_concurrency=args.trial_concurrency,
        shared_concurrency=args.shared_concurrency,
        output_root=str(Path(args.output_root).resolve()),
        universe=universe,
        schedule=schedule,
    )
    if not job.job_id:
        from fintel.models.ids import new_job_id

        job = job.model_copy(
            update={"job_id": new_job_id(strategy=package.name, agent=args.agent)}
        )

    from fintel.market.settings import MarketConfig
    from fintel.simulate import run_job

    cache_root = (
        Path(args.cache_root).expanduser()
        if args.cache_root
        else Path(job.output_root) / "cache"
    )
    market = MarketConfig.from_env(cache_root=cache_root)
    if args.offline:
        market = MarketConfig(
            cache_root=cache_root,
            offline=True,
            massive_api_key=market.massive_api_key,
            brave_api_key=market.brave_api_key,
        )
    if args.no_prefetch:
        job = job.model_copy(update={"prefetch": False})
    if args.prefetch_workers != 8:
        job = job.model_copy(update={"prefetch_workers": args.prefetch_workers})

    job_root = Path(job.output_root) / job.job_id
    use_watch = (not args.no_watch) and sys.stdout.isatty() and not args.quiet

    if use_watch:
        # One CLI, all features: background job (quiet) + live multi-track dashboard.
        # One track per repeat (r1..rK); preflight/probe shown as a header.
        result = _run_with_dashboard(job, market, job_root, getattr(args, "watch_mode", "auto"))
    else:
        # Synchronous: verbose nerve lines (or quiet), no dashboard. Used for
        # --no-watch, non-tty/CI, and --quiet.
        result = run_job(job, market_config=market, quiet=args.quiet)

    _print_decisions(job_root)

    # Non-zero exit when harness or job failed — so CI / scripts can gate.
    if result.health == "broken" or result.status == "failed":
        return 1
    if result.status == "partial" or result.health == "degraded":
        return 2
    return 0


def _run_with_dashboard(job: JobConfig, market, job_root: Path, watch_mode: str = "auto"):
    """Run the job in a background thread (quiet — logs only) and show the live
    in-place dashboard in the foreground. One track per repeat (r1..rK); the
    job-level preflight/probe is shown as a shared header drained from job.log.
    Returns the JobResult."""
    from fintel.cli.watch import watch_run_logs
    from fintel.simulate import run_job

    run_logs = [job_root / f"r{k}" / "run.log" for k in range(1, job.k_repeats + 1)]
    tags = [f"r{k}" for k in range(1, job.k_repeats + 1)]
    job_log = job_root / "job.log"
    box: dict = {}

    def _runner() -> None:
        # No progress= passed → run_job uses per-run nerves (r{k}/run.log) + a
        # job nerve (job.log). quiet=True so the dashboard is the only display.
        box["result"] = run_job(job, market_config=market, quiet=True)

    t = threading.Thread(target=_runner, name="fintel-job", daemon=False)
    t.start()
    # Blocks until every run reports done (or 'q'/Ctrl-C). Preflight header from job.log.
    watch_run_logs(run_logs, tags=tags, job_log=job_log, mode=watch_mode)
    t.join()
    return box["result"]


def _print_decisions(job_root: Path) -> None:
    trials = job_root / "r1" / "trials"
    if not trials.is_dir():
        return
    for trial_dir in sorted(trials.iterdir()):
        from fintel.cli.present import print_decision_block

        print_decision_block(trial_dir)
