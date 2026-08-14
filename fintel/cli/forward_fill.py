"""CLI handler for ``fintel forward-fill``.

Add new decision dates to a finished run. Loads the frozen run config,
computes new dates from the schedule (or takes an explicit list), runs the
agent for the whole universe on each new date, and merges the new trials
into the existing run + job results on disk.
"""

from __future__ import annotations

import sys
from datetime import date as Date


def run_forward_fill_cli(args) -> int:
    from pathlib import Path

    from fintel.environment.nerve import Nerve
    from fintel.environment.progress import NullProgress
    from fintel.market.settings import MarketConfig
    from fintel.models.paths import JobPaths
    from fintel.simulate.forward_fill import run_forward_fill

    job_paths = JobPaths.under(args.output_root, args.job_id)
    if not job_paths.root.is_dir():
        print(f"job not found: {job_paths.root}", file=sys.stderr)
        return 1

    through = None
    if args.through:
        through = Date.fromisoformat(args.through)

    explicit_dates = None
    if args.dates:
        explicit_dates = [Date.fromisoformat(d.strip()) for d in args.dates.split(",") if d.strip()]

    schedule_override = None
    if getattr(args, "schedule_kind", None):
        from fintel.models.market import ScheduleRef

        params: dict = {}
        if getattr(args, "schedule_start", None):
            params["start"] = args.schedule_start
        if getattr(args, "schedule_anchor", None):
            params["anchor"] = args.schedule_anchor
        schedule_override = ScheduleRef(kind=args.schedule_kind, **params)

    if args.quiet:
        progress = NullProgress()
    else:
        progress = Nerve(
            run_root=job_paths.root,
            log_path=job_paths.root / "forward_fill.log",
            verbose=True,
        )

    market_config = MarketConfig.from_env(cache_root=Path(args.output_root) / "cache")

    try:
        report = run_forward_fill(
            job_id=args.job_id,
            run_index=args.run_index,
            through=through,
            dates=explicit_dates,
            cell_concurrency=args.cell_concurrency,
            output_root=args.output_root,
            market_config=market_config,
            schedule_override=schedule_override,
            progress=progress,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if report.n_new_dates == 0:
        print(f"forward-fill {args.job_id}/r{args.run_index}: up to date (no new dates)")
        return 0

    print(
        f"forward-fill {args.job_id}/r{args.run_index}: "
        f"{report.n_new_dates} new dates, "
        f"{report.n_cells_run} cells run, "
        f"{report.n_ok} ok, "
        f"{report.n_failed} failed"
    )
    if report.new_dates:
        print(f"  new dates: {', '.join(report.new_dates)}")

    if report.n_failed > 0:
        return 2
    return 0
