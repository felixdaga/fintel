"""CLI handler for ``fintel backfill``.

Rerun error cells from a finished job.  Loads the frozen run config,
isolates cells with non-ok/non-skipped status, reruns them in a flat pool,
and re-reduces the affected trials + run + job summaries on disk.
"""

from __future__ import annotations

import sys


def run_backfill_cli(args) -> int:
    from pathlib import Path

    from fintel.environment.nerve import Nerve
    from fintel.environment.progress import NullProgress
    from fintel.market.settings import MarketConfig
    from fintel.models.paths import JobPaths
    from fintel.simulate.backfill import run_backfill

    job_paths = JobPaths.under(args.output_root, args.job_id)
    if not job_paths.root.is_dir():
        print(f"job not found: {job_paths.root}", file=sys.stderr)
        return 1

    if args.quiet:
        progress = NullProgress()
    else:
        progress = Nerve(
            run_root=job_paths.root,
            log_path=job_paths.root / "backfill.log",
            verbose=True,
        )

    market_config = MarketConfig.from_env(
        cache_root=Path(args.output_root) / "cache"
    )

    try:
        report = run_backfill(
            job_id=args.job_id,
            run_index=args.run_index,
            cell_concurrency=args.cell_concurrency,
            output_root=args.output_root,
            market_config=market_config,
            progress=progress,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"backfill {args.job_id}/r{args.run_index}: "
        f"{report.n_error_cells} error cells, "
        f"{report.n_reran} reran, "
        f"{report.n_fixed} fixed, "
        f"{report.n_still_failed} still failing"
    )
    if report.affected_dates:
        print(f"  affected dates: {', '.join(report.affected_dates)}")

    if report.n_still_failed > 0:
        return 2
    return 0
