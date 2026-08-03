"""``fintel runs list|show`` — browse finished job artifacts."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from fintel.cli.present import job_summary_line, print_job_artifacts


def run_runs(args: Namespace) -> int:
    root = Path(args.output_root)
    if args.runs_command == "list":
        if not root.is_dir():
            print(f"(no jobs under {root})")
            return 0
        for path in sorted(p for p in root.iterdir() if p.is_dir()):
            print(job_summary_line(path))
        return 0

    if args.runs_command == "show":
        job = root / args.job_id
        if not job.is_dir():
            raise SystemExit(f"job not found: {job}")
        print_job_artifacts(job)
        return 0

    if args.runs_command == "watch":
        from fintel.cli.watch import run_watch

        return run_watch(args)

    return 2
