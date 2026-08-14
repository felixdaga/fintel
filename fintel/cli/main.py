"""``fintel`` entry point — parse flags, call libraries, print progress."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fintel",
        description="Agent evaluation where the benchmark is an investment outcome.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    simulation = sub.add_parser(
        "simulation", help="Run a strategy package simulation against an agent"
    )
    simulation.add_argument("package", help="Path to strategy package directory")
    simulation.add_argument("--agent", required=True, help="Agent name (openclaw, llm, …)")
    simulation.add_argument(
        "--agent-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Agent option (repeatable), e.g. profile=delorean",
    )
    simulation.add_argument("--job-id", default=None, help="Stable job id (default: generated)")
    simulation.add_argument("--k", type=int, default=1, dest="k_repeats", help="K repeats")
    simulation.add_argument(
        "--universe",
        default=None,
        help="Comma-separated symbols (overrides package universe)",
    )
    simulation.add_argument(
        "--dates",
        default=None,
        help="Comma-separated decision dates YYYY-MM-DD (overrides package schedule)",
    )
    simulation.add_argument(
        "--cell-concurrency",
        type=int,
        default=None,
        help="Concurrent cells per date (default: auto=universe size; use 1 for openclaw)",
    )
    simulation.add_argument(
        "--trial-concurrency",
        type=int,
        default=1,
        help="Concurrent dates within a run (default 1)",
    )
    simulation.add_argument(
        "--shared-concurrency",
        type=int,
        default=None,
        help=(
            "Job-wide cell pool across dates and K repeats (keep N cells in "
            "flight; slots roll to the next run/date/ticker as they free). "
            "Same concurrency primitive as --cell-concurrency / "
            "--trial-concurrency, flattened. Replaces cell×trial fan-out; "
            "blocked when memory/feedback is on"
        ),
    )
    simulation.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Concurrent K repeats (default: auto = all K)",
    )
    simulation.add_argument(
        "--output-root",
        default="runs",
        help="Directory for job artifacts (default: runs)",
    )
    simulation.add_argument(
        "--model",
        default="",
        help="Model id pin (empty = agent/profile default)",
    )
    simulation.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live progress (still writes job.log)",
    )
    simulation.add_argument(
        "--no-watch",
        action="store_true",
        help="Disable the live in-place dashboard (run synchronously with verbose lines)",
    )
    simulation.add_argument(
        "--watch-mode",
        choices=["auto", "alt", "stream"],
        default="auto",
        help="Dashboard render mode: auto (default; stream in Cursor, alt elsewhere), "
        "alt (full-screen, needs a real terminal), stream (in-place without alt screen)",
    )
    simulation.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Skip cache warm-up (cold fills at cell time; faster start, slower cells)",
    )
    simulation.add_argument(
        "--prefetch-workers",
        type=int,
        default=8,
        help="Parallel symbol workers for cache warm-up (default 8)",
    )
    simulation.add_argument(
        "--cache-root",
        default=None,
        help="Cache directory (default: <output-root>/cache). "
        "Shared across jobs; one central cache root.",
    )
    simulation.add_argument(
        "--offline",
        action="store_true",
        help="Cache-only; a miss is an error instead of a network call",
    )
    simulation.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Do not load keys from .env/keys.env",
    )

    backfill = sub.add_parser("backfill", help="Rerun error cells from a finished job")
    backfill.add_argument("job_id", help="Job id under --output-root")
    backfill.add_argument(
        "--run",
        type=int,
        default=1,
        dest="run_index",
        help="Which repeat (rK) to backfill (default 1)",
    )
    backfill.add_argument(
        "--cell-concurrency",
        type=int,
        default=1,
        help="Flat pool size for rerunning error cells (default 1)",
    )
    backfill.add_argument(
        "--output-root",
        default="runs",
        help="Directory for job artifacts (default: runs)",
    )
    backfill.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live progress (still writes logs)",
    )

    health = sub.add_parser("health", help="Audit access traces for a finished job")
    health.add_argument("target", help="Job id under --output-root, or path to job/session dir")
    health.add_argument("--output-root", default="runs")
    health.add_argument(
        "--no-expect-tools",
        action="store_true",
        help="Do not treat zero reads as broken (pack / constant agents)",
    )

    runs = sub.add_parser("runs", help="List, show, or watch jobs")
    runs_sub = runs.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_sub.add_parser("list", help="List jobs under output-root")
    runs_list.add_argument("--output-root", default="runs")
    runs_show = runs_sub.add_parser("show", help="Show one job result + health")
    runs_show.add_argument("job_id")
    runs_show.add_argument("--output-root", default="runs")
    runs_watch = runs_sub.add_parser("watch", help="Live in-place dashboard for one or more jobs")
    runs_watch.add_argument(
        "job_ids",
        nargs="+",
        help="job id(s) or nerve log path(s) (run.log / backfill.log)",
    )
    runs_watch.add_argument("--output-root", default="runs")
    runs_watch.add_argument(
        "--watch-mode",
        choices=["auto", "alt", "stream"],
        default="auto",
        help="Render mode: auto (default; stream in Cursor, alt elsewhere), alt, stream",
    )
    runs_watch.add_argument(
        "--wait",
        type=int,
        default=0,
        metavar="SECONDS",
        help="If the job(s) don't exist yet, poll up to SECONDS for them to appear "
        "(lets you start the TUI before the simulation). Default 0 (no wait).",
    )

    rep = sub.add_parser(
        "report", help="Evaluate a finished job (KPI + stochasticity + holdings + agent eval)"
    )
    rep.add_argument("job_id", help="Job id under --output-root")
    rep.add_argument("--output-root", default="runs")
    rep.add_argument(
        "--cache-root",
        default=None,
        help="Price cache root (default: <output-root>/cache)",
    )
    rep.add_argument(
        "--shared-concurrency",
        type=int,
        default=None,
        help="Parallelism for agent-on-agent eval cells (default: sequential)",
    )

    cache = sub.add_parser("cache", help="Inspect the central data cache (read-only)")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)
    cache_status = cache_sub.add_parser(
        "status", help="Show cached date ranges per kind/symbol (gap-aware)"
    )
    cache_status.add_argument(
        "--source",
        default=None,
        help="Show only this catalog source (default: all registered sources)",
    )
    cache_status.add_argument(
        "--symbol",
        default=None,
        help="Show only this symbol (default: all symbols on disk)",
    )
    cache_status.add_argument(
        "--window",
        default=None,
        metavar="FROM..TO",
        help="Highlight gaps inside this window (ISO dates, e.g. 2024-01-01..2025-12-31)",
    )
    cache_status.add_argument(
        "--cache-root",
        default=None,
        help="Cache directory (default: <output-root>/cache)",
    )
    cache_status.add_argument("--output-root", default="runs", help="Output root (default: runs)")

    forward_fill = sub.add_parser("forward-fill", help="Add new decision dates to a finished run")
    forward_fill.add_argument("job_id", help="Job id under --output-root")
    forward_fill.add_argument(
        "--run",
        type=int,
        default=1,
        dest="run_index",
        help="Which repeat (rK) to forward-fill (default 1)",
    )
    forward_fill.add_argument(
        "--through",
        default=None,
        help="Run all scheduled dates up to this date (ISO, default: today)",
    )
    forward_fill.add_argument(
        "--dates",
        default=None,
        help="Explicit comma-separated ISO dates to add (overrides schedule)",
    )
    forward_fill.add_argument(
        "--schedule-kind",
        default=None,
        help="Override the run's schedule kind (e.g. weekly_fridays)",
    )
    forward_fill.add_argument(
        "--schedule-start",
        default=None,
        help="Start date for the override schedule (ISO)",
    )
    forward_fill.add_argument(
        "--schedule-anchor",
        default=None,
        help="Phase date for biweekly_fridays (ISO Friday of the fortnight)",
    )
    forward_fill.add_argument(
        "--cell-concurrency",
        type=int,
        default=1,
        help="Flat pool size for running cells (default 1)",
    )
    forward_fill.add_argument(
        "--output-root",
        default="runs",
        help="Directory for job artifacts (default: runs)",
    )
    forward_fill.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live progress (still writes logs)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "simulation":
        from fintel.cli.simulation import run_simulation

        return run_simulation(args)
    if args.command == "backfill":
        from fintel.cli.backfill import run_backfill_cli

        return run_backfill_cli(args)
    if args.command == "forward-fill":
        from fintel.cli.forward_fill import run_forward_fill_cli

        return run_forward_fill_cli(args)
    if args.command == "health":
        from fintel.cli.health import run_health

        return run_health(args)
    if args.command == "runs":
        from fintel.cli.runs import run_runs

        return run_runs(args)
    if args.command == "report":
        from fintel.cli.report import run_report

        return run_report(args)
    if args.command == "cache":
        from fintel.cli.cache import run_cache

        return run_cache(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
