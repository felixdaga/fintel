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
        "--max-concurrent",
        type=int,
        default=1,
        help="Concurrent K repeats (default 1)",
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
        "Point at delorean's cache to reuse it.",
    )
    simulation.add_argument(
        "--offline",
        action="store_true",
        help="Cache-only; a miss is an error instead of a network call",
    )
    simulation.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Do not load keys from ~/.openclaw profiles",
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
    runs_watch.add_argument("job_ids", nargs="+", help="job id(s) or run.log path(s)")
    runs_watch.add_argument("--output-root", default="runs")

    rep = sub.add_parser("report", help="Evaluate a finished job (KPI + stochasticity + holdings)")
    rep.add_argument("job_id", help="Job id under --output-root")
    rep.add_argument("--output-root", default="runs")
    rep.add_argument(
        "--cache-root",
        default=None,
        help="Price cache root (default: <output-root>/cache)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "simulation":
        from fintel.cli.simulation import run_simulation

        return run_simulation(args)
    if args.command == "health":
        from fintel.cli.health import run_health

        return run_health(args)
    if args.command == "runs":
        from fintel.cli.runs import run_runs

        return run_runs(args)
    if args.command == "report":
        from fintel.cli.report import run_report

        return run_report(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
