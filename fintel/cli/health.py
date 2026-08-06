"""``fintel health`` — re-audit access traces for a finished job or session."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from fintel.cli.present import print_json_block
from fintel.environment.health import audit_job, audit_session


def run_health(args: Namespace) -> int:
    target = Path(args.target)
    if not target.exists():
        candidate = Path(args.output_root) / args.target
        if candidate.exists():
            target = candidate
        else:
            raise SystemExit(f"not found: {args.target}")

    expect_tools = not args.no_expect_tools
    if (target / "access.jsonl").is_file() or (target / "cell.json").is_file():
        report = audit_session(target, expect_tools=expect_tools).to_dict()
    else:
        report = audit_job(target, expect_tools=expect_tools)

    print_json_block("health", report)
    status = report.get("status", "ok")
    if status == "broken":
        return 1
    if status == "degraded":
        return 2
    return 0
