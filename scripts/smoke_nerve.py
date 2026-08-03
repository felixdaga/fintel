"""Nerve smoke: a 2-ticker × 1-date run that exercises the full central
nervous system offline — preflight, probe, run echo, live agent staging
(tool calls + grid), and results — with no LLM and no API key.

Uses the scripted agent on the 'tools' channel so it makes real tool reads
through env.tools.call, which now emits agent_tool_call / agent_tool_result
staging events to the nerve. Combined with the nerve's StageTracker, this
produces run_grid snapshots and (if anything stalls) agent_stalled events in
run.log and the terminal.

Run:  python scripts/smoke_nerve.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fintel.market.settings import MarketConfig
from fintel.models.agent import AgentSpec, ModelSpec
from fintel.models.job import JobConfig
from fintel.models.market import DataBinding, ScheduleRef, UniverseRef
from fintel.simulate import run_job

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "packages" / "systematic_stockrate_djia"
OUT = ROOT / "runs"


def main() -> None:
    job = JobConfig(
        job_id="nerve-smoke-0001",
        strategy=str(PACKAGE),
        agent=AgentSpec(
            name="scripted",
            model=ModelSpec(),
            options={
                "score": 0.4,
                "channel": "tools",
                "reads": ("prices",),
            },
        ),
        k_repeats=1,
        max_concurrent=1,
        cell_concurrency=None,
        output_root=str(OUT),
        universe=UniverseRef(symbols=["AAPL", "MSFT"]),
        schedule=ScheduleRef(kind="custom_dates", dates=["2025-01-02"]),
        data=[DataBinding(kind="prices", source="synthetic_prices")],
    )

    print(f"== fintel nerve smoke: {job.job_id}")
    print(f"   agent: scripted (tools channel)  data: synthetic_prices")
    print()

    result = run_job(job, market_config=MarketConfig(cache_root=OUT / "cache", offline=True))

    print(f"== result: {result.status}")
    if result.runs:
        print(f"   run: {result.runs[0].status}")
    print()

    run_root = OUT / job.job_id / "r1"
    log_path = run_root / "run.log"
    echo_path = run_root / "echo.json"
    print(f"   echo.json:  {'present' if echo_path.is_file() else 'MISSING'}")
    print(f"   run.log:    {'present' if log_path.is_file() else 'MISSING'}")
    if log_path.is_file():
        events = [json.loads(l)["event"] for l in log_path.read_text().splitlines() if l.strip()]
        from collections import Counter
        counts = Counter(events)
        print(f"   run.log events: {dict(counts)}")
        for want in ("run_echo", "agent_tool_call", "agent_tool_result", "run_grid"):
            print(f"     {'OK ' if want in events else '!! missing: '}{want}")


if __name__ == "__main__":
    main()
