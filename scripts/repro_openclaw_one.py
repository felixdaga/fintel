"""Single-cell openclaw reproduction: one ticker, one date, short timeout,
full stdout/stderr capture so we can see exactly what the agent did.

Reproduces the 0005 failures (AAPL: exit 0 / no result.json; MSFT: timeout)
in a controlled way and dumps the full openclaw JSON envelope for inspection.

Run:  python scripts/repro_openclaw_one.py [SYMBOL] [TIMEOUT_S]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fintel.utils.secrets import bootstrap_env

bootstrap_env()

from fintel.simulate import run_job  # noqa: E402
from fintel.market.settings import MarketConfig  # noqa: E402
from fintel.models.agent import AgentSpec, ModelSpec  # noqa: E402
from fintel.models.job import JobConfig  # noqa: E402
from fintel.models.market import ScheduleRef, UniverseRef  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "packages" / "systematic_stockrate_djia"
OUT = ROOT / "runs"

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
TIMEOUT = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
JOB_ID = f"repro-oc-{SYMBOL.lower()}-0001"


def main() -> None:
    job = JobConfig(
        job_id=JOB_ID,
        strategy=str(PACKAGE),
        agent=AgentSpec(
            name="openclaw",
            model=ModelSpec(),
            options={
                "profile": "delorean",
                "agent_id": "main",
                "timeout_s": TIMEOUT,
            },
        ),
        k_repeats=1,
        max_concurrent=1,
        cell_concurrency=1,
        output_root=str(OUT),
        universe=UniverseRef(symbols=[SYMBOL]),
        schedule=ScheduleRef(kind="custom_dates", dates=["2025-01-02"]),
    )

    print(f"== fintel single-cell openclaw repro: {JOB_ID}")
    print(f"   symbol:  {SYMBOL}")
    print(f"   timeout: {TIMEOUT}s")
    print(f"   data:    massive (prices/fundamentals/news)")
    print()

    result = run_job(job, market_config=MarketConfig.from_env(cache_root=OUT / "cache"))

    print(f"== result: {result.status}  health: {result.health}")
    if result.runs:
        r = result.runs[0]
        print(f"   run: {r.status}  error: {r.error}")
    print()

    # Dump the full access log + cell record + any result.json
    session = OUT / JOB_ID / "r1" / "sessions" / f"{JOB_ID}-r1" / "2025-01-02" / SYMBOL
    print(f"== session dir: {session}")
    print()

    access_path = session / "access.jsonl"
    if access_path.is_file():
        print("--- access.jsonl ---")
        for line in access_path.read_text().splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            # Truncate long fields for readability
            for k in ("stdout_tail", "stderr_tail"):
                if k in ev and ev[k]:
                    ev[k] = ev[k][:200] + ("..." if len(ev[k]) > 200 else "")
            print(json.dumps(ev, indent=2))
        print()

    # The subprocess_done event carries the openclaw stdout tail — pull the
    # full envelope from the job.log too.
    job_log = OUT / JOB_ID / "job.log"
    if job_log.is_file():
        print("--- job.log (subprocess_done events) ---")
        for line in job_log.read_text().splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            if ev.get("event") == "cell_done" and ev.get("detail"):
                print(f"   cell_done {ev.get('cell')}: outcome={ev.get('outcome')} "
                      f"health={ev.get('health')} detail={ev['detail']}")
        print()

    cell_path = OUT / JOB_ID / "r1" / "trials" / "2025-01-02" / "cells" / f"{SYMBOL}.json"
    if cell_path.is_file():
        print("--- cell record ---")
        cell = json.loads(cell_path.read_text())
        print(json.dumps(cell, indent=2))
        print()

    result_path = session / "result.json"
    print(f"== result.json present: {result_path.is_file()}")
    if result_path.is_file():
        print(json.dumps(json.loads(result_path.read_text()), indent=2))


if __name__ == "__main__":
    main()
