"""2-ticker openclaw test: AAPL + MSFT, 1 date, sequential cells.

Confirms the fix scales beyond one ticker. In run 0005, MSFT timed out at
600s after its cold news fill hung inside the agent loop. With the prefetch
now warming news in ~1.4s per ticker, both cells should produce real
decisions.

Run:  python scripts/repro_openclaw_2t.py
"""

from __future__ import annotations

import json
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
JOB_ID = "repro-oc-2t-0001"


def main() -> None:
    job = JobConfig(
        job_id=JOB_ID,
        strategy=str(PACKAGE),
        agent=AgentSpec(
            name="openclaw",
            model=ModelSpec(),
            options={"profile": "delorean", "agent_id": "main", "timeout_s": 300.0},
        ),
        k_repeats=1,
        max_concurrent=1,
        cell_concurrency=1,  # sequential: per-cell profile patch isn't concurrency-safe
        output_root=str(OUT),
        universe=UniverseRef(symbols=["AAPL", "MSFT"]),
        schedule=ScheduleRef(kind="custom_dates", dates=["2025-01-02"]),
    )

    print(f"== fintel 2-ticker openclaw: {JOB_ID}")
    print(f"   universe: AAPL, MSFT   date: 2025-01-02   timeout: 300s/cell")
    print()

    result = run_job(job, market_config=MarketConfig.from_env(cache_root=OUT / "cache"))

    print(f"== result: {result.status}  health: {result.health}")
    if result.runs:
        print(f"   run: {result.runs[0].status}  error: {result.runs[0].error}")
    print()

    run_dir = OUT / JOB_ID / "r1"
    trials = run_dir / "trials"
    if trials.is_dir():
        for d in sorted(trials.iterdir()):
            decision_path = d / "decision.json"
            if decision_path.is_file():
                decision = json.loads(decision_path.read_text())
                print(f"== decision {d.name}: {list(decision)}")
                for sym, view in decision.items():
                    print(f"   {sym}: score={view.get('score')}  "
                          f"rationale={str(view.get('rationale', ''))[:90]}")
    print()

    # Per-cell outcome summary
    cells_dir = run_dir / "trials" / "2025-01-02" / "cells"
    if cells_dir.is_dir():
        print("== per-cell outcomes:")
        for cf in sorted(cells_dir.glob("*.json")):
            cell = json.loads(cf.read_text())
            env = cell.get("environment", {})
            health = env.get("health", {})
            print(f"   {cell['cell']}: outcome={cell['outcome']} n_views={cell['n_views']} "
                  f"reads={env.get('n_reads')} health={health.get('status')} "
                  f"elapsed={cell.get('elapsed_ms')}ms")
    print()

    if result.status == "ok" and result.health == "ok":
        print("OK -- both tickers produced real decisions with healthy environments")
    else:
        print(f"NOT OK -- status={result.status} health={result.health}")


if __name__ == "__main__":
    main()
