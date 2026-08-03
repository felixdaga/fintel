"""2t2d smoke: 2 tickers × 2 dates, the delorean-next smoke test, ported to fintel.

Runs the systematic_stockrate_djia package with:
  · universe override: AAPL, MSFT (2 tickers)
  · schedule override: 2025-01-02, 2025-04-01 (2 dates)
  · data override:    synthetic_prices (no API key needed)
  · agent:            scripted (no LLM cost)

Proves the fintel pipeline end-to-end: package load → preflight → lock →
job → run → trial → cell → decisions on disk. Scoring/report are not wired
yet, so this stops at the decision artifacts.

Run:  python scripts/smoke_2t2d.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fintel.simulate import run_job
from fintel.market.settings import MarketConfig
from fintel.models.agent import AgentSpec, ModelSpec
from fintel.models.job import JobConfig
from fintel.models.market import DataBinding, ScheduleRef, UniverseRef

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "packages" / "systematic_stockrate_djia"
OUT = ROOT / "runs"


def main() -> None:
    job = JobConfig(
        job_id="2t2d-smoke-0001",
        strategy=str(PACKAGE),
        agent=AgentSpec(name="scripted", model=ModelSpec(), options={"score": 0.3, "reads": ("prices",)}),
        k_repeats=1,
        max_concurrent=1,        # one run
        cell_concurrency=None,   # auto: both tickers at once
        output_root=str(OUT),
        universe=UniverseRef(symbols=["AAPL", "MSFT"]),
        schedule=ScheduleRef(kind="custom_dates", dates=["2025-01-02", "2025-04-01"]),
        data=[DataBinding(kind="prices", source="synthetic_prices")],
    )

    print(f"== fintel 2t2d smoke: {job.job_id}")
    print(f"   package:  {PACKAGE.name}")
    print(f"   universe: {job.universe.symbols}")
    print(f"   dates:    {job.schedule.params.get('dates')}")
    print(f"   agent:    {job.agent.name}  (no LLM cost)")
    print(f"   data:     synthetic_prices  (no API key)")
    print()

    result = run_job(job, market_config=MarketConfig(cache_root=OUT / "cache", offline=True))

    print(f"== result: {result.status}")
    print(f"   runs: {len(result.runs)}  status: {result.runs[0].status if result.runs else '-'}")
    print()

    job_root = OUT / job.job_id
    print("== artifact tree:")
    for p in sorted(job_root.rglob("*")):
        if p.is_file():
            print(f"   {p.relative_to(job_root)}")
    print()

    # Show the decisions for each date
    run_dir = job_root / "r1"
    for trial_dir in sorted(run_dir.iterdir()):
        if not trial_dir.is_dir() or trial_dir.name != "trials":
            continue
        for d in sorted((run_dir / "trials").iterdir()):
            decision_path = d / "decision.json"
            if decision_path.is_file():
                decision = json.loads(decision_path.read_text())
                print(f"== decision {d.name}: {list(decision)}")
                for sym, view in decision.items():
                    print(f"   {sym}: score={view['score']}  rationale={view.get('rationale', '')[:60]}")
    print()
    print("OK — pipeline produced decisions for 2 tickers × 2 dates.")


if __name__ == "__main__":
    main()
