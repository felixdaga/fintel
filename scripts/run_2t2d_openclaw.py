"""2t2d on openclaw: 2 tickers x 2 dates, the delorean openclaw agent + massive data.

Uses the openclaw "delorean" agent the way delorean did: ``openclaw --profile
delorean agent --json --agent main``. The model/billing is the profile's
concern (its gateway carries the key), so the adapter sets no model. The fintel
MCP server is patched into the profile per cell, pointing at that cell's
session dir.

Cells run sequentially (cell_concurrency=1) because the per-cell profile patch
is not safe under concurrency without a slot pool (deliberately deferred).

Run:  python scripts/run_2t2d_openclaw.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fintel.utils.secrets import bootstrap_env

bootstrap_env()  # MASSIVE_API_KEY for the MCP server env block

from fintel.simulate import run_job  # noqa: E402
from fintel.market.settings import MarketConfig  # noqa: E402
from fintel.models.agent import AgentSpec, ModelSpec  # noqa: E402
from fintel.models.job import JobConfig  # noqa: E402
from fintel.models.market import ScheduleRef, UniverseRef  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "packages" / "systematic_stockrate_djia"
OUT = ROOT / "runs"


def main() -> None:
    # 1 date × 2 tickers — first live openclaw decision test (cheap + clear).
    job = JobConfig(
        job_id="1d2t-openclaw-0004",
        strategy=str(PACKAGE),
        agent=AgentSpec(
            name="openclaw",
            model=ModelSpec(),  # profile owns the model
            options={"profile": "delorean", "agent_id": "main"},
        ),
        k_repeats=1,
        max_concurrent=1,
        cell_concurrency=1,   # sequential: per-cell profile patch isn't concurrency-safe
        output_root=str(OUT),
        universe=UniverseRef(symbols=["AAPL", "MSFT"]),
        schedule=ScheduleRef(kind="custom_dates", dates=["2025-01-02"]),
    )

    print(f"== fintel 1d2t openclaw: {job.job_id}")
    print(f"   package:  {PACKAGE.name}")
    print(f"   universe: {job.universe.symbols}")
    print(f"   dates:    {job.schedule.params.get('dates')}")
    print(f"   agent:    openclaw (profile=delorean, agent=main) -- gateway billing")
    print(f"   data:     massive (prices/fundamentals/news)")
    print(f"   cells:    sequential (per-cell profile patch + PIT deny)")
    print()

    result = run_job(job, market_config=MarketConfig.from_env(cache_root=OUT / "cache"))

    print(f"== result: {result.status}")
    if result.runs:
        print(f"   runs: {len(result.runs)}  status: {result.runs[0].status}")
        if result.runs[0].error:
            print(f"   error: {result.runs[0].error}")
    print()

    job_root = OUT / job.job_id
    print("== artifact tree:")
    for p in sorted(job_root.rglob("*")):
        if p.is_file():
            print(f"   {p.relative_to(job_root)}")
    print()

    run_dir = job_root / "r1"
    trials = run_dir / "trials"
    if trials.is_dir():
        for d in sorted(trials.iterdir()):
            decision_path = d / "decision.json"
            if decision_path.is_file():
                decision = json.loads(decision_path.read_text())
                print(f"== decision {d.name}: {list(decision)}")
                for sym, view in decision.items():
                    print(
                        f"   {sym}: score={view.get('score')}  "
                        f"rationale={view.get('rationale', '')[:80]}"
                    )
    print()
    print("OK -- openclaw simulation produced decisions for 2 tickers x 1 date.")


if __name__ == "__main__":
    main()
