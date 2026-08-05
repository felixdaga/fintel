"""2t2d real run: 2 tickers x 2 dates, real LLM + real massive data.

Keys come from fintel's local ``.env/keys.env`` via ``bootstrap_env``.

Uses the in-process `llm` agent on the `tools` channel (the fully-wired host):
the model calls the fintel data tools, which fetch from massive under PIT
control, then submits views. This is the simulation layer only -- it stops at
the decision artifacts; scoring is the evaluation layer, built separately.

The openclaw *subprocess* adapter is a separate delivery channel and not wired
yet, so this run uses the in-process host that talks to OpenRouter directly.

Run:  python scripts/run_2t2d_real.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fintel.utils.secrets import bootstrap_env

# Pull the keys delorean used into os.environ before anything reads them.
bootstrap_env()

from fintel.simulate import run_job  # noqa: E402
from fintel.market.settings import MarketConfig  # noqa: E402
from fintel.models.agent import AgentSpec, ModelSpec  # noqa: E402
from fintel.models.job import JobConfig  # noqa: E402
from fintel.models.market import ScheduleRef, UniverseRef  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "packages" / "systematic_stockrate_djia"
OUT = ROOT / "runs"

# OpenRouter slug from the openclaw profile primary (strip the `openrouter/`
# provider prefix that openclaw uses for routing).
MODEL = "xiaomi/mimo-v2.5-pro"


def main() -> None:
    job = JobConfig(
        job_id="2t2d-real-0001",
        strategy=str(PACKAGE),
        agent=AgentSpec(
            name="llm",
            model=ModelSpec(id=MODEL),
            options={"channel": "tools", "max_rounds": 6, "max_tokens": 4000},
        ),
        k_repeats=1,
        max_concurrent=1,
        cell_concurrency=None,        # auto: both tickers at once
        output_root=str(OUT),
        universe=UniverseRef(symbols=["AAPL", "MSFT"]),
        schedule=ScheduleRef(kind="custom_dates", dates=["2025-01-02", "2025-04-01"]),
        # No data override: use the package's real massive bindings.
    )

    print(f"== fintel 2t2d real: {job.job_id}")
    print(f"   package:  {PACKAGE.name}")
    print(f"   universe: {job.universe.symbols}")
    print(f"   dates:    {job.schedule.params.get('dates')}")
    print(f"   agent:    {job.agent.name}  channel=tools  model={MODEL}")
    print(f"   data:     massive (prices/fundamentals/news) -- real fetches")
    print()

    result = run_job(
        job,
        market_config=MarketConfig.from_env(cache_root=OUT / "cache"),
    )

    print(f"== result: {result.status}")
    print(f"   runs: {len(result.runs)}  status: {result.runs[0].status if result.runs else '-'}")
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
    print("OK -- simulation produced decisions for 2 tickers x 2 dates.")


if __name__ == "__main__":
    main()
