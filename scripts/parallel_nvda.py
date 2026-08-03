"""Two parallel openclaw runs: NVDA across Jan and Apr 2026.

Each run runs in its own thread with its own `Nerve` writing to a shared
terminal through a line-prefixing stream, so an operator can watch both
runs' steps/turns live without the two streams garbling each other.

Per-cell profile isolation (each cell forks the operator's profile onto a
unique gateway port) means the two parallel runs cannot share a gateway or a
stale fintel mcp server — the cause of the earlier intermittent "0 reads".

Run:  python scripts/parallel_nvda.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from fintel.utils.secrets import bootstrap_env

bootstrap_env()

from fintel.environment.nerve import Nerve  # noqa: E402
from fintel.market.settings import MarketConfig  # noqa: E402
from fintel.models.agent import AgentSpec, ModelSpec  # noqa: E402
from fintel.models.job import JobConfig  # noqa: E402
from fintel.models.market import ScheduleRef, UniverseRef  # noqa: E402
from fintel.simulate import run_job  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs"

# One global lock so the two runs never interleave mid-line.
_PRINT_LOCK = threading.Lock()


class _PrefixedStream:
    """A minimal TextIO: each write() is split into lines and each line is
    emitted to the real stdout prefixed with the run's tag, under one lock."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            with _PRINT_LOCK:
                sys.stdout.write(f"{self.tag} {line}\n")
                sys.stdout.flush()
        return len(s)

    def flush(self) -> None:
        if self._buf:
            with _PRINT_LOCK:
                sys.stdout.write(f"{self.tag} {self._buf}\n")
                sys.stdout.flush()
            self._buf = ""


def _job(job_id: str, date: str) -> JobConfig:
    return JobConfig(
        job_id=job_id,
        strategy=str(ROOT / "packages" / "systematic_stockrate_djia"),
        agent=AgentSpec(
            name="openclaw",
            model=ModelSpec(),
            options={"profile": "delorean", "agent_id": "main", "timeout_s": 300.0},
        ),
        k_repeats=1,
        max_concurrent=1,
        cell_concurrency=1,
        output_root=str(OUT),
        universe=UniverseRef(symbols=["NVDA"]),
        schedule=ScheduleRef(kind="custom_dates", dates=[date]),
    )


def _run(tag: str, job_id: str, date: str, market: MarketConfig) -> dict:
    # verbose=False: the nerve only writes run.log (no scrolling terminal text).
    # Display is the `fintel runs watch` dashboard, which tails run.log and
    # renders a single in-place screen — far more readable than interleaved
    # prefixed lines from two parallel runs.
    nerve = Nerve(
        run_root=OUT / job_id / "r1",
        stream=_PrefixedStream(tag),
        verbose=False,
    )
    result = run_job(_job(job_id, date), market_config=market, progress=nerve)
    run = result.runs[0] if result.runs else None
    return {
        "tag": tag,
        "job_id": job_id,
        "status": result.status,
        "health": result.health,
        "run_status": run.status if run else None,
        "run_error": run.error if run else None,
    }


def main() -> None:
    market = MarketConfig.from_env(cache_root=OUT / "cache")
    specs = [
        ("[jan]", "nvda-parallel-0002-jan", "2026-01-02"),
        ("[apr]", "nvda-parallel-0002-apr", "2026-04-01"),
    ]
    print("== fintel parallel openclaw: NVDA x2 periods (Jan + Apr 2026)")
    print("   per-cell profile isolation (unique gateway port) — no shared gateway\n")
    results: list[dict] = [None, None]  # type: ignore[list-item]
    threads = [
        threading.Thread(
            target=lambda i=i, tag=tag, jid=jid, d=d: results.__setitem__(i, _run(tag, jid, d, market))
        )
        for i, (tag, jid, d) in enumerate(specs)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("\n== summary")
    for r in results:
        print(f"   {r['tag']} {r['job_id']}: status={r['status']} health={r['health']} "
              f"run={r['run_status']} err={r['run_error']}")


if __name__ == "__main__":
    main()
