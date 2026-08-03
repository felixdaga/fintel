"""Full 3-kind prefetch probe for AAPL: prices + fundamentals + news, timed
per kind. The prices-only probe finished in 1.7s, so the hang (if any) must be
in fundamentals or news — or in the orchestration around them.

Run:  python scripts/probe_prefetch_full.py
"""

from __future__ import annotations

import sys
import time
from datetime import date as D
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintel.utils.secrets import bootstrap_env

bootstrap_env()

from fintel.market.settings import MarketConfig
from fintel.market.factory import build_data_sources
from fintel.market.prefetch import prefetch, prefetch_window
from fintel.models.market import DataBinding

OUT = ROOT / "runs"
CFG = MarketConfig.from_env(cache_root=OUT / "cache")


def main() -> None:
    bindings = [
        DataBinding(kind="prices", source="massive_prices"),
        DataBinding(kind="fundamentals", source="massive_fundamentals"),
        DataBinding(kind="news", source="massive_news"),
    ]
    dates = [D(2025, 1, 2)]

    print("== building data sources (no network yet)...", flush=True)
    t = time.perf_counter()
    sources = build_data_sources(bindings, config=CFG)
    print(f"   build_data_sources: {time.perf_counter()-t:.3f}s -> {list(sources)}", flush=True)

    t = time.perf_counter()
    pfrom, pthrough = prefetch_window(dates, sources, bindings)
    print(f"   prefetch_window: {time.perf_counter()-t:.3f}s -> {pfrom} .. {pthrough}", flush=True)

    print()
    print("== full prefetch (AAPL, all kinds, workers=3)...", flush=True)
    t = time.perf_counter()
    res = prefetch(
        symbols=["AAPL"], sources=sources, from_date=pfrom, through_date=pthrough,
        workers=3, decision_dates=dates,
    )
    print(f"   prefetch total: {time.perf_counter()-t:.3f}s", flush=True)
    print(f"   warmed: {sorted(res.warmed)}", flush=True)
    print(f"   failed: {res.failed}", flush=True)

    print()
    print("== now a warm re-run (should be instant)...", flush=True)
    t = time.perf_counter()
    res2 = prefetch(
        symbols=["AAPL"], sources=sources, from_date=pfrom, through_date=pthrough,
        workers=3, decision_dates=dates,
    )
    print(f"   warm re-run: {time.perf_counter()-t:.3f}s  warmed={len(res2.warmed)} failed={len(res2.failed)}", flush=True)

    # Show per-kind windows actually used
    print()
    print("== per-kind warm windows (news should be ~90 days, not 4 years):", flush=True)
    from fintel.market.prefetch import _kind_from
    for kind in sorted(sources):
        kf = _kind_from(sources, kind, dates)
        print(f"   {kind:14s} from {kf}  through {pthrough}  ({(pthrough - kf).days} days)", flush=True)


if __name__ == "__main__":
    main()
