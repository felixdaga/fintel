"""Step-by-step probe of the AAPL prices prefetch path, with timing per step.

The Massive API answers in <1s in isolation, but the full prefetch hangs for
>90s. This runs each layer in sequence with a wall-clock print after every
step so the hang localises immediately.

Run:  python scripts/probe_prefetch.py
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

from fintel.market.data.store import PriceStore
from fintel.market.data import coverage as cov
from fintel.market.data.http import MassiveClient
from fintel.market.data.massive import MassivePrices, default_history_start
from fintel.market.settings import MarketConfig

OUT = ROOT / "runs"
CFG = MarketConfig.from_env(cache_root=OUT / "cache")


def _t(label: str, t0: float) -> float:
    print(f"  {label:40s} {time.perf_counter()-t0:7.3f}s", flush=True)
    return time.perf_counter()


def main() -> None:
    symbol = "AAPL"
    through = D(2025, 1, 1)
    print(f"== probe prefetch path for {symbol} through {through}", flush=True)

    t = time.perf_counter()
    store = PriceStore(root=CFG.cache_root)
    t = _t("PriceStore constructed", t)

    cov_spans = store.coverage(symbol)
    t = _t(f"store.coverage({symbol}) -> {cov_spans}", t)

    need_from = default_history_start()
    t = _t(f"default_history_start() -> {need_from}", t)

    gaps = cov.missing(cov_spans, need_from, through)
    t = _t(f"cov.missing -> {len(gaps)} gap(s): {gaps[:3]}", t)

    if not gaps:
        print("  no gaps — cache already warm; nothing to probe")
        return

    print(f"  -> fetching {len(gaps)} gap(s) from the API...", flush=True)
    key = CFG.require_key("MASSIVE_API_KEY", CFG.massive_api_key)
    client = MassiveClient(key)
    t = _t("MassiveClient constructed", t)

    lo, hi = gaps[0]
    print(f"  -> _fetch_bars({symbol}, {lo}, {hi}) ...", flush=True)
    path = f"/v2/aggs/ticker/{symbol}/range/1/day/{lo.isoformat()}/{hi.isoformat()}"
    t_fetch = time.perf_counter()
    results = client.paginate(path, {"adjusted": "true", "sort": "asc", "limit": 50000})
    t = _t(f"client.paginate -> {len(results)} results", t_fetch)
    print(f"     client.n_requests = {client.n_requests}", flush=True)

    if not results:
        print("  -> API returned no results; skipping merge")
        return

    import pandas as pd

    rows = [
        {
            "date": pd.Timestamp(r["t"], unit="ms").date(),
            "open": r.get("o"), "high": r.get("h"), "low": r.get("l"),
            "close": r.get("c"), "volume": r.get("v"),
        }
        for r in results if r.get("t") is not None
    ]
    fresh = pd.DataFrame(rows)
    t = _t(f"built DataFrame -> {len(fresh)} rows", t)

    print(f"  -> store.merge({symbol}, fresh, ({lo}, {hi})) ...", flush=True)
    merged = store.merge(symbol, fresh, (lo, hi))
    t = _t(f"store.merge -> {len(merged)} rows", t)

    print(f"  -> store.read({symbol}) ...", flush=True)
    back = store.read(symbol)
    t = _t(f"store.read -> {len(back) if back is not None else 0} rows", t)

    print()
    print("== probe complete — every step timed above")


if __name__ == "__main__":
    main()
