"""Probe the news fetch strategies side by side:

  A) fintel's current path: client.paginate(limit=100) over a 90-day window
     — follows next_url cursors sequentially.
  B) delorean's path: client.get(limit=1000) per 30-day chunk, concurrent
     — one round-trip per chunk, no cursor following.

This confirms whether the hang is in the paginate-over-news path and whether
the chunked approach is fast, before porting delorean's strategy into fintel.

Run:  python scripts/probe_news_fetch.py
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as D, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintel.utils.secrets import bootstrap_env

bootstrap_env()

from fintel.market.data.http import MassiveClient
from fintel.market.settings import MarketConfig

CFG = MarketConfig.from_env(cache_root=ROOT / "runs" / "cache")
KEY = CFG.require_key("MASSIVE_API_KEY", CFG.massive_api_key)


def _time(label: str, t0: float) -> float:
    print(f"  {label:55s} {time.perf_counter()-t0:7.3f}s", flush=True)
    return time.perf_counter()


def main() -> None:
    symbol = "AAPL"
    # A 90-day window — what an agent at 2025-01-02 would actually ask for.
    gte = D(2024, 10, 4)
    lte = D(2025, 1, 1)
    print(f"== news fetch probe: {symbol}  {gte} .. {lte}  (90 days)", flush=True)

    client = MassiveClient(KEY)

    # ── Strategy A: fintel's current path (paginate, limit=100) ──────────────
    print("\n[A] fintel current: client.paginate(limit=100) over the full window", flush=True)
    t = time.perf_counter()
    try:
        results_a = client.paginate(
            "/v2/reference/news",
            {
                "ticker": symbol,
                "published_utc.gte": gte.isoformat(),
                "published_utc.lte": lte.isoformat(),
                "limit": 100,
                "sort": "published_utc",
            },
        )
        _time(f"paginate(limit=100) -> {len(results_a)} results, {client.n_requests} reqs", t)
    except Exception as exc:
        _time(f"paginate(limit=100) FAILED: {type(exc).__name__}: {exc}", t)

    # ── Strategy B: delorean's path (chunked, limit=1000, concurrent) ────────
    print("\n[B] delorean strategy: 30-day chunks, client.get(limit=1000), concurrent", flush=True)
    chunk_days = 30
    chunks: list[tuple[D, D]] = []
    cursor = gte
    while cursor <= lte:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), lte)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    print(f"   {len(chunks)} chunk(s): {chunks}", flush=True)

    client_b = MassiveClient(KEY)

    def _fetch_chunk(cgte: D, clte: D) -> list[dict]:
        return client_b.get("/v2/reference/news", {
            "ticker": symbol,
            "published_utc.gte": cgte.isoformat(),
            "published_utc.lte": clte.isoformat(),
            "order": "desc",
            "limit": 1000,
        }).get("results") or []

    t = time.perf_counter()
    seen: set[str] = set()
    articles_b: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(10, len(chunks))) as pool:
        futures = {pool.submit(_fetch_chunk, cg, cl): (cg, cl) for cg, cl in chunks}
        for fut in as_completed(futures):
            for art in fut.result():
                aid = art.get("id", "")
                if aid and aid in seen:
                    continue
                if aid:
                    seen.add(aid)
                articles_b.append(art)
    _time(f"chunked get(limit=1000) -> {len(articles_b)} articles, {client_b.n_requests} reqs", t)

    print(f"\n== verdict: A={len(results_a)} articles  B={len(articles_b)} articles", flush=True)
    print("   (B should be >= A; A may have truncated via cursor caps or dropped older)", flush=True)


if __name__ == "__main__":
    main()
