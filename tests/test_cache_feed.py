"""Feed-level cache policy unit tests (no vendor I/O)."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pandas as pd
import pytest

from fintel.market.cache import ensure_prices, ensure_query_blob, ensure_records
from fintel.market.data.base import DataError, EntitlementError
from fintel.market.data.store import PriceStore, RecordCache


def test_ensure_records_hit_skips_fetch(tmp_path: Path):
    cache = RecordCache(root=tmp_path, kind="news")
    cache.merge(
        "AAPL",
        [{"id": "1", "published_at": "2026-04-01", "title": "x"}],
        [(Date(2026, 4, 1), Date(2026, 4, 23))],
        key=lambda r: r["id"],
        sort=lambda r: r["published_at"],
    )
    calls: list[tuple[Date, Date]] = []

    def fetch_span(lo: Date, hi: Date) -> list[dict]:
        calls.append((lo, hi))
        return []

    out = ensure_records(
        cache,
        "AAPL",
        Date(2026, 4, 1),
        Date(2026, 4, 23),
        fetch_span=fetch_span,
        identity=lambda r: r["id"],
        sort=lambda r: r["published_at"],
        online=True,
        source_name="test",
        kind_label="news",
    )
    assert len(out) == 1
    assert calls == []


def test_ensure_records_offline_miss_raises(tmp_path: Path):
    cache = RecordCache(root=tmp_path, kind="news")
    with pytest.raises(DataError, match="nothing cached"):
        ensure_records(
            cache,
            "AAPL",
            Date(2026, 4, 1),
            Date(2026, 4, 23),
            fetch_span=lambda lo, hi: [],
            identity=lambda r: r["id"],
            sort=lambda r: r.get("published_at", ""),
            online=False,
            source_name="test",
            kind_label="news",
        )


def test_ensure_records_fills_gap(tmp_path: Path):
    cache = RecordCache(root=tmp_path, kind="news")
    calls: list[tuple[Date, Date]] = []

    def fetch_span(lo: Date, hi: Date) -> list[dict]:
        calls.append((lo, hi))
        return [{"id": f"{lo}", "published_at": lo.isoformat(), "title": "n"}]

    out = ensure_records(
        cache,
        "AAPL",
        Date(2026, 4, 1),
        Date(2026, 4, 3),
        fetch_span=fetch_span,
        identity=lambda r: r["id"],
        sort=lambda r: r["published_at"],
        online=True,
        source_name="test",
        kind_label="news",
    )
    assert calls == [(Date(2026, 4, 1), Date(2026, 4, 3))]
    assert len(out) == 1
    # Second call is a hit.
    calls.clear()
    out2 = ensure_records(
        cache,
        "AAPL",
        Date(2026, 4, 1),
        Date(2026, 4, 3),
        fetch_span=fetch_span,
        identity=lambda r: r["id"],
        sort=lambda r: r["published_at"],
        online=True,
        source_name="test",
        kind_label="news",
    )
    assert calls == []
    assert out2 == out


def test_ensure_prices_records_empty_span(tmp_path: Path):
    store = PriceStore(root=tmp_path)
    calls = 0

    def fetch_bars(lo: Date, hi: Date) -> pd.DataFrame | None:
        nonlocal calls
        calls += 1
        return None

    out = ensure_prices(
        store,
        "AAPL",
        Date(2024, 1, 1),
        Date(2024, 1, 5),
        fetch_bars=fetch_bars,
        online=True,
        source_name="test",
    )
    assert out is None
    assert calls == 1
    # Empty span recorded → second ensure does not re-fetch.
    calls = 0
    ensure_prices(
        store,
        "AAPL",
        Date(2024, 1, 1),
        Date(2024, 1, 5),
        fetch_bars=fetch_bars,
        online=True,
        source_name="test",
    )
    assert calls == 0


def test_ensure_prices_entitlement_serves_cached_bars(tmp_path: Path):
    """Unentitled early gap must not abort a read when later bars are cached."""
    store = PriceStore(root=tmp_path)
    cached = pd.DataFrame(
        {
            "date": [Date(2024, 2, 1), Date(2024, 2, 2)],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.1],
            "volume": [100.0, 110.0],
        }
    )
    store.write("WBA", cached, [(Date(2024, 2, 1), Date(2024, 2, 2))])
    calls: list[tuple[Date, Date]] = []

    def fetch_bars(lo: Date, hi: Date) -> pd.DataFrame | None:
        calls.append((lo, hi))
        raise EntitlementError(f"plan blocked [{lo}, {hi}]")

    out = ensure_prices(
        store,
        "WBA",
        Date(2024, 1, 1),
        Date(2024, 2, 2),
        fetch_bars=fetch_bars,
        online=True,
        source_name="test",
    )
    assert out is not None
    assert list(out["date"]) == [Date(2024, 2, 1), Date(2024, 2, 2)]
    assert calls == [(Date(2024, 1, 1), Date(2024, 1, 31))]
    # Gap recorded → no second fetch.
    calls.clear()
    out2 = ensure_prices(
        store,
        "WBA",
        Date(2024, 1, 1),
        Date(2024, 2, 2),
        fetch_bars=fetch_bars,
        online=True,
        source_name="test",
    )
    assert calls == []
    assert out2 is not None
    assert len(out2) == 2


def test_ensure_query_blob_roundtrip(tmp_path: Path):
    path = tmp_path / "web_search" / "q.json"
    n = 0

    def fetch() -> dict:
        nonlocal n
        n += 1
        return {"query": "x", "sources": [1]}

    a = ensure_query_blob(
        path,
        online=True,
        fetch=fetch,
        source_name="web_search",
        miss_detail="miss",
    )
    b = ensure_query_blob(
        path,
        online=False,
        fetch=fetch,
        source_name="web_search",
        miss_detail="miss",
    )
    assert a == b == {"query": "x", "sources": [1]}
    assert n == 1


def test_ensure_query_blob_offline_miss(tmp_path: Path):
    with pytest.raises(DataError, match="miss"):
        ensure_query_blob(
            tmp_path / "missing.json",
            online=False,
            fetch=lambda: {},
            source_name="web_search",
            miss_detail="miss",
        )
