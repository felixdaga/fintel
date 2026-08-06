"""Cache warm-up: one pass over the union of symbols, parallel, skip-if-warm."""

from __future__ import annotations

from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from fintel.market.data.massive import FUNDAMENTALS, MassivePrices, MassiveRecords
from fintel.market.data.store import PriceStore, RecordCache
from fintel.market.prefetch import prefetch, prefetch_window


class StubPrices(MassivePrices):
    """MassivePrices with an in-memory store and no client — pure cache-hit path."""

    def __init__(self, root: Path) -> None:
        super().__init__(store=PriceStore(root=root), client=None)


class StubRecords(MassiveRecords):
    def __init__(self, spec, root: Path) -> None:
        super().__init__(
            spec=spec,
            cache=RecordCache(root=root, kind=spec.kind),
            client=None,
            name=f"stub_{spec.kind}",
        )


def test_prefetch_window_covers_widest_lookback():
    dates = [Date(2025, 1, 2), Date(2025, 4, 1)]
    src = {"fundamentals": StubRecords(FUNDAMENTALS, Path("/tmp"))}
    lo, hi = prefetch_window(dates, src)
    assert hi == Date(2025, 4, 1) - timedelta(days=1)
    # fundamentals lookback is 730 days; floor = first date - 730
    assert lo <= Date(2025, 1, 2) - timedelta(days=730)


def test_prefetch_is_skip_if_warm(tmp_path: Path):
    # Pre-seed a price parquet so _ensure is a no-op.
    import pandas as pd

    store = PriceStore(root=tmp_path)
    df = pd.DataFrame(
        {
            "date": [Date(2024, 1, 2), Date(2024, 1, 3)],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [0, 0],
        }
    )
    store.merge("AAPL", df, (Date(2020, 1, 1), Date(2025, 1, 1)))
    src = {"prices": StubPrices(tmp_path)}
    result = prefetch(
        symbols=["AAPL"],
        sources=src,
        from_date=Date(2020, 1, 1),
        through_date=Date(2025, 1, 1),
        workers=2,
    )
    assert result.failed == {}
    assert "AAPL:prices" in result.warmed


def test_prefetch_records_failure_not_fatal(tmp_path: Path):
    """A symbol with no cache and no client raises inside _ensure; prefetch
    records it as failed and continues — it does not abort the warm-up."""
    src = {"fundamentals": StubRecords(FUNDAMENTALS, tmp_path)}
    result = prefetch(
        symbols=["NOPE"],
        sources=src,
        from_date=Date(2024, 1, 1),
        through_date=Date(2025, 1, 1),
        workers=2,
    )
    assert "NOPE:fundamentals" in result.failed
    assert result.warmed == []


def test_runtime_only_kinds_are_not_prefetchable():
    from fintel.market.prefetch import PREFETCHABLE, is_prefetchable, is_runtime_only

    assert "web_search" not in PREFETCHABLE
    assert is_runtime_only("web_search")
    assert not is_prefetchable("web_search")
    assert is_prefetchable("prices")
