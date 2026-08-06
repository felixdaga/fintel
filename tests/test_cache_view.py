"""Cache introspection: read gap-aware coverage back from disk, read-only."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pandas as pd

from fintel.market import catalog
from fintel.market.cache_view import coverage_for_kind, coverage_summary
from fintel.market.data.store import PriceStore, RecordCache


def _write_prices(cache_root: Path, symbol: str, dates: list[Date]) -> None:
    store = PriceStore(root=cache_root)
    df = pd.DataFrame(
        {"date": dates, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}
    )
    span = (dates[0], dates[-1])
    store.write(symbol, df, [span])


def _write_records(
    cache_root: Path, kind: str, symbol: str, spans: list[tuple[Date, Date]]
) -> None:
    cache = RecordCache(root=cache_root, kind=kind)
    # Records carry a cutoff_field the source clamps on; coverage is what we test.
    records = [
        {
            "id": f"{symbol}-{i}",
            "filing_date": a.isoformat(),
            "published_at": a.isoformat(),
        }
        for i, (a, b) in enumerate(spans)
    ]
    cache.write(symbol, spans, records)


def test_prices_coverage_reads_the_sidecar(tmp_path: Path) -> None:
    _write_prices(tmp_path, "AAPL", [Date(2024, 1, 2), Date(2024, 1, 3), Date(2024, 6, 1)])
    cov = coverage_for_kind(kind="prices", source_name="massive_prices", cache_root=tmp_path)
    assert cov.kind == "prices"
    assert "AAPL" in cov.symbols
    assert cov.symbols["AAPL"] == [(Date(2024, 1, 2), Date(2024, 6, 1))]


def test_record_coverage_is_gap_aware(tmp_path: Path) -> None:
    # Two disjoint spans with a hole between them.
    _write_records(
        tmp_path,
        "fundamentals",
        "MSFT",
        [(Date(2024, 1, 1), Date(2024, 3, 31)), (Date(2025, 1, 1), Date(2025, 3, 31))],
    )
    cov = coverage_for_kind(
        kind="fundamentals",
        source_name="massive_fundamentals",
        cache_root=tmp_path,
    )
    spans = cov.symbols["MSFT"]
    assert len(spans) == 2
    # A request across the hole reports the gap.
    gaps = cov.gaps("MSFT", Date(2024, 1, 1), Date(2025, 6, 30))
    assert (Date(2024, 4, 1), Date(2024, 12, 31)) in gaps


def test_computed_kinds_have_no_own_cache(tmp_path: Path) -> None:
    catalog.register_builtins()
    cov = coverage_for_kind(kind="ratios", source_name="valuation_ratios", cache_root=tmp_path)
    assert cov.symbols == {}


def test_coverage_summary_looks_up_kind_from_catalog(tmp_path: Path) -> None:
    _write_prices(tmp_path, "AAPL", [Date(2024, 1, 2)])
    cov = coverage_summary(source_name="massive_prices", cache_root=tmp_path)
    assert cov.kind == "prices"
    assert "AAPL" in cov.symbols


def test_symbol_filter_only_reports_that_symbol(tmp_path: Path) -> None:
    _write_prices(tmp_path, "AAPL", [Date(2024, 1, 2)])
    _write_prices(tmp_path, "MSFT", [Date(2024, 1, 2)])
    cov = coverage_for_kind(
        kind="prices", source_name="massive_prices", cache_root=tmp_path, symbol="AAPL"
    )
    assert set(cov.symbols) == {"AAPL"}


def test_empty_cache_reports_nothing(tmp_path: Path) -> None:
    cov = coverage_for_kind(kind="prices", source_name="massive_prices", cache_root=tmp_path)
    assert cov.symbols == {}
    assert cov.n_symbols == 0
