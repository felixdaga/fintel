"""Read cached coverage back from disk, gap-aware.

The catalog declares *what data exists*; this module answers *what is on disk
right now*. Coverage is stored per symbol as a list of inclusive
``[from, through]`` intervals (see ``market/data/coverage.py``), not a single
min/max — a cache warmed through 2026 with a hole in 2024 is not "covered" for a
2024 request, and a lone upper bound would lie about that.

This is read-only: it reads the coverage sidecars that ``PriceStore`` and
``RecordCache`` already maintain on every write. It does not fetch, and does
not mutate the cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

from fintel.market.data import coverage as cov
from fintel.market.data.store import PriceStore, RecordCache

Span = cov.Span


@dataclass(frozen=True)
class KindCoverage:
    """What's cached for one kind, across the symbols on disk."""

    kind: str
    source: str
    cache_dir: Path
    symbols: dict[str, list[Span]]  # symbol -> coalesced intervals

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    def gaps(self, symbol: str, need_from: Date, need_through: Date) -> list[Span]:
        return cov.missing(self.symbols.get(symbol, []), need_from, need_through)


def coverage_for_kind(
    *,
    kind: str,
    source_name: str,
    cache_root: Path,
    symbol: str | None = None,
) -> KindCoverage:
    """Read cached coverage for one kind.

    Layout follows the source's on-disk shape:

    * ``prices`` — parquet per symbol with a ``.coverage.json`` sidecar.
    * ``fundamentals`` / ``news`` / ``filing_text`` — JSON per symbol with an
      embedded ``_coverage`` field.
    * ``macro`` — JSON per FRED series ID (same RecordCache shape; keys are
      series IDs like ``FEDFUNDS``, not equity tickers).
    * ``ratios`` / ``news_sentiment`` — computed at fetch time, no own cache;
      report upstream coverage instead (caller's job).
    * ``web_search`` — keyed by query, not symbol; report the cached-file count
      and the min/max freshness window as a single span.

    Unknown kinds (a custom ``module:Callable`` source with its own layout)
    return an empty result — the catalog can't introspect a layout it didn't
    define.
    """
    root = Path(cache_root)
    if kind == "prices":
        store = PriceStore(root=root)
        symbols = {s: store.coverage(s) for s in (store.symbols() if symbol is None else [symbol])}
        return KindCoverage(
            kind=kind, source=source_name, cache_dir=root / "prices", symbols=symbols
        )

    if kind in {"fundamentals", "news", "filing_text", "macro"}:
        cache = RecordCache(root=root, kind=kind)
        if symbol is not None:
            spans, _ = cache.read(symbol)
            symbols = {symbol: spans}
        else:
            d = root / kind
            symbols = {}
            if d.is_dir():
                for p in sorted(d.glob("*.json")):
                    # Skip sibling meta files (e.g. macro/{SERIES}.meta.json).
                    if p.name.endswith(".meta.json"):
                        continue
                    spans, _ = cache.read(p.stem)
                    if spans:
                        symbols[p.stem] = spans
        return KindCoverage(kind=kind, source=source_name, cache_dir=root / kind, symbols=symbols)

    if kind == "web_search":
        d = root / "web_search"
        symbols: dict[str, list[Span]] = {}
        if d.is_dir():
            files = sorted(d.glob("*.json"))
            lo: Date | None = None
            hi: Date | None = None
            for p in files:
                # Slug: {to}_{from}_{hash}.json
                parts = p.stem.split("_")
                if len(parts) >= 2:
                    try:
                        to = Date.fromisoformat(parts[0])
                        fr = Date.fromisoformat(parts[1])
                        lo = fr if lo is None else min(lo, fr)
                        hi = to if hi is None else max(hi, to)
                    except ValueError:
                        continue
            # No per-symbol coverage; report the cached freshness window and
            # file count under a synthetic key so the caller can still print it.
            if lo is not None and hi is not None:
                symbols[f"({len(files)} cached queries)"] = [(lo, hi)]
        return KindCoverage(kind=kind, source=source_name, cache_dir=d, symbols=symbols)

    # Computed kinds (ratios, news_sentiment) have no own cache; unknown custom
    # kinds have an opaque layout. Either way, nothing to report here.
    return KindCoverage(kind=kind, source=source_name, cache_dir=root / kind, symbols={})


def coverage_summary(
    *,
    source_name: str,
    cache_root: Path,
    symbol: str | None = None,
) -> KindCoverage:
    """Convenience: look up the kind from the catalog, then read coverage."""
    from fintel.market import catalog

    info = catalog.source(source_name)
    return coverage_for_kind(
        kind=info.kind, source_name=source_name, cache_root=cache_root, symbol=symbol
    )
