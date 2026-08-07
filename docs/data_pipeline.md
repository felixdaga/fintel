# Data pipeline

How a strategy's data declaration becomes a warm cache, a tool surface, and a
PIT-clamped read — and where the lookback comes from.

## Feed-level cache (`fintel/market/cache/`)

Caching policy lives in **one module**, not inside each vendor file:

| Helper | Shape | Used by |
|--------|-------|---------|
| `ensure_records` / `CachedRecordsFeed` | gap-aware `RecordCache` | fundamentals, news, filing_text, macro |
| `ensure_prices` / `CachedPricesFeed` | gap-aware `PriceStore` | prices |
| `ensure_query_blob` | exact-key JSON file | web_search |

Vendor sources implement **network fetch only** (`_fetch_span` / `_fetch_bars` /
`_search`). They call the cache helper with a few lines — offline miss, short
cache warn, and merge under lock are shared. Prefetch warms via each source's
public `ensure` / `warm`, not private `_ensure` isinstance branches.

## One cache root

`runs/cache/` is the single central cache (default `<output-root>/cache`;
override with `--cache-root` or the `FINTEL_CACHE` env var). Every kind lives
under it:

```
runs/cache/
  prices/{SYMBOL}.parquet        + {SYMBOL}.coverage.json sidecar
  fundamentals/{SYMBOL}.json      (coverage embedded as _coverage)
  news/{SYMBOL}.json
  filing_text/{SYMBOL}.json
  macro/{SERIES_ID}.json          + {SERIES_ID}.meta.json (FRED; series-keyed)
  web_search/{to}_{from}_{hash}.json   (keyed by query, not symbol)
```

There is no second cache tree. `fintel cache status` reads the coverage sidecars
back out, gap-aware:

```
fintel cache status
fintel cache status --source massive_news --symbol AAPL --window 2024-01-01..2026-01-01
```

Coverage is a list of `[from, through]` intervals, not a lone min/max — a cache
warmed through 2026 with a hole in 2024 honestly reports the gap, and a request
in the hole fetches instead of returning empty.

## One lookback per kind

The strategy's `[[data]].lookback_days` is the data range. The catalog
`Param.default` is the fallback when the strategy omits it. **That single value
is what every caller uses.** `catalog.resolve_lookback(binding)` is the one
resolver; the factory bakes the resolved value into each source instance, and
every consumer reads it from there:

| Consumer | Where it reads the lookback |
|----------|-----------------------------|
| Prefetch warm window | source instance `lookback_days` (binding-baked) |
| Probe query | source instance `lookback_days` |
| Tool schema default | source instance `lookback_days` (not the catalog default) |
| Access cap | `policy.lookback_caps[kind]` (callers may request less, never more) |
| Evidence pack (optimized) | `env.policy.lookback_cap_map[kind]` |

There is no separate `spec.lookback_days` override, no `EvidenceConfig.*_lookback_days`,
and no tunable `max_lookback_days` to drift out of agreement. `max_lookback_days`
remains only as a fallback cap for kinds with no binding-declared lookback (a
custom `module:Callable` source).

`window_days` (ratios) is **not** a lookback — it's a computation param (the
trailing P/E averaging window) and stays strategy-owned, `per_call=False`.
`filings_lookback_days` (ratios) is internal to `ValuationRatios`, derived from
`lookback + window`, and not strategy-visible.

## The flow

```
strategy.toml [[data]] blocks
        │
        ▼
preflight: catalog.check_bindings  +  catalog.required_env
        │
        ▼
factory.build_data_sources
  · catalog.resolve_lookback(binding) → lookback baked into each source
  · PriceStore / RecordCache / WebSearch wired to the central cache_root
        │
        ▼
probe  (market/probe.py)   — one fetch per kind, source-instance lookback
        │
        ▼
prefetch (market/prefetch.py) — warm [from, through] per kind, source-instance lookback
        │
        ▼
build_environment  (environment/factory.py)
  · policy.lookback_caps ← from each source's lookback_days
  · ToolSurface ← catalog SourceInfo + source-instance lookback as the default
        │
        ▼
DataAccess.read  (environment/access.py)
  · policy.check_kind / check_symbol / clamp_query(kind, query)
  · source.fetch(clamped, cutoff)   — PIT injected here, nowhere else
```

## Strategy-supplied data

A package may bind a custom source two ways:

1. **`module:Callable`** in `[[data]].source` — the factory resolves and builds
   it; the source gets the platform's cache root, PIT clamp, and access policy
   for free. Its tool is readable but not advertised as a typed tool (no
   catalog entry to describe it).
2. **`catalog.register_source()`** at import — a package ships a `catalog.py`
   that registers its kinds with fields, params, cache layout, and a
   `module:Callable` target. The strategy then picks by name, and the tool
   surface is generated from the catalog entry.

Either way, the lookback rule is the same: `[[data]].lookback_days` wins,
catalog `Param.default` falls back.
