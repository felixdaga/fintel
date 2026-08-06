# Guide: add a strategy package

A strategy package is an external directory the platform loads, validates, and
freezes. It owns the investment question (universe, schedule, data surface,
mission, output contract, scoring). The platform owns the harness, PIT clamp,
concurrency, and evaluation mechanics.

Companion docs: [`architecture.md`](architecture.md) §3 / §9,
[`data_pipeline.md`](data_pipeline.md).

## 1. Package layout

```
packages/my_strategy/
  strategy.toml          # required — manifest
  mission.md             # required — agent character / scoring rubric
  output_schema.json     # required — submit_views / View contract
  strategy.lock          # written by load_and_prepare (identity digests)
  company_names.json     # optional — display names for evidence packs
  my_sources.py          # optional — bring-your-own DataSource(s)
  scoring.py             # optional — custom signal / KPI callables
```

Minimal `strategy.toml`:

```toml
schema_version = 1
name = "my_strategy"
description = "…"

[universe]
preset = "dow30"                 # or: symbols = ["AAPL", "MSFT"]

[decision]
scope = "single_name"            # or portfolio
schedule = { kind = "custom_dates", start = "2026-01-01", end = "2026-06-30", dates = [
    "2026-01-02", "2026-01-09",  # …
] }
# also: schedule = { kind = "quarterly", start = "2022-07-01", end = "2026-04-01" }

[[data]]
kind = "prices"
source = "massive_prices"
lookback_days = 365

[[data]]
kind = "fundamentals"
source = "massive_fundamentals"
lookback_days = 540

[scoring]
signal = "single_name"
kpi = "single_name_ir"
metric_key = "icir"
transform = "single_name"
horizons = [1, 2, 4, 8]          # steps on the decision-date grid
params = { holdings = true, active_budget = 0.5, cost_bps = 5.0 }
```

Copy an existing package (`packages/systematic_stockrate_djia_weekly/`) when you
want the full fundamental/trajectory data surface (macro, news_sentiment, news,
web_search) and mission pillars.

## 2. Mission + output schema

- **`mission.md`** — what “good” means for the agent. Passed through unchanged;
  the platform never edits agent character.
- **`output_schema.json`** — JSON Schema for each submitted `View`. At minimum
  the evaluation layer needs `symbol` + `score` in `[-1, 1]`. Extra fields
  (`rationale`, `key_factors`, `sources_cited`, …) are for audit / pitfall mining.

Ship a schema that matches what your agent actually emits. Unused fields cost
tokens and invite schema failures.

## 3. Bind catalog data

`[[data]]` selects a **kind** (what the agent asks for) and a **source** (who
answers). Browse the library:

```python
from fintel.market import catalog
catalog.register_builtins()
for s in catalog.sources():
    print(s.name, s.kind, s.required_env)
```

Rules:

- Unknown source names fail preflight (typos are errors).
- A kind may be bound once per package.
- Computed kinds (`valuation_ratios`, `news_sentiment`) declare upstreams —
  those upstream kinds must also be bound explicitly.
- `lookback_days` on the binding is the single lookback every consumer uses
  (prefetch, probe, tools, evidence). See [`data_pipeline.md`](data_pipeline.md).

Env vars for bound sources are reported by preflight (`MASSIVE_API_KEY`,
`BRAVE_API_KEY`, `FRED_API_KEY`, …). Put them in `.env/keys.env` (see
`.env.example`).

## 4. Bring your own data

Two extension seams — pick one.

### A. Unregistered `module:Callable` source (package-local)

Point `source` at an import path. Loading the manifest does not import it;
preflight / build resolve it when needed.

```toml
[[data]]
kind = "prices"
source = "mypackage.my_sources:MyPrices"
lookback_days = 365
# any extra keys become constructor params
root = "data/prices"
```

Your callable must satisfy the `DataSource` protocol
(`fintel.market.data.base.DataSource`):

```python
# packages/my_strategy/my_sources.py
from datetime import date as Date
from fintel.pit import Cutoff

class MyPrices:
    name = "my_prices"
    kinds = ("prices",)

    def __init__(self, *, lookback_days: int = 365, root: str = "data/prices"):
        self.lookback_days = lookback_days
        self.root = root

    def fetch(self, query: dict, cutoff: Cutoff) -> list[dict]:
        symbol = query["symbol"]
        # MUST respect cutoff.decision_date — no bars on/after that date
        ...
        return [{"date": "...", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}]
```

`fetch` takes a `Cutoff`, never a bare date. PIT is your responsibility for
custom sources — the platform will not silently fix a leaky clamp.

Put the package (or its parent) on `PYTHONPATH`, or install it as a local
package, so `mypackage.my_sources:MyPrices` resolves.

### B. Register in the catalog (reusable across packages)

```python
from fintel.market.catalog import Field, Param, SourceInfo, register_source

register_source(SourceInfo(
    name="my_prices",
    kind="prices",
    target="mypackage.my_sources:MyPrices",
    fields=(Field("date", "date"), Field("open", "number"), …),
    params=(Param("lookback_days", "int", default=365),),
    required_env=(),          # or ("MY_API_KEY",)
))
```

Then packages bind `source = "my_prices"` like any builtin. Prefer catalog
registration when multiple strategies will share the source.

### Ship data with the package

Custom sources may read files under the package directory (CSV/parquet/json).
Keep vendor API keys out of the package — use env vars via `required_env` /
`MarketConfig`. Never commit secrets; `.env/` is gitignored.

## 5. Custom signal / KPI (optional)

Builtins cover single-name score → IR. For anything else, point
`[scoring].signal` / `kpi` / `transform` at `module:Callable`:

```toml
[scoring]
signal = "mypackage.scoring:my_signal"
kpi = "mypackage.scoring:my_kpi"
transform = "rank_range"
horizons = [1, 2, 3]
params = { holdings = true }
```

Signatures match `fintel/evaluate/signals.py` and `fintel/evaluate/kpi.py`.
The platform never inspects the math — see architecture §12.

## 6. Validate

```bash
# From repo root, with keys bootstrapped or already in the shell:
python - <<'PY'
from fintel.strategy import load_and_prepare
paths, result, lock = load_and_prepare("packages/my_strategy", write_lock=True)
print(result.ok, result.problems, result.warnings)
print("required_env:", result.required_env)
print("lock:", lock.strategy_digest)
PY
```

Fix every preflight problem before spending LLM tokens. Then smoke with a tiny
universe/date override:

```bash
fintel simulation packages/my_strategy \
  --agent optimized \
  --universe AAPL,MSFT \
  --dates 2026-01-02 \
  --job-id my-strategy-smoke \
  --cell-concurrency 2
```

## Checklist

- [ ] `strategy.toml` name unique; schedule dates intentional
- [ ] Every `[[data]]` source resolves (catalog or `module:Callable`)
- [ ] Computed-kind upstreams bound explicitly
- [ ] `mission.md` + `output_schema.json` agree with the agent
- [ ] Required env vars documented / present in `.env/keys.env`
- [ ] Custom sources clamp on `cutoff.decision_date`
- [ ] `load_and_prepare` is green; `strategy.lock` written
- [ ] Smoke run on 1–2 names × 1 date before a full job
