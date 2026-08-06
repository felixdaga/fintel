# Guide: add a strategy package

A strategy package is an external directory the platform loads, validates, and
freezes. It owns the investment strategy — universe, schedule, data surface,
mission, output contract, scoring. The platform owns the harness: PIT clamp,
concurrency, and evaluation mechanics.

This guide walks through the ideas using the demo package
`packages/systematic_stockrate_djia_weekly/` — a weekly single-name rating of
the Dow 30. Copy that package when you want a full fundamental + trajectory
data surface to start from.

Companion docs: [`architecture.md`](architecture.md) §3 / §9,
[`data_pipeline.md`](data_pipeline.md).

## 1. What a package is

A directory with three required files and any optional helpers:

```
packages/systematic_stockrate_djia_weekly/
  strategy.toml          # required — manifest (universe, schedule, data, scoring)
  mission.md             # required — agent character / scoring rubric
  output_schema.json     # required — submit_views / View contract
  strategy.lock          # written by load_and_prepare (identity digests)
  company_names.json     # optional — display names for evidence packs
  my_sources.py          # optional — bring-your-own DataSource(s)
  scoring.py             # optional — custom signal / KPI callables
```

The platform never edits these files. It loads them, validates them, and
freezes a `strategy.lock` recording what was loaded. Three concerns map to
three files:

- **`strategy.toml`** — *what* and *when*: universe, schedule, data kinds,
  scoring wiring.
- **`mission.md`** — *how to judge*: the rubric the agent reasons against.
- **`output_schema.json`** — *what to hand back*: the shape of a `View`.

## 2. `strategy.toml` — the manifest

The manifest is the wiring. Walk through the demo's:

```toml
schema_version = 1
name = "systematic_stockrate_djia_weekly"
description = "Score each DJIA-30 constituent independently, weekly …"

[universe]
preset = "dow30"                  # the Dow 30; or: symbols = ["AAPL", "MSFT"]

[decision]
scope = "single_name"             # one cell per ticker; or portfolio (one cell for all)
schedule = { kind = "custom_dates", start = "2026-04-24", end = "2026-07-31", dates = [
    "2026-04-24", "2026-05-01", "2026-05-08", …   # 15 weekly Fridays, holiday-snapped
] }
```

### Universe

`preset = "dow30"` resolves the Dow 30 membership *at each decision date* —
membership changes over time, so the platform re-resolves per trial. Use
`symbols = [...]` for a fixed list, or register your own universe preset
(see §6).

### Schedule

The demo uses `kind = "custom_dates"` because there is no builtin `weekly`
cadence — the dates are the source of truth. `--start`/`--end` on the CLI may
narrow the grid but never widen it. The builtin alternative is
`kind = "quarterly"` with `start`/`end`, which generates quarter-end dates.

### Data bindings

Each `[[data]]` block binds a **kind** (what the agent asks for) to a
**source** (who answers). The demo binds seven kinds across two lanes:

```toml
# quantitative lane — fundamentals + trajectory
[[data]]
kind = "prices"
source = "massive_prices"
lookback_days = 365

[[data]]
kind = "fundamentals"
source = "massive_fundamentals"
lookback_days = 540

[[data]]
kind = "ratios"                    # computed from prices + fundamentals
source = "valuation_ratios"
window_days = 365
lookback_days = 365

[[data]]
kind = "macro"
source = "fred_macro"
lookback_days = 90

[[data]]
kind = "news_sentiment"            # computed from news
source = "news_sentiment"
lookback_days = 90

# qualitative lane — context
[[data]]
kind = "news"
source = "massive_news"
lookback_days = 14

[[data]]
kind = "web_search"
source = "web_search"
lookback_days = 30
max_results = 5
clamp_by_age = true                # PIT knob: drop results whose age is outside the window
```

Rules:

- Unknown source names fail preflight — typos are errors, not silent.
- A kind may be bound once per package.
- Computed kinds (`ratios`, `news_sentiment`) declare upstreams — those
  upstream kinds (`prices`, `fundamentals`, `news`) must also be bound
  explicitly. The platform will not default them.
- `lookback_days` is the single lookback every consumer uses — prefetch,
  probe, tools, evidence — so they can't drift. See
  [`data_pipeline.md`](data_pipeline.md).
- Extra keys (e.g. `max_results`, `clamp_by_age`, `window_days`) become
  params for the resolved source — the extension seam.

Browse the catalog before picking:

```python
from fintel.market import catalog

catalog.register_builtins()
for s in catalog.sources():
    print(s.name, s.kind, s.required_env)
```

Env vars for bound sources are reported by preflight (`MASSIVE_API_KEY`,
`BRAVE_API_KEY`, `FRED_API_KEY`, …). Put them in `.env/keys.env` (see
`.env.example`).

### Scoring

The `[scoring]` block is wiring only — it names which callables run, it
doesn't contain the math:

```toml
[scoring]
signal = "single_name"             # builtin: view score is the signal
kpi = "single_name_ir"            # builtin: cross-sectional Spearman IC + ICIR
metric_key = "icir"
transform = "single_name"          # builtin: identity for single-name
horizons = [1, 2, 4, 8]          # steps on the decision-date grid (weeks here)
params = { holdings = true, active_budget = 0.5, cost_bps = 5.0 }
```

`horizons` are the measurable forward returns the KPI scores — they are
*not* the agent's view horizon. A fundamental view plays out over quarters;
the KPI scores it on whatever forward grid exists. The platform owns the
mechanics around the signal (transform, ensemble across K, holdings, returns);
the package owns what the signal *is* and what *good* means. See
[`architecture.md`](architecture.md) §7.

## 3. `mission.md` — the rubric

The mission is the agent's character. The platform passes it through
unchanged — it never edits it. The demo's mission opens:

> You are a systematic equity research analyst. You will be asked,
> independently for one company at a time, to rate that company's
> fundamental attractiveness and the trajectory of its business on a
> continuous scale from -1 to +1.

It then names three pillars in order of importance (fundamental health &
trajectory, valuation vs own history, near-term rerating chance), gives
score anchors (`+1.0` extremely attractive … `−1.0` extremely unattractive),
sets discipline rules (start from pillar 1, apply pillar 3 last as a modest
adjuster), and enforces anti-momentum + output hygiene.

A good mission tells the agent:

- **What to judge** — the pillars and their order.
- **How to score** — anchors on the `[-1, +1]` range, with permission to use
  in-between values.
- **What to ignore** — e.g. "price momentum alone is never a rerating
  catalyst."
- **What to cite** — point-in-time evidence only, grounded in what was
  actually shown.
- **What not to do** — e.g. "do not discuss position sizing; that is handled
  downstream."

## 4. `output_schema.json` — the View contract

JSON Schema for each submitted `View`. The evaluation layer needs at minimum
`symbol` + `score` in `[-1, 1]`. Extra fields are for audit / pitfall mining.

The demo's schema requires `symbol`, `score`, `rationale`, and optionally
accepts `key_factors` (array of strings) and `sources_cited` (array of
`SourceRef` objects pointing at the datum that influenced the view):

```json
{
  "properties": {
    "symbol": { "type": "string" },
    "score": { "type": "number", "minimum": -1.0, "maximum": 1.0 },
    "rationale": { "type": "string" },
    "key_factors": { "type": "array", "items": { "type": "string" } },
    "sources_cited": { "type": "array", "items": { "$ref": "#/$defs/SourceRef" } }
  },
  "required": ["symbol", "score", "rationale"]
}
```

Ship a schema that matches what your agent actually emits. Unused fields
cost tokens and invite schema failures. The demo deliberately omits
`conviction` / `time_horizon` — unused by its signal and not worth the PM
tokens.

## 5. Bring your own data

The demo binds builtin catalog sources (`massive_prices`, `fred_macro`, …).
To wire your own data, pick one of two extension seams.

### A. Unregistered `module:Callable` source (package-local)

Point `source` at an import path. Loading the manifest does not import it;
preflight / build resolve it when needed.

```toml
[[data]]
kind = "prices"
source = "mypackage.my_sources:MyPrices"
lookback_days = 365
root = "data/prices"              # extra keys become constructor params
```

Your callable must satisfy the `DataSource` protocol
(`fintel.market.data.base.DataSource`):

```python
# packages/my_strategy/my_sources.py
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

## 6. Custom signal / KPI / universe (optional)

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
The platform never inspects the math — see [`architecture.md`](architecture.md) §7.

Custom universe presets register the same way:

```python
from fintel.market.catalog import UniverseInfo, register_universe

register_universe(UniverseInfo(name="my_basket", target="mypackage.universes:my_basket"))
```

## 7. Validate

```bash
# From repo root, with keys bootstrapped or already in the shell:
python - <<'PY'
from fintel.strategy import load_and_prepare
paths, result, lock = load_and_prepare(
    "packages/systematic_stockrate_djia_weekly", write_lock=True
)
print(result.ok, result.problems, result.warnings)
print("required_env:", result.required_env)
print("lock:", lock.strategy_digest)
PY
```

Fix every preflight problem before spending LLM tokens. Then smoke with a
tiny universe/date override:

```bash
fintel simulation packages/systematic_stockrate_djia_weekly \
  --agent optimized \
  --universe AAPL,MSFT \
  --dates 2026-05-01 \
  --job-id djia-smoke \
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
