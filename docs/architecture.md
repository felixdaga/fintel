# fintel — schema & architecture

Agent evaluation where the benchmark is an investment outcome. You bring a
**strategy package**, you name an **agent**, the platform runs the evaluation
and scores it.

## 1. The boundary

Three owners, not two. Getting the third one wrong is what makes a platform
leak strategy assumptions.

**Platform** — what a strategy cannot be trusted to enforce against itself,
plus the plumbing every strategy would otherwise rebuild:

- point-in-time guarantee: clamping, and the post-run trace audit
- isolation and reproducibility: `config.json` / `lock.json` / `result.json`
- orchestration: Job → Run → Trial → Cell, retry, resume, K repeats
- data serving: cache, PIT clamp, kind → tool / evidence exposure
- agent adapters
- the generic K-repeat / ensemble scoring harness
- the generic renderer

**Strategy package** — the investment judgment:

- mission text and output contract
- decision scope (single-name vs portfolio)
- which data kinds, and their params
- the KPI: what "good" means
- any extra scoring or report sections it contributes

**Agent / runtime** — how the mission actually gets executed:

- model, sampling, thinking depth
- what the agent consumes from what's offered: whether memory is fed, whether
  it reads tools or a pre-rendered evidence pack, how it decomposes internally
- anything in `AgentSpec.options`, which the platform never inspects

The same package run under two agents is the same evaluation; the agents may
consume the offered world differently. That is the agent's business, declared by
its adapter — not the package's, and not the platform's.

**Conformance test.** Two dissimilar packages must both run end to end with zero
edits under `fintel/`. If one needs a platform change, the boundary is wrong.

## 2. Execution hierarchy

```
Job          one `fintel backtest` invocation: package × agent × market × K
 └─ Run      one of K repeats — the stochasticity unit
     └─ Trial        one decision date
         └─ Cell     one agent invocation
                     scope=single_name  -> one cell per symbol
                     scope=portfolio    -> one cell for the whole universe
```

Cell is a real orchestration level. That is what makes fan-out agent-agnostic,
puts missing-result retry where the missing thing actually lives, and lets one
concurrency engine cover every level.

## 3. The manifest

```toml
name = "systematic_stockrate_djia"
schema_version = 1

[decision]
scope = "single_name"

[decision.schedule]
kind = "custom_dates"
dates = ["2022-07-01", "2022-10-01"]

[universe]
preset = "dow30"              # active constituents as of each decision date

[[data]]
kind = "prices"
source = "massive_prices"     # swap for any source serving `prices`
lookback_days = 365

[[data]]
kind = "fundamentals"
source = "massive_fundamentals"

[scoring]
kpi = "single_name_ir"        # builtin name, or "mypkg.scoring:FactorNeutralIR"
transform = "single_name"
horizons = [1, 2, 3, 4]
```

`extra="forbid"` on the manifest, so a typo is an error rather than a silently
ignored key. Unknown keys inside a `[[data]]` / `[universe]` / `[decision.schedule]`
block are kept as `params` and handed to whatever the factory resolves — that is
the extension seam, and it means a new source needs no schema change here.

## 4. The market library

Three things are separable, and keeping them separate is what makes providers
swappable:

| | | |
|---|---|---|
| **kind** | what the agent asks for | `prices` |
| **source** | who answers | `massive_prices`, `yfinance_prices`, `synthetic_prices` |
| **fields** | what comes back | `open`, `high`, `low`, `close`, `volume` |

`market/catalog.py` is the library: every source registers its kind, provider,
field roster, accepted params and required env vars, so you can browse what
exists before picking. `market.catalog.sources(kind="prices")` answers "what
could serve this", and registering a `SourceInfo` adds to the library without
editing the platform. A source with no declared fields is refused — an
undocumented source can't be picked from a catalog.

The manifest then binds kind → source, and the factory rejects a binding whose
source serves a different kind, or params the source doesn't accept, before any
agent asks a question and silently gets nothing.

**Computed kinds** are sources like any other; they just declare
`derives_from=("prices", "fundamentals")` instead of a vendor. The factory
builds plain sources first, then injects the upstreams. Upstream kinds must be
bound explicitly in the same manifest — defaulting them would let a package's
ratios quietly change provider.

```
DataSource.fetch(query, cutoff)         kind-keyed, PIT-clamped
        │
   environment/access.py                the ONE agent-facing data path
        ├── mcp/            get_data(kind, **q) + per-kind tools, generated
        │                   from the run's declared bindings
        └── evidence/       pre-rendered text pack for non-tool agents
```

`fetch` takes a `Cutoff`, never a bare date, so a source cannot be called
without the point-in-time boundary and the boundary can't be mistaken for an
ordinary parameter.

### The one deliberate exception

Scoring must read *after* the decision date — that's the measurement.
`market/realized.py` is the only module allowed to, and it is deliberately
separate: different module, different name (`PriceLookup.price_at`, not
`fetch`), and a test forbidding `environment/`, `agents/` and `pit/` from
importing it. In the old code the clamped and unclamped paths were two methods
on one class, one typo apart.

## 5. Layers

Imports flow one way only. Enforced by `tests/test_architecture.py`.

```
L0   models/       pure schemas. no I/O.
L1   utils/        generic. no domain knowledge.
L2   pit/          the platform guarantee: clamp + audit
L3   market/       calendar, universes, schedules, data catalog, realized prices
L4   environment/  session, access, mcp, evidence, isolation
L5   agents/       adapters + agent-side context
L6   strategy/     package load, lock, preflight
L7   evaluate/     job, run, trial, cell, queue, store
L8   scoring/      KPI protocol + generic ensemble harness + builtins
L9   report/       renderers
L10  cli/          parse flags, build config, call evaluate. nothing else.
```

`scoring/` reads artifacts, never imports `evaluate/`. That keeps it pure and
re-runnable against finished runs.

## 6. Extension points

Two shapes, both resolving `module:Callable` through `utils/import_path`:

- **Catalog registration** for things you browse before choosing — data sources
  and universes. Carries metadata, so `--list-sources` is possible.
- **A `factory.py` name map** for things chosen structurally — schedules, agents,
  KPIs, environments.

Either way the live path goes through the same resolver, and a custom callable
receives its declared params plus `config=` only if it asks for one — so a
trivial extension stays trivial while a cache-backed one still gets the cache
root.

`strategy` · `market.catalog` (sources, universes) · `market.schedule` ·
`environment` · `agents` · `scoring` · `environment.evidence`

## 7. Artifacts

Every level writes the same trio: `config.json` (asked), `lock.json`
(digests, for replay comparison), `result.json` (happened).

```
runs/<job_id>/
  config.json  result.json  job.log
  r1/ … rK/
    config.json  lock.json  result.json  run.log
    trials/<YYYY-MM-DD>/
      decision.json
      result.json
      trace/<cell>.jsonl
  report/
    report.md  report.json
```

`r1` exists even when K=1 — predictable tooling beats a special case.
`RunConfig` records the effective world post-override and is self-describing, so
`fintel report <job_id>` reads identity from the lock and needs no `--strategy`.

## 8. CLI

```
fintel backtest <package> --agent <name> [--k N] [--universe ...] [--dry-run]
                                        [--agent-opt k=v]
fintel report   <job_id|run_dir...> [--horizons 1,2,3] [--format console|markdown]
fintel runs     list | show <job_id>
```

Agent-specific settings go through `--agent-opt`, validated against a schema the
agent adapter declares. Adding an agent requires no CLI edit.

## 9. Invariants

Each has a test in `tests/test_architecture.py`. Convention alone is what let the
previous attempt drift.

1. No data at or after `decision_date` reaches an agent.
2. The platform never edits the agent's character — mission is passed in.
3. `models/` imports no logic.
4. No cross-package `_private` imports.
5. `scoring/` and `pit/` never import `evaluate/`.
6. No scoring change merges without the ICIR reference test passing.
7. One naming scheme per identity, in `models/ids.py`.
8. Every `module:Callable` a factory or the catalog advertises resolves.
9. Nothing outside `_legacy/` imports `_legacy/`.
10. Only `market/realized.py` may read past the decision date, and nothing
    agent-facing may import it.
11. Every registered data source declares its fields.

## 10. Not yet wired

Nothing enters the tree without a consumer and a test. These are deliberately
absent rather than stubbed, because an unwired declaration is worse than a
missing one — it reads as finished.

| Absent | Decide when |
|---|---|
| **Capability declaration** (memory fed / tools vs evidence / web) | The first agent adapter lands. It belongs to the agent, so it should be shaped by a real adapter, not guessed at in `models/`. |
| **Portfolio state** — `Portfolio`, `Position`, `TargetWeights`, `Decision.portfolio_before` | A package + agent pair actually needs position carryover. How state is retrieved is strategy-dependent and not yet specified; the old code shipped tools serving a book that never advanced. |
| **Sizing** — views → weights | Same trigger. Sizing is judgment, so it lands as strategy-owned when portfolio state does. |
| `initial_cash`, `cost_bps` on the manifest | There is no execution engine to consume them. |
| **`EnvironmentSpec`** / isolation options in `JobConfig` | `environment/factory.py` exists. Until then slot sizing has no home and shouldn't get a placeholder. |
| **Memory levels** beyond what an adapter uses | With capabilities agent-owned, the old `off/log/agent_authored/feedback` ladder is an agent concern. `agent_authored` was accepted in config and raised at runtime in the old repo — don't reintroduce that shape. |
| **Valuation ratios** as a computed kind | The composition seam is built and tested; the port is ~600 lines of pure TTM math (`valuation/ratios.py`, with a `RATIO_FIELDS` roster that becomes the catalog field list). Deliberately a separate step so the math can be diffed against the original rather than paraphrased. |
| **`yfinance_prices`** | One `SourceInfo` plus a bars fetcher writing to the same `PriceStore`. The seam is proven by a second registered `prices` source in the tests; this is the real vendor, and it wants a network test to be worth anything. |
| **`filing_text`, `news_sentiment`, `web_search`** | All three exist in the old repo and none are agent-reachable yet, since `environment/access.py` is what exposes a kind. Port alongside that layer, not before. |
| **Symbol renames** (`META`/`FB` pre-2022-06-09) | The old code handled this with a SPLICE map in study scripts, so the runtime path silently served garbage for pre-rename `META`. Belongs in the data layer, once there's a rename table to key it on. |
| **Index weights** | The constituents dataset carries membership only. `UniverseReport.weights_available` is `False` and nothing pretends otherwise; a price/cap-weighted benchmark needs a different source. |
