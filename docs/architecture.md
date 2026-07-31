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
source = "massive_prices"
params = { lookback_days = 365 }

[scoring]
kpi = "single_name_ir"        # builtin name, or "mypkg.scoring:FactorNeutralIR"
transform = "single_name"
horizons = [1, 2, 3, 4]
```

`extra="forbid"` on the manifest, so a typo is an error rather than a silently
ignored key. Unknown keys inside a `[[data]]` / `[universe]` / `[decision.schedule]`
block are kept as `params` and handed to whatever the factory resolves — that is
the extension seam, and it means a new source needs no schema change here.

## 4. Data access — one path, two presentations

```
DataSource.fetch(query, as_of)          kind-keyed, PIT-clamped
        │
   environment/access.py                the ONE data path
        ├── mcp/            get_data(kind, **q) + per-kind tools, generated
        │                   from the run's declared data bindings
        └── evidence/       pre-rendered text pack for non-tool agents
```

Nothing downstream hardcodes a kind name. A package that ships its own
`DataSource` for a novel kind gets caching, PIT, a tool, and an evidence
section without touching the platform.

## 5. Layers

Imports flow one way only. Enforced by `tests/test_architecture.py`.

```
L0  models/       pure schemas. no I/O.
L1  utils/        generic. no domain knowledge.
L2  pit/          the platform guarantee: clamp + trace audit
    market/       universes, schedule, data catalog, valuation
L3  environment/  session, access, mcp, evidence, isolation
L4  agents/       adapters + agent-side context
L5  strategy/     package load, lock, preflight
L6  evaluate/     job, run, trial, cell, queue, store
L7  scoring/      KPI protocol + generic ensemble harness + builtins
L8  report/       renderers
L9  cli/          parse flags, build config, call evaluate. nothing else.
```

`scoring/` reads artifacts, never imports `evaluate/`. That keeps it pure and
re-runnable against finished runs.

## 6. Extension points

One shape everywhere: a `factory.py` with a builtin `{name: "module:Class"}`
map plus `module:Class` import-path resolution, via `utils/import_path.build`.
The live path must go through it — a factory nothing calls is a dead factory.

`strategy` · `market.data` · `market.universe` · `market.schedule` ·
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
8. Every `module:Class` in a factory's `BUILTINS` resolves.
9. Nothing outside `_legacy/` imports `_legacy/`.

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
