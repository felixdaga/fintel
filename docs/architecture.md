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

The catalog is the whole library, not a subset — a kind is in it only once it's
fetchable under PIT control, because "declared but unwired" is the failure mode
where a strategy runs and quietly sees nothing:

| kind | source | PIT clamp |
|---|---|---|
| `prices` | `massive_prices`, `synthetic_prices` | bar date `< decision_date` |
| `fundamentals` | `massive_fundamentals` | `filing_date < decision_date` |
| `news` | `massive_news` | `published_at < decision_date` |
| `filing_text` | `massive_filing_text` | `filing_date < decision_date` |
| `ratios` | `valuation_ratios` (computed) | inherited from upstreams |
| `news_sentiment` | `news_sentiment` (computed) | inherited from upstream |
| `web_search` | `web_search` | provider freshness window ends `decision_date - 1` |

`web_search` is the one kind PIT can't be enforced on after the fact: results
carry no reliable date to clamp, so the provider's freshness window *is* the
control, and the cache is keyed by that window so a replay can't widen it.

### Strategy selects, catalog validates

The manifest binds kind → source and may set params; it cannot invent a source.
`catalog.check_bindings()` returns every finding at once — unknown source, wrong
kind for the source, a param the source doesn't accept, a kind bound twice, a
computed kind whose upstream is unbound — so one preflight tells you everything
to fix rather than one thing per run. `catalog.required_env()` reports the
credentials a binding list needs. A bare name not in the catalog is a typo and is
rejected; a `module:Callable` is how a package ships its own source and is
allowed.

**Computed kinds** are sources like any other; they just declare
`derives_from=("prices", "fundamentals")` instead of a vendor. The factory
builds plain sources first, then injects the upstreams. Upstream kinds must be
bound explicitly in the same manifest — defaulting them would let a package's
ratios quietly change provider.

Computed kinds derive on demand from already-clamped upstream data. The old
pipeline instead precomputed daily `ratios/` and `news_sentiment/` series from
the full cache and re-filtered per day, which meant a second PIT implementation
to keep correct alongside the first.

```
DataSource.fetch(query, cutoff)         kind-keyed, PIT-clamped
        │
   environment/access.py                the ONE agent-facing data path
        ├── tools.py        typed tools, generated from the run's bindings
        └── evidence.py     pre-rendered text pack for non-tool agents
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

## 5. The environment

Where the three inputs meet, for exactly one agent invocation:

| | brings |
|---|---|
| **strategy** | which kinds, with which params; what one cell decides on |
| **market** | the universe *at the decision date*, and the bound clamped sources |
| **runtime** | where the cell may write, and how much it may ask for |

A `Cell` is the identity of one invocation — run, date, symbols, scope — and it
owns the `Cutoff`. That's the load-bearing decision in this layer. The old MCP
server worked out which cell it was serving by walking process ancestry for the
CLI's pid and reading `cli-pids/<pid>/bundle.json`; under a gateway that spawns
the server itself the walk finds nothing, so every concurrent cell fell back to
one shared directory and contaminated its neighbours. It then cached the first
bundle it ever loaded, so a reused process served a stale decision date. Cell
identity is now constructed, frozen and passed explicitly, and a test forbids any
module but `cell.py` from building a `Cutoff`.

**Reads are policy-checked, then clamped, then recorded** — one path,
`DataAccess.read`. The old code implemented the clamp in the data source, again
in the bundle builder, again per-tool in the server, again in the dossier
renderers and again in the memory store. They disagreed: `get_fundamentals` used
`filing_date <= decision_date` where its source used `<`, `get_filings` had no
upper bound and trusted its input, the bundled price path served frozen bars
without re-checking, and a `web_search` cache hit skipped the check its live path
performed.

**Absence and failure are different answers.** Every read returns a `Reading`
whose status is `ok`, `empty`, `failed` or `denied`. The old tools returned `[]`
for a missing API key, a source that raised, an unconfigured provider, and a
company with genuinely no news — indistinguishable to the agent and to a reviewer
reading the trace afterwards. `AccessLog.summary()` reports `degraded` so a run
that half-failed is never scored as a run against a quiet market.

**The universe is enforced on reads.** The old server gated it only at submission
— "data tools accept any symbol" — so an agent scored on the Dow could read
anything cached. Widening it is a strategy's explicit choice via `peers`, which
grants reads without granting decisions.

**The tool surface is generated** from the bindings and the catalog, so a run that
binds three kinds has three tools, each described by the entry of the source
actually bound. The old list of twenty static tools had to be kept in agreement
with the catalog, the builder and the prompt text, and drifted from all three. A
param the strategy owns — the trailing window behind a P/E — is marked
`per_call=False` and never offered to an agent, since two readings in one run must
mean the same thing.

### Three delivery channels, one path

Agents differ in how they want data, not in what they may see. All three channels
are thin presentations of `DataAccess.read`:

| channel | for | built from |
|---|---|---|
| in-process call | a desk that fetches directly in Python | `access.read(kind, **q)` |
| typed tools | anything doing function-calling — MCP, an LLM's native tools, or an in-process framework wrapper | `ToolSurface.descriptors()` + `call()` |
| rendered text | a single-turn agent that needs everything in one dump | `evidence.build(access)` |

This is why `tools.py` carries no transport. The old repo had an MCP server *and*
a LangChain `Toolkit` reimplementing the same tools over the same session, each
with its own PIT filters to keep in agreement — two implementations of one idea,
which is how they drifted apart. A multi-role desk narrows its per-role tool set
with `ToolSurface.subset(kinds)`, which shares the underlying access, so a role
boundary can't become a second data path with its own rules.

There is no data bundle. Pre-freezing market data into a per-cell file was one of
three jobs the old `bundle.json` did, and it was already vestigial: `fundamentals`
and `ratios` were written on every cell and read by no tool, and `news` was always
`{}`. The other two jobs survive — cell identity is `cell.json`, and a subprocess
that needs to rebuild access gets the bindings, which are declarative and small.

Repeated identical reads inside one cell are memoized, so an agent re-reading
prices mid-reasoning cannot see two different answers. Failures are never
memoized — a transient blip must not become an outage for the rest of the cell.

### Nothing shared between concurrent cells

Every artifact a cell writes is named after that cell: `cells/<cell>.json` and
`trace/<cell>.jsonl`. The old layout had every symbol on a date write into one
`decisions/<date>.json`, so concurrent cells overwrote each other and a run could
finish with views missing and no error at all. `decision.json` still exists but is
a *reduction*, written once after the fan-in.

The shared cache needs two separate guarantees, and it turned out to be missing
both:

* **Atomic writes**, so a reader — which takes no lock — can never observe a
  half-written file. Parquet was being written in place, so a concurrent reader
  saw a truncated file, logged it as unreadable, and treated a populated cache as
  a miss.
* **A lock around read-modify-write**, because atomicity alone doesn't prevent a
  lost update: two cells fetching different spans of one symbol both read the old
  coverage, both append their own, and the last writer erases the other's records.
  Merging therefore lives on the store, which re-reads inside the lock, rather
  than in each source.

`tests/test_concurrency.py` covers both. The lost-update test was verified to
fail 25 times out of 25 with the lock removed, so it is a real guard and not a
test that merely passes.

## 6. Layers

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

## 7. Extension points

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

### Where agent adapters live

`agents/` holds one adapter per agent, the way Harbor's `agents/installed/`
holds Claude Code, Cursor, OpenClaw and ~27 others behind one `AgentFactory`
name map. Registration copies Harbor almost verbatim: a `name -> "module:Class"`
map plus a `module:Class` escape hatch, so a custom agent needs no entry.

The contract is `decide(environment) -> AgentResponse`. A returned value, not a
`submit()` callback — the platform can then check what came back, and an adapter
cannot forget to produce anything. The `Environment` is the whole capability
surface, so an adapter chooses a channel rather than reaching for data itself.

Where Harbor's shape does **not** transfer: its adapters vary in how an agent is
*installed and launched*, never in how it gets data, because a Harbor agent gets
a terminal and the data is the filesystem. Ours vary mostly in *data delivery*,
and some have no CLI at all — an in-process LangGraph desk needs no install step
and no argv. So `BaseInstalledAgent` maps to one family of ours, not to all of
them, and it belongs in `agents/installed/` rather than at the root.

Worth taking from Harbor:

* **Declarative `CLI_FLAGS` / `ENV_VARS`** descriptors that build argv and env
  from kwargs, instead of a hand-rolled config class per adapter.
* **`ERROR_PATTERNS` mapping output to typed errors.** The important one. It is
  the same empty-vs-failed distinction `DataAccess` already makes, applied to
  agents: a rate limit is retryable, a safety refusal is a real abstention and
  must *not* be retried, and a context-window overflow is a config bug, not a
  flake. The old runner collapsed all three into empty views and retried blindly.
* **`SUPPORTS_*` class flags**, checked before launch so a mismatch fails fast.
  This is what "capabilities" honestly is — a property the adapter declares, not
  something a strategy requests.
* **`install()` / `setup()` split from `run()`**, so a CLI is installed once per
  trial rather than per cell.

Deliberately *not* in the adapter: per-symbol fan-out. Four old adapters each
grew their own `ThreadPoolExecutor` and progress plumbing. Cells are the
platform's unit, so `evaluate/` fans out and an adapter only ever sees one cell.

### Seven agents, two hosts

The old repo's agents look like seven kinds of thing and are really points in a
small cube: *where the code runs*, *how data arrives*, *how the answer returns*.
Collapsed, only two hosts matter.

* **In-process** — the LangGraph desks, the single-call HTTP agent, the
  baselines. These consume `DataAccess` and tool objects directly, which is why
  we do not copy Harbor's choice to subprocess its LangGraph adapter; Harbor does
  that because its agents need a terminal and arbitrary dependencies.
* **Subprocess CLI + MCP config** — OpenClaw and Claude Code. Same host, and
  they differ only in config file format and whether the instruction arrives by
  argv or stdin.

`openclaw_per_ticker` was never a host, just fan-out. Scope lives on the `Cell`,
so one adapter serves both single-name and portfolio instead of two config kinds.

### What makes agents comparable

Running one strategy against several agents is the product, so the axis that
must not vary is the environment. Four things enforce that:

1. **`agents/` cannot import `market/` or `pit/`.** Guarded in
   `tests/test_architecture.py`. An adapter able to build a `DataSource` or a
   `Cutoff` can fetch its own data, and then a comparison measures the evidence
   channel instead of the agent. The old repo lost precisely this: one agent's
   toolkit reached the live web while another had no web access at all.
2. **Typed outcomes.** `Outcome` separates `abstained` and `refused` — real
   answers — from `timeout`, `rate_limited`, `parse_error` and `crashed`. Only
   `RETRYABLE` outcomes are retried, because re-rolling a refusal does not
   recover an answer, it manufactures a different one. `AgentResponse` refuses to
   be `ok` with no views, so the ambiguity the old runner lived with is
   unrepresentable rather than merely discouraged.
3. **A cost basis.** `Usage.basis` records whether a price was `reported` by the
   provider or `estimated` from a rate card, and a sum of both degrades to
   `mixed` and stops being `comparable`. A subprocess CLI usually gives tokens
   but no price while an HTTP agent gives the real charge, so this is the normal
   case, not an edge one. The old rollup stamped the total `authoritative`
   whenever *any* leg reported a cost.
4. **The channel is a platform knob.** Because tools and pack are two renderings
   of one `DataAccess`, the same adapter can be run over either. The old code
   could only ablate this *inside* one agent, via `evidence_mode`, so the
   comparison never transferred to another agent.

`agents/run.py:invoke` is the only place an agent is called and the only place an
exception becomes an outcome. It never raises, because in the old runner an
adapter exception aborted the whole date and erased every other symbol's work.

`ScriptedAgent` can reach every outcome and every channel on demand, so error
handling is exercised on every commit rather than only when a provider happens to
rate-limit us. `ConstantAgent` is a genuine baseline, not a stub: an agent that
cannot beat a fixed score has shown nothing.

## 8. Artifacts

Every level writes the same trio: `config.json` (asked), `lock.json`
(digests, for replay comparison), `result.json` (happened).

```
runs/<job_id>/
  config.json  result.json  job.log
  r1/ … rK/
    config.json  lock.json  result.json  run.log
    trials/<YYYY-MM-DD>/
      decision.json          reduced once, after the fan-in
      result.json
      cells/<cell>.json      one writer each
      trace/<cell>.jsonl     one writer each
  report/
    report.md  report.json
```

`r1` exists even when K=1 — predictable tooling beats a special case.
`RunConfig` records the effective world post-override and is self-describing, so
`fintel report <job_id>` reads identity from the lock and needs no `--strategy`.

## 9. CLI

```
fintel backtest <package> --agent <name> [--k N] [--universe ...] [--dry-run]
                                        [--agent-opt k=v]
fintel report   <job_id|run_dir...> [--horizons 1,2,3] [--format console|markdown]
fintel runs     list | show <job_id>
```

Agent-specific settings go through `--agent-opt`, validated against a schema the
agent adapter declares. Adding an agent requires no CLI edit.

## 10. Invariants

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
12. Only `environment/cell.py` may construct a `Cutoff`. Everywhere else it
    arrives from the cell, so PIT has exactly one point of decision.

## 11. Not yet wired

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
| **`yfinance_prices`** | One `SourceInfo` plus a bars fetcher writing to the same `PriceStore`. The seam is proven by a second registered `prices` source in the tests; this is the real vendor, and it wants a network test to be worth anything. |
| **Tool transport** (MCP wiring) | `environment/tools.py` produces the descriptors and dispatches calls; binding those to a protocol needs a real agent adapter, so it lands with `agents/`. Kept transport-free so the surface is testable without a subprocess. |
| **Process-level isolation** — slot pools, gateway restarts | `environment/` isolates a cell's *state*: its directory, its policy, its cutoff. Isolating concurrent *processes* is launch mechanics specific to one CLI, so it belongs with that adapter. The old repo's pool was the only thing actually preventing concurrent cross-talk, while the pid-based scheme it sat next to did nothing. **The pool did not fix sequential staleness**: a reused process kept the first cell it ever loaded, and `_require_bundle` never reloaded despite a docstring saying it did. When the transport lands it must carry the rule below. |

**Rule for whoever builds the tool transport:** a server process serves exactly
one cell and refuses a second. Its session directory path already encodes the
cell, so a reused process pointed at a new cell is detectable — it must fail
loudly and let the adapter restart it, never serve the cell it remembers. That is
the failure the slot pool masked rather than fixed.
| **Memory** as a readable kind | It's an agent capability, not market data, so it isn't in the catalog. The environment will expose it once an adapter defines what memory means for a strategy. |
| **Factor returns** (Ken French) | Used by the old repo for factor neutralisation, which is strategy-owned judgment rather than a served kind. It wants a home in the scoring/strategy layer, not the catalog. |
| **Symbol renames** (`META`/`FB` pre-2022-06-09) | The old code handled this with a SPLICE map in study scripts, so the runtime path silently served garbage for pre-rename `META`. Belongs in the data layer, once there's a rename table to key it on. |
| **Index weights** | The constituents dataset carries membership only. `UniverseReport.weights_available` is `False` and nothing pretends otherwise; a price/cap-weighted benchmark needs a different source. |
