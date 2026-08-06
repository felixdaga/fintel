# fintel — architecture

Agent evaluation where the benchmark is an investment outcome. You bring a
**strategy package**, you name an **agent**, the platform runs the evaluation to identify optimal configurations.

## 1. The boundary

**Platform** — the infrastucture and plumbing

- point-in-time guarantee: clamping, and the post-run trace audit
- isolation and reproducibility: `config.json` / `result.json` (fingerprint sealed in config)
- orchestration: Job → Run → Trial → Cell, retry, resume, K repeats
- data serving: cache, PIT clamp, kind → tool / evidence exposure
- agent adapters
- the evaluation layer: KPI, stochasticity, holdings, report — read-only over finished runs
- the generic K-repeat / ensemble scoring harness
- the generic renderer
- the nervous system: preflight, run echo, live staging, durable logs

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
Job          one `fintel simulation` invocation: package × agent × market × K
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
schema_version = 1
name = "my_strategy"
description = "Score each name independently; scored via pooled single-name IR."

[universe]
preset = "dow30"              # or symbols = ["AAPL", "MSFT", …]

[decision]
scope = "single_name"
schedule = { kind = "custom_dates", start = "2026-01-01", end = "2026-06-30", dates = [...] }
# also: kind = "quarterly" with start/end

[[data]]
kind = "prices"
source = "massive_prices"     # swap for any source serving `prices`

[[data]]
kind = "fundamentals"
source = "massive_fundamentals"

[[data]]
kind = "news"
source = "massive_news"

[scoring]
# The signal: how the decision's metrics become THE signal. The platform owns
# the mechanics around it (transform, ensemble, holdings); the package owns
# what the signal *is*. Builtin name or "mypkg.scoring:my_signal_fn".
signal = "single_name"
# The KPI: what "good" means over signal + prices. Builtin name or
# "mypkg.scoring:my_kpi_fn". The platform runs the mechanics around it; the
# package owns the math.
kpi = "single_name_ir"
metric_key = "icir"
transform = "single_name"        # a post-step on the signal; builtin or module:Callable
horizons = [1, 2, 4, 8]          # steps on the decision-date grid
# Strategy-owned params passed to the KPI + holdings. `holdings=true` opts into
# the default signal->weights->returns mechanics.
params = { holdings = true, active_budget = 0.5, cost_bps = 5.0 }
```

`extra="forbid"` on the manifest, so a typo is an error rather than a silently
ignored key. Unknown keys inside a `[[data]]` / `[universe]` / `[decision.schedule]`
block are kept as `params` and handed to whatever the factory resolves — that is
the extension seam, and it means a new source needs no schema change here.

## 4. The market library

Three things are separable, and keeping them separate is what makes providers
swappable:


|            |                         |                                          |
| ---------- | ----------------------- | ---------------------------------------- |
| **kind**   | what the agent asks for | `prices`                                 |
| **source** | who answers             | `massive_prices`, `synthetic_prices`     |
| **fields** | what comes back         | `open`, `high`, `low`, `close`, `volume` |


`market/catalog.py` is the library: every source registers its kind, provider,
field roster, accepted params and required env vars, so you can browse what
exists before picking. `market.catalog.sources(kind="prices")` answers "what
could serve this", and registering a `SourceInfo` adds to the library without
editing the platform. A source with no declared fields is refused — an
undocumented source can't be picked from a catalog.

The catalog is the whole library, not a subset — a kind is in it only once it's
fetchable under PIT control, because "declared but unwired" is the failure mode
where a strategy runs and quietly sees nothing:


| kind             | source                               | PIT clamp                                          |
| ---------------- | ------------------------------------ | -------------------------------------------------- |
| `prices`         | `massive_prices`, `synthetic_prices` | bar date `< decision_date`                         |
| `fundamentals`   | `massive_fundamentals`               | `filing_date < decision_date`                      |
| `news`           | `massive_news`, `alphavantage_news`  | `published_at < decision_date`                     |
| `filing_text`    | `massive_filing_text`                | `filing_date < decision_date`                      |
| `macro`          | `fred_macro`                         | observation date `< decision_date`                 |
| `ratios`         | `valuation_ratios` (computed)        | inherited from upstreams                           |
| `news_sentiment` | `news_sentiment` (computed)          | inherited from upstream                            |
| `web_search`     | `web_search`                         | provider freshness window ends `decision_date - 1`; optional post-clamp on Brave `sources[url].age` (`clamp_by_age`, default on) |


`web_search` PIT is two layers: the provider freshness window (cache-keyed so a
replay can't widen it), then — when catalog/`[[data]]` `clamp_by_age` is on —
a post-filter using Brave's `sources[url].age` YYYY-MM-DD against that window.
Undated results are kept. Strategies drive the knob like other source params.

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
ratios quietly change provider. Computed kinds derive on demand from
already-clamped upstream data, so there is a single PIT implementation.

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
importing it. The clamped and unclamped paths are different names in different
modules, not two methods one typo apart.

## 5. The environment

Where the three inputs meet, for exactly one agent invocation. The environment is
*built* from strategy + market + runtime, then *handed to* the agent adapter —
the third owner from §1 — which never reaches for data itself but chooses one of
the delivery channels and emits staging events back to the nerve.

|              | role | brings |
| ------------ | ---- | ------ |
| **strategy** | input | which kinds, with which params; what one cell decides on |
| **market**   | input | the universe *at the decision date*, and the bound clamped sources |
| **runtime**  | input | where the cell may write, and how much it may ask for |
| **agent**    | consumer | receives the `Environment`, picks a channel (`tools` / `pack` / in-process), runs `decide(env) -> AgentResponse`, and emits `agent_stage` / `agent_tool_call` / `agent_tool_result` to `env.nerve` |

The agent is not an input to building the environment — that's the point of the
boundary: the same environment can be handed to any agent, so a comparison
measures the agent, not the world it was given. The adapter's only structural
job is to consume what's offered and report back; it is barred from `market/`
and `pit/` (§13), so it cannot fetch its own data or build a `Cutoff`.

A `Cell` is the identity of one invocation — run, date, symbols, scope — and it
owns the `Cutoff`. That's the load-bearing decision in this layer. Cell identity
is constructed, frozen and passed explicitly, and a test forbids any module but
`cell.py` from building a `Cutoff`.

### How the modules come together

`factory.build_environment` is the one assembler. It takes the three inputs
(strategy kinds, market sources + universe, runtime session) and wires every
environment module into one `Environment` object an agent adapter consumes.
Two streams leave it: the **PIT audit stream** (`log`, an `AccessLog`, one record
per read) and the **live run stream** (`nerve`, the `Nerve`, staging events).

```mermaid
flowchart TD
    subgraph inputs["inputs (built outside, passed in)"]
        CELL["cell.py<br/>Cell — owns the Cutoff<br/>(only place a Cutoff is built)"]
        SRC["market DataSource(s)<br/>kind → fetch(query, cutoff)"]
        UNI["universe at the decision date"]
        NERVEIN["nerve (Progress)<br/>the run emit surface"]
    end

    FACT["factory.build_environment"]
    POL["policy.py<br/>AccessPolicy / PolicyBuilder<br/>what this cell may read & decide"]
    SESS["session.py<br/>SessionDir — where the cell writes<br/>holds the trace path"]
    LOG["trace.py<br/>AccessLog — the PIT audit stream<br/>one record per read (on_read hook)"]
    ACC["access.py<br/>DataAccess — the ONE agent-facing path<br/>policy-check → clamp → record"]
    TOOLS["tools.py<br/>ToolSurface — typed tools<br/>(generated from bindings + catalog)"]
    EVID["evidence.py<br/>evidence.build — rendered text pack"]
    ENV["base.py<br/>Environment — what one invocation may see"]

    CELL --> FACT
    SRC --> FACT
    UNI --> FACT
    NERVEIN --> FACT
    FACT --> POL
    FACT --> SESS
    FACT --> LOG
    POL --> ACC
    SRC --> ACC
    LOG --- "on_read" --> ACC
    ACC --> TOOLS
    ACC --> EVID
    FACT --> ENV
    CELL --> ENV
    ACC --> ENV
    POL --> ENV
    LOG --> ENV
    TOOLS --> ENV
    SESS --> ENV
    NERVEIN --> ENV

    subgraph channels["three delivery channels — one path, three presentations of DataAccess.read"]
        TOOLS
        EVID
        DIRECT["in-process<br/>access.read(kind, **q)"]
    end
    ACC --> DIRECT

    AGENT["agents/<br/>adapter — decide(env) -> AgentResponse<br/>picks a channel, never fetches itself"]
    ENV --> AGENT
    TOOLS --> AGENT
    EVID --> AGENT
    DIRECT --> AGENT
    AGENT -. "agent_stage / agent_tool_call / agent_tool_result" .-> NERVEIN

    subgraph subprocess["subprocess CLI agent path"]
        MCPS["mcp_server.py<br/>rebuilds Environment from session dir<br/>(cell.json + bindings.json)<br/>exposes ToolSurface tools, writes result.json"]
    end
    ENV -. "session dir + bindings" .-> MCPS
    MCPS -. "attaches reads to" .-> LOG
    MCPS -. "serves tools to" .-> AGENT

    subgraph observability["nervous system (§6)"]
        NERVE["nerve.py<br/>Nerve — JSONL log + live terminal"]
        STG["staging.py<br/>StageTracker — grid + stuck heuristics"]
        ECHO["echo.py<br/>run input snapshot"]
        PROG["progress.py<br/>Progress protocol + NullProgress"]
        HLTH["health.py<br/>post-run audit of access.jsonl"]
    end
    NERVEIN -.-> NERVE
    NERVE --> STG
    PROG -.-> NERVE
    ECHO -. "reads tool specs" .-> TOOLS
    LOG -. "audit input" .-> HLTH
```

The invariants the diagram encodes: `agents/` never reaches `market/` or `pit/`
(an adapter consumes `Environment`, it does not build a `DataSource` or a
`Cutoff`); `tools` and `evidence` share one `access` so they cannot drift apart;
and the audit stream (`log`) and the live stream (`nerve`) are two concerns with
two sinks but one owner — the environment module.

**Reads are policy-checked, then clamped, then recorded** — one path,
`DataAccess.read`. Every read returns a `Reading` whose status is `ok`,
`empty`, `failed` or `denied`. Absence and failure are different answers: a
missing API key, a source that raised, an unconfigured provider, and a company
with genuinely no news are four distinct statuses, distinguishable to the agent
and to a reviewer reading the trace afterwards. `AccessLog.summary()` reports
`degraded` so a run that half-failed is never scored as a run against a quiet
market.

**The universe is enforced on reads**, not only at submission. Widening it is a
strategy's explicit choice via `peers`, which grants reads without granting
decisions.

**The tool surface is generated** from the bindings and the catalog, so a run
that binds three kinds has three tools, each described by the entry of the
source actually bound. A param the strategy owns — the trailing window behind a
P/E — is marked `per_call=False` and never offered to an agent, since two
readings in one run must mean the same thing.

### Three delivery channels, one path

Agents differ in how they want data, not in what they may see. All three
channels are thin presentations of `DataAccess.read`:


| channel         | for                                                                                              | built from                             |
| --------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------- |
| in-process call | a desk that fetches directly in Python                                                           | `access.read(kind, **q)`               |
| typed tools     | anything doing function-calling — MCP, an LLM's native tools, or an in-process framework wrapper | `ToolSurface.descriptors()` + `call()` |
| rendered text   | a single-turn agent that needs everything in one dump                                            | `evidence.build(access)`               |


This is why `tools.py` carries no transport. A multi-role desk narrows its
per-role tool set with `ToolSurface.subset(kinds)`, which shares the underlying
access, so a role boundary can't become a second data path with its own rules.

Repeated identical reads inside one cell are memoized, so an agent re-reading
prices mid-reasoning cannot see two different answers. Failures are never
memoized — a transient blip must not become an outage for the rest of the cell.

### Nothing shared between concurrent cells

Every artifact a cell writes is named after that cell: `cells/<cell>.json` and
`trace/<cell>.jsonl`. `decision.json` still exists but is a *reduction*, written
once after the fan-in.

The shared cache needs two separate guarantees:

- **Atomic writes**, so a reader — which takes no lock — can never observe a
half-written file. Parquet is written to a unique temp file and replaced, not
written in place.
- **A lock around read-modify-write**, because atomicity alone doesn't prevent a
lost update: two cells fetching different spans of one symbol both read the
old coverage, both append their own, and the last writer erases the other's
records. Merging lives on the store, which re-reads inside the lock, rather
than in each source.

`tests/test_concurrency.py` covers both. The lost-update test was verified to
fail 25 times out of 25 with the lock removed, so it is a real guard and not a
test that merely passes.

## 6. The nervous system

The environment module owns run-time observability — the single emit surface
for every run event, durable to disk and live to the terminal. It exists so a
run is never blind: preflight catches a missing piece before any LLM call is
paid for, the echo records every input before any cell runs, and live staging
shows what each agent is doing while it does it.

### Nerve (`environment/nerve.py`)

`Nerve` implements the `Progress` protocol (`environment/progress.py`) and is the
one sink every layer emits through. Each event is written as a JSONL line to a
durable log and, when verbose, printed to the terminal. There are two scopes:

- **Job nerve** — `runs/<job>/job.log`. Owns job-level events: `job_start`,
`preflight_ok`, `probe_kind`, `prefetch_done`, `job_done`.
- **Run nerve** — `runs/<job>/rK/run.log`. Owns run-level events: `run_start`,
`run_echo`, `trial_start`, `cell_start`, the staging events, `cell_done`,
`run_done`.

`run_job` builds the job nerve; `run_run` builds each run nerve. Passing a single
`progress=` down is supported (one log for everything) but the default is
per-run nerves, which is what the multi-track dashboard needs.

`NullProgress` (`environment/progress.py`) is the no-op sink for tests and
`--quiet`.

### Echo (`environment/echo.py`)

Before any cell runs, `build_echo()` gathers every input that shapes the run
into one snapshot: agent + model, PIT mode and deny list, strategy name + digest,
universe, schedule/dates, bound data kinds, the generated tool surface with each
tool's param schema, the mission/schema/instruction sizes, and the fingerprint.
`render_echo()` turns it into the human-readable block printed at run start and
recorded on the `run_echo` nerve event in `run.log`. It mirrors agent prompt
construction exactly, so the echo is a faithful record of what the agent was
actually given. Prompt helpers are inlined here to keep `environment/` from
importing `agents/`. The fingerprint digest is also sealed into `rK/config.json`.

### Staging (`environment/staging.py`)

`StageTracker` accumulates per-cell state from the staging events
(`cell_start`, `agent_stage`, `agent_tool_call`, `agent_tool_result`,
`cell_done`) and derives two things:

- **A live grid** — one row per cell: `cell / date / stage / round / calls / err / idle / last tool or reasoning`. Rendered by `StageTracker.grid()`.
- **Stuck heuristics** — `stalled(threshold_s)` flags a cell whose last event is
older than the threshold; `extra_calls(threshold)` flags a cell making an
unusual number of tool calls.

The nerve owns one `StageTracker` and, on a throttle (`grid_interval_s`), emits a
derived `run_grid` event (the rendered grid) and `agent_stalled` events for
newly stalled cells. Derived events are emitted outside the nerve lock so a
listener can't deadlock the emitter. The staging events carry `decision_date`
so a multi-date run (same symbol, different dates) is disambiguated in the grid.

### Probe (`market/probe.py`)

A run-level reachability check for the bound data sources, run before any cell.
For each kind it calls `DataSource.fetch(query, cutoff)` with a synthetic query
using that kind's resolved lookback — the strategy's `[[data]].lookback_days`,
falling back to the catalog `Param.default` (so quarterly fundamentals reliably
return data, not an empty 30-day window) — and classifies the result as `ok`,
`empty`, or `failed`. `empty` is reachable — a quiet kind is not a broken kind.
The probe emits one `probe_kind` event per kind and a `probe_ok` summary. It
lives in `market/` (not `environment/`) because it calls the market primitive
directly, keeping the layer ladder intact.

### Prefetch (`market/prefetch.py`)

Warms the cache for an entire run before any cell runs, so a cell never pays the
latency of a cold fetch mid-reasoning. Two design points:

- **One lookback per kind.** The strategy's `[[data]].lookback_days` (catalog
default when omitted) is the single value every caller uses — prefetch, probe,
the tool schema default, the access cap, and evidence packs all read the same
number, baked into each source instance by the factory. There is no separate
`spec.lookback_days`, `EvidenceConfig.*_lookback_days`, or `max_lookback_days`
knob to drift out of agreement. A caller may request *less* history, never more
than the strategy declared; the access policy clamps to that cap.
- **Chunked concurrent fetch for high-volume kinds.** `MassiveRecords` (news)
splits a span into `chunk_days`-day chunks and fetches them concurrently with
`chunk_workers` at `limit=1000`, instead of paginating sequentially at
`limit=100`. A span that hung for minutes now completes in ~1.4s.

`prefetch` reports a warmed/failed count and writes `runs/<job>/prefetch.json`.

## 7. Layers

Imports flow one way only. Enforced by `tests/test_architecture.py`.

```
L0   models/       pure schemas. no I/O.
L1   utils/        generic. no domain knowledge.
L2   pit/          the platform guarantee: clamp + audit
L3   market/       calendar, universes, schedules, data catalog, realized prices, probe, prefetch
L4   environment/  session, access, mcp, evidence, isolation, nerve, echo, staging, progress
L5   agents/       adapters + agent-side context
L6   strategy/     package load, lock, preflight
L7   simulate/    job, run, trial, cell, queue, store, artifacts — the simulation, up to decisions
L8   evaluate/    read-only analytics over finished runs: signal, KPI, stochasticity, holdings, report
L9   cli/         parse flags, build config, call simulate/evaluate, present results, render dashboard
```

`evaluate/` (scoring) sits above `simulate/`, reads its artifacts, and imports it
never — so a re-score needs no re-run. The `evaluate → simulate` forbidden edge is
enforced in `tests/test_architecture.py`, and a dedicated conformance test
(`tests/test_evaluate.py`) asserts no `evaluate/` module imports `simulate/`.

## 8. Extension points

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

`agents/` holds one adapter per agent, the way Harbor's `agents/adapters/`
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
and some have no CLI at all — an in-process desk needs no install step and no
argv. So `BaseInstalledAgent` maps to one family of ours, not to all of them,
and it belongs in `agents/adapters/` rather than at the root.

Worth taking from Harbor:

- **Declarative `CLI_FLAGS` / `ENV_VARS`** descriptors that build argv and env
from kwargs, instead of a hand-rolled config class per adapter.
- `**ERROR_PATTERNS` mapping output to typed errors.** The important one. It is
the same empty-vs-failed distinction `DataAccess` already makes, applied to
agents: a rate limit is retryable, a safety refusal is a real abstention and
must *not* be retried, and a context-window overflow is a config bug, not a
flake.
- `**SUPPORTS_*` class flags**, checked before launch so a mismatch fails fast.
This is what "capabilities" honestly is — a property the adapter declares, not
something a strategy requests.
- `**install()` / `setup()` split from `run()**`, so a CLI is installed once per
trial rather than per cell.

Deliberately *not* in the adapter: per-symbol fan-out. Cells are the platform's
unit, so `simulate/` fans out and an adapter only ever sees one cell.

### Two hosts

Agents vary on three axes — *where the code runs*, *how data arrives*, *how the
answer returns* — but only two hosts matter:

- **In-process** — the desks and baselines. These consume `DataAccess` and tool
objects directly.
- **Subprocess CLI + MCP config** — OpenClaw and Claude Code. Same host; they
differ only in config file format and whether the instruction arrives by
argv or stdin.

Scope lives on the `Cell`, so one adapter serves both single-name and portfolio.

### What makes agents comparable

Running one strategy against several agents is the product, so the axis that
must not vary is the environment. Four things enforce that:

1. `**agents/` cannot import `market/` or `pit/`.** Guarded in
  `tests/test_architecture.py`. An adapter able to build a `DataSource` or a
   `Cutoff` can fetch its own data, and then a comparison measures the evidence
   channel instead of the agent.
2. **Typed outcomes.** `Outcome` separates `abstained` and `refused` — real
  answers — from `timeout`, `rate_limited`, `parse_error` and `crashed`. Only
   `RETRYABLE` outcomes are retried, because re-rolling a refusal does not
   recover an answer, it manufactures a different one. `AgentResponse` refuses to
   be `ok` with no views, so the ambiguity is unrepresentable rather than merely
   discouraged.
3. **A cost basis.** `Usage.basis` records whether a price was `reported` by the
  provider or `estimated` from a rate card, and a sum of both degrades to
   `mixed` and stops being `comparable`. A subprocess CLI usually gives tokens
   but no price while an HTTP agent gives the real charge, so this is the normal
   case, not an edge one.
4. **The channel is a platform knob.** Because tools and pack are two renderings
  of one `DataAccess`, the same adapter can be run over either. Switching it is
   a measurement, not an internal tweak.

`agents/run.py:invoke` is the only place an agent is called and the only place an
exception becomes an outcome. It never raises, because an adapter exception
should never abort the whole date and erase every other symbol's work.

`ScriptedAgent` can reach every outcome and every channel on demand, so error
handling is exercised on every commit rather than only when a provider happens
to rate-limit. `ConstantAgent` is a genuine baseline, not a stub: an agent that
cannot beat a fixed score has shown nothing.

`tests/test_agents.py` holds the conformance suite. It is parametrised over the
registry and asserts that the registry and the suite's roster are the same set,
so an adapter cannot be added without being held to the contract.

### The in-process host

`LLMAgent` is one adapter covering both the single-call dossier agent and the
tool-loop analyst. `channel` picks `pack` or `tools` and nothing else changes,
which is what makes switching it a measurement.

- `agents/emit.py` is the only parser from model output to `View`. Scores are
clamped rather than rejected, since a model emitting 1.4 means "very positive"
and the adjustment is recorded. The schema gives the model an explicit
`abstain`, because without one "no views" means both "no opinion" and
"something broke".
- `agents/llm.py` names provider failures. A rate limit, a policy refusal and a
prompt that didn't fit are three events. Some arrive as a *successful* response
— `finish_reason: content_filter` is a refusal, `length` is a truncation — so
the body is inspected too.

Cost is `reported` only when the provider states it. There is no rate card in the
tree, so `basis` stays `unknown` rather than quietly becoming an estimate that
later gets summed with a measurement. Adding an estimator is a labelled change.

### The subprocess host

`agents/adapters/` holds the CLI agents — OpenClaw and Claude Code. One host
underneath (`SubprocessAgent`), one transport (the fintel MCP server in
`environment/mcp_server.py`), two thin adapters that differ only in argv and
config-file format.

The MCP server lives in `environment/` rather than `agents/` because it rebuilds
the environment from the session directory — which requires `market.factory` —
and `agents/` is barred from importing `market/`. The server reads `cell.json`
and `bindings.json` from its session dir, rebuilds the `Environment`, and exposes
exactly the tools the strategy declared. It writes the agent's answer to
`result.json` and exits.

**One cell per process is structural, not policed.** The MCP server is a stdio
subprocess that dies when the CLI exits, so a reused gateway cannot keep serving
the first cell it ever loaded. The agent CLI is launched in its own session
(`start_new_session=True`) so a crash inside it cannot signal-kill the fintel
parent — the failure that once made parallel jobs die silently right after
`cell_start`.

**Per-cell profile isolation.** Each cell forks a minimal copy of the operator's
profile (`openclaw.json` + the agent config) to a unique temporary profile, sets
a freshly assigned free gateway port, and launches against it. This gives every
cell its own fresh gateway and fintel MCP server, so concurrent cells never share
a gateway that could serve a stale session dir. The isolated profile is removed
when the cell ends. Atomic profile writes use a PID + UUID temp filename so
concurrent modifications never collide on a shared `.tmp`.

Error patterns map CLI output to typed outcomes, the same way `llm.py` maps
provider responses. A rate limit, a safety refusal and a context overflow are
three different events.

### Transcript catcher (`agents/adapters/catcher.py`)

For a CLI agent the model's reasoning turns are not in-process; the catcher
tails the agent's transcript file and emits staging events to the nerve:
`agent_stage` (per reasoning round, including `thinking` content),
`agent_tool_call`, and `agent_tool_result`. It also appends to the cell's
`access.jsonl` audit. The catcher is constructed after the session id is known,
so it watches the right transcript, and it carries `decision_date` so multi-date
runs disambiguate.

### Fingerprint

`agents/fingerprint.py` covers every adapter uniformly: agent name and version,
model pin, channel, prompt hash, data kinds, and adapter parameters. The
channel is part of the digest, so a channel ablation is a different run, not the
same run that happened to use a different path.

`run_run` seals one into `r1/config.json` (`fingerprint` field) before any trial
runs, built from the declared `RunConfig` (not a live agent instance, so
fingerprinting has none of the side effects — HTTP client setup, profile reads —
that building the real adapter does) plus the mission text's hash. Two runs of
the same package produce the same digest; a changed mission, model, or channel
changes it.

### Mission, tools manual, and output schema

`agents/prompts.py` is the one composer every tool-calling adapter uses to turn
a strategy pack's `mission.md` and `output_schema.json` into the text an agent
actually sees. `run_job` reads both files once and threads them down unchanged —
`run_run` → `run_trial` → `run_cell` → `build_agent` — so every builtin agent
accepts `mission_text`/`output_schema_text` uniformly. A custom `module:Class`
adapter that doesn't declare them is built without them rather than failing.

`render_tools()` builds the manual from `ToolSurface.descriptors()`, the same
catalog-derived source the tools themselves are built from, so it cannot
describe a tool that doesn't exist or omit one that does — and it is offered to
*every* tool-calling delivery (OpenClaw, Claude Code, the LLM host's `tools`
channel), never to a non-tool-calling one (the LLM host's `pack` channel, which
has nothing to call).

For a CLI agent the whole thing collapses into one string — `mission` +
`## Tools` + `## Output schema` — because a subprocess argv/stdin instruction
has no system/user split. `LLMAgent` keeps that split: the mission replaces the
generic system-prompt fallback, and the tools manual / output schema land in the
user message alongside the native `tools=` function-calling schemas that
actually govern what it can call.

### Agent preflight

`run_job` checks two things before any cell runs, not one: the strategy
(`strategy.preflight`) and the agent (`agents.factory.preflight`). An adapter
declares a `preflight_checks(**params)` hook if it has something to check
without building or invoking itself — `LLMAgent` checks `OPENROUTER_API_KEY`
(skipped if a client was already supplied), `SubprocessAgent` checks its binary
is on `PATH`, `OpenClawAgent` additionally checks its profile config exists. An
adapter with no hook (`scripted`, `constant`) is assumed ready. Both checks raise
the same `PreflightError` on failure, so "can this run at all" is answered once,
before any LLM call is paid for.

### PIT tool policy (standard adapter requirement)

Every adapter must declare `pit_enforcement`:

- `**access**` — in-process hosts (`llm`, `scripted`, `constant`). No native
tool surface; every read goes through `Environment.access` (already
PIT-clamped). Declaring this is enough.
- `**cli_deny**` — subprocess CLIs (`openclaw`, `claude-code`). At every cell
launch the adapter must (1) apply the platform deny list for *threat*
channels only — uncontrolled web, free filesystem, shell egress, non-PIT
memory — and (2) isolate MCP to the `fintel` server, stashing the operator's
other servers and restoring them when the cell ends. Sub-agents, plan tools,
and session status stay on: those are ability, not knowledge egress.

The deny lists live in `agents/pit_policy.py` (`OPENCLAW_DENY`,
`CLAUDE_CODE_DENY`). OpenClaw merges them into `tools.deny` on the profile;
Claude Code passes them as session-scoped `--disallowedTools` so the operator's
daily Claude is not permanently crippled. Preflight fails closed if a `cli_deny`
adapter does not override `enforce_pit_policy`. The effective mode + deny list
is part of the run fingerprint.

## 9. Strategy packages

A strategy package is an external directory — `strategy.toml`, a `mission.md`,
and an `output_schema.json`. The platform never edits it; it loads, validates,
and freezes it. Three composable steps, one normal entry point:

- `load` parses the manifest. No I/O beyond the file, so `--list-strategy`
doesn't pay for validation.
- `preflight` reports *every* reason the package cannot run — bad bindings,
missing env, missing mission, unknown schedule — before any LLM call. One
command, all findings, not the first failure. It checks the declared world
is resolvable; it does not fetch, because fetching is the cost preflight
exists to avoid.
- `build_lock` freezes the package's identity: a digest of the manifest text,
the mission, the output schema, and the catalog state it was checked against.

`load_and_prepare` composes all three and writes `strategy.lock` into the
package root. A package may ship its own data sources as `module:Callable`;
loading does not import them, so a package referencing a source whose module
isn't installed still loads — it only fails at preflight or build, where the
error is actionable.

The scoring KPI is format-checked only (non-empty, or `module:Class`). Builtin
KPI validation lands with the scoring layer; preflight does not import scoring,
which would make it non-hermetic.

## 10. Artifacts

Every level writes the same pair: `config.json` (asked + fingerprint digest)
and `result.json` (happened). The run echo is emitted to the nerve / `run.log`
at start (not a sibling JSON file). Package-level `strategy.lock` is separate
from the run folder.

```
runs/<job_id>/
  config.json  result.json  job.log  prefetch.json  health.json
  r1/ … rK/
    config.json  result.json  run.log
    trials/<YYYY-MM-DD>/
      decision.json          reduced once, after the fan-in
      result.json
      cells/<cell>.json      one writer each  (CellRecord)
      trace/<cell>.jsonl     one writer each
    sessions/…               agent workspace (access.jsonl, dossier, …)
  report/                   evaluation output (fintel report)
    report.json  report.md
  cache/                     shared central cache (one root, atomic, lock-merged)
```

`r1` exists even when K=1 — predictable tooling beats a special case.
`RunConfig` records the effective world post-override (including `fingerprint`)
and is self-describing.

**One central cache root.** `runs/cache/` is the single shared cache (default
`<output-root>/cache`, overridable with `--cache-root` or `FINTEL_CACHE`).
Every kind lives under it as `<cache_root>/<kind>/<symbol>.{parquet,json}`.
`fintel cache status` reads the gap-aware coverage sidecars back out — per
symbol, coverage is a list of `[from, through]` intervals, not a lone min/max,
so a cache warmed through 2026 with a hole in 2024 honestly reports the gap.

### Semantic writers (`simulate/artifacts.py`)

One typed writer per artifact kind — `write_cell`, `write_decision`,
`write_trial_result`, `write_run_result`, `write_job_result`, `write_run_config`,
`write_prefetch`, `write_health`. Each wraps `store.write_json`/`store.write_model`
with knowledge of the specific model and path, so a `CellRecord` (with its typed
`SourceRef`s) round-trips correctly and callers never touch raw JSON. This is the
only place that knows which model goes in which file.

### CLI presenter (`cli/present.py`)

The reader side of the artifacts: typed loaders (`load_job_result`,
`load_run_result`, `load_cell_record`, `load_decision`, `load_health`) that
encapsulate `store.read_json` + `model_validate`, with graceful degradation for
corrupt files, and formatters (`job_summary_line`, `print_job_artifacts`,
`print_decision_block`, `print_json_block`) for the CLI. `fintel runs` and
`fintel health` read finished runs through this layer, never inline `json.loads`.

## 11. The fan-out

`simulate/` owns the hierarchy: **Job → Run → Trial → Cell**. Each level builds
the next level's config, fans out with bounded concurrency (`map_parallel`), and
reduces the results into one record written *after* its children are done. The
simulation covers everything up to the agent's decision; scoring those decisions
is the evaluation layer, built separately and read-only over the artifacts this
produces.

- A **Job** loads and preflights the package once, freezes a `RunConfig` per
repeat, and fans the K runs out with `max_concurrent` as the bound.
- A **Run** builds the universe, schedule, and data sources once (the cache is
the expensive part — a source that fetched a symbol's prices for one date
must not re-fetch for the next), then walks the schedule.
- A **Trial** resolves the universe *at the decision date* (an index's
membership changes), fans out cells, and reduces them into `decision.json` —
the single writer, after the fan-in.
- A **Cell** builds its environment, invokes the agent through `agents.invoke`
(which never raises — a failed cell is a recorded outcome), and writes its
own `cells/<cell>.json`.

A failure at any level is a recorded outcome, not a lost run. `map_parallel`
returns `None` for a failed item rather than raising, so one bad cell doesn't
abort its date, and one bad run doesn't abort the job. The reducers (`record.py`)
are pure — they read results, not the filesystem — so a re-reduction needs no
re-run.

The agent is built fresh per cell. For a scripted agent this is free; for an LLM
agent it constructs an HTTP client, which is cheap next to the call itself. The
reason is correctness: `LLMAgent` accumulates trace steps on `self`, so two
threads sharing one instance would corrupt each other's trace. Building per cell
makes the agent re-entrant by construction, so `cell_concurrency > 1` is safe.

### Two axes of concurrency

Runtime concurrency is two independent nested axes, plus an optional flat pool:

- **Cells within a trial** (`cell_concurrency`) — tickers decide concurrently.
Auto = the universe size at each date, so all tickers run at once. Safe
regardless of memory: memory writes happen after the whole trial completes,
never per-cell, so intra-date concurrency never races on shared state.
- **Runs across K repeats** (`max_concurrent`) — repeats run in parallel. Auto
= K, so all repeats run at once.
- **Flat shared pool** (`shared_concurrency`) — keep N cells in flight across
*all* (date, ticker) pairs in a run. A finished cell immediately takes the next
work item from any remaining date. Replaces the nested cell × trial fan-out.
Blocked when memory or feedback couples dates (independent cells only).

Peak in-flight sessions without shared = resolved `max_concurrent` × resolved
`cell_concurrency` (e.g. 3 × 10 = 30). With shared = `max_concurrent` ×
`shared_concurrency`.

### The sequential requirement

**Trials (dates) within a run are sequential by default**
(`trial_concurrency=1`), because a date's session carries the prior date's
portfolio + memory. The `memory_on` guard in `run_run` forces
`trial_concurrency` back to 1 when memory is on, so a misconfigured job that
asked for parallel dates can't race on shared state. Memory isn't wired yet, so
the guard is dormant; it activates the moment a strategy declares memory.
`shared_concurrency` is harder: it raises if memory or feedback is on.

> **Concurrency vs parallelism.** The runtime uses threads, which give
> *concurrency* for the I/O-bound work that dominates a backtest (LLM HTTP
> calls, subprocess waits — the GIL is released, so many are in flight at
> once). For subprocess agents this is also real *parallelism*, since each
> in-flight cell is a separate OS process. It is **not** CPU parallelism: the
> GIL serialises Python, so CPU-bound work (scoring, returns) gets no speedup
> from threads. That belongs to the scoring layer, which will use processes.

## 12. The evaluation layer

Read-only analytics over a finished job, run after `simulate/` and never
importing it. The strategy defines **what the signal is** and **what good
means**; the platform owns the mechanics around them. This is the §1 boundary
applied to scoring: a new strategy needs only its package, never an
`evaluate/` edit — enforced by a conformance test that runs a dissimilar
package (portfolio scope, custom `module:Callable` KPI) through the full
pipeline with zero platform changes.

### The abstraction

The decision carries quantitative metrics (the `View`: `score`, `conviction`,
…). The strategy owns two callables resolved from `ScoringSpec`; the platform
owns everything that wraps them:

| strategy defines | signature | owns |
|---|---|---|
| **the signal** | `signal_fn(views) -> {symbol: float}` | how decision metrics become THE signal |
| **the KPI** | `kpi_fn(signal_by_date, prices, horizons, params) -> dict` | what "good" means over signal + prices |

| platform computes | from | owns |
|---|---|---|
| ensemble signal | per-run signals | cell-mean across K (mechanical) |
| holdings | signal | signal → weights (default rule, opt-in) |
| returns | holdings + prices | gross + net (cost-adjusted) compounding |
| behaviour (L1) | traces | always on, no-op if no traces |
| output variance (L2) | signals | always on |
| report | all of the above | rendering only |

The platform never inspects what's inside `signal_fn` or `kpi_fn`. Builtins
cover the common cases with no code; `module:Callable` covers anything else
with no platform change — the same seam data sources, universes, schedules
and agents already use.

### Where the computation lives

- **Builtins** (platform-owned): `evaluate/signals.py` (`single_name_signal`),
  `evaluate/transforms.py` (`identity`, `rank_range`, `zscore`),
  `evaluate/kpi.py` (`single_name_ir` — raw IC), `evaluate/holdings.py`
  (weights + gross/net NAV + cost), `evaluate/behaviour.py` +
  `evaluate/variance.py` (stochasticity).
- **Strategy-owned custom math**: in the package, wherever the author puts
  it, referenced from `strategy.toml` as `module:Callable`. The manifest's
  `[scoring]` block is wiring only — it names which callable runs, it doesn't
  contain the math.

### Pipeline

```
read(job_dir) -> RunData{views, behaviour} per repeat
  -> signals: signal_fn(views) -> transform -> ensemble (cell-mean)   # platform mechanics
  -> behaviour(RunData)                       # L1, no-op if no traces
  -> variance(per_run signals)                # L2
  -> kpi.compute(ensemble + per_run, prices) # strategy-selected metric
  -> holdings.build(signals, prices) if opted-in  # default weights + NAV
  -> ReportPayload -> report.json + report.md
```

`read.py` is the one fintel-specific seam (it knows the on-disk layout);
everything above it is portable math. `prices.py` builds the unclamped
`PriceLookup` from the job's cache — the one module allowed to read past the
decision date (§4), now consumed by the KPI.

### Layers (always-on vs strategy-dependent)

| layer | status | notes |
|---|---|---|
| **L1 Behaviour** | always on | no-op for agents with no tool-call trace (scripted / constant) — reports `n/a`, not zeros that would read as "perfectly stable" |
| **L2 Output variance** | always on | score / sign / rank dispersion across K repeats |
| **KPI** | strategy-dependent (`scoring.kpi`) | `single_name_ir` (raw IC) for the current package; any `module:Callable` |
| **Holdings** | opt-in (`scoring.params["holdings"]`) | default long-only tilt around equal-weight + gross/net NAV |

There is no world-validity gate and no full MVO / factor attribution in the MVP
— those are post-MVP, along with `fintel compare` (paired ΔIC across two
jobs with the same strategy/period/universe).

### Conformance

`tests/test_evaluate.py::test_dissimilar_package_runs_end_to_end` is the §1
conformance test for the evaluation layer: a second, dissimilar strategy
(portfolio scope, `rank_range` transform, a custom `module:Callable` KPI, no
`single_name` anywhere) runs through the full pipeline with zero edits to
`fintel/evaluate/`. The only contract the package must satisfy is the
decision shape — `decision.json` keyed by symbol → `View` (score in [-1, 1]).

## 13. CLI

One entry point (`fintel`). Operator guides live under `docs/`:

| Guide | Covers |
| ----- | ------ |
| [`add_strategy_package_guide.md`](add_strategy_package_guide.md) | Author a package; bind catalog data or bring your own source |
| [`run_and_monitor_guide.md`](run_and_monitor_guide.md) | `simulation`, TUI watch, backfill, health, cache |
| [`evaluate_results_guide.md`](evaluate_results_guide.md) | Artifacts, `report`, KPIs, holdings, deeper notebook analytics |
| [`add_new_agents_guide.md`](add_new_agents_guide.md) | Wire a new agent adapter |
| [`data_pipeline.md`](data_pipeline.md) | Cache, lookbacks, prefetch → tools → PIT |

```
fintel simulation <package> --agent <name> [--k N] [--max-concurrent N]
                                          [--universe AAPL,MSFT] [--dates 2025-01-02]
                                          [--agent-opt k=v] [--cell-concurrency N]
                                          [--trial-concurrency N] [--shared-concurrency N]
                                          [--job-id …] [--watch-mode auto|alt|stream]
                                          [--no-prefetch] [--prefetch-workers 8]
                                          [--cache-root …] [--offline] [--quiet] [--no-watch]
                                          [--no-bootstrap]
fintel backfill <job_id> [--run K] [--cell-concurrency N]
fintel runs     list | show <job_id> | watch <job_id…>
fintel health   <job_id|session_dir>
fintel report   <job_id> [--cache-root …]
fintel cache    status [--source …] [--symbol …] [--window FROM..TO]
```

`fintel simulation` is the one entry point for a run. In a real terminal (TTY)
it launches the job in a background thread and renders the live in-place
dashboard in the foreground. With `--no-watch`, or in a non-TTY/CI context, it
runs synchronously with verbose nerve lines to stderr and no dashboard.

`fintel backfill` reruns only failed cells on a finished repeat and rewrites
cell/trial/run/job artifacts in place.

`fintel report <job_id>` runs the evaluation layer over a finished job: it
reads the strategy's `ScoringSpec` from the frozen run config (`rK/config.json`),
runs the full pipeline (§12), and writes `report.json` + `report.md` under
`<job>/report/`, printing the markdown to stdout. The strategy declares the
KPI/signal/transform/horizons/params; the platform runs the mechanics.

`fintel cache status` is read-only inspection of the central cache (gap-aware
coverage sidecars). See [`data_pipeline.md`](data_pipeline.md).

Agent-specific settings go through `--agent-opt`, validated against a schema the
agent adapter declares. Adding an agent requires no CLI edit. Secrets load from
`.env/keys.env` unless `--no-bootstrap` is set.

### The live dashboard (`cli/watch.py`)

`fintel runs watch <job_id…>` (and the `simulation` foreground) tails one or more
`run.log` / `backfill.log` files and renders a single updating screen: a shared
`preflight` header drained from `job.log` (probe[kind:status]), then **one track
per repeat** (r1…rK), each showing its cells with
`date / stage / round / calls / err / idle / last tool or reasoning`. Cells are
keyed by `(cell, decision_date)` so a multi-date run with the same symbol doesn't
collide. Modes: `alt` (full-screen TTY), `stream` (in-place, Cursor-friendly),
`auto` (picks). Press `q` to detach (the job keeps running; re-attach with
`fintel runs watch`).

### Health vs outcome

Agent **outcome** (`ok` / `abstained` / …) is what the model did. Environment
**health** (`ok` / `degraded` / `broken`) is whether the harness and data path
worked: failed reads, schema denials (`requires ['symbol']; got ['kwargs']`),
zero successful tool reads for a CLI agent, empty-only data, and PIT-suspect
query fields. A job with stuffed views but dead tools is `health=broken` and
must not report as a clean run. The MCP server **attaches** to the cell's
`access.jsonl` so subprocess tool calls are in the same audit trail.

## 14. Invariants

Each has a test in `tests/test_architecture.py`. Convention alone is what lets a
codebase drift.

1. No data at or after `decision_date` reaches an agent.
2. The platform never edits the agent's character — mission is passed in.
3. `models/` imports no logic.
4. No cross-package `_private` imports.
5. `pit/` never imports `simulate/`. (The `evaluate/` → `simulate/` edge is
  guarded too, for when scoring lands.)
6. No scoring change merges without the ICIR reference test passing.
7. One naming scheme per identity, in `models/ids.py`.
8. Every `module:Callable` a factory or the catalog advertises resolves.
9. Only `market/realized.py` may read past the decision date, and nothing
  agent-facing may import it.
10. Every registered data source declares its fields.
11. Only `environment/cell.py` may construct a `Cutoff`. Everywhere else it
  arrives from the cell, so PIT has exactly one point of decision.

