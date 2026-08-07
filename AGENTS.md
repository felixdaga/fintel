# AGENTS.md

> **Breaking changes**: see `CHANGELOG.md` (when it lands) for migration notes.
> Until then, the invariants in `docs/architecture.md` §8 are the contract.

## Project overview

fintel is a benchmark evaluation platform for financial agents. A **strategy
package** is the benchmark (universe, schedule, data, mission, output contract,
signal, KPI). An **agent** is whatever you bring (in-process LLM desk, subprocess
CLI, scripted baseline). The platform runs the simulation under point-in-time
controls and scores the decisions post-run.

Read [`docs/architecture.md`](docs/architecture.md) first — especially §1
(the boundary and the financial-vs-terminal eval distinction) and §8
(invariants & layer map). The layer map is enforced by
`tests/test_architecture.py`; do not add an import that crosses it.

## Quick start

```bash
# Dev install
uv sync --all-extras

# Smoke run (2 names × 1 date)
fintel simulation packages/systematic_stockrate_djia_weekly \
  --agent optimized \
  --job-id smoke \
  --universe AAPL,MSFT \
  --dates 2026-04-24 \
  --shared-concurrency 4

# Score it
fintel report smoke

# Tests
uv run pytest tests/
```

Secrets: `mkdir -p .env && cp .env.example .env/keys.env`, then fill keys.
`.env/` is gitignored. Never commit keys.

## Repository structure

```
fintel/
├── fintel/                # the platform
│   ├── models/            # L0 — pure schemas, no I/O
│   ├── utils/             # L1 — generic helpers
│   ├── pit/               # L2 — the platform guarantee: clamp + audit
│   ├── market/            # L3 — catalog, universes, schedules, data sources, realized prices, probe, prefetch
│   ├── environment/       # L4 — session, access, mcp, evidence, nerve, echo, staging, cell (Cutoff owner)
│   ├── agents/            # L5 — adapters + agent-side context (barred from market/ and pit/)
│   ├── strategy/          # L6 — package load, lock, preflight
│   ├── simulate/          # L7 — job, run, trial, cell, queue, store, artifacts, backfill
│   ├── evaluate/          # L8 — read-only analytics: signals, kpi, holdings, behaviour, variance, report
│   └── cli/               # L9 — parse flags, build config, call simulate/evaluate, present, watch
├── packages/              # strategy packages (external, loaded not imported by platform)
├── docs/                  # architecture + guides
├── runs/                  # job output (gitignored; demo run negated in)
├── cache/                 # central data cache (gitignored)
└── tests/                 # pytest suite, flat
```

Imports flow one way only (L0 → L9). Enforced by `tests/test_architecture.py`.

## Key concepts

### Strategy package

An external directory the platform loads, validates, and freezes:

```
packages/my_strategy/
  strategy.toml          # manifest: universe, schedule, data bindings, scoring
  mission.md             # agent character / rubric
  output_schema.json     # View contract (symbol + score in [-1,1] minimum)
  strategy.lock          # written by load_and_prepare
  my_sources.py          # optional — bring-your-own DataSource
  scoring.py             # optional — custom signal / KPI callables
```

The platform never edits it. `load` parses; `preflight` reports every reason
it can't run; `build_lock` freezes identity. See
[`docs/add_strategy_package_guide.md`](docs/add_strategy_package_guide.md).

### Execution hierarchy

```
Job → Run → Trial → Cell
```

- **Job** — one `fintel simulation` invocation: package × agent × market × K
- **Run** — one of K repeats (the stochasticity unit)
- **Trial** — one decision date
- **Cell** — one agent invocation (per-symbol for `single_name`, whole universe for `portfolio`)

Cell is the load-bearing orchestration level: it owns the `Cutoff` (only
`environment/cell.py` may construct one), and it's where retry lives.

### Agents

`agents/` holds one adapter per agent behind an `AgentFactory` name map.
Contract: `decide(environment) -> AgentResponse` — a returned value, not a
callback. Two hosts:

- **in-process** (`llm`, `scripted`, `constant`, `optimized`) — consume
  `DataAccess` / tool objects directly.
- **subprocess CLI + MCP** (`openclaw`, `claude_code`) — fintel MCP server
  rebuilds `Environment` from the session dir.

Every adapter declares `pit_enforcement`: `access` (in-process, already
clamped) or `cli_deny` (subprocess, deny list + MCP isolation per cell).

### Evaluation

Read-only over finished jobs, never imports `simulate/`. The strategy owns
the signal (`signal_fn(views) → {symbol: float}`) and KPI
(`kpi_fn(signal_by_date, prices, horizons, params) → dict`); the platform owns
ensemble (cell-mean across K), holdings, returns, behaviour (L1), variance (L2),
report rendering.

## Development setup

```bash
git clone https://github.com/felixdaga/fintel.git
cd fintel
uv sync --all-extras
```

Python ≥ 3.11. Ruff for format + lint, pytest for tests.

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pytest tests/
```

Always run `ruff check --fix .` and `ruff format .` after code changes.

## Testing

Tests are flat in `tests/`, no unit/integration split. Most are hermetic
(no network, no real LLM calls); a few rely on fixtures or env vars.

```bash
# Full suite
uv run pytest tests/

# One file
uv run pytest tests/test_simulate.py

# By keyword
uv run pytest -k backfill
```

Key files to know:

- `tests/test_architecture.py` — the layer map + invariants. If your change
  adds a forbidden import, this fails first.
- `tests/test_agents.py` — agent conformance suite, parametrised over the
  registry. Adding an adapter without registering it fails here.
- `tests/test_evaluate.py` — the evaluation-layer conformance test (a
  dissimilar package runs end-to-end with zero platform edits).
- `tests/test_pit.py` — point-in-time guarantees.
- `tests/test_concurrency.py` — atomic writes + lock-merge (the lost-update
  guard).

## Code style

- **Ruff** — format + lint (`E`, `F`, `I`, `B`, `UP`), line length 100,
  target `py311`.
- **No cross-package `_private` imports** — invariant #4.
- **No `assert` for runtime guards** — use explicit `if` + clear error; asserts
  vanish under `-O`.
- **`models/` imports no logic** — pure schemas, no I/O.
- **`pit/` never imports `simulate/`**; `evaluate/` never imports `simulate/`.
- **Only `market/realized.py` may read past the decision date**, and nothing
  agent-facing may import it.
- **Only `environment/cell.py` may construct a `Cutoff`** — PIT has exactly
  one point of decision.

## Common tasks

### Add a data source

Either register in the catalog (`fintel/market/catalog.py`,
`register_source(SourceInfo(...))`) or ship as `module:Callable` in the
package. Custom sources must clamp on `cutoff.decision_date`. See
[`docs/add_strategy_package_guide.md`](docs/add_strategy_package_guide.md) §5.

### Add an agent adapter

1. Create `fintel/agents/installed/{name}.py` (in-process) or
   `fintel/agents/adapters/{name}.py` (subprocess CLI).
2. Extend the right base (`LLMAgent` / `BaseInstalledAgent` / `ScriptedAgent`).
3. Register in `fintel/agents/factory.py` name map.
4. Declare `pit_enforcement` (`access` or `cli_deny`).
5. `tests/test_agents.py` will pick it up via the registry parametrisation —
   it must pass the conformance suite.

See [`docs/add_new_agents_guide.md`](docs/add_new_agents_guide.md).

### Add a KPI / signal / transform

Point `[scoring]` at `module:Callable`. Signatures match
`fintel/evaluate/signals.py`, `fintel/evaluate/kpi.py`, `fintel/evaluate/transforms.py`.
The platform never inspects the math. Builtins cover single-name score → IR;
anything else lives in the package.

### Add a universe preset

`register_universe(UniverseInfo(name=..., target="module:Callable"))` in
`fintel/market/catalog.py`. The callable resolves membership at a decision
date (membership changes over time).

## Important notes

- The `fintel` CLI is the only operator surface — no `scripts/` one-offs.
- `runs/` and `cache/` are gitignored; the demo run `runs/systematic_stockrate_djia_weekly_optimized_20260806_mimo_v2_5_pro_k1_0001/`
  is negated in `.gitignore` so it ships.
- Secrets load from `.env/keys.env` unless `--no-bootstrap` is set.
- The MCP server lives in `environment/` (not `agents/`) because it rebuilds
  the environment — which requires `market.factory` — and `agents/` is barred
  from importing `market/`.
- The agent is built fresh per cell — `LLMAgent` accumulates trace on `self`,
  so shared instances would race across threads.
