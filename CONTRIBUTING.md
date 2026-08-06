# Contributing to fintel

Thanks for considering a contribution. fintel is a benchmark evaluation
platform for financial agents, and the boundary between **strategy package**,
**agent**, and **platform** is what keeps it general — please read
[`docs/architecture.md`](docs/architecture.md) §1 (the boundary) and §8
(invariants & layer map) before opening a PR. Most "what should I respect"
questions are answered there.

## Before you start

- Python ≥ 3.11, [uv](https://docs.astral.sh/uv/) for dependency management.
- `uv sync --all-extras` to install.
- Secrets: `mkdir -p .env && cp .env.example .env/keys.env`, fill keys.
  `.env/` is gitignored — never commit keys.

## The contract

The layer map (L0 `models/` → L9 `cli/`) is enforced by
`tests/test_architecture.py`. Imports flow one way only. The invariants in
[`docs/architecture.md`](docs/architecture.md) §8 each have a test. Two
load-bearing ones to keep in mind:

- **`agents/` cannot import `market/` or `pit/`.** An adapter that can fetch
  its own data measures the evidence channel, not the agent.
- **Only `environment/cell.py` may construct a `Cutoff`**, and only
  `market/realized.py` may read past the decision date. PIT has exactly one
  point of decision and one point of measurement.

A change that needs a platform edit to make a new strategy package run is a
boundary violation — the conformance test
(`tests/test_evaluate.py::test_dissimilar_package_runs_end_to_end`) catches
this.

## Development workflow

```bash
# Lint + format
uv run ruff check --fix .
uv run ruff format .

# Tests
uv run pytest tests/ -q
uv run pytest tests/test_architecture.py   # the layer map
uv run pytest tests/test_agents.py         # agent conformance
uv run pytest -k backfill                  # by keyword
```

Always run `ruff check --fix .` and `ruff format .` after code changes. CI
runs the same gates (`.github/workflows/ci.yml`).

### Code style

- Ruff (`E`, `F`, `I`, `B`, `UP`), line length 100, target `py311`.
- No `assert` for runtime guards — use explicit `if` + clear error; asserts
  vanish under `-O`.
- No cross-package `_private` imports.
- `models/` imports no logic; `pit/` never imports `simulate/`; `evaluate/`
  never imports `simulate/`.

## Common contributions

### Add or fix a strategy package

Strategy packages live under `packages/` and are external to the platform —
the platform loads, validates, and freezes them but never edits them. See
[`docs/add_strategy_package_guide.md`](docs/add_strategy_package_guide.md).
A package change should not require any `fintel/` edit.

### Add a data source

Either register in the catalog (`fintel/market/catalog.py`,
`register_source(SourceInfo(...))`) or ship as `module:Callable` in the
package. Custom sources must clamp on `cutoff.decision_date` — PIT is your
responsibility for custom sources.

### Add an agent adapter

1. `fintel/agents/installed/{name}.py` (in-process) or
   `fintel/agents/adapters/{name}.py` (subprocess CLI).
2. Extend the right base (`LLMAgent` / `BaseInstalledAgent` / `ScriptedAgent`).
3. Register in `fintel/agents/factory.py`.
4. Declare `pit_enforcement` (`access` or `cli_deny`).
5. `tests/test_agents.py` picks it up via the registry parametrisation — it
   must pass the conformance suite.

See [`docs/add_new_agents_guide.md`](docs/add_new_agents_guide.md).

### Add a KPI / signal / transform

Point `[scoring]` at `module:Callable`. Signatures match
`fintel/evaluate/signals.py`, `fintel/evaluate/kpi.py`,
`fintel/evaluate/transforms.py`. The platform never inspects the math.

## Opening a PR

- Keep the change focused — one concern per PR.
- Run `ruff check --fix .`, `ruff format .`, and `uv run pytest tests/`
  locally before pushing. CI runs the same gates.
- Reference the invariant or layer your change touches in the description.
- If the change is user-visible, add an entry under `## [Unreleased]` in
  [`CHANGELOG.md`](CHANGELOG.md).

## Reporting issues

- State the package, agent, and CLI command (with flags) you ran.
- Attach the `job.log` / `run.log` tail and the failing cell's
  `cells/<cell>.json` if you have one — they contain the outcome and health
  status without needing a re-run.
- Redact any keys / personal paths before pasting.
