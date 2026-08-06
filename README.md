# fintel

Agent evaluation where the benchmark is an investment outcome.

You ship a **strategy package** (universe, schedule, data, mission, scoring),
name an **agent**, and fintel runs the simulation under point-in-time data
controls — then scores decisions with the package’s KPI.

## Install

```bash
uv sync
# or: pip install -e ".[dev,llm,mcp]"
```

Python ≥ 3.11. Copy secrets:

```bash
mkdir -p .env && cp .env.example .env/keys.env
# fill OPENROUTER_API_KEY, MASSIVE_API_KEY, BRAVE_API_KEY, FRED_API_KEY, …
```

`.env/` is gitignored. Never commit keys.

## Quick start

```bash
fintel simulation packages/systematic_stockrate_djia_weekly \
  --agent optimized \
  --job-id smoke \
  --universe AAPL,MSFT \
  --dates 2026-04-24 \
  --shared-concurrency 4

fintel report smoke
```

## CLI

| Command | Purpose |
| ------- | ------- |
| `fintel simulation <package> --agent …` | Run a job (live TUI by default) |
| `fintel runs watch <job_id>` | Attach / re-attach the dashboard |
| `fintel backfill <job_id>` | Rerun failed cells |
| `fintel health <job_id>` | Audit tool/data path |
| `fintel report <job_id>` | KPI + stochasticity + holdings |
| `fintel cache status` | Inspect the central cache |

## Docs

| Doc | Contents |
| --- | -------- |
| [`docs/architecture.md`](docs/architecture.md) | System design and invariants |
| [`docs/add_strategy_package_guide.md`](docs/add_strategy_package_guide.md) | Author a package; bring your own data |
| [`docs/run_and_monitor_guide.md`](docs/run_and_monitor_guide.md) | Run, TUI, backfill, cache |
| [`docs/evaluate_results_guide.md`](docs/evaluate_results_guide.md) | Artifacts, report, what we offer |
| [`docs/add_new_agents_guide.md`](docs/add_new_agents_guide.md) | Wire a new agent |
| [`docs/data_pipeline.md`](docs/data_pipeline.md) | Cache / lookback / PIT path |

## Example packages

- `packages/systematic_stockrate_djia_weekly` — weekly DJIA single-name rating
- `packages/systematic_stockrate_djia_monthly` — same stack, quarter-start grid

## Repository hygiene

- Secrets: only `.env.example` is tracked; real keys live in `.env/keys.env`
- Artifacts: `runs/` and `cache/` are gitignored
- Operator surface: the `fintel` CLI (no `scripts/` one-offs)
