# fintel

[![CI](https://github.com/felixdaga/fintel/actions/workflows/ci.yml/badge.svg)](https://github.com/felixdaga/fintel/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/felixdaga/fintel/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

A benchmark evaluation platform for financial agents. You bring a **strategy
package** (the benchmark), name an **agent**, and fintel runs the simulation
under point-in-time data controls — then scores the decisions with the
package's KPI.

## What it does

- **Evaluate arbitrary agents** — `optimized` (in-process LLM desk), `openclaw`,
`claude-code` (subprocess CLIs), or any `module:Class` adapter you ship.
- **Bring your own benchmark** — a strategy package is the investment strategy:
universe, schedule, data surface, mission, output contract, signal, KPI,
horizons. The platform never inspects the math.
- **Run controlled backtests** — point-in-time data clamp, per-cell isolation,
K-repeat stochasticity, bounded concurrency across cells / dates / repeats.
- **Score post-run** — strategy-dependent metrics, e.g. cross-sectional IC / ICIR, holdings + gross/net NAV,  
behaviour + output-variance layers, rendered to `report.md`.



### Financial agent eval vs terminal agent eval

Terminal benchmarks (SWE-Bench, Terminal-Bench, Aider Polyglot) measure whether
an agent *completes a task* in a container. Financial agent eval is different:


|                         | Terminal benchmarks               | Financial agent eval (fintel)                            |
| ----------------------- | --------------------------------- | -------------------------------------------------------- |
| **Task**                | Complete a terminal task          | Make a point-in-time investment decision                 |
| **Environment**         | Container / shell                 | PIT-controlled data with real market complexity          |
| **Success**             | Runtime — task completes = done   | Post-run — cross-sectional IC, final NAV                 |
| **Cell independence**   | Each task independent             | Cell outcome depends on strategy + cross-sectional cells |
| **Benchmark ownership** | Developer-designed, generalizable | Personal — your strategy, your benchmark                 |


Investors leveraging AI to make financial decisions need to own their own
benchmark. Hence the architecture: the strategy package **is** the benchmark.
See `[docs/architecture.md](docs/architecture.md)` §1 for the full framing.

## Install

External (from git):

```bash
uv pip install git+https://github.com/felixdaga/fintel.git
# or: pip install git+https://github.com/felixdaga/fintel.git
```

Dev (clone + editable):

```bash
git clone https://github.com/felixdaga/fintel.git
cd fintel
uv sync --all-extras
```

Python ≥ 3.11. Then copy secrets:

```bash
mkdir -p .env && cp .env.example .env/keys.env
# fill OPENROUTER_API_KEY, MASSIVE_API_KEY, BRAVE_API_KEY, FRED_API_KEY, …
```

`.env/` is gitignored. Never commit keys.

## Quick start

Smoke (2 names × 1 date, ~minutes):

```bash
fintel simulation packages/systematic_stockrate_djia_weekly \
  --agent optimized \
  --job-id smoke \
  --universe AAPL,MSFT \
  --dates 2026-04-24 \
  --shared-concurrency 4

fintel report smoke
```

Full demo (Dow 30 × 15 weekly Fridays, the run shipped in `runs/`):

```bash
fintel simulation packages/systematic_stockrate_djia_weekly \
  --agent optimized \
  --job-id djia-w-full \
  --shared-concurrency 8

fintel report djia-w-full-r1-0001
```

The shipped demo run lives at `runs/djia-w-full-r1-0001/` — inspect it directly
or open `fintel/evaluate/run_analytics.ipynb` for deeper analysis.

## CLI


| Command                                 | Purpose                                        |
| --------------------------------------- | ---------------------------------------------- |
| `fintel simulation <package> --agent …` | Run a job (live TUI by default)                |
| `fintel runs watch <job_id…>`           | Attach / re-attach the dashboard               |
| `fintel backfill <job_id>`              | Rerun failed cells, rewrite artifacts in place |
| `fintel health <job_id>`                | Audit tool / data path                         |
| `fintel report <job_id>`                | KPI + stochasticity + holdings → `report.md`   |
| `fintel cache status`                   | Inspect the central cache (gap-aware)          |




## Docs


| Doc                                                                        | Contents                              |
| -------------------------------------------------------------------------- | ------------------------------------- |
| `[docs/architecture.md](docs/architecture.md)`                             | System design, invariants, layer map  |
| `[docs/add_strategy_package_guide.md](docs/add_strategy_package_guide.md)` | Author a package; bring your own data |
| `[docs/run_and_monitor_guide.md](docs/run_and_monitor_guide.md)`           | Run, TUI, backfill, cache             |
| `[docs/evaluate_results_guide.md](docs/evaluate_results_guide.md)`         | Artifacts, report, what we offer      |
| `[docs/add_new_agents_guide.md](docs/add_new_agents_guide.md)`             | Wire a new agent adapter              |
| `[docs/data_pipeline.md](docs/data_pipeline.md)`                           | Cache / lookback / PIT path           |




## Example packages

- `packages/systematic_stockrate_djia_weekly` — weekly DJIA single-name rating
- `packages/systematic_stockrate_djia_monthly` — same stack, quarter-start grid



## Repository hygiene

- Secrets: only `.env.example` is tracked; real keys live in `.env/keys.env`
- Artifacts: `runs/` and `cache/` are gitignored (the demo run is negated in)
- Operator surface: the `fintel` CLI (no `scripts/` one-offs)
- Contributing: see [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md)
- Changes: see [`CHANGELOG.md`](CHANGELOG.md)

