# Guide: run and monitor

How to launch a simulation, watch it live, recover failed cells, and inspect
cache/health. All of this is the `fintel` CLI — there is no separate script
layer.

## 0. Setup

```bash
# install
uv sync                        # or: pip install -e ".[dev,llm,mcp]"

# secrets (gitignored)
mkdir -p .env && cp .env.example .env/keys.env
# fill OPENROUTER_API_KEY, MASSIVE_API_KEY, BRAVE_API_KEY, FRED_API_KEY, …

# confirm the package preflights
python -c "from fintel.strategy import load_and_prepare; print(load_and_prepare('packages/systematic_stockrate_djia_weekly')[1])"
```

Keys load from `.env/keys.env` automatically. Pass `--no-bootstrap` to skip.
Shell-exported vars always win over the file.

## 1. Run a simulation

```bash
fintel simulation packages/systematic_stockrate_djia_weekly \
  --agent optimized \
  --job-id djia-w-smoke \
  --universe AAPL,MSFT,NVDA \
  --dates 2026-04-24,2026-05-01 \
  --shared-concurrency 6
```

Useful flags:

| Flag | Role |
| ---- | ---- |
| `--agent` | `optimized`, `llm`, `openclaw`, `claude_code`, … |
| `--agent-opt k=v` | Adapter options (repeatable) |
| `--k` | Stochastic repeats (writes `r1`…`rK`) |
| `--universe` / `--dates` | Override package universe/schedule (narrow only) |
| `--cell-concurrency` | Concurrent names per date |
| `--trial-concurrency` | Concurrent dates within a run |
| `--shared-concurrency` | Flat cell pool across dates (preferred for throughput) |
| `--max-concurrent` | Concurrent K repeats |
| `--cache-root` | Shared cache (default `<output-root>/cache`) |
| `--offline` | Cache-only; miss → error |
| `--no-prefetch` | Skip warm-up (cold fills at cell time) |
| `--watch-mode` | `auto` \| `alt` \| `stream` |
| `--no-watch` | Sync run, verbose lines, no dashboard |
| `--quiet` | Suppress live progress (still logs) |

Artifacts land under `runs/<job-id>/` (gitignored). See architecture §10.

### Concurrency quick pick

- **Interactive / OpenClaw:** `--cell-concurrency 1` (session-sensitive agents).
- **In-process optimized / llm:** `--shared-concurrency N` with N ≈ comfortable
  parallel LLM calls (e.g. 8–30 depending on provider limits).
- **Memory/feedback on:** shared pool is blocked — use cell × trial instead.

## 2. Live TUI

### Attached (default)

In a real terminal, `fintel simulation …` starts the job and opens the
dashboard in the foreground. One command.

### Detached / second pane

```bash
# terminal A — start without the foreground dashboard
fintel simulation packages/systematic_stockrate_djia_weekly \
  --agent optimized --job-id djia-w-full --no-watch --shared-concurrency 20

# terminal B — attach (can start before the job exists)
fintel runs watch djia-w-full --wait 60 --watch-mode stream
```

Watch modes:

| Mode | When |
| ---- | ---- |
| `auto` | Default — `stream` inside Cursor-like hosts, `alt` in a real TTY |
| `stream` | In-place redraw, no alt-screen (best for IDE terminals) |
| `alt` | Full-screen alt-screen + cbreak (best for iTerm/Terminal.app) |

Press `q` to detach; the job keeps running. Re-attach anytime with
`fintel runs watch <job_id>`.

For backfill jobs the watcher prefers `backfill.log` over `run.log` when both
exist.

### Multi-job

```bash
fintel runs watch job-a job-b --watch-mode stream
```

## 3. Inspect while / after running

```bash
fintel runs list
fintel runs show <job_id>
fintel health <job_id>
```

**Outcome vs health**

- Outcome (`ok`, `parse_error`, `rate_limited`, …) — what the agent did.
- Health (`ok`, `degraded`, `broken`) — whether tools/data/PIT path worked.

A run full of scores with `health=broken` (dead tools) is not a clean result.

## 4. Backfill failed cells

After a job finishes with error cells:

```bash
fintel backfill <job_id> --run 1 --cell-concurrency 20
fintel runs watch <job_id> --watch-mode stream   # tails backfill.log
```

Backfill reruns every cell whose status is not `ok`/`skipped`, rebuilds trial
decisions, and re-reduces run/job results in place. Strategy-agnostic — uses
the frozen `RunConfig`.

## 5. Cache

```bash
fintel cache status
fintel cache status --source massive_news --symbol AAPL --window 2024-01-01..2026-01-01
```

One central root (`runs/cache/` by default). Prefetch warms it; cells read
through PIT. Details: [`data_pipeline.md`](data_pipeline.md).

## 6. Typical full-run recipe

```bash
# 1) smoke
fintel simulation packages/systematic_stockrate_djia_weekly \
  --agent optimized --job-id smoke --universe AAPL,MSFT \
  --dates 2026-04-24 --shared-concurrency 4

# 2) full job (separate TUI)
fintel simulation packages/systematic_stockrate_djia_weekly \
  --agent optimized --job-id djia-w-full-r1 \
  --shared-concurrency 24 --no-watch &

fintel runs watch djia-w-full-r1 --wait 120 --watch-mode stream

# 3) repair stragglers
fintel backfill djia-w-full-r1 --cell-concurrency 24

# 4) evaluate
fintel report djia-w-full-r1
```

Next: [`evaluate_results_guide.md`](evaluate_results_guide.md).
