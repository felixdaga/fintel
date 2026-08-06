# Changelog

All notable changes to fintel are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `fintel backfill` — rerun only the failed cells of a finished run, rewriting
  cell / trial / run / job artifacts in place.
- `--shared-concurrency` flat pool: N cells in flight across all (date, ticker)
  pairs in a run, replacing the nested cell × trial fan-out.
- `fintel cache status` — read-only inspection of the central cache with
  gap-aware coverage sidecars.
- Live TUI dashboard: per-repeat tracks, per-cell staging grid, stuck
  heuristics. Modes: `auto` / `alt` / `stream`.
- `fintel/evaluate/run_analytics.ipynb` — deeper analytics notebook (decision
  import, stochasticity, KPI, holdings, attribution).
- `docs/` guides: architecture, add strategy package, run & monitor,
  evaluate results, add agents, data pipeline.
- `AGENTS.md`, `CONTRIBUTING.md`, `CHANGELOG.md`.
- CI workflow (`.github/workflows/ci.yml`): ruff + pytest on push/PR.
- Demo run shipped in `runs/djia-w-full-r1-0001/`.

### Changed
- Replaced the legacy monthly DJIA package with one aligned to the weekly
  package (same data surface, mission, output contract; date / horizon only).
- `classify_status` / `classify_exit` tightened to avoid false-positive
  refusals from model prose in stdout.
- Verifier tool-call failures are now flagged rather than crashing the cell;
  `AgentResult.verification_flag` added.
- PM failure to call `submit_views` is now `parse_error` (retryable) instead
  of `crashed`.

### Removed
- Legacy one-off scripts under `scripts/` — the `fintel` CLI is the only
  operator surface.

## [0.1.0] — Initial public release

- Strategy package model: `strategy.toml` + `mission.md` + `output_schema.json`.
- Execution hierarchy: Job → Run → Trial → Cell.
- Point-in-time data layer: catalog, clamp, audit, prefetch, probe.
- Agent adapters: in-process (`llm`, `scripted`, `constant`, `optimized`)
  and subprocess CLI (`openclaw`, `claude-code`) via MCP.
- Evaluation layer: signals, transforms, KPI (single_name_ir), holdings,
  behaviour (L1), variance (L2), report.
- Layer map enforced by `tests/test_architecture.py`.
