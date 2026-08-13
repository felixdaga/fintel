# Guide: understand output and evaluate results

What a finished job writes, how to read it, and what `fintel report` (plus the
analytics notebook) currently offer.

## 1. On-disk layout

```
runs/<job_id>/
  config.json  result.json  job.log  prefetch.json  health.json
  r1/ … rK/
    config.json  result.json  run.log          # (+ backfill.log if used)
    trials/<YYYY-MM-DD>/
      decision.json          # symbol → View (the investment output)
      result.json
      cells/<SYMBOL>.json    # CellRecord (outcome, usage, health, …)
      trace/<SYMBOL>.jsonl
    sessions/…               # agent workspace / access.jsonl
  report/                    # written by `fintel report`
    report.json  report.md
  cache/                     # shared central cache (or --cache-root)
```

`r1` exists even when K=1. Frozen identity lives in `rK/config.json`
(effective universe, schedule, scoring, fingerprint).

### What to open first

| Question | File |
| -------- | ---- |
| Did the job finish cleanly? | `result.json`, `health.json` |
| What did the agent decide? | `rK/trials/<date>/decision.json` |
| Why did a cell fail? | `rK/trials/<date>/cells/<SYM>.json` |
| What did tools return? | `rK/sessions/…/access.jsonl` + traces |
| Is the signal any good? | `report/report.md` after `fintel report` |

## 2. Cell outcomes

Each cell has an agent **outcome** and environment **health**.

Common outcomes: `ok`, `abstained`, `empty`, `parse_error`, `rate_limited`,
`timeout`, `transient`, `refused`, `crashed`.

Retryable outcomes (`empty`, `timeout`, `rate_limited`, `transient`,
`parse_error`) are retried once during the run (default) and are the main
targets for `fintel backfill`.

Health `broken` / `degraded` means the harness or data path failed (tool
schema denials, zero successful reads, PIT-suspect queries). Do not treat a
broken job as a clean alpha test.

```bash
fintel runs show <job_id>
fintel health <job_id>
```

## 3. `fintel report` — what we offer today

```bash
fintel report <job_id>
# writes runs/<job_id>/report/{report.json,report.md} and prints the markdown
```

The report reads the frozen `ScoringSpec` from `rK/config.json` and runs the
evaluation pipeline (architecture §12):

| Layer | Status | What you get |
| ----- | ------ | ------------ |
| **KPI** (`single_name_ir` by default) | strategy-selected | Per-horizon cross-sectional Spearman IC, mean IC, raw ICIR (`mean/std`, not annualized), IC series |
| **L1 Behaviour** | always on | Tool-call / stage stability across cells (n/a if no traces) |
| **L2 Output variance** | always on | Score / sign / rank dispersion across K repeats (informative when K>1) |
| **Holdings** | opt-in via `scoring.params.holdings` | Naive long-only tilt around equal-weight → gross/net NAV with turnover cost |

### Horizons

`horizons = [1, 2, 4, 8]` are **steps on the decision-date grid**, not calendar
weeks/months by themselves. On a weekly Friday grid, `h=1` ≈ one week; on a
monthly first-trading-day grid, `h=1` ≈ one month; on a quarter-start grid,
`h=1` ≈ one quarter. Forward returns use **open→open** prices from the cache
(`market/realized.py`).

### Holdings (MVP)

Not MVO. Default weights:

`w_i = max(0, 1/N + active_budget · s_i / Σ|s|)`, renormalized.

NAV compounds horizon-1 grid returns; net subtracts `cost_bps` × turnover.
Full mean-variance / factor attribution is post-MVP (see the analytics notebook
for exploratory MVO).

### Custom KPI / signal

Packages can point `[scoring].kpi` / `signal` / `transform` at
`module:Callable`. The platform runs whatever is declared — see
[`add_strategy_package_guide.md`](add_strategy_package_guide.md).

## 4. Reading `report.md`

Typical sections:

1. **Job meta** — strategy, K, signal/transform/KPI, horizons, dates, universe.
2. **KPI table** — mean IC / ICIR per horizon (ensemble; per-run if K>1).
3. **Behaviour** — whether the agent path was stable.
4. **Variance** — cross-repeat disagreement (skip concern when K=1).
5. **Holdings** — gross/net NAV path vs the naive tilt (if opted in).

ICIR here is **raw** (mean IC / std IC over the dated IC series). It is not a
Sharpe of portfolio returns.

## 5. Deeper analytics notebook

For exploration beyond the MVP report:

```
fintel/evaluate/run_analytics.ipynb
```

Swap `JOB_DIR` at the top. Sections:

1. Decisions → tidy DataFrame (all View fields except `sources_cited`)
2. Stochasticity (K>1): cross-run rank corr, ticker score spread
3. Score distributions over time
4. Spearman **and** Pearson IC (flexible horizons)
5. Performance: price-weighted DJIA benchmark, score-/equal-weight long
   baskets (empty → cash), naive tilt, MVO overlay
6. Name-level active attribution vs benchmark

Requires the same Python env as fintel (`pandas`, `matplotlib`, optional
`cvxpy` for MVO). This notebook is research tooling — not a substitute for
`fintel report` in CI.

## 6. Reproducibility checklist

- [ ] `health.json` / `fintel health` is clean (or failures understood)
- [ ] Error cells backfilled or explicitly accepted
- [ ] `rK/config.json` scoring matches the package you think you ran
- [ ] Report horizons match the grid cadence you care about
- [ ] For claims about alpha: quote IC/ICIR **and** holdings net NAV, with
      K and date window attached
- [ ] Cache root recorded if not the default (`--cache-root` / `FINTEL_CACHE`)

## 7. What we do **not** ship yet

- Paired job compare (`fintel compare`)
- Production Barra / factor attribution in `fintel report`
- World-validity gate on tool evidence quality

Those are post-MVP; use the notebook for exploratory portfolio construction
and name attribution in the meantime.
