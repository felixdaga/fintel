# fintel report — systematic_stockrate_djia_weekly_optimized_20260806_mimo_v2_5_pro_k1_0001

strategy: `packages/systematic_stockrate_djia_weekly`  agent repeats: 1  signal: `single_name`  transform: `single_name`  kpi: `single_name_ir`
dates: 15  universe: 31 (AAPL, AMGN, AMZN, AXP, BA, CAT, CRM, CSCO…)

## KPI
ensemble (single_name_ir, metric_key=icir):

| horizon | mean_ic | raw_icir | n_periods |
|---|---|---|---|
| 1 | 0.0149 | 0.0752 | 14 |
| 2 | 0.0341 | 0.1707 | 13 |
| 4 | 0.0251 | 0.1295 | 11 |
| 8 | -0.0129 | -0.1046 | 7 |

## Output variance (L2)
_need >= 2 runs for cross-run variance_

## Behaviour (L1)
cells: 0  mean_call_count_std: 0.0

## Holdings & returns (opt-in)
active_budget: 0.5  cost_bps: 5.0  turnover_total: 3.2153
ensemble NAV: gross 1.061719  net 1.060018
