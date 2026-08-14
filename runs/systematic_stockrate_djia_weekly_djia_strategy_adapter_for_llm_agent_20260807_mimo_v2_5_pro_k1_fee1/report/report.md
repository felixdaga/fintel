# fintel report — systematic_stockrate_djia_weekly_djia_strategy_adapter_for_llm_agent_20260807_mimo_v2_5_pro_k1_fee1

strategy: `/Users/lucky/.openclaw/workspace/projects/fintel/packages/systematic_stockrate_djia_weekly`  agent repeats: 1  signal: `single_name`  transform: `single_name`  kpi: `single_name_ir`
dates: 15  universe: 31 (AAPL, AMGN, AMZN, AXP, BA, CAT, CRM, CSCO…)

## KPI
ensemble (single_name_ir, metric_key=icir):

| horizon | mean_ic | raw_icir | n_periods |
|---|---|---|---|
| 1 | 0.0298 | 0.1362 | 14 |
| 2 | 0.0295 | 0.1522 | 13 |
| 4 | 0.0620 | 0.3225 | 11 |
| 8 | -0.0231 | -0.2262 | 7 |

## Output variance (L2)
_need >= 2 runs for cross-run variance_

## Behaviour (L1)
cells: 0  mean_call_count_std: 0.0

## Holdings & returns (opt-in)
active_budget: 0.5  cost_bps: 5.0  turnover_total: 1.6745
ensemble NAV: gross 1.116395  net 1.115464
