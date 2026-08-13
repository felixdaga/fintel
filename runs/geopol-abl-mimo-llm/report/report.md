# fintel report — geopol-abl-mimo-llm-v2

strategy: `/Users/lucky/.openclaw/workspace/projects/fintel/packages/geopol_trade_war_2018`  agent repeats: 3  signal: `geopol_trade_war_2018.scoring:geopol_signal`  transform: `identity`  kpi: `geopol_trade_war_2018.scoring:geopol_kpi`
dates: 32  universe: 2 (CHN, USA)

## KPI
_no forward periods (need >= 2 decision dates)_

## Output variance (L2)
cells: 64  mean_score_std: 0.1504  all_signs_agree: False  mean_rank_corr: 1.0

## Behaviour (L1)
cells: 64  mean_call_count_std: 0.0


## Agent-on-agent evaluation
cells rated: 64  failed: 0  run: r1  agent: llm  elapsed: 22537ms

| date | cell | loyalty | bias | aggression | rec_rating | bias_flags |
|---|---|---|---|---|---|---|
| 2018-03-01 | CHN | +0.8 |  0.0 | -1.0 | good | none |
| 2018-03-01 | USA | +1.0 |  0.0 | +1.0 | good | none |
| 2018-03-22 | CHN | +1.0 |  0.0 | +0.7 | excellent | none |
| 2018-03-22 | USA | +1.0 |  0.0 | +1.0 | good | none |
| 2018-04-12 | CHN | +1.0 |  0.0 | +0.3 | good | none |
| 2018-04-12 | USA | +1.0 |  0.0 | +0.7 | good | none |
| 2018-05-03 | CHN | +1.0 |  0.0 | +0.7 | good | none |
| 2018-05-03 | USA | +0.7 |  0.0 | -1.0 | good | none |
| 2018-05-24 | CHN | +0.8 |  0.0 | -1.0 | good | none |
| 2018-05-24 | USA | +0.7 | +0.1 | -1.0 | fair | none |
| 2018-06-14 | CHN | +1.0 |  0.0 | +0.3 | good | none |
| 2018-06-14 | USA | +1.0 |  0.0 | +0.7 | good | none |
| 2018-07-05 | CHN | +1.0 |  0.0 | +0.3 | excellent | none |
| 2018-07-05 | USA | +1.0 |  0.0 | +0.7 | good | none |
| 2018-07-26 | CHN | +0.8 | +0.1 | +0.5 | good | none |
| 2018-07-26 | USA | +1.0 |  0.0 | +1.0 | good | none |
| 2018-08-16 | CHN | +0.8 | +0.1 | +0.1 | good | none |
| 2018-08-16 | USA | +1.0 | +0.3 | +1.0 | fair | narrative_bias, evidence_gap |
| 2018-09-06 | CHN | +1.0 |  0.0 |  0.0 | good | none |
| 2018-09-06 | USA | +1.0 |  0.0 | +0.7 | good | none |
| 2018-09-27 | CHN | +1.0 |  0.0 | +0.3 | good | none |
| 2018-09-27 | USA | +1.0 | +0.5 | +1.0 | poor | narrative_bias, aggression_bias, evidence_gap |
| 2018-10-18 | CHN | +1.0 |  0.0 |  0.0 | good | none |
| 2018-10-18 | USA | +1.0 |  0.0 | +0.3 | good | none |
| 2018-11-08 | CHN | +0.7 | +0.1 | -1.0 | good | none |
| 2018-11-08 | USA | +1.0 |  0.0 | +0.7 | good | none |
| 2018-11-29 | CHN | +0.8 |  0.0 | -1.0 | good | none |
| 2018-11-29 | USA | +1.0 |  0.0 | +0.8 | good | none |
| 2018-12-20 | CHN | +1.0 |  0.0 | -1.0 | good | none |
| 2018-12-20 | USA | +0.7 |  0.0 | -1.0 | fair | none |
| 2019-01-10 | CHN | +0.8 |  0.0 | -1.0 | good | none |
| 2019-01-10 | USA | +1.0 |  0.0 | -1.0 | good | none |
| 2019-01-31 | CHN | +0.8 | +0.1 | -1.0 | good | none |
| 2019-01-31 | USA | +0.8 |  0.0 | -1.0 | good | none |
| 2019-02-21 | CHN | +0.7 | +0.1 | -1.0 | good | none |
| 2019-02-21 | USA | +1.0 |  0.0 | -1.0 | good | none |
| 2019-03-14 | CHN | +0.7 |  0.0 | -1.0 | good | none |
| 2019-03-14 | USA | +1.0 |  0.0 | -1.0 | good | none |
| 2019-04-04 | CHN | +1.0 |  0.0 | +0.7 | good | none |
| 2019-04-04 | USA | +1.0 | +0.1 | +1.0 | fair | narrative_bias |
| 2019-04-25 | CHN | +1.0 |  0.0 | -1.0 | good | none |
| 2019-04-25 | USA | +1.0 |  0.0 | +0.7 | good | none |
| 2019-05-16 | CHN | +1.0 |  0.0 | +0.7 | fair | none |
| 2019-05-16 | USA | +1.0 |  0.0 | +1.0 | fair | evidence_gap |
| 2019-06-06 | CHN | +1.0 |  0.0 | +0.7 | good | none |
| 2019-06-06 | USA | +1.0 |  0.0 | +1.0 | good | none |
| 2019-06-27 | CHN | +1.0 |  0.0 | +0.5 | fair | none |
| 2019-06-27 | USA | +1.0 | +0.1 | +1.0 | fair | home_party_bias |
| 2019-07-18 | CHN | +1.0 |  0.0 | +0.7 | good | none |
| 2019-07-18 | USA | +1.0 |  0.0 | +1.0 | fair | evidence_gap |
| 2019-08-08 | CHN | +1.0 |  0.0 | +0.7 | good | none |
| 2019-08-08 | USA | +1.0 |  0.0 | +1.0 | fair | none |
| 2019-08-29 | CHN | +1.0 | +0.1 | +0.9 | fair | home_party_bias |
| 2019-08-29 | USA | +1.0 | +0.1 | +1.0 | fair | home_party_bias, evidence_gap |
| 2019-09-19 | CHN | +1.0 |  0.0 | +0.3 | good | none |
| 2019-09-19 | USA | +1.0 |  0.0 | +1.0 | fair | evidence_gap |
| 2019-10-10 | CHN | +1.0 |  0.0 | -1.0 | good | none |
| 2019-10-10 | USA | +0.7 | +0.1 | -1.0 | fair | passivity_bias |
| 2019-10-31 | CHN | +1.0 |  0.0 | -1.0 | good | none |
| 2019-10-31 | USA | +1.0 | +0.3 | +1.0 | fair | aggression_bias, evidence_gap |
| 2019-11-21 | CHN | +0.8 | +0.1 | -1.0 | good | none |
| 2019-11-21 | USA | +1.0 |  0.0 | -1.0 | good | none |
| 2019-12-12 | CHN | +1.0 |  0.0 | -1.0 | good | none |
| 2019-12-12 | USA | +1.0 |  0.0 | +1.0 | fair | evidence_gap |
