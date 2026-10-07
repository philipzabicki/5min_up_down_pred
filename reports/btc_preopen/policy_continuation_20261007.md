# BTC pre-open policy continuation — 2026-10-07

All policies remain inactive. This retrospective development period has earlier selection exposure and is not an independent holdout.

## Findings

- Position-state correction: 0 changed decisions across 15 policies; the corrected logs contain no decision moment with a known-but-locked payout. The event schedule explains why: resolutions arrive 314–752 seconds after start and cash releases at the later of resolution or +300 seconds, plus 60 seconds.
- DRK intensity: selected η by outer fold `{'1': 0.0, '2': 0.25, '3': 0.25}`. The selected-η outer portfolio PnL was $230.43, versus $312.08 for η=0 point Kelly; this is a development comparison, not evidence of stable improvement.
- RCK: the corrected-state α=0.5 correlated comparison matches v1 at $33.52. Its wealth denominator remains cost-basis equity, an explicit approximation rather than mark-to-market value; paired sizing separates stake effects from constraint rejections.
- Optimal stopping: no exit rule dominates PnL and risk. `hold_to_resolution` has the highest full-portfolio mean daily log growth (0.091518). On independently funded identical buys, ridge was positive in 3/3 folds, tree in 3/3, and simple expected value in 1/3; fold PnL is not a portfolio return. These are development results, not a holdout.
- All full-portfolio exit policies use identical legacy exact-$5 entry selection and reinvest sale proceeds. The isolated analysis uses identical accepted buys/stakes but each trade is independently funded from $100.

## Full-portfolio risk and growth

| Exit rule | PnL | Mean daily log growth | Cost-basis DD | Liquidation DD | Liquidation coverage | Max exposure | Max underwater | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hold_to_resolution | $461.42 | 0.091518 | 75.37% | 77.18% | 86.7% | $15.00 | 456894s | 2285 |
| simple_expected_value | $6.94 | 0.009108 | 74.41% | 75.09% | 95.7% | $10.00 | 1065295s | 2285 |
| ridge | $146.62 | 0.042837 | 55.74% | 55.82% | 94.9% | $10.00 | 614760s | 2285 |
| tree | $43.94 | 0.021830 | 42.56% | 43.93% | 96.7% | $10.00 | 831344s | 2285 |

## Separate reports

- `position_state_comparison_20261007.md` — v1 against corrected position state.
- `paired_sizing_20261007.md` — common/one-sided feasibility and paired PnL, including actual fixed-$5 admitted amounts.
- `drk_intensity_20261007.md` — causal internal η selection and outer comparison.
- `optimal_stopping_20261007.md` — identical-buy exit comparison, reinvested portfolio, and execution coverage.
- `trajectory_coverage_os_20261007.csv` — per-market PMXT/Kacho coverage.
- `audit.json` — local PMXT schema/event provenance and the audit of other locally available Parquet files.

Outcome availability uses `resolved_at_utc` as an explicit proxy because the source does not record the first receipt time of the official label. Entry fee rate is held constant for post-entry sales because per-tick fee history is absent. The saved Kacho data required no network retrieval; added data cost was $0.
