# DRK intensity selection — 2026-10-07

The frozen grid is η ∈ {0, 0.25, 0.5, 0.75, 1}. Intervals are expanded to include the point estimate before interpolation; DOWN uses the complement of the adjusted UP interval. η=0 is point Kelly and η=1 is the prior full 5–95% DRK rule.

| Outer fold | Internal validation rows | Internal prior rows | Selected η |
|---:|---:|---:|---:|
| 1 | 891 | 889 | 0.00 |
| 2 | 890 | 1781 | 0.25 |
| 3 | 891 | 2671 | 0.25 |

## Internal validation scores

| Fold | η | Mean daily log growth | PnL | Trades |
|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.002332 | $1.88 | 22 |
| 1 | 0.25 | 0.002332 | $1.88 | 22 |
| 1 | 0.50 | 0.002332 | $1.88 | 22 |
| 1 | 0.75 | 0.002332 | $1.88 | 22 |
| 1 | 1.00 | 0.002332 | $1.88 | 22 |
| 2 | 0.00 | -0.044898 | $-30.18 | 648 |
| 2 | 0.25 | -0.035584 | $-24.77 | 593 |
| 2 | 0.50 | -0.040478 | $-27.66 | 466 |
| 2 | 0.75 | -0.035989 | $-25.02 | 257 |
| 2 | 1.00 | -0.073373 | $-44.40 | 58 |
| 3 | 0.00 | 0.056633 | $48.65 | 472 |
| 3 | 0.25 | 0.084210 | $80.30 | 477 |
| 3 | 0.50 | 0.028313 | $21.92 | 344 |
| 3 | 0.75 | 0.014346 | $10.56 | 222 |
| 3 | 1.00 | -0.011721 | $-7.88 | 110 |

## Outer comparison

| Variant | PnL | Mean daily log growth | Cost-basis max DD | Trades | Max exposure |
|---|---:|---:|---:|---:|---:|
| point_eta0 | $312.08 | 0.067431 | 86.139% | 1824 | $40.00 |
| full_interval_eta1 | $-49.07 | -0.032128 | 70.240% | 285 | $15.41 |
| selected_eta | $230.43 | 0.056915 | 83.324% | 1595 | $39.69 |

## Outer fold economic results

| Variant | Fold | η | Trades | Mean admitted stake | Total admitted stake | Expected net PnL at entry | Realized PnL |
|---|---:|---:|---:|---:|---:|---:|---:|
| point_eta0 | 1 | 0.00 | 649 | $4.13 | $2677.41 | $126.59 | $-27.73 |
| point_eta0 | 2 | 0.00 | 505 | $3.72 | $1877.74 | $97.94 | $98.26 |
| point_eta0 | 3 | 0.00 | 670 | $7.89 | $5283.61 | $290.60 | $241.55 |
| selected_eta | 1 | 0.00 | 649 | $4.13 | $2677.41 | $126.59 | $-27.73 |
| selected_eta | 2 | 0.25 | 398 | $3.53 | $1404.04 | $80.11 | $77.09 |
| selected_eta | 3 | 0.25 | 548 | $6.27 | $3436.68 | $210.34 | $181.07 |
| full_interval_eta1 | 1 | 1.00 | 55 | $2.94 | $161.84 | $22.41 | $-46.84 |
| full_interval_eta1 | 2 | 1.00 | 48 | $2.55 | $122.50 | $10.06 | $-7.50 |
| full_interval_eta1 | 3 | 1.00 | 182 | $2.81 | $510.55 | $39.81 | $5.27 |

The internal calibrator used fixed C=1.0 and only labels available before that fold's validation block. Bootstrap intervals use the unchanged 3-day block length and 500 replicates. This remains development-period research, not a new holdout.
