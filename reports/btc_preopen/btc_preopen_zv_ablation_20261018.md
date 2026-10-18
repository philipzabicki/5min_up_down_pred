# BTC pre-open Z/V feature ablation — 2026-10-18

## Scope and frozen comparison

BASE64 is the exact ordered set of 64 source features. BASE70 is the saved B0 model: BASE64 plus taker_Z and taker_V at 1, 5, and 15 minutes. B1 is outside scope. This is a development-only comparison.

The comparison was frozen before fitting. It uses the completed taker-flow run's exact common mask and saved folds: 3,174,805 training rows, including 634,957 decision rows. The mask excludes 11,335 otherwise eligible rows; BASE64 did not restore them. Proxy OOF: 318,560 paired forecasts, 1,107 UTC days, 2023-06-20T16:45:00+00:00 through 2026-06-30T23:50:00+00:00. Official development: 21,951 resolved outcomes, 77 UTC days, 2026-04-15T17:05:00+00:00 through 2026-06-30T23:50:00+00:00.

The completed BASE64 CV and quality checkpoints were reused after fixing the forecast projection and economics report assembly; their hashes matched the unchanged frozen data/model identity.

The saved full-development BASE64 artifacts were incompatible because they used the larger unmasked sample and lacked fold boosters fitted on the shared rows. The only new fit was one fixed-parameter 10-fold BASE64 CV. Its selected iteration was 25, using mean fold log loss + 0.5 × population fold standard deviation, 50 early-stopping rounds, and a 500-round limit. No Optuna or extra configurations were used. BASE70 remains the original B0 at iteration 31; its missing full iteration history was not reconstructed by retraining.

Target: target_preopen_5m_up_binance_index_proxy (Close[Opened+6m] >= Open[Opened+2m]); decision at T−60. The fixed training weights were 0.5900 for decision rows and 0.1025 for auxiliary rows. Training labels were purged at each first validation decision.

## Proxy OOF quality

Lower log loss and Brier are better; higher AUC is better. Deltas are BASE70−BASE64.

| Fold | Rows | BASE64 LL | BASE70 LL | Δ LL | BASE64 Brier | BASE70 Brier | Δ Brier | BASE64 AUC | BASE70 AUC | Δ AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 31,858 | 0.691760 | 0.691638 | -0.000122 | 0.249307 | 0.249246 | -0.000061 | 0.5291 | 0.5308 | +0.0017 |
| 1 | 31,841 | 0.692303 | 0.692130 | -0.000172 | 0.249578 | 0.249492 | -0.000086 | 0.5246 | 0.5266 | +0.0020 |
| 2 | 31,862 | 0.692202 | 0.692164 | -0.000037 | 0.249528 | 0.249509 | -0.000018 | 0.5257 | 0.5263 | +0.0006 |
| 3 | 31,861 | 0.691759 | 0.691798 | +0.000039 | 0.249307 | 0.249326 | +0.000019 | 0.5300 | 0.5295 | -0.0005 |
| 4 | 31,842 | 0.692200 | 0.692174 | -0.000026 | 0.249527 | 0.249514 | -0.000013 | 0.5246 | 0.5247 | +0.0001 |
| 5 | 31,856 | 0.691962 | 0.692029 | +0.000067 | 0.249408 | 0.249442 | +0.000034 | 0.5271 | 0.5262 | -0.0008 |
| 6 | 31,861 | 0.692733 | 0.692722 | -0.000011 | 0.249793 | 0.249787 | -0.000006 | 0.5156 | 0.5170 | +0.0014 |
| 7 | 31,858 | 0.692903 | 0.692884 | -0.000019 | 0.249878 | 0.249868 | -0.000009 | 0.5136 | 0.5143 | +0.0007 |
| 8 | 31,860 | 0.692063 | 0.691997 | -0.000066 | 0.249458 | 0.249426 | -0.000033 | 0.5257 | 0.5262 | +0.0005 |
| 9 | 31,861 | 0.691948 | 0.691807 | -0.000141 | 0.249401 | 0.249330 | -0.000071 | 0.5289 | 0.5305 | +0.0017 |

Overall BASE64: LL 0.692183, Brier 0.249519, AUC 0.5244. BASE70: LL 0.692134, Brier 0.249494, AUC 0.5251.

Paired 3-day UTC block bootstrap, 2,000 replications (seed 20261018), BASE70−BASE64 95% intervals: LL [-0.000107, +0.000008], Brier [-0.000053, +0.000004], AUC [+0.0000, +0.0014].

## Official development quality and calibration

| Score | Rows | BASE64 LL | BASE70 LL | BASE64 Brier | BASE70 Brier | BASE64 AUC | BASE70 AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 21,951 | 0.692129 | 0.692020 | 0.249491 | 0.249437 | 0.5268 | 0.5278 |
| Daily Platt, common mask | 20,718 | 0.692215 | 0.692152 | 0.249534 | 0.249503 | 0.5259 | 0.5266 |

Each model had its own daily Platt calibrator. It used only earlier official labels released before that UTC day's first decision, with at least 1,000 prior labels and both classes. Recreated BASE70 probabilities match the saved cache on 21,951 rows, with max absolute difference 3.33e-16.

| UTC month | Score | Rows | BASE64 LL | BASE70 LL | Δ LL | BASE64 Brier | BASE70 Brier | Δ Brier | BASE64 AUC | BASE70 AUC | Δ AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-04 | Raw | 4,401 | 0.691826 | 0.691574 | -0.000252 | 0.249340 | 0.249214 | -0.000126 | 0.5271 | 0.5330 | +0.0059 |
| 2026-05 | Raw | 8,917 | 0.692226 | 0.692259 | +0.000032 | 0.249540 | 0.249556 | +0.000016 | 0.5258 | 0.5246 | -0.0012 |
| 2026-06 | Raw | 8,633 | 0.692183 | 0.692000 | -0.000184 | 0.249518 | 0.249427 | -0.000092 | 0.5282 | 0.5296 | +0.0015 |
| 2026-04 | Daily Platt | 3,168 | 0.691800 | 0.691558 | -0.000241 | 0.249328 | 0.249208 | -0.000120 | 0.5274 | 0.5309 | +0.0035 |
| 2026-05 | Daily Platt | 8,917 | 0.692342 | 0.692439 | +0.000097 | 0.249598 | 0.249646 | +0.000048 | 0.5243 | 0.5234 | -0.0009 |
| 2026-06 | Daily Platt | 8,633 | 0.692237 | 0.692073 | -0.000165 | 0.249545 | 0.249463 | -0.000082 | 0.5266 | 0.5280 | +0.0014 |

Official raw bootstrap 95% intervals for BASE70−BASE64: LL [-0.000309, +0.000078], Brier [-0.000154, +0.000039], AUC [-0.0018, +0.0039]. Daily-Platt common-mask intervals: LL [-0.000324, +0.000214], Brier [-0.000162, +0.000107], AUC [-0.0023, +0.0036].

BASE70 had lower proxy log loss in 8/10 folds, lower raw official log loss in 2/3 months, and lower calibrated official log loss in 2/3 months. Monthly results are descriptive and did not select thresholds or periods.

## Fixed-rule economics

Each model chose its own side and qualification on the same qualifying development calendar: independent $5 orders, 1% net-EV hurdle, T−60 decisions, T−59 execution, saved fees, depth, and settlement cash-release cutoff. Missing predictions and missing official outcome or settlement data remain separate statuses. These orders are not a portfolio return and are not scaled from $100.

| Model | Decisions | Fills | Settlements | Turnover | Fees | Net PnL | PnL / turnover | Forecast EV on settled fills |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE64 | 21,962 | 1,445 | 1,445 | $7,225.00 | $258.64 | $194.27 | 2.6889% | $355.36 |
| BASE70 | 21,962 | 1,452 | 1,452 | $7,260.00 | $259.71 | $260.05 | 3.5819% | $356.58 |

BASE70 reproduces the saved anchor: 1,452 fills, $7,260.00 turnover, $260.05 net PnL. Its hash-verified completed ledger was reused. BASE64 skip reasons and missing-data statuses are in economic_comparison.json.

| UTC decision month | Model | Decisions | Fills | Turnover | Fees | Net PnL | Forecast EV |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026-04 | BASE64 | 4,402 | 144 | $720.00 | $25.73 | $69.19 | $25.09 |
| 2026-05 | BASE64 | 8,927 | 574 | $2,870.00 | $105.36 | $57.29 | $196.61 |
| 2026-06 | BASE64 | 8,633 | 727 | $3,635.00 | $127.55 | $67.79 | $133.66 |
| 2026-04 | BASE70 | 4,402 | 163 | $815.00 | $29.16 | $22.29 | $27.94 |
| 2026-05 | BASE70 | 8,927 | 565 | $2,825.00 | $103.70 | $99.49 | $198.69 |
| 2026-06 | BASE70 | 8,633 | 724 | $3,620.00 | $126.85 | $138.26 | $129.95 |

Five best decision days were 120.2% of BASE64 net PnL and 100.8% of BASE70 net PnL.

The paired economic bootstrap used all 77 UTC days, including days without trades, in shared circular 3-day blocks (2,000 replications; seed 20261018):

| Quantity | Estimate | 95% interval |
|---|---:|---:|
| BASE64 mean daily PnL | $2.5230 | $-1.3239 to $6.5371 |
| BASE70 mean daily PnL | $3.3772 | $-0.2390 to $7.4847 |
| BASE70−BASE64 mean daily PnL | $0.8542 | $-2.7327 to $4.7227 |
| BASE64 PnL / turnover | 2.6889% | -1.6770% to 6.7488% |
| BASE70 PnL / turnover | 3.5819% | -0.3409% to 7.0498% |

Valid ratio replications: BASE64 2,000/2,000; BASE70 2,000/2,000. Replications with zero turnover have no ratio estimate.

## Conclusion and limits

1. BASE70 had slightly better aggregate point estimates for log loss, Brier, and AUC on proxy OOF and official development scores. The proxy AUC interval is positive; proxy log-loss and Brier intervals and all official intervals include zero. This is a limited proxy AUC signal, not broad evidence of a reliable quality improvement.
2. The effect was **distributed across multiple periods**: BASE70 won 8/10 proxy folds and 2/3 calibrated official months by log loss.
3. BASE70 PnL minus BASE64 was $65.78; the interval for the mean daily PnL difference was $-2.7327 to $4.7227.
4. BASE64 net PnL under the same sample and calibration was $194.27; a positive result was already present without the additions.

The 64 selected features, hyperparameters, and iteration selection used development data. These intervals are conditional on that procedure and are not independent confirmation of generalization. Holdout labels and PnL were not evaluated; no deployment refit was run.

End-to-end time: 1351.9s. Sampled peak process working set: 12.26 GiB. BASE64 CV peak working set: 12.264 GiB. Hardware, stage times, and artifact hashes are in the run manifest.

Artifacts: data/analysis/polymarket/BTC/btc_preopen_zv_ablation_20261018/run_manifest.json, reports/btc_preopen/btc_preopen_zv_ablation_20261018_manifest.json, BASE64 fold models and CV history under data/analysis/polymarket/BTC/btc_preopen_zv_ablation_20261018/models/BASE64, and OOF, tables, and daily PnL under data/analysis/polymarket/BTC/btc_preopen_zv_ablation_20261018. The original BASE70 full iteration history was unavailable and was not recreated.
