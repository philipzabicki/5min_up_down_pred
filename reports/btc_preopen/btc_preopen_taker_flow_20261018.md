# BTC pre-open taker-flow experiment — 2026-10-18

## Run and source

- Status: `completed`. The model comparison uses the completed binary full-development run `794a69e369a3f82d2ec161af839cdb7992339e34a6606b8bce7712da7fd610f7` and its exact 64-feature order.
- Source training rows: 3,186,140; decision rows: 637,228; common complete rows for both variants: 3,174,805 (634,957 decision rows).
- The source's 10 chronological fold fingerprints were recovered and matched exactly before applying the shared completeness mask. Validation rows remain decision minutes only, and training labels are purged at the source fold boundary.
- Saved target remains `target_preopen_5m_up_binance_index_proxy`: `Close[Opened+6m] >= Open[Opened+2m]`. For a noon market, the last closed candle opens at 11:58 UTC, the information boundary is 11:59, and the market window is 12:00–12:05 UTC. No features use an unclosed candle.
- Model parameters: `{"feature_fraction": 0.7962744599393133, "lambda_l1": 9.508616956284627, "lambda_l2": 30.729843292781943, "learning_rate": 0.06418546619439842, "min_data_in_leaf": 12089, "num_leaves": 81}`; full-development fit seed `20261015`; fixed decision/auxiliary weights `0.59` / `0.1025`. Execution used 8 LightGBM threads to preserve headroom; fold logic, weights, GPU backend, and selected hyperparameters were held fixed.
- The control variant B0 uses the 64 source features plus `taker_Z` and `taker_V` at 1, 5, and 15 minutes (70 total). B1 adds `taker_I`, `taker_U`, `taker_beta`, and `taker_A` at those horizons (82 total). Both variants use the same rows.

## Result

The taker-flow additions did not improve this frozen pipeline. Proxy OOF scores were nearly tied and their paired 95% bootstrap intervals include zero; on the 21,951 official development outcomes, B1 had worse log loss, Brier score, and AUC, with all three paired intervals favoring B0. In the fixed-$5 replay, net PnL was $115.67 for B1 versus $260.05 for B0. These results give no evidence to add the new features to the current model; the comparison used development data only and does not establish holdout performance.

## Binance USD-M Futures source

Data came from official Binance Public Data monthly BTCUSDT 1-minute USD-M Futures kline archives, June 2020 through June 2026. Monthly ZIP files were checked against their SHA-256 sidecars. We used only OHLC, quote volume, taker-buy quote volume, and source timestamps; no trades or order-book data were downloaded. Binance documents the monthly archive and kline fields in its [Public Data README](https://github.com/binance/binance-public-data).

- Monthly archives checked: 73; valid rows on the expected monthly range: 3,198,240; missing or invalid minutes on the full grid: 0.
- Invalid OHLC rows: 0; rows where quote volume or taker-buy volume failed `0 ≤ taker_buy_quote_volume ≤ quote_volume`: 0; duplicate timestamp rows excluded: 0.
- Missing bars and invalid quote/taker values remain missing. No forward fill or clipping is applied.

For each horizon `h`, `Q=sum(quote_volume)`, `D=sum(2*taker_buy_quote_volume−quote_volume)`, `I=D/Q`, and `U=D/(h*qbar)`, with `qbar` the mean quote volume in the preceding 1,440 minutes. `Z=log(Close_end/Open_start)/(sqrt(h)*max(sigma,1e-6))`, where `sigma` is the preceding 60-minute standard deviation of BTCUSDT log returns. `V=log1p(Q/(h*qbar))`. `beta` is the rolling slope of `Z` on `U` over valid pairs in the previous 10,080 minutes (minimum 1,440 pairs), excluding the current horizon; `A` is the regression residual. Near-zero `U` variance sets beta to zero. The old `candle_signed_vol` feature is candle-body direction (`sign(close-open)`), and `session_close_impulse_BVC` is a separate proxy; neither contains Binance taker-buy quote volume.

## Development proxy OOF quality

One fixed-parameter 10-fold CV produced both the selected iteration and the fold boosters used for OOF; there were no Optuna trials and no second OOF fit.

| Variant | Features | Best iteration | CV log loss mean | CV fold SD | CV objective | OOF rows | OOF log loss | OOF Brier | OOF AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 70 | 31 | 0.692134 | 0.000375 | 0.692322 | 318,560 | 0.692134 | 0.249494 | 0.5251 |
| B1 | 82 | 28 | 0.692161 | 0.000359 | 0.692341 | 318,560 | 0.692161 | 0.249507 | 0.5246 |

B0's post-fit summary was recovered from its saved fold boosters and persisted OOF predictions; the boosters reproduced those predictions and no fit was repeated. Its full per-iteration history was not persisted, and its runtime is inferred from the OOF file timestamp.

| Fold | Rows | B0 log loss | B1 log loss | B0 Brier | B1 Brier | B0 AUC | B1 AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 31,858 | 0.691638 | 0.691699 | 0.249246 | 0.249277 | 0.5308 | 0.5299 |
| 1 | 31,841 | 0.692130 | 0.692244 | 0.249492 | 0.249549 | 0.5266 | 0.5244 |
| 2 | 31,862 | 0.692164 | 0.692054 | 0.249509 | 0.249454 | 0.5263 | 0.5270 |
| 3 | 31,861 | 0.691798 | 0.691708 | 0.249326 | 0.249281 | 0.5295 | 0.5312 |
| 4 | 31,842 | 0.692174 | 0.692195 | 0.249514 | 0.249524 | 0.5247 | 0.5243 |
| 5 | 31,856 | 0.692029 | 0.691991 | 0.249442 | 0.249423 | 0.5262 | 0.5274 |
| 6 | 31,861 | 0.692722 | 0.692775 | 0.249787 | 0.249814 | 0.5170 | 0.5152 |
| 7 | 31,858 | 0.692884 | 0.692798 | 0.249868 | 0.249825 | 0.5143 | 0.5160 |
| 8 | 31,860 | 0.691997 | 0.692188 | 0.249426 | 0.249521 | 0.5262 | 0.5233 |
| 9 | 31,861 | 0.691807 | 0.691956 | 0.249330 | 0.249405 | 0.5305 | 0.5286 |

## Official development OOF quality

These metrics use the official Polymarket labels available for development markets from 2026-04-15T17:05:00+00:00 to before 2026-07-01T00:00:00+00:00. Raw metrics score all joined official outcomes. Platt calibration is fitted separately for B0 and B1 once per UTC day, using only earlier OOF official labels whose release time is before that day's first decision; the calibrator stays unavailable until it has 1,000 prior labels and both classes.

| Forecast | Rows | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|
| B0 raw | 21,951 | 0.692020 | 0.249437 | 0.5278 |
| B1 raw | 21,951 | 0.692294 | 0.249574 | 0.5244 |
| B0 daily Platt | 20,718 | 0.692152 | 0.249503 | 0.5266 |
| B1 daily Platt | 20,718 | 0.692389 | 0.249621 | 0.5237 |

The proxy label matched the official outcome on 21,259/21,951 joined development OOF markets (96.85%).

| Fold | Official rows | B0 log loss | B1 log loss | B0 Brier | B1 Brier | B0 AUC | B1 AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 21,951 | 0.692020 | 0.692294 | 0.249437 | 0.249574 | 0.5278 | 0.5244 |

## Paired uncertainty and forecast changes

Intervals below are for B1 minus B0 on identical rows, using 2,000 paired circular moving-block bootstrap replications over 3-day UTC blocks. Negative log-loss/Brier deltas favor B1; positive AUC deltas favor B1.

| Scope | Rows | Log-loss delta 95% interval | Brier delta 95% interval | AUC delta 95% interval | Replications |
|---|---:|---:|---:|---:|---:|
| Proxy OOF | 318,560 | [-0.000038, +0.000090] | [-0.000019, +0.000045] | [-0.00135, +0.00030] | 2000 |
| Official raw OOF | 21,951 | [+0.000075, +0.000484] | [+0.000037, +0.000241] | [-0.00677, -0.00021] | 2000 |

Raw probability delta (`B1−B0`): n=318,560, mean=+0.000637, SD=0.009601, mean absolute=0.007581, p01/p05/median/p95/p99=-0.021655/-0.014870/+0.000524/+0.016518/+0.024081; positive on 52.19% of rows.

## Use of the additional features

B1 used 12 of 12 added features in at least one tree split, and 12 in every fold. The joint permutation diagnostic permuted the 12 added columns together within each validation fold without refitting; mean log-loss increase was 0.000115.

| Added feature | Split count across folds | Folds with splits | Gain across folds |
|---|---:|---:|---:|
| `taker_I_1m` | 81 | 10/10 | 491.657 |
| `taker_U_1m` | 104 | 10/10 | 602.776 |
| `taker_beta_1m` | 266 | 10/10 | 1710.729 |
| `taker_A_1m` | 42 | 10/10 | 192.776 |
| `taker_I_5m` | 265 | 10/10 | 2909.534 |
| `taker_U_5m` | 465 | 10/10 | 6927.229 |
| `taker_beta_5m` | 278 | 10/10 | 1810.183 |
| `taker_A_5m` | 141 | 10/10 | 748.629 |
| `taker_I_15m` | 142 | 10/10 | 858.804 |
| `taker_U_15m` | 298 | 10/10 | 2600.660 |
| `taker_beta_15m` | 393 | 10/10 | 3086.965 |
| `taker_A_15m` | 185 | 10/10 | 974.034 |

## Fixed-$5 development economics

The existing causal portfolio engine independently recomputed each model over qualifying official markets, with $5 gross stake, a 1% T-60 net-EV hurdle, T-59 execution snapshots, existing fees/depth rules, and settlement cash release. The saved T-60/T-59 snapshots were hash-verified and reused; no book replay was run. Forecast EV sums T-60 net EV for filled trades; realized PnL uses T-59 fills and official settlements.

| Variant | Decisions | Fills | Settlements | Turnover | Fees | Net PnL | PnL / turnover | Forecast EV | Realized PnL | Audit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B0 | 21,962 | 1,452 | 1,452 | $7,260.00 | $259.71 | $260.05 | 0.0358 | $356.58 | $260.05 | passed |
| B1 | 21,962 | 1,070 | 1,070 | $5,350.00 | $193.63 | $115.67 | 0.0216 | $288.78 | $115.67 | passed |

## Runtime and artifacts

- Archive validation and features: 35.3s; peak 1.73 GiB.
- Binance ZIP acquisition: at least 176.9s from the first archive file creation to the last ZIP write; the exact first-attempt stage timer was overwritten on resume. Later ZIP/checksum revalidation took 0.9s across 73 monthly files.
- Shared rows and fold reconstruction: 27.0s; peak 11.67 GiB.
- B0 CV: 39.2s; peak unavailable; observed at least 12.98 GiB during CV. B1 CV: 40.3s; peak 16.45 GiB. B1 estimate after B0 was 45.9 seconds.
- OOF, bootstrap, calibration, and economics: 888.8s; sampled working set reached at least 5.58 GiB, but continuous peak was not sampled.
- Total experiment elapsed from the first attempt through completion: 1481.4s.
- Run manifest: `data/analysis/polymarket/BTC/btc_preopen_taker_flow_20261018/run_manifest.json`; source data audit: `data/analysis/polymarket/BTC/btc_preopen_taker_flow_20261018/binance_source_validation.json`; quality and economics: `data/analysis/polymarket/BTC/btc_preopen_taker_flow_20261018/quality_and_economics.json`.
- CV models and OOF predictions are saved under `data/analysis/polymarket/BTC/btc_preopen_taker_flow_20261018`. The experiment uses no holdout outcomes or deployment refit.
- The comparison is conditional on the source run's already selected feature set and hyperparameters; it is a controlled feature addition within that frozen pipeline.
