# BTC pre-open return regression experiment — 2026-10-17

## Result

One Huber return-regression challenger was fitted on the frozen 64-feature dataset. The development and historically exposed holdout scores are conditional on the already selected generators, weights, feature order, and regression hyperparameters.

- Main development comparison (proxy, same causal calibrated sample): regression log-loss `0.6922059499906476` vs binary Platt `0.6922129045696145`; paired regression-minus-binary difference `-6.9545789668237425e-06` with 95% block interval `[-7.064349458003576e-05, 5.423921645339799e-05]`.
- Holdout proxy comparison: regression `0.6924641894730769` vs binary `0.6925435821900641` log-loss on 28,224 common rows.
- Holdout official comparison: regression `0.6924642510924999` vs binary `0.6925030351573465` log-loss on 28,220 outcomes released by the frozen cutoff.

## Frozen target and data

For each 12:00–12:05 market, the decision remains 11:59:00 UTC (T−60), the prediction deadline remains 11:59:01 (T−59), and `target_preopen_5m_return_binance_index_proxy` is `Close[12:04] / Open[12:00] − 1`, available at 12:05. The return was reconstructed from cached minute OHLC using the contract window; cached binary labels verify the direction and tie rule.

At row `Opened`, sigma uses only close-to-close returns from adjacent one-minute candles ending at that row’s close, available at `Opened + 1 minute`, which equals the decision time:

```text
r = target_preopen_5m_return_binance_index_proxy
sigma_5m = sqrt(5) * population_std(last 60 valid one-minute simple returns; ddof=0)
z = r / max(sigma_5m, 0.000001)
```

The 60-return window is fixed. A timestamp gap or missing/nonpositive close invalidates affected windows until 60 contiguous returns are present. Missing targets remain missing, zero sigma uses the specified positive denominator floor, and z is not clipped.

- Binary-eligible development rows: 3,186,140; regression-eligible rows: 3,186,080; excluded for unavailable return or sigma: 60.
- Development return range: `-0.09755207949260503` to `0.08928108646143484`; z range: `-58.94429110552288` to `75.1888359097728`.
- Cached source-minute gaps: 0; missing sigma training rows: 60; zero-volatility training rows: 0.
- Exact 64-feature order, selected VP/RP settings, sample weights, and 10 saved fold fingerprints are recorded in `data/analysis/polymarket/BTC/btc_preopen_return_regression_20261017/frozen_manifest.json`.

## Development probability comparison

| Target | Model | Rows | Log-loss | Brier | AUC | ECE (10 bins) |
|---|---|---:|---:|---:|---:|---:|
| official | return_regression_causal_logistic | 21,951 | 0.692272 | 0.249563 | 0.5249 | 0.004288 |
| official | binary_causal_platt | 21,951 | 0.692279 | 0.249567 | 0.5239 | 0.004577 |
| official | binary_raw | 21,951 | 0.692227 | 0.249541 | 0.5239 | 0.003041 |
| official | constant_prior | 21,951 | 0.693163 | 0.250008 | 0.5000 | 0.002904 |
| proxy | return_regression_causal_logistic | 286,752 | 0.692206 | 0.249530 | 0.5246 | 0.002675 |
| proxy | binary_causal_platt | 286,752 | 0.692213 | 0.249533 | 0.5242 | 0.002730 |
| proxy | binary_raw | 286,752 | 0.692214 | 0.249534 | 0.5241 | 0.003477 |
| proxy | constant_prior | 286,752 | 0.693151 | 0.250002 | 0.4995 | 0.001163 |

The first internal OOF fold is excluded from calibrated probability scoring because no earlier OOF predictions exist to fit its calibrator. Later fold calibrators use only prior OOF predictions whose proxy labels were available by the first decision in that fold. The constant-probability comparator uses the same earlier available labels. Official Polymarket outcomes are scored separately; they did not fit either calibrator.

Regression OOF diagnostics on 318,614 actual decision rows: Huber(alpha=0.9) `0.463554`, MAE `0.85431`, predicted-z standard deviation `0.052692`, Pearson(predicted z, actual z) `0.01266`, and Pearson(predicted z, actual return) `0.01498`.

Fold metrics, calibration bins, predicted-z quantiles, and actual return/UP frequency by predicted-z decile are in `development_results.json`.

## Frozen single holdout evaluation

The fit, parameters, and final development OOF calibrator were frozen at `2026-10-17T00:41:35.341044+00:00` before scoring. Holdout was historically exposed in earlier repository work and is descriptive. Cutoff: `2026-10-07T00:00:00Z`. No development or holdout result changed the model or policy.

| Target | Forecast | Rows | Log-loss | Brier | AUC | ECE (10 bins) |
|---|---|---:|---:|---:|---:|---:|
| official | return_regression direction_probability | 28,220 | 0.692464 | 0.249659 | 0.5217 | 0.002598 |
| official | binary raw | 28,220 | 0.692527 | 0.249690 | 0.5217 | 0.001496 |
| official | binary calibrated | 28,220 | 0.692503 | 0.249678 | 0.5217 | 0.002796 |
| official | A_frozen raw | 28,220 | 0.692425 | 0.249639 | 0.5220 | 0.002532 |
| official | A_frozen calibrated | 28,220 | 0.692466 | 0.249659 | 0.5220 | 0.004245 |
| proxy | return_regression direction_probability | 28,224 | 0.692464 | 0.249659 | 0.5215 | 0.003767 |
| proxy | binary raw | 28,224 | 0.692565 | 0.249709 | 0.5212 | 0.002665 |
| proxy | binary calibrated | 28,224 | 0.692544 | 0.249698 | 0.5212 | 0.003965 |
| proxy | A_frozen raw | 28,224 | 0.692477 | 0.249665 | 0.5211 | 0.003950 |
| proxy | A_frozen calibrated | 28,224 | 0.692521 | 0.249687 | 0.5211 | 0.006017 |

## Economics and cost

Both forecasts use the existing 1% minimum net-EV rule, T−60 decision and T−59 snapshots. Independent orders request $5 each; the separate portfolio starts at $100, stakes 5% of free cash, and does not borrow. No archived order-book replay was run.

| Forecast | Independent $5 signals | Filled / settled | Net PnL | 5%-free-cash portfolio trades | Ending cash | PnL | Minimum-order skips |
|---|---:|---:|---:|---:|---:|---:|---:|
| binary | 3,005 | 2,798 / 2,798 | $-261.48 | 2 | $99.59 | $-0.41 | 23,104 |
| return_regression | 3,110 | 2,895 / 2,895 | $-249.25 | 2 | $99.40 | $-0.60 | 23,104 |

Decision skip reasons, execution fills, minimum-order diagnostics, accounting audits, and monthly portfolio rows are in `holdout_economics.json` and its per-model artifacts.

- Regression tuning: 37 study records (35 complete, 2 failed, 0 pruned), 2700.6 s; best criterion `0.48273402` = mean Huber `0.46352841` + 0.5 × fold SD `0.038411234`; selected iteration 95. Failed/partial trials did not enter selection.
- Trial exclusions: trial 0: stale RUNNING seed trial from the earlier initialization attempt was marked failed; no score was produced; trial 36: wall-clock deadline expired during a CV fold; partial trial was excluded from selection.
- Regression final fit: 10.73 s; 10-fold development OOF fits: 74.12 s; warm holdout inference is benchmarked in `costs.json`.
- Pre-run profile: one full CV fold took 6.25 s; the linear 60-trial, 10-fold projection with 25% overhead was 23452.4 s. Actual tuning took 2700.6 s and stopped at the authorized 2,700-second cap. Peak process working set: `unavailable (the process counter returned no sample)`.
- The main binary model tuning took 1,343.5 s, its final fit 3.1 s, and its ten-fold OOF predictions 19.9 s in the completed 2026-10-16 run.
- Limits: calibration and feature/config selection are conditional on development; official development labels cover only the market-index period; the historical holdout is exposed; return regression and volatility definition are one tested variant only.

## Conclusion

On the primary development proxy comparison, regression log-loss was 0.692206 versus 0.692213; the paired interval spans zero, so the small point-estimate advantage is inconclusive. On the single holdout proxy assessment it was 0.692464 versus 0.692544. These results do not establish a material improvement or reject other continuous targets.

Reproduction from the repository root: `python run_btc_preopen_return_regression.py`.
Frozen manifest: `data/analysis/polymarket/BTC/btc_preopen_return_regression_20261017/frozen_manifest.json`. Development predictions: `data/analysis/polymarket/BTC/btc_preopen_return_regression_20261017/development_oof_predictions.parquet`.
