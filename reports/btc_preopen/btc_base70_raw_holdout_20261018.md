# RAW BASE70 vs BASE64 - frozen historical assessment

## Status

The frozen holdout assessment is complete for 2026-07-01 through 2026-10-06, with exclusive data cutoff `2026-10-07T00:00:00Z`. The original run was blocked because raw flow fields were missing for 2026-10-02 through 2026-10-06. Following the authorized data completion, all five daily archives and official CHECKSUM sidecars were verified. The original blocker remains in `stages.holdout_flow_preflight`; its resolution is recorded in `data_completion_history`.

This is a historical assessment on a previously exposed holdout. Downloading the historical archives later does not change candle information time. T-59 execution is an assumption one second after the T-60 decision; actual one-second inference latency was not measured.

## Frozen protocol and checkpoints

- Protocol SHA-256: `f894db142fe8c82f2a451d21c26e742ed5e3c7518146e7ddeb28b45f81dbb428`.
- Frozen period, features, raw probabilities, policy, and cutoff were preserved. No model refit was performed.
- BASE64: 64 features and 25 iterations. BASE70: 70 features and 31 iterations.
- Shared development data: 3,174,805 rows, 634,957 decision rows, latest label availability `2026-06-30T23:59:00Z`.

| Model | Features | Iterations | Original fit seconds | Model SHA-256 |
| --- | --- | --- | --- | --- |
| BASE64 | 64 | 25 | 4.482 | f297a56a0fc841da29c2c21d316ab08fe1f30566306da24a79257840bf7f227f |
| BASE70 | 70 | 31 | 4.171 | 004cdacc495552604d5d91ed7122c8eeaa45a3b887117315daaea428f0f0776b |

## Binance data completion

The five archives came from Binance's official USD-M daily BTCUSDT 1m public data directory, documented in the [Binance Public Data README](https://github.com/binance/binance-public-data/blob/master/README.md). The downloader verified each archive against its official CHECKSUM file. Both source URLs and SHA-256 digests, including sidecar file hashes, are recorded here and in the manifest.

| UTC day | Archive URL | Official archive SHA-256 | CHECKSUM file SHA-256 | Verified archive SHA-256 | Valid minutes |
| --- | --- | --- | --- | --- | --- |
| 2026-10-02 | https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-10-02.zip | 9a831b07b2ce50e54e99490c3549b2e426955bc2608c78c31d3761d989e1cf28 | f7383c952f7ff41b67faab675971f6b9293f246d252cf22a62cf5feeecbacffd | 9a831b07b2ce50e54e99490c3549b2e426955bc2608c78c31d3761d989e1cf28 | 1440 |
| 2026-10-03 | https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-10-03.zip | 7cc6d272810c07d24c4d2dd35efa53b979198f8bcd918cbb7b5fec3e0cd31be2 | 09cae7d9efbe49c53a680339b1114858b3dc7305ba3b268954cb235aafb5dd41 | 7cc6d272810c07d24c4d2dd35efa53b979198f8bcd918cbb7b5fec3e0cd31be2 | 1440 |
| 2026-10-04 | https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-10-04.zip | e20f611db97170301b316de8e71a56cd8a3771e3f3c272b0ab2532a99ad1dafd | 931b3b7604d3a326d0664028b589ee9e4507100ae814e5b085aeadfdd6d6f29d | e20f611db97170301b316de8e71a56cd8a3771e3f3c272b0ab2532a99ad1dafd | 1440 |
| 2026-10-05 | https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-10-05.zip | 701e9dfaafc643d6d2c98bf76b5c7e39fc9adcc631928fb96d282bd42bb8648c | 88772c6af5b3ffdbf16b4405550ebf219ae660bac12a998b33ebac3bf6421279 | 701e9dfaafc643d6d2c98bf76b5c7e39fc9adcc631928fb96d282bd42bb8648c | 1440 |
| 2026-10-06 | https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-10-06.zip | 05cd282da17466b5e032c5a3bef179d1816ff2e13fa1ebd98c75e0188a141360 | 02519319272410a4af85d09d7f2072d10f19c13a3f9720b1a1dc312ed42aa502 | 05cd282da17466b5e032c5a3bef179d1816ff2e13fa1ebd98c75e0188a141360 | 1440 |

Validation confirmed USD-M BTCUSDT, 1m interval, millisecond Unix timestamps in UTC, candle close times, and a continuous 7,200-minute grid across the five days. Quote volume and taker-buy quote volume were present, non-negative, and checked using absolute tolerance `1e-10` and relative tolerance `1e-12`. There were no missing or duplicate minutes and no imputation or approximation.

| UTC day | Expected minutes | Valid candles | Missing/invalid | Duplicates | Volume violations outside tolerance |
| --- | --- | --- | --- | --- | --- |
| 2026-10-02 | 1440 | 1440 | 0 | 0 | 0 |
| 2026-10-03 | 1440 | 1440 | 0 | 0 | 0 |
| 2026-10-04 | 1440 | 1440 | 0 | 0 | 0 |
| 2026-10-05 | 1440 | 1440 | 0 | 0 | 0 |
| 2026-10-06 | 1440 | 1440 | 0 | 0 | 0 |

The continuous raw history begins `2026-06-01T00:00:00+00:00` and provides 43,198 warm-up minutes before the first holdout feature row. Recomputed Z/V finite masks exactly match the saved development cache over June 24-30; maximum numerical differences are `{"taker_Z_1m": {"rows": 10080, "finite_rows": 10080, "max_abs_difference": 2.0525803279269894e-11}, "taker_V_1m": {"rows": 10080, "finite_rows": 10080, "max_abs_difference": 4.440892098500626e-16}, "taker_Z_5m": {"rows": 10080, "finite_rows": 10080, "max_abs_difference": 1.4533263481553149e-11}, "taker_V_5m": {"rows": 10080, "finite_rows": 10080, "max_abs_difference": 8.004708007547379e-14}, "taker_Z_15m": {"rows": 10080, "finite_rows": 10080, "max_abs_difference": 1.9618084934336366e-11}, "taker_V_15m": {"rows": 10080, "finite_rows": 10080, "max_abs_difference": 4.08006961549745e-15}}`. The original feature windows, shifts, and T-60 information boundary are recorded as `{"horizons_minutes": [1, 5, 15], "qbar_window_minutes": 1440, "sigma_window_minutes": 60, "sigma_ddof": 0, "sigma_floor": 1e-06, "beta_window_minutes": 10080, "minimum_beta_pairs": 1440, "beta_variance_tolerance": 1e-12, "historical_beta_excludes_current_window": true}`.

## Prediction samples and quality

The shared complete-feature sample contains 28,224 of 28,224 candidate rows. Proxy labels available by cutoff: 28,224. Official labels available by cutoff: 28,220. BASE64, BASE70, and the constant `p=0.5` control use the same sample for each target.

| Target | Forecast | Rows | Log loss | Brier | AUC |
| --- | --- | --- | --- | --- | --- |
| Binance index proxy | BASE64 | 28224 | 0.692666 | 0.249759 | 0.518380 |
| Binance index proxy | BASE70 | 28224 | 0.692511 | 0.249682 | 0.521491 |
| Binance index proxy | HALF | 28224 | 0.693147 | 0.250000 | 0.500000 |
| Binance index proxy | BASE70 minus BASE64 | 28224 | -0.000155 | -0.000078 | 0.003111 |
| Official Polymarket | BASE64 | 28220 | 0.692640 | 0.249746 | 0.518702 |
| Official Polymarket | BASE70 | 28220 | 0.692521 | 0.249687 | 0.521450 |
| Official Polymarket | HALF | 28220 | 0.693147 | 0.250000 | 0.500000 |
| Official Polymarket | BASE70 minus BASE64 | 28220 | -0.000119 | -0.000060 | 0.002747 |

Paired bootstrap differences (point estimate and 95% interval) for both model comparisons and the `p=0.5` control:

| Target | Contrast | Metric | Point difference | 95% interval | Replicates |
| --- | --- | --- | --- | --- | --- |
| proxy | BASE70_MINUS_BASE64 | log_loss | -0.000155 | [-0.000315, -0.000006] | 2000 |
| proxy | BASE70_MINUS_BASE64 | brier | -0.000078 | [-0.000157, -0.000003] | 2000 |
| proxy | BASE70_MINUS_BASE64 | auc | 0.003111 | [0.001211, 0.005190] | 2000 |
| proxy | BASE64_MINUS_HALF | log_loss | -0.000481 | [-0.000976, 0.000040] | 2000 |
| proxy | BASE64_MINUS_HALF | brier | -0.000241 | [-0.000488, 0.000020] | 2000 |
| proxy | BASE64_MINUS_HALF | auc | 0.018380 | [0.011218, 0.025440] | 2000 |
| proxy | BASE70_MINUS_HALF | log_loss | -0.000636 | [-0.001187, -0.000083] | 2000 |
| proxy | BASE70_MINUS_HALF | brier | -0.000318 | [-0.000593, -0.000042] | 2000 |
| proxy | BASE70_MINUS_HALF | auc | 0.021491 | [0.014847, 0.028295] | 2000 |
| official | BASE70_MINUS_BASE64 | log_loss | -0.000119 | [-0.000284, 0.000035] | 2000 |
| official | BASE70_MINUS_BASE64 | brier | -0.000060 | [-0.000142, 0.000017] | 2000 |
| official | BASE70_MINUS_BASE64 | auc | 0.002747 | [0.000875, 0.004715] | 2000 |
| official | BASE64_MINUS_HALF | log_loss | -0.000508 | [-0.001011, -0.000009] | 2000 |
| official | BASE64_MINUS_HALF | brier | -0.000254 | [-0.000505, -0.000005] | 2000 |
| official | BASE64_MINUS_HALF | auc | 0.018702 | [0.011987, 0.025398] | 2000 |
| official | BASE70_MINUS_HALF | log_loss | -0.000627 | [-0.001192, -0.000075] | 2000 |
| official | BASE70_MINUS_HALF | brier | -0.000313 | [-0.000596, -0.000038] | 2000 |
| official | BASE70_MINUS_HALF | auc | 0.021450 | [0.014938, 0.028065] | 2000 |

The bootstrap used 2,000 shared circular 3-day UTC block replicates with seed `20261018`.

Proxy and official labels agreed on 26,007 of 28,220 jointly available rows (92.16%). The confusion matrix is in the manifest. Monthly scores are separate; October is incomplete.

| Target | UTC month | Rows | LL 64 | LL 70 | LL .5 | Brier 64 | Brier 70 | Brier .5 | AUC 64 | AUC 70 | AUC .5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binance_index_proxy | 2026-07 | 8928 | 0.693267 | 0.693206 | 0.693147 | 0.250059 | 0.250028 | 0.250000 | 0.510821 | 0.514529 | 0.500000 |
| binance_index_proxy | 2026-08 | 8928 | 0.692386 | 0.692166 | 0.693147 | 0.249620 | 0.249510 | 0.250000 | 0.520720 | 0.524301 | 0.500000 |
| binance_index_proxy | 2026-09 | 8640 | 0.692175 | 0.691929 | 0.693147 | 0.249514 | 0.249391 | 0.250000 | 0.526780 | 0.529671 | 0.500000 |
| binance_index_proxy | 2026-10 | 1728 | 0.693463 | 0.693610 | 0.693147 | 0.250158 | 0.250231 | 0.250000 | 0.507310 | 0.507359 | 0.500000 |
| official_polymarket | 2026-07 | 8927 | 0.693438 | 0.693462 | 0.693147 | 0.250145 | 0.250156 | 0.250000 | 0.509119 | 0.511831 | 0.500000 |
| official_polymarket | 2026-08 | 8927 | 0.692257 | 0.691967 | 0.693147 | 0.249555 | 0.249411 | 0.250000 | 0.522864 | 0.527269 | 0.500000 |
| official_polymarket | 2026-09 | 8639 | 0.692113 | 0.691987 | 0.693147 | 0.249483 | 0.249420 | 0.250000 | 0.526683 | 0.528260 | 0.500000 |
| official_polymarket | 2026-10 | 1727 | 0.693123 | 0.693180 | 0.693147 | 0.249988 | 0.250016 | 0.250000 | 0.510922 | 0.512568 | 0.500000 |

## Independent $5 orders

The existing T-60/T-59 snapshots and ledger engine were used with a 1% net-EV threshold, T-60 decisions, and assumed T-59 execution. Orders were independent; no portfolio capital limit was applied.

| Model | Decisions | Intents | Fills | Settled | Unresolved | Gross turnover | Fees | Net PnL | Forecast EV on settled fills | PnL minus EV | PnL / settled turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE64 | 28224 | 3086 | 2880 | 2879 | 1 | $14400.00 | $531.20 | -$529.30 | $1099.74 | -$1629.04 | -3.677% |
| BASE70 | 28224 | 4003 | 3738 | 3737 | 1 | $18690.00 | $677.55 | -$546.22 | $1242.61 | -$1788.83 | -2.923% |

BASE70 minus BASE64 economic bootstrap: mean daily net PnL difference `{"point_estimate": -0.17265743815920157, "estimate_median": -0.11640485957451727, "95pct": [-4.19183167454023, 3.8954522753988012], "valid_replications": 2000}`; difference in PnL / settled turnover `{"estimate_median": 0.007517577927153118, "95pct": [-0.017728303712884422, 0.032445671620813715], "valid_replications": 2000}`. Monthly results:

| Model | UTC month | Intents | Fills | Settled | Unresolved | Gross turnover | UP fills | DOWN fills | UP net PnL | DOWN net PnL | Fees | Net PnL | PnL / turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE64 | 2026-07 | 972 | 954 | 954 | 0 | $4770.00 | 460 | 494 | -$102.04 | -$446.69 | $175.52 | -$548.73 | -11.504% |
| BASE64 | 2026-08 | 959 | 890 | 890 | 0 | $4450.00 | 438 | 452 | $48.35 | -$66.68 | $167.78 | -$18.33 | -0.412% |
| BASE64 | 2026-09 | 909 | 791 | 791 | 0 | $3955.00 | 358 | 433 | $18.53 | $141.31 | $144.41 | $159.84 | 4.041% |
| BASE64 | 2026-10 | 246 | 245 | 244 | 1 | $1225.00 | 121 | 124 | $36.85 | -$158.93 | $43.50 | -$122.08 | -10.006% |
| BASE70 | 2026-07 | 1273 | 1244 | 1244 | 0 | $6220.00 | 606 | 638 | -$58.74 | -$450.72 | $225.10 | -$509.47 | -8.191% |
| BASE70 | 2026-08 | 1202 | 1117 | 1117 | 0 | $5585.00 | 553 | 564 | $80.33 | -$104.68 | $206.64 | -$24.35 | -0.436% |
| BASE70 | 2026-09 | 1203 | 1053 | 1053 | 0 | $5265.00 | 456 | 597 | -$22.19 | $159.92 | $188.76 | $137.74 | 2.616% |
| BASE70 | 2026-10 | 325 | 324 | 323 | 1 | $1620.00 | 160 | 164 | -$29.05 | -$121.09 | $57.05 | -$150.14 | -9.296% |

Daily PnL concentration (top five profitable days, remaining days, and total):

| Model | Top five days | Top five PnL | Remaining days PnL | Total PnL |
| --- | --- | --- | --- | --- |
| BASE64 | 2026-08-09, 2026-08-20, 2026-09-10, 2026-09-03, 2026-09-30 | $276.56 | -$805.86 | -$529.30 |
| BASE70 | 2026-08-09, 2026-09-10, 2026-07-08, 2026-09-12, 2026-09-26 | $290.36 | -$836.58 | -$546.22 |

BASE70 minus BASE64 realized PnL decomposition:

| Settled fill match category | Market count | PnL difference |
| --- | --- | --- |
| same_side | 2510 | $0.00 |
| only_one_variant | 1596 | -$16.92 |
| opposite_sides | 0 | $0.00 |

On the 2,510 resolved market/side fills shared by both independent policies, total forecast EV was $1069.41 for BASE64 and $1128.24 for BASE70 (BASE70 minus BASE64: $58.83).

Accounting audits:

| Model | Audit status | Errors |
| --- | --- | --- |
| BASE64 | passed | [] |
| BASE70 | passed | [] |

The accounting audit covers balances, fees, turnover, PnL, settlements, and independent-order mode. Its one-second timing check applies to the assumed T-59 timestamp and does not measure actual inference latency.

## Run history and outputs

- Original blocker: `stages.holdout_flow_preflight.status = blocked`; its reason and missing paths are retained.
- Data completion: `stages.holdout_data_completion`; only the five authorized daily archives and CHECKSUM sidecars were downloaded.
- Both model checkpoints were reused. Predictions, samples, bootstrap outputs, execution calendar, ledgers, and audits are checkpointed, so report-generation errors do not refit or rescore.
- No Polymarket archive replay, other dataset downloads, or post-holdout tuning were performed.
- Report: `reports/btc_preopen/btc_base70_raw_holdout_20261018.md`. Versioned manifest: `reports/btc_preopen/btc_base70_raw_holdout_20261018_manifest.json`. Runtime outputs: `data/analysis/polymarket/BTC/btc_base70_raw_holdout_20261018/`.
