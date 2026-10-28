# E0039 BTC chronology reconstruction run

- Batch run manifest status: `completed_folds_economics_runtime_partial`
- E0039 status after the follow-on runtime benchmark: batch folds/economics and offline runtime parity complete; T0009 source/readiness remains blocked.
- Run identity: `508858c48b11b9252e4ec3aa0e7814579f2fe770ef8840e32fd86dca6c6bd8d0`
- Protocol: `configs/research/btc_chronology_reconstruction_20261027_v7.json` (`9c6aca98034ca010cda16e326053ce83e2c78dbab0120445be8ec1180392baf6`)
- Parent protocol: `configs/research/btc_chronology_reconstruction_20261027_v6.json` (`2facc83e0b68ae0dc26874ec5dfd85b70559d1fa17a2ec4156744099bd580d3c`)
- Runtime seconds: `28354.1491446`
- Protected holdout rows materialized: `0`
- Follow-on offline runtime benchmark: `completed_streaming_runtime_benchmark` (`300/300` windows; detailed below). The original batch manifest status remains unchanged.

## Fold evidence

### F1_20260415_20260501

- Status: `chronological_fit_completed`
- Selection cutoff: `2025-07-01T00:00:00Z`
- Outer fit cutoff: `2026-04-15T00:00:00Z`
- Selected base features: `64`
- OOF rows: `{'A': 82934, 'B': 82934, 'C': 82934, 'D': 82934}`
- Outer raw rows: `{'A': 4608, 'B': 4608, 'C': 4608, 'D': 4608}`
- Calibrated rows: `{'A': 0, 'B': 0, 'C': 0, 'D': 0}`
- Fitting resource sample: `{"detail": null, "elapsed_seconds": 324.669171, "finished_utc": "2026-10-27T19:54:31.982093+00:00", "runner_peak_working_set_gb_sampled": 9.509, "sampling_interval_seconds": 2.0, "status": "completed", "system_memory_includes_unrelated_processes": true, "system_minimum_available_memory_gb_sampled": 40.591}`
- Fold resource sample: `{"detail": null, "elapsed_seconds": 8687.557046, "finished_utc": "2026-10-27T19:54:31.983505+00:00", "runner_peak_working_set_gb_sampled": 45.067, "sampling_interval_seconds": 2.0, "status": "completed", "system_memory_includes_unrelated_processes": true, "system_minimum_available_memory_gb_sampled": 4.116}`

### F2_20260501_20260601

- Status: `chronological_fit_completed`
- Selection cutoff: `2025-08-01T00:00:00Z`
- Outer fit cutoff: `2026-05-01T00:00:00Z`
- Selected base features: `64`
- OOF rows: `{'A': 78606, 'B': 78606, 'C': 78606, 'D': 78606}`
- Outer raw rows: `{'A': 8928, 'B': 8928, 'C': 8928, 'D': 8928}`
- Calibrated rows: `{'A': 8928, 'B': 8928, 'C': 8928, 'D': 8928}`
- Fitting resource sample: `{"detail": null, "elapsed_seconds": 349.62869, "finished_utc": "2026-10-27T22:45:37.853138+00:00", "runner_peak_working_set_gb_sampled": 5.236, "sampling_interval_seconds": 2.0, "status": "completed", "system_memory_includes_unrelated_processes": true, "system_minimum_available_memory_gb_sampled": 42.695}`
- Fold resource sample: `{"detail": null, "elapsed_seconds": 10265.742942, "finished_utc": "2026-10-27T22:45:37.854201+00:00", "runner_peak_working_set_gb_sampled": 46.818, "sampling_interval_seconds": 2.0, "status": "completed", "system_memory_includes_unrelated_processes": true, "system_minimum_available_memory_gb_sampled": 0.752}`

### F3_20260601_20260701

- Status: `chronological_fit_completed`
- Selection cutoff: `2025-09-01T00:00:00Z`
- Outer fit cutoff: `2026-06-01T00:00:00Z`
- Selected base features: `64`
- OOF rows: `{'A': 78616, 'B': 78616, 'C': 78616, 'D': 78616}`
- Outer raw rows: `{'A': 8640, 'B': 8640, 'C': 8640, 'D': 8640}`
- Calibrated rows: `{'A': 8640, 'B': 8640, 'C': 8640, 'D': 8640}`
- Fitting resource sample: `{"detail": null, "elapsed_seconds": 322.490373, "finished_utc": "2026-10-28T01:21:29.159081+00:00", "runner_peak_working_set_gb_sampled": 5.648, "sampling_interval_seconds": 2.0, "status": "completed", "system_memory_includes_unrelated_processes": true, "system_minimum_available_memory_gb_sampled": 42.884}`
- Fold resource sample: `{"detail": null, "elapsed_seconds": 9351.072611, "finished_utc": "2026-10-28T01:21:29.161159+00:00", "runner_peak_working_set_gb_sampled": 41.984, "sampling_interval_seconds": 2.0, "status": "completed", "system_memory_includes_unrelated_processes": true, "system_minimum_available_memory_gb_sampled": 5.763}`

## Source-clean economics

- Frozen E0038 whitelist: `5604`
- Common fully calibrated markets: `5148`
- Coverage by model: `{"A": {"markets_in_outer_folds": 5604, "officially_calibrated_predictions": 5148, "raw_predictions": 5604, "source_clean_whitelist_markets": 5604, "status_counts": {"minimum_not_met": 456, "predicted_calibrated": 5148}}, "B": {"markets_in_outer_folds": 5604, "officially_calibrated_predictions": 5148, "raw_predictions": 5604, "source_clean_whitelist_markets": 5604, "status_counts": {"minimum_not_met": 456, "predicted_calibrated": 5148}}, "C": {"markets_in_outer_folds": 5604, "officially_calibrated_predictions": 5148, "raw_predictions": 5604, "source_clean_whitelist_markets": 5604, "status_counts": {"minimum_not_met": 456, "predicted_calibrated": 5148}}, "D": {"markets_in_outer_folds": 5604, "officially_calibrated_predictions": 5148, "raw_predictions": 5604, "source_clean_whitelist_markets": 5604, "status_counts": {"minimum_not_met": 456, "predicted_calibrated": 5148}}}`
- Paired independent-$5 differences on common calibrated coverage: `{"A-B": {"common_markets": 5148, "left_minus_right_pnl_usd": 50.851431798043436, "left_realized_pnl_usd": 139.945271869044, "markets_positive_zero_negative": {"negative": 369, "positive": 366, "zero": 4413}, "right_realized_pnl_usd": 89.09384007100057}, "A-C": {"common_markets": 5148, "left_minus_right_pnl_usd": -78.02451960799655, "left_realized_pnl_usd": 139.945271869044, "markets_positive_zero_negative": {"negative": 316, "positive": 296, "zero": 4536}, "right_realized_pnl_usd": 217.96979147704053}, "A-D": {"common_markets": 5148, "left_minus_right_pnl_usd": -134.629513324161, "left_realized_pnl_usd": 139.945271869044, "markets_positive_zero_negative": {"negative": 373, "positive": 336, "zero": 4439}, "right_realized_pnl_usd": 274.574785193205}, "B-C": {"common_markets": 5148, "left_minus_right_pnl_usd": -128.87595140603995, "left_realized_pnl_usd": 89.09384007100057, "markets_positive_zero_negative": {"negative": 359, "positive": 342, "zero": 4447}, "right_realized_pnl_usd": 217.96979147704053}, "B-D": {"common_markets": 5148, "left_minus_right_pnl_usd": -185.48094512220442, "left_realized_pnl_usd": 89.09384007100057, "markets_positive_zero_negative": {"negative": 285, "positive": 251, "zero": 4612}, "right_realized_pnl_usd": 274.574785193205}, "C-D": {"common_markets": 5148, "left_minus_right_pnl_usd": -56.604993716164444, "left_realized_pnl_usd": 217.96979147704053, "markets_positive_zero_negative": {"negative": 273, "positive": 256, "zero": 4619}, "right_realized_pnl_usd": 274.574785193205}}`
- Monthly and paired ledgers: `data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027/runs/508858c48b11b925/economics/monthly_paths.csv`, `data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027/runs/508858c48b11b925/economics/paired_monthly_differences.csv`
- Unknown coverage rows remain unknown and are not labelled `no_trade`.

## Offline runtime

- Status: `completed_streaming_runtime_benchmark`; 300/300 frozen windows passed.
- Benchmark elapsed time: `649.725 s`; runtime sample SHA-256: `a3f62ce585f17aa125e69c6494ab139425cfcfc2580007217914c98d8341b7b0`.
- Harness: `run_btc_chronology_runtime_benchmark_20261028.py`, SHA-256 `6c8d54c4820a8ba7b3ba8bb0db543c5118bd4d6b35a03f514b91e73c67735b64`.
- Local machine output: `data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027/runs/508858c48b11b925/runtime_benchmark/runtime_benchmark.json` (SHA-256 `be893f5f1a6f7fdd4f66dd786699da36bd9a9e60a88d9aa1309f133990d095df`) and `runtime_latencies.csv` (SHA-256 `b2cda4ba6485fc6f9e8e8c2d02d2da59a83beb31807fef3b689e78cd881bd037`). These run artifacts remain ignored under `data/analysis`.
- Inputs were the full fold-local raw prefix and locally archived Binance kline rows; batch feature, dynamic-feature, flow and model outputs were read only as parity references. No protected rows were materialized. Offline-only; no network or collector was used.
- Path timed at each sampled decision: one raw candle's BASE/VP/RP/flow/memory/ESN state update, selected BASE64 snapshot, all A/B/C/D bundle members, then the frozen UTC-day Platt calibration or the frozen no-fit fallback. Fold-local bundles and state were reinitialized for F1/F2/F3; F2 and F3 record the switch/reset from the prior fold.

### Startup and causal warmup

| Fold | Historical candles | Profile states (s) | BASE64 adapter (s) | Memory/ESN states (s) | Flow state (s) | Model load (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 3,076,320 | 1.217 | 0.480 | 75.240 | 0.495 | 0.241 |
| F2 | 3,098,345 | 0.796 | 0.466 | 75.077 | 0.460 | 0.287 |
| F3 | 3,142,945 | 3.125 | 0.507 | 76.415 | 0.481 | 0.323 |

Verified local Binance archive loading took `0.2 s` for the three source months. Memory/ESN bootstrap is a separate fold-start cost and is excluded from decision latency.

### Latency distributions

All values are milliseconds. `>1 s` is the percent of measured values over 1,000 ms. `Updates between sample decisions` is the summed cost of one-minute updates since the prior sample boundary; the first measurement in each fold contains the first sample's one-candle update. The total can exceed one second because it spans many candle updates.

| Fold | Path | Count | p50 | p95 | p99 | Max | >1 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | One-candle state update | 21,971 | 3.350 | 5.504 | 6.794 | 60.387 | 0.0% |
| F1 | BASE64 feature snapshot | 100 | 6.303 | 8.611 | 15.138 | 48.476 | 0.0% |
| F1 | Ensemble and calibration | 100 | 1.565 | 1.920 | 3.327 | 3.873 | 0.0% |
| F1 | Full current path | 100 | 11.299 | 16.745 | 26.389 | 112.736 | 0.0% |
| F1 | Updates between sample decisions | 100 | 580.494 | 1,859.656 | 3,374.846 | 4,947.543 | 30.0% |
| F2 | One-candle state update | 44,596 | 3.338 | 4.835 | 6.700 | 16.191 | 0.0% |
| F2 | BASE64 feature snapshot | 100 | 7.670 | 9.285 | 13.602 | 13.906 | 0.0% |
| F2 | Ensemble and calibration | 100 | 1.466 | 1.892 | 2.680 | 3.192 | 0.0% |
| F2 | Full current path | 100 | 12.620 | 15.556 | 20.132 | 22.633 | 0.0% |
| F2 | Updates between sample decisions | 100 | 503.450 | 5,805.564 | 9,102.064 | 9,453.529 | 32.0% |
| F3 | One-candle state update | 43,176 | 3.402 | 5.227 | 7.073 | 14.004 | 0.0% |
| F3 | BASE64 feature snapshot | 100 | 5.463 | 8.571 | 11.379 | 26.210 | 0.0% |
| F3 | Ensemble and calibration | 100 | 1.421 | 2.138 | 3.348 | 3.610 | 0.0% |
| F3 | Full current path | 100 | 10.431 | 14.182 | 20.827 | 32.824 | 0.0% |
| F3 | Updates between sample decisions | 100 | 588.693 | 708.161 | 2,327.896 | 98,907.131 | 2.0% |
| All | Full current path | 300 | 11.478 | 15.324 | 22.661 | 112.736 | 0.0% |

The largest adjacent sample gap in F3 is `27,205 minutes`; the `98.907 s` cumulative update total spans that unsampled interval. It is not the latency of one candle update or one current decision path.

### Batch/stream parity

- Feature tolerance: absolute `2e-6`. Maximum absolute difference: selected BASE64 `3.775e-9`, six Z/V `1.323e-11`, memory PCA8 `0`, ESN PCA8 `0`; finite-mask mismatches: zero.
- Raw ensemble probability tolerance: absolute `2e-6`; maximum difference over A/B/C/D and all 300 windows: `2.221e-16`.
- Calibrated probability maximum difference: `3.331e-16`. F1 used the frozen no-fit fallback for all sampled models; F2/F3 each matched 100 calibrated outputs per model.
- Runtime tests: focused `3/3` passed; full `python -m unittest discover -s tests` passed `736/736` in `46.215 s`; `py_compile` passed.

This measures offline local computation on the frozen windows. It does not establish historical T-59 input readiness, event/network delay, production operation, or a repeatable trading advantage. Historical T-59 readiness remains `UNKNOWN`.

## Interpretation limits

- This is a retrospective chronological reproduction conditional on the previously exposed A/B/C/D family search.
- Historical T-59 readiness and network delay remain `UNKNOWN`.
- E0037 remains historical `FAILED`; E0038 artifacts are unchanged.
- No holdout, confirmation, PMXT replay, source acquisition, shadow run, or order was performed.
