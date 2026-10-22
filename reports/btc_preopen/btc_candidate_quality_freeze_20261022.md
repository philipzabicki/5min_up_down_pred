# BTC candidate quality freeze and development comparison — 2026-10-22

> All scores below use the previously reused `search_validation` interval. They are development selection results, not independent confirmation.

## Frozen procedures

A is BASE70; B adds the fixed simple-memory PCA8 representation; C adds the focused-search ESN PCA8 candidate `8f74a7951e10c11830baba75`; D adds both frozen representations in the preregistered order. D was frozen in `configs/research/btc_candidate_quality_freeze_20261022.json` before fitting.

The common target is the Binance index proxy `Close[Opened+6m] >= Open[Opened+2m]`, available at Opened+7m. Nominal decisions are at Opened+1m, the T−60 point. LightGBM uses fit-core only, weights 0.59 on decision rows and 0.1025 otherwise, the unchanged one-minute label-availability purge, 31 boosting rounds, no early stopping, and seeds 20261101–20261103. C/D cross those seeds with reservoir seeds 20261023–20261025.

The complete ordered features, transforms, seed matrices, model files, source hashes, and fit provenance are in the local bundle manifests under `data/analysis/polymarket/BTC/btc_candidate_quality_freeze_20261022/bundles/`.

### Frozen representation parameters

- B memory inputs: `['log_return_1m', 'log_return_5m', 'log_return_15m', 'log_return_60m', 'candle_body_log_return', 'candle_close_location', 'candle_upper_wick_fraction', 'candle_lower_wick_fraction', 'log_volume', 'volume_ratio_to_prior_30m_median', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']`; lags `[1, 5, 15, 60, 240, 720]` minutes; EWM half-lives `[15, 60, 240, 720]` minutes; deviations `['current_scaled_input - EMA(60m)', 'current_scaled_input - EMA(720m)']`. Input scaling and PCA8 use 20,000 fit-core rows, randomized PCA seed `20261028`, and the exact scaler/PCA arrays in the copied transform bundle.
- C selected ESN configuration: `{"architecture": "single", "bias_scaling": 0.006897627342216597, "branch_count": 1, "component_bias_scalings": [0.006897627342216597], "component_input_groups": [["returns", "candle", "volume", "time"]], "component_input_scalings": [0.015346966516162231], "component_leak_rates": [0.001047464483131801], "component_sizes": [256], "component_sparse_degrees": [4], "component_spectral_radii": [0.6522888765744748], "component_topologies": ["sparse"], "components": [{"bias_scaling": 0.006897627342216597, "index": 0, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.015346966516162231, "leak_rate": 0.001047464483131801, "source_component": -1, "sparse_degree": 4, "spectral_radius": 0.6522888765744748, "topology": "sparse", "units": 256}], "connectivity": 0.15, "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.015346966516162231, "layer_count": 1, "layers": null, "leak_rate": 0.001047464483131801, "n_reservoirs": 1, "reservoir_size_per_layer_or_reservoir": 256, "selected_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "spectral_radius": 0.6522888765744748, "topology": "sparse", "total_units": 256}`; raw inputs `['log_return_1m', 'log_return_5m', 'log_return_15m', 'log_return_60m', 'candle_body_log_return', 'candle_close_location', 'candle_upper_wick_fraction', 'candle_lower_wick_fraction', 'log_volume', 'volume_ratio_to_prior_30m_median', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']`. Reservoir-specific matrices, input scaler, PCA8, warm-up counts, and projection seeds are frozen for seeds `[20261023, 20261024, 20261025]` in the C manifest.
- Every procedure manifest stores the exact BASE70 order and one source-backed definition/source hash per column, plus complete model parameters per LightGBM seed.

## Common observations and metrics

The comparison uses 52,120 identical decision timestamps and labels. Decision SHA-256: `b8134c4abd481b29f2fe8d45b50c75a2020b4bdd761b7a5c02e1af356e1ac690`; target SHA-256: `a333332b36c3f173f83886003274d0ea48a11a27a07c8ae71c72f2d1989e4d0e`. Feature readiness and label availability were intersected once before scoring.

Mean individual-model metrics are kept separate from metrics on the uniform ensemble prediction.

| Procedure | Fits | Mean individual LL | Mean individual Brier | Mean individual AUC | Uniform ensemble LL | Uniform ensemble Brier | Uniform ensemble AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 3 | 0.692412060 | 0.249632746 | 0.524030 | 0.692375219 | 0.249614437 | 0.524217 |
| B | 3 | 0.692248648 | 0.249551260 | 0.525291 | 0.692210070 | 0.249532059 | 0.525545 |
| C | 9 | 0.692313645 | 0.249583671 | 0.524363 | 0.692255246 | 0.249554627 | 0.524743 |
| D | 9 | 0.692191292 | 0.249522749 | 0.525370 | 0.692135666 | 0.249495066 | 0.525778 |

Positive LL/Brier deltas and positive AUC deltas favor the procedure on the right.

| Pair | ΔLL | Paired 95% block CI | ΔBrier | Paired 95% block CI | ΔAUC | Paired 95% block CI |
|---|---:|---|---:|---|---:|---|
| B-A | +0.000165149 | [+0.000047547, +0.000271143] | +0.000082378 | [+0.000023966, +0.000135137] | +0.001328 | [+0.000295773, +0.002263948] |
| C-A | +0.000119973 | [+0.000021977, +0.000216514] | +0.000059810 | [+0.000011125, +0.000107774] | +0.000527 | [-0.000308956, +0.001324021] |
| C-B | -0.000045176 | [-0.000159401, +0.000072658] | -0.000022568 | [-0.000079193, +0.000036088] | -0.000801 | [-0.001732429, +0.000197990] |
| D-B | +0.000074403 | [-0.000003351, +0.000156894] | +0.000036993 | [-0.000001878, +0.000077983] | +0.000233 | [-0.000511901, +0.000947986] |
| D-C | +0.000119580 | [+0.000028704, +0.000209767] | +0.000059561 | [+0.000014062, +0.000104315] | +0.001034 | [+0.000178067, +0.001892316] |
| D-A | +0.000239552 | [+0.000107102, +0.000372376] | +0.000119371 | [+0.000053263, +0.000185519] | +0.001561 | [+0.000399484, +0.002697431] |

The paired intervals use 2,000 shared non-circular moving calendar-day block draws via `utils.btc_oof_audit.moving_calendar_block_multiplicity`, block length 3 calendar days, seed 20261023. They describe variability within this reused interval and do not correct the earlier search-selection effect.

### Matched LightGBM seed comparisons

For C−B and D−B, the three ESN-seed predictions are averaged within each LightGBM seed; B appears once per LightGBM seed.

| LightGBM seed | A LL | B LL | C LL (mean reservoir prediction) | D LL (mean reservoir prediction) |
|---:|---:|---:|---:|---:|
| 20261101 | 0.692455191 | 0.692204449 | 0.692259076 | 0.692187656 |
| 20261102 | 0.692405381 | 0.692287978 | 0.692285020 | 0.692145481 |
| 20261103 | 0.692375609 | 0.692253518 | 0.692277476 | 0.692133202 |

### Individual fitted-model metrics

| Procedure | Reservoir seed | LightGBM seed | LL | Brier | AUC |
|---|---:|---:|---:|---:|---:|
| A | — | 20261101 | 0.692455191 | 0.249654282 | 0.523662 |
| A | — | 20261102 | 0.692405381 | 0.249629403 | 0.523987 |
| A | — | 20261103 | 0.692375609 | 0.249614554 | 0.524442 |
| B | — | 20261101 | 0.692204449 | 0.249529110 | 0.525835 |
| B | — | 20261102 | 0.692287978 | 0.249570897 | 0.524808 |
| B | — | 20261103 | 0.692253518 | 0.249553772 | 0.525229 |
| C | 20261023 | 20261101 | 0.692281312 | 0.249567487 | 0.524697 |
| C | 20261024 | 20261101 | 0.692313719 | 0.249583857 | 0.523992 |
| C | 20261025 | 20261101 | 0.692307823 | 0.249580811 | 0.524191 |
| C | 20261023 | 20261102 | 0.692324890 | 0.249589331 | 0.524334 |
| C | 20261024 | 20261102 | 0.692336314 | 0.249594932 | 0.524510 |
| C | 20261025 | 20261102 | 0.692311535 | 0.249582705 | 0.524254 |
| C | 20261023 | 20261103 | 0.692280065 | 0.249566896 | 0.524731 |
| C | 20261024 | 20261103 | 0.692328925 | 0.249591182 | 0.524285 |
| C | 20261025 | 20261103 | 0.692338223 | 0.249595833 | 0.524272 |
| D | 20261023 | 20261101 | 0.692179910 | 0.249517083 | 0.525516 |
| D | 20261023 | 20261102 | 0.692148657 | 0.249501566 | 0.525655 |
| D | 20261023 | 20261103 | 0.692119690 | 0.249487144 | 0.525945 |
| D | 20261024 | 20261101 | 0.692239148 | 0.249546596 | 0.525027 |
| D | 20261024 | 20261102 | 0.692201720 | 0.249527845 | 0.525429 |
| D | 20261024 | 20261103 | 0.692191916 | 0.249523074 | 0.525332 |
| D | 20261025 | 20261101 | 0.692248882 | 0.249551307 | 0.525033 |
| D | 20261025 | 20261102 | 0.692191733 | 0.249523014 | 0.525071 |
| D | 20261025 | 20261103 | 0.692199972 | 0.249527116 | 0.525318 |

## ESN incremental-information decision

D's point estimate is positive beyond B, but the paired interval includes zero; this development run does not establish incremental information beyond B.

## Comparability audit

- Exact common row IDs and labels: **True** (52,120 rows; identical row and label hashes).
- Same input cutoff and BASE70 matrix: **True**; T−60 inputs only, no label available at or after the search cutoff.
- Common fit parameters, weights, seeds, and purge: **True**; all procedure models use the same fit-row hash.
- Scaler/PCA fit provenance: **True**; memory scaler/PCA and ESN scaler/PCA are fit-core-only artifacts.
- Fit-core scaler reconstruction matches the frozen memory/ESN scalers: **True**; memory PCA sample rows: 20,000; ESN PCA sample rows per seed: [20261023, 20261024, 20261025] × 20,000.
- Maximum A/B/C saved-probability replay error: `0.0382369757`; maximum model save/load replay error: `0`.
- Three orchestration fixes: the auxiliary matrix is reconstructed for every seed fit; monthly pair-trigger comparisons read each `delta_logloss` field; latency output maps candidate aliases to saved bundle keys. The first can affect auxiliary PCA4 and simple-memory fits, the second only the optional pair trigger, and the third only latency lookup. No conditional pair was part of this A/B/C comparison; the old PCA4 result remains a historical artifact outside it. Historical probability files do not carry a per-artifact runner hash, so their exact producing orchestration revision cannot be established from run identity alone. Each procedure uses its old saved probabilities only when all fit-core model replays match within the preregistered tolerance; otherwise all members of that procedure use the fit-core-only refit probabilities. The bundle records the exact current runner/source hashes used for its model exports.
- Scoring probability sources under the frozen replay rule: `{'A': 'fit_core_common_mask_refit', 'B': 'existing_saved_prediction_artifacts', 'C': 'fit_core_common_mask_refit', 'D': 'fit_core_common_mask_refit'}`.

## Bundle replay and streaming checks

Fresh-process checks: `{"batch_stream_esn_equal": true, "batch_stream_esn_projection_equal": true, "batch_stream_memory_equal": true, "batch_stream_memory_projection_equal": true, "esn_checkpoint_resume_equal": true, "feature_order": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "fresh_process": true, "full_search_prefix_bundle_replay": {"all_saved_representation_rows_reproduced": true, "esn_by_reservoir_seed": {"20261023": {"max_abs_error": 0.0, "rows": 2642867, "warmup_steps": 17305}, "20261024": {"max_abs_error": 0.0, "rows": 2642745, "warmup_steps": 17427}, "20261025": {"max_abs_error": 0.0, "rows": 2641786, "warmup_steps": 18386}}, "memory": {"max_abs_error": 0.0, "rows": 2649226}, "raw_stream_rows": 2660546, "scope_end_exclusive_utc": "2025-07-01T00:00:00Z"}, "future_perturbation_memory_prefix_invariant": true, "future_perturbation_prefix_invariant": true, "memory_checkpoint_resume_equal": true, "stream_rows": 128, "valid_reset_case_included": true}`.
Fit-core model export time (A/B/C refits for bundle export plus D): `76.36s`; D-only fit time: `29.10s`.
Fresh full search-prefix bundle-to-feature replay time: `58.03s`.
Main runner active time: `156.74s`; frozen task cap: `14400s`.
Source-recorded full-stream representation generation: B `6.59s`; C `64.72s`; both `71.30s`. Validation batch prediction time for 52,120 rows: `{'A': 0.11244270001770929, 'B': 0.0945138999959454, 'C': 0.3102510000462644, 'D': 0.31368570000631735}`.
Measured end-to-end feature-generation-plus-prediction totals from the existing BASE70 and raw 14-column arrays: `{'A_BASE70': 0.11244270001770929, 'B_BASE70_plus_memory': 6.681185400113463, 'C_BASE70_plus_ESN': 65.02725720009767, 'D_BASE70_plus_memory_plus_ESN': 71.61736340017524}`.
Process peak working set: `3102990336` bytes. Resource snapshot before fit: `{'cpu_logical_processors': 20, 'cpu_model': 'Intel64 Family 6 Model 183 Stepping 1, GenuineIntel', 'platform': 'Windows-11-10.0.26300-SP0', 'physical_memory_bytes': 68366831616, 'available_memory_bytes': 52515291136, 'disk_free_bytes': 324298723328, 'gpu_detected': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'reservoir_gpu_used': False, 'workers': 1, 'native_threads': 4}`; after replay: `{'cpu_logical_processors': 20, 'cpu_model': 'Intel64 Family 6 Model 183 Stepping 1, GenuineIntel', 'platform': 'Windows-11-10.0.26300-SP0', 'physical_memory_bytes': 68366831616, 'available_memory_bytes': 51310452736, 'disk_free_bytes': 324298596352, 'gpu_detected': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'reservoir_gpu_used': False, 'workers': 1, 'native_threads': 4}`.
Per-decision ensemble inference latency (256 rows): `{'A': {'samples': 256, 'p50_us': 67.0, 'p95_us': 96.075, 'p99_us': 159.5849999999995}, 'B': {'samples': 256, 'p50_us': 72.4, 'p95_us': 109.525, 'p99_us': 143.5749999999999}, 'C': {'samples': 256, 'p50_us': 212.8, 'p95_us': 322.65000000000003, 'p99_us': 427.6699999999993}, 'D': {'samples': 256, 'p50_us': 222.0, 'p95_us': 329.525, 'p99_us': 432.1549999999998}}`. Cold-start and normal state-update/projection latency: `{'measurement_scope': 'causal 14-column development feature stream through memory/ESN state and PCA; excludes upstream BASE70 raw-market feature construction', 'pre_validation_contiguous_development_rows': 2399907, 'memory': {'cold_start_rows': 978, 'warmup_rows_to_first_feature': 721, 'cold_start_seconds_after_runtime_initialization': 0.0014868999714963138, 'features_emitted_during_cold_start': 258, 'normal_update_and_projection_us': {'samples': 257, 'p50': 177.3, 'p95': 204.02, 'p99': 232.356}}, 'esn_by_reservoir_seed': {'20261023': {'cold_start_rows': 17562, 'warmup_rows_to_first_feature': 17306, 'cold_start_seconds_after_runtime_initialization': 0.07801989995641634, 'features_emitted_during_cold_start': 257, 'normal_update_and_projection_us': {'samples': 256, 'p50': 15.0, 'p95': 17.825, 'p99': 22.35499999999998}}, '20261024': {'cold_start_rows': 17684, 'warmup_rows_to_first_feature': 17428, 'cold_start_seconds_after_runtime_initialization': 0.07842480001272634, 'features_emitted_during_cold_start': 257, 'normal_update_and_projection_us': {'samples': 256, 'p50': 14.7, 'p95': 16.55, 'p99': 22.33499999999993}}, '20261025': {'cold_start_rows': 18643, 'warmup_rows_to_first_feature': 18387, 'cold_start_seconds_after_runtime_initialization': 0.08488009998109192, 'features_emitted_during_cold_start': 257, 'normal_update_and_projection_us': {'samples': 256, 'p50': 14.5, 'p95': 16.575, 'p99': 25.66999999999999}}}}`.
Feature generation starts from the cached BASE70 and raw 14-column development arrays. The upstream raw-market-to-BASE70 build was not rerun or timed in this comparison; its complete source code, configuration, ordered feature definitions, and hashes are included in the bundle manifests.

## Test outside search_validation

The former one-path split designated 2025-07-01 through 2026-07-01 as confirmation. Its configurations and manifests show that it has already been used for prior model comparisons, so it must not be called a fresh test or reused to make this result look independent. The project holdout is 2026-07-01 through exclusive 2026-10-07. The project research contract records prior exposure of historical holdout results; it is also not unseen. This task did not read any confirmation or holdout inputs, predictions, labels, or scores.

The only eligible existing external period for a one-time frozen transfer of these candidates is the protected 2026-07-01–2026-10-07 holdout. Keep it protected until an explicitly authorized run. At the test start, refit A/B/C/D once using all pre-test rows whose labels were available by 2026-07-01T00:00:00Z, preserving all feature, model, seed, ensemble, target, weight, and purge settings. Fit memory/ESN scalers and PCA only on those allowed training rows. Fix 31 iterations with no test early stopping. Carry only causal feature state into the first decision, warm it using pre-test history, and never use a later test observation for an earlier decision.

Use one common mask of test decision timestamps with valid BASE70, memory, all ESN members, and a label released before the frozen exclusive cutoff. Log loss is primary; Brier and AUC are secondary. Report B−A, C−A, C−B, D−B, D−C, and D−A using equal-weight ensembles; use the same 3-calendar-day paired moving block bootstrap from `utils.btc_oof_audit.moving_calendar_block_multiplicity` and publish all exclusions. Execute once, save input/code/model hashes before scoring, do not revise settings after seeing results, and label the interval historical/exposed rather than fresh.

Development comparison command: `python run_btc_candidate_quality_20261022.py`.
Future protected-test gate command: `python run_btc_candidate_holdout_20261022.py`. Its separate config defaults to `allow_protected_data: false`; this gate reads only that config and fails before importing any data pipeline or opening a protected file. The protected scorer is intentionally not implemented or run here; it must be supplied as part of a separately authorized evaluation.

## Provenance and limits

- Source code commit: `f9a372a0a33fd2f9a873c83f5eeee0f657e0f926`; focused-search original frozen source commit: `95cc942b42b28561317bfa74379e1a8c83869429`.
- Runner code hash: `6e60bee961457991db3384e8add5bebbcdf7b0c1c98c9ac15b534e697d6879ae`; freeze-config hash: `0fbade42eb91d48e876d4178d15de1da6a86e185c44f915bc0620603f907e175`.
- Prior development selection results remain +0.000136588 mean ΔLL for B and +0.000101431 mean ΔLL for C; both came from repeated use of search_validation.
- Output: `data\analysis\polymarket\BTC\btc_candidate_quality_freeze_20261022`; report: `reports\btc_preopen\btc_candidate_quality_freeze_20261022.md`.
