# E0039 chronology runner v5 — fold dataset path failure

## Status

**F1 stopped during final feature dataset construction.** The resource gate passed, all four pinned source projection hashes matched, and the indicator, reaction-profile, and volume-profile tuning stages completed. The fold could not build its final feature dataset because profile-state path resolution could not infer the symbol and interval from the temporal prefix filename `fold_20260415_20260501.csv`.

Protocol: [E0039 v5](../../configs/research/btc_chronology_reconstruction_20261027_v5.json), SHA-256 `7ff752eaef17774725349d2f4a8d5cd7cb456ebecb575969bf8eb5c65279fcb9`. Parent: [E0039 v4](../../configs/research/btc_chronology_reconstruction_20261027_v4.json), SHA-256 `6821739db72dec443186f2fdb6b8c99960af4b5ad96310fc500e44bb54cf691e`.

Run identity: `442a0de18fda1bfadd9759bff0f2d7c320266c8e536fb1e3a29cca957dd3eea6`. Full run manifest: [`run_manifest.json`](../../data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027/runs/442a0de18fda1bfa/run_manifest.json). Total runtime was 3,064.967 seconds; F1 runtime before failure was 3,047.335 seconds. The in-run resource snapshot recorded 49.15 GiB available RAM, 105.90 GiB free disk, and 6,393 MiB free VRAM before fitting. During F1, the wrapper sampled a 12.346 GiB minimum available system RAM and a 15.032 GiB runner peak working set; unrelated-process memory was included in the system minimum.

## F1 evidence

- Fold: `[2026-04-15, 2026-05-01)`, source prefix `fold_20260415_20260501.csv`.
- Indicator selection completed in 289.8 seconds. Six 1m indicator result files were saved.
- `reaction_profile` completed 96 trials in 1,517.2 seconds: 84 complete, 12 pruned, 0 errors.
- `volume_profile` completed 96 trials in 1,163.6 seconds: 96 complete, 0 pruned, 0 errors.
- `final_feature_dataset` then failed in `resolve_volume_profile_modeling_state_path` with `ValueError: Could not derive symbol/interval ... from base_data_file='fold_20260415_20260501.csv'`. The selected dataset settings already contain `symbol=BTCUSD` and `interval=1m`; v6 passes these through as the fallback identity for temporal-prefix filenames. Existing symbol-bearing filenames keep their current parsing behavior.
- No final fold feature dataset or predictive LightGBM model was produced. No candidate OOF output, official calibration result, outer prediction, or economic replay was produced. No holdout rows were materialized.

The observed four source projection hashes remain those pinned in v4/v5 and were rechecked by the v5 runner. The run did not change the clean E0038 cohort, folds, labels, selection recipe, accounting semantics, or source set.

## Offline runtime

- Status: `PARTIAL_NOT_IMPLEMENTED`.
- Detail: The selected-generator/VP/RP streaming adapter and frozen 300-window latency benchmark remain outstanding.

## Interpretation limits

- This is a retrospective chronological reproduction conditional on the previously exposed A/B/C/D family search.
- Historical T-59 readiness and network delay remain `UNKNOWN`.
- E0037 remains historical `FAILED`; E0038 artifacts are unchanged.
- T0009 remains blocked overall by source coverage and readiness/provenance gaps; T0010 remains deferred.
- No holdout, confirmation run, PMXT replay, source acquisition, shadow run, or order was performed.

## Next step

Commit and push the v6 state-path fallback and protocol identity before rerunning the exact no-argument runner. The run must repeat the resource and source-hash gates, then resume F1 and continue F2, F3, the continuous economic ledger, and the frozen runtime work without changing the research procedures.
