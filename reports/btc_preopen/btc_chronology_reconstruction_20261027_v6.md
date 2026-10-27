# E0039 chronology run v6 — causal join timestamp failure

## Status

**F1 selection and final dataset completed; causal feature join failed before predictive fitting.** The exact v6 runner passed the live resource gate, reverified all four frozen source projection hashes and the 5,604-ID clean cohort, then completed indicators, reaction-profile tuning, volume-profile tuning, and final dataset construction. It stopped when the causal feature join cast the base dataset's string `Opened` values directly to `int64`.

Protocol: [E0039 v6](../../configs/research/btc_chronology_reconstruction_20261027_v6.json), SHA-256 `2facc83e0b68ae0dc26874ec5dfd85b70559d1fa17a2ec4156744099bd580d3c`. Parent: [E0039 v5](../../configs/research/btc_chronology_reconstruction_20261027_v5.json), SHA-256 `7ff752eaef17774725349d2f4a8d5cd7cb456ebecb575969bf8eb5c65279fcb9`.

Run identity: `eb07a734607052ac66e245b38a608d57f1d6c80662b4aa89d47657142678269e`. Ignored manifest: [`run_manifest.json`](../../data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027/runs/eb07a734607052ac/run_manifest.json). Runtime to failure was 3,184.845 seconds. The pre-fit snapshot recorded 51.01 GiB available RAM, 105.90 GiB free disk, and 6,375 MiB free VRAM; no gate failures or memory-blocking Python processes were reported. Holdout rows materialized: zero.

## F1 evidence

- Fold: `[2026-04-15, 2026-05-01)`, prefix `fold_20260415_20260501.csv`.
- Indicator selection completed in 271.7 seconds; six indicator result files were saved.
- `reaction_profile` completed 96 trials in 1,489.1 seconds: 84 complete, 12 pruned, 0 errors.
- `volume_profile` completed 96/96 trials in 1,209.6 seconds, with no errors.
- `final_feature_dataset` completed in 190.7 seconds. It built the 3,098,307-row dataset and saved VP/RP state files under the configured `BTCUSD_1m_*_modeling_end` identities, confirming the v6 path fallback worked.
- `causal_candidate_feature_join` then failed with `ArrowInvalid: Failed to parse string: '2020-09-08 10:04:00' as a scalar of type int64`. The base `Opened` field is a UTC-naive candle timestamp string; the causal feature timestamps are Arrow timestamp values. The join's existing direct integer cast handled timestamp arrays but not strings.

The v7 correction parses the canonical `%Y-%m-%d %H:%M:%S` string form as UTC candle time to nanosecond timestamps before the exact join. It retains typed Arrow timestamp support, ordering/uniqueness checks, exact matching, and NaN for unmatched causal rows. The regression first reproduced the direct-cast error, then passed with the correction.

No causal-joined dataset, predictive LightGBM fit, candidate OOF output, official calibration result, outer prediction, economic replay, or runtime benchmark was produced. No confirmation evaluation output or holdout rows were materialized. E0037 remains historically `FAILED`; E0038 artifacts and T0009 source/readiness limitations are unchanged; T0010 remains deferred.

## Next step

Commit and push the v7 timestamp parser and frozen protocol before rerunning the exact no-argument runner. It must repeat the live resource gate, clean-ID check, and four source projection hash checks before fitting. Do not use holdout data or change the research procedure.
