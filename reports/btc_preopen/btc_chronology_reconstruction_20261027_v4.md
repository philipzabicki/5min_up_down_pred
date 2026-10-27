# E0039 chronology runner v4 — PyArrow compatibility correction

## Status

**FROZEN_AFTER_READER_FIX; FIT NOT STARTED.** The v3 retry passed resource and identity checks and computed all frozen source projection hashes, but stopped before F1 because PyArrow rejected the tuple-valued official Gamma column list. V4 converts the same ordered columns to a list and pins the observed source projection hashes. The v4 change must be committed and pushed before the next run.

Protocol: [E0039 v4](../../configs/research/btc_chronology_reconstruction_20261027_v4.json), SHA-256 `6821739db72dec443186f2fdb6b8c99960af4b5ad96310fc500e44bb54cf691e`. Parent: [E0039 v3](../../configs/research/btc_chronology_reconstruction_20261027_v3.json), SHA-256 `d3a5dd3d916c117e627c95683865a66a3e0685bac7862d901307f92a77060047`.

## Frozen source hashes

V4 pins the four values computed by the v3 run before its loader error. The runner will recompute these exact projections after the resource gate and reject any mismatch before fitting:

- `official_gamma`: `8a57353f8ab2fa192dda31d5848c08d8a942ac2cfecba42c3fd02c3d6b97cc88`
- `economics_gamma`: `2376742f69af58ae09abd2f97a69a790e8a4478b534e89c489650bd2e03e79f1`
- `t60_books`: `18bcea350e8857c42d236fe44d9753c98fd8c1bf9ab7f4d1e1d6060eca5f4165`
- `t59_asks`: `8cd8b338718a6c2aea41624015723ae6f7246232a114db40643d51052e5676f7`

These are source identities, not model or economic results. The official Gamma hash covers its frozen calibration source range `[2025-07-01, 2026-07-01)`; the economic hashes cover the E0038 whitelist within the three outer windows `[2026-04-15, 2026-07-01)`. The earlier Gamma slice supports the frozen earlier-only OOF calibration lineage; no new confirmation scoring or selection was performed. The E0038 cohort, source filters, UTC boundaries, labels, folds, model procedure, budgets, accounting and 300-window sample are unchanged.

## Validation

- Focused E0039 tests: **19/19 passed**, including tuple-column reading and half-open window filtering.
- Full `python -m unittest discover -s tests`: **730/730 passed** in **45.116 seconds**, exit code 0.
- `py_compile`: passed.
- V4 identity check: **53** pinned files verified; the 5,604-ID whitelist matches `e2946aa8e45c2ed050347e18756408a0175fa0eb9859a120f8df7c173051edda`.

These checks validate the reader compatibility fix and frozen identity only. There are no F1/F2/F3 results, official OOF calibration, outer predictions, economic ledger, or runtime measurements yet. T0009 remains blocked overall; the offline streaming adapter is still pending.

## Next step

After commit and push, run the exact no-argument command:

```powershell
python run_btc_chronology_reconstruction_20261027.py
```

It must repeat the live resource gate, recompute and verify all four source hashes, then run F1, F2, F3 and the one continuous economic ledger under the unchanged frozen protocol. Do not use holdout data or run source acquisition, PMXT replay, live/shadow, DNS checks or orders.
