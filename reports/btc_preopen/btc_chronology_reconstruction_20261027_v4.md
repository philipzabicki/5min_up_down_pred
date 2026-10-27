# E0039 chronology runner v4 — execution attempt

## Status

**FAILED_BEFORE_SELECTION.** V4 passed the resource gate, recomputed and verified the four frozen source projections, loaded the allowed Gamma/economic snapshots, and entered the F1 wrapper. It then failed before fold-local selection or fitting while initializing a new runtime ledger: the empty-ledger branch attempted to create a set containing a dictionary. No fold work, feature search, fit, OOF prediction, calibration, outer prediction, or economic replay ran.

Protocol: [E0039 v4](../../configs/research/btc_chronology_reconstruction_20261027_v4.json), SHA-256 `6821739db72dec443186f2fdb6b8c99960af4b5ad96310fc500e44bb54cf691e`. Parent: [E0039 v3](../../configs/research/btc_chronology_reconstruction_20261027_v3.json), SHA-256 `d3a5dd3d916c117e627c95683865a66a3e0685bac7862d901307f92a77060047`.

## Resource and source identity

The runner's same-process snapshot passed: 50.08 GiB available RAM, 105.90 GiB free disk, RTX 4060 Laptop GPU with 6,414 MiB free, and no blocking Python process. Four small Python processes were present, each using at most 0.05 GiB. The runner recomputed and matched all four source-projection hashes pinned in v4. It loaded the permitted pre-holdout Gamma calibration slice and the 5,604-market E0038 economic source projections; it did not evaluate confirmation metrics or materialize holdout rows.

Run identity: `3d81fb7714425ace6e62de28fd7be29ba8914fe3a012a6b03b5fca015e07d9af`. The ignored manifest records zero completed folds, zero confirmation evaluation output, and zero holdout rows materialized. The exact failure was `TypeError: cannot use 'dict' as a set element (unhashable type: 'dict')` at `_start_runtime_ledger`.

## Fix and validation

V5 replaces the malformed nested set/dictionary expression with the intended new-ledger dictionary. A regression starts and finishes a new ledger with its identity and recorded attempt.

- Focused E0039 tests: **20/20 passed**.
- Full `python -m unittest discover -s tests`: **731/731 passed** in **66.208 seconds**, exit code 0.
- `py_compile`: passed.
- V5 identity: 53 pinned files verified; 5,604 clean IDs and the four source hashes match their frozen values.

V4 produced no fold result. These checks validate implementation mechanics, not research outcomes. The fix and unchanged source hashes are frozen in [E0039 v5](../../configs/research/btc_chronology_reconstruction_20261027_v5.json), SHA-256 `7ff752eaef17774725349d2f4a8d5cd7cb456ebecb575969bf8eb5c65279fcb9`.

## Next step

Commit and push v5, then rerun the exact no-argument command. It must repeat the resource gate and verify the four source hashes before F1. No fold budget, source, chronology boundary, model procedure, calibration rule, or economic rule changes.
