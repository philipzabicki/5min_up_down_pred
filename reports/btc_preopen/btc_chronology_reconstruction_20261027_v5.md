# E0039 chronology runner v5 — runtime-ledger correction

## Status

**FROZEN_AFTER_LEDGER_FIX; FIT NOT STARTED.** V4 verified the resource gate and all source hashes, loaded the allowed source projections, then stopped before F1 selection because its empty runtime-ledger initializer used a set containing a dictionary. V5 corrects the initializer and adds a regression for starting and finishing a new ledger. This change must be committed and pushed before the next run.

Protocol: [E0039 v5](../../configs/research/btc_chronology_reconstruction_20261027_v5.json), SHA-256 `7ff752eaef17774725349d2f4a8d5cd7cb456ebecb575969bf8eb5c65279fcb9`. Parent: [E0039 v4](../../configs/research/btc_chronology_reconstruction_20261027_v4.json), SHA-256 `6821739db72dec443186f2fdb6b8c99960af4b5ad96310fc500e44bb54cf691e`.

## Change boundary

V5 replaces the malformed empty-ledger expression with a plain dictionary. Existing ledger loading, identity checks, budget accounting, resume behavior, sampler, fold procedures, and output paths are unchanged. The runner still uses the four observed E0039 source projection hashes; it will recompute and verify them before fitting. No new data, sample, split, model, calibration, trading, or accounting rule is introduced.

The E0038 clean whitelist remains 5,604 markets with canonical ID hash `e2946aa8e45c2ed050347e18756408a0175fa0eb9859a120f8df7c173051edda`. The four source hashes remain the values recorded in the v3 pre-fit manifest and pinned in v4.

## Validation

- Focused E0039 tests: **20/20 passed**, including the runtime-ledger initialization/completion regression.
- Full `python -m unittest discover -s tests`: **731/731 passed** in **66.208 seconds**, exit code 0.
- `py_compile`: passed.
- V5 identity check: **53** pinned files verified; all **5,604** clean IDs match the frozen hash.

These checks validate the ledger correction and runner mechanics only. No fold, OOF calibration, outer prediction, economic replay, or runtime benchmark has completed.

## Next step

After commit and push, run:

```powershell
python run_btc_chronology_reconstruction_20261027.py
```

The runner must repeat the current resource gate, verify all four pinned source hashes, then complete F1, F2, F3 and the continuous E0038 economic ledger under the frozen budgets. The offline adapter and 300-window latency measurement remain after the batch path.
