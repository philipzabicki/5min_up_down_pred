# E0039 chronology runner v7 — parse causal join timestamps

## Status

**FROZEN_AND_VALIDATED; v7 has not yet been run.** V6 completed F1 indicators, reaction-profile tuning, volume-profile tuning, and the final feature dataset, then failed in the exact causal candidate feature join because the base dataset stores `Opened` as a UTC-naive string while the join cast it directly to `int64`.

Protocol: [E0039 v7](../../configs/research/btc_chronology_reconstruction_20261027_v7.json), SHA-256 `9c6aca98034ca010cda16e326053ce83e2c78dbab0120445be8ec1180392baf6`. Parent: [E0039 v6](../../configs/research/btc_chronology_reconstruction_20261027_v6.json), SHA-256 `2facc83e0b68ae0dc26874ec5dfd85b70559d1fa17a2ec4156744099bd580d3c`.

## Change boundary

- The causal join converts Arrow timestamp fields to nanosecond ticks and parses canonical `%Y-%m-%d %H:%M:%S` `Opened` strings with the candle-file UTC convention.
- Exact timestamp matching, ordered/unique timestamp validation, one-minute alignment, and NaN for unmatched rows remain unchanged.
- Source files and hashes, the 5,604-market clean cohort, fold boundaries, labels, feature set, trial budgets, fit/selection/calibration procedures, economics, and accounting remain unchanged.
- No holdout, confirmation evaluation, PMXT replay, source acquisition, shadow run, or order is part of this protocol.

## Validation

- Regression: reproduced the Arrow string-to-int cast failure before the fix; **1/1 passed** after the fix, including exact join and gap preservation.
- E0039 chronology tests: **20/20 passed**.
- Project configuration tests: **23/23 passed**.
- Full `python -m unittest discover -s tests`: **733/733 passed** in **48.443 seconds**, exit code 0.
- `py_compile`: passed for the causal pipeline, chronology runner, and pinned E0039 test file.
- Read-only identity verification passed for the v6 parent, all **53** pinned implementation/source/config files, all three frozen fold boundaries, and the four frozen source-hash entries; protected holdout use remains forbidden.

These checks validate implementation mechanics only. No v7 predictive fit, OOF/calibration result, outer prediction, economic result, or runtime benchmark exists.

## Next step

Commit and push v7, then run:

```powershell
python run_btc_chronology_reconstruction_20261027.py
```

The exact runner must repeat its live resource gate, verify the clean E0038 IDs and four pinned source projection hashes, then continue F1 and complete F2/F3 and the shared continuous economic ledger under the unchanged recipe. The offline streaming adapter and frozen 300-window latency benchmark remain `PARTIAL_NOT_IMPLEMENTED` and follow the batch path. T0009 remains blocked overall; T0010 remains deferred.
