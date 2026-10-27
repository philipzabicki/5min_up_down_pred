# E0039 chronology runner v6 — fold state identity fallback

## Status

**FROZEN_AND_VALIDATED; v6 has not yet been run.** V5 completed F1 indicator selection, reaction-profile tuning, and volume-profile tuning, then failed when final feature dataset construction could not infer symbol/interval from the temporal-prefix filename. V6 passes the already configured dataset identity to the VP/RP state-path resolvers when a filename has no source identity.

Protocol: [E0039 v6](../../configs/research/btc_chronology_reconstruction_20261027_v6.json), SHA-256 `2facc83e0b68ae0dc26874ec5dfd85b70559d1fa17a2ec4156744099bd580d3c`. Parent: [E0039 v5](../../configs/research/btc_chronology_reconstruction_20261027_v5.json), SHA-256 `7ff752eaef17774725349d2f4a8d5cd7cb456ebecb575969bf8eb5c65279fcb9`.

## Change boundary

- The existing `symbol + interval` filename parser remains first choice. If a fold-prefix filename does not match it, state paths use `settings["symbol"]` and `settings["interval"]`.
- For this BTC dataset, the fallback is `BTCUSD` and `1m`; the run-local state names become `BTCUSD_1m_*_modeling_end`.
- The v5 failure, E0038 clean cohort, source files and hashes, fold boundaries, labels, features, trial budgets, candidate procedures, calibration, economics, and accounting semantics are unchanged.
- The four filtered source-row hashes remain pinned as observed and must be recomputed by the exact runner before fitting.

## Validation

- Fold-state path tests: **23/23 passed**, including the fold-filename fallback for both VP and RP state paths.
- E0039 chronology tests: **20/20 passed**.
- Full `python -m unittest discover -s tests`: **733/733 passed** in **40.668 seconds**, exit code 0.
- `py_compile`: passed for `create_modeling_dataset.py`, the chronology runner, and both pinned test files.
- Read-only identity verification: passed for the v5 parent and all **53** pinned implementation/source/config paths; all three frozen folds and four source projection hashes are present; holdout use is forbidden.

These checks validate implementation mechanics only. No v6 fit, prediction, calibration, economic result, or runtime benchmark exists.

## Next step

Commit and push v6, then run the exact no-argument command:

```powershell
python run_btc_chronology_reconstruction_20261027.py
```

The run must repeat the live resource gate, verify the 5,604 clean IDs and four pinned source projection hashes, resume F1, and continue F2/F3 and the continuous economic ledger. After the batch path, complete the selected-generator/VP/RP offline adapter and frozen 300-window latency benchmark. No holdout, confirmation run, source acquisition, PMXT replay, shadow run, or order is authorized by this protocol.
