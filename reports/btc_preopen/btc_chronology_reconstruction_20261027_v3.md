# E0039 chronology runner v3 — execution attempt

## Status

**FAILED_BEFORE_FIT.** The v3 protocol corrected the clean-ID hash mismatch and was pushed before this retry. The runner passed the live resource gate, verified its frozen identities, computed the frozen source projection hashes, then stopped while loading official Gamma rows because this PyArrow version requires `columns` to be a list or dictionary rather than a tuple. No fold started and no fit, OOF prediction, calibration, outer prediction, or economic replay ran.

Protocol: [E0039 v3](../../configs/research/btc_chronology_reconstruction_20261027_v3.json), SHA-256 `d3a5dd3d916c117e627c95683865a66a3e0685bac7862d901307f92a77060047`. Parent: [E0039 v2](../../configs/research/btc_chronology_reconstruction_20261027_v2.json), SHA-256 `7b042861fd46af05d1a5f0de4ecc6aeba0f141f20f8b9aabe83dc313d271d220`. The E0039 v1/v2 protocols and the v2 resource-gate report remain unchanged.

## Resource gate and identity

The same-interpreter snapshot passed: 50.06 GiB available RAM, 105.91 GiB free disk, RTX 4060 Laptop GPU with 6,453 MiB free, 20 logical CPUs, and no other Python processes. All 53 frozen input and implementation hashes verified; the 5,604-ID E0038 whitelist hash matched v3.

Run identity: `2bc77d0d309e39df6e1baf98741473a0cc678b8cec271ac0852368245367d4ce`. The local ignored manifest records status `interrupted_or_failed_with_checkpoints`, zero folds, `confirmation_rows_materialized=0` (no confirmation evaluation output), and zero holdout rows materialized. Its exact failure was `ValueError: Expected a list or a dict for 'columns', got <class 'tuple'> instead.`

## Source projection hashes

Before the loader failure, the runner computed these hashes from the frozen projections and recorded them in the ignored run manifest:

- `official_gamma`: `8a57353f8ab2fa192dda31d5848c08d8a942ac2cfecba42c3fd02c3d6b97cc88`
- `economics_gamma`: `2376742f69af58ae09abd2f97a69a790e8a4478b534e89c489650bd2e03e79f1`
- `t60_books`: `18bcea350e8857c42d236fe44d9753c98fd8c1bf9ab7f4d1e1d6060eca5f4165`
- `t59_asks`: `8cd8b338718a6c2aea41624015723ae6f7246232a114db40643d51052e5676f7`

The economic projections follow the frozen whitelist and date limits. The official Gamma input slice is frozen for the earlier-only OOF calibration lineage; its rows were read here only for the protocol's source-hash check. No calibration was fit, no confirmation scoring or choice was made, and no holdout row was materialized. V4 pins these observed hashes and recomputes/verifies them before fitting.

## Validation and fix

The reader now converts the unchanged ordered column tuple to a list before calling PyArrow. A regression confirms tuple input works and preserves the frozen half-open development window `[2026-04-15, 2026-07-01)`.

- Focused E0039 tests: **19/19 passed**.
- Full `python -m unittest discover -s tests`: **730/730 passed** in **45.116 seconds**, exit code 0.
- `py_compile`: passed.
- V4 identity check: 53 pinned files verified; all 5,604 whitelist IDs match.

V3 still contains no fold result. The source hashes are inputs to the next run, not economic results or evidence of model quality. The PyArrow fix and hashes are frozen in [E0039 v4](../../configs/research/btc_chronology_reconstruction_20261027_v4.json), SHA-256 `6821739db72dec443186f2fdb6b8c99960af4b5ad96310fc500e44bb54cf691e`.

## Next step

Commit and push the v4 implementation and protocol correction, then rerun the exact no-argument command. It must pass the resource gate and recompute all four pinned source hashes before F1. Preserve the three chronological folds, complete OOF/calibration work, and shared economic ledger. No holdout, new confirmation evaluation, source acquisition, PMXT replay, live/shadow operation, DNS check, or order is authorized.
