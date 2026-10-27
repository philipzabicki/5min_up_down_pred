# E0039 chronology runner v3 — protocol correction

## Status

**FROZEN_AFTER_HASH_CORRECTION; FIT NOT STARTED.** The v2 resource gate originally stopped at 3.79 GiB available RAM. A later exact-command retry passed the current resource gate and verified its frozen files, then stopped at the E0038 clean-ID hash check before fitting. This v3 protocol corrects the hash identity and must be committed and pushed before the next research run.

Protocol: [E0039 v3](../../configs/research/btc_chronology_reconstruction_20261027_v3.json), SHA-256 `d3a5dd3d916c117e627c95683865a66a3e0685bac7862d901307f92a77060047`. Parent: [E0039 v2](../../configs/research/btc_chronology_reconstruction_20261027_v2.json), SHA-256 `7b042861fd46af05d1a5f0de4ecc6aeba0f141f20f8b9aabe83dc313d271d220`. The original [v2 execution report](btc_chronology_reconstruction_20261027_v2.md) and protocol remain unchanged.

## Hash correction

The v2 protocol stored `b92ed75e56a1d5fa438642590b4d13bf6d2745871cc66c606a910d564532d62e` in `clean_ids_sha256`. That value is the SHA-256 of the complete local E0038 clean-cohort CSV. The runner's frozen check hashes sorted lowercase `condition_id` values, which produced `e2946aa8e45c2ed050347e18756408a0175fa0eb9859a120f8df7c173051edda`.

The 5,604 IDs selected from the frozen source-quality manifest and the CSV have identical sets and no duplicate CSV IDs. The v3 correction changes the hash field, parent protocol identity, and runner/test identity hashes. It does not change the whitelist, source data, count, filters, chronology boundaries, labels, features, fold budgets, candidate procedures, calibration, economic rules, or runtime sample.

## Retry evidence

The retry's same-interpreter resource snapshot passed the frozen gates: 50.15 GiB available RAM, 105.91 GiB free disk, 6,206 MiB free on the RTX 4060 Laptop GPU, and no other Python process at the 1 GiB working-set threshold. The runner verified 53 frozen file and implementation hashes before loading the clean whitelist. It then raised `RuntimeError: E0038 clean ID allowlist hash changed`.

The retry loaded only the clean-whitelist columns after the gate. It did not load model-fit targets, Gamma outcomes, T-60/T-59 economic rows, confirmation rows, or holdout rows. No development-only projected source-row hashes, fold fits, OOF calibration, outer predictions, portfolio replay, or runtime measurement were produced. The frozen v2 report and gate record remain historical evidence; v3 will repeat the resource gate.

## Validation

- Focused E0039 chronology tests: **18/18 passed**.
- Full `python -m unittest discover -s tests`: **729/729 passed** in **48.090 seconds**, exit code 0.
- `python -m py_compile run_btc_chronology_reconstruction_20261027.py tests/test_btc_chronology_reconstruction_20261027.py`: passed.
- V3 identity check: **53** pinned files verified; all **5,604** whitelist IDs match the canonical hash above.

These checks validate the hash correction and code mechanics only. They do not validate fold reproduction, model quality, economic results, or runtime readiness.

## Next step

After this v3 protocol/code/test change is committed and pushed, rerun the exact no-argument command:

```powershell
python run_btc_chronology_reconstruction_20261027.py
```

The runner must recompute its resource gate, verify v3 identities, and compute exact development-only projected source-row hashes before F1. Do not change the frozen procedure or use protected holdout data. T0009 remains blocked overall; the offline streaming adapter and 300-window benchmark remain pending after the batch path.
