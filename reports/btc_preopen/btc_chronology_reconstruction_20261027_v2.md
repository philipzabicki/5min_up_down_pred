# E0039 chronology runner v2

## Status

**BLOCKED_BEFORE_FIT.** The batch runner and protocol were committed and pushed before the required command was run. The runner stopped at its pre-fit resource gate, before reading fit data or labels. No F1/F2/F3 model fit, OOF prediction, official calibration, outer prediction, economic replay, or runtime benchmark was produced.

The frozen protocol is [E0039 v2](../../configs/research/btc_chronology_reconstruction_20261027_v2.json), SHA-256 `7b042861fd46af05d1a5f0de4ecc6aeba0f141f20f8b9aabe83dc313d271d220`. Its parent is the unchanged [E0039 v1 protocol](../../configs/research/btc_chronology_reconstruction_20261027.json), SHA-256 `6b901e753aa05ac5172fd753568638c4ff3880d3d6126b914c51db864866399f`.

## Implementation and validation

Commit `439fbcb15f177681b4fae1347486c494345756ef` (`feat(btc-research): add E0039 chronology runner`) is on `audit/btc-dynamic-history-20261020` and was pushed to its configured upstream before the runner invocation.

The frozen work adds the fold-local BASE70 plus six causal flow-feature selection pipeline, memory/ESN state construction, complete A/B/C/D ensembles, earlier-only official Gamma OOF calibration, source-clean E0038 accounting, and identity-checked checkpoints. Hard search stages cannot be marked complete below their frozen targets. The strict-positive E0038 EV rule is recorded; exact-zero EV is rejected and an exact side tie selects UP. E0037 remains **FAILED** and E0038 artifacts remain unchanged.

- Focused E0039 tests: **17/17 passed**, including real synthetic LightGBM fits for all A/B/C/D members, Platt calibration, one-class/minimum behavior, chronology and current-outer-label independence.
- Full repository suite: **728/728 passed** in **54.381 seconds**.
- These tests validate mechanics only; no full research fit ran.

The per-fold time allocation is 47,700 seconds for the shared selection chain, 7,200 seconds for all 48 inner/outer A/B/C/D members plus calibration and fold assessment, and 17,100 seconds for state construction, I/O and economics, totaling 72,000 seconds. Trial, generation and weight-candidate targets remain fixed in v2.

## Pre-fit resource gate

The exact command `python run_btc_chronology_reconstruction_20261027.py` returned `BLOCKED_BEFORE_FIT` on 2026-10-27. The versioned [pre-fit gate record](btc_chronology_reconstruction_20261027_pre_fit_gate.json) reports:

| Resource | Observed | Gate |
|---|---:|---:|
| Available RAM | 3.79 GiB | At least 20 GiB |
| Free D: disk | 105.97 GiB | At least 80 GiB |
| Other Python processes at or above 1 GiB | PIDs 42800, 46484, 41684, 50164, 47096; each used 9.53–10.06 GiB working set | None |
| GPU | RTX 4060 Laptop GPU, 6,927 / 8,188 MiB free | Available |

The failure is limited to RAM availability and active Python workloads. The runner recorded `data_or_label_reads_for_fit: false` and `holdout_rows_materialized: 0`; it did not open the calibration or economic datasets. Consequently, the development-only projected source-row hashes have not been computed yet. No process was stopped.

## Remaining status and resume

Chronology is **implemented but blocked before fit**. Source coverage remains conditional on E0038's 5,604 clean markets; T0009 remains **BLOCKED** overall and T0010 remains deferred. The offline streaming adapter and its 300-window parity/latency benchmark remain **PARTIAL_NOT_IMPLEMENTED**; this runtime status is separate from the batch-fit resource gate.

After the active Python workloads exit and the frozen RAM/disk/GPU gate passes, resume with the same no-argument command:

```powershell
python run_btc_chronology_reconstruction_20261027.py
```

It will compute the exact development-only input row hashes before any fit, then run F1, F2 and F3 in the frozen order and continue the single cross-fold E0038 economic ledger. Do not use protected holdout data or alter the frozen recipe.
