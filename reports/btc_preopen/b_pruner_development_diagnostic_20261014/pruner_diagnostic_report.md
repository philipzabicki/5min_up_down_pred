# BTC pre-open: development pruning continuation diagnostic

> This is a small diagnostic continuation on the three existing development B folds. It is not a new independent OOS test, does not score the holdout, and does not select a model or policy.

## Frozen procedure and provenance

- Frozen manifest: `reports/btc_preopen/b_pruner_development_diagnostic_20261014/pruner_diagnostic_manifest.json` (identity `b9f24e2b32671c46f1b6406b19660f18d7e4c0946e85c33c5e0a4cb7e91c175f`).
- The five parameter sets are the already-frozen PRUNED trials 57, 9, 44, 42, 23 from the final-fit B study. That study's objective labels ended by `2026-06-30T23:59:00Z`; its study contract excludes holdout labels.
- Each development fold uses its own B feature matrix, fit boundary, target-weight vector, and saved ten-fold purged chronological CV splits. The observed input checks match the saved tuning split and training-weight hashes.
- Metric: unweighted actual-decision-row binary log loss; the minimized objective is mean plus 0.5 times the CV standard deviation. Maximum 500 iterations, early stopping 50, seed 20261004, backend gpu, 20 threads.
- Existing pruner checkpoints: iterations 25, 50; Optuna pruning was disabled for these fixed runs while its reporting callback remained active. No TPE suggestions, feature selection, generator fitting, or holdout evaluation ran.
- No hypothetical MedianPruner decisions are replayed, so no false-pruning percentage is estimated. The comparison is against completed configs from each same development B study.

## Continued configurations

| Fold | Config | Source trial | Learning rate | Source prune iteration | Source / continued objective at checkpoints | Continued best objective | Best iteration | Best binary log loss mean | Time (s) | Same-study COMPLETE best / median | Delta vs best |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| dev_2026q2_a | B_PRUNED_LR_Q1 | 57 | 0.026590 | 25 | 25:0.692454/0.692508 | 0.692478 | 34 | 0.692213 | 66.6 | 0.692437 / 0.692484 (7 COMPLETE) | +0.000042 |
| dev_2026q2_a | B_PRUNED_LR_Q2 | 9 | 0.036137 | 50 | 25:0.692393/0.692517, 50:0.692365/0.692580 | 0.692509 | 24 | 0.692243 | 56.0 | 0.692437 / 0.692484 (7 COMPLETE) | +0.000072 |
| dev_2026q2_a | B_PRUNED_LR_Q3 | 44 | 0.045212 | 50 | 25:0.692419/0.692509, 50:0.692456/0.692619 | 0.692504 | 24 | 0.692241 | 41.0 | 0.692437 / 0.692484 (7 COMPLETE) | +0.000067 |
| dev_2026q2_a | B_PRUNED_LR_Q4 | 42 | 0.057655 | 50 | 25:0.692382/0.692554, 50:0.692419/0.692748 | 0.692500 | 17 | 0.692249 | 50.5 | 0.692437 / 0.692484 (7 COMPLETE) | +0.000063 |
| dev_2026q2_a | B_PRUNED_LR_Q5 | 23 | 0.081080 | 50 | 25:0.692422/0.692738, 50:0.692558/0.693092 | 0.692556 | 11 | 0.692292 | 52.1 | 0.692437 / 0.692484 (7 COMPLETE) | +0.000119 |
| dev_2026q2_b | B_PRUNED_LR_Q1 | 57 | 0.026590 | 25 | 25:0.692454/0.692449 | 0.692291 | 55 | 0.692090 | 113.1 | 0.692283 / 0.692309 (5 COMPLETE) | +0.000008 |
| dev_2026q2_b | B_PRUNED_LR_Q2 | 9 | 0.036137 | 50 | 25:0.692393/0.692394, 50:0.692365/0.692320 | 0.692316 | 40 | 0.692097 | 73.0 | 0.692283 / 0.692309 (5 COMPLETE) | +0.000033 |
| dev_2026q2_b | B_PRUNED_LR_Q3 | 44 | 0.045212 | 50 | 25:0.692419/0.692368, 50:0.692456/0.692348 | 0.692340 | 54 | 0.692105 | 61.2 | 0.692283 / 0.692309 (5 COMPLETE) | +0.000057 |
| dev_2026q2_b | B_PRUNED_LR_Q4 | 42 | 0.057655 | 50 | 25:0.692382/0.692332, 50:0.692419/0.692381 | 0.692332 | 25 | 0.692119 | 59.6 | 0.692283 / 0.692309 (5 COMPLETE) | +0.000049 |
| dev_2026q2_b | B_PRUNED_LR_Q5 | 23 | 0.081080 | 50 | 25:0.692422/0.692416, 50:0.692558/0.692527 | 0.692386 | 15 | 0.692200 | 59.9 | 0.692283 / 0.692309 (5 COMPLETE) | +0.000103 |
| dev_2026q2_c | B_PRUNED_LR_Q1 | 57 | 0.026590 | 25 | 25:0.692454/0.692446 | 0.692240 | 73 | 0.692053 | 103.8 | 0.692271 / 0.692278 (5 COMPLETE) | -0.000031 |
| dev_2026q2_c | B_PRUNED_LR_Q2 | 9 | 0.036137 | 50 | 25:0.692393/0.692421, 50:0.692365/0.692326 | 0.692321 | 55 | 0.692136 | 75.5 | 0.692271 / 0.692278 (5 COMPLETE) | +0.000050 |
| dev_2026q2_c | B_PRUNED_LR_Q3 | 44 | 0.045212 | 50 | 25:0.692419/0.692360, 50:0.692456/0.692304 | 0.692275 | 78 | 0.692072 | 70.6 | 0.692271 / 0.692278 (5 COMPLETE) | +0.000004 |
| dev_2026q2_c | B_PRUNED_LR_Q4 | 42 | 0.057655 | 50 | 25:0.692382/0.692387, 50:0.692419/0.692349 | 0.692341 | 47 | 0.692146 | 74.3 | 0.692271 / 0.692278 (5 COMPLETE) | +0.000070 |
| dev_2026q2_c | B_PRUNED_LR_Q5 | 23 | 0.081080 | 50 | 25:0.692422/0.692360, 50:0.692558/0.692425 | 0.692342 | 28 | 0.692154 | 69.0 | 0.692271 / 0.692278 (5 COMPLETE) | +0.000071 |

## Interpretation

- Continued runs completed: 15/15. Of those, 1 reached or beat the best COMPLETE configuration in their same-fold development study by the original CV objective.
- This verdict applies only to these five parameter sets on the listed development CV folds. The sampled continuations can show that pruned parameter sets may later be competitive on development CV; they do not estimate how often the MedianPruner wrongly prunes, since these source decisions came from another chronological fit and are not replayed here.
- Full available per-iteration validation metric curves and exact checkpoint values are in `pruner_diagnostic_results.json`; the completed reference configurations are in `completed_development_trials.csv`.

## Runtime and reproduction

- Measured end-to-end runtime including input verification and data loading: 1052.2s (17.54 min); model configuration trials: 1026.1s across 15 completed configurations.
- Resume from saved config checkpoints without CLI arguments: `python run_btc_preopen_pruner_development_diagnostic.py`.
- The runner skips completed fold/configuration pairs and writes each completed curve atomically. An interrupted, incomplete configuration may be rerun; completed configurations are not repeated.
