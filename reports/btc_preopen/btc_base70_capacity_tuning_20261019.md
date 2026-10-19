# BASE70 main-model capacity tuning — 2026-10-19

## Scope and result

A single development-only CV study tested whether the inherited main-model settings restrict the existing 70-feature matrix. It changed only `num_leaves`, `min_data_in_leaf`, `lambda_l1`, and `lambda_l2`; the target, feature order, common mask, folds, purge, weights, learning rate, feature fraction, GPU backend, and `max_bin` stayed frozen. The candidate schedule was frozen before training. The study did not read holdout outcomes or use official labels/economics for selection.

BASE70 inherited the same effective parameter dictionary as the BASE64/main-model run (`data/analysis/polymarket/BTC/btc_preopen_taker_flow_20261018/B0_cv_summary.json`). The earlier BASE70 seed-stability fits used 31 fixed rounds without early stopping; FULL_FEATURES also fixed 31 rounds and changed `feature_fraction`. The original B0 summary records 500 maximum rounds and patience 50 but says its full history was not persisted after a post-fit NameError. Its saved 31-tree boosters were therefore not treated as completed early-stopping runs; the three-seed control was retrained.

The prior empty-partition audit is closed with evidence limited to **three UTC hours**; no remaining PMXT archives were downloaded. See [that report](btc_empty_partition_provenance_20261019.md).

Frozen matrix: 3,174,805 rows × 70 features; shared OOF 318,560 rows; ten chronological folds; weight SHA-256 `3a21a8cfecaeb10cdf0af1e63fd9833eef034954f6f530005bab71da6f1397ca`. The trial score is the arithmetic mean over all three seed-specific minima of mean unweighted fold log-loss + 0.5 × population fold standard deviation. No best seed is chosen and no average-probability ensemble enters selection.

## Frozen study and CV selection

The control was trial 0 with original parameters and reselected iterations: {'lambda_l1': 9.508616956284627, 'lambda_l2': 30.729843292781943, 'min_data_in_leaf': 12089, 'num_leaves': 81}. The selected configuration was trial 0: {'lambda_l1': 9.508616956284627, 'lambda_l2': 30.729843292781943, 'min_data_in_leaf': 12089, 'num_leaves': 81}; mean seed CV criterion 0.69232917 versus control 0.69232917. Difference (selected − control) across seeds: mean +0.00000000, population SD 0.00000000, range [+0.00000000, +0.00000000]. Lower is better. Improvement in the seed criterion occurred in 0/3 seeds.

Boundary hits for the selected setting: `not applicable: selected setting is the original control outside candidate levels`. Selected per-seed tree counts: `{'20261015': 31, '20261016': 26, '20261017': 34}`; control per-seed counts: `{'20261015': 31, '20261016': 26, '20261017': 34}`. Complete trials: 17; started configurations: 18; incomplete trials excluded: 1. Pruning was disabled. Study elapsed time: 1681.2s; peak process working set: 11.32 GiB.

Best completed candidate alternative: trial 13 `capacity_13`, parameters {'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 12089, 'num_leaves': 63}, criterion 0.69233247 (delta versus control +0.00000330). Its seed scores {'20261015': 0.692337431596273, '20261016': 0.6923495802949808, '20261017': 0.6923103908907787} versus control {'20261015': 0.6923218068124216, '20261016': 0.6923493132337768, '20261017': 0.6923163770781005}; mean selected-minus-control delta +0.00000330, population SD 0.00000908, improvement in 1/3 seeds. No completed candidate beat the control. Candidate boundary hits: `{'num_leaves': False, 'min_data_in_leaf': False, 'lambda_l1': False, 'lambda_l2': True}`.
For the completed candidates below the inherited 81 leaves (12 trials), the best used 63 leaves and scored 0.69233247 (delta +0.00000330; seed deltas [1.5624783851420077e-05, 2.670612040400755e-07, -5.986187321727066e-06], better in 1/3). For the higher 127-leaf group (4 trials), the best scored 0.69235099 (delta +0.00002183; seed deltas [3.02194817641821e-05, 1.8278060427312504e-05, 1.6980587048465168e-05], better in 0/3). Other candidate parameters co-varied within the frozen grid.

### Proxy OOF quality

The table reports all three seeds for control and selected configuration; historical `SINGLE_31` is listed separately. Seed-level proxy metrics are computed on the common 318,560-row OOF sample.

| Role | Seed | Trees | CV criterion | Proxy log-loss | Brier | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 20261015 | 31 | 0.69232181 | 0.69213430 | 0.24949413 | 0.525079 |
| control | 20261016 | 26 | 0.69234931 | 0.69216288 | 0.24950843 | 0.524538 |
| control | 20261017 | 34 | 0.69231638 | 0.69213950 | 0.24949671 | 0.525161 |
| selected | 20261015 | 31 | 0.69232181 | 0.69213430 | 0.24949413 | 0.525079 |
| selected | 20261016 | 26 | 0.69234931 | 0.69216288 | 0.24950843 | 0.524538 |
| selected | 20261017 | 34 | 0.69231638 | 0.69213950 | 0.24949671 | 0.525161 |
| historical_SINGLE_31 | 20261015 | 31 | — | 0.69213431 | 0.24949413 | 0.525079 |

The full per-seed, per-fold log-loss/Brier/AUC table and every iteration’s mean, fold SD, and criterion are saved in the local trial checkpoints. The report manifest records their paths and hashes.

### Paired proxy changes and fold results

Changes are selected minus control on each same seed; negative log-loss/Brier and positive AUC are better.

| Seed | CV criterion delta | Proxy log-loss delta | Brier delta | AUC delta |
| ---: | ---: | ---: | ---: | ---: |
| 20261015 | +0.00000000 | +0.00000000 | +0.00000000 | +0.000000 |
| 20261016 | +0.00000000 | +0.00000000 | +0.00000000 | +0.000000 |
| 20261017 | +0.00000000 | +0.00000000 | +0.00000000 | +0.000000 |
| Mean (population SD) | +0.00000000 (0.00000000) | +0.00000000 (0.00000000) | +0.00000000 (0.00000000) | +0.000000 (0.000000) |

Per-fold proxy metrics for the control and selected setting:

| Role | Seed | Fold | Trees | Validation rows | Log-loss | Brier | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 20261015 | 0 | 31 | 31858 | 0.69163805 | 0.24924645 | 0.530830 |
| control | 20261015 | 1 | 31 | 31841 | 0.69213038 | 0.24949185 | 0.526625 |
| control | 20261015 | 2 | 31 | 31862 | 0.69216427 | 0.24950925 | 0.526339 |
| control | 20261015 | 3 | 31 | 31861 | 0.69179773 | 0.24932601 | 0.529494 |
| control | 20261015 | 4 | 31 | 31842 | 0.69217446 | 0.24951416 | 0.524657 |
| control | 20261015 | 5 | 31 | 31856 | 0.69202887 | 0.24944166 | 0.526227 |
| control | 20261015 | 6 | 31 | 31861 | 0.69272171 | 0.24978743 | 0.517030 |
| control | 20261015 | 7 | 31 | 31858 | 0.69288389 | 0.24986840 | 0.514301 |
| control | 20261015 | 8 | 31 | 31860 | 0.69199685 | 0.24942565 | 0.526246 |
| control | 20261015 | 9 | 31 | 31861 | 0.69180683 | 0.24933045 | 0.530545 |
| control | 20261016 | 0 | 26 | 31858 | 0.69161806 | 0.24923690 | 0.530159 |
| control | 20261016 | 1 | 26 | 31841 | 0.69224761 | 0.24955055 | 0.524803 |
| control | 20261016 | 2 | 26 | 31862 | 0.69224495 | 0.24954938 | 0.524892 |
| control | 20261016 | 3 | 26 | 31861 | 0.69171472 | 0.24928443 | 0.531094 |
| control | 20261016 | 4 | 26 | 31842 | 0.69212878 | 0.24949144 | 0.524973 |
| control | 20261016 | 5 | 26 | 31856 | 0.69206407 | 0.24945906 | 0.526198 |
| control | 20261016 | 6 | 26 | 31861 | 0.69269750 | 0.24977547 | 0.516435 |
| control | 20261016 | 7 | 26 | 31858 | 0.69288203 | 0.24986734 | 0.515205 |
| control | 20261016 | 8 | 26 | 31860 | 0.69211888 | 0.24948660 | 0.524178 |
| control | 20261016 | 9 | 26 | 31861 | 0.69191223 | 0.24938314 | 0.528831 |
| control | 20261017 | 0 | 34 | 31858 | 0.69166338 | 0.24925929 | 0.530587 |
| control | 20261017 | 1 | 34 | 31841 | 0.69221977 | 0.24953658 | 0.525424 |
| control | 20261017 | 2 | 34 | 31862 | 0.69222860 | 0.24954129 | 0.525832 |
| control | 20261017 | 3 | 34 | 31861 | 0.69182324 | 0.24933849 | 0.529764 |
| control | 20261017 | 4 | 34 | 31842 | 0.69225901 | 0.24955621 | 0.524343 |
| control | 20261017 | 5 | 34 | 31856 | 0.69199172 | 0.24942328 | 0.526607 |
| control | 20261017 | 6 | 34 | 31861 | 0.69266687 | 0.24976014 | 0.517345 |
| control | 20261017 | 7 | 34 | 31858 | 0.69279776 | 0.24982517 | 0.516704 |
| control | 20261017 | 8 | 34 | 31860 | 0.69197217 | 0.24941339 | 0.526167 |
| control | 20261017 | 9 | 34 | 31861 | 0.69177257 | 0.24931333 | 0.530445 |
| selected | 20261015 | 0 | 31 | 31858 | 0.69163805 | 0.24924645 | 0.530830 |
| selected | 20261015 | 1 | 31 | 31841 | 0.69213038 | 0.24949185 | 0.526625 |
| selected | 20261015 | 2 | 31 | 31862 | 0.69216427 | 0.24950925 | 0.526339 |
| selected | 20261015 | 3 | 31 | 31861 | 0.69179773 | 0.24932601 | 0.529494 |
| selected | 20261015 | 4 | 31 | 31842 | 0.69217446 | 0.24951416 | 0.524657 |
| selected | 20261015 | 5 | 31 | 31856 | 0.69202887 | 0.24944166 | 0.526227 |
| selected | 20261015 | 6 | 31 | 31861 | 0.69272171 | 0.24978743 | 0.517030 |
| selected | 20261015 | 7 | 31 | 31858 | 0.69288389 | 0.24986840 | 0.514301 |
| selected | 20261015 | 8 | 31 | 31860 | 0.69199685 | 0.24942565 | 0.526246 |
| selected | 20261015 | 9 | 31 | 31861 | 0.69180683 | 0.24933045 | 0.530545 |
| selected | 20261016 | 0 | 26 | 31858 | 0.69161806 | 0.24923690 | 0.530159 |
| selected | 20261016 | 1 | 26 | 31841 | 0.69224761 | 0.24955055 | 0.524803 |
| selected | 20261016 | 2 | 26 | 31862 | 0.69224495 | 0.24954938 | 0.524892 |
| selected | 20261016 | 3 | 26 | 31861 | 0.69171472 | 0.24928443 | 0.531094 |
| selected | 20261016 | 4 | 26 | 31842 | 0.69212878 | 0.24949144 | 0.524973 |
| selected | 20261016 | 5 | 26 | 31856 | 0.69206407 | 0.24945906 | 0.526198 |
| selected | 20261016 | 6 | 26 | 31861 | 0.69269750 | 0.24977547 | 0.516435 |
| selected | 20261016 | 7 | 26 | 31858 | 0.69288203 | 0.24986734 | 0.515205 |
| selected | 20261016 | 8 | 26 | 31860 | 0.69211888 | 0.24948660 | 0.524178 |
| selected | 20261016 | 9 | 26 | 31861 | 0.69191223 | 0.24938314 | 0.528831 |
| selected | 20261017 | 0 | 34 | 31858 | 0.69166338 | 0.24925929 | 0.530587 |
| selected | 20261017 | 1 | 34 | 31841 | 0.69221977 | 0.24953658 | 0.525424 |
| selected | 20261017 | 2 | 34 | 31862 | 0.69222860 | 0.24954129 | 0.525832 |
| selected | 20261017 | 3 | 34 | 31861 | 0.69182324 | 0.24933849 | 0.529764 |
| selected | 20261017 | 4 | 34 | 31842 | 0.69225901 | 0.24955621 | 0.524343 |
| selected | 20261017 | 5 | 34 | 31856 | 0.69199172 | 0.24942328 | 0.526607 |
| selected | 20261017 | 6 | 34 | 31861 | 0.69266687 | 0.24976014 | 0.517345 |
| selected | 20261017 | 7 | 34 | 31858 | 0.69279776 | 0.24982517 | 0.516704 |
| selected | 20261017 | 8 | 34 | 31860 | 0.69197217 | 0.24941339 | 0.526167 |
| selected | 20261017 | 9 | 34 | 31861 | 0.69177257 | 0.24931333 | 0.530445 |

Every evaluated iteration's mean, fold SD, and criterion are saved in each local seed checkpoint, including rounds after the best iteration up to early stopping. The report manifest records their paths and hashes.

## Official development and fixed policy, after selection

Only after `selection.json` was written, the selected configuration and control were scored on the existing official development sample and replayed on the same saved 20,735-row calendar. Results use each seed separately and raw model probabilities with the existing $5 gross stake, 1% net-EV hurdle, T−60 decision, T−59 execution, snapshots, fee rules, and accounting audit. Holdout starts are excluded.

Frozen snapshot identities: decision T−60 `38259fadf50c28e455f5d4523610cbac827886e35922076dbba5cb33bd55e0b1`; execution T−59 `abee0fdaf31eaf1455b4343e985bf9d5ee38a84dd8eb7e06adf1e08fc0c380d0`. The policy calendar has 20,735 rows; 20,718 had official outcomes and cash releases by cutoff, while 17 missing outcomes remained excluded.

Missing source inputs remain explicit as rejections rather than fills:

| Seed | Calendar rows | Outcome available | Missing official outcome | No Gamma market record | No priceable native ask |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20261015 | 20735 | 20718 | 17 | 7 | 11910 |
| 20261016 | 20735 | 20718 | 17 | 7 | 11910 |
| 20261017 | 20735 | 20718 | 17 | 7 | 11910 |

| Role | Seed | Trees | Official LL | Brier | AUC | Fills | Settled | Fees USD | Net PnL USD | PnL / turnover | Audit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 20261015 | 31 | 0.69205172 | 0.24945273 | 0.527437 | 1038 | 1038 | 188.84 | 267.44 | 5.1530% | True |
| selected | 20261015 | 31 | 0.69205172 | 0.24945273 | 0.527437 | 1038 | 1038 | 188.84 | 267.44 | 5.1530% | True |
| control | 20261016 | 26 | 0.69232581 | 0.24958949 | 0.524294 | 919 | 919 | 168.05 | 102.28 | 2.2260% | True |
| selected | 20261016 | 26 | 0.69232581 | 0.24958949 | 0.524294 | 919 | 919 | 168.05 | 102.28 | 2.2260% | True |
| control | 20261017 | 34 | 0.69212628 | 0.24948975 | 0.526847 | 1193 | 1193 | 215.43 | 44.57 | 0.7472% | True |
| selected | 20261017 | 34 | 0.69212628 | 0.24948975 | 0.526847 | 1193 | 1193 | 215.43 | 44.57 | 0.7472% | True |

Paired changes (selected minus control) on the same official target and calendar:

| Seed | Official LL delta | Brier delta | AUC delta | Fills delta | Net PnL delta USD | PnL/turnover delta pp |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261015 | +0.00000000 | +0.00000000 | +0.000000 | +0 | +0.00 | +0.000 |
| 20261016 | +0.00000000 | +0.00000000 | +0.000000 | +0 | +0.00 | +0.000 |
| 20261017 | +0.00000000 | +0.00000000 | +0.000000 | +0 | +0.00 | +0.000 |

| historical SINGLE_31 | 20261015 | 31 | 0.69205172 | 0.24945273 | 0.527437 | 1038 | 1038 | 188.84 | 267.44 | 5.1530% | reused saved audited ledger |

A proxy improvement is not sufficient if it does not carry to the official development target and policy. These development results are conditional on earlier feature selection and the current tuning study; they are not independent confirmation of an edge. No holdout evaluation or deployment refit was run.

## Artifacts and limitations

Frozen protocol: `configs/research/btc_base70_capacity_tuning_20261019.json`. Run manifest: `data/analysis/polymarket/BTC/btc_base70_capacity_tuning_20261019/run_manifest.json`. Optuna SQLite study: `data/analysis/polymarket/BTC/btc_base70_capacity_tuning_20261019/study.sqlite3`. CV-only selection record: `data/analysis/polymarket/BTC/btc_base70_capacity_tuning_20261019/selection.json`. Saved fold models and OOF files are under `data/analysis/polymarket/BTC/btc_base70_capacity_tuning_20261019/trials`; official development quality and replay outputs are under `data/analysis/polymarket/BTC/btc_base70_capacity_tuning_20261019`.

Actual CV work: 540 fold fits started; 530 fully completed; 18 configurations started, 17 completed. Peak process working set was 11.32 GiB. The study was limited to its frozen candidate schedule and time budget; no space extension was run.
