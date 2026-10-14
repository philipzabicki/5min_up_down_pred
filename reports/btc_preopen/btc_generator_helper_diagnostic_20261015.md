# BTC VP/RP helper-model diagnostic — 2026-10-15

## Frozen design

- Development fold: `dev_2026q2_a`; fit labels stop at the frozen exclusive boundary in `data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/contracts/dev_2026q2_a.json`.
- Candidates: 6 existing COMPLETE generator configurations, selected before new CV by best score plus parameter distance and distinct best-iteration regions.
- Helper settings: 6 fixed variants; `6` CPU threads; 10 purged chronological folds; recency weighting off to match the effective model-update pipeline.
- Outcomes are development-only generator CV scores. There is no feature selection, main-model tuning, holdout evaluation, or prospective collection in this pilot.
- The report is resumable from the script constants with `python run_btc_generator_helper_diagnostic.py`; each completed candidate/helper pair is atomically checkpointed.

## Selected generator configurations

| Family | Candidate | Source trial | Previous objective | Previous best iteration | Selection basis |
|---|---|---:|---:|---:|---|
| VP | `vp_1` | 15 | 0.69279827 | 14 | lowest existing development CV objective |
| VP | `vp_2` | 4 | 0.69313558 | 9 | parameter-distant existing trial with a distinct best-iteration region |
| VP | `vp_3` | 7 | 0.69286093 | 19 | parameter-distant existing trial with a distinct best-iteration region |
| RP | `rp_1` | 20 | 0.69284666 | 14 | lowest existing development CV objective |
| RP | `rp_2` | 3 | 0.69312776 | 7 | parameter-distant existing trial with a distinct best-iteration region |
| RP | `rp_3` | 6 | 0.69306901 | 12 | parameter-distant existing trial with a distinct best-iteration region |

## Helper comparison

Scores use the production mean validation log loss plus 0.75 × population fold standard deviation. Lower is better.

| Family | Candidate | Helper | Objective | Mean log loss | Fold std | Best iter | Cap hit | CV seconds | Matrix MiB |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| RP | `rp_1` | `current` | 0.69284673 | 0.69267128 | 0.00023393 | 14 | no | 6.0 (+2.3 build) | 65.7 |
| RP | `rp_1` | `higher_capacity` | 0.69287198 | 0.69271156 | 0.00021389 | 13 | no | 7.2 (+2.3 build) | 65.7 |
| RP | `rp_1` | `higher_tree_cap` | 0.69284673 | 0.69267128 | 0.00023393 | 14 | no | 6.0 (+2.3 build) | 65.7 |
| RP | `rp_1` | `lower_lr_current_cap` | 0.69283512 | 0.69265394 | 0.00024158 | 34 | no | 8.2 (+2.3 build) | 65.7 |
| RP | `rp_1` | `lower_lr_higher_cap` | 0.69283512 | 0.69265394 | 0.00024158 | 34 | no | 8.2 (+2.3 build) | 65.7 |
| RP | `rp_1` | `stronger_regularization` | 0.69285273 | 0.69266658 | 0.00024820 | 15 | no | 5.9 (+2.3 build) | 65.7 |
| RP | `rp_2` | `current` | 0.69312776 | 0.69306467 | 0.00008412 | 7 | no | 5.2 (+13.5 build) | 65.7 |
| RP | `rp_2` | `higher_capacity` | 0.69314008 | 0.69309062 | 0.00006595 | 7 | no | 6.5 (+13.5 build) | 65.7 |
| RP | `rp_2` | `higher_tree_cap` | 0.69312776 | 0.69306467 | 0.00008412 | 7 | no | 5.1 (+13.5 build) | 65.7 |
| RP | `rp_2` | `lower_lr_current_cap` | 0.69311850 | 0.69307635 | 0.00005620 | 11 | no | 5.7 (+13.5 build) | 65.7 |
| RP | `rp_2` | `lower_lr_higher_cap` | 0.69311850 | 0.69307635 | 0.00005620 | 11 | no | 5.6 (+13.5 build) | 65.7 |
| RP | `rp_2` | `stronger_regularization` | 0.69310000 | 0.69303692 | 0.00008411 | 9 | no | 5.0 (+13.5 build) | 65.7 |
| RP | `rp_3` | `current` | 0.69306910 | 0.69297726 | 0.00012246 | 12 | no | 5.8 (+11.3 build) | 65.7 |
| RP | `rp_3` | `higher_capacity` | 0.69311035 | 0.69301756 | 0.00012372 | 8 | no | 6.8 (+11.3 build) | 65.7 |
| RP | `rp_3` | `higher_tree_cap` | 0.69306910 | 0.69297726 | 0.00012246 | 12 | no | 5.8 (+11.3 build) | 65.7 |
| RP | `rp_3` | `lower_lr_current_cap` | 0.69306511 | 0.69298182 | 0.00011105 | 19 | no | 6.8 (+11.3 build) | 65.7 |
| RP | `rp_3` | `lower_lr_higher_cap` | 0.69306511 | 0.69298182 | 0.00011105 | 19 | no | 6.6 (+11.3 build) | 65.7 |
| RP | `rp_3` | `stronger_regularization` | 0.69302637 | 0.69292057 | 0.00014106 | 19 | no | 6.1 (+11.3 build) | 65.7 |
| VP | `vp_1` | `current` | 0.69279827 | 0.69260709 | 0.00025490 | 14 | no | 8.5 (+1.3 build) | 122.0 |
| VP | `vp_1` | `higher_capacity` | 0.69292191 | 0.69270583 | 0.00028811 | 12 | no | 10.7 (+1.3 build) | 122.0 |
| VP | `vp_1` | `higher_tree_cap` | 0.69279827 | 0.69260709 | 0.00025490 | 14 | no | 9.0 (+1.3 build) | 122.0 |
| VP | `vp_1` | `lower_lr_current_cap` | 0.69283013 | 0.69262678 | 0.00027113 | 27 | no | 11.2 (+1.3 build) | 122.0 |
| VP | `vp_1` | `lower_lr_higher_cap` | 0.69283013 | 0.69262678 | 0.00027113 | 27 | no | 10.8 (+1.3 build) | 122.0 |
| VP | `vp_1` | `stronger_regularization` | 0.69283057 | 0.69260934 | 0.00029497 | 16 | no | 9.0 (+1.3 build) | 122.0 |
| VP | `vp_2` | `current` | 0.69313558 | 0.69306962 | 0.00008795 | 9 | no | 11.3 (+1.2 build) | 215.9 |
| VP | `vp_2` | `higher_capacity` | 0.69316373 | 0.69313725 | 0.00003531 | 1 | no | 11.4 (+1.2 build) | 215.9 |
| VP | `vp_2` | `higher_tree_cap` | 0.69313558 | 0.69306962 | 0.00008795 | 9 | no | 10.3 (+1.2 build) | 215.9 |
| VP | `vp_2` | `lower_lr_current_cap` | 0.69314192 | 0.69306731 | 0.00009948 | 20 | no | 13.1 (+1.2 build) | 215.9 |
| VP | `vp_2` | `lower_lr_higher_cap` | 0.69314192 | 0.69306731 | 0.00009948 | 20 | no | 13.6 (+1.2 build) | 215.9 |
| VP | `vp_2` | `stronger_regularization` | 0.69311193 | 0.69303125 | 0.00010758 | 14 | no | 11.0 (+1.2 build) | 215.9 |
| VP | `vp_3` | `current` | 0.69286093 | 0.69267550 | 0.00024724 | 19 | no | 13.5 (+1.2 build) | 215.9 |
| VP | `vp_3` | `higher_capacity` | 0.69292015 | 0.69279537 | 0.00016638 | 9 | no | 15.8 (+1.2 build) | 215.9 |
| VP | `vp_3` | `higher_tree_cap` | 0.69286093 | 0.69267550 | 0.00024724 | 19 | no | 13.2 (+1.2 build) | 215.9 |
| VP | `vp_3` | `lower_lr_current_cap` | 0.69285626 | 0.69268423 | 0.00022937 | 32 | no | 17.4 (+1.2 build) | 215.9 |
| VP | `vp_3` | `lower_lr_higher_cap` | 0.69285626 | 0.69268423 | 0.00022937 | 32 | no | 17.0 (+1.2 build) | 215.9 |
| VP | `vp_3` | `stronger_regularization` | 0.69283616 | 0.69264216 | 0.00025867 | 20 | no | 13.9 (+1.2 build) | 215.9 |

## Ranking agreement

### VP

| Helper variant | Spearman vs current | Top-1 same | Top-2 overlap | Best score delta vs current |
|---|---:|:---:|---:|---:|
| `current` | 1.000 | yes | 1.00 | +0.00000000 |
| `higher_tree_cap` | 1.000 | yes | 1.00 | +0.00000000 |
| `lower_lr_current_cap` | 1.000 | yes | 1.00 | +0.00003186 |
| `lower_lr_higher_cap` | 1.000 | yes | 1.00 | +0.00003186 |
| `higher_capacity` | 0.500 | no | 1.00 | +0.00012189 |
| `stronger_regularization` | 1.000 | yes | 1.00 | +0.00003230 |

### RP

| Helper variant | Spearman vs current | Top-1 same | Top-2 overlap | Best score delta vs current |
|---|---:|:---:|---:|---:|
| `current` | 1.000 | yes | 1.00 | +0.00000000 |
| `higher_tree_cap` | 1.000 | yes | 1.00 | +0.00000000 |
| `lower_lr_current_cap` | 1.000 | yes | 1.00 | -0.00001160 |
| `lower_lr_higher_cap` | 1.000 | yes | 1.00 | -0.00001160 |
| `higher_capacity` | 1.000 | yes | 1.00 | +0.00002525 |
| `stronger_regularization` | 1.000 | yes | 1.00 | +0.00000601 |

## Resource and interpretation

- Pilot elapsed time so far: 370.3s (6.2 min); completed comparisons: 36/36.
- Peak process working set: 2596569088 bytes; maximum candidate matrix: 215.9 MiB.
- Three configurations per family make rank correlations coarse. This pilot can flag ranking sensitivity; it cannot establish that the current helper ranks all possible generators reliably.
- Existing source-trial CV curves were used only to preselect development candidates. All reported comparison scores were recomputed with the aligned callback and the frozen helper variants.
