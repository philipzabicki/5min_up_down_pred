# RAW BASE70 seed stability and five-model ensemble - 2026-10-18

## Conclusion

**Keep SINGLE as the research baseline.**

- Probability quality: The official sample does not show a clear, interval-supported improvement across probability scores.
- After-cost economics: The after-cost evidence does not clear a paired, directional improvement for both daily PnL and PnL/turnover.
- SINGLE was retained exactly from its original 20261015 OOF and fold boosters. The four new seeds used 31 rounds on the same ten folds, common rows, target, sample weights, purge, and feature order.
- The ensemble is the arithmetic mean of the five saved raw probabilities. It uses no calibration, learned weights, seed selection, or prediction-dispersion filter.
- This is conditional on the previously selected features, hyperparameters, and 31 rounds; it is not independent validation of the whole model-selection process. Historical development outcomes have been examined previously.

## Frozen comparison and seed behavior

- Shared BASE70 matrix: 3,174,805 rows, 70 features, ten saved chronological folds. All 40 new fits were exactly 31 iterations with no early stopping.
- Target: `target_preopen_5m_up_binance_index_proxy`; decision-time validation rows and label-availability purge match the source folds. Sample weights hash: `3a21a8cfecaeb10cdf0af1e63fd9833eef034954f6f530005bab71da6f1397ca`.
- Original SINGLE parameters and boosters were reused. For new fits, `feature_fraction_seed` controls per-tree feature subsampling; `data_random_seed` controls the dataset's histogram-bin sampling and the dataset was rebuilt once per new seed with that seed. `bagging_seed` is set but inactive because bagging remains off (`bagging_fraction=1`, `bagging_freq=0`). The explicit child seeds override the root `seed` for their respective sources. `extra_seed`, `drop_seed`, and `objective_seed` remain inactive under the unchanged gbdt/binary settings. These LightGBM 4.6.0 definitions are documented in the [official parameter reference](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html).
- The full effective fit parameters for SINGLE and each new seed were frozen in the versioned manifest before training. The source control was not retrained to normalize parameters.

### Single-seed quality range

The observed seed range is sensitivity to training randomness, not a confidence interval for true probabilities.

| Sample | Metric | SINGLE | Min | Median | Max |
| --- | --- | --- | --- | --- | --- |
| official | auc | 0.527437 | 0.523827 | 0.525139 | 0.527437 |
| official | brier | 0.249453 | 0.249453 | 0.249553 | 0.249599 |
| official | log_loss | 0.692052 | 0.692052 | 0.692254 | 0.692344 |
| proxy_oof | auc | 0.525079 | 0.524764 | 0.524922 | 0.525203 |
| proxy_oof | brier | 0.249494 | 0.249494 | 0.249502 | 0.249505 |
| proxy_oof | log_loss | 0.692134 | 0.692134 | 0.692150 | 0.692157 |

### Probability quality: overall

Lower log loss and Brier are better; higher AUC is better. The official sample exactly matches the prior 20,718 labeled rows. Proxy OOF contains every common validation row across all ten folds.

| Sample | Model | Rows | Log loss | Brier | AUC |
| --- | --- | --- | --- | --- | --- |
| official | SINGLE | 20718 | 0.692052 | 0.249453 | 0.5274 |
| official | SEED_20261016 | 20718 | 0.692254 | 0.249553 | 0.5251 |
| official | SEED_20261017 | 20718 | 0.692112 | 0.249483 | 0.5272 |
| official | SEED_20261018 | 20718 | 0.692344 | 0.249599 | 0.5238 |
| official | SEED_20261019 | 20718 | 0.692255 | 0.249554 | 0.5247 |
| official | ENSEMBLE | 20718 | 0.692154 | 0.249503 | 0.5265 |
| proxy_oof | SINGLE | 318560 | 0.692134 | 0.249494 | 0.5251 |
| proxy_oof | SEED_20261016 | 318560 | 0.692157 | 0.249505 | 0.5248 |
| proxy_oof | SEED_20261017 | 318560 | 0.692151 | 0.249502 | 0.5249 |
| proxy_oof | SEED_20261018 | 318560 | 0.692150 | 0.249502 | 0.5252 |
| proxy_oof | SEED_20261019 | 318560 | 0.692144 | 0.249499 | 0.5249 |
| proxy_oof | ENSEMBLE | 318560 | 0.692097 | 0.249475 | 0.5255 |

### Probability quality: fold metrics

The table contains every model's log loss, Brier, and AUC for each non-empty fold in the proxy and official samples. The complete machine-readable table is saved locally in `fold_quality.csv`.

| Sample | Fold | Model | Rows | Log loss | Brier | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| official | 9 | SINGLE | 20718 | 0.692052 | 0.249453 | 0.5274 |
| official | 9 | SEED_20261016 | 20718 | 0.692254 | 0.249553 | 0.5251 |
| official | 9 | SEED_20261017 | 20718 | 0.692112 | 0.249483 | 0.5272 |
| official | 9 | SEED_20261018 | 20718 | 0.692344 | 0.249599 | 0.5238 |
| official | 9 | SEED_20261019 | 20718 | 0.692255 | 0.249554 | 0.5247 |
| official | 9 | ENSEMBLE | 20718 | 0.692154 | 0.249503 | 0.5265 |
| proxy_oof | 0 | SINGLE | 31858 | 0.691638 | 0.249246 | 0.5308 |
| proxy_oof | 0 | SEED_20261016 | 31858 | 0.691580 | 0.249218 | 0.5305 |
| proxy_oof | 0 | SEED_20261017 | 31858 | 0.691669 | 0.249262 | 0.5302 |
| proxy_oof | 0 | SEED_20261018 | 31858 | 0.691630 | 0.249243 | 0.5306 |
| proxy_oof | 0 | SEED_20261019 | 31858 | 0.691713 | 0.249284 | 0.5297 |
| proxy_oof | 0 | ENSEMBLE | 31858 | 0.691595 | 0.249226 | 0.5308 |
| proxy_oof | 1 | SINGLE | 31841 | 0.692130 | 0.249492 | 0.5266 |
| proxy_oof | 1 | SEED_20261016 | 31841 | 0.692236 | 0.249545 | 0.5254 |
| proxy_oof | 1 | SEED_20261017 | 31841 | 0.692250 | 0.249552 | 0.5249 |
| proxy_oof | 1 | SEED_20261018 | 31841 | 0.692292 | 0.249572 | 0.5254 |
| proxy_oof | 1 | SEED_20261019 | 31841 | 0.692227 | 0.249540 | 0.5257 |
| proxy_oof | 1 | ENSEMBLE | 31841 | 0.692176 | 0.249515 | 0.5260 |
| proxy_oof | 2 | SINGLE | 31862 | 0.692164 | 0.249509 | 0.5263 |
| proxy_oof | 2 | SEED_20261016 | 31862 | 0.692312 | 0.249583 | 0.5249 |
| proxy_oof | 2 | SEED_20261017 | 31862 | 0.692196 | 0.249525 | 0.5259 |
| proxy_oof | 2 | SEED_20261018 | 31862 | 0.692226 | 0.249540 | 0.5261 |
| proxy_oof | 2 | SEED_20261019 | 31862 | 0.692171 | 0.249513 | 0.5262 |
| proxy_oof | 2 | ENSEMBLE | 31862 | 0.692164 | 0.249509 | 0.5262 |
| proxy_oof | 3 | SINGLE | 31861 | 0.691798 | 0.249326 | 0.5295 |
| proxy_oof | 3 | SEED_20261016 | 31861 | 0.691706 | 0.249280 | 0.5311 |
| proxy_oof | 3 | SEED_20261017 | 31861 | 0.691826 | 0.249340 | 0.5297 |
| proxy_oof | 3 | SEED_20261018 | 31861 | 0.691775 | 0.249314 | 0.5306 |
| proxy_oof | 3 | SEED_20261019 | 31861 | 0.691674 | 0.249264 | 0.5310 |
| proxy_oof | 3 | ENSEMBLE | 31861 | 0.691706 | 0.249280 | 0.5309 |
| proxy_oof | 4 | SINGLE | 31842 | 0.692174 | 0.249514 | 0.5247 |
| proxy_oof | 4 | SEED_20261016 | 31842 | 0.692155 | 0.249505 | 0.5250 |
| proxy_oof | 4 | SEED_20261017 | 31842 | 0.692240 | 0.249547 | 0.5243 |
| proxy_oof | 4 | SEED_20261018 | 31842 | 0.692153 | 0.249503 | 0.5254 |
| proxy_oof | 4 | SEED_20261019 | 31842 | 0.692168 | 0.249511 | 0.5251 |
| proxy_oof | 4 | ENSEMBLE | 31842 | 0.692127 | 0.249490 | 0.5253 |
| proxy_oof | 5 | SINGLE | 31856 | 0.692029 | 0.249442 | 0.5262 |
| proxy_oof | 5 | SEED_20261016 | 31856 | 0.692058 | 0.249456 | 0.5261 |
| proxy_oof | 5 | SEED_20261017 | 31856 | 0.692023 | 0.249439 | 0.5265 |
| proxy_oof | 5 | SEED_20261018 | 31856 | 0.691861 | 0.249358 | 0.5288 |
| proxy_oof | 5 | SEED_20261019 | 31856 | 0.691944 | 0.249400 | 0.5271 |
| proxy_oof | 5 | ENSEMBLE | 31856 | 0.691933 | 0.249394 | 0.5275 |
| proxy_oof | 6 | SINGLE | 31861 | 0.692722 | 0.249787 | 0.5170 |
| proxy_oof | 6 | SEED_20261016 | 31861 | 0.692698 | 0.249776 | 0.5168 |
| proxy_oof | 6 | SEED_20261017 | 31861 | 0.692703 | 0.249778 | 0.5167 |
| proxy_oof | 6 | SEED_20261018 | 31861 | 0.692678 | 0.249766 | 0.5170 |
| proxy_oof | 6 | SEED_20261019 | 31861 | 0.692726 | 0.249790 | 0.5164 |
| proxy_oof | 6 | ENSEMBLE | 31861 | 0.692655 | 0.249754 | 0.5172 |
| proxy_oof | 7 | SINGLE | 31858 | 0.692884 | 0.249868 | 0.5143 |
| proxy_oof | 7 | SEED_20261016 | 31858 | 0.692907 | 0.249880 | 0.5154 |
| proxy_oof | 7 | SEED_20261017 | 31858 | 0.692808 | 0.249831 | 0.5162 |
| proxy_oof | 7 | SEED_20261018 | 31858 | 0.692874 | 0.249863 | 0.5160 |
| proxy_oof | 7 | SEED_20261019 | 31858 | 0.692872 | 0.249862 | 0.5157 |
| proxy_oof | 7 | ENSEMBLE | 31858 | 0.692820 | 0.249837 | 0.5162 |
| proxy_oof | 8 | SINGLE | 31860 | 0.691997 | 0.249426 | 0.5262 |
| proxy_oof | 8 | SEED_20261016 | 31860 | 0.692087 | 0.249471 | 0.5242 |
| proxy_oof | 8 | SEED_20261017 | 31860 | 0.692011 | 0.249433 | 0.5257 |
| proxy_oof | 8 | SEED_20261018 | 31860 | 0.692032 | 0.249443 | 0.5262 |
| proxy_oof | 8 | SEED_20261019 | 31860 | 0.692046 | 0.249450 | 0.5257 |
| proxy_oof | 8 | ENSEMBLE | 31860 | 0.691983 | 0.249419 | 0.5263 |
| proxy_oof | 9 | SINGLE | 31861 | 0.691807 | 0.249330 | 0.5305 |
| proxy_oof | 9 | SEED_20261016 | 31861 | 0.691830 | 0.249342 | 0.5295 |
| proxy_oof | 9 | SEED_20261017 | 31861 | 0.691785 | 0.249320 | 0.5307 |
| proxy_oof | 9 | SEED_20261018 | 31861 | 0.691977 | 0.249415 | 0.5277 |
| proxy_oof | 9 | SEED_20261019 | 31861 | 0.691896 | 0.249375 | 0.5285 |
| proxy_oof | 9 | ENSEMBLE | 31861 | 0.691809 | 0.249331 | 0.5302 |

### Official monthly quality

| UTC month | Model | Rows | Log loss | Brier | AUC |
| --- | --- | --- | --- | --- | --- |
| 2026-04 | SINGLE | 3168 | 0.691610 | 0.249233 | 0.5325 |
| 2026-04 | SEED_20261016 | 3168 | 0.691772 | 0.249314 | 0.5284 |
| 2026-04 | SEED_20261017 | 3168 | 0.691667 | 0.249260 | 0.5330 |
| 2026-04 | SEED_20261018 | 3168 | 0.692022 | 0.249438 | 0.5262 |
| 2026-04 | SEED_20261019 | 3168 | 0.691806 | 0.249330 | 0.5300 |
| 2026-04 | ENSEMBLE | 3168 | 0.691726 | 0.249290 | 0.5308 |
| 2026-05 | SINGLE | 8917 | 0.692259 | 0.249556 | 0.5246 |
| 2026-05 | SEED_20261016 | 8917 | 0.692494 | 0.249674 | 0.5210 |
| 2026-05 | SEED_20261017 | 8917 | 0.692276 | 0.249565 | 0.5240 |
| 2026-05 | SEED_20261018 | 8917 | 0.692449 | 0.249651 | 0.5218 |
| 2026-05 | SEED_20261019 | 8917 | 0.692398 | 0.249626 | 0.5209 |
| 2026-05 | ENSEMBLE | 8917 | 0.692325 | 0.249589 | 0.5232 |
| 2026-06 | SINGLE | 8633 | 0.692000 | 0.249427 | 0.5296 |
| 2026-06 | SEED_20261016 | 8633 | 0.692182 | 0.249518 | 0.5286 |
| 2026-06 | SEED_20261017 | 8633 | 0.692106 | 0.249479 | 0.5293 |
| 2026-06 | SEED_20261018 | 8633 | 0.692354 | 0.249603 | 0.5259 |
| 2026-06 | SEED_20261019 | 8633 | 0.692271 | 0.249562 | 0.5271 |
| 2026-06 | ENSEMBLE | 8633 | 0.692133 | 0.249493 | 0.5289 |

### Paired quality bootstrap: ENSEMBLE minus SINGLE

Negative log loss/Brier differences favor ENSEMBLE; positive AUC differences favor ENSEMBLE. Draws use shared circular 3-day UTC blocks, 2,000 replications, seed 20261018.

| Sample | Metric | Point difference | 95% interval | Valid draws |
| --- | --- | --- | --- | --- |
| official | auc | -0.000959 | -0.002904 to 0.000988 | 2000 |
| official | brier | 0.000051 | -0.000012 to 0.000109 | 2000 |
| official | log_loss | 0.000102 | -0.000024 to 0.000219 | 2000 |
| proxy_oof | auc | 0.000389 | -0.000011 to 0.000795 | 2000 |
| proxy_oof | brier | -0.000019 | -0.000035 to -0.000002 | 2000 |
| proxy_oof | log_loss | -0.000037 | -0.000069 to -0.000004 | 2000 |

### Seed predictions and dispersion

Pearson correlations use all common proxy OOF rows.

| Seed | SINGLE | SEED_20261016 | SEED_20261017 | SEED_20261018 | SEED_20261019 |
| --- | --- | --- | --- | --- | --- |
| SINGLE | 1.00000 | 0.95188 | 0.95278 | 0.95269 | 0.95149 |
| SEED_20261016 | 0.95188 | 1.00000 | 0.95092 | 0.95120 | 0.95147 |
| SEED_20261017 | 0.95278 | 0.95092 | 1.00000 | 0.95127 | 0.95153 |
| SEED_20261018 | 0.95269 | 0.95120 | 0.95127 | 1.00000 | 0.95205 |
| SEED_20261019 | 0.95149 | 0.95147 | 0.95153 | 0.95205 | 1.00000 |

| Sample | Rows | Mean SD | SD p25 | SD p50 | SD p75 | SD p90 | SD p95 | SD p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official | 20718 | 0.004670 | 0.003415 | 0.004542 | 0.005774 | 0.006963 | 0.007718 | 0.009168 |
| proxy_oof | 318560 | 0.004698 | 0.003442 | 0.004564 | 0.005805 | 0.007003 | 0.007750 | 0.009242 |

The standard deviation is across the five raw component probabilities for each market. Ensemble quality relative to the arithmetic mean of the five component scores:

| Sample | Metric | Mean component score | Ensemble score | Ensemble minus component mean |
| --- | --- | --- | --- | --- |
| official | auc | 0.525653 | 0.526478 | 0.000825 |
| official | brier | 0.249528 | 0.249503 | -0.000025 |
| official | log_loss | 0.692203 | 0.692154 | -0.000050 |
| proxy_oof | auc | 0.524976 | 0.525469 | 0.000493 |
| proxy_oof | brier | 0.249501 | 0.249475 | -0.000025 |
| proxy_oof | log_loss | 0.692147 | 0.692097 | -0.000050 |

## Independent-order economics

The same 20,735 decisions over 72 UTC days were replayed after the prior Platt warm-up. Each model independently selected its side and orders at the same $5 gross amount, 1% net-EV hurdle, T-60 decision and T-59 execution, with the same snapshots, fees, depth, outcome cutoff, and accounting engine. Raw predictions were used without Platt. No common-fill filter was applied; a lower trade count alone is not treated as an improvement.

SINGLE reproduced the previous RAW anchor: 1,038 fills, $5,190.00 turnover, and $267.44 net PnL. All six accounting audits passed.

| Variant | Intents | Fills | Settled | Turnover | Fees | Net PnL | PnL/turnover | EV on settled fills | PnL minus EV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SINGLE | 1047 | 1038 | 1038 | $5,190.00 | $188.84 | $267.44 | 5.15% | $279.60 | $-12.17 |
| SEED_20261016 | 1078 | 1067 | 1067 | $5,335.00 | $193.65 | $134.09 | 2.51% | $284.59 | $-150.50 |
| SEED_20261017 | 1113 | 1101 | 1101 | $5,505.00 | $199.57 | $52.27 | 0.95% | $284.01 | $-231.74 |
| SEED_20261018 | 1080 | 1073 | 1073 | $5,365.00 | $195.11 | $46.13 | 0.86% | $281.73 | $-235.60 |
| SEED_20261019 | 1025 | 1015 | 1015 | $5,075.00 | $184.86 | $-64.27 | -1.27% | $275.59 | $-339.85 |
| ENSEMBLE | 968 | 960 | 960 | $4,800.00 | $175.10 | $130.41 | 2.72% | $265.31 | $-134.91 |

### Monthly economics

| UTC month | Variant | Fills | Turnover | Fees | Net PnL | PnL/turnover | EV | PnL minus EV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04 | SINGLE | 63 | $315.00 | $11.45 | $-10.39 | -3.30% | $9.40 | $-19.78 |
| 2026-05 | SINGLE | 447 | $2,235.00 | $83.85 | $66.12 | 2.96% | $177.30 | $-111.18 |
| 2026-06 | SINGLE | 528 | $2,640.00 | $93.54 | $211.71 | 8.02% | $92.91 | $118.80 |
| 2026-04 | SEED_20261016 | 65 | $325.00 | $11.76 | $-2.57 | -0.79% | $10.37 | $-12.94 |
| 2026-05 | SEED_20261016 | 458 | $2,290.00 | $85.74 | $22.23 | 0.97% | $178.71 | $-156.47 |
| 2026-06 | SEED_20261016 | 544 | $2,720.00 | $96.14 | $114.43 | 4.21% | $95.51 | $18.91 |
| 2026-04 | SEED_20261017 | 66 | $330.00 | $11.95 | $-6.63 | -2.01% | $10.23 | $-16.86 |
| 2026-05 | SEED_20261017 | 466 | $2,330.00 | $86.99 | $0.28 | 0.01% | $177.26 | $-176.98 |
| 2026-06 | SEED_20261017 | 569 | $2,845.00 | $100.64 | $58.61 | 2.06% | $96.52 | $-37.91 |
| 2026-04 | SEED_20261018 | 72 | $360.00 | $13.05 | $-65.24 | -18.12% | $11.11 | $-76.34 |
| 2026-05 | SEED_20261018 | 481 | $2,405.00 | $89.85 | $-46.95 | -1.95% | $179.17 | $-226.11 |
| 2026-06 | SEED_20261018 | 520 | $2,600.00 | $92.20 | $158.31 | 6.09% | $91.46 | $66.85 |
| 2026-04 | SEED_20261019 | 65 | $325.00 | $11.78 | $-30.95 | -9.52% | $10.05 | $-41.01 |
| 2026-05 | SEED_20261019 | 433 | $2,165.00 | $81.30 | $-108.61 | -5.02% | $175.15 | $-283.77 |
| 2026-06 | SEED_20261019 | 517 | $2,585.00 | $91.79 | $75.30 | 2.91% | $90.38 | $-15.08 |
| 2026-04 | ENSEMBLE | 60 | $300.00 | $10.88 | $-25.05 | -8.35% | $9.22 | $-34.27 |
| 2026-05 | ENSEMBLE | 413 | $2,065.00 | $77.79 | $-29.58 | -1.43% | $170.73 | $-200.31 |
| 2026-06 | ENSEMBLE | 487 | $2,435.00 | $86.42 | $185.04 | 7.60% | $85.36 | $99.68 |

### UP and DOWN outcomes

| Variant | Side | Fills | Settled | Turnover | Fees | Net PnL | EV | PnL minus EV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SINGLE | UP | 493 | 493 | $2,465.00 | $88.80 | $183.41 | $134.64 | $48.77 |
| SINGLE | DOWN | 545 | 545 | $2,725.00 | $100.05 | $84.03 | $144.97 | $-60.94 |
| SEED_20261016 | UP | 499 | 499 | $2,495.00 | $89.74 | $178.89 | $132.87 | $46.02 |
| SEED_20261016 | DOWN | 568 | 568 | $2,840.00 | $103.90 | $-44.80 | $151.72 | $-196.52 |
| SEED_20261017 | UP | 555 | 555 | $2,775.00 | $99.56 | $49.33 | $139.48 | $-90.16 |
| SEED_20261017 | DOWN | 546 | 546 | $2,730.00 | $100.01 | $2.94 | $144.53 | $-141.59 |
| SEED_20261018 | UP | 525 | 525 | $2,625.00 | $94.47 | $137.22 | $135.84 | $1.38 |
| SEED_20261018 | DOWN | 548 | 548 | $2,740.00 | $100.64 | $-91.09 | $145.89 | $-236.99 |
| SEED_20261019 | UP | 483 | 483 | $2,415.00 | $87.18 | $76.95 | $133.14 | $-56.19 |
| SEED_20261019 | DOWN | 532 | 532 | $2,660.00 | $97.68 | $-141.22 | $142.44 | $-283.66 |
| ENSEMBLE | UP | 469 | 469 | $2,345.00 | $84.73 | $151.06 | $128.23 | $22.83 |
| ENSEMBLE | DOWN | 491 | 491 | $2,455.00 | $90.37 | $-20.65 | $137.08 | $-157.73 |

### Concentration and paired economic uncertainty

| Variant | Top-five-day PnL | Share of total PnL | Other days PnL |
| --- | --- | --- | --- |
| ENSEMBLE | $196.43 | 150.6% | $-66.03 |
| SEED_20261016 | $173.24 | 129.2% | $-39.15 |
| SEED_20261017 | $197.28 | 377.4% | $-145.01 |
| SEED_20261018 | $216.83 | 470.0% | $-170.70 |
| SEED_20261019 | $147.07 | -228.8% | $-211.34 |
| SINGLE | $218.21 | 81.6% | $49.23 |

- ENSEMBLE minus SINGLE mean daily PnL: $-1.9032; 95% paired interval $-3.9185 to $0.0148.
- ENSEMBLE minus SINGLE PnL/settled-turnover: -2.44%; 95% paired interval -6.52% to 0.37% (2000/2000 valid draws).
- PnL difference decomposition: same-side fills $0.00; fills present in one variant $-137.03; opposite sides $0.00. Components reconcile to the full settled PnL difference.

## Runtime, artifacts, and limits

- Elapsed seconds by stage: `{"common_predictions": 0.68, "economics_ENSEMBLE": 23.0, "economics_SEED_20261016": 20.16, "economics_SEED_20261017": 19.8, "economics_SEED_20261018": 21.83, "economics_SEED_20261019": 22.72, "economics_SINGLE": 21.52, "economics_summary": 130.35, "execution_inputs": 5.62, "quality": 51.46, "report": 0.0, "source_verification": 2.52, "training": 84.57}`.
- Peak process working set: 5.046 GiB. New-fit cumulative seconds: 70.7; quality/bootstrap: 51.5; replay: 130.4.
- Matrix and folds were loaded once for this training pass. Per-fold models, OOF checkpoints, trades and decisions are kept locally under `data/analysis/polymarket/BTC/btc_base70_seed_stability_20261018`; only the compact config, report, versioned manifest, runner and tests are versioned.
- The development sample and feature/model settings were selected previously. These results do not establish future or live performance. No holdout was scored, no deployment refit was run, and no live artifacts were changed.

The runner is `python run_btc_base70_seed_stability_20261018.py`; no command-line arguments are required.
