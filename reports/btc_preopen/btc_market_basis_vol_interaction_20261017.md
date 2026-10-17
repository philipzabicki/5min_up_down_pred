# BTC-market interaction correction — 2026-10-17

> The July–October holdout had been examined in earlier repository work. Its result here is descriptive and was not used for feature, configuration, or policy selection.

## Question and protocol

M0 is a market-offset correction using the market logit, BTC index-futures basis, BTC volatility, and both T-60 spreads. M0 therefore retains BTC information through basis and volatility. M1 adds only `d = logit(p_BTC_A_frozen) − logit(p_market)`. The primary comparison is M1−M0.

Both models use `p = sigmoid(logit(p_market) + f(X))`, a row-aligned LightGBM `init_score`, official Polymarket outcomes, binary log-loss, 100 iterations, and the same complete rows and purged May/June folds. The fitted tree correction is raw margin; the market offset is added exactly once at prediction. No weights, monotonic constraints, early stopping, Optuna, generator rerun, BTC model/OOF refit, data fetch, or archive replay were used.

## Inputs and freeze

Common prediction rows: 8,675 development OOF markets and 24,735 holdout markets. Development starts are before 2026-07-01T00:00:00+00:00; holdout starts are before 2026-10-07T00:00:00+00:00. All predictions and quality metrics use rows shared with the previous frozen D table.

The pre-holdout manifest was frozen at `2026-10-17T05:06:33.148595+00:00` before holdout prediction scoring. Manifest SHA-256: `78d745411b8fe7678cf85a95cce46f0827d3b6d0ff126087cb4880efa249e88d`. Input/code identities and winning settings are recorded in `data/analysis/polymarket/BTC/btc_market_basis_vol_interaction_20261017/pre_holdout_manifest.json`.

Reporting resumed at `8ebcc3bc3b090cd074d5d5acdcba8692ea66455c447eb7314f7bc7ce47685031` after the first report attempt hit empty-only June calendar bootstrap draws. Each empty-only draw was discarded; all comparisons then used 2,000 valid replicates. The frozen holdout predictions were reused from their one-time score at `2026-10-17T05:06:33.384641+00:00`, with the original final-model and prediction hashes intact. The frozen manifest retains the original runner hash.

BTC probabilities came from the existing causal A_frozen development OOF predictions and frozen holdout predictions, validated against the saved walk-forward manifest and hashes in the prior experiment config. They were not regenerated or replaced. The previous D comparator uses the saved development and holdout prediction tables; their hashes are recorded in the pre-holdout manifest.

### Frozen periods and folds

| Period/fold | Markets or train rows | Validation rows | First market | Last market | Latest eligible train-label release |
|---|---:|---:|---|---|---|
| development | 10,794 | — | 2026-04-15T17:05:00+00:00 | 2026-06-30T23:50:00+00:00 | 2026-06-30T23:56:22+00:00 |
| holdout | 24,735 | — | 2026-07-01T00:00:00+00:00 | 2026-10-06T23:50:00+00:00 | 2026-10-06T23:56:53+00:00 |
| dev_2026q2_b | 2,119 | 5,253 | 2026-05-01T00:00:00+00:00 | 2026-06-01T00:00:00+00:00 | 2026-04-30T23:56:52+00:00 |
| dev_2026q2_c | 7,371 | 3,422 | 2026-06-01T00:00:00+00:00 | 2026-07-01T00:00:00+00:00 | 2026-05-31T23:56:15+00:00 |

The May and June validation folds use the frozen monthly boundaries. Training requires `market_start_utc < validation_start_utc` and `label_available_at_utc <= validation_start_utc`; all variants share the same complete-input mask.

## Development configuration results

Selection minimizes mean fold log-loss + 0.5 × population SD. Every configuration is shown; scores are development results after configuration selection.

| Model | Leaves | Depth | L2 | May log-loss | June log-loss | Mean + 0.5 SD | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| M0 | 4 | 2 | 100 | 0.69380422 | 0.69359747 | 0.69375254 | yes |
| M0 | 4 | 2 | 10 | 0.69555419 | 0.69341517 | 0.69501943 |  |
| M0 | 8 | 3 | 100 | 0.69479511 | 0.69363218 | 0.69450438 |  |
| M0 | 8 | 3 | 10 | 0.69711537 | 0.69315541 | 0.69612538 |  |
| M1 | 4 | 2 | 100 | 0.69410097 | 0.69234759 | 0.69366263 | yes |
| M1 | 4 | 2 | 10 | 0.69476166 | 0.69227690 | 0.69414047 |  |
| M1 | 8 | 3 | 100 | 0.69405141 | 0.69326349 | 0.69385443 |  |
| M1 | 8 | 3 | 10 | 0.69684533 | 0.69335093 | 0.69597173 |  |

## Probability quality

Negative paired differences favor the left-hand model. Intervals use paired moving 3-day UTC blocks, 2,000 replicates, seed 20261017. The selected development folds are not independent test results; their intervals are conditional on the selected predictions and do not remove selection optimism.

### dev_2026q2_b

| Model | n | Log-loss | Brier |
|---|---:|---:|---:|
| p_market | 5,253 | 0.69282027 | 0.24983451 |
| previous_D | 5,253 | 0.69273111 | 0.24978981 |
| M0 | 5,253 | 0.69380422 | 0.25032739 |
| M1 | 5,253 | 0.69410097 | 0.25047526 |

| Paired difference | Δ log-loss | 95% block CI | Δ Brier | 95% block CI |
|---|---:|---:|---:|---:|
| M1-M0 | +0.00029675 | [-0.00109858, +0.00101012] | +0.00014787 | [-0.00054474, +0.00050224] |
| M1-p_market | +0.00128070 | [-0.00033095, +0.00261492] | +0.00064076 | [-0.00016311, +0.00130870] |
| M1-previous_D | +0.00136986 | [-0.00028998, +0.00261952] | +0.00068545 | [-0.00014320, +0.00130878] |

### dev_2026q2_c

| Model | n | Log-loss | Brier |
|---|---:|---:|---:|
| p_market | 3,422 | 0.69324384 | 0.25004331 |
| previous_D | 3,422 | 0.69269216 | 0.24976948 |
| M0 | 3,422 | 0.69359747 | 0.25021879 |
| M1 | 3,422 | 0.69234759 | 0.24959530 |

| Paired difference | Δ log-loss | 95% block CI | Δ Brier | 95% block CI |
|---|---:|---:|---:|---:|
| M1-M0 | -0.00124988 | [-0.00273261, -0.00008894] | -0.00062349 | [-0.00135894, -0.00004565] |
| M1-p_market | -0.00089625 | [-0.00265446, +0.00045985] | -0.00044801 | [-0.00132416, +0.00022837] |
| M1-previous_D | -0.00034457 | [-0.00124704, +0.00023526] | -0.00017418 | [-0.00062521, +0.00011514] |

### development_cv_selected

| Model | n | Log-loss | Brier |
|---|---:|---:|---:|
| p_market | 8,675 | 0.69298735 | 0.24991687 |
| previous_D | 8,675 | 0.69271574 | 0.24978179 |
| M0 | 8,675 | 0.69372267 | 0.25028455 |
| M1 | 8,675 | 0.69340932 | 0.25012815 |

| Paired difference | Δ log-loss | 95% block CI | Δ Brier | 95% block CI |
|---|---:|---:|---:|---:|
| M1-M0 | -0.00031335 | [-0.00141510, +0.00095350] | -0.00015640 | [-0.00070540, +0.00047347] |
| M1-p_market | +0.00042197 | [-0.00091346, +0.00182818] | +0.00021127 | [-0.00045514, +0.00091480] |
| M1-previous_D | +0.00069358 | [-0.00034598, +0.00183334] | +0.00034635 | [-0.00017266, +0.00091674] |

### holdout_descriptive

| Model | n | Log-loss | Brier |
|---|---:|---:|---:|
| p_market | 24,735 | 0.69158191 | 0.24923384 |
| previous_D | 24,735 | 0.69119693 | 0.24903925 |
| M0 | 24,735 | 0.69168580 | 0.24928509 |
| M1 | 24,735 | 0.69136706 | 0.24912258 |

| Paired difference | Δ log-loss | 95% block CI | Δ Brier | 95% block CI |
|---|---:|---:|---:|---:|
| M1-M0 | -0.00031874 | [-0.00089345, +0.00011986] | -0.00016251 | [-0.00044720, +0.00005764] |
| M1-p_market | -0.00021485 | [-0.00085901, +0.00032098] | -0.00011126 | [-0.00043455, +0.00015658] |
| M1-previous_D | +0.00017013 | [-0.00016291, +0.00044914] | +0.00008333 | [-0.00008166, +0.00022188] |

## Fixed-$5 economics

Decision and EV eligibility use the causal T-60 snapshot and existing fee logic. A qualified order is executed from the actual T-59 snapshot using existing depth/full-book or top-of-book proxy behavior. PnL uses fills settled by the period cutoff; fees are already included in cash debit and are not deducted from EV a second time.

### development_cv_selected (settlement cutoff 2026-07-01T00:00:00+00:00)

| Model | Priceable at T-60 | Positive-EV markets | Positive-EV sides | EV ≥ 1% | T-59 fills | Settled | Turnover | Fees | Net PnL | PnL/turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 7,409 | 1,740 | 1,740 | 1,700 | 1,679 | 1,679 | $8,395.00 | $291.5516 | $-249.5610 | -2.973% |
| M1 | 7,409 | 2,268 | 2,268 | 1,841 | 1,823 | 1,823 | $9,115.00 | $320.5370 | $-199.6242 | -2.190% |
| p_market | 7,409 | 0 | 0 | 0 | 0 | 0 | $0.00 | $0.0000 | $+0.0000 | — |
| previous_D | 7,409 | 490 | 490 | 131 | 131 | 131 | $655.00 | $23.0502 | $+17.5218 | 2.675% |

Resolved fills by execution depth:

| Model | Depth source | Fills | Shares | Turnover | Fees | Wins | Losses | Net PnL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | full_book | 1679 | 16608.9319 | $8395.00 | $291.5516 | 854 | 825 | $-249.5610 |
| M1 | full_book | 1823 | 18285.9494 | $9115.00 | $320.5370 | 923 | 900 | $-199.6242 |
| previous_D | full_book | 131 | 1322.3307 | $655.00 | $23.0502 | 69 | 62 | $+17.5218 |

Incremental adverse execution cost stress, applied to settled gross turnover after the modeled fee treatment:

| Model | 0 bps | 5 bps | 10 bps | 25 bps | Extra cost to erase positive PnL |
|---|---:|---:|---:|---:|---:|
| M0 | $-249.5610 | $-253.7585 | $-257.9560 | $-270.5485 | $0.0000 (0.00 bps) |
| M1 | $-199.6242 | $-204.1817 | $-208.7392 | $-222.4117 | $0.0000 (0.00 bps) |
| p_market | $+0.0000 | $+0.0000 | $+0.0000 | $+0.0000 | $0.0000 (0.00 bps) |
| previous_D | $+17.5218 | $+17.1943 | $+16.8668 | $+15.8843 | $17.5218 (267.51 bps) |

T-59 ask prices repriced adversely while keeping the side and $5 order intent selected at T-60:

| Model | 0 bps PnL (fills) | +5 bps | +10 bps | +25 bps |
|---|---:|---:|---:|---:|
| M0 | $-249.5610 (1679 fills, 0 misses) | $-253.6255 (1679 fills, 0 misses) | $-257.6905 (1679 fills, 0 misses) | $-269.8474 (1679 fills, 0 misses) |
| M1 | $-199.6242 (1823 fills, 0 misses) | $-204.0771 (1823 fills, 0 misses) | $-208.5323 (1823 fills, 0 misses) | $-221.8510 (1823 fills, 0 misses) |
| p_market | $+0.0000 (0 fills, 0 misses) | $+0.0000 (0 fills, 0 misses) | $+0.0000 (0 fills, 0 misses) | $+0.0000 (0 fills, 0 misses) |
| previous_D | $+17.5218 (131 fills, 0 misses) | $+17.1859 (131 fills, 0 misses) | $+16.8501 (131 fills, 0 misses) | $+15.8454 (131 fills, 0 misses) |

M1−M0 paired daily net-PnL difference: $+49.9368 total across 61 calendar days, including days without fills.

### holdout_descriptive (settlement cutoff 2026-10-07T00:00:00+00:00)

| Model | Priceable at T-60 | Positive-EV markets | Positive-EV sides | EV ≥ 1% | T-59 fills | Settled | Turnover | Fees | Net PnL | PnL/turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 23,110 | 2,514 | 2,514 | 567 | 549 | 549 | $2,745.00 | $95.5892 | $-1.9380 | -0.071% |
| M1 | 23,110 | 8,796 | 8,796 | 5,150 | 4,946 | 4,946 | $24,730.00 | $871.1063 | $-216.9903 | -0.877% |
| p_market | 23,110 | 0 | 0 | 0 | 0 | 0 | $0.00 | $0.0000 | $+0.0000 | — |
| previous_D | 23,110 | 37 | 37 | 0 | 0 | 0 | $0.00 | $0.0000 | $+0.0000 | — |

Resolved fills by execution depth:

| Model | Depth source | Fills | Shares | Turnover | Fees | Wins | Losses | Net PnL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | full_book | 549 | 5473.6085 | $2745.00 | $95.5892 | 286 | 263 | $-1.9380 |
| M1 | full_book | 4946 | 50064.0350 | $24730.00 | $871.1063 | 2526 | 2420 | $-216.9903 |

Incremental adverse execution cost stress, applied to settled gross turnover after the modeled fee treatment:

| Model | 0 bps | 5 bps | 10 bps | 25 bps | Extra cost to erase positive PnL |
|---|---:|---:|---:|---:|---:|
| M0 | $-1.9380 | $-3.3105 | $-4.6830 | $-8.8005 | $0.0000 (0.00 bps) |
| M1 | $-216.9903 | $-229.3553 | $-241.7203 | $-278.8153 | $0.0000 (0.00 bps) |
| p_market | $+0.0000 | $+0.0000 | $+0.0000 | $+0.0000 | $0.0000 (0.00 bps) |
| previous_D | $+0.0000 | $+0.0000 | $+0.0000 | $+0.0000 | $0.0000 (0.00 bps) |

T-59 ask prices repriced adversely while keeping the side and $5 order intent selected at T-60:

| Model | 0 bps PnL (fills) | +5 bps | +10 bps | +25 bps |
|---|---:|---:|---:|---:|
| M0 | $-1.9380 (549 fills, 0 misses) | $-3.3074 (549 fills, 0 misses) | $-4.6781 (549 fills, 0 misses) | $-8.7733 (549 fills, 0 misses) |
| M1 | $-216.9903 (4946 fills, 0 misses) | $-229.2369 (4946 fills, 0 misses) | $-241.4859 (4946 fills, 0 misses) | $-278.1123 (4946 fills, 0 misses) |
| p_market | $+0.0000 (0 fills, 0 misses) | $+0.0000 (0 fills, 0 misses) | $+0.0000 (0 fills, 0 misses) | $+0.0000 (0 fills, 0 misses) |
| previous_D | $+0.0000 (0 fills, 0 misses) | $+0.0000 (0 fills, 0 misses) | $+0.0000 (0 fills, 0 misses) | $+0.0000 (0 fills, 0 misses) |

M1−M0 paired daily net-PnL difference: $-215.0523 total across 98 calendar days, including days without fills.

## Required free-cash portfolio control

This separate control starts each period with $100 and stakes 5% of free cash at T-60, subject to the existing $5 minimum-order fallback and cutoff settlement accounting.

### development_cv_selected

| Model | Qualified decisions | Filled entries | Settled fills | Turnover | Fees | Net PnL | PnL/turnover | Cash at cutoff | Unsettled locked gross |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 1 | 1 | 1 | $5.00 | $0.1800 | $-5.1800 | -103.600% | $94.8200 | $0.0000 |
| M1 | 1 | 1 | 1 | $5.00 | $0.1836 | $-5.1836 | -103.672% | $94.8164 | $0.0000 |
| p_market | 0 | 0 | 0 | $0.00 | $0.0000 | $+0.0000 | — | $100.0000 | $0.0000 |
| previous_D | 21 | 21 | 21 | $120.01 | $4.1099 | $-3.9685 | -3.307% | $96.0315 | $0.0000 |

### holdout_descriptive

| Model | Qualified decisions | Filled entries | Settled fills | Turnover | Fees | Net PnL | PnL/turnover | Cash at cutoff | Unsettled locked gross |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 1 | 1 | 1 | $5.00 | $0.1750 | $-5.1750 | -103.500% | $94.8250 | $0.0000 |
| M1 | 1 | 1 | 1 | $5.00 | $0.1750 | $-5.1750 | -103.500% | $94.8250 | $0.0000 |
| p_market | 0 | 0 | 0 | $0.00 | $0.0000 | $+0.0000 | — | $100.0000 | $0.0000 |
| previous_D | 0 | 0 | 0 | $0.00 | $0.0000 | $+0.0000 | — | $100.0000 | $0.0000 |

## What the selected M1 learned

Feature gains describe split use; a feature gain by itself does not prove interaction. Interaction counts below require both features to appear along one root-to-leaf path in the same final development tree.

| Feature | Splits | Gain | Gain share |
|---|---:|---:|---:|
| logit_p_market | 18 | 47.861040 | 4.7% |
| futures_index_basis_rel_1m | 94 | 270.503551 | 26.5% |
| btc_sigma_5m | 63 | 169.861220 | 16.6% |
| spread_up | 0 | 0.000000 | 0.0% |
| spread_down | 0 | 0.000000 | 0.0% |
| d_btc_minus_market | 108 | 534.422251 | 52.3% |

| Interaction pair on the same path | Paths | Trees with ≥1 path |
|---|---:|---:|
| d and basis | 116 | 51 |
| d and sigma | 98 | 42 |

The final-refit M1 correction surfaces use five development quantiles per axis (25 points each), with remaining features fixed at development medians. Full values are in [interaction_grids.csv](reports/btc_preopen/btc_market_basis_vol_interaction_20261017/interaction_grids.csv). These are model mechanism views, not causal effects; their ranges were fixed before looking at holdout predictions.

## Descriptive holdout note

The holdout was scored once after the manifest freeze. It was historically exposed before this experiment, so it is descriptive and cannot validate a new edge or guide a change in this task.

Successful reporting-resume invocation: 197.19 s; peak process working set: 4.041 GiB. It reused all 18 verified fit artifacts and the one-time holdout prediction; no fits or holdout predictions were repeated. The initial fit/score invocation stopped during report serialization, so its elapsed time was not captured.

## Conclusions

1. Development-selected M1−M0 Δ log-loss is -0.00031335 (95% block CI [-0.00141510, +0.00095350]); Δ Brier is -0.00015640. The direction favors M1 in only one validation periods (May +0.00029675; June -0.00124988). The interval includes zero, so the point change is not statistically clear under this block bootstrap.
2. The final M1 trees put d with basis on a root-to-leaf path in 116 paths (51 trees), and d with sigma in 98 paths (42 trees).
3. Development fixed-$5 M0 had 1700 qualifying decisions, 1679 fills, and net PnL $-249.5610; M1 had 1841 qualifying decisions, 1823 fills, and $-199.6242. On holdout M0 had 567 qualifying decisions, 549 fills, and $-1.9380; M1 had 5150 qualifying decisions, 4946 fills, and $-216.9903. M1 was not profitable after modeled costs on holdout; this is descriptive.
4. Descriptive holdout M1−M0 Δ log-loss is -0.00031874 (95% block CI [-0.00089345, +0.00011986]); Δ Brier is -0.00016251. The holdout was previously viewed and did not influence any choice.

Reproduce without CLI arguments: `python run_btc_market_basis_vol_interaction_20261017.py`. Prediction checkpoints and run details are under `data/analysis/polymarket/BTC/btc_market_basis_vol_interaction_20261017`.
