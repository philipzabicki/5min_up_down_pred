# Full-development BTC pre-open fit - 2026-10-16

## Main answers

- The new model did not demonstrate a holdout quality improvement over `A_frozen`. On the 28,224 common proxy observations and 28,220 official outcomes available by the cutoff, every paired 95% interval for log-loss, Brier, and AUC includes zero. Point estimates slightly favor `A_frozen` on log-loss and Brier.
- The frozen 1% net-EV rule still loses money in the no-credit portfolio: **-$0.41 from $100**, leaving $99.59, with a 5.17% maximum drawdown. Independent $5 orders under the same rule lost **$256.28** across 2,797 settled fills.

## Run status

- Status: completed; one full-development model pipeline and one historical holdout evaluation.
- Historically exposed holdout: July-October outcomes and overlapping BTC observations were viewed in prior repository experiments; one descriptive evaluation only, no choices may change from its result.
- Development markets: starts before `2026-07-01T00:00:00Z`. Holdout market starts: `2026-07-01T00:00:00Z` through `2026-10-06T23:55:00Z` inclusive. Exclusive data cutoff: `2026-10-07T00:00:00Z`.
- Model fit's latest eligible label time: `2026-06-30T23:59:00+00:00`; the frozen contract allows labels only through the first holdout decision.
- Calibration: `development chronological CV predictions conditional on selected generators and features`; 318,614 development rows. Provenance: `['development_selected_configuration_chronological_oof_conditional']`. These validation predictions are conditional on generator settings, weights, features, and hyperparameters selected on development; they are not an independent evaluation of the whole selection process.
- Calibration prediction folds: 10; rows: 318,614; holdout rows included: `False`.
- Frozen bundle: `data/analysis/polymarket/BTC/btc_preopen_full_development_fit_20261016/pipeline/794a69e369a3f82d/stages/final_model_c3c05535e1df/model_bundle.json` (SHA-256 `7dca4d49cb2412d3616dd3877bec62ffba4d9aceb98b4439c61c070c4f067213`). Freeze: `data/analysis/polymarket/BTC/btc_preopen_full_development_fit_20261016/pipeline/794a69e369a3f82d/evaluated_bundle_freeze.json` (SHA-256 `0e54fca8103897420760690d682485086b5f34a813a842c971f418b41a906d32`).
- Frozen policy: `A_BTC_calibrated_ev_1.0000%`, 1.00% minimum net EV; no holdout selection.

### Recovery history

The first pipeline attempt failed in `feature_selection` after 31,436.2 seconds. The preserved traceback records `KeyError: 'k'` in `select_features.py` while sorting the top-k sweep results. The same stage input signature completed on retry with 17 top-k evaluations and 64 selected features. The retry skipped previously verified stages; model tuning, final fitting, calibration OOF, Platt calibration, external evaluation, bundle verification, and the outer holdout-evaluation stage all finished with `completed` status. The top-level manifest retains its historical `error` value, and the original stdout, stderr, and stage attempt records remain intact.

## One-pass development stages

| Stage | Studies | Trials evaluated | COMPLETE | PRUNED | Errors |
|---|---:|---:|---:|---:|---:|
| volume_profile | 1 | 96 | 96 | 0 | 0 |
| reaction_profile | 1 | 96 | 77 | 19 | 0 |
| main model | 1 | 60 | 22 | 38 | 0 |

The main-model study used 10 internal chronological folds, as recorded in the effective run configuration. An earlier generated sentence counted the two stored train/validation fingerprint keys as folds; that display count is corrected here. Its development CV score is a tuning result conditional on development selection. No external loop repeats the complete generator, weighting, selection, or tuning stages.

## Holdout probability quality

The standalone evaluator table below reports the official-outcome score recorded by the pipeline on 28,221 labelled rows. The common-sample comparison later in this report uses 28,220 official outcomes whose cash release was available by the frozen cutoff; one additional labelled outcome was released after that cutoff.

| Forecast | Rows | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|
| Raw model | 28,221 | 0.692526 | 0.249689 | 0.5217 |
| Platt calibrated | 28,221 | 0.692502 | 0.249678 | 0.5217 |

## Snapshot scenarios from the single-model evaluator

The single-model evaluation reused the hash-verified T-60 decision and T-59 execution snapshots. It did not replay the book-event archive. It scored 28,224 holdout market rows, with 28,220 official outcomes and cash releases available by the frozen data cutoff. The selected-policy row uses the frozen 1% hurdle; the two rows labelled `control` use a 0% hurdle.

| Scenario | Trades | Ending cash | Net PnL | Turnover | Fees | Max drawdown | Accounting |
|---|---:|---:|---:|---:|---:|---:|---|
| `A_BTC_calibrated_ev_1.0000%` | 2 | $99.59 | $-0.41 | $10.25 | $0.36 | 5.17% | passed |
| `fixed_5_usd_control` | 89 | $2.44 | $-97.56 | $445.00 | $15.44 | 98.00% | passed |
| `free_cash_5pct_control` | 4 | $98.99 | $-1.01 | $20.98 | $0.74 | 10.08% | passed |

The holdout evaluation also records T-60/T-59 decision and fill coverage, full-book versus top-of-book proxy sources, rejection reasons, and adverse-price sensitivity under the existing portfolio accounting rules. None of these holdout results changed the model, calibrator, or policy.

## Common-sample comparison with `A_frozen`

Both saved prediction files contain exactly the same 28,224 unique market starts, from 2026-07-01 00:00 UTC through 2026-10-06 23:55 UTC. Their decision timestamps, target windows, and Binance proxy labels match row for row. Both sets of predictions were available at T-59. The official Polymarket target was joined separately from the frozen market index and scored only where its resolution and cash release were available by the 2026-10-07 00:00 UTC cutoff.

| Coverage item | Count |
|---|---:|
| Predictions per model | 28,224 |
| Identical common market starts | 28,224 |
| Complete Binance price-proxy labels | 28,224 |
| Official labels present in the market index | 28,221 |
| Official labels and cash releases available by cutoff | 28,220 |
| Official outcomes unavailable by cutoff | 4 (3 missing; 1 released after cutoff) |

### Binance price-proxy labels

| Forecast | Rows | Raw log-loss | Raw Brier | Raw AUC | Calibrated log-loss | Calibrated Brier | Calibrated AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_frozen` | 28,224 | 0.692477 | 0.249665 | 0.521106 | 0.692521 | 0.249687 | 0.521106 |
| New model | 28,224 | 0.692565 | 0.249709 | 0.521151 | 0.692544 | 0.249698 | 0.521151 |

### Official Polymarket settlement labels

| Forecast | Rows | Raw log-loss | Raw Brier | Raw AUC | Calibrated log-loss | Calibrated Brier | Calibrated AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_frozen` | 28,220 | 0.692425 | 0.249639 | 0.522040 | 0.692466 | 0.249659 | 0.522040 |
| New model | 28,220 | 0.692527 | 0.249690 | 0.521710 | 0.692503 | 0.249678 | 0.521710 |

### Paired differences, new model minus `A_frozen`

Intervals are 95% paired circular moving-block bootstrap intervals using 3-day UTC blocks, 2,000 replications, and seed `20261012`. Negative log-loss/Brier differences favor the new model; positive AUC differences favor it.

| Target | Forecast | Log-loss delta [95% CI] | Brier delta [95% CI] | AUC delta [95% CI] |
|---|---|---:|---:|---:|
| Proxy | Raw | +0.0000885 [−0.0001214, +0.0002962] | +0.0000439 [−0.0000608, +0.0001474] | +0.0000446 [−0.0024728, +0.0025558] |
| Proxy | Calibrated | +0.0000228 [−0.0001938, +0.0002233] | +0.0000113 [−0.0000968, +0.0001112] | +0.0000446 [−0.0024728, +0.0025558] |
| Official | Raw | +0.0001021 [−0.0000937, +0.0003019] | +0.0000507 [−0.0000468, +0.0001503] | −0.0003299 [−0.0025329, +0.0018881] |
| Official | Calibrated | +0.0000369 [−0.0001572, +0.0002363] | +0.0000185 [−0.0000785, +0.0001179] | −0.0003299 [−0.0025329, +0.0018881] |

All paired intervals include zero. The new model's point estimates are slightly worse on log-loss and Brier for both targets; AUC moves slightly upward on the proxy and downward on official outcomes. Platt calibration is monotonic, so it leaves each model's AUC unchanged. The two label sources disagree on 2,213 of the 28,220 available official outcomes; the proxy and official scores are kept separate. This holdout was historically exposed and remains descriptive.

## Development-only tuning checkpoints

Generator studies minimize mean decision-row log-loss plus 0.75 times the population standard deviation across chronological folds. The main-model study minimizes mean log-loss plus 0.5 times that standard deviation. These are development tuning scores, separate from the holdout evaluation above.

| Study | Evaluated trials (complete/pruned) | Best CV criterion | Mean fold log-loss | Fold SD | Change from previous checkpoint | Cumulative elapsed |
|---|---:|---:|---:|---:|---:|---:|
| Reaction profile 24 | 24 (23/1) | 0.69284429 | 0.69270037 | 0.00019188 | — | 524.1 s |
| Reaction profile 48 | 48 (42/6) | 0.69280134 | 0.69264459 | 0.00020901 | −0.00004294 | 976.3 s |
| Reaction profile 96 | 96 (77/19) | 0.69278666 | 0.69260870 | 0.00023728 | −0.00001468 | 1,923.2 s |
| Volume profile 24 | 24 (24/0) | 0.69280067 | 0.69264500 | 0.00020755 | — | 362.5 s |
| Volume profile 48 | 48 (48/0) | 0.69279229 | 0.69262375 | 0.00022472 | −0.00000838 | 737.1 s |
| Volume profile 96 | 96 (96/0) | 0.69276711 | 0.69258109 | 0.00024802 | −0.00002518 | 1,441.7 s |
| Main model 10 | 10 (4/6) | 0.69233681 | 0.69213743 | 0.00039876 | — | 237.1 s |
| Main model 20 | 20 (6/14) | 0.69229600 | 0.69209770 | 0.00039660 | −0.00004081 | 428.6 s |
| Main model 30 | 30 (10/20) | 0.69229600 | 0.69209770 | 0.00039660 | 0.00000000 | 662.5 s |
| Main model 40 | 40 (16/24) | 0.69229600 | 0.69209770 | 0.00039660 | 0.00000000 | 909.2 s |
| Main model 50 | 50 (20/30) | 0.69229600 | 0.69209770 | 0.00039660 | 0.00000000 | 1,136.2 s |
| Main model 60 | 60 (22/38) | 0.69229600 | 0.69209770 | 0.00039660 | 0.00000000 | 1,342.7 s |

At the final generator checkpoint, the criterion improved by 0.00005763 for RP and 0.00003356 for VP from trial 24. The main-model criterion improved by 0.00004081 through trial 20 and stayed flat through trial 60. The final main-model fit used 3,186,140 eligible rows, 64 features, and 31 boosting iterations. No holdout labels entered these choices.

## Actual stage durations

| Stage | Recorded elapsed |
|---|---:|
| Indicators, verified cache reuse | 0.03 s |
| Reaction profile, 96 trials | 1,924.0 s |
| Volume profile, 96 trials | 1,442.0 s |
| Final candidate feature dataset | 242.1 s |
| Target weights | 2,619.0 s |
| Feature selection, failed attempt then successful retry | 25,175.8 s + 1,310.7 s |
| Main-model tuning, 60 trials | 1,343.5 s |
| Final model fit | 3.1 s |
| Calibration OOF predictions | 19.9 s |
| Platt calibration fit | 0.36 s |
| External holdout prediction scoring | 2.27 s |
| Fresh-process bundle verification | 230.5 s |
| Snapshot holdout stage | about 669 s (pipeline completion to outer-stage completion) |

The first pipeline attempt lasted 31,436.2 seconds and failed at feature selection. The resumed completed attempt lasted 2,944.7 seconds; verified earlier stages were skipped. Including the snapshot evaluation, recorded active elapsed time was about 9 h 44 min. The wall-clock span from the first start to completion was 15 h 33 min, including the 5 h 49 min wait before resume. The snapshot-stage duration is derived from manifest timestamps because that stage does not store a separate monotonic elapsed field.

## Same frozen policy: independent $5 orders and no-credit portfolio

Both model rows use the same frozen 1% minimum net-EV hurdle, calibrated probabilities, T-60 side selection, T-59 execution snapshots, accepted fee rules, and 2026-10-07 cutoff. Independent orders each use $5 gross stake without a shared balance. The portfolio starts with $100, stakes 5% of free cash, and never borrows.

| Forecast | Independent signals | Filled / settled orders | Filled turnover | Filled fee equivalent | Realized PnL on settled fills | Starting capital / ending cash | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_frozen`, independent $5 | 4,272 | 4,018 / 4,017 | $20,090.00 | $725.31 | −$726.97 | N/A | N/A |
| New model, independent $5 | 3,005 | 2,798 / 2,797 | $13,990.00 | $517.58 | −$256.28 | N/A | N/A |
| `A_frozen`, 5% free-cash portfolio | — | 2 / 2 | $10.24 | $0.36 | −$0.60 | $100.00 / $99.40 | 5.17% |
| New model, 5% free-cash portfolio | — | 2 / 2 | $10.25 | $0.36 | −$0.41 | $100.00 / $99.59 | 5.17% |

Independent-order turnover and fee equivalents include every T-59 fill; realized PnL uses the 4,017 and 2,797 outcomes released by cutoff. One final $5 fill per model had no cash release by cutoff. Independent orders have no common equity curve, so portfolio capital and drawdown do not apply. Both $100 portfolio accounting audits passed with non-negative free cash. The new-model portfolio figures reconcile to the saved holdout ledger within $0.00000001; the `A_frozen` values also match the prior saved $100 ledger under the same snapshot hashes.

This policy still loses money in the bankroll-constrained evaluation: the new model finishes **$0.41 below** its $100 starting balance. Its smaller independent-order loss also remains negative and is not evidence of positive expected return.

## Verification artifacts

- `A_frozen` holdout predictions: `data/analysis/polymarket/BTC/preopen_research_20261011_full_compute20_retry/runs/evaluated_holdout/794a69e369a3f82d/stages/external_evaluation_8ecf7dad07f7/external_test_predictions.parquet` (SHA-256 `b8febf2da37695cae6727737abc9cd650c7a69052b97b6ac72d27254801426f6`); frozen A bundle SHA-256 `4820e23493efb1743863b58b411352006a71ff7ce8c66068c17c8c88e65e4583`.
- New-model predictions: `data/analysis/polymarket/BTC/btc_preopen_full_development_fit_20261016/pipeline/794a69e369a3f82d/stages/external_evaluation_66a0ee79c911/external_test_predictions.parquet` (SHA-256 `294309cfc91216ddb357817d82947b2c08b65335c89c71bb14bb3d8c59255bf1`); new bundle SHA-256 `7dca4d49cb2412d3616dd3877bec62ffba4d9aceb98b4439c61c070c4f067213`.
- Official market index SHA-256 `f828da19aa9c1811118abadd3d58a00b51ff2311ceb7492138ccfc1da5dd2973`. T-60 and T-59 snapshot SHA-256 values are `38259fadf50c28e455f5d4523610cbac827886e35922076dbba5cb33bd55e0b1` and `abee0fdaf31eaf1455b4343e985bf9d5ee38a84dd8eb7e06adf1e08fc0c380d0`.
- The paired analysis and its metric intervals are in `reports/btc_preopen/btc_preopen_full_development_fit_20261016/holdout_comparison_20261016.json`. The analysis used saved predictions, market labels, and T-60/T-59 snapshots; it did not train a model or replay archived book events.

## Cost and artifacts

- Runtime estimate recorded before the run: `data/analysis/polymarket/BTC/btc_preopen_full_development_fit_20261016/cost_estimate.json` (forecast 5-8 hours from measured full-development stages; separate from the superseded multi-pipeline estimate).
- Pipeline manifest: `data/analysis/polymarket/BTC/btc_preopen_full_development_fit_20261016/pipeline/794a69e369a3f82d/run_manifest.json`.
- Holdout economics: `data/analysis/polymarket/BTC/btc_preopen_full_development_fit_20261016/holdout_evaluation/holdout_portfolio_results.json`.
- Frozen fit rows: 3,186,140; features: 64; best iteration: 31.
