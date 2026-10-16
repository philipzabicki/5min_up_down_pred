# BTC modeling audit — 2026-10-15

> **Scope correction (2026-10-16):** The four-family ablation described below was started after this audit, then interrupted during `calibration_seed/base_plus_rp/feature_selection` when the objective changed. It is not a completed comparison and will not be resumed. Its completed indicator artifacts and compatible generator parameter sets were audited; only the indicators and parameter values are reused by the single joint VP+RP tuning pipeline documented in [BTC_VP_RP_TUNING_20261016.md](BTC_VP_RP_TUNING_20261016.md). The partial selector output and old CV scores are not reused.

## Scope and decisions

Audit started from `main` at `d8e2540f9c5fff2936b61a4568fb4689e186de45`, tracking `origin/main`, with a clean worktree and no staged changes. The machine profile recorded for this repository is an i7-13650HX, 20 logical processors, 64 GiB RAM, and an RTX 4060 Laptop GPU. No training or replay process was active at the initial check. The existing A/B run artifacts, Optuna databases, reports, `performance.md`, and [the binding BTC research contract](BTC_RESEARCH_CONTRACT.md) were inspected.

The highest impact confirmed defects were (1) LightGBM stopped on mean validation log loss while the chosen iteration and trial ranking used mean plus a fold-dispersion penalty, and (2) the pre-open batch contract admitted predictions through market start although the frozen contract allows only one second after nominal T-60. Both are corrected in source. The standalone `nightly_v1` profile's old 45-second deadline was also changed to the contract's one second; the model-update runner already overrode this to one second in its effective runs. New work uses independent artifacts; historical study databases and results were not edited. No prospective collector was started.

## Effective pipeline and lineage

| Order | Entry points and artifacts | Effective behavior in the 2026-10-12 model update |
|---|---|---|
| Inputs and target | `run_btc_preopen_model_update.py` → `run_btc_preopen_experiment.py` → `features/btc_preopen_contract.py`; source manifest and fold contract in `data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/` | 1-minute Binance COIN-M BTCUSD index proxy. Inputs run from `2020-06-09 09:33 UTC`; each fold fits only labels available by its exclusive boundary. Pre-open decisions are the scheduled rows, with all five future label candles required. Effective prediction budget is 1 s in the update runner and now in the standalone profile. |
| Indicators | `fit_indicators.py`; per-fold indicator results in each run's `stages/indicators_*` | Frozen profile is re-fit on each fold's eligible prefix. The update profile uses population 64, 8 generations, 20 workers, with an indicator stage budget of 21,600 s. |
| VP and RP search | `fit_volume_profile.py` and `fit_reaction_profile.py`, called by the experiment runner with fold-local development data | The profile override sets 24 trials per generator, 2,700 s per stage, 10 chronological inner folds, a minimum of 5 successful trials, and GPU backend. Generator tuning rows are eligible, scheduled decisions only; their target weight is constant 0.4625. The runner uses TPE with 5 startup trials and a MedianPruner with 3 startup trials, 10 warm-up iterations, and a report every 10 iterations. The generator helper model is 300 trees maximum, 40-round stopping, learning rate 0.05, 63 leaves, depth 6, minimum leaf size 128, L2=5, no bagging, and 1.0 feature fraction. |
| Feature generation | `create_modeling_dataset.py`, cached under `shared_runs/<fold>/<run>/stages/final_feature_dataset_*` | After fitting the generators, the pipeline builds indicators, reaction-profile, volume-profile, candle, session, volatility, and basis features. It verifies the stored decision mask against the timestamp contract. Feature data are cached by input/config/code signatures. |
| Target weights and selection | `optimize_target_weights.py`, then `select_features.py` | The target-weight stage evaluates the frozen 16-candidate search on fold-local data. Feature selection runs once in each chronological fold on its training partition; A/B variants reuse that fold's selector result. In the completed A/B update, best k equaled the selector recommendation in all three development folds; the required 20-feature savings gate was not met. |
| Main model tuning and fit | `optimize_lgbm_hyperparameters.py` then `train_lgbm.py` | 10 requested trials per model/outer fold, minimum 5 successful, 10 inner CV folds, max 500 trees, 50-round stopping, and 2,700 s tuning-and-fit stage budget. The registered six-dimensional search ranges and one seed trial are in the run's `effective_config.json`. Fold-recency weighting is explicitly disabled by the update runner. The objective uses unweighted mean log loss plus 0.5 times unweighted population fold standard deviation. Validation uses decision rows without auxiliary observation weights. |
| Calibration and policy | Earlier-fold OOF calibration in `run_btc_preopen_model_update.py`; policy uses frozen research code and quotes | Calibration used 52,127 earlier-fold OOF rows through `2026-06-30T23:55:00Z`; it excluded evaluated holdout. Policy was selected on development before the holdout. |
| Final evaluation | `run_btc_preopen_research.py`; `reports/btc_preopen/btc_preopen_model_update_20261012_partial.md` and local run manifests | Existing July–October evaluation is historically exposed, descriptive only, and cannot select new variants. The final B model stage is incomplete and remains rejected. No deployment refit or order was run. |

**Generator definitions:** VP is the fixed-range volume profile implemented in `features/volume_profile_fixed_range.py`: candle volume is accumulated over price bins and expressed as local density/share and above/below-volume features. RP is the fixed-grid reaction profile in `features/reaction_profile_fixed_grid.py`, fit by `fit_reaction_profile.py`: candle wick reactions are accumulated on a fixed price grid with horizon windows/decay, producing seven support/resistance, imbalance, and center-distance features per short/medium/long/all horizon. RP is not a guessed abbreviation or a model architecture.

Each nested run's `effective_config.json` records effective profile, contract, and input identities. Its `run_manifest.json` records stage input signatures, attempts, outputs and SHA-256 values; the fold contract JSON records time boundaries and label eligibility. The model-update top-level `model_update_manifest.json` records fold prediction hashes and the frozen development/holdout identities. These files remain alongside the local study DBs under `data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/`.

The three outer development market periods are A (`2026-04-15T17:05Z`–`2026-04-30T23:55Z`), B (`2026-05-01T00:00Z`–`2026-05-31T23:55Z`), and C (`2026-06-01T00:00Z`–`2026-06-30T23:55Z`). Their latest eligible training-label times are respectively `2026-04-15T17:04Z`, `2026-04-30T23:59Z`, and `2026-05-31T23:59Z`; fit starts at `2020-06-09T09:33Z`. The evaluated period begins `2026-07-01T00:00Z` and ends at the frozen `2026-10-07T00:00Z` exclusive cutoff. In main-model trials A/B/C, training rows were 3,076,285 / 3,098,300 / 3,142,940 and scheduled decision rows were 615,257 / 619,660 / 628,588.

## Target and availability check

For a market opening at `12:00 UTC`:

| Event | Timestamp / value |
|---|---|
| Input candle `Opened` | `11:58:00` |
| Last feature candle event close | `11:59:00` (`Opened + 1m`) |
| Nominal decision (T-60) | `11:59:00` |
| Prediction/execution deadline (T-59) | `11:59:01` |
| Market interval | `[12:00:00, 12:05:00)` |
| Proxy reference | `Open[12:00]=101.00`, `Close[12:04]=101.00` |
| Proxy label and availability | UP on a tie; label available at `12:05:00` (`Opened + 7m`) |
| Official Polymarket label | Not available for this synthetic example; it is a separate Chainlink-settled outcome in the official market archive. |

Batch candles contain event timestamps, not exchange receive timestamps. The batch contract therefore cannot prove when a boundary candle arrived; it now records actual prediction availability as unknown and permits only the explicit one-second compute budget. The public collector records candle receive time. Its three-market smoke met the one-second budget at 0.480–0.573 s, but that is operational validation, not evidence over a prospective sample.

`utils/data.compute_binary_close_target_from_opened` remains the generic close-to-close horizon helper. The BTC pre-open path calls `build_preopen_contract_frame` and compares the market window's start open with its final candle close. `purge_unavailable_training_rows` and fold checks purge using label availability (`Opened + 7m <= boundary`), not a row-count gap. Gaps invalidate proxy labels; UTC offsets and day/hour boundary examples are covered by tests. This change affects new generated batches and their cache identity; old artifacts retain their recorded contract/hash and are not relabeled in place.

The training frame contains every eligible minute label. For each five-minute market, target-weight tuning allocates total weight mass between the scheduled decision minute and the other four minutes: with decision weight `w`, each auxiliary minute receives `(1-w)/4`. The frozen candidate search contains 16 decision-weight values; the historical holdout model selected `w=0.29` from development. Generator fitting uses only scheduled decision rows, so its per-row target weight is constant and does not treat the four auxiliary minutes as independent generator examples. Main-model fitting uses the chosen minute weights, while validation loss is unweighted on actual decision rows.

## Actual search budgets and outcomes

Generator counts below are from the frozen per-fold SQLite studies, not standalone fitter defaults. Each study had unique parameter configurations and no duplicate trial parameters. The three development studies each sampled 24 of 20 VP or 12 RP dimensions; the top five COMPLETE trials had only 2/300 VP and 2/180 RP parameter values at a search boundary (under 1.2%). This is a small sample relative to dimensionality, so low boundary occupancy does not prove the spaces are adequately searched.

| Development fold | VP COMPLETE / PRUNED / FAIL | VP wall time; best objective | RP COMPLETE / PRUNED / FAIL | RP wall time; best objective |
|---|---:|---:|---:|---:|
| A | 24 / 0 / 0 | 356 s; 0.6927983 | 22 / 2 / 0 | 429 s; 0.6928467 |
| B | 24 / 0 / 0 | 349 s; 0.6927921 | 22 / 2 / 0 | 448 s; 0.6929440 |
| C | 24 / 0 / 0 | 340 s; 0.6927777 | 22 / 2 / 0 | 448 s; 0.6928309 |

The best objectives are inner-fold generator scores, not external model-quality results. VP trial best iterations across the three studies ranged 4–23; RP ranged 1–19. None reached the 300-tree cap. The profile stages ended at their requested trial count, not their 2,700 s cap. No completed checkpoint was double-counted on resume. The calibration seed and historically exposed holdout generator studies also used 24 trials; those are not new development evidence.

| Main-model tuning | COMPLETE / PRUNED / FAIL, A / B | Distinct configurations, A / B | Best-iteration range, A / B | Study duration, A / B |
|---|---:|---:|---:|---:|
| Dev A | 7/3/0; 8/2/0 | 10 / 10; no duplicates | 18–39 / 14–42 | 329 / 346 s |
| Dev B | 5/5/0; 5/10/0 | 10 / 15; no duplicates | 20–63 / 25–74 | 333 / 388 s |
| Dev C | 5/14/0; 5/21/0 | 19 / 26; no duplicates | 27–97 / 30–81 | 764 / 1,142 s |
| Evaluated period | 6/9/0; 4/73/0 | 15 / 77; no duplicates | 21–59 / 26–62 | 547 / 2,951 s from first study start to last completion |

The calibration-seed generator studies also ran 24 configurations each: both VP studies had 24 COMPLETE, and both RP studies had 23 COMPLETE / 1 PRUNED. Its two model studies had 5 COMPLETE / 10 PRUNED and 5 COMPLETE / 11 PRUNED. The evaluated-period VP study had 24 COMPLETE and RP had 23 COMPLETE / 1 PRUNED. These counts are historical provenance; the evaluated-period results are not development evidence.

The main-model search space was `learning_rate` 0.02–0.12 (log), `num_leaves` 63–255, `min_data_in_leaf` 1,024–65,536 (log), `feature_fraction` 0.65–0.95, and `lambda_l1`/`lambda_l2` 0.1–32 (log). Its registered seed was 0.04 learning rate, 160 leaves, 8,192 minimum leaf rows, 0.8 feature fraction, and L1/L2 of 4. These are the effective update settings, not limits restored from the older `modeling.json`.

Main-model completed trials (55 total) selected best iterations from 14 to 97; none hit 500. The holdout B study's cumulative attempt accounting was about 2,435.7 s against a 2,430 s tuning allowance, and it reached only 4 successful trials against the required 5. It did not reach final fit. Do not resume that frozen B with changed rules. The raw study-wall interval includes pauses between attempts; the cumulative-budget manifest is authoritative for budget exhaustion.

Existing development evidence does not establish that the current pruner is generally harmful. The separate fixed continuation diagnostic selected five already-pruned configs across learning-rate quantiles and continued them on the three development B folds: 15/15 completed in 1,052.2 s end-to-end. One configuration beat the best COMPLETE trial in its same development fold by 0.000031; the other 14 did not. Six best iterations occurred after the source prune checkpoints. Because the source trials and comparison folds were not a same-study replay, this does not estimate a false-prune rate. Keep the frozen pruner for now; a same-study, same-objective replay is required before changing it. See [the diagnostic report](../reports/btc_preopen/b_pruner_development_diagnostic_20261014/pruner_diagnostic_report.md).

## Objective repair and verification

Historically, VP/RP and main-model `lgb.cv` early stopping watched mean `binary_logloss`; the final iteration used mean plus 0.75×fold standard deviation for VP/RP or 0.5×standard deviation for the main model. Recency-weighted profile search applied fold weights only after that iteration had already been selected. The Optuna callback reported a different aggregate. A new shared callback now uses one metric curve for per-iteration reporting, pruning, early stopping, iteration selection, and final trial value. Where recency weights are active, fold scores are emitted from LightGBM validation folds each iteration; no extra fold prediction pass is needed. Existing coefficients and weights are preserved. The 2026-10-12 model-update runner turns recency weighting off, so its effective CV base is the unweighted fold mean plus the registered standard-deviation penalty.

Validation: the new CV objective tests cover a dispersion-driven iteration change, a recency-driven iteration change, matching prune/stop/rank values, LightGBM CV integration, and both real VP/RP CV runners on a small CPU fixture. A 12,000×24, 10-fold, 60-round CPU microbenchmark ran twice: baseline CV plus post-CV fold predictions took 0.409/0.424 s; per-fold metrics inside CV took 0.409/0.419 s. Median difference was -0.6% (noise at this scale); no added prediction pass was used. This is a bounded microbenchmark, not a full data performance claim.

## Existing full-pipeline comparisons and measured helper pilot

The historical 2026-10-12 A/B development comparison used 10,795 common official-outcome markets:

| Forecast | Log loss | Brier | AUC |
|---|---:|---:|---:|
| A_frozen | 0.692246 | 0.249550 | 0.524894 |
| A_reoptimized | 0.691908 | 0.249381 | 0.532245 |
| B_dynamic_price_volume | 0.692125 | 0.249490 | 0.527404 |

Paired log-loss differences from `A_frozen` were -0.000338 for A-reoptimized and -0.000121 for B; both confidence intervals crossed zero. B added 5–6 candidate features after selection. The exposed holdout A-reoptimized minus A-frozen official-label log-loss difference was +0.000001 with a 95% block interval spanning zero. These are historical results under the former CV iteration rule and do not identify VP's or RP's incremental value. The earlier Polymarket-target comparison also did not show a demonstrated advantage; it is not repeated here.

The helper-model pilot is complete. It used three preselected configurations from each fold-A development study: the prior best plus parameter-distant configurations from lower and higher best-iteration regions. Each was rescored under six fixed CPU helpers: current; 600-tree cap; lower learning rate with 300/600 caps; higher capacity; and stronger regularization. All 36 comparisons used the same 615,257 decision rows, ten purged chronological folds, constant generator weight 0.4625, and the aligned mean log-loss + 0.75 × population fold standard deviation objective. The outer fold stopped at `2026-04-15T17:04:00Z`; decision-only generator rows had labels available through `2026-04-15T17:00:00Z`. The holdout was not used.

Across the six VP/RP configurations, all helpers selected best iterations below 35; none of the 300- or 600-tree caps was approached. Raising the cap alone produced identical scores and iterations. Halving the learning rate moved best iterations to 11–34 but left the candidate ordering unchanged. The higher-capacity variant changed the VP top-1 ordering by only 0.0000018 objective (Spearman 0.5, top-two overlap 1.0); the other helper variants had Spearman 1.0 and preserved both top-two configurations. RP retained the same top configuration in all six variants. The lower-capacity/stronger-regularization variant improved the third RP configuration by 0.000043 versus current but did not change the winner. These are coarse rankings from three configurations per family, so they support keeping the current cheap helper for screening but do not establish universal ranking reliability. The full scores, iterations, per-configuration times, matrix sizes, learning curves, and rank comparisons are in the [pilot report](../reports/btc_preopen/btc_generator_helper_diagnostic_20261015.md); its hash-bound checkpoint is local under `data/analysis/polymarket/BTC/btc_generator_helper_diagnostic_20261015/`.

The 36 comparisons completed in 370.3 s with six CPU threads. Individual CV fits took 5.0–17.4 s; candidate feature generation took 1.2–13.5 s; the largest matrix was 215.9 MiB for 615,257 rows and 92 VP features. Peak process working set was 2,596,569,088 bytes (2.42 GiB). This is a bounded, fold-A helper screening experiment, not a full pipeline ablation.

At the time of this audit, the clean `base / base+VP / base+RP / base+VP+RP` nested comparison had not started. A ready, resumable runner existed at [run_btc_preopen_feature_ablation.py](../run_btc_preopen_feature_ablation.py), with its frozen four-family configuration at [btc_preopen_vp_rp_ablation_20261015.json](../configs/research/btc_preopen_vp_rp_ablation_20261015.json). It was subsequently started and stopped as described in the scope correction above. Its partial work does not establish a four-family comparison.

At the time of this audit the full nested run had not launched. Its then-current projected duration was roughly 25–29 hours. That four-family plan was superseded by the single joint VP+RP tuning pipeline in [BTC_VP_RP_TUNING_20261016.md](BTC_VP_RP_TUNING_20261016.md). The completed helper and pruning diagnostics remain historical evidence; neither was repeated.

## Data coverage and prospective observation

The frozen research cutoff is `2026-10-07T00:00:00Z`; the trailing 240 calendar days begin `2026-02-09T00:00:00Z`. The verified BTC price input is continuous through `2026-10-06T23:59:00Z`. The market archive has 50,184 Gamma-confirmed starts and 50,184 paired T-60/T-59 snapshots over 84,331 five-minute calendar slots from `2025-12-18T04:25:00Z`; the official-outcome index has 50,183 labels, only from `2026-04-15T17:05:00Z` through `2026-10-06T23:55:00Z`. Thus proxy prediction coverage is longer than official outcome and economic-label coverage. The final 240 days are historically exposed and are not an untouched holdout.

The 240-day prospective observation is **deferred and not started**. The 2026-10-14 protocol and public collector remain preserved as historical artifacts. Its 3-market public smoke is excluded from prospective quality; it met the one-second deadline with 0.480–0.573 s decision-to-prediction latency. See [protocol and deferred status](BTC_PROSPECTIVE_TEST_20261014.md) and [smoke summary](../reports/btc_prospective_test_20261014/public_smoke_summary.json).

## Next experiment and run policy

1. The four-family ablation was stopped after the scope correction; do not resume it. Follow the single joint VP+RP pipeline and bounded budgets in [BTC_VP_RP_TUNING_20261016.md](BTC_VP_RP_TUNING_20261016.md).
2. Freeze development selection before the one historical holdout diagnostic. Do not use holdout results to revise features, generator configs, or model parameters.
3. Keep the current helper and pruner; no new helper or pruning experiment is part of this work.

No full main-model training/selection run, deployment refit, live change, order, or prospective collection was launched during this audit. The new workloads were the 36-fit helper-model CPU pilot, the small CV correctness/performance fixture, and targeted tests. The subsequent ablation interruption and replacement pipeline are documented above and in the linked 2026-10-16 status.
