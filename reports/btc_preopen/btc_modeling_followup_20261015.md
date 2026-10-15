# BTC modeling follow-up — 2026-10-15

## Status

Preflight audit and focused validation are complete. The four-family VP/RP ablation started at `2026-10-15T00:45:13Z` (local PID 9964) after the separate four-worker refit exited. At the `00:47:24Z` checkpoint it was on outer fold `calibration_seed`, variant `base_only`; the 1m ADX indicator search had completed generation 8/8 and saved its result, while the full indicator stage remained in progress. This is progress, not completion. Resume from the repository root with `python run_btc_preopen_feature_ablation.py`; stage manifests and Optuna studies are under the run directory. Standard output and errors are captured in ignored `launcher.stdout.log` and `launcher.stderr.log` files there. The feature-parity and store-memory jobs remained active and untouched.

Starting point: clean `main` at `320e79531379cc015a3fa88d79b0145e3a32a3d5`, tracking `origin/main`. The host reports 20 logical processors, 64 GiB RAM, RTX 4060 8 GiB, and about 514 GB free on `D:`. The separate stage-5 adapter parity task is also active. Neither process was changed or stopped.

## Effective tuning actually used

The source is the 2026-10-12 model-update run manifests and their Optuna SQLite studies, cross-checked against the saved model-update manifest. The row labeled `M` uses the runner's 2,700-second model-tuning-and-fit stage override, of which 270 seconds are reserved for final fitting; the effective cumulative tuning allowance is 2,430 seconds. It requests 10 trials and requires at least 5 successful COMPLETE trials. Pruned attempts and resumptions mean 10 is not a hard ceiling on the stored trial count. Main-model CV used 10 chronological folds, six parameters, a 500-tree cap, and 50-round early stopping. The `G` rows use 24 requested trials, a 2,700-second stage cap, at least 5 COMPLETE trials, 10 CV folds, and a 300-tree helper with 40-round early stopping. VP has 20 dimensions and RP 12.

“Last improvement” below means the last strict best-so-far change in the final CV objective, with the elapsed time from the first trial start and the improvement size. No statistical or practical significance threshold was preregistered. Boundary counts inspect parameter values in the five best COMPLETE trials (fewer than five when fewer than five trials completed). The cap column counts COMPLETE trials whose recorded best iteration reached the tree limit; PRUNED runs are not tree-cap hits.

| Scope / fold | Variant and study ID | Limit | COMPLETE / PRUNED / FAIL | Study wall time | Tuned parameters | Last strict improvement | Best-five boundary values | Tree-cap hits |
|---|---|---|---:|---:|---:|---|---:|---:|
| Development A | Shared VP, `volume_profile_f031ac524e97` | G-VP | 24 / 0 / 0 | 356 s | 20 | #15 @ 240 s, Δ 0.0000575 | 0 / 100 | 0 / 24 |
| Development B | Shared VP, `volume_profile_72282e3aced8` | G-VP | 24 / 0 / 0 | 350 s | 20 | #23 @ 349 s, Δ 0.0000040 | 2 / 100 | 0 / 24 |
| Development C | Shared VP, `volume_profile_dcab0f4c83a1` | G-VP | 24 / 0 / 0 | 341 s | 20 | #12 @ 194 s, Δ 0.0000491 | 0 / 100 | 0 / 24 |
| Development A | Shared RP, `reaction_profile_fc9f0f1d1ffd` | G-RP | 22 / 2 / 0 | 429 s | 12 | #20 @ 392 s, Δ 0.0000541 | 1 / 60 | 0 / 22 |
| Development B | Shared RP, `reaction_profile_27867cc98f6f` | G-RP | 22 / 2 / 0 | 448 s | 12 | #22 @ 429 s, Δ 0.0000157 | 1 / 60 | 0 / 22 |
| Development C | Shared RP, `reaction_profile_3b2f7fd09fe7` | G-RP | 22 / 2 / 0 | 448 s | 12 | #16 @ 330 s, Δ 0.0001750 | 0 / 60 | 0 / 22 |
| Development A | `A_reoptimized`, `model_tuning_5ef757555e14` | M | 8 / 2 / 0 | 346 s | 6 | #4 @ 224 s, Δ 0.0000117 | 0 / 30 | 0 / 8 |
| Development A | `B_dynamic_price_volume`, `model_tuning_07440f35c4ff` | M | 7 / 3 / 0 | 329 s | 6 | #4 @ 227 s, Δ 0.0000406 | 0 / 30 | 0 / 7 |
| Development B | `A_reoptimized`, `model_tuning_962eeed2eb17` | M | 5 / 10 / 0 | 388 s | 6 | #2 @ 138 s, Δ 0.0000592 | 0 / 30 | 0 / 5 |
| Development B | `B_dynamic_price_volume`, `model_tuning_31f5c387b911` | M | 5 / 5 / 0 | 333 s | 6 | #9 @ 333 s, Δ 0.0000064 | 0 / 30 | 0 / 5 |
| Development C | `A_reoptimized`, `model_tuning_d73a56a073c1` | M | 5 / 21 / 0 | 1,142 s | 6 | #11 @ 336 s, Δ 0.0000082 | 0 / 30 | 0 / 5 |
| Development C | `B_dynamic_price_volume`, `model_tuning_7dd6e1872dea` | M | 5 / 14 / 0 | 764 s | 6 | #18 @ 764 s, Δ 0.0000041 | 0 / 30 | 0 / 5 |
| Calibration seed | Shared VP, `volume_profile_65b21bb53dc7` | G-VP | 24 / 0 / 0 | 324 s | 20 | #3 @ 52 s, Δ 0.0000599 | 0 / 100 | 0 / 24 |
| Calibration seed | Shared RP, `reaction_profile_0064a83e0d95` | G-RP | 23 / 1 / 0 | 466 s | 12 | #22 @ 448 s, Δ 0.0000222 | 0 / 60 | 0 / 23 |
| Calibration seed | `A_reoptimized`, `model_tuning_79426f3820b7` | M | 5 / 10 / 0 | 435 s | 6 | #14 @ 435 s, Δ 0.0000347 | 0 / 30 | 0 / 5 |
| Calibration seed | `B_dynamic_price_volume`, `model_tuning_dd2c60592a5d` | M | 5 / 11 / 0 | 1,254 s | 6 | #6 @ 281 s, Δ 0.0000269 | 0 / 30 | 0 / 5 |
| Final fit before exposed holdout | Shared VP, `volume_profile_bb5a9f51a1e3` | G-VP | 24 / 0 / 0 | 483 s | 20 | #18 @ 378 s, Δ 0.0000312 | 2 / 100 | 0 / 24 |
| Final fit before exposed holdout | Shared RP, `reaction_profile_e0c4389b9858` | G-RP | 23 / 1 / 0 | 474 s | 12 | #13 @ 274 s, Δ 0.0002279 | 0 / 60 | 0 / 23 |
| Final fit before exposed holdout | `A_reoptimized`, `model_tuning_3c18194850fc` | M | 6 / 9 / 0 | 547 s | 6 | #2 @ 176 s, Δ 0.0000654 | 0 / 30 | 0 / 6 |
| Final fit before exposed holdout | `B_dynamic_price_volume`, `model_tuning_0d770e6a6293` | M | 4 / 73 / 0 | 2,951 s* | 6 | #72 @ 2,814 s*, Δ 0.0000097 | 0 / 24 | 0 / 4 |

The calibration-seed and evaluated-holdout rows are not development-fold evidence. In particular, the final B model study used only pre-July training labels but did not produce a model: its cumulative tuning clock was about 2,435.7 seconds against the 2,430-second tuning allowance and stopped at 4/5 minimum COMPLETE trials. The `*` wall spans include pauses between attempts; the cumulative-budget manifest is authoritative. Two additional calibration-seed VP/RP SQLite files are duplicate histories with identical candidate objectives; the saved A/B bundles use the study IDs shown above.

The prior audit's main-model A/B counts were transposed. Matching each saved `model_tuning.best_value` and trial number in `model_update_manifest.json` to its SQLite study gives Development A: A `8/2`, B `7/3`; B: A `5/10`, B `5/5`; C: A `5/21`, B `5/14`. The earlier audit listed those pairs in the opposite variant order. Across the three development folds, the top-five COMPLETE configurations hit a search boundary 2/300 VP parameter values and 2/180 RP values; no main-model top-five value hit a bound. No best COMPLETE trial hit a tree cap: VP best iterations were 4–23, RP 1–19, and main-model 14–97. These small generator samples across 20/12 dimensions and main studies with only 5–8 COMPLETE trials indicate constrained search; they do not prove that a wider search improves external quality.

## Target, deadline, and market mapping

For the scheduled market `[12:00, 12:05)` UTC, the row with `Opened=11:58` has its last input candle close nominally at 11:59. The decision is 11:59:00 (T−60). Its `target_window_start_at`/`market_start_utc` is 12:00, used to join the market calendar and preserve that market's `condition_id`. The deadline is 11:59:01 (T−59). The label is `Close[Opened+6m] >= Open[Opened+2m]`: for this row, `Close[12:04] >= Open[12:00]`; with both equal to 101.00, the result is UP and becomes available at 12:05. The market ID join uses this 12:00 market start; it does not key the prediction to the 11:58 candle timestamp.

`tests/test_btc_preopen_contract.py::test_12_00_market_uses_12_00_open_and_12_04_close` checks the hand-computable target, tie rule, 11:59 feature close, and 11:59:01 deadline. I added `test_candidate_join_keeps_the_12_00_market_id_for_its_t_minus_60_prediction` to cover the missing link from the 12:00 forecast start to the market calendar's `condition_id`. The existing fold validator also checks `market_start = nominal_decision + 1 minute`, prediction readiness at `decision + 1 second`, and label availability at `market_start + 5 minutes`.

The one-second deadline is a frozen end-to-end contract. Historical batch rows contain candle event time, not receipt time; the batch's one-second field is a nominal readiness rule, not measured latency. The separate three-market public smoke measured T−60-to-prediction at 0.480–0.573 s, including its observed candle-pair wait, feature construction, and inference. It did not submit an order or measure order acknowledgement/fill, so the whole live decision-to-execution path has not been measured. No T−59 redefinition or target correction is needed; dependent historical artifacts remain unchanged.

## CV objective, saved models, and pruning

The corrected shared callback keeps the existing objective forms: main model unweighted mean decision-row log loss plus `0.5 ×` population fold standard deviation; VP/RP mean plus `0.75 ×` fold standard deviation. The 2026-10-12 runner disabled recency fold weights. The same per-iteration score is used for the Optuna report/pruner, early stopping, the returned CV curve's best iteration, and final trial ranking. The real VP/RP runner integration tests and callback tests passed.

The saved-fold check is direct: all nine existing completed A/B fit bundles have `Booster.current_iteration()` equal to the trial's selected iteration—seed A/B 57/31; Dev A 42/39; Dev B 74/63; Dev C 46/29; final-fit A 59. Their bundle raw-replay verification status is `verified`; the final-fit B study has no saved model. The production fit passes the selected best iteration as `num_boost_round`, saves that many trees, and prediction uses the loaded model's default (all saved trees). The objective repair's 12,000×24, ten-fold, 60-round CPU benchmark was already measured twice: median 0.416 s before versus 0.414 s after (−0.6%, within timing noise), with no extra fold-prediction pass.

The earlier pruning continuation was a real completion diagnostic, separate from the helper-capacity pilot: five previously PRUNED configurations from different learning-rate regions were run without Optuna pruning on each of three development B folds (15/15, 1,052.2 s end-to-end; 1,026.1 s in model fits). Six runs peaked after their original prune checkpoint; one beat its same-development-fold best COMPLETE objective by 0.000031 and 14 did not. Because source trials came from the separate final-fit B study, this is not a same-study false-prune rate. It is sufficient to answer whether pruned trials were continued; it does not justify changing the frozen pruner. The helper pilot separately reused six fixed generator configurations and did not re-tune VP/RP.

## What the prepared ablation measures

The config selects exactly `base`, `base+VP`, `base+RP`, and `base+VP+RP`. `base` is the modeling dataset's numeric feature set after reserved target/timestamp/raw OHLCV columns are removed, with columns prefixed `vp_` and `rp_` excluded. It retains fold-selected indicator features, candle, session, realized-volatility, basis/premium, and generic candle-volume features; it contains no second VP/RP generator output. The filter runs before target-weight search, feature selection, and main-model tuning. The four variants force no extra candidate features.

This is a comparison of the **full fold-local learning procedure with and without the VP/RP feature families**, not a test of one fixed historical VP/RP configuration. The pipeline re-tunes indicators, VP and RP using only each outer fold's earlier price-proxy labels, then separately tunes target weights, selects features, tunes/fits the main model, and fits calibration from earlier chronological OOF predictions. VP/RP and indicator training use no outer validation labels. Model, selection, and target-weight inner CV purge labels by availability time. Official Polymarket outcomes are used only for the external development comparison and are frozen before the single descriptive historical holdout evaluation.

All four variants use the same five outer fold boundaries and the same scheduled market starts. `_merge_predictions` joins one-to-one on `market_start_utc` and checks nominal decision, proxy label, label availability, and fold identity. Evaluation uses the same official outcome target, calibrated probability columns, metrics, and common market sample. The runner's content signatures include raw-data/contract/code identity and fold-local generator artifacts; generator and complete feature-dataset stages are shared within a fold, while target-weight, selector, tuner, calibrator, and model artifacts remain variant-specific. These signatures include the config, features and fold provenance; no learned fit is shared across folds.

## Runtime reconstruction frozen before launch

No full ablation or full variant was run to estimate runtime. The old model-update manifests give complete measured stages for five folds; multiplying only where each variant must refit yields this estimate:

| Component | Workload | Measured basis | Estimated elapsed |
|---|---|---|---:|
| Shared feature generation | 5 folds, once per fold: indicators + VP + RP + feature dataset | 6,609 s summed from the five recorded fold manifests | 1.84 h |
| Target-weight search | 4 variants × 5 folds = 20 | 2,185–2,771 s per old variant/fold; mean 2,478 s | 12.14–15.39 h |
| Feature selection | 4 × 5 = 20 | 1,602–1,804 s per old variant/fold; mean 1,687 s | 8.90–10.02 h |
| Main-model tuning | 4 × 5 = 20 | Existing per-fold A/B study wall spans; holdout B capped at its 2,430 s tuning allowance when projecting active work | 4.43–4.72 h |
| Final model fit | 20 | Existing fold final fits 2.7–7.1 s each | about 0.02 h |
| Bundle raw-replay verification | 20, since each model is different | 164–267 s per old pipeline; mean about 209 s | about 1.16 h |

The reconstructed end-to-end projection is about **28.5–33.2 hours**, plus small final comparison/report overhead. It is higher than the earlier 25–29 hour headline because this sum explicitly includes the per-variant point-in-time bundle verifications and the five-fold main-model tuning estimate. The range is a sum of observed stage minima/maxima, not a statistical interval. Existing data provide representative measurements for each expensive stage, so no isolated full-variant timing run is needed. The config still freezes the same 24 generator trials, 16 weight candidates, selector settings, ten-fold CV, and ten requested main-model trials per variant/fold.

## Run results and progress

Launched under Python 3.14.4 with the frozen 20-thread/GPU profile. The launch snapshot showed 53.8 GiB free RAM and 969 MiB of 8,188 MiB GPU memory used; the parent process initially held 2.98 GiB RSS. During the first indicator batch, the 20-worker process tree briefly summed to 40.3 GiB working set and free RAM reached 15.5 GiB; after the batch workers exited, free RAM returned to 51.8 GiB. The observed 15-second aggregate CPU use was 203 seconds. I set this run's process tree to BelowNormal priority while the unrelated user tasks remained active; its tuning/data settings were not changed. At `00:47:24Z`, the 1m ADX search had saved generation 8/8, but the indicators stage was still running and the outer-fold manifest remained in progress. Current progress must be read from the top-level `model_update_manifest.json`, per-fold `run_manifest.json`/`run.log`, and stage Optuna SQLite databases under `data/analysis/polymarket/BTC/btc_preopen_vp_rp_ablation_20261015/`; stdout and stderr logs are in that ignored run directory. The model-update runner records completed folds and resumes compatible stages by their frozen signatures. No 240-day observation, real trade, or deployment refit is part of this run.

### Sources

- [Prior modeling audit](../../docs/BTC_MODELING_AUDIT_20261015.md)
- [Prior helper diagnostic](btc_generator_helper_diagnostic_20261015.md)
- [Prior pruner continuation report](b_pruner_development_diagnostic_20261014/pruner_diagnostic_report.md)
- [Frozen ablation config](../../configs/research/btc_preopen_vp_rp_ablation_20261015.json)
