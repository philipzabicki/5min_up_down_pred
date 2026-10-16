# BTC joint VP+RP tuning — 2026-10-16

> **Superseded by the single full-development fit.** This document below is the historical plan for the five-pipeline procedure. PID `25464` was controlled-interrupted on 2026-10-16 at 02:18:52 UTC after the scope changed. Its manifests are marked `interrupted_scope_change`; the old runner refuses to resume them. See [the replacement procedure](BTC_FULL_DEVELOPMENT_PIPELINE_20261016.md) and the [current research status](BTC_RESEARCH_STATUS_20261010.md).

## Scope and current state

The four-family `base / base+VP / base+RP / base+VP+RP` ablation was stopped after the scope changed. Its last saved location was `calibration_seed/base_plus_rp/feature_selection`. `base_only` and `base_plus_vp` completed only for the calibration seed; `base_plus_rp` feature selection was partial, and `base_plus_vp_rp` did not complete. No four-way comparison exists. The root manifest records `interrupted_scope_change`; donor artifact contents and source hashes, databases, and checkpoints were preserved.

This replacement runs one candidate, `base_plus_vp_rp`, through all five existing outer folds: calibration seed, development A/B/C, and one final historical holdout diagnostic. Existing feature families, fold boundaries, purge rules, target, corrected CV objective, helper model, and pruning settings remain in force. The candidate is selected only on common development observations against the frozen A baseline. July–October outcomes were previously viewed, so the single holdout evaluation is descriptive and cannot change the model or policy. No 240-day observation, deployment refit, live change, or order is part of this work.

**Execution state (2026-10-16 00:32:31 UTC):** the single run is active as PID `25464`, started at `00:32:02 UTC`, using `run_btc_preopen_vp_rp_tuning.py`. The top-level manifest is `running`; the active nested run is `calibration_seed`, stage `reaction_profile`. The compatible indicator stage was verified and reused in 0.0 s. The fresh RP study `btc_preopen_reaction_profile_e2a733d08d94` has 1 COMPLETE, 0 PRUNED, 0 errors, and 23 queued parameter-only seeds; its first recomputed score is 0.69316685 after 16.9 s. The launch snapshot showed 52.5 GiB free RAM, 368 GiB free disk, and 1.0 GiB of 8.2 GiB GPU memory used. At the 00:32:19 UTC resource check, 48.1 GiB RAM was free and GPU use was 2.2 GiB at 20%. No unrelated process was stopped. This is an active checkpoint, not a completed pipeline; the current progress file is under `data/analysis/polymarket/BTC/btc_preopen_vp_rp_tuning_20261016/shared_runs/calibration_seed/9ca4755fd7b45c9f/stages/reaction_profile_e2a733d08d94/trial_progress.json`.

## Audited reuse and recomputation

| Artifact or result | Treatment | Reason |
|---|---|---|
| 64-individual × 8-generation base-indicator fits for each of the five folds | Reuse, with source manifest/config and stage signatures checked | Exact raw-data, target, fold, timestamp, seed, population, generation, and indicator-profile compatibility. Source paths and hashes are frozen in the experiment config. |
| Earlier VP/RP study parameter values, 24 per generator and fold | Enqueue as parameter-only seed trials into fresh studies | Some source studies used the earlier CV ranking. Regardless of source, every seeded candidate is rescored under the corrected mean log-loss + 0.75 × population fold-standard-deviation objective; no prior score enters the new ranking. |
| Existing raw market data, official outcome index, T-60/T-59 snapshots, fold contracts, and frozen A predictions | Reuse after hash and contract checks | These inputs are independent of the newly selected VP/RP values and are required for common-sample development comparison. |
| Old generator scores, VP/RP-dependent datasets, target-weight results, selector output, model trials/fits, calibration, or candidate predictions | Recompute | Generator changes invalidate all dependent stages. The interrupted selector is partial; no downstream ablation artifact is treated as a compatible result. |
| Existing frozen A baseline model and its historical report | Reuse | It is the registered comparison point, not a new training run. Its development reference was log loss 0.692246, Brier 0.249550, AUC 0.524894 under the earlier CV iteration rule. |

The seed parameters and indicator donor identities are hash-checked by `run_btc_preopen_model_update.py` before training. The new output directory and Optuna studies are separate from both the interrupted ablation and the historical baseline. The source audit and per-fold donor hashes are in [`btc_preopen_vp_rp_tuning_20261016.json`](../configs/research/btc_preopen_vp_rp_tuning_20261016.json).

## Frozen budgets and checkpoints

| Stage | Hard evaluated-trial cap | Cumulative time cap | Checkpoints | Expected result |
|---|---:|---:|---|---|
| VP, each fold | 96 | 2,700 s per generator stage | 24, 48, 96 | Time cap can stop the study before 96; report COMPLETE, PRUNED, errors, pending, elapsed time, best CV score, fold mean/std, improvement, best iteration, and boundary hits. |
| RP, each fold | 96 | 2,700 s per generator stage | 24, 48, 96 | Same rules and corrected objective as VP. |
| Main model, each fold | 60 | 7,200 s for tuning | 10, 20, 30, 40, 50, 60 | 15–24 COMPLETE trials estimated from earlier trial durations; this is an estimate, not a minimum guarantee. A separate 300 s reserve is held for the final fit. |

These are cumulative limits across resume attempts. The generator limit is at most 96 evaluations, not 96 additional trials after each checkpoint. VP/RP uses mean decision-row log loss plus 0.75 × population fold standard deviation; main-model CV uses the existing mean plus 0.5 × population fold standard deviation. Main-model CV retains the six-dimensional search, 10 chronological folds, 500-tree cap, early stopping, helper, and pruning policy. Feature selection remains after the complete VP/RP candidate dataset has been built, once per fold. Generator, dataset, selector, main-model fit, calibration, and validation retain their existing order and causal fold-local inputs.

The proxy target remains `Close[Opened+6m] >= Open[Opened+2m]`, with the label available at `Opened+7m`. Nominal decision time is T-60; the readiness/execution snapshot is T-59. No supervised stage may use labels from its scored outer period. Parameter-space bounds, objective coefficients, target weights, purge rules, and feature exclusions are unchanged.

The measured historical VP/RP searches took about 300–480 seconds for 24 trials. Prior 2,430-second main-model windows produced about 5–8 COMPLETE trials. The 7,200-second cap projects 15–24 COMPLETE trials. Summing bounded new search budgets and historical target-weight, selector, feature-build, fit, and verification measurements gives an estimated end-to-end runtime of roughly 28–34 hours across all folds; this is a planning estimate, not a run result. The run will use one process and cumulative stage manifests; resume with the same command only after confirming no copy is active.

## Evaluation and cost reporting

Development selection uses identical official-outcome rows for the candidate and frozen A. It reports log loss, Brier, AUC, calibration bins, per-fold quality, paired differences with the preregistered 3-day block intervals, coverage, feature counts, checkpoint curves, and recorded training time. The candidate passes the frozen gate only if its paired 95% development interval versus A has an upper bound below zero; otherwise A remains selected. Holdout results do not change features, tuning, calibration, or policy.

Training costs will compare the recorded end-to-end nested model-pipeline elapsed seconds for all five candidate and existing A folds. This is historical elapsed time, not a controlled benchmark; the old A runs may have reused cached stages. Inference cost will be measured once on the same holdout feature samples for the candidate and A bundles, with one warm single-row CPU prediction thread. The report separates LightGBM prediction from Platt calibration and excludes feature generation, vector allocation, and network/order submission.

Final tables, checkpoint curves, evaluation summary, and the generated report are written under `reports/btc_preopen/btc_preopen_vp_rp_tuning_20261016/` and `reports/btc_preopen/btc_preopen_vp_rp_tuning_20261016.md`. Until the run completes, quality, training cost, and inference cost outcomes remain pending.

## Run and resume

From the repository root, run one command with no console arguments:

```powershell
python run_btc_preopen_vp_rp_tuning.py
```

The process resumes compatible fold stages, Optuna studies, and cumulative budgets from the isolated output directory `data/analysis/polymarket/BTC/btc_preopen_vp_rp_tuning_20261016/`. Do not start a second copy. No additional four-way ablation or 240-day observation is authorized by this plan.
