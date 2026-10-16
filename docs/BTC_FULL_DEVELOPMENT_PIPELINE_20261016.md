# BTC full-development fit and single holdout evaluation

## Frozen scope

- Development market starts are before `2026-07-01T00:00:00Z`.
- Holdout market starts run from `2026-07-01T00:00:00Z` through `2026-10-06T23:55:00Z`, inclusive.
- The source data cutoff is exclusive at `2026-10-07T00:00:00Z`.
- The first holdout nominal decision is `2026-06-30T23:59:00Z`. The fit may use a label only when its actual availability is no later than that decision; this excludes late-June labels that become available later.
- The Binance COIN-M proxy target remains `Close[Opened+6m] >= Open[Opened+2m]`, with its label available at `Opened+7m`. Nominal decisions use T-60; readiness and hypothetical execution use T-59. Availability-time purge rules remain those in [the BTC research contract](BTC_RESEARCH_CONTRACT.md).
- The July-October period and overlapping BTC observations were viewed in prior repository experiments. The holdout is one descriptive evaluation. Its results cannot change the model or policy.

## One pipeline

The replacement has a separate config, output directory, and manifest. It calls the existing resumable training engine once, on the complete permitted development range. It has no outer fold list, no required `calibration_seed`, and no A/B/C completion gate.

The pipeline reuses the compatible, hash-verified base-indicator artifact and the existing full-development VP/RP study values only as parameter seeds. Those seed values are rescored in fresh studies. The interrupted shorter-range trial scores are not imported. The RP and VP studies each have one study and at most 96 total trials, with checkpoints at 24/48/96 and the existing chronological CV objective, helper, and pruning behavior.

After the generators are frozen, the engine builds one dependent candidate dataset, evaluates 16 target-weight candidates, runs one feature-selection sweep, and runs one main-model study. The main-model study has at most 60 total trials and a cumulative 7,200-second tuning budget, plus a separate 300-second final-fit reserve. Each tuning candidate is scored on 10 chronological folds; the final model is then fit once on all rows whose labels are available by the frozen fit boundary.

| Work | Studies or evaluations | Internal fits/evaluations |
|---|---:|---:|
| Reaction profile | 1 study, at most 96 trials | At most 960 chronological fold evaluations |
| Volume profile | 1 study, at most 96 trials | At most 960 chronological fold evaluations |
| Target weights | 16 candidates | 160 chronological fold evaluations |
| Feature selection | 1 sweep, at most 60 top-k candidates | At most 600 top-k fold evaluations, plus the selector's ranking fits |
| Main model | 1 study, at most 60 trials | At most 600 chronological fold fits; pruning and early stopping can reduce this |
| Calibration predictions | No search | 10 fixed-parameter main-model fits on the main study's chronological folds |
| Final model | No search | 1 full-development fit |
| Holdout | No search | 1 probability evaluation and 3 frozen-policy portfolio scenarios |

The calibration fits reuse the selected model parameters and iteration count. They do not rerun generator searches, target-weight selection, feature selection, or hyperparameter search. Their predictions are chronological validation predictions conditional on the generators, weights, features, and model parameters selected on development. They support calibration; they are not an independent estimate of the full adaptive selection process. The final model is separately fit on the complete eligible development set.

No holdout rows enter generator tuning, target-weight selection, feature selection, model tuning, or calibration. The model and calibrator are frozen before holdout scoring. The single holdout portfolio evaluation uses the verified T-60/T-59 snapshots and existing accounting logic; it does not replay archived book events.

## Reuse and recomputation

- **Reused:** raw candles, official market outcomes, source contracts, frozen policy, T-60/T-59 snapshots, and compatible base indicators, after their existing hash and contract checks.
- **Parameter seeds only:** full-development VP/RP candidate parameter values. Their old scores do not enter the new studies.
- **Recomputed:** VP/RP scores for the target development scope, the selected-generator feature dataset, target weights, feature selection, main-model tuning, conditional OOF calibration predictions, final model, calibrator, and the one holdout evaluation. These depend on the new single-pipeline fit or on the selected generator configuration.
- **Preserved as history:** the interrupted five-pipeline run, its completed stages, Optuna databases, and partial feature-selection artifacts. Its manifest status is `interrupted_scope_change`; its launcher refuses to resume later folds.

## Runtime estimate recorded before training

The estimate uses measured durations from the existing full-development pipeline and the interrupted 96-trial shorter-range generator studies. The prior 28-34 hour estimate for five complete pipelines does not apply.

| Stage | Executions | Measurement basis | Estimated time |
|---|---|---|---:|
| Base indicators | 1 signature check; reuse cached output | Reused stage took 0.03 s | 0-5 s |
| Reaction profile | 1 study, up to 96 x 10 CV evaluations | Interrupted 96-trial study took 1,528 s; 2,700 s cap | 1,800-2,700 s |
| Volume profile | 1 study, up to 96 x 10 CV evaluations | Interrupted 96-trial study took 1,261 s; 2,700 s cap | 1,500-2,700 s |
| Target weights | 16 x 10 CV evaluations | Full-development stage took 8,171 s; 10,800 s cap | 8,171-10,800 s |
| Candidate dataset | 1 rebuild | Full-development build took 223 s | 223-600 s |
| Feature selection | 1 sweep, 10 folds | Full-development stage took 1,802 s; 2,400 s cap | 1,802-2,400 s |
| Main-model tuning | 1 study, up to 60 x 10 folds | Prior 40-trial study took 945 s; 7,200 s cap | 3,000-7,200 s |
| Calibration predictions | 10 fixed-parameter fits | 300 s reserved | 30-300 s |
| Final fit and calibration | 1 final fit and 1 calibrator fit | Prior fit took 2.8 s; final-fit reserve 300 s | 5-300 s |
| Holdout and snapshot economics | 1 score, consistency check, 3 policies | Prior consistency check took 270 s and snapshot diagnostic 142 s | 420-900 s |
| **End-to-end estimate** | **One pipeline and one holdout evaluation** | **Stages above; cache reuse included** | **About 5-8 hours** |

The estimate is planning guidance, not a completion promise. The machine resource snapshot is recorded in the new run manifest. The run uses one process and the configured 20 threads; cumulative stage budgets and trial checkpoints are stored by the existing stage engine.

## Resume command

From the repository root, run the same command to start or resume the experiment:

```powershell
python run_btc_preopen_full_development_fit.py
```

The script accepts no command-line arguments. It checks the separate experiment identity, frozen inputs, snapshots, and artifact hashes before continuing the new manifest. The run completed on 2026-10-16; its results and recovery history are in [the final report](../reports/btc_preopen/btc_preopen_full_development_fit_20261016.md) and [current research status](BTC_RESEARCH_STATUS_20261010.md). Do not resume the superseded runner.
