# BTC pre-open experiment v1

## Result

This run completed a causal, full-minute **raw-feature LightGBM baseline** for the one-minute-before-market task. It did not refit the repository's indicator, reaction-profile, volume-profile, observation-weight, feature-selection, or Optuna layers. Those historical artifacts encode a different target and decision minute, so the baseline deliberately did not consume them. The corresponding target-specific searches remain outstanding; this report does not call the run a completed full-stack experiment.

The baseline has a small positive historical prediction result against official Polymarket outcomes: calibrated log loss was 0.69238 versus 0.69315 for constant 0.5, Brier score was 0.24962 versus 0.25, and AUC was 0.51961 versus 0.5. The paired three-day block-bootstrap 95% interval for the calibrated model's log-loss difference versus 0.5 was [-0.001495, -0.000018]. The upper bound is close to zero. The period was analyzed in earlier repository experiments, so this is historical evidence with prior exposure, not a pristine holdout.

There is no historical evidence here that the signal could be traded profitably after costs. The archived market-book snapshots start at market open, not one minute before it; there are zero eligible pre-start quote rows. The report therefore contains no economic replay or assumed 0.50 fills.

## Shared task contract

The machine-readable contract and split are in [btc_preopen_v1.json](../configs/btc_preopen_v1.json). The training artifact and evaluation lineage are in [btc_preopen_experiment_manifest_20261004.json](btc_preopen_experiment_manifest_20261004.json).

For the requested example, the row timestamp is the candle's `Opened` time:

| Event | UTC time | Use |
|---|---:|---|
| Feature candle opens | 16:43 | `Opened = 16:43` |
| Feature candle closes | 16:44 | Its full OHLCV may be used only after close and receipt |
| Nominal decision | 16:44 | Prediction availability is measured separately at runtime |
| Polymarket window opens | 16:45 | One minute after nominal decision |
| Polymarket window closes | 16:50 | Five-minute position resolves using the official market outcome |
| Proxy target prices | `Open[16:45]` through `Close[16:49]` | Equivalent to `Close[Opened+6m] >= Open[Opened+2m]` |
| Proxy label available | 16:50 | The close of the last target candle; later than the nominal decision |

The minute rows retain the same rolling target at every `Opened`: the future five-minute window starts at `Opened + 2m` and ends at `Opened + 7m`. Only rows with UTC `Opened.minute % 5 == 3` correspond to real one-minute-prestart decisions. Every row remains in the training set; non-decision rows are auxiliary examples. The principal evaluation uses only real decision rows. Ties are UP. A proxy label is missing if any of the five target candles is absent. The proxy target is Binance COIN-M BTCUSD index price, while the external label is the official Polymarket result resolved from the market's Chainlink source.

The collector records the nominal decision, candle receive time, prediction start and finish, quote request and receive times, source book timestamp, prediction-window boundaries, and a hash-derived bundle version. Its timestamps preserve source and local receipt separately; historical batch rows do not contain measured inference or execution latency.

## Data split and artifact lineage

The split was frozen in the contract before the model run:

- Fit and iteration selection use label-availability times before 2026-01-01 00:00 UTC.
- Calibration uses 30,155 actual decision rows from 2026-01-01 through the decision at 2026-04-15 16:54 UTC. Its latest proxy label was available at 17:00 UTC.
- The first external decision is 2026-04-15 17:04 UTC, for the market starting at 17:05 UTC. The external market period ends at 2026-05-18 10:30 UTC.
- At the internal fit/validation boundary, rows were purged by comparing `target_label_available_at` with the first validation decision timestamp. Five rows were removed. Calibration labels were also required to be available before the first external decision.
- No external-test labels were used for model fitting, iteration selection, or calibration in this run. Earlier repository artifacts and analyses had already used this test period and overlapping BTC observations; the period is marked historically exposed.

The runner consumed 3,321,148 one-minute rows and 29 causal raw-candle features. Final fitting used 2,925,224 eligible minute rows, with unit sample weights. The selected iteration count was 19. The Platt calibrator was fit only on later development proxy labels. The model, local metadata, calibrator, feature-source manifest, verified feature cache, predictions, smoke output, and evaluation are stored under the paths listed in the JSON manifest; model and data files are local ignored artifacts and are not committed.

The internal smoke fit used 19,995 earlier rows and 5,000 validation rows, selected before the main run. It confirmed finite predictions, the target timestamp alignment, no external-test rows in fitting, and timestamp-based label purging. The reusable feature cache is about 688 MB and is keyed by raw-data, target/contract, feature-code hashes, and ordered features.

The run used all eligible one-minute training rows with an 8-thread CPU LightGBM budget on the available 20 logical processors. The model parameters, seed, thread count, round budget, row counts, and artifact hashes are in the manifest. Earlier fit invocations were resumed, but the original process-level timing and peak memory were not retained; the logged 2.0-second resume is not the end-to-end training time. I leave those measurements unreported rather than present a cached resume as the full cost.

## Target-dependent pipeline audit

| Existing stage | What it learns and its historical target/split | Causality and pre-open handling |
|---|---|---|
| Indicators (`fit_indicators.py`, `configs/indicator_fit.json`) | Genetic searches across ADX, Bollinger Bands, Chaikin Oscillator, Keltner Channel, MACD, and Stochastic Oscillator. Candidate features are scored against configured auxiliary `ahead_ret` or `candle_up` proxy targets over time segments; the default selection score is extremes-versus-middle information ratio. | Feature values are computed from candle history available at the current closed row. The learned candidate parameters and selected feature list depend on labels. The saved search outputs were not reused because they were fit on broader/later history and different target definitions. The existing segment/gap setup is row-based rather than purging from label-availability timestamps. |
| Reaction Profile (`fit_reaction_profile.py`) | 500 Optuna trials, 10 walk-forward folds, LightGBM scoring, old `target_5m_candle_up`, and the existing decision-only weight threshold. | The profile state is updated as each candle closes; a reaction is incorporated only after the candle that reveals it. State construction is causal. The fit target is still the old close-to-close five-minute label, and folds do not purge by the new target's label-availability time. The old fitted parameters were not reused. |
| Volume Profile (`fit_volume_profile.py`) | 500 Optuna trials and 10 walk-forward folds, scored through LightGBM on the same old five-minute target and existing decision-only weights. | Profile values use closed-candle history and are causal. Learned profile parameters and old decision-row timing do not match this task; old fit outputs were not reused. |
| Observation weights (`optimize_target_weights.py`) | Candidate per-row weights are searched using walk-forward OOF predictions and balanced accuracy on the configured decision subset; the existing decision subset is UTC minute modulo 5 equal to 4. The script has a shorter proxy search and a final 10-fold evaluation. | This target/decision convention differs from the new modulo-3 pre-start rows. The existing weight artifact was not used. The baseline uses unit weights, and all comparison metrics are fixed, unweighted metrics on the same official market rows. |
| Feature selection (`select_features.py`) | Feature ranking, permutation importance, and top-K comparisons over separate 10-fold walk-forward evaluations, using the existing target weights and target. | Learned selection may depend on labels and inherited feature artifacts. Existing selections were not consumed. No new feature search was run for this raw-only baseline. |
| LightGBM search and OOF (`optimize_lgbm_hyperparameters.py`, `train_lgbm.py`) | The repository's search requests 25 Optuna trials over 10 walk-forward folds; the training stage emits weighted OOF and fits a final model with the configured target, weights, and features. | The prior pipeline uses row-index folds and does not apply the new timestamp-based purge at each fold boundary. Its old OOF, fitted model, and hyperparameters were not reused. The baseline instead selected 19 iterations on an earlier chronological development validation and purged unavailable labels at its boundary. |
| Calibration and comparison (`run_btc_new_model_comparison.py`) | Historical model-comparison outputs evaluate saved models and calibration on earlier repository split definitions. | Those outputs were not used. This run fits a new Platt calibrator from later development predictions against the new price proxy; it does not fit to external official labels. |

In particular, `utils.data.compute_binary_close_target_from_opened` defines the old target as future close versus current close, while `compute_target_weights_from_opened` gives higher weight to rows with minute remainder 4. Neither convention is the pre-start contract above. The reaction and volume profile calculations themselves are causal, but causal feature values do not make target-dependent fitted parameters independent of a later test.

## External prediction results

The price proxy was available for 9,426 real decision rows. Nineteen had no matching official market record. The shared evaluation uses the 9,407 matched markets and compares every model/baseline on identical rows. The proxy and official outcome disagreed on 400 rows (4.25%).

| Label and forecast | Log loss | Brier | AUC |
|---|---:|---:|---:|
| Official Polymarket — constant 0.5 | 0.693147 | 0.250000 | 0.5000 |
| Official Polymarket — development prevalence | 0.693147 | 0.250000 | 0.5000 |
| Official Polymarket — raw model | 0.692475 | 0.249664 | 0.5196 |
| Official Polymarket — Platt model | 0.692384 | 0.249620 | 0.5196 |
| Binance index proxy — constant 0.5 | 0.693147 | 0.250000 | 0.5000 |
| Binance index proxy — raw model | 0.692228 | 0.249541 | 0.5254 |
| Binance index proxy — Platt model | 0.691865 | 0.249362 | 0.5254 |

Calibration reports, reliability bins, all paired bootstrap intervals, and full-precision metrics are in `data/analysis/polymarket/BTC/preopen_v1/evaluation.json`. Paired uncertainty uses 2,000 circular three-day calendar-block bootstrap replications with seed 20261004. The improvement on official labels is modest and should be treated cautiously because the period is historically exposed and the calibrator learned the proxy, not official outcomes.

## Economic data and prospective collection

No retained book observation was timestamped before market start in the selected historical period. The closest archived snapshots begin at market start, which is too late for the requested one-minute-before entry. A portfolio replay cannot be estimated from those snapshots; missing pre-start asks were not imputed.

The separate observation-only runner is `python run_btc_preopen_collection.py`. It uses the same raw-feature function and frozen model/calibrator, records both sides' book depth and timestamps, and polls later official outcomes. Its current database is `data/analysis/polymarket/BTC/preopen_collection_20261004_v3/shadow.sqlite3`; the earlier v1 and v2 session data are preserved in separate paths. Orders are disabled, hypothetical trade decisions are disabled, and no active model configuration is changed. The existing entry-state collector database and process are separate and are not reused by this runner.

The first prospective record was market `btc-updown-5m-1791132300`, starting at 16:45 UTC on 2026-10-04. The 16:43 candle arrived at 16:44:01.661; prediction finished at 16:44:01.686; all book inputs were received by 16:44:02.553. UP showed 0.48 bid / 0.49 ask and DOWN showed 0.51 bid / 0.52 ask. The recorded source book timestamp was 16:44:02.008 and local receives were recorded separately. The calibrated UP probability was 0.50560, bundle version `btc-preopen-v1:4ed7e34d94bc42a0`, and no trade decision was enabled. Polymarket later resolved the market UP at 16:50:53 UTC. This is one prospective observation, not evidence of forecast quality or tradable profit.

The next two v2 snapshots were preserved with their source timestamps, local request/receive times, bids, asks, sizes, and depth, but the old freshness check marked them invalid because source timestamps were 60–87 ms later than local receive timestamps. That comparison treated the two clocks as synchronized. The v3 collector accepts source times up to one second ahead of local receipt as suspected clock skew, still rejects timestamps older than the configured 30-second limit or more than one second into the future, and records the raw signed difference and both clocks. The v2 database is retained; collection now uses a new v3 database so the earlier session is not rewritten.

To reproduce the historical fit and evaluation, run `python run_btc_preopen_experiment.py`; the script resumes verified matching cache/checkpoints and does not accept console arguments. To observe new markets, start `python run_btc_preopen_collection.py` in a separate process after the model bundle exists. The collection script is prospective and does not alter the training result above.

## Conclusion

The one-minute-before signal has a small historical predictive signal in this baseline, including on the matched official outcomes. The evidence does not establish a robust edge: the test period was previously exposed, AUC is only 0.5196, and the calibrated log-loss confidence interval barely excludes no improvement. There are no pre-start historical bids/asks to show whether the forecasts could have been acted on profitably after fees, spread, depth, latency, and cash-lock constraints. Prospective collection is therefore needed for an execution-cost assessment. The complete target-specific fitted stack remains pending.
