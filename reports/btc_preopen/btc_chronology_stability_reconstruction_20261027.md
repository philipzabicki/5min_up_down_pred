# BTC chronology and E0038 stability audit — 2026-10-27

## Scope and decision

This is E0039, a development-only continuation of E0038. E0038 diagnostics below use only its hash-verified `summary.json`, `market_decisions.csv`, and `monthly_paths.csv`; the PMXT book replay was not repeated. No new model fit, outer prediction, economic replay, holdout evaluation, acquisition, pilot, shadow process, DNS request, or order was run.

The E0038 diagnostic and fold-cutoff audit are complete. The requested A/B/C/D chronological reproduction and full runtime benchmark are **blocked before fit/measurement** by specific missing offline orchestration, described below. I froze the folds, data hashes, original procedure budgets, and resume order in [the E0039 protocol](../../configs/research/btc_chronology_reconstruction_20261027.json) before any new fit. The next step is to implement and test the fold-local BASE70 + flow + memory/ESN + official-OOF path, then commit/push that code and a protocol amendment before fitting.

E0038 remains retrospective and coverage-conditional. The 5,604 clean markets are not a complete development calendar. Preserve E0037's historical **FAILED** gate and E0038 v1/v2/v3 unchanged. T0009 remains **BLOCKED** overall; T0010 remains deferred.

## E0038 stability and concentration

### Frozen inputs and cohort

| Artifact | SHA-256 |
|---|---|
| E0038 v3 protocol | `5852107c07e0bb1768fcfe9693322bd8b9b377963110424c8f09ac08e28d8008` |
| Summary | `67f46ab02dcf91b4faf566610b308e89ddff1888dd03bd168ead2b06aee2f84e` |
| `market_decisions.csv` | `10c5e05b564ab351524338e3d14de6412aa11429fd298ab6298c57ff1ad40694` |
| `monthly_paths.csv` | `f4b2f18fe28cfa48fa6e10d8e1b4d5718cfe168cad1c6c3ec2a7fcc25302f813` |

The common clean cohort is 5,604 markets: April 456, May 2,315, June 2,833; first market start 2026-04-15 17:35 UTC, last 2026-06-30 23:35 UTC. It is a subset selected by the frozen source-quality rule, not full-calendar coverage. For every model and control, every cohort market has a decision row. A known `no_trade` contributes zero in the paired diagnostics; a missing market is not converted to zero.

### Independent fixed-$5 results by entry month

Each entry signal executed at the frozen T-59 snapshot; independent diagnostics have no cash constraint. Delay does not change their PnL. “Fee equivalent” is the E0038 combined cash/share fee accounting value; raw fee columns and every control/delay/month are in [the 108-row monthly flow file](btc_chronology_monthly_flows_20261027.csv), SHA-256 `5cd323dcedb3f02b71bd1b5cde4dd8b8523879a3dacae11b997734c075f91209`.

| Model | Month | Cohort | Signals = fills | Gross | Fee equivalent | Payout | Net PnL |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | Apr | 456 | 167 | $835.00 | $29.66 | $802.22 | **−$35.82** |
| A | May | 2,315 | 817 | $4,085.00 | $146.04 | $4,114.34 | **−$116.70** |
| A | Jun | 2,833 | 1,051 | $5,255.00 | $183.50 | $5,665.47 | **+$226.98** |
| B | Apr | 456 | 156 | $780.00 | $27.71 | $820.81 | **+$38.47** |
| B | May | 2,315 | 778 | $3,890.00 | $139.42 | $4,013.02 | **−$16.40** |
| B | Jun | 2,833 | 1,012 | $5,060.00 | $176.83 | $5,436.58 | **+$199.75** |
| C | Apr | 456 | 138 | $690.00 | $24.54 | $707.33 | **+$15.17** |
| C | May | 2,315 | 768 | $3,840.00 | $137.48 | $4,016.85 | **+$39.38** |
| C | Jun | 2,833 | 943 | $4,715.00 | $165.15 | $5,099.02 | **+$218.87** |
| D | Apr | 456 | 149 | $745.00 | $26.46 | $783.14 | **+$35.63** |
| D | May | 2,315 | 763 | $3,815.00 | $136.93 | $3,988.71 | **+$36.78** |
| D | Jun | 2,833 | 950 | $4,750.00 | $166.30 | $5,129.80 | **+$213.50** |

Over the full conditional cohort, independent net PnL is A **+$74.46**, B **+$221.83**, C **+$273.42**, D **+$285.90**. Each result is strongly month-sensitive: June contributes more than the total net for A because April/May are net losses; June contributes 90.0%, 80.1%, and 74.7% of B/C/D total net, respectively. This retrospective split is descriptive and does not define a month filter or sizing rule.

### Concentration by day and trade

| Model | Positive gross PnL | Top five positive days | Share of positive gross | Best day | Worst day | Top 10 winning trades | Share of positive gross |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | $5,144.63 | $326.19 | 6.34% | Jun 9 +$90.14 | May 29 −$135.19 | $101.25 | 1.97% |
| B | $4,999.12 | $380.81 | 7.62% | Jun 9 +$97.16 | May 29 −$134.63 | $101.25 | 2.03% |
| C | $4,787.95 | $309.40 | 6.46% | May 28 +$83.74 | May 29 −$101.20 | $101.25 | 2.11% |
| D | $4,825.99 | $340.27 | 7.05% | May 28 +$102.79 | May 29 −$88.86 | $101.25 | 2.10% |

Positive gross PnL here means the sum of positive market outcomes before netting negative outcomes. The top-five-day measure is not the share of total net. The top-ten trade amount is similar across candidates because the payout structure caps a $5 binary-market trade's upside. Largest negative days are not hidden by the positive total.

### Paired independent-$5 differences

Both comparisons use all **5,604 common known/assessed markets**, including known no-trades as zero. They are not restricted to trades where both models entered. No confidence interval is reported: this is a retrospective comparison with no prespecified time-series interval method, and trade-level bootstrap independence is unjustified.

| Pair | Total difference | Markets positive / zero / negative | Apr / May / Jun | Days positive / zero / negative | Five largest daily differences |
|---|---:|---:|---:|---:|---|
| B−A | +$147.37 | 352 / 4,925 / 327 | +$74.30 / +$100.30 / −$27.23 | 32 / 6 / 22 | May 28 +$72.94; Jun 6 +$63.87; Apr 24 +$33.37; Jun 4 +$25.21; May 31 +$25.15 |
| D−B | +$64.07 | 197 / 5,218 / 189 | −$2.85 / +$53.17 / +$13.75 | 28 / 10 / 22 | May 29 +$45.77; Jun 10 +$40.19; Jun 8 +$28.26; May 23 +$25.15; May 30 +$19.55 |

Largest negative daily differences were B−A: Jun 10 −$53.66, May 30 −$48.82, Jun 1 −$35.80, Jun 11 −$33.86, Jun 5 −$27.36; and D−B: May 27 −$32.15, May 3 −$23.71, Jun 9 −$22.67, Jun 7 −$19.67, Jun 6 −$18.32.

### $100 funded accounts and monthly event snapshots

The following equity is **cash plus locked gross cost basis**, not mark-to-market. The event-ledger reconstruction matches E0038's terminal equity and cost-basis maximum drawdown; peak/trough times are event times. PnL is attributed to entry-month in the separate monthly flow table, while cash/equity changes occur at entry and payout events. All positions settled by the June-end account close; no locked balance remained.

| Model | Delay (s) | Fills / cash rejections | End equity (net PnL) | Cost-basis max DD | Peak equity at UTC | Trough equity at UTC |
|---|---:|---:|---:|---:|---|---|
| A | 60 | 285 / 1,750 | $4.544 (−$95.456) | 96.236% | $120.731 Apr 19 16:26:22 | $4.544 May 7 23:31:20 |
| A | 300 | 285 / 1,750 | $4.544 (−$95.456) | 96.236% | $120.731 Apr 19 16:30:22 | $4.544 May 7 23:35:20 |
| A | 1,800 | 282 / 1,753 | $0.258 (−$99.742) | 99.786% | $120.731 Apr 19 16:55:22 | $0.258 May 8 00:00:20 |
| A | 7,200 | 261 / 1,774 | $0.269 (−$99.731) | 99.778% | $120.731 Apr 19 18:25:22 | $0.269 May 7 13:45:52 |
| B | 60 | 1,946 / 0 | $321.828 (+$221.828) | 78.300% | $451.453 Jun 9 23:06:18 | $35.291 May 26 08:44:01 |
| B | 300 | 1,946 / 0 | $321.828 (+$221.828) | 78.300% | $451.264 Jun 9 23:10:18 | $35.291 May 26 08:44:01 |
| B | 1,800 | 1,946 / 0 | $321.828 (+$221.828) | 78.726% | $450.697 Jun 9 23:35:18 | $34.598 May 26 09:14:01 |
| B | 7,200 | 1,940 / 6 | $323.258 (+$223.258) | 78.579% | $451.259 Jun 10 01:05:18 | $34.763 May 26 10:14:01 |
| C | 60 | 1,849 / 0 | $373.416 (+$273.416) | 78.088% | $439.038 Jun 9 20:11:25 | $29.969 May 13 06:01:53 |
| C | 300 | 1,849 / 0 | $373.416 (+$273.416) | 78.088% | $439.038 Jun 9 20:15:25 | $29.969 May 13 06:05:53 |
| C | 1,800 | 1,849 / 0 | $373.416 (+$273.416) | 78.216% | $438.678 Jun 9 20:40:25 | $29.794 May 13 06:30:53 |
| C | 7,200 | 1,842 / 7 | $370.214 (+$270.214) | 78.344% | $434.740 Jun 9 22:10:25 | $29.619 May 13 06:39:01 |
| D | 60 | 1,862 / 0 | $385.902 (+$285.902) | 74.636% | $477.874 Jun 9 23:06:18 | $36.846 May 13 05:09:01 |
| D | 300 | 1,862 / 0 | $385.902 (+$285.902) | 74.636% | $477.685 Jun 9 23:10:18 | $36.846 May 13 05:09:01 |
| D | 1,800 | 1,862 / 0 | $385.902 (+$285.902) | 74.636% | $477.118 Jun 9 23:35:18 | $36.846 May 13 05:09:01 |
| D | 7,200 | 1,859 / 3 | $391.241 (+$291.241) | 75.051% | $481.760 Jun 10 01:05:18 | $36.153 May 13 07:04:01 |

Every month-end row for the fixed-$5 and 5%-of-free-cash accounts is recorded in [the 96-row account snapshot file](btc_chronology_account_snapshots_20261027.csv), SHA-256 `c4be0e3dbffc3ccdfa74ad9c9d0c8e28ee1ccdffc91605735f41544be24ccee2`. It includes month-end free cash, locked cost, equity, cumulative payouts and fees, drawdown, and unsettled positions. The month-end convention follows event time (`Opened`/decision ledger), which is two minutes before `market_start_utc`; the monthly PnL table instead groups by `market_start_utc`. Thus one boundary market can fall in the prior `Opened` month for account snapshots while remaining in the next month for PnL attribution. Those are different views of the same ledger, not a discrepancy.

The 5%-of-free-cash requests remain subject to the frozen $5 minimum. A/B/D each made one $5 fill and ended at $95 (−$5); C made two fills totaling $10.223 gross and ended at $99.235 (−$0.765). More than 5,500 requests were below the minimum for each model; no minimum was lowered and no request was rounded up. Across the four delay cases C's cash availability changed which below-minimum requests reached the minimum, but its accepted fills and final PnL did not change.

## Pipeline cutoff matrix

| Stage | Observed scope/cutoff and evidence | Status for Apr–Jun 2026 external folds |
|---|---|---|
| Raw BTC and proxy target | E0039 pins exact byte prefixes ending Apr 30, May 31, and Jun 30 23:59 UTC. The contract is `Close[Opened+6m] >= Open[Opened+2m]`, ties UP; target label is available at `Opened+7m`. The raw prefix hashes/row counts are in the [protocol](../../configs/research/btc_chronology_reconstruction_20261027.json). No raw row after Jun 30 was materialized for this audit. | Input and target timing are verified. New fold labels must pass the timestamp mask; no historical receive-time is inferred. |
| Indicator/generator search | The full-development run used population 64 × 8 generations; the 2026-10-16 config records one reused indicator artifact and VP/RP study inputs selected/scored on the full development range through 2026-07-01. The full-development `selected_features.json` hash is pinned in E0039. | **Later development selection.** Refit indicator, volume-profile, and reaction-profile procedures on each earlier prefix. Do not transfer the full-development generator parameters. |
| Taker flow / six Z/V features | The 2026-10-18 config uses source-derived `taker_Z/V` at 1/5/15m, with causal rolling windows and a full-development materialization cutoff 2026-07-01. Safe fold prefixes end at each last decision candle; hashes and timestamps are in E0039. | Inputs are causally available, but the old BASE70 selected-feature list and weights are later-selected. Recompute the feature list and weight search inside each fold. |
| Feature selection | The 64 non-flow BASE70 columns were selected from the full-development feature set; exact artifact SHA is `4b7a2b48…f73d0a81`. | **Later development selection.** Re-run the same bounded selector before every outer window; store fold-specific feature order. |
| Memory scaler/PCA and memory state | E0031 candidate manifests plus E0039 transform pins verify memory scaler/PCA fit on 20,000 fit-core rows ending before 2024-07-01. Transformation uses fixed lags 1/5/15/60/240/720, EWM 15/60/240/720, and resets on invalid inputs or timestamp gaps. | Fit/transform choice is earlier than all outer windows. Reuse the exact frozen scaler/PCA; rebuild state across every available minute before and through each fold. |
| ESN scaler/PCA/config/state | E0031 run manifest verifies fit-core-only scaler/PCA through 2024-07-01. The selected ESN config's search validation ends 2025-07-01, earlier than the external folds. Three reservoir seeds and transforms are hash-pinned in E0039. | Earlier transform and configuration selection are documented for these folds; candidate-family selection remains historically exposed. Stream state through all minutes; do not fit PCA after the cutoff. |
| Model, weights, iterations, ensemble | Frozen E0031 base members were trained on 2,088,350 rows through 2024-07-01 with 31 rounds and seeds 20261101–03; C/D cross seeds 20261023–25. The 0.59/0.1025 target weights and 31-round/model settings were sourced from later full-development Takerflow work; the 64 features were also full-development selected. | Historical base fit is earlier, but model inputs and training choices are not. Refit feature list, weights, and hyperparameters using only inner chronological validation before each outer fold. Retain frozen seed identities/uniform averaging and report changes; no outer early stopping. |
| Official calibration and entry choice | Takerflow's daily Platt rule uses prior official OOF labels released at `max(resolved_at, market_start+5m)+60s`, at least 1,000 outcomes, and both classes. E0038 has no measured prediction-ready timestamps; nominal T-60 availability is a simulation assumption. | **No saved official OOF forecasts exist for the new fold-local A/B/C/D fits.** A new daily calibrator must consume earlier-only OOF predictions, not fit in-sample or use the outer period. Preserve the strictly positive EV policy (`EV > $0`, exact zero rejected, exact tie selects UP) and fee/minimum rule from E0038 v3's `select_positive_ev_side`. |

Evidence classes are explicit: (1) earlier transform fits and ESN configuration selection are supported by their manifests; (2) raw/flow fold prefixes and the chronological label mask are verified for this reconstruction; (3) generator parameters, feature selection, target weights, model tuning and official calibration used later development in the historical candidates; (4) exact E0039 fold-local model/OOF/calibrator hashes are **NOT_CREATED** because the runner is blocked before fit. Missing `prediction_available_at` is not the blocker: the new chronology is a labeled simulation and does not backdate old predictions.

The label-mask helper added to `run_btc_preopen_experiment.py` makes both fit-mask construction sites use the same timestamp rule. Unit tests cover the exact exclusive cutoff and invariant fit prefixes after future rows/labels are appended. This guard alone does not cure later feature/model selection.

## Chronological reproduction status and concrete block

E0039 freezes the requested windows `[2026-04-15, 2026-05-01)`, `[2026-05-01, 2026-06-01)`, and `[2026-06-01, 2026-07-01)` UTC. Each outer fit is cutoff at the window's UTC start, with labels released strictly before that cutoff. Only the E0038 clean whitelist may enter economic comparisons. The planned chain is F1 complete, then F2/F3, one shared $100 path per model/control/delay, no monthly or fold resets, and no PnL-driven revisions.

The first fold was **not** fit because the required E0039 runner is **NOT_IMPLEMENTED**. The available `run_btc_preopen_experiment.run_training_pipeline` is not the frozen A/B/C/D procedure: it builds a single main model and its internal calibrator is for the Binance proxy; it does not join the six exact Takerflow columns or compose the frozen memory/ESN variants and uniform seed ensembles. The existing A/B/C/D features instead depend on a full-development-selected 64-feature set, 0.59/0.1025 target weights, and later model settings. Reusing these choices would leak the outer period. The newly frozen procedure also requires official daily Platt from earlier-only OOF forecasts for the refit model, and no such per-fold A/B/C/D OOF stream or runner is present. A single missing chronology guard is fixed; the remaining work is a concrete multi-stage implementation, not a data-provenance mystery.

The source inputs are present: three exact development raw prefixes, three causal flow-feature prefixes, the E0038 clean-market allowlist and saved E0038 execution/source snapshots. No holdout metric/output was opened for E0039. One earlier repository audit turn did accidentally materialize the mixed confirmation/holdout `confirmation_market_matches.parquet` while checking its interface; only its row count/schema and the first confirmation row were displayed, and no result or choice was derived. The canonical project history also contains prior E0030 holdout metrics and E0031 mixed-manifest counts. This is disclosed exposure, not new evidence; no holdout rows/metrics were used in this reconstruction.

## Offline runtime benchmark status

The deterministic sample is [300 timestamp-only windows](btc_chronology_runtime_sample_20261027.csv), 100 per month, including the first/last available row in each month and nearest available E0038 rows around the fold-update boundaries. Its SHA-256 is `a3f62ce585f17aa125e69c6494ab139425cfcfc2580007217914c98d8341b7b0`. Selection did not inspect PnL. The environment at freeze had an i7-13650HX (14 cores/20 logical processors), 64 GiB RAM, RTX 4060 Laptop GPU, about 151 GB free on D:, and no heavy competing training job. Any completed benchmark must use one worker, bounded native threads, a monotonic clock, report count/p50/p95/p99/max/%>1s, and distinguish startup/warmup from one-decision updates.

The **full** hot path is not measured. The exact selected BASE70 64-feature set includes generated indicator, VP/RP and other source features plus six Z/V flow values. The research path builds these in batch; the current runtime path does not expose an offline per-decision update that produces this exact ordered feature vector and joins it with streaming memory/ESN states. A ready row from the full-development feature cache would hide feature construction, so it is excluded from a full-path claim. Without the adapter, timing only the LightGBM files, a proxy vector, or a partial base-14 memory/ESN update would not meet the frozen requirement. No latency percentiles or one-second conclusion are claimed.

## Validation and status by axis

- Added `chronological_fit_mask` and used it at both fit-mask construction sites. Focused test run: **51/51 PASS** (`test_btc_preopen_contract`, development boundary, ESN experiment contract, candidate checkpoint/restart, and resolution portfolio modules).
- Existing regressions already cover future-row invariance for baseline causal features, memory/ESN gap resets and restart parity, model feature schema/identity checks, and resolution ledger accounting. The new tests add exact exclusive-cutoff and appended-future-label mask cases.
- No new model fit, OOF calibration, outer prediction, accounting replay, or benchmark measurement was run. Full `python -m unittest discover -s tests` passed **711/711** in **41.334 seconds** (exit 0); this validates code mechanics, not fold reproduction or runtime readiness.
- **T0009/source coverage:** BLOCKED overall; E0038 is limited to its 5,604 clean markets, 53,660 candidate calendar rows lack Gamma IDs, the 24 local PMXT hours remain deferred under T0010, and actual prediction-ready timestamps remain unknown.
- **T0009/chronology:** PARTIAL; fold boundaries/data prefixes/mask are frozen and tested, but A/B/C/D fold fitting and official OOF calibration are blocked before fit.
- **T0009/runtime:** BLOCKED for the complete offline hot path; selected generated feature update is missing from an offline streaming adapter. No live/shadow timing is inferred.
- **T0010:** DEFERRED; no acquisition or gap-count adjustment.
- **E0037:** historical FAILED source gate retained; it is not represented as passed by this chronology work.

## Next action

Implement and test one identity-checked offline runner that consumes the frozen E0039 fold prefixes, refits the full bounded base-generator/feature-selection/weight/model stages, composes BASE70+Z/V with the frozen earlier memory/ESN transforms, emits causal earlier-only official OOF predictions/calibration, and exposes the same ordered features through a per-decision offline update. Then issue and push an E0039 protocol amendment with runner/code hashes before any fit. Start F1 only after that guard passes.
