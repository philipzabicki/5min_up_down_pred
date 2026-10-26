# BTC Research Contract

Status: binding research decisions, recorded 2026-10-10 (Europe/Warsaw). All computation timestamps and interval boundaries below are UTC. This contract governs BTC prediction training, learned features, calibration, policy selection, historical replay, and the evaluated holdout. It does not authorize live orders.

## Data periods and validation

- Development market starts are strictly earlier than `2026-07-01T00:00:00Z`. This includes June 30 in full. Holdout market starts are at or after that instant and before the frozen exclusive data cutoff recorded in the run manifest.
- At this contract's initial freeze, the verified local Polymarket archive declares exclusive cutoff `2026-10-07T00:00:00Z`: 50,184 Gamma-confirmed market starts over a 5-minute calendar from `2025-12-18T04:25:00Z` through `2026-10-06T23:55:00Z`. The run must record source coverage, gaps, exclusions, and actual cutoff; later local data may extend it only if included in a newly frozen input manifest before evaluation.
- BTC price training rows may use the entire available price history whose target labels are available by the applicable fit boundary. Do not impose a three-year maximum. Whole-history training is the baseline; shorter windows or recency weighting are development-only comparisons.
- The BTC model target is the Binance COIN-M BTCUSD index proxy: `Close[Opened+6m] >= Open[Opened+2m]`, over candles beginning at `Opened+2m` and ending at the exclusive `Opened+7m` boundary. The label is available at `Opened+7m`, requires all five one-minute candles, and resolves ties UP. `Opened+1m` is the nominal decision and the market starts at `Opened+2m`; this proxy is not the official Chainlink settlement label. Build these price labels without requiring a Polymarket quote or market record; apply quote availability only when forming the economic evaluation sample.
- Optimize indicator-generator settings, volume/reaction profile settings, feature selection, model parameters, calibration, and trading policy only on development. Policy optimization additionally requires development execution quotes and resolved outcomes. Do not use holdout PnL or labels to select or revise any choice.
- Use chronological walk-forward validation. In every fold, all learned components—including generator choices, feature selection, model fitting, calibration, and policy fitting—must use only the earlier prefix available at that fold. Deterministic causal features may warm up from earlier candles. Do not fit a generator or select features on all development and then present earlier model folds as independent validation of that full pipeline.
- A training label is eligible only when its recorded availability time is no later than the applicable fit/decision boundary. Purge rows whose labels are not yet available; do not purge earlier feature history needed for warm-up. Every fold manifest records the fit interval, validation interval, latest eligible label-availability time, feature warm-up interval, and prediction provenance.
- Fit calibration on out-of-sample predictions. Evaluate policy choices using causally calibrated walk-forward predictions and outcomes available before each policy-selection boundary. Development fold predictions may enter the final development fit after selection is complete.
- Freeze configuration, artifacts, code/data identities, and the exclusive holdout cutoff before holdout evaluation. Historical holdout results have been inspected previously; this evaluation is not described as unseen. Record that exposure once in the experiment report, and do not use holdout PnL to tune the strategy.

## Decision and simulated execution times

- Make the nominal decision at `T−60s` using only data available then. The combined budget for Binance websocket receipt, feature construction, prediction, order submission, and execution is one second.
- Simulate execution at `T−59s`. Do not add another second, the former 45-second budget, or a further 5-second execution delay. Earlier live observations of about 500 ms per cycle are an existing user assumption, not a measurement from this experiment.
- Replay book state causally through `T−59s`. Choose side, eligibility, and stake from information available at `T−60s`. Use the `T−59s` book only to determine whether/how the already-decided order could execute; it cannot change the earlier decision. If a limit or edge condition is used, record separately its decision-time test and its post-decision execution feasibility test.

## Quote continuity and age

- Do not reject an entry or exit solely because the last ask/bid price change is more than one second old. In a continuous event replay, a level remains active until changed or removed; unchanged price is not evidence of a data gap.
- Detect gaps or inconsistencies from source-coverage/sequence evidence. For sampled sources, carry forward only the latest causally available observation within known source coverage and within the same market. Record that it is a proxy and its age. A later update must not be required to validate an earlier decision.
- Never carry a quote across a documented source gap or between markets. Missing ask/bid, invalid prices, and a genuinely inconsistent book remain separate exclusion reasons. Do not use future observations to qualify an entry or exit.
- Apply the same information and continuity rules to ask-side buys and bid-side sells. Missing historical hypothetical-order ACKs do not block a counterfactual backtest.

## Taker execution and depth

- On a verified complete replayed book, price a taker order by walking the relevant levels and computing size-specific VWAP. Known insufficient depth is an execution constraint; match the existing live order type and its full/partial-fill semantics.
- On top-of-book or otherwise incomplete depth, price at the latest available ask for a buy or bid for a sell as an explicit proxy assumption. Unknown depth is not zero liquidity and alone does not make the order unpriceable.
- Compare requested gross BUY amount in USDC with Gamma `orderMinSize` when that market metadata is available; otherwise use the accepted `$5` USDC fallback. Historical Gamma minimums are not available, so report this limitation. A partial fill may end below the minimum after the requested order passed the check.
- Never label a top-of-book proxy as full-book VWAP. Every ledger entry records `execution_source`, `depth_known`, and `execution_assumption`, plus requested/filled size and the specific no-fill/partial-fill reason when applicable.
- Report full-book versus proxy trade shares and outcomes. Run adverse-price/slippage sensitivity on the completed path and estimate the incremental cost that removes the observed advantage. Apply the same information rules on exit bids.
- Historical recorded liquidity is not a live-liquidity forecast; retain that limitation in the report without using it to omit the requested historical analysis.

## Fees and portfolio accounting

- Preserve the accepted fee estimate: rate `0.072` before `2026-05-06T00:00:00Z`, rate `0.07` from that instant onward, with the existing V1/V2 regime split and established price-dependent fee formula, rounding, share/cash debit, and no double charge. Do not reopen fee-change-date research.
- Keep rebates as a separately reported sensitivity; they do not participate in primary strategy selection.
- Each evaluated period is one continuous portfolio initialized with `$100`, with no credit, deposits, or withdrawals. Under the current sale-based decision (D0016), lock the buy cost until an actual simulated sale executes. Do not use official resolution or a binary payout as an exit or a fallback. If a position cannot be reliably closed, retain its shares, paid fees, and locked cost in the ledger; mark the position open and the economic result incomplete. Start a separate `$100` holdout portfolio at `2026-07-01T00:00:00Z`; carry causal feature/model state from earlier history, but never carry development portfolio equity into holdout.
- Required sizing controls are (1) fixed `$5` gross stake and (2) `5%` of free cash at the decision, with no additional dollar cap. The percentage stake remains locked until an actual sale; any unsold amount remains locked. Call it “5% of free cash,” not 5% of total equity. Any additional learned policy is selected on development only.
- Preserve share/cash fee treatment, price-dependent fees, share precision, rounding, and one-time debiting. Audit cash lock/release, no debt, sale proceeds and fees, open positions, and the continuous cost-basis equity path.

## Artifacts and reporting

- Cache identity includes data/source hashes, schema, code version, feature/generator configuration, and fold boundaries. Reuse only complete, identity-matching stages; checkpoint writes are atomic and resumable. A model change requires new predictions, but does not by itself require replaying an unchanged book archive.
- Freeze an `evaluated_bundle` for the holdout and keep it separate from any later `deployment_candidate`. Prepare a resumable final-refit procedure with frozen settings for all available history, including the historical holdout; do not run that refit as part of holdout evaluation or replace the evaluated bundle with its output. Calibration and learned policy continue to use causal walk-forward predictions. Refit is not another independent backtest. Do not activate a model or submit orders.
- The report records exact periods for generators, model, calibrator, policy, and holdout; chosen settings and development walk-forward basis; market and trade counts with exclusion reasons; entries recovered by quote/depth corrections; holdout equity, PnL, drawdown, turnover, fees, trade count, and monthly results; both required controls; full-book/proxy shares; slippage sensitivity; runtime, memory, cache reuse, data gaps, and a clear conclusion.
- Keep progress and the next resumable step in `docs/BTC_RESEARCH_STATUS_20261010.md`, updating it at each checkpoint. Before long runs, report a measured runtime estimate and counters; do not replace the authorized full run with a smoke test or benchmark.
