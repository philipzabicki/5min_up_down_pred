# Historical BTC Polymarket contract

## Timing, established before implementation

`Opened` is the opening time of a CLOSED Binance index one-minute candle (naive artifacts mean UTC). Its OHLC and features are available no earlier than `Opened + 1 minute`. Live `_predict_next_bucket` consumes that candle, prepares features, predicts, then evaluates the quotes. Only minute % 5 == 4 rows are live decision rows; market start is `Opened + 1 minute`, end is start + 5 minutes. The proxy compares Close at Opened with Close at Opened + 5 minutes (ties UP). Target weights emphasize these decision rows; other minutes are NOT extra live entries.

Historical wall-clock receive/inference times are unavailable. Scenarios 0, 1, 2 seconds are explicit lower-bound / latency assumptions, not measured live availability. `decision_available_at = Opened + 1 minute + scenario latency`. Select the first recorded quote AT OR AFTER this instant, within an explicit delay limit and strictly before expiry. Never nearest, interpolation, or forward fill. Quote delay is measured from decision_available_at. Kacho samples a cached book at 1 Hz: the original exchange book age and actual fill cannot be proven. Age is unknown, never reported as fresh.

## Data layers

Raw immutable revision-pinned Hugging Face files and official Gamma responses reside under data/raw/polymarket. Manifests contain SHA256, size, revision, download time and inspected coverage/counts. Canonical market/quote Parquet and decision datasets reside under data/datasets/polymarket/BTC, reports under data/analysis/polymarket/BTC. Providers remain separate.

Kacho outcome is a final-bid inference: retain it ONLY for audit, never settlement truth. Official resolved metadata supplies settlement, token order, boundaries and historical per-market feeSchedule. Gamma startDate is listing time; use eventStartTime / events.startTime and slug epoch for trading-window start. Unknown or mismatched identities/boundaries are excluded. Official response cache includes fetch time; unresolved cached markets require explicit cache removal to refresh.

Kacho records ASK top size and BID aggregate depth only. Top-size-constrained ask execution is possible; deeper ASK VWAP is unavailable. Aggregate bid depth cannot finance BUY execution. Counterfactual buckets are unavailable when ask size is insufficient. Fees reuse utils.polymarket and net shares follow live build_trade_intent: (gross stake - fee) / ask. Stakes include fees. Settlement timing is retained; backtests lock stake until official resolution and allow one position per market.

## Prediction provenance and evaluation

Legacy OOF artifacts contain timestamps but no row-level fold evidence and select early stopping on the evaluated fold. They are audited for coverage but rejected for policy evaluation. A dedicated deterministic CPU LightGBM historical run uses ten predeclared existing causal candle/volatility/basis features (no fitted indicators or retrospectively selected feature list), fixed documented model parameters, past-only train/validation, a label embargo, and future prediction blocks. Each prediction exports source model, fold, training/validation label availability cutoff and feature/parameter hashes. No final fitted predictions. Existing feature definitions and target remain unchanged.

The deployed model feature configurations were selected after this historical window. Their historical selection independence cannot be proven, so legacy probabilities and retrospectively fitted indicators are excluded. The reported probability model is an explicit fixed LightGBM benchmark on existing causal features, not a reconstruction of the deployed model. Current live POLICY code is reused on these OOS probabilities. This distinction is retained in reports; no claim of prospective strategy performance is made.

Probability calibration uses official outcome and only settled earlier markets. Policy choice uses a separate later historical validation block; evaluation is the untouched next chronological block. Final block never selects a variant. Data quality gate is applied before policy search. Reports include all predefined baselines, calibration diagnostics, bank paths, execution rejections and bucket attribution.

Architecture: long-history predictive modeling -> shorter-history Polymarket settlement/execution calibration -> economic entry and sizing policy. Polymarket prices are never probability-model features.
