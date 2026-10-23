# Frozen BTC candidate target and economic attribution — 2026-10-23

## Findings

1. **The target disagreement is mechanically real, but its price-source cause is unproven.** All 133,340 saved Binance labels reproduce exactly from the frozen COIN-M candles, and all 79,678 matched Gamma records pass the exact market-start, series, slug-epoch, event-start, condition-ID, and token-orientation checks. Among 79,650 resolved official labels, Binance differs on 1,980/51,430 confirmation markets (3.85%) and 2,213/28,220 holdout markets (7.84%). No alignment or implementation error was found. Gamma stores the final one-hot outcome, not the Chainlink prices and timestamps used to settle it. The discrepancies are consistent with the contracts using different price constructions, but none can be confirmed as source-caused with the available price evidence.
2. **A’s loss versus B is a decision-selection and realized-outcome difference, not a side reversal.** On the 1,143 common priceable decisions, A made 424 trades and lost $108.80 independently; B made 396 and earned $28.01. They took the same side on all 342 shared trades; there were no opposite-side decisions. A-only trades lost $57.47, while B-only trades earned $79.35. B’s realized advantage is therefore $136.82 on this sparse sample. The T−60 expected net PnL for both exclusive groups was positive, so the realized gap is not evidence of a robust edge for B.
3. **Adding ESN to memory changed 86 decisions, all between trade and no-trade.** D made 35 trades B skipped (+$16.00 realized) and skipped 51 B trades that earned +$21.35, for D−B = −$5.35. There were no side changes. The difference is spread across the changed rows: the five largest absolute contributions account for 6.20% of their total absolute contribution. This is a small, retrospective diagnostic, not evidence of stable ESN harm.
4. **Next experiment: reconstruct the official target from the actual Chainlink reports over the frozen official-label windows.** This directly extends the earlier 9,407-window target audit with the A/B/C/D prediction windows and the observed July–September mismatch increase. Do not refit models or revise policy.

## Frozen scope and source boundaries

This analysis uses only the saved v3 A/B/C/D predictions, Gamma labels, Binance OHLCV, economic decision ledger, and self-financing ledger. Procedures remain A = BASE70, B = BASE70 + simple memory PCA8, C = BASE70 + ESN PCA8, and D = BASE70 + memory + ESN PCA8. Both evaluated periods were previously exposed. No training, Optuna, replay, split change, policy change, or trading was run. The analysis input hashes are frozen in `btc_candidate_attribution_20261023_protocol.json`.

The older target study compared 9,407 Kacho windows and found 405 disagreements (4.31%); its Chainlink cache covered none of those windows. Its earlier OOF target/timing and smaller economic-window coverage differ from this frozen A/B/C/D comparison, so the rates are not a like-for-like time trend. New here are the exact reconstruction of the v3 Binance target across all saved candidate rows, the wider official-label sample in both periods, row-level mismatch attribution, and the link to the existing 1,143-row decision/account ledgers. No Chainlink price history was added.

## Target contracts

| Item | Frozen Binance proxy label | Official Polymarket label |
|---|---|---|
| Instrument / source | BTCUSD COIN-M index, Binance 1-minute OHLCV | BTC Up/Down 5-minute market, Gamma series `10684`; market slug `btc-updown-5m-{epoch}` |
| Market identity | Prediction row maps `Opened + 2m` to market start | Exact UTC start, condition ID, UP/DOWN token IDs, series, and slug epoch recorded per market in the observation ledger |
| Compared values | `Open[Opened+2m]` versus `Close[Opened+6m]` | Chainlink BTC/USD 60-second TWAP stream over the title’s five-minute range versus the price at the beginning of that range |
| Window | Five 1-minute candles `[T,T+1m)` through `[T+4m,T+5m)`; `T = Opened+2m`; end boundary `Opened+7m` | Exact market start `T` through `T+5m` |
| Rule / tie | UP when final close `>=` first open; equality is UP | UP when the Chainlink range TWAP is `>=` its opening reference; equality is UP |
| Label availability | `Opened+7m`, after all five candles close | Gamma’s saved final, one-hot outcome; the underlying Chainlink price reports are not in Gamma |
| Saved evidence | Original proxy target plus exact OHLC reconstruction | UP/DOWN `outcomes`, token mapping, one-hot `outcome_prices_current`, resolution time and closed status |

The market page names the Chainlink BTC/USD TWAP 60s stream and describes the UP rule as a range TWAP greater than or equal to the opening price ([Polymarket market rules](https://polymarket.com/event/btc-updown-5m-1788862200), [Chainlink stream](https://data.chain.link/streams/btc-usd-twap-60s-streams)). The Gamma one-hot value is settlement evidence, **not** a record of either price used in that comparison.

### Mechanical join and label checks

| Check | Result |
|---|---:|
| Saved candidate prediction rows | 133,340 |
| Exact `Opened+2m` market start and `Opened+1m` nominal T−60 mapping | 133,340 / 133,340 |
| Five Binance candles present; target reconstructed equal to saved label | 133,340 / 133,340 for each check |
| Exact Gamma market-start matches | 79,678 |
| Matched Gamma rows with series 10684, aligned slug epoch and event start | 79,678 / 79,678 |
| Matched condition IDs and distinct UP/DOWN token IDs | All passed |
| Matched rows with valid final one-hot official label | 79,650 |
| Matched rows excluded from official scoring | 28: 14 not closed/no resolution time, 13 ambiguous outcomes/prices, 1 non-binary prices |
| Exact Binance price ties among official rows | 5; one disagrees with Gamma |

The error classification is therefore: **implementation/matching errors: 0; disagreements confirmed by direct oracle-price evidence: 0; mechanically valid disagreements whose price-level cause remains unresolved: 4,193.** The common target contracts differ in source and price construction, but without the corresponding Chainlink reports it would overstate the evidence to label each mismatch as caused by that difference. No historical rule-version record is archived, so the rate increase cannot be attributed to a rule change or data-quality change either.

The Chainlink REST API supports timestamped and sequential report requests, but its report endpoints require authenticated requests. No matching report payloads or API credentials are part of the frozen inputs, and the older local 1,426-report cache ends outside these market windows. I did not substitute another exchange or invent opening/settlement prices. See [Chainlink Data Streams API authentication and report endpoints](https://docs.chain.link/data-streams/reference/data-streams-api/interface-api).

## Disagreement patterns

The denominators are resolved official Gamma labels only. A Binance-up/Polymarket-down count and its converse are nearly balanced in both periods, which is inconsistent with a simple UP/DOWN orientation inversion.

| Period | n | Disagreements | Rate | Binance UP → PM DOWN | Binance DOWN → PM UP |
|---|---:|---:|---:|---:|---:|
| Confirmation | 51,430 | 1,980 | 3.85% | 973 | 1,007 |
| Holdout | 28,220 | 2,213 | 7.84% | 1,054 | 1,159 |

| Month (UTC) | n | Disagreements | Rate | Binance UP → PM DOWN | Binance DOWN → PM UP |
|---|---:|---:|---:|---:|---:|
| 2025-12 (partial coverage) | 3,962 | 169 | 4.27% | 92 | 77 |
| 2026-01 | 7,469 | 334 | 4.47% | 165 | 169 |
| 2026-02 | 4,887 | 233 | 4.77% | 113 | 120 |
| 2026-03 | 8,928 | 362 | 4.05% | 188 | 174 |
| 2026-04 | 8,634 | 364 | 4.22% | 175 | 189 |
| 2026-05 | 8,917 | 355 | 3.98% | 176 | 179 |
| 2026-06 | 8,633 | 163 | 1.89% | 77 | 86 |
| 2026-07 | 8,927 | 234 | 2.62% | 104 | 130 |
| 2026-08 | 8,927 | 794 | 8.89% | 383 | 411 |
| 2026-09 | 8,639 | 1,002 | 11.60% | 510 | 492 |
| 2026-10 (partial) | 1,727 | 183 | 10.60% | 93 | 90 |

Mismatch incidence is concentrated in the smallest observed Binance moves. The bins below were fixed in the protocol before analysis; they are descriptive, not candidate-selection cutoffs.

| Absolute Binance return | Confirmation: n / disagreements / rate | Holdout: n / disagreements / rate |
|---|---:|---:|
| Exactly 0 bp | 2 / 1 / 50.00% | 3 / 0 / 0.00% |
| (0, 1] bp | 6,322 / 1,668 / 26.38% | 4,260 / 1,171 / 27.49% |
| (1, 5] bp | 17,938 / 306 / 1.71% | 10,918 / 893 / 8.18% |
| (5, 10] bp | 12,424 / 5 / 0.04% | 6,844 / 126 / 1.84% |
| (10, 25] bp | 11,622 / 0 / 0.00% | 5,267 / 22 / 0.42% |
| (25, 50] bp | 2,635 / 0 / 0.00% | 816 / 1 / 0.12% |
| > 50 bp | 487 / 0 / 0.00% | 112 / 0 / 0.00% |

For all official labels, the absolute Binance-return mean / median / p90 were 8.57 / 5.44 / 19.51 bp in confirmation and 6.90 / 4.46 / 15.65 bp in holdout. Within the disagreement rows, median / p90 were 0.40 / 1.24 bp and 0.90 / 4.05 bp, respectively. At most 5 bp from flat: 1,975/1,980 confirmation mismatches (99.75%) and 2,064/2,213 holdout mismatches (93.27%). This supports describing the mismatches as near-flat proxy windows; it does not identify which price or sampling point caused an individual settlement.

## Quality metrics and target attribution

All entries use the same official-label rows and saved probabilities within each period. Delta convention: B−A = loss(A) − loss(B), D−B = loss(B) − loss(D); positive favors the right-hand model. Values are point estimates, with no new uncertainty test.

| Period | Target | n | B−A log loss | B−A Brier | D−B log loss | D−B Brier |
|---|---|---:|---:|---:|---:|---:|
| Confirmation | Binance | 51,430 | +0.000036 | +0.000018 | −0.000050 | −0.000025 |
| Confirmation | Polymarket | 51,430 | +0.000056 | +0.000028 | −0.000044 | −0.000022 |
| Holdout | Binance | 28,220 | +0.000058 | +0.000028 | +0.000033 | +0.000016 |
| Holdout | Polymarket | 28,220 | +0.000018 | +0.000009 | +0.000076 | +0.000038 |

Switching between Binance and Polymarket labels does **not** reverse either comparison’s sign within confirmation or within holdout. D−B does reverse sign between the two periods under both target definitions. The existing v3 Bonferroni-adjusted intervals for these primary comparisons cross zero; the subgroup attribution below is explanatory arithmetic, not new evidence of significance.

Weighted contributions to each full official-row delta, split by whether the two labels agree:

| Period | Comparison | Agreement contribution: LL / Brier | Disagreement contribution: LL / Brier | Total: LL / Brier |
|---|---|---:|---:|---:|
| Confirmation | B−A | +0.000039 / +0.000019 | +0.000017 / +0.000008 | +0.000056 / +0.000028 |
| Confirmation | D−B | −0.000051 / −0.000026 | +0.000007 / +0.000004 | −0.000044 / −0.000022 |
| Holdout | B−A | +0.000026 / +0.000013 | −0.000008 / −0.000004 | +0.000018 / +0.000009 |
| Holdout | D−B | +0.000045 / +0.000023 | +0.000031 / +0.000015 | +0.000076 / +0.000038 |

Thus confirmation’s negative D−B is driven mainly by D’s loss on **agreement** rows, not the disagreements. In holdout, both groups favor D. For B−A, agreement rows favor B in both periods; disagreement rows strengthen its point estimate in confirmation and reduce it in holdout. The modest target-dependent changes in magnitude are not a within-period rank reversal.

## Economic sample and A versus B

The only priceable economic sample is 1,143 confirmation decisions, from 2026-04-15 17:35 UTC through 2026-05-18 09:00 UTC. It is 2.22% of the 51,430 resolved confirmation labels and contains no holdout markets. This selection is tied to saved T−60/T−59 book availability, not a new filter. The T−60 prediction availability time itself is not archived.

| Sample | n | Date range (UTC) | Target mismatch | Median / p90 absolute Binance move | Share ≤ 5 bp |
|---|---:|---|---:|---:|---:|
| All official confirmation labels | 51,430 | 2025-12-18 to 2026-06-30 | 3.85% | 5.44 / 19.51 bp | 47.17% |
| Outside sparse economic subset | 50,287 | 2025-12-18 to 2026-06-30 | 3.84% | 5.47 / 19.64 bp | 47.01% |
| Sparse economic subset | 1,143 | 2026-04-15 to 2026-05-18 | 4.46% | 4.50 / 14.68 bp | 54.33% |
| All official holdout labels | 28,220 | 2026-07-01 to 2026-10-06 | 7.84% | 4.46 / 15.65 bp | 53.80% |

| Decision group | Markets | A/B actions | Gross notional | Cash fees / V1 share-fee equivalent | Mean T−59 VWAP | T−60 expected net PnL A / B | Realized PnL A / B | B−A contribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Both trade same side | 342 | 342 / 342 | $1,710 each | $38.15 / $22.17 each | 0.50380 each | +$54.50 / +$49.04 | −$51.33 / −$51.33 | $0.00 |
| A only | 82 | 82 / 0 | $410 (A) | $9.94 / $4.45 (A) | 0.50622 (A) | +$3.85 / $0.00 | −$57.47 / $0.00 | +$57.47 |
| B only | 54 | 0 / 54 | $270 (B) | $6.37 / $3.21 (B) | 0.50130 (B) | $0.00 / +$2.06 | $0.00 / +$79.35 | +$79.35 |
| Opposite sides | 0 | 0 / 0 | $0 | $0 / $0 | — | $0 / $0 | $0 / $0 | $0.00 |
| Both no trade | 665 | 0 / 0 | $0 | $0 / $0 | — | $0 / $0 | $0 / $0 | $0.00 |

Independent trades total to A: 424, $2,120 gross, $48.09 cash fees plus $26.62 V1 share-fee equivalent, +$58.35 expected net PnL and −$108.80 realized PnL. B: 396, $1,980 gross, $44.52 cash fees plus $25.38 share-fee equivalent, +$51.10 expected and +$28.01 realized. The exact +$136.82 B−A realized difference is the sum of the exclusive-group contributions above; the shared trades cancel and no trade changed sides.

B made 28 fewer trades. Outside `[0.45, 0.55]`, the saved UP probability occurred on 4.81% of A rows and 3.50% of B rows in this sample, so B is somewhat less extreme by this descriptive measure. But both exclusive groups had positive aggregate T−60 expected net PnL; A-only subsequently lost $57.47 while B-only won $79.35. The ledger therefore supports selection and realized settlement outcomes as the accounting explanation; it does not establish that B’s reduced tail probability or lower participation will generalize.

### Capital and fee reconciliation

The account simulation starts each candidate at $100, locks a $5 gross cost basis until the saved release time, and has no borrowing. Cash is debited by `$5 + V2 cash fee`; V1 fees reduce payout shares and are reported separately as a dollar equivalent, not charged twice. V3’s existing 7.2%/7.0% fee-regime estimate and V1/V2 handling are retained.

| Model | Signals / independent trades | Accepted | Cash-rejected | Independent PnL | Self-financing PnL | Ending cash/equity | Accepted gross | Cash debit | Payouts released |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 424 | 285 | 139 | −$108.80 | −$95.42 | $4.58 | $1,425 | $1,448.86 | $1,353.44 |
| B | 396 | 396 | 0 | +$28.01 | +$28.01 | $128.01 | $1,980 | $2,024.52 | $2,052.53 |
| D | 380 | 380 | 0 | +$22.66 | +$22.66 | $122.66 | $1,900 | $1,943.16 | $1,965.82 |

The identity `initial cash − cash debits + released payouts = ending free cash` reconciles to within `8e−12` USD, with zero open positions at sample end. A’s 139 rejected signals have −$13.38 counterfactual PnL, excluded from its actual funded PnL. The first rejection was decision time 2026-05-09 05:44 UTC (market start 05:45), with $4.575836 free cash, no locked position, and a $5.175 debit required. Rejections continued through market start 2026-05-18 08:15; there was no credit, borrowing, or liquidation. The balance reached $4.58 because the account absorbed realized losses and then could not fund another fixed order.

B and D have identical independent and funded PnL because every selected $5 trade was accepted in each account; neither had a cash rejection. Settlement release is idealized as `max(Gamma resolution time, market end) + 60 seconds`; actual redemption latency is not archived.

## What ESN changed: D versus B

| Decision group | Markets | B / D trades | Gross notional | Cash fees / V1 share-fee equivalent | Mean T−59 VWAP | Expected net PnL B / D | Realized PnL B / D | D−B realized contribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Both trade same side | 345 | 345 / 345 | $1,725 each | $38.26 / $22.71 each | 0.50313 each | +$49.64 / +$46.40 | +$6.66 / +$6.66 | $0.00 |
| B only | 51 | 51 / 0 | $255 (B) | $6.26 / $2.67 (B) | 0.50569 (B) | +$1.46 / $0.00 | +$21.35 / $0.00 | −$21.35 |
| D only | 35 | 0 / 35 | $175 (D) | $4.90 / $1.25 (D) | 0.50486 (D) | $0.00 / +$0.66 | $0.00 / +$16.00 | +$16.00 |
| Both no trade | 712 | 0 / 0 | $0 | $0 / $0 | — | $0 / $0 | $0 / $0 | $0.00 |
| Opposite sides | 0 | 0 / 0 | $0 | $0 / $0 | — | $0 / $0 | $0 / $0 | $0.00 |

D’s selected trade count fell from 396 to 380. Mean absolute `p(D)−p(B)` was 0.00365 (median 0.00297; p90 0.00762) across the economic sample; the mean absolute probability change was 0.00648 on B-only rows and 0.00514 on D-only rows. In the frozen positive-EV policy, every B-only row was above zero for B and at/below zero for D; every D-only row showed the reverse. Median distances to the zero-EV boundary were $0.0162 (B on B-only), $0.0315 (D on those rows), $0.0262 (B on D-only), and $0.0150 (D on D-only). These are descriptions of threshold crossings, not a search for a better threshold.

The net −$5.35 is not concentrated in a few settlements: 42 changed rows contributed positively and 44 negatively to D−B, with the five largest absolute contributions making up 6.20% of total absolute changed-row contribution. D’s total T−60 expected net PnL was $47.07 versus B’s $51.10; realized PnL was +$22.66 versus +$28.01. This sample does not establish persistent deterioration from ESN.

## Concentration, drawdown, and limitations

| Model | Independent trades | Largest win / loss | Top 5 / top 10 absolute trade-PnL share | Best / worst daily PnL | Cost-basis max drawdown |
|---|---:|---:|---:|---:|---:|
| A | 424 | +$5.23 / −$5.24 | 1.26% / 2.52% | May 10 +$27.88 / May 7 −$28.17 | $116.16 (96.21%) |
| B | 396 | +$5.91 / −$5.24 | 1.39% / 2.74% | May 10 +$32.86 / May 7 −$32.95 | $100.07 (55.99%) |
| D | 380 | +$5.91 / −$5.24 | 1.45% / 2.85% | May 15 +$23.20 / May 3 −$27.67 | $101.60 (72.53%) |

Drawdown equity is explicitly `free cash + gross cost basis of unsettled positions`, sampled at decision and release events; it is **not** mark-to-market drawdown. Peak→trough times were A: 2026-04-19 16:26:22 → 2026-05-09 05:16:15 UTC; B: 2026-05-06 17:46:16 → 2026-05-13 05:09:00; D: 2026-05-01 05:01:54 → 2026-05-13 05:09:00.

The economic ledger is sparse and confirmation-only: 1,143 markets, no holdout book data, with assumed T−60 prediction availability because actual prediction timestamps were not retained. The account release time is idealized. Both periods and the evaluated candidate procedures are historically exposed. PnL concentration is modest by individual trade but daily outcomes remain volatile; positive B/D PnL does not establish live readiness.

## Recommended next experiment

**Hypothesis:** reconstructing the official label from the exact Chainlink TWAP reports will explain a material share of the near-flat Binance/Gamma mismatches and determine whether the July–September mismatch increase persists under direct source-price comparison.

- **Test:** obtain the correct Chainlink stream/feed IDs and report payloads covering the opening reference and each frozen five-minute market window; retain each report’s valid-from and observation timestamps. Reconstruct the official rule, compare it with all 79,650 saved Gamma outcomes, and report per-window agreement, mismatch moves, and reason codes. Rescore the same saved A/B/C/D probabilities on those reconstructed labels as a diagnostic; do not refit, recalibrate, change thresholds, or replay trades.
- **Data:** authenticated historical Data Streams reports plus Gamma’s condition/token IDs, market slug and exact UTC window. Keep reports and provenance in a separate immutable supplement.
- **Periods:** the same confirmation `[2025-07-01, 2026-07-01)` and holdout `[2026-07-01, 2026-10-07)` market-start bounds. Treat both as exposed historical intervals, not new tests.
- **Cost:** little local computation; first verify historical report coverage and obtain authorized API access. Chainlink documents timestamped and paged historical report endpoints, while requiring authentication for report requests; public documentation does not establish data retention for all requested dates or quote a price. Do not provision paid access until coverage and cost are known.
- **Interpretation limit:** exact label reconstruction can separate proxy disagreement from sampling/settlement details, but it does not supply missing historical prediction-availability timestamps, expand the 1,143-market execution sample, or establish live performance.

## Reproducibility and artifacts

Runner: `run_btc_candidate_attribution_20261023.py`. Frozen input manifest: `reports/btc_preopen/btc_candidate_attribution_20261023_protocol.json`. Analysis outputs are kept locally under `data/analysis/polymarket/BTC/btc_candidate_attribution_20261023/` and are not part of the commit. The output set includes:

- `target_observation_ledger.parquet`, `target_disagreements.csv`
- `quality_metrics_by_target_and_group.csv`, `target_metric_delta_attribution.csv`, `target_metric_decomposition_checks.csv`, `probability_change_summary.csv`
- `monthly_target_agreement.csv`, `binance_return_move_bins.csv`, `binance_return_quantiles.json`
- `economic_attribution_ledger.parquet`, `pair_decision_attribution.csv`, `changed_decisions_B_D.csv`, `changed_decision_threshold_summary.csv`
- `portfolio_accounting_reconciliation.csv`, `portfolio_cash_path.csv`, `portfolio_event_ledger.csv`
- `pnl_concentration.csv`, `daily_pnl_by_scenario.csv`, `economic_sample_vs_full_official.csv`, `economic_probability_distribution.csv`
- `analysis_summary.json`, `validation.json`

Validation: Python syntax compilation passed; runner reconciliation assertions passed for all 133,340 target windows, all 79,678 Gamma matches, full-sample loss decomposition, fixed-$5 fee/cash identities, and portfolio cash/equity identities. No training or replay ran. End-to-end runtime was 7.07 seconds. Sampled peak process working set was 613,040,128 bytes (584.7 MiB), sampled every 0.1 seconds.
