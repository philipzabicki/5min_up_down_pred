# BTC conditional market-offset execution diagnostic - 2026-10-18

> This is a read-only diagnostic of the frozen experiment in `btc_conditional_market_offset_20261018.md`. The holdout was historically exposed. No training, tuning, new threshold, replay, data collection, or policy change was run.

## Findings

The frozen holdout has 24,736 economic decision rows; 24,735 have official labels released strictly before the exclusive cutoff, 0 release exactly at the cutoff, and one additional row releases after cutoff. Development has 8,675 saved OOF predictions; the 2,119-row April prefix trained the fold models and has no out-of-fold D prediction.
At fixed $5 and T-60, D has 1,625 markets with no priceable side, 23,074 with a priceable side but no positive net EV, 37 with positive EV below 1%, and 0 at or above the 1% hurdle. The 37 below-hurdle IDs exactly match the frozen portfolio decisions. The prior independent-order aggregate combined unpriceable markets with nonpositive EV under `no_positive_net_ev_at_t60`; its 24,699 count is 1,625 + 23,074.
For those 37 positive-EV intents, the 1% rule requires $0.05 net EV on the requested $5 gross amount. Their actual T-60 EV and distance to that hurdle are summarized below. None qualifies; therefore the frozen policy makes no T-59 order and reports zero realized trades for D. All 37 labels are cutoff-safe; a same-intent T-59 shadow is priceable for 35 and has cutoff-safe outcomes for 35 fills, with descriptive shadow PnL $+1.73. These are not executed trades or a new policy.
In the fixed-order waterfall for those 37 cases, median net EV at the normalized p_market reference is $0.0577; at the T-60 best ask it is $0.0079; and after walking saved T-60 depth it is $0.0079. The observed median drop occurs at the ask, while the depth walk adds no further median drop in this sample.
On saved D predictions, development OOF delta log-loss is -0.000272 and delta Brier is -0.000135; holdout deltas are -0.000385 and -0.000195. Development OOF includes 131 T-60 cases at the frozen 1% hurdle (delta log-loss -0.002055, Brier -0.000998); holdout has 0. Thus development quality gains do occur in a small set that clears the fixed execution rule, while holdout has no trade-qualified quality sample. All months, including October, remain in the monthly table.
The order engine has no explicit price limit. It fixes side and stake at T-60, then walks the T-59 asks as a FAK-style marketable BUY, including partial fills. The model's probability correction is small relative to the cost of paying the native ask plus fees for most rows; the exact price waterfall shows which changes account for the net EV.

## Frozen input audit

The saved final D model has 43 trees and reproduces the holdout raw margins and probabilities exactly. Development quality uses 8,675 saved fold-specific OOF predictions; the final full-development model is not expected to recreate those fold-model outputs. All 33,411 saved official labels match the hash-verified market index for outcome, start, resolution (including matching missing resolutions), and release time. The observed label rows hash is `96c9e9b89c23da572c1e54183a8e325b945b8bab0569cfa5659e131b1363edf9`; the prior experiment report hash confirms it was not overwritten.

| Source | SHA-256 | Bytes |
|---|---|---:|
| development_cv_predictions (`data/analysis/polymarket/BTC/btc_conditional_market_offset_20261018/development_cv_predictions.parquet`) | `2167633bc213d12c338d8238d9ae61a0086ba3b02325256832570cc4d3536d26` | 2728042 |
| holdout_predictions (`data/analysis/polymarket/BTC/btc_conditional_market_offset_20261018/holdout_predictions.parquet`) | `a619e7a3db176586c29552310672299c815b112ddca76a7141ca8e12484b77a9` | 7574474 |
| conditional_model (`data/analysis/polymarket/BTC/btc_conditional_market_offset_20261018/conditional_correction_lightgbm.txt`) | `0031f7868d364719d311e595102b3f7a7dadbf6dcef35f8e1100b3a968cca565` | 19485 |
| t59_snapshots (`data/analysis/polymarket/BTC/preopen_research_20261011/book_replay_t60_t59/t59_ask_ladders_with_source_coverage.parquet`) | `abee0fdaf31eaf1455b4343e985bf9d5ee38a84dd8eb7e06adf1e08fc0c380d0` | 26051278 |
| t60_snapshots (`data/analysis/polymarket/BTC/preopen_research_20261011/book_replay_t60_t59/t60_decision_books_with_source_coverage.parquet`) | `38259fadf50c28e455f5d4523610cbac827886e35922076dbba5cb33bd55e0b1` | 25908421 |
| official_market_labels (`data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/pmxt_extended/market_index.parquet`) | `f828da19aa9c1811118abadd3d58a00b51ff2311ceb7492138ccfc1da5dd2973` | 7486947 |
| frozen_protocol (`configs/research/btc_conditional_market_offset_20261018.json`) | `8e10fd35dbe9da9bb2ed3524428120958331e7930b6e24d63f726f0fbd1a5d99` |  |
| frozen_runner (`run_btc_conditional_market_offset_20261018.py`) | `f392ee0e7e8ac33aae3a06ace415b4e136b1b17941cc72ff74f73e14867f891d` |  |
| historical_experiment_report (`reports/btc_preopen/btc_conditional_market_offset_20261018.md`) | `f4d331e3cd43d29744c352c30215d69f6b621138b37146f8a34f6392c787ffe6` |  |
| observed_label_rows (`saved development OOF and holdout prediction artifacts`) | `96c9e9b89c23da572c1e54183a8e325b945b8bab0569cfa5659e131b1363edf9` |  |

## Sample and T-60 funnel

| period | population | markets | quality_label_rows | no_priceable_t60_side | nonpositive_net_ev | positive_ev_below_1pct | at_or_above_1pct | positive_intents_priced_at_t59 | t59_priceable | t59_partial_fills |
|---|---|---|---|---|---|---|---|---|---|---|
| development_oof | 8675 | 8675 | 8675 | 1266 | 6919 | 359 | 131 | 490 | 488 | 0 |
| holdout_historical_diagnostic | 24736 | 24736 | 24735 | 1625 | 23074 | 37 | 0 | 37 | 35 | 0 |

Status is assigned without filtering on the outcome: a side is selected only by the frozen positive-net-EV rule; the 1% criterion is applied afterward. The T-60 native ask reasons are broken out in `t60_unpriceable_side_reasons.csv`. The holdout quality sample excludes one economics row whose official label availability is after cutoff; the economic decision universe retains it, matching the frozen run.

The frozen portfolio decision table reports 24,736 rows: 1,625 `no_priceable_native_ask_side`, 23,074 `no_positive_net_expected_value`, 37 `below_minimum_net_ev_fraction`, and zero accepted orders. The independent fixed-$5 evaluator's `no_positive_net_ev_at_t60=24,699` merges the first two categories because it increments that reason whenever no side is selected. The matching 37 below-1% cases occur in both calculations because the portfolio never spends cash and therefore its 5% free-cash request remains $5.

## EV and execution definitions

| item | rule |
|---|---|
| p_market | At T-60: mid_up=(UP best bid+ask)/2; mid_down=(DOWN best bid+ask)/2; p_market=mid_up/(mid_up+mid_down). For DOWN use 1-p_market. |
| probability change versus EV percent | A probability-point correction is not percent EV on stake; EV also depends on executable shares, ask price, depth, and fees. |
| gross request | $5 gross USDC requested for the fixed-order comparison; denominator of the original EV fraction |
| side selection | At T-60, price both native ask sides for $5; choose the side with the greatest strictly positive after-fee EV; exact tie favors UP |
| qualification | net_ev_usd / requested_gross_usd >= 1%; original implementation allows 1e-12 numerical tolerance |
| price cap | None in the existing engine: FAK-style marketable BUY walks saved asks at execution; no T-60-derived limit price is passed |
| book depth | Walk complete saved ask ladder when depth is known; otherwise assume full requested fill at saved best ask; known insufficient depth may partially fill |
| minimum order | Current Gamma orderMinSize when present, otherwise $5 USDC fallback; checked against requested gross amount, not the eventual partial fill; historical minimum metadata is unavailable |
| fee rate | Date-assigned BTC5m schedule: 7.2% before 2026-05-06, 7.0% from 2026-05-06; exponent 1; transition date is a stated estimate |
| fee formula | Per price level: shares * rate * price * (1-price); the fee engine uses Decimal and its documented rounding |
| V1 BUY fee | Before the assigned V1/V2 collection change, fee is outcome shares, floored to 1e-6 per saved aggregate price level; cash debit is gross |
| V2 BUY fee | Cash fee rounded half-up to 5 decimals per order with 0.00001 minimum; net shares equal gross shares; cash debit is gross plus fee |
| expected payout | p_D(selected side) * net shares; a winning token pays $1 per net share |
| break-even | cash_debit_usd / net_shares; the exact p_D at which net expected value is zero |
| T-59 | Use the T-60 chosen side and same requested stake; price its T-59 ask ladder only. T-59 cannot select another side or retroactively qualify the decision |

The original percent-EV denominator is the requested gross stake, not the fee-inclusive cash debit and not the filled amount. For a full $5 scenario, `EV_fraction = (p_D * net_shares - cash_debit) / 5`. For a partial T-59 fill the diagnostic keeps that same $5 denominator and separately reports filled gross. The exact zero-EV threshold is `p_break_even = cash_debit / net_shares`; the zero-EV condition is algebraically equivalent to `p_D = p_break_even`.

### Hand-reconciled example

Median-by-T-60-EV holdout example (chosen before looking at outcome): `2026-09-05T04:30:00+00:00`, side `up`. D assigns `0.52830082` to that side, versus normalized market reference `0.50500000`. At executable T-60 depth the VWAP is `0.51000000` for gross filled `$5.000000`; gross shares are `9.80392157`, fee shares `0.00000000`, net shares `9.80392157`, cash fee `$0.17150000`, and cash debit `$5.17150000` (gross plus fee). Expected payout is `0.52830082 * 9.80392157 = $5.17941977`. Subtracting cash debit gives net EV `$0.00791977`; it is `0.1584%` of the requested $5 gross stake and `0.1531%` of the fee-inclusive cash debit. Break-even is `$5.17150000 / 9.80392157 = 0.52749300`. The gap to the 1% hurdle is `$0.04208023`, or `0.8416` percentage points. T-59 same-side VWAP is `0.51` and the hypothetical fill status is `full`; it does not change the T-60 decision.

### EV waterfall for positive T-60 intents

| period | scenario | intent_rows | priceable_rows | positive_net_ev_rows | at_or_above_1pct_rows | partial_fills | median_vwap | median_fee_usd_equivalent | median_filled_gross_usd | net_ev_usd_p10 | net_ev_usd_median | net_ev_usd_p90 | net_ev_usd_max | probability_minus_break_even_p10 | probability_minus_break_even_median | probability_minus_break_even_p90 | net_ev_fraction_requested_median | net_ev_fraction_requested_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| development_oof | normalized_p_market_reference | 490 | 490 | 490 | 490 | 0 | 0.505 | 0.17325 | 5 | 0.0596959 | 0.0664156 | 0.158309 | 0.226623 | 0.00614868 | 0.00634177 | 0.0157874 | 0.0132831 | 0.0453246 |
| development_oof | native_midpoint | 490 | 490 | 490 | 490 | 0 | 0.505 | 0.17325 | 5 | 0.0596959 | 0.0664156 | 0.158309 | 0.226623 | 0.00614868 | 0.00634177 | 0.0157874 | 0.0132831 | 0.0453246 |
| development_oof | t60_best_ask | 490 | 490 | 490 | 131 | 0 | 0.51 | 0.1715 | 5 | 0.0111628 | 0.0139744 | 0.104937 | 0.165258 | 0.00116093 | 0.0013186 | 0.0106656 | 0.00279488 | 0.0330516 |
| development_oof | t60_executable_depth | 490 | 490 | 490 | 131 | 0 | 0.51 | 0.1715 | 5 | 0.0111628 | 0.0139744 | 0.104937 | 0.165258 | 0.00116093 | 0.0013186 | 0.0106656 | 0.00279488 | 0.0330516 |
| development_oof | t59_same_intent | 490 | 488 | 482 | 129 | 0 | 0.51 | 0.1715 | 5 | 0.0111628 | 0.0139744 | 0.108828 | 0.188567 | 0.00115338 | 0.0013186 | 0.0109002 | 0.00279488 | 0.0377134 |
| holdout_historical_diagnostic | normalized_p_market_reference | 37 | 37 | 37 | 37 | 0 | 0.495 | 0.17675 | 5 | 0.0569027 | 0.0576711 | 0.0675591 | 0.121815 | 0.00561532 | 0.00580257 | 0.00586437 | 0.0115342 | 0.024363 |
| holdout_historical_diagnostic | native_midpoint | 37 | 37 | 37 | 37 | 0 | 0.495 | 0.17675 | 5 | 0.0569027 | 0.0576711 | 0.0675591 | 0.121815 | 0.00561532 | 0.00580257 | 0.00586437 | 0.0115342 | 0.024363 |
| holdout_historical_diagnostic | t60_best_ask | 37 | 37 | 37 | 0 | 0 | 0.5 | 0.175 | 5 | 0.00637895 | 0.00791977 | 0.00911277 | 0.0351392 | 0.000589041 | 0.000807817 | 0.000838342 | 0.00158395 | 0.00702783 |
| holdout_historical_diagnostic | t60_executable_depth | 37 | 37 | 37 | 0 | 0 | 0.5 | 0.175 | 5 | 0.00637895 | 0.00791977 | 0.00911277 | 0.0351392 | 0.000589041 | 0.000807817 | 0.000838342 | 0.00158395 | 0.00702783 |
| holdout_historical_diagnostic | t59_same_intent | 37 | 35 | 35 | 0 | 0 | 0.51 | 0.1715 | 5 | 0.00635016 | 0.00791977 | 0.00913316 | 0.0351392 | 0.000587809 | 0.000807817 | 0.000838476 | 0.00158395 | 0.00702783 |

Each reported stage uses the unchanged T-60 D probability and selected side. The normalized p_market stage is theoretical; native midpoint is a non-executable reference; best ask isolates crossing the spread; full depth adds ask-ladder costs; T-59 adds the one-second quote/depth change. For each adjacent step, `change in net EV = change in pre-fee EV + change in fee effect`; the per-intent CSV contains both terms and the exact net-EV difference. If T-59 fills less than $5, the report shows the smaller filled gross and net shares separately; that difference is not labelled only as slippage. Missing stages stay missing.

`positive_ev_below_1pct_cases.csv` lists all 37 holdout cases, with their timestamp, fixed side, costs, p_break_even, T-59 shadow execution, and cutoff-safe realized outcome. Realized outcomes are descriptive only and do not imply a strategy.

### Distribution across the 37 below-hurdle cases

| measure | n | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|
| selected_side_D_probability | 37 | 0.306208 | 0.468052 | 0.49831 | 0.518323 | 0.528301 | 0.532186 | 0.53826 |
| p_break_even_T60 | 37 | 0.304413 | 0.467325 | 0.497472 | 0.5175 | 0.527493 | 0.531485 | 0.537472 |
| p_D_minus_p_break_even_T60 | 37 | 0.000465617 | 0.000589041 | 0.000807817 | 0.000807817 | 0.000832659 | 0.000838342 | 0.00297823 |
| T60_net_EV_USD | 37 | 0.00521684 | 0.00637895 | 0.00791977 | 0.00791977 | 0.00849652 | 0.00911277 | 0.0351392 |
| T60_EV_fraction_requested | 37 | 0.00104337 | 0.00127579 | 0.00158395 | 0.00158395 | 0.0016993 | 0.00182255 | 0.00702783 |
| T60_cash_debit_USD | 37 | 5.168 | 5.1701 | 5.1715 | 5.175 | 5.182 | 5.1925 | 5.2485 |
| T60_fee_equivalent_USD | 37 | 0.168 | 0.1701 | 0.1715 | 0.175 | 0.182 | 0.1925 | 0.2485 |
| T60_VWAP | 37 | 0.29 | 0.45 | 0.48 | 0.5 | 0.51 | 0.514 | 0.52 |
| T59_cash_debit_USD_shadow | 35 | 5.168 | 5.1694 | 5.1715 | 5.1715 | 5.18375 | 5.1925 | 5.2485 |
| T59_VWAP_shadow | 35 | 0.29 | 0.45 | 0.475 | 0.51 | 0.51 | 0.516 | 0.52 |

## Where probability quality improves

| period | n | delta_log_loss_D_minus_A | delta_brier_D_minus_A | min_signed_probability_correction | median_signed_probability_correction | max_signed_probability_correction | mean_abs_probability_correction | median_abs_probability_correction | p95_abs_probability_correction |
|---|---|---|---|---|---|---|---|---|---|
| development_oof | 8675 | -0.000271607 | -0.000135081 | -0.0268208 | 0.0037011 | 0.0368392 | 0.00788878 | 0.00576886 | 0.0236743 |
| holdout_historical_diagnostic | 24735 | -0.000384979 | -0.000194594 | -0.02021 | -0.0105555 | 0.0254502 | 0.0125319 | 0.0126156 | 0.0202079 |

### By existing training-defined economic status

| period | economic_status | n | delta_log_loss_D_minus_A | delta_brier_D_minus_A | mean_abs_probability_correction |
|---|---|---|---|---|---|
| development_oof | at_or_above_1pct | 131 | -0.00205506 | -0.000997749 | 0.0322339 |
| development_oof | no_priceable_t60_side | 1266 | -0.000200367 | -0.000100162 | 0.0061931 |
| development_oof | nonpositive_net_ev | 6919 | -0.00017803 | -8.90651e-05 | 0.0068876 |
| development_oof | positive_ev_below_1pct | 359 | -0.00167555 | -0.000830296 | 0.0242806 |
| holdout_historical_diagnostic | no_priceable_t60_side | 1625 | -0.00137497 | -0.000689035 | 0.0126641 |
| holdout_historical_diagnostic | nonpositive_net_ev | 23073 | -0.000312656 | -0.000158423 | 0.0125054 |
| holdout_historical_diagnostic | positive_ev_below_1pct | 37 | -0.00200562 | -0.00103528 | 0.0232446 |

Status is mechanically defined by the frozen $5 T-60 evaluator, not selected by outcomes. A zero count for the 1%-qualified holdout tier means there is no holdout evidence about quality specifically among trade-qualified cases.

### All months

| period | month_utc | n | delta_log_loss_D_minus_A | delta_brier_D_minus_A | mean_abs_probability_correction |
|---|---|---|---|---|---|
| development_oof | 2026-05 | 5253 | -8.91587e-05 | -4.46926e-05 | 0.00465009 |
| development_oof | 2026-06 | 3422 | -0.000551678 | -0.000273833 | 0.0128604 |
| holdout_historical_diagnostic | 2026-07 | 6795 | -0.000593916 | -0.000302913 | 0.0127109 |
| holdout_historical_diagnostic | 2026-08 | 7576 | -0.000611543 | -0.000306616 | 0.0125953 |
| holdout_historical_diagnostic | 2026-09 | 8639 | -0.000335605 | -0.000167985 | 0.0123932 |
| holdout_historical_diagnostic | 2026-10 | 1725 | 0.00118583 | 0.000590816 | 0.0122426 |

October is retained: read its delta columns with their signs; positive means D is worse than A. The full 3-context-by-period table in `quality_by_existing_training_bins.csv` preserves the saved d, price, and volatility bins; no new holdout quantiles or profitable subsets were searched.

### Full-sample quality versus correction size, spread, and ask premium

| period | associated_measure | quality_measure | n | pearson_correlation |
|---|---|---|---|---|
| development_oof | mean_priceable_side_spread | delta_log_loss_D_minus_A | 7409 | 1.38145e-05 |
| development_oof | mean_priceable_side_spread | delta_brier_D_minus_A | 7409 | 5.56398e-05 |
| development_oof | mean_priceable_side_spread | abs_probability_correction | 7409 | 0.13189 |
| development_oof | mean_priceable_side_ask_premium | delta_log_loss_D_minus_A | 7409 | 9.93783e-05 |
| development_oof | mean_priceable_side_ask_premium | delta_brier_D_minus_A | 7409 | 0.000139331 |
| development_oof | mean_priceable_side_ask_premium | abs_probability_correction | 7409 | 0.132342 |
| development_oof | abs_probability_correction | delta_log_loss_D_minus_A | 8675 | -0.0229745 |
| development_oof | abs_probability_correction | delta_brier_D_minus_A | 8675 | -0.0228363 |
| holdout_historical_diagnostic | mean_priceable_side_spread | delta_log_loss_D_minus_A | 23110 | -0.00671476 |
| holdout_historical_diagnostic | mean_priceable_side_spread | delta_brier_D_minus_A | 23110 | -0.00684643 |
| holdout_historical_diagnostic | mean_priceable_side_spread | abs_probability_correction | 23110 | 0.0294116 |
| holdout_historical_diagnostic | mean_priceable_side_ask_premium | delta_log_loss_D_minus_A | 23110 | -0.00627945 |
| holdout_historical_diagnostic | mean_priceable_side_ask_premium | delta_brier_D_minus_A | 23110 | -0.00641358 |
| holdout_historical_diagnostic | mean_priceable_side_ask_premium | abs_probability_correction | 23110 | 0.0295091 |
| holdout_historical_diagnostic | abs_probability_correction | delta_log_loss_D_minus_A | 24735 | -0.0122414 |
| holdout_historical_diagnostic | abs_probability_correction | delta_brier_D_minus_A | 24735 | -0.0123978 |

These Pearson associations use every quality-labeled row in each period and average the available priceable-side quote measures at T-60; they do not create bins or select subgroups. A negative correlation for a D-minus-A loss means the score difference tends to favor D as the associated measure increases. Ask premium is native best ask minus normalized p_market on each side; it is descriptive, not a realized cost estimate. The full-sample loss associations with correction magnitude, spread, and ask premium are close to zero, so these data do not show that quality gains are concentrated among small or large corrections, or at high spread/cost.

### Existing training-defined context buckets

| period | context | training_defined_bin | n | delta_log_loss_D_minus_A | delta_brier_D_minus_A | mean_abs_probability_correction |
|---|---|---|---|---|---|---|
| development_oof | d_btc_minus_market | Q1 | 2317 | -0.000344823 | -0.000173248 | 0.00650224 |
| development_oof | d_btc_minus_market | Q2 | 1956 | -0.000206094 | -0.000103146 | 0.0046434 |
| development_oof | d_btc_minus_market | Q3 | 1933 | -4.59753e-05 | -2.26585e-05 | 0.00588018 |
| development_oof | d_btc_minus_market | Q4 | 2386 | -0.000456032 | -0.000224356 | 0.0136221 |
| development_oof | d_btc_minus_market | <NA> | 83 | 0.000275225 | 0.000125968 | 0.00504 |
| development_oof | p_market | Q1 | 2044 | -0.00039044 | -0.000190225 | 0.00862819 |
| development_oof | p_market | Q2 | 4271 | -0.000660104 | -0.000329841 | 0.00789175 |
| development_oof | p_market | Q3 | 1501 | -0.000233617 | -0.000116674 | 0.00488647 |
| development_oof | p_market | Q4 | 625 | 0.00285499 | 0.00142081 | 0.0134847 |
| development_oof | p_market | <NA> | 234 | -0.000737336 | -0.000372368 | 0.0056878 |
| development_oof | btc_sigma_5m | Q1 | 2545 | -0.000235042 | -0.000117469 | 0.00461421 |
| development_oof | btc_sigma_5m | Q2 | 1219 | -0.000259167 | -0.000129567 | 0.00535965 |
| development_oof | btc_sigma_5m | Q3 | 1242 | 0.000366863 | 0.000183243 | 0.00725952 |
| development_oof | btc_sigma_5m | Q4 | 3511 | -0.000552054 | -0.000274253 | 0.011203 |
| development_oof | btc_sigma_5m | <NA> | 158 | 0.00025653 | 0.000129027 | 0.0114454 |
| holdout_historical_diagnostic | d_btc_minus_market | Q1 | 6816 | -0.00099832 | -0.000502985 | 0.0173498 |
| holdout_historical_diagnostic | d_btc_minus_market | Q2 | 5419 | 0.000319036 | 0.000159322 | 0.0119016 |
| holdout_historical_diagnostic | d_btc_minus_market | Q3 | 5329 | -5.96735e-05 | -3.00454e-05 | 0.0113509 |
| holdout_historical_diagnostic | d_btc_minus_market | Q4 | 7105 | -0.000609127 | -0.000306792 | 0.00926491 |
| holdout_historical_diagnostic | d_btc_minus_market | <NA> | 66 | 0.00301657 | 0.00138745 | 0.0137702 |
| holdout_historical_diagnostic | p_market | Q1 | 9826 | -0.000248283 | -0.000124778 | 0.0127239 |
| holdout_historical_diagnostic | p_market | Q2 | 10300 | -0.00100878 | -0.000504298 | 0.0121698 |
| holdout_historical_diagnostic | p_market | Q3 | 258 | 0.00361667 | 0.0018071 | 0.012087 |
| holdout_historical_diagnostic | p_market | Q4 | 4267 | 0.000518513 | 0.000250511 | 0.0129823 |
| holdout_historical_diagnostic | p_market | <NA> | 84 | 0.00192914 | 0.00085592 | 0.012956 |
| holdout_historical_diagnostic | btc_sigma_5m | Q1 | 8070 | -0.000325737 | -0.000167715 | 0.0120659 |
| holdout_historical_diagnostic | btc_sigma_5m | Q2 | 7401 | -0.00107103 | -0.000535139 | 0.0127619 |
| holdout_historical_diagnostic | btc_sigma_5m | Q3 | 5652 | 0.000172183 | 8.56507e-05 | 0.0126925 |
| holdout_historical_diagnostic | btc_sigma_5m | Q4 | 3073 | -7.99905e-05 | -4.06285e-05 | 0.0129228 |
| holdout_historical_diagnostic | btc_sigma_5m | <NA> | 539 | 0.00056694 | 0.000262522 | 0.0124379 |

## What the 43 stumps learned

The `d_btc_minus_market` feature supplies 28 of 43 splits, 72.1% of split gain, and 59.7% of grouped absolute centered contribution on the holdout final-model predictions. Basis contributes 34.0% and `btc_sigma_5m` contributes 6.3% of that grouped absolute amount. No split uses raw market logit, BTC returns, or spreads. This is a piecewise correction mainly along the BTC-market difference, with basis and volatility context; because `d` includes market logit, it does not isolate incremental BTC information.

The frozen final model has 43 one-split trees and 43 splits. Centering subtracts each stump's unweighted mean of its two leaf values; the sum of those removed values is the fixed constant `0.0105166645363`. Grouped centered tree contributions plus this constant reproduce every holdout final-model raw margin; maximum observed reconstruction error is `2.78e-17`. Development's saved OOF probabilities came from fold-specific models and are used for quality metrics, not this final-model decomposition.

| feature | split_count | total_gain | gain_share | thresholds |
|---|---|---|---|---|
| logit_p_market | 0 | 0 | 0 |  |
| d_btc_minus_market | 28 | 293.984 | 0.720942 | 0.03585358885; 0.03585358885; 0.03585358885; 0.03585358885; 0.03585358885; 0.03585358885; 0.03585358885; 0.03585358885; 0.03585358885; 0.03585358885; 0.03585358885; 0.03585358885; 0.03585358885; -0.07436317208; 0.03585358885; -0.07436317208; 0.03585358885; -0.07436317208; 0.03585358885; -0.07436317208; 0.03585358885; -0.07436317208; 0.03585358885; -0.07436317208; 0.03585358885; -0.07436317208; 0.1514089196; 0.03585358885 |
| btc_return_1m | 0 | 0 | 0 |  |
| btc_return_5m | 0 | 0 | 0 |  |
| btc_return_15m | 0 | 0 | 0 |  |
| btc_sigma_5m | 3 | 17.3217 | 0.0424783 | 0.0006556677785; 0.0006556677785; 0.0006556677785 |
| futures_index_basis_rel_1m | 12 | 96.4715 | 0.236579 | -0.0005947036698; -0.0005947036698; -0.0005947036698; -0.0005947036698; -0.0005947036698; -0.0005947036698; -0.0005947036698; -0.0005947036698; -0.0005947036698; -0.0005947036698; -0.0005947036698; -0.0005947036698 |
| spread_up | 0 | 0 | 0 |  |
| spread_down | 0 | 0 | 0 |  |

Per-period grouped feature contribution shares are in `feature_contributions.csv`; the exact 43 thresholds, gains, and leaf values are in `tree_splits.csv`. Gain and additive contribution are descriptive, not causal. The feature `d = logit(p_BTC)-logit(p_market)` already contains the market probability, so its use does not isolate incremental BTC information.

## Reproduction and limitations

Run from the repository root: `python run_btc_conditional_market_offset_diagnostics_20261018.py`. It verifies frozen hashes, loads saved predictions/model/snapshots, reproduces the existing T-60 rule, and writes separate diagnostic outputs. It does not train, tune, refit, fetch data, replay archives, or alter the original experiment artifacts.

The exact fee amount and V1/V2 cutover remain estimates in the accepted fee registry; ask depth is a saved historical snapshot, and incomplete-depth rows use the engine's top-of-book full-fill proxy. Current Gamma minimums are not historical. `btc_sigma_5m` is sqrt(5) times the population standard deviation of the prior 60 contiguous one-minute close returns. T-59 is a one-second diagnostic, not a live-liquidity forecast. Positive forecast EV is not evidence of realized profit.

Diagnostic runtime: 22.7s. Detailed read-only market scenarios are under `data/analysis/polymarket/BTC/btc_conditional_market_offset_20261018/diagnostics/`. Summary tables are in `reports/btc_preopen/btc_conditional_market_offset_20261018_diagnostics/`.
