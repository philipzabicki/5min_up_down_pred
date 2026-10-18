# BASE70 information value and Platt calibration — 2026-10-18

## Conclusion

This is an exploratory comparison on the frozen development sample and the existing fixed-$5 policy. It does not establish a deployment edge. RAW BASE70 is evaluated against HALF and the causal daily PRIOR; PLATT is compared with RAW on identical scored rows.

- The saved BASE70 anchor ledger was reproduced before evaluation: 1,452 fills, $7,260.00 turnover, $259.71 fees, and $260.05 net settled PnL. All 1,452 fills in that ledger were settled by cutoff.
- Common forecast calendar: 20,735 rows with all four probabilities, 72 UTC days, from `2026-04-20T00:00:00+00:00` through `2026-06-30T23:50:00+00:00`.
- Saved OOF timestamp provenance: 0 common-calendar rows have a recorded prediction timestamp; T−59 availability is assumed for 20,735 rows under the frozen one-second contract.
- Probability-quality sample: 20,718 common rows with official labels released by cutoff, 10,323 UP and 10,395 DOWN, from `2026-04-20T00:00:00+00:00` through `2026-06-30T23:50:00+00:00`.

Key findings:

- RAW improves all three probability scores over HALF and PRIOR on the shared 20,718-label sample; the 95% paired intervals exclude zero for both contrasts on log loss, Brier, and AUC.
- Under the frozen independent-$5 policy, RAW returns $267.44, versus $-101.27 for HALF and $-137.08 for PRIOR. Mean daily PnL contrast intervals are $0.33 to $10.97 (RAW-HALF) and $0.73 to $11.62 (RAW-PRIOR). These development comparisons remain exploratory.
- PLATT does not improve on RAW in this comparison: its point log loss and Brier are slightly worse and AUC slightly lower, all three paired intervals include zero, and its PnL is $-7.39 lower. The mean daily PnL contrast interval is $-3.43 to $2.84.
- The RAW advantage over HALF/PRIOR is driven by selecting different fills: one-variant settled-fill components are $379.32 and $415.13; opposite-side components are $-10.61 and $-10.61.
- Results are uncertain and concentrated: RAW's five best days contribute 81.6% of its PnL, while PLATT's five best days contribute 100.8%; the remaining-day PnL is $49.23 and $-2.07, respectively. Both RAW and PLATT mean-daily-PnL and PnL/turnover bootstrap intervals include zero.

## Probability quality

Lower log loss and Brier are better; higher AUC is better. RAW and PLATT use exactly the same rows. HALF AUC equals 0.5 on this two-class sample. PRIOR is fixed within each UTC day, so its AUC reflects differences between daily prior values.

| Variant | Log loss | Brier | AUC |
|---|---:|---:|---:|
| HALF | 0.693147 | 0.250000 | 0.5000 |
| PRIOR | 0.693213 | 0.250033 | 0.4979 |
| RAW | 0.692052 | 0.249453 | 0.5274 |
| PLATT | 0.692152 | 0.249503 | 0.5266 |

Paired differences are left minus right. The 95% intervals use 2,000 shared circular three-day UTC block resamples (seed 20261018); negative log-loss/Brier differences and positive AUC differences favor the left variant.

| Contrast | Metric | Point difference | 95% interval |
|---|---|---:|---:|
| PLATT_MINUS_PRIOR | auc | 0.028781 | 0.019283 to 0.039562 |
| PLATT_MINUS_PRIOR | brier | -0.000530 | -0.000885 to -0.000175 |
| PLATT_MINUS_PRIOR | log_loss | -0.001062 | -0.001773 to -0.000350 |
| PLATT_MINUS_RAW | auc | -0.000805 | -0.002340 to 0.000848 |
| PLATT_MINUS_RAW | brier | 0.000050 | -0.000035 to 0.000138 |
| PLATT_MINUS_RAW | log_loss | 0.000100 | -0.000071 to 0.000278 |
| RAW_MINUS_HALF | auc | 0.027437 | 0.019792 to 0.035312 |
| RAW_MINUS_HALF | brier | -0.000547 | -0.000843 to -0.000259 |
| RAW_MINUS_HALF | log_loss | -0.001095 | -0.001689 to -0.000518 |
| RAW_MINUS_PRIOR | auc | 0.029586 | 0.019099 to 0.040998 |
| RAW_MINUS_PRIOR | brier | -0.000580 | -0.000878 to -0.000294 |
| RAW_MINUS_PRIOR | log_loss | -0.001162 | -0.001758 to -0.000587 |

Monthly common-sample scores:

| UTC month | Rows | HALF LL | HALF Brier | HALF AUC | PRIOR LL | PRIOR Brier | PRIOR AUC | RAW LL | RAW Brier | RAW AUC | PLATT LL | PLATT Brier | PLATT AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-04 | 3,168 | 0.693147 | 0.250000 | 0.5000 | 0.693149 | 0.250001 | 0.4969 | 0.691610 | 0.249233 | 0.5325 | 0.691558 | 0.249208 | 0.5309 |
| 2026-05 | 8,917 | 0.693147 | 0.250000 | 0.5000 | 0.693240 | 0.250046 | 0.4863 | 0.692259 | 0.249556 | 0.5246 | 0.692439 | 0.249646 | 0.5234 |
| 2026-06 | 8,633 | 0.693147 | 0.250000 | 0.5000 | 0.693210 | 0.250031 | 0.4833 | 0.692000 | 0.249427 | 0.5296 | 0.692073 | 0.249463 | 0.5280 |

## Fixed-policy economics

Each variant independently chooses the side and qualifies using the same fixed gross $5 order and T-60 net-EV hurdle, then the existing T-59 execution engine. Orders are independent: no $100 portfolio or available-cash limit is applied. Intents are counted from the decision-time EV test before T-59 execution. Unresolved fills are included in fill, turnover, and fee counts, then reported separately; they are excluded from settled PnL, settled turnover, and the PnL-EV gap.

| Variant | Intents | Fills | Settlements | Unresolved fills | Filled turnover | Fees | Net settled PnL | PnL / settled turnover | EV on settled fills | PnL − EV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HALF | 642 | 635 | 635 | 0 | $3,175.00 | $122.38 | $-101.27 | -3.19% | $259.07 | $-360.34 |
| PRIOR | 668 | 660 | 660 | 0 | $3,300.00 | $126.89 | $-137.08 | -4.15% | $255.20 | $-392.28 |
| RAW | 1,047 | 1,038 | 1,038 | 0 | $5,190.00 | $188.84 | $267.44 | 5.15% | $279.60 | $-12.17 |
| PLATT | 1,466 | 1,452 | 1,452 | 0 | $7,260.00 | $259.71 | $260.05 | 3.58% | $356.58 | $-96.53 |

| Variant | UTC month | Intents | Fills (UP/DOWN) | Settlements (UP/DOWN) | Unresolved fills | Filled turnover | Fees | Net settled PnL (UP/DOWN) | PnL / settled turnover | EV on settled fills | PnL-EV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HALF | 2026-04 | 5 | 5 (0/5) | 5 (0/5) | 0 | $25.00 | $1.00 | $-0.54 ($0.00/$-0.54) | -2.15% | $2.69 | $-3.23 |
| HALF | 2026-05 | 346 | 341 (115/226) | 341 (115/226) | 0 | $1,705.00 | $66.68 | $-2.99 ($8.58/$-11.56) | -0.18% | $179.75 | $-182.74 |
| HALF | 2026-06 | 291 | 289 (122/167) | 289 (122/167) | 0 | $1,445.00 | $54.70 | $-97.75 ($-74.34/$-23.40) | -6.76% | $76.63 | $-174.38 |
| PRIOR | 2026-04 | 5 | 4 (3/1) | 4 (3/1) | 0 | $20.00 | $0.79 | $-5.58 ($-15.00/$9.42) | -27.90% | $2.15 | $-7.73 |
| PRIOR | 2026-05 | 361 | 356 (132/224) | 356 (132/224) | 0 | $1,780.00 | $69.40 | $2.41 ($3.60/$-1.18) | 0.14% | $177.84 | $-175.43 |
| PRIOR | 2026-06 | 302 | 300 (129/171) | 300 (129/171) | 0 | $1,500.00 | $56.70 | $-133.91 ($-89.78/$-44.13) | -8.93% | $75.21 | $-209.12 |
| RAW | 2026-04 | 63 | 63 (14/49) | 63 (14/49) | 0 | $315.00 | $11.45 | $-10.39 ($25.32/$-35.70) | -3.30% | $9.40 | $-19.78 |
| RAW | 2026-05 | 455 | 447 (124/323) | 447 (124/323) | 0 | $2,235.00 | $83.85 | $66.12 ($4.86/$61.26) | 2.96% | $177.30 | $-111.18 |
| RAW | 2026-06 | 529 | 528 (355/173) | 528 (355/173) | 0 | $2,640.00 | $93.54 | $211.71 ($153.24/$58.47) | 8.02% | $92.91 | $118.80 |
| PLATT | 2026-04 | 166 | 163 (110/53) | 163 (110/53) | 0 | $815.00 | $29.16 | $22.29 ($59.28/$-36.98) | 2.74% | $27.94 | $-5.65 |
| PLATT | 2026-05 | 573 | 565 (240/325) | 565 (240/325) | 0 | $2,825.00 | $103.70 | $99.49 ($110.15/$-10.66) | 3.52% | $198.69 | $-99.20 |
| PLATT | 2026-06 | 727 | 724 (439/285) | 724 (439/285) | 0 | $3,620.00 | $126.85 | $138.26 ($141.76/$-3.50) | 3.82% | $129.95 | $8.32 |

Filled and settled sides over the full sample:

| Variant | UP fills | DOWN fills | UP settlements | DOWN settlements |
|---|---:|---:|---:|---:|
| HALF | 237 | 398 | 237 | 398 |
| PRIOR | 264 | 396 | 264 | 396 |
| RAW | 493 | 545 | 493 | 545 |
| PLATT | 789 | 663 | 789 | 663 |

Decision-time skips and missing-data reasons:

| Variant | Rejection / missing-data reasons |
|---|---|
| HALF | ambiguous_equal_source_timestamp_order: 5; below_minimum_net_ev_fraction: 529; crossed_token_book_at_entry: 1; native_ask_bbo_reconciliation_mismatch: 1; no_gamma_market_record: 7; no_positive_net_expected_value: 7,647; no_priceable_native_ask_side: 11,910
| PRIOR | ambiguous_equal_source_timestamp_order: 5; below_minimum_net_ev_fraction: 317; crossed_token_book_at_entry: 1; native_ask_bbo_reconciliation_mismatch: 2; no_gamma_market_record: 7; no_positive_net_expected_value: 7,833; no_priceable_native_ask_side: 11,910
| RAW | ambiguous_equal_source_timestamp_order: 4; below_minimum_net_ev_fraction: 608; crossed_token_book_at_entry: 1; native_ask_bbo_reconciliation_mismatch: 4; no_gamma_market_record: 7; no_positive_net_expected_value: 7,163; no_priceable_native_ask_side: 11,910
| PLATT | ambiguous_equal_source_timestamp_order: 6; below_minimum_net_ev_fraction: 698; crossed_token_book_at_entry: 3; native_ask_bbo_reconciliation_mismatch: 5; no_gamma_market_record: 7; no_positive_net_expected_value: 6,654; no_priceable_native_ask_side: 11,910

Execution-side decomposition uses settled executions only. Unresolved executions remain unpriced and are counted separately. The components reconcile exactly to each full settled-PnL difference.

| Contrast | Same side in both | Settled fill in one variant | Opposite sides | Total PnL difference |
|---|---:|---:|---:|---:|
| RAW_MINUS_HALF | $0.00 | $379.32 | $-10.61 | $368.71 |
| RAW_MINUS_PRIOR | $0.00 | $415.13 | $-10.61 | $404.52 |
| PLATT_MINUS_RAW | $0.00 | $-7.39 | $0.00 | $-7.39 |
| PLATT_MINUS_PRIOR | $0.00 | $393.16 | $3.97 | $397.13 |

## Uncertainty and concentration

The economic bootstrap resamples the same complete daily calendar for every variant, including days with no fills. It reports mean daily settled PnL and PnL divided by settled turnover; each ratio is recomputed from resampled sums. Zero-turnover replicas are undefined.

| Variant | Mean daily PnL and 95% interval | PnL / settled turnover and 95% interval |
|---|---:|---:|
| HALF | $-1.4065 ($-4.2781, $1.3566) | -3.19% (-11.48%, 2.77%); 2000/2000 valid |
| PRIOR | $-1.9039 ($-4.8874, $0.8257) | -4.15% (-12.98%, 1.59%); 2000/2000 valid |
| RAW | $3.7144 ($-0.1967, $8.2680) | 5.15% (-0.36%, 9.42%); 2000/2000 valid |
| PLATT | $3.6118 ($-0.2642, $7.9719) | 3.58% (-0.31%, 7.21%); 2000/2000 valid |

| Contrast | Mean daily PnL difference | 95% interval |
|---|---:|---:|
| RAW_MINUS_HALF | $5.1210 | $0.3333 to $10.9728 |
| RAW_MINUS_PRIOR | $5.6183 | $0.7328 to $11.6210 |
| PLATT_MINUS_RAW | $-0.1027 | $-3.4282 to $2.8383 |
| PLATT_MINUS_PRIOR | $5.5156 | $0.3096 to $12.0861 |

| Contrast | PnL / settled turnover difference | 95% interval | Valid replications |
|---|---:|---:|---:|
| RAW_MINUS_HALF | 8.34% | 0.61% to 17.84% | 2000/2000 valid |
| RAW_MINUS_PRIOR | 9.31% | 1.54% to 18.96% | 2000/2000 valid |
| PLATT_MINUS_RAW | -1.57% | -5.18% to 2.80% | 2000/2000 valid |
| PLATT_MINUS_PRIOR | 7.74% | -0.05% to 18.19% | 2000/2000 valid |

| Variant | Top-five-day PnL | Share of total PnL | Remaining-day PnL |
|---|---:|---:|---:|
| HALF | $107.59 | N/A | $-208.86 |
| PRIOR | $101.37 | N/A | $-238.45 |
| RAW | $218.21 | 81.6% | $49.23 |
| PLATT | $262.11 | 100.8% | $-2.07 |

The share is shown as N/A when total PnL is non-positive; in those cases the table still reports the top-five and remaining-day PnL amounts.

## Availability and provenance

- Common calendar availability: 20,735 forecasts; 20,728 market records; 20,728 T−60 snapshots; 20,728 T−59 snapshots; 20,718 official outcomes available by the cutoff. These are recorded separately.
- The saved BASE70 OOF files contain no populated `prediction_available_at_utc` values. Forecasts are present in the saved OOF rows; availability at T−59 is a timing assumption from the one-second frozen contract, not an observed runtime timestamp.
- PRIOR and Platt both use the official-label history beginning `2026-04-15T17:05:00+00:00`, the release rule `max(resolved_at_utc, market_start_utc + 5 minutes) + 60 seconds`, at least 1,000 earlier labels, and both classes. An exact-boundary label is excluded because the source calibrator requires release time strictly before the day's first decision.
- The source Platt cache was reproduced with the same finite mask and maximum probability difference `3.33e-16`.
- Accounting audits passed for all four engine ledgers. The saved BASE70 settled trades were matched to the new PLATT settled trades before reporting.
- Stage elapsed seconds: `{"common_predictions": 1.0948041000010562, "economics_HALF": 16.6678161000018, "economics_PLATT": 30.57035039999755, "economics_PRIOR": 31.018347999997786, "economics_RAW": 30.587201499998628, "economics_summary": 3.5089002000022447, "execution_calendar": 19.26094490000105, "quality_and_bootstrap": 65.62151689999882, "report": 0.006805199998780154}`. Peak process working set: `3.554` GiB. No feature matrix was loaded, model was trained, archive replay was run, data was downloaded, holdout was evaluated, or live configuration was changed.

Development selection, prior research, and policy choices limit the interpretation of these intervals. A positive PnL point estimate does not establish a future or live advantage.

Prediction and daily PnL tables, source hashes, per-variant ledgers, and bootstrap draws are stored locally under `data/analysis/polymarket/BTC/btc_base70_information_value_20261018`. Reproduce with `python run_btc_base70_information_value_20261018.py`; no command-line arguments are required.
