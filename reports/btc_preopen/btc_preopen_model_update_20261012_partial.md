# BTC pre-open model update: partial report

> This report was built from hash-verified predictions, model artifacts, and corrected T-60/T-59 snapshots already on disk. It did not run training, a model refit, an event replay, or live execution. The evaluated July–October period was previously inspected and is historical diagnostic evidence.

## Status and frozen decision

- Development selected `A_frozen` on 10,795 common markets. `A_frozen` is the earlier reference; `A_reoptimized` is the new challenger; `B_dynamic_price_volume` adds causal price/volume candidates.
- Frozen selection rule: Select the lowest-mean-log-loss newly trained candidate only if its Bonferroni-adjusted 97.5% paired delta-log-loss upper bound versus A_frozen is below zero; otherwise keep A_frozen.
- The best new development point estimate had lower mean log loss than A_frozen, but neither candidate's adjusted interval cleared zero. Keeping A_frozen is the pre-registered uncertainty rule, not evidence that all new models have worse mean metrics.
- B holdout is incomplete: its model tuning ended at 4/5 required successful trials. No B holdout prediction or score is reported.
- The frozen comparison includes A_frozen and A_reoptimized on the same holdout prediction markets; quality metrics use official labels and Binance proxy labels separately.

### Development quality on the frozen common sample

| Forecast | Target | n | Log loss | Brier | AUC |
|---|---|---|---|---|---|
| A_frozen | official_polymarket_outcome | 10795 | 0.692246 | 0.249550 | 0.524894 |
| A_frozen | binance_index_price_proxy | 10795 | 0.692316 | 0.249585 | 0.523694 |
| A_reoptimized | official_polymarket_outcome | 10795 | 0.691908 | 0.249381 | 0.532245 |
| A_reoptimized | binance_index_price_proxy | 10795 | 0.691965 | 0.249409 | 0.530989 |
| B_dynamic_price_volume | official_polymarket_outcome | 10795 | 0.692125 | 0.249490 | 0.527404 |
| B_dynamic_price_volume | binance_index_price_proxy | 10795 | 0.692180 | 0.249517 | 0.526233 |

### Development fold metrics

| Fold | Forecast | Target | n | Log loss | Brier | AUC |
|---|---|---|---|---|---|---|
| dev_2026q2_a | A_frozen | official_polymarket_outcome | 2119 | 0.691967 | 0.249411 | 0.523886 |
| dev_2026q2_a | A_frozen | binance_index_price_proxy | 2119 | 0.692218 | 0.249536 | 0.520029 |
| dev_2026q2_a | A_reoptimized | official_polymarket_outcome | 2119 | 0.692113 | 0.249484 | 0.522204 |
| dev_2026q2_a | A_reoptimized | binance_index_price_proxy | 2119 | 0.692165 | 0.249509 | 0.522033 |
| dev_2026q2_a | B_dynamic_price_volume | official_polymarket_outcome | 2119 | 0.692109 | 0.249482 | 0.523235 |
| dev_2026q2_a | B_dynamic_price_volume | binance_index_price_proxy | 2119 | 0.692287 | 0.249570 | 0.519714 |
| dev_2026q2_b | A_frozen | official_polymarket_outcome | 5253 | 0.692381 | 0.249617 | 0.520995 |
| dev_2026q2_b | A_frozen | binance_index_price_proxy | 5253 | 0.692334 | 0.249594 | 0.521360 |
| dev_2026q2_b | A_reoptimized | official_polymarket_outcome | 5253 | 0.692207 | 0.249530 | 0.528130 |
| dev_2026q2_b | A_reoptimized | binance_index_price_proxy | 5253 | 0.692159 | 0.249506 | 0.528306 |
| dev_2026q2_b | B_dynamic_price_volume | official_polymarket_outcome | 5253 | 0.692336 | 0.249595 | 0.525031 |
| dev_2026q2_b | B_dynamic_price_volume | binance_index_price_proxy | 5253 | 0.692224 | 0.249539 | 0.526495 |
| dev_2026q2_c | A_frozen | official_polymarket_outcome | 3423 | 0.692212 | 0.249532 | 0.531386 |
| dev_2026q2_c | A_frozen | binance_index_price_proxy | 3423 | 0.692348 | 0.249600 | 0.529222 |
| dev_2026q2_c | A_reoptimized | official_polymarket_outcome | 3423 | 0.691322 | 0.249088 | 0.542247 |
| dev_2026q2_c | A_reoptimized | binance_index_price_proxy | 3423 | 0.691543 | 0.249199 | 0.539196 |
| dev_2026q2_c | B_dynamic_price_volume | official_polymarket_outcome | 3423 | 0.691811 | 0.249333 | 0.534816 |
| dev_2026q2_c | B_dynamic_price_volume | binance_index_price_proxy | 3423 | 0.692044 | 0.249450 | 0.531671 |

### Paired development comparisons against A_frozen

| Candidate | n | Delta log loss | Bonferroni-adjusted 97.5% interval |
|---|---|---|---|
| A_reoptimized-A_frozen | 10795 | -0.000338 | [-0.0009445541267960575, 0.00018752105681461552] |
| B_dynamic_price_volume-A_frozen | 10795 | -0.000121 | [-0.0005665552567170468, 0.0003326279193560822] |

The intervals are the frozen paired circular moving 3-UTC-day bootstrap (2,000 replicates, seed 20261012), with the Bonferroni-adjusted 97.5% upper-bound gate. A candidate would replace A_frozen only if its adjusted upper delta-log-loss bound were below zero.

### Feature selection audit

| Fold | Variant | Best k | Selector recommendation | Model features used | Added B candidates | Savings | Required | Gate |
|---|---|---|---|---|---|---|---|---|
| dev_2026q2_a | A_reoptimized | 40 | 40 | 40 | 0 | 0 | 20 | False |
| dev_2026q2_a | B_dynamic_price_volume | 40 | 40 | 45 | 5 | 0 | 20 | False |
| dev_2026q2_b | A_reoptimized | 56 | 56 | 56 | 0 | 0 | 20 | False |
| dev_2026q2_b | B_dynamic_price_volume | 56 | 56 | 61 | 5 | 0 | 20 | False |
| dev_2026q2_c | A_reoptimized | 64 | 64 | 64 | 0 | 0 | 20 | False |
| dev_2026q2_c | B_dynamic_price_volume | 64 | 64 | 70 | 6 | 0 | 20 | False |

In each development fold, best k equalled recommended k, so the 20-feature plateau-savings requirement was not met. B then added candidate features that were not already selected. The selector was run once per fold and reused by both variants.

### Holdout probability quality

| Forecast | Target | n | Log loss | Brier | AUC |
|---|---|---|---|---|---|
| A_frozen | official_polymarket_outcome | 28221 | 0.692466 | 0.249659 | 0.522043 |
| A_frozen | binance_index_price_proxy | 28224 | 0.692521 | 0.249687 | 0.521106 |
| A_reoptimized | official_polymarket_outcome | 28221 | 0.692467 | 0.249660 | 0.521866 |
| A_reoptimized | binance_index_price_proxy | 28224 | 0.692470 | 0.249661 | 0.521066 |

| Target | n | Delta log loss: A_reoptimized minus A_frozen | 95% block CI | Delta Brier |
|---|---|---|---|---|
| official_outcome_up | 28221 | 0.000001 | [-0.00022444626356855462, 0.00023720069543607585] | 0.000001 |
| target_binance_proxy_up | 28224 | -0.000051 | [-0.0002746734345350816, 0.00016056014812011822] | -0.000025 |

Holdout intervals use the same paired circular 3-day UTC block structure and 2,000 replicates as the development procedure; they are descriptive because this period was historically exposed.

## Economic evaluation at the frozen policy

Policy remains 1% net EV at T-60, same side executed at T-59, independent gross order size USD 5, with the accepted fee schedule, depth, partial-fill, and settlement accounting. No T-59 observation selects the side or changes the requested amount. The separate portfolios use 5% of free cash at T-60 and lock stake until settlement.
- Locally verified corrected snapshots: 50,184 T-60 and 50,184 T-59 markets; hashes 38259fadf50c28e455f5d4523610cbac827886e35922076dbba5cb33bd55e0b1 and abee0fdaf31eaf1455b4343e985bf9d5ee38a84dd8eb7e06adf1e08fc0c380d0. Training's prediction-quality evaluator does not run economic replay; the snapshots do exist and were reused here. Missing quote coverage is reported per market in the funnel.
- The earlier A/EV 1% result comparison status is `exact_match`. The exact row-by-row values are in `holdout_fixed_5_economics.csv`.
- Previous market-level ledger comparison: `exact_match`; previous/recomputed qualified signals 4272/4272, previous-only/recomputed-only market IDs 0/0, changed common ledgers 0. Any differing market rows are listed in `previous_a_ledger_mismatches.csv` before interpreting challenger results.

| Forecast | Opportunities | Signals | Fills | Mean forecast P(win) | Observed win rate | Turnover USD | Fee equivalent USD | Expected net T-59 USD | Realized net PnL USD | PnL/turnover % | Wins |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_frozen | 24736 | 4272 | 4018 | 0.526405 | 0.485565 | 20090.000000 | 725.305770 | 1242.331391 | -732.163484 | -3.644418 | 1951 |
| A_reoptimized | 24736 | 2665 | 2521 | 0.514475 | 0.475208 | 12605.000000 | 471.090280 | 1033.957943 | -294.718276 | -2.338106 | 1198 |

### Funnel and EV reconciliation

| Forecast | Confirmed markets | Predictions | T-60 priceable | 1% signals | T-59 fills | Resolved | Wins | Losses | Mean forecast P(win) | Observed win rate | Expected net T-60 USD | Expected net T-59 USD | T-60 to T-59 net EV change USD | Gross repricing USD | Payout residual USD | Realized fee effect USD | Expected minus realized net USD | Expected minus realized USD/fill | 95% block CI USD/fill | Realized net PnL USD | Exclusions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_frozen | 28222 | 28224 | 23113 | 4272 | 4018 | 4018 | 1951 | 2067 | 0.526405 | 0.485565 | 1269.082032 | 1242.331391 | -26.750641 | -27.609931 | -1974.494875 | 725.305770 | 1974.494875 | 0.491412 | [0.2735707835872054, 0.6801005873426893] | -732.163484 | {"ambiguous_equal_source_timestamp_order": 250, "below_frozen_ev_hurdle": 1976, "future_source_timestamp_at_entry": 5, "native_ask_bbo_reconciliation_mismatch": 3, "no_gamma_market_record": 2, "no_native_book_snapshot": 3017, "no_positive_t60_net_ev": 16865, "no_priceable_native_ask_side": 2088} |
| A_reoptimized | 28222 | 28224 | 23113 | 2665 | 2521 | 2521 | 1198 | 1323 | 0.514475 | 0.475208 | 1052.184267 | 1033.957943 | -18.226324 | -18.780354 | -1328.676219 | 471.090280 | 1328.676219 | 0.527043 | [0.21411107001752278, 0.7744330529275738] | -294.718276 | {"ambiguous_equal_source_timestamp_order": 143, "below_frozen_ev_hurdle": 1437, "future_source_timestamp_at_entry": 4, "native_ask_bbo_reconciliation_mismatch": 1, "no_gamma_market_record": 2, "no_native_book_snapshot": 3017, "no_positive_t60_net_ev": 19011, "no_priceable_native_ask_side": 2088} |

For A_frozen, the 1% hurdle was a model-based forecast filter: mean chosen-side P(win) on resolved fills was 52.64% versus a 48.56% realized win rate. Expected net at T-60 was USD 1,269.08; T-59 repricing changed it by USD -26.75 to USD 1,242.33. Realized net PnL was USD -732.16, leaving an expected-minus-realized gap of USD 1,974.49 (USD 0.491/fill; 95% moving-block interval USD 0.274 to 0.680/fill). This is evidence of an outcome/forecast gap among the selected fills in this historical period; it does not identify one cause. The bridge separates decision-time EV from diagnostic T-59 expected EV, and the group ledger breaks results out by side, development-frozen T-60 price/EV bins, and month, including full-depth/proxy shares.

### Monthly independent-order economics

| Forecast | Month UTC | Signals | Fills | Turnover USD | Fees USD | Expected net T-59 USD | Realized net PnL USD |
|---|---|---|---|---|---|---|---|
| A_frozen | 2026-07 | 1287 | 1264 | 6320.000000 | 228.607100 | 389.315684 | -665.669380 |
| A_frozen | 2026-08 | 1345 | 1266 | 6330.000000 | 232.187680 | 570.578742 | 97.686066 |
| A_frozen | 2026-09 | 1290 | 1139 | 5695.000000 | 203.071650 | 226.070313 | 26.922269 |
| A_frozen | 2026-10 | 350 | 349 | 1745.000000 | 61.439340 | 56.366651 | -191.102439 |
| A_reoptimized | 2026-07 | 802 | 780 | 3900.000000 | 145.935780 | 315.314681 | -342.394113 |
| A_reoptimized | 2026-08 | 828 | 780 | 3900.000000 | 149.352450 | 497.491075 | 26.641092 |
| A_reoptimized | 2026-09 | 834 | 761 | 3805.000000 | 139.676790 | 182.180029 | 140.549964 |
| A_reoptimized | 2026-10 | 201 | 200 | 1000.000000 | 36.125260 | 38.972158 | -119.515218 |

Development-frozen group boundaries and all grouped rows are in `holdout_economics_by_side_price_ev_month.csv`.

### Separate continuous portfolios

| Forecast | Starting cash | Trades | Turnover USD | Fees USD | Net PnL USD | Ending cash | Max drawdown | Audit |
|---|---|---|---|---|---|---|---|---|
| A_frozen | 100.000000 | 2 | 10.241250 | 0.358440 | -0.599690 | 99.400310 | 0.051750 | passed |
| A_frozen | 500.000000 | 251 | 3486.255948 | 125.147930 | -404.159498 | 95.840502 | 0.816852 | passed |
| A_frozen | 1000.000000 | 322 | 7306.299677 | 261.669100 | -901.378929 | 98.621071 | 0.905769 | passed |
| A_reoptimized | 100.000000 | 1 | 5.000000 | 0.175000 | -5.175000 | 94.825000 | 0.051750 | passed |
| A_reoptimized | 500.000000 | 156 | 1709.487374 | 62.132070 | -403.171289 | 96.828711 | 0.806343 | passed |
| A_reoptimized | 1000.000000 | 297 | 4592.815532 | 168.297060 | -901.995705 | 98.004295 | 0.901996 | passed |

Each account starts independently at USD 100/500/1,000. Its detailed monthly diagnostics, decisions, trades, equity path, and accounting audit are in the ignored local ledger directory.

### USD 100 portfolio rejection and stop audit

| Forecast | Minimum-order skips | Locked-cash skips | Other skips | Last trade entry UTC | Last decision UTC | Free cash USD | Cost-basis equity USD | 5% stake USD | $5 minimum check |
|---|---|---|---|---|---|---|---|---|---|
| A_frozen | 23104 | 0 | 1630 | 2026-07-01 04:39:01+00:00 | 2026-10-06T23:54:00+00:00 | 99.400310 | 99.400310 | 4.970016 | passed |
| A_reoptimized | 23107 | 0 | 1628 | 2026-07-01 04:19:01+00:00 | 2026-10-06T23:54:00+00:00 | 94.825000 | 94.825000 | 4.741250 | passed |

Both USD 100 ledgers show that every below-minimum skip requested exactly 5% of pre-decision free cash and that each requested amount was below the accepted USD 5 fallback; there were no locked-cash skips. A_frozen's 1,630 other skips were 1,625 unpriceable native asks and 5 no-positive-net-EV decisions. A_reoptimized's 1,628 were 1,625 unpriceable asks, 2 no-positive-net-EV decisions, and 1 below-minimum-net-EV decision. Historical Gamma order minimum metadata is unavailable, so the USD 5 fallback is the auditable threshold. The small portfolio losses reflect a few trades followed by stakes below minimum, not evidence of strategy robustness.

| Forecast | Starting cash | Month UTC | Trades | Turnover USD | Fees USD | Monthly equity PnL USD | Closing equity USD |
|---|---|---|---|---|---|---|---|
| A_frozen | 100.000000 | 2026-06 | 0 | 0 | 0 | 0.000000 | 100.000000 |
| A_frozen | 100.000000 | 2026-07 | 2 | 10.241250 | 0.358440 | -0.599690 | 99.400310 |
| A_frozen | 100.000000 | 2026-08 | 0 | 0 | 0 | 0.000000 | 99.400310 |
| A_frozen | 100.000000 | 2026-09 | 0 | 0 | 0 | 0.000000 | 99.400310 |
| A_frozen | 100.000000 | 2026-10 | 0 | 0 | 0 | 0.000000 | 99.400310 |
| A_frozen | 500.000000 | 2026-06 | 0 | 0 | 0 | 0.000000 | 500.000000 |
| A_frozen | 500.000000 | 2026-07 | 251 | 3486.255948 | 125.147930 | -404.159498 | 95.840502 |
| A_frozen | 500.000000 | 2026-08 | 0 | 0 | 0 | 0.000000 | 95.840502 |
| A_frozen | 500.000000 | 2026-09 | 0 | 0 | 0 | 0.000000 | 95.840502 |
| A_frozen | 500.000000 | 2026-10 | 0 | 0 | 0 | 0.000000 | 95.840502 |
| A_frozen | 1000.000000 | 2026-06 | 0 | 0 | 0 | 0.000000 | 1000.000000 |
| A_frozen | 1000.000000 | 2026-07 | 322 | 7306.299677 | 261.669100 | -901.378929 | 98.621071 |
| A_frozen | 1000.000000 | 2026-08 | 0 | 0 | 0 | 0.000000 | 98.621071 |
| A_frozen | 1000.000000 | 2026-09 | 0 | 0 | 0 | 0.000000 | 98.621071 |
| A_frozen | 1000.000000 | 2026-10 | 0 | 0 | 0 | 0.000000 | 98.621071 |
| A_reoptimized | 100.000000 | 2026-06 | 0 | 0 | 0 | 0.000000 | 100.000000 |
| A_reoptimized | 100.000000 | 2026-07 | 1 | 5.000000 | 0.175000 | -5.175000 | 94.825000 |
| A_reoptimized | 100.000000 | 2026-08 | 0 | 0 | 0 | 0.000000 | 94.825000 |
| A_reoptimized | 100.000000 | 2026-09 | 0 | 0 | 0 | 0.000000 | 94.825000 |
| A_reoptimized | 100.000000 | 2026-10 | 0 | 0 | 0 | 0.000000 | 94.825000 |
| A_reoptimized | 500.000000 | 2026-06 | 0 | 0 | 0 | 0.000000 | 500.000000 |
| A_reoptimized | 500.000000 | 2026-07 | 156 | 1709.487374 | 62.132070 | -403.171289 | 96.828711 |
| A_reoptimized | 500.000000 | 2026-08 | 0 | 0 | 0 | 0.000000 | 96.828711 |
| A_reoptimized | 500.000000 | 2026-09 | 0 | 0 | 0 | 0.000000 | 96.828711 |
| A_reoptimized | 500.000000 | 2026-10 | 0 | 0 | 0 | 0.000000 | 96.828711 |
| A_reoptimized | 1000.000000 | 2026-06 | 0 | 0 | 0 | 0.000000 | 1000.000000 |
| A_reoptimized | 1000.000000 | 2026-07 | 297 | 4592.815532 | 168.297060 | -901.995705 | 98.004295 |
| A_reoptimized | 1000.000000 | 2026-08 | 0 | 0 | 0 | 0.000000 | 98.004295 |
| A_reoptimized | 1000.000000 | 2026-09 | 0 | 0 | 0 | 0.000000 | 98.004295 |
| A_reoptimized | 1000.000000 | 2026-10 | 0 | 0 | 0 | 0.000000 | 98.004295 |

## B pruning and tuning budget

- Holdout B study: 4 COMPLETE, 73 PRUNED, 0 FAIL. The saved continuation preflight had 15 trials and 3 successes; the final study has 77 trials and 4/5 required successes. The latest tuning stage used 1938.8s and stopped at the frozen tuning deadline.
- Median observed duration was 73.6s per completed trial and 28.3s per pruned trial. The first pruning checkpoint is iteration 25 (Optuna step 24); trial curves, all parameters, durations, best iterations, and checkpoint values are in `b_holdout_trials.csv`.
- A high pruned share alone does not prove harmful pruning. 4 of 4 completed trials recorded their best iteration after the first pruning checkpoint (iteration 25), so later gains are possible. Pruned curves end at the pruning checkpoint; their counterfactual later values are unknown. This is consistent with a possible early-pruning risk, not proof that a slower configuration was wrongly discarded.
- Budget reconstruction: the saved resume log shows 1933.1s remaining from the 2,430s tuning allowance, implying about 496.9s in the first B attempt. The resumed stage manifest records 1938.816s wall time, so cumulative B-stage wall time is about 2435.7s (roughly 5.7s beyond the resumed tuning clock, with ±0.05s rounding uncertainty). This separates tuning from resume/cleanup overhead; B final fit was not reached (0s). The 2,430s tuning allowance was exhausted before 5/5 successes; do not interpret the broader 2,700s tuning-plus-fit envelope as extra tuning budget.
- The frozen runner requires at least 5 successful trials before final fit. If the budget ends after that minimum but before the requested 10 total trials, the current acceptance gate can use the best successful trial while recording `search_complete=false`; these are distinct outcomes. This B study stopped at 4/5, so it correctly produced no final model or holdout score.
- The separately authorized development-only continuation diagnostic completed 15 fixed configurations across the three development B folds, without TPE suggestions, feature selection, generator optimization, or a holdout score. It records full available validation curves, both source and continued checkpoint values, and same-fold COMPLETE-study comparisons. No hypothetical MedianPruner decision or false-prune rate is claimed; see `reports/btc_preopen/b_pruner_development_diagnostic_20261014/pruner_diagnostic_report.md`. The diagnostic did not change model or policy selection.
- Six of 15 continuations found their best iteration after the corresponding source trial's prune checkpoint; only one of those six beat its same-fold COMPLETE-study best. Development fold C / Q1 (source trial 57) reached 0.692240 at iteration 73, 0.000031 below the best COMPLETE objective. This is limited evidence that a configuration from a historically PRUNED final-fit-B trial can be competitive on a development fold. It is not evidence that a specific development-fold MedianPruner decision was wrong: the source decision came from another chronological fit, and no hypothetical pruning was replayed. The other 14 continuations did not beat their same-fold best.

| Config | Source trial | Pruned at iteration | Selection rule | Frozen params JSON |
|---|---|---|---|---|
| B_PRUNED_LR_Q1 | 57 | 25 | median learning_rate-ranked PRUNED trial in quintile; ties by trial_number | {"feature_fraction": 0.7584827895375871, "lambda_l1": 2.682842333496145, "lambda_l2": 0.3760227836163543, "learning_rate": 0.026590164242171966, "min_data_in_leaf": 3092.0, "num_leaves": 155.0} |
| B_PRUNED_LR_Q2 | 9 | 50 | median learning_rate-ranked PRUNED trial in quintile; ties by trial_number | {"feature_fraction": 0.8470010931970855, "lambda_l1": 0.49632356271715267, "lambda_l2": 2.651423241611079, "learning_rate": 0.03613682679309368, "min_data_in_leaf": 1083.0, "num_leaves": 117.0} |
| B_PRUNED_LR_Q3 | 44 | 50 | median learning_rate-ranked PRUNED trial in quintile; ties by trial_number | {"feature_fraction": 0.9275539494242215, "lambda_l1": 3.1105231969259206, "lambda_l2": 0.23624620695083579, "learning_rate": 0.04521244910698924, "min_data_in_leaf": 14373.0, "num_leaves": 183.0} |
| B_PRUNED_LR_Q4 | 42 | 50 | median learning_rate-ranked PRUNED trial in quintile; ties by trial_number | {"feature_fraction": 0.8581576996128096, "lambda_l1": 7.3897905538404, "lambda_l2": 2.5732930185481844, "learning_rate": 0.057655482491433624, "min_data_in_leaf": 8909.0, "num_leaves": 150.0} |
| B_PRUNED_LR_Q5 | 23 | 50 | median learning_rate-ranked PRUNED trial in quintile; ties by trial_number | {"feature_fraction": 0.9123575896347449, "lambda_l1": 13.686793961081149, "lambda_l2": 21.15846563027234, "learning_rate": 0.08107954002830947, "min_data_in_leaf": 3484.0, "num_leaves": 156.0} |

## Artifacts and reproduction

- Report tables: `reports/btc_preopen/btc_preopen_model_update_20261012_partial`.
- Detailed decision and portfolio ledgers: `data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/partial_report_ledgers` (local research data).
- Report provenance is recorded separately at `partial_report_ledgers/report_provenance.json`; the frozen config, original model-update manifest, and development freeze were read-only inputs.
- Rebuild only this report, without training or command-line arguments: `python build_btc_preopen_model_update_report.py`.
- Live remains off. No deployment refit, order, or policy change was made.
