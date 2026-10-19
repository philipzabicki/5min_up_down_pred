# RAW BASE70 EV–PnL gap attribution — 2026-10-19

> Development is exploratory; July–October holdout was exposed previously. This fixed analysis is retrospective and does not create independent confirmation.

## Scope and saved-ledger verification

Only the existing RAW BASE70 prediction, T−60 decision, T−59 execution, and trade-ledger artifacts were read. No model or forecast was regenerated; the book archive was not replayed. Development and holdout were analyzed separately with their original cutoffs. The saved fixed-$5, 1% net-EV policy was retained.

| period | all_fills | settled_fills | unresolved_fills | settled_turnover_usd | forecast_ev_on_settled_fills_usd | net_pnl_usd | pnl_minus_ev_usd |
|---|---|---|---|---|---|---|---|
| development | 1038 | 1038 | 0 | 5190.0000 | 279.6050 | 267.4394 | -12.1656 |
| holdout | 3738 | 3737 | 1 | 18685.0000 | 1242.6112 | -546.2172 | -1788.8284 |

## Decision and label availability funnel

| period | stage | markets | share_of_decisions |
|---|---|---|---|
| development | RAW BASE70 decisions | 20735 | 1.0000 |
| development | Binance proxy label available by cutoff | 20735 | 1.0000 |
| development | Official label available by cutoff | 20718 | 0.9992 |
| development | Both labels available by cutoff | 20718 | 0.9992 |
| development | At least one T-60 $5 side valuation | 8818 | 0.4253 |
| development | Valid T-60 p_market | 10148 | 0.4894 |
| development | Both labels plus valid p_market | 10144 | 0.4892 |
| development | Frozen policy buy intent | 1038 | 0.0501 |
| development | Policy intent with both labels and p_market | 988 | 0.0476 |
| development | Priceable UP/DOWN candidate sides | 14550 |  |
| development | Actual fills | 1038 | 0.0501 |
| development | Settled actual fills | 1038 | 0.0501 |
| development | Settled fills with both labels (decomposition subset) | 1038 | 0.0501 |
| holdout | RAW BASE70 decisions | 28224 | 1.0000 |
| holdout | Binance proxy label available by cutoff | 28224 | 1.0000 |
| holdout | Official label available by cutoff | 28220 | 0.9999 |
| holdout | Both labels available by cutoff | 28220 | 0.9999 |
| holdout | At least one T-60 $5 side valuation | 23113 | 0.8189 |
| holdout | Valid T-60 p_market | 24737 | 0.8765 |
| holdout | Both labels plus valid p_market | 24735 | 0.8764 |
| holdout | Frozen policy buy intent | 3738 | 0.1324 |
| holdout | Policy intent with both labels and p_market | 3737 | 0.1324 |
| holdout | Priceable UP/DOWN candidate sides | 41763 |  |
| holdout | Actual fills | 3738 | 0.1324 |
| holdout | Settled actual fills | 3737 | 0.1324 |
| holdout | Settled fills with both labels (decomposition subset) | 3737 | 0.1324 |

The UP and DOWN candidates are two related observations for one market. Candidate counts therefore include side rows, while unique market counts are reported separately; block bootstrap draws retain both sides of each market together.

## Exact settled-execution decomposition

For each matched settled execution: p and labels are oriented to the purchased side; s is the engine’s payout-unit net_shares; c is the ledger cash_debit_usd. The share fee is already removed from net_shares, and any cash fee is already included in cash_debit_usd. The diagnostic never subtracts fee_usd again. T−60 EV and execution terms use the saved ledger and decision values.

RP − E60 = A + B + C, where A = E59 − E60, B = (yB − p) × s59, and C = (yP − yB) × s59. B includes outcome realization noise; a negative B alone does not establish miscalibration.

| period | executions | sum_E60_usd | mean_E60_usd | sum_E59_usd | mean_E59_usd | sum_R_binance_usd | mean_R_binance_usd | sum_R_official_usd | mean_R_official_usd | sum_A_execution_usd | mean_A_execution_usd | sum_B_binance_residual_usd | mean_B_binance_residual_usd | sum_C_official_minus_proxy_usd | mean_C_official_minus_proxy_usd | sum_identity_gap_usd | mean_identity_gap_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| development | 1038 | 279.6050 | 0.2694 | 281.9753 | 0.2717 | 239.5107 | 0.2307 | 267.4394 | 0.2576 | 2.3703 | 0.0023 | -42.4646 | -0.0409 | 27.9287 | 0.0269 | -12.1656 | -0.0117 |
| holdout | 3737 | 1242.6112 | 0.3325 | 1218.0850 | 0.3260 | -307.4899 | -0.0823 | -546.2172 | -0.1462 | -24.5262 | -0.0066 | -1525.5749 | -0.4082 | -238.7272 | -0.0639 | -1788.8284 | -0.4787 |

Monthly, UP/DOWN, and month-by-side sums and per-execution means:

| grouping | period | utc_month | side | executions | sum_A_execution_usd | mean_A_execution_usd | sum_B_binance_residual_usd | mean_B_binance_residual_usd | sum_C_official_minus_proxy_usd | mean_C_official_minus_proxy_usd | sum_identity_gap_usd | mean_identity_gap_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| month | development | 2026-04 |  | 63 | 0.0000 | 0.0000 | -0.4943 | -0.0078 | -19.2874 | -0.3061 | -19.7817 | -0.3140 |
| month | development | 2026-05 |  | 447 | 3.1403 | 0.0070 | -162.4846 | -0.3635 | 48.1636 | 0.1077 | -111.1807 | -0.2487 |
| month | development | 2026-06 |  | 528 | -0.7700 | -0.0015 | 120.5142 | 0.2282 | -0.9474 | -0.0018 | 118.7968 | 0.2250 |
| side | development |  | down | 545 | 3.4038 | 0.0062 | -60.9782 | -0.1119 | -3.3626 | -0.0062 | -60.9370 | -0.1118 |
| side | development |  | up | 493 | -1.0335 | -0.0021 | 18.5136 | 0.0376 | 31.2914 | 0.0635 | 48.7714 | 0.0989 |
| month_side | development | 2026-04 | down | 49 | 0.0000 | 0.0000 | -43.0734 | -0.8790 | -0.1894 | -0.0039 | -43.2628 | -0.8829 |
| month_side | development | 2026-04 | up | 14 | 0.0000 | 0.0000 | 42.5792 | 3.0414 | -19.0980 | -1.3641 | 23.4811 | 1.6772 |
| month_side | development | 2026-05 | down | 323 | 3.3189 | 0.0103 | -59.7147 | -0.1849 | 8.4380 | 0.0261 | -47.9579 | -0.1485 |
| month_side | development | 2026-05 | up | 124 | -0.1785 | -0.0014 | -102.7698 | -0.8288 | 39.7256 | 0.3204 | -63.2228 | -0.5099 |
| month_side | development | 2026-06 | down | 173 | 0.0850 | 0.0005 | 41.8100 | 0.2417 | -11.6113 | -0.0671 | 30.2837 | 0.1751 |
| month_side | development | 2026-06 | up | 355 | -0.8550 | -0.0024 | 78.7042 | 0.2217 | 10.6639 | 0.0300 | 88.5131 | 0.2493 |
| month | holdout | 2026-07 |  | 1244 | -0.7698 | -0.0006 | -877.7393 | -0.7056 | -25.6163 | -0.0206 | -904.1254 | -0.7268 |
| month | holdout | 2026-08 |  | 1117 | -13.4977 | -0.0121 | -502.8712 | -0.4502 | -73.2623 | -0.0656 | -589.6311 | -0.5279 |
| month | holdout | 2026-09 |  | 1053 | -7.5562 | -0.0072 | 5.2903 | 0.0050 | -84.5663 | -0.0803 | -86.8322 | -0.0825 |
| month | holdout | 2026-10 |  | 323 | -2.7027 | -0.0084 | -150.2547 | -0.4652 | -55.2823 | -0.1712 | -208.2397 | -0.6447 |
| side | holdout |  | down | 1963 | -14.8901 | -0.0076 | -1113.7844 | -0.5674 | -39.6721 | -0.0202 | -1168.3466 | -0.5952 |
| side | holdout |  | up | 1774 | -9.6361 | -0.0054 | -411.7905 | -0.2321 | -199.0552 | -0.1122 | -620.4818 | -0.3498 |
| month_side | holdout | 2026-07 | down | 638 | -1.3189 | -0.0021 | -661.8580 | -1.0374 | -4.3748 | -0.0069 | -667.5516 | -1.0463 |
| month_side | holdout | 2026-07 | up | 606 | 0.5491 | 0.0009 | -215.8813 | -0.3562 | -21.2416 | -0.0351 | -236.5738 | -0.3904 |
| month_side | holdout | 2026-08 | down | 564 | -8.7053 | -0.0154 | -286.2006 | -0.5074 | -87.6575 | -0.1554 | -382.5634 | -0.6783 |
| month_side | holdout | 2026-08 | up | 553 | -4.7924 | -0.0087 | -216.6706 | -0.3918 | 14.3952 | 0.0260 | -207.0677 | -0.3744 |
| month_side | holdout | 2026-09 | down | 597 | -3.3860 | -0.0057 | -37.4430 | -0.0627 | 74.9330 | 0.1255 | 34.1040 | 0.0571 |
| month_side | holdout | 2026-09 | up | 456 | -4.1702 | -0.0091 | 42.7334 | 0.0937 | -159.4993 | -0.3498 | -120.9361 | -0.2652 |
| month_side | holdout | 2026-10 | down | 164 | -1.4800 | -0.0090 | -128.2827 | -0.7822 | -22.5728 | -0.1376 | -152.3355 | -0.9289 |
| month_side | holdout | 2026-10 | up | 159 | -1.2227 | -0.0077 | -21.9720 | -0.1382 | -32.7095 | -0.2057 | -55.9042 | -0.3516 |

Paired 3-day block intervals for A/B/C:

| period | component | sum_95pct | mean_95pct | valid_replications |
|---|---|---|---|---|
| development | A_execution_usd | [-1.789261189787555, 9.184711455326028] | [-0.0014861675527376701, 0.008519857783530737] | 2000 |
| development | B_binance_residual_usd | [-270.8809937165756, 200.67844067485004] | [-0.32637758300786734, 0.17347665903967668] | 2000 |
| development | C_official_minus_proxy_usd | [-92.46074611906498, 151.8753720852296] | [-0.09268192179081808, 0.16410698303050766] | 2000 |
| holdout | A_execution_usd | [-38.41115374253261, -12.139351333908131] | [-0.010170883981707133, -0.003274399991777103] | 2000 |
| holdout | B_binance_residual_usd | [-2518.1488863743275, -698.7157852688024] | [-0.6279203848002627, -0.19928427802321838] | 2000 |
| holdout | C_official_minus_proxy_usd | [-569.9207848614923, 71.49114553621297] | [-0.15152425421767196, 0.01959854848539749] | 2000 |

The subset-to-ledger reconciliation is in reconciliation.json.

## Candidate EV bins, policy intents, and actual fills

EV bins are the frozen net T−60 EV divided by the requested $5 gross stake. Candidate rows are valuations, not simulated trades. The policy-intent and actual-fill roles are shown separately. An actual-fill PnL includes only fills settled before cutoff; unresolved fills remain counted and separate.

| period | role | EV_bin | side | unique_markets | side_observations_or_intents_or_fills | fills | settled_fills | unresolved_fills | mean_p_side | mean_binance_win_rate | mean_official_win_rate | mean_p_break_even | mean_binance_error_y_minus_p | mean_official_error_y_minus_p | forecast_E60_usd_on_settled_fills | realized_official_net_pnl_usd_on_settled_fills |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| development | both_priceable_sides | [2%,4%) | up | 182 | 182 | 0 | 0 | 0 | 0.5282 | 0.5659 | 0.5549 | 0.5142 | 0.0378 | 0.0268 |  |  |
| development | both_priceable_sides | ≤0% | up | 6232 | 6232 | 0 | 0 | 0 | 0.5011 | 0.4923 | 0.4907 | 0.5284 | -0.0088 | -0.0104 |  |  |
| development | both_priceable_sides | (0%,1%) | up | 265 | 265 | 0 | 0 | 0 | 0.5233 | 0.5245 | 0.5396 | 0.5208 | 0.0013 | 0.0164 |  |  |
| development | both_priceable_sides | ≥4% | up | 141 | 141 | 0 | 0 | 0 | 0.5249 | 0.5248 | 0.5319 | 0.4712 | -0.0001 | 0.0070 |  |  |
| development | both_priceable_sides | [1%,2%) | up | 172 | 172 | 0 | 0 | 0 | 0.5254 | 0.5058 | 0.5291 | 0.5182 | -0.0195 | 0.0037 |  |  |
| development | both_priceable_sides | ≤0% | down | 6663 | 6663 | 0 | 0 | 0 | 0.4929 | 0.5032 | 0.5030 | 0.5199 | 0.0103 | 0.0101 |  |  |
| development | both_priceable_sides | (0%,1%) | down | 343 | 343 | 0 | 0 | 0 | 0.5137 | 0.5073 | 0.5102 | 0.5116 | -0.0064 | -0.0035 |  |  |
| development | both_priceable_sides | [2%,4%) | down | 195 | 195 | 0 | 0 | 0 | 0.5199 | 0.4872 | 0.5077 | 0.5064 | -0.0327 | -0.0122 |  |  |
| development | both_priceable_sides | [1%,2%) | down | 203 | 203 | 0 | 0 | 0 | 0.5169 | 0.5517 | 0.5567 | 0.5097 | 0.0348 | 0.0398 |  |  |
| development | both_priceable_sides | ≥4% | down | 154 | 154 | 0 | 0 | 0 | 0.5158 | 0.4805 | 0.4545 | 0.4623 | -0.0353 | -0.0613 |  |  |
| development | frozen_policy_intent | [2%,4%) | up | 181 | 181 | 0 | 0 | 0 | 0.5282 | 0.5635 | 0.5525 | 0.5142 | 0.0353 | 0.0243 |  |  |
| development | frozen_policy_intent | ≥4% | up | 141 | 141 | 0 | 0 | 0 | 0.5249 | 0.5248 | 0.5319 | 0.4712 | -0.0001 | 0.0070 |  |  |
| development | frozen_policy_intent | [1%,2%) | up | 171 | 171 | 0 | 0 | 0 | 0.5253 | 0.5088 | 0.5322 | 0.5181 | -0.0165 | 0.0069 |  |  |
| development | frozen_policy_intent | [2%,4%) | down | 193 | 193 | 0 | 0 | 0 | 0.5199 | 0.4870 | 0.5078 | 0.5064 | -0.0329 | -0.0121 |  |  |
| development | frozen_policy_intent | [1%,2%) | down | 202 | 202 | 0 | 0 | 0 | 0.5170 | 0.5495 | 0.5545 | 0.5099 | 0.0325 | 0.0374 |  |  |
| development | frozen_policy_intent | ≥4% | down | 150 | 150 | 0 | 0 | 0 | 0.5159 | 0.4800 | 0.4533 | 0.4625 | -0.0359 | -0.0626 |  |  |
| development | actual_fill | [2%,4%) | up | 181 | 181 | 181 | 181 | 0 | 0.5282 | 0.5635 | 0.5525 | 0.5145 | 0.0353 | 0.0243 | 25.4935 | 70.5171 |
| development | actual_fill | [2%,4%) | down | 193 | 193 | 193 | 193 | 0 | 0.5199 | 0.4870 | 0.5078 | 0.5066 | -0.0329 | -0.0121 | 26.6396 | 1.6785 |
| development | actual_fill | [1%,2%) | down | 202 | 202 | 202 | 202 | 0 | 0.5170 | 0.5495 | 0.5545 | 0.5092 | 0.0325 | 0.0374 | 14.6330 | 91.1097 |
| development | actual_fill | ≥4% | down | 150 | 150 | 150 | 150 | 0 | 0.5159 | 0.4800 | 0.4533 | 0.4619 | -0.0359 | -0.0626 | 103.6931 | -8.7594 |
| development | actual_fill | ≥4% | up | 141 | 141 | 141 | 141 | 0 | 0.5249 | 0.5248 | 0.5319 | 0.4714 | -0.0001 | 0.0070 | 96.9111 | 93.4292 |
| development | actual_fill | [1%,2%) | up | 171 | 171 | 171 | 171 | 0 | 0.5253 | 0.5088 | 0.5322 | 0.5183 | -0.0165 | 0.0069 | 12.2347 | 19.4643 |
| holdout | both_priceable_sides | ≤0% | up | 17704 | 17704 | 0 | 0 | 0 | 0.4970 | 0.4969 | 0.4975 | 0.5266 | -0.0001 | 0.0006 |  |  |
| holdout | both_priceable_sides | [2%,4%) | up | 719 | 719 | 0 | 0 | 0 | 0.5313 | 0.5243 | 0.5229 | 0.5172 | -0.0070 | -0.0084 |  |  |
| holdout | both_priceable_sides | [1%,2%) | up | 651 | 651 | 0 | 0 | 0 | 0.5276 | 0.5284 | 0.5223 | 0.5202 | 0.0009 | -0.0053 |  |  |
| holdout | both_priceable_sides | ≥4% | up | 608 | 608 | 0 | 0 | 0 | 0.5182 | 0.4836 | 0.4728 | 0.4598 | -0.0347 | -0.0455 |  |  |
| holdout | both_priceable_sides | (0%,1%) | up | 932 | 932 | 0 | 0 | 0 | 0.5241 | 0.5000 | 0.5075 | 0.5217 | -0.0241 | -0.0166 |  |  |
| holdout | both_priceable_sides | [2%,4%) | down | 678 | 678 | 0 | 0 | 0 | 0.5304 | 0.4941 | 0.5015 | 0.5160 | -0.0363 | -0.0290 |  |  |
| holdout | both_priceable_sides | (0%,1%) | down | 864 | 864 | 0 | 0 | 0 | 0.5230 | 0.5104 | 0.5093 | 0.5206 | -0.0126 | -0.0137 |  |  |
| holdout | both_priceable_sides | [1%,2%) | down | 643 | 643 | 0 | 0 | 0 | 0.5266 | 0.5194 | 0.5054 | 0.5192 | -0.0072 | -0.0212 |  |  |
| holdout | both_priceable_sides | ≥4% | down | 704 | 704 | 0 | 0 | 0 | 0.5210 | 0.4233 | 0.4276 | 0.4646 | -0.0977 | -0.0934 |  |  |
| holdout | both_priceable_sides | ≤0% | down | 18260 | 18260 | 0 | 0 | 0 | 0.4942 | 0.5037 | 0.5043 | 0.5244 | 0.0095 | 0.0101 |  |  |
| holdout | frozen_policy_intent | [2%,4%) | up | 638 | 638 | 0 | 0 | 0 | 0.5304 | 0.5125 | 0.5110 | 0.5164 | -0.0179 | -0.0195 |  |  |
| holdout | frozen_policy_intent | [1%,2%) | up | 582 | 582 | 0 | 0 | 0 | 0.5273 | 0.5292 | 0.5172 | 0.5200 | 0.0019 | -0.0101 |  |  |
| holdout | frozen_policy_intent | ≥4% | up | 555 | 555 | 0 | 0 | 0 | 0.5167 | 0.4793 | 0.4639 | 0.4559 | -0.0374 | -0.0528 |  |  |
| holdout | frozen_policy_intent | [2%,4%) | down | 654 | 654 | 0 | 0 | 0 | 0.5304 | 0.4908 | 0.4985 | 0.5160 | -0.0396 | -0.0320 |  |  |
| holdout | frozen_policy_intent | [1%,2%) | down | 627 | 627 | 0 | 0 | 0 | 0.5264 | 0.5199 | 0.5056 | 0.5190 | -0.0065 | -0.0208 |  |  |
| holdout | frozen_policy_intent | ≥4% | down | 682 | 682 | 0 | 0 | 0 | 0.5209 | 0.4238 | 0.4296 | 0.4643 | -0.0972 | -0.0913 |  |  |
| holdout | actual_fill | [2%,4%) | down | 654 | 654 | 654 | 654 | 0 | 0.5304 | 0.4908 | 0.4985 | 0.5165 | -0.0396 | -0.0320 | 94.6894 | -112.4356 |
| holdout | actual_fill | [1%,2%) | down | 627 | 627 | 627 | 627 | 0 | 0.5264 | 0.5199 | 0.5056 | 0.5196 | -0.0065 | -0.0208 | 46.4157 | -78.5683 |
| holdout | actual_fill | ≥4% | down | 682 | 682 | 682 | 682 | 0 | 0.5209 | 0.4238 | 0.4296 | 0.4652 | -0.0972 | -0.0913 | 510.6705 | -325.5671 |
| holdout | actual_fill | [2%,4%) | up | 638 | 638 | 638 | 638 | 0 | 0.5304 | 0.5125 | 0.5110 | 0.5169 | -0.0179 | -0.0195 | 90.1649 | -30.8559 |
| holdout | actual_fill | [1%,2%) | up | 582 | 582 | 582 | 582 | 0 | 0.5273 | 0.5292 | 0.5172 | 0.5202 | 0.0019 | -0.0101 | 42.3902 | -21.9571 |
| holdout | actual_fill | ≥4% | up | 555 | 555 | 555 | 554 | 1 | 0.5167 | 0.4793 | 0.4639 | 0.4566 | -0.0374 | -0.0528 | 458.2806 | 23.1668 |

Paired 3-day block intervals for candidate, policy-selected, and actual-fill y−p by EV bin:

| period | role | EV_bin | side | target | side_observations | mean_y_minus_p | mean_95pct | valid_replications | weakly_estimated |
|---|---|---|---|---|---|---|---|---|---|
| development | both_priceable_sides | [2%,4%) | up | binance_proxy | 182 | 0.0378 | [-0.012915746692698613, 0.09549963397639284] | 2000 | False |
| development | both_priceable_sides | [2%,4%) | up | official | 182 | 0.0268 | [-0.037535981564445695, 0.08034521181693151] | 2000 | False |
| development | both_priceable_sides | ≤0% | up | binance_proxy | 6232 | -0.0088 | [-0.02661039938997282, 0.009794972349920685] | 2000 | False |
| development | both_priceable_sides | ≤0% | up | official | 6230 | -0.0104 | [-0.029206677866711734, 0.009165185930971645] | 2000 | False |
| development | both_priceable_sides | (0%,1%) | up | binance_proxy | 265 | 0.0013 | [-0.05107131750301595, 0.06225483852895972] | 2000 | False |
| development | both_priceable_sides | (0%,1%) | up | official | 265 | 0.0164 | [-0.03639006016254365, 0.07780493373470927] | 2000 | False |
| development | both_priceable_sides | ≥4% | up | binance_proxy | 141 | -0.0001 | [-0.1012984732702627, 0.06619958810490637] | 2000 | False |
| development | both_priceable_sides | ≥4% | up | official | 141 | 0.0070 | [-0.08039418001235886, 0.07194741898192539] | 2000 | False |
| development | both_priceable_sides | [1%,2%) | up | binance_proxy | 172 | -0.0195 | [-0.13233637087182742, 0.07451064578567942] | 2000 | False |
| development | both_priceable_sides | [1%,2%) | up | official | 172 | 0.0037 | [-0.10005091707826706, 0.09923828260202343] | 2000 | False |
| development | both_priceable_sides | ≤0% | down | binance_proxy | 6663 | 0.0103 | [-0.004109067051662898, 0.023066660986997296] | 2000 | False |
| development | both_priceable_sides | ≤0% | down | official | 6660 | 0.0101 | [-0.005449035603580923, 0.0236985387923144] | 2000 | False |
| development | both_priceable_sides | (0%,1%) | down | binance_proxy | 343 | -0.0064 | [-0.06229730703079249, 0.052087099723643185] | 2000 | False |
| development | both_priceable_sides | (0%,1%) | down | official | 343 | -0.0035 | [-0.054144361308401556, 0.05085468358515077] | 2000 | False |
| development | both_priceable_sides | [2%,4%) | down | binance_proxy | 195 | -0.0327 | [-0.09683969142608324, 0.02053237276305401] | 2000 | False |
| development | both_priceable_sides | [2%,4%) | down | official | 195 | -0.0122 | [-0.08047721121972536, 0.05047792270933226] | 2000 | False |
| development | both_priceable_sides | [1%,2%) | down | binance_proxy | 203 | 0.0348 | [-0.04505039410734618, 0.1201232404733028] | 2000 | False |
| development | both_priceable_sides | [1%,2%) | down | official | 203 | 0.0398 | [-0.044838700572982795, 0.13280643906138365] | 2000 | False |
| development | both_priceable_sides | ≥4% | down | binance_proxy | 154 | -0.0353 | [-0.1319537429818692, 0.04735165753197681] | 2000 | False |
| development | both_priceable_sides | ≥4% | down | official | 154 | -0.0613 | [-0.14756331815474646, 0.01538483934520287] | 2000 | False |
| development | frozen_policy_intent | [2%,4%) | up | binance_proxy | 181 | 0.0353 | [-0.014161119516089695, 0.09384859258474691] | 2000 | False |
| development | frozen_policy_intent | [2%,4%) | up | official | 181 | 0.0243 | [-0.038003586567312925, 0.0768330488927405] | 2000 | False |
| development | frozen_policy_intent | ≥4% | up | binance_proxy | 141 | -0.0001 | [-0.1012984732702627, 0.06619958810490637] | 2000 | False |
| development | frozen_policy_intent | ≥4% | up | official | 141 | 0.0070 | [-0.08039418001235886, 0.07194741898192539] | 2000 | False |
| development | frozen_policy_intent | [1%,2%) | up | binance_proxy | 171 | -0.0165 | [-0.12645750139145226, 0.07783975552058321] | 2000 | False |
| development | frozen_policy_intent | [1%,2%) | up | official | 171 | 0.0069 | [-0.09602446993466095, 0.1043379716117165] | 2000 | False |
| development | frozen_policy_intent | [2%,4%) | down | binance_proxy | 193 | -0.0329 | [-0.09633244175527363, 0.02176654961273591] | 2000 | False |
| development | frozen_policy_intent | [2%,4%) | down | official | 193 | -0.0121 | [-0.08056586708957333, 0.0509504506628996] | 2000 | False |
| development | frozen_policy_intent | [1%,2%) | down | binance_proxy | 202 | 0.0325 | [-0.047774941712514396, 0.11775307151353583] | 2000 | False |
| development | frozen_policy_intent | [1%,2%) | down | official | 202 | 0.0374 | [-0.04706740263054549, 0.13008913670673602] | 2000 | False |
| development | frozen_policy_intent | ≥4% | down | binance_proxy | 150 | -0.0359 | [-0.12898595486575895, 0.045682867973206184] | 2000 | False |
| development | frozen_policy_intent | ≥4% | down | official | 150 | -0.0626 | [-0.14653552178541818, 0.01232645570809077] | 2000 | False |
| development | actual_fill | [2%,4%) | up | binance_proxy | 181 | 0.0353 | [-0.014161119516089695, 0.09384859258474691] | 2000 | False |
| development | actual_fill | [2%,4%) | up | official | 181 | 0.0243 | [-0.038003586567312925, 0.0768330488927405] | 2000 | False |
| development | actual_fill | [2%,4%) | down | binance_proxy | 193 | -0.0329 | [-0.09633244175527363, 0.02176654961273591] | 2000 | False |
| development | actual_fill | [2%,4%) | down | official | 193 | -0.0121 | [-0.08056586708957333, 0.0509504506628996] | 2000 | False |
| development | actual_fill | [1%,2%) | down | binance_proxy | 202 | 0.0325 | [-0.047774941712514396, 0.11775307151353583] | 2000 | False |
| development | actual_fill | [1%,2%) | down | official | 202 | 0.0374 | [-0.04706740263054549, 0.13008913670673602] | 2000 | False |
| development | actual_fill | ≥4% | down | binance_proxy | 150 | -0.0359 | [-0.12898595486575895, 0.045682867973206184] | 2000 | False |
| development | actual_fill | ≥4% | down | official | 150 | -0.0626 | [-0.14653552178541818, 0.01232645570809077] | 2000 | False |
| development | actual_fill | ≥4% | up | binance_proxy | 141 | -0.0001 | [-0.1012984732702627, 0.06619958810490637] | 2000 | False |
| development | actual_fill | ≥4% | up | official | 141 | 0.0070 | [-0.08039418001235886, 0.07194741898192539] | 2000 | False |
| development | actual_fill | [1%,2%) | up | binance_proxy | 171 | -0.0165 | [-0.12645750139145226, 0.07783975552058321] | 2000 | False |
| development | actual_fill | [1%,2%) | up | official | 171 | 0.0069 | [-0.09602446993466095, 0.1043379716117165] | 2000 | False |
| holdout | both_priceable_sides | ≤0% | up | binance_proxy | 17704 | -0.0001 | [-0.007530654825937078, 0.007755474100003753] | 2000 | False |
| holdout | both_priceable_sides | ≤0% | up | official | 17703 | 0.0006 | [-0.006571886043761915, 0.007584113614612712] | 2000 | False |
| holdout | both_priceable_sides | [2%,4%) | up | binance_proxy | 719 | -0.0070 | [-0.041815028177996906, 0.02577044358099194] | 2000 | False |
| holdout | both_priceable_sides | [2%,4%) | up | official | 719 | -0.0084 | [-0.04193078955438153, 0.02266734922779022] | 2000 | False |
| holdout | both_priceable_sides | [1%,2%) | up | binance_proxy | 651 | 0.0009 | [-0.03739017485867537, 0.03987686669724775] | 2000 | False |
| holdout | both_priceable_sides | [1%,2%) | up | official | 651 | -0.0053 | [-0.04225173258552054, 0.035058928056852505] | 2000 | False |
| holdout | both_priceable_sides | ≥4% | up | binance_proxy | 608 | -0.0347 | [-0.07725757105934622, 0.009200942194464556] | 2000 | False |
| holdout | both_priceable_sides | ≥4% | up | official | 607 | -0.0455 | [-0.09000773952214769, 0.001554552926000381] | 2000 | False |
| holdout | both_priceable_sides | (0%,1%) | up | binance_proxy | 932 | -0.0241 | [-0.05164237014023039, 0.004327407636154938] | 2000 | False |
| holdout | both_priceable_sides | (0%,1%) | up | official | 932 | -0.0166 | [-0.049953856399691424, 0.019014844708565043] | 2000 | False |
| holdout | both_priceable_sides | [2%,4%) | down | binance_proxy | 678 | -0.0363 | [-0.07993510687444685, 0.013148997943127403] | 2000 | False |
| holdout | both_priceable_sides | [2%,4%) | down | official | 678 | -0.0290 | [-0.07293102621557038, 0.01930083599718978] | 2000 | False |
| holdout | both_priceable_sides | (0%,1%) | down | binance_proxy | 864 | -0.0126 | [-0.03940419127226295, 0.012998619116690383] | 2000 | False |
| holdout | both_priceable_sides | (0%,1%) | down | official | 864 | -0.0137 | [-0.041909211477158544, 0.012025721479076067] | 2000 | False |
| holdout | both_priceable_sides | [1%,2%) | down | binance_proxy | 643 | -0.0072 | [-0.044125108552652795, 0.0273716670504959] | 2000 | False |
| holdout | both_priceable_sides | [1%,2%) | down | official | 643 | -0.0212 | [-0.05856121616202485, 0.01651524396594586] | 2000 | False |
| holdout | both_priceable_sides | ≥4% | down | binance_proxy | 704 | -0.0977 | [-0.13839254204220355, -0.05566822592419509] | 2000 | False |
| holdout | both_priceable_sides | ≥4% | down | official | 704 | -0.0934 | [-0.1358618964142581, -0.04784751093425454] | 2000 | False |
| holdout | both_priceable_sides | ≤0% | down | binance_proxy | 18260 | 0.0095 | [0.00217720601456059, 0.01665701837482853] | 2000 | False |
| holdout | both_priceable_sides | ≤0% | down | official | 18258 | 0.0101 | [0.0033861227744824817, 0.01685167494335576] | 2000 | False |
| holdout | frozen_policy_intent | [2%,4%) | up | binance_proxy | 638 | -0.0179 | [-0.0607404135578531, 0.021746661379958298] | 2000 | False |
| holdout | frozen_policy_intent | [2%,4%) | up | official | 638 | -0.0195 | [-0.0601172892114749, 0.0179347719004498] | 2000 | False |
| holdout | frozen_policy_intent | [1%,2%) | up | binance_proxy | 582 | 0.0019 | [-0.038717745092493004, 0.0435131796047725] | 2000 | False |
| holdout | frozen_policy_intent | [1%,2%) | up | official | 582 | -0.0101 | [-0.04848440671060997, 0.03181741113226757] | 2000 | False |
| holdout | frozen_policy_intent | ≥4% | up | binance_proxy | 555 | -0.0374 | [-0.08417692368245237, 0.009921851896190121] | 2000 | False |
| holdout | frozen_policy_intent | ≥4% | up | official | 554 | -0.0528 | [-0.09820321770220074, -0.0034408653668033693] | 2000 | False |
| holdout | frozen_policy_intent | [2%,4%) | down | binance_proxy | 654 | -0.0396 | [-0.08484308638835832, 0.010488192234237253] | 2000 | False |
| holdout | frozen_policy_intent | [2%,4%) | down | official | 654 | -0.0320 | [-0.07689650959380223, 0.016220279769939405] | 2000 | False |
| holdout | frozen_policy_intent | [1%,2%) | down | binance_proxy | 627 | -0.0065 | [-0.044176916701574316, 0.029327151227658853] | 2000 | False |
| holdout | frozen_policy_intent | [1%,2%) | down | official | 627 | -0.0208 | [-0.05909974960071975, 0.017740238718191602] | 2000 | False |
| holdout | frozen_policy_intent | ≥4% | down | binance_proxy | 682 | -0.0972 | [-0.1378433693256283, -0.05464751286666812] | 2000 | False |
| holdout | frozen_policy_intent | ≥4% | down | official | 682 | -0.0913 | [-0.13465111709261723, -0.04511320852956836] | 2000 | False |
| holdout | actual_fill | [2%,4%) | down | binance_proxy | 654 | -0.0396 | [-0.08484308638835832, 0.010488192234237253] | 2000 | False |
| holdout | actual_fill | [2%,4%) | down | official | 654 | -0.0320 | [-0.07689650959380223, 0.016220279769939405] | 2000 | False |
| holdout | actual_fill | [1%,2%) | down | binance_proxy | 627 | -0.0065 | [-0.044176916701574316, 0.029327151227658853] | 2000 | False |
| holdout | actual_fill | [1%,2%) | down | official | 627 | -0.0208 | [-0.05909974960071975, 0.017740238718191602] | 2000 | False |
| holdout | actual_fill | ≥4% | down | binance_proxy | 682 | -0.0972 | [-0.1378433693256283, -0.05464751286666812] | 2000 | False |
| holdout | actual_fill | ≥4% | down | official | 682 | -0.0913 | [-0.13465111709261723, -0.04511320852956836] | 2000 | False |
| holdout | actual_fill | [2%,4%) | up | binance_proxy | 638 | -0.0179 | [-0.0607404135578531, 0.021746661379958298] | 2000 | False |
| holdout | actual_fill | [2%,4%) | up | official | 638 | -0.0195 | [-0.0601172892114749, 0.0179347719004498] | 2000 | False |
| holdout | actual_fill | [1%,2%) | up | binance_proxy | 582 | 0.0019 | [-0.038717745092493004, 0.0435131796047725] | 2000 | False |
| holdout | actual_fill | [1%,2%) | up | official | 582 | -0.0101 | [-0.04848440671060997, 0.03181741113226757] | 2000 | False |
| holdout | actual_fill | ≥4% | up | binance_proxy | 555 | -0.0374 | [-0.08417692368245237, 0.009921851896190121] | 2000 | False |
| holdout | actual_fill | ≥4% | up | official | 554 | -0.0528 | [-0.09820321770220074, -0.0034408653668033693] | 2000 | False |

Groups with fewer than 50 side observations are marked weakly estimated. Proxy/official disagreements within each EV bin are reported by unique market in target_disagreement.csv.

## BASE70 versus T−60 market forecast

p_market uses the existing normalized midpoint definition and is used only as a forecast benchmark; it is not substituted for an ask or execution price. The common sample contains unique markets with both cutoff-available labels and valid p_market. Selected-side results are conditional on BASE70’s policy selection.

| period | sample_role | target | forecast | markets | positive_rate | log_loss | brier | auc |
|---|---|---|---|---|---|---|---|---|
| development | all_valid_T60_markets | binance_proxy | RAW_BASE70 | 10144 | 0.4983 | 0.6922 | 0.2495 | 0.5252 |
| development | all_valid_T60_markets | binance_proxy | p_market | 10144 | 0.4983 | 0.6930 | 0.2499 | 0.5134 |
| development | all_valid_T60_markets | binance_proxy | p=0.5 | 10144 | 0.4983 | 0.6931 | 0.2500 | 0.5000 |
| development | all_valid_T60_markets | official | RAW_BASE70 | 10144 | 0.4973 | 0.6921 | 0.2495 | 0.5274 |
| development | all_valid_T60_markets | official | p_market | 10144 | 0.4973 | 0.6929 | 0.2499 | 0.5138 |
| development | all_valid_T60_markets | official | p=0.5 | 10144 | 0.4973 | 0.6931 | 0.2500 | 0.5000 |
| holdout | all_valid_T60_markets | binance_proxy | RAW_BASE70 | 24735 | 0.4978 | 0.6924 | 0.2496 | 0.5230 |
| holdout | all_valid_T60_markets | binance_proxy | p_market | 24735 | 0.4978 | 0.6919 | 0.2494 | 0.5200 |
| holdout | all_valid_T60_markets | binance_proxy | p=0.5 | 24735 | 0.4978 | 0.6931 | 0.2500 | 0.5000 |
| holdout | all_valid_T60_markets | official | RAW_BASE70 | 24735 | 0.4983 | 0.6925 | 0.2497 | 0.5225 |
| holdout | all_valid_T60_markets | official | p_market | 24735 | 0.4983 | 0.6916 | 0.2492 | 0.5201 |
| holdout | all_valid_T60_markets | official | p=0.5 | 24735 | 0.4983 | 0.6931 | 0.2500 | 0.5000 |

Conditional point metrics on BASE70-selected sides and the actual settled fills (all three forecasts):

| period | sample_role | target | forecast | markets | positive_rate | log_loss | brier | auc |
|---|---|---|---|---|---|---|---|---|
| development | frozen_policy_selected_side | binance_proxy | RAW_BASE70 | 988 | 0.5283 | 0.6889 | 0.2479 | 0.5493 |
| development | frozen_policy_selected_side | binance_proxy | p_market | 988 | 0.5283 | 0.6945 | 0.2506 | 0.5500 |
| development | frozen_policy_selected_side | binance_proxy | p=0.5 | 988 | 0.5283 | 0.6931 | 0.2500 | 0.5000 |
| development | frozen_policy_selected_side | official | RAW_BASE70 | 988 | 0.5314 | 0.6885 | 0.2477 | 0.5492 |
| development | frozen_policy_selected_side | official | p_market | 988 | 0.5314 | 0.6929 | 0.2499 | 0.5536 |
| development | frozen_policy_selected_side | official | p=0.5 | 988 | 0.5314 | 0.6931 | 0.2500 | 0.5000 |
| development | actually_filled_and_settled_side | binance_proxy | RAW_BASE70 | 988 | 0.5283 | 0.6889 | 0.2479 | 0.5493 |
| development | actually_filled_and_settled_side | binance_proxy | p_market | 988 | 0.5283 | 0.6945 | 0.2506 | 0.5500 |
| development | actually_filled_and_settled_side | binance_proxy | p=0.5 | 988 | 0.5283 | 0.6931 | 0.2500 | 0.5000 |
| development | actually_filled_and_settled_side | official | RAW_BASE70 | 988 | 0.5314 | 0.6885 | 0.2477 | 0.5492 |
| development | actually_filled_and_settled_side | official | p_market | 988 | 0.5314 | 0.6929 | 0.2499 | 0.5536 |
| development | actually_filled_and_settled_side | official | p=0.5 | 988 | 0.5314 | 0.6931 | 0.2500 | 0.5000 |
| holdout | frozen_policy_selected_side | binance_proxy | RAW_BASE70 | 3737 | 0.4916 | 0.6938 | 0.2503 | 0.5304 |
| holdout | frozen_policy_selected_side | binance_proxy | p_market | 3737 | 0.4916 | 0.6882 | 0.2475 | 0.5455 |
| holdout | frozen_policy_selected_side | binance_proxy | p=0.5 | 3737 | 0.4916 | 0.6931 | 0.2500 | 0.5000 |
| holdout | frozen_policy_selected_side | official | RAW_BASE70 | 3737 | 0.4870 | 0.6939 | 0.2504 | 0.5347 |
| holdout | frozen_policy_selected_side | official | p_market | 3737 | 0.4870 | 0.6858 | 0.2464 | 0.5483 |
| holdout | frozen_policy_selected_side | official | p=0.5 | 3737 | 0.4870 | 0.6931 | 0.2500 | 0.5000 |
| holdout | actually_filled_and_settled_side | binance_proxy | RAW_BASE70 | 3737 | 0.4916 | 0.6938 | 0.2503 | 0.5304 |
| holdout | actually_filled_and_settled_side | binance_proxy | p_market | 3737 | 0.4916 | 0.6882 | 0.2475 | 0.5455 |
| holdout | actually_filled_and_settled_side | binance_proxy | p=0.5 | 3737 | 0.4916 | 0.6931 | 0.2500 | 0.5000 |
| holdout | actually_filled_and_settled_side | official | RAW_BASE70 | 3737 | 0.4870 | 0.6939 | 0.2504 | 0.5347 |
| holdout | actually_filled_and_settled_side | official | p_market | 3737 | 0.4870 | 0.6858 | 0.2464 | 0.5483 |
| holdout | actually_filled_and_settled_side | official | p=0.5 | 3737 | 0.4870 | 0.6931 | 0.2500 | 0.5000 |

Paired 3-day block intervals for RAW BASE70 minus p_market on the common market sample:

| period | target | metric | point_difference | 95pct | valid_replications |
|---|---|---|---|---|---|
| development | binance_proxy | log_loss | -0.0008 | [-0.0014959600002752843, -0.00010439851705296163] | 2000 |
| development | binance_proxy | brier | -0.0004 | [-0.0007339400596174612, -5.286897872682776e-05] | 2000 |
| development | binance_proxy | auc | 0.0117 | [-0.00025571769583527235, 0.025700583280924953] | 2000 |
| development | official | log_loss | -0.0008 | [-0.0014731704442063564, -0.00014748901359503216] | 2000 |
| development | official | brier | -0.0004 | [-0.0007338273684582092, -7.211002314343338e-05] | 2000 |
| development | official | auc | 0.0136 | [0.0010910665965921497, 0.028976735905406316] | 2000 |
| holdout | binance_proxy | log_loss | 0.0006 | [-0.0003594356586860109, 0.0017593990008285762] | 2000 |
| holdout | binance_proxy | brier | 0.0003 | [-0.00017533381504826056, 0.000860288903109037] | 2000 |
| holdout | binance_proxy | auc | 0.0031 | [-0.004286064083262172, 0.01031542495681564] | 2000 |
| holdout | official | log_loss | 0.0009 | [-8.293160197534732e-05, 0.002091310039418945] | 2000 |
| holdout | official | brier | 0.0004 | [-4.523368152511697e-05, 0.0010145618278970196] | 2000 |
| holdout | official | auc | 0.0024 | [-0.004735464221150676, 0.009407792144144625] | 2000 |

Negative log-loss/Brier differences and positive AUC differences favor BASE70.

## Proxy versus official target; costs

| period | scope | markets_with_both_labels | proxy_up_official_down | proxy_down_official_up | total_disagreements | disagreement_rate |
|---|---|---|---|---|---|---|
| development | all_common_labeled_markets | 20718 | 322 | 329 | 651 | 0.0314 |
| development | at_least_one_priceable_T60_side | 8814 | 151 | 141 | 292 | 0.0331 |
| development | frozen_policy_buy_intents | 1038 | 19 | 21 | 40 | 0.0385 |
| development | all_actual_fills | 1038 | 19 | 21 | 40 | 0.0385 |
| development | settled_actual_fills | 1038 | 19 | 21 | 40 | 0.0385 |
| development | candidate_EV_[2%,4%) | 377 | 9 | 3 | 12 | 0.0318 |
| development | candidate_EV_≤0% | 8494 | 142 | 136 | 278 | 0.0327 |
| development | candidate_EV_(0%,1%) | 608 | 7 | 10 | 17 | 0.0280 |
| development | candidate_EV_≥4% | 295 | 3 | 8 | 11 | 0.0373 |
| development | candidate_EV_[1%,2%) | 375 | 7 | 10 | 17 | 0.0453 |
| holdout | all_common_labeled_markets | 28220 | 1090 | 1123 | 2213 | 0.0784 |
| holdout | at_least_one_priceable_T60_side | 23111 | 949 | 955 | 1904 | 0.0824 |
| holdout | frozen_policy_buy_intents | 3737 | 155 | 138 | 293 | 0.0784 |
| holdout | all_actual_fills | 3737 | 155 | 138 | 293 | 0.0784 |
| holdout | settled_actual_fills | 3737 | 155 | 138 | 293 | 0.0784 |
| holdout | candidate_EV_≤0% | 22589 | 920 | 927 | 1847 | 0.0818 |
| holdout | candidate_EV_[2%,4%) | 1397 | 60 | 54 | 114 | 0.0816 |
| holdout | candidate_EV_[1%,2%) | 1294 | 52 | 57 | 109 | 0.0842 |
| holdout | candidate_EV_≥4% | 1311 | 52 | 42 | 94 | 0.0717 |
| holdout | candidate_EV_(0%,1%) | 1796 | 61 | 69 | 130 | 0.0724 |

Before-fee gross PnL and engine net PnL on unchanged settled executions are shown below. Fee effect is computed as the difference between these results under the engine’s share/cash convention; reported USD-equivalent fees are not subtracted a second time.

| period | fills | settled_fills | unresolved_fills | gross_PnL_before_fees_official_usd | engine_net_PnL_after_fees_official_usd | fee_effect_under_engine_convention_usd | ledger_fee_USD_equivalent_sum | cash_fee_usd_sum | share_fee_units_sum |
|---|---|---|---|---|---|---|---|---|---|
| development | 1038 | 1038 | 0 | 456.2401 | 267.4394 | -188.8007 | 188.8425 | 177.9417 | 22.1420 |
| holdout | 3738 | 3737 | 1 | 131.1386 | -546.2172 | -677.3558 | 677.5483 | 677.5483 | 0.0000 |

Monthly and side fee/PnL rows are in fee_summary.csv. All detailed table artifacts and the frozen source manifest are recorded below.

## Interpretation and limits

**Decomposition.** Development settled PnL was $267.44 versus $279.60 T-60 EV (gap $-12.17). A/B/C were $2.37/$-42.46/$27.93; A+B left a $-40.09 gap at the Binance proxy, and C moved it to the official $-12.17 gap. Holdout settled PnL was $-546.22 versus $1242.61 EV (gap $-1788.83). A/B/C were $-24.53/$-1525.57/$-238.73; execution plus proxy realization left $-1550.10 at the Binance proxy, and the official-target difference added $-238.73. B is the main observed holdout component, but it includes outcome realization noise and is not, by itself, proof of miscalibration.

**EV selection.** On holdout policy-selected sides, official y-p averaged -1.6%, -2.6%, and -7.4% in the 1–2%, 2–4%, and ≥4% bins (valid-label n=1209, 1292, 1236); Binance y-p was -0.2%, -2.9%, and -7.0%. The ≥4% official intervals by side were UP [-0.098, -0.003]; DOWN [-0.135, -0.045]; both exclude zero, while the lower-bin side intervals include zero. In the ≥4% actual-fill bin, $968.95 forecast EV became $-302.40 net PnL. This supports EV-associated overestimation in the exposed holdout, concentrated at ≥4%, but does not establish a general calibration law.

**Market benchmark.** On the selected holdout sample, RAW BASE70 had worse official log loss and Brier than p_market by 0.008048 and 0.003921, and lower AUC by 0.013624. These are descriptive point comparisons conditional on BASE70 selection. On all common holdout markets, RAW-minus-market log loss/Brier/AUC differences were +0.000898 [-0.000083, +0.002091], +0.000432 [-0.000045, +0.001015], and +0.002357 [-0.004735, +0.009408]. Each paired 3-day block interval includes zero.

**Targets and fees.** Label disagreement rose from 3.14% of common development labels to 7.84% in holdout, and was 7.84% on settled holdout fills. Before fees, holdout gross PnL was $131.14; engine fees reduced it by $677.36 to $-546.22. The aggregate loss therefore appeared after fees; fees do not explain the large EV-to-realized-outcome gap. The decomposition covers 1,038/1,038 development and 3,737/3,737 holdout settled fills with both labels. Holdout results were previously exposed and are retrospective, not an independent test; no policy change follows from this report.

The exact identity passed; maximum absolute identity error was 1.78e-15 USD. Runtime: 94.53 s; sampled peak RSS: 0.470 GiB.

Frozen manifest: reports/btc_preopen/btc_base70_ev_pnl_attribution_20261019_manifest.json. Detailed tables and ledger-level decomposition: data/analysis/polymarket/BTC/btc_base70_ev_pnl_attribution_20261019/. Reproduce with python analyze_btc_base70_ev_pnl_attribution_20261019.py (no arguments).
