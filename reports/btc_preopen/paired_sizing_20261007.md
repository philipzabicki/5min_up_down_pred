# Paired sizing diagnostics — 2026-10-07

Frozen side/time comes from the legacy candidate_platt exact-$5 positive expected-profit chooser. Each order gets a fresh $100 cash state; there is no shared cash path or reinvestment. The figures below are diagnostics, not returns achievable by one $100 portfolio.

| First | Second | Shared feasible | First only | Second only | Both rejected | Paired PnL first | Paired PnL second | Difference | Fixed-$5 common PnL / admitted gross |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| point_kelly_btc_only | drk_btc_only | 485 | 581 | 0 | 1219 | $37.44 | $-3.29 | $+40.72 | $-4.38 / $2388.08 |
| point_kelly_btc_market | drk_btc_market | 343 | 590 | 0 | 1352 | $2.16 | $-18.80 | $+20.96 | $-10.14 / $1682.04 |
| point_kelly_btc_market | rck_alpha_0.5_correlated | 405 | 528 | 0 | 1352 | $52.69 | $18.01 | $+34.69 | $24.13 / $1983.40 |
| point_kelly_btc_market | rck_alpha_0.7_correlated | 54 | 879 | 0 | 1352 | $84.02 | $19.65 | $+64.37 | $46.15 / $266.34 |
| point_kelly_btc_market | rck_alpha_0.8_correlated | 4 | 929 | 0 | 1352 | $-36.13 | $-8.68 | $-27.46 | $-10.20 / $20.00 |
| portfolio_kelly_independent | rck_alpha_0.5_independent | 405 | 528 | 0 | 1352 | $52.69 | $18.01 | $+34.69 | $24.13 / $1983.40 |
| portfolio_kelly_independent | rck_alpha_0.7_independent | 54 | 879 | 0 | 1352 | $84.02 | $19.65 | $+64.37 | $46.15 / $266.34 |
| portfolio_kelly_independent | rck_alpha_0.8_independent | 4 | 929 | 0 | 1352 | $-36.13 | $-8.68 | $-27.46 | $-10.20 / $20.00 |
| fractional_kelly | fixed_5 | 290 | 0 | 1954 | 41 | $112.38 | $196.41 | $-84.03 | $196.41 / $1409.53 |

The common-set method stakes are actual admitted gross: first and second are reported separately in JSON alongside the fixed-$5 accepted gross above.

## One-sided feasible orders and rejections

| Pair | First-only PnL | Rejected by second | Second-only PnL | Rejected by first | Both rejected | Reasons: first / second |
|---|---:|---|---:|---|---:|---|
| point_kelly_btc_only vs drk_btc_only | $264.52 | rounded_order_not_beneficial_or_feasible: 581 | $0.00 | none | 1219 | rounded_order_not_beneficial_or_feasible: 1178, no_positive_log_growth: 41 / rounded_order_not_beneficial_or_feasible: 1178, no_positive_log_growth: 41 |
| point_kelly_btc_market vs drk_btc_market | $258.61 | rounded_order_not_beneficial_or_feasible: 590 | $0.00 | none | 1352 | rounded_order_not_beneficial_or_feasible: 1311, no_positive_log_growth: 41 / rounded_order_not_beneficial_or_feasible: 1311, no_positive_log_growth: 41 |
| point_kelly_btc_market vs rck_alpha_0.5_correlated | $208.08 | no_positive_log_growth: 528 | $0.00 | none | 1352 | rounded_order_not_beneficial_or_feasible: 1311, no_positive_log_growth: 41 / no_positive_log_growth: 1352 |
| point_kelly_btc_market vs rck_alpha_0.7_correlated | $176.75 | no_positive_log_growth: 879 | $0.00 | none | 1352 | rounded_order_not_beneficial_or_feasible: 1311, no_positive_log_growth: 41 / no_positive_log_growth: 1352 |
| point_kelly_btc_market vs rck_alpha_0.8_correlated | $296.90 | no_positive_log_growth: 929 | $0.00 | none | 1352 | rounded_order_not_beneficial_or_feasible: 1311, no_positive_log_growth: 41 / no_positive_log_growth: 1352 |
| portfolio_kelly_independent vs rck_alpha_0.5_independent | $208.08 | no_positive_log_growth: 528 | $0.00 | none | 1352 | rounded_order_not_beneficial_or_feasible: 1311, no_positive_log_growth: 41 / no_positive_log_growth: 1352 |
| portfolio_kelly_independent vs rck_alpha_0.7_independent | $176.75 | no_positive_log_growth: 879 | $0.00 | none | 1352 | rounded_order_not_beneficial_or_feasible: 1311, no_positive_log_growth: 41 / no_positive_log_growth: 1352 |
| portfolio_kelly_independent vs rck_alpha_0.8_independent | $296.90 | no_positive_log_growth: 929 | $0.00 | none | 1352 | rounded_order_not_beneficial_or_feasible: 1311, no_positive_log_growth: 41 / no_positive_log_growth: 1352 |
| fractional_kelly vs fixed_5 | $0.00 | none | $264.74 | minimum_order_shares: 854, below_minimum_gross_notional: 1100 | 41 | insufficient_top_of_book: 17, minimum_order_shares: 15, below_minimum_gross_notional: 9 / insufficient_top_of_book: 17, minimum_order_shares: 24 |

Dollar stakes are the actual admitted gross amounts after cash and top-of-book limits; fixed $5 uses the same execution function and its actual admitted amount. No shared cash path or reinvestment is used in this diagnostic.
