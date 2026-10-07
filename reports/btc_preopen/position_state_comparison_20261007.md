# Position state correction comparison — 2026-10-07

The original report and files remain the v1 baseline. The v2 runner used a separate cache and output set. The outcome timestamp is `resolved_at_utc` copied into `outcome_available_at_utc`; it is an explicit proxy, not the recorded receipt time of the official result.

| Policy | v1 PnL | v2 PnL | v1 max DD | v2 max DD | v1/v2 trades | changed decisions | known locked moments | RCK rejections | max exposure v1/v2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_trade | $0.00 | $0.00 | 0.000% | 0.000% | 0/0 | 0 | 0 | 0 | $0.00/$0.00 |
| legacy_best_level_fixed_5 | $461.15 | $461.15 | 63.816% | 63.816% | 2244/2244 | 0 | 0 | 0 | $15.00/$15.00 |
| legacy_exact_fixed_5 | $461.42 | $461.42 | 75.366% | 75.366% | 2285/2285 | 0 | 0 | 0 | $15.00/$15.00 |
| fractional_kelly_half_10pct | $209.65 | $209.65 | 32.621% | 32.621% | 529/529 | 0 | 0 | 0 | $23.33/$23.33 |
| point_kelly_btc_only | $453.87 | $453.87 | 89.517% | 89.517% | 1779/1779 | 0 | 0 | 0 | $40.00/$40.00 |
| drk_btc_only | $-14.39 | $-14.39 | 62.199% | 62.199% | 265/265 | 0 | 0 | 0 | $17.40/$17.40 |
| point_kelly_btc_market | $312.08 | $312.08 | 86.139% | 86.139% | 1824/1824 | 0 | 0 | 0 | $40.00/$40.00 |
| drk_btc_market | $-49.07 | $-49.07 | 70.240% | 70.240% | 285/285 | 0 | 0 | 0 | $15.41/$15.41 |
| portfolio_kelly_independent | $319.19 | $319.19 | 86.361% | 86.361% | 1827/1827 | 0 | 0 | 0 | $40.00/$40.00 |
| rck_alpha_0.5_correlated | $33.52 | $33.52 | 39.770% | 39.770% | 303/303 | 0 | 0 | 43 | $24.27/$24.27 |
| rck_alpha_0.5_independent | $29.88 | $29.88 | 42.787% | 42.787% | 281/281 | 0 | 0 | 39 | $24.26/$24.26 |
| rck_alpha_0.7_correlated | $-4.18 | $-4.18 | 21.862% | 21.862% | 26/26 | 0 | 0 | 2 | $12.87/$12.87 |
| rck_alpha_0.7_independent | $-4.22 | $-4.22 | 21.862% | 21.862% | 26/26 | 0 | 0 | 2 | $12.87/$12.87 |
| rck_alpha_0.8_correlated | $-5.95 | $-5.95 | 10.860% | 10.860% | 4/4 | 0 | 0 | 0 | $9.12/$9.12 |
| rck_alpha_0.8_independent | $-5.95 | $-5.95 | 10.860% | 10.860% | 4/4 | 0 | 0 | 0 | $9.12/$9.12 |

No evaluated decision timestamp fell between result availability and cash release. Decisions are 300 seconds apart; result availability was 314–752 seconds after market start, while release is `max(resolved_at, market_start+300s)+60s`. At the +541-second decision, an early result had already released or a late result was not yet available; by +841 seconds even the latest release had occurred. Thus the state correction changes no decision, RCK candidate rejection, trade, PnL, drawdown, or exposure in this schedule.

A known locked win contributes its deterministic payout once to terminal scenario wealth; a known loss contributes zero. Neither is current cash. The release event removes the position and adds payout to cash once.
