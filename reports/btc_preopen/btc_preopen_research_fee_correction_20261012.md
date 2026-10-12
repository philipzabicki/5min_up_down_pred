# BTC pre-open fee and minimum-order correction — 2026-10-12

> **Correction notice.** This report supersedes the portfolio valuations in [`btc_preopen_research_20261011_full_compute20_retry.md`](btc_preopen_research_20261011_full_compute20_retry.md). That earlier report used archived snapshot fee metadata as the effective schedule and misread Gamma `orderMinSize` as shares. This correction applies the accepted dated fee rules and interprets Gamma's minimum as USDC.

## Evaluation scope

The existing development OOF predictions, evaluated-holdout predictions, and verified T-60/T-59 book snapshots were reused. The holdout model was not refit; the book replay was not rerun; no live settings or orders were changed. Only development policy portfolios, the frozen policy file, holdout portfolios, and downstream reports/manifests were regenerated. The new artifacts are under `data/analysis/polymarket/BTC/preopen_research_20261012_fee_correction/`.

The evaluated holdout had already been inspected in the previous report, so it is historically exposed. Its corrected results are a policy re-evaluation, not an unseen confirmatory holdout.

## Corrected execution assumptions

- The shared [`polymarket_btc5m_fee_rules.py`](../../utils/polymarket_btc5m_fee_rules.py) assigns the effective rate from execution time: 7.2% before 2026-05-06 00:00 UTC and 7.0% from that boundary. Archived snapshot bps and collection mode remain audit fields only.
- Collection follows the accepted exchange schedule: outcome-share fees before the estimated 2026-04-28 11:00 UTC cutover, no orders during the 11:00–12:00 maintenance window, and collateral cash fees after 12:00 UTC. Fees are rounded at each V1 share level or once per V2 order. T-60 EV includes the assigned fee; T-59 accounting charges only actual FAK fills, including partial fills.
- Polymarket documents Gamma `orderMinSize` in USDC and describes `/book` `min_order_size` separately in shares. FAK market BUY `amount` is USDC, and a FAK can partially fill before cancelling its remainder. The replay therefore compares the requested FAK USDC amount with current Gamma `orderMinSize` (or the documented $5 USDC fallback) and allows a resulting partial fill below that threshold. This metadata is current, not historical. The configured live profiles use FAK, and [`run.py`](../../run.py) builds BUY `MarketOrderArgs.amount` from `intent['bet_usdc']` in USDC; the live path was reviewed for units and left unchanged. See [Market Details](https://docs.polymarket.com/market-data/market-details) and [Place Orders](https://docs.polymarket.com/trading/place-orders).

## Development policy selection

The best active challenger is ranked by ending cash, then lower maximum drawdown, then fewer trades. It is selected only when it beats the $100 no-trade baseline under that ordering. The fixed-$5 control is reported separately and is not part of the selectable free-cash hurdle grid; the zero-hurdle 5% policy is both a grid candidate and the required uncapped 5% control.

| Development policy | Ending cash | Net PnL | Trades | Fees | Max drawdown |
|---|---:|---:|---:|---:|---:|
| free_cash_5pct_ev_0.0000pct | $99.41 | -$0.59 | 2 | $0.37 | 5.00% |
| free_cash_5pct_ev_0.2500pct | $99.41 | -$0.59 | 2 | $0.37 | 5.00% |
| free_cash_5pct_ev_0.5000pct | $95.00 | -$5.00 | 1 | $0.18 | 5.00% |
| free_cash_5pct_ev_1.0000pct | $99.41 | -$0.59 | 2 | $0.37 | 5.00% |
| free_cash_5pct_ev_2.0000pct | $95.00 | -$5.00 | 1 | $0.18 | 5.00% |
| no_trade | $100.00 | $0.00 | 0 | $0.00 | 0.00% |
| fixed_5_usd_control | $316.29 | $216.29 | 1,797 | $321.87 | 97.04% |
| free_cash_5pct_control | $99.41 | -$0.59 | 2 | $0.37 | 5.00% |

**Frozen selection:** `no_trade`. **Best active challenger:** `free_cash_5pct_ev_0.0000pct`. **No-trade baseline:** $100.00.

## Corrected holdout portfolio results

Every active policy below was simulated chronologically from the $100 initial balance. The controls are fixed $5 and 5% of free cash without a dollar cap. The selected policy, best active challenger when no-trade wins development, and controls were fully rerun with the seven-point T-59 slippage grid.

| Holdout policy role | Ending cash | Net PnL | Trades | Gross turnover | Fees | Max drawdown | Accounting audit |
|---|---:|---:|---:|---:|---:|---:|---|
| no_trade_baseline | $100.00 | $0.00 | 0 | $0.00 | $0.00 | 0.00% | not_applicable |
| selected_policy | $100.00 | $0.00 | 0 | $0.00 | $0.00 | 0.00% | not_applicable |
| best_active_challenger | $99.40 | -$0.60 | 2 | $10.24 | $0.36 | 5.17% | passed |
| fixed_5_usd_control | $1.18 | -$98.82 | 566 | $2,830.00 | $100.37 | 99.01% | passed |
| free_cash_5pct_control | $99.40 | -$0.60 | 2 | $10.24 | $0.36 | 5.17% | passed |

The previous holdout policy `free_cash_5pct_ev_0.5000pct` reported $25.60 ending cash, -$74.40 PnL, 138 trades, and $0.00 fees. The corrected selected-policy result is $100.00 ending cash, $0.00 PnL, 0 trades, and $0.00 fees.

| Slippage result | $0.0000/share | $0.0010/share | $0.0025/share | $0.0050/share | $0.0100/share | $0.0250/share | $0.0500/share |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_trade_baseline | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 |
| selected_policy | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 |
| best_active_challenger | -$0.60 | -$0.62 | -$0.65 | -$0.69 | -$0.78 | -$1.03 | -$1.43 |
| fixed_5_usd_control | -$98.82 | -$97.34 | -$98.20 | -$97.48 | -$95.14 | -$99.54 | -$99.86 |
| free_cash_5pct_control | -$0.60 | -$0.62 | -$0.65 | -$0.69 | -$0.78 | -$1.03 | -$1.43 |

| Trade count | $0.0000/share | $0.0010/share | $0.0025/share | $0.0050/share | $0.0100/share | $0.0250/share | $0.0500/share |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_trade_baseline | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| selected_policy | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| best_active_challenger | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| fixed_5_usd_control | 566 | 416 | 415 | 407 | 100 | 64 | 57 |
| free_cash_5pct_control | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

Each point restarts with $100 and recomputes execution eligibility, fills, later free-cash stakes, settlements, and the resulting trade set; T-60 information and decisions remain causal and fixed. Complete stateful paths and decision ledgers are in [`holdout_fee_corrected_portfolio_results.json`](../../data/analysis/polymarket/BTC/preopen_research_20261012_fee_correction/runs/fee_corrected_holdout/holdout_fee_corrected_portfolio_results.json).

Holdout decision eligibility and rejection counts:

| Policy role | Decisions | T-60 priceable markets | T-60 chosen side executable at T-59 | Fills | Skip reasons |
|---|---:|---:|---:|---:|---|
| selected_policy | 0 | 0 | 0 | 0 | no-trade policy; no market decisions |
| best_active_challenger / free_cash_5pct_control | 28,222 | 7 | 2 | 2 | below_minimum_order_usd=23,104, missing_official_outcome_or_settlement_time=2, no_positive_net_expected_value=5, no_priceable_native_ask_side=5,109 |
| fixed_5_usd_control | 28,222 | 23,111 | 5,870 | 566 | ambiguous_equal_source_timestamp_order=372, future_source_timestamp_at_entry=1, insufficient_free_cash_for_cash_debit=5,304, missing_official_outcome_or_settlement_time=2, native_ask_bbo_reconciliation_mismatch=4, no_positive_net_expected_value=16,864, no_priceable_native_ask_side=5,109 |

Monthly holdout results by selected policy and control:

| Policy role | Month UTC | Trades | Gross turnover | Fees | Monthly PnL | Entry cohort PnL | Monthly drawdown | Opening equity | Closing equity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| selected_policy | 2026-06 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00% | $100.00 | $100.00 |
| selected_policy | 2026-07 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00% | $100.00 | $100.00 |
| selected_policy | 2026-08 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00% | $100.00 | $100.00 |
| selected_policy | 2026-09 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00% | $100.00 | $100.00 |
| selected_policy | 2026-10 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00% | $100.00 | $100.00 |
| best_active_challenger / free_cash_5pct_control | 2026-06 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00% | $100.00 | $100.00 |
| best_active_challenger / free_cash_5pct_control | 2026-07 | 2 | $10.24 | $0.36 | -$0.60 | -$0.60 | 5.17% | $100.00 | $99.40 |
| best_active_challenger / free_cash_5pct_control | 2026-08 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00% | $99.40 | $99.40 |
| best_active_challenger / free_cash_5pct_control | 2026-09 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00% | $99.40 | $99.40 |
| best_active_challenger / free_cash_5pct_control | 2026-10 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00% | $99.40 | $99.40 |
| fixed_5_usd_control | 2026-06 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00% | $100.00 | $100.00 |
| fixed_5_usd_control | 2026-07 | 566 | $2,830.00 | $100.37 | -$98.82 | -$98.82 | 99.01% | $100.00 | $1.18 |
| fixed_5_usd_control | 2026-08 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00% | $1.18 | $1.18 |
| fixed_5_usd_control | 2026-09 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00% | $1.18 | $1.18 |
| fixed_5_usd_control | 2026-10 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00% | $1.18 | $1.18 |

Filled-trade depth and outcome counts:

| Policy role | Execution source | Full book | Trades | Wins | Losses | Gross shares | Net PnL |
|---|---|---:|---:|---:|---:|---:|---:|
| selected_policy | no fills | n/a | 0 | 0 | 0 | 0.00 | $0.00 |
| best_active_challenger / free_cash_5pct_control | pmxt_v2 | True | 2 | 1 | 1 | 20.4825 | -$0.60 |
| fixed_5_usd_control | pmxt_v2 | True | 566 | 280 | 286 | 5,761.5808 | -$98.82 |

Both active holdout policies used full-book PMXT depth; there were no top-of-book proxy fills.

The selected policy is no-trade, so it has no fill ledger and an accounting audit is not applicable. Active control accounting audits are reported in the holdout summary table. Filled trades by assigned collection mode for the selected policy were none. Skipped-decision reasons for the selected policy were none. Active trades span these UTC months: none.

## Provenance

- Reused development OOF manifest SHA-256: `001dcd816a83ea5bda001595ddd6e57d977cee0624c2eff5a585e9b2498be88a`.
- Reused evaluated-holdout prediction SHA-256: `b8febf2da37695cae6727737abc9cd650c7a69052b97b6ac72d27254801426f6` (28,224 rows); previously frozen model bundle SHA-256: `4820e23493efb1743863b58b411352006a71ff7ce8c66068c17c8c88e65e4583`.
- T-59 snapshot SHA-256: `abee0fdaf31eaf1455b4343e985bf9d5ee38a84dd8eb7e06adf1e08fc0c380d0`; T-60 snapshot SHA-256: `38259fadf50c28e455f5d4523610cbac827886e35922076dbba5cb33bd55e0b1`.
- Correction computation time: 565.4 seconds; observed process working set peaked at approximately 5.9 GiB in samples. No training, model refit, broad replay, paid compute, hosted model inference, or market-data service was used.
- Machine-readable policy freeze: [`selected_development_policy.json`](../../data/analysis/polymarket/BTC/preopen_research_20261012_fee_correction/selected_development_policy.json); correction run manifest: [`research_manifest.json`](../../data/analysis/polymarket/BTC/preopen_research_20261012_fee_correction/research_manifest.json); holdout provenance: [`fee_corrected_holdout_manifest.json`](../../data/analysis/polymarket/BTC/preopen_research_20261012_fee_correction/fee_corrected_holdout_manifest.json).
