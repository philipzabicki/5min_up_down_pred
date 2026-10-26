# E0038 — T0009 source quality continuation

## Frozen checkpoint before economics

Protocol: [E0038 v3 frozen correction](../../configs/research/btc_t0009_source_quality_continuation_20261026_v3.json), SHA256 `5852107c07e0bb1768fcfe9693322bd8b9b377963110424c8f09ac08e28d8008`. V2 remains at [its frozen protocol](../../configs/research/btc_t0009_source_quality_continuation_20261026_v2.json), SHA256 `98934fda651524a8da36c31e78c3970780c24b55f6e2a7cdfae37a2349f3dcae`; v1 remains at [its original protocol](../../configs/research/btc_t0009_source_quality_continuation_20261026.json), SHA256 `1dd1fcfad459a45318f3b8cd7936acd7bbdd32d140c088135c2cd5245929eee1`. Parent E0037 remains **FAILED** at its saved gate and its protocol, report, summary, and prior E0036 economics are unchanged.

The v1 freeze commit `30ce75b` was pushed before the first economics launch. Its runner stopped at the committed-protocol guard because Git normalized CRLF to LF; it stopped before loading source or outcome rows. V2, committed and pushed as `ac4acc4`, changed only the guard's newline comparison, then completed economics on the frozen 5,604 rows. Its E0036 comparison reported decision mismatches because pandas missing `selected_side` values became `NaN` and were read as the literal `"nan"` rather than `no_trade`. A direct join with missing values mapped to `no_trade` confirms zero decision differences for all four candidates on the 1,141 common rows; probabilities and fill/fee values already matched exactly. V3 changes only this comparison mapping and preserves the v2 run as prior evidence. The source rule, cohort, candidates, costs, sizing, fills, payouts, and account paths are unchanged. V3 was then committed and pushed as `e500553` before the corrected replay.

### T−60 source trace

For condition `0xe4d40fb41d0acedbcae8592f00ec3ac0f6e662f89260c11a65443385996537a3` (start 2026-04-30 07:00 UTC), the full UP book was initialized from `2026-04-29T22.parquet`, row group 0, row 86,417. Replaying 3,915 events through the 06:59 UTC receive-time cutoff leaves the UP ask at $0.51 for 236.98 shares. The last such level update is in `2026-04-30T06.parquet`, row group 0, row 112,555 (source 06:58:37.664, receive 06:58:37.778). No later $0.51 level update appears before the cutoff. The later reported-BBO group (rows 113,166–113,169; receive 06:58:59.863) reports $0.52.

The production parser and a small independent replay agree on the $0.51 book. Reported BBO is checked as evidence but does not mutate the replayed levels. The local source conflict is **unresolved**: this trace does not prove an implementation bug or a missing archive event. The market was already outside the 5,684-row base intersection because two-sided native T−60 BBO coverage is unavailable, so it is not subtracted again.

Representative T−59 traces show two further quarantined classes: a native ask/BBO mismatch ($0.50 book ask versus $0.51 reported BBO) that independently replays to the cached ask; and two conflicting same-source-time updates at $0.53 (sizes 164.99 and 174.99 shares), which make that ask level ambiguous. [Official source documentation](https://docs.polymarket.com/market-data/realtime-data) supports full-book and price-change message fields; the [source guide](https://github.com/Polymarket/agent-skills/blob/main/websocket.md) documents zero-size removal. The local replay uses level-size replacement semantics. These sources do not establish that every reported BBO must equal a locally reconstructed book.

### Parity layers

- **Source values:** across all 1,143 E0036 controls, 17 T−60 fields yielded 19,431 comparisons with zero mismatches; 23 T−59 fields yielded 26,289 comparisons with zero mismatches. Fields include prices, sizes/ladders, token identity, entry times, fee values/mode, and causal/quality values.
- **Causal and reconstruction checks:** the 5,684-row base cohort has no future receive-time or source-time event at either cutoff, and no archive gap after book or fee initialization. Entry-hour/freshness and known-fee checks are present. Wider 25-hour unprocessed-window flags affect 210 rows but are not post-initialization gaps.
- **Validation coverage:** old T−60 checks were 0 for 1,118 rows, 1 for 7, and 2 for 18; zero checks are `NOT_CHECKED`, not a pass. New checks cover both sides for all 1,143 rows: 1,142 pass and the traced conflict fails. T−59 has 1 partial-check row and 1,142 full-check rows, with no reported mismatches. E0037 remains **FAILED** as recorded.

### Frozen quarantine and cohort

The source-only intersection is 5,684 markets: joint source coverage, valid official label, and saved prediction. Apply the same rule to both native tokens at both cutoffs, regardless of labels, PnL, or model advantage: require valid ask reasons, direct token identity, comparable/reconciled native BBO, no ask ambiguity/cross, no future event, known fees, available entry-hour/freshness checks, and no unknown or missing archive hour after book/fee initialization.

This removes 80 markets: 62 with T−59 equal-source-time ask ambiguity, 16 with T−59 ask/BBO mismatch, and 2 with an uncomparable ask BBO. The categories do not overlap. The clean cohort is **5,604** (April 456, May 2,315, June 2,833). E0036 has 1,143 old control markets; 1,141 remain in the clean common sample. The other two have only one native T−60 side and cannot support a two-sided decision. Do not require old stateful ending equity to match after changing the sample.

The coverage manifest contains 105,118 rows: 51,458 Gamma markets and 53,660 candidate calendar rows without Gamma IDs. There are 51,430 valid Gamma labels and 105,117 rows with predictions. Unmapped rows are not confirmed missing markets and are not strategic no-trades. Only 5,604 qualify for the frozen conditional cohort; this is not full-development calendar coverage.

### Prediction classification and economics freeze

The unchanged saved A/B/C/D probabilities are tied to run identity `ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b` and candidate manifest SHA256 `16e66363f35390d520e5a3ca7ce58fa07689ac15e66244a313c50f1a29017503`. No measured prediction-ready timestamps exist. Nominal T−60 availability is a retrospective simulation assumption; T−59 runtime readiness is `UNKNOWN`, and full-pipeline selection/calibration independence is not established under D0005. Results cannot be called independent confirmation.

E0038 v3 carries forward v2's pinned sources and economic protocol: unchanged A/B/C/D decisions, T−59 fills, fixed-$5 independent diagnostic, separate $100 fixed-$5 and 5%-of-free-cash accounts, D0019's $5 USDC fallback, and payout delays of 60/300/1,800/7,200 seconds after `max(market end, Gamma resolved_at_utc)`. A v2 run exists; the v3 corrected replay has not run at this checkpoint. Delays are scenarios; actual redemption availability remains `UNKNOWN`. The runner requires the exact v3 protocol in `HEAD` and verifies pinned inputs/code before reading development labels or computing PnL.

Validation: focused E0038 tests **13/13** and full discovery **709/709** pass after the v3 comparison correction (**43.327 seconds**, exit code 0). `git diff --check` passed.

T0009 remains **BLOCKED** for full-development coverage and causal/readiness provenance. No holdout outcomes, training, tuning, acquisition, sale policy, live/shadow, orders, or production claims are in scope.

## Economic result - E0038 v3

The v3 protocol was committed and pushed as `e500553` before the corrected replay. The run completed in 31.339 seconds on the frozen 5,604-market source-quality cohort, from 2026-04-15 17:35 UTC through 2026-06-30 23:35 UTC. This is a coverage-conditional subset of the 2025-07 through 2026-06 development interval, not a full-calendar portfolio. Holdout rows used: 0; holdout outcome columns projected: false.

On the E0036 common clean sample (1,141 markets), all row-level probability, decision, gross, VWAP, net-share, cash-debit, and fee comparisons are exact for A/B/C/D (zero differences in every field). The v2 and v3 market-decision and monthly-path files have identical SHA256 hashes; v3 corrects only the saved `no_trade` label comparison.

### Independent fixed-$5 diagnostic

This is independent per-market accounting without a shared capital limit. Payout timing does not change these totals.

| Model | Signals/fills | Gross turnover | Cash fees | Share fees | Fee equivalent | Payouts | Net PnL |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 2,035/2,035 | $10,175.00 | $332.57 | 52.520 | $359.19 | $10,582.03 | $74.46 |
| B | 1,946/1,946 | $9,730.00 | $318.58 | 50.069 | $343.96 | $10,270.41 | $221.83 |
| C | 1,849/1,849 | $9,245.00 | $304.78 | 44.176 | $327.16 | $9,823.20 | $273.42 |
| D | 1,862/1,862 | $9,310.00 | $305.74 | 47.232 | $329.69 | $9,901.65 | $285.90 |

### $100 account, fixed $5 gross per signal

Signals are positive-EV selections at T-60. The no-positive-EV count is among the 5,604 eligible markets; cash rejections are signaled trades that could not be funded after fees. Ending equity includes free cash plus any locked cost basis. All scenarios finish with $0 locked cost basis and 0 unsettled positions.

| Model | Payout delay | Signals/fills | No-positive-EV | Cash rejects | Cash fees | Share fees | Fee equivalent | Payouts | Ending equity / net PnL | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 60s | 2,035/285 | 3,569 | 1,750 | $23.90 | 52.520 | $50.52 | $1,353.44 | $4.54 / -$95.46 | 0.962 |
| A | 300s | 2,035/285 | 3,569 | 1,750 | $23.90 | 52.520 | $50.52 | $1,353.44 | $4.54 / -$95.46 | 0.962 |
| A | 1800s | 2,035/282 | 3,569 | 1,753 | $23.38 | 52.520 | $50.00 | $1,333.64 | $0.26 / -$99.74 | 0.998 |
| A | 7200s | 2,035/261 | 3,569 | 1,774 | $19.74 | 52.520 | $46.36 | $1,225.01 | $0.27 / -$99.73 | 0.998 |
| B | 60s | 1,946/1,946 | 3,658 | 0 | $318.58 | 50.069 | $343.96 | $10,270.41 | $321.83 / $221.83 | 0.783 |
| B | 300s | 1,946/1,946 | 3,658 | 0 | $318.58 | 50.069 | $343.96 | $10,270.41 | $321.83 / $221.83 | 0.783 |
| B | 1800s | 1,946/1,946 | 3,658 | 0 | $318.58 | 50.069 | $343.96 | $10,270.41 | $321.83 / $221.83 | 0.787 |
| B | 7200s | 1,946/1,940 | 3,658 | 6 | $317.53 | 50.069 | $342.91 | $10,240.79 | $323.26 / $223.26 | 0.786 |
| C | 60s | 1,849/1,849 | 3,755 | 0 | $304.78 | 44.176 | $327.16 | $9,823.20 | $373.42 / $273.42 | 0.781 |
| C | 300s | 1,849/1,849 | 3,755 | 0 | $304.78 | 44.176 | $327.16 | $9,823.20 | $373.42 / $273.42 | 0.781 |
| C | 1800s | 1,849/1,849 | 3,755 | 0 | $304.78 | 44.176 | $327.16 | $9,823.20 | $373.42 / $273.42 | 0.782 |
| C | 7200s | 1,849/1,842 | 3,755 | 7 | $303.56 | 44.176 | $325.94 | $9,783.77 | $370.21 / $270.21 | 0.783 |
| D | 60s | 1,862/1,862 | 3,742 | 0 | $305.74 | 47.232 | $329.69 | $9,901.65 | $385.90 / $285.90 | 0.746 |
| D | 300s | 1,862/1,862 | 3,742 | 0 | $305.74 | 47.232 | $329.69 | $9,901.65 | $385.90 / $285.90 | 0.746 |
| D | 1800s | 1,862/1,862 | 3,742 | 0 | $305.74 | 47.232 | $329.69 | $9,901.65 | $385.90 / $285.90 | 0.746 |
| D | 7200s | 1,862/1,859 | 3,742 | 3 | $305.20 | 47.232 | $329.15 | $9,891.44 | $391.24 / $291.24 | 0.751 |

### $100 account, 5% of free cash

The $5 minimum is applied without upward rounding. Most requests fall below it after initial fees, so this frozen control places very few trades.

| Model | Payout delay | Signals/fills | Below-$5 rejections | No-positive-EV | Cash rejects | Cash fees / share fees | Payouts | Ending equity / net PnL | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 60s | 1/1 | 5,598 | 5 | 0 | $0.00 / 0.360 | $0.00 | $95.00 / -$5.00 | 0.050 |
| A | 300s | 1/1 | 5,598 | 5 | 0 | $0.00 / 0.360 | $0.00 | $95.00 / -$5.00 | 0.050 |
| A | 1800s | 1/1 | 5,598 | 5 | 0 | $0.00 / 0.360 | $0.00 | $95.00 / -$5.00 | 0.050 |
| A | 7200s | 1/1 | 5,598 | 5 | 0 | $0.00 / 0.360 | $0.00 | $95.00 / -$5.00 | 0.050 |
| B | 60s | 1/1 | 5,598 | 5 | 0 | $0.00 / 0.360 | $0.00 | $95.00 / -$5.00 | 0.050 |
| B | 300s | 1/1 | 5,598 | 5 | 0 | $0.00 / 0.360 | $0.00 | $95.00 / -$5.00 | 0.050 |
| B | 1800s | 1/1 | 5,598 | 5 | 0 | $0.00 / 0.360 | $0.00 | $95.00 / -$5.00 | 0.050 |
| B | 7200s | 1/1 | 5,598 | 5 | 0 | $0.00 / 0.360 | $0.00 | $95.00 / -$5.00 | 0.050 |
| C | 60s | 2/2 | 5,582 | 20 | 0 | $0.00 / 0.707 | $9.46 | $99.24 / -$0.76 | 0.050 |
| C | 300s | 2/2 | 5,583 | 19 | 0 | $0.00 / 0.707 | $9.46 | $99.24 / -$0.76 | 0.050 |
| C | 1800s | 2/2 | 5,585 | 17 | 0 | $0.00 / 0.707 | $9.46 | $99.24 / -$0.76 | 0.050 |
| C | 7200s | 2/2 | 5,585 | 17 | 0 | $0.00 / 0.707 | $9.46 | $99.24 / -$0.76 | 0.050 |
| D | 60s | 1/1 | 5,597 | 6 | 0 | $0.00 / 0.360 | $0.00 | $95.00 / -$5.00 | 0.050 |
| D | 300s | 1/1 | 5,597 | 6 | 0 | $0.00 / 0.360 | $0.00 | $95.00 / -$5.00 | 0.050 |
| D | 1800s | 1/1 | 5,597 | 6 | 0 | $0.00 / 0.360 | $0.00 | $95.00 / -$5.00 | 0.050 |
| D | 7200s | 1/1 | 5,597 | 6 | 0 | $0.00 / 0.360 | $0.00 | $95.00 / -$5.00 | 0.050 |

### Monthly path and interpretation

Clean source rows occur only in April (456), May (2,315), and June 2026 (2,833). The 432-row monthly artifact contains independent diagnostic totals and all 12-month account snapshots for each model/control/delay, including free cash, locked cost basis, equity, fees, payouts, drawdown, and unsettled positions. Months with zero clean source rows indicate unavailable cohort coverage, not strategy no-trades. All ending account positions are settled by the development cutoff.

Ignored local monthly path: `data/analysis/polymarket/BTC/btc_t0009_source_quality_continuation_20261026_5852107c07e0/monthly_paths.csv` (SHA256 `f4b2f18fe28cfa48fa6e10d8e1b4d5718cfe168cad1c6c3ec2a7fcc25302f813`). The 268,992-row market-decision ledger is `data/analysis/polymarket/BTC/btc_t0009_source_quality_continuation_20261026_5852107c07e0/market_decisions.csv` (SHA256 `10c5e05b564ab351524338e3d14de6412aa11429fd298ab6298c57ff1ad40694`). The v3 local summary is `data/analysis/polymarket/BTC/btc_t0009_source_quality_continuation_20261026_5852107c07e0/summary.json` (SHA256 `67f46ab02dcf91b4faf566610b308e89ddff1888dd03bd168ead2b06aee2f84e`); run manifest SHA256 `301a0dbd8909ae56b611db5b9b33654c8676d93213f398e681303efc9bdcc7e2`.

These are retrospective/conditional results on unchanged saved historical predictions. The zero measured prediction-ready timestamps, unknown T-59 runtime readiness, and unestablished full-pipeline selection/calibration independence prevent an independent confirmation or live-readiness claim. Positive A/B/C/D subset PnL does not close T0009 or authorize holdout use, a pilot, or production. No portfolio bootstrap or inferential uncertainty interval was run.

### Separate source task T0010

The exact 24-hour local PMXT gap inventory remains a separate deferred source task, [T0010](../../TASKS.md), recorded in the [T0009 coverage report](btc_t0009_resolution_coverage_20261026.md) and [machine summary](btc_t0009_resolution_coverage_20261026.json): 21 hours from 2026-04-13 19:00 through 2026-04-14 15:00 UTC and three hours from 2026-06-11 04:00 through 06:00 UTC. E0038 performed no acquisition. This inventory is distinct from the 80 market-level quarantines and is not an extra subtraction from the 5,684-market source intersection.
