# E0038 — T0009 source quality continuation

## Frozen checkpoint before economics

Protocol: [E0038 v3 frozen correction](../../configs/research/btc_t0009_source_quality_continuation_20261026_v3.json), SHA256 `5852107c07e0bb1768fcfe9693322bd8b9b377963110424c8f09ac08e28d8008`. V2 remains at [its frozen protocol](../../configs/research/btc_t0009_source_quality_continuation_20261026_v2.json), SHA256 `98934fda651524a8da36c31e78c3970780c24b55f6e2a7cdfae37a2349f3dcae`; v1 remains at [its original protocol](../../configs/research/btc_t0009_source_quality_continuation_20261026.json), SHA256 `1dd1fcfad459a45318f3b8cd7936acd7bbdd32d140c088135c2cd5245929eee1`. Parent E0037 remains **FAILED** at its saved gate and its protocol, report, summary, and prior E0036 economics are unchanged.

The v1 freeze commit `30ce75b` was pushed before the first economics launch. Its runner stopped at the committed-protocol guard because Git normalized CRLF to LF; it stopped before loading source or outcome rows. V2, committed and pushed as `ac4acc4`, changed only the guard's newline comparison, then completed economics on the frozen 5,604 rows. Its E0036 comparison reported decision mismatches because pandas missing `selected_side` values became `NaN` and were read as the literal `"nan"` rather than `no_trade`. A direct join with missing values mapped to `no_trade` confirms zero decision differences for all four candidates on the 1,141 common rows; probabilities and fill/fee values already matched exactly. V3 changes only this comparison mapping and preserves the v2 run as prior evidence. The source rule, cohort, candidates, costs, sizing, fills, payouts, and account paths are unchanged. V3 must be committed and pushed before the corrected replay.

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
