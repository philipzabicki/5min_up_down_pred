# Binance–Polymarket disagreement by settlement regime

Date: 2026-10-23

Status: complete on existing saved labels; Chainlink price reconstruction not available without authorized stream access.

## Scope and evidence

This analysis reuses the frozen common sample in `target_observation_ledger.parquet`: 79,650 resolved binary labels from 2025-12-18 04:25 UTC through 2026-10-06 23:50 UTC. It does not recompute model predictions, economics, or the A/B/C/D comparison. The 28 Gamma-matched rows without an unambiguous binary official label remain excluded as in the previous report.

The local Gamma archive keeps outcome and timing fields but no market rule description or resolution-source text. I therefore keep two ideas separate:

- **Calendar-assigned regime:** all 79,650 rows are grouped using the supplied transition dates, checked against individual Polymarket rules pages around both boundaries.
- **Market-page evidence:** the rule text was directly inspected for seven markets. The other 79,643 rows have no per-market rule text in the saved Gamma data and were not individually page-checked. Their assigned regime remains an inference from date, not a market-specific metadata fact.

The seven inspected pages all match the supplied calendar; no conflict was observed. Two directly adjacent pages bracket each transition. The available [Predictions Changelog capture](sources/polymarket_predictions_changelog_capture_20261023.md) does not show the Aug 7 or Aug 14 entries. It is a text excerpt returned by a reader marked as crawled two months earlier, not a raw page archive. The boundary dates are therefore supported here by the individual market pages, not by an accessible copy of those changelog entries.

| Strict evidence classification | Markets | Treatment |
|---|---:|---|
| Rule text directly inspected on a market page | 7 | Market-confirmed; all match the supplied calendar |
| Assigned directly from accessible changelog entry | 0 | Neither transition entry appears in the accessible capture |
| Market-specific rule evidence unavailable | 79,643 | Unknown at row level; date-assigned only for the conditional regime summaries below |
| Conflicting rule text | 0 | No conflict among the seven inspected pages |

| Market start (UTC) | Role | Market page and declared feed | Window | Tie | Evidence |
|---|---|---|---:|---|---|
| 2026-08-06 23:50 | Regime anchor | [Market rules](https://polymarket.com/event/btc-updown-5m-1786060200); [BTC/USD feed](https://data.chain.link/streams/btc-usd) | 0 s | UP (`>=`) | Market page |
| 2026-08-06 23:55 | Immediately before Aug 7 | [Market rules](https://polymarket.com/event/btc-updown-5m-1786060500); [BTC/USD feed](https://data.chain.link/streams/btc-usd) | 0 s | UP (`>=`) | Market page |
| 2026-08-07 00:00 | Immediately after Aug 7 | [Market rules](https://polymarket.com/event/btc-updown-5m-1786060800); [BTC/USD TWAP30 feed](https://data.chain.link/streams/btc-usd-twap-30s-streams) | 30 s | UP (`>=`) | Market page |
| 2026-08-12 18:45 | TWAP30 anchor | [Market rules](https://polymarket.com/event/btc-updown-5m-1786560300); [BTC/USD TWAP30 feed](https://data.chain.link/streams/btc-usd-twap-30s-streams) | 30 s | UP (`>=`) | Market page |
| 2026-08-13 23:55 | Immediately before Aug 14 | [Market rules](https://polymarket.com/vi/event/btc-updown-5m-1786665300); [BTC/USD TWAP30 feed](https://data.chain.link/streams/btc-usd-twap-30s-streams) | 30 s | UP (`>=`) | Market page |
| 2026-08-14 00:00 | Immediately after Aug 14 | [Market rules](https://polymarket.com/zh-hant/event/btc-updown-5m-1786665600); [BTC/USD TWAP60 feed](https://data.chain.link/streams/btc-usd-twap-60s-streams) | 60 s | UP (`>=`) | Market page |
| 2026-09-18 20:50 | TWAP60 anchor | [Market rules](https://polymarket.com/event/btc-updown-5m-1789764600); [BTC/USD TWAP60 feed](https://data.chain.link/streams/btc-usd-twap-60s-streams) | 60 s | UP (`>=`) | Market page |

The market pages say the UP outcome wins when the listed Chainlink value for the interval is greater than or equal to the value at the beginning. Thus, each boundary comparison uses the source/feed named on that specific market page; the September page is only a TWAP60 spot check, not evidence for earlier markets.

The machine-readable 43-market sample is [versioned here](btc_discrepancy_regimes_20261023_sample_manifest.csv). It contains 12 seeded rows per date-assigned regime (six label agreements and six disagreements), four immediately adjacent boundary markets, and three additional rule anchors. Seed: `20261023`. The sample was not selected by PnL or model quality and is not an estimate of population disagreement rates. The local full assignment ledger records rule, feed URL, TWAP length, tie rule, direct-page evidence status, and assignment basis for every common-sample row: `data/analysis/polymarket/BTC/btc_discrepancy_regimes_20261023/market_contract_assignment_ledger.csv.gz`.

## Correction to the earlier report

The previous [candidate attribution report](btc_candidate_attribution_20261023.md) described a 60-second TWAP contract across the full sample. That description was too broad. The historical contract is single price before 2026-08-07 00:00 UTC, TWAP30 from that boundary through 2026-08-14 00:00 UTC, then TWAP60. The saved Gamma labels and prior disagreement rates were not constructed from a reconstructed Chainlink price, so this was a contract-description error, not an error in the target-label construction or the overall disagreement-rate calculations. No labels or A/B/C/D scores were changed.

I also found a separate arithmetic error in the previous report's aggregate direction cells. The monthly direction cells sum to the corrected totals below; the previous report's aggregate direction figures should be superseded by these. The disagreement counts and rates are unchanged.

| Period | Previous Binance UP → PM DOWN / Binance DOWN → PM UP | Corrected direction counts |
|---|---:|---:|
| Confirmation | 973 / 1,007 | 986 / 994 |
| Holdout | 1,054 / 1,159 | 1,090 / 1,123 |

## Disagreements by calendar-assigned regime

Movement quantiles are Binance absolute return in basis points. “Disagreement move” quantiles refer only to markets where the two labels differ. Directions are counts of Binance UP → Polymarket DOWN and the converse.

| Period / assigned regime | Markets | Disagreements | Rate | Direction UP→DOWN / DOWN→UP | Abs. return mean / p25 / median / p75 / p90 (bp) | Disagreement abs. return median / p90 (bp) |
|---|---:|---:|---:|---:|---:|---:|
| Confirmation / single price | 51,430 | 1,980 | 3.85% | 986 / 994 | 8.57 / 2.19 / 5.44 / 11.15 / 19.51 | 0.40 / 1.24 |
| Confirmation / TWAP30 | 0 | 0 | — | 0 / 0 | — | — |
| Confirmation / TWAP60 | 0 | 0 | — | 0 / 0 | — | — |
| Holdout / single price | 10,654 | 276 | 2.59% | 125 / 151 | 6.62 / 1.73 / 4.42 / 8.94 / 15.16 | 0.20 / 0.75 |
| Holdout / TWAP30 | 2,016 | 175 | 8.68% | 76 / 99 | 4.23 / 0.93 / 2.55 / 5.89 / 10.01 | 0.38 / 1.83 |
| Holdout / TWAP60 | 15,550 | 1,762 | 11.33% | 889 / 873 | 7.43 / 2.01 / 4.82 / 9.66 / 16.78 | 1.23 / 4.47 |
| **All / single price** | **62,084** | **2,256** | **3.63%** | **1,111 / 1,145** | **8.24 / 2.10 / 5.26 / 10.74 / 18.74** | **0.37 / 1.20** |
| **All / TWAP30** | **2,016** | **175** | **8.68%** | **76 / 99** | **4.23 / 0.93 / 2.55 / 5.89 / 10.01** | **0.38 / 1.83** |
| **All / TWAP60** | **15,550** | **1,762** | **11.33%** | **889 / 873** | **7.43 / 2.01 / 4.82 / 9.66 / 16.78** | **1.23 / 4.47** |

Holdout split at the supplied transition dates:

| Holdout slice | Markets | Disagreements | Rate | Direction UP→DOWN / DOWN→UP | Abs. return mean / median / p90 (bp) |
|---|---:|---:|---:|---:|---:|
| July, single price | 8,927 | 234 | 2.62% | 104 / 130 | 6.90 / 4.69 / 15.59 |
| Aug 1–6, before first change | 1,727 | 42 | 2.43% | 21 / 21 | 5.17 / 3.46 / 11.97 |
| TWAP30 | 2,016 | 175 | 8.68% | 76 / 99 | 4.23 / 2.55 / 10.01 |
| TWAP60 through last saved market | 15,550 | 1,762 | 11.33% | 889 / 873 | 7.43 / 4.82 / 16.78 |

The two post-transition groups together contain 17,566 markets and 1,937 disagreements (11.03%). They are 62.2% of holdout markets and 87.5% of holdout disagreements. This concentration is consistent with the rule change being relevant to the observed rise, but it does not identify a causal effect.

The single-price holdout is lower than confirmation (2.59% vs 3.85%). Its disagreement rate is lower in the exact-zero and (0,10] bp bins; the larger bins have zero or one disagreement in each period. This does not show a pre-change rise. The increase in the aggregate holdout rate is concentrated in the post-transition date-assigned groups. The regime and movement tables do not establish that TWAP caused the increase: calendar period, market conditions, and sample composition changed together.

### Existing Binance movement bins

Each cell is `markets / disagreements / disagreement rate`. All seven frozen bins have nonempty markets in each date-assigned regime, so comparisons by these bins have common support. A bin with zero disagreements is still reported; it is not evidence of a zero underlying probability.

| Absolute Binance return | Confirmation, single | Holdout, single | Holdout, TWAP30 | Holdout, TWAP60 |
|---|---:|---:|---:|---:|
| Exactly 0 bp | 2 / 1 / 50.00% | 1 / 0 / 0.00% | 1 / 0 / 0.00% | 1 / 0 / 0.00% |
| (0, 1] bp | 6,322 / 1,668 / 26.38% | 1,656 / 261 / 15.76% | 528 / 133 / 25.19% | 2,076 / 777 / 37.43% |
| (1, 5] bp | 17,938 / 306 / 1.71% | 4,118 / 14 / 0.34% | 899 / 39 / 4.34% | 5,901 / 840 / 14.23% |
| (5, 10] bp | 12,424 / 5 / 0.04% | 2,586 / 0 / 0.00% | 384 / 3 / 0.78% | 3,874 / 123 / 3.18% |
| (10, 25] bp | 11,622 / 0 / 0.00% | 2,019 / 1 / 0.05% | 184 / 0 / 0.00% | 3,064 / 21 / 0.69% |
| (25, 50] bp | 2,635 / 0 / 0.00% | 241 / 0 / 0.00% | 19 / 0 / 0.00% | 556 / 1 / 0.18% |
| >50 bp | 487 / 0 / 0.00% | 33 / 0 / 0.00% | 1 / 0 / 0.00% | 78 / 0 / 0.00% |

In particular, disagreement rates within the same frozen movement bins are higher in the post-change holdout groups, including bins above 1 bp. This reduces the chance that the aggregate shift is only due to more near-zero Binance moves; it does not isolate a causal effect. July and Aug 1–6 versus post-change holdout bin counts/rates are retained in `data/analysis/polymarket/BTC/btc_discrepancy_regimes_20261023/movement_bin_disagreements.csv`. The exact-zero bin is absent in Aug 1–6, so comparisons for that slice have six of seven bins; all other holdout-slice pairs have six or seven shared bins as recorded in `common_move_bin_coverage.csv`.

The pre-change holdout movement bins are shown separately because their mix differs from the post-change periods:

| Absolute Binance return | July | Aug 1–6, before change |
|---|---:|---:|
| Exactly 0 bp | 1 / 0 / 0.00% | 0 / 0 / — |
| (0, 1] bp | 1,308 / 221 / 16.90% | 348 / 40 / 11.49% |
| (1, 5] bp | 3,347 / 12 / 0.36% | 771 / 2 / 0.26% |
| (5, 10] bp | 2,224 / 0 / 0.00% | 362 / 0 / 0.00% |
| (10, 25] bp | 1,800 / 1 / 0.06% | 219 / 0 / 0.00% |
| (25, 50] bp | 216 / 0 / 0.00% | 25 / 0 / 0.00% |
| >50 bp | 31 / 0 / 0.00% | 2 / 0 / 0.00% |

Cells use the same `markets / disagreements / rate` format. The empty exact-zero cell in Aug 1–6 is not a zero-rate observation.

## Chainlink reconstruction feasibility

No sample report was fetched. The available environment had no Chainlink/Data Streams credential variable names, and the local cache does not contain the target reports. I made no authenticated API request, purchased no access, and did not substitute Binance, a generic BTCUSD stream, or an independently calculated average.

| Question | Finding |
|---|---|
| Correct target streams | The market pages link the names `btc-usd`, `btc-usd-twap-30s-streams`, and `btc-usd-twap-60s-streams`. The exact Chainlink feed IDs for the 30s and 60s TWAP streams were not discoverable in the accessible official listing, so they remain unverified. The locally saved `data/chainlink/metadata/BTCUSD.json` is a regular BTC/USD CEX-price stream, not either TWAP feed and not the classic on-chain Data Feed. |
| Report schema and price scale | Polymarket's TWAP integration documentation specifies Chainlink report schema V2 for these custom TWAP feeds and an exact signed E18 price. Decode `decoded.price` / `full_accuracy_value` with integer or decimal arithmetic at 18 decimals; do not round to the UI display. The generic crypto v3 schema is not the target TWAP schema. |
| Timestamps | For the Data Streams API, `validFromTimestamp` is the earliest time the report's value is valid and `observationsTimestamp` is the latest valid/Chainlink observation time, in seconds. RTDS supplies the Chainlink observation timestamp in `payload.timestamp` (milliseconds); its outer event timestamp is publisher send time. |
| Endpoint and history | The API documents timestamp lookup (`/api/v1/reports?feedID&timestamp`) and sequential pagination (`/reports/page?feedID&startTimestamp&limit`), but these docs do not state retention or availability for the requested historical timestamps. Polymarket RTDS also exposes live BTC/USD TWAP30 and TWAP60 updates without credentials, but subscriptions start with the next update and offer no snapshot, history, or replay. RTDS cannot reconstruct these past markets now. |
| Authentication and authorization | Direct Chainlink Data Streams requests require a UUID authorization value, millisecond `X-Authorization-Timestamp`, and HMAC-SHA256 signature using a shared secret; feed access is subscription-scoped. The public Polymarket RTDS live stream requires no credentials. No usable local Chainlink credentials were found. |
| Cost | Polymarket RTDS is documented as credential-free for live updates. Direct Data Streams uses subscription billing; the official billing page does not publish a mainnet price and directs customers to contact Chainlink. The available documentation does not support a historical-access cost estimate. No account or paid service was created. |
| Local cache | `data/chainlink/raw_reports/BTCUSD_reports.csv` has 1,426 rows from 2026-03-21 18:51:45 to 21:26:57 UTC with only `ObservedAt, Price, Bid, Ask`. It has no raw signed report, checksum, or TWAP values and covers only a short window; it cannot reconstruct this sample. |

The relevant [Polymarket Chainlink TWAP documentation](https://docs.polymarket.com/market-data/chainlink-twap) documents credential-free live RTDS, the no-history behavior, the V2 signed-E18 TWAP report, and how to find the feed ID in Chainlink's catalog. The catalog itself returned HTTP 403 in this review, so the exact 30s and 60s feed IDs remain unverified. The [Chainlink Data Streams API](https://docs.chain.link/data-streams/reference/data-streams-api/interface-api) documents authenticated historical report endpoints and timestamp fields; [Data Streams billing](https://docs.chain.link/data-streams/billing) describes subscription pricing.

Polymarket's TWAP documentation also says Chainlink does not publish the custom feed's sampling boundaries, weighting, rounding, or missing-input behavior. I therefore did not attempt to synthesize TWAP values from point prices; exact retrospective reconstruction needs the correct official TWAP feed reports.

The immediate next step with the highest value is to obtain the two exact TWAP feed IDs plus written historical-retention and price confirmation from Chainlink for the sample's boundary timestamps. If an existing authorized subscription covers them, the 43-market sample needs at most 86 endpoint price lookups before reusing shared boundaries; otherwise stop before any paid request. This would establish whether the correct feed prices explain individual Gamma/Binance disagreements. A full reconstruction of the post-change common sample would cover 17,566 markets and require at most 35,132 endpoint report lookups before shared-boundary reuse. The present public documentation does not support a reliable monetary cost estimate, so that broader retrieval is not scoped or priced here.

## Conclusion

The supplied regime dates are supported by official market pages on both sides of both transitions. The changelog text currently accessible to this run omits the cited entries, and row-specific rule text is absent for 79,643 of 79,650 common-sample markets; those rows are therefore calendar-assigned rather than individually confirmed. The earlier 60s-for-all description is corrected, the stored Gamma labels and disagreement rates remain intact, and the previous aggregate direction cells are corrected. Disagreement rates do not rise in the single-price holdout; they rise in the post-change calendar-assigned groups and remain higher within the frozen Binance movement bins. Causation and exact price reconstruction remain untested pending the correct feed IDs, history entitlement, and pricing.
