# T0009 full-development source coverage — 2026-10-26

## Result

E0037 completed its source-only reconstruction for the frozen development interval. T0009 remains **BLOCKED**: E0036 T−60 source parity failed on one market, exact source gaps remain, and no prediction-ready timestamp exists for any matched Gamma market. The 60/300/1800/7200-second payout scenarios and all new economic calculations were not run because the frozen parity gate did not pass.

The [frozen protocol](../../configs/research/btc_t0009_resolution_coverage_20261026.json) is unchanged at SHA256 `38a6221f0d904112ead40197583dd3310bb27d9702ec4524e43df0144dca3dc1`. The source-coverage runner hash was `31f5165feb76608682295a331a8e0fda72c5f780b5723bf9aaeecc46e4e37057`. The frozen interval is `[2025-07-01T00:00:00Z, 2026-07-01T00:00:00Z)`. No holdout partitions, outcomes, or PnL were read.

## Coverage

The verified local PMXT replay covered 51,458 Gamma markets across 1,853 hourly partitions and applied 104,859,384 event rows. The complete replay took 4,942.4 seconds (82.4 minutes) with six workers. Its archive fingerprint matched the current archive and its cached snapshots and provenance were verified; this report's coverage refresh reused those snapshots and took 12.9 seconds. Peak memory was not recorded.

| Measure | Count |
|---|---:|
| Gamma markets in scope | 51,458 |
| Official Gamma labels valid | 51,430 |
| Candidate calendar rows | 105,117 |
| Candidate rows with exact Gamma condition ID | 51,457 |
| Candidate rows without Gamma condition ID | 53,660 |
| Gamma markets without a saved candidate row | 1 |
| Candidate rows missing any A/B/C/D probability | 0 |
| Matched Gamma markets with saved prediction-ready timestamp | 0 |
| At least one native fixed-$5 T−60 BBO side | 9,250 |
| Both native fixed-$5 T−60 BBO sides | 5,688 |
| At least one native fixed-$5 T−59 depth side | 9,333 |
| Joint both-sided T−60 and T−59 source | 5,685 |
| Valid-label, prediction-present, joint-source rows | 5,684 |

The 5,684 rows with labels and source coverage are 11.05% of valid Gamma labels. This is a retrospective source-conditional count, not a proven causal or tradable cohort: per-row readiness is unknown, and D0005 does not establish independent full-pipeline selection/calibration. All 53,660 unmapped candidate rows have A/B/C/D probabilities, but none is mapped to a Gamma market and none can be treated as a trade.

First-exclusion counts for the 51,458 Gamma markets:

| First exclusion | Count |
|---|---:|
| No usable two-sided native T−60 BBO at fixed $5 | 45,743 |
| Source-covered under retrospective assumption | 5,684 |
| Official label invalid or unavailable | 27 |
| No usable native T−59 depth at fixed $5 | 3 |
| Missing saved candidate probabilities/row | 1 |

The 53,660 candidate rows with no Gamma mapping are additional manifest rows with first exclusion `no_gamma_market_mapping`; they are not included in the Gamma-market exclusion totals above.

| Month UTC | Gamma | Valid labels | Both-sided T−60 and T−59 source | Valid-label source rows | Unmapped candidate rows |
|---|---:|---:|---:|---:|---:|
| 2025-07 | 0 | 0 | 0 | 0 | 8,928 |
| 2025-08 | 0 | 0 | 0 | 0 | 8,928 |
| 2025-09 | 0 | 0 | 0 | 0 | 8,640 |
| 2025-10 | 0 | 0 | 0 | 0 | 8,928 |
| 2025-11 | 0 | 0 | 0 | 0 | 8,639 |
| 2025-12 | 3,969 | 3,962 | 0 | 0 | 4,959 |
| 2026-01 | 7,475 | 7,469 | 0 | 0 | 1,452 |
| 2026-02 | 4,889 | 4,887 | 0 | 0 | 3,175 |
| 2026-03 | 8,928 | 8,928 | 0 | 0 | 0 |
| 2026-04 | 8,636 | 8,634 | 462 | 462 | 4 |
| 2026-05 | 8,927 | 8,917 | 2,353 | 2,352 | 1 |
| 2026-06 | 8,634 | 8,633 | 2,870 | 2,870 | 6 |

Additional manifest flags include 3,562 Gamma markets with only one usable native T−60 side; 29,245 T−60 and 29,245 T−59 expected 25-hour source gaps; 29,234 markets at each cutoff without the archive entry hour; 45 documented gaps after book initialization at each cutoff; and 3 T−59 gaps after a fee update. Readiness is explicitly unknown on all 105,118 manifest rows. Side-level rejection reasons and each market's first exclusion/additional flags are in the row-level manifest.

The archive manifest advertises 1,877 development hours, of which 1,853 are local and verified. Twenty-one missing local hours span 2026-04-13 19:00 through 2026-04-14 15:00 UTC and appear in the public PMXT checksum index. Three more advertised local hours, 2026-06-11 04:00–06:00 UTC, are absent locally. No archive data was downloaded.

## E0036 source parity and mismatch investigation

T−59 parity passed: 27,432 fields across 1,143 markets. T−60 parity failed: 20,574 fields were compared across the same markets, with one mismatch in `bbo_at_entry_ask_mismatches` for condition `0xe4d40fb41d0acedbcae8592f00ec3ac0f6e662f89260c11a65443385996537a3`, starting 2026-04-30 07:00 UTC. The new replay records 1 mismatch; the saved entry snapshot records 0, but its saved check count was also zero, so that artifact did not actually verify the BBO comparison.

At the T−60 cutoff, the reconstructed UP ask was 0.51 while the latest processed PMXT-reported UP best ask was 0.52. The archived UP ask level at 0.51 remained in the reconstructed book; its last recorded update before the cutoff was at 06:58:37.664 UTC. Later processed events reported 0.52, but the local stream contains no subsequent removal/update for the 0.51 level before the cutoff. The available data cannot establish whether the cause is an omitted archive event or another source behavior. No replay rule or cutoff was changed to force parity. The mismatch and unresolved source semantics keep the economics gate closed.

## Prediction provenance

The frozen A/B/C/D [candidate manifest](btc_candidate_polymarket_target_20261023_v3_manifest.json) has run identity `ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b`: A is BASE70, B adds simple-memory PCA8, C adds ESN PCA8, and D combines the two additions across 24 model members. The saved probabilities were reused; no inference was run. A/C/D probabilities were regenerated using a common fit-core mask and B uses its existing saved prediction artifact; the exact historical runner revision for every probability artifact is not established. All matched candidate rows have A/B/C/D values, but none has a retained measured prediction-ready timestamp. The old nominal decision-time field is not runtime telemetry. Existing candidate quality notes identify full-pipeline nested selection/calibration independence as unproven; results therefore remain retrospective/conditional.

## Minimum-order unit audit

The local 80,127-row Gamma file has only `order_min_size_shares_current`, always 5.0, and contains no market-metadata capture timestamp. It cannot establish historical market-specific USDC minimums for this interval. Current Polymarket documentation labels Gamma `orderMinSize` in USDC and CLOB book `min_order_size` in shares; the distinction is also raised in the still-open [py-sdk issue #302](https://github.com/Polymarket/py-sdk/issues/302). See [Market Details](https://docs.polymarket.com/market-data/market-details) and [Place Orders](https://docs.polymarket.com/trading/place-orders).

The T−59 helper was corrected after the E0037 source-only run: only an explicitly USD/USDC-labeled Gamma field can override the frozen $5 fallback. A share-labeled field is ignored, and requests below the threshold remain rejected without rounding up. This clarifies D0004 under D0019; it does not backfill historical minimums. The pinned E0037 input hash and result remain unchanged. Any later economics run must freeze the corrected source identity in a controlled protocol/addendum and first pass the parity gate.

## Other source and payout limits

The sampled Obadiaha source had no joint T−60/T−59 market within its 10-second sampling rule across 5,472 March markets. The Kacho within-market tick source had zero matched pre-open T−60/T−59 rows across its 15,682 markets and has no ask depth. Neither source replaces PMXT coverage.

The frozen payout delays remain 60, 300, 1,800, and 7,200 seconds after `max(market end, Gamma resolved_at_utc)`. None was evaluated. Gamma resolution/outcome time is not archived wallet redemption availability; Polymarket describes an UMA proposal and challenge process, including a two-hour challenge period ([resolution help](https://help.polymarket.com/en/articles/13364518-how-are-prediction-markets-resolved)). The prior E0036 economics were not reproduced or changed.

Two deterministic source-only smoke checks passed. The PMXT exact sample was condition `0xfe21d75161d71bec075cedee0f0d6b2cd57beeebbba939ece415da2ef965f978`, 2026-05-18 09:15 UTC; replay reproduced the saved rows, while the saved source-quality checks reject the sample. The Obadiaha sample was condition `0x63aa6cc4782925cf7ea6631422ddcfc0acd1770fb4ae6c046bd4f9af53effa08`, 2026-03-01 20:05 UTC; both native books were 4.095 seconds before their cutoffs and are classified only as a 10-second proxy without event/receive timestamps. No outcome/PnL fields were read and neither smoke sample became a trade.

## Validation and next gate

Focused tests passed: T0009 coverage 10/10 and T−59 reassessment 34/34. Full discovery passed 696/696 in 55.837 seconds. `git diff --check` is required before commit. The local row-level manifest is `data/analysis/polymarket/BTC/btc_t0009_resolution_coverage_20261026_38a6221f0d90/coverage_manifest.parquet` (SHA256 `ea0228ad28dedc1f72f8f51eb35d4332a7e67973e0d1f80346c43b2340e09007`); the JSON summary is tracked beside this report.

Resume T0009 only after the T−60 discrepancy is explained or repaired from source-backed evidence and the full E0036 parity check passes. Preserve the frozen interval and timestamps. Before any economics, freeze the corrected minimum-order source identity and establish or explicitly classify the missing prediction-readiness provenance. Do not use holdout data or compute payout-sensitivity account paths while the parity gate remains failed.
