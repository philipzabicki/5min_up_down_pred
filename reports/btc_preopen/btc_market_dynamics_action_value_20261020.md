# BTC market dynamics and action value — 2026-10-20

**Status: `blocked_missing_dynamic_history`.** The development cutoff is exclusive at `2026-07-01T00:00:00Z`.

## Outcome

The frozen pre-fit feasibility minima were not met, so models A/B/C/D, bootstrap comparisons, and economic scoring were not run. This is a feasibility result; it provides no profitability conclusion and selects no model or policy.

## Replay and T−60 reconstruction

- The official development market index ran from 2026-04-15T17:05:00Z through 2026-06-30T23:55:00Z. The replay covered 6,796 static candidates and all 1,409 nonempty event partitions in scope. It selected 39,013,290 event rows; 444 metadata-known empty partitions were skipped. No event rows at or after the cutoff were materialized.
- Replayed T−60 states matched all 6,796 saved snapshots: zero quote, event-count, or book-count mismatches; maximum absolute price difference was 0.
- Complete S+H+RAW history was available for 2,393 markets. The remaining 4,403 candidates were excluded for: 2,751 unresolved order/history windows at 60 s; 992 book resets inside the 60 s depletion window; 659 invalid native quote windows at 60 s; and 1 invalid native quote window at 15 s. Missing and unverified source hours remain excluded; an empty partition was not treated as proof that the book was unchanged.

## Frozen availability gate

The evaluation starts on 2026-05-15 after the 30-day warm-up. Fold 1 covers [2026-05-15T00:00Z, 2026-06-01T00:00Z) and its first T−60 decision is 2026-05-14T23:59Z; fold 2 covers [2026-06-01T00:00Z, 2026-07-01T00:00Z) and its first T−60 decision is 2026-05-31T23:59Z. The evaluation contains 633 markets across 22 observed days, below the required 2,000 evaluation markets; the 20-day minimum passes. Both monthly blocks are nonempty, and their released training labels pass the 1,000-label minimum.

| Fold | Evaluation markets | Observed days | Released training labels | Training markets with known q=$5 targets on both sides |
| --- | ---: | ---: | ---: | ---: |
| 1 (May 15–May 31) | 520 | 12 | 1,760 | 908 / 1,000 required |
| 2 (June) | 113 | 10 | 2,279 | 1,207 / 1,000 required |

The overall evaluation-market minimum and fold 1 q=$5 two-sided training minimum failed. There were 410 evaluation markets with known q=$5 targets on both sides. Official labels were available for 21,962 of 21,962 development index rows and 2,392 of the 2,393 common-input markets. These counts do not change the frozen gate.

**Next allowed step:** only after a separately verified source-history backfill can provide at least 1,367 additional valid evaluation markets and 92 additional pre-fold q=$5 two-sided training markets, rerun this same pre-fit gate with thresholds unchanged. Do not treat unverified empty partitions as unchanged books or relax the minima.

## Frozen model plan (not fitted)

All decisions were defined at T−60 with hypothetical execution at T−59. The frozen size grid was q = {$5, $10, $25, $50, $100}; q=$5 was the primary action. Static S has 10 features: RAW and market logits and their difference, both spreads, midpoint-sum deviation, ask-band notionals, and $5 slippage for both sides. Dynamic H has exactly 21 features: seven quote-change/depletion features at each of 5, 15, and 60 s.

A uses S to classify the official UP outcome; B uses S plus action descriptors to regress net action return; C uses S+H for the official UP outcome; D uses S+H plus the same action descriptors for net action return. The fixed controls were RAW BASE70, raw market probability, and no-trade. Because the frozen availability gate failed, none of these fits or controls were scored. There are no probability, economic, bootstrap, portfolio-sizing, stress, or promotion-gate results from this run.

## Scope and limitations

RAW BASE70 is conditional OOF following earlier development selection, so this experiment would not be an independent nested evaluation of the full BTC feature/model-selection pipeline. No holdout market rows were read or scored. A pre-existing historical holdout summary was incidentally visible in the status document before this work and was not used for any decision or analysis.

`holdout not read/scored` (no holdout market rows or scoring; see the historical-summary exposure note above). `live unchanged`. `orders not submitted`.

## Compute

The full four-worker development event replay took 2,417.5 s (40.3 min) for 39.0 million selected rows. No model fitting ran. The runner did not capture a numeric peak working set; a PowerShell sample during replay showed about 41.6 GB system memory available and worker processes around 0.4 GB each.

## Artifacts

- Frozen protocol: `data/analysis/polymarket/BTC/btc_market_dynamics_action_value_20261020/frozen_protocol.json`
- Versioned summary manifest: `reports/btc_preopen/btc_market_dynamics_action_value_20261020_manifest.json`
- Availability and fold counts: `data/analysis/polymarket/BTC/btc_market_dynamics_action_value_20261020/availability.json`
- Run manifest and T−60 reconstruction metrics: `data/analysis/polymarket/BTC/btc_market_dynamics_action_value_20261020/run_manifest.json`
- Large development-only tables and replay checkpoints are under the configured `data/analysis/...` output directory and are not staged.
