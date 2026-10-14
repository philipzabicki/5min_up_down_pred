# BTC-plus-market prospective test (2026-10-14)

## Status

**Deferred — not started.** The protocol and public collector are preserved as frozen historical artifacts. The 240-day prospective collection has **not started** and is deferred by the 2026-10-15 modeling task. The public smoke is operational validation and is excluded from the prospective sample. The protocol freeze time is `2026-10-14T03:26:18Z`. No prospective start date or evaluation calendar date exists.

The historical information-value result remains “no demonstrated incremental information,” not proof of no information. `A_frozen` remains historically selected; `A_reoptimized` did not demonstrate improvement on official holdout labels; B remains incomplete and was not resumed; the historical holdout was exposed and is excluded from fitting, planning power, and the new sample. The prior conditional information analysis covered `A_frozen` only, not both A variants.

## Question and frozen comparisons

The primary question is whether the frozen market-plus-BTC probability `D` improves paired log loss over raw `p_market` on new common markets. The primary difference is `D − raw p_market`; negative favors D. Raw market is the primary benchmark because the earlier market-only logistic correction scored worse than raw market.

The secondary comparison is `D − C_market_only`, which tests BTC's contribution under the same logistic model form. `A_frozen` is a descriptive benchmark. The coefficients, feature inputs, calibrator, and source identities are fixed in [the protocol](../configs/research/btc_prospective_test_20261014.json) and [the frozen artifact manifest](../reports/btc_prospective_test_20261014/frozen_artifact_manifest.json). No additional model search or historical holdout fit is part of this test.

## Duration and planning power

Development planning used 8,676 paired OOF markets across 61 calendar days, of which 45 had at least one row; 16 days were empty. Coverage was 49.4% of scheduled five-minute slots, or 142.2 markets per calendar day and 192.8 per covered day. The exposed holdout did not inform the planned power.

Power is based on centered paired daily loss differences and circular moving-block resampling of 1, 3, or 7 UTC days (5,000 replicates, seed `20261014`, two-sided α=0.05). The effect grid is a planning scenario, not an assumption about the true effect or an economic threshold.

| Primary D minus raw market | Block | Days for 80% power | Days for 90% power |
|---:|---:|---:|---:|
| 0.00025 | 1 day | 2,585 | 3,566 |
| 0.00025 | 3 days | 2,324 | 3,164 |
| 0.00025 | 7 days | 1,651 | 2,176 |
| 0.00050 | 1 day | 674 | 880 |
| 0.00050 | 3 days | 552 | 733 |
| 0.00050 | 7 days | 384 | 532 |
| 0.00100 | 1 day | 169 | 227 |
| 0.00100 | 3 days | 147 | 195 |
| 0.00100 | 7 days | 100 | 145 |

The fixed 240-calendar-day horizon is aimed at an improvement around 0.001: estimated power is 91.6%, 94.7%, and 98.7% at block lengths of 1, 3, and 7 days. At 0.0005, estimated power is 40.0%, 44.8%, and 61.9%; at 0.00025, it is 13.2%, 15.2%, and 22.8%. A 240-day test should therefore not be presented as a reliable test of very small improvements. The [power report](../reports/btc_prospective_test_20261014/power_and_observation_plan.md) shows primary and secondary MDEs at 30, 60, and 90 actual coverage days, plus the separate 240-calendar-day power table.

These are uncertain planning estimates: development has only 61 calendar days, contains a long coverage gap, and provides few independent time blocks. Regime shifts, serial dependence, and future coverage can differ. Resampling far beyond the development period is not a precise success-date forecast. Log-loss improvements are not mechanically converted to profit.

The paired development mean's moving-block bootstrap variability is below. Variance and standard error refer to the 61-calendar-day mean paired log-loss difference; they are planning estimates from only 8,676 development markets, not a prospective result.

| Comparison | Block | Variance of mean difference | Bootstrap SE |
|---|---:|---:|---:|
| D minus raw market | 1 day | 3.50e-7 | 0.000591 |
| D minus raw market | 3 days | 3.22e-7 | 0.000568 |
| D minus raw market | 7 days | 2.48e-7 | 0.000498 |
| D minus market-only C | 1 day | 3.32e-7 | 0.000576 |
| D minus market-only C | 3 days | 2.64e-7 | 0.000514 |
| D minus market-only C | 7 days | 1.94e-7 | 0.000440 |

## Fixed evaluation rules

- Observe exactly 240 elapsed calendar days from the first scheduled market recorded by the collector. The deadline does not move.
- After that boundary, allow a fixed 24-hour grace to retrieve official outcomes only for markets that started before the deadline. Do not add markets or replace missing ones.
- Score the primary paired log-loss difference with a two-sided 95% circular moving-block bootstrap using 3 UTC-day blocks, 5,000 replicates, and the registered seed.
- Call the primary result **positive** if the entire 95% interval is below zero, **negative** if the entire interval is above zero, and **inconclusive** otherwise. A coverage-gate failure is inconclusive. The secondary result cannot override the primary.
- Require attempt records for at least 90% of scheduled markets, at least 168 UTC dates with 100 or more common quality markets, and at least 32,000 common quality markets. Keep late and missing cycles in the operational denominator. Quality eligibility does not depend on the frozen policy selecting a trade or a T−59 quote being executable.
- Do not inspect significance while collecting. Do not extend the horizon or replace observations because a result is near a threshold.

Brier score, descriptive calibration bias and fixed 10-bin ECE, and simulated economics are secondary. Calibration is not refit. Economics are separate: `A_frozen` chooses the side at T−60 using the frozen 1% net-EV hurdle; the same side is assessed on the T−59 public book under the fixed fee, depth, and minimum-order rules. Report independent USD 5 orders and the separate 5%-of-free-cash portfolio with no credit and unchanged minimum size. Shadow observations cannot establish actual ACKs or fills; execution remains a simulation on observed books.

## Collector and run command

The collector warms from the verified frozen BTC bundle, connects the configured public Binance index and futures candle streams, and REST-backfills through the latest closed candle after the feeds connect and before scheduling the first market. It pairs subsequent closed candles with source and receive times, waits briefly for the paired WebSocket candle at T−60, and uses public REST to repair remaining gaps. At nominal T−60 it records both Polymarket books, uses the candle opened at market start minus two minutes, generates all frozen probabilities, and records the same-side T−59 execution snapshot. It measures request, feed wait, feature, inference, policy, and total decision latency against the one-second T−59 availability budget.

Market attempts and official outcomes are append-only in a WAL SQLite database. A unique market key prevents duplicate attempts on resume; collector checkpoints persist the predictor state. Only public reads are present in this path. It has no order submission, cancellation, or signing capability and requires no private credentials.

From the repository root, start or resume the single collector with:

```powershell
python run_btc_prospective_shadow.py
```

Do not run a second copy against the same database. The collector records the frozen protocol hash and full artifact hash manifest in its session metadata. The development-only smoke is kept outside that database.

## Public smoke

The committed [three-market smoke summary](../reports/btc_prospective_test_20261014/public_smoke_summary.json) reports probabilities, actual measured latency, official outcomes, and exceptions; the full public capture is stored under `data/analysis/polymarket/BTC/btc_prospective_test_20261014/smoke/`. The run is excluded from the prospective quality sample. It uses public market metadata, books, and BTC candles only; orders and private credentials are disabled. The third official outcome arrived after the smoke's built-in polling deadline and was attached by a separate public read with its actual retrieval time.

## Reproduction and source files

- Power analysis: `python analyze_btc_prospective_power_20261014.py`
- Final scoring after collection and the settlement grace: `python analyze_btc_prospective_results_20261014.py`
- Public smoke (development validation only): `python smoke_btc_prospective_public_20261014.py`
- Machine-readable protocol: [btc_prospective_test_20261014.json](../configs/research/btc_prospective_test_20261014.json)
- Artifact hashes: [frozen_artifact_manifest.json](../reports/btc_prospective_test_20261014/frozen_artifact_manifest.json)
- Power calculations: [power_analysis.json](../reports/btc_prospective_test_20261014/power_analysis.json)
