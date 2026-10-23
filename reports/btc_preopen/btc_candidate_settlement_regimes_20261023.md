# Frozen A/B/C/D quality under historical Polymarket settlement rules — 2026-10-23

## Findings

- The memory addition B does **not** improve A in every settlement regime. On Gamma labels, B−A log-loss improvement is +0.000056 in confirmation single-price and +0.000063 in TWAP60, but −0.000010 in holdout single-price and −0.000178 in TWAP30. Every paired 95% interval includes zero.
- On TWAP60, D is slightly better than B on Gamma labels by +0.000020 log loss, +0.000010 Brier, and +0.000305 AUC. The paired 95% intervals include zero: log loss [−0.000100, +0.000141], Brier [−0.000050, +0.000070], AUC [−0.000848, +0.001404]. The point estimate does not establish an ESN gain.
- The procedure ranking changes across period × regime slices. The confirmation / holdout split and the official target both matter; the aggregate single-price score hides that split. Dates and settlement regimes are confounded, so this is not a causal estimate of TWAP's effect.
- Binance/Gamma disagreements explain the **change in paired ranking** when the outcome is switched. In TWAP60, the B−A Gamma gain is smaller than its Binance gain, while D−B changes sign from a small Binance loss to a small Gamma gain. The weighted attribution assigns that shift to the disagreement rows, as expected because agreement rows have identical labels.
- The sample is too short to resolve the small TWAP60 differences: 15,550 five-minute decisions span 54 calendar days (18 non-overlapping three-day blocks). TWAP30 spans only seven days (two non-overlapping blocks). The pointwise intervals remain broad; repeated five-minute decisions are not independent historical replications.

## Frozen sample and rule assignment

Only the saved A/B/C/D ensembles and official Gamma labels were scored. A is BASE70; B is BASE70 + simple memory PCA8; C is BASE70 + ESN PCA8; D is BASE70 + memory + ESN PCA8. The Binance label remains the saved COIN-M BTCUSD proxy. No model, calibration, threshold, seed, or policy was changed. Both periods have prior exposure and this analysis is retrospective, not a new independent test or a tuning set.

The saved prediction ledger contains 133,340 rows. Gamma matched 79,678; 79,650 have a valid, one-hot official label. The remaining 28 matched rows are excluded for the already recorded reasons: 14 not closed or missing resolution time, 13 ambiguous outcomes/prices, and one nonbinary/ambiguous outcome-price record. All 79,650 official rows have finite A/B/C/D predictions on the same common mask, a unique `(condition_id, market_start_utc)` key, and one matching row in the existing assignment ledger. The counts reconcile exactly to the previous 79,650 official labels: 51,430 confirmation and 28,220 holdout. There are 4,193 Binance/Gamma disagreements.

The supplied rule calendar is single price before 2026-08-07 00:00 UTC, TWAP30 from that instant through 2026-08-14 00:00 UTC, and TWAP60 thereafter. All row assignments match those boundaries; no unknown or conflicting calendar assignment occurs. Seven market pages were checked directly and match the calendar. The other 79,643 rows are calendar-assigned because the local Gamma archive does not retain per-market rule text; the seven pages are not evidence that every market was individually verified.

| Group | Requested interval, UTC `[start,end)` | Observed first → last market start | Markets; active / span days | UP Binance / Gamma | Binance–Gamma disagreements | Direct page rows; verified execution rows |
|---|---|---|---:|---:|---:|---:|
| Confirmation / single price | 2025-07-01 → 2026-07-01 | 2025-12-18 04:25 → 2026-06-30 23:50 | 51,430; 179 / 195 | 50.303% / 50.319% | 1,980 (3.85%) | 0; 1,143 |
| Holdout / single price | 2026-07-01 → 2026-08-07 | 2026-07-01 00:00 → 2026-08-06 23:55 | 10,654; 37 / 37 | 49.446% / 49.690% | 276 (2.59%) | 2; 0 |
| Holdout / TWAP30 | 2026-08-07 → 2026-08-14 | 2026-08-07 00:00 → 2026-08-13 23:55 | 2,016; 7 / 7 | 49.405% / 50.546% | 175 (8.68%) | 3; 0 |
| Holdout / TWAP60 | 2026-08-14 → 2026-10-07 | 2026-08-14 00:00 → 2026-10-06 23:50 | 15,550; 54 / 54 | 50.193% / 50.090% | 1,762 (11.33%) | 2; 0 |
| All / single price | 2025-07-01 → 2026-08-07 | 2025-12-18 04:25 → 2026-08-06 23:55 | 62,084; 216 / 232 | 50.156% / 50.211% | 2,256 (3.63%) | 2; 1,143 |

“Active / span days” distinguishes UTC dates with at least one scored market from all calendar days between the first and last scored market. Missing-label dates remain empty days in the time-block bootstrap. The all-single-price row is descriptive and does not replace the separate confirmation and holdout results.

## Quality by outcome target

Each model cell is `log loss / Brier / AUC`; lower losses and higher AUC are better. Constant `p=0.5` is fixed without fitting on any evaluated labels and scores 0.693147 / 0.250000 / 0.500000 in every group. Every group has both outcome classes, so AUC is defined.

### Official Gamma outcome

| Group (n) | A | B | C | D |
|---|---:|---:|---:|---:|
| Confirmation / single price (51,430) | 0.692214 / 0.249533 / 0.525334 | 0.692158 / 0.249506 / 0.525512 | 0.692141 / 0.249497 / 0.525797 | 0.692202 / 0.249528 / 0.524997 |
| Holdout / single price (10,654) | 0.693465 / 0.250157 / 0.512257 | 0.693475 / 0.250163 / 0.511194 | 0.693348 / 0.250099 / 0.512118 | 0.693307 / 0.250079 / 0.512067 |
| Holdout / TWAP30 (2,016) | 0.691090 / 0.248973 / 0.540207 | 0.691268 / 0.249062 / 0.537162 | 0.691434 / 0.249144 / 0.536180 | 0.691243 / 0.249049 / 0.537071 |
| Holdout / TWAP60 (15,550) | 0.692103 / 0.249479 / 0.525683 | 0.692041 / 0.249448 / 0.525709 | 0.692066 / 0.249460 / 0.525811 | 0.692021 / 0.249438 / 0.526014 |
| All / single price (62,084) | 0.692428 / 0.249640 / 0.523151 | 0.692384 / 0.249618 / 0.523140 | 0.692348 / 0.249601 / 0.523537 | 0.692391 / 0.249622 / 0.522890 |

### Saved Binance proxy outcome

| Group (n) | A | B | C | D |
|---|---:|---:|---:|---:|
| Confirmation / single price (51,430) | 0.691912 / 0.249383 / 0.528337 | 0.691877 / 0.249366 / 0.528322 | 0.691859 / 0.249357 / 0.528785 | 0.691927 / 0.249391 / 0.527812 |
| Holdout / single price (10,654) | 0.693260 / 0.250055 / 0.514173 | 0.693309 / 0.250080 / 0.512756 | 0.693176 / 0.250014 / 0.513531 | 0.693162 / 0.250007 / 0.513420 |
| Holdout / TWAP30 (2,016) | 0.691168 / 0.249012 / 0.538514 | 0.691347 / 0.249101 / 0.535398 | 0.691597 / 0.249225 / 0.533784 | 0.691452 / 0.249154 / 0.534098 |
| Holdout / TWAP60 (15,550) | 0.692310 / 0.249581 / 0.524201 | 0.692148 / 0.249501 / 0.525044 | 0.692243 / 0.249548 / 0.524240 | 0.692176 / 0.249515 / 0.524680 |
| All / single price (62,084) | 0.692144 / 0.249499 / 0.525985 | 0.692123 / 0.249488 / 0.525761 | 0.692085 / 0.249470 / 0.526287 | 0.692139 / 0.249496 / 0.525474 |

## Paired differences and uncertainty

For log loss and Brier, `Δ = loss(left) − loss(right)`; for AUC, `Δ = AUC(right) − AUC(left)`. Positive values favor the right-hand procedure. The intervals below are pointwise percentile 95% intervals from 2,000 paired circular moving-block resamples of whole UTC days, using three-day blocks. Each group has its own draws, shared across procedures and both targets. No block crosses a settlement regime; no observation outside the evaluated group enters its interval. These are descriptive intervals, not formal significance declarations; no multiple-comparison-adjusted claims are made.

| Group | Comparison | Gamma ΔLL [95% interval] | Gamma ΔBrier [95% interval] | Gamma ΔAUC [95% interval] |
|---|---|---:|---:|---:|
| Confirmation / single price | B−A | +0.000056 [−0.000068, +0.000176] | +0.000028 [−0.000034, +0.000088] | +0.000178 [−0.001109, +0.001415] |
| Confirmation / single price | D−B | −0.000044 [−0.000125, +0.000042] | −0.000022 [−0.000062, +0.000021] | −0.000515 [−0.001371, +0.000355] |
| Holdout / single price | B−A | −0.000010 [−0.000331, +0.000318] | −0.000005 [−0.000165, +0.000158] | −0.001064 [−0.004676, +0.002870] |
| Holdout / single price | D−B | +0.000168 [−0.000026, +0.000366] | +0.000084 [−0.000013, +0.000182] | +0.000874 [−0.001287, +0.003005] |
| Holdout / TWAP30 | B−A | −0.000178 [−0.000617, +0.000295] | −0.000089 [−0.000308, +0.000148] | −0.003045 [−0.012173, +0.006761] |
| Holdout / TWAP30 | D−B | +0.000025 [−0.000167, +0.000205] | +0.000013 [−0.000083, +0.000102] | −0.000090 [−0.004014, +0.003958] |
| Holdout / TWAP60 | B−A | +0.000063 [−0.000169, +0.000268] | +0.000031 [−0.000084, +0.000133] | +0.000026 [−0.002042, +0.001881] |
| Holdout / TWAP60 | D−B | +0.000020 [−0.000100, +0.000141] | +0.000010 [−0.000050, +0.000070] | +0.000305 [−0.000848, +0.001404] |
| All / single price | B−A | +0.000044 [−0.000076, +0.000160] | +0.000022 [−0.000038, +0.000080] | −0.000011 [−0.001263, +0.001209] |
| All / single price | D−B | −0.000007 [−0.000090, +0.000077] | −0.000004 [−0.000045, +0.000038] | −0.000250 [−0.001066, +0.000605] |

The auxiliary C−A and D−A comparisons, plus the same paired intervals on the Binance target, are retained in the machine-readable output. In every row shown above the interval includes zero. TWAP30 has only seven possible circular block starts and two non-overlapping three-day blocks; its 2,016 five-minute rows do not provide a large independent sample. TWAP60 has 54 possible starts and 18 non-overlapping blocks.

## Separating time and target changes

The same frozen probabilities were scored against each row's Binance and Gamma labels. On TWAP60, B−A log-loss improvement is +0.000161 on Binance but +0.000063 on Gamma. The target switch changes the delta by −0.000098; the disagreement rows contribute that entire change, while agreeing labels contribute zero. On those 1,762 disagreement rows, B−A changes from +0.000615 on Binance to −0.000254 on Gamma.

For D−B on TWAP60, the Binance-target delta is −0.000028 and the Gamma-target delta is +0.000020. The +0.000047 target-shift contribution comes from disagreement rows: their mean D−B delta is −0.000079 on Binance and +0.000340 on Gamma. On the 13,788 agreeing rows D−B is −0.000021 for either label. Thus the apparent sign change is associated with the changed label, but this does not show why Chainlink and Binance labels differ; Chainlink report prices were not fetched.

The regime rankings also vary by date. TWAP60 monthly Gamma B−A log-loss improvements are −0.000165 in partial August (Aug 14–31), +0.000079 in September, and +0.000660 in partial October (through Oct 6); D−B is −0.000067, +0.000080, and −0.000020. Fixed UTC weekly diagnostics show further sign changes. The Aug 10–17 and Oct 5–12 weeks are marked partial in the output; the latter includes only Oct 5–6 within this holdout.

Across TWAP60, mean probability / standard deviation and mean logit / standard deviation are: A 0.500966 / 0.028291 and 0.003870 / 0.113488; B 0.501811 / 0.026509 and 0.007263 / 0.106311; C 0.501636 / 0.025493 and 0.006556 / 0.102208; D 0.501647 / 0.025304 and 0.006603 / 0.101456. The fixed ten-bin calibration diagnostic has populated bins only in [0.4, 0.5) and [0.5, 0.6); its descriptive ECE is 0.005606, 0.004397, 0.003253, and 0.002261 for A/B/C/D. No calibrator was fit. Per-member results are auxiliary only: A/B each have three saved members and C/D nine; the ensemble averages reconcile exactly to those members, which are not treated as independent historical repetitions.

## Existing execution sample

The existing, validated execution ledger contains 1,143 markets from 2026-04-15 17:35 through 2026-05-18 09:00 UTC. All are confirmation / single-price: 1,143 of 51,430 labels (2.22%); there are zero execution-ledger markets in holdout single-price, TWAP30, or TWAP60. Do not recalculate economics from this grouping. The previously reported +$28.01 for B and +$22.66 for D therefore do **not** evaluate the economics of the current TWAP60 settlement rule. Missing TWAP60 book coverage does not block the prediction metrics above.

## Limits and next step

The calendar rule changes coincide with time, market conditions, and a different label construction. The seven checked market pages support the boundary assignment but do not validate all 79,650 individual market rules. The observed label disagreements are not attributed to Chainlink price values without the exact historical reports. Holdout outcomes have previously been inspected. The pointwise intervals are wide, particularly for TWAP30, and repeated five-minute markets cluster within days. No result here proves a causal TWAP effect, new independent confirmation, or economic advantage.

**Correction to the earlier next-step duration:** the 180-day minimum was arbitrary and is withdrawn. The collector implementation, historical L2 source review, one-market Vike L2 reconstruction, and blocked live preflight are documented in the [shadow source and pilot report](btc_candidate_shadow_sources_and_pilot_20261023.md). Its precision plan pre-registers a $0.10 per-$5 primary MDE, Bonferroni family-wise alpha 0.05 over A-vs-cash and D-vs-B, and one final look after the transaction and three-day-block thresholds; those thresholds are calculated from the old sparse execution sample and are explicitly not representative of TWAP60. No economic interim looks or model/policy changes are allowed.

## Reproducibility artifacts

- Analysis extension: [`extend_btc_candidate_attribution_regimes_20261023.py`](../../extend_btc_candidate_attribution_regimes_20261023.py)
- Local output directory: `data/analysis/polymarket/BTC/btc_candidate_settlement_regimes_20261023/`
- Outputs: `common_settlement_regime_market_table.parquet`; `quality_by_period_regime_target_procedure.csv`; `paired_differences_block_bootstrap.csv`; `target_shift_attribution.csv`; `monthly_diagnostics.csv`; `twap60_weekly_diagnostics.csv`; `fixed_bin_calibration.csv`; `member_seed_auxiliary_metrics.csv`; `execution_coverage.csv`; `analysis_manifest.json`.
- Inputs: the saved `target_observation_ledger.parquet`, existing 1,143-row `economic_attribution_ledger.parquet`, the row-level calendar assignment ledger, the prior rule audit report, and the frozen candidate-attribution protocol. Input SHA-256 hashes are in the local analysis manifest.
- Validation: 79,650 rows and unique market keys reconcile; the period × regime partition sums exactly; all ensemble member means reconcile; AUC ties, single-class NA, delta signs, seven-day bootstrap bounds, and target-shift contribution identities pass. An independent scikit-learn check matches the TWAP60 A point AUC and a paired bootstrap replicate.
- Runtime: 68.5 seconds active; sampled peak working set 727,691,264 bytes (694 MiB). No model fitting, refit, Chainlink fetch, book replay, live activation, or economic recomputation occurred.
