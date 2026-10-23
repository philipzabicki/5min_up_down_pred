# Frozen A/B/C/D on official Polymarket BTC 5m outcomes — 2026-10-23

## Result

The saved procedures do not show a consistent, statistically supported improvement on official Polymarket outcomes. B is slightly better than A by log loss in both periods, while D is worse than B in confirmation and better in holdout. Every prespecified Bonferroni-adjusted 97.5% interval for the two primary comparisons includes zero, on both the full official-label sample and the T−59 execution-quality subset.

The only priceable economic comparison is a sparse confirmation-period diagnostic over 1,143 markets from 2026-04-15 17:35 UTC through 2026-05-18 09:00 UTC. On that sample D produced 380 trades and +$22.66 independent-trade PnL; B produced 396 trades and +$28.01. D changed 86 of 1,143 decisions relative to B and those changes reduced PnL by $5.35. There are no exact saved T−60 books in the holdout, so there is no holdout economic result or portfolio claim.

These results are retrospective. Both periods were previously exposed. T−60 availability timestamps for the saved A/B/C/D predictions are not retained, and the T−60 books contain only native best asks and displayed size, not full ladders. The economic figures assume the predictions were available at T−60 and are not a full-coverage execution backtest or evidence of live readiness.

## Frozen scope and procedure

- Run identity: ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b. A=BASE70; B=BASE70 + simple memory PCA8; C=BASE70 + ESN PCA8 (auxiliary); D=BASE70 + memory + ESN PCA8. No fitting, refitting, calibration, feature replay, ESN search, or parameter changes were performed.
- Periods are bounded by market start: confirmation [2025-07-01, 2026-07-01) UTC and holdout [2026-07-01, 2026-10-07) UTC. The saved rows map Opened+2 minutes to market start and Opened+1 minute to T−60 for all 133,340 predictions.
- Official UP/DOWN outcomes were derived from the saved Gamma BTC 5m series 10684 market index. The join requires exact UTC market start, canonical market slug, unique condition and token IDs, UP/DOWN outcome orientation, one-hot resolved outcome prices, closed status, and resolution at or after market end. Binance proxy labels were never substituted for missing official results.
- The saved target column remains the existing Binance COIN-M proxy; the derived official Polymarket target is stored separately in the market-match artifacts. Both are compared only where an official resolution exists.
- The historical fixed-$5 reference policy is reports/btc_preopen/policy.json. It selects the side with the highest positive expected net payout, breaks exact positive ties toward UP, and otherwise does not trade. Its earlier selection and both evaluation periods are exposed. The binding BTC contract supplies T−60 decision / T−59 execution timing for this diagnostic.
- Decisions use T−60 saved native best ask and displayed size only when that one level can cover the complete $5 gross order. T−59 execution walks the verified native ask ladders and requires a complete $5 fill on both sides. Missing price, depth, fee, or source evidence is excluded from the common economic mask, not counted as a no-trade decision.
- Historical fee calculation uses the saved fee registry and existing replay helper: estimated 0.072 rate before 2026-05-06 and 0.07 thereafter; V1 share fees before the April 28 maintenance window, the archived maintenance pause, then V2 cash fees. The V2 cash fee is additional to the $5 purchase amount; V1 fee shares reduce payout. The rate transition is an estimate. Historical order minimum and tick-size evidence are unavailable; the accepted $5 fallback was used and current Gamma minimums were not projected backward.
- Paired uncertainty uses the existing non-circular 3-calendar-day moving-block bootstrap: 2,000 replicates, seed 20261023, common multiplicities across candidates and pairs. The established Bonferroni correction is applied only to D−B and B−A, not to the prior research history. Calibration bins are fixed deciles and diagnostic only; no calibrator was fitted.

The protocol manifest was frozen before the v3 scoring run. It records hashes for the saved predictions, Gamma and book sources, manifests, policies, fee rules, analysis helpers, 24 frozen model artifacts, and 7 transforms. The separate dynamic-history/RAW OOF blockers CLOSED_BLOCKED_DATA_AND_RAW_PROVENANCE and blocked_missing_dynamic_history remain active for that experiment. They do not block scoring these saved predictions against available official labels; they continue to limit claims about underlying historical raw-source availability and live readiness.

## Coverage and target agreement

| Market-start period | Saved predictions | Unique Gamma matches | Official resolutions | Matched without official label | T−59 snapshots matched to official rows | Provenance-qualified two-sided full $5 fills | Existing-policy T−59 subset (≤30s age and clean aggregate BBO) | Exact saved T−60 snapshots | T−60 rows passing fixed-$5 checks | Common T−60/T−59 economic rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Confirmation | 105,117 | 51,457 | 51,430 | 27 | 21,951 | 5,609 | 4,824 | 9,407 | 4,066 | 1,143 |
| Holdout | 28,223 | 28,221 | 28,220 | 1 | 28,220 | 17,564 | 17,497 | 0 | 0 | 0 |

The 27 confirmation exclusions are 14 markets not closed or without a resolution time and 13 with ambiguous outcome labels/prices. The one holdout exclusion has non-binary outcome prices. Unmatched predictions are 53,660 in confirmation and 2 in holdout. All saved prediction rows meet the T−60 timing mapping, but that mapping does not establish the runtime availability of each prediction.

The prior Binance target differs from the official Polymarket label on 1,980/51,430 confirmation rows (3.85%) and 2,213/28,220 holdout rows (7.84%). This target change is kept distinct from the sample selection caused by book coverage.

The T−59 replay contains 50,184 snapshots, all marked t59_native_asks with entry_time exactly market_start−59 seconds; its replay manifest verifies provenance and snapshot integrity. The stricter quality subset additionally requires source coverage without gaps, no future event, a native and unambiguous two-sided book, known fees, known complete ask depth, and complete $5 fills. It covers 10.91% of official confirmation labels and 62.24% of official holdout labels. The different time coverage and selection rates mean the T−59 subset is not interchangeable with the full period.

The saved T−60 artifact covers 9,407 markets from 2026-04-15 through 2026-05-18. It retains best ask and displayed size rather than full ladders, so only the single-level $5 cases can be repriced with the accepted fee helper. The adjacent older T−59 BBO matches the verified replay on both native tokens for 9,400/9,407 IDs; 7 are missing or do not reconcile. This source boundary leaves no T−60 book for holdout.

## Official-label quality

Metrics use the same rows for A/B/C/D. Higher AUC and lower log loss/Brier are better. All numbers are retrospective on previously exposed data.

| Period and sample | n | Model | Log loss | Brier | AUC |
|---|---:|:---:|---:|---:|---:|
| Confirmation, all official labels | 51,430 | A | 0.692214 | 0.249533 | 0.525334 |
|  |  | B | 0.692158 | 0.249506 | 0.525512 |
|  |  | C | 0.692141 | 0.249497 | 0.525797 |
|  |  | D | 0.692202 | 0.249528 | 0.524997 |
| Confirmation, T−59 qualified subset | 5,609 | A | 0.692120 | 0.249487 | 0.526096 |
|  |  | B | 0.692061 | 0.249458 | 0.527027 |
|  |  | C | 0.692188 | 0.249521 | 0.524961 |
|  |  | D | 0.692148 | 0.249502 | 0.524863 |
| Holdout, all official labels | 28,220 | A | 0.692545 | 0.249699 | 0.521425 |
|  |  | B | 0.692527 | 0.249690 | 0.520943 |
|  |  | C | 0.692505 | 0.249679 | 0.521348 |
|  |  | D | 0.692451 | 0.249652 | 0.521538 |
| Holdout, T−59 qualified subset | 17,564 | A | 0.692972 | 0.249912 | 0.516874 |
|  |  | B | 0.692944 | 0.249898 | 0.516293 |
|  |  | C | 0.692914 | 0.249883 | 0.516402 |
|  |  | D | 0.692840 | 0.249846 | 0.516856 |

Primary paired log-loss deltas follow the frozen convention: D−B = LL(B)−LL(D), B−A = LL(A)−LL(B). Positive values favor the first model named in the delta.

| Period and sample | D−B delta LL | D−B corrected 97.5% interval | B−A delta LL | B−A corrected 97.5% interval |
|---|---:|---:|---:|---:|
| Confirmation, all official labels | −0.000043842 | [−0.000138141, +0.000055194] | +0.000055587 | [−0.000084719, +0.000199606] |
| Confirmation, T−59 qualified subset | −0.000087530 | [−0.000387361, +0.000192921] | +0.000059470 | [−0.000241573, +0.000445013] |
| Holdout, all official labels | +0.000076370 | [−0.000042918, +0.000183945] | +0.000018034 | [−0.000199836, +0.000197347] |
| Holdout, T−59 qualified subset | +0.000104068 | [−0.000060901, +0.000244040] | +0.000028256 | [−0.000261622, +0.000275136] |

Every corrected interval crosses zero. On the full confirmation sample, C has the best point log loss, Brier, and AUC; on holdout, D has the best point values for all three. D−B changes sign between periods. Restricting to T−59-qualified markets also changes the point rankings, but does not produce a supported primary improvement.

### Fixed-decile calibration

Each cell is count; mean predicted UP probability → observed UP rate. Only the [0.4, 0.5) and [0.5, 0.6) bins contain predictions; the remaining fixed deciles are empty. Bins are left-closed/right-open, with the final [0.9, 1.0] bin including 1.0. Calibration was not fitted.

| Period/sample | Bin | A | B | C | D |
|---|---|---|---|---|---|
| Confirmation, all | [0.4, 0.5) | 24,864; .47714→.48488 | 24,402; .47871→.48496 | 24,754; .47926→.48509 | 24,704; .47958→.48603 |
|  | [0.5, 0.6) | 26,566; .52348→.52033 | 27,028; .52265→.51965 | 26,676; .52170→.51998 | 26,726; .52153→.51905 |
| Confirmation, T−59 subset | [0.4, 0.5) | 2,658; .47825→.47705 | 2,582; .47876→.47676 | 2,648; .47951→.47998 | 2,648; .47949→.47885 |
|  | [0.5, 0.6) | 2,951; .52151→.51372 | 3,027; .52087→.51305 | 2,961; .52034→.51098 | 2,961; .52027→.51199 |
| Holdout, all | [0.4, 0.5) | 13,756; .47923→.48401 | 13,473; .48052→.48319 | 13,554; .48096→.48384 | 13,541; .48124→.48527 |
|  | [0.5, 0.6) | 14,464; .52190→.51466 | 14,747; .52088→.51482 | 14,666; .52021→.51439 | 14,679; .51993→.51305 |
| Holdout, T−59 subset | [0.4, 0.5) | 8,618; .47838→.48584 | 8,405; .47985→.48507 | 8,436; .48025→.48589 | 8,430; .48064→.48790 |
|  | [0.5, 0.6) | 8,946; .52241→.51073 | 9,159; .52121→.51086 | 9,128; .52053→.51019 | 9,134; .52018→.50832 |

## Economic diagnostic

This is the common official-label/T−60/T−59 sample only: 1,143 confirmation markets, all with the same prices, fee rules, fixed-$5 policy, and financing assumptions across A/B/C/D. It is not a full-period backtest. A/B/C/D prediction availability at T−60 is assumed because a per-prediction availability timestamp is absent.

### Independent $5 trades

| Model | Trades | Gross turnover | Cash fees | V1 share-fee equivalent | Total fee equivalent | Sum of T−60 expected net PnL | Realized PnL | Mean PnL/trade | PnL/turnover |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 424 | $2,120 | $48.09 | $26.62 | $74.71 | +$58.35 | −$108.80 | −$0.257 | −5.13% |
| B | 396 | $1,980 | $44.52 | $25.38 | $69.90 | +$51.10 | +$28.01 | +$0.071 | +1.41% |
| C | 375 | $1,875 | $43.82 | $22.38 | $66.19 | +$46.73 | +$20.42 | +$0.054 | +1.09% |
| D | 380 | $1,900 | $43.16 | $23.95 | $67.11 | +$47.07 | +$22.66 | +$0.060 | +1.19% |

The expected value uses the frozen probability and T−60 ask cost; realized PnL uses the chosen side’s T−59 net shares, T−59 cash debit, and official Gamma result. V1 share fees are shown as a cash-equivalent amount for comparison; they reduce shares and are not an extra cash debit. No bootstrap treats these trades as independent.

### D versus B decisions and PnL attribution

On the 1,143 common markets, D and B make the same decision 1,057 times and differ 86 times. They share 345 same-side trades; there are no different-side trades. D adds 35 trades with +$16.00 PnL, while D omits 51 B trades that earned +$21.35, for −$5.35 incremental PnL from changed participation. B trades 396 times and D 380 times. This diagnostic therefore does not show D improving economic decisions over B.

### Self-financing $100 accounts

Each model starts a separate account with $100, spends the same $5 gross order plus any V2 cash fee, receives no credit, and releases payout at max(Gamma resolved time, market end)+60 seconds. All accepted trades settle before the sample period ends under that idealized release rule; actual redemption latency is not archived.

| Model | Accepted trades | Rejected for insufficient cash | Free cash/equity at period end | Unsettled positions |
|:---:|---:|---:|---:|---:|
| A | 285 | 139 | $4.58 | 0 |
| B | 396 | 0 | $128.01 | 0 |
| C | 375 | 0 | $120.42 | 0 |
| D | 380 | 0 | $122.66 | 0 |
| Cash hold | 0 | 0 | $100.00 | 0 |

The A account runs short of cash after realized losses and skips 139 otherwise valid signals. B, C, and D keep cash nonnegative in this sparse sample. None of these account outcomes estimates a full-period capital path; the T−60 source ends before holdout begins.

### PnL concentration

| Model | Largest absolute trade PnL | Top 5 trades’ share of absolute PnL | Top 5 days’ share of absolute PnL |
|:---:|---:|---:|---:|
| A | $5.24 | 1.26% | 5.60% |
| B | $5.91 | 1.39% | 6.68% |
| C | $5.91 | 1.47% | 6.71% |
| D | $5.91 | 1.45% | 6.14% |

Daily PnL contributions are retained in the local ledger. These are concentration descriptions, not a bootstrap or a full uncertainty analysis of the portfolio path.

## Artifacts, limits, and conclusion

- Frozen protocol: reports/btc_preopen/btc_candidate_polymarket_target_20261023_v3_manifest.json; SHA-256 16e66363f35390d520e5a3ca7ce58fa07689ac15e66244a313c50f1a29017503.
- Report: reports/btc_preopen/btc_candidate_polymarket_target_20261023.md.
- Local outputs: data/analysis/polymarket/BTC/btc_candidate_polymarket_target_20261023/quality_summary.json, economic_summary.json, confirmation_market_matches.parquet, holdout_market_matches.parquet, economic_decision_ledger.parquet, self_financing_portfolio_ledger.csv, and daily_pnl_contributions.csv. Large data ledgers remain local and are not versioned.
- Runner: run_btc_candidate_polymarket_target_20261023.py. Completed in 101.62 seconds; sampled peak working set was 1.80 GiB. Python syntax compilation and Git whitespace validation passed; no test suite or additional model run was launched.

Memory improves on A only by very small and inconsistent point differences. Adding ESN to memory (D versus B) does not produce a supported official-target quality improvement, and D underperforms B on the available economic diagnostic. T−59 selection reduces or shifts the sample, while the absence of T−60 data for holdout prevents an economic holdout conclusion. Historical source availability and timing remain insufficient to establish a full historical execution backtest or live readiness.
