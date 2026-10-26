# BTC 5m sale-based historical evaluation gate — 2026-10-26

**Status: BLOCKED_CONTRACT_GAP.** No current final sale rule or maximum position horizon is frozen, so the active strategy cannot yet be replayed without inventing exit policy. This is not a negative profitability result. No new economic replay, holdout evaluation, training, tuning, download, collector run, or network diagnostic was performed.

## Current contract

- **Information and entry timing:** D0003 keeps side, eligibility, and stake based on information available at T−60. Candle receipt, feature construction, and prediction may complete through T−59 within the existing one-second budget. T−59 is the execution snapshot; it cannot alter the earlier decision.
- **Entry reference:** E0031's frozen fixed-$5 diagnostic selected the side with the highest positive expected net payout, broke a positive tie toward UP, and otherwise skipped. It used T−60 native asks/displayed size for the decision and T−59 native ask depth for the $5 execution simulation. The economic cohort required complete fills; the T−60 snapshots do not contain full ladders. D0006 also requires both fixed-$5 and 5%-of-free-cash portfolio controls for a full economic comparison.
- **Fees and capital:** preserve D0006's estimated fee rates (0.072 before 2026-05-06 UTC and 0.07 afterward), V1/V2 collection, established rounding, and one-time debit. Each period starts with $100, no credit or deposits; overlapping positions consume free cash and remain locked until sale. Historical fee estimates and book simulations are not real fills.
- **Exit:** D0006's old settlement release is superseded for future evaluation by D0016. A position must close through an executed simulated sale with costs. An unavailable or unfilled sale leaves the bought shares, entry fees, and locked cost in the ledger; the result is incomplete. There is no active rule specifying exit decision times/triggers, retries, maximum holding horizon, exit order type/delay, or partial-fill handling. The historical `close_at_60s`, take-profit, and stop-loss variants are not the current contract.
- **Order and fills:** existing entry studies use counterfactual taker depth walks and generally require the complete requested order. Historical ACKs, actual partial fills, and archived per-market minimums are absent. D0004's full-depth versus top-of-book-proxy labels remain binding. No current exit execution rule is frozen.

## Frozen candidate identity

The candidate letters are only aliases. The binding identity is [the freeze config](../../configs/research/btc_candidate_quality_freeze_20261022.json), SHA-256 `0fbade42eb91d48e876d4178d15de1da6a86e185c44f915bc0620603f907e175`, and the local frozen procedure manifests. Candidate manifest hashes below are recorded in the E0031 v3 manifest (SHA-256 `16e66363f35390d520e5a3ca7ce58fa07689ac15e66244a313c50f1a29017503`).

| Alias | Frozen procedure | Candidate ID | Procedure manifest SHA-256 |
|---|---|---|---|
| A | BASE70 | `BASE70` | `28368afa62f824b5d2592eb61f87cb7a48ba8b66d0afc0a97ccdf6f334c9f1ef` |
| B | BASE70 + simple-memory PCA8 | `simple_memory_pca8` | `66eddd315dc38326f6f64959cd4a4fd8d053a6cd189ef7cceac5b3c35bd6a567` |
| C | BASE70 + focused ESN PCA8 (auxiliary ablation) | `8f74a7951e10c11830baba75` | `f5ca0f29908e473b7e67be344c9430da0163cfd80f29d8bba9ce3931985f4d45` |
| D | BASE70 + simple-memory PCA8 + focused ESN PCA8 | `D_memory_plus_8f74a7951e10c11830baba75` | `c4910eb9d4fff03680fe36242c6d338abb2dcbc422803e94bd15c700869e6119` |

The existing E0030 prediction run identity is `ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b`. Its confirmation and holdout prediction artifact hashes are respectively `ac81b1fb7a88a01ac1135b14c84d642f3d90615ad4b374928434c9af8ef881be` and `6469e6d95d93cb6e8cef3659437b97c52482cd0b4e9747dd9cfc843765553d3b`. These artifacts are local, ignored dependencies; their records were inspected, but prediction rows were not loaded in this task.

## Existing studies relevant to sale economics

| Study / candidate | Period and prediction source | Entry | Exit | Costs and capital | Existing result / status | Remaining limit for this task |
|---|---|---|---|---|---|---|
| E0006 legacy policy/exit studies | Historical development; older candidate and saved predictions | Legacy T−59 entry selector, exact $5 buys, older one-second timing | Old `close_at_60/180/240s` and threshold variants; prior simulator settled positions after a failed/no sale | Historical fee reconstruction; $100 continuous portfolio plus separate independently funded bet diagnostics | CLOSED_INCONCLUSIVE. `optimal_stopping_20261007.md` reports 2,285 identical buys; ridge sold 1,961 and reported +$146.62 total PnL. The total included positions not sold and therefore cannot be read as sale-only PnL. | Different candidate/entry protocol; Kacho is sampled top-of-book, PMXT exit paths are not mapped to frozen A/B/C/D, and the previous no-sale fallback was invalid under D0016. |
| E0030 / E0031 frozen A/B/C/D | Confirmation `[2025-07-01, 2026-07-01)` and previously exposed holdout `[2026-07-01, 2026-10-07)`; retrospective frozen prediction artifacts | E0031 fixed-$5 T−60 side choice and T−59 native-ask simulation; positive expected net payout or no trade | **Hold to official resolution** in the v3 manifest | Estimated historical fees; independent $5 diagnostics plus separate $100 portfolios | CLOSED_INCONCLUSIVE. E0031 has 1,143 common confirmation entry markets only; B +$28.01 and D +$22.66 were settlement-based diagnostics. There is no holdout economic result. | Not sale economics; no full exit path/portfolio result. Prior scores and holdout exposure are retained, not rerun or used for selection. |
| E0021 RAW BASE64/70 holdout | Previously exposed `[2026-07-01, 2026-10-07)`; saved RAW predictions | Fixed-$5 historical entry and resolution outcome | Resolution-based payoff | After-fee independent trades, not a sale ledger | CLOSED_NEGATIVE: BASE64 −$529.30; BASE70 −$546.22. | Different models/period protocol and no sale exit. It is not a result for the current sale-based strategy. |
| E0024 dynamic history/action value | Frozen development folds only; saved RAW OOF | No fitted action-value policy | No economic exit evaluation | No portfolio result | BLOCKED; no model result. | RAW prediction availability/provenance gate remains closed; see counts below. |

E0030's prediction scores are predictive evidence only. E0031's settlement-based gains and E0006's mixed sale/settlement totals are not evidence that the current sale strategy is profitable.

## Prediction provenance

Three distinct provenance claims must remain separate:

1. **Actually available at the historical decision:** not established for the frozen candidate predictions. E0031 confirms nominal market-start/T−60 alignment for saved rows, but not actual input receipt or prediction-ready timestamps. No timestamp was fabricated here.
2. **Reproducible today from a historical data cutoff:** frozen A/B/C/D bundle and saved prediction identities exist; E0029 documents representation parity. That does not establish historical compute/readiness time or the independence of the full selection pipeline.
3. **Potential future selection or availability contamination:** D0005 records that full-pipeline nested validation is not established. For E0024's B0 RAW OOF, feature/generator selection and tuning used full development. The 318,560 rows all have missing `prediction_available_at_utc`; after applying that required fail-closed condition, the final RAW-qualified gate is zero. No refit can retroactively prove when old predictions were ready.

The E0024 counters are not an eligible cohort: **637/2,000** is the saved book/H evaluation count before RAW availability; **908/1,000** is fold-1 two-sided $5 training count before RAW availability; **633** is the earlier unrecovered count; and **44** is a small independent verifier sample (40 prior cases plus 4 recovered), not the full 637 or a qualifying sample. The final provenance-qualified count remains **0**. The frozen gate and folds were not changed.

## Book and position-path coverage

| Evidence | Known count/range | What it establishes | What remains unknown |
|---|---|---|---|
| E0031 frozen candidate entry artifacts | 50,184 T−59 native-ask snapshots; 9,407 exact T−60 snapshots; 4,066 T−60 fixed-$5 checks; 5,609 provenance-qualified full-$5 T−59 fills in confirmation; 1,143 common T−60/T−59 confirmation markets; zero exact T−60 holdout snapshots | Entry diagnostics on the existing cohort only | No sale path, repeated exit decision, or current-contract position lifecycle was counted. Snapshot availability does not establish later bid depth. |
| E0006 legacy exit diagnostic | Kacho 300-second paths and post-start five-second decisions for 4,454/4,454 legacy markets; 23,931 pre-start and 126,675 post-start full-position exit quote observations | A historical path sample for a different legacy policy | Kacho is sampled top-of-book with displayed top size, not a complete continuous full-depth event tape. It is not an A/B/C/D market-path join. |
| E0024 PMXT development audit | 1,853/1,856 expected hours checked; 1,409 non-empty and 444 confirmed empty; three missing hours on 2026-06-11 | Coverage for the audited E0024 windows; those three hours were outside all 26-hour windows in that audit | Does not establish PMXT completeness for another market cohort or exit horizon. The 44-case verifier is only a control sample. |
| E0034 source review | Local PMXT v2 ends 2026-08-10, before TWAP60 starts 2026-08-14. Vike's BTC/5m family layout misses 2026-08-14–21; a mixed-layout target-window fetch was estimated at 101.88 GB. PMData access/quota and Marketlens target coverage/price remain unverified. | Known historical source limits; public CLOB is current-book data, not historical full-depth replay | No data was acquired. Holdout ranges here are structural source notes only; no holdout economics or label-based sample selection was performed. |

The PMXT empty-bid initialization issue is already addressed: a full `book` snapshot with `bids=[]` and a populated native ask initializes the token book. `tests.test_btc_preopen_economic_replay.BtcPreopenEconomicReplayTests.test_empty_bid_snapshot_still_initializes_native_ask_book` passed before and after this task's code change. No further patch was needed for it.

An exhaustive per-market exit-gap list cannot be truthfully generated until T0008 freezes the sale decision schedule and maximum holding interval. The accompanying JSON records that unknown explicitly and separates known source ranges from the unresolved market-specific requirement. No 101.88 GB download was started.

## Confirmed accounting defect and repair

Before the repair, `run_btc_historical_policy_study._simulate_exit_portfolio` scheduled an official-resolution payout whenever `_exit_plan` returned no sale. The prior test encoded this by expecting `time_exit_unfilled_hold` to become `actual_exit_type=resolution`; a hand-calculable fixture with a $5 buy, $0.175 entry fee, and insufficient bid depth produced a $104.825 ending cash balance by using the outcome payout.

The simulator now records an unfilled time exit as `time_exit_unfilled_open`, preserves the $5 locked cost and $0.175 paid fee, leaves $94.825 free cash, and returns `economic_result_status=incomplete_open_positions` with `net_pnl_usd=null`. It no longer requires an outcome to create the purchase or includes resolution payout fields. The focused fixture passes with ending cost-basis equity $99.825 and one open position. The focused replay/accounting modules passed 30/30 tests; full discovery (`python -m unittest discover -s tests`) passed 680/680 in 40.247 seconds, exit code 0. This mechanics fix does not select `close_at_60s` as the active rule and does not produce a historical strategy result.

## Outcome and next action

- **Development result:** none. The exact blocker is an unspecified sale trigger/redecision schedule and maximum holding horizon, followed by unresolved order type/delay and partial-fill semantics. No result from official resolution was used.
- **Holdout:** no new evaluation; prior exposure remains. The existing E0031 holdout has no entry T−60 economic cohort, and its reported quality scores do not satisfy D0017.
- **Pilot/live:** none. T0004/E0034 remain historically blocked at 0/3; D0017 removes the former “repair DNS then pilot” recommendation from the active queue.
- **One next action:** freeze the sale rule and position lifecycle under T0008. Only then can the required quote intervals and per-market data-gap manifest be determined.
