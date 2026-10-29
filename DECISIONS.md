# Decisions and protected invariants

Recovered 2026-10-25 from code, configurations, artifacts and history at `cf885b0`. IDs are stable. Dates below identify the source decision or recovery date; this audit does not invent earlier approvals. The binding [BTC research contract](docs/BTC_RESEARCH_CONTRACT.md) is preserved. A decision being ACTIVE does not certify its implementation or experimental success.

## D0001 - Preserve split boundaries and prior exposure

Status: ACTIVE. Date: 2026-10-10; exposure updated 2026-10-22/23.
Source: [contract](docs/BTC_RESEARCH_CONTRACT.md), [candidate gate](configs/research/btc_candidate_holdout_gate_20261022.json), `run_btc_candidate_holdout_20261022.py::validate_scope`.
Supersedes: no earlier ID.
Decision: development market starts < 2026-07-01 UTC; frozen historical holdout [2026-07-01, 2026-10-07) UTC. Candidate confirmation [2025-07-01, 2026-07-01) was already exposed. The saved one-time approval covers frozen retrospective scoring; it is not blanket authorization for more protected-data runs.
Reason: prevent development/test contamination and false independent-test claims.
Consequences: inspect existing published results/metadata for recovery; do not rerun protected scorers, raw-row analysis, tuning or evaluation in an audit. New boundaries require a new recorded protocol and exposure account.

## D0002 - Protect target and information times

Status: ACTIVE. Date: 2026-10-04; binding 2026-10-10.
Source: contract; `features/btc_preopen_contract.py::build_preopen_contract_frame`, `build_preopen_target`, `purge_unavailable_training_rows`; `tests/test_btc_preopen_contract.py`.
Supersedes: historical market-open target/timing for new pre-open work only.
Decision: Binance COIN-M BTCUSD index proxy is `Close[Opened+6m] >= Open[Opened+2m]`, five complete minute candles, ties UP, label available at Opened+7m. Nominal feature/decision time Opened+1m; market start Opened+2m.
Reason: causal alignment and explicit separation from official Chainlink/Gamma settlement.
Consequences: never rename the proxy as the official label; purge unavailable labels without discarding valid warm-up history.

## D0003 - Decision at T-60, execution at T-59

Status: ACTIVE. Date: 2026-10-10/11.
Source: contract; commit `aaacebd`; `run_btc_preopen_t59_reassessment.py`; candidate shadow `market_timing`, `prediction_timeliness`.
Supersedes: single T-59 decision in `reports/btc_preopen/policy_final_research_20261007.json` and former extra delays.
Decision: side/eligibility/stake use information available at T-60. T-59 only tests execution of the existing decision. Total receipt-to-execution budget is one second; historical assumed timing is not measured live timing.
Reason: stop execution snapshots leaking into selection.
Consequences: late signals retain actual readiness and are ineligible. No backdating or implicit extra 5/45 seconds.

## D0004 - Quote continuity, depth and minimum-order units

Status: ACTIVE. Date: 2026-10-10/12.
Source: contract; `1b01c9f`; `run_btc_preopen_t59_reassessment.py`; cashflow/fee/replay tests.
Supersedes: rejecting solely on one-second changed-price age; interpreting Gamma `orderMinSize` as shares.
Decision: unchanged quotes remain valid within proven same-market source continuity. Never carry across a documented gap. Walk verified full-depth ladders; label incomplete-depth best-price execution as a proxy. Compare requested gross BUY USDC with Gamma minimum USDC, fallback $5.
Reason: price age is not a coverage gap; unknown depth is not zero liquidity.
Consequences: preserve source, depth and assumption fields, fees, full/partial-fill semantics and exclusion reasons. Old age-rejection counts cannot establish absence of exchange liquidity.

## D0005 - Causal learning and conditional evidence

Status: ACTIVE. Date: 2026-10-10; one-pass procedure explicitly adopted 2026-10-16.
Source: contract; `0f96f5b`; [full-development report](reports/btc_preopen/btc_preopen_full_development_fit_20261016.md); candidate protocols.
Supersedes: no change to the requirement for calling full-pipeline validation independent.
Decision: learned generators, feature selection, model, calibrator and policy need earlier-only fitting for independent chronological validation. One-pass full-development search plus conditional chronological OOF is implemented as a separate, explicitly conditional experiment.
Reason: earlier model validation folds do not undo full-development feature/generator selection.
Consequences: historical conditional OOF may support the named conditional comparison, never a claim that the complete pipeline has nested independent validation. Contract vs one-pass architecture remains a documented methodological deviation, not silently resolved.

## D0006 - Fees and continuous portfolio accounting

Status: ACTIVE. Date: 2026-10-10/12.
Source: contract; [fee regimes v2](configs/polymarket_btc5m_fee_regimes_v2.json); `utils/polymarket_btc5m_fee_rules.py`; cashflow and fee tests.
Supersedes: per-market archived current fee metadata as historical proof; old share-minimum valuations.
Decision: accepted estimated rates 0.072 before 2026-05-06 UTC, 0.07 afterward; preserve V1/V2 collection, rounding and one-time debit. Rebates are separate sensitivity. Each development/holdout portfolio starts independently at $100, no debt/deposits/withdrawals, stakes locked to settlement/redemption. Required controls: $5 gross and 5% of free cash with no extra dollar cap.
Reason: prevent fee double-counting and unfinanceable independent-trade PnL being presented as portfolio returns.
Consequences: $5 gross can require >$5 cash with collateral fees. Historical fee assignment is an explicit estimate, not real-fill validation. D0018 governs the active hold-to-resolution account lifecycle. D0016 still governs inactive sale simulations so an unfilled sale remains open and is never converted to a settlement payout.

## D0007 - Keep evaluated and deployment artifacts separate

Status: ACTIVE. Date: 2026-10-10/11.
Source: contract; [refit config](configs/research/btc_preopen_deployment_refit_20261011.json); `run_btc_preopen_deployment_refit.py::run_deployment_refit`.
Supersedes: no earlier ID.
Decision: frozen evaluated bundle cannot be replaced by later refit on all available history. Refit is not another independent backtest. No candidate activation or live orders authorized by research/audit.
Reason: preserve evaluation lineage and operational scope.
Consequences: runtime currently points to older BTC/ETH/SOL model metadata, not frozen A/B/C/D. No configuration changes in this audit. Missing configured refit prerequisites remain T0006.

## D0008 - Freeze A/B/C/D procedures and primary comparisons

Status: ACTIVE. Date: 2026-10-22/23.
Source: [quality freeze](configs/research/btc_candidate_quality_freeze_20261022.json), [historical gate](configs/research/btc_candidate_holdout_gate_20261022.json); scorer `predict_members`, `common_observation_mask`, `score_period`.
Supersedes: earlier candidate selection only for this frozen comparison, not runtime activation.
Decision: A=BASE70, B=A+memory PCA8, C=A+ESN PCA8, D=A+memory+ESN. A/B each have three LightGBM seeds; C/D each have a 3x3 reservoir/LightGBM grid; uniform ensembles. Primary B-A/D-B comparisons use common rows and the protocol's paired uncertainty correction.
Reason: prevent seed, transform or sample selection after scoring.
Consequences: preserve 24 model members, transforms and ordered schemas. Positive point estimates with adjusted intervals including zero remain inconclusive. Reused search-validation intervals are selection evidence.

## D0009 - Earlier prospective market-plus-BTC protocol is deferred

Status: ACTIVE (deferred, not running). Date: 2026-10-14/15.
Source: [prospective protocol](docs/BTC_PROSPECTIVE_TEST_20261014.md), [config](configs/research/btc_prospective_test_20261014.json).
Supersedes: no earlier ID; distinct from D0010.
Decision: preserve the older fixed 240-day/24-hour-grace, no-interim-significance protocol. Its collection was not started and was deferred for modeling work.
Reason: avoid treating implemented collection/planning power as observed prospective evidence.
Consequences: E0010 is PLANNED/deferred. This does not describe the different A/B/C/D shadow's final-look rule.

## D0010 - A/B/C/D shadow stopping rule replaces arbitrary duration

Status: ACTIVE. Date: 2026-10-23.
Source: [shadow config](configs/research/btc_candidate_shadow_20261023.json), [source/pilot report](reports/btc_preopen/btc_candidate_shadow_sources_and_pilot_20261023.md), latest shadow section of checkpoint log.
Supersedes: D0015.
Decision: no orders, public observations, actual readiness, official Gamma outcomes; technical pilot precedes collection claims. Primary A-vs-cash and D-vs-B; one final look after both trade/dependence and complete-day thresholds. Current planning scenario is 2% of $5 MDE, 80% power, two comparisons with Bonferroni adjustment.
Reason: a duration alone does not establish adequate power.
Consequences: 22,701/4,501 independent or 41,997/8,327 dependence-adjusted planning counts are not observed results. The 7,000-day scenario and 2,334 three-day blocks (7,002 days) expose poor feasibility; neither is a forecast or a prerequisite blocking other research. No interim economic stopping; no automatic model changes; 50 GB hard stop, no automatic data deletion.
T0007 interpretation recorded 2026-10-25: the nominal decision-information cutoff is T-60; the scheduled candle-close event must be no later than that cutoff; actual candle receipt, feature construction and prediction may complete through T-59 under the existing one-second budget; T-59 is the hypothetical execution snapshot. This follows D0003 and `features/btc_preopen_contract.py`, and does not change the frozen A/B/C/D protocol or D0003. The collector preserves UTC source/event/receive/ready timestamps, measures durations monotonically, and refuses to price a REST book whose receipt or source timestamp is after its execution deadline. Unit coverage passes. Real eligibility and latency remain UNKNOWN because Gamma/CLOB DNS prevented the live pilot; see T0004/T0007 and E0034.

## D0011 - Dynamic-history gate stays closed

Status: ACTIVE. Date: 2026-10-20.
Source: [integrity report](reports/btc_preopen/btc_dynamic_history_integrity_20261020.md), [feasibility](reports/btc_preopen/btc_dynamic_history_feasibility_20261020.md); `run_btc_market_dynamics_action_value_20261020.py::_prediction_availability_status`, `_availability_report`.
Supersedes: original 633-market diagnostic by corrected 637 count; does not reopen the gate.
Decision: preserve frozen folds/minima and fail closed on unproven RAW OOF information time. No action-value model/promotion/economics until source provenance and unchanged gate pass.
Reason: 637 < 2,000 evaluation markets; fold-1 $5 both-side training 908 < 1,000; nominal timestamps do not prove prediction readiness.
Consequences: independent audit recovered four markets, not the missing research result. A new source backfill must have genuine provenance; no fabricated availability timestamps.

## D0012 - Distinguish official outcomes from settlement-rule reconstruction

Status: ACTIVE. Date: 2026-10-23.
Source: [regime analysis](reports/btc_preopen/btc_discrepancy_regimes_20261023.md), [candidate regime report](reports/btc_preopen/btc_candidate_settlement_regimes_20261023.md); saved analysis manifests.
Supersedes: describing TWAP60 as the rule for the full historical sample; earlier incorrect aggregate direction cells.
Decision: calendar assignment is single-price before August 7 UTC, TWAP30 August 7-14, TWAP60 from August 14. Seven inspected market pages support boundaries; 79,643 other individual rules are not directly confirmed. Gamma one-hot outcomes do not contain the underlying Chainlink TWAP price reports.
Reason: target mismatch and temporal regime shift do not identify a price-level cause.
Consequences: preserve corrected 4,193 disagreement total; obtain exact-feed historical evidence before causal claims. Existing 1,143-market economics cannot support TWAP60 trading performance.

## D0013 - Baselines are conditional, not promoted winners

Status: ACTIVE. Date: 2026-10-18 through 2026-10-22.
Source: seed-stability, FULL_FEATURES, RECENT365, ESN architecture/focused and candidate-freeze reports in EXPERIMENTS.md.
Supersedes: no runtime decision.
Decision: original SINGLE remains a historical research reference where comparisons failed promotion; RECENT365 closes only that frozen window procedure. Later memory/ESN A/B/C/D are distinct frozen comparisons, not an authorization to replace live models.
Reason: development selection and seed stability are not deployment validation.
Consequences: do not repeat closed variants unchanged or present a probe portfolio as the best downstream model.

## D0014 - Canonical state and audit scope

Status: ACTIVE. Date: 2026-10-25.
Source: user's state-recovery request; [AGENTS.md](AGENTS.md); this audit.
Supersedes: ambiguous current-state readings of historical checkpoints.
Decision: PROJECT_STATE is the snapshot, this file the decision/invariant register, EXPERIMENTS the evidence/status register, TASKS the acceptance-based backlog. The existing BTC status remains the checkpoint history; contract remains binding. Audit changes are documentation/evidence only; bugs become tasks.
Reason: prevent false completion and competing state narratives.
Consequences: future work updates applicable records, protects invariants and uses explicit evidence levels. Commit and normal push follow AGENTS.md; that grants no deployment/merge/trading authority.

## D0015 - Earlier 180-day A/B/C/D shadow recommendation

Status: SUPERSEDED. Date: 2026-10-23. Superseded by: D0010.
Source: `btc_candidate_settlement_regimes_20261023.md` and its checkpoint section.
Decision: earlier recommendation of at least 180 complete UTC days.
Reason: later source/pilot work derived explicit precision and dependence requirements.
Consequences: preserve the historical recommendation as history; do not treat it as the current stopping rule.

## D0016 - Sale simulations require executed exits; unresolved positions stay open

Status: SUPERSEDED FOR ACTIVE STRATEGY SCOPE by D0018; retained for sale simulations. Date: 2026-10-26.
Source: user-provided `instrukcja_dalszych_prac_btc_20261025.md` (2026-10-26); current-contract review and regression evidence in [E0035](EXPERIMENTS.md) and [sale-strategy gate report](reports/btc_preopen/btc_historical_sale_strategy_gate_20261026.md).
Supersedes: no longer controls the active strategy objective. Its accounting rule for an inactive sale simulation remains in force.
Decision: when a sale-based simulation is run, only an executed sale closes that simulated sale position. A missing or unfilled exit must not be replaced by a binary settlement payout. Retain the purchase, paid fees, shares, and locked balance; mark the position open and result incomplete when it cannot be closed under the frozen sale contract. Do not choose an exit trigger, retry schedule, or maximum holding horizon from observed results.
Reason: a settlement payout must not conceal a failed sale in a sale-based ledger.
Consequences: sale-policy research is deferred and outside the active scope. The legacy `close_at_60s` and threshold variants are not current policy. The no-fill correction remains covered by the E0035 regression.

## D0017 - Holdout economics precede any live or shadow consideration

Status: ACTIVE. Date: 2026-10-26.
Source: user-provided `instrukcja_dalszych_prac_btc_20261025.md` (2026-10-26).
Supersedes: the 2026-10-25 next-step recommendation to repair DNS and then run the three-window shadow pilot. The historical T0004/E0034 blocker and 0/3 result are retained.
Decision: live or shadow may be considered only after a positive economic result for the active buy-and-hold-to-resolution strategy on the fixed holdout. Earlier sale is not a prerequisite. A positive result alone does not authorize a pilot; separate explicit user authorization is required. Do not move the split, use holdout results for development choices, or run a new holdout evaluation as part of this task.
Reason: a shadow pilot is not a substitute for the required economic evidence and is not authorized by a research result.
Consequences: T0004 is not in the active next-action queue. No collector, trader, order, or DNS pilot diagnostic is authorized here. Historical holdout exposure remains governed by D0001; any future eligible result must report that exposure and use a separately approved evaluation scope.

## D0018 - Buy and hold through official resolution is the active strategy

Status: ACTIVE. Date: 2026-10-26.
Source: user's explicit request to execute `instrukcja_rozstrzygniecie_btc_20261026.md` (2026-10-26); [E0036](EXPERIMENTS.md) and its frozen protocol.
Supersedes: D0016's sale-only restriction on the active objective and the sale-based next-step recommendation in T0008. D0016's no-settlement-fallback rule for inactive sale simulations remains unchanged.
Decision: evaluate purchase of UP/DOWN at the frozen T-59 execution for a T-60 decision, then hold the position until official Gamma resolution and its binary payout. The payout is the planned closing event for this strategy. Keep the gross purchase cost locked until the frozen payout-availability time; distinguish market end, official outcome-known time, and assumed payout availability. Use official Gamma results, not the Binance proxy. Preserve separate $100 portfolios for fixed $5 gross and 5% of free cash, no credit/deposits, accepted fee rules, and the historical $5 minimum-order fallback when Gamma minimum data are unavailable.
Reason: the current research question is buy-and-hold economics through official resolution; sale timing is outside the active scope.
Consequences: D0017's holdout gate is resolution-based economics. Current E0036 results are a sparse, previously exposed confirmation diagnostic, not full-development validation. Actual payout availability and historical T-60 prediction-ready timestamps remain unverified. No new holdout run, live/shadow process, or order is authorized by this decision.

## D0019 - Minimum-order checks require verified USDC units

Status: ACTIVE. Date: 2026-10-26.
Source: T0009 minimum-order audit; D0004; [Polymarket Market Details](https://docs.polymarket.com/market-data/market-details), [Place Orders](https://docs.polymarket.com/trading/place-orders), and [py-sdk issue #302](https://github.com/Polymarket/py-sdk/issues/302).
Supersedes: no decision; clarifies D0004's minimum-order unit rule.
Decision: compare a requested gross BUY in USDC only with a minimum whose unit is verified as USDC. Ignore share-labeled minimum fields for that check. If historical USDC minimum data are unavailable or unverified, use the frozen $5 USDC fallback; reject below-minimum requests without rounding them upward. Do not project current market metadata backward as historical evidence.
Reason: available Gamma and CLOB fields use different unit labels, the local historical file contains only a share-labeled field without capture times, and the minimum-order constraint cannot be treated as unit-equivalent without evidence.
Consequences: the T-59 helper now ignores the share-labeled Gamma field. This code correction does not establish historical market minima or change the frozen E0037 source-only result. Any later economics run must pin the corrected source identity and pass the source-parity gate first.

## D0020 - E0039 follow-up uses 1/5/15 second settlement scenarios

Status: ACTIVE for E0040. Date: 2026-10-28.
Source: user's explicit authorization in `instrukcja_E0039_opoznienia_1_5_15s_20261028.md`.
Supersedes: no prior decision; this records the follow-up scenario set and its event ordering without changing E0039's saved protocols or results.
Decision: E0040 uses payout delays `[1, 5, 15]` seconds as simulation assumptions. Payout availability remains `max(market_end_utc, resolved_at_utc) + delay_seconds`, with market end at `market_start_utc + 5 minutes`. The old 60/300/1800/7200-second scenarios remain historical and are not treated as current operational settings or measured Polymarket timing. At T-60, release payouts available by that instant, then freeze the decision, side, EV, and 5%-of-free-cash size. At T-59, release payouts available through that timestamp, inclusively, immediately before testing the already-fixed full debit. A later payout cannot fund that purchase. Payout delay does not pause or move prediction, feature/state, calibration, or market-evaluation timestamps. Keep buy-and-hold through official Gamma resolution; no early sale or cash-out.
Reason: the user approved these three hypothetical settlement assumptions and required correct event-time cash availability for second-level delays.
Consequences: E0040 replays saved E0039 predictions and snapshots over its unchanged 5,604/5,148 coverage. It does not establish measured redemption availability, historical readiness, or production readiness; T0009, E0037, and T0010 retain their existing statuses.

## D0021 - Freeze one D economic evaluation protocol

Status: ACTIVE. Date: 2026-10-28.
Source: user's explicit request to execute `instrukcja_D_przygotowanie_holdoutu_20261028.md`; [E0041 protocol](configs/research/btc_D_holdout_preparation_20261028.json).
Supersedes: no earlier decision. D0001's fixed split and historical exposure, D0008's frozen family/seed conventions, D0017/D0018's holdout-before-pilot and resolution economics, D0019's verified-USDC minimum, and D0020's 1/5/15-second event order remain active.
Decision: E0041 prepares a single D pipeline for the fixed market-start interval `[2026-07-01T00:00:00Z, 2026-10-07T00:00:00Z)`. Selection is limited to labels available strictly before `2025-10-01T00:00:00Z`; the final D fit uses labels available strictly before the first decision at `2026-06-30T23:59:00Z`. D is BASE70 + memory PCA8 + ESN PCA8, with three LightGBM seeds × three reservoir seeds and a uniform nine-member mean. Fit one terminal Platt calibrator from causal D OOF predictions joined to official Gamma labels released by the first holdout decision; require at least 1,000 rows and both classes, with no proxy fallback. Freeze all six `$100` account paths (fixed `$5` and 5% of free cash, each at `[1, 5, 15]` second assumed payout delays), the E0038 symmetric source rule, complete half-open interval, and fixed positive-result criterion before any new holdout evaluation.
Reason: specify one reproducible future economic test for the development-selected D pipeline while preserving timing, source-quality, accounting, and historical-exposure boundaries.
Consequences: this decision authorizes preparation code, synthetic/development tests, and a development-only fit attempt behind the frozen resource gate. It does not authorize any holdout payload read, prediction, scoring, economic replay, source acquisition, live/shadow run, or order. The holdout is previously exposed, not independent. No result can authorize a pilot; a later explicit evaluation authorization is required, and a positive result alone is insufficient.

## D0022 - E0041 v2 cumulative-budget continuation

Status: ACTIVE. Date: 2026-10-28.
Source: user's explicit authorization in `instrukcja_E0041_budzet_v2_i_kontynuacja_20261028.md`; [E0041 v2 protocol](configs/research/btc_D_holdout_preparation_20261028_v2.json).
Supersedes: D0021 only for the execution-budget ceilings and versioned continuation of its saved development checkpoint. All D0021 experiment, cutoff, target, model, calibration, accounting, source-quality, and holdout boundaries remain active.
Decision: authorize one versioned continuation of E0041 from the preserved v1 state. Cumulative RP budget is 7,200 seconds (v1: 2,700; 2,704.5527 seconds already used, leaving 4,495.4473); the cumulative selection-chain allocation is 52,200 seconds (v1: 47,700); the cumulative whole-preparation limit is 76,500 seconds (v1: 72,000). The 96-terminal-trial target, minimum five valid COMPLETE trials, VP 2,700-second cap, and all remaining stage caps stay unchanged. COMPLETE, PRUNED, and FAIL count as terminal toward 96; FAIL and crash-penalty trials do not count toward the five valid successes, and any pending/running trial blocks completion. Preserve the 46 terminal v1 trials (44 COMPLETE, 2 PRUNED) and continue with 50 additional proposals. Reuse the verified completed indicator artifacts and safe raw prefix. Adopt only compatible artifacts into the distinct v2 run after hash, profile, input, and SQLite-backup checks. Record `sampler_path_guaranteed_identical_after_resume=false` because the TPE RNG state is not preserved.
Reason: the v1 reaction-profile stage exhausted its cumulative 2,700-second allowance after 46/96 trials; its complete study and preprocessing checkpoint make a budget-only, versioned continuation possible without resetting the sample.
Consequences: all prior v1 execution time and each later execution attempt count toward cumulative limits; idle time between sessions does not. Checkpoint migration counts only against the whole-preparation limit, not again against the RP stage. This authorizes development-only preparation under the unchanged resource gate. `evaluation_authorized=false`; no holdout payload, predictions, economics, confirmation evaluation, source acquisition, book replay, live/shadow activity, or orders are authorized. E0037 remains FAILED and T0009 remains BLOCKED.

## Protected invariant register

Changes require an explicit recorded decision and applicable authorization, never an incidental implementation edit. Sources refer to the entries above and their linked original evidence.

| Invariant | Reason | Source | Protected since | Can change only if |
|---|---|---|---|---|
| UTC half-open development/holdout and exposure ledger | No test reuse concealed as new evidence | D0001 | 2026-10-10 | New frozen protocol records all prior exposure and approval |
| Proxy five-candle target, ties UP, Opened+7m release | Preserve labels and causality | D0002 | 2026-10-04 | Separate methodological decision with target/parity evidence |
| T-60 choice / T-59 execution, actual readiness retained | No execution-time leakage | D0003 | 2026-10-10 | Explicit timing protocol and measured availability |
| Same-market causal continuity, no future qualification | No invented quotes across gaps | D0004 | 2026-10-10 | Source-backed revised continuity rule, equivalence review |
| Full-depth VWAP vs explicitly labeled proxy | No fictitious liquidity claim | D0004 | 2026-10-10 | Verified depth/source semantics and ledger audit |
| Gamma minimum in gross USDC, fallback $5 | Prevent unit errors | D0004/D0019 | 2026-10-12/26 | Authoritative changed rule and boundary tests |
| Accepted dated fee estimate, precision/rounding and single debit | Comparable net cashflow | D0006 | 2026-10-10/12 | Explicit fee decision; no reopening date research incidentally |
| Independent $100 portfolios, cash locking, no credit | Feasible account path | D0006 | 2026-10-10 | New accounting protocol with ledgers |
| $5 gross and 5% free-cash controls, no added cap | Preserve required comparisons | D0006 | 2026-10-10 | Explicit revised experiment scope |
| Earlier-only learned state for independent claims | Prevent selection leakage | D0005 | 2026-10-10 | Different method labeled honestly, never retroactively independent |
| Ordered feature schemas, seed grid, uniform ensemble, frozen transforms | Reproducible candidate identity | D0008 | 2026-10-22 | New experiment ID/freeze before outcomes |
| RAW OOF provenance and unchanged dynamic gate | Prevent fabricated data sufficiency | D0011 | 2026-10-20 | Proven source backfill and same-gate verification |
| Evaluated/deployment bundle separation; no live activation | Preserve evaluation and authority | D0007 | 2026-10-10 | Explicit later activation authorization and operational validation |
| Append-only outcome exposure / no interim economic stopping | Prevent optional stopping | D0001/D0010 | 2026-10-22/23 | New preregistration with exposure disclosure |
| E0040's settlement delay assumptions and T-60/T-59 cash-event ordering | Keep seconds-level simulation causal and explicit | D0020 | 2026-10-28 | New explicit user-approved protocol; measured availability remains separate |
| Active hold-to-resolution strategy closes by official payout; sale simulations do not use settlement as an unfilled-exit fallback | Separate the two accounting paths | D0016/D0018 | 2026-10-26 | Explicit revised user decision and accounting regression tests |
| No live/shadow consideration before positive fixed-holdout resolution economics and separate approval | Keep research results distinct from pilot authorization | D0017/D0018 | 2026-10-26 | Separate explicit user authorization after the stated economic gate |

## D0023 - E0043 source-only acquisition for the 68-hour book gap

Status: ACTIVE. Date: 2026-10-29.
Source: user's explicit request to execute `instrukcja_uzupelnienie_68h_zrodel_holdoutu_20261029.md`; [E0043 protocol](configs/research/btc_D_holdout_book_gap_fill_20261029.json).
Supersedes: D0021 only for its prohibition on source acquisition, and D0001 only for its prohibition on protected raw-row reads during an audit, within the exact scope below. All other holdout, split, exposure, model, feature, timing, and economic invariants remain active.
Decision: E0043 may inspect publisher indexes/object metadata and, only if a documented qualifying source exists, acquire and read native full-book event records for `[2026-08-15T10:00:00Z,2026-08-18T06:00:00Z)`. The only permitted payload use is source validation of schema and event semantics, native condition/token identity, event/receive times, initialization, ordering, continuity, and source provenance. Adjacent hours may be accessed only when a frozen manifest proves they are necessary for initialization or a boundary check. Do not read outcome/resolution/payout fields, forecasts, cached feature values, or historical economic measures; do not generate features/predictions/evaluation rows or calculate EV, fills, trades, PnL, or predictive metrics. Do not run E0041's evaluator. Keep `evaluation_authorized=false`, `activated=false`, and `HOLDOUT_NOT_RUN`.
Reason: E0042 found 68 missing full-venue hours and 1,115 affected source-history windows. A separate explicit source-only scope is needed to test remote availability without turning source repair into development or holdout evaluation.
Consequences: freeze E0043's provider/index/parser/range/resource rules before payload transfer; retain the E0042 gap evidence and archive files; write any acquired files only under a new ignored local staging directory; version only code, small manifests, and reports. Do not replace full native depth/timing with candles, trades, partial-depth snapshots, or nominal timestamps. No source availability result changes E0037 FAILED, T0009 BLOCKED, T0010 DEFERRED, or the E0041 D bundle.
