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
Consequences: $5 gross can require >$5 cash with collateral fees. Historical fee assignment is an explicit estimate, not real-fill validation. D0016 supersedes the settlement/redemption release assumption for future sale-based economic evaluation; prior resolution-based artifacts remain historical evidence only.

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

## D0016 - Sale execution is required; unresolved positions stay open

Status: ACTIVE. Date: 2026-10-26.
Source: user-provided `instrukcja_dalszych_prac_btc_20261025.md` (2026-10-26); current-contract review and regression evidence in [E0035](EXPERIMENTS.md) and [sale-strategy gate report](reports/btc_preopen/btc_historical_sale_strategy_gate_20261026.md).
Supersedes: only the resolution-based exit/release assumption in D0006 and prior BTC economic reports for future strategy evaluation. Their frozen results and statuses remain unchanged as historical records.
Decision: do not evaluate a strategy that holds to official resolution. A reported economic outcome must be based on historically simulated executed buys and sells, including their costs. A missing or unfilled exit must not be replaced by a binary settlement payout. Retain the executed purchase, paid fees, remaining shares, and locked balance in the ledger; mark the position open and the result incomplete when it cannot be closed under the frozen sale contract. Do not choose an exit trigger, retry schedule, or maximum holding horizon from observed results.
Reason: settlement is not a sale and cannot establish the requested realized-sale economics.
Consequences: the binding contract now requires sale-only closure for future evaluations. Exact exit trigger/redecision schedule, maximum holding horizon, order type/delay, and partial-fill handling remain unspecified; new economic replay stays blocked until those terms are frozen. The legacy `close_at_60s` and threshold variants are not current policy. The unfilled-sale ledger correction is verified by the small synthetic regression; no historical replay was rerun.

## D0017 - Holdout economics precede any live or shadow consideration

Status: ACTIVE. Date: 2026-10-26.
Source: user-provided `instrukcja_dalszych_prac_btc_20261025.md` (2026-10-26).
Supersedes: the 2026-10-25 next-step recommendation to repair DNS and then run the three-window shadow pilot. The historical T0004/E0034 blocker and 0/3 result are retained.
Decision: live or shadow may be considered only after a positive economic result for the sale-based strategy on the fixed holdout. That result alone does not authorize a pilot; separate explicit user authorization is required. Do not move the split, use holdout results for development choices, or run a new holdout evaluation as part of this task.
Reason: a shadow pilot is not a substitute for the required economic evidence and is not authorized by a research result.
Consequences: T0004 is not in the active next-action queue. No collector, trader, order, or DNS pilot diagnostic is authorized here. Historical holdout exposure remains governed by D0001; any future eligible result must report that exposure and use an approved evaluation scope.

## Protected invariant register

Changes require an explicit recorded decision and applicable authorization, never an incidental implementation edit. Sources refer to the entries above and their linked original evidence.

| Invariant | Reason | Source | Protected since | Can change only if |
|---|---|---|---|---|
| UTC half-open development/holdout and exposure ledger | No test reuse concealed as new evidence | D0001 | 2026-10-10 | New frozen protocol records all prior exposure and approval |
| Proxy five-candle target, ties UP, Opened+7m release | Preserve labels and causality | D0002 | 2026-10-04 | Separate methodological decision with target/parity evidence |
| T-60 choice / T-59 execution, actual readiness retained | No execution-time leakage | D0003 | 2026-10-10 | Explicit timing protocol and measured availability |
| Same-market causal continuity, no future qualification | No invented quotes across gaps | D0004 | 2026-10-10 | Source-backed revised continuity rule, equivalence review |
| Full-depth VWAP vs explicitly labeled proxy | No fictitious liquidity claim | D0004 | 2026-10-10 | Verified depth/source semantics and ledger audit |
| Gamma minimum in gross USDC, fallback $5 | Prevent unit errors | D0004 | 2026-10-12 | Authoritative changed rule and boundary tests |
| Accepted dated fee estimate, precision/rounding and single debit | Comparable net cashflow | D0006 | 2026-10-10/12 | Explicit fee decision; no reopening date research incidentally |
| Independent $100 portfolios, cash locking, no credit | Feasible account path | D0006 | 2026-10-10 | New accounting protocol with ledgers |
| $5 gross and 5% free-cash controls, no added cap | Preserve required comparisons | D0006 | 2026-10-10 | Explicit revised experiment scope |
| Earlier-only learned state for independent claims | Prevent selection leakage | D0005 | 2026-10-10 | Different method labeled honestly, never retroactively independent |
| Ordered feature schemas, seed grid, uniform ensemble, frozen transforms | Reproducible candidate identity | D0008 | 2026-10-22 | New experiment ID/freeze before outcomes |
| RAW OOF provenance and unchanged dynamic gate | Prevent fabricated data sufficiency | D0011 | 2026-10-20 | Proven source backfill and same-gate verification |
| Evaluated/deployment bundle separation; no live activation | Preserve evaluation and authority | D0007 | 2026-10-10 | Explicit later activation authorization and operational validation |
| Append-only outcome exposure / no interim economic stopping | Prevent optional stopping | D0001/D0010 | 2026-10-22/23 | New preregistration with exposure disclosure |
| Sale-based economics only; no settlement fallback; failed exits remain open and incomplete | Do not report settlement payouts as sales | D0016 | 2026-10-26 | Explicit revised user decision and new accounting tests |
| No live/shadow consideration before positive fixed-holdout sale economics and separate approval | Keep research results distinct from pilot authorization | D0017 | 2026-10-26 | Separate explicit user authorization after the stated economic gate |
