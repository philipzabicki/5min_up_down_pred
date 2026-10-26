# PROJECT STATE

Last audited: 2026-10-26 (Europe/Warsaw).
Audit commit: the commit introducing this file, `docs: recover canonical project state` (resolve with `git log -1 --format=%H -- PROJECT_STATE.md`; no self-referential hash).
Audit input HEAD: `e74d70997ef75eba0f818fb18d209817897e2f36` on `audit/btc-dynamic-history-20261020`; upstream was `origin/audit/btc-dynamic-history-20261020` and the tree was clean before this task.

## Objective

Predict BTC five-minute Binance-index direction before a Polymarket market opens, then assess official-outcome prediction quality and executable after-fee economics separately. The active BTC economic strategy is buy UP/DOWN at the frozen T-59 execution and hold to official Gamma resolution. The repository also contains ETH/SOL live and general modeling tooling.

## Current phase

Historical research and frozen A/B/C/D comparison; **no demonstrated production readiness or repeatable trading advantage**. A=BASE70, B=A+simple-memory PCA8, C=A+ESN PCA8, D=A+memory+ESN. E0036's 1,143-market confirmation result remains sparse, previously exposed and unchanged. E0037's saved source-parity gate remains **FAILED**. E0038 separates value parity from checker coverage and defines a symmetric source-quality rule for a 5,604-market retrospective, coverage-conditional subset. V1 was committed/pushed; its runner stopped at the commit guard before reading source/outcome rows due to CRLF normalization. V2 preserves the methodology and must be committed/pushed before conditional economics. T0009 remains blocked for complete development coverage and causal/readiness provenance. T0008/E0035 sale-policy work is deferred/out-of-scope under D0018. The old A/B/C/D no-order pilot remains historically blocked at 0/3 and is not an active next step under D0017. Dynamic-history/action-value research remains independently blocked before fitting.

Registers: [DECISIONS](DECISIONS.md), [EXPERIMENTS](EXPERIMENTS.md), [TASKS](TASKS.md). Details/matrices/provenance: [audit report](docs/PROJECT_STATE_AUDIT_20261025.md), [evidence](reports/project_state_recovery_20261025.json). The [BTC status](docs/BTC_RESEARCH_STATUS_20261010.md) remains the checkpoint history; [research contract](docs/BTC_RESEARCH_CONTRACT.md) remains binding.

## Verified working

- Full discovery at audited HEAD: **675 tests, 665 PASS, 2 FAIL, 8 ERROR, 0 SKIP, 0 XFAIL**, 51.735 s runner / 62.653 s wall. This is the preserved baseline; follow-up results below supersede its test status.
- Follow-up on 2026-10-25: **680 tests, 680 PASS, 0 FAIL, 0 ERROR, 0 SKIP, 0 XFAIL**, 45.157 s, exit code 0 after fixes T0001-T0003.
- Sale-ledger regression on 2026-10-26: the focused PMXT initialization and exit-accounting modules passed **30/30** tests after correcting the no-sale settlement fallback; full discovery passed **680/680** tests in **40.247 s**, exit code 0. This validates mechanics only.
- E0036 resolution ledger: settlement regression **3/3**, existing replay/accounting tests **24/24**, and full discovery **683/683** passed (51.332 s, exit code 0). The hash-verified sparse confirmation run reproduced E0031 fixed-$5 economics and separately computed the 5%-of-free-cash control, cost-basis drawdown and event accounting. This validates code mechanics, not strategy profitability or production readiness.
- E0037 source-only validation on 2026-10-26: focused T0009 tests **10/10**, T-59 reassessment tests **34/34**, and full discovery **696/696** passed (55.837 s, exit code 0). The full local development PMXT snapshot was hash/provenance verified and reused. E0036 T-60 source parity still failed; no economic results were produced.
- E0038 v1 pre-economics validation: focused source/economics tests **10/10** and full discovery **706/706** passed (52.155 s, exit code 0); its runner stopped at a protocol commit guard before data/outcome loading. E0038 v2 guard tests pass **12/12**, and full discovery passes **708/708** (45.048 s, exit code 0). No economic result has been generated.
- Read-only candidate shadow SQLite: two preflight events, zero data/market/checkpoint rows (E0034).
- Completed frozen scorer/period/approval/source records exist and were hashed; historical scores were not recomputed (E0030).

## Implemented but not validated

- Acquisition/features/training/calibration/replay/accounting/live orders/redemption have code/tests; complete reference/live validation UNKNOWN.
- Saved memory/ESN parity covers 2,660,546 development raw rows; large replay not repeated (E0029).
- All primary adjusted B-A/D-B historical intervals cross zero (E0030/E0031).
- E0031's 1,143 confirmation-market fixed-$5 diagnostic is resolution-based and has no holdout economic cohort. Under D0018, its B +$28.01 / D +$22.66 independent PnL is a relevant diagnostic, but it does not establish complete development economics or a repeatable edge. E0036 adds separate $100 account paths and the required 5%-of-free-cash control.

## Partially implemented

- Complete independent nested pipeline validation is not established by conditional OOF or one-pass fitting (D0005/E0013).
- Earlier model-update B stopped below its successful-trial minimum; no B holdout score (E0009).
- Shadow timing code now uses the T-60 information cutoff and T-59 readiness deadline, retains actual event/receive/ready timestamps, and rejects late books; live timing remains unmeasured because endpoint preflight failed (T0007/T0004).
- Deployment refit code exists; configured original holdout prerequisite/output absent (T0006).
- Sale accounting now leaves an unfilled exit open, with purchase fees and cost basis retained and PnL marked incomplete; this is a tested mechanics correction, not an active exit policy or historical result (E0035).

## Cleared test failures

- T0001 ESN shared-module state is restored after the candidate checkpoint test; suspect test order now passes.
- T0002 stale test expectations were aligned to the fixed 2-tick code/runtime value; runtime policy was not changed.
- T0003 fake submit fixture now matches real trader initialization. This was not a real exchange failure.
- README names absent `optimize_lgbm_optuna.py` and `.python-version`; current optimizer is `optimize_lgbm_hyperparameters.py` (T0006).

## Blocked

- Dynamic gate: 637/2,000 evaluation; fold1 $5 training 908/1,000; RAW timing provenance missing. No model result (E0024).
- Resolution economics: E0038 confirms 19,431 T−60 and 26,289 T−59 source-value parity comparisons with no mismatches, while distinguishing old unchecked rows from passed checks. Its symmetric source-quality rule leaves 5,604 of the 5,684 retrospective source intersection; the T−60 ask/BBO conflict remains unresolved and quarantined. No measured prediction-ready timestamps exist, full-pipeline selection/calibration independence is not established, and 53,660 prediction calendar rows have no Gamma ID. Conditional subset economics do not close T0009. Actual redemption availability remains unknown.
- Sale-policy study: T0008/E0035 are deferred/out-of-scope, not blockers for the active resolution strategy. The sale simulator still keeps unfilled exits open and never treats them as settlement.
- Shadow: bounded 2026-10-25 same-interpreter preflight reached Binance time (HTTP 200, valid schema) but Gamma and CLOB DNS resolution failed. No collector was started; pilot remains 0/3. Under D0017 this is retained history, not an active next step (T0004).
- Exact historical Chainlink TWAP access/coverage/cost and TWAP60 L2 missing; no acquisition authorized (T0005).

## Unknown / needs verification

Real fills/ACKs/fees and operational live readiness; full-pipeline independent OOS performance; prospective profitability; raw archive completeness beyond inspected manifests; individual rule text for 79,643 of 79,650 calendar-assigned markets; exact cause of 4,193 Binance/Gamma label disagreements; reproducibility on a clean clone and another OS. Large ignored artifacts are local dependencies, not delivered by Git.

## Active experiments

E0036 PARTIAL on the available confirmation subset. E0037 PARTIAL with its original FAILED parity gate preserved. E0038 v2 is frozen before economics with 5,604 source-quality-clean markets; its pre-result protocol must be committed and pushed before the authorized conditional economics run. T0009 remains BLOCKED for full-development coverage and candidate-readiness provenance. E0034/E0024 BLOCKED; E0010 PLANNED/deferred. Other outcomes: EXPERIMENTS. Stale `running` manifests do not prove active processes.

## Active decisions

D0001-D0008, D0011-D0014, D0017-D0019 govern splits/exposure, targets/timing, execution/fees/accounting, causal selection, candidate identity, source provenance, active resolution economics, the holdout-before-pilot gate, and verified minimum-order units. D0016 continues to govern inactive sale-simulation accounting only. D0010 supersedes the old arbitrary 180-day A/B/C/D shadow recommendation. D0009 is the separate deferred older prospective protocol.

## Protected invariants

UTC boundaries and label availability; historical exposure of confirmation/holdout; no development on protected data; frozen feature/model/seed identities; T-60 decision vs T-59 execution; continuity without future quotes; correct fee debit/minimum units; $100 self-financing portfolios; $5 and 5%-of-free-cash controls; evaluated/deployment bundle separation; no live activation/order authority. Exact sources, dates and change conditions: DECISIONS.md invariant register.

## Superseded / abandoned approaches

Legacy market-open/Kacho, original fees/minimum units/T59 selection, interrupted VP/RP ablation/tuning, negative RECENT365 and unpromoted ensemble/FULL_FEATURES/probe portfolios: E0001-E0034 and audit section 10. Their history is retained.

## Next recommended actions

1. T0009: commit and push E0038 v2 before computing its conditional A/B/C/D economics on the 5,604-market clean subset; then retain the full-development coverage and readiness blockers. Do not use holdout data or acquire sources.
