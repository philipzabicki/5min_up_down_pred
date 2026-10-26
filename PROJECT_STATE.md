# PROJECT STATE

Last audited: 2026-10-26 (Europe/Warsaw).
Audit commit: the commit introducing this file, `docs: recover canonical project state` (resolve with `git log -1 --format=%H -- PROJECT_STATE.md`; no self-referential hash).
Audit input HEAD: `e74d70997ef75eba0f818fb18d209817897e2f36` on `audit/btc-dynamic-history-20261020`; upstream was `origin/audit/btc-dynamic-history-20261020` and the tree was clean before this task.

## Objective

Predict BTC five-minute Binance-index direction before a Polymarket market opens, then assess official-outcome prediction quality and executable after-fee economics separately. The active BTC economic strategy is buy UP/DOWN at the frozen T-59 execution and hold to official Gamma resolution. The repository also contains ETH/SOL live and general modeling tooling.

## Current phase

Historical research and frozen A/B/C/D comparison; **no demonstrated production readiness or repeatable trading advantage**. A=BASE70, B=A+simple-memory PCA8, C=A+ESN PCA8, D=A+memory+ESN. E0036 calculated the existing 1,143-market confirmation intersection under resolution accounting; it is sparse, previously exposed and not full-development evidence. Full development economics remain blocked by T-60 source coverage and causal-provenance gaps (T0009), not by absent sale rules. T0008/E0035 sale-policy work is deferred/out-of-scope under D0018. The old A/B/C/D no-order pilot remains historically blocked at 0/3 and is not an active next step under D0017. Dynamic-history/action-value research remains independently blocked before fitting.

Registers: [DECISIONS](DECISIONS.md), [EXPERIMENTS](EXPERIMENTS.md), [TASKS](TASKS.md). Details/matrices/provenance: [audit report](docs/PROJECT_STATE_AUDIT_20261025.md), [evidence](reports/project_state_recovery_20261025.json). The [BTC status](docs/BTC_RESEARCH_STATUS_20261010.md) remains the checkpoint history; [research contract](docs/BTC_RESEARCH_CONTRACT.md) remains binding.

## Verified working

- Full discovery at audited HEAD: **675 tests, 665 PASS, 2 FAIL, 8 ERROR, 0 SKIP, 0 XFAIL**, 51.735 s runner / 62.653 s wall. This is the preserved baseline; follow-up results below supersede its test status.
- Follow-up on 2026-10-25: **680 tests, 680 PASS, 0 FAIL, 0 ERROR, 0 SKIP, 0 XFAIL**, 45.157 s, exit code 0 after fixes T0001-T0003.
- Sale-ledger regression on 2026-10-26: the focused PMXT initialization and exit-accounting modules passed **30/30** tests after correcting the no-sale settlement fallback; full discovery passed **680/680** tests in **40.247 s**, exit code 0. This validates mechanics only.
- E0036 resolution ledger: settlement regression **3/3**, existing replay/accounting tests **24/24**, and full discovery **683/683** passed (51.332 s, exit code 0). The hash-verified sparse confirmation run reproduced E0031 fixed-$5 economics and separately computed the 5%-of-free-cash control, cost-basis drawdown and event accounting. This validates code mechanics, not strategy profitability or production readiness.
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
- Resolution economics: full-development T-60 entry coverage and historical prediction-ready telemetry remain incomplete. E0036 covers only 1,143 confirmation markets (2.22% of confirmation official labels), from 2026-04-15 through 2026-05-18; T0009 is the active gate. Actual redemption availability is unknown; E0036 reused the existing 60-second payout-availability assumption and does not claim it bounds real redemption latency.
- Sale-policy study: T0008/E0035 are deferred/out-of-scope, not blockers for the active resolution strategy. The sale simulator still keeps unfilled exits open and never treats them as settlement.
- Shadow: bounded 2026-10-25 same-interpreter preflight reached Binance time (HTTP 200, valid schema) but Gamma and CLOB DNS resolution failed. No collector was started; pilot remains 0/3. Under D0017 this is retained history, not an active next step (T0004).
- Exact historical Chainlink TWAP access/coverage/cost and TWAP60 L2 missing; no acquisition authorized (T0005).

## Unknown / needs verification

Real fills/ACKs/fees and operational live readiness; full-pipeline independent OOS performance; prospective profitability; raw archive completeness beyond inspected manifests; individual rule text for 79,643 of 79,650 calendar-assigned markets; exact cause of 4,193 Binance/Gamma label disagreements; reproducibility on a clean clone and another OS. Large ignored artifacts are local dependencies, not delivered by Git.

## Active experiments

E0036 PARTIAL on the available confirmation subset. E0034/E0024 BLOCKED; E0010 PLANNED/deferred. Other outcomes: EXPERIMENTS. Stale `running` manifests do not prove active processes.

## Active decisions

D0001-D0008, D0011-D0014, D0017, and D0018 govern splits/exposure, targets/timing, execution/fees/accounting, causal selection, candidate identity, source provenance, active resolution economics, and the holdout-before-pilot gate. D0016 continues to govern inactive sale-simulation accounting only. D0010 supersedes the old arbitrary 180-day A/B/C/D shadow recommendation. D0009 is the separate deferred older prospective protocol.

## Protected invariants

UTC boundaries and label availability; historical exposure of confirmation/holdout; no development on protected data; frozen feature/model/seed identities; T-60 decision vs T-59 execution; continuity without future quotes; correct fee debit/minimum units; $100 self-financing portfolios; $5 and 5%-of-free-cash controls; evaluated/deployment bundle separation; no live activation/order authority. Exact sources, dates and change conditions: DECISIONS.md invariant register.

## Superseded / abandoned approaches

Legacy market-open/Kacho, original fees/minimum units/T59 selection, interrupted VP/RP ablation/tuning, negative RECENT365 and unpromoted ensemble/FULL_FEATURES/probe portfolios: E0001-E0034 and audit section 10. Their history is retained.

## Next recommended actions

1. T0009: inventory and verify existing local development T-60/T-59 sources, market/token mapping, official outcomes and causal cutoffs; produce exact coverage gaps before any additional development economics. No holdout evaluation or data acquisition is included.
