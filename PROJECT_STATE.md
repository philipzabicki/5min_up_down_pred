# PROJECT STATE

Last audited: 2026-10-25 (Europe/Warsaw).
Audit commit: the commit introducing this file, `docs: recover canonical project state` (resolve with `git log -1 --format=%H -- PROJECT_STATE.md`; no self-referential hash).
Current HEAD at audit input: `cf885b0428e0c6041d471e6bca97597ff04dc4ec` on `audit/btc-dynamic-history-20261020`.

## Objective

Predict BTC five-minute Binance-index direction before a Polymarket market opens, then assess official-outcome prediction quality and executable after-fee economics separately. The repository also contains ETH/SOL live and general modeling tooling.

## Current phase

Historical research and frozen A/B/C/D comparison; **no demonstrated production readiness or repeatable trading advantage**. A=BASE70, B=A+simple-memory PCA8, C=A+ESN PCA8, D=A+memory+ESN. The latest no-order shadow collector is implemented but its pilot is blocked; its database contains zero live windows. Dynamic-history/action-value research is independently blocked before fitting.

Registers: [DECISIONS](DECISIONS.md), [EXPERIMENTS](EXPERIMENTS.md), [TASKS](TASKS.md). Details/matrices/provenance: [audit report](docs/PROJECT_STATE_AUDIT_20261025.md), [evidence](reports/project_state_recovery_20261025.json). The [BTC status](docs/BTC_RESEARCH_STATUS_20261010.md) remains the checkpoint history; [research contract](docs/BTC_RESEARCH_CONTRACT.md) remains binding.

## Verified working

- Full discovery at audited HEAD: **675 tests, 665 PASS, 2 FAIL, 8 ERROR, 0 SKIP, 0 XFAIL**, 51.735 s runner / 62.653 s wall. This is the preserved baseline; follow-up results below supersede its test status.
- Follow-up on 2026-10-25: **680 tests, 680 PASS, 0 FAIL, 0 ERROR, 0 SKIP, 0 XFAIL**, 45.157 s, exit code 0 after fixes T0001-T0003.
- Read-only candidate shadow SQLite: two preflight events, zero data/market/checkpoint rows (E0034).
- Completed frozen scorer/period/approval/source records exist and were hashed; historical scores were not recomputed (E0030).

## Implemented but not validated

- Acquisition/features/training/calibration/replay/accounting/live orders/redemption have code/tests; complete reference/live validation UNKNOWN.
- Saved memory/ESN parity covers 2,660,546 development raw rows; large replay not repeated (E0029).
- All primary adjusted B-A/D-B historical intervals cross zero (E0030/E0031).
- Sparse economics: 1,143 confirmation/single-price markets, zero holdout/TWAP markets; B +$28.01 / D +$22.66 simulated independent PnL, actual readiness unknown (E0031/E0032).

## Partially implemented

- Complete independent nested pipeline validation is not established by conditional OOF or one-pass fitting (D0005/E0013).
- Earlier model-update B stopped below its successful-trial minimum; no B holdout score (E0009).
- Shadow timing code now uses the T-60 information cutoff and T-59 readiness deadline, retains actual event/receive/ready timestamps, and rejects late books; live timing remains unmeasured because endpoint preflight failed (T0007/T0004).
- Deployment refit code exists; configured original holdout prerequisite/output absent (T0006).

## Cleared test failures

- T0001 ESN shared-module state is restored after the candidate checkpoint test; suspect test order now passes.
- T0002 stale test expectations were aligned to the fixed 2-tick code/runtime value; runtime policy was not changed.
- T0003 fake submit fixture now matches real trader initialization. This was not a real exchange failure.
- README names absent `optimize_lgbm_optuna.py` and `.python-version`; current optimizer is `optimize_lgbm_hyperparameters.py` (T0006).

## Blocked

- Dynamic gate: 637/2,000 evaluation; fold1 $5 training 908/1,000; RAW timing provenance missing. No model result (E0024).
- Shadow: bounded 2026-10-25 same-interpreter preflight reached Binance time (HTTP 200, valid schema) but Gamma and CLOB DNS resolution failed. No collector was started; pilot remains 0/3 (T0004).
- Exact historical Chainlink TWAP access/coverage/cost and TWAP60 L2 missing; no acquisition authorized (T0005).

## Unknown / needs verification

Real fills/ACKs/fees and operational live readiness; full-pipeline independent OOS performance; prospective profitability; raw archive completeness beyond inspected manifests; individual rule text for 79,643 of 79,650 calendar-assigned markets; exact cause of 4,193 Binance/Gamma label disagreements; reproducibility on a clean clone and another OS. Large ignored artifacts are local dependencies, not delivered by Git.

## Active experiments

No new run. E0034/E0024 BLOCKED; E0010 PLANNED/deferred. Other outcomes: EXPERIMENTS. Stale `running` manifests do not prove active processes.

## Active decisions

D0001-D0008 and D0011-D0014 govern splits/exposure, targets/timing, execution/fees/accounting, causal selection, candidate identity, source provenance, and audit scope. D0010 supersedes the old arbitrary 180-day A/B/C/D shadow recommendation. D0009 is the separate deferred older prospective protocol.

## Protected invariants

UTC boundaries and label availability; historical exposure of confirmation/holdout; no development on protected data; frozen feature/model/seed identities; T-60 decision vs T-59 execution; continuity without future quotes; correct fee debit/minimum units; $100 self-financing portfolios; $5 and 5%-of-free-cash controls; evaluated/deployment bundle separation; no live activation/order authority. Exact sources, dates and change conditions: DECISIONS.md invariant register.

## Superseded / abandoned approaches

Legacy market-open/Kacho, original fees/minimum units/T59 selection, interrupted VP/RP ablation/tuning, negative RECENT365 and unpromoted ensemble/FULL_FEATURES/probe portfolios: E0001-E0034 and audit section 10. Their history is retained.

## Next recommended actions

1. T0004/T0007: after Gamma/CLOB DNS access is restored, rerun bounded schema preflight and complete the frozen no-order three-window pilot; collect real receipt/compute/book timing before closing T0007.
2. T0005: verify exact historical TWAP/L2 source access, identity, coverage and cost before any separately authorized acquisition.
