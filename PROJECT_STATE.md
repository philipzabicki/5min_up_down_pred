# PROJECT STATE

Last audited: 2026-10-27 (Europe/Warsaw).
Audit commit: the commit introducing this file, `docs: recover canonical project state` (resolve with `git log -1 --format=%H -- PROJECT_STATE.md`; no self-referential hash).
Audit input HEAD: `e74d70997ef75eba0f818fb18d209817897e2f36` on `audit/btc-dynamic-history-20261020`; upstream was `origin/audit/btc-dynamic-history-20261020` and the tree was clean before this task.

## Objective

Predict BTC five-minute Binance-index direction before a Polymarket market opens, then assess official-outcome prediction quality and executable after-fee economics separately. The active BTC economic strategy is buy UP/DOWN at the frozen T-59 execution and hold to official Gamma resolution. The repository also contains ETH/SOL live and general modeling tooling.

## Current phase

Historical research and frozen A/B/C/D comparison; **no demonstrated production readiness or repeatable trading advantage**. A=BASE70, B=A+simple-memory PCA8, C=A+ESN PCA8, D=A+memory+ESN. E0036's 1,143-market confirmation result remains sparse and unchanged. E0037's saved source-parity gate remains **FAILED**. E0038 completed retrospective, coverage-conditional economics on 5,604 source-quality-clean markets. Its corrected comparison matches all row-level probability, decision, fill, and fee values on 1,141 common E0036 markets. E0039 records month/concentration and paired diagnostics, freezes three development chronology folds, and adds a tested label-availability fit mask. Its first v2 invocation stopped at the saved RAM gate; a later retry passed the live resource gate but stopped at an allowlist-hash mismatch before any fit, label, Gamma outcome, or economic-source read. A v3 protocol corrects the recorded hash semantics without changing the 5,604-ID cohort or procedure; its regression and full-suite checks pass. F1/F2/F3, OOF calibration, outer predictions, economics, and the offline runtime benchmark remain unrun. See the [E0039 audit](reports/btc_preopen/btc_chronology_stability_reconstruction_20261027.md), [v2 execution report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v2.md), [v3 correction report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v3.md), and [v3 protocol](configs/research/btc_chronology_reconstruction_20261027_v3.json). T0009 remains blocked overall; source, chronology and runtime axes have separate statuses. T0008/E0035 sale-policy work is deferred/out-of-scope under D0018. The old A/B/C/D no-order pilot remains historically blocked at 0/3 and is not an active next step under D0017. Dynamic-history/action-value research remains independently blocked before fitting.

Registers: [DECISIONS](DECISIONS.md), [EXPERIMENTS](EXPERIMENTS.md), [TASKS](TASKS.md). Details/matrices/provenance: [audit report](docs/PROJECT_STATE_AUDIT_20261025.md), [evidence](reports/project_state_recovery_20261025.json). The [BTC status](docs/BTC_RESEARCH_STATUS_20261010.md) remains the checkpoint history; [research contract](docs/BTC_RESEARCH_CONTRACT.md) remains binding.

## Verified working

- Full discovery at audited HEAD: **675 tests, 665 PASS, 2 FAIL, 8 ERROR, 0 SKIP, 0 XFAIL**, 51.735 s runner / 62.653 s wall. This is the preserved baseline; follow-up results below supersede its test status.
- Follow-up on 2026-10-25: **680 tests, 680 PASS, 0 FAIL, 0 ERROR, 0 SKIP, 0 XFAIL**, 45.157 s, exit code 0 after fixes T0001-T0003.
- Sale-ledger regression on 2026-10-26: the focused PMXT initialization and exit-accounting modules passed **30/30** tests after correcting the no-sale settlement fallback; full discovery passed **680/680** tests in **40.247 s**, exit code 0. This validates mechanics only.
- E0036 resolution ledger: settlement regression **3/3**, existing replay/accounting tests **24/24**, and full discovery **683/683** passed (51.332 s, exit code 0). The hash-verified sparse confirmation run reproduced E0031 fixed-$5 economics and separately computed the 5%-of-free-cash control, cost-basis drawdown and event accounting. This validates code mechanics, not strategy profitability or production readiness.
- E0037 source-only validation on 2026-10-26: focused T0009 tests **10/10**, T-59 reassessment tests **34/34**, and full discovery **696/696** passed (55.837 s, exit code 0). The full local development PMXT snapshot was hash/provenance verified and reused. E0036 T-60 source parity still failed; no economic results were produced.
- E0038 validation: focused source/economics tests 13/13 and full discovery 709/709 passed (43.327 s, exit code 0). V1 stopped before data/outcome loading; v2 and v3 were committed/pushed before their runs. V3 results match v2 economics and correct its row-comparison label; all portfolio accounting invariants passed.
- E0039 chronology-mask focused checks: **51/51 PASS** across pre-open contract, development-boundary, ESN experiment-contract, candidate checkpoint/restart, and resolution-portfolio modules. Full `python -m unittest discover -s tests`: **711/711 PASS**, 0 FAIL/ERROR/SKIP/XFAIL, **41.334 s**, exit 0. This validates code mechanics, not chronology reproduction or runtime readiness.
- E0039 v2 runner focused tests: **17/17 PASS**. Full `python -m unittest discover -s tests`: **728/728 PASS**, 0 FAIL/ERROR/SKIP/XFAIL, **54.381 s**, exit 0; `py_compile` also passed. The required no-argument runner then stopped before fit at its resource gate; see the [v2 run report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v2.md). Tests validate mechanics only, not research results.
- E0039 v3 hash correction: focused chronology tests **18/18 PASS**; full discovery **729/729 PASS** in **48.090 s** (exit 0); `py_compile` passed. The v3 freeze verified 53 implementation/source files and the 5,604-ID whitelist hash `e2946aa8e45c2ed050347e18756408a0175fa0eb9859a120f8df7c173051edda`. This validates the correction and runner mechanics only; chronology results remain unrun.
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
- T0009 chronology axis (E0039): the v2 retry passed the live resource gate and verified its frozen files, then stopped before fitting at a clean-ID hash mismatch. The v3 protocol now pins the canonical sorted-ID hash and updates the runner's protocol identity; its test and full suite pass. No fold fit, official OOF calibration, outer predictions, or economic replay has run. Commit and push v3 before rerunning `python run_btc_chronology_reconstruction_20261027.py`; the runner must repeat the resource gate. Missing historical `prediction_available_at` is not the chronology blocker.
- T0009 runtime axis (E0039): the batch runner is implemented, but the offline streaming adapter remains `PARTIAL_NOT_IMPLEMENTED`; no full-path benchmark or latency percentiles have been measured. The deterministic 300-window sample is frozen. No one-second conclusion is claimed.
- Resolution economics: E0038 confirms 19,431 T−60 and 26,289 T−59 source-value parity comparisons with no mismatches, while distinguishing old unchecked rows from passed checks. Its symmetric source-quality rule leaves 5,604 of the 5,684 retrospective source intersection; the T−60 ask/BBO conflict remains unresolved and quarantined. The completed A/B/C/D economic paths and four payout scenarios are coverage-conditional; no measured prediction-ready timestamps exist, full-pipeline selection/calibration independence is not established, and 53,660 prediction calendar rows have no Gamma ID. The 24 missing local PMXT hours remain separately inventoried under T0010. These results do not close T0009. Actual redemption availability remains unknown.
- Sale-policy study: T0008/E0035 are deferred/out-of-scope, not blockers for the active resolution strategy. The sale simulator still keeps unfilled exits open and never treats them as settlement.
- Shadow: bounded 2026-10-25 same-interpreter preflight reached Binance time (HTTP 200, valid schema) but Gamma and CLOB DNS resolution failed. No collector was started; pilot remains 0/3. Under D0017 this is retained history, not an active next step (T0004).
- Exact historical Chainlink TWAP access/coverage/cost and TWAP60 L2 missing; no acquisition authorized (T0005).

## Unknown / needs verification

Real fills/ACKs/fees and operational live readiness; full-pipeline independent OOS performance; prospective profitability; raw archive completeness beyond inspected manifests; individual rule text for 79,643 of 79,650 calendar-assigned markets; exact cause of 4,193 Binance/Gamma label disagreements; reproducibility on a clean clone and another OS. Large ignored artifacts are local dependencies, not delivered by Git.

## Active experiments

E0036 PARTIAL on the available confirmation subset. E0037 PARTIAL with its original FAILED parity gate preserved. E0038 completed its corrected retrospective economics on 5,604 source-quality-clean markets. E0039 completed the E0038 stability diagnosis and implemented/pushed the v2 chronology runner and protocol. Its resource gate now passes, but the v2 retry exposed a protocol ID-hash mismatch; v3 corrects it and is validated but not yet run. No fold results exist. The offline runtime adapter/benchmark remains partial. T0009 remains BLOCKED overall for source coverage/readiness; chronology and runtime have separate statuses. E0034/E0024 BLOCKED; T0010 DEFERRED. Other outcomes: EXPERIMENTS. Stale running manifests do not prove active processes.

## Active decisions

D0001-D0008, D0011-D0014, D0017-D0019 govern splits/exposure, targets/timing, execution/fees/accounting, causal selection, candidate identity, source provenance, active resolution economics, the holdout-before-pilot gate, and verified minimum-order units. D0016 continues to govern inactive sale-simulation accounting only. D0010 supersedes the old arbitrary 180-day A/B/C/D shadow recommendation. D0009 is the separate deferred older prospective protocol.

## Protected invariants

UTC boundaries and label availability; historical exposure of confirmation/holdout; no development on protected data; frozen feature/model/seed identities; T-60 decision vs T-59 execution; continuity without future quotes; correct fee debit/minimum units; $100 self-financing portfolios; $5 and 5%-of-free-cash controls; evaluated/deployment bundle separation; no live activation/order authority. Exact sources, dates and change conditions: DECISIONS.md invariant register.

## Superseded / abandoned approaches

Legacy market-open/Kacho, original fees/minimum units/T59 selection, interrupted VP/RP ablation/tuning, negative RECENT365 and unpromoted ensemble/FULL_FEATURES/probe portfolios: E0001-E0034 and audit section 10. Their history is retained.

## Next recommended actions

1. E0039 / T0009 chronology: commit and push the v3 protocol/hash correction, repeat the live resource gate, then resume with `python run_btc_chronology_reconstruction_20261027.py`. It computes development-only source-row hashes before fitting F1/F2/F3. Keep E0038's conditional classification; no holdout or source acquisition. Track the still-partial offline runtime adapter separately.
