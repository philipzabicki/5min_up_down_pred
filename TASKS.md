# Evidence-based tasks

Recovered 2026-10-25 at `cf885b0`. These are observed gaps, not authorization to implement or run experiments. IDs are stable. DONE requires the stated acceptance evidence; a passing test subset cannot close a full-suite task. Consult [PROJECT_STATE](PROJECT_STATE.md), [DECISIONS](DECISIONS.md), and the relevant [experiment](EXPERIMENTS.md) first.

## Superseded continuation plan - 2026-10-25

Scope: continue T0001-T0005 and T0007 from the recovered HEAD; preserve D0001-D0014 and frozen A/B/C/D. No protected-data evaluation, training, tuning, replay, orders, or paid acquisition.

Disposition 2026-10-26: the former step 4 (repair DNS, then run the three-window shadow pilot) is superseded by D0017. Preserve its T0004/E0034 historical blocker and 0/3 result; do not resume it as the next action.

1. Reproduce and classify the eight T0001 errors and two T0002/T0003 failures, using fresh and suspect-order runs where relevant; resolve intended behavior from code, tests, active decisions and history before editing. Acceptance: focused regressions pass without masking or changing protected semantics.
2. Check the current VP optimization input boundary and available manifests only; establish current admissibility separately from historical execution/model lineage. Acceptance: evidence-backed protection result, or explicit UNKNOWN, without reading protected feature rows.
3. Trace T0007 event/receive/compute/ready/book times against D0003 and the binding contract. Make only a contract-aligned minimal correction if the authority is unambiguous; cover deadline, late/missing data or book, and restart idempotency. Acceptance: actual timestamps remain causal and late windows retain rejection reasons.
4. If critical timing and no-order checks pass, freeze the existing shadow manifest before one predeclared series of three consecutive BTC 5m windows, run bounded public preflight/collection with the dedicated collector, then close it and report all planned/observed/complete/timely/resolved counts. Stop on a recorded blocker; do not substitute windows or retry until success.
5. Independently inspect existing TWAP/L2 indexes, manifests and bounded samples plus RAW OOF provenance; do not acquire data. Acceptance: short source-availability matrix, exact missing evidence, and correct interpretation of the 637/2,000 and 908/1,000 gates.
6. Run the required focused checks and one full unittest discovery after fixes; record environment and PASS/FAIL/ERROR/SKIP/XFAIL counts. Update the existing task, experiment, decision (only if needed), and project-state records with evidence. Commit and normally push only this work.

## Current continuation plan - 2026-10-26

Scope: continue the historical BTC evaluation under D0017/D0018. Preserve frozen candidates, official Gamma outcomes, fee rules and split boundaries. No new holdout evaluation, model scoring/training, tuning, full replay, external data acquisition, collector, DNS pilot check, shadow/live process, or order.

1. E0036 reuses the hash-identified E0031 confirmation ledger and calculates economics only for its 1,143-market common T-60/T-59 subset. Its positive B/C/D points are sparse, previously exposed diagnostics; they do not close full-development validation.
2. T0009: establish exact local T-60 entry and causal source coverage for the full frozen development interval. Reuse existing PMXT/Kacho and token mapping; record gaps before any additional calculation. Do not fabricate prediction-ready timestamps or use holdout results.
3. Prepare a separate fixed-holdout resolution-economics evaluation only after the development coverage gate is met. The existing holdout is exposed and current exact T-60 economic coverage is zero; any new evaluation requires a separately approved scope.
4. T0004 remains outside the active queue. Do not start live/shadow consideration before a positive fixed-holdout result and separate explicit authorization.

## T0000 - Recover canonical project state

Status: DONE.
Priority: P0.
Reason: conflicting historical checkpoint/status claims and no unified current snapshot/registers.
Depends on: none.
Acceptance criteria: inspect initial Git state/full available history, distinguish implementation/tests/validation and experiment outcomes, run safe existing tests, record data protection/provenance, decisions/invariants, false completion and unknowns; provide canonical navigation and a documentation-only commit.
Evidence required: [audit report](docs/PROJECT_STATE_AUDIT_20261025.md), [machine evidence](reports/project_state_recovery_20261025.json), four canonical registers and agent rules. Commit is the one introducing this task; resolve with `git log -1 --format=%H -- TASKS.md`. Publishing outcome is reported by the audit completion message.
Completion scope: state recovery completed even though tests are red and research/operational validation remains incomplete. No implementation bug was fixed.

## T0001 - Restore ESN test isolation across full discovery

Status: DONE (2026-10-25).
Priority: P1.
Reason: full discovery has eight ESN errors although architecture (7 tests) and expansion (8) each pass in isolated processes. `test_btc_candidate_holdout_20261022.py::test_gap_and_checkpoint_restart_match_uninterrupted_dynamic_states` calls `quality._configure_source_run`; that calls `focused._configure_engines`, overwriting shared expansion/architecture globals and callbacks without test restoration.
Depends on: none. Relevant: E0026-E0030, D0008/D0014.
Acceptance criteria: under unchanged candidate methods/artifacts, full discovery has no ESN state-dependent errors; affected tests pass alone and after the candidate checkpoint test; no writes to frozen local research outputs; no protected-row reads introduced.
Evidence required: exact commands, counts and timings for full suite and the order-sensitive reproduction; scoped diff explaining restored global state. Resolve the separate failures in T0002/T0003 before claiming the entire suite passes.
Evidence: scoped `patch.dict` restoration in the candidate checkpoint test restores the focused ESN, expansion ESN, and expansion engine-runner module globals. Before the fix, fresh ESN tests passed (15/15), candidate-checkpoint-first reproduced all 8 errors, and reverse order passed; afterward the candidate checkpoint followed by the ESN modules passed (16/16). Full discovery after all three repairs: 680 tests, 680 PASS, 0 FAIL/ERROR/SKIP/XFAIL, 45.157 s, exit code 0.

## T0002 - Resolve fixed-slippage test/configuration discrepancy

Status: DONE (2026-10-25).
Priority: P1.
Reason: `test_optimize_trade_policy.py::OptimizeTradePolicyTests::test_run_optimization_saves_runtime_config_with_fixed_slippage_ticks` expects 3; `optimize_trade_policy.py::SUBMITTED_PRICE_SLIPPAGE_TICKS` is 2. It fails both full and isolated runs. An audit cannot decide whether the test expectation or intended policy should change.
Depends on: none. Relevant: D0003/D0006/D0007.
Acceptance criteria: establish intended fixed-slippage semantics from policy authority/history, align implementation/test accordingly without incidental live configuration changes; preserve per-asset write isolation and unchanged search scope.
Evidence required: source of the intended value and passing focused/full checks; any methodological change needs its own recorded decision. Do not optimize new policies to resolve a constant assertion.
Evidence: `optimize_trade_policy.py::SUBMITTED_PRICE_SLIPPAGE_TICKS`, the active BTC/ETH/SOL runtime JSON, and same-commit history all set 2; the test's expectation of 3 was stale. Updated only those fixed-value test expectations. `tests.test_optimize_trade_policy`: 6/6 PASS; full discovery: 680/680 PASS. No policy/runtime config changed and no optimization ran.

## T0003 - Restore mocked submit-success telemetry coverage

Status: DONE (2026-10-25).
Priority: P1.
Reason: `test_run_multi_asset_latency.py::LiveSubmitTelemetryTests::test_successful_submit_response_is_distinct_from_send_and_fill_events` constructs a trader with `__new__` and lacks `pm_user_trade_monitor`; `_maybe_submit_order` returns `submission_error` instead of `submitted_fok`. Fake-only diagnostic reproduced the missing-attribute error. This is not evidence of an actual exchange rejection.
Depends on: none. Relevant: E0007, D0007.
Acceptance criteria: identify intended initialization behavior; correct fixture or implementation in separate work; successful mocked response has call bounds/order identity and leaves unobserved transport/ACK/fill timestamps unknown; no network/order calls required.
Evidence required: passing success/error telemetry tests and full discovery; no live order submitted.
Evidence: the fake trader fixture now initializes `pm_user_trade_monitor = None`, matching the production initializer while retaining the existing mocked submit assertions. `tests.test_run_multi_asset_latency`: 15/15 PASS; full discovery: 680/680 PASS. No client or order call was made.

## T0004 - Verify endpoint reachability and complete no-order shadow pilot

Status: BLOCKED (2026-10-25; public Gamma/CLOB DNS failure).
Priority: P1.
Reason: the bounded same-interpreter preflight at 2026-10-25T22:44Z reached Binance time (HTTP 200, valid `serverTime` JSON) but DNS resolution failed for `gamma-api.polymarket.com` and `clob.polymarket.com`. Gamma token/schema lookup and CLOB book schema therefore could not be verified. Current DB remains at two prior preflight events and zero market, snapshot, prediction, attempt, outcome, candle, or checkpoint rows. No collector was started.
Depends on: T0007 for readiness semantics; preserve D0008/D0010 frozen identities; perform independent mock tests after T0001 if implementation changes are needed.
Acceptance criteria: diagnose public endpoint access without credentials/orders; under separately continued collection scope capture at least three complete scheduled windows with exact Gamma market/token mapping, timely actual prediction timestamps, both T60/T59 snapshots and later official outcomes; report all failed/late attempts, measured latency and storage growth. Leave actual fills UNKNOWN. Preserve protocol and no-order behavior.
Evidence required: preflight responses/timestamps, read-only DB counts/sample identities, runner/model/config hashes, per-stage runtime timing, restart/checkpoint evidence, and explicit collector process state/stop procedure. A synthetic test or successful Binance-only probe cannot close this task.
2026-10-25 disposition: no window was scheduled or substituted after the public endpoint gate failed. Actual pilot count remains 0/3; observed/complete/timely/resolved counts are all zero. Fills remain UNKNOWN. Resume only after Gamma and CLOB DNS access is restored and a fresh bounded schema preflight passes.
2026-10-26 disposition under D0017: retain the blocked 0/3 history, but do not resume this pilot or run DNS diagnostics as the next step. Reconsider only after positive sale-based economics on the fixed holdout and separate explicit user authorization.

## T0005 - Establish exact historical TWAP and L2 source feasibility

Status: BLOCKED.
Priority: P2.
Reason: 4,193 saved label disagreements lack official price-print reconstruction; exact TWAP30/TWAP60 historical feed IDs/access/retention/cost are unconfirmed. Local PMXT ends before TWAP60; current historical candidate execution ledger has no TWAP60 rows.
Depends on: explicit source/access scope and, if acquisition is needed, authorization for any account, credentials, fee or large download; none granted by this audit. Independent of T0004 and unrelated research.
Acceptance criteria: document exact official feed identities and obtainable historical periods for the frozen 43-market sample; separately document two-token L2 coverage and initialized event continuity for the target historical period; obtain concrete access/cost/size information before acquisition. If unavailable, preserve UNKNOWN price cause and absent economics. Do not tune models or change thresholds to fill a data gap.
Evidence required: source metadata, sample/manifest checksums, timestamps/coverage and quote orientation, written cost/entitlement evidence, then separately authorized reconstruction checks. Provider marketing or one out-of-period Vike sample is insufficient for target-period validation.
2026-10-25 reinspection: retained source matrix and RAW OOF report remain unchanged; no new public historical evidence, account access, purchase, or download was obtained. Existing PMXT/Vike/PMData/Marketlens and RAW OOF limitations remain as stated in E0034 and E0024.

## T0006 - Reconcile legacy entrypoints and reproducibility prerequisites

Status: READY.
Priority: P2.
Reason: README references missing `optimize_lgbm_optuna.py`/`.python-version`; current optimizer name differs. Deployment refit config points to original `preopen_research_20261011/evaluated_holdout_manifest.json`, absent locally, whereas completed retry/correction records are elsewhere. Most data/models/caches are ignored and requirements are unpinned.
Depends on: audit registers and D0001/D0007; no refit execution authorized.
Acceptance criteria: document correct commands/interpreter/dependency prerequisites and intended refit lineage without substituting bundles silently; identify all required local artifacts and how to restore them within data/authorization constraints. Distinguish a read-only reproducibility check from running training or refit.
Evidence required: verified links/entrypoints, manifest compatibility and model/config identity review, documented missing assets; clean-clone/cross-platform status remains NOT_VERIFIED_IN_THIS_AUDIT until actually tested. Do not close by installing a different GPU stack or replacing the user's environment.

## T0007 - Reconcile shadow readiness deadline with the one-second contract

Status: PARTIAL (contract/code/tests resolved; real timing measurement blocked with T0004).
Priority: P1.
Reason: binding D0003 and `features/btc_preopen_contract.py` distinguish the nominal T-60 information cutoff from prediction readiness through T-59. The former shadow implementation incorrectly used T-60 as the readiness cutoff. Its book requests are one-shot public REST snapshots; a response received after its T-60/T-59 boundary cannot be treated as available at that boundary.
Depends on: D0003/D0010 and frozen protocol review; precedes T0004 pilot-validation claims.
Acceptance criteria: establish and record the intended difference (or required correction) between nominal decision time, data cutoff, actual prediction readiness and execution time. Preserve causality and actual timestamps; do not simply backdate or loosen a gate to obtain passing windows. If behavior changes, use explicit decision/protocol identity and separately authorized implementation work.
Evidence required: source-of-authority comparison, deterministic boundary cases and then actual no-order receipt/compute timing; report whether the intended path can produce eligible decisions. A unit assertion encoding the current strict deadline is not reference validation of that deadline.
Evidence: candidate shadow now accepts only T-120 input candles whose two Binance kline close events are no later than the T-60 information cutoff, retains actual UTC receive/event/ready times, measures durations and pair skew with `time.monotonic`, and uses T-59 as the prediction-ready deadline. T-60/T-59 REST books retain actual request/receive/source timestamps; late or missing books are not priceable. Boundary, late candle/readiness/book, missing-book and duplicate-window tests pass (`tests.test_btc_candidate_shadow_20261023`: 14/14; `tests.test_btc_prospective_shadow`: 19/19). Real receipt/compute timing remains unmeasured because T0004 public preflight failed DNS.

## T0008 - Freeze sale-based position lifecycle before economics

Status: DEFERRED / OUT-OF-SCOPE for the active strategy (D0018); historical sale gate remains BLOCKED.
Priority: P0.
Reason: the binding contract previously released positions at official resolution, which D0016 now prohibits. The existing A/B/C/D economic diagnostic used settlement outcomes; older `close_at_60s` and threshold studies are not the active strategy. No current sale trigger/redecision schedule or maximum position horizon is frozen.
Depends on: D0003/D0004/D0006/D0016; preserve D0001/D0008/D0011 and D0017. No holdout or live/shadow work.
Acceptance criteria: record the exact sale trigger and decision times, maximum holding horizon, order type and delay, partial/unfilled-fill rules, fee handling, remaining-share treatment, and overlapping-position cash lock/release. Before any outcome review, freeze the candidate IDs, cohort/splits, entry policy, execution path, capital controls, costs, coverage gates, and metrics. Then use existing manifests to produce per-market/token/hour required-range gaps; do not invent missing quotes or use official resolution as an exit.
Evidence required: source-backed active contract, frozen protocol/hash, testable execution semantics, and exact market-path inventory. Do not run a development replay while the exit contract is unresolved. Any later holdout result requires its own approved scope; it does not authorize a pilot.
2026-10-26 historical evidence: E0035 and [the sale-strategy gate report](reports/btc_preopen/btc_historical_sale_strategy_gate_20261026.md) confirmed that no sale trigger/redecision schedule or maximum holding horizon was frozen. The no-fill-to-resolution fallback was corrected and remains covered. D0018 supersedes sale-policy work for the active objective; missing sale terms are not an active project blocker. Do not resume unless the user later reopens sale research.

## T0009 - Establish full-development coverage for resolution economics

Status: BLOCKED (entry-source coverage and prediction-readiness provenance incomplete; E0036).
Priority: P0.
Reason: the reusable E0031 economic ledger contains 1,143 common confirmation markets (2026-04-15 through 2026-05-18), not the full frozen development interval. Exact saved T-60 books do not cover other development dates; per-prediction ready timestamps are absent. E0036 therefore reports a sparse previously exposed diagnostic, not complete development economics.
Depends on: D0001/D0003/D0004/D0005/D0006/D0008/D0012/D0018; preserve E0024's historical status and all split boundaries. No holdout evaluation or live/shadow work.
Acceptance criteria: inventory existing local PMXT/Kacho, market/token mapping, T-60 decision snapshots, T-59 native ask-depth sources, Gamma official outcomes and saved model/procedure hashes for the full development interval; verify T-60 feature, training, transformation, selection and calibration cutoffs without fabricating historical runtime telemetry. Produce exact market/token/time coverage and exclusion manifests before additional economic computation. If source gaps remain, keep them explicit and do not reduce the interval or change nominal timestamps. Reuse existing caches/ledgers; run fixed-$5 and 5%-of-free-cash $100 resolution portfolios only for newly evidenced development coverage, with payout timing assumptions frozen before outcomes.
Evidence required: source-manifest hashes, causal cutoff audit, exact gap counts, candidate common-sample accounting, independent fixed-$5 diagnostic, both self-financing controls, fee and payout event reconciliation, locked capital, drawdown and unresolved-position reporting. Do not use holdout outcomes, tune choices, acquire paid data, or infer live readiness.
