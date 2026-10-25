# Evidence-based tasks

Recovered 2026-10-25 at `cf885b0`. These are observed gaps, not authorization to implement or run experiments. IDs are stable. DONE requires the stated acceptance evidence; a passing test subset cannot close a full-suite task. Consult [PROJECT_STATE](PROJECT_STATE.md), [DECISIONS](DECISIONS.md), and the relevant [experiment](EXPERIMENTS.md) first.

## T0000 - Recover canonical project state

Status: DONE.
Priority: P0.
Reason: conflicting historical checkpoint/status claims and no unified current snapshot/registers.
Depends on: none.
Acceptance criteria: inspect initial Git state/full available history, distinguish implementation/tests/validation and experiment outcomes, run safe existing tests, record data protection/provenance, decisions/invariants, false completion and unknowns; provide canonical navigation and a documentation-only commit.
Evidence required: [audit report](docs/PROJECT_STATE_AUDIT_20261025.md), [machine evidence](reports/project_state_recovery_20261025.json), four canonical registers and agent rules. Commit is the one introducing this task; resolve with `git log -1 --format=%H -- TASKS.md`. Publishing outcome is reported by the audit completion message.
Completion scope: state recovery completed even though tests are red and research/operational validation remains incomplete. No implementation bug was fixed.

## T0001 - Restore ESN test isolation across full discovery

Status: READY.
Priority: P1.
Reason: full discovery has eight ESN errors although architecture (7 tests) and expansion (8) each pass in isolated processes. `test_btc_candidate_holdout_20261022.py::test_gap_and_checkpoint_restart_match_uninterrupted_dynamic_states` calls `quality._configure_source_run`; that calls `focused._configure_engines`, overwriting shared expansion/architecture globals and callbacks without test restoration.
Depends on: none. Relevant: E0026-E0030, D0008/D0014.
Acceptance criteria: under unchanged candidate methods/artifacts, full discovery has no ESN state-dependent errors; affected tests pass alone and after the candidate checkpoint test; no writes to frozen local research outputs; no protected-row reads introduced.
Evidence required: exact commands, counts and timings for full suite and the order-sensitive reproduction; scoped diff explaining restored global state. Resolve the separate failures in T0002/T0003 before claiming the entire suite passes.

## T0002 - Resolve fixed-slippage test/configuration discrepancy

Status: READY.
Priority: P1.
Reason: `test_optimize_trade_policy.py::OptimizeTradePolicyTests::test_run_optimization_saves_runtime_config_with_fixed_slippage_ticks` expects 3; `optimize_trade_policy.py::SUBMITTED_PRICE_SLIPPAGE_TICKS` is 2. It fails both full and isolated runs. An audit cannot decide whether the test expectation or intended policy should change.
Depends on: none. Relevant: D0003/D0006/D0007.
Acceptance criteria: establish intended fixed-slippage semantics from policy authority/history, align implementation/test accordingly without incidental live configuration changes; preserve per-asset write isolation and unchanged search scope.
Evidence required: source of the intended value and passing focused/full checks; any methodological change needs its own recorded decision. Do not optimize new policies to resolve a constant assertion.

## T0003 - Restore mocked submit-success telemetry coverage

Status: READY.
Priority: P1.
Reason: `test_run_multi_asset_latency.py::LiveSubmitTelemetryTests::test_successful_submit_response_is_distinct_from_send_and_fill_events` constructs a trader with `__new__` and lacks `pm_user_trade_monitor`; `_maybe_submit_order` returns `submission_error` instead of `submitted_fok`. Fake-only diagnostic reproduced the missing-attribute error. This is not evidence of an actual exchange rejection.
Depends on: none. Relevant: E0007, D0007.
Acceptance criteria: identify intended initialization behavior; correct fixture or implementation in separate work; successful mocked response has call bounds/order identity and leaves unobserved transport/ACK/fill timestamps unknown; no network/order calls required.
Evidence required: passing success/error telemetry tests and full discovery; no live order submitted.

## T0004 - Verify endpoint reachability and complete no-order shadow pilot

Status: READY (collector E0034 remains BLOCKED pending evidence).
Priority: P1.
Reason: saved October 23 Gamma/CLOB DNS failures prevented runtime initialization; current database has zero live windows. Current network reachability is UNKNOWN because audit did not retry or start collector.
Depends on: T0007 for readiness semantics; preserve D0008/D0010 frozen identities; perform independent mock tests after T0001 if implementation changes are needed.
Acceptance criteria: diagnose public endpoint access without credentials/orders; under separately continued collection scope capture at least three complete scheduled windows with exact Gamma market/token mapping, timely actual prediction timestamps, both T60/T59 snapshots and later official outcomes; report all failed/late attempts, measured latency and storage growth. Leave actual fills UNKNOWN. Preserve protocol and no-order behavior.
Evidence required: preflight responses/timestamps, read-only DB counts/sample identities, runner/model/config hashes, per-stage runtime timing, restart/checkpoint evidence, and explicit collector process state/stop procedure. A synthetic test or successful Binance-only probe cannot close this task.

## T0005 - Establish exact historical TWAP and L2 source feasibility

Status: BLOCKED.
Priority: P2.
Reason: 4,193 saved label disagreements lack official price-print reconstruction; exact TWAP30/TWAP60 historical feed IDs/access/retention/cost are unconfirmed. Local PMXT ends before TWAP60; current historical candidate execution ledger has no TWAP60 rows.
Depends on: explicit source/access scope and, if acquisition is needed, authorization for any account, credentials, fee or large download; none granted by this audit. Independent of T0004 and unrelated research.
Acceptance criteria: document exact official feed identities and obtainable historical periods for the frozen 43-market sample; separately document two-token L2 coverage and initialized event continuity for the target historical period; obtain concrete access/cost/size information before acquisition. If unavailable, preserve UNKNOWN price cause and absent economics. Do not tune models or change thresholds to fill a data gap.
Evidence required: source metadata, sample/manifest checksums, timestamps/coverage and quote orientation, written cost/entitlement evidence, then separately authorized reconstruction checks. Provider marketing or one out-of-period Vike sample is insufficient for target-period validation.

## T0006 - Reconcile legacy entrypoints and reproducibility prerequisites

Status: READY.
Priority: P2.
Reason: README references missing `optimize_lgbm_optuna.py`/`.python-version`; current optimizer name differs. Deployment refit config points to original `preopen_research_20261011/evaluated_holdout_manifest.json`, absent locally, whereas completed retry/correction records are elsewhere. Most data/models/caches are ignored and requirements are unpinned.
Depends on: audit registers and D0001/D0007; no refit execution authorized.
Acceptance criteria: document correct commands/interpreter/dependency prerequisites and intended refit lineage without substituting bundles silently; identify all required local artifacts and how to restore them within data/authorization constraints. Distinguish a read-only reproducibility check from running training or refit.
Evidence required: verified links/entrypoints, manifest compatibility and model/config identity review, documented missing assets; clean-clone/cross-platform status remains NOT_VERIFIED_IN_THIS_AUDIT until actually tested. Do not close by installing a different GPU stack or replacing the user's environment.

## T0007 - Reconcile shadow readiness deadline with the one-second contract

Status: READY.
Priority: P1.
Reason: `run_btc_candidate_shadow_20261023.py::market_timing` defines readiness deadline T-60; `_predict_market` applies it to actual completion after receipt of a candle closing nominally at T-60. `features/btc_preopen_contract.py::build_preopen_contract_frame` defines prediction deadline T-59. The current shadow test explicitly rejects T-60+250 ms. Positive delay therefore appears incompatible with timely classification under shadow despite the generic one-second budget.
Depends on: D0003/D0010 and frozen protocol review; precedes T0004 pilot-validation claims.
Acceptance criteria: establish and record the intended difference (or required correction) between nominal decision time, data cutoff, actual prediction readiness and execution time. Preserve causality and actual timestamps; do not simply backdate or loosen a gate to obtain passing windows. If behavior changes, use explicit decision/protocol identity and separately authorized implementation work.
Evidence required: source-of-authority comparison, deterministic boundary cases and then actual no-order receipt/compute timing; report whether the intended path can produce eligible decisions. A unit assertion encoding the current strict deadline is not reference validation of that deadline.
