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

## Historical continuation plan - 2026-10-26

Scope: continue the historical BTC evaluation under D0017/D0018. Preserve frozen candidates, official Gamma outcomes, fee rules and split boundaries. No new holdout evaluation, model scoring/training, tuning, full replay, external data acquisition, collector, DNS pilot check, shadow/live process, or order.

1. E0036 reuses the hash-identified E0031 confirmation ledger and calculates economics only for its 1,143-market common T-60/T-59 subset. Its positive B/C/D points are sparse, previously exposed diagnostics; they do not close full-development validation.
2. E0039 / T0009 chronology axis: the 2026-10-27 attached instruction later authorized the specified earlier-only F1/F2/F3 work. Current status and the exact pre-fit resource gate are recorded below. Keep source coverage and runtime as separate T0009 axes; do not use holdout, acquire sources, or run live/shadow.
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

Status: BLOCKED overall (source coverage and prediction-readiness provenance remain incomplete; E0039 v6 completed F1 feature generation but stopped before predictive fitting, v7 is validated and awaits commit/push, and the complete runtime hot path is BLOCKED).
Priority: P0.
Reason: E0038 separates source-value parity from checker coverage and retains the unresolved E0036 T-60 book/BBO conflict in quarantine. Of 5,684 label/prediction/source-intersection markets, 5,604 pass a symmetric source-only quality rule; source and prediction coverage remain sparse and retrospective, 53,660 candidate calendar rows lack Gamma IDs, the 24 missing local PMXT hours are tracked separately under T0010, and measured prediction-ready timestamps are absent. Conditional economics on 5,604 markets cannot establish complete development economics.
Depends on: D0001/D0003/D0004/D0005/D0006/D0008/D0012/D0018/D0019; preserve E0024's historical status and all split boundaries. No holdout evaluation or live/shadow work.
Acceptance criteria: inventory existing local PMXT/Kacho, market/token mapping, T-60 decision snapshots, T-59 native ask-depth sources, Gamma official outcomes and saved model/procedure hashes for the full development interval; verify T-60 feature, training, transformation, selection and calibration cutoffs without fabricating historical runtime telemetry. Produce exact market/token/time coverage and exclusion manifests before additional economic computation. If source gaps remain, keep them explicit and do not reduce the interval or change nominal timestamps. Reuse existing caches/ledgers; run fixed-$5 and 5%-of-free-cash $100 resolution portfolios only for newly evidenced development coverage, with payout timing assumptions frozen before outcomes.
Evidence required: source-manifest hashes, causal cutoff audit, exact gap counts, candidate common-sample accounting, independent fixed-$5 diagnostic, both self-financing controls, fee and payout event reconciliation, locked capital, drawdown and unresolved-position reporting. Do not use holdout outcomes, tune choices, acquire paid data, or infer live readiness.

2026-10-26 scope under the prior user instruction authorized E0038 source-quality continuation and conditional economics, with no new training/tuning; that history remains unchanged. The user's 2026-10-27 attached instruction supersedes that restriction for the three specified earlier-only development folds and offline benchmark. It still forbids holdout use, acquisition, T0010 work, sale policy, DNS, live/shadow, and orders. E0038 v1/v2/v3 and E0037's saved FAILED gate remain unchanged.

2026-10-26 result: E0037 replayed the full local development interval and wrote a 105,118-row manifest (51,458 Gamma rows and 53,660 unmapped candidate rows). The labeled, prediction-present, two-sided T-60/T-59 source cohort is 5,684, with zero measured prediction-ready timestamps. T-59 E0036 parity passed; T-60 parity failed on one BBO ask-mismatch count, so no new economics or payout-delay account paths were run. The T−60 sample's archived raw stream leaves an UP 0.51 ask level while later processed BBO fields report 0.52; the local data do not resolve the cause. D0019 clarifies that share-labeled minimum fields cannot substitute for the D0004 USDC minimum; the $5 fallback remains. The full test suite passed 696/696. No holdout, acquisition, outcomes/PnL, training/tuning, sale-policy, DNS, live/shadow, or order work occurred.

E0038 source-quality freeze checkpoint: value parity is exact across 19,431 T−60 and 26,289 T−59 comparisons. Old zero-check rows are `NOT_CHECKED`; the new checker finds the traced BBO conflict. A source-only symmetric rule quarantines 80 of 5,684 markets (62 ambiguous ask ladders, 16 T−59 ask/BBO mismatches, 2 uncomparable BBOs), leaving 5,604 and 1,141 E0036 common markets. The instructed T−60 market was already outside the 5,684 base. The independent local replay agrees with the cached $0.51 ask while the later reported BBO is $0.52; cause remains unresolved. E0038 is frozen before economics; no new outcomes/PnL have been read at this checkpoint. Full-development coverage and causal/readiness provenance remain blocked.

E0038 result: the corrected v3 replay ran only after its protocol was committed and pushed. On the 5,604 clean markets, independent fixed-$5 net PnL was A +$74.46, B +$221.83, C +$273.42, and D +$285.90. For the $100 fixed-$5 accounts, at the 60-second payout scenario net PnL was A -$95.46, B +$221.83, C +$273.42, and D +$285.90; at 7,200 seconds it was A -$99.73, B +$223.26, C +$270.21, and D +$291.24. The 5%-of-free-cash control made only 1-2 fills per model because requests below the frozen $5 minimum were rejected without rounding; its net PnL ranged from -$5.00 to -$0.76. All four payout scenarios and complete month paths are in the [E0038 report](reports/btc_preopen/btc_t0009_source_quality_continuation_20261026.md); detailed local ledgers remain ignored under data/analysis. Corrected E0036 common-sample comparison has zero row-level differences across all fields for A/B/C/D on 1,141 markets. No holdout rows or columns were read. These are retrospective/conditional results; no portfolio bootstrap or independent predictive confirmation is claimed. T0009 remains BLOCKED for full-development source coverage and readiness/provenance.

2026-10-27 E0039 initial freeze: the E0038 stability/concentration and paired B-A/D-B diagnostics are in [the chronology report](reports/btc_preopen/btc_chronology_stability_reconstruction_20261027.md); all monthly flows and account snapshots are linked there. The new fit mask uses label-availability timestamps and has exact-cutoff/future-row regression tests. The three outer windows, raw/flow prefix hashes, E0038 allowlist and 300-window runtime sample were frozen in [E0039 v1](configs/research/btc_chronology_reconstruction_20261027.json). At that initial freeze, the new runner was not yet implemented; focused checks passed 51/51 and full discovery passed 711/711 in 41.334 seconds.

2026-10-27 E0039 v2 execution: the identity-checked fold-local runner and amended protocol were committed and pushed before invocation. See the [v2 report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v2.md), [v2 protocol](configs/research/btc_chronology_reconstruction_20261027_v2.json), and versioned [pre-fit resource gate](reports/btc_preopen/btc_chronology_reconstruction_20261027_pre_fit_gate.json). The exact command returned `BLOCKED_BEFORE_FIT`: available RAM was 3.79 GiB against 20 GiB required, with five other Python processes using 9.53-10.06 GiB each. Free disk and GPU gates passed. The gate records `data_or_label_reads_for_fit: false` and zero holdout rows. Development-only source-row projection hashes, folds, OOF calibration, economics and runtime measurement therefore remain unstarted. Focused v2 tests passed 17/17; full discovery passed 728/728 in 54.381 seconds; `py_compile` passed.

2026-10-27 E0039 v2 retry and v3 protocol correction: the live gate passed (50.15 GiB available RAM, 105.91 GiB free disk, 6,206 MiB free GPU memory, no other Python processes using at least 1 GiB), and 53 frozen files verified. The command then stopped before fitting at `E0038 clean ID allowlist hash changed`. Investigation confirmed the v2 expected value `b92ed75e...` was the SHA-256 of the whole local cohort CSV, while the runner correctly hashes sorted lowercase condition IDs. The 5,604 IDs selected from the frozen quality manifest equal the CSV ID set; the canonical ID hash is `e2946aa8e45c2ed050347e18756408a0175fa0eb9859a120f8df7c173051edda`. No fit labels, Gamma outcomes, economic source rows, or holdout rows were read; no fold, OOF calibration, outer prediction, or economic replay ran. Versioned [E0039 v3](configs/research/btc_chronology_reconstruction_20261027_v3.json) corrects only hash semantics and parent/code identity; v1/v2 remain unchanged. Focused E0039 tests passed 18/18, full discovery 729/729 in 48.090 seconds, and `py_compile` passed. See the [v3 correction report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v3.md).

2026-10-27 E0039 v3 retry: resource and identity checks passed. The runner computed the four frozen source projection hashes and wrote them to the ignored v3 run manifest, then failed before F1 with PyArrow's `Expected a list or a dict for 'columns', got <class 'tuple'> instead.` The run recorded zero folds, `confirmation_rows_materialized=0` (no confirmation evaluation output), and zero holdout rows materialized. The official Gamma source projection was read only for its frozen source hash; no confirmation score or choice was made. The observed hashes are now pinned in [E0039 v4](configs/research/btc_chronology_reconstruction_20261027_v4.json); the v4 runner normalizes the same ordered tuple to a list and recomputes/compares those hashes before fitting. No fold or economic result was produced. See the [v3 execution report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v3.md) and [v4 protocol report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v4.md).

2026-10-27 E0039 v4 retry: the gate passed (50.08 GiB available RAM, 105.90 GiB free disk, 6,414 MiB free VRAM) and the four pinned source projection hashes matched. The runner loaded the permitted official Gamma and E0038 economic snapshots, then failed before selection with `TypeError: cannot use 'dict' as a set element (unhashable type: 'dict')` in `_start_runtime_ledger`. The manifest records zero folds, zero confirmation evaluation output, and zero holdout rows materialized. No fit, confirmation score, OOF result, calibration, outer prediction, or economic result was produced. The failure is preserved in the [v4 report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v4.md).

V5 changes only the malformed empty-ledger initialization and pins the same four source hashes. Its runtime-ledger regression passed 20/20 focused tests; full discovery passed 731/731 in 66.208 seconds, and `py_compile` passed. See [E0039 v5](configs/research/btc_chronology_reconstruction_20261027_v5.json) and the [v5 report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v5.md).

2026-10-27 E0039 v5 execution: the live gate and four pinned source hashes passed. F1 indicator selection completed in 289.8 seconds, reaction profile completed 96 trials (84 complete, 12 pruned), and volume profile completed 96/96. Final feature dataset construction failed because profile-state path resolution could not derive symbol/interval from `fold_20260415_20260501.csv`. The wrapper recorded a minimum available system memory sample of 12.346 GiB during F1; the gate had passed before fitting. No final fold dataset, predictive LightGBM fit, OOF/calibration result, outer prediction, economic result, or holdout rows were produced. See the [v5 execution report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v5.md).

V6 passes the configured dataset `symbol`/`interval` to the VP/RP state resolvers when temporal-prefix filenames do not encode them; existing source-name parsing remains unchanged. It keeps all data, source hashes, folds, labels, budgets, fit/selection/calibration procedures, and accounting semantics. Focused path tests passed 23/23; chronology tests passed 20/20; full discovery passed 733/733 in 40.668 seconds; `py_compile` passed. Read-only identity verification passed for the v5 parent and 53 pinned implementation/source/config paths. See [E0039 v6](configs/research/btc_chronology_reconstruction_20261027_v6.json) and the [v5 execution report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v5.md).

2026-10-27 E0039 v6 execution: the exact runner passed its resource gate (51.01 GiB available RAM, 105.90 GiB free disk, 6,375 MiB free VRAM), verified the 5,604 clean-ID identity and four frozen source projection hashes, completed F1 indicators (271.7 seconds), reaction profile (96 trials: 84 complete, 12 pruned), volume profile (96/96), and the 3,098,307-row final dataset (190.7 seconds). It failed in `causal_candidate_feature_join` because the base Parquet `Opened` values are strings and were cast directly to `int64`. The run manifest records zero completed folds and zero holdout rows. No predictive fit, OOF/calibration result, outer prediction, or economic replay was produced. See the [v6 execution report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v6.md) and ignored [run manifest](data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027/runs/eb07a734607052ac/run_manifest.json).

V7 parses the canonical UTC-naive candle string form to nanosecond timestamps while retaining typed Arrow timestamp support and exact join/gap behavior. The regression reproduced the original Arrow error before the fix and passed 1/1 after it. Chronology tests passed 20/20, project-config tests 23/23, full discovery passed 733/733 in 48.443 seconds, and `py_compile` passed. Read-only identity verification passed for the v6 parent, all 53 pinned files, the three fold boundaries, and four frozen source-hash entries. See [E0039 v7](configs/research/btc_chronology_reconstruction_20261027_v7.json) and the [v7 protocol report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v7.md).

Chronology axis: commit and push v7 before the next fit. The exact runner must repeat the resource gate, verify the 5,604 clean IDs and four pinned source projection hashes, then continue F1 and complete F2/F3 and the shared economic ledger with the unchanged research recipe. Do not reduce the workload or alter the protocol.

Runtime axis: offline streaming adapter remains `PARTIAL_NOT_IMPLEMENTED`; the full hot path has no latency percentiles or one-second conclusion. The 300-window sample remains frozen. This runtime status is separate from the batch resource gate.

Source axis: unchanged and BLOCKED overall. E0038 remains a 5,604-market conditional subset; 53,660 candidate calendar rows lack Gamma IDs and 24 local PMXT hours remain deferred under T0010. No source acquisition was performed. Missing historical prediction-ready timestamps are not the chronology blocker; E0039 does not backdate them.

## T0010 - Reconcile 24 missing local PMXT development hours

Status: DEFERRED (inventory preserved; external acquisition is outside the authorized scope).
Priority: P1.
Reason: the T0009 coverage inventory records 1,877 advertised PMXT development hours, 1,853 locally processed hours, and 24 missing local hours: 21 public-indexed hours from 2026-04-13 19:00 through 2026-04-14 15:00 UTC and three hours on 2026-06-11 04:00–06:00 UTC. This is a source-coverage follow-up, separate from E0038's 80 market-level quarantines and not an additional subtraction from its 5,684-market base.
Depends on: T0009 and its source-coverage evidence. Do not download or acquire data without a separately authorized scope.
Acceptance criteria: retain the exact missing-hour inventory from the [T0009 coverage report](reports/btc_preopen/btc_t0009_resolution_coverage_20261026.md) and [machine summary](reports/btc_preopen/btc_t0009_resolution_coverage_20261026.json); document whether existing local sources can resolve any entries and the exact affected coverage. Do not treat obtaining these hours as restoring months before the archive begins or as closing T0009 by itself.
Evidence required: source manifest and exact UTC hour list, source availability/provenance, affected-market coverage, and an explicit no-acquisition or separately authorized acquisition record.
2026-10-26: carried forward from the user's T0009 instruction. No acquisition was performed in E0038.
