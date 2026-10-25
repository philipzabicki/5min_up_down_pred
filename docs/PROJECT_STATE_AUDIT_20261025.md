PROJECT STATE RECOVERY: COMPLETE

Recovery date: 2026-10-25, Europe/Warsaw. All experiment/data boundaries are UTC. Scope: recover repository state, not certify model quality or production readiness. COMPLETE means the state and its uncertainties are recorded; the test suite is **failing**, research gates remain blocked, and independent/live validation is incomplete.

## 1. Git / repository state

Initial state, before any audit edit:

| Item | Observed value |
|---|---|
| Repository | `D:\Cloud\filips19\github\5min_up_down_pred` |
| Branch | `audit/btc-dynamic-history-20261020` |
| HEAD | `cf885b0428e0c6041d471e6bca97597ff04dc4ec` |
| Upstream | `origin/audit/btc-dynamic-history-20261020`, same SHA |
| Remote | `https://github.com/philipzabicki/5min_up_down_pred.git` |
| Relative state | ahead 0 / behind 0; additionally confirmed by `git ls-remote --heads --tags origin` |
| Working tree / staged changes | clean; modified paths: none; staged paths: none; untracked paths reported by Git: none |
| Local unpushed commits | none at start |
| Other local/remote branch | `main` / `origin/main` at `3d95f641288f57e5f7d4c9ed77c621b5f678daaf` |
| History inspected | all 262 available commits, from `c07d854` (2026-03-19) to audited HEAD (2026-10-23), plus milestone file stats/code and current artifacts |
| Tags / releases | no local/remote tags; GitHub releases GET returned `[]` on audit date |
| Explicit Revert commits | none found by anchored `^Revert` search; semantic replacement exists and is listed below |
| Other experimental branches | no additional refs in available local/remote branch inventory; deleted/unavailable refs UNKNOWN |

Ignored files are **not** the same as untracked files reported by ordinary `git status`. There is substantial ignored local data, models, databases, caches and logs; `.env` exists and its contents were not read. Nothing pre-existing was removed, reset, stashed, staged or attributed to this audit. Existing Python workloads were left alone. The documentation-only audit commit is identifiable with `git log -1 --format=%H -- PROJECT_STATE.md`; its own hash cannot be embedded in that same commit. The completion message records exact commit/push/final working-tree status.

### Historical reconstruction

Commit subjects served as navigation only; code, file changes, configurations and available result files support the reconstructed scope.

| Stage | Intended / actually present | Evidence | Current status |
|---|---|---|---|
| March-May original pipeline | Indicators/profiles, feature selection, LightGBM and live trading; actual source modules plus later replacements | `c07d854` adds original 42-file pipeline; current `features/`, acquisition/training/runtime code; `1469985` removes older policies | LEGACY procedures; core capabilities remain implemented, original performance not revalidated |
| June-July multiasset/config migration | Migrate utilities and add per-asset runtime; source moves, multiasset launcher and asset maps exist | `ac2c531` moves 8 utilities without content changes; `cad7cf7`, `90fef84`, `d0d22cc`; `utils/project_config.py`, runtime manifest | ACTIVE infrastructure, ETH/SOL trading quality UNKNOWN |
| October 2 pipeline/profile controls | Add VP/RP fit steps and model switches | `1720603` actual five-file diff; `run_deployment_pipeline.py` and tests | IMPLEMENTED; does not prove completed deployment |
| October 3-4 market-open and causal pre-open split | Audit newer BTC, collect state, create a separate causal prestart target | E0001-E0004; target module and versioned protocols; `bb72dc0`, `571019c` | Old market-open path LEGACY; pre-open code ACTIVE |
| October 6-10 execution/fees/policy | Reconstruct books, compare sizing/exits and assess operational readiness | E0005-E0007; current fee/cashflow code, saved reports; dated contract | Several old valuations SUPERSEDED; operational readiness NO-GO historically, no newer positive live proof |
| October 11-14 causal research / model update | Complete walk-forward pipeline, fix fees/minimum units, compare models and plan prospective test | E0008-E0010; completed retry/correction records; B trial shortfall; prospective deferral | Negative corrected selected-policy result; model B partial; prospective PLANNED |
| October 15-16 ablation -> one-pass fit | Initially full family comparison, explicitly changed scope to one full-development fit | E0011-E0013; interrupted manifests; `0f96f5b`; successful freeze/run | Original experiments SUPERSEDED; single fit completed with conditional OOF and negative selected-policy PnL |
| October 17-19 target/model/market variants | Returns, offsets, shrinkage, flow, RAW BASE70, seeds/recency/capacity and microstructure | E0014-E0023; separate code/config/manifest/result sets | Mixed negative/inconclusive/conditional positive results; no general promotion |
| October 20 dynamic-history gate | Construct S/H and action values after data sufficiency/provenance | E0024; actual failed gate and RAW availability audit; `25a013b`, `80020c1`, `11927f3` | BLOCKED before fitting; counts corrected, thresholds unchanged |
| October 20-22 ESN evolution | Portfolio -> architecture audit -> expanded -> focused -> memory control -> frozen A/B/C/D | E0025-E0029; source/seed/config identities and completed summaries | Development search completed; best probe portfolio not downstream winner; no independent ESN advantage established |
| October 23 frozen scoring/targets/regimes/shadow | Score approved exposed periods, assess Gamma targets and diagnose limited economics; begin public shadow pilot | E0030-E0034; current scorer, approval true, completed period records; zero-row live DB | Historical scoring completed/inconclusive; source-price reconstruction and actual pilot BLOCKED |

## 2. Executive state summary

The codebase is substantial research and live automation software, with a distinct BTC pre-open research path and older BTC/ETH/SOL trading paths. It is not merely a planned project. Acquisition, features, tuning, OOF, calibration, replay, portfolio accounting, bundle freezing and read-only collectors have implementations and tests. Actual runtime activation still refers to older BTC `20261003_043549`, ETH `20260807_181931` and SOL `20260807_201205` metadata, not A/B/C/D. All three runtime assets are enabled; their selected live profiles have paper mode false and submission disabling false. These are inspected configuration values, **not evidence that a live process or real trading is currently operating**. This audit never ran `run.py` or changed those settings.

What is established now: test execution outcomes; present metadata/artifact identities; explicit gate failures; no candidate shadow windows in the inspected DB; code/protocol and documentation conflicts. What is historical recorded evidence: parity results, prediction scores, bootstrap intervals, source checks and economic ledgers. They were inspected as saved outputs, not recalculated. What remains unestablished: general independent OOS advantage, current TWAP60 profitability, full end-to-end deadline compliance, actual ACK/fill/fee behavior and production readiness.

Canonical snapshot and registers: [PROJECT_STATE](../PROJECT_STATE.md), [DECISIONS](../DECISIONS.md), [EXPERIMENTS](../EXPERIMENTS.md), [TASKS](../TASKS.md). The existing contract stays binding and existing BTC status remains the checkpoint log. Audit evidence: [JSON inventory](../reports/project_state_recovery_20261025.json).

## 3. Verified components

Implementation status is about concrete code. Test status is from the current run, with named failing exceptions. Validation is independent of both. `UNKNOWN` means the required kind of evidence was not found/established; it is not a PASS. A passing synthetic test is never called exchange reference validation.

| Component | Implemented | Tested in audit | Validated | Evidence | Status |
|---|---|---|---|---|---|
| Asset/config loading and pipeline orchestration | IMPLEMENTED | Config and pipeline assertions pass | Full deployment UNKNOWN | `utils/project_config.py::load_enabled_runtime_asset_settings`; `run_deployment_pipeline.py::run_pipeline`; `test_project_config`, `test_run_deployment_pipeline`; `1720603` | A for tested contracts, B for deployment |
| Binance/Chainlink acquisition and raw repair | IMPLEMENTED | Some parsing/source fixtures; no live acquisition | Current endpoint/data completeness UNKNOWN | `data/binance_sources.py`, `data/chainlink_sources.py`, `data/raw_ohlcv_repair.py`, `fetch_data.py`; taker-flow source tests | B |
| General candle/indicator/VP/RP/basis features | IMPLEMENTED | Feature assertions pass | Full offline/live reference parity UNKNOWN for all configs/assets | `create_modeling_dataset.py::build_dataset_from_settings`; `features/`; candle/basis/reaction/parity tests | A narrowly; B globally |
| Pre-open target, labels and boundary guards | IMPLEMENTED | Contract/development-boundary/target tests pass | Saved historical proxy reconstruction matches; not independently rerun | `features/btc_preopen_contract.py::build_preopen_contract_frame`, `purge_unavailable_training_rows`; E0032 validation JSON | A for fixture semantics, PARTIALLY VERIFIED historically |
| General LightGBM/CV/feature and weight selection | IMPLEMENTED_WITH_KNOWN_DEVIATIONS | CV objective/selector tests pass | Independent full-pipeline OOS UNKNOWN | `train_lgbm.py::evaluate_walk_forward_variant` uses validation fold for early stopping and scores same fold; `select_features.py`; E0001/E0013 | B; legacy OOF is conditional |
| Nested research / causal calibration/policy | IMPLEMENTED | Walk-forward/research tests pass | Recorded completed retry/corrected results; not rerun | `run_btc_preopen_walk_forward.py::_write_calibration_pool`, `run_btc_preopen_research.py::_select_development_policy`; E0008 | B, historical economic result negative |
| One-pass full-development fitting | IMPLEMENTED_WITH_KNOWN_DEVIATIONS | Its frozen-config/prediction tests pass | Conditional CV, not independent nested selection | `run_btc_preopen_full_development_fit.py::_validate_frozen_config`, `_run_or_resume`; E0013; D0005 | B |
| Book replay/T60-T59 execution | IMPLEMENTED_WITH_KNOWN_DEVIATIONS | Replay/continuity/checkpoint fixtures pass | Reference complete event history/real fills UNKNOWN | `run_btc_preopen_t59_reassessment.py`; `features/btc_market_dynamics_action_value.py`; E0005/E0024 | B, source restrictions remain |
| Fees/minimum/cashflow/accounting | IMPLEMENTED | Fee/cashflow/trade/portfolio fixture assertions pass | Saved accounting passes on named ledgers; actual charged fees UNKNOWN | `utils/polymarket_btc5m_fee_rules.py::resolve_market_fee_rule`, `fee_for_price_level`; E0008/E0031/E0032 | A for tested arithmetic; B for exchange equivalence |
| Optimizer fixed-slippage export | IMPLEMENTED_WITH_KNOWN_DEVIATIONS | FAIL persists alone | Intended fixed value UNKNOWN | `optimize_trade_policy.py::SUBMITTED_PRICE_SLIPPAGE_TICKS`; `test_optimize_trade_policy...fixed_slippage_ticks` | Known discrepancy T0002 |
| Dynamic S/H and action-value experiment | PARTIAL | Mechanics/provenance assertions pass | Gate FAIL, no model fitted/scored | `run_btc_market_dynamics_action_value_20261020.py::_availability_report`, `_fit_fold` behind gate; E0024 | D |
| Memory/ESN engines and frozen projection | IMPLEMENTED | Engine/memory/checkpoint assertions pass | Saved development parity only; full rerun not done | `features/esn_reservoir.py::ESNEngine`, `FeatureScaler`, `FrozenProjection`; `features/causal_memory_control.py::CausalMemoryState`; E0029 | A for tested mechanics, B for independent quality |
| ESN search runner/test composition | IMPLEMENTED_WITH_KNOWN_DEVIATIONS | Eight full-suite ERROR; isolated architecture/expansion PASS | Search summaries exist; independence UNKNOWN | `focused._configure_engines`, `quality._configure_source_run`; T0001 | Known shared-global isolation problem |
| A/B/C/D frozen bundle/scorer | IMPLEMENTED | Scorer mechanics pass; checkpoint test pollutes other modules | Completed historical exposed scores only | `run_btc_candidate_holdout_20261022.py::validate_scope`, `predict_members`, `save_period_checkpoint`; E0029/E0030 | B; statistical superiority unresolved |
| Official-target/regime attribution | IMPLEMENTED | Saved manifest inspection; not a new full analysis run | Exact-price reference UNKNOWN | `run_btc_candidate_attribution_20261023.py`; `extend_btc_candidate_attribution_regimes_20261023.py`; E0031-E0033 | PARTIALLY VERIFIED output provenance |
| Candidate no-order shadow | IMPLEMENTED_WITH_KNOWN_DEVIATIONS | Ten shadow mechanics/static assertions pass | Pilot 0/3, real timing UNKNOWN; cutoff conflict | `run_btc_candidate_shadow_20261023.py::market_timing`, `_predict_market`, `_collect_market`; E0034; T0007 | C/D |
| Older prospective collector / evaluation | IMPLEMENTED | Prospective fixture assertions pass | Prospective study not started | `run_btc_prospective_shadow.py`, `analyze_btc_prospective_results_20261014.py`; E0010 | B, experiment PLANNED/deferred |
| Live multiasset orders/user stream/redemption | IMPLEMENTED_WITH_KNOWN_DEVIATIONS | One mocked submit-success FAIL; other relevant assertions pass | Actual exchange/V2/fill/latency UNKNOWN | `run.py::LivePredictor`, `PolymarketLiveTrader`, `_run_enabled_runtime_assets`; `utils/polymarket_user_stream.py`; redemption tests | B; T0003, historical NO-GO |
| Separate deployment refit | IMPLEMENTED; configured workflow PARTIAL | Research guard assertions only; no full refit | No completed configured output found | `run_btc_preopen_deployment_refit.py::run_deployment_refit`; missing original holdout prerequisite; D0007 | C/T0006 |
| Fully independent full-stack OOS validation / production qualification | NOT_IMPLEMENTED as a completed evidenced gate | NOT_VERIFIED_IN_THIS_AUDIT | UNKNOWN | Existing conditional/exposed results cannot satisfy this claim | E |

### Validation ledger

| Element | Reference / data | Code identity and result | Does it still apply? |
|---|---|---|---|
| Timestamp/target mechanics | Synthetic boundary/missing/tie/purge fixtures | Current audited HEAD; assertions pass | YES, only tested semantics, not actual Binance receive timing |
| BASE70 raw-to-frozen prediction replay | Saved development raw stream ending before July 2025 | [development replay JSON](../reports/btc_preopen/btc_candidate_holdout_20261022_development_replay.json): exact BASE70 finite masks/features; model tolerance <=saved threshold | PARTIALLY VERIFIED: file inspected, full replay not repeated; applies only to frozen sources/models |
| Memory/ESN representation, resume and future prefix | 2,660,546 saved development raw rows; each representation's valid mask | E0029 local stream verification; max errors 0; fresh process true | PARTIALLY VERIFIED: local JSON hashed; current mechanics tests pass, large reference run not repeated |
| Historical frozen prediction robustness | Exposed 105,117/28,223 decisions | E0030 score `cf572c4`, finalize `262b7d0`; all adjusted primary intervals include zero | Applicable as inconclusive historical evidence only; no fresh sample |
| Official outcome mapping | Saved Gamma series 10684 identities, one-hot labels and Binance windows | E0032 saved validation 133,340/79,650 checks; no detected join/target error | Partial evidence, no exact Chainlink price reconstruction |
| Economic accounting | Saved 1,143 common confirmation markets | E0031/E0032; identity/decomposition checks true | Only sparse single-price retrospective assumptions, not TWAP60/fills |
| Full-depth replay vs exchange reference | Archived initialized book plus updates; reference gaps/ambiguity | Synthetic checks and independent development audit; failed dynamic source gate | No comprehensive reference PASS |
| Live no-order pipeline | Real endpoints, actual ready/book/outcome observations | E0034: two failed preflights, zero windows | No validation; current network not retried |
| Cross-platform / real production | Another OS or real observed order lifecycle | No such run in this audit | UNKNOWN / NOT_VERIFIED_IN_THIS_AUDIT |

### Data and provenance audit

No protected raw CSV/Parquet rows or protected prediction Parquets were opened, hashed, scored or recalculated by this audit. Existing reports, prior-use manifests, summary JSON and file metadata were inspected. Both protected periods are explicitly recorded as already used under a previous one-time protocol; reading their published results does not restore their independence. Existing tests use synthetic/temp data and, for one checkpoint test, local development-only frozen cache/bundles. No new research training/tuning, source acquisition, full replay or long collector was launched.

| Dataset/source | Identity, period and version evidence | Observed limits / reproducibility |
|---|---|---|
| COIN-M BTCUSD index and USD-M BTCUSDT OHLCV | `configs/datasets.json`; raw BTC CSVs exist; cutoff file `BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007.csv` is 324,192,033 bytes | Metadata only; actual row coverage/checksums not rerun. Non-cutoff files may grow. Exact run manifests are authoritative, filename alone is not |
| Target-specific feature matrix / BASE64/70 | Frozen full-development pipeline `794a69e369a3f82d...`; 64 selected features plus six Z/V; source/fold/config hashes in experiment manifests | Learned choices used development. Reusing matrix for earlier validation is conditional. Missing configured development CSV is not repaired here |
| ESN development cache | `btc_esn_portfolio_20261020_r3/prepared/search/prepared_manifest.json` exists; config fit core [2020-06-09 09:33,2024-07-01), correction [2024-07-01,2025-01-01), search [2025-01-01,2025-07-01) UTC | These intervals have distinct purposes; search reused repeatedly. Source of BASE70 selection itself used later development, so full-pipeline independence does not follow from fit-core-only PCA/models |
| Frozen candidate confirmation | [2025-07-01,2026-07-01); expected 105,120, common 105,117; two invalid BASE70 and one late-to-end label exclusion in saved report | Already used by E0025 and other comparisons; exposed, not fresh confirmation |
| Frozen candidate holdout | [2026-07-01,2026-10-07); common 28,223; saved pre-read/approval/attempts/period artifacts | Already historically exposed. Old RAW holdout uses 28,224, different exact decision/end mask; these counts must not be equated |
| Taker-flow raw completion | E0030: 98 daily API payload partitions, 141,113 minutes + 11,550 causal warm-up minutes; exact per-partition hashes in source-completion manifest | Manifest independently found/hashed; payload completeness claims not recalculated. Separate E0021 repaired five October archive partitions |
| Gamma official outcomes | E0031/E0032: 79,678 matches, 79,650 valid labels; 28 exclusions: 14 unresolved/missing resolution, 13 ambiguous outcomes/prices, 1 nonbinary/ambiguous prices | Final one-hot outcomes/metadata are not historical price reports. Source payloads can change; preserve captured identities/hashes |
| Settlement-rule assignment | Seven saved inspected pages and deterministic 43-market sample; August 7/14 UTC boundaries | 79,643 remaining individual rules date-assigned, not directly confirmed. UNKNOWN per-market truth is not erased by zero conflicts under supplied calendar |
| Kacho / Obadiaha older snapshots | `docs/polymarket_history_contract.md`, `docs/polymarket_btc_experiment.md`; captured provider manifests/checksums | Kacho outcome requires official resolution; top sizes are shares; Obadiaha full-book-best confirmation insufficient for old market-value sample. Historical timing is not pre-open coverage |
| PMXT v2 / supplemental native or proxy snapshots | Dated T59/data manifests and dynamic-history audit | Source holes, ambiguous event order, missing initializing snapshots and RAW provenance cause genuine gates. Empty partition != no event or unchanged book. Local PMXT end precedes TWAP60; raw archive was not rescanned |
| A/B/C/D execution intersection | Saved common native T60/T59 size-qualified 1,143 markets | All confirmation/single-price; zero holdout/TWAP30/TWAP60. T60 prediction actual readiness absent; no full ladders in T60 source. This is not a representative full market calendar |
| Official Chainlink reports | Existing short nonoverlapping 1,426-report cache described in source audit | Exact target TWAP feeds/history/entitlement/cost unavailable; source claim not refreshed, no API access performed |
| Live/shadow | Current read-only candidate DB contains no market observations | No new live dataset. Earlier collectors' start manifests do not establish continuing collection. Read-only SQLite table counts recorded in audit JSON |

TRAIN / VALIDATION / CONFIRMATION / HOLDOUT / OOS / LIVE are different roles. The original raw baseline used pre-January-2026 fit, later calibration and April-May historical external evaluation; it must not be conflated with the later July split or ESN fit-core split. OOS claims must state exactly which learned components were fit earlier; conditional chronological model predictions are not independent OOS of the complete selection process.

Leakage risks found: general `train_lgbm.py` scores the fold used for early stopping; selected full-development generators/features feed earlier conditional OOF; confirmation and holdout were reused; saved RAW nominal times lack actual ready/fit provenance for the strict dynamic gate. Evidence supports these limitations, **not a claim that every pipeline leaks in every way**. Current pre-open contract and frozen scorer have explicit label-availability and common-mask checks. No protected-data audit scan was run to infer more.

Raw/model binaries/caches are mostly ignored; all 51 inspected local JSONs and SHA-256s are listed in the evidence record. This supports finding records again on this workstation. It does not supply missing source data to a clone or guarantee whole-run reproducibility. Requirements are unpinned, `.python-version` is absent, and optional CLOB distribution metadata was not available under its declared package name. Reproducibility across installations/OS is UNKNOWN.

## 4. Experiments

Detailed hypothesis/commit/data/protocol/input/artifact/result/reuse/limitations/replacement fields are in [EXPERIMENTS.md](../EXPERIMENTS.md). This index includes failures and incomplete work.

| ID | Experiment | Status | Result | Reusable evidence |
|---|---|---|---|---|
| E0001-E0002 | Market-open models/shadow; state collection versions | CLOSED_INCONCLUSIVE / UNKNOWN completion | Prediction gains do not establish economic success; start manifests only for some collectors | PARTIAL |
| E0003 | Causal raw pre-open baseline | CLOSED_POSITIVE, prediction only | Small historical gain vs constant; no original prestart economics | PARTIAL |
| E0004-E0006 | Full training/parity; original replay; sizing/exit studies | CLOSED_INCONCLUSIVE / SUPERSEDED | Real code and limited parity, old valuation/timing replaced, no stable policy replacement | PARTIAL |
| E0007-E0008 | Live readiness; causal research/fee corrections | CLOSED_NEGATIVE | NO-GO; corrected policy selects cash; active controls lose | PARTIAL, corrected historical scope |
| E0009-E0010 | Model update; prospective test | CLOSED_INCONCLUSIVE (B partial) / PLANNED deferred | B 4/5 trials; no B holdout; no prospective sample | PARTIAL / planning only |
| E0011-E0012 | VP/RP ablation and joint tuning | SUPERSEDED | Interrupted scope change, no completed original design | Stage costs/provenance only |
| E0013 | Full-development one-pass fit | CLOSED_NEGATIVE economic claim | Fit complete, selected 2-trade -$0.41 holdout | Conditional historical evidence |
| E0014-E0017 | Return/market-offset/nested/interactions/shrinkage | CLOSED_INCONCLUSIVE; E0017a economic path INVALID | Mixed scores/zero trades/negative costs; invalid missing-token path corrected | PARTIAL; invalid economics NO |
| E0018-E0019 | Flow features and shared-mask ablation | CLOSED_NEGATIVE / CLOSED_POSITIVE conditional | Different row/anchor protocols give different conclusions | PARTIAL, preserve both |
| E0020a-d | Baseline/seeds/full features/RECENT365 | INCONCLUSIVE; RECENT365 CLOSED_NEGATIVE | No replacement established; exact recent-window procedure worse | PARTIAL |
| E0021 | RAW BASE64/70 historical holdout | CLOSED_NEGATIVE economic claim | -$529.30 / -$546.22 independent-order PnL | PARTIAL, exposed |
| E0022a-c | Attribution / market increment / frozen transfer | CLOSED_INCONCLUSIVE overall | Conditional development gain; LL transfer intervals cross zero | PARTIAL |
| E0023a-c | Microstructure / partition provenance / capacity | CLOSED_NEGATIVE / limited diagnosis / INCONCLUSIVE | Interaction worse; source holes; capacity selects control | PARTIAL |
| E0024 | Dynamic/action-value feasibility | BLOCKED | 637/2,000; fold1 908/1,000; RAW provenance unproven | YES for failed gate, NO for model result |
| E0025-E0028 | ESN portfolio, architecture, expanded, focused | CLOSED_INCONCLUSIVE independent gain | Searches complete; memory control strong; no independent winner | PARTIAL development only |
| E0029 | Frozen A/B/C/D technical/quality comparison | CLOSED_INCONCLUSIVE | D-B interval crosses zero; saved parity checks pass | PARTIAL |
| E0030 | Historical candidate robustness | CLOSED_INCONCLUSIVE | Both primary adjusted intervals cross zero in both periods | PARTIAL exposed scores |
| E0031-E0032 | Gamma target / sparse economic attribution | CLOSED_INCONCLUSIVE | B/D modest sparse positive PnL; 4,193 price causes unresolved | PARTIAL |
| E0033a-b | Settlement-rule and per-regime prediction scores | CLOSED_INCONCLUSIVE; price reconstruction BLOCKED | Regime association, no resolved TWAP60 ESN gain, no TWAP economics | PARTIAL |
| E0034 | No-order shadow and L2 source/sample pilot | BLOCKED | 0/3 windows; DNS failure records; stricter deadline conflict | YES for zero observed pilot, NO for live result |

No experiment is marked RUNNING solely because a stale manifest says so. Process enumeration observed other Python processes; their work was not interrupted, identified as these experiments, or treated as proof of completion. No new roadmap/experiment was started.

## 5. Active decisions

Full records including dates/reasons/consequences and replacement links: [DECISIONS.md](../DECISIONS.md).

| ID | Decision | Status |
|---|---|---|
| D0001-D0002 | UTC splits/exposure; exact proxy label/release semantics | ACTIVE |
| D0003-D0004 | T60 choice/T59 execution; causal quote/depth/minimum units | ACTIVE |
| D0005 | Earlier-only independent evaluation; one-pass results explicitly conditional | ACTIVE, unresolved methodological deviation documented |
| D0006-D0007 | Estimated dated fees/self-financing controls; evaluated vs deployment separation | ACTIVE |
| D0008 | Freeze 24 members, transforms, feature order and primary comparisons | ACTIVE |
| D0009 | Older 240-day prospective protocol | ACTIVE, deferred |
| D0010 | A/B/C/D no-order precision/final-look/storage rules | ACTIVE; readiness deadline conflict T0007 unresolved |
| D0011-D0012 | Failed dynamic provenance gate; official labels vs date-assigned settlement regimes | ACTIVE |
| D0013-D0014 | Baselines not deployed winners; canonical state/audit-only scope | ACTIVE |
| D0015 | Earlier arbitrary 180-day A/B/C/D recommendation | SUPERSEDED by D0010 |

No invented REVOKED decisions. Conflicts are recorded instead of guessing a current authority where none is explicit.

## 6. Protected invariants

The complete register in DECISIONS includes **Invariant / Reason / Source / Protected since / Can change only if** for 14 protected elements. Preserve label/timestamps/splits/exposure, earlier-only learned fitting, native/proxy distinction, source continuity, order-minimum units, dated fee arithmetic/rounding, continuous $100 paths, $5 and 5%-free-cash controls, frozen features/seeds/transforms, unchanged dynamic gates and actual RAW provenance, evaluated/deployment separation and no interim significance/unauthorized trading. None changed in this audit.

## 7. False / overstated completion claims

The label **FALSE / OVERSTATED COMPLETION CLAIM** applies to the expanded claim in the left column, not automatically to its cited source. Several current reports already state the limitations correctly: no unqualified repository-wide `PRODUCTION READY` claim was established by this audit. The first group below records unsupported completion inferences that must not enter canonical state; it does not assert that the cited documents actually made those expanded claims. Actual stale/conflicting statements are identified separately in the remaining rows. A historical subset PASS is not disproved merely because today's full suite fails.

| Claim / source | What evidence actually supports | Missing evidence / recovered classification |
|---|---|---|
| **FALSE / OVERSTATED COMPLETION CLAIM:** global reading of `complete`, saved `passed`, or old small test counts in status/ESN reports as current all-tests PASS | Current discovery: 675, 665 PASS, 2 FAIL, 8 ERROR. Isolated ESN subset passes | Full current suite is FAIL; T0001-T0003. Historical subset PASS remains dated subset evidence |
| **FALSE / OVERSTATED COMPLETION CLAIM:** `complete_search_validation_only` or candidate freeze interpreted as independently validated ESN advantage | Search-selection results, saved technical parity; adjusted primary historical intervals include zero | Independent new-sample quality/economic result absent; E0025-E0030 |
| **FALSE / OVERSTATED COMPLETION CLAIM:** probe-selected ESN portfolio described as downstream-best | E0026 portfolio is +0.000158274 in probe, -0.000046807 downstream; best selected single +0.000078732 | Downstream selection was not changed to single winner; probe scores are not LightGBM outcomes |
| **FALSE / OVERSTATED COMPLETION CLAIM:** original full-stack completion inferred from pre-open v1 raw baseline or completed one-pass pipeline | Baseline explicitly excludes learned old layers; one-pass OOF conditional on development selection | Complete independently nested generator/feature/model/policy validation absent; D0005 |
| **FALSE / OVERSTATED COMPLETION CLAIM:** completed four-family ablation or original multi-fold VP/RP study inferred from successor fit | Both manifests say interrupted_scope_change; successor has distinct procedure | Original comparisons incomplete and SUPERSEDED, E0011/E0012 |
| **FALSE / OVERSTATED COMPLETION CLAIM:** all model-update candidates evaluated on holdout | B tuning 4/5 minimum, no B holdout score; completed A comparisons exist | B incomplete, E0009 |
| **FALSE / OVERSTATED COMPLETION CLAIM:** +$28.01 B / +$22.66 D supports holdout/current TWAP60 execution | 1,143 sparse confirmation/single-price markets, no holdout economic rows; no actual T60 readiness | No TWAP60 depth/economics, no real fills; E0031-E0034 |
| **FALSE / OVERSTATED COMPLETION CLAIM:** accounting PASS validates inferred timing, live fills or true fee schedule | Fixture/saved ledger arithmetic and assumed timestamps only | Operational receipt/ACK/fill/reference fees unknown; E0007/E0021/E0031 |
| **FALSE / OVERSTATED COMPLETION CLAIM:** implemented shadow equals successful live collection | Direct SQLite counts zero market/data tables, saved DNS preflight errors | 0/3 pilot; no latency/growth result; current reachability UNKNOWN |
| **FALSE / OVERSTATED COMPLETION CLAIM:** empty source partition/zero economic rows establishes zero exchange liquidity | Source/initialization gaps and archive not covering target dates | Actual source coverage missing; cannot infer order-book state from absence |
| Stale negative implementation claim: status October 22 says `allow_protected_data: false`, scorer unimplemented | Current gate is true with exact prior approval; scorer + completed period records exist | Explicitly superseded by October 23, not grounds to rerun approved scorer during audit |
| Stale control-flow fields: original research manifest says running; focused manifest frozen_before_screening; completed manifest retains old selector error | Successful retry/correction, focused completed summary and separate failure archive show later state | Fields are stage/attempt-specific; neither automatic RUNNING nor automatic FAILED inferred |
| Stale rule claim: TWAP60 applied to full historical sample, earlier aggregate direction cells | Later seven-page/rule-calendar analysis and corrected 986/994, 1,090/1,123 totals | Earlier generalized rule description and wrong aggregates superseded; individual rule verification incomplete |
| Conflicting readiness rules: shadow deadline T60 vs generic pre-open deadline T59 | Both code paths and current tests encode their own rules; input closes at T60 | Practical equivalence UNKNOWN. Inference: positive receipt/compute delay could make every normal signal late under strict shadow rule; T0007, no repair in audit |
| README present-state instructions mention missing optimizer filename / Python version file | Actual script `optimize_lgbm_hyperparameters.py`, interpreter detected; referenced files absent | Setup/workflow not fully reproducible as written, T0006 |

## 8. Known blockers

1. **Dynamic-history research:** corrected 637 markets against minimum 2,000, fold1 908 two-sided-$5 training against 1,000, and unproven RAW prediction availability. Required source/provenance evidence is missing; no model fit should follow. Old original report uses 633; final audit adds four, not enough to reopen.
2. **Candidate live pilot:** saved Gamma/CLOB DNS failures prevent initialization and DB has zero windows. Current DNS not retested. Readiness semantics additionally need reconciliation before an eligible pilot can validate timing (T0007 -> T0004).
3. **Historical official-price reconstruction:** exact TWAP feeds, authorized access/retention/cost unknown; no price report covering target windows. Gamma outcomes alone cannot answer price-level cause (T0005).
4. **TWAP historical economics:** local book archive ends before relevant settlement regime. No target-period source data recovered; provider sample outside period only proves limited feasibility (T0005).
5. **Audit test acceptance:** full discovery is red; T0001-T0003. This blocks a green-suite claim, not this documentation-only recovery.
6. **Configured legacy refit:** original holdout manifest absent; configured later candidate output not found. Do not silently swap to a different evaluated bundle or run refit (T0006).

## 9. Unknowns requiring verification

- Actual current Gamma/CLOB reachability, full cold/warm BASE70 update latency under live feed, actual readiness/fill times and safe production behavior.
- Whether independent future data support memory or ESN increments; all historical primary adjusted comparisons inconclusive and search/exposure uncorrected by a new sample.
- Exact reason for 4,193 label disagreements; direct individual rule confirmation outside seven anchors; completeness and event ordering for all raw partitions.
- Complete asset-wide batch/live parity, ETH/SOL current model quality and active process purposes. Enabled settings are not proof of running/trading.
- Whether arbitrary earlier local outputs, checkpoints and manifests remain internally hash-equivalent to every dependent raw/model file. Audit hashed selected metadata, not multi-GB/bulk datasets.
- Whole-run clean-clone/cross-platform reproducibility; exact complete historical run timers/peak memory where original records did not preserve them. Cached resume time must not be reported as original full-run time.

### Gap analysis

| Category | What belongs here | Evidence / limit |
|---|---|---|
| A. VERIFIED | Current test outcomes; narrow passing mechanics; existence/hashes of inspected records; zero candidate shadow observations; failed gate values in saved record | Tests executed now and read-only inspection; this does not upgrade historical numerical results to recomputed truth |
| B. IMPLEMENTED BUT NOT VALIDATED | General pipeline, acquisition, bundle/scorer, fees/replay, live orders/redemption; historical conditional model comparisons | Real code and tests, limited saved validation; no full exchange/independent/production proof |
| C. PARTIAL | Interrupted ablation/model B, whole independent nested validation, no live pilot, legacy refit workflow | Named missing stages/prerequisites; future work must not start from a false completion |
| D. BLOCKED | Dynamic history/RAW gate, shadow pilot, exact Chainlink and historical TWAP depth, green test-suite claim | Specific gate/source/network/test blockers above |
| E. UNKNOWN | Generalizable edge, exact price cause, all-market rule/coverage, real fills, current other process purposes, clean-clone/cross-platform behavior | Explicit absent/unrun evidence; no UNKNOWN converted into PASS |

## 10. Superseded / dead work

| Path/procedure | Classification | Replacement / reason |
|---|---|---|
| Current source infrastructure, pre-open contract, evaluated bundles and candidate frozen scorer | ACTIVE | Explicit current code and frozen protocols; ACTIVE does not imply profitable or running |
| Older market-open/Kacho comparison, old single-asset architecture | LEGACY | Pre-open research and configured multiasset runtime for new work; original evidence retained |
| Original strict-book-age rejections, T59 selection, old fee/share-minimum valuations | SUPERSEDED | Contract D0003/D0004/D0006 and corrected paths E0008 |
| Four-family ablation/joint outer profile search | SUPERSEDED, interrupted | E0013 scope replacement; do not resume old plan accidentally |
| Invalid shrinkage economic path `prior_invalid_execution` | SUPERSEDED / INVALID economic evidence | Corrected token-oriented evaluation; quality subset not invalidated |
| ESN initial/r2 and uncorrected r4 input-group artifacts | SUPERSEDED attempts | Successful r3 portfolio / corrected r4 inputs; do not delete history |
| ESN probe portfolio as performance winner | SUPERSEDED interpretation | Downstream comparisons show better single candidates; later searches freeze new candidates |
| RECENT365 exact procedure | LEGACY, CLOSED_NEGATIVE | No promoted replacement; not an inference about all adaptation methods |
| Older 240-day prospective collection | PLANNED/deferred | Separate from A/B/C/D shadow, not silently restarted |
| Stale original research `running` manifest and configured refit paths | UNKNOWN control-flow freshness / LEGACY lineage | Completed retry/correction has separate identity; missing refit prerequisites stay explicit |
| Deleted original entrypoints such as `live_trade.py`, moved root utilities, old optimizer filename | DEAD at HEAD / historic paths | `run.py`, `utils/`, `optimize_lgbm_hyperparameters.py`; no deletion by audit |
| Other unavailable/deleted experimental branch work | UNKNOWN | Available refs do not establish deleted history |

## 11. Files created/updated

Created: root PROJECT_STATE, DECISIONS, EXPERIMENTS, TASKS; this audit report; compact metadata/test/source evidence JSON. Updated: AGENTS instructions and navigation from README/existing BTC checkpoint log. No competing second checkpoint history created, no old report removed. Exact path list is in section 13.

## 12. Tests/checks executed

### Environment and test scope

Detected interpreter: `C:\Users\philipz\.pyenv\pyenv-win\versions\3.14.4\python.exe`, Python 3.14.4, Windows. Libraries: NumPy 2.4.4, pandas 3.0.2, LightGBM 4.6.0, PyArrow 24.0.0, Numba 0.65.1, Optuna 4.8.0, scikit-learn 1.8.0. Pytest and psutil were not found; built-in unittest is the repository's actual test harness. No package installation or environment modification.

Resources inspected: i7-13650HX, 14 cores / 20 logical processors, ~63.7 GiB RAM with ~44.6 GiB free at initial reading; D: ~147 GiB free; RTX 4060 Laptop 8,188 MiB, 1,602 MiB in use at sampled reading. Other Python processes were present; none terminated. No heavy experiment was run and no benchmark/performance optimization claim is made.

| Exact Python command (repository root) | Tests | PASS | FAIL | ERROR | SKIP | XFAIL | unittest seconds | Command wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `python -m unittest discover -s tests -v` | 675 | 665 | 2 | 8 | 0 | 0 | 51.735 | 62.653 |
| `python -m unittest discover -s tests -p test_btc_esn_architecture_audit.py -v` | 7 | 7 | 0 | 0 | 0 | 0 | 0.515 | 2.922 |
| `python -m unittest discover -s tests -p test_btc_esn_expansion.py -v` | 8 | 8 | 0 | 0 | 0 | 0 | 0.930 | 3.166 |
| `python -m unittest discover -s tests -p test_optimize_trade_policy.py -v` | 6 | 5 | 1 | 0 | 0 | 0 | 0.024 | 0.955 |
| `python -m unittest discover -s tests -p test_run_multi_asset_latency.py -v` | 15 | 14 | 1 | 0 | 0 | 0 | 0.005 | 2.270 |

The 36 isolated executions overlap the 675-test suite; do not add them as new unique tests. PowerShell wrappers used `Stopwatch` and redirected each stream to `%TEMP%/5min_...log`; exact local log paths and SHA-256s are in audit JSON. Both stdout/stderr were retained. No skipped/expected-failure/unexpected-success result was reported. There are 80 `test_*.py` files. These are existing test commands, not newly introduced argument parsing in project scripts.

The suite has no formal disjoint unit/integration/regression/property/performance/E2E/cross-platform markers. Therefore exact per-category test counts are UNKNOWN rather than fabricated:

| Kind | Evidence executed / not executed |
|---|---|
| Unit | Many synthetic target/feature/fee/policy/config assertions in full discovery |
| Integration | Temp Parquet/SQLite round trips, config/orchestration and local frozen development-cache checkpoint test; included in discovery. Requires ignored local assets; not a pure clean-clone unit suite |
| Regression | Existing timestamp-resolution, mixed-schema, cashflow, source-gate, streaming/checkpoint and telemetry cases; included, with failures below |
| Property/invariant | Handwritten future-perturbation/prefix/batch-stream/cash assertions included; no separate property-testing framework/generative property suite established |
| Performance | Dedicated representative benchmark runners not executed; NOT_VERIFIED_IN_THIS_AUDIT. Historic cost outputs inspected only |
| End-to-end | No full training/replay/deployment or real endpoint/order lifecycle executed; NOT_VERIFIED_IN_THIS_AUDIT |
| Cross-platform | Windows only; cross-platform NOT_VERIFIED_IN_THIS_AUDIT |

Warnings: pandas `Pandas4Warning` for deprecated `copy` keyword (`features/candle_features.py:941` and `audit_feature_readiness.py:2968`); SQLite `ResourceWarning` for an unclosed connection. They were not hidden or repaired. No training/full-backtest rerun for documentation validation.

### Exact failures and diagnostic interpretation

Full discovery errors (all are also recorded in the local full log):

- `test_btc_esn_architecture_audit.ESNArchitectureAuditTests.test_controlled_anchor_pool_covers_every_family_and_topology_at_128_units`: KeyError `common_anchor`.
- `...test_downstream_resume_skips_cached_seed_without_deleting_missing_baseline`: RuntimeError missing frozen baseline validation probabilities.
- `...test_input_group_stage_covers_each_family_and_each_group_separately`: KeyError `input_groups`.
- `test_btc_esn_expansion.BTCESNExpansionTests.test_boundary_report_contains_family_and_unit_budget_rates`: KeyError `initial_candidate_sampler_seed`.
- `...test_downstream_exploration_keeps_thirty_percent_probe_independent`: same missing key.
- `...test_downstream_pool_resume_reuses_checkpointed_ids_and_order`: same missing key.
- `...test_inherited_component_counts_match_architecture_shape_and_budget`: KeyError `parallel_branch_choices`.
- `...test_initial_pool_is_unique_and_stratified_by_family`: KeyError `initial_candidate_sampler_seed`.

Evidence for state contamination, not eight independent missing production configs: earlier candidate checkpoint test calls `_configure_source_run` -> `_configure_engines`, which sets `expansion.CONFIG` and shared engine state to focused-study values; expected original architecture/expansion keys then differ. All 15 tests of those two modules pass in fresh processes. This diagnostic does not waive the full-suite errors or fix them (T0001).

Persistent FAILs:

- `test_optimize_trade_policy.OptimizeTradePolicyTests.test_run_optimization_saves_runtime_config_with_fixed_slippage_ticks`: expected 3, actual 2. Current module constant is 2, unpatched fixture expects 3. Intended authority unresolved, T0002.
- `test_run_multi_asset_latency.LiveSubmitTelemetryTests.test_successful_submit_response_is_distinct_from_send_and_fill_events`: expected `submitted_fok`, actual `submission_error`. Calling only its existing fake trader/response/helpers showed error `'PolymarketLiveTrader' object has no attribute 'pm_user_trade_monitor'`. No client/network invoked. Test bypasses real initializer via `__new__`; this is a fixture/initialization-coverage discrepancy, not proof of an actual rejected order (T0003).

### Other diagnostics

Exact Git/environment inspection commands: `git status --porcelain=v2 --branch`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'`, `git remote -v`, `git branch -avv`, `git tag -n`, `git ls-remote --heads --tags origin`, `git log --all --reverse --date=short --format='%h %ad %s'`, `git rev-list --all --count`, milestone `git show --format=short --stat`, `python --version`, module discovery/version metadata, `Get-CimInstance Win32_OperatingSystem`, `Get-CimInstance Win32_Processor`, `Get-CimInstance Win32_Process -Filter "name='python.exe'"`, `Get-PSDrive D`, `nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader`.

GitHub connector read-only GET `https://api.github.com/repos/philipzabicki/5min_up_down_pred/releases?per_page=100` returned `[]`. No messages, PRs, releases or settings were written.

Read-only metadata evidence generation used Python `Path.read_bytes`/SHA-256 on the 51 exact JSON paths in audit evidence, reading their recorded status fields, plus `sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)`. Tables were enumerated with `SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'`, counted using `SELECT COUNT(*)`, and the two `collector_events` read as observed time/type/payload. No collector connection in write mode and no market/prediction data read from protected Parquets.

Documentation acceptance checks passed: 180 local Markdown links, zero missing targets; 34 root experiment records (plus explicit variants), 15 decision records, 8 task records, all 14 required report sections; JSON parsing, 51 present local metadata hashes and 23 unchanged source/config/test hashes; test-count reconciliation, secret-pattern scan and `git diff --check`. Largest new file is about 54 KB. Final staged-diff/whitespace review precedes the focused commit. Diagnostic shell quoting was corrected after one search expanded outside its intended scope; it was stopped, produced no implementation changes and its unrelated output is not used as audit evidence.

## 13. Git changes made by this audit

Only these intended paths belong in the audit commit:

- `PROJECT_STATE.md` (new concise snapshot).
- `DECISIONS.md` (new decision/invariant register).
- `EXPERIMENTS.md` (new experiment/status/provenance register).
- `TASKS.md` (new acceptance-based tasks, including recovered issues).
- `docs/PROJECT_STATE_AUDIT_20261025.md` (this report).
- `reports/project_state_recovery_20261025.json` (small evidence record; no raw data/models/secret values).
- `AGENTS.md` (canonical-state workflow rules).
- `README.md` (current-state navigation only).
- `docs/BTC_RESEARCH_STATUS_20261010.md` (current-state/supersession notice, old history retained).

No source algorithm, feature definition, data split, model, runtime trading setting, test, raw data or earlier result changed. Temp unittest logs stay outside Git. The normal push is required by the existing AGENTS Git rules; only the new audit commit may be pushed, after checking outgoing commits. No merge/release/deployment/order is authorized.

## 14. Recommended next tasks

1. T0001-T0003: repair full-suite state isolation and resolve the two persistent discrepancies in separately scoped implementation work, preserving model/experiment semantics.
2. T0007 -> T0004: resolve nominal decision vs actual readiness deadline, then verify public connectivity and complete a three-window no-order pilot with actual timing and failure denominators.
3. T0005: independently verify exact historical TWAP feed/L2 identity, coverage, entitlement and cost before any separately authorized acquisition/reconstruction. Do not repeat old tuning or relax failed gates to conceal missing data.

T0006 is a smaller reproducibility/documentation prerequisite before using legacy training/refit instructions. No new milestone or experiment is automatically started by this list.
