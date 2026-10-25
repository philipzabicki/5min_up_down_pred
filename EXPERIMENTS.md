# Experiment register

Recovered at `cf885b0`, 2026-10-25. IDs are durable; suffixes identify separate variants within a historical family. No experiment was run by this audit. CLOSED describes the stated hypothesis and sample, never production readiness. A completed runner can have a negative or inconclusive result.

Evidence levels: **YES** = reusable for the explicitly limited claim, **PARTIAL** = conditional, incomplete, superseded in part, or report-only, **NO** = no result supporting the proposed claim. Historical numeric results below were read from saved reports/result JSON, **NOT_RECOMPUTED_IN_THIS_AUDIT**. The [evidence inventory](reports/project_state_recovery_20261025.json) records which local JSONs were independently found and hashed. That verifies file identity, not the historical calculations. Unchecked old report results are **CLAIMED PREVIOUSLY - NOT INDEPENDENTLY VERIFIED**. Local `data/analysis/...` outputs are ignored by Git; their presence is not clean-clone reproducibility.

`Code commit` gives the source/introduction or explicitly recorded execution revision; it is not silently treated as the execution revision if none was captured. Full command/test evidence is in the [audit](docs/PROJECT_STATE_AUDIT_20261025.md). Historical periods were exposed; do not rerun protected inputs merely to refresh this register.

## E0001 - Market-open/Kacho old vs new BTC and shadow

Status: CLOSED_INCONCLUSIVE. Name: old/new BTC, market-only and market-plus-BTC comparison.
Hypothesis: newer BTC adds official-outcome information and profitable entries beyond market-only.
Code commit: `327324a`; parity corrections `1941e3a`, `97eeee3`.
Data: common 9,407 Kacho windows, April 15-May 18, 2026; historical OOF and official Gamma labels.
Protocol / inputs: past-selected logistic market layer, delayed top-of-book execution; old/new BTC OOF; distinct market-open timing, not the later pre-open protocol.
Output artifacts: [new model report](docs/btc_new_model_evaluation_20261003.md), [shadow v2 audit](docs/btc_shadow_evaluation_20261004.md), frozen models/configs and local `new_model_comparison`, `entry_selection_audit_20261003`, shadow sessions.
Result: reported new-vs-old LL improvement and small market-plus-new improvement; entry diagnostic reports 810 independent BTC-Platt signals losing $383.72, while a funded $100 path takes 73 and ends $2.11. These are different estimands.
Conclusion: prediction comparison is not evidence of an executable trading advantage.
Reusable evidence: PARTIAL, report/manifest lineage and preserved mechanics; historical values not independently recalculated.
Known limitations: source OOF selection/early stopping, simulated fills, changed feature-history definitions; v1/v2 shadow durations differ. Current success/current process state UNKNOWN.
Superseded by: pre-open E0003 onward for the active research question; v1 shadow schedule by v2 for that older collector.

## E0002 - Entry-state and pre-open data collection versions

Status: UNKNOWN (collection completion); implemented collectors are not completed experiments.
Hypothesis: record causal price, model, quote and outcome state for later diagnostics.
Code commit: `4e8d47f`, `bb72dc0`, `90016fc`.
Data: local `entry_state_collection_20261004_v1/v2`, `preopen_collection_20261004_v1/v2/v3`; session manifests exist.
Protocol / inputs: versioned collection protocols in `configs/btc_entry_state_collection_protocol_*` and `configs/btc_preopen_collection_protocol_*`; frozen model and public feeds.
Output artifacts: [entry-state report](docs/btc_entry_state_experiment_20261004.md), local session databases/manifests.
Result: start manifests establish attempts; they do not establish continuous operation, resolved coverage or final study success.
Conclusion: retain separate version lineage and explicit UNKNOWN final status.
Reusable evidence: PARTIAL, implemented mechanics and recorded starts.
Known limitations: full session tables and all running processes were not audited for historical completion; no live requests made.
Superseded by: later candidate shadow E0034 for A/B/C/D collection; older records retained.

## E0003 - Causal pre-open raw-feature baseline

Status: CLOSED_POSITIVE (historical prediction comparison only).
Hypothesis: a causal minute-candle baseline beats constant 0.5 on official outcomes.
Code commit: `bb72dc0`; local `preopen_v1/run_manifest.json` records completed status.
Data: 3,321,148 minute rows; fit/iteration selection before 2026-01-01; later calibration; April 15-May 18 external historical sample.
Protocol / inputs: 29 raw causal features; 19 selected rounds; target/release in D0002; labels purged by availability. No old fitted indicator/VP/RP artifacts reused.
Output artifacts: [baseline report](docs/btc_preopen_experiment_20261004.md), [manifest](docs/btc_preopen_experiment_manifest_20261004.json), local `preopen_v1/evaluation.json`.
Result: reported calibrated LL 0.69238 vs 0.69315; 95% paired delta interval [-0.001495, -0.000018]. Zero eligible prestart quotes in that original source.
Conclusion: a historical prediction result; no economic result and no full-stack completion at that point.
Reusable evidence: PARTIAL; manifest independently present, numerical result not rerun; prior exposure applies.
Known limitations: no measured full-run runtime/peak RAM and no prestart execution coverage in this experiment.
Superseded by: E0004/E0008/E0013 as fuller pipelines, not retroactive validation of them.

## E0004 - Target-specific resumable training and candidate parity

Status: CLOSED_INCONCLUSIVE (technical implementation complete, advantage unresolved).
Hypothesis: build a target-specific indicator/profile/selection/training bundle with reproducible live features.
Code commit: `571019c`, `1d83317`, `206cc71`.
Data: frozen pre-open training configs and local target-specific stages; distinct from old market-open artifacts.
Protocol / inputs: `run_btc_preopen_experiment.py`, `configs/btc_preopen_training.json`; indicator/VP/RP/weights/feature selection/model stages, bundle hashes and parity checks.
Output artifacts: [training protocol](docs/btc_preopen_training.md), [audit](reports/btc_preopen/audit.json), `artifact_manifest.json`, `original_bundle_verification.json`, `live_feature_parity.json`, runtime compatibility report.
Result: saved audit verdict `correct_with_limits`; later fixes distinguish original bundle, research candidate and actual runtime. Test existence or a saved parity PASS does not certify actual order execution.
Conclusion: substantial pipeline/bundle code exists; whole-system production validation UNKNOWN.
Reusable evidence: PARTIAL, limited bundle/parity checks and code; not rerun in full.
Known limitations: target-specific fitted artifact lineage and measured runtime parity apply only to identified bundles.
Superseded by: later evaluated bundles for their own experiments; original bundle remains preserved.

## E0005 - Original event economics and T-59 book corrections

Status: SUPERSEDED (original eligibility/timing conclusions).
Hypothesis: price historical pre-open entries from archived books at executable times.
Code commit: `b9e8fad`, `206cc71`, `d7557bc`, `a3eb722`.
Data: PMXT events, supplemental snapshots, frozen model predictions, Gamma outcomes.
Protocol / inputs: successive event replay, locked-book handling, BBO reconciliation and T-59 reassessment versions.
Output artifacts: [summary](reports/btc_preopen/SUMMARY.md), `economic_replay.json`, `event_replay_summary.json`, `fresh_bbo_replay_summary.json`, [T59 report](reports/btc_preopen/t59_reassessment_20261008/t59_reassessment_20261008.md).
Result: earlier strict crossing/age/depth exclusions changed after corrections; source incompleteness did not prove no exchange liquidity. Old apparent successful replay is not current causal economic evidence.
Conclusion: reuse source traces/checkpoints only with their identity and semantics; original valuation/timing claims are superseded.
Reusable evidence: PARTIAL, source provenance and diagnosis; NO for current timing/fee profitability claims.
Known limitations: missing historical receive/ACK/fill times, archive ambiguity and changing replay semantics.
Superseded by: D0003/D0004/D0006, E0007/E0008 and corrected diagnostic paths.

## E0006 - Policy, robust Kelly, position-state and exit studies

Status: CLOSED_INCONCLUSIVE.
Hypothesis: sizing, robust intensity and early exits improve frozen pre-open economics.
Code commit: `68f69ad`, `90c9277`, `4b23a1a`.
Data: development historical snapshots/predictions/outcomes; original policy timing and fee definitions.
Protocol / inputs: 58 policy configurations, walk-forward selection, DRK intensity, paired sizing, v1/v2 position state, optimal-stopping comparisons; no refit of the BTC signal.
Output artifacts: [continuation report](reports/btc_preopen/policy_continuation_20261007.md), [policy study](reports/btc_preopen/policy_study.json), `advanced_policy_*`, `drk_intensity_*`, `paired_sizing_*`, `optimal_stopping_*`, `position_state_comparison_*`.
Result: no stable policy replacement; selected DRK PnL $230.43 vs point Kelly $312.08 in that development setup; no exit dominates risk and PnL. Final policy artifact is explicitly inactive.
Conclusion: completed searches, not validated deployment or proof of optimized profitability.
Reusable evidence: PARTIAL; preserve comparisons within old protocol, not current fee/timing valuations.
Known limitations: reused development, prior protocol assumptions and high path dependence. Numeric historical results not recomputed.
Superseded by: causal contract/research for new evaluations; individual studies remain historical evidence.

## E0007 - Historical fee reconstruction and live readiness

Status: CLOSED_NEGATIVE (live-readiness question).
Hypothesis: reconstructed fees/replay and deployed runtime suffice for live operation.
Code commit: `0319d58`, `28e1e48`, `41d7004`, research contract `9d57864`.
Data: saved historical fee source captures, Gamma metadata, replay and runtime manifests.
Protocol / inputs: historical dated V1/V2 fees, order/depth assumptions, runtime vs candidate identity review.
Output artifacts: [readiness NO-GO](reports/btc_live_readiness_20261010/readiness_report.md), [fee history](reports/btc_fee_history_20261010/historical_fee_report.md), [policy fee report](reports/btc_fee_policy_20261010/historical_policy_report.md), fee registries.
Result: readiness explicitly NO-GO; active bundle differed; V2 signing/fills/fee behavior not verified. Later contract superseded price-age rejection and fixed accepted estimated fee dates.
Conclusion: code and retrospective accounting do not establish live readiness.
Reusable evidence: PARTIAL; dated limitations/decision and saved source captures; no current external-source verification.
Known limitations: exact per-market historical fees and real fills unknown; audit did not query exchange or submit orders.
Superseded by: D0003/D0004/D0006 for conflicting old assumptions; NO-GO has no later successful operational evidence.

## E0008 - Causal walk-forward research and fee/signal corrections

Status: CLOSED_NEGATIVE (selected active economic advantage).
Hypothesis: a causal full research pipeline supports an after-fee policy beyond cash.
Code commit: `f4424ca`, `3c90264`, `1b01c9f`, `c734ea2`, `8de9135`.
Data: development before July 1; historically exposed July-October holdout; full_compute20_retry predictions and T60/T59 snapshots.
Protocol / inputs: nested research runner, development-selected calibration/policy, isolated $100 holdout; later fixed historical fees and minimum USDC units; signal diagnostic corrected execution-time EV.
Output artifacts: [original](reports/btc_preopen/btc_preopen_research_20261011_full_compute20_retry.md), [fee correction](reports/btc_preopen/btc_preopen_research_fee_correction_20261012.md), [signal correction](reports/btc_preopen/btc_preopen_signal_diagnostic_correction_20261012.md); local corrected manifests.
Result: corrected selection is no-trade; fixed-$5 control lost $98.82 on 566 funded trades; 5%-free-cash control lost $0.60 on 2. Earlier valuations are superseded.
Conclusion: completed causal research does not yield positive selected after-fee holdout PnL.
Reusable evidence: PARTIAL, corrected historical comparison and accounting, not independent holdout.
Known limitations: ignored source artifacts, simulated execution; original path still says `running` and lacks configured evaluated-holdout input, unlike completed retry/correction paths.
Superseded by: corrected fee/signal outputs for numerical valuations; later models are separate experiments.

## E0009 - Bounded model update and pruning diagnostic

Status: CLOSED_INCONCLUSIVE; B holdout sub-run BLOCKED/incomplete.
Hypothesis: A_reoptimized or dynamic price-volume B improves A_frozen.
Code commit: `a4eb863`, `06036c9`, `c39f405`.
Data: 10,795 common development markets; historical holdout for available A models.
Protocol / inputs: bounded successful-trial minima; 97.5% paired adjusted interval for challenger promotion; same shared feature-selection path.
Output artifacts: [partial report](reports/btc_preopen/btc_preopen_model_update_20261012_partial.md), local `development_model_freeze.json`, `model_update_manifest.json`; [pruner diagnostic](reports/btc_preopen/b_pruner_development_diagnostic_20261014/pruner_diagnostic_report.md).
Result: no challenger clears promotion interval; retain A_frozen. B holdout tuning stopped at 4/5 required successful trials, no B holdout score. Pruner analysis is a separate development diagnostic, not a completed B model evaluation.
Conclusion: explicit partial completion; never substitute A's result for B.
Reusable evidence: PARTIAL, completed A/development pieces and failure diagnosis.
Known limitations: unfinished B; earlier stages/resource pauses and successful-trial accounting matter.
Superseded by: later model studies as new hypotheses; no evidence B holdout was completed.

## E0010 - Older prospective BTC-plus-market test

Status: PLANNED (deferred, not started).
Hypothesis: frozen market-plus-BTC improves new-sample paired LL beyond market-only.
Code commit: `d8e2540`.
Data: power planning from 8,676 development markets / 61 calendar days; no new 240-day sample established.
Protocol / inputs: fixed 240 days, 24-hour outcome grace, 5,000 bootstrap replications, no interim significance, fixed coverage gates.
Output artifacts: [protocol/status](docs/BTC_PROSPECTIVE_TEST_20261014.md), [power report](reports/btc_prospective_test_20261014/power_and_observation_plan.md), frozen manifest/public smoke summary.
Result: planning and public smoke exist; long collection explicitly deferred by modeling checkpoint.
Conclusion: no prospective experimental outcome.
Reusable evidence: YES for planning/protocol only; NO for prospective advantage.
Known limitations: power extrapolation from sparse, reused historical development; distinct from E0034.
Superseded by: none for its own frozen protocol; not automatically reactivated.

## E0011 - Four-family VP/RP ablation and helper diagnostic

Status: SUPERSEDED (unfinished ablation).
Hypothesis: compare base-only, VP, RP and combined features through full selection.
Code commit: `320e795`, `bae5d50`.
Data: target-specific development folds, calibration-seed stages.
Protocol / inputs: outer-fold ablation in `run_btc_preopen_feature_ablation.py`; helper diagnostic is a separate small pilot.
Output artifacts: [followup report](reports/btc_preopen/btc_modeling_followup_20261015.md), [helper diagnostic](reports/btc_preopen/btc_generator_helper_diagnostic_20261015.md), local ablation manifest.
Result: only base_only/base_plus_vp calibration-seed pipelines completed; RP selection partial; no complete four-family outer comparison. Manifest `interrupted_scope_change` confirms interruption.
Conclusion: cannot infer family ranking from this incomplete design.
Reusable evidence: PARTIAL for costs/completed stages; NO for full ablation conclusion.
Known limitations: helper pilot is not an authorized full workload substitute.
Superseded by: E0013 after explicit scope change.

## E0012 - Joint bounded VP/RP tuning

Status: SUPERSEDED (interrupted).
Hypothesis: optimize profile settings through repeated outer pipelines.
Code commit: `fbebc80`; scope replacement `0f96f5b`.
Data: development pipeline inputs; no final independent family comparison.
Protocol / inputs: [tuning specification](docs/BTC_VP_RP_TUNING_20261016.md), frozen research config and checkpoints.
Output artifacts: local `btc_preopen_vp_rp_tuning_20261016/model_update_manifest.json`, historical status checkpoints.
Result: manifest `interrupted_scope_change`; later single full-development run is a different procedure.
Conclusion: no completed original multi-fold tuning claim.
Reusable evidence: PARTIAL for preserved stages/provenance; NO for original full experiment.
Known limitations: no full repeated-search validation.
Superseded by: E0013.

## E0013 - One-pass full-development fit

Status: CLOSED_NEGATIVE (selected policy net gain); training pipeline completed.
Hypothesis: full-development target-specific fit and calibrated 1% EV policy produce positive after-fee results.
Code commit: `0f96f5b`; report `8a556a8`; local run ID `794a69e369a3f82d2ec161af839cdb7992339e34a6606b8bce7712da7fd610f7`.
Data: eligible price history through development; 318,614 conditional calibration OOF rows; historical holdout before October 7.
Protocol / inputs: one VP study (96 trials), RP study (96; 77 complete/19 pruned), model study (60; 22 complete/38 pruned); 10 internal folds, 64 final features, no repeated outer complete pipeline.
Output artifacts: [report](reports/btc_preopen/btc_preopen_full_development_fit_20261016.md), [freeze/evaluation files](reports/btc_preopen/btc_preopen_full_development_fit_20261016/frozen_bundle_manifest.json), local run/freeze manifests.
Result: saved selected-policy result 2 trades, $99.59 ending cash / -$0.41. Recovered selector `KeyError: 'k'`; completed manifest retains old error text and separate failed attempt archive.
Conclusion: completed technical fit, negative policy result, conditional OOF. D0005 deviation from full nested evaluation remains explicit.
Reusable evidence: PARTIAL; current manifests found/hashed, scores not rerun.
Known limitations: exposed holdout, generator/feature/model selection conditionality; old error field alone does not mean final run is still failed.
Superseded by: BASE64/BASE70/ESN variants for separate comparisons; no runtime promotion.

## E0014 - Huber return regression

Status: CLOSED_INCONCLUSIVE.
Hypothesis: normalized future-return regression plus causal logistic calibration improves binary probability estimates.
Code commit: `36375d5`, `cc5e85e`.
Data: frozen 64 features, 3,186,080 eligible development rows, fixed folds and historical holdout.
Protocol / inputs: 60-return causal sigma, Huber z target, unchanged weights/features, earlier-only OOF calibration.
Output artifacts: [report](reports/btc_preopen/btc_preopen_return_regression_20261017.md), local frozen/input/run manifests and holdout results.
Result: development paired LL difference -0.00000695, interval crosses zero; historical holdout point LL improves but is exposed.
Conclusion: no resolved independent superiority from this comparison.
Reusable evidence: PARTIAL, conditional prediction/target mechanics.
Known limitations: preselected generators/features and tuned regression parameters; historical metrics not recomputed.
Superseded by: none; retained diagnostic.

## E0015 - Conditional market-offset model and execution diagnosis

Status: CLOSED_INCONCLUSIVE (quality); no accepted frozen-policy holdout trades.
Hypothesis: conditional BTC correction to market logit improves official predictions and executable edge.
Code commit: `1387a13`, `84a5675`; `run_btc_conditional_market_offset_20261018.py`.
Data: causal development folds and exposed historical holdout, shared valid T60/T59 sample.
Protocol / inputs: six correction configurations, fixed offsets added once, mean LL + 0.5 SD selection, frozen hurdle.
Output artifacts: [model report](reports/btc_preopen/btc_conditional_market_offset_20261018.md), [execution diagnostic](reports/btc_preopen/btc_conditional_market_offset_20261018_execution_diagnostic.md), local completed run manifest.
Result: selected leaves2_l2_10, 43 trees; saved holdout decision table 24,736 rows and zero accepted orders. Prediction improvement does not imply executable 1% EV.
Conclusion: distinguish probability diagnostics from zero-trade economics; no validated trading edge.
Reusable evidence: PARTIAL, matched quality and execution exclusions under stated protocol.
Known limitations: reused selection history, no live fills.
Superseded by: none; later interactions are separate hypotheses.

## E0016 - Market/basis/volatility interactions and nested correction family

Status: CLOSED_INCONCLUSIVE.
Hypothesis: BTC adds information beyond market probability, basis, volatility and spreads; simpler/nested probability corrections can improve policy choice.
Code commit: `8de9135`, `dbe5183`, `0f61dfc` (three distinct frozen procedures).
Data: development OOF 8,676/8,675 common rows depending on eligibility; ~24,735-24,736 exposed holdout rows.
Protocol / inputs: market-only/logistic/combined baselines; nested E/F blends/offsets; basis-volatility M0/M1 freeze and paired bootstrap. Do not combine their sample counts or selection scores.
Output artifacts: [basic comparison](reports/btc_preopen/btc_market_probability_experiment_20261012.md), [nested](reports/btc_preopen/btc_market_probability_nested_20261012.md), [interaction](reports/btc_preopen/btc_market_basis_vol_interaction_20261017.md), separate configs/manifests.
Result: basic historical holdout raw market LL 0.69158, better than fitted corrections; nested development selects A_frozen policy, E-B/E-D intervals cross zero. Interaction results remain conditional historical diagnostics.
Conclusion: no general BTC-over-market/after-fee deployment conclusion from these distinct comparisons.
Reusable evidence: PARTIAL; individual protocols/results retained, no cross-protocol pooling.
Known limitations: exposed holdout and baseline-selection conditionality; changing coverage.
Superseded by: later BASE70 market-layer study E0022 for that model only.

## E0017 - One-parameter M1 shrinkage

Status: CLOSED_INCONCLUSIVE.
Hypothesis: shrink a previously selected market correction with one May-fitted alpha.
Code commit: `16eaf5c`.
Data: 5,252 May available-label fit rows; June frozen evaluation; no new model or holdout training.
Protocol / inputs: alpha in [0,1], exact saved raw correction, official label release, fixed economics after technical correction.
Output artifacts: [report](reports/btc_preopen/btc_m1_correction_shrinkage_20261018/report.md), local frozen manifest and run metadata.
Result: alpha 0.074270374954; exact endpoint reconstruction for alpha=0/1. May is fitting data, June was previously involved in selecting base M1.
Invalid prior sub-run E0017a: Status INVALID for its economic result only; missing UP/DOWN token IDs prevented all qualification. Preserved in local `prior_invalid_execution`; same alpha/probabilities and valid quality metrics were reused for corrected economics. Corrected calibrated variant has zero trades; original M1 has 792 fills and -$38.27. Do not cite the invalid zero-fill path as evidence about the policy.
Conclusion: valid conditional shrinkage diagnostic, not independent confirmation of the whole procedure.
Reusable evidence: PARTIAL; no independent robust economic gain inferred.
Known limitations: reused base-model selection and correction history.
Superseded by: none.

## E0018 - Six taker-flow feature additions

Status: CLOSED_NEGATIVE (the frozen comparison).
Hypothesis: taker-flow Z/V additions improve the selected 64-feature pipeline.
Code commit: `322716e`.
Data: fixed full-development binary run and Binance USD-M volume/quote-volume inputs; 21,951 official development outcomes.
Protocol / inputs: same weights/folds, causal source-derived flow features, B0/B1 quality and fixed-$5 replay.
Output artifacts: [report](reports/btc_preopen/btc_preopen_taker_flow_20261018.md), local source validation and `quality_and_economics.json`, completed run manifest.
Result: proxy OOF nearly tied; official LL/Brier/AUC worsened with intervals favoring B0.
Conclusion: closes this addition procedure; not a universal rejection of flow or of later shared-mask BASE70 comparison.
Reusable evidence: PARTIAL, conditional negative result.
Known limitations: selected feature matrix and changed common-row requirements motivate E0019 as a distinct comparison.
Superseded by: E0019 for shared-mask BASE64/70 analysis, not retroactive erasure of the negative result.

## E0019 - Shared-mask BASE64 vs BASE70 Z/V ablation

Status: CLOSED_POSITIVE (conditional development comparison).
Hypothesis: compare six flow features on identical valid rows/folds, avoiding original mismatched BASE64 anchor.
Code commit: `a0095c6`.
Data: 3,174,805 common rows; fixed BASE70 CV and newly fitted BASE64 on shared rows.
Protocol / inputs: 10 fixed folds; BASE64 25 rounds vs BASE70 31; causal labels, same policy; RAW/Platt separately.
Output artifacts: [report](reports/btc_preopen/btc_preopen_zv_ablation_20261018.md), versioned manifest, local quality/economics/run records.
Result: BASE70 lower proxy LL in 8/10 folds; reproduced 1,452-fill $260.05 anchor. This differs from post-calibration-warm-up subsets used in later studies.
Conclusion: limited positive shared-mask development evidence; does not establish holdout or full-pipeline independence.
Reusable evidence: PARTIAL, conditional comparison with source provenance.
Known limitations: preselected pipeline; different round counts; no runtime readiness proof.
Superseded by: E0021 for frozen historical transfer, not by relabeling development as unseen.

## E0020 - BASE70 probability baseline, seed, feature-access and recency studies

Status: CLOSED_INCONCLUSIVE (family synthesis; exact variant statuses below).
Hypothesis: improve frozen RAW SINGLE with calibration, seed ensemble, full per-tree feature access or a 365-day fit window.
Code commit: `f0fd2e0` (E0020a information value), `a3cd794` (E0020b seeds), `ba04cbe` (E0020c full features), `cf910af` (E0020d RECENT365).
Data: shared BASE70 matrix, 10 saved folds; official/sample economic masks differ from full proxy OOF. E0020b/c/d use fixed 31 rounds, no early stopping in new fits.
Protocol / inputs: RAW/HALF/causal PRIOR/PLATT; four extra seed grids with equal-weight five-model ensemble; feature_fraction=1 challenger; 10 recent-window fits with preserved weight mass.
Output artifacts: [information value](reports/btc_preopen/btc_base70_information_value_20261018.md), [seeds](reports/btc_preopen/btc_base70_seed_stability_20261018.md), [full features](reports/btc_preopen/btc_base70_full_features_20261018.md), [RECENT365](reports/btc_preopen/btc_base70_recent365_20261018.md), four configs/manifests and local completed run records.
Result: E0020a RAW/PLATT daily-PnL intervals include zero and gains concentrate in few days. E0020b does not support ensemble promotion. E0020c $243.44 vs SINGLE $267.44, paired daily-PnL interval crosses zero. E0020d official scores worse across all 3 months, $327.92 less PnL with negative paired interval.
Conclusion: retain SINGLE as a historical reference, no refit/promotion; RECENT365 closes only the exact tested procedure.
Reusable evidence: PARTIAL, each named conditional comparison; do not pool seed variability as time-sample uncertainty.
Known limitations: all depend on preselected BASE70; historical development reused; summary `next_step` can be stale even with completed results.
Superseded by: none as comparisons; later frozen candidates use their own protocol.

| Permanent variant ID | Name / hypothesis | Status | Code commit / exact source | Result / reusable evidence / superseded by |
|---|---|---|---|---|
| E0020a | RAW/HALF/PRIOR/PLATT information value | CLOSED_INCONCLUSIVE | `f0fd2e0`; `btc_base70_information_value_20261018` config/report/local output | Concentrated uncertain PnL; PARTIAL; none |
| E0020b | Five-seed ensemble stabilizes SINGLE | CLOSED_INCONCLUSIVE | `a3cd794`; `btc_base70_seed_stability_20261018` config/report/local output | No supported promotion; PARTIAL; none |
| E0020c | FULL_FEATURES improves original feature sampling | CLOSED_INCONCLUSIVE | `ba04cbe`; `btc_base70_full_features_20261018` config/report/local output | Quality/economic intervals cross zero; PARTIAL; none |
| E0020d | RECENT365 improves fixed whole-history procedure | CLOSED_NEGATIVE | `cf910af`; `btc_base70_recent365_20261018` config/report/local output | Official quality and daily PnL worsen; PARTIAL; none |

Each variant inherits only the shared frozen-matrix/fold input and limitations above; its separate named config, manifest and output directory define its protocol/data mask. No pooled result is substituted for an individual outcome.

## E0021 - RAW BASE64/70 historical holdout

Status: CLOSED_NEGATIVE (after-fee profitability).
Hypothesis: frozen RAW shared-mask BASE70 quality/economics transfer to July-October.
Code commit: `a54e786`, `d03763a`.
Data: previously exposed [2026-07-01, 2026-10-07); five missing October flow archives completed with official checksums.
Protocol / inputs: frozen 64/70 features, 25/31 rounds, identical quote/fee/cash rules, no holdout tuning; separate independent-order accounting.
Output artifacts: [report](reports/btc_preopen/btc_base70_raw_holdout_20261018.md), versioned manifest, local completed historical assessment / flow-validation records.
Result: 28,224 complete proxy rows, 28,220 official available outcomes; independent fixed-$5 PnL BASE64 -$529.30, BASE70 -$546.22.
Conclusion: completed negative after-fee historical result; repaired source absence is no longer the active blocker for this experiment.
Reusable evidence: PARTIAL, conditional and exposed; not the different 28,223-row A/B/C/D scorer.
Known limitations: assumed one-second execution, prior selection and historical quote availability; true fill latency not measured.
Superseded by: none; retain as negative evidence.

## E0022 - EV/PnL attribution and BASE70 market information/transfer

Status: CLOSED_INCONCLUSIVE overall; development M1-vs-M0 information result positive on its conditional sample.
Hypothesis: explain expected-vs-realized PnL and test incremental RAW information beyond market forecast.
Code commit: `37831fc` (E0022a attribution), `190a88a` (E0022b development), `d7d81c5` (E0022c frozen transfer); cash-debit reporting correction `a5fd05e`.
Data: existing saved ledgers/snapshots, 9,694 common development quality rows for daily M0/M1; 24,735 common historical holdout labels for transfer.
Protocol / inputs: exact EV/execution/outcome/target-switch decomposition; causal daily development logistic M0/M1; separate once-fitted frozen coefficients for holdout. Market midpoint is a forecast, not an ask.
Output artifacts: [attribution](reports/btc_preopen/btc_base70_ev_pnl_attribution_20261019.md), [development](reports/btc_preopen/btc_market_base70_incremental_20261019.md), [transfer](reports/btc_preopen/btc_market_base70_frozen_holdout_20261019.md), local reconciliation and run manifests.
Result: development M1-M0 LL -0.000492, interval [-0.000778,-0.000209]. Historical transfer improves AUC, but LL/Brier intervals include zero. Decomposition separates target mismatch, execution and realized noise; negative realization alone does not prove miscalibration.
Conclusion: limited conditional development information; no general stable economic advantage.
Reusable evidence: PARTIAL, compare each exact sample/procedure; independent and funded PnL remain separate.
Known limitations: hypothesis followed holdout exposure; RAW OOF global feature selection and availability provenance unresolved for stricter dynamic claim.
Superseded by: no replacement; corrected gross-notional/cash-debit wording governs these ledgers.

| Permanent variant ID | Name / hypothesis | Status | Code commit | Specific result / reusable evidence |
|---|---|---|---|---|
| E0022a | Exact EV/execution/outcome/target-switch attribution | CLOSED_INCONCLUSIVE | `37831fc` | Reconciles contributions, not a stable causal explanation; PARTIAL |
| E0022b | Daily development RAW information beyond market M0 | CLOSED_POSITIVE | `190a88a` | Conditional M1-M0 LL interval below zero; PARTIAL, reused selected development |
| E0022c | Once-fitted frozen M1 information transfers historically | CLOSED_INCONCLUSIVE | `d7d81c5` | AUC gain, LL/Brier intervals cross zero; PARTIAL, exposed holdout |

Data, inputs, protocol and output paths for these variants are the three separately linked reports above and their same-named config/local directories; all share only the stated parent limitations. None supersedes the other's different estimand.

## E0023 - Microstructure design/development, empty partitions and capacity

Status: CLOSED_INCONCLUSIVE (family synthesis; exact variant statuses below).
Hypothesis: (a) BTC/microstructure interactions add information; (b) missing June inputs have identifiable provenance; (c) changing BASE70 capacity improves frozen quality.
Code commit: `1810b37`, `dcf64df`, `8313c3a` (E0023a); `0979185`, `22bfed6` (E0023b partition audit); `b118f27` (E0023c capacity).
Data: development snapshots/events and original RAW predictions; empty partition source manifests; reused 70-feature/fold cache for capacity.
Protocol / inputs: static/interaction models and common masks; distinguish no markets/no available inputs/no execution; bounded 3-seed capacity search, no holdout.
Output artifacts: [design](reports/btc_preopen/btc_market_microstructure_design_20261019.md), [development](reports/btc_preopen/btc_market_microstructure_development_20261019.md), [partition provenance](reports/btc_preopen/btc_empty_partition_provenance_20261019.md), [capacity](reports/btc_preopen/btc_base70_capacity_tuning_20261019.md), separate local manifests.
Result: X-C LL +0.012049, interval [0.002681,0.025441] (worse); C/X/MARKET have no fills on 5,083 decision rows, RAW 782. Sixteen days have markets but no available inputs. Capacity completed 17 trials, selected original control trial 0.
Conclusion: no promoted interaction/capacity replacement; unverified empty input does not prove unchanged book.
Reusable evidence: PARTIAL, negative conditional model result and explicit coverage/source accounting.
Known limitations: source holes and unresolved RAW information-time proof; capacity stopping/budget differs from unbounded tuning.
Superseded by: corrected diagnostics, then E0024 for dynamic feasibility; no fabricated fill result from empty inputs.

| Permanent variant ID | Name / hypothesis | Status | Code commit | Specific result / reusable evidence |
|---|---|---|---|---|
| E0023a | Microstructure interactions add BTC information | CLOSED_NEGATIVE | `1810b37`, `dcf64df`, `8313c3a` | Worse paired quality; PARTIAL conditional model evidence |
| E0023b | June empty-partition source diagnosis | CLOSED_INCONCLUSIVE | `0979185`, `22bfed6` | Data availability separated from zero trading; complete source truth remains unresolved; PARTIAL diagnosis |
| E0023c | Bounded capacity search beats BASE70 control | CLOSED_INCONCLUSIVE | `b118f27` | Original control selected, no supported improvement; PARTIAL selection evidence |

Each variant's named report/config/local directory above is its output/input manifest authority, with the shared scope limitations retained. E0024 supersedes only further dynamic feasibility assumptions, not these completed conditional comparisons.

## E0024 - Dynamic history and action-value feasibility

Status: BLOCKED (closed blocked attempt, not running).
Hypothesis: dynamic H adds prediction/action-value information beyond static S with executable size targets.
Code commit: `3d95f64`; reset correction `3760107`, integrity `25a013b`, RAW availability `80020c1`; closure `11927f3`.
Data: development-only event replay; S=10, H=21 features; q={$5,$10,$25,$50,$100}; unchanged two evaluation folds.
Protocol / inputs: provenance-qualified T60/T59, complete native history and released labels; frozen >=2,000 evaluation markets, >=20 days, >=1,000 two-sided $5 training per fold.
Output artifacts: [initial report](reports/btc_preopen/btc_market_dynamics_action_value_20261020.md), [audit](reports/btc_preopen/btc_dynamic_history_audit_20261020.md), [integrity](reports/btc_preopen/btc_dynamic_history_integrity_20261020.md), [final feasibility](reports/btc_preopen/btc_dynamic_history_feasibility_20261020.md), local `final_prefit_gate.json` and blocked run manifest.
Result: original count 633; independent recovery adds 4 -> 637/2,000 across 22 days; training two-sided counts 908/1,207. RAW provenance adds a separate fail-closed availability condition. Original 633/410 counts remain historical, not the final corrected gate.
Conclusion: no classifier/regressor fit, no economic/portfolio/bootstrap/promotion result for this study. Original RAW timing assumption is INVALID as proof of actual causal availability.
Reusable evidence: YES for failed gate/source diagnosis, PARTIAL for synthetic mechanics; NO for model advantage.
Known limitations: missing genuine pre-initialization history, ambiguous/stale events, unproven RAW prediction readiness; cannot invent it from model identity or nominal row time.
Superseded by: corrected final gate replaces original counts; separate A/B/C/D saved-score analysis is not blocked by this distinct model-fit gate.

## E0025 - Initial ESN portfolio and NVAR comparison

Status: CLOSED_INCONCLUSIVE.
Hypothesis: reservoir portfolio or NVAR adds to BASE70 on later development.
Code commit: `855958a`; final successful local revision r3, earlier roots retained.
Data: fit core before 2024-07-01, correction/search through 2025-07-01, confirmation through 2026-07-01; 105,113 common confirmation decisions.
Protocol / inputs: seed-matched BASE70, selected ESN portfolio and NVAR, paired seven-day bootstrap; no independent full-pipeline selection claim.
Output artifacts: [report](reports/btc_preopen/btc_esn_portfolio_20261020.md), local r3 `run_state.json`, `final_comparison.json`; benchmark/manifest.
Result: ESN A-B LL benefit +0.000022, 95% [-0.000024,+0.000069]; NVAR -0.000032, interval also crosses zero.
Conclusion: no resolved incremental confirmation result; confirms confirmation was exposed before E0030.
Reusable evidence: PARTIAL, selected-BASE70 conditional comparison.
Known limitations: baseline generator/feature tuning used full development; no trading-profit claim.
Superseded by: later ESN search/freeze experiments; r1/r2 are superseded attempts.

## E0026 - ESN architecture audit and corrected input groups

Status: CLOSED_INCONCLUSIVE (independent advantage); search completed.
Hypothesis: distinguish architecture/input/probe selection from downstream LightGBM performance.
Code commit: `8a55169`, `77be163`; corrected local r4.
Data: same reused search-validation through 2025-07-01; correction-fit and fit-core as frozen.
Protocol / inputs: architecture/topology anchors, probe selection, downstream seed grids, corrected input-group stage; no new confirmation/holdout score.
Output artifacts: [audit report](reports/btc_preopen/btc_esn_architecture_audit_20261020.md), [initial trials](reports/btc_preopen/btc_esn_architecture_audit_20261020/initial_48_trials.json), local corrected/uncorrected JSONs and r4 run state.
Result: best selected downstream single/ring 1x128 delta LL +0.000078732; probe portfolio +0.000158274 probe but -0.000046807 downstream. Runner did not switch selection to downstream single winner.
Conclusion: probe winner is not downstream portfolio superiority; corrected input stage supersedes uncorrected conclusions.
Reusable evidence: PARTIAL for search diagnostics, NO for independent portfolio advantage.
Known limitations: multiple search/reused validation; only selected candidates receive full downstream evaluation.
Superseded by: corrected r4 input results and subsequent E0027/E0028; raw attempts retained.

## E0027 - Expanded ESN search

Status: CLOSED_INCONCLUSIVE (independent advantage); development search completed.
Hypothesis: wider reservoir exploration and downstream selection improve baseline/search choices.
Code commit: `253e3a8`; resume fixes `d78189f`; report `94bf228`.
Data: frozen development search cache only through 2025-07-01.
Protocol / inputs: 300 valid screens, 54 rejected proposals, 60 downstream candidates including 18 probe-independent, 5 finalists on 3x3 seed grid; recorded exploratory-stage budget amendment.
Output artifacts: [report](reports/btc_preopen/btc_esn_expansion_20261021.md), [budget amendment](reports/btc_preopen/btc_esn_expansion_20261021_budget_amendment.md), local frozen manifest/final comparison/runtime ledger.
Result: best_single delta LL +0.000104617, positive 8/9 seeds; tested pairs below single; active time 18,487.5 s including 7 resumes.
Conclusion: completed development selection, not confirmation.
Reusable evidence: PARTIAL, search identity and seed results; isolated tests currently 8/8 pass.
Known limitations: reused validation and amended stage cap; full-suite global-config isolation fails.
Superseded by: E0028 for focused exploration; no erasure of search exposure.

## E0028 - Focused ESN and simple-memory controls

Status: CLOSED_INCONCLUSIVE (independent gain); development search completed.
Hypothesis: focused sparse/512 neighborhood beats prior single/ring and simple causal-history control.
Code commit: `95cc942`; orchestration recovery `0ed9015` and `execution_hotfix.json`.
Data: same pre-July-2025 search-validation cache; no fresh sample.
Protocol / inputs: 120 proposals, 30 preliminary downstream, 9 finalists on full 3x3 grid; PCA4 and memory PCA8 controls; optional pair only if frozen trigger.
Output artifacts: [protocol](reports/btc_preopen/btc_esn_focused_20261022_protocol.md), [report](reports/btc_preopen/btc_esn_focused_20261022.md), local `run_summary.json` and execution hotfix/runtime ledger.
Result: 114 valid/120; simple_memory_PCA8 +0.000136588 (3/3) exceeds best local ESN +0.000101431 (9/9); pair trigger not met; 10,404.1 active seconds.
Conclusion: development leader/control identified, no independent result. Static `run_manifest` says frozen_before_screening while completed summary/downstream outputs establish later completion.
Reusable evidence: PARTIAL, development comparisons and source identity.
Known limitations: repeated search-validation and preserved execution fix history.
Superseded by: E0029 freezes procedures without retroactively making this search independent.

## E0029 - A/B/C/D quality freeze and technical bundle validation

Status: CLOSED_INCONCLUSIVE (incremental ESN beyond memory); technical freeze completed.
Hypothesis: ESN adds to memory on a frozen common BASE70 sample; bundles reproduce their selected representations.
Code commit: `9dab58a`; source alignment `8f36b71`, `176230a`, bootstrap labels `f9a372a`; recorded final runtime source `f9a372a0a33fd2f9a873c83f5eeee0f657e0f926`.
Data: reused search-validation 52,120 decisions; 2,660,546-row development raw stream; fit-core transforms only.
Protocol / inputs: D frozen before fitting, uniform ensembles with 24 members, paired LL/Brier/AUC and bundle export/stream/checkpoint/future-perturbation checks.
Output artifacts: [report](reports/btc_preopen/btc_candidate_quality_freeze_20261022.md), freeze config; local `candidate_comparison.json`, `stream_verification.json`, `runtime.json`, bundles.
Result: A/B/C/D LL 0.692375219 / 0.692210070 / 0.692255246 / 0.692135666. D-B benefit +0.000074403, 95% [-0.000003351,+0.000156894]. Saved full memory/ESN replay matches exactly on respective valid rows.
Conclusion: technical parity evidence is limited to frozen representation/bundle identity; no independent ESN benefit beyond memory established.
Reusable evidence: PARTIAL; local result/verification files independently present/hashed; small mechanics covered in current tests, full replay not repeated.
Known limitations: reused search set, local ignored bundles; candidate freeze is not live readiness.
Superseded by: E0030 adds exposed retrospective scoring, not a replacement of the immutable freeze.

## E0030 - Frozen historical A/B/C/D scoring

Status: CLOSED_INCONCLUSIVE.
Hypothesis: B-A and D-B predictive improvements persist in confirmation/holdout.
Code commit: scoring `cf572c495e683860cd8ae52764a4b2deb4231257`; finalizer fix `262b7d0`; report `831e518`.
Data: confirmation [2025-07-01,2026-07-01), holdout [2026-07-01,2026-10-07), both historically exposed; 98 completed USD-M daily partitions supplement missing flow.
Protocol / inputs: authorized frozen scorer, pre-read manifest, same 24 models/transforms, common masks, 2,000 paired non-circular 3-day blocks, adjusted primary comparisons; no refit.
Output artifacts: [report](reports/btc_preopen/btc_candidate_holdout_20261022.md), [development replay](reports/btc_preopen/btc_candidate_holdout_20261022_development_replay.json); local run `ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b`, pre-read, attempts, source and period checkpoints.
Result: 105,117 confirmation and 28,223 holdout common rows. B-A LL benefit +0.000065819/+0.000058329; D-B +0.000068634/+0.000032457; all adjusted primary intervals include zero. Source completion records 141,113 minutes with no missing/invalid rows.
Conclusion: technical scoring complete, statistical superiority unresolved. Failed schema/timestamp/finalizer attempts were recovered and retained, not hidden.
Reusable evidence: PARTIAL, saved conditional historical scoring; no new scoring/raw-row access in audit.
Known limitations: previous exposure, tiny effects, varying monthly signs; source checksums/candidate predictions not recomputed here.
Superseded by: E0031/E0033 add other target/regime views, not independent new samples.

## E0031 - Official Gamma target and sparse economics

Status: CLOSED_INCONCLUSIVE.
Hypothesis: frozen candidate gains survive official outcomes and executable historical subset.
Code commit: `c42c86e`; v3 protocol/runner `run_btc_candidate_polymarket_target_20261023.py`.
Data: 133,340 saved predictions; 79,678 exact Gamma matches, 79,650 official valid outcomes; provenance-qualified T60/T59 intersection 1,143 confirmation markets, zero holdout.
Protocol / inputs: frozen probabilities, exact market/token/outcome joins, no refit/replay, fixed-$5 independent and separate $100 portfolios.
Output artifacts: [report](reports/btc_preopen/btc_candidate_polymarket_target_20261023.md), [v3 manifest](reports/btc_preopen/btc_candidate_polymarket_target_20261023_v3_manifest.json), local completed run/quality/economic summaries.
Result: all adjusted primary Gamma intervals include zero. Sparse independent PnL A -$108.80, B +$28.01, C +$20.42, D +$22.66; funded A ends $4.58 after 139 insufficient-cash rejections; B/C/D end $128.01/$120.42/$122.66.
Conclusion: sparse retrospective simulation; no holdout economic or current TWAP60 execution result.
Reusable evidence: PARTIAL, sample-specific quality/accounting; not production/availability validation.
Known limitations: historical T-60 probability-readiness timestamps absent; native T60 only best ask/size; assumes saved probabilities existed at deadline. Not full-depth continuous live behavior.
Superseded by: E0032 explains contributions; E0033 qualifies settlement regimes.

## E0032 - Frozen target and economic attribution

Status: CLOSED_INCONCLUSIVE (price cause / stable economic explanation); accounting reconstruction completed.
Hypothesis: distinguish label implementation/join mistakes, official-target disagreement and changed trading decisions.
Code commit: `cfb5490`; correction `bd5ce99`.
Data: E0030 predictions and all five OHLC target candles; E0031 saved 1,143-row economics, no new replay.
Protocol / inputs: exact Binance target reconstruction, Gamma identity/orientation, decomposition of losses/decision participation/fees/cash paths.
Output artifacts: [report](reports/btc_preopen/btc_candidate_attribution_20261023.md), [protocol](reports/btc_preopen/btc_candidate_attribution_20261023_protocol.json), local `validation.json`, `analysis_summary.json`.
Result: saved check records all 133,340 proxy targets match; 1,980/51,430 confirmation and 2,213/28,220 holdout disagree with Gamma. D vs B differs on 86 decisions, no side reversals; net -$5.35, accounting identities pass.
Conclusion: no detected target/join implementation error under those checks; 4,193 disagreements have unresolved Chainlink price-level cause. A different target and participation explain components, not a causal general profit theory.
Reusable evidence: PARTIAL; saved validation and accounting record independently inspected/hashed, OHLC reconstruction not repeated.
Known limitations: no underlying official TWAP prices/reports; later aggregate direction correction and rule-scope correction must be retained.
Superseded by: E0033 for regime interpretation and corrected direction totals, not deletion of underlying validated rows.

## E0033 - Settlement-rule disagreement and regime candidate scores

Status: CLOSED_INCONCLUSIVE; exact Chainlink price reconstruction BLOCKED.
Hypothesis: historical settlement rule explains target disagreement patterns and candidate quality changes.
Code commit: `bd5ce99` (E0033a disagreement/rule correction), `4627442` (E0033b saved-score extension).
Data: 79,650 official common rows; seven directly inspected market rule pages; deterministic 43-market sample; regime counts 62,084 single-price / 2,016 TWAP30 / 15,550 TWAP60.
Protocol / inputs: calendar boundaries August 7 and August 14 UTC; common A/B/C/D masks and saved labels; paired circular 3-day bootstrap by group, no cross-group blocks; no Chainlink retrieval.
Output artifacts: [disagreements](reports/btc_preopen/btc_discrepancy_regimes_20261023.md), [43-market sample](reports/btc_preopen/btc_discrepancy_regimes_20261023_sample_manifest.csv), [candidate regimes](reports/btc_preopen/btc_candidate_settlement_regimes_20261023.md), local analysis manifests and table outputs.
Result: disagreement 3.63%/8.68%/11.33%; exact cause unresolved. TWAP60 D-B LL benefit +0.000020, interval [-0.000100,+0.000141]; zero execution-ledger rows for either TWAP regime. Corrected disagreement directions: confirmation 986/994, holdout 1,090/1,123.
Conclusion: regime association and target-shift decomposition, not causal rule effect or profitable ESN gain.
Reusable evidence: PARTIAL, saved official-label metrics and seven-page boundary support; no direct per-market rule confirmation for remaining 79,643.
Known limitations: time/regime confounding; TWAP30 only 7 days / 2 nonoverlapping blocks, TWAP60 54 / 18; exact feed IDs/access/history/cost unconfirmed.
Superseded by: E0034 replaces arbitrary 180-day next-step recommendation with registered precision scenarios; historical scores retained.

## E0034 - A/B/C/D no-order shadow and L2 source pilot

Status: BLOCKED.
Hypothesis: collect timely frozen candidate probabilities, T60/T59 depth and official outcomes, then assess new-sample economics.
Code commit: `cf885b0428e0c6041d471e6bca97597ff04dc4ec`.
Data: local shadow SQLite, source-review captures and one October 22 Vike public L2 sample outside the historical TWAP60 evaluation range.
Protocol / inputs: [frozen config](configs/research/btc_candidate_shadow_20261023.json), `run_btc_candidate_shadow_20261023.py`; public GET/WebSocket only, no authenticated trading client; actual deadline/latency and same-market source evidence; D0010 stopping/storage rules.
Output artifacts: [source/pilot report](reports/btc_preopen/btc_candidate_shadow_sources_and_pilot_20261023.md), local `shadow.sqlite3` preflight events.
Result: both saved Polymarket probes fail DNS, Binance time succeeds; audit independently counts 2 preflight events and zero all market/data/checkpoint tables. Pilot 0/3, no live latency or growth measurement. Public Vike sample demonstrates limited reconstruction feasibility, not target-period coverage.
Conclusion: implemented collector, blocked real pilot; **no live performance result**. Historical local PMXT ends August 10, before TWAP60 August 14, so empty TWAP60 economic ledger reflects missing raw history rather than a proven selector defect.
Reusable evidence: YES for zero local pilot and recorded preflight failures; PARTIAL for source/sample feasibility; NO for live availability/profitability.
Known limitations: current connectivity not reprobed; 101.9 GB archive plan and free/paid access claims are dated provider review, not audit-confirmed current offers. No accounts, purchases, source downloads or background collectors started. Local process enumeration found other Python processes but their work was not terminated or attributed.
Known timing deviation: shadow readiness must be <=T-60 (`market_timing`, `_predict_market`), even though its input candle closes nominally at T-60; generic pre-open frame permits prediction until T-59. Existing test deliberately marks T-60+250 ms late. Inference: positive receipt/compute delay could make otherwise one-second-budget-compliant predictions ineligible. This is an unresolved protocol/feasibility conflict, not an observed live failure; T0007 precedes pilot validation.
Superseded by: none. Next technical step is T0004; historical source prerequisite is T0005, without changing frozen models/policy.
