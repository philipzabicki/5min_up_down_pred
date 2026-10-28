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
Known limitations: a bounded same-interpreter probe on 2026-10-25 reached Binance time (HTTP 200; valid `serverTime` JSON) but Gamma and CLOB DNS resolution failed. No accounts, purchases, source downloads or collector runs were started. The historical 101.9 GB Vike archive estimate and free/paid access claims remain from the 2026-10-23 provider review, not current offers.
T0007 code correction on 2026-10-25: T-60 is the decision-information cutoff and T-59 the prediction-ready/execution deadline. The candidate collector records both Binance kline close events and actual local UTC receipts, uses monotonic durations, and rejects post-deadline REST book observations for pricing. Boundary/late/missing/idempotency tests pass. No real market timing was observed because the endpoint preflight failed; no pilot window was scheduled. Read-only shadow DB still has two historical preflight events and zero market/candle/snapshot/prediction/attempt/outcome/checkpoint rows. Pilot result remains 0/3; no actual fill or live-latency claim.
Historical-source reinspection on 2026-10-25 found no new artifacts: PMXT still ends before TWAP60; Vike has a target-family/date hole and requires a free API key for the estimated 101.88 GB mixed-layout acquisition; PMData's reported quota and access requirements remain unconfirmed against an account; Marketlens exact Pro price/target-period sample remain unknown; public CLOB exposes current books, not historical full depth. RAW OOF still has 318,560/318,560 missing `prediction_available_at_utc`; the 637/2,000 and 908/1,000 counts remain auxiliary pre-RAW gates, not eligible observations. No historical data were acquired and no gate was relaxed.
2026-10-26 disposition under D0017: preserve the blocked status, 0/3 pilot, and all prior source findings. The former recommendation to make T0004 the next step is superseded; no DNS diagnostics, collector, or pilot is in the active queue. T0005 remains a historical source-feasibility blocker; no acquisition is planned.

## E0035 - Sale-based historical strategy gate and open-position accounting

Status: BLOCKED HISTORICALLY (sale contract incomplete; sale study is now deferred/out-of-scope under D0018).
Hypothesis: existing frozen A/B/C/D evidence and local quote sources can support a historical strategy assessment using executed simulated sales, fees, and a self-financing portfolio.
Code commit: the E0035 implementation commit containing `run_btc_historical_policy_study.py` and its regression test.
Data: existing E0006/E0021/E0029-E0034 reports and manifests, local frozen candidate manifests/prediction checksums, and small synthetic unit fixtures only. No raw model rows, holdout rows, full quote archive, or historical replay were opened/run.
Protocol / inputs: D0016 sale-only outcome rule; D0001/D0003/D0004/D0006/D0008/D0011 preserved. No exit policy was inferred. No training, tuning, replay, holdout evaluation, or source acquisition.
Output artifacts: [sale-strategy gate report](reports/btc_preopen/btc_historical_sale_strategy_gate_20261026.md), [machine-readable source/gate inventory](reports/btc_preopen/btc_historical_sale_source_gaps_20261026.json), updated contract and registers.
Result: current A/B/C/D diagnostics contain buys with resolution-based outcomes, not completed sale-based trades. The present active exit trigger/redecision schedule and maximum holding horizon are unspecified, so exact current per-market required exit intervals and a development economic result cannot be produced without inventing policy. The PMXT empty-bid initialization is already covered by a passing regression. A confirmed no-fill path previously converted an unfilled time sale into a settlement payout; a synthetic case reproduced it, and the simulator now keeps the purchase, entry fee, shares, and locked cost open with no PnL. The focused replay/accounting modules passed 30/30 tests; full `python -m unittest discover -s tests` passed 680/680 in 40.247 seconds (exit 0).
Conclusion: no development or holdout economic result was produced. This is a contract blocker, not an estimate or negative finding about strategy profitability.
Reusable evidence: current entry snapshot counts and hashes, existing source coverage diagnostics, PMXT initialization regression, and the small open-position accounting regression. Historical candidate labels/scores remain as previously recorded and are not reinterpreted.
Known limitations: saved prediction-ready timestamps are absent; the full BASE70 pipeline's independent fold-local selection is not established; current exit source coverage is not countable without a frozen sale horizon; historical PMXT does not include TWAP60; no outcome from a sale is calculated. No holdout metric was generated.
Disposition under D0018: the missing sale trigger and maximum horizon do not block active buy-and-hold-to-resolution economics. T0008 is deferred. The no-fill sale correction remains a separate invariant; see E0036 for the active resolution ledger.

## E0036 - Sparse development economics for hold-to-official-resolution

Status: PARTIAL (available confirmation subset calculated; full-development coverage and payout timing remain incomplete).
Hypothesis: the frozen BASE70/memory/ESN/memory+ESN candidates can produce positive executable, self-financing economics when positions are held to official Gamma resolution.
Protocol: [frozen E0036 protocol](configs/research/btc_resolution_development_20261026.json), SHA-256 `6e5812ca9fee860313c2e64fdd4a39111a03df55a2da37134df86c17fad04ed6`. D0001/D0003/D0004/D0006/D0008/D0012/D0018; no holdout rows or outcomes used.
Candidate identity: source E0031 run `ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b`; procedure-manifest SHA-256 A `28368afa62f824b5d2592eb61f87cb7a48ba8b66d0afc0a97ccdf6f334c9f1ef`, B `66eddd315dc38326f6f64959cd4a4fd8d053a6cd189ef7cceac5b3c35bd6a567`, C `f5ca0f29908e473b7e67be344c9430da0163cfd80f29d8bba9ce3931985f4d45`, D `c4910eb9d4fff03680fe36242c6d338abb2dcbc422803e94bd15c700869e6119`. The full 24 member-model and transform hashes remain in the E0031 v3 manifest.
Data: hash-verified E0031 `economic_decision_ledger.parquet` (SHA-256 `c0432980574e91ecd87529c8367e89f1b08c0cc119b81c345bb7de5acc8fc1a4`), 1,143 common confirmation markets from 2026-04-15 17:35 through 2026-05-18 09:00 UTC. T-60 saved snapshots cover 9,407 markets and T-59 provenance-qualified two-sided full-$5 books cover 5,609; their official-label/T-60/T-59 economic intersection is 1,143. This is 2.22% of the 51,430 confirmation official labels, not full-development coverage. No holdout exact T-60 economic rows exist.
Inputs: E0031 manifest SHA-256 `16e66363f35390d520e5a3ca7ce58fa07689ac15e66244a313c50f1a29017503`; T-60 source SHA-256 `1df26fe04d2e8d0f0fb9a3820184ce4a5b10c42383b8d424975dde44ef6a84b9`; T-59 ask-ladder SHA-256 `c90bb2364fff225b6e5047ff761f5ea150dda35afdd63a88ed0fef6f6bd6c941`; policy SHA-256 `baac4806dedfced36c548321cc7e50a628334c75bc54df436e29b0d99789961c`; fee config SHA-256 `b47cb594512b3671f67ab3cee3a673ff43b7dcd15f7514803e1bcde8238ec1e2`. T-60/T-59 Arrow reads filtered by the frozen confirmation IDs and development cutoff before materializing projected quote/quality columns. The T-59 source file spans into holdout; no holdout label/outcome columns were projected or used.
Code/output: runner `run_btc_resolution_development_20261026.py` SHA-256 `11fd99afa71a787cae135e1334416d9ccbc4c7bb4e0275edd1de17a361b28414`; settlement ledger `utils/btc_resolution_portfolio.py` SHA-256 `4af144f307db129f208687dd4ef7c38d002382b0b24048066772ef0d4964c9c1`; cached outputs `data/analysis/polymarket/BTC/btc_resolution_development_20261026_6e5812ca9fee/`, summary SHA-256 `56a1d8e137657a7af197e9286ed7f996debba8f99e7c98d411f136573000684e`. The runner reused saved E0031 decisions, T-60 snapshots and T-59 ask ladders; Arrow applied the frozen condition-ID, development-cutoff and entry-case filters before materializing T-60/T-59 quote fields. It did no model inference, training, quote refetch or full book replay. Runtime 0.40 seconds after Python startup.
Result — independent fixed-$5 trade diagnostics: A 424 trades, −$108.80; B 396, +$28.01; C 375, +$20.42; D 380, +$22.66. These per-trade sums ignore account funding limits.
Result — separate $100 fixed-$5 portfolios (accepted/rejected for cash; ending equity/PnL; fee equivalent; max drawdown; peak locked gross): A 285/139; $4.58/−$95.42; $50.49; 96.21%; $10. B 396/0; $128.01/+$28.01; $69.90; 55.99%; $10. C 375/0; $120.42/+$20.42; $66.19; 69.32%; $10. D 380/0; $122.66/+$22.66; $67.11; 72.53%; $10. All accepted positions received one payout and were settled by the development cutoff under the assumed release schedule; no cost remained locked. The cash ledger stayed nonnegative.
Result — separate $100 5%-of-free-cash portfolios, with the D0004 $5 minimum fallback: A 1 accepted, 1,137 below minimum, $95.00 ending equity/−$5.00; B 1, 1,137, $95.00/−$5.00; C 2, 1,121, $99.24/−$0.76; D 1, 1,136, $95.00/−$5.00. There were no T-60 size failures, T-59 no-fills or cash-overdraft rejections. Each account stopped taking trades after free cash fell below the fallback minimum. Fees were $0 cash and $0.18/$0.18/$0.36/$0.18 equivalent for A/B/C/D. Peak locked gross was $5/$5/$5.22/$5; maximum cost-basis drawdown was 5% each. These sparse outcomes expose the interaction between 5%-of-free-cash sizing and the accepted historical minimum fallback.
Accounting: free cash plus locked gross cost basis; fees debited once; winning token receives net shares, losing token receives zero; one official-resolution payout closes each position. For each position the report/ledger distinguishes market end, Gamma `resolved_at_utc`, and assumed payout availability `max(market end, resolved_at_utc)+60s`. Actual redemption availability is not archived; the reused 60-second assumption is not a verified conservative bound. Sale simulation code remains separate and still leaves an unfilled sale open.
Conclusion: B/C/D are positive only on the existing sparse confirmation sample under the fixed-$5 control; this is not full-development validation or independent evidence. E0030/E0031 adjusted predictive intervals include zero. No economic PnL bootstrap was defined for this stateful portfolio path, so no trade-independence interval is claimed. Full development is blocked on entry-source coverage and causal provenance (T0009), not on sale rules.
Known limitations: the confirmation cohort was previously exposed; predictions lack historical per-row ready timestamps, so T-60 availability is assumed and not a live-readiness result; exact T-60 books cover only 2026-04-15 through 2026-05-18; actual payout availability is unknown; E0035/E0033 rule-source caveats and D0005 full-pipeline validation limits remain; previously negative results for other model families remain unchanged. No new holdout evaluation, model scoring/training, tuning, sale-policy research, live/shadow or orders were run.
Next step: T0009, establish full-development T-60/T-59 source coverage and causal cutoffs using existing local data before any additional economics.
Validation: `python -m unittest discover -s tests -p "test_btc_resolution_portfolio.py"` passed 3/3; `python -m unittest discover -s tests -p "test_btc_preopen_economic_replay.py"` passed 24/24; `python -m unittest discover -s tests` passed **683/683** in 51.332 seconds (exit 0).

## E0037 - T0009 development source reconstruction and resolution sensitivity

Status: PARTIAL / source-only coverage completed; E0036 T-60 parity failed, so economics remain blocked.
Objective: extend documented development entry-source coverage beyond E0036 while keeping candidate identities, entry mechanics, minimum fallback, accounting, and split boundaries fixed; audit saved prediction provenance and payout timing sensitivity.
Protocol: [frozen E0037 protocol](configs/research/btc_t0009_resolution_coverage_20261026.json). Decisions D0001/D0003/D0004/D0005/D0006/D0008/D0012/D0018 apply. E0036 remains the parity control and is not rewritten.
Scope: canonical development `[2025-07-01, 2026-07-01)` only. Use existing saved A/B/C/D probabilities without new inference. Preliminary identity audit found 105,117 confirmation rows, 53,660 without Gamma condition IDs, and 51,457 unique Gamma condition IDs with no duplicate IDs or missing A/B/C/D probabilities. Existing E0031/E0036 records list 51,430 official-label-valid markets; recount only as a coverage flag after the freeze. Holdout rows/files are excluded.
Frozen sources: Gamma market index; E0031 confirmation matches and manifest; E0036 ledger/T-60/T-59 artifacts; PMXT archive manifest and its 1,853 local development event parts (1,242,757,525 bytes; 152,378,189 source rows; part-list/content/schema fingerprint `3fe6416a8b6a4d1c511bc9c75c11c9bed2bfc52a3411d58b780304828b04622c`); Obadiaha March 1-19 sampled top-10 books; Kacho within-market ticks as a negative pre-open coverage control. Input hashes are recorded in the protocol.
Pre-result checks: one PMXT market at 2026-05-18 09:15 UTC selected as the earliest saved market after E0036's 09:00 maximum; one Obadiaha market at 2026-03-01 20:05 UTC selected as the earliest dual-native-book sample within 10 seconds at both cutoffs. Selection uses time/source only, not outcomes or PnL. Reconstruct both cutoffs, verify native token mapping, no-future checks, source quality/depth, compare the PMXT quote with saved rows, then reproduce E0036 before additional economics.
Frozen accounting: separate $100 fixed-$5 and 5%-of-free-cash accounts, no credit/deposits or added cap. Historical minimum value/unit remains UNKNOWN; preserve the $5 fallback and reject below-minimum sizes without rounding them up. Re-run the full chronological account path at `max(market end, Gamma resolved_at_utc) + {60,300,1800,7200}s`; actual redemption availability is not archived. PMXT exact replay and Obadiaha sampled proxy are separate cohorts. Missing-source opportunities remain unknown, not no-trade, and cannot be reported as full-development economics.
Provenance limit: E0031 per-row `prediction_available_at_utc` is absent and full-pipeline fold-local selection/calibration independence is not established (D0005). Do not invent timestamps. Unless saved lineage proves otherwise, economic output is retrospective/conditional, not independent validation. Preserve E0024's historical gate.
Limits: no holdout outcomes, paid/external acquisition, broad tuning/training, new entry-policy search, sale work, live/shadow/pilot/DNS/orders, or production claim. Output artifacts go to ignored `data/analysis`; tracked report/summary paths are fixed in the protocol. T0009 stays blocked if material coverage or provenance gaps remain.

## E0037 result record

Freeze checkpoint: no new source replay, sample result, or economic result had been produced when the protocol was frozen. The dated result below updates that checkpoint; preserve the frozen protocol and record any code repair, protocol deviation, and its effect separately.

2026-10-26 result — source-only, no outcomes/PnL read. The full local development PMXT replay covered 51,458 markets across 1,853 parts and applied 104,859,384 events. The verified complete replay took 4,942.4 seconds with six workers; the source-coverage refresh reused its verified snapshot. The row-level manifest contains 105,118 rows: 51,458 Gamma markets and 53,660 candidate rows without Gamma IDs. Of the Gamma rows, 51,430 have valid official labels, 5,688 have both native fixed-$5 T-60 BBO sides, 9,333 have at least one native fixed-$5 T-59 depth side, and 5,684 have labels, predictions, and joint T-60/T-59 source coverage. There are zero measured per-row prediction-ready timestamps. These counts are retrospective source coverage, not an economic cohort.

First-exclusion totals for Gamma rows: 45,743 lack usable two-sided native fixed-$5 T-60 BBO; 5,684 are source-covered under the retrospective assumption; 27 have invalid/unavailable official labels; 3 lack usable native T-59 depth; and 1 lacks a saved candidate row/probability. The additional 53,660 unmapped candidate rows are explicitly retained in the manifest with their month and no-Gamma-ID exclusion. Additional quality flags and monthly counts are in the [report](reports/btc_preopen/btc_t0009_resolution_coverage_20261026.md) and [summary](reports/btc_preopen/btc_t0009_resolution_coverage_20261026.json); row-level manifest SHA256 is `ea0228ad28dedc1f72f8f51eb35d4332a7e67973e0d1f80346c43b2340e09007` in ignored local data.

E0036 T-59 source parity passed (27,432 comparisons across 1,143 markets); T-60 parity failed (20,574 comparisons) on condition `0xe4d40fb41d0acedbcae8592f00ec3ac0f6e662f89260c11a65443385996537a3`, field `bbo_at_entry_ask_mismatches` (new 1, saved 0). The saved row had zero BBO checks and did not establish agreement. The raw local event stream retains an UP ask at 0.51 while processed BBO data report 0.52. The source does not identify whether an archive event is missing or explain the discrepancy. E0037's source-parity gate therefore blocked all economic results; the frozen 60/300/1800/7200-second payout sensitivities and monthly account paths were not run. E0036's prior economics remain unchanged.

The two deterministic source-only smoke checks passed and did not read outcomes/PnL. A minimum-order audit found 80,127 local Gamma rows with only a share-labeled current minimum of 5.0 and no capture timestamp. Under D0019, the T-59 helper now accepts only explicitly USD/USDC-labeled values and otherwise uses the frozen $5 fallback; the corrected helper was not part of E0037's pinned source hash and no portfolio run was performed with it. Current public documentation labels Gamma `orderMinSize` in USDC and the CLOB book `min_order_size` in shares; see [Market Details](https://docs.polymarket.com/market-data/market-details), [Place Orders](https://docs.polymarket.com/trading/place-orders), and [py-sdk issue #302](https://github.com/Polymarket/py-sdk/issues/302).

The candidate manifest and A/B/C/D probabilities were reused without inference. Candidate IDs/probabilities are present for 51,457 exact Gamma matches; all 53,660 rows without a Gamma ID have A/B/C/D probabilities but cannot be traded. Full-pipeline selection/calibration independence remains unproven under D0005; report predictions as retrospective/conditional. The payout delay is not a verified wallet availability time; Polymarket's resolution guidance describes an UMA challenge process ([resolution help](https://help.polymarket.com/en/articles/13364518-how-are-prediction-markets-resolved)).

Validation: focused T0009 tests 10/10, T-59 reassessment tests 34/34, and full `python -m unittest discover -s tests` 696/696 in 55.837 seconds. T0009 remains BLOCKED pending a source-backed resolution of the T-60 mismatch/parity failure and prediction-readiness provenance. Do not run payout or economic paths before the gate passes.

## E0038 — T0009 source-quality continuation

Status: completed for the source-quality-clean subset; full-development T0009 remains blocked. The saved E0037 gate remains **FAILED** and unchanged.
Protocol: [E0038 v3 correction](configs/research/btc_t0009_source_quality_continuation_20261026_v3.json), with frozen v2 at [v2](configs/research/btc_t0009_source_quality_continuation_20261026_v2.json) and initial v1 at [v1](configs/research/btc_t0009_source_quality_continuation_20261026.json); source checkpoint: [report](reports/btc_preopen/btc_t0009_source_quality_continuation_20261026.md) and [summary](reports/btc_preopen/btc_t0009_source_quality_continuation_20261026.json). This is a new versioned continuation, not a retroactive edit to E0037. Preserve E0036's previous economics.

Source-value parity across all 1,143 E0036 controls is 19,431/19,431 T−60 field comparisons and 26,289/26,289 T−59 comparisons with no mismatches. Checker coverage is reported separately: old T−60 check counts were 0 for 1,118 rows, 1 for 7, and 2 for 18; zero checks are `NOT_CHECKED`. The new checker evaluates both T−60 sides for all 1,143 rows and identifies one BBO conflict. T−59 has one partial checker row and 1,142 complete rows without mismatch.

The instructed market's cached and independent event replay both retain an UP $0.51 ask for 236.98 shares through T−60 while a later same-cutoff reported BBO is $0.52. The source conflict remains unresolved; no confirmed parser defect or missing archive event was established. That market is already outside the 5,684 base because two-sided native T−60 BBO coverage is unavailable. The symmetric rule excludes 80 of the 5,684 base markets: 62 T−59 equal-source-time ask ambiguities, 16 T−59 native ask/BBO mismatches, and 2 uncomparable ask BBOs; categories are disjoint. The remaining 5,604 are source-quality-clean, including 1,141 of E0036's controls. The two other E0036 rows lack a second native T−60 side. The 53,660 unmapped candidate calendar rows are not asserted missing markets or strategy no-trades.

Prediction provenance: 0 measured ready timestamps; nominal T−60 availability is only a simulation assumption, T−59 runtime readiness is `UNKNOWN`, and full-pipeline selection/calibration independence is not established under D0005. E0038 therefore supports only retrospective, coverage-conditional accounting and is not independent predictive confirmation. The frozen economics use unchanged saved A/B/C/D probabilities, T−60 decision/T−59 execution, fixed-$5 independent diagnostics, separate $100 fixed-$5 and 5%-of-free-cash accounts, the D0019 $5 fallback, and payout scenarios of 60/300/1,800/7,200 seconds from `max(market end, Gamma resolution)`. Actual redemption availability is unknown. At the pre-economics freeze checkpoint, no E0038 economic output existed; the protocol was committed and pushed first.

V1's runner stopped at its protocol commit guard before source/outcome loading. V2 (`ac4acc4`) was committed and pushed, then ran economics. Review found its E0036 decision mismatch counts were a reporting defect: missing current sides were represented as pandas `NaN`, not saved `no_trade`. A follow-up join mapping missing values to `no_trade` confirms zero side-decision differences for all candidates on the 1,141 common clean markets; probabilities and fill/fee values already matched. V3 changes only the comparator and reruns the unchanged frozen economics after its commit and push.

Validation: focused E0038 tests **13/13** and full discovery **709/709** pass after the v3 comparison correction (43.327 seconds, exit code 0); `git diff --check` passes. The runner checks pinned inputs/code and projects only development outcomes for the frozen clean ID set.

## E0038 result record

V3 protocol SHA256: 5852107c07e0bb1768fcfe9693322bd8b9b377963110424c8f09ac08e28d8008; committed/pushed as e500553 before the corrected replay. V3 completed in 31.339 seconds on 5,604 source-quality-clean markets spanning 2026-04-15 17:35 through 2026-06-30 23:35 UTC. No holdout rows or outcome columns were projected. Actual redemption availability remains UNKNOWN; delays are independent scenarios. E0037 remains FAILED at its saved gate, and E0036 prior economics remain unchanged.

Independent $5 diagnostic net PnL was A +$74.46, B +$221.83, C +$273.42, D +$285.90. Fixed-$5 $100 account net PnL by payout delay 60/300/1,800/7,200 seconds was A -$95.46/-$95.46/-$99.74/-$99.73; B +$221.83/+221.83/+221.83/+223.26; C +$273.42/+273.42/+273.42/+270.21; D +$285.90/+285.90/+285.90/+291.24. The free-cash 5% control made only 1-2 fills per model because sub-$5 requests were rejected without rounding; its PnL was -$5.00 for A/B/D and -$0.76 for C across delays. Detailed entries, fees, payouts, cash, drawdown, and monthly snapshots are in the E0038 report, tracked summary, and ignored local market/monthly ledgers.

The v2 and v3 economic controls, market-decision ledger, and monthly path are identical. V3 fixes only the E0036 decision comparison missing-value label: all probability, decision, gross, VWAP, net-share, debit, and fee differences are zero on the 1,141 common clean markets. End-of-run accounting checks passed; every account ended with zero locked cost basis and zero unsettled positions. The monthly ledger has 432 rows for the 12-month development calendar and available cohort months; zero-cohort months are not classified as strategy no-trades.

Coverage carry-forward: the exact 24-hour local PMXT gap list in the [T0009 coverage machine summary](reports/btc_preopen/btc_t0009_resolution_coverage_20261026.json) remains a separate deferred source task, [T0010](TASKS.md). It comprises 21 hours from 2026-04-13 19:00 through 2026-04-14 15:00 UTC and three hours from 2026-06-11 04:00 through 06:00 UTC. E0038 did not acquire data or alter its 5,684-market base / 80-market quarantine counts for this inventory.

Interpretation remains retrospective/conditional. No prediction-ready timestamps are available, T-59 runtime readiness is UNKNOWN, and full-pipeline selection/calibration independence is not established under D0005. No portfolio bootstrap was run. Positive subset results do not close T0009 or authorize holdout use, a pilot, or production.

## E0039 — BTC chronology and E0038 stability reconstruction

Status: E0039 v7 batch chronology/economics and the follow-on offline 300-window runtime benchmark completed. T0009 remains BLOCKED overall because source coverage and historical prediction-readiness provenance remain incomplete; runtime results do not establish production readiness.

Protocol: [E0039 v1 freeze](configs/research/btc_chronology_reconstruction_20261027.json), SHA-256 `6b901e753aa05ac5172fd753568638c4ff3880d3d6126b914c51db864866399f`; amended [E0039 v2 protocol](configs/research/btc_chronology_reconstruction_20261027_v2.json), SHA-256 `7b042861fd46af05d1a5f0de4ecc6aeba0f141f20f8b9aabe83dc313d271d220`. Reports: [chronology/stability report](reports/btc_preopen/btc_chronology_stability_reconstruction_20261027.md) and [v2 execution report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v2.md). Monthly flows: [108-row aggregate CSV](reports/btc_preopen/btc_chronology_monthly_flows_20261027.csv), SHA-256 `5cd323dcedb3f02b71bd1b5cde4dd8b8523879a3dacae11b997734c075f91209`. Account event snapshots: [96-row CSV](reports/btc_preopen/btc_chronology_account_snapshots_20261027.csv), SHA-256 `c4be0e3dbffc3ccdfa74ad9c9d0c8e28ee1ccdffc91605735f41544be24ccee2`. The preselected 300-window runtime sample is [here](reports/btc_preopen/btc_chronology_runtime_sample_20261027.csv), SHA-256 `a3f62ce585f17aa125e69c6494ab139425cfcfc2580007217914c98d8341b7b0`.

E0038 artifact hashes were reverified against its v3 summary. Its conditional clean cohort is 5,604 markets: April 456, May 2,315, June 2,833. Independent fixed-$5 net PnL remains A +$74.46, B +$221.83, C +$273.42, D +$285.90. June contributes more than the total A net because Apr/May are negative, and 74.7–90.0% of B/C/D net respectively; see the report for top/bottom days, paired B−A/D−B over all 5,604 common known markets, and the 16 fixed-account payout scenarios with event-time extremes/cost-basis drawdown. No CI or trade bootstrap was added. Monthly PnL is by market start; account snapshots are event-time cash/locked/equity and reconcile to the E0038 ledger.

Chronology plan: outer windows are `[2026-04-15, 2026-05-01)`, `[2026-05-01, 2026-06-01)`, and `[2026-06-01, 2026-07-01)` UTC, each fit cutoff at its start. Exact raw/flow prefixes, E0038 clean ID whitelist, candidate transform manifests, procedure budgets, and checkpoint order are frozen. Added `chronological_fit_mask` and used it at both pre-open training fit-mask sites. Focused regression run: 51/51 PASS, including exact exclusive label cutoff, append-future-row fit-prefix invariance, existing causal-feature future-mutation and memory/ESN restart tests.

Initial v1 block and current v2 disposition: the first freeze correctly identified that the generic `run_btc_preopen_experiment.run_training_pipeline` emits one model and Binance-proxy calibration, and does not compose fold-local BASE70+six Z/V, A/B/C/D memory/ESN variants and uniform ensembles, or earlier-only official Gamma OOF daily Platt calibration. The historic feature list, target weights and model settings could not be transferred. The v2 runner now implements that chronology and its protocol was pushed before invocation. The exact command `python run_btc_chronology_reconstruction_20261027.py` returned `BLOCKED_BEFORE_FIT` at the resource gate: 3.79 GiB RAM available versus 20 GiB required, while five other Python processes used 9.53-10.06 GiB each. Free disk was 105.97 GiB against an 80 GiB minimum; 6,927 MiB GPU memory was free. The versioned [gate record](reports/btc_preopen/btc_chronology_reconstruction_20261027_pre_fit_gate.json) says `data_or_label_reads_for_fit: false` and zero holdout rows. No development-only source-row hashes, fold fits, OOF calibration, outer predictions, economic replay, or runtime benchmark were produced. Resume with the exact command once the frozen gate passes.

Runtime axis is separately blocked: there is no offline per-decision builder for the exact ordered generated BASE70 features plus flow and dynamic states. The full-development feature cache is not used to disguise this missing hot-path work; no latency distribution or one-second conclusion is reported. The 300 samples are deterministic by timestamp and include the first/last available row per month; selection did not use PnL.

Validation: v1 chronology-mask focused checks passed 51/51 and full discovery passed 711/711 in 41.334 seconds. For v2, focused E0039 tests passed **17/17**, full `python -m unittest discover -s tests` passed **728/728** in **54.381 seconds** (exit 0), and `py_compile` passed. These validate code mechanics only; they do not validate fold reproduction, profitability, or runtime readiness.

Axis status at v2: T0009 source remains BLOCKED (E0038's subset is conditional, missing Gamma mapping/gap inventory and prediction-ready times remain); chronology execution was blocked before fit by the saved RAM gate; the offline streaming adapter remains `PARTIAL_NOT_IMPLEMENTED`, and full runtime is unmeasured; T0010 remains DEFERRED. E0037's saved FAILED gate is preserved. No new metrics, source acquisition, PMXT replay, live/shadow, DNS, or order operation occurred in E0039. Prior repository exposure remains as already recorded; an earlier audit turn also materialized the mixed confirmation/holdout matches file but derived no metric/choice from it. No holdout rows or metrics were newly used here.

2026-10-27 E0039 v2 retry and v3 protocol correction: the current resource gate passed (50.15 GiB available RAM, 105.91 GiB free disk, 6,206 MiB free GPU memory, no other Python process at the 1 GiB blocking threshold), and all 53 frozen file/implementation hashes verified. The no-argument runner then stopped at `E0038 clean ID allowlist hash changed`, before any fit, OOF, prediction, or economics. The v2 value `b92ed75e56a1d5fa438642590b4d13bf6d2745871cc66c606a910d564532d62e` is the SHA-256 of the complete E0038 cohort CSV; the runner expects SHA-256 of sorted lowercase condition IDs. The 5,604 IDs from the frozen quality manifest and CSV are the same set, with no duplicate CSV IDs. Their canonical ID hash is `e2946aa8e45c2ed050347e18756408a0175fa0eb9859a120f8df7c173051edda`.

Versioned [E0039 v3](configs/research/btc_chronology_reconstruction_20261027_v3.json), SHA-256 `d3a5dd3d916c117e627c95683865a66a3e0685bac7862d901307f92a77060047`, corrects the hash value and records v2 as its parent. It preserves the E0038 cohort, data sources, filters, boundaries, budgets, model procedure and accounting. V1/v2 remain unchanged. The runner now pins v3 and writes its execution record to the [v3 report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v3.md). A regression for canonical ID hashing passed 18/18 focused E0039 tests; full discovery passed 729/729 in 48.090 seconds, and `py_compile` passed. Independent identity verification passed for 53 pinned files and all 5,604 whitelist IDs. These validate implementation mechanics only.

No model fit data, labels, Gamma outcomes, economic source rows, confirmation rows, or holdout rows were read by the failed retry; only the clean-whitelist columns were loaded after the gate. No development-only projected source-row hashes, F1/F2/F3 results, official OOF calibration, outer predictions, economic replay, or runtime measurement exist yet. V3 must be committed and pushed before resuming the exact no-argument command. T0009 remains BLOCKED overall; the offline adapter remains PARTIAL_NOT_IMPLEMENTED, T0010 remains DEFERRED, and E0037's saved FAILED gate remains unchanged.

2026-10-27 E0039 v3 execution and v4 correction: after v3 was pushed, the retry passed the live resource gate (50.06 GiB available RAM, 105.91 GiB free disk, 6,453 MiB free VRAM, no other Python processes) and verified 53 frozen hashes. It read the frozen source projections to compute their row hashes, then failed before F1 when the official Gamma reader passed tuple-valued columns to this PyArrow version. The run manifest records zero folds, `confirmation_rows_materialized=0` (no confirmation evaluation output), and zero holdout rows materialized. The official Gamma input slice was read only for its frozen hash; no confirmation score or model choice was made. The four observed hashes are preserved in the ignored manifest and pinned in [E0039 v4](configs/research/btc_chronology_reconstruction_20261027_v4.json), SHA-256 `6821739db72dec443186f2fdb6b8c99960af4b5ad96310fc500e44bb54cf691e`. V4 converts the same ordered tuple to a list and verifies those hashes before fitting; no source, sample, filter, or research procedure changed. Its focused tests passed 19/19, full discovery 730/730 in 45.116 seconds, and `py_compile` passed. See the [v3 execution report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v3.md) and [v4 report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v4.md).

No F1/F2/F3 fit, OOF calibration, outer predictions, continuous economics, or runtime benchmark has run. The next step is to commit and push v4, repeat the gate, then resume the no-argument runner. T0009 remains BLOCKED overall; the offline adapter remains PARTIAL_NOT_IMPLEMENTED, T0010 remains DEFERRED, and E0037's saved FAILED gate is unchanged.

2026-10-27 E0039 v4 execution and v5 correction: the v4 runner passed its live resource gate (50.08 GiB available RAM, 105.90 GiB free disk, 6,414 MiB free VRAM) and recomputed all four pinned source hashes. It loaded the frozen Gamma calibration slice and the 5,604-market E0038 economic source snapshots, then stopped before F1 selection at `_start_runtime_ledger` with `TypeError: cannot use 'dict' as a set element (unhashable type: 'dict')`. The ignored manifest records zero completed folds, zero confirmation evaluation output, and zero holdout rows materialized. No confirmation score, model fit, OOF prediction, calibration result, outer prediction, or economic replay was produced. V5 removes the malformed set wrapper and adds a new-ledger regression. It keeps the same hashes and all research budgets/procedures. [E0039 v5](configs/research/btc_chronology_reconstruction_20261027_v5.json), SHA-256 `7ff752eaef17774725349d2f4a8d5cd7cb456ebecb575969bf8eb5c65279fcb9`, passed 20/20 focused tests, 731/731 full discovery in 66.208 seconds, and `py_compile`. See the [v4 attempt report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v4.md) and [v5 protocol report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v5.md).

No fold, OOF calibration, outer prediction, continuous economics, or offline runtime benchmark exists. Commit and push v5 before the next exact runner invocation. T0009 remains BLOCKED overall; the offline adapter remains PARTIAL_NOT_IMPLEMENTED, T0010 remains DEFERRED, and E0037's saved FAILED gate is unchanged.

2026-10-27 E0039 v5 execution: the exact runner passed its resource/source gates and completed F1 indicator selection (289.8 s), reaction-profile tuning (96 trials: 84 complete, 12 pruned), and volume-profile tuning (96/96). It failed during final feature dataset construction because `fold_20260415_20260501.csv` does not encode the symbol/interval needed for generator-state filenames. The wrapper recorded 12.346 GiB minimum available system RAM during the run; the pre-fit gate itself passed. No final fold dataset, predictive LightGBM fit, OOF/calibration result, outer prediction, economic result, or holdout row was produced. See the [v5 execution report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v5.md) and ignored [run manifest](data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027/runs/442a0de18fda1bfa/run_manifest.json).

Versioned [E0039 v6](configs/research/btc_chronology_reconstruction_20261027_v6.json), SHA-256 `2facc83e0b68ae0dc26874ec5dfd85b70559d1fa17a2ec4156744099bd580d3c`, records v5 as parent. It passes configured dataset `symbol`/`interval` to both profile-state resolvers as a fallback for fold-prefix filenames; existing filename parsing, frozen inputs, source hashes, fold boundaries, labels, trial budgets, model/selection/calibration procedures, and accounting are unchanged. The fallback regression tests passed 23/23, chronology tests passed 20/20, full discovery passed 733/733 in 40.668 seconds, and `py_compile` passed. Read-only identity verification passed for the v5 parent and all 53 pinned implementation/source/config paths. These checks validate code mechanics only.

V6 was committed and pushed before its exact run; the v6 execution evidence and v7 correction are recorded below.

2026-10-27 E0039 v6 execution: the exact runner passed its resource gate (51.01 GiB available RAM, 105.90 GiB free disk, 6,375 MiB free VRAM), verified the 5,604 clean-ID identity and all four frozen source projection hashes, completed F1 indicators (271.7 s), reaction-profile tuning (96 trials: 84 complete, 12 pruned), volume-profile tuning (96/96), and final dataset construction (190.7 s; 3,098,307 rows). The causal candidate feature join then failed because the base Parquet `Opened` column contains UTC-naive strings, which the join attempted to cast directly to `int64`. The ignored manifest records zero completed folds and zero holdout rows. No predictive fit, OOF/calibration result, outer prediction, or economic replay was produced. See the [v6 run report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v6.md) and ignored [run manifest](data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027/runs/eb07a734607052ac/run_manifest.json).

Versioned [E0039 v7](configs/research/btc_chronology_reconstruction_20261027_v7.json), SHA-256 `9c6aca98034ca010cda16e326053ce83e2c78dbab0120445be8ec1180392baf6`, records v6 as parent. The join parses canonical `%Y-%m-%d %H:%M:%S` strings to nanosecond candle timestamps while retaining typed Arrow timestamps, exact matching, ordering checks, and unmatched-row gaps. No data, cohort, fold, label, feature, budget, fitting, selection, calibration, economic, or accounting rule changed. The regression reproduced the Arrow error before the fix and passed 1/1 after it; chronology tests passed 20/20, project-config tests 23/23, full discovery 733/733 in 48.443 seconds, and `py_compile` passed. Read-only identity verification passed for the v6 parent and all 53 pinned implementation/source/config files; all three fold boundaries validate and holdout use remains forbidden. These checks validate mechanics only.

2026-10-28 E0039 v7 execution and offline runtime result: the exact batch command completed all three frozen chronology folds, official OOF calibration, outer predictions and the continuous coverage-conditional economics ledger. Run identity: `508858c48b11b9252e4ec3aa0e7814579f2fe770ef8840e32fd86dca6c6bd8d0`; protocol v7 SHA-256: `9c6aca98034ca010cda16e326053ce83e2c78dbab0120445be8ec1180392baf6`. All 5,604 allowlisted markets have raw predictions per model; 5,148/model have official calibration and 456/model remain `minimum_not_met`.

The standalone offline adapter used the frozen 300-window sample, each fold's full causal raw prefix and its fold-local saved model bundle. It computed selected BASE64, VP/RP, six Z/V flow features from locally archived Binance source rows, memory/ESN PCA8 updates, every A/B/C/D ensemble member, and the frozen daily calibration or no-fit fallback. Frozen batch caches were used only as parity references. Feature and raw/calibrated prediction differences stayed within the `2e-6` absolute tolerance; protected holdout and confirmation rows materialized: zero. The benchmark was offline-only and used no network or collector.

Full current decision path across 300 windows: p50 `11.478 ms`, p95 `15.324 ms`, p99 `22.661 ms`, max `112.736 ms`, `0.0%` above 1,000 ms. Per-fold state-update and cumulative update-between-window distributions, warmup times, parity maxima and bundle-reset evidence are in the [v7 execution/runtime report](reports/btc_preopen/btc_chronology_reconstruction_20261027_v7.md). The F3 sample has a largest adjacent-window gap of 27,205 minutes, so its maximum inter-window update total (`98.907 s`) covers that entire unsampled interval and is not a single decision-path latency. Historical T-59 readiness remains `UNKNOWN`; E0037's source-parity gate remains `FAILED`; T0009 remains BLOCKED overall and T0010 remains deferred.

Runtime-focused tests passed 3/3; full `python -m unittest discover -s tests` passed 736/736 in 46.215 seconds; `py_compile` and `git diff --check` passed. This completes the authorized E0039 chronology/runtime work without resolving broader source coverage or live-readiness provenance.

2026-10-28 E0039 saved-economics account/stability report: [report](reports/btc_preopen/btc_e0039_accounts_and_stability_20261028.md). Its report-only ledger reconstruction verifies the 32 model/control/delay accounts against `controls_results.terminal_account_after_all_payouts` and the 96 monthly snapshots against `economics/monthly_paths.csv`; it uses saved full fills, cash debits, payout values and payout-availability timestamps only. No signals, eligibility, fills, execution prices, or fees were recalculated. Cash debits were not charged fees a second time; accepted positions were each settled exactly once, with zero terminal open positions and no negative cash, deposits, or credit. The continuous 60-second chart is labeled as a report reconstruction. Core saved economics output hashes were checked against the existing run manifest; the source/run identity and derived artifact hashes are in the report package JSON.

On the common 5,148 calibrated markets, the independent fixed-$5 diagnostic reproduced A +$139.945271869044, B +$89.093840071001, C +$217.969791477041, D +$274.574785193205 and decomposed by May/June. The distinct $100 self-financing fixed-$5 account at 60 seconds ended at $4.940048 (A), $0.002942 (B), $1.136197 (C), and $374.574785 (D); all 32 terminal accounts and four payout-delay scenarios are in the CSV. Each 5%-of-free-cash account made one $5 fill, then its next nominal request was $4.741 and below the frozen minimum; the following 5,147 decisions were rejected for minimum size. The 456 April/F1 rows with `minimum_not_met` remain an untested economic interval, not zero PnL. These retrospective conditional results do not establish statistical support or resolve historical redemption availability. T0009 remains BLOCKED, E0037 FAILED, historical prediction readiness UNKNOWN, and T0010 DEFERRED; no holdout or confirmation rows were used.

Report package verification: focused accounting tests passed 3/3; `py_compile` passed for the report generator and its tests; all 9 report links resolve, all 5 derived output hashes match the package JSON, the aggregate CSV files contain 32/32/96/40 rows, and the SVG parses as XML. No full research run was performed.

## E0040 — E0039 settlement-delay follow-up at 1/5/15 seconds

Status: COMPLETED_LOCAL_RETROSPECTIVE_CONDITIONAL. E0040 ran only after its frozen protocol/code/tests and source-hash correction were committed and pushed.

Protocol: [E0040 frozen protocol](configs/research/btc_e0039_settlement_1_5_15s_20261028.json), SHA-256 `2f069cac9f58997860ed956c1b41850abc740c2c52e17d28688a5599292649ce`. It records D0020 and T0011, the exact E0039 run identity `508858c48b11b9252e4ec3aa0e7814579f2fe770ef8840e32fd86dca6c6bd8d0`, source v7 protocol, saved prediction/calibration files, 5,604-ID whitelist, Gamma and T-60/T-59 inputs, E0039 economic output hashes, and the prior report/account CSV. E0039's 60/300/1800/7200-second protocol and results remain unchanged.

Pre-replay provenance check: the first hash audit found a mistyped F2 `outer_calibrated.parquet` hash in the draft protocol. It was corrected to the on-disk SHA-256; a read-only comparison then matched saved fold probabilities, calibration statuses, and calibrator identities against E0039's combined predictions (22,416 rows, 5,604 raw/5,148 calibrated per model). No economic replay ran before this correction.

Scope: replay existing calibrated predictions for all 5,148 common markets per model with 24 continuous $100 portfolios: A/B/C/D × fixed $5 / 5% of free cash × `[1, 5, 15]` seconds. The 456 F1/April `minimum_not_met` markets remain uncovered economically. Payout availability stays `max(market_start_utc + 5 minutes, resolved_at_utc) + delay_seconds`, and delays are simulation assumptions rather than measured Polymarket timing. No feature, prediction, state, calibration, or market-evaluation timestamps are shifted.

Method changes were frozen separately: (1) the authorized delay list is `[1, 5, 15]`; (2) payouts available through T-59 are released immediately before cash coverage for an already-fixed T-60 purchase. T-60 free cash still fixes 5% sizing; a payout after T-59 cannot fund that buy. This corrects E0039's earlier T-60-only account-release event and is distinguished from the delay change in the report. Saved independent fixed-$5 trades match E0039 row-by-row and reproduce A `+$139.945271869044`, B `+$89.093840071001`, C `+$217.969791477041`, and D `+$274.574785193205`.

Results: [report](reports/btc_preopen/btc_e0039_settlement_1_5_15s_20261028.md), [machine summary](reports/btc_preopen/btc_e0039_settlement_1_5_15s_20261028.json), [24 accounts](reports/btc_preopen/btc_e0039_settlement_1_5_15s_accounts_20261028.csv), [monthly snapshots](reports/btc_preopen/btc_e0039_settlement_1_5_15s_monthly_snapshots_20261028.csv), [monthly cash flows](reports/btc_preopen/btc_e0039_settlement_1_5_15s_monthly_flows_20261028.csv), [independent-$5 parity](reports/btc_preopen/btc_e0039_settlement_1_5_15s_independent_fixed5_20261028.csv), and [equity chart](reports/btc_preopen/btc_e0039_settlement_1_5_15s_equity_20261028.svg). Run identity: `27634bcbcc1068ae0b4907d4ffa3e2deddb312a17089e16aeb2c4963a3b99522`; 24 accounts / 123,552 decisions; 26 frozen source/dependency hashes matched; replay elapsed 32.293 s. Output hash checks passed 12/12; 72 month-end and 24 terminal snapshots reconcile; all monthly net cash flows reconcile to account PnL; all accounts settled with nonnegative terminal cash.

All 1/5/15-second account outcomes matched one another and the historical 60-second terminal comparison. Fixed-$5 ending equity / net PnL / fills: A `$4.940048 / -$95.059952 / 189`; B `$0.002942 / -$99.997058 / 212`; C `$1.136197 / -$98.863803 / 210`; D `$374.574785 / +$274.574785 / 1,589`. Each 5%-of-free-cash account made one `$5` fill, ended at `$94.82`, and had 5,147 below-minimum rejections. Independent-$5 E0039 row-level differences are zero for all models; transaction signatures also match across all three delays.

Implementation freeze: [dedicated replay runner](run_btc_e0039_settlement_1_5_15s_20261028.py), SHA-256 `3a24f7ec3452536c2d2cde7ecd02ec67d9e3811ceb33c90cae22d3b9d35091c0`; [focused regressions](tests/test_btc_e0039_settlement_1_5_15s_20261028.py), SHA-256 `f071fdd5c7b62f0a71acca6965a9fcc0ae469442ac1701088972cc99db5ccd8e`. Focused mechanics tests passed 10/10; full `python -m unittest discover -s tests` passed 749/749 in 67.812 seconds; `py_compile` passed. Freeze commits `025e32e` and `2906e2a` were pushed before replay. The report package passed output and link checks. Result commit d2b5bf9 was pushed to origin/audit/btc-dynamic-history-20261020. No holdout/confirmation, training/selection/calibration, feature generation, raw-book replay, runtime measurement, network, live/shadow, or T0046 operation is part of this follow-up.

## E0041 - Fixed D holdout preparation (2026-10-28)

Status: `D_BUNDLE_BLOCKED_RESOURCE` on the v2 pre-fit gate. The v1 run remains historically `D_BUNDLE_BLOCKED_TUNING_BUDGET`; v2 is authorized under D0022. `evaluation_authorized=false`; `EVALUATION_CODE_READY`; no v2 bundle yet; `HOLDOUT_NOT_RUN`.

Current protocol: [E0041 v2](configs/research/btc_D_holdout_preparation_20261028_v2.json), associated with D0022/T0012. Historical protocol: [E0041 v1](configs/research/btc_D_holdout_preparation_20261028.json), associated with D0021. Fixed holdout starts `[2026-07-01T00:00:00Z,2026-10-07T00:00:00Z)`; first decision `2026-06-30T23:59:00Z`. Selection cutoff is `2025-10-01T00:00:00Z`; all training labels must be strictly available before their fit cutoff. Final candidate is D = BASE70 + memory PCA8 + ESN PCA8, three LightGBM seeds × three reservoir seeds, uniform mean. One terminal Platt calibration uses earlier D OOF predictions and only eligible official Gamma outcomes released by the first holdout decision; minimum 1,000 rows, both classes, no fallback.

Exposure and rationale: the fixed interval and A/B/C/D family were previously examined. This is not a virgin or fully independent test. Per the user's instruction, D is selected from development evidence; E0039/E0040's fixed-$5 D development account ended at `$374.574785` from `$100` with `64.656%` cost-basis drawdown. These development values do not describe the holdout.

Frozen future method: use the complete interval without subperiod selection; source-qualify symmetrically using `utils.btc_t0009_source_audit.source_quarantine_reasons`; run D only through the E0040 T-60 decision/T-59 full native-ask fill/accounting path; maintain six continuous independent `$100` accounts (fixed `$5` and 5% of free cash at 1/5/15 seconds); settle terminally after the last eligible market without post-window entries. The common positive criterion is fixed-$5 terminal equity strictly above `$100` in all three delays. This amount criterion does not imply statistical support or production readiness.

Preparation boundary: the preparation reader is limited to development and uses an unbuffered CSV stream, stopping after the first excluded `Opened` timestamp and before its OHLCV delimiter payload. The future evaluator checks the authorization field before opening the bundle or evaluation inputs. No holdout payload was read, no holdout predictions or economics were generated, and no source content validation was performed in this preparation. See the [preparation report](reports/btc_preopen/btc_D_holdout_preparation_20261028.md) and [metadata coverage table](reports/btc_preopen/btc_D_holdout_coverage_metadata_20261028.csv) for the resource result and exact source gaps.

Frozen preparation/evaluator/tests were committed as `cb05ee640d80e9471c97eafdbe4f39164c77aeca` and pushed before the gate attempt. The runner wrote only the local, Git-ignored `data/analysis/polymarket/BTC/btc_D_holdout_preparation_20261028/pre_fit_gate.json`, then stopped in 0.684785 s: free disk 69.64 GiB versus the frozen 80.0 GiB requirement; available RAM 47.73/63.67 GiB, no Python processes, RTX 4060 Laptop GPU detected. `run_identity_sha256=null`, `holdout_rows_read=0`, and `holdout_predictions_generated=0`. No selection, model fit, OOF, calibration, or bundle was produced. The first-attempt recommendation was to resume after the resource gate passed; the subsequent run recorded below passed that gate but exhausted the frozen cumulative tuning budget. Do not lower the threshold, remove/move files, or reset the tuning history.

The first-attempt report and canonical gate record were committed as `1e80fd6b0ab9bd13175420c8ca40e66e4f92c0e9` and pushed to `origin/audit/btc-dynamic-history-20261020`.

2026-10-28 storage recovery and preparation: four exact duplicate development Parquet files were replaced by SHA-256-verified hardlinks at their existing paths. Measured free-space gain was 19,934,871,552 B (18.5657 GiB). The unchanged 80 GiB resource gate passed at 88.1991 GiB free. Run identity `149ca92fac1ae8c5de3f14b74c30c8a26d9d82c8c1427f3fce08f84b43f49877` copied the safe development prefix (3,186,145 rows; SHA-256 `25481546299f72c159ffe0c0b4f7e6c2c94b914b52973f9a49a29eb13427829a`) and stopped before the excluded OHLCV payload. The reaction-profile stage used 2,704.5527 of 2,700 seconds and ended after 46/96 trials (44 complete, 2 pruned); the cumulative budget is exhausted and the resume sampler path is not guaranteed identical. No selection, final fit, OOF, Platt calibrator, or D bundle exists. Holdout rows materialized/read and predictions generated are zero. `evaluation_authorized=false`; `EVALUATION_CODE_READY`; `D_BUNDLE_BLOCKED_TUNING_BUDGET`; `HOLDOUT_NOT_RUN`. Focused tests passed 43 tests plus 52 subtests. At the v1 checkpoint, do not rerun/reset under v1; continuation required a separately versioned protocol and decision, now recorded in D0022. See the [storage recovery and preparation report](reports/btc_preopen/btc_D_holdout_preparation_20261028_storage_recovery_20261028.md).

Status vocabulary: `EVALUATION_CODE_READY` means implementation/tests passed while authorization remains false; `D_BUNDLE_READY` requires a fitted, hashed nine-member model and official calibrator; `D_BUNDLE_BLOCKED_RESOURCE` records an attempt stopped before data access at the unchanged machine-resource gate; `D_BUNDLE_BLOCKED_TUNING_BUDGET` describes the historical v1 attempt only; the current v2 state is `D_BUNDLE_BLOCKED_RESOURCE`; `DATA_PRESENT_BY_METADATA`, `DATA_BLOCKED`, and `DATA_NOT_VERIFIED` are separate; holdout remains `HOLDOUT_NOT_RUN`. E0037 remains FAILED and T0009 remains BLOCKED.

2026-10-28 budget-only continuation (D0022): E0041 v2 changes only the cumulative whole-preparation, selection-chain, and reaction-profile limits to 76,500, 52,200, and 7,200 seconds. VP stays at 2,700 seconds; the 96 terminal-trial target, five valid COMPLETE minimum, objective, target, features, cutoffs, searches, seeds, calibration and economics remain frozen. The v1 study has 46 terminal trials (44 COMPLETE, 2 PRUNED) and remains intact; v2 adopts a coherent SQLite backup and continues with 50 proposals. Prior RP use of 2,704.5527 seconds leaves 4,495.4473 seconds. Whole-preparation use includes v1 execution and each later active attempt, excludes idle gaps, and counts artifact migration once at the whole-run level. The sampler RNG state is not saved, so `sampler_path_guaranteed_identical_after_resume=false`; the configured TPE method continues from the existing trial history. COMPLETE, PRUNED, and FAIL count as terminal toward 96; FAIL and crash-penalty trials do not count among the five valid COMPLETE minimum, and pending/running states block completion.

The existing-study timing audit found 46 trials with a 13.639-second minimum, 19.815-second median, 33.231-second p90, 524.619-second maximum, 58.646-second mean, and 2,697.721 seconds summed trial duration. Four trials exceeded 400 seconds. The RP stage used 2,704.553 seconds; indicators used 400.144 seconds; the whole v1 preparation used 3,518.952 seconds and its selection chain 3,111.075 seconds. The pinned profile used one worker, 20 threads, and GPU LightGBM. Source review confirms the common input arrays and folds are prepared once per stage and passed into the objective; each trial still performs the parameter-dependent feature/objective work. No redundant full input reload was identified, and no timing-based objective/PnL selection or new trials were used in this audit.

The exact post-push command `python run_btc_D_holdout_preparation_20261028.py` first returned `blocked_by_resource_gate` in 1.165558 seconds. After ledger-fix commit `49c8ee0` was pushed, a second exact command returned the same status in 1.102481 seconds: 83.97 GiB free disk (80 GiB minimum), 46.27 GiB available RAM, 7,271 MiB free of 8,188 MiB GPU memory, and 20 logical CPUs. Active T0048 PID 49516 used 2.39 GiB; the unchanged gate blocked before input access. The gate record has `run_identity_sha256=null`, zero holdout rows read, and zero predictions. Both attempts (2.268039 seconds total) are preserved and will be charged once to the cumulative whole-preparation ledger after the gate passes; idle time is excluded. T0046/T0048 were not altered. Pre-fit validation passed: focused E0041 tests 20/20, full discovery 769/769 in 122.977 seconds, `py_compile`, v2 JSON and pinned-code-hash checks, and `git diff --check`. This is not reference validation or a real fit. At the pre-fit gate checkpoint, no final D fit had started. No holdout payload or predictions were accessed; `evaluation_authorized=false`, `HOLDOUT_NOT_RUN`; E0037 remains FAILED and T0009 remains BLOCKED.

2026-10-28 E0041 v2 execution after gate pass: run identity `189049366841f6d025e3d9b080bb69dd0e441bb5638e451708f27c0635754c44` passed the unchanged resource gate and completed the selection-only chain in 9,981.3369 s. RP finished at 96 terminal trials (83 COMPLETE, 13 PRUNED; the 46-trial v1 prefix was preserved and the five-COMPLETE minimum passed). VP completed in 1,357.4 s; target weights in 3,051.2 s; feature selection in 2,400.6 s (recommended K=256, best K=320); model tuning in 1,877.8 s (best trial 41, objective value 0.6922568800280187). The runner skipped general calibration and external prediction as specified for the selection-only path, then raised `KeyError: minimum_successful_trials` at preparation line 1150: saved `v1_artifact_migration.json` omitted the threshold carried in the frozen v2 protocol. The ignored runtime metadata was repaired using the frozen value 5. `v1_artifact_migration.repair_log.json` records original/corrected migration hashes and source protocol hash. No source code, protocol, search, or model setting changed.

An exact retry at 2026-10-28T18:46:54.898240Z returned `blocked_by_resource_gate` in 0.6277065 s because disk free was 73.84 GiB versus the unchanged 80 GiB requirement; available RAM was 51.4 GiB and GPU free memory 6,676 MiB. The invocation stopped before data access with `run_identity_sha256=null`, zero holdout rows read, and zero predictions. The latest gate time is pending one-time inclusion in the cumulative whole-preparation ledger upon a future pass; current recorded use is 13,511.5405/76,500 s, with 62,988.4595 s left before that pending gate entry. No final nine-member D fit, OOF, calibrator, or D bundle exists. `evaluation_authorized=false`; `HOLDOUT_NOT_RUN`. Free disk must return to at least 80 GiB before resuming; do not lower the gate or move/delete data/checkpoints. E0037 remains FAILED; T0009 remains BLOCKED.
