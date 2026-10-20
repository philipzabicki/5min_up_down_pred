"""Run or resume one full-development BTC pre-open fit and holdout evaluation."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_preopen_experiment as experiment
import run_btc_preopen_model_update as model_update
import run_btc_preopen_research as research
import run_btc_preopen_walk_forward as walk_forward


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_preopen_full_development_fit_20261016.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_dir"]
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
CONTRACT_PATH = OUTPUT_DIR / "contracts/development_holdout.json"
REPORT_PATH = ROOT / CONFIG["report_path"]
TABLE_DIR = ROOT / CONFIG["table_dir"]
STATUS_PATH = ROOT / "docs/BTC_RESEARCH_STATUS_20261010.md"
STATUS_BEGIN = "<!-- BTC_FULL_DEVELOPMENT_RUN_STATUS_BEGIN -->"
STATUS_END = "<!-- BTC_FULL_DEVELOPMENT_RUN_STATUS_END -->"

MODEL_PROVENANCE = "development_selected_configuration_chronological_oof_conditional"

COST_ESTIMATE = [
    {
        "stage": "Base indicators",
        "executions": "1 signature and artifact verification; reuse the compatible 64×8 fit",
        "basis": "Reused indicator stage completed in 0.03 s during the interrupted run",
        "estimated_seconds": [0, 5],
    },
    {
        "stage": "Reaction profile",
        "executions": "1 Optuna study; up to 96 candidates × 10 chronological CV folds",
        "basis": "96-trial interrupted study took 1,528 s on its shorter fit range; full-development study has a 2,700 s cap",
        "estimated_seconds": [1800, 2700],
    },
    {
        "stage": "Volume profile",
        "executions": "1 Optuna study; up to 96 candidates × 10 chronological CV folds",
        "basis": "96-trial interrupted study took 1,261 s on its shorter fit range; full-development study has a 2,700 s cap",
        "estimated_seconds": [1500, 2700],
    },
    {
        "stage": "Target weights",
        "executions": "1 study; 16 candidates × 10 chronological CV folds",
        "basis": "Same full-development pipeline measured 8,171 s; new cumulative stage budget is 10,800 s",
        "estimated_seconds": [8171, 10800],
    },
    {
        "stage": "Full candidate dataset",
        "executions": "1 build after the selected VP/RP settings are frozen",
        "basis": "Same full-development pipeline measured 223 s; generator changes require this dependent rebuild",
        "estimated_seconds": [223, 600],
    },
    {
        "stage": "Feature selection",
        "executions": "1 ranking/sweep; 10 chronological folds; existing limit of 60 top-k evaluations",
        "basis": "Same full-development pipeline measured 1,802 s; new stage budget is 2,400 s",
        "estimated_seconds": [1802, 2400],
    },
    {
        "stage": "Main-model tuning",
        "executions": "1 Optuna study; at most 60 trials × 10 chronological CV folds",
        "basis": "Prior 40-trial study had 18 COMPLETE trials in 945 s; current cumulative tuning cap is 7,200 s",
        "estimated_seconds": [3000, 7200],
    },
    {
        "stage": "Calibration predictions",
        "executions": "10 fixed-parameter model fits, one per internal chronological validation fold",
        "basis": "No parameter search; prior final full-development fit took 2.8 s at 34 trees; reserve 300 s for this block",
        "estimated_seconds": [30, 300],
    },
    {
        "stage": "Final model fit and calibration",
        "executions": "1 full-development fit, then 1 Platt fit; final-fit reserve is 300 s",
        "basis": "Prior final fit and calibrator took 2.8 s and 0.7 s respectively",
        "estimated_seconds": [5, 300],
    },
    {
        "stage": "Frozen holdout and snapshot economics",
        "executions": "1 holdout score, 1 bundle consistency check, and 3 frozen portfolio scenarios with existing T-60/T-59 snapshots",
        "basis": "Prior bundle consistency check took 270 s; prior five-scenario snapshot diagnostic took 142 s",
        "estimated_seconds": [420, 900],
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _now_utc() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _validate_frozen_config() -> None:
    if "outer_folds" in CONFIG:
        raise RuntimeError("The full-development configuration cannot contain outer pipeline folds")
    if len(CONFIG["variants"]) != 1:
        raise RuntimeError("The full-development run requires exactly one candidate pipeline")
    if CONFIG["development_cutoff_exclusive_utc"] != "2026-07-01T00:00:00Z":
        raise RuntimeError("Development cutoff differs from the accepted research contract")
    if CONFIG["data_cutoff_exclusive_utc"] != "2026-10-07T00:00:00Z":
        raise RuntimeError("Data cutoff differs from the accepted research contract")
    development_raw_path = Path(CONFIG["development_raw_data_path"])
    full_history_raw_path = Path(CONFIG["raw_data_path"])
    if development_raw_path == full_history_raw_path:
        raise RuntimeError("Full-development fitting requires a separate development-only candle input")
    if "development_cutoff_20260701" not in development_raw_path.stem:
        raise RuntimeError("Development candle input name must identify the frozen 2026-07-01 cutoff")
    if (
        CONFIG["holdout_market_start_first_inclusive_utc"] != "2026-07-01T00:00:00Z"
        or CONFIG["holdout_market_start_last_inclusive_utc"] != "2026-10-06T23:55:00Z"
    ):
        raise RuntimeError("Holdout market interval differs from the accepted freeze")
    if CONFIG["generator_search"]["trials_per_generator"] != 96:
        raise RuntimeError("VP/RP must share one 96-trial budget per generator for the full run")
    if CONFIG["main_model_search"]["trial_limit"] != 60:
        raise RuntimeError("Main-model study must use the configured 60-trial run-wide limit")
    if CONFIG["main_model_search"]["time_limit_seconds"] != 7200:
        raise RuntimeError("Main-model tuning limit differs from the accepted 7,200-second budget")
    if CONFIG["main_model_search"]["final_fit_reserve_seconds"] != 300:
        raise RuntimeError("The final model fit must retain its separate 300-second reserve")
    if CONFIG["reuse"]["generator_seed_sources"].keys() != {"reaction_profile", "volume_profile"}:
        raise RuntimeError("Only the existing VP and RP parameter seed sources are expected")

    import optimize_lgbm_hyperparameters as model_tuning

    model_tuning.validate_lgbm_search_space(CONFIG["main_model_search"]["search_space"])
    model_tuning.validate_seed_trial_params(
        CONFIG["main_model_search"]["seed_trial_params"],
        CONFIG["main_model_search"]["search_space"],
    )


def _holdout_fold() -> dict:
    folds = walk_forward.CONFIG["folds"]
    holdout = next(item for item in folds if item["fold_id"] == "evaluated_holdout")
    if (
        holdout["market_start_first_utc"] != CONFIG["holdout_market_start_first_inclusive_utc"]
        or holdout["market_start_last_inclusive_utc"] != CONFIG["holdout_market_start_last_inclusive_utc"]
    ):
        raise RuntimeError("Source holdout fold differs from the fixed date interval")
    return holdout


def _build_contract() -> dict:
    source_contract = json.loads((ROOT / CONFIG["source_contract"]).read_text(encoding="utf-8"))
    holdout = _holdout_fold()
    contract = walk_forward._fold_contract(source_contract, holdout)
    split = contract["chronological_split_frozen_before_tuning"]
    first_decision = pd.Timestamp(split["first_external_test_nominal_decision_utc"])
    latest_eligible_label = pd.Timestamp(split["fit_and_tuning_end_exclusive_utc"]) - pd.Timedelta(nanoseconds=1)
    if latest_eligible_label != first_decision:
        raise RuntimeError("Fit boundary must allow labels only through the first holdout decision")
    if split["external_test_market_start_last_inclusive_utc"] != "2026-10-06T23:55:00Z":
        raise RuntimeError("Holdout end differs from the fixed market interval")
    return contract


def _build_profile(contract_path: Path) -> dict:
    training = json.loads((ROOT / "configs/btc_preopen_training.json").read_text(encoding="utf-8"))
    profile = dict(training["profiles"]["nightly_v1"])
    variant = CONFIG["variants"][0]
    source = CONFIG["reuse"]["indicator_artifact_source"]
    generators = CONFIG["generator_search"]
    main = CONFIG["main_model_search"]
    runtime = CONFIG["runtime"]
    selector = CONFIG["feature_selection"]
    profile.update({
        "contract_path": contract_path.resolve().relative_to(ROOT).as_posix(),
        "raw_data_path": CONFIG["development_raw_data_path"],
        "development_input_path": CONFIG["development_raw_data_path"],
        "source_raw_data_path": CONFIG["raw_data_path"],
        "development_cutoff_exclusive_utc": CONFIG["development_cutoff_exclusive_utc"],
        "official_markets_path": CONFIG["official_markets_path"],
        "prediction_compute_budget_seconds": 1,
        "output_dir": f"{CONFIG['output_dir']}/pipeline",
        "runtime_budget_seconds": int(runtime["runtime_budget_seconds"]),
        "stage_runtime_budgets_seconds": dict(runtime["stage_budgets_seconds"]),
        "threads": int(runtime["threads"]),
        "compute_backend": runtime["compute_backend"],
        "indicator_fit": dict(CONFIG["indicator_fit"]),
        "indicator_seed": int(source["indicator_seed"]),
        "indicator_artifact_source": source,
        "generator_seed_trial_sources": CONFIG["reuse"]["generator_seed_sources"],
        "generator_tuning_time_limit_seconds": int(generators["time_limit_seconds_per_generator"]),
        "generator_checkpoint_trials": list(generators["checkpoint_trials"]),
        "generator_trial_limit_is_hard": True,
        "reaction_profile_trials": int(generators["trials_per_generator"]),
        "volume_profile_trials": int(generators["trials_per_generator"]),
        "minimum_successful_trials": int(generators["minimum_successful_trials"]),
        "cv_folds": int(main["cv_folds"]),
        "generator_cv_folds": int(main["cv_folds"]),
        "weight_cv_folds": int(main["cv_folds"]),
        "selector_cv_folds": int(main["cv_folds"]),
        "model_cv_folds": int(main["cv_folds"]),
        "target_weight_candidates": int(CONFIG["target_weight_search"]["candidates"]),
        "feature_selection": {
            "max_sweep_evaluations": int(selector["max_sweep_evaluations"]),
            "max_refinement_rounds": int(selector["max_refinement_rounds"]),
            "permutation_feature_fraction": float(selector["permutation_feature_fraction"]),
            "permutation_repeats": int(selector["permutation_repeats"]),
        },
        "model_trials": int(main["trial_limit"]),
        "minimum_successful_trials": int(main["minimum_successful_trials"]),
        "maximum_estimators": int(main["maximum_estimators"]),
        "early_stopping_rounds": int(main["early_stopping_rounds"]),
        "seed": int(main["random_seed"]),
        "model_trial_limit_is_hard": True,
        "main_model_tuning_time_limit_seconds": int(main["time_limit_seconds"]),
        "main_model_final_fit_reserve_seconds": int(main["final_fit_reserve_seconds"]),
        "model_checkpoint_trials": list(main["checkpoint_trials"]),
        "model_search_space_override": dict(main["search_space"]),
        "model_seed_trial_params_override": [dict(main["seed_trial_params"])],
        "model_feature_candidate_features": list(variant["candidate_features"]),
        "model_feature_families": list(CONFIG["model_feature_families"]),
        "calibration_oof_mode": "selected_development_cv",
        "frozen_policy_path": CONFIG["frozen_policy"]["source_path"],
    })
    return profile


def _source_identity(source_inputs: tuple, contract_path: Path) -> dict:
    source_identity = source_inputs[-1]
    code_paths = (
        "run_btc_preopen_full_development_fit.py",
        "run_btc_preopen_experiment.py",
        "run_btc_preopen_model_update.py",
        "optimize_lgbm_hyperparameters.py",
        "fit_reaction_profile.py",
        "fit_volume_profile.py",
        "optimize_target_weights.py",
        "select_features.py",
        "train_lgbm.py",
        "features/btc_preopen_contract.py",
    )
    return {
        "experiment_id": CONFIG["experiment_id"],
        "config_sha256": _sha256(CONFIG_PATH),
        "contract_sha256": _sha256(contract_path),
        "source_identity": source_identity,
        "source_walk_forward_manifest_sha256": source_identity["source_walk_forward_manifest_sha256"],
        "source_holdout_manifest_sha256": source_identity["source_holdout_manifest_sha256"],
        "t59_snapshot_sha256": source_identity["corrected_t59_snapshot_sha256"],
        "t60_snapshot_sha256": source_identity["corrected_t60_snapshot_sha256"],
        "frozen_policy_sha256": _sha256(ROOT / CONFIG["frozen_policy"]["source_path"]),
        "training_code_sha256": {name: _sha256(ROOT / name) for name in code_paths},
    }


def _write_progress_document(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = STATUS_PATH.read_text(encoding="utf-8")
    block = "\n".join([
        STATUS_BEGIN,
        "### Full-development run checkpoint",
        f"- Updated UTC: `{payload.get('updated_utc')}`.",
        f"- Status: `{payload.get('status')}`; active pipeline stage: `{payload.get('active_stage') or 'none'}`.",
        f"- Completed stages: `{', '.join(payload.get('completed_stages', [])) or 'none'}`.",
        f"- Trial checkpoint: `{json.dumps(payload.get('trial_checkpoint', {}), sort_keys=True)}`.",
        f"- Next action: `{payload.get('next_step')}`.",
        f"- Run manifest: `{payload.get('manifest_path')}`.",
        STATUS_END,
    ])
    if STATUS_BEGIN in existing and STATUS_END in existing:
        before, remainder = existing.split(STATUS_BEGIN, 1)
        _, after = remainder.split(STATUS_END, 1)
        updated = before + block + after
    else:
        updated = block + "\n\n" + existing
    temporary = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(STATUS_PATH)


def _monitor_pipeline(run_root: Path, stop: threading.Event) -> None:
    last_signature = None
    while not stop.wait(30):
        manifest_path = run_root / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            pipeline = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stages = pipeline.get("stages", {})
        active = next((name for name, stage in stages.items() if stage.get("status") == "running"), None)
        completed = [name for name, stage in stages.items() if stage.get("status") == "completed"]
        trial_checkpoint = {}
        if active:
            progress_paths = list((run_root / "stages").glob(f"{active}_*/trial_progress.json"))
            if progress_paths:
                newest = max(progress_paths, key=lambda path: path.stat().st_mtime)
                try:
                    progress = json.loads(newest.read_text(encoding="utf-8"))
                    trial_checkpoint = {
                        key: progress.get(key)
                        for key in (
                            "checkpoint_trial_count", "evaluated_trials", "cumulative_elapsed_seconds",
                            "target_trials", "trial_counts", "best_cv_criterion", "best_iteration",
                        )
                        if key in progress
                    }
                except (OSError, json.JSONDecodeError):
                    trial_checkpoint = {}
        signature = (active, tuple(completed), json.dumps(trial_checkpoint, sort_keys=True, default=str))
        if signature == last_signature:
            continue
        last_signature = signature
        _write_progress_document({
            "updated_utc": _now_utc(),
            "status": pipeline.get("status"),
            "active_stage": active,
            "completed_stages": completed,
            "trial_checkpoint": trial_checkpoint,
            "next_step": f"Resume with `{CONFIG['reproduction']['runner']}` after confirming no copy is active.",
            "manifest_path": (run_root / "run_manifest.json").resolve().relative_to(ROOT).as_posix(),
        })


def _holdout_economics(
    predictions_path: Path,
    correction_manifest: dict,
    *,
    input_identity: dict,
) -> dict:
    predictions = pd.read_parquet(predictions_path)
    snapshots, decision_snapshots, book_record = research._load_enriched_book_snapshots(correction_manifest)
    calendar, _ = research._build_market_calendar(predictions)
    calendar, _ = research.replay._market_statuses(calendar, snapshots, decision_snapshots)
    holdout_rows, period_summary = research._asof_market_rows(
        calendar,
        pd.Timestamp(CONFIG["development_cutoff_exclusive_utc"]),
        pd.Timestamp(CONFIG["data_cutoff_exclusive_utc"]),
    )
    if holdout_rows.empty or holdout_rows.p_candidate_platt.notna().sum() == 0:
        raise RuntimeError("The frozen model produced no holdout rows for the existing T-60/T-59 snapshots")
    output_root = OUTPUT_DIR / "holdout_evaluation"
    selected_name = str(CONFIG["frozen_policy"]["policy_name"])
    policies = [
        (selected_name, "free_cash_5pct", float(CONFIG["frozen_policy"]["minimum_net_ev_fraction"])),
        ("fixed_5_usd_control", "fixed_5_usd", 0.0),
        ("free_cash_5pct_control", "free_cash_5pct", 0.0),
    ]
    identity = {
        **input_identity,
        "holdout_predictions_sha256": _sha256(predictions_path),
        "t59_coverage_sha256": book_record["t59_coverage_sha256"],
        "t60_coverage_sha256": book_record["t60_coverage_sha256"],
        "frozen_policy_sha256": _sha256(ROOT / CONFIG["frozen_policy"]["source_path"]),
        "period_start_utc": CONFIG["development_cutoff_exclusive_utc"],
        "period_cutoff_exclusive_utc": CONFIG["data_cutoff_exclusive_utc"],
        "policies": policies,
    }
    result = research._evaluate_portfolio_period(
        correction_manifest,
        predictions,
        start=pd.Timestamp(CONFIG["development_cutoff_exclusive_utc"]),
        cutoff=pd.Timestamp(CONFIG["data_cutoff_exclusive_utc"]),
        policies=policies,
        output_root=output_root,
        period_name="holdout",
        input_identity=identity,
    )
    result["period_summary"] = period_summary
    result["historically_exposed"] = True
    result["selection_from_holdout"] = False
    result_path = output_root / "holdout_portfolio_results.json"
    _write_json(result_path, result)
    return {
        "status": "completed",
        "results_path": result_path.resolve().relative_to(ROOT).as_posix(),
        "results_sha256": _sha256(result_path),
        "holdout_market_rows": int(period_summary["market_rows_in_period"]),
        "holdout_outcomes_available": int(period_summary["official_outcomes_and_cash_releases_available_by_cutoff"]),
        "policies": list(result["policies"]),
        "holdout_selection": False,
    }


def _write_final_report(manifest: dict, pipeline_manifest: dict, economics: dict) -> None:
    run_root = ROOT / manifest["pipeline_run_root"]
    bundle_path = ROOT / manifest["bundle_path"]
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    freeze_path = ROOT / bundle["evaluated_bundle_freeze_path"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    evaluation_stage = pipeline_manifest["stages"]["external_evaluation"]
    evaluation_path = ROOT / evaluation_stage["result"]["metrics_path"]
    if not evaluation_path.is_file():
        json_outputs = [
            ROOT / output["path"]
            for output in evaluation_stage.get("outputs", [])
            if str(output.get("path", "")).lower().endswith(".json")
        ]
        evaluation_path = next((path for path in json_outputs if path.is_file()), evaluation_path)
    if not evaluation_path.is_file():
        raise RuntimeError("Completed external evaluation metrics JSON is missing")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    economics_path = ROOT / economics["results_path"]
    economic_result = json.loads(economics_path.read_text(encoding="utf-8"))
    model_training_path = ROOT / bundle["model_path"]
    model_training = json.loads((model_training_path.parent / "training_manifest.json").read_text(encoding="utf-8"))
    calibrator = json.loads((ROOT / bundle["calibrator_path"]).read_text(encoding="utf-8"))
    pipeline_stages = pipeline_manifest.get("stages", {})
    generator_rows = []
    for stage_name in ("volume_profile", "reaction_profile"):
        stage = pipeline_stages.get(stage_name, {}).get("result", {})
        generator_rows.append(
            f"| {stage_name} | 1 | {stage.get('trials_total', 0)} | {stage.get('successful_trials', 0)} | "
            f"{stage.get('pruned_trials', 0)} | {stage.get('error_trials', 0)} |"
        )
    model_stage = pipeline_stages.get("model_tuning", {}).get("result", {})
    oof_stage = pipeline_stages.get("calibration_oof", {}).get("result", {})
    official = evaluation.get("metrics", {}).get("official_market_data", {}).get("metrics", {})
    raw = official.get("raw_model", {})
    platt = official.get("platt_model", {})
    economic_lines = []
    for name, item in economic_result["policies"].items():
        summary = item["summary"]
        economic_lines.append(
            f"| `{name}` | {summary.get('trade_count', 0)} | ${summary.get('ending_cash_after_settlement_usd', 0.0):,.2f} "
            f"| ${summary.get('net_pnl_usd', 0.0):,.2f} | ${summary.get('gross_turnover_usd', 0.0):,.2f} "
            f"| ${summary.get('fees_paid_estimated_usd', 0.0):,.2f} | {summary.get('max_drawdown_cost_basis_equity', 0.0):.2%} "
            f"| {summary.get('accounting_audit', {}).get('status', 'unavailable')} |"
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(TABLE_DIR / "evaluation_summary.json", evaluation)
    _write_json(TABLE_DIR / "holdout_economics.json", economic_result)
    _write_json(TABLE_DIR / "frozen_bundle_manifest.json", freeze)
    report = f"""# Full-development BTC pre-open fit - 2026-10-16

## Run status

- Status: completed; one full-development model pipeline and one historical holdout evaluation.
- Historically exposed holdout: {CONFIG['historical_holdout_status']}
- Development markets: starts before `{CONFIG['development_cutoff_exclusive_utc']}`. Holdout market starts: `{CONFIG['holdout_market_start_first_inclusive_utc']}` through `{CONFIG['holdout_market_start_last_inclusive_utc']}` inclusive. Exclusive data cutoff: `{CONFIG['data_cutoff_exclusive_utc']}`.
- Model fit's latest eligible label time: `{model_training.get('latest_label_available_at')}`; the frozen contract allows labels only through the first holdout decision.
- Calibration: `{calibrator.get('calibration_source')}`; {calibrator.get('training_rows', 0):,} development rows. Provenance: `{calibrator.get('prediction_provenance')}`. These validation predictions are conditional on generator settings, weights, features, and hyperparameters selected on development; they are not an independent evaluation of the whole selection process.
- Calibration prediction folds: {oof_stage.get('fold_count', 0)}; rows: {oof_stage.get('rows', 0):,}; holdout rows included: `{oof_stage.get('holdout_rows_included')}`.
- Frozen bundle: `{manifest['bundle_path']}` (SHA-256 `{manifest['bundle_sha256']}`). Freeze: `{bundle.get('evaluated_bundle_freeze_path')}` (SHA-256 `{bundle.get('evaluated_bundle_freeze_sha256')}`).
- Frozen policy: `{CONFIG['frozen_policy']['policy_name']}`, {float(CONFIG['frozen_policy']['minimum_net_ev_fraction']):.2%} minimum net EV; no holdout selection.

## One-pass development stages

| Stage | Studies | Trials evaluated | COMPLETE | PRUNED | Errors |
|---|---:|---:|---:|---:|---:|
{chr(10).join(generator_rows)}
| main model | 1 | {model_stage.get('trials_total', 0)} | {model_stage.get('successful_trials', 0)} | {model_stage.get('pruned_trials', 0)} | {model_stage.get('error_trials', 0)} |

The main-model study used {len(model_stage.get('fold_indices', [])) or CONFIG['main_model_search']['cv_folds']} internal chronological folds. Its development CV score is a tuning result conditional on development selection. No external loop repeats the complete generator, weighting, selection, or tuning stages.

## Holdout probability quality

| Forecast | Rows | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|
| Raw model | {raw.get('n', 0):,} | {raw.get('log_loss', float('nan')):.6f} | {raw.get('brier', float('nan')):.6f} | {raw.get('auc', float('nan')):.4f} |
| Platt calibrated | {platt.get('n', 0):,} | {platt.get('log_loss', float('nan')):.6f} | {platt.get('brier', float('nan')):.6f} | {platt.get('auc', float('nan')):.4f} |

## Frozen-policy economics from existing snapshots

The single holdout portfolio evaluation reused the hash-verified T-60 decision and T-59 execution snapshots. It did not replay the book archive. It scored {economics['holdout_market_rows']:,} holdout market rows, with {economics['holdout_outcomes_available']:,} official outcomes and cash releases available by the frozen data cutoff.

| Scenario | Trades | Ending cash | Net PnL | Turnover | Fees | Max drawdown | Accounting |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(economic_lines)}

The holdout evaluation also records T-60/T-59 decision and fill coverage, full-book versus top-of-book proxy sources, rejection reasons, and adverse-price sensitivity under the existing portfolio accounting rules. None of these holdout results changed the model, calibrator, or policy.

## Cost and artifacts

- Runtime estimate recorded before the run: `data/analysis/polymarket/BTC/btc_preopen_full_development_fit_20261016/cost_estimate.json` (forecast 5-8 hours from measured full-development stages; separate from the superseded multi-pipeline estimate).
- Pipeline manifest: `{(run_root / 'run_manifest.json').resolve().relative_to(ROOT).as_posix()}`.
- Holdout economics: `{economics['results_path']}`.
- Frozen fit rows: {model_training.get('training_rows', 0):,}; features: {len(bundle.get('feature_order', []))}; best iteration: {model_training.get('best_iteration')}.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def _run_or_resume() -> dict:
    _validate_frozen_config()
    model_update.configure_model_update(CONFIG_PATH)
    source_inputs = model_update._source_inputs()
    correction_manifest = source_inputs[5]
    book_record = source_inputs[6]
    policy_freeze = json.loads((ROOT / CONFIG["frozen_policy"]["source_path"]).read_text(encoding="utf-8"))
    selected_policy = policy_freeze["policy_selection"]["selected_policy"]
    if (
        selected_policy.get("policy") != CONFIG["frozen_policy"]["policy_name"]
        or not np.isclose(float(selected_policy.get("minimum_net_ev_fraction")), float(CONFIG["frozen_policy"]["minimum_net_ev_fraction"]))
        or policy_freeze["policy_selection"].get("holdout_used_for_selection") is not False
    ):
        raise RuntimeError("The frozen development policy does not match this experiment configuration")

    contract = _build_contract()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    walk_forward._write_json(CONTRACT_PATH, contract)
    identity = _source_identity(source_inputs, CONTRACT_PATH)
    _write_json(OUTPUT_DIR / "cost_estimate.json", {
        "recorded_before_training_start_utc": _now_utc(),
        "estimated_runtime_hours": [5, 8],
        "stages": COST_ESTIMATE,
        "source_of_measurements": {
            "full_development_pipeline": "data/analysis/polymarket/BTC/preopen_research_20261011_full_compute20_retry/runs/evaluated_holdout/794a69e369a3f82d/run_manifest.json",
            "interrupted_scope_change": "data/analysis/polymarket/BTC/btc_preopen_vp_rp_tuning_20261016/model_update_manifest.json",
        },
    })
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.is_file() else {
        "schema_version": 1,
        "experiment_id": CONFIG["experiment_id"],
        "stages": {},
    }
    if manifest.get("input_identity") not in (None, identity):
        raise RuntimeError("Existing full-development output belongs to different frozen inputs or code")
    if manifest.get("status") == "interrupted_scope_change":
        raise RuntimeError("This full-development run was interrupted after a scope change and is not resumable")
    manifest.update({
        "input_identity": identity,
        "status": "running",
        "started_utc": manifest.get("started_utc", _now_utc()),
        "updated_utc": _now_utc(),
        "cost_estimate_path": (OUTPUT_DIR / "cost_estimate.json").resolve().relative_to(ROOT).as_posix(),
        "cost_estimate_sha256": _sha256(OUTPUT_DIR / "cost_estimate.json"),
        "resource_snapshot_at_start": model_update._resource_snapshot(),
        "development_cutoff_exclusive_utc": CONFIG["development_cutoff_exclusive_utc"],
        "holdout_market_start_first_inclusive_utc": CONFIG["holdout_market_start_first_inclusive_utc"],
        "holdout_market_start_last_inclusive_utc": CONFIG["holdout_market_start_last_inclusive_utc"],
        "data_cutoff_exclusive_utc": CONFIG["data_cutoff_exclusive_utc"],
        "holdout_historically_exposed": True,
        "holdout_used_for_selection": False,
    })
    _write_json(MANIFEST_PATH, manifest)
    _write_progress_document({
        "updated_utc": _now_utc(),
        "status": "running",
        "active_stage": "preflight complete; one full-development pipeline starting",
        "completed_stages": [],
        "trial_checkpoint": {},
        "next_step": f"Run `{CONFIG['reproduction']['runner']}` to resume this manifest.",
        "manifest_path": MANIFEST_PATH.resolve().relative_to(ROOT).as_posix(),
    })

    profile = _build_profile(CONTRACT_PATH)
    pipeline_record = manifest.get("stages", {}).get("model_pipeline", {})
    if pipeline_record.get("status") == "completed":
        run_root = ROOT / pipeline_record["run_root"]
        pipeline_manifest_path = run_root / "run_manifest.json"
        if not pipeline_manifest_path.is_file():
            raise RuntimeError("Completed pipeline checkpoint is missing its manifest")
        pipeline_manifest = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
        if pipeline_manifest.get("status") != "completed":
            raise RuntimeError("Top-level pipeline checkpoint and nested run manifest disagree")
        bundle_path = ROOT / pipeline_record["bundle_path"]
        predictions_path = ROOT / pipeline_record["predictions_path"]
        if _sha256(bundle_path) != pipeline_record["bundle_sha256"] or _sha256(predictions_path) != pipeline_record["predictions_sha256"]:
            raise RuntimeError("Completed model or holdout prediction checksum changed")
    else:
        run_root_hint = ROOT / profile["output_dir"]
        run_root_hint.mkdir(parents=True, exist_ok=True)
        monitor_stop = threading.Event()
        monitor = threading.Thread(
            target=_monitor_pipeline,
            args=(run_root_hint, monitor_stop),
            daemon=True,
            name="btc-full-development-progress",
        )
        monitor.start()
        manifest["stages"]["model_pipeline"] = {"status": "running", "started_utc": _now_utc()}
        _write_json(MANIFEST_PATH, manifest)
        try:
            result = experiment.run_training_pipeline(profile_override=profile)
        finally:
            monitor_stop.set()
            monitor.join(timeout=5)
        run_root = result["run_root"]
        prediction_path, pipeline_manifest = walk_forward._completed_predictions(run_root)
        bundle_path = result["bundle_path"]
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        training_manifest_path = ROOT / bundle["model_path"]
        training_manifest = json.loads((training_manifest_path.parent / "training_manifest.json").read_text(encoding="utf-8"))
        fit_label_limit = pd.Timestamp(contract["chronological_split_frozen_before_tuning"]["fit_label_available_through_inclusive_utc"])
        if pd.Timestamp(training_manifest["latest_label_available_at"]) > fit_label_limit:
            raise RuntimeError("Final model used a label unavailable at the first holdout decision")
        calibration = json.loads((ROOT / bundle["calibrator_path"]).read_text(encoding="utf-8"))
        if (
            calibration.get("conditional_on_development_selection") is not True
            or MODEL_PROVENANCE not in calibration.get("prediction_provenance", [])
        ):
            raise RuntimeError("Final calibrator lacks conditional development-CV prediction provenance")
        freeze_path = ROOT / bundle["evaluated_bundle_freeze_path"]
        record = {
            "status": "completed",
            "run_root": run_root.resolve().relative_to(ROOT).as_posix(),
            "bundle_path": bundle_path.resolve().relative_to(ROOT).as_posix(),
            "bundle_sha256": _sha256(bundle_path),
            "freeze_path": freeze_path.resolve().relative_to(ROOT).as_posix(),
            "freeze_sha256": _sha256(freeze_path),
            "predictions_path": prediction_path.resolve().relative_to(ROOT).as_posix(),
            "predictions_sha256": _sha256(prediction_path),
            "training_manifest_path": training_manifest_path.parent.joinpath("training_manifest.json").resolve().relative_to(ROOT).as_posix(),
            "latest_training_label_available_at_utc": training_manifest["latest_label_available_at"],
            "calibration_source": calibration["calibration_source"],
            "calibration_training_rows": int(calibration["training_rows"]),
            "calibration_oof_provenance": calibration["prediction_provenance"],
            "calibration_oof_conditional": True,
            "pipeline_manifest_sha256": _sha256(run_root / "run_manifest.json"),
            "finished_utc": _now_utc(),
        }
        manifest.setdefault("stages", {})["model_pipeline"] = record
        manifest["pipeline_run_root"] = record["run_root"]
        manifest["bundle_path"] = record["bundle_path"]
        manifest["bundle_sha256"] = record["bundle_sha256"]
        manifest["freeze_path"] = record["freeze_path"]
        manifest["freeze_sha256"] = record["freeze_sha256"]
        manifest["predictions_path"] = record["predictions_path"]
        manifest["predictions_sha256"] = record["predictions_sha256"]
        manifest["updated_utc"] = _now_utc()
        _write_json(MANIFEST_PATH, manifest)

    economics_record = manifest.get("stages", {}).get("holdout_evaluation", {})
    if economics_record.get("status") == "completed":
        economics_path = ROOT / economics_record["results_path"]
        if not economics_path.is_file() or _sha256(economics_path) != economics_record["results_sha256"]:
            raise RuntimeError("Completed holdout economics checkpoint checksum mismatch")
    else:
        manifest.setdefault("stages", {})["holdout_evaluation"] = {"status": "running", "started_utc": _now_utc()}
        manifest["updated_utc"] = _now_utc()
        _write_json(MANIFEST_PATH, manifest)
        economics_record = _holdout_economics(
            ROOT / manifest["predictions_path"],
            correction_manifest,
            input_identity=identity,
        )
        manifest["stages"]["holdout_evaluation"] = economics_record | {"finished_utc": _now_utc()}
        manifest["updated_utc"] = _now_utc()
        _write_json(MANIFEST_PATH, manifest)

    pipeline_manifest_path = ROOT / manifest["pipeline_run_root"] / "run_manifest.json"
    pipeline_manifest = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
    _write_final_report(manifest, pipeline_manifest, economics_record)
    manifest.update({
        "status": "completed",
        "finished_utc": _now_utc(),
        "updated_utc": _now_utc(),
        "report_path": REPORT_PATH.resolve().relative_to(ROOT).as_posix(),
        "report_sha256": _sha256(REPORT_PATH),
        "holdout_selection": False,
        "resource_snapshot_at_completion": model_update._resource_snapshot(),
    })
    _write_json(MANIFEST_PATH, manifest)
    _write_progress_document({
        "updated_utc": _now_utc(),
        "status": "completed",
        "active_stage": "none",
        "completed_stages": ["model_pipeline", "holdout_evaluation"],
        "trial_checkpoint": {},
        "next_step": "The holdout is closed to model and policy changes for this run.",
        "manifest_path": MANIFEST_PATH.resolve().relative_to(ROOT).as_posix(),
    })
    return manifest


def run() -> dict:
    started = time.perf_counter()
    try:
        return _run_or_resume()
    except KeyboardInterrupt:
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.is_file() else {}
        existing.update({"status": "interrupted", "updated_utc": _now_utc(), "next_step": CONFIG["reproduction"]["runner"]})
        _write_json(MANIFEST_PATH, existing)
        raise
    except Exception as error:
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.is_file() else {}
        existing.update({"status": "failed", "updated_utc": _now_utc(), "error": f"{type(error).__name__}: {error}"})
        _write_json(MANIFEST_PATH, existing)
        raise
    finally:
        print(f"[btc-full-development] elapsed_seconds={time.perf_counter() - started:.1f}", flush=True)


if __name__ == "__main__":
    run()
