"""Continue five frozen, formerly-pruned parameter sets on development folds."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

import optimize_lgbm_hyperparameters as model_tuning
import run_btc_preopen_experiment as preopen
import run_btc_preopen_model_update as model_update
import run_btc_preopen_research as research


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "reports/btc_preopen/b_pruner_development_diagnostic_20261014"
MANIFEST_PATH = OUTPUT_DIR / "pruner_diagnostic_manifest.json"
RESULTS_PATH = OUTPUT_DIR / "pruner_diagnostic_results.json"
REPORT_PATH = OUTPUT_DIR / "pruner_diagnostic_report.md"
REFERENCE_TRIALS_PATH = OUTPUT_DIR / "completed_development_trials.csv"
CONFIG_CSV = ROOT / "reports/btc_preopen/btc_preopen_model_update_20261012_partial/b_pruned_configs_for_development_test.csv"
OUTER_MANIFEST_PATH = model_update.MANIFEST_PATH
FOLDS = ("dev_2026q2_a", "dev_2026q2_b", "dev_2026q2_c")
VARIANT = "B_dynamic_price_volume"
FEATURES = (
    ROOT / "run_btc_preopen_experiment.py",
    ROOT / "optimize_lgbm_hyperparameters.py",
    ROOT / "features/btc_preopen_contract.py",
    ROOT / "utils/metrics.py",
    ROOT / "train_lgbm.py",
)
INT_PARAMS = {"min_data_in_leaf", "num_leaves"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _study_rows(path: Path) -> tuple[str, list[dict]]:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        study_name = connection.execute("SELECT study_name FROM studies").fetchone()[0]
        trial_rows = connection.execute(
            "SELECT trial_id, number, state, datetime_start, datetime_complete "
            "FROM trials ORDER BY number"
        ).fetchall()
        values = dict(connection.execute(
            "SELECT trial_id, value FROM trial_values WHERE objective=0"
        ).fetchall())
        params: dict[int, dict] = {}
        for trial_id, name, value in connection.execute(
            "SELECT trial_id, param_name, param_value FROM trial_params"
        ).fetchall():
            params.setdefault(int(trial_id), {})[name] = float(value)
        attrs: dict[int, dict] = {}
        for trial_id, name, value_json in connection.execute(
            "SELECT trial_id, key, value_json FROM trial_user_attributes"
        ).fetchall():
            attrs.setdefault(int(trial_id), {})[name] = json.loads(value_json)
        curves: dict[int, list[dict]] = {}
        for trial_id, step, value in connection.execute(
            "SELECT trial_id, step, intermediate_value FROM trial_intermediate_values "
            "WHERE intermediate_value_type='FINITE' ORDER BY trial_id, step"
        ).fetchall():
            curves.setdefault(int(trial_id), []).append({
                "step": int(step), "iteration": int(step) + 1, "value": float(value),
            })
    finally:
        connection.close()

    rows = []
    for trial_id, number, state, started, completed in trial_rows:
        duration = None
        if started and completed:
            duration = (pd.Timestamp(completed) - pd.Timestamp(started)).total_seconds()
        rows.append({
            "trial_number": int(number),
            "state": str(state),
            "duration_seconds": duration,
            "objective_value": float(values[trial_id]) if trial_id in values else None,
            "best_iteration": attrs.get(int(trial_id), {}).get("best_iteration"),
            "params": params.get(int(trial_id), {}),
            "intermediate_curve": curves.get(int(trial_id), []),
        })
    return str(study_name), rows


def _read_candidate_configs(source_study_path: Path) -> tuple[list[dict], int, list[int]]:
    source_name, source_trials = _study_rows(source_study_path)
    trials_by_number = {row["trial_number"]: row for row in source_trials}
    configs = []
    with CONFIG_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            params = json.loads(row["params"])
            normalized = {
                name: int(value) if name in INT_PARAMS else float(value)
                for name, value in params.items()
            }
            source = trials_by_number[int(row["source_trial_number"])]
            if source["state"] != "PRUNED":
                raise RuntimeError(f"Source trial {row['source_trial_number']} is no longer PRUNED")
            for name, value in normalized.items():
                actual = source["params"].get(name)
                if actual is None or not math.isclose(float(value), float(actual), rel_tol=0.0, abs_tol=1e-12):
                    raise RuntimeError(f"Frozen config differs from source trial {source['trial_number']} ({name})")
            pruned_at = source["intermediate_curve"][-1]["iteration"]
            if pruned_at != int(row["source_pruned_at_iteration"]):
                raise RuntimeError(f"Pruning checkpoint differs for source trial {source['trial_number']}")
            configs.append({
                "config_id": row["test_config_id"],
                "source_trial_number": int(row["source_trial_number"]),
                "source_study_name": source_name,
                "source_state": source["state"],
                "source_pruned_at_iteration": pruned_at,
                "source_pruning_checkpoint_value": source["intermediate_curve"][-1]["value"],
                "source_intermediate_curve": source["intermediate_curve"],
                "selection_rule": row["selection_rule"],
                "params": normalized,
            })
    if len(configs) != 5 or len({item["config_id"] for item in configs}) != 5:
        raise RuntimeError("The frozen development diagnostic must contain exactly five configurations")
    checkpoints = sorted({
        point["iteration"]
        for row in source_trials if row["state"] == "PRUNED"
        for point in row["intermediate_curve"]
    })
    intervals = [right - left for left, right in zip(checkpoints, checkpoints[1:])]
    if not intervals or len(set(intervals)) != 1:
        raise RuntimeError("Could not identify a single frozen pruning checkpoint interval")
    return configs, intervals[0], checkpoints


def _find_study_path(run_root: Path, stage: dict) -> Path:
    for artifact in stage.get("outputs", []):
        path = ROOT / artifact["path"]
        if path.name == "study.sqlite3" and path.is_file():
            return path
    signature = str(stage.get("input_signature", ""))
    if signature:
        path = run_root / "stages" / f"model_tuning_{signature[:12]}" / "study.sqlite3"
        if path.is_file():
            return path
    raise RuntimeError(f"Model-tuning study database is missing under {run_root}")


def _stage_output(stage: dict, filename: str) -> dict:
    for artifact in stage.get("outputs", []):
        if Path(artifact["path"]).name == filename:
            return artifact
    raise RuntimeError(f"Stage manifest has no {filename} output")


def _frozen_inputs() -> dict:
    outer = _read_json(OUTER_MANIFEST_PATH)
    source_holdout = outer["folds"]["evaluated_holdout"]["A_reoptimized"]
    holdout_run_root = ROOT / source_holdout["run_root"]
    holdout_run = _read_json(holdout_run_root / "run_manifest.json")
    holdout_tuning = holdout_run["stages"]["model_tuning"]
    source_study = _find_study_path(holdout_run_root, holdout_tuning)
    configs, checkpoint_interval, source_checkpoints = _read_candidate_configs(source_study)

    fold_inputs = []
    reference_trials = []
    for fold_id in FOLDS:
        record = outer["folds"][fold_id][VARIANT]
        run_root = ROOT / record["run_root"]
        run = _read_json(run_root / "run_manifest.json")
        tuning_stage = run["stages"]["model_tuning"]
        tuning_result = tuning_stage["result"]
        model_manifest_path = ROOT / record["training_manifest_path"]
        model_manifest = _read_json(model_manifest_path)
        feature_order_path = ROOT / record["bundle_path"].replace("model_bundle.json", "feature_order.json")
        feature_order = model_manifest["feature_order"]
        feature_order_payload = _read_json(feature_order_path)
        if isinstance(feature_order_payload, dict):
            feature_order_payload = feature_order_payload.get("feature_order")
        if feature_order_payload != feature_order:
            raise RuntimeError(f"B model feature order differs from its training manifest for {fold_id}")
        dataset_artifact = _stage_output(run["stages"]["final_feature_dataset"], Path(
            run["stages"]["final_feature_dataset"]["result"]["dataset_path"]
        ).name)
        dataset_path = ROOT / dataset_artifact["path"]
        contract_path = ROOT / record["contract_path"]
        contract = _read_json(contract_path)
        tuning_db = _find_study_path(run_root, tuning_stage)
        _, completed = _study_rows(tuning_db)
        completed = [row for row in completed if row["state"] == "COMPLETE"]
        if not completed:
            raise RuntimeError(f"Development B study has no completed reference trials for {fold_id}")
        reference_trials.extend({
            "fold_id": fold_id,
            "trial_number": row["trial_number"],
            "objective_value": row["objective_value"],
            "best_iteration": row["best_iteration"],
            "duration_seconds": row["duration_seconds"],
            "learning_rate": row["params"].get("learning_rate"),
            "params": json.dumps(row["params"], sort_keys=True),
        } for row in completed)
        fold_inputs.append({
            "fold_id": fold_id,
            "run_root": record["run_root"],
            "fit_start_utc": record["model_fit_start_utc"],
            "fit_end_exclusive_utc": record["fit_and_tuning_end_exclusive_utc"],
            "latest_training_label_available_at_utc": record["latest_training_label_available_at_utc"],
            "training_rows_expected": int(record["fit_rows"]),
            "decision_rows_expected": int(record["decision_rows"]),
            "decision_weight": float(model_manifest["decision_weight"]),
            "feature_order": feature_order,
            "feature_count": len(feature_order),
            "dataset_path": dataset_artifact["path"],
            "dataset_sha256": dataset_artifact["sha256"],
            "dataset_size_bytes": int(dataset_artifact["size_bytes"]),
            "feature_dataset_stage_sha256": _sha256(run_root / "stages" / f"final_feature_dataset_{run['stages']['final_feature_dataset']['input_signature'][:12]}" / "feature_dataset_stage.json"),
            "feature_order_path": feature_order_path.resolve().relative_to(ROOT).as_posix(),
            "feature_order_sha256": _sha256(feature_order_path),
            "model_training_manifest_path": model_manifest_path.resolve().relative_to(ROOT).as_posix(),
            "model_training_manifest_sha256": _sha256(model_manifest_path),
            "fold_contract_path": record["contract_path"],
            "fold_contract_sha256": _sha256(contract_path),
            "contract_fit_boundary_utc": contract["chronological_split_frozen_before_tuning"]["fit_label_available_through_inclusive_utc"],
            "tuning_study_path": tuning_db.resolve().relative_to(ROOT).as_posix(),
            "tuning_study_sha256": _sha256(tuning_db),
            "tuning_stage_input_signature": tuning_stage["input_signature"],
            "fold_index_fingerprints": tuning_result["fold_indices"],
            "training_weight_sha256": tuning_result["training_weight_sha256"],
            "reference_completed_trials": [row["trial_number"] for row in completed],
            "reference_completed_best_objective": min(float(row["objective_value"]) for row in completed),
            "reference_completed_median_objective": float(np.median([float(row["objective_value"]) for row in completed])),
            "reference_completed_count": len(completed),
        })

    _, source_trials = _study_rows(source_study)
    source_rows = {row["trial_number"]: row for row in source_trials}
    source_contract_path = ROOT / model_update.CONTRACT_DIR / "evaluated_holdout.json"
    source_contract = _read_json(source_contract_path)
    source_fit = source_contract["chronological_split_frozen_before_tuning"]
    source_parameters = [{
        **config,
        "source_objective_value": source_rows[config["source_trial_number"]]["objective_value"],
        "source_learning_rate": config["params"]["learning_rate"],
    } for config in configs]
    return {
        "schema_version": 1,
        "experiment_id": "btc_preopen_pruner_development_diagnostic_20261014",
        "status": "prepared_before_trials",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "06036c93c59c36052d8a2357db72cb06e5e0b246",
        "source_parameter_study": {
            "path": source_study.resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha256(source_study),
            "study_name": _study_rows(source_study)[0],
            "source_fit_label_available_through_utc": source_fit["fit_label_available_through_inclusive_utc"],
            "source_fit_end_exclusive_utc": source_fit["fit_and_tuning_end_exclusive_utc"],
            "objective_scope": "binary price-proxy training/CV rows whose labels were available by the frozen fit boundary; no post-holdout labels",
            "holdout_labels_in_objective": False,
            "source_contract_path": source_contract_path.resolve().relative_to(ROOT).as_posix(),
            "source_contract_sha256": _sha256(source_contract_path),
        },
        "source_config_csv": {
            "path": CONFIG_CSV.resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha256(CONFIG_CSV),
            "selection_rule": "one median-learning-rate-ranked PRUNED trial from each quintile; ties by trial number",
            "configurations": source_parameters,
        },
        "source_data_identity": {
            "raw_price_data_path": model_update.CONFIG["raw_data_path"],
            "raw_price_data_sha256": outer["source_identity"]["raw_data_sha256"],
            "dataset_cutoff_exclusive_utc": outer["source_identity"]["data_cutoff_exclusive_utc"],
        },
        "folds": fold_inputs,
        "procedure": {
            "scope": "fixed-parameter continuation on three development folds; diagnostic CV only, not a new independent OOS test",
            "model_variant": VARIANT,
            "training_target": "target_preopen_5m_up_binance_index_proxy; Opened+2m open versus Opened+6m close; label available Opened+7m",
            "metric": "unweighted actual-decision-row binary_logloss",
            "objective_aggregation": "mean binary_logloss + 0.5 * unweighted fold std; minimized",
            "validation_rows": "the same ten chronological, purged internal folds as each development B tuning study; decision rows only and no validation observation weights",
            "training_weights": "fold-specific frozen decision weight; auxiliary rows receive (1 - decision_weight) / 4",
            "model_cv_folds": 10,
            "test_to_train_ratio": 0.1,
            "maximum_iterations": int(model_update.CONFIG["model_search"]["maximum_estimators"]),
            "early_stopping_rounds": int(model_update.CONFIG["model_search"]["early_stopping_rounds"]),
            "pruner_checkpoint_iterations": source_checkpoints,
            "pruner_report_interval_iterations": checkpoint_interval,
            "original_pruner": {
                "class": "optuna.pruners.MedianPruner",
                "n_startup_trials": 3,
                "n_warmup_steps": 10,
                "interval_steps": 1,
                "n_min_trials": 1,
            },
            "diagnostic_pruning_behavior": "disabled only: the production objective/report callback is retained; should_prune always returns false",
            "new_tpe_proposals": 0,
            "new_feature_selection_or_generator_optimization": False,
            "random_seed": int(model_update.CONFIG["model_search"]["random_seed"]),
            "compute_backend": model_update.CONFIG["runtime_profile"]["compute_backend"],
            "threads": int(model_update.CONFIG["runtime_profile"]["threads"]),
            "comparison": "continued objective and best iteration versus the COMPLETE configurations in the same development B Optuna study; report exact best and median completed objectives",
            "decision_replay": False,
            "holdout_evaluation": False,
        },
        "settings_hashes": {
            "model_update_config_sha256": _sha256(model_update.CONFIG_PATH),
            "diagnostic_runner_sha256": _sha256(Path(__file__)),
            **{path.relative_to(ROOT).as_posix(): _sha256(path) for path in FEATURES},
        },
        "reference_completed_trials": reference_trials,
        "frozen_identity_sha256": None,
    }


def _write_frozen_manifest() -> dict:
    frozen = _frozen_inputs()
    identity = _canonical_sha256({key: value for key, value in frozen.items() if key not in {
        "status", "prepared_utc", "frozen_identity_sha256",
    }})
    frozen["frozen_identity_sha256"] = identity
    if MANIFEST_PATH.is_file():
        existing = _read_json(MANIFEST_PATH)
        if existing.get("frozen_identity_sha256") != identity:
            existing_results = _read_json(RESULTS_PATH) if RESULTS_PATH.is_file() else {"trials": {}}
            if existing.get("status") not in {"prepared_before_trials", "running"} or existing_results.get("trials"):
                raise RuntimeError("Existing diagnostic manifest has different frozen inputs; refusing to reuse results")
            _write_json(MANIFEST_PATH, frozen)
            return frozen
        return existing
    _write_json(MANIFEST_PATH, frozen)
    return frozen


def _validate_fold_data(fold: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    dataset_path = ROOT / fold["dataset_path"]
    if _sha256(dataset_path) != fold["dataset_sha256"]:
        raise RuntimeError(f"Feature dataset hash changed for {fold['fold_id']}")
    feature_order = fold["feature_order"]
    contract_columns = [
        "Opened", preopen.TARGET_COL, preopen.TARGET_AVAILABLE_COL,
        preopen.DECISION_COL, "feature_candle_close_at", "nominal_decision_at",
        "nominal_prediction_available_at", "actual_prediction_available_at",
        "prediction_deadline_at",
    ]
    frame = pd.read_parquet(dataset_path, columns=[*contract_columns, *feature_order])
    contract = frame.loc[:, contract_columns]
    decision_all = preopen.preopen_decision_mask(contract)
    stored_decision = contract[preopen.DECISION_COL].to_numpy(dtype=bool, copy=False)
    if not np.array_equal(decision_all, stored_decision):
        raise RuntimeError(f"Point-in-time decision mask changed for {fold['fold_id']}")
    opened = pd.DatetimeIndex(pd.to_datetime(contract["Opened"], utc=True, errors="raise"))
    label_available = pd.DatetimeIndex(pd.to_datetime(
        contract[preopen.TARGET_AVAILABLE_COL], utc=True, errors="coerce",
    ))
    fit_end = pd.Timestamp(fold["fit_end_exclusive_utc"])
    fit_start = pd.Timestamp(fold["fit_start_utc"])
    y_full = pd.to_numeric(contract[preopen.TARGET_COL], errors="coerce").to_numpy(
        dtype=np.float64, copy=False,
    )
    fit_mask = (
        np.isfinite(y_full)
        & label_available.notna()
        & (opened >= fit_start)
        & (label_available < fit_end)
    )
    rows = np.flatnonzero(fit_mask)
    if len(rows) != fold["training_rows_expected"]:
        raise RuntimeError(f"Fit-row count differs from the frozen B study for {fold['fold_id']}")
    development_features = frame.loc[:, feature_order].iloc[rows].reset_index(drop=True)
    development_features = development_features.replace([np.inf, -np.inf], np.nan)
    x_np = development_features.to_numpy(dtype=np.float32, copy=False)
    y_np = y_full[rows].astype(np.int8, copy=False)
    development_decision = decision_all[rows]
    if int(development_decision.sum()) != fold["decision_rows_expected"]:
        raise RuntimeError(f"Decision-row count differs from the frozen B study for {fold['fold_id']}")
    sample_weight_np = preopen._preopen_sample_weights(
        development_decision, fold["decision_weight"],
    )
    opened_fit = opened[rows]
    decision_at = pd.DatetimeIndex(pd.to_datetime(
        contract["nominal_decision_at"], utc=True, errors="raise",
    ))[rows]
    folds = preopen._purged_walk_forward_folds(
        model_tuning,
        opened_fit,
        decision_at,
        int(model_update.CONFIG["model_search"]["model_cv_folds"]),
        0.1,
        validation_mask=development_decision,
    )
    fold_indices = model_tuning.build_fold_indices(folds)
    fingerprints = preopen._fold_index_fingerprints(folds)
    if fingerprints != fold["fold_index_fingerprints"]:
        raise RuntimeError(f"Internal CV splits differ from the frozen B study for {fold['fold_id']}")
    weight_sha = preopen._array_sha256(sample_weight_np, dtype=np.float32)
    if weight_sha != fold["training_weight_sha256"]:
        raise RuntimeError(f"Observation-weight hash differs from the frozen B study for {fold['fold_id']}")
    validation_rows = sum(len(valid) for _, valid in fold_indices)
    if validation_rows <= 0:
        raise RuntimeError(f"No internal decision validation rows for {fold['fold_id']}")
    return x_np, y_np, sample_weight_np, folds, {
        "rows": int(len(rows)),
        "decision_rows": int(development_decision.sum()),
        "feature_count": len(feature_order),
        "fold_count": len(folds),
        "fold_indices": fingerprints,
        "training_weight_sha256": weight_sha,
        "maximum_label_available_at_utc": label_available[rows].max().isoformat(),
    }


class _FixedTrial:
    def __init__(self, params: dict):
        self.params = params
        self.study = type("StudyDirection", (), {"direction": optuna.study.StudyDirection.MINIMIZE})()
        self.user_attrs = {}
        self.checkpoints = []

    def suggest_float(self, name, low, high, *, step=None, log=False):
        return float(self.params[name])

    def suggest_int(self, name, low, high, *, step=1, log=False):
        return int(self.params[name])

    def suggest_categorical(self, name, choices):
        return self.params[name]

    def report(self, value, step):
        self.checkpoints.append({
            "step": int(step), "iteration": int(step) + 1, "objective_value": float(value),
        })

    def should_prune(self):
        return False

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


def _configure_tuning() -> None:
    profile = model_update.CONFIG["model_search"]
    runtime = model_update.CONFIG["runtime_profile"]
    model_tuning.CV_FOLDS = int(profile["model_cv_folds"])
    model_tuning.WF_TEST_TO_TRAIN_RATIO = 0.1
    model_tuning.MAX_N_ESTIMATORS = int(profile["maximum_estimators"])
    model_tuning.EARLY_STOPPING_ROUNDS = int(profile["early_stopping_rounds"])
    model_tuning.PRUNE_REPORT_EVERY_N_ITER = 25
    model_tuning.CV_OBJECTIVE_BASE_METRIC = "binary_logloss"
    model_tuning.CV_OBJECTIVE_IS_HIGHER_BETTER = False
    model_tuning.CV_STD_PENALTY = 0.5
    model_tuning.ENABLE_FOLD_RECENCY_WEIGHTING = False
    model_tuning.SEED = int(profile["random_seed"])
    model_tuning.LGBM_DEVICE_TYPE = str(runtime["compute_backend"])
    model_tuning.LGBM_NUM_THREADS = int(runtime["threads"])
    model_tuning.GPU_MAX_BIN_LIMIT = 63


def _run_trial(fold: dict, config: dict, x_np, y_np, weights, folds) -> dict:
    started = time.perf_counter()
    feature_order = fold["feature_order"]
    fold_indices = model_tuning.build_fold_indices(folds)
    fold_weights = model_tuning.build_fold_recency_weights(folds)
    train_set = lgb.Dataset(
        x_np, label=y_np, weight=weights, feature_name=feature_order, free_raw_data=False,
    )
    objective = model_tuning.make_objective(
        train_set=train_set,
        feature_names=feature_order,
        x_np=x_np,
        y_np=y_np,
        sample_weight_np=weights,
        folds=folds,
        fold_indices=fold_indices,
        fold_weight_by_id=fold_weights,
        search_space=model_update.CONFIG["model_search"]["search_space"],
        unweighted_validation=True,
    )
    trial = _FixedTrial(config["params"])
    captured = {}
    original_cv = model_tuning.lgb.cv

    def capture_cv(*args, **kwargs):
        full_curves: dict[str, list[float]] = {}

        def capture_iteration(env):
            for evaluation in env.evaluation_result_list or []:
                if len(evaluation) != 5:
                    continue
                valid_name, metric_name, mean_value, _, std_value = evaluation
                if valid_name not in {"valid", "cv_agg"}:
                    continue
                prefix = "valid" if valid_name == "cv_agg" else valid_name
                full_curves.setdefault(f"{prefix} {metric_name}-mean", []).append(float(mean_value))
                full_curves.setdefault(f"{prefix} {metric_name}-stdv", []).append(float(std_value))

        capture_iteration.order = -100
        kwargs["callbacks"] = [capture_iteration, *kwargs.get("callbacks", [])]
        result = original_cv(*args, **kwargs)
        captured["curves"] = full_curves
        captured["returned_curves"] = {
            key: np.asarray(value, dtype=np.float64).tolist()
            for key, value in result.items()
            if key != "cvbooster"
        }
        return result

    model_tuning.lgb.cv = capture_cv
    try:
        objective_value = float(objective(trial))
    finally:
        model_tuning.lgb.cv = original_cv
    curves = captured.get("curves")
    if not curves:
        raise RuntimeError("LightGBM did not return the full validation curves")
    metric = model_tuning.CV_OBJECTIVE_BASE_METRIC
    mean_curve = curves[f"valid {metric}-mean"]
    std_curve = curves[f"valid {metric}-stdv"]
    objective_curve = [
        mean + model_tuning.CV_STD_PENALTY * std
        for mean, std in zip(mean_curve, std_curve)
    ]
    best_index = int(np.argmin(objective_curve))
    if not math.isclose(objective_value, objective_curve[best_index], rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Returned objective differs from the minimum of its captured CV curve")
    checkpoints = trial.checkpoints
    for checkpoint in checkpoints:
        iteration = checkpoint["iteration"]
        curve_value = objective_curve[iteration - 1] if iteration <= len(objective_curve) else None
        if curve_value is None or not math.isclose(
            checkpoint["objective_value"], curve_value, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Pruner checkpoint does not match the full captured objective curve: "
                f"iteration={iteration} callback={checkpoint['objective_value']!r} "
                f"curve={curve_value!r} delta="
                f"{abs(checkpoint['objective_value'] - curve_value) if curve_value is not None else 'n/a'}"
            )
    return {
        "fold_id": fold["fold_id"],
        "config_id": config["config_id"],
        "source_trial_number": config["source_trial_number"],
        "learning_rate": config["params"]["learning_rate"],
        "params": config["params"],
        "source_pruned_at_iteration": config["source_pruned_at_iteration"],
        "source_trial_checkpoint_value": config["source_pruning_checkpoint_value"],
        "duration_seconds": float(time.perf_counter() - started),
        "best_iteration": best_index + 1,
        "best_objective": objective_value,
        "best_binary_logloss_mean": float(mean_curve[best_index]),
        "best_binary_logloss_std": float(std_curve[best_index]),
        "checkpoint_values": checkpoints,
        "validation_curve": curves,
        "best_iteration_curve": objective_curve,
        "trial_attributes": trial.user_attrs,
        "status": "completed",
    }


def _reference_by_fold(manifest: dict) -> dict[str, dict]:
    return {fold["fold_id"]: fold for fold in manifest["folds"]}


def _write_report(manifest: dict, results: dict, elapsed_seconds: float) -> None:
    reference = _reference_by_fold(manifest)
    lines = [
        "# BTC pre-open: development pruning continuation diagnostic",
        "",
        "> This is a small diagnostic continuation on the three existing development B folds. It is not a new independent OOS test, does not score the holdout, and does not select a model or policy.",
        "",
        "## Frozen procedure and provenance",
        "",
        f"- Frozen manifest: `{MANIFEST_PATH.relative_to(ROOT).as_posix()}` (identity `{manifest['frozen_identity_sha256']}`).",
        f"- The five parameter sets are the already-frozen PRUNED trials {', '.join(str(item['source_trial_number']) for item in manifest['source_config_csv']['configurations'])} from the final-fit B study. That study's objective labels ended by `{manifest['source_parameter_study']['source_fit_label_available_through_utc']}`; its study contract excludes holdout labels.",
        f"- Each development fold uses its own B feature matrix, fit boundary, target-weight vector, and saved ten-fold purged chronological CV splits. The observed input checks match the saved tuning split and training-weight hashes.",
        f"- Metric: unweighted actual-decision-row binary log loss; the minimized objective is mean plus 0.5 times the CV standard deviation. Maximum {manifest['procedure']['maximum_iterations']} iterations, early stopping {manifest['procedure']['early_stopping_rounds']}, seed {manifest['procedure']['random_seed']}, backend {manifest['procedure']['compute_backend']}, {manifest['procedure']['threads']} threads.",
        f"- Existing pruner checkpoints: iterations {', '.join(str(value) for value in manifest['procedure']['pruner_checkpoint_iterations'])}; Optuna pruning was disabled for these fixed runs while its reporting callback remained active. No TPE suggestions, feature selection, generator fitting, or holdout evaluation ran.",
        "- No hypothetical MedianPruner decisions are replayed, so no false-pruning percentage is estimated. The comparison is against completed configs from each same development B study.",
        "",
        "## Continued configurations",
        "",
        "| Fold | Config | Source trial | Learning rate | Source prune iteration | Source / continued objective at checkpoints | Continued best objective | Best iteration | Best binary log loss mean | Time (s) | Same-study COMPLETE best / median | Delta vs best |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fold_id in FOLDS:
        for config in manifest["source_config_csv"]["configurations"]:
            row = results["trials"].get(f"{fold_id}/{config['config_id']}")
            if not row:
                lines.append(f"| {fold_id} | {config['config_id']} | {config['source_trial_number']} | n/a | {config['source_pruned_at_iteration']} | n/a | INCOMPLETE | n/a | n/a | n/a | n/a | n/a |")
                continue
            continued_checkpoints = {
                point["iteration"]: point["objective_value"]
                for point in row["checkpoint_values"]
            }
            checkpoints = ", ".join(
                f"{point['iteration']}:{point['value']:.6f}/"
                f"{continued_checkpoints[point['iteration']]:.6f}"
                if point["iteration"] in continued_checkpoints
                else f"{point['iteration']}:{point['value']:.6f}/n/a"
                for point in config["source_intermediate_curve"]
            )
            completed_best = reference[fold_id]["reference_completed_best_objective"]
            completed_median = reference[fold_id]["reference_completed_median_objective"]
            delta = row["best_objective"] - completed_best
            lines.append(
                f"| {fold_id} | {config['config_id']} | {config['source_trial_number']} | {row['learning_rate']:.6f} | {row['source_pruned_at_iteration']} | {checkpoints} | {row['best_objective']:.6f} | {row['best_iteration']} | {row['best_binary_logloss_mean']:.6f} | {row['duration_seconds']:.1f} | {completed_best:.6f} / {completed_median:.6f} ({reference[fold_id]['reference_completed_count']} COMPLETE) | {delta:+.6f} |"
            )
    completed = [row for row in results["trials"].values() if row.get("status") == "completed"]
    competitive = [
        row for row in completed
        if row["best_objective"] <= reference[row["fold_id"]]["reference_completed_best_objective"]
    ]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Continued runs completed: {len(completed)}/15. Of those, {len(competitive)} reached or beat the best COMPLETE configuration in their same-fold development study by the original CV objective.",
        "- This verdict applies only to these five parameter sets on the listed development CV folds. The sampled continuations can show that pruned parameter sets may later be competitive on development CV; they do not estimate how often the MedianPruner wrongly prunes, since these source decisions came from another chronological fit and are not replayed here.",
        "- Full available per-iteration validation metric curves and exact checkpoint values are in `pruner_diagnostic_results.json`; the completed reference configurations are in `completed_development_trials.csv`.",
        "",
        "## Runtime and reproduction",
        "",
        f"- Measured end-to-end runtime including input verification and data loading: {elapsed_seconds:.1f}s ({elapsed_seconds / 60:.2f} min); model configuration trials: {sum(row['duration_seconds'] for row in completed):.1f}s across {len(completed)} completed configurations.",
        f"- Resume from saved config checkpoints without CLI arguments: `python {Path(__file__).name}`.",
        "- The runner skips completed fold/configuration pairs and writes each completed curve atomically. An interrupted, incomplete configuration may be rerun; completed configurations are not repeated.",
        "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    run_started = time.perf_counter()
    manifest = _write_frozen_manifest()
    print(
        f"[pruner-diagnostic] frozen manifest={MANIFEST_PATH.relative_to(ROOT).as_posix()} "
        f"configurations={len(manifest['source_config_csv']['configurations'])} folds={len(FOLDS)}",
        flush=True,
    )
    if RESULTS_PATH.is_file():
        results = _read_json(RESULTS_PATH)
        if results.get("frozen_identity_sha256") != manifest["frozen_identity_sha256"]:
            if results.get("trials"):
                raise RuntimeError("Diagnostic checkpoint belongs to different frozen inputs")
            results = {"frozen_identity_sha256": manifest["frozen_identity_sha256"], "trials": {}}
            _write_json(RESULTS_PATH, results)
    else:
        results = {
            "frozen_identity_sha256": manifest["frozen_identity_sha256"],
            "trials": {},
        }
        _write_json(RESULTS_PATH, results)

    _configure_tuning()
    total_trials = len(FOLDS) * len(manifest["source_config_csv"]["configurations"])
    for fold in manifest["folds"]:
        fold_id = fold["fold_id"]
        pending = [
            config for config in manifest["source_config_csv"]["configurations"]
            if f"{fold_id}/{config['config_id']}" not in results["trials"]
        ]
        if not pending:
            continue
        x_np, y_np, weights, folds, observed = _validate_fold_data(fold)
        fold["verified_inputs"] = observed
        fold["verified_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "running"
        _write_json(MANIFEST_PATH, manifest)
        for config in pending:
            key = f"{fold_id}/{config['config_id']}"
            print(
                f"[pruner-diagnostic] start {key} lr={config['params']['learning_rate']:.6f} "
                f"remaining={total_trials - len(results['trials'])}/{total_trials}",
                flush=True,
            )
            row = _run_trial(fold, config, x_np, y_np, weights, folds)
            reference = next(item for item in manifest["folds"] if item["fold_id"] == fold_id)
            row["same_study_reference"] = {
                "completed_count": reference["reference_completed_count"],
                "best_objective": reference["reference_completed_best_objective"],
                "median_objective": reference["reference_completed_median_objective"],
                "delta_vs_best": row["best_objective"] - reference["reference_completed_best_objective"],
                "reached_or_beat_best": row["best_objective"] <= reference["reference_completed_best_objective"],
            }
            results["trials"][key] = row
            results["last_completed_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(RESULTS_PATH, results)
            done = len(results["trials"])
            durations = [value["duration_seconds"] for value in results["trials"].values()]
            average = float(np.mean(durations)) if durations else 0.0
            eta = max(0, total_trials - done) * average
            print(
                f"[pruner-diagnostic] done {key} best={row['best_objective']:.6f} "
                f"iteration={row['best_iteration']} elapsed={row['duration_seconds']:.1f}s "
                f"progress={done}/{total_trials} mean_trial={average:.1f}s ETA={eta / 60:.1f}m",
                flush=True,
            )
            _write_report(manifest, results, time.perf_counter() - run_started)
        del x_np, y_np, weights, folds
        import gc

        gc.collect()

    manifest["status"] = "completed" if len(results["trials"]) == total_trials else "incomplete_resumable"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["elapsed_seconds"] = float(time.perf_counter() - run_started)
    _write_json(MANIFEST_PATH, manifest)
    results["elapsed_seconds"] = manifest["elapsed_seconds"]
    _write_json(RESULTS_PATH, results)
    pd.DataFrame(manifest["reference_completed_trials"]).to_csv(REFERENCE_TRIALS_PATH, index=False)
    _write_report(manifest, results, manifest["elapsed_seconds"])
    print(
        f"[pruner-diagnostic] status={manifest['status']} completed={len(results['trials'])}/{total_trials} "
        f"elapsed={manifest['elapsed_seconds'] / 60:.2f}m report={REPORT_PATH.relative_to(ROOT).as_posix()}",
        flush=True,
    )
    return {"manifest": manifest, "results": results}


if __name__ == "__main__":
    run()
