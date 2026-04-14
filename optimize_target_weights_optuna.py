import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from common_config_utils import path_to_portable_str
from modeling_dataset_utils import (
    load_excluded_feature_names_from_settings,
    load_feature_subset_from_settings,
    load_modeling_dataset_settings,
    resolve_modeling_dataset_output_paths,
    resolve_modeling_float_dtype,
    resolve_modeling_float_dtype_name,
    summarize_feature_subset,
)
from optuna_run_utils import make_timestamped_artifact_path, resolve_run_study_name
from target_weights import TARGET_WEIGHT_COL, compute_decision_mask_from_opened, summarize_target_weights
from train_lgbm import (
    CV_FOLDS as DEFAULT_CV_FOLDS,
    EARLY_STOPPING_ROUNDS,
    LGBM_DEFAULT_PARAMS,
    LGBM_OPTUNA_BEST_PARAMS,
    N_ESTIMATORS,
    TARGET_COL,
    WF_TEST_TO_TRAIN_RATIO as DEFAULT_WF_TEST_TO_TRAIN_RATIO,
    classification_metrics,
    evaluate_walk_forward_variant,
    load_walk_forward_training_frame,
    make_walk_forward_folds,
)

DECISION_ROW_OBJECTIVE_METRIC = "balanced_accuracy"
DECISION_ROW_STD_PENALTY = 1.0
DEFAULT_STUDY_NAME_PREFIX = "lgbm_target_weights_decision_rows_balanced_accuracy"
DEFAULT_STORAGE = "sqlite:///data/optuna/databases/lgbm_target_weights_tpe_gpu.db"
DEFAULT_OUTPUT_DIR = Path("data/optuna/target_weights")
BEST_RESULT_STEM = "lgbm_target_weights_optuna_best_decision_rows_balanced_accuracy"
TRIALS_CSV_STEM = "lgbm_target_weights_optuna_trials_decision_rows_balanced_accuracy"
BEST_OOF_STEM = "lgbm_target_weights_optuna_best_oof"
SEED = 37
OPTUNA_N_JOBS = 1
DECISION_WEIGHT_LOW = 0.20
DECISION_WEIGHT_HIGH = 0.999
ENABLE_FOLD_RECENCY_WEIGHTING = True
FOLD_RECENCY_WEIGHTING_MODE = "linear"
FOLD_RECENCY_WEIGHT_MIN = 1.0
FOLD_RECENCY_WEIGHT_MAX = 1.4


def is_nontrivial_fold_recency_weighting_enabled():
    return bool(ENABLE_FOLD_RECENCY_WEIGHTING) and not np.isclose(
        float(FOLD_RECENCY_WEIGHT_MIN),
        float(FOLD_RECENCY_WEIGHT_MAX),
    )


def build_fold_recency_weights(folds):
    if not folds:
        raise ValueError("Fold recency weights require at least one fold.")

    fold_ids = [int(fold["fold_id"]) for fold in folds]
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("Fold ids must be unique to build recency weights.")

    min_weight = float(FOLD_RECENCY_WEIGHT_MIN)
    max_weight = float(FOLD_RECENCY_WEIGHT_MAX)
    if min_weight <= 0.0 or max_weight <= 0.0:
        raise ValueError("Fold recency weights must be strictly positive.")
    if max_weight < min_weight:
        raise ValueError(
            "FOLD_RECENCY_WEIGHT_MAX must be >= FOLD_RECENCY_WEIGHT_MIN."
        )

    if len(folds) == 1 or not bool(ENABLE_FOLD_RECENCY_WEIGHTING):
        weights = np.ones(len(fold_ids), dtype=np.float64)
    else:
        mode = str(FOLD_RECENCY_WEIGHTING_MODE).strip().lower()
        if mode == "linear":
            weights = np.linspace(
                min_weight,
                max_weight,
                num=len(fold_ids),
                dtype=np.float64,
            )
        else:
            raise ValueError(
                f"Unsupported FOLD_RECENCY_WEIGHTING_MODE: {FOLD_RECENCY_WEIGHTING_MODE}"
            )

    return pd.Series(weights, index=fold_ids, name="fold_weight", dtype=np.float64)


def resolve_fold_weight_array(folds, fold_weight_by_id):
    if fold_weight_by_id is None:
        raise ValueError("fold_weight_by_id is required.")

    fold_ids = [int(fold["fold_id"]) for fold in folds]
    resolved = pd.Series(fold_weight_by_id, dtype=np.float64).reindex(fold_ids)
    if resolved.isna().any():
        missing_fold_ids = resolved.index[resolved.isna()].tolist()
        raise ValueError(f"Missing fold weights for fold ids: {missing_fold_ids}")

    weights = resolved.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(weights).all():
        raise ValueError("Fold weights contain non-finite values.")
    if np.any(weights <= 0.0):
        raise ValueError("Fold weights must be strictly positive.")
    return weights


def weighted_mean_vector(values, weights):
    values_arr = np.asarray(values, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    if values_arr.ndim != 1:
        raise ValueError("weighted_mean_vector expects a 1D array.")
    if values_arr.shape[0] != weights_arr.shape[0]:
        raise ValueError(
            "weighted_mean_vector length mismatch: "
            f"{values_arr.shape[0]} != {weights_arr.shape[0]}"
        )
    return float(np.average(values_arr, weights=weights_arr))


def fold_weight_items_for_summary(folds, fold_weight_by_id):
    weights = resolve_fold_weight_array(folds, fold_weight_by_id)
    return [
        {
            "fold_id": int(fold["fold_id"]),
            "weight": float(weight),
        }
        for fold, weight in zip(folds, weights)
    ]


def decision_objective_base_score_label():
    if is_nontrivial_fold_recency_weighting_enabled():
        return f"{DECISION_ROW_OBJECTIVE_METRIC}_weighted_mean"
    return f"{DECISION_ROW_OBJECTIVE_METRIC}_mean"


def resolve_decision_objective_name():
    return f"decision_rows_{decision_objective_base_score_label()}_minus_std_penalty"


def decision_objective_aggregation_description():
    return (
        f"fold_{decision_objective_base_score_label()} - std_penalty * "
        f"fold_{DECISION_ROW_OBJECTIVE_METRIC}_std"
    )


def summarize_decision_fold_metric_scores(
    metric_values,
    folds,
    fold_weight_by_id,
    *,
    std_penalty,
):
    metric_values_arr = np.asarray(metric_values, dtype=np.float64)
    if metric_values_arr.ndim != 1:
        raise ValueError("metric_values must be a 1D array.")
    if metric_values_arr.shape[0] != len(folds):
        raise ValueError(
            "metric_values length mismatch: "
            f"{metric_values_arr.shape[0]} != {len(folds)}"
        )

    metric_mean = float(np.mean(metric_values_arr))
    metric_weighted_mean = weighted_mean_vector(
        metric_values_arr,
        resolve_fold_weight_array(folds, fold_weight_by_id),
    )
    metric_std = float(np.std(metric_values_arr, ddof=0))
    objective_base_value = (
        metric_weighted_mean
        if is_nontrivial_fold_recency_weighting_enabled()
        else metric_mean
    )
    return {
        "decision_metric_mean": metric_mean,
        "decision_metric_weighted_mean": metric_weighted_mean,
        "decision_metric_std": metric_std,
        "decision_metric_base_value": float(objective_base_value),
        "decision_metric_objective": float(
            objective_base_value - (float(std_penalty) * metric_std)
        ),
    }


def build_cli_args():
    parser = argparse.ArgumentParser(
        description=(
            "Optuna optimization of target row weights for LGBM. "
            "Training uses all rows, but the objective is evaluated only on decision rows."
        )
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Optional Optuna timeout in seconds.",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Optional Optuna study name. Defaults to an auto-generated timestamped name.",
    )
    parser.add_argument(
        "--storage",
        default=DEFAULT_STORAGE,
        help="Optuna storage URL.",
    )
    parser.add_argument(
        "--load-if-exists",
        action="store_true",
        help="Resume an existing Optuna study if it already exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON/CSV artifacts.",
    )
    parser.add_argument(
        "--params-source",
        choices=("optuna", "default"),
        default="optuna",
        help="Base LGBM parameter set to use during weight optimization.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help="Walk-forward fold count.",
    )
    parser.add_argument(
        "--test-to-train-ratio",
        type=float,
        default=DEFAULT_WF_TEST_TO_TRAIN_RATIO,
        help="Walk-forward test/train ratio.",
    )
    parser.add_argument(
        "--device-type",
        choices=("gpu", "cpu"),
        default="gpu",
        help="LightGBM device type.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Optional override for LightGBM thread count.",
    )
    parser.add_argument(
        "--decision-weight-low",
        type=float,
        default=DECISION_WEIGHT_LOW,
        help="Lower bound for decision row weight.",
    )
    parser.add_argument(
        "--decision-weight-high",
        type=float,
        default=DECISION_WEIGHT_HIGH,
        help="Upper bound for decision row weight. "
        "Non-decision rows share the remaining block mass equally.",
    )
    parser.add_argument(
        "--decision-weight-step",
        type=float,
        default=None,
        help="Optional Optuna step for decision_weight. Omit for continuous search.",
    )
    parser.add_argument(
        "--objective-std-penalty",
        type=float,
        default=DECISION_ROW_STD_PENALTY,
        help="Objective = mean(metric) - std_penalty * std(metric) across folds.",
    )
    parser.add_argument(
        "--save-best-oof",
        action="store_true",
        help="Re-run the best weight configuration at the end and save its OOF predictions.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.n_trials < 1:
        raise ValueError("--n-trials must be >= 1.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")
    if not (0.0 < float(args.test_to_train_ratio) < 1.0):
        raise ValueError("--test-to-train-ratio must be in (0, 1).")
    if float(args.objective_std_penalty) < 0.0:
        raise ValueError("--objective-std-penalty must be >= 0.")
    low = float(args.decision_weight_low)
    high = float(args.decision_weight_high)
    if not (0.0 < low < 1.0):
        raise ValueError("--decision-weight-low must be in (0, 1).")
    if not (0.0 < high < 1.0):
        raise ValueError("--decision-weight-high must be in (0, 1).")
    if low >= high:
        raise ValueError("--decision-weight-low must be < --decision-weight-high.")
    if args.decision_weight_step is not None and float(args.decision_weight_step) <= 0.0:
        raise ValueError("--decision-weight-step must be > 0 when provided.")


def build_model_param_overrides(args):
    if args.params_source == "optuna":
        params = dict(LGBM_OPTUNA_BEST_PARAMS)
    else:
        params = dict(LGBM_DEFAULT_PARAMS)
    params["device_type"] = str(args.device_type)
    if args.n_jobs is not None:
        params["n_jobs"] = int(args.n_jobs)
    return params


def build_weight_config(decision_weight, *, other_rows_per_block=4):
    decision_weight_value = float(decision_weight)
    if not np.isfinite(decision_weight_value):
        raise ValueError("decision_weight must be finite.")
    if not (0.0 < decision_weight_value < 1.0):
        raise ValueError("decision_weight must be in (0, 1).")
    other_rows = int(other_rows_per_block)
    if other_rows <= 0:
        raise ValueError("other_rows_per_block must be >= 1.")

    remaining_weight = 1.0 - decision_weight_value
    other_weight_value = remaining_weight / other_rows
    if not np.isfinite(other_weight_value) or other_weight_value <= 0.0:
        raise ValueError(
            "Derived non-decision row weight must be finite and > 0. "
            f"Got {other_weight_value!r}."
        )

    return {
        "decision_weight": decision_weight_value,
        "other_weight": other_weight_value,
        "other_rows_per_block": other_rows,
        "total_block_weight": float(
            decision_weight_value + (other_rows * other_weight_value)
        ),
        "non_decision_total_weight": float(other_rows * other_weight_value),
    }


def build_sample_weight_series(decision_mask, weight_config, *, index, float_dtype):
    decision_mask_np = np.asarray(decision_mask, dtype=bool)
    weight_values = np.where(
        decision_mask_np,
        float(weight_config["decision_weight"]),
        float(weight_config["other_weight"]),
    ).astype(float_dtype, copy=False)
    if not np.isfinite(weight_values).all():
        raise ValueError("Constructed sample weights contain non-finite values.")
    if np.any(weight_values <= 0.0):
        raise ValueError("Constructed sample weights must be strictly positive.")
    return pd.Series(weight_values, index=index, name=TARGET_WEIGHT_COL)


def compute_decision_row_fold_metrics(*, y, oof_pred_proba, decision_mask, folds):
    y_np = np.asarray(y, dtype=np.int8)
    oof_pred_np = np.asarray(oof_pred_proba, dtype=np.float64)
    decision_mask_np = np.asarray(decision_mask, dtype=bool)
    fold_rows = []

    for fold in folds:
        te_s = int(fold["test_start"])
        te_e = int(fold["test_end"])
        fold_pred = oof_pred_np[te_s:te_e]
        fold_y = y_np[te_s:te_e]
        fold_decision_mask = decision_mask_np[te_s:te_e]
        valid_mask = np.isfinite(fold_pred) & fold_decision_mask
        if not np.any(valid_mask):
            raise RuntimeError(
                f"Fold {fold['fold_id']} produced no valid decision-row OOF predictions."
            )

        fold_metrics = classification_metrics(
            fold_y[valid_mask],
            fold_pred[valid_mask],
            sample_weight=None,
        )
        fold_rows.append(
            {
                "fold_id": int(fold["fold_id"]),
                "decision_rows": int(valid_mask.sum()),
                **{metric_name: float(metric_value) for metric_name, metric_value in fold_metrics.items()},
            }
        )

    return pd.DataFrame(fold_rows)


def evaluate_weight_config(
    *,
    x,
    y,
    decision_mask,
    folds,
    fold_weight_by_id,
    param_overrides,
    weight_config,
    std_penalty,
    float_dtype,
    model_variant,
):
    sample_weight = build_sample_weight_series(
        decision_mask,
        weight_config,
        index=y.index,
        float_dtype=float_dtype,
    )
    cv_result, oof_pred_proba, _ = evaluate_walk_forward_variant(
        x=x,
        y=y,
        sample_weight=sample_weight,
        folds=folds,
        param_overrides=param_overrides,
        model_variant=model_variant,
        collect_oof_predictions=True,
        collect_feature_importance=False,
        early_stopping_verbose=False,
        float_dtype=float_dtype,
    )
    if oof_pred_proba is None:
        raise RuntimeError("OOF predictions were not collected.")

    decision_mask_np = np.asarray(decision_mask, dtype=bool)
    oof_mask = np.isfinite(oof_pred_proba)
    decision_oof_mask = oof_mask & decision_mask_np
    if not np.any(decision_oof_mask):
        raise RuntimeError("No OOF decision-row predictions were produced.")

    y_np = y.to_numpy(dtype=np.int8, copy=False)
    decision_oof_metrics = classification_metrics(
        y_np[decision_oof_mask],
        np.asarray(oof_pred_proba, dtype=np.float64)[decision_oof_mask],
        sample_weight=None,
    )
    decision_fold_metrics = compute_decision_row_fold_metrics(
        y=y_np,
        oof_pred_proba=oof_pred_proba,
        decision_mask=decision_mask_np,
        folds=folds,
    )
    objective_metric_values = decision_fold_metrics[DECISION_ROW_OBJECTIVE_METRIC].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    decision_metric_summary = summarize_decision_fold_metric_scores(
        objective_metric_values,
        folds=folds,
        fold_weight_by_id=fold_weight_by_id,
        std_penalty=std_penalty,
    )

    return {
        "sample_weight": sample_weight,
        "sample_weight_summary": summarize_target_weights(
            sample_weight.to_numpy(dtype=np.float64, copy=False)
        ),
        "cv_result": cv_result,
        "oof_pred_proba": np.asarray(oof_pred_proba, dtype=np.float64),
        "decision_oof_metrics": decision_oof_metrics,
        "decision_fold_metrics": decision_fold_metrics,
        "folds": list(folds),
        "fold_weight_by_id": fold_weight_by_id.copy(),
        **decision_metric_summary,
    }
def suggest_weight_config(trial, args):
    suggest_kwargs = {
        "name": "decision_weight",
        "low": float(args.decision_weight_low),
        "high": float(args.decision_weight_high),
    }
    if args.decision_weight_step is not None:
        suggest_kwargs["step"] = float(args.decision_weight_step)
    decision_weight = trial.suggest_float(**suggest_kwargs)
    return build_weight_config(decision_weight)


def enqueue_seed_trials(study, args):
    candidate_weights = [0.20, 0.50, 0.80, 0.90, 0.95, 0.98]
    low = float(args.decision_weight_low)
    high = float(args.decision_weight_high)
    step = None if args.decision_weight_step is None else float(args.decision_weight_step)

    for candidate in candidate_weights:
        if candidate < low or candidate > high:
            continue
        if step is not None:
            snapped = low + (round((candidate - low) / step) * step)
            candidate = float(min(high, max(low, snapped)))
        study.enqueue_trial({"decision_weight": float(candidate)})


def build_trials_dataframe(study):
    rows = []
    for trial in study.get_trials(deepcopy=False):
        if trial.value is None:
            objective_value = None
        else:
            objective_value = float(trial.value)
        row = {
            "trial_number": int(trial.number),
            "state": str(trial.state.name),
            "objective_value": objective_value,
            "decision_weight": trial.params.get("decision_weight"),
            "other_weight": trial.user_attrs.get("other_weight"),
            "decision_rows_balanced_accuracy_mean": trial.user_attrs.get(
                "decision_rows_balanced_accuracy_mean"
            ),
            "decision_rows_balanced_accuracy_weighted_mean": trial.user_attrs.get(
                "decision_rows_balanced_accuracy_weighted_mean"
            ),
            "decision_rows_balanced_accuracy_std": trial.user_attrs.get(
                "decision_rows_balanced_accuracy_std"
            ),
            "decision_rows_oof_brier": trial.user_attrs.get("decision_rows_oof_brier"),
            "decision_rows_oof_logloss": trial.user_attrs.get("decision_rows_oof_logloss"),
            "decision_rows_oof_accuracy": trial.user_attrs.get("decision_rows_oof_accuracy"),
            "decision_rows_oof_balanced_accuracy": trial.user_attrs.get(
                "decision_rows_oof_balanced_accuracy"
            ),
            "decision_rows_oof_f1": trial.user_attrs.get("decision_rows_oof_f1"),
            "decision_rows_oof_precision": trial.user_attrs.get("decision_rows_oof_precision"),
            "decision_rows_oof_recall": trial.user_attrs.get("decision_rows_oof_recall"),
            "decision_rows_total": trial.user_attrs.get("decision_rows_total"),
            "mean_best_iteration": trial.user_attrs.get("mean_best_iteration"),
            "params_source": trial.user_attrs.get("params_source"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_best_result_payload(
    *,
    args,
    run_info,
    data_path,
    feature_subset,
    excluded_features,
    class_distribution,
    decision_mask,
    best_trial,
    best_result,
):
    decision_mask_np = np.asarray(decision_mask, dtype=bool)
    best_weight_config = build_weight_config(best_trial.params["decision_weight"])
    return {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "study_name": run_info["study_name"],
        "study_name_source": run_info["study_name_source"],
        "storage": str(args.storage),
        "data_path": path_to_portable_str(data_path),
        "target_col": TARGET_COL,
        "sample_weight_col": TARGET_WEIGHT_COL,
        "decision_row_definition": {
            "expression": "Opened.minute % 5 == 4",
            "rows": int(decision_mask_np.sum()),
            "total_rows": int(decision_mask_np.size),
            "share": float(decision_mask_np.mean()),
        },
        "feature_selection": summarize_feature_subset(
            feature_subset,
            excluded_features=excluded_features,
        ),
        "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
        "train_pipeline_alignment": {
            "source_script": "train_lgbm.py",
            "cv_folds": int(args.cv_folds),
            "walk_forward_test_to_train_ratio": float(args.test_to_train_ratio),
            "params_source": str(args.params_source),
            "n_estimators": int(N_ESTIMATORS),
            "early_stopping_rounds": int(EARLY_STOPPING_ROUNDS),
            "device_type": str(args.device_type),
        },
        "weight_search_space": {
            "parameterization": (
                "decision_weight in (0,1); "
                "other_weight = (1 - decision_weight) / 4"
            ),
            "decision_weight_low": float(args.decision_weight_low),
            "decision_weight_high": float(args.decision_weight_high),
            "decision_weight_step": (
                None if args.decision_weight_step is None else float(args.decision_weight_step)
            ),
            "objective_metric": DECISION_ROW_OBJECTIVE_METRIC,
            "objective_scope": "decision_rows_only",
            "objective_aggregation": decision_objective_aggregation_description(),
            "objective_std_penalty": float(args.objective_std_penalty),
        },
        "fold_recency_weighting": {
            "enabled": bool(ENABLE_FOLD_RECENCY_WEIGHTING),
            "active": bool(is_nontrivial_fold_recency_weighting_enabled()),
            "mode": str(FOLD_RECENCY_WEIGHTING_MODE),
            "min_weight": float(FOLD_RECENCY_WEIGHT_MIN),
            "max_weight": float(FOLD_RECENCY_WEIGHT_MAX),
            "std_score_aggregation": "unweighted",
            "fold_weights": fold_weight_items_for_summary(
                best_result["folds"],
                best_result["fold_weight_by_id"],
            ),
        },
        "best_trial": {
            "trial_number": int(best_trial.number),
            "objective_value": float(best_trial.value),
            "decision_weight": float(best_weight_config["decision_weight"]),
            "other_weight": float(best_weight_config["other_weight"]),
            "non_decision_total_weight": float(best_weight_config["non_decision_total_weight"]),
            "total_block_weight": float(best_weight_config["total_block_weight"]),
            "decision_rows_balanced_accuracy_mean": float(best_result["decision_metric_mean"]),
            "decision_rows_balanced_accuracy_weighted_mean": float(
                best_result["decision_metric_weighted_mean"]
            ),
            "decision_rows_balanced_accuracy_std": float(best_result["decision_metric_std"]),
            "decision_rows_objective_base_value": float(
                best_result["decision_metric_base_value"]
            ),
            "decision_rows_oof_metrics": {
                metric_name: float(metric_value)
                for metric_name, metric_value in best_result["decision_oof_metrics"].items()
            },
            "mean_best_iteration": int(best_result["cv_result"]["mean_best_iteration"]),
            "sample_weight_summary": best_result["sample_weight_summary"],
        },
    }


def save_best_oof_predictions(
    *,
    output_path,
    df,
    decision_mask,
    best_result,
):
    export_df = df.loc[:, [col for col in ("Opened", "Open", "High", "Low", "Close", "Volume", TARGET_COL) if col in df.columns]].copy()
    export_df["decision_row"] = np.asarray(decision_mask, dtype=np.int8)
    export_df[TARGET_WEIGHT_COL] = best_result["sample_weight"].to_numpy(dtype=np.float64, copy=False)
    export_df["oof_pred_proba_up"] = best_result["oof_pred_proba"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_parquet(output_path, index=False)
    return export_df


def main():
    args = build_cli_args()
    validate_args(args)

    dataset_settings = load_modeling_dataset_settings()
    modeling_float_dtype = resolve_modeling_float_dtype(dataset_settings)
    modeling_float_dtype_name = resolve_modeling_float_dtype_name(dataset_settings)
    data_path = resolve_modeling_dataset_output_paths(dataset_settings)["parquet"]
    feature_subset = load_feature_subset_from_settings(dataset_settings)
    excluded_features = load_excluded_feature_names_from_settings(dataset_settings)

    training_data = load_walk_forward_training_frame(
        data_path=data_path,
        feature_subset=feature_subset,
        excluded_features=excluded_features,
        float_dtype=modeling_float_dtype,
    )
    df = training_data["df"].reset_index(drop=True)
    x = training_data["x"].reset_index(drop=True)
    y = training_data["y"].reset_index(drop=True)
    class_distribution = training_data["class_distribution"]
    decision_mask = compute_decision_mask_from_opened(df["Opened"])
    folds = make_walk_forward_folds(
        n_rows=len(x),
        n_folds=int(args.cv_folds),
        test_to_train_ratio=float(args.test_to_train_ratio),
    )
    fold_weight_by_id = build_fold_recency_weights(folds)
    param_overrides = build_model_param_overrides(args)

    run_info = resolve_run_study_name(
        args.study_name,
        default_prefix=DEFAULT_STUDY_NAME_PREFIX,
    )
    best_result_path = make_timestamped_artifact_path(
        args.output_dir,
        stem=BEST_RESULT_STEM,
        suffix=".json",
        timestamp=run_info["run_timestamp"],
    )
    trials_csv_path = make_timestamped_artifact_path(
        args.output_dir,
        stem=TRIALS_CSV_STEM,
        suffix=".csv",
        timestamp=run_info["run_timestamp"],
    )
    best_oof_path = make_timestamped_artifact_path(
        args.output_dir,
        stem=BEST_OOF_STEM,
        suffix=".parquet",
        timestamp=run_info["run_timestamp"],
    )

    print(
        f"start optimize_target_weights | rows={len(df)} features={x.shape[1]} "
        f"decision_rows={int(np.asarray(decision_mask, dtype=bool).sum())} "
        f"cv_folds={len(folds)} params_source={args.params_source} "
        f"objective={resolve_decision_objective_name()} "
        f"float_precision={modeling_float_dtype_name}"
    )
    print(
        "start optimize_target_weights | "
        f"decision_weight_range=({float(args.decision_weight_low):.6f}, "
        f"{float(args.decision_weight_high):.6f}) "
        f"decision_weight_step={args.decision_weight_step} "
        f"std_penalty={float(args.objective_std_penalty):.4f} "
        f"device_type={args.device_type}"
    )
    print(
        "start optimize_target_weights | "
        f"fold weighting | enabled={bool(ENABLE_FOLD_RECENCY_WEIGHTING)} "
        f"active={is_nontrivial_fold_recency_weighting_enabled()} "
        f"mode={FOLD_RECENCY_WEIGHTING_MODE} "
        f"min={float(FOLD_RECENCY_WEIGHT_MIN):.4f} "
        f"max={float(FOLD_RECENCY_WEIGHT_MAX):.4f} "
        f"std=unweighted"
    )

    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        multivariate=False,
    )
    study = optuna.create_study(
        study_name=run_info["study_name"],
        storage=str(args.storage),
        direction="maximize",
        sampler=sampler,
        load_if_exists=bool(args.load_if_exists),
    )
    enqueue_seed_trials(study, args)

    def objective(trial):
        weight_config = suggest_weight_config(trial, args)
        result = evaluate_weight_config(
            x=x,
            y=y,
            decision_mask=decision_mask,
            folds=folds,
            fold_weight_by_id=fold_weight_by_id,
            param_overrides=param_overrides,
            weight_config=weight_config,
            std_penalty=float(args.objective_std_penalty),
            float_dtype=modeling_float_dtype,
            model_variant=f"trial_{trial.number}",
        )
        objective_value = float(result["decision_metric_objective"])

        trial.set_user_attr("decision_weight", float(weight_config["decision_weight"]))
        trial.set_user_attr("other_weight", float(weight_config["other_weight"]))
        trial.set_user_attr(
            "non_decision_total_weight",
            float(weight_config["non_decision_total_weight"]),
        )
        trial.set_user_attr(
            "decision_rows_balanced_accuracy_mean",
            float(result["decision_metric_mean"]),
        )
        trial.set_user_attr(
            "decision_rows_balanced_accuracy_std",
            float(result["decision_metric_std"]),
        )
        trial.set_user_attr(
            "decision_rows_balanced_accuracy_weighted_mean",
            float(result["decision_metric_weighted_mean"]),
        )
        trial.set_user_attr(
            "decision_rows_oof_brier",
            float(result["decision_oof_metrics"]["brier_score"]),
        )
        trial.set_user_attr(
            "decision_rows_oof_logloss",
            float(result["decision_oof_metrics"]["binary_logloss"]),
        )
        trial.set_user_attr(
            "decision_rows_oof_accuracy",
            float(result["decision_oof_metrics"]["accuracy"]),
        )
        trial.set_user_attr(
            "decision_rows_oof_balanced_accuracy",
            float(result["decision_oof_metrics"]["balanced_accuracy"]),
        )
        trial.set_user_attr(
            "decision_rows_oof_precision",
            float(result["decision_oof_metrics"]["precision"]),
        )
        trial.set_user_attr(
            "decision_rows_oof_recall",
            float(result["decision_oof_metrics"]["recall"]),
        )
        trial.set_user_attr(
            "decision_rows_oof_f1",
            float(result["decision_oof_metrics"]["f1"]),
        )
        trial.set_user_attr(
            "decision_rows_total",
            int(result["decision_fold_metrics"]["decision_rows"].sum()),
        )
        trial.set_user_attr(
            "mean_best_iteration",
            int(result["cv_result"]["mean_best_iteration"]),
        )
        trial.set_user_attr("params_source", str(args.params_source))

        print(
            f"trial={trial.number} "
            f"decision_weight={weight_config['decision_weight']:.6f} "
            f"other_weight={weight_config['other_weight']:.6f} "
            f"decision_bal_acc_mean={result['decision_metric_mean']:.6f} "
            f"decision_bal_acc_weighted_mean={result['decision_metric_weighted_mean']:.6f} "
            f"decision_bal_acc_std={result['decision_metric_std']:.6f} "
            f"objective={objective_value:.6f}"
        )

        del result
        gc.collect()
        return objective_value

    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=args.timeout_seconds,
        n_jobs=OPTUNA_N_JOBS,
        gc_after_trial=True,
        show_progress_bar=True,
        catch=(lgb.basic.LightGBMError, OSError, ValueError),
    )

    best_trial = study.best_trial
    best_weight_config = build_weight_config(best_trial.params["decision_weight"])
    best_result = evaluate_weight_config(
        x=x,
        y=y,
        decision_mask=decision_mask,
        folds=folds,
        fold_weight_by_id=fold_weight_by_id,
        param_overrides=param_overrides,
        weight_config=best_weight_config,
        std_penalty=float(args.objective_std_penalty),
        float_dtype=modeling_float_dtype,
        model_variant="best_trial_recheck",
    )

    payload = build_best_result_payload(
        args=args,
        run_info=run_info,
        data_path=data_path,
        feature_subset=feature_subset,
        excluded_features=excluded_features,
        class_distribution=class_distribution,
        decision_mask=decision_mask,
        best_trial=best_trial,
        best_result=best_result,
    )
    payload["artifacts"] = {
        "best_result_json": path_to_portable_str(best_result_path),
        "trials_csv": path_to_portable_str(trials_csv_path),
        "best_oof_parquet": (
            path_to_portable_str(best_oof_path) if args.save_best_oof else None
        ),
    }

    best_result_path.parent.mkdir(parents=True, exist_ok=True)
    best_result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    trials_df = build_trials_dataframe(study)
    trials_csv_path.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(trials_csv_path, index=False)

    if args.save_best_oof:
        save_best_oof_predictions(
            output_path=best_oof_path,
            df=df,
            decision_mask=decision_mask,
            best_result=best_result,
        )

    print(
        f"best target weights | trial={best_trial.number} "
        f"decision_weight={best_weight_config['decision_weight']:.6f} "
        f"other_weight={best_weight_config['other_weight']:.6f} "
        f"decision_rows_bal_acc={best_result['decision_oof_metrics']['balanced_accuracy']:.6f} "
        f"decision_rows_bal_acc_weighted_mean={best_result['decision_metric_weighted_mean']:.6f} "
        f"objective={best_trial.value:.6f}"
    )
    print(
        f"artifacts | json={path_to_portable_str(best_result_path)} "
        f"csv={path_to_portable_str(trials_csv_path)}"
    )


if __name__ == "__main__":
    main()
