import json
from bisect import bisect_left
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common_config_utils import path_to_portable_str
from features.candle_features import CANDLE_PATTERN_COLS, RAW_OHLCV_COLS
from features.session_open_features import add_session_open_features
from features.volume_profile_fixed_range import (
    is_volume_profile_feature,
    validate_volume_profile_dataset_metadata,
    validate_volume_profile_feature_columns,
)
from modeling_dataset_utils import (
    load_modeling_dataset_artifact_metadata,
    load_excluded_feature_names_from_settings,
    load_feature_subset,
    load_feature_subset_from_settings,
    load_modeling_dataset_settings,
    resolve_modeling_float_dtype,
    resolve_modeling_float_dtype_name,
    split_feature_subset,
    summarize_feature_subset,
)
from optuna_run_utils import make_timestamped_artifact_path, resolve_run_study_name
from target_weights import (
    TARGET_WEIGHT_COL,
    TARGET_WEIGHT_DECISION_VALUE,
    compute_decision_mask_from_opened,
    summarize_target_weights,
)
from train_lgbm import (
    CV_FOLDS as DEFAULT_CV_FOLDS,
    EARLY_STOPPING_EVAL_METRIC,
    EARLY_STOPPING_ROUNDS,
    LGBM_DEFAULT_PARAMS,
    LGBM_OPTUNA_BEST_PARAMS,
    N_ESTIMATORS,
    OOF_EXPORT_BASE_COLS,
    TARGET_COL,
    WF_TEST_TO_TRAIN_RATIO as DEFAULT_WF_TEST_TO_TRAIN_RATIO,
    build_lgbm_model,
    classification_metrics,
    clean_and_impute_fold,
    make_walk_forward_folds,
    resolve_sample_weight_series,
)

DECISION_ROW_OBJECTIVE_METRIC = "balanced_accuracy"
DECISION_ROW_STD_PENALTY = 1.0
DEFAULT_STUDY_NAME_PREFIX = "lgbm_target_weight_search_decision_rows_balanced_accuracy"
DEFAULT_OUTPUT_DIR = Path("data/optuna/target_weights")
BEST_RESULT_STEM = "lgbm_target_weight_search_best_decision_rows_balanced_accuracy"
SEARCH_RESULTS_CSV_STEM = "lgbm_target_weight_search_proxy_candidates"
SEARCH_CONTEXT_RESULTS_CSV_STEM = "lgbm_target_weight_search_proxy_contexts"
SEARCH_FOLD_METRICS_CSV_STEM = "lgbm_target_weight_search_proxy_fold_metrics"
FINAL_RESULTS_CSV_STEM = "lgbm_target_weight_search_final_candidates"
FINAL_CONTEXT_RESULTS_CSV_STEM = "lgbm_target_weight_search_final_contexts"
FINAL_FOLD_METRICS_CSV_STEM = "lgbm_target_weight_search_final_fold_metrics"
BEST_OOF_STEM = "lgbm_target_weight_search_best_oof"

ROW_MODE_ALL_ROWS = "all_rows"
ROW_MODE_DECISION_ONLY = "decision_rows_only"
EVAL_SCOPE_ALL_ROWS = "all_rows"
EVAL_SCOPE_DECISION_ONLY = "decision_rows_only"
PREDICTION_SCOPE_ALL_ROWS = "all_rows"
PREDICTION_SCOPE_DECISION_ONLY = "decision_rows_only"
STRATEGY_ALL_ROWS_WEIGHTED = "all_rows_weighted"
STRATEGY_DECISION_ONLY_BASELINE = "decision_rows_only_baseline"

SEED = 37
DECISION_WEIGHT_LOW = 0.20
DECISION_WEIGHT_HIGH = 0.999
DECISION_WEIGHT_STEP = None

ENABLE_FOLD_RECENCY_WEIGHTING = True
FOLD_RECENCY_WEIGHTING_MODE = "linear"
FOLD_RECENCY_WEIGHT_MIN = 1.0
FOLD_RECENCY_WEIGHT_MAX = 1.4

STUDY_NAME = None
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
PARAMS_SOURCE = "target_weight_robust"
CV_FOLDS = 8
TEST_TO_TRAIN_RATIO = DEFAULT_WF_TEST_TO_TRAIN_RATIO
DEVICE_TYPE = "gpu"
LGBM_N_JOBS = 14
OBJECTIVE_STD_PENALTY = DECISION_ROW_STD_PENALTY
SAVE_BEST_OOF = False

FEATURE_SELECTOR_ARTIFACT_ROOT = Path("data/analysis/feature_selector")
FEATURE_SUBSET_RECENT_LIMIT = 3
MAX_FEATURE_SUBSET_CANDIDATES = 3
FEATURE_SUBSET_CANDIDATE_MODE_ACTIVE_ONLY = "active_only"
FEATURE_SUBSET_CANDIDATE_MODE_ACTIVE_PLUS_RECENT = "active_plus_recent"
FEATURE_VIEW_ALL_FEATURES = "all_features"
FEATURE_VIEW_ACTIVE_SUBSET = "active_subset"
FEATURE_VIEW_RANDOM_SUBSET = "random_subset"
PARAM_PROFILE_OPTUNA = "optuna"
PARAM_PROFILE_DEFAULT = "default"
PARAM_PROFILE_TARGET_WEIGHT_ROBUST = "target_weight_robust"
DEFAULT_TARGET_WEIGHT_PARAM_PROFILES = (
    PARAM_PROFILE_TARGET_WEIGHT_ROBUST,
)
DEFAULT_INCLUDE_ALL_FEATURES_VIEW = False
DEFAULT_INCLUDE_ACTIVE_FEATURE_SUBSET_VIEW = False
DEFAULT_RANDOM_FEATURE_SUBSETS = 10
DEFAULT_RANDOM_FEATURE_SUBSET_SIZE = 256
DEFAULT_TARGET_WEIGHT_LOOKBACK_DAYS = 365
CONTEXT_OBJECTIVE_STD_PENALTY = 1.0

TARGET_WEIGHT_SEARCH_ROBUST_PARAMS = {
    "learning_rate": 0.01,
    "num_leaves": 127,
    "min_data_in_leaf": 128,
    "max_depth": 12,
    "feature_fraction": 0.4,
    "bagging_fraction": 0.8,
    "bagging_freq": 8,
    "lambda_l2": 12.0,
    "lambda_l1": 4.0,
    "min_sum_hessian_in_leaf": 0.1,
    "min_gain_to_split": 0.5,
    "feature_fraction_bynode": 0.7,
    "path_smooth": 20.0,
    "extra_trees": False,
}

SEARCH_CV_FOLDS = 3
SEARCH_N_ESTIMATORS = 400
SEARCH_EARLY_STOPPING_ROUNDS = 30
SEARCH_REFINEMENT_ROUNDS = 2
SEARCH_TOP_PARENT_WEIGHTS = 2
TOP_WEIGHT_CANDIDATES_PER_SUBSET = 2
FINAL_TOP_WEIGHTED_CANDIDATES = 2

INITIAL_WEIGHT_GRID = (
    0.20,
    0.35,
    0.50,
    0.65,
    0.80,
    TARGET_WEIGHT_DECISION_VALUE,
    0.95,
    0.98,
)


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


def _format_candidate_weight(value):
    return f"{float(value):.6f}"


def _dedupe_float_candidates(values):
    out = []
    for value in values:
        candidate = float(value)
        if any(np.isclose(candidate, existing) for existing in out):
            continue
        out.append(candidate)
    return out


def _clip_weight_to_search_space(value):
    clipped = float(min(float(DECISION_WEIGHT_HIGH), max(float(DECISION_WEIGHT_LOW), value)))
    if DECISION_WEIGHT_STEP is not None:
        step = float(DECISION_WEIGHT_STEP)
        low = float(DECISION_WEIGHT_LOW)
        snapped = low + (round((clipped - low) / step) * step)
        clipped = float(min(float(DECISION_WEIGHT_HIGH), max(low, snapped)))
    return clipped


def build_initial_weight_candidates():
    base_values = [
        float(DECISION_WEIGHT_LOW),
        *[float(value) for value in INITIAL_WEIGHT_GRID],
        float(DECISION_WEIGHT_HIGH),
    ]
    clipped = [_clip_weight_to_search_space(value) for value in base_values]
    return tuple(sorted(_dedupe_float_candidates(clipped)))


def build_refined_weight_candidates(evaluated_weights, top_parent_weights):
    evaluated = sorted(_dedupe_float_candidates(evaluated_weights))
    if not evaluated:
        return tuple()

    proposed = []
    bounds = [float(DECISION_WEIGHT_LOW), *evaluated, float(DECISION_WEIGHT_HIGH)]
    for parent in top_parent_weights:
        parent_value = float(parent)
        insert_idx = bisect_left(bounds, parent_value)
        if insert_idx == 0 or insert_idx >= len(bounds):
            continue

        lower = bounds[insert_idx - 1]
        upper = bounds[insert_idx + 1] if insert_idx + 1 < len(bounds) else bounds[-1]
        if not np.isclose(lower, parent_value):
            proposed.append((lower + parent_value) / 2.0)
        if not np.isclose(upper, parent_value):
            proposed.append((parent_value + upper) / 2.0)

    proposed = [_clip_weight_to_search_space(value) for value in proposed]
    return tuple(
        sorted(
            value
            for value in _dedupe_float_candidates(proposed)
            if not any(np.isclose(value, existing) for existing in evaluated)
        )
    )


def validate_config():
    if CV_FOLDS < 2:
        raise ValueError("CV_FOLDS must be >= 2.")
    if SEARCH_CV_FOLDS < 2:
        raise ValueError("SEARCH_CV_FOLDS must be >= 2.")
    if SEARCH_CV_FOLDS > CV_FOLDS:
        raise ValueError("SEARCH_CV_FOLDS must be <= CV_FOLDS.")
    if not (0.0 < float(TEST_TO_TRAIN_RATIO) < 1.0):
        raise ValueError("TEST_TO_TRAIN_RATIO must be in (0, 1).")
    if float(OBJECTIVE_STD_PENALTY) < 0.0:
        raise ValueError("OBJECTIVE_STD_PENALTY must be >= 0.")
    if PARAMS_SOURCE not in {
        PARAM_PROFILE_TARGET_WEIGHT_ROBUST,
        PARAM_PROFILE_OPTUNA,
        PARAM_PROFILE_DEFAULT,
    }:
        raise ValueError(
            "PARAMS_SOURCE must be one of: "
            f"'{PARAM_PROFILE_TARGET_WEIGHT_ROBUST}', "
            f"'{PARAM_PROFILE_OPTUNA}', '{PARAM_PROFILE_DEFAULT}'."
        )
    if DEVICE_TYPE not in {"gpu", "cpu"}:
        raise ValueError("DEVICE_TYPE must be one of: 'gpu', 'cpu'.")
    if SEARCH_N_ESTIMATORS < 1:
        raise ValueError("SEARCH_N_ESTIMATORS must be >= 1.")
    if SEARCH_EARLY_STOPPING_ROUNDS < 0:
        raise ValueError("SEARCH_EARLY_STOPPING_ROUNDS must be >= 0.")
    if TOP_WEIGHT_CANDIDATES_PER_SUBSET < 1:
        raise ValueError("TOP_WEIGHT_CANDIDATES_PER_SUBSET must be >= 1.")
    if FINAL_TOP_WEIGHTED_CANDIDATES < 1:
        raise ValueError("FINAL_TOP_WEIGHTED_CANDIDATES must be >= 1.")
    low = float(DECISION_WEIGHT_LOW)
    high = float(DECISION_WEIGHT_HIGH)
    if not (0.0 < low < 1.0):
        raise ValueError("DECISION_WEIGHT_LOW must be in (0, 1).")
    if not (0.0 < high < 1.0):
        raise ValueError("DECISION_WEIGHT_HIGH must be in (0, 1).")
    if low >= high:
        raise ValueError("DECISION_WEIGHT_LOW must be < DECISION_WEIGHT_HIGH.")
    if DECISION_WEIGHT_STEP is not None and float(DECISION_WEIGHT_STEP) <= 0.0:
        raise ValueError("DECISION_WEIGHT_STEP must be > 0 when provided.")


def build_model_param_overrides(param_source=None):
    resolved_param_source = str(
        PARAMS_SOURCE if param_source is None else param_source
    ).strip().lower()
    if resolved_param_source == PARAM_PROFILE_TARGET_WEIGHT_ROBUST:
        params = dict(TARGET_WEIGHT_SEARCH_ROBUST_PARAMS)
    elif resolved_param_source == PARAM_PROFILE_OPTUNA:
        params = dict(LGBM_OPTUNA_BEST_PARAMS)
    elif resolved_param_source == PARAM_PROFILE_DEFAULT:
        params = dict(LGBM_DEFAULT_PARAMS)
    else:
        raise ValueError(
            f"Unsupported param_source={resolved_param_source!r}. "
            "Expected one of: "
            f"{PARAM_PROFILE_TARGET_WEIGHT_ROBUST}, "
            f"{PARAM_PROFILE_OPTUNA}, {PARAM_PROFILE_DEFAULT}."
        )
    params["device_type"] = str(DEVICE_TYPE)
    if LGBM_N_JOBS is not None:
        params["n_jobs"] = int(LGBM_N_JOBS)
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


def build_unit_weight_series(index, *, float_dtype):
    weight_values = np.ones(len(index), dtype=float_dtype)
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
                **{
                    metric_name: float(metric_value)
                    for metric_name, metric_value in fold_metrics.items()
                },
            }
        )

    return pd.DataFrame(fold_rows)


def summarize_boolean_mask(mask):
    mask_np = np.asarray(mask, dtype=bool)
    total_rows = int(mask_np.size)
    matched_rows = int(mask_np.sum())
    return {
        "rows": matched_rows,
        "total_rows": total_rows,
        "share": float(matched_rows / total_rows) if total_rows > 0 else 0.0,
    }


def evaluate_unweighted_prediction_scopes(y_true, y_pred_proba, decision_mask):
    y_true_np = np.asarray(y_true, dtype=np.int8)
    y_pred_np = np.asarray(y_pred_proba, dtype=np.float64)
    decision_mask_np = np.asarray(decision_mask, dtype=bool)

    if y_true_np.shape[0] != y_pred_np.shape[0]:
        raise ValueError("Prediction scope evaluation received mismatched y_true/y_pred.")
    if y_true_np.shape[0] != decision_mask_np.shape[0]:
        raise ValueError(
            "Prediction scope evaluation received mismatched y_true/decision_mask."
        )

    scopes = {}
    scope_masks = {
        "all_rows": np.ones(y_true_np.shape[0], dtype=bool),
        "decision_rows": decision_mask_np,
    }
    for scope_name, scope_mask in scope_masks.items():
        rows = int(scope_mask.sum())
        if rows == 0:
            scopes[scope_name] = {
                "rows": 0,
                "positive_rate": None,
                "metrics": None,
            }
            continue

        y_scope = y_true_np[scope_mask]
        pred_scope = y_pred_np[scope_mask]
        scopes[scope_name] = {
            "rows": rows,
            "positive_rate": float(np.mean(y_scope)),
            "metrics": classification_metrics(
                y_scope,
                pred_scope,
                sample_weight=None,
            ),
        }
    return scopes


def select_strategy_fold_indices(
    train_indices,
    test_indices,
    decision_mask,
    *,
    row_mode,
    eval_scope,
    prediction_scope,
):
    train_indices = np.asarray(train_indices, dtype=np.int32)
    test_indices = np.asarray(test_indices, dtype=np.int32)
    decision_mask_np = np.asarray(decision_mask, dtype=bool)

    if row_mode == ROW_MODE_ALL_ROWS:
        train_used = train_indices
    elif row_mode == ROW_MODE_DECISION_ONLY:
        train_used = train_indices[decision_mask_np[train_indices]]
    else:
        raise ValueError(f"Unsupported row_mode: {row_mode}")

    if eval_scope == EVAL_SCOPE_ALL_ROWS:
        eval_used = test_indices
    elif eval_scope == EVAL_SCOPE_DECISION_ONLY:
        eval_used = test_indices[decision_mask_np[test_indices]]
    else:
        raise ValueError(f"Unsupported eval_scope: {eval_scope}")

    if prediction_scope == PREDICTION_SCOPE_ALL_ROWS:
        predict_used = test_indices
    elif prediction_scope == PREDICTION_SCOPE_DECISION_ONLY:
        predict_used = test_indices[decision_mask_np[test_indices]]
    else:
        raise ValueError(f"Unsupported prediction_scope: {prediction_scope}")

    return {
        "train_indices": train_used,
        "eval_indices": eval_used,
        "predict_indices": predict_used,
    }


def _strategy_sample_weight_series(
    *,
    row_mode,
    decision_mask,
    index,
    float_dtype,
    weight_config,
):
    if row_mode == ROW_MODE_ALL_ROWS:
        if weight_config is None:
            raise ValueError("weight_config is required for all-rows weighted strategy.")
        return build_sample_weight_series(
            decision_mask,
            weight_config,
            index=index,
            float_dtype=float_dtype,
        )
    if row_mode == ROW_MODE_DECISION_ONLY:
        return build_unit_weight_series(index, float_dtype=float_dtype)
    raise ValueError(f"Unsupported row_mode: {row_mode}")


def evaluate_strategy(
    *,
    x,
    y,
    decision_mask,
    folds,
    fold_weight_by_id,
    param_overrides,
    row_mode,
    eval_scope,
    prediction_scope,
    weight_config,
    std_penalty,
    float_dtype,
    model_variant,
    n_estimators,
    early_stopping_rounds,
):
    decision_mask_np = np.asarray(decision_mask, dtype=bool)
    sample_weight = _strategy_sample_weight_series(
        row_mode=row_mode,
        decision_mask=decision_mask_np,
        index=y.index,
        float_dtype=float_dtype,
        weight_config=weight_config,
    )
    sample_weight_np = sample_weight.to_numpy(dtype=float_dtype, copy=False)
    y_np = y.to_numpy(dtype=np.int8, copy=False)
    oof_pred_proba = np.full(shape=len(x), fill_value=np.nan, dtype=np.float64)
    fold_rows = []
    best_iterations = []

    for fold in folds:
        train_indices = np.arange(
            int(fold["train_start"]),
            int(fold["train_end"]),
            dtype=np.int32,
        )
        test_indices = np.arange(
            int(fold["test_start"]),
            int(fold["test_end"]),
            dtype=np.int32,
        )
        selected = select_strategy_fold_indices(
            train_indices=train_indices,
            test_indices=test_indices,
            decision_mask=decision_mask_np,
            row_mode=row_mode,
            eval_scope=eval_scope,
            prediction_scope=prediction_scope,
        )
        train_used = selected["train_indices"]
        eval_used = selected["eval_indices"]
        predict_used = selected["predict_indices"]

        if train_used.size == 0:
            raise ValueError(
                f"Fold {fold['fold_id']} produced 0 training rows for row_mode={row_mode}."
            )
        if eval_used.size == 0:
            raise ValueError(
                f"Fold {fold['fold_id']} produced 0 validation rows for eval_scope={eval_scope}."
            )
        if predict_used.size == 0:
            raise ValueError(
                f"Fold {fold['fold_id']} produced 0 prediction rows for prediction_scope={prediction_scope}."
            )

        y_train = y_np[train_used]
        if np.unique(y_train).size < 2:
            raise ValueError(
                "Target has only one class inside a fold after row filtering. "
                f"fold_id={fold['fold_id']} row_mode={row_mode}"
            )

        x_train_raw = x.iloc[train_used]
        x_eval_raw = x.iloc[eval_used]
        x_predict_raw = x.iloc[predict_used]
        x_train, x_eval, _, dropped_nan_features, _ = clean_and_impute_fold(
            x_train_raw,
            x_eval_raw,
            float_dtype=float_dtype,
        )
        x_predict = x_predict_raw.drop(
            columns=dropped_nan_features,
            errors="ignore",
        ).astype(float_dtype, copy=False)

        model = build_lgbm_model(
            n_estimators=int(n_estimators),
            param_overrides=param_overrides,
        )
        fit_kwargs = {
            "X": x_train,
            "y": y_train,
            "sample_weight": sample_weight_np[train_used],
            "eval_set": [(x_eval, y_np[eval_used])],
            "eval_sample_weight": [sample_weight_np[eval_used]],
            "eval_metric": EARLY_STOPPING_EVAL_METRIC,
        }
        if int(early_stopping_rounds) > 0:
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(
                    stopping_rounds=int(early_stopping_rounds),
                    verbose=False,
                )
            ]
        model.fit(**fit_kwargs)

        best_iteration = int(model.best_iteration_ or int(n_estimators))
        best_iterations.append(best_iteration)
        fold_pred = model.predict_proba(
            x_predict,
            num_iteration=best_iteration,
        )[:, 1]
        oof_pred_proba[predict_used] = fold_pred.astype(np.float64, copy=False)
        fold_rows.append(
            {
                "fold_id": int(fold["fold_id"]),
                "row_mode": row_mode,
                "eval_scope": eval_scope,
                "prediction_scope": prediction_scope,
                "train_rows_used": int(train_used.size),
                "eval_rows_used": int(eval_used.size),
                "predict_rows_used": int(predict_used.size),
                "predict_decision_rows": int(decision_mask_np[predict_used].sum()),
                "best_iteration": best_iteration,
                "train_opened_min_index": int(train_used[0]),
                "train_opened_max_index": int(train_used[-1]),
                "test_opened_min_index": int(test_indices[0]),
                "test_opened_max_index": int(test_indices[-1]),
            }
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

    oof_mask = np.isfinite(oof_pred_proba)
    decision_oof_mask = oof_mask & decision_mask_np
    if not np.any(decision_oof_mask):
        raise RuntimeError("No OOF decision-row predictions were produced.")

    prediction_scope_metrics = classification_metrics(
        y_np[oof_mask],
        oof_pred_proba[oof_mask],
        sample_weight=None,
    )
    decision_oof_metrics = classification_metrics(
        y_np[decision_oof_mask],
        oof_pred_proba[decision_oof_mask],
        sample_weight=None,
    )
    oof_scope_metrics = None
    if prediction_scope == PREDICTION_SCOPE_ALL_ROWS:
        oof_scope_metrics = evaluate_unweighted_prediction_scopes(
            y_true=y_np[oof_mask],
            y_pred_proba=oof_pred_proba[oof_mask],
            decision_mask=decision_mask_np[oof_mask],
        )

    fold_df = pd.DataFrame(fold_rows).merge(
        decision_fold_metrics,
        on="fold_id",
        how="left",
    )
    return {
        "row_mode": row_mode,
        "eval_scope": eval_scope,
        "prediction_scope": prediction_scope,
        "sample_weight": sample_weight,
        "sample_weight_summary": summarize_target_weights(
            sample_weight.to_numpy(dtype=np.float64, copy=False)
        ),
        "oof_pred_proba": oof_pred_proba,
        "oof_coverage_ratio": float(oof_mask.mean()),
        "decision_oof_metrics": decision_oof_metrics,
        "prediction_scope_metrics": prediction_scope_metrics,
        "oof_scope_metrics": oof_scope_metrics,
        "decision_fold_metrics": decision_fold_metrics,
        "fold_metrics": fold_df,
        "folds": list(folds),
        "fold_weight_by_id": fold_weight_by_id.copy(),
        "mean_best_iteration": int(np.round(np.mean(best_iterations))),
        **decision_metric_summary,
    }


def build_result_row(
    *,
    stage,
    feature_subset_candidate,
    strategy_name,
    result,
    weight_config,
    n_estimators,
    cv_folds,
    search_round=None,
):
    row = {
        "stage": stage,
        "feature_subset_id": feature_subset_candidate["id"],
        "feature_subset_label": feature_subset_candidate["label"],
        "feature_subset_path": feature_subset_candidate["path"],
        "feature_count": int(feature_subset_candidate["feature_count"]),
        "is_active_feature_subset": bool(feature_subset_candidate["is_active"]),
        "strategy_name": strategy_name,
        "row_mode": result["row_mode"],
        "eval_scope": result["eval_scope"],
        "prediction_scope": result["prediction_scope"],
        "cv_folds": int(cv_folds),
        "n_estimators": int(n_estimators),
        "search_round": search_round,
        "objective_value": float(result["decision_metric_objective"]),
        "decision_metric_mean": float(result["decision_metric_mean"]),
        "decision_metric_weighted_mean": float(result["decision_metric_weighted_mean"]),
        "decision_metric_std": float(result["decision_metric_std"]),
        "decision_metric_base_value": float(result["decision_metric_base_value"]),
        "decision_rows_oof_accuracy": float(result["decision_oof_metrics"]["accuracy"]),
        "decision_rows_oof_balanced_accuracy": float(
            result["decision_oof_metrics"]["balanced_accuracy"]
        ),
        "decision_rows_oof_precision": float(
            result["decision_oof_metrics"]["precision"]
        ),
        "decision_rows_oof_recall": float(result["decision_oof_metrics"]["recall"]),
        "decision_rows_oof_f1": float(result["decision_oof_metrics"]["f1"]),
        "decision_rows_oof_brier": float(
            result["decision_oof_metrics"]["brier_score"]
        ),
        "decision_rows_oof_logloss": float(
            result["decision_oof_metrics"]["binary_logloss"]
        ),
        "prediction_scope_accuracy": float(result["prediction_scope_metrics"]["accuracy"]),
        "prediction_scope_balanced_accuracy": float(
            result["prediction_scope_metrics"]["balanced_accuracy"]
        ),
        "prediction_scope_brier": float(
            result["prediction_scope_metrics"]["brier_score"]
        ),
        "prediction_scope_logloss": float(
            result["prediction_scope_metrics"]["binary_logloss"]
        ),
        "oof_coverage_ratio": float(result["oof_coverage_ratio"]),
        "mean_best_iteration": int(result["mean_best_iteration"]),
    }
    if result["oof_scope_metrics"] is not None:
        all_rows_metrics = result["oof_scope_metrics"]["all_rows"]["metrics"]
        if all_rows_metrics is not None:
            row["all_rows_oof_accuracy"] = float(all_rows_metrics["accuracy"])
            row["all_rows_oof_balanced_accuracy"] = float(
                all_rows_metrics["balanced_accuracy"]
            )
            row["all_rows_oof_brier"] = float(all_rows_metrics["brier_score"])
            row["all_rows_oof_logloss"] = float(all_rows_metrics["binary_logloss"])
    if weight_config is None:
        row["decision_weight"] = None
        row["other_weight"] = None
        row["non_decision_total_weight"] = None
        row["total_block_weight"] = None
    else:
        row["decision_weight"] = float(weight_config["decision_weight"])
        row["other_weight"] = float(weight_config["other_weight"])
        row["non_decision_total_weight"] = float(
            weight_config["non_decision_total_weight"]
        )
        row["total_block_weight"] = float(weight_config["total_block_weight"])
    return row


def attach_metadata_to_fold_metrics(
    fold_df,
    *,
    stage,
    feature_subset_candidate,
    strategy_name,
    weight_config,
):
    out = fold_df.copy()
    out.insert(0, "stage", stage)
    out.insert(1, "feature_subset_id", feature_subset_candidate["id"])
    out.insert(2, "feature_subset_label", feature_subset_candidate["label"])
    out.insert(3, "strategy_name", strategy_name)
    out.insert(4, "feature_count", int(feature_subset_candidate["feature_count"]))
    out["decision_weight"] = (
        None if weight_config is None else float(weight_config["decision_weight"])
    )
    return out


def discover_recent_feature_subset_paths(*, root_dir, limit):
    root_path = Path(root_dir)
    if not root_path.exists():
        return []
    paths = sorted(
        root_path.glob("*/recommended_features.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return paths[: int(limit)]


def load_target_weight_search_settings(dataset_settings):
    output_dir = Path(dataset_settings["modeling_output_dir"])
    output_suffix = str(dataset_settings["output_suffix"]).strip()
    if not output_suffix:
        raise ValueError("dataset_settings.output_suffix cannot be empty.")

    param_profiles = tuple(DEFAULT_TARGET_WEIGHT_PARAM_PROFILES)
    include_all_features_view = bool(DEFAULT_INCLUDE_ALL_FEATURES_VIEW)
    include_active_feature_subset_view = bool(
        DEFAULT_INCLUDE_ACTIVE_FEATURE_SUBSET_VIEW
    )
    random_feature_subsets = int(DEFAULT_RANDOM_FEATURE_SUBSETS)
    random_feature_subset_size = int(DEFAULT_RANDOM_FEATURE_SUBSET_SIZE)
    random_feature_subset_fraction = None
    random_feature_subset_min_features = None
    context_std_penalty = float(CONTEXT_OBJECTIVE_STD_PENALTY)
    lookback_days = int(DEFAULT_TARGET_WEIGHT_LOOKBACK_DAYS)

    if random_feature_subsets < 0:
        raise ValueError("DEFAULT_RANDOM_FEATURE_SUBSETS must be >= 0.")
    if random_feature_subset_size < 1:
        raise ValueError("DEFAULT_RANDOM_FEATURE_SUBSET_SIZE must be >= 1.")
    if context_std_penalty < 0.0:
        raise ValueError("CONTEXT_OBJECTIVE_STD_PENALTY must be >= 0.")
    if lookback_days < 1:
        raise ValueError("DEFAULT_TARGET_WEIGHT_LOOKBACK_DAYS must be >= 1.")
    if (
        not include_all_features_view
        and not include_active_feature_subset_view
        and random_feature_subsets < 1
    ):
        raise ValueError("Target-weight search must enable at least one feature view.")

    return {
        "output_dir": output_dir,
        "output_suffix": output_suffix,
        "param_profiles": param_profiles,
        "include_all_features_view": include_all_features_view,
        "include_active_feature_subset_view": include_active_feature_subset_view,
        "random_feature_subsets": random_feature_subsets,
        "random_feature_subset_size": random_feature_subset_size,
        "random_feature_subset_fraction": random_feature_subset_fraction,
        "random_feature_subset_min_features": random_feature_subset_min_features,
        "context_std_penalty": context_std_penalty,
        "lookback_days": lookback_days,
    }


def resolve_target_weight_search_dataset_path(dataset_settings, search_settings):
    output_dir = Path(search_settings["output_dir"])
    output_stem = (
        f"{Path(dataset_settings['base_data_file']).stem}"
        f"{search_settings['output_suffix']}"
    )
    return output_dir / f"{output_stem}.parquet"


def resolve_target_weight_search_time_window(data_path, *, lookback_days):
    if lookback_days is None:
        return None

    opened_df = pd.read_parquet(data_path, columns=["Opened"])
    if "Opened" not in opened_df.columns or opened_df.empty:
        raise ValueError(
            "Cannot resolve target-weight search time window because 'Opened' is missing."
        )
    opened_series = pd.to_datetime(opened_df["Opened"], utc=True, errors="coerce")
    latest_opened = opened_series.max()
    if pd.isna(latest_opened):
        raise ValueError("Opened column contains no valid timestamps.")
    opened_start_utc = latest_opened - pd.Timedelta(days=int(lookback_days))
    return {
        "lookback_days": int(lookback_days),
        "opened_start_utc": opened_start_utc,
        "opened_end_utc": latest_opened,
        "rows_total": int(len(opened_df)),
        "rows_in_window": int((opened_series >= opened_start_utc).sum()),
    }


def _normalize_parquet_filter_timestamp(value):
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


def load_target_weight_training_frame(
    *,
    data_path,
    excluded_features=None,
    float_dtype=np.float64,
    opened_start_utc=None,
    opened_end_utc=None,
):
    excluded_feature_names = (
        tuple(excluded_features["features"]) if excluded_features else tuple()
    )
    excluded_feature_set = set(excluded_feature_names)

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"Loading dataset: {data_path}")
    if excluded_features:
        preview = ", ".join(excluded_feature_names[:5])
        print(
            "Feature exclusions active: "
            f"count={excluded_features['count']} preview=[{preview}]"
        )
    if opened_start_utc is not None or opened_end_utc is not None:
        print(
            "Time filter active: "
            f"opened_start_utc={opened_start_utc} opened_end_utc={opened_end_utc}"
        )

    read_kwargs = {}
    parquet_filters = []
    if opened_start_utc is not None:
        parquet_filters.append(
            ("Opened", ">=", _normalize_parquet_filter_timestamp(opened_start_utc))
        )
    if opened_end_utc is not None:
        parquet_filters.append(
            ("Opened", "<=", _normalize_parquet_filter_timestamp(opened_end_utc))
        )
    if parquet_filters:
        read_kwargs["filters"] = parquet_filters

    df = pd.read_parquet(data_path, **read_kwargs)
    print(f"Loaded dataset: rows={len(df)} cols={len(df.columns)}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")

    df = df[df[TARGET_COL].notna()].copy()
    if len(df) == 0:
        raise ValueError("No rows left after TARGET_COL non-null filtering.")

    dropped_legacy_cdl = [col for col in CANDLE_PATTERN_COLS if col in df.columns]
    if dropped_legacy_cdl:
        df = df.drop(columns=dropped_legacy_cdl)
        print(f"Dropped legacy 1m CDL columns: {len(dropped_legacy_cdl)}")

    df = add_session_open_features(df)
    sample_weight, sample_weight_source, sample_weight_summary = (
        resolve_sample_weight_series(df, float_dtype=float_dtype)
    )
    df[TARGET_WEIGHT_COL] = sample_weight

    y = df[TARGET_COL].astype(np.int8, copy=False)
    class_distribution = y.value_counts().sort_index().to_dict()
    weighted_class_distribution = {
        str(int(class_id)): float(sample_weight.loc[y.index[y == class_id]].sum())
        for class_id in sorted(class_distribution.keys())
    }

    dropped_raw_ohlcv_features = [col for col in RAW_OHLCV_COLS if col in df.columns]
    x_drop_cols = [TARGET_COL, TARGET_WEIGHT_COL, *dropped_raw_ohlcv_features]
    x = df.drop(columns=x_drop_cols, errors="ignore")
    x = x.replace([np.inf, -np.inf], np.nan)

    non_numeric_features = [
        col for col in x.columns if not pd.api.types.is_numeric_dtype(x[col])
    ]
    if non_numeric_features:
        x = x.drop(columns=non_numeric_features)

    if excluded_feature_set:
        excluded_present_features = [
            col for col in x.columns if col in excluded_feature_set
        ]
        excluded_missing_features = [
            col for col in excluded_feature_names if col not in x.columns
        ]
        if excluded_present_features:
            x = x.drop(columns=excluded_present_features)
        print(
            "Feature exclusions applied: "
            f"dropped_feature_cols={len(excluded_present_features)} "
            f"missing_requested={len(excluded_missing_features)}"
        )

    x = x.astype(float_dtype, copy=False)
    validate_volume_profile_feature_columns(
        x.columns,
        source_label=f"modeling dataset features at {data_path}",
    )
    volume_profile_feature_columns = tuple(
        col for col in x.columns if is_volume_profile_feature(col)
    )
    if volume_profile_feature_columns:
        dataset_metadata, metadata_path = load_modeling_dataset_artifact_metadata(
            data_path
        )
        validate_volume_profile_dataset_metadata(
            dataset_metadata,
            feature_columns=volume_profile_feature_columns,
            cfg=load_modeling_dataset_settings().get("volume_profile_fixed_range"),
            source_label=f"modeling dataset metadata {metadata_path}",
        )

    return {
        "df": df,
        "x": x,
        "y": y,
        "sample_weight": sample_weight,
        "sample_weight_source": sample_weight_source,
        "sample_weight_summary": sample_weight_summary,
        "class_distribution": class_distribution,
        "weighted_class_distribution": weighted_class_distribution,
        "dropped_raw_ohlcv_features": dropped_raw_ohlcv_features,
        "dropped_non_numeric_features": non_numeric_features,
    }


def load_parquet_column_names(data_path):
    dataset_path = Path(data_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    import pyarrow.parquet as pq

    return tuple(str(name) for name in pq.ParquetFile(dataset_path).schema_arrow.names)


def resolve_feature_subset_required_parquet_columns(subset_info):
    if subset_info is None:
        return tuple(OOF_EXPORT_BASE_COLS)

    selected_feature_columns = list(subset_info["features"])
    subset_parts = split_feature_subset(
        selected_feature_columns,
        source_label=f"feature subset {subset_info['path']}",
    )
    session_feature_set = set(subset_parts["session_feature_cols"])
    return tuple(
        dict.fromkeys(
            [
                *OOF_EXPORT_BASE_COLS,
                *(
                    feature_name
                    for feature_name in selected_feature_columns
                    if feature_name not in session_feature_set
                ),
            ]
        )
    )


def resolve_missing_parquet_columns_for_subset(
    subset_info,
    available_parquet_columns,
):
    available_column_set = set(available_parquet_columns)
    required_columns = resolve_feature_subset_required_parquet_columns(subset_info)
    return tuple(
        column_name
        for column_name in required_columns
        if column_name not in available_column_set
    )


def _feature_subset_signature(subset_info):
    if subset_info is None:
        return ("__all_numeric_features__",)
    return tuple(str(feature) for feature in subset_info["features"])


def _feature_subset_candidate_label(subset_info, *, is_active):
    if subset_info is None:
        return "all_numeric_features"
    parent_name = Path(subset_info["path"]).parent.name
    if is_active:
        return f"active:{parent_name}"
    return f"recent:{parent_name}"


def resolve_feature_subset_candidates(
    dataset_settings,
    excluded_features,
    *,
    available_parquet_columns=None,
    candidate_mode=FEATURE_SUBSET_CANDIDATE_MODE_ACTIVE_ONLY,
    recent_limit=FEATURE_SUBSET_RECENT_LIMIT,
    max_candidates=MAX_FEATURE_SUBSET_CANDIDATES,
):
    candidate_mode = str(candidate_mode).strip().lower()
    if candidate_mode not in {
        FEATURE_SUBSET_CANDIDATE_MODE_ACTIVE_ONLY,
        FEATURE_SUBSET_CANDIDATE_MODE_ACTIVE_PLUS_RECENT,
    }:
        raise ValueError(
            f"Unsupported candidate_mode={candidate_mode!r} for feature subset selection."
        )

    active_subset = load_feature_subset_from_settings(dataset_settings)
    active_subset_path = None if active_subset is None else Path(active_subset["path"])
    candidates = []
    seen_signatures = set()

    def register_candidate(subset_info, *, is_active):
        if (
            subset_info is not None
            and available_parquet_columns is not None
        ):
            missing_parquet_columns = resolve_missing_parquet_columns_for_subset(
                subset_info,
                available_parquet_columns,
            )
            if missing_parquet_columns:
                preview = ", ".join(missing_parquet_columns[:10])
                subset_path = path_to_portable_str(subset_info["path"])
                if is_active:
                    raise ValueError(
                        "Active feature subset is incompatible with the current modeling dataset. "
                        f"path={subset_path} missing_count={len(missing_parquet_columns)} "
                        f"preview=[{preview}]"
                    )
                print(
                    "Skipping incompatible recent feature subset: "
                    f"path={subset_path} missing_count={len(missing_parquet_columns)} "
                    f"preview=[{preview}]"
                )
                return

        signature = _feature_subset_signature(subset_info)
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        label = _feature_subset_candidate_label(subset_info, is_active=is_active)
        candidates.append(
            {
                "id": f"subset_{len(candidates):02d}",
                "label": label,
                "path": (
                    None
                    if subset_info is None
                    else path_to_portable_str(subset_info["path"])
                ),
                "subset_info": subset_info,
                "feature_count": (
                    0 if subset_info is None else int(subset_info["count"])
                ),
                "is_active": bool(is_active),
                "summary": summarize_feature_subset(
                    subset_info,
                    excluded_features=excluded_features,
                ),
            }
        )

    if active_subset is not None:
        register_candidate(active_subset, is_active=True)
    if candidate_mode == FEATURE_SUBSET_CANDIDATE_MODE_ACTIVE_PLUS_RECENT:
        for subset_path in discover_recent_feature_subset_paths(
            root_dir=FEATURE_SELECTOR_ARTIFACT_ROOT,
            limit=int(recent_limit),
        ):
            if active_subset_path is not None and Path(subset_path) == active_subset_path:
                continue
            subset_info = load_feature_subset(
                subset_path,
                list_key=dataset_settings.get("feature_subset_list_key"),
            )
            excluded_feature_names = tuple(
                (excluded_features or {}).get("features") or tuple()
            )
            if excluded_feature_names:
                excluded_feature_set = set(excluded_feature_names)
                filtered = tuple(
                    feature
                    for feature in subset_info["features"]
                    if feature not in excluded_feature_set
                )
                subset_info = {
                    **subset_info,
                    "features": filtered,
                    "count": len(filtered),
                    "source_count": int(subset_info["count"]),
                    "excluded_feature_names": excluded_feature_names,
                    "excluded_count": len(excluded_feature_names),
                    "excluded_from_subset_count": int(
                        subset_info["count"] - len(filtered)
                    ),
                }
                if not subset_info["features"]:
                    continue
            register_candidate(subset_info, is_active=False)
            if len(candidates) >= int(max_candidates):
                break

    if not candidates:
        register_candidate(None, is_active=False)

    return candidates[: int(max_candidates)]


def build_union_feature_subset(feature_subset_candidates):
    subset_infos = [
        candidate["subset_info"]
        for candidate in feature_subset_candidates
        if candidate["subset_info"] is not None
    ]
    if not subset_infos:
        return None
    if len(subset_infos) == 1:
        return dict(subset_infos[0])

    union_features = []
    seen_features = set()
    for subset_info in subset_infos:
        for feature_name in subset_info["features"]:
            if feature_name in seen_features:
                continue
            union_features.append(feature_name)
            seen_features.add(feature_name)

    return {
        "path": Path("<virtual_union_feature_subset>"),
        "features": tuple(union_features),
        "count": len(union_features),
        "format": "virtual",
        "list_key": None,
        "created_utc": None,
        "source_data_path": None,
        "metadata": {
            "virtual_union": True,
            "source_paths": [
                path_to_portable_str(candidate["subset_info"]["path"])
                for candidate in feature_subset_candidates
                if candidate["subset_info"] is not None
            ],
        },
        "source_count": len(union_features),
        "excluded_feature_names": tuple(),
        "excluded_count": 0,
        "excluded_from_subset_count": 0,
    }


def build_feature_matrix_by_subset(x_union, feature_subset_candidates):
    matrices = {}
    for candidate in feature_subset_candidates:
        subset_info = candidate["subset_info"]
        if subset_info is None:
            matrices[candidate["id"]] = x_union
            continue

        missing_features = [
            feature_name
            for feature_name in subset_info["features"]
            if feature_name not in x_union.columns
        ]
        if missing_features:
            preview = ", ".join(missing_features[:10])
            raise ValueError(
                "Loaded union feature matrix is missing required subset features. "
                f"subset_id={candidate['id']} missing_count={len(missing_features)} "
                f"preview=[{preview}]"
            )
        matrices[candidate["id"]] = x_union.loc[:, list(subset_info["features"])].copy()
    return matrices


def build_feature_view_candidates(
    *,
    x_all,
    active_subset,
    include_all_features_view,
    include_active_feature_subset_view,
    random_feature_subsets,
    random_feature_subset_size,
    random_feature_subset_fraction,
    random_feature_subset_min_features,
):
    candidates = []
    seen_signatures = set()
    all_columns = tuple(str(column_name) for column_name in x_all.columns)
    all_column_set = set(all_columns)

    def register_candidate(
        feature_names,
        *,
        label,
        source_kind,
        path=None,
        is_active=False,
        summary=None,
        metadata=None,
    ):
        feature_names = all_columns if feature_names is None else tuple(feature_names)
        missing_features = [
            feature_name
            for feature_name in feature_names
            if feature_name not in all_column_set
        ]
        if missing_features:
            preview = ", ".join(missing_features[:10])
            raise ValueError(
                "Feature view references columns missing from the loaded dataset. "
                f"label={label} missing_count={len(missing_features)} preview=[{preview}]"
            )
        if not feature_names:
            return
        signature = tuple(feature_names)
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        candidates.append(
            {
                "id": f"view_{len(candidates):02d}",
                "label": str(label),
                "path": None if path is None else path_to_portable_str(path),
                "source_kind": str(source_kind),
                "is_active": bool(is_active),
                "feature_count": int(len(feature_names)),
                "features": signature,
                "summary": summary,
                "metadata": metadata or {},
            }
        )

    if include_all_features_view:
        register_candidate(
            None,
            label=FEATURE_VIEW_ALL_FEATURES,
            source_kind=FEATURE_VIEW_ALL_FEATURES,
            summary={
                "path": None,
                "count": int(len(all_columns)),
                "source_count": int(len(all_columns)),
                "format": "generated",
                "list_key": None,
                "created_utc": None,
                "source_data_path": None,
                "excluded_from_subset_count": 0,
            },
            metadata={"generated_from": "loaded_modeling_dataset"},
        )

    if include_active_feature_subset_view and active_subset is not None:
        register_candidate(
            active_subset["features"],
            label=f"active:{Path(active_subset['path']).parent.name}",
            source_kind=FEATURE_VIEW_ACTIVE_SUBSET,
            path=active_subset["path"],
            is_active=True,
            summary=summarize_feature_subset(active_subset),
            metadata={"generated_from": "active_feature_subset"},
        )

    total_feature_count = len(all_columns)
    if random_feature_subset_size is not None:
        subset_size = int(random_feature_subset_size)
    else:
        if random_feature_subset_fraction is None:
            raise ValueError(
                "random_feature_subset_fraction is required when random_feature_subset_size is not set."
            )
        subset_size = int(
            round(total_feature_count * float(random_feature_subset_fraction))
        )
        if random_feature_subset_min_features is not None:
            subset_size = max(int(random_feature_subset_min_features), subset_size)
    subset_size = min(total_feature_count, subset_size)
    if total_feature_count > 0 and subset_size > 0:
        rng = np.random.default_rng(SEED)
        for random_idx in range(int(random_feature_subsets)):
            selected_positions = np.sort(
                rng.choice(total_feature_count, size=subset_size, replace=False)
            )
            selected_features = tuple(
                all_columns[int(position)] for position in selected_positions
            )
            register_candidate(
                selected_features,
                label=f"random:{random_idx:02d}",
                source_kind=FEATURE_VIEW_RANDOM_SUBSET,
                summary={
                    "path": None,
                    "count": int(len(selected_features)),
                    "source_count": int(total_feature_count),
                    "format": "generated",
                    "list_key": None,
                    "created_utc": None,
                    "source_data_path": None,
                    "excluded_from_subset_count": 0,
                },
                metadata={
                    "generated_from": "all_features",
                    "random_seed": int(SEED),
                    "random_subset_index": int(random_idx),
                    "subset_fraction": (
                        None
                        if random_feature_subset_fraction is None
                        else float(random_feature_subset_fraction)
                    ),
                    "subset_size": int(len(selected_features)),
                },
            )

    if not candidates:
        raise ValueError("No feature views were generated for target-weight search.")
    return candidates


def build_feature_matrix_by_view(x_all, feature_views):
    matrices = {}
    all_columns = tuple(str(column_name) for column_name in x_all.columns)
    for feature_view in feature_views:
        feature_names = tuple(feature_view["features"])
        if feature_names == all_columns:
            matrices[feature_view["id"]] = x_all
            continue
        missing_features = [
            feature_name
            for feature_name in feature_names
            if feature_name not in x_all.columns
        ]
        if missing_features:
            preview = ", ".join(missing_features[:10])
            raise ValueError(
                "Loaded feature matrix is missing feature-view columns. "
                f"feature_view_id={feature_view['id']} missing_count={len(missing_features)} "
                f"preview=[{preview}]"
            )
        matrices[feature_view["id"]] = x_all.loc[:, list(feature_names)].copy()
    return matrices


def build_param_profile_candidates(search_settings):
    candidates = []
    for profile_name in search_settings["param_profiles"]:
        params = build_model_param_overrides(param_source=profile_name)
        candidates.append(
            {
                "id": f"params_{len(candidates):02d}",
                "name": str(profile_name),
                "label": str(profile_name),
                "params": params,
            }
        )
    return candidates


def build_evaluation_contexts(feature_views, x_by_view, param_profiles):
    contexts = []
    for feature_view in feature_views:
        for param_profile in param_profiles:
            contexts.append(
                {
                    "id": f"context_{len(contexts):02d}",
                    "label": f"{feature_view['label']} | {param_profile['label']}",
                    "feature_view": feature_view,
                    "feature_view_id": feature_view["id"],
                    "feature_view_label": feature_view["label"],
                    "param_profile": param_profile,
                    "param_profile_name": param_profile["name"],
                    "param_profile_id": param_profile["id"],
                    "x": x_by_view[feature_view["id"]],
                    "param_overrides": param_profile["params"],
                    "is_primary": len(contexts) == 0,
                }
            )
    if not contexts:
        raise ValueError("No evaluation contexts were built for target-weight search.")
    return contexts


def build_context_result_row(
    *,
    stage,
    evaluation_context,
    strategy_name,
    result,
    weight_config,
    n_estimators,
    cv_folds,
    search_round=None,
):
    row = {
        "stage": stage,
        "context_id": evaluation_context["id"],
        "context_label": evaluation_context["label"],
        "feature_view_id": evaluation_context["feature_view"]["id"],
        "feature_view_label": evaluation_context["feature_view"]["label"],
        "feature_view_path": evaluation_context["feature_view"]["path"],
        "feature_view_source_kind": evaluation_context["feature_view"]["source_kind"],
        "feature_count": int(evaluation_context["feature_view"]["feature_count"]),
        "is_active_feature_view": bool(evaluation_context["feature_view"]["is_active"]),
        "param_profile_id": evaluation_context["param_profile"]["id"],
        "param_profile_name": evaluation_context["param_profile"]["name"],
        "is_primary_context": bool(evaluation_context["is_primary"]),
        "strategy_name": str(strategy_name),
        "row_mode": result["row_mode"],
        "eval_scope": result["eval_scope"],
        "prediction_scope": result["prediction_scope"],
        "cv_folds": int(cv_folds),
        "n_estimators": int(n_estimators),
        "search_round": search_round,
        "objective_value": float(result["decision_metric_objective"]),
        "decision_metric_mean": float(result["decision_metric_mean"]),
        "decision_metric_weighted_mean": float(result["decision_metric_weighted_mean"]),
        "decision_metric_std": float(result["decision_metric_std"]),
        "decision_metric_base_value": float(result["decision_metric_base_value"]),
        "decision_rows_oof_accuracy": float(result["decision_oof_metrics"]["accuracy"]),
        "decision_rows_oof_balanced_accuracy": float(
            result["decision_oof_metrics"]["balanced_accuracy"]
        ),
        "decision_rows_oof_precision": float(
            result["decision_oof_metrics"]["precision"]
        ),
        "decision_rows_oof_recall": float(result["decision_oof_metrics"]["recall"]),
        "decision_rows_oof_f1": float(result["decision_oof_metrics"]["f1"]),
        "decision_rows_oof_brier": float(
            result["decision_oof_metrics"]["brier_score"]
        ),
        "decision_rows_oof_logloss": float(
            result["decision_oof_metrics"]["binary_logloss"]
        ),
        "prediction_scope_accuracy": float(result["prediction_scope_metrics"]["accuracy"]),
        "prediction_scope_balanced_accuracy": float(
            result["prediction_scope_metrics"]["balanced_accuracy"]
        ),
        "prediction_scope_brier": float(
            result["prediction_scope_metrics"]["brier_score"]
        ),
        "prediction_scope_logloss": float(
            result["prediction_scope_metrics"]["binary_logloss"]
        ),
        "oof_coverage_ratio": float(result["oof_coverage_ratio"]),
        "mean_best_iteration": int(result["mean_best_iteration"]),
    }
    if result["oof_scope_metrics"] is not None:
        all_rows_metrics = result["oof_scope_metrics"]["all_rows"]["metrics"]
        if all_rows_metrics is not None:
            row["all_rows_oof_accuracy"] = float(all_rows_metrics["accuracy"])
            row["all_rows_oof_balanced_accuracy"] = float(
                all_rows_metrics["balanced_accuracy"]
            )
            row["all_rows_oof_brier"] = float(all_rows_metrics["brier_score"])
            row["all_rows_oof_logloss"] = float(
                all_rows_metrics["binary_logloss"]
            )
    if weight_config is None:
        row["decision_weight"] = None
        row["other_weight"] = None
        row["non_decision_total_weight"] = None
        row["total_block_weight"] = None
    else:
        row["decision_weight"] = float(weight_config["decision_weight"])
        row["other_weight"] = float(weight_config["other_weight"])
        row["non_decision_total_weight"] = float(
            weight_config["non_decision_total_weight"]
        )
        row["total_block_weight"] = float(weight_config["total_block_weight"])
    return row


def attach_context_metadata_to_fold_metrics(
    fold_df,
    *,
    stage,
    evaluation_context,
    strategy_name,
    weight_config,
):
    out = fold_df.copy()
    out.insert(0, "stage", stage)
    out.insert(1, "context_id", evaluation_context["id"])
    out.insert(2, "context_label", evaluation_context["label"])
    out.insert(3, "feature_view_id", evaluation_context["feature_view"]["id"])
    out.insert(4, "feature_view_label", evaluation_context["feature_view"]["label"])
    out.insert(5, "param_profile_name", evaluation_context["param_profile"]["name"])
    out.insert(6, "is_primary_context", bool(evaluation_context["is_primary"]))
    out.insert(7, "strategy_name", str(strategy_name))
    out["decision_weight"] = (
        None if weight_config is None else float(weight_config["decision_weight"])
    )
    return out


def build_context_weight_summary_row(
    context_rows,
    *,
    stage,
    strategy_name,
    weight_config,
    n_estimators,
    cv_folds,
    search_round,
    context_std_penalty,
):
    if not context_rows:
        raise ValueError("context_rows must not be empty.")

    objective_values = np.asarray(
        [row["objective_value"] for row in context_rows],
        dtype=np.float64,
    )
    decision_bal_acc_values = np.asarray(
        [row["decision_rows_oof_balanced_accuracy"] for row in context_rows],
        dtype=np.float64,
    )
    mean_best_iterations = np.asarray(
        [row["mean_best_iteration"] for row in context_rows],
        dtype=np.float64,
    )
    primary_row = next(
        (row for row in context_rows if bool(row.get("is_primary_context"))),
        context_rows[0],
    )
    context_objective_mean = float(np.mean(objective_values))
    context_objective_std = float(np.std(objective_values, ddof=0))
    decision_bal_acc_mean = float(np.mean(decision_bal_acc_values))
    decision_bal_acc_std = float(np.std(decision_bal_acc_values, ddof=0))

    return {
        "stage": stage,
        "strategy_name": str(strategy_name),
        "row_mode": primary_row["row_mode"],
        "eval_scope": primary_row["eval_scope"],
        "prediction_scope": primary_row["prediction_scope"],
        "cv_folds": int(cv_folds),
        "n_estimators": int(n_estimators),
        "search_round": search_round,
        "contexts_evaluated": int(len(context_rows)),
        "objective_value": float(
            context_objective_mean - (float(context_std_penalty) * context_objective_std)
        ),
        "context_objective_mean": context_objective_mean,
        "context_objective_std": context_objective_std,
        "context_objective_min": float(np.min(objective_values)),
        "context_objective_max": float(np.max(objective_values)),
        "decision_rows_bal_acc_mean": decision_bal_acc_mean,
        "decision_rows_bal_acc_std": decision_bal_acc_std,
        "decision_rows_bal_acc_min": float(np.min(decision_bal_acc_values)),
        "decision_rows_bal_acc_max": float(np.max(decision_bal_acc_values)),
        "primary_context_id": primary_row["context_id"],
        "primary_context_label": primary_row["context_label"],
        "primary_context_objective_value": float(primary_row["objective_value"]),
        "primary_context_decision_rows_oof_balanced_accuracy": float(
            primary_row["decision_rows_oof_balanced_accuracy"]
        ),
        "mean_best_iteration": int(np.round(np.mean(mean_best_iterations))),
        "decision_weight": (
            None if weight_config is None else float(weight_config["decision_weight"])
        ),
        "other_weight": (
            None if weight_config is None else float(weight_config["other_weight"])
        ),
        "non_decision_total_weight": (
            None
            if weight_config is None
            else float(weight_config["non_decision_total_weight"])
        ),
        "total_block_weight": (
            None if weight_config is None else float(weight_config["total_block_weight"])
        ),
    }


def _top_aggregate_weight_results(weight_rows, *, limit):
    if not weight_rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(weight_rows)
        .sort_values(
            by=[
                "objective_value",
                "context_objective_mean",
                "decision_rows_bal_acc_mean",
                "decision_weight",
            ],
            ascending=[False, False, False, True],
            kind="stable",
        )
        .head(int(limit))
        .reset_index(drop=True)
    )


def sort_context_results_df(context_results_df):
    if context_results_df.empty:
        return context_results_df

    out = context_results_df.copy()
    out["_decision_weight_sort"] = pd.to_numeric(
        out["decision_weight"],
        errors="coerce",
    ).fillna(-1.0)
    out = out.sort_values(
        by=[
            "context_label",
            "strategy_name",
            "_decision_weight_sort",
            "objective_value",
            "decision_rows_oof_balanced_accuracy",
        ],
        ascending=[True, True, True, False, False],
        kind="stable",
    ).drop(columns=["_decision_weight_sort"])
    return out.reset_index(drop=True)


def sort_aggregate_strategy_results(results_df):
    if results_df.empty:
        return results_df

    out = results_df.copy()
    out["_decision_weight_sort"] = pd.to_numeric(
        out["decision_weight"],
        errors="coerce",
    ).fillna(-1.0)
    out = out.sort_values(
        by=[
            "objective_value",
            "context_objective_mean",
            "decision_rows_bal_acc_mean",
            "strategy_name",
            "_decision_weight_sort",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).drop(columns=["_decision_weight_sort"])
    return out.reset_index(drop=True)


def enrich_context_results_with_baseline_deltas(context_results_df):
    if context_results_df.empty:
        return context_results_df

    out = context_results_df.copy()
    out["baseline_objective_value"] = np.nan
    out["baseline_decision_rows_oof_balanced_accuracy"] = np.nan
    out["objective_delta_vs_context_baseline"] = np.nan
    out["decision_bal_acc_delta_vs_context_baseline"] = np.nan

    baseline_rows = (
        out[out["strategy_name"] == STRATEGY_DECISION_ONLY_BASELINE]
        .set_index("context_id")
        .to_dict(orient="index")
    )
    for row_idx, row in out.iterrows():
        baseline = baseline_rows.get(row["context_id"])
        if baseline is None:
            continue
        out.at[row_idx, "baseline_objective_value"] = float(
            baseline["objective_value"]
        )
        out.at[row_idx, "baseline_decision_rows_oof_balanced_accuracy"] = float(
            baseline["decision_rows_oof_balanced_accuracy"]
        )
        out.at[row_idx, "objective_delta_vs_context_baseline"] = float(
            row["objective_value"] - baseline["objective_value"]
        )
        out.at[row_idx, "decision_bal_acc_delta_vs_context_baseline"] = float(
            row["decision_rows_oof_balanced_accuracy"]
            - baseline["decision_rows_oof_balanced_accuracy"]
        )
    return out


def enrich_aggregate_results_with_baseline_deltas(
    aggregate_results_df,
    *,
    context_results_df,
):
    if aggregate_results_df.empty:
        return aggregate_results_df

    out = aggregate_results_df.copy()
    out["baseline_objective_value"] = np.nan
    out["baseline_context_objective_mean"] = np.nan
    out["baseline_context_objective_std"] = np.nan
    out["baseline_decision_rows_bal_acc_mean"] = np.nan
    out["objective_delta_vs_baseline"] = np.nan
    out["context_objective_mean_delta_vs_baseline"] = np.nan
    out["decision_rows_bal_acc_mean_delta_vs_baseline"] = np.nan
    out["contexts_beating_baseline"] = np.nan
    out["contexts_losing_to_baseline"] = np.nan
    out["contexts_tying_baseline"] = np.nan

    baseline_df = out[out["strategy_name"] == STRATEGY_DECISION_ONLY_BASELINE]
    if baseline_df.empty:
        return out

    baseline_row = baseline_df.iloc[0]
    baseline_objective = float(baseline_row["objective_value"])
    baseline_context_mean = float(baseline_row["context_objective_mean"])
    baseline_context_std = float(baseline_row["context_objective_std"])
    baseline_decision_bal_acc_mean = float(baseline_row["decision_rows_bal_acc_mean"])
    baseline_context_rows = context_results_df[
        context_results_df["strategy_name"] == STRATEGY_DECISION_ONLY_BASELINE
    ]
    baseline_context_by_id = baseline_context_rows.set_index("context_id").to_dict(
        orient="index"
    )

    for row_idx, row in out.iterrows():
        out.at[row_idx, "baseline_objective_value"] = baseline_objective
        out.at[row_idx, "baseline_context_objective_mean"] = baseline_context_mean
        out.at[row_idx, "baseline_context_objective_std"] = baseline_context_std
        out.at[row_idx, "baseline_decision_rows_bal_acc_mean"] = (
            baseline_decision_bal_acc_mean
        )
        out.at[row_idx, "objective_delta_vs_baseline"] = float(
            row["objective_value"] - baseline_objective
        )
        out.at[row_idx, "context_objective_mean_delta_vs_baseline"] = float(
            row["context_objective_mean"] - baseline_context_mean
        )
        out.at[row_idx, "decision_rows_bal_acc_mean_delta_vs_baseline"] = float(
            row["decision_rows_bal_acc_mean"] - baseline_decision_bal_acc_mean
        )
        if row["strategy_name"] != STRATEGY_ALL_ROWS_WEIGHTED:
            continue

        row_contexts = context_results_df[
            (context_results_df["strategy_name"] == STRATEGY_ALL_ROWS_WEIGHTED)
            & np.isclose(
                pd.to_numeric(
                    context_results_df["decision_weight"],
                    errors="coerce",
                ).to_numpy(dtype=np.float64),
                float(row["decision_weight"]),
                equal_nan=False,
            )
        ]
        wins = 0
        losses = 0
        ties = 0
        for _, context_row in row_contexts.iterrows():
            baseline_context = baseline_context_by_id.get(context_row["context_id"])
            if baseline_context is None:
                continue
            context_delta = float(
                context_row["objective_value"] - baseline_context["objective_value"]
            )
            if context_delta > 0.0:
                wins += 1
            elif context_delta < 0.0:
                losses += 1
            else:
                ties += 1
        out.at[row_idx, "contexts_beating_baseline"] = int(wins)
        out.at[row_idx, "contexts_losing_to_baseline"] = int(losses)
        out.at[row_idx, "contexts_tying_baseline"] = int(ties)
    return out


def build_baseline_recommendation_summary(final_results_df):
    if final_results_df.empty:
        return None

    baseline_df = final_results_df[
        final_results_df["strategy_name"] == STRATEGY_DECISION_ONLY_BASELINE
    ]
    weighted_df = final_results_df[
        final_results_df["strategy_name"] == STRATEGY_ALL_ROWS_WEIGHTED
    ]
    baseline_row = None if baseline_df.empty else baseline_df.iloc[0]
    best_weighted_row = None if weighted_df.empty else weighted_df.iloc[0]

    if baseline_row is None:
        recommended_row = best_weighted_row
        recommendation_reason = "no_context_baseline"
    elif best_weighted_row is None:
        recommended_row = baseline_row
        recommendation_reason = "no_weighted_candidate"
    elif float(best_weighted_row["objective_value"]) > float(
        baseline_row["objective_value"]
    ):
        recommended_row = best_weighted_row
        recommendation_reason = "weighted_beats_context_baseline"
    else:
        recommended_row = baseline_row
        recommendation_reason = "context_baseline_beats_weighted"

    if recommended_row is None:
        return None

    return {
        "baseline_strategy_name": (
            None if baseline_row is None else str(baseline_row["strategy_name"])
        ),
        "baseline_objective_value": (
            None if baseline_row is None else float(baseline_row["objective_value"])
        ),
        "baseline_decision_rows_bal_acc_mean": (
            None
            if baseline_row is None
            else float(baseline_row["decision_rows_bal_acc_mean"])
        ),
        "best_weighted_decision_weight": (
            None
            if best_weighted_row is None
            else float(best_weighted_row["decision_weight"])
        ),
        "best_weighted_objective_value": (
            None
            if best_weighted_row is None
            else float(best_weighted_row["objective_value"])
        ),
        "best_weighted_decision_rows_bal_acc_mean": (
            None
            if best_weighted_row is None
            else float(best_weighted_row["decision_rows_bal_acc_mean"])
        ),
        "recommended_strategy_name": str(recommended_row["strategy_name"]),
        "recommended_decision_weight": (
            None
            if pd.isna(recommended_row["decision_weight"])
            else float(recommended_row["decision_weight"])
        ),
        "recommended_objective_value": float(recommended_row["objective_value"]),
        "recommended_decision_rows_bal_acc_mean": float(
            recommended_row["decision_rows_bal_acc_mean"]
        ),
        "recommendation_reason": recommendation_reason,
    }


def evaluate_strategy_across_contexts(
    *,
    stage,
    evaluation_contexts,
    strategy_name,
    row_mode,
    weight_config,
    y,
    decision_mask,
    folds,
    fold_weight_by_id,
    float_dtype,
    n_estimators,
    early_stopping_rounds,
    prediction_scope,
    search_round,
    context_std_penalty,
):
    context_rows = []
    fold_frames = []
    context_results = []

    for evaluation_context in evaluation_contexts:
        model_variant_suffix = (
            "baseline"
            if weight_config is None
            else _format_candidate_weight(weight_config["decision_weight"])
        )
        context_result = evaluate_strategy(
            x=evaluation_context["x"],
            y=y,
            decision_mask=decision_mask,
            folds=folds,
            fold_weight_by_id=fold_weight_by_id,
            param_overrides=evaluation_context["param_overrides"],
            row_mode=row_mode,
            eval_scope=EVAL_SCOPE_DECISION_ONLY,
            prediction_scope=prediction_scope,
            weight_config=weight_config,
            std_penalty=float(OBJECTIVE_STD_PENALTY),
            float_dtype=float_dtype,
            model_variant=(
                f"{evaluation_context['id']}_{stage}_{strategy_name}_"
                f"{model_variant_suffix}"
            ),
            n_estimators=int(n_estimators),
            early_stopping_rounds=int(early_stopping_rounds),
        )
        context_row = build_context_result_row(
            stage=stage,
            evaluation_context=evaluation_context,
            strategy_name=strategy_name,
            result=context_result,
            weight_config=weight_config,
            n_estimators=int(n_estimators),
            cv_folds=len(folds),
            search_round=search_round,
        )
        context_rows.append(context_row)
        fold_frames.append(
            attach_context_metadata_to_fold_metrics(
                context_result["fold_metrics"],
                stage=stage,
                evaluation_context=evaluation_context,
                strategy_name=strategy_name,
                weight_config=weight_config,
            )
        )
        context_results.append(
            {
                "context": evaluation_context,
                "row": context_row,
                "result": context_result,
                "weight_config": weight_config,
            }
        )

    aggregate_row = build_context_weight_summary_row(
        context_rows,
        stage=stage,
        strategy_name=strategy_name,
        weight_config=weight_config,
        n_estimators=int(n_estimators),
        cv_folds=len(folds),
        search_round=search_round,
        context_std_penalty=float(context_std_penalty),
    )
    return {
        "aggregate_row": aggregate_row,
        "context_rows": context_rows,
        "fold_metrics": pd.concat(fold_frames, ignore_index=True),
        "context_results": context_results,
        "weight_config": weight_config,
    }


def evaluate_weight_candidate_across_contexts(
    *,
    stage,
    evaluation_contexts,
    decision_weight,
    y,
    decision_mask,
    folds,
    fold_weight_by_id,
    float_dtype,
    n_estimators,
    early_stopping_rounds,
    prediction_scope,
    search_round,
    context_std_penalty,
):
    weight_config = build_weight_config(decision_weight)
    return evaluate_strategy_across_contexts(
        stage=stage,
        evaluation_contexts=evaluation_contexts,
        strategy_name=STRATEGY_ALL_ROWS_WEIGHTED,
        row_mode=ROW_MODE_ALL_ROWS,
        weight_config=weight_config,
        y=y,
        decision_mask=decision_mask,
        folds=folds,
        fold_weight_by_id=fold_weight_by_id,
        float_dtype=float_dtype,
        n_estimators=int(n_estimators),
        early_stopping_rounds=int(early_stopping_rounds),
        prediction_scope=prediction_scope,
        search_round=search_round,
        context_std_penalty=float(context_std_penalty),
    )


def evaluate_baseline_across_contexts(
    *,
    stage,
    evaluation_contexts,
    y,
    decision_mask,
    folds,
    fold_weight_by_id,
    float_dtype,
    n_estimators,
    early_stopping_rounds,
    prediction_scope,
    context_std_penalty,
):
    return evaluate_strategy_across_contexts(
        stage=stage,
        evaluation_contexts=evaluation_contexts,
        strategy_name=STRATEGY_DECISION_ONLY_BASELINE,
        row_mode=ROW_MODE_DECISION_ONLY,
        weight_config=None,
        y=y,
        decision_mask=decision_mask,
        folds=folds,
        fold_weight_by_id=fold_weight_by_id,
        float_dtype=float_dtype,
        n_estimators=int(n_estimators),
        early_stopping_rounds=int(early_stopping_rounds),
        prediction_scope=prediction_scope,
        search_round=None,
        context_std_penalty=float(context_std_penalty),
    )


def run_proxy_weight_search_across_contexts(
    *,
    evaluation_contexts,
    y,
    decision_mask,
    folds,
    fold_weight_by_id,
    float_dtype,
    context_std_penalty,
):
    search_rows = []
    search_context_rows = []
    search_fold_frames = []
    evaluated_by_weight = {}
    pending_weights = list(build_initial_weight_candidates())

    for round_idx in range(int(SEARCH_REFINEMENT_ROUNDS) + 1):
        if not pending_weights:
            break

        for decision_weight in pending_weights:
            evaluated = evaluate_weight_candidate_across_contexts(
                stage="proxy_search",
                evaluation_contexts=evaluation_contexts,
                decision_weight=decision_weight,
                y=y,
                decision_mask=decision_mask,
                folds=folds,
                fold_weight_by_id=fold_weight_by_id,
                float_dtype=float_dtype,
                n_estimators=int(SEARCH_N_ESTIMATORS),
                early_stopping_rounds=int(SEARCH_EARLY_STOPPING_ROUNDS),
                prediction_scope=PREDICTION_SCOPE_DECISION_ONLY,
                search_round=round_idx,
                context_std_penalty=float(context_std_penalty),
            )
            evaluated_by_weight[float(decision_weight)] = evaluated
            search_rows.append(evaluated["aggregate_row"])
            search_context_rows.extend(evaluated["context_rows"])
            search_fold_frames.append(evaluated["fold_metrics"])
            print(
                "proxy weight_search "
                f"round={round_idx} decision_weight={float(decision_weight):.6f} "
                f"objective={evaluated['aggregate_row']['objective_value']:.6f} "
                f"context_mean={evaluated['aggregate_row']['context_objective_mean']:.6f} "
                f"context_std={evaluated['aggregate_row']['context_objective_std']:.6f} "
                f"decision_bal_acc_mean={evaluated['aggregate_row']['decision_rows_bal_acc_mean']:.6f}"
            )

        if round_idx >= int(SEARCH_REFINEMENT_ROUNDS):
            break

        top_parent_weights = _top_aggregate_weight_results(
            search_rows,
            limit=int(SEARCH_TOP_PARENT_WEIGHTS),
        )["decision_weight"].tolist()
        pending_weights = list(
            build_refined_weight_candidates(
                evaluated_by_weight.keys(),
                top_parent_weights,
            )
        )

    shortlist_df = _top_aggregate_weight_results(
        search_rows,
        limit=int(FINAL_TOP_WEIGHTED_CANDIDATES),
    )
    ranked_search_df = _top_aggregate_weight_results(
        search_rows,
        limit=len(search_rows),
    )
    shortlist_weights = _dedupe_float_candidates(
        [float(DECISION_WEIGHT_LOW), *shortlist_df["decision_weight"].tolist()]
    )
    shortlist_rows = []
    for shortlist_weight in shortlist_weights:
        aggregate_row = evaluated_by_weight[float(shortlist_weight)]["aggregate_row"].copy()
        proxy_rank_series = ranked_search_df.index[
            np.isclose(
                ranked_search_df["decision_weight"].to_numpy(dtype=np.float64),
                float(shortlist_weight),
            )
        ]
        aggregate_row["proxy_rank"] = int(proxy_rank_series[0] + 1)
        shortlist_rows.append(aggregate_row)

    return {
        "search_rows": search_rows,
        "search_context_rows": search_context_rows,
        "search_fold_metrics": pd.concat(search_fold_frames, ignore_index=True),
        "weighted_shortlist": shortlist_rows,
        "evaluated_by_weight": evaluated_by_weight,
    }


def run_final_weight_recheck_across_contexts(
    *,
    evaluation_contexts,
    shortlist_rows,
    y,
    decision_mask,
    folds,
    fold_weight_by_id,
    float_dtype,
    context_std_penalty,
):
    final_rows = []
    final_context_rows = []
    final_fold_frames = []
    evaluated_by_weight = {}

    for shortlist_row in shortlist_rows:
        decision_weight = float(shortlist_row["decision_weight"])
        evaluated = evaluate_weight_candidate_across_contexts(
            stage="final_recheck",
            evaluation_contexts=evaluation_contexts,
            decision_weight=decision_weight,
            y=y,
            decision_mask=decision_mask,
            folds=folds,
            fold_weight_by_id=fold_weight_by_id,
            float_dtype=float_dtype,
            n_estimators=int(N_ESTIMATORS),
            early_stopping_rounds=int(EARLY_STOPPING_ROUNDS),
            prediction_scope=PREDICTION_SCOPE_ALL_ROWS,
            search_round=None,
            context_std_penalty=float(context_std_penalty),
        )
        aggregate_row = {
            **evaluated["aggregate_row"],
            "proxy_objective_value": float(shortlist_row["objective_value"]),
            "proxy_rank": int(shortlist_row["proxy_rank"]),
        }
        evaluated_by_weight[decision_weight] = evaluated
        final_rows.append(aggregate_row)
        final_context_rows.extend(evaluated["context_rows"])
        final_fold_frames.append(evaluated["fold_metrics"])
        print(
            "final weight_search "
            f"decision_weight={decision_weight:.6f} "
            f"objective={aggregate_row['objective_value']:.6f} "
            f"context_mean={aggregate_row['context_objective_mean']:.6f} "
            f"context_std={aggregate_row['context_objective_std']:.6f} "
            f"decision_bal_acc_mean={aggregate_row['decision_rows_bal_acc_mean']:.6f}"
        )

    final_results_df = _top_aggregate_weight_results(
        final_rows,
        limit=len(final_rows),
    )
    final_context_results_df = pd.DataFrame(final_context_rows)
    final_context_results_df = final_context_results_df.sort_values(
        by=[
            "decision_weight",
            "objective_value",
            "decision_rows_oof_balanced_accuracy",
        ],
        ascending=[True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return {
        "final_results_df": final_results_df,
        "final_context_results_df": final_context_results_df,
        "final_fold_metrics_df": pd.concat(final_fold_frames, ignore_index=True),
        "evaluated_by_weight": evaluated_by_weight,
    }


def _top_weight_results(weight_rows, *, limit):
    return (
        pd.DataFrame(weight_rows)
        .sort_values(
            by=[
                "objective_value",
                "decision_metric_weighted_mean",
                "decision_rows_oof_balanced_accuracy",
                "decision_weight",
            ],
            ascending=[False, False, False, True],
            kind="stable",
        )
        .head(int(limit))
    )


def run_proxy_weight_search_for_subset(
    *,
    feature_subset_candidate,
    x,
    y,
    decision_mask,
    folds,
    fold_weight_by_id,
    param_overrides,
    float_dtype,
):
    search_rows = []
    search_fold_frames = []

    baseline_result = evaluate_strategy(
        x=x,
        y=y,
        decision_mask=decision_mask,
        folds=folds,
        fold_weight_by_id=fold_weight_by_id,
        param_overrides=param_overrides,
        row_mode=ROW_MODE_DECISION_ONLY,
        eval_scope=EVAL_SCOPE_DECISION_ONLY,
        prediction_scope=PREDICTION_SCOPE_DECISION_ONLY,
        weight_config=None,
        std_penalty=float(OBJECTIVE_STD_PENALTY),
        float_dtype=float_dtype,
        model_variant=f"{feature_subset_candidate['id']}_baseline_proxy",
        n_estimators=int(SEARCH_N_ESTIMATORS),
        early_stopping_rounds=int(SEARCH_EARLY_STOPPING_ROUNDS),
    )
    baseline_row = build_result_row(
        stage="proxy_search",
        feature_subset_candidate=feature_subset_candidate,
        strategy_name=STRATEGY_DECISION_ONLY_BASELINE,
        result=baseline_result,
        weight_config=None,
        n_estimators=int(SEARCH_N_ESTIMATORS),
        cv_folds=len(folds),
        search_round=0,
    )
    search_rows.append(baseline_row)
    search_fold_frames.append(
        attach_metadata_to_fold_metrics(
            baseline_result["fold_metrics"],
            stage="proxy_search",
            feature_subset_candidate=feature_subset_candidate,
            strategy_name=STRATEGY_DECISION_ONLY_BASELINE,
            weight_config=None,
        )
    )

    weighted_results = {}
    pending_weights = list(build_initial_weight_candidates())
    for round_idx in range(int(SEARCH_REFINEMENT_ROUNDS) + 1):
        if not pending_weights:
            break

        for decision_weight in pending_weights:
            weight_config = build_weight_config(decision_weight)
            weighted_result = evaluate_strategy(
                x=x,
                y=y,
                decision_mask=decision_mask,
                folds=folds,
                fold_weight_by_id=fold_weight_by_id,
                param_overrides=param_overrides,
                row_mode=ROW_MODE_ALL_ROWS,
                eval_scope=EVAL_SCOPE_DECISION_ONLY,
                prediction_scope=PREDICTION_SCOPE_DECISION_ONLY,
                weight_config=weight_config,
                std_penalty=float(OBJECTIVE_STD_PENALTY),
                float_dtype=float_dtype,
                model_variant=(
                    f"{feature_subset_candidate['id']}_proxy_round_{round_idx}_"
                    f"w_{_format_candidate_weight(decision_weight)}"
                ),
                n_estimators=int(SEARCH_N_ESTIMATORS),
                early_stopping_rounds=int(SEARCH_EARLY_STOPPING_ROUNDS),
            )
            weighted_results[float(decision_weight)] = {
                "weight_config": weight_config,
                "result": weighted_result,
            }
            search_rows.append(
                build_result_row(
                    stage="proxy_search",
                    feature_subset_candidate=feature_subset_candidate,
                    strategy_name=STRATEGY_ALL_ROWS_WEIGHTED,
                    result=weighted_result,
                    weight_config=weight_config,
                    n_estimators=int(SEARCH_N_ESTIMATORS),
                    cv_folds=len(folds),
                    search_round=round_idx,
                )
            )
            search_fold_frames.append(
                attach_metadata_to_fold_metrics(
                    weighted_result["fold_metrics"],
                    stage="proxy_search",
                    feature_subset_candidate=feature_subset_candidate,
                    strategy_name=STRATEGY_ALL_ROWS_WEIGHTED,
                    weight_config=weight_config,
                )
            )
            print(
                f"proxy subset={feature_subset_candidate['label']} "
                f"round={round_idx} decision_weight={decision_weight:.6f} "
                f"objective={weighted_result['decision_metric_objective']:.6f} "
                f"decision_bal_acc={weighted_result['decision_oof_metrics']['balanced_accuracy']:.6f}"
            )

        if round_idx >= int(SEARCH_REFINEMENT_ROUNDS):
            break

        weighted_rows = [
            row
            for row in search_rows
            if row["strategy_name"] == STRATEGY_ALL_ROWS_WEIGHTED
        ]
        weighted_df = _top_weight_results(
            weighted_rows,
            limit=int(SEARCH_TOP_PARENT_WEIGHTS),
        )
        top_parent_weights = weighted_df["decision_weight"].tolist()
        pending_weights = list(
            build_refined_weight_candidates(
                weighted_results.keys(),
                top_parent_weights,
            )
        )

    weighted_rows = [
        row
        for row in search_rows
        if row["strategy_name"] == STRATEGY_ALL_ROWS_WEIGHTED
    ]
    top_weight_df = _top_weight_results(
        weighted_rows,
        limit=int(TOP_WEIGHT_CANDIDATES_PER_SUBSET),
    )
    return {
        "feature_subset_candidate": feature_subset_candidate,
        "baseline_row": baseline_row,
        "weighted_shortlist": top_weight_df.to_dict(orient="records"),
        "search_rows": search_rows,
        "search_fold_metrics": pd.concat(search_fold_frames, ignore_index=True),
    }


def build_final_candidate_specs(proxy_subset_results):
    final_specs = []
    for subset_result in proxy_subset_results:
        final_specs.append(
            {
                "feature_subset_candidate": subset_result["feature_subset_candidate"],
                "strategy_name": STRATEGY_DECISION_ONLY_BASELINE,
                "decision_weight": None,
            }
        )

    weighted_specs = []
    for subset_result in proxy_subset_results:
        for row in subset_result["weighted_shortlist"]:
            weighted_specs.append(
                {
                    "feature_subset_candidate": subset_result["feature_subset_candidate"],
                    "strategy_name": STRATEGY_ALL_ROWS_WEIGHTED,
                    "decision_weight": float(row["decision_weight"]),
                    "proxy_objective_value": float(row["objective_value"]),
                }
            )

    weighted_specs = sorted(
        weighted_specs,
        key=lambda row: (
            -float(row["proxy_objective_value"]),
            float(row["decision_weight"]),
        ),
    )[: int(FINAL_TOP_WEIGHTED_CANDIDATES)]
    final_specs.extend(weighted_specs)
    return final_specs


def evaluate_final_candidate(
    *,
    candidate_spec,
    x,
    y,
    decision_mask,
    folds,
    fold_weight_by_id,
    param_overrides,
    float_dtype,
):
    strategy_name = candidate_spec["strategy_name"]
    feature_subset_candidate = candidate_spec["feature_subset_candidate"]
    if strategy_name == STRATEGY_DECISION_ONLY_BASELINE:
        row_mode = ROW_MODE_DECISION_ONLY
        weight_config = None
    elif strategy_name == STRATEGY_ALL_ROWS_WEIGHTED:
        row_mode = ROW_MODE_ALL_ROWS
        weight_config = build_weight_config(candidate_spec["decision_weight"])
    else:
        raise ValueError(f"Unsupported strategy_name: {strategy_name}")

    result = evaluate_strategy(
        x=x,
        y=y,
        decision_mask=decision_mask,
        folds=folds,
        fold_weight_by_id=fold_weight_by_id,
        param_overrides=param_overrides,
        row_mode=row_mode,
        eval_scope=EVAL_SCOPE_DECISION_ONLY,
        prediction_scope=PREDICTION_SCOPE_ALL_ROWS,
        weight_config=weight_config,
        std_penalty=float(OBJECTIVE_STD_PENALTY),
        float_dtype=float_dtype,
        model_variant=(
            f"{feature_subset_candidate['id']}_final_{strategy_name}_"
            f"{'baseline' if weight_config is None else _format_candidate_weight(weight_config['decision_weight'])}"
        ),
        n_estimators=int(N_ESTIMATORS),
        early_stopping_rounds=int(EARLY_STOPPING_ROUNDS),
    )
    result_row = build_result_row(
        stage="final_recheck",
        feature_subset_candidate=feature_subset_candidate,
        strategy_name=strategy_name,
        result=result,
        weight_config=weight_config,
        n_estimators=int(N_ESTIMATORS),
        cv_folds=len(folds),
    )
    fold_df = attach_metadata_to_fold_metrics(
        result["fold_metrics"],
        stage="final_recheck",
        feature_subset_candidate=feature_subset_candidate,
        strategy_name=strategy_name,
        weight_config=weight_config,
    )
    return {
        "row": result_row,
        "result": result,
        "fold_metrics": fold_df,
        "weight_config": weight_config,
    }


def enrich_final_results_with_baseline_deltas(final_results_df):
    if final_results_df.empty:
        return final_results_df

    out = final_results_df.copy()
    out["objective_delta_vs_subset_baseline"] = np.nan
    out["decision_bal_acc_delta_vs_subset_baseline"] = np.nan
    baseline_rows = (
        out[out["strategy_name"] == STRATEGY_DECISION_ONLY_BASELINE]
        .set_index("feature_subset_id")
        .to_dict(orient="index")
    )
    for row_idx, row in out.iterrows():
        baseline = baseline_rows.get(row["feature_subset_id"])
        if baseline is None:
            continue
        out.at[row_idx, "objective_delta_vs_subset_baseline"] = float(
            row["objective_value"] - baseline["objective_value"]
        )
        out.at[row_idx, "decision_bal_acc_delta_vs_subset_baseline"] = float(
            row["decision_rows_oof_balanced_accuracy"]
            - baseline["decision_rows_oof_balanced_accuracy"]
        )
    return out


def build_subset_summary_rows(final_results_df):
    summary_rows = []
    if final_results_df.empty:
        return pd.DataFrame(summary_rows)

    for subset_id, subset_df in final_results_df.groupby("feature_subset_id", sort=False):
        subset_df = subset_df.sort_values(
            by=[
                "objective_value",
                "decision_metric_weighted_mean",
                "decision_rows_oof_balanced_accuracy",
            ],
            ascending=[False, False, False],
            kind="stable",
        )
        baseline_df = subset_df[
            subset_df["strategy_name"] == STRATEGY_DECISION_ONLY_BASELINE
        ]
        weighted_df = subset_df[
            subset_df["strategy_name"] == STRATEGY_ALL_ROWS_WEIGHTED
        ]
        baseline_row = None if baseline_df.empty else baseline_df.iloc[0]
        best_weighted_row = None if weighted_df.empty else weighted_df.iloc[0]

        if baseline_row is None:
            recommended_row = best_weighted_row
            recommendation_reason = "no_subset_baseline"
        elif best_weighted_row is None:
            recommended_row = baseline_row
            recommendation_reason = "no_weighted_candidate"
        elif float(best_weighted_row["objective_value"]) > float(
            baseline_row["objective_value"]
        ):
            recommended_row = best_weighted_row
            recommendation_reason = "weighted_beats_subset_baseline"
        else:
            recommended_row = baseline_row
            recommendation_reason = "subset_baseline_beats_weighted"

        summary_rows.append(
            {
                "feature_subset_id": subset_id,
                "feature_subset_label": subset_df["feature_subset_label"].iloc[0],
                "feature_subset_path": subset_df["feature_subset_path"].iloc[0],
                "feature_count": int(subset_df["feature_count"].iloc[0]),
                "is_active_feature_subset": bool(
                    subset_df["is_active_feature_subset"].iloc[0]
                ),
                "baseline_objective_value": (
                    None if baseline_row is None else float(baseline_row["objective_value"])
                ),
                "baseline_decision_bal_acc": (
                    None
                    if baseline_row is None
                    else float(baseline_row["decision_rows_oof_balanced_accuracy"])
                ),
                "best_weighted_objective_value": (
                    None
                    if best_weighted_row is None
                    else float(best_weighted_row["objective_value"])
                ),
                "best_weighted_decision_weight": (
                    None
                    if best_weighted_row is None
                    else float(best_weighted_row["decision_weight"])
                ),
                "best_weighted_decision_bal_acc": (
                    None
                    if best_weighted_row is None
                    else float(best_weighted_row["decision_rows_oof_balanced_accuracy"])
                ),
                "recommended_strategy_name": str(recommended_row["strategy_name"]),
                "recommended_objective_value": float(
                    recommended_row["objective_value"]
                ),
                "recommended_decision_weight": recommended_row["decision_weight"],
                "recommended_decision_bal_acc": float(
                    recommended_row["decision_rows_oof_balanced_accuracy"]
                ),
                "recommendation_reason": recommendation_reason,
            }
        )
    return pd.DataFrame(summary_rows)


def json_safe_value(value):
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe_value(item) for item in value]
    if isinstance(value, pd.Series):
        return json_safe_value(value.to_dict())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    return value


def build_best_result_payload(
    *,
    run_info,
    data_path,
    class_distribution,
    decision_mask,
    time_window,
    excluded_features,
    active_feature_subset,
    feature_views,
    param_profiles,
    evaluation_contexts,
    target_weight_search_settings,
    final_results_df,
    final_context_results_df,
    best_row,
):
    decision_mask_np = np.asarray(decision_mask, dtype=bool)
    best_strategy_name = str(best_row["strategy_name"])
    best_context_rows = final_context_results_df[
        final_context_results_df["strategy_name"] == best_strategy_name
    ].copy()
    if best_strategy_name == STRATEGY_ALL_ROWS_WEIGHTED:
        best_weight = float(best_row["decision_weight"])
        best_context_rows = best_context_rows[
            np.isclose(
                pd.to_numeric(
                    best_context_rows["decision_weight"],
                    errors="coerce",
                ).to_numpy(dtype=np.float64),
                best_weight,
                equal_nan=False,
            )
        ].copy()
    best_context_rows = best_context_rows.sort_values(
        by=[
            "objective_value",
            "decision_rows_oof_balanced_accuracy",
            "context_label",
        ],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    baseline_context_rows = final_context_results_df[
        final_context_results_df["strategy_name"] == STRATEGY_DECISION_ONLY_BASELINE
    ].copy()
    baseline_context_rows = baseline_context_rows.sort_values(
        by=[
            "objective_value",
            "decision_rows_oof_balanced_accuracy",
            "context_label",
        ],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    recommendation = build_baseline_recommendation_summary(final_results_df)
    baseline_df = final_results_df[
        final_results_df["strategy_name"] == STRATEGY_DECISION_ONLY_BASELINE
    ]
    baseline_row = None if baseline_df.empty else baseline_df.iloc[0]
    return {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "study_name": run_info["study_name"],
        "study_name_source": run_info["study_name_source"],
        "search_method": "deterministic_proxy_then_final_recheck_multi_context",
        "search_target": "decision_rows_weight_robust_across_feature_views_and_param_profiles",
        "data_path": path_to_portable_str(data_path),
        "target_col": TARGET_COL,
        "sample_weight_col": TARGET_WEIGHT_COL,
        "decision_row_definition": {
            "expression": "Opened.minute % 5 == 4",
            "rows": int(decision_mask_np.sum()),
            "total_rows": int(decision_mask_np.size),
            "share": float(decision_mask_np.mean()),
        },
        "time_window": (
            None
            if time_window is None
            else {
                "lookback_days": int(time_window["lookback_days"]),
                "opened_start_utc": pd.Timestamp(
                    time_window["opened_start_utc"]
                ).isoformat(),
                "opened_end_utc": pd.Timestamp(
                    time_window["opened_end_utc"]
                ).isoformat(),
                "rows_total": int(time_window["rows_total"]),
                "rows_in_window": int(time_window["rows_in_window"]),
            }
        ),
        "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
        "active_feature_selection": (
            None
            if active_feature_subset is None
            else summarize_feature_subset(
                active_feature_subset,
                excluded_features=excluded_features,
            )
        ),
        "feature_views": [
            {
                "id": feature_view["id"],
                "label": feature_view["label"],
                "path": feature_view["path"],
                "source_kind": feature_view["source_kind"],
                "feature_count": int(feature_view["feature_count"]),
                "is_active": bool(feature_view["is_active"]),
                "summary": feature_view["summary"],
                "metadata": feature_view["metadata"],
            }
            for feature_view in feature_views
        ],
        "param_profiles": [
            {
                "id": param_profile["id"],
                "name": param_profile["name"],
                "params": param_profile["params"],
            }
            for param_profile in param_profiles
        ],
        "evaluation_contexts": [
            {
                "id": evaluation_context["id"],
                "label": evaluation_context["label"],
                "feature_view_id": evaluation_context["feature_view"]["id"],
                "feature_view_label": evaluation_context["feature_view"]["label"],
                "param_profile_id": evaluation_context["param_profile"]["id"],
                "param_profile_name": evaluation_context["param_profile"]["name"],
                "is_primary": bool(evaluation_context["is_primary"]),
            }
            for evaluation_context in evaluation_contexts
        ],
        "proxy_search": {
            "cv_folds": int(SEARCH_CV_FOLDS),
            "n_estimators": int(SEARCH_N_ESTIMATORS),
            "early_stopping_rounds": int(SEARCH_EARLY_STOPPING_ROUNDS),
            "initial_weight_grid": [float(value) for value in build_initial_weight_candidates()],
            "refinement_rounds": int(SEARCH_REFINEMENT_ROUNDS),
            "top_parent_weights": int(SEARCH_TOP_PARENT_WEIGHTS),
            "final_top_weighted_candidates": int(FINAL_TOP_WEIGHTED_CANDIDATES),
        },
        "final_recheck": {
            "cv_folds": int(CV_FOLDS),
            "n_estimators": int(N_ESTIMATORS),
            "early_stopping_rounds": int(EARLY_STOPPING_ROUNDS),
            "device_type": str(DEVICE_TYPE),
        },
        "objective": {
            "metric": DECISION_ROW_OBJECTIVE_METRIC,
            "scope": "decision_rows_only",
            "fold_aggregation": decision_objective_aggregation_description(),
            "fold_std_penalty": float(OBJECTIVE_STD_PENALTY),
            "context_aggregation": (
                "context_objective_mean - context_std_penalty * context_objective_std"
            ),
            "context_std_penalty": float(
                target_weight_search_settings["context_std_penalty"]
            ),
            "fold_recency_weighting": {
                "enabled": bool(ENABLE_FOLD_RECENCY_WEIGHTING),
                "active": bool(is_nontrivial_fold_recency_weighting_enabled()),
                "mode": str(FOLD_RECENCY_WEIGHTING_MODE),
                "min_weight": float(FOLD_RECENCY_WEIGHT_MIN),
                "max_weight": float(FOLD_RECENCY_WEIGHT_MAX),
            },
        },
        "target_weight_search_settings": {
            "param_profiles": list(target_weight_search_settings["param_profiles"]),
            "include_all_features_view": bool(
                target_weight_search_settings["include_all_features_view"]
            ),
            "include_active_feature_subset_view": bool(
                target_weight_search_settings["include_active_feature_subset_view"]
            ),
            "random_feature_subsets": int(
                target_weight_search_settings["random_feature_subsets"]
            ),
            "random_feature_subset_size": target_weight_search_settings[
                "random_feature_subset_size"
            ],
            "random_feature_subset_fraction": target_weight_search_settings[
                "random_feature_subset_fraction"
            ],
            "random_feature_subset_min_features": target_weight_search_settings[
                "random_feature_subset_min_features"
            ],
            "lookback_days": target_weight_search_settings["lookback_days"],
        },
        "final_leaderboard": sort_aggregate_strategy_results(final_results_df).to_dict(
            orient="records"
        ),
        "baseline_context_breakdown": baseline_context_rows.to_dict(orient="records"),
        "best_context_breakdown": best_context_rows.to_dict(orient="records"),
        "baseline_configuration": (
            None if baseline_row is None else baseline_row.to_dict()
        ),
        "recommendation": recommendation,
        "best_configuration": best_row.to_dict(),
    }


def save_best_oof_predictions(
    *,
    output_path,
    df,
    decision_mask,
    final_result,
):
    export_df = df.loc[
        :,
        [
            col
            for col in ("Opened", "Open", "High", "Low", "Close", "Volume", TARGET_COL)
            if col in df.columns
        ],
    ].copy()
    export_df["decision_row"] = np.asarray(decision_mask, dtype=np.int8)
    export_df[TARGET_WEIGHT_COL] = final_result["result"]["sample_weight"].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    export_df["oof_pred_proba_up"] = final_result["result"]["oof_pred_proba"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_parquet(output_path, index=False)
    return export_df


def main():
    validate_config()

    dataset_settings = load_modeling_dataset_settings()
    target_weight_search_settings = load_target_weight_search_settings(dataset_settings)
    modeling_float_dtype = resolve_modeling_float_dtype(dataset_settings)
    modeling_float_dtype_name = resolve_modeling_float_dtype_name(dataset_settings)
    data_path = resolve_target_weight_search_dataset_path(
        dataset_settings,
        target_weight_search_settings,
    )
    time_window = resolve_target_weight_search_time_window(
        data_path,
        lookback_days=target_weight_search_settings["lookback_days"],
    )
    excluded_features = load_excluded_feature_names_from_settings(dataset_settings)
    active_feature_subset = load_feature_subset_from_settings(dataset_settings)
    training_data = load_target_weight_training_frame(
        data_path=data_path,
        excluded_features=excluded_features,
        float_dtype=modeling_float_dtype,
        opened_start_utc=(
            None if time_window is None else time_window["opened_start_utc"]
        ),
        opened_end_utc=None if time_window is None else time_window["opened_end_utc"],
    )
    df = training_data["df"].reset_index(drop=True)
    x_all = training_data["x"].reset_index(drop=True)
    y = training_data["y"].reset_index(drop=True)
    class_distribution = training_data["class_distribution"]
    decision_mask = compute_decision_mask_from_opened(df["Opened"])
    decision_summary = summarize_boolean_mask(decision_mask)
    feature_views = build_feature_view_candidates(
        x_all=x_all,
        active_subset=active_feature_subset,
        include_all_features_view=target_weight_search_settings[
            "include_all_features_view"
        ],
        include_active_feature_subset_view=target_weight_search_settings[
            "include_active_feature_subset_view"
        ],
        random_feature_subsets=target_weight_search_settings[
            "random_feature_subsets"
        ],
        random_feature_subset_size=target_weight_search_settings[
            "random_feature_subset_size"
        ],
        random_feature_subset_fraction=target_weight_search_settings[
            "random_feature_subset_fraction"
        ],
        random_feature_subset_min_features=target_weight_search_settings[
            "random_feature_subset_min_features"
        ],
    )
    x_by_view = build_feature_matrix_by_view(x_all, feature_views)
    param_profiles = build_param_profile_candidates(target_weight_search_settings)
    evaluation_contexts = build_evaluation_contexts(
        feature_views,
        x_by_view,
        param_profiles,
    )

    proxy_fold_count = min(int(SEARCH_CV_FOLDS), int(CV_FOLDS))
    proxy_folds = make_walk_forward_folds(
        n_rows=len(x_all),
        n_folds=proxy_fold_count,
        test_to_train_ratio=float(TEST_TO_TRAIN_RATIO),
    )
    final_folds = make_walk_forward_folds(
        n_rows=len(x_all),
        n_folds=int(CV_FOLDS),
        test_to_train_ratio=float(TEST_TO_TRAIN_RATIO),
    )
    proxy_fold_weight_by_id = build_fold_recency_weights(proxy_folds)
    final_fold_weight_by_id = build_fold_recency_weights(final_folds)

    run_info = resolve_run_study_name(
        STUDY_NAME,
        default_prefix=DEFAULT_STUDY_NAME_PREFIX,
    )
    best_result_path = make_timestamped_artifact_path(
        OUTPUT_DIR,
        stem=BEST_RESULT_STEM,
        suffix=".json",
        timestamp=run_info["run_timestamp"],
    )
    search_results_csv_path = make_timestamped_artifact_path(
        OUTPUT_DIR,
        stem=SEARCH_RESULTS_CSV_STEM,
        suffix=".csv",
        timestamp=run_info["run_timestamp"],
    )
    search_context_results_csv_path = make_timestamped_artifact_path(
        OUTPUT_DIR,
        stem=SEARCH_CONTEXT_RESULTS_CSV_STEM,
        suffix=".csv",
        timestamp=run_info["run_timestamp"],
    )
    search_fold_metrics_csv_path = make_timestamped_artifact_path(
        OUTPUT_DIR,
        stem=SEARCH_FOLD_METRICS_CSV_STEM,
        suffix=".csv",
        timestamp=run_info["run_timestamp"],
    )
    final_results_csv_path = make_timestamped_artifact_path(
        OUTPUT_DIR,
        stem=FINAL_RESULTS_CSV_STEM,
        suffix=".csv",
        timestamp=run_info["run_timestamp"],
    )
    final_context_results_csv_path = make_timestamped_artifact_path(
        OUTPUT_DIR,
        stem=FINAL_CONTEXT_RESULTS_CSV_STEM,
        suffix=".csv",
        timestamp=run_info["run_timestamp"],
    )
    final_fold_metrics_csv_path = make_timestamped_artifact_path(
        OUTPUT_DIR,
        stem=FINAL_FOLD_METRICS_CSV_STEM,
        suffix=".csv",
        timestamp=run_info["run_timestamp"],
    )
    best_oof_path = make_timestamped_artifact_path(
        OUTPUT_DIR,
        stem=BEST_OOF_STEM,
        suffix=".parquet",
        timestamp=run_info["run_timestamp"],
    )

    print(
        f"start target_weight_search | rows={len(df)} features_all={x_all.shape[1]} "
        f"decision_rows={decision_summary['rows']} decision_share={decision_summary['share']:.4f} "
        f"feature_views={len(feature_views)} param_profiles={len(param_profiles)} "
        f"contexts={len(evaluation_contexts)} "
        f"proxy_cv_folds={len(proxy_folds)} final_cv_folds={len(final_folds)} "
        f"objective={resolve_decision_objective_name()} float_precision={modeling_float_dtype_name}"
    )
    print(
        "target-weight data source | "
        f"path={path_to_portable_str(data_path)} "
        f"param_profiles={list(target_weight_search_settings['param_profiles'])} "
        f"random_feature_subsets={int(target_weight_search_settings['random_feature_subsets'])} "
        f"random_feature_subset_size={target_weight_search_settings['random_feature_subset_size']} "
        f"lookback_days={target_weight_search_settings['lookback_days']}"
    )
    if time_window is not None:
        print(
            "target-weight time window | "
            f"opened_start_utc={time_window['opened_start_utc']} "
            f"opened_end_utc={time_window['opened_end_utc']} "
            f"rows_total={time_window['rows_total']} "
            f"rows_in_window={time_window['rows_in_window']}"
        )
    if active_feature_subset is not None:
        print(
            "active feature subset | "
            f"path={path_to_portable_str(active_feature_subset['path'])} "
            f"features={int(active_feature_subset['count'])}"
        )
    for feature_view in feature_views:
        print(
            f"feature view | id={feature_view['id']} label={feature_view['label']} "
            f"source={feature_view['source_kind']} features={feature_view['feature_count']} "
            f"path={feature_view['path']}"
        )
    for param_profile in param_profiles:
        print(
            f"param profile | id={param_profile['id']} name={param_profile['name']} "
            f"param_count={len(param_profile['params'])}"
        )
    print(
        "search config | "
        f"decision_weight_range=({float(DECISION_WEIGHT_LOW):.6f}, {float(DECISION_WEIGHT_HIGH):.6f}) "
        f"proxy_n_estimators={int(SEARCH_N_ESTIMATORS)} final_n_estimators={int(N_ESTIMATORS)} "
        f"context_std_penalty={float(target_weight_search_settings['context_std_penalty']):.6f} "
        f"proxy_eval=decision_rows_only final_eval=decision_rows_only final_predict=all_rows"
    )

    proxy_search_result = run_proxy_weight_search_across_contexts(
        evaluation_contexts=evaluation_contexts,
        y=y,
        decision_mask=decision_mask,
        folds=proxy_folds,
        fold_weight_by_id=proxy_fold_weight_by_id,
        float_dtype=modeling_float_dtype,
        context_std_penalty=target_weight_search_settings["context_std_penalty"],
    )
    search_results_df = _top_aggregate_weight_results(
        proxy_search_result["search_rows"],
        limit=len(proxy_search_result["search_rows"]),
    )
    search_context_results_df = pd.DataFrame(proxy_search_result["search_context_rows"])
    search_context_results_df = search_context_results_df.sort_values(
        by=[
            "decision_weight",
            "objective_value",
            "decision_rows_oof_balanced_accuracy",
        ],
        ascending=[True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    search_fold_metrics_df = proxy_search_result["search_fold_metrics"]

    final_recheck_result = run_final_weight_recheck_across_contexts(
        evaluation_contexts=evaluation_contexts,
        shortlist_rows=proxy_search_result["weighted_shortlist"],
        y=y,
        decision_mask=decision_mask,
        folds=final_folds,
        fold_weight_by_id=final_fold_weight_by_id,
        float_dtype=modeling_float_dtype,
        context_std_penalty=target_weight_search_settings["context_std_penalty"],
    )
    baseline_recheck_result = evaluate_baseline_across_contexts(
        stage="final_recheck",
        evaluation_contexts=evaluation_contexts,
        y=y,
        decision_mask=decision_mask,
        folds=final_folds,
        fold_weight_by_id=final_fold_weight_by_id,
        float_dtype=modeling_float_dtype,
        n_estimators=int(N_ESTIMATORS),
        early_stopping_rounds=int(EARLY_STOPPING_ROUNDS),
        prediction_scope=PREDICTION_SCOPE_ALL_ROWS,
        context_std_penalty=target_weight_search_settings["context_std_penalty"],
    )
    baseline_aggregate_row = {
        **baseline_recheck_result["aggregate_row"],
        "proxy_objective_value": np.nan,
        "proxy_rank": np.nan,
    }
    print(
        "final baseline | "
        f"strategy={baseline_aggregate_row['strategy_name']} "
        f"objective={baseline_aggregate_row['objective_value']:.6f} "
        f"context_mean={baseline_aggregate_row['context_objective_mean']:.6f} "
        f"context_std={baseline_aggregate_row['context_objective_std']:.6f} "
        f"decision_bal_acc_mean={baseline_aggregate_row['decision_rows_bal_acc_mean']:.6f}"
    )
    final_results_df = pd.concat(
        [
            pd.DataFrame([baseline_aggregate_row]),
            final_recheck_result["final_results_df"],
        ],
        ignore_index=True,
    )
    final_context_results_df = pd.concat(
        [
            pd.DataFrame(baseline_recheck_result["context_rows"]),
            final_recheck_result["final_context_results_df"],
        ],
        ignore_index=True,
    )
    final_fold_metrics_df = pd.concat(
        [
            baseline_recheck_result["fold_metrics"],
            final_recheck_result["final_fold_metrics_df"],
        ],
        ignore_index=True,
    )
    final_context_results_df = sort_context_results_df(
        enrich_context_results_with_baseline_deltas(final_context_results_df)
    )
    final_results_df = sort_aggregate_strategy_results(
        enrich_aggregate_results_with_baseline_deltas(
            final_results_df,
            context_results_df=final_context_results_df,
        )
    )
    if final_results_df.empty:
        raise RuntimeError("No final target-weight candidates were evaluated.")
    best_row = final_results_df.iloc[0]
    recommendation = build_baseline_recommendation_summary(final_results_df)

    payload = build_best_result_payload(
        run_info=run_info,
        data_path=data_path,
        class_distribution=class_distribution,
        decision_mask=decision_mask,
        time_window=time_window,
        excluded_features=excluded_features,
        active_feature_subset=active_feature_subset,
        feature_views=feature_views,
        param_profiles=param_profiles,
        evaluation_contexts=evaluation_contexts,
        target_weight_search_settings=target_weight_search_settings,
        final_results_df=final_results_df,
        final_context_results_df=final_context_results_df,
        best_row=best_row,
    )
    payload["artifacts"] = {
        "best_result_json": path_to_portable_str(best_result_path),
        "proxy_candidates_csv": path_to_portable_str(search_results_csv_path),
        "proxy_context_results_csv": path_to_portable_str(
            search_context_results_csv_path
        ),
        "proxy_fold_metrics_csv": path_to_portable_str(search_fold_metrics_csv_path),
        "final_candidates_csv": path_to_portable_str(final_results_csv_path),
        "final_context_results_csv": path_to_portable_str(
            final_context_results_csv_path
        ),
        "final_fold_metrics_csv": path_to_portable_str(final_fold_metrics_csv_path),
        "best_oof_parquet": (
            path_to_portable_str(best_oof_path) if SAVE_BEST_OOF else None
        ),
    }
    payload = json_safe_value(payload)

    best_result_path.parent.mkdir(parents=True, exist_ok=True)
    best_result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    search_results_df.to_csv(search_results_csv_path, index=False)
    search_context_results_df.to_csv(search_context_results_csv_path, index=False)
    search_fold_metrics_df.to_csv(search_fold_metrics_csv_path, index=False)
    final_results_df.to_csv(final_results_csv_path, index=False)
    final_context_results_df.to_csv(final_context_results_csv_path, index=False)
    final_fold_metrics_df.to_csv(final_fold_metrics_csv_path, index=False)

    if SAVE_BEST_OOF:
        if str(best_row["strategy_name"]) == STRATEGY_DECISION_ONLY_BASELINE:
            best_strategy_result = baseline_recheck_result
        else:
            best_weight = float(best_row["decision_weight"])
            best_strategy_result = final_recheck_result["evaluated_by_weight"][
                best_weight
            ]
        primary_context_result = next(
            (
                item
                for item in best_strategy_result["context_results"]
                if bool(item["context"]["is_primary"])
            ),
            best_strategy_result["context_results"][0],
        )
        save_best_oof_predictions(
            output_path=best_oof_path,
            df=df,
            decision_mask=decision_mask,
            final_result={"result": primary_context_result["result"]},
        )

    print(
        "best strategy | "
        f"strategy={best_row['strategy_name']} "
        f"decision_weight={best_row['decision_weight']} "
        f"objective={best_row['objective_value']:.6f} "
        f"context_mean={best_row['context_objective_mean']:.6f} "
        f"context_std={best_row['context_objective_std']:.6f} "
        f"decision_rows_bal_acc_mean={best_row['decision_rows_bal_acc_mean']:.6f}"
    )
    if recommendation is not None:
        print(
            "recommendation | "
            f"strategy={recommendation['recommended_strategy_name']} "
            f"decision_weight={recommendation['recommended_decision_weight']} "
            f"reason={recommendation['recommendation_reason']}"
        )
    print(
        f"artifacts | json={path_to_portable_str(best_result_path)} "
        f"proxy_csv={path_to_portable_str(search_results_csv_path)} "
        f"final_csv={path_to_portable_str(final_results_csv_path)}"
    )


if __name__ == "__main__":
    main()
