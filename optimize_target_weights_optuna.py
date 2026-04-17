import json
from bisect import bisect_left
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common_config_utils import path_to_portable_str
from modeling_dataset_utils import (
    load_excluded_feature_names_from_settings,
    load_feature_subset,
    load_feature_subset_from_settings,
    load_modeling_dataset_settings,
    resolve_modeling_dataset_output_paths,
    resolve_modeling_float_dtype,
    resolve_modeling_float_dtype_name,
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
    TARGET_COL,
    WF_TEST_TO_TRAIN_RATIO as DEFAULT_WF_TEST_TO_TRAIN_RATIO,
    build_lgbm_model,
    classification_metrics,
    clean_and_impute_fold,
    load_walk_forward_training_frame,
    make_walk_forward_folds,
)

DECISION_ROW_OBJECTIVE_METRIC = "balanced_accuracy"
DECISION_ROW_STD_PENALTY = 1.0
DEFAULT_STUDY_NAME_PREFIX = "lgbm_target_weight_search_decision_rows_balanced_accuracy"
DEFAULT_OUTPUT_DIR = Path("data/optuna/target_weights")
BEST_RESULT_STEM = "lgbm_target_weight_search_best_decision_rows_balanced_accuracy"
SEARCH_RESULTS_CSV_STEM = "lgbm_target_weight_search_proxy_candidates"
SEARCH_FOLD_METRICS_CSV_STEM = "lgbm_target_weight_search_proxy_fold_metrics"
FINAL_RESULTS_CSV_STEM = "lgbm_target_weight_search_final_candidates"
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
PARAMS_SOURCE = "optuna"
CV_FOLDS = DEFAULT_CV_FOLDS
TEST_TO_TRAIN_RATIO = DEFAULT_WF_TEST_TO_TRAIN_RATIO
DEVICE_TYPE = "gpu"
LGBM_N_JOBS = 14
OBJECTIVE_STD_PENALTY = DECISION_ROW_STD_PENALTY
SAVE_BEST_OOF = False

FEATURE_SELECTOR_ARTIFACT_ROOT = Path("data/analysis/feature_selector")
FEATURE_SUBSET_RECENT_LIMIT = 3
MAX_FEATURE_SUBSET_CANDIDATES = 3

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
    if PARAMS_SOURCE not in {"optuna", "default"}:
        raise ValueError("PARAMS_SOURCE must be one of: 'optuna', 'default'.")
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


def build_model_param_overrides():
    if PARAMS_SOURCE == "optuna":
        params = dict(LGBM_OPTUNA_BEST_PARAMS)
    else:
        params = dict(LGBM_DEFAULT_PARAMS)
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


def resolve_feature_subset_candidates(dataset_settings, excluded_features):
    active_subset = load_feature_subset_from_settings(dataset_settings)
    active_subset_path = None if active_subset is None else Path(active_subset["path"])
    candidates = []
    seen_signatures = set()

    def register_candidate(subset_info, *, is_active):
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
    for subset_path in discover_recent_feature_subset_paths(
        root_dir=FEATURE_SELECTOR_ARTIFACT_ROOT,
        limit=int(FEATURE_SUBSET_RECENT_LIMIT),
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
        if len(candidates) >= int(MAX_FEATURE_SUBSET_CANDIDATES):
            break

    if not candidates:
        register_candidate(None, is_active=False)

    return candidates[: int(MAX_FEATURE_SUBSET_CANDIDATES)]


def build_union_feature_subset(feature_subset_candidates):
    subset_infos = [
        candidate["subset_info"]
        for candidate in feature_subset_candidates
        if candidate["subset_info"] is not None
    ]
    if not subset_infos:
        return None

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
    excluded_features,
    feature_subset_candidates,
    final_results_df,
    subset_summary_df,
    best_row,
):
    decision_mask_np = np.asarray(decision_mask, dtype=bool)
    best_subset_candidate = next(
        candidate
        for candidate in feature_subset_candidates
        if candidate["id"] == best_row["feature_subset_id"]
    )
    return {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "study_name": run_info["study_name"],
        "study_name_source": run_info["study_name_source"],
        "search_method": "deterministic_proxy_then_final_recheck",
        "search_target": "decision_rows_weight_vs_decision_only_baseline",
        "data_path": path_to_portable_str(data_path),
        "target_col": TARGET_COL,
        "sample_weight_col": TARGET_WEIGHT_COL,
        "decision_row_definition": {
            "expression": "Opened.minute % 5 == 4",
            "rows": int(decision_mask_np.sum()),
            "total_rows": int(decision_mask_np.size),
            "share": float(decision_mask_np.mean()),
        },
        "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
        "feature_subset_candidates": [
            {
                "id": candidate["id"],
                "label": candidate["label"],
                "path": candidate["path"],
                "feature_count": int(candidate["feature_count"]),
                "is_active": bool(candidate["is_active"]),
                "summary": candidate["summary"],
            }
            for candidate in feature_subset_candidates
        ],
        "proxy_search": {
            "cv_folds": int(SEARCH_CV_FOLDS),
            "n_estimators": int(SEARCH_N_ESTIMATORS),
            "early_stopping_rounds": int(SEARCH_EARLY_STOPPING_ROUNDS),
            "initial_weight_grid": [float(value) for value in build_initial_weight_candidates()],
            "refinement_rounds": int(SEARCH_REFINEMENT_ROUNDS),
            "top_parent_weights": int(SEARCH_TOP_PARENT_WEIGHTS),
            "top_weight_candidates_per_subset": int(TOP_WEIGHT_CANDIDATES_PER_SUBSET),
        },
        "final_recheck": {
            "cv_folds": int(CV_FOLDS),
            "n_estimators": int(N_ESTIMATORS),
            "early_stopping_rounds": int(EARLY_STOPPING_ROUNDS),
            "params_source": str(PARAMS_SOURCE),
            "device_type": str(DEVICE_TYPE),
            "final_top_weighted_candidates": int(FINAL_TOP_WEIGHTED_CANDIDATES),
        },
        "objective": {
            "metric": DECISION_ROW_OBJECTIVE_METRIC,
            "scope": "decision_rows_only",
            "aggregation": decision_objective_aggregation_description(),
            "std_penalty": float(OBJECTIVE_STD_PENALTY),
            "fold_recency_weighting": {
                "enabled": bool(ENABLE_FOLD_RECENCY_WEIGHTING),
                "active": bool(is_nontrivial_fold_recency_weighting_enabled()),
                "mode": str(FOLD_RECENCY_WEIGHTING_MODE),
                "min_weight": float(FOLD_RECENCY_WEIGHT_MIN),
                "max_weight": float(FOLD_RECENCY_WEIGHT_MAX),
            },
        },
        "feature_selection": summarize_feature_subset(
            best_subset_candidate["subset_info"],
            excluded_features=excluded_features,
        ),
        "subset_recommendations": subset_summary_df.to_dict(orient="records"),
        "final_leaderboard": final_results_df.sort_values(
            by=[
                "objective_value",
                "decision_metric_weighted_mean",
                "decision_rows_oof_balanced_accuracy",
            ],
            ascending=[False, False, False],
            kind="stable",
        ).to_dict(orient="records"),
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
    modeling_float_dtype = resolve_modeling_float_dtype(dataset_settings)
    modeling_float_dtype_name = resolve_modeling_float_dtype_name(dataset_settings)
    data_path = resolve_modeling_dataset_output_paths(dataset_settings)["parquet"]
    excluded_features = load_excluded_feature_names_from_settings(dataset_settings)
    feature_subset_candidates = resolve_feature_subset_candidates(
        dataset_settings,
        excluded_features,
    )
    if not feature_subset_candidates:
        raise ValueError("No feature subset candidates were resolved for target-weight search.")

    union_feature_subset = build_union_feature_subset(feature_subset_candidates)
    training_data = load_walk_forward_training_frame(
        data_path=data_path,
        feature_subset=union_feature_subset,
        excluded_features=excluded_features,
        float_dtype=modeling_float_dtype,
    )
    df = training_data["df"].reset_index(drop=True)
    x_union = training_data["x"].reset_index(drop=True)
    y = training_data["y"].reset_index(drop=True)
    class_distribution = training_data["class_distribution"]
    decision_mask = compute_decision_mask_from_opened(df["Opened"])
    decision_summary = summarize_boolean_mask(decision_mask)
    x_by_subset = build_feature_matrix_by_subset(x_union, feature_subset_candidates)

    proxy_fold_count = min(int(SEARCH_CV_FOLDS), int(CV_FOLDS))
    proxy_folds = make_walk_forward_folds(
        n_rows=len(x_union),
        n_folds=proxy_fold_count,
        test_to_train_ratio=float(TEST_TO_TRAIN_RATIO),
    )
    final_folds = make_walk_forward_folds(
        n_rows=len(x_union),
        n_folds=int(CV_FOLDS),
        test_to_train_ratio=float(TEST_TO_TRAIN_RATIO),
    )
    proxy_fold_weight_by_id = build_fold_recency_weights(proxy_folds)
    final_fold_weight_by_id = build_fold_recency_weights(final_folds)
    param_overrides = build_model_param_overrides()

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
        f"start target_weight_search | rows={len(df)} features_union={x_union.shape[1]} "
        f"decision_rows={decision_summary['rows']} decision_share={decision_summary['share']:.4f} "
        f"feature_subset_candidates={len(feature_subset_candidates)} "
        f"proxy_cv_folds={len(proxy_folds)} final_cv_folds={len(final_folds)} "
        f"objective={resolve_decision_objective_name()} float_precision={modeling_float_dtype_name}"
    )
    for candidate in feature_subset_candidates:
        print(
            f"subset candidate | id={candidate['id']} label={candidate['label']} "
            f"features={candidate['feature_count']} path={candidate['path']}"
        )
    print(
        "search config | "
        f"decision_weight_range=({float(DECISION_WEIGHT_LOW):.6f}, {float(DECISION_WEIGHT_HIGH):.6f}) "
        f"proxy_n_estimators={int(SEARCH_N_ESTIMATORS)} final_n_estimators={int(N_ESTIMATORS)} "
        f"proxy_eval=decision_rows_only final_eval=decision_rows_only final_predict=all_rows"
    )

    proxy_subset_results = []
    search_result_rows = []
    search_fold_metrics_frames = []
    for feature_subset_candidate in feature_subset_candidates:
        subset_proxy_result = run_proxy_weight_search_for_subset(
            feature_subset_candidate=feature_subset_candidate,
            x=x_by_subset[feature_subset_candidate["id"]],
            y=y,
            decision_mask=decision_mask,
            folds=proxy_folds,
            fold_weight_by_id=proxy_fold_weight_by_id,
            param_overrides=param_overrides,
            float_dtype=modeling_float_dtype,
        )
        proxy_subset_results.append(subset_proxy_result)
        search_result_rows.extend(subset_proxy_result["search_rows"])
        search_fold_metrics_frames.append(subset_proxy_result["search_fold_metrics"])

    search_results_df = pd.DataFrame(search_result_rows)
    search_fold_metrics_df = pd.concat(search_fold_metrics_frames, ignore_index=True)

    final_candidate_specs = build_final_candidate_specs(proxy_subset_results)
    final_results = []
    final_fold_metrics_frames = []
    for candidate_spec in final_candidate_specs:
        feature_subset_candidate = candidate_spec["feature_subset_candidate"]
        evaluated = evaluate_final_candidate(
            candidate_spec=candidate_spec,
            x=x_by_subset[feature_subset_candidate["id"]],
            y=y,
            decision_mask=decision_mask,
            folds=final_folds,
            fold_weight_by_id=final_fold_weight_by_id,
            param_overrides=param_overrides,
            float_dtype=modeling_float_dtype,
        )
        final_results.append(
            {
                **evaluated["row"],
                "proxy_objective_value": candidate_spec.get("proxy_objective_value"),
            }
        )
        final_fold_metrics_frames.append(evaluated["fold_metrics"])
        print(
            f"final subset={feature_subset_candidate['label']} "
            f"strategy={candidate_spec['strategy_name']} "
            f"decision_weight={candidate_spec.get('decision_weight')} "
            f"objective={evaluated['row']['objective_value']:.6f} "
            f"decision_bal_acc={evaluated['row']['decision_rows_oof_balanced_accuracy']:.6f}"
        )

    final_results_df = pd.DataFrame(final_results)
    final_results_df = enrich_final_results_with_baseline_deltas(final_results_df)
    final_results_df = final_results_df.sort_values(
        by=[
            "objective_value",
            "decision_metric_weighted_mean",
            "decision_rows_oof_balanced_accuracy",
        ],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    final_fold_metrics_df = pd.concat(final_fold_metrics_frames, ignore_index=True)
    subset_summary_df = build_subset_summary_rows(final_results_df)
    if subset_summary_df.empty:
        raise RuntimeError("No subset summary rows were produced.")

    recommended_results_df = final_results_df.merge(
        subset_summary_df.loc[:, ["feature_subset_id", "recommended_strategy_name"]],
        left_on=["feature_subset_id", "strategy_name"],
        right_on=["feature_subset_id", "recommended_strategy_name"],
        how="inner",
    )
    if recommended_results_df.empty:
        raise RuntimeError("No final recommended candidates were identified.")
    best_row = (
        recommended_results_df.sort_values(
            by=[
                "objective_value",
                "decision_metric_weighted_mean",
                "decision_rows_oof_balanced_accuracy",
            ],
            ascending=[False, False, False],
            kind="stable",
        )
        .iloc[0]
        .drop(labels=["recommended_strategy_name"])
    )

    payload = build_best_result_payload(
        run_info=run_info,
        data_path=data_path,
        class_distribution=class_distribution,
        decision_mask=decision_mask,
        excluded_features=excluded_features,
        feature_subset_candidates=feature_subset_candidates,
        final_results_df=final_results_df,
        subset_summary_df=subset_summary_df,
        best_row=best_row,
    )
    payload["artifacts"] = {
        "best_result_json": path_to_portable_str(best_result_path),
        "proxy_candidates_csv": path_to_portable_str(search_results_csv_path),
        "proxy_fold_metrics_csv": path_to_portable_str(search_fold_metrics_csv_path),
        "final_candidates_csv": path_to_portable_str(final_results_csv_path),
        "final_fold_metrics_csv": path_to_portable_str(final_fold_metrics_csv_path),
        "best_oof_parquet": (
            path_to_portable_str(best_oof_path) if SAVE_BEST_OOF else None
        ),
    }
    payload = json_safe_value(payload)

    best_result_path.parent.mkdir(parents=True, exist_ok=True)
    best_result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    search_results_df.to_csv(search_results_csv_path, index=False)
    search_fold_metrics_df.to_csv(search_fold_metrics_csv_path, index=False)
    final_results_df.to_csv(final_results_csv_path, index=False)
    final_fold_metrics_df.to_csv(final_fold_metrics_csv_path, index=False)

    if SAVE_BEST_OOF:
        best_feature_subset_id = str(best_row["feature_subset_id"])
        best_strategy_name = str(best_row["strategy_name"])
        if best_strategy_name == STRATEGY_DECISION_ONLY_BASELINE:
            best_candidate_spec = {
                "feature_subset_candidate": next(
                    candidate
                    for candidate in feature_subset_candidates
                    if candidate["id"] == best_feature_subset_id
                ),
                "strategy_name": STRATEGY_DECISION_ONLY_BASELINE,
                "decision_weight": None,
            }
        else:
            best_candidate_spec = {
                "feature_subset_candidate": next(
                    candidate
                    for candidate in feature_subset_candidates
                    if candidate["id"] == best_feature_subset_id
                ),
                "strategy_name": STRATEGY_ALL_ROWS_WEIGHTED,
                "decision_weight": float(best_row["decision_weight"]),
            }
        best_final_result = evaluate_final_candidate(
            candidate_spec=best_candidate_spec,
            x=x_by_subset[best_feature_subset_id],
            y=y,
            decision_mask=decision_mask,
            folds=final_folds,
            fold_weight_by_id=final_fold_weight_by_id,
            param_overrides=param_overrides,
            float_dtype=modeling_float_dtype,
        )
        save_best_oof_predictions(
            output_path=best_oof_path,
            df=df,
            decision_mask=decision_mask,
            final_result=best_final_result,
        )

    print(
        f"best strategy | subset={best_row['feature_subset_label']} "
        f"strategy={best_row['strategy_name']} "
        f"decision_weight={best_row['decision_weight']} "
        f"objective={best_row['objective_value']:.6f} "
        f"decision_rows_bal_acc={best_row['decision_rows_oof_balanced_accuracy']:.6f}"
    )
    print(
        f"artifacts | json={path_to_portable_str(best_result_path)} "
        f"proxy_csv={path_to_portable_str(search_results_csv_path)} "
        f"final_csv={path_to_portable_str(final_results_csv_path)}"
    )


if __name__ == "__main__":
    main()
