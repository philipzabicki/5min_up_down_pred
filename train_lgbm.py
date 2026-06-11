import json
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from features.candle_features import (
    CANDLE_PATTERN_COLS,
    RAW_OHLCV_COLS,
)
from features.session_open_features import add_session_open_features
from features.volume_profile_fixed_range import (
    is_volume_profile_feature,
    normalize_config as normalize_volume_profile_config,
    validate_volume_profile_dataset_metadata,
    validate_volume_profile_feature_columns,
)
from utils.config import path_to_portable_str
from utils.data import (
    TARGET_WEIGHT_COL,
    compute_target_weights_from_opened,
    summarize_target_weights,
)
from utils.data import (
    load_modeling_dataset_artifact_metadata,
    load_excluded_feature_names_from_settings,
    load_feature_subset_from_settings,
    load_modeling_dataset_settings,
    resolve_modeling_float_dtype,
    resolve_modeling_float_dtype_name,
    resolve_modeling_dataset_output_paths,
    resolve_oof_prediction_output_paths,
    split_feature_subset,
    summarize_feature_subset,
    validate_parquet_magic_bytes,
)
from utils.metrics import (
    make_sklearn_binary_logloss_eval,
    weighted_binary_logloss,
    weighted_brier_score,
)
from utils.project_config import active_asset_path

TARGET_COL = "target_5m_candle_up"
OOF_PRED_COL = "oof_pred_proba_up"
OOF_EXPORT_BASE_COLS = [
    "Opened",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    TARGET_COL,
    TARGET_WEIGHT_COL,
]
OOF_PREVIEW_ROWS = 1000
CV_FOLDS = 10
N_ESTIMATORS = 3000
EARLY_STOPPING_ROUNDS = 25
SEED = 37
N_JOBS = 16
MODELS_DIR = active_asset_path("data/models/{asset}")
LGBM_DEVICE_TYPE = "gpu"
LGBM_VERBOSITY = -1
PREDICTION_THRESHOLD = 0.5
LGBM_MAX_BIN = 63
PRIMARY_REPORTING_METRIC = "binary_logloss"
EARLY_STOPPING_METRIC = PRIMARY_REPORTING_METRIC
EARLY_STOPPING_EVAL_METRIC = make_sklearn_binary_logloss_eval(EARLY_STOPPING_METRIC)


def resolve_walk_forward_test_to_train_ratio():
    settings = load_modeling_dataset_settings()
    train_lgbm_settings = settings.get("train_lgbm") or {}
    return float(train_lgbm_settings["walk_forward_test_to_train_ratio"])


WF_TEST_TO_TRAIN_RATIO = resolve_walk_forward_test_to_train_ratio()

# Wklej tutaj najlepsze parametry z optimize_generic_lgbm_optuna.py.
# Zostaw pusty dict, aby używać domyślnych parametrów LightGBM.s
LGBM_OPTUNA_BEST_PARAMS = {
      "learning_rate": 0.028370028338368332,
      "num_leaves": 201,
      "min_data_in_leaf": 2527,
      "max_depth": 246,
      "feature_fraction": 0.7371325954810048,
      "bagging_fraction": 0.9308115159624847,
      "bagging_freq": 25,
      "lambda_l2": 39.51068447389412,
      "lambda_l1": 5.269826622414657,
      "min_sum_hessian_in_leaf": 1.5455745495230206,
      "min_gain_to_split": 0.09322721688630903,
      "feature_fraction_bynode": 0.38804968275175306,
      "path_smooth": 39.06532785785736,
      "extra_trees": False,
      "monotone_constraints_method": "basic",
      "monotone_penalty": 0.0
    }
LGBM_DEFAULT_PARAMS = {
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "max_depth": -1,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "lambda_l2": 0.0,
    "lambda_l1": 0.0,
    "min_sum_hessian_in_leaf": 0.001,
    "min_gain_to_split": 0.0,
    "feature_fraction_bynode": 1.0,
    "path_smooth": 0.0,
    "extra_trees": False,
}

_ACTIVE_MONOTONE_CONSTRAINTS_BY_FEATURE = None


def load_active_lgbm_monotone_constraints():
    global _ACTIVE_MONOTONE_CONSTRAINTS_BY_FEATURE
    if _ACTIVE_MONOTONE_CONSTRAINTS_BY_FEATURE is None:
        settings = load_modeling_dataset_settings()
        train_lgbm_settings = settings.get("train_lgbm") or {}
        _ACTIVE_MONOTONE_CONSTRAINTS_BY_FEATURE = dict(
            train_lgbm_settings.get("monotone_constraints") or {}
        )
    return dict(_ACTIVE_MONOTONE_CONSTRAINTS_BY_FEATURE)


def build_lgbm_monotone_constraint_vector(
        feature_names,
        constraints_by_feature=None,
):
    if constraints_by_feature is None:
        constraints_by_feature = load_active_lgbm_monotone_constraints()
    constraints_by_feature = dict(constraints_by_feature or {})
    if not constraints_by_feature:
        return None
    return [
        int(constraints_by_feature.get(str(feature_name), 0))
        for feature_name in feature_names
    ]


def make_lgbm_monotone_constraint_params(
        feature_names,
        constraints_by_feature=None,
):
    constraint_vector = build_lgbm_monotone_constraint_vector(
        feature_names,
        constraints_by_feature=constraints_by_feature,
    )
    if constraint_vector is None:
        return {}
    return {"monotone_constraints": constraint_vector}


def summarize_lgbm_monotone_constraints(
        feature_names,
        constraints_by_feature=None,
):
    if constraints_by_feature is None:
        constraints_by_feature = load_active_lgbm_monotone_constraints()
    constraints_by_feature = dict(constraints_by_feature or {})
    feature_name_list = [str(feature_name) for feature_name in feature_names]
    feature_name_set = set(feature_name_list)
    applied_constraints = {
        feature_name: int(constraints_by_feature[feature_name])
        for feature_name in feature_name_list
        if int(constraints_by_feature.get(feature_name, 0)) != 0
    }
    missing_configured_features = [
        feature_name
        for feature_name in constraints_by_feature
        if feature_name not in feature_name_set
    ]
    return {
        "configured_feature_count": int(len(constraints_by_feature)),
        "applied_feature_count": int(len(applied_constraints)),
        "missing_configured_feature_count": int(len(missing_configured_features)),
        "configured_constraints": constraints_by_feature,
        "applied_constraints": applied_constraints,
        "missing_configured_features": missing_configured_features,
    }


def format_lgbm_monotone_constraint_summary(summary):
    return (
        f"configured={summary['configured_feature_count']} "
        f"applied={summary['applied_feature_count']} "
        f"missing_configured={summary['missing_configured_feature_count']}"
    )


def build_lgbm_model(
        n_estimators,
        param_overrides=None,
        feature_names=None,
        monotone_constraints_by_feature=None,
):
    params = {
        "objective": "binary",
        "metric": "None",
        "n_estimators": n_estimators,
        "random_state": SEED,
        "n_jobs": N_JOBS,
        "device_type": LGBM_DEVICE_TYPE,
        "verbosity": LGBM_VERBOSITY,
        "max_bin": LGBM_MAX_BIN,
    }
    if param_overrides:
        params.update(param_overrides)
    if feature_names is not None:
        params.update(
            make_lgbm_monotone_constraint_params(
                feature_names,
                constraints_by_feature=monotone_constraints_by_feature,
            )
        )
    return lgb.LGBMClassifier(**params)


def make_walk_forward_folds(
        n_rows,
        n_folds,
        test_to_train_ratio,
):
    ratio_inv = 1.0 / test_to_train_ratio
    test_len = int(np.floor(n_rows / (n_folds + ratio_inv)))
    train_len = int(np.floor(test_len / test_to_train_ratio))

    folds = []
    for fold_id in range(n_folds):
        train_start = fold_id * test_len
        train_end = train_start + train_len
        test_start = train_end
        test_end = test_start + test_len
        if test_end > n_rows:
            break
        folds.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

    return folds


def classification_metrics(
        y_true,
        y_pred_proba,
        sample_weight=None,
        threshold=PREDICTION_THRESHOLD,
):
    y_true_i = np.asarray(y_true, dtype=np.int8)
    y_pred_proba_f = np.asarray(y_pred_proba, dtype=np.float64)
    y_pred_i = (y_pred_proba_f >= threshold).astype(np.int8)
    if sample_weight is None:
        w = np.ones(shape=len(y_true_i), dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)
        if w.shape[0] != len(y_true_i):
            raise ValueError(
                "Sample weights length mismatch in classification_metrics: "
                f"{w.shape[0]} != {len(y_true_i)}"
            )

    tp = float(np.sum(w[(y_true_i == 1) & (y_pred_i == 1)]))
    tn = float(np.sum(w[(y_true_i == 0) & (y_pred_i == 0)]))
    fp = float(np.sum(w[(y_true_i == 0) & (y_pred_i == 1)]))
    fn = float(np.sum(w[(y_true_i == 1) & (y_pred_i == 0)]))

    total = float(np.sum(w))
    accuracy = float((tp + tn) / total) if total > 0 else float("nan")
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float((2 * precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    tpr = recall
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_accuracy = float((tpr + tnr) / 2.0)

    brier_score = float(
        weighted_brier_score(
            y_true=y_true_i,
            y_pred_proba=y_pred_proba_f,
            sample_weight=w,
        )
    )
    binary_logloss = float(
        weighted_binary_logloss(
            y_true=y_true_i,
            y_pred_proba=y_pred_proba_f,
            sample_weight=w,
        )
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": brier_score,
        "binary_logloss": binary_logloss,
    }


def resolve_sample_weight_series(df, float_dtype=np.float64):
    if TARGET_WEIGHT_COL in df.columns:
        sample_weight = pd.to_numeric(df[TARGET_WEIGHT_COL], errors="raise")
        source = "dataset_column"
    else:
        if "Opened" not in df.columns:
            raise ValueError(
                f"Cannot derive sample weights because 'Opened' is missing and '{TARGET_WEIGHT_COL}' is absent."
            )
        sample_weight = pd.Series(
            compute_target_weights_from_opened(df["Opened"]),
            index=df.index,
            name=TARGET_WEIGHT_COL,
        )
        source = "derived_from_opened"

    sample_weight = sample_weight.astype(float_dtype)
    sample_weight_np = sample_weight.to_numpy(dtype=float_dtype, copy=False)
    if sample_weight_np.shape[0] != len(df):
        raise ValueError("Sample weights length mismatch.")
    if not np.isfinite(sample_weight_np).all():
        raise ValueError("Sample weights contain non-finite values.")
    if np.any(sample_weight_np <= 0.0):
        raise ValueError("Sample weights must be strictly positive.")

    return (
        pd.Series(sample_weight_np, index=df.index, name=TARGET_WEIGHT_COL),
        source,
        summarize_target_weights(sample_weight_np),
    )


def clean_and_impute_fold(
        x_train_raw,
        x_test_raw,
        float_dtype=np.float64,
):
    all_nan_train_features = x_train_raw.columns[x_train_raw.isna().all()].tolist()
    if all_nan_train_features:
        x_train_raw = x_train_raw.drop(columns=all_nan_train_features)
        x_test_raw = x_test_raw.drop(columns=all_nan_train_features)

    x_train = x_train_raw.astype(float_dtype)
    x_test = x_test_raw.astype(float_dtype)
    return x_train, x_test, pd.Series(dtype=float_dtype), all_nan_train_features, 0


def load_walk_forward_training_frame(
        data_path,
        feature_subset=None,
        excluded_features=None,
        float_dtype=np.float64,
):
    excluded_feature_names = (
        tuple(excluded_features["features"]) if excluded_features else tuple()
    )
    excluded_feature_set = set(excluded_feature_names)
    selected_feature_columns = (
        list(feature_subset["features"]) if feature_subset else None
    )
    subset_parts = (
        split_feature_subset(
            selected_feature_columns,
            source_label=f"feature subset {feature_subset['path']}",
        )
        if feature_subset
        else None
    )
    parquet_columns = None
    if selected_feature_columns is not None:
        session_feature_set = set(subset_parts["session_feature_cols"])
        parquet_columns = list(
            dict.fromkeys(
                [
                    *OOF_EXPORT_BASE_COLS,
                    *(
                        col
                        for col in selected_feature_columns
                        if col not in session_feature_set
                    ),
                ]
            )
        )

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    validate_parquet_magic_bytes(data_path)

    print(f"Loading dataset: {data_path}")
    if feature_subset:
        print(
            "Feature subset active: "
            f"path={feature_subset['path']} count={feature_subset['count']}"
        )
    if excluded_features:
        preview = ", ".join(excluded_feature_names[:5])
        print(
            "Feature exclusions active: "
            f"count={excluded_features['count']} preview=[{preview}]"
        )

    if parquet_columns is not None:
        try:
            parquet_column_set = set(pq.ParquetFile(data_path).schema_arrow.names)
        except Exception as exc:
            raise ValueError(
                "Dataset file is not a readable parquet file. "
                "Rebuild it with create_modeling_dataset.py. "
                f"path={data_path}"
            ) from exc

        missing_parquet_columns = [
            col for col in parquet_columns if col not in parquet_column_set
        ]
        if missing_parquet_columns:
            preview = ", ".join(missing_parquet_columns[:10])
            raise ValueError(
                "Dataset is missing columns required by train_lgbm.py. "
                "Rebuild it with create_modeling_dataset.py for the active feature subset. "
                f"Missing_count={len(missing_parquet_columns)} preview=[{preview}]"
            )

    try:
        df = pd.read_parquet(data_path, columns=parquet_columns)
    except Exception as exc:
        raise ValueError(
            "Failed to read modeling dataset parquet. "
            "Rebuild it with create_modeling_dataset.py. "
            f"path={data_path} root_cause={type(exc).__name__}: {exc}"
        ) from exc
    print(f"Loaded dataset: rows={len(df)} cols={len(df.columns)}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")

    df = df[df[TARGET_COL].notna()]
    if len(df) == 0:
        raise ValueError("No rows left after TARGET_COL non-null filtering.")

    dropped_legacy_cdl = [col for col in CANDLE_PATTERN_COLS if col in df.columns]
    if dropped_legacy_cdl:
        df = df.drop(columns=dropped_legacy_cdl)
        print(f"Dropped legacy 1m CDL columns: {len(dropped_legacy_cdl)}")

    if subset_parts:
        missing_subset_session_features = [
            col for col in subset_parts["session_feature_cols"] if col not in df.columns
        ]
        if missing_subset_session_features:
            df = add_session_open_features(
                df,
                feature_cols=missing_subset_session_features,
            )
    else:
        df = add_session_open_features(df)

    sample_weight, sample_weight_source, sample_weight_summary = (
        resolve_sample_weight_series(df, float_dtype=float_dtype)
    )
    df[TARGET_WEIGHT_COL] = sample_weight

    y = df[TARGET_COL].astype(np.int8)
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

    if selected_feature_columns is not None:
        missing_selected_features = [
            col for col in selected_feature_columns if col not in x.columns
        ]
        if missing_selected_features:
            preview = ", ".join(missing_selected_features[:10])
            raise ValueError(
                "Dataset is missing configured subset features for training. "
                f"Missing_count={len(missing_selected_features)} preview=[{preview}]"
            )
        x = x.loc[:, selected_feature_columns]

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

    x = x.astype(float_dtype)
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


def evaluate_walk_forward_variant(
        x,
        y,
        sample_weight,
        folds,
        param_overrides,
        model_variant,
        collect_oof_predictions=True,
        collect_feature_importance=True,
        early_stopping_verbose=True,
        float_dtype=np.float64,
):
    fold_results = []
    best_iterations = []
    oof_pred_proba = (
        np.full(shape=len(x), fill_value=np.nan, dtype=float_dtype)
        if collect_oof_predictions
        else None
    )
    cv_gain_sum = {} if collect_feature_importance else None
    cv_split_sum = {} if collect_feature_importance else None
    cv_fold_presence = {} if collect_feature_importance else None
    covered_test_rows = 0

    for fold in folds:
        fold_id = fold["fold_id"]
        tr_s, tr_e = fold["train_start"], fold["train_end"]
        te_s, te_e = fold["test_start"], fold["test_end"]

        x_train_raw = x.iloc[tr_s:tr_e]
        y_train = y.iloc[tr_s:tr_e]
        w_train = sample_weight.iloc[tr_s:tr_e].to_numpy(
            dtype=float_dtype, copy=False
        )
        x_test_raw = x.iloc[te_s:te_e]
        y_test = y.iloc[te_s:te_e]
        w_test = sample_weight.iloc[te_s:te_e].to_numpy(
            dtype=float_dtype, copy=False
        )

        x_train, x_test, _, dropped_nan_features, _ = clean_and_impute_fold(
            x_train_raw, x_test_raw, float_dtype=float_dtype
        )

        model = build_lgbm_model(
            n_estimators=N_ESTIMATORS,
            param_overrides=param_overrides,
            feature_names=x_train.columns,
        )
        model.fit(
            x_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(x_test, y_test)],
            eval_sample_weight=[w_test],
            eval_metric=EARLY_STOPPING_EVAL_METRIC,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=EARLY_STOPPING_ROUNDS,
                    first_metric_only=True,
                    verbose=early_stopping_verbose,
                )
            ],
        )

        best_iteration = int(model.best_iteration_ or N_ESTIMATORS)
        y_pred_proba = model.predict_proba(x_test, num_iteration=best_iteration)[:, 1]
        metrics = classification_metrics(
            y_test.to_numpy(),
            y_pred_proba,
            sample_weight=w_test,
        )

        covered_test_rows += len(x_test)
        if oof_pred_proba is not None:
            oof_pred_proba[te_s:te_e] = y_pred_proba

        if collect_feature_importance:
            for feature_name, gain_value, split_value in zip(
                    model.booster_.feature_name(),
                    model.booster_.feature_importance(importance_type="gain"),
                    model.booster_.feature_importance(importance_type="split"),
            ):
                cv_gain_sum[feature_name] = cv_gain_sum.get(feature_name, 0.0) + float(
                    gain_value
                )
                cv_split_sum[feature_name] = cv_split_sum.get(
                    feature_name, 0.0
                ) + float(split_value)
                cv_fold_presence[feature_name] = (
                        cv_fold_presence.get(feature_name, 0) + 1
                )

        best_iterations.append(best_iteration)
        fold_results.append(
            {
                "fold_id": fold_id,
                "train_size": len(x_train),
                "test_size": len(x_test),
                "dropped_all_nan_train_features_count": len(dropped_nan_features),
                "best_iteration": best_iteration,
                "metrics": metrics,
            }
        )

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "brier_score",
        "binary_logloss",
    ]
    metric_arrays = {
        m: np.array([f["metrics"][m] for f in fold_results], dtype=np.float64)
        for m in metric_names
    }

    if oof_pred_proba is None:
        oof_rows = int(covered_test_rows)
        oof_coverage_ratio = (
            float(covered_test_rows / len(x)) if len(x) > 0 else float("nan")
        )
    else:
        oof_rows = int(np.isfinite(oof_pred_proba).sum())
        oof_coverage_ratio = float(np.isfinite(oof_pred_proba).mean())

    result = {
        "mean_best_iteration": int(np.round(np.mean(best_iterations))),
        "cv_mean_metrics": {m: float(np.mean(metric_arrays[m])) for m in metric_names},
        "cv_std_metrics": {m: float(np.std(metric_arrays[m])) for m in metric_names},
        "folds": fold_results,
        "oof_predictions_enabled": bool(collect_oof_predictions),
        "oof_rows": oof_rows if collect_oof_predictions else 0,
        "oof_coverage_ratio": (
            oof_coverage_ratio if collect_oof_predictions else 0.0
        ),
    }
    cv_feature_importance = None
    if collect_feature_importance:
        cv_feature_importance = pd.DataFrame(
            [
                {
                    "model_variant": model_variant,
                    "feature": feature_name,
                    "fold_presence": int(cv_fold_presence[feature_name]),
                    "importance_gain_sum": float(cv_gain_sum[feature_name]),
                    "importance_gain_mean": float(
                        cv_gain_sum[feature_name] / len(folds)
                    ),
                    "importance_split_sum": float(cv_split_sum[feature_name]),
                    "importance_split_mean": float(
                        cv_split_sum[feature_name] / len(folds)
                    ),
                }
                for feature_name in cv_gain_sum
            ]
        ).sort_values("importance_gain_mean", ascending=False)
    return result, oof_pred_proba, cv_feature_importance


def main():
    dataset_settings = load_modeling_dataset_settings()
    normalized_volume_profile_cfg = normalize_volume_profile_config(
        dataset_settings.get("volume_profile_fixed_range")
    )
    modeling_float_dtype = resolve_modeling_float_dtype(dataset_settings)
    modeling_float_dtype_name = resolve_modeling_float_dtype_name(dataset_settings)
    data_path = resolve_modeling_dataset_output_paths(dataset_settings)["parquet"]
    oof_output_paths = resolve_oof_prediction_output_paths(
        dataset_settings,
        preview_rows=OOF_PREVIEW_ROWS,
    )
    oof_output_path = oof_output_paths["parquet"]
    feature_subset = load_feature_subset_from_settings(dataset_settings)
    excluded_features = load_excluded_feature_names_from_settings(dataset_settings)
    train_lgbm_settings = dict(dataset_settings.get("train_lgbm") or {})
    train_default_model = bool(train_lgbm_settings.get("train_default_model", True))
    save_oof_predictions = bool(train_lgbm_settings.get("save_oof_predictions", True))
    monotone_constraints_by_feature = dict(
        train_lgbm_settings.get("monotone_constraints") or {}
    )
    training_data = load_walk_forward_training_frame(
        data_path=data_path,
        feature_subset=feature_subset,
        excluded_features=excluded_features,
        float_dtype=modeling_float_dtype,
    )
    df = training_data["df"]
    x = training_data["x"]
    y = training_data["y"]
    sample_weight = training_data["sample_weight"]
    sample_weight_source = training_data["sample_weight_source"]
    sample_weight_summary = training_data["sample_weight_summary"]
    class_distribution = training_data["class_distribution"]
    weighted_class_distribution = training_data["weighted_class_distribution"]
    dropped_raw_ohlcv_features = training_data["dropped_raw_ohlcv_features"]
    non_numeric_features = training_data["dropped_non_numeric_features"]

    folds = make_walk_forward_folds(
        n_rows=len(x),
        n_folds=CV_FOLDS,
        test_to_train_ratio=WF_TEST_TO_TRAIN_RATIO,
    )
    print(
        f"Walk-forward CV: folds={len(folds)}, "
        f"test/train={WF_TEST_TO_TRAIN_RATIO:.3f}"
    )
    print(
        f"Sample weights | source={sample_weight_source} "
        f"summary={sample_weight_summary}"
    )
    print(
        "Numeric precision | "
        f"configured_float_precision={modeling_float_dtype_name} "
        f"train_feature_matrix={modeling_float_dtype_name} "
        f"sample_weight={modeling_float_dtype_name}"
    )
    print(
        "Train switches | "
        f"default_model={train_default_model} "
        f"save_oof_predictions={save_oof_predictions}"
    )
    monotone_constraint_summary = summarize_lgbm_monotone_constraints(
        x.columns,
        constraints_by_feature=monotone_constraints_by_feature,
    )
    print(
        "Monotone constraints | "
        f"{format_lgbm_monotone_constraint_summary(monotone_constraint_summary)}"
    )

    cv_variants = {}
    if train_default_model:
        cv_variants["default"] = LGBM_DEFAULT_PARAMS
    cv_variants["optuna"] = LGBM_OPTUNA_BEST_PARAMS
    cv_results = {}
    cv_oof_predictions = {}
    cv_feature_importance_by_variant = {}

    for model_variant, param_overrides in cv_variants.items():
        collect_oof_predictions = save_oof_predictions and model_variant == "optuna"
        cv_result, oof_pred_proba, cv_feature_importance = (
            evaluate_walk_forward_variant(
                x=x,
                y=y,
                sample_weight=sample_weight,
                folds=folds,
                param_overrides=param_overrides,
                model_variant=model_variant,
                collect_oof_predictions=collect_oof_predictions,
                float_dtype=modeling_float_dtype,
            )
        )
        cv_results[model_variant] = cv_result
        cv_oof_predictions[model_variant] = oof_pred_proba
        cv_feature_importance_by_variant[model_variant] = cv_feature_importance

        print(
            f"CV[{model_variant}] | "
            f"brier={cv_result['cv_mean_metrics']['brier_score']:.6f}, "
            f"logloss={cv_result['cv_mean_metrics']['binary_logloss']:.6f}, "
            f"acc={cv_result['cv_mean_metrics']['accuracy']:.4f}, "
            f"bal_acc={cv_result['cv_mean_metrics']['balanced_accuracy']:.4f}, "
            f"precision={cv_result['cv_mean_metrics']['precision']:.4f}, "
            f"recall={cv_result['cv_mean_metrics']['recall']:.4f}, "
            f"f1={cv_result['cv_mean_metrics']['f1']:.4f}"
        )
        print(
            f"OOF[{model_variant}] | "
            + (
                f"rows={cv_result['oof_rows']}, "
                f"coverage={cv_result['oof_coverage_ratio']:.4f}"
                if cv_result["oof_predictions_enabled"]
                else "disabled"
            )
        )

    cv_result = cv_results["optuna"]
    oof_pred_proba = cv_oof_predictions["optuna"]
    oof_export = None
    oof_head_csv = None
    oof_tail_csv = None
    if save_oof_predictions:
        if oof_pred_proba is None:
            raise RuntimeError(
                "save_oof_predictions=True but optuna OOF predictions were not collected."
            )
        oof_mask = np.isfinite(oof_pred_proba)
        oof_export = df.loc[oof_mask, OOF_EXPORT_BASE_COLS].assign(
            **{
                OOF_PRED_COL: oof_pred_proba[oof_mask].astype(
                    modeling_float_dtype, copy=False
                )
            }
        )
        oof_output_path.parent.mkdir(parents=True, exist_ok=True)
        oof_export.to_parquet(oof_output_path, index=False)
        oof_head_csv = oof_output_paths["head_csv"]
        oof_tail_csv = oof_output_paths["tail_csv"]
        oof_export.head(OOF_PREVIEW_ROWS).to_csv(oof_head_csv, index=False)
        oof_export.tail(OOF_PREVIEW_ROWS).to_csv(oof_tail_csv, index=False)

    best_iteration = max(10, int(cv_result["mean_best_iteration"]))

    x_full, _, _, all_nan_train_features, _ = clean_and_impute_fold(
        x, x, float_dtype=modeling_float_dtype
    )
    final_monotone_constraint_summary = summarize_lgbm_monotone_constraints(
        x_full.columns,
        constraints_by_feature=monotone_constraints_by_feature,
    )

    model = build_lgbm_model(
        n_estimators=best_iteration,
        param_overrides=LGBM_OPTUNA_BEST_PARAMS,
        feature_names=x_full.columns,
        monotone_constraints_by_feature=monotone_constraints_by_feature,
    )
    model.fit(
        x_full,
        y,
        sample_weight=sample_weight.to_numpy(
            dtype=modeling_float_dtype, copy=False
        ),
    )

    y_full_pred_proba = model.predict_proba(x_full)[:, 1]
    full_fit_metrics = classification_metrics(
        y.to_numpy(),
        y_full_pred_proba,
        sample_weight=sample_weight.to_numpy(
            dtype=modeling_float_dtype, copy=False
        ),
    )

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MODELS_DIR / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / f"lgbm_{run_timestamp}.txt"
    meta_path = run_dir / f"lgbm_meta_{run_timestamp}.json"
    fi_path = run_dir / f"lgbm_feature_importance_{run_timestamp}.csv"
    cv_fi_paths = {
        model_variant: run_dir
                       / f"lgbm_cv_feature_importance_{model_variant}_{run_timestamp}.csv"
        for model_variant in cv_variants
    }

    model.booster_.save_model(str(model_path))

    fi_df = pd.DataFrame(
        {
            "feature": x_full.columns,
            "importance_gain": model.booster_.feature_importance(
                importance_type="gain"
            ),
            "importance_split": model.booster_.feature_importance(
                importance_type="split"
            ),
        }
    ).sort_values("importance_gain", ascending=False)
    fi_df.to_csv(fi_path, index=False)

    for (
            model_variant,
            cv_feature_importance,
    ) in cv_feature_importance_by_variant.items():
        cv_feature_importance.drop(columns=["model_variant"]).to_csv(
            cv_fi_paths[model_variant],
            index=False,
        )

    meta_payload = {
        "active_asset": str(dataset_settings.get("active_asset", "")),
        "data_path": path_to_portable_str(data_path),
        "target_col": TARGET_COL,
        "rows_used": len(df),
        "rows_after_target_notna": len(df),
        "decision_row_filter": {
            "enabled": False,
        },
        "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
        "weighted_class_distribution": weighted_class_distribution,
        "sample_weight": {
            "column": TARGET_WEIGHT_COL,
            "used": True,
            "source": sample_weight_source,
            **sample_weight_summary,
        },
        "feature_count": int(x_full.shape[1]),
        "dropped_pseudo_targets": [],
        "dropped_raw_ohlcv_features": dropped_raw_ohlcv_features,
        "dropped_non_numeric_features": non_numeric_features,
        "dropped_all_nan_train_features": all_nan_train_features,
        "best_iteration": best_iteration,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "oof_predictions": {
            "enabled": save_oof_predictions,
            "path": (
                path_to_portable_str(oof_output_path)
                if save_oof_predictions
                else None
            ),
            "prediction_col": OOF_PRED_COL if save_oof_predictions else None,
            "model_variant": "optuna" if save_oof_predictions else None,
            "rows": len(oof_export) if oof_export is not None else 0,
            "rows_without_oof": (
                len(df) - len(oof_export) if oof_export is not None else len(df)
            ),
            "coverage_ratio": (
                float(len(oof_export) / len(df))
                if oof_export is not None and len(df) > 0
                else 0.0
            ),
        },
        "metrics": {
            "cv": {
                model_variant: {
                    "mean": cv_data["cv_mean_metrics"],
                    "std": cv_data["cv_std_metrics"],
                }
                for model_variant, cv_data in cv_results.items()
            },
            "full_fit_optuna": full_fit_metrics,
        },
        "feature_selection": summarize_feature_subset(
            feature_subset,
            excluded_features=excluded_features,
        ),
        "numeric_precision": {
            "configured_float_precision": modeling_float_dtype_name,
            "parquet_float_columns": modeling_float_dtype_name,
            "train_feature_matrix": modeling_float_dtype_name,
            "sample_weight": modeling_float_dtype_name,
            "oof_prediction": modeling_float_dtype_name,
        },
        "feature_columns": list(x_full.columns),
        "monotone_constraints": final_monotone_constraint_summary,
        "volume_profile_fixed_range": (
            normalized_volume_profile_cfg
            if any(is_volume_profile_feature(col) for col in x_full.columns)
            else None
        ),
        "model_hyperparameters": {
            "base": {
                "objective": "binary",
                "n_estimators_cv": N_ESTIMATORS,
                "n_estimators_final": best_iteration,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "early_stopping_metric": EARLY_STOPPING_METRIC,
                "device_type": LGBM_DEVICE_TYPE,
                "verbosity": LGBM_VERBOSITY,
                "random_state": SEED,
                "n_jobs": N_JOBS,
                "max_bin": LGBM_MAX_BIN,
            },
            "cv_variants_trained": list(cv_variants.keys()),
            "cv_default_params": LGBM_DEFAULT_PARAMS if train_default_model else None,
            "cv_optuna_params": LGBM_OPTUNA_BEST_PARAMS,
            "final_model_variant": "optuna",
        },
        "train_config": {
            "cv_folds": CV_FOLDS,
            "walk_forward_test_to_train_ratio": WF_TEST_TO_TRAIN_RATIO,
            "primary_reporting_metric": PRIMARY_REPORTING_METRIC,
            "train_default_model": train_default_model,
            "save_oof_predictions": save_oof_predictions,
        },
        "walk_forward_folds": folds,
        "walk_forward_details": {
            model_variant: cv_data["folds"]
            for model_variant, cv_data in cv_results.items()
        },
        "artifacts": {
            "run_dir": path_to_portable_str(run_dir),
            "final_model_path": path_to_portable_str(model_path),
            "final_feature_importance_csv": path_to_portable_str(fi_path),
            "cv_feature_importance_csv": {
                model_variant: path_to_portable_str(path)
                for model_variant, path in cv_fi_paths.items()
            },
            "oof_predictions_path": (
                path_to_portable_str(oof_output_path)
                if save_oof_predictions
                else None
            ),
        },
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Feature importance saved (csv): {fi_path}")
    for model_variant, cv_fi_path in cv_fi_paths.items():
        print(f"CV feature importance saved ({model_variant}): {cv_fi_path}")
    if save_oof_predictions:
        print(f"OOF predictions saved (parquet): {oof_output_path}")
        print(f"OOF preview head saved: {oof_head_csv}")
        print(f"OOF preview tail saved: {oof_tail_csv}")
    else:
        print("OOF predictions export disabled")

    metrics_summary_parts = []
    if "default" in cv_results:
        metrics_summary_parts.append(
            "CV[default] "
            f"brier={cv_results['default']['cv_mean_metrics']['brier_score']:.6f}, "
            f"logloss={cv_results['default']['cv_mean_metrics']['binary_logloss']:.6f}"
        )
    metrics_summary_parts.append(
        "CV[optuna] "
        f"brier={cv_results['optuna']['cv_mean_metrics']['brier_score']:.6f}, "
        f"logloss={cv_results['optuna']['cv_mean_metrics']['binary_logloss']:.6f}"
    )
    metrics_summary_parts.append(
        "FULL_FIT "
        f"brier={full_fit_metrics['brier_score']:.6f}, "
        f"logloss={full_fit_metrics['binary_logloss']:.6f}, "
        f"acc={full_fit_metrics['accuracy']:.4f}, "
        f"bal_acc={full_fit_metrics['balanced_accuracy']:.4f}, "
        f"precision={full_fit_metrics['precision']:.4f}, "
        f"recall={full_fit_metrics['recall']:.4f}, "
        f"f1={full_fit_metrics['f1']:.4f}"
    )
    print("Metrics | " + " | ".join(metrics_summary_parts))


if __name__ == "__main__":
    main()
