import json
import faulthandler
import warnings
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from features.candle_features import RAW_OHLCV_COLS
from features.reaction_profile_fixed_grid import (
    build_reaction_profile_feature_matrix_from_arrays,
    normalize_config as normalize_reaction_profile_config,
)
from train_lgbm import (
    WF_TEST_TO_TRAIN_RATIO as DEFAULT_WF_TEST_TO_TRAIN_RATIO,
    format_lgbm_monotone_constraint_summary,
    make_lgbm_monotone_constraint_params,
    summarize_lgbm_monotone_constraints,
)
from utils.data import (
    TARGET_WEIGHT_COL,
    TARGET_WEIGHT_DECISION_VALUE,
    add_target_weights,
    compute_binary_close_target_from_opened,
    summarize_target_weights,
)
from utils.data import drop_frozen_ohlc_blocks
from utils.data import load_modeling_dataset_settings
from utils.metrics import (
    make_lightgbm_binary_balanced_accuracy_eval,
    make_lightgbm_binary_brier_eval,
    make_lightgbm_binary_logloss_eval,
    weighted_balanced_accuracy_score,
    weighted_binary_logloss,
)
from utils.optuna import (
    make_timestamped_artifact_path,
    resolve_run_study_name,
)
from utils.project_config import active_asset_path

TARGET_TIME_COL = "Opened"
TARGET_PRICE_COL = "Close"
TARGET_COL = "target_5m_candle_up"
TARGET_HORIZON_MINUTES = 5

MODELING_DATASET_SETTINGS = load_modeling_dataset_settings()
BASE_DATA_PATH = Path(MODELING_DATASET_SETTINGS["raw_data_dir"]) / str(
    MODELING_DATASET_SETTINGS["base_data_file"]
)
SEED = 37

CV_FOLDS = 10
WF_TEST_TO_TRAIN_RATIO = DEFAULT_WF_TEST_TO_TRAIN_RATIO
ENABLE_FOLD_RECENCY_WEIGHTING = True
FOLD_RECENCY_WEIGHTING_MODE = "linear"
FOLD_RECENCY_WEIGHT_MIN = 1.0
FOLD_RECENCY_WEIGHT_MAX = 1.5
MIN_SAMPLE_WEIGHT = float(TARGET_WEIGHT_DECISION_VALUE)

MAX_N_ESTIMATORS = 300
EARLY_STOPPING_ROUNDS = 40
PRUNE_REPORT_EVERY_N_ITER = 10

LGBM_NUM_THREADS = 16
OPTUNA_OPTIMIZE_N_JOBS = 1
LGBM_DEVICE_TYPE = "gpu"
LGBM_VERBOSITY = -1
GPU_MAX_BIN_LIMIT = 63
LGBM_GPU_USE_DP = True

LGBM_DEFAULT_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 128,
    "max_depth": 6,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "lambda_l2": 5.0,
    "lambda_l1": 0.0,
    "min_sum_hessian_in_leaf": 0.001,
    "min_gain_to_split": 0.0,
    "feature_fraction_bynode": 1.0,
    "path_smooth": 0.0,
    "extra_trees": False,
}

REACTION_PROFILE_OPTUNA_SEARCH_SPACE = {
    "bin_size": {"type": "float", "low": 0.5, "high": 100.0, "log": True},
    "neighbor_bins": {"type": "float", "low": 0.0, "high": 16.0, "log": False},
    "short_local_window": {"type": "int", "low": 1, "high": 128, "log": False},
    "medium_local_window": {"type": "int", "low": 1, "high": 128, "log": False},
    "long_local_window": {"type": "int", "low": 1, "high": 128, "log": False},
    "all_local_window": {"type": "int", "low": 1, "high": 128, "log": False},
    "short_half_life_candles": {
        "type": "int",
        "low": 10,
        "high": 2400,
        "log": True,
    },
    "medium_half_life_candles": {
        "type": "int",
        "low": 2401,
        "high": 14400,
        "log": True,
    },
    "long_half_life_candles": {
        "type": "int",
        "low": 14401,
        "high": 86400,
        "log": True,
    },
    "min_reaction_strength": {"type": "float", "low": 0.0, "high": 0.001, "log": False},
    "wick_power": {"type": "float", "low": 0.25, "high": 4.0, "log": True},
    "distance_power": {"type": "float", "low": 0.25, "high": 4.0, "log": True},
}

OPTUNA_SEED_TRIAL_PARAMS = [
    {
    "bin_size": 88.13347937442842,
    "neighbor_bins": 14.475523355248354,
    "short_local_window": 67,
    "medium_local_window": 103,
    "long_local_window": 40,
    "all_local_window": 81,
    "short_half_life_candles": 57,
    "medium_half_life_candles": 3702,
    "long_half_life_candles": 21450,
    "min_reaction_strength": 7.776895291626223e-05,
    "wick_power": 1.481640187824231,
    "distance_power": 0.26262391136729163
    }
]

N_TRIALS = 500
TIMEOUT_SECONDS = None
LOAD_IF_EXISTS = True
TPE_STARTUP_TRIALS = int(N_TRIALS * 0.1)

CV_OBJECTIVE_BASE_METRIC = "binary_logloss"
EARLY_STOPPING_METRIC = CV_OBJECTIVE_BASE_METRIC
CV_OBJECTIVE_IS_HIGHER_BETTER = False
CV_STD_PENALTY = 0.75
CRASH_PENALTY = float("inf")
DEFAULT_STUDY_NAME_PREFIX = "reaction_profile_binary_logloss_mean_std"
# Leave empty for a fresh timestamped study. Set only to continue an existing one.
STUDY_NAME = "reaction_profile_binary_logloss_mean_std_20260620_230533"
STORAGE = (
        "sqlite:///"
        + active_asset_path("data/optuna/databases/{asset}/reaction_profile.db").as_posix()
)
ARTIFACT_OUTPUT_DIR = active_asset_path("data/optuna/reaction_profile/{asset}")
BEST_RESULT_STEM = "reaction_profile_best_binary_logloss_mean_std"
TRIALS_CSV_STEM = "reaction_profile_trials_binary_logloss_mean_std"
NATIVE_CRASH_LOG_PATH = ARTIFACT_OUTPUT_DIR / "reaction_profile_native_crash.log"


def enable_native_crash_log():
    NATIVE_CRASH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    crash_log_file = NATIVE_CRASH_LOG_PATH.open("a", encoding="utf-8")
    crash_log_file.write(
        f"\n=== fit_reaction_profile start {datetime.now(timezone.utc).isoformat()} ===\n"
    )
    crash_log_file.flush()
    faulthandler.enable(file=crash_log_file, all_threads=True)
    print(f"native crash log -> {NATIVE_CRASH_LOG_PATH}")
    return crash_log_file


def require_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def build_target_frame(df):
    out = df.copy()
    out[TARGET_TIME_COL] = pd.to_datetime(out[TARGET_TIME_COL], errors="raise")
    out = out.sort_values(TARGET_TIME_COL).reset_index(drop=True)
    out[TARGET_COL] = compute_binary_close_target_from_opened(
        opened_values=out[TARGET_TIME_COL],
        close_values=out[TARGET_PRICE_COL],
        horizon_minutes=TARGET_HORIZON_MINUTES,
    )
    out = add_target_weights(
        out, opened_col=TARGET_TIME_COL, weight_col=TARGET_WEIGHT_COL
    )
    return out


def validate_sample_weight_array(sample_weight_np):
    if sample_weight_np.ndim != 1:
        raise ValueError("Sample weights must be a 1D array.")
    if sample_weight_np.size == 0:
        raise ValueError("Sample weights array is empty.")
    if not np.isfinite(sample_weight_np).all():
        raise ValueError("Sample weights contain non-finite values.")
    if np.any(sample_weight_np <= 0.0):
        raise ValueError("Sample weights must be strictly positive.")


def load_base_ohlcv_frame(data_path):
    if not data_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {data_path}")

    required_columns = [TARGET_TIME_COL, *RAW_OHLCV_COLS]
    print(f"load raw data | path={data_path}")
    df = pd.read_csv(data_path, usecols=required_columns)
    require_columns(df, required_columns)
    raw_rows = len(df)
    df, drop_frozen_summary = drop_frozen_ohlc_blocks(
        df,
        raw_config=MODELING_DATASET_SETTINGS.get("drop_frozen_ohlc_blocks"),
    )
    if drop_frozen_summary["enabled"]:
        print(
            "load raw data | drop_frozen_ohlc_blocks "
            f"min_block_len={drop_frozen_summary['min_block_len']} "
            f"removed_rows={drop_frozen_summary['rows_removed']} "
            f"removed_blocks={drop_frozen_summary['blocks_removed']} "
            f"largest_block_len={drop_frozen_summary['largest_block_len']} "
            f"rows_after={drop_frozen_summary['rows_after']}"
        )

    df = build_target_frame(df)
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    rows_after_target_notna = len(df)
    if rows_after_target_notna == 0:
        raise ValueError("No rows left after target construction.")

    sample_weight_full = pd.to_numeric(df[TARGET_WEIGHT_COL], errors="raise").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    validate_sample_weight_array(sample_weight_full)

    keep_mask = sample_weight_full >= float(MIN_SAMPLE_WEIGHT)
    filtered_rows = int(keep_mask.sum())
    if filtered_rows == 0:
        raise ValueError(
            f"No rows left after row filter {TARGET_WEIGHT_COL}>={MIN_SAMPLE_WEIGHT:.2f}."
        )

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        df = df.astype({col: np.float32 for col in numeric_columns})

    y_full = df[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
    y_filtered = y_full[keep_mask]
    sample_weight_filtered = sample_weight_full[keep_mask]
    class_distribution = {
        int(cls): int(count)
        for cls, count in zip(
            *np.unique(y_filtered.astype(np.int8), return_counts=True)
        )
    }
    weighted_class_distribution = {
        str(int(class_id)): float(
            sample_weight_filtered[y_filtered == float(class_id)].sum()
        )
        for class_id in sorted(class_distribution.keys())
    }
    row_filter_info = {
        "enabled": True,
        "weight_col": TARGET_WEIGHT_COL,
        "min_weight": float(MIN_SAMPLE_WEIGHT),
        "rows_before": int(rows_after_target_notna),
        "rows_after": int(filtered_rows),
        "rows_removed": int(rows_after_target_notna - filtered_rows),
    }
    open_np = df["Open"].to_numpy(dtype=np.float64, copy=False)
    high_np = df["High"].to_numpy(dtype=np.float64, copy=False)
    low_np = df["Low"].to_numpy(dtype=np.float64, copy=False)
    close_np = df["Close"].to_numpy(dtype=np.float64, copy=False)

    print(
        f"load raw data | raw_rows={raw_rows} rows_after_target_notna={rows_after_target_notna}"
    )
    print(
        f"load raw data | row_filter {TARGET_WEIGHT_COL}>={MIN_SAMPLE_WEIGHT:.2f} "
        f"removed={row_filter_info['rows_removed']} remaining={row_filter_info['rows_after']}"
    )
    print(
        f"load raw data | filtered_class_distribution={class_distribution} "
        f"filtered_weighted_class_distribution={weighted_class_distribution}"
    )

    return {
        "df": df,
        "keep_mask": keep_mask,
        "open_np": open_np,
        "high_np": high_np,
        "low_np": low_np,
        "close_np": close_np,
        "y_filtered": y_filtered.astype(np.float32, copy=False),
        "raw_rows": raw_rows,
        "rows_after_target_notna": rows_after_target_notna,
        "sample_weight_full": sample_weight_full,
        "sample_weight_filtered": sample_weight_filtered,
        "sample_weight_summary": summarize_target_weights(sample_weight_filtered),
        "class_distribution": class_distribution,
        "weighted_class_distribution": weighted_class_distribution,
        "row_filter_info": row_filter_info,
    }


def make_walk_forward_folds(n_rows, n_folds, test_to_train_ratio):
    if n_rows < 100:
        raise ValueError(f"Dataset too small for walk-forward CV: {n_rows} rows.")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    if not (0.0 < test_to_train_ratio < 1.0):
        raise ValueError("test_to_train_ratio must be in (0, 1).")

    ratio_inv = 1.0 / test_to_train_ratio
    test_len = int(np.floor(n_rows / (n_folds + ratio_inv)))
    train_len = int(np.floor(test_len / test_to_train_ratio))

    if test_len <= 0 or train_len <= 0:
        raise ValueError(
            f"Cannot create valid folds for n_rows={n_rows}, "
            f"n_folds={n_folds}, ratio={test_to_train_ratio}."
        )

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

    if len(folds) != n_folds:
        raise ValueError(
            f"Created {len(folds)} folds, expected {n_folds}. "
            "Increase dataset size or lower folds."
        )
    return folds


def build_fold_indices(folds):
    return [
        (
            np.arange(fold["train_start"], fold["train_end"], dtype=np.int32),
            np.arange(fold["test_start"], fold["test_end"], dtype=np.int32),
        )
        for fold in folds
    ]


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


def cv_objective_base_score_label():
    if is_nontrivial_fold_recency_weighting_enabled():
        return f"cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean"
    return f"cv_{CV_OBJECTIVE_BASE_METRIC}_mean"


def resolve_cv_objective_name():
    suffix = "minus_std_penalty" if CV_OBJECTIVE_IS_HIGHER_BETTER else "plus_std_penalty"
    return f"{cv_objective_base_score_label()}_{suffix}"


def combine_metric_mean_std(mean_value, std_value, *, higher_is_better, std_penalty):
    if bool(higher_is_better):
        return float(mean_value - (float(std_penalty) * float(std_value)))
    return float(mean_value + (float(std_penalty) * float(std_value)))


def summarize_cv_fold_scores(fold_scores, folds, fold_weight_by_id, std_penalty):
    fold_scores_arr = np.asarray(fold_scores, dtype=np.float64)
    if fold_scores_arr.ndim != 1:
        raise ValueError("fold_scores must be a 1D array.")
    if fold_scores_arr.shape[0] != len(folds):
        raise ValueError(
            "fold_scores length mismatch: "
            f"{fold_scores_arr.shape[0]} != {len(folds)}"
        )

    mean_score = float(np.mean(fold_scores_arr))
    weighted_mean_score = weighted_mean_vector(
        fold_scores_arr,
        resolve_fold_weight_array(folds, fold_weight_by_id),
    )
    std_score = float(np.std(fold_scores_arr))
    objective_base_value = (
        weighted_mean_score
        if is_nontrivial_fold_recency_weighting_enabled()
        else mean_score
    )
    return {
        f"cv_{CV_OBJECTIVE_BASE_METRIC}_mean": mean_score,
        f"cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean": weighted_mean_score,
        f"cv_{CV_OBJECTIVE_BASE_METRIC}_std": std_score,
        "objective_base_value": float(objective_base_value),
        "objective_value": combine_metric_mean_std(
            objective_base_value,
            std_score,
            higher_is_better=CV_OBJECTIVE_IS_HIGHER_BETTER,
            std_penalty=std_penalty,
        ),
    }


def score_cv_objective_metric(y_true, y_pred_proba, sample_weight):
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred_proba, dtype=np.float64)
    sample_weight_arr = np.asarray(sample_weight, dtype=np.float64)
    if CV_OBJECTIVE_BASE_METRIC == "balanced_accuracy":
        return float(
            weighted_balanced_accuracy_score(
                y_true=y_true_arr,
                y_pred_proba=y_pred_arr,
                sample_weight=sample_weight_arr,
            )
        )
    if CV_OBJECTIVE_BASE_METRIC == "binary_logloss":
        return float(
            weighted_binary_logloss(
                y_true=y_true_arr,
                y_pred_proba=y_pred_arr,
                sample_weight=sample_weight_arr,
            )
        )
    raise ValueError(
        "Per-fold rescoring supports CV_OBJECTIVE_BASE_METRIC in "
        "{'balanced_accuracy', 'binary_logloss'}."
    )


def compute_cv_fold_scores_at_iteration(
        cvbooster,
        x_np,
        y_np,
        sample_weight_np,
        folds,
        best_iteration,
):
    boosters = getattr(cvbooster, "boosters", None)
    if boosters is None:
        raise ValueError("cvbooster is missing boosters.")
    if len(boosters) != len(folds):
        raise ValueError(
            f"cvbooster fold count mismatch: {len(boosters)} != {len(folds)}"
        )

    fold_scores = []
    for booster, fold in zip(boosters, folds):
        valid_start = int(fold["test_start"])
        valid_end = int(fold["test_end"])
        y_pred_proba = booster.predict(
            x_np[valid_start:valid_end],
            num_iteration=int(best_iteration),
        )
        fold_scores.append(
            score_cv_objective_metric(
                y_true=y_np[valid_start:valid_end],
                y_pred_proba=y_pred_proba,
                sample_weight=sample_weight_np[valid_start:valid_end],
            )
        )

    return np.asarray(fold_scores, dtype=np.float64)


class ObjectiveAlignedLightGBMPruningCallback:
    def __init__(
            self,
            trial,
            metric,
            std_penalty,
            valid_name="valid_0",
            report_interval=1,
    ):
        self._trial = trial
        self._valid_name = valid_name
        self._metric = metric
        self._std_penalty = std_penalty
        self._report_interval = report_interval

    def _find_evaluation_result(self, target_valid_names, env):
        evaluation_result_list = env.evaluation_result_list
        if evaluation_result_list is None:
            return None

        for evaluation_result in evaluation_result_list:
            valid_name, metric = evaluation_result[:2]
            if valid_name not in target_valid_names:
                continue
            if metric != self._metric and metric != f"valid {self._metric}":
                continue
            return evaluation_result

        return None

    def __call__(self, env):
        if (env.iteration + 1) % self._report_interval != 0:
            return

        evaluation_result_list = env.evaluation_result_list
        is_cv = (
                evaluation_result_list is not None
                and len(evaluation_result_list) > 0
                and len(evaluation_result_list[0]) == 5
        )
        target_valid_names = ("cv_agg", "valid") if is_cv else (self._valid_name,)
        evaluation_result = self._find_evaluation_result(target_valid_names, env)
        if evaluation_result is None:
            raise ValueError(
                'The entry associated with the validation names "{}" and the metric name "{}" '
                "is not found in the evaluation result list {}.".format(
                    ", ".join(target_valid_names),
                    self._metric,
                    str(env.evaluation_result_list),
                )
            )

        _, _, current_score, is_higher_better = evaluation_result[:4]
        if is_higher_better:
            if self._trial.study.direction != optuna.study.StudyDirection.MAXIMIZE:
                raise ValueError(
                    "The intermediate values are inconsistent with the objective values "
                    "in terms of study directions."
                )
        else:
            if self._trial.study.direction != optuna.study.StudyDirection.MINIMIZE:
                raise ValueError(
                    "The intermediate values are inconsistent with the objective values "
                    "in terms of study directions."
                )

        if is_cv:
            current_mean = float(evaluation_result[2])
            current_std = float(evaluation_result[4])
            current_objective = combine_metric_mean_std(
                current_mean,
                current_std,
                higher_is_better=is_higher_better,
                std_penalty=self._std_penalty,
            )
        else:
            current_objective = float(current_score)

        self._trial.report(current_objective, step=env.iteration)

        if self._trial.should_prune():
            raise optuna.TrialPruned(f"Trial was pruned at iteration {env.iteration}.")


def get_supported_reaction_profile_param_names(normalized_base):
    param_names = {
        "bin_size",
        "neighbor_bins",
        "min_reaction_strength",
        "wick_power",
        "distance_power",
    }
    for horizon_name in normalized_base["horizon_names"]:
        param_names.add(f"{horizon_name}_local_window")
        if normalized_base["horizons"][horizon_name]["half_life_candles"] is not None:
            param_names.add(f"{horizon_name}_half_life_candles")
    return param_names


def validate_optuna_search_spec(name, spec):
    if not isinstance(spec, dict):
        raise ValueError(f"Search space spec for {name!r} must be a dict.")

    spec_type = str(spec.get("type", "")).strip().lower()
    if spec_type not in {"int", "float"}:
        raise ValueError(
            f"Search space spec for {name!r} must define type='int' or type='float'."
        )

    if "low" not in spec or "high" not in spec:
        raise ValueError(f"Search space spec for {name!r} must define low and high.")

    low = spec["low"]
    high = spec["high"]
    log = bool(spec.get("log", False))

    if spec_type == "int":
        low_i = int(low)
        high_i = int(high)
        step = int(spec.get("step", 1))
        if step <= 0:
            raise ValueError(f"Integer search space step must be > 0 for {name!r}.")
        if log and step != 1:
            raise ValueError(
                f"Integer log search space cannot use step != 1 for {name!r}."
            )
        if log and low_i < 1:
            raise ValueError(
                f"Integer log search space requires low >= 1 for {name!r}."
            )
        if high_i < low_i:
            raise ValueError(f"Integer search space requires high >= low for {name!r}.")
        return

    low_f = float(low)
    high_f = float(high)
    step = spec.get("step")
    if step is not None and float(step) <= 0.0:
        raise ValueError(f"Float search space step must be > 0 for {name!r}.")
    if log and step is not None:
        raise ValueError(f"Float log search space cannot use step for {name!r}.")
    if log and low_f <= 0.0:
        raise ValueError(f"Float log search space requires low > 0 for {name!r}.")
    if high_f < low_f:
        raise ValueError(f"Float search space requires high >= low for {name!r}.")


def validate_reaction_profile_search_space(normalized_base, search_space):
    supported_param_names = get_supported_reaction_profile_param_names(normalized_base)
    search_space_names = set(search_space)
    unknown_names = sorted(search_space_names - supported_param_names)
    if unknown_names:
        raise ValueError(
            "Unsupported reaction profile Optuna search params: "
            f"{unknown_names}. Supported={sorted(supported_param_names)}"
        )

    for name, spec in search_space.items():
        validate_optuna_search_spec(name, spec)


def suggest_value_from_spec(trial, name, spec):
    spec_type = str(spec["type"]).strip().lower()
    log = bool(spec.get("log", False))
    if spec_type == "int":
        return int(
            trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=log,
            )
        )

    step = spec.get("step")
    return float(
        trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            step=float(step) if step is not None else None,
            log=log,
        )
    )


def validate_seed_trial_params(seed_params, search_space):
    unknown_names = sorted(set(seed_params) - set(search_space))
    if unknown_names:
        raise ValueError(
            f"Seed trial contains params not present in search space: {unknown_names}"
        )

    for name, value in seed_params.items():
        spec = search_space[name]
        spec_type = str(spec["type"]).strip().lower()
        if spec_type == "int":
            value_i = int(value)
            low_i = int(spec["low"])
            high_i = int(spec["high"])
            step_i = int(spec.get("step", 1))
            if value_i < low_i or value_i > high_i:
                raise ValueError(
                    f"Seed trial param {name!r}={value_i} is outside [{low_i}, {high_i}]."
                )
            if ((value_i - low_i) % step_i) != 0:
                raise ValueError(
                    f"Seed trial param {name!r}={value_i} does not match step={step_i}."
                )
            continue

        value_f = float(value)
        low_f = float(spec["low"])
        high_f = float(spec["high"])
        if value_f < low_f or value_f > high_f:
            raise ValueError(
                f"Seed trial param {name!r}={value_f} is outside [{low_f}, {high_f}]."
            )
        step_f = spec.get("step")
        if step_f is not None:
            scaled = (value_f - low_f) / float(step_f)
            if not np.isclose(scaled, round(scaled), atol=1e-9):
                raise ValueError(
                    f"Seed trial param {name!r}={value_f} does not match step={step_f}."
                )


def enqueue_seed_trials(study, seed_trial_params, normalized_base, search_space):
    validate_reaction_profile_search_space(normalized_base, search_space)

    for seed_index, params in enumerate(seed_trial_params, start=1):
        validate_seed_trial_params(params, search_space)
        build_reaction_profile_config_from_params(normalized_base, params)
        study.enqueue_trial(
            params=params,
            user_attrs={"seed_trial_index": int(seed_index)},
            skip_if_exists=True,
        )


def suggest_reaction_profile_config(trial, base_config, search_space):
    normalized_base = normalize_reaction_profile_config(base_config)
    validate_reaction_profile_search_space(normalized_base, search_space)
    params = {
        name: suggest_value_from_spec(trial, name, spec)
        for name, spec in search_space.items()
    }
    return build_reaction_profile_config_from_params(normalized_base, params)


def build_reaction_profile_config_from_params(base_config, params):
    normalized_base = normalize_reaction_profile_config(base_config)
    config = {
        "enabled": True,
        "price_min": float(normalized_base["price_min"]),
        "price_max": float(normalized_base["price_max"]),
        "bin_size": float(params.get("bin_size", normalized_base["bin_size"])),
        "neighbor_bins": float(
            params.get("neighbor_bins", normalized_base["neighbor_bins"])
        ),
        "eps": float(normalized_base["eps"]),
        "min_reaction_strength": float(
            params.get(
                "min_reaction_strength",
                normalized_base["min_reaction_strength"],
            )
        ),
        "wick_power": float(params.get("wick_power", normalized_base["wick_power"])),
        "distance_power": float(
            params.get("distance_power", normalized_base["distance_power"])
        ),
        "horizons": {},
    }

    for horizon_name in normalized_base["horizon_names"]:
        base_horizon = normalized_base["horizons"][horizon_name]
        base_half_life = normalized_base["horizons"][horizon_name]["half_life_candles"]
        if base_half_life is None:
            half_life = None
        else:
            half_life = int(
                params.get(f"{horizon_name}_half_life_candles", base_half_life)
            )
        config["horizons"][horizon_name] = {
            "local_window": int(
                params.get(
                    f"{horizon_name}_local_window",
                    base_horizon["local_window"],
                )
            ),
            "half_life_candles": half_life,
        }

    return normalize_reaction_profile_config(config)


def _drop_none_fields(mapping):
    return {key: value for key, value in mapping.items() if value is not None}


def _first_present(mapping, *keys):
    for key in keys:
        if key is None:
            continue
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _numeric_subset(source, *, int_keys=(), float_keys=()):
    out = {}
    for key in int_keys:
        value = source.get(key)
        if value is not None:
            out[key] = int(value)
    for key in float_keys:
        value = source.get(key)
        if value is not None:
            out[key] = float(value)
    return out


def build_editable_reaction_profile_config(cfg):
    normalized = normalize_reaction_profile_config(cfg)
    return {
        "enabled": bool(normalized["enabled"]),
        "price_min": float(normalized["price_min"]),
        "price_max": float(normalized["price_max"]),
        "bin_size": float(normalized["bin_size"]),
        "neighbor_bins": float(normalized["neighbor_bins"]),
        "eps": float(normalized["eps"]),
        "min_reaction_strength": float(normalized["min_reaction_strength"]),
        "wick_power": float(normalized["wick_power"]),
        "distance_power": float(normalized["distance_power"]),
        "horizons": {
            horizon_name: {
                "local_window": int(
                    normalized["horizons"][horizon_name]["local_window"]
                ),
                "half_life_candles": normalized["horizons"][horizon_name][
                    "half_life_candles"
                ],
            }
            for horizon_name in normalized["horizon_names"]
        },
    }


def build_reaction_profile_derived_summary(cfg):
    normalized = normalize_reaction_profile_config(cfg)
    return {
        "version": str(normalized["version"]),
        "feature_count": int(len(normalized["feature_columns"])),
        "features_per_horizon": 7,
        "bins": int(normalized["bins"]),
        "horizons": {
            horizon_name: {
                "decay": float(normalized["horizons"][horizon_name]["decay"]),
            }
            for horizon_name in normalized["horizon_names"]
        },
    }


def compact_reaction_profile_artifact_payload(payload):
    cv_objective = payload.get("cv_objective") or {}
    base_metric = str(
        cv_objective.get("base_metric", CV_OBJECTIVE_BASE_METRIC)
    ).strip() or CV_OBJECTIVE_BASE_METRIC
    mean_key = f"cv_{base_metric}_mean"
    weighted_mean_key = f"cv_{base_metric}_weighted_mean"
    std_key = f"cv_{base_metric}_std"

    best_trial = payload.get("best_trial") or {}
    objective_name = cv_objective.get("name")
    objective_key = f"objective_{objective_name}" if objective_name else None
    best_cfg = best_trial.get("reaction_profile_fixed_grid")
    base_cfg = payload.get("base_reaction_profile_fixed_grid")
    objective_value = _first_present(best_trial, "objective_value", objective_key)
    if objective_value is None:
        objective_value = next(
            (
                value
                for key, value in best_trial.items()
                if str(key).startswith("objective_") and value is not None
            ),
            None,
        )

    sample_weight_summary = payload.get("sample_weight")
    compact_sample_weight = sample_weight_summary
    if isinstance(sample_weight_summary, dict):
        compact_sample_weight = _drop_none_fields(
            {
                "used": sample_weight_summary.get("used"),
                "source": sample_weight_summary.get("source"),
                **_numeric_subset(
                    sample_weight_summary,
                    float_keys=("min", "max", "mean", "sum"),
                ),
            }
        )
        distribution = sample_weight_summary.get("distribution")
        if isinstance(distribution, dict) and 1 < len(distribution) <= 8:
            compact_sample_weight["distribution"] = distribution

    fold_recency_weighting = payload.get("fold_recency_weighting")
    compact_fold_recency_weighting = fold_recency_weighting
    if isinstance(fold_recency_weighting, dict):
        fold_weights = fold_recency_weighting.get("fold_weights") or ()
        compact_fold_recency_weighting = _drop_none_fields(
            {
                "enabled": fold_recency_weighting.get("enabled"),
                "active": fold_recency_weighting.get("active"),
                "mode": fold_recency_weighting.get("mode"),
                "std_score_aggregation": fold_recency_weighting.get(
                    "std_score_aggregation"
                ),
                **_numeric_subset(
                    fold_recency_weighting,
                    float_keys=("min_weight", "max_weight"),
                ),
                "fold_count": int(len(fold_weights)) if fold_weights else None,
                "first_fold_weight": (
                    float(fold_weights[0]["weight"]) if fold_weights else None
                ),
                "last_fold_weight": (
                    float(fold_weights[-1]["weight"]) if fold_weights else None
                ),
            }
        )

    compact_best_trial = _drop_none_fields(
        {
            "trial_number": (
                int(_first_present(best_trial, "trial_number", "number"))
                if _first_present(best_trial, "trial_number", "number") is not None
                else None
            ),
            "objective_name": objective_name,
            "objective_value": (
                float(objective_value) if objective_value is not None else None
            ),
            **_numeric_subset(
                best_trial,
                int_keys=("best_iteration", "feature_count"),
                float_keys=(mean_key, weighted_mean_key, std_key),
            ),
        }
    )
    if "feature_count" not in compact_best_trial and best_cfg is not None:
        compact_best_trial["feature_count"] = int(
            len(normalize_reaction_profile_config(best_cfg)["feature_columns"])
        )

    return _drop_none_fields(
        {
            "created_utc": payload.get("created_utc"),
            "study_name": payload.get("study_name"),
            "study_name_source": payload.get("study_name_source"),
            "run_timestamp_utc": payload.get("run_timestamp_utc"),
            "best_trial": compact_best_trial,
            "best_params_flat": dict(best_trial.get("params") or {}),
            "best_reaction_profile_fixed_grid": (
                build_editable_reaction_profile_config(best_cfg)
                if best_cfg is not None
                else None
            ),
            "best_reaction_profile_derived": (
                build_reaction_profile_derived_summary(best_cfg)
                if best_cfg is not None
                else None
            ),
            "objective": cv_objective or None,
            "dataset": _drop_none_fields(
                {
                    "base_data_path": payload.get("base_data_path"),
                    "target_col": payload.get("target_col"),
                    "sample_weight_col": payload.get("sample_weight_col"),
                    "sample_weight": compact_sample_weight,
                    "class_distribution": payload.get("class_distribution"),
                    "weighted_class_distribution": payload.get(
                        "weighted_class_distribution"
                    ),
                    **_numeric_subset(
                        payload,
                        int_keys=("rows_raw", "rows_after_target_notna"),
                    ),
                    "decision_row_filter": payload.get("decision_row_filter"),
                }
            ),
            "feature_set": payload.get("feature_set"),
            "optimization": _drop_none_fields(
                {
                    "storage": payload.get("storage"),
                    "timeout_seconds": payload.get("timeout_seconds"),
                    "fold_recency_weighting": compact_fold_recency_weighting,
                    **_numeric_subset(
                        payload,
                        int_keys=(
                            "n_trials_requested",
                            "cv_folds",
                            "max_n_estimators",
                            "early_stopping_rounds",
                            "prune_report_every_n_iteration",
                        ),
                        float_keys=("walk_forward_test_to_train_ratio",),
                    ),
                    "reaction_profile_optuna_search_space": payload.get(
                        "reaction_profile_optuna_search_space"
                    ),
                    "optuna_seed_trial_count": int(
                        len(payload.get("optuna_seed_trial_params") or ())
                    ),
                }
            ),
            "lgbm_params": payload.get("lgbm_params"),
            "base_reaction_profile_fixed_grid": (
                build_editable_reaction_profile_config(base_cfg)
                if base_cfg is not None
                else None
            ),
            "artifacts": payload.get("artifacts"),
        }
    )


def make_lgbm_cv_params(feature_names=None):
    monotone_params = (
        make_lgbm_monotone_constraint_params(feature_names)
        if feature_names is not None
        else {}
    )
    params = {
        "objective": "binary",
        "metric": "None",
        "boosting_type": "gbdt",
        "device_type": LGBM_DEVICE_TYPE,
        "verbosity": LGBM_VERBOSITY,
        "num_threads": LGBM_NUM_THREADS,
        "max_bin": GPU_MAX_BIN_LIMIT,
        "num_iterations": MAX_N_ESTIMATORS,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
        "feature_pre_filter": True,
        **LGBM_DEFAULT_PARAMS,
        **monotone_params,
    }
    if is_lgbm_gpu_enabled():
        params["gpu_use_dp"] = LGBM_GPU_USE_DP
    return params


def is_lgbm_gpu_enabled():
    return str(LGBM_DEVICE_TYPE).strip().lower() == "gpu"


def build_filtered_training_arrays(
        open_np,
        high_np,
        low_np,
        close_np,
        keep_mask,
        y_filtered,
        sample_weight_filtered,
        normalized_rp_config,
):
    x_np, _ = build_reaction_profile_feature_matrix_from_arrays(
        open_=open_np,
        high=high_np,
        low=low_np,
        close=close_np,
        cfg=normalized_rp_config,
        keep_mask=keep_mask,
    )
    if is_lgbm_gpu_enabled():
        # LightGBM GPU can hard-crash on this Windows setup when fed the RP
        # matrix as float64, so keep the GPU training matrix explicitly float32.
        x_np = np.ascontiguousarray(x_np, dtype=np.float32)
    y_np = y_filtered
    sample_weight_np = sample_weight_filtered

    if x_np.shape[0] == 0:
        raise ValueError("No rows left in filtered training matrix.")
    if x_np.shape[1] == 0:
        raise ValueError("No features left after reaction profile build.")
    validate_sample_weight_array(sample_weight_np)

    y_unique = np.unique(y_np)
    if len(y_unique) < 2:
        raise ValueError(
            "Target has only one class after weight filtering. "
            f"Found classes={y_unique.tolist()}"
        )

    return x_np, y_np, sample_weight_np


def run_lightgbm_cv(
        x_np,
        y_np,
        sample_weight_np,
        folds,
        fold_indices,
        fold_weight_by_id,
        feature_names,
        trial=None,
        return_cvbooster=False,
):
    train_set = lgb.Dataset(
        data=x_np,
        label=y_np,
        weight=sample_weight_np,
        feature_name=list(feature_names),
        free_raw_data=True,
    )

    callbacks = [
        lgb.early_stopping(
            stopping_rounds=EARLY_STOPPING_ROUNDS,
            first_metric_only=True,
            verbose=False,
        )
    ]
    if trial is not None:
        callbacks.append(
            ObjectiveAlignedLightGBMPruningCallback(
                trial=trial,
                metric=CV_OBJECTIVE_BASE_METRIC,
                std_penalty=CV_STD_PENALTY,
                report_interval=PRUNE_REPORT_EVERY_N_ITER,
            )
        )

    need_cvbooster = bool(return_cvbooster) or is_nontrivial_fold_recency_weighting_enabled()
    cv_results = lgb.cv(
        params=make_lgbm_cv_params(feature_names=feature_names),
        train_set=train_set,
        folds=fold_indices,
        stratified=False,
        shuffle=False,
        feval=[
            make_lightgbm_binary_logloss_eval(CV_OBJECTIVE_BASE_METRIC),
            make_lightgbm_binary_balanced_accuracy_eval("balanced_accuracy"),
            make_lightgbm_binary_brier_eval("brier_score"),
        ],
        callbacks=callbacks,
        return_cvbooster=need_cvbooster,
        seed=SEED,
    )

    mean_series = np.asarray(
        cv_results[f"valid {CV_OBJECTIVE_BASE_METRIC}-mean"], dtype=np.float64
    )
    std_series = np.asarray(
        cv_results[f"valid {CV_OBJECTIVE_BASE_METRIC}-stdv"], dtype=np.float64
    )
    objective_series = (
        mean_series - (CV_STD_PENALTY * std_series)
        if CV_OBJECTIVE_IS_HIGHER_BETTER
        else mean_series + (CV_STD_PENALTY * std_series)
    )
    best_index = int(
        np.argmax(objective_series)
        if CV_OBJECTIVE_IS_HIGHER_BETTER
        else np.argmin(objective_series)
    )
    best_iteration = int(best_index + 1)

    result = {"best_iteration": best_iteration}
    if need_cvbooster:
        # LightGBM CV exposes only aggregated per-iteration metrics, so recency
        # weighting is applied by rescoring each fold at the chosen iteration.
        cvbooster = cv_results["cvbooster"]
        fold_scores = compute_cv_fold_scores_at_iteration(
            cvbooster=cvbooster,
            x_np=x_np,
            y_np=y_np,
            sample_weight_np=sample_weight_np,
            folds=folds,
            best_iteration=best_iteration,
        )
        result.update(
            summarize_cv_fold_scores(
                fold_scores=fold_scores,
                folds=folds,
                fold_weight_by_id=fold_weight_by_id,
                std_penalty=CV_STD_PENALTY,
            )
        )
        if return_cvbooster:
            result["cvbooster"] = cvbooster
    else:
        result.update(
            {
                f"cv_{CV_OBJECTIVE_BASE_METRIC}_mean": float(mean_series[best_index]),
                f"cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean": float(
                    mean_series[best_index]
                ),
                f"cv_{CV_OBJECTIVE_BASE_METRIC}_std": float(std_series[best_index]),
                "objective_base_value": float(mean_series[best_index]),
                "objective_value": float(objective_series[best_index]),
            }
        )
    return result


def get_best_successful_trial(study):
    successful_trials = []
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if trial.user_attrs.get("trial_status") != "ok":
            continue
        if trial.value is None or not np.isfinite(float(trial.value)):
            continue
        successful_trials.append(trial)

    if not successful_trials:
        raise RuntimeError(
            "No successful Optuna trials completed. "
            "All completed trials were pruned or ended with crash_penalty."
        )

    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        return max(successful_trials, key=lambda trial: float(trial.value))
    return min(successful_trials, key=lambda trial: float(trial.value))


def make_objective(
        base_data,
        folds,
        fold_indices,
        fold_weight_by_id,
        base_rp_config,
        search_space,
):
    def objective(trial):
        normalized_rp_config = None
        x_np = None

        try:
            normalized_rp_config = suggest_reaction_profile_config(
                trial=trial,
                base_config=base_rp_config,
                search_space=search_space,
            )
            x_np, y_np, sample_weight_np = build_filtered_training_arrays(
                open_np=base_data["open_np"],
                high_np=base_data["high_np"],
                low_np=base_data["low_np"],
                close_np=base_data["close_np"],
                keep_mask=base_data["keep_mask"],
                y_filtered=base_data["y_filtered"],
                sample_weight_filtered=base_data["sample_weight_filtered"],
                normalized_rp_config=normalized_rp_config,
            )
            cv_result = run_lightgbm_cv(
                x_np=x_np,
                y_np=y_np,
                sample_weight_np=sample_weight_np,
                folds=folds,
                fold_indices=fold_indices,
                fold_weight_by_id=fold_weight_by_id,
                feature_names=normalized_rp_config["feature_columns"],
                trial=trial,
                return_cvbooster=False,
            )

            trial.set_user_attr("trial_status", "ok")
            trial.set_user_attr(
                f"cv_{CV_OBJECTIVE_BASE_METRIC}_mean",
                cv_result[f"cv_{CV_OBJECTIVE_BASE_METRIC}_mean"],
            )
            trial.set_user_attr(
                f"cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean",
                cv_result[f"cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean"],
            )
            trial.set_user_attr(
                f"cv_{CV_OBJECTIVE_BASE_METRIC}_std",
                cv_result[f"cv_{CV_OBJECTIVE_BASE_METRIC}_std"],
            )
            trial.set_user_attr(
                "objective_base_value",
                cv_result["objective_base_value"],
            )
            trial.set_user_attr("best_iteration", cv_result["best_iteration"])
            trial.set_user_attr("feature_count", int(x_np.shape[1]))
            trial.set_user_attr(
                "config_signature",
                str(normalized_rp_config["config_signature"]),
            )
            return cv_result["objective_value"]
        except (lgb.basic.LightGBMError, OSError) as e:
            trial.set_user_attr("trial_status", "crash_penalty")
            trial.set_user_attr("crash_type", type(e).__name__)
            trial.set_user_attr("crash_message", str(e)[:1000])
            if normalized_rp_config is not None:
                trial.set_user_attr(
                    "config_signature",
                    str(normalized_rp_config["config_signature"]),
                )
            if x_np is not None:
                trial.set_user_attr("feature_count", int(x_np.shape[1]))
            return CRASH_PENALTY

    return objective


def run_optuna_optimization():
    optuna.logging.set_verbosity(optuna.logging.INFO)
    run_info = resolve_run_study_name(
        STUDY_NAME,
        default_prefix=DEFAULT_STUDY_NAME_PREFIX,
    )
    study_name = run_info["study_name"]
    study_name_source = run_info["study_name_source"]
    run_timestamp = run_info["run_timestamp"]
    best_result_path = make_timestamped_artifact_path(
        ARTIFACT_OUTPUT_DIR,
        stem=BEST_RESULT_STEM,
        suffix=".json",
        timestamp=run_timestamp,
    )
    trials_csv_path = make_timestamped_artifact_path(
        ARTIFACT_OUTPUT_DIR,
        stem=TRIALS_CSV_STEM,
        suffix=".csv",
        timestamp=run_timestamp,
    )

    dataset_settings = load_modeling_dataset_settings()
    base_rp_config = dataset_settings.get("reaction_profile_fixed_grid") or {}
    normalized_base_rp_config = normalize_reaction_profile_config(base_rp_config)
    monotone_constraint_summary = summarize_lgbm_monotone_constraints(
        normalized_base_rp_config["feature_columns"]
    )

    base_data = load_base_ohlcv_frame(BASE_DATA_PATH)
    base_df = base_data["df"]
    filtered_rows = int(base_data["row_filter_info"]["rows_after"])
    folds = make_walk_forward_folds(
        n_rows=filtered_rows,
        n_folds=CV_FOLDS,
        test_to_train_ratio=WF_TEST_TO_TRAIN_RATIO,
    )
    fold_indices = build_fold_indices(folds)
    fold_weight_by_id = build_fold_recency_weights(folds)

    if STORAGE.startswith("sqlite:///"):
        Path(STORAGE.replace("sqlite:///", "", 1)).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=optuna.exceptions.ExperimentalWarning,
            message=r"Argument ``multivariate`` is an experimental feature\..*",
        )
        sampler = optuna.samplers.TPESampler(
            seed=SEED,
            n_startup_trials=TPE_STARTUP_TRIALS,
            multivariate=True,
        )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=100,
        max_resource=MAX_N_ESTIMATORS,
        reduction_factor=3,
        bootstrap_count=2,
    )

    print(
        f"start optimize | base_rows={len(base_df)} filtered_rows={filtered_rows} "
        f"folds={len(fold_indices)} trials={N_TRIALS} timeout={TIMEOUT_SECONDS} "
        f"objective={resolve_cv_objective_name()} std_penalty={CV_STD_PENALTY:.4f} "
        f"lgbm_device={LGBM_DEVICE_TYPE} "
        f"study_name={study_name} study_name_source={study_name_source} "
        f"load_if_exists={LOAD_IF_EXISTS}"
    )
    print(
        "start optimize | "
        f"base_reaction_profile_feature_count={len(normalized_base_rp_config['feature_columns'])} "
        f"row_filter_min_weight={MIN_SAMPLE_WEIGHT:.2f} "
        f"bin_size={normalized_base_rp_config['bin_size']:.6f} "
        f"neighbor_bins={normalized_base_rp_config['neighbor_bins']:.6f}"
    )
    print(
        "start optimize | horizon_params="
        + "; ".join(
            (
                f"{horizon_name}:"
                f"local_window={normalized_base_rp_config['horizons'][horizon_name]['local_window']},"
                f"half_life={normalized_base_rp_config['horizons'][horizon_name]['half_life_candles']}"
            )
            for horizon_name in normalized_base_rp_config["horizon_names"]
        )
    )
    print(
        "start optimize | "
        f"search_params={sorted(REACTION_PROFILE_OPTUNA_SEARCH_SPACE)} "
        f"seed_trials_configured={len(OPTUNA_SEED_TRIAL_PARAMS)}"
    )
    print(
        f"fold weighting | enabled={bool(ENABLE_FOLD_RECENCY_WEIGHTING)} "
        f"active={is_nontrivial_fold_recency_weighting_enabled()} "
        f"mode={FOLD_RECENCY_WEIGHTING_MODE} "
        f"min={float(FOLD_RECENCY_WEIGHT_MIN):.4f} "
        f"max={float(FOLD_RECENCY_WEIGHT_MAX):.4f} "
        f"std=unweighted"
    )
    print(
        "monotone constraints | "
        f"{format_lgbm_monotone_constraint_summary(monotone_constraint_summary)}"
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=STORAGE,
        direction="maximize" if CV_OBJECTIVE_IS_HIGHER_BETTER else "minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=LOAD_IF_EXISTS,
    )
    enqueue_seed_trials(
        study=study,
        seed_trial_params=OPTUNA_SEED_TRIAL_PARAMS,
        normalized_base=normalized_base_rp_config,
        search_space=REACTION_PROFILE_OPTUNA_SEARCH_SPACE,
    )

    study.optimize(
        make_objective(
            base_data=base_data,
            folds=folds,
            fold_indices=fold_indices,
            fold_weight_by_id=fold_weight_by_id,
            base_rp_config=base_rp_config,
            search_space=REACTION_PROFILE_OPTUNA_SEARCH_SPACE,
        ),
        n_trials=N_TRIALS,
        timeout=TIMEOUT_SECONDS,
        n_jobs=OPTUNA_OPTIMIZE_N_JOBS,
        gc_after_trial=True,
        show_progress_bar=True,
        catch=(lgb.basic.LightGBMError, OSError),
    )

    best_trial = get_best_successful_trial(study)
    best_normalized_rp_config = build_reaction_profile_config_from_params(
        base_config=base_rp_config,
        params=best_trial.params,
    )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_data_path": str(BASE_DATA_PATH),
        "target_col": TARGET_COL,
        "sample_weight_col": TARGET_WEIGHT_COL,
        "sample_weight": {
            "used": True,
            "source": "dataset_column",
            **base_data["sample_weight_summary"],
        },
        "class_distribution": base_data["class_distribution"],
        "weighted_class_distribution": base_data["weighted_class_distribution"],
        "rows_raw": int(base_data["raw_rows"]),
        "rows_after_target_notna": int(base_data["rows_after_target_notna"]),
        "decision_row_filter": base_data["row_filter_info"],
        "feature_set": {
            "mode": "reaction_profile_fixed_grid_only",
            "base_feature_count": len(normalized_base_rp_config["feature_columns"]),
            "best_feature_count": len(best_normalized_rp_config["feature_columns"]),
        },
        "study_name": study_name,
        "study_name_source": study_name_source,
        "storage": STORAGE,
        "run_timestamp_utc": run_timestamp,
        "n_trials_requested": int(N_TRIALS),
        "timeout_seconds": TIMEOUT_SECONDS,
        "cv_folds": CV_FOLDS,
        "walk_forward_test_to_train_ratio": WF_TEST_TO_TRAIN_RATIO,
        "fold_recency_weighting": {
            "enabled": bool(ENABLE_FOLD_RECENCY_WEIGHTING),
            "active": bool(is_nontrivial_fold_recency_weighting_enabled()),
            "mode": str(FOLD_RECENCY_WEIGHTING_MODE),
            "min_weight": float(FOLD_RECENCY_WEIGHT_MIN),
            "max_weight": float(FOLD_RECENCY_WEIGHT_MAX),
            "std_score_aggregation": "unweighted",
            "fold_weights": fold_weight_items_for_summary(folds, fold_weight_by_id),
        },
        "max_n_estimators": MAX_N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "prune_report_every_n_iteration": PRUNE_REPORT_EVERY_N_ITER,
        "cv_objective": {
            "name": resolve_cv_objective_name(),
            "base_metric": CV_OBJECTIVE_BASE_METRIC,
            "base_score": cv_objective_base_score_label(),
            "aggregation": (
                f"{cv_objective_base_score_label()} "
                f"{'-' if CV_OBJECTIVE_IS_HIGHER_BETTER else '+'} "
                "std_penalty * cv_std"
            ),
            "std_penalty": float(CV_STD_PENALTY),
        },
        "lgbm_params": make_lgbm_cv_params(
            feature_names=normalized_base_rp_config["feature_columns"]
        ),
        "monotone_constraints": monotone_constraint_summary,
        "base_reaction_profile_fixed_grid": normalized_base_rp_config,
        "reaction_profile_optuna_search_space": REACTION_PROFILE_OPTUNA_SEARCH_SPACE,
        "optuna_seed_trial_params": OPTUNA_SEED_TRIAL_PARAMS,
        "best_trial": {
            "number": int(best_trial.number),
            f"objective_{resolve_cv_objective_name()}": float(best_trial.value),
            f"cv_{CV_OBJECTIVE_BASE_METRIC}_mean": float(
                best_trial.user_attrs.get(f"cv_{CV_OBJECTIVE_BASE_METRIC}_mean")
            ),
            f"cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean": float(
                best_trial.user_attrs.get(f"cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean")
            ),
            f"cv_{CV_OBJECTIVE_BASE_METRIC}_std": float(
                best_trial.user_attrs.get(f"cv_{CV_OBJECTIVE_BASE_METRIC}_std")
            ),
            "best_iteration": int(best_trial.user_attrs.get("best_iteration")),
            "feature_count": int(best_trial.user_attrs.get("feature_count")),
            "params": best_trial.params,
            "reaction_profile_fixed_grid": best_normalized_rp_config,
        },
        "artifacts": {
            "best_result_path": str(best_result_path),
            "trials_csv_path": str(trials_csv_path),
        },
    }
    payload = compact_reaction_profile_artifact_payload(payload)

    best_result_path.parent.mkdir(parents=True, exist_ok=True)
    best_result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    trials_csv_path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(trials_csv_path, index=False)

    print(
        f"best trial | number={best_trial.number} objective={best_trial.value:.8f} "
        f"{CV_OBJECTIVE_BASE_METRIC}_mean={float(best_trial.user_attrs.get(f'cv_{CV_OBJECTIVE_BASE_METRIC}_mean')):.8f} "
        f"{CV_OBJECTIVE_BASE_METRIC}_weighted_mean={float(best_trial.user_attrs.get(f'cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean')):.8f} "
        f"{CV_OBJECTIVE_BASE_METRIC}_std={float(best_trial.user_attrs.get(f'cv_{CV_OBJECTIVE_BASE_METRIC}_std')):.8f} "
        f"best_iteration={int(best_trial.user_attrs.get('best_iteration'))} "
        f"feature_count={int(best_trial.user_attrs.get('feature_count'))}"
    )
    print(f"saved best payload -> {best_result_path}")
    print(f"saved trials csv -> {trials_csv_path}")


def main():
    crash_log_file = enable_native_crash_log()
    try:
        run_optuna_optimization()
    finally:
        faulthandler.disable()
        crash_log_file.close()


if __name__ == "__main__":
    main()
