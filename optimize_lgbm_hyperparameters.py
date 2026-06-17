import gc
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from features.candle_features import RAW_OHLCV_COLS
from features.volume_profile_fixed_range import validate_volume_profile_feature_columns
from train_lgbm import (
    CV_FOLDS as FINAL_CV_FOLDS,
    WF_TEST_TO_TRAIN_RATIO as FINAL_WF_TEST_TO_TRAIN_RATIO,
    evaluate_walk_forward_variant,
    format_lgbm_monotone_constraint_summary,
    load_walk_forward_training_frame,
    make_lgbm_monotone_constraint_params,
    make_walk_forward_folds as make_final_walk_forward_folds,
    summarize_lgbm_monotone_constraints,
)
from utils.data import (
    TARGET_WEIGHT_COL,
    summarize_target_weights,
)
from utils.data import (
    load_excluded_feature_names_from_settings,
    load_feature_subset_from_settings,
    load_modeling_dataset_settings,
    resolve_modeling_float_dtype,
    resolve_modeling_float_dtype_name,
    resolve_modeling_dataset_output_paths,
    summarize_feature_subset,
    validate_parquet_magic_bytes,
)
from utils.metrics import (
    make_lightgbm_binary_balanced_accuracy_eval,
    make_lightgbm_binary_brier_eval,
    make_lightgbm_binary_logloss_eval,
    weighted_balanced_accuracy_score,
    weighted_binary_logloss,
)
from utils.optuna import (
    make_timestamped_artifact_path,
    resolve_existing_study_name,
    resolve_run_study_name,
)
from utils.project_config import active_asset_path

TARGET_COL = "target_5m_candle_up"

CV_FOLDS = 10
WF_TEST_TO_TRAIN_RATIO = FINAL_WF_TEST_TO_TRAIN_RATIO
ENABLE_FOLD_RECENCY_WEIGHTING = True
FOLD_RECENCY_WEIGHTING_MODE = "linear"
FOLD_RECENCY_WEIGHT_MIN = 1.0
FOLD_RECENCY_WEIGHT_MAX = 1.5

MAX_N_ESTIMATORS = 4000
EARLY_STOPPING_ROUNDS = 20
PRUNE_REPORT_EVERY_N_ITER = 5

SEED = 37
LGBM_NUM_THREADS = 17
OPTUNA_OPTIMIZE_N_JOBS = 1
LGBM_DEVICE_TYPE = "gpu"
LGBM_VERBOSITY = -1
GPU_MAX_BIN_LIMIT = 63

LGBM_OPTUNA_SEARCH_SPACE = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 0.5, "log": True},
    "num_leaves": {"type": "int", "low": 16, "high": 384},
    "min_data_in_leaf": {"type": "int", "low": 2, "high": 65_536, "log": False},
    "max_depth": {"type": "int", "low": 9, "high": 256},
    "feature_fraction": {"type": "float", "low": 0.01, "high": 1.0},
    "bagging_fraction": {"type": "float", "low": 0.01, "high": 1.0},
    "bagging_freq": {"type": "int", "low": 0, "high": 32},
    "lambda_l2": {"type": "float", "low": 1e-6, "high": 256.0, "log": True},
    "lambda_l1": {"type": "float", "low": 1e-6, "high": 256.0, "log": True},
    "min_sum_hessian_in_leaf": {
        "type": "float",
        "low": 1e-5,
        "high": 100.0,
        "log": True,
    },
    "min_gain_to_split": {"type": "float", "low": 0.0, "high": 10.0},
    "feature_fraction_bynode": {"type": "float", "low": 0.01, "high": 1.0},
    "path_smooth": {"type": "float", "low": 1e-6, "high": 256.0, "log": True},
    "extra_trees": {"type": "categorical", "choices": [True, False]},
    "monotone_constraints_method": {
        "type": "categorical",
        "choices": ["basic", "intermediate", "advanced"],
    },
    "monotone_penalty": {"type": "float", "low": 0.0, "high": 10.0},
}

OPTUNA_SEED_TRIAL_DEFAULT_PARAMS = {
    "monotone_constraints_method": "basic",
    "monotone_penalty": 0.0,
}

# Seed trials are injected before optimization starts.
OPTUNA_SEED_TRIAL_PARAMS = [
    {
        "learning_rate": 0.02082784518014535,
        "num_leaves": 37,
        "min_data_in_leaf": 103,
        "max_depth": 169,
        "feature_fraction": 0.37752257586311444,
        "bagging_fraction": 0.6274488270891571,
        "bagging_freq": 23,
        "lambda_l2": 53.12614038557139,
        "lambda_l1": 10.31812331923178,
        "min_sum_hessian_in_leaf": 0.0061350132104521764,
        "min_gain_to_split": 0.07525941090726794,
        "feature_fraction_bynode": 0.7170577314073263,
        "path_smooth": 1.124852812228145,
        "extra_trees": False,
    },
    {
        "learning_rate": 0.040166858620227654,
        "num_leaves": 97,
        "min_data_in_leaf": 34,
        "max_depth": 219,
        "feature_fraction": 0.49152697810227164,
        "bagging_fraction": 0.6687218276054827,
        "bagging_freq": 22,
        "lambda_l2": 34.27447477551414,
        "lambda_l1": 16.29642005808772,
        "min_sum_hessian_in_leaf": 4.6370061892768994,
        "min_gain_to_split": 0.6705780188504484,
        "feature_fraction_bynode": 0.5237714962797345,
        "path_smooth": 1.8920129570355595,
        "extra_trees": False
    },
    {
        "learning_rate": 0.0232055790116649,
        "num_leaves": 153,
        "min_data_in_leaf": 16,
        "max_depth": 204,
        "feature_fraction": 0.4431937228828896,
        "bagging_fraction": 0.8322310712611087,
        "bagging_freq": 20,
        "lambda_l2": 28.76389554546722,
        "lambda_l1": 7.383529216274445,
        "min_sum_hessian_in_leaf": 0.006286030943849801,
        "min_gain_to_split": 0.16999148557266552,
        "feature_fraction_bynode": 0.8624029035837287,
        "path_smooth": 19.83320951468675,
        "extra_trees": False
    },
    {
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
        "extra_trees": False
    },
    {
        "learning_rate": 0.014324759771509326,
        "num_leaves": 219,
        "min_data_in_leaf": 134,
        "max_depth": 166,
        "feature_fraction": 0.9004727141904951,
        "bagging_fraction": 0.8582425675795976,
        "bagging_freq": 16,
        "lambda_l2": 96.08153533203745,
        "lambda_l1": 1.3495783734260116,
        "min_sum_hessian_in_leaf": 0.005819496553267834,
        "min_gain_to_split": 0.11935855858916433,
        "feature_fraction_bynode": 0.4278157008307606,
        "path_smooth": 91.00059177549922,
        "extra_trees": False
    },
    {
        "learning_rate": 0.03601725310962062,
        "num_leaves": 232,
        "min_data_in_leaf": 1284,
        "max_depth": 238,
        "feature_fraction": 0.6199037025636075,
        "bagging_fraction": 0.823971026512396,
        "bagging_freq": 25,
        "lambda_l2": 60.953190284251875,
        "lambda_l1": 29.676819923146972,
        "min_sum_hessian_in_leaf": 0.12748346400557883,
        "min_gain_to_split": 0.4087732850139867,
        "feature_fraction_bynode": 0.35610397169696534,
        "path_smooth": 34.47657007831167,
        "extra_trees": False,
        "monotone_constraints_method": "advanced",
        "monotone_penalty": 3.9096906129788724
    },
    {
        "learning_rate": 0.04640319526767822,
        "num_leaves": 268,
        "min_data_in_leaf": 5306,
        "max_depth": 13,
        "feature_fraction": 0.7752615284806654,
        "bagging_fraction": 0.7083041610957306,
        "bagging_freq": 12,
        "lambda_l2": 99.45821431148829,
        "lambda_l1": 8.798400892318645,
        "min_sum_hessian_in_leaf": 1.0877816428524054e-05,
        "min_gain_to_split": 0.4520600669558363,
        "feature_fraction_bynode": 0.9773524734598936,
        "path_smooth": 54.99959785886466,
        "extra_trees": False,
        "monotone_constraints_method": "intermediate",
        "monotone_penalty": 2.00943387231227
    },
    {
        "learning_rate": 0.018911429915801383,
        "num_leaves": 249,
        "min_data_in_leaf": 33,
        "max_depth": 182,
        "feature_fraction": 0.8947859540147098,
        "bagging_fraction": 0.7422077424372853,
        "bagging_freq": 16,
        "lambda_l2": 98.80079501643887,
        "lambda_l1": 12.85240865670112,
        "min_sum_hessian_in_leaf": 7.238167505794533e-05,
        "min_gain_to_split": 0.1025385907005768,
        "feature_fraction_bynode": 0.6666400250804688,
        "path_smooth": 78.42447945281934,
        "extra_trees": False,
        "monotone_constraints_method": "basic",
        "monotone_penalty": 1.4043678835536488
    },
    {
        "learning_rate": 0.0065094262862249175,
        "num_leaves": 215,
        "min_data_in_leaf": 97,
        "max_depth": 195,
        "feature_fraction": 0.8985980039678766,
        "bagging_fraction": 0.7035525517200346,
        "bagging_freq": 3,
        "lambda_l2": 84.66552493332907,
        "lambda_l1": 3.8623890143572983,
        "min_sum_hessian_in_leaf": 0.0006044966264154735,
        "min_gain_to_split": 0.23134325634193995,
        "feature_fraction_bynode": 0.3141763321857879,
        "path_smooth": 60.87851990364099,
        "extra_trees": False,
        "monotone_constraints_method": "basic",
        "monotone_penalty": 0.648819578510189
    },
    {
        "learning_rate": 0.004387225197481959,
        "num_leaves": 133,
        "min_data_in_leaf": 720,
        "max_depth": 206,
        "feature_fraction": 0.8974595250202884,
        "bagging_fraction": 0.79292032203349,
        "bagging_freq": 20,
        "lambda_l2": 11.605081849728947,
        "lambda_l1": 6.882639835280196,
        "min_sum_hessian_in_leaf": 7.72484450824706,
        "min_gain_to_split": 0.3243665400449587,
        "feature_fraction_bynode": 0.42688754866797657,
        "path_smooth": 33.372266902988066,
        "extra_trees": False,
        "monotone_constraints_method": "basic",
        "monotone_penalty": 1.2685005945295802
    },
    {
      "learning_rate": 0.003395029882596675,
      "num_leaves": 171,
      "min_data_in_leaf": 5661,
      "max_depth": 176,
      "feature_fraction": 0.672056207538789,
      "bagging_fraction": 0.8653821128720749,
      "bagging_freq": 19,
      "lambda_l2": 10.417085682028684,
      "lambda_l1": 1.3534862331319992,
      "min_sum_hessian_in_leaf": 0.03007168649377586,
      "min_gain_to_split": 0.5171112104391531,
      "feature_fraction_bynode": 0.7173489958572304,
      "path_smooth": 30.034997255070344,
      "extra_trees": False,
      "monotone_constraints_method": "basic",
      "monotone_penalty": 0.7275790716785062
    }
]

N_TRIALS = 15
TIMEOUT_SECONDS = None
CV_OBJECTIVE_BASE_METRIC = "binary_logloss"
EARLY_STOPPING_METRIC = CV_OBJECTIVE_BASE_METRIC
CV_OBJECTIVE_IS_HIGHER_BETTER = False
CV_STD_PENALTY = 0.5
RECHECK_OBJECTIVE_BASE_METRIC = "binary_logloss"
RECHECK_OBJECTIVE_IS_HIGHER_BETTER = False
RECHECK_STD_PENALTY = 0.75
DEFAULT_STUDY_NAME_PREFIX = "lgbm_generic_binary_logloss_mean_std"
# Leave empty for a fresh timestamped study. Set only to continue an existing one.
STUDY_NAME = None
STORAGE = (
        "sqlite:///"
        + active_asset_path(
    "data/optuna/databases/{asset}/lgbm_generic_tpe_hyperband_gpu.db"
).as_posix()
)
LOAD_IF_EXISTS = True
ARTIFACT_OUTPUT_DIR = active_asset_path("data/optuna/lgbm/{asset}")
BEST_RESULT_STEM = "lgbm_generic_optuna_best_mean_std"
TRIALS_CSV_STEM = "lgbm_generic_optuna_trials_mean_std"
RUN_MODE = "optimize"  # "optimize" or "recheck-topn"
# Set only when running "recheck-topn". Falls back to STUDY_NAME when provided.
RECHECK_STUDY_NAME = None
RECHECK_STORAGE = STORAGE
TOP_TRIALS_RECHECK_N = 20
TOP_TRIALS_RECHECK_OUTPUT_DIR = active_asset_path("data/optuna/lgbm/{asset}/recheck")
TOP_TRIALS_RECHECK_OUTPUT_JSON_PATH = None
TOP_TRIALS_RECHECK_OUTPUT_CSV_PATH = None

PRUNER_MIN_RESOURCE = 100
PRUNER_REDUCTION_FACTOR = 3
PRUNER_BOOTSTRAP_COUNT = 0
TPE_STARTUP_TRIALS = int(N_TRIALS * 0.1)

FEATURE_HORIZON_RE = re.compile(r"(?:_fit_|_target_)(\d+)m(?:_ahead_ret)?")
TARGET_HORIZON_RE = re.compile(r"target_(\d+)m")


def make_walk_forward_folds(
        n_rows,
        n_folds,
        test_to_train_ratio,
):
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


def load_generic_training_data(
        data_path,
        feature_subset=None,
        excluded_features=None,
        float_dtype=np.float32,
):
    excluded_feature_names = (
        tuple(excluded_features["features"]) if excluded_features else tuple()
    )
    excluded_feature_set = set(excluded_feature_names)
    selected_feature_columns = (
        list(feature_subset["features"]) if feature_subset else None
    )
    parquet_columns = None
    if selected_feature_columns is not None:
        parquet_columns = list(
            dict.fromkeys([TARGET_COL, TARGET_WEIGHT_COL, *selected_feature_columns])
        )

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    validate_parquet_magic_bytes(data_path)

    print(f"load data | path={data_path}")
    try:
        df = pd.read_parquet(data_path, columns=parquet_columns)
    except Exception as exc:
        if parquet_columns is None:
            raise
        preview = ", ".join(parquet_columns[:10])
        raise ValueError(
            "Dataset is missing columns required by optimize_lgbm_optuna.py. "
            "Rebuild it with create_modeling_dataset.py for the active feature subset. "
            f"Requested_count={len(parquet_columns)} preview=[{preview}]"
        ) from exc
    raw_rows, raw_cols = df.shape
    print(f"load data | raw_rows={raw_rows} raw_cols={raw_cols}")

    df = df[df[TARGET_COL].notna()]
    rows_after_target_notna = len(df)
    if rows_after_target_notna == 0:
        raise ValueError("No rows left after TARGET_COL non-null filtering.")
    if TARGET_WEIGHT_COL not in df.columns:
        raise ValueError(
            f"Dataset is missing required sample weight column '{TARGET_WEIGHT_COL}'. "
            "Rebuild it with create_modeling_dataset.py."
        )
    sample_weight_full = pd.to_numeric(df[TARGET_WEIGHT_COL], errors="raise").to_numpy(
        dtype=float_dtype,
        copy=False,
    )
    sample_weight_source = "dataset_column"
    print(f"load data | rows_after_target_notna={rows_after_target_notna}")

    dropped_raw_ohlcv_features = [col for col in RAW_OHLCV_COLS if col in df.columns]
    x = df.drop(
        columns=[TARGET_COL, TARGET_WEIGHT_COL, *dropped_raw_ohlcv_features],
        errors="ignore",
    )
    x = x.select_dtypes(include=[np.number])
    if selected_feature_columns is not None:
        missing_selected_features = [
            col for col in selected_feature_columns if col not in x.columns
        ]
        if missing_selected_features:
            preview = ", ".join(missing_selected_features[:10])
            raise ValueError(
                "Dataset is missing configured subset features for optimization. "
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
            "load data | exclusions "
            f"dropped={len(excluded_present_features)} "
            f"missing_requested={len(excluded_missing_features)}"
        )
    if x.shape[1] == 0:
        raise ValueError("No numeric feature columns left after preprocessing.")
    validate_volume_profile_feature_columns(
        x.columns,
        source_label=f"optimization dataset features at {data_path}",
    )
    print(
        f"load data | numeric_features={x.shape[1]} "
        f"dropped_raw_ohlcv={len(dropped_raw_ohlcv_features)}"
    )

    feature_cols = x.columns.tolist()
    feature_horizons = [
        int(m.group(1))
        for col in feature_cols
        for m in [FEATURE_HORIZON_RE.search(col)]
        if m is not None
    ]
    max_feature_horizon = max(feature_horizons) if feature_horizons else 0

    target_horizon_match = TARGET_HORIZON_RE.search(TARGET_COL)
    target_horizon = int(target_horizon_match.group(1)) if target_horizon_match else 0

    x_np_full = x.to_numpy(dtype=float_dtype, copy=False)
    invalid_full = ~np.isfinite(x_np_full)
    n_rows, n_features = x_np_full.shape
    n_invalid_full = int(invalid_full.sum())

    max_leading_invalid = 0
    max_trailing_invalid = 0
    for j in range(n_features):
        col_invalid = invalid_full[:, j]
        if col_invalid[0]:
            if col_invalid.all():
                leading = n_rows
            else:
                leading = int(np.argmax(~col_invalid))
            if leading > max_leading_invalid:
                max_leading_invalid = leading

        if col_invalid[-1]:
            rev_invalid = col_invalid[::-1]
            if rev_invalid.all():
                trailing = n_rows
            else:
                trailing = int(np.argmax(~rev_invalid))
            if trailing > max_trailing_invalid:
                max_trailing_invalid = trailing

    head_trim = max_feature_horizon
    tail_trim = max(max_feature_horizon, target_horizon)
    end_idx = n_rows - tail_trim if tail_trim > 0 else n_rows
    if end_idx <= head_trim:
        raise ValueError(
            "No rows left after horizon trim. "
            f"rows={n_rows} head_trim={head_trim} tail_trim={tail_trim}"
        )

    x_np = np.asarray(x_np_full[head_trim:end_idx], dtype=float_dtype)
    finite_by_col = np.isfinite(x_np).any(axis=0)
    dropped_all_invalid_feature_names = []
    if not finite_by_col.all():
        dropped_all_invalid_feature_names = [
            feature_cols[j] for j in range(n_features) if not finite_by_col[j]
        ]
        x_np = x_np[:, finite_by_col]
        feature_cols = [
            feature_cols[j] for j in range(n_features) if bool(finite_by_col[j])
        ]
    if x_np.shape[1] == 0:
        raise ValueError(
            "No usable features left after dropping fully invalid columns."
        )

    invalid = ~np.isfinite(x_np)
    invalid_after_trim = int(invalid.sum())
    x_np = np.where(np.isinf(x_np), np.nan, x_np).astype(float_dtype, copy=False)
    nan_after_trim = int(np.isnan(x_np).sum())

    y_np_full = df[TARGET_COL].to_numpy(dtype=float_dtype, copy=False)
    y_np = np.asarray(y_np_full[head_trim:end_idx], dtype=float_dtype)
    sample_weight_np = np.asarray(
        sample_weight_full[head_trim:end_idx], dtype=float_dtype
    )

    print(
        f"data trim | head={head_trim} tail={tail_trim} "
        f"lead_invalid={max_leading_invalid} trail_invalid={max_trailing_invalid} "
        f"max_feature_horizon={max_feature_horizon} target_horizon={target_horizon}"
    )
    if dropped_all_invalid_feature_names:
        preview = ", ".join(dropped_all_invalid_feature_names[:5])
        print(
            f"load data | dropped_all_invalid_features={len(dropped_all_invalid_feature_names)} "
            f"preview=[{preview}]"
        )
    print(
        f"load data | invalid_full={n_invalid_full} invalid_after_trim={invalid_after_trim} "
        f"nan_after_trim={nan_after_trim}"
    )
    print(
        f"load data | final_rows={x_np.shape[0]} features={x_np.shape[1]} "
        f"dtypes(x/y)=({x_np.dtype}/{y_np.dtype})"
    )
    if x_np.shape[0] == 0:
        raise ValueError("No rows left in training matrix after preprocessing.")
    if sample_weight_np.shape[0] != x_np.shape[0]:
        raise ValueError(
            "Sample weights length mismatch after preprocessing: "
            f"{sample_weight_np.shape[0]} != {x_np.shape[0]}"
        )
    if not np.isfinite(sample_weight_np).all():
        raise ValueError("Sample weights contain non-finite values.")
    if np.any(sample_weight_np <= 0.0):
        raise ValueError("Sample weights must be strictly positive.")

    y_unique = np.unique(y_np)
    if len(y_unique) < 2:
        raise ValueError(
            "Target has only one class after preprocessing. "
            f"Found classes={y_unique.tolist()}"
        )

    return (
        x_np,
        y_np,
        sample_weight_np,
        rows_after_target_notna,
        sample_weight_source,
        summarize_target_weights(sample_weight_np),
        tuple(feature_cols),
    )


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


def objective_base_score_label(base_metric):
    if is_nontrivial_fold_recency_weighting_enabled():
        return f"{base_metric}_weighted_mean"
    return f"{base_metric}_mean"


def resolve_cv_objective_name():
    suffix = "minus_std_penalty" if CV_OBJECTIVE_IS_HIGHER_BETTER else "plus_std_penalty"
    return f"{objective_base_score_label(CV_OBJECTIVE_BASE_METRIC)}_{suffix}"


def resolve_recheck_objective_name():
    suffix = (
        "minus_std_penalty"
        if RECHECK_OBJECTIVE_IS_HIGHER_BETTER
        else "plus_std_penalty"
    )
    return (
        f"{objective_base_score_label(RECHECK_OBJECTIVE_BASE_METRIC)}_{suffix}"
    )


def objective_aggregation_description(base_metric):
    is_higher_better = base_metric in {"accuracy", "balanced_accuracy", "precision", "recall", "f1"}
    operator = "-" if is_higher_better else "+"
    return (
        f"cv_{objective_base_score_label(base_metric)} {operator} std_penalty * "
        f"cv_{base_metric}_std"
    )


def score_cv_objective_metric(y_true, y_pred_proba, sample_weight):
    if CV_OBJECTIVE_BASE_METRIC == "balanced_accuracy":
        return float(
            weighted_balanced_accuracy_score(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                sample_weight=sample_weight,
            )
        )
    if CV_OBJECTIVE_BASE_METRIC == "binary_logloss":
        return float(
            weighted_binary_logloss(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                sample_weight=sample_weight,
            )
        )
    raise ValueError(
        "Per-fold rescoring supports CV_OBJECTIVE_BASE_METRIC in "
        "{'balanced_accuracy', 'binary_logloss'}."
    )


def summarize_cv_fold_scores(
        fold_scores,
        folds,
        fold_weight_by_id,
        *,
        base_metric,
        higher_is_better,
        std_penalty,
):
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
        f"cv_{base_metric}_mean": mean_score,
        f"cv_{base_metric}_weighted_mean": weighted_mean_score,
        f"cv_{base_metric}_std": std_score,
        "objective_base_value": float(objective_base_value),
        "objective_value": combine_metric_mean_std(
            objective_base_value,
            std_score,
            higher_is_better=higher_is_better,
            std_penalty=std_penalty,
        ),
    }


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


def summarize_cv_result_metric(
        cv_result,
        folds,
        fold_weight_by_id,
        *,
        base_metric,
        higher_is_better,
        std_penalty,
):
    fold_scores = [
        float(fold["metrics"][base_metric])
        for fold in cv_result["folds"]
    ]
    return summarize_cv_fold_scores(
        fold_scores=fold_scores,
        folds=folds,
        fold_weight_by_id=fold_weight_by_id,
        base_metric=base_metric,
        higher_is_better=higher_is_better,
        std_penalty=std_penalty,
    )


def combine_metric_mean_std(mean_value, std_value, *, higher_is_better, std_penalty):
    if bool(higher_is_better):
        return float(mean_value - (float(std_penalty) * float(std_value)))
    return float(mean_value + (float(std_penalty) * float(std_value)))


def objective_study_direction(*, higher_is_better):
    return "maximize" if bool(higher_is_better) else "minimize"


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
        if is_cv:
            # LightGBM 4.6.0 reports CV metrics under "valid".
            # Accept "cv_agg" as well for compatibility with older assumptions.
            target_valid_names = ("cv_agg", "valid")
        else:
            target_valid_names = (self._valid_name,)

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
                    "in terms of study directions. Please specify a metric to be "
                    "maximized for ObjectiveAlignedLightGBMPruningCallback."
                )
        else:
            if self._trial.study.direction != optuna.study.StudyDirection.MINIMIZE:
                raise ValueError(
                    "The intermediate values are inconsistent with the objective values "
                    "in terms of study directions. Please specify a metric to be "
                    "minimized for ObjectiveAlignedLightGBMPruningCallback."
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


def validate_optuna_search_spec(name, spec):
    if not isinstance(spec, dict):
        raise ValueError(f"Search space spec for {name!r} must be a dict.")

    spec_type = str(spec.get("type", "")).strip().lower()
    if spec_type not in {"int", "float", "categorical"}:
        raise ValueError(
            f"Search space spec for {name!r} must define type='int', 'float', or 'categorical'."
        )

    if spec_type == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, (list, tuple)) or len(choices) == 0:
            raise ValueError(
                f"Categorical search space spec for {name!r} must define non-empty choices."
            )
        return

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


def validate_lgbm_search_space(search_space):
    for name, spec in search_space.items():
        validate_optuna_search_spec(name, spec)


def suggest_value_from_spec(trial, name, spec):
    spec_type = str(spec["type"]).strip().lower()
    if spec_type == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))

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


def suggest_lgbm_hyperparams(trial, search_space):
    validate_lgbm_search_space(search_space)
    return {
        name: suggest_value_from_spec(trial, name, spec)
        for name, spec in search_space.items()
    }


def make_seed_trial_params(seed_params, search_space):
    resolved = {
        name: value
        for name, value in OPTUNA_SEED_TRIAL_DEFAULT_PARAMS.items()
        if name in search_space
    }
    resolved.update(seed_params)
    return resolved


def validate_seed_trial_params(seed_params, search_space):
    unknown_names = sorted(set(seed_params) - set(search_space))
    if unknown_names:
        raise ValueError(
            f"Seed trial contains params not present in search space: {unknown_names}"
        )

    for name, value in seed_params.items():
        spec = search_space[name]
        spec_type = str(spec["type"]).strip().lower()
        if spec_type == "categorical":
            choices = list(spec["choices"])
            if not any(value == choice for choice in choices):
                raise ValueError(
                    f"Seed trial param {name!r}={value!r} is not in choices={choices!r}."
                )
            continue

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


def enqueue_seed_trials(study, seed_trial_params, search_space):
    validate_lgbm_search_space(search_space)

    for seed_index, params in enumerate(seed_trial_params, start=1):
        params = make_seed_trial_params(params, search_space)
        validate_seed_trial_params(params, search_space)
        study.enqueue_trial(
            params=params,
            user_attrs={"seed_trial_index": int(seed_index)},
            skip_if_exists=True,
        )


def make_objective(
        train_set,
        feature_names,
        x_np,
        y_np,
        sample_weight_np,
        folds,
        fold_indices,
        fold_weight_by_id,
        search_space,
):
    def objective(trial):
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
            "feature_pre_filter": False,
            "gpu_use_dp": False,
            **suggest_lgbm_hyperparams(trial, search_space),
            **make_lgbm_monotone_constraint_params(feature_names),
        }

        need_cvbooster = is_nontrivial_fold_recency_weighting_enabled()
        cv_results = lgb.cv(
            params=params,
            train_set=train_set,
            folds=fold_indices,
            stratified=False,
            shuffle=False,
            feval=[
                make_lightgbm_binary_logloss_eval(CV_OBJECTIVE_BASE_METRIC),
                make_lightgbm_binary_balanced_accuracy_eval("balanced_accuracy"),
                make_lightgbm_binary_brier_eval("brier_score"),
            ],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=EARLY_STOPPING_ROUNDS,
                    first_metric_only=True,
                    verbose=True,
                ),
                ObjectiveAlignedLightGBMPruningCallback(
                    trial=trial,
                    metric=CV_OBJECTIVE_BASE_METRIC,
                    std_penalty=CV_STD_PENALTY,
                    report_interval=PRUNE_REPORT_EVERY_N_ITER,
                ),
            ],
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
        best_iteration = best_index + 1
        if need_cvbooster:
            fold_score_summary = summarize_cv_fold_scores(
                fold_scores=compute_cv_fold_scores_at_iteration(
                    cvbooster=cv_results["cvbooster"],
                    x_np=x_np,
                    y_np=y_np,
                    sample_weight_np=sample_weight_np,
                    folds=folds,
                    best_iteration=best_iteration,
                ),
                folds=folds,
                fold_weight_by_id=fold_weight_by_id,
                base_metric=CV_OBJECTIVE_BASE_METRIC,
                higher_is_better=CV_OBJECTIVE_IS_HIGHER_BETTER,
                std_penalty=CV_STD_PENALTY,
            )
            cv_metric_mean = float(
                fold_score_summary[f"cv_{CV_OBJECTIVE_BASE_METRIC}_mean"]
            )
            cv_metric_weighted_mean = float(
                fold_score_summary[f"cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean"]
            )
            cv_metric_std = float(
                fold_score_summary[f"cv_{CV_OBJECTIVE_BASE_METRIC}_std"]
            )
            objective_base_value = float(fold_score_summary["objective_base_value"])
            objective_value = float(fold_score_summary["objective_value"])
        else:
            cv_metric_mean = float(mean_series[best_index])
            cv_metric_weighted_mean = float(cv_metric_mean)
            cv_metric_std = float(std_series[best_index])
            objective_base_value = float(cv_metric_mean)
            objective_value = float(objective_series[best_index])
        del cv_results
        gc.collect()
        trial.set_user_attr("best_iteration", best_iteration)
        trial.set_user_attr(f"cv_{CV_OBJECTIVE_BASE_METRIC}_mean", cv_metric_mean)
        trial.set_user_attr(
            f"cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean",
            cv_metric_weighted_mean,
        )
        trial.set_user_attr(f"cv_{CV_OBJECTIVE_BASE_METRIC}_std", cv_metric_std)
        trial.set_user_attr("objective_base_value", objective_base_value)
        return objective_value

    return objective


def make_top_trial_recheck_output_paths(
        study_name,
        top_n,
        output_json=None,
        output_csv=None,
):
    if output_json is not None and output_csv is not None:
        return output_json, output_csv

    safe_study_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", study_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"{safe_study_name}_top{top_n}_final_cv_{timestamp}"
    json_path = output_json or (TOP_TRIALS_RECHECK_OUTPUT_DIR / f"{stem}.json")
    csv_path = output_csv or (TOP_TRIALS_RECHECK_OUTPUT_DIR / f"{stem}.csv")
    return json_path, csv_path


def build_top_trial_recheck_summary_row(result):
    row = {
        "recheck_rank": int(result["recheck_rank"]),
        "optuna_rank": int(result["optuna_rank"]),
        "trial_number": int(result["trial_number"]),
        "optuna_objective": float(result["optuna_objective"]),
        "optuna_best_iteration": result["optuna_best_iteration"],
        f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_mean": result[
            f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_mean"
        ],
        f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean": result.get(
            f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean"
        ),
        f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_std": result[
            f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_std"
        ],
        "recheck_objective": float(result["recheck_objective"]),
        "recheck_mean_best_iteration": int(result["recheck_mean_best_iteration"]),
    }
    row.update(result["recheck_metric_summary"])
    for metric_name, metric_value in result["recheck_cv_mean_metrics"].items():
        row[f"recheck_cv_mean_{metric_name}"] = float(metric_value)
    for metric_name, metric_value in result["recheck_cv_std_metrics"].items():
        row[f"recheck_cv_std_{metric_name}"] = float(metric_value)
    return row


def build_top_trial_recheck_best_trial(result):
    best_metric_mean = result["recheck_cv_mean_metrics"].get(
        RECHECK_OBJECTIVE_BASE_METRIC
    )
    best_metric_weighted_mean = result["recheck_metric_summary"].get(
        f"cv_{RECHECK_OBJECTIVE_BASE_METRIC}_weighted_mean"
    )
    best_metric_std = result["recheck_cv_std_metrics"].get(
        RECHECK_OBJECTIVE_BASE_METRIC
    )
    optuna_metric_mean = result.get(f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_mean")
    optuna_metric_weighted_mean = result.get(
        f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean"
    )
    optuna_metric_std = result.get(f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_std")
    return {
        "recheck_rank": int(result["recheck_rank"]),
        "optuna_rank": int(result["optuna_rank"]),
        "number": int(result["trial_number"]),
        f"objective_{resolve_recheck_objective_name()}": float(
            result["recheck_objective"]
        ),
        f"cv_{RECHECK_OBJECTIVE_BASE_METRIC}_mean": (
            float(best_metric_mean) if best_metric_mean is not None else None
        ),
        f"cv_{RECHECK_OBJECTIVE_BASE_METRIC}_weighted_mean": (
            float(best_metric_weighted_mean)
            if best_metric_weighted_mean is not None
            else None
        ),
        f"cv_{RECHECK_OBJECTIVE_BASE_METRIC}_std": (
            float(best_metric_std) if best_metric_std is not None else None
        ),
        f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_mean": (
            float(optuna_metric_mean) if optuna_metric_mean is not None else None
        ),
        f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean": (
            float(optuna_metric_weighted_mean)
            if optuna_metric_weighted_mean is not None
            else None
        ),
        f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_std": (
            float(optuna_metric_std) if optuna_metric_std is not None else None
        ),
        "mean_best_iteration": int(result["recheck_mean_best_iteration"]),
        "params": result["params"],
    }


def run_top_trials_recheck(
        study_name,
        storage,
        top_n,
        output_json=None,
        output_csv=None,
):
    if top_n < 1:
        raise ValueError("top_n must be >= 1.")

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
    x = training_data["x"]
    y = training_data["y"]
    sample_weight = training_data["sample_weight"]
    monotone_constraint_summary = summarize_lgbm_monotone_constraints(x.columns)
    folds = make_final_walk_forward_folds(
        n_rows=len(x),
        n_folds=FINAL_CV_FOLDS,
        test_to_train_ratio=FINAL_WF_TEST_TO_TRAIN_RATIO,
    )
    fold_weight_by_id = build_fold_recency_weights(folds)

    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )
    completed_trials = [
        trial
        for trial in study.get_trials(
            deepcopy=False,
            states=(optuna.trial.TrialState.COMPLETE,),
        )
        if trial.value is not None
    ]
    completed_trials.sort(
        key=lambda trial: float(trial.value),
        reverse=bool(CV_OBJECTIVE_IS_HIGHER_BETTER),
    )
    selected_trials = completed_trials[:top_n]
    if not selected_trials:
        raise ValueError(
            f"No completed trials found for study_name={study_name!r} in storage={storage!r}."
        )

    print(
        f"start recheck | study_name={study_name} storage={storage} "
        f"completed_trials={len(completed_trials)} selected_trials={len(selected_trials)} "
        f"rows={len(x)} features={x.shape[1]} folds={len(folds)} "
        f"test/train={FINAL_WF_TEST_TO_TRAIN_RATIO:.3f} "
        f"float_precision={modeling_float_dtype_name}"
    )
    print(
        "start recheck | "
        f"fold weighting | enabled={bool(ENABLE_FOLD_RECENCY_WEIGHTING)} "
        f"active={is_nontrivial_fold_recency_weighting_enabled()} "
        f"mode={FOLD_RECENCY_WEIGHTING_MODE} "
        f"min={float(FOLD_RECENCY_WEIGHT_MIN):.4f} "
        f"max={float(FOLD_RECENCY_WEIGHT_MAX):.4f} "
        f"std=unweighted"
    )
    if feature_subset:
        print(
            "start recheck | "
            f"feature_subset_path={feature_subset['path']} count={feature_subset['count']}"
        )
    print(
        "start recheck | "
        f"monotone_constraints={format_lgbm_monotone_constraint_summary(monotone_constraint_summary)}"
    )

    recheck_results = []
    for optuna_rank, trial in enumerate(selected_trials, start=1):
        print(
            f"recheck trial {optuna_rank}/{len(selected_trials)} | "
            f"trial_number={trial.number} optuna_objective={float(trial.value):.8f}"
        )
        cv_result, _, _ = evaluate_walk_forward_variant(
            x=x,
            y=y,
            sample_weight=sample_weight,
            folds=folds,
            param_overrides=trial.params,
            model_variant=f"trial_{trial.number}",
            collect_oof_predictions=False,
            collect_feature_importance=False,
            early_stopping_verbose=False,
            float_dtype=modeling_float_dtype,
        )
        recheck_metric_summary = summarize_cv_result_metric(
            cv_result=cv_result,
            folds=folds,
            fold_weight_by_id=fold_weight_by_id,
            base_metric=RECHECK_OBJECTIVE_BASE_METRIC,
            higher_is_better=RECHECK_OBJECTIVE_IS_HIGHER_BETTER,
            std_penalty=RECHECK_STD_PENALTY,
        )
        recheck_metric_mean = float(
            recheck_metric_summary[f"cv_{RECHECK_OBJECTIVE_BASE_METRIC}_mean"]
        )
        recheck_metric_weighted_mean = float(
            recheck_metric_summary[
                f"cv_{RECHECK_OBJECTIVE_BASE_METRIC}_weighted_mean"
            ]
        )
        recheck_metric_std = float(
            recheck_metric_summary[f"cv_{RECHECK_OBJECTIVE_BASE_METRIC}_std"]
        )
        recheck_objective = float(recheck_metric_summary["objective_value"])
        trial_result = {
            "optuna_rank": int(optuna_rank),
            "trial_number": int(trial.number),
            "optuna_objective": float(trial.value),
            "optuna_best_iteration": trial.user_attrs.get("best_iteration"),
            f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_mean": trial.user_attrs.get(
                f"cv_{CV_OBJECTIVE_BASE_METRIC}_mean"
            ),
            f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean": trial.user_attrs.get(
                f"cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean"
            ),
            f"optuna_cv_{CV_OBJECTIVE_BASE_METRIC}_std": trial.user_attrs.get(
                f"cv_{CV_OBJECTIVE_BASE_METRIC}_std"
            ),
            "params": trial.params,
            "recheck_objective": recheck_objective,
            "recheck_mean_best_iteration": int(cv_result["mean_best_iteration"]),
            "recheck_metric_summary": recheck_metric_summary,
            "recheck_cv_mean_metrics": cv_result["cv_mean_metrics"],
            "recheck_cv_std_metrics": cv_result["cv_std_metrics"],
            "recheck_folds": cv_result["folds"],
        }
        recheck_results.append(trial_result)
        result_parts = [
            f"recheck result | trial_number={trial.number}",
            f"{RECHECK_OBJECTIVE_BASE_METRIC}={recheck_metric_mean:.8f}",
            (
                f"{RECHECK_OBJECTIVE_BASE_METRIC}_weighted_mean="
                f"{recheck_metric_weighted_mean:.8f}"
            ),
            f"{RECHECK_OBJECTIVE_BASE_METRIC}_std={recheck_metric_std:.8f}",
        ]
        if RECHECK_OBJECTIVE_BASE_METRIC != "binary_logloss":
            result_parts.extend(
                [
                    f"logloss={cv_result['cv_mean_metrics']['binary_logloss']:.8f}",
                    (
                        "logloss_std="
                        f"{cv_result['cv_std_metrics']['binary_logloss']:.8f}"
                    ),
                ]
            )
        result_parts.extend(
            [
                f"objective={recheck_objective:.8f}",
                f"mean_best_iteration={cv_result['mean_best_iteration']}",
            ]
        )
        print(" ".join(result_parts))

    recheck_results.sort(
        key=lambda item: float(item["recheck_objective"]),
        reverse=bool(RECHECK_OBJECTIVE_IS_HIGHER_BETTER),
    )
    for recheck_rank, result in enumerate(recheck_results, start=1):
        result["recheck_rank"] = int(recheck_rank)

    json_path, csv_path = make_top_trial_recheck_output_paths(
        study_name=study_name,
        top_n=top_n,
        output_json=output_json,
        output_csv=output_csv,
    )
    summary_rows = [
        build_top_trial_recheck_summary_row(result) for result in recheck_results
    ]
    best_result = recheck_results[0]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_name": study_name,
        "storage": storage,
        "top_n_requested": int(top_n),
        "top_n_evaluated": len(recheck_results),
        "data_path": str(data_path),
        "feature_selection": summarize_feature_subset(
            feature_subset,
            excluded_features=excluded_features,
        ),
        "monotone_constraints": monotone_constraint_summary,
        "sample_weight": {
            "used": True,
            "source": training_data["sample_weight_source"],
            **training_data["sample_weight_summary"],
        },
        "train_pipeline_alignment": {
            "source_script": "train_lgbm.py",
            "cv_folds": FINAL_CV_FOLDS,
            "walk_forward_test_to_train_ratio": FINAL_WF_TEST_TO_TRAIN_RATIO,
        },
        "fold_recency_weighting": {
            "enabled": bool(ENABLE_FOLD_RECENCY_WEIGHTING),
            "active": bool(is_nontrivial_fold_recency_weighting_enabled()),
            "mode": str(FOLD_RECENCY_WEIGHTING_MODE),
            "min_weight": float(FOLD_RECENCY_WEIGHT_MIN),
            "max_weight": float(FOLD_RECENCY_WEIGHT_MAX),
            "std_score_aggregation": "unweighted",
            "fold_weights": fold_weight_items_for_summary(folds, fold_weight_by_id),
        },
        "recheck_objective": {
            "name": resolve_recheck_objective_name(),
            "base_metric": RECHECK_OBJECTIVE_BASE_METRIC,
            "aggregation": objective_aggregation_description(
                RECHECK_OBJECTIVE_BASE_METRIC
            ),
            "std_penalty": float(RECHECK_STD_PENALTY),
        },
        "best_trial": build_top_trial_recheck_best_trial(best_result),
        "best_params": best_result["params"],
        "results": recheck_results,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    print(
        f"best recheck | trial_number={best_result['trial_number']} "
        f"recheck_objective={best_result['recheck_objective']:.8f} "
        f"{RECHECK_OBJECTIVE_BASE_METRIC}="
        f"{best_result['recheck_cv_mean_metrics'][RECHECK_OBJECTIVE_BASE_METRIC]:.8f} "
        f"{RECHECK_OBJECTIVE_BASE_METRIC}_weighted_mean="
        f"{best_result['recheck_metric_summary'][f'cv_{RECHECK_OBJECTIVE_BASE_METRIC}_weighted_mean']:.8f} "
        f"{RECHECK_OBJECTIVE_BASE_METRIC}_std="
        f"{best_result['recheck_cv_std_metrics'][RECHECK_OBJECTIVE_BASE_METRIC]:.8f}"
    )
    print(f"Saved recheck payload: {json_path}")
    print(f"Saved recheck csv: {csv_path}")


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
    modeling_float_dtype = resolve_modeling_float_dtype(dataset_settings)
    modeling_float_dtype_name = resolve_modeling_float_dtype_name(dataset_settings)
    data_path = resolve_modeling_dataset_output_paths(dataset_settings)["parquet"]
    feature_subset = load_feature_subset_from_settings(dataset_settings)
    excluded_features = load_excluded_feature_names_from_settings(dataset_settings)

    (
        x_np,
        y_np,
        sample_weight_np,
        rows_after_target_notna,
        sample_weight_source,
        sample_weight_summary,
        feature_names,
    ) = load_generic_training_data(
        data_path=data_path,
        feature_subset=feature_subset,
        excluded_features=excluded_features,
        float_dtype=modeling_float_dtype,
    )
    monotone_constraint_summary = summarize_lgbm_monotone_constraints(feature_names)
    folds = make_walk_forward_folds(
        n_rows=len(x_np),
        n_folds=CV_FOLDS,
        test_to_train_ratio=WF_TEST_TO_TRAIN_RATIO,
    )
    fold_indices = build_fold_indices(folds)
    fold_weight_by_id = build_fold_recency_weights(folds)

    train_set = lgb.Dataset(
        data=x_np,
        label=y_np,
        weight=sample_weight_np,
        feature_name=list(feature_names),
        free_raw_data=True,
    )

    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=TPE_STARTUP_TRIALS,
        multivariate=True,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=PRUNER_MIN_RESOURCE,
        max_resource=MAX_N_ESTIMATORS,
        reduction_factor=PRUNER_REDUCTION_FACTOR,
        bootstrap_count=PRUNER_BOOTSTRAP_COUNT,
    )

    if STORAGE.startswith("sqlite:///"):
        Path(STORAGE.replace("sqlite:///", "", 1)).parent.mkdir(
            parents=True, exist_ok=True
        )

    print(
        f"start optimize | rows={len(x_np)} features={x_np.shape[1]} folds={len(fold_indices)} "
        f"trials={N_TRIALS} timeout={TIMEOUT_SECONDS} prune_every={PRUNE_REPORT_EVERY_N_ITER} "
        f"pruner_bootstrap_count={PRUNER_BOOTSTRAP_COUNT} "
        f"float_precision={modeling_float_dtype_name} "
        f"sample_weight_source={sample_weight_source} "
        f"objective={resolve_cv_objective_name()} std_penalty={CV_STD_PENALTY:.4f} "
        f"study_name={study_name} study_name_source={study_name_source} "
        f"load_if_exists={LOAD_IF_EXISTS}"
    )
    print(
        "start optimize | "
        f"search_params={sorted(LGBM_OPTUNA_SEARCH_SPACE)} "
        f"seed_trials_configured={len(OPTUNA_SEED_TRIAL_PARAMS)}"
    )
    print(
        "start optimize | "
        f"monotone_constraints={format_lgbm_monotone_constraint_summary(monotone_constraint_summary)}"
    )
    print(
        "start optimize | "
        f"fold weighting | enabled={bool(ENABLE_FOLD_RECENCY_WEIGHTING)} "
        f"active={is_nontrivial_fold_recency_weighting_enabled()} "
        f"mode={FOLD_RECENCY_WEIGHTING_MODE} "
        f"min={float(FOLD_RECENCY_WEIGHT_MIN):.4f} "
        f"max={float(FOLD_RECENCY_WEIGHT_MAX):.4f} "
        f"std=unweighted"
    )
    if feature_subset:
        print(
            "start optimize | "
            f"feature_subset_path={feature_subset['path']} count={feature_subset['count']}"
        )

    study = optuna.create_study(
        study_name=study_name,
        storage=STORAGE,
        direction=objective_study_direction(
            higher_is_better=CV_OBJECTIVE_IS_HIGHER_BETTER
        ),
        sampler=sampler,
        pruner=pruner,
        load_if_exists=LOAD_IF_EXISTS,
    )
    enqueue_seed_trials(
        study=study,
        seed_trial_params=OPTUNA_SEED_TRIAL_PARAMS,
        search_space=LGBM_OPTUNA_SEARCH_SPACE,
    )

    objective = make_objective(
        train_set=train_set,
        feature_names=feature_names,
        x_np=x_np,
        y_np=y_np,
        sample_weight_np=sample_weight_np,
        folds=folds,
        fold_indices=fold_indices,
        fold_weight_by_id=fold_weight_by_id,
        search_space=LGBM_OPTUNA_SEARCH_SPACE,
    )
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT_SECONDS,
        n_jobs=OPTUNA_OPTIMIZE_N_JOBS,
        gc_after_trial=True,
        show_progress_bar=True,
        catch=(lgb.basic.LightGBMError, OSError),
    )

    best = study.best_trial
    best_cv_metric_mean = best.user_attrs.get(f"cv_{CV_OBJECTIVE_BASE_METRIC}_mean")
    best_cv_metric_weighted_mean = best.user_attrs.get(
        f"cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean"
    )
    best_cv_metric_std = best.user_attrs.get(f"cv_{CV_OBJECTIVE_BASE_METRIC}_std")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "target_col": TARGET_COL,
        "sample_weight_col": TARGET_WEIGHT_COL,
        "sample_weight": {
            "used": True,
            "source": sample_weight_source,
            **sample_weight_summary,
        },
        "feature_selection": summarize_feature_subset(
            feature_subset,
            excluded_features=excluded_features,
        ),
        "monotone_constraints": monotone_constraint_summary,
        "rows_after_target_notna": int(rows_after_target_notna),
        "decision_row_filter": {
            "enabled": False,
        },
        "study_name": study_name,
        "study_name_source": study_name_source,
        "storage": STORAGE,
        "run_timestamp_utc": run_timestamp,
        "lgbm_optuna_search_space": LGBM_OPTUNA_SEARCH_SPACE,
        "optuna_seed_trial_default_params": OPTUNA_SEED_TRIAL_DEFAULT_PARAMS,
        "optuna_seed_trial_params": OPTUNA_SEED_TRIAL_PARAMS,
        "cv_objective": {
            "name": resolve_cv_objective_name(),
            "base_metric": CV_OBJECTIVE_BASE_METRIC,
            "aggregation": objective_aggregation_description(
                CV_OBJECTIVE_BASE_METRIC
            ),
            "std_penalty": float(CV_STD_PENALTY),
        },
        "recommended_final_selection": {
            "name": resolve_recheck_objective_name(),
            "base_metric": RECHECK_OBJECTIVE_BASE_METRIC,
            "aggregation": objective_aggregation_description(
                RECHECK_OBJECTIVE_BASE_METRIC
            ),
            "std_penalty": float(RECHECK_STD_PENALTY),
            "workflow": "run_top_trials_recheck",
        },
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
        "pruner_bootstrap_count": PRUNER_BOOTSTRAP_COUNT,
        "best_trial": {
            "number": int(best.number),
            f"objective_{resolve_cv_objective_name()}": float(best.value),
            f"cv_{CV_OBJECTIVE_BASE_METRIC}_mean": (
                float(best_cv_metric_mean)
                if best_cv_metric_mean is not None
                else None
            ),
            f"cv_{CV_OBJECTIVE_BASE_METRIC}_weighted_mean": (
                float(best_cv_metric_weighted_mean)
                if best_cv_metric_weighted_mean is not None
                else None
            ),
            f"cv_{CV_OBJECTIVE_BASE_METRIC}_std": (
                float(best_cv_metric_std) if best_cv_metric_std is not None else None
            ),
            "best_iteration": best.user_attrs.get("best_iteration"),
            "params": best.params,
        },
        "artifacts": {
            "best_result_path": str(best_result_path),
            "trials_csv_path": str(trials_csv_path),
        },
    }

    best_result_path.parent.mkdir(parents=True, exist_ok=True)
    best_result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    trials_csv_path.parent.mkdir(parents=True, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_csv_path, index=False)

    print(
        f"Best trial: #{best.number} "
        f"objective={best.value:.8f} "
        f"{CV_OBJECTIVE_BASE_METRIC}_mean={float(best_cv_metric_mean):.8f} "
        f"{CV_OBJECTIVE_BASE_METRIC}_weighted_mean={float(best_cv_metric_weighted_mean):.8f} "
        f"{CV_OBJECTIVE_BASE_METRIC}_std={float(best_cv_metric_std):.8f} "
        f"iter={best.user_attrs.get('best_iteration')}"
    )
    print(
        "Recommended final selection | "
        f"workflow=run_top_trials_recheck "
        f"objective={resolve_recheck_objective_name()} "
        f"std_penalty={RECHECK_STD_PENALTY:.4f}"
    )
    print(f"Saved best payload: {best_result_path}")
    print(f"Saved trials csv: {trials_csv_path}")


def main():
    if RUN_MODE == "recheck-topn":
        run_top_trials_recheck(
            study_name=resolve_existing_study_name(
                RECHECK_STUDY_NAME,
                STUDY_NAME,
                setting_name="RECHECK_STUDY_NAME",
            ),
            storage=RECHECK_STORAGE,
            top_n=TOP_TRIALS_RECHECK_N,
            output_json=TOP_TRIALS_RECHECK_OUTPUT_JSON_PATH,
            output_csv=TOP_TRIALS_RECHECK_OUTPUT_CSV_PATH,
        )
        return

    if RUN_MODE != "optimize":
        raise ValueError(
            f"Unsupported RUN_MODE={RUN_MODE!r}. Expected 'optimize' or 'recheck-topn'."
        )

    run_optuna_optimization()


if __name__ == "__main__":
    main()
