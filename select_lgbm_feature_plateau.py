import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

from features.candle_features import CANDLE_PATTERN_COLS, RAW_OHLCV_COLS
from features.volume_profile_fixed_range import validate_volume_profile_feature_columns
from metrics_utils import make_sklearn_binary_brier_eval, weighted_brier_score
from target_weights import (
    TARGET_WEIGHT_COL,
    compute_target_weights_from_opened,
    summarize_target_weights,
)
from modeling_dataset_utils import (
    resolve_modeling_dataset_parquet_path,
)

DATA_PATH = resolve_modeling_dataset_parquet_path()
OUTPUT_ROOT = Path("data/analysis/feature_selector")

TARGET_COL = "target_5m_candle_up"
FEATURE_COLS = None
EXCLUDE_COLS = []
TIME_COLS = ["Opened"]
DROP_RAW_OHLCV = True

USE_SAMPLE_WEIGHTS = True
SAMPLE_WEIGHT_COL = TARGET_WEIGHT_COL
FALLBACK_TO_UNIT_WEIGHTS = False
MIN_SAMPLE_WEIGHT = 0.80

RANKING_N_SPLITS = 20
TOPK_N_SPLITS = 20
WF_TEST_TO_TRAIN_RATIO = 0.2
ENABLE_FOLD_RECENCY_WEIGHTING = True
FOLD_RECENCY_WEIGHTING_MODE = "linear"
FOLD_RECENCY_WEIGHT_MIN = 1.0
FOLD_RECENCY_WEIGHT_MAX = 1.4

LGBM_DEVICE_TYPE = "gpu"
LGBM_MAX_BIN = 63
LGBM_GPU_USE_DP = False
MODEL_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 6,
    "min_data_in_leaf": 128,
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
    "n_jobs": 14,
    "verbosity": -1,
    "device_type": LGBM_DEVICE_TYPE,
    "max_bin": LGBM_MAX_BIN,
    "gpu_use_dp": LGBM_GPU_USE_DP,
}
N_ESTIMATORS = 300
EARLY_STOPPING_ROUNDS = 40
RANDOM_SEEDS = [37]

SCORER = {
    "name": "balanced_accuracy",
    "greater_is_better": True,
}

TOPK_SELECTION_MODE = "mean_plus_std"
TOPK_SELECTION_STD_COEF = 0.5

COARSE_GRID_BASE = 2
MIN_COARSE_K = 1
MAX_REFINEMENT_ROUNDS = None
MAX_SWEEP_EVALUATIONS = None
MIN_REFINEMENT_INTERVAL = 2

TOLERANCE_MODE = "abs"
ABS_TOL = 0.000005
REL_TOL = 0.0001
MIN_PLATEAU_FEATURE_SAVINGS = 20

MIN_NONZERO_IMPORTANCE_FOLDS = 10
PERMUTATION_TOP_N = 768
# Keep permutation reranking more stable than a single shuffle without making
# selector runtime explode as aggressively as larger repeat counts.
PERMUTATION_N_REPEATS = 4
PERMUTATION_BASE_SEED = 37
MAX_MISSING_RATIO = 0.05
NEAR_CONSTANT_THRESHOLD = 0.9999
DROP_DUPLICATE_COLUMNS = True
MAX_ABS_CORRELATION = 0.999
ENABLE_CORRELATION_FILTER = True
CORRELATION_FILTER_MODE = "screened_exact"  # modes: "screened_exact" or "full_matrix"
CORRELATION_SCREEN_SAMPLE_ROWS = 150_000  # rows used for the initial correlation screening
CORRELATION_SCREEN_MARGIN = 0.05  # lowers the screening threshold below the final drop threshold
CORRELATION_SCREEN_MIN_ROWS = 300_000  # minimum row count required to use screened_exact
CORRELATION_SCREEN_MIN_COLS = 64  # minimum feature count required to use screened_exact
CORRELATION_SCREEN_MIN_THRESHOLD = 0.98  # minimum MAX_ABS_CORRELATION required to use screened_exact

CONSOLE_PREVIEW_FEATURES = 50


def format_score_for_cli(value):
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{float(value):.8f}"


def format_feature_preview_for_cli(label, features, limit):
    preview = list(features[: int(limit)])
    if not preview:
        return f"{label}: none"
    return "\n".join([f"{label}:"] + [f"  {feature}" for feature in preview])


def print_prefilter_report_cli(filter_report_df):
    print("prefilter")
    for _, row in filter_report_df.iterrows():
        duration = float(row.get("duration_sec", np.nan))
        duration_text = ""
        if np.isfinite(duration):
            duration_text = f" time={duration:.3f}s"
        print(
            f"  {row['step']}: removed={int(row['removed_count'])} "
            f"remaining={int(row['remaining_count'])}{duration_text}"
        )


def load_dataframe(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported data file format: {path}")


def prepare_binary_target(y_raw):
    categories = pd.Categorical(y_raw)
    if np.any(categories.codes < 0):
        raise ValueError("Binary classification target still contains NaNs.")

    category_values = categories.categories.tolist()
    if len(category_values) != 2:
        raise ValueError(
            "Expected exactly 2 target classes for binary classification, got "
            f"{len(category_values)}: {category_values}"
        )

    y_encoded = pd.Series(categories.codes.astype(np.int8), index=y_raw.index)
    class_mapping = {
        int(code): str(label) for code, label in enumerate(category_values)
    }
    return y_encoded, class_mapping


def resolve_sample_weight_series(df):
    if not USE_SAMPLE_WEIGHTS:
        weights = np.ones(len(df), dtype=np.float32)
        return (
            pd.Series(weights, index=df.index, name=SAMPLE_WEIGHT_COL),
            "unit_weights_disabled_in_config",
            summarize_target_weights(weights),
        )

    if SAMPLE_WEIGHT_COL in df.columns:
        sample_weight = pd.to_numeric(df[SAMPLE_WEIGHT_COL], errors="raise")
        source = "dataset_column"
    elif SAMPLE_WEIGHT_COL == TARGET_WEIGHT_COL and "Opened" in df.columns:
        sample_weight = pd.Series(
            compute_target_weights_from_opened(df["Opened"]),
            index=df.index,
            name=SAMPLE_WEIGHT_COL,
        )
        source = "derived_from_opened"
    elif FALLBACK_TO_UNIT_WEIGHTS:
        sample_weight = pd.Series(
            np.ones(len(df), dtype=np.float32),
            index=df.index,
            name=SAMPLE_WEIGHT_COL,
        )
        source = "unit_weights_fallback"
    else:
        raise ValueError(
            f"Sample weight column '{SAMPLE_WEIGHT_COL}' missing and fallback disabled."
        )

    sample_weight = sample_weight.astype(np.float32, copy=False)
    sample_weight_np = sample_weight.to_numpy(dtype=np.float32, copy=False)
    if sample_weight_np.shape[0] != len(df):
        raise ValueError("Sample weights length mismatch.")
    if not np.isfinite(sample_weight_np).all():
        raise ValueError("Sample weights contain non-finite values.")
    if np.any(sample_weight_np <= 0.0):
        raise ValueError("Sample weights must be strictly positive.")

    return sample_weight, source, summarize_target_weights(sample_weight_np)


def filter_rows_by_min_sample_weight(df, context_label):
    sample_weight, source, _ = resolve_sample_weight_series(df)
    min_weight = float(MIN_SAMPLE_WEIGHT)
    keep_mask = sample_weight >= min_weight
    filtered_df = df.loc[keep_mask].copy()
    filtered_weight = sample_weight.loc[keep_mask].astype(np.float32, copy=False)

    if filtered_df.empty:
        raise ValueError(
            f"{context_label} has no rows left after sample-weight filtering: "
            f"{SAMPLE_WEIGHT_COL} >= {min_weight:.2f}."
        )

    row_filter_info = {
        "enabled": True,
        "weight_col": SAMPLE_WEIGHT_COL,
        "min_weight": min_weight,
        "rows_before": len(df),
        "rows_after": len(filtered_df),
        "rows_removed": int((~keep_mask).sum()),
    }
    print(
        f"{context_label} | row_filter {SAMPLE_WEIGHT_COL}>={min_weight:.2f} "
        f"removed={row_filter_info['rows_removed']} remaining={row_filter_info['rows_after']}"
    )

    return (
        filtered_df,
        filtered_weight,
        source,
        summarize_target_weights(
            filtered_weight.to_numpy(dtype=np.float32, copy=False)
        ),
        row_filter_info,
    )


def resolve_feature_columns(df):
    if FEATURE_COLS is not None:
        legacy_1m_cdl = [col for col in FEATURE_COLS if col in CANDLE_PATTERN_COLS]
        if legacy_1m_cdl:
            raise ValueError(
                "Configured FEATURE_COLS include disabled 1m candle pattern columns: "
                + ", ".join(legacy_1m_cdl)
            )
        missing = [col for col in FEATURE_COLS if col not in df.columns]
        if missing:
            raise ValueError(f"Configured FEATURE_COLS missing in dataset: {missing}")
        feature_cols = list(FEATURE_COLS)
        non_numeric = [
            col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric:
            raise ValueError(
                "Configured FEATURE_COLS contain non-numeric columns: "
                + ", ".join(non_numeric)
            )
        validate_volume_profile_feature_columns(
            feature_cols,
            source_label=f"FEATURE_COLS for dataset {DATA_PATH}",
        )
        return feature_cols, []

    excluded = set(EXCLUDE_COLS)
    excluded.add(TARGET_COL)
    if SAMPLE_WEIGHT_COL in df.columns:
        excluded.add(SAMPLE_WEIGHT_COL)
    excluded.update(TIME_COLS)
    excluded.update(CANDLE_PATTERN_COLS)
    if DROP_RAW_OHLCV:
        excluded.update(RAW_OHLCV_COLS)

    non_numeric = []
    feature_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
        else:
            non_numeric.append(col)

    if not feature_cols:
        raise ValueError("No numeric feature columns left after autodetection.")
    validate_volume_profile_feature_columns(
        feature_cols,
        source_label=f"feature selector dataset {DATA_PATH}",
    )

    return feature_cols, non_numeric


def compute_column_digest(series):
    hashed = pd.util.hash_pandas_object(series, index=False, categorize=False)
    digest = hashlib.blake2b(
        hashed.to_numpy(dtype=np.uint64, copy=False).tobytes(),
        digest_size=16,
    )
    digest.update(str(series.dtype).encode("utf-8"))
    return digest.hexdigest()


def find_duplicate_columns(df):
    duplicates = []
    duplicate_map = {}
    digests = {}

    for col in df.columns:
        series = df[col]
        digest = compute_column_digest(series)
        if digest not in digests:
            digests[digest] = [col]
            continue

        matched = None
        for existing_col in digests[digest]:
            if series.equals(df[existing_col]):
                matched = existing_col
                break

        if matched is None:
            digests[digest].append(col)
            continue

        duplicates.append(col)
        duplicate_map[col] = matched

    return duplicates, duplicate_map


def find_near_constant_columns(df, threshold, nunique=None):
    if df.shape[1] == 0:
        return []

    threshold = float(threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("NEAR_CONSTANT_THRESHOLD must be in [0, 1].")

    n_rows = len(df)
    if n_rows == 0:
        return []

    min_dominant_count = int(np.ceil(threshold * n_rows))
    max_non_dominant = max(0, n_rows - min_dominant_count)
    max_candidate_unique = max_non_dominant + 1
    if nunique is None:
        nunique = df.nunique(dropna=False)

    candidate_cols = nunique[nunique <= max_candidate_unique].index.tolist()
    near_constant_cols = []
    for col in candidate_cols:
        dominant_count = int(df[col].value_counts(dropna=False, sort=False).max())
        if dominant_count >= min_dominant_count:
            near_constant_cols.append(col)

    return near_constant_cols


def should_use_screened_correlation(df, threshold):
    if CORRELATION_FILTER_MODE != "screened_exact":
        return False
    if (
        CORRELATION_SCREEN_SAMPLE_ROWS is None
        or int(CORRELATION_SCREEN_SAMPLE_ROWS) <= 0
    ):
        return False
    return (
        len(df) >= int(CORRELATION_SCREEN_MIN_ROWS)
        and df.shape[1] >= int(CORRELATION_SCREEN_MIN_COLS)
        and float(threshold) >= float(CORRELATION_SCREEN_MIN_THRESHOLD)
    )


def build_evenly_spaced_row_index(n_rows, sample_rows):
    sample_rows = int(sample_rows)
    if sample_rows <= 0 or n_rows <= sample_rows:
        return np.arange(n_rows, dtype=np.int64)

    sample_idx = np.linspace(0, n_rows - 1, num=sample_rows, dtype=np.int64)
    return np.unique(sample_idx)


def find_highly_correlated_columns_full(df, threshold):
    if df.shape[1] <= 1:
        return [], {}, {"mode": "full_matrix", "exact_pair_checks": 0}

    abs_corr = df.corr(method="pearson", min_periods=3).abs()
    if abs_corr.empty:
        return [], {}, {"mode": "full_matrix", "exact_pair_checks": 0}

    col_names = abs_corr.columns.to_list()
    corr_values = abs_corr.to_numpy(copy=False)
    kept_mask = np.ones(len(col_names), dtype=bool)
    dropped_cols = []
    drop_map = {}
    threshold = float(threshold)

    for col_idx in range(1, len(col_names)):
        prior_corr = corr_values[:col_idx, col_idx]
        candidate_mask = kept_mask[:col_idx] & np.greater_equal(prior_corr, threshold)
        if not np.any(candidate_mask):
            continue

        keeper_idx = int(np.flatnonzero(candidate_mask)[0])
        dropped_col = col_names[col_idx]
        kept_col = col_names[keeper_idx]
        kept_mask[col_idx] = False
        dropped_cols.append(dropped_col)
        drop_map[dropped_col] = kept_col

    return dropped_cols, drop_map, {"mode": "full_matrix", "exact_pair_checks": 0}


def find_highly_correlated_columns_screened(df, threshold):
    threshold = float(threshold)
    screen_threshold = max(0.0, threshold - float(CORRELATION_SCREEN_MARGIN))
    sample_idx = build_evenly_spaced_row_index(len(df), CORRELATION_SCREEN_SAMPLE_ROWS)
    sample_df = df.iloc[sample_idx]
    abs_corr = sample_df.corr(method="pearson", min_periods=3).abs()
    if abs_corr.empty:
        return [], {}, {
            "mode": "screened_exact",
            "screen_sample_rows": int(len(sample_idx)),
            "screen_threshold": screen_threshold,
            "screen_candidate_pairs": 0,
            "exact_pair_checks": 0,
        }

    col_names = abs_corr.columns.to_list()
    corr_values = abs_corr.to_numpy(copy=False)
    screened_prior_map = {}
    screen_candidate_pairs = 0

    for col_idx in range(1, len(col_names)):
        prior_corr = corr_values[:col_idx, col_idx]
        candidate_idx = np.flatnonzero(np.greater_equal(prior_corr, screen_threshold))
        if candidate_idx.size == 0:
            continue
        screened_prior_map[col_idx] = candidate_idx
        screen_candidate_pairs += int(candidate_idx.size)

    if not screened_prior_map:
        return [], {}, {
            "mode": "screened_exact",
            "screen_sample_rows": int(len(sample_idx)),
            "screen_threshold": screen_threshold,
            "screen_candidate_pairs": 0,
            "exact_pair_checks": 0,
        }

    kept_mask = np.ones(len(col_names), dtype=bool)
    dropped_cols = []
    drop_map = {}
    exact_pair_checks = 0

    series_by_idx = [df.iloc[:, idx] for idx in range(df.shape[1])]
    for col_idx in range(1, len(col_names)):
        candidate_idx = screened_prior_map.get(col_idx)
        if candidate_idx is None:
            continue

        for keeper_idx in candidate_idx:
            if not kept_mask[int(keeper_idx)]:
                continue

            exact_pair_checks += 1
            corr = series_by_idx[int(keeper_idx)].corr(
                series_by_idx[col_idx],
                method="pearson",
                min_periods=3,
            )
            if pd.isna(corr) or abs(float(corr)) < threshold:
                continue

            kept_mask[col_idx] = False
            dropped_col = col_names[col_idx]
            kept_col = col_names[int(keeper_idx)]
            dropped_cols.append(dropped_col)
            drop_map[dropped_col] = kept_col
            break

    return dropped_cols, drop_map, {
        "mode": "screened_exact",
        "screen_sample_rows": int(len(sample_idx)),
        "screen_threshold": screen_threshold,
        "screen_candidate_pairs": int(screen_candidate_pairs),
        "exact_pair_checks": int(exact_pair_checks),
    }


def find_highly_correlated_columns(df, threshold):
    if df.shape[1] <= 1:
        return [], {}, {"mode": "skipped", "exact_pair_checks": 0}

    if should_use_screened_correlation(df, threshold):
        return find_highly_correlated_columns_screened(df, threshold)

    return find_highly_correlated_columns_full(df, threshold)


def make_prefilter_report_row(
    step,
    removed_features,
    remaining_count,
    duration_sec,
    details=None,
):
    if details is None:
        details = {}
    return {
        "step": step,
        "removed_count": len(removed_features),
        "remaining_count": int(remaining_count),
        "duration_sec": float(duration_sec),
        "removed_features_json": json.dumps(removed_features),
        "details_json": json.dumps(details),
    }


def prefilter_features(x):
    work = x
    report_rows = [
        {
            "step": "input",
            "removed_count": 0,
            "remaining_count": int(work.shape[1]),
            "duration_sec": 0.0,
            "removed_features_json": json.dumps([]),
            "details_json": json.dumps({}),
        }
    ]
    duplicate_map = {}
    high_corr_drop_map = {}

    started = time.perf_counter()
    all_missing_cols = [col for col in work.columns if work[col].isna().all()]
    if all_missing_cols:
        work = work.drop(columns=all_missing_cols)
    report_rows.append(
        make_prefilter_report_row(
            step="all_missing",
            removed_features=all_missing_cols,
            remaining_count=work.shape[1],
            duration_sec=time.perf_counter() - started,
        )
    )

    started = time.perf_counter()
    nunique = work.nunique(dropna=False) if work.shape[1] > 0 else pd.Series(dtype=np.int64)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        work = work.drop(columns=constant_cols)
        nunique = nunique.drop(labels=constant_cols)
    report_rows.append(
        make_prefilter_report_row(
            step="constant",
            removed_features=constant_cols,
            remaining_count=work.shape[1],
            duration_sec=time.perf_counter() - started,
        )
    )

    started = time.perf_counter()
    near_constant_cols = []
    if work.shape[1] > 0 and NEAR_CONSTANT_THRESHOLD is not None:
        near_constant_cols = find_near_constant_columns(
            work,
            threshold=NEAR_CONSTANT_THRESHOLD,
            nunique=nunique,
        )
        if near_constant_cols:
            work = work.drop(columns=near_constant_cols)
            nunique = nunique.drop(labels=near_constant_cols)
    report_rows.append(
        make_prefilter_report_row(
            step="near_constant",
            removed_features=near_constant_cols,
            remaining_count=work.shape[1],
            duration_sec=time.perf_counter() - started,
        )
    )

    started = time.perf_counter()
    high_missing_cols = []
    if work.shape[1] > 0 and MAX_MISSING_RATIO is not None:
        missing_ratio = work.isna().mean()
        high_missing_cols = missing_ratio[
            missing_ratio > float(MAX_MISSING_RATIO)
        ].index.tolist()
        if high_missing_cols:
            work = work.drop(columns=high_missing_cols)
            nunique = nunique.drop(labels=high_missing_cols, errors="ignore")
    report_rows.append(
        make_prefilter_report_row(
            step="high_missing_ratio",
            removed_features=high_missing_cols,
            remaining_count=work.shape[1],
            duration_sec=time.perf_counter() - started,
        )
    )

    started = time.perf_counter()
    duplicate_cols = []
    if work.shape[1] > 0 and DROP_DUPLICATE_COLUMNS:
        duplicate_cols, duplicate_map = find_duplicate_columns(work)
        if duplicate_cols:
            work = work.drop(columns=duplicate_cols)
            nunique = nunique.drop(labels=duplicate_cols, errors="ignore")
    report_rows.append(
        make_prefilter_report_row(
            step="duplicate_columns",
            removed_features=duplicate_cols,
            remaining_count=work.shape[1],
            duration_sec=time.perf_counter() - started,
        )
    )

    started = time.perf_counter()
    high_corr_cols = []
    high_corr_details = {}
    if (
        work.shape[1] > 1
        and ENABLE_CORRELATION_FILTER
        and MAX_ABS_CORRELATION is not None
    ):
        (
            high_corr_cols,
            high_corr_drop_map,
            high_corr_details,
        ) = find_highly_correlated_columns(
            work,
            threshold=MAX_ABS_CORRELATION,
        )
        if high_corr_cols:
            work = work.drop(columns=high_corr_cols)
    report_rows.append(
        make_prefilter_report_row(
            step="very_high_correlation",
            removed_features=high_corr_cols,
            remaining_count=work.shape[1],
            duration_sec=time.perf_counter() - started,
            details=high_corr_details,
        )
    )

    return work, pd.DataFrame(report_rows), duplicate_map, high_corr_drop_map


def make_walk_forward_folds(n_rows, n_splits, test_to_train_ratio):
    if n_rows < 100:
        raise ValueError(f"Dataset too small for walk-forward CV: {n_rows} rows.")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if not (0.0 < test_to_train_ratio < 1.0):
        raise ValueError("WF_TEST_TO_TRAIN_RATIO must be in (0, 1).")

    ratio_inv = 1.0 / test_to_train_ratio
    test_len = int(np.floor(n_rows / (n_splits + ratio_inv)))
    train_len = int(np.floor(test_len / test_to_train_ratio))

    if test_len <= 0 or train_len <= 0:
        raise ValueError(
            f"Cannot create walk-forward folds for n_rows={n_rows}, "
            f"n_splits={n_splits}, ratio={test_to_train_ratio}."
        )

    folds = []
    for fold_id in range(n_splits):
        train_start = fold_id * test_len
        train_end = train_start + train_len
        valid_start = train_end
        valid_end = valid_start + test_len
        if valid_end > n_rows:
            break
        folds.append(
            {
                "fold_id": fold_id,
                "train_idx": np.arange(train_start, train_end, dtype=np.int32),
                "valid_idx": np.arange(valid_start, valid_end, dtype=np.int32),
            }
        )

    if len(folds) != n_splits:
        raise ValueError(
            f"Created {len(folds)} walk-forward folds, expected {n_splits}. "
            "Increase dataset size or lower n_splits."
        )

    return folds


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


def weighted_mean_rows(table, weights):
    table_arr = np.asarray(table, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    if table_arr.ndim != 2:
        raise ValueError("weighted_mean_rows expects a 2D array.")
    if table_arr.shape[1] != weights_arr.shape[0]:
        raise ValueError(
            "weighted_mean_rows weight mismatch: "
            f"{table_arr.shape[1]} != {weights_arr.shape[0]}"
        )
    return np.average(table_arr, axis=1, weights=weights_arr)


def fold_weight_items_for_summary(folds, fold_weight_by_id):
    weights = resolve_fold_weight_array(folds, fold_weight_by_id)
    return [
        {
            "fold_id": int(fold["fold_id"]),
            "weight": float(weight),
        }
        for fold, weight in zip(folds, weights)
    ]


def topk_selection_base_score_label():
    if is_nontrivial_fold_recency_weighting_enabled():
        return "weighted_mean_score"
    return "mean_score"


def prepare_fold_features(x_train_raw, x_valid_raw):
    drop_cols = x_train_raw.columns[x_train_raw.isna().all()].tolist()

    if drop_cols:
        x_train_raw = x_train_raw.drop(columns=drop_cols)
        x_valid_raw = x_valid_raw.drop(columns=drop_cols)

    return x_train_raw, x_valid_raw, drop_cols


def make_estimator(seed):
    params = {
        **MODEL_PARAMS,
        "random_state": int(seed),
        "n_estimators": int(N_ESTIMATORS),
        "objective": "binary",
    }
    return lgb.LGBMClassifier(**params)


def resolve_eval_metric():
    return make_sklearn_binary_brier_eval("brier_score")


def fit_model(
    model,
    x_train,
    y_train,
    w_train,
    x_valid,
    y_valid,
    w_valid,
):
    fit_kwargs = {
        "sample_weight": w_train,
    }
    callbacks = []
    if EARLY_STOPPING_ROUNDS is not None and EARLY_STOPPING_ROUNDS > 0:
        fit_kwargs["eval_set"] = [(x_valid, y_valid)]
        fit_kwargs["eval_sample_weight"] = [w_valid]
        fit_kwargs["eval_metric"] = resolve_eval_metric()
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=int(EARLY_STOPPING_ROUNDS),
                verbose=False,
            )
        )
    if callbacks:
        fit_kwargs["callbacks"] = callbacks

    model.fit(x_train, y_train, **fit_kwargs)
    return model


def get_best_iteration(model):
    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None or best_iteration <= 0:
        return int(N_ESTIMATORS)
    return int(best_iteration)


def predict_for_scoring(model, x_valid, best_iteration):
    y_pred = model.predict(x_valid, num_iteration=best_iteration)
    y_pred_proba = model.predict_proba(x_valid, num_iteration=best_iteration)
    return np.asarray(y_pred), np.asarray(y_pred_proba)


def score_predictions(
    scorer_cfg,
    y_true,
    y_pred,
    y_pred_proba,
    sample_weight,
):
    custom_scorer = scorer_cfg.get("callable")
    if custom_scorer is not None:
        return float(
            custom_scorer(
                y_true=y_true,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba,
                sample_weight=sample_weight,
            )
        )

    metric_name = str(scorer_cfg["name"]).strip().lower()
    metric_name = metric_name.replace("-", "_")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)

    if metric_name in {"f1", "f1_score"}:
        return float(
            f1_score(
                y_true,
                y_pred,
                sample_weight=sample_weight,
                average="binary",
                zero_division=0,
            )
        )

    if metric_name in {"balanced_accuracy", "balanced_acc"}:
        if y_pred is not None:
            return float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                    sample_weight=sample_weight,
                )
            )
        if y_pred_proba is None or y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != 2:
            raise ValueError(
                f"Metric '{metric_name}' requires binary predictions or class probabilities."
            )
        return float(
            balanced_accuracy_score(
                y_true,
                (y_pred_proba[:, 1] >= 0.5).astype(np.int8),
                sample_weight=sample_weight,
            )
        )

    if y_pred_proba is None or y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != 2:
        raise ValueError(
            f"Metric '{metric_name}' requires binary class probabilities with 2 columns."
        )

    if metric_name in {
        "brier",
        "brier_score",
    }:
        return float(
            weighted_brier_score(
                y_true=y_true,
                y_pred_proba=y_pred_proba[:, 1],
                sample_weight=sample_weight,
            )
        )

    if metric_name in {
        "logloss",
        "log_loss",
        "binary_logloss",
    }:
        return float(
            log_loss(
                y_true,
                y_pred_proba[:, 1],
                sample_weight=sample_weight,
                labels=[0, 1],
            )
        )

    raise ValueError(
        "Unsupported scorer. Allowed scorers: balanced_accuracy, brier_score, "
        "log_loss/binary_logloss, f1_score."
    )


def binary_logloss_from_positive_class_proba(y_true, y_pred_proba_pos, sample_weight):
    return float(
        log_loss(
            y_true,
            y_pred_proba_pos,
            sample_weight=sample_weight,
            labels=[0, 1],
        )
    )


def importance_series(model, feature_order, importance_type):
    booster = model.booster_
    return pd.Series(
        booster.feature_importance(importance_type=importance_type),
        index=booster.feature_name(),
        dtype=np.float64,
    ).reindex(feature_order, fill_value=0.0)


def topk_selection_score(mean_score, std_score):
    mode = str(TOPK_SELECTION_MODE).strip().lower()
    mean_score = float(mean_score)
    std_score = float(std_score)
    if mode == "mean_only":
        return mean_score
    if mode == "mean_plus_std":
        penalty = float(TOPK_SELECTION_STD_COEF) * std_score
        if SCORER["greater_is_better"]:
            return mean_score - penalty
        return mean_score + penalty
    raise ValueError(f"Unsupported TOPK_SELECTION_MODE: {TOPK_SELECTION_MODE}")


def topk_selection_formula():
    mode = str(TOPK_SELECTION_MODE).strip().lower()
    base_score_label = topk_selection_base_score_label()
    if mode == "mean_only":
        return base_score_label
    if mode == "mean_plus_std":
        sign = "-" if SCORER["greater_is_better"] else "+"
        return (
            f"{base_score_label} {sign} "
            f"{float(TOPK_SELECTION_STD_COEF):.4f} * unweighted_std_score"
        )
    raise ValueError(f"Unsupported TOPK_SELECTION_MODE: {TOPK_SELECTION_MODE}")


def sort_feature_table_prescreen(df):
    return df.sort_values(
        by=[
            "eligible_for_selection",
            "used_folds",
            "weighted_used_ratio",
            "used_ratio",
            "median_gain",
            "weighted_mean_gain",
            "mean_gain",
            "median_split",
            "weighted_mean_split",
            "mean_split",
            "feature",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)


def sort_feature_table_permutation(df):
    if df.empty:
        return df.reset_index(drop=True)

    return df.sort_values(
        by=[
            "permutation_weighted_mean_delta_logloss",
            "permutation_mean_delta_logloss",
            "permutation_median_delta_logloss",
            "permutation_std_delta_logloss",
            "used_folds",
            "weighted_used_ratio",
            "used_ratio",
            "median_gain",
            "weighted_mean_gain",
            "mean_gain",
            "feature",
        ],
        ascending=[
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)


def sort_feature_table_final(df):
    permutation_df = sort_feature_table_permutation(
        df[df["ranking_stage"] == "permutation"].copy()
    )
    tail_df = (
        df[df["ranking_stage"] == "prescreen_tail"]
        .copy()
        .sort_values(by=["prescreen_rank"], ascending=[True], kind="stable")
        .reset_index(drop=True)
    )
    return pd.concat([permutation_df, tail_df], ignore_index=True)


def make_permutation_seed(fold_id, feature_idx, repeat_idx):
    seed = (
        int(PERMUTATION_BASE_SEED)
        + (int(fold_id) + 1) * 1_000_003
        + (int(feature_idx) + 1) * 10_007
        + (int(repeat_idx) + 1) * 101
    )
    return int(seed % (2**32 - 1))


def build_fold_ranking(fold_gain_series, fold_used_series):
    fold_df = pd.DataFrame(
        {
            "feature": fold_gain_series.index,
            "fold_gain": fold_gain_series.to_numpy(dtype=np.float64, copy=False),
            "fold_used": fold_used_series.to_numpy(dtype=np.int8, copy=False),
        }
    )
    fold_df = fold_df.sort_values(
        by=["fold_used", "fold_gain", "feature"],
        ascending=[False, False, True],
        kind="stable",
    )
    return fold_df["feature"].tolist()


def run_feature_prescreen(x, y, sample_weight, folds, fold_weight_by_id):
    feature_order = list(x.columns)
    fold_gain_table = pd.DataFrame(index=feature_order)
    fold_split_table = pd.DataFrame(index=feature_order)
    fold_used_table = pd.DataFrame(index=feature_order)
    fold_rankings = {}
    fold_metadata = []
    fold_weight_array = resolve_fold_weight_array(folds, fold_weight_by_id)

    print(
        f"ranking | prescreen start features={len(feature_order)} "
        f"folds={len(folds)} seeds={len(RANDOM_SEEDS)}"
    )

    for fold_pos, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        train_idx = fold["train_idx"]
        valid_idx = fold["valid_idx"]
        fold_weight = float(fold_weight_by_id.loc[fold_id])
        print(
            f"ranking | prescreen fold={fold_pos}/{len(folds)} "
            f"id={fold_id} train={len(train_idx)} valid={len(valid_idx)} "
            f"weight={fold_weight:.4f}"
        )

        x_train_raw = x.iloc[train_idx]
        x_valid_raw = x.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        w_train = sample_weight.iloc[train_idx].to_numpy(dtype=np.float32, copy=False)
        w_valid = sample_weight.iloc[valid_idx].to_numpy(dtype=np.float32, copy=False)

        x_train, x_valid, dropped_all_nan = prepare_fold_features(
            x_train_raw,
            x_valid_raw,
        )

        seed_gain_series = []
        seed_split_series = []
        best_iterations = []
        for seed in RANDOM_SEEDS:
            model = make_estimator(seed)
            fit_model(
                model=model,
                x_train=x_train,
                y_train=y_train,
                w_train=w_train,
                x_valid=x_valid,
                y_valid=y_valid,
                w_valid=w_valid,
            )
            best_iterations.append(get_best_iteration(model))
            seed_gain_series.append(importance_series(model, feature_order, "gain"))
            seed_split_series.append(importance_series(model, feature_order, "split"))

        gain_frame = pd.concat(seed_gain_series, axis=1)
        split_frame = pd.concat(seed_split_series, axis=1)
        fold_gain_mean = gain_frame.mean(axis=1)
        fold_split_mean = split_frame.mean(axis=1)
        fold_used = split_frame.gt(0.0).any(axis=1).astype(np.int8)

        fold_gain_table[f"fold_{fold_id:02d}_gain"] = fold_gain_mean
        fold_split_table[f"fold_{fold_id:02d}_split"] = fold_split_mean
        fold_used_table[f"fold_{fold_id:02d}_used"] = fold_used
        fold_rankings[fold_id] = build_fold_ranking(fold_gain_mean, fold_used)
        fold_metadata.append(
            {
                "fold_id": fold_id,
                "train_size": len(train_idx),
                "valid_size": len(valid_idx),
                "fold_weight": fold_weight,
                "dropped_all_nan_train_features_count": len(dropped_all_nan),
                "mean_best_iteration": float(np.mean(best_iterations)),
            }
        )
        print(
            f"ranking | prescreen fold={fold_pos}/{len(folds)} done "
            f"mean_best_iteration={int(np.round(np.mean(best_iterations)))} "
            f"used_features={int(fold_used.sum())} "
            f"dropped_all_nan={len(dropped_all_nan)}"
        )

    ranking_df = pd.DataFrame(
        {
            "feature": feature_order,
            "weighted_mean_gain": weighted_mean_rows(
                fold_gain_table.to_numpy(dtype=np.float64, copy=False),
                fold_weight_array,
            ),
            "mean_gain": fold_gain_table.mean(axis=1).to_numpy(
                dtype=np.float64, copy=False
            ),
            "median_gain": fold_gain_table.median(axis=1).to_numpy(
                dtype=np.float64, copy=False
            ),
            "weighted_mean_split": weighted_mean_rows(
                fold_split_table.to_numpy(dtype=np.float64, copy=False),
                fold_weight_array,
            ),
            "mean_split": fold_split_table.mean(axis=1).to_numpy(
                dtype=np.float64, copy=False
            ),
            "median_split": fold_split_table.median(axis=1).to_numpy(
                dtype=np.float64, copy=False
            ),
            "used_folds": fold_used_table.sum(axis=1).to_numpy(
                dtype=np.int32, copy=False
            ),
            "weighted_used_ratio": weighted_mean_rows(
                fold_used_table.to_numpy(dtype=np.float64, copy=False),
                fold_weight_array,
            ),
            "used_ratio": fold_used_table.mean(axis=1).to_numpy(
                dtype=np.float64, copy=False
            ),
            "gain_per_fold_json": fold_gain_table.apply(
                lambda row: json.dumps([float(v) for v in row.tolist()]),
                axis=1,
            ).to_numpy(),
            "split_per_fold_json": fold_split_table.apply(
                lambda row: json.dumps([float(v) for v in row.tolist()]),
                axis=1,
            ).to_numpy(),
        }
    )
    ranking_df["eligible_for_selection"] = ranking_df["used_folds"] >= int(
        MIN_NONZERO_IMPORTANCE_FOLDS
    )
    ranking_df = sort_feature_table_prescreen(ranking_df)
    ranking_df["prescreen_rank"] = np.arange(
        1, len(ranking_df) + 1, dtype=np.int32
    )
    permutation_candidates = (
        ranking_df.loc[ranking_df["eligible_for_selection"], "feature"]
        .head(int(PERMUTATION_TOP_N))
        .tolist()
    )
    ranking_df["prescreen_candidate"] = ranking_df["feature"].isin(
        permutation_candidates
    )
    ranking_df["ranking_stage"] = np.where(
        ranking_df["prescreen_candidate"],
        "permutation",
        "prescreen_tail",
    )
    print(
        f"ranking | prescreen done eligible={int(ranking_df['eligible_for_selection'].sum())} "
        f"candidates={len(permutation_candidates)}\n"
        + format_feature_preview_for_cli(
            label="prescreen_top_features",
            features=ranking_df["feature"].tolist(),
            limit=CONSOLE_PREVIEW_FEATURES,
        )
    )

    return ranking_df, fold_rankings, fold_metadata


def run_permutation_reranking(
    x,
    y,
    sample_weight,
    folds,
    fold_weight_by_id,
    ranking_df,
):
    if int(PERMUTATION_N_REPEATS) <= 0:
        raise ValueError("PERMUTATION_N_REPEATS must be > 0.")

    ranking_df = ranking_df.copy()
    ranking_df["permutation_weighted_mean_delta_logloss"] = np.nan
    ranking_df["permutation_mean_delta_logloss"] = np.nan
    ranking_df["permutation_median_delta_logloss"] = np.nan
    ranking_df["permutation_std_delta_logloss"] = np.nan
    ranking_df["permutation_fold_deltas_json"] = pd.Series(
        [np.nan] * len(ranking_df), index=ranking_df.index, dtype=object
    )
    fold_weight_array = resolve_fold_weight_array(folds, fold_weight_by_id)

    candidate_features = ranking_df.loc[
        ranking_df["prescreen_candidate"], "feature"
    ].tolist()
    print(f"ranking | permutation candidates={len(candidate_features)}")
    if not candidate_features:
        return ranking_df

    feature_to_idx = {feature: idx for idx, feature in enumerate(x.columns)}
    fold_deltas_by_feature = {feature: [] for feature in candidate_features}
    candidate_feature_set = set(candidate_features)

    for fold_pos, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        train_idx = fold["train_idx"]
        valid_idx = fold["valid_idx"]

        x_train_raw = x.iloc[train_idx]
        x_valid_raw = x.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        y_valid_np = y_valid.to_numpy(copy=False)
        w_train = sample_weight.iloc[train_idx].to_numpy(dtype=np.float32, copy=False)
        w_valid = sample_weight.iloc[valid_idx].to_numpy(dtype=np.float32, copy=False)

        x_train, x_valid, dropped_all_nan = prepare_fold_features(
            x_train_raw,
            x_valid_raw,
        )

        candidate_positions = {
            feature: idx
            for idx, feature in enumerate(x_valid.columns)
            if feature in candidate_feature_set
        }
        seed_baseline_loglosses = []
        seed_feature_deltas = {feature: [] for feature in candidate_features}

        for seed in RANDOM_SEEDS:
            model = make_estimator(seed)
            fit_model(
                model=model,
                x_train=x_train,
                y_train=y_train,
                w_train=w_train,
                x_valid=x_valid,
                y_valid=y_valid,
                w_valid=w_valid,
            )
            best_iteration = get_best_iteration(model)
            baseline_proba = model.predict_proba(x_valid, num_iteration=best_iteration)
            baseline_logloss = binary_logloss_from_positive_class_proba(
                y_true=y_valid_np,
                y_pred_proba_pos=baseline_proba[:, 1],
                sample_weight=w_valid,
            )
            seed_baseline_loglosses.append(float(baseline_logloss))

            x_valid_work = x_valid.copy()
            for feature in candidate_features:
                feature_pos = candidate_positions.get(feature)
                if feature_pos is None:
                    seed_feature_deltas[feature].append(0.0)
                    continue

                original_values = x_valid_work.iloc[:, feature_pos].to_numpy(copy=True)
                repeat_deltas = []
                for repeat_idx in range(int(PERMUTATION_N_REPEATS)):
                    permuted_values = original_values.copy()
                    rng = np.random.default_rng(
                        make_permutation_seed(
                            fold_id=fold_id,
                            feature_idx=feature_to_idx[feature],
                            repeat_idx=repeat_idx,
                        )
                    )
                    rng.shuffle(permuted_values)
                    x_valid_work.iloc[:, feature_pos] = permuted_values
                    permuted_proba = model.predict_proba(
                        x_valid_work, num_iteration=best_iteration
                    )
                    permuted_logloss = binary_logloss_from_positive_class_proba(
                        y_true=y_valid_np,
                        y_pred_proba_pos=permuted_proba[:, 1],
                        sample_weight=w_valid,
                    )
                    repeat_deltas.append(
                        float(permuted_logloss) - float(baseline_logloss)
                    )

                x_valid_work.iloc[:, feature_pos] = original_values
                seed_feature_deltas[feature].append(float(np.mean(repeat_deltas)))

        print(
            f"ranking | permutation fold={fold_pos}/{len(folds)} "
            f"id={fold_id} baseline_logloss={format_score_for_cli(np.mean(seed_baseline_loglosses))}"
        )
        for feature in candidate_features:
            fold_deltas_by_feature[feature].append(
                float(np.mean(seed_feature_deltas[feature]))
            )

    for feature, fold_deltas in fold_deltas_by_feature.items():
        fold_deltas_arr = np.asarray(fold_deltas, dtype=np.float64)
        feature_mask = ranking_df["feature"] == feature
        ranking_df.loc[
            feature_mask, "permutation_weighted_mean_delta_logloss"
        ] = weighted_mean_vector(fold_deltas_arr, fold_weight_array)
        ranking_df.loc[feature_mask, "permutation_mean_delta_logloss"] = float(
            np.mean(fold_deltas_arr)
        )
        ranking_df.loc[feature_mask, "permutation_median_delta_logloss"] = float(
            np.median(fold_deltas_arr)
        )
        ranking_df.loc[feature_mask, "permutation_std_delta_logloss"] = float(
            np.std(fold_deltas_arr)
        )
        ranking_df.loc[feature_mask, "permutation_fold_deltas_json"] = json.dumps(
            [float(v) for v in fold_deltas]
        )

    return ranking_df


def run_feature_ranking(x, y, sample_weight, folds, fold_weight_by_id):
    ranking_df, fold_rankings, fold_metadata = run_feature_prescreen(
        x=x,
        y=y,
        sample_weight=sample_weight,
        folds=folds,
        fold_weight_by_id=fold_weight_by_id,
    )
    ranking_df = run_permutation_reranking(
        x=x,
        y=y,
        sample_weight=sample_weight,
        folds=folds,
        fold_weight_by_id=fold_weight_by_id,
        ranking_df=ranking_df,
    )
    ranking_df = sort_feature_table_final(ranking_df)
    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1, dtype=np.int32))
    print(
        f"ranking | permutation done ranked_features={int(ranking_df['ranking_stage'].eq('permutation').sum())}\n"
        + format_feature_preview_for_cli(
            label="permutation_top_features",
            features=ranking_df["feature"].tolist(),
            limit=CONSOLE_PREVIEW_FEATURES,
        )
    )
    return ranking_df, fold_rankings, fold_metadata


def build_coarse_k_grid(pool_size):
    pool_size = int(pool_size)
    if pool_size <= 0:
        raise ValueError("pool_size must be > 0 for coarse top-k grid.")

    base = int(COARSE_GRID_BASE)
    min_coarse_k = int(MIN_COARSE_K)
    if base <= 1:
        raise ValueError("COARSE_GRID_BASE must be > 1.")
    if min_coarse_k <= 0:
        raise ValueError("MIN_COARSE_K must be > 0.")

    resolved = {pool_size}
    current = min_coarse_k
    while current < pool_size:
        resolved.add(int(current))
        next_k = int(current) * base
        if next_k <= current:
            raise ValueError("Coarse grid progression must be strictly increasing.")
        current = next_k

    resolved.add(min(min_coarse_k, pool_size))
    return sorted(resolved)


def resolve_max_refinement_rounds(pool_size):
    pool_size = int(pool_size)
    if pool_size <= 0:
        raise ValueError("pool_size must be > 0 for refinement rounds.")

    if MAX_REFINEMENT_ROUNDS is not None:
        resolved = int(MAX_REFINEMENT_ROUNDS)
        if resolved < 0:
            raise ValueError("MAX_REFINEMENT_ROUNDS must be >= 0.")
        return resolved

    return int(np.ceil(np.log2(max(1, pool_size)))) + 2


def resolve_max_sweep_evaluations():
    if MAX_SWEEP_EVALUATIONS is None:
        return None

    resolved = int(MAX_SWEEP_EVALUATIONS)
    if resolved <= 0:
        raise ValueError("MAX_SWEEP_EVALUATIONS must be > 0.")
    return resolved


def score_topk_subset(
    x,
    y,
    sample_weight,
    folds,
    fold_weight_by_id,
    global_feature_order,
    k,
    phase,
):
    k = int(k)
    if k <= 0:
        raise ValueError("k must be > 0.")
    if k > len(global_feature_order):
        raise ValueError("k cannot exceed the global feature pool size.")

    selected_features = list(global_feature_order[:k])
    fold_scores = []
    fold_best_iterations = []
    fold_seed_scores = {}

    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_idx = fold["train_idx"]
        valid_idx = fold["valid_idx"]

        x_train_raw = x.iloc[train_idx][selected_features]
        x_valid_raw = x.iloc[valid_idx][selected_features]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        w_train = sample_weight.iloc[train_idx].to_numpy(dtype=np.float32, copy=False)
        w_valid = sample_weight.iloc[valid_idx].to_numpy(dtype=np.float32, copy=False)

        x_train, x_valid, _ = prepare_fold_features(x_train_raw, x_valid_raw)

        seed_scores = []
        seed_best_iterations = []
        for seed in RANDOM_SEEDS:
            model = make_estimator(seed)
            fit_model(
                model=model,
                x_train=x_train,
                y_train=y_train,
                w_train=w_train,
                x_valid=x_valid,
                y_valid=y_valid,
                w_valid=w_valid,
            )
            best_iteration = get_best_iteration(model)
            y_pred, y_pred_proba = predict_for_scoring(
                model=model,
                x_valid=x_valid,
                best_iteration=best_iteration,
            )
            score_value = score_predictions(
                scorer_cfg=SCORER,
                y_true=y_valid.to_numpy(),
                y_pred=y_pred,
                y_pred_proba=y_pred_proba,
                sample_weight=w_valid,
            )
            seed_scores.append(float(score_value))
            seed_best_iterations.append(float(best_iteration))

        fold_scores.append(float(np.mean(seed_scores)))
        fold_best_iterations.append(float(np.mean(seed_best_iterations)))
        fold_seed_scores[str(fold_id)] = seed_scores

    mean_score = float(np.mean(fold_scores))
    weighted_mean_score = weighted_mean_vector(
        fold_scores,
        resolve_fold_weight_array(folds, fold_weight_by_id),
    )
    std_score = float(np.std(fold_scores))
    selection_base_score = (
        weighted_mean_score
        if is_nontrivial_fold_recency_weighting_enabled()
        else mean_score
    )
    selection_score = float(topk_selection_score(selection_base_score, std_score))

    return {
        "k": k,
        "feature_count": k,
        "mean_score": mean_score,
        "weighted_mean_score": weighted_mean_score,
        "selection_base_score": selection_base_score,
        "std_score": std_score,
        "selection_score": selection_score,
        "fold_scores_json": json.dumps([float(v) for v in fold_scores]),
        "fold_seed_scores_json": json.dumps(fold_seed_scores),
        "mean_best_iteration": float(np.mean(fold_best_iterations)),
        "feature_list_json": json.dumps(selected_features),
        "phase": str(phase),
    }


def pick_best_row(results_df):
    if results_df.empty:
        raise ValueError("No top-K results available.")
    sort_by = [
        "selection_score",
        "selection_base_score",
        "weighted_mean_score",
        "mean_score",
        "std_score",
        "k",
    ]
    if SCORER["greater_is_better"]:
        ascending = [False, False, False, False, True, True]
    else:
        ascending = [True, True, True, True, True, True]
    return (
        results_df.sort_values(sort_by, ascending=ascending, kind="stable")
        .reset_index(drop=True)
        .iloc[0]
    )


def get_midpoint_k(left_k, right_k):
    left_k = int(left_k)
    right_k = int(right_k)
    if right_k <= left_k:
        raise ValueError("right_k must be greater than left_k.")

    if right_k - left_k <= 1:
        return None
    return left_k + ((right_k - left_k) // 2)


def find_refinement_candidates(results_df):
    if results_df.empty:
        return []

    interval_limit = int(MIN_REFINEMENT_INTERVAL)
    if interval_limit < 1:
        raise ValueError("MIN_REFINEMENT_INTERVAL must be >= 1.")

    sorted_df = results_df.sort_values("k", ascending=True, kind="stable").reset_index(
        drop=True
    )
    best_row = pick_best_row(sorted_df)
    best_k = int(best_row["k"])
    best_score = float(best_row["selection_score"])
    best_std = float(best_row["std_score"])
    evaluated_ks = set(sorted_df["k"].astype(int).tolist())

    candidates = set()
    for left_row, right_row in zip(
        sorted_df.iloc[:-1].itertuples(index=False),
        sorted_df.iloc[1:].itertuples(index=False),
    ):
        left_k = int(left_row.k)
        right_k = int(right_row.k)
        if right_k - left_k <= interval_limit:
            continue

        left_ok = is_acceptable(left_row.selection_score, best_score, best_std)
        right_ok = is_acceptable(right_row.selection_score, best_score, best_std)
        interval_contains_best = left_k <= best_k <= right_k
        interval_is_wide = right_k - left_k > interval_limit
        should_refine = (
            (left_ok != right_ok)
            or (left_ok and right_ok and interval_is_wide)
            or interval_contains_best
        )
        if not should_refine:
            continue

        midpoint_k = get_midpoint_k(left_k, right_k)
        if midpoint_k is None or midpoint_k in evaluated_ks:
            continue
        candidates.add(int(midpoint_k))

    return sorted(candidates)


def run_topk_sweep(
    x,
    y,
    sample_weight,
    folds,
    fold_weight_by_id,
    global_feature_order,
):
    pool_size = len(global_feature_order)
    coarse_ks = build_coarse_k_grid(pool_size)
    max_refinement_rounds = resolve_max_refinement_rounds(pool_size)
    max_sweep_evaluations = resolve_max_sweep_evaluations()
    if max_sweep_evaluations is not None and len(coarse_ks) > int(
        max_sweep_evaluations
    ):
        raise ValueError(
            "MAX_SWEEP_EVALUATIONS is smaller than the required coarse grid size."
        )

    rows = []
    seen = set()
    refined_ks = []

    print(f"topk | coarse_grid={coarse_ks}")

    for k in coarse_ks:
        row = score_topk_subset(
            x=x,
            y=y,
            sample_weight=sample_weight,
            folds=folds,
            fold_weight_by_id=fold_weight_by_id,
            global_feature_order=global_feature_order,
            k=k,
            phase="coarse",
        )
        rows.append(row)
        print(
            f"topk | phase=coarse k={int(row['k'])} "
            f"score={format_score_for_cli(row['selection_base_score'])} "
            f"raw={format_score_for_cli(row['mean_score'])} "
            f"std={format_score_for_cli(row['std_score'])} "
            f"select={format_score_for_cli(row['selection_score'])}"
        )
        seen.add(int(k))

    refinement_rounds_completed = 0
    for round_idx in range(1, max_refinement_rounds + 1):
        results_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
        candidates = [
            k for k in find_refinement_candidates(results_df) if int(k) not in seen
        ]
        if not candidates:
            break

        if max_sweep_evaluations is not None and (
            len(rows) + len(candidates) > int(max_sweep_evaluations)
        ):
            print(
                f"topk | stop=max_sweep_evaluations limit={int(max_sweep_evaluations)} "
                f"next_round_candidates={candidates}"
            )
            break

        refinement_rounds_completed = round_idx
        print(f"topk | refine_round={round_idx} candidates={candidates}")
        for k in candidates:
            row = score_topk_subset(
                x=x,
                y=y,
                sample_weight=sample_weight,
                folds=folds,
                fold_weight_by_id=fold_weight_by_id,
                global_feature_order=global_feature_order,
                k=k,
                phase=f"refine_round_{round_idx}",
            )
            rows.append(row)
            refined_ks.append(int(k))
            seen.add(int(k))
            print(
                f"topk | phase=refine_round_{round_idx} k={int(row['k'])} "
                f"score={format_score_for_cli(row['selection_base_score'])} "
                f"raw={format_score_for_cli(row['mean_score'])} "
                f"std={format_score_for_cli(row['std_score'])} "
                f"select={format_score_for_cli(row['selection_score'])}"
            )

    results_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    evaluated_ks = sorted(int(k) for k in seen)
    print(f"topk | evaluated_ks={evaluated_ks}")
    results_df.attrs["sweep_metadata"] = {
        "k_search_strategy": "log_coarse_plus_midpoint_refinement",
        "coarse_grid_base": int(COARSE_GRID_BASE),
        "min_coarse_k": int(MIN_COARSE_K),
        "max_refinement_rounds_used": int(max_refinement_rounds),
        "refinement_rounds_completed": int(refinement_rounds_completed),
        "total_k_evaluations": len(rows),
        "evaluated_ks": evaluated_ks,
        "coarse_ks": [int(k) for k in coarse_ks],
        "refined_ks": sorted(set(int(k) for k in refined_ks)),
    }
    return results_df


def tolerance_value(best_score, best_std):
    mode = str(TOLERANCE_MODE).strip().lower()
    if mode == "abs":
        return float(ABS_TOL)
    if mode == "rel":
        return abs(float(best_score)) * float(REL_TOL)
    if mode == "max_of_abs_rel":
        return max(float(ABS_TOL), abs(float(best_score)) * float(REL_TOL))
    if mode == "best_minus_std":
        return float(best_std)
    raise ValueError(f"Unsupported TOLERANCE_MODE: {TOLERANCE_MODE}")


def is_acceptable(score_value, best_score, best_std):
    tol_value = tolerance_value(best_score, best_std)
    if SCORER["greater_is_better"]:
        return float(score_value) >= float(best_score) - tol_value
    return float(score_value) <= float(best_score) + tol_value


def choose_recommended_row(results_df):
    best_row = pick_best_row(results_df)
    acceptable = results_df[
        results_df.apply(
            lambda row: is_acceptable(
                score_value=row["selection_score"],
                best_score=best_row["selection_score"],
                best_std=best_row["std_score"],
            ),
            axis=1,
        )
    ].copy()
    acceptable = acceptable.sort_values("k", ascending=True, kind="stable").reset_index(
        drop=True
    )
    best_k = int(best_row["k"])
    min_feature_savings = max(0, int(MIN_PLATEAU_FEATURE_SAVINGS))
    if min_feature_savings > 0:
        acceptable_with_savings = acceptable[
            (best_k - acceptable["k"].astype(int)) >= min_feature_savings
        ].reset_index(drop=True)
    else:
        acceptable_with_savings = acceptable

    if acceptable_with_savings.empty:
        recommended_row = best_row
    else:
        recommended_row = acceptable_with_savings.iloc[0]
    print(
        f"plateau | acceptable_candidates={len(acceptable)} "
        f"acceptable_with_min_savings={len(acceptable_with_savings)} "
        f"best_k={int(best_row['k'])} "
        f"best_score={format_score_for_cli(best_row['selection_base_score'])} "
        f"best_raw={format_score_for_cli(best_row['mean_score'])} "
        f"best_select={format_score_for_cli(best_row['selection_score'])} "
        f"recommended_k={int(recommended_row['k'])} "
        f"recommended_score={format_score_for_cli(recommended_row['selection_base_score'])} "
        f"recommended_raw={format_score_for_cli(recommended_row['mean_score'])} "
        f"recommended_select={format_score_for_cli(recommended_row['selection_score'])} "
        f"min_feature_savings={min_feature_savings} "
        f"tolerance={format_score_for_cli(tolerance_value(best_row['selection_score'], best_row['std_score']))}"
    )
    return best_row, recommended_row


def loss_vs_best(best_score, candidate_score):
    if SCORER["greater_is_better"]:
        return float(best_score) - float(candidate_score)
    return float(candidate_score) - float(best_score)


def pct_loss_vs_best(best_score, candidate_score):
    denom = abs(float(best_score))
    if denom <= 1e-12:
        return float("nan")
    return loss_vs_best(best_score, candidate_score) / denom


def write_outputs(
    output_dir,
    filter_report_df,
    ranking_df,
    topk_df,
    summary_payload,
    recommended_features,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    filter_report_df.to_csv(output_dir / "feature_filter_report.csv", index=False)
    ranking_df.to_csv(output_dir / "feature_ranking.csv", index=False)
    topk_df.to_csv(output_dir / "topk_cv_results.csv", index=False)

    txt_path = output_dir / "recommended_features.txt"
    txt_lines = [
        f"created_utc={summary_payload['created_utc']}",
        f"data_path={summary_payload['data_path']}",
        f"target_col={summary_payload['target_col']}",
        f"input_features={summary_payload['input_feature_count']}",
        f"prefiltered_features={summary_payload['prefilter_feature_count']}",
        f"ranking_n_splits={summary_payload['ranking_n_splits']}",
        f"topk_n_splits={summary_payload['topk_n_splits']}",
        f"topk_selection_mode={summary_payload['topk_selection_mode']}",
        f"topk_selection_base_score={summary_payload['topk_selection_base_score']}",
        f"topk_selection_formula={summary_payload['topk_selection_formula']}",
        f"best_k={summary_payload['best_k']}",
        f"best_score={summary_payload['best_score']}",
        f"best_unweighted_score={summary_payload['best_unweighted_score']}",
        f"best_weighted_score={summary_payload['best_weighted_score']}",
        f"best_selection_score={summary_payload['best_selection_score']}",
        f"recommended_k={summary_payload['recommended_k']}",
        f"recommended_score={summary_payload['recommended_score']}",
        f"recommended_unweighted_score={summary_payload['recommended_unweighted_score']}",
        f"recommended_weighted_score={summary_payload['recommended_weighted_score']}",
        f"recommended_selection_score={summary_payload['recommended_selection_score']}",
        f"tolerance_mode={summary_payload['tolerance_mode']}",
        f"min_plateau_feature_savings={summary_payload['min_plateau_feature_savings']}",
        f"selection_statement={summary_payload['selection_statement']}",
        "",
        *recommended_features,
    ]
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    json_path = output_dir / "recommended_features.json"
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print(f"load data | path={DATA_PATH}")
    df = load_dataframe(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")

    df = df[df[TARGET_COL].notna()].copy()
    if df.empty:
        raise ValueError("No rows left after TARGET_COL non-null filtering.")

    (
        df,
        sample_weight,
        sample_weight_source,
        sample_weight_summary,
        row_filter_info,
    ) = filter_rows_by_min_sample_weight(df, context_label="load data")

    raw_input_feature_cols, dropped_non_numeric = resolve_feature_columns(df)
    x_raw = df[raw_input_feature_cols].replace([np.inf, -np.inf], np.nan)
    (
        x_prefilter,
        filter_report_df,
        duplicate_map,
        high_corr_drop_map,
    ) = prefilter_features(x_raw)
    y_raw = df[TARGET_COL]

    y, class_mapping = prepare_binary_target(y_raw)
    ranking_folds = make_walk_forward_folds(
        n_rows=len(df),
        n_splits=RANKING_N_SPLITS,
        test_to_train_ratio=WF_TEST_TO_TRAIN_RATIO,
    )
    topk_folds = make_walk_forward_folds(
        n_rows=len(df),
        n_splits=TOPK_N_SPLITS,
        test_to_train_ratio=WF_TEST_TO_TRAIN_RATIO,
    )
    ranking_fold_weights = build_fold_recency_weights(ranking_folds)
    topk_fold_weights = build_fold_recency_weights(topk_folds)

    print(
        f"load data | rows={len(df)} input_features={len(raw_input_feature_cols)} "
        f"after_prefilter={x_prefilter.shape[1]} "
        f"sample_weight_source={sample_weight_source}"
    )
    print_prefilter_report_cli(filter_report_df)
    print(
        f"cv setup | ranking_folds={len(ranking_folds)} "
        f"topk_folds={len(topk_folds)} "
        f"test_to_train_ratio={WF_TEST_TO_TRAIN_RATIO} "
        f"scorer={SCORER['name']} greater_is_better={SCORER['greater_is_better']}"
    )
    print(
        f"fold weighting | enabled={bool(ENABLE_FOLD_RECENCY_WEIGHTING)} "
        f"active={is_nontrivial_fold_recency_weighting_enabled()} "
        f"mode={FOLD_RECENCY_WEIGHTING_MODE} "
        f"min={float(FOLD_RECENCY_WEIGHT_MIN):.4f} "
        f"max={float(FOLD_RECENCY_WEIGHT_MAX):.4f} "
        f"topk_std=unweighted"
    )

    ranking_df, _, fold_metadata = run_feature_ranking(
        x=x_prefilter,
        y=y,
        sample_weight=sample_weight,
        folds=ranking_folds,
        fold_weight_by_id=ranking_fold_weights,
    )
    global_feature_order = ranking_df["feature"].tolist()

    topk_df = run_topk_sweep(
        x=x_prefilter,
        y=y,
        sample_weight=sample_weight,
        folds=topk_folds,
        fold_weight_by_id=topk_fold_weights,
        global_feature_order=global_feature_order,
    )
    topk_sweep_metadata = dict(topk_df.attrs.get("sweep_metadata", {}))
    best_row, recommended_row = choose_recommended_row(topk_df)
    recommended_k = int(recommended_row["k"])
    recommended_features = global_feature_order[:recommended_k]
    best_k = int(best_row["k"])
    best_score = float(best_row["selection_base_score"])
    best_selection_score = float(best_row["selection_score"])
    recommended_score = float(recommended_row["selection_base_score"])
    recommended_selection_score = float(recommended_row["selection_score"])
    score_loss = loss_vs_best(best_score, recommended_score)
    best_unweighted_score = float(best_row["mean_score"])
    recommended_unweighted_score = float(recommended_row["mean_score"])
    best_weighted_score = float(best_row["weighted_mean_score"])
    recommended_weighted_score = float(recommended_row["weighted_mean_score"])
    unweighted_score_loss = loss_vs_best(
        best_unweighted_score,
        recommended_unweighted_score,
    )
    feature_reduction_ratio = 1.0 - (recommended_k / max(1, len(global_feature_order)))
    if recommended_k == best_k:
        selection_statement = (
            "to jest najlepszy zbior; mniejsze plateau nie dalo wymaganej redukcji cech"
        )
    else:
        selection_statement = (
            "to jest najmniejszy sensowny zbior cech spelniajacy minimalna redukcje"
        )
    print(f"topk | best_k={best_k} recommended_k={recommended_k}")

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / run_timestamp
    print(f"output | dir={output_dir}")

    summary_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(DATA_PATH),
        "target_col": TARGET_COL,
        "scorer": SCORER["name"],
        "ranking_method": (
            "prescreen_used_folds_recency_weighted_mean_gain_"
            "then_permutation_recency_weighted_delta_logloss"
        ),
        "input_feature_count": len(raw_input_feature_cols),
        "prefilter_feature_count": int(x_prefilter.shape[1]),
        "eligible_feature_count": int(ranking_df["eligible_for_selection"].sum()),
        "permutation_top_n": int(PERMUTATION_TOP_N),
        "permutation_n_repeats": int(PERMUTATION_N_REPEATS),
        "permutation_candidate_count": int(ranking_df["prescreen_candidate"].sum()),
        "permutation_ranked_feature_count": int(
            ranking_df["permutation_mean_delta_logloss"].notna().sum()
        ),
        "dropped_non_numeric_features": dropped_non_numeric,
        "duplicate_column_map": duplicate_map,
        "high_corr_drop_map": high_corr_drop_map,
        "sample_weight": {
            "used": bool(USE_SAMPLE_WEIGHTS),
            "source": sample_weight_source,
            **sample_weight_summary,
        },
        "row_filter": row_filter_info,
        "ranking_n_splits": len(ranking_folds),
        "topk_n_splits": len(topk_folds),
        "walk_forward_test_to_train_ratio": float(WF_TEST_TO_TRAIN_RATIO),
        "random_seeds": [int(seed) for seed in RANDOM_SEEDS],
        "fold_recency_weighting": {
            "enabled": bool(ENABLE_FOLD_RECENCY_WEIGHTING),
            "active": bool(is_nontrivial_fold_recency_weighting_enabled()),
            "mode": str(FOLD_RECENCY_WEIGHTING_MODE),
            "min_weight": float(FOLD_RECENCY_WEIGHT_MIN),
            "max_weight": float(FOLD_RECENCY_WEIGHT_MAX),
            "topk_std_score_aggregation": "unweighted",
            "ranking_fold_weights": fold_weight_items_for_summary(
                ranking_folds,
                ranking_fold_weights,
            ),
            "topk_fold_weights": fold_weight_items_for_summary(
                topk_folds,
                topk_fold_weights,
            ),
        },
        "topk_selection_mode": TOPK_SELECTION_MODE,
        "topk_selection_std_coef": float(TOPK_SELECTION_STD_COEF),
        "topk_selection_formula": topk_selection_formula(),
        "topk_selection_base_score": topk_selection_base_score_label(),
        "k_search_strategy": topk_sweep_metadata["k_search_strategy"],
        "coarse_grid_base": int(topk_sweep_metadata["coarse_grid_base"]),
        "min_coarse_k": int(topk_sweep_metadata["min_coarse_k"]),
        "max_refinement_rounds_used": int(
            topk_sweep_metadata["max_refinement_rounds_used"]
        ),
        "total_k_evaluations": int(topk_sweep_metadata["total_k_evaluations"]),
        "evaluated_ks": [int(k) for k in topk_sweep_metadata["evaluated_ks"]],
        "coarse_ks": [int(k) for k in topk_sweep_metadata["coarse_ks"]],
        "refined_ks": [int(k) for k in topk_sweep_metadata["refined_ks"]],
        "refinement_rounds_completed": int(
            topk_sweep_metadata["refinement_rounds_completed"]
        ),
        "best_k": best_k,
        "recommended_k": recommended_k,
        "best_score": best_score,
        "best_unweighted_score": best_unweighted_score,
        "best_weighted_score": best_weighted_score,
        "best_selection_score": best_selection_score,
        "recommended_score": recommended_score,
        "recommended_unweighted_score": recommended_unweighted_score,
        "recommended_weighted_score": recommended_weighted_score,
        "recommended_selection_score": recommended_selection_score,
        "score_loss_vs_best": score_loss,
        "score_loss_pct_vs_best": pct_loss_vs_best(best_score, recommended_score),
        "unweighted_score_loss_vs_best": unweighted_score_loss,
        "unweighted_score_loss_pct_vs_best": pct_loss_vs_best(
            best_unweighted_score,
            recommended_unweighted_score,
        ),
        "feature_reduction_ratio_vs_prefilter": feature_reduction_ratio,
        "tolerance_mode": TOLERANCE_MODE,
        "abs_tol": float(ABS_TOL),
        "rel_tol": float(REL_TOL),
        "min_plateau_feature_savings": int(MIN_PLATEAU_FEATURE_SAVINGS),
        "tolerance_value": tolerance_value(
            best_selection_score, float(best_row["std_score"])
        ),
        "selection_statement": selection_statement,
        "final_feature_list": recommended_features,
        "class_mapping": class_mapping,
        "fold_ranking_metadata": fold_metadata,
    }

    write_outputs(
        output_dir=output_dir,
        filter_report_df=filter_report_df,
        ranking_df=ranking_df,
        topk_df=topk_df,
        summary_payload=summary_payload,
        recommended_features=recommended_features,
    )

    print("summary")
    print(f"input features: {len(raw_input_feature_cols)}")
    print(f"features after prefilter: {x_prefilter.shape[1]}")
    print(
        f"best K by selection metric: {best_k} | "
        f"score={best_score:.8f} raw={best_unweighted_score:.8f} "
        f"weighted={best_weighted_score:.8f} select={best_selection_score:.8f}"
    )
    print(
        f"recommended K and score: {recommended_k} | "
        f"score={recommended_score:.8f} raw={recommended_unweighted_score:.8f} "
        f"weighted={recommended_weighted_score:.8f} "
        f"select={recommended_selection_score:.8f}"
    )
    print(f"delta vs best: {score_loss:.8f}")
    print(f"feature reduction vs prefilter: {feature_reduction_ratio:.2%}")
    print(f"decision: {selection_statement}")
    print(
        format_feature_preview_for_cli(
            label=f"first {CONSOLE_PREVIEW_FEATURES} recommended features",
            features=recommended_features,
            limit=CONSOLE_PREVIEW_FEATURES,
        )
    )
    print(f"artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
