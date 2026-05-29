import hashlib
import json
import math
import re
import textwrap
from datetime import timezone
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common_config_utils import coerce_path, path_to_portable_str
from optuna_run_utils import make_utc_run_timestamp, sanitize_run_name
from project_config import load_runtime_artifact_paths
from target_weights import TARGET_WEIGHT_COL, TARGET_WEIGHT_DECISION_VALUE

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# MODEL_ARTIFACT_PATH=None uses configs/runtime/active.json artifacts.model_meta_path.
MODEL_ARTIFACT_PATH = None
DATA_PATH_OVERRIDE = None
OUTPUT_ROOT = Path("data/analysis/model_one_way")

SAMPLE_MODE = "recent_uniform"
RECENT_DAYS = 365.0
RECENT_ROWS = 365 * 24 * 60
MAX_SAMPLE_ROWS = 250_000
GRID_POINTS = 25
BIN_COUNT = 25
MIN_ONE_WAY_GROUP_ROWS = 2
BATCH_SIZE = 65_536
RANDOM_SEED = 37
LIMIT_FEATURES = 0

SUSPICIOUS_TOP_N = 30
SUSPICIOUS_MIN_ABS_PDP_SLOPE = 0.0005
SUSPICIOUS_MIN_ABS_BASELINE_SLOPE = 0.0005
SUSPICIOUS_MIN_ABS_TARGET_SLOPE = 0.003
SUSPICIOUS_MIN_ABS_KENDALL = 0.10

TIME_COL = "Opened"
TARGET_COL_OVERRIDE = None
TARGET_COL_FALLBACK = "target_5m_candle_up"
WEIGHT_COL_OVERRIDE = None
WEIGHT_COL_FALLBACK = TARGET_WEIGHT_COL
USE_SAMPLE_WEIGHT = True
DECISION_ROWS_ONLY = True
MIN_DECISION_WEIGHT = TARGET_WEIGHT_DECISION_VALUE
FLOAT_DTYPE = "float32"

PLOT_X_AXIS_MODE = "full"  # modes: "central_quantile" or "full"
PLOT_X_VISIBLE_QUANTILES = (0.05, 0.95)
PLOT_X_ZOOM_MAX_RANGE_FRACTION = 0.80
PLOT_X_MIN_FINITE_ROWS = 20
PLOT_TARGET_RATE_SECONDARY_AXIS = True
PLOT_PROBABILITY_CENTER = 0.5
PLOT_PRIMARY_Y_PAD = 0.015
PLOT_TARGET_Y_PAD = 0.030
TARGET_LOESS_FRAC = 0.45
TARGET_LOESS_MIN_POINTS = 5
TARGET_LOESS_CONFIDENCE_Z = 1.96

PLOT_COLORS = {
    "pdp": "#0072B2",
    "sample_baseline": "#5f6368",
    "bin_pred": "#009E73",
    "target_rate": "#D55E00",
    "grid": "#d0d7de",
    "annotation": "#555555",
}

SUMMARY_FLOAT_SIGNIFICANT_DIGITS = 6

SUPPORTED_SAMPLE_MODES = (
    "recent_uniform",
    "recent_random",
    "all_uniform",
    "all_random",
)


def _read_json_object(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _find_sibling_meta(path):
    path = Path(path)
    candidates = list(path.parent.glob("lgbm_meta_*.json"))
    if not candidates:
        return None

    timestamp_match = re.search(r"(\d{8}_\d{6})", path.name)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
        matching = [candidate for candidate in candidates if timestamp in candidate.name]
        if matching:
            return matching[0]

    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def _find_sibling_model(meta_path):
    meta_path = Path(meta_path)
    timestamp_match = re.search(r"(\d{8}_\d{6})", meta_path.name)
    candidates = list(meta_path.parent.glob("lgbm_*.txt"))
    if not candidates:
        return None
    if timestamp_match:
        timestamp = timestamp_match.group(1)
        matching = [candidate for candidate in candidates if timestamp in candidate.name]
        if matching:
            return matching[0]
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def resolve_model_artifact(model_artifact):
    artifact_path = Path(model_artifact)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    if artifact_path.is_dir():
        meta_candidates = list(artifact_path.glob("lgbm_meta_*.json"))
        if meta_candidates:
            artifact_path = max(
                meta_candidates,
                key=lambda candidate: candidate.stat().st_mtime,
            )
        else:
            model_candidates = list(artifact_path.glob("lgbm_*.txt"))
            if not model_candidates:
                raise FileNotFoundError(
                    f"No lgbm_meta_*.json or lgbm_*.txt found in {artifact_path}"
                )
            artifact_path = max(
                model_candidates,
                key=lambda candidate: candidate.stat().st_mtime,
            )

    meta_path = None
    meta = None
    model_path = None

    if artifact_path.suffix.lower() == ".json":
        payload = _read_json_object(artifact_path)
        runtime_artifacts = payload.get("artifacts")
        if (
            isinstance(runtime_artifacts, dict)
            and "model_meta_path" in runtime_artifacts
            and "feature_columns" not in payload
        ):
            return resolve_model_artifact(coerce_path(runtime_artifacts["model_meta_path"]))

        meta_path = artifact_path
        meta = payload
        final_model_path = None
        if isinstance(runtime_artifacts, dict):
            final_model_path = runtime_artifacts.get("final_model_path")
        if final_model_path:
            model_path = coerce_path(final_model_path)
        else:
            model_path = _find_sibling_model(meta_path)
    elif artifact_path.suffix.lower() == ".txt":
        model_path = artifact_path
        meta_path = _find_sibling_meta(model_path)
        meta = _read_json_object(meta_path) if meta_path is not None else None
    else:
        raise ValueError(
            "Unsupported model artifact. Expected .json metadata, .txt LightGBM model, "
            f"or a run directory: {artifact_path}"
        )

    if model_path is None or not Path(model_path).exists():
        raise FileNotFoundError(
            "Could not resolve LightGBM model file from artifact. "
            "Pass lgbm_*.txt directly or use metadata with artifacts.final_model_path."
        )

    return {
        "model_path": Path(model_path),
        "meta_path": Path(meta_path) if meta_path is not None else None,
        "meta": meta or {},
    }


def resolve_configured_model_artifact_path():
    if MODEL_ARTIFACT_PATH is not None:
        return Path(MODEL_ARTIFACT_PATH)
    return load_runtime_artifact_paths()["model_meta_path"]


def resolve_data_path(meta):
    if DATA_PATH_OVERRIDE is not None:
        data_path = coerce_path(DATA_PATH_OVERRIDE)
    else:
        raw_data_path = meta.get("data_path")
        if not raw_data_path:
            raise ValueError(
                "DATA_PATH_OVERRIDE is required when model metadata does not contain data_path."
            )
        data_path = coerce_path(raw_data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Modeling dataset not found: {data_path}")
    return data_path


def resolve_feature_columns(booster, meta):
    feature_columns = meta.get("feature_columns")
    if isinstance(feature_columns, list) and feature_columns:
        features = [str(feature) for feature in feature_columns]
    else:
        features = [str(feature) for feature in booster.feature_name()]

    expected_count = int(booster.num_feature())
    if len(features) != expected_count:
        raise ValueError(
            "Feature count mismatch between metadata and model. "
            f"metadata_or_booster_features={len(features)} model_features={expected_count}"
        )
    return features


def resolve_aux_columns(meta, available_columns):
    target_col = TARGET_COL_OVERRIDE or str(meta.get("target_col") or TARGET_COL_FALLBACK)
    weight_col = WEIGHT_COL_OVERRIDE
    if weight_col is None:
        sample_weight_meta = meta.get("sample_weight")
        if isinstance(sample_weight_meta, dict):
            weight_col = sample_weight_meta.get("column")
    weight_col = str(weight_col or WEIGHT_COL_FALLBACK)

    aux_columns = []
    for column in (TIME_COL, target_col, weight_col):
        if column and column in available_columns and column not in aux_columns:
            aux_columns.append(column)

    return {
        "time_col": TIME_COL,
        "target_col": target_col,
        "weight_col": weight_col,
        "aux_columns": aux_columns,
    }


def _iso_or_none(value):
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if hasattr(value, "tzinfo"):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
    return value.isoformat()


def _select_uniform(values, max_rows):
    if max_rows <= 0 or len(values) <= max_rows:
        return values
    positions = np.linspace(0, len(values) - 1, int(max_rows), dtype=np.int64)
    return values[positions]


def _select_random(values, max_rows, rng):
    if max_rows <= 0 or len(values) <= max_rows:
        return values
    chosen = rng.choice(values, size=int(max_rows), replace=False)
    chosen.sort()
    return chosen


def select_sample_indices(
    data_path,
    parquet_file,
    available_columns,
    *,
    decision_rows_only=False,
    decision_weight_col=None,
    min_decision_weight=None,
):
    total_rows = int(parquet_file.metadata.num_rows)
    rng = np.random.default_rng(int(RANDOM_SEED))
    sample_mode = str(SAMPLE_MODE)
    use_recent = sample_mode.startswith("recent_")
    sample_source = "all_rows"
    max_time = None
    cutoff_time = None
    decision_filter_summary = {"enabled": bool(decision_rows_only)}

    if use_recent:
        eligible_indices = None
        if (
            float(RECENT_DAYS) > 0.0
            and TIME_COL
            and TIME_COL in available_columns
        ):
            time_frame = pd.read_parquet(data_path, columns=[TIME_COL])
            times = pd.to_datetime(time_frame[TIME_COL], errors="coerce")
            valid_times = times.notna()
            if bool(valid_times.any()):
                max_time = times.loc[valid_times].max()
                cutoff_time = max_time - pd.Timedelta(days=float(RECENT_DAYS))
                mask = valid_times & (times >= cutoff_time)
                eligible_indices = np.flatnonzero(mask.to_numpy()).astype(np.int64)
                sample_source = "recent_days"

        if eligible_indices is None or len(eligible_indices) == 0:
            recent_rows = max(1, int(RECENT_ROWS))
            start_idx = max(0, total_rows - recent_rows)
            eligible_indices = np.arange(start_idx, total_rows, dtype=np.int64)
            sample_source = "recent_rows"
    else:
        eligible_indices = np.arange(total_rows, dtype=np.int64)

    if decision_rows_only:
        if not decision_weight_col:
            raise ValueError("Decision row filtering requires a sample weight column.")
        if decision_weight_col not in available_columns:
            raise ValueError(
                "Decision row filtering requires dataset column "
                f"{decision_weight_col!r}."
            )

        if min_decision_weight is None:
            min_decision_weight = TARGET_WEIGHT_DECISION_VALUE
        threshold = float(min_decision_weight)
        weight_frame = pd.read_parquet(data_path, columns=[decision_weight_col])
        decision_weights = pd.to_numeric(
            weight_frame[decision_weight_col],
            errors="coerce",
        ).to_numpy(dtype=np.float64, copy=False)
        if decision_weights.shape[0] != total_rows:
            raise ValueError(
                "Decision weight row count mismatch: "
                f"weights={decision_weights.shape[0]} parquet_rows={total_rows}"
            )

        decision_indices = np.flatnonzero(
            np.isfinite(decision_weights) & (decision_weights >= threshold)
        ).astype(np.int64)
        eligible_rows_before_filter = int(len(eligible_indices))
        eligible_indices = np.intersect1d(
            eligible_indices,
            decision_indices,
            assume_unique=True,
        )
        sample_source = f"{sample_source}_decision_rows"
        decision_filter_summary = {
            "enabled": True,
            "weight_col": str(decision_weight_col),
            "min_weight": threshold,
            "decision_rows_total": int(len(decision_indices)),
            "eligible_rows_before_filter": eligible_rows_before_filter,
            "eligible_rows_after_filter": int(len(eligible_indices)),
            "eligible_rows_removed": int(
                eligible_rows_before_filter - len(eligible_indices)
            ),
        }
        if len(eligible_indices) == 0:
            raise ValueError(
                "Decision row filtering selected zero rows: "
                f"{decision_weight_col}>={threshold}."
            )

    if sample_mode.endswith("_random"):
        sample_indices = _select_random(
            eligible_indices,
            int(MAX_SAMPLE_ROWS),
            rng,
        )
    else:
        sample_indices = _select_uniform(
            eligible_indices,
            int(MAX_SAMPLE_ROWS),
        )
    sample_indices = np.unique(sample_indices.astype(np.int64))

    if len(sample_indices) == 0:
        raise ValueError("Sampling selected zero rows.")

    return sample_indices, {
        "mode": sample_mode,
        "source": sample_source,
        "total_rows": total_rows,
        "eligible_rows": int(len(eligible_indices)),
        "sample_rows": int(len(sample_indices)),
        "max_sample_rows": int(MAX_SAMPLE_ROWS),
        "decision_row_filter": decision_filter_summary,
        "recent_days": float(RECENT_DAYS),
        "recent_rows": int(RECENT_ROWS),
        "time_col": TIME_COL if TIME_COL in available_columns else None,
        "max_time": _iso_or_none(max_time),
        "cutoff_time": _iso_or_none(cutoff_time),
        "first_row_position": int(sample_indices[0]),
        "last_row_position": int(sample_indices[-1]),
        "random_seed": int(RANDOM_SEED),
    }


def _row_group_bounds(parquet_file):
    bounds = []
    start = 0
    for row_group_idx in range(parquet_file.num_row_groups):
        row_count = int(parquet_file.metadata.row_group(row_group_idx).num_rows)
        end = start + row_count
        bounds.append((row_group_idx, start, end))
        start = end
    return bounds


def read_parquet_rows_by_position(
    data_path,
    *,
    columns,
    row_indices,
    batch_size,
):
    parquet_file = pq.ParquetFile(data_path)
    row_indices = np.unique(np.asarray(row_indices, dtype=np.int64))
    frames = []

    for row_group_idx, group_start, group_end in _row_group_bounds(parquet_file):
        left = int(np.searchsorted(row_indices, group_start, side="left"))
        right = int(np.searchsorted(row_indices, group_end, side="left"))
        if left >= right:
            continue

        group_indices = row_indices[left:right]
        batch_start = group_start
        for batch in parquet_file.iter_batches(
            batch_size=int(batch_size),
            columns=columns,
            row_groups=[row_group_idx],
        ):
            batch_end = batch_start + int(batch.num_rows)
            batch_left = int(np.searchsorted(group_indices, batch_start, side="left"))
            batch_right = int(np.searchsorted(group_indices, batch_end, side="left"))
            if batch_left < batch_right:
                local_positions = group_indices[batch_left:batch_right] - batch_start
                table = pa.Table.from_batches([batch]).take(
                    pa.array(local_positions, type=pa.int64())
                )
                frames.append(table.to_pandas())
            batch_start = batch_end

    if not frames:
        raise ValueError("No sampled rows were read from parquet.")

    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != len(row_indices):
        raise ValueError(
            "Sample row read count mismatch: "
            f"read={len(frame)} expected={len(row_indices)}"
        )
    frame.insert(0, "__row_position", row_indices)
    return frame


def prepare_feature_matrix(sample_frame, feature_columns, dtype_name):
    x = sample_frame.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan)
    x = x.apply(pd.to_numeric, errors="coerce")
    dtype = np.float32 if dtype_name == "float32" else np.float64
    return np.ascontiguousarray(x.to_numpy(dtype=dtype, copy=True))


def resolve_sample_weights(sample_frame, weight_col, *, use_sample_weight):
    if not use_sample_weight or weight_col not in sample_frame.columns:
        return np.ones(len(sample_frame), dtype=np.float64), {
            "used": False,
            "source": "unit_weights",
            "column": None,
        }

    weights = pd.to_numeric(sample_frame[weight_col], errors="coerce").to_numpy(
        dtype=np.float64,
        copy=False,
    )
    valid = np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return np.ones(len(sample_frame), dtype=np.float64), {
            "used": False,
            "source": "unit_weights_invalid_dataset_column",
            "column": weight_col,
        }
    if not np.all(valid):
        replacement = float(np.nanmedian(weights[valid]))
        weights = weights.copy()
        weights[~valid] = replacement

    return weights.astype(np.float64, copy=False), {
        "used": True,
        "source": "dataset_column",
        "column": weight_col,
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "mean": float(np.mean(weights)),
        "sum": float(np.sum(weights)),
    }


def resolve_target_values(sample_frame, target_col):
    if target_col not in sample_frame.columns:
        return None
    return pd.to_numeric(sample_frame[target_col], errors="coerce").to_numpy(
        dtype=np.float64,
        copy=False,
    )


def predict_proba(booster, x):
    predictions = np.asarray(booster.predict(x), dtype=np.float64)
    if predictions.ndim == 2:
        if predictions.shape[1] < 2:
            predictions = predictions[:, 0]
        else:
            predictions = predictions[:, 1]
    return predictions.reshape(-1)


def weighted_mean(values, weights=None, mask=None):
    values = np.asarray(values, dtype=np.float64)
    if mask is None:
        mask = np.isfinite(values)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if weights is None:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(values[mask]))

    weights = np.asarray(weights, dtype=np.float64)
    mask = mask & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _equal_count_group_indices(values, group_count):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return []

    group_count = max(1, min(int(group_count), int(values.size)))
    order = np.argsort(values, kind="mergesort")
    return [
        chunk.astype(np.int64, copy=False)
        for chunk in np.array_split(order, group_count)
        if chunk.size > 0
    ]


def _merge_small_adjacent_groups(values, groups, min_group_rows):
    min_group_rows = max(1, int(min_group_rows))
    if min_group_rows <= 1 or len(groups) <= 1:
        return groups

    groups = list(groups)
    while len(groups) > 1:
        small_group_idx = next(
            (idx for idx, group in enumerate(groups) if group.size < min_group_rows),
            None,
        )
        if small_group_idx is None:
            break

        if small_group_idx == 0:
            merge_into_idx = 1
        elif small_group_idx == len(groups) - 1:
            merge_into_idx = small_group_idx - 1
        else:
            group_values = values[groups[small_group_idx]]
            previous_values = values[groups[small_group_idx - 1]]
            next_values = values[groups[small_group_idx + 1]]
            previous_gap = float(np.min(group_values) - np.max(previous_values))
            next_gap = float(np.min(next_values) - np.max(group_values))
            merge_into_idx = (
                small_group_idx - 1
                if previous_gap <= next_gap
                else small_group_idx + 1
            )

        if merge_into_idx < small_group_idx:
            groups[merge_into_idx] = np.concatenate(
                [groups[merge_into_idx], groups[small_group_idx]]
            )
            del groups[small_group_idx]
        else:
            groups[small_group_idx] = np.concatenate(
                [groups[small_group_idx], groups[merge_into_idx]]
            )
            del groups[merge_into_idx]

    return groups


def _tie_aware_equal_count_group_indices(values, group_count, *, min_group_rows=1):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return []

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    run_starts = np.r_[
        0,
        np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1,
    ].astype(np.int64)
    run_ends = np.r_[run_starts[1:], sorted_values.size].astype(np.int64)
    unique_count = int(run_starts.size)
    group_count = max(1, min(int(group_count), unique_count))

    if unique_count <= group_count:
        groups = [
            order[int(start) : int(end)].astype(np.int64, copy=False)
            for start, end in zip(run_starts, run_ends)
        ]
        return _merge_small_adjacent_groups(values, groups, min_group_rows)

    groups = []
    start_run = 0
    remaining_rows = int(values.size)
    remaining_groups = int(group_count)
    while remaining_groups > 0 and start_run < unique_count:
        if remaining_groups == 1:
            end_run = unique_count
        else:
            max_end_run = unique_count - (remaining_groups - 1)
            target_rows = remaining_rows / float(remaining_groups)
            end_run = start_run + 1
            best_size = int(run_ends[end_run - 1] - run_starts[start_run])
            best_error = abs(float(best_size) - target_rows)
            for candidate_end_run in range(start_run + 2, max_end_run + 1):
                candidate_size = int(
                    run_ends[candidate_end_run - 1] - run_starts[start_run]
                )
                candidate_error = abs(float(candidate_size) - target_rows)
                if candidate_error > best_error:
                    break
                end_run = candidate_end_run
                best_size = candidate_size
                best_error = candidate_error

        group = order[int(run_starts[start_run]) : int(run_ends[end_run - 1])]
        groups.append(group)
        remaining_rows -= int(group.size)
        remaining_groups -= 1
        start_run = end_run

    groups = [group.astype(np.int64, copy=False) for group in groups if group.size > 0]
    return _merge_small_adjacent_groups(values, groups, min_group_rows)


def build_grid(feature_values, grid_points):
    finite_values = np.asarray(feature_values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.array([], dtype=np.float64)

    group_values = [
        float(np.median(finite_values[group_indices]))
        for group_indices in _tie_aware_equal_count_group_indices(
            finite_values,
            grid_points,
            min_group_rows=MIN_ONE_WAY_GROUP_ROWS,
        )
    ]
    return np.unique(np.asarray(group_values, dtype=np.float64)).astype(
        np.float64, copy=False
    )


def build_feature_quantiles(feature_values):
    finite_values = np.asarray(feature_values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return {}
    return {
        f"p{int(q * 100):02d}": float(np.quantile(finite_values, q))
        for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    }


def build_observed_bins(
    feature_values,
    baseline_pred,
    target_values,
    weights,
    *,
    bin_count,
):
    feature_values = np.asarray(feature_values, dtype=np.float64)
    baseline_pred = np.asarray(baseline_pred, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite_mask = (
        np.isfinite(feature_values)
        & np.isfinite(baseline_pred)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not np.any(finite_mask):
        return []

    values = feature_values[finite_mask]
    preds = baseline_pred[finite_mask]
    bin_weights = weights[finite_mask]
    targets = (
        np.asarray(target_values, dtype=np.float64)[finite_mask]
        if target_values is not None
        else None
    )

    rows = []
    total_weight = float(np.sum(bin_weights))
    groups = _tie_aware_equal_count_group_indices(
        values,
        bin_count,
        min_group_rows=MIN_ONE_WAY_GROUP_ROWS,
    )
    for bin_id, group_indices in enumerate(groups):
        group_values = values[group_indices]
        group_preds = preds[group_indices]
        weights_in_bin = bin_weights[group_indices]
        count = int(group_indices.size)
        weight_sum = float(np.sum(weights_in_bin))
        target_rate = None
        target_count = 0
        if targets is not None:
            group_targets = targets[group_indices]
            target_mask = np.isfinite(group_targets) & np.isin(group_targets, [0.0, 1.0])
            target_count = int(np.count_nonzero(target_mask))
            if target_count > 0:
                target_rate = weighted_mean(group_targets, weights_in_bin, target_mask)

        rows.append(
            {
                "bin_id": int(bin_id),
                "feature_left": float(np.min(group_values)),
                "feature_right": float(np.max(group_values)),
                "feature_center": float(np.median(group_values)),
                "feature_mean": weighted_mean(group_values, weights_in_bin),
                "row_count": count,
                "weight_sum": weight_sum,
                "weight_fraction": (
                    float(weight_sum / total_weight) if total_weight > 0.0 else None
                ),
                "baseline_pred_mean": weighted_mean(group_preds, weights_in_bin),
                "target_rate": target_rate,
                "target_count": target_count,
            }
        )
    return rows


def summarize_observed_bins(observed_bins):
    valid_target_bins = [
        row for row in observed_bins if row.get("target_rate") is not None
    ]
    if not observed_bins:
        return {
            "bin_count": 0,
            "min_baseline_pred_bin": None,
            "max_baseline_pred_bin": None,
            "min_target_rate_bin": None,
            "max_target_rate_bin": None,
        }

    min_pred = min(observed_bins, key=lambda row: row["baseline_pred_mean"])
    max_pred = max(observed_bins, key=lambda row: row["baseline_pred_mean"])
    min_target = (
        min(valid_target_bins, key=lambda row: row["target_rate"])
        if valid_target_bins
        else None
    )
    max_target = (
        max(valid_target_bins, key=lambda row: row["target_rate"])
        if valid_target_bins
        else None
    )
    return {
        "bin_count": int(len(observed_bins)),
        "min_baseline_pred_bin": {
            "feature_center": min_pred["feature_center"],
            "feature_mean": min_pred["feature_mean"],
            "baseline_pred_mean": min_pred["baseline_pred_mean"],
        },
        "max_baseline_pred_bin": {
            "feature_center": max_pred["feature_center"],
            "feature_mean": max_pred["feature_mean"],
            "baseline_pred_mean": max_pred["baseline_pred_mean"],
        },
        "min_target_rate_bin": (
            {
                "feature_center": min_target["feature_center"],
                "feature_mean": min_target["feature_mean"],
                "target_rate": min_target["target_rate"],
            }
            if min_target is not None
            else None
        ),
        "max_target_rate_bin": (
            {
                "feature_center": max_target["feature_center"],
                "feature_mean": max_target["feature_mean"],
                "target_rate": max_target["target_rate"],
            }
            if max_target is not None
            else None
        ),
    }


def classify_curve(grid_values, pdp_values):
    grid_values = np.asarray(grid_values, dtype=np.float64)
    pdp_values = np.asarray(pdp_values, dtype=np.float64)
    valid = np.isfinite(grid_values) & np.isfinite(pdp_values)
    grid_values = grid_values[valid]
    pdp_values = pdp_values[valid]
    if pdp_values.size < 2:
        return "single_point"

    amplitude = float(np.max(pdp_values) - np.min(pdp_values))
    if amplitude < 0.001:
        return "flat"

    diffs = np.diff(pdp_values)
    step_threshold = max(0.0001, amplitude * 0.05)
    significant = np.abs(diffs) >= step_threshold
    if np.any(significant):
        positive_share = float(np.mean(diffs[significant] > 0.0))
        negative_share = float(np.mean(diffs[significant] < 0.0))
    else:
        positive_share = 0.0
        negative_share = 0.0

    endpoint_change = float(pdp_values[-1] - pdp_values[0])
    if positive_share >= 0.70 and endpoint_change > 0.0:
        return "mostly_increasing"
    if negative_share >= 0.70 and endpoint_change < 0.0:
        return "mostly_decreasing"

    peak_idx = int(np.argmax(pdp_values))
    trough_idx = int(np.argmin(pdp_values))
    interior_peak = 0 < peak_idx < pdp_values.size - 1
    interior_trough = 0 < trough_idx < pdp_values.size - 1
    edge_mean = float((pdp_values[0] + pdp_values[-1]) / 2.0)
    if interior_peak and float(pdp_values[peak_idx] - edge_mean) >= step_threshold:
        return "inverted_u"
    if interior_trough and float(edge_mean - pdp_values[trough_idx]) >= step_threshold:
        return "u_shape"
    return "mixed"


def _format_probability(value):
    if value is None or not np.isfinite(float(value)):
        return "brak danych"
    return f"{float(value):.4f}"


def _format_probability_change_pp(value, *, signed=True):
    if value is None or not np.isfinite(float(value)):
        return "brak danych"
    number = float(value) * 100.0
    return f"{number:+.2f} pp" if signed else f"{number:.2f} pp"


def _format_feature_value(value):
    if value is None or not np.isfinite(float(value)):
        return "brak danych"
    return f"{float(value):.6g}"


def _finite_number_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _abs_or_none(value):
    number = _finite_number_or_none(value)
    return abs(number) if number is not None else None


def _number_or_zero(value):
    number = _finite_number_or_none(value)
    return number if number is not None else 0.0


def _direction(value):
    number = _finite_number_or_none(value)
    if number is None or number == 0.0:
        return 0
    return 1 if number > 0.0 else -1


def _has_min_abs(value, threshold):
    number = _finite_number_or_none(value)
    return number is not None and abs(number) >= float(threshold)


def _series_points(
    rows,
    feature_value_key,
    metric_key,
    *,
    weight_key=None,
    require_positive_target_count=False,
):
    points = []
    for row in rows or []:
        feature_value = _finite_number_or_none(row.get(feature_value_key))
        metric_value = _finite_number_or_none(row.get(metric_key))
        if feature_value is None or metric_value is None:
            continue

        weight = 1.0
        if weight_key is not None:
            weight = _finite_number_or_none(row.get(weight_key))
            if weight is None or weight <= 0.0:
                continue

        if require_positive_target_count:
            target_count = _finite_number_or_none(row.get("target_count"))
            if target_count is None or target_count <= 0.0:
                continue

        points.append((feature_value, metric_value, weight))

    points.sort(key=lambda point: point[0])
    return points


def _weighted_slope(points):
    if len(points) < 2:
        return None
    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    w = np.asarray([point[2] for point in points], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    if np.count_nonzero(valid) < 2:
        return None
    x = x[valid]
    y = y[valid]
    w = w[valid]
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_range = x_max - x_min
    if not math.isfinite(x_range) or x_range <= 0.0:
        return None
    x_norm = (x - x_min) / x_range
    x_mean = float(np.average(x_norm, weights=w))
    y_mean = float(np.average(y, weights=w))
    x_centered = x_norm - x_mean
    denominator = float(np.sum(w * x_centered * x_centered))
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    numerator = float(np.sum(w * x_centered * (y - y_mean)))
    slope = numerator / denominator
    return slope if math.isfinite(slope) else None


def _kendall_tau_b(points):
    if len(points) < 3:
        return None
    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        return None
    x = x[valid]
    y = y[valid]
    if float(np.max(x) - np.min(x)) <= 0.0:
        return None
    if float(np.max(y) - np.min(y)) <= 0.0:
        return None

    concordant = 0
    discordant = 0
    tied_x = 0
    tied_y = 0
    for left_idx in range(len(x) - 1):
        dx = x[left_idx + 1 :] - x[left_idx]
        dy = y[left_idx + 1 :] - y[left_idx]
        tied_both_mask = (dx == 0.0) & (dy == 0.0)
        tied_x_mask = (dx == 0.0) & ~tied_both_mask
        tied_y_mask = (dy == 0.0) & ~tied_both_mask
        comparable = ~(tied_both_mask | tied_x_mask | tied_y_mask)

        tied_x += int(np.count_nonzero(tied_x_mask))
        tied_y += int(np.count_nonzero(tied_y_mask))
        products = dx[comparable] * dy[comparable]
        concordant += int(np.count_nonzero(products > 0.0))
        discordant += int(np.count_nonzero(products < 0.0))

    denominator = math.sqrt(
        float(concordant + discordant + tied_x)
        * float(concordant + discordant + tied_y)
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    tau = float((concordant - discordant) / denominator)
    return tau if math.isfinite(tau) else None


def _direction_with_threshold(value, threshold):
    number = _finite_number_or_none(value)
    if number is None or abs(number) < float(threshold):
        return 0
    return 1 if number > 0.0 else -1


def _opposes(left_direction, right_direction):
    return (
        int(left_direction) != 0
        and int(right_direction) != 0
        and int(left_direction) == -int(right_direction)
    )


def _series_thresholds(series_kind):
    if series_kind == "target":
        return {"slope": float(SUSPICIOUS_MIN_ABS_TARGET_SLOPE)}
    if series_kind == "baseline":
        return {"slope": float(SUSPICIOUS_MIN_ABS_BASELINE_SLOPE)}
    return {"slope": float(SUSPICIOUS_MIN_ABS_PDP_SLOPE)}


def _build_direction_metrics(points, series_kind):
    thresholds = _series_thresholds(series_kind)
    weighted_slope = _weighted_slope(points)
    kendall_tau = _kendall_tau_b(points)

    directions = {
        "weighted_slope": _direction_with_threshold(
            weighted_slope,
            thresholds["slope"],
        ),
        "kendall_tau": _direction_with_threshold(
            kendall_tau,
            float(SUSPICIOUS_MIN_ABS_KENDALL),
        ),
    }
    direction_conflict = _opposes(
        directions["weighted_slope"],
        directions["kendall_tau"],
    )
    if direction_conflict:
        robust_direction = 0
    elif directions["kendall_tau"] != 0:
        robust_direction = directions["kendall_tau"]
    else:
        robust_direction = directions["weighted_slope"]

    return {
        "weighted_slope": weighted_slope,
        "kendall_tau": kendall_tau,
        "directions": directions,
        "robust_direction": int(robust_direction),
        "direction_conflict": bool(direction_conflict),
    }


def _core_vote_count(series_metrics, expected_direction):
    directions = series_metrics.get("directions", {})
    return sum(
        1
        for key in ("weighted_slope", "kendall_tau")
        if int(directions.get(key) or 0) == int(expected_direction)
    )


def _series_strength(series_metrics, series_kind):
    thresholds = _series_thresholds(series_kind)
    values = [
        (
            _abs_or_none(series_metrics.get("weighted_slope")),
            thresholds["slope"],
        ),
        (
            _abs_or_none(series_metrics.get("kendall_tau")),
            float(SUSPICIOUS_MIN_ABS_KENDALL),
        ),
    ]
    scaled = [
        float(value) / float(threshold)
        for value, threshold in values
        if value is not None and threshold > 0.0
    ]
    return max(scaled) if scaled else 0.0


def _series_abs_signal(series_metrics):
    value = _abs_or_none(series_metrics.get("weighted_slope"))
    return float(value) if value is not None else 0.0


def _target_has_required_strength(target_metrics):
    return _has_min_abs(
        target_metrics.get("weighted_slope"),
        SUSPICIOUS_MIN_ABS_TARGET_SLOPE,
    ) or _has_min_abs(
        target_metrics.get("kendall_tau"),
        SUSPICIOUS_MIN_ABS_KENDALL,
    )


def _classify_suspicious_feature(pdp_metrics, baseline_metrics, target_metrics):
    target_dir = int(target_metrics["robust_direction"])
    baseline_dir = int(baseline_metrics["robust_direction"])
    pdp_dir = int(pdp_metrics["robust_direction"])

    direction_conflict_warning = any(
        (
            pdp_metrics["direction_conflict"],
            baseline_metrics["direction_conflict"],
            target_metrics["direction_conflict"],
        )
    )
    target_strong = _target_has_required_strength(target_metrics)
    baseline_against_target_votes = _core_vote_count(baseline_metrics, -target_dir)
    pdp_against_target_votes = _core_vote_count(pdp_metrics, -target_dir)

    if (
        target_dir != 0
        and target_strong
        and baseline_dir == -target_dir
        and pdp_dir == -target_dir
        and not direction_conflict_warning
        and baseline_against_target_votes >= 2
        and pdp_against_target_votes >= 2
    ):
        return (
            "robust_pdp_and_baseline_vs_target_conflict",
            "exclude_candidate",
            None,
        )

    if (
        target_dir != 0
        and baseline_dir == target_dir
        and pdp_dir == -target_dir
    ):
        return (
            "robust_pdp_vs_target_baseline_agrees_target",
            "monotonic_constraint_candidate",
            int(target_dir),
        )

    pdp_strength = _series_strength(pdp_metrics, "pdp")
    target_strength = _series_strength(target_metrics, "target")
    pdp_abs_signal = _series_abs_signal(pdp_metrics)
    target_abs_signal = _series_abs_signal(target_metrics)
    pdp_strong = pdp_dir != 0 and pdp_strength >= 1.0
    target_weak = target_dir == 0 or not target_strong
    pdp_much_stronger_than_target = (
        pdp_strong
        and pdp_abs_signal >= max(
            float(SUSPICIOUS_MIN_ABS_PDP_SLOPE) * 4.0,
            target_abs_signal * 3.0,
        )
    )
    any_robust_direction = any(direction != 0 for direction in (target_dir, baseline_dir, pdp_dir))
    robust_zero_count = sum(1 for direction in (target_dir, baseline_dir, pdp_dir) if direction == 0)
    pdp_target_conflict = target_dir != 0 and pdp_dir == -target_dir
    baseline_target_conflict = target_dir != 0 and baseline_dir == -target_dir

    if pdp_strong and (target_weak or pdp_much_stronger_than_target):
        return (
            "strong_pdp_weak_or_smaller_target_signal",
            "monitor_calibration_or_ablation",
            None,
        )

    if direction_conflict_warning and any_robust_direction:
        return (
            "slope_kendall_direction_conflict",
            "inspect_interactions_do_not_auto_exclude",
            None,
        )

    if (
        pdp_target_conflict
        or baseline_target_conflict
        or (robust_zero_count >= 2 and any_robust_direction)
    ):
        return (
            "mixed_or_inconclusive_robust_directions",
            "inspect_interactions_do_not_auto_exclude",
            None,
        )

    return ("none", "none", None)


def _compact_metric(value):
    return _compact_number(value)


def _weighted_mean_or_none(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return None
    return float(np.average(values[valid], weights=weights[valid]))


def _sum_or_none(*values):
    numbers = []
    for value in values:
        number = _finite_number_or_none(value)
        if number is None:
            return None
        numbers.append(number)
    return float(sum(numbers))


def _interpolate_pdp_at_feature_values(grid_rows, feature_values):
    points = _series_points(grid_rows, "feature_value", "mean_pred")
    if not points:
        return None

    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    unique_x = np.unique(x)
    if unique_x.size == 1:
        return np.full_like(feature_values, fill_value=float(y[0]), dtype=np.float64)

    if unique_x.size != x.size:
        unique_y = np.empty(unique_x.size, dtype=np.float64)
        for idx, x_value in enumerate(unique_x):
            unique_y[idx] = float(np.mean(y[x == x_value]))
    else:
        unique_y = y

    return np.interp(
        feature_values,
        unique_x,
        unique_y,
        left=float(unique_y[0]),
        right=float(unique_y[-1]),
    )


def build_target_alignment_metrics(observed_bins, grid_rows):
    target_rows = []
    for row in observed_bins or []:
        feature_center = _finite_number_or_none(row.get("feature_center"))
        target_rate = _finite_number_or_none(row.get("target_rate"))
        baseline_pred = _finite_number_or_none(row.get("baseline_pred_mean"))
        target_count = _finite_number_or_none(row.get("target_count"))
        if (
            feature_center is None
            or target_rate is None
            or baseline_pred is None
            or target_count is None
            or target_count <= 0.0
        ):
            continue
        target_rows.append((feature_center, target_rate, baseline_pred, target_count))

    base = {
        "method": "target_bin_weighted_mean_absolute_error",
        "main_metric": "target_alignment_score",
        "lower_is_better": True,
        "weight": "target_count",
        "pdp_at_target_bin": "linear_interpolation_clamped_to_pdp_grid",
        "point_count": int(len(target_rows)),
        "weight_sum": 0.0,
        "target_alignment_score": None,
        "baseline_target_wmae": None,
        "pdp_target_wmae": None,
    }
    if not target_rows:
        return base

    feature_values = np.asarray([row[0] for row in target_rows], dtype=np.float64)
    target_rates = np.asarray([row[1] for row in target_rows], dtype=np.float64)
    baseline_preds = np.asarray([row[2] for row in target_rows], dtype=np.float64)
    weights = np.asarray([row[3] for row in target_rows], dtype=np.float64)

    baseline_errors = np.abs(baseline_preds - target_rates)
    pdp_preds = _interpolate_pdp_at_feature_values(grid_rows, feature_values)
    pdp_errors = (
        np.abs(pdp_preds - target_rates)
        if pdp_preds is not None
        else np.asarray([], dtype=np.float64)
    )

    base["weight_sum"] = float(np.sum(weights))
    base["baseline_target_wmae"] = _weighted_mean_or_none(baseline_errors, weights)
    if pdp_preds is not None:
        base["pdp_target_wmae"] = _weighted_mean_or_none(pdp_errors, weights)
    base["target_alignment_score"] = _sum_or_none(
        base["pdp_target_wmae"],
        base["baseline_target_wmae"],
    )
    return base


def build_feature_suspicion_diagnostics(feature_summary):
    one_way = feature_summary.get("one_way", {})
    observed_bins = feature_summary.get("observed_bins", [])
    target_alignment = feature_summary.get("target_alignment") or {}

    pdp_metrics = _build_direction_metrics(
        _series_points(one_way.get("grid", []), "feature_value", "mean_pred"),
        "pdp",
    )
    baseline_metrics = _build_direction_metrics(
        _series_points(
            observed_bins,
            "feature_center",
            "baseline_pred_mean",
            weight_key="row_count",
        ),
        "baseline",
    )
    target_metrics = _build_direction_metrics(
        _series_points(
            observed_bins,
            "feature_center",
            "target_rate",
            weight_key="target_count",
            require_positive_target_count=True,
        ),
        "target",
    )

    category, suggested_action, monotonic_constraint = _classify_suspicious_feature(
        pdp_metrics,
        baseline_metrics,
        target_metrics,
    )
    target_dir = int(target_metrics["robust_direction"])
    baseline_dir = int(baseline_metrics["robust_direction"])
    pdp_dir = int(pdp_metrics["robust_direction"])
    direction_conflict_warning = any(
        (
            pdp_metrics["direction_conflict"],
            baseline_metrics["direction_conflict"],
            target_metrics["direction_conflict"],
        )
    )

    row = {
        "feature": feature_summary.get("feature"),
        "plot": Path(feature_summary.get("plot_path") or "").name,
        "category": category,
        "suggested_action": suggested_action,
        "main_metric": "target_alignment_score",
        "target_alignment_score": _compact_metric(
            target_alignment.get("target_alignment_score")
        ),
        "pdp_target_wmae": _compact_metric(target_alignment.get("pdp_target_wmae")),
        "baseline_target_wmae": _compact_metric(
            target_alignment.get("baseline_target_wmae")
        ),
        "target_alignment_points": int(target_alignment.get("point_count") or 0),
        "target_dir": target_dir,
        "baseline_dir": baseline_dir,
        "pdp_dir": pdp_dir,
        "slope_target": _compact_metric(target_metrics["weighted_slope"]),
        "slope_baseline": _compact_metric(baseline_metrics["weighted_slope"]),
        "slope_pdp": _compact_metric(pdp_metrics["weighted_slope"]),
        "kendall_tau_target": _compact_metric(target_metrics["kendall_tau"]),
        "kendall_tau_baseline": _compact_metric(baseline_metrics["kendall_tau"]),
        "kendall_tau_pdp": _compact_metric(pdp_metrics["kendall_tau"]),
        "direction_conflict_warning": bool(direction_conflict_warning),
        "baseline_target_direction_agree": (
            target_dir != 0 and baseline_dir == target_dir
        ),
        "pdp_target_direction_agree": target_dir != 0 and pdp_dir == target_dir,
    }
    if monotonic_constraint is not None:
        row["monotonic_constraint"] = int(monotonic_constraint)
    return row


def build_suspicious_feature_report(feature_summaries):
    diagnostics = [
        build_feature_suspicion_diagnostics(feature_summary)
        for feature_summary in feature_summaries
    ]
    suspicious = [
        row for row in diagnostics if row["suggested_action"] != "none"
    ]
    suspicious.sort(
        key=lambda row: (
            -_number_or_zero(row.get("target_alignment_score")),
            str(row.get("feature") or ""),
        )
    )
    ranked_features = suspicious[: int(SUSPICIOUS_TOP_N)]

    excluded_feature_names_candidate = []
    monotonic_constraint_candidates = {}
    for row in ranked_features:
        feature = row.get("feature")
        if not feature:
            continue
        if row.get("suggested_action") == "exclude_candidate":
            excluded_feature_names_candidate.append(feature)
        if (
            row.get("suggested_action") == "monotonic_constraint_candidate"
            and int(row.get("monotonic_constraint") or 0) != 0
        ):
            monotonic_constraint_candidates[feature] = int(row["monotonic_constraint"])

    return {
        "description": (
            "Ranking of features whose PDP direction may conflict with observed "
            "baseline prediction or target_rate direction. The main feature score is "
            "PDP WMAE plus baseline WMAE to target_rate by observed target bin."
        ),
        "metrics_version": "target_alignment_kendall_v1",
        "main_metric": {
            "name": "target_alignment_score",
            "lower_is_better": True,
            "weight": "target_count",
            "components": ["pdp_target_wmae", "baseline_target_wmae"],
        },
        "ranked_features_sort": "target_alignment_score_desc",
        "thresholds": {
            "min_abs_slope": float(SUSPICIOUS_MIN_ABS_PDP_SLOPE),
            "min_abs_baseline_slope": float(SUSPICIOUS_MIN_ABS_BASELINE_SLOPE),
            "min_abs_target_slope": float(SUSPICIOUS_MIN_ABS_TARGET_SLOPE),
            "min_abs_kendall_tau": float(SUSPICIOUS_MIN_ABS_KENDALL),
        },
        "ranked_features": ranked_features,
        "excluded_feature_names_candidate": excluded_feature_names_candidate,
        "monotonic_constraint_candidates": monotonic_constraint_candidates,
    }


def print_suspicious_feature_preview(report):
    top = list(report.get("ranked_features", []))[:10]
    print("[one-way] suspicious feature preview top 10:")
    if not top:
        print("[one-way]   no suspicious features")
        return
    for idx, row in enumerate(top, start=1):
        print(
            "[one-way]   "
            f"{idx}. {row.get('feature')} | {row.get('category')} | "
            f"score={_format_probability_change_pp(row.get('target_alignment_score'), signed=False)} "
            f"pdp_wmae={_format_probability_change_pp(row.get('pdp_target_wmae'), signed=False)} "
            f"baseline_wmae={_format_probability_change_pp(row.get('baseline_target_wmae'), signed=False)} "
            f"dirs t/b/p={row.get('target_dir')}/{row.get('baseline_dir')}/{row.get('pdp_dir')} | "
            f"{row.get('suggested_action')}"
        )


def build_llm_summary(feature, summary):
    direction_labels = {
        "single_point": "jednopunktowa",
        "flat": "praktycznie plaska",
        "mostly_increasing": "glownie rosnaca",
        "mostly_decreasing": "glownie malejaca",
        "inverted_u": "w ksztalcie odwroconego U",
        "u_shape": "w ksztalcie U",
        "mixed": "mieszana",
    }
    one_way = summary["one_way"]
    stats = summary["feature_stats"]
    observed = summary["observed_bins_summary"]
    alignment = summary.get("target_alignment", {})

    parts = [
        (
            f"Cecha '{feature}' ma krzywa one-way {direction_labels.get(one_way['direction'], one_way['direction'])}. "
            f"Srednia predykcja bazowa na probce to {_format_probability(one_way['baseline_pred_mean'])}; "
            f"PDP zmienia sie od {_format_probability(one_way['min_mean_pred'])} "
            f"do {_format_probability(one_way['max_mean_pred'])}, czyli amplituda "
            f"{_format_probability_change_pp(one_way['amplitude'], signed=False)}."
        ),
        (
            f"Najwyzsza srednia predykcja wypada przy wartosci okolo "
            f"{_format_feature_value(one_way['max_grid_value'])}, a najnizsza przy "
            f"{_format_feature_value(one_way['min_grid_value'])}."
        ),
        (
            f"W probce wartosci skonczone stanowia {stats['finite_count']}/{stats['row_count']} "
            f"wierszy; missing_ratio={stats['missing_ratio']:.4f}."
        ),
    ]

    if observed.get("max_target_rate_bin") is not None:
        parts.append(
            "W binach obserwacyjnych najwyzszy target_rate jest przy srodkowej wartosci "
            f"{_format_feature_value(observed['max_target_rate_bin']['feature_center'])} "
            f"i wynosi {_format_probability(observed['max_target_rate_bin']['target_rate'])}; "
            "najnizszy target_rate jest przy srodkowej wartosci "
            f"{_format_feature_value(observed['min_target_rate_bin']['feature_center'])} "
            f"i wynosi {_format_probability(observed['min_target_rate_bin']['target_rate'])}."
        )
    else:
        parts.append("Target rate nie zostal policzony, bo target jest niedostepny lub pusty w probce.")

    if alignment.get("pdp_target_wmae") is not None:
        parts.append(
            "Glowny score dopasowania do target_rate to suma wazonych srednich bledow bezwzglednych: "
            f"score={_format_probability_change_pp(alignment.get('target_alignment_score'), signed=False)}, "
            f"PDP={_format_probability_change_pp(alignment['pdp_target_wmae'], signed=False)}, "
            f"baseline={_format_probability_change_pp(alignment.get('baseline_target_wmae'), signed=False)}."
        )

    return " ".join(parts)


def make_safe_plot_name(feature, idx):
    stem = sanitize_run_name(feature, default=f"feature_{idx:04d}")
    digest = hashlib.sha1(feature.encode("utf-8")).hexdigest()[:10]
    if len(stem) > 90:
        stem = stem[:90].rstrip("._-")
    return f"{idx:04d}_{stem}_{digest}.png"


def _finite_float_values(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _padded_limits(min_value, max_value, *, pad_fraction):
    min_value = float(min_value)
    max_value = float(max_value)
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return None
    if max_value < min_value:
        min_value, max_value = max_value, min_value
    value_range = max_value - min_value
    if value_range <= 0.0:
        pad = max(abs(min_value) * 0.05, 1e-9)
    else:
        pad = value_range * float(pad_fraction)
    return (float(min_value - pad), float(max_value + pad))


def build_plot_x_axis(feature_values, grid_values, observed_bins=None, target_loess=None):
    finite_values = _finite_float_values(feature_values)
    finite_grid = _finite_float_values(grid_values)
    axis_parts = [finite_grid]
    if observed_bins:
        axis_parts.append(
            _finite_float_values([row.get("feature_center") for row in observed_bins])
        )
    if target_loess:
        axis_parts.append(
            _finite_float_values([row.get("feature_value") for row in target_loess])
        )
    finite_axis_values = (
        np.concatenate([part for part in axis_parts if part.size > 0])
        if any(part.size > 0 for part in axis_parts)
        else finite_values
    )
    if finite_axis_values.size == 0:
        return {
            "mode": "empty",
            "zoomed": False,
            "visible_min": None,
            "visible_max": None,
            "full_min": None,
            "full_max": None,
            "raw_feature_min": None,
            "raw_feature_max": None,
            "clipped_sample_rows": 0,
            "clipped_sample_ratio": 0.0,
            "clipped_grid_points": 0,
        }

    full_min = float(np.min(finite_axis_values))
    full_max = float(np.max(finite_axis_values))
    raw_min = float(np.min(finite_values)) if finite_values.size > 0 else None
    raw_max = float(np.max(finite_values)) if finite_values.size > 0 else None
    full_limits = _padded_limits(full_min, full_max, pad_fraction=0.04)
    if (
        PLOT_X_AXIS_MODE == "full"
        or finite_axis_values.size < int(PLOT_X_MIN_FINITE_ROWS)
        or full_limits is None
    ):
        return {
            "mode": "full",
            "zoomed": False,
            "visible_min": full_limits[0] if full_limits else full_min,
            "visible_max": full_limits[1] if full_limits else full_max,
            "full_min": full_min,
            "full_max": full_max,
            "raw_feature_min": raw_min,
            "raw_feature_max": raw_max,
            "clipped_sample_rows": 0,
            "clipped_sample_ratio": 0.0,
            "clipped_grid_points": 0,
        }

    q_low, q_high = PLOT_X_VISIBLE_QUANTILES
    q_low = float(q_low)
    q_high = float(q_high)
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("PLOT_X_VISIBLE_QUANTILES must satisfy 0 <= low < high <= 1.")

    visible_min = float(np.quantile(finite_axis_values, q_low))
    visible_max = float(np.quantile(finite_axis_values, q_high))
    clipped_mask = (
        (finite_values < visible_min) | (finite_values > visible_max)
        if finite_values.size > 0
        else np.array([], dtype=bool)
    )
    clipped_sample_rows = int(np.count_nonzero(clipped_mask))
    clipped_grid_points = int(
        np.count_nonzero((finite_grid < visible_min) | (finite_grid > visible_max))
    )

    full_range = full_max - full_min
    visible_range = visible_max - visible_min
    should_zoom = clipped_sample_rows > 0
    if full_range > 0.0 and visible_range > 0.0:
        should_zoom = should_zoom and (
            visible_range <= full_range * float(PLOT_X_ZOOM_MAX_RANGE_FRACTION)
        )

    if not should_zoom:
        return {
            "mode": "full",
            "zoomed": False,
            "visible_min": full_limits[0],
            "visible_max": full_limits[1],
            "full_min": full_min,
            "full_max": full_max,
            "raw_feature_min": raw_min,
            "raw_feature_max": raw_max,
            "clipped_sample_rows": 0,
            "clipped_sample_ratio": 0.0,
            "clipped_grid_points": 0,
        }

    zoom_limits = _padded_limits(visible_min, visible_max, pad_fraction=0.06)
    if zoom_limits is None:
        zoom_limits = _padded_limits(visible_min, visible_min, pad_fraction=0.06)

    return {
        "mode": "central_quantile",
        "zoomed": True,
        "visible_min": zoom_limits[0],
        "visible_max": zoom_limits[1],
        "central_quantile_min": visible_min,
        "central_quantile_max": visible_max,
        "central_quantiles": [q_low, q_high],
        "full_min": full_min,
        "full_max": full_max,
        "raw_feature_min": raw_min,
        "raw_feature_max": raw_max,
        "clipped_sample_rows": clipped_sample_rows,
        "clipped_sample_ratio": (
            float(clipped_sample_rows / finite_values.size)
            if finite_values.size > 0
            else 0.0
        ),
        "clipped_grid_points": clipped_grid_points,
    }


def _filter_xy_for_xlim(x_values, y_values, x_axis):
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if x_axis.get("zoomed"):
        mask = (
            mask
            & (x >= float(x_axis["visible_min"]))
            & (x <= float(x_axis["visible_max"]))
        )
    return x[mask].tolist(), y[mask].tolist()


def _set_probability_axis_limits(ax, values, *, pad, clamp=True, center_value=None):
    finite_values = _finite_float_values(values)
    if finite_values.size == 0:
        return
    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    if center_value is not None:
        center = float(center_value)
        if np.isfinite(center):
            y_min = min(y_min, center)
            y_max = max(y_max, center)
    limits = _padded_limits(y_min, y_max, pad_fraction=0.0)
    if limits is None:
        return
    y_min, y_max = limits
    y_min -= float(pad)
    y_max += float(pad)
    if center_value is not None:
        center = float(center_value)
        if np.isfinite(center) and y_min < center < y_max:
            center_distance = max(center - y_min, y_max - center)
            y_min = center - center_distance
            y_max = center + center_distance
    if clamp:
        y_min = max(0.0, y_min)
        y_max = min(1.0, y_max)
    if y_max > y_min:
        ax.set_ylim(y_min, y_max)


def build_target_loess(observed_bins):
    target_bins = [
        row
        for row in observed_bins
        if row.get("target_rate") is not None and int(row.get("target_count") or 0) > 0
    ]
    if len(target_bins) < 3:
        return []

    x = np.asarray([row["feature_center"] for row in target_bins], dtype=np.float64)
    y = np.asarray([row["target_rate"] for row in target_bins], dtype=np.float64)
    weights = np.asarray([row["target_count"] for row in target_bins], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0.0)
    if np.count_nonzero(mask) < 3:
        return []

    x = x[mask]
    y = y[mask]
    weights = weights[mask]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    weights = weights[order]
    if np.unique(x).size < 2:
        return []

    n_points = int(x.size)
    span_count = max(
        int(math.ceil(n_points * float(TARGET_LOESS_FRAC))),
        int(TARGET_LOESS_MIN_POINTS),
        3,
    )
    span_count = min(span_count, n_points)
    z_value = float(TARGET_LOESS_CONFIDENCE_Z)
    rows = []
    for x0 in x:
        distances = np.abs(x - x0)
        nearest = np.argsort(distances, kind="mergesort")[:span_count]
        bandwidth = float(np.max(distances[nearest]))
        if bandwidth > 0.0:
            scaled = np.clip(distances[nearest] / bandwidth, 0.0, 1.0)
            local_weights = weights[nearest] * ((1.0 - scaled**3) ** 3)
        else:
            local_weights = weights[nearest].copy()

        local_mask = np.isfinite(local_weights) & (local_weights > 0.0)
        local_x = x[nearest][local_mask] - float(x0)
        local_y = y[nearest][local_mask]
        local_weights = local_weights[local_mask]
        if local_y.size < 2:
            continue

        design = np.column_stack([np.ones(local_x.size), local_x])
        sqrt_weights = np.sqrt(local_weights)
        if np.unique(local_x).size >= 2:
            weighted_design = design * sqrt_weights[:, None]
            weighted_y = local_y * sqrt_weights
            beta = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
            y_hat = float(beta[0])
            residuals = local_y - (design @ beta)
            xtwx = design.T @ (local_weights[:, None] * design)
            covariance_scale = float(np.linalg.pinv(xtwx)[0, 0])
            dof_offset = 2.0
        else:
            y_hat = float(np.average(local_y, weights=local_weights))
            residuals = local_y - y_hat
            covariance_scale = 1.0 / float(np.sum(local_weights))
            dof_offset = 1.0

        weight_sum = float(np.sum(local_weights))
        weight_square_sum = float(np.sum(local_weights * local_weights))
        effective_n = (
            (weight_sum * weight_sum) / weight_square_sum
            if weight_square_sum > 0.0
            else 1.0
        )
        dof = max(effective_n - dof_offset, 1.0)
        sigma2 = float(np.sum(local_weights * residuals * residuals) / dof)
        se = math.sqrt(max(sigma2 * covariance_scale, 0.0))
        loess_value = float(np.clip(y_hat, 0.0, 1.0))
        ci_low = max(0.0, min(loess_value, y_hat - (z_value * se)))
        ci_high = min(1.0, max(loess_value, y_hat + (z_value * se)))
        rows.append(
            {
                "feature_value": float(x0),
                "target_loess": loess_value,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
    return rows


def plot_feature_one_way(feature, feature_summary, plot_path):
    grid = feature_summary["one_way"]["grid"]
    observed_bins = feature_summary["observed_bins"]
    x_axis = feature_summary.get("plot_axis", {}).get("x", {})
    grid_x = [row["feature_value"] for row in grid]
    grid_y = [row["mean_pred"] for row in grid]
    visible_grid_x, visible_grid_y = _filter_xy_for_xlim(grid_x, grid_y, x_axis)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    lines = []
    labels = []

    (line,) = ax.plot(
        visible_grid_x,
        visible_grid_y,
        color=PLOT_COLORS["pdp"],
        marker="o",
        linewidth=2.2,
        markersize=4.8,
        label="PDP avg model p(up)",
    )
    lines.append(line)
    labels.append(line.get_label())

    baseline_line = ax.axhline(
        feature_summary["one_way"]["baseline_pred_mean"],
        color=PLOT_COLORS["sample_baseline"],
        linestyle="--",
        linewidth=1.4,
        label="sample baseline avg",
    )
    lines.append(baseline_line)
    labels.append(baseline_line.get_label())

    visible_bin_pred_y = []
    target_axis = None
    visible_target_y = []
    if observed_bins:
        bin_x = [row["feature_center"] for row in observed_bins]
        bin_pred = [row["baseline_pred_mean"] for row in observed_bins]
        visible_bin_x, visible_bin_pred_y = _filter_xy_for_xlim(
            bin_x,
            bin_pred,
            x_axis,
        )
        if visible_bin_x:
            (bin_line,) = ax.plot(
                visible_bin_x,
                visible_bin_pred_y,
                color=PLOT_COLORS["bin_pred"],
                marker=".",
                linewidth=1.7,
                markersize=6.0,
                alpha=0.90,
                label="baseline pred by bin",
            )
            lines.append(bin_line)
            labels.append(bin_line.get_label())

        target_bins = [row for row in observed_bins if row["target_rate"] is not None]
        if target_bins:
            target_x, target_y = _filter_xy_for_xlim(
                [row["feature_center"] for row in target_bins],
                [row["target_rate"] for row in target_bins],
                x_axis,
            )
            if target_x:
                if PLOT_TARGET_RATE_SECONDARY_AXIS:
                    target_axis = ax.twinx()
                    target_plot_axis = target_axis
                else:
                    target_plot_axis = ax
                (target_line,) = target_plot_axis.plot(
                    target_x,
                    target_y,
                    color=PLOT_COLORS["target_rate"],
                    marker=".",
                    linewidth=1.7,
                    markersize=6.0,
                    alpha=0.90,
                    label=(
                        "target rate by bin (right axis)"
                        if PLOT_TARGET_RATE_SECONDARY_AXIS
                        else "target rate by bin"
                    ),
                )
                visible_target_y = target_y
                lines.append(target_line)
                labels.append(target_line.get_label())

                target_loess = feature_summary.get("target_loess", [])
                if target_loess:
                    loess_x = np.asarray(
                        [row["feature_value"] for row in target_loess],
                        dtype=np.float64,
                    )
                    loess_y = np.asarray(
                        [row["target_loess"] for row in target_loess],
                        dtype=np.float64,
                    )
                    loess_low = np.asarray(
                        [row["ci_low"] for row in target_loess],
                        dtype=np.float64,
                    )
                    loess_high = np.asarray(
                        [row["ci_high"] for row in target_loess],
                        dtype=np.float64,
                    )
                    loess_mask = (
                        np.isfinite(loess_x)
                        & np.isfinite(loess_y)
                        & np.isfinite(loess_low)
                        & np.isfinite(loess_high)
                    )
                    if x_axis.get("zoomed"):
                        loess_mask = (
                            loess_mask
                            & (loess_x >= float(x_axis["visible_min"]))
                            & (loess_x <= float(x_axis["visible_max"]))
                        )
                    if np.any(loess_mask):
                        loess_x = loess_x[loess_mask]
                        loess_y = loess_y[loess_mask]
                        loess_low = loess_low[loess_mask]
                        loess_high = loess_high[loess_mask]
                        ci_band = target_plot_axis.fill_between(
                            loess_x,
                            loess_low,
                            loess_high,
                            color=PLOT_COLORS["target_rate"],
                            alpha=0.10,
                            linewidth=0.0,
                            label="target LOESS 95% CI",
                        )
                        (loess_line,) = target_plot_axis.plot(
                            loess_x,
                            loess_y,
                            color=PLOT_COLORS["target_rate"],
                            linestyle="--",
                            linewidth=2.0,
                            alpha=0.72,
                            label="target LOESS",
                        )
                        visible_target_y = [
                            *visible_target_y,
                            *loess_y.tolist(),
                            *loess_low.tolist(),
                            *loess_high.tolist(),
                        ]
                        lines.append(ci_band)
                        labels.append(ci_band.get_label())
                        lines.append(loess_line)
                        labels.append(loess_line.get_label())

    primary_y = [
        *visible_grid_y,
        *visible_bin_pred_y,
        feature_summary["one_way"]["baseline_pred_mean"],
    ]
    _set_probability_axis_limits(
        ax,
        primary_y,
        pad=float(PLOT_PRIMARY_Y_PAD),
        clamp=True,
        center_value=float(PLOT_PROBABILITY_CENTER),
    )
    if target_axis is not None:
        target_axis.set_ylabel("target rate", color=PLOT_COLORS["target_rate"])
        target_axis.tick_params(axis="y", colors=PLOT_COLORS["target_rate"])
        target_axis.spines["right"].set_color(PLOT_COLORS["target_rate"])
        _set_probability_axis_limits(
            target_axis,
            visible_target_y,
            pad=float(PLOT_TARGET_Y_PAD),
            clamp=True,
            center_value=float(PLOT_PROBABILITY_CENTER),
        )

    if x_axis.get("visible_min") is not None and x_axis.get("visible_max") is not None:
        ax.set_xlim(float(x_axis["visible_min"]), float(x_axis["visible_max"]))

    ax.set_title(textwrap.fill(feature, width=96), fontsize=10)
    ax.set_xlabel("feature value")
    ax.set_ylabel("model probability")
    ax.grid(True, alpha=0.25)
    ax.legend(lines, labels, loc="best", fontsize=8)
    if x_axis.get("zoomed"):
        q_low, q_high = x_axis.get("central_quantiles", [None, None])
        q_label = ""
        if q_low is not None and q_high is not None:
            q_label = f"p{int(float(q_low) * 100):02d}-p{int(float(q_high) * 100):02d}"
        note = (
            f"x zoom {q_label}; hidden rows="
            f"{x_axis['clipped_sample_rows']} ({x_axis['clipped_sample_ratio']:.1%}), "
            f"hidden PDP pts={x_axis['clipped_grid_points']}"
        )
        ax.text(
            0.01,
            0.015,
            note,
            transform=ax.transAxes,
            fontsize=8,
            color=PLOT_COLORS["annotation"],
            va="bottom",
            ha="left",
        )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def analyze_feature(
    booster,
    x_base,
    baseline_pred,
    target_values,
    weights,
    feature_columns,
    feature_idx,
    *,
    grid_points,
    bin_count,
    plot_path,
):
    feature = feature_columns[feature_idx]
    values = x_base[:, feature_idx].astype(np.float64, copy=False)
    finite_mask = np.isfinite(values)
    missing_mask = ~finite_mask
    grid_values = build_grid(values, grid_points)
    baseline_mean = weighted_mean(baseline_pred, weights)

    grid_rows = []
    if grid_values.size > 0:
        x_work = x_base.copy()
        for feature_value in grid_values:
            x_work[:, feature_idx] = feature_value
            pred = predict_proba(booster, x_work)
            mean_pred = weighted_mean(pred, weights)
            grid_rows.append(
                {
                    "feature_value": float(feature_value),
                    "mean_pred": mean_pred,
                }
            )

    observed_bins = build_observed_bins(
        values,
        baseline_pred,
        target_values,
        weights,
        bin_count=bin_count,
    )
    target_loess = build_target_loess(observed_bins)
    target_alignment = build_target_alignment_metrics(observed_bins, grid_rows)

    pdp_values = np.asarray([row["mean_pred"] for row in grid_rows], dtype=np.float64)
    direction = classify_curve(grid_values, pdp_values)
    if pdp_values.size:
        min_idx = int(np.nanargmin(pdp_values))
        max_idx = int(np.nanargmax(pdp_values))
        min_mean_pred = float(pdp_values[min_idx])
        max_mean_pred = float(pdp_values[max_idx])
        min_grid_value = float(grid_values[min_idx])
        max_grid_value = float(grid_values[max_idx])
        amplitude = float(max_mean_pred - min_mean_pred)
        mean_abs_step = (
            float(np.mean(np.abs(np.diff(pdp_values))))
            if pdp_values.size >= 2
            else 0.0
        )
    else:
        min_mean_pred = float("nan")
        max_mean_pred = float("nan")
        min_grid_value = float("nan")
        max_grid_value = float("nan")
        amplitude = float("nan")
        mean_abs_step = float("nan")

    feature_summary = {
        "feature": feature,
        "plot_path": path_to_portable_str(plot_path),
        "feature_stats": {
            "row_count": int(values.size),
            "finite_count": int(np.count_nonzero(finite_mask)),
            "missing_count": int(np.count_nonzero(missing_mask)),
            "missing_ratio": float(np.mean(missing_mask)) if values.size else 0.0,
            "missing_baseline_pred_mean": weighted_mean(
                baseline_pred,
                weights,
                missing_mask,
            ),
            "quantiles": build_feature_quantiles(values),
        },
        "one_way": {
            "method": "partial_dependence_replace_one_feature_weighted_average",
            "grid_method": "tie_aware_equal_count_group_median_feature_values",
            "grid_point_count": int(len(grid_rows)),
            "baseline_pred_mean": baseline_mean,
            "min_mean_pred": min_mean_pred,
            "max_mean_pred": max_mean_pred,
            "amplitude": amplitude,
            "min_grid_value": min_grid_value,
            "max_grid_value": max_grid_value,
            "mean_abs_step": mean_abs_step,
            "direction": direction,
            "grid": grid_rows,
        },
        "observed_bins_summary": summarize_observed_bins(observed_bins),
        "observed_bins": observed_bins,
        "target_alignment": target_alignment,
        "target_loess": target_loess,
        "plot_axis": {
            "x": build_plot_x_axis(
                values,
                grid_values,
                observed_bins=observed_bins,
                target_loess=target_loess,
            ),
        },
    }
    feature_summary["llm_summary"] = build_llm_summary(feature, feature_summary)
    plot_feature_one_way(feature, feature_summary, plot_path)
    return feature_summary


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return path_to_portable_str(value)
    return value


def _compact_number(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    compact = float(f"{value:.{int(SUMMARY_FLOAT_SIGNIFICANT_DIGITS)}g}")
    if compact.is_integer():
        return int(compact)
    return compact


def build_compact_feature_summary(feature_summary):
    one_way = feature_summary["one_way"]
    observed_bins = feature_summary["observed_bins"]
    target_loess = feature_summary.get("target_loess", [])
    target_alignment = feature_summary.get("target_alignment", {})
    x_axis = feature_summary.get("plot_axis", {}).get("x", {})

    return {
        "f": feature_summary["feature"],
        "plot": Path(feature_summary["plot_path"]).name,
        "base": _compact_number(one_way["baseline_pred_mean"]),
        "align": [
            _compact_number(target_alignment.get("target_alignment_score")),
            _compact_number(target_alignment.get("pdp_target_wmae")),
            _compact_number(target_alignment.get("baseline_target_wmae")),
            int(target_alignment.get("point_count") or 0),
        ],
        "x": [
            _compact_number(x_axis.get("visible_min")),
            _compact_number(x_axis.get("visible_max")),
            1 if bool(x_axis.get("zoomed", False)) else 0,
            int(x_axis.get("clipped_sample_rows") or 0),
            _compact_number(x_axis.get("clipped_sample_ratio") or 0.0),
            int(x_axis.get("clipped_grid_points") or 0),
        ],
        "pdp": [
            [
                _compact_number(row["feature_value"]),
                _compact_number(row["mean_pred"]),
            ]
            for row in one_way["grid"]
        ],
        "bins": [
            [
                _compact_number(row["feature_left"]),
                _compact_number(row["feature_right"]),
                _compact_number(row["feature_center"]),
                _compact_number(row["feature_mean"]),
                _compact_number(row["baseline_pred_mean"]),
                _compact_number(row["target_rate"]),
                int(row["row_count"]),
                int(row["target_count"]),
            ]
            for row in observed_bins
        ],
        "target_loess": [
            [
                _compact_number(row["feature_value"]),
                _compact_number(row["target_loess"]),
                _compact_number(row["ci_low"]),
                _compact_number(row["ci_high"]),
            ]
            for row in target_loess
        ],
    }


def validate_settings():
    if SAMPLE_MODE not in SUPPORTED_SAMPLE_MODES:
        supported = ", ".join(SUPPORTED_SAMPLE_MODES)
        raise ValueError(f"Unsupported SAMPLE_MODE={SAMPLE_MODE!r}. Expected one of: {supported}")
    if FLOAT_DTYPE not in {"float32", "float64"}:
        raise ValueError("FLOAT_DTYPE must be 'float32' or 'float64'.")
    if int(GRID_POINTS) < 2:
        raise ValueError("GRID_POINTS must be >= 2.")
    if int(BIN_COUNT) < 1:
        raise ValueError("BIN_COUNT must be >= 1.")
    if int(MIN_ONE_WAY_GROUP_ROWS) < 1:
        raise ValueError("MIN_ONE_WAY_GROUP_ROWS must be >= 1.")
    if int(BATCH_SIZE) <= 0:
        raise ValueError("BATCH_SIZE must be > 0.")
    if not isinstance(DECISION_ROWS_ONLY, bool):
        raise ValueError("DECISION_ROWS_ONLY must be a boolean.")
    if not math.isfinite(float(MIN_DECISION_WEIGHT)):
        raise ValueError("MIN_DECISION_WEIGHT must be finite.")
    if PLOT_X_AXIS_MODE not in {"central_quantile", "full"}:
        raise ValueError("PLOT_X_AXIS_MODE must be 'central_quantile' or 'full'.")
    if not isinstance(PLOT_TARGET_RATE_SECONDARY_AXIS, bool):
        raise ValueError("PLOT_TARGET_RATE_SECONDARY_AXIS must be a boolean.")
    if not (0.0 < float(TARGET_LOESS_FRAC) <= 1.0):
        raise ValueError("TARGET_LOESS_FRAC must satisfy 0 < frac <= 1.")
    if int(TARGET_LOESS_MIN_POINTS) < 3:
        raise ValueError("TARGET_LOESS_MIN_POINTS must be >= 3.")
    if float(TARGET_LOESS_CONFIDENCE_Z) <= 0.0:
        raise ValueError("TARGET_LOESS_CONFIDENCE_Z must be > 0.")


def main():
    validate_settings()
    run_timestamp = make_utc_run_timestamp()
    artifact = resolve_model_artifact(resolve_configured_model_artifact_path())
    booster = lgb.Booster(model_file=str(artifact["model_path"]))
    meta = artifact["meta"]
    data_path = resolve_data_path(meta)
    model_feature_columns = resolve_feature_columns(booster, meta)
    if int(LIMIT_FEATURES) > 0:
        analyzed_feature_indices = list(range(min(int(LIMIT_FEATURES), len(model_feature_columns))))
    else:
        analyzed_feature_indices = list(range(len(model_feature_columns)))

    parquet_file = pq.ParquetFile(data_path)
    available_columns = set(parquet_file.schema_arrow.names)
    missing_features = [
        feature for feature in model_feature_columns if feature not in available_columns
    ]
    if missing_features:
        preview = ", ".join(missing_features[:10])
        raise ValueError(
            "Dataset is missing model feature columns. "
            f"missing_count={len(missing_features)} preview=[{preview}]"
        )

    aux = resolve_aux_columns(meta, available_columns)
    read_columns = list(dict.fromkeys([*model_feature_columns, *aux["aux_columns"]]))
    sample_indices, sampling_summary = select_sample_indices(
        data_path,
        parquet_file,
        available_columns,
        decision_rows_only=bool(DECISION_ROWS_ONLY),
        decision_weight_col=aux["weight_col"],
        min_decision_weight=float(MIN_DECISION_WEIGHT),
    )

    output_dir = Path(OUTPUT_ROOT) / run_timestamp
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=False)

    print(
        "[one-way] loading sample | "
        f"rows={len(sample_indices)} cols={len(read_columns)} data={data_path}"
    )
    sample_frame = read_parquet_rows_by_position(
        data_path,
        columns=read_columns,
        row_indices=sample_indices,
        batch_size=int(BATCH_SIZE),
    )
    if aux["time_col"] in sample_frame.columns and len(sample_frame) > 0:
        sample_times = pd.to_datetime(sample_frame[aux["time_col"]], errors="coerce")
        sampling_summary["sample_min_time"] = _iso_or_none(sample_times.min())
        sampling_summary["sample_max_time"] = _iso_or_none(sample_times.max())

    x_base = prepare_feature_matrix(sample_frame, model_feature_columns, FLOAT_DTYPE)
    weights, weight_summary = resolve_sample_weights(
        sample_frame,
        aux["weight_col"],
        use_sample_weight=bool(USE_SAMPLE_WEIGHT),
    )
    target_values = resolve_target_values(sample_frame, aux["target_col"])
    baseline_pred = predict_proba(booster, x_base)

    feature_summaries = []
    for analysis_idx, feature_idx in enumerate(analyzed_feature_indices):
        feature = model_feature_columns[feature_idx]
        plot_path = plots_dir / make_safe_plot_name(feature, analysis_idx + 1)
        print(
            "[one-way] feature "
            f"{analysis_idx + 1}/{len(analyzed_feature_indices)} | {feature}"
        )
        feature_summaries.append(
            analyze_feature(
                booster,
                x_base,
                baseline_pred,
                target_values,
                weights,
                model_feature_columns,
                feature_idx,
                grid_points=int(GRID_POINTS),
                bin_count=int(BIN_COUNT),
                plot_path=plot_path,
            )
        )

    suspicious_feature_report = build_suspicious_feature_report(feature_summaries)
    summary_path = output_dir / "one_way_summary.json"
    run_summary = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_path": path_to_portable_str(artifact["model_path"]),
        "model_meta_path": (
            path_to_portable_str(artifact["meta_path"])
            if artifact["meta_path"] is not None
            else None
        ),
        "data_path": path_to_portable_str(data_path),
        "plots_dir": path_to_portable_str(plots_dir),
        "feature_count": int(len(analyzed_feature_indices)),
        "target_col": aux["target_col"] if aux["target_col"] in sample_frame.columns else None,
        "sampling": sampling_summary,
        "sample_weight": weight_summary,
        "suspicious_feature_report": suspicious_feature_report,
        "legend": {
            "f": "feature_name",
            "plot": "png_file_under_plots_dir",
            "base": "sample_baseline_avg_model_probability",
            "align": "target_alignment_pdp_and_baseline_wmae_to_target_rate",
            "x": "plot_x_axis_metadata",
            "pdp": "PDP_avg_model_probability_at_equal_count_group_medians",
            "bins": "baseline_pred_by_bin_and_target_rate_by_bin",
            "target_loess": "LOESS_target_rate_with_approx_95pct_confidence_interval",
        },
        "cols": {
            "align": [
                "target_alignment_score",
                "pdp_target_wmae",
                "baseline_target_wmae",
                "target_alignment_point_count",
            ],
            "x": [
                "visible_min",
                "visible_max",
                "zoomed_0_1",
                "hidden_rows",
                "hidden_row_ratio",
                "hidden_pdp_points",
            ],
            "pdp": ["feature_value", "model_probability"],
            "target_loess": [
                "feature_value",
                "target_loess",
                "ci_low",
                "ci_high",
            ],
            "bins": [
                "feature_left",
                "feature_right",
                "feature_center",
                "feature_mean",
                "model_probability",
                "target_rate",
                "row_count",
                "target_count",
            ],
        },
        "features": [
            build_compact_feature_summary(feature_summary)
            for feature_summary in feature_summaries
        ],
    }
    summary_path.write_text(
        json.dumps(
            _json_safe(run_summary),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[one-way] summary saved: {summary_path}")
    print(f"[one-way] plots saved: {plots_dir}")
    print_suspicious_feature_preview(suspicious_feature_report)


if __name__ == "__main__":
    main()
