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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Zmieniaj te stale przed uruchomieniem skryptu.
MODEL_ARTIFACT_PATH = Path("data\\models\\20260505_183214\\lgbm_meta_20260505_183214.json")
DATA_PATH_OVERRIDE = None
OUTPUT_ROOT = Path("data/analysis/model_one_way")

SAMPLE_MODE = "recent_uniform"
RECENT_DAYS = 365.0
RECENT_ROWS = 365 * 24 * 60
MAX_SAMPLE_ROWS = 50_000
GRID_POINTS = 25
BIN_COUNT = 25
BATCH_SIZE = 65_536
RANDOM_SEED = 37
LIMIT_FEATURES = 0

SUSPICIOUS_TOP_N = 30
SUSPICIOUS_EDGE_FRACTION = 0.20
SUSPICIOUS_MIN_ABS_PDP_DELTA = 0.001
SUSPICIOUS_MIN_ABS_TARGET_DELTA = 0.005
SUSPICIOUS_MIN_ABS_BASELINE_DELTA = 0.005
SUSPICIOUS_FLAT_PDP_DELTA = 0.0005

TIME_COL = "Opened"
TARGET_COL_OVERRIDE = None
TARGET_COL_FALLBACK = "target_5m_candle_up"
WEIGHT_COL_OVERRIDE = None
WEIGHT_COL_FALLBACK = "target_5m_weight"
USE_SAMPLE_WEIGHT = True
FLOAT_DTYPE = "float32"

PLOT_X_AXIS_MODE = "central_quantile"  # modes: "central_quantile" or "full"
PLOT_X_VISIBLE_QUANTILES = (0.05, 0.95)
PLOT_X_ZOOM_MAX_RANGE_FRACTION = 0.80
PLOT_X_MIN_FINITE_ROWS = 20
PLOT_TARGET_RATE_SECONDARY_AXIS = True
PLOT_PRIMARY_Y_PAD = 0.015
PLOT_TARGET_Y_PAD = 0.030

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


def resolve_data_path(meta):
    if DATA_PATH_OVERRIDE is not None:
        data_path = Path(DATA_PATH_OVERRIDE)
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


def select_sample_indices(data_path, parquet_file, available_columns):
    total_rows = int(parquet_file.metadata.num_rows)
    rng = np.random.default_rng(int(RANDOM_SEED))
    sample_mode = str(SAMPLE_MODE)
    use_recent = sample_mode.startswith("recent_")
    sample_source = "all_rows"
    max_time = None
    cutoff_time = None

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


def build_grid(feature_values, grid_points):
    finite_values = np.asarray(feature_values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.array([], dtype=np.float64)

    unique_values = np.unique(finite_values)
    if unique_values.size <= int(grid_points):
        return unique_values.astype(np.float64, copy=False)

    quantiles = np.linspace(0.0, 1.0, int(grid_points), dtype=np.float64)
    return np.unique(np.quantile(finite_values, quantiles)).astype(
        np.float64,
        copy=False,
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

    unique_values = np.unique(values)
    bins = []
    if unique_values.size <= int(bin_count):
        for value in unique_values:
            mask = values == value
            bins.append((float(value), float(value), mask))
    else:
        edges = np.unique(
            np.quantile(values, np.linspace(0.0, 1.0, int(bin_count) + 1))
        )
        if edges.size < 2:
            mask = np.ones(values.shape[0], dtype=bool)
            bins.append((float(values[0]), float(values[0]), mask))
        else:
            bin_ids = np.searchsorted(edges[1:-1], values, side="right")
            for bin_id in range(edges.size - 1):
                mask = bin_ids == bin_id
                if np.any(mask):
                    bins.append((float(edges[bin_id]), float(edges[bin_id + 1]), mask))

    rows = []
    total_weight = float(np.sum(bin_weights))
    for bin_id, (left, right, mask) in enumerate(bins):
        count = int(np.count_nonzero(mask))
        weights_in_bin = bin_weights[mask]
        weight_sum = float(np.sum(weights_in_bin))
        target_rate = None
        target_count = 0
        if targets is not None:
            target_mask = mask & np.isfinite(targets) & np.isin(targets, [0.0, 1.0])
            target_count = int(np.count_nonzero(target_mask))
            if target_count > 0:
                target_rate = weighted_mean(targets, bin_weights, target_mask)

        rows.append(
            {
                "bin_id": int(bin_id),
                "feature_left": left,
                "feature_right": right,
                "feature_mean": weighted_mean(values, bin_weights, mask),
                "row_count": count,
                "weight_sum": weight_sum,
                "weight_fraction": (
                    float(weight_sum / total_weight) if total_weight > 0.0 else None
                ),
                "baseline_pred_mean": weighted_mean(preds, bin_weights, mask),
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
            "feature_mean": min_pred["feature_mean"],
            "baseline_pred_mean": min_pred["baseline_pred_mean"],
        },
        "max_baseline_pred_bin": {
            "feature_mean": max_pred["feature_mean"],
            "baseline_pred_mean": max_pred["baseline_pred_mean"],
        },
        "min_target_rate_bin": (
            {
                "feature_mean": min_target["feature_mean"],
                "target_rate": min_target["target_rate"],
            }
            if min_target is not None
            else None
        ),
        "max_target_rate_bin": (
            {
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

    endpoint_delta = float(pdp_values[-1] - pdp_values[0])
    if positive_share >= 0.70 and endpoint_delta > 0.0:
        return "mostly_increasing"
    if negative_share >= 0.70 and endpoint_delta < 0.0:
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


def _format_delta_pp(value):
    if value is None or not np.isfinite(float(value)):
        return "brak danych"
    return f"{float(value) * 100.0:+.2f} pp"


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


def _edge_mean(points, use_weights):
    if not points:
        return None
    if not use_weights:
        return float(sum(point[1] for point in points) / len(points))

    total_weight = float(sum(point[2] for point in points))
    if total_weight <= 0.0:
        return None
    return float(sum(point[1] * point[2] for point in points) / total_weight)


def calculate_edge_delta(
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

        weight = None
        if weight_key is not None:
            weight = _finite_number_or_none(row.get(weight_key))
            if weight is None or weight <= 0.0:
                continue

        if require_positive_target_count:
            target_count = _finite_number_or_none(row.get("target_count"))
            if target_count is None or target_count <= 0.0:
                continue

        points.append((feature_value, metric_value, weight))

    if not points:
        return None

    points.sort(key=lambda point: point[0])
    edge_count = max(
        1,
        int(math.ceil(len(points) * float(SUSPICIOUS_EDGE_FRACTION))),
    )
    edge_count = min(edge_count, len(points))
    low_mean = _edge_mean(points[:edge_count], weight_key is not None)
    high_mean = _edge_mean(points[-edge_count:], weight_key is not None)
    if low_mean is None or high_mean is None:
        return None
    return float(high_mean - low_mean)


def build_feature_suspicion_diagnostics(feature_summary):
    one_way = feature_summary.get("one_way", {})
    observed_bins = feature_summary.get("observed_bins", [])

    delta_pdp = calculate_edge_delta(
        one_way.get("grid", []),
        "feature_value",
        "mean_pred",
    )
    delta_baseline = calculate_edge_delta(
        observed_bins,
        "feature_mean",
        "baseline_pred_mean",
        weight_key="weight_sum",
    )
    delta_target = calculate_edge_delta(
        observed_bins,
        "feature_mean",
        "target_rate",
        weight_key="weight_sum",
        require_positive_target_count=True,
    )

    abs_delta_pdp = _abs_or_none(delta_pdp)
    abs_delta_baseline = _abs_or_none(delta_baseline)
    abs_delta_target = _abs_or_none(delta_target)

    pdp_direction = _direction(delta_pdp)
    baseline_direction = _direction(delta_baseline)
    target_direction = _direction(delta_target)

    pdp_target_opposite = (
        pdp_direction != 0
        and target_direction != 0
        and pdp_direction == -target_direction
    )
    pdp_baseline_same_target_opposite = (
        pdp_direction != 0
        and baseline_direction == pdp_direction
        and target_direction == -pdp_direction
    )
    baseline_target_opposite = (
        baseline_direction != 0
        and target_direction != 0
        and baseline_direction == -target_direction
    )

    pdp_has_amplitude = _has_min_abs(delta_pdp, SUSPICIOUS_MIN_ABS_PDP_DELTA)
    baseline_has_amplitude = _has_min_abs(
        delta_baseline,
        SUSPICIOUS_MIN_ABS_BASELINE_DELTA,
    )
    target_has_amplitude = _has_min_abs(
        delta_target,
        SUSPICIOUS_MIN_ABS_TARGET_DELTA,
    )
    pdp_is_flat = (
        abs_delta_pdp is not None
        and abs_delta_pdp <= float(SUSPICIOUS_FLAT_PDP_DELTA)
    )

    if (
        pdp_has_amplitude
        and baseline_has_amplitude
        and target_has_amplitude
        and pdp_baseline_same_target_opposite
    ):
        category = "pdp_and_baseline_vs_target_opposite"
    elif pdp_has_amplitude and target_has_amplitude and pdp_target_opposite:
        category = "pdp_vs_target_opposite"
    elif (
        pdp_is_flat
        and baseline_has_amplitude
        and target_has_amplitude
        and baseline_target_opposite
    ):
        category = "flat_pdp_strong_baseline_mismatch"
    elif pdp_has_amplitude and not target_has_amplitude:
        category = "overconfident_pdp_weak_target"
    else:
        category = "ok_or_unclear"

    suggested_actions = {
        "pdp_vs_target_opposite": "exclude_or_monotonic_constraint_candidate",
        "pdp_and_baseline_vs_target_opposite": "exclude_candidate",
        "flat_pdp_strong_baseline_mismatch": "inspect_interactions_do_not_auto_exclude",
        "overconfident_pdp_weak_target": "monitor_calibration_or_ablation",
        "ok_or_unclear": "none",
    }

    suspicion_score = (
        _number_or_zero(abs_delta_pdp) * 10000.0
        + _number_or_zero(abs_delta_target) * 5000.0
        + _number_or_zero(abs_delta_baseline) * 2000.0
    )
    if pdp_is_flat and baseline_target_opposite:
        suspicion_score += 20.0
    else:
        if pdp_target_opposite:
            suspicion_score += 100.0
        if pdp_baseline_same_target_opposite:
            suspicion_score += 50.0

    return {
        "feature": feature_summary.get("feature"),
        "plot": Path(feature_summary.get("plot_path") or "").name,
        "delta_pdp": delta_pdp,
        "delta_baseline": delta_baseline,
        "delta_target": delta_target,
        "abs_delta_pdp": abs_delta_pdp,
        "abs_delta_baseline": abs_delta_baseline,
        "abs_delta_target": abs_delta_target,
        "pdp_direction": int(pdp_direction),
        "baseline_direction": int(baseline_direction),
        "target_direction": int(target_direction),
        "category": category,
        "suspicion_score": float(suspicion_score),
        "suggested_action": suggested_actions[category],
    }


def _is_very_strong_pdp_target_opposite(diagnostics):
    return (
        diagnostics.get("category") == "pdp_vs_target_opposite"
        and _number_or_zero(diagnostics.get("abs_delta_pdp"))
        >= float(SUSPICIOUS_MIN_ABS_PDP_DELTA) * 2.0
        and _number_or_zero(diagnostics.get("abs_delta_target"))
        >= float(SUSPICIOUS_MIN_ABS_TARGET_DELTA) * 2.0
    )


def build_suspicious_feature_report(feature_summaries):
    diagnostics = [
        build_feature_suspicion_diagnostics(feature_summary)
        for feature_summary in feature_summaries
    ]
    suspicious = [
        row for row in diagnostics if row["category"] != "ok_or_unclear"
    ]
    suspicious.sort(
        key=lambda row: (
            -float(row["suspicion_score"]),
            str(row.get("feature") or ""),
        )
    )
    top = suspicious[: int(SUSPICIOUS_TOP_N)]

    excluded_feature_names_candidate = []
    monotonic_constraint_candidates = {}
    for row in top:
        feature = row.get("feature")
        if not feature:
            continue
        if (
            row.get("suggested_action") == "exclude_candidate"
            or _is_very_strong_pdp_target_opposite(row)
        ):
            excluded_feature_names_candidate.append(feature)
        if (
            row.get("category") == "pdp_vs_target_opposite"
            and int(row.get("target_direction") or 0) != 0
        ):
            monotonic_constraint_candidates[feature] = int(row["target_direction"])

    return {
        "description": (
            "Heuristic ranking of features whose PDP edge direction may conflict with "
            "observed baseline prediction or target_rate edge direction. Review before "
            "excluding; this report uses existing PDP grid and observed bins only."
        ),
        "edge_fraction": float(SUSPICIOUS_EDGE_FRACTION),
        "thresholds": {
            "min_abs_pdp_delta": float(SUSPICIOUS_MIN_ABS_PDP_DELTA),
            "min_abs_target_delta": float(SUSPICIOUS_MIN_ABS_TARGET_DELTA),
            "min_abs_baseline_delta": float(SUSPICIOUS_MIN_ABS_BASELINE_DELTA),
            "flat_pdp_delta": float(SUSPICIOUS_FLAT_PDP_DELTA),
        },
        "top": top,
        "excluded_feature_names_candidate": excluded_feature_names_candidate,
        "monotonic_constraint_candidates": monotonic_constraint_candidates,
    }


def print_suspicious_feature_preview(report):
    top = list(report.get("top", []))[:10]
    print("[one-way] suspicious feature preview top 10:")
    if not top:
        print("[one-way]   no suspicious features")
        return
    for idx, row in enumerate(top, start=1):
        print(
            "[one-way]   "
            f"{idx}. {row.get('feature')} | {row.get('category')} | "
            f"pdp={_format_delta_pp(row.get('delta_pdp'))} "
            f"target={_format_delta_pp(row.get('delta_target'))} "
            f"baseline={_format_delta_pp(row.get('delta_baseline'))} | "
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

    parts = [
        (
            f"Cecha '{feature}' ma krzywa one-way {direction_labels.get(one_way['direction'], one_way['direction'])}. "
            f"Srednia predykcja bazowa na probce to {_format_probability(one_way['baseline_pred_mean'])}; "
            f"PDP zmienia sie od {_format_probability(one_way['min_mean_pred'])} "
            f"do {_format_probability(one_way['max_mean_pred'])}, czyli amplituda "
            f"{_format_delta_pp(one_way['amplitude'])}."
        ),
        (
            f"Najwyzsza srednia predykcja wypada przy wartosci okolo "
            f"{_format_feature_value(one_way['max_grid_value'])}, a najnizsza przy "
            f"{_format_feature_value(one_way['min_grid_value'])}. "
            f"Przejscie od najnizszego do najwyzszego punktu siatki daje "
            f"{_format_delta_pp(one_way['endpoint_delta'])}."
        ),
        (
            f"W probce wartosci skonczone stanowia {stats['finite_count']}/{stats['row_count']} "
            f"wierszy; missing_ratio={stats['missing_ratio']:.4f}."
        ),
    ]

    if observed.get("max_target_rate_bin") is not None:
        parts.append(
            "W binach obserwacyjnych najwyzszy target_rate jest przy sredniej wartosci "
            f"{_format_feature_value(observed['max_target_rate_bin']['feature_mean'])} "
            f"i wynosi {_format_probability(observed['max_target_rate_bin']['target_rate'])}; "
            "najnizszy target_rate jest przy sredniej wartosci "
            f"{_format_feature_value(observed['min_target_rate_bin']['feature_mean'])} "
            f"i wynosi {_format_probability(observed['min_target_rate_bin']['target_rate'])}."
        )
    else:
        parts.append("Target rate nie zostal policzony, bo target jest niedostepny lub pusty w probce.")

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


def build_plot_x_axis(feature_values, grid_values):
    finite_values = _finite_float_values(feature_values)
    finite_grid = _finite_float_values(grid_values)
    if finite_values.size == 0:
        return {
            "mode": "empty",
            "zoomed": False,
            "visible_min": None,
            "visible_max": None,
            "full_min": None,
            "full_max": None,
            "clipped_sample_rows": 0,
            "clipped_sample_ratio": 0.0,
            "clipped_grid_points": 0,
        }

    full_min = float(np.min(finite_values))
    full_max = float(np.max(finite_values))
    full_limits = _padded_limits(full_min, full_max, pad_fraction=0.04)
    if (
        PLOT_X_AXIS_MODE == "full"
        or finite_values.size < int(PLOT_X_MIN_FINITE_ROWS)
        or full_limits is None
    ):
        return {
            "mode": "full",
            "zoomed": False,
            "visible_min": full_limits[0] if full_limits else full_min,
            "visible_max": full_limits[1] if full_limits else full_max,
            "full_min": full_min,
            "full_max": full_max,
            "clipped_sample_rows": 0,
            "clipped_sample_ratio": 0.0,
            "clipped_grid_points": 0,
        }

    q_low, q_high = PLOT_X_VISIBLE_QUANTILES
    q_low = float(q_low)
    q_high = float(q_high)
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("PLOT_X_VISIBLE_QUANTILES must satisfy 0 <= low < high <= 1.")

    visible_min = float(np.quantile(finite_values, q_low))
    visible_max = float(np.quantile(finite_values, q_high))
    clipped_mask = (finite_values < visible_min) | (finite_values > visible_max)
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
        "clipped_sample_rows": clipped_sample_rows,
        "clipped_sample_ratio": float(clipped_sample_rows / finite_values.size),
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


def _set_probability_axis_limits(ax, values, *, pad, clamp=True):
    finite_values = _finite_float_values(values)
    if finite_values.size == 0:
        return
    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    limits = _padded_limits(y_min, y_max, pad_fraction=0.0)
    if limits is None:
        return
    y_min, y_max = limits
    y_min -= float(pad)
    y_max += float(pad)
    if clamp:
        y_min = max(0.0, y_min)
        y_max = min(1.0, y_max)
    if y_max > y_min:
        ax.set_ylim(y_min, y_max)


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
        bin_x = [row["feature_mean"] for row in observed_bins]
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
                [row["feature_mean"] for row in target_bins],
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
                    "delta_vs_baseline": float(mean_pred - baseline_mean),
                }
            )

    observed_bins = build_observed_bins(
        values,
        baseline_pred,
        target_values,
        weights,
        bin_count=bin_count,
    )

    pdp_values = np.asarray([row["mean_pred"] for row in grid_rows], dtype=np.float64)
    direction = classify_curve(grid_values, pdp_values)
    if pdp_values.size:
        min_idx = int(np.nanargmin(pdp_values))
        max_idx = int(np.nanargmax(pdp_values))
        min_mean_pred = float(pdp_values[min_idx])
        max_mean_pred = float(pdp_values[max_idx])
        min_grid_value = float(grid_values[min_idx])
        max_grid_value = float(grid_values[max_idx])
        endpoint_delta = (
            float(pdp_values[-1] - pdp_values[0]) if pdp_values.size >= 2 else 0.0
        )
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
        endpoint_delta = float("nan")
        amplitude = float("nan")
        mean_abs_step = float("nan")

    if grid_values.size >= 2 and np.nanstd(grid_values) > 0.0 and np.nanstd(pdp_values) > 0.0:
        linear_corr = float(np.corrcoef(grid_values, pdp_values)[0, 1])
    else:
        linear_corr = float("nan")

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
            "grid_point_count": int(len(grid_rows)),
            "baseline_pred_mean": baseline_mean,
            "min_mean_pred": min_mean_pred,
            "max_mean_pred": max_mean_pred,
            "amplitude": amplitude,
            "min_grid_value": min_grid_value,
            "max_grid_value": max_grid_value,
            "endpoint_delta": endpoint_delta,
            "mean_abs_step": mean_abs_step,
            "linear_corr_grid_vs_pdp": linear_corr,
            "direction": direction,
            "grid": grid_rows,
        },
        "observed_bins_summary": summarize_observed_bins(observed_bins),
        "observed_bins": observed_bins,
        "plot_axis": {
            "x": build_plot_x_axis(values, grid_values),
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
    x_axis = feature_summary.get("plot_axis", {}).get("x", {})

    return {
        "f": feature_summary["feature"],
        "plot": Path(feature_summary["plot_path"]).name,
        "base": _compact_number(one_way["baseline_pred_mean"]),
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
                _compact_number(row["feature_mean"]),
                _compact_number(row["baseline_pred_mean"]),
                _compact_number(row["target_rate"]),
                int(row["row_count"]),
                int(row["target_count"]),
            ]
            for row in observed_bins
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
    if int(BATCH_SIZE) <= 0:
        raise ValueError("BATCH_SIZE must be > 0.")
    if PLOT_X_AXIS_MODE not in {"central_quantile", "full"}:
        raise ValueError("PLOT_X_AXIS_MODE must be 'central_quantile' or 'full'.")
    if not isinstance(PLOT_TARGET_RATE_SECONDARY_AXIS, bool):
        raise ValueError("PLOT_TARGET_RATE_SECONDARY_AXIS must be a boolean.")


def main():
    validate_settings()
    run_timestamp = make_utc_run_timestamp()
    artifact = resolve_model_artifact(MODEL_ARTIFACT_PATH)
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
        "suspicious_feature_report": suspicious_feature_report,
        "legend": {
            "f": "feature_name",
            "plot": "png_file_under_plots_dir",
            "base": "sample_baseline_avg_model_probability",
            "x": "plot_x_axis_metadata",
            "pdp": "PDP_avg_model_probability",
            "bins": "baseline_pred_by_bin_and_target_rate_by_bin",
            "delta_pdp": "top-edge PDP mean minus bottom-edge PDP mean",
            "delta_baseline": (
                "top-edge observed baseline prediction minus bottom-edge observed "
                "baseline prediction"
            ),
            "delta_target": "top-edge target_rate minus bottom-edge target_rate",
        },
        "cols": {
            "x": [
                "visible_min",
                "visible_max",
                "zoomed_0_1",
                "hidden_rows",
                "hidden_row_ratio",
                "hidden_pdp_points",
            ],
            "pdp": ["feature_value", "model_probability"],
            "bins": [
                "feature_left",
                "feature_right",
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
