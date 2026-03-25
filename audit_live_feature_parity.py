from __future__ import annotations

import copy
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from create_modeling_dataset import (
    add_indicator_values,
    concat_feature_frame,
    parse_fit_results,
)
from features.candle_features import (
    RAW_OHLCV_COLS,
    SUPPORTED_CANDLE_FEATURE_COLS,
    add_candle_derived_features,
    add_candle_streak_features,
    build_latest_candle_derived_feature_dict_fast,
    build_latest_candle_pattern_feature_dict,
    build_latest_candle_streak_feature_dict_fast,
    resolve_candle_derived_feature_cols,
    resolve_candle_pattern_feature_cols,
    resolve_streak_interval_to_rule,
)
from features.session_open_features import (
    add_session_counter_features,
    build_latest_session_counter_feature_dict_fast,
)
from features.volume_profile_fixed_range import (
    FEATURE_VERSION as VP_FEATURE_VERSION,
    AUDIT_ANCHOR_STATE_DIR as VP_AUDIT_ANCHOR_STATE_DIR,
    PSEUDO_LIVE_AUDIT_MODELING_STATE_DIR as VP_PSEUDO_LIVE_AUDIT_MODELING_STATE_DIR,
    PSEUDO_LIVE_AUDIT_RUNTIME_STATE_DIR as VP_PSEUDO_LIVE_AUDIT_RUNTIME_STATE_DIR,
    build_volume_profile_features,
    bootstrap_state_from_history,
    extract_features_from_state,
    load_state as load_volume_profile_state,
    normalize_config as normalize_volume_profile_config,
    save_state as save_volume_profile_state,
    update_state_with_candle as update_volume_profile_state_with_candle,
)
from live_predict_binance import (
    INDICATOR_STABILITY_SUMMARY_PATH,
    INTERVAL,
    KELLY_CONFIG_PATH,
    LIVE_INITIAL_BANKROLL_USDC,
    MODELING_DATASET_SETTINGS,
    MODEL_META_PATH,
    OHLCV_COLS,
    SYMBOL,
    LivePredictor,
    interval_to_timedelta,
    load_kelly_runtime_config,
    load_indicator_specs,
    load_model_and_meta,
    load_required_stable_window,
    parse_target_bucket_minutes,
)
from modeling_dataset_utils import (
    resolve_modeling_dataset_parquet_path,
    split_feature_subset,
)


INTERVAL_DELTA = interval_to_timedelta(INTERVAL)
DEFAULT_AUDIT_DAYS_BACK = 30
DEFAULT_BOOTSTRAP_CANDLES = 20_000
DEFAULT_MAX_KEEP = DEFAULT_BOOTSTRAP_CANDLES
PREDICTION_DIFF_TOL = 1e-12
REL_DIFF_DENOM_FLOOR = 1e-12

# Runtime settings
AUDIT_DAYS_BACK = DEFAULT_AUDIT_DAYS_BACK
AUDIT_BOOTSTRAP_CANDLES = DEFAULT_BOOTSTRAP_CANDLES
AUDIT_MAX_STEPS = 10_080
AUDIT_MAX_KEEP = AUDIT_BOOTSTRAP_CANDLES
AUDIT_MODEL_META_PATH = Path(
    "data\\models\\runs\\20260325_024605\\lgbm_meta_20260325_024605.json"
)
AUDIT_PARQUET_PATH = None
AUDIT_USE_ANCHOR_VP_STATE = True
AUDIT_OVERWRITE_ANCHOR_VP_STATE = False
AUDIT_OUTPUT_DIR = None
AUDIT_DRILLDOWN_FEATURE = None
AUDIT_TOP_N = 50
AUDIT_PROGRESS_ENABLED = True
AUDIT_PROGRESS_EVERY_STEPS = 60


def _ensure_utc_opened(series: pd.Series) -> pd.Series:
    opened = pd.to_datetime(series)
    if getattr(opened.dt, "tz", None) is None:
        return opened.dt.tz_localize("UTC")
    return opened.dt.tz_convert("UTC")


def _naive_utc_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)


def _optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def resolve_anchor_volume_profile_state_path(anchor_candle_opened: pd.Timestamp) -> Path:
    stamp = pd.Timestamp(anchor_candle_opened).strftime("%Y%m%d_%H%M")
    return (
        VP_AUDIT_ANCHOR_STATE_DIR
        / f"{SYMBOL}_{INTERVAL}_{VP_FEATURE_VERSION}_audit_anchor_{stamp}"
    )


def _feature_group_map(feature_columns: list[str]) -> dict[str, str]:
    parts = split_feature_subset(feature_columns)
    group_by_feature: dict[str, str] = {}
    for feature in parts["raw_ohlcv_cols"]:
        group_by_feature[feature] = "raw_ohlcv"
    for feature in parts["candle_feature_cols"]:
        group_by_feature[feature] = "candle"
    for feature in parts["streak_feature_cols"]:
        group_by_feature[feature] = "streak"
    for feature in parts["session_feature_cols"]:
        group_by_feature[feature] = "session"
    for feature in parts["indicator_feature_cols"]:
        group_by_feature[feature] = "indicator"
    for feature in parts["volume_profile_feature_cols"]:
        group_by_feature[feature] = "volume_profile"
    for feature in parts["unclassified_feature_cols"]:
        group_by_feature[feature] = "unclassified"
    return group_by_feature


def _safe_rowwise_mean(values: np.ndarray) -> np.ndarray:
    counts = np.isfinite(values).sum(axis=1)
    sums = np.nansum(values, axis=1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _is_live_decision_opened(opened: pd.Timestamp, bucket_minutes: int) -> bool:
    opened = pd.Timestamp(opened)
    bucket_start = opened.floor(f"{int(bucket_minutes)}min")
    bucket_end = bucket_start + pd.Timedelta(minutes=int(bucket_minutes) - 1)
    return opened == bucket_end


def _safe_colwise_mean(values: np.ndarray) -> np.ndarray:
    counts = np.isfinite(values).sum(axis=0)
    sums = np.nansum(values, axis=0)
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def _safe_colwise_rmse(values: np.ndarray) -> np.ndarray:
    counts = np.isfinite(values).sum(axis=0)
    sums = np.nansum(np.square(values, dtype=np.float64), axis=0)
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = np.sqrt(sums[valid] / counts[valid])
    return out


def _safe_nanmax_axis1(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    safe = np.where(valid, values, -np.inf)
    out = safe.max(axis=1)
    out[~valid.any(axis=1)] = np.nan
    return out.astype(np.float64, copy=False)


def _safe_nanmax_axis0(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    safe = np.where(valid, values, -np.inf)
    out = safe.max(axis=0)
    out[~valid.any(axis=0)] = np.nan
    return out.astype(np.float64, copy=False)


def _safe_relative_diff(
    candidate_values: np.ndarray,
    reference_values: np.ndarray,
    *,
    denom_floor: float = REL_DIFF_DENOM_FLOOR,
) -> np.ndarray:
    candidate = np.asarray(candidate_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    out = np.abs(candidate - reference)
    out /= np.maximum(np.abs(reference), float(denom_floor))
    out[~(np.isfinite(candidate) & np.isfinite(reference))] = np.nan
    return out


def _safe_argmax_axis1(values: np.ndarray) -> np.ndarray:
    safe = np.where(np.isfinite(values), values, -np.inf)
    return safe.argmax(axis=1)


def _safe_argmax_axis0(values: np.ndarray) -> np.ndarray:
    safe = np.where(np.isfinite(values), values, -np.inf)
    return safe.argmax(axis=0)


def _column_values_match(candidate_values: np.ndarray, reference_values: np.ndarray) -> bool:
    candidate = np.asarray(candidate_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    same_mask = (candidate == reference) | (np.isnan(candidate) & np.isnan(reference))
    return bool(same_mask.all())


def _fit_indicator_config_map(feature_columns: list[str]) -> dict[str, dict[str, Any]]:
    fit_results_dir = Path(MODELING_DATASET_SETTINGS["fit_results_dir"])
    configs = parse_fit_results(fit_results_dir)
    selected = set(feature_columns)
    return {cfg["feature_col"]: cfg for cfg in configs if cfg["feature_col"] in selected}


def _feature_builder_frame(feature_columns: list[str]) -> pd.DataFrame:
    feature_parts = split_feature_subset(feature_columns)
    indicator_config_map = _fit_indicator_config_map(feature_columns)
    candle_pattern_cols = set(resolve_candle_pattern_feature_cols(feature_parts["candle_feature_cols"]))

    records: list[dict[str, Any]] = []
    for feature in feature_columns:
        if feature in RAW_OHLCV_COLS:
            records.append(
                {
                    "feature": feature,
                    "builder_family": "raw_ohlcv",
                    "builder_name": "raw_ohlcv_passthrough",
                    "builder_source": None,
                }
            )
            continue

        if feature in feature_parts["streak_feature_cols"]:
            records.append(
                {
                    "feature": feature,
                    "builder_family": "streak",
                    "builder_name": "add_candle_streak_features",
                    "builder_source": None,
                }
            )
            continue

        if feature in feature_parts["session_feature_cols"]:
            records.append(
                {
                    "feature": feature,
                    "builder_family": "session",
                    "builder_name": "add_session_counter_features",
                    "builder_source": None,
                }
            )
            continue

        if feature in feature_parts["volume_profile_feature_cols"]:
            records.append(
                {
                    "feature": feature,
                    "builder_family": "volume_profile",
                    "builder_name": "build_volume_profile_features",
                    "builder_source": "volume_profile_fixed_range",
                }
            )
            continue

        if feature in feature_parts["candle_feature_cols"]:
            builder_name = (
                "build_latest_candle_pattern_feature_dict"
                if feature in candle_pattern_cols
                else "build_latest_candle_derived_feature_dict_fast"
            )
            records.append(
                {
                    "feature": feature,
                    "builder_family": "candle",
                    "builder_name": builder_name,
                    "builder_source": None,
                }
            )
            continue

        cfg = indicator_config_map.get(feature)
        if cfg is not None:
            records.append(
                {
                    "feature": feature,
                    "builder_family": "indicator",
                    "builder_name": str(cfg["indicator"]),
                    "builder_source": str(cfg["json_path"]),
                }
            )
            continue

        records.append(
            {
                "feature": feature,
                "builder_family": "unknown",
                "builder_name": "unknown",
                "builder_source": None,
            }
        )

    return pd.DataFrame.from_records(records)


def _build_single_feature_prediction_impact_report(
    *,
    candidate_label: str,
    reference_label: str,
    candidate_matrix: np.ndarray,
    reference_matrix: np.ndarray,
    candidate_pred: np.ndarray,
    reference_pred: np.ndarray,
    audit_df: pd.DataFrame,
    feature_columns: list[str],
    feature_group_by_name: dict[str, str],
    feature_builder_frame: pd.DataFrame,
    model,
) -> dict[str, Any]:
    row_count, feature_count = candidate_matrix.shape
    if feature_count != len(feature_columns):
        raise ValueError("Feature count mismatch for prediction impact audit.")

    builder_meta = (
        feature_builder_frame.set_index("feature")
        .reindex(feature_columns)
        .reset_index(drop=False)
    )
    feature_names = np.asarray(feature_columns, dtype=object)
    feature_groups = np.asarray(
        [feature_group_by_name.get(col, "unknown") for col in feature_columns],
        dtype=object,
    )
    builder_families = builder_meta["builder_family"].fillna("unknown").to_numpy(dtype=object)
    builder_names = builder_meta["builder_name"].fillna("unknown").to_numpy(dtype=object)
    builder_sources = builder_meta["builder_source"].to_numpy(dtype=object)

    base_abs_gap = np.abs(candidate_pred - reference_pred)
    base_signal_mismatch = (candidate_pred >= 0.5) != (reference_pred >= 0.5)
    base_drift_mask = base_abs_gap > PREDICTION_DIFF_TOL

    fixed_pred_matrix = np.empty((row_count, feature_count), dtype=np.float64)
    proba_shift_matrix = np.empty((row_count, feature_count), dtype=np.float64)
    gap_reduction_matrix = np.empty((row_count, feature_count), dtype=np.float64)
    working_matrix = candidate_matrix.copy()

    feature_rows: list[dict[str, Any]] = []
    for feature_idx, feature_name in enumerate(feature_columns):
        candidate_col = candidate_matrix[:, feature_idx]
        reference_col = reference_matrix[:, feature_idx]
        if _column_values_match(candidate_col, reference_col):
            fixed_pred = candidate_pred
        else:
            working_matrix[:, feature_idx] = reference_col
            fixed_pred = np.asarray(model.predict(working_matrix), dtype=np.float64).reshape(-1)
            if fixed_pred.shape[0] != row_count:
                raise ValueError(
                    f"Prediction length mismatch for feature impact audit: {fixed_pred.shape[0]} != {row_count}"
                )
            working_matrix[:, feature_idx] = candidate_col

        fixed_pred_matrix[:, feature_idx] = fixed_pred
        proba_shift = candidate_pred - fixed_pred
        fixed_abs_gap = np.abs(fixed_pred - reference_pred)
        gap_reduction = base_abs_gap - fixed_abs_gap
        proba_shift_matrix[:, feature_idx] = proba_shift
        gap_reduction_matrix[:, feature_idx] = gap_reduction

        fixed_signal_mismatch = (fixed_pred >= 0.5) != (reference_pred >= 0.5)
        fixed_drift_mask = fixed_abs_gap > PREDICTION_DIFF_TOL
        best_gap_reduction_idx = int(np.argmax(gap_reduction))

        feature_rows.append(
            {
                "feature": feature_name,
                "group": feature_groups[feature_idx],
                "builder_family": builder_families[feature_idx],
                "builder_name": builder_names[feature_idx],
                "builder_source": builder_sources[feature_idx],
                "mean_abs_proba_shift_if_fixed": float(np.mean(np.abs(proba_shift))),
                "max_abs_proba_shift_if_fixed": float(np.max(np.abs(proba_shift))),
                "mean_signed_proba_shift_if_fixed": float(np.mean(proba_shift)),
                "net_abs_proba_gap_reduction_if_fixed": float(np.sum(gap_reduction)),
                "mean_abs_proba_gap_reduction_if_fixed": float(np.mean(gap_reduction)),
                "max_abs_proba_gap_reduction_if_fixed": float(np.max(gap_reduction)),
                "rows_abs_proba_gap_reduced_if_fixed": int(
                    (gap_reduction > PREDICTION_DIFF_TOL).sum()
                ),
                "rows_abs_proba_gap_worsened_if_fixed": int(
                    (gap_reduction < -PREDICTION_DIFF_TOL).sum()
                ),
                "rows_proba_diff_gt_tol_resolved_if_fixed": int(
                    (base_drift_mask & ~fixed_drift_mask).sum()
                ),
                "rows_proba_diff_gt_tol_introduced_if_fixed": int(
                    (~base_drift_mask & fixed_drift_mask).sum()
                ),
                "rows_signal_mismatch_resolved_if_fixed": int(
                    (base_signal_mismatch & ~fixed_signal_mismatch).sum()
                ),
                "rows_signal_mismatch_introduced_if_fixed": int(
                    (~base_signal_mismatch & fixed_signal_mismatch).sum()
                ),
                "worst_gap_reduction_opened": audit_df["Opened"].iloc[best_gap_reduction_idx],
                f"worst_gap_reduction_{candidate_label}_value": float(
                    candidate_col[best_gap_reduction_idx]
                ),
                f"worst_gap_reduction_{reference_label}_value": float(
                    reference_col[best_gap_reduction_idx]
                ),
                f"worst_gap_reduction_{candidate_label}_proba_up": float(
                    candidate_pred[best_gap_reduction_idx]
                ),
                f"worst_gap_reduction_{reference_label}_proba_up": float(
                    reference_pred[best_gap_reduction_idx]
                ),
                "worst_gap_reduction_proba_up_if_fixed": float(
                    fixed_pred[best_gap_reduction_idx]
                ),
            }
        )

    feature_summary_df = (
        pd.DataFrame.from_records(feature_rows)
        .sort_values(
            [
                "rows_signal_mismatch_resolved_if_fixed",
                "mean_abs_proba_gap_reduction_if_fixed",
                "max_abs_proba_gap_reduction_if_fixed",
                "rows_abs_proba_gap_reduced_if_fixed",
                "rows_signal_mismatch_introduced_if_fixed",
                "mean_abs_proba_shift_if_fixed",
                "max_abs_proba_shift_if_fixed",
            ],
            ascending=[False, False, False, False, True, False, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    group_summary_df = (
        feature_summary_df.groupby("group", dropna=False)
        .agg(
            feature_count=("feature", "count"),
            max_mean_abs_proba_shift_if_fixed=("mean_abs_proba_shift_if_fixed", "max"),
            mean_mean_abs_proba_shift_if_fixed=("mean_abs_proba_shift_if_fixed", "mean"),
            max_mean_abs_proba_gap_reduction_if_fixed=(
                "mean_abs_proba_gap_reduction_if_fixed",
                "max",
            ),
            mean_mean_abs_proba_gap_reduction_if_fixed=(
                "mean_abs_proba_gap_reduction_if_fixed",
                "mean",
            ),
            total_rows_abs_proba_gap_reduced_if_fixed=(
                "rows_abs_proba_gap_reduced_if_fixed",
                "sum",
            ),
            total_rows_abs_proba_gap_worsened_if_fixed=(
                "rows_abs_proba_gap_worsened_if_fixed",
                "sum",
            ),
            total_rows_signal_mismatch_resolved_if_fixed=(
                "rows_signal_mismatch_resolved_if_fixed",
                "sum",
            ),
            total_rows_signal_mismatch_introduced_if_fixed=(
                "rows_signal_mismatch_introduced_if_fixed",
                "sum",
            ),
        )
        .sort_values(
            [
                "max_mean_abs_proba_gap_reduction_if_fixed",
                "mean_mean_abs_proba_gap_reduction_if_fixed",
                "max_mean_abs_proba_shift_if_fixed",
                "total_rows_signal_mismatch_resolved_if_fixed",
            ],
            ascending=[False, False, False, False],
            kind="stable",
        )
        .reset_index()
    )

    builder_summary_df = (
        feature_summary_df.groupby(["builder_family", "builder_name"], dropna=False)
        .agg(
            feature_count=("feature", "count"),
            max_mean_abs_proba_shift_if_fixed=("mean_abs_proba_shift_if_fixed", "max"),
            mean_mean_abs_proba_shift_if_fixed=("mean_abs_proba_shift_if_fixed", "mean"),
            max_mean_abs_proba_gap_reduction_if_fixed=(
                "mean_abs_proba_gap_reduction_if_fixed",
                "max",
            ),
            mean_mean_abs_proba_gap_reduction_if_fixed=(
                "mean_abs_proba_gap_reduction_if_fixed",
                "mean",
            ),
            total_rows_abs_proba_gap_reduced_if_fixed=(
                "rows_abs_proba_gap_reduced_if_fixed",
                "sum",
            ),
            total_rows_abs_proba_gap_worsened_if_fixed=(
                "rows_abs_proba_gap_worsened_if_fixed",
                "sum",
            ),
            total_rows_signal_mismatch_resolved_if_fixed=(
                "rows_signal_mismatch_resolved_if_fixed",
                "sum",
            ),
            total_rows_signal_mismatch_introduced_if_fixed=(
                "rows_signal_mismatch_introduced_if_fixed",
                "sum",
            ),
        )
        .sort_values(
            [
                "max_mean_abs_proba_gap_reduction_if_fixed",
                "mean_mean_abs_proba_gap_reduction_if_fixed",
                "max_mean_abs_proba_shift_if_fixed",
                "total_rows_signal_mismatch_resolved_if_fixed",
            ],
            ascending=[False, False, False, False],
            kind="stable",
        )
        .reset_index()
    )

    row_best_idx = _safe_argmax_axis1(gap_reduction_matrix)
    row_best_gap_reduction = gap_reduction_matrix[np.arange(row_count), row_best_idx]
    row_has_helpful_single_feature_fix = row_best_gap_reduction > PREDICTION_DIFF_TOL
    row_summary_df = pd.DataFrame(
        {
            "top_prediction_impact_feature": np.where(
                row_has_helpful_single_feature_fix,
                feature_names[row_best_idx],
                None,
            ),
            "top_prediction_impact_group": np.where(
                row_has_helpful_single_feature_fix,
                feature_groups[row_best_idx],
                None,
            ),
            "top_prediction_impact_builder_family": np.where(
                row_has_helpful_single_feature_fix,
                builder_families[row_best_idx],
                None,
            ),
            "top_prediction_impact_builder_name": np.where(
                row_has_helpful_single_feature_fix,
                builder_names[row_best_idx],
                None,
            ),
            "top_prediction_impact_abs_proba_gap_reduction_if_fixed": np.where(
                row_has_helpful_single_feature_fix,
                row_best_gap_reduction,
                0.0,
            ),
            "top_prediction_impact_abs_proba_shift_if_fixed": np.where(
                row_has_helpful_single_feature_fix,
                np.abs(proba_shift_matrix[np.arange(row_count), row_best_idx]),
                np.nan,
            ),
            "top_prediction_impact_signed_proba_shift_if_fixed": np.where(
                row_has_helpful_single_feature_fix,
                proba_shift_matrix[np.arange(row_count), row_best_idx],
                np.nan,
            ),
            "top_prediction_impact_signal_mismatch_resolved_if_fixed": np.where(
                row_has_helpful_single_feature_fix,
                (
                    base_signal_mismatch
                    & ~(
                        (fixed_pred_matrix[np.arange(row_count), row_best_idx] >= 0.5)
                        != (reference_pred >= 0.5)
                    )
                ),
                False,
            ),
            f"top_prediction_impact_{candidate_label}_value": np.where(
                row_has_helpful_single_feature_fix,
                candidate_matrix[np.arange(row_count), row_best_idx],
                np.nan,
            ),
            f"top_prediction_impact_{reference_label}_value": np.where(
                row_has_helpful_single_feature_fix,
                reference_matrix[np.arange(row_count), row_best_idx],
                np.nan,
            ),
            "top_prediction_impact_proba_up_if_fixed": np.where(
                row_has_helpful_single_feature_fix,
                fixed_pred_matrix[np.arange(row_count), row_best_idx],
                np.nan,
            ),
        }
    )

    top_feature = None if feature_summary_df.empty else feature_summary_df.iloc[0]
    summary = pd.Series(
        {
            "rows_with_helpful_single_feature_fix": int(row_has_helpful_single_feature_fix.sum()),
            "top_prediction_impact_feature": None
            if top_feature is None
            else top_feature["feature"],
            "top_prediction_impact_group": None if top_feature is None else top_feature["group"],
            "top_prediction_impact_builder_family": None
            if top_feature is None
            else top_feature["builder_family"],
            "top_prediction_impact_builder_name": None
            if top_feature is None
            else top_feature["builder_name"],
            "max_mean_abs_proba_shift_if_fixed": 0.0
            if feature_summary_df.empty
            else float(feature_summary_df["mean_abs_proba_shift_if_fixed"].max()),
            "max_mean_abs_proba_gap_reduction_if_fixed": 0.0
            if feature_summary_df.empty
            else float(feature_summary_df["mean_abs_proba_gap_reduction_if_fixed"].max()),
            "max_rows_signal_mismatch_resolved_if_fixed": 0
            if feature_summary_df.empty
            else int(feature_summary_df["rows_signal_mismatch_resolved_if_fixed"].max()),
        }
    )

    return {
        "summary": summary,
        "row_summary_df": row_summary_df,
        "feature_summary_df": feature_summary_df,
        "group_summary_df": group_summary_df,
        "builder_summary_df": builder_summary_df,
        "fixed_pred_matrix": fixed_pred_matrix,
        "proba_shift_matrix": proba_shift_matrix,
        "gap_reduction_matrix": gap_reduction_matrix,
    }


def resolve_recent_history_tail_window(
    *,
    parquet_path: Path,
    audit_end: pd.Timestamp,
    tail_fraction: float,
) -> tuple[pd.Timestamp, int, int]:
    tail_fraction = float(tail_fraction)
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError("tail_fraction must be in (0, 1].")

    opened = _load_opened_series(parquet_path)
    opened = pd.to_datetime(opened)
    audit_end = pd.Timestamp(audit_end)
    eligible = opened.loc[opened <= audit_end]
    if eligible.empty:
        raise ValueError("No parquet rows are available at or before audit_end.")

    total_rows = int(len(eligible))
    keep_rows = int(np.ceil(total_rows * tail_fraction))
    keep_rows = max(1, min(total_rows, keep_rows))
    tail_start = pd.Timestamp(eligible.iloc[total_rows - keep_rows])
    return tail_start, keep_rows, total_rows


def load_modeling_raw_history_frame(
    *,
    parquet_path: Path,
    audit_end: pd.Timestamp,
    history_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    filters = []
    if history_start is not None:
        filters.append(
            (
                "Opened",
                ">=",
                _naive_utc_timestamp(history_start),
            )
        )
    filters.append(
        (
            "Opened",
            "<=",
            _naive_utc_timestamp(audit_end),
        )
    )
    frame = pd.read_parquet(
        parquet_path,
        columns=["Opened", *RAW_OHLCV_COLS],
        filters=filters,
    )
    frame["Opened"] = _ensure_utc_opened(frame["Opened"])
    frame = frame.sort_values("Opened").drop_duplicates(subset=["Opened"]).reset_index(drop=True)
    return frame


def build_current_recomputed_feature_history(
    *,
    raw_history_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    feature_parts = split_feature_subset(feature_columns)
    feature_frame = raw_history_df.loc[:, ["Opened", *RAW_OHLCV_COLS]].copy()

    configured_rules = resolve_streak_interval_to_rule(
        list(MODELING_DATASET_SETTINGS.get("candle_streak_intervals", []))
    )
    if feature_parts["streak_intervals"]:
        missing_streak_intervals = [
            label for label in feature_parts["streak_intervals"] if label not in configured_rules
        ]
        if missing_streak_intervals:
            raise ValueError(
                "Missing configured candle streak intervals for recompute: "
                f"{missing_streak_intervals}"
            )
        feature_frame = add_candle_streak_features(
            feature_frame,
            interval_to_rule={
                label: configured_rules[label] for label in feature_parts["streak_intervals"]
            },
        )

    if feature_parts["candle_feature_cols"]:
        feature_frame = add_candle_derived_features(
            feature_frame,
            feature_cols=feature_parts["candle_feature_cols"],
        )

    if feature_parts["session_feature_cols"]:
        feature_frame = add_session_counter_features(
            feature_frame,
            feature_cols=feature_parts["session_feature_cols"],
        )

    vp_cfg = normalize_volume_profile_config(MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range"))
    if feature_parts["volume_profile_feature_cols"]:
        if not vp_cfg["enabled"]:
            raise ValueError("Volume profile features requested but disabled in modeling config.")
        vp_features_df, _vp_state = build_volume_profile_features(feature_frame, vp_cfg)
        vp_feature_frame = pd.DataFrame(
            {
                feature_col: vp_features_df[feature_col].to_numpy(dtype=np.float64, copy=False)
                for feature_col in feature_parts["volume_profile_feature_cols"]
            },
            index=feature_frame.index,
        )
        feature_frame = concat_feature_frame(
            feature_frame,
            vp_feature_frame,
            context="Current recompute volume profile features",
        )

    if feature_parts["indicator_feature_cols"]:
        indicator_config_map = _fit_indicator_config_map(feature_columns)
        indicator_configs = [
            indicator_config_map[feature]
            for feature in feature_parts["indicator_feature_cols"]
            if feature in indicator_config_map
        ]
        missing_indicator_cols = [
            feature
            for feature in feature_parts["indicator_feature_cols"]
            if feature not in indicator_config_map
        ]
        if missing_indicator_cols:
            raise ValueError(
                "Missing indicator configs for current recompute features: "
                f"{missing_indicator_cols[:10]}"
            )
        ohlcv_np = feature_frame[list(OHLCV_COLS)].to_numpy(dtype=np.float64, copy=True)
        feature_frame = add_indicator_values(feature_frame, ohlcv_np, indicator_configs)

    keep_cols = ["Opened", *RAW_OHLCV_COLS]
    keep_cols.extend(col for col in feature_columns if col not in keep_cols)
    return feature_frame.loc[:, keep_cols].copy()


def align_feature_frame_to_audit_rows(
    *,
    audit_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    aligned = audit_df.loc[:, ["Opened"]].merge(
        feature_frame.loc[:, ["Opened", *feature_columns]],
        on="Opened",
        how="left",
        sort=False,
    )
    if aligned[feature_columns].isna().all(axis=1).any():
        missing_rows = aligned.loc[aligned[feature_columns].isna().all(axis=1), "Opened"]
        raise RuntimeError(
            "Current recompute frame is missing rows for audit timestamps. "
            f"First missing={missing_rows.iloc[0]!s}"
        )
    return aligned


def _advance_kelly_rng_once(predictor: "PseudoLiveAuditPredictor") -> None:
    predictor.price_rng.standard_normal()


def _compare_kelly_decisions(
    *,
    predictor: "PseudoLiveAuditPredictor",
    candidate_proba: np.ndarray,
    reference_proba: np.ndarray,
    candidate_label: str,
    reference_label: str,
) -> pd.DataFrame:
    bankroll = float(LIVE_INITIAL_BANKROLL_USDC)
    rows: list[dict[str, Any]] = []
    for candidate_prob, reference_prob in zip(candidate_proba, reference_proba, strict=True):
        rng_state = copy.deepcopy(predictor.price_rng.bit_generator.state)
        candidate_decision = predictor.evaluate_kelly_recommendation(
            float(candidate_prob),
            bankroll=bankroll,
            rng_state=rng_state,
        )
        reference_decision = predictor.evaluate_kelly_recommendation(
            float(reference_prob),
            bankroll=bankroll,
            rng_state=rng_state,
        )
        _advance_kelly_rng_once(predictor)

        candidate_stake = float(candidate_decision.get("bet_usdc", 0.0) or 0.0)
        reference_stake = float(reference_decision.get("bet_usdc", 0.0) or 0.0)
        candidate_trade = int(candidate_stake > 0.0)
        reference_trade = int(reference_stake > 0.0)
        candidate_side = str(candidate_decision.get("side", "none"))
        reference_side = str(reference_decision.get("side", "none"))
        candidate_reason = str(candidate_decision.get("reason", ""))
        reference_reason = str(reference_decision.get("reason", ""))
        rows.append(
            {
                f"{candidate_label}_kelly_side": candidate_side,
                f"{reference_label}_kelly_side": reference_side,
                f"{candidate_label}_kelly_reason": candidate_reason,
                f"{reference_label}_kelly_reason": reference_reason,
                f"{candidate_label}_stake_usdc": candidate_stake,
                f"{reference_label}_stake_usdc": reference_stake,
                f"{candidate_label}_trade_flag": candidate_trade,
                f"{reference_label}_trade_flag": reference_trade,
                "kelly_side_mismatch": int(candidate_side != reference_side),
                "kelly_reason_mismatch": int(candidate_reason != reference_reason),
                "kelly_trade_flag_mismatch": int(candidate_trade != reference_trade),
                "kelly_stake_abs_diff": abs(candidate_stake - reference_stake),
                "kelly_decision_mismatch": int(
                    candidate_side != reference_side
                    or candidate_reason != reference_reason
                    or candidate_trade != reference_trade
                    or abs(candidate_stake - reference_stake) > 1e-9
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_matrix_comparison_report(
    *,
    candidate_label: str,
    reference_label: str,
    candidate_matrix: np.ndarray,
    reference_matrix: np.ndarray,
    audit_df: pd.DataFrame,
    feature_columns: list[str],
    feature_group_by_name: dict[str, str],
    feature_builder_frame: pd.DataFrame,
    model,
    kelly_predictor: "PseudoLiveAuditPredictor" | None = None,
) -> dict[str, Any]:
    feature_names = np.asarray(feature_columns, dtype=object)
    builder_meta = (
        feature_builder_frame.set_index("feature")
        .reindex(feature_columns)
        .reset_index(drop=False)
    )
    builder_groups = builder_meta["builder_family"].fillna("unknown").to_numpy(dtype=object)
    builder_names = builder_meta["builder_name"].fillna("unknown").to_numpy(dtype=object)
    feature_groups = np.asarray(
        [feature_group_by_name.get(col, "unknown") for col in feature_columns],
        dtype=object,
    )
    candidate_nonfinite_mask = ~np.isfinite(candidate_matrix)
    reference_nonfinite_mask = ~np.isfinite(reference_matrix)
    finite_pair_mask = np.isfinite(candidate_matrix) & np.isfinite(reference_matrix)
    finite_status_mismatch_mask = np.logical_xor(candidate_nonfinite_mask, reference_nonfinite_mask)
    diff_matrix = np.abs(candidate_matrix - reference_matrix)
    diff_matrix[~finite_pair_mask] = np.nan
    rel_diff_matrix = _safe_relative_diff(candidate_matrix, reference_matrix)

    candidate_pred = model.predict(candidate_matrix).astype(np.float64, copy=False)
    reference_pred = model.predict(reference_matrix).astype(np.float64, copy=False)
    pred_abs_diff = np.abs(candidate_pred - reference_pred)
    prediction_impact_report = _build_single_feature_prediction_impact_report(
        candidate_label=candidate_label,
        reference_label=reference_label,
        candidate_matrix=candidate_matrix,
        reference_matrix=reference_matrix,
        candidate_pred=candidate_pred,
        reference_pred=reference_pred,
        audit_df=audit_df,
        feature_columns=feature_columns,
        feature_group_by_name=feature_group_by_name,
        feature_builder_frame=feature_builder_frame,
        model=model,
    )

    row_worst_idx = _safe_argmax_axis1(diff_matrix)
    row_has_finite = np.isfinite(diff_matrix).any(axis=1)
    row_worst_feature = np.where(row_has_finite, feature_names[row_worst_idx], None)
    row_worst_group = np.where(row_has_finite, feature_groups[row_worst_idx], None)
    row_worst_builder_family = np.where(row_has_finite, builder_groups[row_worst_idx], None)
    row_worst_builder_name = np.where(row_has_finite, builder_names[row_worst_idx], None)
    row_worst_candidate = np.where(
        row_has_finite,
        candidate_matrix[np.arange(len(audit_df)), row_worst_idx],
        np.nan,
    )
    row_worst_reference = np.where(
        row_has_finite,
        reference_matrix[np.arange(len(audit_df)), row_worst_idx],
        np.nan,
    )

    step_summary_df = pd.DataFrame(
        {
            "Opened": audit_df["Opened"],
            "feature_max_abs_diff": _safe_nanmax_axis1(diff_matrix),
            "feature_mean_abs_diff": _safe_rowwise_mean(diff_matrix),
            "feature_max_rel_diff": _safe_nanmax_axis1(rel_diff_matrix),
            "feature_mean_rel_diff": _safe_rowwise_mean(rel_diff_matrix),
            "worst_feature": row_worst_feature,
            "worst_group": row_worst_group,
            "worst_builder_family": row_worst_builder_family,
            "worst_builder_name": row_worst_builder_name,
            f"worst_feature_{candidate_label}_value": row_worst_candidate,
            f"worst_feature_{reference_label}_value": row_worst_reference,
            f"{candidate_label}_nonfinite_count": candidate_nonfinite_mask.sum(axis=1).astype(
                np.int32
            ),
            f"{reference_label}_nonfinite_count": reference_nonfinite_mask.sum(axis=1).astype(
                np.int32
            ),
            "finite_status_mismatch_count": finite_status_mismatch_mask.sum(axis=1).astype(
                np.int32
            ),
            f"{candidate_label}_proba_up": candidate_pred,
            f"{reference_label}_proba_up": reference_pred,
            "proba_up_abs_diff": pred_abs_diff,
            "signal_mismatch": (
                (candidate_pred >= 0.5) != (reference_pred >= 0.5)
            ).astype(np.int8),
        }
    )
    step_summary_df = pd.concat(
        [
            step_summary_df.reset_index(drop=True),
            prediction_impact_report["row_summary_df"].reset_index(drop=True),
        ],
        axis=1,
        copy=False,
    )

    if kelly_predictor is not None:
        kelly_df = _compare_kelly_decisions(
            predictor=kelly_predictor,
            candidate_proba=candidate_pred,
            reference_proba=reference_pred,
            candidate_label=candidate_label,
            reference_label=reference_label,
        )
        step_summary_df = pd.concat(
            [step_summary_df.reset_index(drop=True), kelly_df.reset_index(drop=True)],
            axis=1,
            copy=False,
        )
        step_summary_df["business_decision_mismatch"] = (
            (step_summary_df["signal_mismatch"] > 0)
            | (step_summary_df["kelly_side_mismatch"] > 0)
            | (step_summary_df["kelly_reason_mismatch"] > 0)
            | (step_summary_df["kelly_trade_flag_mismatch"] > 0)
        ).astype(np.int8)
        step_summary_df["stake_only_kelly_mismatch"] = (
            (step_summary_df["kelly_decision_mismatch"] > 0)
            & (step_summary_df["business_decision_mismatch"] == 0)
        ).astype(np.int8)
    else:
        step_summary_df["business_decision_mismatch"] = step_summary_df["signal_mismatch"].astype(
            np.int8
        )
        step_summary_df["stake_only_kelly_mismatch"] = np.zeros(
            len(step_summary_df), dtype=np.int8
        )

    col_has_finite = np.isfinite(diff_matrix).any(axis=0)
    col_worst_idx = _safe_argmax_axis0(diff_matrix)
    feature_summary_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "group": [feature_group_by_name.get(col, "unknown") for col in feature_columns],
            "max_abs_diff": _safe_nanmax_axis0(diff_matrix),
            "mean_abs_diff": _safe_colwise_mean(diff_matrix),
            "max_rel_diff": _safe_nanmax_axis0(rel_diff_matrix),
            "mean_rel_diff": _safe_colwise_mean(rel_diff_matrix),
            "rmse_abs_diff": _safe_colwise_rmse(diff_matrix),
            "finite_diff_count": np.isfinite(diff_matrix).sum(axis=0).astype(np.int32),
            f"{candidate_label}_nonfinite_count": candidate_nonfinite_mask.sum(axis=0).astype(
                np.int32
            ),
            f"{reference_label}_nonfinite_count": reference_nonfinite_mask.sum(axis=0).astype(
                np.int32
            ),
            "finite_status_mismatch_count": finite_status_mismatch_mask.sum(axis=0).astype(
                np.int32
            ),
            "worst_opened": np.where(
                col_has_finite,
                audit_df["Opened"].iloc[col_worst_idx].to_numpy(),
                pd.NaT,
            ),
            f"worst_{candidate_label}_value": np.where(
                col_has_finite,
                candidate_matrix[col_worst_idx, np.arange(len(feature_columns))],
                np.nan,
            ),
            f"worst_{reference_label}_value": np.where(
                col_has_finite,
                reference_matrix[col_worst_idx, np.arange(len(feature_columns))],
                np.nan,
            ),
        }
    )
    feature_impact_summary_df = prediction_impact_report["feature_summary_df"].drop(
        columns=["group", "builder_family", "builder_name", "builder_source"],
        errors="ignore",
    )
    feature_summary_df = feature_summary_df.merge(
        feature_builder_frame,
        on="feature",
        how="left",
        sort=False,
    ).merge(
        feature_impact_summary_df,
        on="feature",
        how="left",
        sort=False,
    ).sort_values(
        [
            "rows_signal_mismatch_resolved_if_fixed",
            "mean_abs_proba_gap_reduction_if_fixed",
            "max_abs_proba_gap_reduction_if_fixed",
            "mean_abs_proba_shift_if_fixed",
            "finite_status_mismatch_count",
            "max_rel_diff",
            "mean_rel_diff",
            "max_abs_diff",
            "mean_abs_diff",
        ],
        ascending=[False, False, False, False, False, False, False, False, False],
        kind="stable",
    )

    group_summary_df = (
        feature_summary_df.groupby("group", dropna=False)
        .agg(
            feature_count=("feature", "count"),
            max_abs_diff=("max_abs_diff", "max"),
            mean_abs_diff=("mean_abs_diff", "mean"),
            max_rel_diff=("max_rel_diff", "max"),
            mean_rel_diff=("mean_rel_diff", "mean"),
            rmse_abs_diff=("rmse_abs_diff", "mean"),
            total_candidate_nonfinite_count=(f"{candidate_label}_nonfinite_count", "sum"),
            total_reference_nonfinite_count=(f"{reference_label}_nonfinite_count", "sum"),
            total_finite_status_mismatch_count=("finite_status_mismatch_count", "sum"),
        )
        .reset_index()
    )
    group_prediction_impact_df = prediction_impact_report["group_summary_df"].drop(
        columns=["feature_count"],
        errors="ignore",
    )
    group_summary_df = group_summary_df.merge(
        group_prediction_impact_df,
        on="group",
        how="left",
        sort=False,
    ).sort_values(
        [
            "max_mean_abs_proba_gap_reduction_if_fixed",
            "mean_mean_abs_proba_gap_reduction_if_fixed",
            "max_mean_abs_proba_shift_if_fixed",
            "max_rel_diff",
            "mean_rel_diff",
            "max_abs_diff",
            "mean_abs_diff",
        ],
        ascending=[False, False, False, False, False, False, False],
        kind="stable",
    )

    builder_summary_df = (
        feature_summary_df.groupby(["builder_family", "builder_name"], dropna=False)
        .agg(
            feature_count=("feature", "count"),
            max_abs_diff=("max_abs_diff", "max"),
            mean_abs_diff=("mean_abs_diff", "mean"),
            max_rel_diff=("max_rel_diff", "max"),
            mean_rel_diff=("mean_rel_diff", "mean"),
            rmse_abs_diff=("rmse_abs_diff", "mean"),
            total_finite_status_mismatch_count=("finite_status_mismatch_count", "sum"),
        )
        .reset_index()
    )
    builder_prediction_impact_df = prediction_impact_report["builder_summary_df"].drop(
        columns=["feature_count"],
        errors="ignore",
    )
    builder_summary_df = builder_summary_df.merge(
        builder_prediction_impact_df,
        on=["builder_family", "builder_name"],
        how="left",
        sort=False,
    ).sort_values(
        [
            "max_mean_abs_proba_gap_reduction_if_fixed",
            "mean_mean_abs_proba_gap_reduction_if_fixed",
            "max_mean_abs_proba_shift_if_fixed",
            "max_rel_diff",
            "mean_rel_diff",
            "max_abs_diff",
            "mean_abs_diff",
        ],
        ascending=[False, False, False, False, False, False, False],
        kind="stable",
    )

    summary_payload = {
        "decision_row_count": len(audit_df),
        "feature_count": len(feature_columns),
        f"rows_with_{candidate_label}_nonfinite": int(
            (step_summary_df[f"{candidate_label}_nonfinite_count"] > 0).sum()
        ),
        f"rows_with_{reference_label}_nonfinite": int(
            (step_summary_df[f"{reference_label}_nonfinite_count"] > 0).sum()
        ),
        "rows_with_finite_status_mismatch": int(
            (step_summary_df["finite_status_mismatch_count"] > 0).sum()
        ),
        "rows_with_signal_mismatch": int(step_summary_df["signal_mismatch"].sum()),
        "rows_with_business_decision_mismatch": int(
            step_summary_df["business_decision_mismatch"].sum()
        ),
        "max_feature_abs_diff": float(step_summary_df["feature_max_abs_diff"].max()),
        "mean_feature_abs_diff": float(step_summary_df["feature_mean_abs_diff"].mean()),
        "max_feature_rel_diff": float(step_summary_df["feature_max_rel_diff"].max()),
        "mean_feature_rel_diff": float(step_summary_df["feature_mean_rel_diff"].mean()),
        "max_proba_up_abs_diff": float(step_summary_df["proba_up_abs_diff"].max()),
        "mean_proba_up_abs_diff": float(step_summary_df["proba_up_abs_diff"].mean()),
        "rows_with_proba_diff_gt_tol": int(
            (step_summary_df["proba_up_abs_diff"] > PREDICTION_DIFF_TOL).sum()
        ),
    }
    summary_payload.update(prediction_impact_report["summary"].to_dict())
    if "kelly_decision_mismatch" in step_summary_df.columns:
        summary_payload.update(
            {
                "kelly_audit_bankroll_usdc": float(LIVE_INITIAL_BANKROLL_USDC),
                "rows_with_kelly_side_mismatch": int(step_summary_df["kelly_side_mismatch"].sum()),
                "rows_with_kelly_reason_mismatch": int(
                    step_summary_df["kelly_reason_mismatch"].sum()
                ),
                "rows_with_kelly_trade_flag_mismatch": int(
                    step_summary_df["kelly_trade_flag_mismatch"].sum()
                ),
                "rows_with_stake_only_kelly_mismatch": int(
                    step_summary_df["stake_only_kelly_mismatch"].sum()
                ),
                "rows_with_any_kelly_mismatch": int(
                    step_summary_df["kelly_decision_mismatch"].sum()
                ),
                "max_kelly_stake_abs_diff": float(step_summary_df["kelly_stake_abs_diff"].max()),
                "mean_kelly_stake_abs_diff": float(
                    step_summary_df["kelly_stake_abs_diff"].mean()
                ),
            }
        )

    return {
        "summary": pd.Series(summary_payload),
        "step_summary_df": step_summary_df,
        "feature_summary_df": feature_summary_df.reset_index(drop=True),
        "group_summary_df": group_summary_df,
        "builder_summary_df": builder_summary_df,
        "audit_df": audit_df,
        "feature_columns": feature_columns,
        "candidate_feature_frame": pd.DataFrame(
            candidate_matrix,
            columns=feature_columns,
            copy=False,
        ),
        "prediction_impact_feature_summary_df": prediction_impact_report["feature_summary_df"],
        "prediction_impact_group_summary_df": prediction_impact_report["group_summary_df"],
        "prediction_impact_builder_summary_df": prediction_impact_report["builder_summary_df"],
        "single_feature_fixed_pred_matrix": prediction_impact_report["fixed_pred_matrix"],
        "single_feature_proba_shift_matrix": prediction_impact_report["proba_shift_matrix"],
        "single_feature_gap_reduction_matrix": prediction_impact_report["gap_reduction_matrix"],
        "diff_matrix": diff_matrix,
        "rel_diff_matrix": rel_diff_matrix,
        "candidate_nonfinite_mask": candidate_nonfinite_mask,
        "reference_nonfinite_mask": reference_nonfinite_mask,
        "finite_status_mismatch_mask": finite_status_mismatch_mask,
        "candidate_label": candidate_label,
        "reference_label": reference_label,
    }


def build_live_drift_reason_report(
    report: dict[str, Any],
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    step_summary_df = report["step_summary_df"].copy()
    feature_summary_df = report["feature_summary_df"].copy()
    diff_matrix = np.asarray(report["diff_matrix"], dtype=np.float64)
    rel_diff_matrix = np.asarray(report["rel_diff_matrix"], dtype=np.float64)
    fixed_pred_matrix = report.get("single_feature_fixed_pred_matrix")
    proba_shift_matrix = report.get("single_feature_proba_shift_matrix")
    gap_reduction_matrix = report.get("single_feature_gap_reduction_matrix")
    if fixed_pred_matrix is not None:
        fixed_pred_matrix = np.asarray(fixed_pred_matrix, dtype=np.float64)
    if proba_shift_matrix is not None:
        proba_shift_matrix = np.asarray(proba_shift_matrix, dtype=np.float64)
    if gap_reduction_matrix is not None:
        gap_reduction_matrix = np.asarray(gap_reduction_matrix, dtype=np.float64)
    feature_columns = list(report["feature_columns"])
    feature_names = np.asarray(feature_columns, dtype=object)
    candidate_label = str(report.get("candidate_label", "candidate"))
    reference_label = str(report.get("reference_label", "reference"))
    candidate_proba_col = f"{candidate_label}_proba_up"
    reference_proba_col = f"{reference_label}_proba_up"
    history_shortfall = step_summary_df.get(
        "history_shortfall",
        pd.Series(0, index=step_summary_df.index, dtype=np.int32),
    )

    drift_mask = step_summary_df["proba_up_abs_diff"] > PREDICTION_DIFF_TOL
    business_mask = (
        step_summary_df["business_decision_mismatch"] > 0
        if "business_decision_mismatch" in step_summary_df.columns
        else step_summary_df["signal_mismatch"] > 0
    )
    kelly_mask = (
        step_summary_df["kelly_decision_mismatch"] > 0
        if "kelly_decision_mismatch" in step_summary_df.columns
        else pd.Series(False, index=step_summary_df.index)
    )
    stake_only_mask = (
        step_summary_df["stake_only_kelly_mismatch"] > 0
        if "stake_only_kelly_mismatch" in step_summary_df.columns
        else pd.Series(False, index=step_summary_df.index)
    )
    explain_mask = business_mask
    explanation_basis = "business_decision_mismatch"
    if not bool(explain_mask.any()) and bool(stake_only_mask.any()):
        explain_mask = stake_only_mask
        explanation_basis = "stake_only_kelly_mismatch"
    if not bool(explain_mask.any()) and bool(drift_mask.any()):
        explain_mask = drift_mask
        explanation_basis = "proba_diff_gt_tol"
    if not bool(explain_mask.any()):
        explain_mask = step_summary_df["feature_max_abs_diff"] > 0
        explanation_basis = "feature_diff_fallback"

    dominant_feature_col = (
        "top_prediction_impact_feature"
        if "top_prediction_impact_feature" in step_summary_df.columns
        else "worst_feature"
    )
    dominant_group_col = (
        "top_prediction_impact_group"
        if "top_prediction_impact_group" in step_summary_df.columns
        else "worst_group"
    )
    dominant_builder_family_col = (
        "top_prediction_impact_builder_family"
        if "top_prediction_impact_builder_family" in step_summary_df.columns
        else "worst_builder_family"
    )
    dominant_builder_name_col = (
        "top_prediction_impact_builder_name"
        if "top_prediction_impact_builder_name" in step_summary_df.columns
        else "worst_builder_name"
    )
    row_sort_cols = [
        col
        for col in [
            "top_prediction_impact_abs_proba_gap_reduction_if_fixed",
            "proba_up_abs_diff",
            "feature_max_abs_diff",
            "finite_status_mismatch_count",
        ]
        if col in step_summary_df.columns
    ]
    explain_rows_df = (
        step_summary_df.loc[explain_mask]
        .sort_values(
            row_sort_cols,
            ascending=[False] * len(row_sort_cols),
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if explain_rows_df.empty:
        summary = pd.Series(
            {
                "rows_selected_for_explanation": 0,
                "rows_with_proba_diff_gt_tol": int(drift_mask.sum()),
                "rows_with_business_decision_mismatch": int(business_mask.sum()),
                "rows_with_stake_only_kelly_mismatch": int(stake_only_mask.sum()),
                "rows_with_any_kelly_mismatch": int(kelly_mask.sum()),
                "explanation_basis": explanation_basis,
                "rows_with_history_shortfall": int(history_shortfall.gt(0).sum()),
                "prediction_drift_rows_with_history_shortfall": 0,
                "dominant_prediction_impact_group": None,
                "dominant_prediction_impact_builder_family": None,
                "dominant_prediction_impact_builder_name": None,
                "dominant_prediction_impact_feature": None,
                "dominant_worst_group": None,
                "dominant_worst_builder_family": None,
                "dominant_worst_builder_name": None,
                "dominant_worst_feature": None,
            }
        )
        return {
            "summary": summary,
            "row_summary_df": explain_rows_df,
            "group_summary_df": pd.DataFrame(),
            "builder_summary_df": pd.DataFrame(),
            "feature_summary_df": pd.DataFrame(),
        }

    explain_row_mask = explain_mask.to_numpy(dtype=bool, copy=False)
    if len(explain_rows_df) == len(step_summary_df):
        explain_diff_matrix = diff_matrix
        explain_rel_diff_matrix = rel_diff_matrix
    else:
        explain_diff_matrix = diff_matrix[explain_row_mask, :]
        explain_rel_diff_matrix = rel_diff_matrix[explain_row_mask, :]

    explain_group_summary_df = (
        explain_rows_df.groupby(dominant_group_col, dropna=False)
        .agg(
            row_count=("Opened", "count"),
            max_proba_up_abs_diff=("proba_up_abs_diff", "max"),
            mean_proba_up_abs_diff=("proba_up_abs_diff", "mean"),
            max_top_prediction_impact_abs_proba_gap_reduction_if_fixed=(
                "top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "max",
            ),
            mean_top_prediction_impact_abs_proba_gap_reduction_if_fixed=(
                "top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "mean",
            ),
            max_top_prediction_impact_abs_proba_shift_if_fixed=(
                "top_prediction_impact_abs_proba_shift_if_fixed",
                "max",
            ),
            mean_top_prediction_impact_abs_proba_shift_if_fixed=(
                "top_prediction_impact_abs_proba_shift_if_fixed",
                "mean",
            ),
        )
        .sort_values(
            [
                "row_count",
                "max_top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "mean_top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "max_proba_up_abs_diff",
                "max_top_prediction_impact_abs_proba_shift_if_fixed",
            ],
            ascending=[False, False, False, False, False],
            kind="stable",
        )
        .reset_index()
        .rename(columns={dominant_group_col: "prediction_impact_group"})
    )
    explain_builder_summary_df = (
        explain_rows_df.groupby(
            [dominant_builder_family_col, dominant_builder_name_col],
            dropna=False,
        )
        .agg(
            row_count=("Opened", "count"),
            max_proba_up_abs_diff=("proba_up_abs_diff", "max"),
            mean_proba_up_abs_diff=("proba_up_abs_diff", "mean"),
            max_top_prediction_impact_abs_proba_gap_reduction_if_fixed=(
                "top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "max",
            ),
            mean_top_prediction_impact_abs_proba_gap_reduction_if_fixed=(
                "top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "mean",
            ),
            max_top_prediction_impact_abs_proba_shift_if_fixed=(
                "top_prediction_impact_abs_proba_shift_if_fixed",
                "max",
            ),
            mean_top_prediction_impact_abs_proba_shift_if_fixed=(
                "top_prediction_impact_abs_proba_shift_if_fixed",
                "mean",
            ),
        )
        .sort_values(
            [
                "row_count",
                "max_top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "mean_top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "max_proba_up_abs_diff",
                "max_top_prediction_impact_abs_proba_shift_if_fixed",
            ],
            ascending=[False, False, False, False, False],
            kind="stable",
        )
        .reset_index()
        .rename(
            columns={
                dominant_builder_family_col: "prediction_impact_builder_family",
                dominant_builder_name_col: "prediction_impact_builder_name",
            }
        )
    )

    explain_feature_summary_df: pd.DataFrame
    if (
        fixed_pred_matrix is not None
        and proba_shift_matrix is not None
        and gap_reduction_matrix is not None
        and candidate_proba_col in step_summary_df.columns
        and reference_proba_col in step_summary_df.columns
    ):
        explain_fixed_pred_matrix = fixed_pred_matrix[explain_row_mask, :]
        explain_abs_proba_shift_matrix = np.abs(proba_shift_matrix[explain_row_mask, :])
        explain_gap_reduction_matrix = gap_reduction_matrix[explain_row_mask, :]
        explain_candidate_pred = step_summary_df.loc[explain_mask, candidate_proba_col].to_numpy(
            dtype=np.float64,
            copy=False,
        )
        explain_reference_pred = step_summary_df.loc[explain_mask, reference_proba_col].to_numpy(
            dtype=np.float64,
            copy=False,
        )
        explain_signal_mismatch = (
            (explain_candidate_pred >= 0.5) != (explain_reference_pred >= 0.5)
        )[:, None]
        explain_fixed_signal_mismatch = (
            explain_fixed_pred_matrix >= 0.5
        ) != explain_reference_pred[:, None]

        explain_feature_summary_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "max_abs_proba_gap_reduction_if_fixed_on_explained_rows": _safe_nanmax_axis0(
                        explain_gap_reduction_matrix
                    ),
                    "mean_abs_proba_gap_reduction_if_fixed_on_explained_rows": _safe_colwise_mean(
                        explain_gap_reduction_matrix
                    ),
                    "max_abs_proba_shift_if_fixed_on_explained_rows": _safe_nanmax_axis0(
                        explain_abs_proba_shift_matrix
                    ),
                    "mean_abs_proba_shift_if_fixed_on_explained_rows": _safe_colwise_mean(
                        explain_abs_proba_shift_matrix
                    ),
                    "rows_abs_proba_gap_reduced_if_fixed_on_explained_rows": (
                        explain_gap_reduction_matrix > PREDICTION_DIFF_TOL
                    ).sum(axis=0).astype(np.int32),
                    "rows_signal_mismatch_resolved_if_fixed_on_explained_rows": (
                        explain_signal_mismatch & ~explain_fixed_signal_mismatch
                    ).sum(axis=0).astype(np.int32),
                    "max_rel_diff_on_explained_rows": _safe_nanmax_axis0(explain_rel_diff_matrix),
                    "mean_rel_diff_on_explained_rows": _safe_colwise_mean(explain_rel_diff_matrix),
                    "max_abs_diff_on_explained_rows": _safe_nanmax_axis0(explain_diff_matrix),
                    "mean_abs_diff_on_explained_rows": _safe_colwise_mean(explain_diff_matrix),
                    "rmse_abs_diff_on_explained_rows": _safe_colwise_rmse(explain_diff_matrix),
                }
            )
            .merge(
                feature_summary_df[["feature", "group", "builder_family", "builder_name"]],
                on="feature",
                how="left",
                sort=False,
            )
            .sort_values(
                [
                    "rows_signal_mismatch_resolved_if_fixed_on_explained_rows",
                    "mean_abs_proba_gap_reduction_if_fixed_on_explained_rows",
                    "max_abs_proba_gap_reduction_if_fixed_on_explained_rows",
                    "mean_abs_proba_shift_if_fixed_on_explained_rows",
                    "max_abs_proba_shift_if_fixed_on_explained_rows",
                    "max_abs_diff_on_explained_rows",
                ],
                ascending=[False, False, False, False, False, False],
                kind="stable",
            )
            .reset_index(drop=True)
        )
    else:
        feature_max = _safe_nanmax_axis0(explain_diff_matrix)
        feature_mean = _safe_colwise_mean(explain_diff_matrix)
        feature_rmse = _safe_colwise_rmse(explain_diff_matrix)
        feature_rel_max = _safe_nanmax_axis0(explain_rel_diff_matrix)
        feature_rel_mean = _safe_colwise_mean(explain_rel_diff_matrix)
        explain_feature_summary_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "max_rel_diff_on_explained_rows": feature_rel_max,
                    "mean_rel_diff_on_explained_rows": feature_rel_mean,
                    "max_abs_diff_on_explained_rows": feature_max,
                    "mean_abs_diff_on_explained_rows": feature_mean,
                    "rmse_abs_diff_on_explained_rows": feature_rmse,
                }
            )
            .merge(
                feature_summary_df[["feature", "group", "builder_family", "builder_name"]],
                on="feature",
                how="left",
                sort=False,
            )
            .sort_values(
                [
                    "max_rel_diff_on_explained_rows",
                    "mean_rel_diff_on_explained_rows",
                    "max_abs_diff_on_explained_rows",
                    "mean_abs_diff_on_explained_rows",
                    "rmse_abs_diff_on_explained_rows",
                ],
                ascending=[False, False, False, False, False],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    dominant_group = (
        None
        if explain_group_summary_df.empty
        else explain_group_summary_df.iloc[0]["prediction_impact_group"]
    )
    dominant_builder_family = (
        None
        if explain_builder_summary_df.empty
        else explain_builder_summary_df.iloc[0]["prediction_impact_builder_family"]
    )
    dominant_builder_name = (
        None
        if explain_builder_summary_df.empty
        else explain_builder_summary_df.iloc[0]["prediction_impact_builder_name"]
    )
    dominant_feature = (
        None
        if explain_feature_summary_df.empty
        else explain_feature_summary_df.iloc[0]["feature"]
    )

    summary = pd.Series(
        {
            "rows_selected_for_explanation": int(explain_mask.sum()),
            "rows_with_proba_diff_gt_tol": int(drift_mask.sum()),
            "rows_with_business_decision_mismatch": int(business_mask.sum()),
            "rows_with_stake_only_kelly_mismatch": int(stake_only_mask.sum()),
            "rows_with_any_kelly_mismatch": int(kelly_mask.sum()),
            "explanation_basis": explanation_basis,
            "rows_with_history_shortfall": int(history_shortfall.gt(0).sum()),
            "prediction_drift_rows_with_history_shortfall": int(
                explain_rows_df.get("history_shortfall", pd.Series(dtype=np.int32)).gt(0).sum()
            ),
            "dominant_prediction_impact_group": dominant_group,
            "dominant_prediction_impact_builder_family": dominant_builder_family,
            "dominant_prediction_impact_builder_name": dominant_builder_name,
            "dominant_prediction_impact_feature": dominant_feature,
            "dominant_worst_group": dominant_group,
            "dominant_worst_builder_family": dominant_builder_family,
            "dominant_worst_builder_name": dominant_builder_name,
            "dominant_worst_feature": dominant_feature,
        }
    )

    return {
        "summary": summary,
        "row_summary_df": explain_rows_df.head(int(top_n)).copy(),
        "group_summary_df": explain_group_summary_df.head(int(top_n)).copy(),
        "builder_summary_df": explain_builder_summary_df.head(int(top_n)).copy(),
        "feature_summary_df": explain_feature_summary_df.head(int(top_n)).copy(),
    }


@dataclass(frozen=True)
class AuditWindow:
    bootstrap_start: pd.Timestamp
    audit_start: pd.Timestamp
    audit_end: pd.Timestamp
    bootstrap_rows: int
    audit_rows: int
    requested_days_back: int
    max_steps: int | None


class PseudoLiveAuditPredictor(LivePredictor):
    def __init__(
        self,
        bootstrap_df: pd.DataFrame,
        *,
        model_meta_path: Path = MODEL_META_PATH,
        max_keep: int = DEFAULT_MAX_KEEP,
        volume_profile_state: dict[str, Any] | None = None,
    ) -> None:
        self.model, meta = load_model_and_meta(model_meta_path)
        self.feature_columns = list(meta.get("feature_columns", []))
        if not self.feature_columns:
            raise ValueError("Missing feature_columns in model metadata.")

        self.candle_feature_columns = [
            col for col in self.feature_columns if col in SUPPORTED_CANDLE_FEATURE_COLS
        ]
        self.candle_derived_feature_columns = tuple(
            resolve_candle_derived_feature_cols(self.candle_feature_columns)
        )
        self.candle_pattern_feature_columns = tuple(
            resolve_candle_pattern_feature_cols(self.candle_feature_columns)
        )
        self.prediction_threshold = 0.5
        self.target_col = str(meta.get("target_col", "target_5m_candle_up"))
        self.target_bucket_minutes = parse_target_bucket_minutes(self.target_col)
        self.kelly_runtime = load_kelly_runtime_config(KELLY_CONFIG_PATH)
        self.live_bankroll_usdc = float(LIVE_INITIAL_BANKROLL_USDC)
        self.price_rng = np.random.default_rng(int(self.kelly_runtime["seed"]))

        feature_parts = split_feature_subset(self.feature_columns)
        if feature_parts["streak_intervals"]:
            self.streak_interval_to_rule = resolve_streak_interval_to_rule(
                feature_parts["streak_intervals"]
            )
        else:
            self.streak_interval_to_rule = {}
        self.session_feature_columns = tuple(feature_parts["session_feature_cols"])
        self.volume_profile_feature_columns = tuple(
            feature_parts["volume_profile_feature_cols"]
        )
        self.volume_profile_cfg = normalize_volume_profile_config(
            MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")
        )
        self.volume_profile_enabled = bool(
            self.volume_profile_feature_columns and self.volume_profile_cfg["enabled"]
        )
        self.volume_profile_state_path = (
            VP_PSEUDO_LIVE_AUDIT_RUNTIME_STATE_DIR
            / f"{SYMBOL}_{INTERVAL}_{VP_FEATURE_VERSION}"
        )
        self.volume_profile_modeling_state_path = (
            VP_PSEUDO_LIVE_AUDIT_MODELING_STATE_DIR
            / f"{SYMBOL}_{INTERVAL}_{VP_FEATURE_VERSION}_modeling_end"
        )
        self.volume_profile_state_source_path = None

        self.indicator_specs = load_indicator_specs(self.feature_columns)
        self.required_stable_window = load_required_stable_window(
            INDICATOR_STABILITY_SUMMARY_PATH
        )
        self.bootstrap_candles = int(len(bootstrap_df))
        self.max_keep = int(max_keep)

        bootstrap_df = bootstrap_df.copy()
        bootstrap_df["Opened"] = _ensure_utc_opened(bootstrap_df["Opened"])
        bootstrap_df = bootstrap_df.sort_values("Opened").reset_index(drop=True)
        if bootstrap_df.empty:
            raise ValueError("bootstrap_df cannot be empty.")

        self.opened_candles = deque(pd.Timestamp(v) for v in bootstrap_df["Opened"])
        self.ohlcv_np = bootstrap_df[OHLCV_COLS].to_numpy(dtype=np.float64, copy=True)
        self.opened_ns_np = np.fromiter(
            (opened.value for opened in self.opened_candles),
            dtype=np.int64,
            count=len(self.opened_candles),
        )
        self.candle_open_close = {
            opened: (float(self.ohlcv_np[i, 0]), float(self.ohlcv_np[i, 3]))
            for i, opened in enumerate(self.opened_candles)
        }
        self.records = []
        self.predicted_buckets = set()
        self.local_tz = None
        self.last_indicator_nan_cols = []
        self.last_processed_closed_opened = (
            self.opened_candles[-1] if self.opened_candles else None
        )
        self.predictions_path = None

        self.volume_profile_state = None
        if self.volume_profile_enabled:
            if volume_profile_state is not None:
                self.volume_profile_state = copy.deepcopy(volume_profile_state)
            else:
                self.volume_profile_state = bootstrap_state_from_history(
                    bootstrap_df.loc[:, ["Opened", "High", "Low", "Volume"]],
                    self.volume_profile_cfg,
                )

    def _save_runtime_volume_profile_state(self, log: bool = False, context: str = "state"):
        return None

    def evaluate_kelly_recommendation(
        self,
        prob_up_raw: float,
        *,
        bankroll: float | None = None,
        rng_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prev_bankroll = float(self.live_bankroll_usdc)
        prev_rng_state = copy.deepcopy(self.price_rng.bit_generator.state)
        try:
            if bankroll is not None:
                self.live_bankroll_usdc = float(bankroll)
            if rng_state is not None:
                self.price_rng.bit_generator.state = copy.deepcopy(rng_state)
            return dict(self._recommend_kelly_bet(prob_up_raw=float(prob_up_raw)))
        finally:
            self.live_bankroll_usdc = prev_bankroll
            self.price_rng.bit_generator.state = prev_rng_state

    def build_feature_snapshot(self, volume_profile_values: dict[str, float] | None = None):
        vector = self._build_feature_vector(volume_profile_values=volume_profile_values)
        nonfinite_feature_indices = tuple(
            int(idx) for idx in np.flatnonzero(~np.isfinite(vector[0, :]))
        )
        return {
            "vector": vector,
            "nonfinite_feature_indices": nonfinite_feature_indices,
            "indicator_nan_cols": tuple(self.last_indicator_nan_cols),
        }


def _load_opened_series(parquet_path: Path) -> pd.Series:
    opened = pd.read_parquet(parquet_path, columns=["Opened"])["Opened"]
    return pd.to_datetime(opened)


def resolve_audit_window(
    *,
    parquet_path: Path,
    days_back: int = DEFAULT_AUDIT_DAYS_BACK,
    bootstrap_candles: int = DEFAULT_BOOTSTRAP_CANDLES,
    max_steps: int | None = None,
) -> AuditWindow:
    opened = _load_opened_series(parquet_path)
    if opened.empty:
        raise ValueError(f"Modeling parquet has no rows: {parquet_path}")

    audit_end = pd.Timestamp(opened.iloc[-1])
    requested_start = audit_end - pd.Timedelta(days=int(days_back))
    start_idx = int(opened.searchsorted(requested_start, side="left"))
    if start_idx >= len(opened):
        raise ValueError("Resolved audit_start is beyond the dataset end.")
    if start_idx < int(bootstrap_candles):
        raise ValueError(
            "Not enough history before the requested audit start. "
            f"Need bootstrap_candles={bootstrap_candles}, got only {start_idx} rows."
        )

    bootstrap_start_idx = start_idx - int(bootstrap_candles)
    if max_steps is None:
        end_idx = len(opened) - 1
    else:
        end_idx = min(len(opened) - 1, start_idx + int(max_steps) - 1)

    if end_idx < start_idx:
        raise ValueError("Audit window resolved to zero rows.")

    return AuditWindow(
        bootstrap_start=pd.Timestamp(opened.iloc[bootstrap_start_idx]),
        audit_start=pd.Timestamp(opened.iloc[start_idx]),
        audit_end=pd.Timestamp(opened.iloc[end_idx]),
        bootstrap_rows=int(bootstrap_candles),
        audit_rows=int(end_idx - start_idx + 1),
        requested_days_back=int(days_back),
        max_steps=None if max_steps is None else int(max_steps),
    )


def load_modeling_audit_frame(
    *,
    parquet_path: Path,
    feature_columns: list[str],
    audit_window: AuditWindow,
) -> pd.DataFrame:
    columns = ["Opened", *RAW_OHLCV_COLS, *feature_columns]
    frame = pd.read_parquet(
        parquet_path,
        columns=columns,
        filters=[
            ("Opened", ">=", audit_window.bootstrap_start.to_pydatetime().replace(tzinfo=None)),
            ("Opened", "<=", audit_window.audit_end.to_pydatetime().replace(tzinfo=None)),
        ],
    )
    frame["Opened"] = _ensure_utc_opened(frame["Opened"])
    frame = frame.sort_values("Opened").drop_duplicates(subset=["Opened"]).reset_index(drop=True)
    return frame


def load_anchor_volume_profile_history(
    *,
    parquet_path: Path,
    anchor_candle_opened: pd.Timestamp,
) -> pd.DataFrame:
    anchor_candle_opened = pd.Timestamp(anchor_candle_opened)
    frame = pd.read_parquet(
        parquet_path,
        columns=["Opened", "High", "Low", "Volume"],
        filters=[
            (
                "Opened",
                "<=",
                _naive_utc_timestamp(anchor_candle_opened),
            ),
        ],
    )
    frame["Opened"] = _ensure_utc_opened(frame["Opened"])
    frame = frame.sort_values("Opened").drop_duplicates(subset=["Opened"]).reset_index(drop=True)
    return frame


def build_or_load_anchor_volume_profile_state(
    *,
    parquet_path: Path,
    anchor_candle_opened: pd.Timestamp,
    overwrite: bool = False,
) -> tuple[dict[str, Any], Path]:
    state_path = resolve_anchor_volume_profile_state_path(anchor_candle_opened)
    if not overwrite and state_path.with_suffix(".npz").exists() and state_path.with_suffix(".json").exists():
        return load_volume_profile_state(state_path), state_path

    history_df = load_anchor_volume_profile_history(
        parquet_path=parquet_path,
        anchor_candle_opened=anchor_candle_opened,
    )
    state = bootstrap_state_from_history(
        history_df.loc[:, ["Opened", "High", "Low", "Volume"]],
        normalize_volume_profile_config(MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")),
    )
    save_volume_profile_state(state, state_path)
    return load_volume_profile_state(state_path), state_path


def run_stored_modeling_vs_current_recompute_audit(
    *,
    days_back: int = DEFAULT_AUDIT_DAYS_BACK,
    max_steps: int | None = None,
    history_tail_fraction: float = 1.0,
    model_meta_path: str | Path = MODEL_META_PATH,
    parquet_path: str | Path | None = None,
) -> dict[str, Any]:
    model_meta_path = Path(model_meta_path)
    parquet_path = _optional_path(parquet_path) or resolve_modeling_dataset_parquet_path()

    model, meta = load_model_and_meta(model_meta_path)
    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        raise ValueError("Missing feature_columns in model metadata.")

    audit_window = resolve_audit_window(
        parquet_path=parquet_path,
        days_back=days_back,
        bootstrap_candles=0,
        max_steps=max_steps,
    )
    stored_audit_df = load_modeling_audit_frame(
        parquet_path=parquet_path,
        feature_columns=feature_columns,
        audit_window=audit_window,
    )
    history_start, history_rows, history_total_rows = resolve_recent_history_tail_window(
        parquet_path=parquet_path,
        audit_end=audit_window.audit_end,
        tail_fraction=history_tail_fraction,
    )
    if audit_window.audit_start < history_start:
        raise ValueError(
            "stored_vs_current_recompute history tail is too short for the requested audit window. "
            f"audit_start={audit_window.audit_start.isoformat()} "
            f"history_start={history_start.isoformat()} "
            f"tail_fraction={float(history_tail_fraction):.6f}"
        )
    raw_history_df = load_modeling_raw_history_frame(
        parquet_path=parquet_path,
        audit_end=audit_window.audit_end,
        history_start=history_start,
    )
    recomputed_history_df = build_current_recomputed_feature_history(
        raw_history_df=raw_history_df,
        feature_columns=feature_columns,
    )
    recomputed_audit_df = align_feature_frame_to_audit_rows(
        audit_df=stored_audit_df,
        feature_frame=recomputed_history_df,
        feature_columns=feature_columns,
    )

    feature_group_by_name = _feature_group_map(feature_columns)
    feature_builder_frame = _feature_builder_frame(feature_columns)
    kelly_predictor = PseudoLiveAuditPredictor(
        raw_history_df.tail(max(1, min(len(raw_history_df), DEFAULT_BOOTSTRAP_CANDLES))).copy(),
        model_meta_path=model_meta_path,
        max_keep=max(DEFAULT_MAX_KEEP, len(raw_history_df.tail(DEFAULT_BOOTSTRAP_CANDLES))),
        volume_profile_state=None,
    )
    report = build_matrix_comparison_report(
        candidate_label="recomputed",
        reference_label="stored",
        candidate_matrix=recomputed_audit_df[feature_columns].to_numpy(dtype=np.float64, copy=True),
        reference_matrix=stored_audit_df[feature_columns].to_numpy(dtype=np.float64, copy=True),
        audit_df=stored_audit_df,
        feature_columns=feature_columns,
        feature_group_by_name=feature_group_by_name,
        feature_builder_frame=feature_builder_frame,
        model=model,
        kelly_predictor=kelly_predictor,
    )
    report["summary"] = pd.concat(
        [
            pd.Series(
                {
                    "parquet_path": str(parquet_path),
                    "audit_start": audit_window.audit_start.isoformat(),
                    "audit_end": audit_window.audit_end.isoformat(),
                    "audit_rows": audit_window.audit_rows,
                    "days_back": audit_window.requested_days_back,
                    "max_steps": audit_window.max_steps,
                    "history_tail_fraction": float(history_tail_fraction),
                    "history_start": history_start.isoformat(),
                    "history_rows": int(history_rows),
                    "history_total_rows": int(history_total_rows),
                    "report_type": "stored_modeling_vs_current_recompute",
                }
            ),
            report["summary"],
        ]
    )
    report["audit_window"] = audit_window
    report["audit_df"] = stored_audit_df
    report["stored_audit_df"] = stored_audit_df
    report["recomputed_audit_df"] = recomputed_audit_df
    report["candidate_feature_frame"] = recomputed_audit_df.loc[:, feature_columns].copy()
    report["feature_columns"] = feature_columns
    return report


def run_live_modeling_feature_audit(
    *,
    days_back: int = DEFAULT_AUDIT_DAYS_BACK,
    bootstrap_candles: int = DEFAULT_BOOTSTRAP_CANDLES,
    max_steps: int | None = None,
    max_keep: int = DEFAULT_MAX_KEEP,
    model_meta_path: str | Path = MODEL_META_PATH,
    parquet_path: str | Path | None = None,
    use_anchor_vp_state: bool = True,
    overwrite_anchor_vp_state: bool = False,
) -> dict[str, Any]:
    model_meta_path = Path(model_meta_path)
    parquet_path = _optional_path(parquet_path) or resolve_modeling_dataset_parquet_path()
    progress_enabled = bool(AUDIT_PROGRESS_ENABLED)
    progress_every = max(1, int(AUDIT_PROGRESS_EVERY_STEPS))

    if progress_enabled:
        print(
            "[audit] starting live_vs_stored "
            f"days_back={int(days_back)} "
            f"bootstrap_candles={int(bootstrap_candles)} "
            f"max_steps={max_steps if max_steps is not None else 'all'} "
            f"max_keep={int(max_keep)}"
        )
        print(
            "[audit] inputs "
            f"model_meta={model_meta_path} "
            f"parquet={parquet_path}"
        )

    model, meta = load_model_and_meta(model_meta_path)
    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        raise ValueError("Missing feature_columns in model metadata.")

    audit_window = resolve_audit_window(
        parquet_path=parquet_path,
        days_back=days_back,
        bootstrap_candles=bootstrap_candles,
        max_steps=max_steps,
    )
    if progress_enabled:
        print(
            "[audit] window "
            f"bootstrap_start={audit_window.bootstrap_start.isoformat()} "
            f"audit_start={audit_window.audit_start.isoformat()} "
            f"audit_end={audit_window.audit_end.isoformat()} "
            f"bootstrap_rows={int(audit_window.bootstrap_rows)} "
            f"audit_rows={int(audit_window.audit_rows)}"
        )
    modeling_frame = load_modeling_audit_frame(
        parquet_path=parquet_path,
        feature_columns=feature_columns,
        audit_window=audit_window,
    )

    if len(modeling_frame) != audit_window.bootstrap_rows + audit_window.audit_rows:
        raise RuntimeError(
            "Unexpected audit frame length. "
            f"Expected {audit_window.bootstrap_rows + audit_window.audit_rows}, "
            f"got {len(modeling_frame)}."
        )

    bootstrap_df = modeling_frame.iloc[: audit_window.bootstrap_rows].copy()
    audit_df = modeling_frame.iloc[audit_window.bootstrap_rows :].copy().reset_index(drop=True)
    if audit_df.empty:
        raise RuntimeError("Audit dataframe is empty after splitting bootstrap rows.")

    anchor_candle_opened = pd.Timestamp(bootstrap_df["Opened"].iloc[-1])
    anchor_vp_state = None
    anchor_vp_state_path = None
    if use_anchor_vp_state:
        if progress_enabled:
            print(
                "[audit] loading anchor vp state "
                f"anchor_opened={anchor_candle_opened.isoformat()} "
                f"overwrite={bool(overwrite_anchor_vp_state)}"
            )
        anchor_vp_state, anchor_vp_state_path = build_or_load_anchor_volume_profile_state(
            parquet_path=parquet_path,
            anchor_candle_opened=anchor_candle_opened,
            overwrite=overwrite_anchor_vp_state,
        )
        if progress_enabled:
            print(
                "[audit] anchor vp state ready "
                f"path={anchor_vp_state_path.with_suffix('.npz')}"
            )

    predictor = PseudoLiveAuditPredictor(
        bootstrap_df,
        model_meta_path=model_meta_path,
        max_keep=max_keep,
        volume_profile_state=anchor_vp_state,
    )
    predictor.model = model
    if progress_enabled:
        print(
            "[audit] predictor ready "
            f"required_stable_window={int(predictor.required_stable_window)} "
            f"bootstrap_rows_loaded={int(predictor.bootstrap_candles)} "
            f"initial_buffer_rows={int(len(predictor.opened_candles))}"
        )

    feature_group_by_name = _feature_group_map(feature_columns)
    feature_builder_frame = _feature_builder_frame(feature_columns)
    indicator_nan_count_rows: list[int] = []
    history_rows_used_rows: list[int] = []
    history_shortfall_rows: list[int] = []
    decision_row_indices: list[int] = []
    live_vector_rows: list[np.ndarray] = []
    live_nonfinite_rows: list[np.ndarray] = []

    ohlcv_matrix = audit_df[list(RAW_OHLCV_COLS)].to_numpy(dtype=np.float64, copy=False)
    opened_values = audit_df["Opened"].to_list()
    loop_started_at = time.perf_counter()
    total_steps = len(opened_values)
    decision_steps = 0

    for row_idx, opened in enumerate(opened_values):
        predictor._append_new_candle(
            pd.Timestamp(opened),
            tuple(float(v) for v in ohlcv_matrix[row_idx, :]),
        )
        volume_profile_values = predictor._extract_volume_profile_features_for_latest_candle()
        is_decision_step = _is_live_decision_opened(opened, predictor.target_bucket_minutes)
        if is_decision_step:
            snapshot = predictor.build_feature_snapshot(volume_profile_values)
            live_vector_rows.append(snapshot["vector"][0, :].astype(np.float64, copy=True))
            nonfinite_row = np.zeros(len(feature_columns), dtype=bool)
            if snapshot["nonfinite_feature_indices"]:
                nonfinite_row[list(snapshot["nonfinite_feature_indices"])] = True
            live_nonfinite_rows.append(nonfinite_row)
            indicator_nan_count_rows.append(len(snapshot["indicator_nan_cols"]))
            current_history_rows = int(len(predictor.opened_candles))
            history_rows_used_rows.append(current_history_rows)
            history_shortfall_rows.append(
                max(0, int(predictor.required_stable_window) - current_history_rows)
            )
            decision_row_indices.append(row_idx)
            decision_steps += 1
        predictor._update_volume_profile_state_for_latest_candle(opened)
        completed_steps = row_idx + 1
        if progress_enabled and (
            completed_steps == 1
            or completed_steps == total_steps
            or completed_steps % progress_every == 0
        ):
            elapsed_sec = time.perf_counter() - loop_started_at
            steps_per_sec = completed_steps / elapsed_sec if elapsed_sec > 0 else float("nan")
            remaining_steps = total_steps - completed_steps
            eta_sec = (
                remaining_steps / steps_per_sec
                if np.isfinite(steps_per_sec) and steps_per_sec > 0
                else float("nan")
            )
            print(
                "[audit] progress "
                f"{completed_steps}/{total_steps} "
                f"({(completed_steps / total_steps) * 100.0:.1f}%) "
                f"opened={pd.Timestamp(opened).isoformat()} "
                f"buffer_rows={int(len(predictor.opened_candles))} "
                f"decision_rows={decision_steps} "
                f"elapsed={_format_duration(elapsed_sec)} "
                f"eta={_format_duration(eta_sec) if np.isfinite(eta_sec) else 'n/a'}"
            )

    decision_audit_df = audit_df.iloc[decision_row_indices].copy().reset_index(drop=True)
    if decision_audit_df.empty:
        raise RuntimeError("No live decision rows were selected for the audit window.")
    modeling_matrix = decision_audit_df[feature_columns].to_numpy(dtype=np.float64, copy=True)
    live_matrix = np.vstack(live_vector_rows).astype(np.float64, copy=False)
    live_nonfinite_mask = np.vstack(live_nonfinite_rows).astype(bool, copy=False)
    indicator_nan_count = np.asarray(indicator_nan_count_rows, dtype=np.int32)
    history_rows_used = np.asarray(history_rows_used_rows, dtype=np.int32)
    history_shortfall = np.asarray(history_shortfall_rows, dtype=np.int32)

    if progress_enabled:
        print("[audit] building comparison report")
    live_report = build_matrix_comparison_report(
        candidate_label="live",
        reference_label="stored",
        candidate_matrix=live_matrix,
        reference_matrix=modeling_matrix,
        audit_df=decision_audit_df,
        feature_columns=feature_columns,
        feature_group_by_name=feature_group_by_name,
        feature_builder_frame=feature_builder_frame,
        model=model,
        kelly_predictor=predictor,
    )
    live_report["step_summary_df"]["indicator_nan_count"] = indicator_nan_count
    live_report["step_summary_df"]["history_rows_used"] = history_rows_used
    live_report["step_summary_df"]["history_shortfall"] = history_shortfall
    live_report["summary"] = pd.concat(
        [
            pd.Series(
                {
                    "parquet_path": str(parquet_path),
                    "audit_start": audit_window.audit_start.isoformat(),
                    "audit_end": audit_window.audit_end.isoformat(),
                    "bootstrap_rows": audit_window.bootstrap_rows,
                    "audit_rows_total_1m": audit_window.audit_rows,
                    "decision_rows": len(decision_audit_df),
                    "days_back": audit_window.requested_days_back,
                    "max_steps": audit_window.max_steps,
                    "max_keep": int(max_keep),
                    "required_stable_window": int(predictor.required_stable_window),
                    "use_anchor_vp_state": bool(use_anchor_vp_state),
                    "anchor_vp_state_path": None
                    if anchor_vp_state_path is None
                    else str(anchor_vp_state_path.with_suffix(".npz")),
                    "report_type": "pseudo_live_vs_stored_modeling_decision_only",
                }
            ),
            live_report["summary"],
        ]
    )

    live_feature_frame = pd.DataFrame(
        live_matrix,
        columns=feature_columns,
        index=decision_audit_df.index,
    )
    live_report["audit_window"] = audit_window
    live_report["audit_df"] = decision_audit_df
    live_report["live_feature_frame"] = live_feature_frame
    live_report["candidate_feature_frame"] = live_feature_frame
    live_report["feature_columns"] = feature_columns
    if progress_enabled:
        print("[audit] building drift reason report")
    live_report["drift_reason_report"] = build_live_drift_reason_report(
        live_report,
        top_n=AUDIT_TOP_N,
    )

    return {
        "summary": live_report["summary"],
        "audit_window": audit_window,
        "modeling_frame": modeling_frame,
        "audit_df": decision_audit_df,
        "live_feature_frame": live_feature_frame,
        "step_summary_df": live_report["step_summary_df"],
        "feature_summary_df": live_report["feature_summary_df"],
        "group_summary_df": live_report["group_summary_df"],
        "builder_summary_df": live_report["builder_summary_df"],
        "live_nonfinite_mask": live_report["candidate_nonfinite_mask"],
        "modeling_nonfinite_mask": live_report["reference_nonfinite_mask"],
        "finite_status_mismatch_mask": live_report["finite_status_mismatch_mask"],
        "feature_columns": feature_columns,
        "feature_builder_frame": feature_builder_frame,
        "live_vs_stored_report": live_report,
        "drift_reason_report": live_report["drift_reason_report"],
    }


def feature_drilldown(
    results: dict[str, Any],
    feature_name: str,
    *,
    report_key: str | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    report = results if report_key is None else results[report_key]
    feature_columns = list(report["feature_columns"])
    if feature_name not in feature_columns:
        raise KeyError(f"Unknown feature_name={feature_name!r}")

    feature_idx = feature_columns.index(feature_name)
    audit_df = report["audit_df"]
    candidate_feature_frame = report.get("candidate_feature_frame", report.get("live_feature_frame"))
    step_summary_df = report["step_summary_df"]
    candidate_nonfinite_mask = report.get(
        "candidate_nonfinite_mask",
        report.get("live_nonfinite_mask"),
    )
    reference_nonfinite_mask = report.get(
        "reference_nonfinite_mask",
        report.get("modeling_nonfinite_mask"),
    )
    candidate_series = candidate_feature_frame[feature_name].to_numpy(dtype=np.float64, copy=False)
    reference_series = audit_df[feature_name].to_numpy(dtype=np.float64, copy=False)
    abs_diff = np.abs(candidate_series - reference_series)
    abs_diff[~(np.isfinite(candidate_series) & np.isfinite(reference_series))] = np.nan
    rel_diff = _safe_relative_diff(candidate_series, reference_series)
    candidate_label = str(report.get("candidate_label", "candidate"))
    reference_label = str(report.get("reference_label", "reference"))
    candidate_proba_col = f"{candidate_label}_proba_up"
    reference_proba_col = f"{reference_label}_proba_up"

    drilldown_df = pd.DataFrame(
        {
            "Opened": audit_df["Opened"],
            "Open": audit_df["Open"],
            "High": audit_df["High"],
            "Low": audit_df["Low"],
            "Close": audit_df["Close"],
            "Volume": audit_df["Volume"],
            "stored_value": reference_series,
            "candidate_value": candidate_series,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "candidate_nonfinite": candidate_nonfinite_mask[:, feature_idx],
            "stored_nonfinite": reference_nonfinite_mask[:, feature_idx],
            "finite_status_mismatch": report["finite_status_mismatch_mask"][:, feature_idx],
            "proba_up_abs_diff": step_summary_df["proba_up_abs_diff"],
        }
    )
    proba_cols = [col for col in step_summary_df.columns if col.endswith("_proba_up")]
    for col in proba_cols:
        drilldown_df[col] = step_summary_df[col]
    fixed_pred_matrix = report.get("single_feature_fixed_pred_matrix")
    proba_shift_matrix = report.get("single_feature_proba_shift_matrix")
    gap_reduction_matrix = report.get("single_feature_gap_reduction_matrix")
    if (
        fixed_pred_matrix is not None
        and proba_shift_matrix is not None
        and gap_reduction_matrix is not None
        and candidate_proba_col in step_summary_df.columns
        and reference_proba_col in step_summary_df.columns
    ):
        fixed_pred_series = np.asarray(fixed_pred_matrix, dtype=np.float64)[:, feature_idx]
        proba_shift_series = np.asarray(proba_shift_matrix, dtype=np.float64)[:, feature_idx]
        gap_reduction_series = np.asarray(gap_reduction_matrix, dtype=np.float64)[:, feature_idx]
        reference_pred = step_summary_df[reference_proba_col].to_numpy(dtype=np.float64, copy=False)
        candidate_pred = step_summary_df[candidate_proba_col].to_numpy(dtype=np.float64, copy=False)
        base_signal_mismatch = (candidate_pred >= 0.5) != (reference_pred >= 0.5)
        fixed_signal_mismatch = (fixed_pred_series >= 0.5) != (reference_pred >= 0.5)
        drilldown_df["proba_up_if_feature_fixed"] = fixed_pred_series
        drilldown_df["proba_up_shift_if_feature_fixed"] = proba_shift_series
        drilldown_df["abs_proba_gap_reduction_if_feature_fixed"] = gap_reduction_series
        drilldown_df["signal_mismatch_resolved_if_feature_fixed"] = (
            base_signal_mismatch & ~fixed_signal_mismatch
        ).astype(np.int8)
        drilldown_df["signal_mismatch_introduced_if_feature_fixed"] = (
            ~base_signal_mismatch & fixed_signal_mismatch
        ).astype(np.int8)

    sort_spec = [
        ("abs_proba_gap_reduction_if_feature_fixed", False),
        ("proba_up_abs_diff", False),
        ("abs_diff", False),
    ]
    sort_cols = [col for col, _ascending in sort_spec if col in drilldown_df.columns]
    sort_ascending = [ascending for col, ascending in sort_spec if col in drilldown_df.columns]
    return drilldown_df.sort_values(
        sort_cols,
        ascending=sort_ascending,
        na_position="last",
        kind="stable",
    ).head(int(top_n))


def _build_feature_prediction_impact_export_df(
    feature_summary_df: pd.DataFrame,
    *,
    top_k: int | None = None,
    only_impactful: bool = False,
) -> pd.DataFrame:
    slim_columns = [
        "rank",
        "feature",
        "group",
        "builder",
        "mean_pred_gap_reduction",
        "max_pred_gap_reduction",
        "mean_pred_shift",
        "max_pred_shift",
        "signal_flips_resolved",
        "proba_drift_rows_resolved",
        "rows_helped",
        "rows_hurt",
    ]
    if set(slim_columns).issubset(feature_summary_df.columns):
        export_df = (
            feature_summary_df.loc[:, slim_columns]
            .sort_values(["rank"], ascending=[True], kind="stable")
            .reset_index(drop=True)
        )
        if only_impactful and not export_df.empty:
            impact_mask = (
                export_df["rows_helped"].gt(0)
                | export_df["signal_flips_resolved"].gt(0)
                | export_df["proba_drift_rows_resolved"].gt(0)
                | export_df["max_pred_shift"].abs().gt(PREDICTION_DIFF_TOL)
            )
            export_df = export_df.loc[impact_mask].reset_index(drop=True)
            export_df["rank"] = np.arange(1, len(export_df) + 1, dtype=np.int32)
            export_df = export_df.loc[:, slim_columns]
        if top_k is not None:
            export_df = export_df.head(int(top_k)).reset_index(drop=True)
        return export_df

    export_columns = [
        "feature",
        "group",
        "builder_name",
        "mean_abs_proba_gap_reduction_if_fixed",
        "max_abs_proba_gap_reduction_if_fixed",
        "mean_abs_proba_shift_if_fixed",
        "max_abs_proba_shift_if_fixed",
        "rows_signal_mismatch_resolved_if_fixed",
        "rows_proba_diff_gt_tol_resolved_if_fixed",
        "rows_abs_proba_gap_reduced_if_fixed",
        "rows_abs_proba_gap_worsened_if_fixed",
    ]
    rename_map = {
        "builder_name": "builder",
        "mean_abs_proba_gap_reduction_if_fixed": "mean_pred_gap_reduction",
        "max_abs_proba_gap_reduction_if_fixed": "max_pred_gap_reduction",
        "mean_abs_proba_shift_if_fixed": "mean_pred_shift",
        "max_abs_proba_shift_if_fixed": "max_pred_shift",
        "rows_signal_mismatch_resolved_if_fixed": "signal_flips_resolved",
        "rows_proba_diff_gt_tol_resolved_if_fixed": "proba_drift_rows_resolved",
        "rows_abs_proba_gap_reduced_if_fixed": "rows_helped",
        "rows_abs_proba_gap_worsened_if_fixed": "rows_hurt",
    }
    if feature_summary_df.empty:
        return pd.DataFrame(columns=slim_columns)

    sort_spec = [
        ("mean_abs_proba_gap_reduction_if_fixed", False),
        ("max_abs_proba_gap_reduction_if_fixed", False),
        ("mean_abs_proba_shift_if_fixed", False),
        ("max_abs_proba_shift_if_fixed", False),
        ("rows_signal_mismatch_resolved_if_fixed", False),
        ("rows_proba_diff_gt_tol_resolved_if_fixed", False),
        ("rows_abs_proba_gap_reduced_if_fixed", False),
        ("feature", True),
    ]
    sort_cols = [col for col, _ascending in sort_spec if col in feature_summary_df.columns]
    sort_ascending = [ascending for col, ascending in sort_spec if col in feature_summary_df.columns]
    export_df = (
        feature_summary_df.sort_values(
            sort_cols,
            ascending=sort_ascending,
            na_position="last",
            kind="stable",
        )
        .loc[:, export_columns]
        .rename(columns=rename_map)
        .reset_index(drop=True)
    )
    export_df.insert(0, "rank", np.arange(1, len(export_df) + 1, dtype=np.int32))
    if only_impactful and not export_df.empty:
        impact_mask = (
            export_df["rows_helped"].gt(0)
            | export_df["signal_flips_resolved"].gt(0)
            | export_df["proba_drift_rows_resolved"].gt(0)
            | export_df["max_pred_shift"].abs().gt(PREDICTION_DIFF_TOL)
        )
        export_df = export_df.loc[impact_mask].reset_index(drop=True)
        export_df["rank"] = np.arange(1, len(export_df) + 1, dtype=np.int32)
        export_df = export_df.loc[:, slim_columns]
    if top_k is not None:
        export_df = export_df.head(int(top_k)).reset_index(drop=True)
    return export_df


def _feature_prediction_impact_records(
    feature_summary_df: pd.DataFrame,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    export_df = _build_feature_prediction_impact_export_df(
        feature_summary_df,
        top_k=top_k,
        only_impactful=True,
    )
    return json.loads(export_df.to_json(orient="records"))


def save_audit_outputs(
    results: dict[str, Any],
    *,
    output_dir: Path,
    drilldown_feature_name: str | None = None,
    top_n: int = 50,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: dict[str, Path] = {}
    report = results["live_vs_stored_report"]
    drift_reason_report = results["drift_reason_report"]

    written_paths["summary_json"] = output_dir / "live_vs_stored_summary.json"
    written_paths["decision_rows_csv"] = output_dir / "live_vs_stored_decision_rows.csv"
    written_paths["feature_prediction_impacts_csv"] = (
        output_dir / "live_vs_stored_feature_prediction_impacts.csv"
    )
    written_paths["feature_prediction_impacts_full_csv"] = (
        output_dir / "live_vs_stored_feature_prediction_impacts_full.csv"
    )
    written_paths["builder_prediction_impacts_csv"] = (
        output_dir / "live_vs_stored_builder_prediction_impacts.csv"
    )
    written_paths["group_prediction_impacts_csv"] = (
        output_dir / "live_vs_stored_group_prediction_impacts.csv"
    )
    feature_export_df = _build_feature_prediction_impact_export_df(
        report["feature_summary_df"],
        only_impactful=True,
    )
    top10_feature_records = _feature_prediction_impact_records(
        report["feature_summary_df"],
        top_k=10,
    )
    top25_feature_records = _feature_prediction_impact_records(
        report["feature_summary_df"],
        top_k=25,
    )

    written_paths["summary_json"].write_text(
        json.dumps(
            {
                "live_vs_stored": report["summary"].to_dict(),
                "drift_reason": drift_reason_report["summary"].to_dict(),
                "prediction_impact_feature_rank_metric": "mean_pred_gap_reduction",
                "top10_prediction_impact_features": top10_feature_records,
                "top25_prediction_impact_features": top25_feature_records,
            },
            indent=2,
            ensure_ascii=True,
            default=str,
        ),
        encoding="utf-8",
    )
    decision_sort_spec = [
        ("top_prediction_impact_abs_proba_gap_reduction_if_fixed", False),
        ("proba_up_abs_diff", False),
        ("top_prediction_impact_abs_proba_shift_if_fixed", False),
        ("feature_mean_rel_diff", False),
        ("feature_mean_abs_diff", False),
        ("Opened", True),
    ]
    decision_sort_cols = [col for col, _ascending in decision_sort_spec if col in report["step_summary_df"]]
    decision_sort_ascending = [
        ascending for col, ascending in decision_sort_spec if col in report["step_summary_df"]
    ]
    decision_rows_export_df = report["step_summary_df"].sort_values(
        decision_sort_cols,
        ascending=decision_sort_ascending,
        kind="stable",
        na_position="last",
    )
    decision_rows_export_df.to_csv(written_paths["decision_rows_csv"], index=False)
    feature_export_df.to_csv(
        written_paths["feature_prediction_impacts_csv"],
        index=False,
    )
    report["feature_summary_df"].to_csv(
        written_paths["feature_prediction_impacts_full_csv"],
        index=False,
    )
    report["builder_summary_df"].to_csv(
        written_paths["builder_prediction_impacts_csv"],
        index=False,
    )
    report["group_summary_df"].to_csv(
        written_paths["group_prediction_impacts_csv"],
        index=False,
    )

    if drilldown_feature_name:
        drilldown_path = output_dir / f"live_vs_stored_drilldown_{drilldown_feature_name}.csv"
        feature_drilldown(
            results,
            drilldown_feature_name,
            report_key="live_vs_stored_report",
            top_n=top_n,
        ).to_csv(drilldown_path, index=False)
        written_paths["drilldown_csv"] = drilldown_path

    return written_paths


def _default_output_dir() -> Path:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    return Path("data/analysis/live_feature_parity") / stamp


def main() -> None:
    started_at = time.perf_counter()
    results = run_live_modeling_feature_audit(
        days_back=AUDIT_DAYS_BACK,
        bootstrap_candles=AUDIT_BOOTSTRAP_CANDLES,
        max_steps=AUDIT_MAX_STEPS,
        max_keep=AUDIT_MAX_KEEP,
        model_meta_path=AUDIT_MODEL_META_PATH,
        parquet_path=AUDIT_PARQUET_PATH,
        use_anchor_vp_state=AUDIT_USE_ANCHOR_VP_STATE,
        overwrite_anchor_vp_state=AUDIT_OVERWRITE_ANCHOR_VP_STATE,
    )
    output_dir = _optional_path(AUDIT_OUTPUT_DIR) or _default_output_dir()
    if AUDIT_PROGRESS_ENABLED:
        print(f"[audit] saving outputs -> {output_dir}")
    written_paths = save_audit_outputs(
        results,
        output_dir=output_dir,
        drilldown_feature_name=AUDIT_DRILLDOWN_FEATURE,
        top_n=AUDIT_TOP_N,
    )
    if AUDIT_PROGRESS_ENABLED:
        print(
            "[audit] finished "
            f"elapsed={_format_duration(time.perf_counter() - started_at)}"
        )

    summary = results["live_vs_stored_report"]["summary"]
    reason_summary = results["drift_reason_report"]["summary"]
    business_keys = [
        "decision_row_count",
        "rows_with_signal_mismatch",
        "rows_with_business_decision_mismatch",
        "rows_with_kelly_side_mismatch",
        "rows_with_kelly_trade_flag_mismatch",
        "rows_with_kelly_reason_mismatch",
        "rows_with_stake_only_kelly_mismatch",
        "rows_with_proba_diff_gt_tol",
        "max_proba_up_abs_diff",
        "max_mean_abs_proba_gap_reduction_if_fixed",
        "max_mean_abs_proba_shift_if_fixed",
        "max_kelly_stake_abs_diff",
    ]
    available_business_keys = [key for key in business_keys if key in summary.index]
    print("Live vs stored business summary:")
    print(summary.loc[available_business_keys].to_string())
    print()
    print("Dominant prediction-impact source:")
    dominant_keys = [
        "explanation_basis",
        "dominant_prediction_impact_group",
        "dominant_prediction_impact_builder_family",
        "dominant_prediction_impact_builder_name",
        "dominant_prediction_impact_feature",
    ]
    available_dominant_keys = [key for key in dominant_keys if key in reason_summary.index]
    print(reason_summary.loc[available_dominant_keys].to_string())
    print()
    print("Top drift rows:")
    print(results["drift_reason_report"]["row_summary_df"].to_string(index=False))
    print()
    print("Top prediction-impact features:")
    print(
        _build_feature_prediction_impact_export_df(
            results["live_vs_stored_report"]["feature_summary_df"],
            top_k=min(25, AUDIT_TOP_N),
            only_impactful=True,
        ).to_string(index=False)
    )
    print()
    print("Top prediction-impact builders:")
    print(results["drift_reason_report"]["builder_summary_df"].to_string(index=False))
    print()
    print("Top prediction-impact groups:")
    print(results["drift_reason_report"]["group_summary_df"].to_string(index=False))
    if AUDIT_DRILLDOWN_FEATURE:
        print()
        print(f"Live vs stored drilldown [{AUDIT_DRILLDOWN_FEATURE}]:")
        print(
            feature_drilldown(
                results,
                AUDIT_DRILLDOWN_FEATURE,
                report_key="live_vs_stored_report",
                top_n=AUDIT_TOP_N,
            ).to_string(index=False)
        )
    print()
    print("Wrote:")
    for label, path in written_paths.items():
        print(f"  {label}: {path}")


__all__ = [
    "AuditWindow",
    "DEFAULT_AUDIT_DAYS_BACK",
    "DEFAULT_BOOTSTRAP_CANDLES",
    "DEFAULT_MAX_KEEP",
    "PseudoLiveAuditPredictor",
    "build_or_load_anchor_volume_profile_state",
    "build_live_drift_reason_report",
    "feature_drilldown",
    "load_modeling_audit_frame",
    "resolve_audit_window",
    "resolve_anchor_volume_profile_state_path",
    "run_live_modeling_feature_audit",
    "save_audit_outputs",
]


if __name__ == "__main__":
    main()
