import copy
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from create_modeling_dataset import (
    add_indicator_values,
    concat_feature_frame,
    parse_fit_results,
)
from features.basis_premium_features import validate_basis_premium_feature_columns
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
    add_session_open_features,
    build_latest_session_open_feature_dict_fast,
)
from features.realized_volatility import add_realized_volatility_features
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
    state_matches_config as volume_profile_state_matches_config,
    update_state_with_candle as update_volume_profile_state_with_candle,
    validate_volume_profile_feature_columns,
    validate_volume_profile_model_metadata,
)
from live_predict_binance import (
    INDICATOR_HISTORY_REQUIREMENTS_PATH,
    INTERVAL,
    TRADE_POLICY_CONFIG_PATH,
    LIVE_INITIAL_BANKROLL_USDC,
    MODELING_DATASET_SETTINGS,
    MODEL_META_PATH,
    OHLCV_COLS,
    SYMBOL,
    LivePredictor,
    interval_to_timedelta,
    load_indicator_history_requirements,
    load_trade_policy_runtime_config,
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
DEFAULT_BOOTSTRAP_CANDLES = 21_600
DEFAULT_MAX_KEEP = DEFAULT_BOOTSTRAP_CANDLES
PREDICTION_DIFF_TOL = 1e-6
REL_DIFF_DENOM_FLOOR = 1e-6
FEATURE_DROP_MAX_ABS_DIFF_TOL = 1e-2
FEATURE_DROP_MEAN_ABS_DIFF_TOL = 1e-3
FEATURE_DROP_MAX_REL_DIFF_TOL = 1e-2

# Runtime settings
AUDIT_DAYS_BACK = DEFAULT_AUDIT_DAYS_BACK
AUDIT_BOOTSTRAP_CANDLES = DEFAULT_BOOTSTRAP_CANDLES
AUDIT_MAX_STEPS = 10_080
AUDIT_MAX_KEEP = AUDIT_BOOTSTRAP_CANDLES
AUDIT_MODEL_META_PATH = MODEL_META_PATH
AUDIT_PARQUET_PATH = None
AUDIT_USE_ANCHOR_VP_STATE = True
AUDIT_OVERWRITE_ANCHOR_VP_STATE = False
AUDIT_ALLOW_UNSTABLE_INDICATOR_SUMMARY = True
AUDIT_OUTPUT_DIR = None
AUDIT_DRILLDOWN_FEATURE = None
AUDIT_TOP_N = 50
AUDIT_PROGRESS_ENABLED = True
AUDIT_PROGRESS_EVERY_STEPS = 60
AUDIT_PROGRESS_MAX_UPDATES = 12
AUDIT_CONSOLE_TOP_N = 5
AUDIT_CONSOLE_LABEL_MAX_LEN = 100
AUDIT_MAX_MEAN_PROBA_ABS_DIFF = None
AUDIT_MAX_MAX_PROBA_ABS_DIFF = None
AUDIT_MAX_SIGNAL_MISMATCH_RATE = None
AUDIT_MAX_BUSINESS_MISMATCH_RATE = None


def _env_optional_float(name, default=None):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


AUDIT_MAX_MEAN_PROBA_ABS_DIFF = _env_optional_float(
    "AUDIT_MAX_MEAN_PROBA_ABS_DIFF",
    AUDIT_MAX_MEAN_PROBA_ABS_DIFF,
)
AUDIT_MAX_MAX_PROBA_ABS_DIFF = _env_optional_float(
    "AUDIT_MAX_MAX_PROBA_ABS_DIFF",
    AUDIT_MAX_MAX_PROBA_ABS_DIFF,
)
AUDIT_MAX_SIGNAL_MISMATCH_RATE = _env_optional_float(
    "AUDIT_MAX_SIGNAL_MISMATCH_RATE",
    AUDIT_MAX_SIGNAL_MISMATCH_RATE,
)
AUDIT_MAX_BUSINESS_MISMATCH_RATE = _env_optional_float(
    "AUDIT_MAX_BUSINESS_MISMATCH_RATE",
    AUDIT_MAX_BUSINESS_MISMATCH_RATE,
)


def _ensure_utc_opened(series):
    opened = pd.to_datetime(series)
    if getattr(opened.dt, "tz", None) is None:
        return opened.dt.tz_localize("UTC")
    return opened.dt.tz_convert("UTC")


def _naive_utc_timestamp(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)


def _optional_path(value):
    if value is None:
        return None
    return Path(value)


def _load_model_feature_importance_frame(meta):
    artifacts = dict(meta.get("artifacts") or {})
    raw_path = str(artifacts.get("final_feature_importance_csv") or "").strip()
    if not raw_path:
        return None

    path = Path(raw_path)
    if not path.exists():
        return None

    frame = pd.read_csv(path)
    required_cols = {"feature", "importance_gain", "importance_split"}
    if not required_cols.issubset(frame.columns):
        return None
    validate_volume_profile_feature_columns(
        frame["feature"].tolist(),
        source_label=f"feature importance artifact {path}",
    )
    validate_basis_premium_feature_columns(
        frame["feature"].tolist(),
        source_label=f"feature importance artifact {path}",
    )

    out = frame.loc[:, ["feature", "importance_gain", "importance_split"]].copy()
    out["importance_gain"] = pd.to_numeric(
        out["importance_gain"], errors="coerce"
    ).fillna(0.0)
    out["importance_split"] = pd.to_numeric(
        out["importance_split"], errors="coerce"
    ).fillna(0.0)
    return out


def resolve_anchor_volume_profile_state_path(anchor_candle_opened):
    stamp = pd.Timestamp(anchor_candle_opened).strftime("%Y%m%d_%H%M")
    return (
        VP_AUDIT_ANCHOR_STATE_DIR
        / f"{SYMBOL}_{INTERVAL}_{VP_FEATURE_VERSION}_audit_anchor_{stamp}"
    )


def _feature_group_map(feature_columns):
    parts = split_feature_subset(feature_columns, source_label="audit feature columns")
    group_by_feature = {}
    for feature in parts["raw_ohlcv_cols"]:
        group_by_feature[feature] = "raw_ohlcv"
    for feature in parts["candle_feature_cols"]:
        group_by_feature[feature] = "candle"
    for feature in parts["streak_feature_cols"]:
        group_by_feature[feature] = "streak"
    for feature in parts["session_feature_cols"]:
        group_by_feature[feature] = "session"
    for feature in parts["realized_volatility_feature_cols"]:
        group_by_feature[feature] = "realized_volatility"
    for feature in parts["basis_premium_feature_cols"]:
        group_by_feature[feature] = "basis_premium"
    for feature in parts["indicator_feature_cols"]:
        group_by_feature[feature] = "indicator"
    for feature in parts["volume_profile_feature_cols"]:
        group_by_feature[feature] = "volume_profile"
    for feature in parts["unclassified_feature_cols"]:
        group_by_feature[feature] = "unclassified"
    return group_by_feature


def _ignored_basis_premium_feature_columns(feature_columns):
    parts = split_feature_subset(feature_columns, source_label="audit feature columns")
    return tuple(parts["basis_premium_feature_cols"])


def _audited_feature_columns(feature_columns):
    ignored = set(_ignored_basis_premium_feature_columns(feature_columns))
    return [feature for feature in feature_columns if feature not in ignored]


def _copy_ignored_basis_premium_features(candidate_df, reference_df, feature_columns):
    ignored_cols = _ignored_basis_premium_feature_columns(feature_columns)
    if not ignored_cols:
        return candidate_df

    missing_reference_cols = [col for col in ignored_cols if col not in reference_df]
    if missing_reference_cols:
        raise ValueError(
            "Stored audit frame is missing ignored basis premium columns: "
            f"{missing_reference_cols[:10]}"
        )

    out = candidate_df.copy()
    for col in ignored_cols:
        out[col] = reference_df[col].to_numpy(dtype=np.float64, copy=False)

    leading_cols = [col for col in out.columns if col not in feature_columns]
    return out.loc[:, [*leading_cols, *feature_columns]].copy()


def _safe_rowwise_mean(values):
    counts = np.isfinite(values).sum(axis=1)
    sums = np.nansum(values, axis=1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def _format_duration(seconds):
    seconds = max(0.0, float(seconds))
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _summary_rate(count, total):
    total = int(total)
    if total <= 0:
        return float("nan")
    return float(count) / float(total)


def _evaluate_guardrail_violations(summary):
    decision_row_count = int(summary.get("decision_row_count", 0) or 0)
    signal_mismatch_rate = float(
        summary.get(
            "signal_mismatch_rate",
            _summary_rate(summary.get("rows_with_signal_mismatch", 0), decision_row_count),
        )
    )
    business_mismatch_rate = float(
        summary.get(
            "business_decision_mismatch_rate",
            _summary_rate(
                summary.get("rows_with_business_decision_mismatch", 0),
                decision_row_count,
            ),
        )
    )

    checks = [
        (
            AUDIT_MAX_MEAN_PROBA_ABS_DIFF,
            "mean_proba_up_abs_diff",
            float(summary.get("mean_proba_up_abs_diff", np.nan)),
        ),
        (
            AUDIT_MAX_MAX_PROBA_ABS_DIFF,
            "max_proba_up_abs_diff",
            float(summary.get("max_proba_up_abs_diff", np.nan)),
        ),
        (
            AUDIT_MAX_SIGNAL_MISMATCH_RATE,
            "signal_mismatch_rate",
            signal_mismatch_rate,
        ),
        (
            AUDIT_MAX_BUSINESS_MISMATCH_RATE,
            "business_decision_mismatch_rate",
            business_mismatch_rate,
        ),
    ]

    violations = []
    for limit, label, value in checks:
        if limit is None or not np.isfinite(float(limit)):
            continue
        if np.isfinite(value) and float(value) > float(limit):
            violations.append(
                f"{label}={float(value):.6f} exceeds limit={float(limit):.6f}"
            )

    return violations


def _truncate_console_label(value, *, max_len=AUDIT_CONSOLE_LABEL_MAX_LEN):
    text = str(value or "").strip()
    if not text:
        return "n/a"
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def _format_console_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        text = value.strip()
        return text or "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric_value):
        return "n/a"
    abs_value = abs(numeric_value)
    if abs_value >= 1000.0 or (0.0 < abs_value < 1e-4):
        return f"{numeric_value:.3e}"
    return f"{numeric_value:.6f}".rstrip("0").rstrip(".")


def _format_console_rate(value):
    if value is None:
        return "n/a"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric_value):
        return "n/a"
    return f"{numeric_value * 100.0:.3f}%"


def _print_console_ranked_rows(
    title,
    frame,
    *,
    label_col,
    metric_specs,
    limit=AUDIT_CONSOLE_TOP_N,
):
    if frame is None or frame.empty:
        return

    print()
    print(f"{title}:")
    display_df = frame.head(int(limit)).reset_index(drop=True)
    for row_idx, row in display_df.iterrows():
        parts = [f"{row_idx + 1}. {_truncate_console_label(row.get(label_col))}"]
        for column_name, display_label, formatter in metric_specs:
            if column_name not in display_df.columns:
                continue
            parts.append(f"{display_label}={formatter(row.get(column_name))}")
        print(f"  {' | '.join(parts)}")


def _print_audit_console_summary(results, written_paths):
    summary = results["live_vs_stored_report"]["summary"]
    reason_summary = results["drift_reason_report"]["summary"]
    violations = _evaluate_guardrail_violations(summary)

    print("Audit summary:")
    print(
        "  window: "
        f"{summary.get('audit_start', 'n/a')} -> {summary.get('audit_end', 'n/a')} | "
        f"bootstrap_rows={_format_console_value(summary.get('bootstrap_rows'))} | "
        "audit_rows_total_1m="
        f"{_format_console_value(summary.get('audit_rows_total_1m'))} | "
        f"decision_rows={_format_console_value(summary.get('decision_row_count'))}"
    )
    print(
        "  prediction drift: "
        f"mean_abs_proba={_format_console_value(summary.get('mean_proba_up_abs_diff'))} | "
        f"max_abs_proba={_format_console_value(summary.get('max_proba_up_abs_diff'))} | "
        "rows_gt_tol="
        f"{_format_console_value(summary.get('rows_with_proba_diff_gt_tol'))}"
    )
    print(
        "  business drift: "
        f"signal={_format_console_value(summary.get('rows_with_signal_mismatch'))} "
        f"({_format_console_rate(summary.get('signal_mismatch_rate'))}) | "
        "decision="
        f"{_format_console_value(summary.get('rows_with_business_decision_mismatch'))} "
        f"({_format_console_rate(summary.get('business_decision_mismatch_rate'))}) | "
        "policy_any="
        f"{_format_console_value(summary.get('rows_with_any_policy_mismatch'))} "
        f"({_format_console_rate(summary.get('any_policy_mismatch_rate'))})"
    )
    ignored_basis_count = int(summary.get("ignored_basis_premium_feature_count", 0) or 0)
    if ignored_basis_count:
        print(
            "  ignored features: "
            f"basis_premium={_format_console_value(ignored_basis_count)}"
        )
    print(
        "  dominant source: "
        f"basis={_format_console_value(reason_summary.get('explanation_basis'))} | "
        f"group={_format_console_value(reason_summary.get('dominant_prediction_impact_group'))} | "
        f"builder={_format_console_value(reason_summary.get('dominant_prediction_impact_builder_name'))} | "
        "feature="
        f"{_truncate_console_label(reason_summary.get('dominant_prediction_impact_feature'))}"
    )
    if violations:
        print("  guardrails: VIOLATED")
        for violation in violations:
            print(f"    - {violation}")
    else:
        print("  guardrails: OK")

    _print_console_ranked_rows(
        "Top prediction-impact features",
        _build_feature_prediction_impact_export_df(
            results["live_vs_stored_report"]["feature_summary_df"],
            top_k=min(int(AUDIT_CONSOLE_TOP_N), int(AUDIT_TOP_N)),
            only_impactful=True,
        ),
        label_col="feature",
        metric_specs=[
            ("builder", "builder", _format_console_value),
            ("proba_drift_rows_resolved", "drift_rows", _format_console_value),
            ("rows_helped", "rows_helped", _format_console_value),
            ("net_pred_gap_reduction", "net_gap_reduction", _format_console_value),
            ("max_pred_shift", "max_pred_shift", _format_console_value),
        ],
    )
    _print_console_ranked_rows(
        "Top feature drop candidates",
        _build_feature_drop_candidates_df(
            results["live_vs_stored_report"]["feature_summary_df"],
            top_k=min(int(AUDIT_CONSOLE_TOP_N), int(AUDIT_TOP_N)),
        ),
        label_col="feature",
        metric_specs=[
            ("builder", "builder", _format_console_value),
            ("drop_candidate_reason", "reason", _truncate_console_label),
            ("max_abs_diff", "max_abs_diff", _format_console_value),
            ("mean_abs_diff", "mean_abs_diff", _format_console_value),
            ("proba_drift_rows_resolved", "drift_rows", _format_console_value),
        ],
    )

    print()
    print("Artifacts:")
    for label, path in written_paths.items():
        print(f"  {label}: {path}")

    return violations


def _is_live_decision_opened(opened, bucket_minutes):
    opened = pd.Timestamp(opened)
    bucket_start = opened.floor(f"{int(bucket_minutes)}min")
    bucket_end = bucket_start + pd.Timedelta(minutes=int(bucket_minutes) - 1)
    return opened == bucket_end


def _safe_colwise_mean(values):
    counts = np.isfinite(values).sum(axis=0)
    sums = np.nansum(values, axis=0)
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def _safe_colwise_rmse(values):
    counts = np.isfinite(values).sum(axis=0)
    sums = np.nansum(np.square(values, dtype=np.float64), axis=0)
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = np.sqrt(sums[valid] / counts[valid])
    return out


def _safe_nanmax_axis1(values):
    valid = np.isfinite(values)
    safe = np.where(valid, values, -np.inf)
    out = safe.max(axis=1)
    out[~valid.any(axis=1)] = np.nan
    return out.astype(np.float64, copy=False)


def _safe_nanmax_axis0(values):
    valid = np.isfinite(values)
    safe = np.where(valid, values, -np.inf)
    out = safe.max(axis=0)
    out[~valid.any(axis=0)] = np.nan
    return out.astype(np.float64, copy=False)


def _safe_relative_diff(
    candidate_values,
    reference_values,
    *,
    denom_floor=REL_DIFF_DENOM_FLOOR,
):
    candidate = np.asarray(candidate_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    out = np.abs(candidate - reference)
    out /= np.maximum(np.abs(reference), float(denom_floor))
    out[~(np.isfinite(candidate) & np.isfinite(reference))] = np.nan
    return out


def _safe_argmax_axis1(values):
    safe = np.where(np.isfinite(values), values, -np.inf)
    return safe.argmax(axis=1)


def _safe_argmax_axis0(values):
    safe = np.where(np.isfinite(values), values, -np.inf)
    return safe.argmax(axis=0)


def _column_values_match(candidate_values, reference_values):
    candidate = np.asarray(candidate_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    same_mask = (candidate == reference) | (np.isnan(candidate) & np.isnan(reference))
    return bool(same_mask.all())


def _fit_indicator_config_map(feature_columns):
    fit_results_dir = Path(MODELING_DATASET_SETTINGS["fit_results_dir"])
    configs = parse_fit_results(fit_results_dir)
    selected = set(feature_columns)
    return {
        cfg["feature_col"]: cfg for cfg in configs if cfg["feature_col"] in selected
    }


def _feature_builder_frame(feature_columns):
    feature_parts = split_feature_subset(
        feature_columns,
        source_label="audit feature columns",
    )
    indicator_config_map = _fit_indicator_config_map(feature_columns)
    candle_pattern_cols = set(
        resolve_candle_pattern_feature_cols(feature_parts["candle_feature_cols"])
    )

    records = []
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
                    "builder_name": "add_session_open_features",
                    "builder_source": None,
                }
            )
            continue

        if feature in feature_parts["realized_volatility_feature_cols"]:
            records.append(
                {
                    "feature": feature,
                    "builder_family": "realized_volatility",
                    "builder_name": "add_realized_volatility_features",
                    "builder_source": "realized_volatility",
                }
            )
            continue

        if feature in feature_parts["basis_premium_feature_cols"]:
            records.append(
                {
                    "feature": feature,
                    "builder_family": "basis_premium",
                    "builder_name": "add_basis_premium_features",
                    "builder_source": "basis_premium_features",
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
    candidate_label,
    reference_label,
    candidate_matrix,
    reference_matrix,
    candidate_pred,
    reference_pred,
    audit_df,
    feature_columns,
    feature_group_by_name,
    feature_builder_frame,
    model,
):
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
    builder_families = (
        builder_meta["builder_family"].fillna("unknown").to_numpy(dtype=object)
    )
    builder_names = (
        builder_meta["builder_name"].fillna("unknown").to_numpy(dtype=object)
    )
    builder_sources = builder_meta["builder_source"].to_numpy(dtype=object)

    base_abs_gap = np.abs(candidate_pred - reference_pred)
    base_signal_mismatch = (candidate_pred >= 0.5) != (reference_pred >= 0.5)
    base_drift_mask = base_abs_gap > PREDICTION_DIFF_TOL

    fixed_pred_matrix = np.empty((row_count, feature_count), dtype=np.float64)
    proba_shift_matrix = np.empty((row_count, feature_count), dtype=np.float64)
    gap_reduction_matrix = np.empty((row_count, feature_count), dtype=np.float64)
    working_matrix = candidate_matrix.copy()

    feature_rows = []
    for feature_idx, feature_name in enumerate(feature_columns):
        candidate_col = candidate_matrix[:, feature_idx]
        reference_col = reference_matrix[:, feature_idx]
        if _column_values_match(candidate_col, reference_col):
            fixed_pred = candidate_pred
        else:
            working_matrix[:, feature_idx] = reference_col
            fixed_pred = np.asarray(
                model.predict(working_matrix), dtype=np.float64
            ).reshape(-1)
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
        helpful_gap_mask = gap_reduction > PREDICTION_DIFF_TOL
        harmful_gap_mask = gap_reduction < -PREDICTION_DIFF_TOL
        drift_gap_values = gap_reduction[base_drift_mask]
        helpful_gap_values = gap_reduction[helpful_gap_mask]
        if base_drift_mask.any():
            drift_base_gap_values = base_abs_gap[base_drift_mask]
            drift_gap_reduction_ratio = np.divide(
                drift_gap_values,
                drift_base_gap_values,
                out=np.zeros(drift_gap_values.shape[0], dtype=np.float64),
                where=drift_base_gap_values > PREDICTION_DIFF_TOL,
            )
            mean_abs_proba_gap_reduction_on_drift_rows_if_fixed = float(
                np.mean(drift_gap_values)
            )
            max_abs_proba_gap_reduction_on_drift_rows_if_fixed = float(
                np.max(drift_gap_values)
            )
            mean_gap_reduction_ratio_on_drift_rows_if_fixed = float(
                np.mean(drift_gap_reduction_ratio)
            )
            max_gap_reduction_ratio_on_drift_rows_if_fixed = float(
                np.max(drift_gap_reduction_ratio)
            )
        else:
            mean_abs_proba_gap_reduction_on_drift_rows_if_fixed = 0.0
            max_abs_proba_gap_reduction_on_drift_rows_if_fixed = 0.0
            mean_gap_reduction_ratio_on_drift_rows_if_fixed = 0.0
            max_gap_reduction_ratio_on_drift_rows_if_fixed = 0.0

        mean_abs_proba_gap_reduction_on_helped_rows_if_fixed = (
            float(np.mean(helpful_gap_values)) if helpful_gap_mask.any() else 0.0
        )
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
                "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed": (
                    mean_abs_proba_gap_reduction_on_drift_rows_if_fixed
                ),
                "max_abs_proba_gap_reduction_on_drift_rows_if_fixed": (
                    max_abs_proba_gap_reduction_on_drift_rows_if_fixed
                ),
                "mean_abs_proba_gap_reduction_on_helped_rows_if_fixed": (
                    mean_abs_proba_gap_reduction_on_helped_rows_if_fixed
                ),
                "mean_gap_reduction_ratio_on_drift_rows_if_fixed": (
                    mean_gap_reduction_ratio_on_drift_rows_if_fixed
                ),
                "max_gap_reduction_ratio_on_drift_rows_if_fixed": (
                    max_gap_reduction_ratio_on_drift_rows_if_fixed
                ),
                "mean_abs_proba_gap_reduction_if_fixed": float(np.mean(gap_reduction)),
                "max_abs_proba_gap_reduction_if_fixed": float(np.max(gap_reduction)),
                "rows_abs_proba_gap_reduced_if_fixed": int(helpful_gap_mask.sum()),
                "rows_abs_proba_gap_worsened_if_fixed": int(harmful_gap_mask.sum()),
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
                "worst_gap_reduction_opened": audit_df["Opened"].iloc[
                    best_gap_reduction_idx
                ],
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
                "rows_proba_diff_gt_tol_resolved_if_fixed",
                "net_abs_proba_gap_reduction_if_fixed",
                "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
                "mean_gap_reduction_ratio_on_drift_rows_if_fixed",
                "rows_abs_proba_gap_reduced_if_fixed",
                "rows_signal_mismatch_introduced_if_fixed",
                "mean_abs_proba_shift_if_fixed",
                "max_abs_proba_shift_if_fixed",
            ],
            ascending=[False, False, False, False, False, False, True, False, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    group_summary_df = (
        feature_summary_df.groupby("group", dropna=False)
        .agg(
            feature_count=("feature", "count"),
            max_mean_abs_proba_shift_if_fixed=("mean_abs_proba_shift_if_fixed", "max"),
            mean_mean_abs_proba_shift_if_fixed=(
                "mean_abs_proba_shift_if_fixed",
                "mean",
            ),
            total_net_abs_proba_gap_reduction_if_fixed=(
                "net_abs_proba_gap_reduction_if_fixed",
                "sum",
            ),
            max_mean_abs_proba_gap_reduction_if_fixed=(
                "mean_abs_proba_gap_reduction_if_fixed",
                "max",
            ),
            mean_mean_abs_proba_gap_reduction_if_fixed=(
                "mean_abs_proba_gap_reduction_if_fixed",
                "mean",
            ),
            max_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed=(
                "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
                "max",
            ),
            mean_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed=(
                "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
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
            total_rows_proba_diff_gt_tol_resolved_if_fixed=(
                "rows_proba_diff_gt_tol_resolved_if_fixed",
                "sum",
            ),
            total_rows_signal_mismatch_introduced_if_fixed=(
                "rows_signal_mismatch_introduced_if_fixed",
                "sum",
            ),
        )
        .sort_values(
            [
                "total_rows_signal_mismatch_resolved_if_fixed",
                "total_rows_proba_diff_gt_tol_resolved_if_fixed",
                "total_net_abs_proba_gap_reduction_if_fixed",
                "max_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
                "max_mean_abs_proba_shift_if_fixed",
            ],
            ascending=[False, False, False, False, False],
            kind="stable",
        )
        .reset_index()
    )

    builder_summary_df = (
        feature_summary_df.groupby(["builder_family", "builder_name"], dropna=False)
        .agg(
            feature_count=("feature", "count"),
            max_mean_abs_proba_shift_if_fixed=("mean_abs_proba_shift_if_fixed", "max"),
            mean_mean_abs_proba_shift_if_fixed=(
                "mean_abs_proba_shift_if_fixed",
                "mean",
            ),
            total_net_abs_proba_gap_reduction_if_fixed=(
                "net_abs_proba_gap_reduction_if_fixed",
                "sum",
            ),
            max_mean_abs_proba_gap_reduction_if_fixed=(
                "mean_abs_proba_gap_reduction_if_fixed",
                "max",
            ),
            mean_mean_abs_proba_gap_reduction_if_fixed=(
                "mean_abs_proba_gap_reduction_if_fixed",
                "mean",
            ),
            max_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed=(
                "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
                "max",
            ),
            mean_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed=(
                "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
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
            total_rows_proba_diff_gt_tol_resolved_if_fixed=(
                "rows_proba_diff_gt_tol_resolved_if_fixed",
                "sum",
            ),
            total_rows_signal_mismatch_introduced_if_fixed=(
                "rows_signal_mismatch_introduced_if_fixed",
                "sum",
            ),
        )
        .sort_values(
            [
                "total_rows_signal_mismatch_resolved_if_fixed",
                "total_rows_proba_diff_gt_tol_resolved_if_fixed",
                "total_net_abs_proba_gap_reduction_if_fixed",
                "max_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
                "max_mean_abs_proba_shift_if_fixed",
            ],
            ascending=[False, False, False, False, False],
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
            "rows_with_helpful_single_feature_fix": int(
                row_has_helpful_single_feature_fix.sum()
            ),
            "top_prediction_impact_feature": (
                None if top_feature is None else top_feature["feature"]
            ),
            "top_prediction_impact_group": (
                None if top_feature is None else top_feature["group"]
            ),
            "top_prediction_impact_builder_family": (
                None if top_feature is None else top_feature["builder_family"]
            ),
            "top_prediction_impact_builder_name": (
                None if top_feature is None else top_feature["builder_name"]
            ),
            "max_mean_abs_proba_shift_if_fixed": (
                0.0
                if feature_summary_df.empty
                else float(feature_summary_df["mean_abs_proba_shift_if_fixed"].max())
            ),
            "max_mean_abs_proba_gap_reduction_if_fixed": (
                0.0
                if feature_summary_df.empty
                else float(
                    feature_summary_df["mean_abs_proba_gap_reduction_if_fixed"].max()
                )
            ),
            "max_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed": (
                0.0
                if feature_summary_df.empty
                else float(
                    feature_summary_df[
                        "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed"
                    ].max()
                )
            ),
            "max_net_abs_proba_gap_reduction_if_fixed": (
                0.0
                if feature_summary_df.empty
                else float(
                    feature_summary_df["net_abs_proba_gap_reduction_if_fixed"].max()
                )
            ),
            "max_rows_signal_mismatch_resolved_if_fixed": (
                0
                if feature_summary_df.empty
                else int(
                    feature_summary_df["rows_signal_mismatch_resolved_if_fixed"].max()
                )
            ),
            "max_rows_proba_diff_gt_tol_resolved_if_fixed": (
                0
                if feature_summary_df.empty
                else int(
                    feature_summary_df[
                        "rows_proba_diff_gt_tol_resolved_if_fixed"
                    ].max()
                )
            ),
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
    parquet_path,
    audit_end,
    tail_fraction,
):
    tail_fraction = float(tail_fraction)
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError("tail_fraction must be in (0, 1].")

    opened = _load_opened_series(parquet_path)
    opened = pd.to_datetime(opened)
    audit_end = pd.Timestamp(audit_end)
    eligible = opened.loc[opened <= audit_end]
    if eligible.empty:
        raise ValueError("No parquet rows are available at or before audit_end.")

    total_rows = len(eligible)
    keep_rows = int(np.ceil(total_rows * tail_fraction))
    keep_rows = max(1, min(total_rows, keep_rows))
    tail_start = pd.Timestamp(eligible.iloc[total_rows - keep_rows])
    return tail_start, keep_rows, total_rows


def load_modeling_raw_history_frame(
    *,
    parquet_path,
    audit_end,
    history_start=None,
):
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
    frame = (
        frame.sort_values("Opened")
        .drop_duplicates(subset=["Opened"])
        .reset_index(drop=True)
    )
    return frame


def build_current_recomputed_feature_history(
    *,
    raw_history_df,
    feature_columns,
):
    feature_parts = split_feature_subset(
        feature_columns,
        source_label="audit feature columns",
    )
    recomputed_feature_columns = _audited_feature_columns(feature_columns)
    feature_frame = raw_history_df.loc[:, ["Opened", *RAW_OHLCV_COLS]].copy()

    configured_rules = resolve_streak_interval_to_rule(
        MODELING_DATASET_SETTINGS.get("candle_streak_intervals", {})
    )
    if feature_parts["streak_intervals"]:
        missing_streak_intervals = [
            label
            for label in feature_parts["streak_intervals"]
            if label not in configured_rules
        ]
        if missing_streak_intervals:
            raise ValueError(
                "Missing configured candle streak intervals for recompute: "
                f"{missing_streak_intervals}"
            )
        feature_frame = add_candle_streak_features(
            feature_frame,
            interval_to_rule={
                label: configured_rules[label]
                for label in feature_parts["streak_intervals"]
            },
        )

    if feature_parts["candle_feature_cols"]:
        feature_frame = add_candle_derived_features(
            feature_frame,
            feature_cols=feature_parts["candle_feature_cols"],
        )

    if feature_parts["session_feature_cols"]:
        feature_frame = add_session_open_features(
            feature_frame,
            feature_cols=feature_parts["session_feature_cols"],
        )

    if feature_parts["realized_volatility_feature_cols"]:
        feature_frame = add_realized_volatility_features(feature_frame)

    vp_cfg = normalize_volume_profile_config(
        MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")
    )
    if feature_parts["volume_profile_feature_cols"]:
        if not vp_cfg["enabled"]:
            raise ValueError(
                "Volume profile features requested but disabled in modeling config."
            )
        vp_features_df, _vp_state = build_volume_profile_features(feature_frame, vp_cfg)
        vp_feature_frame = pd.DataFrame(
            {
                feature_col: vp_features_df[feature_col].to_numpy(
                    dtype=np.float64, copy=False
                )
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
        indicator_config_map = _fit_indicator_config_map(recomputed_feature_columns)
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
    keep_cols.extend(col for col in recomputed_feature_columns if col not in keep_cols)
    return feature_frame.loc[:, keep_cols].copy()


def align_feature_frame_to_audit_rows(
    *,
    audit_df,
    feature_frame,
    feature_columns,
):
    aligned = audit_df.loc[:, ["Opened"]].merge(
        feature_frame.loc[:, ["Opened", *feature_columns]],
        on="Opened",
        how="left",
        sort=False,
    )
    if feature_columns and aligned[feature_columns].isna().all(axis=1).any():
        missing_rows = aligned.loc[
            aligned[feature_columns].isna().all(axis=1), "Opened"
        ]
        raise RuntimeError(
            "Current recompute frame is missing rows for audit timestamps. "
            f"First missing={missing_rows.iloc[0]!s}"
        )
    return aligned


def _compare_policy_decisions(
    *,
    predictor,
    candidate_proba,
    reference_proba,
    candidate_label,
    reference_label,
):
    bankroll = float(LIVE_INITIAL_BANKROLL_USDC)
    rows = []
    for candidate_prob, reference_prob in zip(
        candidate_proba, reference_proba, strict=True
    ):
        candidate_decision = predictor.evaluate_policy_decision(
            float(candidate_prob),
            bankroll=bankroll,
        )
        reference_decision = predictor.evaluate_policy_decision(
            float(reference_prob),
            bankroll=bankroll,
        )

        candidate_stake = float(candidate_decision.get("bet_usdc", 0.0) or 0.0)
        reference_stake = float(reference_decision.get("bet_usdc", 0.0) or 0.0)
        candidate_trade = int(candidate_stake > 0.0)
        reference_trade = int(reference_stake > 0.0)
        candidate_side = str(candidate_decision.get("trade_side", "none"))
        reference_side = str(reference_decision.get("trade_side", "none"))
        candidate_reason = str(
            candidate_decision.get(
                "final_reason",
                candidate_decision.get("reason", ""),
            )
        )
        reference_reason = str(
            reference_decision.get(
                "final_reason",
                reference_decision.get("reason", ""),
            )
        )
        rows.append(
            {
                f"{candidate_label}_policy_side": candidate_side,
                f"{reference_label}_policy_side": reference_side,
                f"{candidate_label}_policy_reason": candidate_reason,
                f"{reference_label}_policy_reason": reference_reason,
                f"{candidate_label}_stake_usdc": candidate_stake,
                f"{reference_label}_stake_usdc": reference_stake,
                f"{candidate_label}_trade_flag": candidate_trade,
                f"{reference_label}_trade_flag": reference_trade,
                "policy_side_mismatch": int(candidate_side != reference_side),
                "policy_reason_mismatch": int(candidate_reason != reference_reason),
                "policy_trade_flag_mismatch": int(candidate_trade != reference_trade),
                "policy_stake_abs_diff": abs(candidate_stake - reference_stake),
                "policy_decision_mismatch": int(
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
    candidate_label,
    reference_label,
    candidate_matrix,
    reference_matrix,
    audit_df,
    feature_columns,
    feature_group_by_name,
    feature_builder_frame,
    model,
    feature_importance_df=None,
    policy_predictor=None,
):
    feature_names = np.asarray(feature_columns, dtype=object)
    builder_meta = (
        feature_builder_frame.set_index("feature")
        .reindex(feature_columns)
        .reset_index(drop=False)
    )
    builder_groups = (
        builder_meta["builder_family"].fillna("unknown").to_numpy(dtype=object)
    )
    builder_names = (
        builder_meta["builder_name"].fillna("unknown").to_numpy(dtype=object)
    )
    feature_groups = np.asarray(
        [feature_group_by_name.get(col, "unknown") for col in feature_columns],
        dtype=object,
    )
    candidate_nonfinite_mask = ~np.isfinite(candidate_matrix)
    reference_nonfinite_mask = ~np.isfinite(reference_matrix)
    finite_pair_mask = np.isfinite(candidate_matrix) & np.isfinite(reference_matrix)
    finite_status_mismatch_mask = np.logical_xor(
        candidate_nonfinite_mask, reference_nonfinite_mask
    )
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
    row_worst_builder_family = np.where(
        row_has_finite, builder_groups[row_worst_idx], None
    )
    row_worst_builder_name = np.where(
        row_has_finite, builder_names[row_worst_idx], None
    )
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
            f"{candidate_label}_nonfinite_count": candidate_nonfinite_mask.sum(
                axis=1
            ).astype(np.int32),
            f"{reference_label}_nonfinite_count": reference_nonfinite_mask.sum(
                axis=1
            ).astype(np.int32),
            "finite_status_mismatch_count": finite_status_mismatch_mask.sum(
                axis=1
            ).astype(np.int32),
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

    if policy_predictor is not None:
        policy_df = _compare_policy_decisions(
            predictor=policy_predictor,
            candidate_proba=candidate_pred,
            reference_proba=reference_pred,
            candidate_label=candidate_label,
            reference_label=reference_label,
        )
        step_summary_df = pd.concat(
            [step_summary_df.reset_index(drop=True), policy_df.reset_index(drop=True)],
            axis=1,
            copy=False,
        )
        step_summary_df["business_decision_mismatch"] = (
            (step_summary_df["signal_mismatch"] > 0)
            | (step_summary_df["policy_side_mismatch"] > 0)
            | (step_summary_df["policy_reason_mismatch"] > 0)
            | (step_summary_df["policy_trade_flag_mismatch"] > 0)
        ).astype(np.int8)
        step_summary_df["stake_only_policy_mismatch"] = (
            (step_summary_df["policy_decision_mismatch"] > 0)
            & (step_summary_df["business_decision_mismatch"] == 0)
        ).astype(np.int8)
    else:
        step_summary_df["business_decision_mismatch"] = step_summary_df[
            "signal_mismatch"
        ].astype(np.int8)
        step_summary_df["stake_only_policy_mismatch"] = np.zeros(
            len(step_summary_df), dtype=np.int8
        )

    col_has_finite = np.isfinite(diff_matrix).any(axis=0)
    col_worst_idx = _safe_argmax_axis0(diff_matrix)
    feature_summary_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "group": [
                feature_group_by_name.get(col, "unknown") for col in feature_columns
            ],
            "max_abs_diff": _safe_nanmax_axis0(diff_matrix),
            "mean_abs_diff": _safe_colwise_mean(diff_matrix),
            "max_rel_diff": _safe_nanmax_axis0(rel_diff_matrix),
            "mean_rel_diff": _safe_colwise_mean(rel_diff_matrix),
            "rmse_abs_diff": _safe_colwise_rmse(diff_matrix),
            "finite_diff_count": np.isfinite(diff_matrix).sum(axis=0).astype(np.int32),
            f"{candidate_label}_nonfinite_count": candidate_nonfinite_mask.sum(
                axis=0
            ).astype(np.int32),
            f"{reference_label}_nonfinite_count": reference_nonfinite_mask.sum(
                axis=0
            ).astype(np.int32),
            "finite_status_mismatch_count": finite_status_mismatch_mask.sum(
                axis=0
            ).astype(np.int32),
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
    feature_summary_df = (
        feature_summary_df.merge(
            feature_builder_frame,
            on="feature",
            how="left",
            sort=False,
        )
        .merge(
            feature_impact_summary_df,
            on="feature",
            how="left",
            sort=False,
        )
        .sort_values(
            [
                "rows_signal_mismatch_resolved_if_fixed",
                "rows_proba_diff_gt_tol_resolved_if_fixed",
                "net_abs_proba_gap_reduction_if_fixed",
                "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
                "mean_gap_reduction_ratio_on_drift_rows_if_fixed",
                "mean_abs_proba_shift_if_fixed",
                "finite_status_mismatch_count",
                "max_rel_diff",
                "mean_rel_diff",
                "max_abs_diff",
                "mean_abs_diff",
                "rmse_abs_diff",
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
                False,
                False,
            ],
            kind="stable",
        )
    )
    if feature_importance_df is not None and not feature_importance_df.empty:
        feature_summary_df = feature_summary_df.merge(
            feature_importance_df,
            on="feature",
            how="left",
            sort=False,
        )
    if "importance_gain" not in feature_summary_df.columns:
        feature_summary_df["importance_gain"] = 0.0
    if "importance_split" not in feature_summary_df.columns:
        feature_summary_df["importance_split"] = 0.0
    feature_summary_df["importance_gain"] = pd.to_numeric(
        feature_summary_df["importance_gain"], errors="coerce"
    ).fillna(0.0)
    feature_summary_df["importance_split"] = pd.to_numeric(
        feature_summary_df["importance_split"], errors="coerce"
    ).fillna(0.0)

    group_summary_df = (
        feature_summary_df.groupby("group", dropna=False)
        .agg(
            feature_count=("feature", "count"),
            max_abs_diff=("max_abs_diff", "max"),
            mean_abs_diff=("mean_abs_diff", "mean"),
            max_rel_diff=("max_rel_diff", "max"),
            mean_rel_diff=("mean_rel_diff", "mean"),
            rmse_abs_diff=("rmse_abs_diff", "mean"),
            total_candidate_nonfinite_count=(
                f"{candidate_label}_nonfinite_count",
                "sum",
            ),
            total_reference_nonfinite_count=(
                f"{reference_label}_nonfinite_count",
                "sum",
            ),
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
            "total_rows_signal_mismatch_resolved_if_fixed",
            "total_rows_proba_diff_gt_tol_resolved_if_fixed",
            "total_net_abs_proba_gap_reduction_if_fixed",
            "max_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
            "max_mean_abs_proba_shift_if_fixed",
            "max_rel_diff",
            "mean_rel_diff",
            "max_abs_diff",
            "mean_abs_diff",
        ],
        ascending=[False, False, False, False, False, False, False, False, False],
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
            "total_rows_signal_mismatch_resolved_if_fixed",
            "total_rows_proba_diff_gt_tol_resolved_if_fixed",
            "total_net_abs_proba_gap_reduction_if_fixed",
            "max_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
            "max_mean_abs_proba_shift_if_fixed",
            "max_rel_diff",
            "mean_rel_diff",
            "max_abs_diff",
            "mean_abs_diff",
        ],
        ascending=[False, False, False, False, False, False, False, False, False],
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
        "signal_mismatch_rate": _summary_rate(
            int(step_summary_df["signal_mismatch"].sum()),
            len(audit_df),
        ),
        "rows_with_business_decision_mismatch": int(
            step_summary_df["business_decision_mismatch"].sum()
        ),
        "business_decision_mismatch_rate": _summary_rate(
            int(step_summary_df["business_decision_mismatch"].sum()),
            len(audit_df),
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
    if "policy_decision_mismatch" in step_summary_df.columns:
        summary_payload.update(
            {
                "policy_audit_bankroll_usdc": float(LIVE_INITIAL_BANKROLL_USDC),
                "rows_with_policy_side_mismatch": int(
                    step_summary_df["policy_side_mismatch"].sum()
                ),
                "policy_side_mismatch_rate": _summary_rate(
                    int(step_summary_df["policy_side_mismatch"].sum()),
                    len(audit_df),
                ),
                "rows_with_policy_reason_mismatch": int(
                    step_summary_df["policy_reason_mismatch"].sum()
                ),
                "policy_reason_mismatch_rate": _summary_rate(
                    int(step_summary_df["policy_reason_mismatch"].sum()),
                    len(audit_df),
                ),
                "rows_with_policy_trade_flag_mismatch": int(
                    step_summary_df["policy_trade_flag_mismatch"].sum()
                ),
                "policy_trade_flag_mismatch_rate": _summary_rate(
                    int(step_summary_df["policy_trade_flag_mismatch"].sum()),
                    len(audit_df),
                ),
                "rows_with_stake_only_policy_mismatch": int(
                    step_summary_df["stake_only_policy_mismatch"].sum()
                ),
                "stake_only_policy_mismatch_rate": _summary_rate(
                    int(step_summary_df["stake_only_policy_mismatch"].sum()),
                    len(audit_df),
                ),
                "rows_with_any_policy_mismatch": int(
                    step_summary_df["policy_decision_mismatch"].sum()
                ),
                "any_policy_mismatch_rate": _summary_rate(
                    int(step_summary_df["policy_decision_mismatch"].sum()),
                    len(audit_df),
                ),
                "max_policy_stake_abs_diff": float(
                    step_summary_df["policy_stake_abs_diff"].max()
                ),
                "mean_policy_stake_abs_diff": float(
                    step_summary_df["policy_stake_abs_diff"].mean()
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
        "prediction_impact_feature_summary_df": prediction_impact_report[
            "feature_summary_df"
        ],
        "prediction_impact_group_summary_df": prediction_impact_report[
            "group_summary_df"
        ],
        "prediction_impact_builder_summary_df": prediction_impact_report[
            "builder_summary_df"
        ],
        "single_feature_fixed_pred_matrix": prediction_impact_report[
            "fixed_pred_matrix"
        ],
        "single_feature_proba_shift_matrix": prediction_impact_report[
            "proba_shift_matrix"
        ],
        "single_feature_gap_reduction_matrix": prediction_impact_report[
            "gap_reduction_matrix"
        ],
        "diff_matrix": diff_matrix,
        "rel_diff_matrix": rel_diff_matrix,
        "candidate_nonfinite_mask": candidate_nonfinite_mask,
        "reference_nonfinite_mask": reference_nonfinite_mask,
        "finite_status_mismatch_mask": finite_status_mismatch_mask,
        "candidate_label": candidate_label,
        "reference_label": reference_label,
    }


def build_live_drift_reason_report(
    report,
    *,
    top_n=20,
):
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
    policy_mask = (
        step_summary_df["policy_decision_mismatch"] > 0
        if "policy_decision_mismatch" in step_summary_df.columns
        else pd.Series(False, index=step_summary_df.index)
    )
    stake_only_mask = (
        step_summary_df["stake_only_policy_mismatch"] > 0
        if "stake_only_policy_mismatch" in step_summary_df.columns
        else pd.Series(False, index=step_summary_df.index)
    )
    explain_mask = business_mask
    explanation_basis = "business_decision_mismatch"
    if not bool(explain_mask.any()) and bool(stake_only_mask.any()):
        explain_mask = stake_only_mask
        explanation_basis = "stake_only_policy_mismatch"
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
                "rows_with_stake_only_policy_mismatch": int(stake_only_mask.sum()),
                "rows_with_any_policy_mismatch": int(policy_mask.sum()),
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
        explain_candidate_pred = step_summary_df.loc[
            explain_mask, candidate_proba_col
        ].to_numpy(
            dtype=np.float64,
            copy=False,
        )
        explain_reference_pred = step_summary_df.loc[
            explain_mask, reference_proba_col
        ].to_numpy(
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
                    )
                    .sum(axis=0)
                    .astype(np.int32),
                    "rows_signal_mismatch_resolved_if_fixed_on_explained_rows": (
                        explain_signal_mismatch & ~explain_fixed_signal_mismatch
                    )
                    .sum(axis=0)
                    .astype(np.int32),
                    "max_rel_diff_on_explained_rows": _safe_nanmax_axis0(
                        explain_rel_diff_matrix
                    ),
                    "mean_rel_diff_on_explained_rows": _safe_colwise_mean(
                        explain_rel_diff_matrix
                    ),
                    "max_abs_diff_on_explained_rows": _safe_nanmax_axis0(
                        explain_diff_matrix
                    ),
                    "mean_abs_diff_on_explained_rows": _safe_colwise_mean(
                        explain_diff_matrix
                    ),
                    "rmse_abs_diff_on_explained_rows": _safe_colwise_rmse(
                        explain_diff_matrix
                    ),
                }
            )
            .merge(
                feature_summary_df[
                    ["feature", "group", "builder_family", "builder_name"]
                ],
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
                feature_summary_df[
                    ["feature", "group", "builder_family", "builder_name"]
                ],
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
            "rows_with_stake_only_policy_mismatch": int(stake_only_mask.sum()),
            "rows_with_any_policy_mismatch": int(policy_mask.sum()),
            "explanation_basis": explanation_basis,
            "rows_with_history_shortfall": int(history_shortfall.gt(0).sum()),
            "prediction_drift_rows_with_history_shortfall": int(
                explain_rows_df.get("history_shortfall", pd.Series(dtype=np.int32))
                .gt(0)
                .sum()
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


class AuditWindow:
    __slots__ = (
        "bootstrap_start",
        "audit_start",
        "audit_end",
        "bootstrap_rows",
        "audit_rows",
        "requested_days_back",
        "max_steps",
    )

    def __init__(
        self,
        bootstrap_start,
        audit_start,
        audit_end,
        bootstrap_rows,
        audit_rows,
        requested_days_back,
        max_steps,
    ):
        self.bootstrap_start = bootstrap_start
        self.audit_start = audit_start
        self.audit_end = audit_end
        self.bootstrap_rows = bootstrap_rows
        self.audit_rows = audit_rows
        self.requested_days_back = requested_days_back
        self.max_steps = max_steps


class PseudoLiveAuditPredictor(LivePredictor):
    def __init__(
        self,
        bootstrap_df,
        *,
        model_meta_path=MODEL_META_PATH,
        max_keep=DEFAULT_MAX_KEEP,
        volume_profile_state=None,
        allow_unstable_indicator_summary=AUDIT_ALLOW_UNSTABLE_INDICATOR_SUMMARY,
    ):
        self.model, meta = load_model_and_meta(model_meta_path)
        self.feature_columns = list(meta.get("feature_columns", []))
        if not self.feature_columns:
            raise ValueError("Missing feature_columns in model metadata.")
        validate_volume_profile_feature_columns(
            self.feature_columns,
            source_label=f"model metadata {model_meta_path}",
        )
        validate_basis_premium_feature_columns(
            self.feature_columns,
            source_label=f"model metadata {model_meta_path}",
        )

        self.candle_feature_columns = [
            col for col in self.feature_columns if col in SUPPORTED_CANDLE_FEATURE_COLS
        ]
        self.candle_derived_feature_columns = tuple(
            resolve_candle_derived_feature_cols(self.candle_feature_columns)
        )
        self.candle_pattern_feature_columns = tuple(
            resolve_candle_pattern_feature_cols(self.candle_feature_columns)
        )
        self.target_col = str(meta.get("target_col", "target_5m_candle_up"))
        self.target_bucket_minutes = parse_target_bucket_minutes(self.target_col)
        self.trade_policy_runtime = load_trade_policy_runtime_config(
            TRADE_POLICY_CONFIG_PATH
        )
        self.live_bankroll_usdc = float(LIVE_INITIAL_BANKROLL_USDC)

        feature_parts = split_feature_subset(
            self.feature_columns,
            source_label=f"model metadata {model_meta_path}",
        )
        if feature_parts["streak_intervals"]:
            self.streak_interval_to_rule = resolve_streak_interval_to_rule(
                feature_parts["streak_intervals"]
            )
        else:
            self.streak_interval_to_rule = {}
        self.session_feature_columns = tuple(feature_parts["session_feature_cols"])
        self.realized_volatility_feature_columns = tuple(
            feature_parts["realized_volatility_feature_cols"]
        )
        self.basis_premium_feature_columns = tuple(
            feature_parts["basis_premium_feature_cols"]
        )
        # The audit copies stored basis-premium features back into the candidate vector.
        self.basis_premium_feature_columns = ()
        self.basis_premium_cfg = {}
        self.basis_premium_interval_to_rule = {}
        self.basis_index_close_col = ""
        self.basis_index_ohlcv_idx = None
        self.basis_futures_close_col = ""
        self.basis_futures_close_np = None
        self.volume_profile_feature_columns = tuple(
            feature_parts["volume_profile_feature_cols"]
        )
        self.volume_profile_cfg = normalize_volume_profile_config(
            MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")
        )
        validate_volume_profile_model_metadata(
            meta,
            feature_columns=self.volume_profile_feature_columns,
            cfg=self.volume_profile_cfg,
            source_label=f"model metadata {model_meta_path}",
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
        self.volume_profile_save_pool = None

        self.indicator_specs = load_indicator_specs(
            self.feature_columns,
            source_label=f"model metadata {model_meta_path}",
        )
        requirements_indicator_specs = self.indicator_specs
        if allow_unstable_indicator_summary:
            requirements_payload = json.loads(
                Path(INDICATOR_HISTORY_REQUIREMENTS_PATH).read_text(encoding="utf-8")
            )
            unstable_feature_cols = {
                str(feature_col).strip()
                for feature_col in requirements_payload.get("unstable_features", [])
                if str(feature_col).strip()
            }
            if not unstable_feature_cols:
                summary_path_raw = requirements_payload.get("analysis_summary_path")
                if summary_path_raw not in (None, ""):
                    summary_path = Path(summary_path_raw)
                    if summary_path.exists():
                        summary_payload = json.loads(
                            summary_path.read_text(encoding="utf-8")
                        )
                        unstable_feature_cols = {
                            str(feature_col).strip()
                            for feature_col in summary_payload.get(
                                "unstable_features", []
                            )
                            if str(feature_col).strip()
                        }
            filtered_specs = [
                spec
                for spec in self.indicator_specs
                if str(spec.feature_col) not in unstable_feature_cols
            ]
            if len(filtered_specs) != len(self.indicator_specs):
                print(
                    "[audit] ignoring unstable features in per-feature history validation "
                    f"count={len(self.indicator_specs) - len(filtered_specs)}"
                )
                requirements_indicator_specs = filtered_specs
        self.indicator_history_requirements = load_indicator_history_requirements(
            INDICATOR_HISTORY_REQUIREMENTS_PATH,
            indicator_specs=requirements_indicator_specs,
            allow_unstable=allow_unstable_indicator_summary,
        )
        self.required_stable_window = int(
            self.indicator_history_requirements["global_required_runtime_window"]
        )
        self.required_stable_window_raw = int(
            self.indicator_history_requirements["global_required_stable_window"]
        )
        self.indicator_stable_window_by_feature = dict(
            self.indicator_history_requirements["stable_window_by_feature"]
        )
        self.indicator_runtime_window_by_feature = dict(
            self.indicator_history_requirements["runtime_window_by_feature"]
        )
        self.bootstrap_candles = len(bootstrap_df)
        self.max_keep = max(int(max_keep), int(self.required_stable_window))

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
        self.realized_volatility_state = None
        self.latest_realized_volatility_values = {}
        self._initialize_realized_volatility_state()

        self.volume_profile_state = None
        if self.volume_profile_enabled:
            if volume_profile_state is not None:
                self.volume_profile_state = copy.deepcopy(volume_profile_state)
            else:
                self.volume_profile_state = bootstrap_state_from_history(
                    bootstrap_df.loc[:, ["Opened", "High", "Low", "Volume"]],
                    self.volume_profile_cfg,
                )

    def _save_runtime_volume_profile_state(self, log=False, context="state"):
        return None

    def evaluate_policy_decision(
        self,
        prob_up_raw,
        *,
        bankroll=None,
    ):
        prev_bankroll = float(self.live_bankroll_usdc)
        try:
            if bankroll is not None:
                self.live_bankroll_usdc = float(bankroll)
            return dict(self._build_policy_intent(proba_up=float(prob_up_raw)))
        finally:
            self.live_bankroll_usdc = prev_bankroll

    def build_feature_snapshot(self, volume_profile_values=None):
        vector = self._build_feature_vector(volume_profile_values=volume_profile_values)
        nonfinite_feature_indices = tuple(
            int(idx) for idx in np.flatnonzero(~np.isfinite(vector[0, :]))
        )
        return {
            "vector": vector,
            "nonfinite_feature_indices": nonfinite_feature_indices,
            "indicator_nan_cols": tuple(self.last_indicator_nan_cols),
        }


def _load_opened_series(parquet_path):
    opened = pd.read_parquet(parquet_path, columns=["Opened"])["Opened"]
    return pd.to_datetime(opened)


def resolve_audit_window(
    *,
    parquet_path,
    days_back=DEFAULT_AUDIT_DAYS_BACK,
    bootstrap_candles=DEFAULT_BOOTSTRAP_CANDLES,
    max_steps=None,
):
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
    parquet_path,
    feature_columns,
    audit_window,
):
    columns = ["Opened", *RAW_OHLCV_COLS, *feature_columns]
    frame = pd.read_parquet(
        parquet_path,
        columns=columns,
        filters=[
            (
                "Opened",
                ">=",
                audit_window.bootstrap_start.to_pydatetime().replace(tzinfo=None),
            ),
            (
                "Opened",
                "<=",
                audit_window.audit_end.to_pydatetime().replace(tzinfo=None),
            ),
        ],
    )
    frame["Opened"] = _ensure_utc_opened(frame["Opened"])
    frame = (
        frame.sort_values("Opened")
        .drop_duplicates(subset=["Opened"])
        .reset_index(drop=True)
    )
    return frame


def load_anchor_volume_profile_history(
    *,
    parquet_path,
    anchor_candle_opened,
):
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
    frame = (
        frame.sort_values("Opened")
        .drop_duplicates(subset=["Opened"])
        .reset_index(drop=True)
    )
    return frame


def build_or_load_anchor_volume_profile_state(
    *,
    parquet_path,
    anchor_candle_opened,
    overwrite=False,
):
    state_path = resolve_anchor_volume_profile_state_path(anchor_candle_opened)
    vp_cfg = normalize_volume_profile_config(
        MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")
    )
    if (
        not overwrite
        and state_path.with_suffix(".npz").exists()
        and state_path.with_suffix(".json").exists()
    ):
        try:
            state = load_volume_profile_state(state_path)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(
                "[audit] existing anchor vp state is unreadable; rebuilding "
                f"path={state_path.with_suffix('.npz')} error={exc}"
            )
        else:
            if volume_profile_state_matches_config(state, vp_cfg):
                return state, state_path
            print(
                "[audit] existing anchor vp state config mismatch; rebuilding "
                f"path={state_path.with_suffix('.npz')}"
            )

    history_df = load_anchor_volume_profile_history(
        parquet_path=parquet_path,
        anchor_candle_opened=anchor_candle_opened,
    )
    state = bootstrap_state_from_history(
        history_df.loc[:, ["Opened", "High", "Low", "Volume"]],
        vp_cfg,
    )
    save_volume_profile_state(state, state_path)
    return load_volume_profile_state(state_path), state_path


def run_stored_modeling_vs_current_recompute_audit(
    *,
    days_back=DEFAULT_AUDIT_DAYS_BACK,
    max_steps=None,
    history_tail_fraction=1.0,
    model_meta_path=MODEL_META_PATH,
    parquet_path=None,
):
    model_meta_path = Path(model_meta_path)
    parquet_path = (
        _optional_path(parquet_path) or resolve_modeling_dataset_parquet_path()
    )

    model, meta = load_model_and_meta(model_meta_path)
    feature_importance_df = _load_model_feature_importance_frame(meta)
    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        raise ValueError("Missing feature_columns in model metadata.")
    validate_volume_profile_feature_columns(
        feature_columns,
        source_label=f"model metadata {model_meta_path}",
    )
    validate_basis_premium_feature_columns(
        feature_columns,
        source_label=f"model metadata {model_meta_path}",
    )
    ignored_basis_feature_columns = _ignored_basis_premium_feature_columns(
        feature_columns
    )
    recomputed_feature_columns = _audited_feature_columns(feature_columns)
    validate_volume_profile_model_metadata(
        meta,
        feature_columns=feature_columns,
        cfg=normalize_volume_profile_config(
            MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")
        ),
        source_label=f"model metadata {model_meta_path}",
    )

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
    history_start, history_rows, history_total_rows = (
        resolve_recent_history_tail_window(
            parquet_path=parquet_path,
            audit_end=audit_window.audit_end,
            tail_fraction=history_tail_fraction,
        )
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
        feature_columns=recomputed_feature_columns,
    )
    recomputed_audit_df = align_feature_frame_to_audit_rows(
        audit_df=stored_audit_df,
        feature_frame=recomputed_history_df,
        feature_columns=recomputed_feature_columns,
    )
    recomputed_audit_df = _copy_ignored_basis_premium_features(
        recomputed_audit_df,
        stored_audit_df,
        feature_columns,
    )

    feature_group_by_name = _feature_group_map(feature_columns)
    feature_builder_frame = _feature_builder_frame(feature_columns)
    policy_predictor = PseudoLiveAuditPredictor(
        raw_history_df.tail(
            max(1, min(len(raw_history_df), DEFAULT_BOOTSTRAP_CANDLES))
        ).copy(),
        model_meta_path=model_meta_path,
        max_keep=max(
            DEFAULT_MAX_KEEP, len(raw_history_df.tail(DEFAULT_BOOTSTRAP_CANDLES))
        ),
        volume_profile_state=None,
    )
    report = build_matrix_comparison_report(
        candidate_label="recomputed",
        reference_label="stored",
        candidate_matrix=recomputed_audit_df[feature_columns].to_numpy(
            dtype=np.float64, copy=True
        ),
        reference_matrix=stored_audit_df[feature_columns].to_numpy(
            dtype=np.float64, copy=True
        ),
        audit_df=stored_audit_df,
        feature_columns=feature_columns,
        feature_group_by_name=feature_group_by_name,
        feature_builder_frame=feature_builder_frame,
        model=model,
        feature_importance_df=feature_importance_df,
        policy_predictor=policy_predictor,
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
                    "ignored_basis_premium_feature_count": len(
                        ignored_basis_feature_columns
                    ),
                    "ignored_basis_premium_features": ", ".join(
                        ignored_basis_feature_columns
                    ),
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
    report["candidate_feature_frame"] = recomputed_audit_df.loc[
        :, feature_columns
    ].copy()
    report["feature_columns"] = feature_columns
    return report


def run_live_modeling_feature_audit(
    *,
    days_back=DEFAULT_AUDIT_DAYS_BACK,
    bootstrap_candles=DEFAULT_BOOTSTRAP_CANDLES,
    max_steps=None,
    max_keep=DEFAULT_MAX_KEEP,
    model_meta_path=MODEL_META_PATH,
    parquet_path=None,
    use_anchor_vp_state=True,
    overwrite_anchor_vp_state=False,
):
    model_meta_path = Path(model_meta_path)
    parquet_path = (
        _optional_path(parquet_path) or resolve_modeling_dataset_parquet_path()
    )
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
        if AUDIT_ALLOW_UNSTABLE_INDICATOR_SUMMARY:
            print(
                "[audit] indicator history requirements artifact contains unstable features; "
                "continuing because audit override is enabled"
            )
        print(
            "[audit] inputs " f"model_meta={model_meta_path} " f"parquet={parquet_path}"
        )

    model, meta = load_model_and_meta(model_meta_path)
    feature_importance_df = _load_model_feature_importance_frame(meta)
    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        raise ValueError("Missing feature_columns in model metadata.")
    validate_volume_profile_feature_columns(
        feature_columns,
        source_label=f"model metadata {model_meta_path}",
    )
    validate_basis_premium_feature_columns(
        feature_columns,
        source_label=f"model metadata {model_meta_path}",
    )
    ignored_basis_feature_columns = _ignored_basis_premium_feature_columns(
        feature_columns
    )
    validate_volume_profile_model_metadata(
        meta,
        feature_columns=feature_columns,
        cfg=normalize_volume_profile_config(
            MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")
        ),
        source_label=f"model metadata {model_meta_path}",
    )

    required_stable_window = load_required_stable_window(
        INDICATOR_HISTORY_REQUIREMENTS_PATH,
        allow_unstable=AUDIT_ALLOW_UNSTABLE_INDICATOR_SUMMARY,
    )
    resolved_bootstrap_candles = max(
        int(bootstrap_candles), int(required_stable_window)
    )
    resolved_max_keep = max(int(max_keep), int(required_stable_window))
    if progress_enabled and (
        resolved_bootstrap_candles != int(bootstrap_candles)
        or resolved_max_keep != int(max_keep)
    ):
        print(
            "[audit] adjusted_history_windows "
            f"required_stable_window={int(required_stable_window)} "
            f"bootstrap_candles={int(bootstrap_candles)}->{resolved_bootstrap_candles} "
            f"max_keep={int(max_keep)}->{resolved_max_keep}"
        )
    bootstrap_candles = resolved_bootstrap_candles
    max_keep = resolved_max_keep

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
    audit_df = (
        modeling_frame.iloc[audit_window.bootstrap_rows :].copy().reset_index(drop=True)
    )
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
        anchor_vp_state, anchor_vp_state_path = (
            build_or_load_anchor_volume_profile_state(
                parquet_path=parquet_path,
                anchor_candle_opened=anchor_candle_opened,
                overwrite=overwrite_anchor_vp_state,
            )
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
            f"initial_buffer_rows={len(predictor.opened_candles)}"
        )

    feature_group_by_name = _feature_group_map(feature_columns)
    feature_builder_frame = _feature_builder_frame(feature_columns)
    indicator_nan_count_rows = []
    history_rows_used_rows = []
    history_shortfall_rows = []
    decision_row_indices = []
    live_vector_rows = []
    live_nonfinite_rows = []

    ignored_basis_feature_indices = [
        feature_columns.index(feature)
        for feature in ignored_basis_feature_columns
        if feature in feature_columns
    ]
    ohlcv_matrix = audit_df[list(RAW_OHLCV_COLS)].to_numpy(dtype=np.float64, copy=False)
    opened_values = audit_df["Opened"].to_list()
    loop_started_at = time.perf_counter()
    total_steps = len(opened_values)
    decision_steps = 0
    progress_interval = max(1, int(progress_every))
    if total_steps > 0:
        progress_interval = max(
            progress_interval,
            int(np.ceil(total_steps / float(AUDIT_PROGRESS_MAX_UPDATES))),
        )

    for row_idx, opened in enumerate(opened_values):
        predictor._append_new_candle(
            pd.Timestamp(opened),
            tuple(float(v) for v in ohlcv_matrix[row_idx, :]),
        )
        volume_profile_values = (
            predictor._prepare_volume_profile_features_for_latest_candle(opened)
        )
        is_decision_step = _is_live_decision_opened(
            opened, predictor.target_bucket_minutes
        )
        if is_decision_step:
            snapshot = predictor.build_feature_snapshot(volume_profile_values)
            live_vector = snapshot["vector"][0, :].astype(np.float64, copy=True)
            if ignored_basis_feature_indices:
                live_vector[ignored_basis_feature_indices] = audit_df.loc[
                    row_idx, list(ignored_basis_feature_columns)
                ].to_numpy(dtype=np.float64, copy=False)
            live_vector_rows.append(live_vector)
            nonfinite_row = ~np.isfinite(live_vector)
            live_nonfinite_rows.append(nonfinite_row)
            indicator_nan_count_rows.append(len(snapshot["indicator_nan_cols"]))
            current_history_rows = len(predictor.opened_candles)
            history_rows_used_rows.append(current_history_rows)
            history_shortfall_rows.append(
                max(0, int(predictor.required_stable_window) - current_history_rows)
            )
            decision_row_indices.append(row_idx)
            decision_steps += 1
        completed_steps = row_idx + 1
        if progress_enabled and (
            completed_steps == 1
            or completed_steps == total_steps
            or completed_steps % progress_interval == 0
        ):
            elapsed_sec = time.perf_counter() - loop_started_at
            steps_per_sec = (
                completed_steps / elapsed_sec if elapsed_sec > 0 else float("nan")
            )
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
                f"buffer_rows={len(predictor.opened_candles)} "
                f"decision_rows={decision_steps} "
                f"elapsed={_format_duration(elapsed_sec)} "
                f"eta={_format_duration(eta_sec) if np.isfinite(eta_sec) else 'n/a'}"
            )

    decision_audit_df = (
        audit_df.iloc[decision_row_indices].copy().reset_index(drop=True)
    )
    if decision_audit_df.empty:
        raise RuntimeError("No live decision rows were selected for the audit window.")
    modeling_matrix = decision_audit_df[feature_columns].to_numpy(
        dtype=np.float64, copy=True
    )
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
        feature_importance_df=feature_importance_df,
        policy_predictor=predictor,
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
                    "ignored_basis_premium_feature_count": len(
                        ignored_basis_feature_columns
                    ),
                    "ignored_basis_premium_features": ", ".join(
                        ignored_basis_feature_columns
                    ),
                    "use_anchor_vp_state": bool(use_anchor_vp_state),
                    "anchor_vp_state_path": (
                        None
                        if anchor_vp_state_path is None
                        else str(anchor_vp_state_path.with_suffix(".npz"))
                    ),
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
    results,
    feature_name,
    *,
    report_key=None,
    top_n=20,
):
    report = results if report_key is None else results[report_key]
    feature_columns = list(report["feature_columns"])
    if feature_name not in feature_columns:
        raise KeyError(f"Unknown feature_name={feature_name!r}")

    feature_idx = feature_columns.index(feature_name)
    audit_df = report["audit_df"]
    candidate_feature_frame = report.get(
        "candidate_feature_frame", report.get("live_feature_frame")
    )
    step_summary_df = report["step_summary_df"]
    candidate_nonfinite_mask = report.get(
        "candidate_nonfinite_mask",
        report.get("live_nonfinite_mask"),
    )
    reference_nonfinite_mask = report.get(
        "reference_nonfinite_mask",
        report.get("modeling_nonfinite_mask"),
    )
    candidate_series = candidate_feature_frame[feature_name].to_numpy(
        dtype=np.float64, copy=False
    )
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
            "finite_status_mismatch": report["finite_status_mismatch_mask"][
                :, feature_idx
            ],
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
        fixed_pred_series = np.asarray(fixed_pred_matrix, dtype=np.float64)[
            :, feature_idx
        ]
        proba_shift_series = np.asarray(proba_shift_matrix, dtype=np.float64)[
            :, feature_idx
        ]
        gap_reduction_series = np.asarray(gap_reduction_matrix, dtype=np.float64)[
            :, feature_idx
        ]
        reference_pred = step_summary_df[reference_proba_col].to_numpy(
            dtype=np.float64, copy=False
        )
        candidate_pred = step_summary_df[candidate_proba_col].to_numpy(
            dtype=np.float64, copy=False
        )
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
    sort_ascending = [
        ascending for col, ascending in sort_spec if col in drilldown_df.columns
    ]
    return drilldown_df.sort_values(
        sort_cols,
        ascending=sort_ascending,
        na_position="last",
        kind="stable",
    ).head(int(top_n))


def _build_feature_prediction_impact_export_df(
    feature_summary_df,
    *,
    top_k=None,
    only_impactful=False,
):
    feature_summary_df = feature_summary_df.copy()
    if (
        "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed"
        not in feature_summary_df.columns
    ):
        feature_summary_df["mean_abs_proba_gap_reduction_on_drift_rows_if_fixed"] = (
            feature_summary_df.get("mean_abs_proba_gap_reduction_if_fixed", 0.0)
        )
    if (
        "mean_abs_proba_gap_reduction_on_helped_rows_if_fixed"
        not in feature_summary_df.columns
    ):
        feature_summary_df["mean_abs_proba_gap_reduction_on_helped_rows_if_fixed"] = (
            feature_summary_df.get("mean_abs_proba_gap_reduction_if_fixed", 0.0)
        )
    if "mean_gap_reduction_ratio_on_drift_rows_if_fixed" not in feature_summary_df.columns:
        feature_summary_df["mean_gap_reduction_ratio_on_drift_rows_if_fixed"] = 0.0
    if "max_gap_reduction_ratio_on_drift_rows_if_fixed" not in feature_summary_df.columns:
        feature_summary_df["max_gap_reduction_ratio_on_drift_rows_if_fixed"] = 0.0

    slim_columns = [
        "rank",
        "feature",
        "group",
        "builder",
        "net_pred_gap_reduction",
        "mean_pred_gap_reduction_on_drift_rows",
        "mean_pred_gap_reduction_on_helped_rows",
        "mean_gap_reduction_ratio_on_drift_rows",
        "max_gap_reduction_ratio_on_drift_rows",
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
                | export_df["net_pred_gap_reduction"].gt(PREDICTION_DIFF_TOL)
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
        "net_abs_proba_gap_reduction_if_fixed",
        "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
        "mean_abs_proba_gap_reduction_on_helped_rows_if_fixed",
        "mean_gap_reduction_ratio_on_drift_rows_if_fixed",
        "max_gap_reduction_ratio_on_drift_rows_if_fixed",
        "mean_abs_proba_shift_if_fixed",
        "max_abs_proba_shift_if_fixed",
        "rows_signal_mismatch_resolved_if_fixed",
        "rows_proba_diff_gt_tol_resolved_if_fixed",
        "rows_abs_proba_gap_reduced_if_fixed",
        "rows_abs_proba_gap_worsened_if_fixed",
    ]
    rename_map = {
        "builder_name": "builder",
        "net_abs_proba_gap_reduction_if_fixed": "net_pred_gap_reduction",
        "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed": "mean_pred_gap_reduction_on_drift_rows",
        "mean_abs_proba_gap_reduction_on_helped_rows_if_fixed": "mean_pred_gap_reduction_on_helped_rows",
        "mean_gap_reduction_ratio_on_drift_rows_if_fixed": "mean_gap_reduction_ratio_on_drift_rows",
        "max_gap_reduction_ratio_on_drift_rows_if_fixed": "max_gap_reduction_ratio_on_drift_rows",
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
        ("rows_signal_mismatch_resolved_if_fixed", False),
        ("rows_proba_diff_gt_tol_resolved_if_fixed", False),
        ("net_abs_proba_gap_reduction_if_fixed", False),
        ("mean_abs_proba_gap_reduction_on_drift_rows_if_fixed", False),
        ("mean_gap_reduction_ratio_on_drift_rows_if_fixed", False),
        ("mean_abs_proba_shift_if_fixed", False),
        ("max_abs_proba_shift_if_fixed", False),
        ("rows_abs_proba_gap_reduced_if_fixed", False),
        ("feature", True),
    ]
    sort_cols = [
        col for col, _ascending in sort_spec if col in feature_summary_df.columns
    ]
    sort_ascending = [
        ascending for col, ascending in sort_spec if col in feature_summary_df.columns
    ]
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
            | export_df["net_pred_gap_reduction"].gt(PREDICTION_DIFF_TOL)
        )
        export_df = export_df.loc[impact_mask].reset_index(drop=True)
        export_df["rank"] = np.arange(1, len(export_df) + 1, dtype=np.int32)
        export_df = export_df.loc[:, slim_columns]
    if top_k is not None:
        export_df = export_df.head(int(top_k)).reset_index(drop=True)
    return export_df


def _feature_prediction_impact_records(
    feature_summary_df,
    *,
    top_k,
):
    export_df = _build_feature_prediction_impact_export_df(
        feature_summary_df,
        top_k=top_k,
        only_impactful=True,
    )
    return json.loads(export_df.to_json(orient="records"))


def _build_feature_drop_candidates_df(
    feature_summary_df,
    *,
    top_k=None,
):
    if feature_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "feature",
                "group",
                "builder",
                "drop_candidate_reason",
                "importance_gain",
                "importance_split",
                "max_abs_diff",
                "mean_abs_diff",
                "rmse_abs_diff",
                "max_rel_diff",
                "mean_rel_diff",
                "finite_status_mismatch_count",
                "mean_pred_shift",
                "max_pred_shift",
                "signal_flips_resolved",
                "proba_drift_rows_resolved",
            ]
        )

    export_df = feature_summary_df.copy()
    if "builder_name" in export_df.columns:
        export_df["builder"] = export_df["builder_name"]
    elif "builder" not in export_df.columns:
        export_df["builder"] = "unknown"

    if "importance_gain" not in export_df.columns:
        export_df["importance_gain"] = 0.0
    if "importance_split" not in export_df.columns:
        export_df["importance_split"] = 0.0
    if "mean_abs_proba_shift_if_fixed" not in export_df.columns:
        export_df["mean_abs_proba_shift_if_fixed"] = 0.0
    if "max_abs_proba_shift_if_fixed" not in export_df.columns:
        export_df["max_abs_proba_shift_if_fixed"] = 0.0
    if "rows_signal_mismatch_resolved_if_fixed" not in export_df.columns:
        export_df["rows_signal_mismatch_resolved_if_fixed"] = 0
    if "rows_proba_diff_gt_tol_resolved_if_fixed" not in export_df.columns:
        export_df["rows_proba_diff_gt_tol_resolved_if_fixed"] = 0

    export_df["flag_finite_status_mismatch"] = (
        export_df["finite_status_mismatch_count"] > 0
    )
    export_df["flag_max_abs_diff"] = export_df["max_abs_diff"].abs().gt(
        FEATURE_DROP_MAX_ABS_DIFF_TOL
    )
    export_df["flag_mean_abs_diff"] = export_df["mean_abs_diff"].abs().gt(
        FEATURE_DROP_MEAN_ABS_DIFF_TOL
    )
    export_df["flag_max_rel_diff"] = export_df["max_rel_diff"].gt(
        FEATURE_DROP_MAX_REL_DIFF_TOL
    )
    export_df["flag_prediction_impact"] = (
        export_df["max_abs_proba_shift_if_fixed"].gt(PREDICTION_DIFF_TOL)
        | export_df["rows_signal_mismatch_resolved_if_fixed"].gt(0)
        | export_df["rows_proba_diff_gt_tol_resolved_if_fixed"].gt(0)
    )
    export_df["flag_abs_diff_material"] = (
        export_df["flag_max_abs_diff"] | export_df["flag_mean_abs_diff"]
    )
    export_df["drop_candidate"] = (
        export_df["flag_finite_status_mismatch"]
        | export_df["flag_max_rel_diff"]
        | (
            export_df["flag_abs_diff_material"]
            & export_df["flag_prediction_impact"]
        )
    )

    reason_cols = [
        ("flag_finite_status_mismatch", "finite_status_mismatch"),
        (
            "flag_max_abs_diff",
            f"max_abs_diff>{FEATURE_DROP_MAX_ABS_DIFF_TOL:g}",
        ),
        (
            "flag_mean_abs_diff",
            f"mean_abs_diff>{FEATURE_DROP_MEAN_ABS_DIFF_TOL:g}",
        ),
        (
            "flag_max_rel_diff",
            f"max_rel_diff>{FEATURE_DROP_MAX_REL_DIFF_TOL:g}",
        ),
    ]
    reasons = []
    for row in export_df.itertuples(index=False):
        parts = [label for flag_name, label in reason_cols if getattr(row, flag_name)]
        reasons.append("; ".join(parts))
    export_df["drop_candidate_reason"] = reasons

    export_df = export_df.loc[export_df["drop_candidate"]].copy()
    export_df = export_df.sort_values(
        [
            "finite_status_mismatch_count",
            "max_rel_diff",
            "mean_rel_diff",
            "max_abs_diff",
            "mean_abs_diff",
            "importance_gain",
            "importance_split",
            "max_abs_proba_shift_if_fixed",
            "mean_abs_proba_shift_if_fixed",
            "feature",
        ],
        ascending=[False, False, False, False, False, False, False, False, False, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    export_df = export_df.loc[
        :,
        [
            "feature",
            "group",
            "builder",
            "drop_candidate_reason",
            "importance_gain",
            "importance_split",
            "max_abs_diff",
            "mean_abs_diff",
            "rmse_abs_diff",
            "max_rel_diff",
            "mean_rel_diff",
            "finite_status_mismatch_count",
            "mean_abs_proba_shift_if_fixed",
            "max_abs_proba_shift_if_fixed",
            "rows_signal_mismatch_resolved_if_fixed",
            "rows_proba_diff_gt_tol_resolved_if_fixed",
        ],
    ].rename(
        columns={
            "mean_abs_proba_shift_if_fixed": "mean_pred_shift",
            "max_abs_proba_shift_if_fixed": "max_pred_shift",
            "rows_signal_mismatch_resolved_if_fixed": "signal_flips_resolved",
            "rows_proba_diff_gt_tol_resolved_if_fixed": "proba_drift_rows_resolved",
        }
    )
    export_df.insert(0, "rank", np.arange(1, len(export_df) + 1, dtype=np.int32))
    if top_k is not None:
        export_df = export_df.head(int(top_k)).reset_index(drop=True)
    return export_df


def save_audit_outputs(
    results,
    *,
    output_dir,
    drilldown_feature_name=None,
    top_n=50,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = {}
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
    written_paths["feature_drop_candidates_csv"] = (
        output_dir / "live_vs_stored_feature_drop_candidates.csv"
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
    drop_candidate_df = _build_feature_drop_candidates_df(report["feature_summary_df"])
    top10_drop_candidate_records = json.loads(
        drop_candidate_df.head(10).to_json(orient="records")
    )
    top25_drop_candidate_records = json.loads(
        drop_candidate_df.head(25).to_json(orient="records")
    )

    written_paths["summary_json"].write_text(
        json.dumps(
            {
                "live_vs_stored": report["summary"].to_dict(),
                "drift_reason": drift_reason_report["summary"].to_dict(),
                "feature_drop_candidate_thresholds": {
                    "max_abs_diff_gt": FEATURE_DROP_MAX_ABS_DIFF_TOL,
                    "mean_abs_diff_gt": FEATURE_DROP_MEAN_ABS_DIFF_TOL,
                    "max_rel_diff_gt": FEATURE_DROP_MAX_REL_DIFF_TOL,
                },
                "feature_drop_candidate_count": int(len(drop_candidate_df)),
                "top10_feature_drop_candidates": top10_drop_candidate_records,
                "top25_feature_drop_candidates": top25_drop_candidate_records,
                "prediction_impact_feature_rank_metric": (
                    "rows_proba_diff_gt_tol_resolved_then_net_pred_gap_reduction"
                ),
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
    decision_sort_cols = [
        col
        for col, _ascending in decision_sort_spec
        if col in report["step_summary_df"]
    ]
    decision_sort_ascending = [
        ascending
        for col, ascending in decision_sort_spec
        if col in report["step_summary_df"]
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
    drop_candidate_df.to_csv(
        written_paths["feature_drop_candidates_csv"],
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
        drilldown_path = (
            output_dir / f"live_vs_stored_drilldown_{drilldown_feature_name}.csv"
        )
        feature_drilldown(
            results,
            drilldown_feature_name,
            report_key="live_vs_stored_report",
            top_n=top_n,
        ).to_csv(drilldown_path, index=False)
        written_paths["drilldown_csv"] = drilldown_path

    return written_paths


def _default_output_dir():
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    return Path("data/analysis/live_feature_parity") / stamp


def main():
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

    violations = _print_audit_console_summary(results, written_paths)
    if violations:
        raise RuntimeError(
            "Live-vs-stored audit exceeded configured drift guardrails. "
            "See console summary above."
        )


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

