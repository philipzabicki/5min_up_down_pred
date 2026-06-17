import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from features.basis_premium_features import (
    is_basis_premium_feature,
    validate_basis_premium_feature_columns,
)
from features.candle_features import (
    RAW_OHLCV_COLS,
    STREAK_FEATURE_PREFIX,
    SUPPORTED_CANDLE_FEATURE_COLS,
    is_deprecated_candle_feature_col,
)
from features.realized_volatility import (
    is_realized_volatility_feature,
)
from features.session_open_features import (
    SUPPORTED_SESSION_OPEN_FEATURE_COLS,
    is_session_open_feature,
)
from features.volume_profile_fixed_range import is_volume_profile_feature
from features.volume_profile_fixed_range import validate_volume_profile_feature_columns
from utils.collections import dedupe_ordered as _dedupe_ordered
from utils.project_config import (
    ACTIVE_CONFIG_PATH,
    MODELING_CONFIG_PATH,
    load_modeling_settings,
)

MODELING_DATASET_CONFIG_FILE = MODELING_CONFIG_PATH
ACTIVE_PROFILE_CONFIG_FILE = ACTIVE_CONFIG_PATH
FEATURE_SUBSET_JSON_KEYS = (
    "final_feature_list",
    "recommended_features",
    "feature_columns",
)
SUPPORTED_MODELING_FLOAT_PRECISIONS = ("float32", "float64")
_TXT_METADATA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
PARQUET_MAGIC_BYTES = b"PAR1"
DEFAULT_DROP_FROZEN_OHLC_BLOCKS_CONFIG = {
    "enabled": False,
    "min_block_len": 3,
}
TARGET_WEIGHT_COL = "target_5m_weight"
TARGET_WEIGHT_MINUTE_MODULO = 5
TARGET_WEIGHT_MINUTE_REMAINDER = 4
TARGET_WEIGHT_DECISION_VALUE = 0.4625
TARGET_WEIGHT_OTHER_VALUE = 0.5375 / 4


def _format_weight_key(value):
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def compute_decision_mask_from_opened(
        opened_values,
        minute_modulo=TARGET_WEIGHT_MINUTE_MODULO,
        minute_remainder=TARGET_WEIGHT_MINUTE_REMAINDER,
):
    opened_index = pd.DatetimeIndex(pd.to_datetime(opened_values, errors="raise"))
    opened_minute = opened_index.minute.to_numpy(dtype=np.int16, copy=False)
    return (opened_minute % int(minute_modulo)) == int(minute_remainder)


def compute_target_weights_from_opened(opened_values, dtype=np.float64):
    decision_mask = compute_decision_mask_from_opened(opened_values)
    weights = np.where(
        decision_mask,
        TARGET_WEIGHT_DECISION_VALUE,
        TARGET_WEIGHT_OTHER_VALUE,
    )
    return weights.astype(dtype, copy=False)


def compute_binary_close_target_from_opened(
        opened_values,
        close_values,
        horizon_minutes,
        dtype=np.float64,
):
    horizon = int(horizon_minutes)
    if horizon <= 0:
        raise ValueError(f"horizon_minutes must be > 0, got: {horizon_minutes}")

    opened_index = pd.DatetimeIndex(pd.to_datetime(opened_values, errors="raise"))
    if opened_index.has_duplicates:
        dup_count = int(opened_index.duplicated().sum())
        raise ValueError(f"Duplicate Opened values found: {dup_count}")

    close_np = pd.to_numeric(close_values, errors="coerce").to_numpy(
        dtype=np.float64,
        copy=False,
    )
    close_series = pd.Series(close_np, index=opened_index)
    future_opened = opened_index + pd.Timedelta(minutes=horizon)
    future_close = close_series.reindex(future_opened).to_numpy(
        dtype=np.float64, copy=False
    )
    current_close = close_series.to_numpy(dtype=np.float64, copy=False)

    target = np.full(len(close_series), np.nan, dtype=dtype)
    valid_mask = np.isfinite(current_close) & np.isfinite(future_close)
    if np.any(valid_mask):
        # Keep target semantics aligned with Polymarket settlement: ties resolve Up.
        target[valid_mask] = (
                future_close[valid_mask] >= current_close[valid_mask]
        ).astype(dtype, copy=False)
    return target


def add_target_weights(
        df,
        opened_col="Opened",
        weight_col=TARGET_WEIGHT_COL,
        dtype=np.float64,
):
    if opened_col not in df.columns:
        raise ValueError(
            f"Cannot build target weights without opened column '{opened_col}'."
        )

    out = df.copy(deep=False)
    out[weight_col] = compute_target_weights_from_opened(
        out[opened_col],
        dtype=dtype,
    )
    return out


def summarize_target_weights(weights):
    weights_np = np.asarray(weights, dtype=np.float64)
    if weights_np.ndim != 1:
        raise ValueError("Target weights summary expects a 1D array.")
    if weights_np.size == 0:
        raise ValueError("Cannot summarize empty target weights.")

    unique_weights, counts = np.unique(weights_np, return_counts=True)
    distribution = {
        _format_weight_key(weight): int(count)
        for weight, count in zip(unique_weights, counts)
    }
    return {
        "min": float(np.min(weights_np)),
        "max": float(np.max(weights_np)),
        "mean": float(np.mean(weights_np)),
        "sum": float(np.sum(weights_np)),
        "distribution": distribution,
    }


def normalize_drop_frozen_ohlc_blocks_config(raw_config):
    if raw_config is None:
        return dict(DEFAULT_DROP_FROZEN_OHLC_BLOCKS_CONFIG)
    if not isinstance(raw_config, dict):
        raise ValueError("drop_frozen_ohlc_blocks config must be a JSON object.")

    enabled = bool(raw_config.get("enabled", False))
    min_block_len = int(
        raw_config.get(
            "min_block_len",
            DEFAULT_DROP_FROZEN_OHLC_BLOCKS_CONFIG["min_block_len"],
        )
    )
    if min_block_len < 1:
        raise ValueError(
            f"drop_frozen_ohlc_blocks.min_block_len must be >= 1, got: {min_block_len}"
        )

    return {
        "enabled": enabled,
        "min_block_len": int(min_block_len),
    }


def drop_frozen_ohlc_blocks(
        df,
        raw_config=None,
        opened_col="Opened",
        ohlc_cols=("Open", "High", "Low", "Close"),
):
    config = normalize_drop_frozen_ohlc_blocks_config(raw_config)
    summary = {
        "enabled": bool(config["enabled"]),
        "min_block_len": int(config["min_block_len"]),
        "rows_before": len(df),
        "rows_removed": 0,
        "rows_after": len(df),
        "blocks_removed": 0,
        "largest_block_len": 0,
        "first_removed_opened": None,
        "last_removed_opened": None,
    }
    if not config["enabled"] or df.empty:
        return df, summary

    required_cols = [col for col in (opened_col, *ohlc_cols) if col not in df.columns]
    if required_cols:
        raise ValueError(
            "Missing required columns for drop_frozen_ohlc_blocks: "
            + ", ".join(required_cols)
        )

    same_prev_mask = (
        df.loc[:, list(ohlc_cols)].eq(df.loc[:, list(ohlc_cols)].shift(1)).all(axis=1)
    )
    same_prev_mask = same_prev_mask.fillna(False)
    run_group = (~same_prev_mask).cumsum()
    run_lengths = same_prev_mask.groupby(run_group).transform("sum")
    drop_mask = same_prev_mask & (run_lengths >= int(config["min_block_len"]))

    if not bool(drop_mask.any()):
        return df.reset_index(drop=True), summary

    removed_opened = pd.to_datetime(
        df.loc[drop_mask, opened_col],
        errors="coerce",
    )
    block_starts = drop_mask & ~drop_mask.shift(1, fill_value=False)
    largest_block_len = int(run_lengths.loc[drop_mask].max())

    summary.update(
        {
            "rows_removed": int(drop_mask.sum()),
            "rows_after": int((~drop_mask).sum()),
            "blocks_removed": int(block_starts.sum()),
            "largest_block_len": largest_block_len,
            "first_removed_opened": (
                None if removed_opened.empty else str(removed_opened.iloc[0])
            ),
            "last_removed_opened": (
                None if removed_opened.empty else str(removed_opened.iloc[-1])
            ),
        }
    )
    filtered = df.loc[~drop_mask].reset_index(drop=True)
    return filtered, summary


def _normalize_feature_names(features, source_path):
    normalized = []
    for raw_feature in features:
        feature = str(raw_feature).strip()
        if not feature:
            raise ValueError(
                f"Feature subset contains an empty feature name: {source_path}"
            )
        normalized.append(feature)

    normalized = _dedupe_ordered(normalized)
    if not normalized:
        raise ValueError(f"Feature subset is empty: {source_path}")
    validate_volume_profile_feature_columns(
        normalized,
        source_label=f"feature subset {source_path}",
    )
    validate_basis_premium_feature_columns(
        normalized,
        source_label=f"feature subset {source_path}",
    )
    return tuple(normalized)


def _normalize_optional_feature_names(features, source_label):
    if features is None:
        return tuple()
    if not isinstance(features, list):
        raise ValueError(
            f"Invalid {source_label}: expected a JSON array of feature names."
        )

    normalized = []
    for raw_feature in features:
        feature = str(raw_feature).strip()
        if not feature:
            raise ValueError(f"{source_label} contains an empty feature name.")
        normalized.append(feature)
    return tuple(_dedupe_ordered(normalized))


def _normalize_modeling_float_precision(raw_value, *, source_label):
    if raw_value is None:
        allowed = ", ".join(SUPPORTED_MODELING_FLOAT_PRECISIONS)
        raise ValueError(
            f"Missing required {source_label}. Expected one of: {allowed}."
        )

    dtype_name = str(raw_value).strip().lower()
    if dtype_name not in SUPPORTED_MODELING_FLOAT_PRECISIONS:
        allowed = ", ".join(SUPPORTED_MODELING_FLOAT_PRECISIONS)
        raise ValueError(
            f"Invalid {source_label}: {raw_value!r}. Expected one of: {allowed}."
        )
    return dtype_name


def _exclude_features(feature_names, excluded_feature_names, *, source_label):
    if not excluded_feature_names:
        return tuple(feature_names), tuple()

    excluded_feature_set = set(excluded_feature_names)
    kept = tuple(
        feature_name
        for feature_name in feature_names
        if feature_name not in excluded_feature_set
    )
    removed = tuple(
        feature_name
        for feature_name in feature_names
        if feature_name in excluded_feature_set
    )
    if not kept:
        raise ValueError(
            f"All features were excluded after applying excluded_feature_names to {source_label}."
        )
    return kept, removed


def load_modeling_dataset_settings(
        config_path=MODELING_DATASET_CONFIG_FILE,
        *,
        asset=None,
        dataset_profile_name=None,
        modeling_profile_name=None,
):
    if Path(config_path) != MODELING_DATASET_CONFIG_FILE:
        raise ValueError(
            "Custom modeling config path overrides are no longer supported. "
            f"Expected: {MODELING_DATASET_CONFIG_FILE}"
        )

    settings = load_modeling_settings(
        active_config_path=ACTIVE_PROFILE_CONFIG_FILE,
        asset=asset,
        dataset_profile_name=dataset_profile_name,
        modeling_profile_name=modeling_profile_name,
    )
    excluded_feature_names = _normalize_optional_feature_names(
        list(settings.get("excluded_feature_names") or ()),
        source_label="modeling.feature_selection.excluded_feature_names",
    )
    float_precision = _normalize_modeling_float_precision(
        settings.get("float_precision"),
        source_label="modeling.float_precision",
    )

    return {
        "active_asset": str(settings["active_asset"]),
        "symbol": str(settings["symbol"]),
        "interval": str(settings["interval"]),
        "market": str(settings["market"]),
        "volume_symbol": str(settings["volume_symbol"]),
        "volume_market": str(settings["volume_market"]),
        "raw_data_dir": Path(settings["raw_data_dir"]),
        "data_dir": Path(settings["raw_data_dir"]),
        "base_data_file": str(settings["base_data_file"]),
        "modeling_output_dir": Path(settings["modeling_output_dir"]),
        "output_suffix": str(settings["output_suffix"]),
        "fit_results_dir": Path(settings["fit_results_dir"]),
        "preview_rows": int(settings["preview_rows"]),
        "candle_streak_intervals": dict(settings["candle_streak_intervals"]),
        "feature_intervals": dict(settings.get("feature_intervals") or {}),
        "basis_premium_features": dict(settings.get("basis_premium_features") or {}),
        "feature_subset_path": settings.get("feature_subset_path"),
        "feature_subset_list_key": settings.get("feature_subset_list_key"),
        "excluded_feature_names": excluded_feature_names,
        "float_precision": float_precision,
        "volume_profile_fixed_range": settings.get("volume_profile_fixed_range"),
        "drop_frozen_ohlc_blocks": normalize_drop_frozen_ohlc_blocks_config(
            settings.get("drop_frozen_ohlc_blocks")
        ),
        "train_lgbm": dict(settings.get("train_lgbm") or {}),
    }


def resolve_modeling_dataset_output_stem(settings):
    return f"{Path(settings['base_data_file']).stem}{settings['output_suffix']}"


def resolve_raw_dataset_input_path(settings):
    return Path(settings["raw_data_dir"]) / str(settings["base_data_file"])


def resolve_modeling_dataset_output_paths(settings):
    output_dir = Path(settings["modeling_output_dir"])
    output_stem = resolve_modeling_dataset_output_stem(settings)
    preview_rows = int(settings["preview_rows"])
    return {
        "parquet": output_dir / f"{output_stem}.parquet",
        "metadata_json": output_dir / f"{output_stem}_metadata.json",
        "head_csv": output_dir / f"{output_stem}_head{preview_rows}.csv",
        "tail_csv": output_dir / f"{output_stem}_tail{preview_rows}.csv",
    }


def resolve_modeling_dataset_parquet_path(config_path=MODELING_DATASET_CONFIG_FILE):
    settings = load_modeling_dataset_settings(config_path=config_path)
    return resolve_modeling_dataset_output_paths(settings)["parquet"]


def resolve_modeling_dataset_metadata_path(config_path=MODELING_DATASET_CONFIG_FILE):
    settings = load_modeling_dataset_settings(config_path=config_path)
    return resolve_modeling_dataset_output_paths(settings)["metadata_json"]


def resolve_modeling_dataset_metadata_path_for_parquet(parquet_path):
    parquet_path = Path(parquet_path)
    return parquet_path.with_name(f"{parquet_path.stem}_metadata.json")


def validate_parquet_magic_bytes(parquet_path):
    parquet_path = Path(parquet_path)
    file_stat = parquet_path.stat()
    file_size = file_stat.st_size
    if file_size < len(PARQUET_MAGIC_BYTES) * 2:
        raise ValueError(
            "Invalid Parquet dataset: file is too small to contain a Parquet "
            f"header/footer. path={parquet_path} size_bytes={file_size}. "
            "Rebuild it with create_modeling_dataset.py."
        )

    with parquet_path.open("rb") as fh:
        header = fh.read(len(PARQUET_MAGIC_BYTES))
        fh.seek(-len(PARQUET_MAGIC_BYTES), 2)
        footer = fh.read(len(PARQUET_MAGIC_BYTES))

    if header != PARQUET_MAGIC_BYTES or footer != PARQUET_MAGIC_BYTES:
        modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat(
            sep=" ",
            timespec="seconds",
        )
        raise ValueError(
            "Invalid Parquet dataset: missing Parquet magic bytes at "
            f"{'header' if header != PARQUET_MAGIC_BYTES else 'footer'}. "
            f"path={parquet_path} size_bytes={file_size} modified_at={modified_at} "
            f"header={header!r} footer={footer!r}. The file is probably truncated "
            "or was not fully written. Rebuild it with create_modeling_dataset.py."
        )


def load_modeling_dataset_artifact_metadata(parquet_path):
    metadata_path = resolve_modeling_dataset_metadata_path_for_parquet(parquet_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Modeling dataset metadata not found: {metadata_path}")

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Unsupported modeling dataset metadata payload type: {type(payload)!r}"
        )
    return payload, metadata_path


def resolve_oof_prediction_output_paths(settings, *, preview_rows):
    output_dir = Path(settings["modeling_output_dir"])
    output_stem = f"{Path(settings['base_data_file']).stem}_oof_predictions"
    return {
        "parquet": output_dir / f"{output_stem}.parquet",
        "head_csv": output_dir / f"{output_stem}_head{int(preview_rows)}.csv",
        "tail_csv": output_dir / f"{output_stem}_tail{int(preview_rows)}.csv",
    }


def resolve_modeling_float_dtype_name(settings):
    return _normalize_modeling_float_precision(
        settings.get("float_precision"),
        source_label="settings['float_precision']",
    )


def resolve_modeling_float_dtype(settings):
    dtype_name = resolve_modeling_float_dtype_name(settings)
    if dtype_name == "float64":
        return np.float64
    return np.float32


def _load_feature_subset_from_json(path, list_key=None):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        feature_names = payload
        list_key_used = None
        metadata = {}
    elif isinstance(payload, dict):
        keys_to_try = [list_key] if list_key else list(FEATURE_SUBSET_JSON_KEYS)
        keys_to_try = [key for key in keys_to_try if key]
        feature_names = None
        list_key_used = None
        for key in keys_to_try:
            candidate = payload.get(key)
            if isinstance(candidate, list):
                feature_names = candidate
                list_key_used = key
                break
        if feature_names is None:
            tried = ", ".join(keys_to_try)
            raise ValueError(
                f"Could not find a feature list in {path}. Tried keys: {tried}"
            )
        metadata = payload
    else:
        raise ValueError(
            f"Unsupported feature subset JSON payload type in {path}: {type(payload)!r}"
        )

    return {
        "features": _normalize_feature_names(feature_names, source_path=path),
        "format": "json",
        "list_key": list_key_used,
        "metadata": metadata,
    }


def _load_feature_subset_from_text(path):
    metadata = {}
    feature_names = []
    in_feature_section = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            if metadata or feature_names:
                in_feature_section = True
            continue
        if not in_feature_section and _TXT_METADATA_RE.match(line):
            key, value = line.split("=", 1)
            metadata[key] = value
            continue
        in_feature_section = True
        feature_names.append(line)

    return {
        "features": _normalize_feature_names(feature_names, source_path=path),
        "format": "text",
        "list_key": None,
        "metadata": metadata,
    }


def load_feature_subset(path, list_key=None):
    subset_path = Path(path)
    if not subset_path.exists():
        raise FileNotFoundError(f"Feature subset file not found: {subset_path}")

    suffix = subset_path.suffix.lower()
    if suffix == ".json":
        loaded = _load_feature_subset_from_json(subset_path, list_key=list_key)
    elif suffix in {".txt", ".lst"}:
        loaded = _load_feature_subset_from_text(subset_path)
    else:
        raise ValueError(
            f"Unsupported feature subset file extension for {subset_path}. "
            "Use .json or .txt."
        )

    metadata = loaded["metadata"]
    return {
        "path": subset_path,
        "features": loaded["features"],
        "count": len(loaded["features"]),
        "format": loaded["format"],
        "list_key": loaded["list_key"],
        "created_utc": metadata.get("created_utc"),
        "source_data_path": metadata.get("data_path"),
        "metadata": metadata,
    }


def load_feature_subset_from_settings(settings):
    subset_path = settings.get("feature_subset_path")
    if not subset_path:
        return None
    subset_info = load_feature_subset(
        subset_path,
        list_key=settings.get("feature_subset_list_key"),
    )
    excluded_feature_names = tuple(settings.get("excluded_feature_names") or ())
    filtered_features, removed_features = _exclude_features(
        subset_info["features"],
        excluded_feature_names,
        source_label=f"feature_subset_path={subset_path}",
    )
    deprecated_removed_features = tuple(
        feature
        for feature in filtered_features
        if is_deprecated_candle_feature_col(feature)
    )
    if deprecated_removed_features:
        filtered_features = tuple(
            feature
            for feature in filtered_features
            if not is_deprecated_candle_feature_col(feature)
        )
    return {
        **subset_info,
        "features": filtered_features,
        "count": len(filtered_features),
        "source_count": int(subset_info["count"]),
        "excluded_feature_names": excluded_feature_names,
        "excluded_count": len(excluded_feature_names),
        "excluded_from_subset_count": len(removed_features),
        "deprecated_removed_feature_names": deprecated_removed_features,
        "deprecated_removed_count": len(deprecated_removed_features),
    }


def load_excluded_feature_names_from_settings(settings):
    excluded_feature_names = tuple(settings.get("excluded_feature_names") or ())
    if not excluded_feature_names:
        return None
    return {
        "features": excluded_feature_names,
        "count": len(excluded_feature_names),
    }


def split_feature_subset(feature_names, *, source_label="feature columns"):
    validate_volume_profile_feature_columns(
        feature_names,
        source_label=source_label,
    )
    validate_basis_premium_feature_columns(
        feature_names,
        source_label=source_label,
    )
    raw_ohlcv_cols = []
    candle_feature_cols = []
    streak_feature_cols = []
    streak_intervals = []
    session_feature_cols = []
    realized_volatility_feature_cols = []
    basis_premium_feature_cols = []
    indicator_feature_cols = []
    volume_profile_feature_cols = []
    unclassified_feature_cols = []

    candle_feature_set = set(SUPPORTED_CANDLE_FEATURE_COLS)
    raw_ohlcv_set = set(RAW_OHLCV_COLS)
    session_feature_set = set(SUPPORTED_SESSION_OPEN_FEATURE_COLS)

    for feature_name in feature_names:
        feature_name = str(feature_name).strip()
        if feature_name in raw_ohlcv_set:
            raw_ohlcv_cols.append(feature_name)
            continue
        if feature_name in candle_feature_set:
            candle_feature_cols.append(feature_name)
            continue
        if feature_name.startswith(STREAK_FEATURE_PREFIX):
            streak_feature_cols.append(feature_name)
            streak_intervals.append(feature_name[len(STREAK_FEATURE_PREFIX):])
            continue
        if feature_name in session_feature_set or is_session_open_feature(
                feature_name
        ):
            session_feature_cols.append(feature_name)
            continue
        if is_realized_volatility_feature(feature_name):
            realized_volatility_feature_cols.append(feature_name)
            continue
        if is_volume_profile_feature(feature_name):
            volume_profile_feature_cols.append(feature_name)
            continue
        if is_basis_premium_feature(feature_name):
            basis_premium_feature_cols.append(feature_name)
            continue
        if "_fit_" in feature_name:
            indicator_feature_cols.append(feature_name)
            continue
        unclassified_feature_cols.append(feature_name)

    return {
        "raw_ohlcv_cols": tuple(raw_ohlcv_cols),
        "candle_feature_cols": tuple(candle_feature_cols),
        "streak_feature_cols": tuple(streak_feature_cols),
        "streak_intervals": tuple(_dedupe_ordered(streak_intervals)),
        "session_feature_cols": tuple(session_feature_cols),
        "realized_volatility_feature_cols": tuple(realized_volatility_feature_cols),
        "basis_premium_feature_cols": tuple(basis_premium_feature_cols),
        "indicator_feature_cols": tuple(indicator_feature_cols),
        "volume_profile_feature_cols": tuple(volume_profile_feature_cols),
        "unclassified_feature_cols": tuple(unclassified_feature_cols),
    }


def summarize_feature_subset(subset_info, excluded_features=None):
    excluded_feature_names = tuple()
    if excluded_features is not None:
        excluded_feature_names = tuple(excluded_features["features"])
    elif subset_info is not None:
        excluded_feature_names = tuple(subset_info.get("excluded_feature_names") or ())

    if subset_info is None and not excluded_feature_names:
        return {"enabled": False}

    payload = {
        "enabled": True,
        "subset_enabled": subset_info is not None,
        "exclusions_enabled": bool(excluded_feature_names),
        "excluded_count": len(excluded_feature_names),
        "excluded_feature_names": list(excluded_feature_names),
    }
    if subset_info is None:
        payload.update(
            {
                "path": None,
                "count": None,
                "source_count": None,
                "format": None,
                "list_key": None,
                "created_utc": None,
                "source_data_path": None,
                "excluded_from_subset_count": 0,
                "deprecated_removed_count": 0,
                "deprecated_removed_feature_names": [],
            }
        )
        return payload

    payload.update(
        {
            "path": str(subset_info["path"]),
            "count": int(subset_info["count"]),
            "source_count": int(subset_info.get("source_count", subset_info["count"])),
            "format": subset_info["format"],
            "list_key": subset_info["list_key"],
            "created_utc": subset_info.get("created_utc"),
            "source_data_path": subset_info.get("source_data_path"),
            "excluded_from_subset_count": int(
                subset_info.get("excluded_from_subset_count", 0)
            ),
            "deprecated_removed_count": int(
                subset_info.get("deprecated_removed_count", 0)
            ),
            "deprecated_removed_feature_names": list(
                subset_info.get("deprecated_removed_feature_names") or ()
            ),
        }
    )
    return payload
