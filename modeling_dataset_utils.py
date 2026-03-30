import json
import re
from pathlib import Path

import numpy as np

from data_quality_filters import normalize_drop_frozen_ohlc_blocks_config
from features.candle_features import (
    RAW_OHLCV_COLS,
    STREAK_FEATURE_PREFIX,
    SUPPORTED_CANDLE_FEATURE_COLS,
)
from features.session_open_features import (
    SUPPORTED_SESSION_COUNTER_COLS,
    is_session_counter_feature,
)
from features.realized_volatility import is_realized_volatility_feature
from features.volume_profile_fixed_range import is_volume_profile_feature
from project_config import (
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


def _dedupe_ordered(values):
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


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


def load_modeling_dataset_settings(config_path=MODELING_DATASET_CONFIG_FILE):
    if Path(config_path) != MODELING_DATASET_CONFIG_FILE:
        raise ValueError(
            "Custom modeling config path overrides are no longer supported. "
            f"Expected: {MODELING_DATASET_CONFIG_FILE}"
        )

    settings = load_modeling_settings(active_config_path=ACTIVE_PROFILE_CONFIG_FILE)
    excluded_feature_names = _normalize_optional_feature_names(
        list(settings.get("excluded_feature_names") or ()),
        source_label="modeling.feature_selection.excluded_feature_names",
    )
    float_precision = _normalize_modeling_float_precision(
        settings.get("float_precision"),
        source_label="modeling.float_precision",
    )

    return {
        "raw_data_dir": Path(settings["raw_data_dir"]),
        "data_dir": Path(settings["raw_data_dir"]),
        "base_data_file": str(settings["base_data_file"]),
        "modeling_output_dir": Path(settings["modeling_output_dir"]),
        "output_suffix": str(settings["output_suffix"]),
        "fit_results_dir": Path(settings["fit_results_dir"]),
        "preview_rows": int(settings["preview_rows"]),
        "candle_streak_intervals": [
            str(v) for v in settings["candle_streak_intervals"]
        ],
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
        "head_csv": output_dir / f"{output_stem}_head{preview_rows}.csv",
        "tail_csv": output_dir / f"{output_stem}_tail{preview_rows}.csv",
    }


def resolve_modeling_dataset_parquet_path(config_path=MODELING_DATASET_CONFIG_FILE):
    settings = load_modeling_dataset_settings(config_path=config_path)
    return resolve_modeling_dataset_output_paths(settings)["parquet"]


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
    return {
        **subset_info,
        "features": filtered_features,
        "count": len(filtered_features),
        "source_count": int(subset_info["count"]),
        "excluded_feature_names": excluded_feature_names,
        "excluded_count": len(excluded_feature_names),
        "excluded_from_subset_count": len(removed_features),
    }


def load_excluded_feature_names_from_settings(settings):
    excluded_feature_names = tuple(settings.get("excluded_feature_names") or ())
    if not excluded_feature_names:
        return None
    return {
        "features": excluded_feature_names,
        "count": len(excluded_feature_names),
    }


def split_feature_subset(feature_names):
    raw_ohlcv_cols = []
    candle_feature_cols = []
    streak_feature_cols = []
    streak_intervals = []
    session_feature_cols = []
    realized_volatility_feature_cols = []
    indicator_feature_cols = []
    volume_profile_feature_cols = []
    unclassified_feature_cols = []

    candle_feature_set = set(SUPPORTED_CANDLE_FEATURE_COLS)
    raw_ohlcv_set = set(RAW_OHLCV_COLS)
    session_feature_set = set(SUPPORTED_SESSION_COUNTER_COLS)

    for feature_name in feature_names:
        if feature_name in raw_ohlcv_set:
            raw_ohlcv_cols.append(feature_name)
            continue
        if feature_name in candle_feature_set:
            candle_feature_cols.append(feature_name)
            continue
        if feature_name.startswith(STREAK_FEATURE_PREFIX):
            streak_feature_cols.append(feature_name)
            streak_intervals.append(feature_name[len(STREAK_FEATURE_PREFIX) :])
            continue
        if feature_name in session_feature_set or is_session_counter_feature(
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
        }
    )
    return payload
