import json
import re
from pathlib import Path

from common_config_utils import load_json_object, require_positive_int, require_text
from features.candle_features import (
    RAW_OHLCV_COLS,
    STREAK_FEATURE_PREFIX,
    SUPPORTED_CANDLE_FEATURE_COLS,
)
from features.session_open_features import (
    SUPPORTED_SESSION_COUNTER_COLS,
    is_session_counter_feature,
)
from features.volume_profile_fixed_range import is_volume_profile_feature


MODELING_DATASET_CONFIG_FILE = Path("configs/modeling_dataset_config.json")
FEATURE_SUBSET_JSON_KEYS = (
    "final_feature_list",
    "recommended_features",
    "feature_columns",
)
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
            raise ValueError(f"Feature subset contains an empty feature name: {source_path}")
        normalized.append(feature)

    normalized = _dedupe_ordered(normalized)
    if not normalized:
        raise ValueError(f"Feature subset is empty: {source_path}")
    return tuple(normalized)


def load_modeling_dataset_settings(config_path=MODELING_DATASET_CONFIG_FILE):
    payload = load_json_object(config_path)
    streak_intervals = payload.get("candle_streak_intervals")
    if not isinstance(streak_intervals, list) or not streak_intervals:
        raise ValueError(
            "Missing or invalid config key: candle_streak_intervals (non-empty list required)."
        )

    feature_subset_path = payload.get("feature_subset_path")
    if feature_subset_path is None or str(feature_subset_path).strip() == "":
        feature_subset_path = None
    else:
        feature_subset_path = Path(str(feature_subset_path).strip())

    feature_subset_list_key = str(payload.get("feature_subset_list_key", "")).strip()
    if not feature_subset_list_key:
        feature_subset_list_key = None

    return {
        "data_dir": Path(require_text(payload, "data_dir")),
        "base_data_file": require_text(payload, "base_data_file"),
        "output_suffix": require_text(payload, "output_suffix"),
        "fit_results_dir": Path(require_text(payload, "fit_results_dir")),
        "preview_rows": require_positive_int(payload, "preview_rows"),
        "candle_streak_intervals": [str(v) for v in streak_intervals],
        "feature_subset_path": feature_subset_path,
        "feature_subset_list_key": feature_subset_list_key,
        "volume_profile_fixed_range": payload.get("volume_profile_fixed_range"),
    }


def resolve_modeling_dataset_output_stem(settings):
    return f"{Path(settings['base_data_file']).stem}{settings['output_suffix']}"


def resolve_modeling_dataset_output_paths(settings):
    data_dir = Path(settings["data_dir"])
    output_stem = resolve_modeling_dataset_output_stem(settings)
    preview_rows = int(settings["preview_rows"])
    return {
        "parquet": data_dir / f"{output_stem}.parquet",
        "head_csv": data_dir / f"{output_stem}_head{preview_rows}.csv",
        "tail_csv": data_dir / f"{output_stem}_tail{preview_rows}.csv",
    }


def resolve_modeling_dataset_parquet_path(config_path=MODELING_DATASET_CONFIG_FILE):
    settings = load_modeling_dataset_settings(config_path=config_path)
    return resolve_modeling_dataset_output_paths(settings)["parquet"]


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
    return load_feature_subset(
        subset_path,
        list_key=settings.get("feature_subset_list_key"),
    )


def split_feature_subset(feature_names):
    raw_ohlcv_cols = []
    candle_feature_cols = []
    streak_feature_cols = []
    streak_intervals = []
    session_feature_cols = []
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
        if feature_name in session_feature_set or is_session_counter_feature(feature_name):
            session_feature_cols.append(feature_name)
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
        "indicator_feature_cols": tuple(indicator_feature_cols),
        "volume_profile_feature_cols": tuple(volume_profile_feature_cols),
        "unclassified_feature_cols": tuple(unclassified_feature_cols),
    }


def summarize_feature_subset(subset_info):
    if subset_info is None:
        return {"enabled": False}

    return {
        "enabled": True,
        "path": str(subset_info["path"]),
        "count": int(subset_info["count"]),
        "format": subset_info["format"],
        "list_key": subset_info["list_key"],
        "created_utc": subset_info.get("created_utc"),
        "source_data_path": subset_info.get("source_data_path"),
    }
