import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

FEATURE_VERSION = "rp_fixed_grid_v1"
RP_FEATURE_PREFIX = "rp_"
STATE_DIR = Path("data/features/state/reaction_profile")
RUNTIME_STATE_DIR = STATE_DIR / "runtime"
MODELING_STATE_DIR = STATE_DIR / "modeling"
AUDIT_ANCHOR_STATE_DIR = STATE_DIR / "audit_anchor"
PSEUDO_LIVE_AUDIT_STATE_DIR = STATE_DIR / "pseudo_live_audit"
PSEUDO_LIVE_AUDIT_RUNTIME_STATE_DIR = PSEUDO_LIVE_AUDIT_STATE_DIR / "runtime"
PSEUDO_LIVE_AUDIT_MODELING_STATE_DIR = PSEUDO_LIVE_AUDIT_STATE_DIR / "modeling"

DEFAULT_CONFIG = {
    "enabled": True,
    "price_min": 0.0,
    "price_max": 2000000.0,
    "bin_size": 5.0,
    "neighbor_bins": 3.0,
    "eps": 1e-12,
    "min_reaction_strength": 0.0,
    "wick_power": 1.0,
    "distance_power": 1.0,
    "horizons": {
        "short": {"local_window": 64, "half_life_candles": 60},
        "medium": {"local_window": 64, "half_life_candles": 3600},
        "long": {"local_window": 64, "half_life_candles": 20000},
        "all": {"local_window": 64, "half_life_candles": None},
    },
}
_HORIZON_ORDER = ("short", "medium", "long", "all")
_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close")
_RENORMALIZE_SCALE_MIN = 1e-3
_RP_HORIZON_PATTERN = r"(?:short|medium|long|all)"
_RP_CANONICAL_FEATURE_RE = re.compile(
    rf"^rp_{_RP_HORIZON_PATTERN}_("
    r"support_below"
    r"|resistance_above"
    r"|support_above"
    r"|resistance_below"
    r"|imbalance"
    r"|support_center_dist"
    r"|resistance_center_dist"
    r")$"
)
_FEATURE_SUFFIXES = (
    "support_below",
    "resistance_above",
    "support_above",
    "resistance_below",
    "imbalance",
    "support_center_dist",
    "resistance_center_dist",
)
_ALLOWED_TOP_LEVEL_CONFIG_KEYS = {
    "enabled",
    "price_min",
    "price_max",
    "bin_size",
    "neighbor_bins",
    "eps",
    "min_reaction_strength",
    "wick_power",
    "distance_power",
    "horizons",
}
_ALLOWED_HORIZON_CONFIG_KEYS = {
    "local_window",
    "half_life_candles",
}


def is_reaction_profile_feature(feature_name):
    return bool(_RP_CANONICAL_FEATURE_RE.match(str(feature_name).strip()))


def validate_reaction_profile_feature_columns(feature_names, source_label):
    invalid_feature_cols = []
    for raw_feature_name in feature_names:
        feature_name = str(raw_feature_name).strip()
        if feature_name.startswith(RP_FEATURE_PREFIX) and not is_reaction_profile_feature(
                feature_name
        ):
            invalid_feature_cols.append(feature_name)

    if not invalid_feature_cols:
        return tuple(str(feature_name).strip() for feature_name in feature_names)

    preview = ", ".join(invalid_feature_cols[:10])
    raise ValueError(
        f"Unsupported reaction profile feature columns in {source_label}. "
        "Only the canonical RP naming schema produced by "
        "features.reaction_profile_fixed_grid.get_feature_columns(...) is supported. "
        "Regenerate any dataset, state, model, feature subset, or report artifact "
        "that still uses incompatible RP feature names. "
        f"Invalid_count={len(invalid_feature_cols)} preview=[{preview}]"
    )


def _normalize_positive_int(value, *, field_name):
    value_f = float(value)
    if not np.isfinite(value_f) or value_f <= 0.0:
        raise ValueError(f"reaction profile config requires {field_name} > 0.")
    if not value_f.is_integer():
        raise ValueError(
            f"reaction profile config requires integer-valued {field_name}."
        )
    return int(value_f)


def _normalize_positive_float(value, *, field_name):
    value_f = float(value)
    if not np.isfinite(value_f) or value_f <= 0.0:
        raise ValueError(f"reaction profile config requires {field_name} > 0.")
    return float(value_f)


def _normalize_non_negative_float(value, *, field_name):
    value_f = float(value)
    if not np.isfinite(value_f) or value_f < 0.0:
        raise ValueError(f"reaction profile config requires {field_name} >= 0.")
    return float(value_f)


def _normalize_horizon_config(horizon_name, cfg):
    local_window = _normalize_positive_int(
        cfg["local_window"],
        field_name=f"horizons.{horizon_name}.local_window",
    )

    half_life = cfg.get("half_life_candles")
    if horizon_name == "all":
        if half_life is not None:
            raise ValueError(
                "reaction profile horizon 'all' requires half_life_candles=null."
            )
        normalized_half_life = None
        decay = 1.0
    else:
        normalized_half_life = _normalize_positive_int(
            half_life,
            field_name=f"horizons.{horizon_name}.half_life_candles",
        )
        decay = math.exp(math.log(0.5) / float(normalized_half_life))

    return {
        "horizon_name": horizon_name,
        "local_window": int(local_window),
        "half_life_candles": normalized_half_life,
        "decay": float(decay),
    }


def normalize_config(raw_config=None):
    if (
            isinstance(raw_config, dict)
            and "horizon_names" in raw_config
            and "horizons" in raw_config
    ):
        raw_config = {
            "enabled": bool(raw_config.get("enabled", True)),
            "price_min": raw_config["price_min"],
            "price_max": raw_config["price_max"],
            "bin_size": raw_config["bin_size"],
            "neighbor_bins": raw_config["neighbor_bins"],
            "eps": raw_config["eps"],
            "min_reaction_strength": raw_config["min_reaction_strength"],
            "wick_power": raw_config["wick_power"],
            "distance_power": raw_config["distance_power"],
            "horizons": {
                horizon_name: {
                    "local_window": raw_config["horizons"][horizon_name][
                        "local_window"
                    ],
                    "half_life_candles": raw_config["horizons"][horizon_name][
                        "half_life_candles"
                    ],
                }
                for horizon_name in _HORIZON_ORDER
            },
        }

    user_cfg = dict(raw_config or {})
    unknown_top_level_keys = sorted(set(user_cfg) - _ALLOWED_TOP_LEVEL_CONFIG_KEYS)
    if unknown_top_level_keys:
        raise ValueError(
            "Unsupported reaction_profile_fixed_grid config keys: "
            f"{unknown_top_level_keys}"
        )

    user_horizons = user_cfg.pop("horizons", None)
    if user_horizons is not None and not isinstance(user_horizons, dict):
        raise ValueError("reaction profile config field 'horizons' must be a dict.")

    merged = {
        key: value for key, value in DEFAULT_CONFIG.items() if key != "horizons"
    }
    merged.update(user_cfg)

    price_min = float(merged["price_min"])
    price_max = float(merged["price_max"])
    if not np.isfinite(price_min) or not np.isfinite(price_max):
        raise ValueError(
            "reaction profile config requires finite price_min and price_max."
        )
    if price_max <= price_min:
        raise ValueError("reaction profile config requires price_max > price_min.")

    bin_size = _normalize_positive_float(merged["bin_size"], field_name="bin_size")
    bins = int(math.ceil((price_max - price_min) / bin_size)) + 1
    if bins <= 0:
        raise ValueError(
            "reaction profile config produced no bins. "
            "Check price_min, price_max, and bin_size."
        )

    neighbor_bins = _normalize_non_negative_float(
        merged["neighbor_bins"],
        field_name="neighbor_bins",
    )
    neighbor_radius = int(math.ceil(neighbor_bins))
    eps = _normalize_positive_float(merged["eps"], field_name="eps")
    min_reaction_strength = _normalize_non_negative_float(
        merged["min_reaction_strength"],
        field_name="min_reaction_strength",
    )
    wick_power = _normalize_positive_float(
        merged["wick_power"],
        field_name="wick_power",
    )
    distance_power = _normalize_positive_float(
        merged["distance_power"],
        field_name="distance_power",
    )

    horizon_overrides = user_horizons if isinstance(user_horizons, dict) else {}
    unknown_horizon_names = sorted(set(horizon_overrides) - set(_HORIZON_ORDER))
    if unknown_horizon_names:
        raise ValueError(
            "Unsupported reaction profile horizon names: "
            f"{unknown_horizon_names}. Expected one of {list(_HORIZON_ORDER)}"
        )

    horizons = {}
    horizon_names = []
    horizon_local_windows = []
    half_lives = []
    decays = []
    for horizon_name in _HORIZON_ORDER:
        horizon_cfg = dict(DEFAULT_CONFIG["horizons"][horizon_name])
        user_horizon_cfg = horizon_overrides.get(horizon_name)
        if user_horizon_cfg is not None:
            if not isinstance(user_horizon_cfg, dict):
                raise ValueError(
                    f"reaction profile horizon '{horizon_name}' config must be a dict."
                )
            unknown_horizon_keys = sorted(
                set(user_horizon_cfg) - _ALLOWED_HORIZON_CONFIG_KEYS
            )
            if unknown_horizon_keys:
                raise ValueError(
                    f"Unsupported reaction profile keys in horizons.{horizon_name}: "
                    f"{unknown_horizon_keys}"
                )
            horizon_cfg.update(user_horizon_cfg)

        normalized_horizon = _normalize_horizon_config(horizon_name, horizon_cfg)
        horizons[horizon_name] = normalized_horizon
        horizon_names.append(horizon_name)
        horizon_local_windows.append(int(normalized_horizon["local_window"]))
        half_lives.append(normalized_horizon["half_life_candles"])
        decays.append(float(normalized_horizon["decay"]))

    normalized = {
        "enabled": bool(merged.get("enabled", True)),
        "price_min": float(price_min),
        "price_max": float(price_max),
        "bin_size": float(bin_size),
        "bins": int(bins),
        "neighbor_bins": float(neighbor_bins),
        "neighbor_radius": int(neighbor_radius),
        "eps": float(eps),
        "min_reaction_strength": float(min_reaction_strength),
        "wick_power": float(wick_power),
        "distance_power": float(distance_power),
        "horizons": horizons,
        "horizon_names": tuple(horizon_names),
        "horizon_local_windows": tuple(horizon_local_windows),
        "half_lives": tuple(half_lives),
        "decays": tuple(decays),
        "version": FEATURE_VERSION,
    }
    normalized["feature_columns"] = get_feature_columns(normalized)
    normalized["config_signature"] = _config_signature_from_normalized(normalized)
    return normalized


def config_signature(cfg=None):
    normalized = cfg if cfg and "horizon_names" in cfg else normalize_config(cfg)
    return _config_signature_from_normalized(normalized)


def _config_signature_from_normalized(normalized):
    payload = {
        "version": normalized["version"],
        "enabled": bool(normalized["enabled"]),
        "price_min": float(normalized["price_min"]),
        "price_max": float(normalized["price_max"]),
        "bin_size": float(normalized["bin_size"]),
        "bins": int(normalized["bins"]),
        "neighbor_bins": float(normalized["neighbor_bins"]),
        "neighbor_radius": int(normalized["neighbor_radius"]),
        "eps": float(normalized["eps"]),
        "min_reaction_strength": float(normalized["min_reaction_strength"]),
        "wick_power": float(normalized["wick_power"]),
        "distance_power": float(normalized["distance_power"]),
        "horizons": {
            name: {
                "local_window": int(normalized["horizons"][name]["local_window"]),
                "half_life_candles": normalized["horizons"][name][
                    "half_life_candles"
                ],
                "decay": float(normalized["horizons"][name]["decay"]),
            }
            for name in normalized["horizon_names"]
        },
        "feature_columns": list(normalized["feature_columns"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def get_feature_columns(cfg=None):
    normalized = cfg if cfg and "horizon_names" in cfg else normalize_config(cfg)
    cols = []
    for horizon_name in normalized["horizon_names"]:
        for suffix in _FEATURE_SUFFIXES:
            cols.append(f"rp_{horizon_name}_{suffix}")
    return tuple(cols)


def _format_horizon_summary(source):
    return ", ".join(
        (
            f"{name}(bins={source['bins']},"
            f"window={source['horizons'][name]['local_window']},"
            f"hl={source['horizons'][name]['half_life_candles']})"
        )
        for name in source["horizon_names"]
    )


def create_empty_state(cfg=None):
    normalized = cfg if cfg and "horizon_names" in cfg else normalize_config(cfg)
    horizons = {}
    for horizon_name in normalized["horizon_names"]:
        horizon_cfg = normalized["horizons"][horizon_name]
        horizons[horizon_name] = {
            "horizon_name": horizon_name,
            "half_life_candles": horizon_cfg["half_life_candles"],
            "decay": float(horizon_cfg["decay"]),
            "bins": int(normalized["bins"]),
            "local_window": int(horizon_cfg["local_window"]),
            "support_profile": np.zeros(int(normalized["bins"]), dtype=np.float64),
            "resistance_profile": np.zeros(int(normalized["bins"]), dtype=np.float64),
            "global_scale": 1.0,
        }

    return {
        "enabled": bool(normalized["enabled"]),
        "version": normalized["version"],
        "price_min": float(normalized["price_min"]),
        "price_max": float(normalized["price_max"]),
        "bin_size": float(normalized["bin_size"]),
        "bins": int(normalized["bins"]),
        "neighbor_bins": float(normalized["neighbor_bins"]),
        "neighbor_radius": int(normalized["neighbor_radius"]),
        "eps": float(normalized["eps"]),
        "min_reaction_strength": float(normalized["min_reaction_strength"]),
        "wick_power": float(normalized["wick_power"]),
        "distance_power": float(normalized["distance_power"]),
        "horizon_names": tuple(normalized["horizon_names"]),
        "feature_columns": tuple(normalized["feature_columns"]),
        "config_signature": normalized["config_signature"],
        "horizons": horizons,
        "last_candle_time": None,
    }


def state_matches_config(state, cfg):
    return str(state.get("config_signature", "")) == config_signature(cfg)


def _state_base_path(path):
    base_path = Path(path)
    if base_path.suffix.lower() in {".npz", ".json"}:
        return base_path.with_suffix("")
    return base_path


def _state_metadata_dict(state):
    return {
        "version": str(state["version"]),
        "config_signature": str(state["config_signature"]),
        "enabled": bool(state["enabled"]),
        "price_min": float(state["price_min"]),
        "price_max": float(state["price_max"]),
        "bin_size": float(state["bin_size"]),
        "bins": int(state["bins"]),
        "neighbor_bins": float(state["neighbor_bins"]),
        "neighbor_radius": int(state["neighbor_radius"]),
        "eps": float(state["eps"]),
        "min_reaction_strength": float(state["min_reaction_strength"]),
        "wick_power": float(state["wick_power"]),
        "distance_power": float(state["distance_power"]),
        "horizon_names": list(state["horizon_names"]),
        "feature_columns": list(state["feature_columns"]),
        "horizons": {
            name: {
                "horizon_name": str(state["horizons"][name]["horizon_name"]),
                "half_life_candles": state["horizons"][name]["half_life_candles"],
                "decay": float(state["horizons"][name]["decay"]),
                "bins": int(state["horizons"][name]["bins"]),
                "local_window": int(state["horizons"][name]["local_window"]),
                "global_scale": float(state["horizons"][name]["global_scale"]),
            }
            for name in state["horizon_names"]
        },
        "last_candle_time": state.get("last_candle_time"),
    }


def _support_profile_npz_key(horizon_name):
    return f"support_profile_{horizon_name}"


def _resistance_profile_npz_key(horizon_name):
    return f"resistance_profile_{horizon_name}"


def save_state(state, base_path):
    base_path = _state_base_path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    npz_path = base_path.with_suffix(".npz")
    json_path = base_path.with_suffix(".json")
    arrays = {}
    for name in state["horizon_names"]:
        arrays[_support_profile_npz_key(name)] = np.asarray(
            state["horizons"][name]["support_profile"],
            dtype=np.float64,
        )
        arrays[_resistance_profile_npz_key(name)] = np.asarray(
            state["horizons"][name]["resistance_profile"],
            dtype=np.float64,
        )

    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(
        json.dumps(
            _state_metadata_dict(state),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return {"npz": npz_path, "json": json_path}


def load_state(base_path):
    base_path = _state_base_path(base_path)
    npz_path = base_path.with_suffix(".npz")
    json_path = base_path.with_suffix(".json")
    if not npz_path.exists():
        raise FileNotFoundError(f"reaction profile state npz not found: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"reaction profile state json not found: {json_path}")

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if str(meta.get("version", "")) != FEATURE_VERSION:
        raise ValueError(
            f"Unsupported reaction profile state version: {meta.get('version')!r}"
        )

    horizon_names = tuple(meta.get("horizon_names") or ())
    if horizon_names != _HORIZON_ORDER:
        raise ValueError(
            "reaction profile state horizon_names do not match the canonical order. "
            f"Expected={list(_HORIZON_ORDER)} actual={list(horizon_names)}"
        )

    cfg = {
        "enabled": bool(meta.get("enabled", True)),
        "price_min": meta["price_min"],
        "price_max": meta["price_max"],
        "bin_size": meta["bin_size"],
        "neighbor_bins": meta["neighbor_bins"],
        "eps": meta["eps"],
        "min_reaction_strength": meta["min_reaction_strength"],
        "wick_power": meta["wick_power"],
        "distance_power": meta["distance_power"],
        "horizons": {},
    }
    for horizon_name in horizon_names:
        horizon_meta = meta.get("horizons", {}).get(horizon_name)
        if not isinstance(horizon_meta, dict):
            raise ValueError(
                f"reaction profile state metadata is missing horizons.{horizon_name}"
            )
        cfg["horizons"][horizon_name] = {
            "local_window": horizon_meta["local_window"],
            "half_life_candles": horizon_meta["half_life_candles"],
        }

    state = create_empty_state(cfg)
    meta_feature_columns = tuple(meta.get("feature_columns") or ())
    validate_reaction_profile_feature_columns(
        meta_feature_columns,
        source_label=f"reaction profile state metadata {json_path}",
    )
    expected_feature_columns = tuple(state["feature_columns"])
    if meta_feature_columns != expected_feature_columns:
        raise ValueError(
            "reaction profile state metadata feature_columns do not match the current "
            "canonical RP schema. Regenerate the saved RP state artifact. "
            f"path={json_path} expected_count={len(expected_feature_columns)} "
            f"actual_count={len(meta_feature_columns)}"
        )

    actual_signature = str(meta.get("config_signature", ""))
    expected_signature = str(state["config_signature"])
    if actual_signature != expected_signature:
        raise ValueError(
            "reaction profile state config_signature does not match the current RP "
            "configuration. Regenerate the saved RP state artifact. "
            f"path={json_path}"
        )

    with np.load(npz_path) as data:
        for horizon_name in state["horizon_names"]:
            for array_key, state_key in (
                    (_support_profile_npz_key(horizon_name), "support_profile"),
                    (_resistance_profile_npz_key(horizon_name), "resistance_profile"),
            ):
                if array_key not in data:
                    raise ValueError(
                        f"reaction profile state npz is missing {array_key}: {npz_path}"
                    )
                profile = np.asarray(data[array_key], dtype=np.float64)
                expected_shape = state["horizons"][horizon_name][state_key].shape
                if profile.shape != expected_shape:
                    raise ValueError(
                        "reaction profile shape mismatch for "
                        f"{horizon_name}.{state_key}: {profile.shape} != {expected_shape}"
                    )
                state["horizons"][horizon_name][state_key][:] = profile

    for horizon_name in state["horizon_names"]:
        global_scale = float(meta["horizons"][horizon_name].get("global_scale", 1.0))
        if not np.isfinite(global_scale) or global_scale <= 0.0:
            raise ValueError(
                "reaction profile state requires finite positive global_scale for "
                f"horizon={horizon_name}"
            )
        state["horizons"][horizon_name]["global_scale"] = global_scale

    state["last_candle_time"] = meta.get("last_candle_time")
    return state


def validate_reaction_profile_model_metadata(
        meta,
        *,
        feature_columns,
        cfg,
        source_label,
):
    feature_columns = tuple(str(feature_name).strip() for feature_name in feature_columns)
    validate_reaction_profile_feature_columns(feature_columns, source_label)
    if not any(is_reaction_profile_feature(col) for col in feature_columns):
        return None

    raw_rp_cfg = meta.get("reaction_profile_fixed_grid")
    if not isinstance(raw_rp_cfg, dict):
        raise ValueError(
            f"{source_label} is missing reaction_profile_fixed_grid config metadata. "
            "Regenerate the modeling dataset and retrain the model for the current "
            f"RP architecture ({FEATURE_VERSION})."
        )

    expected_cfg = cfg if cfg and "horizon_names" in cfg else normalize_config(cfg)
    try:
        model_cfg = normalize_config(raw_rp_cfg)
    except Exception as exc:
        raise ValueError(
            f"{source_label} contains incompatible reaction_profile_fixed_grid config "
            "metadata. Regenerate the modeling dataset and retrain the model for "
            f"the current RP architecture ({FEATURE_VERSION})."
        ) from exc

    if str(model_cfg["config_signature"]) != str(expected_cfg["config_signature"]):
        raise ValueError(
            f"{source_label} reaction_profile_fixed_grid config does not match the "
            "active RP config. Regenerate the model or switch to the matching RP config."
        )
    return model_cfg


def validate_reaction_profile_dataset_metadata(
        metadata,
        *,
        feature_columns,
        cfg,
        source_label,
):
    feature_columns = tuple(str(feature_name).strip() for feature_name in feature_columns)
    validate_reaction_profile_feature_columns(feature_columns, source_label)
    if not any(is_reaction_profile_feature(col) for col in feature_columns):
        return None

    raw_rp_cfg = metadata.get("reaction_profile_fixed_grid")
    if not isinstance(raw_rp_cfg, dict):
        raise ValueError(
            f"{source_label} is missing reaction_profile_fixed_grid dataset metadata. "
            "Regenerate the modeling dataset for the current RP architecture."
        )

    expected_cfg = cfg if cfg and "horizon_names" in cfg else normalize_config(cfg)
    try:
        dataset_cfg = normalize_config(raw_rp_cfg)
    except Exception as exc:
        raise ValueError(
            f"{source_label} contains incompatible reaction_profile_fixed_grid dataset "
            "metadata. Regenerate the modeling dataset for the current RP architecture."
        ) from exc

    if str(dataset_cfg["config_signature"]) != str(expected_cfg["config_signature"]):
        raise ValueError(
            f"{source_label} reaction_profile_fixed_grid config does not match the "
            "active RP config. Regenerate the modeling dataset."
        )

    meta_feature_columns = tuple(
        metadata.get("reaction_profile_feature_columns") or ()
    )
    if meta_feature_columns and meta_feature_columns != tuple(
            expected_cfg["feature_columns"]
    ):
        raise ValueError(
            f"{source_label} reaction_profile_feature_columns do not match the active "
            "RP schema. Regenerate the modeling dataset."
        )
    return dataset_cfg


def _require_dataframe_columns(df, required):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"reaction profile dataframe missing required columns: {missing}"
        )


def _price_to_bin_index(price, price_min, price_max, bin_size, bins):
    price = float(price)
    if not np.isfinite(price) or price < price_min or price > price_max:
        return None
    idx = int(math.floor((price - price_min) / bin_size))
    if idx < 0 or idx >= bins:
        return None
    return idx


@njit(cache=True)
def _price_to_bin_index_numba(price, price_min, price_max, bin_size, bins):
    if not np.isfinite(price) or price < price_min or price > price_max:
        return -1
    idx = int(math.floor((price - price_min) / bin_size))
    if idx < 0 or idx >= bins:
        return -1
    return idx


@njit(cache=True)
def _fill_out_of_range_feature_row_numba(out_row, cursor):
    out_row[cursor] = 0.0
    out_row[cursor + 1] = 0.0
    out_row[cursor + 2] = 0.0
    out_row[cursor + 3] = 0.0
    out_row[cursor + 4] = 0.0
    out_row[cursor + 5] = np.nan
    out_row[cursor + 6] = np.nan


@njit(cache=True)
def _compute_reaction_strengths_numba(
        open_value,
        high,
        low,
        close,
        eps,
        min_reaction_strength,
        wick_power,
):
    if (
            not np.isfinite(open_value)
            or not np.isfinite(high)
            or not np.isfinite(low)
            or not np.isfinite(close)
    ):
        return 0.0, 0.0
    candle_range = high - low
    if candle_range <= eps:
        return 0.0, 0.0
    if low <= 0.0 or high <= 0.0 or close <= 0.0:
        return 0.0, 0.0

    body_low = open_value if open_value < close else close
    lower_wick = body_low - low
    if lower_wick < 0.0:
        lower_wick = 0.0
    lower_wick_ratio = lower_wick / (candle_range if candle_range > eps else eps)
    support_log = math.log(close / low)
    if support_log < 0.0:
        support_log = 0.0
    support_strength = (lower_wick_ratio ** wick_power) * support_log
    if support_strength < min_reaction_strength:
        support_strength = 0.0

    body_high = open_value if open_value > close else close
    upper_wick = high - body_high
    if upper_wick < 0.0:
        upper_wick = 0.0
    upper_wick_ratio = upper_wick / (candle_range if candle_range > eps else eps)
    resistance_log = math.log(high / close)
    if resistance_log < 0.0:
        resistance_log = 0.0
    resistance_strength = (upper_wick_ratio ** wick_power) * resistance_log
    if resistance_strength < min_reaction_strength:
        resistance_strength = 0.0

    return support_strength, resistance_strength


@njit(cache=True)
def _apply_event_kernel_numba(
        profile_buffer,
        offset,
        bins,
        center_bin,
        strength,
        neighbor_bins,
        neighbor_radius,
        eps,
        scale,
):
    if center_bin < 0 or strength <= 0.0 or not np.isfinite(strength):
        return

    for kernel_offset in range(-neighbor_radius, neighbor_radius + 1):
        target_bin = center_bin + kernel_offset
        if target_bin < 0 or target_bin >= bins:
            continue
        distance = abs(kernel_offset)
        if neighbor_bins <= eps:
            weight = 1.0 if distance == 0 else 0.0
        else:
            x = float(distance) / neighbor_bins
            weight = 1.0 - x
            if weight < 0.0:
                weight = 0.0
        if weight <= 0.0:
            continue
        profile_buffer[offset + target_bin] += (strength * weight) / scale


@njit(cache=True)
def _update_state_with_candle_numba(
        support_profile_buffer,
        resistance_profile_buffer,
        global_scales,
        open_value,
        high,
        low,
        close,
        price_min,
        price_max,
        bin_size,
        bins,
        neighbor_bins,
        neighbor_radius,
        eps,
        min_reaction_strength,
        wick_power,
        horizon_decays,
        renormalize_scale_min,
):
    support_strength, resistance_strength = _compute_reaction_strengths_numba(
        open_value=open_value,
        high=high,
        low=low,
        close=close,
        eps=eps,
        min_reaction_strength=min_reaction_strength,
        wick_power=wick_power,
    )
    support_bin = _price_to_bin_index_numba(low, price_min, price_max, bin_size, bins)
    resistance_bin = _price_to_bin_index_numba(
        high,
        price_min,
        price_max,
        bin_size,
        bins,
    )

    horizon_count = global_scales.shape[0]
    for horizon_idx in range(horizon_count):
        offset = horizon_idx * bins
        decay = horizon_decays[horizon_idx]
        if decay != 1.0:
            global_scales[horizon_idx] *= decay
            scale = global_scales[horizon_idx]
            if scale < renormalize_scale_min:
                for bin_idx in range(bins):
                    support_profile_buffer[offset + bin_idx] = (
                            support_profile_buffer[offset + bin_idx] * scale
                    )
                    resistance_profile_buffer[offset + bin_idx] = (
                            resistance_profile_buffer[offset + bin_idx] * scale
                    )
                global_scales[horizon_idx] = 1.0
                scale = 1.0
        else:
            scale = global_scales[horizon_idx]

        _apply_event_kernel_numba(
            support_profile_buffer,
            offset,
            bins,
            support_bin,
            support_strength,
            neighbor_bins,
            neighbor_radius,
            eps,
            scale,
        )
        _apply_event_kernel_numba(
            resistance_profile_buffer,
            offset,
            bins,
            resistance_bin,
            resistance_strength,
            neighbor_bins,
            neighbor_radius,
            eps,
            scale,
        )


@njit(cache=True)
def _extract_feature_row_array_numba(
        support_profile_buffer,
        resistance_profile_buffer,
        global_scales,
        close,
        price_min,
        price_max,
        bin_size,
        bins,
        distance_power,
        horizon_local_windows,
        out_row,
):
    current_bin = _price_to_bin_index_numba(
        close,
        price_min,
        price_max,
        bin_size,
        bins,
    )
    cursor = 0
    horizon_count = global_scales.shape[0]

    for horizon_idx in range(horizon_count):
        if current_bin < 0:
            _fill_out_of_range_feature_row_numba(out_row, cursor)
            cursor += 7
            continue

        offset = horizon_idx * bins
        local_window = horizon_local_windows[horizon_idx]
        scale = global_scales[horizon_idx]
        left = current_bin - local_window
        if left < 0:
            left = 0
        right = current_bin + local_window
        if right >= bins:
            right = bins - 1

        support_below = 0.0
        resistance_above = 0.0
        support_above = 0.0
        resistance_below = 0.0
        support_center_numer = 0.0
        resistance_center_numer = 0.0

        for bin_idx in range(left, right + 1):
            if bin_idx == current_bin:
                continue
            distance_bins = abs(bin_idx - current_bin)
            if distance_bins <= 0:
                continue
            distance_weight = 1.0 / ((1.0 + float(distance_bins)) ** distance_power)
            support_value = (
                    support_profile_buffer[offset + bin_idx]
                    * scale
                    * distance_weight
            )
            resistance_value = (
                    resistance_profile_buffer[offset + bin_idx]
                    * scale
                    * distance_weight
            )
            if bin_idx < current_bin:
                support_below += support_value
                resistance_below += resistance_value
                support_center_numer += float(distance_bins) * support_value
            else:
                support_above += support_value
                resistance_above += resistance_value
                resistance_center_numer += float(distance_bins) * resistance_value

        out_row[cursor] = support_below
        cursor += 1
        out_row[cursor] = resistance_above
        cursor += 1
        out_row[cursor] = support_above
        cursor += 1
        out_row[cursor] = resistance_below
        cursor += 1
        out_row[cursor] = (
                support_below + resistance_below - resistance_above - support_above
        )
        cursor += 1
        out_row[cursor] = (
            support_center_numer / support_below if support_below > 0.0 else np.nan
        )
        cursor += 1
        out_row[cursor] = (
            resistance_center_numer / resistance_above
            if resistance_above > 0.0
            else np.nan
        )
        cursor += 1


@njit(cache=True)
def _build_reaction_profile_feature_matrix_numba(
        open_values,
        high,
        low,
        close,
        keep_mask,
        out_rows,
        price_min,
        price_max,
        bin_size,
        bins,
        neighbor_bins,
        neighbor_radius,
        eps,
        min_reaction_strength,
        wick_power,
        distance_power,
        horizon_local_windows,
        horizon_decays,
        feature_count,
        renormalize_scale_min,
):
    horizon_count = horizon_decays.shape[0]
    total_bins = horizon_count * bins
    support_profile_buffer = np.zeros(total_bins, dtype=np.float64)
    resistance_profile_buffer = np.zeros(total_bins, dtype=np.float64)
    global_scales = np.ones(horizon_count, dtype=np.float64)
    feature_matrix = np.empty((out_rows, feature_count), dtype=np.float64)

    out_idx = 0
    row_count = close.shape[0]
    for row_idx in range(row_count):
        _update_state_with_candle_numba(
            support_profile_buffer=support_profile_buffer,
            resistance_profile_buffer=resistance_profile_buffer,
            global_scales=global_scales,
            open_value=open_values[row_idx],
            high=high[row_idx],
            low=low[row_idx],
            close=close[row_idx],
            price_min=price_min,
            price_max=price_max,
            bin_size=bin_size,
            bins=bins,
            neighbor_bins=neighbor_bins,
            neighbor_radius=neighbor_radius,
            eps=eps,
            min_reaction_strength=min_reaction_strength,
            wick_power=wick_power,
            horizon_decays=horizon_decays,
            renormalize_scale_min=renormalize_scale_min,
        )

        if keep_mask[row_idx]:
            _extract_feature_row_array_numba(
                support_profile_buffer=support_profile_buffer,
                resistance_profile_buffer=resistance_profile_buffer,
                global_scales=global_scales,
                close=close[row_idx],
                price_min=price_min,
                price_max=price_max,
                bin_size=bin_size,
                bins=bins,
                distance_power=distance_power,
                horizon_local_windows=horizon_local_windows,
                out_row=feature_matrix[out_idx],
            )
            out_idx += 1

    return feature_matrix, support_profile_buffer, resistance_profile_buffer, global_scales


def _normalize_keep_mask(keep_mask, row_count):
    if keep_mask is None:
        return np.ones(row_count, dtype=np.bool_)

    keep_mask_np = np.asarray(keep_mask, dtype=np.bool_)
    if keep_mask_np.ndim != 1 or keep_mask_np.shape[0] != row_count:
        raise ValueError(
            "reaction profile keep_mask must be a 1D boolean array with the same length as the inputs."
        )
    return np.ascontiguousarray(keep_mask_np)


def _normalized_horizon_arrays(normalized):
    return {
        "horizon_local_windows": np.ascontiguousarray(
            np.asarray(normalized["horizon_local_windows"], dtype=np.int64)
        ),
        "horizon_decays": np.ascontiguousarray(
            np.asarray(normalized["decays"], dtype=np.float64)
        ),
    }


def _apply_flat_profiles_to_state(
        state,
        support_profile_buffer,
        resistance_profile_buffer,
        global_scales,
):
    bins = int(state["bins"])
    for horizon_idx, horizon_name in enumerate(state["horizon_names"]):
        offset = horizon_idx * bins
        state["horizons"][horizon_name]["support_profile"][:] = (
            support_profile_buffer[offset: offset + bins]
        )
        state["horizons"][horizon_name]["resistance_profile"][:] = (
            resistance_profile_buffer[offset: offset + bins]
        )
        state["horizons"][horizon_name]["global_scale"] = float(
            global_scales[horizon_idx]
        )


def build_reaction_profile_feature_matrix_from_arrays(
        open_,
        high,
        low,
        close,
        cfg,
        keep_mask=None,
):
    normalized = normalize_config(cfg)
    state = create_empty_state(normalized)
    row_count = len(open_)
    if not state["enabled"]:
        return np.empty((0, 0), dtype=np.float64), state

    open_np = np.ascontiguousarray(np.asarray(open_, dtype=np.float64))
    high_np = np.ascontiguousarray(np.asarray(high, dtype=np.float64))
    low_np = np.ascontiguousarray(np.asarray(low, dtype=np.float64))
    close_np = np.ascontiguousarray(np.asarray(close, dtype=np.float64))
    if (
            high_np.shape[0] != row_count
            or low_np.shape[0] != row_count
            or close_np.shape[0] != row_count
    ):
        raise ValueError("reaction profile input arrays must have the same length.")

    keep_mask_np = _normalize_keep_mask(keep_mask, row_count=row_count)
    out_rows = int(keep_mask_np.sum())
    horizon_arrays = _normalized_horizon_arrays(normalized)

    (
        feature_matrix,
        support_profile_buffer,
        resistance_profile_buffer,
        global_scales,
    ) = _build_reaction_profile_feature_matrix_numba(
        open_values=open_np,
        high=high_np,
        low=low_np,
        close=close_np,
        keep_mask=keep_mask_np,
        out_rows=out_rows,
        price_min=float(state["price_min"]),
        price_max=float(state["price_max"]),
        bin_size=float(state["bin_size"]),
        bins=int(state["bins"]),
        neighbor_bins=float(state["neighbor_bins"]),
        neighbor_radius=int(state["neighbor_radius"]),
        eps=float(state["eps"]),
        min_reaction_strength=float(state["min_reaction_strength"]),
        wick_power=float(state["wick_power"]),
        distance_power=float(state["distance_power"]),
        horizon_local_windows=horizon_arrays["horizon_local_windows"],
        horizon_decays=horizon_arrays["horizon_decays"],
        feature_count=len(state["feature_columns"]),
        renormalize_scale_min=float(_RENORMALIZE_SCALE_MIN),
    )

    _apply_flat_profiles_to_state(
        state,
        support_profile_buffer=support_profile_buffer,
        resistance_profile_buffer=resistance_profile_buffer,
        global_scales=global_scales,
    )
    return feature_matrix, state


def _compute_reaction_strengths(
        open_value,
        high,
        low,
        close,
        eps,
        min_reaction_strength,
        wick_power,
):
    support_strength, resistance_strength = _compute_reaction_strengths_numba(
        float(open_value),
        float(high),
        float(low),
        float(close),
        float(eps),
        float(min_reaction_strength),
        float(wick_power),
    )
    return float(support_strength), float(resistance_strength)


def _apply_event_kernel(
        profile,
        center_bin,
        strength,
        *,
        neighbor_bins,
        neighbor_radius,
        eps,
        scale,
):
    if center_bin is None or strength <= 0.0 or not np.isfinite(strength):
        return
    for kernel_offset in range(-neighbor_radius, neighbor_radius + 1):
        target_bin = int(center_bin) + int(kernel_offset)
        if target_bin < 0 or target_bin >= profile.shape[0]:
            continue
        distance = abs(kernel_offset)
        if neighbor_bins <= eps:
            weight = 1.0 if distance == 0 else 0.0
        else:
            weight = max(1.0 - (float(distance) / float(neighbor_bins)), 0.0)
        if weight <= 0.0:
            continue
        profile[target_bin] += (float(strength) * weight) / float(scale)


def update_state_with_candle(state, open, high, low, close):
    if not state["enabled"]:
        return state

    support_strength, resistance_strength = _compute_reaction_strengths(
        open_value=float(open),
        high=float(high),
        low=float(low),
        close=float(close),
        eps=float(state["eps"]),
        min_reaction_strength=float(state["min_reaction_strength"]),
        wick_power=float(state["wick_power"]),
    )
    support_bin = _price_to_bin_index(
        float(low),
        float(state["price_min"]),
        float(state["price_max"]),
        float(state["bin_size"]),
        int(state["bins"]),
    )
    resistance_bin = _price_to_bin_index(
        float(high),
        float(state["price_min"]),
        float(state["price_max"]),
        float(state["bin_size"]),
        int(state["bins"]),
    )

    for horizon_name in state["horizon_names"]:
        horizon_state = state["horizons"][horizon_name]
        if float(horizon_state["decay"]) != 1.0:
            horizon_state["global_scale"] *= float(horizon_state["decay"])
            if horizon_state["global_scale"] < _RENORMALIZE_SCALE_MIN:
                horizon_state["support_profile"] *= horizon_state["global_scale"]
                horizon_state["resistance_profile"] *= horizon_state["global_scale"]
                horizon_state["global_scale"] = 1.0

        scale = float(horizon_state["global_scale"])
        _apply_event_kernel(
            horizon_state["support_profile"],
            support_bin,
            support_strength,
            neighbor_bins=float(state["neighbor_bins"]),
            neighbor_radius=int(state["neighbor_radius"]),
            eps=float(state["eps"]),
            scale=scale,
        )
        _apply_event_kernel(
            horizon_state["resistance_profile"],
            resistance_bin,
            resistance_strength,
            neighbor_bins=float(state["neighbor_bins"]),
            neighbor_radius=int(state["neighbor_radius"]),
            eps=float(state["eps"]),
            scale=scale,
        )

    return state


def _extract_feature_row_array(state, close):
    if not state["enabled"]:
        return np.empty(0, dtype=np.float64)

    out = np.empty(len(state["feature_columns"]), dtype=np.float64)
    cursor = 0
    current_bin = _price_to_bin_index(
        float(close),
        float(state["price_min"]),
        float(state["price_max"]),
        float(state["bin_size"]),
        int(state["bins"]),
    )

    for horizon_name in state["horizon_names"]:
        if current_bin is None:
            out[cursor: cursor + 5] = 0.0
            out[cursor + 5] = np.nan
            out[cursor + 6] = np.nan
            cursor += 7
            continue

        horizon_state = state["horizons"][horizon_name]
        bins = int(horizon_state["bins"])
        local_window = int(horizon_state["local_window"])
        scale = float(horizon_state["global_scale"])
        support_profile = horizon_state["support_profile"]
        resistance_profile = horizon_state["resistance_profile"]
        left = max(0, int(current_bin) - local_window)
        right = min(bins - 1, int(current_bin) + local_window)
        support_below = 0.0
        resistance_above = 0.0
        support_above = 0.0
        resistance_below = 0.0
        support_center_numer = 0.0
        resistance_center_numer = 0.0

        for bin_idx in range(left, right + 1):
            if bin_idx == int(current_bin):
                continue
            distance_bins = abs(bin_idx - int(current_bin))
            if distance_bins <= 0:
                continue
            distance_weight = 1.0 / (
                    (1.0 + float(distance_bins)) ** float(state["distance_power"])
            )
            support_value = (
                    float(support_profile[bin_idx]) * scale * distance_weight
            )
            resistance_value = (
                    float(resistance_profile[bin_idx]) * scale * distance_weight
            )
            if bin_idx < int(current_bin):
                support_below += support_value
                resistance_below += resistance_value
                support_center_numer += float(distance_bins) * support_value
            else:
                support_above += support_value
                resistance_above += resistance_value
                resistance_center_numer += float(distance_bins) * resistance_value

        out[cursor] = support_below
        cursor += 1
        out[cursor] = resistance_above
        cursor += 1
        out[cursor] = support_above
        cursor += 1
        out[cursor] = resistance_below
        cursor += 1
        out[cursor] = (
                support_below + resistance_below - resistance_above - support_above
        )
        cursor += 1
        out[cursor] = (
            support_center_numer / support_below if support_below > 0.0 else np.nan
        )
        cursor += 1
        out[cursor] = (
            resistance_center_numer / resistance_above
            if resistance_above > 0.0
            else np.nan
        )
        cursor += 1

    return out


def extract_features_from_state(state, close):
    values = _extract_feature_row_array(state, close=close)
    return {
        feature_col: float(values[idx])
        for idx, feature_col in enumerate(state["feature_columns"])
    }


def build_reaction_profile_features(df, cfg):
    _require_dataframe_columns(df, _REQUIRED_COLUMNS)
    normalized = normalize_config(cfg)
    state = create_empty_state(normalized)
    if not state["enabled"]:
        return pd.DataFrame(index=df.index), state

    print(
        "[rp] build start | "
        f"rows={len(df)} horizons={_format_horizon_summary(normalized)}"
    )
    feature_matrix, state = build_reaction_profile_feature_matrix_from_arrays(
        open_=df["Open"].to_numpy(dtype=np.float64, copy=False),
        high=df["High"].to_numpy(dtype=np.float64, copy=False),
        low=df["Low"].to_numpy(dtype=np.float64, copy=False),
        close=df["Close"].to_numpy(dtype=np.float64, copy=False),
        cfg=normalized,
    )

    if "Opened" in df.columns and len(df) > 0:
        state["last_candle_time"] = str(pd.Timestamp(df["Opened"].iloc[-1]).isoformat())

    feature_df = pd.DataFrame(
        feature_matrix, columns=state["feature_columns"], index=df.index
    )
    print("[rp] build columns: " + ", ".join(state["feature_columns"]))
    return feature_df, state


def bootstrap_state_from_history(df_hist, cfg=None):
    _require_dataframe_columns(df_hist, _REQUIRED_COLUMNS)
    normalized = normalize_config(cfg)
    state = create_empty_state(normalized)
    if not state["enabled"]:
        return state

    print(
        "[rp] bootstrap state start | "
        f"rows={len(df_hist)} horizons={_format_horizon_summary(normalized)}"
    )
    keep_mask = np.zeros(len(df_hist), dtype=np.bool_)
    _, state = build_reaction_profile_feature_matrix_from_arrays(
        open_=df_hist["Open"].to_numpy(dtype=np.float64, copy=False),
        high=df_hist["High"].to_numpy(dtype=np.float64, copy=False),
        low=df_hist["Low"].to_numpy(dtype=np.float64, copy=False),
        close=df_hist["Close"].to_numpy(dtype=np.float64, copy=False),
        cfg=normalized,
        keep_mask=keep_mask,
    )
    if "Opened" in df_hist.columns and len(df_hist) > 0:
        state["last_candle_time"] = str(
            pd.Timestamp(df_hist["Opened"].iloc[-1]).isoformat()
        )
    print("[rp] bootstrap state done")
    return state


def check_batch_live_consistency(df, cfg=None, atol=1e-9, rtol=1e-9):
    _require_dataframe_columns(df, _REQUIRED_COLUMNS)
    batch_df, _ = build_reaction_profile_features(df, cfg)
    live_state = create_empty_state(cfg)

    open_np = df["Open"].to_numpy(dtype=np.float64, copy=False)
    high_np = df["High"].to_numpy(dtype=np.float64, copy=False)
    low_np = df["Low"].to_numpy(dtype=np.float64, copy=False)
    close_np = df["Close"].to_numpy(dtype=np.float64, copy=False)

    live_matrix = np.empty_like(batch_df.to_numpy(dtype=np.float64, copy=True))
    for row_idx in range(len(df)):
        update_state_with_candle(
            live_state,
            open=float(open_np[row_idx]),
            high=float(high_np[row_idx]),
            low=float(low_np[row_idx]),
            close=float(close_np[row_idx]),
        )
        live_matrix[row_idx, :] = _extract_feature_row_array(
            live_state,
            close=float(close_np[row_idx]),
        )

    batch_matrix = batch_df.to_numpy(dtype=np.float64, copy=False)
    allclose = bool(
        np.allclose(
            batch_matrix,
            live_matrix,
            atol=float(atol),
            rtol=float(rtol),
            equal_nan=True,
        )
    )
    max_abs_diff = (
        float(np.nanmax(np.abs(batch_matrix - live_matrix)))
        if batch_matrix.size
        else 0.0
    )
    return {
        "ok": allclose,
        "max_abs_diff": max_abs_diff,
        "atol": float(atol),
        "rtol": float(rtol),
        "rows": len(df),
        "feature_count": int(batch_matrix.shape[1]) if batch_matrix.ndim == 2 else 0,
    }


__all__ = [
    "AUDIT_ANCHOR_STATE_DIR",
    "FEATURE_VERSION",
    "MODELING_STATE_DIR",
    "PSEUDO_LIVE_AUDIT_MODELING_STATE_DIR",
    "PSEUDO_LIVE_AUDIT_RUNTIME_STATE_DIR",
    "PSEUDO_LIVE_AUDIT_STATE_DIR",
    "RP_FEATURE_PREFIX",
    "RUNTIME_STATE_DIR",
    "STATE_DIR",
    "bootstrap_state_from_history",
    "build_reaction_profile_feature_matrix_from_arrays",
    "build_reaction_profile_features",
    "check_batch_live_consistency",
    "config_signature",
    "create_empty_state",
    "extract_features_from_state",
    "get_feature_columns",
    "is_reaction_profile_feature",
    "load_state",
    "normalize_config",
    "save_state",
    "state_matches_config",
    "update_state_with_candle",
    "validate_reaction_profile_dataset_metadata",
    "validate_reaction_profile_feature_columns",
    "validate_reaction_profile_model_metadata",
]
