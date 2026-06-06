import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

FEATURE_VERSION = "vp_fixed_range_v3"
VP_FEATURE_PREFIX = "vp_"
STATE_DIR = Path("data/features/state/volume_profile")
RUNTIME_STATE_DIR = STATE_DIR / "runtime"
MODELING_STATE_DIR = STATE_DIR / "modeling"
AUDIT_ANCHOR_STATE_DIR = STATE_DIR / "audit_anchor"
PSEUDO_LIVE_AUDIT_STATE_DIR = STATE_DIR / "pseudo_live_audit"
PSEUDO_LIVE_AUDIT_RUNTIME_STATE_DIR = PSEUDO_LIVE_AUDIT_STATE_DIR / "runtime"
PSEUDO_LIVE_AUDIT_MODELING_STATE_DIR = PSEUDO_LIVE_AUDIT_STATE_DIR / "modeling"

_DEFAULT_HORIZON_BASE = {
    "step": 5,
    "local_window": 2,
    "sigma_divisor": 4.0,
    "min_sigma": 2.5,
}

DEFAULT_CONFIG = {
    "enabled": True,
    "price_min": 1.0,
    "price_max": 200000.0,
    "neighbor_bins": 2,
    "eps": 1e-6,
    "horizons": {
        "short": {**_DEFAULT_HORIZON_BASE, "half_life_candles": 1440},
        "medium": {**_DEFAULT_HORIZON_BASE, "half_life_candles": 10080},
        "long": {**_DEFAULT_HORIZON_BASE, "half_life_candles": 43200},
        "all": {**_DEFAULT_HORIZON_BASE, "half_life_candles": None},
    },
}
_HORIZON_ORDER = ("short", "medium", "long", "all")
_REQUIRED_COLUMNS = ("High", "Low", "Volume")
_RENORMALIZE_SCALE_MIN = 1e-3
_SQRT_TWO = math.sqrt(2.0)
_VP_HORIZON_PATTERN = r"(?:short|medium|long|all)"
_VP_CANONICAL_FEATURE_RE = re.compile(
    rf"^vp_{_VP_HORIZON_PATTERN}_("
    r"log_density_ratio_to_current_bin_(?:minus|plus)_[1-9]\d*"
    r"|local_above_below_volume_log_ratio"
    r"|current_bin_volume_share_of_local_window"
    r"|current_bin_volume_share_of_local_peak"
    r")$"
)
_ALLOWED_TOP_LEVEL_CONFIG_KEYS = {
    "enabled",
    "price_min",
    "price_max",
    "neighbor_bins",
    "eps",
    "horizons",
}
_LEGACY_GLOBAL_ONLY_KEYS = {
    "step",
    "bins",
    "local_window",
    "sigma_divisor",
    "min_sigma",
    "half_lives",
    "decays",
}
_ALLOWED_HORIZON_CONFIG_KEYS = {
    "step",
    "local_window",
    "sigma_divisor",
    "min_sigma",
    "half_life_candles",
}


def is_volume_profile_feature(feature_name):
    return bool(_VP_CANONICAL_FEATURE_RE.match(str(feature_name).strip()))


def validate_volume_profile_feature_columns(feature_names, *, source_label):
    invalid_feature_cols = []
    for raw_feature_name in feature_names:
        feature_name = str(raw_feature_name).strip()
        if feature_name.startswith(VP_FEATURE_PREFIX) and not is_volume_profile_feature(
                feature_name
        ):
            invalid_feature_cols.append(feature_name)

    if not invalid_feature_cols:
        return tuple(str(feature_name).strip() for feature_name in feature_names)

    preview = ", ".join(invalid_feature_cols[:10])
    raise ValueError(
        f"Unsupported volume profile feature columns in {source_label}. "
        "Only the canonical VP naming schema produced by "
        "features.volume_profile_fixed_range.get_feature_columns(...) is supported. "
        "Regenerate any dataset, state, model, feature subset, or report artifact "
        "that still uses incompatible VP feature names. "
        f"Invalid_count={len(invalid_feature_cols)} preview=[{preview}]"
    )


def _normalize_positive_int(value, *, field_name):
    value_f = float(value)
    if not np.isfinite(value_f) or value_f <= 0.0:
        raise ValueError(f"volume profile config requires {field_name} > 0.")
    if not value_f.is_integer():
        raise ValueError(
            f"volume profile config requires integer-valued {field_name}."
        )
    return int(value_f)


def _normalize_positive_float(value, *, field_name):
    value_f = float(value)
    if not np.isfinite(value_f) or value_f <= 0.0:
        raise ValueError(f"volume profile config requires {field_name} > 0.")
    return value_f


def _normalize_horizon_config(horizon_name, cfg, *, price_min, price_max, offset):
    step = _normalize_positive_int(
        cfg["step"],
        field_name=f"horizons.{horizon_name}.step",
    )
    bins = int(math.ceil((price_max - price_min) / float(step)))
    if bins <= 0:
        raise ValueError(
            f"volume profile horizon '{horizon_name}' produced no bins. "
            "Check price_min, price_max, and step."
        )

    local_window = _normalize_positive_int(
        cfg["local_window"],
        field_name=f"horizons.{horizon_name}.local_window",
    )
    sigma_divisor = _normalize_positive_float(
        cfg["sigma_divisor"],
        field_name=f"horizons.{horizon_name}.sigma_divisor",
    )
    min_sigma = _normalize_positive_float(
        cfg["min_sigma"],
        field_name=f"horizons.{horizon_name}.min_sigma",
    )

    half_life = cfg.get("half_life_candles")
    if half_life is None:
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
        "step": int(step),
        "bins": int(bins),
        "offset": int(offset),
        "local_window": int(local_window),
        "sigma_divisor": float(sigma_divisor),
        "min_sigma": float(min_sigma),
        "half_life_candles": normalized_half_life,
        "decay": float(decay),
    }


def normalize_config(cfg=None):
    if isinstance(cfg, dict) and "horizon_names" in cfg and "horizons" in cfg:
        cfg = {
            "enabled": bool(cfg.get("enabled", True)),
            "price_min": cfg["price_min"],
            "price_max": cfg["price_max"],
            "neighbor_bins": cfg["neighbor_bins"],
            "eps": cfg["eps"],
            "horizons": {
                horizon_name: {
                    "step": cfg["horizons"][horizon_name]["step"],
                    "local_window": cfg["horizons"][horizon_name]["local_window"],
                    "sigma_divisor": cfg["horizons"][horizon_name]["sigma_divisor"],
                    "min_sigma": cfg["horizons"][horizon_name]["min_sigma"],
                    "half_life_candles": cfg["horizons"][horizon_name][
                        "half_life_candles"
                    ],
                }
                for horizon_name in _HORIZON_ORDER
            },
        }

    user_cfg = dict(cfg or {})

    legacy_global_keys = sorted(set(user_cfg) & _LEGACY_GLOBAL_ONLY_KEYS)
    if legacy_global_keys:
        raise ValueError(
            "volume_profile_fixed_range no longer supports global-only VP parameters. "
            "Move step/local_window/sigma_divisor/min_sigma into per-horizon config. "
            f"Found legacy keys={legacy_global_keys}"
        )

    unknown_top_level_keys = sorted(set(user_cfg) - _ALLOWED_TOP_LEVEL_CONFIG_KEYS)
    if unknown_top_level_keys:
        raise ValueError(
            "Unsupported volume_profile_fixed_range config keys: "
            f"{unknown_top_level_keys}"
        )

    user_horizons = user_cfg.pop("horizons", None)
    if user_horizons is not None and not isinstance(user_horizons, dict):
        raise ValueError("volume profile config field 'horizons' must be a dict.")

    merged = dict(DEFAULT_CONFIG)
    merged.update(user_cfg)

    price_min = float(merged["price_min"])
    price_max = float(merged["price_max"])
    if not np.isfinite(price_min) or not np.isfinite(price_max):
        raise ValueError(
            "volume profile config requires finite price_min and price_max."
        )
    if price_max <= price_min:
        raise ValueError("volume profile config requires price_max > price_min.")

    neighbor_bins = _normalize_positive_int(
        merged["neighbor_bins"],
        field_name="neighbor_bins",
    )
    eps = _normalize_positive_float(merged["eps"], field_name="eps")

    horizon_overrides = user_horizons if isinstance(user_horizons, dict) else {}
    unknown_horizon_names = sorted(set(horizon_overrides) - set(_HORIZON_ORDER))
    if unknown_horizon_names:
        raise ValueError(
            "Unsupported volume profile horizon names: "
            f"{unknown_horizon_names}. Expected one of {list(_HORIZON_ORDER)}"
        )

    horizons = {}
    horizon_names = []
    horizon_offsets = []
    horizon_bins = []
    horizon_steps = []
    horizon_local_windows = []
    horizon_sigma_divisors = []
    horizon_min_sigmas = []
    half_lives = []
    decays = []
    total_bins = 0

    for horizon_name in _HORIZON_ORDER:
        horizon_cfg = dict(DEFAULT_CONFIG["horizons"][horizon_name])
        user_horizon_cfg = horizon_overrides.get(horizon_name)
        if user_horizon_cfg is not None:
            if not isinstance(user_horizon_cfg, dict):
                raise ValueError(
                    f"volume profile horizon '{horizon_name}' config must be a dict."
                )
            unknown_horizon_keys = sorted(
                set(user_horizon_cfg) - _ALLOWED_HORIZON_CONFIG_KEYS
            )
            if unknown_horizon_keys:
                raise ValueError(
                    f"Unsupported volume profile keys in horizons.{horizon_name}: "
                    f"{unknown_horizon_keys}"
                )
            horizon_cfg.update(user_horizon_cfg)

        normalized_horizon = _normalize_horizon_config(
            horizon_name,
            horizon_cfg,
            price_min=price_min,
            price_max=price_max,
            offset=total_bins,
        )
        horizons[horizon_name] = normalized_horizon
        horizon_names.append(horizon_name)
        horizon_offsets.append(int(normalized_horizon["offset"]))
        horizon_bins.append(int(normalized_horizon["bins"]))
        horizon_steps.append(int(normalized_horizon["step"]))
        horizon_local_windows.append(int(normalized_horizon["local_window"]))
        horizon_sigma_divisors.append(float(normalized_horizon["sigma_divisor"]))
        horizon_min_sigmas.append(float(normalized_horizon["min_sigma"]))
        half_lives.append(normalized_horizon["half_life_candles"])
        decays.append(float(normalized_horizon["decay"]))
        total_bins += int(normalized_horizon["bins"])

    normalized = {
        "enabled": bool(merged.get("enabled", True)),
        "price_min": price_min,
        "price_max": price_max,
        "neighbor_bins": neighbor_bins,
        "eps": eps,
        "horizons": horizons,
        "horizon_names": tuple(horizon_names),
        "horizon_offsets": tuple(horizon_offsets),
        "horizon_bins": tuple(horizon_bins),
        "horizon_steps": tuple(horizon_steps),
        "horizon_local_windows": tuple(horizon_local_windows),
        "horizon_sigma_divisors": tuple(horizon_sigma_divisors),
        "horizon_min_sigmas": tuple(horizon_min_sigmas),
        "half_lives": tuple(half_lives),
        "decays": tuple(decays),
        "total_bins": int(total_bins),
        "max_bins": int(max(horizon_bins)),
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
        "neighbor_bins": int(normalized["neighbor_bins"]),
        "eps": float(normalized["eps"]),
        "horizons": {
            name: {
                "step": int(normalized["horizons"][name]["step"]),
                "bins": int(normalized["horizons"][name]["bins"]),
                "local_window": int(normalized["horizons"][name]["local_window"]),
                "sigma_divisor": float(
                    normalized["horizons"][name]["sigma_divisor"]
                ),
                "min_sigma": float(normalized["horizons"][name]["min_sigma"]),
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
    neighbor_bins = int(normalized["neighbor_bins"])
    for horizon_name in normalized["horizon_names"]:
        for shift in range(-neighbor_bins, 0):
            cols.append(
                f"vp_{horizon_name}_log_density_ratio_to_current_bin_minus_{abs(shift)}"
            )
        for shift in range(1, neighbor_bins + 1):
            cols.append(
                f"vp_{horizon_name}_log_density_ratio_to_current_bin_plus_{shift}"
            )
        cols.append(f"vp_{horizon_name}_local_above_below_volume_log_ratio")
        cols.append(f"vp_{horizon_name}_current_bin_volume_share_of_local_window")
        cols.append(f"vp_{horizon_name}_current_bin_volume_share_of_local_peak")
    return tuple(cols)


def _format_horizon_summary(source):
    return ", ".join(
        (
            f"{name}(step={source['horizons'][name]['step']},"
            f"bins={source['horizons'][name]['bins']},"
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
            "step": int(horizon_cfg["step"]),
            "bins": int(horizon_cfg["bins"]),
            "local_window": int(horizon_cfg["local_window"]),
            "sigma_divisor": float(horizon_cfg["sigma_divisor"]),
            "min_sigma": float(horizon_cfg["min_sigma"]),
            # Each horizon keeps an independent grid/profile for batch/live parity.
            "raw_profile": np.zeros(int(horizon_cfg["bins"]), dtype=np.float64),
            "global_scale": 1.0,
        }

    return {
        "enabled": bool(normalized["enabled"]),
        "version": normalized["version"],
        "price_min": float(normalized["price_min"]),
        "price_max": float(normalized["price_max"]),
        "neighbor_bins": int(normalized["neighbor_bins"]),
        "eps": float(normalized["eps"]),
        "horizon_names": tuple(normalized["horizon_names"]),
        "feature_columns": tuple(normalized["feature_columns"]),
        "config_signature": normalized["config_signature"],
        "horizons": horizons,
        "last_candle_time": None,
    }


def state_matches_config(state, cfg=None):
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
        "neighbor_bins": int(state["neighbor_bins"]),
        "eps": float(state["eps"]),
        "horizon_names": list(state["horizon_names"]),
        "feature_columns": list(state["feature_columns"]),
        "horizons": {
            name: {
                "horizon_name": str(state["horizons"][name]["horizon_name"]),
                "half_life_candles": state["horizons"][name]["half_life_candles"],
                "decay": float(state["horizons"][name]["decay"]),
                "step": int(state["horizons"][name]["step"]),
                "bins": int(state["horizons"][name]["bins"]),
                "local_window": int(state["horizons"][name]["local_window"]),
                "sigma_divisor": float(state["horizons"][name]["sigma_divisor"]),
                "min_sigma": float(state["horizons"][name]["min_sigma"]),
                "global_scale": float(state["horizons"][name]["global_scale"]),
            }
            for name in state["horizon_names"]
        },
        "last_candle_time": state.get("last_candle_time"),
    }


def _raw_profile_npz_key(horizon_name):
    return f"raw_profile_{horizon_name}"


def save_state(state, path):
    base_path = _state_base_path(path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    npz_path = base_path.with_suffix(".npz")
    json_path = base_path.with_suffix(".json")

    np.savez_compressed(
        npz_path,
        **{
            _raw_profile_npz_key(name): np.asarray(
                state["horizons"][name]["raw_profile"],
                dtype=np.float64,
            )
            for name in state["horizon_names"]
        },
    )
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


def load_state(path):
    base_path = _state_base_path(path)
    npz_path = base_path.with_suffix(".npz")
    json_path = base_path.with_suffix(".json")
    if not npz_path.exists():
        raise FileNotFoundError(f"volume profile state npz not found: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"volume profile state json not found: {json_path}")

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if str(meta.get("version", "")) != FEATURE_VERSION:
        raise ValueError(
            f"Unsupported volume profile state version: {meta.get('version')!r}"
        )

    horizon_names = tuple(meta.get("horizon_names") or ())
    if horizon_names != _HORIZON_ORDER:
        raise ValueError(
            "volume profile state horizon_names do not match the canonical order. "
            f"Expected={list(_HORIZON_ORDER)} actual={list(horizon_names)}"
        )

    cfg = {
        "enabled": bool(meta.get("enabled", True)),
        "price_min": meta["price_min"],
        "price_max": meta["price_max"],
        "neighbor_bins": meta["neighbor_bins"],
        "eps": meta["eps"],
        "horizons": {},
    }
    for horizon_name in horizon_names:
        horizon_meta = meta.get("horizons", {}).get(horizon_name)
        if not isinstance(horizon_meta, dict):
            raise ValueError(
                f"volume profile state metadata is missing horizons.{horizon_name}"
            )
        cfg["horizons"][horizon_name] = {
            "step": horizon_meta["step"],
            "local_window": horizon_meta["local_window"],
            "sigma_divisor": horizon_meta["sigma_divisor"],
            "min_sigma": horizon_meta["min_sigma"],
            "half_life_candles": horizon_meta["half_life_candles"],
        }

    state = create_empty_state(cfg)
    meta_feature_columns = tuple(meta.get("feature_columns") or ())
    validate_volume_profile_feature_columns(
        meta_feature_columns,
        source_label=f"volume profile state metadata {json_path}",
    )
    expected_feature_columns = tuple(state["feature_columns"])
    if meta_feature_columns != expected_feature_columns:
        raise ValueError(
            "volume profile state metadata feature_columns do not match the current "
            "canonical VP schema. Regenerate the saved VP state artifact. "
            f"path={json_path} expected_count={len(expected_feature_columns)} "
            f"actual_count={len(meta_feature_columns)}"
        )

    actual_signature = str(meta.get("config_signature", ""))
    expected_signature = str(state["config_signature"])
    if actual_signature != expected_signature:
        raise ValueError(
            "volume profile state config_signature does not match the current VP "
            "configuration. Regenerate the saved VP state artifact. "
            f"path={json_path}"
        )

    with np.load(npz_path) as data:
        for horizon_name in state["horizon_names"]:
            array_key = _raw_profile_npz_key(horizon_name)
            if array_key not in data:
                raise ValueError(
                    f"volume profile state npz is missing {array_key}: {npz_path}"
                )
            raw_profile = np.asarray(data[array_key], dtype=np.float64)
            expected_shape = state["horizons"][horizon_name]["raw_profile"].shape
            if raw_profile.shape != expected_shape:
                raise ValueError(
                    "volume profile raw_profile shape mismatch for "
                    f"{horizon_name}: {raw_profile.shape} != {expected_shape}"
                )
            state["horizons"][horizon_name]["raw_profile"][:] = raw_profile

    for horizon_name in state["horizon_names"]:
        global_scale = float(meta["horizons"][horizon_name].get("global_scale", 1.0))
        if not np.isfinite(global_scale) or global_scale <= 0.0:
            raise ValueError(
                "volume profile state requires finite positive global_scale for "
                f"horizon={horizon_name}"
            )
        state["horizons"][horizon_name]["global_scale"] = global_scale

    state["last_candle_time"] = meta.get("last_candle_time")
    return state


def validate_volume_profile_model_metadata(
        metadata_payload,
        *,
        feature_columns,
        cfg=None,
        source_label,
):
    feature_columns = tuple(str(feature_name).strip() for feature_name in feature_columns)
    validate_volume_profile_feature_columns(
        feature_columns,
        source_label=source_label,
    )
    if not any(is_volume_profile_feature(col) for col in feature_columns):
        return None

    raw_vp_cfg = metadata_payload.get("volume_profile_fixed_range")
    if not isinstance(raw_vp_cfg, dict):
        raise ValueError(
            f"{source_label} is missing volume_profile_fixed_range config metadata. "
            "Regenerate the modeling dataset and retrain the model for the current "
            f"VP architecture ({FEATURE_VERSION})."
        )

    expected_cfg = cfg if cfg and "horizon_names" in cfg else normalize_config(cfg)
    try:
        model_cfg = normalize_config(raw_vp_cfg)
    except Exception as exc:
        raise ValueError(
            f"{source_label} contains incompatible volume_profile_fixed_range config "
            "metadata. Regenerate the modeling dataset and retrain the model for "
            f"the current VP architecture ({FEATURE_VERSION})."
        ) from exc

    if str(model_cfg["config_signature"]) != str(expected_cfg["config_signature"]):
        raise ValueError(
            f"{source_label} volume_profile_fixed_range config does not match the "
            "active VP config. Regenerate the model or switch to the matching VP config."
        )
    return model_cfg


def validate_volume_profile_dataset_metadata(
        metadata_payload,
        *,
        feature_columns,
        cfg=None,
        source_label,
):
    feature_columns = tuple(str(feature_name).strip() for feature_name in feature_columns)
    validate_volume_profile_feature_columns(
        feature_columns,
        source_label=source_label,
    )
    if not any(is_volume_profile_feature(col) for col in feature_columns):
        return None

    raw_vp_cfg = metadata_payload.get("volume_profile_fixed_range")
    if not isinstance(raw_vp_cfg, dict):
        raise ValueError(
            f"{source_label} is missing volume_profile_fixed_range dataset metadata. "
            "Regenerate the modeling dataset for the current VP architecture."
        )

    expected_cfg = cfg if cfg and "horizon_names" in cfg else normalize_config(cfg)
    try:
        dataset_cfg = normalize_config(raw_vp_cfg)
    except Exception as exc:
        raise ValueError(
            f"{source_label} contains incompatible volume_profile_fixed_range dataset "
            "metadata. Regenerate the modeling dataset for the current VP architecture."
        ) from exc

    if str(dataset_cfg["config_signature"]) != str(expected_cfg["config_signature"]):
        raise ValueError(
            f"{source_label} volume_profile_fixed_range config does not match the "
            "active VP config. Regenerate the modeling dataset."
        )

    meta_feature_columns = tuple(
        metadata_payload.get("volume_profile_feature_columns") or ()
    )
    if meta_feature_columns and meta_feature_columns != tuple(
            expected_cfg["feature_columns"]
    ):
        raise ValueError(
            f"{source_label} volume_profile_feature_columns do not match the active "
            "VP schema. Regenerate the modeling dataset."
        )
    return dataset_cfg


def _require_dataframe_columns(df, required):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"volume profile dataframe missing required columns: {missing}"
        )


def _price_to_bin_index(price, price_min, step, bins):
    idx = int((float(price) - price_min) / step)
    if idx < 0:
        return 0
    if idx >= bins:
        return bins - 1
    return idx


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(float(x) / _SQRT_TWO))


@njit(cache=True)
def _price_to_bin_index_numba(price, price_min, step, bins):
    idx = int((price - price_min) / step)
    if idx < 0:
        return 0
    if idx >= bins:
        return bins - 1
    return idx


@njit(cache=True)
def _normal_cdf_numba(x):
    return 0.5 * (1.0 + math.erf(x / _SQRT_TWO))


@njit(cache=True)
def _extract_feature_row_array_numba(
        raw_profile_buffer,
        global_scales,
        high,
        low,
        price_min,
        neighbor_bins,
        eps,
        horizon_offsets,
        horizon_bins,
        horizon_steps,
        horizon_local_windows,
        out_row,
):
    price_ref = 0.5 * (high + low)
    cursor = 0
    horizon_count = global_scales.shape[0]

    for horizon_idx in range(horizon_count):
        offset = horizon_offsets[horizon_idx]
        bins = horizon_bins[horizon_idx]
        step = horizon_steps[horizon_idx]
        local_window = horizon_local_windows[horizon_idx]
        bin_idx = _price_to_bin_index_numba(price_ref, price_min, step, bins)
        scale = global_scales[horizon_idx]
        curr = float(raw_profile_buffer[offset + bin_idx]) * scale

        for shift in range(-neighbor_bins, 0):
            neighbor_idx = bin_idx + shift
            if neighbor_idx < 0:
                neighbor_idx = 0
            elif neighbor_idx >= bins:
                neighbor_idx = bins - 1
            neigh = float(raw_profile_buffer[offset + neighbor_idx]) * scale
            out_row[cursor] = math.log((neigh + eps) / (curr + eps))
            cursor += 1

        for shift in range(1, neighbor_bins + 1):
            neighbor_idx = bin_idx + shift
            if neighbor_idx < 0:
                neighbor_idx = 0
            elif neighbor_idx >= bins:
                neighbor_idx = bins - 1
            neigh = float(raw_profile_buffer[offset + neighbor_idx]) * scale
            out_row[cursor] = math.log((neigh + eps) / (curr + eps))
            cursor += 1

        left = bin_idx - local_window
        if left < 0:
            left = 0
        right = bin_idx + local_window + 1
        if right > bins:
            right = bins

        above = 0.0
        below = 0.0
        local_sum = 0.0
        local_max = 0.0
        for local_idx in range(left, right):
            value = float(raw_profile_buffer[offset + local_idx]) * scale
            local_sum += value
            if value > local_max:
                local_max = value
            if local_idx < bin_idx:
                below += value
            elif local_idx > bin_idx:
                above += value

        out_row[cursor] = math.log((above + eps) / (below + eps))
        cursor += 1
        out_row[cursor] = curr / (local_sum + eps)
        cursor += 1
        out_row[cursor] = curr / (local_max + eps)
        cursor += 1


@njit(cache=True)
def _update_state_with_candle_numba(
        raw_profile_buffer,
        global_scales,
        horizon_offsets,
        horizon_bins,
        horizon_steps,
        horizon_sigma_divisors,
        horizon_min_sigmas,
        horizon_decays,
        high,
        low,
        volume,
        price_min,
        renormalize_scale_min,
        weight_buffer,
):
    if not np.isfinite(high) or not np.isfinite(low) or not np.isfinite(volume):
        return
    if volume == 0.0:
        return

    hl2 = 0.5 * (high + low)
    horizon_count = global_scales.shape[0]

    for horizon_idx in range(horizon_count):
        offset = horizon_offsets[horizon_idx]
        bins = horizon_bins[horizon_idx]
        step = horizon_steps[horizon_idx]
        sigma_divisor = horizon_sigma_divisors[horizon_idx]
        min_sigma = horizon_min_sigmas[horizon_idx]

        start_idx = 0
        length = 1
        total_weight = 1.0

        if high <= low or abs(high - low) <= 1e-12:
            start_idx = _price_to_bin_index_numba(hl2, price_min, step, bins)
            weight_buffer[0] = 1.0
        else:
            start_idx = _price_to_bin_index_numba(low, price_min, step, bins)
            end_idx = _price_to_bin_index_numba(high, price_min, step, bins)
            length = end_idx - start_idx + 1
            mu = hl2
            sigma = (high - low) / sigma_divisor
            if sigma < min_sigma:
                sigma = min_sigma

            total_weight = 0.0
            for weight_idx in range(length):
                bin_idx = start_idx + weight_idx
                bin_left = price_min + float(bin_idx) * step
                bin_right = bin_left + step
                left = low if low > bin_left else bin_left
                right = high if high < bin_right else bin_right
                if right <= left:
                    weight_buffer[weight_idx] = 0.0
                    continue
                weight = _normal_cdf_numba((right - mu) / sigma) - _normal_cdf_numba(
                    (left - mu) / sigma
                )
                if weight <= 0.0 or not np.isfinite(weight):
                    weight_buffer[weight_idx] = 0.0
                    continue
                weight_buffer[weight_idx] = weight
                total_weight += weight

            if total_weight <= 0.0 or not np.isfinite(total_weight):
                start_idx = _price_to_bin_index_numba(hl2, price_min, step, bins)
                length = 1
                total_weight = 1.0
                weight_buffer[0] = 1.0

        decay = horizon_decays[horizon_idx]
        if decay != 1.0:
            global_scales[horizon_idx] *= decay
            scale = global_scales[horizon_idx]
            if scale < renormalize_scale_min:
                for bin_idx in range(bins):
                    raw_profile_buffer[offset + bin_idx] = (
                            raw_profile_buffer[offset + bin_idx] * scale
                    )
                global_scales[horizon_idx] = 1.0
                scale = 1.0
        else:
            scale = global_scales[horizon_idx]

        base_increment = float(volume) / total_weight
        inv_scale = base_increment / scale
        for weight_idx in range(length):
            raw_profile_buffer[offset + start_idx + weight_idx] += (
                    weight_buffer[weight_idx] * inv_scale
            )


@njit(cache=True)
def _build_volume_profile_feature_matrix_numba(
        high,
        low,
        volume,
        keep_mask,
        out_rows,
        price_min,
        neighbor_bins,
        eps,
        horizon_offsets,
        horizon_bins,
        horizon_steps,
        horizon_local_windows,
        horizon_sigma_divisors,
        horizon_min_sigmas,
        horizon_decays,
        feature_count,
        total_bins,
        max_bins,
        renormalize_scale_min,
):
    raw_profile_buffer = np.zeros(total_bins, dtype=np.float64)
    global_scales = np.ones(horizon_bins.shape[0], dtype=np.float64)
    feature_matrix = np.empty((out_rows, feature_count), dtype=np.float64)
    weight_buffer = np.empty(max_bins, dtype=np.float64)

    out_idx = 0
    row_count = high.shape[0]
    for row_idx in range(row_count):
        row_high = high[row_idx]
        row_low = low[row_idx]
        _update_state_with_candle_numba(
            raw_profile_buffer=raw_profile_buffer,
            global_scales=global_scales,
            horizon_offsets=horizon_offsets,
            horizon_bins=horizon_bins,
            horizon_steps=horizon_steps,
            horizon_sigma_divisors=horizon_sigma_divisors,
            horizon_min_sigmas=horizon_min_sigmas,
            horizon_decays=horizon_decays,
            high=row_high,
            low=row_low,
            volume=volume[row_idx],
            price_min=price_min,
            renormalize_scale_min=renormalize_scale_min,
            weight_buffer=weight_buffer,
        )

        if keep_mask[row_idx]:
            _extract_feature_row_array_numba(
                raw_profile_buffer=raw_profile_buffer,
                global_scales=global_scales,
                high=row_high,
                low=row_low,
                price_min=price_min,
                neighbor_bins=neighbor_bins,
                eps=eps,
                horizon_offsets=horizon_offsets,
                horizon_bins=horizon_bins,
                horizon_steps=horizon_steps,
                horizon_local_windows=horizon_local_windows,
                out_row=feature_matrix[out_idx],
            )
            out_idx += 1

    return feature_matrix, raw_profile_buffer, global_scales


def _normalize_keep_mask(keep_mask, row_count):
    if keep_mask is None:
        return np.ones(row_count, dtype=np.bool_)

    keep_mask_np = np.asarray(keep_mask, dtype=np.bool_)
    if keep_mask_np.ndim != 1 or keep_mask_np.shape[0] != row_count:
        raise ValueError(
            "volume profile keep_mask must be a 1D boolean array with the same length as the inputs."
        )
    return np.ascontiguousarray(keep_mask_np)


def _normalized_horizon_arrays(normalized):
    return {
        "horizon_offsets": np.ascontiguousarray(
            np.asarray(normalized["horizon_offsets"], dtype=np.int64)
        ),
        "horizon_bins": np.ascontiguousarray(
            np.asarray(normalized["horizon_bins"], dtype=np.int64)
        ),
        "horizon_steps": np.ascontiguousarray(
            np.asarray(normalized["horizon_steps"], dtype=np.float64)
        ),
        "horizon_local_windows": np.ascontiguousarray(
            np.asarray(normalized["horizon_local_windows"], dtype=np.int64)
        ),
        "horizon_sigma_divisors": np.ascontiguousarray(
            np.asarray(normalized["horizon_sigma_divisors"], dtype=np.float64)
        ),
        "horizon_min_sigmas": np.ascontiguousarray(
            np.asarray(normalized["horizon_min_sigmas"], dtype=np.float64)
        ),
        "horizon_decays": np.ascontiguousarray(
            np.asarray(normalized["decays"], dtype=np.float64)
        ),
    }


def _apply_flat_profiles_to_state(state, normalized, raw_profile_buffer, global_scales):
    for horizon_idx, horizon_name in enumerate(state["horizon_names"]):
        offset = int(normalized["horizons"][horizon_name]["offset"])
        bins = int(normalized["horizons"][horizon_name]["bins"])
        state["horizons"][horizon_name]["raw_profile"][:] = raw_profile_buffer[
            offset: offset + bins
        ]
        state["horizons"][horizon_name]["global_scale"] = float(
            global_scales[horizon_idx]
        )


def build_volume_profile_feature_matrix_from_arrays(
        high,
        low,
        volume,
        cfg=None,
        keep_mask=None,
):
    normalized = normalize_config(cfg)
    state = create_empty_state(normalized)
    row_count = len(high)
    if not state["enabled"]:
        return np.empty((0, 0), dtype=np.float64), state

    high_np = np.ascontiguousarray(np.asarray(high, dtype=np.float64))
    low_np = np.ascontiguousarray(np.asarray(low, dtype=np.float64))
    volume_np = np.ascontiguousarray(np.asarray(volume, dtype=np.float64))
    if low_np.shape[0] != row_count or volume_np.shape[0] != row_count:
        raise ValueError("volume profile input arrays must have the same length.")

    keep_mask_np = _normalize_keep_mask(keep_mask, row_count=row_count)
    out_rows = int(keep_mask_np.sum())
    horizon_arrays = _normalized_horizon_arrays(normalized)

    feature_matrix, raw_profile_buffer, global_scales = (
        _build_volume_profile_feature_matrix_numba(
            high=high_np,
            low=low_np,
            volume=volume_np,
            keep_mask=keep_mask_np,
            out_rows=out_rows,
            price_min=float(state["price_min"]),
            neighbor_bins=int(state["neighbor_bins"]),
            eps=float(state["eps"]),
            horizon_offsets=horizon_arrays["horizon_offsets"],
            horizon_bins=horizon_arrays["horizon_bins"],
            horizon_steps=horizon_arrays["horizon_steps"],
            horizon_local_windows=horizon_arrays["horizon_local_windows"],
            horizon_sigma_divisors=horizon_arrays["horizon_sigma_divisors"],
            horizon_min_sigmas=horizon_arrays["horizon_min_sigmas"],
            horizon_decays=horizon_arrays["horizon_decays"],
            feature_count=len(state["feature_columns"]),
            total_bins=int(normalized["total_bins"]),
            max_bins=int(normalized["max_bins"]),
            renormalize_scale_min=float(_RENORMALIZE_SCALE_MIN),
        )
    )

    _apply_flat_profiles_to_state(
        state,
        normalized,
        raw_profile_buffer=raw_profile_buffer,
        global_scales=global_scales,
    )
    return feature_matrix, state


def _build_candle_contribution_slice(
        high,
        low,
        volume,
        price_min,
        step,
        bins,
        sigma_divisor,
        min_sigma,
):
    if not np.isfinite(high) or not np.isfinite(low) or not np.isfinite(volume):
        return None, np.empty(0, dtype=np.float64)
    if volume == 0.0:
        return None, np.empty(0, dtype=np.float64)

    hl2 = 0.5 * (high + low)
    if high <= low or abs(high - low) <= 1e-12:
        bin_idx = _price_to_bin_index(hl2, price_min, step, bins)
        return bin_idx, np.asarray([float(volume)], dtype=np.float64)

    start_idx = _price_to_bin_index(low, price_min, step, bins)
    end_idx = _price_to_bin_index(high, price_min, step, bins)
    weights = np.zeros(end_idx - start_idx + 1, dtype=np.float64)
    mu = hl2
    sigma = max((high - low) / sigma_divisor, min_sigma)

    total_weight = 0.0
    for offset, bin_idx in enumerate(range(start_idx, end_idx + 1)):
        bin_left = price_min + float(bin_idx) * step
        bin_right = bin_left + step
        left = max(low, bin_left)
        right = min(high, bin_right)
        if right <= left:
            continue
        weight = _normal_cdf((right - mu) / sigma) - _normal_cdf((left - mu) / sigma)
        if weight <= 0.0 or not np.isfinite(weight):
            continue
        weights[offset] = weight
        total_weight += weight

    if total_weight <= 0.0 or not np.isfinite(total_weight):
        bin_idx = _price_to_bin_index(hl2, price_min, step, bins)
        return bin_idx, np.asarray([float(volume)], dtype=np.float64)

    deltas = np.asarray((float(volume) * (weights / total_weight)), dtype=np.float64)
    return start_idx, deltas


def _extract_feature_row_array(
        state,
        high,
        low,
):
    if not state["enabled"]:
        return np.empty(0, dtype=np.float64)

    neighbor_bins = int(state["neighbor_bins"])
    eps = float(state["eps"])
    price_ref = 0.5 * (float(high) + float(low))

    out = np.empty(len(state["feature_columns"]), dtype=np.float64)
    cursor = 0

    for horizon_name in state["horizon_names"]:
        horizon_state = state["horizons"][horizon_name]
        bins = int(horizon_state["bins"])
        step = float(horizon_state["step"])
        local_window = int(horizon_state["local_window"])
        raw_profile = horizon_state["raw_profile"]
        scale = float(horizon_state["global_scale"])
        bin_idx = _price_to_bin_index(price_ref, float(state["price_min"]), step, bins)
        curr = float(raw_profile[bin_idx]) * scale

        for shift in range(-neighbor_bins, 0):
            neighbor_idx = min(max(bin_idx + shift, 0), bins - 1)
            neigh = float(raw_profile[neighbor_idx]) * scale
            out[cursor] = math.log((neigh + eps) / (curr + eps))
            cursor += 1
        for shift in range(1, neighbor_bins + 1):
            neighbor_idx = min(max(bin_idx + shift, 0), bins - 1)
            neigh = float(raw_profile[neighbor_idx]) * scale
            out[cursor] = math.log((neigh + eps) / (curr + eps))
            cursor += 1

        left = max(0, bin_idx - local_window)
        right = min(bins, bin_idx + local_window + 1)
        local_slice = raw_profile[left:right]
        center = bin_idx - left

        above = float(local_slice[center + 1:].sum(dtype=np.float64)) * scale
        below = float(local_slice[:center].sum(dtype=np.float64)) * scale
        local_sum = float(local_slice.sum(dtype=np.float64)) * scale
        local_max = float(local_slice.max()) * scale

        out[cursor] = math.log((above + eps) / (below + eps))
        cursor += 1
        out[cursor] = curr / (local_sum + eps)
        cursor += 1
        out[cursor] = curr / (local_max + eps)
        cursor += 1

    return out


def extract_features_from_state(
        state,
        high,
        low,
):
    values = _extract_feature_row_array(state, high=high, low=low)
    return {
        feature_col: float(values[idx])
        for idx, feature_col in enumerate(state["feature_columns"])
    }


def update_state_with_candle(
        state,
        high,
        low,
        volume,
):
    if not state["enabled"]:
        return state

    for horizon_name in state["horizon_names"]:
        horizon_state = state["horizons"][horizon_name]
        start_idx, deltas = _build_candle_contribution_slice(
            high=float(high),
            low=float(low),
            volume=float(volume),
            price_min=float(state["price_min"]),
            step=float(horizon_state["step"]),
            bins=int(horizon_state["bins"]),
            sigma_divisor=float(horizon_state["sigma_divisor"]),
            min_sigma=float(horizon_state["min_sigma"]),
        )
        if start_idx is None or deltas.size == 0:
            continue

        if float(horizon_state["decay"]) != 1.0:
            horizon_state["global_scale"] *= float(horizon_state["decay"])
            if horizon_state["global_scale"] < _RENORMALIZE_SCALE_MIN:
                horizon_state["raw_profile"] *= horizon_state["global_scale"]
                horizon_state["global_scale"] = 1.0

        stop_idx = int(start_idx) + int(deltas.shape[0])
        horizon_state["raw_profile"][start_idx:stop_idx] += np.asarray(
            deltas / float(horizon_state["global_scale"]),
            dtype=np.float64,
        )

    return state


def bootstrap_state_from_history(
        df_hist,
        cfg=None,
):
    _require_dataframe_columns(df_hist, _REQUIRED_COLUMNS)
    normalized = normalize_config(cfg)
    state = create_empty_state(normalized)
    if not state["enabled"]:
        return state

    print(
        "[vp] bootstrap state start | "
        f"rows={len(df_hist)} horizons={_format_horizon_summary(normalized)}"
    )
    keep_mask = np.zeros(len(df_hist), dtype=np.bool_)
    _, state = build_volume_profile_feature_matrix_from_arrays(
        high=df_hist["High"].to_numpy(dtype=np.float64, copy=False),
        low=df_hist["Low"].to_numpy(dtype=np.float64, copy=False),
        volume=df_hist["Volume"].to_numpy(dtype=np.float64, copy=False),
        cfg=normalized,
        keep_mask=keep_mask,
    )

    if "Opened" in df_hist.columns and len(df_hist) > 0:
        state["last_candle_time"] = str(
            pd.Timestamp(df_hist["Opened"].iloc[-1]).isoformat()
        )
    print("[vp] bootstrap state done")
    return state


def build_volume_profile_features(
        df,
        cfg=None,
        verbose=True,
):
    _require_dataframe_columns(df, _REQUIRED_COLUMNS)
    normalized = normalize_config(cfg)
    state = create_empty_state(normalized)
    if not state["enabled"]:
        return pd.DataFrame(index=df.index), state

    row_count = len(df)
    if verbose:
        print(
            "[vp] build start | "
            f"rows={row_count} horizons={_format_horizon_summary(normalized)}"
        )

    feature_matrix, state = build_volume_profile_feature_matrix_from_arrays(
        high=df["High"].to_numpy(dtype=np.float64, copy=False),
        low=df["Low"].to_numpy(dtype=np.float64, copy=False),
        volume=df["Volume"].to_numpy(dtype=np.float64, copy=False),
        cfg=normalized,
    )

    if "Opened" in df.columns and row_count > 0:
        state["last_candle_time"] = str(pd.Timestamp(df["Opened"].iloc[-1]).isoformat())

    feature_df = pd.DataFrame(
        feature_matrix, columns=state["feature_columns"], index=df.index
    )
    if verbose:
        print("[vp] build columns: " + ", ".join(state["feature_columns"]))
    return feature_df, state


def check_batch_live_consistency(
        df,
        cfg=None,
        atol=1e-6,
        rtol=1e-6,
):
    _require_dataframe_columns(df, _REQUIRED_COLUMNS)
    batch_df, _ = build_volume_profile_features(df, cfg)
    live_state = create_empty_state(cfg)

    high = df["High"].to_numpy(dtype=np.float64, copy=False)
    low = df["Low"].to_numpy(dtype=np.float64, copy=False)
    volume = df["Volume"].to_numpy(dtype=np.float64, copy=False)

    live_matrix = np.empty_like(batch_df.to_numpy(dtype=np.float64, copy=True))
    for row_idx in range(len(df)):
        update_state_with_candle(
            live_state,
            high=float(high[row_idx]),
            low=float(low[row_idx]),
            volume=float(volume[row_idx]),
        )
        live_matrix[row_idx, :] = _extract_feature_row_array(
            live_state,
            high=float(high[row_idx]),
            low=float(low[row_idx]),
        )

    batch_matrix = batch_df.to_numpy(dtype=np.float64, copy=False)
    live_matrix64 = live_matrix.astype(np.float64, copy=False)
    allclose = bool(
        np.allclose(
            batch_matrix,
            live_matrix64,
            atol=float(atol),
            rtol=float(rtol),
            equal_nan=True,
        )
    )
    max_abs_diff = (
        float(np.nanmax(np.abs(batch_matrix - live_matrix64)))
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
    "RUNTIME_STATE_DIR",
    "STATE_DIR",
    "VP_FEATURE_PREFIX",
    "build_volume_profile_feature_matrix_from_arrays",
    "build_volume_profile_features",
    "bootstrap_state_from_history",
    "check_batch_live_consistency",
    "config_signature",
    "create_empty_state",
    "extract_features_from_state",
    "get_feature_columns",
    "is_volume_profile_feature",
    "load_state",
    "normalize_config",
    "save_state",
    "state_matches_config",
    "update_state_with_candle",
    "validate_volume_profile_dataset_metadata",
    "validate_volume_profile_feature_columns",
    "validate_volume_profile_model_metadata",
]
