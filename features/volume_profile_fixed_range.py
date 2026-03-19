from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit


FEATURE_VERSION = "vp_fixed_range_v1"
VP_FEATURE_PREFIX = "vp_"
STATE_DIR = Path("data/feature_state/volume_profile")

DEFAULT_CONFIG = {
    "enabled": True,
    "price_min": 1.0,
    "price_max": 200000.0,
    "step": 5,
    "neighbor_bins": 2,
    "local_window": 2,
    "sigma_divisor": 4.0,
    "min_sigma": 2.5,
    "eps": 1e-6,
    "horizons": {
        "short": {"half_life_candles": 1440},
        "medium": {"half_life_candles": 10080},
        "long": {"half_life_candles": 43200},
        "all": {"half_life_candles": None},
    },
}

_HORIZON_ORDER = ("short", "medium", "long", "all")
_REQUIRED_COLUMNS = ("High", "Low", "Volume")
_RENORMALIZE_SCALE_MIN = 1e-3
_SQRT_TWO = math.sqrt(2.0)


def is_volume_profile_feature(feature_name: str) -> bool:
    return str(feature_name).startswith(VP_FEATURE_PREFIX)


def normalize_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    user_cfg = dict(cfg or {})
    user_horizons = user_cfg.pop("horizons", None)

    merged = dict(DEFAULT_CONFIG)
    merged.update(user_cfg)

    horizons: dict[str, dict[str, Any]] = {}
    user_horizons = user_horizons if isinstance(user_horizons, dict) else {}
    for horizon_name in _HORIZON_ORDER:
        horizon_cfg = dict(DEFAULT_CONFIG["horizons"][horizon_name])
        user_horizon_cfg = user_horizons.get(horizon_name)
        if isinstance(user_horizon_cfg, dict):
            horizon_cfg.update(user_horizon_cfg)
        horizons[horizon_name] = horizon_cfg

    price_min = float(merged["price_min"])
    price_max = float(merged["price_max"])
    step_raw = float(merged["step"])
    if not np.isfinite(price_min) or not np.isfinite(price_max):
        raise ValueError("volume profile config requires finite price_min and price_max.")
    if price_max <= price_min:
        raise ValueError("volume profile config requires price_max > price_min.")
    if not np.isfinite(step_raw) or step_raw <= 0.0:
        raise ValueError("volume profile config requires step > 0.")
    if not float(step_raw).is_integer():
        raise ValueError("volume profile config requires integer-valued step.")
    step = int(step_raw)

    bins = int(math.ceil((price_max - price_min) / step))
    if bins <= 0:
        raise ValueError("volume profile config produced no bins.")

    neighbor_bins = int(merged["neighbor_bins"])
    local_window = int(merged["local_window"])
    sigma_divisor = float(merged["sigma_divisor"])
    min_sigma = float(merged["min_sigma"])
    eps = float(merged["eps"])

    if neighbor_bins <= 0:
        raise ValueError("volume profile config requires neighbor_bins > 0.")
    if local_window <= 0:
        raise ValueError("volume profile config requires local_window > 0.")
    if sigma_divisor <= 0.0:
        raise ValueError("volume profile config requires sigma_divisor > 0.")
    if min_sigma <= 0.0:
        raise ValueError("volume profile config requires min_sigma > 0.")
    if eps <= 0.0:
        raise ValueError("volume profile config requires eps > 0.")

    half_lives: list[int | None] = []
    decays: list[float] = []
    horizon_names: list[str] = []
    for horizon_name in _HORIZON_ORDER:
        half_life = horizons[horizon_name].get("half_life_candles")
        if half_life is None:
            decay = 1.0
            normalized_half_life = None
        else:
            normalized_half_life = int(half_life)
            if normalized_half_life <= 0:
                raise ValueError(
                    f"volume profile horizon '{horizon_name}' requires half_life_candles > 0 or null."
                )
            decay = math.exp(math.log(0.5) / float(normalized_half_life))
        horizons[horizon_name] = {
            "half_life_candles": normalized_half_life,
            "decay": float(decay),
        }
        horizon_names.append(horizon_name)
        half_lives.append(normalized_half_life)
        decays.append(float(decay))

    normalized = {
        "enabled": bool(merged.get("enabled", True)),
        "price_min": price_min,
        "price_max": price_max,
        "step": step,
        "bins": bins,
        "neighbor_bins": neighbor_bins,
        "local_window": local_window,
        "sigma_divisor": sigma_divisor,
        "min_sigma": min_sigma,
        "eps": eps,
        "horizons": horizons,
        "horizon_names": tuple(horizon_names),
        "half_lives": tuple(half_lives),
        "decays": tuple(decays),
        "version": FEATURE_VERSION,
    }
    normalized["feature_columns"] = get_feature_columns(normalized)
    normalized["config_signature"] = _config_signature_from_normalized(normalized)
    return normalized


def config_signature(cfg: dict[str, Any] | None = None) -> str:
    normalized = cfg if cfg and "horizon_names" in cfg else normalize_config(cfg)
    return _config_signature_from_normalized(normalized)


def _config_signature_from_normalized(normalized: dict[str, Any]) -> str:
    payload = {
        "version": normalized["version"],
        "enabled": bool(normalized["enabled"]),
        "price_min": float(normalized["price_min"]),
        "price_max": float(normalized["price_max"]),
        "step": float(normalized["step"]),
        "bins": int(normalized["bins"]),
        "neighbor_bins": int(normalized["neighbor_bins"]),
        "local_window": int(normalized["local_window"]),
        "sigma_divisor": float(normalized["sigma_divisor"]),
        "min_sigma": float(normalized["min_sigma"]),
        "eps": float(normalized["eps"]),
        "horizons": {
            name: {
                "half_life_candles": normalized["horizons"][name]["half_life_candles"],
                "decay": float(normalized["horizons"][name]["decay"]),
            }
            for name in normalized["horizon_names"]
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def get_feature_columns(cfg: dict[str, Any] | None = None) -> tuple[str, ...]:
    normalized = cfg if cfg and "horizon_names" in cfg else normalize_config(cfg)
    cols: list[str] = []
    neighbor_bins = int(normalized["neighbor_bins"])
    for horizon_name in normalized["horizon_names"]:
        for shift in range(-neighbor_bins, 0):
            cols.append(f"vp_{horizon_name}_lr_m{abs(shift)}")
        for shift in range(1, neighbor_bins + 1):
            cols.append(f"vp_{horizon_name}_lr_p{shift}")
        cols.append(f"vp_{horizon_name}_imbalance")
        cols.append(f"vp_{horizon_name}_curr_share")
        cols.append(f"vp_{horizon_name}_peak_ratio")
    return tuple(cols)


def create_empty_state(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = cfg if cfg and "horizon_names" in cfg else normalize_config(cfg)
    horizon_count = len(normalized["horizon_names"])
    bins = int(normalized["bins"])
    return {
        "enabled": bool(normalized["enabled"]),
        "version": normalized["version"],
        "price_min": float(normalized["price_min"]),
        "price_max": float(normalized["price_max"]),
        "step": float(normalized["step"]),
        "bins": bins,
        "neighbor_bins": int(normalized["neighbor_bins"]),
        "local_window": int(normalized["local_window"]),
        "sigma_divisor": float(normalized["sigma_divisor"]),
        "min_sigma": float(normalized["min_sigma"]),
        "eps": float(normalized["eps"]),
        "horizon_names": tuple(normalized["horizon_names"]),
        "half_lives": tuple(normalized["half_lives"]),
        "decays": np.asarray(normalized["decays"], dtype=np.float64),
        "raw_profiles": np.zeros((horizon_count, bins), dtype=np.float32),
        "global_scales": np.ones(horizon_count, dtype=np.float64),
        "feature_columns": tuple(normalized["feature_columns"]),
        "config_signature": normalized["config_signature"],
        "last_candle_time": None,
    }


def state_matches_config(state: dict[str, Any], cfg: dict[str, Any] | None = None) -> bool:
    return str(state.get("config_signature", "")) == config_signature(cfg)


def _state_base_path(path: str | Path) -> Path:
    base_path = Path(path)
    if base_path.suffix.lower() in {".npz", ".json"}:
        return base_path.with_suffix("")
    return base_path


def _state_metadata_dict(state: dict[str, Any]) -> dict[str, Any]:
    horizon_meta = {}
    for idx, name in enumerate(state["horizon_names"]):
        horizon_meta[name] = {
            "half_life_candles": state["half_lives"][idx],
            "decay": float(state["decays"][idx]),
            "global_scale": float(state["global_scales"][idx]),
        }

    return {
        "version": str(state["version"]),
        "config_signature": str(state["config_signature"]),
        "price_min": float(state["price_min"]),
        "price_max": float(state["price_max"]),
        "step": float(state["step"]),
        "bins": int(state["bins"]),
        "neighbor_bins": int(state["neighbor_bins"]),
        "local_window": int(state["local_window"]),
        "sigma_divisor": float(state["sigma_divisor"]),
        "min_sigma": float(state["min_sigma"]),
        "eps": float(state["eps"]),
        "horizon_names": list(state["horizon_names"]),
        "feature_columns": list(state["feature_columns"]),
        "horizons": horizon_meta,
        "last_candle_time": state.get("last_candle_time"),
    }


def save_state(state: dict[str, Any], path: str | Path) -> dict[str, Path]:
    base_path = _state_base_path(path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    npz_path = base_path.with_suffix(".npz")
    json_path = base_path.with_suffix(".json")

    np.savez_compressed(
        npz_path,
        raw_profiles=state["raw_profiles"],
        global_scales=state["global_scales"],
        decays=state["decays"],
    )
    json_path.write_text(
        json.dumps(_state_metadata_dict(state), indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    return {"npz": npz_path, "json": json_path}


def load_state(path: str | Path) -> dict[str, Any]:
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

    cfg = {
        "enabled": True,
        "price_min": meta["price_min"],
        "price_max": meta["price_max"],
        "step": meta["step"],
        "neighbor_bins": meta["neighbor_bins"],
        "local_window": meta["local_window"],
        "sigma_divisor": meta["sigma_divisor"],
        "min_sigma": meta["min_sigma"],
        "eps": meta["eps"],
        "horizons": {
            name: {
                "half_life_candles": meta["horizons"][name]["half_life_candles"],
            }
            for name in meta["horizon_names"]
        },
    }
    state = create_empty_state(cfg)
    with np.load(npz_path) as data:
        raw_profiles = np.asarray(data["raw_profiles"], dtype=np.float32)
        global_scales = np.asarray(data["global_scales"], dtype=np.float64)
        decays = np.asarray(data["decays"], dtype=np.float64)

    expected_shape = state["raw_profiles"].shape
    if raw_profiles.shape != expected_shape:
        raise ValueError(
            f"volume profile raw_profiles shape mismatch: {raw_profiles.shape} != {expected_shape}"
        )
    if global_scales.shape != state["global_scales"].shape:
        raise ValueError(
            "volume profile global_scales shape mismatch: "
            f"{global_scales.shape} != {state['global_scales'].shape}"
        )
    if decays.shape != state["decays"].shape:
        raise ValueError(
            f"volume profile decays shape mismatch: {decays.shape} != {state['decays'].shape}"
        )

    state["raw_profiles"][:] = raw_profiles
    state["global_scales"][:] = global_scales
    state["decays"][:] = decays
    state["last_candle_time"] = meta.get("last_candle_time")
    state["config_signature"] = str(meta.get("config_signature", state["config_signature"]))
    return state


def _require_dataframe_columns(df: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"volume profile dataframe missing required columns: {missing}")


def _price_to_bin_index(price: float, price_min: float, step: float, bins: int) -> int:
    idx = int((float(price) - price_min) / step)
    if idx < 0:
        return 0
    if idx >= bins:
        return bins - 1
    return idx


def _normal_cdf(x: float) -> float:
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
    raw_profiles,
    global_scales,
    high,
    low,
    price_min,
    step,
    bins,
    neighbor_bins,
    local_window,
    eps,
    out_row,
):
    price_ref = 0.5 * (high + low)
    bin_idx = _price_to_bin_index_numba(price_ref, price_min, step, bins)
    cursor = 0
    horizon_count = raw_profiles.shape[0]

    for horizon_idx in range(horizon_count):
        scale = global_scales[horizon_idx]
        curr = float(raw_profiles[horizon_idx, bin_idx]) * scale

        for shift in range(-neighbor_bins, 0):
            neighbor_idx = bin_idx + shift
            if neighbor_idx < 0:
                neighbor_idx = 0
            elif neighbor_idx >= bins:
                neighbor_idx = bins - 1
            neigh = float(raw_profiles[horizon_idx, neighbor_idx]) * scale
            out_row[cursor] = np.float32(math.log((neigh + eps) / (curr + eps)))
            cursor += 1

        for shift in range(1, neighbor_bins + 1):
            neighbor_idx = bin_idx + shift
            if neighbor_idx < 0:
                neighbor_idx = 0
            elif neighbor_idx >= bins:
                neighbor_idx = bins - 1
            neigh = float(raw_profiles[horizon_idx, neighbor_idx]) * scale
            out_row[cursor] = np.float32(math.log((neigh + eps) / (curr + eps)))
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
            value = float(raw_profiles[horizon_idx, local_idx]) * scale
            local_sum += value
            if value > local_max:
                local_max = value
            if local_idx < bin_idx:
                below += value
            elif local_idx > bin_idx:
                above += value

        out_row[cursor] = np.float32(math.log((above + eps) / (below + eps)))
        cursor += 1
        out_row[cursor] = np.float32(curr / (local_sum + eps))
        cursor += 1
        out_row[cursor] = np.float32(curr / (local_max + eps))
        cursor += 1


@njit(cache=True)
def _update_state_with_candle_numba(
    raw_profiles,
    global_scales,
    decays,
    high,
    low,
    volume,
    price_min,
    step,
    bins,
    sigma_divisor,
    min_sigma,
    renormalize_scale_min,
    weight_buffer,
):
    if not np.isfinite(high) or not np.isfinite(low) or not np.isfinite(volume):
        return
    if volume == 0.0:
        return

    hl2 = 0.5 * (high + low)
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
        for offset in range(length):
            bin_idx = start_idx + offset
            bin_left = price_min + float(bin_idx) * step
            bin_right = bin_left + step
            left = low if low > bin_left else bin_left
            right = high if high < bin_right else bin_right
            if right <= left:
                weight_buffer[offset] = 0.0
                continue
            weight = _normal_cdf_numba((right - mu) / sigma) - _normal_cdf_numba(
                (left - mu) / sigma
            )
            if weight <= 0.0 or not np.isfinite(weight):
                weight_buffer[offset] = 0.0
                continue
            weight_buffer[offset] = weight
            total_weight += weight

        if total_weight <= 0.0 or not np.isfinite(total_weight):
            start_idx = _price_to_bin_index_numba(hl2, price_min, step, bins)
            length = 1
            total_weight = 1.0
            weight_buffer[0] = 1.0

    base_increment = float(volume) / total_weight
    horizon_count = raw_profiles.shape[0]

    for horizon_idx in range(horizon_count):
        decay = decays[horizon_idx]
        if decay != 1.0:
            global_scales[horizon_idx] *= decay
            scale = global_scales[horizon_idx]
            if scale < renormalize_scale_min:
                for bin_idx in range(bins):
                    raw_profiles[horizon_idx, bin_idx] = np.float32(
                        raw_profiles[horizon_idx, bin_idx] * scale
                    )
                global_scales[horizon_idx] = 1.0
                scale = 1.0
        else:
            scale = global_scales[horizon_idx]

        inv_scale = base_increment / scale
        for offset in range(length):
            raw_profiles[horizon_idx, start_idx + offset] += np.float32(
                weight_buffer[offset] * inv_scale
            )


@njit(cache=True)
def _build_volume_profile_feature_matrix_numba(
    high,
    low,
    volume,
    keep_mask,
    out_rows,
    price_min,
    step,
    bins,
    neighbor_bins,
    local_window,
    sigma_divisor,
    min_sigma,
    eps,
    decays,
    feature_count,
    renormalize_scale_min,
):
    horizon_count = decays.shape[0]
    raw_profiles = np.zeros((horizon_count, bins), dtype=np.float32)
    global_scales = np.ones(horizon_count, dtype=np.float64)
    feature_matrix = np.empty((out_rows, feature_count), dtype=np.float32)
    weight_buffer = np.empty(bins, dtype=np.float64)

    out_idx = 0
    row_count = high.shape[0]
    for row_idx in range(row_count):
        row_high = high[row_idx]
        row_low = low[row_idx]
        if keep_mask[row_idx]:
            _extract_feature_row_array_numba(
                raw_profiles=raw_profiles,
                global_scales=global_scales,
                high=row_high,
                low=row_low,
                price_min=price_min,
                step=step,
                bins=bins,
                neighbor_bins=neighbor_bins,
                local_window=local_window,
                eps=eps,
                out_row=feature_matrix[out_idx],
            )
            out_idx += 1

        _update_state_with_candle_numba(
            raw_profiles=raw_profiles,
            global_scales=global_scales,
            decays=decays,
            high=row_high,
            low=row_low,
            volume=volume[row_idx],
            price_min=price_min,
            step=step,
            bins=bins,
            sigma_divisor=sigma_divisor,
            min_sigma=min_sigma,
            renormalize_scale_min=renormalize_scale_min,
            weight_buffer=weight_buffer,
        )

    return feature_matrix, raw_profiles, global_scales


def _normalize_keep_mask(keep_mask, row_count: int) -> np.ndarray:
    if keep_mask is None:
        return np.ones(row_count, dtype=np.bool_)

    keep_mask_np = np.asarray(keep_mask, dtype=np.bool_)
    if keep_mask_np.ndim != 1 or keep_mask_np.shape[0] != row_count:
        raise ValueError(
            "volume profile keep_mask must be a 1D boolean array with the same length as the inputs."
        )
    return np.ascontiguousarray(keep_mask_np)


def build_volume_profile_feature_matrix_from_arrays(
    high,
    low,
    volume,
    cfg: dict[str, Any] | None = None,
    keep_mask=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    normalized = normalize_config(cfg)
    state = create_empty_state(normalized)
    row_count = len(high)
    if not state["enabled"]:
        return np.empty((0, 0), dtype=np.float32), state

    high_np = np.ascontiguousarray(np.asarray(high, dtype=np.float64))
    low_np = np.ascontiguousarray(np.asarray(low, dtype=np.float64))
    volume_np = np.ascontiguousarray(np.asarray(volume, dtype=np.float64))
    if low_np.shape[0] != row_count or volume_np.shape[0] != row_count:
        raise ValueError("volume profile input arrays must have the same length.")

    keep_mask_np = _normalize_keep_mask(keep_mask, row_count=row_count)
    out_rows = int(keep_mask_np.sum())

    feature_matrix, raw_profiles, global_scales = (
        _build_volume_profile_feature_matrix_numba(
            high=high_np,
            low=low_np,
            volume=volume_np,
            keep_mask=keep_mask_np,
            out_rows=out_rows,
            price_min=float(state["price_min"]),
            step=float(state["step"]),
            bins=int(state["bins"]),
            neighbor_bins=int(state["neighbor_bins"]),
            local_window=int(state["local_window"]),
            sigma_divisor=float(state["sigma_divisor"]),
            min_sigma=float(state["min_sigma"]),
            eps=float(state["eps"]),
            decays=np.ascontiguousarray(state["decays"], dtype=np.float64),
            feature_count=len(state["feature_columns"]),
            renormalize_scale_min=float(_RENORMALIZE_SCALE_MIN),
        )
    )

    state["raw_profiles"][:] = raw_profiles
    state["global_scales"][:] = global_scales
    return feature_matrix, state


def _build_candle_contribution_slice(
    high: float,
    low: float,
    volume: float,
    price_min: float,
    step: float,
    bins: int,
    sigma_divisor: float,
    min_sigma: float,
) -> tuple[int | None, np.ndarray]:
    if not np.isfinite(high) or not np.isfinite(low) or not np.isfinite(volume):
        return None, np.empty(0, dtype=np.float32)
    if volume == 0.0:
        return None, np.empty(0, dtype=np.float32)

    hl2 = 0.5 * (high + low)
    if high <= low or abs(high - low) <= 1e-12:
        bin_idx = _price_to_bin_index(hl2, price_min, step, bins)
        return bin_idx, np.asarray([float(volume)], dtype=np.float32)

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
        return bin_idx, np.asarray([float(volume)], dtype=np.float32)

    deltas = np.asarray((float(volume) * (weights / total_weight)), dtype=np.float32)
    return start_idx, deltas


def _extract_feature_row_array(
    state: dict[str, Any],
    high: float,
    low: float,
) -> np.ndarray:
    if not state["enabled"]:
        return np.empty(0, dtype=np.float32)

    neighbor_bins = int(state["neighbor_bins"])
    local_window = int(state["local_window"])
    eps = float(state["eps"])
    bins = int(state["bins"])
    price_ref = 0.5 * (float(high) + float(low))
    bin_idx = _price_to_bin_index(price_ref, float(state["price_min"]), float(state["step"]), bins)

    out = np.empty(len(state["feature_columns"]), dtype=np.float32)
    cursor = 0

    for horizon_idx in range(len(state["horizon_names"])):
        raw_profile = state["raw_profiles"][horizon_idx]
        scale = float(state["global_scales"][horizon_idx])
        curr_raw = float(raw_profile[bin_idx])
        curr = curr_raw * scale

        for shift in range(-neighbor_bins, 0):
            neighbor_idx = min(max(bin_idx + shift, 0), bins - 1)
            neigh = float(raw_profile[neighbor_idx]) * scale
            out[cursor] = np.float32(math.log((neigh + eps) / (curr + eps)))
            cursor += 1
        for shift in range(1, neighbor_bins + 1):
            neighbor_idx = min(max(bin_idx + shift, 0), bins - 1)
            neigh = float(raw_profile[neighbor_idx]) * scale
            out[cursor] = np.float32(math.log((neigh + eps) / (curr + eps)))
            cursor += 1

        left = max(0, bin_idx - local_window)
        right = min(bins, bin_idx + local_window + 1)
        local_slice = raw_profile[left:right]
        center = bin_idx - left

        above = float(local_slice[center + 1 :].sum(dtype=np.float64)) * scale
        below = float(local_slice[:center].sum(dtype=np.float64)) * scale
        local_sum = float(local_slice.sum(dtype=np.float64)) * scale
        local_max = float(local_slice.max()) * scale

        out[cursor] = np.float32(math.log((above + eps) / (below + eps)))
        cursor += 1
        out[cursor] = np.float32(curr / (local_sum + eps))
        cursor += 1
        out[cursor] = np.float32(curr / (local_max + eps))
        cursor += 1

    return out


def extract_features_from_state(
    state: dict[str, Any],
    high: float,
    low: float,
) -> dict[str, float]:
    values = _extract_feature_row_array(state, high=high, low=low)
    return {
        feature_col: float(values[idx])
        for idx, feature_col in enumerate(state["feature_columns"])
    }


def update_state_with_candle(
    state: dict[str, Any],
    high: float,
    low: float,
    volume: float,
) -> dict[str, Any]:
    if not state["enabled"]:
        return state

    start_idx, deltas = _build_candle_contribution_slice(
        high=float(high),
        low=float(low),
        volume=float(volume),
        price_min=float(state["price_min"]),
        step=float(state["step"]),
        bins=int(state["bins"]),
        sigma_divisor=float(state["sigma_divisor"]),
        min_sigma=float(state["min_sigma"]),
    )
    if start_idx is None or deltas.size == 0:
        return state

    stop_idx = int(start_idx) + int(deltas.shape[0])
    scaled_deltas = deltas.astype(np.float64, copy=False)

    for horizon_idx in range(len(state["horizon_names"])):
        decay = float(state["decays"][horizon_idx])
        if decay != 1.0:
            state["global_scales"][horizon_idx] *= decay
            scale = float(state["global_scales"][horizon_idx])
            if scale < _RENORMALIZE_SCALE_MIN:
                state["raw_profiles"][horizon_idx] *= np.float32(scale)
                state["global_scales"][horizon_idx] = 1.0
        scale = float(state["global_scales"][horizon_idx])
        increment = np.asarray(scaled_deltas / scale, dtype=np.float32)
        state["raw_profiles"][horizon_idx, start_idx:stop_idx] += increment

    return state


def bootstrap_state_from_history(
    df_hist: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _require_dataframe_columns(df_hist, _REQUIRED_COLUMNS)
    normalized = normalize_config(cfg)
    state = create_empty_state(normalized)
    if not state["enabled"]:
        return state

    print(
        "[vp] bootstrap state start | "
        f"rows={len(df_hist)} bins={state['bins']} horizons={','.join(state['horizon_names'])}"
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
        state["last_candle_time"] = str(pd.Timestamp(df_hist["Opened"].iloc[-1]).isoformat())
    print("[vp] bootstrap state done")
    return state


def build_volume_profile_features(
    df: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_dataframe_columns(df, _REQUIRED_COLUMNS)
    normalized = normalize_config(cfg)
    state = create_empty_state(normalized)
    if not state["enabled"]:
        return pd.DataFrame(index=df.index), state

    row_count = len(df)
    if verbose:
        print(
            "[vp] build start | "
            f"rows={row_count} bins={state['bins']} horizons={','.join(state['horizon_names'])}"
        )

    feature_matrix, state = build_volume_profile_feature_matrix_from_arrays(
        high=df["High"].to_numpy(dtype=np.float64, copy=False),
        low=df["Low"].to_numpy(dtype=np.float64, copy=False),
        volume=df["Volume"].to_numpy(dtype=np.float64, copy=False),
        cfg=normalized,
    )

    if "Opened" in df.columns and row_count > 0:
        state["last_candle_time"] = str(pd.Timestamp(df["Opened"].iloc[-1]).isoformat())

    feature_df = pd.DataFrame(feature_matrix, columns=state["feature_columns"], index=df.index)
    if verbose:
        print("[vp] build columns: " + ", ".join(state["feature_columns"]))
    return feature_df, state


def check_batch_live_consistency(
    df: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    _require_dataframe_columns(df, _REQUIRED_COLUMNS)
    batch_df, _ = build_volume_profile_features(df, cfg)
    live_state = create_empty_state(cfg)

    high = df["High"].to_numpy(dtype=np.float64, copy=False)
    low = df["Low"].to_numpy(dtype=np.float64, copy=False)
    volume = df["Volume"].to_numpy(dtype=np.float64, copy=False)

    live_matrix = np.empty_like(batch_df.to_numpy(dtype=np.float32, copy=True))
    for row_idx in range(len(df)):
        live_matrix[row_idx, :] = _extract_feature_row_array(
            live_state,
            high=float(high[row_idx]),
            low=float(low[row_idx]),
        )
        update_state_with_candle(
            live_state,
            high=float(high[row_idx]),
            low=float(low[row_idx]),
            volume=float(volume[row_idx]),
        )

    batch_matrix = batch_df.to_numpy(dtype=np.float64, copy=False)
    live_matrix64 = live_matrix.astype(np.float64, copy=False)
    allclose = bool(
        np.allclose(batch_matrix, live_matrix64, atol=float(atol), rtol=float(rtol), equal_nan=True)
    )
    max_abs_diff = float(np.nanmax(np.abs(batch_matrix - live_matrix64))) if batch_matrix.size else 0.0
    return {
        "ok": allclose,
        "max_abs_diff": max_abs_diff,
        "atol": float(atol),
        "rtol": float(rtol),
        "rows": int(len(df)),
        "feature_count": int(batch_matrix.shape[1]) if batch_matrix.ndim == 2 else 0,
    }


__all__ = [
    "FEATURE_VERSION",
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
]
