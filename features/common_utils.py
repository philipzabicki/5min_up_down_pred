import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Type

import numpy as np
from numba import njit
from pymoo.core.variable import Binary, Choice, Integer, Real

NAN_RATIO_THRESHOLD = 0.05
NAN_PENALTY = 10.0
DEBUG_DIR = Path(__file__).resolve().parents[1] / "data" / "nan_debug"

REAL_VAR_WEIGHT = 8.0
BINARY_VAR_WEIGHT = 1.0


def _to_serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def score_nan_stats(score):
    arr = np.asarray(score, dtype=np.float64).reshape(-1)
    total_count = int(arr.size)
    if total_count == 0:
        return 1.0, 0, 0
    invalid_count = int((~np.isfinite(arr)).sum())
    invalid_ratio = invalid_count / total_count
    return invalid_ratio, invalid_count, total_count


def score_nan_ratio(score):
    arr = np.asarray(score, dtype=np.float64).reshape(-1)
    total_count = int(arr.size)
    if total_count == 0:
        return 1.0
    return float((~np.isfinite(arr)).sum() / total_count)


def log_nan_debug(indicator_name, params, nan_ratio, nan_count, total_count):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEBUG_DIR / f"{indicator_name}_nan_debug_pid{os.getpid()}.csv"
    needs_header = not out_path.exists()

    with out_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if needs_header:
            writer.writerow(["nan_ratio", "nan_count", "total_count", "params_json"])
        writer.writerow(
            [
                f"{nan_ratio:.10f}",
                nan_count,
                total_count,
                json.dumps(_to_serializable(params), ensure_ascii=True, sort_keys=True),
            ]
        )


@njit(cache=True, nogil=True)
def _quantile_sorted_numba(sorted_arr, q):
    n = sorted_arr.shape[0]
    if n == 0:
        return np.nan
    if q <= 0.0:
        return sorted_arr[0]
    if q >= 1.0:
        return sorted_arr[n - 1]

    pos = q * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return sorted_arr[lo]

    weight_hi = pos - lo
    weight_lo = 1.0 - weight_hi
    return sorted_arr[lo] * weight_lo + sorted_arr[hi] * weight_hi


@njit(cache=True, nogil=True)
def _bucket_stat_numba(values, stat_code, clip_q):
    n = values.shape[0]
    if n == 0:
        return np.nan

    sorted_vals = np.sort(values)

    if stat_code == 1:
        mid = n // 2
        if n % 2 == 1:
            return sorted_vals[mid]
        return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])

    clip = clip_q
    if clip <= 0.0:
        return np.sum(sorted_vals) / n

    lo = _quantile_sorted_numba(sorted_vals, clip)
    hi = _quantile_sorted_numba(sorted_vals, 1.0 - clip)
    left_idx = int(np.searchsorted(sorted_vals, lo, side="left"))
    right_idx = int(np.searchsorted(sorted_vals, hi, side="right"))

    middle_sum = 0.0
    if right_idx > left_idx:
        middle_sum = np.sum(sorted_vals[left_idx:right_idx])

    total = lo * left_idx + middle_sum + hi * (n - right_idx)
    return total / n


@njit(cache=True, nogil=True)
def _segment_score_extremes_vs_mid_oof_numba(
    x_arr,
    y_arr,
    start,
    end,
    train_frac,
    gap,
    q_ext,
    q_mid,
    stat_code,
    clip_q,
    min_bucket_size,
):
    seg_len = end - start
    cut = start + int(train_frac * seg_len)
    if (cut - start) < 2 or cut >= end:
        return np.nan

    test_start = cut + gap
    if test_start >= end:
        return np.nan

    sorted_train_x = np.sort(x_arr[start:cut])
    lo_ext = _quantile_sorted_numba(sorted_train_x, q_ext)
    hi_ext = _quantile_sorted_numba(sorted_train_x, 1.0 - q_ext)
    mid_lo = _quantile_sorted_numba(sorted_train_x, 0.5 - q_mid)
    mid_hi = _quantile_sorted_numba(sorted_train_x, 0.5 + q_mid)
    if lo_ext >= hi_ext or mid_lo > mid_hi:
        return np.nan

    test_x = x_arr[test_start:end]
    test_y = y_arr[test_start:end]

    bot_mask = test_x < lo_ext
    top_mask = test_x > hi_ext
    mid_mask = (test_x >= mid_lo) & (test_x <= mid_hi)

    bot_count = int(np.sum(bot_mask))
    top_count = int(np.sum(top_mask))
    mid_count = int(np.sum(mid_mask))

    if (
        bot_count < min_bucket_size
        or top_count < min_bucket_size
        or mid_count < min_bucket_size
    ):
        return np.nan

    top_y = test_y[top_mask]
    bot_y = test_y[bot_mask]
    mid_y = test_y[mid_mask]

    mu_top = _bucket_stat_numba(top_y, stat_code, clip_q)
    mu_bot = _bucket_stat_numba(bot_y, stat_code, clip_q)
    mu_mid = _bucket_stat_numba(mid_y, stat_code, clip_q)

    raw = np.abs(mu_top - mu_mid) + np.abs(mu_bot - mu_mid)

    # expected |diff of means| under null ~ sqrt(2/pi) * std * sqrt(1/n1 + 1/n2)
    noise = np.sqrt(2 / np.pi) * np.std(test_y) * (
        np.sqrt(1.0 / top_count + 1.0 / mid_count) +
        np.sqrt(1.0 / bot_count + 1.0 / mid_count)
    )

    score = raw - noise
    return 0.0 if (score <= 0.0 or not np.isfinite(score)) else score


@njit(cache=True, nogil=True)
def _extremes_vs_mid_ir_oof_numba(
    x_arr,
    y_arr,
    segments_count,
    train_frac,
    gap,
    q_ext,
    q_mid,
    stat_code,
    clip_q,
    min_bucket_size,
    min_valid_segments,
):
    n = x_arr.shape[0]
    k = int(segments_count)
    gap_val = int(gap)
    min_bucket = int(min_bucket_size)
    min_valid = int(min_valid_segments)
    if min_valid < 2:
        min_valid = 2

    scores = np.empty(k, dtype=np.float64)
    valid_count = 0

    for seg_idx in range(k):
        start = (seg_idx * n) // k
        end = ((seg_idx + 1) * n) // k

        seg_score = _segment_score_extremes_vs_mid_oof_numba(
            x_arr=x_arr,
            y_arr=y_arr,
            start=start,
            end=end,
            train_frac=train_frac,
            gap=gap_val,
            q_ext=q_ext,
            q_mid=q_mid,
            stat_code=stat_code,
            clip_q=clip_q,
            min_bucket_size=min_bucket,
        )
        if np.isnan(seg_score):
            continue

        scores[valid_count] = seg_score
        valid_count += 1

    if valid_count < min_valid:
        return 0.0

    mean_score = 0.0
    for i in range(valid_count):
        mean_score += scores[i]
    mean_score /= valid_count
    if not np.isfinite(mean_score):
        return 0.0

    var_sum = 0.0
    for i in range(valid_count):
        d = scores[i] - mean_score
        var_sum += d * d
    std_score = np.sqrt(var_sum / (valid_count - 1))
    # print(f"valid_count={valid_count}, mean_score={mean_score:.6f}, std_score={std_score:.6f}")
    if not np.isfinite(std_score) or std_score == 0.0:
        return 0.0

    ir = mean_score / std_score
    if not np.isfinite(ir) or ir < 0.0:
        return 0.0
    return ir


def extremes_vs_mid_ir_oof(
    x,
    y,
    segments_count=12,
    train_frac=0.80,
    gap=1500,
    q_ext=0.10,
    q_mid=0.10,
    stat="mean_clip",
    clip_q=0.01,
    min_bucket_size=50,
    min_valid_segments=2,
):
    stat_txt = str(stat).strip().lower()
    if stat_txt == "mean_clip":
        stat_code = 0
    elif stat_txt == "median":
        stat_code = 1
    else:
        raise ValueError(f"Unsupported stat='{stat}'. Use 'mean_clip' or 'median'.")

    segments_count_val = int(segments_count)
    train_frac_val = float(train_frac)
    gap_val = int(gap)
    q_ext_val = float(q_ext)
    q_mid_val = float(q_mid)
    clip_q_val = float(clip_q)
    min_bucket_size_val = int(min_bucket_size)
    min_valid_segments_val = int(min_valid_segments)

    if segments_count_val < 1:
        return 0.0
    if not (0.0 < train_frac_val < 1.0):
        return 0.0
    if gap_val < 0:
        return 0.0
    if not (0.0 < q_ext_val < 0.5):
        return 0.0
    if not (0.0 < q_mid_val < 0.5):
        return 0.0
    if not (0.5 - q_mid_val > q_ext_val):
        return 0.0
    if not (0.0 <= clip_q_val < 0.5):
        return 0.0
    if min_bucket_size_val < 1:
        return 0.0
    if min_valid_segments_val < 1:
        return 0.0

    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr.shape[0] != y_arr.shape[0] or x_arr.shape[0] < 3:
        return 0.0

    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(finite_mask):
        return 0.0
    if not np.all(finite_mask):
        x_arr = x_arr[finite_mask]
        y_arr = y_arr[finite_mask]
        if x_arr.shape[0] < 3:
            return 0.0

    return float(
        _extremes_vs_mid_ir_oof_numba(
            x_arr=x_arr,
            y_arr=y_arr,
            segments_count=segments_count_val,
            train_frac=train_frac_val,
            gap=gap_val,
            q_ext=q_ext_val,
            q_mid=q_mid_val,
            stat_code=int(stat_code),
            clip_q=clip_q_val,
            min_bucket_size=min_bucket_size_val,
            min_valid_segments=min_valid_segments_val,
        )
    )


def _safe_positive_int(value, field_name):
    out = int(value)
    if out <= 0:
        raise ValueError(f"{field_name} must be > 0, got: {value}")
    return out


def normalize_indicators(indicators_cfg):
    if isinstance(indicators_cfg, dict):
        names = [str(k) for k in indicators_cfg.keys()]
    elif isinstance(indicators_cfg, list):
        names = [str(v) for v in indicators_cfg]
    else:
        raise ValueError(
            "interval.indicators must be either dict (indicator names as keys) "
            "or list[str]."
        )

    names = sorted(set(names))
    if not names:
        raise ValueError("interval.indicators cannot be empty.")
    return names


def resolve_base_pop_size(interval_cfg, pair_cfg, indicators_cfg):
    base = interval_cfg.get("base_pop_size", pair_cfg.get("base_pop_size"))
    if base is not None:
        return _safe_positive_int(base, "base_pop_size")

    raise ValueError(
        "Missing base_pop_size in fit indicators config. "
        "Set it on pair or interval level."
    )


def _variable_score(var):
    if isinstance(var, Choice):
        options_count = len(getattr(var, "options", []) or [])
        if options_count <= 1:
            return 0.0
        return math.log2(float(options_count))

    if isinstance(var, Integer):
        bounds = getattr(var, "bounds", None)
        if not bounds or len(bounds) != 2:
            return 1.0
        low, high = int(bounds[0]), int(bounds[1])
        count = max(1, high - low + 1)
        if count <= 1:
            return 0.0
        return math.log2(float(count))

    if isinstance(var, Real):
        return float(REAL_VAR_WEIGHT)

    if isinstance(var, Binary):
        return float(BINARY_VAR_WEIGHT)

    return 1.0


def indicator_space_score(
    indicator_name,
    problem_map,
):
    if indicator_name not in problem_map:
        raise KeyError(f"Unsupported indicator in problem_map: {indicator_name}")

    problem = problem_map[indicator_name]()
    vars_map = getattr(problem, "vars", None)
    if not isinstance(vars_map, dict) or not vars_map:
        raise ValueError(
            f"Indicator {indicator_name} has no variable definition in problem."
        )

    return sum(_variable_score(var) for var in vars_map.values())


def compute_indicator_pop_sizes(
    indicator_names,
    problem_map,
    base_pop_size,
):
    base = _safe_positive_int(int(base_pop_size), "base_pop_size")
    names = sorted(set(indicator_names))

    raw_scores = {
        name: indicator_space_score(
            name,
            problem_map=problem_map,
        )
        for name in names
    }
    mean_score = sum(raw_scores.values()) / len(raw_scores)
    if mean_score <= 0:
        raise ValueError("Invalid indicator space score mean <= 0.")

    result = {}
    for name in names:
        adjusted = float(base) * (raw_scores[name] / mean_score)
        result[name] = max(1, int(round(adjusted)))
    return result
