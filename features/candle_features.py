from functools import lru_cache

import numpy as np
import pandas as pd
import talib
from talib import abstract as talib_abstract
from numba import njit
from pandas.tseries.frequencies import to_offset
from project_config import load_modeling_profile

RAW_OHLCV_COLS = ("Open", "High", "Low", "Close", "Volume")
_CANONICAL_BODY_RANGE_COL = "candle_body_pressure"
MULTI_INTERVAL_CANDLE_DERIVED_COLS = (
    "candle_signed_vol",
    "candle_up_down_vol_log_ratio",
    "candle_wick_asym",
    "candle_close_location_value",
    _CANONICAL_BODY_RANGE_COL,
)
ALL_CANDLE_DERIVED_COLS = (
    "candle_ret_co",
    "candle_range_ho",
    "candle_log_volume",
    "candle_body_abs_open",
    _CANONICAL_BODY_RANGE_COL,
    "candle_signed_vol",
    "candle_up_down_vol_log_ratio",
    "candle_wick_asym",
    "candle_close_location_value",
)
BASE_CANDLE_INTERVAL_LABEL = "1m"
CONFIGURABLE_INTERVAL_DERIVED_BASE_COLS = {
    "5m": ALL_CANDLE_DERIVED_COLS,
    "15m": MULTI_INTERVAL_CANDLE_DERIVED_COLS,
    "30m": MULTI_INTERVAL_CANDLE_DERIVED_COLS,
    "1h": MULTI_INTERVAL_CANDLE_DERIVED_COLS,
    "4h": MULTI_INTERVAL_CANDLE_DERIVED_COLS,
    "1d": MULTI_INTERVAL_CANDLE_DERIVED_COLS,
}


def _derived_feature_col(base_col):
    return f"{base_col}_{BASE_CANDLE_INTERVAL_LABEL}"


def _derived_lag_feature_col(base_col, lag):
    return f"{base_col}_{BASE_CANDLE_INTERVAL_LABEL}_lag{int(lag)}"


def _interval_derived_lag_feature_col(base_col, interval_label, lag):
    return f"{base_col}_{interval_label}_lag{int(lag)}"


def _normalize_candle_interval_lag_config(
    raw_config,
    *,
    source_label="candle_streak_intervals",
):
    if not isinstance(raw_config, dict) or not raw_config:
        raise ValueError(f"{source_label} must be a non-empty JSON object.")

    normalized = {}
    for raw_interval, raw_lag_count in raw_config.items():
        interval_label = str(raw_interval).strip()
        if not interval_label:
            raise ValueError(f"{source_label} cannot contain empty interval labels.")
        if interval_label in normalized:
            raise ValueError(
                f"{source_label} contains duplicate interval label: {interval_label!r}."
            )
        if interval_label not in INTERVAL_TO_RULE:
            supported = ", ".join(sorted(INTERVAL_TO_RULE.keys()))
            raise ValueError(
                f"Unsupported candle interval in {source_label}: {interval_label!r}. "
                f"Supported: {supported}"
            )
        if isinstance(raw_lag_count, bool) or not isinstance(raw_lag_count, int):
            raise ValueError(
                f"{source_label}[{interval_label!r}] must be an integer, got "
                f"{raw_lag_count!r}."
            )
        if raw_lag_count < 0:
            raise ValueError(
                f"{source_label}[{interval_label!r}] must be >= 0, got {raw_lag_count}."
            )
        normalized[interval_label] = int(raw_lag_count)
    return normalized


@lru_cache(maxsize=1)
def _load_configured_candle_interval_lag_counts():
    modeling_profile = load_modeling_profile()
    return _normalize_candle_interval_lag_config(
        modeling_profile.get("candle_streak_intervals"),
        source_label="modeling.candle_streak_intervals",
    )


def _iter_configured_interval_derived_specs(interval_lag_counts):
    for interval_label, lag_count in interval_lag_counts.items():
        if interval_label == BASE_CANDLE_INTERVAL_LABEL or lag_count <= 0:
            continue
        base_cols = CONFIGURABLE_INTERVAL_DERIVED_BASE_COLS.get(interval_label)
        if base_cols is None:
            continue
        for base_col in base_cols:
            for lag in range(1, int(lag_count) + 1):
                yield interval_label, base_col, lag


CANDLE_PATTERN_COLS = tuple(talib.get_function_groups().get("Pattern Recognition", ()))
OPENED_COL = "Opened"
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOLUME_COL = "Volume"
STREAK_FEATURE_PREFIX = "candle_streak_"

# Standard Binance kline intervals.
INTERVAL_TO_RULE = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1D",
    "3d": "3D",
    "1w": "1W-MON",
    "1M": "1MS",
}
PATTERN_INTERVAL_TO_RULE = {
    "5m": INTERVAL_TO_RULE["5m"],
}
DIRECT_PATTERN_INTERVAL_LABELS = ("1m",)
DEFAULT_PATTERN_INTERVAL_LABELS = DIRECT_PATTERN_INTERVAL_LABELS + tuple(
    PATTERN_INTERVAL_TO_RULE.keys()
)
INTERVAL_CANDLE_PATTERN_COLS = tuple(
    f"{col}_{interval_label}"
    for interval_label in DEFAULT_PATTERN_INTERVAL_LABELS
    for col in CANDLE_PATTERN_COLS
)
DEFAULT_CANDLE_PATTERN_COLS = INTERVAL_CANDLE_PATTERN_COLS


def _build_candle_feature_catalog():
    interval_lag_counts = _load_configured_candle_interval_lag_counts()
    base_lag_count = int(interval_lag_counts.get(BASE_CANDLE_INTERVAL_LABEL, 0))

    direct_derived_cols = tuple(
        _derived_feature_col(base_col) for base_col in ALL_CANDLE_DERIVED_COLS
    )
    lagged_derived_cols = tuple(
        _derived_lag_feature_col(base_col, lag)
        for base_col in ALL_CANDLE_DERIVED_COLS
        for lag in range(1, base_lag_count + 1)
    )
    interval_derived_cols = tuple(
        _interval_derived_lag_feature_col(base_col, interval_label, lag)
        for interval_label, base_col, lag in _iter_configured_interval_derived_specs(
            interval_lag_counts
        )
    )
    candle_derived_cols = (
        direct_derived_cols + lagged_derived_cols + interval_derived_cols
    )
    candle_feature_cols = candle_derived_cols + DEFAULT_CANDLE_PATTERN_COLS
    supported_feature_cols = (
        candle_derived_cols + CANDLE_PATTERN_COLS + INTERVAL_CANDLE_PATTERN_COLS
    )

    derived_feature_spec_by_col = {
        _derived_feature_col(base_col): (base_col, None, 0)
        for base_col in ALL_CANDLE_DERIVED_COLS
    }
    derived_feature_spec_by_col.update(
        {
            _derived_lag_feature_col(base_col, lag): (base_col, None, lag)
            for base_col in ALL_CANDLE_DERIVED_COLS
            for lag in range(1, base_lag_count + 1)
        }
    )
    derived_feature_spec_by_col.update(
        {
            _interval_derived_lag_feature_col(base_col, interval_label, lag): (
                base_col,
                interval_label,
                lag,
            )
            for interval_label, base_col, lag in _iter_configured_interval_derived_specs(
                interval_lag_counts
            )
        }
    )
    return {
        "interval_lag_counts": dict(interval_lag_counts),
        "direct_derived_cols": direct_derived_cols,
        "lagged_derived_cols": lagged_derived_cols,
        "interval_derived_cols": interval_derived_cols,
        "candle_derived_cols": candle_derived_cols,
        "candle_feature_cols": candle_feature_cols,
        "supported_feature_cols": supported_feature_cols,
        "derived_feature_spec_by_col": derived_feature_spec_by_col,
    }


_EPS = 1e-12
_DERIVED_BASE_COL_TO_INDEX = {
    col: idx for idx, col in enumerate(ALL_CANDLE_DERIVED_COLS)
}
_PATTERN_FUNC_BY_COL = {col: getattr(talib, col) for col in CANDLE_PATTERN_COLS}
_PATTERN_LOOKBACK_BY_COL = {
    col: int(talib_abstract.Function(col).lookback) + 1 for col in CANDLE_PATTERN_COLS
}
_PATTERN_FEATURE_SPEC_BY_COL = {col: (col, None) for col in CANDLE_PATTERN_COLS}
_PATTERN_FEATURE_SPEC_BY_COL.update(
    {
        f"{col}_{interval_label}": (col, interval_label)
        for interval_label in DEFAULT_PATTERN_INTERVAL_LABELS
        for col in CANDLE_PATTERN_COLS
    }
)
_MINUTE_NS = pd.Timedelta(minutes=1).value
_CANDLE_FEATURE_CATALOG = _build_candle_feature_catalog()
CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS = dict(
    _CANDLE_FEATURE_CATALOG["interval_lag_counts"]
)
DIRECT_CANDLE_DERIVED_COLS = _CANDLE_FEATURE_CATALOG["direct_derived_cols"]
LAGGED_CANDLE_DERIVED_COLS = _CANDLE_FEATURE_CATALOG["lagged_derived_cols"]
INTERVAL_CANDLE_DERIVED_COLS = _CANDLE_FEATURE_CATALOG["interval_derived_cols"]
CANDLE_DERIVED_COLS = _CANDLE_FEATURE_CATALOG["candle_derived_cols"]
CANDLE_FEATURE_COLS = _CANDLE_FEATURE_CATALOG["candle_feature_cols"]
SUPPORTED_CANDLE_FEATURE_COLS = _CANDLE_FEATURE_CATALOG["supported_feature_cols"]
_DERIVED_FEATURE_SPEC_BY_COL = _CANDLE_FEATURE_CATALOG["derived_feature_spec_by_col"]


def _dedupe_ordered(values):
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return tuple(out)


def resolve_candle_feature_cols(feature_cols=None):
    if feature_cols is None:
        return CANDLE_FEATURE_COLS

    requested = [str(col).strip() for col in feature_cols]
    if not requested:
        raise ValueError("feature_cols cannot be empty.")

    requested = _dedupe_ordered(requested)
    unsupported = [col for col in requested if col not in SUPPORTED_CANDLE_FEATURE_COLS]
    if unsupported:
        supported = ", ".join(SUPPORTED_CANDLE_FEATURE_COLS)
        raise ValueError(
            "Unsupported candle feature columns: "
            + ", ".join(unsupported)
            + f". Supported: {supported}"
        )
    return requested


def resolve_candle_derived_feature_cols(feature_cols=None):
    if feature_cols is not None:
        feature_cols = tuple(feature_cols)
        if not feature_cols:
            return ()
    return tuple(
        col
        for col in resolve_candle_feature_cols(feature_cols)
        if col in _DERIVED_FEATURE_SPEC_BY_COL
    )


def _split_derived_feature_cols(feature_cols=None):
    direct_feature_cols = []
    lag_feature_cols = {}
    interval_to_feature_cols = {}
    for feature_col in resolve_candle_derived_feature_cols(feature_cols):
        base_col, interval_label, lag = _DERIVED_FEATURE_SPEC_BY_COL[feature_col]
        if interval_label is None and lag == 0:
            direct_feature_cols.append(feature_col)
            continue
        if interval_label is None:
            lag_feature_cols.setdefault(lag, []).append(feature_col)
            continue
        interval_to_feature_cols.setdefault(interval_label, []).append(feature_col)
    return (
        tuple(direct_feature_cols),
        {
            int(lag): tuple(feature_cols)
            for lag, feature_cols in lag_feature_cols.items()
        },
        {
            interval_label: tuple(feature_cols)
            for interval_label, feature_cols in interval_to_feature_cols.items()
        },
    )


def resolve_candle_pattern_feature_cols(pattern_cols=None):
    if pattern_cols is not None:
        pattern_cols = tuple(pattern_cols)
        if not pattern_cols:
            return ()
    return tuple(
        col
        for col in resolve_candle_feature_cols(pattern_cols)
        if col in _PATTERN_FEATURE_SPEC_BY_COL
    )


def _split_pattern_feature_cols(pattern_cols=None):
    base_pattern_cols = []
    direct_pattern_feature_cols = []
    interval_to_feature_cols = {}
    for feature_col in resolve_candle_pattern_feature_cols(pattern_cols):
        base_col, interval_label = _PATTERN_FEATURE_SPEC_BY_COL[feature_col]
        if interval_label is None:
            base_pattern_cols.append(base_col)
            continue
        if interval_label in DIRECT_PATTERN_INTERVAL_LABELS:
            direct_pattern_feature_cols.append(feature_col)
            continue
        interval_to_feature_cols.setdefault(interval_label, []).append(feature_col)
    return (
        tuple(base_pattern_cols),
        tuple(direct_pattern_feature_cols),
        {
            interval_label: tuple(feature_cols)
            for interval_label, feature_cols in interval_to_feature_cols.items()
        },
    )


def _safe_divide(num, den):
    den_arr = np.asarray(den, dtype=np.float64)
    safe_den = np.where(np.abs(den_arr) > _EPS, den_arr, np.nan)
    return np.asarray(num, dtype=np.float64) / safe_den


def _compute_up_down_volume_arrays(open_, close, volume):
    body = np.asarray(close, dtype=np.float64) - np.asarray(open_, dtype=np.float64)
    volume_arr = np.asarray(volume, dtype=np.float64)
    up_volume = np.where(body > 0.0, volume_arr, 0.0)
    down_volume = np.where(body < 0.0, volume_arr, 0.0)
    return up_volume, down_volume


@njit(cache=True)
def _compute_derived_feature_matrix(open_arr, high_arr, low_arr, close_arr, volume_arr):
    n = len(open_arr)
    out = np.empty((n, len(ALL_CANDLE_DERIVED_COLS)), dtype=np.float64)
    for i in range(n):
        open_value = float(open_arr[i])
        high_value = float(high_arr[i])
        low_value = float(low_arr[i])
        close_value = float(close_arr[i])
        volume_value = float(volume_arr[i])
        body = close_value - open_value
        range_hl = high_value - low_value
        upper_wick = high_value - max(open_value, close_value)
        lower_wick = min(open_value, close_value) - low_value
        up_volume = volume_value if body > 0.0 else 0.0
        down_volume = volume_value if body < 0.0 else 0.0
        safe_open = open_value if abs(open_value) > _EPS else np.nan
        range_eps = range_hl + _EPS

        out[i, 0] = body / safe_open
        out[i, 1] = range_hl / safe_open
        out[i, 2] = np.log(volume_value if volume_value > _EPS else _EPS)
        out[i, 3] = abs(body) / safe_open
        out[i, 4] = body / range_eps
        out[i, 5] = volume_value * np.sign(body)
        out[i, 6] = np.log1p(up_volume) - np.log1p(down_volume)
        out[i, 7] = (upper_wick - lower_wick) / range_eps
        out[i, 8] = (close_value - low_value) / range_eps
    return out


@njit(cache=True)
def _compute_derived_feature_matrix_with_volume_split(
    open_arr,
    high_arr,
    low_arr,
    close_arr,
    volume_arr,
    up_volume_arr,
    down_volume_arr,
):
    n = len(open_arr)
    out = np.empty((n, len(ALL_CANDLE_DERIVED_COLS)), dtype=np.float64)
    for i in range(n):
        open_value = float(open_arr[i])
        high_value = float(high_arr[i])
        low_value = float(low_arr[i])
        close_value = float(close_arr[i])
        volume_value = float(volume_arr[i])
        up_volume_value = float(up_volume_arr[i])
        down_volume_value = float(down_volume_arr[i])
        body = close_value - open_value
        range_hl = high_value - low_value
        upper_wick = high_value - max(open_value, close_value)
        lower_wick = min(open_value, close_value) - low_value
        safe_open = open_value if abs(open_value) > _EPS else np.nan
        range_eps = range_hl + _EPS

        out[i, 0] = body / safe_open
        out[i, 1] = range_hl / safe_open
        out[i, 2] = np.log(volume_value if volume_value > _EPS else _EPS)
        out[i, 3] = abs(body) / safe_open
        out[i, 4] = body / range_eps
        out[i, 5] = volume_value * np.sign(body)
        out[i, 6] = np.log1p(up_volume_value) - np.log1p(down_volume_value)
        out[i, 7] = (upper_wick - lower_wick) / range_eps
        out[i, 8] = (close_value - low_value) / range_eps
    return out


def build_candle_derived_features_from_series(
    open_, high, low, close, volume, *, up_volume=None, down_volume=None
):
    open_arr = np.asarray(open_, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)
    body = close_arr - open_arr
    range_hl = high_arr - low_arr
    volume_arr = np.asarray(volume, dtype=np.float64)
    upper_wick = high_arr - np.maximum(open_arr, close_arr)
    lower_wick = np.minimum(open_arr, close_arr) - low_arr
    if up_volume is None or down_volume is None:
        up_volume_arr, down_volume_arr = _compute_up_down_volume_arrays(
            open_arr,
            close_arr,
            volume_arr,
        )
    else:
        up_volume_arr = np.asarray(up_volume, dtype=np.float64)
        down_volume_arr = np.asarray(down_volume, dtype=np.float64)
    range_eps = range_hl + _EPS

    out = {
        "candle_ret_co": _safe_divide(body, open_arr),
        "candle_range_ho": _safe_divide(range_hl, open_arr),
        "candle_log_volume": np.log(np.clip(volume_arr, _EPS, None)),
        "candle_body_abs_open": _safe_divide(np.abs(body), open_arr),
        "candle_body_pressure": body / range_eps,
        "candle_signed_vol": volume_arr * np.sign(body),
        "candle_up_down_vol_log_ratio": np.log1p(up_volume_arr)
        - np.log1p(down_volume_arr),
        "candle_wick_asym": (upper_wick - lower_wick) / range_eps,
        "candle_close_location_value": (close_arr - low_arr) / range_eps,
    }
    return out


def build_candle_pattern_features_from_series(
    open_, high, low, close, pattern_cols=None
):
    (
        selected_pattern_cols,
        direct_pattern_feature_cols,
        _,
    ) = _split_pattern_feature_cols(pattern_cols)
    direct_pattern_cols = tuple(
        _PATTERN_FEATURE_SPEC_BY_COL[feature_col][0]
        for feature_col in direct_pattern_feature_cols
    )
    requested_pattern_cols = _dedupe_ordered(
        (*selected_pattern_cols, *direct_pattern_cols)
    )
    if not requested_pattern_cols:
        return {}

    open_arr = np.asarray(open_, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)
    if not (len(open_arr) == len(high_arr) == len(low_arr) == len(close_arr)):
        raise ValueError("OHLC inputs must have the same length for candle patterns.")

    out = {}
    for col in requested_pattern_cols:
        pattern_fn = _PATTERN_FUNC_BY_COL[col]
        out[col] = np.asarray(
            pattern_fn(open_arr, high_arr, low_arr, close_arr), dtype=np.int32
        )
    return out


def _resample_complete_interval_frame(
    base_df,
    rule,
    required_cols,
    agg_spec,
    dropna_cols,
    count_col,
    opened_col=OPENED_COL,
    context="interval candles",
):
    missing = [col for col in required_cols if col not in base_df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for {context}: " + ", ".join(missing)
        )

    work = base_df[list(required_cols)].copy()
    work[opened_col] = pd.to_datetime(work[opened_col], errors="raise")
    work = work.sort_values(opened_col)
    if work[opened_col].duplicated().any():
        dup_count = int(work[opened_col].duplicated().sum())
        raise ValueError(f"Duplicate {opened_col} values found: {dup_count}")

    work = work.set_index(opened_col)
    agg = work.resample(rule, label="left", closed="left").agg(agg_spec)
    agg["__count"] = work[count_col].resample(rule, label="left", closed="left").size()
    agg = agg.dropna(subset=list(dropna_cols)).copy()
    if agg.empty:
        return agg

    bucket_start = pd.DatetimeIndex(agg.index)
    bucket_end = bucket_start + to_offset(rule)
    expected_count = np.asarray(
        (bucket_end - bucket_start) / pd.Timedelta(minutes=1),
        dtype=np.float64,
    )

    # Only expose fully built HTF candles. This uses the actual bucket width,
    # so the last candle is kept when it is complete and dropped otherwise.
    is_complete = agg["__count"].to_numpy(dtype=np.float64) == expected_count
    agg = agg.loc[is_complete].copy()
    if agg.empty:
        return agg

    bucket_end = bucket_end[is_complete]
    agg["__close_opened"] = bucket_end - pd.Timedelta(minutes=1)
    return agg


def _merge_interval_events(base_df, events, feature_cols, opened_col=OPENED_COL):
    if not feature_cols:
        return pd.DataFrame(index=base_df.index)

    base_opened = base_df[[opened_col]].copy()
    base_opened[opened_col] = pd.to_datetime(
        base_opened[opened_col], errors="raise", utc=True
    )
    base_merge_key = pd.DatetimeIndex(base_opened[opened_col]).as_unit("ns")
    if base_merge_key.hasnans:
        raise ValueError(f"{opened_col} contains NaT values in base_df.")
    base_opened["__merge_opened_ns"] = base_merge_key.asi8
    base_opened = base_opened.sort_values("__merge_opened_ns").reset_index()

    events = events.copy()
    events[opened_col] = pd.to_datetime(events[opened_col], errors="coerce", utc=True)
    events = events.dropna(subset=[opened_col])
    event_merge_key = pd.DatetimeIndex(events[opened_col]).as_unit("ns")
    events["__merge_opened_ns"] = event_merge_key.asi8
    events = events.sort_values("__merge_opened_ns").reset_index(drop=True)
    merged = pd.merge_asof(
        base_opened,
        events,
        on="__merge_opened_ns",
        direction="backward",
    )
    out = merged[list(feature_cols)]
    out.index = merged["index"].to_numpy()
    return out.reindex(base_df.index)


def _empty_interval_pattern_frame(base_index, feature_cols):
    return pd.DataFrame(
        {
            feature_col: np.zeros(len(base_index), dtype=np.int32)
            for feature_col in feature_cols
        },
        index=base_index,
    )


def _empty_interval_derived_frame(base_index, feature_cols, float_dtype=np.float64):
    return pd.DataFrame(
        {
            feature_col: np.full(len(base_index), np.nan, dtype=float_dtype)
            for feature_col in feature_cols
        },
        index=base_index,
    )


def _compute_interval_patterns(
    base_df,
    interval_label,
    rule,
    feature_cols,
    opened_col=OPENED_COL,
    open_col=OPEN_COL,
    high_col=HIGH_COL,
    low_col=LOW_COL,
    close_col=CLOSE_COL,
):
    if not feature_cols:
        return pd.DataFrame(index=base_df.index)

    agg = _resample_complete_interval_frame(
        base_df=base_df,
        rule=rule,
        required_cols=[opened_col, open_col, high_col, low_col, close_col],
        agg_spec={
            open_col: "first",
            high_col: "max",
            low_col: "min",
            close_col: "last",
        },
        dropna_cols=[open_col, high_col, low_col, close_col],
        count_col=open_col,
        opened_col=opened_col,
        context="interval candle patterns",
    )
    if agg.empty:
        return _empty_interval_pattern_frame(base_df.index, feature_cols)

    base_pattern_cols = tuple(
        _PATTERN_FEATURE_SPEC_BY_COL[feature_col][0] for feature_col in feature_cols
    )
    patterns = build_candle_pattern_features_from_series(
        open_=agg[open_col].to_numpy(dtype=np.float64, copy=False),
        high=agg[high_col].to_numpy(dtype=np.float64, copy=False),
        low=agg[low_col].to_numpy(dtype=np.float64, copy=False),
        close=agg[close_col].to_numpy(dtype=np.float64, copy=False),
        pattern_cols=base_pattern_cols,
    )

    events = pd.DataFrame(
        {
            opened_col: pd.to_datetime(
                agg["__close_opened"], errors="coerce"
            ).to_numpy(),
            **{
                feature_col: patterns[_PATTERN_FEATURE_SPEC_BY_COL[feature_col][0]]
                for feature_col in feature_cols
            },
        }
    )
    out = (
        _merge_interval_events(
            base_df=base_df,
            events=events,
            feature_cols=feature_cols,
            opened_col=opened_col,
        )
        .fillna(0)
        .astype(np.int32, copy=False)
    )
    return out.reindex(base_df.index)


def _compute_interval_derived_features(
    base_df,
    interval_label,
    rule,
    feature_cols,
    float_dtype=np.float64,
    opened_col=OPENED_COL,
    open_col=OPEN_COL,
    high_col=HIGH_COL,
    low_col=LOW_COL,
    close_col=CLOSE_COL,
    volume_col=VOLUME_COL,
):
    if not feature_cols:
        return pd.DataFrame(index=base_df.index)

    needs_underlying_volume_split = any(
        _DERIVED_FEATURE_SPEC_BY_COL[feature_col][0]
        == "candle_up_down_vol_log_ratio"
        for feature_col in feature_cols
    )
    interval_base = base_df[
        [
            opened_col,
            open_col,
            high_col,
            low_col,
            close_col,
            volume_col,
        ]
    ].copy()
    required_cols = [
        opened_col,
        open_col,
        high_col,
        low_col,
        close_col,
        volume_col,
    ]
    agg_spec = {
        open_col: "first",
        high_col: "max",
        low_col: "min",
        close_col: "last",
        volume_col: "sum",
    }
    if needs_underlying_volume_split:
        up_volume_arr, down_volume_arr = _compute_up_down_volume_arrays(
            interval_base[open_col].to_numpy(dtype=np.float64, copy=False),
            interval_base[close_col].to_numpy(dtype=np.float64, copy=False),
            interval_base[volume_col].to_numpy(dtype=np.float64, copy=False),
        )
        interval_base["__up_volume"] = up_volume_arr
        interval_base["__down_volume"] = down_volume_arr
        required_cols.extend(["__up_volume", "__down_volume"])
        agg_spec["__up_volume"] = "sum"
        agg_spec["__down_volume"] = "sum"

    agg = _resample_complete_interval_frame(
        base_df=interval_base,
        rule=rule,
        required_cols=required_cols,
        agg_spec=agg_spec,
        dropna_cols=[open_col, high_col, low_col, close_col, volume_col],
        count_col=open_col,
        opened_col=opened_col,
        context=f"{interval_label} candle derived features",
    )
    if agg.empty:
        return _empty_interval_derived_frame(
            base_df.index,
            feature_cols,
            float_dtype=float_dtype,
        )

    derived_kwargs = {}
    if needs_underlying_volume_split:
        derived_kwargs = {
            "up_volume": agg["__up_volume"].to_numpy(dtype=np.float64, copy=False),
            "down_volume": agg["__down_volume"].to_numpy(dtype=np.float64, copy=False),
        }

    derived = build_candle_derived_features_from_series(
        open_=agg[open_col].to_numpy(dtype=np.float64, copy=False),
        high=agg[high_col].to_numpy(dtype=np.float64, copy=False),
        low=agg[low_col].to_numpy(dtype=np.float64, copy=False),
        close=agg[close_col].to_numpy(dtype=np.float64, copy=False),
        volume=agg[volume_col].to_numpy(dtype=np.float64, copy=False),
        **derived_kwargs,
    )
    derived_df = pd.DataFrame(derived, index=agg.index)

    event_values = {
        opened_col: pd.to_datetime(agg["__close_opened"], errors="coerce").to_numpy(),
    }
    for feature_col in feature_cols:
        base_col, _, lag = _DERIVED_FEATURE_SPEC_BY_COL[feature_col]
        # For HTF features lag1 means the latest fully closed HTF candle visible
        # at the current 1m row, lag2 the previous HTF candle, and so on.
        event_values[feature_col] = (
            derived_df[base_col]
            .shift(int(lag) - 1)
            .to_numpy(
                dtype=float_dtype,
                copy=False,
            )
        )
    events = pd.DataFrame(event_values)

    return _merge_interval_events(
        base_df=base_df,
        events=events,
        feature_cols=feature_cols,
        opened_col=opened_col,
    )


def add_candle_derived_features(df, feature_cols=None, float_dtype=np.float64):
    missing = [col for col in RAW_OHLCV_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required OHLCV columns for candle features: " + ", ".join(missing)
        )

    selected_cols = resolve_candle_feature_cols(feature_cols)
    open_arr = df[OPEN_COL].to_numpy(dtype=np.float64, copy=False)
    high_arr = df[HIGH_COL].to_numpy(dtype=np.float64, copy=False)
    low_arr = df[LOW_COL].to_numpy(dtype=np.float64, copy=False)
    close_arr = df[CLOSE_COL].to_numpy(dtype=np.float64, copy=False)
    volume_arr = df[VOLUME_COL].to_numpy(dtype=np.float64, copy=False)

    feature_values = {}
    (
        direct_derived_feature_cols,
        lag_to_derived_feature_cols,
        interval_to_derived_feature_cols,
    ) = _split_derived_feature_cols(selected_cols)
    if (
        direct_derived_feature_cols
        or lag_to_derived_feature_cols
        or interval_to_derived_feature_cols
    ):
        derived = build_candle_derived_features_from_series(
            open_=open_arr,
            high=high_arr,
            low=low_arr,
            close=close_arr,
            volume=volume_arr,
        )
        derived_df = pd.DataFrame(derived, index=df.index)
        for feature_col in direct_derived_feature_cols:
            base_col, _, _ = _DERIVED_FEATURE_SPEC_BY_COL[feature_col]
            feature_values[feature_col] = derived_df[base_col].to_numpy(
                dtype=float_dtype,
                copy=False,
            )
        for lag, lag_feature_cols in lag_to_derived_feature_cols.items():
            for feature_col in lag_feature_cols:
                base_col, _, _ = _DERIVED_FEATURE_SPEC_BY_COL[feature_col]
                feature_values[feature_col] = (
                    derived_df[base_col]
                    .shift(int(lag))
                    .to_numpy(
                        dtype=float_dtype,
                        copy=False,
                    )
                )

    if interval_to_derived_feature_cols:
        if OPENED_COL not in df.columns:
            raise ValueError(
                f"Missing required column for interval candle derived features: {OPENED_COL}"
            )
        derived_base = df[
            [OPENED_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL]
        ].copy()
        for (
            interval_label,
            interval_feature_cols,
        ) in interval_to_derived_feature_cols.items():
            interval_rule = INTERVAL_TO_RULE[interval_label]
            interval_derived = _compute_interval_derived_features(
                base_df=derived_base,
                interval_label=interval_label,
                rule=interval_rule,
                feature_cols=interval_feature_cols,
                float_dtype=float_dtype,
                opened_col=OPENED_COL,
                open_col=OPEN_COL,
                high_col=HIGH_COL,
                low_col=LOW_COL,
                close_col=CLOSE_COL,
                volume_col=VOLUME_COL,
            )
            for col in interval_feature_cols:
                feature_values[col] = interval_derived[col].to_numpy(
                    dtype=float_dtype,
                    copy=False,
                )

    selected_pattern_cols = resolve_candle_pattern_feature_cols(selected_cols)
    (
        base_pattern_cols,
        direct_pattern_feature_cols,
        interval_to_pattern_feature_cols,
    ) = _split_pattern_feature_cols(selected_pattern_cols)
    direct_pattern_cols = tuple(
        _PATTERN_FEATURE_SPEC_BY_COL[feature_col][0]
        for feature_col in direct_pattern_feature_cols
    )
    direct_compute_cols = _dedupe_ordered((*base_pattern_cols, *direct_pattern_cols))
    if direct_compute_cols:
        patterns = build_candle_pattern_features_from_series(
            open_=open_arr,
            high=high_arr,
            low=low_arr,
            close=close_arr,
            pattern_cols=direct_compute_cols,
        )
        for col in base_pattern_cols:
            if col in patterns:
                feature_values[col] = patterns[col]
        for feature_col in direct_pattern_feature_cols:
            base_col = _PATTERN_FEATURE_SPEC_BY_COL[feature_col][0]
            if base_col in patterns:
                feature_values[feature_col] = patterns[base_col]

    if interval_to_pattern_feature_cols:
        if OPENED_COL not in df.columns:
            raise ValueError(
                f"Missing required column for interval candle patterns: {OPENED_COL}"
            )
        pattern_base = df[[OPENED_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]].copy()
        for (
            interval_label,
            interval_feature_cols,
        ) in interval_to_pattern_feature_cols.items():
            interval_rule = PATTERN_INTERVAL_TO_RULE[interval_label]
            interval_patterns = _compute_interval_patterns(
                base_df=pattern_base,
                interval_label=interval_label,
                rule=interval_rule,
                feature_cols=interval_feature_cols,
                opened_col=OPENED_COL,
                open_col=OPEN_COL,
                high_col=HIGH_COL,
                low_col=LOW_COL,
                close_col=CLOSE_COL,
            )
            for col in interval_feature_cols:
                feature_values[col] = interval_patterns[col].to_numpy(
                    dtype=np.int32, copy=False
                )

    if not feature_values:
        return df.copy(deep=False)

    feature_frame = pd.DataFrame(feature_values, index=df.index)
    duplicate_cols = [col for col in feature_frame.columns if col in df.columns]
    if duplicate_cols:
        base_df = df.drop(columns=duplicate_cols)
    else:
        base_df = df
    return pd.concat([base_df, feature_frame], axis=1, copy=False)


def build_latest_candle_derived_feature_dict(
    opened_values,
    open_values,
    high_values,
    low_values,
    close_values,
    volume_values,
    feature_cols=None,
):
    selected_cols = resolve_candle_derived_feature_cols(feature_cols)
    if not selected_cols:
        return {}
    if len(open_values) == 0:
        return {}

    base = pd.DataFrame(
        {
            OPENED_COL: pd.to_datetime(opened_values, errors="raise"),
            OPEN_COL: np.asarray(open_values, dtype=np.float64),
            HIGH_COL: np.asarray(high_values, dtype=np.float64),
            LOW_COL: np.asarray(low_values, dtype=np.float64),
            CLOSE_COL: np.asarray(close_values, dtype=np.float64),
            VOLUME_COL: np.asarray(volume_values, dtype=np.float64),
        }
    )
    derived_df = add_candle_derived_features(base, feature_cols=selected_cols)
    return {col: float(derived_df[col].iloc[-1]) for col in selected_cols}


def _compute_latest_complete_interval_derived_matrix_fast(
    opened_ns,
    open_values,
    high_values,
    low_values,
    close_values,
    volume_values,
    *,
    bucket_ns,
    expected_count,
    needed_complete,
):
    if needed_complete <= 0 or len(opened_ns) == 0:
        return np.empty((0, len(ALL_CANDLE_DERIVED_COLS)), dtype=np.float64)

    bucket_start_ns = (opened_ns // bucket_ns) * bucket_ns
    segment_breaks = np.empty(len(bucket_start_ns), dtype=bool)
    segment_breaks[0] = True
    segment_breaks[1:] = bucket_start_ns[1:] != bucket_start_ns[:-1]
    segment_starts = np.flatnonzero(segment_breaks)
    segment_ends = np.empty(len(segment_starts), dtype=np.int64)
    segment_ends[:-1] = segment_starts[1:]
    segment_ends[-1] = len(opened_ns)

    agg_open = []
    agg_high = []
    agg_low = []
    agg_close = []
    agg_volume = []
    agg_up_volume = []
    agg_down_volume = []
    for start, end in zip(segment_starts, segment_ends):
        if int(end - start) != int(expected_count):
            continue
        up_volume = 0.0
        down_volume = 0.0
        for pos in range(start, end):
            candle_body = float(close_values[pos]) - float(open_values[pos])
            candle_volume = float(volume_values[pos])
            if candle_body > 0.0:
                up_volume += candle_volume
            elif candle_body < 0.0:
                down_volume += candle_volume
        agg_open.append(float(open_values[start]))
        agg_high.append(float(np.max(high_values[start:end])))
        agg_low.append(float(np.min(low_values[start:end])))
        agg_close.append(float(close_values[end - 1]))
        agg_volume.append(float(np.sum(volume_values[start:end])))
        agg_up_volume.append(up_volume)
        agg_down_volume.append(down_volume)

    if len(agg_open) < needed_complete:
        return np.empty((0, len(ALL_CANDLE_DERIVED_COLS)), dtype=np.float64)

    take_slice = slice(len(agg_open) - needed_complete, len(agg_open))
    return _compute_derived_feature_matrix_with_volume_split(
        np.asarray(agg_open[take_slice], dtype=np.float64),
        np.asarray(agg_high[take_slice], dtype=np.float64),
        np.asarray(agg_low[take_slice], dtype=np.float64),
        np.asarray(agg_close[take_slice], dtype=np.float64),
        np.asarray(agg_volume[take_slice], dtype=np.float64),
        np.asarray(agg_up_volume[take_slice], dtype=np.float64),
        np.asarray(agg_down_volume[take_slice], dtype=np.float64),
    )


def build_latest_candle_derived_feature_dict_fast(
    opened_values,
    opened_ns_values,
    open_values,
    high_values,
    low_values,
    close_values,
    volume_values,
    feature_cols=None,
):
    selected_cols = resolve_candle_derived_feature_cols(feature_cols)
    if not selected_cols:
        return {}
    if len(open_values) == 0:
        return {}

    open_arr = np.asarray(open_values, dtype=np.float64)
    high_arr = np.asarray(high_values, dtype=np.float64)
    low_arr = np.asarray(low_values, dtype=np.float64)
    close_arr = np.asarray(close_values, dtype=np.float64)
    volume_arr = np.asarray(volume_values, dtype=np.float64)
    opened_ns = np.asarray(opened_ns_values, dtype=np.int64)

    if not (
        len(opened_ns)
        == len(open_arr)
        == len(high_arr)
        == len(low_arr)
        == len(close_arr)
        == len(volume_arr)
    ):
        return build_latest_candle_derived_feature_dict(
            opened_values=opened_values,
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            volume_values=volume_values,
            feature_cols=selected_cols,
        )

    if len(opened_ns) > 1 and np.any(np.diff(opened_ns) <= 0):
        return build_latest_candle_derived_feature_dict(
            opened_values=opened_values,
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            volume_values=volume_values,
            feature_cols=selected_cols,
        )

    out = {}
    (
        direct_derived_feature_cols,
        lag_to_derived_feature_cols,
        interval_to_derived_feature_cols,
    ) = _split_derived_feature_cols(selected_cols)

    max_direct_lag = max(lag_to_derived_feature_cols.keys(), default=0)
    direct_tail_len = max(1, max_direct_lag + 1)
    direct_matrix = _compute_derived_feature_matrix(
        open_arr[-direct_tail_len:],
        high_arr[-direct_tail_len:],
        low_arr[-direct_tail_len:],
        close_arr[-direct_tail_len:],
        volume_arr[-direct_tail_len:],
    )

    for feature_col in direct_derived_feature_cols:
        base_col, _, _ = _DERIVED_FEATURE_SPEC_BY_COL[feature_col]
        out[feature_col] = float(
            direct_matrix[-1, _DERIVED_BASE_COL_TO_INDEX[base_col]]
        )
    for lag, lag_feature_cols in lag_to_derived_feature_cols.items():
        if lag >= direct_matrix.shape[0]:
            return build_latest_candle_derived_feature_dict(
                opened_values=opened_values,
                open_values=open_values,
                high_values=high_values,
                low_values=low_values,
                close_values=close_values,
                volume_values=volume_values,
                feature_cols=selected_cols,
            )
        row = direct_matrix[-(int(lag) + 1)]
        for feature_col in lag_feature_cols:
            base_col, _, _ = _DERIVED_FEATURE_SPEC_BY_COL[feature_col]
            out[feature_col] = float(row[_DERIVED_BASE_COL_TO_INDEX[base_col]])

    for (
        interval_label,
        interval_feature_cols,
    ) in interval_to_derived_feature_cols.items():
        rule = INTERVAL_TO_RULE.get(interval_label)
        if rule is None:
            return build_latest_candle_derived_feature_dict(
                opened_values=opened_values,
                open_values=open_values,
                high_values=high_values,
                low_values=low_values,
                close_values=close_values,
                volume_values=volume_values,
                feature_cols=selected_cols,
            )
        offset = to_offset(rule)
        bucket_ns = int(offset.nanos)
        if bucket_ns <= 0 or bucket_ns % _MINUTE_NS != 0:
            return build_latest_candle_derived_feature_dict(
                opened_values=opened_values,
                open_values=open_values,
                high_values=high_values,
                low_values=low_values,
                close_values=close_values,
                volume_values=volume_values,
                feature_cols=selected_cols,
            )
        expected_count = int(bucket_ns // _MINUTE_NS)
        if expected_count <= 0:
            return build_latest_candle_derived_feature_dict(
                opened_values=opened_values,
                open_values=open_values,
                high_values=high_values,
                low_values=low_values,
                close_values=close_values,
                volume_values=volume_values,
                feature_cols=selected_cols,
            )

        max_interval_lag = max(
            int(_DERIVED_FEATURE_SPEC_BY_COL[feature_col][2])
            for feature_col in interval_feature_cols
        )
        tail_len = min(len(opened_ns), expected_count * (max_interval_lag + 2))
        interval_matrix = _compute_latest_complete_interval_derived_matrix_fast(
            opened_ns[-tail_len:],
            open_arr[-tail_len:],
            high_arr[-tail_len:],
            low_arr[-tail_len:],
            close_arr[-tail_len:],
            volume_arr[-tail_len:],
            bucket_ns=bucket_ns,
            expected_count=expected_count,
            needed_complete=max_interval_lag,
        )
        if interval_matrix.shape[0] < max_interval_lag:
            return build_latest_candle_derived_feature_dict(
                opened_values=opened_values,
                open_values=open_values,
                high_values=high_values,
                low_values=low_values,
                close_values=close_values,
                volume_values=volume_values,
                feature_cols=selected_cols,
            )
        for feature_col in interval_feature_cols:
            base_col, _, lag = _DERIVED_FEATURE_SPEC_BY_COL[feature_col]
            out[feature_col] = float(
                interval_matrix[-int(lag), _DERIVED_BASE_COL_TO_INDEX[base_col]]
            )

    return out


def build_latest_candle_pattern_feature_dict(
    opened_values, open_values, high_values, low_values, close_values, pattern_cols=None
):
    selected_pattern_cols = resolve_candle_pattern_feature_cols(pattern_cols)
    if not selected_pattern_cols:
        return {}
    if len(open_values) == 0:
        return {}

    (
        base_pattern_cols,
        direct_pattern_feature_cols,
        interval_to_pattern_feature_cols,
    ) = _split_pattern_feature_cols(selected_pattern_cols)
    out = {}

    direct_pattern_cols = tuple(
        _PATTERN_FEATURE_SPEC_BY_COL[feature_col][0]
        for feature_col in direct_pattern_feature_cols
    )
    direct_compute_cols = _dedupe_ordered((*base_pattern_cols, *direct_pattern_cols))
    if direct_compute_cols:
        pattern_values = build_candle_pattern_features_from_series(
            open_=open_values,
            high=high_values,
            low=low_values,
            close=close_values,
            pattern_cols=direct_compute_cols,
        )
        out.update({col: int(pattern_values[col][-1]) for col in base_pattern_cols})
        for feature_col in direct_pattern_feature_cols:
            base_col = _PATTERN_FEATURE_SPEC_BY_COL[feature_col][0]
            out[feature_col] = int(pattern_values[base_col][-1])

    if not interval_to_pattern_feature_cols:
        return out

    base = pd.DataFrame(
        {
            OPENED_COL: pd.to_datetime(opened_values, errors="raise"),
            OPEN_COL: np.asarray(open_values, dtype=np.float64),
            HIGH_COL: np.asarray(high_values, dtype=np.float64),
            LOW_COL: np.asarray(low_values, dtype=np.float64),
            CLOSE_COL: np.asarray(close_values, dtype=np.float64),
        }
    )
    for (
        interval_label,
        interval_feature_cols,
    ) in interval_to_pattern_feature_cols.items():
        interval_rule = PATTERN_INTERVAL_TO_RULE[interval_label]
        interval_patterns = _compute_interval_patterns(
            base_df=base,
            interval_label=interval_label,
            rule=interval_rule,
            feature_cols=interval_feature_cols,
            opened_col=OPENED_COL,
            open_col=OPEN_COL,
            high_col=HIGH_COL,
            low_col=LOW_COL,
            close_col=CLOSE_COL,
        )
        for feature_col in interval_feature_cols:
            out[feature_col] = int(interval_patterns[feature_col].iloc[-1])
    return out


def _compute_latest_complete_interval_pattern_values_fast(
    opened_ns,
    open_values,
    high_values,
    low_values,
    close_values,
    *,
    bucket_ns,
    expected_count,
    base_pattern_cols,
):
    if not base_pattern_cols or len(opened_ns) == 0:
        return {}

    needed_complete = max(_PATTERN_LOOKBACK_BY_COL[col] for col in base_pattern_cols)
    tail_len = min(len(opened_ns), int(expected_count) * (int(needed_complete) + 1))
    opened_ns = opened_ns[-tail_len:]
    open_values = open_values[-tail_len:]
    high_values = high_values[-tail_len:]
    low_values = low_values[-tail_len:]
    close_values = close_values[-tail_len:]

    bucket_start_ns = (opened_ns // bucket_ns) * bucket_ns
    segment_breaks = np.empty(len(bucket_start_ns), dtype=bool)
    segment_breaks[0] = True
    segment_breaks[1:] = bucket_start_ns[1:] != bucket_start_ns[:-1]
    segment_starts = np.flatnonzero(segment_breaks)
    segment_ends = np.empty(len(segment_starts), dtype=np.int64)
    segment_ends[:-1] = segment_starts[1:]
    segment_ends[-1] = len(opened_ns)

    agg_open = []
    agg_high = []
    agg_low = []
    agg_close = []
    for start, end in zip(segment_starts, segment_ends):
        if int(end - start) != int(expected_count):
            continue
        agg_open.append(float(open_values[start]))
        agg_high.append(float(np.max(high_values[start:end])))
        agg_low.append(float(np.min(low_values[start:end])))
        agg_close.append(float(close_values[end - 1]))

    if not agg_open:
        return {col: 0 for col in base_pattern_cols}

    tail_len = max(_PATTERN_LOOKBACK_BY_COL[col] for col in base_pattern_cols)
    take_slice = slice(max(len(agg_open) - tail_len, 0), len(agg_open))
    patterns = build_candle_pattern_features_from_series(
        open_=np.asarray(agg_open[take_slice], dtype=np.float64),
        high=np.asarray(agg_high[take_slice], dtype=np.float64),
        low=np.asarray(agg_low[take_slice], dtype=np.float64),
        close=np.asarray(agg_close[take_slice], dtype=np.float64),
        pattern_cols=base_pattern_cols,
    )
    return {col: int(patterns[col][-1]) for col in base_pattern_cols}


def build_latest_candle_pattern_feature_dict_fast(
    opened_values,
    opened_ns_values,
    open_values,
    high_values,
    low_values,
    close_values,
    pattern_cols=None,
):
    selected_pattern_cols = resolve_candle_pattern_feature_cols(pattern_cols)
    if not selected_pattern_cols:
        return {}
    if len(open_values) == 0:
        return {}

    opened_ns = np.asarray(opened_ns_values, dtype=np.int64)
    open_arr = np.asarray(open_values, dtype=np.float64)
    high_arr = np.asarray(high_values, dtype=np.float64)
    low_arr = np.asarray(low_values, dtype=np.float64)
    close_arr = np.asarray(close_values, dtype=np.float64)
    if not (
        len(opened_ns)
        == len(open_arr)
        == len(high_arr)
        == len(low_arr)
        == len(close_arr)
    ):
        return build_latest_candle_pattern_feature_dict(
            opened_values=opened_values,
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            pattern_cols=selected_pattern_cols,
        )
    if len(opened_ns) > 1 and np.any(np.diff(opened_ns) <= 0):
        return build_latest_candle_pattern_feature_dict(
            opened_values=opened_values,
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            pattern_cols=selected_pattern_cols,
        )

    (
        base_pattern_cols,
        direct_pattern_feature_cols,
        interval_to_pattern_feature_cols,
    ) = _split_pattern_feature_cols(selected_pattern_cols)
    out = {}

    direct_pattern_cols = tuple(
        _PATTERN_FEATURE_SPEC_BY_COL[feature_col][0]
        for feature_col in direct_pattern_feature_cols
    )
    direct_compute_cols = _dedupe_ordered((*base_pattern_cols, *direct_pattern_cols))
    if direct_compute_cols:
        direct_tail_len = max(_PATTERN_LOOKBACK_BY_COL[col] for col in direct_compute_cols)
        pattern_values = build_candle_pattern_features_from_series(
            open_=open_arr[-direct_tail_len:],
            high=high_arr[-direct_tail_len:],
            low=low_arr[-direct_tail_len:],
            close=close_arr[-direct_tail_len:],
            pattern_cols=direct_compute_cols,
        )
        out.update({col: int(pattern_values[col][-1]) for col in base_pattern_cols})
        for feature_col in direct_pattern_feature_cols:
            base_col = _PATTERN_FEATURE_SPEC_BY_COL[feature_col][0]
            out[feature_col] = int(pattern_values[base_col][-1])

    for (
        interval_label,
        interval_feature_cols,
    ) in interval_to_pattern_feature_cols.items():
        rule = PATTERN_INTERVAL_TO_RULE.get(interval_label)
        if rule is None:
            return build_latest_candle_pattern_feature_dict(
                opened_values=opened_values,
                open_values=open_values,
                high_values=high_values,
                low_values=low_values,
                close_values=close_values,
                pattern_cols=selected_pattern_cols,
            )
        offset = to_offset(rule)
        bucket_ns = int(offset.nanos)
        if bucket_ns <= 0 or bucket_ns % _MINUTE_NS != 0:
            return build_latest_candle_pattern_feature_dict(
                opened_values=opened_values,
                open_values=open_values,
                high_values=high_values,
                low_values=low_values,
                close_values=close_values,
                pattern_cols=selected_pattern_cols,
            )
        expected_count = int(bucket_ns // _MINUTE_NS)
        if expected_count <= 0:
            return build_latest_candle_pattern_feature_dict(
                opened_values=opened_values,
                open_values=open_values,
                high_values=high_values,
                low_values=low_values,
                close_values=close_values,
                pattern_cols=selected_pattern_cols,
            )

        interval_base_pattern_cols = _dedupe_ordered(
            _PATTERN_FEATURE_SPEC_BY_COL[feature_col][0]
            for feature_col in interval_feature_cols
        )
        interval_values = _compute_latest_complete_interval_pattern_values_fast(
            opened_ns=opened_ns,
            open_values=open_arr,
            high_values=high_arr,
            low_values=low_arr,
            close_values=close_arr,
            bucket_ns=bucket_ns,
            expected_count=expected_count,
            base_pattern_cols=interval_base_pattern_cols,
        )
        for feature_col in interval_feature_cols:
            base_col = _PATTERN_FEATURE_SPEC_BY_COL[feature_col][0]
            out[feature_col] = int(interval_values.get(base_col, 0))

    return out


def _streak_feature_col(interval_label):
    return f"{STREAK_FEATURE_PREFIX}{interval_label}"


def resolve_streak_interval_to_rule(interval_labels):
    if isinstance(interval_labels, dict):
        labels = list(
            _normalize_candle_interval_lag_config(
                interval_labels,
                source_label="candle_streak_intervals",
            ).keys()
        )
    else:
        labels = [str(x).strip() for x in interval_labels]
    if not labels:
        raise ValueError("candle_streak_intervals cannot be empty.")

    unique_labels = []
    seen = set()
    for label in labels:
        if not label:
            raise ValueError("candle_streak_intervals cannot contain empty values.")
        if label in seen:
            continue
        if label not in INTERVAL_TO_RULE:
            supported = ", ".join(sorted(INTERVAL_TO_RULE.keys()))
            raise ValueError(
                f"Unsupported candle streak interval: '{label}'. "
                f"Supported: {supported}"
            )
        unique_labels.append(label)
        seen.add(label)
    return {label: INTERVAL_TO_RULE[label] for label in unique_labels}


def signed_streak_from_signs(signs):
    out = np.zeros(len(signs), dtype=np.int32)
    prev_sign = 0
    prev_len = 0

    for i, sign in enumerate(signs):
        sign = int(sign)
        if sign == 0:
            out[i] = 0
            prev_sign = 0
            prev_len = 0
            continue

        if sign == prev_sign:
            prev_len += 1
            out[i] = sign * prev_len
        else:
            prev_sign = sign
            prev_len = 1
            out[i] = sign

    return out


def _compute_interval_streak(
    base_df,
    interval_label,
    rule,
    opened_col=OPENED_COL,
    open_col=OPEN_COL,
    close_col=CLOSE_COL,
):
    agg = _resample_complete_interval_frame(
        base_df=base_df,
        rule=rule,
        required_cols=[opened_col, open_col, close_col],
        agg_spec={open_col: "first", close_col: "last"},
        dropna_cols=[open_col, close_col],
        count_col=open_col,
        opened_col=opened_col,
        context=f"{interval_label} candle streaks",
    )
    if agg.empty:
        return pd.Series(np.zeros(len(base_df), dtype=np.int32), index=base_df.index)

    candle_diff = agg[close_col].to_numpy(dtype=np.float64) - agg[open_col].to_numpy(
        dtype=np.float64
    )
    signs = np.sign(candle_diff).astype(np.int8, copy=False)
    streak = signed_streak_from_signs(signs)

    close_opened = pd.to_datetime(agg["__close_opened"], errors="coerce").to_numpy()
    events = pd.DataFrame(
        {
            opened_col: close_opened,
            _streak_feature_col(interval_label): streak,
        }
    )

    merged = _merge_interval_events(
        base_df=base_df,
        events=events,
        feature_cols=[_streak_feature_col(interval_label)],
        opened_col=opened_col,
    )
    series = (
        merged[_streak_feature_col(interval_label)]
        .fillna(0)
        .astype(np.int32)
        .rename(_streak_feature_col(interval_label))
    )
    return series.reindex(base_df.index)


def validate_signed_streak_logic():
    signs = np.array([1, 1, 1, -1, 1], dtype=np.int8)
    expected = np.array([1, 2, 3, -1, 1], dtype=np.int32)
    got = signed_streak_from_signs(signs)
    if not np.array_equal(got, expected):
        raise RuntimeError(
            f"Signed streak logic invalid: got={got}, expected={expected}"
        )


def add_candle_streak_features(
    df,
    interval_to_rule,
    opened_col=OPENED_COL,
    open_col=OPEN_COL,
    close_col=CLOSE_COL,
):
    required = [opened_col, open_col, close_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for candle streaks: {missing}")

    opened = pd.to_datetime(df[opened_col], errors="raise")
    if opened.duplicated().any():
        dup_count = int(opened.duplicated().sum())
        raise ValueError(f"Duplicate {opened_col} values found: {dup_count}")

    if opened.is_monotonic_increasing:
        base_df = df
        base = pd.DataFrame(
            {
                opened_col: opened,
                open_col: df[open_col].to_numpy(copy=False),
                close_col: df[close_col].to_numpy(copy=False),
            },
            index=df.index,
        )
    else:
        order = np.argsort(opened.to_numpy())
        base_df = df.iloc[order].reset_index(drop=True)
        base = pd.DataFrame(
            {
                opened_col: opened.iloc[order].to_numpy(),
                open_col: df[open_col].iloc[order].to_numpy(copy=False),
                close_col: df[close_col].iloc[order].to_numpy(copy=False),
            },
            index=base_df.index,
        )

    feature_values = {}
    for interval_label, rule in interval_to_rule.items():
        out_col = _streak_feature_col(interval_label)
        feature_values[out_col] = _compute_interval_streak(
            base_df=base,
            interval_label=interval_label,
            rule=rule,
            opened_col=opened_col,
            open_col=open_col,
            close_col=close_col,
        ).to_numpy(dtype=np.int32, copy=False)

    feature_frame = pd.DataFrame(feature_values, index=base_df.index)
    base_df = base_df.drop(columns=list(feature_values.keys()), errors="ignore")
    return pd.concat([base_df, feature_frame], axis=1, copy=False)


def _prepare_latest_streak_base(
    opened_index,
    open_values,
    close_values,
    open_col=OPEN_COL,
    close_col=CLOSE_COL,
):
    return pd.DataFrame(
        {
            open_col: np.asarray(open_values, dtype=np.float64),
            close_col: np.asarray(close_values, dtype=np.float64),
        },
        index=opened_index,
    )


def _prepare_latest_streak_inputs(
    opened_values,
    open_values,
    close_values,
    opened_col=OPENED_COL,
):
    opened_index = pd.DatetimeIndex(pd.to_datetime(opened_values, errors="raise"))
    open_arr = np.asarray(open_values, dtype=np.float64)
    close_arr = np.asarray(close_values, dtype=np.float64)

    if not (len(opened_index) == len(open_arr) == len(close_arr)):
        raise ValueError("Opened/Open/Close inputs must have the same length.")

    if not opened_index.is_monotonic_increasing:
        order = np.argsort(opened_index.asi8, kind="stable")
        opened_index = opened_index.take(order)
        open_arr = open_arr[order]
        close_arr = close_arr[order]

    if opened_index.has_duplicates:
        dup_count = int(opened_index.duplicated().sum())
        raise ValueError(f"Duplicate {opened_col} values found: {dup_count}")

    return opened_index, open_arr, close_arr


def _latest_signed_streak_value(signs):
    if len(signs) == 0:
        return 0

    last_sign = int(signs[-1])
    if last_sign == 0:
        return 0

    streak_len = 1
    for i in range(len(signs) - 2, -1, -1):
        if int(signs[i]) != last_sign:
            break
        streak_len += 1
    return int(last_sign * streak_len)


def _compute_latest_interval_streak_value_fast(
    opened_ns,
    open_values,
    close_values,
    rule,
):
    open_arr = np.asarray(open_values, dtype=np.float64)
    close_arr = np.asarray(close_values, dtype=np.float64)

    try:
        offset = to_offset(rule)
        bucket_ns = offset.nanos
    except ValueError:
        return None

    if bucket_ns <= 0 or bucket_ns % _MINUTE_NS != 0:
        return None

    expected_count = int(bucket_ns // _MINUTE_NS)
    if expected_count <= 0:
        return None

    if expected_count == 1:
        candle_diff = close_arr - open_arr
        signs = np.sign(candle_diff).astype(np.int8, copy=False)
        return _latest_signed_streak_value(signs)

    # Group 1m candles into fixed-width buckets without building a resampled DataFrame.
    bucket_start_ns = (opened_ns // bucket_ns) * bucket_ns
    segment_breaks = np.empty(len(bucket_start_ns), dtype=bool)
    segment_breaks[0] = True
    segment_breaks[1:] = bucket_start_ns[1:] != bucket_start_ns[:-1]
    segment_starts = np.flatnonzero(segment_breaks)
    segment_ends = np.empty(len(segment_starts), dtype=np.int64)
    segment_ends[:-1] = segment_starts[1:]
    segment_ends[-1] = len(opened_ns)

    counts = segment_ends - segment_starts
    is_complete = counts == expected_count
    if not np.any(is_complete):
        return 0

    first_pos = segment_starts[is_complete]
    last_pos = segment_ends[is_complete] - 1
    candle_diff = close_arr[last_pos] - open_arr[first_pos]
    signs = np.sign(candle_diff).astype(np.int8, copy=False)
    return _latest_signed_streak_value(signs)


def _compute_latest_interval_streak_value_pandas(
    indexed_base,
    interval_label,
    rule,
    open_col=OPEN_COL,
    close_col=CLOSE_COL,
):
    agg = indexed_base.resample(rule, label="left", closed="left").agg(
        {open_col: "first", close_col: "last"}
    )
    agg["__count"] = (
        indexed_base[open_col].resample(rule, label="left", closed="left").size()
    )
    agg = agg.dropna(subset=[open_col, close_col]).copy()
    if agg.empty:
        return 0

    bucket_start = pd.DatetimeIndex(agg.index)
    bucket_end = bucket_start + to_offset(rule)
    expected_count = np.asarray(
        (bucket_end - bucket_start) / pd.Timedelta(minutes=1),
        dtype=np.float64,
    )

    is_complete = agg["__count"].to_numpy(dtype=np.float64) == expected_count
    agg = agg.loc[is_complete].copy()
    if agg.empty:
        return 0

    candle_diff = agg[close_col].to_numpy(dtype=np.float64) - agg[open_col].to_numpy(
        dtype=np.float64
    )
    signs = np.sign(candle_diff).astype(np.int8, copy=False)
    streak = signed_streak_from_signs(signs)
    if len(streak) == 0:
        return 0
    return int(streak[-1])


def build_latest_candle_streak_feature_dict(
    opened_values,
    open_values,
    close_values,
    interval_to_rule,
    opened_col=OPENED_COL,
    open_col=OPEN_COL,
    close_col=CLOSE_COL,
):
    if len(opened_values) == 0:
        return {}

    opened_index, open_arr, close_arr = _prepare_latest_streak_inputs(
        opened_values=opened_values,
        open_values=open_values,
        close_values=close_values,
        opened_col=opened_col,
    )

    opened_ns = opened_index.asi8
    indexed_base = None
    out = {}
    for interval_label, rule in interval_to_rule.items():
        value = _compute_latest_interval_streak_value_fast(
            opened_ns=opened_ns,
            open_values=open_arr,
            close_values=close_arr,
            rule=rule,
        )
        if value is None:
            if indexed_base is None:
                indexed_base = _prepare_latest_streak_base(
                    opened_index=opened_index,
                    open_values=open_arr,
                    close_values=close_arr,
                    open_col=open_col,
                    close_col=close_col,
                )
            value = _compute_latest_interval_streak_value_pandas(
                indexed_base=indexed_base,
                interval_label=interval_label,
                rule=rule,
                open_col=open_col,
                close_col=close_col,
            )
        out[_streak_feature_col(interval_label)] = int(value)
    return out


def build_latest_candle_streak_feature_dict_fast(
    opened_values,
    opened_ns_values,
    open_values,
    close_values,
    interval_to_rule,
    opened_col=OPENED_COL,
    open_col=OPEN_COL,
    close_col=CLOSE_COL,
):
    if len(open_values) == 0:
        return {}

    opened_ns = np.asarray(opened_ns_values, dtype=np.int64)
    open_arr = np.asarray(open_values, dtype=np.float64)
    close_arr = np.asarray(close_values, dtype=np.float64)
    if not (len(opened_ns) == len(open_arr) == len(close_arr)):
        return build_latest_candle_streak_feature_dict(
            opened_values=opened_values,
            open_values=open_values,
            close_values=close_values,
            interval_to_rule=interval_to_rule,
            opened_col=opened_col,
            open_col=open_col,
            close_col=close_col,
        )
    if len(opened_ns) > 1 and np.any(np.diff(opened_ns) <= 0):
        return build_latest_candle_streak_feature_dict(
            opened_values=opened_values,
            open_values=open_values,
            close_values=close_values,
            interval_to_rule=interval_to_rule,
            opened_col=opened_col,
            open_col=open_col,
            close_col=close_col,
        )

    indexed_base = None
    out = {}
    for interval_label, rule in interval_to_rule.items():
        value = _compute_latest_interval_streak_value_fast(
            opened_ns=opened_ns,
            open_values=open_arr,
            close_values=close_arr,
            rule=rule,
        )
        if value is None:
            if indexed_base is None:
                opened_index = pd.DatetimeIndex(
                    pd.to_datetime(opened_values, errors="raise")
                )
                indexed_base = _prepare_latest_streak_base(
                    opened_index=opened_index,
                    open_values=open_arr,
                    close_values=close_arr,
                    open_col=open_col,
                    close_col=close_col,
                )
            value = _compute_latest_interval_streak_value_pandas(
                indexed_base=indexed_base,
                interval_label=interval_label,
                rule=rule,
                open_col=open_col,
                close_col=close_col,
            )
        out[_streak_feature_col(interval_label)] = int(value)
    return out
