from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from utils.collections import dedupe_ordered_tuple as _dedupe_ordered

OPENED_COL = "Opened"
SESSION_FLOW_FEATURE_PREFIX = "session_flow_"
SESSION_OPEN_IMPULSE_FEATURE_PREFIX = "session_open_impulse_"
SESSION_CLOSE_IMPULSE_FEATURE_PREFIX = "session_close_impulse_"
DEFAULT_SESSION_WEEKDAYS = (0, 1, 2, 3, 4)
_SESSION_OPEN_IMPULSE_TAU_MINUTES = 30.0
_SESSION_CLOSE_IMPULSE_TAU_MINUTES = 30.0
_SESSION_FEATURE_DTYPE = np.float32

_RAW_SESSIONS = {
    # North America
    "NYSE": {
        "start": time(9, 30),
        "end": time(16, 0),
        "timezone": "America/New_York",
        "avg_turnover_gold_oz": 36027392,
    },
    # "NASDAQ": {  # 1:1 overlap with NYSE in UTC all year; keep the higher-turnover session.
    #     "start": time(9, 30),
    #     "end": time(16, 0),
    #     "timezone": "America/New_York",
    #     "avg_turnover_gold_oz": 33997447,
    # },
    # Europe
    # "LSE": {  # 1:1 overlap with Xetra in UTC all year; keep the higher-turnover session.
    #     "start": time(8, 0),
    #     "end": time(16, 30),
    #     "timezone": "Europe/London",
    #     "avg_turnover_gold_oz": 1478288,
    # },
    "Xetra": {
        "start": time(9, 0),
        "end": time(17, 30),
        "timezone": "Europe/Berlin",
        "avg_turnover_gold_oz": 1598603,
    },
    # Asia-Pacific
    "TSE_Morning": {
        "start": time(9, 0),
        "end": time(11, 30),
        "timezone": "Asia/Tokyo",
        "avg_turnover_gold_oz": 4975573,
    },
    "TSE_Afternoon": {
        "start": time(12, 30),
        "end": time(15, 30),
        "timezone": "Asia/Tokyo",
        "avg_turnover_gold_oz": 5970688,
    },
    "SSE_Morning": {
        "start": time(9, 30),
        "end": time(11, 30),
        "timezone": "Asia/Shanghai",
        "avg_turnover_gold_oz": 9569137,
    },
    "SSE_Afternoon": {
        "start": time(13, 0),
        "end": time(15, 0),
        "timezone": "Asia/Shanghai",
        "avg_turnover_gold_oz": 9569137,
    },
    "BSE": {
        "start": time(9, 15),
        "end": time(15, 30),
        "timezone": "Asia/Kolkata",
        "avg_turnover_gold_oz": 140737,
    },
    "ASX": {
        "start": time(10, 0),
        "end": time(16, 0),
        "timezone": "Australia/Sydney",
        "avg_turnover_gold_oz": 977696,
    },
    "HOSE_Morning": {
        "start": time(9, 15),
        "end": time(11, 30),
        "timezone": "Asia/Ho_Chi_Minh",
        "avg_turnover_gold_oz": 135064,
    },
    "HOSE_Afternoon": {
        "start": time(13, 0),
        "end": time(14, 30),
        "timezone": "Asia/Ho_Chi_Minh",
        "avg_turnover_gold_oz": 90043,
    },
    "PSE_Morning": {
        "start": time(9, 30),
        "end": time(12, 0),
        "timezone": "Asia/Manila",
        "avg_turnover_gold_oz": 11878,
    },
    "PSE_Afternoon": {
        "start": time(13, 0),
        "end": time(14, 45),
        "timezone": "Asia/Manila",
        "avg_turnover_gold_oz": 8315,
    },
    "PSX": {
        "start": time(9, 32),
        "end": time(15, 30),
        "timezone": "Asia/Karachi",
        "avg_turnover_gold_oz": 35793,
    },
    "SET_Morning": {
        "start": time(10, 0),
        "end": time(12, 30),
        "timezone": "Asia/Bangkok",
        "avg_turnover_gold_oz": 96693,
    },
    "SET_Afternoon": {
        "start": time(14, 0),
        "end": time(16, 30),
        "timezone": "Asia/Bangkok",
        "avg_turnover_gold_oz": 96693,
    },
    "IDX": {
        "start": time(9, 0),
        "end": time(15, 50),
        "timezone": "Asia/Jakarta",
        "avg_turnover_gold_oz": 145719,
    },
    # Middle East and Africa
    "DFM": {
        "start": time(10, 0),
        "end": time(15, 0),
        "timezone": "Asia/Dubai",
        "avg_turnover_gold_oz": 34152,
    },
    "NSE_Nigeria": {
        "start": time(10, 0),
        "end": time(14, 20),
        "timezone": "Africa/Lagos",
        "avg_turnover_gold_oz": 2526,
    },
    "BIST_Morning": {
        "start": time(9, 30),
        "end": time(12, 30),
        "timezone": "Europe/Istanbul",
        "avg_turnover_gold_oz": 379348,
    },
    "BIST_Afternoon": {
        "start": time(14, 0),
        "end": time(17, 30),
        "timezone": "Europe/Istanbul",
        "avg_turnover_gold_oz": 442573,
    },
    "NSE_Kenya": {
        "start": time(9, 0),
        "end": time(15, 0),
        "timezone": "Africa/Nairobi",
        "avg_turnover_gold_oz": 733,
    },
    # South America
    "B3": {
        "start": time(10, 0),
        "end": time(16, 55),
        "timezone": "America/Sao_Paulo",
        "avg_turnover_gold_oz": 780526,
    },
    "BCBA": {
        "start": time(10, 30),
        "end": time(17, 0),
        "timezone": "America/Argentina/Buenos_Aires",
        "avg_turnover_gold_oz": 1742627,
    },
    "BVC": {
        "start": time(9, 30),
        "end": time(15, 55),
        "timezone": "America/Bogota",
        "avg_turnover_gold_oz": 7363,
    },
}


def _normalize_sessions(raw_sessions):
    normalized = {}
    for session_name, cfg in raw_sessions.items():
        start_time = cfg["start"]
        end_time = cfg["end"]
        timezone_name = str(cfg["timezone"]).strip()
        weekdays = tuple(int(v) for v in cfg.get("weekdays", DEFAULT_SESSION_WEEKDAYS))

        if "avg_turnover_gold_oz" not in cfg:
            raise ValueError(
                f"Session '{session_name}' is missing required avg_turnover_gold_oz."
            )
        avg_turnover_gold_oz = float(cfg["avg_turnover_gold_oz"])

        if not timezone_name:
            raise ValueError(f"Session '{session_name}' has an empty timezone.")
        if not weekdays:
            raise ValueError(
                f"Session '{session_name}' must define at least one weekday."
            )
        if any(day < 0 or day > 6 for day in weekdays):
            raise ValueError(
                f"Session '{session_name}' weekdays must be integers in [0, 6]."
            )
        if not np.isfinite(avg_turnover_gold_oz):
            raise ValueError(
                f"Session '{session_name}' has non-finite avg_turnover_gold_oz."
            )

        start_minute = int(start_time.hour) * 60 + int(start_time.minute)
        end_minute = int(end_time.hour) * 60 + int(end_time.minute)
        duration_minutes = end_minute - start_minute
        if duration_minutes <= 0:
            raise ValueError(
                f"Session '{session_name}' must have end > start within the same local day."
            )

        weekdays = tuple(sorted(set(weekdays)))
        if avg_turnover_gold_oz <= 0.0:
            avg_turnover_gold_oz_per_min = 0.0
            log_avg_turnover_gold_oz_per_min = 0.0
        else:
            avg_turnover_gold_oz_per_min = avg_turnover_gold_oz / float(
                duration_minutes
            )
            log_avg_turnover_gold_oz_per_min = float(
                np.log1p(avg_turnover_gold_oz_per_min)
            )

        normalized[session_name] = {
            "name": session_name,
            "start_time": start_time,
            "end_time": end_time,
            "start_minute": start_minute,
            "end_minute": end_minute,
            "duration_minutes": duration_minutes,
            "timezone": timezone_name,
            "tzinfo": ZoneInfo(timezone_name),
            "weekdays": weekdays,
            "weekdays_arr": np.asarray(weekdays, dtype=np.int8),
            "avg_turnover_gold_oz": avg_turnover_gold_oz,
            "avg_turnover_gold_oz_per_min": avg_turnover_gold_oz_per_min,
            "log_avg_turnover_gold_oz_per_min": log_avg_turnover_gold_oz_per_min,
        }
    return normalized


SESSIONS = _normalize_sessions(_RAW_SESSIONS)


def _build_session_feature_name_maps():
    supported_cols = []
    session_name_by_feature_col = {}
    feature_kind_by_col = {}

    for session_name in SESSIONS.keys():
        feature_cols = {
            "flow": f"{SESSION_FLOW_FEATURE_PREFIX}{session_name}",
            "open_impulse": f"{SESSION_OPEN_IMPULSE_FEATURE_PREFIX}{session_name}",
            "close_impulse": f"{SESSION_CLOSE_IMPULSE_FEATURE_PREFIX}{session_name}",
        }
        for feature_kind, feature_col in feature_cols.items():
            supported_cols.append(feature_col)
            session_name_by_feature_col[feature_col] = session_name
            feature_kind_by_col[feature_col] = feature_kind

    return tuple(supported_cols), session_name_by_feature_col, feature_kind_by_col


(
    SUPPORTED_SESSION_OPEN_FEATURE_COLS,
    SESSION_NAME_BY_OPEN_FEATURE_COL,
    _SESSION_OPEN_FEATURE_KIND_BY_COL,
) = _build_session_feature_name_maps()


def is_session_open_feature(feature_name):
    return str(feature_name).strip() in SESSION_NAME_BY_OPEN_FEATURE_COL


def resolve_session_open_feature_cols(feature_cols=None):
    if feature_cols is None:
        return SUPPORTED_SESSION_OPEN_FEATURE_COLS

    requested = [str(col).strip() for col in feature_cols]
    if not requested:
        raise ValueError("feature_cols cannot be empty.")

    requested = _dedupe_ordered(requested)
    unsupported = [
        col for col in requested if col not in SESSION_NAME_BY_OPEN_FEATURE_COL
    ]
    if unsupported:
        supported = ", ".join(SUPPORTED_SESSION_OPEN_FEATURE_COLS)
        raise ValueError(
            "Unsupported session open feature columns: "
            + ", ".join(unsupported)
            + f". Supported: {supported}"
        )
    return requested


def _coerce_opened_to_utc_series(opened_values):
    opened_utc = pd.to_datetime(opened_values, errors="raise", utc=True)
    if isinstance(opened_utc, pd.Series):
        return opened_utc
    return pd.Series(opened_utc)


def _group_feature_cols_by_session(selected_cols):
    grouped = {}
    for feature_col in selected_cols:
        session_name = SESSION_NAME_BY_OPEN_FEATURE_COL[feature_col]
        feature_kind = _SESSION_OPEN_FEATURE_KIND_BY_COL[feature_col]
        grouped.setdefault(session_name, []).append((feature_col, feature_kind))
    return grouped


def _get_timezone_local_parts(opened_utc, timezone_name, tz_parts_cache):
    cached = tz_parts_cache.get(timezone_name)
    if cached is not None:
        return cached

    opened_local = opened_utc.dt.tz_convert(timezone_name)
    local_naive = opened_local.dt.tz_localize(None)
    cached = {
        "weekday": local_naive.dt.dayofweek.to_numpy(dtype=np.int8, copy=False),
        "minute_of_day": (
            local_naive.dt.hour.to_numpy(dtype=np.int16, copy=False) * 60
            + local_naive.dt.minute.to_numpy(dtype=np.int16, copy=False)
        ),
    }
    tz_parts_cache[timezone_name] = cached
    return cached


def _compute_session_open_feature_arrays(session_cfg, tz_local_parts, requested_kinds):
    if not requested_kinds:
        return {}

    minute_of_day = tz_local_parts["minute_of_day"]
    weekday = tz_local_parts["weekday"]
    values_by_kind = {
        feature_kind: np.zeros(minute_of_day.shape[0], dtype=_SESSION_FEATURE_DTYPE)
        for feature_kind in requested_kinds
    }
    if minute_of_day.size == 0:
        return values_by_kind

    is_active = np.isin(weekday, session_cfg["weekdays_arr"]) & (
        (minute_of_day >= int(session_cfg["start_minute"]))
        & (minute_of_day < int(session_cfg["end_minute"]))
    )
    if not np.any(is_active):
        return values_by_kind

    weight = _SESSION_FEATURE_DTYPE(session_cfg["log_avg_turnover_gold_oz_per_min"])
    if weight <= 0.0:
        return values_by_kind

    if "flow" in requested_kinds:
        values_by_kind["flow"][is_active] = weight

    if "open_impulse" in requested_kinds or "close_impulse" in requested_kinds:
        active_minutes = minute_of_day[is_active].astype(
            _SESSION_FEATURE_DTYPE,
            copy=False,
        )
        minutes_since_open = active_minutes - _SESSION_FEATURE_DTYPE(
            session_cfg["start_minute"]
        )
        minutes_to_close = _SESSION_FEATURE_DTYPE(
            session_cfg["end_minute"]
        ) - active_minutes

        if "open_impulse" in requested_kinds:
            values_by_kind["open_impulse"][is_active] = weight * np.exp(
                -minutes_since_open
                / _SESSION_FEATURE_DTYPE(_SESSION_OPEN_IMPULSE_TAU_MINUTES)
            )
        if "close_impulse" in requested_kinds:
            values_by_kind["close_impulse"][is_active] = weight * np.exp(
                -minutes_to_close
                / _SESSION_FEATURE_DTYPE(_SESSION_CLOSE_IMPULSE_TAU_MINUTES)
            )

    return values_by_kind


def add_session_open_features(df, feature_cols=None, opened_col=OPENED_COL):
    if opened_col not in df.columns:
        raise ValueError(
            f"Missing required column for session open features: {opened_col}"
        )

    selected_cols = resolve_session_open_feature_cols(feature_cols)
    opened_utc = _coerce_opened_to_utc_series(df[opened_col])
    tz_parts_cache = {}
    feature_values = {}

    for (
        session_name,
        feature_specs,
    ) in _group_feature_cols_by_session(selected_cols).items():
        session_cfg = SESSIONS[session_name]
        tz_local_parts = _get_timezone_local_parts(
            opened_utc,
            session_cfg["timezone"],
            tz_parts_cache,
        )
        values_by_kind = _compute_session_open_feature_arrays(
            session_cfg,
            tz_local_parts,
            {feature_kind for _, feature_kind in feature_specs},
        )
        for feature_col, feature_kind in feature_specs:
            feature_values[feature_col] = values_by_kind[feature_kind]

    feature_frame = pd.DataFrame(
        {feature_col: feature_values[feature_col] for feature_col in selected_cols},
        index=df.index,
    )
    duplicate_cols = [col for col in feature_frame.columns if col in df.columns]
    if duplicate_cols:
        base_df = df.drop(columns=duplicate_cols)
    else:
        base_df = df
    return pd.concat([base_df, feature_frame], axis=1, copy=False)


def build_latest_session_open_feature_dict(
    opened_values,
    feature_cols=None,
    opened_col=OPENED_COL,
):
    if len(opened_values) == 0:
        return {}

    selected_cols = resolve_session_open_feature_cols(feature_cols)
    latest_df = pd.DataFrame({opened_col: [opened_values[-1]]})
    latest_df = add_session_open_features(
        latest_df,
        feature_cols=selected_cols,
        opened_col=opened_col,
    )
    return {
        feature_col: float(latest_df[feature_col].iloc[-1])
        for feature_col in selected_cols
    }


def _compute_latest_session_open_feature_values(local_dt, session_cfg, requested_kinds):
    values_by_kind = {
        feature_kind: _SESSION_FEATURE_DTYPE(0.0) for feature_kind in requested_kinds
    }
    if not requested_kinds:
        return values_by_kind

    minute_of_day = int(local_dt.hour) * 60 + int(local_dt.minute)
    weekday = int(local_dt.weekday())
    is_active = (
        weekday in session_cfg["weekdays"]
        and int(session_cfg["start_minute"])
        <= minute_of_day
        < int(session_cfg["end_minute"])
    )
    if not is_active:
        return values_by_kind

    weight = _SESSION_FEATURE_DTYPE(session_cfg["log_avg_turnover_gold_oz_per_min"])
    if weight <= 0.0:
        return values_by_kind

    minutes_since_open = _SESSION_FEATURE_DTYPE(
        minute_of_day - int(session_cfg["start_minute"])
    )
    minutes_to_close = _SESSION_FEATURE_DTYPE(
        int(session_cfg["end_minute"]) - minute_of_day
    )

    if "flow" in requested_kinds:
        values_by_kind["flow"] = weight
    if "open_impulse" in requested_kinds:
        values_by_kind["open_impulse"] = weight * np.exp(
            -minutes_since_open
            / _SESSION_FEATURE_DTYPE(_SESSION_OPEN_IMPULSE_TAU_MINUTES)
        )
    if "close_impulse" in requested_kinds:
        values_by_kind["close_impulse"] = weight * np.exp(
            -minutes_to_close
            / _SESSION_FEATURE_DTYPE(_SESSION_CLOSE_IMPULSE_TAU_MINUTES)
        )
    return values_by_kind


def build_latest_session_open_feature_dict_fast(latest_opened, feature_cols=None):
    selected_cols = resolve_session_open_feature_cols(feature_cols)
    if not selected_cols:
        return {}

    latest_ts = pd.Timestamp(latest_opened)
    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.tz_localize("UTC")
    else:
        latest_ts = latest_ts.tz_convert("UTC")

    local_dt_cache = {}
    out = {}
    for (
        session_name,
        feature_specs,
    ) in _group_feature_cols_by_session(selected_cols).items():
        session_cfg = SESSIONS[session_name]
        timezone_name = session_cfg["timezone"]
        local_dt = local_dt_cache.get(timezone_name)
        if local_dt is None:
            local_dt = latest_ts.tz_convert(session_cfg["tzinfo"]).to_pydatetime()
            local_dt_cache[timezone_name] = local_dt
        values_by_kind = _compute_latest_session_open_feature_values(
            local_dt,
            session_cfg,
            {feature_kind for _, feature_kind in feature_specs},
        )
        for feature_col, feature_kind in feature_specs:
            out[feature_col] = float(values_by_kind[feature_kind])
    return {feature_col: out[feature_col] for feature_col in selected_cols}
