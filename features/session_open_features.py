from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OPENED_COL = "Opened"
SESSION_COUNTER_FEATURE_PREFIX = "session_counter_"
DEFAULT_SESSION_WEEKDAYS = (0, 1, 2, 3, 4)
_NS_PER_MINUTE = 60 * 1_000_000_000
_UTC_ZONEINFO = ZoneInfo("UTC")

_RAW_SESSIONS = {
    # North America
    "NYSE": {
        "start": time(9, 30),
        "end": time(16, 0),
        "timezone": "America/New_York",
    },
    "NASDAQ": {
        "start": time(9, 30),
        "end": time(16, 0),
        "timezone": "America/New_York",
    },
    # Europe
    "LSE": {
        "start": time(8, 0),
        "end": time(16, 30),
        "timezone": "Europe/London",
    },
    "Xetra": {
        "start": time(9, 0),
        "end": time(17, 30),
        "timezone": "Europe/Berlin",
    },
    # Asia-Pacific
    "TSE_Morning": {
        "start": time(9, 0),
        "end": time(11, 30),
        "timezone": "Asia/Tokyo",
    },
    "TSE_Afternoon": {
        "start": time(12, 30),
        "end": time(15, 30),
        "timezone": "Asia/Tokyo",
    },
    "SSE_Morning": {
        "start": time(9, 30),
        "end": time(11, 30),
        "timezone": "Asia/Shanghai",
    },
    "SSE_Afternoon": {
        "start": time(13, 0),
        "end": time(15, 0),
        "timezone": "Asia/Shanghai",
    },
    "BSE": {
        "start": time(9, 15),
        "end": time(15, 30),
        "timezone": "Asia/Kolkata",
    },
    "ASX": {
        "start": time(10, 0),
        "end": time(16, 0),
        "timezone": "Australia/Sydney",
    },
    "HOSE_Morning": {
        "start": time(9, 15),
        "end": time(11, 30),
        "timezone": "Asia/Ho_Chi_Minh",
    },
    "HOSE_Afternoon": {
        "start": time(13, 0),
        "end": time(14, 30),
        "timezone": "Asia/Ho_Chi_Minh",
    },
    "PSE_Morning": {
        "start": time(9, 30),
        "end": time(12, 0),
        "timezone": "Asia/Manila",
    },
    "PSE_Afternoon": {
        "start": time(13, 0),
        "end": time(14, 45),
        "timezone": "Asia/Manila",
    },
    "PSX": {
        "start": time(9, 32),
        "end": time(15, 30),
        "timezone": "Asia/Karachi",
    },
    "SET_Morning": {
        "start": time(10, 0),
        "end": time(12, 30),
        "timezone": "Asia/Bangkok",
    },
    "SET_Afternoon": {
        "start": time(14, 0),
        "end": time(16, 30),
        "timezone": "Asia/Bangkok",
    },
    "IDX": {
        "start": time(9, 0),
        "end": time(15, 50),
        "timezone": "Asia/Jakarta",
    },
    # Middle East and Africa
    "DFM": {
        "start": time(10, 0),
        "end": time(15, 0),
        "timezone": "Asia/Dubai",
    },
    "NSE_Nigeria": {
        "start": time(10, 0),
        "end": time(14, 20),
        "timezone": "Africa/Lagos",
    },
    "BIST_Morning": {
        "start": time(9, 30),
        "end": time(12, 30),
        "timezone": "Europe/Istanbul",
    },
    "BIST_Afternoon": {
        "start": time(14, 0),
        "end": time(17, 30),
        "timezone": "Europe/Istanbul",
    },
    "NSE_Kenya": {
        "start": time(9, 0),
        "end": time(15, 0),
        "timezone": "Africa/Nairobi",
    },
    # South America
    "B3": {
        "start": time(10, 0),
        "end": time(16, 55),
        "timezone": "America/Sao_Paulo",
    },
    "BCBA": {
        "start": time(11, 0),
        "end": time(17, 0),
        "timezone": "America/Argentina/Buenos_Aires",
    },
    "BVC": {
        "start": time(9, 30),
        "end": time(15, 55),
        "timezone": "America/Bogota",
    },
}


def _dedupe_ordered(values):
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return tuple(out)


def _normalize_sessions(raw_sessions):
    normalized = {}
    for session_name, cfg in raw_sessions.items():
        start_time = cfg["start"]
        end_time = cfg["end"]
        timezone_name = str(cfg["timezone"]).strip()
        weekdays = tuple(int(v) for v in cfg.get("weekdays", DEFAULT_SESSION_WEEKDAYS))

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

        start_minute = int(start_time.hour) * 60 + int(start_time.minute)
        end_minute = int(end_time.hour) * 60 + int(end_time.minute)
        if end_minute <= start_minute:
            raise ValueError(
                f"Session '{session_name}' must have end > start within the same local day."
            )

        normalized[session_name] = {
            "name": session_name,
            "start_time": start_time,
            "end_time": end_time,
            "start_minute": start_minute,
            "end_minute": end_minute,
            "timezone": timezone_name,
            "tzinfo": ZoneInfo(timezone_name),
            "weekdays": tuple(sorted(set(weekdays))),
        }
    return normalized


SESSIONS = _normalize_sessions(_RAW_SESSIONS)
SUPPORTED_SESSION_COUNTER_COLS = tuple(
    f"{SESSION_COUNTER_FEATURE_PREFIX}{session_name}"
    for session_name in SESSIONS.keys()
)
SESSION_NAME_BY_FEATURE_COL = {
    feature_col: feature_col[len(SESSION_COUNTER_FEATURE_PREFIX) :]
    for feature_col in SUPPORTED_SESSION_COUNTER_COLS
}


def is_session_counter_feature(feature_name):
    return str(feature_name).strip() in SESSION_NAME_BY_FEATURE_COL


def resolve_session_counter_feature_cols(feature_cols=None):
    if feature_cols is None:
        return SUPPORTED_SESSION_COUNTER_COLS

    requested = [str(col).strip() for col in feature_cols]
    if not requested:
        raise ValueError("feature_cols cannot be empty.")

    requested = _dedupe_ordered(requested)
    unsupported = [col for col in requested if col not in SESSION_NAME_BY_FEATURE_COL]
    if unsupported:
        supported = ", ".join(SUPPORTED_SESSION_COUNTER_COLS)
        raise ValueError(
            "Unsupported session counter feature columns: "
            + ", ".join(unsupported)
            + f". Supported: {supported}"
        )
    return requested


def _coerce_opened_to_utc_series(opened_values):
    opened_utc = pd.to_datetime(opened_values, errors="raise", utc=True)
    if isinstance(opened_utc, pd.Series):
        return opened_utc
    return pd.Series(opened_utc)


def _build_local_boundary_utc_ns(day_value, boundary_time, tzinfo):
    day = pd.Timestamp(day_value).date()
    boundary_local = datetime(
        day.year,
        day.month,
        day.day,
        boundary_time.hour,
        boundary_time.minute,
        tzinfo=tzinfo,
    )
    return int(pd.Timestamp(boundary_local).tz_convert("UTC").value)


def _build_trading_calendar(local_date_days, session_cfg):
    calendar_start = local_date_days.min() - np.timedelta64(7, "D")
    calendar_end = local_date_days.max()
    calendar_days = np.arange(
        calendar_start,
        calendar_end + np.timedelta64(1, "D"),
        np.timedelta64(1, "D"),
    )
    weekday_arr = pd.DatetimeIndex(
        calendar_days.astype("datetime64[ns]")
    ).dayofweek.to_numpy(
        dtype=np.int8,
        copy=False,
    )
    trading_day_mask = np.isin(
        weekday_arr,
        np.asarray(session_cfg["weekdays"], dtype=np.int8),
    )

    prev_trading_pos = np.full(calendar_days.shape[0], -1, dtype=np.int32)
    last_trading_pos = -1
    for idx in range(calendar_days.shape[0]):
        prev_trading_pos[idx] = last_trading_pos
        if trading_day_mask[idx]:
            last_trading_pos = idx

    open_utc_ns = np.full(calendar_days.shape[0], -1, dtype=np.int64)
    end_utc_ns = np.full(calendar_days.shape[0], -1, dtype=np.int64)
    trading_positions = np.flatnonzero(trading_day_mask)
    for idx in trading_positions:
        day_value = calendar_days[idx]
        open_utc_ns[idx] = _build_local_boundary_utc_ns(
            day_value=day_value,
            boundary_time=session_cfg["start_time"],
            tzinfo=session_cfg["tzinfo"],
        )
        end_utc_ns[idx] = _build_local_boundary_utc_ns(
            day_value=day_value,
            boundary_time=session_cfg["end_time"],
            tzinfo=session_cfg["tzinfo"],
        )

    return calendar_days, trading_day_mask, prev_trading_pos, open_utc_ns, end_utc_ns


def _compute_session_counter_values(opened_utc, session_cfg, tz_parts_cache):
    timezone_name = session_cfg["timezone"]
    if timezone_name not in tz_parts_cache:
        opened_local = opened_utc.dt.tz_convert(timezone_name)
        local_naive = opened_local.dt.tz_localize(None)
        tz_parts_cache[timezone_name] = {
            "local_date_days": local_naive.to_numpy(dtype="datetime64[D]"),
            "minute_of_day": (
                local_naive.dt.hour.to_numpy(dtype=np.int32, copy=False) * 60
                + local_naive.dt.minute.to_numpy(dtype=np.int32, copy=False)
            ),
        }

    local_date_days = tz_parts_cache[timezone_name]["local_date_days"]
    minute_of_day = tz_parts_cache[timezone_name]["minute_of_day"]
    if local_date_days.size == 0:
        return np.empty(0, dtype=np.int32)

    (
        calendar_days,
        trading_day_mask,
        prev_trading_pos,
        open_utc_ns,
        end_utc_ns,
    ) = _build_trading_calendar(local_date_days, session_cfg)

    row_date_pos = (
        (local_date_days - calendar_days[0])
        .astype("timedelta64[D]")
        .astype(np.int32, copy=False)
    )
    row_is_trading_day = trading_day_mask[row_date_pos]
    start_minute = int(session_cfg["start_minute"])
    end_minute = int(session_cfg["end_minute"])
    is_active = (
        row_is_trading_day
        & (minute_of_day >= start_minute)
        & (minute_of_day < end_minute)
    )

    opened_utc_ns = opened_utc.array.asi8
    values = np.empty(opened_utc_ns.shape[0], dtype=np.int32)

    if np.any(is_active):
        active_pos = row_date_pos[is_active]
        active_diff = (
            opened_utc_ns[is_active] - open_utc_ns[active_pos]
        ) // _NS_PER_MINUTE
        values[is_active] = active_diff.astype(np.int32, copy=False)

    inactive_mask = ~is_active
    if np.any(inactive_mask):
        inactive_date_pos = row_date_pos[inactive_mask]
        inactive_minute = minute_of_day[inactive_mask]
        inactive_is_trading_day = row_is_trading_day[inactive_mask]
        close_date_pos = np.where(
            inactive_is_trading_day & (inactive_minute >= end_minute),
            inactive_date_pos,
            prev_trading_pos[inactive_date_pos],
        ).astype(np.int32, copy=False)
        if np.any(close_date_pos < 0):
            raise ValueError(
                f"Could not resolve previous trading close for session '{session_cfg['name']}'."
            )
        inactive_diff = (
            opened_utc_ns[inactive_mask] - end_utc_ns[close_date_pos]
        ) // _NS_PER_MINUTE
        values[inactive_mask] = -inactive_diff.astype(np.int32, copy=False)

    return values


def add_session_counter_features(df, feature_cols=None, opened_col=OPENED_COL):
    if opened_col not in df.columns:
        raise ValueError(
            f"Missing required column for session counter features: {opened_col}"
        )

    selected_cols = resolve_session_counter_feature_cols(feature_cols)
    opened_utc = _coerce_opened_to_utc_series(df[opened_col])
    tz_parts_cache = {}
    feature_values = {}
    for feature_col in selected_cols:
        session_name = SESSION_NAME_BY_FEATURE_COL[feature_col]
        session_cfg = SESSIONS[session_name]
        feature_values[feature_col] = _compute_session_counter_values(
            opened_utc=opened_utc,
            session_cfg=session_cfg,
            tz_parts_cache=tz_parts_cache,
        )

    feature_frame = pd.DataFrame(feature_values, index=df.index)
    base_df = df.drop(columns=list(feature_values.keys()), errors="ignore")
    return pd.concat([base_df, feature_frame], axis=1, copy=False)


def build_latest_session_counter_feature_dict(
    opened_values,
    feature_cols=None,
    opened_col=OPENED_COL,
):
    if len(opened_values) == 0:
        return {}

    latest_df = pd.DataFrame({opened_col: [opened_values[-1]]})
    latest_df = add_session_counter_features(
        latest_df,
        feature_cols=feature_cols,
        opened_col=opened_col,
    )
    selected_cols = resolve_session_counter_feature_cols(feature_cols)
    return {col: int(latest_df[col].iloc[-1]) for col in selected_cols}


def _previous_trading_date(local_date, weekdays):
    weekday_set = set(int(day) for day in weekdays)
    for offset in range(1, 8):
        candidate = local_date - pd.Timedelta(days=offset)
        if int(candidate.dayofweek) in weekday_set:
            return candidate.date()
    raise ValueError("Could not resolve previous trading day.")


def _build_local_session_boundary(local_date, boundary_time, tzinfo):
    return datetime(
        local_date.year,
        local_date.month,
        local_date.day,
        boundary_time.hour,
        boundary_time.minute,
        tzinfo=tzinfo,
    )


def _compute_latest_session_counter_value(local_dt, session_cfg):
    minute_of_day = int(local_dt.hour) * 60 + int(local_dt.minute)
    weekday = int(local_dt.weekday())
    weekdays = tuple(int(day) for day in session_cfg["weekdays"])
    start_minute = int(session_cfg["start_minute"])
    end_minute = int(session_cfg["end_minute"])
    local_date = local_dt.date()
    tzinfo = session_cfg["tzinfo"]

    if weekday in weekdays and start_minute <= minute_of_day < end_minute:
        open_local = _build_local_session_boundary(
            local_date, session_cfg["start_time"], tzinfo
        )
        return int(
            (
                local_dt.astimezone(_UTC_ZONEINFO)
                - open_local.astimezone(_UTC_ZONEINFO)
            ).total_seconds()
            // 60
        )

    if weekday in weekdays and minute_of_day >= end_minute:
        close_date = local_date
    else:
        close_date = _previous_trading_date(pd.Timestamp(local_date), weekdays)

    close_local = _build_local_session_boundary(
        close_date, session_cfg["end_time"], tzinfo
    )
    return -int(
        (
            local_dt.astimezone(_UTC_ZONEINFO) - close_local.astimezone(_UTC_ZONEINFO)
        ).total_seconds()
        // 60
    )


def build_latest_session_counter_feature_dict_fast(latest_opened, feature_cols=None):
    selected_cols = resolve_session_counter_feature_cols(feature_cols)
    if not selected_cols:
        return {}

    latest_ts = pd.Timestamp(latest_opened)
    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.tz_localize("UTC")
    else:
        latest_ts = latest_ts.tz_convert("UTC")

    local_dt_cache = {}
    out = {}
    for feature_col in selected_cols:
        session_name = SESSION_NAME_BY_FEATURE_COL[feature_col]
        session_cfg = SESSIONS[session_name]
        timezone_name = session_cfg["timezone"]
        local_dt = local_dt_cache.get(timezone_name)
        if local_dt is None:
            local_dt = latest_ts.tz_convert(session_cfg["tzinfo"]).to_pydatetime()
            local_dt_cache[timezone_name] = local_dt
        out[feature_col] = _compute_latest_session_counter_value(local_dt, session_cfg)
    return out
