import json
from pathlib import Path

import numpy as np
import pandas as pd

LIVE_RECORD_TIMESTAMP_COLUMNS = (
    "record_snapshot_at",
    "prediction_time",
    "resolved_at",
    "bucket_start",
    "bucket_end",
)

LIVE_PREDICTION_EXPORT_COLUMNS = (
    "record_id",
    "record_snapshot_at",
    "pm_model_hash",
    "pm_run_started_at_utc",
    "prediction_time",
    "resolved_at",
    "bucket_start",
    "bucket_end",
    "proba_up",
    "signal_up",
    "kelly_side",
    "kelly_reason",
    "kelly_edge",
    "kelly_fraction",
    "stake_usdc",
    "entry_price",
    "entry_fee_usdc",
    "bankroll_after_entry",
    "bankroll_after_resolve",
    "bucket_open_price",
    "bucket_close_price",
    "actual_up",
    "is_correct",
    "trade_is_win",
    "payout_usdc",
    "pnl_usdc",
    "win_rate_resolved",
    "win_rate_traded",
)

LIVE_TRADE_EXPORT_COLUMNS = LIVE_PREDICTION_EXPORT_COLUMNS + (
    "pm_mode",
    "pm_market_slug",
    "pm_order_status",
    "pm_order_error",
    "pm_settlement_status",
    "shares_net",
    "pm_up_best_bid",
    "pm_up_best_ask",
    "pm_down_best_bid",
    "pm_down_best_ask",
)


def as_utc_timestamp(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def interval_to_timedelta(interval):
    if interval.endswith("m"):
        return pd.Timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return pd.Timedelta(hours=int(interval[:-1]))
    if interval.endswith("d"):
        return pd.Timedelta(days=int(interval[:-1]))
    raise ValueError(f"Unsupported interval: {interval}")


def interval_to_floor_rule(interval):
    if interval.endswith("m"):
        return f"{int(interval[:-1])}min"
    if interval.endswith("h"):
        return f"{int(interval[:-1])}h"
    if interval.endswith("d"):
        return f"{int(interval[:-1])}d"
    raise ValueError(f"Unsupported floor rule interval: {interval}")


def build_live_trade_records_path(
    live_trade_dir,
    symbol,
    interval,
    run_started_at_utc,
    model_hash,
    kelly_config_hash,
    modeling_dataset_config_hash,
):
    return Path(live_trade_dir) / (
        f"live_trade_polymarket_{symbol}_{interval}_"
        f"model_{model_hash}_kelly_{kelly_config_hash}_"
        f"modeling_{modeling_dataset_config_hash}_{run_started_at_utc}.csv"
    )


def compute_running_win_rates(records, *, is_resolved, is_traded):
    resolved_rates = []
    traded_rates = []
    resolved_count = 0
    resolved_wins = 0
    traded_count = 0
    traded_wins = 0

    for record in records:
        if not is_resolved(record):
            resolved_rates.append(np.nan)
            traded_rates.append(np.nan)
            continue

        resolved_count += 1
        resolved_wins += int(record.get("is_correct") or 0)
        if is_traded(record):
            traded_count += 1
            traded_wins += int(record.get("trade_is_win") or 0)

        resolved_rates.append(float(resolved_wins / resolved_count))
        traded_rates.append(
            float(traded_wins / traded_count) if traded_count else np.nan
        )

    return resolved_rates, traded_rates


def serialize_timestamp_columns(frame, columns):
    for col in columns:
        if col not in frame.columns:
            continue
        frame[col] = frame[col].map(
            lambda value: value.isoformat() if isinstance(value, pd.Timestamp) else ""
        )


def records_to_export_frame(
    records,
    *,
    export_columns,
    timestamp_columns=LIVE_RECORD_TIMESTAMP_COLUMNS,
    is_resolved,
    is_traded,
):
    out = pd.DataFrame(records)
    if out.empty:
        return pd.DataFrame(columns=list(export_columns))

    resolved_rates, traded_rates = compute_running_win_rates(
        records,
        is_resolved=is_resolved,
        is_traded=is_traded,
    )
    out["win_rate_resolved"] = resolved_rates
    out["win_rate_traded"] = traded_rates

    for col in export_columns:
        if col not in out.columns:
            out[col] = np.nan
    out = out.loc[:, list(export_columns)]
    serialize_timestamp_columns(out, timestamp_columns)
    return out


def write_records_csv(
    records,
    path,
    *,
    export_columns,
    timestamp_columns=LIVE_RECORD_TIMESTAMP_COLUMNS,
    is_resolved,
    is_traded,
):
    frame = records_to_export_frame(
        records,
        export_columns=export_columns,
        timestamp_columns=timestamp_columns,
        is_resolved=is_resolved,
        is_traded=is_traded,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value if isinstance(value, str) else str(value)


def write_records_state(records, path):
    payload = [_json_safe(record) for record in records]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )


def read_records_state(path, timestamp_columns=LIVE_RECORD_TIMESTAMP_COLUMNS):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Live records state must contain a list.")

    loaded = []
    for raw_record in records:
        if not isinstance(raw_record, dict):
            continue
        record = dict(raw_record)
        for col in timestamp_columns:
            value = record.get(col)
            if value in {None, ""}:
                record[col] = None
                continue
            record[col] = pd.to_datetime(value, errors="coerce", utc=True)
            if pd.isna(record[col]):
                record[col] = None
        loaded.append(record)
    return loaded
