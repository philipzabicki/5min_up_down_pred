import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

LIVE_ROOT_DIR = Path("data/live")
LIVE_SHARED_MARKET_DATA_FILENAME = "polymarket_5m.csv"
LIVE_SHARED_MARKET_DATA_KEY_COLUMNS = (
    "pm_run_started_at_utc",
    "pm_model_hash",
    "record_id",
)
LIVE_RECORD_TIMESTAMP_COLUMNS = (
    "record_snapshot_at",
    "prediction_time",
    "resolved_at",
    "bucket_start",
    "bucket_end",
    "pm_account_sync_at",
    "pm_account_sync_at_entry",
    "pm_account_sync_at_resolve",
)

POLICY_DIAGNOSTIC_COLUMNS = (
    "policy_proba_up",
    "policy_ask_yes",
    "policy_ask_no",
    "policy_fee_yes",
    "policy_fee_no",
    "policy_extra_buffer",
    "policy_ev_yes",
    "policy_ev_no",
    "policy_best_ev",
    "policy_decision",
    "policy_reason",
)

STAKE_DIAGNOSTIC_COLUMNS = (
    "stake_multiplier",
    "required_stake_usdc",
    "effective_stake_usdc",
)

LATENCY_DIAGNOSTIC_COLUMNS = (
    "ws_price_event_delay_ms",
    "ws_volume_event_delay_ms",
    "ws_price_receive_delay_ms",
    "ws_volume_receive_delay_ms",
    "ws_event_delay_ms",
    "ws_receive_delay_ms",
    "ws_component_sync_ms",
    "feature_prep_ms",
    "feature_vector_ms",
    "model_predict_ms",
    "policy_compute_ms",
    "market_prefetch_hit",
    "market_prefetch_age_ms",
    "market_lookup_source",
)

LIVE_PREDICTION_EXPORT_COLUMNS = (
    "record_id",
    "record_snapshot_at",
    "pm_model_hash",
    "pm_policy_hash",
    "pm_run_started_at_utc",
    "prediction_time",
    "resolved_at",
    "bucket_start",
    "bucket_end",
    "proba_up",
    "trade_side",
    "stake_usdc",
    "stake_multiplier",
    "required_stake_usdc",
    "effective_stake_usdc",
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
    "win_rate_policy_resolved",
    "win_rate_closed_trade",
) + POLICY_DIAGNOSTIC_COLUMNS

LIVE_TRADE_EXPORT_COLUMNS = LIVE_PREDICTION_EXPORT_COLUMNS + (
    "entry_fee_raw_usdc",
    "shares_net",
    "pm_mode",
    "pm_market_slug",
    "pm_fees_enabled",
    "pm_fee_rate_bps",
    "pm_fee_source",
    "pm_fee_rate",
    "pm_fee_exponent",
    "pm_fee_round_decimals",
    "pm_min_fee_usdc",
    "pm_order_status",
    *LATENCY_DIAGNOSTIC_COLUMNS,
    "decision_delay_ms",
    "market_lookup_ms",
    "submit_order_ms",
    "execution_ms",
    "pm_order_error",
    "pm_settlement_status",
    "pm_up_best_bid",
    "pm_up_best_ask",
    "pm_down_best_bid",
    "pm_down_best_ask",
    "pm_account_sync_at",
    "pm_account_cash_balance_usdc",
    "pm_account_positions_value_usdc",
    "pm_account_sync_at_entry",
    "pm_account_cash_balance_entry_usdc",
    "pm_account_positions_value_entry_usdc",
    "pm_account_sync_at_resolve",
    "pm_account_cash_balance_resolve_usdc",
    "pm_account_positions_value_resolve_usdc",
)

LIVE_SHARED_MARKET_DATA_COLUMNS = (
    "record_id",
    "pm_model_hash",
    "pm_policy_hash",
    "pm_run_started_at_utc",
    "prediction_time",
    "resolved_at",
    "bucket_start",
    "bucket_end",
    "proba_up",
    "trade_side",
    "price_eps",
    "price_slip",
    "ask_yes",
    "ask_no",
    "stake_usdc",
    "entry_price",
    "entry_fee_usdc",
    "entry_fee_raw_usdc",
    "shares_net",
    "bankroll_after_entry",
    "bankroll_after_resolve",
    "btc_open",
    "btc_high",
    "btc_low",
    "btc_close",
    "btc_volume",
    "bucket_open_price",
    "bucket_close_price",
    "pm_order_price_cap",
    "pm_fees_enabled",
    "pm_fee_rate_bps",
    "pm_fee_source",
    "pm_fee_rate",
    "pm_fee_exponent",
    "pm_fee_round_decimals",
    "pm_min_fee_usdc",
    "pm_up_best_bid",
    "pm_up_best_ask",
    "pm_down_best_bid",
    "pm_down_best_ask",
    "pm_order_status",
    "pm_settlement_status",
    "actual_up",
    "is_correct",
    "trade_is_win",
    "payout_usdc",
    "pnl_usdc",
) + POLICY_DIAGNOSTIC_COLUMNS + LATENCY_DIAGNOSTIC_COLUMNS

LIVE_TRADE_RECORD_PATH_RE = re.compile(
    r"^live_trade_polymarket_"
    r"(?P<symbol>.+?)_"
    r"(?P<interval>\d+[mhd])_"
    r"model_(?P<model_hash>[0-9a-f]{12})_"
    r"policy_(?P<policy_config_hash>[0-9a-f]{12})_"
    r"modeling_(?P<modeling_dataset_config_hash>[0-9a-f]{12})_"
    r"(?P<run_started_at_utc>\d{8}_\d{6})\.csv$"
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
    policy_config_hash,
    modeling_dataset_config_hash,
):
    return Path(live_trade_dir) / (
        f"live_trade_polymarket_{symbol}_{interval}_"
        f"model_{model_hash}_policy_{policy_config_hash}_"
        f"modeling_{modeling_dataset_config_hash}_{run_started_at_utc}.csv"
    )


def build_live_market_data_path(live_root_dir=LIVE_ROOT_DIR):
    return Path(live_root_dir) / LIVE_SHARED_MARKET_DATA_FILENAME


def parse_live_trade_records_path(path):
    match = LIVE_TRADE_RECORD_PATH_RE.match(Path(path).name)
    if match is None:
        return None
    return dict(match.groupdict())


def compute_running_win_rates(records, *, is_resolved, is_traded):
    policy_resolved_rates = []
    closed_trade_rates = []
    policy_resolved_count = 0
    policy_resolved_wins = 0
    closed_trade_count = 0
    closed_trade_wins = 0

    for record in records:
        if not is_resolved(record):
            policy_resolved_rates.append(np.nan)
            closed_trade_rates.append(np.nan)
            continue

        policy_resolved_count += 1
        policy_resolved_wins += int(record.get("is_correct") or 0)
        if is_traded(record):
            closed_trade_count += 1
            closed_trade_wins += int(record.get("trade_is_win") or 0)

        policy_resolved_rates.append(
            float(policy_resolved_wins / policy_resolved_count)
        )
        closed_trade_rates.append(
            float(closed_trade_wins / closed_trade_count)
            if closed_trade_count
            else np.nan
        )

    return policy_resolved_rates, closed_trade_rates


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

    policy_resolved_rates, closed_trade_rates = compute_running_win_rates(
        records,
        is_resolved=is_resolved,
        is_traded=is_traded,
    )
    out["win_rate_policy_resolved"] = policy_resolved_rates
    out["win_rate_closed_trade"] = closed_trade_rates

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


def _dedupe_key_part(value):
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def _acquire_csv_lock(lock_path, timeout_sec=30.0, poll_sec=0.05):
    deadline = time.monotonic() + float(timeout_sec)
    while True:
        try:
            return os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for CSV lock: {lock_path}")
            time.sleep(float(poll_sec))


def _release_csv_lock(fd, lock_path):
    try:
        os.close(fd)
    finally:
        try:
            Path(lock_path).unlink()
        except FileNotFoundError:
            pass


def upsert_records_csv(
    records,
    path,
    *,
    export_columns,
    key_columns=LIVE_SHARED_MARKET_DATA_KEY_COLUMNS,
    timestamp_columns=LIVE_RECORD_TIMESTAMP_COLUMNS,
    is_resolved,
    is_traded,
    record_filter=None,
):
    filtered_records = list(records)
    if record_filter is not None:
        filtered_records = [record for record in filtered_records if record_filter(record)]
    if not filtered_records:
        return

    frame = records_to_export_frame(
        filtered_records,
        export_columns=export_columns,
        timestamp_columns=timestamp_columns,
        is_resolved=is_resolved,
        is_traded=is_traded,
    )
    if frame.empty:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{path}.lock")
    lock_fd = _acquire_csv_lock(lock_path)
    tmp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{int(time.time() * 1_000_000)}.tmp"
    )
    try:
        if path.exists() and path.stat().st_size > 0:
            existing = pd.read_csv(path, dtype=object, keep_default_na=False)
        else:
            existing = pd.DataFrame(columns=list(export_columns))

        for col in export_columns:
            if col not in existing.columns:
                existing[col] = np.nan

        if existing.empty:
            combined = frame.copy()
        else:
            combined = pd.concat((existing, frame), ignore_index=True, sort=False)
        for col in export_columns:
            if col not in combined.columns:
                combined[col] = np.nan

        combined_keys = combined.apply(
            lambda row: tuple(_dedupe_key_part(row.get(col)) for col in key_columns),
            axis=1,
        )
        combined = combined.loc[~combined_keys.duplicated(keep="last")].copy()

        sort_columns = [
            col
            for col in (
                "pm_run_started_at_utc",
                "prediction_time",
                "bucket_start",
                "record_id",
            )
            if col in combined.columns
        ]
        if sort_columns:
            combined = combined.sort_values(
                by=sort_columns,
                kind="stable",
                na_position="last",
            )

        combined = combined.loc[:, list(export_columns)]
        combined.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except FileNotFoundError:
            pass
        _release_csv_lock(lock_fd, lock_path)


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
