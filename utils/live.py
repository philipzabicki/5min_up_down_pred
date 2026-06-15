import atexit
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from utils.project_config import load_active_asset

LIVE_ROOT_DIR = Path("data/live")
LIVE_LOGS_DIR = LIVE_ROOT_DIR / "logs"
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
    "pm_redeem_submitted_at",
    "pm_redeem_confirmed_at",
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
    "intended_stake_usdc",
    "submitted_stake_usdc",
    "filled_stake_usdc",
    "stake_multiplier",
    "required_stake_usdc",
    "effective_stake_usdc",
)

LATENCY_DIAGNOSTIC_COLUMNS = (
    "signal_ready_delay_ms",
    "decision_ready_delay_ms",
    "cycle_complete_delay_ms",
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

LIVE_BASE_EXPORT_COLUMNS = (
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
                               *STAKE_DIAGNOSTIC_COLUMNS,
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

LIVE_TRADE_EXPORT_COLUMNS = LIVE_BASE_EXPORT_COLUMNS + (
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
    "pm_tick_size",
    "pm_order_min_size",
    "pm_order_price_cap",
    "pm_submitted_price",
    "pm_submitted_price_mode",
    "pm_submitted_price_slippage_ticks",
    "pm_submitted_price_error",
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
    "pm_redeem_condition_id",
    "pm_redeem_collateral_token",
    "pm_redeem_ctf_address",
    "pm_redeem_target_address",
    "pm_redeem_relayer_tx_type",
    "pm_redeem_index_sets",
    "pm_redeem_signer_address",
    "pm_redeem_funder_address",
    "pm_redeem_nonce",
    "pm_redeem_tx_id",
    "pm_redeem_tx_hash",
    "pm_redeem_tx_state",
    "pm_redeem_error",
    "pm_redeem_submitted_at",
    "pm_redeem_confirmed_at",
    "pm_settlement_payout_source",
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
                                      *STAKE_DIAGNOSTIC_COLUMNS,
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
                                      "pm_submitted_price",
                                      "pm_submitted_price_mode",
                                      "pm_submitted_price_slippage_ticks",
                                      "pm_submitted_price_error",
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
TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"
TELEGRAM_CONSOLE_MAX_MESSAGE_CHARS = 3900


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


def _telegram_api_post(bot_token, method, payload=None, timeout=5.0):
    response = requests.post(
        f"https://api.telegram.org/bot{bot_token}/{method}",
        data=payload or {},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _resolve_telegram_chat_id(
        bot_token,
        chat_id=None,
        *,
        timeout=5.0,
        api_post=_telegram_api_post,
):
    chat_id = str(chat_id or "").strip()
    if chat_id:
        return chat_id

    try:
        updates = api_post(
            bot_token,
            "getUpdates",
            {
                "limit": 20,
                "timeout": 0,
                "allowed_updates": json.dumps(
                    ["message", "edited_message", "channel_post", "my_chat_member"]
                ),
            },
            timeout,
        ).get("result") or []
    except Exception:
        return None

    chats = []
    for update in updates:
        if not isinstance(update, dict):
            continue
        for payload in (
                update.get("message"),
                update.get("edited_message"),
                update.get("channel_post"),
                update.get("my_chat_member"),
        ):
            if not isinstance(payload, dict):
                continue
            chat = payload.get("chat")
            if isinstance(chat, dict) and "id" in chat:
                chats.append((str(chat["id"]), chat.get("type")))

    chat_ids = list(dict.fromkeys(chat_id for chat_id, _ in chats))
    private_chat_ids = list(
        dict.fromkeys(chat_id for chat_id, chat_type in chats if chat_type == "private")
    )
    if len(private_chat_ids) == 1:
        return private_chat_ids[0]
    if len(chat_ids) == 1:
        return chat_ids[0]
    return None


def send_telegram_message(
        text,
        *,
        bot_token=None,
        chat_id=None,
        timeout=5.0,
        api_post=_telegram_api_post,
):
    message = str(text or "").strip()
    if not message:
        return False

    resolved_bot_token = str(
        bot_token or os.environ.get(TELEGRAM_BOT_TOKEN_ENV, "")
    ).strip()
    if not resolved_bot_token:
        return False

    configured_chat_id = str(
        chat_id or os.environ.get(TELEGRAM_CHAT_ID_ENV, "")
    ).strip()
    resolved_chat_id = _resolve_telegram_chat_id(
        resolved_bot_token,
        chat_id=configured_chat_id,
        timeout=timeout,
        api_post=api_post,
    )
    if not resolved_chat_id:
        return False

    try:
        response = api_post(
            resolved_bot_token,
            "sendMessage",
            {
                "chat_id": resolved_chat_id,
                "text": message[:TELEGRAM_CONSOLE_MAX_MESSAGE_CHARS],
                "disable_web_page_preview": "true",
            },
            timeout,
        )
    except Exception:
        return False
    return not isinstance(response, dict) or bool(response.get("ok", True))


class _TelegramConsoleSink:
    def __init__(
            self,
            bot_token,
            chat_id,
            *,
            timeout=5.0,
            max_message_chars=TELEGRAM_CONSOLE_MAX_MESSAGE_CHARS,
            max_queue_size=1000,
            api_post=_telegram_api_post,
    ):
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.timeout = timeout
        self.max_message_chars = max_message_chars
        self.api_post = api_post
        self.lock = threading.RLock()
        self.buffer = ""
        self.closed = False
        self.stop_marker = object()
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.worker = threading.Thread(
            target=self._run,
            name="telegram-console-mirror",
            daemon=True,
        )
        self.worker.start()

    def write(self, text):
        if not text:
            return 0
        with self.lock:
            if self.closed:
                return len(text)
            self.buffer += str(text)
            lines = self.buffer.splitlines(keepends=True)
            if lines and not lines[-1].endswith(("\n", "\r")):
                self.buffer = lines.pop()
            else:
                self.buffer = ""
        for line in lines:
            self._enqueue(line)
        return len(text)

    def flush(self):
        with self.lock:
            text = self.buffer
            self.buffer = ""
        if text:
            self._enqueue(text)

    def close(self):
        self.flush()
        with self.lock:
            if self.closed:
                return
            self.closed = True
        try:
            self.queue.put(self.stop_marker, timeout=1.0)
        except queue.Full:
            return
        self.worker.join(timeout=2.0)

    def _enqueue(self, text):
        try:
            self.queue.put_nowait(str(text))
        except queue.Full:
            pass

    def _run(self):
        while True:
            item = self.queue.get()
            if item is self.stop_marker:
                break
            for message in self._split_message(item):
                self._send_message(message)

    def _split_message(self, text):
        message = str(text).rstrip("\r\n")
        if not message.strip():
            return []
        return [
            message[start: start + self.max_message_chars]
            for start in range(0, len(message), self.max_message_chars)
        ]

    def _send_message(self, text):
        try:
            self.api_post(
                self.bot_token,
                "sendMessage",
                {
                    "chat_id": self.chat_id,
                    "text": text,
                    "disable_web_page_preview": "true",
                },
                self.timeout,
            )
        except (requests.RequestException, ValueError):
            return


class _TeeStream:
    def __init__(self, primary_stream, log_stream, lock, extra_streams=()):
        self.primary_stream = primary_stream
        self.log_stream = log_stream
        self.lock = lock
        self.extra_streams = tuple(extra_streams or ())

    def write(self, text):
        with self.lock:
            self.primary_stream.write(text)
            self.log_stream.write(text)
            for stream in self.extra_streams:
                stream.write(text)
        return len(text)

    def flush(self):
        with self.lock:
            self.primary_stream.flush()
            self.log_stream.flush()
            for stream in self.extra_streams:
                stream.flush()

    def __getattr__(self, name):
        return getattr(self.primary_stream, name)


def build_live_console_log_path(run_name, run_started_at_utc=None, logs_dir=LIVE_LOGS_DIR):
    timestamp = (
        str(run_started_at_utc)
        if run_started_at_utc is not None
        else pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    )
    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_name).strip())
    safe_run_name = safe_run_name.strip("._-") or "live"
    return Path(logs_dir) / f"{safe_run_name}_{timestamp}.log"


def setup_live_console_logging(
        run_name,
        run_started_at_utc=None,
        logs_dir=LIVE_LOGS_DIR,
        *,
        telegram=False,
        telegram_bot_token=None,
        telegram_chat_id=None,
):
    log_path = build_live_console_log_path(
        run_name,
        run_started_at_utc=run_started_at_utc,
        logs_dir=logs_dir,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    lock = threading.RLock()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    telegram_sink = None
    telegram_status = None
    if telegram:
        bot_token = str(
            telegram_bot_token or os.environ.get(TELEGRAM_BOT_TOKEN_ENV, "")
        ).strip()
        configured_chat_id = str(
            telegram_chat_id or os.environ.get(TELEGRAM_CHAT_ID_ENV, "")
        ).strip()
        if not bot_token:
            telegram_status = (
                f"[telegram] console mirror disabled: missing {TELEGRAM_BOT_TOKEN_ENV}"
            )
        else:
            resolved_chat_id = _resolve_telegram_chat_id(
                bot_token,
                chat_id=configured_chat_id,
            )
            if resolved_chat_id:
                telegram_sink = _TelegramConsoleSink(bot_token, resolved_chat_id)
                telegram_status = "[telegram] console mirror enabled"
            else:
                telegram_status = (
                    "[telegram] console mirror disabled: missing "
                    f"{TELEGRAM_CHAT_ID_ENV} and auto-detect found no unique chat"
                )

    extra_streams = (telegram_sink,) if telegram_sink is not None else ()
    stdout_tee = _TeeStream(original_stdout, log_file, lock, extra_streams)
    stderr_tee = _TeeStream(original_stderr, log_file, lock, extra_streams)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    def close_log_file():
        stdout_tee.flush()
        stderr_tee.flush()
        if telegram_sink is not None:
            telegram_sink.close()
        if sys.stdout is stdout_tee:
            sys.stdout = original_stdout
        if sys.stderr is stderr_tee:
            sys.stderr = original_stderr
        log_file.close()

    atexit.register(close_log_file)
    print(f"[log] console output file: {log_path}")
    if telegram_status:
        print(telegram_status)
    return log_path


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


def build_live_market_data_path(live_root_dir=LIVE_ROOT_DIR, asset=None):
    asset_name = str(asset or "").strip().upper()
    if not asset_name:
        try:
            asset_name = load_active_asset()
        except Exception:
            asset_name = ""
    if asset_name:
        return Path(live_root_dir) / asset_name / LIVE_SHARED_MARKET_DATA_FILENAME
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


def _finite_float(value, default=np.nan):
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _first_positive_float(*values):
    for value in values:
        parsed = _finite_float(value)
        if np.isfinite(parsed) and parsed > 0.0:
            return float(parsed)
    return float("nan")


def _zero_near(value, tolerance=1e-6):
    if np.isfinite(value) and abs(value) <= float(tolerance):
        return 0.0
    return value


def resolve_polymarket_closed_position_settlement(
        record,
        closed_position,
        *,
        prefer_data_api_pnl=False,
):
    record = record or {}
    closed_position = closed_position or {}

    avg_price = _finite_float(closed_position.get("avgPrice"))
    closed_shares = _finite_float(closed_position.get("totalBought"))
    realized_pnl = _finite_float(closed_position.get("realizedPnl"))

    shares_net = _first_positive_float(
        closed_shares,
        record.get("shares_net"),
        record.get("entry_shares_net_orig"),
    )
    stake_usdc = _first_positive_float(
        record.get("filled_stake_usdc"),
        record.get("entry_stake_usdc_orig"),
        record.get("submitted_stake_usdc"),
        record.get("intended_stake_usdc"),
        record.get("stake_usdc"),
    )
    if (
            not np.isfinite(stake_usdc)
            and np.isfinite(shares_net)
            and np.isfinite(avg_price)
            and avg_price > 0.0
    ):
        stake_usdc = float(shares_net * avg_price)

    actual_up = _finite_float(record.get("actual_up"))
    trade_side = str(record.get("trade_side", "") or "").strip().lower()
    side_won = None
    if actual_up in {0.0, 1.0}:
        if trade_side == "yes":
            side_won = bool(actual_up == 1.0)
        elif trade_side == "no":
            side_won = bool(actual_up == 0.0)

    payout_usdc = float("nan")
    pnl_usdc = float("nan")
    payout_source = "data_api_closed_positions"

    if prefer_data_api_pnl and np.isfinite(realized_pnl):
        pnl_usdc = float(realized_pnl)
        if np.isfinite(stake_usdc):
            payout_usdc = _zero_near(float(stake_usdc + pnl_usdc))
    elif side_won is not None and np.isfinite(stake_usdc):
        payout_usdc = float(shares_net) if side_won and np.isfinite(shares_net) else 0.0
        pnl_usdc = float(payout_usdc - stake_usdc)
        payout_source = "settlement_outcome_shares"
    elif np.isfinite(realized_pnl):
        pnl_usdc = float(realized_pnl)
        if np.isfinite(stake_usdc):
            payout_usdc = _zero_near(float(stake_usdc + pnl_usdc))

    trade_is_win = int(pnl_usdc > 0.0) if np.isfinite(pnl_usdc) else None

    return {
        "closed_avg_price": avg_price,
        "closed_total_bought": closed_shares,
        "closed_realized_pnl": realized_pnl,
        "stake_usdc": stake_usdc,
        "shares_net": shares_net,
        "payout_usdc": payout_usdc,
        "pnl_usdc": pnl_usdc,
        "trade_is_win": trade_is_win,
        "payout_source": payout_source,
    }


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
