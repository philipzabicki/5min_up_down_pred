import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import requests
from eth_abi import encode as abi_encode
from eth_account import Account
from eth_utils import keccak, to_checksum_address
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    AssetType,
    BalanceAllowanceParams,
    MarketOrderArgs,
    PartialCreateOrderOptions,
    OrderType,
)
from py_clob_client.http_helpers import helpers as pyclob_http_helpers
from py_clob_client.order_builder.constants import BUY, SELL
from websocket import WebSocketApp

try:
    from py_builder_relayer_client.builder.safe import build_safe_transaction_request
    from py_builder_relayer_client.config import (
        get_contract_config as get_relayer_contract_config,
    )
    from py_builder_relayer_client.models import (
        OperationType as RelayerOperationType,
        SafeTransaction as RelayerSafeTransaction,
        SafeTransactionArgs as RelayerSafeTransactionArgs,
    )
    from py_builder_relayer_client.signer import Signer as RelayerSigner

    RELAYER_SDK_AVAILABLE = True
except ImportError:
    build_safe_transaction_request = None
    get_relayer_contract_config = None
    RelayerOperationType = None
    RelayerSafeTransaction = None
    RelayerSafeTransactionArgs = None
    RelayerSigner = None
    RELAYER_SDK_AVAILABLE = False

from live_predict_binance import (
    INTERVAL_DELTA,
    LIVE_TRADE_DIR,
    LivePredictor,
    MAX_WS_RECONNECT_DELAY_SEC,
    PRICE_SOURCE,
    VOLUME_SOURCE,
    PREDICTIONS_EXPORT_COLUMNS as BASE_PREDICTIONS_EXPORT_COLUMNS,
    WS_PING_INTERVAL_SEC,
    WS_PING_TIMEOUT_SEC,
    WS_URL,
    build_live_trade_records_path,
)
from kelly_utils import adjust_probability_for_kelly


DEFAULT_GAMMA_HOST = "https://gamma-api.polymarket.com"
DEFAULT_CLOB_HOST = "https://clob.polymarket.com"
DEFAULT_DATA_API_HOST = "https://data-api.polymarket.com"
DEFAULT_RELAYER_HOST = "https://relayer-v2.polymarket.com"
ENV_FILE_PATH = Path(__file__).resolve().with_name(".env")
POLYMARKET_ORDER_PRICE_CAP = 0.537
POLYMARKET_CRYPTO_FEE_EXPONENT = 2.0
POLYMARKET_FEE_ROUND_DECIMALS = 4
POLYMARKET_MIN_FEE_USDC = 0.0001
POLYMARKET_COLLATERAL_DECIMALS = 6
POLYMARKET_VALID_TICK_SIZES = {"0.1", "0.01", "0.001", "0.0001"}
POLYMARKET_CLOSED_POSITIONS_PAGE_LIMIT = 50
POLYMARKET_BACKGROUND_SYNC_MIN_INTERVAL_SEC = 2.0
POLYMARKET_USDC_E_ADDRESS = to_checksum_address(
    "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
)
POLYMARKET_CTF_ADDRESS = to_checksum_address(
    "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
)
POLYMARKET_ZERO_BYTES32 = "0x" + ("0" * 64)
POLYMARKET_BINARY_INDEX_SETS = (1, 2)
POLYMARKET_RELAYER_TERMINAL_STATES = {
    "STATE_CONFIRMED",
    "STATE_FAILED",
    "STATE_INVALID",
}
POLYMARKET_RELAYER_PENDING_STATES = {
    "STATE_NEW",
    "STATE_EXECUTED",
    "STATE_MINED",
}
LIVE_TRADE_LATENCY_TIMESTAMP_COLUMNS = (
    "pm_binance_event_at",
    "pm_ws_received_at",
    "pm_predict_started_at",
    "pm_submit_ready_at",
    "pm_order_dispatch_at",
    "pm_order_response_at",
)
LIVE_TRADE_LATENCY_NUMERIC_COLUMNS = (
    "pm_market_lookup_attempts",
    "pm_latency_sync_missing_candles_ms",
    "pm_latency_upsert_closed_candle_ms",
    "pm_latency_volume_profile_ms",
    "pm_latency_feature_vector_ms",
    "pm_latency_model_predict_ms",
    "pm_latency_market_lookup_total_ms",
    "pm_latency_market_lookup_queue_ms",
    "pm_latency_market_lookup_wait_ms",
    "pm_latency_market_lookup_retry_sleep_ms",
    "pm_latency_market_lookup_gamma_ms",
    "pm_latency_market_lookup_up_book_ms",
    "pm_latency_market_lookup_down_book_ms",
    "pm_latency_market_lookup_fee_rate_ms",
    "pm_latency_recommendation_ms",
    "pm_latency_submit_total_ms",
    "pm_latency_create_market_order_ms",
    "pm_latency_post_order_ms",
    "pm_latency_predict_total_ms",
    "pm_latency_on_message_total_ms",
    "pm_market_start_to_binance_event_ms",
    "pm_market_start_to_ws_received_ms",
    "pm_market_start_to_submit_ready_ms",
    "pm_market_start_to_order_dispatch_ms",
    "pm_market_start_to_order_response_ms",
    "pm_latency_binance_event_to_ws_received_ms",
    "pm_latency_ws_to_submit_ready_ms",
    "pm_latency_ws_to_order_dispatch_ms",
    "pm_latency_ws_to_order_response_ms",
)
LIVE_TRADE_EXPORT_COLUMNS = (
    "record_id",
    "pm_model_hash",
    "pm_run_started_at_utc",
    "record_snapshot_at",
) + tuple(BASE_PREDICTIONS_EXPORT_COLUMNS) + (
    "pm_mode",
    "pm_series_slug",
    "pm_execution_mode",
    "pm_market_slug",
    "pm_market_question",
    "pm_market_end",
    "pm_condition_id",
    "pm_up_token_id",
    "pm_down_token_id",
    "pm_selected_token_id",
    "pm_accepting_orders",
    "pm_restricted",
    "pm_fees_enabled",
    "pm_fee_rate_bps",
    "pm_tick_size",
    "pm_order_min_size",
    "pm_order_price_cap",
    "pm_position_size",
    "pm_position_current_value",
    "pm_position_redeemable",
    "pm_settlement_status",
    "pm_account_sync_at",
    "pm_account_sync_reason",
    "pm_redeem_tx_id",
    "pm_redeem_tx_hash",
    "pm_redeem_tx_state",
    "pm_redeem_error",
    "pm_up_best_bid",
    "pm_up_best_ask",
    "pm_down_best_bid",
    "pm_down_best_ask",
    "pm_seconds_to_close",
    "pm_order_status",
    "pm_order_response",
    "pm_allowance_info",
    "pm_market_error",
    "pm_order_error",
    "pm_exit_order_status",
    "pm_exit_order_response",
    "pm_exit_order_error",
    "pm_exit_reason",
    "pm_exit_price",
    "pm_exit_shares",
    "pm_exit_fee_usdc",
    "pm_exit_proceeds_usdc",
    *LIVE_TRADE_LATENCY_TIMESTAMP_COLUMNS,
    *LIVE_TRADE_LATENCY_NUMERIC_COLUMNS,
)


def _env_text(name, default=""):
    raw = os.getenv(name)
    return raw.strip() if raw is not None else default


def _env_int(name, default):
    raw = os.getenv(name)
    return int(raw) if raw is not None and raw.strip() else int(default)


def _env_float(name, default):
    raw = os.getenv(name)
    return float(raw) if raw is not None and raw.strip() else float(default)


def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _strip_wrapping_quotes(value):
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_env_file(env_path=ENV_FILE_PATH):
    if not env_path.exists():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ[key] = _strip_wrapping_quotes(value)
    return True


_load_env_file()


def _configure_clob_http_client(timeout_sec):
    try:
        timeout_sec = float(timeout_sec)
    except (TypeError, ValueError):
        return
    if not np.isfinite(timeout_sec) or timeout_sec <= 0.0:
        return

    previous_client = getattr(pyclob_http_helpers, "_http_client", None)
    pyclob_http_helpers._http_client = httpx.Client(
        http2=True,
        timeout=httpx.Timeout(timeout=timeout_sec),
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
    )
    if previous_client is not None and previous_client is not pyclob_http_helpers._http_client:
        close_fn = getattr(previous_client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def _safe_float(value, default=float("nan")):
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_text(value, default=""):
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return default
        return text
    try:
        if bool(pd.isna(value)):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def _parse_json_list(value):
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Expected JSON list payload, got: {type(value).__name__}")


def _best_price(levels, side):
    if not levels:
        return float("nan")
    prices = [_safe_float(level.get("price")) for level in levels]
    prices = [price for price in prices if np.isfinite(price)]
    if not prices:
        return float("nan")
    return float(min(prices) if side == "ask" else max(prices))


def _json_compact(payload):
    try:
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    except TypeError:
        return str(payload)


def _http_status_code(exc):
    response = getattr(exc, "response", None)
    if response is None:
        return None
    return int(getattr(response, "status_code", 0) or 0)


def _utc_now():
    return pd.Timestamp.now(tz="UTC")


def _elapsed_ms(start_perf, end_perf=None):
    if start_perf is None:
        return float("nan")
    end_perf = time.perf_counter() if end_perf is None else float(end_perf)
    return float(max(end_perf - float(start_perf), 0.0) * 1000.0)


def _timestamp_diff_ms(start_ts, end_ts):
    if start_ts is None or end_ts is None:
        return float("nan")
    try:
        start_ts = pd.Timestamp(start_ts)
        end_ts = pd.Timestamp(end_ts)
    except Exception:
        return float("nan")
    return float((end_ts - start_ts).total_seconds() * 1000.0)


def _timestamp_from_ms(value):
    if value is None or value == "":
        return None
    try:
        return pd.to_datetime(int(value), unit="ms", utc=True)
    except (TypeError, ValueError):
        return None


def _timed_call(func, *args, **kwargs):
    started_perf = time.perf_counter()
    value = func(*args, **kwargs)
    return value, _elapsed_ms(started_perf)


def _latency_trace_defaults():
    payload = {col: None for col in LIVE_TRADE_LATENCY_TIMESTAMP_COLUMNS}
    payload.update({col: np.nan for col in LIVE_TRADE_LATENCY_NUMERIC_COLUMNS})
    return payload


def _latency_accumulate_ms(trace, key, elapsed_ms):
    if trace is None:
        return
    try:
        elapsed_ms = float(elapsed_ms)
    except (TypeError, ValueError):
        return
    if not np.isfinite(elapsed_ms):
        return
    current = trace.get(key, 0.0)
    try:
        current = float(current)
    except (TypeError, ValueError):
        current = 0.0
    if not np.isfinite(current):
        current = 0.0
    trace[key] = float(current + elapsed_ms)


def _latency_increment(trace, key, amount=1):
    if trace is None:
        return
    current = trace.get(key, 0)
    try:
        current = int(current)
    except (TypeError, ValueError):
        current = 0
    trace[key] = int(current + int(amount))


def _latency_set_from_perf(trace, key, started_perf, ended_perf):
    if trace is None:
        return
    elapsed_ms = _elapsed_ms(started_perf, ended_perf)
    if np.isfinite(elapsed_ms):
        trace[key] = float(elapsed_ms)


def _stable_record_id(record):
    record_id = _safe_text(record.get("record_id"))
    if record_id:
        return record_id

    bucket_start = record.get("bucket_start")
    if isinstance(bucket_start, pd.Timestamp):
        return f"bucket:{bucket_start.isoformat()}"

    asset = _safe_text(record.get("pm_selected_token_id"))
    if asset:
        return f"external:{asset}"

    prediction_time = record.get("prediction_time")
    if isinstance(prediction_time, pd.Timestamp):
        return f"prediction:{prediction_time.isoformat()}"

    return ""


def _record_state_signature(record):
    def _normalize(value):
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, (np.floating, float)):
            value = float(value)
            if not np.isfinite(value):
                return None
            return value
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if value is None:
            return None
        return str(value)

    payload = {
        key: _normalize(value)
        for key, value in record.items()
        if key != "record_snapshot_at"
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _mark_record_dirty(dirty_ids, record):
    record_id = _stable_record_id(record)
    if record_id:
        dirty_ids.add(record_id)


def _latency_snapshot(bucket_start, trace):
    payload = _latency_trace_defaults()
    if trace:
        for col in LIVE_TRADE_LATENCY_TIMESTAMP_COLUMNS:
            payload[col] = trace.get(col)
        for col in LIVE_TRADE_LATENCY_NUMERIC_COLUMNS:
            value = trace.get(col)
            if value is None:
                continue
            if col == "pm_market_lookup_attempts":
                try:
                    payload[col] = int(value)
                except (TypeError, ValueError):
                    payload[col] = np.nan
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float("nan")
            payload[col] = value if np.isfinite(value) else np.nan

    market_start = pd.Timestamp(bucket_start)
    payload["pm_market_start_to_binance_event_ms"] = _timestamp_diff_ms(
        market_start, payload["pm_binance_event_at"]
    )
    payload["pm_market_start_to_ws_received_ms"] = _timestamp_diff_ms(
        market_start, payload["pm_ws_received_at"]
    )
    payload["pm_market_start_to_submit_ready_ms"] = _timestamp_diff_ms(
        market_start, payload["pm_submit_ready_at"]
    )
    payload["pm_market_start_to_order_dispatch_ms"] = _timestamp_diff_ms(
        market_start, payload["pm_order_dispatch_at"]
    )
    payload["pm_market_start_to_order_response_ms"] = _timestamp_diff_ms(
        market_start, payload["pm_order_response_at"]
    )
    if not np.isfinite(payload["pm_latency_binance_event_to_ws_received_ms"]):
        payload["pm_latency_binance_event_to_ws_received_ms"] = _timestamp_diff_ms(
            payload["pm_binance_event_at"], payload["pm_ws_received_at"]
        )
    if not np.isfinite(payload["pm_latency_ws_to_submit_ready_ms"]):
        payload["pm_latency_ws_to_submit_ready_ms"] = _timestamp_diff_ms(
            payload["pm_ws_received_at"], payload["pm_submit_ready_at"]
        )
    if not np.isfinite(payload["pm_latency_ws_to_order_dispatch_ms"]):
        payload["pm_latency_ws_to_order_dispatch_ms"] = _timestamp_diff_ms(
            payload["pm_ws_received_at"], payload["pm_order_dispatch_at"]
        )
    if not np.isfinite(payload["pm_latency_ws_to_order_response_ms"]):
        payload["pm_latency_ws_to_order_response_ms"] = _timestamp_diff_ms(
            payload["pm_ws_received_at"], payload["pm_order_response_at"]
        )
    return payload


def _polymarket_fee_rate_from_bps(base_fee_bps):
    # Crypto markets expose a base fee in bps, while the fee formula uses the
    # corresponding fee-rate scalar. For Polymarket crypto this is 1000 bps -> 0.25.
    return max(float(base_fee_bps), 0.0) / 4000.0


def _collateral_balance_to_usdc(raw_balance):
    if raw_balance is None or raw_balance == "":
        return float("nan")

    try:
        if isinstance(raw_balance, str):
            raw_balance = raw_balance.strip()
        balance_int = int(raw_balance)
        return float(balance_int / (10**POLYMARKET_COLLATERAL_DECIMALS))
    except (TypeError, ValueError):
        return _safe_float(raw_balance)


def _tick_size_literal(value):
    tick_size = _safe_float(value)
    if not np.isfinite(tick_size) or tick_size <= 0.0:
        return None
    text = format(float(tick_size), ".4f").rstrip("0").rstrip(".")
    return text if text in POLYMARKET_VALID_TICK_SIZES else None


def _partial_create_order_options(tick_size, neg_risk):
    kwargs = {}
    tick_size_text = _tick_size_literal(tick_size)
    if tick_size_text is not None:
        kwargs["tick_size"] = tick_size_text
    if neg_risk is not None:
        kwargs["neg_risk"] = bool(neg_risk)
    return PartialCreateOrderOptions(**kwargs) if kwargs else None


class PolymarketSettings:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class PolymarketMarketSnapshot:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _path_compare_key(path):
    return os.path.normcase(str(Path(path).resolve()))


def resolve_polymarket_records_path(default_records_path, model_hash):
    override_text = _env_text("POLY_RECORDS_PATH", "")
    if not override_text:
        return default_records_path

    candidate = Path(override_text)
    if _path_compare_key(candidate.parent) != _path_compare_key(LIVE_TRADE_DIR):
        print(
            "Ignoring POLY_RECORDS_PATH outside data/live/trade: "
            f"{candidate}. Using default: {default_records_path}"
        )
        return default_records_path

    required_model_token = f"model_{model_hash}"
    if required_model_token not in candidate.stem:
        print(
            "Ignoring POLY_RECORDS_PATH without required model hash token "
            f"'{required_model_token}': {candidate.name}. "
            f"Using default: {default_records_path.name}"
        )
        return default_records_path

    return candidate


def load_polymarket_settings(default_records_path, model_hash):
    return PolymarketSettings(
        gamma_host=_env_text("POLY_GAMMA_HOST", DEFAULT_GAMMA_HOST),
        clob_host=_env_text("POLY_CLOB_HOST", DEFAULT_CLOB_HOST),
        data_api_host=_env_text("POLY_DATA_API_HOST", DEFAULT_DATA_API_HOST),
        relayer_host=_env_text("POLY_RELAYER_HOST", DEFAULT_RELAYER_HOST),
        series_slug=_env_text("POLY_SERIES_SLUG", "btc-up-or-down-5m"),
        market_slug_prefix=_env_text("POLY_MARKET_SLUG_PREFIX", "btc-updown-5m"),
        market_slug_override=_env_text("POLY_MARKET_SLUG_OVERRIDE", ""),
        paper_mode=_env_bool("POLY_PAPER_MODE", True),
        disable_order_submission=_env_bool("POLY_DISABLE_ORDER_SUBMIT", False),
        signature_type=_env_int("POLY_SIGNATURE_TYPE", 0),
        chain_id=_env_int("POLY_CHAIN_ID", 137),
        private_key=_env_text("POLY_PRIVATE_KEY", ""),
        funder=_env_text("POLY_FUNDER_ADDRESS", ""),
        max_exposure_usdc=_env_float("POLY_MAX_EXPOSURE_USDC", math.inf),
        max_bankroll_usdc=_env_float("POLY_MAX_BANKROLL_USDC", math.inf),
        no_trade_last_seconds=_env_int("POLY_NO_TRADE_LAST_SECONDS", 20),
        start_bankroll_usdc=_env_float("POLY_START_BANKROLL_USDC", 1000.0),
        records_path=resolve_polymarket_records_path(default_records_path, model_hash),
        market_request_timeout_sec=_env_float("POLY_MARKET_REQUEST_TIMEOUT_SEC", 3.0),
        clob_http_timeout_sec=_env_float(
            "POLY_CLOB_HTTP_TIMEOUT_SEC",
            _env_float("POLY_MARKET_REQUEST_TIMEOUT_SEC", 3.0),
        ),
        market_lookup_max_wait_ms=_env_int("POLY_MARKET_LOOKUP_MAX_WAIT_MS", 2500),
        market_lookup_retry_ms=_env_int("POLY_MARKET_LOOKUP_RETRY_MS", 100),
        market_lookup_prefetch_lead_ms=_env_int(
            "POLY_MARKET_LOOKUP_PREFETCH_LEAD_MS", 1200
        ),
        market_lookup_prefetch_max_age_ms=_env_int(
            "POLY_MARKET_LOOKUP_PREFETCH_MAX_AGE_MS", 2500
        ),
        execution_mode=_env_text("POLY_EXECUTION_MODE", "fok").lower(),
        relayer_api_key=_env_text("POLY_RELAYER_API_KEY", ""),
        relayer_api_key_address=_env_text("POLY_RELAYER_API_KEY_ADDRESS", ""),

        resume_existing_records=_env_bool("POLY_RESUME_EXISTING_RECORDS", True),
        import_untracked_open_positions=_env_bool("POLY_IMPORT_UNTRACKED_OPEN_POSITIONS", False),

        enable_exit_orders=_env_bool("POLY_ENABLE_EXIT_ORDERS", True),
        exit_min_profit_usdc=_env_float("POLY_EXIT_MIN_PROFIT_USDC", 0.15),
        exit_min_roi=_env_float("POLY_EXIT_MIN_ROI", 0.01),
        exit_min_seconds_to_close=_env_int("POLY_EXIT_MIN_SECONDS_TO_CLOSE", 45),

        redeem_resolved_positions=_env_bool("POLY_REDEEM_RESOLVED_POSITIONS", True),
    )


class PolymarketLiveTrader(LivePredictor):
    def __init__(self):
        super().__init__()
        default_records_path = build_live_trade_records_path(
            run_started_at_utc=self.run_started_at_utc,
            model_hash=self.model_hash,
            kelly_config_hash=self.kelly_config_hash,
            modeling_dataset_config_hash=self.modeling_dataset_config_hash,
        )
        self.pm_cfg = load_polymarket_settings(default_records_path, self.model_hash)
        _configure_clob_http_client(self.pm_cfg.clob_http_timeout_sec)
        self.live_kelly_sizing = {
            "fractional_kelly": float(self.kelly_runtime["fractional_kelly"]),
            "cap": float(self.kelly_runtime["cap"]),
            "min_edge": float(self.kelly_runtime["min_edge"]),
            "prob_shrink": float(self.kelly_runtime["prob_shrink"]),
            "min_stake_usdc": float(self.kelly_runtime["min_stake_usdc"]),
        }
        if self.pm_cfg.execution_mode not in {"fok"}:
            raise NotImplementedError(
                "Unsupported POLY_EXECUTION_MODE. Supported values: ['fok']"
            )
        self.pm_signer_address = (
            Account.from_key(self.pm_cfg.private_key).address
            if self.pm_cfg.private_key
            else ""
        )
        self.pm_relayer_api_key_address = (
            self.pm_cfg.relayer_api_key_address or self.pm_signer_address
        )
        self.live_bankroll_usdc = self._capped_trading_bankroll_usdc(
            self.pm_cfg.start_bankroll_usdc
        )
        self.predictions_path = self.pm_cfg.records_path
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)

        self.pm_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        self.pm_session.mount("https://", adapter)
        self.pm_session.headers.update(
            {"User-Agent": "5min_up_down_pred/live_trade.py"}
        )
        self.pm_io_pool = ThreadPoolExecutor(max_workers=3)
        self.pm_lookup_pool = ThreadPoolExecutor(max_workers=1)
        self.pm_bg_pool = ThreadPoolExecutor(max_workers=1)
        self.records_lock = threading.Lock()
        self.pm_save_lock = threading.Lock()
        self.pm_bg_lock = threading.Lock()
        self.pm_market_prefetch_lock = threading.Lock()
        self.pm_bg_future = None
        self.pm_bg_pending_reason = ""
        self.pm_bg_last_started = 0.0
        self.pm_market_prefetch_timer = None
        self.pm_market_prefetch_bucket_start = None
        self.pm_market_prefetch_future = None
        self.pm_cash_balance_usdc = float("nan")
        self.pm_positions_value_usdc = float("nan")
        self.pm_last_account_sync_at = ""
        self.pm_last_account_sync_reason = ""
        self.pm_relayer_contract_config = (
            get_relayer_contract_config(self.pm_cfg.chain_id)
            if RELAYER_SDK_AVAILABLE
            else None
        )
        self.pm_relayer_signer = (
            RelayerSigner(self.pm_cfg.private_key, self.pm_cfg.chain_id)
            if RELAYER_SDK_AVAILABLE and self.pm_cfg.private_key
            else None
        )
        self.pm_relayer_warning_printed = False
        self.pm_client = None
        self.pm_allowance_info = ""
        self.pm_dirty_record_ids = set()
        self.pm_persisted_record_signatures = {}
        self.pm_storage_requires_rewrite = False
        self.bankroll_source = self._bankroll_source_label("env_start_bankroll")
        self._load_existing_records()
        if not self.pm_cfg.paper_mode:
            self.pm_client = self._build_live_client()
            self._refresh_live_cash_state(sync_bankroll=True)

    def _capped_trading_bankroll_usdc(self, bankroll_usdc):
        bankroll_usdc = float(bankroll_usdc)
        if np.isfinite(self.pm_cfg.max_bankroll_usdc):
            bankroll_usdc = min(bankroll_usdc, float(self.pm_cfg.max_bankroll_usdc))
        return float(bankroll_usdc)

    def _bankroll_source_label(self, base_source):
        if np.isfinite(self.pm_cfg.max_bankroll_usdc):
            return f"{base_source}_capped"
        return str(base_source)

    def _build_live_client(self):
        if not self.pm_cfg.private_key:
            raise ValueError("POLY_PRIVATE_KEY is required when POLY_PAPER_MODE=0.")
        if not self.pm_cfg.funder:
            raise ValueError("POLY_FUNDER_ADDRESS is required when POLY_PAPER_MODE=0.")
        if self.pm_cfg.signature_type not in {0, 1, 2}:
            raise ValueError(
                "POLY_SIGNATURE_TYPE must be one of {0, 1, 2} "
                "(0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE)."
            )
        signer_address = Account.from_key(self.pm_cfg.private_key).address
        signer_matches_funder = signer_address.lower() == self.pm_cfg.funder.lower()
        if self.pm_cfg.signature_type in {1, 2} and signer_matches_funder:
            print(
                "[warn] POLY_SIGNATURE_TYPE uses a proxy-wallet flow, but "
                "POLY_FUNDER_ADDRESS matches the signer address. "
                "For type 1/2, POLY_FUNDER_ADDRESS should usually be the proxy "
                "wallet address from polymarket.com/settings, not the private-key address."
            )

        client = ClobClient(
            self.pm_cfg.clob_host,
            key=self.pm_cfg.private_key,
            chain_id=self.pm_cfg.chain_id,
            signature_type=self.pm_cfg.signature_type,
            funder=self.pm_cfg.funder,
        )
        client.set_api_creds(client.create_or_derive_api_creds())
        return client

    def _fetch_live_balance_allowance(self):
        if (
            self.pm_client is None
            or BalanceAllowanceParams is None
            or AssetType is None
        ):
            raise RuntimeError("pm_client_not_initialized")
        return self.pm_client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )

    def _fetch_open_positions(self):
        payload = self._get_json(
            self.pm_cfg.data_api_host,
            "/positions",
            {"user": self.pm_cfg.funder, "sizeThreshold": 0},
        )
        return payload if isinstance(payload, list) else []

    def _fetch_closed_positions(self, condition_ids=None):
        condition_ids = [str(x) for x in (condition_ids or []) if str(x)]
        if not condition_ids:
            return []

        rows = []
        chunk_size = 20  # bezpiecznie, żeby query string nie urósł za bardzo
        for i in range(0, len(condition_ids), chunk_size):
            chunk = condition_ids[i : i + chunk_size]
            offset = 0

            while True:
                payload = self._get_json(
                    self.pm_cfg.data_api_host,
                    "/closed-positions",
                    {
                        "user": self.pm_cfg.funder,
                        "market": ",".join(chunk),
                        "sortBy": "TIMESTAMP",
                        "sortDirection": "DESC",
                        "limit": POLYMARKET_CLOSED_POSITIONS_PAGE_LIMIT,
                        "offset": offset,
                    },
                )
                page = payload if isinstance(payload, list) else []
                if not page:
                    break

                rows.extend(page)
                if len(page) < POLYMARKET_CLOSED_POSITIONS_PAGE_LIMIT:
                    break
                offset += POLYMARKET_CLOSED_POSITIONS_PAGE_LIMIT

        return rows

    def _refresh_live_cash_state(self, sync_bankroll):
        try:
            payload = self._fetch_live_balance_allowance()
        except Exception as exc:
            self.pm_allowance_info = f"allowance_lookup_failed:{exc}"
            return float("nan")

        self.pm_allowance_info = _json_compact(payload)
        balance_usdc = _collateral_balance_to_usdc(payload.get("balance"))
        self.pm_cash_balance_usdc = (
            float(balance_usdc) if np.isfinite(balance_usdc) else float("nan")
        )
        if sync_bankroll and np.isfinite(balance_usdc):
            self.live_bankroll_usdc = self._capped_trading_bankroll_usdc(balance_usdc)
            if np.isfinite(self.pm_cfg.max_bankroll_usdc) and float(
                balance_usdc
            ) > float(self.live_bankroll_usdc):
                self.bankroll_source = "polymarket_cash_balance_capped"
            else:
                self.bankroll_source = "polymarket_cash_balance"
        return balance_usdc

    def _load_existing_records(self):
        if not self.predictions_path.exists():
            return
        try:
            df = pd.read_csv(self.predictions_path, low_memory=False)
        except Exception as exc:
            print(f"[pm] failed to load existing records: {exc}")
            return
        
        if not self.pm_cfg.resume_existing_records:
            return

        if "pm_model_hash" in df.columns:
            df["pm_model_hash"] = df["pm_model_hash"].map(_safe_text)
            df = df.loc[df["pm_model_hash"] == self.model_hash].copy()
        else:
            self.pm_storage_requires_rewrite = True

        existing_columns = list(df.columns)
        expected_columns = list(LIVE_TRADE_EXPORT_COLUMNS)
        if existing_columns != expected_columns:
            self.pm_storage_requires_rewrite = True

        if "record_id" in df.columns:
            df["record_id"] = df["record_id"].map(_safe_text)
            initial_row_count = len(df)
            if df["record_id"].eq("").any():
                self.pm_storage_requires_rewrite = True
            else:
                df = df.drop_duplicates(subset=["record_id"], keep="last")
                if len(df) != initial_row_count:
                    self.pm_storage_requires_rewrite = True
        else:
            self.pm_storage_requires_rewrite = True

        timestamp_cols = {
            "record_snapshot_at",
            "prediction_time",
            "bucket_start",
            "bucket_end",
            "resolved_at",
            *LIVE_TRADE_LATENCY_TIMESTAMP_COLUMNS,
        }
        for col in timestamp_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        for col in (
            "pm_redeem_tx_id",
            "pm_redeem_tx_hash",
            "pm_redeem_tx_state",
            "pm_redeem_error",
            "pm_settlement_status",
            "pm_account_sync_at",
            "pm_account_sync_reason",
        ):
            if col in df.columns:
                df[col] = df[col].map(_safe_text)

        df = df.where(pd.notna(df), None)
        loaded_records = df.to_dict("records")
        for rec in loaded_records:
            rec["record_id"] = _stable_record_id(rec)
        with self.records_lock:
            self.records = loaded_records
        self.predicted_buckets = {
            ts
            for ts in df.get("bucket_start", pd.Series(dtype="object"))
            if isinstance(ts, pd.Timestamp)
        }
        self.pm_persisted_record_signatures = {
            rec["record_id"]: _record_state_signature(rec)
            for rec in loaded_records
            if rec.get("record_id")
        }

    def _relayer_is_configured(self):
        return bool(
            RELAYER_SDK_AVAILABLE
            and self.pm_cfg.relayer_api_key
            and self.pm_relayer_api_key_address
            and self.pm_relayer_signer is not None
            and self.pm_relayer_contract_config is not None
        )

    def _warn_relayer_unavailable_once(self, reason):
        if self.pm_relayer_warning_printed or self.pm_cfg.paper_mode:
            return
        print(f"[pm] auto-redeem disabled: {reason}")
        self.pm_relayer_warning_printed = True

    def _relayer_headers(self):
        if not self.pm_cfg.relayer_api_key:
            raise RuntimeError("POLY_RELAYER_API_KEY is missing")
        if not self.pm_relayer_api_key_address:
            raise RuntimeError("POLY_RELAYER_API_KEY_ADDRESS is missing")
        return {
            "RELAYER_API_KEY": self.pm_cfg.relayer_api_key,
            "RELAYER_API_KEY_ADDRESS": self.pm_relayer_api_key_address,
        }

    def _relayer_get_json(self, path, params=None, require_auth=True):
        url = f"{self.pm_cfg.relayer_host.rstrip('/')}/{path.lstrip('/')}"
        headers = self._relayer_headers() if require_auth else None
        response = self.pm_session.get(
            url,
            params=params,
            headers=headers,
            timeout=float(self.pm_cfg.market_request_timeout_sec),
        )
        response.raise_for_status()
        return response.json()

    def _relayer_post_json(self, path, payload):
        url = f"{self.pm_cfg.relayer_host.rstrip('/')}/{path.lstrip('/')}"
        response = self.pm_session.post(
            url,
            json=payload,
            headers=self._relayer_headers(),
            timeout=float(self.pm_cfg.market_request_timeout_sec),
        )
        response.raise_for_status()
        return response.json()

    def _relayer_get_nonce(self):
        payload = self._relayer_get_json(
            "/nonce",
            {"address": self.pm_signer_address, "type": "SAFE"},
            require_auth=False,
        )
        nonce = payload.get("nonce")
        if nonce is None or str(nonce) == "":
            raise RuntimeError("relayer_nonce_missing")
        return str(nonce)

    def _relayer_safe_is_deployed(self):
        payload = self._relayer_get_json(
            "/deployed",
            {"address": self.pm_cfg.funder},
            require_auth=False,
        )
        return bool(payload.get("deployed", False))

    def _relayer_get_transaction_state(self, tx_id):
        payload = self._relayer_get_json(
            "/transaction", {"id": tx_id}, require_auth=True
        )
        if isinstance(payload, list):
            return payload[0] if payload else {}
        return payload if isinstance(payload, dict) else {}

    def _encode_redeem_positions_call(self, condition_id):
        condition_hex = str(condition_id).strip()
        if not condition_hex.startswith("0x") or len(condition_hex) != 66:
            raise ValueError(
                f"Invalid conditionId for redeemPositions: {condition_id!r}"
            )
        selector = keccak(text="redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
        encoded_args = abi_encode(
            ["address", "bytes32", "bytes32", "uint256[]"],
            [
                POLYMARKET_USDC_E_ADDRESS,
                bytes.fromhex(POLYMARKET_ZERO_BYTES32[2:]),
                bytes.fromhex(condition_hex[2:]),
                list(POLYMARKET_BINARY_INDEX_SETS),
            ],
        )
        return "0x" + (selector + encoded_args).hex()

    def _build_redeem_transactions(self, candidates):
        transactions = []
        condition_ids = []
        seen_conditions = set()
        for item in candidates:
            condition_id = str(item.get("conditionId", ""))
            asset_id = str(item.get("asset", ""))
            if not condition_id or condition_id in seen_conditions:
                continue
            if bool(item.get("negativeRisk", False)):
                continue
            tx = RelayerSafeTransaction(
                to=POLYMARKET_CTF_ADDRESS,
                operation=RelayerOperationType.Call,
                data=self._encode_redeem_positions_call(condition_id),
                value="0",
            )
            transactions.append(tx)
            seen_conditions.add(condition_id)
            condition_ids.append(condition_id)
        return transactions, condition_ids

    def _submit_redeem_batch(self, candidates):
        if not candidates:
            return
        if self.pm_cfg.disable_order_submission:
            self._warn_relayer_unavailable_once(
                "live writes disabled via POLY_DISABLE_ORDER_SUBMIT=1"
            )
            return
        if not self._relayer_is_configured():
            self._warn_relayer_unavailable_once(
                "missing relayer SDK or POLY_RELAYER_API_KEY credentials"
            )
            return
        if not self._relayer_safe_is_deployed():
            raise RuntimeError(f"Relayer safe {self.pm_cfg.funder} is not deployed")

        transactions, condition_ids = self._build_redeem_transactions(candidates)
        if not transactions:
            return

        nonce = self._relayer_get_nonce()
        tx_args = RelayerSafeTransactionArgs(
            from_address=self.pm_signer_address,
            nonce=nonce,
            chain_id=self.pm_cfg.chain_id,
            transactions=transactions,
        )
        request_body = build_safe_transaction_request(
            signer=self.pm_relayer_signer,
            args=tx_args,
            config=self.pm_relayer_contract_config,
            metadata=f"redeem {self.pm_cfg.market_slug_prefix}",
        ).to_dict()
        response = self._relayer_post_json("/submit", request_body)
        tx_id = str(response.get("transactionID", "") or "")
        tx_hash = str(response.get("transactionHash", "") or "")
        if not tx_id:
            raise RuntimeError(f"relayer_submit_missing_transaction_id:{response}")
        self._mark_redeem_submission(
            condition_ids=condition_ids,
            tx_id=tx_id,
            tx_hash=tx_hash,
            tx_state="STATE_NEW",
            error="",
        )

    def _mark_redeem_submission(
        self,
        *,
        condition_ids,
        tx_id,
        tx_hash,
        tx_state,
        error,
    ):
        target_conditions = set(str(x) for x in condition_ids if str(x))
        with self.records_lock:
            for rec in self.records:
                condition_id = _safe_text(rec.get("pm_condition_id"))
                if condition_id not in target_conditions:
                    continue
                rec["pm_redeem_tx_id"] = tx_id
                rec["pm_redeem_tx_hash"] = tx_hash
                rec["pm_redeem_tx_state"] = tx_state
                rec["pm_redeem_error"] = error
                rec["pm_settlement_status"] = "redeem_submitted"
                _mark_record_dirty(self.pm_dirty_record_ids, rec)

    def _update_redeem_transaction_state(self, *, tx_id, tx_hash, tx_state, error=""):
        with self.records_lock:
            for rec in self.records:
                if _safe_text(rec.get("pm_redeem_tx_id")) != _safe_text(tx_id):
                    continue
                rec["pm_redeem_tx_hash"] = tx_hash or rec.get("pm_redeem_tx_hash", "")
                rec["pm_redeem_tx_state"] = tx_state
                rec["pm_redeem_error"] = error
                if tx_state == "STATE_CONFIRMED":
                    rec["pm_settlement_status"] = "redeem_confirmed_waiting_close_sync"
                elif tx_state in {"STATE_FAILED", "STATE_INVALID"}:
                    rec["pm_settlement_status"] = "redeem_failed"
                _mark_record_dirty(self.pm_dirty_record_ids, rec)

    def _poll_redeem_transactions(self):
        pending_ids = set()
        for rec in self._records_snapshot():
            tx_id = _safe_text(rec.get("pm_redeem_tx_id"))
            tx_state = _safe_text(rec.get("pm_redeem_tx_state"))
            if tx_id and tx_state not in POLYMARKET_RELAYER_TERMINAL_STATES:
                pending_ids.add(tx_id)
        for tx_id in pending_ids:
            tx = self._relayer_get_transaction_state(tx_id)
            if not tx:
                continue
            tx_state = str(tx.get("state", "") or "")
            tx_hash = str(tx.get("transactionHash", "") or "")
            error = str(tx.get("error", "") or tx.get("errorMessage", "") or "")
            if tx_state:
                self._update_redeem_transaction_state(
                    tx_id=tx_id,
                    tx_hash=tx_hash,
                    tx_state=tx_state,
                    error=error,
                )

    def _is_managed_position(self, position):
        slug = str(position.get("slug", "") or position.get("eventSlug", "") or "")
        return slug.startswith(self.pm_cfg.market_slug_prefix)
    
    def _tracked_condition_ids(self):
        condition_ids = set()
        for rec in self._records_snapshot():
            if str(rec.get("pm_mode", "")) != "live":
                continue
            condition_id = _safe_text(rec.get("pm_condition_id"))
            if condition_id:
                condition_ids.add(condition_id)
        return sorted(condition_ids)
    
    def _build_external_position_record(self, position, sync_at):
        asset = str(position.get("asset", ""))
        initial_value = _safe_float(position.get("initialValue"))
        size = _safe_float(position.get("size"))
        current_value = _safe_float(position.get("currentValue"))
        redeemable = bool(position.get("redeemable", False))

        return {
            "record_id": f"external:{asset}" if asset else "",
            "pm_model_hash": self.model_hash,
            "pm_run_started_at_utc": self.run_started_at_utc,
            "prediction_time": pd.Timestamp.now(tz="UTC"),
            "bucket_start": None,
            "bucket_end": None,
            "proba_up": np.nan,
            "threshold": self.prediction_threshold,
            "signal_up": None,
            "kelly_side": str(position.get("outcome", "")).lower(),
            "kelly_fraction": np.nan,
            "kelly_bet_usdc": initial_value,
            "kelly_edge": np.nan,
            "kelly_prob_win_adj": np.nan,
            "kelly_prob_win_raw": np.nan,
            "kelly_reason": "external_position",
            "stake_usdc": initial_value,
            "entry_price": _safe_float(position.get("avgPrice")),
            "entry_fee_usdc": np.nan,
            "entry_fee_raw_usdc": np.nan,
            "shares_net": size,
            "kelly_c_eff": np.nan,
            "kelly_eff_rate": np.nan,
            "price_eps": np.nan,
            "price_slip": np.nan,
            "bankroll_before_entry": np.nan,
            "bankroll_after_entry": np.nan,
            "bankroll_after_resolve": None,
            "trade_is_win": None,
            "payout_usdc": None,
            "pnl_usdc": None,
            "bucket_open_price": None,
            "bucket_close_price": None,
            "actual_up": None,
            "is_correct": None,
            "resolved_at": None,
            "pm_mode": "live",
            "pm_series_slug": self.pm_cfg.series_slug,
            "pm_execution_mode": "external_position",
            "pm_market_slug": str(position.get("slug", "") or ""),
            "pm_market_question": str(position.get("title", "") or ""),
            "pm_market_end": str(position.get("endDate", "") or ""),
            "pm_condition_id": str(position.get("conditionId", "") or ""),
            "pm_up_token_id": str(position.get("oppositeAsset", "") or ""),
            "pm_down_token_id": asset,
            "pm_selected_token_id": asset,
            "pm_accepting_orders": False,
            "pm_restricted": False,
            "pm_fees_enabled": False,
            "pm_fee_rate_bps": 0,
            "pm_tick_size": np.nan,
            "pm_order_min_size": np.nan,
            "pm_order_price_cap": float(POLYMARKET_ORDER_PRICE_CAP),
            "pm_position_size": size,
            "pm_position_current_value": current_value,
            "pm_position_redeemable": redeemable,
            "pm_settlement_status": "redeemable_open" if redeemable else "open",
            "pm_account_sync_at": sync_at,
            "pm_account_sync_reason": "startup_external_position",
            "pm_redeem_tx_id": "",
            "pm_redeem_tx_hash": "",
            "pm_redeem_tx_state": "",
            "pm_redeem_error": "",
            "pm_up_best_bid": np.nan,
            "pm_up_best_ask": np.nan,
            "pm_down_best_bid": np.nan,
            "pm_down_best_ask": np.nan,
            "pm_seconds_to_close": np.nan,
            "pm_order_status": "external_position",
            "pm_order_response": "",
            "pm_allowance_info": self.pm_allowance_info,
            "pm_market_error": "",
            "pm_order_error": "",
        }

    def _ensure_records_for_open_positions(self, open_positions, sync_at):
        tracked_assets = {
            str(rec.get("pm_selected_token_id", ""))
            for rec in self._records_snapshot()
            if str(rec.get("pm_selected_token_id", ""))
        }
        external_records = []
        for pos in open_positions:
            if not self._is_managed_position(pos):
                continue
            asset = str(pos.get("asset", ""))
            if not asset or asset in tracked_assets:
                continue
            external_records.append(self._build_external_position_record(pos, sync_at))
            tracked_assets.add(asset)

        if not external_records:
            return
        with self.records_lock:
            self.records.extend(external_records)
            for rec in external_records:
                _mark_record_dirty(self.pm_dirty_record_ids, rec)
    
    def _estimate_sell_proceeds(self, shares, price, fee_rate_bps):
        shares = float(shares)
        price = float(price)
        fee_rate = _polymarket_fee_rate_from_bps(int(fee_rate_bps))
        eff_rate = fee_rate * float((price * (1.0 - price)) ** POLYMARKET_CRYPTO_FEE_EXPONENT)

        gross = shares * price
        fee_raw = gross * eff_rate
        fee = round(fee_raw, POLYMARKET_FEE_ROUND_DECIMALS)
        if fee < POLYMARKET_MIN_FEE_USDC:
            fee = 0.0

        return {
            "gross_usdc": float(gross),
            "fee_usdc": float(fee),
            "net_usdc": float(gross - fee),
            "eff_rate": float(eff_rate),
        }
    
    def _collect_exit_candidates(self, open_positions):
        records_by_asset = {
            _safe_text(rec.get("pm_selected_token_id")): rec
            for rec in self._records_snapshot()
            if str(rec.get("pm_mode", "")) == "live" and _safe_text(rec.get("pm_selected_token_id"))
        }

        candidates = []
        for pos in open_positions:
            asset = _safe_text(pos.get("asset"))
            if not asset:
                continue

            rec = records_by_asset.get(asset)
            if rec is None:
                continue

            # tylko pozycje nadal otwarte i jeszcze nierozliczone
            if self._has_binary_flag(rec.get("actual_up")):
                continue
            if _safe_text(rec.get("pm_settlement_status")) in {
                "exit_submitted",
                "redeem_submitted",
                "redeem_confirmed_waiting_close_sync",
                "closed",
            }:
                continue

            shares = _safe_float(pos.get("size"))
            if not np.isfinite(shares) or shares <= 0.0:
                continue

            best_bid_book = self._fetch_order_book_summary(asset)
            best_bid = _safe_float(best_bid_book.get("best_bid"))
            if not np.isfinite(best_bid) or best_bid <= 0.0:
                continue

            stake_usdc = _safe_float(rec.get("stake_usdc"))
            if not np.isfinite(stake_usdc) or stake_usdc <= 0.0:
                continue

            fee_rate_bps = int(rec.get("pm_fee_rate_bps", 0) or 0)
            proceeds = self._estimate_sell_proceeds(
                shares=shares,
                price=best_bid,
                fee_rate_bps=fee_rate_bps,
            )
            pnl_usdc = float(proceeds["net_usdc"] - stake_usdc)
            roi = float(pnl_usdc / stake_usdc)

            market_end = pd.Timestamp(rec.get("pm_market_end"))
            seconds_to_close = float((market_end - _utc_now()).total_seconds())
            if seconds_to_close <= float(self.pm_cfg.exit_min_seconds_to_close):
                continue

            if pnl_usdc < float(self.pm_cfg.exit_min_profit_usdc):
                continue
            if roi < float(self.pm_cfg.exit_min_roi):
                continue

            candidates.append(
                {
                    "asset": asset,
                    "shares": float(shares),
                    "price": float(best_bid),
                    "fee_rate_bps": fee_rate_bps,
                    "tick_size": rec.get("pm_tick_size"),
                    "neg_risk": False,
                    "stake_usdc": float(stake_usdc),
                    "pnl_usdc": pnl_usdc,
                    "proceeds_net_usdc": float(proceeds["net_usdc"]),
                    "fee_usdc": float(proceeds["fee_usdc"]),
                }
            )

        return candidates

    def _submit_exit_candidates(self, open_positions):
        candidates = self._collect_exit_candidates(open_positions)
        if not candidates:
            return

        for candidate in candidates:
            self._submit_single_exit_candidate(candidate)

    def _submit_single_exit_candidate(self, candidate):
        asset = str(candidate["asset"])
        shares = float(candidate["shares"])
        price = float(candidate["price"])

        status = "skipped"
        response_txt = ""
        error_txt = ""

        try:
            if self.pm_cfg.paper_mode:
                status = "paper_exit_intent"
            elif self.pm_cfg.disable_order_submission:
                status = "exit_submission_disabled"
            elif self.pm_client is None:
                status = "exit_client_unavailable"
                error_txt = "pm_client_not_initialized"
            else:
                options = _partial_create_order_options(
                    candidate.get("tick_size"),
                    candidate.get("neg_risk"),
                )
                order = MarketOrderArgs(
                    token_id=asset,
                    amount=shares,           # SELL => shares, nie USDC
                    side=SELL,
                    price=price,            # floor = obecny best bid
                    fee_rate_bps=int(candidate.get("fee_rate_bps", 0) or 0),
                    order_type=OrderType.FOK,
                )
                signed_order = self.pm_client.create_market_order(order, options=options)
                response = self.pm_client.post_order(signed_order, OrderType.FOK)
                response_txt = _json_compact(response)
                if isinstance(response, dict) and bool(response.get("success", False)):
                    status = "submitted_fok"
                else:
                    status = "submission_rejected"
        except Exception as exc:
            status = "submission_error"
            error_txt = str(exc)

        with self.records_lock:
            for rec in self.records:
                if _safe_text(rec.get("pm_selected_token_id")) != asset:
                    continue
                rec["pm_exit_order_status"] = status
                rec["pm_exit_order_response"] = response_txt
                rec["pm_exit_order_error"] = error_txt
                rec["pm_exit_reason"] = "profit_take"
                rec["pm_exit_price"] = price
                rec["pm_exit_shares"] = shares
                rec["pm_exit_fee_usdc"] = float(candidate["fee_usdc"])
                rec["pm_exit_proceeds_usdc"] = float(candidate["proceeds_net_usdc"])
                if status == "submitted_fok":
                    rec["pm_settlement_status"] = "exit_submitted"
                _mark_record_dirty(self.pm_dirty_record_ids, rec)

    def _collect_redeem_candidates(self, open_positions):
        records_by_condition = {
            _safe_text(rec.get("pm_condition_id")): rec
            for rec in self._records_snapshot()
            if str(rec.get("pm_mode", "")) == "live" and _safe_text(rec.get("pm_condition_id"))
        }

        candidates = []
        for pos in open_positions:
            if not self._is_managed_position(pos):
                continue
            if bool(pos.get("negativeRisk", False)):
                continue

            condition_id = _safe_text(pos.get("conditionId"))
            if not condition_id:
                continue

            rec = records_by_condition.get(condition_id)
            if rec is None:
                continue

            # redeem dopiero po resolution
            if rec.get("resolved_at") is None and not self._has_binary_flag(rec.get("actual_up")):
                continue

            tx_state = _safe_text(rec.get("pm_redeem_tx_state"))
            settlement_status = _safe_text(rec.get("pm_settlement_status"))

            # skip tylko gdy tx naprawdę jeszcze pending albo rekord już finalnie zamknięty
            if tx_state in POLYMARKET_RELAYER_PENDING_STATES:
                continue
            if settlement_status in {"redeem_submitted", "redeem_confirmed_waiting_close_sync", "closed"}:
                continue

            # NIE wymagaj pos.get("redeemable")==True
            # chcemy też spalić losing balances jako cleanup
            candidates.append(pos)

        return candidates

    def _poll_background_sync(self):
        future = None
        pending_reason = ""
        with self.pm_bg_lock:
            if self.pm_bg_future is None or not self.pm_bg_future.done():
                return
            future = self.pm_bg_future
            self.pm_bg_future = None
            pending_reason = self.pm_bg_pending_reason
            self.pm_bg_pending_reason = ""

        try:
            future.result()
        except Exception as exc:
            print(f"[pm] background sync failed: {exc}")

        if pending_reason:
            self._schedule_background_sync(pending_reason, force=True)

    def _schedule_background_sync(self, reason, force=False):
        if self.pm_cfg.paper_mode or self.pm_client is None:
            return

        self._poll_background_sync()
        now = time.monotonic()
        with self.pm_bg_lock:
            if self.pm_bg_future is not None and not self.pm_bg_future.done():
                self.pm_bg_pending_reason = reason
                return
            if (
                not force
                and now - self.pm_bg_last_started
                < POLYMARKET_BACKGROUND_SYNC_MIN_INTERVAL_SEC
            ):
                self.pm_bg_pending_reason = reason
                return
            self.pm_bg_last_started = now
            self.pm_bg_future = self.pm_bg_pool.submit(
                self._run_background_sync, reason
            )

    def _run_background_sync(self, reason):
        if self._relayer_is_configured():
            self._poll_redeem_transactions()
        elif not RELAYER_SDK_AVAILABLE:
            self._warn_relayer_unavailable_once(
                "py-builder-relayer-client is not installed"
            )
        elif not self.pm_cfg.relayer_api_key:
            self._warn_relayer_unavailable_once("POLY_RELAYER_API_KEY is missing")

        cash_balance_usdc = self._refresh_live_cash_state(sync_bankroll=True)
        open_positions = self._fetch_open_positions()
        tracked_condition_ids = self._tracked_condition_ids()
        closed_positions = self._fetch_closed_positions(condition_ids=tracked_condition_ids)
        sync_at = pd.Timestamp.now(tz="UTC").isoformat()

        if self.pm_cfg.import_untracked_open_positions:
            self._ensure_records_for_open_positions(open_positions, sync_at)
        self.pm_positions_value_usdc = float(
            sum(
                _safe_float(item.get("currentValue"), 0.0) or 0.0
                for item in open_positions
                if np.isfinite(_safe_float(item.get("currentValue"), float("nan")))
            )
        )
        self.pm_last_account_sync_at = sync_at
        self.pm_last_account_sync_reason = reason
        self._reconcile_live_records(
            cash_balance_usdc=cash_balance_usdc,
            open_positions=open_positions,
            closed_positions=closed_positions,
            sync_at=sync_at,
            reason=reason,
        )
        if self.pm_cfg.enable_exit_orders:
            self._submit_exit_candidates(open_positions)

        if self.pm_cfg.redeem_resolved_positions:
            self._submit_redeem_batch(self._collect_redeem_candidates(open_positions))
        self._save_records()

    def _reconcile_live_records(
        self,
        *,
        cash_balance_usdc,
        open_positions,
        closed_positions,
        sync_at,
        reason,
    ):
        open_by_asset = {
            str(item.get("asset", "")): item
            for item in open_positions
            if str(item.get("asset", ""))
        }
        closed_by_asset = {}
        for item in closed_positions:
            asset = str(item.get("asset", ""))
            if not asset:
                continue
            prev = closed_by_asset.get(asset)
            prev_ts = int(prev.get("timestamp", 0) or 0) if prev is not None else -1
            item_ts = int(item.get("timestamp", 0) or 0)
            if item_ts >= prev_ts:
                closed_by_asset[asset] = item

        with self.records_lock:
            for rec in self.records:
                if str(rec.get("pm_mode", "")) != "live":
                    continue
                asset = str(rec.get("pm_selected_token_id", ""))
                if not asset:
                    continue
                rec["pm_account_sync_at"] = sync_at
                rec["pm_account_sync_reason"] = reason
                closed_pos = closed_by_asset.get(asset)
                if closed_pos is not None:
                    avg_price = _safe_float(closed_pos.get("avgPrice"))
                    total_bought = _safe_float(closed_pos.get("totalBought"))
                    realized_pnl = _safe_float(closed_pos.get("realizedPnl"))
                    shares_net = (
                        float(total_bought / avg_price)
                        if np.isfinite(total_bought)
                        and np.isfinite(avg_price)
                        and avg_price > 0
                        else float(rec.get("shares_net", 0.0) or 0.0)
                    )
                    payout = (
                        float(total_bought + realized_pnl)
                        if np.isfinite(total_bought) and np.isfinite(realized_pnl)
                        else np.nan
                    )
                    if np.isfinite(total_bought):
                        rec["stake_usdc"] = float(total_bought)
                    if np.isfinite(avg_price):
                        rec["entry_price"] = float(avg_price)
                    if np.isfinite(shares_net):
                        rec["shares_net"] = float(shares_net)
                    rec["trade_is_win"] = (
                        int(realized_pnl > 0.0)
                        if np.isfinite(realized_pnl)
                        else rec["trade_is_win"]
                    )
                    rec["payout_usdc"] = (
                        float(payout) if np.isfinite(payout) else rec.get("payout_usdc")
                    )
                    rec["pnl_usdc"] = (
                        float(realized_pnl)
                        if np.isfinite(realized_pnl)
                        else rec.get("pnl_usdc")
                    )
                    effective_bankroll_usdc = self._capped_trading_bankroll_usdc(
                        cash_balance_usdc
                    )
                    rec["bankroll_after_resolve"] = (
                        float(effective_bankroll_usdc)
                        if np.isfinite(cash_balance_usdc)
                        else rec.get("bankroll_after_resolve")
                    )
                    rec["pm_position_size"] = 0.0
                    rec["pm_position_current_value"] = 0.0
                    rec["pm_position_redeemable"] = False
                    rec["pm_settlement_status"] = "closed"
                    if _safe_text(rec.get("pm_exit_order_status")) == "submitted_fok":
                        rec["pm_settlement_status"] = "closed"
                    elif _safe_text(rec.get("pm_redeem_tx_state")) == "STATE_CONFIRMED":
                        rec["pm_settlement_status"] = "closed"
                    else:
                        rec["pm_settlement_status"] = "closed"
                    continue

                open_pos = open_by_asset.get(asset)
                if open_pos is None:
                    settlement_status = _safe_text(rec.get("pm_settlement_status"))

                    if settlement_status == "exit_submitted":
                        rec["pm_settlement_status"] = "awaiting_exit_close_sync"
                    elif settlement_status == "redeem_submitted":
                        rec["pm_settlement_status"] = "awaiting_redeem_close_sync"
                    elif rec.get("pm_order_status") == "submitted_fok":
                        rec["pm_settlement_status"] = (
                            "awaiting_close_sync"
                            if rec.get("resolved_at") is not None
                            else "awaiting_entry_sync"
                        )
                    continue

                pos_size = _safe_float(open_pos.get("size"))
                avg_price = _safe_float(open_pos.get("avgPrice"))
                initial_value = _safe_float(open_pos.get("initialValue"))
                current_value = _safe_float(open_pos.get("currentValue"))
                redeemable = bool(open_pos.get("redeemable", False))

                if np.isfinite(initial_value):
                    rec["stake_usdc"] = float(initial_value)
                if np.isfinite(avg_price):
                    rec["entry_price"] = float(avg_price)
                if np.isfinite(pos_size):
                    rec["shares_net"] = float(pos_size)
                if np.isfinite(cash_balance_usdc) and rec.get(
                    "pm_settlement_status"
                ) not in {"open", "redeemable_open", "closed"}:
                    rec["bankroll_after_entry"] = self._capped_trading_bankroll_usdc(
                        cash_balance_usdc
                    )
                rec["pm_position_size"] = (
                    float(pos_size) if np.isfinite(pos_size) else np.nan
                )
                rec["pm_position_current_value"] = (
                    float(current_value) if np.isfinite(current_value) else np.nan
                )
                rec["pm_position_redeemable"] = bool(redeemable)
                current_status = _safe_text(rec.get("pm_settlement_status"))
                tx_state = _safe_text(rec.get("pm_redeem_tx_state"))

                if current_status == "exit_submitted":
                    rec["pm_settlement_status"] = "exit_submitted"
                elif tx_state in POLYMARKET_RELAYER_PENDING_STATES:
                    rec["pm_settlement_status"] = "redeem_submitted"
                elif tx_state == "STATE_CONFIRMED":
                    rec["pm_settlement_status"] = "redeem_confirmed_waiting_close_sync"
                elif tx_state in {"STATE_FAILED", "STATE_INVALID"}:
                    rec["pm_settlement_status"] = "redeem_failed"
                elif rec.get("resolved_at") is not None:
                    rec["pm_settlement_status"] = "resolved_waiting_settlement"
                else:
                    rec["pm_settlement_status"] = (
                        "redeemable_open" if redeemable else "open"
                    )

    def _records_snapshot(self):
        with self.records_lock:
            return [dict(rec) for rec in self.records]

    def _bucket_start_for_latest_candle(self):
        minute_open = self.opened_candles[-1]
        return minute_open.floor(f"{self.target_bucket_minutes}min") + pd.Timedelta(
            minutes=self.target_bucket_minutes
        )

    def _next_unpredicted_bucket_start(self):
        bucket_start = self._bucket_start_for_latest_candle()
        while bucket_start in self.predicted_buckets:
            bucket_start += pd.Timedelta(minutes=self.target_bucket_minutes)
        return bucket_start

    def _cancel_market_prefetch_timer_locked(self):
        timer = self.pm_market_prefetch_timer
        self.pm_market_prefetch_timer = None
        if timer is not None:
            timer.cancel()

    def _submit_market_snapshot_prefetch(self, bucket_start):
        bucket_start = pd.Timestamp(bucket_start)
        with self.pm_market_prefetch_lock:
            if bucket_start in self.predicted_buckets:
                return None
            if self.pm_market_prefetch_bucket_start == bucket_start:
                future = self.pm_market_prefetch_future
                if future is not None:
                    return future
            self._cancel_market_prefetch_timer_locked()
            self.pm_market_prefetch_bucket_start = bucket_start
            self.pm_market_prefetch_future = self.pm_lookup_pool.submit(
                self._fetch_market_snapshot_prefetch_payload,
                bucket_start,
            )
            return self.pm_market_prefetch_future

    def _market_prefetch_timer_callback(self, bucket_start):
        bucket_start = pd.Timestamp(bucket_start)
        with self.pm_market_prefetch_lock:
            if self.pm_market_prefetch_bucket_start != bucket_start:
                return
            self.pm_market_prefetch_timer = None
        try:
            self._submit_market_snapshot_prefetch(bucket_start)
        except Exception:
            pass

    def _schedule_market_snapshot_prefetch(self, bucket_start):
        lead_ms = max(int(self.pm_cfg.market_lookup_prefetch_lead_ms), 0)
        if lead_ms <= 0:
            return

        bucket_start = pd.Timestamp(bucket_start)
        if bucket_start in self.predicted_buckets:
            return

        prefetch_at = bucket_start - pd.Timedelta(milliseconds=lead_ms)
        delay_sec = float((prefetch_at - pd.Timestamp.now(tz="UTC")).total_seconds())

        timer_to_start = None
        submit_now = False
        with self.pm_market_prefetch_lock:
            same_bucket = self.pm_market_prefetch_bucket_start == bucket_start
            if same_bucket:
                if self.pm_market_prefetch_future is not None:
                    return
                if self.pm_market_prefetch_timer is not None:
                    return
            self._cancel_market_prefetch_timer_locked()
            self.pm_market_prefetch_bucket_start = bucket_start
            self.pm_market_prefetch_future = None
            if delay_sec <= 0.0:
                submit_now = True
            else:
                timer = threading.Timer(
                    delay_sec,
                    self._market_prefetch_timer_callback,
                    args=(bucket_start,),
                )
                timer.daemon = True
                self.pm_market_prefetch_timer = timer
                timer_to_start = timer
        if timer_to_start is not None:
            timer_to_start.start()
            return
        if submit_now:
            self._submit_market_snapshot_prefetch(bucket_start)

    def _market_lookup_future_for_bucket(self, bucket_start, latency_trace=None):
        bucket_start = pd.Timestamp(bucket_start)
        if latency_trace is not None:
            latency_trace["_perf_market_lookup_submitted"] = time.perf_counter()

        with self.pm_market_prefetch_lock:
            if self.pm_market_prefetch_bucket_start == bucket_start:
                future = self.pm_market_prefetch_future
                if future is not None:
                    if latency_trace is not None:
                        latency_trace["pm_latency_market_lookup_queue_ms"] = 0.0
                    return future
                self._cancel_market_prefetch_timer_locked()

        return self.pm_lookup_pool.submit(
            self._fetch_market_snapshot_traced,
            bucket_start,
            latency_trace,
        )

    def _merge_prefetched_market_latency(self, latency_trace, latency_payload):
        if latency_trace is None or not isinstance(latency_payload, dict):
            return

        for key in (
            "pm_market_lookup_attempts",
            "pm_latency_market_lookup_total_ms",
            "pm_latency_market_lookup_retry_sleep_ms",
            "pm_latency_market_lookup_gamma_ms",
            "pm_latency_market_lookup_up_book_ms",
            "pm_latency_market_lookup_down_book_ms",
            "pm_latency_market_lookup_fee_rate_ms",
        ):
            value = latency_payload.get(key)
            if value is None:
                continue
            latency_trace[key] = value

    def _prefetched_market_payload_is_fresh(self, bucket_start, payload):
        if not isinstance(payload, dict):
            return False
        if pd.Timestamp(payload.get("bucket_start")) != pd.Timestamp(bucket_start):
            return False

        max_age_ms = max(int(self.pm_cfg.market_lookup_prefetch_max_age_ms), 0)
        fetched_at = payload.get("fetched_at")
        if max_age_ms <= 0 or fetched_at is None:
            return True

        age_ms = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(fetched_at)).total_seconds()
        return age_ms * 1000.0 <= float(max_age_ms)

    def _fetch_market_snapshot_prefetch_payload(self, bucket_start):
        latency_trace = {}
        snapshot = self._fetch_market_snapshot_traced(
            bucket_start, latency_trace=latency_trace
        )
        latency_payload = _latency_snapshot(bucket_start, latency_trace)
        latency_payload["pm_latency_market_lookup_queue_ms"] = 0.0
        return {
            "bucket_start": pd.Timestamp(bucket_start),
            "snapshot": snapshot,
            "latency_payload": latency_payload,
            "fetched_at": _utc_now(),
        }

    def _market_slug_for_bucket(self, bucket_start):
        if self.pm_cfg.market_slug_override:
            return self.pm_cfg.market_slug_override
        return f"{self.pm_cfg.market_slug_prefix}-{int(bucket_start.timestamp())}"

    def _get_json(self, host, path, params=None):
        url = f"{host.rstrip('/')}/{path.lstrip('/')}"
        response = self.pm_session.get(
            url,
            params=params,
            timeout=float(self.pm_cfg.market_request_timeout_sec),
        )
        response.raise_for_status()
        return response.json()

    def _fetch_order_book_summary(self, token_id):
        payload = self._get_json(self.pm_cfg.clob_host, "/book", {"token_id": token_id})
        return {
            "best_bid": _best_price(payload.get("bids", []), side="bid"),
            "best_ask": _best_price(payload.get("asks", []), side="ask"),
            "last_trade_price": _safe_float(payload.get("last_trade_price")),
            "min_order_size": _safe_float(payload.get("min_order_size")),
            "tick_size": _safe_float(payload.get("tick_size")),
            "neg_risk": bool(payload.get("neg_risk", False)),
        }

    def _fetch_fee_rate_bps(self, token_id):
        payload = self._get_json(
            self.pm_cfg.clob_host, "/fee-rate", {"token_id": token_id}
        )
        return int(payload.get("base_fee", 0))

    def _fetch_market_snapshot(self, bucket_start, latency_trace=None):
        market_slug = self._market_slug_for_bucket(bucket_start)
        gamma_started_perf = time.perf_counter()
        market = self._get_json(self.pm_cfg.gamma_host, f"/markets/slug/{market_slug}")
        _latency_accumulate_ms(
            latency_trace,
            "pm_latency_market_lookup_gamma_ms",
            _elapsed_ms(gamma_started_perf),
        )

        outcomes = [str(x).strip() for x in _parse_json_list(market.get("outcomes"))]
        token_ids = [
            str(x).strip() for x in _parse_json_list(market.get("clobTokenIds"))
        ]
        if len(outcomes) != len(token_ids):
            raise ValueError(
                f"Outcome/token length mismatch for market={market_slug}: "
                f"outcomes={len(outcomes)} token_ids={len(token_ids)}"
            )

        token_by_outcome = {
            outcome.lower(): token_id for outcome, token_id in zip(outcomes, token_ids)
        }
        up_token_id = token_by_outcome.get("up", "")
        down_token_id = token_by_outcome.get("down", "")
        if not up_token_id or not down_token_id:
            raise ValueError(
                f"Expected Up/Down outcomes for market={market_slug}, got={outcomes}"
            )

        up_book_future = self.pm_io_pool.submit(
            _timed_call, self._fetch_order_book_summary, up_token_id
        )
        down_book_future = self.pm_io_pool.submit(
            _timed_call, self._fetch_order_book_summary, down_token_id
        )
        fee_rate_future = self.pm_io_pool.submit(
            _timed_call, self._fetch_fee_rate_bps, up_token_id
        )
        up_book, up_book_ms = up_book_future.result()
        down_book, down_book_ms = down_book_future.result()
        fee_rate_bps, fee_rate_ms = fee_rate_future.result()
        _latency_accumulate_ms(
            latency_trace, "pm_latency_market_lookup_up_book_ms", up_book_ms
        )
        _latency_accumulate_ms(
            latency_trace, "pm_latency_market_lookup_down_book_ms", down_book_ms
        )
        _latency_accumulate_ms(
            latency_trace, "pm_latency_market_lookup_fee_rate_ms", fee_rate_ms
        )

        return PolymarketMarketSnapshot(
            market_slug=market_slug,
            market_question=str(market.get("question", "")),
            bucket_start=pd.Timestamp(bucket_start).isoformat(),
            market_end=str(market.get("endDate", "")),
            condition_id=str(market.get("conditionId", "")),
            restricted=bool(market.get("restricted", False)),
            accepting_orders=bool(market.get("acceptingOrders", False)),
            fees_enabled=bool(market.get("feesEnabled", False)),
            order_min_size=max(
                _safe_float(market.get("orderMinSize"), 0.0),
                _safe_float(up_book.get("min_order_size"), 0.0),
                _safe_float(down_book.get("min_order_size"), 0.0),
            ),
            tick_size=max(
                _safe_float(market.get("orderPriceMinTickSize"), 0.0),
                _safe_float(up_book.get("tick_size"), 0.0),
                _safe_float(down_book.get("tick_size"), 0.0),
            ),
            neg_risk=bool(market.get("negRisk", False))
            or bool(up_book.get("neg_risk"))
            or bool(down_book.get("neg_risk")),
            fee_rate_bps=fee_rate_bps,
            up_token_id=up_token_id,
            down_token_id=down_token_id,
            up_best_bid=_safe_float(up_book.get("best_bid")),
            up_best_ask=_safe_float(up_book.get("best_ask")),
            up_last_trade_price=_safe_float(up_book.get("last_trade_price")),
            down_best_bid=_safe_float(down_book.get("best_bid")),
            down_best_ask=_safe_float(down_book.get("best_ask")),
            down_last_trade_price=_safe_float(down_book.get("last_trade_price")),
        )

    def _fetch_market_snapshot_with_retry(self, bucket_start, latency_trace=None):
        deadline = time.perf_counter() + (
            max(int(self.pm_cfg.market_lookup_max_wait_ms), 0) / 1000.0
        )
        retry_sleep_sec = max(int(self.pm_cfg.market_lookup_retry_ms), 0) / 1000.0
        last_error = None

        while True:
            _latency_increment(latency_trace, "pm_market_lookup_attempts")
            try:
                return self._fetch_market_snapshot(
                    bucket_start, latency_trace=latency_trace
                )
            except requests.HTTPError as exc:
                last_error = exc
                status_code = _http_status_code(exc)
                if status_code != 404 or time.perf_counter() >= deadline:
                    raise
            except requests.RequestException as exc:
                last_error = exc
                if time.perf_counter() >= deadline:
                    raise

            if time.perf_counter() >= deadline:
                if last_error is not None:
                    raise last_error
                raise RuntimeError("market_lookup_retry_exhausted")
            retry_sleep_started_perf = time.perf_counter()
            time.sleep(retry_sleep_sec)
            _latency_accumulate_ms(
                latency_trace,
                "pm_latency_market_lookup_retry_sleep_ms",
                _elapsed_ms(retry_sleep_started_perf),
            )

    def _fetch_market_snapshot_traced(self, bucket_start, latency_trace=None):
        lookup_started_perf = time.perf_counter()
        if latency_trace is not None:
            latency_trace["_perf_market_lookup_started"] = lookup_started_perf
            submitted_perf = latency_trace.get("_perf_market_lookup_submitted")
            if submitted_perf is not None:
                _latency_set_from_perf(
                    latency_trace,
                    "pm_latency_market_lookup_queue_ms",
                    submitted_perf,
                    lookup_started_perf,
                )
        try:
            return self._fetch_market_snapshot_with_retry(
                bucket_start, latency_trace=latency_trace
            )
        finally:
            lookup_completed_perf = time.perf_counter()
            if latency_trace is not None:
                latency_trace["_perf_market_lookup_completed"] = lookup_completed_perf
                _latency_set_from_perf(
                    latency_trace,
                    "pm_latency_market_lookup_total_ms",
                    lookup_started_perf,
                    lookup_completed_perf,
                )

    def _score_outcome(
        self,
        *,
        side,
        token_id,
        prob_win_raw,
        prob_win_adj,
        price,
        fee_rate_bps,
        order_min_size,
    ):
        fee_rate = _polymarket_fee_rate_from_bps(int(fee_rate_bps))
        fee_exponent = POLYMARKET_CRYPTO_FEE_EXPONENT
        eff_rate = fee_rate * float((price * (1.0 - price)) ** fee_exponent)
        if eff_rate >= 0.99:
            return {"reason": "eff_rate_too_high", "side": side, "token_id": token_id}

        c_eff = price / (1.0 - eff_rate)
        edge = prob_win_adj - c_eff
        f_star = max(float((prob_win_adj - c_eff) / (1.0 - c_eff)), 0.0)
        fraction = min(
            float(self.live_kelly_sizing["cap"]),
            float(self.live_kelly_sizing["fractional_kelly"]) * f_star,
        )
        stake = float(self.live_bankroll_usdc) * fraction
        if np.isfinite(self.pm_cfg.max_exposure_usdc):
            stake = min(stake, float(self.pm_cfg.max_exposure_usdc))
        fee_raw = stake * eff_rate
        fee = round(fee_raw, POLYMARKET_FEE_ROUND_DECIMALS)
        if fee < POLYMARKET_MIN_FEE_USDC:
            fee = 0.0

        shares_net = (stake - fee) / price if price > 0.0 else 0.0
        return {
            "reason": "ok",
            "side": side,
            "token_id": token_id,
            "edge": float(edge),
            "fraction": float(fraction),
            "bet_usdc": float(stake),
            "prob_win_raw": float(prob_win_raw),
            "prob_win_adj": float(prob_win_adj),
            "entry_price": float(price),
            "fee_usdc": float(fee),
            "fee_raw_usdc": float(fee_raw),
            "shares_net": float(shares_net),
            "c_eff": float(c_eff),
            "eff_rate": float(eff_rate),
            "order_min_size": float(order_min_size),
            "fee_rate_bps": int(fee_rate_bps),
        }

    def _recommend_polymarket_bet(self, prob_up_raw, market):
        bankroll = float(self.live_bankroll_usdc)
        if bankroll <= 0.0:
            return {"reason": "bankroll_non_positive"}
        if not market.accepting_orders:
            return {"reason": "market_not_accepting_orders"}

        market_end = pd.Timestamp(market.market_end)
        seconds_to_close = float(
            (market_end - pd.Timestamp.now(tz="UTC")).total_seconds()
        )
        if seconds_to_close <= float(self.pm_cfg.no_trade_last_seconds):
            return {
                "reason": "too_close_to_market_end",
                "seconds_to_close": float(seconds_to_close),
            }

        p = float(
            adjust_probability_for_kelly(
                float(prob_up_raw),
                prob_shrink=float(self.live_kelly_sizing["prob_shrink"]),
                min_clip=1e-6,
            )
        )

        candidates = []
        if np.isfinite(market.up_best_ask) and 0.0 < market.up_best_ask < 1.0:
            candidates.append(
                self._score_outcome(
                    side="up",
                    token_id=market.up_token_id,
                    prob_win_raw=float(prob_up_raw),
                    prob_win_adj=float(p),
                    price=float(market.up_best_ask),
                    fee_rate_bps=int(market.fee_rate_bps),
                    order_min_size=float(market.order_min_size),
                )
            )
        if np.isfinite(market.down_best_ask) and 0.0 < market.down_best_ask < 1.0:
            candidates.append(
                self._score_outcome(
                    side="down",
                    token_id=market.down_token_id,
                    prob_win_raw=float(prob_up_raw),
                    prob_win_adj=float(1.0 - p),
                    price=float(market.down_best_ask),
                    fee_rate_bps=int(market.fee_rate_bps),
                    order_min_size=float(market.order_min_size),
                )
            )

        valid_candidates = [
            candidate for candidate in candidates if candidate.get("reason") == "ok"
        ]
        if not valid_candidates:
            return {
                "reason": "no_valid_orderbook_quotes",
                "up_best_ask": float(market.up_best_ask),
                "down_best_ask": float(market.down_best_ask),
            }

        best = max(valid_candidates, key=lambda item: float(item["edge"]))
        if float(best["edge"]) < float(self.live_kelly_sizing["min_edge"]):
            best["reason"] = "edge_below_min"
            return best
        if float(best["fraction"]) <= 0.0:
            best["reason"] = "fraction_non_positive"
            return best
        if float(best["bet_usdc"]) < float(self.live_kelly_sizing["min_stake_usdc"]):
            best["reason"] = "stake_below_min"
            return best
        if float(best["entry_price"]) > POLYMARKET_ORDER_PRICE_CAP:
            best["reason"] = "price_above_order_cap"
            best["order_price_cap"] = float(POLYMARKET_ORDER_PRICE_CAP)
            return best
        if float(best["fee_usdc"]) >= float(best["bet_usdc"]):
            best["reason"] = "fee_ge_stake"
            return best
        if float(best["shares_net"]) < float(best["order_min_size"]):
            best["reason"] = "shares_below_order_min"
            return best
        best["seconds_to_close"] = float(seconds_to_close)
        best["tick_size"] = float(market.tick_size)
        best["neg_risk"] = bool(market.neg_risk)
        best["order_price_cap"] = float(POLYMARKET_ORDER_PRICE_CAP)
        return best

    def _submit_result(
        self,
        *,
        commit_bankroll,
        status,
        response="",
        error="",
        allowance_info=None,
    ):
        return {
            "commit_bankroll": bool(commit_bankroll),
            "status": str(status),
            "response": str(response),
            "error": str(error),
            "allowance_info": (
                self.pm_allowance_info
                if allowance_info is None
                else str(allowance_info)
            ),
        }

    def _prime_pm_client_order_metadata(self, intent):
        if self.pm_client is None:
            return

        token_id = str(intent.get("token_id", "") or "").strip()
        if not token_id:
            return

        tick_size_text = _tick_size_literal(intent.get("tick_size"))
        fee_rate_bps = intent.get("fee_rate_bps")
        neg_risk = intent.get("neg_risk")

        tick_sizes = getattr(self.pm_client, "_ClobClient__tick_sizes", None)
        tick_size_timestamps = getattr(
            self.pm_client, "_ClobClient__tick_size_timestamps", None
        )
        fee_rates = getattr(self.pm_client, "_ClobClient__fee_rates", None)
        neg_risk_cache = getattr(self.pm_client, "_ClobClient__neg_risk", None)

        if tick_size_text is not None and isinstance(tick_sizes, dict):
            tick_sizes[token_id] = tick_size_text
            if isinstance(tick_size_timestamps, dict):
                tick_size_timestamps[token_id] = time.monotonic()
        if isinstance(fee_rates, dict):
            try:
                fee_rate_bps = int(fee_rate_bps)
            except (TypeError, ValueError):
                fee_rate_bps = None
            if fee_rate_bps is not None and fee_rate_bps >= 0:
                fee_rates[token_id] = fee_rate_bps
        if neg_risk is not None and isinstance(neg_risk_cache, dict):
            neg_risk_cache[token_id] = bool(neg_risk)

    def _maybe_submit_order(self, intent, latency_trace=None):
        submit_started_perf = time.perf_counter()
        if latency_trace is not None:
            latency_trace["pm_submit_ready_at"] = _utc_now()
            ws_received_perf = latency_trace.get("_perf_ws_received")
            if ws_received_perf is not None:
                _latency_set_from_perf(
                    latency_trace,
                    "pm_latency_ws_to_submit_ready_ms",
                    ws_received_perf,
                    submit_started_perf,
                )
        try:
            if intent.get("reason") != "ok":
                return self._submit_result(commit_bankroll=False, status="skipped")
            if self.pm_cfg.paper_mode:
                return self._submit_result(commit_bankroll=True, status="paper_intent")
            if self.pm_cfg.disable_order_submission:
                return self._submit_result(
                    commit_bankroll=False,
                    status="submission_disabled",
                )
            if self.pm_client is None:
                return self._submit_result(
                    commit_bankroll=False,
                    status="client_unavailable",
                    error="pm_client_not_initialized",
                )

            execution_mode = str(self.pm_cfg.execution_mode).lower()
            order_options = _partial_create_order_options(
                intent.get("tick_size"), intent.get("neg_risk")
            )
            if execution_mode == "fok":
                self._prime_pm_client_order_metadata(intent)
                create_started_perf = time.perf_counter()
                try:
                    order = MarketOrderArgs(
                        token_id=str(intent["token_id"]),
                        amount=float(intent["bet_usdc"]),
                        side=BUY,
                        price=float(
                            intent.get("order_price_cap", POLYMARKET_ORDER_PRICE_CAP)
                        ),
                        fee_rate_bps=int(intent.get("fee_rate_bps", 0) or 0),
                        order_type=OrderType.FOK,
                    )
                    signed_order = self.pm_client.create_market_order(
                        order, options=order_options
                    )
                finally:
                    if latency_trace is not None:
                        latency_trace["pm_latency_create_market_order_ms"] = (
                            _elapsed_ms(create_started_perf)
                        )

                if latency_trace is not None:
                    latency_trace["pm_order_dispatch_at"] = _utc_now()
                    ws_received_perf = latency_trace.get("_perf_ws_received")
                    if ws_received_perf is not None:
                        _latency_set_from_perf(
                            latency_trace,
                            "pm_latency_ws_to_order_dispatch_ms",
                            ws_received_perf,
                            time.perf_counter(),
                        )
                post_started_perf = time.perf_counter()
                try:
                    response = self.pm_client.post_order(signed_order, OrderType.FOK)
                finally:
                    if latency_trace is not None:
                        latency_trace["pm_latency_post_order_ms"] = _elapsed_ms(
                            post_started_perf
                        )
                        latency_trace["pm_order_response_at"] = _utc_now()
                        ws_received_perf = latency_trace.get("_perf_ws_received")
                        if ws_received_perf is not None:
                            _latency_set_from_perf(
                                latency_trace,
                                "pm_latency_ws_to_order_response_ms",
                                ws_received_perf,
                                time.perf_counter(),
                            )
                success_status = "submitted_fok"
            else:
                raise NotImplementedError(
                    "Unsupported POLY_EXECUTION_MODE: "
                    f"{self.pm_cfg.execution_mode!r}"
                )
            response_txt = _json_compact(response)
            response_success = (
                bool(response.get("success", False))
                if isinstance(response, dict)
                else "error" not in response_txt.lower()
            )
            if not response_success:
                return self._submit_result(
                    commit_bankroll=False,
                    status="submission_rejected",
                    response=response_txt,
                )
            return self._submit_result(
                commit_bankroll=True,
                status=success_status,
                response=response_txt,
            )
        except Exception as exc:
            return self._submit_result(
                commit_bankroll=False,
                status="submission_error",
                error=str(exc),
            )
        finally:
            if latency_trace is not None:
                latency_trace["pm_latency_submit_total_ms"] = _elapsed_ms(
                    submit_started_perf
                )

    def _resolve_pending(self):
        if not self.records:
            return 0

        with self.records_lock:
            pending_boundaries = self._pending_settlement_boundaries(self.records)
            pending_records = [dict(rec) for rec in self.records]
        if self.settlement_source == "polymarket":
            self._refresh_polymarket_markets(pending_records)
        else:
            self._refresh_settlement_candles(pending_boundaries)

        resolved_now = 0
        resolved_at = pd.Timestamp.now(tz="UTC")
        with self.records_lock:
            for rec in self.records:
                if rec["actual_up"] is not None:
                    continue

                if not self._resolve_record_outcome_from_settlement_truth(
                    rec,
                    resolved_at=resolved_at,
                ):
                    continue

                rec["trade_is_win"] = None
                rec["payout_usdc"] = None
                rec["pnl_usdc"] = None
                rec["bankroll_after_resolve"] = None

                if rec.get("pm_order_status") == "submitted_fok":
                    rec["pm_settlement_status"] = "resolved_waiting_settlement"
                else:
                    rec["pm_settlement_status"] = rec.get("pm_settlement_status") or "resolved_no_position"
                _mark_record_dirty(self.pm_dirty_record_ids, rec)
                resolved_now += 1

        return resolved_now

    def _resolve_market_snapshot(self, bucket_start, market_future, latency_trace=None):
        wait_started_perf = time.perf_counter()
        try:
            result = market_future.result()
            if self._prefetched_market_payload_is_fresh(bucket_start, result):
                self._merge_prefetched_market_latency(
                    latency_trace,
                    result.get("latency_payload"),
                )
                return result["snapshot"]
            if isinstance(result, dict):
                return self._fetch_market_snapshot_traced(
                    bucket_start, latency_trace=latency_trace
                )
            return result
        except Exception:
            return self._fetch_market_snapshot_traced(
                bucket_start, latency_trace=latency_trace
            )
        finally:
            resolved_perf = time.perf_counter()
            if latency_trace is not None:
                latency_trace["pm_latency_market_lookup_wait_ms"] = _elapsed_ms(
                    wait_started_perf, resolved_perf
                )

    def _evaluate_prediction_execution(
        self,
        *,
        bucket_start,
        proba_up,
        market_future,
        latency_trace=None,
    ):
        market = None
        market_error = ""
        intent = {"reason": "not_evaluated"}
        submit_result = self._submit_result(
            commit_bankroll=False,
            status="not_attempted",
        )

        try:
            market = self._resolve_market_snapshot(
                bucket_start,
                market_future,
                latency_trace=latency_trace,
            )
            recommendation_started_perf = time.perf_counter()
            intent = self._recommend_polymarket_bet(prob_up_raw=proba_up, market=market)
            if latency_trace is not None:
                latency_trace["pm_latency_recommendation_ms"] = _elapsed_ms(
                    recommendation_started_perf
                )
            submit_result = self._maybe_submit_order(
                intent, latency_trace=latency_trace
            )
        except Exception as exc:
            market_error = str(exc)
            intent = {"reason": "market_lookup_failed"}
            submit_result = self._submit_result(
                commit_bankroll=False,
                status="market_lookup_failed",
                error=str(exc),
            )

        return {
            "market": market,
            "market_error": market_error,
            "intent": intent,
            "submit_result": submit_result,
        }

    def _build_prediction_record(
        self,
        *,
        bucket_start,
        bucket_end,
        proba_up,
        signal_up,
        bankroll_before_entry,
        bankroll_after_entry,
        stake_usdc,
        market,
        market_error,
        intent,
        submit_result,
        latency_trace,
    ):
        order_status = str(submit_result["status"])
        latency_payload = _latency_snapshot(bucket_start, latency_trace)

        return {
            "record_id": f"bucket:{pd.Timestamp(bucket_start).isoformat()}",
            "pm_model_hash": self.model_hash,
            "pm_run_started_at_utc": self.run_started_at_utc,
            "prediction_time": _utc_now(),
            "bucket_start": bucket_start,
            "bucket_end": bucket_end,
            "proba_up": proba_up,
            "threshold": self.prediction_threshold,
            "signal_up": signal_up,
            "kelly_side": str(intent.get("side", "none")),
            "kelly_fraction": float(intent.get("fraction", 0.0) or 0.0),
            "kelly_bet_usdc": float(intent.get("bet_usdc", 0.0) or 0.0),
            "kelly_edge": float(intent.get("edge", np.nan)),
            "kelly_prob_win_adj": float(intent.get("prob_win_adj", np.nan)),
            "kelly_prob_win_raw": float(intent.get("prob_win_raw", np.nan)),
            "kelly_reason": str(intent.get("reason", "")),
            "stake_usdc": float(stake_usdc),
            "entry_price": float(intent.get("entry_price", np.nan)),
            "entry_fee_usdc": float(intent.get("fee_usdc", 0.0) or 0.0),
            "entry_fee_raw_usdc": float(intent.get("fee_raw_usdc", 0.0) or 0.0),
            "shares_net": float(intent.get("shares_net", 0.0) or 0.0),
            "kelly_c_eff": float(intent.get("c_eff", np.nan)),
            "kelly_eff_rate": float(intent.get("eff_rate", np.nan)),
            "price_eps": np.nan,
            "price_slip": np.nan,
            "bankroll_before_entry": float(bankroll_before_entry),
            "bankroll_after_entry": float(bankroll_after_entry),
            "bankroll_after_resolve": None,
            "trade_is_win": None,
            "payout_usdc": None,
            "pnl_usdc": None,
            "bucket_open_price": None,
            "bucket_close_price": None,
            "actual_up": None,
            "is_correct": None,
            "resolved_at": None,
            "pm_mode": "paper" if self.pm_cfg.paper_mode else "live",
            "pm_series_slug": self.pm_cfg.series_slug,
            "pm_execution_mode": self.pm_cfg.execution_mode,
            "pm_market_slug": "" if market is None else market.market_slug,
            "pm_market_question": "" if market is None else market.market_question,
            "pm_market_end": "" if market is None else market.market_end,
            "pm_condition_id": "" if market is None else market.condition_id,
            "pm_up_token_id": "" if market is None else market.up_token_id,
            "pm_down_token_id": "" if market is None else market.down_token_id,
            "pm_selected_token_id": str(intent.get("token_id", "")),
            "pm_accepting_orders": False if market is None else market.accepting_orders,
            "pm_restricted": False if market is None else market.restricted,
            "pm_fees_enabled": False if market is None else market.fees_enabled,
            "pm_fee_rate_bps": 0 if market is None else market.fee_rate_bps,
            "pm_tick_size": np.nan if market is None else float(market.tick_size),
            "pm_order_min_size": (
                np.nan if market is None else float(market.order_min_size)
            ),
            "pm_order_price_cap": float(
                intent.get("order_price_cap", POLYMARKET_ORDER_PRICE_CAP)
            ),
            "pm_position_size": np.nan,
            "pm_position_current_value": np.nan,
            "pm_position_redeemable": False,
            "pm_settlement_status": (
                "entry_submitted" if order_status == "submitted_fok" else ""
            ),
            "pm_account_sync_at": self.pm_last_account_sync_at,
            "pm_account_sync_reason": self.pm_last_account_sync_reason,
            "pm_redeem_tx_id": "",
            "pm_redeem_tx_hash": "",
            "pm_redeem_tx_state": "",
            "pm_redeem_error": "",
            "pm_up_best_bid": np.nan if market is None else float(market.up_best_bid),
            "pm_up_best_ask": np.nan if market is None else float(market.up_best_ask),
            "pm_down_best_bid": (
                np.nan if market is None else float(market.down_best_bid)
            ),
            "pm_down_best_ask": (
                np.nan if market is None else float(market.down_best_ask)
            ),
            "pm_seconds_to_close": float(intent.get("seconds_to_close", np.nan)),
            "pm_order_status": order_status,
            "pm_order_response": str(submit_result["response"]),
            "pm_allowance_info": str(submit_result["allowance_info"]),
            "pm_market_error": market_error,
            "pm_order_error": str(submit_result["error"]),
            "pm_exit_order_status": "",
            "pm_exit_order_response": "",
            "pm_exit_order_error": "",
            "pm_exit_reason": "",
            "pm_exit_price": np.nan,
            "pm_exit_shares": np.nan,
            "pm_exit_fee_usdc": np.nan,
            "pm_exit_proceeds_usdc": np.nan,
            **latency_payload,
        }

    def _build_prediction_summary(
        self,
        *,
        minute_close,
        bucket_start,
        bucket_end,
        proba_up,
        signal_up,
        stake_usdc,
        bankroll_before_entry,
        bankroll_after_entry,
        intent,
        submit_result,
        latency_trace,
    ):
        latency_payload = _latency_snapshot(bucket_start, latency_trace)
        return {
            "decision_local": minute_close.tz_convert(self.local_tz).isoformat(),
            "bucket_start": bucket_start,
            "bucket_end": bucket_end,
            "proba_up": proba_up,
            "signal_up": signal_up,
            "kelly_side": str(intent.get("side", "none")),
            "kelly_fraction": float(intent.get("fraction", 0.0) or 0.0),
            "kelly_bet_usdc": float(intent.get("bet_usdc", 0.0) or 0.0),
            "stake_usdc": float(stake_usdc),
            "bankroll_before_entry": float(bankroll_before_entry),
            "bankroll_after_entry": float(bankroll_after_entry),
            "kelly_reason": str(intent.get("reason", "")),
            "kelly_edge": float(intent.get("edge", np.nan)),
            "pm_order_status": str(submit_result["status"]),
            "pm_order_error": str(submit_result["error"]),
            "pm_market_lookup_attempts": latency_payload["pm_market_lookup_attempts"],
            "pm_market_start_to_ws_received_ms": float(
                latency_payload["pm_market_start_to_ws_received_ms"]
            ),
            "pm_market_start_to_submit_ready_ms": float(
                latency_payload["pm_market_start_to_submit_ready_ms"]
            ),
            "pm_market_start_to_order_dispatch_ms": float(
                latency_payload["pm_market_start_to_order_dispatch_ms"]
            ),
            "pm_latency_binance_event_to_ws_received_ms": float(
                latency_payload["pm_latency_binance_event_to_ws_received_ms"]
            ),
            "pm_latency_market_lookup_total_ms": float(
                latency_payload["pm_latency_market_lookup_total_ms"]
            ),
            "pm_latency_market_lookup_queue_ms": float(
                latency_payload["pm_latency_market_lookup_queue_ms"]
            ),
            "pm_latency_market_lookup_wait_ms": float(
                latency_payload["pm_latency_market_lookup_wait_ms"]
            ),
            "pm_latency_submit_total_ms": float(
                latency_payload["pm_latency_submit_total_ms"]
            ),
            "pm_latency_create_market_order_ms": float(
                latency_payload["pm_latency_create_market_order_ms"]
            ),
            "pm_latency_post_order_ms": float(
                latency_payload["pm_latency_post_order_ms"]
            ),
            "pm_latency_ws_to_submit_ready_ms": float(
                latency_payload["pm_latency_ws_to_submit_ready_ms"]
            ),
            "pm_latency_predict_total_ms": float(
                latency_payload["pm_latency_predict_total_ms"]
            ),
        }

    def _predict_next_bucket(self, volume_profile_values=None, latency_trace=None):
        predict_started_perf = time.perf_counter()
        latency_trace = {} if latency_trace is None else latency_trace
        if latency_trace.get("pm_predict_started_at") is None:
            latency_trace["pm_predict_started_at"] = _utc_now()
        minute_open = self.opened_candles[-1]
        minute_close = minute_open + pd.Timedelta(minutes=1)
        bucket_start = self._bucket_start_for_latest_candle()
        bucket_end = bucket_start + pd.Timedelta(minutes=self.target_bucket_minutes - 1)
        market_future = self._market_lookup_future_for_bucket(
            bucket_start, latency_trace=latency_trace
        )
        if volume_profile_values is None:
            volume_profile_started_perf = time.perf_counter()
            volume_profile_values = self._extract_volume_profile_features_for_latest_candle()
            latency_trace["pm_latency_volume_profile_ms"] = _elapsed_ms(
                volume_profile_started_perf
            )

        feature_vector_started_perf = time.perf_counter()
        feature_vector = self._build_feature_vector(
            volume_profile_values=volume_profile_values
        )
        latency_trace["pm_latency_feature_vector_ms"] = _elapsed_ms(
            feature_vector_started_perf
        )
        model_predict_started_perf = time.perf_counter()
        proba_up = float(self.model.predict(feature_vector)[0])
        latency_trace["pm_latency_model_predict_ms"] = _elapsed_ms(
            model_predict_started_perf
        )
        signal_up = int(proba_up >= self.prediction_threshold)
        bankroll_before_entry = float(self.live_bankroll_usdc)
        execution = self._evaluate_prediction_execution(
            bucket_start=bucket_start,
            proba_up=proba_up,
            market_future=market_future,
            latency_trace=latency_trace,
        )
        latency_trace["pm_latency_predict_total_ms"] = _elapsed_ms(predict_started_perf)
        intent = execution["intent"]
        submit_result = execution["submit_result"]
        stake_usdc = (
            float(intent.get("bet_usdc", 0.0) or 0.0)
            if bool(submit_result["commit_bankroll"])
            else 0.0
        )
        if self.pm_cfg.paper_mode and stake_usdc > 0.0:
            self.live_bankroll_usdc -= stake_usdc
        bankroll_after_entry = float(self.live_bankroll_usdc)

        record = self._build_prediction_record(
            bucket_start=bucket_start,
            bucket_end=bucket_end,
            proba_up=proba_up,
            signal_up=signal_up,
            bankroll_before_entry=bankroll_before_entry,
            bankroll_after_entry=bankroll_after_entry,
            stake_usdc=stake_usdc,
            market=execution["market"],
            market_error=str(execution["market_error"]),
            intent=intent,
            submit_result=submit_result,
            latency_trace=latency_trace,
        )
        with self.records_lock:
            self.records.append(record)
            _mark_record_dirty(self.pm_dirty_record_ids, record)
        self.predicted_buckets.add(bucket_start)

        return self._build_prediction_summary(
            minute_close=minute_close,
            bucket_start=bucket_start,
            bucket_end=bucket_end,
            proba_up=proba_up,
            signal_up=signal_up,
            stake_usdc=stake_usdc,
            bankroll_before_entry=bankroll_before_entry,
            bankroll_after_entry=bankroll_after_entry,
            intent=intent,
            submit_result=submit_result,
            latency_trace=latency_trace,
        )

    def _has_binary_flag(self, value):
        return value is not None and pd.notna(value)

    def _record_is_resolved(self, record):
        return self._has_binary_flag(record.get("actual_up")) and self._has_binary_flag(
            record.get("is_correct")
        )

    def _record_is_traded(self, record):
        return (
            _safe_text(record.get("pm_settlement_status")) == "closed"
            and self._has_binary_flag(record.get("trade_is_win"))
            and record.get("pnl_usdc") is not None
        )

    def _running_win_rates(self, records):
        resolved_rates = []
        traded_rates = []
        resolved_count = 0
        resolved_wins = 0
        traded_count = 0
        traded_wins = 0

        for record in records:
            if not self._record_is_resolved(record):
                resolved_rates.append(np.nan)
                traded_rates.append(np.nan)
                continue

            resolved_count += 1
            resolved_wins += int(record["is_correct"])
            if self._record_is_traded(record):
                traded_count += 1
                traded_wins += int(record["trade_is_win"])

            resolved_rates.append(float(resolved_wins / resolved_count))
            traded_rates.append(
                float(traded_wins / traded_count) if traded_count else np.nan
            )

        return resolved_rates, traded_rates

    def _serialize_timestamp_columns(self, frame, columns):
        for col in columns:
            frame[col] = frame[col].map(
                lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else ""
            )

    def _records_to_output_frame(
        self, records, snapshot_at=None, rates_by_record_id=None
    ):
        out = pd.DataFrame(records)
        if "record_id" not in out.columns:
            out["record_id"] = [
                _stable_record_id(record) for record in records
            ]
        if snapshot_at is None:
            out["record_snapshot_at"] = ""
        else:
            out["record_snapshot_at"] = pd.Timestamp(snapshot_at)
        if rates_by_record_id is None:
            win_rate_resolved, win_rate_traded = self._running_win_rates(records)
            out["win_rate_resolved"] = win_rate_resolved
            out["win_rate_traded"] = win_rate_traded
        else:
            out["win_rate_resolved"] = [
                rates_by_record_id.get(_safe_text(record.get("record_id")), (np.nan, np.nan))[0]
                for record in records
            ]
            out["win_rate_traded"] = [
                rates_by_record_id.get(_safe_text(record.get("record_id")), (np.nan, np.nan))[1]
                for record in records
            ]

        for col in LIVE_TRADE_EXPORT_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan
        out = out.loc[:, list(LIVE_TRADE_EXPORT_COLUMNS)]
        self._serialize_timestamp_columns(
            out,
            [
                "record_snapshot_at",
                "prediction_time",
                "bucket_start",
                "bucket_end",
                "resolved_at",
                *LIVE_TRADE_LATENCY_TIMESTAMP_COLUMNS,
            ],
        )
        return out

    def _format_rate(self, value):
        return "n/a" if not np.isfinite(value) else f"{value * 100:.2f}%"

    def _stats(self):
        records = self._records_snapshot()
        resolved = sum(1 for rec in records if self._has_binary_flag(rec["actual_up"]))
        resolved_wins = sum(
            int(rec["is_correct"]) for rec in records if self._record_is_resolved(rec)
        )
        traded = sum(1 for rec in records if self._record_is_traded(rec))
        traded_wins = sum(
            int(rec["trade_is_win"]) for rec in records if self._record_is_traded(rec)
        )
        resolved_win_rate = (
            float(resolved_wins / resolved) if resolved else float("nan")
        )
        traded_win_rate = float(traded_wins / traded) if traded else float("nan")
        total_pnl = float(
            sum(
                float(rec.get("pnl_usdc", 0.0) or 0.0)
                for rec in records
                if rec["actual_up"] is not None
            )
        )
        return {
            "resolved": resolved,
            "resolved_wins": resolved_wins,
            "resolved_losses": resolved - resolved_wins,
            "resolved_win_rate": resolved_win_rate,
            "traded": traded,
            "traded_wins": traded_wins,
            "traded_losses": traded - traded_wins,
            "traded_win_rate": traded_win_rate,
            "total_pnl": total_pnl,
        }

    def _save_records(self):
        with self.pm_save_lock:
            records = self._records_snapshot()
            if not records:
                return
            for rec in records:
                rec["record_id"] = _stable_record_id(rec)

            snapshot_at = _utc_now()
            if self.pm_storage_requires_rewrite or not self.predictions_path.exists():
                frame = self._records_to_output_frame(records, snapshot_at=snapshot_at)
                frame.to_csv(self.predictions_path, index=False)
                self.pm_storage_requires_rewrite = False
                self.pm_dirty_record_ids.clear()
                self.pm_persisted_record_signatures = {
                    rec["record_id"]: _record_state_signature(rec)
                    for rec in records
                    if rec.get("record_id")
                }
                return

            dirty_ids = {
                record_id for record_id in self.pm_dirty_record_ids if record_id
            }
            candidate_records = (
                [
                    rec
                    for rec in records
                    if _safe_text(rec.get("record_id")) in dirty_ids
                ]
                if dirty_ids
                else records
            )
            changed_records = []
            for rec in candidate_records:
                record_id = _safe_text(rec.get("record_id"))
                if not record_id:
                    continue
                signature = _record_state_signature(rec)
                if signature == self.pm_persisted_record_signatures.get(record_id):
                    continue
                changed_records.append(rec)

            if not changed_records:
                return

            resolved_rates, traded_rates = self._running_win_rates(records)
            rates_by_record_id = {
                _safe_text(record.get("record_id")): (
                    resolved_rates[idx],
                    traded_rates[idx],
                )
                for idx, record in enumerate(records)
            }
            # Persist one latest snapshot per record_id; appending snapshots caused
            # the CSV to balloon with duplicate rows for the same bucket.
            frame = self._records_to_output_frame(
                records,
                snapshot_at=snapshot_at,
                rates_by_record_id=rates_by_record_id,
            )
            frame.to_csv(self.predictions_path, index=False)
            self.pm_storage_requires_rewrite = False
            self.pm_dirty_record_ids.clear()
            self.pm_persisted_record_signatures = {
                _safe_text(rec.get("record_id")): _record_state_signature(rec)
                for rec in records
                if _safe_text(rec.get("record_id"))
            }

    def _log(self, tag, pred=None):
        stats = self._stats()
        resolved_win_rate_txt = self._format_rate(stats["resolved_win_rate"])
        traded_win_rate_txt = self._format_rate(stats["traded_win_rate"])
        ts = (
            str(pred["decision_local"])
            if pred is not None
            else pd.Timestamp.now(tz="UTC").tz_convert(self.local_tz).isoformat()
        )

        if pred is not None:
            self._print_recent_candle_buffer(count=5)

        msg = [
            ts,
            f"[{tag}]",
            f"resolved={stats['resolved']}",
            f"resolved_wins={stats['resolved_wins']}",
            f"resolved_losses={stats['resolved_losses']}",
            f"win_rate_resolved={resolved_win_rate_txt}",
            f"traded={stats['traded']}",
            f"traded_wins={stats['traded_wins']}",
            f"traded_losses={stats['traded_losses']}",
            f"win_rate_traded={traded_win_rate_txt}",
            f"total_pnl={stats['total_pnl']:.2f}",
            f"bankroll={self.live_bankroll_usdc:.2f}",
        ]
        if pred is not None:
            msg.extend(
                [
                    f"proba_up={pred['proba_up']:.6f}",
                    f"signal_up={pred['signal_up']}",
                    f"kelly_side={pred['kelly_side']}",
                    f"kelly_edge={pred['kelly_edge']:.6f}",
                    f"kelly_reason={pred['kelly_reason']}",
                    f"stake_usdc={pred['stake_usdc']:.2f}",
                    f"pm_order_status={pred['pm_order_status']}",
                ]
            )
            if pred.get("pm_order_error"):
                msg.append(f"pm_order_error={pred['pm_order_error']}")
            for label, key in (
                ("market_lookup_attempts", "pm_market_lookup_attempts"),
                ("lookup_wall_ms", "pm_latency_market_lookup_total_ms"),
                ("lookup_queue_ms", "pm_latency_market_lookup_queue_ms"),
                ("lookup_wait_ms", "pm_latency_market_lookup_wait_ms"),
                ("ws_to_ready_ms", "pm_latency_ws_to_submit_ready_ms"),
                ("submit_ms", "pm_latency_submit_total_ms"),
                ("create_ms", "pm_latency_create_market_order_ms"),
                ("post_ms", "pm_latency_post_order_ms"),
                ("predict_ms", "pm_latency_predict_total_ms"),
                ("handler_ms", "pm_latency_on_message_total_ms"),
            ):
                value = pred.get(key, np.nan)
                if np.isfinite(value):
                    if key == "pm_market_lookup_attempts":
                        msg.append(f"{label}={int(value)}")
                    else:
                        msg.append(f"{label}={float(value):.1f}")
        if np.isfinite(self.pm_cash_balance_usdc):
            msg.append(f"cash_balance={self.pm_cash_balance_usdc:.2f}")
        if np.isfinite(self.pm_positions_value_usdc):
            msg.append(f"positions_value={self.pm_positions_value_usdc:.2f}")
        print(" ".join(msg))
        if pred is not None:
            self._print_indicator_nan_status()

    def _maybe_sync_missing_candles(self, opened_from_ws):
        if not self.opened_candles:
            return

        expected_next = self.opened_candles[-1] + INTERVAL_DELTA
        if opened_from_ws > expected_next:
            self._sync_closed_candles_from_rest(stop_before_opened=opened_from_ws)

    def _maybe_predict_closed_bucket(self, opened, volume_profile_values, latency_trace=None):
        bucket_start = opened.floor(f"{self.target_bucket_minutes}min")
        bucket_end = bucket_start + pd.Timedelta(minutes=self.target_bucket_minutes - 1)
        if opened != bucket_end:
            return None

        next_bucket_start = bucket_start + pd.Timedelta(
            minutes=self.target_bucket_minutes
        )
        if next_bucket_start in self.predicted_buckets:
            return None

        return self._predict_next_bucket(
            volume_profile_values=volume_profile_values,
            latency_trace=latency_trace,
        )

    def _schedule_post_cycle_syncs(self, pred, resolved_now):
        if pred is not None and pred.get("pm_order_status") == "submitted_fok":
            self._schedule_background_sync("post_submit", force=True)
        if resolved_now > 0:
            self._schedule_background_sync("post_resolve", force=True)
        if not self.pm_cfg.paper_mode:
            self._schedule_background_sync("maintenance", force=False)

    def _persist_cycle_results(self, pred, resolved_now):
        if resolved_now <= 0 and pred is None:
            return
        self._save_records()
        self._log("resolve+pred" if pred else "resolve", pred=pred)

    def _print_live_runtime_configuration(self):
        if self.pm_cfg.disable_order_submission:
            print(
                "Live test mode is enabled | "
                "external writes disabled via POLY_DISABLE_ORDER_SUBMIT=1 "
                f"request_timeout_sec={self.pm_cfg.market_request_timeout_sec:.2f} "
                f"clob_http_timeout_sec={self.pm_cfg.clob_http_timeout_sec:.2f} "
                f"market_lookup_max_wait_ms={self.pm_cfg.market_lookup_max_wait_ms} "
                f"market_prefetch_lead_ms={self.pm_cfg.market_lookup_prefetch_lead_ms} "
                f"exposure_cap_usdc={self.pm_cfg.max_exposure_usdc} "
                f"bankroll_cap_usdc={self.pm_cfg.max_bankroll_usdc}"
            )
        else:
            print(
                "Live submission mode is enabled | "
                f"request_timeout_sec={self.pm_cfg.market_request_timeout_sec:.2f} "
                f"clob_http_timeout_sec={self.pm_cfg.clob_http_timeout_sec:.2f} "
                f"market_lookup_max_wait_ms={self.pm_cfg.market_lookup_max_wait_ms} "
                f"market_prefetch_lead_ms={self.pm_cfg.market_lookup_prefetch_lead_ms} "
                f"exposure_cap_usdc={self.pm_cfg.max_exposure_usdc} "
                f"bankroll_cap_usdc={self.pm_cfg.max_bankroll_usdc}"
            )

        if self.pm_allowance_info:
            print(f"Allowance snapshot: {self.pm_allowance_info}")
        if np.isfinite(self.pm_cash_balance_usdc):
            print(f"Polymarket cash balance: {self.pm_cash_balance_usdc:.2f}")
        if self._relayer_is_configured():
            print(
                "Auto-redeem background mode is enabled | "
                f"relayer_host={self.pm_cfg.relayer_host} "
                f"relayer_api_key_address={self.pm_relayer_api_key_address}"
            )
        else:
            self._warn_relayer_unavailable_once(
                "set POLY_RELAYER_API_KEY and optionally POLY_RELAYER_API_KEY_ADDRESS"
            )

    def _print_runtime_configuration(self):
        print(
            "Polymarket execution | "
            f"mode={'paper' if self.pm_cfg.paper_mode else 'live'} "
            f"price_source={PRICE_SOURCE} volume_source={VOLUME_SOURCE} "
            f"series_slug={self.pm_cfg.series_slug} "
            f"market_slug_prefix={self.pm_cfg.market_slug_prefix} "
            f"settlement_source={self.settlement_source} "
            f"execution_mode={self.pm_cfg.execution_mode} "
            f"order_price_cap={POLYMARKET_ORDER_PRICE_CAP:.3f} "
            f"records={self.predictions_path}"
        )
        if self.pm_cfg.paper_mode:
            print(
                "Paper mode bankroll source | "
                f"POLY_START_BANKROLL_USDC={self.pm_cfg.start_bankroll_usdc:.2f} "
                f"bankroll_cap_usdc={self.pm_cfg.max_bankroll_usdc}"
            )
        else:
            self._print_live_runtime_configuration()

        print(
            "Kelly sizing | "
            f"bankroll={self.live_bankroll_usdc:.2f} "
            f"fractional_kelly={self.live_kelly_sizing['fractional_kelly']:.6f} "
            f"cap={self.live_kelly_sizing['cap']:.6f} "
            f"min_edge={self.live_kelly_sizing['min_edge']:.6f} "
            f"prob_shrink={self.live_kelly_sizing['prob_shrink']:.6f} "
            f"min_stake_usdc={self.live_kelly_sizing['min_stake_usdc']:.2f}"
        )
        print(f"Bankroll source: {self.bankroll_source}")
        print(f"Records file: {self.predictions_path}")

    def _run_websocket_once(self):
        ws = WebSocketApp(
            WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        ws.run_forever(
            ping_interval=WS_PING_INTERVAL_SEC,
            ping_timeout=WS_PING_TIMEOUT_SEC,
        )

    def _on_message(self, _ws, message):
        try:
            self._poll_background_sync()
            message_started_perf = time.perf_counter()
            ws_received_perf = time.perf_counter()
            ws_received_at = _utc_now()
            payload = json.loads(message)
            closed_candle, live_minute_opened, event_at = self._consume_ws_payload(
                payload
            )
            if closed_candle is None:
                return
            latency_trace = {
                "_perf_ws_received": ws_received_perf,
                "pm_ws_received_at": ws_received_at,
                "pm_binance_event_at": event_at,
            }

            opened_from_ws = pd.to_datetime(int(closed_candle["t"]), unit="ms", utc=True)
            sync_started_perf = time.perf_counter()
            self._maybe_sync_missing_candles(opened_from_ws)
            latency_trace["pm_latency_sync_missing_candles_ms"] = _elapsed_ms(
                sync_started_perf
            )

            upsert_started_perf = time.perf_counter()
            opened = self._upsert_closed_candle(closed_candle)
            latency_trace["pm_latency_upsert_closed_candle_ms"] = _elapsed_ms(
                upsert_started_perf
            )
            if opened is None:
                return
            if (
                self.last_processed_closed_opened is not None
                and opened <= self.last_processed_closed_opened
            ):
                return
            if PRICE_SOURCE == "index":
                sync_after_started_perf = time.perf_counter()
                self._maybe_sync_missing_candles(live_minute_opened)
                latency_trace["pm_latency_sync_missing_candles_ms"] = float(
                    latency_trace["pm_latency_sync_missing_candles_ms"]
                    + _elapsed_ms(sync_after_started_perf)
                )

            volume_profile_started_perf = time.perf_counter()
            volume_profile_values = self._extract_volume_profile_features_for_latest_candle()
            latency_trace["pm_latency_volume_profile_ms"] = _elapsed_ms(
                volume_profile_started_perf
            )
            pred = self._maybe_predict_closed_bucket(
                opened,
                volume_profile_values,
                latency_trace=latency_trace,
            )
            resolved_now = self._resolve_pending()

            self._update_volume_profile_state_for_latest_candle(opened)
            self.last_processed_closed_opened = opened
            self._schedule_market_snapshot_prefetch(self._next_unpredicted_bucket_start())
            latency_trace["pm_latency_on_message_total_ms"] = _elapsed_ms(
                message_started_perf
            )
            if pred is not None:
                pred["pm_latency_on_message_total_ms"] = float(
                    latency_trace["pm_latency_on_message_total_ms"]
                )
                with self.records_lock:
                    if self.records:
                        self.records[-1]["pm_latency_on_message_total_ms"] = float(
                            latency_trace["pm_latency_on_message_total_ms"]
                        )

            self._schedule_post_cycle_syncs(pred, resolved_now)
            self._persist_cycle_results(pred, resolved_now)

            self._poll_background_sync()
        except Exception as exc:
            print(f"[pred] message handling failed: {exc}")

    def run_forever(self):
        self._print_runtime_configuration()
        now_utc = pd.Timestamp.now(tz="UTC")
        next_bucket_start = now_utc.floor(f"{self.target_bucket_minutes}min") + pd.Timedelta(
            minutes=self.target_bucket_minutes
        )
        while next_bucket_start in self.predicted_buckets:
            next_bucket_start += pd.Timedelta(minutes=self.target_bucket_minutes)
        self._schedule_market_snapshot_prefetch(next_bucket_start)
        self._schedule_background_sync("startup", force=True)

        delay = 1
        while True:
            try:
                self._run_websocket_once()
            except Exception as exc:
                print(f"[ws] run failed: {exc}")

            print(f"[ws] reconnect in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, MAX_WS_RECONNECT_DELAY_SEC)


def main():
    trader = PolymarketLiveTrader()

    now_utc = pd.Timestamp.now(tz="UTC")
    next_resolve = now_utc.floor(f"{trader.target_bucket_minutes}min") + pd.Timedelta(
        minutes=trader.target_bucket_minutes
    )
    print(f"[wait] first resolve+pred around {next_resolve.isoformat()}")
    trader.run_forever()


if __name__ == "__main__":
    main()
