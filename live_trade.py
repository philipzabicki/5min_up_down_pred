import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime as std_datetime, timedelta as std_timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import requests
import py_clob_client_v2.headers.headers as pyclob_headers
from eth_account import Account
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
from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import (
    AssetType,
    BalanceAllowanceParams,
    MarketOrderArgs,
    PartialCreateOrderOptions,
    OrderType,
)
from py_clob_client_v2.http_helpers import helpers as pyclob_http_helpers
from py_clob_client_v2.order_builder.constants import BUY, SELL

from live_utils import (
    LIVE_SHARED_MARKET_DATA_COLUMNS,
    LIVE_TRADE_EXPORT_COLUMNS,
    build_live_market_data_path,
    build_live_trade_records_path,
    read_records_state,
    upsert_records_csv,
    write_records_csv,
    write_records_state,
)
from project_env import load_repo_env

load_repo_env()

from live_predict_binance import (
    INTERVAL_DELTA,
    INTERVAL,
    LIVE_PROFILE,
    LIVE_TRADE_DIR,
    LivePredictor,
    PRICE_SOURCE,
    PRICE_MARKET,
    resolve_model_accuracy_from_proba,
    resolve_model_side_from_proba,
    resolve_record_accuracy_from_side,
    SYMBOL,
    VOLUME_MARKET,
    VOLUME_SYMBOL,
    VOLUME_SOURCE,
    WS_TARGETS,
    _load_ws_payload,
)
from polymarket_fee_utils import (
    DEFAULT_POLYMARKET_FEE_ROUND_DECIMALS,
    DEFAULT_POLYMARKET_MIN_FEE_USDC,
    normalize_polymarket_fee_model,
    polymarket_fee_model_from_market,
    polymarket_taker_fee_usdc_from_shares,
)
from polymarket_redeem_utils import (
    POLYMARKET_BINARY_INDEX_SETS,
    POLYMARKET_RELAYER_PENDING_STATES,
    POLYMARKET_RELAYER_TERMINAL_STATES,
    build_redeem_transactions as build_redeem_transaction_specs,
    collect_redeem_candidates as collect_redeem_candidate_specs,
    encode_redeem_positions_call,
    resolve_redeem_collateral_address,
    resolve_redeem_ctf_address,
    resolve_redeem_target_address,
    resolve_relayer_tx_type,
)
from trade_policy import (
    build_trade_intent,
    decide_trade_from_model_direction,
    decide_trade_from_ev,
    resolve_fee_fractions_from_quotes,
)

DEFAULT_GAMMA_HOST = "https://gamma-api.polymarket.com"
DEFAULT_CLOB_HOST = "https://clob.polymarket.com"
DEFAULT_DATA_API_HOST = "https://data-api.polymarket.com"
DEFAULT_RELAYER_HOST = "https://relayer-v2.polymarket.com"
POLYMARKET_COLLATERAL_DECIMALS = 6
POLYMARKET_VALID_TICK_SIZES = {"0.1", "0.01", "0.001", "0.0001"}
POLYMARKET_CLOSED_POSITIONS_PAGE_LIMIT = 50
POLYMARKET_BACKGROUND_SYNC_MIN_INTERVAL_SEC = 2.0
POLYMARKET_AUTH_CLOCK_SKEW_WARN_SEC = 5.0
POLYMARKET_EXECUTION_ORDER_TYPES = {
    "fok": OrderType.FOK,
    "fak": OrderType.FAK,
}
POLYMARKET_SUBMITTED_ORDER_STATUSES = frozenset(
    f"submitted_{mode}" for mode in POLYMARKET_EXECUTION_ORDER_TYPES
)
POLYMARKET_RETRYABLE_SUBMISSION_STATUS = "submission_retryable_425"
POLYMARKET_POST_ORDER_RETRYABLE_STATUS_CODES = frozenset({425})
POLYMARKET_POST_ORDER_MAX_RETRIES = 3
POLYMARKET_POST_ORDER_RETRY_INITIAL_DELAY_SEC = 1.0
POLYMARKET_POST_ORDER_RETRY_MAX_DELAY_SEC = 4.0

_PYCLOB_AUTH_TIME_OFFSET_SEC = 0.0


class _OffsetDatetime:
    @classmethod
    def now(cls, tz=None):
        return std_datetime.now(tz=tz) + std_timedelta(
            seconds=float(_PYCLOB_AUTH_TIME_OFFSET_SEC)
        )


def _env_text(name, default=""):
    raw = os.getenv(name)
    return raw.strip() if raw is not None else default


def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _profile_text(profile, key, default=""):
    raw = profile.get(key, default)
    return str(raw).strip() if raw is not None else str(default).strip()


def _profile_float(profile, key, default):
    raw = profile.get(key, default)
    return float(raw)


def _profile_int(profile, key, default):
    raw = profile.get(key, default)
    return int(raw)


def _profile_bool(profile, key, default):
    raw = profile.get(key, default)
    if isinstance(raw, bool):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


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
    if (
        previous_client is not None
        and previous_client is not pyclob_http_helpers._http_client
    ):
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
        if not text or text.lower() in {"nan", "nat", "none", "null"}:
            return default
        return text
    try:
        if bool(pd.isna(value)):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "null"}:
        return default
    return text


def _supported_polymarket_execution_modes():
    return sorted(POLYMARKET_EXECUTION_ORDER_TYPES)


def _polymarket_order_type_for_execution_mode(execution_mode):
    mode = _safe_text(execution_mode).lower()
    order_type = POLYMARKET_EXECUTION_ORDER_TYPES.get(mode)
    if order_type is None:
        raise NotImplementedError(
            "Unsupported live.polymarket_execution_mode. Supported values: "
            f"{_supported_polymarket_execution_modes()}; got {execution_mode!r}"
        )
    return order_type


def _polymarket_submitted_status_for_execution_mode(execution_mode):
    mode = _safe_text(execution_mode).lower()
    if mode not in POLYMARKET_EXECUTION_ORDER_TYPES:
        raise NotImplementedError(
            "Unsupported live.polymarket_execution_mode. Supported values: "
            f"{_supported_polymarket_execution_modes()}; got {execution_mode!r}"
        )
    return f"submitted_{mode}"


def _is_polymarket_submitted_status(status):
    return _safe_text(status) in POLYMARKET_SUBMITTED_ORDER_STATUSES


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


def _best_size(levels, side):
    best_price = _best_price(levels, side=side)
    if not np.isfinite(best_price):
        return float("nan")
    size_sum = 0.0
    found = False
    for level in levels or []:
        level_price = _safe_float(level.get("price"))
        if not np.isfinite(level_price) or abs(level_price - best_price) > 1e-12:
            continue
        level_size = _safe_float(level.get("size"))
        if not np.isfinite(level_size):
            continue
        size_sum += float(level_size)
        found = True
    return float(size_sum) if found else float("nan")


def _resolve_submitted_buy_price(
    *,
    entry_price,
    order_price_cap,
    submitted_price_mode,
):
    entry_price_value = _safe_float(entry_price)
    if not np.isfinite(entry_price_value) or not (0.0 < entry_price_value < 1.0):
        return float("nan"), "invalid_entry_price"

    submitted_price_mode_text = _safe_text(submitted_price_mode).lower()
    if submitted_price_mode_text in {"", "entry_price"}:
        return float(entry_price_value), ""
    if submitted_price_mode_text != "order_price_cap":
        return float("nan"), f"unsupported_submitted_price_mode:{submitted_price_mode_text}"

    order_price_cap_value = _safe_float(order_price_cap)
    if not np.isfinite(order_price_cap_value) or not (0.0 < order_price_cap_value < 1.0):
        return float("nan"), "invalid_order_price_cap"
    if entry_price_value > order_price_cap_value + 1e-12:
        return float("nan"), "entry_price_above_order_price_cap"

    return float(order_price_cap_value), ""


def _json_compact(payload):
    try:
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    except TypeError:
        return str(payload)


def _http_status_code(exc):
    status_code = getattr(exc, "status_code", None)
    try:
        if status_code is not None:
            return int(status_code)
    except (TypeError, ValueError):
        pass
    response = getattr(exc, "response", None)
    if response is None:
        return None
    return int(getattr(response, "status_code", 0) or 0)


def _submission_error_status_from_exception(exc):
    if _http_status_code(exc) in POLYMARKET_POST_ORDER_RETRYABLE_STATUS_CODES:
        return POLYMARKET_RETRYABLE_SUBMISSION_STATUS
    return "submission_error"


def _utc_now():
    return pd.Timestamp.now(tz="UTC")


def _delay_ms_since(timestamp, *, now=None):
    started_at = pd.Timestamp(timestamp)
    finished_at = _utc_now() if now is None else pd.Timestamp(now)
    delay_ms = (finished_at - started_at).total_seconds() * 1000.0
    return float(max(delay_ms, 0.0))


def _elapsed_ms(started_perf):
    return float(max((time.perf_counter() - float(started_perf)) * 1000.0, 0.0))


def _delay_ms_between(start, end):
    if start is None or end is None:
        return float("nan")
    return _delay_ms_since(start, now=end)


def _is_missing_orderbook_http_error(exc):
    if not isinstance(exc, requests.HTTPError):
        return False
    if _http_status_code(exc) != 404:
        return False
    response = getattr(exc, "response", None)
    if response is None:
        return False
    text = str(getattr(response, "text", "") or "")
    return "No orderbook exists for the requested token id" in text


def _set_pyclob_auth_time_offset(offset_sec):
    global _PYCLOB_AUTH_TIME_OFFSET_SEC
    _PYCLOB_AUTH_TIME_OFFSET_SEC = float(offset_sec)
    pyclob_headers.datetime = _OffsetDatetime


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


def _response_payload(response):
    if isinstance(response, dict):
        return response
    if not isinstance(response, str):
        return None
    text = response.strip()
    if not text or text[0] not in "{[":
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _extract_buy_fill_metrics_from_response(order_response):
    payload = _response_payload(order_response)
    if payload is None:
        return float("nan"), float("nan")
    return (
        _safe_float(payload.get("takingAmount")),
        _safe_float(payload.get("makingAmount")),
    )


def _resolve_buy_record_fields(intent, submit_result):
    requested_stake = _safe_float(intent.get("bet_usdc"))
    requested_price = _safe_float(intent.get("entry_price"))
    requested_fee = _safe_float(intent.get("entry_fee_usdc"))
    requested_fee_raw = _safe_float(intent.get("entry_fee_raw_usdc"))
    requested_shares = _safe_float(intent.get("shares_net"))
    committed = bool(submit_result.get("commit_bankroll")) and (
        _safe_text(intent.get("final_reason")) == "ok"
    )
    filled_stake = _safe_float(submit_result.get("filled_stake_usdc"))
    filled_shares = _safe_float(submit_result.get("filled_shares"))

    if np.isfinite(filled_stake):
        actual_stake = float(filled_stake)
    elif committed and np.isfinite(requested_stake):
        actual_stake = float(requested_stake)
    else:
        actual_stake = 0.0

    if np.isfinite(filled_shares):
        actual_shares = float(filled_shares)
    elif committed and np.isfinite(requested_shares):
        actual_shares = float(requested_shares)
    else:
        actual_shares = 0.0

    return {
        "stake_usdc": float(actual_stake),
        "entry_price": float(requested_price) if np.isfinite(requested_price) else np.nan,
        "entry_fee_usdc": float(requested_fee) if np.isfinite(requested_fee) else 0.0,
        "entry_fee_raw_usdc": (
            float(requested_fee_raw) if np.isfinite(requested_fee_raw) else 0.0
        ),
        "shares_net": float(actual_shares),
        "entry_stake_usdc_orig": (
            float(requested_stake)
            if np.isfinite(requested_stake)
            else float(actual_stake)
        ),
        "entry_price_orig": float(requested_price) if np.isfinite(requested_price) else np.nan,
        "entry_fee_usdc_orig": (
            float(requested_fee) if np.isfinite(requested_fee) else 0.0
        ),
        "entry_fee_raw_usdc_orig": (
            float(requested_fee_raw) if np.isfinite(requested_fee_raw) else 0.0
        ),
        "entry_shares_net_orig": (
            float(requested_shares)
            if np.isfinite(requested_shares)
            else float(actual_shares)
        ),
    }


def _backfill_record_analysis_fields(record):
    shares_from_response, stake_from_response = _extract_buy_fill_metrics_from_response(
        record.get("pm_order_response")
    )
    current_stake = _safe_float(record.get("stake_usdc"))
    current_price = _safe_float(record.get("entry_price"))
    current_fee = _safe_float(record.get("entry_fee_usdc"))
    current_fee_raw = _safe_float(record.get("entry_fee_raw_usdc"))
    current_shares = _safe_float(record.get("shares_net"))
    current_bankroll_before = _safe_float(record.get("bankroll_before_entry"))
    current_bankroll_after = _safe_float(record.get("bankroll_after_entry"))
    if not np.isfinite(_safe_float(record.get("entry_stake_usdc_orig"))):
        if np.isfinite(stake_from_response):
            record["entry_stake_usdc_orig"] = float(stake_from_response)
        elif np.isfinite(current_stake):
            record["entry_stake_usdc_orig"] = float(current_stake)
        else:
            record["entry_stake_usdc_orig"] = np.nan

    if not np.isfinite(_safe_float(record.get("entry_price_orig"))):
        record["entry_price_orig"] = (
            float(current_price) if np.isfinite(current_price) else np.nan
        )

    if not np.isfinite(_safe_float(record.get("entry_fee_usdc_orig"))):
        record["entry_fee_usdc_orig"] = (
            float(current_fee) if np.isfinite(current_fee) else np.nan
        )

    if not np.isfinite(_safe_float(record.get("entry_fee_raw_usdc_orig"))):
        record["entry_fee_raw_usdc_orig"] = (
            float(current_fee_raw) if np.isfinite(current_fee_raw) else np.nan
        )

    if not np.isfinite(_safe_float(record.get("entry_shares_net_orig"))):
        if np.isfinite(shares_from_response):
            record["entry_shares_net_orig"] = float(shares_from_response)
        elif np.isfinite(current_shares):
            record["entry_shares_net_orig"] = float(current_shares)
        else:
            record["entry_shares_net_orig"] = np.nan

    if not np.isfinite(_safe_float(record.get("bankroll_before_entry_orig"))):
        record["bankroll_before_entry_orig"] = (
            float(current_bankroll_before)
            if np.isfinite(current_bankroll_before)
            else np.nan
        )

    if not np.isfinite(_safe_float(record.get("bankroll_after_entry_orig"))):
        record["bankroll_after_entry_orig"] = (
            float(current_bankroll_after) if np.isfinite(current_bankroll_after) else np.nan
        )

    actual_up = record.get("actual_up")
    if actual_up is not None and pd.notna(actual_up):
        record["is_correct"] = resolve_record_accuracy_from_side(
            record,
            actual_up=int(actual_up),
        )

    record.setdefault("pm_order_response", "")
    record.setdefault("pm_exit_order_response", "")
    record.setdefault("pm_position_avg_price", np.nan)
    record.setdefault("pm_position_initial_value_usdc", np.nan)
    record.setdefault("pm_closed_avg_price", np.nan)
    record.setdefault("pm_closed_total_bought_usdc", np.nan)
    record.setdefault("pm_closed_realized_pnl_usdc", np.nan)
    record.setdefault("pm_closed_payout_usdc", np.nan)
    record.setdefault("pm_account_cash_balance_usdc", np.nan)
    record.setdefault("pm_account_positions_value_usdc", np.nan)
    record.setdefault("pm_account_sync_at_entry", None)
    record.setdefault("pm_account_cash_balance_entry_usdc", np.nan)
    record.setdefault("pm_account_positions_value_entry_usdc", np.nan)
    record.setdefault("pm_account_sync_at_resolve", None)
    record.setdefault("pm_account_cash_balance_resolve_usdc", np.nan)
    record.setdefault("pm_account_positions_value_resolve_usdc", np.nan)
    record.setdefault("decision_delay_ms", np.nan)
    record.setdefault("ws_price_event_delay_ms", np.nan)
    record.setdefault("ws_volume_event_delay_ms", np.nan)
    record.setdefault("ws_price_receive_delay_ms", np.nan)
    record.setdefault("ws_volume_receive_delay_ms", np.nan)
    record.setdefault("ws_event_delay_ms", np.nan)
    record.setdefault("ws_receive_delay_ms", np.nan)
    record.setdefault("ws_component_sync_ms", np.nan)
    record.setdefault("feature_prep_ms", np.nan)
    record.setdefault("feature_vector_ms", np.nan)
    record.setdefault("model_predict_ms", np.nan)
    record.setdefault("policy_compute_ms", np.nan)
    record.setdefault("market_prefetch_hit", False)
    record.setdefault("market_prefetch_age_ms", np.nan)
    record.setdefault("market_lookup_source", "")
    record.setdefault("market_lookup_ms", np.nan)
    record.setdefault("submit_order_ms", np.nan)
    record.setdefault("execution_ms", np.nan)
    record.setdefault("pm_fee_source", "")
    record.setdefault("pm_fee_rate", np.nan)
    record.setdefault("pm_fee_exponent", np.nan)
    record.setdefault("pm_fee_round_decimals", np.nan)
    record.setdefault("pm_min_fee_usdc", np.nan)
    record.setdefault("pm_exit_decision_at", None)
    record.setdefault("pm_exit_best_bid", np.nan)
    record.setdefault("pm_exit_seconds_to_close", np.nan)
    record.setdefault("pm_exit_candidate_pnl_usdc", np.nan)
    record.setdefault("pm_exit_candidate_roi", np.nan)
    record.setdefault("pm_exit_redeem_pnl_usdc", np.nan)
    record.setdefault("pm_exit_min_allowed_pnl_usdc", np.nan)
    record.setdefault("pm_redeem_condition_id", _safe_text(record.get("pm_condition_id")))
    record.setdefault("pm_redeem_collateral_token", "")
    record.setdefault("pm_redeem_ctf_address", "")
    record.setdefault("pm_redeem_target_address", "")
    record.setdefault("pm_redeem_relayer_tx_type", "")
    record.setdefault("pm_redeem_index_sets", "")
    record.setdefault("pm_redeem_signer_address", "")
    record.setdefault("pm_redeem_funder_address", "")
    record.setdefault("pm_redeem_nonce", "")
    record.setdefault("pm_redeem_tx_id", "")
    record.setdefault("pm_redeem_tx_hash", "")
    record.setdefault("pm_redeem_tx_state", "")
    record.setdefault("pm_redeem_error", "")
    record.setdefault("pm_redeem_submitted_at", None)
    record.setdefault("pm_redeem_confirmed_at", None)
    record.setdefault("pm_settlement_payout_source", "")

def _fee_model_from_record(record):
    rate = _safe_float(record.get("pm_fee_rate"))
    exponent = _safe_float(record.get("pm_fee_exponent"))
    if not np.isfinite(rate) or rate < 0.0:
        return None
    if not np.isfinite(exponent) or exponent <= 0.0:
        return None

    return normalize_polymarket_fee_model(
        {
            "rate": float(rate),
            "exponent": float(exponent),
            "fee_round_decimals": int(
                _safe_float(
                    record.get("pm_fee_round_decimals"),
                    DEFAULT_POLYMARKET_FEE_ROUND_DECIMALS,
                )
            ),
            "min_fee": float(
                _safe_float(
                    record.get("pm_min_fee_usdc"),
                    DEFAULT_POLYMARKET_MIN_FEE_USDC,
                )
            ),
            "source": str(record.get("pm_fee_source", "trade_record") or "trade_record"),
        },
        context="trade record fee model",
    )


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


def load_polymarket_settings(trade_records_path):
    live_profile = dict(LIVE_PROFILE)
    order_price_cap = _profile_float(
        live_profile,
        "polymarket_order_price_cap",
        0.56,
    )
    if not np.isfinite(order_price_cap) or not (0.0 < order_price_cap < 1.0):
        raise ValueError(
            "live.polymarket_order_price_cap must be a finite float strictly "
            "between 0 and 1."
        )
    signature_type = _profile_int(live_profile, "polymarket_signature_type", 0)
    redeem_resolved_positions = _env_bool(
        "POLY_REDEEM_RESOLVED_POSITIONS",
        _profile_bool(
            live_profile,
            "polymarket_redeem_resolved_positions",
            True,
        ),
    )
    return PolymarketSettings(
        gamma_host=_profile_text(
            live_profile,
            "polymarket_gamma_host",
            DEFAULT_GAMMA_HOST,
        ),
        clob_host=_profile_text(
            live_profile,
            "polymarket_clob_host",
            DEFAULT_CLOB_HOST,
        ),
        data_api_host=_profile_text(
            live_profile,
            "polymarket_data_api_host",
            DEFAULT_DATA_API_HOST,
        ),
        relayer_host=_profile_text(
            live_profile,
            "polymarket_relayer_host",
            DEFAULT_RELAYER_HOST,
        ),
        series_slug=_profile_text(
            live_profile,
            "polymarket_series_slug",
            "btc-up-or-down-5m",
        ),
        market_slug_prefix=_profile_text(
            live_profile,
            "polymarket_market_slug_prefix",
            "btc-updown-5m",
        ),
        market_slug_override=_profile_text(
            live_profile,
            "polymarket_market_slug_override",
            "",
        ),
        paper_mode=_profile_bool(live_profile, "polymarket_paper_mode", True),
        disable_order_submission=_profile_bool(
            live_profile,
            "polymarket_disable_order_submission",
            False,
        ),
        signature_type=signature_type,
        chain_id=_profile_int(live_profile, "polymarket_chain_id", 137),
        private_key=_env_text("POLY_PRIVATE_KEY", ""),
        funder=_env_text("POLY_FUNDER_ADDRESS", ""),
        max_exposure_usdc=_profile_float(
            live_profile,
            "polymarket_max_exposure_usdc",
            math.inf,
        ),
        max_bankroll_usdc=_profile_float(
            live_profile,
            "polymarket_max_bankroll_usdc",
            math.inf,
        ),
        no_trade_last_seconds=_profile_int(
            live_profile,
            "polymarket_no_trade_last_seconds",
            20,
        ),
        start_bankroll_usdc=_profile_float(
            live_profile,
            "polymarket_start_bankroll_usdc",
            1000.0,
        ),
        trade_records_path=Path(trade_records_path),
        market_request_timeout_sec=_profile_float(
            live_profile,
            "polymarket_market_request_timeout_sec",
            3.0,
        ),
        clob_http_timeout_sec=_profile_float(
            live_profile,
            "polymarket_clob_http_timeout_sec",
            _profile_float(
                live_profile,
                "polymarket_market_request_timeout_sec",
                3.0,
            ),
        ),
        market_lookup_max_wait_ms=_profile_int(
            live_profile,
            "polymarket_market_lookup_max_wait_ms",
            2500,
        ),
        market_lookup_retry_ms=_profile_int(
            live_profile,
            "polymarket_market_lookup_retry_ms",
            100,
        ),
        market_lookup_prefetch_lead_ms=_profile_int(
            live_profile,
            "polymarket_market_lookup_prefetch_lead_ms",
            1200,
        ),
        market_lookup_prefetch_max_age_ms=_profile_int(
            live_profile,
            "polymarket_market_lookup_prefetch_max_age_ms",
            2500,
        ),
        execution_mode=_profile_text(
            live_profile,
            "polymarket_execution_mode",
            "fok",
        ),
        order_price_cap=float(order_price_cap),
        relayer_api_key=_env_text("POLY_RELAYER_API_KEY", ""),
        relayer_api_key_address=_env_text("POLY_RELAYER_API_KEY_ADDRESS", ""),
        relayer_tx_type=resolve_relayer_tx_type(
            os.environ,
            signature_type=signature_type,
        ),
        redeem_collateral_token_address=resolve_redeem_collateral_address(
            os.environ
        ),
        redeem_ctf_address=resolve_redeem_ctf_address(os.environ),
        redeem_target_address=resolve_redeem_target_address(os.environ),
        redeem_require_redeemable=_env_bool(
            "POLY_REDEEM_REQUIRE_REDEEMABLE",
            True,
        ),
        import_untracked_open_positions=_profile_bool(
            live_profile,
            "polymarket_import_untracked_open_positions",
            False,
        ),
        enable_exit_orders=_profile_bool(
            live_profile,
            "polymarket_enable_exit_orders",
            True,
        ),
        exit_min_profit_usdc=_profile_float(
            live_profile,
            "polymarket_exit_min_profit_usdc",
            0.15,
        ),
        exit_min_roi=_profile_float(
            live_profile,
            "polymarket_exit_min_roi",
            0.01,
        ),
        exit_min_seconds_to_close=_profile_int(
            live_profile,
            "polymarket_exit_min_seconds_to_close",
            45,
        ),
        exit_redeem_profit_tolerance=_profile_float(
            live_profile,
            "polymarket_exit_redeem_profit_tolerance",
            0.01,
        ),
        redeem_resolved_positions=redeem_resolved_positions,
    )


class PolymarketLiveTrader(LivePredictor):
    def __init__(self):
        super().__init__()
        default_trade_records_path = build_live_trade_records_path(
            live_trade_dir=LIVE_TRADE_DIR,
            symbol=SYMBOL,
            interval=INTERVAL,
            run_started_at_utc=self.run_started_at_utc,
            model_hash=self.model_hash,
            policy_config_hash=self.trade_policy_config_hash,
            modeling_dataset_config_hash=self.modeling_dataset_config_hash,
        )
        self.pm_cfg = load_polymarket_settings(default_trade_records_path)
        _configure_clob_http_client(self.pm_cfg.clob_http_timeout_sec)
        self.live_trade_policy = self.trade_policy_runtime
        if self.pm_cfg.execution_mode not in POLYMARKET_EXECUTION_ORDER_TYPES:
            raise NotImplementedError(
                "Unsupported live.polymarket_execution_mode. Supported values: "
                f"{_supported_polymarket_execution_modes()}; "
                f"got {self.pm_cfg.execution_mode!r}"
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
        self.trade_records_path = self.pm_cfg.trade_records_path
        self.trade_records_path.parent.mkdir(parents=True, exist_ok=True)
        self.market_data_path = build_live_market_data_path()
        self.market_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.trade_records_state_path = self.trade_records_path.with_suffix(
            ".state.json"
        )

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
        self.pm_redeem_submit_lock = threading.Lock()
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
        self.pm_relayer_contract_config = get_relayer_contract_config(
            self.pm_cfg.chain_id
        )
        self.pm_relayer_signer = (
            RelayerSigner(self.pm_cfg.private_key, self.pm_cfg.chain_id)
            if self.pm_cfg.private_key
            else None
        )
        self.pm_relayer_warning_printed = False
        self.pm_client = None
        self.pm_allowance_info = ""
        self.bankroll_source = self._bankroll_source_label("profile_start_bankroll")
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

    def _sync_pyclob_auth_clock(self):
        try:
            response = self.pm_session.get(
                f"{self.pm_cfg.clob_host.rstrip('/')}/time",
                timeout=float(self.pm_cfg.market_request_timeout_sec),
            )
            response.raise_for_status()
            server_ts = float(str(response.text).strip())
        except Exception as exc:
            print(f"[pm] auth clock sync skipped: {exc}")
            _set_pyclob_auth_time_offset(0.0)
            return 0.0

        local_ts = float(time.time())
        offset_sec = float(server_ts - local_ts)
        _set_pyclob_auth_time_offset(offset_sec)
        if abs(offset_sec) >= POLYMARKET_AUTH_CLOCK_SKEW_WARN_SEC:
            print(
                "[pm] auth clock offset detected | "
                f"local_vs_clob={offset_sec:+.1f}s "
                "using server-aligned timestamps for py_clob_client_v2 auth"
            )
        return offset_sec

    def _build_live_client(self):
        if not self.pm_cfg.private_key:
            raise ValueError(
                "POLY_PRIVATE_KEY is required when live.polymarket_paper_mode=false."
            )
        if not self.pm_cfg.funder:
            raise ValueError(
                "POLY_FUNDER_ADDRESS is required when live.polymarket_paper_mode=false."
            )
        if self.pm_cfg.signature_type not in {0, 1, 2}:
            raise ValueError(
                "live.polymarket_signature_type must be one of {0, 1, 2} "
                "(0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE)."
            )
        signer_address = Account.from_key(self.pm_cfg.private_key).address
        signer_matches_funder = signer_address.lower() == self.pm_cfg.funder.lower()
        if self.pm_cfg.signature_type in {1, 2} and signer_matches_funder:
            print(
                "[warn] live.polymarket_signature_type uses a proxy-wallet flow, "
                "but the configured funder address matches the signer address. "
                "For type 1/2, the configured funder should usually be the proxy "
                "wallet address from polymarket.com/settings, not the private-key address."
            )
        signature_type_labels = {
            0: "EOA",
            1: "POLY_PROXY",
            2: "POLY_GNOSIS_SAFE",
        }
        auth_clock_offset_sec = self._sync_pyclob_auth_clock()

        try:
            client = ClobClient(
                self.pm_cfg.clob_host,
                key=self.pm_cfg.private_key,
                chain_id=self.pm_cfg.chain_id,
                signature_type=self.pm_cfg.signature_type,
                funder=self.pm_cfg.funder,
            )
            client.set_api_creds(client.create_or_derive_api_key())
            return client
        except Exception as exc:
            msg = str(exc)
            if "Invalid L1 Request headers" in msg:
                raise RuntimeError(
                    "Polymarket L1 auth failed: Invalid L1 Request headers. "
                    "Docs and py_clob_client_v2 source indicate L1 auth is signed only by "
                    "POLY_PRIVATE_KEY + live.polymarket_chain_id, before "
                    "funder/signature_type are "
                    "used for order building. "
                    f"derived_signer={signer_address} "
                    f"funder={self.pm_cfg.funder} "
                    f"signature_type={self.pm_cfg.signature_type}"
                    f"({signature_type_labels.get(self.pm_cfg.signature_type, 'unknown')}) "
                    f"auth_clock_offset_sec={auth_clock_offset_sec:+.1f}. "
                    "If this account is a normal EOA, set "
                    "live.polymarket_signature_type=0 and configure "
                    "POLY_FUNDER_ADDRESS to the same address as the private key. "
                    "If this is a proxy/safe setup, keep the proxy funder address from "
                    "polymarket.com/settings and make sure the private key belongs to the "
                    "linked signer wallet for that account."
                ) from exc
            raise

    def _fetch_live_balance_allowance(self):
        if self.pm_client is None:
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
        if not self.trade_records_state_path.exists():
            return

        try:
            loaded_records = read_records_state(self.trade_records_state_path)
        except Exception as exc:
            print(f"[pm] failed to load existing state: {exc}")
            return

        for rec in loaded_records:
            rec["record_id"] = _stable_record_id(rec)
            _backfill_record_analysis_fields(rec)
        with self.records_lock:
            self.records = loaded_records
        self.predicted_buckets = {
            rec["bucket_start"]
            for rec in loaded_records
            if isinstance(rec.get("bucket_start"), pd.Timestamp)
        }

    def _relayer_is_configured(self):
        return bool(
            self.pm_cfg.relayer_api_key
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
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                "relayer_get_failed "
                f"path={path} status={response.status_code} body={response.text}"
            ) from exc
        return response.json()

    def _relayer_post_json(self, path, payload):
        url = f"{self.pm_cfg.relayer_host.rstrip('/')}/{path.lstrip('/')}"
        response = self.pm_session.post(
            url,
            json=payload,
            headers=self._relayer_headers(),
            timeout=float(self.pm_cfg.market_request_timeout_sec),
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                "relayer_post_failed "
                f"path={path} status={response.status_code} body={response.text}"
            ) from exc
        return response.json()

    def _relayer_get_nonce(self):
        payload = self._relayer_get_json(
            "/nonce",
            {"address": self.pm_signer_address, "type": self.pm_cfg.relayer_tx_type},
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
        return encode_redeem_positions_call(
            condition_id,
            collateral_token_address=self.pm_cfg.redeem_collateral_token_address,
            index_sets=POLYMARKET_BINARY_INDEX_SETS,
        )

    def _build_redeem_transactions(self, candidates):
        specs = build_redeem_transaction_specs(
            candidates,
            collateral_token_address=self.pm_cfg.redeem_collateral_token_address,
            ctf_address=self.pm_cfg.redeem_ctf_address,
            target_address=self.pm_cfg.redeem_target_address,
            relayer_tx_type=self.pm_cfg.relayer_tx_type,
            index_sets=POLYMARKET_BINARY_INDEX_SETS,
        )
        transactions = []
        for spec in specs:
            transactions.append(
                RelayerSafeTransaction(
                    to=spec["to"],
                    operation=RelayerOperationType.Call,
                    data=spec["data"],
                    value=spec["value"],
                )
            )
            print(
                "[pm] redeem tx built | "
                f"conditionId={spec['conditionId']} "
                f"collateralToken={spec['collateralToken']} "
                f"ctfAddress={spec['ctfAddress']} "
                f"targetAddress={spec['to']} "
                f"indexSets={spec['indexSets']} "
                f"relayerTxType={spec['relayerTxType']} "
                f"signer={self.pm_signer_address} "
                f"funder={self.pm_cfg.funder}"
            )
        condition_ids = [spec["conditionId"] for spec in specs]
        return transactions, condition_ids

    def _mark_redeem_submit_error(self, *, condition_ids, error):
        if not condition_ids:
            return
        now = pd.Timestamp.now(tz="UTC")
        target_conditions = set(str(x) for x in condition_ids if str(x))
        with self.records_lock:
            for rec in self.records:
                condition_id = _safe_text(rec.get("pm_condition_id"))
                if condition_id not in target_conditions:
                    continue
                rec["pm_redeem_condition_id"] = condition_id
                rec["pm_redeem_collateral_token"] = (
                    self.pm_cfg.redeem_collateral_token_address
                )
                rec["pm_redeem_ctf_address"] = self.pm_cfg.redeem_ctf_address
                rec["pm_redeem_target_address"] = self.pm_cfg.redeem_target_address
                rec["pm_redeem_relayer_tx_type"] = self.pm_cfg.relayer_tx_type
                rec["pm_redeem_index_sets"] = _json_compact(
                    list(POLYMARKET_BINARY_INDEX_SETS)
                )
                rec["pm_redeem_signer_address"] = self.pm_signer_address
                rec["pm_redeem_funder_address"] = self.pm_cfg.funder
                rec["pm_redeem_tx_state"] = "SUBMIT_FAILED"
                rec["pm_redeem_error"] = str(error)
                rec["pm_redeem_submitted_at"] = now
                rec["pm_settlement_status"] = "redeem_submit_failed"

    def _submit_redeem_batch(self, candidates):
        if not candidates:
            return
        if not self.pm_redeem_submit_lock.acquire(blocking=False):
            print("[pm] redeem skip | reason=redeem_submit_already_running")
            return
        try:
            candidates = self._collect_redeem_candidates(
                [item.get("position", item) for item in candidates],
                log_decisions=False,
            )
            if not candidates:
                return
            condition_ids = [str(item.get("conditionId", "")) for item in candidates]
            if self.pm_cfg.disable_order_submission:
                print(
                    "[pm] redeem skip | "
                    "reason=order_submission_disabled "
                    f"conditionIds={condition_ids}"
                )
                return
            if not self._relayer_is_configured():
                self._warn_relayer_unavailable_once(
                    "missing relayer credentials or relayer signer config"
                )
                return
            if self.pm_cfg.relayer_tx_type != "SAFE":
                self._mark_redeem_submit_error(
                    condition_ids=condition_ids,
                    error=(
                        "unsupported_relayer_tx_type_for_redeem:"
                        f"{self.pm_cfg.relayer_tx_type}. SAFE is supported by this "
                        "implementation; WALLET deposit-wallet batches require the "
                        "new relayer client flow."
                    ),
                )
                return
            if not self._relayer_safe_is_deployed():
                self._mark_redeem_submit_error(
                    condition_ids=condition_ids,
                    error=f"relayer_safe_not_deployed:{self.pm_cfg.funder}",
                )
                return

            transactions, condition_ids = self._build_redeem_transactions(candidates)
            if not transactions:
                return

            nonce = self._relayer_get_nonce()
            print(
                "[pm] redeem nonce | "
                f"relayerTxType={self.pm_cfg.relayer_tx_type} "
                f"signer={self.pm_signer_address} "
                f"funder={self.pm_cfg.funder} "
                f"nonce={nonce} "
                f"conditionIds={condition_ids}"
            )
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
            tx_state = str(response.get("state", "") or "STATE_NEW")
            print(
                "[pm] redeem submitted | "
                f"transactionID={tx_id} "
                f"transactionHash={tx_hash} "
                f"state={tx_state} "
                f"conditionIds={condition_ids}"
            )
            if not tx_id:
                raise RuntimeError(f"relayer_submit_missing_transaction_id:{response}")
            self._mark_redeem_submission(
                condition_ids=condition_ids,
                tx_id=tx_id,
                tx_hash=tx_hash,
                tx_state=tx_state,
                error="",
                nonce=nonce,
            )
        except Exception as exc:
            self._mark_redeem_submit_error(
                condition_ids=[
                    str(item.get("conditionId", ""))
                    for item in candidates
                    if str(item.get("conditionId", ""))
                ],
                error=str(exc),
            )
            print(
                "[pm] redeem submit failed | "
                f"relayerTxType={self.pm_cfg.relayer_tx_type} "
                f"signer={self.pm_signer_address} "
                f"funder={self.pm_cfg.funder} "
                f"error={exc}"
            )
        finally:
            self.pm_redeem_submit_lock.release()

    def _mark_redeem_submission(
        self,
        *,
        condition_ids,
        tx_id,
        tx_hash,
        tx_state,
        error,
        nonce,
    ):
        target_conditions = set(str(x) for x in condition_ids if str(x))
        submitted_at = pd.Timestamp.now(tz="UTC")
        with self.records_lock:
            for rec in self.records:
                condition_id = _safe_text(rec.get("pm_condition_id"))
                if condition_id not in target_conditions:
                    continue
                rec["pm_redeem_condition_id"] = condition_id
                rec["pm_redeem_collateral_token"] = (
                    self.pm_cfg.redeem_collateral_token_address
                )
                rec["pm_redeem_ctf_address"] = self.pm_cfg.redeem_ctf_address
                rec["pm_redeem_target_address"] = self.pm_cfg.redeem_target_address
                rec["pm_redeem_relayer_tx_type"] = self.pm_cfg.relayer_tx_type
                rec["pm_redeem_index_sets"] = _json_compact(
                    list(POLYMARKET_BINARY_INDEX_SETS)
                )
                rec["pm_redeem_signer_address"] = self.pm_signer_address
                rec["pm_redeem_funder_address"] = self.pm_cfg.funder
                rec["pm_redeem_nonce"] = nonce
                rec["pm_redeem_tx_id"] = tx_id
                rec["pm_redeem_tx_hash"] = tx_hash
                rec["pm_redeem_tx_state"] = tx_state
                rec["pm_redeem_error"] = error
                rec["pm_redeem_submitted_at"] = submitted_at
                rec["pm_settlement_status"] = "redeem_submitted"

    def _update_redeem_transaction_state(self, *, tx_id, tx_hash, tx_state, error=""):
        confirmed_at = (
            pd.Timestamp.now(tz="UTC") if tx_state == "STATE_CONFIRMED" else None
        )
        with self.records_lock:
            for rec in self.records:
                if _safe_text(rec.get("pm_redeem_tx_id")) != _safe_text(tx_id):
                    continue
                rec["pm_redeem_tx_hash"] = tx_hash or rec.get("pm_redeem_tx_hash", "")
                rec["pm_redeem_tx_state"] = tx_state
                rec["pm_redeem_error"] = error
                if tx_state == "STATE_CONFIRMED":
                    rec["pm_redeem_confirmed_at"] = confirmed_at
                    rec["pm_settlement_status"] = "redeem_confirmed_waiting_close_sync"
                elif tx_state in {"STATE_FAILED", "STATE_INVALID"}:
                    rec["pm_settlement_status"] = "redeem_failed"
        print(
            "[pm] redeem poll | "
            f"transactionID={tx_id} "
            f"transactionHash={tx_hash} "
            f"state={tx_state} "
            f"error={error}"
        )

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
            "pm_policy_hash": self.trade_policy_config_hash,
            "pm_run_started_at_utc": self.run_started_at_utc,
            "prediction_time": pd.Timestamp.now(tz="UTC"),
            "bucket_start": None,
            "bucket_end": None,
            "proba_up": np.nan,
            "trade_side": str(position.get("outcome", "")).lower(),
            "stake_usdc": initial_value,
            "entry_price": _safe_float(position.get("avgPrice")),
            "entry_fee_usdc": np.nan,
            "entry_fee_raw_usdc": np.nan,
            "shares_net": size,
            "entry_stake_usdc_orig": initial_value,
            "entry_price_orig": _safe_float(position.get("avgPrice")),
            "entry_fee_usdc_orig": np.nan,
            "entry_fee_raw_usdc_orig": np.nan,
            "entry_shares_net_orig": size,
            "price_eps": np.nan,
            "price_slip": np.nan,
            "ask_yes": np.nan,
            "ask_no": np.nan,
            "policy_proba_up": np.nan,
            "policy_ask_yes": np.nan,
            "policy_ask_no": np.nan,
            "policy_fee_yes": np.nan,
            "policy_fee_no": np.nan,
            "policy_extra_buffer": float(self.trade_policy_runtime["extra_buffer"]),
            "policy_ev_yes": np.nan,
            "policy_ev_no": np.nan,
            "policy_best_ev": np.nan,
            "policy_decision": "no_trade",
            "policy_reason": "external_position",
            "bankroll_before_entry": np.nan,
            "bankroll_after_entry": np.nan,
            "bankroll_before_entry_orig": np.nan,
            "bankroll_after_entry_orig": np.nan,
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
            "pm_fee_source": "",
            "pm_fee_rate": np.nan,
            "pm_fee_exponent": np.nan,
            "pm_fee_round_decimals": np.nan,
            "pm_min_fee_usdc": np.nan,
            "pm_tick_size": np.nan,
            "pm_order_min_size": np.nan,
            "pm_order_price_cap": float(self.pm_cfg.order_price_cap),
            "pm_position_size": size,
            "pm_position_current_value": current_value,
            "pm_position_redeemable": redeemable,
            "pm_position_avg_price": _safe_float(position.get("avgPrice")),
            "pm_position_initial_value_usdc": initial_value,
            "pm_closed_avg_price": np.nan,
            "pm_closed_total_bought_usdc": np.nan,
            "pm_closed_realized_pnl_usdc": np.nan,
            "pm_closed_payout_usdc": np.nan,
            "pm_settlement_status": "redeemable_open" if redeemable else "open",
            "pm_account_sync_at": sync_at,
            "pm_account_sync_reason": "startup_external_position",
            "pm_account_cash_balance_usdc": float(self.pm_cash_balance_usdc)
            if np.isfinite(self.pm_cash_balance_usdc)
            else np.nan,
            "pm_account_positions_value_usdc": float(self.pm_positions_value_usdc)
            if np.isfinite(self.pm_positions_value_usdc)
            else np.nan,
            "pm_account_sync_at_entry": sync_at,
            "pm_account_cash_balance_entry_usdc": float(self.pm_cash_balance_usdc)
            if np.isfinite(self.pm_cash_balance_usdc)
            else np.nan,
            "pm_account_positions_value_entry_usdc": float(self.pm_positions_value_usdc)
            if np.isfinite(self.pm_positions_value_usdc)
            else np.nan,
            "pm_account_sync_at_resolve": None,
            "pm_account_cash_balance_resolve_usdc": np.nan,
            "pm_account_positions_value_resolve_usdc": np.nan,
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
            "pm_order_error": "",
            "pm_order_response": "",
            "pm_exit_decision_at": None,
            "pm_exit_best_bid": np.nan,
            "pm_exit_seconds_to_close": np.nan,
            "pm_exit_candidate_pnl_usdc": np.nan,
            "pm_exit_candidate_roi": np.nan,
            "pm_exit_redeem_pnl_usdc": np.nan,
            "pm_exit_min_allowed_pnl_usdc": np.nan,
            "pm_exit_order_response": "",
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

    def _estimate_sell_proceeds(self, shares, price, fee_model):
        shares = float(shares)
        price = float(price)
        gross = shares * price
        fee_result = polymarket_taker_fee_usdc_from_shares(shares, price, fee_model)
        fee_raw = float(fee_result["fee_raw_usdc"])
        fee = float(fee_result["fee_usdc"])

        return {
            "gross_usdc": float(gross),
            "fee_usdc": float(fee),
            "fee_raw_usdc": float(fee_raw),
            "net_usdc": float(gross - fee),
            "eff_rate": float(fee_result["eff_rate"]),
        }

    def _estimate_redeem_proceeds(self, shares):
        shares = float(shares)
        if not np.isfinite(shares) or shares <= 0.0:
            return {"net_usdc": 0.0}
        return {"net_usdc": float(shares)}

    def _resolve_record_fee_model(self, record):
        stored_fee_model = _fee_model_from_record(record)
        if stored_fee_model is not None:
            return stored_fee_model

        market_slug = _safe_text(record.get("pm_market_slug"))
        if not market_slug:
            return None

        try:
            market = self._get_json(self.pm_cfg.gamma_host, f"/markets/slug/{market_slug}")
        except Exception:
            return None

        try:
            return polymarket_fee_model_from_market(
                market,
                default_round_decimals=DEFAULT_POLYMARKET_FEE_ROUND_DECIMALS,
                default_min_fee=DEFAULT_POLYMARKET_MIN_FEE_USDC,
            )
        except ValueError:
            return None

    def _collect_exit_candidates(self, open_positions):
        records_by_asset = {
            _safe_text(rec.get("pm_selected_token_id")): rec
            for rec in self._records_snapshot()
            if str(rec.get("pm_mode", "")) == "live"
            and _safe_text(rec.get("pm_selected_token_id"))
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

            try:
                best_bid_book = self._fetch_order_book_summary(asset)
            except requests.HTTPError as exc:
                if _is_missing_orderbook_http_error(exc):
                    continue
                raise
            best_bid = _safe_float(best_bid_book.get("best_bid"))
            if not np.isfinite(best_bid) or best_bid <= 0.0:
                continue

            stake_usdc = _safe_float(rec.get("entry_stake_usdc_orig"))
            if not np.isfinite(stake_usdc) or stake_usdc <= 0.0:
                stake_usdc = _safe_float(rec.get("stake_usdc"))
            if not np.isfinite(stake_usdc) or stake_usdc <= 0.0:
                continue

            fee_model = self._resolve_record_fee_model(rec)
            if fee_model is None:
                continue

            fee_rate_bps = int(rec.get("pm_fee_rate_bps", 0) or 0)
            proceeds = self._estimate_sell_proceeds(
                shares=shares,
                price=best_bid,
                fee_model=fee_model,
            )
            redeem = self._estimate_redeem_proceeds(shares=shares)
            redeem_pnl_usdc = float(redeem["net_usdc"] - stake_usdc)
            redeem_profit_tolerance = min(
                max(float(self.pm_cfg.exit_redeem_profit_tolerance), 0.0),
                1.0,
            )

            pnl_usdc = float(proceeds["net_usdc"] - stake_usdc)
            roi = float(pnl_usdc / stake_usdc)

            # Allow pre-resolution exit only if it preserves at least
            # (1 - tolerance) of the winner-redeem profit.
            if redeem_pnl_usdc > 0.0:
                min_exit_pnl_usdc = (1.0 - redeem_profit_tolerance) * redeem_pnl_usdc
                if pnl_usdc < min_exit_pnl_usdc:
                    continue
            else:
                min_exit_pnl_usdc = float("-inf")

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
                    "fee_model": fee_model,
                    "tick_size": rec.get("pm_tick_size"),
                    "neg_risk": False,
                    "stake_usdc": float(stake_usdc),
                    "pnl_usdc": pnl_usdc,
                    "roi": roi,
                    "seconds_to_close": float(seconds_to_close),
                    "redeem_pnl_usdc": float(redeem_pnl_usdc),
                    "min_exit_pnl_usdc": float(min_exit_pnl_usdc),
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
        order_type = _polymarket_order_type_for_execution_mode(
            self.pm_cfg.execution_mode
        )
        submitted_status = _polymarket_submitted_status_for_execution_mode(
            self.pm_cfg.execution_mode
        )

        status = "skipped"
        error_txt = ""
        response_txt = ""

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
                    amount=shares,  # SELL => shares, nie USDC
                    side=SELL,
                    price=price,  # floor = obecny best bid
                    order_type=order_type,
                )
                response = self._create_and_post_market_order_with_retry(
                    order, options, order_type
                )
                response_txt = _json_compact(response)
                if isinstance(response, dict) and bool(response.get("success", False)):
                    status = submitted_status
                else:
                    status = "submission_rejected"
        except Exception as exc:
            status = _submission_error_status_from_exception(exc)
            error_txt = str(exc)

        with self.records_lock:
            for rec in self.records:
                if _safe_text(rec.get("pm_selected_token_id")) != asset:
                    continue
                rec["pm_exit_order_status"] = status
                rec["pm_exit_order_error"] = error_txt
                rec["pm_exit_reason"] = "profit_take"
                rec["pm_exit_decision_at"] = _utc_now()
                rec["pm_exit_best_bid"] = price
                rec["pm_exit_seconds_to_close"] = float(
                    candidate.get("seconds_to_close", np.nan)
                )
                rec["pm_exit_candidate_pnl_usdc"] = float(
                    candidate.get("pnl_usdc", np.nan)
                )
                rec["pm_exit_candidate_roi"] = float(candidate.get("roi", np.nan))
                rec["pm_exit_redeem_pnl_usdc"] = float(
                    candidate.get("redeem_pnl_usdc", np.nan)
                )
                rec["pm_exit_min_allowed_pnl_usdc"] = float(
                    candidate.get("min_exit_pnl_usdc", np.nan)
                )
                rec["pm_exit_price"] = price
                rec["pm_exit_shares"] = shares
                rec["pm_exit_fee_usdc"] = float(candidate["fee_usdc"])
                rec["pm_exit_proceeds_usdc"] = float(candidate["proceeds_net_usdc"])
                rec["pm_exit_order_response"] = response_txt
                if status == submitted_status:
                    rec["pm_settlement_status"] = "exit_submitted"

    def _collect_redeem_candidates(self, open_positions, *, log_decisions=True):
        candidates, diagnostics = collect_redeem_candidate_specs(
            open_positions,
            self._records_snapshot(),
            market_slug_prefix=self.pm_cfg.market_slug_prefix,
            require_redeemable=self.pm_cfg.redeem_require_redeemable,
        )
        if log_decisions:
            for diag in diagnostics:
                reason = str(diag.get("reason", ""))
                action = str(diag.get("action", ""))
                if action == "candidate" or reason in {
                    "redeem_already_confirmed",
                    "redeem_already_pending",
                    "redeem_tx_pending",
                    "redeem_tx_state_unknown",
                    "negative_risk_unsupported",
                    "invalid_condition_id",
                    "not_redeemable",
                }:
                    print(
                        "[pm] redeem decision | "
                        f"action={action} "
                        f"reason={reason} "
                        f"conditionId={diag.get('conditionId', '')} "
                        f"asset={diag.get('asset', '')} "
                        f"redeemable={diag.get('redeemable', False)} "
                        f"negativeRisk={diag.get('negativeRisk', False)} "
                        f"requireRedeemable={self.pm_cfg.redeem_require_redeemable}"
                    )
        return candidates

    def _poll_background_sync(self, *, reschedule_pending=True):
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

        if pending_reason and reschedule_pending:
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
        elif not self.pm_cfg.relayer_api_key:
            self._warn_relayer_unavailable_once("POLY_RELAYER_API_KEY is missing from .env")

        cash_balance_usdc = self._refresh_live_cash_state(sync_bankroll=True)
        open_positions = self._fetch_open_positions()
        tracked_condition_ids = self._tracked_condition_ids()
        closed_positions = self._fetch_closed_positions(
            condition_ids=tracked_condition_ids
        )
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
                _backfill_record_analysis_fields(rec)
                rec["pm_account_sync_at"] = sync_at
                rec["pm_account_sync_reason"] = reason
                rec["pm_account_cash_balance_usdc"] = (
                    float(cash_balance_usdc) if np.isfinite(cash_balance_usdc) else np.nan
                )
                rec["pm_account_positions_value_usdc"] = (
                    float(self.pm_positions_value_usdc)
                    if np.isfinite(self.pm_positions_value_usdc)
                    else np.nan
                )
                if not _safe_text(rec.get("pm_account_sync_at_entry")):
                    rec["pm_account_sync_at_entry"] = _safe_text(
                        rec.get("pm_account_sync_at")
                    ) or sync_at
                if not np.isfinite(
                    _safe_float(rec.get("pm_account_cash_balance_entry_usdc"))
                ):
                    rec["pm_account_cash_balance_entry_usdc"] = (
                        float(rec["pm_account_cash_balance_usdc"])
                        if np.isfinite(_safe_float(rec.get("pm_account_cash_balance_usdc")))
                        else np.nan
                    )
                if not np.isfinite(
                    _safe_float(rec.get("pm_account_positions_value_entry_usdc"))
                ):
                    rec["pm_account_positions_value_entry_usdc"] = (
                        float(rec["pm_account_positions_value_usdc"])
                        if np.isfinite(
                            _safe_float(rec.get("pm_account_positions_value_usdc"))
                        )
                        else np.nan
                    )
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
                    rec["pm_closed_avg_price"] = (
                        float(avg_price) if np.isfinite(avg_price) else np.nan
                    )
                    rec["pm_closed_total_bought_usdc"] = (
                        float(total_bought) if np.isfinite(total_bought) else np.nan
                    )
                    rec["pm_closed_realized_pnl_usdc"] = (
                        float(realized_pnl) if np.isfinite(realized_pnl) else np.nan
                    )
                    rec["pm_closed_payout_usdc"] = (
                        float(payout) if np.isfinite(payout) else np.nan
                    )
                    rec["pm_settlement_payout_source"] = "data_api_closed_positions"
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
                    if not _safe_text(rec.get("pm_account_sync_at_resolve")):
                        rec["pm_account_sync_at_resolve"] = sync_at
                    if not np.isfinite(
                        _safe_float(rec.get("pm_account_cash_balance_resolve_usdc"))
                    ):
                        rec["pm_account_cash_balance_resolve_usdc"] = (
                            float(cash_balance_usdc)
                            if np.isfinite(cash_balance_usdc)
                            else np.nan
                        )
                    if not np.isfinite(
                        _safe_float(rec.get("pm_account_positions_value_resolve_usdc"))
                    ):
                        rec["pm_account_positions_value_resolve_usdc"] = (
                            float(self.pm_positions_value_usdc)
                            if np.isfinite(self.pm_positions_value_usdc)
                            else np.nan
                        )
                    rec["pm_position_size"] = 0.0
                    rec["pm_position_current_value"] = 0.0
                    rec["pm_position_redeemable"] = False
                    rec["pm_settlement_status"] = "closed"
                    if _is_polymarket_submitted_status(rec.get("pm_exit_order_status")):
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
                    elif _is_polymarket_submitted_status(rec.get("pm_order_status")):
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

                rec["pm_position_size"] = (
                    float(pos_size) if np.isfinite(pos_size) else np.nan
                )
                rec["pm_position_current_value"] = (
                    float(current_value) if np.isfinite(current_value) else np.nan
                )
                rec["pm_position_redeemable"] = bool(redeemable)
                if np.isfinite(pos_size):
                    rec["shares_net"] = float(pos_size)
                rec["pm_position_avg_price"] = (
                    float(avg_price) if np.isfinite(avg_price) else np.nan
                )
                if np.isfinite(avg_price):
                    rec["entry_price"] = float(avg_price)
                rec["pm_position_initial_value_usdc"] = (
                    float(initial_value) if np.isfinite(initial_value) else np.nan
                )
                if np.isfinite(initial_value):
                    rec["stake_usdc"] = float(initial_value)
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

    def _market_lookup_future_for_bucket(self, bucket_start):
        bucket_start = pd.Timestamp(bucket_start)

        with self.pm_market_prefetch_lock:
            if self.pm_market_prefetch_bucket_start == bucket_start:
                future = self.pm_market_prefetch_future
                if future is not None:
                    return future
                self._cancel_market_prefetch_timer_locked()

        return self.pm_lookup_pool.submit(
            self._fetch_market_snapshot_with_retry, bucket_start
        )

    def _market_future_timeout_sec(self):
        request_timeout_sec = float(self.pm_cfg.market_request_timeout_sec)
        lookup_wait_sec = max(int(self.pm_cfg.market_lookup_max_wait_ms), 0) / 1000.0
        return max(1.0, lookup_wait_sec + request_timeout_sec * 3.0 + 1.0)

    def _future_result_with_timeout(self, future, label, timeout_sec=None):
        timeout = (
            self._market_future_timeout_sec()
            if timeout_sec is None
            else max(float(timeout_sec), 1.0)
        )
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            raise TimeoutError(f"{label}_timeout_after_{timeout:.1f}s") from exc

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
        snapshot = self._fetch_market_snapshot_with_retry(bucket_start)
        return {
            "bucket_start": pd.Timestamp(bucket_start),
            "snapshot": snapshot,
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
            "best_bid_size": _best_size(payload.get("bids", []), side="bid"),
            "best_ask": _best_price(payload.get("asks", []), side="ask"),
            "best_ask_size": _best_size(payload.get("asks", []), side="ask"),
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

    def _fetch_market_snapshot(self, bucket_start):
        market_slug = self._market_slug_for_bucket(bucket_start)
        market = self._get_json(self.pm_cfg.gamma_host, f"/markets/slug/{market_slug}")

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
            self._fetch_order_book_summary, up_token_id
        )
        down_book_future = self.pm_io_pool.submit(
            self._fetch_order_book_summary, down_token_id
        )
        fee_rate_future = self.pm_io_pool.submit(self._fetch_fee_rate_bps, up_token_id)
        io_timeout_sec = max(float(self.pm_cfg.market_request_timeout_sec) + 1.0, 1.0)
        up_book = self._future_result_with_timeout(
            up_book_future, "up_order_book", timeout_sec=io_timeout_sec
        )
        down_book = self._future_result_with_timeout(
            down_book_future, "down_order_book", timeout_sec=io_timeout_sec
        )
        fee_rate_bps = self._future_result_with_timeout(
            fee_rate_future, "fee_rate", timeout_sec=io_timeout_sec
        )
        fee_model = polymarket_fee_model_from_market(
            market,
            default_round_decimals=DEFAULT_POLYMARKET_FEE_ROUND_DECIMALS,
            default_min_fee=DEFAULT_POLYMARKET_MIN_FEE_USDC,
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
            fee_model=fee_model,
            up_token_id=up_token_id,
            down_token_id=down_token_id,
            up_best_bid=_safe_float(up_book.get("best_bid")),
            up_best_bid_size=_safe_float(up_book.get("best_bid_size")),
            up_best_ask=_safe_float(up_book.get("best_ask")),
            up_best_ask_size=_safe_float(up_book.get("best_ask_size")),
            up_last_trade_price=_safe_float(up_book.get("last_trade_price")),
            down_best_bid=_safe_float(down_book.get("best_bid")),
            down_best_bid_size=_safe_float(down_book.get("best_bid_size")),
            down_best_ask=_safe_float(down_book.get("best_ask")),
            down_best_ask_size=_safe_float(down_book.get("best_ask_size")),
            down_last_trade_price=_safe_float(down_book.get("last_trade_price")),
        )

    def _fetch_market_snapshot_with_retry(self, bucket_start):
        deadline = time.perf_counter() + (
            max(int(self.pm_cfg.market_lookup_max_wait_ms), 0) / 1000.0
        )
        retry_sleep_sec = max(int(self.pm_cfg.market_lookup_retry_ms), 0) / 1000.0
        last_error = None

        while True:
            try:
                return self._fetch_market_snapshot(bucket_start)
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
            time.sleep(retry_sleep_sec)

    def _recommend_polymarket_bet(self, prob_up_raw, market):
        bankroll = float(self.live_bankroll_usdc)
        trade_policy_mode = str(self.live_trade_policy.get("mode", "ev")).lower()
        if bankroll <= 0.0:
            return {
                "decision": "no_trade",
                "trade_side": "none",
                "final_reason": "bankroll_non_positive",
                "policy_reason": "bankroll_non_positive",
            }
        if not market.accepting_orders:
            return {
                "decision": "no_trade",
                "trade_side": "none",
                "final_reason": "market_not_accepting_orders",
                "policy_reason": "market_not_accepting_orders",
            }

        market_end = pd.Timestamp(market.market_end)
        seconds_to_close = float(
            (market_end - pd.Timestamp.now(tz="UTC")).total_seconds()
        )
        if seconds_to_close <= float(self.pm_cfg.no_trade_last_seconds):
            return {
                "decision": "no_trade",
                "trade_side": "none",
                "final_reason": "too_close_to_market_end",
                "policy_reason": "too_close_to_market_end",
                "seconds_to_close": float(seconds_to_close),
            }

        fee_fractions = resolve_fee_fractions_from_quotes(
            ask_yes=float(market.up_best_ask),
            ask_no=float(market.down_best_ask),
            fee_model=market.fee_model,
        )
        if trade_policy_mode == "model_direction_min_stake":
            policy_result = decide_trade_from_model_direction(
                proba_up=float(prob_up_raw),
                threshold=float(self.prediction_threshold),
                ask_yes=float(market.up_best_ask),
                ask_no=float(market.down_best_ask),
                fee_yes=float(fee_fractions["fee_yes"]),
                fee_no=float(fee_fractions["fee_no"]),
                extra_buffer=float(self.live_trade_policy["extra_buffer"]),
                min_decision_margin=float(
                    self.live_trade_policy.get("min_decision_margin", 0.0)
                ),
                min_decision_margin_up=self.live_trade_policy.get(
                    "min_decision_margin_up"
                ),
                min_decision_margin_down=self.live_trade_policy.get(
                    "min_decision_margin_down"
                ),
            )
        else:
            policy_result = decide_trade_from_ev(
                float(prob_up_raw),
                float(market.up_best_ask),
                float(market.down_best_ask),
                float(fee_fractions["fee_yes"]),
                float(fee_fractions["fee_no"]),
                float(self.live_trade_policy["extra_buffer"]),
            )
        intent = build_trade_intent(
            policy_result=policy_result,
            bankroll=float(self.live_bankroll_usdc),
            stake_multiplier=float(self.live_trade_policy["stake_multiplier"]),
            fee_model=market.fee_model,
            order_min_size=float(market.order_min_size),
            external_stake_cap_usdc=float(self.pm_cfg.max_exposure_usdc),
        )

        side = str(intent.get("trade_side", "") or "").lower()
        if side == "yes":
            intent["token_id"] = str(market.up_token_id)
        elif side == "no":
            intent["token_id"] = str(market.down_token_id)

        intent["seconds_to_close"] = float(seconds_to_close)
        intent["tick_size"] = float(market.tick_size)
        intent["neg_risk"] = bool(market.neg_risk)
        intent["order_price_cap"] = float(self.pm_cfg.order_price_cap)
        intent["fee_rate_bps"] = int(market.fee_rate_bps)
        intent["fee_model"] = market.fee_model
        intent["ask_yes"] = float(market.up_best_ask)
        intent["ask_no"] = float(market.down_best_ask)
        if str(intent.get("decision", "")).lower() == "no_trade":
            intent["submitted_price"] = float("nan")
            intent["submitted_price_error"] = ""
            return intent

        submitted_price, submitted_price_error = _resolve_submitted_buy_price(
            entry_price=intent.get("entry_price", np.nan),
            order_price_cap=self.pm_cfg.order_price_cap,
            submitted_price_mode=self.live_trade_policy.get(
                "submitted_price_mode", "entry_price"
            ),
        )
        if not np.isfinite(submitted_price):
            intent["decision"] = "no_trade"
            intent["trade_side"] = "none"
            intent["token_id"] = ""
            intent["bet_usdc"] = 0.0
            intent["final_reason"] = submitted_price_error
            intent["submitted_price"] = float("nan")
            intent["submitted_price_error"] = str(submitted_price_error)
            return intent

        intent["submitted_price"] = float(submitted_price)
        intent["submitted_price_error"] = ""
        return intent

    def _submit_result(
        self,
        *,
        commit_bankroll,
        status,
        error="",
        response_text="",
        filled_stake_usdc=np.nan,
        filled_shares=np.nan,
    ):
        filled_stake_value = _safe_float(filled_stake_usdc)
        filled_shares_value = _safe_float(filled_shares)
        return {
            "commit_bankroll": bool(commit_bankroll),
            "status": str(status),
            "error": str(error),
            "response_text": str(response_text),
            "filled_stake_usdc": (
                float(filled_stake_value)
                if np.isfinite(filled_stake_value)
                else np.nan
            ),
            "filled_shares": (
                float(filled_shares_value)
                if np.isfinite(filled_shares_value)
                else np.nan
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

    def _create_and_post_market_order_with_retry(self, order, options, order_type):
        if self.pm_client is None:
            raise RuntimeError("pm_client_not_initialized")

        delay_sec = float(POLYMARKET_POST_ORDER_RETRY_INITIAL_DELAY_SEC)
        max_attempts = max(int(POLYMARKET_POST_ORDER_MAX_RETRIES), 0) + 1

        for attempt_idx in range(max_attempts):
            try:
                return self.pm_client.create_and_post_market_order(
                    order_args=order,
                    options=options,
                    order_type=order_type,
                )
            except Exception as exc:
                status_code = _http_status_code(exc)
                is_retryable = (
                    status_code in POLYMARKET_POST_ORDER_RETRYABLE_STATUS_CODES
                )
                if not is_retryable or attempt_idx >= max_attempts - 1:
                    raise
                print(
                    "[pm] create_and_post_market_order retry | "
                    f"status_code={status_code} "
                    f"delay_sec={delay_sec:.1f} "
                    f"attempt={attempt_idx + 1}/{max_attempts - 1}"
                )
                time.sleep(delay_sec)
                delay_sec = min(
                    delay_sec * 2.0,
                    float(POLYMARKET_POST_ORDER_RETRY_MAX_DELAY_SEC),
                )

    def _maybe_submit_order(self, intent):
        try:
            if intent.get("final_reason") != "ok":
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
            order_type = _polymarket_order_type_for_execution_mode(execution_mode)
            success_status = _polymarket_submitted_status_for_execution_mode(
                execution_mode
            )
            order_options = _partial_create_order_options(
                intent.get("tick_size"), intent.get("neg_risk")
            )
            if execution_mode in POLYMARKET_EXECUTION_ORDER_TYPES:
                self._prime_pm_client_order_metadata(intent)
                order = MarketOrderArgs(
                    token_id=str(intent["token_id"]),
                    amount=float(intent["bet_usdc"]),
                    side=BUY,
                    price=float(
                        intent.get(
                            "submitted_price",
                            intent.get("entry_price", self.pm_cfg.order_price_cap),
                        )
                    ),
                    order_type=order_type,
                    user_usdc_balance=float(self.pm_cash_balance_usdc),
                )
                response = self._create_and_post_market_order_with_retry(
                    order, order_options, order_type
                )
            else:
                raise NotImplementedError(
                    "Unsupported live.polymarket_execution_mode: "
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
                    response_text=response_txt,
                )
            filled_shares, filled_stake_usdc = _extract_buy_fill_metrics_from_response(
                response
            )
            commit_bankroll = execution_mode == "fok" or (
                np.isfinite(filled_stake_usdc) and float(filled_stake_usdc) > 0.0
            )
            return self._submit_result(
                commit_bankroll=commit_bankroll,
                status=success_status,
                response_text=response_txt,
                filled_stake_usdc=filled_stake_usdc,
                filled_shares=filled_shares,
            )
        except Exception as exc:
            return self._submit_result(
                commit_bankroll=False,
                status=_submission_error_status_from_exception(exc),
                error=str(exc),
            )

    def _resolve_pending(self):
        if not self.records:
            return 0

        with self.records_lock:
            pending_records = [dict(rec) for rec in self.records]
        self._refresh_polymarket_markets(pending_records)

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

                if _is_polymarket_submitted_status(rec.get("pm_order_status")):
                    rec["pm_settlement_status"] = "resolved_waiting_settlement"
                else:
                    rec["pm_settlement_status"] = (
                        rec.get("pm_settlement_status") or "resolved_no_position"
                    )
                resolved_now += 1

        return resolved_now

    def _resolve_market_snapshot(self, bucket_start, market_future):
        metadata = {
            "market_prefetch_hit": False,
            "market_prefetch_age_ms": np.nan,
            "market_lookup_source": "future_snapshot",
        }
        try:
            result = self._future_result_with_timeout(
                market_future, "market_snapshot"
            )
            if self._prefetched_market_payload_is_fresh(bucket_start, result):
                fetched_at = result.get("fetched_at") if isinstance(result, dict) else None
                metadata["market_prefetch_hit"] = True
                metadata["market_prefetch_age_ms"] = _delay_ms_between(
                    fetched_at,
                    _utc_now(),
                )
                metadata["market_lookup_source"] = "prefetched_snapshot"
                return result["snapshot"], metadata
            if isinstance(result, dict):
                metadata["market_lookup_source"] = "stale_prefetch_refetch"
                return self._fetch_market_snapshot_with_retry(bucket_start), metadata
            return result, metadata
        except Exception:
            metadata["market_lookup_source"] = "future_error_refetch"
            return self._fetch_market_snapshot_with_retry(bucket_start), metadata

    def _evaluate_prediction_execution(
        self,
        *,
        bucket_start,
        proba_up,
        market_future,
    ):
        execution_started_perf = time.perf_counter()
        market_lookup_started_perf = execution_started_perf
        market = None
        intent = {"reason": "not_evaluated"}
        submit_result = self._submit_result(
            commit_bankroll=False,
            status="not_attempted",
        )
        market_lookup_ms = np.nan
        submit_order_ms = np.nan
        policy_compute_ms = np.nan
        market_prefetch_hit = False
        market_prefetch_age_ms = np.nan
        market_lookup_source = ""

        try:
            market, market_meta = self._resolve_market_snapshot(bucket_start, market_future)
            market_lookup_ms = _elapsed_ms(market_lookup_started_perf)
            market_prefetch_hit = bool(market_meta.get("market_prefetch_hit", False))
            market_prefetch_age_ms = float(
                market_meta.get("market_prefetch_age_ms", np.nan)
            )
            market_lookup_source = str(market_meta.get("market_lookup_source", ""))
            policy_started_perf = time.perf_counter()
            intent = self._recommend_polymarket_bet(prob_up_raw=proba_up, market=market)
            policy_compute_ms = _elapsed_ms(policy_started_perf)
            submit_started_perf = time.perf_counter()
            submit_result = self._maybe_submit_order(intent)
        except Exception as exc:
            if not np.isfinite(market_lookup_ms):
                market_lookup_ms = _elapsed_ms(market_lookup_started_perf)
            intent = {"reason": "market_lookup_failed"}
            submit_result = self._submit_result(
                commit_bankroll=False,
                status="market_lookup_failed",
                error=str(exc),
            )
        else:
            submit_order_ms = _elapsed_ms(submit_started_perf)

        return {
            "market": market,
            "intent": intent,
            "submit_result": submit_result,
            "policy_compute_ms": float(policy_compute_ms),
            "market_prefetch_hit": bool(market_prefetch_hit),
            "market_prefetch_age_ms": float(market_prefetch_age_ms),
            "market_lookup_source": str(market_lookup_source),
            "market_lookup_ms": float(market_lookup_ms),
            "submit_order_ms": float(submit_order_ms),
            "execution_ms": _elapsed_ms(execution_started_perf),
        }

    def _build_prediction_record(
        self,
        *,
        bucket_start,
        bucket_end,
        proba_up,
        bankroll_before_entry,
        bankroll_after_entry,
        stake_usdc,
        market,
        intent,
        submit_result,
        decision_delay_ms,
        latency_metrics,
    ):
        order_status = str(submit_result["status"])
        buy_record_fields = _resolve_buy_record_fields(intent, submit_result)
        btc_snapshot = self._latest_btc_snapshot()

        record = {
            "record_id": f"bucket:{pd.Timestamp(bucket_start).isoformat()}",
            "pm_model_hash": self.model_hash,
            "pm_policy_hash": self.trade_policy_config_hash,
            "pm_run_started_at_utc": self.run_started_at_utc,
            "prediction_time": _utc_now(),
            "bucket_start": bucket_start,
            "bucket_end": bucket_end,
            "proba_up": proba_up,
            "model_side": resolve_model_side_from_proba(
                proba_up,
                threshold=self.prediction_threshold,
            ),
            "trade_side": str(intent.get("trade_side", "none")),
            "stake_usdc": float(buy_record_fields["stake_usdc"]),
            "stake_multiplier": float(intent.get("stake_multiplier", np.nan)),
            "required_stake_usdc": float(intent.get("required_stake_usdc", np.nan)),
            "effective_stake_usdc": float(
                intent.get("effective_stake_usdc", np.nan)
            ),
            "entry_price": float(buy_record_fields["entry_price"]),
            "entry_fee_usdc": float(buy_record_fields["entry_fee_usdc"]),
            "entry_fee_raw_usdc": float(buy_record_fields["entry_fee_raw_usdc"]),
            "shares_net": float(buy_record_fields["shares_net"]),
            "entry_stake_usdc_orig": float(
                buy_record_fields["entry_stake_usdc_orig"]
            ),
            "entry_price_orig": float(buy_record_fields["entry_price_orig"]),
            "entry_fee_usdc_orig": float(buy_record_fields["entry_fee_usdc_orig"]),
            "entry_fee_raw_usdc_orig": float(
                buy_record_fields["entry_fee_raw_usdc_orig"]
            ),
            "entry_shares_net_orig": float(
                buy_record_fields["entry_shares_net_orig"]
            ),
            "price_eps": np.nan,
            "price_slip": np.nan,
            "ask_yes": np.nan if market is None else float(market.up_best_ask),
            "ask_no": np.nan if market is None else float(market.down_best_ask),
            "policy_proba_up": float(intent.get("proba_up", np.nan)),
            "policy_ask_yes": float(intent.get("ask_yes", np.nan)),
            "policy_ask_no": float(intent.get("ask_no", np.nan)),
            "policy_fee_yes": float(intent.get("fee_yes", np.nan)),
            "policy_fee_no": float(intent.get("fee_no", np.nan)),
            "policy_extra_buffer": float(intent.get("extra_buffer", np.nan)),
            "policy_ev_yes": float(intent.get("ev_yes", np.nan)),
            "policy_ev_no": float(intent.get("ev_no", np.nan)),
            "policy_best_ev": float(intent.get("best_ev", np.nan)),
            "policy_decision": str(intent.get("decision", "no_trade")),
            "policy_reason": str(
                intent.get("final_reason") or intent.get("reason", "")
            ),
            "bankroll_before_entry": float(bankroll_before_entry),
            "bankroll_after_entry": float(bankroll_after_entry),
            "bankroll_before_entry_orig": float(bankroll_before_entry),
            "bankroll_after_entry_orig": float(bankroll_after_entry),
            "bankroll_after_resolve": None,
            "trade_is_win": None,
            "payout_usdc": None,
            "pnl_usdc": None,
            "btc_open": float(btc_snapshot["btc_open"]),
            "btc_high": float(btc_snapshot["btc_high"]),
            "btc_low": float(btc_snapshot["btc_low"]),
            "btc_close": float(btc_snapshot["btc_close"]),
            "btc_volume": float(btc_snapshot["btc_volume"]),
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
            "pm_fee_source": (
                ""
                if market is None
                else str(getattr(market, "fee_model", {}).get("source", ""))
            ),
            "pm_fee_rate": (
                np.nan
                if market is None
                else float(getattr(market, "fee_model", {}).get("rate", np.nan))
            ),
            "pm_fee_exponent": (
                np.nan
                if market is None
                else float(getattr(market, "fee_model", {}).get("exponent", np.nan))
            ),
            "pm_fee_round_decimals": (
                np.nan
                if market is None
                else int(
                    getattr(market, "fee_model", {}).get(
                        "fee_round_decimals",
                        DEFAULT_POLYMARKET_FEE_ROUND_DECIMALS,
                    )
                )
            ),
            "pm_min_fee_usdc": (
                np.nan
                if market is None
                else float(
                    getattr(market, "fee_model", {}).get(
                        "min_fee",
                        DEFAULT_POLYMARKET_MIN_FEE_USDC,
                    )
                )
            ),
            "pm_tick_size": np.nan if market is None else float(market.tick_size),
            "pm_order_min_size": (
                np.nan if market is None else float(market.order_min_size)
            ),
            "pm_order_price_cap": float(
                intent.get("order_price_cap", self.pm_cfg.order_price_cap)
            ),
            "pm_position_size": np.nan,
            "pm_position_current_value": np.nan,
            "pm_position_redeemable": False,
            "pm_position_avg_price": np.nan,
            "pm_position_initial_value_usdc": np.nan,
            "pm_closed_avg_price": np.nan,
            "pm_closed_total_bought_usdc": np.nan,
            "pm_closed_realized_pnl_usdc": np.nan,
            "pm_closed_payout_usdc": np.nan,
            "pm_settlement_status": (
                "entry_submitted" if _is_polymarket_submitted_status(order_status) else ""
            ),
            "pm_account_sync_at": self.pm_last_account_sync_at,
            "pm_account_sync_reason": self.pm_last_account_sync_reason,
            "pm_account_cash_balance_usdc": float(self.pm_cash_balance_usdc)
            if np.isfinite(self.pm_cash_balance_usdc)
            else np.nan,
            "pm_account_positions_value_usdc": float(self.pm_positions_value_usdc)
            if np.isfinite(self.pm_positions_value_usdc)
            else np.nan,
            "pm_account_sync_at_entry": self.pm_last_account_sync_at,
            "pm_account_cash_balance_entry_usdc": float(self.pm_cash_balance_usdc)
            if np.isfinite(self.pm_cash_balance_usdc)
            else np.nan,
            "pm_account_positions_value_entry_usdc": float(self.pm_positions_value_usdc)
            if np.isfinite(self.pm_positions_value_usdc)
            else np.nan,
            "pm_account_sync_at_resolve": None,
            "pm_account_cash_balance_resolve_usdc": np.nan,
            "pm_account_positions_value_resolve_usdc": np.nan,
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
            "decision_delay_ms": float(decision_delay_ms),
            "pm_order_error": str(submit_result["error"]),
            "pm_order_response": str(submit_result.get("response_text", "")),
            "pm_exit_order_status": "",
            "pm_exit_order_error": "",
            "pm_exit_reason": "",
            "pm_exit_decision_at": None,
            "pm_exit_best_bid": np.nan,
            "pm_exit_seconds_to_close": np.nan,
            "pm_exit_candidate_pnl_usdc": np.nan,
            "pm_exit_candidate_roi": np.nan,
            "pm_exit_redeem_pnl_usdc": np.nan,
            "pm_exit_min_allowed_pnl_usdc": np.nan,
            "pm_exit_price": np.nan,
            "pm_exit_shares": np.nan,
            "pm_exit_fee_usdc": np.nan,
            "pm_exit_proceeds_usdc": np.nan,
            "pm_exit_order_response": "",
        }
        record.update(latency_metrics)
        return record

    def _build_prediction_summary(
        self,
        *,
        minute_close,
        bucket_start,
        bucket_end,
        proba_up,
        stake_usdc,
        bankroll_before_entry,
        bankroll_after_entry,
        intent,
        submit_result,
        decision_delay_ms,
        latency_metrics,
    ):
        summary = {
            "decision_local": minute_close.tz_convert(self.local_tz).isoformat(),
            "bucket_start": bucket_start,
            "bucket_end": bucket_end,
            "proba_up": proba_up,
            "model_side": resolve_model_side_from_proba(
                proba_up,
                threshold=self.prediction_threshold,
            ),
            "trade_side": str(intent.get("trade_side", "none")),
            "stake_usdc": float(stake_usdc),
            "stake_multiplier": float(intent.get("stake_multiplier", np.nan)),
            "required_stake_usdc": float(intent.get("required_stake_usdc", np.nan)),
            "effective_stake_usdc": float(
                intent.get("effective_stake_usdc", np.nan)
            ),
            "bankroll_before_entry": float(bankroll_before_entry),
            "bankroll_after_entry": float(bankroll_after_entry),
            "policy_decision": str(intent.get("decision", "no_trade")),
            "policy_reason": str(intent.get("final_reason") or intent.get("reason", "")),
            "policy_ev_yes": float(intent.get("ev_yes", np.nan)),
            "policy_ev_no": float(intent.get("ev_no", np.nan)),
            "policy_best_ev": float(intent.get("best_ev", np.nan)),
            "pm_order_status": str(submit_result["status"]),
            "pm_order_error": str(submit_result["error"]),
            "decision_delay_ms": float(decision_delay_ms),
        }
        summary.update(latency_metrics)
        return summary

    def _predict_next_bucket(self, volume_profile_values=None, delay_timing=None):
        minute_open = self.opened_candles[-1]
        minute_close = minute_open + pd.Timedelta(minutes=1)
        bucket_start = self._bucket_start_for_latest_candle()
        bucket_end = bucket_start + pd.Timedelta(minutes=self.target_bucket_minutes - 1)
        print(
            "[pred] starting | "
            f"bucket_start={bucket_start.isoformat()} "
            f"minute_open={minute_open.isoformat()}",
            flush=True,
        )
        market_future = self._market_lookup_future_for_bucket(bucket_start)
        latency_metrics = {}
        if delay_timing is not None:
            for key in (
                "ws_price_event_delay_ms",
                "ws_volume_event_delay_ms",
                "ws_price_receive_delay_ms",
                "ws_volume_receive_delay_ms",
                "ws_event_delay_ms",
                "ws_receive_delay_ms",
                "ws_component_sync_ms",
                "feature_prep_ms",
            ):
                if key in delay_timing:
                    latency_metrics[key] = delay_timing[key]
        if volume_profile_values is None:
            feature_prep_started_perf = time.perf_counter()
            volume_profile_values = self._prepare_volume_profile_features_for_latest_candle(
                minute_open
            )
            latency_metrics["feature_prep_ms"] = _elapsed_ms(feature_prep_started_perf)
        else:
            latency_metrics.setdefault("feature_prep_ms", np.nan)

        feature_vector_started_perf = time.perf_counter()
        feature_vector = self._build_feature_vector(
            volume_profile_values=volume_profile_values
        )
        latency_metrics["feature_vector_ms"] = _elapsed_ms(feature_vector_started_perf)
        model_predict_started_perf = time.perf_counter()
        proba_up = float(self.model.predict(feature_vector)[0])
        latency_metrics["model_predict_ms"] = _elapsed_ms(model_predict_started_perf)
        bankroll_before_entry = float(self.live_bankroll_usdc)
        execution = self._evaluate_prediction_execution(
            bucket_start=bucket_start,
            proba_up=proba_up,
            market_future=market_future,
        )
        decision_delay_ms = _delay_ms_since(bucket_start)
        latency_metrics.setdefault("ws_price_event_delay_ms", np.nan)
        latency_metrics.setdefault("ws_volume_event_delay_ms", np.nan)
        latency_metrics.setdefault("ws_price_receive_delay_ms", np.nan)
        latency_metrics.setdefault("ws_volume_receive_delay_ms", np.nan)
        latency_metrics.setdefault("ws_event_delay_ms", np.nan)
        latency_metrics.setdefault("ws_receive_delay_ms", np.nan)
        latency_metrics.setdefault("ws_component_sync_ms", np.nan)
        latency_metrics["policy_compute_ms"] = float(execution["policy_compute_ms"])
        latency_metrics["market_prefetch_hit"] = bool(execution["market_prefetch_hit"])
        latency_metrics["market_prefetch_age_ms"] = float(
            execution["market_prefetch_age_ms"]
        )
        latency_metrics["market_lookup_source"] = str(execution["market_lookup_source"])
        latency_metrics["market_lookup_ms"] = float(execution["market_lookup_ms"])
        latency_metrics["submit_order_ms"] = float(execution["submit_order_ms"])
        latency_metrics["execution_ms"] = float(execution["execution_ms"])
        intent = execution["intent"]
        submit_result = execution["submit_result"]
        filled_stake_usdc = _safe_float(submit_result.get("filled_stake_usdc"))
        if np.isfinite(filled_stake_usdc):
            stake_usdc = float(filled_stake_usdc)
        else:
            stake_usdc = (
                float(intent.get("bet_usdc", 0.0) or 0.0)
                if bool(submit_result["commit_bankroll"])
                and str(intent.get("final_reason", "")) == "ok"
                else 0.0
            )
        if self.pm_cfg.paper_mode and stake_usdc > 0.0:
            self.live_bankroll_usdc -= stake_usdc
        bankroll_after_entry = float(self.live_bankroll_usdc)

        record = self._build_prediction_record(
            bucket_start=bucket_start,
            bucket_end=bucket_end,
            proba_up=proba_up,
            bankroll_before_entry=bankroll_before_entry,
            bankroll_after_entry=bankroll_after_entry,
            stake_usdc=stake_usdc,
            market=execution["market"],
            intent=intent,
            submit_result=submit_result,
            decision_delay_ms=decision_delay_ms,
            latency_metrics=latency_metrics,
        )
        with self.records_lock:
            self.records.append(record)
        self.predicted_buckets.add(bucket_start)

        return self._build_prediction_summary(
            minute_close=minute_close,
            bucket_start=bucket_start,
            bucket_end=bucket_end,
            proba_up=proba_up,
            stake_usdc=stake_usdc,
            bankroll_before_entry=bankroll_before_entry,
            bankroll_after_entry=bankroll_after_entry,
            intent=intent,
            submit_result=submit_result,
            decision_delay_ms=decision_delay_ms,
            latency_metrics=latency_metrics,
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

    def _format_rate(self, value):
        return "n/a" if not np.isfinite(value) else f"{value * 100:.2f}%"

    def _stats(self):
        records = self._records_snapshot()
        model_resolved = 0
        model_wins = 0
        for rec in records:
            model_accuracy = resolve_model_accuracy_from_proba(
                rec,
                threshold=self.prediction_threshold,
            )
            if model_accuracy is None:
                continue
            model_resolved += 1
            model_wins += int(model_accuracy)
        policy_resolved = sum(1 for rec in records if self._record_is_resolved(rec))
        policy_resolved_wins = sum(
            int(rec["is_correct"]) for rec in records if self._record_is_resolved(rec)
        )
        closed_trades = sum(1 for rec in records if self._record_is_traded(rec))
        closed_trade_wins = sum(
            int(rec["trade_is_win"]) for rec in records if self._record_is_traded(rec)
        )
        win_rate_policy_resolved = (
            float(policy_resolved_wins / policy_resolved)
            if policy_resolved
            else float("nan")
        )
        win_rate_model = (
            float(model_wins / model_resolved) if model_resolved else float("nan")
        )
        win_rate_closed_trade = (
            float(closed_trade_wins / closed_trades)
            if closed_trades
            else float("nan")
        )
        total_pnl = float(
            sum(
                float(rec.get("pnl_usdc", 0.0) or 0.0)
                for rec in records
                if rec["actual_up"] is not None
            )
        )
        return {
            "model_resolved": model_resolved,
            "model_wins": model_wins,
            "model_losses": model_resolved - model_wins,
            "win_rate_model": win_rate_model,
            "policy_resolved": policy_resolved,
            "policy_resolved_wins": policy_resolved_wins,
            "policy_resolved_losses": policy_resolved - policy_resolved_wins,
            "win_rate_policy_resolved": win_rate_policy_resolved,
            "closed_trades": closed_trades,
            "closed_trade_wins": closed_trade_wins,
            "closed_trade_losses": closed_trades - closed_trade_wins,
            "win_rate_closed_trade": win_rate_closed_trade,
            "total_pnl": total_pnl,
        }

    def _save_records(self):
        with self.pm_save_lock:
            records = self._records_snapshot()
            if not records:
                return
            snapshot_at = _utc_now()
            for rec in records:
                rec["record_id"] = _stable_record_id(rec)
                rec["record_snapshot_at"] = snapshot_at
                self._backfill_bucket_price_bounds(rec)
                _backfill_record_analysis_fields(rec)
            write_records_csv(
                records,
                self.trade_records_path,
                export_columns=LIVE_TRADE_EXPORT_COLUMNS,
                is_resolved=self._record_is_resolved,
                is_traded=self._record_is_traded,
            )
            upsert_records_csv(
                records,
                self.market_data_path,
                export_columns=LIVE_SHARED_MARKET_DATA_COLUMNS,
                is_resolved=self._record_is_resolved,
                is_traded=self._record_is_traded,
                record_filter=lambda rec: str(rec.get("record_id", "")).startswith(
                    "bucket:"
                ),
            )
            write_records_state(records, self.trade_records_state_path)

    @staticmethod
    def _print_log_fields(section, fields):
        rendered = []
        for key, value in fields:
            if value is None:
                continue
            value_txt = str(value).replace("\n", " ").strip()
            if not value_txt:
                continue
            rendered.append(f"{key}={value_txt}")
        if rendered:
            print(f"  {section:<10}| " + " | ".join(rendered))

    def _print_indicator_nan_status(self):
        if not self.last_indicator_nan_cols:
            return
        cols = ", ".join(self.last_indicator_nan_cols)
        self._print_log_fields(
            "indicators",
            [
                ("latest_nan_count", len(self.last_indicator_nan_cols)),
                ("cols", cols),
            ],
        )

    def _log(self, tag, pred=None):
        stats = self._stats()
        model_win_rate_txt = self._format_rate(stats["win_rate_model"])
        policy_resolved_win_rate_txt = self._format_rate(
            stats["win_rate_policy_resolved"]
        )
        closed_trade_win_rate_txt = self._format_rate(
            stats["win_rate_closed_trade"]
        )
        ts = (
            str(pred["decision_local"])
            if pred is not None
            else pd.Timestamp.now(tz="UTC").tz_convert(self.local_tz).isoformat()
        )

        print(f"[{tag}] {ts}")
        self._print_log_fields(
            "model",
            [
                ("resolved", stats["model_resolved"]),
                ("wins", stats["model_wins"]),
                ("losses", stats["model_losses"]),
                ("win_rate", model_win_rate_txt),
            ],
        )
        self._print_log_fields(
            "resolved",
            [
                ("policy_resolved", stats["policy_resolved"]),
                ("wins", stats["policy_resolved_wins"]),
                ("losses", stats["policy_resolved_losses"]),
                ("win_rate", policy_resolved_win_rate_txt),
            ],
        )
        self._print_log_fields(
            "trades",
            [
                ("closed", stats["closed_trades"]),
                ("wins", stats["closed_trade_wins"]),
                ("losses", stats["closed_trade_losses"]),
                ("win_rate", closed_trade_win_rate_txt),
            ],
        )
        if pred is not None:
            self._print_log_fields(
                "decision",
                [
                    ("proba_up", f"{pred['proba_up']:.6f}"),
                    ("model_side", pred.get("model_side", "none")),
                    ("policy_decision", pred["policy_decision"]),
                    ("trade_side", pred["trade_side"]),
                    ("policy_best_ev", f"{pred['policy_best_ev']:.6f}"),
                    ("stake_usdc", f"{pred['stake_usdc']:.2f}"),
                    ("stake_multiplier", f"{pred['stake_multiplier']:.4f}"),
                    (
                        "required_stake_usdc",
                        (
                            f"{pred['required_stake_usdc']:.2f}"
                            if np.isfinite(
                                _safe_float(pred.get("required_stake_usdc", np.nan))
                            )
                            else None
                        ),
                    ),
                    ("effective_stake_usdc", f"{pred['effective_stake_usdc']:.2f}"),
                ],
            )
            execution_fields = [
                ("pm_order_status", pred["pm_order_status"]),
                ("policy_reason", pred["policy_reason"]),
                ("decision_delay_ms", f"{pred['decision_delay_ms']:.0f}"),
            ]
            if np.isfinite(_safe_float(pred.get("ws_receive_delay_ms", np.nan))):
                execution_fields.append(
                    ("ws_receive_delay_ms", f"{pred['ws_receive_delay_ms']:.0f}")
                )
            if np.isfinite(_safe_float(pred.get("ws_component_sync_ms", np.nan))):
                execution_fields.append(
                    ("ws_component_sync_ms", f"{pred['ws_component_sync_ms']:.0f}")
                )
            if np.isfinite(_safe_float(pred.get("feature_prep_ms", np.nan))):
                execution_fields.append(
                    ("feature_prep_ms", f"{pred['feature_prep_ms']:.0f}")
                )
            if np.isfinite(_safe_float(pred.get("model_predict_ms", np.nan))):
                execution_fields.append(
                    ("model_predict_ms", f"{pred['model_predict_ms']:.0f}")
                )
            if np.isfinite(_safe_float(pred.get("market_lookup_ms", np.nan))):
                execution_fields.append(
                    ("market_lookup_ms", f"{pred['market_lookup_ms']:.0f}")
                )
            if np.isfinite(_safe_float(pred.get("submit_order_ms", np.nan))):
                execution_fields.append(
                    ("submit_order_ms", f"{pred['submit_order_ms']:.0f}")
                )
            self._print_log_fields("execution", execution_fields)
            if pred.get("pm_order_error"):
                self._print_log_fields(
                    "error",
                    [("pm_order_error", pred["pm_order_error"])],
                )
        account_fields = []
        account_fields.append(("total_pnl", f"{stats['total_pnl']:.2f}"))
        account_fields.append(("bankroll", f"{self.live_bankroll_usdc:.2f}"))
        if np.isfinite(self.pm_cash_balance_usdc):
            account_fields.append(("cash_balance", f"{self.pm_cash_balance_usdc:.2f}"))
        if np.isfinite(self.pm_positions_value_usdc):
            account_fields.append(
                ("positions_value", f"{self.pm_positions_value_usdc:.2f}")
            )
        self._print_log_fields("account", account_fields)
        if pred is not None:
            self._print_indicator_nan_status()
        print()

    def _maybe_sync_missing_candles(self, opened_from_ws):
        if not self.opened_candles:
            return

        expected_next = self.opened_candles[-1] + INTERVAL_DELTA
        if opened_from_ws > expected_next:
            self._sync_closed_candles_from_rest(stop_before_opened=opened_from_ws)

    def _maybe_predict_closed_bucket(
        self,
        opened,
        volume_profile_values,
        *,
        delay_timing=None,
    ):
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
            delay_timing=delay_timing,
        )

    def _schedule_post_cycle_syncs(self, pred, resolved_now):
        if pred is not None and _is_polymarket_submitted_status(
            pred.get("pm_order_status")
        ):
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
                "external writes disabled via "
                "live.polymarket_disable_order_submission=true "
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
        if self._relayer_is_configured() and self.pm_cfg.relayer_tx_type == "SAFE":
            print(
                "Auto-redeem background mode is enabled | "
                f"relayer_host={self.pm_cfg.relayer_host} "
                f"relayer_api_key_address={self.pm_relayer_api_key_address} "
                f"relayer_tx_type={self.pm_cfg.relayer_tx_type} "
                f"collateralToken={self.pm_cfg.redeem_collateral_token_address} "
                f"ctfAddress={self.pm_cfg.redeem_ctf_address} "
                f"targetAddress={self.pm_cfg.redeem_target_address} "
                f"requireRedeemable={self.pm_cfg.redeem_require_redeemable}"
            )
        elif self._relayer_is_configured():
            self._warn_relayer_unavailable_once(
                "auto-redeem submission currently supports POLY_RELAYER_TX_TYPE=SAFE; "
                f"got {self.pm_cfg.relayer_tx_type}"
            )
        else:
            self._warn_relayer_unavailable_once(
                "set relayer secrets in .env: POLY_RELAYER_API_KEY and optionally "
                "POLY_RELAYER_API_KEY_ADDRESS"
            )

    def _print_runtime_configuration(self):
        print(
            "Polymarket execution | "
            f"mode={'paper' if self.pm_cfg.paper_mode else 'live'} "
            f"price_symbol={SYMBOL} price_market={PRICE_MARKET} "
            f"volume_symbol={VOLUME_SYMBOL} volume_market={VOLUME_MARKET} "
            f"price_source={PRICE_SOURCE} volume_source={VOLUME_SOURCE} "
            f"series_slug={self.pm_cfg.series_slug} "
            f"market_slug_prefix={self.pm_cfg.market_slug_prefix} "
            f"settlement_source={self.settlement_source} "
            f"execution_mode={self.pm_cfg.execution_mode} "
            f"order_price_cap={self.pm_cfg.order_price_cap:.3f} "
            f"records={self.trade_records_path}"
        )
        print(
            "Websocket targets | "
            + ", ".join(
                f"{target['label']}->{target['url']}" for target in WS_TARGETS
            )
        )
        if self.pm_cfg.paper_mode:
            print(
                "Paper mode bankroll source | "
                f"live.polymarket_start_bankroll_usdc={self.pm_cfg.start_bankroll_usdc:.2f} "
                f"bankroll_cap_usdc={self.pm_cfg.max_bankroll_usdc}"
            )
        else:
            self._print_live_runtime_configuration()

        print(
            "Trade policy | "
            f"bankroll={self.live_bankroll_usdc:.2f} "
            f"mode={self.live_trade_policy.get('mode', 'ev')} "
            f"stake_multiplier={self.live_trade_policy['stake_multiplier']:.4f} "
            f"extra_buffer={self.live_trade_policy['extra_buffer']:.6f} "
            f"submitted_price_mode={self.live_trade_policy.get('submitted_price_mode', 'entry_price')}"
        )
        print(
            "Exit policy | "
            f"enable_exit_orders={self.pm_cfg.enable_exit_orders} "
            f"exit_min_profit_usdc={self.pm_cfg.exit_min_profit_usdc:.4f} "
            f"exit_min_roi={self.pm_cfg.exit_min_roi:.4f} "
            f"exit_redeem_profit_tolerance={self.pm_cfg.exit_redeem_profit_tolerance:.4f}"
        )
        print(f"Trade policy hash: {self.trade_policy_config_hash}")
        print(f"Bankroll source: {self.bankroll_source}")
        print(f"Records file: {self.trade_records_path}")
        print(f"Shared market data file: {self.market_data_path}")

    def _on_message(self, _ws, message):
        with self.ws_message_lock:
            try:
                # Keep completed background-sync bookkeeping, but do not start any new
                # network work before the current predict->submit cycle finishes.
                self._poll_background_sync(reschedule_pending=False)
                payload = _load_ws_payload(message)
                closed_candle, live_minute_opened, _event_at, ws_timing = (
                    self._consume_ws_payload(payload)
                )
                if closed_candle is None:
                    return

                opened_from_ws = pd.to_datetime(
                    int(closed_candle["t"]), unit="ms", utc=True
                )
                self._maybe_sync_missing_candles(opened_from_ws)

                opened = self._upsert_closed_candle(closed_candle)
                if opened is None:
                    return
                if (
                    self.last_processed_closed_opened is not None
                    and opened <= self.last_processed_closed_opened
                ):
                    return
                if PRICE_SOURCE == "index":
                    self._maybe_sync_missing_candles(live_minute_opened)

                feature_prep_started_perf = time.perf_counter()
                volume_profile_values = self._prepare_volume_profile_features_for_latest_candle(
                    opened
                )
                delay_timing = {} if ws_timing is None else dict(ws_timing)
                delay_timing["feature_prep_ms"] = _elapsed_ms(feature_prep_started_perf)
                pred = self._maybe_predict_closed_bucket(
                    opened,
                    volume_profile_values,
                    delay_timing=delay_timing,
                )
                resolved_now = self._resolve_pending()

                self.last_processed_closed_opened = opened
                self._schedule_market_snapshot_prefetch(
                    self._next_unpredicted_bucket_start()
                )

                self._schedule_post_cycle_syncs(pred, resolved_now)
                self._persist_cycle_results(pred, resolved_now)

                self._poll_background_sync()
            except Exception as exc:
                print(f"[pred] message handling failed: {exc}")

    def run_forever(self):
        self._print_runtime_configuration()
        now_utc = pd.Timestamp.now(tz="UTC")
        next_bucket_start = now_utc.floor(
            f"{self.target_bucket_minutes}min"
        ) + pd.Timedelta(minutes=self.target_bucket_minutes)
        while next_bucket_start in self.predicted_buckets:
            next_bucket_start += pd.Timedelta(minutes=self.target_bucket_minutes)
        self._schedule_market_snapshot_prefetch(next_bucket_start)
        self._schedule_background_sync("startup", force=True)

        self._run_all_websocket_targets_forever()


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
