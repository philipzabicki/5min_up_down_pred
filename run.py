import hashlib
import json
import math
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime
from datetime import datetime as std_datetime, timedelta as std_timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from websocket import WebSocketApp

from create_modeling_dataset import (
    parse_fit_results,
    resolve_volume_profile_modeling_state_path,
)
from features.ADX import get_adx_values
from features.BollingerBands import get_bollinger_bands_values
from features.ChaikinOsc import get_chaikin_oscillator_values
from features.KeltnerChannel import get_keltner_channel_values
from features.MACD import get_macd_values
from features.StochOsc import get_stochastic_oscillator_values
from features.basis_premium_features import (
    add_basis_premium_features,
    is_basis_premium_feature,
    resolve_futures_close_col,
    validate_basis_premium_feature_columns,
)
from features.candle_features import (
    RAW_OHLCV_COLS,
    STREAK_FEATURE_PREFIX,
    SUPPORTED_CANDLE_FEATURE_COLS,
    build_latest_candle_derived_feature_dict_fast,
    build_latest_candle_pattern_feature_dict_fast,
    build_latest_candle_streak_feature_dict_fast,
    resolve_candle_derived_feature_cols,
    resolve_candle_pattern_feature_cols,
    resolve_streak_interval_to_rule,
)
from features.feature_intervals import FEATURE_INTERVAL_TO_RULE
from features.live_indicator_runtime import (
    LATEST_VALUE_BUILDERS as LIVE_LATEST_VALUE_BUILDERS,
    IndicatorFullHistoryScratch,
    IndicatorWindowScratch,
)
from features.realized_volatility import (
    REALIZED_VOLATILITY_FEATURE_COLUMNS,
    RealizedVolatilityRuntimeState,
)
from features.session_open_features import (
    SUPPORTED_SESSION_OPEN_FEATURE_COLS,
    build_latest_session_open_feature_dict_fast,
)
from features.volume_profile_fixed_range import (
    FEATURE_VERSION as VP_FEATURE_VERSION,
    RUNTIME_STATE_DIR as VP_RUNTIME_STATE_DIR,
    extract_features_from_state,
    is_volume_profile_feature,
    load_state as load_volume_profile_state,
    normalize_config as normalize_volume_profile_config,
    save_state as save_volume_profile_state,
    state_matches_config as volume_profile_state_matches_config,
    update_state_with_candle as update_volume_profile_state_with_candle,
    validate_volume_profile_feature_columns,
    validate_volume_profile_model_metadata,
)
from utils.config import coerce_path
from utils.data import (
    MODELING_DATASET_CONFIG_FILE,
    load_modeling_dataset_settings,
    split_feature_subset,
)
from utils.live import (
    LIVE_SHARED_MARKET_DATA_COLUMNS,
    as_utc_timestamp,
    build_live_market_data_path,
    interval_to_floor_rule,
    interval_to_timedelta,
    resolve_polymarket_closed_position_settlement,
    setup_live_console_logging,
    upsert_records_csv,
    write_records_csv,
)
from utils.project_config import (
    load_dataset_profile,
    load_live_profile,
    load_runtime_artifact_paths,
)
from utils.trading import (
    build_trade_intent,
    decide_trade_from_ev,
    load_trade_policy_runtime_config,
)

TRADING_IMPORT_ERROR = None
try:
    import httpx
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
except ModuleNotFoundError as exc:
    TRADING_IMPORT_ERROR = exc
    httpx = None
    pyclob_headers = None
    Account = None
    build_safe_transaction_request = None
    get_relayer_contract_config = None
    RelayerOperationType = None
    RelayerSafeTransaction = None
    RelayerSafeTransactionArgs = None
    RelayerSigner = None
    ClobClient = None
    AssetType = None
    BalanceAllowanceParams = None
    MarketOrderArgs = None
    PartialCreateOrderOptions = None
    pyclob_http_helpers = None
    BUY = None
    SELL = None


    class OrderType:
        FOK = "FOK"
        FAK = "FAK"

from utils.live import (
    LIVE_TRADE_EXPORT_COLUMNS,
    build_live_trade_records_path,
    read_records_state,
    write_records_state,
)
from utils.config import load_repo_env
from utils.polymarket import (
    DEFAULT_POLYMARKET_FEE_ROUND_DECIMALS,
    DEFAULT_POLYMARKET_MIN_FEE_USDC,
    normalize_polymarket_fee_model,
    polymarket_fee_model_from_market,
    polymarket_taker_fee_usdc_from_shares,
)
from utils.polymarket import (
    POLYMARKET_BINARY_INDEX_SETS,
    POLYMARKET_RELAYER_PENDING_STATES,
    POLYMARKET_RELAYER_TERMINAL_STATES,
    build_redeem_transactions as build_redeem_transaction_specs,
    collect_redeem_candidates as collect_redeem_candidate_specs,
    encode_redeem_positions_call,
    polymarket_market_slug_matches_prefix,
    resolve_redeem_collateral_address,
    resolve_redeem_ctf_address,
    resolve_redeem_target_address,
    resolve_relayer_tx_type,
)
from utils.trading import (
    decide_trade_from_model_direction,
    resolve_fee_fractions_from_quotes,
)

load_repo_env()
DEFAULT_MODEL_PREDICTION_THRESHOLD = 0.5


def normalize_live_market_type(value, field_name):
    normalized = str(value).strip().lower()
    if normalized not in {"spot", "um", "cm"}:
        raise ValueError(f"{field_name} must be one of {{'spot', 'um', 'cm'}}")
    return normalized


def _delay_ms_between(start, end):
    if start is None or end is None:
        return float("nan")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return float(max((end_ts - start_ts).total_seconds() * 1000.0, 0.0))


LIVE_PROFILE = load_live_profile()
DATASET_PROFILE = load_dataset_profile()
RUNTIME_ARTIFACT_PATHS = load_runtime_artifact_paths()

SYMBOL = str(DATASET_PROFILE["symbol"]).strip().upper()
INTERVAL = str(DATASET_PROFILE["interval"]).strip()
PRICE_MARKET = normalize_live_market_type(DATASET_PROFILE["market"], "dataset.market")
VOLUME_SYMBOL = str(DATASET_PROFILE["volume_symbol"]).strip().upper()
VOLUME_MARKET = normalize_live_market_type(
    DATASET_PROFILE["volume_market"],
    "dataset.volume_market",
)
INDEX_PRICE_SYNTHETIC_VOLUME_DEFAULT = float(
    LIVE_PROFILE["index_price_synthetic_volume_default"]
)
MODEL_META_PATH = Path(RUNTIME_ARTIFACT_PATHS["model_meta_path"])
TRADE_POLICY_CONFIG_PATH = Path(
    RUNTIME_ARTIFACT_PATHS["trade_policy_path"]
)
INDICATOR_HISTORY_REQUIREMENTS_PATH = Path(
    RUNTIME_ARTIFACT_PATHS["indicator_history_requirements_path"]
)
# Polymarket 5m up/down markets resolve from the market itself.
SETTLEMENT_SOURCE = "polymarket"
POLYMARKET_GAMMA_HOST = str(LIVE_PROFILE["polymarket_gamma_host"]).strip().rstrip("/")
POLYMARKET_SERIES_SLUG = str(LIVE_PROFILE["polymarket_series_slug"]).strip()
POLYMARKET_MARKET_SLUG_PREFIX = str(
    LIVE_PROFILE["polymarket_market_slug_prefix"]
).strip()
POLYMARKET_MARKET_REQUEST_TIMEOUT_SEC = float(
    LIVE_PROFILE["polymarket_market_request_timeout_sec"]
)
LIVE_ROOT_DIR = Path("data/live")
LIVE_TRADE_DIR = LIVE_ROOT_DIR / "trade"
RUN_STARTED_AT_UTC = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
VOLUME_PROFILE_RUNTIME_STATE_PATH = (
        VP_RUNTIME_STATE_DIR / f"{SYMBOL}_{INTERVAL}_{VP_FEATURE_VERSION}"
)

DEFAULT_BOOTSTRAP_CANDLES = int(LIVE_PROFILE["default_bootstrap_candles"])
DEFAULT_INDICATOR_HISTORY_MARGIN_RATIO = float(
    LIVE_PROFILE["indicator_history_margin_ratio"]
)
DEFAULT_INDICATOR_HISTORY_MIN_EXTRA_CANDLES = int(
    LIVE_PROFILE["indicator_history_min_extra_candles"]
)
MAX_WS_RECONNECT_DELAY_SEC = int(LIVE_PROFILE["max_ws_reconnect_delay_sec"])
WS_PING_INTERVAL_SEC = int(LIVE_PROFILE["ws_ping_interval_sec"])
WS_PING_TIMEOUT_SEC = int(LIVE_PROFILE["ws_ping_timeout_sec"])

LIVE_INITIAL_BANKROLL_USDC = float(LIVE_PROFILE["live_initial_bankroll_usdc"])

OHLCV_COLS = list(RAW_OHLCV_COLS)
BASE_FEATURE_COLS = (
        set(OHLCV_COLS)
        | set(SUPPORTED_CANDLE_FEATURE_COLS)
        | set(SUPPORTED_SESSION_OPEN_FEATURE_COLS)
        | set(REALIZED_VOLATILITY_FEATURE_COLUMNS)
)
VALUE_BUILDERS = {
    "ADX": get_adx_values,
    "BollingerBands": get_bollinger_bands_values,
    "ChaikinOsc": get_chaikin_oscillator_values,
    "KeltnerChannel": get_keltner_channel_values,
    "MACD": get_macd_values,
    "StochOsc": get_stochastic_oscillator_values,
}


def _basis_premium_futures_close_label():
    cfg = dict(MODELING_DATASET_SETTINGS.get("basis_premium_features") or {})
    return str(cfg.get("futures_close_col", "") or "").strip() or (
        "<auto-detected futures close>"
    )


def _basis_premium_live_error():
    futures_close_col = _basis_premium_futures_close_label()
    return (
        "Basis premium live prediction requires futures close column "
        f"'{futures_close_col}' in raw history; rebuild live input with auxiliary "
        "source columns."
    )


def _basis_premium_config():
    cfg = dict(MODELING_DATASET_SETTINGS.get("basis_premium_features") or {})
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "index_close_col": str(cfg.get("index_close_col", "Close") or "Close").strip(),
        "futures_close_col": str(cfg.get("futures_close_col", "") or "").strip(),
        "eps": float(cfg.get("eps", 1e-12)),
    }


def _basis_premium_intervals_from_feature_cols(feature_cols):
    intervals = []
    seen = set()
    unsupported = []
    for raw_col in feature_cols:
        interval = str(raw_col).strip().rsplit("_", 1)[-1]
        if interval not in FEATURE_INTERVAL_TO_RULE:
            unsupported.append(interval)
            continue
        if interval in seen:
            continue
        intervals.append(interval)
        seen.add(interval)
    if unsupported or not intervals:
        raise ValueError(
            "Unsupported basis premium live feature intervals: "
            f"{unsupported}. Supported: {', '.join(FEATURE_INTERVAL_TO_RULE.keys())}"
        )
    return tuple(intervals)


def _auxiliary_ohlc_cols(market_type, symbol):
    market = str(market_type).strip().upper()
    symbol = str(symbol).strip().upper()
    return {
        "Open": f"{market}_{symbol}_Open",
        "High": f"{market}_{symbol}_High",
        "Low": f"{market}_{symbol}_Low",
        "Close": f"{market}_{symbol}_Close",
    }


def _live_auxiliary_ohlc_cols():
    if PRICE_SOURCE == VOLUME_SOURCE:
        return {}
    if VOLUME_SOURCE != "trade" or VOLUME_MARKET not in {"um", "cm"}:
        return {}
    return _auxiliary_ohlc_cols(VOLUME_MARKET, VOLUME_SYMBOL)


KLINE_PRICE_KEY_BY_COL = {
    "Open": "o",
    "High": "h",
    "Low": "l",
    "Close": "c",
}


def _extract_live_auxiliary_ohlc_from_kline(kline, auxiliary_ohlc_cols=None):
    auxiliary_ohlc_cols = (
        LIVE_AUXILIARY_OHLC_COLS
        if auxiliary_ohlc_cols is None
        else dict(auxiliary_ohlc_cols)
    )
    if not auxiliary_ohlc_cols:
        return {}

    values = {}
    for source_col, auxiliary_col in auxiliary_ohlc_cols.items():
        kline_key = KLINE_PRICE_KEY_BY_COL[source_col]
        values[auxiliary_col] = float(kline[kline_key])
    return values


class IndicatorSpec:
    __slots__ = (
        "indicator",
        "feature_col",
        "builder",
        "latest_builder",
        "params",
        "required_candles",
    )

    def __init__(
            self,
            indicator,
            feature_col,
            builder,
            latest_builder,
            params,
            required_candles,
    ):
        self.indicator = indicator
        self.feature_col = feature_col
        self.builder = builder
        self.latest_builder = latest_builder
        self.params = params
        self.required_candles = required_candles


MODELING_DATASET_SETTINGS = load_modeling_dataset_settings()
FIT_RESULTS_DIR = MODELING_DATASET_SETTINGS["fit_results_dir"]
VOLUME_PROFILE_MODELING_STATE_PATH = resolve_volume_profile_modeling_state_path(
    MODELING_DATASET_SETTINGS["base_data_file"]
)


def normalize_live_source_selection(price_source, volume_source):
    normalized_price_source = str(price_source).strip().lower()
    if normalized_price_source not in {"trade", "index"}:
        raise ValueError("Configured price_source must be one of {'trade', 'index'}")

    normalized_volume_source = str(volume_source).strip().lower()
    if normalized_volume_source in {"", "same"}:
        normalized_volume_source = normalized_price_source

    if normalized_volume_source not in {"trade", "index"}:
        raise ValueError(
            "Configured volume_source must be one of {'same', 'trade', 'index'}"
        )

    if normalized_price_source == "trade" and normalized_volume_source == "index":
        raise ValueError(
            "volume_source='index' is not supported when price_source='trade'."
        )

    return normalized_price_source, normalized_volume_source


def parse_json_listish(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return [part.strip() for part in text.split(",") if part.strip()]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def resolve_polymarket_market_slug(bucket_start, market_slug=""):
    market_slug_text = str(market_slug or "").strip()
    if market_slug_text:
        return market_slug_text
    bucket_start_ts = as_utc_timestamp(bucket_start)
    return f"{POLYMARKET_MARKET_SLUG_PREFIX}-{int(bucket_start_ts.timestamp())}"


def fetch_polymarket_market_by_slug(session, market_slug):
    response = session.get(
        f"{POLYMARKET_GAMMA_HOST}/markets/slug/{market_slug}",
        timeout=POLYMARKET_MARKET_REQUEST_TIMEOUT_SEC,
    )
    response.raise_for_status()
    return response.json()


def resolve_polymarket_actual_up_from_market_payload(market_payload):
    if not isinstance(market_payload, dict):
        return None

    resolution_status = (
        str(market_payload.get("umaResolutionStatus", "") or "").strip().lower()
    )
    winning_outcome = (
        str(market_payload.get("winning_outcome", "") or "").strip().lower()
    )
    if winning_outcome in {"up", "yes"}:
        return 1
    if winning_outcome in {"down", "no"}:
        return 0

    outcomes = [
        str(item).strip() for item in parse_json_listish(market_payload.get("outcomes"))
    ]
    if len(outcomes) != 2:
        return None

    outcome_prices = []
    for item in parse_json_listish(market_payload.get("outcomePrices")):
        try:
            outcome_prices.append(float(item))
        except (TypeError, ValueError):
            return None

    if len(outcome_prices) != len(outcomes):
        return None

    winning_idx = None
    for idx, price in enumerate(outcome_prices):
        if abs(price - 1.0) <= 1e-9:
            winning_idx = idx
            break
    if winning_idx is None:
        max_price = max(outcome_prices)
        if resolution_status != "resolved" or outcome_prices.count(max_price) != 1:
            return None
        winning_idx = int(np.argmax(np.asarray(outcome_prices, dtype=np.float64)))

    winning_label = outcomes[winning_idx].lower()
    if winning_label in {"up", "yes"}:
        return 1
    if winning_label in {"down", "no"}:
        return 0
    return None


def resolve_record_accuracy_from_side(record, actual_up=None):
    if actual_up is None:
        actual_up = record.get("actual_up")
    if actual_up is None or pd.isna(actual_up):
        return None

    side = str(record.get("trade_side", "") or "").strip().lower()
    if side == "yes":
        return int(int(actual_up) == 1)
    if side == "no":
        return int(int(actual_up) == 0)
    return None


def resolve_model_side_from_proba(proba_up, threshold=DEFAULT_MODEL_PREDICTION_THRESHOLD):
    try:
        proba_value = float(proba_up)
    except (TypeError, ValueError):
        return "none"
    if not np.isfinite(proba_value):
        return "none"
    return "up" if proba_value >= float(threshold) else "down"


def resolve_model_accuracy_from_proba(
        record,
        *,
        actual_up=None,
        threshold=DEFAULT_MODEL_PREDICTION_THRESHOLD,
):
    if actual_up is None:
        actual_up = record.get("actual_up")
    if actual_up is None or pd.isna(actual_up):
        return None

    model_side = resolve_model_side_from_proba(
        record.get("proba_up"),
        threshold=threshold,
    )
    if model_side == "up":
        return int(int(actual_up) == 1)
    if model_side == "down":
        return int(int(actual_up) == 0)
    return None


def resolve_rest_klines_endpoint(price_source, market_type):
    market_type = normalize_live_market_type(market_type, "market_type")
    if price_source == "trade":
        if market_type == "spot":
            return "https://api.binance.com/api/v3/klines", "symbol"
        if market_type == "um":
            return "https://fapi.binance.com/fapi/v1/klines", "symbol"
        return "https://dapi.binance.com/dapi/v1/klines", "symbol"
    if price_source == "index":
        if market_type == "um":
            return "https://fapi.binance.com/fapi/v1/indexPriceKlines", "pair"
        if market_type == "cm":
            return "https://dapi.binance.com/dapi/v1/indexPriceKlines", "pair"
        raise ValueError(
            "Unsupported market_type for price_source='index': "
            f"{market_type!r}. Expected one of {{'um', 'cm'}}."
        )
    raise ValueError(f"Unsupported price_source: {price_source}")


def resolve_ws_stream_name(symbol, interval, source):
    if source == "trade":
        return f"{symbol.lower()}@kline_{interval}"
    if source == "index":
        return f"{symbol.lower()}@indexPriceKline_{interval}"
    raise ValueError(f"Unsupported source for websocket stream: {source}")


def resolve_ws_base_url(market_type):
    market_type = normalize_live_market_type(market_type, "market_type")
    if market_type == "spot":
        return "wss://stream.binance.com:9443/ws"
    if market_type == "um":
        return "wss://fstream.binance.com/market/ws"
    return "wss://dstream.binance.com/ws"


def build_ws_targets(
        *,
        interval,
        price_symbol,
        price_market,
        price_source,
        volume_symbol,
        volume_market,
        volume_source,
):
    candidates = [
        {
            "role": "price",
            "market_type": price_market,
            "symbol": price_symbol,
            "source": price_source,
        },
        {
            "role": "volume",
            "market_type": volume_market,
            "symbol": volume_symbol,
            "source": volume_source,
        },
    ]

    targets = []
    seen = set()
    for candidate in candidates:
        stream_name = resolve_ws_stream_name(
            candidate["symbol"],
            interval,
            candidate["source"],
        )
        target_key = (candidate["market_type"], stream_name)
        if target_key in seen:
            continue
        seen.add(target_key)
        url = f"{resolve_ws_base_url(candidate['market_type'])}/{stream_name}"
        targets.append(
            {
                "role": candidate["role"],
                "market_type": candidate["market_type"],
                "symbol": candidate["symbol"],
                "source": candidate["source"],
                "stream_name": stream_name,
                "url": url,
                "label": (
                    f"{candidate['role']}:{candidate['market_type']}:"
                    f"{candidate['symbol']}:{candidate['source']}"
                ),
            }
        )
    return targets


# OHLC/V source must stay consistent with the dataset used to build the modeling set.
PRICE_SOURCE, VOLUME_SOURCE = normalize_live_source_selection(
    DATASET_PROFILE["price_source"],
    DATASET_PROFILE["volume_source"],
)
LIVE_AUXILIARY_OHLC_COLS = _live_auxiliary_ohlc_cols()
PRICE_STREAM_NAME = resolve_ws_stream_name(SYMBOL, INTERVAL, PRICE_SOURCE)
VOLUME_STREAM_NAME = resolve_ws_stream_name(VOLUME_SYMBOL, INTERVAL, VOLUME_SOURCE)
WS_TARGETS = build_ws_targets(
    interval=INTERVAL,
    price_symbol=SYMBOL,
    price_market=PRICE_MARKET,
    price_source=PRICE_SOURCE,
    volume_symbol=VOLUME_SYMBOL,
    volume_market=VOLUME_MARKET,
    volume_source=VOLUME_SOURCE,
)
INTERVAL_DELTA = interval_to_timedelta(INTERVAL)
INTERVAL_FLOOR_RULE = interval_to_floor_rule(INTERVAL)
INDICATOR_HISTORY_MARGIN_RATIO = float(DEFAULT_INDICATOR_HISTORY_MARGIN_RATIO)
INDICATOR_HISTORY_MIN_EXTRA_CANDLES = int(
    DEFAULT_INDICATOR_HISTORY_MIN_EXTRA_CANDLES
)

if (
        not np.isfinite(INDICATOR_HISTORY_MARGIN_RATIO)
        or INDICATOR_HISTORY_MARGIN_RATIO < 0.0
):
    raise ValueError(
        "live.indicator_history_margin_ratio must be a finite number >= 0."
    )
if INDICATOR_HISTORY_MIN_EXTRA_CANDLES < 0:
    raise ValueError("live.indicator_history_min_extra_candles must be >= 0.")


def parse_target_bucket_minutes(target_col):
    match = re.search(r"target_(\d+)m", target_col)
    return int(match.group(1)) if match else 5


def _hash_path_contents(path):
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash missing path: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        normalized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()[:12]

    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _load_ws_payload(message):
    if isinstance(message, (dict, list)):
        return message
    return json.loads(message)


def resolve_model_meta_and_path(meta_path):
    meta_path = coerce_path(meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    candidate_paths = []

    artifacts = meta.get("artifacts")
    if isinstance(artifacts, dict):
        final_model_path = artifacts.get("final_model_path")
        if isinstance(final_model_path, str) and final_model_path.strip():
            candidate_paths.append(coerce_path(final_model_path))

    candidate_paths.extend(
        [
            meta_path.with_name(meta_path.name.replace("_meta_", "_")).with_suffix(
                ".txt"
            ),
            meta_path.with_name(meta_path.stem.replace("_meta", "")).with_suffix(
                ".txt"
            ),
            meta_path.with_name(meta_path.name.replace("_meta.json", ".txt")),
        ]
    )

    model_path = next((path for path in candidate_paths if path.exists()), None)
    if model_path is None:
        searched = ", ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(
            f"Model file not found for metadata {meta_path}. Candidates: {searched}"
        )

    return meta, model_path


def load_model_and_meta(meta_path):
    meta, model_path = resolve_model_meta_and_path(meta_path)
    return lgb.Booster(model_file=str(model_path)), meta


def estimate_required_candles(indicator, params):
    periods = [
        int(v)
        for k, v in params.items()
        if "period" in k.lower() and isinstance(v, (int, np.integer))
    ]
    if not periods:
        return 0

    required = max(periods)
    if indicator == "MACD":
        required = max(
            required,
            int(params.get("slow_period", required))
            + int(params.get("signal_period", 0))
            + 50,
        )
    elif indicator in {"ADX", "BollingerBands", "KeltnerChannel", "StochOsc"}:
        required += 50
    return required


def apply_indicator_history_margin(
        required_window,
        *,
        margin_ratio=INDICATOR_HISTORY_MARGIN_RATIO,
        min_extra_candles=INDICATOR_HISTORY_MIN_EXTRA_CANDLES,
):
    base_window = int(required_window or 0)
    if base_window <= 0:
        return 0

    extra = max(
        int(min_extra_candles),
        int(np.ceil(float(base_window) * float(margin_ratio))),
    )
    return int(base_window + extra)


def _normalize_feature_window_map(raw_map):
    if not isinstance(raw_map, dict):
        return {}

    normalized = {}
    for raw_feature_col, raw_window in raw_map.items():
        feature_col = str(raw_feature_col).strip()
        if not feature_col:
            continue
        try:
            window = int(raw_window or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Indicator history requirements artifact contains a non-integer per-feature "
                f"window for '{feature_col}'."
            ) from exc
        if window > 0:
            normalized[feature_col] = int(window)
    return normalized


def load_indicator_history_requirements(
        artifact_path,
        *,
        indicator_specs=None,
        allow_unstable=False,
):
    artifact_path = coerce_path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            "Indicator history requirements artifact is required for live runtime: "
            f"{artifact_path}"
        )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            "Indicator history requirements artifact must contain a JSON object: "
            f"{artifact_path}"
        )

    if "unstable_feature_count" not in payload:
        raise ValueError(
            "Indicator history requirements artifact is missing unstable_feature_count: "
            f"{artifact_path}"
        )
    unstable_count = int(payload["unstable_feature_count"])
    if unstable_count != 0 and not allow_unstable:
        raise ValueError(
            "Indicator history requirements artifact reports unstable features; "
            "live runtime should not proceed."
        )

    if "global_required_stable_window" not in payload:
        raise ValueError(
            "Indicator history requirements artifact is missing global_required_stable_window: "
            f"{artifact_path}"
        )
    global_required_stable_window = int(payload["global_required_stable_window"])
    if global_required_stable_window <= 0:
        raise ValueError(
            "Indicator history requirements artifact contains invalid "
            "global_required_stable_window: "
            f"{artifact_path}"
        )

    stable_window_by_feature = _normalize_feature_window_map(
        payload.get("required_stable_window_by_feature")
    )
    if not stable_window_by_feature:
        raise ValueError(
            "Indicator history requirements artifact is missing "
            "required_stable_window_by_feature: "
            f"{artifact_path}"
        )

    stable_window_max = max(stable_window_by_feature.values())
    if global_required_stable_window != stable_window_max:
        raise ValueError(
            "Indicator history requirements artifact is inconsistent: "
            "global_required_stable_window must equal max(required_stable_window_by_feature). "
            f"path={artifact_path} "
            f"global_required_stable_window={global_required_stable_window} "
            f"stable_window_max={stable_window_max}"
        )

    if indicator_specs is not None:
        specs = list(indicator_specs)
        missing_feature_cols = sorted(
            {
                str(spec.feature_col)
                for spec in specs
                if str(spec.feature_col) not in stable_window_by_feature
            }
        )
        if missing_feature_cols:
            preview = ", ".join(missing_feature_cols[:10])
            raise ValueError(
                "Indicator history requirements artifact is missing per-feature windows "
                f"for {len(missing_feature_cols)} model features. "
                f"path={artifact_path} preview=[{preview}]"
            )

    runtime_window_by_feature = {
        feature_col: apply_indicator_history_margin(required_window)
        for feature_col, required_window in stable_window_by_feature.items()
    }
    global_required_runtime_window = max(
        runtime_window_by_feature.values(),
        default=apply_indicator_history_margin(global_required_stable_window),
    )
    if global_required_runtime_window <= 0:
        raise ValueError(
            "Indicator history requirements artifact produced an invalid runtime "
            f"window: {artifact_path}"
        )

    return {
        "payload": payload,
        "global_required_stable_window": int(global_required_stable_window),
        "global_required_runtime_window": int(global_required_runtime_window),
        "stable_window_by_feature": stable_window_by_feature,
        "runtime_window_by_feature": runtime_window_by_feature,
    }


def load_required_stable_window(
        artifact_path,
        *,
        allow_unstable=False,
        indicator_specs=None,
):
    requirements = load_indicator_history_requirements(
        artifact_path,
        indicator_specs=indicator_specs,
        allow_unstable=allow_unstable,
    )
    return int(requirements["global_required_runtime_window"])


def load_indicator_specs(feature_columns, *, source_label=None):
    source_label = source_label or f"model metadata feature_columns at {MODEL_META_PATH}"
    validate_volume_profile_feature_columns(
        feature_columns,
        source_label=source_label,
    )
    validate_basis_premium_feature_columns(
        feature_columns,
        source_label=source_label,
    )
    fit_configs = parse_fit_results(FIT_RESULTS_DIR)
    fit_by_feature_col = {cfg["feature_col"]: cfg for cfg in fit_configs}

    specs = []
    missing_features = []

    for col in feature_columns:
        if (
                col in BASE_FEATURE_COLS
                or col.startswith(STREAK_FEATURE_PREFIX)
                or is_volume_profile_feature(col)
                or is_basis_premium_feature(col)
        ):
            continue

        cfg = fit_by_feature_col.get(col)
        if cfg is None:
            missing_features.append(col)
            continue

        indicator = str(cfg["indicator"])
        params = cfg["params"]
        builder = VALUE_BUILDERS.get(indicator)
        if builder is None:
            raise ValueError(
                f"Indicator '{indicator}' not supported by live VALUE_BUILDERS for feature '{col}'."
            )
        latest_builder = LIVE_LATEST_VALUE_BUILDERS.get(indicator)
        if latest_builder is None:
            raise ValueError(
                f"Indicator '{indicator}' not supported by live latest value builders for feature '{col}'."
            )

        specs.append(
            IndicatorSpec(
                indicator=indicator,
                feature_col=col,
                builder=builder,
                latest_builder=latest_builder,
                params=params,
                required_candles=estimate_required_candles(indicator, params),
            )
        )

    if missing_features:
        preview = ", ".join(missing_features[:10])
        raise FileNotFoundError(
            "Missing fit configs for model feature columns in fit_results_dir "
            f"{FIT_RESULTS_DIR.resolve()}. Missing_count={len(missing_features)} "
            f"preview=[{preview}]"
        )

    return specs


def _rest_kline_params(
        source,
        market_type,
        symbol,
        limit,
        start_time_ms=None,
        end_time_ms=None,
):
    rest_url, symbol_param = resolve_rest_klines_endpoint(source, market_type)
    params = {
        symbol_param: str(symbol).strip().upper(),
        "interval": INTERVAL,
        "limit": int(limit),
    }
    if start_time_ms is not None:
        params["startTime"] = int(start_time_ms)
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    return rest_url, params


def _rest_limit_for_market_type(market_type, requested_limit):
    market_type = normalize_live_market_type(market_type, "market_type")
    max_limit = 1000 if market_type == "spot" else 1500
    return max(1, min(int(requested_limit), int(max_limit)))


def _volume_from_rest_row(row, source):
    if source == "index":
        try:
            value = float(row[8])
        except (IndexError, TypeError, ValueError):
            value = 0.0
        return float(value if value > 0.0 else INDEX_PRICE_SYNTHETIC_VOLUME_DEFAULT)
    return float(row[5])


def _merge_price_and_volume_frames(price_df, volume_df, auxiliary_ohlc_cols=None):
    auxiliary_ohlc_cols = (
        LIVE_AUXILIARY_OHLC_COLS
        if auxiliary_ohlc_cols is None
        else dict(auxiliary_ohlc_cols)
    )
    volume_cols = ["Opened", "Volume"]
    volume_cols.extend(auxiliary_ohlc_cols.keys())

    merged = price_df.loc[:, ["Opened", "Open", "High", "Low", "Close"]].merge(
        volume_df.loc[:, volume_cols],
        on="Opened",
        how="inner",
        suffixes=("", "__volume_source"),
    )
    if auxiliary_ohlc_cols:
        drop_cols = []
        for source_col, auxiliary_col in auxiliary_ohlc_cols.items():
            merged_col = f"{source_col}__volume_source"
            merged[auxiliary_col] = merged[merged_col]
            drop_cols.append(merged_col)
        merged = merged.drop(columns=drop_cols)

    merged = (
        merged.drop_duplicates(subset=["Opened"], keep="last")
        .sort_values("Opened")
        .reset_index(drop=True)
    )
    ordered_cols = ["Opened", *OHLCV_COLS, *auxiliary_ohlc_cols.values()]
    if merged.empty:
        return pd.DataFrame(columns=ordered_cols)
    return merged.loc[:, ordered_cols]


def _fetch_single_source_closed_ohlcv_range(
        session,
        start_opened,
        end_opened=None,
        limit=1000,
        source="trade",
        symbol="",
        market_type="um",
):
    start_ts = pd.Timestamp(start_opened)
    if end_opened is None:
        end_ts = pd.Timestamp.now(tz="UTC").floor(INTERVAL_FLOOR_RULE) - INTERVAL_DELTA
    else:
        end_ts = pd.Timestamp(end_opened)
    if end_ts is not None and end_ts < start_ts:
        return pd.DataFrame(columns=["Opened", *OHLCV_COLS])

    now_ms = int(time.time() * 1000)
    all_rows = []
    next_end = pd.Timestamp(end_ts)
    effective_limit = _rest_limit_for_market_type(market_type, limit)
    interval_ms = int(INTERVAL_DELTA.total_seconds() * 1000)

    while True:
        rest_url, params = _rest_kline_params(
            source=source,
            market_type=market_type,
            symbol=symbol,
            limit=effective_limit,
            end_time_ms=int(next_end.value // 1_000_000) + interval_ms - 1,
        )

        response = session.get(rest_url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not data:
            break

        all_rows.extend(data)
        first_opened = pd.to_datetime(int(data[0][0]), unit="ms", utc=True)
        if first_opened <= start_ts:
            break
        if len(data) < effective_limit:
            break
        next_end = first_opened - INTERVAL_DELTA
        time.sleep(0.05)

    if not all_rows:
        return pd.DataFrame(columns=["Opened", *OHLCV_COLS])

    rows = []
    for row in all_rows:
        if int(row[6]) >= now_ms:
            continue
        opened = pd.to_datetime(int(row[0]), unit="ms", utc=True)
        if opened < start_ts:
            continue
        if end_ts is not None and opened > end_ts:
            continue
        rows.append(
            {
                "Opened": opened,
                "Open": float(row[1]),
                "High": float(row[2]),
                "Low": float(row[3]),
                "Close": float(row[4]),
                "Volume": _volume_from_rest_row(row, source=source),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Opened", *OHLCV_COLS])

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["Opened"])
        .sort_values("Opened")
        .reset_index(drop=True)
    )


def fetch_historical_ohlcv(session, candles):
    end_opened = pd.Timestamp.now(tz="UTC").floor(INTERVAL_FLOOR_RULE) - INTERVAL_DELTA
    start_opened = end_opened - ((int(candles) - 1) * INTERVAL_DELTA)
    out = fetch_closed_ohlcv_range(
        session,
        start_opened=start_opened,
        end_opened=end_opened,
    )
    if out.empty:
        raise RuntimeError("REST bootstrap returned no candles.")
    if len(out) < int(candles):
        raise RuntimeError(
            f"REST bootstrap returned only {len(out)} candles, expected at least {candles}."
        )
    return out.tail(int(candles)).reset_index(drop=True)


def fetch_closed_ohlcv_range(session, start_opened, end_opened=None, limit=1000):
    price_df = _fetch_single_source_closed_ohlcv_range(
        session,
        start_opened=start_opened,
        end_opened=end_opened,
        limit=limit,
        source=PRICE_SOURCE,
        symbol=SYMBOL,
        market_type=PRICE_MARKET,
    )
    if VOLUME_SOURCE == PRICE_SOURCE:
        return price_df

    volume_df = _fetch_single_source_closed_ohlcv_range(
        session,
        start_opened=start_opened,
        end_opened=end_opened,
        limit=limit,
        source=VOLUME_SOURCE,
        symbol=VOLUME_SYMBOL,
        market_type=VOLUME_MARKET,
    )
    return _merge_price_and_volume_frames(price_df, volume_df)


class LivePredictor:
    def __init__(self):
        self.run_started_at_utc = RUN_STARTED_AT_UTC
        self.model_meta_path = MODEL_META_PATH
        meta, self.model_file_path = resolve_model_meta_and_path(MODEL_META_PATH)
        self.model = lgb.Booster(model_file=str(self.model_file_path))
        self.model_hash = _hash_path_contents(self.model_file_path)
        self.trade_policy_config_hash = _hash_path_contents(TRADE_POLICY_CONFIG_PATH)
        self.modeling_dataset_config_hash = _hash_path_contents(
            MODELING_DATASET_CONFIG_FILE
        )
        self.feature_columns = list(meta.get("feature_columns", []))
        if not self.feature_columns:
            raise ValueError("Missing feature_columns in model metadata.")
        validate_volume_profile_feature_columns(
            self.feature_columns,
            source_label=f"model metadata {self.model_meta_path}",
        )
        validate_basis_premium_feature_columns(
            self.feature_columns,
            source_label=f"model metadata {self.model_meta_path}",
        )
        print(f"using fit results dir: {FIT_RESULTS_DIR.resolve()}")
        self.candle_feature_columns = [
            col for col in self.feature_columns if col in SUPPORTED_CANDLE_FEATURE_COLS
        ]
        self.candle_derived_feature_columns = tuple(
            resolve_candle_derived_feature_cols(self.candle_feature_columns)
        )
        self.candle_pattern_feature_columns = tuple(
            resolve_candle_pattern_feature_cols(self.candle_feature_columns)
        )

        self.target_col = str(meta.get("target_col", "target_5m_candle_up"))
        self.target_bucket_minutes = parse_target_bucket_minutes(self.target_col)
        self.prediction_threshold = float(
            meta.get("prediction_threshold", DEFAULT_MODEL_PREDICTION_THRESHOLD)
        )

        self.trade_policy_runtime = load_trade_policy_runtime_config(
            TRADE_POLICY_CONFIG_PATH
        )
        self.live_bankroll_usdc = LIVE_INITIAL_BANKROLL_USDC

        feature_parts = split_feature_subset(
            self.feature_columns,
            source_label=f"model metadata {self.model_meta_path}",
        )
        if feature_parts["streak_intervals"]:
            self.streak_interval_to_rule = resolve_streak_interval_to_rule(
                feature_parts["streak_intervals"]
            )
        else:
            self.streak_interval_to_rule = {}
        self.session_feature_columns = tuple(feature_parts["session_feature_cols"])
        self.realized_volatility_feature_columns = tuple(
            feature_parts["realized_volatility_feature_cols"]
        )
        self.basis_premium_feature_columns = tuple(
            feature_parts["basis_premium_feature_cols"]
        )
        self.basis_premium_cfg = _basis_premium_config()
        self.basis_premium_interval_to_rule = {}
        self.basis_index_close_col = ""
        self.basis_index_ohlcv_idx = None
        self.basis_futures_close_col = ""
        self.basis_futures_close_np = None
        if self.basis_premium_feature_columns:
            if not self.basis_premium_cfg["enabled"]:
                raise ValueError(
                    "Model requires basis premium features but "
                    "basis_premium_features.enabled is false."
                )
            self.basis_index_close_col = self.basis_premium_cfg["index_close_col"]
            if self.basis_index_close_col not in OHLCV_COLS:
                raise ValueError(
                    "Live basis premium prediction requires index_close_col to be "
                    f"one of raw OHLCV columns {OHLCV_COLS}; got "
                    f"{self.basis_index_close_col!r}."
                )
            self.basis_index_ohlcv_idx = OHLCV_COLS.index(
                self.basis_index_close_col
            )
            basis_intervals = _basis_premium_intervals_from_feature_cols(
                self.basis_premium_feature_columns
            )
            self.basis_premium_interval_to_rule = {
                interval: FEATURE_INTERVAL_TO_RULE[interval]
                for interval in basis_intervals
            }
        self.volume_profile_feature_columns = tuple(
            feature_parts["volume_profile_feature_cols"]
        )
        self.volume_profile_cfg = normalize_volume_profile_config(
            MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")
        )
        validate_volume_profile_model_metadata(
            meta,
            feature_columns=self.volume_profile_feature_columns,
            cfg=self.volume_profile_cfg,
            source_label=f"model metadata {self.model_meta_path}",
        )
        if (
                self.volume_profile_feature_columns
                and not self.volume_profile_cfg["enabled"]
        ):
            raise ValueError(
                "Model requires volume profile features but volume_profile_fixed_range.enabled is false."
            )
        self.volume_profile_enabled = bool(
            self.volume_profile_feature_columns and self.volume_profile_cfg["enabled"]
        )
        self.volume_profile_state_path = VOLUME_PROFILE_RUNTIME_STATE_PATH
        self.volume_profile_modeling_state_path = VOLUME_PROFILE_MODELING_STATE_PATH
        self.volume_profile_save_pool = (
            ThreadPoolExecutor(max_workers=1) if self.volume_profile_enabled else None
        )

        self.indicator_specs = load_indicator_specs(
            self.feature_columns,
            source_label=f"model metadata {self.model_meta_path}",
        )
        self.indicator_history_requirements = load_indicator_history_requirements(
            INDICATOR_HISTORY_REQUIREMENTS_PATH,
            indicator_specs=self.indicator_specs,
        )
        self.required_stable_window = int(
            self.indicator_history_requirements["global_required_runtime_window"]
        )
        self.required_stable_window_raw = int(
            self.indicator_history_requirements["global_required_stable_window"]
        )
        self.indicator_stable_window_by_feature = dict(
            self.indicator_history_requirements["stable_window_by_feature"]
        )
        self.indicator_runtime_window_by_feature = dict(
            self.indicator_history_requirements["runtime_window_by_feature"]
        )
        max_needed = max((s.required_candles for s in self.indicator_specs), default=0)
        self.bootstrap_candles = max(
            DEFAULT_BOOTSTRAP_CANDLES,
            self.required_stable_window,
            max_needed * 3,
        )
        self.max_keep = int(self.bootstrap_candles)

        self.session = requests.Session()
        self.ws_message_lock = threading.Lock()
        self.pending_ws_price_candles = {}
        self.pending_ws_volume_by_opened = {}

        bootstrap_df = fetch_historical_ohlcv(self.session, self.bootstrap_candles)
        if bootstrap_df.empty:
            raise RuntimeError("Bootstrap dataframe is empty.")

        self.opened_candles = deque(
            pd.Timestamp(opened) for opened in bootstrap_df["Opened"]
        )
        self.ohlcv_np = bootstrap_df[OHLCV_COLS].to_numpy(dtype=np.float64, copy=True)
        if len(self.opened_candles) != self.ohlcv_np.shape[0]:
            raise RuntimeError("Opened/OHLCV length mismatch after bootstrap load.")
        if len(self.opened_candles) > self.max_keep:
            drop_count = len(self.opened_candles) - self.max_keep
            for _ in range(drop_count):
                self.opened_candles.popleft()
            self.ohlcv_np = self.ohlcv_np[-self.max_keep:, :]
        basis_bootstrap_df = bootstrap_df.tail(len(self.opened_candles)).reset_index(
            drop=True
        )
        self._initialize_basis_premium_state(basis_bootstrap_df)
        self.opened_ns_np = np.fromiter(
            (opened.value for opened in self.opened_candles),
            dtype=np.int64,
            count=len(self.opened_candles),
        )

        self.candle_open_close = {
            opened: (float(self.ohlcv_np[i, 0]), float(self.ohlcv_np[i, 3]))
            for i, opened in enumerate(self.opened_candles)
        }

        self.records = []
        self.predicted_buckets = set()
        self.local_tz = datetime.now().astimezone().tzinfo
        self.last_indicator_nan_cols = []
        self.settlement_source = SETTLEMENT_SOURCE
        self.settlement_market_cache = {}
        self.last_processed_closed_opened = (
            self.opened_candles[-1] if self.opened_candles else None
        )

        self.realized_volatility_state = None
        self.latest_realized_volatility_values = {}
        self._initialize_realized_volatility_state()
        self.volume_profile_state = None
        if self.volume_profile_enabled:
            self._initialize_volume_profile_state(bootstrap_df)

    def _resolve_indicator_window_len(self, feature_col):
        if not self.indicator_runtime_window_by_feature:
            return int(self.required_stable_window)
        return int(
            self.indicator_runtime_window_by_feature.get(
                feature_col,
                self.required_stable_window,
            )
        )

    def _slice_indicator_ohlcv_window(self, feature_col):
        window_len = max(2, int(self._resolve_indicator_window_len(feature_col)))
        if self.ohlcv_np.shape[0] <= window_len:
            return self.ohlcv_np
        return self.ohlcv_np[-window_len:, :]

    def _compute_latest_indicator_value(
            self, spec, full_history_scratch, window_scratch_by_len
    ):
        window_len = max(2, int(self._resolve_indicator_window_len(spec.feature_col)))
        window_scratch = window_scratch_by_len.get(window_len)
        if window_scratch is None:
            window_scratch = IndicatorWindowScratch(full_history_scratch, window_len)
            window_scratch_by_len[window_len] = window_scratch
        return float(spec.latest_builder(spec.params, window_scratch))

    def _initialize_realized_volatility_state(self):
        if not self.realized_volatility_feature_columns:
            self.realized_volatility_state = None
            self.latest_realized_volatility_values = {}
            return

        self.realized_volatility_state = RealizedVolatilityRuntimeState()
        latest_values = {}
        for close_value in self.ohlcv_np[:, 3]:
            latest_values = self.realized_volatility_state.update(float(close_value))
        self.latest_realized_volatility_values = latest_values

    def _initialize_basis_premium_state(self, history_df):
        if not self.basis_premium_feature_columns:
            self.basis_futures_close_col = ""
            self.basis_futures_close_np = None
            return

        try:
            futures_close_col = resolve_futures_close_col(
                history_df,
                index_close_col=self.basis_index_close_col,
                futures_close_col=self.basis_premium_cfg["futures_close_col"],
            )
        except ValueError as exc:
            raise ValueError(_basis_premium_live_error()) from exc

        futures_close = pd.to_numeric(
            history_df[futures_close_col],
            errors="coerce",
        ).to_numpy(dtype=np.float64, copy=True)
        if futures_close.shape[0] != len(self.opened_candles):
            raise RuntimeError(
                "Basis premium futures close history is not aligned with OHLCV "
                f"history: {futures_close.shape[0]} != {len(self.opened_candles)}."
            )

        non_finite = int((~np.isfinite(futures_close)).sum())
        if non_finite:
            raise ValueError(
                "Basis premium live prediction requires finite futures close "
                f"history in {futures_close_col!r}; non_finite_count={non_finite}."
            )

        self.basis_futures_close_col = futures_close_col
        self.basis_futures_close_np = futures_close
        print(
            "basis/premium live features: "
            f"feature_count={len(self.basis_premium_feature_columns)} "
            f"index_close_col={self.basis_index_close_col} "
            f"futures_close_col={self.basis_futures_close_col} "
            f"intervals={list(self.basis_premium_interval_to_rule.keys())}"
        )

    def _coerce_basis_futures_close(self, value, *, context):
        if not self.basis_premium_feature_columns:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Basis premium live prediction requires futures close column "
                f"{self.basis_futures_close_col!r} for {context}."
            ) from exc
        if not np.isfinite(out):
            raise RuntimeError(
                "Basis premium live prediction received a non-finite futures close "
                f"for {context}: {out!r}."
            )
        return out

    def _basis_futures_close_from_live_candle(self, kline, ohlcv):
        if not self.basis_premium_feature_columns:
            return None
        opened = pd.to_datetime(int(kline["t"]), unit="ms", utc=True)
        if self.basis_futures_close_col in OHLCV_COLS:
            idx = OHLCV_COLS.index(self.basis_futures_close_col)
            return self._coerce_basis_futures_close(
                ohlcv[idx],
                context=f"closed candle {opened}",
            )
        if self.basis_futures_close_col not in kline:
            raise RuntimeError(
                "Closed live candle is missing basis futures close column "
                f"{self.basis_futures_close_col!r}. Check that live volume_source "
                "is the same auxiliary futures trade source used for modeling."
            )
        return self._coerce_basis_futures_close(
            kline[self.basis_futures_close_col],
            context=f"closed candle {opened}",
        )

    def _build_latest_basis_premium_features(self):
        if not self.basis_premium_feature_columns:
            return {}
        if self.basis_futures_close_np is None:
            raise RuntimeError("Basis premium live state is not initialized.")
        if self.basis_futures_close_np.shape[0] != len(self.opened_candles):
            raise RuntimeError(
                "Basis premium futures close history is not aligned with opened "
                f"candles: {self.basis_futures_close_np.shape[0]} != "
                f"{len(self.opened_candles)}."
            )

        history_df = pd.DataFrame(
            {
                "Opened": tuple(self.opened_candles),
                self.basis_index_close_col: self.ohlcv_np[
                    :, self.basis_index_ohlcv_idx
                ],
                self.basis_futures_close_col: self.basis_futures_close_np,
            }
        )
        feature_df = add_basis_premium_features(
            history_df,
            opened_col="Opened",
            index_close_col=self.basis_index_close_col,
            futures_close_col=self.basis_futures_close_col,
            interval_to_rule=self.basis_premium_interval_to_rule,
            feature_cols=self.basis_premium_feature_columns,
            eps=self.basis_premium_cfg["eps"],
        )
        latest = feature_df.iloc[-1]
        return {
            feature_col: float(latest[feature_col])
            for feature_col in self.basis_premium_feature_columns
        }

    def _append_new_candle(self, opened, ohlcv, basis_futures_close=None):
        if self.basis_premium_feature_columns:
            basis_futures_close = self._coerce_basis_futures_close(
                basis_futures_close,
                context=f"closed candle {pd.Timestamp(opened).isoformat()}",
            )
        ohlcv_row = np.asarray(ohlcv, dtype=np.float64).reshape(1, len(OHLCV_COLS))
        opened_ns_row = np.asarray([pd.Timestamp(opened).value], dtype=np.int64)
        if self.ohlcv_np.size == 0:
            self.ohlcv_np = ohlcv_row
            self.opened_ns_np = opened_ns_row
        else:
            self.ohlcv_np = np.vstack((self.ohlcv_np, ohlcv_row))
            self.opened_ns_np = np.concatenate((self.opened_ns_np, opened_ns_row))

        self.opened_candles.append(opened)
        self.candle_open_close[opened] = (
            float(ohlcv_row[0, 0]),
            float(ohlcv_row[0, 3]),
        )
        if self.basis_premium_feature_columns:
            basis_row = np.asarray([basis_futures_close], dtype=np.float64)
            if (
                    self.basis_futures_close_np is None
                    or self.basis_futures_close_np.size == 0
            ):
                self.basis_futures_close_np = basis_row
            else:
                self.basis_futures_close_np = np.concatenate(
                    (self.basis_futures_close_np, basis_row)
                )
        if self.realized_volatility_state is not None:
            self.latest_realized_volatility_values = self.realized_volatility_state.update(
                float(ohlcv_row[0, 3])
            )

        if len(self.opened_candles) > self.max_keep:
            dropped_opened = self.opened_candles.popleft()
            self.candle_open_close.pop(dropped_opened, None)
            self.ohlcv_np = self.ohlcv_np[-self.max_keep:, :]
            self.opened_ns_np = self.opened_ns_np[-self.max_keep:]
            if self.basis_premium_feature_columns:
                self.basis_futures_close_np = self.basis_futures_close_np[
                    -self.max_keep:
                ]

    def _latest_btc_snapshot(self):
        if self.ohlcv_np.size == 0:
            return {
                "btc_open": np.nan,
                "btc_high": np.nan,
                "btc_low": np.nan,
                "btc_close": np.nan,
                "btc_volume": np.nan,
            }

        last_row = self.ohlcv_np[-1, :]
        return {
            "btc_open": float(last_row[0]),
            "btc_high": float(last_row[1]),
            "btc_low": float(last_row[2]),
            "btc_close": float(last_row[3]),
            "btc_volume": float(last_row[4]),
        }

    def _backfill_bucket_price_bounds(self, record):
        bucket_start = pd.to_datetime(record.get("bucket_start"), errors="coerce", utc=True)
        bucket_end = pd.to_datetime(record.get("bucket_end"), errors="coerce", utc=True)

        if pd.notna(bucket_start):
            candle = self.candle_open_close.get(bucket_start)
            if candle is not None and pd.isna(record.get("bucket_open_price")):
                record["bucket_open_price"] = float(candle[0])

        if pd.notna(bucket_end):
            candle = self.candle_open_close.get(bucket_end)
            if candle is not None and pd.isna(record.get("bucket_close_price")):
                record["bucket_close_price"] = float(candle[1])

    def _pending_ws_candle_opened(self, candle):
        return pd.to_datetime(int(candle["t"]), unit="ms", utc=True)

    def _cleanup_pending_ws_state(self, reference_opened):
        cutoff = pd.Timestamp(reference_opened) - (INTERVAL_DELTA * 5)
        self.pending_ws_price_candles = {
            opened: candle
            for opened, candle in self.pending_ws_price_candles.items()
            if opened >= cutoff
        }
        self.pending_ws_volume_by_opened = {
            opened: volume
            for opened, volume in self.pending_ws_volume_by_opened.items()
            if opened >= cutoff
        }

    def _build_ws_closed_candle_timing(
            self,
            *,
            opened,
            live_minute_opened,
            price_meta=None,
            volume_meta=None,
            completed_at=None,
    ):
        completed_at = (
            pd.Timestamp.now(tz="UTC") if completed_at is None else pd.Timestamp(completed_at)
        )
        price_event_at = None if price_meta is None else price_meta.get("event_at")
        volume_event_at = None if volume_meta is None else volume_meta.get("event_at")
        price_received_at = (
            None if price_meta is None else price_meta.get("received_at")
        )
        volume_received_at = (
            None if volume_meta is None else volume_meta.get("received_at")
        )
        component_received = [
            pd.Timestamp(ts)
            for ts in (price_received_at, volume_received_at)
            if ts is not None
        ]
        component_event = [
            pd.Timestamp(ts)
            for ts in (price_event_at, volume_event_at)
            if ts is not None
        ]
        last_component_event_at = max(component_event) if component_event else None

        return {
            "ws_price_event_delay_ms": _delay_ms_between(
                live_minute_opened,
                price_event_at,
            ),
            "ws_volume_event_delay_ms": _delay_ms_between(
                live_minute_opened,
                volume_event_at,
            ),
            "ws_price_receive_delay_ms": _delay_ms_between(
                live_minute_opened,
                price_received_at,
            ),
            "ws_volume_receive_delay_ms": _delay_ms_between(
                live_minute_opened,
                volume_received_at,
            ),
            "ws_event_delay_ms": _delay_ms_between(
                live_minute_opened,
                last_component_event_at,
            ),
            "ws_receive_delay_ms": _delay_ms_between(
                live_minute_opened,
                completed_at,
            ),
            "ws_component_sync_ms": (
                0.0
                if len(component_received) <= 1
                else _delay_ms_between(
                    min(component_received),
                    max(component_received),
                )
            ),
            "ws_ready_at": completed_at,
            "ws_candle_opened": pd.Timestamp(opened),
            "ws_bucket_start": pd.Timestamp(live_minute_opened),
        }

    def _store_pending_ws_price_candle(
            self,
            price_candle,
            *,
            event_at=None,
            received_at=None,
    ):
        opened = self._pending_ws_candle_opened(price_candle)
        if self.opened_candles and opened <= self.opened_candles[-1]:
            return None, None
        price_meta = {
            "candle": dict(price_candle),
            "event_at": event_at,
            "received_at": received_at,
        }
        live_minute_opened = opened + INTERVAL_DELTA
        if VOLUME_SOURCE == PRICE_SOURCE:
            return price_candle, self._build_ws_closed_candle_timing(
                opened=opened,
                live_minute_opened=live_minute_opened,
                price_meta=price_meta,
                completed_at=received_at,
            )

        volume_meta = self.pending_ws_volume_by_opened.pop(opened, None)
        if volume_meta is None:
            self.pending_ws_price_candles[opened] = price_meta
            self._cleanup_pending_ws_state(opened)
            return None, None

        price_candle["v"] = float(volume_meta["volume"])
        price_candle.update(volume_meta.get("auxiliary_ohlc") or {})
        self.pending_ws_price_candles.pop(opened, None)
        completed_at = max(
            pd.Timestamp(ts)
            for ts in (received_at, volume_meta.get("received_at"))
            if ts is not None
        )
        return price_candle, self._build_ws_closed_candle_timing(
            opened=opened,
            live_minute_opened=live_minute_opened,
            price_meta=price_meta,
            volume_meta=volume_meta,
            completed_at=completed_at,
        )

    def _store_pending_ws_volume(
            self,
            opened,
            volume,
            *,
            auxiliary_ohlc=None,
            event_at=None,
            received_at=None,
    ):
        opened = pd.Timestamp(opened)
        if self.opened_candles and opened <= self.opened_candles[-1]:
            return None, None

        volume_meta = {
            "volume": float(volume),
            "auxiliary_ohlc": dict(auxiliary_ohlc or {}),
            "event_at": event_at,
            "received_at": received_at,
        }
        price_meta = self.pending_ws_price_candles.pop(opened, None)
        if price_meta is None:
            self.pending_ws_volume_by_opened[opened] = volume_meta
            self._cleanup_pending_ws_state(opened)
            return None, None

        price_candle = dict(price_meta["candle"])
        price_candle["v"] = float(volume)
        price_candle.update(volume_meta.get("auxiliary_ohlc") or {})
        live_minute_opened = opened + INTERVAL_DELTA
        completed_at = max(
            pd.Timestamp(ts)
            for ts in (received_at, price_meta.get("received_at"))
            if ts is not None
        )
        return price_candle, self._build_ws_closed_candle_timing(
            opened=opened,
            live_minute_opened=live_minute_opened,
            price_meta=price_meta,
            volume_meta=volume_meta,
            completed_at=completed_at,
        )

    def _unwrap_ws_payload(self, payload):
        if isinstance(payload, dict) and "stream" in payload and "data" in payload:
            return str(payload.get("stream") or ""), payload.get("data") or {}
        if not isinstance(payload, dict):
            return "", {}
        event_type = str(payload.get("e") or "")
        if event_type == "kline":
            kline = payload.get("k") or {}
            stream_symbol = str(kline.get("s") or payload.get("s") or "").strip().upper()
            if stream_symbol:
                return resolve_ws_stream_name(stream_symbol, INTERVAL, "trade"), payload
            return resolve_ws_stream_name(SYMBOL, INTERVAL, "trade"), payload
        if event_type == "indexPrice_kline":
            pair = str(payload.get("ps") or "").strip().upper()
            if pair:
                return resolve_ws_stream_name(pair, INTERVAL, "index"), payload
            return resolve_ws_stream_name(SYMBOL, INTERVAL, "index"), payload
        return "", payload

    def _extract_closed_index_price_candle(self, payload):
        kline = payload.get("k", {})
        if not kline or not bool(kline.get("x", False)):
            return None, None
        try:
            opened = pd.to_datetime(int(kline["t"]), unit="ms", utc=True)
        except (KeyError, TypeError, ValueError):
            return None, None
        try:
            synthetic_volume = max(
                int(kline.get("n", 0) or 0),
                int(INDEX_PRICE_SYNTHETIC_VOLUME_DEFAULT),
            )
        except (TypeError, ValueError):
            synthetic_volume = int(INDEX_PRICE_SYNTHETIC_VOLUME_DEFAULT)
        closed_candle = {
            "t": int(kline["t"]),
            "o": float(kline["o"]),
            "h": float(kline["h"]),
            "l": float(kline["l"]),
            "c": float(kline["c"]),
            "v": float(synthetic_volume),
        }
        live_minute_opened = opened + INTERVAL_DELTA
        return closed_candle, live_minute_opened

    def _consume_ws_payload(self, payload):
        stream_name, data = self._unwrap_ws_payload(payload)
        if not isinstance(data, dict):
            return None, None, None, None

        event_at = None
        received_at = pd.Timestamp.now(tz="UTC")
        raw_event_ms = data.get("E")
        if raw_event_ms not in (None, ""):
            try:
                event_at = pd.to_datetime(int(raw_event_ms), unit="ms", utc=True)
            except (TypeError, ValueError):
                event_at = None

        if stream_name == PRICE_STREAM_NAME and PRICE_SOURCE == "trade":
            kline = data.get("k", {})
            if not kline or not bool(kline.get("x", False)):
                return None, None, event_at, None

            opened = pd.to_datetime(int(kline["t"]), unit="ms", utc=True)
            live_minute_opened = opened + INTERVAL_DELTA
            closed_candle, timing = self._store_pending_ws_price_candle(
                {
                    "t": int(kline["t"]),
                    "o": float(kline["o"]),
                    "h": float(kline["h"]),
                    "l": float(kline["l"]),
                    "c": float(kline["c"]),
                    "v": float(kline["v"]),
                },
                event_at=event_at,
                received_at=received_at,
            )
            return closed_candle, live_minute_opened, event_at, timing

        if stream_name == VOLUME_STREAM_NAME and VOLUME_SOURCE == "trade":
            kline = data.get("k", {})
            if not kline or not bool(kline.get("x", False)):
                return None, None, event_at, None

            opened = pd.to_datetime(int(kline["t"]), unit="ms", utc=True)
            live_minute_opened = opened + INTERVAL_DELTA
            closed_candle, timing = self._store_pending_ws_volume(
                opened,
                float(kline["v"]),
                auxiliary_ohlc=_extract_live_auxiliary_ohlc_from_kline(kline),
                event_at=event_at,
                received_at=received_at,
            )
            return closed_candle, live_minute_opened, event_at, timing

        if stream_name == PRICE_STREAM_NAME and PRICE_SOURCE == "index":
            price_candle, live_minute_opened = self._extract_closed_index_price_candle(
                data
            )
            if price_candle is None:
                return None, live_minute_opened, event_at, None
            closed_candle, timing = self._store_pending_ws_price_candle(
                price_candle,
                event_at=event_at,
                received_at=received_at,
            )
            return closed_candle, live_minute_opened, event_at, timing

        return None, None, event_at, None

    def _volume_profile_state_last_candle_timestamp(self, state):
        raw_value = state.get("last_candle_time")
        if not raw_value:
            return None
        ts = pd.Timestamp(raw_value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _save_runtime_volume_profile_state(self, log=False, context="state"):
        paths = save_volume_profile_state(
            self.volume_profile_state,
            self.volume_profile_state_path,
        )
        if log:
            print(f"[vp] saved {context} -> {paths['npz']}")
        return paths

    def _snapshot_volume_profile_state_for_save(self):
        state = self.volume_profile_state
        if state is None:
            raise RuntimeError("volume profile state is not initialized")

        snapshot = {
            "enabled": bool(state["enabled"]),
            "version": str(state["version"]),
            "price_min": float(state["price_min"]),
            "price_max": float(state["price_max"]),
            "neighbor_bins": int(state["neighbor_bins"]),
            "eps": float(state["eps"]),
            "horizon_names": tuple(state["horizon_names"]),
            "feature_columns": tuple(state["feature_columns"]),
            "config_signature": str(state["config_signature"]),
            "last_candle_time": state.get("last_candle_time"),
            "horizons": {},
        }
        for horizon_name in state["horizon_names"]:
            horizon_state = state["horizons"][horizon_name]
            snapshot["horizons"][horizon_name] = {
                "horizon_name": str(horizon_state["horizon_name"]),
                "half_life_candles": horizon_state["half_life_candles"],
                "decay": float(horizon_state["decay"]),
                "step": int(horizon_state["step"]),
                "bins": int(horizon_state["bins"]),
                "local_window": int(horizon_state["local_window"]),
                "sigma_divisor": float(horizon_state["sigma_divisor"]),
                "min_sigma": float(horizon_state["min_sigma"]),
                "global_scale": float(horizon_state["global_scale"]),
                "raw_profile": np.array(
                    horizon_state["raw_profile"], dtype=np.float64, copy=True
                ),
            }
        return snapshot

    def _save_runtime_volume_profile_state_async(self, log=False, context="state"):
        if self.volume_profile_state is None:
            return None
        save_pool = getattr(self, "volume_profile_save_pool", None)
        if save_pool is None:
            return self._save_runtime_volume_profile_state(log=log, context=context)

        state_snapshot = self._snapshot_volume_profile_state_for_save()
        future = save_pool.submit(
            save_volume_profile_state,
            state_snapshot,
            self.volume_profile_state_path,
        )
        if log:
            def _log_saved(done_future, *, save_context=context):
                try:
                    paths = done_future.result()
                except Exception as exc:
                    print(f"[vp] async save failed ({save_context}): {exc}")
                    return
                print(f"[vp] saved {save_context} -> {paths['npz']}")

            future.add_done_callback(_log_saved)
        return future

    def _load_required_modeling_volume_profile_state(self, bootstrap_df):
        history_last_opened = pd.Timestamp(bootstrap_df["Opened"].iloc[-1])
        try:
            state = load_volume_profile_state(self.volume_profile_modeling_state_path)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "volume profile live state initialization failed: missing modeling-end "
                f"state {self.volume_profile_modeling_state_path.with_suffix('.npz')}"
            ) from exc

        if not volume_profile_state_matches_config(state, self.volume_profile_cfg):
            raise RuntimeError(
                "volume profile live state initialization failed: modeling-end state "
                "config does not match active volume_profile_fixed_range config."
            )

        last_candle_ts = self._volume_profile_state_last_candle_timestamp(state)
        if last_candle_ts is None:
            raise RuntimeError(
                "volume profile live state initialization failed: modeling-end state "
                "is missing last_candle_time."
            )
        if last_candle_ts > history_last_opened:
            raise RuntimeError(
                "volume profile modeling-end state is ahead of bootstrap history "
                f"({last_candle_ts.isoformat()} > {history_last_opened.isoformat()})"
            )
        return state

    def _sync_volume_profile_state_with_history(self, history_df):
        if not self.volume_profile_enabled or history_df.empty:
            return

        sync_df = history_df.loc[:, ["Opened", "High", "Low", "Volume"]].copy()
        last_candle_ts = self._volume_profile_state_last_candle_timestamp(
            self.volume_profile_state
        )
        if last_candle_ts is None:
            raise RuntimeError(
                "volume profile state is missing last_candle_time. "
                "Live VP requires a saved modeling-end state plus incremental catch-up."
            )
        first_history_opened = pd.Timestamp(sync_df["Opened"].iloc[0])
        gap_start = last_candle_ts + INTERVAL_DELTA
        gap_end = first_history_opened - INTERVAL_DELTA
        if gap_start <= gap_end:
            gap_df = fetch_closed_ohlcv_range(
                self.session,
                start_opened=gap_start,
                end_opened=gap_end,
            )
            if gap_df.empty:
                raise RuntimeError(
                    "volume profile catch-up gap is missing candles "
                    f"from {gap_start.isoformat()} to {gap_end.isoformat()}"
                )
            gap_opened = gap_df["Opened"]
            if (
                    pd.Timestamp(gap_opened.iloc[0]) != gap_start
                    or pd.Timestamp(gap_opened.iloc[-1]) != gap_end
                    or not gap_opened.diff().iloc[1:].eq(INTERVAL_DELTA).all()
            ):
                raise RuntimeError(
                    "volume profile catch-up gap is not contiguous "
                    f"for range {gap_start.isoformat()} -> {gap_end.isoformat()}"
                )
            sync_df = pd.concat(
                [
                    gap_df.loc[:, ["Opened", "High", "Low", "Volume"]],
                    sync_df,
                ],
                ignore_index=True,
            )
        sync_df = sync_df.loc[sync_df["Opened"] > last_candle_ts]

        if sync_df.empty:
            if not self.volume_profile_state_path.with_suffix(".npz").exists():
                self._save_runtime_volume_profile_state(
                    log=True, context="runtime state"
                )
            return

        print(f"[vp] catch-up state with {len(sync_df)} candles")
        high = sync_df["High"].to_numpy(dtype=np.float64, copy=False)
        low = sync_df["Low"].to_numpy(dtype=np.float64, copy=False)
        volume = sync_df["Volume"].to_numpy(dtype=np.float64, copy=False)

        for row_idx in range(len(sync_df)):
            update_volume_profile_state_with_candle(
                self.volume_profile_state,
                high=float(high[row_idx]),
                low=float(low[row_idx]),
                volume=float(volume[row_idx]),
            )

        self.volume_profile_state["last_candle_time"] = str(
            pd.Timestamp(sync_df["Opened"].iloc[-1]).isoformat()
        )
        self._save_runtime_volume_profile_state(log=True, context="runtime state")

    def _initialize_volume_profile_state(self, bootstrap_df):
        self.volume_profile_state = self._load_required_modeling_volume_profile_state(
            bootstrap_df
        )
        print(
            "[vp] loaded modeling-end state -> "
            f"{self.volume_profile_modeling_state_path.with_suffix('.npz')}"
        )
        self._sync_volume_profile_state_with_history(bootstrap_df)

    def _extract_volume_profile_features_for_latest_candle(self):
        if not self.volume_profile_enabled:
            return {}
        latest_ohlcv = self.ohlcv_np[-1, :]
        return extract_features_from_state(
            self.volume_profile_state,
            high=float(latest_ohlcv[1]),
            low=float(latest_ohlcv[2]),
        )

    def _prepare_volume_profile_features_for_latest_candle(self, opened):
        if not self.volume_profile_enabled:
            return {}

        opened = pd.Timestamp(opened)
        last_candle_ts = self._volume_profile_state_last_candle_timestamp(
            self.volume_profile_state
        )
        if last_candle_ts is None or last_candle_ts < opened:
            self._update_volume_profile_state_for_latest_candle(opened)
        elif last_candle_ts > opened:
            raise RuntimeError(
                "volume profile state is ahead of the latest closed candle "
                f"({last_candle_ts.isoformat()} > {opened.isoformat()})"
            )

        return self._extract_volume_profile_features_for_latest_candle()

    def _update_volume_profile_state_for_latest_candle(self, opened, *, persist=True):
        if not self.volume_profile_enabled:
            return
        latest_ohlcv = self.ohlcv_np[-1, :]
        update_volume_profile_state_with_candle(
            self.volume_profile_state,
            high=float(latest_ohlcv[1]),
            low=float(latest_ohlcv[2]),
            volume=float(latest_ohlcv[4]),
        )
        self.volume_profile_state["last_candle_time"] = str(
            pd.Timestamp(opened).isoformat()
        )
        if persist:
            self._save_runtime_volume_profile_state_async()

    def _sync_closed_candles_from_rest(self, stop_before_opened=None):
        if not self.opened_candles:
            return 0

        start_opened = self.opened_candles[-1] + INTERVAL_DELTA
        end_opened = None
        if stop_before_opened is not None:
            end_opened = pd.Timestamp(stop_before_opened) - INTERVAL_DELTA
            if end_opened < start_opened:
                return 0

        catchup_df = fetch_closed_ohlcv_range(
            self.session,
            start_opened=start_opened,
            end_opened=end_opened,
        )
        if catchup_df.empty:
            return 0

        basis_futures_close_values = None
        if self.basis_premium_feature_columns:
            if self.basis_futures_close_col not in catchup_df.columns:
                raise RuntimeError(
                    "REST catch-up is missing basis futures close column "
                    f"{self.basis_futures_close_col!r}."
                )
            basis_futures_close_values = catchup_df[
                self.basis_futures_close_col
            ].to_numpy(dtype=np.float64, copy=False)

        added = 0
        for row_idx, row in enumerate(catchup_df.itertuples(index=False)):
            opened = pd.Timestamp(row.Opened)
            if opened <= self.opened_candles[-1]:
                continue
            self._append_new_candle(
                opened,
                (row.Open, row.High, row.Low, row.Close, row.Volume),
                basis_futures_close=(
                    None
                    if basis_futures_close_values is None
                    else basis_futures_close_values[row_idx]
                ),
            )
            self._update_volume_profile_state_for_latest_candle(
                opened,
                persist=False,
            )
            added += 1

        if added > 0:
            self._save_runtime_volume_profile_state_async()
            print(
                "[sync] caught_up_closed_candles="
                f"{added} first={catchup_df['Opened'].iloc[0].isoformat()} "
                f"last={catchup_df['Opened'].iloc[-1].isoformat()}"
            )
        return added

    def _maybe_sync_missing_candles(self, stop_before_opened):
        if not self.opened_candles or stop_before_opened is None:
            return 0

        expected_next = self.opened_candles[-1] + INTERVAL_DELTA
        if pd.Timestamp(stop_before_opened) > expected_next:
            return self._sync_closed_candles_from_rest(
                stop_before_opened=stop_before_opened
            )
        return 0

    def _build_feature_vector(self, volume_profile_values=None):
        latest_ohlcv = self.ohlcv_np[-1, :]
        opened_values = tuple(self.opened_candles)
        values = {
            "Open": float(latest_ohlcv[0]),
            "High": float(latest_ohlcv[1]),
            "Low": float(latest_ohlcv[2]),
            "Close": float(latest_ohlcv[3]),
            "Volume": float(latest_ohlcv[4]),
        }
        if self.candle_derived_feature_columns:
            values.update(
                build_latest_candle_derived_feature_dict_fast(
                    opened_values=opened_values,
                    opened_ns_values=self.opened_ns_np,
                    open_values=self.ohlcv_np[:, 0],
                    high_values=self.ohlcv_np[:, 1],
                    low_values=self.ohlcv_np[:, 2],
                    close_values=self.ohlcv_np[:, 3],
                    volume_values=self.ohlcv_np[:, 4],
                    feature_cols=self.candle_derived_feature_columns,
                )
            )
        if self.candle_pattern_feature_columns:
            values.update(
                build_latest_candle_pattern_feature_dict_fast(
                    opened_values=opened_values,
                    opened_ns_values=self.opened_ns_np,
                    open_values=self.ohlcv_np[:, 0],
                    high_values=self.ohlcv_np[:, 1],
                    low_values=self.ohlcv_np[:, 2],
                    close_values=self.ohlcv_np[:, 3],
                    pattern_cols=self.candle_pattern_feature_columns,
                )
            )
        if self.streak_interval_to_rule:
            values.update(
                build_latest_candle_streak_feature_dict_fast(
                    opened_values=opened_values,
                    opened_ns_values=self.opened_ns_np,
                    open_values=self.ohlcv_np[:, 0],
                    close_values=self.ohlcv_np[:, 3],
                    interval_to_rule=self.streak_interval_to_rule,
                )
            )
        if self.session_feature_columns:
            values.update(
                build_latest_session_open_feature_dict_fast(
                    latest_opened=opened_values[-1],
                    feature_cols=self.session_feature_columns,
                )
            )
        if self.realized_volatility_state is not None:
            values.update(self.latest_realized_volatility_values)
        if self.basis_premium_feature_columns:
            values.update(self._build_latest_basis_premium_features())

        if volume_profile_values:
            values.update(volume_profile_values)

        indicator_nan_cols = []
        indicator_full_history_scratch = None
        indicator_window_scratch_by_len = None
        if self.indicator_specs:
            indicator_full_history_scratch = IndicatorFullHistoryScratch(self.ohlcv_np)
            indicator_window_scratch_by_len = {}
        for spec in self.indicator_specs:
            raw_value = self._compute_latest_indicator_value(
                spec,
                indicator_full_history_scratch,
                indicator_window_scratch_by_len,
            )
            values[spec.feature_col] = raw_value
            if not np.isfinite(raw_value):
                indicator_nan_cols.append(spec.feature_col)

        self.last_indicator_nan_cols = indicator_nan_cols

        vector = np.empty((1, len(self.feature_columns)), dtype=np.float64)
        for i, col in enumerate(self.feature_columns):
            vector[0, i] = float(values.get(col, np.nan))
        return vector

    def _build_policy_intent(self, proba_up):
        bankroll = float(self.live_bankroll_usdc)
        if bankroll <= 0.0:
            return {
                "decision": "no_trade",
                "trade_side": "none",
                "final_reason": "bankroll_non_positive",
                "policy_reason": "bankroll_non_positive",
            }

        if str(self.trade_policy_runtime.get("mode", "ev")).lower() == "model_direction_min_stake":
            return {
                "proba_up": float(proba_up),
                "decision": "no_trade",
                "trade_side": "none",
                "bet_usdc": 0.0,
                "stake_multiplier": float(self.trade_policy_runtime["stake_multiplier"]),
                "stake_multiplier_mode": self.trade_policy_runtime.get(
                    "stake_multiplier_mode", "fixed"
                ),
                "required_stake_usdc": float("nan"),
                "effective_stake_usdc": float("nan"),
                "entry_price": float("nan"),
                "entry_fee_usdc": 0.0,
                "entry_fee_raw_usdc": 0.0,
                "shares_net": 0.0,
                "final_reason": "market_quotes_required_for_min_stake",
                "policy_reason": "market_quotes_required_for_min_stake",
                "price_eps": float("nan"),
                "price_slip": float("nan"),
            }

        policy_result = decide_trade_from_ev(
            float(proba_up),
            None,
            None,
            None,
            None,
            float(self.trade_policy_runtime["extra_buffer"]),
        )
        intent = build_trade_intent(
            policy_result=policy_result,
            bankroll=float(bankroll),
            stake_multiplier=float(self.trade_policy_runtime["stake_multiplier"]),
            fee_model=self.trade_policy_runtime["fee_model"],
            stake_multiplier_mode=self.trade_policy_runtime.get(
                "stake_multiplier_mode", "fixed"
            ),
            initial_bankroll=float(LIVE_INITIAL_BANKROLL_USDC),
            return_multiple_balance=float(bankroll),
        )
        intent["price_eps"] = float("nan")
        intent["price_slip"] = float("nan")
        return intent

    def _upsert_closed_candle(self, kline):
        opened = pd.to_datetime(int(kline["t"]), unit="ms", utc=True)
        ohlcv = (
            float(kline["o"]),
            float(kline["h"]),
            float(kline["l"]),
            float(kline["c"]),
            float(kline["v"]),
        )
        basis_futures_close = self._basis_futures_close_from_live_candle(kline, ohlcv)

        if self.opened_candles:
            last_opened = self.opened_candles[-1]
            if opened < last_opened:
                return None
            if opened == last_opened:
                self.ohlcv_np[-1, :] = np.asarray(ohlcv, dtype=np.float64)
                self.candle_open_close[opened] = (float(ohlcv[0]), float(ohlcv[3]))
                if self.basis_premium_feature_columns:
                    self.basis_futures_close_np[-1] = basis_futures_close
                return opened

        self._append_new_candle(
            opened,
            ohlcv,
            basis_futures_close=basis_futures_close,
        )
        return opened

    def _settlement_http_session(self):
        session = getattr(self, "pm_session", None)
        return session if session is not None else self.session

    def _resolve_polymarket_market_slug_for_record(self, rec):
        return resolve_polymarket_market_slug(
            rec.get("bucket_start"),
            market_slug=rec.get("pm_market_slug", ""),
        )

    def _refresh_polymarket_markets(self, records):
        pending = [rec for rec in records if rec.get("actual_up") is None]
        pending_slugs = set()
        for rec in pending:
            slug = self._resolve_polymarket_market_slug_for_record(rec)
            if slug:
                pending_slugs.add(slug)
        self.settlement_market_cache = {
            slug: entry
            for slug, entry in self.settlement_market_cache.items()
            if slug in pending_slugs
        }
        if not pending_slugs:
            return 0

        refreshed = 0
        for slug in sorted(pending_slugs):
            cached = self.settlement_market_cache.get(slug)
            if cached is not None and cached.get("resolved"):
                continue

            try:
                market = fetch_polymarket_market_by_slug(
                    self._settlement_http_session(),
                    slug,
                )
            except Exception as exc:
                print(f"[settlement] polymarket refresh failed market={slug}: {exc}")
                continue

            actual_up = resolve_polymarket_actual_up_from_market_payload(market)
            summary_key = (
                bool(market.get("closed", False)),
                bool(market.get("acceptingOrders", False)),
                str(market.get("umaResolutionStatus", "") or "").strip().lower(),
                None if actual_up is None else int(actual_up),
            )
            if cached is None or cached.get("summary_key") != summary_key:
                print(
                    "[settlement] refreshed "
                    f"source=polymarket market={slug} "
                    f"closed={summary_key[0]} "
                    f"accepting_orders={summary_key[1]} "
                    f"uma_status={summary_key[2] or 'n/a'} "
                    f"resolved={actual_up is not None}"
                )
            self.settlement_market_cache[slug] = {
                "market": market,
                "actual_up": None if actual_up is None else int(actual_up),
                "resolved": actual_up is not None,
                "summary_key": summary_key,
            }
            refreshed += 1
        return refreshed

    def _resolve_record_outcome_from_polymarket_market(self, rec, resolved_at):
        market_slug = self._resolve_polymarket_market_slug_for_record(rec)
        if not market_slug:
            return False

        cache_entry = self.settlement_market_cache.get(market_slug)
        if not cache_entry or not cache_entry.get("resolved"):
            return False

        actual_up = int(cache_entry["actual_up"])
        rec["actual_up"] = actual_up
        rec["is_correct"] = resolve_record_accuracy_from_side(
            rec,
            actual_up=actual_up,
        )
        rec["resolved_at"] = resolved_at
        if "pm_market_slug" in rec and not rec.get("pm_market_slug"):
            rec["pm_market_slug"] = market_slug
        return True

    def _resolve_record_outcome_from_settlement_truth(self, rec, resolved_at):
        return self._resolve_record_outcome_from_polymarket_market(
            rec,
            resolved_at=resolved_at,
        )

    def _on_open(self, ws):
        label = getattr(ws, "_target_label", "")
        url = getattr(ws, "_target_url", "")
        print(f"[ws] connected: {label or url}")

    def _on_error(self, ws, error):
        label = getattr(ws, "_target_label", "")
        print(f"[ws] error [{label}]: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        label = getattr(ws, "_target_label", "")
        print(f"[ws] closed [{label}]: code={close_status_code}, msg={close_msg}")

    def _run_websocket_target_once(self, target):
        def on_open(ws):
            ws._target_label = target["label"]
            ws._target_url = target["url"]
            self._on_open(ws)

        def on_message(ws, message):
            wrapped_payload = {
                "stream": target["stream_name"],
                "data": _load_ws_payload(message),
            }
            self._on_message(ws, wrapped_payload)

        ws = WebSocketApp(
            target["url"],
            on_open=on_open,
            on_message=on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        ws.run_forever(
            ping_interval=WS_PING_INTERVAL_SEC,
            ping_timeout=WS_PING_TIMEOUT_SEC,
        )

    def _run_websocket_target_forever(self, target):
        delay = 1
        while True:
            try:
                self._run_websocket_target_once(target)
            except Exception as exc:
                print(f"[ws] run failed [{target['label']}]: {exc}")

            print(f"[ws] reconnect [{target['label']}] in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, MAX_WS_RECONNECT_DELAY_SEC)

    def _run_all_websocket_targets_forever(self):
        if not WS_TARGETS:
            raise RuntimeError("No websocket targets configured.")
        for target in WS_TARGETS[1:]:
            thread = threading.Thread(
                target=self._run_websocket_target_forever,
                args=(target,),
                daemon=True,
                name=f"ws-{target['role']}-{target['market_type']}",
            )
            thread.start()
        self._run_websocket_target_forever(WS_TARGETS[0])


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
POLYMARKET_POST_ORDER_RETRY_INITIAL_DELAY_SEC = 0.25
POLYMARKET_POST_ORDER_RETRY_MAX_DELAY_SEC = 1.0

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
        tick_size=None,
        slippage_ticks=0,
):
    entry_price_value = _safe_float(entry_price)
    if not np.isfinite(entry_price_value) or not (0.0 < entry_price_value < 1.0):
        return float("nan"), "invalid_entry_price"

    submitted_price_mode_text = _safe_text(submitted_price_mode).lower()
    if submitted_price_mode_text in {"", "entry_price"}:
        return float(entry_price_value), ""

    order_price_cap_value = _safe_float(order_price_cap)
    if not np.isfinite(order_price_cap_value) or not (0.0 < order_price_cap_value < 1.0):
        return float("nan"), "invalid_order_price_cap"
    if entry_price_value > order_price_cap_value + 1e-12:
        return float("nan"), "entry_price_above_order_price_cap"

    if submitted_price_mode_text == "entry_price_plus_ticks":
        tick_size_value = _safe_float(tick_size)
        try:
            slippage_tick_count = int(slippage_ticks)
        except (TypeError, ValueError):
            return float("nan"), "invalid_submitted_price_slippage_ticks"
        if not np.isfinite(tick_size_value) or tick_size_value <= 0.0:
            return float("nan"), "invalid_tick_size"
        if slippage_tick_count < 0:
            return float("nan"), "invalid_submitted_price_slippage_ticks"
        submitted_price = entry_price_value + slippage_tick_count * tick_size_value
        submitted_price = min(submitted_price, order_price_cap_value)
        submitted_price = round(float(submitted_price), 4)
        if not (0.0 < submitted_price < 1.0):
            return float("nan"), "invalid_submitted_price"
        return float(submitted_price), ""

    if submitted_price_mode_text != "order_price_cap":
        return float("nan"), f"unsupported_submitted_price_mode:{submitted_price_mode_text}"

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
    intended_stake = _safe_float(record.get("intended_stake_usdc"))
    if not np.isfinite(intended_stake):
        intended_stake = _safe_float(record.get("effective_stake_usdc"))
        if not np.isfinite(intended_stake):
            intended_stake = _safe_float(record.get("entry_stake_usdc_orig"))
        record["intended_stake_usdc"] = (
            float(intended_stake) if np.isfinite(intended_stake) else np.nan
        )
    submitted_stake = _safe_float(record.get("submitted_stake_usdc"))
    if not np.isfinite(submitted_stake):
        order_status = _safe_text(record.get("pm_order_status"))
        attempted_statuses = set(POLYMARKET_SUBMITTED_ORDER_STATUSES) | {
            POLYMARKET_RETRYABLE_SUBMISSION_STATUS,
            "submission_error",
            "submission_rejected",
        }
        record["submitted_stake_usdc"] = (
            float(record["intended_stake_usdc"])
            if order_status in attempted_statuses
               and np.isfinite(_safe_float(record.get("intended_stake_usdc")))
            else np.nan
        )
    if not np.isfinite(_safe_float(record.get("filled_stake_usdc"))):
        if np.isfinite(current_stake):
            record["filled_stake_usdc"] = float(current_stake)
        elif np.isfinite(stake_from_response):
            record["filled_stake_usdc"] = float(stake_from_response)
        else:
            record["filled_stake_usdc"] = np.nan
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
    record.setdefault("signal_ready_delay_ms", np.nan)
    record.setdefault("decision_ready_delay_ms", np.nan)
    record.setdefault("cycle_complete_delay_ms", np.nan)
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
    record.setdefault("pm_submitted_price", np.nan)
    record.setdefault("pm_submitted_price_mode", "")
    record.setdefault("pm_submitted_price_slippage_ticks", np.nan)
    record.setdefault("pm_submitted_price_error", "")
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
        return float(balance_int / (10 ** POLYMARKET_COLLATERAL_DECIMALS))
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
        if TRADING_IMPORT_ERROR is not None:
            raise ImportError(
                "Missing live trading dependency. Install requirements.txt before "
                "running run.py."
            ) from TRADING_IMPORT_ERROR
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
            {"User-Agent": "5min_up_down_pred/run.py"}
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
        chunk_size = 20  # bezpiecznie, ĹĽeby query string nie urĂłsĹ‚ za bardzo
        for i in range(0, len(condition_ids), chunk_size):
            chunk = condition_ids[i: i + chunk_size]
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
        return polymarket_market_slug_matches_prefix(
            slug, self.pm_cfg.market_slug_prefix
        )

    def _is_managed_record(self, record):
        return polymarket_market_slug_matches_prefix(
            record.get("pm_market_slug"), self.pm_cfg.market_slug_prefix
        )

    def _tracked_condition_ids(self):
        condition_ids = set()
        for rec in self._records_snapshot():
            if str(rec.get("pm_mode", "")) != "live":
                continue
            if not self._is_managed_record(rec):
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
               and self._is_managed_record(rec)
               and _safe_text(rec.get("pm_selected_token_id"))
        }

        candidates = []
        for pos in open_positions:
            if not self._is_managed_position(pos):
                continue
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
                    "market_slug": _safe_text(pos.get("slug") or pos.get("eventSlug")),
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
        if not polymarket_market_slug_matches_prefix(
                candidate.get("market_slug"), self.pm_cfg.market_slug_prefix
        ):
            print(
                "[pm] exit skip | "
                f"reason=unmanaged_market asset={asset} "
                f"marketSlug={candidate.get('market_slug', '')}"
            )
            return
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
                if not self._is_managed_record(rec):
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
                    settlement = resolve_polymarket_closed_position_settlement(
                        rec,
                        closed_pos,
                        prefer_data_api_pnl=_is_polymarket_submitted_status(
                            rec.get("pm_exit_order_status")
                        ),
                    )
                    payout = (
                        _safe_float(settlement.get("payout_usdc"))
                        if np.isfinite(_safe_float(settlement.get("payout_usdc")))
                        else np.nan
                    )
                    rec["pm_closed_avg_price"] = (
                        float(settlement["closed_avg_price"])
                        if np.isfinite(settlement["closed_avg_price"])
                        else np.nan
                    )
                    rec["pm_closed_total_bought_usdc"] = (
                        float(settlement["closed_total_bought"])
                        if np.isfinite(settlement["closed_total_bought"])
                        else np.nan
                    )
                    rec["pm_closed_realized_pnl_usdc"] = (
                        float(settlement["closed_realized_pnl"])
                        if np.isfinite(settlement["closed_realized_pnl"])
                        else np.nan
                    )
                    rec["pm_closed_payout_usdc"] = (
                        float(payout) if np.isfinite(payout) else np.nan
                    )
                    rec["pm_settlement_payout_source"] = settlement["payout_source"]
                    if np.isfinite(settlement["stake_usdc"]):
                        rec["stake_usdc"] = float(settlement["stake_usdc"])
                    if np.isfinite(settlement["closed_avg_price"]):
                        rec["entry_price"] = float(settlement["closed_avg_price"])
                    if np.isfinite(settlement["shares_net"]):
                        rec["shares_net"] = float(settlement["shares_net"])
                    if settlement["trade_is_win"] is not None:
                        rec["trade_is_win"] = settlement["trade_is_win"]
                    rec["payout_usdc"] = (
                        float(payout) if np.isfinite(payout) else rec.get("payout_usdc")
                    )
                    rec["pnl_usdc"] = (
                        float(settlement["pnl_usdc"])
                        if np.isfinite(settlement["pnl_usdc"])
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
        return_multiple_balance = (
            float(self.pm_cash_balance_usdc)
            if np.isfinite(self.pm_cash_balance_usdc)
            else float(self.live_bankroll_usdc)
        )
        intent = build_trade_intent(
            policy_result=policy_result,
            bankroll=float(self.live_bankroll_usdc),
            stake_multiplier=float(self.live_trade_policy["stake_multiplier"]),
            fee_model=market.fee_model,
            order_min_size=float(market.order_min_size),
            external_stake_cap_usdc=float(self.pm_cfg.max_exposure_usdc),
            stake_multiplier_mode=self.live_trade_policy.get(
                "stake_multiplier_mode", "fixed"
            ),
            initial_bankroll=float(self.pm_cfg.start_bankroll_usdc),
            return_multiple_balance=float(return_multiple_balance),
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
        intent["submitted_price_mode"] = str(
            self.live_trade_policy.get("submitted_price_mode", "entry_price")
        )
        intent["submitted_price_slippage_ticks"] = int(
            self.live_trade_policy.get("submitted_price_slippage_ticks", 0)
        )
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
            tick_size=market.tick_size,
            slippage_ticks=self.live_trade_policy.get(
                "submitted_price_slippage_ticks", 0
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
            submitted_stake_usdc=np.nan,
            filled_stake_usdc=np.nan,
            filled_shares=np.nan,
    ):
        submitted_stake_value = _safe_float(submitted_stake_usdc)
        filled_stake_value = _safe_float(filled_stake_usdc)
        filled_shares_value = _safe_float(filled_shares)
        return {
            "commit_bankroll": bool(commit_bankroll),
            "status": str(status),
            "error": str(error),
            "response_text": str(response_text),
            "submitted_stake_usdc": (
                float(submitted_stake_value)
                if np.isfinite(submitted_stake_value)
                else np.nan
            ),
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
        attempted_stake_usdc = np.nan
        try:
            if intent.get("final_reason") != "ok":
                return self._submit_result(
                    commit_bankroll=False,
                    status="skipped",
                    submitted_stake_usdc=0.0,
                )
            if self.pm_cfg.paper_mode:
                return self._submit_result(
                    commit_bankroll=True,
                    status="paper_intent",
                    submitted_stake_usdc=float(intent.get("bet_usdc", 0.0) or 0.0),
                )
            if self.pm_cfg.disable_order_submission:
                return self._submit_result(
                    commit_bankroll=False,
                    status="submission_disabled",
                    submitted_stake_usdc=0.0,
                )
            if self.pm_client is None:
                return self._submit_result(
                    commit_bankroll=False,
                    status="client_unavailable",
                    error="pm_client_not_initialized",
                    submitted_stake_usdc=0.0,
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
                attempted_stake_usdc = float(intent["bet_usdc"])
                order = MarketOrderArgs(
                    token_id=str(intent["token_id"]),
                    amount=attempted_stake_usdc,
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
                    submitted_stake_usdc=attempted_stake_usdc,
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
                submitted_stake_usdc=attempted_stake_usdc,
                filled_stake_usdc=filled_stake_usdc,
                filled_shares=filled_shares,
            )
        except Exception as exc:
            return self._submit_result(
                commit_bankroll=False,
                status=_submission_error_status_from_exception(exc),
                error=str(exc),
                submitted_stake_usdc=attempted_stake_usdc,
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
        decision_ready_delay_ms = np.nan
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
            decision_ready_delay_ms = _delay_ms_since(bucket_start)
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
            "decision_ready_delay_ms": float(decision_ready_delay_ms),
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
        intended_stake_usdc = _safe_float(intent.get("bet_usdc"), 0.0)
        submitted_stake_usdc = _safe_float(
            submit_result.get("submitted_stake_usdc"),
            np.nan,
        )
        filled_stake_usdc = _safe_float(buy_record_fields["stake_usdc"], 0.0)

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
            "intended_stake_usdc": float(intended_stake_usdc),
            "submitted_stake_usdc": (
                float(submitted_stake_usdc)
                if np.isfinite(submitted_stake_usdc)
                else np.nan
            ),
            "filled_stake_usdc": float(filled_stake_usdc),
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
            "pm_submitted_price": float(intent.get("submitted_price", np.nan)),
            "pm_submitted_price_mode": str(
                intent.get("submitted_price_mode", "")
            ),
            "pm_submitted_price_slippage_ticks": float(
                intent.get("submitted_price_slippage_ticks", np.nan)
            ),
            "pm_submitted_price_error": str(
                intent.get("submitted_price_error", "")
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
        submitted_stake_usdc = _safe_float(
            submit_result.get("submitted_stake_usdc"),
            np.nan,
        )
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
            "intended_stake_usdc": float(_safe_float(intent.get("bet_usdc"), 0.0)),
            "submitted_stake_usdc": (
                float(submitted_stake_usdc)
                if np.isfinite(submitted_stake_usdc)
                else np.nan
            ),
            "filled_stake_usdc": float(stake_usdc),
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
            "pm_submitted_price": float(intent.get("submitted_price", np.nan)),
            "pm_submitted_price_mode": str(intent.get("submitted_price_mode", "")),
            "pm_submitted_price_slippage_ticks": float(
                intent.get("submitted_price_slippage_ticks", np.nan)
            ),
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
        latency_metrics["signal_ready_delay_ms"] = _delay_ms_since(bucket_start)
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
        latency_metrics["decision_ready_delay_ms"] = float(
            execution["decision_ready_delay_ms"]
        )
        latency_metrics["market_prefetch_hit"] = bool(execution["market_prefetch_hit"])
        latency_metrics["market_prefetch_age_ms"] = float(
            execution["market_prefetch_age_ms"]
        )
        latency_metrics["market_lookup_source"] = str(execution["market_lookup_source"])
        latency_metrics["market_lookup_ms"] = float(execution["market_lookup_ms"])
        latency_metrics["submit_order_ms"] = float(execution["submit_order_ms"])
        latency_metrics["execution_ms"] = float(execution["execution_ms"])
        latency_metrics["cycle_complete_delay_ms"] = float(decision_delay_ms)
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

    def _format_number(self, value, decimals=2):
        value = _safe_float(value)
        return "n/a" if not np.isfinite(value) else f"{value:.{int(decimals)}f}"

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
        policy_signals = sum(
            1
            for rec in records
            if _safe_text(rec.get("policy_decision")) in {"buy_yes", "buy_no"}
        )
        policy_no_trade = sum(
            1
            for rec in records
            if _safe_text(rec.get("policy_decision")) == "no_trade"
        )
        order_attempt_statuses = set(POLYMARKET_SUBMITTED_ORDER_STATUSES) | {
            POLYMARKET_RETRYABLE_SUBMISSION_STATUS,
            "submission_error",
            "submission_rejected",
        }
        order_attempts = sum(
            1
            for rec in records
            if _safe_text(rec.get("pm_order_status")) in order_attempt_statuses
        )
        order_submitted = sum(
            1
            for rec in records
            if _is_polymarket_submitted_status(rec.get("pm_order_status"))
        )
        order_filled = sum(
            1
            for rec in records
            if _safe_text(rec.get("policy_decision")) in {"buy_yes", "buy_no"}
            and np.isfinite(_safe_float(rec.get("stake_usdc")))
            and _safe_float(rec.get("stake_usdc")) > 0.0
        )
        order_failed_425 = sum(
            1
            for rec in records
            if _safe_text(rec.get("pm_order_status"))
            == POLYMARKET_RETRYABLE_SUBMISSION_STATUS
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
            "policy_signals": policy_signals,
            "policy_no_trade": policy_no_trade,
            "order_attempts": order_attempts,
            "order_submitted": order_submitted,
            "order_filled": order_filled,
            "order_failed_425": order_failed_425,
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
            "orders",
            [
                ("signals", stats["policy_signals"]),
                ("no_trade", stats["policy_no_trade"]),
                ("attempts", stats["order_attempts"]),
                ("submitted", stats["order_submitted"]),
                ("filled", stats["order_filled"]),
                ("failed_425", stats["order_failed_425"]),
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
                    (
                        "stake_intended",
                        self._format_number(pred.get("intended_stake_usdc"), 2),
                    ),
                    (
                        "stake_submitted",
                        self._format_number(pred.get("submitted_stake_usdc"), 2),
                    ),
                    (
                        "stake_filled",
                        self._format_number(pred.get("filled_stake_usdc"), 2),
                    ),
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
                    (
                        "submitted_price",
                        self._format_number(pred.get("pm_submitted_price"), 4),
                    ),
                ],
            )
            execution_fields = [
                ("pm_order_status", pred["pm_order_status"]),
                ("policy_reason", pred["policy_reason"]),
                (
                    "signal_ready_ms",
                    self._format_number(pred.get("signal_ready_delay_ms"), 0),
                ),
                (
                    "decision_ready_ms",
                    self._format_number(pred.get("decision_ready_delay_ms"), 0),
                ),
                (
                    "cycle_complete_ms",
                    self._format_number(pred.get("cycle_complete_delay_ms"), 0),
                ),
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
            f"basis_features={len(self.basis_premium_feature_columns)} "
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
            f"stake_multiplier_mode={self.live_trade_policy.get('stake_multiplier_mode', 'fixed')} "
            f"extra_buffer={self.live_trade_policy['extra_buffer']:.6f} "
            f"submitted_price_mode={self.live_trade_policy.get('submitted_price_mode', 'entry_price')} "
            f"submitted_price_slippage_ticks={self.live_trade_policy.get('submitted_price_slippage_ticks', 0)}"
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
    setup_live_console_logging(
        f"run_{SYMBOL}_{INTERVAL}",
        run_started_at_utc=RUN_STARTED_AT_UTC,
        telegram=True,
    )
    runner = PolymarketLiveTrader()

    now_utc = pd.Timestamp.now(tz="UTC")
    next_resolve = now_utc.floor(
        f"{runner.target_bucket_minutes}min"
    ) + pd.Timedelta(minutes=runner.target_bucket_minutes)
    print(f"[wait] first resolve+pred around {next_resolve.isoformat()}")
    runner.run_forever()


if __name__ == "__main__":
    main()
