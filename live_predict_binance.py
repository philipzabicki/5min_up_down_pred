import json
import hashlib
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import lightgbm as lgb
from common_config_utils import coerce_path
from live_utils import (
    LIVE_PREDICTION_EXPORT_COLUMNS,
    LIVE_SHARED_MARKET_DATA_COLUMNS,
    as_utc_timestamp,
    build_live_market_data_path,
    interval_to_floor_rule,
    interval_to_timedelta,
    upsert_records_csv,
    write_records_csv,
)
from websocket import WebSocketApp
from create_modeling_dataset import (
    parse_fit_results,
    resolve_volume_profile_modeling_state_path,
)
from project_config import (
    load_dataset_profile,
    load_live_profile,
    load_runtime_artifact_paths,
)
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

from features.ADX import get_adx_values
from features.BollingerBands import get_bollinger_bands_values
from features.ChaikinOsc import get_chaikin_oscillator_values
from features.KeltnerChannel import get_keltner_channel_values
from features.live_indicator_runtime import (
    LATEST_VALUE_BUILDERS as LIVE_LATEST_VALUE_BUILDERS,
    IndicatorFullHistoryScratch,
    IndicatorWindowScratch,
)
from features.MACD import get_macd_values
from features.realized_volatility import (
    REALIZED_VOLATILITY_FEATURE_COLUMNS,
    RealizedVolatilityRuntimeState,
)
from features.session_open_features import (
    SUPPORTED_SESSION_OPEN_FEATURE_COLS,
    build_latest_session_open_feature_dict_fast,
)
from features.StochOsc import get_stochastic_oscillator_values
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
from modeling_dataset_utils import (
    MODELING_DATASET_CONFIG_FILE,
    load_modeling_dataset_settings,
    split_feature_subset,
)
from trade_policy import (
    build_trade_intent,
    decide_trade_from_ev,
    load_trade_policy_runtime_config,
)

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
    RUNTIME_ARTIFACT_PATHS["trade_policy_runtime_config_path"]
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
POLYMARKET_MARKET_SLUG_OVERRIDE = str(
    LIVE_PROFILE.get("polymarket_market_slug_override", "")
).strip()
POLYMARKET_MARKET_REQUEST_TIMEOUT_SEC = float(
    LIVE_PROFILE["polymarket_market_request_timeout_sec"]
)
LIVE_ROOT_DIR = Path("data/live")
LIVE_PREDICTIONS_DIR = LIVE_ROOT_DIR / "predictions"
LIVE_TRADE_DIR = LIVE_ROOT_DIR / "trade"
RUN_STARTED_AT_UTC = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
PREDICTIONS_OUTPUT_PATH = Path(
    LIVE_PREDICTIONS_DIR
    / f"live_predictions_{SYMBOL}_{INTERVAL}_{RUN_STARTED_AT_UTC}.csv"
)
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
    if POLYMARKET_MARKET_SLUG_OVERRIDE:
        return POLYMARKET_MARKET_SLUG_OVERRIDE
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
            self.ohlcv_np = self.ohlcv_np[-self.max_keep :, :]
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

        self.predictions_path = PREDICTIONS_OUTPUT_PATH
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        self.market_data_path = build_live_market_data_path()
        self.market_data_path.parent.mkdir(parents=True, exist_ok=True)
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
            self.ohlcv_np = self.ohlcv_np[-self.max_keep :, :]
            self.opened_ns_np = self.opened_ns_np[-self.max_keep :]
            if self.basis_premium_feature_columns:
                self.basis_futures_close_np = self.basis_futures_close_np[
                    -self.max_keep :
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

    def _resolve_pending(self):
        if not self.records:
            return 0

        self._refresh_polymarket_markets(self.records)
        resolved_now = 0
        resolved_at = pd.Timestamp.now(tz="UTC")

        for rec in self.records:
            if rec["actual_up"] is not None:
                continue

            if not self._resolve_record_outcome_from_settlement_truth(
                rec,
                resolved_at=resolved_at,
            ):
                continue

            stake_usdc = float(rec.get("stake_usdc", 0.0) or 0.0)
            side = str(rec.get("trade_side", "none"))
            actual_up = int(rec["actual_up"])
            if stake_usdc > 0.0 and side in {"yes", "no"}:
                is_trade_win = int(
                    (side == "yes" and actual_up == 1)
                    or (side == "no" and actual_up == 0)
                )
                shares_net = float(rec.get("shares_net", 0.0) or 0.0)
                payout = float(shares_net) if is_trade_win else 0.0

                self.live_bankroll_usdc += float(payout)
                rec["trade_is_win"] = int(is_trade_win)
                rec["payout_usdc"] = float(payout)
                rec["pnl_usdc"] = float(payout - stake_usdc)
            else:
                rec["trade_is_win"] = None
                rec["payout_usdc"] = 0.0
                rec["pnl_usdc"] = 0.0

            rec["bankroll_after_resolve"] = float(self.live_bankroll_usdc)
            resolved_now += 1

        return resolved_now

    def _predict_next_bucket(self, volume_profile_values=None):
        if volume_profile_values is None:
            volume_profile_values = (
                self._prepare_volume_profile_features_for_latest_candle(
                    self.opened_candles[-1]
                )
            )
        proba_up = float(
            self.model.predict(
                self._build_feature_vector(volume_profile_values=volume_profile_values)
            )[0]
        )
        intent = self._build_policy_intent(proba_up=proba_up)
        bankroll_before_entry = float(self.live_bankroll_usdc)
        stake_usdc = (
            float(intent.get("bet_usdc", 0.0) or 0.0)
            if str(intent.get("final_reason", "")) == "ok"
            else 0.0
        )
        if stake_usdc > 0.0:
            self.live_bankroll_usdc -= stake_usdc
        bankroll_after_entry = float(self.live_bankroll_usdc)

        minute_open = self.opened_candles[-1]
        minute_close = minute_open + pd.Timedelta(minutes=1)
        bucket_start = minute_open.floor(
            f"{self.target_bucket_minutes}min"
        ) + pd.Timedelta(minutes=self.target_bucket_minutes)
        bucket_end = bucket_start + pd.Timedelta(minutes=self.target_bucket_minutes - 1)
        btc_snapshot = self._latest_btc_snapshot()

        self.records.append(
            {
                "record_id": f"bucket:{pd.Timestamp(bucket_start).isoformat()}",
                "pm_model_hash": self.model_hash,
                "pm_policy_hash": self.trade_policy_config_hash,
                "pm_run_started_at_utc": self.run_started_at_utc,
                "prediction_time": pd.Timestamp.now(tz="UTC"),
                "bucket_start": bucket_start,
                "bucket_end": bucket_end,
                "proba_up": proba_up,
                "trade_side": str(intent.get("trade_side", "none")),
                "stake_usdc": float(stake_usdc),
                "stake_multiplier": float(intent.get("stake_multiplier", np.nan)),
                "required_stake_usdc": float(
                    intent.get("required_stake_usdc", np.nan)
                ),
                "effective_stake_usdc": float(
                    intent.get("effective_stake_usdc", np.nan)
                ),
                "entry_price": float(intent.get("entry_price", np.nan)),
                "entry_fee_usdc": float(intent.get("entry_fee_usdc", 0.0)),
                "entry_fee_raw_usdc": float(intent.get("entry_fee_raw_usdc", 0.0)),
                "shares_net": float(intent.get("shares_net", 0.0)),
                "price_eps": float(intent.get("price_eps", np.nan)),
                "price_slip": float(intent.get("price_slip", np.nan)),
                "ask_yes": float(intent.get("ask_yes", np.nan)),
                "ask_no": float(intent.get("ask_no", np.nan)),
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
            }
        )
        self.predicted_buckets.add(bucket_start)

        decision_local = minute_close.tz_convert(self.local_tz).isoformat()
        return {
            "decision_local": decision_local,
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
        }

    def _save_records(self):
        if not self.records:
            return
        for rec in self.records:
            self._backfill_bucket_price_bounds(rec)
        write_records_csv(
            self.records,
            self.predictions_path,
            export_columns=LIVE_PREDICTION_EXPORT_COLUMNS,
            is_resolved=lambda rec: rec.get("actual_up") is not None
            and rec.get("is_correct") is not None,
            is_traded=lambda rec: rec.get("actual_up") is not None
            and rec.get("trade_is_win") is not None,
        )
        upsert_records_csv(
            self.records,
            self.market_data_path,
            export_columns=LIVE_SHARED_MARKET_DATA_COLUMNS,
            is_resolved=lambda rec: rec.get("actual_up") is not None
            and rec.get("is_correct") is not None,
            is_traded=lambda rec: rec.get("actual_up") is not None
            and rec.get("trade_is_win") is not None,
            record_filter=lambda rec: str(rec.get("record_id", "")).startswith("bucket:"),
        )

    def _stats(self):
        model_resolved = 0
        model_wins = 0
        for rec in self.records:
            model_accuracy = resolve_model_accuracy_from_proba(
                rec,
                threshold=self.prediction_threshold,
            )
            if model_accuracy is None:
                continue
            model_resolved += 1
            model_wins += int(model_accuracy)
        policy_resolved = sum(
            1
            for rec in self.records
            if rec["actual_up"] is not None and rec["is_correct"] is not None
        )
        policy_resolved_wins = sum(
            int(rec["is_correct"])
            for rec in self.records
            if rec["actual_up"] is not None and rec["is_correct"] is not None
        )
        closed_trades = sum(
            1
            for rec in self.records
            if rec["actual_up"] is not None and rec["trade_is_win"] is not None
        )
        closed_trade_wins = sum(
            int(rec["trade_is_win"])
            for rec in self.records
            if rec["actual_up"] is not None and rec["trade_is_win"] is not None
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
                for rec in self.records
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

    def _print_recent_candle_buffer(self, count=5):
        if not self.opened_candles:
            return

        n = min(int(count), len(self.opened_candles))
        opened_tail = list(self.opened_candles)[-n:]
        ohlcv_tail = self.ohlcv_np[-n:, :]

        print(f"[candles] last {n} candles from buffer:")
        for idx, (opened, row) in enumerate(zip(opened_tail, ohlcv_tail), start=1):
            print(
                f"[candles] {idx}/{n} opened={opened.isoformat()} "
                f"open={float(row[0]):.2f} high={float(row[1]):.2f} "
                f"low={float(row[2]):.2f} close={float(row[3]):.2f} "
                f"volume={float(row[4]):.6f}"
            )

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

    @staticmethod
    def _format_rate(value):
        return "n/a" if not np.isfinite(value) else f"{value * 100:.2f}%"

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

        if pred is not None:
            self._print_recent_candle_buffer(count=5)

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
                            if np.isfinite(float(pred.get("required_stake_usdc", np.nan)))
                            else None
                        ),
                    ),
                    (
                        "effective_stake_usdc",
                        f"{pred['effective_stake_usdc']:.2f}",
                    ),
                    ("policy_reason", pred["policy_reason"]),
                ],
            )
        self._print_log_fields(
            "account",
            [
                ("total_pnl", f"{stats['total_pnl']:.2f}"),
                ("bankroll", f"{self.live_bankroll_usdc:.2f}"),
            ],
        )
        if pred is not None:
            self._print_indicator_nan_status()

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

    def _on_message(self, ws, message):
        with self.ws_message_lock:
            try:
                payload = _load_ws_payload(message)
                closed_candle, live_minute_opened, _event_at, _timing = (
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

                volume_profile_values = (
                    self._prepare_volume_profile_features_for_latest_candle(opened)
                )

                resolved_now = self._resolve_pending()

                bucket_start = opened.floor(f"{self.target_bucket_minutes}min")
                bucket_end = bucket_start + pd.Timedelta(
                    minutes=self.target_bucket_minutes - 1
                )
                pred = None

                if opened == bucket_end:
                    next_bucket_start = bucket_start + pd.Timedelta(
                        minutes=self.target_bucket_minutes
                    )
                    if next_bucket_start not in self.predicted_buckets:
                        pred = self._predict_next_bucket(
                            volume_profile_values=volume_profile_values
                        )

                self.last_processed_closed_opened = opened

                if resolved_now > 0 or pred is not None:
                    self._save_records()
                    self._log("resolve+pred" if pred else "resolve", pred=pred)
            except Exception as exc:
                print(f"[pred] message handling failed: {exc}")

    def run_forever(self):
        print(
            "Starting live predictor | "
            f"price_symbol={SYMBOL} price_market={PRICE_MARKET} "
            f"volume_symbol={VOLUME_SYMBOL} volume_market={VOLUME_MARKET} "
            f"interval={INTERVAL} "
            f"price_source={PRICE_SOURCE} volume_source={VOLUME_SOURCE} "
            f"bootstrap_candles={len(self.opened_candles)} "
            f"target={self.target_col} "
            f"bucket_minutes={self.target_bucket_minutes} "
            f"features={len(self.feature_columns)} "
            f"streak_features={len(self.streak_interval_to_rule)} "
            f"session_features={len(self.session_feature_columns)} "
            f"basis_features={len(self.basis_premium_feature_columns)} "
            f"vp_features={len(self.volume_profile_feature_columns)}"
        )
        market_selector = (
            f"market_slug_override={POLYMARKET_MARKET_SLUG_OVERRIDE}"
            if POLYMARKET_MARKET_SLUG_OVERRIDE
            else f"market_slug_prefix={POLYMARKET_MARKET_SLUG_PREFIX}"
        )
        print(
            "Settlement source | "
            f"source=polymarket gamma_host={POLYMARKET_GAMMA_HOST} "
            f"series_slug={POLYMARKET_SERIES_SLUG} "
            f"{market_selector} "
            "rule=market_resolution"
        )
        if PRICE_SOURCE == "index" and VOLUME_SOURCE == "trade":
            print(
                "Hybrid mode uses /indexPriceKlines for OHLC, /klines for Volume, "
                "and live combines @indexPriceKline_1m with the matching trade-volume "
                "kline stream from the modeling dataset."
            )
        elif PRICE_SOURCE == "index":
            print(
                "Index-price mode uses /indexPriceKlines for history and "
                "@indexPriceKline_1m for live candles; Volume is synthetic basic-count."
            )
        print(
            "Websocket targets | "
            + ", ".join(
                f"{target['label']}->{target['url']}" for target in WS_TARGETS
            )
        )
        print(
            "Trade policy | "
            f"bankroll={self.live_bankroll_usdc:.2f} "
            f"mode={self.trade_policy_runtime.get('mode', 'ev')} "
            f"stake_multiplier={self.trade_policy_runtime['stake_multiplier']:.4f} "
            f"extra_buffer={self.trade_policy_runtime['extra_buffer']:.6f} "
            f"submitted_price_mode={self.trade_policy_runtime.get('submitted_price_mode', 'entry_price')}"
        )
        print(
            "Price/Fee model | "
            f"fee_rate={self.trade_policy_runtime['fee_model']['rate']:.6f} "
            f"fee_exp={self.trade_policy_runtime['fee_model']['exponent']:.3f}"
        )
        print("Policy inputs | live ask_yes/ask_no and side-specific fees required")
        print(f"Trade policy config: {TRADE_POLICY_CONFIG_PATH}")
        print(
            "Runtime hashes | "
            f"model={self.model_hash} "
            f"policy={self.trade_policy_config_hash} "
            f"modeling={self.modeling_dataset_config_hash}"
        )
        print(f"Predictions file: {self.predictions_path}")
        if self.volume_profile_enabled:
            print(
                f"VP modeling state path: {self.volume_profile_modeling_state_path.with_suffix('.npz')}"
            )
            print(
                f"VP runtime state path: {self.volume_profile_state_path.with_suffix('.npz')}"
            )
        self._run_all_websocket_targets_forever()


def main():
    predictor = LivePredictor()

    now_utc = pd.Timestamp.now(tz="UTC")
    next_resolve = now_utc.floor(
        f"{predictor.target_bucket_minutes}min"
    ) + pd.Timedelta(minutes=predictor.target_bucket_minutes)
    print(f"[wait] first resolve+pred around {next_resolve.isoformat()}")
    predictor.run_forever()


if __name__ == "__main__":
    main()
