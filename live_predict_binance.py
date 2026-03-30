import json
import hashlib
import os
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
from live_common import (
    LIVE_PREDICTION_EXPORT_COLUMNS,
    as_utc_timestamp,
    interval_to_floor_rule,
    interval_to_timedelta,
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
from project_env import load_repo_env
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
    SUPPORTED_SESSION_COUNTER_COLS,
    build_latest_session_counter_feature_dict_fast,
)
from features.StochOsc import get_stochastic_oscillator_values
from features.volume_profile_fixed_range import (
    FEATURE_VERSION as VP_FEATURE_VERSION,
    RUNTIME_STATE_DIR as VP_RUNTIME_STATE_DIR,
    bootstrap_state_from_history,
    extract_features_from_state,
    is_volume_profile_feature,
    load_state as load_volume_profile_state,
    normalize_config as normalize_volume_profile_config,
    save_state as save_volume_profile_state,
    state_matches_config as volume_profile_state_matches_config,
    update_state_with_candle as update_volume_profile_state_with_candle,
)
from kelly_utils import adjust_probability_for_kelly
from modeling_dataset_utils import (
    MODELING_DATASET_CONFIG_FILE,
    load_modeling_dataset_settings,
    split_feature_subset,
)


def normalize_live_market_type(value, field_name):
    normalized = str(value).strip().lower()
    if normalized not in {"spot", "um", "cm"}:
        raise ValueError(f"{field_name} must be one of {{'spot', 'um', 'cm'}}")
    return normalized

load_repo_env()
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
KELLY_CONFIG_PATH = Path(RUNTIME_ARTIFACT_PATHS["kelly_runtime_config_path"])
INDICATOR_HISTORY_REQUIREMENTS_PATH = Path(
    RUNTIME_ARTIFACT_PATHS["indicator_history_requirements_path"]
)
# Polymarket 5m up/down markets resolve from the market itself.
SETTLEMENT_SOURCE = "polymarket"
POLYMARKET_GAMMA_HOST = (
    os.getenv(
        "POLY_GAMMA_HOST",
        str(LIVE_PROFILE["polymarket_gamma_host"]),
    )
    .strip()
    .rstrip("/")
)
POLYMARKET_SERIES_SLUG = os.getenv(
    "POLY_SERIES_SLUG",
    str(LIVE_PROFILE["polymarket_series_slug"]),
).strip()
POLYMARKET_MARKET_SLUG_PREFIX = os.getenv(
    "POLY_MARKET_SLUG_PREFIX",
    str(LIVE_PROFILE["polymarket_market_slug_prefix"]),
).strip()
POLYMARKET_MARKET_SLUG_OVERRIDE = os.getenv(
    "POLY_MARKET_SLUG_OVERRIDE",
    str(LIVE_PROFILE.get("polymarket_market_slug_override", "")),
).strip()
POLYMARKET_MARKET_REQUEST_TIMEOUT_SEC = float(
    os.getenv(
        "POLY_MARKET_REQUEST_TIMEOUT_SEC",
        str(LIVE_PROFILE["polymarket_market_request_timeout_sec"]),
    ).strip()
)
LIVE_ROOT_DIR = Path("data/live")
LIVE_PREDICTIONS_DIR = LIVE_ROOT_DIR / "predictions"
LIVE_TRADE_DIR = LIVE_ROOT_DIR / "trade"
RUN_STARTED_AT_UTC = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
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
MIN_PROBA_CLIP = float(LIVE_PROFILE["min_proba_clip"])

OHLCV_COLS = list(RAW_OHLCV_COLS)
BASE_FEATURE_COLS = (
    set(OHLCV_COLS)
    | set(SUPPORTED_CANDLE_FEATURE_COLS)
    | set(SUPPORTED_SESSION_COUNTER_COLS)
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

    side = str(record.get("kelly_side", "") or "").strip().lower()
    if side == "up":
        return int(int(actual_up) == 1)
    if side == "down":
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
        return "wss://fstream.binance.com/ws"
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
INDICATOR_HISTORY_MARGIN_RATIO = float(
    os.getenv(
        "LIVE_INDICATOR_HISTORY_MARGIN_RATIO",
        str(DEFAULT_INDICATOR_HISTORY_MARGIN_RATIO),
    ).strip()
    or str(DEFAULT_INDICATOR_HISTORY_MARGIN_RATIO)
)
INDICATOR_HISTORY_MIN_EXTRA_CANDLES = int(
    os.getenv(
        "LIVE_INDICATOR_HISTORY_MIN_EXTRA_CANDLES",
        str(DEFAULT_INDICATOR_HISTORY_MIN_EXTRA_CANDLES),
    ).strip()
    or str(DEFAULT_INDICATOR_HISTORY_MIN_EXTRA_CANDLES)
)

if (
    not np.isfinite(INDICATOR_HISTORY_MARGIN_RATIO)
    or INDICATOR_HISTORY_MARGIN_RATIO < 0.0
):
    raise ValueError(
        "LIVE_INDICATOR_HISTORY_MARGIN_RATIO must be a finite number >= 0."
    )
if INDICATOR_HISTORY_MIN_EXTRA_CANDLES < 0:
    raise ValueError("LIVE_INDICATOR_HISTORY_MIN_EXTRA_CANDLES must be >= 0.")


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


def load_kelly_runtime_config(config_path):
    config_path = coerce_path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Kelly config not found: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))

    def _read_float(container, key):
        if key not in container:
            raise KeyError(f"Missing '{key}' in Kelly config: {config_path}")
        return float(container[key])

    fee_model = payload.get("fee_model")
    price_sim = payload.get("price_sim")
    cv_meta = payload.get("cv_meta", {})
    if not isinstance(cv_meta, dict):
        cv_meta = {}
    if not isinstance(fee_model, dict):
        raise ValueError(f"Malformed Kelly config, missing fee_model: {config_path}")
    if not isinstance(price_sim, dict):
        raise ValueError(f"Malformed Kelly config, missing price_sim: {config_path}")

    cfg = {
        "fractional_kelly": _read_float(payload, "fractional_kelly"),
        "cap": _read_float(payload, "cap"),
        "min_edge": _read_float(payload, "min_edge"),
        "prob_shrink": _read_float(payload, "prob_shrink"),
        "min_stake_usdc": _read_float(payload, "min_stake_usdc"),
        "fee_rate": _read_float(fee_model, "feeRate"),
        "fee_exponent": _read_float(fee_model, "exponent"),
        "fee_round_decimals": int(fee_model.get("fee_round_decimals", 4)),
        "min_fee": _read_float(fee_model, "min_fee"),
        "price_sim_model": str(price_sim.get("model", "legacy_symmetric_clip")),
        "seed": int(cv_meta.get("seed", 37)),
    }
    if cfg["price_sim_model"] == "neutral_conservative_fixed":
        cfg.update(
            {
                "ask_price": _read_float(price_sim, "ask_price"),
                "order_price_cap": float(
                    price_sim.get("order_price_cap", 0.55) or 0.55
                ),
                "order_min_size": float(
                    price_sim.get("order_min_size", 5.0) or 5.0
                ),
                "price_policy": str(price_sim.get("policy", "")),
            }
        )
    elif cfg["price_sim_model"] == "parametric_market_mid_overround":
        overround_ticks_values = price_sim.get("overround_ticks_values", [])
        overround_ticks_probs = price_sim.get("overround_ticks_probs", [])
        if len(overround_ticks_values) != len(overround_ticks_probs):
            raise ValueError(
                "Kelly config invalid: overround_ticks_values/probs length mismatch."
            )
        if not overround_ticks_values:
            raise ValueError(
                "Kelly config invalid: missing overround_ticks_values for parametric price sim."
            )
        cfg.update(
            {
                "tick_size": _read_float(price_sim, "tick_size"),
                "half_tick": _read_float(price_sim, "half_tick"),
                "mid_intercept_mean": _read_float(price_sim, "mid_intercept_mean"),
                "mid_intercept_std": _read_float(price_sim, "mid_intercept_std"),
                "mid_intercept_min": _read_float(price_sim, "mid_intercept_min"),
                "mid_intercept_max": _read_float(price_sim, "mid_intercept_max"),
                "mid_slope_mean": _read_float(price_sim, "mid_slope_mean"),
                "mid_slope_std": _read_float(price_sim, "mid_slope_std"),
                "mid_slope_min": _read_float(price_sim, "mid_slope_min"),
                "mid_slope_max": _read_float(price_sim, "mid_slope_max"),
                "mid_residual_std": _read_float(price_sim, "mid_residual_std"),
                "mid_residual_abs_q99": _read_float(
                    price_sim,
                    "mid_residual_abs_q99",
                ),
                "overround_ticks_values": [
                    int(value) for value in overround_ticks_values
                ],
                "overround_ticks_probs": [
                    float(value) for value in overround_ticks_probs
                ],
                "order_price_cap": float(
                    price_sim.get("order_price_cap", 0.55) or 0.55
                ),
                "order_min_size": float(
                    price_sim.get("order_min_size", 5.0) or 5.0
                ),
            }
        )
        total_prob = float(sum(cfg["overround_ticks_probs"]))
        if not np.isfinite(total_prob) or total_prob <= 0.0:
            raise ValueError(
                "Kelly config invalid: overround_ticks_probs must sum to a positive value."
            )
        cfg["overround_ticks_probs"] = [
            float(value / total_prob) for value in cfg["overround_ticks_probs"]
        ]
    else:
        cfg.update(
            {
                "sigma": _read_float(payload, "sigma"),
                "spread_half": _read_float(payload, "spread_half"),
                "base_price": _read_float(price_sim, "base_price"),
                "price_clip_lo": _read_float(price_sim, "price_clip_lo"),
                "price_clip_hi": _read_float(price_sim, "price_clip_hi"),
            }
        )
        if cfg["price_clip_lo"] >= cfg["price_clip_hi"]:
            raise ValueError(
                "Kelly config invalid: price_clip_lo must be < price_clip_hi"
            )
    if cfg["cap"] <= 0.0 or cfg["cap"] > 1.0:
        raise ValueError("Kelly config invalid: cap must be in (0, 1].")
    if cfg["fractional_kelly"] <= 0.0:
        raise ValueError("Kelly config invalid: fractional_kelly must be > 0.")
    if cfg["min_stake_usdc"] <= 0.0:
        raise ValueError("Kelly config invalid: min_stake_usdc must be > 0.")
    if cfg["fee_rate"] < 0.0:
        raise ValueError("Kelly config invalid: feeRate must be >= 0.")
    return cfg


def load_indicator_specs(feature_columns):
    fit_configs = parse_fit_results(FIT_RESULTS_DIR)
    fit_by_feature_col = {cfg["feature_col"]: cfg for cfg in fit_configs}

    specs = []
    missing_features = []

    for col in feature_columns:
        if (
            col in BASE_FEATURE_COLS
            or col.startswith(STREAK_FEATURE_PREFIX)
            or is_volume_profile_feature(col)
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


def _merge_price_and_volume_frames(price_df, volume_df):
    merged = (
        price_df.loc[:, ["Opened", "Open", "High", "Low", "Close"]]
        .merge(
            volume_df.loc[:, ["Opened", "Volume"]],
            on="Opened",
            how="inner",
        )
        .drop_duplicates(subset=["Opened"], keep="last")
        .sort_values("Opened")
        .reset_index(drop=True)
    )
    if merged.empty:
        return pd.DataFrame(columns=["Opened", *OHLCV_COLS])
    return merged.loc[:, ["Opened", *OHLCV_COLS]]


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
        self.kelly_config_hash = _hash_path_contents(KELLY_CONFIG_PATH)
        self.modeling_dataset_config_hash = _hash_path_contents(
            MODELING_DATASET_CONFIG_FILE
        )
        self.feature_columns = list(meta.get("feature_columns", []))
        if not self.feature_columns:
            raise ValueError("Missing feature_columns in model metadata.")
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

        self.prediction_threshold = 0.5
        self.target_col = str(meta.get("target_col", "target_5m_candle_up"))
        self.target_bucket_minutes = parse_target_bucket_minutes(self.target_col)

        self.kelly_runtime = load_kelly_runtime_config(KELLY_CONFIG_PATH)
        self.live_bankroll_usdc = LIVE_INITIAL_BANKROLL_USDC
        self.price_rng = np.random.default_rng(int(self.kelly_runtime["seed"]))
        self.price_sim_scenario = self._sample_price_sim_scenario()

        feature_parts = split_feature_subset(self.feature_columns)
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
        self.volume_profile_feature_columns = tuple(
            feature_parts["volume_profile_feature_cols"]
        )
        self.volume_profile_cfg = normalize_volume_profile_config(
            MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")
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
        self.volume_profile_state_source_path = None
        self.volume_profile_save_pool = (
            ThreadPoolExecutor(max_workers=1) if self.volume_profile_enabled else None
        )

        self.indicator_specs = load_indicator_specs(self.feature_columns)
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

    def _append_new_candle(self, opened, ohlcv):
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
        if self.realized_volatility_state is not None:
            self.latest_realized_volatility_values = self.realized_volatility_state.update(
                float(ohlcv_row[0, 3])
            )

        if len(self.opened_candles) > self.max_keep:
            dropped_opened = self.opened_candles.popleft()
            self.candle_open_close.pop(dropped_opened, None)
            self.ohlcv_np = self.ohlcv_np[-self.max_keep :, :]
            self.opened_ns_np = self.opened_ns_np[-self.max_keep :]

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

    def _store_pending_ws_price_candle(self, price_candle):
        opened = self._pending_ws_candle_opened(price_candle)
        if self.opened_candles and opened <= self.opened_candles[-1]:
            return None
        if VOLUME_SOURCE == PRICE_SOURCE:
            return price_candle

        volume = self.pending_ws_volume_by_opened.pop(opened, None)
        if volume is None:
            self.pending_ws_price_candles[opened] = price_candle
            self._cleanup_pending_ws_state(opened)
            return None

        price_candle["v"] = float(volume)
        self.pending_ws_price_candles.pop(opened, None)
        return price_candle

    def _store_pending_ws_volume(self, opened, volume):
        opened = pd.Timestamp(opened)
        if self.opened_candles and opened <= self.opened_candles[-1]:
            return None

        price_candle = self.pending_ws_price_candles.pop(opened, None)
        if price_candle is None:
            self.pending_ws_volume_by_opened[opened] = float(volume)
            self._cleanup_pending_ws_state(opened)
            return None

        price_candle["v"] = float(volume)
        return price_candle

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
            return None, None, None

        event_at = None
        raw_event_ms = data.get("E")
        if raw_event_ms not in (None, ""):
            try:
                event_at = pd.to_datetime(int(raw_event_ms), unit="ms", utc=True)
            except (TypeError, ValueError):
                event_at = None

        if stream_name == PRICE_STREAM_NAME and PRICE_SOURCE == "trade":
            kline = data.get("k", {})
            if not kline or not bool(kline.get("x", False)):
                return None, None, event_at

            opened = pd.to_datetime(int(kline["t"]), unit="ms", utc=True)
            live_minute_opened = opened + INTERVAL_DELTA
            closed_candle = self._store_pending_ws_price_candle(
                {
                    "t": int(kline["t"]),
                    "o": float(kline["o"]),
                    "h": float(kline["h"]),
                    "l": float(kline["l"]),
                    "c": float(kline["c"]),
                    "v": float(kline["v"]),
                }
            )
            return closed_candle, live_minute_opened, event_at

        if stream_name == VOLUME_STREAM_NAME and VOLUME_SOURCE == "trade":
            kline = data.get("k", {})
            if not kline or not bool(kline.get("x", False)):
                return None, None, event_at

            opened = pd.to_datetime(int(kline["t"]), unit="ms", utc=True)
            live_minute_opened = opened + INTERVAL_DELTA
            closed_candle = self._store_pending_ws_volume(
                opened,
                float(kline["v"]),
            )
            return closed_candle, live_minute_opened, event_at

        if stream_name == PRICE_STREAM_NAME and PRICE_SOURCE == "index":
            price_candle, live_minute_opened = self._extract_closed_index_price_candle(
                data
            )
            if price_candle is None:
                return None, live_minute_opened, event_at
            closed_candle = self._store_pending_ws_price_candle(price_candle)
            return closed_candle, live_minute_opened, event_at

        return None, None, event_at

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

        snapshot = dict(state)
        snapshot["horizon_names"] = tuple(state["horizon_names"])
        snapshot["half_lives"] = tuple(state["half_lives"])
        snapshot["feature_columns"] = tuple(state["feature_columns"])
        snapshot["decays"] = np.array(state["decays"], dtype=np.float64, copy=True)
        snapshot["global_scales"] = np.array(
            state["global_scales"], dtype=np.float64, copy=True
        )
        snapshot["raw_profiles"] = np.array(
            state["raw_profiles"], dtype=np.float32, copy=True
        )
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

    def _load_volume_profile_state_candidates(self, bootstrap_df):
        history_last_opened = pd.Timestamp(bootstrap_df["Opened"].iloc[-1])
        candidates = []
        for label, path in (
            ("runtime", self.volume_profile_state_path),
            ("modeling_end", self.volume_profile_modeling_state_path),
        ):
            try:
                state = load_volume_profile_state(path)
                if not volume_profile_state_matches_config(
                    state, self.volume_profile_cfg
                ):
                    raise ValueError("config mismatch")
                last_candle_ts = self._volume_profile_state_last_candle_timestamp(state)
                if last_candle_ts is not None and last_candle_ts > history_last_opened:
                    raise ValueError(
                        "state is ahead of bootstrap history "
                        f"({last_candle_ts.isoformat()} > {history_last_opened.isoformat()})"
                    )
                candidates.append((last_candle_ts, label, path, state))
            except FileNotFoundError:
                continue
            except Exception as exc:
                print(f"[vp] {label} state reload skipped: {exc}")

        if not candidates:
            return None

        def _sort_key(item):
            last_candle_ts, label, _path, _state = item
            ts_value = last_candle_ts.value if last_candle_ts is not None else -1
            label_priority = 1 if label == "runtime" else 0
            return (ts_value, label_priority)

        return max(candidates, key=_sort_key)

    def _sync_volume_profile_state_with_history(self, history_df):
        if not self.volume_profile_enabled or history_df.empty:
            return

        sync_df = history_df.loc[:, ["Opened", "High", "Low", "Volume"]].copy()
        last_candle_ts = self._volume_profile_state_last_candle_timestamp(
            self.volume_profile_state
        )
        if last_candle_ts is not None:
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
            if (
                self.volume_profile_state_source_path != self.volume_profile_state_path
                or not self.volume_profile_state_path.with_suffix(".npz").exists()
            ):
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
        candidate = self._load_volume_profile_state_candidates(bootstrap_df)
        if candidate is not None:
            _last_candle_ts, label, path, state = candidate
            self.volume_profile_state = state
            self.volume_profile_state_source_path = path
            print(f"[vp] loaded {label} state -> {path.with_suffix('.npz')}")
            self._sync_volume_profile_state_with_history(bootstrap_df)
            return

        print("[vp] bootstrap state from historical candles")
        self.volume_profile_state = bootstrap_state_from_history(
            bootstrap_df.loc[:, ["Opened", "High", "Low", "Volume"]],
            self.volume_profile_cfg,
        )
        self._save_runtime_volume_profile_state(log=True, context="runtime state")

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

        added = 0
        for row in catchup_df.itertuples(index=False):
            opened = pd.Timestamp(row.Opened)
            if opened <= self.opened_candles[-1]:
                continue
            self._append_new_candle(
                opened,
                (row.Open, row.High, row.Low, row.Close, row.Volume),
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
                build_latest_session_counter_feature_dict_fast(
                    latest_opened=opened_values[-1],
                    feature_cols=self.session_feature_columns,
                )
            )
        if self.realized_volatility_state is not None:
            values.update(self.latest_realized_volatility_values)

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

    def _sample_price_sim_scenario(self):
        if self.kelly_runtime.get("price_sim_model") != "parametric_market_mid_overround":
            return None

        intercept = float(self.kelly_runtime["mid_intercept_mean"])
        intercept_std = float(self.kelly_runtime["mid_intercept_std"])
        if intercept_std > 0.0:
            intercept += intercept_std * float(self.price_rng.standard_normal())
        intercept = float(
            np.clip(
                intercept,
                float(self.kelly_runtime["mid_intercept_min"]),
                float(self.kelly_runtime["mid_intercept_max"]),
            )
        )

        slope = float(self.kelly_runtime["mid_slope_mean"])
        slope_std = float(self.kelly_runtime["mid_slope_std"])
        if slope_std > 0.0:
            slope += slope_std * float(self.price_rng.standard_normal())
        slope = float(
            np.clip(
                slope,
                float(self.kelly_runtime["mid_slope_min"]),
                float(self.kelly_runtime["mid_slope_max"]),
            )
        )

        return {
            "mid_intercept": float(intercept),
            "mid_slope": float(slope),
        }

    def _simulate_execution_quotes(self, prob_up_raw):
        if self.kelly_runtime.get("price_sim_model") == "neutral_conservative_fixed":
            ask_price = float(self.kelly_runtime["ask_price"])
            return {
                "up_price": float(ask_price),
                "down_price": float(ask_price),
                "market_mid_up": 0.5,
                "price_policy": str(self.kelly_runtime.get("price_policy", "")),
            }

        if self.kelly_runtime.get("price_sim_model") == "parametric_market_mid_overround":
            tick_size = float(self.kelly_runtime["tick_size"])
            half_tick = float(self.kelly_runtime["half_tick"])
            scenario = self.price_sim_scenario or self._sample_price_sim_scenario()

            residual = float(self.kelly_runtime["mid_residual_std"]) * float(
                self.price_rng.standard_normal()
            )
            residual = float(
                np.clip(
                    residual,
                    -float(self.kelly_runtime["mid_residual_abs_q99"]),
                    float(self.kelly_runtime["mid_residual_abs_q99"]),
                )
            )

            ask_mid_up = (
                0.5
                + float(scenario["mid_intercept"])
                + float(scenario["mid_slope"]) * (float(prob_up_raw) - 0.5)
                + residual
            )

            overround_ticks = int(
                self.price_rng.choice(
                    np.asarray(
                        self.kelly_runtime["overround_ticks_values"],
                        dtype=np.int64,
                    ),
                    p=np.asarray(
                        self.kelly_runtime["overround_ticks_probs"],
                        dtype=np.float64,
                    ),
                )
            )

            mid_half_ticks_float = ask_mid_up / half_tick
            mid_half_ticks = int(round(mid_half_ticks_float))
            lower_mid_half_ticks = int(overround_ticks + 2)
            upper_mid_half_ticks = int(198 - overround_ticks)
            mid_half_ticks = int(
                np.clip(mid_half_ticks, lower_mid_half_ticks, upper_mid_half_ticks)
            )
            if (mid_half_ticks + overround_ticks) % 2 != 0:
                mid_half_ticks += 1 if mid_half_ticks_float >= mid_half_ticks else -1

            up_ticks = int((mid_half_ticks + overround_ticks) // 2)
            down_ticks = int((overround_ticks - mid_half_ticks + 200) // 2)
            up_price = float(tick_size * up_ticks)
            down_price = float(tick_size * down_ticks)

            return {
                "up_price": up_price,
                "down_price": down_price,
                "market_mid_up": float(ask_mid_up),
                "overround_ticks": int(overround_ticks),
                "mid_intercept": float(scenario["mid_intercept"]),
                "mid_slope": float(scenario["mid_slope"]),
                "mid_residual": float(residual),
            }

        sigma = float(self.kelly_runtime["sigma"])
        spread_half = float(self.kelly_runtime["spread_half"])
        base_price = float(self.kelly_runtime["base_price"])
        clip_lo = float(self.kelly_runtime["price_clip_lo"])
        clip_hi = float(self.kelly_runtime["price_clip_hi"])
        eps = float(self.price_rng.standard_normal())
        slip = abs(sigma * eps)
        price = float(np.clip(base_price + spread_half + slip, clip_lo, clip_hi))
        return {
            "up_price": float(price),
            "down_price": float(price),
            "eps": float(eps),
            "slip": float(slip),
        }

    def _recommend_kelly_bet(self, prob_up_raw):
        bankroll = float(self.live_bankroll_usdc)
        if bankroll <= 0.0:
            return {"reason": "bankroll_non_positive"}

        p = float(
            adjust_probability_for_kelly(
                float(prob_up_raw),
                prob_shrink=float(self.kelly_runtime["prob_shrink"]),
                min_clip=MIN_PROBA_CLIP,
            )
        )

        quotes = self._simulate_execution_quotes(prob_up_raw=prob_up_raw)
        fee_rate = float(self.kelly_runtime["fee_rate"])
        fee_exponent = float(self.kelly_runtime["fee_exponent"])
        candidates = []
        for side, price, p_side in (
            ("up", float(quotes["up_price"]), float(p)),
            ("down", float(quotes["down_price"]), float(1.0 - p)),
        ):
            eff_rate = fee_rate * float((price * (1.0 - price)) ** fee_exponent)
            if eff_rate >= 0.99:
                continue
            c_eff = price / (1.0 - eff_rate)
            edge = p_side - c_eff
            candidates.append(
                {
                    "side": side,
                    "entry_price": float(price),
                    "prob_win_adj": float(p_side),
                    "edge": float(edge),
                    "c_eff": float(c_eff),
                    "eff_rate": float(eff_rate),
                }
            )

        if not candidates:
            return {
                "reason": "eff_rate_too_high",
                "prob_win_raw": float(prob_up_raw),
                "prob_win_adj": p,
                "up_price": float(quotes["up_price"]),
                "down_price": float(quotes["down_price"]),
                "eps": float(quotes.get("eps", np.nan)),
                "slip": float(quotes.get("slip", np.nan)),
            }

        best = max(candidates, key=lambda item: float(item["edge"]))
        side = str(best["side"])
        selected_edge = float(best["edge"])
        p_side = float(best["prob_win_adj"])
        price = float(best["entry_price"])
        c_eff = float(best["c_eff"])
        eff_rate = float(best["eff_rate"])
        order_price_cap = float(self.kelly_runtime.get("order_price_cap", np.inf))
        order_min_size = float(self.kelly_runtime.get("order_min_size", 0.0))

        if selected_edge < float(self.kelly_runtime["min_edge"]):
            return {
                "reason": "edge_below_min",
                "side": side,
                "edge": float(selected_edge),
                "prob_win_raw": float(prob_up_raw),
                "prob_win_adj": float(p_side),
                "entry_price": float(price),
                "c_eff": float(c_eff),
                "eff_rate": float(eff_rate),
                "up_price": float(quotes["up_price"]),
                "down_price": float(quotes["down_price"]),
                "eps": float(quotes.get("eps", np.nan)),
                "slip": float(quotes.get("slip", np.nan)),
            }

        if price > order_price_cap:
            return {
                "reason": "price_above_cap",
                "side": side,
                "edge": float(selected_edge),
                "prob_win_raw": float(prob_up_raw),
                "prob_win_adj": float(p_side),
                "entry_price": float(price),
                "order_price_cap": float(order_price_cap),
                "c_eff": float(c_eff),
                "eff_rate": float(eff_rate),
                "up_price": float(quotes["up_price"]),
                "down_price": float(quotes["down_price"]),
                "eps": float(quotes.get("eps", np.nan)),
                "slip": float(quotes.get("slip", np.nan)),
            }

        f_star = (p_side - c_eff) / (1.0 - c_eff)
        f_star = max(float(f_star), 0.0)
        f = min(
            float(self.kelly_runtime["cap"]),
            float(self.kelly_runtime["fractional_kelly"]) * f_star,
        )
        if f <= 0.0:
            return {
                "reason": "fraction_non_positive",
                "side": side,
                "edge": float(selected_edge),
                "prob_win_raw": float(prob_up_raw),
                "prob_win_adj": float(p_side),
                "entry_price": float(price),
                "c_eff": float(c_eff),
                "eff_rate": float(eff_rate),
                "up_price": float(quotes["up_price"]),
                "down_price": float(quotes["down_price"]),
                "eps": float(quotes.get("eps", np.nan)),
                "slip": float(quotes.get("slip", np.nan)),
            }

        stake = bankroll * f
        if stake < float(self.kelly_runtime["min_stake_usdc"]):
            return {
                "reason": "stake_below_min",
                "side": side,
                "edge": float(selected_edge),
                "fraction": float(f),
                "bet_usdc": float(stake),
                "prob_win_raw": float(prob_up_raw),
                "prob_win_adj": float(p_side),
                "entry_price": float(price),
                "c_eff": float(c_eff),
                "eff_rate": float(eff_rate),
                "up_price": float(quotes["up_price"]),
                "down_price": float(quotes["down_price"]),
                "eps": float(quotes.get("eps", np.nan)),
                "slip": float(quotes.get("slip", np.nan)),
            }

        fee_raw = stake * eff_rate
        fee = round(fee_raw, int(self.kelly_runtime["fee_round_decimals"]))
        if fee < float(self.kelly_runtime["min_fee"]):
            fee = 0.0
        if fee >= stake:
            return {
                "reason": "fee_ge_stake",
                "side": side,
                "edge": float(selected_edge),
                "fraction": float(f),
                "bet_usdc": float(stake),
                "prob_win_raw": float(prob_up_raw),
                "prob_win_adj": float(p_side),
                "entry_price": float(price),
                "c_eff": float(c_eff),
                "eff_rate": float(eff_rate),
                "eps": eps,
                "slip": slip,
            }

        shares_net = (stake - fee) / price
        if shares_net < order_min_size:
            return {
                "reason": "shares_below_order_min",
                "side": side,
                "edge": float(selected_edge),
                "fraction": float(f),
                "bet_usdc": float(stake),
                "prob_win_raw": float(prob_up_raw),
                "prob_win_adj": float(p_side),
                "entry_price": float(price),
                "fee_usdc": float(fee),
                "fee_raw_usdc": float(fee_raw),
                "shares_net": float(shares_net),
                "order_min_size": float(order_min_size),
                "c_eff": float(c_eff),
                "eff_rate": float(eff_rate),
                "up_price": float(quotes["up_price"]),
                "down_price": float(quotes["down_price"]),
                "eps": float(quotes.get("eps", np.nan)),
                "slip": float(quotes.get("slip", np.nan)),
            }
        return {
            "reason": "ok",
            "side": side,
            "edge": float(selected_edge),
            "fraction": float(f),
            "bet_usdc": float(stake),
            "prob_win_raw": float(prob_up_raw),
            "prob_win_adj": float(p_side),
            "entry_price": float(price),
            "fee_usdc": float(fee),
            "fee_raw_usdc": float(fee_raw),
            "shares_net": float(shares_net),
            "c_eff": float(c_eff),
            "eff_rate": float(eff_rate),
            "up_price": float(quotes["up_price"]),
            "down_price": float(quotes["down_price"]),
            "eps": float(quotes.get("eps", np.nan)),
            "slip": float(quotes.get("slip", np.nan)),
        }

    def _upsert_closed_candle(self, kline):
        opened = pd.to_datetime(int(kline["t"]), unit="ms", utc=True)
        ohlcv = (
            float(kline["o"]),
            float(kline["h"]),
            float(kline["l"]),
            float(kline["c"]),
            float(kline["v"]),
        )

        if self.opened_candles:
            last_opened = self.opened_candles[-1]
            if opened < last_opened:
                return None
            if opened == last_opened:
                self.ohlcv_np[-1, :] = np.asarray(ohlcv, dtype=np.float64)
                self.candle_open_close[opened] = (float(ohlcv[0]), float(ohlcv[3]))
                return opened

        self._append_new_candle(opened, ohlcv)
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
            side = str(rec.get("kelly_side", "none"))
            actual_up = int(rec["actual_up"])
            if stake_usdc > 0.0 and side in {"up", "down"}:
                is_trade_win = int(
                    (side == "up" and actual_up == 1)
                    or (side == "down" and actual_up == 0)
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
        signal_up = int(proba_up >= self.prediction_threshold)
        kelly = self._recommend_kelly_bet(prob_up_raw=proba_up)
        bankroll_before_entry = float(self.live_bankroll_usdc)
        stake_usdc = float(kelly.get("bet_usdc", 0.0) or 0.0)
        if stake_usdc > 0.0:
            self.live_bankroll_usdc -= stake_usdc
        bankroll_after_entry = float(self.live_bankroll_usdc)

        minute_open = self.opened_candles[-1]
        minute_close = minute_open + pd.Timedelta(minutes=1)
        bucket_start = minute_open.floor(
            f"{self.target_bucket_minutes}min"
        ) + pd.Timedelta(minutes=self.target_bucket_minutes)
        bucket_end = bucket_start + pd.Timedelta(minutes=self.target_bucket_minutes - 1)

        self.records.append(
            {
                "prediction_time": pd.Timestamp.now(tz="UTC"),
                "bucket_start": bucket_start,
                "bucket_end": bucket_end,
                "proba_up": proba_up,
                "threshold": self.prediction_threshold,
                "signal_up": signal_up,
                "kelly_side": str(kelly.get("side", "none")),
                "kelly_fraction": float(kelly.get("fraction", 0.0)),
                "kelly_bet_usdc": float(kelly.get("bet_usdc", 0.0)),
                "kelly_edge": float(kelly.get("edge", np.nan)),
                "kelly_prob_win_adj": float(kelly.get("prob_win_adj", np.nan)),
                "kelly_prob_win_raw": float(kelly.get("prob_win_raw", np.nan)),
                "kelly_reason": str(kelly.get("reason", "")),
                "stake_usdc": float(stake_usdc),
                "entry_price": float(kelly.get("entry_price", np.nan)),
                "entry_fee_usdc": float(kelly.get("fee_usdc", 0.0)),
                "entry_fee_raw_usdc": float(kelly.get("fee_raw_usdc", 0.0)),
                "shares_net": float(kelly.get("shares_net", 0.0)),
                "kelly_c_eff": float(kelly.get("c_eff", np.nan)),
                "kelly_eff_rate": float(kelly.get("eff_rate", np.nan)),
                "price_eps": float(kelly.get("eps", np.nan)),
                "price_slip": float(kelly.get("slip", np.nan)),
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
            }
        )
        self.predicted_buckets.add(bucket_start)

        decision_local = minute_close.tz_convert(self.local_tz).isoformat()
        return {
            "decision_local": decision_local,
            "bucket_start": bucket_start,
            "bucket_end": bucket_end,
            "proba_up": proba_up,
            "signal_up": signal_up,
            "kelly_side": str(kelly.get("side", "none")),
            "kelly_fraction": float(kelly.get("fraction", 0.0)),
            "kelly_bet_usdc": float(kelly.get("bet_usdc", 0.0)),
            "stake_usdc": float(stake_usdc),
            "bankroll_before_entry": float(bankroll_before_entry),
            "bankroll_after_entry": float(bankroll_after_entry),
            "kelly_reason": str(kelly.get("reason", "")),
            "kelly_edge": float(kelly.get("edge", np.nan)),
        }

    def _save_records(self):
        if not self.records:
            return
        write_records_csv(
            self.records,
            self.predictions_path,
            export_columns=LIVE_PREDICTION_EXPORT_COLUMNS,
            is_resolved=lambda rec: rec.get("actual_up") is not None
            and rec.get("is_correct") is not None,
            is_traded=lambda rec: rec.get("actual_up") is not None
            and rec.get("trade_is_win") is not None,
        )

    def _stats(self):
        kelly_resolved = sum(
            1
            for rec in self.records
            if rec["actual_up"] is not None and rec["is_correct"] is not None
        )
        kelly_resolved_wins = sum(
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
        win_rate_kelly_resolved = (
            float(kelly_resolved_wins / kelly_resolved)
            if kelly_resolved
            else float("nan")
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
            "kelly_resolved": kelly_resolved,
            "kelly_resolved_wins": kelly_resolved_wins,
            "kelly_resolved_losses": kelly_resolved - kelly_resolved_wins,
            "win_rate_kelly_resolved": win_rate_kelly_resolved,
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
        print(
            f"[indicators] latest_nan_count={len(self.last_indicator_nan_cols)} "
            f"cols={cols}"
        )

    def _log(self, tag, pred=None):
        stats = self._stats()
        kelly_resolved_win_rate_txt = (
            "n/a"
            if not np.isfinite(stats["win_rate_kelly_resolved"])
            else f"{stats['win_rate_kelly_resolved'] * 100:.2f}%"
        )
        closed_trade_win_rate_txt = (
            "n/a"
            if not np.isfinite(stats["win_rate_closed_trade"])
            else f"{stats['win_rate_closed_trade'] * 100:.2f}%"
        )
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
            f"kelly_resolved={stats['kelly_resolved']}",
            f"kelly_resolved_wins={stats['kelly_resolved_wins']}",
            f"kelly_resolved_losses={stats['kelly_resolved_losses']}",
            f"win_rate_kelly_resolved={kelly_resolved_win_rate_txt}",
            f"closed_trades={stats['closed_trades']}",
            f"closed_trade_wins={stats['closed_trade_wins']}",
            f"closed_trade_losses={stats['closed_trade_losses']}",
            f"win_rate_closed_trade={closed_trade_win_rate_txt}",
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
                ]
            )
        print(" ".join(msg))
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
                closed_candle, live_minute_opened, _event_at = self._consume_ws_payload(
                    payload
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
            "Kelly sizing | "
            f"bankroll={self.live_bankroll_usdc:.2f} "
            f"fractional_kelly={self.kelly_runtime['fractional_kelly']:.6f} "
            f"cap={self.kelly_runtime['cap']:.6f} "
            f"min_edge={self.kelly_runtime['min_edge']:.6f} "
            f"prob_shrink={self.kelly_runtime['prob_shrink']:.6f} "
            f"min_stake_usdc={self.kelly_runtime['min_stake_usdc']:.2f}"
        )
        print(
            "Price/Fee model | "
            f"fee_rate={self.kelly_runtime['fee_rate']:.6f} "
            f"fee_exp={self.kelly_runtime['fee_exponent']:.3f}"
        )
        if self.kelly_runtime.get("price_sim_model") == "neutral_conservative_fixed":
            print(
                "Price sim | "
                f"model={self.kelly_runtime['price_sim_model']} "
                f"ask_price={self.kelly_runtime['ask_price']:.6f} "
                f"order_price_cap={self.kelly_runtime['order_price_cap']:.3f} "
                f"order_min_size={self.kelly_runtime['order_min_size']:.2f}"
            )
        elif self.kelly_runtime.get("price_sim_model") == "parametric_market_mid_overround":
            print(
                "Price sim | "
                f"model={self.kelly_runtime['price_sim_model']} "
                f"mid_slope_mean={self.kelly_runtime['mid_slope_mean']:.6f} "
                f"mid_slope_std={self.kelly_runtime['mid_slope_std']:.6f} "
                f"mid_residual_std={self.kelly_runtime['mid_residual_std']:.6f} "
                f"order_price_cap={self.kelly_runtime['order_price_cap']:.3f} "
                f"order_min_size={self.kelly_runtime['order_min_size']:.2f}"
            )
        else:
            print(
                "Price sim | "
                f"model={self.kelly_runtime['price_sim_model']} "
                f"base_price={self.kelly_runtime['base_price']:.6f} "
                f"sigma={self.kelly_runtime['sigma']:.6f} "
                f"spread_half={self.kelly_runtime['spread_half']:.6f} "
                f"clip=[{self.kelly_runtime['price_clip_lo']:.3f},{self.kelly_runtime['price_clip_hi']:.3f}]"
            )
        print(f"Kelly config: {KELLY_CONFIG_PATH}")
        print(
            "Runtime hashes | "
            f"model={self.model_hash} "
            f"kelly={self.kelly_config_hash} "
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
            if self.volume_profile_state_source_path is not None:
                print(
                    "VP source state: "
                    f"{self.volume_profile_state_source_path.with_suffix('.npz')}"
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
