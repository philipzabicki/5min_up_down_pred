import json
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import lightgbm as lgb
from websocket import WebSocketApp
from create_modeling_dataset import parse_fit_results
from features.candle_features import (
    RAW_OHLCV_COLS,
    STREAK_FEATURE_PREFIX,
    SUPPORTED_CANDLE_FEATURE_COLS,
    build_latest_candle_derived_feature_dict_fast,
    build_latest_candle_streak_feature_dict_fast,
    build_latest_candle_pattern_feature_dict,
    resolve_candle_derived_feature_cols,
    resolve_candle_pattern_feature_cols,
    resolve_streak_interval_to_rule,
)

from features.ADX import get_adx_values
from features.BollingerBands import get_bollinger_bands_values
from features.ChaikinOsc import get_chaikin_oscillator_values
from features.KeltnerChannel import get_keltner_channel_values
from features.MACD import get_macd_values
from features.session_open_features import (
    SUPPORTED_SESSION_COUNTER_COLS,
    build_latest_session_counter_feature_dict_fast,
)
from features.StochOsc import get_stochastic_oscillator_values
from features.volume_profile_fixed_range import (
    FEATURE_VERSION as VP_FEATURE_VERSION,
    STATE_DIR as VP_STATE_DIR,
    bootstrap_state_from_history,
    extract_features_from_state,
    is_volume_profile_feature,
    load_state as load_volume_profile_state,
    normalize_config as normalize_volume_profile_config,
    save_state as save_volume_profile_state,
    state_matches_config as volume_profile_state_matches_config,
    update_state_with_candle as update_volume_profile_state_with_candle,
)
from modeling_dataset_utils import (
    MODELING_DATASET_CONFIG_FILE,
    load_modeling_dataset_settings,
    split_feature_subset,
)


SYMBOL = "BTCUSDT"
INTERVAL = "1m"
REST_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
WS_URL = f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@kline_{INTERVAL}"
MODEL_META_PATH = Path("data\models\lgbm_meta_20260318_170305.json")
KELLY_CONFIG_PATH = Path("configs/kelly_config.json")
INDICATOR_STABILITY_SUMMARY_PATH = Path("data/analysis/indicator_stability_summary.json")
RUN_STARTED_AT_UTC = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
PREDICTIONS_OUTPUT_PATH = Path(
    f"data/live_predictions_{SYMBOL}_{INTERVAL}_{RUN_STARTED_AT_UTC}.csv"
)
VOLUME_PROFILE_STATE_PATH = VP_STATE_DIR / f"{SYMBOL}_{INTERVAL}_{VP_FEATURE_VERSION}"

DEFAULT_BOOTSTRAP_CANDLES = 20_000
MAX_WS_RECONNECT_DELAY_SEC = 15
WS_PING_INTERVAL_SEC = 20
WS_PING_TIMEOUT_SEC = 10

LIVE_INITIAL_BANKROLL_USDC = 1000.0
MIN_PROBA_CLIP = 1e-6

OHLCV_COLS = list(RAW_OHLCV_COLS)
PREDICTIONS_EXPORT_COLUMNS = (
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
    "pnl_usdc",
    "win_rate_resolved",
    "win_rate_traded",
)
BASE_FEATURE_COLS = (
    set(OHLCV_COLS)
    | set(SUPPORTED_CANDLE_FEATURE_COLS)
    | set(SUPPORTED_SESSION_COUNTER_COLS)
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
    __slots__ = ("feature_col", "builder", "params", "required_candles")

    def __init__(self, feature_col, builder, params, required_candles):
        self.feature_col = feature_col
        self.builder = builder
        self.params = params
        self.required_candles = required_candles

MODELING_DATASET_SETTINGS = load_modeling_dataset_settings()
FIT_RESULTS_DIR = MODELING_DATASET_SETTINGS["fit_results_dir"]


def interval_to_timedelta(interval):
    if interval.endswith("m"):
        return pd.Timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return pd.Timedelta(hours=int(interval[:-1]))
    if interval.endswith("d"):
        return pd.Timedelta(days=int(interval[:-1]))
    raise ValueError(f"Unsupported interval: {interval}")


INTERVAL_DELTA = interval_to_timedelta(INTERVAL)


def parse_target_bucket_minutes(target_col):
    match = re.search(r"target_(\d+)m", target_col)
    return int(match.group(1)) if match else 5


def load_model_and_meta(meta_path):
    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    candidate_paths = []

    artifacts = meta.get("artifacts")
    if isinstance(artifacts, dict):
        final_model_path = artifacts.get("final_model_path")
        if isinstance(final_model_path, str) and final_model_path.strip():
            candidate_paths.append(Path(final_model_path))

    candidate_paths.extend(
        [
            meta_path.with_name(meta_path.name.replace("_meta_", "_")).with_suffix(".txt"),
            meta_path.with_name(meta_path.stem.replace("_meta", "")).with_suffix(".txt"),
            meta_path.with_name(meta_path.name.replace("_meta.json", ".txt")),
        ]
    )

    model_path = next((path for path in candidate_paths if path.exists()), None)
    if model_path is None:
        searched = ", ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(
            f"Model file not found for metadata {meta_path}. Candidates: {searched}"
        )

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


def load_required_stable_window(summary_path):
    if not summary_path.exists():
        raise FileNotFoundError(
            "Indicator stability summary is required for live runtime: "
            f"{summary_path}"
        )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    required_window = int(payload.get("global_required_stable_window", 0) or 0)
    unstable_count = int(payload.get("unstable_feature_count", 0) or 0)
    if required_window <= 0:
        raise ValueError(
            "Indicator stability summary missing valid global_required_stable_window: "
            f"{summary_path}"
        )
    if unstable_count != 0:
        raise ValueError(
            "Indicator stability summary reports unstable features; "
            "live runtime should not proceed."
        )
    return required_window


def load_kelly_runtime_config(config_path):
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
        "sigma": _read_float(payload, "sigma"),
        "spread_half": _read_float(payload, "spread_half"),
        "fee_rate": _read_float(fee_model, "feeRate"),
        "fee_exponent": _read_float(fee_model, "exponent"),
        "fee_round_decimals": int(fee_model.get("fee_round_decimals", 4)),
        "min_fee": _read_float(fee_model, "min_fee"),
        "base_price": _read_float(price_sim, "base_price"),
        "price_clip_lo": _read_float(price_sim, "price_clip_lo"),
        "price_clip_hi": _read_float(price_sim, "price_clip_hi"),
        "seed": int(cv_meta.get("seed", 37)),
    }
    if cfg["price_clip_lo"] >= cfg["price_clip_hi"]:
        raise ValueError("Kelly config invalid: price_clip_lo must be < price_clip_hi")
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

        specs.append(
            IndicatorSpec(
                feature_col=col,
                builder=builder,
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


def fetch_historical_ohlcv(session, candles):
    all_rows = []
    end_time_ms = None

    while len(all_rows) < candles:
        batch_size = min(1000, candles - len(all_rows))
        params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": batch_size}
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        response = session.get(REST_KLINES_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not data:
            break

        all_rows = data + all_rows
        end_time_ms = int(data[0][0]) - 1
        if len(data) < batch_size:
            break
        time.sleep(0.05)

    if not all_rows:
        raise RuntimeError("REST bootstrap returned no candles.")

    now_ms = int(time.time() * 1000)
    rows = []
    for row in all_rows:
        if int(row[6]) >= now_ms:
            continue
        rows.append(
            {
                "Opened": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                "Open": float(row[1]),
                "High": float(row[2]),
                "Low": float(row[3]),
                "Close": float(row[4]),
                "Volume": float(row[5]),
            }
        )

    if not rows:
        raise RuntimeError("No closed candles found during REST bootstrap.")

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["Opened"])
        .sort_values("Opened")
        .reset_index(drop=True)
    )


def fetch_closed_ohlcv_range(session, start_opened, end_opened=None, limit=1000):
    start_ts = pd.Timestamp(start_opened)
    end_ts = pd.Timestamp(end_opened) if end_opened is not None else None
    if end_ts is not None and end_ts < start_ts:
        return pd.DataFrame(columns=["Opened", *OHLCV_COLS])

    all_rows = []
    next_start = start_ts
    interval_ms = int(INTERVAL_DELTA.total_seconds() * 1000)

    while True:
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "limit": int(limit),
            "startTime": int(next_start.value // 1_000_000),
        }
        if end_ts is not None:
            params["endTime"] = int(end_ts.value // 1_000_000) + interval_ms - 1

        response = session.get(REST_KLINES_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not data:
            break

        all_rows.extend(data)
        if len(data) < limit:
            break

        last_opened = pd.to_datetime(int(data[-1][0]), unit="ms", utc=True)
        next_start = last_opened + INTERVAL_DELTA
        if end_ts is not None and next_start > end_ts:
            break
        time.sleep(0.05)

    if not all_rows:
        return pd.DataFrame(columns=["Opened", *OHLCV_COLS])

    now_ms = int(time.time() * 1000)
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
                "Volume": float(row[5]),
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


class LivePredictor:
    def __init__(self):
        self.model, meta = load_model_and_meta(MODEL_META_PATH)
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

        self.feature_medians = {
            str(k): float(v) for k, v in meta.get("train_feature_medians", {}).items()
        }
        self.prediction_threshold = 0.5
        self.target_col = str(meta.get("target_col", "target_5m_candle_up"))
        self.target_bucket_minutes = parse_target_bucket_minutes(self.target_col)

        self.kelly_runtime = load_kelly_runtime_config(KELLY_CONFIG_PATH)
        self.live_bankroll_usdc = LIVE_INITIAL_BANKROLL_USDC
        self.price_rng = np.random.default_rng(int(self.kelly_runtime["seed"]))

        feature_parts = split_feature_subset(self.feature_columns)
        if feature_parts["streak_intervals"]:
            self.streak_interval_to_rule = resolve_streak_interval_to_rule(
                feature_parts["streak_intervals"]
            )
        else:
            self.streak_interval_to_rule = {}
        self.session_feature_columns = tuple(feature_parts["session_feature_cols"])
        self.volume_profile_feature_columns = tuple(
            feature_parts["volume_profile_feature_cols"]
        )
        self.volume_profile_cfg = normalize_volume_profile_config(
            MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")
        )
        if self.volume_profile_feature_columns and not self.volume_profile_cfg["enabled"]:
            raise ValueError(
                "Model requires volume profile features but volume_profile_fixed_range.enabled is false."
            )
        self.volume_profile_enabled = bool(
            self.volume_profile_feature_columns and self.volume_profile_cfg["enabled"]
        )
        self.volume_profile_state_path = VOLUME_PROFILE_STATE_PATH

        self.indicator_specs = load_indicator_specs(self.feature_columns)
        self.required_stable_window = load_required_stable_window(
            INDICATOR_STABILITY_SUMMARY_PATH
        )
        max_needed = max((s.required_candles for s in self.indicator_specs), default=0)
        self.bootstrap_candles = max(
            DEFAULT_BOOTSTRAP_CANDLES,
            self.required_stable_window,
            max_needed * 3,
        )
        self.max_keep = max(self.bootstrap_candles + 2000, 25000)

        self.session = requests.Session()

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
        self.last_processed_closed_opened = self.opened_candles[-1] if self.opened_candles else None

        self.predictions_path = PREDICTIONS_OUTPUT_PATH
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        self.volume_profile_state = None
        if self.volume_profile_enabled:
            self._initialize_volume_profile_state(bootstrap_df)

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
        self.candle_open_close[opened] = (float(ohlcv_row[0, 0]), float(ohlcv_row[0, 3]))

        if len(self.opened_candles) > self.max_keep:
            dropped_opened = self.opened_candles.popleft()
            self.candle_open_close.pop(dropped_opened, None)
            self.ohlcv_np = self.ohlcv_np[-self.max_keep :, :]
            self.opened_ns_np = self.opened_ns_np[-self.max_keep :]

    def _sync_volume_profile_state_with_history(self, history_df):
        if not self.volume_profile_enabled or history_df.empty:
            return

        sync_df = history_df.loc[:, ["Opened", "High", "Low", "Volume"]]
        last_candle_time = self.volume_profile_state.get("last_candle_time")
        if last_candle_time:
            last_candle_ts = pd.Timestamp(last_candle_time)
            sync_df = sync_df.loc[sync_df["Opened"] > last_candle_ts]

        if sync_df.empty:
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
        paths = save_volume_profile_state(
            self.volume_profile_state,
            self.volume_profile_state_path,
        )
        print(f"[vp] saved state -> {paths['npz']}")

    def _initialize_volume_profile_state(self, bootstrap_df):
        try:
            state = load_volume_profile_state(self.volume_profile_state_path)
            if not volume_profile_state_matches_config(state, self.volume_profile_cfg):
                raise ValueError("config mismatch")
            self.volume_profile_state = state
            print(f"[vp] loaded state -> {self.volume_profile_state_path.with_suffix('.npz')}")
            self._sync_volume_profile_state_with_history(bootstrap_df)
            return
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[vp] state reload fallback: {exc}")

        print("[vp] bootstrap state from historical candles")
        self.volume_profile_state = bootstrap_state_from_history(
            bootstrap_df.loc[:, ["Opened", "High", "Low", "Volume"]],
            self.volume_profile_cfg,
        )
        paths = save_volume_profile_state(
            self.volume_profile_state,
            self.volume_profile_state_path,
        )
        print(f"[vp] saved state -> {paths['npz']}")

    def _extract_volume_profile_features_for_latest_candle(self):
        if not self.volume_profile_enabled:
            return {}
        latest_ohlcv = self.ohlcv_np[-1, :]
        return extract_features_from_state(
            self.volume_profile_state,
            high=float(latest_ohlcv[1]),
            low=float(latest_ohlcv[2]),
        )

    def _update_volume_profile_state_for_latest_candle(self, opened):
        if not self.volume_profile_enabled:
            return
        latest_ohlcv = self.ohlcv_np[-1, :]
        update_volume_profile_state_with_candle(
            self.volume_profile_state,
            high=float(latest_ohlcv[1]),
            low=float(latest_ohlcv[2]),
            volume=float(latest_ohlcv[4]),
        )
        self.volume_profile_state["last_candle_time"] = str(pd.Timestamp(opened).isoformat())
        save_volume_profile_state(self.volume_profile_state, self.volume_profile_state_path)

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
            self._update_volume_profile_state_for_latest_candle(opened)
            added += 1

        if added > 0:
            print(
                "[sync] caught_up_closed_candles="
                f"{added} first={catchup_df['Opened'].iloc[0].isoformat()} "
                f"last={catchup_df['Opened'].iloc[-1].isoformat()}"
            )
        return added

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
                build_latest_candle_pattern_feature_dict(
                    opened_values=opened_values,
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

        if volume_profile_values:
            values.update(volume_profile_values)

        ohlcv_np = self.ohlcv_np
        indicator_nan_cols = []
        for spec in self.indicator_specs:
            series = np.asarray(
                spec.builder(spec.params, ohlcv_np), dtype=np.float64
            ).reshape(-1)
            if series.shape[0] != ohlcv_np.shape[0]:
                raise ValueError(
                    f"Length mismatch for {spec.feature_col}: {series.shape[0]} != {ohlcv_np.shape[0]}"
                )
            raw_value = float(series[-1])
            values[spec.feature_col] = raw_value
            if not np.isfinite(raw_value):
                indicator_nan_cols.append(spec.feature_col)

        self.last_indicator_nan_cols = indicator_nan_cols

        vector = np.empty((1, len(self.feature_columns)), dtype=np.float64)
        for i, col in enumerate(self.feature_columns):
            value = values.get(col, np.nan)
            if not np.isfinite(value):
                value = self.feature_medians.get(col, 0.0)
            vector[0, i] = float(value)
        return vector

    def _simulate_execution_price(self):
        sigma = float(self.kelly_runtime["sigma"])
        spread_half = float(self.kelly_runtime["spread_half"])
        base_price = float(self.kelly_runtime["base_price"])
        clip_lo = float(self.kelly_runtime["price_clip_lo"])
        clip_hi = float(self.kelly_runtime["price_clip_hi"])
        eps = float(self.price_rng.standard_normal())
        slip = abs(sigma * eps)
        price = float(np.clip(base_price + spread_half + slip, clip_lo, clip_hi))
        return price, eps, slip

    def _recommend_kelly_bet(self, prob_up_raw):
        bankroll = float(self.live_bankroll_usdc)
        if bankroll <= 0.0:
            return {"reason": "bankroll_non_positive"}

        p = 0.5 + float(self.kelly_runtime["prob_shrink"]) * (
            float(prob_up_raw) - 0.5
        )
        p = float(np.clip(p, MIN_PROBA_CLIP, 1.0 - MIN_PROBA_CLIP))

        price, eps, slip = self._simulate_execution_price()
        fee_rate = float(self.kelly_runtime["fee_rate"])
        fee_exponent = float(self.kelly_runtime["fee_exponent"])
        eff_rate = fee_rate * float((price * (1.0 - price)) ** fee_exponent)
        if eff_rate >= 0.99:
            return {
                "reason": "eff_rate_too_high",
                "prob_win_raw": float(prob_up_raw),
                "prob_win_adj": p,
                "entry_price": price,
                "eps": eps,
                "slip": slip,
            }

        c_eff = price / (1.0 - eff_rate)
        edge_up = p - c_eff
        edge_down = (1.0 - p) - c_eff
        if edge_up >= edge_down:
            side = "up"
            selected_edge = edge_up
            p_side = p
        else:
            side = "down"
            selected_edge = edge_down
            p_side = 1.0 - p

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
                "eps": eps,
                "slip": slip,
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
                "eps": eps,
                "slip": slip,
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
                "eps": eps,
                "slip": slip,
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
            "eps": eps,
            "slip": slip,
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

    def _resolve_pending(self):
        if not self.records:
            return 0

        resolved_now = 0

        for rec in self.records:
            if rec["actual_up"] is not None:
                continue

            start_candle = self.candle_open_close.get(rec["bucket_start"])
            end_candle = self.candle_open_close.get(rec["bucket_end"])
            if start_candle is None or end_candle is None:
                continue

            bucket_open = float(start_candle[0])
            bucket_close = float(end_candle[1])
            actual_up = int(bucket_close > bucket_open)

            rec["bucket_open_price"] = bucket_open
            rec["bucket_close_price"] = bucket_close
            rec["actual_up"] = actual_up
            rec["is_correct"] = int(actual_up == rec["signal_up"])
            rec["resolved_at"] = pd.Timestamp.now(tz="UTC")

            stake_usdc = float(rec.get("stake_usdc", 0.0) or 0.0)
            side = str(rec.get("kelly_side", "none"))
            if stake_usdc > 0.0 and side in {"up", "down"}:
                is_trade_win = int(
                    (side == "up" and actual_up == 1) or (side == "down" and actual_up == 0)
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

        out = pd.DataFrame(self.records)
        resolved_rates = []
        traded_rates = []
        resolved_count = 0
        resolved_wins = 0
        traded_count = 0
        traded_wins = 0

        for rec in self.records:
            row_resolved = rec["actual_up"] is not None and rec["is_correct"] is not None
            row_traded = (
                rec["actual_up"] is not None and rec["trade_is_win"] is not None
            )
            if not row_resolved:
                resolved_rates.append(np.nan)
                traded_rates.append(np.nan)
                continue

            resolved_count += 1
            resolved_wins += int(rec["is_correct"])
            if row_traded:
                traded_count += 1
                traded_wins += int(rec["trade_is_win"])

            resolved_rates.append(float(resolved_wins / resolved_count))
            traded_rates.append(
                float(traded_wins / traded_count) if traded_count else np.nan
            )

        out["win_rate_resolved"] = resolved_rates
        out["win_rate_traded"] = traded_rates
        out = out.loc[:, list(PREDICTIONS_EXPORT_COLUMNS)]
        for col in ["prediction_time", "bucket_start", "bucket_end", "resolved_at"]:
            out[col] = out[col].map(
                lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else ""
            )
        out.to_csv(self.predictions_path, index=False)

    def _stats(self):
        resolved = sum(1 for rec in self.records if rec["actual_up"] is not None)
        resolved_wins = sum(
            int(rec["is_correct"])
            for rec in self.records
            if rec["actual_up"] is not None and rec["is_correct"] is not None
        )
        traded = sum(
            1
            for rec in self.records
            if rec["actual_up"] is not None and rec["trade_is_win"] is not None
        )
        traded_wins = sum(
            int(rec["trade_is_win"])
            for rec in self.records
            if rec["actual_up"] is not None and rec["trade_is_win"] is not None
        )
        resolved_win_rate = float(resolved_wins / resolved) if resolved else float("nan")
        traded_win_rate = float(traded_wins / traded) if traded else float("nan")
        total_pnl = float(
            sum(
                float(rec.get("pnl_usdc", 0.0) or 0.0)
                for rec in self.records
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
        resolved_win_rate_txt = (
            "n/a"
            if not np.isfinite(stats["resolved_win_rate"])
            else f"{stats['resolved_win_rate'] * 100:.2f}%"
        )
        traded_win_rate_txt = (
            "n/a"
            if not np.isfinite(stats["traded_win_rate"])
            else f"{stats['traded_win_rate'] * 100:.2f}%"
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
                ]
            )
        print(" ".join(msg))
        if pred is not None:
            self._print_indicator_nan_status()

    def _on_open(self, ws):
        print(f"[ws] connected: {WS_URL}")

    def _on_error(self, ws, error):
        print(f"[ws] error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"[ws] closed: code={close_status_code}, msg={close_msg}")

    def _on_message(self, ws, message):
        try:
            payload = json.loads(message)
            kline = payload.get("k", {})
            if not kline or not bool(kline.get("x", False)):
                return

            opened_from_ws = pd.to_datetime(int(kline["t"]), unit="ms", utc=True)
            if self.opened_candles:
                expected_next = self.opened_candles[-1] + INTERVAL_DELTA
                if opened_from_ws > expected_next:
                    self._sync_closed_candles_from_rest(stop_before_opened=opened_from_ws)

            opened = self._upsert_closed_candle(kline)
            if opened is None:
                return
            if (
                self.last_processed_closed_opened is not None
                and opened <= self.last_processed_closed_opened
            ):
                return

            volume_profile_values = self._extract_volume_profile_features_for_latest_candle()

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

            self._update_volume_profile_state_for_latest_candle(opened)
            self.last_processed_closed_opened = opened

            if resolved_now > 0 or pred is not None:
                self._save_records()
                self._log("resolve+pred" if pred else "resolve", pred=pred)
        except Exception as exc:
            print(f"[pred] message handling failed: {exc}")

    def run_forever(self):
        print(
            "Starting live predictor | "
            f"symbol={SYMBOL} interval={INTERVAL} "
            f"bootstrap_candles={len(self.opened_candles)} "
            f"target={self.target_col} "
            f"bucket_minutes={self.target_bucket_minutes} "
            f"features={len(self.feature_columns)} "
            f"streak_features={len(self.streak_interval_to_rule)} "
            f"session_features={len(self.session_feature_columns)} "
            f"vp_features={len(self.volume_profile_feature_columns)}"
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
            f"base_price={self.kelly_runtime['base_price']:.6f} "
            f"sigma={self.kelly_runtime['sigma']:.6f} "
            f"spread_half={self.kelly_runtime['spread_half']:.6f} "
            f"clip=[{self.kelly_runtime['price_clip_lo']:.3f},{self.kelly_runtime['price_clip_hi']:.3f}] "
            f"fee_rate={self.kelly_runtime['fee_rate']:.6f} "
            f"fee_exp={self.kelly_runtime['fee_exponent']:.3f}"
        )
        print(f"Kelly config: {KELLY_CONFIG_PATH}")
        print(f"Predictions file: {self.predictions_path}")
        if self.volume_profile_enabled:
            print(f"VP state path: {self.volume_profile_state_path.with_suffix('.npz')}")

        delay = 1
        while True:
            try:
                ws = WebSocketApp(
                    WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                ws.run_forever(
                    ping_interval=WS_PING_INTERVAL_SEC, ping_timeout=WS_PING_TIMEOUT_SEC
                )
            except Exception as exc:
                print(f"[ws] run failed: {exc}")

            print(f"[ws] reconnect in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, MAX_WS_RECONNECT_DELAY_SEC)

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
