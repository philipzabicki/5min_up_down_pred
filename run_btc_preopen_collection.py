"""Observe one-minute pre-open BTC markets with the pre-open baseline; no orders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit

import run_btc_shadow as shadow
from features.btc_preopen_baseline import FEATURE_COLUMNS, build_causal_preopen_features


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/btc_preopen_collection_protocol_20261004_v3.json"
CONFIG = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
RAW_CANDLES_PATH = ROOT / "data/datasets/raw/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m.csv"
HISTORY_ROWS = 241


class RawPreopenPredictor:
    """Keep only the rolling candle history needed by the frozen raw-feature model."""

    def __init__(self, candles: pd.DataFrame, max_keep: int = HISTORY_ROWS):
        self.max_keep = int(max_keep)
        self.candles = candles.sort_values("Opened").tail(self.max_keep).reset_index(drop=True)
        if len(self.candles) != self.max_keep:
            raise RuntimeError(
                f"Pre-open runtime needs {self.max_keep} contiguous candles; got {len(self.candles)}"
            )
        if not self.candles["Opened"].diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
            raise RuntimeError("Pre-open runtime bootstrap contains a candle gap")
        self.opened_candles = list(pd.to_datetime(self.candles["Opened"], utc=True))

    def _append_new_candle(self, opened, ohlcv, basis_futures_close=None):
        opened = pd.Timestamp(opened)
        opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC")
        expected = self.opened_candles[-1] + pd.Timedelta(minutes=1)
        if opened != expected:
            raise RuntimeError(
                f"Pre-open runtime expected candle {expected.isoformat()}, got {opened.isoformat()}"
            )
        record = {"Opened": opened}
        record.update(dict(zip(shadow.live_runtime.OHLCV_COLS, ohlcv)))
        record["UM_BTCUSDT_Close"] = (
            np.nan if basis_futures_close is None else float(basis_futures_close)
        )
        self.candles = pd.concat(
            [self.candles, pd.DataFrame([record])], ignore_index=True
        ).tail(self.max_keep)
        self.opened_candles.append(opened)
        self.opened_candles = self.opened_candles[-self.max_keep :]

    @staticmethod
    def _prepare_volume_profile_features_for_latest_candle(_opened):
        return None

    @staticmethod
    def _prepare_reaction_profile_features_for_latest_candle(_opened):
        return None

    def build_feature_snapshot(self, volume_profile_values=None, reaction_profile_values=None):
        features = build_causal_preopen_features(self.candles)
        vector = features.loc[:, FEATURE_COLUMNS].iloc[-1].to_numpy(dtype=np.float64)
        invalid = np.flatnonzero(~np.isfinite(vector)).tolist()
        return {"vector": vector.reshape(1, -1), "nonfinite_feature_indices": invalid}


def _load_anchor_candles(anchor_opened: pd.Timestamp) -> pd.DataFrame:
    columns = ["Opened", "Open", "High", "Low", "Close", "Volume", "UM_BTCUSDT_Close"]
    frame = pd.read_csv(RAW_CANDLES_PATH, usecols=columns)
    frame["Opened"] = pd.to_datetime(frame["Opened"], utc=True, errors="raise")
    frame = (
        frame.loc[frame["Opened"].le(anchor_opened)]
        .sort_values("Opened")
        .drop_duplicates("Opened", keep="last")
        .tail(HISTORY_ROWS)
        .reset_index(drop=True)
    )
    if len(frame) != HISTORY_ROWS or frame["Opened"].iloc[-1] != anchor_opened:
        raise RuntimeError(
            "The frozen pre-open anchor lacks the required contiguous raw candle history"
        )
    return frame


def _restore_preopen_predictor(connection, _frozen_hash, required_window):
    if int(required_window) != HISTORY_ROWS:
        raise RuntimeError(f"Pre-open model expects {HISTORY_ROWS} candles, got {required_window}")
    predictor = RawPreopenPredictor(_load_anchor_candles(shadow.ANCHOR_OPENED))
    saved = shadow._read_saved_candles(
        connection, shadow.ANCHOR_OPENED + shadow.live_runtime.INTERVAL_DELTA
    )
    if not saved.empty and not saved["Opened"].diff().iloc[1:].eq(
        shadow.live_runtime.INTERVAL_DELTA
    ).all():
        raise RuntimeError("Saved pre-open collection candles contain a gap")
    for row in saved.itertuples(index=False):
        predictor._append_new_candle(
            pd.Timestamp(row.Opened),
            tuple(float(getattr(row, column)) for column in shadow.live_runtime.OHLCV_COLS),
            getattr(row, "UM_BTCUSDT_Close", None),
        )
    return predictor


def _load_preopen_model():
    meta = json.loads(shadow.MODEL_META_PATH.read_text(encoding="utf-8"))
    model_path = shadow._source_model_path(shadow.MODEL_META_PATH)
    model = lgb.Booster(model_file=str(model_path))
    if list(meta.get("feature_columns", [])) != list(FEATURE_COLUMNS):
        raise ValueError("Pre-open model feature order differs from the live feature builder")
    if list(model.feature_name()) != list(FEATURE_COLUMNS):
        raise ValueError("Pre-open LightGBM feature order differs from the live feature builder")
    return model, meta, model_path


def _load_preopen_calibrator():
    payload = json.loads(shadow.MODEL_WEIGHTS_PATH.read_text(encoding="utf-8"))
    return {
        "btc_platt": {
            "intercept": float(payload["intercept"]),
            "coef": float(payload["coefficient"]),
        }
    }


def _predict_preopen_variants(weights, raw_btc, _quote_fields):
    calibrator = weights["btc_platt"]
    calibrated = float(
        expit(
            calibrator["intercept"]
            + calibrator["coef"] * logit(np.clip(raw_btc, 1e-6, 1.0 - 1e-6))
        )
    )
    return float(raw_btc), calibrated, None, None


def _record_collection_decision(
    connection,
    accounts,
    market_data,
    market_id,
    market_start,
    market_end,
    available_at,
    inference_started,
    inference_finished,
    prediction_values,
    raw_btc,
    quote_observation,
    quote_fields,
    weights,
):
    values = {
        **prediction_values,
        "collection_only": prediction_values["btc_platt"],
        "market_only": None,
        "market_plus_btc": None,
    }
    return _BASE_RECORD_MARKET_DECISION(
        connection,
        accounts,
        market_data,
        market_id,
        market_start,
        market_end,
        available_at,
        inference_started,
        inference_finished,
        values,
        raw_btc,
        quote_observation,
        quote_fields,
        weights,
    )


def _preopen_artifact_hashes():
    paths = (
        PROTOCOL_PATH,
        shadow.SOURCE_MODEL_MANIFEST_PATH,
        shadow.FEATURE_SOURCE_MANIFEST_PATH,
        shadow.MODEL_WEIGHTS_PATH,
        shadow.MODEL_META_PATH,
        shadow._source_model_path(shadow.MODEL_META_PATH),
        ROOT / "features/btc_preopen_baseline.py",
        ROOT / "features/btc_preopen_contract.py",
        ROOT / "run_btc_preopen_collection.py",
        ROOT / "run_btc_shadow.py",
        ROOT / "run.py",
        ROOT / "utils/polymarket.py",
        ROOT / "utils/polymarket_market_value.py",
    )
    return {
        str(path.resolve().relative_to(ROOT)).replace("\\", "/"): shadow._sha256(path)
        for path in paths
    }


def _install_runtime_adapter():
    shadow.PROTOCOL_PATH = PROTOCOL_PATH
    shadow.CONFIG = CONFIG
    shadow.SOURCE_MODEL_MANIFEST_PATH = ROOT / CONFIG["source_model_manifest_path"]
    shadow.FEATURE_SOURCE_MANIFEST_PATH = ROOT / CONFIG["feature_source_manifest_path"]
    shadow.MODEL_WEIGHTS_PATH = ROOT / CONFIG["model_weights_path"]
    shadow.MODEL_META_PATH = ROOT / CONFIG["model_meta_path"]
    shadow.OUTPUT_DIR = ROOT / CONFIG["output_directory"]
    shadow.DATABASE_PATH = shadow.OUTPUT_DIR / "shadow.sqlite3"
    shadow.MANIFEST_PATH = shadow.OUTPUT_DIR / "session_manifest.json"
    shadow.ANCHOR_OPENED = pd.Timestamp(CONFIG["profile_state_anchor_opened_utc"])
    shadow.POLL_SECONDS = int(CONFIG["poll_seconds"])
    shadow.CHECKPOINT_INTERVAL = pd.Timedelta(
        minutes=int(CONFIG["checkpoint_interval_minutes"])
    )
    shadow.MAX_BOOK_AGE_SECONDS = float(
        CONFIG["execution_assumption"]["max_book_age_seconds"]
    )
    shadow.MAX_BOOK_FUTURE_CLOCK_SKEW_SECONDS = float(
        CONFIG["execution_assumption"]["future_source_clock_skew_tolerance_seconds"]
    )
    shadow.VARIANTS = ("collection_only",)

    shadow._restore_predictor = _restore_preopen_predictor
    shadow._load_btc_model = _load_preopen_model
    shadow._load_market_weights = _load_preopen_calibrator
    shadow._predict_variants = _predict_preopen_variants
    shadow._record_market_decision = _record_collection_decision
    shadow._artifact_hashes = _preopen_artifact_hashes
    shadow._save_checkpoint = lambda connection, _predictor, _frozen_hash: connection.commit()
    shadow.audit.load_indicator_specs = lambda _feature_columns: []
    shadow.audit.load_indicator_history_requirements = lambda *args, **kwargs: {
        "global_required_runtime_window": HISTORY_ROWS
    }


_BASE_RECORD_MARKET_DECISION = shadow._record_market_decision


def _set_bundle_version():
    payload = {
        "protocol_sha256": shadow._sha256(PROTOCOL_PATH),
        "source_manifest_sha256": shadow._sha256(shadow.SOURCE_MODEL_MANIFEST_PATH),
        "feature_manifest_sha256": shadow._sha256(shadow.FEATURE_SOURCE_MANIFEST_PATH),
        "model_sha256": shadow._sha256(shadow._source_model_path(shadow.MODEL_META_PATH)),
        "calibrator_sha256": shadow._sha256(shadow.MODEL_WEIGHTS_PATH),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    CONFIG["artifact_bundle_version"] = f"btc-preopen-v1:{digest[:16]}"
    shadow.CONFIG = CONFIG


def main():
    _install_runtime_adapter()
    required_paths = (
        shadow.SOURCE_MODEL_MANIFEST_PATH,
        shadow.FEATURE_SOURCE_MANIFEST_PATH,
        shadow.MODEL_WEIGHTS_PATH,
        shadow.MODEL_META_PATH,
        shadow._source_model_path(shadow.MODEL_META_PATH),
        RAW_CANDLES_PATH,
    )
    for required_path in required_paths:
        if not required_path.is_file():
            raise FileNotFoundError(f"Pre-open collection needs {required_path}")
    _set_bundle_version()
    shadow.run_shadow()


if __name__ == "__main__":
    main()
