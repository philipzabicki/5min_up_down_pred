"""Collection-only BTC entry-state session with versioned full-history ADX."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_shadow as shadow
from features.live_indicator_runtime_entry_state_v1 import (
    get_adx_latest_value_full_history,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/btc_entry_state_collection_protocol_20261004_v2.json"
CONFIG = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

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
    CONFIG["execution_assumption"].get("max_book_age_seconds") or 30.0
)
shadow.VARIANTS = ("collection_only",)

ADX_FEATURE = (
    "ADX_fit_3m_pop128_adxper142_atrper1160_matypadxmama_"
    "matypatrdema_matypnegdmhma_matypposdmfba_negdmper64_"
    "posdmper1855_qe0.1_qm0.3_tf0.8_stmc_sg20"
)
_MODEL_META = json.loads(shadow.MODEL_META_PATH.read_text(encoding="utf-8"))
ADX_FEATURE_INDEX = _MODEL_META["feature_columns"].index(ADX_FEATURE)
ADX_SPEC = shadow.live_runtime.load_indicator_specs(
    [ADX_FEATURE],
    source_label="versioned BTC entry-state ADX runtime",
)[0]
if ADX_SPEC.indicator != "ADX":
    raise RuntimeError(f"Unexpected indicator for {ADX_FEATURE}: {ADX_SPEC.indicator}")
ADX_PARAMS = ADX_SPEC.params
ADX_CONFIG = next(
    config
    for config in shadow.live_runtime.parse_fit_results(
        shadow.live_runtime.FIT_RESULTS_DIR
    )
    if config.get("feature_col") == ADX_FEATURE
)
ADX_CONFIG_PATH = ROOT / ADX_CONFIG["json_path"]


class _FullHistory:
    def __init__(self, opened, ohlcv):
        opened_ns = (
            pd.DatetimeIndex(pd.to_datetime(opened, utc=True))
            .as_unit("ns")
            .asi8.copy()
        )
        values = np.asarray(ohlcv, dtype=np.float64)
        capacity = len(opened_ns) + 100_000
        self.opened_ns = np.empty(capacity, dtype=np.int64)
        self.ohlcv = np.empty((capacity, values.shape[1]), dtype=np.float64)
        self.opened_ns[: len(opened_ns)] = opened_ns
        self.ohlcv[: len(values)] = values
        self.count = len(opened_ns)

    @property
    def last_opened(self):
        return pd.Timestamp(self.opened_ns[self.count - 1], tz="UTC")

    def append(self, opened, ohlcv):
        timestamp = pd.Timestamp(opened).tz_convert("UTC")
        opened_ns = int(timestamp.value)
        if opened_ns <= self.opened_ns[self.count - 1]:
            return
        if timestamp - self.last_opened != shadow.live_runtime.INTERVAL_DELTA:
            raise RuntimeError(
                "Entry-state ADX history is not contiguous: "
                f"{self.last_opened.isoformat()} -> {timestamp.isoformat()}"
            )
        if self.count == len(self.opened_ns):
            capacity = max(self.count + 1, self.count * 2)
            opened_ns_values = np.empty(capacity, dtype=np.int64)
            ohlcv_values = np.empty((capacity, self.ohlcv.shape[1]), dtype=np.float64)
            opened_ns_values[: self.count] = self.opened_ns[: self.count]
            ohlcv_values[: self.count] = self.ohlcv[: self.count]
            self.opened_ns = opened_ns_values
            self.ohlcv = ohlcv_values
        self.opened_ns[self.count] = opened_ns
        self.ohlcv[self.count] = np.asarray(ohlcv, dtype=np.float64)
        self.count += 1

    def through(self, opened):
        timestamp_ns = int(pd.Timestamp(opened).tz_convert("UTC").value)
        end = int(np.searchsorted(self.opened_ns[: self.count], timestamp_ns, side="right"))
        return self.ohlcv[:end]


FULL_HISTORY = None


def _load_full_history():
    global FULL_HISTORY
    raw_path = shadow.audit.resolve_raw_dataset_input_path(
        shadow.audit.MODELING_DATASET_SETTINGS
    )
    columns = ["Opened", *shadow.live_runtime.OHLCV_COLS]
    frame = pd.read_csv(raw_path, usecols=columns)
    frame["Opened"] = pd.to_datetime(frame["Opened"], utc=True, errors="coerce")
    frame = (
        frame.loc[frame["Opened"].le(shadow.ANCHOR_OPENED)]
        .dropna(subset=["Opened"])
        .sort_values("Opened")
        .drop_duplicates(subset=["Opened"], keep="last")
        .reset_index(drop=True)
    )
    if frame.empty or frame["Opened"].iloc[-1] != shadow.ANCHOR_OPENED:
        raise RuntimeError("Raw BTC source does not reach the frozen ADX history anchor")
    if not frame["Opened"].diff().iloc[1:].eq(shadow.live_runtime.INTERVAL_DELTA).all():
        raise RuntimeError("Raw BTC source has a gap in the full-history ADX prefix")
    FULL_HISTORY = _FullHistory(
        frame["Opened"],
        frame[shadow.live_runtime.OHLCV_COLS].to_numpy(dtype=np.float64, copy=False),
    )

    if not shadow.DATABASE_PATH.exists():
        return
    database_uri = f"file:{shadow.DATABASE_PATH.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    try:
        rows = connection.execute(
            "SELECT opened_utc,ohlcv_json FROM btc_candles "
            "WHERE opened_utc>? ORDER BY opened_utc",
            (shadow._utc_iso(shadow.ANCHOR_OPENED),),
        ).fetchall()
    finally:
        connection.close()
    for opened, payload in rows:
        FULL_HISTORY.append(pd.Timestamp(opened), json.loads(payload))


_base_artifact_hashes = shadow._artifact_hashes


def _artifact_hashes():
    result = _base_artifact_hashes()
    for path in (
        Path(__file__),
        ROOT / "features/live_indicator_runtime_entry_state_v1.py",
        ADX_CONFIG_PATH,
    ):
        result[str(path.relative_to(ROOT)).replace("\\", "/")] = shadow._sha256(path)
    return result


shadow._artifact_hashes = _artifact_hashes
_base_make_predictor = shadow._make_predictor


def _make_predictor(*args, **kwargs):
    predictor = _base_make_predictor(*args, **kwargs)
    build_feature_snapshot = predictor.build_feature_snapshot

    def _build_feature_snapshot(*snapshot_args, **snapshot_kwargs):
        snapshot = build_feature_snapshot(*snapshot_args, **snapshot_kwargs)
        full_ohlcv = FULL_HISTORY.through(predictor.opened_candles[-1])
        snapshot["vector"][0, ADX_FEATURE_INDEX] = get_adx_latest_value_full_history(
            ADX_PARAMS,
            full_ohlcv,
        )
        snapshot["nonfinite_feature_indices"] = tuple(
            int(index)
            for index in np.flatnonzero(~np.isfinite(snapshot["vector"][0, :]))
        )
        snapshot["indicator_nan_cols"] = tuple(
            feature
            for feature in snapshot["indicator_nan_cols"]
            if feature != ADX_FEATURE
        )
        return snapshot

    predictor.build_feature_snapshot = _build_feature_snapshot
    return predictor


shadow._make_predictor = _make_predictor
_base_process_decision_candle = shadow._process_decision_candle


def _process_decision_candle(*args, **kwargs):
    opened = args[6] if len(args) > 6 else kwargs["opened"]
    ohlcv = args[7] if len(args) > 7 else kwargs["ohlcv"]
    FULL_HISTORY.append(opened, ohlcv)
    return _base_process_decision_candle(*args, **kwargs)


shadow._process_decision_candle = _process_decision_candle
_base_record_market_decision = shadow._record_market_decision


def _entry_context(connection, market_start, market_end, available_at,
                   inference_finished, prediction_values, quote_observation):
    market_start = pd.Timestamp(market_start).tz_convert("UTC")
    market_end = pd.Timestamp(market_end).tz_convert("UTC")
    candle_opened = market_start - shadow.live_runtime.INTERVAL_DELTA
    row = connection.execute(
        "SELECT ohlcv_json,received_at_utc FROM btc_candles WHERE opened_utc=?",
        (shadow._utc_iso(candle_opened),),
    ).fetchone()
    current_price = None
    candle_received_at = None
    volatility_per_minute = None
    if row:
        current_price = float(json.loads(row[0])[3])
        candle_received_at = pd.to_datetime(row[1], utc=True)
        history_rows = connection.execute(
            "SELECT ohlcv_json FROM btc_candles WHERE opened_utc<=? "
            "ORDER BY opened_utc DESC LIMIT 31",
            (shadow._utc_iso(candle_opened),),
        ).fetchall()
        closes = np.asarray(
            [float(json.loads(item[0])[3]) for item in reversed(history_rows)],
            dtype=np.float64,
        )
        if len(closes) >= 3 and np.isfinite(closes).all() and (closes > 0).all():
            volatility_per_minute = float(
                np.std(np.diff(np.log(closes)), ddof=1)
            )

    quote_available_at = None
    for book in (quote_observation or {}).get("up_book", {}), (quote_observation or {}).get("down_book", {}):
        received = book.get("received_at_utc")
        if received:
            stamp = pd.to_datetime(received, utc=True)
            quote_available_at = stamp if quote_available_at is None else max(quote_available_at, stamp)
    gamma_received = (quote_observation or {}).get("gamma_received_at_utc")
    if gamma_received:
        stamp = pd.to_datetime(gamma_received, utc=True)
        quote_available_at = stamp if quote_available_at is None else max(quote_available_at, stamp)

    resolution_source = (quote_observation or {}).get("gamma_market_payload", {}).get(
        "resolutionSource"
    )
    remaining_seconds = max(0.0, (market_end - market_start).total_seconds())
    horizon_volatility = (
        None
        if volatility_per_minute is None
        else volatility_per_minute * np.sqrt(remaining_seconds / 60.0)
    )
    return {
        "frozen_decision_at_utc": shadow._utc_iso(market_start),
        "collector_recorded_at_utc": shadow._utc_iso(),
        "btc_model_id": "20261003_043549",
        "btc_prediction_kind": "prospective live inference; no historical fold",
        "current_btc_price": current_price,
        "current_btc_price_source": "Binance COIN-M BTCUSD index 1m candle close (proxy)",
        "current_btc_price_period_end_utc": shadow._utc_iso(market_start),
        "current_btc_candle_opened_utc": shadow._utc_iso(candle_opened),
        "current_btc_data_received_at_utc": None if row is None else shadow._utc_iso(candle_received_at),
        "current_btc_data_available_by_frozen_decision": bool(
            candle_received_at is not None and candle_received_at <= market_start
        ),
        "btc_data_received_at_utc": shadow._utc_iso(available_at),
        "btc_data_delay_after_frozen_decision_seconds": (
            pd.Timestamp(available_at).tz_convert("UTC") - market_start
        ).total_seconds(),
        "btc_prediction_finished_at_utc": shadow._utc_iso(inference_finished),
        "btc_prediction_delay_after_frozen_decision_seconds": (
            pd.Timestamp(inference_finished).tz_convert("UTC") - market_start
        ).total_seconds(),
        "btc_prediction_available_by_frozen_decision": bool(
            pd.Timestamp(inference_finished).tz_convert("UTC") <= market_start
        ),
        "reference_level_price": None,
        "reference_level_source": resolution_source,
        "reference_level_missing_reason": (
            "Gamma supplies the Chainlink BTC/USD TWAP 60s resolution URL but no "
            "market-start price; no exact-stream point-in-time archive is configured."
        ),
        "current_price_log_distance_from_reference": None,
        "seconds_to_market_end_at_frozen_decision": remaining_seconds,
        "prior_30_return_volatility_per_minute": volatility_per_minute,
        "projected_log_move_volatility_remaining_window": horizon_volatility,
        "standardized_reference_distance": None,
        "quote_available_at_utc": None if quote_available_at is None else shadow._utc_iso(quote_available_at),
        "quote_delay_after_frozen_decision_seconds": (
            None if quote_available_at is None
            else (quote_available_at - market_start).total_seconds()
        ),
        "quote_available_by_frozen_decision": bool(
            quote_available_at is not None and quote_available_at <= market_start
        ),
        "quote_age_seconds_at_collection": {
            side: (quote_observation or {}).get(f"{side}_book", {}).get(
                "book_timestamp_age_seconds"
            )
            for side in ("up", "down")
        },
        "market_model_probabilities_at_late_quote": {
            "market_only": prediction_values.get("market_only"),
            "market_plus_btc": prediction_values.get("market_plus_btc"),
            "availability_time_utc": None if quote_available_at is None else shadow._utc_iso(quote_available_at),
            "eligible_for_frozen_start_decision": False,
        },
        "collection_only": True,
        "order_submission_enabled": False,
        "redeem_enabled": False,
    }


def _record_market_decision(connection, accounts, market_data, market_id, market_start,
                            market_end, available_at, inference_started,
                            inference_finished, prediction_values, raw_btc,
                            quote_observation, quote_fields, weights):
    context = _entry_context(
        connection,
        market_start,
        market_end,
        available_at,
        inference_finished,
        prediction_values,
        quote_observation,
    )
    observation = dict(quote_observation or {})
    observation["entry_state_context"] = context
    frozen_predictions = dict(prediction_values)
    frozen_predictions["market_only"] = None
    frozen_predictions["market_plus_btc"] = None
    frozen_predictions["collection_only"] = None
    return _base_record_market_decision(
        connection,
        accounts,
        market_data,
        market_id,
        market_start,
        market_end,
        available_at,
        inference_started,
        inference_finished,
        frozen_predictions,
        raw_btc,
        observation,
        quote_fields,
        weights,
    )


shadow._record_market_decision = _record_market_decision


if __name__ == "__main__":
    _load_full_history()
    shadow.run_shadow()
