"""Prospective, public-read-only BTC/Polymarket collection for the frozen test."""
from __future__ import annotations

import hashlib
import io
import json
import queue as queue_module
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.special import expit, logit

import audit_feature_readiness as audit
import run as live_runtime
import run_btc_shadow as shadow
import run_btc_preopen_t59_reassessment as economic_replay
from features.reaction_profile_fixed_grid import load_state as load_rp_state
from features.volume_profile_fixed_range import load_state as load_vp_state
from utils.polymarket import (
    polymarket_fee_model_from_market,
    resolve_polymarket_up_down_tokens,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/research/btc_prospective_test_20261014.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
BUNDLE_PATH = ROOT / PROTOCOL["model_bundle_path"]
SELECTION_PATH = ROOT / PROTOCOL["frozen_selection_path"]
OUTPUT_DIR = ROOT / "data/analysis/polymarket/BTC/btc_prospective_test_20261014"
DB_PATH = OUTPUT_DIR / "prospective.sqlite3"
MANIFEST_PATH = OUTPUT_DIR / "session_manifest.json"
POLL_SECONDS = 0.25
SETTLEMENT_POLL_SECONDS = 30.0
MARKET_PREFIX = "btc-updown-5m"
ANCHOR_OPENED = pd.Timestamp("2026-10-06T23:59:00Z")


class PublicBtcCandleFeed:
    """Read-only consumer for the repository's configured public Binance streams."""

    def __init__(self):
        self.rows = queue_module.Queue(maxsize=300)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.threads = []
        self.active_sockets = {}
        self.pending_price = {}
        self.pending_volume = {}
        self.status = {
            target["label"]: {
                "connected": False, "connected_at_utc": None,
                "last_message_at_utc": None, "last_closed_candle_opened_utc": None,
                "reconnect_count": 0, "last_error": None,
            }
            for target in live_runtime.WS_TARGETS
        }

    def start(self):
        if self.threads:
            return
        if not live_runtime.WS_TARGETS:
            raise RuntimeError("Frozen runtime has no configured public Binance websocket feeds")
        for target in live_runtime.WS_TARGETS:
            thread = threading.Thread(
                target=self._run_target,
                args=(target,),
                daemon=True,
                name=f"btc-shadow-{target['role']}-{target['market_type']}",
            )
            thread.start()
            self.threads.append(thread)

    def stop(self):
        self.stop_event.set()
        with self.lock:
            sockets = list(self.active_sockets.values())
        for socket in sockets:
            try:
                socket.close()
            except Exception:
                pass
        for thread in self.threads:
            thread.join(timeout=2.0)

    def _update_status(self, label, **values):
        with self.lock:
            self.status.setdefault(label, {}).update(values)

    def _run_target(self, target):
        delay = 1.0
        label = target["label"]
        while not self.stop_event.is_set():
            def on_open(socket):
                self._update_status(
                    label, connected=True, connected_at_utc=_iso(), last_error=None,
                )
                with self.lock:
                    self.active_sockets[label] = socket

            def on_message(_socket, message):
                try:
                    self._handle_message(target, message)
                except Exception as exc:
                    self._update_status(label, last_error=f"{type(exc).__name__}:{exc}")

            def on_error(_socket, error):
                self._update_status(label, connected=False, last_error=str(error))

            def on_close(_socket, _code, message):
                self._update_status(label, connected=False, last_error=str(message or "websocket closed"))

            socket = None
            try:
                socket = live_runtime.WebSocketApp(
                    target["url"], on_open=on_open, on_message=on_message,
                    on_error=on_error, on_close=on_close,
                )
                socket.run_forever(
                    ping_interval=live_runtime.WS_PING_INTERVAL_SEC,
                    ping_timeout=live_runtime.WS_PING_TIMEOUT_SEC,
                )
            except Exception as exc:
                self._update_status(label, connected=False, last_error=f"{type(exc).__name__}:{exc}")
            finally:
                with self.lock:
                    self.active_sockets.pop(label, None)
                    state = self.status.setdefault(label, {})
                    state["connected"] = False
                    state["reconnect_count"] = int(state.get("reconnect_count", 0)) + 1
            if self.stop_event.wait(delay):
                break
            delay = min(delay * 2.0, 60.0)

    @staticmethod
    def _event_time(data):
        try:
            return _iso(pd.to_datetime(int(data["E"]), unit="ms", utc=True))
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    def _handle_message(self, target, message):
        data = live_runtime._load_ws_payload(message)
        if not isinstance(data, dict):
            return
        kline = data.get("k") or {}
        if not kline or not bool(kline.get("x", False)):
            return
        opened = _utc(pd.to_datetime(int(kline["t"]), unit="ms", utc=True))
        received = _iso()
        event_at = self._event_time(data)
        label = target["label"]
        self._update_status(
            label, last_message_at_utc=received,
            last_closed_candle_opened_utc=_iso(opened),
        )
        payload_symbol = (
            data.get("ps") if target["source"] == "index" else kline.get("s")
        )
        if payload_symbol and str(payload_symbol).upper() != str(target["symbol"]).upper():
            raise RuntimeError(f"Websocket symbol mismatch for {label}: {payload_symbol}")

        if target["role"] == "price":
            if target["source"] == "index":
                synthetic_volume = max(
                    int(kline.get("n", 0) or 0), int(live_runtime.INDEX_PRICE_SYNTHETIC_VOLUME_DEFAULT)
                )
                price_volume = float(synthetic_volume)
            else:
                price_volume = float(kline["v"])
            price = {
                "t": int(kline["t"]), "o": float(kline["o"]),
                "h": float(kline["h"]), "l": float(kline["l"]),
                "c": float(kline["c"]), "v": price_volume,
                "event_at_utc": event_at, "received_at_utc": received,
            }
            if live_runtime.VOLUME_SOURCE == live_runtime.PRICE_SOURCE:
                self._emit(opened, price, {
                    "volume": price["v"], "auxiliary_ohlc": {},
                    "event_at_utc": price["event_at_utc"],
                    "received_at_utc": price["received_at_utc"],
                })
            else:
                with self.lock:
                    self.pending_price[opened] = price
                    volume = self.pending_volume.pop(opened, None)
                    self._trim_pending(opened)
                if volume is not None:
                    self._emit(opened, price, volume)
        elif target["role"] == "volume":
            volume = {
                "volume": float(kline["v"]),
                "auxiliary_ohlc": live_runtime._extract_live_auxiliary_ohlc_from_kline(kline),
                "event_at_utc": event_at, "received_at_utc": received,
            }
            with self.lock:
                self.pending_volume[opened] = volume
                price = self.pending_price.pop(opened, None)
                self._trim_pending(opened)
            if price is not None:
                self._emit(opened, price, volume)

    def _trim_pending(self, newest):
        cutoff = newest - pd.Timedelta(minutes=30)
        self.pending_price = {key: value for key, value in self.pending_price.items() if key >= cutoff}
        self.pending_volume = {key: value for key, value in self.pending_volume.items() if key >= cutoff}

    def _emit(self, opened, price, volume):
        aux = volume.get("auxiliary_ohlc", {})
        record = {
            "Opened": opened,
            "Open": price["o"], "High": price["h"], "Low": price["l"],
            "Close": price["c"], "Volume": volume["volume"],
            **aux,
        }
        received = max(
            _utc(price["received_at_utc"]), _utc(volume["received_at_utc"])
        )
        metadata = {
            "source": "binance_public_websocket",
            "price_event_at_utc": price.get("event_at_utc"),
            "price_received_at_utc": price["received_at_utc"],
            "volume_event_at_utc": volume.get("event_at_utc"),
            "volume_received_at_utc": volume["received_at_utc"],
            "received_at_utc": _iso(received),
            "price_stream": live_runtime.PRICE_STREAM_NAME,
            "volume_stream": live_runtime.VOLUME_STREAM_NAME,
        }
        try:
            self.rows.put_nowait({"row": record, "received_at_utc": _iso(received), "source_metadata": metadata})
        except queue_module.Full:
            with self.lock:
                state = self.status.setdefault("feed", {})
                state["dropped_closed_candle_count"] = int(state.get("dropped_closed_candle_count", 0)) + 1

    def drain(self):
        result = []
        while True:
            try:
                result.append(self.rows.get_nowait())
            except queue_module.Empty:
                return sorted(result, key=lambda item: item["row"]["Opened"])

    def status_snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.status))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value=None) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _iso(value=None) -> str:
    return _utc(value).isoformat()


def _canonical_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _artifact_manifest() -> dict:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    if bundle.get("status") != "verified" or bundle.get("raw_replay_verification", {}).get("status") != "verified":
        raise RuntimeError("Frozen A_frozen bundle lacks a verified raw replay")
    if bundle.get("raw_replay_verification", {}).get("target_available_for_prediction") is not False:
        raise RuntimeError("Frozen A_frozen replay does not prove causal target availability")
    model_meta = json.loads((ROOT / bundle["model_meta_path"]).read_text(encoding="utf-8"))
    frozen_feature_order = json.loads((ROOT / bundle["feature_order_path"]).read_text(encoding="utf-8"))
    if frozen_feature_order.get("feature_order") != model_meta.get("feature_columns"):
        raise RuntimeError("Frozen model feature-order artifact differs from its metadata")
    paths = {
        "protocol": PROTOCOL_PATH,
        "bundle": BUNDLE_PATH,
        "selection": SELECTION_PATH,
        "model": ROOT / bundle["model_path"],
        "model_meta": ROOT / bundle["model_meta_path"],
        "calibrator": ROOT / bundle["calibrator_path"],
        "feature_order": ROOT / bundle["feature_order_path"],
        "feature_sources": ROOT / bundle["feature_sources_path"],
        "volume_profile_config": ROOT / bundle["volume_profile_config_path"],
        "reaction_profile_config": ROOT / bundle["reaction_profile_config_path"],
        "profile_vp_npz": ROOT / bundle["profile_state_artifacts"][1],
        "profile_vp_json": ROOT / bundle["profile_state_artifacts"][1].replace(".npz", ".json"),
        "profile_rp_npz": ROOT / bundle["profile_state_artifacts"][0],
        "profile_rp_json": ROOT / bundle["profile_state_artifacts"][0].replace(".npz", ".json"),
        "frozen_probability_config": ROOT / PROTOCOL["probability_configuration_path"],
        "nested_oof": ROOT / PROTOCOL["nested_oof_path"],
        "raw_candles": ROOT / bundle["raw_candle_path"],
        "collector": Path(__file__).resolve(),
        "power_analysis_code": ROOT / "analyze_btc_prospective_power_20261014.py",
        "power_analysis_report": ROOT / "reports/btc_prospective_test_20261014/power_and_observation_plan.md",
        "power_analysis_results": ROOT / "reports/btc_prospective_test_20261014/power_analysis.json",
        "result_analysis_code": ROOT / "analyze_btc_prospective_results_20261014.py",
        "public_smoke_runner": ROOT / "smoke_btc_prospective_public_20261014.py",
        "shadow_public_fetch_helpers": ROOT / "run_btc_shadow.py",
        "runtime": ROOT / "run.py",
        "audit_runtime": ROOT / "audit_feature_readiness.py",
        "economic_replay": ROOT / "run_btc_preopen_economic_replay.py",
        "t59_reassessment": ROOT / "run_btc_preopen_t59_reassessment.py",
        "btc5m_fee_rules": ROOT / "utils/polymarket_btc5m_fee_rules.py",
        "polymarket_market_helpers": ROOT / "utils/polymarket.py",
        "contract": ROOT / "features/btc_preopen_contract.py",
        "research_contract": ROOT / "docs/BTC_RESEARCH_CONTRACT.md",
        "performance_contract": ROOT / "performance.md",
    }
    fit_dir = ROOT / bundle["indicator_fit_results_dir"]
    for fit_result in sorted(fit_dir.glob("*.json")):
        paths[f"fitted_indicator:{fit_result.name}"] = fit_result
    for feature_source in sorted((ROOT / "features").rglob("*.py")):
        paths[f"feature_source:{feature_source.relative_to(ROOT).as_posix()}"] = feature_source
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Frozen prospective input missing: {key}: {path}")
    if _sha256(BUNDLE_PATH) != PROTOCOL["model_bundle_sha256"]:
        raise RuntimeError("The frozen evaluated BTC bundle hash changed")
    if _sha256(SELECTION_PATH) != PROTOCOL["frozen_selection_sha256"]:
        raise RuntimeError("The frozen market probability selection hash changed")
    if _sha256(ROOT / PROTOCOL["nested_oof_path"]) != PROTOCOL["nested_oof_sha256"]:
        raise RuntimeError("Nested development OOF identity changed")
    if _sha256(ROOT / bundle["model_path"]) != bundle["model_sha256"]:
        raise RuntimeError("Frozen evaluated model hash differs from its bundle")
    if _sha256(ROOT / bundle["calibrator_path"]) != bundle["calibrator_sha256"]:
        raise RuntimeError("Frozen evaluated calibrator hash differs from its bundle")
    if _sha256(ROOT / bundle["raw_candle_path"]) != bundle["raw_candle_sha256"]:
        raise RuntimeError("Frozen A_frozen raw candle source hash differs from its bundle")
    return {key: _sha256(path) for key, path in paths.items()}


def _connect(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS session (
          id INTEGER PRIMARY KEY CHECK(id=1), started_at_utc TEXT NOT NULL,
          first_market_start_utc TEXT, evaluation_deadline_utc TEXT,
          protocol_sha256 TEXT NOT NULL, artifact_manifest_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS btc_candles (
          opened_utc TEXT PRIMARY KEY, ohlcv_json TEXT NOT NULL,
          futures_close REAL, received_at_utc TEXT NOT NULL,
          source_metadata_json TEXT
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
          id INTEGER PRIMARY KEY CHECK(id=1), last_opened_utc TEXT NOT NULL,
          frozen_hash TEXT NOT NULL, ohlcv_blob BLOB NOT NULL,
          vp_json_blob BLOB NOT NULL, vp_npz_blob BLOB NOT NULL,
          rp_json_blob BLOB NOT NULL, rp_npz_blob BLOB NOT NULL,
          updated_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS market_attempts (
          market_slug TEXT PRIMARY KEY, market_start_utc TEXT NOT NULL,
          decision_deadline_utc TEXT NOT NULL, attempt_recorded_at_utc TEXT NOT NULL,
          attempt_status TEXT NOT NULL, reason TEXT,
          btc_input_opened_utc TEXT, btc_data_received_at_utc TEXT,
          prediction_finished_at_utc TEXT, t60_snapshot_json TEXT,
          t59_snapshot_json TEXT, probabilities_json TEXT, policy_json TEXT,
          timings_json TEXT NOT NULL, source_identity_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS official_outcomes (
          market_slug TEXT PRIMARY KEY, condition_id TEXT, target_up INTEGER NOT NULL,
          resolved_at_utc TEXT, retrieved_at_utc TEXT NOT NULL, payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS collector_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, observed_at_utc TEXT NOT NULL,
          event_type TEXT NOT NULL, payload_json TEXT NOT NULL
        );
        """
    )
    candle_columns = {row[1] for row in connection.execute("PRAGMA table_info(btc_candles)")}
    if "source_metadata_json" not in candle_columns:
        connection.execute("ALTER TABLE btc_candles ADD COLUMN source_metadata_json TEXT")
        connection.commit()
    return connection


def _start_session(connection, artifacts):
    row = connection.execute("SELECT * FROM session WHERE id=1").fetchone()
    if row:
        if row[4] != _sha256(PROTOCOL_PATH) or json.loads(row[5]) != artifacts:
            raise RuntimeError("Existing prospective database has different frozen inputs")
        return _utc(row[1])
    started_at = _utc()
    connection.execute(
        "INSERT INTO session(id,started_at_utc,protocol_sha256,artifact_manifest_json) VALUES (1,?,?,?)",
        (_iso(started_at), _sha256(PROTOCOL_PATH), json.dumps(artifacts, sort_keys=True)),
    )
    connection.commit()
    MANIFEST_PATH.write_text(
        json.dumps({
            "protocol": PROTOCOL,
            "protocol_sha256": _sha256(PROTOCOL_PATH),
            "artifact_sha256": artifacts,
            "session_started_at_utc": _iso(started_at),
            "trading_enabled": False,
            "order_submission_count": 0,
            "smoke_data_included": False,
            "settlement_grace_deadline_utc": None,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return started_at


def _runtime_requirements(meta, fit_results_dir, destination):
    specs = audit.load_indicator_specs(
        meta["feature_columns"], source_label="frozen A_frozen model", fit_results_dir=fit_results_dir
    )
    windows = {
        str(spec.feature_col): max(int(spec.required_candles), 1)
        for spec in specs
    }
    if not windows:
        raise RuntimeError("Frozen evaluated model has no fitted indicator history requirements")
    payload = {
        "validation_status": "parameter_based_warmup_only_empirical_stability_not_audited",
        "unstable_feature_count": 0,
        "global_required_stable_window": max(windows.values()),
        "required_stable_window_by_feature": windows,
        "generated_from_frozen_model": True,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination, audit.load_indicator_history_requirements(
        destination, indicator_specs=specs, allow_unstable=False
    )


def _load_anchor_frame(raw_path: Path, required_rows: int) -> pd.DataFrame:
    cols = ["Opened", *live_runtime.OHLCV_COLS, "UM_BTCUSDT_Close"]
    tail = pd.DataFrame()
    for chunk in pd.read_csv(raw_path, usecols=cols, chunksize=200_000):
        opened = pd.to_datetime(chunk["Opened"], utc=True, errors="coerce")
        chunk = chunk.loc[opened.le(ANCHOR_OPENED)].copy()
        if chunk.empty:
            continue
        chunk["Opened"] = opened.loc[chunk.index]
        tail = pd.concat([tail, chunk], ignore_index=True).tail(required_rows)
    tail = tail.dropna(subset=["Opened"]).sort_values("Opened").drop_duplicates("Opened", keep="last")
    tail = tail.tail(required_rows).reset_index(drop=True)
    if len(tail) != required_rows or pd.Timestamp(tail.Opened.iloc[-1]) != ANCHOR_OPENED:
        raise RuntimeError("Frozen BTC raw source does not contain contiguous A_frozen anchor history")
    if not pd.to_datetime(tail.Opened, utc=True).diff().iloc[1:].eq(live_runtime.INTERVAL_DELTA).all():
        raise RuntimeError("Frozen A_frozen bootstrap history contains a gap")
    return tail


def _feature_runtime_config(bundle):
    vp_payload = json.loads((ROOT / bundle["volume_profile_config_path"]).read_text(encoding="utf-8"))
    rp_payload = json.loads((ROOT / bundle["reaction_profile_config_path"]).read_text(encoding="utf-8"))
    vp = json.loads(vp_payload["config_signature"])
    rp = json.loads(rp_payload["config_signature"])
    vp_config = {
        key: vp[key] for key in ("enabled", "price_min", "price_max", "neighbor_bins", "eps")
    }
    vp_config["horizons"] = {
        name: {key: value for key, value in horizons.items() if key in {
            "step", "local_window", "sigma_divisor", "min_sigma", "half_life_candles"
        }}
        for name, horizons in vp["horizons"].items()
    }
    rp_config = {
        key: rp[key] for key in (
            "enabled", "price_min", "price_max", "bin_size", "neighbor_bins",
            "eps", "min_reaction_strength", "wick_power", "distance_power",
        )
    }
    rp_config["horizons"] = {
        name: {key: value for key, value in horizons.items() if key in {"local_window", "half_life_candles"}}
        for name, horizons in rp["horizons"].items()
    }
    return {
        "volume_profile_fixed_range": vp_config,
        "reaction_profile_fixed_grid": rp_config,
        "basis_premium_features": {
            "enabled": True, "index_close_col": "Close",
            "futures_close_col": "UM_BTCUSDT_Close", "eps": 1e-12,
        },
    }


def _build_predictor(connection, bundle, requirement_path, runtime_window, runtime_dir=None):
    meta_path = ROOT / bundle["model_meta_path"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_path = ROOT / bundle["model_path"]
    runtime_meta = dict(meta)
    runtime_meta["artifacts"] = {"final_model_path": str(model_path.resolve())}
    runtime_meta.update(_feature_runtime_config(bundle))
    runtime_meta_path = (OUTPUT_DIR / "runtime" if runtime_dir is None else runtime_dir) / "lgbm_meta.json"
    runtime_meta_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_meta_path.write_text(json.dumps(runtime_meta, indent=2, sort_keys=True), encoding="utf-8")

    runtime_config = _feature_runtime_config(bundle)
    vp_path = ROOT / bundle["profile_state_artifacts"][1]
    rp_path = ROOT / bundle["profile_state_artifacts"][0]
    vp_state, rp_state = load_vp_state(vp_path), load_rp_state(rp_path)
    raw_path = ROOT / bundle["raw_candle_path"]
    bootstrap = _load_anchor_frame(raw_path, runtime_window)
    predictor = audit.PseudoLiveAuditPredictor(
        bootstrap,
        model_meta_path=runtime_meta_path,
        max_keep=runtime_window,
        volume_profile_state=vp_state,
        reaction_profile_state=rp_state,
        feature_runtime_config=runtime_config,
        fit_results_dir=ROOT / bundle["indicator_fit_results_dir"],
        indicator_history_requirements_path=requirement_path,
        allow_unstable_indicator_summary=False,
    )
    row = connection.execute("SELECT opened_utc FROM btc_candles WHERE opened_utc>? ORDER BY opened_utc", (_iso(ANCHOR_OPENED),)).fetchall()
    if row:
        candles = _read_candles_after(connection, ANCHOR_OPENED)
        _advance_predictor(predictor, candles)
    return predictor, runtime_meta_path


def _read_candles_after(connection, opened):
    rows = connection.execute(
        "SELECT opened_utc,ohlcv_json,futures_close,received_at_utc FROM btc_candles WHERE opened_utc>? ORDER BY opened_utc",
        (_iso(opened),),
    ).fetchall()
    records = []
    for stamp, raw, futures_close, received in rows:
        record = {"Opened": _utc(stamp), "received_at_utc": received}
        record.update(dict(zip(live_runtime.OHLCV_COLS, json.loads(raw))))
        record["UM_BTCUSDT_Close"] = futures_close
        records.append(record)
    return pd.DataFrame(records)


def _advance_predictor(predictor, candles):
    for row in candles.itertuples(index=False):
        opened = _utc(row.Opened)
        predictor._append_new_candle(
            opened,
            tuple(float(getattr(row, col)) for col in live_runtime.OHLCV_COLS),
            basis_futures_close=float(row.UM_BTCUSDT_Close),
        )
        predictor._update_volume_profile_state_for_latest_candle(opened, persist=False)
        predictor._update_reaction_profile_state_for_latest_candle(opened, persist=False)


def _advance_websocket_feed(connection, predictor, feed):
    updates = feed.drain()
    advanced = 0
    for update in updates:
        row = update["row"]
        opened = _utc(row["Opened"])
        last_opened = _utc(predictor.opened_candles[-1])
        if opened <= last_opened:
            continue
        expected = last_opened + live_runtime.INTERVAL_DELTA
        if opened > expected:
            gap_end = opened - live_runtime.INTERVAL_DELTA
            catchup, request_started, received, elapsed = _timed_fetch_catchup_pair(expected, gap_end)
            _advance_predictor(predictor, catchup)
            gap_source = {
                "source": "public_rest_gap_repair",
                "request_started_at_utc": _iso(request_started),
                "received_at_utc": _iso(received),
                "request_elapsed_seconds": elapsed,
            }
            _persist_candles(connection, catchup.itertuples(index=False), received, gap_source)
            last_opened = _utc(predictor.opened_candles[-1])
        if opened != last_opened + live_runtime.INTERVAL_DELTA:
            raise RuntimeError(f"Websocket candle sequence mismatch: expected {last_opened + live_runtime.INTERVAL_DELTA}, got {opened}")
        frame = pd.DataFrame([row])
        _advance_predictor(predictor, frame)
        _persist_candle(
            connection, next(frame.itertuples(index=False)),
            update["received_at_utc"], update["source_metadata"],
        )
        advanced += 1
    return advanced


def _advance_websocket_feed_until(connection, predictor, feed, target_opened, wait_budget_seconds):
    target_opened = _utc(target_opened)
    deadline = time.monotonic() + max(0.0, float(wait_budget_seconds))
    while True:
        _advance_websocket_feed(connection, predictor, feed)
        last_opened = _utc(predictor.opened_candles[-1])
        if last_opened >= target_opened:
            return last_opened
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return last_opened
        time.sleep(min(0.02, remaining))


def _websocket_pair_wait_budget(t59_deadline):
    timing = PROTOCOL["market_timing"]
    remaining = (
        _utc(t59_deadline) - _utc()
    ).total_seconds() - float(timing["t59_compute_reserve_seconds"])
    return min(float(timing["websocket_pair_wait_cap_seconds"]), max(0.0, remaining))


def _wait_for_public_feed_connections(feed, timeout_seconds=30.0):
    connect_deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < connect_deadline:
        states = feed.status_snapshot()
        if states and all(state.get("connected") for state in states.values()):
            return states
        time.sleep(0.1)
    return feed.status_snapshot()


def _backfill_predictor_to_latest_closed(connection, predictor, source):
    latest = _utc().floor("min") - live_runtime.INTERVAL_DELTA
    last_opened = _utc(predictor.opened_candles[-1])
    if latest <= last_opened:
        return None
    candles, request_started, received, elapsed = _timed_fetch_catchup_pair(
        last_opened + live_runtime.INTERVAL_DELTA, latest
    )
    _advance_predictor(predictor, candles)
    metadata = {
        "source": source,
        "request_started_at_utc": _iso(request_started),
        "received_at_utc": _iso(received),
        "request_elapsed_seconds": elapsed,
    }
    _persist_candles(connection, candles.itertuples(index=False), received, metadata)
    return {
        "first_opened_utc": _iso(candles.Opened.iloc[0]),
        "last_opened_utc": _iso(latest),
        "request_elapsed_seconds": elapsed,
        "received_at_utc": _iso(received),
    }


def _restore_checkpoint(connection, frozen_hash, bundle, requirement_path, runtime_window):
    row = connection.execute(
        "SELECT last_opened_utc,frozen_hash,ohlcv_blob,vp_json_blob,vp_npz_blob,rp_json_blob,rp_npz_blob FROM checkpoints WHERE id=1"
    ).fetchone()
    if row is None:
        return None
    if row[1] != frozen_hash:
        raise RuntimeError("Checkpoint has a different frozen input identity")
    with np.load(io.BytesIO(row[2]), allow_pickle=False) as blob:
        opened = pd.to_datetime(blob["opened_ns"], utc=True)
        frame = pd.DataFrame(blob["ohlcv"], columns=live_runtime.OHLCV_COLS)
        frame.insert(0, "Opened", opened)
        if "basis_futures_close" in blob:
            frame["UM_BTCUSDT_Close"] = blob["basis_futures_close"]
    vp_state = shadow._load_profile_blob(row[3], row[4], shadow.load_volume_profile_state)
    rp_state = shadow._load_profile_blob(row[5], row[6], shadow.load_reaction_profile_state)
    predictor = _make_predictor(frame, bundle, requirement_path, runtime_window, vp_state, rp_state)
    later = _read_candles_after(connection, _utc(row[0]))
    if not later.empty:
        _advance_predictor(predictor, later)
    return predictor


def _make_predictor(bootstrap, bundle, requirement_path, runtime_window, vp_state, rp_state):
    meta_path = ROOT / bundle["model_meta_path"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_path = ROOT / bundle["model_path"]
    runtime_meta = dict(meta)
    runtime_meta["artifacts"] = {"final_model_path": str(model_path.resolve())}
    runtime_meta.update(_feature_runtime_config(bundle))
    runtime_meta_path = OUTPUT_DIR / "runtime" / "lgbm_meta.json"
    runtime_meta_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_meta_path.write_text(json.dumps(runtime_meta, indent=2, sort_keys=True), encoding="utf-8")
    runtime_config = _feature_runtime_config(bundle)
    predictor = audit.PseudoLiveAuditPredictor(
        bootstrap,
        model_meta_path=runtime_meta_path,
        max_keep=runtime_window,
        volume_profile_state=vp_state,
        reaction_profile_state=rp_state,
        feature_runtime_config=runtime_config,
        fit_results_dir=ROOT / bundle["indicator_fit_results_dir"],
        indicator_history_requirements_path=requirement_path,
        allow_unstable_indicator_summary=False,
    )
    return predictor, runtime_meta_path


def _encode_candle(row, received_at, source_metadata=None):
    return (
        _iso(row.Opened),
        json.dumps([float(getattr(row, col)) for col in live_runtime.OHLCV_COLS], separators=(",", ":")),
        float(row.UM_BTCUSDT_Close),
        _iso(received_at),
        _json(source_metadata or {"source": "public_rest_catchup", "received_at_utc": _iso(received_at)}),
    )


def _persist_candle(connection, row, received_at, source_metadata=None):
    _persist_candles(connection, [row], received_at, source_metadata)


def _persist_candles(connection, rows, received_at, source_metadata=None):
    records = [_encode_candle(row, received_at, source_metadata) for row in rows]
    if not records:
        return
    connection.executemany(
        "INSERT OR IGNORE INTO btc_candles(opened_utc,ohlcv_json,futures_close,received_at_utc,source_metadata_json) VALUES (?,?,?,?,?)",
        records,
    )
    connection.commit()


def _persisted_candle(connection, opened):
    row = connection.execute(
        "SELECT opened_utc,ohlcv_json,futures_close,received_at_utc,source_metadata_json FROM btc_candles WHERE opened_utc=?",
        (_iso(opened),),
    ).fetchone()
    if row is None:
        return None
    record = {"Opened": _utc(row[0]), "received_at_utc": row[3]}
    record.update(dict(zip(live_runtime.OHLCV_COLS, json.loads(row[1]))))
    record["UM_BTCUSDT_Close"] = row[2]
    record["source_metadata"] = {} if row[4] is None else json.loads(row[4])
    return record


def _next_market_start(now):
    stamp = _utc(now).floor("5min")
    if stamp <= _utc(now):
        stamp += pd.Timedelta(minutes=5)
    if stamp - pd.Timedelta(minutes=1) <= _utc(now):
        stamp += pd.Timedelta(minutes=5)
    return stamp


def _wait_until(target):
    target = _utc(target)
    while True:
        remaining = (target - _utc()).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def _market_metadata(session, market_start):
    slug = f"{MARKET_PREFIX}-{int(_utc(market_start).timestamp())}"
    started = _utc()
    response = session.get(f"https://gamma-api.polymarket.com/markets/slug/{slug}", timeout=8)
    response.raise_for_status()
    market = response.json()
    received = _utc()
    tokens = resolve_polymarket_up_down_tokens(market)
    start_value = market.get("eventStartTime") or next(
        (event.get("startTime") for event in market.get("events", []) if event.get("startTime")), None
    )
    actual_start = None if not start_value else _utc(start_value)
    end_value = market.get("endDate")
    actual_end = None if not end_value else _utc(end_value)
    identity = (
        market.get("slug") == slug and actual_start == _utc(market_start)
        and actual_end == _utc(market_start) + pd.Timedelta(minutes=5)
    )
    return {
        "slug": slug,
        "market": market,
        "tokens": tokens,
        "requested_at_utc": _iso(started),
        "received_at_utc": _iso(received),
        "market_start_utc": None if actual_start is None else _iso(actual_start),
        "market_end_utc": None if actual_end is None else _iso(actual_end),
        "identity_valid": bool(identity),
        "market_accepting_orders": bool(market.get("acceptingOrders", False)),
        "market_closed": bool(market.get("closed", False)),
        "condition_id": str(market.get("conditionId") or ""),
        "fee_model": polymarket_fee_model_from_market(market),
    }


def _collect_book_futures(up_future, down_future):
    books, errors = {}, {}
    for side, future in (("up", up_future), ("down", down_future)):
        try:
            books[side] = future.result()
        except Exception as exc:
            books[side] = None
            errors[side] = f"{type(exc).__name__}:{exc}"
    return books, errors


def _quote_snapshot(metadata, snapshot_name, books=None, book_errors=None):
    started = _utc()
    started_clock = time.perf_counter()
    request_duration = None
    if books is None:
        book_sessions = [requests.Session(), requests.Session()]
        for book_session in book_sessions:
            book_session.headers.update({"User-Agent": "btc-prospective-shadow/1.0"})
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                up_future = pool.submit(shadow._book_observation, book_sessions[0], metadata["tokens"]["up"])
                down_future = pool.submit(shadow._book_observation, book_sessions[1], metadata["tokens"]["down"])
                books, book_errors = _collect_book_futures(up_future, down_future)
            request_duration = time.perf_counter() - started_clock
        finally:
            for book_session in book_sessions:
                book_session.close()
    book_errors = book_errors or {}
    up, down = books["up"], books["down"]
    available_books = [book for book in (up, down) if book is not None]
    started = min(
        (_utc(book["request_started_at_utc"]) for book in available_books), default=started
    )
    received = max(
        (_utc(book["received_at_utc"]) for book in available_books), default=_utc()
    )
    fields = {}
    for side, book in (("up", up), ("down", down)):
        fields[f"{side}_best_bid"] = None if book is None else book["best_bid"]
        fields[f"{side}_best_ask"] = None if book is None else book["best_ask"]
        fields[f"{side}_bid_size"] = None if book is None else book["best_bid_size"]
        fields[f"{side}_ask_size"] = None if book is None else book["best_ask_size"]
    numbers = np.asarray(list(fields.values()), dtype=float)
    valid = bool(
        up is not None and down is not None
        and metadata["identity_valid"] and np.isfinite(numbers).all()
        and metadata["market_accepting_orders"] and not metadata["market_closed"]
        and all(0 < fields[f"{side}_best_bid"] <= fields[f"{side}_best_ask"] < 1 for side in ("up", "down"))
        and all(fields[f"{side}_ask_size"] >= 0 for side in ("up", "down"))
        and all(shadow._book_timestamp_age_is_acceptable(book["book_timestamp_age_seconds"]) for book in available_books)
        and len(available_books) == 2
    )
    mids = {
        side: None if fields[f"{side}_best_bid"] is None or fields[f"{side}_best_ask"] is None
        else (fields[f"{side}_best_bid"] + fields[f"{side}_best_ask"]) / 2
        for side in ("up", "down")
    }
    p_market = float(mids["up"] / (mids["up"] + mids["down"])) if valid else None
    return {
        "snapshot_name": snapshot_name,
        "request_started_at_utc": _iso(started),
        "received_at_utc": _iso(received),
        "request_duration_seconds": request_duration if request_duration is not None else max(
            (float(book.get("request_elapsed_seconds", 0.0)) for book in available_books), default=0.0
        ),
        "market_slug": metadata["slug"],
        "condition_id": metadata["condition_id"],
        "market_identity_valid": metadata["identity_valid"],
        "market_accepting_orders": metadata["market_accepting_orders"],
        "market_closed": metadata["market_closed"],
        "quote_valid": valid,
        "quote_fields": fields,
        "mid_up": mids["up"] if valid else None,
        "mid_down": mids["down"] if valid else None,
        "p_market": p_market,
        "up_book": up,
        "down_book": down,
        "book_errors": book_errors,
        "fee_model": metadata["fee_model"],
        "order_min_size_usd": shadow._float(metadata["market"].get("orderMinSize"), 0.0),
        "gamma_market_payload": metadata["market"],
    }


def _probabilities(p_market, p_a):
    if not (0 < float(p_market) < 1 and 0 < float(p_a) < 1):
        raise ValueError("Frozen probabilities must be strictly between zero and one")
    logit_market = float(logit(np.clip(p_market, 1e-6, 1 - 1e-6)))
    logit_btc = float(logit(np.clip(p_a, 1e-6, 1 - 1e-6)))
    spec = PROTOCOL["model_specification"]
    c = spec["c_market_only"]
    d = spec["d_market_plus_btc"]
    return {
        "p_A_frozen": float(p_a),
        "p_C_market_only": float(expit(c["intercept"] + c["beta_logit_market"] * logit_market)),
        "p_D_market_plus_btc": float(expit(
            d["intercept"] + d["beta_logit_market"] * logit_market
            + d["beta_logit_btc_calibrated"] * logit_btc
        )),
    }


def _quality_row(attempt, outcome):
    if not outcome or outcome.get("target_up") not in (0, 1):
        return False
    if attempt.get("attempt_status") != "complete":
        return False
    t60 = attempt.get("t60_snapshot") or {}
    timings = attempt.get("timings") or {}
    probabilities = attempt.get("probabilities") or {}
    if not t60.get("quote_valid") or not timings.get("quality_inputs_ready_before_t59"):
        return False
    required = ("p_market", "p_A_frozen", "p_C_market_only", "p_D_market_plus_btc")
    for key in required:
        try:
            value = float(probabilities[key])
        except (KeyError, TypeError, ValueError):
            return False
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            return False
    return True


def _primary_verdict(interval, coverage_passed):
    lower, upper = (float(value) for value in interval)
    if not coverage_passed:
        return "inconclusive"
    if upper < 0.0:
        return "positive_incremental_information"
    if lower > 0.0:
        return "negative_incremental_information"
    return "inconclusive"


def _fetch_one_source(start_opened, end_opened, *, source, symbol, market_type):
    with requests.Session() as session:
        session.headers.update({"User-Agent": "btc-prospective-shadow/1.0"})
        return live_runtime._fetch_single_source_closed_ohlcv_range(
            session,
            start_opened=start_opened,
            end_opened=end_opened,
            source=source,
            symbol=symbol,
            market_type=market_type,
        )


def _fetch_catchup_pair(start_opened, end_opened):
    price_args = {
        "source": live_runtime.PRICE_SOURCE,
        "symbol": live_runtime.SYMBOL,
        "market_type": live_runtime.PRICE_MARKET,
    }
    volume_args = {
        "source": live_runtime.VOLUME_SOURCE,
        "symbol": live_runtime.VOLUME_SYMBOL,
        "market_type": live_runtime.VOLUME_MARKET,
    }
    if live_runtime.VOLUME_SOURCE == live_runtime.PRICE_SOURCE:
        return _fetch_one_source(start_opened, end_opened, **price_args)
    with ThreadPoolExecutor(max_workers=2) as pool:
        price_future = pool.submit(_fetch_one_source, start_opened, end_opened, **price_args)
        volume_future = pool.submit(_fetch_one_source, start_opened, end_opened, **volume_args)
        price = price_future.result()
        volume = volume_future.result()
    combined = live_runtime._merge_price_and_volume_frames(price, volume)
    live_runtime.validate_closed_ohlcv_catchup(combined, start_opened, end_opened)
    return combined


def _timed_fetch_catchup_pair(start_opened, end_opened):
    started = _utc()
    clock_started = time.perf_counter()
    frame = _fetch_catchup_pair(start_opened, end_opened)
    return frame, started, _utc(), time.perf_counter() - clock_started


def _execution_snapshot(snapshot, metadata):
    result = {
        "fee_known": True,
        "fee_rate_bps": None,
        "fee_collection_mode": "scheduled_btc5m_fee_rule",
        "no_future_event_at_entry": True,
        "no_future_source_event_at_entry": True,
        "archive_entry_hour_available": True,
        "archive_gap_after_fee_update": False,
        "fee_update_receive_ns": int(_utc(snapshot["received_at_utc"]).value),
    }
    for side in ("up", "down"):
        book = snapshot[f"{side}_book"]
        token_id = str(metadata["tokens"][side])
        result.update({
            f"{side}_quote_source_token_id": token_id,
            f"{side}_quote_complemented": False,
            f"{side}_ask_levels": [
                (float(level["price"]), float(level["size"]))
                for level in book["ask_levels"]
                if 0 < float(level["price"]) < 1 and float(level["size"]) > 0
            ],
            f"{side}_bid_levels": [
                (float(level["price"]), float(level["size"]))
                for level in book["bid_levels"]
                if 0 < float(level["price"]) < 1 and float(level["size"]) > 0
            ],
            f"{side}_best_ask": book["best_ask"],
            f"{side}_best_bid": book["best_bid"],
            f"{side}_ask_age_seconds": book["book_timestamp_age_seconds"],
            f"{side}_bid_ask_crossed": book["best_bid"] > book["best_ask"],
            f"{side}_ask_order_ambiguous": False,
            f"{side}_book_init_receive_ns": int(_utc(book["received_at_utc"]).value),
            f"archive_gap_after_{side}_book_init": False,
        })
    return result


def _policy_market(snapshot, metadata, p_a, market_start):
    return {
        "market_slug": metadata["slug"],
        "condition_id": metadata["condition_id"],
        "market_start_utc": _iso(market_start),
        "up_token_id": str(metadata["tokens"]["up"]),
        "down_token_id": str(metadata["tokens"]["down"]),
        "order_min_size_usd_current": float(snapshot["order_min_size_usd"] or 0.0),
        "p_candidate_platt": float(p_a),
    }


def _policy_at_t60(snapshot, p_a, metadata, market_start):
    stake = float(PROTOCOL["economics"]["independent_gross_stake_usd"])
    hurdle = stake * 0.01
    market = _policy_market(snapshot, metadata, p_a, market_start)
    decision = economic_replay._evaluate_market_sides(
        market, _execution_snapshot(snapshot, metadata), stake
    )
    side = decision["chosen_side"]
    ev = decision["chosen_ev_usd"]
    if side is None or ev is None or float(ev) < hurdle:
        return {
            "selected_side": None,
            "reason": "no_side_passes_frozen_net_ev_hurdle",
            "hurdle_usd": hurdle,
            "t60_side_evaluation": decision["sides"],
        }
    side_result = decision["sides"][side]
    fill = side_result["fill"]
    return {
        "selected_side": side,
        "reason": "frozen_a_frozen_1pct_net_ev_hurdle_passed",
        "decision_at_t60_only": True,
        "stake_usd": stake,
        "net_ev_usd": float(ev),
        "net_ev_fraction": float(ev) / stake,
        "hurdle_usd": hurdle,
        "p_side": float(side_result["probability"]),
        "t60_best_ask": float(snapshot["quote_fields"][f"{side}_best_ask"]),
        "t60_fill": fill,
        "t60_depth_basis": "complete_public_book_levels",
        "policy_market": market,
        "t60_side_evaluation": decision["sides"],
    }


def _policy_at_t59(policy, snapshot, metadata):
    side = policy.get("selected_side")
    if side is None:
        return {
            "selected_side": None,
            "t59_quote_valid": bool(snapshot["quote_valid"]),
            "actual_fill": "unknown",
        }
    if not snapshot.get("quote_valid"):
        return {
            "selected_side": side,
            "t59_quote_valid": False,
            "same_t60_side_retained": True,
            "simulated_execution": {
                "priceable": False,
                "fill": None,
                "no_fill_reason": "invalid_or_incomplete_t59_book",
                "execution_assumption": "unpriceable_public_book",
            },
            "actual_fill": "unknown",
        }
    market = dict(policy["policy_market"])
    market["p_candidate_platt"] = float(policy["p_side"] if side == "up" else 1.0 - policy["p_side"])
    result = economic_replay._price_side(
        _execution_snapshot(snapshot, metadata), market, side,
        float(PROTOCOL["economics"]["independent_gross_stake_usd"]),
    )
    return {
        "selected_side": side,
        "t59_quote_valid": bool(snapshot["quote_valid"]),
        "same_t60_side_retained": True,
        "simulated_execution": result,
        "actual_fill": "unknown",
    }


def _official_outcome(session, slug):
    result = shadow._settlement_payload(session, slug)
    return result


def _append_outcome(connection, market_slug, condition_id, outcome):
    connection.execute(
        "INSERT OR IGNORE INTO official_outcomes(market_slug,condition_id,target_up,resolved_at_utc,retrieved_at_utc,payload_json) VALUES (?,?,?,?,?,?)",
        (
            market_slug, condition_id, int(outcome["target_up"]), outcome["resolved_at_utc"],
            outcome["observed_at_utc"], json.dumps(outcome["source_payload"], sort_keys=True),
        ),
    )
    connection.commit()


def _append_attempt(connection, record):
    connection.execute(
        "INSERT INTO market_attempts(market_slug,market_start_utc,decision_deadline_utc,attempt_recorded_at_utc,attempt_status,reason,btc_input_opened_utc,btc_data_received_at_utc,prediction_finished_at_utc,t60_snapshot_json,t59_snapshot_json,probabilities_json,policy_json,timings_json,source_identity_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            record["market_slug"], record["market_start_utc"], record["decision_deadline_utc"],
            record["attempt_recorded_at_utc"], record["attempt_status"], record.get("reason"),
            record.get("btc_input_opened_utc"), record.get("btc_data_received_at_utc"),
            record.get("prediction_finished_at_utc"), _json(record.get("t60_snapshot")),
            _json(record.get("t59_snapshot")), _json(record.get("probabilities")),
            _json(record.get("policy")), _json(record.get("timings", {})),
            _json(record.get("source_identity", {})),
        ),
    )
    connection.commit()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=shadow._json_default, allow_nan=False)


def _log_event(connection, event_type, payload):
    connection.execute(
        "INSERT INTO collector_events(observed_at_utc,event_type,payload_json) VALUES (?,?,?)",
        (_iso(), event_type, _json(payload)),
    )
    connection.commit()


def _save_checkpoint(connection, predictor, frozen_hash):
    import io
    output = io.BytesIO()
    arrays = {"ohlcv": predictor.ohlcv_np, "opened_ns": predictor.opened_ns_np}
    if predictor.basis_futures_close_np is not None:
        arrays["basis_futures_close"] = predictor.basis_futures_close_np
    np.savez_compressed(output, **arrays)
    vp_json, vp_npz = shadow._save_profile_blob(predictor.volume_profile_state, shadow.save_volume_profile_state)
    rp_json, rp_npz = shadow._save_profile_blob(predictor.reaction_profile_state, shadow.save_reaction_profile_state)
    connection.execute(
        "INSERT INTO checkpoints(id,last_opened_utc,frozen_hash,ohlcv_blob,vp_json_blob,vp_npz_blob,rp_json_blob,rp_npz_blob,updated_at_utc) VALUES (1,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET last_opened_utc=excluded.last_opened_utc,frozen_hash=excluded.frozen_hash,ohlcv_blob=excluded.ohlcv_blob,vp_json_blob=excluded.vp_json_blob,vp_npz_blob=excluded.vp_npz_blob,rp_json_blob=excluded.rp_json_blob,rp_npz_blob=excluded.rp_npz_blob,updated_at_utc=excluded.updated_at_utc",
        (_iso(predictor.opened_candles[-1]), frozen_hash, output.getvalue(), sqlite3.Binary(vp_json), sqlite3.Binary(vp_npz), sqlite3.Binary(rp_json), sqlite3.Binary(rp_npz), _iso()),
    )
    connection.commit()


def _attempt_missing_schedule(connection, session_started, now):
    session_row = connection.execute(
        "SELECT first_market_start_utc,evaluation_deadline_utc FROM session WHERE id=1"
    ).fetchone()
    if not session_row or not session_row[0] or not session_row[1]:
        return
    first = _utc(session_row[0])
    window_deadline = _utc(session_row[1])
    current = _utc(now)
    market_start = first
    while market_start < window_deadline and market_start - pd.Timedelta(minutes=1) <= current:
        slug = f"{MARKET_PREFIX}-{int(market_start.timestamp())}"
        exists = connection.execute("SELECT 1 FROM market_attempts WHERE market_slug=?", (slug,)).fetchone()
        if not exists:
            _append_attempt(connection, {
                "market_slug": slug, "market_start_utc": _iso(market_start),
                "decision_deadline_utc": _iso(market_start - pd.Timedelta(minutes=1)),
                "attempt_recorded_at_utc": _iso(now), "attempt_status": "missed_cycle",
                "reason": "collector_started_late_or_was_unavailable_at_t60",
                "timings": {"schedule_reconstruction": True},
                "source_identity": {"protocol_sha256": _sha256(PROTOCOL_PATH)},
            })
        market_start += pd.Timedelta(minutes=5)


def _decision_attempt(predictor, model, calibrator, metadata, market_start, t60_snapshot, btc_received_at):
    deadline = _utc(market_start) - pd.Timedelta(minutes=1)
    input_opened = _utc(market_start) - pd.Timedelta(minutes=2)
    attempt = {
        "market_slug": metadata["slug"], "market_start_utc": _iso(market_start),
        "decision_deadline_utc": _iso(deadline), "attempt_recorded_at_utc": _iso(),
        "attempt_status": "missing", "reason": None,
        "btc_input_opened_utc": _iso(input_opened), "t60_snapshot": None,
        "t59_snapshot": None, "probabilities": None, "policy": None, "timings": {},
        "source_identity": {"protocol_sha256": _sha256(PROTOCOL_PATH)},
    }
    try:
        t60 = t60_snapshot
        attempt["t60_snapshot"] = t60
        t60_at = _utc(t60["request_started_at_utc"])
        attempt["timings"]["t60_request_start_offset_seconds"] = (t60_at - deadline).total_seconds()
        if not t60["quote_valid"]:
            attempt["reason"] = "invalid_t60_market_book"
            attempt["attempt_status"] = "late_or_incomplete"
        latest_opened = _utc(predictor.opened_candles[-1]) if predictor.opened_candles else None
        if latest_opened != input_opened:
            raise RuntimeError(f"T-60 predictor candle mismatch: expected {input_opened}, got {latest_opened}")
        feature_started = _utc()
        feature_clock = time.perf_counter()
        volume_values = predictor._prepare_volume_profile_features_for_latest_candle(input_opened)
        reaction_values = predictor._prepare_reaction_profile_features_for_latest_candle(input_opened)
        snapshot = predictor.build_feature_snapshot(
            volume_profile_values=volume_values, reaction_profile_values=reaction_values
        )
        feature_finished = _utc()
        feature_seconds = time.perf_counter() - feature_clock
        if snapshot["nonfinite_feature_indices"]:
            raise RuntimeError("A_frozen input contains nonfinite feature values")
        inference_started = _utc()
        inference_clock = time.perf_counter()
        raw_probability = float(model.predict(snapshot["vector"])[0])
        calibration = float(expit(
            float(calibrator["intercept"])
            + float(calibrator["coefficient"]) * float(logit(np.clip(raw_probability, 1e-6, 1 - 1e-6)))
        ))
        prediction_finished = _utc()
        inference_seconds = time.perf_counter() - inference_clock
        attempt["btc_data_received_at_utc"] = _iso(btc_received_at)
        attempt["prediction_finished_at_utc"] = _iso(prediction_finished)
        attempt["timings"].update({
            "t60_input_cycle_started_at_utc": t60["request_started_at_utc"],
            "t60_request_start_offset_seconds": (t60_at - deadline).total_seconds(),
            "btc_data_received_at_utc": _iso(btc_received_at),
            "t60_total_input_to_prediction_seconds": (prediction_finished - deadline).total_seconds(),
            "feature_construction_started_at_utc": _iso(feature_started),
            "feature_construction_finished_at_utc": _iso(feature_finished),
            "feature_construction_seconds": feature_seconds,
            "prediction_started_at_utc": _iso(inference_started),
            "prediction_finished_at_utc": _iso(prediction_finished),
            "prediction_and_calibration_seconds": inference_seconds,
            "t60_snapshot_request_seconds": t60["request_duration_seconds"],
        })
        if t60["p_market"] is None:
            raise RuntimeError("T-60 market probability is unavailable")
        probabilities = _probabilities(t60["p_market"], calibration)
        attempt["probabilities"] = {
            "p_market": float(t60["p_market"]),
            **probabilities,
            "p_A_raw": raw_probability,
        }
        decision_started = _utc()
        decision_clock = time.perf_counter()
        attempt["policy"] = _policy_at_t60(t60, calibration, metadata, market_start)
        decision_finished = _utc()
        decision_seconds = time.perf_counter() - decision_clock
        t59 = _utc(market_start) - pd.Timedelta(seconds=59)
        ready_at = max(_utc(t60["received_at_utc"]), _utc(btc_received_at), decision_finished)
        attempt["timings"]["t60_quality_inputs_ready_at_utc"] = _iso(ready_at)
        attempt["timings"].update({
            "quality_inputs_ready_before_t59": bool(ready_at <= t59),
            "decision_started_at_utc": _iso(decision_started),
            "decision_finished_at_utc": _iso(decision_finished),
            "decision_computation_seconds": decision_seconds,
            "decision_finished_by_t59": decision_finished <= t59,
        })
        attempt["attempt_status"] = (
            "complete" if ready_at <= t59 and t60["quote_valid"] else "late_or_incomplete"
        )
        if attempt["attempt_status"] != "complete" and attempt["reason"] is None:
            attempt["reason"] = "t60_quality_inputs_not_ready_by_t59"
        try:
            _wait_until(t59)
            execution_snapshot = _quote_snapshot(metadata, "T-59")
            attempt["t59_snapshot"] = execution_snapshot
            attempt["timings"]["t59_snapshot_request_seconds"] = execution_snapshot["request_duration_seconds"]
            attempt["timings"]["t59_book_started_at_utc"] = execution_snapshot["request_started_at_utc"]
        except Exception as t59_exc:
            attempt["timings"]["t59_error"] = f"{type(t59_exc).__name__}:{t59_exc}"
            attempt["reason"] = attempt["reason"] or f"t59_execution_observation_error:{type(t59_exc).__name__}:{t59_exc}"
            return attempt
        if attempt.get("policy") is not None:
            attempt["policy"]["t59_execution_observation"] = _policy_at_t59(
                attempt["policy"], execution_snapshot, metadata
            )
        return attempt
    except Exception as exc:
        attempt["reason"] = attempt["reason"] or f"cycle_error:{type(exc).__name__}:{exc}"
        try:
            t59 = _utc(market_start) - pd.Timedelta(seconds=59)
            _wait_until(t59)
            attempt["t59_snapshot"] = _quote_snapshot(metadata, "T-59")
        except Exception as t59_exc:
            attempt["timings"]["t59_error"] = f"{type(t59_exc).__name__}:{t59_exc}"
        return attempt
    finally:
        attempt["attempt_recorded_at_utc"] = _iso()


def _settle_outstanding(connection, session):
    rows = connection.execute(
        "SELECT market_slug,market_start_utc,t60_snapshot_json FROM market_attempts a LEFT JOIN official_outcomes o USING(market_slug) WHERE o.market_slug IS NULL ORDER BY market_start_utc"
    ).fetchall()
    now = _utc()
    for slug, start_text, t60_json in rows:
        if now < _utc(start_text) + pd.Timedelta(minutes=6):
            continue
        try:
            outcome = _official_outcome(session, slug)
            if outcome is not None:
                market = json.loads(t60_json) if t60_json else {}
                _append_outcome(connection, slug, market.get("condition_id"), outcome)
        except requests.RequestException as exc:
            _log_event(connection, "outcome_lookup_error", {"market_slug": slug, "error": f"{type(exc).__name__}:{exc}"})


def _market_cycle(connection, predictor, model, calibrator, session, market_start, feed):
    slug = f"{MARKET_PREFIX}-{int(market_start.timestamp())}"
    if connection.execute("SELECT 1 FROM market_attempts WHERE market_slug=?", (slug,)).fetchone():
        return
    connection.execute(
        "UPDATE session SET first_market_start_utc=COALESCE(first_market_start_utc,?),evaluation_deadline_utc=COALESCE(evaluation_deadline_utc,?) WHERE id=1",
        (_iso(market_start), _iso(_utc(market_start) + pd.Timedelta(days=240))),
    )
    connection.commit()
    session_row = connection.execute(
        "SELECT started_at_utc,first_market_start_utc,evaluation_deadline_utc FROM session WHERE id=1"
    ).fetchone()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["first_market_start_utc"] = session_row[1]
    manifest["evaluation_deadline_utc"] = session_row[2]
    manifest["settlement_grace_deadline_utc"] = _iso(
        _utc(session_row[2]) + pd.Timedelta(hours=int(PROTOCOL["quality_test"]["official_outcome_settlement_grace_hours"]))
    )
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    try:
        metadata = _market_metadata(session, market_start)
    except Exception as exc:
        deadline = _utc(market_start) - pd.Timedelta(minutes=1)
        _wait_until(deadline + pd.Timedelta(seconds=1))
        _append_attempt(connection, {
            "market_slug": slug, "market_start_utc": _iso(market_start),
            "decision_deadline_utc": _iso(deadline), "attempt_recorded_at_utc": _iso(),
            "attempt_status": "late_or_missing", "reason": f"market_metadata_error:{type(exc).__name__}:{exc}",
            "btc_input_opened_utc": _iso(deadline - pd.Timedelta(minutes=1)),
            "timings": {}, "source_identity": {"protocol_sha256": _sha256(PROTOCOL_PATH)},
        })
        _log_event(connection, "market_metadata_error", {"market_slug": slug, "error": f"{type(exc).__name__}:{exc}"})
        return
    deadline = _utc(market_start) - pd.Timedelta(minutes=1)
    input_opened = deadline - pd.Timedelta(minutes=1)
    session_start = _utc(connection.execute("SELECT started_at_utc FROM session WHERE id=1").fetchone()[0])
    t60_snapshot = None
    received_at = None
    source_metadata = None
    websocket_wait_seconds = 0.0
    try:
        # Public BTC streams stay connected before T-60; fetch only both books here.
        _wait_until(deadline)
        request_sessions = [requests.Session() for _ in range(2)]
        for request_session in request_sessions:
            request_session.headers.update({"User-Agent": "btc-prospective-shadow/1.0"})
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                up_book_future = pool.submit(shadow._book_observation, request_sessions[0], metadata["tokens"]["up"])
                down_book_future = pool.submit(shadow._book_observation, request_sessions[1], metadata["tokens"]["down"])
                books, book_errors = _collect_book_futures(up_book_future, down_book_future)
        finally:
            for request_session in request_sessions:
                request_session.close()
        t60_snapshot = _quote_snapshot(metadata, "T-60", books=books, book_errors=book_errors)
        websocket_wait_started = time.perf_counter()
        _advance_websocket_feed_until(
            connection, predictor, feed, input_opened,
            _websocket_pair_wait_budget(_utc(market_start) - pd.Timedelta(seconds=59)),
        )
        websocket_wait_seconds = time.perf_counter() - websocket_wait_started
        last_opened = _utc(predictor.opened_candles[-1])
        if last_opened > input_opened:
            raise RuntimeError(f"T-60 predictor contains future candle {last_opened} > {input_opened}")
        if last_opened < input_opened:
            catchup, request_started, received_at, elapsed = _timed_fetch_catchup_pair(
                last_opened + live_runtime.INTERVAL_DELTA, input_opened
            )
            _advance_predictor(predictor, catchup)
            _persist_candles(connection, catchup.itertuples(index=False), received_at, {
                "source": "public_rest_t60_fallback",
                "request_started_at_utc": _iso(request_started),
                "received_at_utc": _iso(received_at),
                "request_elapsed_seconds": elapsed,
            })
        if _utc(predictor.opened_candles[-1]) != input_opened:
            raise RuntimeError("T-60 BTC source did not return the exact closed input candle")
        stored_input = _persisted_candle(connection, input_opened)
        if stored_input is None:
            raise RuntimeError("T-60 input candle is absent from the durable BTC cache")
        received_at = _utc(stored_input["received_at_utc"])
        source_metadata = stored_input["source_metadata"]
        attempt = _decision_attempt(
            predictor, model, calibrator, metadata, market_start,
            t60_snapshot, received_at,
        )
        attempt["timings"]["btc_source_metadata"] = source_metadata
        attempt["timings"]["btc_websocket_pair_wait_seconds"] = websocket_wait_seconds
        attempt["timings"]["btc_websocket_connections"] = feed.status_snapshot()
        attempt["source_identity"] = {
            "protocol_sha256": _sha256(PROTOCOL_PATH),
            "model_bundle_sha256": _sha256(BUNDLE_PATH),
            "model_sha256": _sha256(ROOT / json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))["model_path"]),
            "calibrator_sha256": _sha256(ROOT / json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))["calibrator_path"]),
        }
        _append_attempt(connection, attempt)
    except Exception as exc:
        try:
            t59_at = _utc(market_start) - pd.Timedelta(seconds=59)
            _wait_until(t59_at)
            t59_snapshot = _quote_snapshot(metadata, "T-59")
        except Exception:
            t59_snapshot = None
        _append_attempt(connection, {
            "market_slug": slug, "market_start_utc": _iso(market_start),
            "decision_deadline_utc": _iso(deadline), "attempt_recorded_at_utc": _iso(),
            "attempt_status": "late_or_missing", "reason": f"{type(exc).__name__}:{exc}",
            "btc_input_opened_utc": _iso(input_opened), "t59_snapshot": t59_snapshot,
            "btc_data_received_at_utc": None if received_at is None else _iso(received_at),
            "t60_snapshot": t60_snapshot,
            "timings": {
                "collector_session_started_at_utc": _iso(session_start),
                "btc_source_metadata": source_metadata,
                "btc_websocket_pair_wait_seconds": websocket_wait_seconds,
                "btc_websocket_connections": feed.status_snapshot(),
            },
            "source_identity": {"protocol_sha256": _sha256(PROTOCOL_PATH)},
        })
        _log_event(connection, "market_cycle_error", {"market_slug": slug, "error": f"{type(exc).__name__}:{exc}"})


def _initialize_runtime(connection, artifacts):
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    temp_req = OUTPUT_DIR / "runtime" / "indicator_history_requirements.json"
    metadata = json.loads((ROOT / bundle["model_meta_path"]).read_text(encoding="utf-8"))
    requirement_path, required = _runtime_requirements(
        metadata, ROOT / bundle["indicator_fit_results_dir"], temp_req
    )
    runtime_window = max(int(required["global_required_runtime_window"]), int(audit.DEFAULT_MAX_KEEP))
    artifacts["generated_indicator_history_requirements"] = _sha256(requirement_path)
    frozen_hash = _canonical_hash({"protocol_sha256": artifacts["protocol"], "artifacts": artifacts})
    checkpoint = connection.execute("SELECT 1 FROM checkpoints WHERE id=1").fetchone()
    if checkpoint:
        predictor = _restore_checkpoint(
            connection, frozen_hash, bundle, requirement_path, runtime_window
        )
        runtime_meta_path = OUTPUT_DIR / "runtime" / "lgbm_meta.json"
    else:
        predictor, runtime_meta_path = _build_predictor(connection, bundle, requirement_path, runtime_window)
    model = predictor.model
    calibration_path = ROOT / bundle["calibrator_path"]
    calibrator = json.loads(calibration_path.read_text(encoding="utf-8"))
    if _sha256(calibration_path) != bundle["calibrator_sha256"]:
        raise RuntimeError("Frozen A_frozen calibrator hash differs from the evaluated bundle")
    # Seed the BTC runtime forward from the frozen profile anchor through the latest closed minute.
    _backfill_predictor_to_latest_closed(connection, predictor, "public_rest_startup_backfill")
    _save_checkpoint(connection, predictor, frozen_hash)
    session_started = _start_session(connection, artifacts)
    return session_started, bundle, predictor, model, calibrator, frozen_hash, runtime_meta_path


def run_shadow():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = _artifact_manifest()
    connection = _connect()
    session_started, bundle, predictor, model, calibrator, frozen_hash, _ = _initialize_runtime(connection, artifacts)
    feed = PublicBtcCandleFeed()
    feed.start()
    feed_connections = _wait_for_public_feed_connections(feed)
    startup_backfill = _backfill_predictor_to_latest_closed(
        connection, predictor, "public_rest_websocket_startup_backfill"
    )
    _save_checkpoint(connection, predictor, frozen_hash)
    session = requests.Session()
    session.headers.update({"User-Agent": "btc-prospective-shadow/1.0"})
    _attempt_missing_schedule(connection, session_started, _utc())
    _log_event(connection, "collector_started", {
        "last_btc_candle_opened_utc": _iso(predictor.opened_candles[-1]),
        "model_bundle_sha256": _sha256(BUNDLE_PATH), "orders_enabled": False,
        "btc_websocket_targets": live_runtime.WS_TARGETS,
        "btc_websocket_connections": feed_connections,
        "post_connect_startup_backfill": startup_backfill,
    })
    next_settlement_poll = 0.0
    try:
        market_start = _next_market_start(_utc())
        while True:
            window_deadline = connection.execute(
                "SELECT evaluation_deadline_utc FROM session WHERE id=1"
            ).fetchone()[0]
            if window_deadline and market_start >= _utc(window_deadline):
                grace_deadline = _utc(window_deadline) + pd.Timedelta(
                    hours=int(PROTOCOL["quality_test"]["official_outcome_settlement_grace_hours"])
                )
                if _utc() < grace_deadline:
                    if time.monotonic() >= next_settlement_poll:
                        _settle_outstanding(connection, session)
                        next_settlement_poll = time.monotonic() + SETTLEMENT_POLL_SECONDS
                    time.sleep(min(SETTLEMENT_POLL_SECONDS, max((grace_deadline - _utc()).total_seconds(), 0.0)))
                    continue
                _settle_outstanding(connection, session)
                _log_event(connection, "evaluation_window_closed", {
                    "market_deadline_utc": window_deadline,
                    "settlement_grace_deadline_utc": _iso(grace_deadline),
                })
                break
            try:
                _market_cycle(connection, predictor, model, calibrator, session, market_start, feed)
                _save_checkpoint(connection, predictor, frozen_hash)
                market_start += pd.Timedelta(minutes=5)
                if time.monotonic() >= next_settlement_poll:
                    _settle_outstanding(connection, session)
                    next_settlement_poll = time.monotonic() + SETTLEMENT_POLL_SECONDS
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                _log_event(connection, "collector_error", {"error": f"{type(exc).__name__}:{exc}"})
                time.sleep(POLL_SECONDS)
    finally:
        feed.stop()
        session.close()
        _save_checkpoint(connection, predictor, frozen_hash)
        connection.close()


if __name__ == "__main__":
    run_shadow()
