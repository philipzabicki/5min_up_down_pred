"""Read-only prospective shadow collector for frozen BTC candidates A/B/C/D."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import audit_feature_readiness as audit
import run as live_runtime
import run_btc_candidate_holdout_20261022 as candidates
import run_btc_candidate_quality_20261022 as candidate_quality
import run_btc_prospective_shadow as prospective
import run_btc_shadow as shadow
from features.btc_taker_flow import compute_taker_flow_features
from features.causal_memory_control import CausalMemoryState
from features.esn_reservoir import ESNEngine, FeatureScaler, INPUT_COLUMNS, build_causal_esn_inputs
from utils import polymarket_btc5m_fee_rules as fee_rules


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/research/btc_candidate_shadow_20261023.json"
CONFIG = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / "data/analysis/polymarket/BTC/btc_candidate_shadow_20261023"
DB_PATH = OUTPUT_DIR / "shadow.sqlite3"
RUNTIME_DIR = OUTPUT_DIR / "runtime"
FLOW_FEATURES = ("taker_Z_1m", "taker_V_1m", "taker_Z_5m", "taker_V_5m", "taker_Z_15m", "taker_V_15m")
FLOW_CONFIG_PATH = ROOT / "configs/research/btc_preopen_taker_flow_20261018.json"
FLOW_CONFIG = json.loads(FLOW_CONFIG_PATH.read_text(encoding="utf-8"))["flow_features"]
MARKET_PREFIX = CONFIG["collection"]["market_prefix"]
ANCHOR_OPENED = prospective.ANCHOR_OPENED
POLL_SECONDS = 0.10
SETTLEMENT_POLL_SECONDS = float(CONFIG["collection"]["settlement_poll_seconds"])
CHECKPOINT_SECONDS = float(CONFIG["runtime"]["checkpoint_frequency_seconds"])
MAX_DATABASE_BYTES = int(CONFIG["runtime"]["maximum_database_bytes"])
GROSS_ORDER_USD = float(CONFIG["collection"]["hypothetical_gross_notional_usd"])


def _utc(value=None) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _iso(value=None) -> str:
    return _utc(value).isoformat()


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=shadow._json_default, allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def market_timing(market_start) -> dict:
    start = _utc(market_start)
    return {
        "market_start_utc": start,
        "decision_deadline_utc": start - pd.Timedelta(seconds=60),
        "execution_snapshot_utc": start - pd.Timedelta(seconds=59),
        "input_opened_utc": start - pd.Timedelta(minutes=2),
    }


def prediction_timeliness(ready_at, decision_deadline) -> dict:
    ready = _utc(ready_at)
    deadline = _utc(decision_deadline)
    return {
        "ready_at_utc": ready,
        "decision_deadline_utc": deadline,
        "timely": bool(ready <= deadline),
        "lateness_seconds": max(0.0, (ready - deadline).total_seconds()),
    }


def outcome_feature_matrix(base64, flow_by_name, base_order):
    base = np.asarray(base64, dtype=np.float32).reshape(-1)
    if len(base_order) != 70 or len(base) != 64:
        raise ValueError("shadow BASE70/BASE64 order differs from the frozen contract")
    values = dict(zip((name for name in base_order if name not in FLOW_FEATURES), base))
    values.update({name: float(flow_by_name[name]) for name in FLOW_FEATURES})
    matrix = np.asarray([values[name] for name in base_order], dtype=np.float32).reshape(1, -1)
    if not np.isfinite(matrix).all():
        raise ValueError("candidate BASE70 row contains unavailable or nonfinite features")
    return matrix


def hypothetical_buy(ask_levels, fee_model, entry_time_utc, gross_usd=GROSS_ORDER_USD) -> dict:
    """Price a $5 gross ask walk with the date-assigned collection unit and Gamma fee curve."""
    clean = {}
    for price, size in ask_levels:
        price, size = float(price), float(size)
        if 0.0 < price < 1.0 and size > 0.0 and math.isfinite(price) and math.isfinite(size):
            clean[price] = clean.get(price, 0.0) + size
    if not clean:
        return {"priceable": False, "reason": "empty_or_invalid_ask_ladder", "actual_fill": "unknown"}
    rule = fee_rules.resolve_market_fee_rule(None, entry_time_utc)
    if not isinstance(fee_model, dict) or not np.isfinite(float(fee_model.get("rate", np.nan))):
        return {"priceable": False, "reason": "unknown_market_fee_schedule", "actual_fill": "unknown"}
    fee_rule = dict(rule)
    fee_rule["rate"] = float(fee_model["rate"])
    fee_rule["exponent"] = float(fee_model.get("exponent", 1.0))
    if rule["collection_mode"] == "maintenance_pause":
        return {
            "priceable": False, "reason": "exchange_maintenance_pause",
            "fee_rule_id": rule["rule_id"], "fee_rule_status": rule["status"],
            "actual_fill": "unknown",
        }
    if rule["collection_mode"] == "cash_collateral":
        fee_rule["v2_minimum_fee"] = float(fee_model.get("min_fee", rule.get("v2_minimum_fee", 0.00001)))
    # Apply the observed Gamma fee curve with the frozen date-assigned collection unit.
    remaining = float(gross_usd)
    raw_fee = fee_cash_v1 = fee_shares_v1 = gross_shares = gross_cash = 0.0
    levels_used = []
    from utils.polymarket_btc5m_fee_rules import fee_for_price_level, round_v2_cash_fee
    for price in sorted(clean):
        take = min(remaining / price, clean[price])
        if take <= 0.0:
            continue
        level = fee_for_price_level(take, price, fee_rule, side="buy")
        gross_shares += take
        gross_cash += take * price
        raw_fee += level["raw_fee_usdc_equivalent"]
        fee_cash_v1 += level["fee_cash_usd"]
        fee_shares_v1 += level["fee_shares"]
        remaining -= take * price
        levels_used.append({"price": price, "shares": take, "gross_usd": take * price})
        if remaining <= 1e-8:
            break
    depth_sufficient = remaining <= 1e-6
    if rule["collection_mode"] == "cash_collateral":
        fee_cash = round_v2_cash_fee(raw_fee, fee_rule)
        fee_shares = 0.0
        net_shares = gross_shares
    else:
        fee_cash = fee_cash_v1
        fee_shares = fee_shares_v1
        net_shares = gross_shares - fee_shares
    return {
        "priceable": bool(depth_sufficient),
        "reason": "full_depth_priceable" if depth_sufficient else "insufficient_displayed_ask_depth",
        "requested_gross_usd": float(gross_usd),
        "gross_usd": gross_cash,
        "unfilled_gross_usd": max(remaining, 0.0),
        "gross_shares": gross_shares,
        "net_shares_after_fee": net_shares,
        "vwap": gross_cash / gross_shares if gross_shares else None,
        "fee_cash_usd": fee_cash,
        "fee_shares": fee_shares,
        "fee_usd_entry_equivalent": fee_shares * (gross_cash / gross_shares) if fee_shares and gross_shares else fee_cash,
        "cash_debit_usd": gross_cash + fee_cash,
        "fee_rate": float(fee_model["rate"]),
        "fee_exponent": float(fee_rule["exponent"]),
        "fee_collection_mode": rule["collection_mode"],
        "fee_rule_id": rule["rule_id"],
        "fee_rule_status": rule["status"],
        "fee_source": fee_model.get("source"),
        "levels_used": levels_used,
        "execution_assumption": "hypothetical_full_displayed_book_walk; fill not guaranteed",
        "actual_fill": "unknown",
    }


def _connect(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS collector_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, observed_at_utc TEXT NOT NULL,
          event_type TEXT NOT NULL, payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS candles (
          opened_utc TEXT PRIMARY KEY, candle_json TEXT NOT NULL,
          received_at_utc TEXT NOT NULL, source_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
          market_start_utc TEXT PRIMARY KEY, input_opened_utc TEXT NOT NULL,
          input_received_at_utc TEXT NOT NULL, feature_started_at_utc TEXT NOT NULL,
          inference_started_at_utc TEXT NOT NULL, ready_at_utc TEXT NOT NULL,
          timely INTEGER NOT NULL, members_json TEXT NOT NULL,
          ensembles_json TEXT NOT NULL, timings_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS markets (
          market_start_utc TEXT PRIMARY KEY, market_slug TEXT NOT NULL,
          condition_id TEXT NOT NULL, tokens_json TEXT NOT NULL,
          metadata_json TEXT NOT NULL, fetched_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS snapshots (
          market_start_utc TEXT NOT NULL, snapshot_name TEXT NOT NULL,
          requested_at_utc TEXT NOT NULL, received_at_utc TEXT NOT NULL,
          snapshot_json TEXT NOT NULL, PRIMARY KEY(market_start_utc,snapshot_name)
        );
        CREATE TABLE IF NOT EXISTS market_attempts (
          market_start_utc TEXT PRIMARY KEY, status TEXT NOT NULL,
          decision_deadline_utc TEXT NOT NULL, execution_snapshot_utc TEXT NOT NULL,
          prediction_json TEXT, t60_snapshot_json TEXT, t59_snapshot_json TEXT,
          execution_json TEXT NOT NULL, timings_json TEXT NOT NULL,
          recorded_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS official_outcomes (
          market_start_utc TEXT PRIMARY KEY, target_up INTEGER NOT NULL,
          resolved_at_utc TEXT, received_at_utc TEXT NOT NULL, payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
          id INTEGER PRIMARY KEY CHECK(id=1), identity_sha256 TEXT NOT NULL,
          last_opened_utc TEXT NOT NULL, state_blob BLOB NOT NULL,
          state_sha256 TEXT NOT NULL, saved_at_utc TEXT NOT NULL
        );
        """
    )
    connection.commit()
    return connection


def _event(connection, event_type, payload):
    connection.execute(
        "INSERT INTO collector_events(observed_at_utc,event_type,payload_json) VALUES (?,?,?)",
        (_iso(), event_type, _json(payload)),
    )
    connection.commit()


def _network_probe(connection) -> tuple[bool, dict]:
    market_start = _next_market_start()
    targets = {
        "binance": "https://fapi.binance.com/fapi/v1/time",
        "gamma": f"https://gamma-api.polymarket.com/markets/slug/{MARKET_PREFIX}-{int(market_start.timestamp())}",
        "clob": "https://clob.polymarket.com/book?token_id=0",
    }
    result = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "btc-candidate-shadow/1.0"})
    for name, url in targets.items():
        try:
            response = session.get(url, timeout=8)
            result[name] = {"url": url, "status_code": int(response.status_code), "transport_ok": response.status_code < 500}
        except requests.RequestException as exc:
            result[name] = {"url": url, "transport_ok": False, "error": f"{type(exc).__name__}: {exc}"}
    session.close()
    _event(connection, "network_preflight", result)
    return all(item["transport_ok"] for item in result.values()), result


def _runtime_identity(bundle):
    paths = {
        "protocol": PROTOCOL_PATH,
        "runner": Path(__file__).resolve(),
        "prospective_runner": ROOT / "run_btc_prospective_shadow.py",
        "candidate_runner": ROOT / "run_btc_candidate_holdout_20261022.py",
        "candidate_quality": ROOT / "run_btc_candidate_quality_20261022.py",
        "audit_runtime": ROOT / "audit_feature_readiness.py",
        "live_runtime": ROOT / "run.py",
        "taker_flow_code": ROOT / "features/btc_taker_flow.py",
        "memory_code": ROOT / "features/causal_memory_control.py",
        "esn_code": ROOT / "features/esn_reservoir.py",
        "taker_flow_config": FLOW_CONFIG_PATH,
        "fee_rules": ROOT / "utils/polymarket_btc5m_fee_rules.py",
        "market_helpers": ROOT / "utils/polymarket.py",
        "shadow_market_reader": ROOT / "run_btc_shadow.py",
        "economic_walk": ROOT / "run_btc_preopen_economic_replay.py",
    }
    old_bundle = json.loads(prospective.BUNDLE_PATH.read_text(encoding="utf-8"))
    old_assets = {
        "protocol_sha256": prospective.PROTOCOL["model_bundle_sha256"],
        "bundle_sha256": _sha256(prospective.BUNDLE_PATH),
        "model_meta_sha256": _sha256(ROOT / old_bundle["model_meta_path"]),
        "model_sha256": _sha256(ROOT / old_bundle["model_path"]),
        "profile_state_sha256": {
            str(path): _sha256(ROOT / path) for path in old_bundle["profile_state_artifacts"]
        },
        "indicator_fit_results_sha256": {
            path.name: _sha256(path) for path in sorted((ROOT / old_bundle["indicator_fit_results_dir"]).glob("*.json"))
        },
        "anchor_opened_utc": str(ANCHOR_OPENED),
    }
    return {
        "candidate_bundle": bundle,
        "base64_feature_runtime": old_assets,
        "runtime_files": {name: _sha256(path) for name, path in paths.items()},
        "protocol_sha256": _sha256(PROTOCOL_PATH),
    }


def _runtime_requirements(feature_columns, fit_dir, destination):
    specs = audit.load_indicator_specs(
        feature_columns, source_label="frozen candidate BASE70 minus six taker-flow controls",
        fit_results_dir=fit_dir,
    )
    payload = {
        "validation_status": "parameter_based_warmup_only_empirical_stability_not_audited",
        "unstable_feature_count": 0,
        "global_required_stable_window": max(int(spec.required_candles) for spec in specs),
        "required_stable_window_by_feature": {
            str(spec.feature_col): max(int(spec.required_candles), 1) for spec in specs
        },
        "generated_from_frozen_candidate_bundle": True,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return audit.load_indicator_history_requirements(
        destination, indicator_specs=specs, allow_unstable=False,
    )


def _build_base_predictor(base_order):
    old_bundle = json.loads(prospective.BUNDLE_PATH.read_text(encoding="utf-8"))
    old_meta_path = ROOT / old_bundle["model_meta_path"]
    meta = json.loads(old_meta_path.read_text(encoding="utf-8"))
    base64_order = [name for name in base_order if name not in FLOW_FEATURES]
    if len(base64_order) != 64:
        raise RuntimeError("frozen BASE70 must contain exactly six external taker-flow features")
    meta["feature_columns"] = base64_order
    meta.update(prospective._feature_runtime_config(old_bundle))
    meta["artifacts"] = {"final_model_path": str((ROOT / old_bundle["model_path"]).resolve())}
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    runtime_meta_path = RUNTIME_DIR / "base64_runtime_meta.json"
    runtime_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    fit_dir = ROOT / old_bundle["indicator_fit_results_dir"]
    requirements_path = RUNTIME_DIR / "indicator_history_requirements.json"
    requirements = _runtime_requirements(base64_order, fit_dir, requirements_path)
    runtime_window = int(requirements["global_required_runtime_window"])
    bootstrap = prospective._load_anchor_frame(ROOT / old_bundle["raw_candle_path"], runtime_window)
    volume_state = shadow.load_volume_profile_state(ROOT / old_bundle["profile_state_artifacts"][1])
    reaction_state = shadow.load_reaction_profile_state(ROOT / old_bundle["profile_state_artifacts"][0])
    predictor = audit.PseudoLiveAuditPredictor(
        bootstrap,
        model_meta_path=runtime_meta_path,
        max_keep=runtime_window,
        volume_profile_state=volume_state,
        reaction_profile_state=reaction_state,
        feature_runtime_config=prospective._feature_runtime_config(old_bundle),
        fit_results_dir=fit_dir,
        indicator_history_requirements_path=requirements_path,
        load_model=False,
        allow_unstable_indicator_summary=False,
    )
    return predictor, base64_order, old_bundle


def _load_dynamic_config(inputs, bundle):
    memory_bundle = json.loads((candidates.BUNDLE_DIR / "transforms/simple_memory_projection.json").read_text(encoding="utf-8"))
    esn_bundles = {
        seed: json.loads((candidates.BUNDLE_DIR / f"transforms/esn_seed_{seed}.json").read_text(encoding="utf-8"))
        for seed in bundle["reservoir_seeds"]
    }
    memory_scaler = FeatureScaler.from_dict(memory_bundle["input_scaler"])
    memory_columns = tuple(memory_bundle["identity"]["input_columns"])
    memory_indices = np.asarray([INPUT_COLUMNS.index(name) for name in memory_columns], dtype=np.int64)
    memory_state = None
    memory_projection = np.asarray(memory_bundle["pca_components"], dtype=np.float64)
    memory_mean = np.asarray(memory_bundle["pca_mean"], dtype=np.float64)
    esn_engines = {}
    esn_indices = np.asarray([INPUT_COLUMNS.index(name) for name in inputs["esn_config"].selected_columns], dtype=np.int64)
    esn_projection = {}
    for seed, item in esn_bundles.items():
        esn_engines[seed] = ESNEngine(
            inputs["esn_config"], seed, FeatureScaler.from_dict(item["scaler"]),
            warmup_steps=int(item["warmup_steps"]),
        )
        esn_projection[seed] = {
            "mean": np.asarray(item["projection"]["mean"], dtype=np.float64),
            "components": np.asarray(item["projection"]["components"], dtype=np.float64),
        }
    return {
        "memory_bundle": memory_bundle,
        "esn_bundles": esn_bundles,
        "memory_scaler": memory_scaler,
        "memory_indices": memory_indices,
        "memory_mean": memory_mean,
        "memory_projection": memory_projection,
        "esn_engines": esn_engines,
        "esn_indices": esn_indices,
        "esn_projection": esn_projection,
        "memory_state": memory_state,
        "last_memory": None,
        "last_esn": {},
        "raw_tail": None,
    }


def _restore_dynamic_state(runtime, payload):
    runtime["memory_state"] = CausalMemoryState.from_dict(payload["memory_state"])
    for seed, saved in payload["esn_state"].items():
        engine = runtime["esn_engines"][int(seed)]
        engine.states = np.asarray(saved["states"], dtype=np.float32)
        engine.steps_since_reset = int(saved["steps_since_reset"])
        engine.last_timestamp = pd.Timestamp(saved["last_timestamp"]) if saved["last_timestamp"] else None
    runtime["raw_tail"] = pd.DataFrame(payload["raw_tail"])


def _dynamic_checkpoint(runtime):
    return {
        "memory_state": runtime["memory_state"].to_dict(),
        "esn_state": {
            str(seed): {
                "states": engine.states,
                "steps_since_reset": engine.steps_since_reset,
                "last_timestamp": None if engine.last_timestamp is None else engine.last_timestamp.isoformat(),
            }
            for seed, engine in runtime["esn_engines"].items()
        },
        "raw_tail": runtime["raw_tail"].to_dict(orient="records"),
    }


def _initialize_dynamic_state(runtime, inputs):
    _, _, state = candidates._generate_dynamic_features(
        inputs["raw_opened"], inputs["raw_x"], inputs["raw_valid"],
        runtime["memory_bundle"], inputs["esn_config"], runtime["esn_bundles"],
        return_states=True,
    )
    runtime["memory_state"] = state["memory_state"]
    runtime["esn_engines"] = state["esn_states"]
    source_path = ROOT / json.loads(candidates.CONFIG_PATH.read_text(encoding="utf-8"))["inputs"]["baseline_feature_dataset"]
    last_opened = pd.Timestamp(int(inputs["raw_opened"][-1]), unit="ns", tz="UTC")
    frame = pd.read_parquet(
        source_path,
        columns=["Opened", "Open", "High", "Low", "Close", "Volume", "futures_index_basis_rel_1m"],
        filters=[("Opened", ">=", (last_opened - pd.Timedelta(minutes=120)).tz_localize(None)),
                 ("Opened", "<=", last_opened.tz_localize(None))],
    )
    frame["Opened"] = pd.to_datetime(frame["Opened"], utc=True, errors="raise")
    runtime["raw_tail"] = frame.sort_values("Opened").tail(120).reset_index(drop=True)
    if runtime["raw_tail"].empty or runtime["raw_tail"].Opened.iloc[-1] != last_opened:
        raise RuntimeError("BASE70 raw tail does not reach the frozen development stream end")


def _raw_feature_row(runtime, candle):
    timestamp = _utc(candle["Opened"])
    close = float(candle["Close"])
    futures_close = float(candle["UM_BTCUSDT_Close"])
    rel_basis = futures_close / close - 1.0 if close > 0 and futures_close > 0 else np.nan
    row = {
        "Opened": timestamp,
        "Open": float(candle["Open"]), "High": float(candle["High"]),
        "Low": float(candle["Low"]), "Close": close, "Volume": float(candle["Volume"]),
        "futures_index_basis_rel_1m": rel_basis,
    }
    raw_tail = pd.concat([runtime["raw_tail"], pd.DataFrame([row])], ignore_index=True).tail(120)
    opened, features = build_causal_esn_inputs(raw_tail)
    raw_x = features.loc[:, INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=False)[-1]
    valid = bool(
        np.isfinite([row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]]).all()
        and row["Open"] > 0 and row["Close"] > 0 and row["High"] >= max(row["Open"], row["Close"])
        and row["Low"] <= min(row["Open"], row["Close"]) and row["High"] >= row["Low"]
        and row["Volume"] >= 0
    )
    runtime["raw_tail"] = raw_tail
    return opened[-1], raw_x, valid


def _advance_dynamic(runtime, candle):
    timestamp, raw_x, valid = _raw_feature_row(runtime, candle)
    memory_input = runtime["memory_scaler"].transform(raw_x[runtime["memory_indices"]].astype(np.float64).reshape(1, -1))[0]
    memory_features = runtime["memory_state"].update_one(timestamp, memory_input, valid=valid)
    if memory_features is not None:
        runtime["last_memory"] = ((memory_features.astype(np.float64) - runtime["memory_mean"]) @ runtime["memory_projection"].T).astype(np.float32)
    esn_raw = raw_x[runtime["esn_indices"]]
    esn_now = {}
    for seed, engine in runtime["esn_engines"].items():
        state = engine.update_one(timestamp, esn_raw, valid=valid)
        if state is not None:
            projection = runtime["esn_projection"][seed]
            esn_now[seed] = ((state.astype(np.float64) - projection["mean"]) @ projection["components"].T).astype(np.float32)
    if esn_now:
        runtime["last_esn"] = esn_now
    return timestamp, raw_x, valid


def _flow_row(connection, opened):
    rows = connection.execute(
        "SELECT opened_utc,candle_json FROM candles WHERE opened_utc<=? ORDER BY opened_utc DESC LIMIT ?",
        (_iso(opened), int(FLOW_CONFIG["beta_window_minutes"]) + 100),
    ).fetchall()
    if not rows:
        return None
    frame_rows = []
    for stamp, payload in reversed(rows):
        candle = json.loads(payload)
        frame_rows.append({
            "Opened": _utc(stamp),
            "open": candle["Open"], "high": candle["High"], "low": candle["Low"],
            "close": candle["Close"], "quote_volume": candle.get("quote_volume"),
            "taker_buy_quote_volume": candle.get("taker_buy_quote_volume"),
        })
    frame = pd.DataFrame(frame_rows).set_index("Opened")
    generated = compute_taker_flow_features(
        frame,
        horizons=tuple(int(v) for v in FLOW_CONFIG["horizons_minutes"]),
        qbar_window=int(FLOW_CONFIG["qbar_window_minutes"]),
        sigma_window=int(FLOW_CONFIG["sigma_window_minutes"]),
        beta_window=int(FLOW_CONFIG["beta_window_minutes"]),
        minimum_beta_pairs=int(FLOW_CONFIG["minimum_beta_pairs"]),
        beta_variance_tolerance=float(FLOW_CONFIG["beta_variance_tolerance"]),
        sigma_floor=float(FLOW_CONFIG["sigma_floor"]),
        sigma_ddof=int(FLOW_CONFIG["sigma_ddof"]),
    )
    stamp = _utc(opened)
    if stamp not in generated.index:
        return None
    return {name: float(generated.loc[stamp, name]) for name in FLOW_FEATURES}


def _predict_market(
    connection, predictor, runtime, boosters, base_order, market_start,
    input_opened, input_received, source_metadata,
):
    timing = market_timing(market_start)
    if _utc(input_opened) != timing["input_opened_utc"]:
        raise ValueError("market signal input candle is not the frozen T-120 open")
    if pd.Timestamp(predictor.opened_candles[-1]) != timing["input_opened_utc"]:
        raise ValueError("BASE64 runtime is not synchronized to the exact decision candle")
    feature_started = _utc()
    feature_clock = time.perf_counter()
    snapshot_clock = time.perf_counter()
    base_snapshot = predictor.build_feature_snapshot()
    base64 = np.asarray(base_snapshot["vector"], dtype=np.float32).reshape(-1)
    snapshot_seconds = time.perf_counter() - snapshot_clock
    flow_clock = time.perf_counter()
    flow = _flow_row(connection, input_opened)
    if flow is None:
        raise ValueError("frozen taker-flow features are not ready for the decision candle")
    flow_seconds = time.perf_counter() - flow_clock
    matrix_clock = time.perf_counter()
    base70 = outcome_feature_matrix(base64, flow, base_order)
    matrix_seconds = time.perf_counter() - matrix_clock
    feature_elapsed = time.perf_counter() - feature_clock
    if runtime["last_memory"] is None or set(runtime["last_esn"]) != set(runtime["esn_engines"]):
        raise ValueError("memory or ESN state is not warm and complete")
    inference_started = _utc()
    inference_clock = time.perf_counter()
    member_rows = candidates.predict_members(
        base70, np.asarray(runtime["last_memory"]).reshape(1, -1),
        {seed: np.asarray(value).reshape(1, -1) for seed, value in runtime["last_esn"].items()},
        boosters,
        lightgbm_seeds=runtime["lightgbm_seeds"],
        reservoir_seeds=tuple(sorted(runtime["esn_engines"])),
    )
    ensembles = candidates.ensemble_predictions(
        member_rows,
        runtime["lightgbm_seeds"],
        tuple(sorted(runtime["esn_engines"])),
    )
    inference_elapsed = time.perf_counter() - inference_clock
    ready_at = _utc()
    deadline = timing["decision_deadline_utc"]
    timely = prediction_timeliness(ready_at, deadline)
    members = {name: {key: float(value[0]) for key, value in rows.items()} for name, rows in member_rows.items()}
    ensemble_values = {name: float(value[0]) for name, value in ensembles.items()}
    source_receipts = [
        _utc(source_metadata[key])
        for key in ("price_received_at_utc", "volume_received_at_utc")
        if source_metadata.get(key)
    ]
    source_events = [
        _utc(source_metadata[key])
        for key in ("price_event_at_utc", "volume_event_at_utc")
        if source_metadata.get(key)
    ]
    timings = {
        **dict(runtime.get("last_update_timings") or {}),
        "price_event_at_utc": source_metadata.get("price_event_at_utc"),
        "volume_event_at_utc": source_metadata.get("volume_event_at_utc"),
        "price_received_at_utc": source_metadata.get("price_received_at_utc"),
        "volume_received_at_utc": source_metadata.get("volume_received_at_utc"),
        "source_pair_receive_skew_seconds": (
            (max(source_receipts) - min(source_receipts)).total_seconds()
            if len(source_receipts) > 1 else 0.0
        ),
        "latest_source_event_to_receive_seconds": (
            (max(source_receipts) - max(source_events)).total_seconds()
            if source_receipts and source_events else None
        ),
        "input_received_at_utc": _iso(input_received),
        "feature_started_at_utc": _iso(feature_started),
        "feature_finished_at_utc": _iso(inference_started),
        "feature_seconds": feature_elapsed,
        "base70_feature_snapshot_seconds": snapshot_seconds,
        "taker_flow_feature_seconds": flow_seconds,
        "base70_matrix_seconds": matrix_seconds,
        "inference_started_at_utc": _iso(inference_started),
        "inference_finished_at_utc": _iso(ready_at),
        "inference_seconds": inference_elapsed,
        "input_to_ready_seconds": max(0.0, (ready_at - _utc(input_received)).total_seconds()),
        "decision_lateness_seconds": timely["lateness_seconds"],
        "feature_nonfinite_columns": list(base_snapshot["nonfinite_feature_indices"]),
    }
    connection.execute(
        "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            _iso(market_start), _iso(input_opened), _iso(input_received),
            _iso(feature_started), _iso(inference_started), _iso(ready_at), int(timely["timely"]),
            _json(members), _json(ensemble_values), _json(timings),
        ),
    )
    connection.commit()
    return {
        "market_start_utc": _iso(market_start),
        "input_opened_utc": _iso(input_opened),
        "input_received_at_utc": _iso(input_received),
        "ready_at_utc": _iso(ready_at),
        "timely": bool(timely["timely"]),
        "members": members,
        "ensembles": ensemble_values,
        "timings": timings,
    }


def _persist_candle(connection, update):
    row = dict(update["row"])
    metadata = dict(update.get("source_metadata") or {})
    metadata["received_at_utc"] = update["received_at_utc"]
    candle = {name: row.get(name) for name in live_runtime.OHLCV_COLS}
    candle.update({
        "UM_BTCUSDT_Close": row.get("UM_BTCUSDT_Close"),
        "quote_volume": metadata.get("quote_volume"),
        "taker_buy_quote_volume": metadata.get("taker_buy_quote_volume"),
    })
    opened = _iso(row["Opened"])
    encoded = _json(candle)
    previous = connection.execute("SELECT candle_json FROM candles WHERE opened_utc=?", (opened,)).fetchone()
    if previous:
        if previous[0] != encoded:
            _event(connection, "conflicting_duplicate_candle", {"opened_utc": opened, "old": json.loads(previous[0]), "new": candle})
            raise RuntimeError(f"conflicting duplicate candle at {opened}")
        _event(connection, "duplicate_candle_ignored", {
            "opened_utc": opened, "source": metadata.get("source"),
            "received_at_utc": update["received_at_utc"], "row": row,
            "source_metadata": metadata,
        })
        return False
    connection.execute(
        "INSERT INTO candles(opened_utc,candle_json,received_at_utc,source_json) VALUES (?,?,?,?)",
        (opened, encoded, update["received_at_utc"], _json(metadata)),
    )
    connection.commit()
    database_bytes = sum(path.stat().st_size for path in (DB_PATH, Path(str(DB_PATH) + "-wal")) if path.exists())
    if database_bytes >= MAX_DATABASE_BYTES:
        _event(connection, "database_size_limit_reached", {"database_bytes": database_bytes, "limit_bytes": MAX_DATABASE_BYTES})
        raise RuntimeError("shadow database reached its configured cap; no data was deleted")
    return True


def _advance_predictor(predictor, candle):
    opened = _utc(candle["Opened"])
    if opened <= _utc(predictor.opened_candles[-1]):
        return
    if opened != _utc(predictor.opened_candles[-1]) + live_runtime.INTERVAL_DELTA:
        raise RuntimeError("BASE64 runtime cannot advance across a missing candle")
    predictor._append_new_candle(
        opened, tuple(float(candle[name]) for name in live_runtime.OHLCV_COLS),
        basis_futures_close=float(candle["UM_BTCUSDT_Close"]),
    )
    predictor._update_volume_profile_state_for_latest_candle(opened, persist=False)
    predictor._update_reaction_profile_state_for_latest_candle(opened, persist=False)


def _process_update(connection, predictor, runtime, update, *, allow_signal):
    row = dict(update["row"])
    opened = _utc(row["Opened"])
    latest = _utc(runtime["esn_engines"][next(iter(runtime["esn_engines"]))].last_timestamp)
    expected = latest + live_runtime.INTERVAL_DELTA
    if opened <= latest:
        _persist_candle(connection, update)
        return
    if opened > expected:
        gap_end = opened - live_runtime.INTERVAL_DELTA
        gap, _, received, elapsed = _fetch_catchup_pair(expected, gap_end)
        _event(connection, "source_gap_repaired", {
            "from_opened_utc": _iso(expected), "through_opened_utc": _iso(gap_end),
            "request_received_at_utc": _iso(received), "elapsed_seconds": elapsed,
            "predictions_backfilled": False,
        })
        replay_started = time.perf_counter()
        for candle in gap.to_dict(orient="records"):
            repaired = _rest_update(candle, received, elapsed)
            _apply_update(connection, predictor, runtime, repaired, allow_signal=False)
        _event(connection, "source_gap_state_replay_complete", {
            "from_opened_utc": _iso(expected), "through_opened_utc": _iso(gap_end),
            "rows": len(gap), "state_replay_seconds": time.perf_counter() - replay_started,
            "predictions_backfilled": False,
        })
    if opened != _utc(runtime["esn_engines"][next(iter(runtime["esn_engines"]))].last_timestamp) + live_runtime.INTERVAL_DELTA:
        raise RuntimeError("source update is duplicate or out of order after gap repair")
    _apply_update(connection, predictor, runtime, update, allow_signal=allow_signal)


def _apply_update(connection, predictor, runtime, update, *, allow_signal, persist=True):
    row = dict(update["row"])
    opened = _utc(row["Opened"])
    processing_started_at = _utc()
    processing_clock = time.perf_counter()
    persist_clock = time.perf_counter()
    if persist and not _persist_candle(connection, update):
        return
    persist_seconds = time.perf_counter() - persist_clock
    dynamic_clock = time.perf_counter()
    _, _, valid = _advance_dynamic(runtime, row)
    dynamic_seconds = time.perf_counter() - dynamic_clock
    base_clock = time.perf_counter()
    if opened > _utc(predictor.opened_candles[-1]):
        _advance_predictor(predictor, row)
    base_seconds = time.perf_counter() - base_clock
    runtime["last_update_timings"] = {
        "input_received_at_utc": _iso(update["received_at_utc"]),
        "update_processing_started_at_utc": _iso(processing_started_at),
        "candle_persist_seconds": persist_seconds,
        "memory_esn_update_seconds": dynamic_seconds,
        "base70_incremental_update_seconds": base_seconds,
        "state_update_finished_at_utc": _iso(),
        "state_update_seconds": time.perf_counter() - processing_clock,
        "source": (update.get("source_metadata") or {}).get("source"),
    }
    if not allow_signal or not valid:
        return
    market_start = opened + pd.Timedelta(minutes=2)
    if market_start.minute % 5 != 0 or market_start.second != 0:
        return
    try:
        _predict_market(
            connection, predictor, runtime, runtime["boosters"], runtime["base_order"],
            market_start, opened, update["received_at_utc"], update.get("source_metadata") or {},
        )
    except ValueError as exc:
        _event(connection, "prediction_unavailable", {
            "market_start_utc": _iso(market_start), "input_opened_utc": _iso(opened),
            "input_received_at_utc": update["received_at_utc"],
            "runtime_timings": runtime.get("last_update_timings"),
            "reason": f"{type(exc).__name__}: {exc}",
        })


def _rest_update(candle, received_at, elapsed):
    opened = _utc(candle["Opened"])
    payload = dict(candle)
    quote = payload.pop("quote_volume")
    taker = payload.pop("taker_buy_quote_volume")
    return {
        "row": payload,
        "received_at_utc": _iso(received_at),
        "source_metadata": {
            "source": "binance_public_rest_gap_repair",
            "request_elapsed_seconds": float(elapsed),
            "event_at_utc": _iso(opened + pd.Timedelta(minutes=1)),
            "quote_volume": float(quote),
            "taker_buy_quote_volume": float(taker),
        },
    }


def _fetch_catchup_pair(start_opened, end_opened):
    start, end = _utc(start_opened), _utc(end_opened)
    with requests.Session() as session:
        session.headers.update({"User-Agent": "btc-candidate-shadow/1.0"})
        index = live_runtime._fetch_single_source_closed_ohlcv_range(
            session, start_opened=start, end_opened=end, source="index",
            symbol=live_runtime.SYMBOL, market_type=live_runtime.PRICE_MARKET,
        )
    futures = _fetch_um_raw(start, end)
    futures = futures.rename(columns={"Volume": "UM_BTCUSDT_Volume"})
    merged = index.merge(futures, on="Opened", how="inner", validate="one_to_one")
    merged["Volume"] = merged["UM_BTCUSDT_Volume"]
    expected = pd.date_range(start, end, freq="min")
    if not pd.DatetimeIndex(merged["Opened"]).equals(expected):
        raise RuntimeError("Binance index and USD-M quote-volume catch-up is incomplete")
    return merged


def _fetch_um_raw(start_opened, end_opened):
    start = _utc(start_opened)
    end = _utc(end_opened)
    interval_ms = 60_000
    start_ms = int(start.value // 1_000_000)
    end_ms = int(end.value // 1_000_000)
    url = "https://fapi.binance.com/fapi/v1/klines"
    rows = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "btc-candidate-shadow/1.0"})
        cursor = start_ms
        while cursor <= end_ms:
            response = session.get(
                url,
                params={"symbol": "BTCUSDT", "interval": "1m", "startTime": cursor,
                        "endTime": end_ms + interval_ms - 1, "limit": 1500},
                timeout=20,
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            rows.extend(batch)
            cursor = int(batch[-1][0]) + interval_ms
    records = []
    for item in rows:
        opened = pd.to_datetime(int(item[0]), unit="ms", utc=True)
        if opened < start or opened > end:
            continue
        records.append({
            "Opened": opened,
            "Volume": float(item[5]),
            "UM_BTCUSDT_Open": float(item[1]),
            "UM_BTCUSDT_High": float(item[2]),
            "UM_BTCUSDT_Low": float(item[3]),
            "UM_BTCUSDT_Close": float(item[4]),
            "quote_volume": float(item[7]),
            "taker_buy_quote_volume": float(item[10]),
        })
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.sort_values("Opened", kind="stable").drop_duplicates("Opened").reset_index(drop=True)


def _latest_candle_open(connection):
    row = connection.execute("SELECT opened_utc FROM candles ORDER BY opened_utc DESC LIMIT 1").fetchone()
    return None if row is None else _utc(row[0])


def _save_checkpoint(connection, runtime, predictor, identity_sha256):
    predictor_state = {
        key: value for key, value in predictor.__dict__.items()
        if key not in {"model", "volume_profile_save_pool", "reaction_profile_save_pool"}
    }
    payload = pickle.dumps({
        "predictor": predictor_state,
        "dynamic": _dynamic_checkpoint(runtime),
        "last_opened_utc": _iso(runtime["memory_state"].last_timestamp_ns),
    }, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(payload).hexdigest()
    connection.execute(
        "INSERT INTO checkpoints(id,identity_sha256,last_opened_utc,state_blob,state_sha256,saved_at_utc) "
        "VALUES (1,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET identity_sha256=excluded.identity_sha256,"
        "last_opened_utc=excluded.last_opened_utc,state_blob=excluded.state_blob,"
        "state_sha256=excluded.state_sha256,saved_at_utc=excluded.saved_at_utc",
        (identity_sha256, _iso(runtime["memory_state"].last_timestamp_ns), sqlite3.Binary(payload), digest, _iso()),
    )
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.commit()


def _restore_checkpoint(connection, runtime, predictor, identity_sha256):
    row = connection.execute("SELECT identity_sha256,state_blob,state_sha256 FROM checkpoints WHERE id=1").fetchone()
    if row is None:
        return False
    if row[0] != identity_sha256:
        raise RuntimeError("shadow checkpoint frozen identity differs; refusing to resume")
    payload = bytes(row[1])
    if hashlib.sha256(payload).hexdigest() != row[2]:
        raise RuntimeError("shadow checkpoint checksum mismatch")
    saved = pickle.loads(payload)
    predictor.__dict__.update(saved["predictor"])
    _restore_dynamic_state(runtime, saved["dynamic"])
    return True


def _initialize_runtime(connection, identity, bundle):
    initialization_clock = time.perf_counter()
    boosters_clock = time.perf_counter()
    base_order = bundle["base_feature_order"]
    boosters = candidates._load_frozen_boosters(bundle)
    booster_load_seconds = time.perf_counter() - boosters_clock
    predictor_clock = time.perf_counter()
    predictor, base64_order, old_bundle = _build_base_predictor(base_order)
    base_predictor_bootstrap_seconds = time.perf_counter() - predictor_clock
    finalist_records = json.loads((candidate_quality.SOURCE_OUT / "focused_finalist_pool.json").read_text(encoding="utf-8"))["finalists"]
    finalist = next(item for item in finalist_records if item.get("role") == "best_local_preliminary")
    esn_config = candidate_quality.expansion.engine_runner._config_from_record(finalist["effective_config"])
    dynamic_config_clock = time.perf_counter()
    runtime = _load_dynamic_config({"esn_config": esn_config}, bundle)
    dynamic_config_seconds = time.perf_counter() - dynamic_config_clock
    runtime["boosters"] = boosters
    runtime["base_order"] = base_order
    runtime["lightgbm_seeds"] = tuple(bundle["lightgbm_seeds"])
    identity_sha = _canonical_hash(identity)
    restore_clock = time.perf_counter()
    restored = _restore_checkpoint(connection, runtime, predictor, identity_sha)
    checkpoint_restore_seconds = time.perf_counter() - restore_clock
    replay_rows = 0
    if not restored:
        config = json.loads(candidates.CONFIG_PATH.read_text(encoding="utf-8"))
        inputs = candidates._load_approved_arrays(config, base_order)
        if inputs["esn_config"].to_dict() != esn_config.to_dict():
            raise RuntimeError("candidate ESN config differs between its frozen artifact and source replay")
        started = time.perf_counter()
        _initialize_dynamic_state(runtime, inputs)
        _event(connection, "cold_start_dynamic_replay", {
            "raw_rows": len(inputs["raw_opened"]),
            "elapsed_seconds": time.perf_counter() - started,
            "raw_last_opened_utc": _iso(pd.Timestamp(int(inputs["raw_opened"][-1]), unit="ns", tz="UTC")),
        })
        del inputs
    else:
        saved_last = _utc(runtime["memory_state"].last_timestamp_ns)
        rows = connection.execute(
            "SELECT opened_utc,candle_json,received_at_utc,source_json FROM candles WHERE opened_utc>? ORDER BY opened_utc",
            (_iso(saved_last),),
        ).fetchall()
        replay_rows = len(rows)
        replay_clock = time.perf_counter()
        for stamp, candle_json, received, source_json in rows:
            candle = json.loads(candle_json)
            candle["Opened"] = _utc(stamp)
            source = json.loads(source_json)
            update = {
                "row": candle,
                "received_at_utc": received,
                "source_metadata": source,
            }
            _apply_update(connection, predictor, runtime, update, allow_signal=False, persist=False)
        _event(connection, "checkpoint_state_replay", {
            "rows": replay_rows,
            "state_replay_seconds": time.perf_counter() - replay_clock,
            "predictions_backfilled": False,
        })
    catchup_fetch_seconds = 0.0
    catchup_replay_seconds = 0.0
    catchup_rows = 0
    last_dynamic = _utc(runtime["memory_state"].last_timestamp_ns)
    latest_closed = _utc().floor("min") - pd.Timedelta(minutes=1)
    if last_dynamic < latest_closed:
        start = last_dynamic + pd.Timedelta(minutes=1)
        started = time.perf_counter()
        catchup = _fetch_catchup_pair(start, latest_closed)
        received = _utc()
        catchup_fetch_seconds = time.perf_counter() - started
        replay_clock = time.perf_counter()
        catchup_rows = len(catchup)
        for candle in catchup.to_dict(orient="records"):
            update = _rest_update(candle, received, catchup_fetch_seconds)
            _apply_update(connection, predictor, runtime, update, allow_signal=False)
        catchup_replay_seconds = time.perf_counter() - replay_clock
        _event(connection, "initial_rest_catchup", {
            "start_opened_utc": _iso(start), "end_opened_utc": _iso(latest_closed),
            "rows": catchup_rows,
            "rest_fetch_seconds": catchup_fetch_seconds,
            "state_replay_seconds": catchup_replay_seconds,
            "prediction_backfill": False,
        })
    _save_checkpoint(connection, runtime, predictor, identity_sha)
    _event(connection, "runtime_initialization_complete", {
        "mode": "checkpoint_resume" if restored else "cold_start",
        "booster_load_seconds": booster_load_seconds,
        "base_predictor_bootstrap_seconds": base_predictor_bootstrap_seconds,
        "dynamic_engine_setup_seconds": dynamic_config_seconds,
        "checkpoint_restore_seconds": checkpoint_restore_seconds,
        "checkpoint_state_replay_rows": replay_rows,
        "initial_catchup_rows": catchup_rows,
        "initial_catchup_rest_fetch_seconds": catchup_fetch_seconds,
        "initial_catchup_state_replay_seconds": catchup_replay_seconds,
        "total_seconds": time.perf_counter() - initialization_clock,
    })
    runtime["identity_sha256"] = identity_sha
    runtime["checkpoint_at_monotonic"] = time.monotonic()
    return predictor, runtime


def _wait_until(target):
    target = _utc(target)
    while True:
        remaining = (target - _utc()).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def _next_market_start(now=None):
    current = _utc(now)
    start = current.ceil("5min")
    if start <= current:
        start += pd.Timedelta(minutes=5)
    return start


def _store_market(connection, market_start, metadata):
    connection.execute(
        "INSERT OR REPLACE INTO markets VALUES (?,?,?,?,?,?)",
        (_iso(market_start), metadata["slug"], metadata["condition_id"],
         _json(metadata["tokens"]), _json(metadata), _iso(metadata["received_at_utc"])),
    )
    connection.commit()


def _store_snapshot(connection, market_start, name, snapshot):
    connection.execute(
        "INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?)",
        (_iso(market_start), name, snapshot["request_started_at_utc"],
         snapshot["received_at_utc"], _json(snapshot)),
    )
    connection.commit()


def _drain_live_updates(feed, connection, predictor, runtime):
    updates = feed.drain()
    for update in updates:
        _process_update(connection, predictor, runtime, update, allow_signal=True)
    return len(updates)


def _wait_until_with_feed(target, feed, connection, predictor, runtime):
    target = _utc(target)
    while True:
        _drain_live_updates(feed, connection, predictor, runtime)
        remaining = (target - _utc()).total_seconds()
        if remaining <= 0:
            _drain_live_updates(feed, connection, predictor, runtime)
            return
        time.sleep(min(POLL_SECONDS, remaining))


def _failed_market_attempt(connection, market_start, status, error):
    timing = market_timing(market_start)
    timings = {
        "failed_at_utc": _iso(),
        "decision_deadline_utc": _iso(timing["decision_deadline_utc"]),
        "execution_snapshot_utc": _iso(timing["execution_snapshot_utc"]),
    }
    connection.execute(
        "INSERT OR REPLACE INTO market_attempts VALUES (?,?,?,?,?,?,?,?,?,?)",
        (_iso(market_start), status, _iso(timing["decision_deadline_utc"]),
         _iso(timing["execution_snapshot_utc"]), None, None, None,
         _json({"status": "unavailable", "actual_fill": "unknown"}),
         _json(timings), _iso()),
    )
    connection.commit()
    _event(connection, "market_window_failed", {
        "market_start_utc": _iso(market_start), "status": status, "error": error,
    })


def _execution_checks(snapshot, metadata, market_start):
    results = {}
    for side in ("up", "down"):
        book = snapshot.get(f"{side}_book")
        if not book:
            results[side] = {"priceable": False, "reason": "book_unavailable", "actual_fill": "unknown"}
            continue
        age = book.get("book_timestamp_age_seconds")
        if not snapshot.get("quote_valid") or not shadow._book_timestamp_age_is_acceptable(age):
            results[side] = {
                "priceable": False,
                "reason": "snapshot_quality_invalid" if not snapshot.get("quote_valid") else "stale_or_missing_book_timestamp",
                "actual_fill": "unknown",
                "token_id": str(metadata["tokens"][side]),
                "book_received_at_utc": book.get("received_at_utc"),
                "book_timestamp_utc": book.get("book_timestamp_utc"),
                "book_age_seconds": age,
            }
            continue
        levels = [(item["price"], item["size"]) for item in book.get("ask_levels", [])]
        results[side] = hypothetical_buy(
            levels, metadata["fee_model"], _utc(market_start) - pd.Timedelta(seconds=59), GROSS_ORDER_USD,
        )
        results[side]["token_id"] = str(metadata["tokens"][side])
        results[side]["book_received_at_utc"] = book.get("received_at_utc")
        results[side]["book_timestamp_utc"] = book.get("book_timestamp_utc")
        results[side]["book_age_seconds"] = book.get("book_timestamp_age_seconds")
    return results


def _collect_market(connection, session, feed, predictor, runtime, market_start):
    timing = market_timing(market_start)
    market_clock = time.perf_counter()
    try:
        metadata = prospective._market_metadata(session, market_start)
    except (requests.RequestException, ValueError) as exc:
        status = "market_metadata_invalid" if isinstance(exc, ValueError) else "market_metadata_unavailable"
        _failed_market_attempt(
            connection, market_start, status,
            f"{type(exc).__name__}: {exc}",
        )
        return None
    metadata_elapsed = time.perf_counter() - market_clock
    gamma = metadata["market"]
    metadata["settlement_metadata"] = {
        "rule_source": "captured Gamma market payload",
        "resolution_source": gamma.get("resolutionSource"),
        "resolution_source_url": gamma.get("resolutionSourceUrl"),
        "description": gamma.get("description"),
        "event_start_time": gamma.get("eventStartTime"),
        "end_date": gamma.get("endDate"),
    }
    metadata["shadow_identity_sha256"] = runtime["identity_sha256"]
    _store_market(connection, market_start, metadata)
    if not metadata["identity_valid"]:
        _failed_market_attempt(
            connection, market_start, "market_identity_invalid",
            "Gamma slug/start/end do not match the requested five-minute UTC market",
        )
        return None
    _wait_until_with_feed(timing["decision_deadline_utc"], feed, connection, predictor, runtime)
    t60 = prospective._quote_snapshot(metadata, "decision_T-60")
    _store_snapshot(connection, market_start, "decision_T-60", t60)
    _wait_until_with_feed(timing["execution_snapshot_utc"], feed, connection, predictor, runtime)
    prediction = connection.execute(
        "SELECT input_opened_utc,ready_at_utc,timely,members_json,ensembles_json,timings_json "
        "FROM predictions WHERE market_start_utc=?", (_iso(market_start),),
    ).fetchone()
    prediction_payload = None
    if prediction:
        prediction_payload = {
            "input_opened_utc": prediction[0], "ready_at_utc": prediction[1],
            "timely": bool(prediction[2]), "members": json.loads(prediction[3]),
            "ensembles": json.loads(prediction[4]), "timings": json.loads(prediction[5]),
        }
    t59 = prospective._quote_snapshot(metadata, "hypothetical_execution_T-59")
    _store_snapshot(connection, market_start, "hypothetical_execution_T-59", t59)
    execution = _execution_checks(t59, metadata, market_start)
    full_books = all(
        isinstance(snapshot.get(f"{side}_book"), dict)
        for snapshot in (t60, t59) for side in ("up", "down")
    )
    if prediction_payload is None:
        status = "prediction_unavailable_by_T-60"
    elif not prediction_payload["timely"]:
        status = "prediction_ready_late"
    elif not full_books:
        status = "book_incomplete"
    elif not t60["quote_valid"] or not t59["quote_valid"]:
        status = "book_quality_invalid"
    else:
        status = "complete"
    timings = {
        "market_metadata_seconds": metadata_elapsed,
        "decision_deadline_utc": _iso(timing["decision_deadline_utc"]),
        "execution_snapshot_utc": _iso(timing["execution_snapshot_utc"]),
        "t60_book_seconds": t60["request_duration_seconds"],
        "t59_book_seconds": t59["request_duration_seconds"],
        "t60_received_at_utc": t60["received_at_utc"],
        "t59_received_at_utc": t59["received_at_utc"],
        "t60_request_offset_from_deadline_seconds": (
            _utc(t60["request_started_at_utc"]) - timing["decision_deadline_utc"]
        ).total_seconds(),
        "t59_request_offset_from_deadline_seconds": (
            _utc(t59["request_started_at_utc"]) - timing["execution_snapshot_utc"]
        ).total_seconds(),
        "t60_snapshot_quality_valid": bool(t60["quote_valid"]),
        "t59_snapshot_quality_valid": bool(t59["quote_valid"]),
        "prediction_timely": None if prediction_payload is None else prediction_payload["timely"],
        "full_depth_both_tokens_at_both_times": full_books,
    }
    connection.execute(
        "INSERT OR REPLACE INTO market_attempts VALUES (?,?,?,?,?,?,?,?,?,?)",
        (_iso(market_start), status, _iso(timing["decision_deadline_utc"]),
         _iso(timing["execution_snapshot_utc"]), _json(prediction_payload) if prediction_payload else None,
         _json(t60), _json(t59), _json(execution), _json(timings), _iso()),
    )
    connection.commit()
    return metadata


def _poll_settlements(connection, session):
    rows = connection.execute(
        "SELECT market_start_utc,market_slug FROM markets WHERE market_start_utc<=? "
        "AND market_start_utc NOT IN (SELECT market_start_utc FROM official_outcomes) "
        "ORDER BY market_start_utc",
        (_iso(_utc() - pd.Timedelta(minutes=6)),),
    ).fetchall()
    for market_start_text, slug in rows:
        try:
            outcome = shadow._settlement_payload(session, slug)
        except requests.RequestException as exc:
            _event(connection, "settlement_lookup_failed", {"market_slug": slug, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if outcome is None:
            continue
        connection.execute(
            "INSERT OR IGNORE INTO official_outcomes VALUES (?,?,?,?,?)",
            (_iso(market_start_text), int(outcome["target_up"]), outcome["resolved_at_utc"],
             outcome["observed_at_utc"], _json(outcome["source_payload"])),
        )
        connection.commit()


def _collector_loop(connection, predictor, runtime):
    feed = prospective.PublicBtcCandleFeed()
    session = requests.Session()
    session.headers.update({"User-Agent": "btc-candidate-shadow/1.0"})
    feed.start()
    _event(connection, "collector_started", {
        "pid": os.getpid(), "started_at_utc": _iso(),
        "trade_calls": 0, "order_submission_count": 0,
        "feed_status": feed.status_snapshot(),
    })
    next_market = _next_market_start()
    next_settlement_poll = 0.0
    try:
        while True:
            for update in feed.drain():
                _process_update(connection, predictor, runtime, update, allow_signal=True)
            now = _utc()
            if now >= next_market - pd.Timedelta(seconds=120):
                _collect_market(connection, session, feed, predictor, runtime, next_market)
                if CONFIG["runtime"].get("checkpoint_after_completed_market"):
                    _save_checkpoint(connection, runtime, predictor, runtime["identity_sha256"])
                    runtime["checkpoint_at_monotonic"] = time.monotonic()
                next_market += pd.Timedelta(minutes=5)
            if time.monotonic() >= next_settlement_poll:
                _poll_settlements(connection, session)
                next_settlement_poll = time.monotonic() + SETTLEMENT_POLL_SECONDS
            if time.monotonic() - runtime["checkpoint_at_monotonic"] >= CHECKPOINT_SECONDS:
                _save_checkpoint(connection, runtime, predictor, runtime["identity_sha256"])
                runtime["checkpoint_at_monotonic"] = time.monotonic()
                _event(connection, "hourly_checkpoint", {"last_opened_utc": _iso(runtime["memory_state"].last_timestamp_ns)})
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        _save_checkpoint(connection, runtime, predictor, runtime["identity_sha256"])
        _event(connection, "collector_stopped", {"reason": "keyboard_interrupt", "pid": os.getpid()})
        raise
    finally:
        feed.stop()
        session.close()


def main():
    connection = _connect()
    ready, network = _network_probe(connection)
    if not ready:
        print(json.dumps({"status": "blocked_network_preflight", "endpoints": network}, indent=2), flush=True)
        connection.close()
        raise SystemExit(2)
    bundle = candidates._bundle_identity()
    identity = _runtime_identity(bundle)
    identity_sha = _canonical_hash(identity)
    _event(connection, "frozen_candidate_identity_verified", {"identity_sha256": identity_sha, "identity": identity})
    predictor, runtime = _initialize_runtime(connection, identity, bundle)
    try:
        _collector_loop(connection, predictor, runtime)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
