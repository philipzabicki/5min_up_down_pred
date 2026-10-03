"""Non-trading prospective BTC/Polymarket shadow collector; no order APIs."""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.special import expit, logit

import audit_feature_readiness as audit
import run as live_runtime
from features.reaction_profile_fixed_grid import (
    load_state as load_reaction_profile_state,
    save_state as save_reaction_profile_state,
)
from features.volume_profile_fixed_range import (
    load_state as load_volume_profile_state,
    save_state as save_volume_profile_state,
)
from utils.polymarket import (
    parse_json_listish,
    polymarket_fee_model_from_market,
    resolve_polymarket_actual_up_from_market_payload,
    resolve_polymarket_up_down_tokens,
)
from utils.polymarket_history import payoff
from utils.polymarket_market_value import decide_at_observed_book, market_features


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/btc_shadow_protocol_20261004.json"
CONFIG = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
SOURCE_MODEL_MANIFEST_PATH = ROOT / CONFIG.get(
    "source_model_manifest_path",
    "docs/btc_new_model_manifest_20261003.json",
)
FEATURE_SOURCE_MANIFEST_PATH = ROOT / CONFIG.get(
    "feature_source_manifest_path",
    SOURCE_MODEL_MANIFEST_PATH,
)
MODEL_WEIGHTS_PATH = ROOT / CONFIG["model_weights_path"]
MODEL_META_PATH = ROOT / CONFIG["model_meta_path"]
OUTPUT_DIR = ROOT / CONFIG["output_directory"]
DATABASE_PATH = OUTPUT_DIR / "shadow.sqlite3"
MANIFEST_PATH = OUTPUT_DIR / "session_manifest.json"
ANCHOR_OPENED = pd.Timestamp(CONFIG["profile_state_anchor_opened_utc"])
POLL_SECONDS = int(CONFIG["poll_seconds"])
CHECKPOINT_INTERVAL = pd.Timedelta(minutes=int(CONFIG["checkpoint_interval_minutes"]))
MAX_BOOK_AGE_SECONDS = float(
    CONFIG["execution_assumption"].get("max_book_age_seconds") or 30.0
)
VARIANTS = tuple(CONFIG["variants"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _artifact_hashes():
    manifest = json.loads(
        FEATURE_SOURCE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    feature_source_paths = []
    for relative_path, expected_hash in manifest["live_parity"]["feature_source_sha256"].items():
        path = ROOT / relative_path
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"Frozen BTC feature source differs from its manifest: {relative_path}")
        feature_source_paths.append(path)
    vp_anchor = audit.resolve_anchor_volume_profile_state_path(
        ANCHOR_OPENED,
        source_label="raw_csv",
    )
    rp_anchor = audit.resolve_anchor_reaction_profile_state_path(
        ANCHOR_OPENED,
        source_label="raw_csv",
    )
    raw_dataset = audit.resolve_raw_dataset_input_path(audit.MODELING_DATASET_SETTINGS)
    paths = [
        PROTOCOL_PATH,
        MODEL_WEIGHTS_PATH,
        MODEL_META_PATH,
        _source_model_path(MODEL_META_PATH),
        SOURCE_MODEL_MANIFEST_PATH,
        FEATURE_SOURCE_MANIFEST_PATH,
        ROOT / live_runtime.INDICATOR_HISTORY_REQUIREMENTS_PATH,
        ROOT / "data/datasets/modeling/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m_model_ready_metadata.json",
        raw_dataset,
        vp_anchor.with_suffix(".json"),
        vp_anchor.with_suffix(".npz"),
        rp_anchor.with_suffix(".json"),
        rp_anchor.with_suffix(".npz"),
        ROOT / "configs/live.json",
        ROOT / "configs/modeling.json",
        ROOT / "configs/datasets.json",
        ROOT / "configs/runtime/active.json",
        ROOT / "configs/runtime/trade_policy_project.json",
        ROOT / "run_btc_shadow.py",
        ROOT / "run.py",
        ROOT / "audit_feature_readiness.py",
        ROOT / "utils/polymarket_market_value.py",
        ROOT / "utils/polymarket_policy.py",
        ROOT / "utils/polymarket_history.py",
        ROOT / "utils/polymarket.py",
        ROOT / "features/live_indicator_runtime.py",
        ROOT / "features/volume_profile_fixed_range.py",
        ROOT / "features/reaction_profile_fixed_grid.py",
        *feature_source_paths,
    ]
    result = {}
    for raw_path in paths:
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Frozen shadow input is missing: {path}")
        result[str(path.relative_to(ROOT)).replace("\\", "/")] = _sha256(path)
    return result


def _source_model_path(meta_path):
    meta_payload = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    model_path = meta_payload.get("model_path") or meta_payload.get("model_file")
    if model_path:
        path = Path(model_path)
        return path if path.is_absolute() else ROOT / path
    model_id = (
        meta_payload.get("run_id")
        or meta_payload.get("model_id")
        or Path(meta_path).parent.name
    )
    if not model_id:
        raise ValueError("BTC model metadata has no model path or model id")
    return ROOT / f"data/models/BTC/{model_id}/lgbm_{model_id}.txt"


def _utc_iso(value=None):
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").isoformat()


def _load_anchor_profile_states():
    vp_path = audit.resolve_anchor_volume_profile_state_path(
        ANCHOR_OPENED,
        source_label="raw_csv",
    )
    rp_path = audit.resolve_anchor_reaction_profile_state_path(
        ANCHOR_OPENED,
        source_label="raw_csv",
    )
    return load_volume_profile_state(vp_path), load_reaction_profile_state(rp_path)


def _load_anchor_bootstrap(required_window):
    raw_path = audit.resolve_raw_dataset_input_path(audit.MODELING_DATASET_SETTINGS)
    meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    dataset_metadata, _ = audit.load_modeling_dataset_artifact_metadata(
        audit.resolve_modeling_dataset_parquet_path()
    )
    aux_columns = audit._basis_premium_auxiliary_columns(
        meta.get("feature_columns", []),
        dataset_metadata,
    )
    usecols = ["Opened", *live_runtime.OHLCV_COLS, *aux_columns]
    frame = pd.read_csv(raw_path, usecols=list(dict.fromkeys(usecols)))
    frame["Opened"] = pd.to_datetime(frame["Opened"], utc=True, errors="coerce")
    frame = (
        frame.loc[frame["Opened"].le(ANCHOR_OPENED)]
        .dropna(subset=["Opened"])
        .sort_values("Opened")
        .drop_duplicates(subset=["Opened"], keep="last")
        .tail(int(required_window))
        .reset_index(drop=True)
    )
    if len(frame) != int(required_window) or frame["Opened"].iloc[-1] != ANCHOR_OPENED:
        raise RuntimeError(
            "The frozen profile anchor does not have the required contiguous BTC history. "
            f"rows={len(frame)} required={required_window} last={frame['Opened'].iloc[-1] if len(frame) else None}"
        )
    if not frame["Opened"].diff().iloc[1:].eq(live_runtime.INTERVAL_DELTA).all():
        raise RuntimeError("The frozen bootstrap candles contain a data gap")
    return frame


def _make_predictor(bootstrap, vp_state, rp_state, *, max_keep):
    predictor = audit.PseudoLiveAuditPredictor(
        bootstrap,
        model_meta_path=MODEL_META_PATH,
        max_keep=max_keep,
        volume_profile_state=vp_state,
        reaction_profile_state=rp_state,
        allow_unstable_indicator_summary=True,
    )
    predictor.trade_policy_runtime = {"mode": "shadow_only"}
    return predictor


def _float(value, default=float("nan")):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _best_level(levels, side):
    sizes_by_price = {}
    for level in levels or []:
        price = _float(level.get("price"))
        size = _float(level.get("size"), 0.0)
        if 0.0 < price < 1.0 and size >= 0.0:
            sizes_by_price[price] = sizes_by_price.get(price, 0.0) + size
    normalized = [
        {"price": price, "size": size}
        for price, size in sizes_by_price.items()
    ]
    if not normalized:
        return float("nan"), 0.0, 0.0, 0.0, 0
    best = max(v["price"] for v in normalized) if side == "bid" else min(
        v["price"] for v in normalized
    )
    at_best = sum(v["size"] for v in normalized if abs(v["price"] - best) <= 1e-12)
    top_five = sorted(normalized, key=lambda v: v["price"], reverse=side == "bid")[:5]
    depth_shares = sum(v["size"] for v in top_five)
    depth_usd = sum(v["size"] * v["price"] for v in top_five)
    return float(best), float(at_best), float(depth_shares), float(depth_usd), len(normalized)


def _book_observation(session, token_id):
    requested_at = pd.Timestamp.now(tz="UTC")
    response = session.get(
        "https://clob.polymarket.com/book",
        params={"token_id": str(token_id)},
        timeout=8,
    )
    response.raise_for_status()
    received_at = pd.Timestamp.now(tz="UTC")
    payload = response.json()
    bid = _best_level(payload.get("bids", []), "bid")
    ask = _best_level(payload.get("asks", []), "ask")
    raw_timestamp = payload.get("timestamp")
    book_timestamp = None
    age_seconds = None
    try:
        book_timestamp = pd.to_datetime(int(raw_timestamp), unit="ms", utc=True)
        age_seconds = (received_at - book_timestamp).total_seconds()
    except (TypeError, ValueError, OverflowError):
        pass
    return {
        "token_id": str(token_id),
        "request_started_at_utc": _utc_iso(requested_at),
        "received_at_utc": _utc_iso(received_at),
        "book_timestamp_raw": raw_timestamp,
        "book_timestamp_utc": None if book_timestamp is None else _utc_iso(book_timestamp),
        "book_timestamp_age_seconds": age_seconds,
        "best_bid": bid[0],
        "best_bid_size": bid[1],
        "top5_bid_depth_shares": bid[2],
        "top5_bid_depth_usd": bid[3],
        "bid_level_count": bid[4],
        "best_ask": ask[0],
        "best_ask_size": ask[1],
        "top5_ask_depth_shares": ask[2],
        "top5_ask_depth_usd": ask[3],
        "ask_level_count": ask[4],
        "minimum_order_size": _float(payload.get("min_order_size"), 0.0),
        "tick_size": _float(payload.get("tick_size")),
        "neg_risk": bool(payload.get("neg_risk", False)),
    }


def _market_observation(session, bucket_start):
    bucket_start = pd.Timestamp(bucket_start).tz_convert("UTC")
    slug = f"{CONFIG['market_slug_prefix']}-{int(bucket_start.timestamp())}"
    received_at = pd.Timestamp.now(tz="UTC")
    response = session.get(
        f"https://gamma-api.polymarket.com/markets/slug/{slug}",
        timeout=8,
    )
    response.raise_for_status()
    market = response.json()
    gamma_received_at = pd.Timestamp.now(tz="UTC")
    tokens = resolve_polymarket_up_down_tokens(market)
    up_book = _book_observation(session, tokens["up"])
    down_book = _book_observation(session, tokens["down"])
    official_fee_model = polymarket_fee_model_from_market(market)
    start_value = market.get("eventStartTime") or next(
        (
            event.get("startTime")
            for event in market.get("events", [])
            if event.get("startTime")
        ),
        None,
    )
    market_start = None if not start_value else pd.to_datetime(start_value, utc=True)
    market_end = None if not market.get("endDate") else pd.to_datetime(market["endDate"], utc=True)
    identity_valid = (
        market.get("slug") == slug
        and market_start == bucket_start
        and market_end == bucket_start + pd.Timedelta(minutes=5)
        and int(bucket_start.timestamp()) % 300 == 0
    )
    quote_fields = {}
    for side, book in (("up", up_book), ("down", down_book)):
        quote_fields[f"{side}_best_bid"] = book["best_bid"]
        quote_fields[f"{side}_best_ask"] = book["best_ask"]
        quote_fields[f"{side}_bid_size"] = book["best_bid_size"]
        quote_fields[f"{side}_ask_size"] = book["best_ask_size"]
    values = list(quote_fields.values())
    quotes_finite = np.isfinite(np.asarray(values, dtype=float)).all()
    prices_valid = all(0.0 < quote_fields[f"{side}_best_bid"] < 1.0
                       and 0.0 < quote_fields[f"{side}_best_ask"] < 1.0
                       and quote_fields[f"{side}_best_bid"] <= quote_fields[f"{side}_best_ask"]
                       for side in ("up", "down"))
    sizes_valid = all(quote_fields[f"{side}_ask_size"] >= 0.0 for side in ("up", "down"))
    ages = [up_book["book_timestamp_age_seconds"], down_book["book_timestamp_age_seconds"]]
    age_valid = all(age is not None and 0.0 <= age <= MAX_BOOK_AGE_SECONDS for age in ages)
    valid = bool(
        identity_valid
        and not market.get("closed", False)
        and market.get("acceptingOrders", False)
        and quotes_finite
        and prices_valid
        and sizes_valid
        and age_valid
    )
    reason = None
    if not identity_valid:
        reason = "market_identity_or_boundary_mismatch"
    elif market.get("closed", False) or not market.get("acceptingOrders", False):
        reason = "market_not_accepting_orders"
    elif not quotes_finite or not prices_valid or not sizes_valid:
        reason = "invalid_or_missing_book"
    elif not age_valid:
        reason = "book_timestamp_missing_or_older_than_limit"
    return {
        "market_slug": slug,
        "condition_id": str(market.get("conditionId", "")),
        "market_question": str(market.get("question", "")),
        "market_received_at_utc": _utc_iso(received_at),
        "gamma_received_at_utc": _utc_iso(gamma_received_at),
        "market_start_utc": None if market_start is None else _utc_iso(market_start),
        "market_end_utc": None if market_end is None else _utc_iso(market_end),
        "up_token_id": str(tokens["up"]),
        "down_token_id": str(tokens["down"]),
        "market_identity_valid": bool(identity_valid),
        "market_accepting_orders": bool(market.get("acceptingOrders", False)),
        "market_closed": bool(market.get("closed", False)),
        "quote_valid": valid,
        "quote_invalid_reason": reason,
        "up_book": up_book,
        "down_book": down_book,
        "quote_fields": quote_fields,
        "order_min_size": max(
            _float(market.get("orderMinSize"), 0.0),
            _float(up_book.get("minimum_order_size"), 0.0),
            _float(down_book.get("minimum_order_size"), 0.0),
        ),
        "tick_size": max(
            _float(market.get("orderPriceMinTickSize"), 0.0),
            _float(up_book.get("tick_size"), 0.0),
            _float(down_book.get("tick_size"), 0.0),
        ),
        "observed_gamma_fee_model": official_fee_model,
        "frozen_fee_model_matches_gamma": all(
            abs(float(official_fee_model[key]) - float(CONFIG["hypothetical_fee_model"][key])) < 1e-12
            for key in ("rate", "exponent", "fee_round_decimals", "min_fee")
        ),
        "gamma_market_payload": market,
    }


def _sigmoid_predict(weights, frame):
    columns = weights["input_features"]
    matrix = frame.loc[:, columns].to_numpy(dtype=np.float64)
    mean = np.asarray(weights["mean"], dtype=np.float64)
    scale = np.asarray(weights["scale"], dtype=np.float64)
    coef = np.asarray(weights["coef"], dtype=np.float64)
    if matrix.shape[1] != len(mean) or len(mean) != len(scale) or len(scale) != len(coef):
        raise ValueError("Frozen shadow logistic model dimensions do not match")
    return float(expit(float(weights["intercept"]) + ((matrix[0] - mean) / scale) @ coef))


def _predict_variants(weights, raw_btc, quote_fields):
    btc_platt = float(
        expit(float(weights["btc_platt"]["intercept"]) +
              float(weights["btc_platt"]["coef"]) * logit(np.clip(raw_btc, 1e-6, 1 - 1e-6)))
    )
    if quote_fields is None:
        return raw_btc, btc_platt, None, None
    quote_frame = pd.DataFrame([quote_fields])
    market = market_features(quote_frame)
    market_only = _sigmoid_predict(weights["market_only"], market)
    combined_input = market.copy()
    combined_input["source_logit"] = logit(np.clip(raw_btc, 1e-6, 1 - 1e-6))
    combo = _sigmoid_predict(weights["market_plus_btc"], combined_input)
    return raw_btc, btc_platt, market_only, combo


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _encode_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default, allow_nan=False)


def _decode_json(payload):
    return json.loads(payload)


def _connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS session_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          event_type TEXT NOT NULL,
          observed_at_utc TEXT NOT NULL,
          payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS btc_candles (
          opened_utc TEXT PRIMARY KEY,
          ohlcv_json TEXT NOT NULL,
          futures_close REAL,
          received_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS decisions (
          condition_key TEXT NOT NULL,
          variant TEXT NOT NULL,
          market_slug TEXT NOT NULL,
          market_start_utc TEXT NOT NULL,
          decision_json TEXT NOT NULL,
          PRIMARY KEY (condition_key, variant)
        );
        CREATE TABLE IF NOT EXISTS settlements (
          condition_key TEXT PRIMARY KEY,
          market_slug TEXT NOT NULL,
          target_up INTEGER NOT NULL,
          resolved_at_utc TEXT,
          observed_at_utc TEXT NOT NULL,
          source_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_ledger (
          event_id TEXT PRIMARY KEY,
          variant TEXT NOT NULL,
          condition_key TEXT NOT NULL,
          event_type TEXT NOT NULL,
          event_time_utc TEXT NOT NULL,
          payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
          id INTEGER PRIMARY KEY CHECK (id = 1),
          last_opened_utc TEXT NOT NULL,
          frozen_hash TEXT NOT NULL,
          ohlcv_blob BLOB NOT NULL,
          vp_json_blob BLOB NOT NULL,
          vp_npz_blob BLOB NOT NULL,
          rp_json_blob BLOB NOT NULL,
          rp_npz_blob BLOB NOT NULL,
          updated_at_utc TEXT NOT NULL
        );
        """
    )
    return connection


def _load_profile_blob(json_blob, npz_blob, loader):
    with tempfile.TemporaryDirectory(prefix="btc-shadow-state-") as directory:
        base = Path(directory) / "state"
        base.with_suffix(".json").write_bytes(json_blob)
        base.with_suffix(".npz").write_bytes(npz_blob)
        return loader(base)


def _save_profile_blob(state, saver):
    with tempfile.TemporaryDirectory(prefix="btc-shadow-state-") as directory:
        base = Path(directory) / "state"
        saver(state, base)
        return base.with_suffix(".json").read_bytes(), base.with_suffix(".npz").read_bytes()


def _encode_checkpoint_ohlcv(predictor):
    output = io.BytesIO()
    arrays = {"ohlcv": predictor.ohlcv_np, "opened_ns": predictor.opened_ns_np}
    if predictor.basis_futures_close_np is not None:
        arrays["basis_futures_close"] = predictor.basis_futures_close_np
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _restore_checkpoint(connection, frozen_hash, model_meta_path, vp_state, rp_state, max_keep):
    row = connection.execute(
        "SELECT last_opened_utc,frozen_hash,ohlcv_blob,vp_json_blob,vp_npz_blob,"
        "rp_json_blob,rp_npz_blob FROM checkpoints WHERE id=1"
    ).fetchone()
    if row is None:
        return None
    if row[1] != frozen_hash:
        raise RuntimeError("Shadow checkpoint hashes differ; create a new experiment version")
    with np.load(io.BytesIO(row[2]), allow_pickle=False) as checkpoint:
        opened = pd.to_datetime(checkpoint["opened_ns"], utc=True)
        frame = pd.DataFrame(checkpoint["ohlcv"], columns=live_runtime.OHLCV_COLS)
        frame.insert(0, "Opened", opened)
        if "basis_futures_close" in checkpoint:
            frame["UM_BTCUSDT_Close"] = checkpoint["basis_futures_close"]
    vp_state = _load_profile_blob(row[3], row[4], load_volume_profile_state)
    rp_state = _load_profile_blob(row[5], row[6], load_reaction_profile_state)
    predictor = _make_predictor(frame, vp_state, rp_state, max_keep=max_keep)
    return predictor


def _save_checkpoint(connection, predictor, frozen_hash):
    vp_json, vp_npz = _save_profile_blob(predictor.volume_profile_state, save_volume_profile_state)
    rp_json, rp_npz = _save_profile_blob(predictor.reaction_profile_state, save_reaction_profile_state)
    connection.execute(
        "INSERT INTO checkpoints(id,last_opened_utc,frozen_hash,ohlcv_blob,vp_json_blob,vp_npz_blob,"
        "rp_json_blob,rp_npz_blob,updated_at_utc) VALUES (1,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET last_opened_utc=excluded.last_opened_utc,"
        "frozen_hash=excluded.frozen_hash,ohlcv_blob=excluded.ohlcv_blob,"
        "vp_json_blob=excluded.vp_json_blob,vp_npz_blob=excluded.vp_npz_blob,"
        "rp_json_blob=excluded.rp_json_blob,rp_npz_blob=excluded.rp_npz_blob,"
        "updated_at_utc=excluded.updated_at_utc",
        (
            _utc_iso(predictor.opened_candles[-1]),
            frozen_hash,
            _encode_checkpoint_ohlcv(predictor),
            sqlite3.Binary(vp_json),
            sqlite3.Binary(vp_npz),
            sqlite3.Binary(rp_json),
            sqlite3.Binary(rp_npz),
            _utc_iso(),
        ),
    )
    connection.commit()


def _account_states(connection):
    accounts = {}
    for variant in VARIANTS:
        accounts[variant] = {
            "cash": float(CONFIG["portfolio"]["initial_virtual_usd"]),
            "locked": 0.0,
            "peak_equity": float(CONFIG["portfolio"]["initial_virtual_usd"]),
            "max_drawdown": 0.0,
            "pending": {},
        }
        last = connection.execute(
            "SELECT payload_json FROM portfolio_ledger WHERE variant=? ORDER BY rowid DESC LIMIT 1",
            (variant,),
        ).fetchone()
        if last:
            payload = _decode_json(last[0])
            accounts[variant]["cash"] = float(payload["cash_after"])
            accounts[variant]["locked"] = float(payload["locked_after"])
        rows = connection.execute(
            "SELECT d.condition_key,d.decision_json FROM decisions d "
            "LEFT JOIN settlements s ON s.condition_key=d.condition_key "
            "WHERE d.variant=? AND s.condition_key IS NULL",
            (variant,),
        ).fetchall()
        for condition_key, decision_json in rows:
            payload = _decode_json(decision_json)
            trade = payload.get("hypothetical_trade")
            if trade:
                accounts[variant]["pending"][condition_key] = trade
        histories = connection.execute(
            "SELECT payload_json FROM portfolio_ledger WHERE variant=? ORDER BY rowid",
            (variant,),
        ).fetchall()
        for (raw,) in histories:
            payload = _decode_json(raw)
            accounts[variant]["peak_equity"] = max(
                accounts[variant]["peak_equity"], float(payload.get("peak_equity_after", 0.0))
            )
            accounts[variant]["max_drawdown"] = max(
                accounts[variant]["max_drawdown"], float(payload.get("max_drawdown_after", 0.0))
            )
    return accounts


def _append_ledger(connection, account, variant, condition_key, event_type, event_time, extra):
    equity = float(account["cash"] + account["locked"])
    account["peak_equity"] = max(float(account["peak_equity"]), equity)
    drawdown = 0.0 if account["peak_equity"] <= 0.0 else max(0.0, 1.0 - equity / account["peak_equity"])
    account["max_drawdown"] = max(float(account["max_drawdown"]), drawdown)
    payload = {
        **extra,
        "cash_after": float(account["cash"]),
        "locked_after": float(account["locked"]),
        "cost_basis_equity_after": equity,
        "peak_equity_after": float(account["peak_equity"]),
        "max_drawdown_after": float(account["max_drawdown"]),
        "actual_fill": "unknown",
        "actual_fee": "unknown",
    }
    event_id = f"{condition_key}:{variant}:{event_type}"
    connection.execute(
        "INSERT OR IGNORE INTO portfolio_ledger(event_id,variant,condition_key,event_type,event_time_utc,payload_json) "
        "VALUES (?,?,?,?,?,?)",
        (event_id, variant, condition_key, event_type, _utc_iso(event_time), _encode_json(payload)),
    )


def _resolve_virtual_settlement(connection, accounts, condition_key, outcome_up, observed_at):
    decision_rows = connection.execute(
        "SELECT variant,decision_json FROM decisions WHERE condition_key=?",
        (condition_key,),
    ).fetchall()
    for variant, raw in decision_rows:
        account = accounts[variant]
        trade = account["pending"].pop(condition_key, None)
        payout = 0.0
        pnl = 0.0
        if trade:
            outcome = int(outcome_up) if trade["side"] == "up" else 1 - int(outcome_up)
            payout = float(trade["shares"]) * outcome
            pnl = payout - float(trade["stake_usd"])
            account["cash"] += payout
            account["locked"] -= float(trade["stake_usd"])
            account["locked"] = max(account["locked"], 0.0)
        _append_ledger(
            connection,
            account,
            variant,
            condition_key,
            "settlement",
            observed_at,
            {
                "target_up": int(outcome_up),
                "hypothetical_payout_usd": payout if trade else None,
                "hypothetical_realized_pnl_usd": pnl if trade else None,
                "official_outcome": int(outcome_up),
            },
        )


def _settlement_payload(session, market_slug):
    response = session.get(
        f"https://gamma-api.polymarket.com/markets/slug/{market_slug}",
        timeout=8,
    )
    response.raise_for_status()
    payload = response.json()
    observed_at = pd.Timestamp.now(tz="UTC")
    if not payload.get("closed") or str(payload.get("umaResolutionStatus", "")).lower() != "resolved":
        return None
    outcome_prices = parse_json_listish(payload.get("outcomePrices"))
    try:
        values = sorted(float(value) for value in outcome_prices)
    except (TypeError, ValueError):
        return None
    if values != [0.0, 1.0]:
        return None
    outcome_up = resolve_polymarket_actual_up_from_market_payload(payload)
    if outcome_up not in (0, 1):
        return None
    resolved_at = payload.get("umaEndDate") or payload.get("closedTime")
    return {
        "target_up": int(outcome_up),
        "resolved_at_utc": None if not resolved_at else _utc_iso(pd.to_datetime(resolved_at, utc=True)),
        "observed_at_utc": _utc_iso(observed_at),
        "source_payload": payload,
    }


def _poll_settlements(connection, session, accounts):
    rows = connection.execute(
        "SELECT DISTINCT d.condition_key,d.market_slug,d.market_start_utc "
        "FROM decisions d LEFT JOIN settlements s ON s.condition_key=d.condition_key "
        "WHERE s.condition_key IS NULL AND d.condition_key != d.market_slug "
        "ORDER BY d.market_start_utc"
    ).fetchall()
    now = pd.Timestamp.now(tz="UTC")
    for condition_key, market_slug, market_start_text in rows:
        market_start = pd.to_datetime(market_start_text, utc=True)
        if now < market_start + pd.Timedelta(minutes=6):
            continue
        try:
            settled = _settlement_payload(session, market_slug)
        except requests.RequestException as exc:
            print(f"[shadow] settlement lookup failed {market_slug}: {exc}", flush=True)
            continue
        if settled is None:
            continue
        connection.execute(
            "INSERT OR IGNORE INTO settlements(condition_key,market_slug,target_up,resolved_at_utc,observed_at_utc,source_json) "
            "VALUES (?,?,?,?,?,?)",
            (
                condition_key,
                market_slug,
                int(settled["target_up"]),
                settled["resolved_at_utc"],
                settled["observed_at_utc"],
                _encode_json(settled["source_payload"]),
            ),
        )
        _resolve_virtual_settlement(
            connection,
            accounts,
            condition_key,
            settled["target_up"],
            settled["observed_at_utc"],
        )
        connection.commit()


def _read_saved_candles(connection, start_opened):
    rows = connection.execute(
        "SELECT opened_utc,ohlcv_json,futures_close FROM btc_candles WHERE opened_utc>=? ORDER BY opened_utc",
        (_utc_iso(start_opened),),
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    records = []
    for opened_text, raw_values, futures_close in rows:
        values = _decode_json(raw_values)
        record = {"Opened": pd.to_datetime(opened_text, utc=True)}
        record.update(dict(zip(live_runtime.OHLCV_COLS, values)))
        if futures_close is not None:
            record["UM_BTCUSDT_Close"] = float(futures_close)
        records.append(record)
    return pd.DataFrame(records)


def _save_candles(connection, frame, received_at):
    if frame.empty:
        return
    for row in frame.itertuples(index=False):
        values = [float(getattr(row, column)) for column in live_runtime.OHLCV_COLS]
        futures_close = getattr(row, "UM_BTCUSDT_Close", None)
        connection.execute(
            "INSERT OR IGNORE INTO btc_candles(opened_utc,ohlcv_json,futures_close,received_at_utc) "
            "VALUES (?,?,?,?)",
            (
                _utc_iso(row.Opened),
                _encode_json(values),
                None if futures_close is None or pd.isna(futures_close) else float(futures_close),
                _utc_iso(received_at),
            ),
        )
    connection.commit()


def _check_duplicate_candles(connection, frame):
    for row in frame.itertuples(index=False):
        existing = connection.execute(
            "SELECT ohlcv_json,futures_close FROM btc_candles WHERE opened_utc=?",
            (_utc_iso(row.Opened),),
        ).fetchone()
        if existing is None:
            continue
        values = np.asarray(_decode_json(existing[0]), dtype=np.float64)
        current = np.asarray([float(getattr(row, column)) for column in live_runtime.OHLCV_COLS], dtype=np.float64)
        if not np.array_equal(values, current):
            raise RuntimeError(f"Previously recorded BTC candle changed: {row.Opened}")


def _record_session(connection, frozen_hash, artifact_hashes):
    row = connection.execute(
        "SELECT payload_json FROM session_events WHERE event_type='session_start' ORDER BY id LIMIT 1"
    ).fetchone()
    if row:
        payload = _decode_json(row[0])
        if payload.get("frozen_hash") != frozen_hash:
            raise RuntimeError("Existing shadow session does not match frozen inputs")
        return payload
    payload = {
        "session_id": CONFIG["session_id"],
        "started_at_utc": _utc_iso(),
        "frozen_hash": frozen_hash,
        "artifact_sha256": artifact_hashes,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "model_weights_sha256": _sha256(MODEL_WEIGHTS_PATH),
        "model_meta_sha256": _sha256(MODEL_META_PATH),
        "base_model_sha256": _sha256(_source_model_path(MODEL_META_PATH)),
        "trading_enabled": False,
        "order_submission_count": 0,
    }
    connection.execute(
        "INSERT INTO session_events(event_type,observed_at_utc,payload_json) VALUES ('session_start',?,?)",
        (payload["started_at_utc"], _encode_json(payload)),
    )
    connection.commit()
    MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return payload


def _restore_account_states(connection):
    accounts = _account_states(connection)
    return accounts


def _record_market_decision(connection, accounts, market_data, market_id, market_start, market_end,
                            available_at, inference_started, inference_finished, prediction_values,
                            raw_btc, quote_observation, quote_fields, weights):
    condition_key = market_id or market_data["market_slug"]
    if connection.execute(
        "SELECT 1 FROM decisions WHERE condition_key=? LIMIT 1", (condition_key,)
    ).fetchone():
        return
    reason = None if quote_observation is None else quote_observation.get("quote_invalid_reason")
    quotes_valid = bool(quote_observation and quote_observation.get("quote_valid"))
    row_values = None
    if quotes_valid:
        row_values = {
            **quote_fields,
            "fee_rate": float(CONFIG["hypothetical_fee_model"]["rate"]),
            "fee_exponent": float(CONFIG["hypothetical_fee_model"]["exponent"]),
            "fee_round_decimals": int(CONFIG["hypothetical_fee_model"]["fee_round_decimals"]),
            "fee_min_fee": float(CONFIG["hypothetical_fee_model"]["min_fee"]),
            "order_min_size": float(quote_observation["order_min_size"]),
        }
    decision_at = _utc_iso()
    common = {
        "condition_key": condition_key,
        "market_slug": market_data["market_slug"],
        "condition_id": market_id or None,
        "market_start_utc": _utc_iso(market_start),
        "market_end_utc": _utc_iso(market_end),
        "btc_candle_opened_utc": _utc_iso(market_start - pd.Timedelta(minutes=1)),
        "btc_candle_assumed_available_at_utc": _utc_iso(market_start),
        "btc_data_received_at_utc": _utc_iso(available_at),
        "prediction_started_at_utc": _utc_iso(inference_started),
        "prediction_finished_at_utc": _utc_iso(inference_finished),
        "decision_at_utc": decision_at,
        "raw_btc_probability_up": float(raw_btc),
        "btc_platt_probability_up": float(prediction_values["btc_platt"]),
        "market_only_probability_up": prediction_values["market_only"],
        "market_plus_btc_probability_up": prediction_values["market_plus_btc"],
        "quote_observation": quote_observation,
        "quote_fields": quote_fields,
        "quote_valid": quotes_valid,
        "quote_invalid_reason": reason,
        "hypothetical_fill_assumption": CONFIG["execution_assumption"],
        "hypothetical_fee_model": CONFIG["hypothetical_fee_model"],
        "actual_fill": "unknown",
        "actual_fee": "unknown",
        "official_outcome": "unknown",
    }
    fee_fields = None if row_values is None else type("FeeRow", (), row_values)()
    for variant in VARIANTS:
        account = accounts[variant]
        probability = prediction_values[variant]
        decision = None
        trade = None
        reject_reason = reason or ("missing_or_stale_quote" if not quotes_valid else None)
        if quotes_valid and probability is not None:
            if account["cash"] < float(CONFIG["portfolio"]["fixed_hypothetical_stake_usd"]) - 1e-9:
                reject_reason = "insufficient_virtual_cash"
            else:
                selection, rejections = decide_at_observed_book(
                    fee_fields,
                    float(probability),
                    float(CONFIG["portfolio"]["expected_pnl_buffer_usd"]),
                )
                if selection is None:
                    reject_reason = ",".join(rejections)
                else:
                    ev, side, limit, theoretical_payout, p_success = selection
                    stake = float(CONFIG["portfolio"]["fixed_hypothetical_stake_usd"])
                    side_ask_size = float(row_values[f"{side}_ask_size"])
                    simulated = payoff(
                        stake,
                        float(limit),
                        1,
                        {
                            "rate": row_values["fee_rate"],
                            "exponent": row_values["fee_exponent"],
                            "fee_round_decimals": row_values["fee_round_decimals"],
                            "min_fee": row_values["fee_min_fee"],
                        },
                        side_ask_size,
                    )
                    if simulated is None:
                        reject_reason = "insufficient_top_level_ask_depth"
                    else:
                        cash_before = float(account["cash"])
                        equity_before = float(account["cash"] + account["locked"])
                        account["cash"] -= stake
                        account["locked"] += stake
                        trade = {
                            "side": str(side),
                            "stake_usd": stake,
                            "entry_price": float(limit),
                            "fee_usd": float(simulated["fee"]),
                            "shares": float(simulated["shares"]),
                            "expected_net_ev_usd": float(ev),
                            "selected_side_probability": float(p_success),
                            "top_level_ask_size_shares": side_ask_size,
                            "cash_before_usd": cash_before,
                            "equity_before_usd": equity_before,
                            "fraction_of_pre_entry_equity": stake / equity_before if equity_before > 0 else None,
                            "hypothetical_only": True,
                        }
                        decision = "hypothetical_trade"
                        reject_reason = None
        record = {
            **common,
            "variant": variant,
            "probability_up": None if probability is None else float(probability),
            "decision": decision or "no_trade",
            "decision_reason": reject_reason,
            "hypothetical_trade": trade,
            "simulated_execution": None if trade is None else {
                "assumed_fill_at_best_ask": trade["entry_price"],
                "assumed_fill_shares": trade["shares"],
                "assumed_fee_usd": trade["fee_usd"],
                "actual_fill": "unknown",
                "actual_fee": "unknown",
            },
        }
        connection.execute(
            "INSERT INTO decisions(condition_key,variant,market_slug,market_start_utc,decision_json) VALUES (?,?,?,?,?)",
            (condition_key, variant, market_data["market_slug"], _utc_iso(market_start), _encode_json(record)),
        )
        _append_ledger(
            connection,
            account,
            variant,
            condition_key,
            "decision",
            decision_at,
            {
                "decision": record["decision"],
                "reason": reject_reason,
                "probability_up": record["probability_up"],
                "hypothetical_stake_usd": None if trade is None else trade["stake_usd"],
                "hypothetical_fee_usd": None if trade is None else trade["fee_usd"],
                "fraction_of_pre_entry_equity": None if trade is None else trade["fraction_of_pre_entry_equity"],
            },
        )
        if trade is not None:
            account["pending"][condition_key] = trade
    connection.commit()


def _process_decision_candle(connection, predictor, model, weights, session, accounts,
                             opened, ohlcv, futures_close, data_received_at):
    predictor._append_new_candle(
        opened,
        ohlcv,
        basis_futures_close=futures_close,
    )
    volume_values = predictor._prepare_volume_profile_features_for_latest_candle(opened)
    reaction_values = predictor._prepare_reaction_profile_features_for_latest_candle(opened)
    if int(pd.Timestamp(opened).minute) % int(predictor.target_bucket_minutes) != int(predictor.target_bucket_minutes) - 1:
        return
    market_start = pd.Timestamp(opened) + live_runtime.INTERVAL_DELTA
    if market_start <= pd.to_datetime(
            _record_session_start(connection), utc=True
    ):
        return
    market_slug = f"{CONFIG['market_slug_prefix']}-{int(market_start.timestamp())}"
    if connection.execute(
        "SELECT 1 FROM decisions WHERE market_slug=? LIMIT 1",
        (market_slug,),
    ).fetchone():
        return
    inference_started = pd.Timestamp.now(tz="UTC")
    snapshot = predictor.build_feature_snapshot(
        volume_profile_values=volume_values,
        reaction_profile_values=reaction_values,
    )
    vector = snapshot["vector"]
    if snapshot["nonfinite_feature_indices"]:
        raise RuntimeError(
            "Shadow model input has non-finite values at "
            f"{opened.isoformat()}: {snapshot['nonfinite_feature_indices'][:10]}"
        )
    raw_btc = float(model.predict(vector)[0])
    inference_finished = pd.Timestamp.now(tz="UTC")
    bucket_start = market_start
    market_data = {
        "market_slug": f"{CONFIG['market_slug_prefix']}-{int(bucket_start.timestamp())}",
        "condition_id": "",
        "market_question": "",
    }
    quote_observation = None
    quote_fields = None
    try:
        quote_observation = _market_observation(session, bucket_start)
        market_data.update(quote_observation)
        quote_fields = quote_observation["quote_fields"]
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        quote_observation = {
            "market_slug": market_data["market_slug"],
            "quote_valid": False,
            "quote_invalid_reason": f"market_or_book_fetch_error:{type(exc).__name__}:{exc}",
            "market_received_at_utc": _utc_iso(),
        }
        market_data.update(quote_observation)
    market_id = str(market_data.get("condition_id") or "")
    if not market_id:
        market_id = str(market_data["market_slug"])
    if quote_observation and quote_observation.get("quote_valid"):
        raw, btc_platt, market_only, combo = _predict_variants(
            weights,
            raw_btc,
            quote_fields,
        )
        prediction_values = {
            "btc_platt": btc_platt,
            "market_only": market_only,
            "market_plus_btc": combo,
        }
    else:
        raw = raw_btc
        btc_platt = float(
            expit(float(weights["btc_platt"]["intercept"]) +
                  float(weights["btc_platt"]["coef"]) * logit(np.clip(raw_btc, 1e-6, 1 - 1e-6)))
        )
        prediction_values = {"btc_platt": btc_platt, "market_only": None, "market_plus_btc": None}
    try:
        market_start_value = pd.to_datetime(market_data.get("market_start_utc") or bucket_start, utc=True)
        market_end_value = pd.to_datetime(
            market_data.get("market_end_utc") or (bucket_start + pd.Timedelta(minutes=5)),
            utc=True,
        )
        _record_market_decision(
            connection,
            accounts,
            market_data,
            market_id,
            market_start_value,
            market_end_value,
            data_received_at,
            inference_started,
            inference_finished,
            prediction_values,
            raw,
            quote_observation,
            quote_fields,
            weights,
        )
        print(
            f"[shadow] decision {market_data['market_slug']} "
            f"quote_valid={bool(quote_observation and quote_observation.get('quote_valid'))} "
            f"BTC={raw:.6f} combined={prediction_values['market_plus_btc']} "
            f"at={_utc_iso()}",
            flush=True,
        )
    except sqlite3.IntegrityError:
        return


def _record_session_start(connection):
    row = connection.execute(
        "SELECT payload_json FROM session_events WHERE event_type='session_start' ORDER BY id LIMIT 1"
    ).fetchone()
    if not row:
        raise RuntimeError("Shadow session start was not recorded")
    return _decode_json(row[0])["started_at_utc"]


def _restore_predictor(connection, frozen_hash, required_window):
    predictor = _restore_checkpoint(
        connection,
        frozen_hash,
        MODEL_META_PATH,
        None,
        None,
        required_window,
    )
    if predictor is not None:
        return predictor

    bootstrap = _load_anchor_bootstrap(required_window)
    vp_state, rp_state = _load_anchor_profile_states()
    predictor = _make_predictor(bootstrap, vp_state, rp_state, max_keep=required_window)
    # Rebuild runtime states from the persisted one-minute rows if this process
    # previously ran before its first hourly checkpoint.
    saved = _read_saved_candles(connection, ANCHOR_OPENED + live_runtime.INTERVAL_DELTA)
    if not saved.empty:
        validate_end = saved["Opened"].iloc[-1]
        if not saved["Opened"].diff().iloc[1:].eq(live_runtime.INTERVAL_DELTA).all():
            raise RuntimeError("Saved shadow BTC candles contain a gap")
        for row in saved.itertuples(index=False):
            opened = pd.Timestamp(row.Opened)
            futures_close = getattr(row, "UM_BTCUSDT_Close", None)
            predictor._append_new_candle(
                opened,
                tuple(float(getattr(row, column)) for column in live_runtime.OHLCV_COLS),
                basis_futures_close=futures_close,
            )
            predictor._update_volume_profile_state_for_latest_candle(opened, persist=False)
            predictor._update_reaction_profile_state_for_latest_candle(opened, persist=False)
    return predictor


def _load_market_weights():
    payload = json.loads(MODEL_WEIGHTS_PATH.read_text(encoding="utf-8"))
    if payload.get("artifact_version") != 1:
        raise ValueError("Unsupported BTC shadow frozen model artifact")
    if payload.get("market_only", {}).get("variant") != "market_only":
        raise ValueError("Frozen market-only model variant identity is invalid")
    if payload.get("market_plus_btc", {}).get("variant") != "market_plus_oof":
        raise ValueError("Frozen market-plus-BTC model variant identity is invalid")
    return payload


def _load_btc_model():
    model, meta = audit.load_model_and_meta(MODEL_META_PATH)
    expected_id = "20261003_043549"
    if expected_id not in str(MODEL_META_PATH) or len(meta.get("feature_columns", [])) != 256:
        raise ValueError("Resolved BTC model does not match the frozen 20261003_043549 model")
    model_path = _source_model_path(MODEL_META_PATH)
    return model, meta, model_path


def _fetch_new_candles(session, last_opened):
    latest_closed = pd.Timestamp.now(tz="UTC").floor(live_runtime.INTERVAL_FLOOR_RULE) - live_runtime.INTERVAL_DELTA
    start = pd.Timestamp(last_opened) + live_runtime.INTERVAL_DELTA
    if start > latest_closed:
        return pd.DataFrame()
    frame = live_runtime.fetch_closed_ohlcv_range(
        session,
        start_opened=start,
        end_opened=latest_closed,
    )
    if frame.empty:
        raise RuntimeError(
            "BTC REST catch-up returned no candles for an expected closed gap "
            f"{start.isoformat()} -> {latest_closed.isoformat()}"
        )
    live_runtime.validate_closed_ohlcv_catchup(frame, start, latest_closed)
    return frame


def run_shadow():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_hashes = _artifact_hashes()
    frozen_hash = _canonical_json_sha256({"config": CONFIG, "artifacts": artifact_hashes})
    weights = _load_market_weights()
    model, meta, model_path = _load_btc_model()
    required = audit.load_indicator_history_requirements(
        live_runtime.INDICATOR_HISTORY_REQUIREMENTS_PATH,
        indicator_specs=audit.load_indicator_specs(meta["feature_columns"]),
        allow_unstable=True,
    )
    required_window = int(required["global_required_runtime_window"])
    connection = _connect(DATABASE_PATH)
    session_info = _record_session(connection, frozen_hash, artifact_hashes)
    predictor = _restore_predictor(connection, frozen_hash, required_window)
    accounts = _restore_account_states(connection)
    session = requests.Session()
    session.headers.update({"User-Agent": "btc-up-down-shadow/1.0"})
    last_checkpoint_at = pd.Timestamp.now(tz="UTC")
    print(
        "[shadow] started "
        f"session={CONFIG['session_id']} at={session_info['started_at_utc']} "
        f"last_btc_candle={predictor.opened_candles[-1].isoformat()} "
        f"model={model_path.name} orders=disabled",
        flush=True,
    )
    try:
        while True:
            try:
                catchup = _fetch_new_candles(session, predictor.opened_candles[-1])
                received_at = pd.Timestamp.now(tz="UTC")
                if not catchup.empty:
                    _check_duplicate_candles(connection, catchup)
                    _save_candles(connection, catchup, received_at)
                    for row in catchup.itertuples(index=False):
                        futures_close = getattr(row, "UM_BTCUSDT_Close", None)
                        _process_decision_candle(
                            connection,
                            predictor,
                            model,
                            weights,
                            session,
                            accounts,
                            pd.Timestamp(row.Opened),
                            tuple(float(getattr(row, column)) for column in live_runtime.OHLCV_COLS),
                            futures_close,
                            received_at,
                        )
                _poll_settlements(connection, session, accounts)
                now = pd.Timestamp.now(tz="UTC")
                if now - last_checkpoint_at >= CHECKPOINT_INTERVAL:
                    _save_checkpoint(connection, predictor, frozen_hash)
                    last_checkpoint_at = now
                    print(f"[shadow] checkpoint {predictor.opened_candles[-1].isoformat()}", flush=True)
            except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
                connection.execute(
                    "INSERT INTO session_events(event_type,observed_at_utc,payload_json) VALUES (?,?,?)",
                    (
                        "collector_error",
                        _utc_iso(),
                        _encode_json({"error_type": type(exc).__name__, "error": str(exc)}),
                    ),
                )
                connection.commit()
                print(f"[shadow] cycle failed; retained last good state: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("[shadow] stopped by operator; decisions remain append-only", flush=True)
    finally:
        _save_checkpoint(connection, predictor, frozen_hash)
        connection.close()
        session.close()


if __name__ == "__main__":
    run_shadow()
