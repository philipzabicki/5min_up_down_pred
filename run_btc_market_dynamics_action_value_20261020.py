"""Development-only BTC market-dynamics and action-value experiment."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import time
import traceback
from collections import defaultdict
import importlib.util
import io
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, mean_squared_error

from features import btc_market_dynamics_action_value as study
import run_btc_preopen_economic_replay as replay


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_market_dynamics_action_value_20261020.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_directory"]
RUN_MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
AVAILABILITY_PATH = OUTPUT_DIR / "availability.json"
FREEZE_PATH = OUTPUT_DIR / "frozen_protocol.json"
STATIC_PATH = OUTPUT_DIR / "static_inputs.parquet"
DYNAMIC_PATH = OUTPUT_DIR / "dynamic_inputs.parquet"
ENDPOINT_PATH = OUTPUT_DIR / "endpoint_states.parquet"
PREDICTION_PATH = OUTPUT_DIR / "evaluation_predictions.parquet"
ECONOMICS_DIR = OUTPUT_DIR / "economics"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
COMMON_PATH = OUTPUT_DIR / "common_inputs.parquet"
EXECUTION_TARGET_PATH = OUTPUT_DIR / "execution_targets.parquet"
DEVELOPMENT_CUTOFF = pd.Timestamp(CONFIG["development_cutoff_exclusive_utc"])
FIRST_POSSIBLE_START = pd.Timestamp("2026-04-15T00:00:00Z")
ENTRY_OFFSET_SECONDS = int(CONFIG["execution_offset_seconds"])
DECISION_OFFSET_SECONDS = int(CONFIG["decision_offset_seconds"])
ACTION_SIZES = tuple(float(x) for x in CONFIG["action_sizes_usd"])
HORIZONS = tuple(int(x) for x in (5, 15, 60))
STATIC_FEATURES = tuple(CONFIG["static_feature_order"])
DYNAMIC_FEATURES = tuple(CONFIG["dynamic_feature_order"])
ACTION_DESCRIPTORS = tuple(CONFIG["action_descriptor_order"])
SHARD_COUNT = 4
CHECKPOINT_PARTS = 4
BOOTSTRAP_REPLICATES = int(CONFIG["bootstrap"]["replications"])
BOOTSTRAP_SEED = int(CONFIG["bootstrap"]["seed"])

MARKET_SOURCE = ROOT / CONFIG["sources"]["market_index_path"]
RAW_SOURCE = ROOT / CONFIG["sources"]["raw_base70_oof_path"]
T60_SOURCE = ROOT / CONFIG["sources"]["t60_snapshot_path"]
T59_SOURCE = ROOT / CONFIG["sources"]["t59_ladders_path"]
EVENT_DIR = ROOT / CONFIG["sources"]["event_parts_dir"]
DEPENDENCY_PATH = ROOT / CONFIG["sources"]["event_dependency_manifest_path"]
REPLAY_MANIFEST_PATH = ROOT / CONFIG["sources"]["t59_replay_manifest_path"]

EVENT_COLUMNS = [
    "timestamp_received", "timestamp", "market", "event_type", "asset_id",
    "bids", "asks", "price", "size", "side", "best_bid", "best_ask",
    "fee_rate_bps",
]
T60_COLUMNS = [
    "condition_id", "market_start_utc", "entry_time_utc", "has_full_snapshot",
    "quote_valid", "archive_entry_hour_available", "archive_freshness_window_available",
    "archive_window_hours_expected", "archive_window_hours_source_available",
    "archive_window_hours_processed", "archive_window_missing_source_hours",
    "archive_window_unprocessed_hours", "market_event_count", "market_book_snapshot_count",
    "no_future_event_at_entry", "no_future_source_event_at_entry",
    "up_best_bid", "up_best_ask", "up_best_ask_size_shares",
    "down_best_bid", "down_best_ask", "down_best_ask_size_shares",
    "up_ask_levels", "down_ask_levels", "up_quote_source_token_id",
    "down_quote_source_token_id", "up_quote_complemented", "down_quote_complemented",
    "up_ask_order_ambiguous", "down_ask_order_ambiguous",
    "up_bbo_ask_comparable", "down_bbo_ask_comparable",
    "up_bbo_ask_mismatch", "down_bbo_ask_mismatch",
    "up_bid_ask_crossed", "down_bid_ask_crossed", "up_bid_ask_locked",
    "down_bid_ask_locked", "up_book_init_receive_ns", "down_book_init_receive_ns",
    "archive_gap_after_up_book_init", "archive_gap_after_down_book_init",
]
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _atomic_pickle(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _dataset_dev(path: Path, columns: list[str], *, start_at=None) -> pa.Table:
    dataset = ds.dataset(str(path), format="parquet")
    predicate = ds.field("market_start_utc") < DEVELOPMENT_CUTOFF.to_pydatetime()
    if start_at is not None:
        predicate = predicate & (ds.field("market_start_utc") >= pd.Timestamp(start_at).to_pydatetime())
    return dataset.to_table(columns=columns, filter=predicate)


def _read_dev_market_index(*, labels: bool = False) -> pd.DataFrame:
    columns = ["condition_id", "market_start_utc", "up_token_id", "down_token_id",
               "archive_history_complete", "archive_freshness_window_available",
               "archive_window_hours_expected", "archive_window_hours_source_available",
               "archive_window_hours_processed", "archive_window_missing_source_hours",
               "archive_window_unprocessed_hours"]
    if labels:
        columns.extend(["resolved_at_utc", "target_polymarket_up"])
    frame = _dataset_dev(MARKET_SOURCE, columns).to_pandas()
    frame["condition_id"] = frame.condition_id.astype(str).str.lower()
    study.assert_development_starts(frame.market_start_utc, context="development market index")
    if frame.condition_id.duplicated().any() or frame.market_start_utc.duplicated().any():
        raise RuntimeError("development market index must contain one row per market start and condition")
    return frame


def _read_raw_oof(start_at=None) -> pd.DataFrame:
    frame = _dataset_dev(
        RAW_SOURCE,
        ["market_start_utc", "nominal_decision_at", "prediction_available_at_utc", "p_model_raw", "fold_id"],
        start_at=start_at,
    ).to_pandas()
    frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True)
    frame["nominal_decision_at"] = pd.to_datetime(frame.nominal_decision_at, utc=True)
    frame["prediction_available_at_utc"] = pd.to_datetime(frame.prediction_available_at_utc, utc=True)
    study.assert_development_starts(frame.market_start_utc, context="RAW BASE70 OOF")
    if frame.market_start_utc.duplicated().any():
        raise RuntimeError("RAW BASE70 must have one forecast for every market start")
    return frame


def _read_t60() -> pd.DataFrame:
    frame = _dataset_dev(T60_SOURCE, T60_COLUMNS).to_pandas()
    frame["condition_id"] = frame.condition_id.astype(str).str.lower()
    frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True)
    frame["entry_time_utc"] = pd.to_datetime(frame.entry_time_utc, utc=True)
    study.assert_development_starts(frame.market_start_utc, context="T-60 snapshots")
    if frame.condition_id.duplicated().any() or frame.market_start_utc.duplicated().any():
        raise RuntimeError("T-60 snapshots must contain one row per development market")
    return frame


def _levels(value) -> dict[float, float]:
    if value is None:
        return {}
    try:
        if pd.isna(value):
            return {}
    except (TypeError, ValueError):
        pass
    result = {}
    for pair in value:
        if pair is None or len(pair) < 2:
            continue
        try:
            price, size = float(pair[0]), float(pair[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and math.isfinite(size) and 0.0 < price < 1.0 and size > 0.0:
            result[round(price, 8)] = size
    return result


def _number(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_bool(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _as_int(value, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        if pd.isna(value):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _market_probability(up_bid, up_ask, down_bid, down_ask):
    values = [_number(x) for x in (up_bid, up_ask, down_bid, down_ask)]
    if any(x is None for x in values):
        return None
    ub, ua, db, da = values
    if ub > ua or db > da:
        return None
    up_mid, down_mid = (ub + ua) / 2.0, (db + da) / 2.0
    total = up_mid + down_mid
    return up_mid / total if total > 0.0 else None


def _static_record(raw, snapshot, market):
    start = pd.Timestamp(market.market_start_utc)
    expected_decision = start + pd.Timedelta(seconds=DECISION_OFFSET_SECONDS)
    if pd.Timestamp(snapshot.entry_time_utc) != expected_decision:
        return None, "t60_time_misaligned"
    p_raw = _number(raw.p_model_raw)
    if p_raw is None or not 0.0 <= p_raw <= 1.0:
        return None, "raw_probability_missing_or_invalid"
    static_checks = {
        "has_full_snapshot": snapshot.has_full_snapshot,
        "quote_valid": snapshot.quote_valid,
        "archive_entry_hour_available": snapshot.archive_entry_hour_available,
        "archive_freshness_window_available": snapshot.archive_freshness_window_available,
        "no_future_event_at_entry": snapshot.no_future_event_at_entry,
        "no_future_source_event_at_entry": snapshot.no_future_source_event_at_entry,
    }
    if not all(_as_bool(value) for value in static_checks.values()):
        return None, "static_source_or_native_book_unavailable"
    if _as_bool(snapshot.up_quote_complemented) or _as_bool(snapshot.down_quote_complemented):
        return None, "static_complemented_book"
    if _as_bool(snapshot.up_ask_order_ambiguous) or _as_bool(snapshot.down_ask_order_ambiguous):
        return None, "static_ask_order_ambiguous"
    if _as_bool(snapshot.up_bid_ask_crossed) or _as_bool(snapshot.down_bid_ask_crossed):
        return None, "static_crossed_book"
    if _as_bool(snapshot.archive_gap_after_up_book_init) or _as_bool(snapshot.archive_gap_after_down_book_init):
        return None, "static_archive_gap_after_book_init"
    if _as_int(snapshot.archive_window_missing_source_hours) > 0 or _as_int(snapshot.archive_window_unprocessed_hours) > 0:
        return None, "static_archive_window_incomplete"
    up_levels, down_levels = _levels(snapshot.up_ask_levels), _levels(snapshot.down_ask_levels)
    up_bid, up_ask = _number(snapshot.up_best_bid), _number(snapshot.up_best_ask)
    down_bid, down_ask = _number(snapshot.down_best_bid), _number(snapshot.down_best_ask)
    if not up_levels or not down_levels or None in (up_bid, up_ask, down_bid, down_ask):
        return None, "static_native_depth_missing"
    if abs(min(up_levels) - up_ask) > 1e-6 or abs(min(down_levels) - down_ask) > 1e-6:
        return None, "static_best_ask_ladder_mismatch"
    p_market = _market_probability(up_bid, up_ask, down_bid, down_ask)
    if p_market is None or not 0.0 <= p_market <= 1.0:
        return None, "static_market_probability_invalid"
    up_vwap, down_vwap = study.gross_vwap(up_levels, 5.0), study.gross_vwap(down_levels, 5.0)
    up_depth, down_depth = study.ask_band_notional(up_levels, up_ask), study.ask_band_notional(down_levels, down_ask)
    if up_vwap is None or down_vwap is None or up_depth <= 0.0 or down_depth <= 0.0:
        return None, "static_five_dollar_depth_missing"
    mid_up, mid_down = (up_bid + up_ask) / 2.0, (down_bid + down_ask) / 2.0
    features = {
        "raw_logit": study.clipped_logit(p_raw),
        "market_logit": study.clipped_logit(p_market),
        "raw_minus_market_logit": study.clipped_logit(p_raw) - study.clipped_logit(p_market),
        "spread_up": up_ask - up_bid,
        "spread_down": down_ask - down_bid,
        "mid_sum_minus_one": mid_up + mid_down - 1.0,
        "log1p_ask_notional_up": math.log1p(up_depth),
        "log1p_ask_notional_down": math.log1p(down_depth),
        "slippage_5_up": up_vwap - up_ask,
        "slippage_5_down": down_vwap - down_ask,
    }
    record = {
        "condition_id": str(market.condition_id),
        "market_start_utc": start,
        "decision_time_utc": expected_decision,
        "p_raw": p_raw,
        "p_market": p_market,
        "up_token_id": str(market.up_token_id),
        "down_token_id": str(market.down_token_id),
        "up_best_bid_t60": up_bid,
        "up_best_ask_t60": up_ask,
        "down_best_bid_t60": down_bid,
        "down_best_ask_t60": down_ask,
        "up_ask_levels_t60": sorted(up_levels.items()),
        "down_ask_levels_t60": sorted(down_levels.items()),
        "expected_t60_event_count": _as_int(snapshot.market_event_count),
        "expected_t60_book_count": _as_int(snapshot.market_book_snapshot_count),
        "up_bid_ask_locked_t60": _as_bool(snapshot.up_bid_ask_locked),
        "down_bid_ask_locked_t60": _as_bool(snapshot.down_bid_ask_locked),
        "raw_fold_id": str(raw.fold_id),
        "raw_prediction_available_at_utc": raw.prediction_available_at_utc,
        "archive_window_hours_expected": _as_int(snapshot.archive_window_hours_expected),
        "archive_window_hours_source_available": _as_int(snapshot.archive_window_hours_source_available),
        "archive_window_hours_processed": _as_int(snapshot.archive_window_hours_processed),
        "archive_window_missing_source_hours": _as_int(snapshot.archive_window_missing_source_hours),
        "archive_window_unprocessed_hours": _as_int(snapshot.archive_window_unprocessed_hours),
        **features,
    }
    if not np.isfinite(np.asarray([record[c] for c in STATIC_FEATURES], dtype=np.float64)).all():
        return None, "static_nonfinite_feature"
    return record, None


def _load_input_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    market_index = _read_dev_market_index(labels=False)
    earliest = market_index.market_start_utc.min()
    raw = _read_raw_oof(start_at=earliest)
    t60 = _read_t60()
    if raw.market_start_utc.min() < earliest:
        raise RuntimeError("RAW OOF unexpectedly contains starts before the official market index")
    raw_map = {pd.Timestamp(row.market_start_utc): row for row in raw.itertuples(index=False)}
    t60_map = {str(row.condition_id).lower(): row for row in t60.itertuples(index=False)}
    records, reasons = [], defaultdict(int)
    rows = []
    for market in market_index.itertuples(index=False):
        raw_row = raw_map.get(pd.Timestamp(market.market_start_utc))
        if raw_row is None:
            reasons["raw_oof_missing"] += 1
            continue
        snapshot = t60_map.get(str(market.condition_id).lower())
        if snapshot is None:
            reasons["t60_snapshot_missing"] += 1
            continue
        expected_decision = pd.Timestamp(market.market_start_utc) + pd.Timedelta(seconds=DECISION_OFFSET_SECONDS)
        if pd.Timestamp(raw_row.nominal_decision_at) != expected_decision:
            reasons["raw_decision_time_misaligned"] += 1
            continue
        if pd.Timestamp(raw_row.prediction_available_at_utc) > expected_decision + pd.Timedelta(seconds=1):
            reasons["raw_prediction_late"] += 1
            continue
        record, reason = _static_record(raw_row, snapshot, market)
        if reason:
            reasons[reason] += 1
            continue
        record.update({
            "archive_history_complete": bool(market.archive_history_complete),
            "archive_freshness_window_available": bool(market.archive_freshness_window_available),
            "archive_window_hours_expected_index": _as_int(market.archive_window_hours_expected),
            "archive_window_hours_source_available_index": _as_int(market.archive_window_hours_source_available),
            "archive_window_hours_processed_index": _as_int(market.archive_window_hours_processed),
            "archive_window_missing_source_hours_index": _as_int(market.archive_window_missing_source_hours),
            "archive_window_unprocessed_hours_index": _as_int(market.archive_window_unprocessed_hours),
        })
        records.append(record)
        rows.append({
            "condition_id": record["condition_id"],
            "market_start_utc": record["market_start_utc"],
            "up_token_id": record["up_token_id"],
            "down_token_id": record["down_token_id"],
        })
    static = pd.DataFrame.from_records(records)
    static.sort_values("market_start_utc", kind="stable", inplace=True, ignore_index=True)
    if static.condition_id.duplicated().any() or static.market_start_utc.duplicated().any():
        raise RuntimeError("static candidate calendar has duplicate markets")
    study.assert_development_starts(static.market_start_utc, context="static feature table")
    audit = {
        "development_market_index_rows": int(len(market_index)),
        "raw_oof_rows_in_official_market_date_range": int(len(raw)),
        "raw_oof_market_rows_joined": int(len(raw_map)),
        "t60_development_snapshot_rows": int(len(t60)),
        "static_complete_rows": int(len(static)),
        "static_exclusion_reasons": dict(sorted(reasons.items())),
        "official_market_first_start_utc": earliest.isoformat(),
        "official_market_last_start_utc": market_index.market_start_utc.max().isoformat(),
        "raw_oof_fold_counts_joined": raw.loc[raw.market_start_utc.isin(market_index.market_start_utc), "fold_id"].value_counts().sort_index().to_dict(),
        "raw_oof_prediction_available_at_le_decision_plus_one_second": True,
        "holdout_rows_materialized": 0,
    }
    _write_parquet(static, STATIC_PATH)
    _write_json(OUTPUT_DIR / "development_market_inputs.json", {
        "rows": rows,
        "market_count": len(rows),
        "first_start_utc": static.market_start_utc.min().isoformat() if len(static) else None,
        "last_start_utc": static.market_start_utc.max().isoformat() if len(static) else None,
    })
    return static, market_index, t60, audit


def _source_manifest_audit(static: pd.DataFrame, market_index: pd.DataFrame) -> dict:
    for path in (MARKET_SOURCE, RAW_SOURCE, T60_SOURCE, T59_SOURCE, DEPENDENCY_PATH, REPLAY_MANIFEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    base_manifest = json.loads((ROOT / "reports/btc_preopen/btc_market_microstructure_design_20261019_manifest.json").read_text(encoding="utf-8"))
    microstructure_manifest = json.loads((ROOT / "reports/btc_preopen/btc_market_microstructure_development_20261019_manifest.json").read_text(encoding="utf-8"))
    raw_record = base_manifest["inputs"]["raw_base70_oof"]
    t60_record = base_manifest["inputs"]["t60_snapshot"]
    micro_inputs = microstructure_manifest["inputs"]
    raw_hash, t60_hash = _sha256(RAW_SOURCE), _sha256(T60_SOURCE)
    if raw_hash != raw_record["sha256"] or T60_SOURCE.stat().st_size != t60_record["bytes"] or t60_hash != t60_record["sha256"]:
        raise RuntimeError("RAW BASE70 or T-60 input failed its existing manifest hash")
    archive_manifest_path = EVENT_DIR.parent / "archive_manifest.json"
    for name, path in (("official_market_index_path", MARKET_SOURCE), ("t59_snapshots_path", T59_SOURCE),
                       ("replay_dependencies_path", DEPENDENCY_PATH),
                       ("source_archive_manifest_path", archive_manifest_path)):
        expected = micro_inputs[name]
        if path.stat().st_size != int(expected["bytes"]) or _sha256(path) != expected["sha256"]:
            raise RuntimeError(f"{name} failed its existing microstructure development manifest hash")
    dependency = json.loads(DEPENDENCY_PATH.read_text(encoding="utf-8"))
    replay_manifest = json.loads(REPLAY_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not replay_manifest.get("replay_dependencies_match_current"):
        raise RuntimeError("existing replay dependency manifest is not current")
    if not replay_manifest.get("snapshot_artifact_integrity_verified"):
        raise RuntimeError("existing T-59 replay snapshot artifact integrity is not verified")
    dep_meta = {item["path"]: item for item in dependency["partition_metadata"]}
    dep_hash = {item["path"]: item["content_sha256"] for item in dependency["partitions"]}
    earliest_hour = (pd.Timestamp(static.market_start_utc.min()).floor("h") - pd.Timedelta(hours=25)).strftime("%Y-%m-%dT%H.parquet")
    expected_start = pd.Timestamp(earliest_hour[:-8].replace("T", " "), tz="UTC")
    files = []
    for path in sorted(EVENT_DIR.glob("*.parquet")):
        hour = pd.Timestamp(path.stem.replace("T", " "), tz="UTC")
        if hour < expected_start or hour >= DEVELOPMENT_CUTOFF:
            continue
        name = path.name
        meta = dep_meta.get(name)
        if meta is None or name not in dep_hash:
            raise RuntimeError(f"event part missing from verified replay dependency manifest: {name}")
        stat = path.stat()
        if int(meta["size_bytes"]) != stat.st_size or int(meta["mtime_ns"]) != stat.st_mtime_ns:
            raise RuntimeError(f"event part metadata changed since replay identity was verified: {name}")
        files.append(path)
    if not files:
        raise RuntimeError("no local event parts cover the development window")
    source_list = [{"path": path.name, "size_bytes": path.stat().st_size, "sha256": dep_hash[path.name]} for path in files]
    return {
        "raw_base70_oof": {"path": RAW_SOURCE.relative_to(ROOT).as_posix(), "bytes": RAW_SOURCE.stat().st_size, "sha256": raw_hash},
        "market_index": {"path": MARKET_SOURCE.relative_to(ROOT).as_posix(), "bytes": MARKET_SOURCE.stat().st_size, "sha256": _sha256(MARKET_SOURCE)},
        "t60_snapshot": {"path": T60_SOURCE.relative_to(ROOT).as_posix(), "bytes": T60_SOURCE.stat().st_size, "sha256": t60_hash},
        "t59_ladders": {"path": T59_SOURCE.relative_to(ROOT).as_posix(), "bytes": T59_SOURCE.stat().st_size, "sha256": _sha256(T59_SOURCE)},
        "market_index_filter": "market_start_utc < 2026-07-01T00:00:00Z applied in Arrow scanner before materializing development rows/labels",
        "event_parts": {
            "dependency_sha256": dependency["dependency_sha256"],
            "archive_source_manifest_sha256": dependency["archive_source_manifest_sha256"],
            "verified_current_replay_identity": replay_manifest["current_archive_content_fingerprint_sha256"],
            "files_in_development_scan": len(files),
            "bytes_in_development_scan": int(sum(item["size_bytes"] for item in source_list)),
            "ordered_hashes_sha256": _canonical_sha256(source_list),
            "individual_part_hashes_from_current_manifest": source_list,
            "metadata_rechecked": True,
        },
        "base70_origin": {
            "conditional_oof": True,
            "source_report": "reports/btc_preopen/btc_preopen_taker_flow_20261018.md",
            "source_run_manifest": "data/analysis/polymarket/BTC/btc_preopen_taker_flow_20261018/run_manifest.json",
            "official_market_index_first_start_utc": market_index.market_start_utc.min().isoformat(),
            "official_market_index_last_start_utc": market_index.market_start_utc.max().isoformat(),
            "static_complete_first_start_utc": static.market_start_utc.min().isoformat(),
            "static_complete_last_start_utc": static.market_start_utc.max().isoformat(),
            "earlier_official_market_prediction_rows": 0,
            "holdout_evaluation": False,
            "microstructure_development_manifest_path": "reports/btc_preopen/btc_market_microstructure_development_20261019_manifest.json",
            "microstructure_development_manifest_sha256": _sha256(ROOT / "reports/btc_preopen/btc_market_microstructure_development_20261019_manifest.json"),
            "archive_source_manifest_path": archive_manifest_path.relative_to(ROOT).as_posix(),
            "archive_source_manifest_sha256": _sha256(archive_manifest_path),
            "market_index_t59_and_replay_dependencies_match_existing_manifest_hashes": True,
        },
    }


def _working_set_bytes() -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize) if ok else None
    except Exception:
        return None


def _endpoint_state(state: dict, market: dict, endpoint_ns: int, runtime: dict) -> dict:
    result = {
        "condition_id": market["condition_id"],
        "market_start_utc": market["market_start_utc"],
        "endpoint_time_utc": pd.Timestamp(endpoint_ns, unit="ns", tz="UTC"),
        "market_event_count": int(state["event_count"]),
        "market_book_snapshot_count": int(state["book_count"]),
        "ambiguous_event_count": int(runtime["ambiguous_event_count"]),
        "book_reset_count": int(runtime["book_reset_count"]),
        "out_of_order_price_change_count": int(state["out_of_order_price_changes"]),
        "out_of_order_book_count": int(state["out_of_order_book_events"]),
        "cross_side_late_price_change_count": int(state["cross_side_late_price_changes"]),
        "cross_side_late_ask_count": int(state["cross_side_late_asks"]),
        "equal_source_timestamp_price_change_count": int(state["equal_source_timestamp_price_changes"]),
        "max_inter_event_gap_seconds": state["max_inter_event_gap_seconds"],
        "no_future_source_event": bool(
            all(token.get("max_source_ns_seen") is None or token["max_source_ns_seen"] <= endpoint_ns
                for token in state["tokens"].values())
            and (state.get("fee_source_ns") is None or state["fee_source_ns"] <= endpoint_ns)
        ),
    }
    for side in ("up", "down"):
        token_id = market[f"{side}_token_id"]
        token = state["tokens"].get(token_id)
        if not token or not token.get("initialized"):
            result.update({
                f"{side}_initialized": False,
                f"{side}_bid": None,
                f"{side}_ask": None,
                f"{side}_ask_levels": [],
                f"{side}_ask_ambiguous": True,
                f"{side}_first_book_receive_ns": runtime["first_book_receive_ns"].get(token_id),
                f"{side}_latest_book_receive_ns": None,
            })
            continue
        bids = {float(price): float(size) for price, size in token["bids"].items() if float(size) > 0.0}
        asks = {float(price): float(size) for price, size in token["asks"].items() if float(size) > 0.0}
        bid = max(bids) if bids else None
        ask = min(asks) if asks else None
        result.update({
            f"{side}_initialized": True,
            f"{side}_bid": bid,
            f"{side}_ask": ask,
            f"{side}_ask_levels": sorted((round(float(price), 8), float(size)) for price, size in asks.items()),
            f"{side}_ask_notional_band": study.ask_band_notional(asks, ask) if ask is not None else 0.0,
            f"{side}_ask_ambiguous": bool(token["ask_order_ambiguous"]),
            f"{side}_first_book_receive_ns": runtime["first_book_receive_ns"].get(token_id),
            f"{side}_latest_book_receive_ns": token.get("book_receive_ns"),
            f"{side}_max_source_ns_seen": token.get("max_source_ns_seen"),
        })
    result["p_market"] = _market_probability(
        result.get("up_bid"), result.get("up_ask"), result.get("down_bid"), result.get("down_ask"),
    )
    result["quote_valid"] = bool(
        result.get("up_initialized") and result.get("down_initialized")
        and result.get("up_bid") is not None and result.get("up_ask") is not None
        and result.get("down_bid") is not None and result.get("down_ask") is not None
        and result["up_bid"] <= result["up_ask"] and result["down_bid"] <= result["down_ask"]
        and not result["up_ask_ambiguous"] and not result["down_ask_ambiguous"]
        and result["no_future_source_event"]
    )
    result["spread_up"] = (
        result["up_ask"] - result["up_bid"]
        if result.get("up_ask") is not None and result.get("up_bid") is not None else None
    )
    result["spread_down"] = (
        result["down_ask"] - result["down_bid"]
        if result.get("down_ask") is not None and result.get("down_bid") is not None else None
    )
    return result


def _start_depletion_tracker(endpoint: dict, horizon: int, start_ns: int) -> dict:
    tracker = {"start_ns": int(start_ns), "horizon_seconds": int(horizon), "sides": {}}
    for side in ("up", "down"):
        ask = endpoint.get(f"{side}_ask")
        notional = endpoint.get(f"{side}_ask_notional_band", 0.0)
        tracker["sides"][side] = {
            "band_low": ask,
            "band_high": min(1.0, ask + 0.02) if ask is not None else None,
            "starting_notional": float(notional or 0.0),
            "removed_notional": 0.0,
        }
    return tracker


def _dynamic_record(market: dict, runtime: dict) -> tuple[dict | None, str | None, list[dict]]:
    endpoints = runtime["endpoints"]
    required = ["tminus120", "tminus75", "tminus65", "tminus60"]
    if any(name not in endpoints for name in required):
        return None, "endpoint_state_missing", []
    oldest = endpoints["tminus120"]
    if not all(study.book_initialization_precedes(oldest.get(f"{side}_first_book_receive_ns"),
                                                  int(pd.Timestamp(market["market_start_utc"]).value) - 120_000_000_000)
               for side in ("up", "down")):
        return None, "book_not_initialized_before_oldest_endpoint", [endpoints[k] for k in required]
    for horizon, baseline_name in ((60, "tminus120"), (15, "tminus75"), (5, "tminus65")):
        baseline, decision = endpoints[baseline_name], endpoints["tminus60"]
        tracker = runtime["depletion_trackers"].get(horizon)
        if not baseline.get("quote_valid") or not decision.get("quote_valid") or tracker is None:
            return None, f"invalid_native_quote_window_{horizon}s", [endpoints[k] for k in required]
        if (decision["ambiguous_event_count"] > baseline["ambiguous_event_count"]
                or decision["out_of_order_price_change_count"] > baseline["out_of_order_price_change_count"]
                or decision["out_of_order_book_count"] > baseline["out_of_order_book_count"]
                or decision["cross_side_late_price_change_count"] > baseline["cross_side_late_price_change_count"]):
            return None, f"unresolved_order_or_history_{horizon}s", [endpoints[k] for k in required]
        if decision["book_reset_count"] > baseline["book_reset_count"]:
            return None, f"book_reset_inside_depletion_window_{horizon}s", [endpoints[k] for k in required]
        for side in ("up", "down"):
            start_notional = tracker["sides"][side]["starting_notional"]
            if start_notional <= 0.0:
                return None, f"nonpositive_start_depth_{horizon}s_{side}", [endpoints[k] for k in required]
    result = {
        "condition_id": market["condition_id"],
        "market_start_utc": market["market_start_utc"],
        "dynamic_valid": True,
        "book_reset_count_5s": endpoints["tminus60"]["book_reset_count"] - endpoints["tminus65"]["book_reset_count"],
        "book_reset_count_15s": endpoints["tminus60"]["book_reset_count"] - endpoints["tminus75"]["book_reset_count"],
        "book_reset_count_60s": endpoints["tminus60"]["book_reset_count"] - endpoints["tminus120"]["book_reset_count"],
    }
    end = endpoints["tminus60"]
    for horizon, baseline_name in ((5, "tminus65"), (15, "tminus75"), (60, "tminus120")):
        baseline = endpoints[baseline_name]
        result[f"p_market_change_{horizon}s"] = end["p_market"] - baseline["p_market"]
        result[f"spread_up_change_{horizon}s"] = end["spread_up"] - baseline["spread_up"]
        result[f"spread_down_change_{horizon}s"] = end["spread_down"] - baseline["spread_down"]
        for side in ("up", "down"):
            result[f"log1p_ask_notional_{side}_change_{horizon}s"] = (
                math.log1p(end[f"{side}_ask_notional_band"])
                - math.log1p(baseline[f"{side}_ask_notional_band"])
            )
            tracker = runtime["depletion_trackers"][horizon]["sides"][side]
            result[f"ask_depletion_relative_{side}_{horizon}s"] = (
                tracker["removed_notional"] / (tracker["starting_notional"] + 1e-12)
            )
    if not np.isfinite(np.asarray([result[name] for name in DYNAMIC_FEATURES], dtype=np.float64)).all():
        return None, "nonfinite_dynamic_feature", [endpoints[k] for k in required]
    endpoint_rows = []
    for name in required:
        row = dict(endpoints[name])
        row["endpoint_name"] = name
        endpoint_rows.append(row)
    return result, None, endpoint_rows


def _worker_identity(shard_id: int, markets: list[dict], parts: list[dict], fingerprint: str) -> dict:
    return {
        "experiment_id": CONFIG["experiment_id"],
        "shard_id": int(shard_id),
        "condition_ids_sha256": _canonical_sha256(sorted(item["condition_id"] for item in markets)),
        "source_parts_fingerprint_sha256": fingerprint,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "protocol_sha256": _sha256(CONFIG_PATH),
        "feature_definition_sha256": _sha256(ROOT / "features/btc_market_dynamics_action_value.py"),
        "parts_count": len(parts),
        "first_part": parts[0]["path"] if parts else None,
        "last_part": parts[-1]["path"] if parts else None,
        "max_market_start_exclusive_utc": DEVELOPMENT_CUTOFF.isoformat(),
    }


def _write_worker_status(path: Path, payload: dict) -> None:
    _write_json(path, payload)


def _process_market_event_group(
    state: dict,
    runtime: dict,
    event_group: pd.DataFrame,
    received_ns: int,
    up_token_id: str,
    down_token_id: str,
) -> None:
    rows = list(event_group.itertuples(index=False))
    conflicts, conflict_count = replay._same_receive_group_conflicts(rows)
    relevant_ids = {str(up_token_id), str(down_token_id)}
    if any(token_id in relevant_ids for token_id, _ in conflicts):
        runtime["ambiguous_event_count"] += 1
    expected = {}
    for row in rows:
        token_id = str(row.asset_id)
        event_type = str(row.event_type)
        token = replay._token_state(state, token_id)
        if event_type == "book":
            if not token["initialized"] and token_id not in runtime["first_book_receive_ns"]:
                runtime["first_book_receive_ns"][token_id] = int(received_ns)
            before_book_count = int(state["book_count"])
            before_initialized = bool(token["initialized"])
            before_bids = dict(token["bids"])
            before_asks = dict(token["asks"])
        before_ambiguous = bool(token["ask_order_ambiguous"])
        before_size = None
        price = _number(getattr(row, "price", None))
        side = str(getattr(row, "side", "")).upper()
        if event_type == "price_change" and side == "SELL" and price is not None and token["initialized"]:
            before_size = token["asks"].get(round(price, 8))
        replay._update_event(state, row, int(received_ns), expected)
        after_token = state["tokens"].get(token_id)
        if event_type == "book" and after_token is not None:
            accepted = int(state["book_count"]) > before_book_count
            changed_full_book = (
                not before_initialized
                or before_bids != after_token["bids"]
                or before_asks != after_token["asks"]
            )
            if accepted and changed_full_book:
                runtime["book_reset_count"] += 1
        if after_token is not None and not before_ambiguous and bool(after_token["ask_order_ambiguous"]):
            if token_id in relevant_ids:
                runtime["ambiguous_event_count"] += 1
        if event_type == "price_change" and side == "SELL" and price is not None and after_token is not None:
            after_size = after_token["asks"].get(round(price, 8))
            new_size = 0.0 if after_size is None else float(after_size)
            for horizon, tracker in runtime["depletion_trackers"].items():
                if not study.in_open_closed_window(received_ns, tracker["start_ns"], runtime["t60_ns"]):
                    continue
                for outcome, token_value in (("up", up_token_id), ("down", down_token_id)):
                    band = tracker["sides"][outcome]
                    inside = (
                        token_id == token_value
                        and band["band_low"] is not None
                        and float(band["band_low"]) <= round(price, 8) <= float(band["band_high"])
                    )
                    band["removed_notional"] += study.level_depletion_notional(
                        before_size, new_size, round(price, 8),
                        event_type=event_type, side=side, native=token_id in relevant_ids,
                        in_fixed_band=inside,
                    )
    for token_id, side in conflicts:
        token = replay._token_state(state, token_id)
        token["bid_order_ambiguous" if side == "BUY" else "ask_order_ambiguous"] = True
    state["equal_source_timestamp_price_changes"] += int(conflict_count)


def _event_replay_shard(shard_id: int, markets: list[dict], parts: list[dict], source_fingerprint: str) -> dict:
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    shard_dir = CHECKPOINT_DIR / f"worker_{shard_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    status_path = shard_dir / "progress.json"
    checkpoint_path = shard_dir / "checkpoint.pkl"
    dynamic_path = shard_dir / "dynamic.parquet"
    endpoints_path = shard_dir / "endpoints.parquet"
    identity = _worker_identity(shard_id, markets, parts, source_fingerprint)
    market_by_id = {item["condition_id"]: item for item in markets}
    id_array_values = [key.encode("ascii") for key in market_by_id]
    endpoint_offsets = {
        "tminus120": -120,
        "tminus75": -75,
        "tminus65": -65,
        "tminus60": -60,
    }
    schedule = []
    for market in markets:
        start_ns = int(pd.Timestamp(market["market_start_utc"]).value)
        market["t60_ns"] = start_ns - 60_000_000_000
        for name, seconds in endpoint_offsets.items():
            schedule.append((start_ns + seconds * 1_000_000_000, market["condition_id"], name))
    schedule.sort(key=lambda item: (item[0], item[1], item[2]))
    schedule_times = [item[0] for item in schedule]
    schedule_by_position = schedule
    states = defaultdict(replay._new_state)
    runtime_by_id = {
        item["condition_id"]: {
            "first_book_receive_ns": {}, "endpoints": {}, "depletion_trackers": {},
            "ambiguous_event_count": 0, "book_reset_count": 0, "t60_ns": item["t60_ns"],
        }
        for item in markets
    }
    dynamic_rows, endpoint_rows = [], []
    schedule_position = 0
    start_position = 0
    total_events = 0
    market_event_groups = 0
    selected_rows = 0
    last_part = None
    started = time.perf_counter()
    if checkpoint_path.is_file():
        with checkpoint_path.open("rb") as handle:
            saved = pickle.load(handle)
        if saved.get("identity") != identity:
            raise RuntimeError(f"worker {shard_id} checkpoint identity mismatch")
        if saved.get("status") == "completed":
            if not dynamic_path.is_file() or not endpoints_path.is_file():
                raise RuntimeError(f"worker {shard_id} completed checkpoint has missing outputs")
            return {"shard_id": shard_id, "status": "reused", "markets": len(markets),
                    "dynamic_path": str(dynamic_path), "endpoints_path": str(endpoints_path)}
        states = defaultdict(replay._new_state, saved["states"])
        runtime_by_id = saved["runtime_by_id"]
        dynamic_rows = saved["dynamic_rows"]
        endpoint_rows = saved["endpoint_rows"]
        schedule_position = int(saved["schedule_position"])
        start_position = int(saved["last_part_position"]) + 1
        total_events = int(saved["total_events"])
        market_event_groups = int(saved["market_event_groups"])
        selected_rows = int(saved["selected_rows"])
        last_part = saved.get("last_part")

    def emit_before(cutoff_ns: int, inclusive: bool = False) -> None:
        nonlocal schedule_position
        while schedule_position < len(schedule_by_position):
            at_ns, condition_id, endpoint_name = schedule_by_position[schedule_position]
            if at_ns > cutoff_ns or (at_ns == cutoff_ns and not inclusive):
                break
            market = market_by_id[condition_id]
            runtime = runtime_by_id[condition_id]
            state = states[condition_id]
            endpoint = _endpoint_state(state, market, at_ns, runtime)
            endpoint["endpoint_name"] = endpoint_name
            runtime["endpoints"][endpoint_name] = endpoint
            if endpoint_name == "tminus120":
                runtime["depletion_trackers"][60] = _start_depletion_tracker(endpoint, 60, at_ns)
            elif endpoint_name == "tminus75":
                runtime["depletion_trackers"][15] = _start_depletion_tracker(endpoint, 15, at_ns)
            elif endpoint_name == "tminus65":
                runtime["depletion_trackers"][5] = _start_depletion_tracker(endpoint, 5, at_ns)
            elif endpoint_name == "tminus60":
                dynamic, reason, saved_endpoints = _dynamic_record(market, runtime)
                if dynamic is None:
                    dynamic_rows.append({
                        "condition_id": condition_id, "market_start_utc": market["market_start_utc"],
                        "dynamic_valid": False, "dynamic_reason": reason,
                    })
                    endpoint_rows.extend(saved_endpoints)
                else:
                    dynamic_rows.append(dynamic)
                    endpoint_rows.extend(saved_endpoints)
                state_counts = {
                    "replayed_t60_event_count": endpoint["market_event_count"],
                    "expected_t60_event_count": market["expected_t60_event_count"],
                    "replayed_t60_book_count": endpoint["market_book_snapshot_count"],
                    "expected_t60_book_count": market["expected_t60_book_count"],
                }
                for record in reversed(dynamic_rows):
                    if record.get("condition_id") == condition_id:
                        record.update(state_counts)
                        break
                states.pop(condition_id, None)
                runtime_by_id.pop(condition_id, None)
            schedule_position += 1

    def save_checkpoint(part_position: int, part_name: str, status: str = "interrupted") -> None:
        payload = {
            "identity": identity, "status": status,
            "last_part_position": int(part_position), "last_part": part_name,
            "schedule_position": int(schedule_position), "states": dict(states),
            "runtime_by_id": runtime_by_id, "dynamic_rows": dynamic_rows,
            "endpoint_rows": endpoint_rows, "total_events": int(total_events),
            "market_event_groups": int(market_event_groups), "selected_rows": int(selected_rows),
            "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        _atomic_pickle(checkpoint_path, payload)
        elapsed = max(time.perf_counter() - started, 1e-9)
        _write_worker_status(status_path, {
            "shard_id": shard_id, "status": "running", "completed_parts": part_position + 1,
            "total_parts": len(parts), "completed_markets": len(dynamic_rows),
            "event_rows_selected": selected_rows, "event_groups": market_event_groups,
            "rows_per_second": selected_rows / elapsed, "elapsed_seconds": elapsed,
            "working_set_bytes": _working_set_bytes(), "last_part": part_name,
        })

    try:
        for part_position, part in enumerate(parts):
            if part_position < start_position:
                continue
            path = ROOT / part["path"]
            dataset = ds.dataset(str(path), format="parquet")
            value_set = pa.array(id_array_values, type=dataset.schema.field("market").type)
            predicate = ds.field("market").isin(value_set)
            predicate = predicate & (ds.field("timestamp_received") < DEVELOPMENT_CUTOFF.to_pydatetime())
            table = dataset.to_table(columns=EVENT_COLUMNS, filter=predicate)
            if table.num_rows:
                frame = table.to_pandas()
                frame["market"] = frame["market"].map(lambda value: value.decode("ascii") if isinstance(value, bytes) else str(value).lower())
                frame["_received_ns"] = replay._timestamp_ns(frame["timestamp_received"])
                frame["_source_ns"] = replay._timestamp_ns(frame["timestamp"])
                limits = frame["market"].map(lambda cid: market_by_id[cid]["t60_ns"]).to_numpy(dtype=np.int64, copy=False)
                receive_ns = frame["_received_ns"].to_numpy(dtype=np.int64, copy=False)
                source_ns = frame["_source_ns"].to_numpy(dtype=np.int64, copy=False)
                valid_receive = receive_ns > np.iinfo(np.int64).min
                bad_received = frame.loc[~valid_receive, "market"].unique()
                for cid in bad_received:
                    runtime_by_id[cid]["ambiguous_event_count"] += 1
                valid_source = source_ns > np.iinfo(np.int64).min
                bad_source = frame.loc[valid_receive & ~valid_source, "market"].unique()
                for cid in bad_source:
                    runtime_by_id[cid]["ambiguous_event_count"] += 1
                frame = frame.loc[valid_receive & valid_source & (receive_ns <= limits)]
                if not frame.empty:
                    frame.sort_values(["_received_ns", "_source_ns", "market", "asset_id", "event_type"], kind="stable", inplace=True)
                    selected_rows += int(len(frame))
                    for received_ns, receive_group in frame.groupby("_received_ns", sort=False):
                        received_ns = int(received_ns)
                        emit_before(received_ns, inclusive=False)
                        for condition_id, event_group in receive_group.groupby("market", sort=False):
                            market = market_by_id[condition_id]
                            runtime = runtime_by_id.get(condition_id)
                            if runtime is None:
                                continue
                            _process_market_event_group(
                                states[condition_id], runtime, event_group, received_ns,
                                market["up_token_id"], market["down_token_id"],
                            )
                            market_event_groups += 1
                        emit_before(received_ns, inclusive=True)
                    total_events += int(len(frame))
            hour = pd.Timestamp(path.stem.replace("T", " "), tz="UTC")
            emit_before(int((hour + pd.Timedelta(hours=1)).value), inclusive=False)
            last_part = path.name
            if (part_position + 1) % CHECKPOINT_PARTS == 0 or part_position + 1 == len(parts):
                save_checkpoint(part_position, path.name)
        if schedule_position < len(schedule_by_position):
            emit_before(schedule_times[-1], inclusive=True)
        dynamic_frame = pd.DataFrame.from_records(dynamic_rows)
        endpoint_frame = pd.DataFrame.from_records(endpoint_rows)
        _write_parquet(dynamic_frame, dynamic_path)
        _write_parquet(endpoint_frame, endpoints_path)
        elapsed = time.perf_counter() - started
        final_payload = {
            "identity": identity, "status": "completed",
            "last_part_position": len(parts) - 1, "last_part": last_part,
            "schedule_position": int(schedule_position), "states": {}, "runtime_by_id": {},
            "dynamic_rows": [], "endpoint_rows": [], "total_events": int(total_events),
            "market_event_groups": int(market_event_groups), "selected_rows": int(selected_rows),
            "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "outputs": {
                "dynamic": {"path": str(dynamic_path), "sha256": _sha256(dynamic_path), "rows": len(dynamic_frame)},
                "endpoints": {"path": str(endpoints_path), "sha256": _sha256(endpoints_path), "rows": len(endpoint_frame)},
            },
        }
        _atomic_pickle(checkpoint_path, final_payload)
        _write_worker_status(status_path, {
            "shard_id": shard_id, "status": "completed", "completed_parts": len(parts),
            "total_parts": len(parts), "completed_markets": len(dynamic_frame),
            "valid_dynamic_markets": int(dynamic_frame.get("dynamic_valid", pd.Series(dtype=bool)).fillna(False).sum()),
            "event_rows_selected": selected_rows, "event_groups": market_event_groups,
            "rows_per_second": selected_rows / max(elapsed, 1e-9), "elapsed_seconds": elapsed,
            "working_set_bytes": _working_set_bytes(), "last_part": last_part,
        })
        return {"shard_id": shard_id, "status": "completed", "markets": len(dynamic_frame),
                "valid_dynamic_markets": int(dynamic_frame.get("dynamic_valid", pd.Series(dtype=bool)).fillna(False).sum()),
                "event_rows": selected_rows, "elapsed_seconds": elapsed,
                "dynamic_path": str(dynamic_path), "endpoints_path": str(endpoints_path)}
    except BaseException:
        if last_part is not None:
            try:
                save_checkpoint(max(start_position, parts.index(next(p for p in parts if p["path"].endswith(last_part)))), last_part, status="failed")
            except Exception:
                pass
        _write_worker_status(status_path, {
            "shard_id": shard_id, "status": "failed", "completed_parts": start_position,
            "total_parts": len(parts), "last_part": last_part, "traceback": traceback.format_exc(),
            "working_set_bytes": _working_set_bytes(),
        })
        raise


def _run_dynamic_replay(static: pd.DataFrame, source_audit: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dep = json.loads(DEPENDENCY_PATH.read_text(encoding="utf-8"))
    base_parts = dep["partition_metadata"]
    expected_start = pd.Timestamp(static.market_start_utc.min()).floor("h") - pd.Timedelta(hours=25)
    parts = []
    known_empty = 0
    for item in base_parts:
        hour = pd.Timestamp(item["path"][:-8].replace("T", " "), tz="UTC")
        if hour < expected_start or hour >= DEVELOPMENT_CUTOFF:
            continue
        if int(item["size_bytes"]) <= 3000:
            known_empty += 1
            continue
        path = EVENT_DIR / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        parts.append({"path": path.relative_to(ROOT).as_posix(), "size_bytes": int(item["size_bytes"])})
    parts.sort(key=lambda item: item["path"])
    if not parts:
        raise RuntimeError("no non-empty development event partitions found")
    source_fingerprint = source_audit["event_parts"]["ordered_hashes_sha256"]
    records = []
    source_coverage_exclusions = []
    covered_part_names = {Path(item["path"]).name for item in parts}
    for row in static.to_dict("records"):
        record = {
            key: row[key] for key in (
                "condition_id", "market_start_utc", "up_token_id", "down_token_id",
                "expected_t60_event_count", "expected_t60_book_count",
            )
        }
        start_ns = int(pd.Timestamp(row["market_start_utc"]).value)
        oldest_ns = start_ns - 120_000_000_000
        t60_ns = start_ns - 60_000_000_000
        required_hours = pd.date_range(
            pd.Timestamp(oldest_ns, unit="ns", tz="UTC").floor("h"),
            pd.Timestamp(t60_ns, unit="ns", tz="UTC").floor("h"),
            freq="h",
        )
        missing_hours = [
            hour.strftime("%Y-%m-%dT%H.parquet")
            for hour in required_hours
            if hour.strftime("%Y-%m-%dT%H.parquet") not in covered_part_names
        ]
        if missing_hours:
            source_coverage_exclusions.append({
                "condition_id": record["condition_id"],
                "market_start_utc": record["market_start_utc"],
                "dynamic_valid": False,
                "dynamic_reason": "source_hour_missing_or_unverified_in_required_window",
                "required_source_hours": missing_hours,
            })
        else:
            records.append(record)
    shards = [records[i::SHARD_COUNT] for i in range(SHARD_COUNT)]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    futures = {}
    results = []
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=SHARD_COUNT, mp_context=ctx) as pool:
        for shard_id, shard_records in enumerate(shards):
            if not shard_records:
                continue
            futures[pool.submit(_event_replay_shard, shard_id, shard_records, parts, source_fingerprint)] = shard_id
        last_signature = None
        while futures:
            done, _ = concurrent.futures.wait(
                list(futures), timeout=30.0, return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                shard_id = futures.pop(future)
                results.append(future.result())
            progress_rows = []
            for shard_id in range(SHARD_COUNT):
                path = CHECKPOINT_DIR / f"worker_{shard_id}" / "progress.json"
                if path.is_file():
                    progress_rows.append(json.loads(path.read_text(encoding="utf-8")))
            signature = tuple((row.get("completed_parts"), row.get("event_rows_selected")) for row in progress_rows)
            if progress_rows and signature != last_signature:
                completed = sum(int(row.get("completed_parts", 0)) for row in progress_rows)
                total = sum(int(row.get("total_parts", len(parts))) for row in progress_rows)
                rows = sum(int(row.get("event_rows_selected", 0)) for row in progress_rows)
                elapsed = max(time.perf_counter() - started, 1e-9)
                rate = rows / elapsed
                estimated_total_rows = int(152_378_189 * (len(records) / max(len(static), 1)))
                eta = max(0.0, (estimated_total_rows - rows) / max(rate, 1.0))
                working_set = sum(int(row.get("working_set_bytes") or 0) for row in progress_rows)
                print(
                    f"[dynamics] parts={completed:,}/{total:,}; selected_events={rows:,}; "
                    f"throughput={rate:,.0f} rows/s; worker_working_set={working_set / 1e9:.2f} GB; "
                    f"ETA~{eta / 60:.1f} min",
                    flush=True,
                )
                last_signature = signature
    dynamic_parts, endpoint_parts = [], []
    for shard_id in range(SHARD_COUNT):
        result_row = next((row for row in results if row["shard_id"] == shard_id), None)
        if result_row is None:
            continue
        dynamic_parts.append(pd.read_parquet(result_row["dynamic_path"]))
        endpoint_parts.append(pd.read_parquet(result_row["endpoints_path"]))
    replayed_dynamic = pd.concat(dynamic_parts, ignore_index=True) if dynamic_parts else pd.DataFrame()
    dynamic = pd.concat(
        [replayed_dynamic, pd.DataFrame.from_records(source_coverage_exclusions)], ignore_index=True,
    )
    endpoints = pd.concat(endpoint_parts, ignore_index=True) if endpoint_parts else pd.DataFrame()
    dynamic.sort_values("market_start_utc", kind="stable", inplace=True, ignore_index=True)
    if not endpoints.empty:
        endpoints.sort_values(["market_start_utc", "endpoint_time_utc"], kind="stable", inplace=True, ignore_index=True)
    else:
        endpoints = pd.DataFrame(columns=["condition_id", "market_start_utc", "endpoint_time_utc", "endpoint_name"])
    study.assert_development_starts(dynamic.market_start_utc, context="dynamic feature table")
    if dynamic.condition_id.duplicated().any() or len(dynamic) != len(static):
        raise RuntimeError("dynamic replay did not emit exactly one row for every static candidate")
    _write_parquet(dynamic, DYNAMIC_PATH)
    _write_parquet(endpoints, ENDPOINT_PATH)
    stats = {
        "status": "completed",
        "candidate_markets": int(len(static)),
        "markets_replayed_after_required_source_hour_check": int(len(records)),
        "source_hour_coverage_excluded_markets": int(len(source_coverage_exclusions)),
        "dynamic_valid_markets": int(dynamic.dynamic_valid.fillna(False).sum()),
        "dynamic_exclusion_reasons": dynamic.loc[~dynamic.dynamic_valid.fillna(False), "dynamic_reason"].value_counts(dropna=False).to_dict(),
        "event_parts_scanned": int(len(parts)),
        "known_empty_event_parts_skipped": int(known_empty),
        "event_rows_selected": int(sum(result.get("event_rows", 0) for result in results)),
        "workers": SHARD_COUNT,
        "elapsed_seconds": time.perf_counter() - started,
        "shards": results,
        "dynamic_path": DYNAMIC_PATH.relative_to(ROOT).as_posix(),
        "endpoint_state_path": ENDPOINT_PATH.relative_to(ROOT).as_posix(),
        "event_rows_after_development_cutoff_materialized": 0,
    }
    return dynamic, endpoints, stats


T59_REQUIRED_COLUMNS = [
    "condition_id", "market_start_utc", "entry_time_utc", "has_full_snapshot", "quote_valid",
    "fee_known", "fee_rate_bps", "fee_collection_mode", "fee_event_ns", "fee_age_seconds",
    "no_future_event_at_entry", "no_future_source_event_at_entry", "archive_source",
    "archive_entry_hour_available", "archive_freshness_window_available",
    "archive_window_hours_expected", "archive_window_hours_source_available",
    "archive_window_hours_processed", "archive_window_missing_source_hours",
    "archive_window_unprocessed_hours", "market_event_count", "market_book_snapshot_count",
    "market_out_of_order_price_change_count", "market_cross_side_late_price_change_count",
    "market_equal_source_timestamp_price_change_count", "bbo_at_entry_ask_mismatches",
    "archive_gap_after_up_book_init", "archive_gap_after_down_book_init", "archive_gap_after_fee_update",
    "up_best_bid", "up_best_ask", "up_best_ask_size_shares", "up_ask_age_seconds",
    "up_quote_source_token_id", "up_quote_complemented", "up_ask_levels", "up_book_init_receive_ns",
    "up_ask_order_ambiguous", "up_bbo_ask_comparable", "up_bbo_ask_mismatch",
    "up_bid_ask_crossed", "up_bid_ask_locked", "up_fill",
    "down_best_bid", "down_best_ask", "down_best_ask_size_shares", "down_ask_age_seconds",
    "down_quote_source_token_id", "down_quote_complemented", "down_ask_levels", "down_book_init_receive_ns",
    "down_ask_order_ambiguous", "down_bbo_ask_comparable", "down_bbo_ask_mismatch",
    "down_bid_ask_crossed", "down_bid_ask_locked", "down_fill",
]


def _read_t59() -> pd.DataFrame:
    frame = _dataset_dev(T59_SOURCE, T59_REQUIRED_COLUMNS).to_pandas()
    frame["condition_id"] = frame.condition_id.astype(str).str.lower()
    frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True)
    frame["entry_time_utc"] = pd.to_datetime(frame.entry_time_utc, utc=True)
    study.assert_development_starts(frame.market_start_utc, context="T-59 execution snapshots")
    if frame.condition_id.duplicated().any():
        raise RuntimeError("T-59 snapshot source must contain one row per development market")
    return frame


def _read_development_labels() -> pd.DataFrame:
    labels = _read_dev_market_index(labels=True)[[
        "condition_id", "market_start_utc", "resolved_at_utc", "target_polymarket_up",
    ]].copy()
    labels["resolved_at_utc"] = pd.to_datetime(labels.resolved_at_utc, utc=True, errors="coerce")
    labels["target_polymarket_up"] = pd.to_numeric(labels.target_polymarket_up, errors="coerce")
    labels["label_release_at_utc"] = pd.concat([
        labels.resolved_at_utc,
        labels.market_start_utc + pd.Timedelta(minutes=5),
    ], axis=1).max(axis=1) + pd.Timedelta(seconds=60)
    labels.loc[labels.target_polymarket_up.isna() | labels.resolved_at_utc.isna(), "label_release_at_utc"] = pd.NaT
    return labels


def _t59_base_reason(snapshot: dict, side: str, token_id: str) -> str:
    if snapshot.get(f"{side}_quote_complemented"):
        return "complement_only_not_executable"
    source_token = snapshot.get(f"{side}_quote_source_token_id")
    if source_token is None or str(source_token) != str(token_id):
        return "no_native_book_snapshot"
    if not _as_bool(snapshot.get("archive_entry_hour_available")):
        return "archive_hour_unavailable"
    if not _as_bool(snapshot.get("archive_freshness_window_available")):
        return "archive_freshness_window_unavailable"
    if not _as_bool(snapshot.get("has_full_snapshot")):
        return "incomplete_native_snapshot"
    if _as_int(snapshot.get("archive_window_missing_source_hours")) > 0:
        return "archive_source_hours_missing"
    if _as_int(snapshot.get("archive_window_unprocessed_hours")) > 0:
        return "archive_source_hours_unprocessed"
    if _as_bool(snapshot.get(f"archive_gap_after_{side}_book_init")):
        return "archive_gap_after_book_initialization"
    if _as_bool(snapshot.get("archive_gap_after_fee_update")):
        return "archive_gap_after_fee_update"
    if not _as_bool(snapshot.get("no_future_event_at_entry")):
        return "future_receive_event_at_entry"
    if not _as_bool(snapshot.get("no_future_source_event_at_entry")):
        return "future_source_timestamp_at_entry"
    if _as_bool(snapshot.get(f"{side}_ask_order_ambiguous")):
        return "ambiguous_equal_source_timestamp_order"
    if _as_bool(snapshot.get(f"{side}_bid_ask_crossed")):
        return "crossed_token_book"
    if (_as_bool(snapshot.get(f"{side}_bbo_ask_comparable"))
            and _as_bool(snapshot.get(f"{side}_bbo_ask_mismatch"))):
        return "native_ask_bbo_mismatch"
    if not _as_bool(snapshot.get("fee_known")):
        return "unknown_historical_fee"
    if snapshot.get("fee_collection_mode") == "maintenance_pause":
        return "exchange_maintenance_pause"
    book_init = snapshot.get(f"{side}_book_init_receive_ns")
    try:
        book_init = None if book_init is None or pd.isna(book_init) else int(book_init)
    except (TypeError, ValueError):
        book_init = None
    if not study.book_initialization_precedes(
        book_init,
        int(pd.Timestamp(snapshot["market_start_utc"]).value) - 59_000_000_000,
    ):
        return "book_initialization_missing_or_late"
    return "eligible"


def _fee_rule(entry_time: pd.Timestamp) -> dict:
    return replay._cached_fee_rule(pd.Timestamp(entry_time).isoformat())


def _price_action(levels: dict, requested_usd: float, entry_time: pd.Timestamp) -> dict:
    rule = _fee_rule(entry_time)
    return replay._walk_asks(
        levels,
        rule["rate"] * 10_000.0,
        rule["collection_mode"],
        gross_order_usd=float(requested_usd),
        entry_time_utc=entry_time,
        allow_partial=True,
    )


def _build_action_targets(
    static: pd.DataFrame, dynamic: pd.DataFrame, t59: pd.DataFrame, labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dynamic_map = dynamic.set_index("condition_id", drop=False)
    t59_map = t59.set_index("condition_id", drop=False)
    labels_map = labels.set_index("condition_id", drop=False)
    common_rows, execution_rows, missing = [], [], defaultdict(int)
    for base in static.to_dict("records"):
        condition_id = base["condition_id"]
        drow = dynamic_map.loc[condition_id]
        if not _as_bool(drow.get("dynamic_valid")):
            missing[f"dynamic:{drow.get('dynamic_reason', 'invalid')}"] += 1
            continue
        if condition_id not in t59_map.index:
            missing["t59_snapshot_missing"] += 1
            continue
        if condition_id not in labels_map.index:
            missing["official_label_row_missing"] += 1
            continue
        market = {**base, **{name: drow[name] for name in DYNAMIC_FEATURES}}
        label = labels_map.loc[condition_id]
        market.update({
            "target_polymarket_up": _number(label.target_polymarket_up),
            "resolved_at_utc": label.resolved_at_utc,
            "label_release_at_utc": label.label_release_at_utc,
            "official_label_known": bool(_number(label.target_polymarket_up) in (0.0, 1.0)
                                          and pd.notna(label.resolved_at_utc)),
        })
        snapshot = t59_map.loc[condition_id].to_dict()
        expected_entry = pd.Timestamp(base["market_start_utc"]) + pd.Timedelta(seconds=ENTRY_OFFSET_SECONDS)
        if pd.Timestamp(snapshot["entry_time_utc"]) != expected_entry:
            raise RuntimeError(f"T-59 execution timestamp misaligned for {condition_id}")
        if pd.Timestamp(snapshot["market_start_utc"]) != pd.Timestamp(base["market_start_utc"]):
            raise RuntimeError(f"T-59 market start mismatch for {condition_id}")
        snapshot["market_start_utc"] = pd.Timestamp(base["market_start_utc"])
        fee_rule = _fee_rule(expected_entry)
        market["t60_trading_allowed"] = fee_rule["collection_mode"] != "maintenance_pause"
        execution = {
            "condition_id": condition_id,
            "market_start_utc": base["market_start_utc"],
            "entry_time_utc": expected_entry,
            "fee_rule_id": fee_rule["rule_id"],
            "fee_rate": fee_rule["rate"],
            "fee_exponent": fee_rule["exponent"],
            "fee_collection_mode": fee_rule["collection_mode"],
            "t59_event_count": _as_int(snapshot.get("market_event_count")),
            "t59_book_count": _as_int(snapshot.get("market_book_snapshot_count")),
            "t59_source": snapshot.get("archive_source"),
        }
        for side in ("up", "down"):
            side_reason = _t59_base_reason(snapshot, side, base[f"{side}_token_id"])
            levels = _levels(snapshot.get(f"{side}_ask_levels"))
            book_init = snapshot.get(f"{side}_book_init_receive_ns")
            try:
                book_init = None if book_init is None or pd.isna(book_init) else int(book_init)
            except (TypeError, ValueError):
                book_init = None
            full_depth_known = (
                side_reason == "eligible"
                and study.book_initialization_precedes(
                    book_init,
                    int(expected_entry.value),
                )
                and bool(snapshot.get("has_full_snapshot"))
            )
            if side_reason == "eligible" and not levels:
                # A complete native book with no asks is a known no-fill, not an unknown execution.
                if full_depth_known:
                    side_reason = "observed_empty_ask_book"
                else:
                    side_reason = "depth_not_proven"
            execution[f"{side}_base_reason"] = side_reason
            execution[f"{side}_depth_known"] = bool(full_depth_known)
            execution[f"{side}_ask_levels"] = sorted(levels.items())
            execution[f"{side}_best_ask_t59"] = _number(snapshot.get(f"{side}_best_ask"))
            execution[f"{side}_ask_age_seconds"] = _number(snapshot.get(f"{side}_ask_age_seconds"))
            for q in ACTION_SIZES:
                key = f"{side}_q{int(q)}"
                if side_reason == "observed_empty_ask_book":
                    fill = None
                    result_fraction, status = 0.0, "observed_no_fill"
                    label_known = bool(full_depth_known and market["official_label_known"])
                    result_fraction = result_fraction if label_known else None
                elif side_reason == "eligible" and full_depth_known:
                    fill = _price_action(levels, q, expected_entry)
                    if float(fill.get("gross_shares") or 0.0) <= 0.0:
                        result_fraction, status = 0.0, "observed_no_fill"
                    else:
                        result_usd, result_status = study.action_return(
                            side=side,
                            outcome_up=market["target_polymarket_up"],
                            net_shares=fill.get("shares"),
                            cash_debit_usd=fill.get("cash_debit_usd"),
                            fill_status="partial_fill" if fill.get("partial_fill") else "full",
                            label_known=market["official_label_known"],
                        )
                        result_fraction = (
                            study.action_return_fraction(result_usd, q)
                            if result_usd is not None else None
                        )
                        status = result_status
                    label_known = result_fraction is not None
                else:
                    fill = None
                    result_fraction, status, label_known = None, side_reason, False
                if fill is None:
                    fill = {"gross_usd": None, "gross_shares": None, "shares": None,
                            "cash_debit_usd": None, "fee_cash_usd": None, "fee_shares": None,
                            "vwap": None, "partial_fill": False, "fill_status": "unknown"}
                else:
                    if status == "observed_no_fill":
                        fill = {**fill, "fill_status": "no_fill"}
                    else:
                        fill = {**fill, "fill_status": "partial_fill" if fill.get("partial_fill") else "full"}
                for field in (
                    "gross_usd", "gross_shares", "shares", "cash_debit_usd", "fee_cash_usd",
                    "fee_shares", "vwap", "partial_fill", "fill_status",
                ):
                    execution[f"{key}_{field}"] = fill.get(field)
                execution[f"{key}_r"] = result_fraction
                execution[f"{key}_label_status"] = status
                execution[f"{key}_label_known"] = bool(label_known)
            # T-60 descriptors are analytic action inputs; T-59 never changes them.
            t60_levels = base[f"{side}_ask_levels_t60"]
            t60_rule = _fee_rule(expected_entry)
            for q in ACTION_SIZES:
                if t60_rule["collection_mode"] == "maintenance_pause":
                    fill = {"gross_usd": 0.0, "gross_shares": 0.0, "shares": 0.0,
                            "cash_debit_usd": 0.0, "fee_cash_usd": 0.0,
                            "vwap": float(base[f"{side}_best_ask_t60"])}
                else:
                    fill = replay._walk_asks(
                        dict(t60_levels), t60_rule["rate"] * 10_000.0, t60_rule["collection_mode"],
                        gross_order_usd=q, entry_time_utc=expected_entry, allow_partial=True,
                    )
                # Static eligibility has already established a complete native T-60 ladder.
                depth_known = True
                descriptor = study.action_descriptor(
                    side=side, requested_usd=q,
                    t60_vwap=fill.get("vwap") if fill.get("vwap") is not None else base[f"{side}_best_ask_t60"],
                    t60_best_ask=base[f"{side}_best_ask_t60"],
                    t60_cash_debit_usd=fill.get("cash_debit_usd") or 0.0,
                    t60_net_shares=fill.get("shares") or 0.0,
                    t60_depth_known=bool(depth_known),
                )
                for name, value in descriptor.items():
                    market[f"{side}_q{int(q)}_{name}"] = value
                market[f"{side}_q{int(q)}_t60_gross_usd"] = fill.get("gross_usd")
                market[f"{side}_q{int(q)}_t60_gross_shares"] = fill.get("gross_shares")
                market[f"{side}_q{int(q)}_t60_fee_cash_usd"] = fill.get("fee_cash_usd")
        execution_rows.append(execution)
        common_rows.append(market)
    common = pd.DataFrame.from_records(common_rows)
    executions = pd.DataFrame.from_records(execution_rows)
    if common.empty:
        common = pd.DataFrame(columns=list(static.columns) + list(DYNAMIC_FEATURES) + [
            "target_polymarket_up", "resolved_at_utc", "label_release_at_utc", "official_label_known",
        ])
    if executions.empty:
        executions = pd.DataFrame(columns=["condition_id", "market_start_utc"])
    if not common.empty:
        common.sort_values("market_start_utc", kind="stable", inplace=True, ignore_index=True)
        study.assert_development_starts(common.market_start_utc, context="common S+H+RAW calendar")
    if not executions.empty:
        executions.sort_values("market_start_utc", kind="stable", inplace=True, ignore_index=True)
    coverage = {
        "common_input_markets": int(len(common)),
        "execution_rows": int(len(executions)),
        "execution_quality_exclusion_reasons": dict(sorted(missing.items())),
        "official_label_known_markets": int(common.get("official_label_known", pd.Series(dtype=bool)).sum()),
        "holdout_rows_materialized": 0,
    }
    return common, executions, coverage


def _validate_t60_reconstruction(static: pd.DataFrame, endpoints: pd.DataFrame) -> dict:
    saved = endpoints.loc[endpoints.endpoint_name.eq("tminus60")].copy()
    joined = static.merge(
        saved, on=["condition_id", "market_start_utc"], how="inner", validate="one_to_one",
        suffixes=("_snapshot", "_replay"),
    )
    valid_ids = set(joined.condition_id)
    good_dynamic = static.loc[static.condition_id.isin(valid_ids)]
    if len(joined) != len(good_dynamic):
        raise RuntimeError("T-60 endpoint reconstruction omitted emitted markets")
    metrics = {"compared_markets": int(len(joined)), "quote_mismatches": 0,
               "event_count_mismatches": 0, "book_count_mismatches": 0,
               "max_abs_price_delta": 0.0}
    deltas = []
    for side in ("up", "down"):
        for kind in ("best_bid", "best_ask"):
            expected = pd.to_numeric(joined[f"{side}_{kind}_t60"], errors="coerce")
            actual = pd.to_numeric(joined[f"{side}_{'bid' if kind == 'best_bid' else 'ask'}"], errors="coerce")
            difference = (expected - actual).abs()
            deltas.extend(difference.dropna().tolist())
            metrics["quote_mismatches"] += int((difference > 1e-6).sum())
    metrics["max_abs_price_delta"] = max(deltas, default=0.0)
    metrics["event_count_mismatches"] = int((
        pd.to_numeric(joined.expected_t60_event_count, errors="coerce")
        != pd.to_numeric(joined.market_event_count, errors="coerce")
    ).sum())
    metrics["book_count_mismatches"] = int((
        pd.to_numeric(joined.expected_t60_book_count, errors="coerce")
        != pd.to_numeric(joined.market_book_snapshot_count, errors="coerce")
    ).sum())
    if (metrics["quote_mismatches"] or metrics["event_count_mismatches"]
            or metrics["book_count_mismatches"]):
        raise RuntimeError(f"T-60 replay failed its saved snapshot comparison: {metrics}")
    return metrics


def _make_folds(common: pd.DataFrame, executions: pd.DataFrame) -> list[dict]:
    if common.empty:
        return []
    first_day = pd.Timestamp(common.market_start_utc.min()).floor("D")
    evaluation_start = first_day + pd.Timedelta(days=int(CONFIG["folds"]["warmup_days"]))
    first_end = evaluation_start + pd.offsets.MonthBegin(1)
    boundaries = []
    start = evaluation_start
    end = first_end
    while start < DEVELOPMENT_CUTOFF:
        boundaries.append((start, min(end, DEVELOPMENT_CUTOFF)))
        start = end
        end = start + pd.offsets.MonthBegin(1)
    exec_by_id = executions.set_index("condition_id", drop=False) if not executions.empty else None
    folds = []
    for fold_number, (start, end) in enumerate(boundaries, start=1):
        evaluation = common.loc[common.market_start_utc.ge(start) & common.market_start_utc.lt(end)]
        first_decision = start + pd.Timedelta(seconds=DECISION_OFFSET_SECONDS)
        training = common.loc[
            common.market_start_utc.lt(start)
            & common.official_label_known.fillna(False)
            & common.label_release_at_utc.lt(first_decision)
        ]
        training_ids = set(training.condition_id)
        train_exec = executions.loc[executions.condition_id.isin(training_ids)] if not executions.empty else executions
        q5_known_both = (
            train_exec.get("up_q5_label_known", pd.Series(dtype=bool)).fillna(False).astype(bool)
            & train_exec.get("down_q5_label_known", pd.Series(dtype=bool)).fillna(False).astype(bool)
            & train_exec.get("up_q5_r", pd.Series(dtype=float)).notna()
            & train_exec.get("down_q5_r", pd.Series(dtype=float)).notna()
        )
        target = pd.to_numeric(training.target_polymarket_up, errors="coerce").dropna()
        folds.append({
            "fold_id": fold_number,
            "evaluation_start_utc": start.isoformat(),
            "evaluation_end_exclusive_utc": end.isoformat(),
            "first_decision_at_utc": first_decision.isoformat(),
            "training_market_count": int(len(training_ids)),
            "released_official_label_count": int(target.shape[0]),
            "released_label_classes": sorted(int(x) for x in target.unique()),
            "training_markets_with_known_q5_both_sides": int(q5_known_both.sum()),
            "evaluation_market_count": int(len(evaluation)),
            "evaluation_observed_days": int(evaluation.market_start_utc.dt.floor("D").nunique()),
            "evaluation_market_ids": evaluation.condition_id.astype(str).tolist(),
        })
    return folds


def _availability_report(
    static: pd.DataFrame, dynamic: pd.DataFrame, common: pd.DataFrame,
    executions: pd.DataFrame, folds: list[dict], source_audit: dict,
) -> dict:
    eval_start = pd.Timestamp(folds[0]["evaluation_start_utc"]) if folds else DEVELOPMENT_CUTOFF
    q5_both = pd.Series(False, index=common.index)
    execution_by_id = executions.set_index("condition_id") if not executions.empty else None
    if execution_by_id is not None:
        q5 = execution_by_id.get("up_q5_label_known", pd.Series(dtype=bool)).fillna(False).astype(bool)
        q5 &= execution_by_id.get("down_q5_label_known", pd.Series(dtype=bool)).fillna(False).astype(bool)
        q5 &= execution_by_id.get("up_q5_r", pd.Series(dtype=float)).notna()
        q5 &= execution_by_id.get("down_q5_r", pd.Series(dtype=float)).notna()
        known_ids = set(execution_by_id.index[q5])
        q5_both = common.condition_id.isin(known_ids) & common.official_label_known.fillna(False)
    evaluation = common.loc[common.market_start_utc.ge(eval_start)]
    evaluation_days = int(evaluation.market_start_utc.dt.floor("D").nunique()) if not evaluation.empty else 0
    nonempty_folds = sum(item["evaluation_market_count"] > 0 for item in folds)
    minimum_input_markets = int(CONFIG["folds"]["minimum_qualifying_input_markets"])
    minimum_days = int(CONFIG["folds"]["minimum_observed_days"])
    minimum_blocks = int(CONFIG["folds"]["minimum_nonempty_evaluation_blocks"])
    fold_minima = []
    for fold in folds:
        fold_minima.append({
            "fold_id": fold["fold_id"],
            "nonempty_evaluation_block": fold["evaluation_market_count"] > 0,
            "released_labels_minimum_met": (
                fold["released_official_label_count"] >= CONFIG["folds"]["minimum_released_training_labels"]
                and len(fold["released_label_classes"]) == 2
            ),
            "known_q5_both_sides_minimum_met": (
                fold["training_markets_with_known_q5_both_sides"]
                >= CONFIG["folds"]["minimum_training_markets_with_known_q5_both_sides"]
            ),
        })
    base_inputs_pass = (
        nonempty_folds >= minimum_blocks
        and len(evaluation) >= minimum_input_markets
        and evaluation_days >= minimum_days
    )
    fold_minima_pass = all(
        item["released_labels_minimum_met"] and item["known_q5_both_sides_minimum_met"]
        for item in fold_minima if item["nonempty_evaluation_block"]
    )
    dynamic_exclusions = dynamic.loc[~dynamic.dynamic_valid.fillna(False), "dynamic_reason"].value_counts().to_dict()
    if not base_inputs_pass:
        status = (
            "blocked_missing_dynamic_history"
            if dynamic_exclusions or len(common) < minimum_input_markets
            else "insufficient_development_coverage"
        )
    elif not fold_minima_pass:
        status = "insufficient_development_coverage"
    else:
        status = "availability_passed"
    daily = common.assign(day=common.market_start_utc.dt.strftime("%Y-%m-%d")).groupby("day").agg(
        common_input_markets=("condition_id", "size"),
        official_labels=("official_label_known", "sum"),
    )
    static_daily = static.assign(day=static.market_start_utc.dt.strftime("%Y-%m-%d")).groupby("day").size()
    dynamic_daily = dynamic.loc[dynamic.dynamic_valid.fillna(False)].assign(
        day=lambda frame: frame.market_start_utc.dt.strftime("%Y-%m-%d")
    ).groupby("day").size()
    q5_daily = common.loc[q5_both].assign(
        day=lambda frame: frame.market_start_utc.dt.strftime("%Y-%m-%d")
    ).groupby("day").size()
    calendar = pd.DataFrame(index=pd.date_range(
        static.market_start_utc.min().floor("D"),
        DEVELOPMENT_CUTOFF - pd.Timedelta(days=1), freq="D", tz="UTC",
    ))
    calendar.index.name = "day_utc"
    calendar["static_complete_markets"] = static_daily
    calendar["dynamic_valid_markets"] = dynamic_daily
    calendar["common_input_markets"] = daily.common_input_markets
    calendar["official_label_markets"] = daily.official_labels
    calendar["known_q5_both_sides_markets"] = q5_daily
    calendar = calendar.fillna(0).astype(int).reset_index()
    return {
        "status": status,
        "development_cutoff_exclusive_utc": DEVELOPMENT_CUTOFF.isoformat(),
        "official_market_index_first_start_utc": source_audit.get("base70_origin", {}).get("official_market_index_first_start_utc"),
        "static_complete_market_count": int(len(static)),
        "dynamic_valid_market_count": int(dynamic.dynamic_valid.fillna(False).sum()),
        "dynamic_exclusion_reasons": dynamic_exclusions,
        "common_input_market_count_all_development": int(len(common)),
        "evaluation_start_utc": eval_start.isoformat(),
        "evaluation_market_count": int(len(evaluation)),
        "evaluation_market_count_known_q5_both_sides": int(q5_both.loc[common.market_start_utc.ge(eval_start)].sum()),
        "evaluation_observed_days": evaluation_days,
        "nonempty_evaluation_blocks": int(nonempty_folds),
        "minimums": {
            "nonempty_blocks_required": minimum_blocks,
            "evaluation_input_markets_required": minimum_input_markets,
            "evaluation_observed_days_required": minimum_days,
            "released_labels_per_fold_required": CONFIG["folds"]["minimum_released_training_labels"],
            "known_q5_both_sides_markets_per_fold_required": CONFIG["folds"]["minimum_training_markets_with_known_q5_both_sides"],
        },
        "folds": [{key: value for key, value in item.items() if key != "evaluation_market_ids"} for item in folds],
        "fold_minima": fold_minima,
        "fold_minima_pass": bool(fold_minima_pass),
        "input_minima_pass": bool(base_inputs_pass),
        "daily_availability_path": "availability_by_day.parquet",
        "daily_availability_rows": calendar.to_dict("records"),
        "source_audit": source_audit,
        "holdout_rows_materialized": 0,
    }, calendar


def _freeze_protocol(availability: dict, folds: list[dict], artifacts: dict, validation: dict) -> dict:
    return {
        "experiment_id": CONFIG["experiment_id"],
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status_at_freeze": availability["status"],
        "development_cutoff_exclusive_utc": DEVELOPMENT_CUTOFF.isoformat(),
        "evaluation_start_utc": availability["evaluation_start_utc"],
        "folds": [{key: value for key, value in item.items() if key != "evaluation_market_ids"} for item in folds],
        "static_features_ordered": list(STATIC_FEATURES),
        "dynamic_features_ordered": list(DYNAMIC_FEATURES),
        "action_descriptor_ordered": list(ACTION_DESCRIPTORS),
        "action_sizes_usd": list(ACTION_SIZES),
        "model": CONFIG["model"],
        "policy": CONFIG["policy"],
        "bootstrap": CONFIG["bootstrap"],
        "promotion_gate": CONFIG["promotion_gate"],
        "source_audit": availability["source_audit"],
        "artifact_hashes": artifacts,
        "required_tests": {
            "status": validation["status"], "tests_run": validation["tests_run"],
            "test_files_sha256": validation["test_files_sha256"],
        },
        "base70_origin_limitation": (
            "RAW BASE70 is conditional OOF after earlier development selection; this experiment does not test that full selection pipeline independently."
        ),
        "prior_exposure_register": {
            "historical_holdout_summary_incidentally_seen_in_status_document": True,
            "holdout_market_rows_or_monthly_tables_read_for_this_experiment": False,
            "holdout_scored_or_used_for_selection": False,
            "exposure_used_for_feature_window_threshold_or_model_selection": False,
        },
        "holdout_evaluation": False,
        "live_actions": False,
        "orders_submitted": False,
    }


def _run_required_tests() -> dict:
    files = [
        "test_btc_market_dynamics_action_value_20261020.py",
        "test_btc_preopen_contract.py",
        "test_btc_preopen_economic_replay.py",
        "test_btc_preopen_t59_reassessment.py",
        "test_polymarket_btc5m_fee_rules.py",
    ]
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    file_hashes = {}
    paths = []
    for filename in files:
        path = ROOT / "tests" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
        file_hashes[path.relative_to(ROOT).as_posix()] = _sha256(path)
    result_path = OUTPUT_DIR / "required_tests.json"
    current_identity = {
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "feature_code_sha256": _sha256(ROOT / "features/btc_market_dynamics_action_value.py"),
        "test_files_sha256": file_hashes,
    }
    if result_path.is_file():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("status") == "passed" and all(saved.get(key) == value for key, value in current_identity.items()):
            return saved
    for path in paths:
        module_name = f"_btc_dynamics_test_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load required test module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite.addTests(loader.loadTestsFromModule(module))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    payload = {
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": int(result.testsRun),
        "failures": [str(item[1]) for item in result.failures],
        "errors": [str(item[1]) for item in result.errors],
        "skipped": [str(item[1]) for item in result.skipped],
        "test_files_sha256": file_hashes,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "feature_code_sha256": _sha256(ROOT / "features/btc_market_dynamics_action_value.py"),
        "output": stream.getvalue(),
        "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write_json(result_path, payload)
    print(f"[dynamics] tests={result.testsRun}; status={payload['status']}", flush=True)
    if not result.wasSuccessful():
        raise RuntimeError(f"required tests failed: {result.failures} {result.errors}")
    return payload


def _model_parameters(objective: str, n_rows: int, seed: int, threads: int) -> dict:
    model_config = CONFIG["model"]
    return {
        "objective": objective,
        "boosting_type": model_config["boosting_type"],
        "learning_rate": model_config["learning_rate"],
        "num_leaves": model_config["num_leaves"],
        "max_depth": model_config["max_depth"],
        "lambda_l1": model_config["lambda_l1"],
        "lambda_l2": model_config["lambda_l2"],
        "n_estimators": model_config["n_estimators"],
        "feature_fraction": model_config["feature_fraction"],
        "bagging_fraction": model_config["bagging_fraction"],
        "bagging_freq": model_config["bagging_freq"],
        "max_bin": model_config["max_bin"],
        "feature_pre_filter": model_config["feature_pre_filter"],
        "min_data_in_leaf": max(50, int(math.ceil(0.02 * n_rows))),
        "random_state": int(seed),
        "data_random_seed": int(seed + 1),
        "feature_fraction_seed": int(seed + 2),
        "bagging_seed": int(seed + 3),
        "deterministic": model_config["deterministic"],
        "force_col_wise": model_config["force_col_wise"],
        "n_jobs": int(threads),
        "verbosity": -1,
    }


def _action_feature_names(base_features: tuple[str, ...]) -> list[str]:
    return list(base_features) + list(ACTION_DESCRIPTORS)


def _action_features_from_market(row: dict, side: str, requested_usd: float) -> dict[str, float]:
    q = float(requested_usd)
    q_int = int(q) if q.is_integer() else None
    prefix = f"{side}_q{q_int}_" if q_int in ACTION_SIZES else None
    if prefix and all(f"{prefix}{name}" in row for name in ACTION_DESCRIPTORS):
        return {name: float(row[f"{prefix}{name}"]) for name in ACTION_DESCRIPTORS}
    entry_time = pd.Timestamp(row["market_start_utc"]) + pd.Timedelta(seconds=ENTRY_OFFSET_SECONDS)
    levels = _levels(row[f"{side}_ask_levels_t60"])
    fill = _price_action(levels, q, entry_time)
    best_ask = float(row[f"{side}_best_ask_t60"])
    vwap = fill.get("vwap") if fill.get("vwap") is not None else best_ask
    descriptor = study.action_descriptor(
        side=side,
        requested_usd=q,
        t60_vwap=vwap,
        t60_best_ask=best_ask,
        t60_cash_debit_usd=float(fill.get("cash_debit_usd") or 0.0),
        t60_net_shares=float(fill.get("shares") or 0.0),
        t60_depth_known=True,
    )
    return descriptor


def _action_matrix_row(row: dict, side: str, requested_usd: float, base_features: tuple[str, ...]) -> list[float]:
    action = _action_features_from_market(row, side, requested_usd)
    return [float(row[name]) for name in base_features] + [float(action[name]) for name in ACTION_DESCRIPTORS]


def _fit_fold(
    common: pd.DataFrame, executions: pd.DataFrame, fold: dict,
    source_identity_sha256: str, threads: int,
) -> tuple[pd.DataFrame, dict]:
    fold_id = int(fold["fold_id"])
    fold_dir = CHECKPOINT_DIR / f"fold_{fold_id:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = fold_dir / "predictions.parquet"
    checkpoint_path = fold_dir / "completed.json"
    model_paths = {name: fold_dir / f"model_{name}.txt" for name in ("A", "B", "C", "D")}
    eval_ids = fold["evaluation_market_ids"]
    start = pd.Timestamp(fold["evaluation_start_utc"])
    train = common.loc[
        common.market_start_utc.lt(start)
        & common.official_label_known.fillna(False)
        & common.label_release_at_utc.lt(pd.Timestamp(fold["first_decision_at_utc"]))
    ].copy()
    evaluation = common.loc[common.condition_id.isin(eval_ids)].copy()
    exec_by_id = executions.set_index("condition_id", drop=False)
    input_identity = {
        "experiment_id": CONFIG["experiment_id"],
        "fold_id": fold_id,
        "fold": {key: value for key, value in fold.items() if key != "evaluation_market_ids"},
        "training_ids_sha256": _canonical_sha256(sorted(train.condition_id.astype(str))),
        "evaluation_ids_sha256": _canonical_sha256(sorted(evaluation.condition_id.astype(str))),
        "source_identity_sha256": source_identity_sha256,
        "config_sha256": _sha256(CONFIG_PATH),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "feature_code_sha256": _sha256(ROOT / "features/btc_market_dynamics_action_value.py"),
        "threads": int(threads),
    }
    if checkpoint_path.is_file() and prediction_path.is_file() and all(path.is_file() for path in model_paths.values()):
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        outputs_match = (
            checkpoint.get("identity") == input_identity
            and checkpoint.get("outputs", {}).get("predictions_sha256") == _sha256(prediction_path)
            and all(checkpoint.get("outputs", {}).get(f"model_{name}_sha256") == _sha256(path)
                    for name, path in model_paths.items())
        )
        if outputs_match:
            predictions = pd.read_parquet(prediction_path)
            return predictions, {
                "fold_id": fold_id, "status": "reused", "checkpoint": checkpoint,
                "model_paths": {name: str(path) for name, path in model_paths.items()},
            }

    common_features = {
        "A": list(STATIC_FEATURES),
        "C": list(STATIC_FEATURES) + list(DYNAMIC_FEATURES),
    }
    train_target = pd.to_numeric(train.target_polymarket_up, errors="coerce").astype(int).to_numpy()
    if set(np.unique(train_target)) != {0, 1}:
        raise RuntimeError(f"fold {fold_id} training labels do not contain both official classes")
    train_summary = {
        "training_markets": int(train.condition_id.nunique()),
        "released_official_labels": int(len(train)),
        "released_label_classes": sorted(int(x) for x in np.unique(train_target)),
        "training_action_rows": 0,
        "training_action_markets": 0,
        "training_action_weight_sum": 0.0,
    }
    estimators = {}
    for index, name in enumerate(("A", "C")):
        x_train = train[common_features[name]].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(x_train).all():
            raise RuntimeError(f"fold {fold_id} {name} has nonfinite training features")
        params = _model_parameters("binary", len(train), 20261020 + fold_id * 100 + index, threads)
        estimator = lgb.LGBMClassifier(**params)
        estimator.fit(x_train, train_target)
        estimators[name] = estimator

    action_rows = []
    action_targets = []
    action_market_ids = []
    for row in train.to_dict("records"):
        execution = exec_by_id.loc[row["condition_id"]]
        for side in ("up", "down"):
            for q in ACTION_SIZES:
                key = f"{side}_q{int(q)}"
                if not _as_bool(execution.get(f"{key}_label_known")):
                    continue
                target = _number(execution.get(f"{key}_r"))
                if target is None:
                    continue
                action_rows.append((row, side, q))
                action_targets.append(target)
                action_market_ids.append(row["condition_id"])
    if not action_rows:
        raise RuntimeError(f"fold {fold_id} has no known action-return training labels")
    for variant, base_features in (("B", tuple(STATIC_FEATURES)),
                                   ("D", tuple(STATIC_FEATURES) + tuple(DYNAMIC_FEATURES))):
        x_train = np.asarray([
            _action_matrix_row(row, side, q, base_features) for row, side, q in action_rows
        ], dtype=np.float32)
        y_train = np.asarray(action_targets, dtype=np.float32)
        weights = study.market_action_weights(action_market_ids)
        action_fold_ids = np.full(len(action_market_ids), fold_id, dtype=np.int32)
        study.assert_single_fold_per_market(action_market_ids, action_fold_ids)
        weight_sums = pd.Series(weights).groupby(pd.Series(action_market_ids)).sum().to_numpy()
        if not np.allclose(weight_sums, 1.0, atol=1e-12):
            raise RuntimeError(f"fold {fold_id} action weights do not sum to one per market")
        train_summary["training_action_rows"] = int(len(action_rows))
        train_summary["training_action_markets"] = int(len(set(action_market_ids)))
        train_summary["training_action_weight_sum"] = float(weights.sum())
        params = _model_parameters(
            "regression_l2", len(action_rows), 20261020 + fold_id * 100 + (2 if variant == "B" else 3), threads,
        )
        estimator = lgb.LGBMRegressor(**params)
        estimator.fit(x_train, y_train, sample_weight=weights)
        estimators[variant] = estimator

    predictions = evaluation[["condition_id", "market_start_utc", "target_polymarket_up", "label_release_at_utc"]].copy()
    predictions["fold_id"] = fold_id
    for name in ("A", "C"):
        x_eval = evaluation[common_features[name]].to_numpy(dtype=np.float32, copy=True)
        probabilities = estimators[name].predict_proba(x_eval, num_threads=threads)[:, 1]
        predictions[f"p_{name}"] = probabilities.astype(np.float64)
    for name, base_features in (("B", tuple(STATIC_FEATURES)),
                                ("D", tuple(STATIC_FEATURES) + tuple(DYNAMIC_FEATURES))):
        for side in ("up", "down"):
            for q in ACTION_SIZES:
                x_eval = np.asarray([
                    _action_matrix_row(row, side, q, base_features)
                    for row in evaluation.to_dict("records")
                ], dtype=np.float32)
                values = estimators[name].predict(x_eval, num_threads=threads)
                predictions[f"pred_r_{name}_{side}_q{int(q)}"] = values.astype(np.float64)
    for name, estimator in estimators.items():
        temp_path = model_paths[name].with_suffix(".txt.tmp")
        estimator.booster_.save_model(str(temp_path))
        temp_path.replace(model_paths[name])
    _write_parquet(predictions, prediction_path)
    checkpoint = {
        "identity": input_identity,
        "status": "completed",
        "training": train_summary,
        "model_params": {
            "A": estimators["A"].get_params(), "B": estimators["B"].get_params(),
            "C": estimators["C"].get_params(), "D": estimators["D"].get_params(),
        },
        "model_paths": {name: path.relative_to(ROOT).as_posix() for name, path in model_paths.items()},
        "outputs": {
            "predictions_sha256": _sha256(prediction_path),
            **{f"model_{name}_sha256": _sha256(path) for name, path in model_paths.items()},
        },
        "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write_json(checkpoint_path, checkpoint)
    return predictions, {
        "fold_id": fold_id, "status": "completed", "checkpoint": checkpoint,
        "model_paths": {name: str(path) for name, path in model_paths.items()},
    }


def _probability_metrics(common: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    joined = common[["condition_id", "target_polymarket_up", "official_label_known", "p_raw", "p_market"]].merge(
        predictions[["condition_id", "p_A", "p_C"]], on="condition_id", how="inner", validate="one_to_one",
    )
    joined = joined.loc[joined.official_label_known.fillna(False)].copy()
    y = joined.target_polymarket_up.astype(int).to_numpy()
    result = {"scored_markets": int(len(joined)), "classes": sorted(int(x) for x in np.unique(y))}
    if not len(joined):
        for model_name in ("RAW", "MARKET", "A", "C"):
            result[model_name] = {"log_loss": None, "brier": None, "auc": None}
        return result
    for model_name, probability_column in (("RAW", "p_raw"), ("MARKET", "p_market"),
                                           ("A", "p_A"), ("C", "p_C")):
        p = joined[probability_column].to_numpy(dtype=np.float64)
        result[model_name] = {
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        }
    return result


def _build_decisions(common: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    pred_map = predictions.set_index("condition_id", drop=False)
    records = []
    probability_models = {
        "RAW": "p_raw", "MARKET": "p_market", "A": "p_A", "C": "p_C",
    }
    models = ("RAW", "MARKET", "A", "B", "C", "D")
    for base in common.to_dict("records"):
        prediction = pred_map.loc[base["condition_id"]].to_dict()
        for model_name in models:
            candidates = []
            if not _as_bool(base.get("t60_trading_allowed", True)):
                chosen = None
                records.append({
                    "condition_id": base["condition_id"],
                    "market_start_utc": base["market_start_utc"],
                    "decision_time_utc": base["decision_time_utc"],
                    "fold_id": int(prediction["fold_id"]),
                    "model": model_name,
                    "decision_probability_up": _number(
                        base["p_raw"] if model_name == "RAW" else base["p_market"] if model_name == "MARKET"
                        else prediction.get(probability_models.get(model_name, ""))
                    ) if model_name in probability_models else None,
                    "chosen_side": None, "requested_usd": 0.0,
                    "predicted_return_fraction": None, "expected_t60_pnl_usd": None,
                    "trade_intent": False, "minimum_return_fraction": float(CONFIG["policy"]["minimum_net_return_fraction"]),
                    "decision_status": "exchange_maintenance_pause",
                })
                continue
            for side in ("up", "down"):
                if model_name in probability_models:
                    probability = _number(
                        base["p_raw"] if model_name == "RAW" else
                        base["p_market"] if model_name == "MARKET" else
                        prediction[probability_models[model_name]]
                    )
                    if probability is None:
                        continue
                    p_side = probability if side == "up" else 1.0 - probability
                    action_prefix = f"{side}_q5_"
                    shares = _number(base.get(f"{action_prefix}t60_predicted_net_shares"))
                    debit = _number(base.get(f"{action_prefix}t60_predicted_cash_debit_usd"))
                    if shares is None or debit is None:
                        continue
                    expected_pnl = p_side * shares - debit
                    expected_return = expected_pnl / 5.0
                else:
                    expected_return = _number(prediction.get(f"pred_r_{model_name}_{side}_q5"))
                    if expected_return is None:
                        continue
                    expected_pnl = expected_return * 5.0
                candidates.append({
                    "side": side, "requested_usd": 5.0,
                    "predicted_return": expected_return,
                    "expected_pnl_usd": expected_pnl,
                    "decision_probability_up": (
                        _number(base["p_raw"] if model_name == "RAW" else
                                base["p_market"] if model_name == "MARKET" else
                                prediction[probability_models[model_name]])
                        if model_name in probability_models else None
                    ),
                })
            chosen = study.choose_action(
                candidates, minimum_return=float(CONFIG["policy"]["minimum_net_return_fraction"]),
            )
            records.append({
                "condition_id": base["condition_id"],
                "market_start_utc": base["market_start_utc"],
                "decision_time_utc": base["decision_time_utc"],
                "fold_id": int(prediction["fold_id"]),
                "model": model_name,
                "decision_probability_up": chosen.get("decision_probability_up") if chosen else (
                    _number(base["p_raw"] if model_name == "RAW" else
                            base["p_market"] if model_name == "MARKET" else
                            prediction.get(probability_models.get(model_name, "")))
                    if model_name in probability_models else None
                ),
                "chosen_side": chosen["side"] if chosen else None,
                "requested_usd": 5.0 if chosen else 0.0,
                "predicted_return_fraction": chosen["predicted_return"] if chosen else None,
                "expected_t60_pnl_usd": chosen["expected_pnl_usd"] if chosen else None,
                "trade_intent": chosen is not None,
                "minimum_return_fraction": float(CONFIG["policy"]["minimum_net_return_fraction"]),
                "decision_status": "action_selected" if chosen is not None else "no_action_over_threshold",
            })
    decisions = pd.DataFrame.from_records(records)
    study.assert_development_starts(decisions.market_start_utc, context="OOF decisions")
    return decisions


def _execution_simulation(execution: dict, side: str, requested_usd: float, stress: float = 0.0) -> dict:
    reason = str(execution.get(f"{side}_base_reason", "missing_execution_data"))
    if reason == "observed_empty_ask_book":
        return {"execution_known": True, "fill_status": "no_fill", "gross_usd": 0.0,
                "gross_shares": 0.0, "shares": 0.0, "cash_debit_usd": 0.0,
                "fee_cash_usd": 0.0, "fee_shares": 0.0, "vwap": None, "reason": reason}
    if reason != "eligible" or not _as_bool(execution.get(f"{side}_depth_known")):
        return {"execution_known": False, "fill_status": "unknown", "reason": reason}
    levels = _levels(execution.get(f"{side}_ask_levels"))
    if stress:
        levels = {
            round(float(price) + float(stress), 8): float(size)
            for price, size in levels.items()
            if float(price) + float(stress) < 1.0
        }
    fill = _price_action(levels, requested_usd, pd.Timestamp(execution["entry_time_utc"]))
    if float(fill.get("gross_shares") or 0.0) <= 0.0:
        return {"execution_known": True, "fill_status": "no_fill", "gross_usd": 0.0,
                "gross_shares": 0.0, "shares": 0.0, "cash_debit_usd": 0.0,
                "fee_cash_usd": 0.0, "fee_shares": 0.0, "vwap": None, "reason": "observed_no_fill"}
    return {
        "execution_known": True,
        "fill_status": "partial_fill" if fill.get("partial_fill") else "full",
        "gross_usd": float(fill["gross_usd"]),
        "gross_shares": float(fill["gross_shares"]),
        "shares": float(fill["shares"]),
        "cash_debit_usd": float(fill["cash_debit_usd"]),
        "fee_cash_usd": float(fill["fee_cash_usd"]),
        "fee_shares": float(fill["fee_shares"]),
        "vwap": float(fill["vwap"]),
        "reason": reason,
    }


def _build_fixed5_ledger(
    decisions: pd.DataFrame, common: pd.DataFrame, executions: pd.DataFrame,
) -> pd.DataFrame:
    common_by_id = common.set_index("condition_id", drop=False)
    execution_by_id = executions.set_index("condition_id", drop=False)
    rows = []
    for decision in decisions.to_dict("records"):
        row = common_by_id.loc[decision["condition_id"]]
        item = {
            **decision,
            "execution_status": "no_intent",
            "fill_status": "no_intent",
            "execution_known": True,
            "official_label_status": "no_intent",
            "settlement_status": "no_position",
            "gross_notional_usd": 0.0,
            "cash_debit_usd": 0.0,
            "fee_cash_usd": 0.0,
            "fee_token_equivalent_usd": 0.0,
            "payout_usd": 0.0,
            "settled_pnl_usd": 0.0,
            "no_fee_pnl_usd": 0.0,
        }
        for stress in CONFIG["policy"]["adverse_price_stress_per_token"]:
            item[f"pnl_stress_{str(stress).replace('.', '_')}"] = 0.0
        if decision["trade_intent"]:
            execution = execution_by_id.loc[decision["condition_id"]].to_dict()
            side = decision["chosen_side"]
            fill = _execution_simulation(execution, side, 5.0)
            item.update({
                "execution_status": fill["reason"],
                "fill_status": fill["fill_status"],
                "execution_known": bool(fill["execution_known"]),
                "t59_entry_time_utc": execution["entry_time_utc"],
                "fee_collection_mode": execution["fee_collection_mode"],
                "gross_notional_usd": fill.get("gross_usd"),
                "cash_debit_usd": fill.get("cash_debit_usd"),
                "fee_cash_usd": fill.get("fee_cash_usd"),
                "fee_token_equivalent_usd": (
                    float(fill.get("fee_shares") or 0.0) * float(fill.get("vwap") or 0.0)
                ),
            })
            official = row.target_polymarket_up
            label_known = bool(row.official_label_known)
            item["official_label_status"] = "known" if label_known else "missing_or_unresolved"
            if not fill["execution_known"]:
                item["settlement_status"] = "execution_unknown"
                item["settled_pnl_usd"] = np.nan
                item["no_fee_pnl_usd"] = np.nan
                for stress in CONFIG["policy"]["adverse_price_stress_per_token"]:
                    item[f"pnl_stress_{str(stress).replace('.', '_')}"] = np.nan
            elif fill["fill_status"] == "no_fill":
                item["settlement_status"] = "observed_no_fill"
                item["payout_usd"] = 0.0
                item["settled_pnl_usd"] = 0.0
                item["no_fee_pnl_usd"] = 0.0
                for stress in CONFIG["policy"]["adverse_price_stress_per_token"]:
                    item[f"pnl_stress_{str(stress).replace('.', '_')}"] = 0.0
            elif not label_known:
                item["settlement_status"] = "unsettled_or_label_unknown"
                item["settled_pnl_usd"] = np.nan
                item["no_fee_pnl_usd"] = np.nan
                for stress in CONFIG["policy"]["adverse_price_stress_per_token"]:
                    item[f"pnl_stress_{str(stress).replace('.', '_')}"] = np.nan
            else:
                item["settlement_status"] = "settled"
                if fill["fill_status"] == "no_fill":
                    pnl = no_fee_pnl = 0.0
                    payout = 0.0
                else:
                    won = (side == "up" and float(official) == 1.0) or (side == "down" and float(official) == 0.0)
                    payout = float(fill["shares"]) if won else 0.0
                    pnl = payout - float(fill["cash_debit_usd"])
                    no_fee_pnl = (float(fill["gross_shares"]) if won else 0.0) - float(fill["gross_usd"])
                item["payout_usd"] = payout
                item["settled_pnl_usd"] = pnl
                item["no_fee_pnl_usd"] = no_fee_pnl
                for stress in CONFIG["policy"]["adverse_price_stress_per_token"]:
                    stressed = _execution_simulation(execution, side, 5.0, stress=float(stress))
                    if not stressed["execution_known"]:
                        stressed_pnl = np.nan
                    elif stressed["fill_status"] == "no_fill":
                        stressed_pnl = 0.0
                    else:
                        stressed_won = (side == "up" and float(official) == 1.0) or (side == "down" and float(official) == 0.0)
                        stressed_payout = float(stressed["shares"]) if stressed_won else 0.0
                        stressed_pnl = stressed_payout - float(stressed["cash_debit_usd"])
                    item[f"pnl_stress_{str(stress).replace('.', '_')}"] = stressed_pnl
            if item["settlement_status"] == "settled":
                if abs(float(item["settled_pnl_usd"]) - (float(item["payout_usd"]) - float(item["cash_debit_usd"]))) > 1e-8:
                    raise RuntimeError("fixed-$5 ledger does not reconcile payout minus cash debit")
                if float(item["settled_pnl_usd"]) < -float(item["cash_debit_usd"]) - 1e-8:
                    raise RuntimeError("settled loss exceeds total cash debit")
        rows.append(item)
    ledger = pd.DataFrame.from_records(rows)
    study.assert_development_starts(ledger.market_start_utc, context="fixed-$5 economic ledger")
    return ledger


def _daily_policy_frame(ledger: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    days = pd.date_range(start.floor("D"), end.floor("D"), freq="D", tz="UTC")
    rows = []
    for model_name in ("RAW", "MARKET", "A", "B", "C", "D"):
        subset = ledger.loc[ledger.model.eq(model_name)].copy()
        subset["day_utc"] = pd.to_datetime(subset.market_start_utc, utc=True).dt.floor("D")
        agg = subset.groupby("day_utc").agg(
            settled_pnl_usd=("settled_pnl_usd", lambda values: float(values.dropna().sum())),
            settled_intents=("settlement_status", lambda values: int(values.isin(["settled", "observed_no_fill"]).sum())),
            fills=("fill_status", lambda values: int(values.isin(["full", "partial_fill"]).sum())),
            unknown_execution_intents=("settlement_status", lambda values: int(values.eq("execution_unknown").sum())),
            unsettled_positions=("settlement_status", lambda values: int(values.eq("unsettled_or_label_unknown").sum())),
            gross_notional_usd=("gross_notional_usd", lambda values: float(pd.to_numeric(values, errors="coerce").fillna(0).sum())),
        )
        for day in days:
            values = agg.loc[day] if day in agg.index else None
            rows.append({
                "model": model_name, "day_utc": day,
                "settled_pnl_usd": float(values.settled_pnl_usd) if values is not None else 0.0,
                "bootstrap_pnl_usd": (
                    np.nan if values is not None and (values.unknown_execution_intents or values.unsettled_positions)
                    else float(values.settled_pnl_usd) if values is not None else 0.0
                ),
                "settled_intents": int(values.settled_intents) if values is not None else 0,
                "fills": int(values.fills) if values is not None else 0,
                "unknown_execution_intents": int(values.unknown_execution_intents) if values is not None else 0,
                "unsettled_positions": int(values.unsettled_positions) if values is not None else 0,
                "gross_notional_usd": float(values.gross_notional_usd) if values is not None else 0.0,
                "day_status": "execution_unknown_or_unsettled" if values is not None and (
                    values.unknown_execution_intents or values.unsettled_positions
                ) else "observed_or_no_trade",
            })
    return pd.DataFrame.from_records(rows)


def _bootstrap_daily(daily: pd.DataFrame) -> dict:
    bootstrap = CONFIG["bootstrap"]
    models = ("RAW", "MARKET", "A", "B", "C", "D")
    pivot_pnl = daily.pivot(index="day_utc", columns="model", values="bootstrap_pnl_usd").reindex(columns=models)
    pivot_turnover = daily.pivot(index="day_utc", columns="model", values="gross_notional_usd").reindex(columns=models)
    pnl = pivot_pnl.to_numpy(dtype=np.float64)
    turnover = pivot_turnover.to_numpy(dtype=np.float64)
    n_days = len(pnl)
    block = int(bootstrap["block_days"])
    reps = int(bootstrap["replications"])
    rng = np.random.default_rng(int(bootstrap["seed"]))
    draws = np.empty((reps, n_days, len(models)), dtype=np.int32)
    blocks_needed = int(math.ceil(n_days / block))
    offsets = np.arange(block, dtype=np.int32)
    for rep in range(reps):
        starts = rng.integers(0, n_days, size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % n_days).reshape(-1)[:n_days]
        draws[rep] = indices
    boot_mean = np.empty((reps, len(models)), dtype=np.float64)
    boot_ratio = np.full((reps, len(models)), np.nan, dtype=np.float64)
    for rep in range(reps):
        idx = draws[rep]
        for model_idx in range(len(models)):
            sample = pnl[idx, model_idx]
            if np.isfinite(sample).all():
                boot_mean[rep, model_idx] = sample.mean()
        denom = turnover[idx].sum(axis=0)
        numer = pnl[idx].sum(axis=0)
        for model_idx in range(len(models)):
            sample = pnl[idx, model_idx]
            if np.isfinite(sample).all() and denom[model_idx] > 0.0:
                boot_ratio[rep, model_idx] = numer[model_idx] / denom[model_idx]
    model_index = {name: idx for idx, name in enumerate(models)}
    contrasts = [
        ("D_minus_A", "D", "A"), ("D_minus_no_trade", "D", None),
        ("C_minus_A", "C", "A"), ("D_minus_B", "D", "B"),
        ("B_minus_A", "B", "A"), ("D_minus_C", "D", "C"),
        ("D_minus_RAW", "D", "RAW"),
    ]
    result = {"replications": reps, "seed": int(bootstrap["seed"]), "block_days": block,
              "calendar_days": int(n_days), "paired": True, "contrasts": {}, "model_return_on_turnover": {}}
    for name, left, right in contrasts:
        left_values = pnl[:, model_index[left]]
        if right is None:
            pair_values = left_values
            distribution = boot_mean[:, model_index[left]]
        else:
            right_values = pnl[:, model_index[right]]
            pair_values = left_values - right_values
            distribution = boot_mean[:, model_index[left]] - boot_mean[:, model_index[right]]
        complete = np.isfinite(pair_values)
        estimate = float(pair_values[complete].mean()) if complete.any() else None
        result["contrasts"][name] = {
            "mean_daily_pnl_usd_on_complete_days": estimate,
            "complete_calendar_days": int(complete.sum()),
            "full_calendar_point_estimate_defined": bool(complete.all()),
            "ci95_low": float(np.nanpercentile(distribution, 2.5)) if np.isfinite(distribution).any() else None,
            "ci95_high": float(np.nanpercentile(distribution, 97.5)) if np.isfinite(distribution).any() else None,
            "undefined_replicates": int(np.isnan(distribution).sum()),
            "valid_replicates": int(np.isfinite(distribution).sum()),
        }
    for name in models:
        distribution = boot_ratio[:, model_index[name]]
        result["model_return_on_turnover"][name] = {
            "ratio_of_sums": (
                float(pnl[:, model_index[name]].sum() / turnover[:, model_index[name]].sum())
                if turnover[:, model_index[name]].sum() > 0.0 else None
            ),
            "ci95_low": float(np.nanpercentile(distribution, 2.5)) if np.isfinite(distribution).any() else None,
            "ci95_high": float(np.nanpercentile(distribution, 97.5)) if np.isfinite(distribution).any() else None,
            "undefined_replicates": int(np.isnan(distribution).sum()),
            "valid_replicates": int(np.isfinite(distribution).sum()),
        }
    return result


def _ledger_summaries(ledger: pd.DataFrame) -> dict:
    summaries = {}
    for model_name, frame in ledger.groupby("model", sort=False):
        settled = frame.loc[frame.settlement_status.eq("settled")]
        summaries[model_name] = {
            "calendar_markets": int(len(frame)),
            "decisions": int(len(frame)),
            "trade_intentions": int(frame.trade_intent.sum()),
            "no_trade_decisions": int((~frame.trade_intent).sum()),
            "full_fills": int(frame.fill_status.eq("full").sum()),
            "partial_fills": int(frame.fill_status.eq("partial_fill").sum()),
            "observed_no_fills": int(frame.fill_status.eq("no_fill").sum()),
            "execution_unknown_intentions": int(frame.settlement_status.eq("execution_unknown").sum()),
            "unsettled_positions": int(frame.settlement_status.eq("unsettled_or_label_unknown").sum()),
            "settled_no_fills": int(frame.settlement_status.eq("observed_no_fill").sum()),
            "settled_positions": int(frame.settlement_status.eq("settled").sum()),
            "settled_fill_days": int(settled.loc[settled.fill_status.isin(["full", "partial_fill"]), "market_start_utc"].dt.floor("D").nunique()),
            "requested_notional_usd": float(pd.to_numeric(frame.requested_usd, errors="coerce").fillna(0).sum()),
            "gross_filled_notional_usd": float(pd.to_numeric(frame.gross_notional_usd, errors="coerce").fillna(0).sum()),
            "cash_debit_usd": float(pd.to_numeric(frame.cash_debit_usd, errors="coerce").fillna(0).sum()),
            "fee_cash_usd": float(pd.to_numeric(frame.fee_cash_usd, errors="coerce").fillna(0).sum()),
            "fee_token_equivalent_usd_informational": float(pd.to_numeric(frame.fee_token_equivalent_usd, errors="coerce").fillna(0).sum()),
            "payout_usd_settled": float(pd.to_numeric(settled.payout_usd, errors="coerce").fillna(0).sum()),
            "settled_pnl_usd": float(pd.to_numeric(settled.settled_pnl_usd, errors="coerce").fillna(0).sum()),
            "no_fee_pnl_same_actions_usd": float(pd.to_numeric(settled.no_fee_pnl_usd, errors="coerce").fillna(0).sum()),
            "stress_pnl_same_actions_usd": {
                str(stress): float(pd.to_numeric(
                    settled[f"pnl_stress_{str(stress).replace('.', '_')}"], errors="coerce",
                ).dropna().sum())
                for stress in CONFIG["policy"]["adverse_price_stress_per_token"]
            },
            "execution_unknown_reasons": frame.loc[
                frame.settlement_status.eq("execution_unknown"), "execution_status"
            ].value_counts().to_dict(),
        }
    return summaries


def _action_return_diagnostics(
    common: pd.DataFrame, executions: pd.DataFrame, predictions: pd.DataFrame,
) -> dict:
    exec_map = executions.set_index("condition_id", drop=False)
    pred_map = predictions.set_index("condition_id", drop=False)
    records = []
    for market in common.to_dict("records"):
        if market["condition_id"] not in exec_map.index:
            continue
        execution = exec_map.loc[market["condition_id"]]
        prediction = pred_map.loc[market["condition_id"]]
        for model_name in ("B", "D"):
            for side in ("up", "down"):
                key = f"{side}_q5"
                actual = _number(execution.get(f"{key}_r"))
                predicted = _number(prediction.get(f"pred_r_{model_name}_{side}_q5"))
                if actual is None or predicted is None:
                    continue
                records.append({"model": model_name, "side": side, "market_start_utc": market["market_start_utc"],
                                "predicted_r": predicted, "realized_r": actual,
                                "error": predicted - actual})
    frame = pd.DataFrame.from_records(records)
    bin_masks = [
        ("(-inf,0]", lambda values: values <= 0.0),
        ("(0,0.01)", lambda values: (values > 0.0) & (values < 0.01)),
        ("[0.01,0.02)", lambda values: (values >= 0.01) & (values < 0.02)),
        ("[0.02,0.04)", lambda values: (values >= 0.02) & (values < 0.04)),
        ("[0.04,inf)", lambda values: values >= 0.04),
    ]
    result = {}
    for model_name in ("B", "D"):
        subset = frame.loc[frame.model.eq(model_name)] if not frame.empty else frame
        if subset.empty:
            result[model_name] = {"known_action_rows": 0, "rmse": None, "bins": []}
            continue
        subset = subset.copy()
        bins = []
        for label, mask_fn in bin_masks:
            in_bin = subset.loc[mask_fn(subset.predicted_r)]
            bins.append({
                "prediction_bin": label,
                "n": int(len(in_bin)),
                "mean_predicted_r": float(in_bin.predicted_r.mean()) if len(in_bin) else None,
                "mean_realized_r": float(in_bin.realized_r.mean()) if len(in_bin) else None,
                "rmse": float(np.sqrt(np.mean(in_bin.error.to_numpy() ** 2))) if len(in_bin) else None,
            })
        result[model_name] = {
            "known_action_rows": int(len(subset)),
            "rmse": float(np.sqrt(np.mean(subset.error.to_numpy() ** 2))),
            "mean_predicted_return": float(subset.predicted_r.mean()),
            "mean_realized_return": float(subset.realized_r.mean()),
            "bins": bins,
            "interpretation": "descriptive action-row calibration; actions are repeated outcomes within a market",
        }
    return result


def _portfolio_decision(
    model_name: str, row: dict, prediction: dict, requested_usd: float,
    estimator_by_fold: dict[int, dict[str, lgb.Booster]],
) -> dict | None:
    if not _as_bool(row.get("t60_trading_allowed", True)):
        return None
    candidates = []
    fold_id = int(prediction["fold_id"])
    for side in ("up", "down"):
        if model_name in {"RAW", "MARKET", "A", "C"}:
            p = _number(
                row["p_raw"] if model_name == "RAW" else
                row["p_market"] if model_name == "MARKET" else
                prediction[f"p_{model_name}"]
            )
            if p is None:
                continue
            features = _action_features_from_market(row, side, requested_usd)
            side_p = p if side == "up" else 1.0 - p
            shares = float(features["t60_predicted_net_shares"])
            debit = float(features["t60_predicted_cash_debit_usd"])
            expected_pnl = side_p * shares - debit
            predicted_return = expected_pnl / float(requested_usd)
        else:
            base_features = tuple(STATIC_FEATURES) if model_name == "B" else tuple(STATIC_FEATURES) + tuple(DYNAMIC_FEATURES)
            x = np.asarray([_action_matrix_row(row, side, requested_usd, base_features)], dtype=np.float32)
            predicted_return = float(estimator_by_fold[fold_id][model_name].predict(x, num_threads=1)[0])
            expected_pnl = predicted_return * float(requested_usd)
        candidates.append({"side": side, "requested_usd": float(requested_usd),
                           "predicted_return": float(predicted_return), "expected_pnl_usd": float(expected_pnl)})
    chosen = study.choose_action(candidates, minimum_return=float(CONFIG["policy"]["minimum_net_return_fraction"]))
    return dict(chosen) if chosen else None


def _run_portfolio(
    model_name: str, sizing: str, common: pd.DataFrame, predictions: pd.DataFrame,
    executions: pd.DataFrame, estimator_by_fold: dict[int, dict[str, lgb.Booster],
    ],
) -> tuple[pd.DataFrame, dict]:
    initial_cash = float(CONFIG["policy"]["portfolio_initial_cash_usd"])
    cash = initial_cash
    pending = []
    settled_pnl = 0.0
    records = []
    pred_map = predictions.set_index("condition_id", drop=False)
    execution_map = executions.set_index("condition_id", drop=False)
    for row in common.sort_values("market_start_utc", kind="stable").to_dict("records"):
        decision_time = pd.Timestamp(row["decision_time_utc"])
        still_pending = []
        for position in pending:
            if position["release_at"] is not None and position["release_at"] <= decision_time:
                cash += position["payout_usd"]
                settled_pnl += float(position["payout_usd"]) - float(position["cash_debit_usd"])
            else:
                still_pending.append(position)
        pending = still_pending
        prediction = pred_map.loc[row["condition_id"]].to_dict()
        requested = 5.0 if sizing == "fixed_5_usd" else cash * 0.05
        outside_range = study.action_size_outside_training_range(requested)
        chosen = _portfolio_decision(model_name, row, prediction, requested, estimator_by_fold)
        record = {
            "condition_id": row["condition_id"], "market_start_utc": row["market_start_utc"],
            "decision_time_utc": decision_time, "fold_id": int(prediction["fold_id"]),
            "model": model_name, "sizing": sizing,
            "free_cash_before_usd": cash, "requested_usd": requested if chosen else 0.0,
            "action_size_outside_training_range": bool(outside_range and chosen),
            "chosen_side": chosen["side"] if chosen else None,
            "predicted_return_fraction": chosen["predicted_return"] if chosen else None,
            "trade_intent": chosen is not None,
            "execution_status": "no_intent", "fill_status": "no_intent",
            "execution_known": True, "settlement_status": "no_position",
            "cash_debit_usd": 0.0, "gross_notional_usd": 0.0,
            "fee_cash_usd": 0.0, "fee_token_equivalent_usd": 0.0,
            "payout_usd": 0.0, "pnl_usd": 0.0,
        }
        if chosen:
            record["requested_usd"] = requested
            if requested < 5.0:
                record.update({"execution_status": "below_minimum_order_size", "fill_status": "not_submitted_below_minimum",
                               "execution_known": True, "settlement_status": "no_position", "pnl_usd": 0.0})
            elif requested > cash:
                record.update({"execution_status": "insufficient_free_cash", "fill_status": "not_submitted_no_credit",
                               "execution_known": True, "settlement_status": "no_position", "pnl_usd": 0.0})
            else:
                execution = execution_map.loc[row["condition_id"]].to_dict()
                fill = _execution_simulation(execution, chosen["side"], requested)
                record.update({
                    "execution_status": fill["reason"], "fill_status": fill["fill_status"],
                    "execution_known": bool(fill["execution_known"]),
                    "gross_notional_usd": fill.get("gross_usd"),
                    "cash_debit_usd": fill.get("cash_debit_usd"),
                    "fee_cash_usd": fill.get("fee_cash_usd"),
                    "fee_token_equivalent_usd": float(fill.get("fee_shares") or 0.0) * float(fill.get("vwap") or 0.0),
                })
                if not fill["execution_known"]:
                    record["pnl_usd"] = np.nan
                    record["settlement_status"] = "execution_unknown"
                elif fill["fill_status"] == "no_fill":
                    record["pnl_usd"] = 0.0
                    record["settlement_status"] = "observed_no_fill"
                elif float(fill["cash_debit_usd"]) > cash + 1e-9:
                    record.update({"execution_status": "insufficient_free_cash_after_fee",
                                   "fill_status": "not_submitted_no_credit", "execution_known": True,
                                   "settlement_status": "no_position",
                                   "gross_notional_usd": 0.0, "cash_debit_usd": 0.0,
                                   "fee_cash_usd": 0.0, "fee_token_equivalent_usd": 0.0,
                                   "pnl_usd": 0.0})
                elif not row["official_label_known"]:
                    record["pnl_usd"] = np.nan
                    record["execution_status"] = "settlement_unknown"
                    record["settlement_status"] = "unsettled_or_label_unknown"
                    cash -= float(fill["cash_debit_usd"])
                    pending.append({"release_at": None, "payout_usd": 0.0,
                                    "cash_debit_usd": float(fill["cash_debit_usd"])})
                else:
                    cash -= float(fill["cash_debit_usd"])
                    official = float(row["target_polymarket_up"])
                    won = (chosen["side"] == "up" and official == 1.0) or (chosen["side"] == "down" and official == 0.0)
                    payout = float(fill["shares"]) if won else 0.0
                    record["payout_usd"] = payout
                    record["pnl_usd"] = payout - float(fill["cash_debit_usd"])
                    record["settlement_status"] = "settled"
                    release_at = row["label_release_at_utc"]
                    pending.append({"release_at": release_at, "payout_usd": payout,
                                    "cash_debit_usd": float(fill["cash_debit_usd"])})
        record["free_cash_after_entry_usd"] = cash
        record["open_positions_after_entry"] = len(pending)
        records.append(record)
    # Settle known development positions after the final development decision; labels are still development-only.
    unresolved_cost_basis = 0.0
    unresolved_count = 0
    for position in pending:
        if position["release_at"] is not None:
            cash += position["payout_usd"]
            settled_pnl += float(position["payout_usd"]) - float(position["cash_debit_usd"])
        else:
            unresolved_count += 1
            unresolved_cost_basis += float(position["cash_debit_usd"])
    ledger = pd.DataFrame.from_records(records)
    study.assert_development_starts(ledger.market_start_utc, context=f"{model_name} {sizing} portfolio ledger")
    unknown = int((ledger.trade_intent & ~ledger.execution_known).sum())
    summary = {
        "initial_cash_usd": initial_cash,
        "ending_cash_after_known_settlement_usd": float(cash),
        "net_pnl_after_known_settlement_usd": float(settled_pnl),
        "unresolved_position_cost_basis_usd": float(unresolved_cost_basis),
        "decision_count": int(len(ledger)),
        "trade_intentions": int(ledger.trade_intent.sum()),
        "fills": int(ledger.fill_status.isin(["full", "partial_fill"]).sum()),
        "partial_fills": int(ledger.fill_status.eq("partial_fill").sum()),
        "observed_no_fills": int(ledger.fill_status.eq("no_fill").sum()),
        "unknown_executions": unknown,
        "unresolved_positions": int(unresolved_count),
        "out_of_training_range_intentions": int(ledger.action_size_outside_training_range.sum()),
        "out_of_training_range_filled_notional_usd": float(pd.to_numeric(
            ledger.loc[ledger.action_size_outside_training_range, "gross_notional_usd"], errors="coerce",
        ).fillna(0).sum()),
        "gross_filled_notional_usd": float(pd.to_numeric(ledger.gross_notional_usd, errors="coerce").fillna(0).sum()),
        "cash_debit_usd": float(pd.to_numeric(ledger.cash_debit_usd, errors="coerce").fillna(0).sum()),
        "fee_cash_usd": float(pd.to_numeric(ledger.fee_cash_usd, errors="coerce").fillna(0).sum()),
        "fee_token_equivalent_usd_informational": float(pd.to_numeric(ledger.fee_token_equivalent_usd, errors="coerce").fillna(0).sum()),
        "accounting_condition": "conditional on skipping unobservable T-59 executions; no credit; holdings keep original actions",
    }
    summary["out_of_training_range_filled_notional_fraction"] = (
        summary["out_of_training_range_filled_notional_usd"] / summary["gross_filled_notional_usd"]
        if summary["gross_filled_notional_usd"] > 0 else None
    )
    return ledger, summary


def _promotion_gate(ledger: pd.DataFrame, daily: pd.DataFrame, bootstrap: dict, accounting_pass: bool) -> dict:
    d = ledger.loc[ledger.model.eq("D")].copy()
    d_settled = d.loc[d.settlement_status.eq("settled")]
    d_days = daily.loc[daily.model.eq("D")]
    pnl_by_day = d_days.set_index("day_utc").settled_pnl_usd.sort_values(ascending=False)
    pnl_excluding_best_five = float(pnl_by_day.iloc[5:].sum()) if len(pnl_by_day) > 5 else 0.0
    stress_column = f"pnl_stress_{str(0.0025).replace('.', '_')}"
    stress_total = float(pd.to_numeric(d_settled[stress_column], errors="coerce").dropna().sum())
    d_fills = d_settled.loc[d_settled.fill_status.isin(["full", "partial_fill"])]
    primary_ci = bootstrap["contrasts"]["D_minus_A"]
    standalone_ci = bootstrap["contrasts"]["D_minus_no_trade"]
    checks = {
        "d_minus_a_lower_95_ci_daily_pnl_gt_zero": primary_ci["ci95_low"] is not None and primary_ci["ci95_low"] > 0.0,
        "d_lower_95_ci_daily_pnl_gt_zero": standalone_ci["ci95_low"] is not None and standalone_ci["ci95_low"] > 0.0,
        "d_minimum_settled_fills": len(d_fills) >= int(CONFIG["promotion_gate"]["d_minimum_settled_fills"]),
        "d_minimum_fill_days": d_fills.market_start_utc.dt.floor("D").nunique() >= int(CONFIG["promotion_gate"]["d_minimum_fill_days"]),
        "d_pnl_after_removing_best_five_days_gt_zero": pnl_excluding_best_five > 0.0,
        "d_pnl_stress_0_0025_gt_zero": stress_total > 0.0,
        "accounting_and_causality_tests_pass": bool(accounting_pass),
        "unknown_execution_does_not_prevent_interpretation": (
            primary_ci["valid_replicates"] > 0 and standalone_ci["valid_replicates"] > 0
        ),
    }
    return {
        "status": "candidate_for_separate_transfer_test" if all(checks.values()) else "development_result_no_promotion",
        "checks": checks,
        "d_settled_fill_count": int(len(d_fills)),
        "d_fill_days": int(d_fills.market_start_utc.dt.floor("D").nunique()),
        "d_pnl_after_removing_best_five_days_usd": pnl_excluding_best_five,
        "d_pnl_stress_0_0025_usd": stress_total,
    }


def _monthly_finance(ledger: pd.DataFrame) -> list[dict]:
    frame = ledger.copy()
    frame["month_utc"] = frame.market_start_utc.dt.strftime("%Y-%m")
    rows = []
    for (model, month), subset in frame.groupby(["model", "month_utc"], sort=True):
        settled = subset.loc[subset.settlement_status.eq("settled")]
        rows.append({
            "model": model, "month_utc": month,
            "calendar_markets": int(len(subset)),
            "intents": int(subset.trade_intent.sum()),
            "fills": int(subset.fill_status.isin(["full", "partial_fill"]).sum()),
            "partial_fills": int(subset.fill_status.eq("partial_fill").sum()),
            "no_fills": int(subset.fill_status.eq("no_fill").sum()),
            "unknown_execution": int(subset.settlement_status.eq("execution_unknown").sum()),
            "unsettled_positions": int(subset.settlement_status.eq("unsettled_or_label_unknown").sum()),
            "gross_filled_notional_usd": float(pd.to_numeric(subset.gross_notional_usd, errors="coerce").fillna(0).sum()),
            "cash_debit_usd": float(pd.to_numeric(subset.cash_debit_usd, errors="coerce").fillna(0).sum()),
            "fee_cash_usd": float(pd.to_numeric(subset.fee_cash_usd, errors="coerce").fillna(0).sum()),
            "fee_token_equivalent_usd": float(pd.to_numeric(subset.fee_token_equivalent_usd, errors="coerce").fillna(0).sum()),
            "payout_usd": float(pd.to_numeric(settled.payout_usd, errors="coerce").fillna(0).sum()),
            "settled_pnl_usd": float(pd.to_numeric(settled.settled_pnl_usd, errors="coerce").fillna(0).sum()),
        })
    return rows


def _fold_finance(ledger: pd.DataFrame) -> list[dict]:
    rows = []
    for (model, fold_id), subset in ledger.groupby(["model", "fold_id"], sort=True):
        settled = subset.loc[subset.settlement_status.eq("settled")]
        rows.append({
            "model": model, "fold_id": int(fold_id),
            "calendar_markets": int(len(subset)), "intents": int(subset.trade_intent.sum()),
            "fills": int(subset.fill_status.isin(["full", "partial_fill"]).sum()),
            "unknown_execution": int(subset.settlement_status.eq("execution_unknown").sum()),
            "unsettled_positions": int(subset.settlement_status.eq("unsettled_or_label_unknown").sum()),
            "gross_filled_notional_usd": float(pd.to_numeric(subset.gross_notional_usd, errors="coerce").fillna(0).sum()),
            "settled_pnl_usd": float(pd.to_numeric(settled.settled_pnl_usd, errors="coerce").fillna(0).sum()),
        })
    return rows


def _fold_probability_metrics(common: pd.DataFrame, predictions: pd.DataFrame) -> list[dict]:
    joined = common[["condition_id", "target_polymarket_up", "official_label_known"]].merge(
        predictions, on="condition_id", how="inner", validate="one_to_one",
    )
    rows = []
    for fold_id, frame in joined.loc[joined.official_label_known.fillna(False)].groupby("fold_id", sort=True):
        y = frame.target_polymarket_up.astype(int).to_numpy()
        for model_name, col in (("A", "p_A"), ("C", "p_C")):
            p = frame[col].to_numpy(dtype=np.float64)
            rows.append({
                "fold_id": int(fold_id), "model": model_name,
                "scored_markets": int(len(frame)),
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "brier": float(brier_score_loss(y, p)),
                "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            })
    return rows


def _markdown_table(rows: list[dict], columns: list[tuple[str, str]], empty: str = "No rows.") -> str:
    if not rows:
        return empty
    header = "| " + " | ".join(title for _, title in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    values = []
    for row in rows:
        values.append("| " + " | ".join(
            str(row.get(key, "")) if row.get(key) is not None else "—" for key, _ in columns
        ) + " |")
    return "\n".join([header, separator, *values])


def _write_report(
    *, status: str, availability: dict, static_audit: dict, dynamic_stats: dict,
    reconstruction: dict | None, folds: list[dict], fit_stats: list[dict],
    probability_metrics: dict | None, q5_probability_metrics: dict | None,
    fold_probability_metrics: list[dict],
    finance: dict | None, monthly: list[dict], fold_finance: list[dict], bootstrap: dict | None,
    gate: dict | None, action_diagnostics: dict | None, portfolios: dict | None,
    run_cost: dict, source_audit: dict,
) -> str:
    lines = [
        "# BTC market dynamics and action value — 2026-10-20",
        "",
        f"**Status: `{status}`.** Development cutoff is exclusive at `2026-07-01T00:00:00Z`.",
        "",
    ]
    if status != "development_result_complete":
        lines.extend([
            "The planned A/B/C/D fits and economic scoring were not run because the frozen feasibility minimums were not met or a required replay check failed.",
            "",
            f"- Common S+H+RAW inputs in development: {availability.get('common_input_market_count_all_development', 0):,}.",
            f"- Evaluation inputs from {availability.get('evaluation_start_utc')}: {availability.get('evaluation_market_count', 0):,} across {availability.get('evaluation_observed_days', 0)} observed days.",
            f"- Dynamic exclusions: `{json.dumps(availability.get('dynamic_exclusion_reasons', {}), sort_keys=True)}`.",
            f"- Availability status: `{availability.get('status')}`; fit minima pass: `{availability.get('fold_minima_pass')}`.",
            "",
            "No profitability conclusion is available from this feasibility run. Missing or unverified event-source hours remain excluded; an empty partition was not treated as proof that the target market was unchanged.",
            "",
        ])
    else:
        prob_rows = []
        for model_name in ("RAW", "MARKET", "A", "C"):
            row = probability_metrics[model_name]
            prob_rows.append({"model": model_name, "log_loss": f"{row['log_loss']:.6f}",
                              "brier": f"{row['brier']:.6f}", "auc": f"{row['auc']:.6f}" if row["auc"] is not None else None})
        financial_rows = []
        for model_name in ("RAW", "MARKET", "A", "B", "C", "D"):
            row = finance[model_name]
            financial_rows.append({
                "model": model_name, "intents": row["trade_intentions"], "fills": row["full_fills"] + row["partial_fills"],
                "partial": row["partial_fills"], "no_fill": row["observed_no_fills"],
                "unknown_exec": row["execution_unknown_intentions"], "unsettled": row["unsettled_positions"],
                "gross": f"${row['gross_filled_notional_usd']:.2f}", "debit": f"${row['cash_debit_usd']:.2f}",
                "fee_cash": f"${row['fee_cash_usd']:.4f}", "token_fee_eq": f"${row['fee_token_equivalent_usd_informational']:.4f}",
                "payout": f"${row['payout_usd_settled']:.2f}", "pnl": f"${row['settled_pnl_usd']:.2f}",
            })
        lines.extend([
            "## Evaluation result",
            "",
            f"Promotion gate: **`{gate['status']}`**. The predeclared challenger was D. Secondary results do not replace D if D fails.",
            "",
            "## Development availability",
            "",
            f"Official calendar: `{availability.get('official_market_index_first_start_utc')}` through `{source_audit.get('base70_origin', {}).get('official_market_index_last_start_utc')}`; there were no earlier official BTC market rows to join, despite older conditional BASE70 forecasts. The evaluation begins `{availability.get('evaluation_start_utc')}` and retains both planned blocks through the exclusive July 1 cutoff.",
            "",
            _markdown_table([
                {"stage": "official index", "count": static_audit["development_market_index_rows"]},
                {"stage": "RAW OOF joined", "count": static_audit["raw_oof_market_rows_joined"]},
                {"stage": "static S complete", "count": static_audit["static_complete_rows"]},
                {"stage": "dynamic H valid", "count": availability["dynamic_valid_market_count"]},
                {"stage": "common S+H+RAW", "count": availability["common_input_market_count_all_development"]},
                {"stage": "evaluation common", "count": availability["evaluation_market_count"]},
                {"stage": "evaluation q=$5 targets both sides", "count": availability["evaluation_market_count_known_q5_both_sides"]},
            ], [("stage", "Stage"), ("count", "Markets")]),
            "",
            "Historical event windows crossing an empty or absent hourly partition were excluded as unverified. Book resets are not counted as depletion, receive/source causality was enforced, and initialization had to precede T−120. Replay endpoint comparisons against the saved T−60 snapshot:",
            "",
            f"`{json.dumps(reconstruction, sort_keys=True)}`",
            "",
            "## Probability models",
            "",
            "A/C metrics use the same OOF evaluation market calendar and official labels. RAW BASE70 and MARKET are unfit controls.",
            "",
            _markdown_table(prob_rows, [("model", "Model"), ("log_loss", "Log loss"), ("brier", "Brier"), ("auc", "AUC")]),
            "",
            f"The same A/C/RAW/MARKET diagnostics restricted to evaluation markets with known q=$5 targets on both sides: `{json.dumps(q5_probability_metrics, sort_keys=True)}`.",
            "",
            "## Fixed $5 independent economic decisions",
            "",
            "Decision is at T−60, execution is at T−59. Every model chooses the side with the highest expected or predicted net return, tie UP, and trades only at ≥1% of requested notional. T−59 affects fill and settlement accounting only. Unknown execution and unresolved positions are not recorded as zero or as losses.",
            "",
            _markdown_table(financial_rows, [
                ("model", "Model"), ("intents", "Intents"), ("fills", "Full/partial fills"),
                ("partial", "Partial"), ("no_fill", "No fill"), ("unknown_exec", "Unknown execution"),
                ("unsettled", "Unsettled"), ("gross", "Gross filled"), ("debit", "Cash debit"),
                ("fee_cash", "Cash fee"), ("token_fee_eq", "Token fee equivalent"),
                ("payout", "Settled payout"), ("pnl", "Settled PnL"),
            ]),
            "",
            "Fee-free PnL keeps the same selected actions; stress values keep the same sides and requested sizes. Token-denominated fees are shown as an informational entry-price equivalent. The ledger uses the shared replay FAK/partial fill kernel and official settlement.",
            "",
            "## OOF probability and action diagnostics",
            "",
            f"Paired circular moving-block bootstrap: {bootstrap['replications']:,} replications, {bootstrap['block_days']} UTC days, seed {bootstrap['seed']}. Primary D−A mean daily settled PnL: `{json.dumps(bootstrap['contrasts']['D_minus_A'], sort_keys=True)}`. D vs no-trade: `{json.dumps(bootstrap['contrasts']['D_minus_no_trade'], sort_keys=True)}`.",
            "",
            f"All secondary paired contrasts: `{json.dumps({k:v for k,v in bootstrap['contrasts'].items() if k not in ('D_minus_A','D_minus_no_trade')}, sort_keys=True)}`.",
            "",
            f"Action return diagnostics by frozen prediction bins: `{json.dumps(action_diagnostics, sort_keys=True)}`.",
            "",
            f"Fold probability metrics: `{json.dumps(fold_probability_metrics, sort_keys=True)}`.",
            "",
            f"Monthly fixed-$5 economics: `{json.dumps(monthly, sort_keys=True)}`.",
            "",
            f"Fold fixed-$5 economics: `{json.dumps(fold_finance, sort_keys=True)}`.",
            "",
            "## Continuous $100 portfolios",
            "",
            "Each policy starts once at $100 at the shared evaluation start; cash is not reset across folds. The 5% path prices its actual requested amount at both T−60 and T−59. Missing execution makes the path conditional on skipping unobservable orders. Orders below the $5 fallback minimum are skipped, never increased to the minimum.",
            "",
            f"`{json.dumps(portfolios, sort_keys=True)}`",
            "",
            "## Protocol, limits, and provenance",
            "",
            f"Availability minima: `{json.dumps(availability['minimums'], sort_keys=True)}`. Fold fits and released label minima: `{json.dumps(availability['folds'], sort_keys=True)}`.",
            "",
            "S contains the clipped logits of RAW and T−60 market probability, their logit difference, both spreads, mid-sum deviation, log ask-band notionals, and $5 T−60 slippage. H contains the seven predeclared market-book changes for each of 5/15/60 seconds, including native fixed-band ask depletion ratios. Models A/C classify the official outcome; B/D regress the net return fraction for side and requested size with per-market action weights summing to one.",
            "",
            "A/B/C/D use the single frozen LightGBM configuration in the protocol, CPU only, no tuning or early stopping. Training is expanding and purged strictly by label release before each block's first T−60 decision. B−A changes both target and task, so the contrast is not a pure loss-function comparison; a target change alone adds no market information.",
            "",
            "RAW BASE70 is conditional OOF after earlier feature/generator/parameter selection on development. This experiment is not an independent nested test of that full pipeline. The result is conditional on the earlier BASE70 selection and does not establish future profitability.",
            "",
            "A historical holdout summary was incidentally visible in the pre-existing status document before this work; it was not used. No holdout market rows or monthly holdout tables were read for this experiment, no holdout scoring was run, and no feature/window/threshold/model choice used holdout performance.",
            "",
            f"Source audit: `{json.dumps({key: value for key, value in source_audit.items() if key != 'event_parts'}, sort_keys=True)}`",
            "",
            f"Event scan: `{json.dumps(dynamic_stats, sort_keys=True)}`",
            "",
            f"Compute: `{json.dumps(run_cost, sort_keys=True)}`",
            "",
            "`holdout not read/scored` (no holdout market rows or scoring; see the historical-summary exposure note above). `live unchanged`. `orders not submitted`.",
            "",
        ])
    lines.extend([
        "## Artifacts",
        "",
        f"- Frozen protocol: `{FREEZE_PATH.relative_to(ROOT).as_posix()}`",
        f"- Availability: `{AVAILABILITY_PATH.relative_to(ROOT).as_posix()}`",
        f"- Run manifest: `{RUN_MANIFEST_PATH.relative_to(ROOT).as_posix()}`",
        *(["- Inactive transfer-test candidate manifest: `reports/btc_preopen/btc_market_dynamics_action_value_20261020_transfer_candidate.json`"]
          if gate and gate["status"] == "candidate_for_separate_transfer_test" else []),
        "- Large development-only tables, predictions, economics, and checkpoints are under the configured `data/analysis/...` output directory and are not staged.",
        "",
    ])
    report = "\n".join(lines)
    report_path = ROOT / CONFIG["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(report_path)
    return report


def _resource_snapshot() -> dict:
    result = {"working_set_bytes": _working_set_bytes(), "cpu_count": os.cpu_count()}
    try:
        import psutil

        memory = psutil.virtual_memory()
        result["available_memory_bytes"] = int(memory.available)
        result["current_process_rss_bytes"] = int(psutil.Process().memory_info().rss)
        result["system_cpu_percent"] = float(psutil.cpu_percent(interval=0.1))
    except Exception:
        pass
    return result


def _set_run_status(payload: dict) -> None:
    _write_json(RUN_MANIFEST_PATH, payload)


def _stage_checkpoint(name: str, identity: dict, outputs: list[Path], status: str = "completed") -> None:
    path = CHECKPOINT_DIR / "stages" / f"{name}.json"
    _write_json(path, {
        "stage": name,
        "status": status,
        "identity": identity,
        "outputs": {
            output.relative_to(ROOT).as_posix(): _sha256(output)
            for output in outputs if output.is_file()
        },
        "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    })


def _write_candidate_manifest(gate: dict, freeze: dict, fit_stats: list[dict]) -> str | None:
    if gate["status"] != "candidate_for_separate_transfer_test":
        return None
    path = ROOT / "reports/btc_preopen/btc_market_dynamics_action_value_20261020_transfer_candidate.json"
    payload = {
        "status": "inactive_candidate_for_separate_transfer_test",
        "experiment_id": CONFIG["experiment_id"],
        "candidate_family": "D",
        "feature_order": list(STATIC_FEATURES) + list(DYNAMIC_FEATURES) + list(ACTION_DESCRIPTORS),
        "action_sizes_usd": list(ACTION_SIZES),
        "model": CONFIG["model"],
        "future_fit_rule": "expanding development prefix with label-release purge; no July fit or score in this experiment",
        "decision": CONFIG["policy"],
        "fit_folds": [{
            "fold_id": item["fold_id"], "training": item["checkpoint"]["training"],
            "model_paths": item["checkpoint"]["model_paths"],
        } for item in fit_stats],
        "freeze_sha256": _canonical_sha256(freeze),
        "holdout_evaluation": False,
        "live_activation": False,
        "orders_submitted": False,
    }
    _write_json(path, payload)
    return path.relative_to(ROOT).as_posix()


def _update_status_document(status: str, gate: dict | None, report_relpath: str) -> None:
    marker_path = CHECKPOINT_DIR / "status_document_update.json"
    if marker_path.is_file():
        return
    status_doc = ROOT / CONFIG["status_path"]
    section = (
        "\n\n## 2026-10-20 — BTC market dynamics and action value\n\n"
        f"Experiment `{CONFIG['experiment_id']}`: **{status}**. "
        f"Promotion: `{gate['status'] if gate else 'not evaluated'}`. "
        f"Report: `{report_relpath}`. Development cutoff: exclusive `2026-07-01T00:00:00Z`. "
        "No holdout market rows were read or scored; a pre-existing historical holdout summary was incidentally seen and was not used. "
        "Live configuration unchanged. Orders not submitted.\n"
    )
    with status_doc.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(section)
    _write_json(marker_path, {"status_document": status_doc.relative_to(ROOT).as_posix(),
                              "experiment_id": CONFIG["experiment_id"], "updated": True})


def main() -> int:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ECONOMICS_DIR.mkdir(parents=True, exist_ok=True)
    initial_resources = _resource_snapshot()
    run = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "running", "current_stage": "source_audit",
        "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "initial_resources": initial_resources,
        "holdout_evaluation": False, "live_actions": False, "orders_submitted": False,
    }
    _set_run_status(run)
    status = "insufficient_development_coverage"
    availability = {"status": status, "dynamic_exclusion_reasons": {}}
    static_audit: dict = {}
    dynamic_stats: dict = {}
    reconstruction = None
    folds: list[dict] = []
    fit_stats: list[dict] = []
    probability_metrics = None
    q5_probability_metrics = None
    fold_probability_metrics: list[dict] = []
    finance = None
    monthly: list[dict] = []
    fold_finance: list[dict] = []
    bootstrap = None
    gate = None
    action_diagnostics = None
    portfolios = None
    source_audit: dict = {}
    common = pd.DataFrame()
    predictions = pd.DataFrame()
    validation = None
    try:
        run.update({"current_stage": "required_tests"})
        _set_run_status(run)
        validation = _run_required_tests()
        _stage_checkpoint("required_tests", {
            "runner_sha256": validation["runner_sha256"],
            "feature_code_sha256": validation["feature_code_sha256"],
            "test_files_sha256": validation["test_files_sha256"],
        }, [OUTPUT_DIR / "required_tests.json"])
        static, market_index, _, static_audit = _load_input_tables()
        source_audit = _source_manifest_audit(static, market_index)
        source_audit_path = AVAILABILITY_PATH.with_name("source_audit.json")
        _write_json(source_audit_path, source_audit)
        _stage_checkpoint("source_audit", {"config_sha256": _sha256(CONFIG_PATH)}, [source_audit_path])
        _stage_checkpoint("static_inputs", {
            "source_audit": source_audit, "runner_sha256": _sha256(Path(__file__).resolve()),
        }, [STATIC_PATH])
        run.update({"current_stage": "dynamic_event_replay", "source_audit": source_audit,
                    "static_market_count": len(static)})
        _set_run_status(run)

        dynamic, endpoints, dynamic_stats = _run_dynamic_replay(static, source_audit)
        _write_parquet(static, STATIC_PATH)
        _stage_checkpoint("dynamic_replay", {
            "source_audit_sha256": _sha256(source_audit_path),
            "runner_sha256": _sha256(Path(__file__).resolve()),
        }, [DYNAMIC_PATH, ENDPOINT_PATH])
        run.update({"current_stage": "T60_reconstruction_validation", "dynamic_stats": dynamic_stats})
        _set_run_status(run)
        try:
            reconstruction = _validate_t60_reconstruction(static, endpoints)
        except Exception as exc:
            availability = {
                "status": "blocked_t60_replay_mismatch",
                "dynamic_valid_market_count": int(dynamic.dynamic_valid.fillna(False).sum()),
                "dynamic_exclusion_reasons": dynamic.loc[~dynamic.dynamic_valid.fillna(False), "dynamic_reason"].value_counts().to_dict(),
                "reconstruction_error": f"{type(exc).__name__}: {exc}",
                "development_cutoff_exclusive_utc": DEVELOPMENT_CUTOFF.isoformat(),
                "holdout_rows_materialized": 0,
            }
            status = availability["status"]
            _write_json(AVAILABILITY_PATH, availability)
        else:
            t59 = _read_t59()
            labels = _read_development_labels()
            common, executions, target_coverage = _build_action_targets(static, dynamic, t59, labels)
            _write_parquet(dynamic, DYNAMIC_PATH)
            _write_parquet(endpoints, ENDPOINT_PATH)
            _write_parquet(common, COMMON_PATH)
            _write_parquet(executions, EXECUTION_TARGET_PATH)
            _stage_checkpoint("action_targets", {
                "static_sha256": _sha256(STATIC_PATH), "dynamic_sha256": _sha256(DYNAMIC_PATH),
                "t59_sha256": _sha256(T59_SOURCE), "runner_sha256": _sha256(Path(__file__).resolve()),
            }, [COMMON_PATH, EXECUTION_TARGET_PATH])
            folds = _make_folds(common, executions)
            availability, daily_availability = _availability_report(
                static, dynamic, common, executions, folds, source_audit,
            )
            availability["target_coverage"] = target_coverage
            availability["t59_snapshot_rows_development"] = int(len(t59))
            availability["t59_source_hash_sha256"] = _sha256(T59_SOURCE)
            availability["development_label_rows"] = int(len(labels))
            availability["development_label_known_rows"] = int(labels.target_polymarket_up.notna().sum())
            status = availability["status"]
            daily_availability_path = OUTPUT_DIR / "availability_by_day.parquet"
            _write_parquet(daily_availability, daily_availability_path)
            _write_json(AVAILABILITY_PATH, availability)
            _stage_checkpoint("availability", {
                "common_sha256": _sha256(COMMON_PATH), "execution_targets_sha256": _sha256(EXECUTION_TARGET_PATH),
                "folds": [{key: value for key, value in item.items() if key != "evaluation_market_ids"} for item in folds],
            }, [AVAILABILITY_PATH, daily_availability_path])
            artifacts = {
                "runner_sha256": _sha256(Path(__file__).resolve()),
                "feature_code_sha256": _sha256(ROOT / "features/btc_market_dynamics_action_value.py"),
                "config_sha256": _sha256(CONFIG_PATH),
                "static_sha256": _sha256(STATIC_PATH),
                "dynamic_sha256": _sha256(DYNAMIC_PATH),
                "endpoint_sha256": _sha256(ENDPOINT_PATH),
                "common_inputs_sha256": _sha256(COMMON_PATH),
                "execution_targets_sha256": _sha256(EXECUTION_TARGET_PATH),
                "daily_availability_sha256": _sha256(daily_availability_path),
            }
            freeze = _freeze_protocol(availability, folds, artifacts, validation)
            if FREEZE_PATH.is_file():
                prior_freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
                if prior_freeze.get("artifact_hashes") != artifacts:
                    raise RuntimeError("existing frozen protocol inputs differ; preserve the old freeze and create an explicit amendment")
                freeze = prior_freeze
            else:
                _write_json(FREEZE_PATH, freeze)
            _stage_checkpoint("frozen_protocol", {
                "availability_sha256": _sha256(AVAILABILITY_PATH), "artifact_hashes": artifacts,
            }, [FREEZE_PATH])
            run.update({"current_stage": "availability_gates", "availability": availability,
                        "freeze_sha256": _sha256(FREEZE_PATH)})
            _set_run_status(run)

            if status == "availability_passed":
                # Keep CPU headroom for active desktop processes and the operating system.
                threads = min(int(CONFIG["model"]["max_threads"]), 4, max(1, (os.cpu_count() or 2) - 2))
                source_identity_sha256 = _canonical_sha256({
                    "source_audit": source_audit, "artifact_hashes": artifacts,
                    "folds": [{key: value for key, value in item.items() if key != "evaluation_market_ids"} for item in folds],
                })
                for fold in folds:
                    if fold["evaluation_market_count"] == 0:
                        continue
                    fold_started = time.perf_counter()
                    fold_predictions, fit_stat = _fit_fold(
                        common, executions, fold, source_identity_sha256, threads,
                    )
                    fit_stat["elapsed_seconds_this_run"] = time.perf_counter() - fold_started
                    fit_stats.append(fit_stat)
                    run.update({"current_stage": f"fit_fold_{fold['fold_id']}", "completed_folds": len(fit_stats),
                                "last_fold_fit": fit_stat, "resources": _resource_snapshot()})
                    _set_run_status(run)
                    fold_prediction_path = CHECKPOINT_DIR / f"fold_{int(fold['fold_id']):02d}" / "predictions.parquet"
                    if not fold_predictions.empty:
                        pass
                predictions = pd.concat([
                    pd.read_parquet(CHECKPOINT_DIR / f"fold_{int(fold['fold_id']):02d}" / "predictions.parquet")
                    for fold in folds if fold["evaluation_market_count"] > 0
                ], ignore_index=True)
                predictions.sort_values("market_start_utc", kind="stable", inplace=True, ignore_index=True)
                study.assert_development_starts(predictions.market_start_utc, context="OOF development predictions")
                if predictions.condition_id.duplicated().any():
                    raise RuntimeError("OOF evaluation market received more than one prediction")
                _write_parquet(predictions, PREDICTION_PATH)
                decisions = _build_decisions(common, predictions)
                eval_start = pd.Timestamp(availability["evaluation_start_utc"])
                decisions = decisions.loc[decisions.market_start_utc.ge(eval_start)].copy()
                eval_common = common.loc[common.market_start_utc.ge(eval_start)].copy()
                eval_predictions = predictions.loc[predictions.market_start_utc.ge(eval_start)].copy()
                eval_executions = executions.loc[executions.condition_id.isin(eval_common.condition_id)].copy()
                _write_parquet(decisions, ECONOMICS_DIR / "fixed5_decisions.parquet")
                ledger = _build_fixed5_ledger(decisions, eval_common, eval_executions)
                _write_parquet(ledger, ECONOMICS_DIR / "fixed5_ledger.parquet")
                daily = _daily_policy_frame(
                    ledger, eval_start, DEVELOPMENT_CUTOFF - pd.Timedelta(days=1),
                )
                _write_parquet(daily, ECONOMICS_DIR / "fixed5_daily.parquet")
                probability_metrics = _probability_metrics(eval_common, eval_predictions)
                known_q5_ids = set(eval_executions.loc[
                    eval_executions.up_q5_label_known.fillna(False).astype(bool)
                    & eval_executions.down_q5_label_known.fillna(False).astype(bool),
                    "condition_id",
                ])
                q5_common = eval_common.loc[eval_common.condition_id.isin(known_q5_ids)].copy()
                q5_predictions = eval_predictions.loc[eval_predictions.condition_id.isin(known_q5_ids)].copy()
                q5_probability_metrics = _probability_metrics(q5_common, q5_predictions)
                fold_probability_metrics = _fold_probability_metrics(eval_common, eval_predictions)
                finance = _ledger_summaries(ledger)
                monthly = _monthly_finance(ledger)
                fold_finance = _fold_finance(ledger)
                bootstrap = _bootstrap_daily(daily)
                action_diagnostics = _action_return_diagnostics(eval_common, eval_executions, eval_predictions)
                estimator_by_fold = {}
                for item in fit_stats:
                    fold_id = item["fold_id"]
                    estimator_by_fold[fold_id] = {
                        name: lgb.Booster(model_file=path) for name, path in item["model_paths"].items()
                    }
                portfolios = {}
                for model_name in ("RAW", "MARKET", "A", "B", "C", "D"):
                    for sizing in ("fixed_5_usd", "5pct_free_cash"):
                        portfolio_ledger, summary = _run_portfolio(
                            model_name, sizing, eval_common, eval_predictions, eval_executions,
                            estimator_by_fold,
                        )
                        portfolios[f"{model_name}_{sizing}"] = summary
                        _write_parquet(
                            portfolio_ledger,
                            ECONOMICS_DIR / f"portfolio_{model_name}_{sizing}.parquet",
                        )
                # The fixed-$5 portfolio must preserve the OOF-selected actions before any cash constraints.
                for model_name in ("RAW", "MARKET", "A", "B", "C", "D"):
                    decision_rows = decisions.loc[decisions.model.eq(model_name)].set_index("condition_id")
                    portfolio_rows = pd.read_parquet(ECONOMICS_DIR / f"portfolio_{model_name}_fixed_5_usd.parquet").set_index("condition_id")
                    overlapping = decision_rows.index.intersection(portfolio_rows.index)
                    if not decision_rows.loc[overlapping, "chosen_side"].fillna("").equals(
                        portfolio_rows.loc[overlapping, "chosen_side"].fillna("")
                    ):
                        raise RuntimeError(f"{model_name} fixed-$5 portfolio changed OOF side selection")
                accounting_pass = all(
                    abs(float(row.settled_pnl_usd) - float(row.payout_usd) + float(row.cash_debit_usd)) <= 1e-8
                    for row in ledger.loc[ledger.settlement_status.eq("settled")].itertuples(index=False)
                )
                causal_pass = (
                    bool(common.market_start_utc.lt(DEVELOPMENT_CUTOFF).all())
                    and bool(predictions.market_start_utc.lt(DEVELOPMENT_CUTOFF).all())
                )
                for fold in folds:
                    start_at = pd.Timestamp(fold["evaluation_start_utc"])
                    first_decision = pd.Timestamp(fold["first_decision_at_utc"])
                    fold_training = common.loc[
                        common.market_start_utc.lt(start_at)
                        & common.official_label_known.fillna(False)
                        & common.label_release_at_utc.lt(first_decision)
                    ]
                    causal_pass = causal_pass and bool(
                        fold_training.market_start_utc.lt(start_at).all()
                        and fold_training.label_release_at_utc.lt(first_decision).all()
                    )
                accounting_pass = accounting_pass and causal_pass and validation["status"] == "passed"
                gate = _promotion_gate(ledger, daily, bootstrap, accounting_pass)
                candidate_path = _write_candidate_manifest(gate, freeze, fit_stats)
                status = "development_result_complete"
                run.update({"current_stage": "evaluation_complete", "fit_stats": fit_stats,
                            "probability_metrics": probability_metrics, "finance": finance,
                            "bootstrap": bootstrap, "promotion_gate": gate,
                            "transfer_candidate_manifest": candidate_path})
                _set_run_status(run)

        total_elapsed = time.perf_counter() - started
        final_resources = _resource_snapshot()
        run_cost = {
            "elapsed_seconds": total_elapsed,
            "initial_resources": initial_resources,
            "final_resources": final_resources,
            "dynamic_replay_elapsed_seconds": dynamic_stats.get("elapsed_seconds"),
            "replay_event_rows_selected": dynamic_stats.get("event_rows_selected"),
            "replay_worker_count": dynamic_stats.get("workers"),
            "fit_elapsed_seconds_current_invocation": float(sum(item.get("elapsed_seconds_this_run", 0.0) for item in fit_stats)),
            "evaluation_market_count": availability.get("evaluation_market_count"),
        }
        report = _write_report(
            status=status, availability=availability, static_audit=static_audit,
            dynamic_stats=dynamic_stats, reconstruction=reconstruction, folds=folds,
            fit_stats=fit_stats, probability_metrics=probability_metrics,
            q5_probability_metrics=q5_probability_metrics,
            fold_probability_metrics=fold_probability_metrics, finance=finance,
            monthly=monthly, bootstrap=bootstrap, gate=gate,
            fold_finance=fold_finance,
            action_diagnostics=action_diagnostics, portfolios=portfolios,
            run_cost=run_cost, source_audit=source_audit,
        )
        _update_status_document(status, gate, CONFIG["report_path"])
        final_manifest = {
            **run,
            "status": status,
            "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "total_elapsed_seconds": total_elapsed,
            "run_cost": run_cost,
            "availability": availability,
            "dynamic_stats": dynamic_stats,
            "t60_reconstruction": reconstruction,
            "fit_stats": fit_stats,
            "probability_metrics": probability_metrics,
            "q5_probability_metrics": q5_probability_metrics,
            "finance": finance,
            "bootstrap": bootstrap,
            "promotion_gate": gate,
            "report_sha256": hashlib.sha256(report.encode("utf-8")).hexdigest(),
            "holdout_evaluation": False, "live_actions": False, "orders_submitted": False,
        }
        _set_run_status(final_manifest)
        return 0
    except BaseException as exc:
        total_elapsed = time.perf_counter() - started
        run.update({"status": "failed", "current_stage": run.get("current_stage"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "failed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "elapsed_seconds": total_elapsed})
        _set_run_status(run)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
