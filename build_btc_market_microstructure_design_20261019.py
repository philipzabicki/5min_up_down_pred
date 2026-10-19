"""Build a development-only T-60 feature table for a proposed BTC/market study."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import run_btc_market_probability_experiment as market_probability


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_market_microstructure_design_20261019.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
START_UTC = pd.Timestamp(CONFIG["development_start_utc"])
CUTOFF_UTC = pd.Timestamp(CONFIG["development_cutoff_exclusive_utc"])
OUTPUT_DIR = ROOT / CONFIG["output_directory"]
TABLE_PATH = ROOT / CONFIG["feature_table_path"]
MANIFEST_PATH = ROOT / CONFIG["manifest_path"]
PROBABILITY_FEATURES = ("p_market", "p_raw", "raw_minus_market_logit")
MICROSTRUCTURE_FEATURES = (
    "spread_up", "spread_down", "up_down_ask_depth_imbalance",
    "slippage_5_up", "slippage_5_down", "mid_sum_minus_one",
)
SNAPSHOT_COLUMNS = [
    "condition_id", "market_start_utc", "entry_time_utc",
    "up_best_bid", "up_best_ask", "up_best_ask_size_shares",
    "down_best_bid", "down_best_ask", "down_best_ask_size_shares",
    "up_ask_levels", "down_ask_levels", "quote_valid", "has_full_snapshot",
    "up_quote_complemented", "down_quote_complemented",
    "archive_entry_hour_available", "archive_freshness_window_available",
    "no_future_event_at_entry", "no_future_source_event_at_entry",
    "archive_gap_after_up_book_init", "archive_gap_after_down_book_init",
    "market_book_snapshot_count", "depth_5usd_valid_both_sides",
    "up_ask_order_ambiguous", "down_ask_order_ambiguous",
    "market_equal_source_timestamp_price_change_count",
    "market_out_of_order_price_change_count", "market_cross_side_late_price_change_count",
    "up_bid_ask_crossed", "down_bid_ask_crossed", "up_bid_ask_locked", "down_bid_ask_locked",
    "up_ask_age_seconds", "down_ask_age_seconds",
    "up_ask_received_age_seconds", "down_ask_received_age_seconds",
]
RAW_COLUMNS = ["market_start_utc", "nominal_decision_at", "p_model_raw"]


def _read_development(path: Path, columns: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    field_type = dataset.schema.field("market_start_utc").type
    start = pa.scalar(START_UTC.to_pydatetime(), type=field_type)
    cutoff = pa.scalar(CUTOFF_UTC.to_pydatetime(), type=field_type)
    date_filter = (ds.field("market_start_utc") >= start) & (ds.field("market_start_utc") < cutoff)
    result = dataset.to_table(columns=columns, filter=date_filter).to_pandas()
    result["market_start_utc"] = pd.to_datetime(result.market_start_utc, utc=True, errors="raise")
    return result


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_true(value) -> bool:
    return value is not None and not pd.isna(value) and bool(value)


def _is_false(value) -> bool:
    return value is not None and not pd.isna(value) and not bool(value)


def _levels(value) -> list[tuple[float, float]]:
    if value is None:
        return []
    result = []
    for level in value:
        if level is None or len(level) != 2:
            continue
        price, shares = _number(level[0]), _number(level[1])
        if price is not None and shares is not None and 0.0 < price < 1.0 and shares > 0.0:
            result.append((price, shares))
    return sorted(result)


def _vwap_for_gross_spend(levels, gross_usd: float = 5.0) -> float | None:
    remaining = float(gross_usd)
    shares = 0.0
    spent = 0.0
    for price, available_shares in levels:
        taken = min(remaining / price, available_shares)
        level_spend = taken * price
        shares += taken
        spent += level_spend
        remaining -= level_spend
        if remaining <= 1e-8:
            break
    if remaining > 1e-6 or shares <= 0.0:
        return None
    return spent / shares


def _eligible_for_microstructure(snapshot: dict, p_market: float | None) -> bool:
    if p_market is None:
        return False
    required_true = (
        "quote_valid", "has_full_snapshot", "archive_entry_hour_available",
        "archive_freshness_window_available", "no_future_event_at_entry",
        "no_future_source_event_at_entry",
    )
    if not all(_is_true(snapshot.get(name)) for name in required_true):
        return False
    if not _is_false(snapshot.get("up_quote_complemented")) or not _is_false(snapshot.get("down_quote_complemented")):
        return False
    if not _is_false(snapshot.get("archive_gap_after_up_book_init")) or not _is_false(snapshot.get("archive_gap_after_down_book_init")):
        return False
    if not _is_false(snapshot.get("up_ask_order_ambiguous")) or not _is_false(snapshot.get("down_ask_order_ambiguous")):
        return False
    if not _is_false(snapshot.get("up_bid_ask_crossed")) or not _is_false(snapshot.get("down_bid_ask_crossed")):
        return False
    return True


def _feature_record(raw_row, snapshot: dict | None) -> dict:
    market_start = pd.Timestamp(raw_row.market_start_utc)
    nominal_decision = pd.Timestamp(raw_row.nominal_decision_at)
    if nominal_decision.tzinfo is None:
        nominal_decision = nominal_decision.tz_localize("UTC")
    else:
        nominal_decision = nominal_decision.tz_convert("UTC")
    if nominal_decision != market_start - pd.Timedelta(seconds=60):
        raise RuntimeError(f"RAW OOF decision is not T-60: {market_start}")

    raw_probability = _number(raw_row.p_model_raw)
    if raw_probability is not None and not 0.0 <= raw_probability <= 1.0:
        raise RuntimeError("Saved RAW BASE70 OOF probability is outside [0, 1]")
    result = {
        "market_start_utc": market_start,
        "condition_id": None,
        "p_raw": raw_probability,
        "p_market": None,
        "raw_minus_market_logit": None,
        **{name: None for name in MICROSTRUCTURE_FEATURES},
        "t60_snapshot_available": snapshot is not None,
        "t60_entry_time_utc": None,
        "microstructure_eligible": False,
        "has_full_snapshot": False,
        "up_quote_complemented": None,
        "down_quote_complemented": None,
        "archive_gap_after_up_book_init": None,
        "archive_gap_after_down_book_init": None,
        "up_ask_order_ambiguous": None,
        "down_ask_order_ambiguous": None,
        "equal_source_timestamp_conflict_count": None,
        "out_of_order_price_change_count": None,
        "cross_side_late_price_change_count": None,
        "up_bid_ask_crossed": None,
        "down_bid_ask_crossed": None,
        "up_bid_ask_locked": None,
        "down_bid_ask_locked": None,
        "depth_5usd_valid_both_sides": None,
        "market_book_snapshot_count": None,
        "up_ask_book_event_age_seconds": None,
        "down_ask_book_event_age_seconds": None,
        "up_ask_received_book_event_age_seconds": None,
        "down_ask_received_book_event_age_seconds": None,
    }
    if snapshot is None:
        return result

    snapshot_start = pd.Timestamp(snapshot["market_start_utc"])
    if snapshot_start.tzinfo is None:
        snapshot_start = snapshot_start.tz_localize("UTC")
    else:
        snapshot_start = snapshot_start.tz_convert("UTC")
    if snapshot_start != market_start:
        raise RuntimeError(f"Snapshot market start does not match RAW OOF row: {market_start}")
    entry_time = pd.Timestamp(snapshot["entry_time_utc"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize("UTC")
    else:
        entry_time = entry_time.tz_convert("UTC")
    if entry_time != market_start - pd.Timedelta(seconds=60):
        raise RuntimeError(f"Snapshot is not T-60: {market_start} at {entry_time}")
    result["condition_id"] = str(snapshot["condition_id"]).lower()
    result["t60_entry_time_utc"] = entry_time

    market = market_probability._snapshot_features(snapshot)
    p_market = _number(market["p_market"])
    result["p_market"] = p_market
    if p_market is not None and raw_probability is not None:
        logit_market = market_probability._logit([p_market])[0]
        logit_raw = market_probability._logit([raw_probability])[0]
        result["raw_minus_market_logit"] = float(logit_raw - logit_market)

    for source, destination in (
        ("has_full_snapshot", "has_full_snapshot"),
        ("up_quote_complemented", "up_quote_complemented"),
        ("down_quote_complemented", "down_quote_complemented"),
        ("archive_gap_after_up_book_init", "archive_gap_after_up_book_init"),
        ("archive_gap_after_down_book_init", "archive_gap_after_down_book_init"),
        ("up_ask_order_ambiguous", "up_ask_order_ambiguous"),
        ("down_ask_order_ambiguous", "down_ask_order_ambiguous"),
        ("market_equal_source_timestamp_price_change_count", "equal_source_timestamp_conflict_count"),
        ("market_out_of_order_price_change_count", "out_of_order_price_change_count"),
        ("market_cross_side_late_price_change_count", "cross_side_late_price_change_count"),
        ("up_bid_ask_crossed", "up_bid_ask_crossed"),
        ("down_bid_ask_crossed", "down_bid_ask_crossed"),
        ("up_bid_ask_locked", "up_bid_ask_locked"),
        ("down_bid_ask_locked", "down_bid_ask_locked"),
        ("depth_5usd_valid_both_sides", "depth_5usd_valid_both_sides"),
        ("market_book_snapshot_count", "market_book_snapshot_count"),
        ("up_ask_age_seconds", "up_ask_book_event_age_seconds"),
        ("down_ask_age_seconds", "down_ask_book_event_age_seconds"),
        ("up_ask_received_age_seconds", "up_ask_received_book_event_age_seconds"),
        ("down_ask_received_age_seconds", "down_ask_received_book_event_age_seconds"),
    ):
        value = snapshot.get(source)
        if "age_seconds" in source or "count" in source:
            result[destination] = _number(value)
        elif value is None or pd.isna(value):
            result[destination] = None
        else:
            result[destination] = bool(value)

    if not _eligible_for_microstructure(snapshot, p_market):
        return result

    prices = {
        side: {
            "bid": _number(snapshot.get(f"{side}_best_bid")),
            "ask": _number(snapshot.get(f"{side}_best_ask")),
            "ask_size": _number(snapshot.get(f"{side}_best_ask_size_shares")),
            "levels": _levels(snapshot.get(f"{side}_ask_levels")),
        }
        for side in ("up", "down")
    }
    if any(
        values["bid"] is None or values["ask"] is None or values["ask_size"] is None
        or not values["levels"] or abs(values["levels"][0][0] - values["ask"]) > 1e-8
        for values in prices.values()
    ):
        return result

    size_sum = prices["up"]["ask_size"] + prices["down"]["ask_size"]
    if size_sum <= 0.0:
        return result
    mid_up = (prices["up"]["bid"] + prices["up"]["ask"]) / 2.0
    mid_down = (prices["down"]["bid"] + prices["down"]["ask"]) / 2.0
    vwap_up = _vwap_for_gross_spend(prices["up"]["levels"])
    vwap_down = _vwap_for_gross_spend(prices["down"]["levels"])
    if vwap_up is None or vwap_down is None:
        return result

    result.update({
        "spread_up": prices["up"]["ask"] - prices["up"]["bid"],
        "spread_down": prices["down"]["ask"] - prices["down"]["bid"],
        "up_down_ask_depth_imbalance": (prices["up"]["ask_size"] - prices["down"]["ask_size"]) / size_sum,
        "slippage_5_up": vwap_up - prices["up"]["ask"],
        "slippage_5_down": vwap_down - prices["down"]["ask"],
        "mid_sum_minus_one": mid_up + mid_down - 1.0,
        "microstructure_eligible": True,
    })
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_working_set_bytes() -> int | None:
    import ctypes
    from ctypes import wintypes

    if not hasattr(ctypes, "WinDLL"):
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t), ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t), ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t), ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t), ("peak_pagefile_usage", ctypes.c_size_t),
            ("private_usage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    process = kernel32.GetCurrentProcess
    process.restype = ctypes.c_void_p
    get_memory_info = psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = (ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD)
    get_memory_info.restype = wintypes.BOOL
    if get_memory_info(process(), ctypes.byref(counters), counters.cb):
        return int(counters.peak_working_set_size)
    return None


def main() -> dict:
    source_paths = {
        "t60_snapshot": ROOT / CONFIG["sources"]["t60_snapshot_path"],
        "raw_base70_oof": ROOT / CONFIG["sources"]["raw_base70_oof_path"],
    }
    code_paths = {
        "config": CONFIG_PATH,
        "runner": Path(__file__),
        "snapshot_builder": ROOT / CONFIG["sources"]["snapshot_builder"],
        "snapshot_state_builder": ROOT / CONFIG["sources"]["snapshot_state_builder"],
        "market_probability_definition": ROOT / "run_btc_market_probability_experiment.py",
    }
    missing = [str(path) for path in (*source_paths.values(), *code_paths.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required saved development inputs are missing: " + ", ".join(missing))

    read_t0 = time.perf_counter()
    snapshots = _read_development(source_paths["t60_snapshot"], SNAPSHOT_COLUMNS)
    snapshot_read_seconds = time.perf_counter() - read_t0
    read_t0 = time.perf_counter()
    raw = _read_development(source_paths["raw_base70_oof"], RAW_COLUMNS)
    oof_read_seconds = time.perf_counter() - read_t0
    if snapshots.market_start_utc.duplicated().any() or raw.market_start_utc.duplicated().any():
        raise RuntimeError("Development T-60 snapshots and RAW OOF must each have unique market starts")
    raw["nominal_decision_at"] = pd.to_datetime(raw.nominal_decision_at, utc=True, errors="raise")
    raw = raw.sort_values("market_start_utc", kind="stable").reset_index(drop=True)
    feature_build_t0 = time.perf_counter()
    snapshot_map = {
        row.market_start_utc: row._asdict()
        for row in snapshots.itertuples(index=False)
    }
    records = [
        _feature_record(row, snapshot_map.get(row.market_start_utc))
        for row in raw.itertuples(index=False)
    ]
    table = pd.DataFrame.from_records(records)
    table["market_logit"] = np.nan
    valid_market = pd.to_numeric(table.p_market, errors="coerce").notna()
    table.loc[valid_market, "market_logit"] = market_probability._logit(table.loc[valid_market, "p_market"])
    feature_build_seconds = time.perf_counter() - feature_build_t0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_t0 = time.perf_counter()
    table.to_parquet(TABLE_PATH, index=False, compression="zstd")
    table_write_seconds = time.perf_counter() - write_t0

    feature_nonmissing = {
        name: int(pd.to_numeric(table[name], errors="coerce").notna().sum())
        for name in (*PROBABILITY_FEATURES, "market_logit", *MICROSTRUCTURE_FEATURES)
    }
    input_records = {
        name: {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for name, path in source_paths.items()
    }
    code_records = {
        name: {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for name, path in code_paths.items()
    }
    summary = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "feature_table_built_no_model_fit",
        "development_start_utc": START_UTC.isoformat(),
        "development_cutoff_exclusive_utc": CUTOFF_UTC.isoformat(),
        "raw_oof_rows": int(len(raw)),
        "t60_snapshot_rows": int(len(snapshots)),
        "raw_oof_rows_with_t60_snapshot": int(table.t60_snapshot_available.sum()),
        "raw_oof_rows_without_t60_snapshot": int((~table.t60_snapshot_available).sum()),
        "t60_entries_aligned_to_market_start_minus_60s": int(
            snapshots.entry_time_utc.eq(snapshots.market_start_utc - pd.Timedelta(seconds=60)).sum()
        ),
        "rows_with_p_market": feature_nonmissing["p_market"],
        "rows_with_all_six_microstructure_features": int(table.microstructure_eligible.sum()),
        "eligible_rows_by_protocol_period": {
            "fit_prefix_before_label_filter": int(
                (table.microstructure_eligible & table.market_start_utc.lt(
                    pd.Timestamp(CONFIG["model_comparison"]["fit_prefix_end_exclusive_utc"])
                )).sum()
            ),
            "evaluation_interval_before_label_filter": int(
                (table.microstructure_eligible & table.market_start_utc.ge(
                    pd.Timestamp(CONFIG["model_comparison"]["evaluation_start_utc"])
                )).sum()
            ),
        },
        "eligible_rows_by_month_before_label_filter": {
            str(month): int(count)
            for month, count in table.loc[table.microstructure_eligible].groupby(
                table.loc[table.microstructure_eligible, "market_start_utc"].dt.strftime("%Y-%m")
            ).size().items()
        },
        "feature_nonmissing_rows": feature_nonmissing,
        "feature_missing_fraction_of_raw_oof": {
            name: 1.0 - count / len(table) for name, count in feature_nonmissing.items()
        },
        "quality_flags": {
            name: int(table[name].fillna(False).astype(bool).sum())
            for name in (
                "has_full_snapshot", "up_quote_complemented", "down_quote_complemented",
                "archive_gap_after_up_book_init", "archive_gap_after_down_book_init",
                "up_ask_order_ambiguous", "down_ask_order_ambiguous",
                "up_bid_ask_crossed", "down_bid_ask_crossed",
                "up_bid_ask_locked", "down_bid_ask_locked", "depth_5usd_valid_both_sides",
            )
        },
        "event_order_diagnostics": {
            "rows_with_equal_source_timestamp_conflicts": int(
                snapshots.market_equal_source_timestamp_price_change_count.fillna(0).gt(0).sum()
            ),
            "rows_with_out_of_order_price_changes": int(
                snapshots.market_out_of_order_price_change_count.fillna(0).gt(0).sum()
            ),
            "rows_with_cross_side_late_price_changes": int(
                snapshots.market_cross_side_late_price_change_count.fillna(0).gt(0).sum()
            ),
            "rows_no_future_receive_event": int(snapshots.no_future_event_at_entry.fillna(False).sum()),
            "rows_no_future_source_event": int(snapshots.no_future_source_event_at_entry.fillna(False).sum()),
            "rows_with_valid_quote_flag": int(snapshots.quote_valid.fillna(False).sum()),
            "rows_with_source_hour_and_freshness_coverage": int(
                (snapshots.archive_entry_hour_available.fillna(False)
                 & snapshots.archive_freshness_window_available.fillna(False)).sum()
            ),
        },
        "snapshot_capabilities": {
            "snapshot_rows_in_development_interval": int(len(snapshots)),
            "up_best_bid_rows": int(snapshots.up_best_bid.notna().sum()),
            "up_best_ask_rows": int(snapshots.up_best_ask.notna().sum()),
            "down_best_bid_rows": int(snapshots.down_best_bid.notna().sum()),
            "down_best_ask_rows": int(snapshots.down_best_ask.notna().sum()),
            "up_best_ask_size_rows": int(snapshots.up_best_ask_size_shares.notna().sum()),
            "down_best_ask_size_rows": int(snapshots.down_best_ask_size_shares.notna().sum()),
            "native_both_token_books": int(snapshots.has_full_snapshot.fillna(False).sum()),
            "up_ask_levels_nonempty_rows": int(snapshots.up_ask_levels.map(lambda levels: len(levels) > 0 if levels is not None else False).sum()),
            "down_ask_levels_nonempty_rows": int(snapshots.down_ask_levels.map(lambda levels: len(levels) > 0 if levels is not None else False).sum()),
            "up_ask_levels_max_levels": int(snapshots.up_ask_levels.map(lambda levels: len(levels) if levels is not None else 0).max()),
            "down_ask_levels_max_levels": int(snapshots.down_ask_levels.map(lambda levels: len(levels) if levels is not None else 0).max()),
            "bid_level_arrays_serialized": False,
            "best_bid_size_fields_serialized": False,
            "source_book_event_count_rows": int(snapshots.market_book_snapshot_count.notna().sum()),
            "explicit_per_side_reset_flag_serialized": False,
            "up_ask_book_event_age_rows": int(snapshots.up_ask_age_seconds.notna().sum()),
            "down_ask_book_event_age_rows": int(snapshots.down_ask_age_seconds.notna().sum()),
            "bid_book_event_age_fields_serialized": False,
            "best_price_level_update_timestamps_serialized": False,
        },
        "output_table": {
            "path": str(TABLE_PATH.relative_to(ROOT)),
            "bytes": TABLE_PATH.stat().st_size,
            "sha256": _sha256(TABLE_PATH),
            "rows": int(len(table)),
            "columns": table.columns.tolist(),
            "contains_official_or_holdout_labels": False,
            "contains_t59_features": False,
        },
        "inputs": input_records,
        "source_code": code_records,
        "timing_seconds": {
            "read_t60_snapshot": snapshot_read_seconds,
            "read_raw_oof": oof_read_seconds,
            "build_feature_table": feature_build_seconds,
            "write_feature_table": table_write_seconds,
        },
        "peak_working_set_bytes": _peak_working_set_bytes(),
        "model_fitting_performed": False,
        "holdout_rows_or_labels_read": False,
        "archive_replayed_or_downloaded": False,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    versioned_manifest_path = ROOT / CONFIG["manifest_path"]
    versioned_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    versioned_manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
