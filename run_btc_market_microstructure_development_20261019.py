"""Fit the frozen development-only BTC/market microstructure interaction study."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import build_btc_market_microstructure_design_20261019 as design
import run_btc_market_base70_incremental_20261019 as existing
import run_btc_preopen_t59_reassessment as replay


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_market_microstructure_development_20261019.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["outputs"]["directory"]
REPORT_PATH = ROOT / CONFIG["outputs"]["report_path"]
MANIFEST_PATH = ROOT / CONFIG["outputs"]["manifest_path"]
START_UTC = pd.Timestamp(CONFIG["development_start_utc"])
CUTOFF_UTC = pd.Timestamp(CONFIG["development_cutoff_exclusive_utc"])
FIT_END_UTC = pd.Timestamp(CONFIG["fit"]["market_start_interval"].split(",")[1].strip(" ])"))
FIRST_TEST_DECISION_UTC = pd.Timestamp(CONFIG["fit"]["first_test_decision_utc"])
EVAL_START_UTC = pd.Timestamp(CONFIG["evaluation"]["market_start_interval"].split(",")[0].strip("[ "))
BASE_FEATURES = (
    "market_logit", "spread_up", "spread_down", "up_down_ask_depth_imbalance",
    "slippage_5_up", "slippage_5_down", "mid_sum_minus_one",
)
MICROSTRUCTURE_FEATURES = design.MICROSTRUCTURE_FEATURES
MAIN_FEATURES = (*BASE_FEATURES, "raw_minus_market_logit")
INTERACTION_FEATURES = tuple(
    f"raw_minus_market_logit_x_{name}" for name in MICROSTRUCTURE_FEATURES
)
QUALITY_VARIANTS = ("MARKET", "RAW", "C", "X")
ECONOMIC_VARIANTS = ("C", "X", "MARKET", "RAW")
EPSILON = 1e-6
BOOTSTRAP_REPLICATIONS = int(CONFIG["bootstrap"]["replications"])
BOOTSTRAP_BLOCK_DAYS = int(CONFIG["bootstrap"]["block_days"])
BOOTSTRAP_SEED = int(CONFIG["bootstrap"]["seed"])

EXECUTION_SNAPSHOT_COLUMNS = [
    "condition_id", "market_start_utc", "entry_time_utc", "fee_collection_mode",
    "fee_rate_bps", "fee_known", "fee_event_ns", "archive_source",
    "archive_entry_hour_available", "archive_freshness_window_available",
    "archive_window_hours_expected", "archive_window_hours_source_available",
    "archive_window_hours_processed", "archive_window_missing_source_hours",
    "archive_window_unprocessed_hours", "archive_gap_after_up_book_init",
    "archive_gap_after_down_book_init", "archive_gap_after_fee_update",
    "up_book_init_receive_ns", "down_book_init_receive_ns",
    "no_future_event_at_entry", "no_future_source_event_at_entry",
    "up_best_bid", "up_best_ask", "up_best_ask_size_shares", "up_ask_age_seconds",
    "up_quote_source_token_id", "up_quote_complemented", "up_ask_levels",
    "up_ask_order_ambiguous", "up_bbo_ask_comparable", "up_bbo_ask_mismatch",
    "up_bid_ask_crossed", "down_best_bid", "down_best_ask",
    "down_best_ask_size_shares", "down_ask_age_seconds", "down_quote_source_token_id",
    "down_quote_complemented", "down_ask_levels", "down_ask_order_ambiguous",
    "down_bbo_ask_comparable", "down_bbo_ask_mismatch", "down_bid_ask_crossed",
]
AUDIT_EXTRA_SNAPSHOT_COLUMNS = [
    "market_event_count", "market_book_snapshot_count", "market_first_event_utc",
    "market_last_event_utc", "archive_window_hours_expected",
    "archive_window_hours_source_available", "archive_window_hours_processed",
    "archive_window_missing_source_hours", "archive_window_unprocessed_hours",
    "archive_gap_after_fee_update", "fee_known", "fee_rate_bps",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def _read_development(path: Path, columns: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    field_type = dataset.schema.field("market_start_utc").type
    start = pa.scalar(START_UTC.to_pydatetime(), type=field_type)
    cutoff = pa.scalar(CUTOFF_UTC.to_pydatetime(), type=field_type)
    date_filter = (ds.field("market_start_utc") >= start) & (ds.field("market_start_utc") < cutoff)
    frame = dataset.to_table(columns=columns, filter=date_filter).to_pandas()
    frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True, errors="raise")
    return frame


def _record(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _as_bool(value) -> bool:
    return value is not None and not pd.isna(value) and bool(value)


def _snapshot_map(frame: pd.DataFrame) -> dict[str, dict]:
    if frame.condition_id.astype("string").str.lower().duplicated().any():
        raise RuntimeError("Execution snapshots contain duplicate condition IDs")
    return {
        str(row.condition_id).lower(): row._asdict()
        for row in frame.itertuples(index=False)
    }


def _archive_partition_audit() -> dict:
    sources = CONFIG["sources"]
    dependency_path = ROOT / sources["replay_dependencies_path"]
    archive_path = ROOT / sources["source_archive_manifest_path"]
    dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    event_parts = Path(archive["event_parts_dir"])
    first_hour = pd.Timestamp("2026-06-13T00:00:00Z")
    stop_hour = pd.Timestamp("2026-06-29T00:00:00Z")
    partitions = [
        row for row in dependency["partition_metadata"]
        if first_hour <= pd.Timestamp(row["path"][:13] + ":00:00Z") < stop_hour
    ]
    if len(partitions) != 384:
        raise RuntimeError(f"Expected 384 saved hourly partitions for June 13–28, found {len(partitions)}")
    counts = []
    for item in partitions:
        path = event_parts / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Saved event partition is missing: {path}")
        counts.append(pq.ParquetFile(path).metadata.num_rows)
    return {
        "start_utc": first_hour.isoformat(),
        "end_exclusive_utc": stop_hour.isoformat(),
        "partition_count": len(partitions),
        "dependency_sizes_bytes": {
            "minimum": min(item["size_bytes"] for item in partitions),
            "maximum": max(item["size_bytes"] for item in partitions),
            "total": sum(item["size_bytes"] for item in partitions),
        },
        "parquet_metadata_rows": {
            "total": int(sum(counts)),
            "empty_partitions": int(sum(count == 0 for count in counts)),
            "nonempty_partitions": int(sum(count > 0 for count in counts)),
        },
        "archive_rows_read_or_replayed": 0,
        "inspection": "Parquet metadata only; no event rows were read or replayed.",
    }


def _audit_availability() -> tuple[dict, pd.DataFrame]:
    sources = CONFIG["sources"]
    snapshot_columns = list(dict.fromkeys(
        design.SNAPSHOT_COLUMNS + AUDIT_EXTRA_SNAPSHOT_COLUMNS
        + ["archive_window_hours_expected", "archive_window_hours_source_available",
           "archive_window_hours_processed", "archive_window_missing_source_hours",
           "archive_window_unprocessed_hours"]
    ))
    snapshots = _read_development(ROOT / sources["t60_snapshots_path"], snapshot_columns)
    raw = _read_development(ROOT / sources["raw_oof_path"], design.RAW_COLUMNS)
    raw["nominal_decision_at"] = pd.to_datetime(raw.nominal_decision_at, utc=True, errors="raise")
    if snapshots.market_start_utc.duplicated().any() or raw.market_start_utc.duplicated().any():
        raise RuntimeError("Saved T-60 snapshots and RAW OOF must have unique market starts")
    snapshot_by_start = {row.market_start_utc: row._asdict() for row in snapshots.itertuples(index=False)}
    records = []
    for row in raw.itertuples(index=False):
        snapshot = snapshot_by_start.get(row.market_start_utc)
        record = design._feature_record(row, snapshot)
        raw_probability = design._number(row.p_model_raw)
        entry_aligned = False
        if snapshot is not None:
            entry = pd.Timestamp(snapshot["entry_time_utc"])
            if entry.tzinfo is None:
                entry = entry.tz_localize("UTC")
            else:
                entry = entry.tz_convert("UTC")
            entry_aligned = (
                entry == row.market_start_utc - pd.Timedelta(seconds=60)
                and row.nominal_decision_at == row.market_start_utc - pd.Timedelta(seconds=60)
            )
        source_event = snapshot is not None and design._number(snapshot.get("market_event_count")) is not None and float(snapshot["market_event_count"]) > 0
        source_coverage = snapshot is not None and all(
            design._is_true(snapshot.get(key))
            for key in (
                "archive_entry_hour_available", "archive_freshness_window_available",
                "no_future_event_at_entry", "no_future_source_event_at_entry",
            )
        ) and all(
            design._is_false(snapshot.get(key))
            for key in ("archive_gap_after_up_book_init", "archive_gap_after_down_book_init")
        )
        native_full = snapshot is not None and design._is_true(snapshot.get("has_full_snapshot"))
        native_pair = native_full and all(
            design._is_false(snapshot.get(key))
            for key in ("up_quote_complemented", "down_quote_complemented")
        )
        unambiguous = snapshot is not None and all(
            design._is_false(snapshot.get(key))
            for key in ("up_ask_order_ambiguous", "down_ask_order_ambiguous")
        )
        uncrossed = snapshot is not None and all(
            design._is_false(snapshot.get(key))
            for key in ("up_bid_ask_crossed", "down_bid_ask_crossed")
        )
        levels = {
            side: design._levels(snapshot.get(f"{side}_ask_levels")) if snapshot is not None else []
            for side in ("up", "down")
        }
        ask_aligned_side = {
            side: (
                snapshot is not None
                and design._number(snapshot.get(f"{side}_best_ask")) is not None
                and bool(levels[side])
                and abs(levels[side][0][0] - float(snapshot[f"{side}_best_ask"])) <= 1e-8
            )
            for side in ("up", "down")
        }
        depth_side = {
            side: design._vwap_for_gross_spend(levels[side]) is not None
            for side in ("up", "down")
        }
        all_features = all(
            record.get(name) is not None and math.isfinite(float(record[name]))
            for name in MICROSTRUCTURE_FEATURES
        )
        records.append({
            "market_start_utc": row.market_start_utc,
            "raw_valid": raw_probability is not None and 0.0 <= raw_probability <= 1.0,
            "snapshot_match": snapshot is not None,
            "t60_aligned": entry_aligned,
            "source_event_observed": source_event,
            "source_coverage_chronology": source_coverage,
            "native_full_snapshot": native_full,
            "native_both_books": native_pair,
            "unambiguous_asks": unambiguous,
            "not_crossed": uncrossed,
            "p_market_valid": record.get("p_market") is not None and math.isfinite(float(record["p_market"])),
            "ask_ladder_aligned": all(ask_aligned_side.values()),
            "up_ask_ladder_aligned": ask_aligned_side["up"],
            "down_ask_ladder_aligned": ask_aligned_side["down"],
            "gross_depth_5_both_sides": all(depth_side.values()),
            "up_gross_depth_5": depth_side["up"],
            "down_gross_depth_5": depth_side["down"],
            "six_features_finite": all_features,
            "builder_eligible": bool(record["microstructure_eligible"]),
            "has_full_snapshot_flag": native_full,
            "up_complemented": _as_bool(snapshot.get("up_quote_complemented")) if snapshot is not None else False,
            "down_complemented": _as_bool(snapshot.get("down_quote_complemented")) if snapshot is not None else False,
            "archive_entry_hour_available": _as_bool(snapshot.get("archive_entry_hour_available")) if snapshot is not None else False,
            "archive_freshness_window_available": _as_bool(snapshot.get("archive_freshness_window_available")) if snapshot is not None else False,
            "no_future_receive_event": _as_bool(snapshot.get("no_future_event_at_entry")) if snapshot is not None else False,
            "no_future_source_event": _as_bool(snapshot.get("no_future_source_event_at_entry")) if snapshot is not None else False,
            "up_gap_after_init": _as_bool(snapshot.get("archive_gap_after_up_book_init")) if snapshot is not None else False,
            "down_gap_after_init": _as_bool(snapshot.get("archive_gap_after_down_book_init")) if snapshot is not None else False,
            "up_ask_ambiguous": _as_bool(snapshot.get("up_ask_order_ambiguous")) if snapshot is not None else False,
            "down_ask_ambiguous": _as_bool(snapshot.get("down_ask_order_ambiguous")) if snapshot is not None else False,
            "up_crossed": _as_bool(snapshot.get("up_bid_ask_crossed")) if snapshot is not None else False,
            "down_crossed": _as_bool(snapshot.get("down_bid_ask_crossed")) if snapshot is not None else False,
            "up_locked": _as_bool(snapshot.get("up_bid_ask_locked")) if snapshot is not None else False,
            "down_locked": _as_bool(snapshot.get("down_bid_ask_locked")) if snapshot is not None else False,
            "quote_valid_flag": _as_bool(snapshot.get("quote_valid")) if snapshot is not None else False,
            "saved_fee_adjusted_depth_flag": _as_bool(snapshot.get("depth_5usd_valid_both_sides")) if snapshot is not None else False,
            "market_event_count": int(design._number(snapshot.get("market_event_count")) or 0) if snapshot is not None else 0,
            "market_book_snapshot_count": int(design._number(snapshot.get("market_book_snapshot_count")) or 0) if snapshot is not None else 0,
            "feature_record": record,
        })

    detail = pd.DataFrame.from_records(records)
    gate_names = (
        "raw_valid", "snapshot_match", "t60_aligned", "source_event_observed",
        "source_coverage_chronology", "native_both_books", "unambiguous_asks",
        "not_crossed", "p_market_valid", "ask_ladder_aligned",
        "gross_depth_5_both_sides", "six_features_finite",
    )
    funnel = []
    alive = pd.Series(True, index=detail.index)
    for name in gate_names:
        before = int(alive.sum())
        alive &= detail[name].fillna(False).astype(bool)
        after = int(alive.sum())
        funnel.append({"gate": name, "before": before, "excluded": before - after, "remaining": after})

    saved_table_path = ROOT / sources["feature_table_path"] if "feature_table_path" in sources else ROOT / CONFIG["sources"]["feature_table_path"]
    saved_table = pd.read_parquet(saved_table_path)
    saved_table["market_start_utc"] = pd.to_datetime(saved_table.market_start_utc, utc=True, errors="raise")
    if saved_table.market_start_utc.duplicated().any() or len(saved_table) != len(detail):
        raise RuntimeError("Saved design feature table does not align one-to-one with the no-label audit cohort")
    rebuilt = pd.DataFrame.from_records([item["feature_record"] for item in records])
    joined = saved_table.merge(rebuilt, on="market_start_utc", how="outer", suffixes=("_saved", "_rebuilt"), validate="one_to_one")
    if joined.microstructure_eligible_saved.fillna(False).astype(bool).ne(joined.microstructure_eligible_rebuilt.fillna(False).astype(bool)).any():
        raise RuntimeError("Saved feature table eligibility differs from the frozen builder logic")
    for name in ("p_raw", "p_market", *MICROSTRUCTURE_FEATURES):
        left = pd.to_numeric(joined[f"{name}_saved"], errors="coerce").to_numpy(dtype=np.float64)
        right = pd.to_numeric(joined[f"{name}_rebuilt"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True):
            raise RuntimeError(f"Saved feature table differs from builder recomputation for {name}")
    rebuilt_eligibility = detail.set_index("market_start_utc").builder_eligible.sort_index().astype(bool)
    saved_eligibility = saved_table.set_index("market_start_utc").microstructure_eligible.sort_index().fillna(False).astype(bool)
    if not rebuilt_eligibility.equals(saved_eligibility):
        raise RuntimeError("Sequential availability funnel does not reproduce the saved builder eligibility")

    flags = {
        "raw_probability_valid": detail.raw_valid,
        "matching_t60_snapshot": detail.snapshot_match,
        "exact_t60_alignment": detail.t60_aligned,
        "at_least_one_source_event_in_replay": detail.source_event_observed,
        "archive_entry_hour_available": detail.archive_entry_hour_available,
        "archive_freshness_window_available": detail.archive_freshness_window_available,
        "no_future_receive_event": detail.no_future_receive_event,
        "no_future_source_event": detail.no_future_source_event,
        "up_gap_after_init_flag_true": detail.up_gap_after_init,
        "down_gap_after_init_flag_true": detail.down_gap_after_init,
        "source_coverage_and_chronology_pass": detail.source_coverage_chronology,
        "full_two_book_snapshot": detail.native_full_snapshot,
        "both_books_native": detail.native_both_books,
        "up_quote_complemented": detail.up_complemented,
        "down_quote_complemented": detail.down_complemented,
        "up_ask_ambiguous": detail.up_ask_ambiguous,
        "down_ask_ambiguous": detail.down_ask_ambiguous,
        "up_book_crossed": detail.up_crossed,
        "down_book_crossed": detail.down_crossed,
        "up_book_locked": detail.up_locked,
        "down_book_locked": detail.down_locked,
        "quote_valid_flag": detail.quote_valid_flag,
        "valid_p_market": detail.p_market_valid,
        "up_best_ask_matches_first_level": detail.up_ask_ladder_aligned,
        "down_best_ask_matches_first_level": detail.down_ask_ladder_aligned,
        "gross_5usd_depth_up": detail.up_gross_depth_5,
        "gross_5usd_depth_down": detail.down_gross_depth_5,
        "saved_fee_adjusted_5usd_depth_flag": detail.saved_fee_adjusted_depth_flag,
        "all_six_features_finite": detail.six_features_finite,
    }
    independent = {
        name: {"true_count": int(series.fillna(False).astype(bool).sum()), "cohort_rows": len(detail)}
        for name, series in flags.items()
    }
    detail["day_utc"] = detail.market_start_utc.dt.floor("D")
    daily = detail.groupby("day_utc", sort=True).agg(
        raw_oof_rows=("raw_valid", "size"),
        raw_oof_valid=("raw_valid", "sum"),
        snapshot_matched=("snapshot_match", "sum"),
        t60_aligned=("t60_aligned", "sum"),
        source_events_observed=("source_event_observed", "sum"),
        source_coverage_chronology=("source_coverage_chronology", "sum"),
        native_both_books=("native_both_books", "sum"),
        unambiguous_asks=("unambiguous_asks", "sum"),
        uncrossed=("not_crossed", "sum"),
        p_market_valid=("p_market_valid", "sum"),
        ask_ladder_aligned=("ask_ladder_aligned", "sum"),
        gross_depth_5_both_sides=("gross_depth_5_both_sides", "sum"),
        six_features_finite=("six_features_finite", "sum"),
        builder_eligible=("builder_eligible", "sum"),
        up_crossed=("up_crossed", "sum"),
        down_crossed=("down_crossed", "sum"),
        up_ask_ambiguous=("up_ask_ambiguous", "sum"),
        down_ask_ambiguous=("down_ask_ambiguous", "sum"),
        full_snapshot=("native_full_snapshot", "sum"),
        saved_fee_adjusted_depth=("saved_fee_adjusted_depth_flag", "sum"),
    )
    calendar = pd.date_range(START_UTC.floor("D"), CUTOFF_UTC.floor("D") - pd.Timedelta(days=1), freq="D", tz="UTC")
    daily = daily.reindex(calendar, fill_value=0).rename_axis("day_utc").reset_index()
    sequential = pd.DataFrame(funnel)
    availability_dir = OUTPUT_DIR / "availability"
    _atomic_csv(daily, availability_dir / "daily_funnel.csv")
    _atomic_csv(sequential, availability_dir / "sequential_funnel.csv")
    _atomic_json(availability_dir / "independent_flags.json", independent)
    partition_audit = _archive_partition_audit()
    _atomic_json(availability_dir / "empty_source_partition_audit.json", partition_audit)
    audit = {
        "raw_oof_rows": int(len(detail)),
        "matched_snapshot_rows": int(detail.snapshot_match.sum()),
        "t60_aligned_rows": int(detail.t60_aligned.sum()),
        "final_feature_rows": int(detail.builder_eligible.sum()),
        "final_rows_match_saved_feature_table": True,
        "feature_table_recomputation_match": True,
        "sequential_funnel": funnel,
        "independent_flag_frequencies": independent,
        "daily_funnel_rows": int(len(daily)),
        "daily_funnel_path": (availability_dir / "daily_funnel.csv").relative_to(ROOT).as_posix(),
        "sequential_funnel_path": (availability_dir / "sequential_funnel.csv").relative_to(ROOT).as_posix(),
        "independent_flags_path": (availability_dir / "independent_flags.json").relative_to(ROOT).as_posix(),
        "empty_source_partition_audit": partition_audit,
        "source_event_semantics": "archive availability flags record hourly file processing; rows with zero observed replay events fail the separate event-presence gate.",
    }
    return audit, saved_table


def _label_release_times(frame: pd.DataFrame) -> pd.Series:
    starts = pd.to_datetime(frame.market_start_utc, utc=True, errors="raise")
    resolved = pd.to_datetime(frame.resolved_at_utc, utc=True, errors="coerce")
    minimum = starts + pd.Timedelta(minutes=5)
    release_base = resolved.where(resolved.ge(minimum), minimum)
    return release_base + pd.Timedelta(seconds=60)


def _training_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.common_input_available
        & frame.market_start_utc.ge(START_UTC)
        & frame.market_start_utc.lt(FIT_END_UTC)
        & frame.target_valid
        & frame.official_label_release_at_utc.lt(FIRST_TEST_DECISION_UTC)
    )


def _evaluation_input_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.common_input_available
        & frame.market_start_utc.ge(EVAL_START_UTC)
        & frame.market_start_utc.lt(CUTOFF_UTC)
    )


def _quality_label_mask(frame: pd.DataFrame) -> pd.Series:
    release = pd.to_datetime(frame.official_label_release_at_utc, utc=True, errors="coerce")
    return frame.target_valid & release.le(CUTOFF_UTC)


def _target_asof_cutoff(frame: pd.DataFrame) -> pd.Series:
    return frame.target_polymarket_up.where(frame.quality_label_available_by_cutoff, np.nan)


def _assert_same_market_index(expected: pd.DataFrame, observed: pd.DataFrame, context: str) -> None:
    if expected.condition_id.isna().any() or observed.condition_id.isna().any():
        raise RuntimeError(f"{context} has missing condition IDs")
    expected_ids = expected.condition_id.astype("string").str.lower()
    observed_ids = observed.condition_id.astype("string").str.lower()
    if expected_ids.duplicated().any() or observed_ids.duplicated().any():
        raise RuntimeError(f"{context} contains duplicate condition IDs")
    expected_starts = pd.to_datetime(expected.market_start_utc, utc=True, errors="raise")
    observed_starts = pd.to_datetime(observed.market_start_utc, utc=True, errors="raise")
    if (
        len(expected) != len(observed)
        or not expected_ids.reset_index(drop=True).equals(observed_ids.reset_index(drop=True))
        or not expected_starts.reset_index(drop=True).equals(observed_starts.reset_index(drop=True))
    ):
        raise RuntimeError(f"{context} do not align by condition ID and market start")


def _load_model_frame(feature_table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    index_path = ROOT / CONFIG["sources"]["official_market_index_path"]
    official = _read_development(
        index_path,
        ["condition_id", "market_start_utc", "resolved_at_utc", "target_polymarket_up",
         "up_token_id", "down_token_id", "order_min_size_shares_current"],
    )
    if official.market_start_utc.duplicated().any():
        raise RuntimeError("Official development market index has duplicate market starts")
    official["condition_id"] = official.condition_id.astype("string").str.lower()
    official["resolved_at_utc"] = pd.to_datetime(official.resolved_at_utc, utc=True, errors="coerce")
    official["target_polymarket_up"] = pd.to_numeric(official.target_polymarket_up, errors="coerce")
    feature_table["market_start_utc"] = pd.to_datetime(feature_table.market_start_utc, utc=True, errors="raise")
    feature_table["condition_id"] = feature_table.condition_id.astype("string").str.lower()
    frame = feature_table.merge(
        official, on="market_start_utc", how="left", validate="one_to_one", suffixes=("_feature", "_official"),
    )
    feature_ids = frame.condition_id_feature.astype("string").str.lower()
    official_ids = frame.condition_id_official.astype("string").str.lower()
    mismatched = feature_ids.notna() & official_ids.notna() & feature_ids.ne(official_ids)
    if mismatched.any():
        raise RuntimeError("T-60 feature snapshot and official market index condition IDs disagree")
    frame["condition_id"] = official_ids.fillna(feature_ids)
    frame["market_logit"] = design.market_probability._logit(
        pd.to_numeric(frame.p_market, errors="raise").to_numpy(dtype=np.float64)
    )
    frame["official_label_release_at_utc"] = _label_release_times(frame)
    finite_probability = (
        pd.to_numeric(frame.p_raw, errors="coerce").between(0.0, 1.0)
        & pd.to_numeric(frame.p_market, errors="coerce").between(0.0, 1.0)
    )
    finite_features = np.isfinite(frame.loc[:, MICROSTRUCTURE_FEATURES].to_numpy(dtype=np.float64)).all(axis=1)
    decision_aligned = pd.to_datetime(frame.t60_entry_time_utc, utc=True, errors="coerce").eq(
        frame.market_start_utc - pd.Timedelta(seconds=60)
    )
    eligible = frame.microstructure_eligible.fillna(False).astype(bool)
    frame["common_input_available"] = eligible & finite_probability & finite_features & decision_aligned
    frame["decision_day_utc"] = frame.market_start_utc.dt.floor("D")
    frame["target_valid"] = frame.target_polymarket_up.isin((0, 1)) & frame.resolved_at_utc.notna()
    frame["quality_label_available_by_cutoff"] = _quality_label_mask(frame)
    fit_mask = _training_mask(frame)
    train = frame.loc[fit_mask].copy().sort_values("market_start_utc", kind="stable").reset_index(drop=True)
    classes = sorted(pd.to_numeric(train.target_polymarket_up, errors="raise").astype(int).unique().tolist())
    if len(train) < int(CONFIG["fit"]["minimum_training_rows"]) or classes != [0, 1]:
        raise RuntimeError(f"Frozen fit requirements failed before any model fit: rows={len(train)}, classes={classes}")
    evaluation = frame.loc[_evaluation_input_mask(frame)].copy().sort_values(
        "market_start_utc", kind="stable",
    ).reset_index(drop=True)
    quality = evaluation.loc[evaluation.quality_label_available_by_cutoff].copy().reset_index(drop=True)
    if quality.target_polymarket_up.nunique() != 2:
        raise RuntimeError("Frozen quality sample must contain both official target classes")
    train_release = pd.to_datetime(train.official_label_release_at_utc, utc=True)
    fit_summary = {
        "training_rows": len(train),
        "training_class_counts": {str(key): int(value) for key, value in train.target_polymarket_up.astype(int).value_counts().sort_index().items()},
        "training_first_market_start_utc": train.market_start_utc.min().isoformat(),
        "training_last_market_start_utc": train.market_start_utc.max().isoformat(),
        "latest_training_label_release_utc": train_release.max().isoformat(),
        "first_test_decision_utc": FIRST_TEST_DECISION_UTC.isoformat(),
        "latest_training_label_strictly_precedes_first_test_decision": bool(train_release.max() < FIRST_TEST_DECISION_UTC),
        "evaluation_input_rows_independent_of_label_release": len(evaluation),
        "quality_rows_released_by_cutoff": len(quality),
        "quality_days_with_rows": int(quality.decision_day_utc.nunique()),
        "evaluation_days_with_input_rows": int(evaluation.decision_day_utc.nunique()),
        "evaluation_input_rows_after_2026_06_12": int(evaluation.market_start_utc.ge(pd.Timestamp("2026-06-13T00:00:00Z")).sum()),
        "evaluation_input_rows_2026_06_29": int(evaluation.market_start_utc.dt.date.eq(pd.Timestamp("2026-06-29").date()).sum()),
        "evaluation_input_rows_2026_06_30": int(evaluation.market_start_utc.dt.date.eq(pd.Timestamp("2026-06-30").date()).sum()),
    }
    if not fit_summary["latest_training_label_strictly_precedes_first_test_decision"]:
        raise RuntimeError("Training label release boundary is not strict")
    return frame, {"train": train, "evaluation": evaluation, "quality": quality, "summary": fit_summary}


def _logit(values) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(probability) - np.log1p(-probability)


def _transform_matrices(frame: pd.DataFrame, main_scaler: StandardScaler, interaction_scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    raw_main = frame.loc[:, MAIN_FEATURES].to_numpy(dtype=np.float64)
    standardized = main_scaler.transform(raw_main)
    interaction_raw = standardized[:, 7:8] * standardized[:, 1:7]
    interactions = interaction_scaler.transform(interaction_raw)
    control = standardized[:, :7]
    challenger = np.column_stack((standardized, interactions))
    return control, challenger


def _fit_models_and_predict(train: pd.DataFrame, evaluation: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    train_main = train.loc[:, MAIN_FEATURES].to_numpy(dtype=np.float64)
    train_y = train.target_polymarket_up.to_numpy(dtype=np.int8)
    main_scaler = StandardScaler()
    train_standardized = main_scaler.fit_transform(train_main)
    train_interaction_raw = train_standardized[:, 7:8] * train_standardized[:, 1:7]
    interaction_scaler = StandardScaler()
    train_interactions = interaction_scaler.fit_transform(train_interaction_raw)
    train_control = train_standardized[:, :7]
    train_challenger = np.column_stack((train_standardized, train_interactions))
    if train_control.shape[1] != 7 or train_challenger.shape[1] != 14:
        raise RuntimeError("Frozen model dimensions do not match C=7 and X=14")
    if not np.array_equal(train_control, train_challenger[:, :7]):
        raise RuntimeError("C and X training inputs do not share identical scaled common columns")

    model_c = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
        class_weight=None, fit_intercept=True,
    )
    model_x = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
        class_weight=None, fit_intercept=True,
    )
    with threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            model_c.fit(train_control, train_y)
            model_x.fit(train_challenger, train_y)
            eval_control, eval_challenger = _transform_matrices(evaluation, main_scaler, interaction_scaler)
            if not np.array_equal(eval_control, eval_challenger[:, :7]):
                raise RuntimeError("C and X evaluation inputs do not share identical scaled common columns")
            p_c = model_c.predict_proba(eval_control)[:, 1]
            p_x = model_x.predict_proba(eval_challenger)[:, 1]
    evaluation_predictions = evaluation[[
        "condition_id", "market_start_utc", "decision_day_utc", "official_label_release_at_utc",
        "quality_label_available_by_cutoff", "target_polymarket_up", "p_market", "p_raw",
    ]].copy()
    evaluation_predictions["target_asof_cutoff"] = _target_asof_cutoff(evaluation_predictions)
    evaluation_predictions["p_C"] = p_c
    evaluation_predictions["p_X"] = p_x
    # The release timestamp is retained for audit; the target is masked until release by cutoff.
    evaluation_predictions.drop(columns="target_polymarket_up", inplace=True)

    main_constant = [name for index, name in enumerate(MAIN_FEATURES) if np.ptp(train_main[:, index]) == 0.0]
    interaction_constant = [
        name for index, name in enumerate(INTERACTION_FEATURES)
        if np.ptp(train_interaction_raw[:, index]) == 0.0
    ]
    coefficient_names = {
        "C": list(BASE_FEATURES),
        "X": [*MAIN_FEATURES, *INTERACTION_FEATURES],
    }
    coefficients = {
        "C": {name: float(value) for name, value in zip(coefficient_names["C"], model_c.coef_[0])},
        "X": {name: float(value) for name, value in zip(coefficient_names["X"], model_x.coef_[0])},
    }
    fit_summary = {
        "fit_count": 2,
        "input_count_C": int(train_control.shape[1]),
        "input_count_X": int(train_challenger.shape[1]),
        "main_effects_scaler_fit_rows": len(train),
        "interaction_scaler_fit_rows": len(train),
        "main_effects_scaler_feature_order": list(MAIN_FEATURES),
        "interaction_feature_order": list(INTERACTION_FEATURES),
        "constant_main_effect_columns_retained": main_constant,
        "constant_interaction_columns_retained": interaction_constant,
        "main_scaler_mean": dict(zip(MAIN_FEATURES, main_scaler.mean_.astype(float).tolist())),
        "main_scaler_scale": dict(zip(MAIN_FEATURES, main_scaler.scale_.astype(float).tolist())),
        "interaction_scaler_mean": dict(zip(INTERACTION_FEATURES, interaction_scaler.mean_.astype(float).tolist())),
        "interaction_scaler_scale": dict(zip(INTERACTION_FEATURES, interaction_scaler.scale_.astype(float).tolist())),
        "model_c_intercept": float(model_c.intercept_[0]),
        "model_x_intercept": float(model_x.intercept_[0]),
        "coefficients_standardized": coefficients,
        "model_c_iterations": int(model_c.n_iter_[0]),
        "model_x_iterations": int(model_x.n_iter_[0]),
        "convergence_warnings": [
            str(item.message) for item in fit_warnings
            if issubclass(item.category, ConvergenceWarning)
        ],
        "other_fit_warnings": [
            str(item.message) for item in fit_warnings
            if not issubclass(item.category, ConvergenceWarning)
        ],
        "common_scaled_values_identical_in_train_and_evaluation": True,
    }
    bundle = {
        "protocol_version": int(CONFIG["protocol_version"]),
        "main_scaler": main_scaler,
        "interaction_scaler": interaction_scaler,
        "model_C": model_c,
        "model_X": model_x,
        "main_feature_order": list(MAIN_FEATURES),
        "interaction_feature_order": list(INTERACTION_FEATURES),
        "training_market_starts_utc": train.market_start_utc.astype(str).tolist(),
    }
    return bundle, evaluation_predictions, fit_summary


def _metric_values(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    if len(labels) == 0:
        return {"log_loss": None, "brier": None, "auc": None}
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    y = np.asarray(labels, dtype=np.float64)
    log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
    brier = float(np.mean(np.square(p - y)))
    auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None
    return {"log_loss": log_loss, "brier": brier, "auc": auc}


def _make_day_draws(days: pd.DatetimeIndex) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((BOOTSTRAP_REPLICATIONS, len(days)), dtype=np.int32)
    for replicate in range(BOOTSTRAP_REPLICATIONS):
        sampled = []
        while len(sampled) < len(days):
            start = int(rng.integers(0, len(days)))
            sampled.extend((start + offset) % len(days) for offset in range(BOOTSTRAP_BLOCK_DAYS))
        draws[replicate] = np.asarray(sampled[:len(days)], dtype=np.int32)
    return draws


def _interval(values: list[float]) -> dict:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"lower_95": None, "upper_95": None, "valid_replicates": 0}
    return {
        "lower_95": float(np.quantile(finite, 0.025)),
        "upper_95": float(np.quantile(finite, 0.975)),
        "valid_replicates": int(finite.size),
    }


def _quality_bootstrap(predictions: pd.DataFrame, draws: np.ndarray, days: pd.DatetimeIndex) -> dict:
    sample = predictions.loc[predictions.quality_label_available_by_cutoff].copy().reset_index(drop=True)
    labels = sample.target_asof_cutoff.to_numpy(dtype=np.int8)
    calendar_days = days.strftime("%Y-%m-%d").to_numpy()
    day_values = pd.to_datetime(sample.decision_day_utc, utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    positions = {
        day: np.flatnonzero(day_values == day)
        for day in calendar_days
    }
    probability_columns = {"MARKET": "p_market", "RAW": "p_raw", "C": "p_C", "X": "p_X"}
    probabilities = {name: sample[column].to_numpy(dtype=np.float64) for name, column in probability_columns.items()}
    variant_values = {name: {metric: [] for metric in ("log_loss", "brier", "auc")} for name in QUALITY_VARIANTS}
    contrast_names = {"X_MINUS_C": ("X", "C"), "C_MINUS_MARKET": ("C", "MARKET"), "X_MINUS_MARKET": ("X", "MARKET")}
    contrast_values = {name: {metric: [] for metric in ("log_loss", "brier", "auc")} for name in contrast_names}
    empty_replicates = 0
    one_class_replicates = 0
    for day_draw in draws:
        row_parts = [positions[calendar_days[index]] for index in day_draw]
        nonempty_parts = [part for part in row_parts if len(part)]
        if not nonempty_parts:
            empty_replicates += 1
            for values in variant_values.values():
                for metric in values:
                    values[metric].append(np.nan)
            for values in contrast_values.values():
                for metric in values:
                    values[metric].append(np.nan)
            continue
        chosen = np.concatenate(nonempty_parts)
        y = labels[chosen]
        one_class = np.unique(y).size < 2
        one_class_replicates += int(one_class)
        metrics = {
            name: _metric_values(y, probability[chosen])
            for name, probability in probabilities.items()
        }
        for name in QUALITY_VARIANTS:
            for metric in ("log_loss", "brier", "auc"):
                value = metrics[name][metric]
                variant_values[name][metric].append(np.nan if value is None else float(value))
        for contrast, (left, right) in contrast_names.items():
            for metric in ("log_loss", "brier", "auc"):
                a, b = metrics[left][metric], metrics[right][metric]
                contrast_values[contrast][metric].append(np.nan if a is None or b is None else float(a - b))
    return {
        "method": CONFIG["bootstrap"]["method"],
        "replications": int(len(draws)),
        "block_days": BOOTSTRAP_BLOCK_DAYS,
        "seed": BOOTSTRAP_SEED,
        "complete_calendar_days": int(len(days)),
        "calendar_includes_zero_row_days": True,
        "empty_replicates": empty_replicates,
        "one_class_replicates": one_class_replicates,
        "variant_intervals": {
            name: {metric: _interval(values) for metric, values in metrics.items()}
            for name, metrics in variant_values.items()
        },
        "paired_contrasts": {
            name: {metric: _interval(values) for metric, values in metrics.items()}
            for name, metrics in contrast_values.items()
        },
    }


def _quality_results(predictions: pd.DataFrame, days: pd.DatetimeIndex, draws: np.ndarray) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    sample = predictions.loc[predictions.quality_label_available_by_cutoff].copy()
    y = sample.target_asof_cutoff.to_numpy(dtype=np.int8)
    metrics = {
        "MARKET": _metric_values(y, sample.p_market.to_numpy(dtype=np.float64)),
        "RAW": _metric_values(y, sample.p_raw.to_numpy(dtype=np.float64)),
        "C": _metric_values(y, sample.p_C.to_numpy(dtype=np.float64)),
        "X": _metric_values(y, sample.p_X.to_numpy(dtype=np.float64)),
    }
    rows = []
    for name in QUALITY_VARIANTS:
        rows.append({"variant": name, "n": len(sample), "observed_days": int(sample.decision_day_utc.nunique()), **metrics[name]})
    monthly = []
    sample["month_utc"] = pd.to_datetime(sample.market_start_utc, utc=True).dt.strftime("%Y-%m")
    for month, group in sample.groupby("month_utc", sort=True):
        labels = group.target_asof_cutoff.to_numpy(dtype=np.int8)
        for name, column in (("MARKET", "p_market"), ("RAW", "p_raw"), ("C", "p_C"), ("X", "p_X")):
            monthly.append({
                "month_utc": month, "variant": name, "n": len(group),
                "observed_days": int(group.decision_day_utc.nunique()),
                **_metric_values(labels, group[column].to_numpy(dtype=np.float64)),
            })
    bootstrap = _quality_bootstrap(predictions, draws, days)
    return {"overall": rows, "bootstrap": bootstrap}, pd.DataFrame(monthly), pd.DataFrame(rows)


def _load_execution_snapshots(condition_ids: set[str]) -> tuple[dict, dict]:
    sources = CONFIG["sources"]
    snapshots = []
    for key in ("t60_snapshots_path", "t59_snapshots_path"):
        frame = _read_development(ROOT / sources[key], EXECUTION_SNAPSHOT_COLUMNS)
        frame["condition_id"] = frame.condition_id.astype("string").str.lower()
        frame["entry_time_utc"] = pd.to_datetime(frame.entry_time_utc, utc=True, errors="coerce")
        frame = frame.loc[frame.condition_id.isin(condition_ids)].copy()
        if frame.condition_id.duplicated().any():
            raise RuntimeError(f"{key} contains duplicate condition IDs")
        snapshots.append(frame)
    t60, t59 = snapshots
    if not t60.empty and not t60.entry_time_utc.eq(t60.market_start_utc - pd.Timedelta(seconds=60)).all():
        raise RuntimeError("A saved T-60 execution decision snapshot is not aligned to market start minus 60 seconds")
    if not t59.empty and not t59.entry_time_utc.eq(t59.market_start_utc - pd.Timedelta(seconds=59)).all():
        raise RuntimeError("A saved T-59 execution snapshot is not aligned to market start minus 59 seconds")
    return _snapshot_map(t60), _snapshot_map(t59)


def _mask_ledger_to_cutoff(trades: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    result = trades.copy()
    if result.empty:
        return result
    release_by_id = evaluation.set_index("condition_id").official_label_release_at_utc
    target_by_id = evaluation.set_index("condition_id").target_valid
    release = pd.to_datetime(result.condition_id.map(release_by_id), utc=True, errors="coerce")
    target_valid = result.condition_id.map(target_by_id).fillna(False).astype(bool)
    unresolved = ~target_valid.to_numpy() | release.isna().to_numpy() | release.gt(CUTOFF_UTC).to_numpy()
    result["settlement_status"] = result.settlement_status.astype("string")
    result.loc[unresolved, "settlement_status"] = "unresolved_at_cutoff"
    for column in ("outcome_up", "won", "payout_usd", "net_pnl_usd", "settled_at_utc", "cash_release_at_utc"):
        if column in result:
            if column in {"outcome_up", "won"}:
                result[column] = result[column].astype("boolean")
                result.loc[unresolved, column] = pd.NA
            elif column in {"settled_at_utc", "cash_release_at_utc"}:
                result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
                result.loc[unresolved, column] = pd.NaT
            else:
                result[column] = pd.to_numeric(result[column], errors="coerce").astype("float64")
                result.loc[unresolved, column] = np.nan
    return result


def _load_raw_ledger(evaluation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sources = CONFIG["sources"]
    ids = set(evaluation.condition_id.dropna().astype(str).str.lower())
    decision_columns = [
        "condition_id", "market_start_utc", "decision_time_utc", "entry_time_utc",
        "prediction_probability_up", "chosen_side", "decision", "skip_reason",
        "net_ev_fraction_at_decision", "chosen_side_ev_usd", "execution_ev_usd_at_t59",
        "gross_usd", "cash_debit_usd",
    ]
    trade_columns = [
        "condition_id", "market_start_utc", "decision_time_utc", "entry_time_utc",
        "prediction_probability_up", "chosen_side", "decision", "skip_reason",
        "net_ev_fraction_at_decision", "chosen_side_ev_usd", "execution_ev_usd_at_t59",
        "gross_usd", "cash_debit_usd", "execution_source", "depth_known",
        "execution_assumption", "fill_status", "fee_usd", "fee_cash_usd", "fee_shares",
        "fee_usd_entry_equivalent", "fee_raw_usd_equivalent", "gross_shares", "net_shares",
        "vwap", "outcome_up", "won", "payout_usd", "net_pnl_usd", "settlement_status",
        "settled_at_utc", "cash_release_at_utc", "fee_charge_unit", "fee_rule_id",
    ]
    decisions = pd.read_parquet(ROOT / sources["raw_ledger_decisions_path"], columns=decision_columns)
    trades = pd.read_parquet(ROOT / sources["raw_ledger_trades_path"], columns=trade_columns)
    for frame in (decisions, trades):
        frame["condition_id"] = frame.condition_id.astype("string").str.lower()
    decisions = decisions.loc[decisions.condition_id.isin(ids)].copy()
    trades = trades.loc[trades.condition_id.isin(ids)].copy()
    if decisions.condition_id.duplicated().any() or set(decisions.condition_id) != ids:
        raise RuntimeError("Existing RAW decision ledger does not cover the exact shared input calendar")
    expected = evaluation.set_index("condition_id")
    observed = decisions.set_index("condition_id")
    expected_probability = pd.to_numeric(expected.loc[observed.index, "p_raw"], errors="raise").to_numpy(dtype=np.float64)
    observed_probability = pd.to_numeric(observed.prediction_probability_up, errors="raise").to_numpy(dtype=np.float64)
    if not np.allclose(expected_probability, observed_probability, rtol=0.0, atol=1e-12):
        raise RuntimeError("Existing RAW ledger probabilities do not match saved BASE70 OOF on the shared calendar")
    for column, offset in (("decision_time_utc", -60), ("entry_time_utc", -59)):
        observed_time = pd.to_datetime(decisions[column], utc=True, errors="raise")
        expected_time = pd.to_datetime(decisions.market_start_utc, utc=True) + pd.Timedelta(seconds=offset)
        if not observed_time.eq(expected_time).all():
            raise RuntimeError(f"Existing RAW ledger violates the {column} timestamp contract")
    decisions["market_start_utc"] = pd.to_datetime(decisions.market_start_utc, utc=True, errors="raise")
    trades["market_start_utc"] = pd.to_datetime(trades.market_start_utc, utc=True, errors="raise")
    decisions["decision_time_utc"] = pd.to_datetime(decisions.decision_time_utc, utc=True, errors="raise")
    decisions["entry_time_utc"] = pd.to_datetime(decisions.entry_time_utc, utc=True, errors="raise")
    trades["decision_time_utc"] = pd.to_datetime(trades.decision_time_utc, utc=True, errors="raise")
    trades["entry_time_utc"] = pd.to_datetime(trades.entry_time_utc, utc=True, errors="raise")
    decisions.sort_values("market_start_utc", kind="stable", inplace=True, ignore_index=True)
    expected_calendar = evaluation.loc[:, ["condition_id", "market_start_utc"]].sort_values(
        "market_start_utc", kind="stable", ignore_index=True,
    )
    _assert_same_market_index(expected_calendar, decisions, "RAW ledger and common evaluation calendar")
    trades = _mask_ledger_to_cutoff(trades, evaluation)
    audit = {
        "source_decisions_path": sources["raw_ledger_decisions_path"],
        "source_trades_path": sources["raw_ledger_trades_path"],
        "shared_decision_rows": int(len(decisions)),
        "shared_fill_rows_before_cutoff_mask": int(len(trades)),
        "probabilities_match_saved_raw_oof_on_shared_calendar": True,
        "decision_and_entry_times_match_t60_t59_contract": True,
        "full_previous_ledger_pnl_aggregated": False,
        "cutoff_mask_applied_only_after_decision_ledger_sample_restriction": True,
    }
    return decisions, trades, audit


def _sum(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _economics_summary(variant: str, decisions: pd.DataFrame, trades: pd.DataFrame) -> dict:
    settled = trades.loc[trades.settlement_status.eq("settled")].copy() if len(trades) else trades.copy()
    unresolved = trades.loc[~trades.settlement_status.eq("settled")].copy() if len(trades) else trades.copy()
    cashflow = existing._cashflow_ledger_summary(trades)
    independent_audit = existing._independent_cashflow_audit(trades)
    intents = pd.to_numeric(decisions.get("net_ev_fraction_at_decision", pd.Series(dtype=float)), errors="coerce")
    result = {
        "variant": variant,
        "decision_rows": int(len(decisions)),
        "decision_days_with_rows": int(pd.to_datetime(decisions.decision_time_utc, utc=True).dt.floor("D").nunique()) if len(decisions) else 0,
        "intent_count_at_or_above_1pct_net_ev": int(intents.ge(0.01).sum()),
        "fill_count": int(len(trades)),
        "full_fills": int(trades.fill_status.eq("full").sum()) if len(trades) and "fill_status" in trades else 0,
        "partial_fills": int(trades.fill_status.eq("partial").sum()) if len(trades) and "fill_status" in trades else 0,
        "settlement_count_by_cutoff": int(len(settled)),
        "unresolved_fill_count": int(len(unresolved)),
        "settled_gross_notional_usd": cashflow["gross_notional_usd"],
        "all_fill_gross_notional_usd": cashflow["all_fill_gross_notional_usd"],
        "settled_cash_debit_usd": cashflow["settled_cash_outflow_usd"],
        "all_fill_cash_debit_usd": cashflow["all_fill_cash_outflow_usd"],
        "unresolved_cash_debit_usd": cashflow["unresolved_cash_outflow_usd"],
        "settled_cash_fees_usd": cashflow["cash_fees_paid_usd"],
        "all_fill_cash_fees_usd": cashflow["all_fill_cash_fees_paid_usd"],
        "settled_outcome_share_fees_shares": _sum(settled.loc[settled.fee_charge_unit.eq("outcome shares")], "fee_shares") if len(settled) else 0.0,
        "all_fill_outcome_share_fees_shares": _sum(trades.loc[trades.fee_charge_unit.eq("outcome shares")], "fee_shares") if len(trades) else 0.0,
        "settled_outcome_share_fee_entry_equivalent_usd": cashflow["share_fee_usd_entry_equivalent_informational"],
        "settled_net_pnl_usd": cashflow["cashflow_pnl_usd"],
        "pnl_per_settled_notional": cashflow["pnl_per_notional"],
        "pnl_per_settled_cash_debit": (
            cashflow["cashflow_pnl_usd"] / cashflow["settled_cash_outflow_usd"]
            if cashflow["settled_cash_outflow_usd"] else None
        ),
        "forecast_net_ev_on_same_settled_fills_usd": _sum(settled, "chosen_side_ev_usd"),
        "execution_net_ev_at_t59_on_same_settled_fills_usd": _sum(settled, "execution_ev_usd_at_t59"),
        "pnl_minus_t60_ev_same_settled_fills_usd": cashflow["cashflow_pnl_usd"] - _sum(settled, "chosen_side_ev_usd"),
        "accounting_summary": cashflow,
        "independent_decimal_cashflow_audit": independent_audit,
    }
    return result


def _daily_pnl(trades_by_variant: dict[str, pd.DataFrame], days: pd.DatetimeIndex) -> pd.DataFrame:
    daily = pd.DataFrame({"day_utc": days})
    day_index = {day: index for index, day in enumerate(days)}
    for variant, trades in trades_by_variant.items():
        values = np.zeros(len(days), dtype=np.float64)
        if len(trades):
            settled = trades.loc[trades.settlement_status.eq("settled")].copy()
            if len(settled):
                settled["day_utc"] = pd.to_datetime(settled.decision_time_utc, utc=True).dt.floor("D")
                for day, value in pd.to_numeric(settled.net_pnl_usd, errors="coerce").groupby(settled.day_utc).sum().items():
                    if day in day_index:
                        values[day_index[day]] = float(value)
        daily[f"{variant}_net_pnl_usd"] = values
    daily["X_MINUS_C_net_pnl_usd"] = daily.X_net_pnl_usd - daily.C_net_pnl_usd
    return daily


def _pnl_bootstrap(daily: pd.DataFrame, draws: np.ndarray) -> dict:
    differences = daily.X_MINUS_C_net_pnl_usd.to_numpy(dtype=np.float64)
    means = np.asarray([differences[row].mean() for row in draws], dtype=np.float64)
    return {
        "calendar_days": int(len(daily)),
        "zero_trade_days_included": True,
        "mean_daily_pnl_difference_X_minus_C_usd": {
            "point_estimate": float(differences.mean()), **_interval(means.tolist()),
        },
        "total_pnl_difference_X_minus_C_usd": {
            "point_estimate": float(differences.sum()),
            **_interval((means * len(daily)).tolist()),
        },
    }


def _monthly_economics(decisions_by_variant: dict[str, pd.DataFrame], trades_by_variant: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for variant in ECONOMIC_VARIANTS:
        decisions = decisions_by_variant[variant].copy()
        trades = trades_by_variant[variant].copy()
        months = sorted(pd.to_datetime(decisions.decision_time_utc, utc=True).dt.strftime("%Y-%m").unique()) if len(decisions) else []
        if len(decisions):
            decisions["month_utc"] = pd.to_datetime(decisions.decision_time_utc, utc=True).dt.strftime("%Y-%m")
        if len(trades):
            trades["month_utc"] = pd.to_datetime(trades.decision_time_utc, utc=True).dt.strftime("%Y-%m")
        for month in months:
            md = decisions.loc[decisions.month_utc.eq(month)]
            mt = trades.loc[trades.month_utc.eq(month)] if len(trades) else trades
            settled = mt.loc[mt.settlement_status.eq("settled")] if len(mt) else mt
            rows.append({
                "month_utc": month, "variant": variant,
                "decision_rows": len(md), "decision_days": int(pd.to_datetime(md.decision_time_utc, utc=True).dt.floor("D").nunique()),
                "intents_at_1pct": int(pd.to_numeric(md.get("net_ev_fraction_at_decision", pd.Series(index=md.index, dtype=float)), errors="coerce").ge(0.01).sum()),
                "fills": len(mt), "settlements": len(settled), "unresolved_fills": len(mt) - len(settled),
                "settled_notional_usd": _sum(settled, "gross_usd"),
                "all_fill_notional_usd": _sum(mt, "gross_usd"),
                "settled_cash_debit_usd": _sum(settled, "cash_debit_usd"),
                "settled_cash_fees_usd": _sum(settled, "fee_cash_usd"),
                "settled_share_fee_shares": _sum(settled.loc[settled.fee_charge_unit.eq("outcome shares")], "fee_shares") if len(settled) else 0.0,
                "net_pnl_usd": _sum(settled, "net_pnl_usd"),
                "pnl_per_notional": _sum(settled, "net_pnl_usd") / _sum(settled, "gross_usd") if _sum(settled, "gross_usd") else None,
                "pnl_per_cash_debit": _sum(settled, "net_pnl_usd") / _sum(settled, "cash_debit_usd") if _sum(settled, "cash_debit_usd") else None,
                "forecast_ev_same_settled_fills_usd": _sum(settled, "chosen_side_ev_usd"),
            })
    return pd.DataFrame(rows)


def _concentration(daily: pd.DataFrame, trades_by_variant: dict[str, pd.DataFrame]) -> dict:
    result = {}
    for variant in ECONOMIC_VARIANTS:
        column = f"{variant}_net_pnl_usd"
        ordered = daily.sort_values(column, ascending=False, kind="stable").head(5)
        total = float(daily[column].sum())
        trade_days = 0
        if len(trades_by_variant[variant]):
            trade_days = int(pd.to_datetime(trades_by_variant[variant].decision_time_utc, utc=True).dt.floor("D").nunique())
        result[variant] = {
            "total_pnl_usd": total,
            "calendar_days": int(len(daily)),
            "trade_days": trade_days,
            "no_trade_days": int(len(daily) - trade_days),
            "top_five_days_pnl_usd": float(ordered[column].sum()),
            "top_five_share_of_total_pnl": float(ordered[column].sum() / total) if total else None,
            "top_five_days": [
                {"day_utc": row.day_utc.isoformat(), "net_pnl_usd": float(getattr(row, column))}
                for row in ordered.itertuples(index=False)
            ],
        }
    return result


def _format(value, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str], labels: dict[str, str] | None = None) -> list[str]:
    labels = labels or {}
    header = "| " + " | ".join(labels.get(column, column) for column in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, divider]
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        cells = []
        for value in row:
            if value is None or pd.isna(value):
                cells.append("—")
            elif isinstance(value, (float, np.floating)):
                cells.append(_format(value))
            else:
                cells.append(f"{value:,}" if isinstance(value, (int, np.integer)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _write_report(
    audit: dict, fit_summary: dict, fit_sample: dict, quality: dict, quality_monthly: pd.DataFrame,
    economics: dict, economics_monthly: pd.DataFrame, daily_pnl: pd.DataFrame,
    pnl_bootstrap: dict, concentration: dict, run_metrics: dict,
) -> None:
    funnel = pd.DataFrame(audit["sequential_funnel"])
    daily_availability = pd.read_csv(OUTPUT_DIR / "availability/daily_funnel.csv")
    independent = pd.DataFrame([
        {"flag": name, **value}
        for name, value in audit["independent_flag_frequencies"].items()
    ])
    quality_overall = pd.DataFrame(quality["overall"])
    quality_boot = quality["bootstrap"]
    quality_contrasts = []
    for contrast in ("X_MINUS_C", "C_MINUS_MARKET", "X_MINUS_MARKET"):
        row = {"contrast": contrast}
        for metric in ("log_loss", "brier", "auc"):
            value = quality_boot["paired_contrasts"][contrast][metric]
            row[f"{metric}_ci"] = f"[{_format(value['lower_95'])}, {_format(value['upper_95'])}]"
            row[f"{metric}_valid"] = value["valid_replicates"]
        quality_contrasts.append(row)
    economics_rows = []
    for variant in ECONOMIC_VARIANTS:
        row = economics[variant]
        economics_rows.append({
            "variant": variant, "decision_rows": row["decision_rows"],
            "intents": row["intent_count_at_or_above_1pct_net_ev"], "fills": row["fill_count"],
            "settled": row["settlement_count_by_cutoff"], "unresolved": row["unresolved_fill_count"],
            "notional": row["settled_gross_notional_usd"], "cash_debit": row["settled_cash_debit_usd"],
            "cash_fee": row["settled_cash_fees_usd"], "share_fee_shares": row["settled_outcome_share_fees_shares"],
            "pnl": row["settled_net_pnl_usd"], "pnl_notional": row["pnl_per_settled_notional"],
            "pnl_debit": row["pnl_per_settled_cash_debit"],
            "ev_same_settled": row["forecast_net_ev_on_same_settled_fills_usd"],
        })
    daily_columns = [
        "day_utc", "raw_oof_rows", "snapshot_matched", "source_events_observed",
        "source_coverage_chronology", "native_both_books", "unambiguous_asks",
        "uncrossed", "p_market_valid", "builder_eligible",
    ]
    report = [
        "# Development: RAW – rynek – mikrostruktura — 2026-10-19",
        "",
        "## Zakres i zamrożona korekta protokołu",
        "",
        "To eksploracyjny test development z hipotezą pochodzącą z wcześniejszych analiz. Holdout nie był odczytywany ani oceniany. Wersję 2 zapisano przed fitami: kontrola C ma 7 wejść, a challenger X ma te same wejścia, d = logit(p_RAW) − logit(p_market) i sześć ustalonych interakcji d × z_j (łącznie 14). Addytywnej wersji z projektu nie dopasowano.",
        "",
        "Oba modele korzystają ze wspólnego StandardScaler dopasowanego na siedmiu bazowych wejściach i d wyłącznie na treningu. Interakcje to iloczyny wystandaryzowanych d i z_j; osobny scaler interakcji jest również dopasowany tylko na treningu. Stałe kolumny zachowano, a StandardScaler mapuje je na zero. Wspólne siedem wejść C i X jest identyczne w treningu i ocenie.",
        "",
        f"Fit: jeden prefiks od 15 kwietnia do startów przed 15 maja; release etykiety musi poprzedzać pierwszą decyzję testową T−60 ({FIRST_TEST_DECISION_UTC.isoformat()}). Ocena obejmuje starty od 15 maja do końca developmentu, bez refitów. Wymagane minimum to 1 000 etykiet i obie klasy.",
        "",
        "## Audyt dostępności bez etykiet",
        "",
        f"Lejek zaczyna się od {audit['raw_oof_rows']:,} wierszy RAW OOF; {audit['matched_snapshot_rows']:,} ma dopasowany snapshot T−60, a {audit['final_feature_rows']:,} przechodzi wszystkie warunki sześciu cech. Ponowne przeliczenie buildera zgadza się z zapisanym feature table pod względem indeksów i wartości.",
        "",
        "### Kolejne wykluczenia",
        "",
        *_markdown_table(
            funnel, ["gate", "before", "excluded", "remaining"],
            {"gate": "Warunek", "before": "Przed", "excluded": "Wykluczono", "remaining": "Pozostało"},
        ),
        "",
        "### Niezależne częstości flag",
        "",
        *_markdown_table(
            independent, ["flag", "true_count", "cohort_rows"],
            {"flag": "Flaga", "true_count": "Prawda", "cohort_rows": "Kohorta"},
        ),
        "",
        f"Dzienny lejek dla pełnego kalendarza 15 kwietnia–30 czerwca znajduje się w {audit['daily_funnel_path']}. Poniżej pokazano dzienne liczebności kluczowych etapów.",
        "",
        *_markdown_table(
            daily_availability, daily_columns,
            {
                "day_utc": "Dzień UTC", "raw_oof_rows": "RAW", "snapshot_matched": "Snapshot T−60",
                "source_events_observed": "Zdarzenia", "source_coverage_chronology": "Pokrycie/chronologia",
                "native_both_books": "Dwie księgi", "unambiguous_asks": "Ask jednoznaczne",
                "uncrossed": "Nie crossed", "p_market_valid": "p_market", "builder_eligible": "Końcowe",
            },
        ),
        "",
        "Flagi dostępności godziny i freshness oznaczają, że partycja została odnaleziona lub przetworzona; nie dowodzą, że zawiera zdarzenia. Obecność zdarzeń sprawdzono osobno.",
        "",
        "W zapisanym archiwum wszystkie 384 partycje godzinowe od 13 do 28 czerwca są puste według metadanych Parquet (łącznie 0 wierszy). Snapshoty T−60 z tego okresu mają market_event_count = 0 i brak bid/ask, więc z zachowanych zdarzeń nie da się odtworzyć stanu. To brak danych źródłowych, a nie dowód błędnej rekonstrukcji. Po 12 czerwca pozostało "
        f"{fit_sample['evaluation_input_rows_after_2026_06_12']:,} kwalifikujących się wejść testowych; 29 czerwca było ich {fit_sample['evaluation_input_rows_2026_06_29']}, a 30 czerwca {fit_sample['evaluation_input_rows_2026_06_30']}.",
        "",
        f"Kod rekonstrukcji grupuje zdarzenia po czasie odbioru i pilnuje porządku czasu źródłowego w obrębie tokenu i strony. Crossing oznacza best_bid > best_ask, locked oznacza równość; spóźnione zmiany cross-side są wykrywane i stosowane, jeśli zachowują porządek danej strony. Sam crossing nie dowodzi błędu. Istniejący quote_valid odrzuca crossed books; walidacji nie poluzowano. Niezależnie crossing wystąpił {audit['independent_flag_frequencies']['up_book_crossed']['true_count']:,} razy dla UP i {audit['independent_flag_frequencies']['down_book_crossed']['true_count']:,} dla DOWN.",
        "",
        "Flaga depth_5usd_valid_both_sides z replayu kontroluje wykonanie wraz z dostępnością historycznej opłaty. Cechy slippage sprawdzono niezależnie na zapisanych drabinkach brutto: poziomy muszą obsłużyć pełne $5. Opłaty zachowano jako osobną część audytu ekonomii.",
        "",
        "## Dopasowanie i jakość prognoz",
        "",
        f"Trening obejmował {fit_sample['training_rows']:,} wierszy, klasy {fit_sample['training_class_counts']}; ostatni release etykiety nastąpił {fit_sample['latest_training_label_release_utc']}. Ocena ma {fit_sample['evaluation_input_rows_independent_of_label_release']:,} wierszy zakwalifikowanych wyłącznie po wejściach. Do oceny jakości release do cutoffu uzyskano {fit_sample['quality_rows_released_by_cutoff']:,} etykiet z {fit_sample['quality_days_with_rows']} dni. Wspólna próba obejmuje też końcówkę czerwca.",
        "",
        *_markdown_table(
            quality_overall, ["variant", "n", "observed_days", "log_loss", "brier", "auc"],
            {"variant": "Wariant", "n": "N", "observed_days": "Dni z obserwacjami", "log_loss": "Log-loss", "brier": "Brier", "auc": "AUC"},
        ),
        "",
        "Miesięczne wyniki na tej samej próbie:",
        "",
        *_markdown_table(
            quality_monthly, ["month_utc", "variant", "n", "observed_days", "log_loss", "brier", "auc"],
            {"month_utc": "Miesiąc UTC", "variant": "Wariant", "n": "N", "observed_days": "Dni", "log_loss": "Log-loss", "brier": "Brier", "auc": "AUC"},
        ),
        "",
        "Sparowane 95% przedziały bootstrapu bloków 3-dniowych (ujemna różnica log-loss/Brier oznacza poprawę po lewej stronie):",
        "",
        *_markdown_table(
            pd.DataFrame(quality_contrasts),
            ["contrast", "log_loss_ci", "log_loss_valid", "brier_ci", "brier_valid", "auc_ci", "auc_valid"],
            {"contrast": "Kontrast", "log_loss_ci": "Log-loss 95% CI", "log_loss_valid": "Ważne", "brier_ci": "Brier 95% CI", "brier_valid": "Ważne", "auc_ci": "AUC 95% CI", "auc_valid": "Ważne"},
        ),
        "",
        f"Bootstrap użył pełnego kalendarza {quality_boot['complete_calendar_days']} dni od 15 maja do 30 czerwca, w tym dni bez kwalifikujących się wierszy. Replikacje: {quality_boot['replications']:,}; puste: {quality_boot['empty_replicates']}; jednoklasowe: {quality_boot['one_class_replicates']} (zachowane dla log-loss/Brier, AUC oznaczone jako niezdefiniowane). Głównym kontrastem jest X−C; X−MARKET porównuje challenger bezpośrednio z benchmarkiem rynku.",
        "",
        "## Ekonomia na wspólnym kalendarzu",
        "",
        "Każdy wariant podejmuje własne decyzje na tych samych wejściowych wierszach. Warianty C, X i MARKET używają wyłącznie T−60 do wyboru strony i progu EV netto co najmniej 1%; snapshot T−59 służy do wykonania wybranej strony. RAW pochodzi z istniejącego ledgeru ograniczonego najpierw do wspólnego kalendarza. Wyniki po cutoffie pozostają nierozliczone.",
        "",
        *_markdown_table(
            pd.DataFrame(economics_rows),
            ["variant", "decision_rows", "intents", "fills", "settled", "unresolved", "notional", "cash_debit", "cash_fee", "share_fee_shares", "pnl", "pnl_notional", "pnl_debit", "ev_same_settled"],
            {"variant": "Wariant", "decision_rows": "Decyzje", "intents": "EV ≥1%", "fills": "Wykonania", "settled": "Rozliczono", "unresolved": "Nierozliczone", "notional": "Notional settled $", "cash_debit": "Debet settled $", "cash_fee": "Opłata cash $", "share_fee_shares": "Opłata tokenowa (shares)", "pnl": "PnL $", "pnl_notional": "PnL/notional", "pnl_debit": "PnL/debet", "ev_same_settled": "EV na settled $"},
        ),
        "",
        "Notional, debet, PnL i EV dotyczą tych samych rozliczonych wykonań. Opłaty gotówkowe pozostają w debecie; opłaty pobrane w outcome shares zmniejszają liczbę wypłaconych tokenów i nie są doliczane do cash debit. Liczby otwartych pozycji i ich debet są również w manifeście i ledgerach.",
        "",
        "Wyniki miesięczne:",
        "",
        *_markdown_table(
            economics_monthly,
            ["month_utc", "variant", "decision_rows", "intents_at_1pct", "fills", "settlements", "unresolved_fills", "settled_notional_usd", "settled_cash_debit_usd", "settled_cash_fees_usd", "settled_share_fee_shares", "net_pnl_usd", "pnl_per_notional", "pnl_per_cash_debit", "forecast_ev_same_settled_fills_usd"],
            {"month_utc": "Miesiąc UTC", "variant": "Wariant", "decision_rows": "Decyzje", "intents_at_1pct": "EV ≥1%", "fills": "Wykonania", "settlements": "Rozliczono", "unresolved_fills": "Nierozliczone", "settled_notional_usd": "Notional $", "settled_cash_debit_usd": "Debet $", "settled_cash_fees_usd": "Fee cash $", "settled_share_fee_shares": "Fee shares", "net_pnl_usd": "PnL $", "pnl_per_notional": "PnL/notional", "pnl_per_cash_debit": "PnL/debet", "forecast_ev_same_settled_fills_usd": "EV settled $"},
        ),
        "",
        f"Dzienna różnica PnL X−C obejmuje wszystkie {pnl_bootstrap['calendar_days']} dni, z zerami w dniach bez transakcji. Średnia dzienna: {_format(pnl_bootstrap['mean_daily_pnl_difference_X_minus_C_usd']['point_estimate'])} USD, 95% CI [{_format(pnl_bootstrap['mean_daily_pnl_difference_X_minus_C_usd']['lower_95'])}, {_format(pnl_bootstrap['mean_daily_pnl_difference_X_minus_C_usd']['upper_95'])}]; różnica sumy: {_format(pnl_bootstrap['total_pnl_difference_X_minus_C_usd']['point_estimate'])} USD, 95% CI [{_format(pnl_bootstrap['total_pnl_difference_X_minus_C_usd']['lower_95'])}, {_format(pnl_bootstrap['total_pnl_difference_X_minus_C_usd']['upper_95'])}].",
        "",
        "Koncentracja dzienna jest zapisana w manifeście. Udział PnL z pięciu najlepszych dni: "
        f"C {_format(concentration['C']['top_five_share_of_total_pnl'])}, "
        f"X {_format(concentration['X']['top_five_share_of_total_pnl'])}, "
        f"MARKET {_format(concentration['MARKET']['top_five_share_of_total_pnl'])}, "
        f"RAW {_format(concentration['RAW']['top_five_share_of_total_pnl'])}. "
        f"Pełne dzienne szeregi: {(OUTPUT_DIR / 'economics/daily_pnl.csv').relative_to(ROOT).as_posix()}.",
        "",
        "## Współczynniki i interpretacja",
        "",
        "Współczynniki X są podane w skali cech po transformacjach zapisanych z treningu. Interakcje są opisowe i nie stanowią osobno potwierdzonych efektów.",
        "",
        *_markdown_table(
            pd.DataFrame([
                {"interaction": name, "coefficient": value}
                for name, value in fit_summary["coefficients_standardized"]["X"].items()
                if name in INTERACTION_FEATURES
            ]),
            ["interaction", "coefficient"],
            {"interaction": "Interakcja", "coefficient": "Współczynnik X"},
        ),
        "",
        f"Wniosek: X nie wnosi potwierdzonej informacji BTC ponad C w tym teście: różnica log-loss X−C wynosi {_format(metrics_delta(quality_overall, 'X', 'C'))}, a jej sparowany 95% CI to [{_format(quality_boot['paired_contrasts']['X_MINUS_C']['log_loss']['lower_95'])}, {_format(quality_boot['paired_contrasts']['X_MINUS_C']['log_loss']['upper_95'])}]. X wypada też gorzej od MARKET: X−MARKET = {_format(metrics_delta(quality_overall, 'X', 'MARKET'))}, 95% CI [{_format(quality_boot['paired_contrasts']['X_MINUS_MARKET']['log_loss']['lower_95'])}, {_format(quality_boot['paired_contrasts']['X_MINUS_MARKET']['log_loss']['upper_95'])}].",
        f"Nie widać dodatniej ekonomii dla C, X ani MARKET: liczba zleceń z EV co najmniej 1% wyniosła odpowiednio {economics['C']['intent_count_at_or_above_1pct_net_ev']}, {economics['X']['intent_count_at_or_above_1pct_net_ev']} i {economics['MARKET']['intent_count_at_or_above_1pct_net_ev']}; żaden z tych wariantów nie miał wykonań. RAW z istniejącego ledgeru na tym samym kalendarzu miał {economics['RAW']['fill_count']:,} wykonań i PnL {economics['RAW']['settled_net_pnl_usd']:.2f} USD po opłatach; ten wynik nie pochodzi z X.",
        f"Zakres wniosku ogranicza luka 16 dni bez zdarzeń źródłowych oraz {fit_sample['evaluation_input_rows_after_2026_06_12']:,} wejść po 12 czerwca. Wynik nie opisuje stabilności w tej luce ani na holdoucie.",
        "",
        "## Koszt i artefakty",
        "",
        f"Czas całkowity: {run_metrics['elapsed_seconds']:.2f} s; szczytowy working set procesu: {run_metrics['peak_working_set_bytes']:,} B. Python {run_metrics['python_version']}; sklearn {run_metrics['sklearn_version']}; CPU logiczne {run_metrics['logical_cpu_count']}; fit ograniczony do jednego wątku BLAS. Modele, scalery i predykcje zapisano przed bootstrapem i raportowaniem; checkpoint: {(OUTPUT_DIR / 'pre_bootstrap_checkpoint.json').relative_to(ROOT).as_posix()}.",
        "",
        f"Raport: {REPORT_PATH.relative_to(ROOT).as_posix()}. Manifest: {MANIFEST_PATH.relative_to(ROOT).as_posix()}. Dane, modele i checkpointy pozostają poza commitem; wersjonowane są protokół, kod, testy, raport i manifest.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
    temporary.write_text("\n".join(report), encoding="utf-8")
    os.replace(temporary, REPORT_PATH)

def metrics_delta(overall: pd.DataFrame, left: str, right: str) -> float:
    a = overall.loc[overall.variant.eq(left), "log_loss"].iloc[0]
    b = overall.loc[overall.variant.eq(right), "log_loss"].iloc[0]
    return float(a - b)


def _source_records() -> tuple[dict, dict]:
    source_paths = {
        name: ROOT / CONFIG["sources"][name]
        for name in (
            "feature_table_path", "design_manifest_path", "raw_oof_path",
            "official_market_index_path", "t60_snapshots_path", "t59_snapshots_path",
            "replay_dependencies_path", "source_archive_manifest_path",
            "raw_ledger_decisions_path", "raw_ledger_trades_path",
        )
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Frozen development inputs are missing: " + ", ".join(missing))
    code_paths = {
        "runner": Path(__file__), "protocol": CONFIG_PATH,
        "design_builder": ROOT / "build_btc_market_microstructure_design_20261019.py",
        "market_probability": ROOT / "run_btc_market_probability_experiment.py",
        "snapshot_state_builder": ROOT / "run_btc_preopen_economic_replay.py",
        "execution_engine": ROOT / "run_btc_preopen_t59_reassessment.py",
        "shared_ledger_audit": ROOT / "run_btc_market_base70_incremental_20261019.py",
        "tests": ROOT / CONFIG["outputs"]["test_path"],
    }
    return ({name: _record(path) for name, path in source_paths.items()},
            {name: _record(path) for name, path in code_paths.items() if path.is_file()})


def _load_prebootstrap_checkpoint(
    checkpoint_path: Path, source_records: dict, protocol_sha256: str, evaluation: pd.DataFrame,
    fit_sample: dict,
) -> tuple[dict, pd.DataFrame, dict, dict] | None:
    if not checkpoint_path.is_file():
        return None
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if (
        checkpoint.get("status") != "models_scalers_predictions_saved_before_bootstrap"
        or checkpoint.get("fit_count") != 2
        or checkpoint.get("protocol_sha256") != protocol_sha256
        or checkpoint.get("input_sha256") != {name: record["sha256"] for name, record in source_records.items()}
        or checkpoint.get("fit_sample") != fit_sample
    ):
        raise RuntimeError("Existing pre-bootstrap checkpoint does not match the frozen inputs and fit sample")
    for name, record in checkpoint["artifacts"].items():
        path = ROOT / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Pre-bootstrap checkpoint artifact failed validation: {name}")
    model_path = ROOT / checkpoint["artifacts"]["models_and_scalers"]["path"]
    predictions_path = ROOT / checkpoint["artifacts"]["evaluation_predictions"]["path"]
    fit_summary_path = ROOT / checkpoint["artifacts"]["model_fit_summary"]["path"]
    bundle = joblib.load(model_path)
    predictions = pd.read_parquet(predictions_path)
    fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
    stored_warnings = fit_summary.get("convergence_warnings", [])
    if stored_warnings and "other_fit_warnings" not in fit_summary:
        convergence_warnings = [message for message in stored_warnings if "converg" in message.lower()]
        other_warnings = [message for message in stored_warnings if "converg" not in message.lower()]
        fit_summary["convergence_warnings"] = convergence_warnings
        fit_summary["other_fit_warnings"] = other_warnings
        checkpoint["model_fit"] = fit_summary
        checkpoint["warning_summary_corrected_without_refit"] = True
        _atomic_json(fit_summary_path, fit_summary)
        checkpoint["artifacts"]["model_fit_summary"] = _record(fit_summary_path)
        _atomic_json(checkpoint_path, checkpoint)
    if (
        bundle.get("protocol_version") != int(CONFIG["protocol_version"])
        or bundle["main_scaler"].n_features_in_ != 8
        or bundle["interaction_scaler"].n_features_in_ != 6
        or bundle["model_C"].n_features_in_ != 7
        or bundle["model_X"].n_features_in_ != 14
        or fit_summary != checkpoint.get("model_fit")
    ):
        raise RuntimeError("Pre-bootstrap model artifacts do not match the frozen C/X dimensions")
    _assert_same_market_index(evaluation, predictions, "Checkpoint predictions and input-only evaluation calendar")
    return bundle, predictions, fit_summary, checkpoint


def main() -> dict:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_seconds = {}
    manifest = {
        "experiment_id": CONFIG["experiment_id"],
        "protocol_version": CONFIG["protocol_version"],
        "status": "running",
        "holdout_rows_or_labels_read": False,
        "base70_or_generator_training_performed": False,
        "archive_replayed_or_downloaded": False,
        "additive_challenger_fit_count": 0,
        "model_fit_count": 0,
        "stages": {},
    }
    source_records, code_records = _source_records()
    manifest["inputs"] = source_records
    manifest["source_code"] = code_records
    manifest["protocol_sha256"] = code_records["protocol"]["sha256"]
    _atomic_json(MANIFEST_PATH, manifest)

    stage_started = time.perf_counter()
    audit, feature_table = _audit_availability()
    stage_seconds["no_label_availability_audit"] = time.perf_counter() - stage_started
    manifest["availability_audit"] = audit
    manifest["stages"]["availability_audit"] = {"status": "completed", "elapsed_seconds": stage_seconds["no_label_availability_audit"]}
    _atomic_json(MANIFEST_PATH, manifest)

    stage_started = time.perf_counter()
    frame, data = _load_model_frame(feature_table)
    manifest["fit_sample"] = data["summary"]
    checkpoint_path = OUTPUT_DIR / "pre_bootstrap_checkpoint.json"
    restored = _load_prebootstrap_checkpoint(
        checkpoint_path, source_records, code_records["protocol"]["sha256"],
        data["evaluation"], data["summary"],
    )
    if restored is None:
        bundle, predictions, fit_summary = _fit_models_and_predict(data["train"], data["evaluation"])
        model_path = OUTPUT_DIR / "models_and_scalers.joblib"
        model_temp = model_path.with_suffix(model_path.suffix + ".tmp")
        joblib.dump(bundle, model_temp, compress=3)
        os.replace(model_temp, model_path)
        predictions_path = OUTPUT_DIR / "evaluation_predictions.parquet"
        _atomic_parquet(predictions, predictions_path)
        fit_summary_path = OUTPUT_DIR / "model_fit_summary.json"
        _atomic_json(fit_summary_path, fit_summary)
        fit_elapsed = time.perf_counter() - stage_started
        artifacts = {
            "models_and_scalers": _record(model_path),
            "evaluation_predictions": _record(predictions_path),
            "model_fit_summary": _record(fit_summary_path),
        }
        checkpoint = {
            "status": "models_scalers_predictions_saved_before_bootstrap",
            "protocol_sha256": code_records["protocol"]["sha256"],
            "input_sha256": {name: record["sha256"] for name, record in source_records.items()},
            "fit_runner_sha256": code_records["runner"]["sha256"],
            "fit_count": 2,
            "artifacts": artifacts,
            "fit_sample": data["summary"],
            "model_fit": fit_summary,
            "fit_elapsed_seconds": fit_elapsed,
            "elapsed_before_bootstrap_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": design._peak_working_set_bytes(),
            "stage_seconds_before_bootstrap": {
                **stage_seconds,
                "two_model_fits_and_prediction_checkpoint": fit_elapsed,
            },
        }
        _atomic_json(checkpoint_path, checkpoint)
    else:
        bundle, predictions, fit_summary, checkpoint = restored
        fit_elapsed = float(checkpoint["fit_elapsed_seconds"])
        restore_seconds = time.perf_counter() - stage_started
        for name, seconds in checkpoint.get("stage_seconds_before_bootstrap", {}).items():
            stage_seconds[name] = stage_seconds.get(name, 0.0) + float(seconds)
        stage_seconds["prebootstrap_checkpoint_restore"] = restore_seconds
        manifest["checkpoint_reused"] = True
        manifest["checkpoint_fit_runner_sha256"] = checkpoint["fit_runner_sha256"]
        manifest["fit_warning_summary_corrected_without_refit"] = checkpoint.get(
            "warning_summary_corrected_without_refit", False,
        )
        manifest["current_runner_differs_from_fit_runner"] = (
            checkpoint["fit_runner_sha256"] != code_records["runner"]["sha256"]
        )
        manifest["stages"]["models_scalers_predictions_checkpoint"] = {
            "status": "restored", "elapsed_seconds": fit_elapsed,
            "checkpoint_path": checkpoint_path.relative_to(ROOT).as_posix(),
        }
    stage_seconds.setdefault("two_model_fits_and_prediction_checkpoint", fit_elapsed)
    manifest["model_fit"] = fit_summary
    manifest["model_fit_count"] = 2
    manifest["model_fit_runner_sha256"] = checkpoint["fit_runner_sha256"]
    manifest["artifacts_before_bootstrap"] = checkpoint["artifacts"]
    if "models_scalers_predictions_checkpoint" not in manifest["stages"]:
        manifest["stages"]["models_scalers_predictions_checkpoint"] = {
            "status": "completed", "elapsed_seconds": fit_elapsed,
            "checkpoint_path": checkpoint_path.relative_to(ROOT).as_posix(),
        }
    _atomic_json(MANIFEST_PATH, manifest)
    if checkpoint["status"] != "models_scalers_predictions_saved_before_bootstrap":
        raise RuntimeError("Pre-bootstrap checkpoint was not committed")
    for name, record in checkpoint["artifacts"].items():
        path = ROOT / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Pre-bootstrap artifact failed its checkpoint hash: {name}")

    eval_start = pd.Timestamp(CONFIG["evaluation"]["market_start_interval"].split(",")[0].strip("[ "))
    days = pd.date_range(eval_start.floor("D"), CUTOFF_UTC.floor("D") - pd.Timedelta(days=1), freq="D", tz="UTC")
    draws = _make_day_draws(days)
    stage_started = time.perf_counter()
    quality, quality_monthly, quality_overall = _quality_results(predictions, days, draws)
    quality_monthly_path = OUTPUT_DIR / "quality_monthly.csv"
    _atomic_csv(quality_monthly, quality_monthly_path)
    _atomic_json(OUTPUT_DIR / "quality_bootstrap.json", quality["bootstrap"])
    stage_seconds["quality_bootstrap"] = time.perf_counter() - stage_started
    manifest["quality"] = quality
    manifest["stages"]["quality_bootstrap"] = {"status": "completed", "elapsed_seconds": stage_seconds["quality_bootstrap"]}
    _atomic_json(MANIFEST_PATH, manifest)

    stage_started = time.perf_counter()
    eval_predictions = predictions.loc[
        pd.to_datetime(predictions.market_start_utc, utc=True).ge(EVAL_START_UTC)
    ].copy()
    evaluation = data["evaluation"].copy().sort_values("market_start_utc", kind="stable").reset_index(drop=True)
    _assert_same_market_index(evaluation, eval_predictions, "Evaluation predictions and input-only economic calendar")
    eval_predictions["decision_day_utc"] = pd.to_datetime(eval_predictions.market_start_utc, utc=True).dt.floor("D")
    evaluation = evaluation.sort_values("market_start_utc", kind="stable").reset_index(drop=True)
    evaluation["outcome_available_by_cutoff"] = evaluation.quality_label_available_by_cutoff
    evaluation["cash_release_at_cutoff_utc"] = evaluation.official_label_release_at_utc
    unavailable = ~evaluation.outcome_available_by_cutoff.fillna(False)
    evaluation["target_polymarket_up"] = _target_asof_cutoff(evaluation)
    evaluation.loc[unavailable, "resolved_at_utc"] = pd.NaT
    condition_ids = set(evaluation.condition_id.dropna().astype(str).str.lower())
    t60_by_id, t59_by_id = _load_execution_snapshots(condition_ids)
    if not set(t60_by_id).issuperset(condition_ids):
        raise RuntimeError("At least one common-input row lacks its T-60 decision snapshot")
    market_by_id = existing.research._snapshot_map(evaluation.loc[evaluation.condition_id.notna()])
    decisions_by_variant = {}
    trades_by_variant = {}
    economics = {}
    for variant in ("C", "X", "MARKET"):
        market_rows = evaluation.copy()
        probability_column = {"C": "p_C", "X": "p_X", "MARKET": "p_market"}[variant]
        market_rows["p_candidate_platt"] = eval_predictions[probability_column].to_numpy(dtype=np.float64)
        summary, decisions, trades = existing._simulate_policy(
            variant, market_rows, t59_by_id, market_by_id, t60_by_id,
        )
        if len(decisions) != len(market_rows):
            raise RuntimeError(f"{variant} did not produce one decision record for every shared input row")
        decisions = pd.DataFrame(decisions)
        trades = pd.DataFrame(trades)
        decisions_by_variant[variant] = decisions
        trades_by_variant[variant] = trades
        economics[variant] = _economics_summary(variant, decisions, trades)
        economics[variant]["execution_engine_summary"] = summary

    raw_decisions, raw_trades, raw_audit = _load_raw_ledger(evaluation)
    decisions_by_variant["RAW"] = raw_decisions
    trades_by_variant["RAW"] = raw_trades
    economics["RAW"] = _economics_summary("RAW", raw_decisions, raw_trades)
    manifest["raw_ledger_common_sample_audit"] = raw_audit

    for variant in ECONOMIC_VARIANTS:
        _atomic_parquet(decisions_by_variant[variant], OUTPUT_DIR / "economics" / f"{variant}_decisions.parquet")
        _atomic_parquet(trades_by_variant[variant], OUTPUT_DIR / "economics" / f"{variant}_trades.parquet")
    daily_pnl = _daily_pnl(trades_by_variant, days)
    daily_pnl_path = OUTPUT_DIR / "economics" / "daily_pnl.csv"
    _atomic_csv(daily_pnl, daily_pnl_path)
    economics_monthly = _monthly_economics(decisions_by_variant, trades_by_variant)
    economics_monthly_path = OUTPUT_DIR / "economics" / "monthly.csv"
    _atomic_csv(economics_monthly, economics_monthly_path)
    pnl_bootstrap = _pnl_bootstrap(daily_pnl, draws)
    concentration = _concentration(daily_pnl, trades_by_variant)
    manifest["economics"] = economics
    manifest["economics_monthly_path"] = economics_monthly_path.relative_to(ROOT).as_posix()
    manifest["paired_daily_pnl_bootstrap"] = pnl_bootstrap
    manifest["daily_pnl_concentration"] = concentration
    stage_seconds["economics_and_existing_raw_ledger_restriction"] = time.perf_counter() - stage_started
    manifest["stages"]["economics"] = {"status": "completed", "elapsed_seconds": stage_seconds["economics_and_existing_raw_ledger_restriction"]}
    _atomic_json(MANIFEST_PATH, manifest)

    current_peak = design._peak_working_set_bytes()
    previous_peak = checkpoint.get("peak_working_set_bytes") if manifest.get("checkpoint_reused") else None
    peak_values = [value for value in (current_peak, previous_peak) if value is not None]
    run_metrics = {
        "elapsed_seconds": 0.0,
        "peak_working_set_bytes": max(peak_values) if peak_values else None,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pyarrow_version": pa.__version__,
        "logical_cpu_count": os.cpu_count(),
        "fit_native_thread_limit": 1,
        "stage_seconds": stage_seconds,
    }
    elapsed_before_resume = (
        float(checkpoint.get("elapsed_before_bootstrap_seconds", 0.0))
        if manifest.get("checkpoint_reused") else 0.0
    )
    run_metrics["elapsed_seconds"] = elapsed_before_resume + time.perf_counter() - started
    manifest["runtime_and_memory"] = run_metrics
    manifest["artifacts"] = {}
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file() and not path.name.endswith(".tmp"):
            manifest["artifacts"][path.relative_to(ROOT).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": _sha256(path),
            }
    manifest["status"] = "completed_development_only"
    _write_report(
        audit, fit_summary, data["summary"], quality, quality_monthly, economics,
        economics_monthly, daily_pnl, pnl_bootstrap, concentration, run_metrics,
    )
    manifest["report"] = _record(REPORT_PATH)
    manifest["runtime_and_memory"]["elapsed_seconds"] = elapsed_before_resume + time.perf_counter() - started
    _atomic_json(MANIFEST_PATH, manifest)
    return manifest


if __name__ == "__main__":
    result = main()
    print(json.dumps({
        "status": result["status"],
        "fit_count": result["model_fit_count"],
        "audit_rows": result["availability_audit"]["final_feature_rows"],
        "quality_rows": result["fit_sample"]["quality_rows_released_by_cutoff"],
        "runtime_seconds": result["runtime_and_memory"]["elapsed_seconds"],
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }, indent=2, sort_keys=True))
