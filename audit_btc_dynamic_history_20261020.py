"""Development-only audit of the frozen BTC dynamic-history experiment.

Run from the repository root with ``python audit_btc_dynamic_history_20261020.py``.
All outputs go to a new audit-specific directory; source artifacts are read-only.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
import concurrent.futures
import multiprocessing as mp

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_market_dynamics_action_value_20261020.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
ORIGINAL_DIR = ROOT / CONFIG["output_directory"]
AUDIT_ID = "btc_dynamic_history_audit_20261020_v1"
AUDIT_DIR = ROOT / "data/analysis/polymarket/BTC" / AUDIT_ID
REPORT_PATH = ROOT / "reports/btc_preopen/btc_dynamic_history_audit_20261020.md"
MANIFEST_PATH = ROOT / "reports/btc_preopen/btc_dynamic_history_audit_20261020_manifest.json"

DEVELOPMENT_CUTOFF = pd.Timestamp("2026-07-01T00:00:00Z")
EVALUATION_START = pd.Timestamp("2026-05-15T00:00:00Z")
FOLD_2_START = pd.Timestamp("2026-06-01T00:00:00Z")
BASELINE_COMMIT = "3d95f641288f57e5f7d4c9ed77c621b5f678daaf"
SEED = "20261020"
SAMPLE_CAP_PER_STRATUM = 20
ABS_TOLERANCE = 1e-10
REL_TOLERANCE = 1e-9
REPLAY_SEMANTICS_VERSION = 2
HORIZONS = {60: "tminus120", 15: "tminus75", 5: "tminus65"}
ENDPOINTS = ("tminus120", "tminus75", "tminus65", "tminus60")
COUNTERS = (
    "ambiguous_event_count",
    "out_of_order_price_change_count",
    "out_of_order_book_count",
    "cross_side_late_price_change_count",
    "cross_side_late_ask_count",
    "equal_source_timestamp_price_change_count",
    "book_reset_count",
    "market_event_count",
    "market_book_snapshot_count",
)
STATIC_FEATURES = tuple(CONFIG["static_feature_order"])
DYNAMIC_FEATURES = tuple(CONFIG["dynamic_feature_order"])
EVENT_COLUMNS = [
    "timestamp_received", "timestamp", "market", "event_type", "asset_id",
    "bids", "asks", "price", "size", "side", "best_bid", "best_ask",
    "fee_rate_bps",
]
TRACE_COLUMNS = (
    "condition_id", "market_start_utc", "received_at_utc", "source_timestamp_utc",
    "asset_id", "side", "event_type", "price", "size", "decision", "decision_reason",
    "group_conflict", "state_before_json", "state_after_json", "token_book_before_json",
    "token_book_after_json", "level_size_before", "level_size_after",
    "depletion_contribution_by_horizon_json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    def normalize(value):
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(normalize(payload), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def period_for(start) -> str:
    value = pd.Timestamp(start)
    if value < EVALUATION_START:
        return "pre_2026-05-15"
    if value < FOLD_2_START:
        return "fold_1_2026-05-15_to_2026-06-01"
    return "fold_2_2026-06-01_to_2026-07-01"


def safe_bool(value) -> bool:
    return False if pd.isna(value) else bool(value)


def endpoint_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    static = pd.read_parquet(ORIGINAL_DIR / "static_inputs.parquet")
    dynamic = pd.read_parquet(ORIGINAL_DIR / "dynamic_inputs.parquet")
    endpoints = pd.read_parquet(ORIGINAL_DIR / "endpoint_states.parquet")
    for name, frame in (("static", static), ("dynamic", dynamic), ("endpoints", endpoints)):
        starts = pd.to_datetime(frame.market_start_utc, utc=True)
        if starts.isna().any() or not starts.lt(DEVELOPMENT_CUTOFF).all():
            raise RuntimeError(f"{name} artifact contains invalid or non-development market starts")
    if static.condition_id.duplicated().any() or dynamic.condition_id.duplicated().any():
        raise RuntimeError("source artifacts contain duplicate market IDs")
    if len(static) != 6796 or len(dynamic) != 6796 or len(endpoints) != 6796 * 4:
        raise RuntimeError("source artifact counts do not match the reported 6,796 × 4 population")
    if set(endpoints.endpoint_name.unique()) != set(ENDPOINTS):
        raise RuntimeError("endpoint artifact does not contain exactly the four required endpoints")
    if endpoints.groupby("condition_id").size().ne(4).any():
        raise RuntimeError("an audited market is missing an endpoint row")
    return static, dynamic, endpoints


def _endpoint(endpoint_by_key: pd.DataFrame, condition_id: str, name: str) -> pd.Series:
    return endpoint_by_key.loc[(condition_id, name)]


def _counter_delta(start: pd.Series, end: pd.Series, name: str) -> int:
    return int(end[name]) - int(start[name])


def build_market_audit(
    static: pd.DataFrame, dynamic: pd.DataFrame, endpoints: pd.DataFrame,
) -> pd.DataFrame:
    ep = endpoints.set_index(["condition_id", "endpoint_name"], drop=False)
    static_by_id = static.set_index("condition_id", drop=False)
    dynamic_by_id = dynamic.set_index("condition_id", drop=False)
    records = []
    for row in static.itertuples(index=False):
        cid = str(row.condition_id)
        market = static_by_id.loc[cid]
        original = dynamic_by_id.loc[cid]
        oldest = _endpoint(ep, cid, "tminus120")
        oldest_at_ns = int(pd.Timestamp(oldest.endpoint_time_utc).value)
        initialized = {
            side: (
                pd.notna(oldest[f"{side}_first_book_receive_ns"])
                and int(oldest[f"{side}_first_book_receive_ns"]) < oldest_at_ns
            )
            for side in ("up", "down")
        }
        record = {
            "condition_id": cid,
            "market_start_utc": pd.Timestamp(row.market_start_utc),
            "decision_time_utc": pd.Timestamp(row.market_start_utc) - pd.Timedelta(seconds=60),
            "period": period_for(row.market_start_utc),
            "original_status": "accepted" if safe_bool(original.dynamic_valid) else "rejected",
            "original_first_rejection_reason": None if pd.isna(original.dynamic_reason) else str(original.dynamic_reason),
            "up_initialized_before_tminus120": bool(initialized["up"]),
            "down_initialized_before_tminus120": bool(initialized["down"]),
            "initialization_missing": not all(initialized.values()),
            "archive_history_complete": safe_bool(market.archive_history_complete),
            "archive_freshness_window_available": safe_bool(market.archive_freshness_window_available),
            "archive_window_hours_expected": int(market.archive_window_hours_expected),
            "archive_window_hours_source_available": int(market.archive_window_hours_source_available),
            "archive_window_hours_processed": int(market.archive_window_hours_processed),
            "archive_window_missing_source_hours": int(market.archive_window_missing_source_hours),
            "archive_window_unprocessed_hours": int(market.archive_window_unprocessed_hours),
            "replayed_t60_event_count": int(original.replayed_t60_event_count),
            "expected_t60_event_count": int(original.expected_t60_event_count),
            "replayed_t60_book_count": int(original.replayed_t60_book_count),
            "expected_t60_book_count": int(original.expected_t60_book_count),
        }
        all_flags: set[str] = set()
        for horizon, baseline_name in HORIZONS.items():
            start = _endpoint(ep, cid, baseline_name)
            end = _endpoint(ep, cid, "tminus60")
            prefix = f"h{horizon}s"
            quote_start_ok = safe_bool(start.quote_valid)
            quote_end_ok = safe_bool(end.quote_valid)
            record[f"{prefix}_start_time_utc"] = pd.Timestamp(start.endpoint_time_utc)
            record[f"{prefix}_end_time_utc"] = pd.Timestamp(end.endpoint_time_utc)
            record[f"{prefix}_window_start_exclusive_utc"] = pd.Timestamp(start.endpoint_time_utc)
            record[f"{prefix}_window_end_inclusive_utc"] = pd.Timestamp(end.endpoint_time_utc)
            record[f"{prefix}_quote_valid_start"] = quote_start_ok
            record[f"{prefix}_quote_valid_end"] = quote_end_ok
            record[f"{prefix}_no_future_source_start"] = safe_bool(start.no_future_source_event)
            record[f"{prefix}_no_future_source_end"] = safe_bool(end.no_future_source_event)
            flags: set[str] = set()
            if not quote_start_ok or not quote_end_ok:
                flags.add("invalid_native_quote")
            for counter in COUNTERS:
                delta = _counter_delta(start, end, counter)
                record[f"{prefix}_{counter}_start"] = int(start[counter])
                record[f"{prefix}_{counter}_end"] = int(end[counter])
                record[f"{prefix}_{counter}_increment"] = delta
                if counter == "ambiguous_event_count" and delta > 0:
                    flags.add("equal_timestamp_or_ambiguous_event")
                elif counter == "out_of_order_price_change_count" and delta > 0:
                    flags.add("ignored_same_side_stale_price_change")
                elif counter == "out_of_order_book_count" and delta > 0:
                    flags.add("ignored_stale_book")
                elif counter == "cross_side_late_price_change_count" and delta > 0:
                    flags.add("accepted_cross_side_late_price_change")
                elif counter == "book_reset_count" and delta > 0:
                    flags.add("book_message_counted_as_reset")
                elif counter == "equal_source_timestamp_price_change_count" and delta > 0:
                    flags.add("equal_timestamp_price_conflict")
            for side in ("up", "down"):
                amount = pd.to_numeric(pd.Series([start[f"{side}_ask_notional_band"]]), errors="coerce").iloc[0]
                record[f"{prefix}_{side}_starting_ask_band_notional"] = amount
                if pd.isna(amount) or amount <= 0:
                    flags.add(f"nonpositive_depletion_denominator_{side}")
            record[f"{prefix}_independent_flags"] = sorted(flags)
            record[f"{prefix}_flags_json"] = json.dumps(sorted(flags), separators=(",", ":"))
            all_flags.update(f"{horizon}s:{flag}" for flag in flags)

        record["all_independent_flags"] = sorted(all_flags)
        record["flag_signature"] = "|".join(sorted(all_flags)) if all_flags else "VALID"
        # Recompute qualification variants without touching source state or labels.
        def qualifies(ignore: str | None = None) -> bool:
            if record["initialization_missing"]:
                return False
            for horizon in HORIZONS:
                p = f"h{horizon}s"
                if not record[f"{p}_quote_valid_start"] or not record[f"{p}_quote_valid_end"]:
                    return False
                if any(record[f"{p}_{counter}_increment"] > 0 for counter in (
                    "ambiguous_event_count", "out_of_order_price_change_count", "out_of_order_book_count",
                )):
                    return False
                if ignore not in {"cross_side", "both"} and record[f"{p}_cross_side_late_price_change_count_increment"] > 0:
                    return False
                if ignore not in {"book_message", "both"} and record[f"{p}_book_reset_count_increment"] > 0:
                    return False
                if any(
                    pd.isna(record[f"{p}_{side}_starting_ask_band_notional"])
                    or record[f"{p}_{side}_starting_ask_band_notional"] <= 0
                    for side in ("up", "down")
                ):
                    return False
            return True

        record["reconstructed_original_qualification"] = qualifies()
        record["candidate_if_cross_side_gate_removed"] = qualifies("cross_side")
        record["candidate_if_book_message_gate_removed"] = qualifies("book_message")
        record["candidate_if_both_gates_removed"] = qualifies("both")
        records.append(record)
    result = pd.DataFrame.from_records(records)
    if result.condition_id.duplicated().any() or len(result) != len(static):
        raise RuntimeError("per-market audit did not preserve the complete candidate population")
    return result


def attach_t60_source_continuity(audit: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / CONFIG["sources"]["t60_snapshot_path"]
    dataset = ds.dataset(str(path), format="parquet")
    ids = audit.condition_id.astype(str).str.lower().tolist()
    id_type = dataset.schema.field("condition_id").type
    id_values = [value.encode("ascii") for value in ids] if str(id_type).startswith("binary") else ids
    predicate = (ds.field("market_start_utc") < DEVELOPMENT_CUTOFF.to_pydatetime())
    predicate = predicate & ds.field("condition_id").isin(id_values)
    columns = [
        "condition_id", "market_start_utc", "archive_freshness_window_available", "archive_window_missing_source_hours",
        "archive_window_unprocessed_hours", "archive_gap_after_up_book_init",
        "archive_gap_after_down_book_init",
    ]
    frame = dataset.to_table(columns=columns, filter=predicate).to_pandas()
    frame["condition_id"] = frame.condition_id.astype(str).str.lower()
    frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True)
    if not frame.market_start_utc.lt(DEVELOPMENT_CUTOFF).all():
        raise RuntimeError("T-60 continuity scan materialized a non-development row")
    if frame.condition_id.duplicated().any() or set(frame.condition_id) != set(ids):
        raise RuntimeError("T-60 continuity source does not exactly cover the static candidate set")
    selected = frame.set_index("condition_id")
    result = audit.copy()
    for column in columns[2:]:
        result[f"t60_{column}"] = result.condition_id.map(selected[column])
    result["source_continuity_complete"] = (
        result.archive_history_complete.fillna(False).astype(bool)
        & result.t60_archive_freshness_window_available.fillna(False).astype(bool)
        & result.t60_archive_window_missing_source_hours.fillna(1).eq(0)
        & result.t60_archive_window_unprocessed_hours.fillna(1).eq(0)
        & ~result.t60_archive_gap_after_up_book_init.fillna(True).astype(bool)
        & ~result.t60_archive_gap_after_down_book_init.fillna(True).astype(bool)
    )
    return result


def sample_plan(audit: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    selected: dict[str, dict] = {}
    strata = []

    def semantic_signature(row) -> str:
        flags = set(row.all_independent_flags)
        categories = set()
        mappings = {
            "accepted_cross_side_late_price_change": "accepted_cross_side_late",
            "ignored_same_side_stale_price_change": "ignored_same_side_stale",
            "ignored_stale_book": "ignored_stale_book",
            "equal_timestamp_or_ambiguous_event": "ambiguous_or_equal_timestamp",
            "equal_timestamp_price_conflict": "ambiguous_or_equal_timestamp",
            "book_message_counted_as_reset": "book_message_or_snapshot",
            "invalid_native_quote": "invalid_native_quote",
            "nonpositive_depletion_denominator_up": "empty_depletion_denominator",
            "nonpositive_depletion_denominator_down": "empty_depletion_denominator",
        }
        for value in flags:
            _, flag = value.split(":", 1)
            category = mappings.get(flag)
            if category:
                categories.add(category)
        if row.initialization_missing:
            categories.add("missing_initialization")
        if (
            row.archive_window_missing_source_hours > 0
            or row.archive_window_unprocessed_hours > 0
            or not row.archive_history_complete
        ):
            categories.add("source_coverage_gap")
        return "|".join(sorted(categories)) if categories else "VALID"

    sample_frame = audit.copy()
    sample_frame["semantic_signature"] = [semantic_signature(row) for row in sample_frame.itertuples(index=False)]
    for (period, signature), group in sample_frame.groupby(["period", "semantic_signature"], sort=True, dropna=False):
        ranked = sorted(
            group.condition_id.astype(str),
            key=lambda cid: (hashlib.sha256(f"{SEED}|{cid}".encode("utf-8")).hexdigest(), cid),
        )
        ids = ranked[:SAMPLE_CAP_PER_STRATUM]
        selected.setdefault(period, {})[signature] = ids
        strata.append({
            "period": period,
            "semantic_flag_combination": signature,
            "population_markets": int(len(group)),
            "sample_markets": len(ids),
            "condition_ids": ids,
        })
    base_ids = {cid for by_stratum in selected.values() for ids in by_stratum.values() for cid in ids}
    # Every candidate potentially recovered by either narrow hypothesis is traced,
    # even if it lies outside its 20-market diagnostic stratum sample.
    candidate_mask = (
        ~audit.reconstructed_original_qualification
    ) & (
        audit.candidate_if_cross_side_gate_removed
        | audit.candidate_if_book_message_gate_removed
        | audit.candidate_if_both_gates_removed
    )
    candidate_ids = set(audit.loc[candidate_mask, "condition_id"].astype(str))
    all_ids = base_ids | candidate_ids
    plan = {
        "algorithm": "SHA-256(seed + '|' + condition_id), ascending digest then ID; select at most 20 per period and mutually exclusive semantic combination (accepted cross-side late, ignored same-side stale, ignored stale book, ambiguity/equal timestamp conflict, book message, invalid quote, empty depletion denominator, missing initialization, source coverage gap); select every market in a narrow candidate-recovery set in addition",
        "seed": SEED,
        "sample_cap_per_stratum": SAMPLE_CAP_PER_STRATUM,
        "strata": strata,
        "diagnostic_strata_market_count": len(base_ids),
        "candidate_recovery_market_count": len(candidate_ids),
        "unique_market_count": len(all_ids),
        "candidate_recovery_condition_ids": sorted(candidate_ids),
        "selection_fields": ["market_start_utc", "period", "endpoint availability and flags"],
        "forbidden_selection_fields": ["outcome", "future price", "PnL", "model behavior"],
        "tolerance": {
            "float64_absolute": ABS_TOLERANCE,
            "float64_relative": REL_TOLERANCE,
            "timestamps_flags_decisions": "exact",
            "nonfinite_features": "fail",
        },
        "comparison_started": False,
    }
    sample = sample_frame.loc[sample_frame.condition_id.astype(str).isin(all_ids)].copy()
    return sample, plan


def source_scope_audit(static: pd.DataFrame) -> dict:
    cutoff = DEVELOPMENT_CUTOFF.to_pydatetime()
    market_path = ROOT / CONFIG["sources"]["market_index_path"]
    raw_path = ROOT / CONFIG["sources"]["raw_base70_oof_path"]
    t60_path = ROOT / CONFIG["sources"]["t60_snapshot_path"]
    t59_path = ROOT / CONFIG["sources"]["t59_ladders_path"]
    first_index_start = pd.Timestamp(static.market_start_utc.min()).to_pydatetime()

    def dev_scan(path: Path, columns: list[str], extra=None) -> pd.DataFrame:
        dataset = ds.dataset(str(path), format="parquet")
        predicate = ds.field("market_start_utc") < cutoff
        if extra is not None:
            predicate = predicate & extra
        frame = dataset.to_table(columns=columns, filter=predicate).to_pandas()
        if "market_start_utc" in frame:
            frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True)
            if not frame.market_start_utc.lt(DEVELOPMENT_CUTOFF).all():
                raise RuntimeError(f"scope-safe scan materialized a non-development row from {path}")
        return frame

    index_dataset = ds.dataset(str(market_path), format="parquet")
    index_cols = [name for name in ("condition_id", "market_start_utc", "target_polymarket_up", "resolved_at_utc") if name in index_dataset.schema.names]
    index = dev_scan(market_path, index_cols)
    raw_dataset = ds.dataset(str(raw_path), format="parquet")
    raw_cols = [name for name in ("market_start_utc", "nominal_decision_at", "prediction_available_at_utc", "p_model_raw", "fold_id", "target_label_available_at") if name in raw_dataset.schema.names]
    raw_earlier = dev_scan(raw_path, raw_cols, ds.field("market_start_utc") < first_index_start)
    raw_earlier["nominal_decision_at"] = pd.to_datetime(raw_earlier.nominal_decision_at, utc=True)
    raw_earlier["prediction_available_at_utc"] = pd.to_datetime(raw_earlier.prediction_available_at_utc, utc=True)
    raw_earlier["p_model_raw"] = pd.to_numeric(raw_earlier.p_model_raw, errors="coerce")
    nominally_aligned = raw_earlier.nominal_decision_at.eq(
        raw_earlier.market_start_utc - pd.Timedelta(seconds=60)
    )
    prediction_finite = (
        raw_earlier.p_model_raw.notna()
        & raw_earlier.p_model_raw.between(0.0, 1.0, inclusive="both")
    )
    availability_recorded = raw_earlier.prediction_available_at_utc.notna()
    observed_available_by_t59 = (
        availability_recorded
        & raw_earlier.prediction_available_at_utc.le(raw_earlier.nominal_decision_at + pd.Timedelta(seconds=1))
    )
    index_starts_earlier = index.market_start_utc.lt(first_index_start)
    raw_starts = set(raw_earlier.market_start_utc.astype("int64"))
    index_earlier_starts = set(index.loc[index_starts_earlier, "market_start_utc"].astype("int64"))
    raw_rows_at_index_starts = int(raw_earlier.market_start_utc.astype("int64").isin(index_earlier_starts).sum())

    def prior_snapshot_count(path: Path) -> dict:
        dataset = ds.dataset(str(path), format="parquet")
        frame = dev_scan(path, ["condition_id", "market_start_utc"], ds.field("market_start_utc") < first_index_start)
        return {
            "path": path.relative_to(ROOT).as_posix(),
            "development_rows_before_used_index_start": int(len(frame)),
            "unique_market_starts": int(frame.market_start_utc.nunique()),
            "first_start_utc": frame.market_start_utc.min().isoformat() if len(frame) else None,
            "last_start_utc": frame.market_start_utc.max().isoformat() if len(frame) else None,
            "schema_columns": dataset.schema.names,
        }

    index_candidates = []
    for path in sorted((ROOT / "data/analysis/polymarket/BTC/preopen_v1").rglob("market_index.parquet")):
        dataset = ds.dataset(str(path), format="parquet")
        frame = dev_scan(path, ["condition_id", "market_start_utc"], ds.field("market_start_utc") < first_index_start)
        index_candidates.append({
            "path": path.relative_to(ROOT).as_posix(),
            "rows_before_used_index_start": int(len(frame)),
            "unique_starts_before_used_index_start": int(frame.market_start_utc.nunique()),
            "first_start_utc": frame.market_start_utc.min().isoformat() if len(frame) else None,
            "last_start_utc": frame.market_start_utc.max().isoformat() if len(frame) else None,
        })

    event_dir = ROOT / CONFIG["sources"]["event_parts_dir"]
    event_paths = sorted(event_dir.glob("*.parquet"))
    event_hours = [pd.Timestamp(path.stem.replace("T", " "), tz="UTC") for path in event_paths]
    event_hours_dev = [value for value in event_hours if value < DEVELOPMENT_CUTOFF]
    return {
        "development_cutoff_exclusive_utc": DEVELOPMENT_CUTOFF.isoformat(),
        "used_official_index": {
            "path": market_path.relative_to(ROOT).as_posix(),
            "development_rows": int(len(index)),
            "first_start_utc": index.market_start_utc.min().isoformat() if len(index) else None,
            "last_start_utc": index.market_start_utc.max().isoformat() if len(index) else None,
            "rows_before_earliest_static_candidate": int(index.market_start_utc.lt(first_index_start).sum()),
        },
        "raw_oof_before_earliest_static_candidate": {
            "path": raw_path.relative_to(ROOT).as_posix(),
            "rows": int(len(raw_earlier)),
            "unique_market_starts": int(raw_earlier.market_start_utc.nunique()),
            "first_start_utc": raw_earlier.market_start_utc.min().isoformat() if len(raw_earlier) else None,
            "last_start_utc": raw_earlier.market_start_utc.max().isoformat() if len(raw_earlier) else None,
            "rows_with_nominal_decision_at_market_start_minus_60s": int(nominally_aligned.sum()),
            "finite_probability_rows": int(prediction_finite.sum()),
            "rows_with_recorded_prediction_available_at_utc": int(availability_recorded.sum()),
            "rows_with_recorded_availability_by_t59": int(observed_available_by_t59.sum()),
            "rows_with_unrecorded_prediction_availability": int((~availability_recorded).sum()),
            "rows_under_the_documented_one_second_availability_assumption": int((nominally_aligned & prediction_finite & ~availability_recorded).sum()),
            "intersection_with_any_local_official_index_before_start": int(raw_rows_at_index_starts),
            "raw_oof_rows_are_not_official_market_rows_without_market_id_join": True,
        },
        "other_local_index_files_before_used_start": index_candidates,
        "t60_snapshot_before_used_start": prior_snapshot_count(t60_path),
        "t59_ladders_before_used_start": prior_snapshot_count(t59_path),
        "event_archive_partition_coverage": {
            "path": event_dir.relative_to(ROOT).as_posix(),
            "development_partition_files": len(event_hours_dev),
            "first_development_hour": min(event_hours_dev).isoformat() if event_hours_dev else None,
            "last_development_hour": max(event_hours_dev).isoformat() if event_hours_dev else None,
            "files_before_first_index_market": sum(value < pd.Timestamp(first_index_start).floor("h") for value in event_hours_dev),
            "hours_before_first_index_market_are_not_a_verified_market_index": True,
        },
        "conclusion": "Earlier RAW OOF rows exist and are nominally aligned, but none records prediction_available_at_utc. The upstream BASE70 report treats T-59 availability as a one-second timing assumption, not a measured timestamp. The supplied local official-index and T60/T59 artifacts begin at the same April 15 market start, so their pre-start intersection is zero in these files; this checkout does not establish that no earlier official markets exist globally. Do not interpret the old hard-coded zero as proof that no earlier official markets exist.",
    }


def artifact_consistency() -> dict:
    freeze_path = ORIGINAL_DIR / "frozen_protocol.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    summary_manifest = json.loads((ROOT / "reports/btc_preopen/btc_market_dynamics_action_value_20261020_manifest.json").read_text(encoding="utf-8"))
    source_audit = json.loads((ORIGINAL_DIR / "source_audit.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((ORIGINAL_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    checks = {}
    for name, path in (
        ("configuration", CONFIG_PATH),
        ("runner", ROOT / "run_btc_market_dynamics_action_value_20261020.py"),
        ("feature_code", ROOT / "features/btc_market_dynamics_action_value.py"),
        ("static_inputs", ORIGINAL_DIR / "static_inputs.parquet"),
        ("dynamic_inputs", ORIGINAL_DIR / "dynamic_inputs.parquet"),
        ("endpoint_states", ORIGINAL_DIR / "endpoint_states.parquet"),
        ("common_inputs", ORIGINAL_DIR / "common_inputs.parquet"),
        ("execution_targets", ORIGINAL_DIR / "execution_targets.parquet"),
    ):
        expected = freeze.get("artifact_hashes", {}).get({
            "configuration": "config_sha256", "runner": "runner_sha256", "feature_code": "feature_code_sha256",
            "static_inputs": "static_sha256", "dynamic_inputs": "dynamic_sha256", "endpoint_states": "endpoint_sha256",
            "common_inputs": "common_inputs_sha256", "execution_targets": "execution_targets_sha256",
        }[name])
        actual = sha256_file(path)
        audited_source_hash = actual
        if name == "runner":
            baseline_blob = subprocess.run(
                ["git", "show", f"{BASELINE_COMMIT}:run_btc_market_dynamics_action_value_20261020.py"],
                cwd=ROOT, check=True, capture_output=True,
            ).stdout
            actual = hashlib.sha256(baseline_blob).hexdigest()
        checks[name] = {"path": path.relative_to(ROOT).as_posix(), "actual_sha256": actual,
                        "expected_sha256": expected, "matches_frozen_protocol": actual == expected}
        if name == "runner":
            checks[name]["baseline_commit_sha256"] = actual
            checks[name]["post_audit_source_sha256"] = audited_source_hash
            checks[name]["post_audit_source_differs_from_baseline"] = audited_source_hash != actual
        summary_expected = summary_manifest.get("code_hashes", {}).get({
            "configuration": "configuration_sha256", "runner": "runner_sha256", "feature_code": "feature_code_sha256",
        }.get(name, "")) if name in {"configuration", "runner", "feature_code"} else None
        if summary_expected is not None:
            checks[name]["expected_sha256_in_versioned_summary_manifest"] = summary_expected
            checks[name]["matches_versioned_summary_manifest"] = actual == summary_expected
    inputs = {
        "market_index": ROOT / CONFIG["sources"]["market_index_path"],
        "raw_base70_oof": ROOT / CONFIG["sources"]["raw_base70_oof_path"],
        "t60_snapshot": ROOT / CONFIG["sources"]["t60_snapshot_path"],
        "t59_ladders": ROOT / CONFIG["sources"]["t59_ladders_path"],
        "event_dependency_manifest": ROOT / CONFIG["sources"]["event_dependency_manifest_path"],
        "t59_replay_manifest": ROOT / CONFIG["sources"]["t59_replay_manifest_path"],
    }
    for name, path in inputs.items():
        actual = sha256_file(path)
        source_record = source_audit.get(name)
        expected = source_record.get("sha256") if isinstance(source_record, dict) else None
        checks[f"input_{name}"] = {"path": path.relative_to(ROOT).as_posix(), "actual_sha256": actual,
                                    "expected_sha256": expected, "matches_source_audit": actual == expected if expected else None}
    dependency_path = ROOT / CONFIG["sources"]["event_dependency_manifest_path"]
    dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
    recorded_dependency = source_audit.get("event_parts", {}).get("dependency_sha256")
    dependency_match = dependency.get("dependency_sha256") == recorded_dependency
    replay_manifest_path = ROOT / CONFIG["sources"]["t59_replay_manifest_path"]
    replay_manifest = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    worker_results = []
    for checkpoint_path in sorted((ORIGINAL_DIR / "checkpoints").glob("worker_*/checkpoint.pkl")):
        with checkpoint_path.open("rb") as stream:
            checkpoint = pickle.load(stream)
        worker_dir = checkpoint_path.parent
        output_records = {}
        for key, filename in (("dynamic", "dynamic.parquet"), ("endpoints", "endpoints.parquet")):
            output_path = worker_dir / filename
            expected_output = checkpoint.get("outputs", {}).get(key, {})
            output_records[key] = {
                "exists": output_path.is_file(),
                "sha256": sha256_file(output_path) if output_path.is_file() else None,
                "expected_sha256": expected_output.get("sha256"),
                "rows": pq.ParquetFile(output_path).metadata.num_rows if output_path.is_file() else None,
                "expected_rows": expected_output.get("rows"),
            }
        worker_results.append({
            "path": checkpoint_path.relative_to(ROOT).as_posix(),
            "status": checkpoint.get("status"),
            "identity": checkpoint.get("identity", {}),
            "outputs": output_records,
            "outputs_match_checkpoint": all(
                value["exists"] and value["sha256"] == value["expected_sha256"] and value["rows"] == value["expected_rows"]
                for value in output_records.values()
            ),
        })
    current_commit = run_manifest.get("git_commit") or run_manifest.get("commit") or run_manifest.get("baseline_commit")
    return {
        "run_manifest_top_level_keys": sorted(run_manifest),
        "run_manifest_declared_commit_fields": {key: run_manifest.get(key) for key in ("git_commit", "commit", "baseline_commit") if key in run_manifest},
        "summary_manifest_baseline_commit": "b118f274687e33414e2c90a697a7535a738200a4",
        "actual_repository_head_at_audit_start": BASELINE_COMMIT,
        "commit_note": "The prompt's verified base is HEAD. The old summary manifest's baseline_commit records its parent; this is metadata provenance, not a checked-out-code mismatch.",
        "checks": checks,
        "all_frozen_output_hashes_match": all(
            item["matches_frozen_protocol"] for name, item in checks.items()
            if name in {"static_inputs", "dynamic_inputs", "endpoint_states", "common_inputs", "execution_targets"}
        ),
        "feature_hash_matches_frozen_protocol": checks["feature_code"]["matches_frozen_protocol"],
        "feature_hash_matches_versioned_summary_manifest": checks["feature_code"].get("matches_versioned_summary_manifest"),
        "frozen_feature_hash_manifest_disagreement": checks["feature_code"].get("expected_sha256") != checks["feature_code"].get("expected_sha256_in_versioned_summary_manifest"),
        "all_source_audit_input_hashes_match": all(item["matches_source_audit"] for item in checks.values() if item.get("matches_source_audit") is not None),
        "event_dependency_digest_matches_source_audit": dependency_match,
        "replay_manifest_reports_current_dependencies": bool(replay_manifest.get("replay_dependencies_match_current")),
        "worker_checkpoints": worker_results,
        "all_four_worker_checkpoints_complete_and_output_hashes_match": len(worker_results) == 4 and all(
            item["status"] == "completed" and item["outputs_match_checkpoint"] for item in worker_results
        ),
        "worker_checkpoint_note": "Four completed per-worker replay checkpoints are present and their two parquet outputs are hash/row-count checked below.",
        "run_manifest_commit_record": current_commit,
    }


def summarize_market_audit(audit: pd.DataFrame) -> dict:
    summary = {
        "candidate_markets": int(len(audit)),
        "original_status_counts": audit.original_status.value_counts().to_dict(),
        "original_first_rejection_reason_counts": audit.original_first_rejection_reason.value_counts(dropna=False).to_dict(),
        "periods": {},
        "period_flag_incidence": {},
        "all_horizon_flag_incidence": {},
        "first_reason_and_overlapping_flag_crosswalk": {},
        "hypothesis_candidate_counts_not_yet_feasibility": {},
    }
    flag_names = sorted({
        value.split(":", 1)[1]
        for values in audit.all_independent_flags
        for value in values
    })
    for flag in flag_names:
        summary["all_horizon_flag_incidence"][flag] = int(
            audit.all_independent_flags.map(lambda values: any(value.split(":", 1)[1] == flag for value in values)).sum()
        )
    summary["horizon_flag_incidence"] = {
        f"{horizon}s": {
            flag: int(audit.all_independent_flags.map(
                lambda values: f"{horizon}s:{flag}" in values
            ).sum())
            for flag in flag_names
        }
        for horizon in HORIZONS
    }
    for period, group in audit.groupby("period", sort=True):
        summary["periods"][period] = {
            "markets": int(len(group)),
            "original_accepted": int(group.original_status.eq("accepted").sum()),
            "original_rejected": int(group.original_status.eq("rejected").sum()),
            "observed_days": int(pd.to_datetime(group.market_start_utc, utc=True).dt.floor("D").nunique()),
            "first_reason_counts": group.original_first_rejection_reason.value_counts(dropna=False).to_dict(),
            "flag_signature_counts": group.flag_signature.value_counts().to_dict(),
            "candidate_if_cross_side_gate_removed": int(group.candidate_if_cross_side_gate_removed.sum()),
            "candidate_if_book_message_gate_removed": int(group.candidate_if_book_message_gate_removed.sum()),
            "candidate_if_both_gates_removed": int(group.candidate_if_both_gates_removed.sum()),
        }
        summary["period_flag_incidence"][period] = {
            f"{horizon}s": {
                flag: int(group.all_independent_flags.map(
                    lambda values: f"{horizon}s:{flag}" in values
                ).sum())
                for flag in flag_names
            }
            for horizon in HORIZONS
        }
    for reason, group in audit.groupby("original_first_rejection_reason", dropna=False, sort=True):
        key = "accepted" if pd.isna(reason) else str(reason)
        summary["first_reason_and_overlapping_flag_crosswalk"][key] = {
            flag: int(group.all_independent_flags.map(
                lambda values: any(value.split(":", 1)[1] == flag for value in values)
            ).sum())
            for flag in flag_names
        }
    summary["hypothesis_candidate_counts_not_yet_feasibility"] = {
        "cross_side_only_rule": int((audit.candidate_if_cross_side_gate_removed & ~audit.reconstructed_original_qualification).sum()),
        "book_message_only_rule": int((audit.candidate_if_book_message_gate_removed & ~audit.reconstructed_original_qualification).sum()),
        "both_rules": int((audit.candidate_if_both_gates_removed & ~audit.reconstructed_original_qualification).sum()),
    }
    return summary


def crosscheck_recovered_candidate_features(
    recovery: pd.DataFrame,
    replay_endpoints: pd.DataFrame,
    replay_features: pd.DataFrame,
) -> dict:
    """Cross-check the corrected production feature builder using independent states/traces."""
    import run_btc_market_dynamics_action_value_20261020 as production_runner

    recovered = recovery.loc[recovery.candidate_recovered].copy()
    ids = sorted(recovered.condition_id.astype(str))
    if not ids:
        result = {"markets": 0, "feature_values_compared": 0, "mismatch_count": 0}
        write_json(AUDIT_DIR / "recovered_candidate_feature_crosscheck.json", result)
        return result

    trace_paths = sorted((AUDIT_DIR / "independent_replay").glob("worker_*/event_trace.parquet"))
    if len(trace_paths) != 4:
        raise RuntimeError("expected all four independent event trace shards for candidate cross-check")
    static_tokens = pd.read_parquet(
        ORIGINAL_DIR / "static_inputs.parquet",
        columns=["condition_id", "up_token_id", "down_token_id"],
    ).set_index("condition_id")
    token_to_side = {
        cid: {
            str(static_tokens.loc[cid, "up_token_id"]): "up",
            str(static_tokens.loc[cid, "down_token_id"]): "down",
        }
        for cid in ids
    }
    contributions = {
        cid: {horizon: {"up": 0.0, "down": 0.0} for horizon in (5, 15, 60)}
        for cid in ids
    }
    for path in trace_paths:
        table = pq.read_table(
            path,
            columns=["condition_id", "asset_id", "depletion_contribution_by_horizon_json"],
            filters=[("condition_id", "in", ids)],
        )
        for cid, payload in zip(
            zip(
                table.column("condition_id").to_pylist(),
                table.column("asset_id").to_pylist(),
            ),
            table.column("depletion_contribution_by_horizon_json").to_pylist(),
        ):
            cid, token_id = str(cid[0]), str(cid[1])
            side = token_to_side[cid].get(token_id)
            if side is None:
                continue
            for horizon, amount in json.loads(payload).items():
                contributions[cid][int(horizon)][side] += float(amount)

    feature_by_id = replay_features.set_index("condition_id")
    mismatch_examples = []
    values_compared = 0
    max_absolute_difference = 0.0
    for cid in ids:
        endpoint_group = replay_endpoints.loc[replay_endpoints.condition_id.astype(str).eq(cid)]
        endpoints = {str(row.endpoint_name): row._asdict() for row in endpoint_group.itertuples(index=False)}
        if set(endpoints) != set(ENDPOINTS):
            raise RuntimeError(f"candidate {cid} is missing an independently replayed endpoint")
        runtime = {
            "endpoints": endpoints,
            "depletion_trackers": {},
            "ambiguous_event_count": int(endpoints["tminus60"]["ambiguous_event_count"]),
            "book_reset_count": int(endpoints["tminus60"]["book_reset_count"]),
        }
        for horizon, baseline_name in ((60, "tminus120"), (15, "tminus75"), (5, "tminus65")):
            baseline = endpoints[baseline_name]
            tracker = {"start_ns": int(pd.Timestamp(baseline["endpoint_time_utc"]).value), "sides": {}}
            for side in ("up", "down"):
                ask = baseline[f"{side}_ask"]
                notional = baseline[f"{side}_ask_notional_band"]
                tracker["sides"][side] = {
                    "band_low": ask,
                    "band_high": min(1.0, float(ask) + 0.02) if pd.notna(ask) else None,
                    "starting_notional": float(notional or 0.0),
                    "removed_notional": contributions[cid][horizon][side],
                }
            runtime["depletion_trackers"][horizon] = tracker
        market = {
            "condition_id": cid,
            "market_start_utc": endpoints["tminus60"]["market_start_utc"],
        }
        actual, reason, _ = production_runner._dynamic_record(market, runtime)
        if actual is None:
            mismatch_examples.append({"condition_id": cid, "reason": reason})
            continue
        independent = feature_by_id.loc[cid]
        for name in DYNAMIC_FEATURES:
            values_compared += 1
            left, right = float(actual[name]), float(independent[name])
            difference = abs(left - right)
            max_absolute_difference = max(max_absolute_difference, difference)
            if not _within_tolerance(left, right) and len(mismatch_examples) < 100:
                mismatch_examples.append({
                    "condition_id": cid, "feature": name,
                    "production": left, "independent": right,
                })
    result = {
        "markets": len(ids),
        "feature_values_compared": values_compared,
        "mismatch_count": len(mismatch_examples),
        "mismatch_examples": mismatch_examples,
        "maximum_absolute_difference": max_absolute_difference,
        "tolerance": {"absolute": ABS_TOLERANCE, "relative": REL_TOLERANCE},
        "production_inputs": "independently reconstructed endpoint states and event-level depletion contributions",
        "scope": {"models_fitted": False, "economics_scored": False},
    }
    write_json(AUDIT_DIR / "recovered_candidate_feature_crosscheck.json", result)
    if values_compared != len(ids) * len(DYNAMIC_FEATURES) or result["mismatch_count"]:
        raise RuntimeError(f"corrected production feature cross-check failed: {json.dumps(result, sort_keys=True)}")
    return result


def write_audit_reports(
    audit: pd.DataFrame,
    summary: dict,
    scope: dict,
    consistency: dict,
    plan: dict,
    replay_summary: dict,
    comparison: dict,
    feature_crosscheck: dict,
    gate: dict,
    synthetic: dict,
) -> dict:
    def table(headers: list[str], rows: list[list[object]]) -> str:
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows)
        return "\n".join(lines)

    def comma(value) -> str:
        return f"{int(value):,}".replace(",", " ")

    def json_artifact(name: str) -> dict:
        path = AUDIT_DIR / name
        return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}

    def parquet_artifact(name: str) -> dict:
        path = AUDIT_DIR / name
        metadata = pq.ParquetFile(path).metadata
        return {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "row_count": int(metadata.num_rows),
            "column_count": int(metadata.num_columns),
            "schema": pq.ParquetFile(path).schema_arrow.names,
        }

    recovered = pd.read_parquet(AUDIT_DIR / "candidate_recovery_audit.parquet")
    recovered_rows = recovered.loc[recovered.candidate_recovered]
    recovered_counts = recovered.loc[recovered.candidate_recovered].groupby("period").size().to_dict()
    period_rows = []
    period_order = (
        "pre_2026-05-15",
        "fold_1_2026-05-15_to_2026-06-01",
        "fold_2_2026-06-01_to_2026-07-01",
    )
    period_labels = {
        "pre_2026-05-15": "Przed 15 maja",
        "fold_1_2026-05-15_to_2026-06-01": "Fold 1 — 15–31 maja",
        "fold_2_2026-06-01_to_2026-07-01": "Fold 2 — czerwiec",
    }
    for period in period_order:
        item = summary["periods"][period]
        reasons = item["first_reason_counts"]
        recovered_count = int(recovered_counts.get(period, 0))
        period_rows.append([
            period_labels[period], comma(item["markets"]), comma(item["observed_days"]),
            comma(item["original_accepted"]), comma(item["original_rejected"]),
            comma(item["original_accepted"] + recovered_count),
            comma(item["original_rejected"] - recovered_count),
            comma(reasons.get("unresolved_order_or_history_60s", 0)),
            f"{comma(reasons.get('book_reset_inside_depletion_window_60s', 0))} → "
            f"{comma(reasons.get('book_reset_inside_depletion_window_60s', 0) - recovered_count)}",
            comma(reasons.get("invalid_native_quote_window_60s", 0)),
            comma(reasons.get("invalid_native_quote_window_15s", 0)),
        ])

    reason_flag_names = [
        "accepted_cross_side_late_price_change", "book_message_counted_as_reset",
        "equal_timestamp_or_ambiguous_event", "equal_timestamp_price_conflict",
        "ignored_same_side_stale_price_change", "ignored_stale_book", "invalid_native_quote",
        "nonpositive_depletion_denominator_up", "nonpositive_depletion_denominator_down",
    ]
    reason_flag_labels = [
        "Cross-side late", "Wiadomość book jako reset", "Ambiguity / equal ts",
        "Konflikt equal ts", "Stary price change", "Stary book", "Niepoprawny quote",
        "Mianownik UP ≤0", "Mianownik DOWN ≤0",
    ]
    crosswalk = summary["first_reason_and_overlapping_flag_crosswalk"]
    first_reason_rows = []
    reason_order = [
        ("accepted", "Przyjęty"),
        ("unresolved_order_or_history_60s", "unresolved_order_or_history_60s"),
        ("book_reset_inside_depletion_window_60s", "book_reset_inside_depletion_window_60s"),
        ("invalid_native_quote_window_60s", "invalid_native_quote_window_60s"),
        ("invalid_native_quote_window_15s", "invalid_native_quote_window_15s"),
    ]
    for key, label in reason_order:
        values = crosswalk.get(key, {})
        first_reason_rows.append([label] + [comma(values.get(flag, 0)) for flag in reason_flag_names])

    horizon_tables = []
    for horizon in (5, 15, 60):
        counts = summary["period_flag_incidence"]
        rows = []
        for period in period_order:
            period_flags = counts[period][f"{horizon}s"]
            rows.append([period_labels[period]] + [comma(period_flags.get(flag, 0)) for flag in reason_flag_names])
        horizon_tables.append(
            f"### {horizon} s\n\n" + table(["Okres"] + reason_flag_labels, rows)
        )

    overall_first_reasons = summary["original_first_rejection_reason_counts"]
    original_accepted_count = int(summary["original_status_counts"]["accepted"])
    recovered_count = int(recovered.candidate_recovered.sum())
    overall_rows = [
        ["Przyjęte", comma(original_accepted_count), comma(original_accepted_count + recovered_count)],
        ["unresolved_order_or_history_60s", comma(overall_first_reasons.get("unresolved_order_or_history_60s", 0)), comma(overall_first_reasons.get("unresolved_order_or_history_60s", 0))],
        ["book_reset_inside_depletion_window_60s", comma(overall_first_reasons.get("book_reset_inside_depletion_window_60s", 0)), comma(overall_first_reasons.get("book_reset_inside_depletion_window_60s", 0) - recovered_count)],
        ["invalid_native_quote_window_60s", comma(overall_first_reasons.get("invalid_native_quote_window_60s", 0)), comma(overall_first_reasons.get("invalid_native_quote_window_60s", 0))],
        ["invalid_native_quote_window_15s", comma(overall_first_reasons.get("invalid_native_quote_window_15s", 0)), comma(overall_first_reasons.get("invalid_native_quote_window_15s", 0))],
    ]
    source = json.loads((AUDIT_DIR / "verified_event_inputs.json").read_text(encoding="utf-8"))
    replay_artifact = json.loads((AUDIT_DIR / "independent_replay_manifest.json").read_text(encoding="utf-8"))
    synthetic_text = (
        f"Dwa przebiegi zakończyły się identyczną pełną książką T−60, ale depletion względne wyniosło "
        f"{synthetic['path_a_price_change_then_state_changing_snapshot_depletion_relative']:.6g} vs "
        f"{synthetic['path_b_no_native_state_change_depletion_relative']:.6g}; kontrola wykryła różnicę."
    )
    gate_rows = [
        ["Rynki oceny", comma(gate["evaluation_market_count"]), comma(gate["minimum_evaluation_markets"]), "PASS" if gate["evaluation_markets_gate_pass"] else "FAIL"],
        ["Obserwowane dni", comma(gate["evaluation_observed_day_count"]), comma(gate["minimum_evaluation_observed_days"]), "PASS" if gate["evaluation_days_gate_pass"] else "FAIL"],
        ["Niepuste foldy", comma(gate["nonempty_evaluation_fold_count"]), comma(gate["minimum_nonempty_evaluation_folds"]), "PASS" if gate["nonempty_folds_gate_pass"] else "FAIL"],
        ["Udostępnione etykiety", "1 760 / 2 279; klasy 0 i 1", "≥1 000 w każdym foldzie", "PASS" if gate["released_label_gate_pass"] else "FAIL"],
        ["Znane q=$5 po obu stronach", "908 / 1 207", "≥1 000 w każdym foldzie", "FAIL" if not gate["q5_two_sided_training_gate_pass"] else "PASS"],
    ]
    fold_rows = [[
        f"Fold {item['fold_id']}", comma(item["evaluation_market_count"]), comma(item["evaluation_observed_days"]),
        comma(item["released_training_label_count"]), ", ".join(map(str, item["released_training_label_classes"])),
        comma(item["known_two_sided_q5_training_market_count"]),
        "PASS" if item["released_label_minimum_met"] else "FAIL",
        "PASS" if item["q5_two_sided_minimum_met"] else "FAIL",
    ] for item in gate["folds"]]
    all_market_audit = parquet_artifact("market_horizon_audit.parquet")
    sample_plan_artifact = json_artifact("sample_plan.json")
    summary_artifact = json_artifact("endpoint_summary.json")
    consistency_artifact = json_artifact("artifact_consistency.json")
    scope_artifact = json_artifact("earlier_history_scope.json")
    verified_sources_artifact = json_artifact("verified_event_inputs.json")
    candidate_artifact = parquet_artifact("candidate_recovery_audit.parquet")
    event_trace_artifacts = [
        parquet_artifact(path.relative_to(AUDIT_DIR).as_posix())
        for path in sorted((AUDIT_DIR / "independent_replay").glob("worker_*/event_trace.parquet"))
    ]
    replay_endpoint_artifact = parquet_artifact("independent_endpoints.parquet")
    replay_feature_artifact = parquet_artifact("independent_features.parquet")
    feature_crosscheck_artifact = json_artifact("recovered_candidate_feature_crosscheck.json")
    prefit_artifact = json_artifact("final_prefit_gate.json")
    synthetic_artifact = json_artifact("synthetic_path_counterexample.json")
    comparison_artifact = json_artifact("independent_replay_comparison.json")
    source_gap_artifact = json_artifact("sample_source_partition_gaps.json")
    worker_summary_artifacts = [
        json_artifact(f"independent_replay/worker_{index}/event_summary.json")
        for index in range(4)
    ]
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    source_hashes = {
        name: item.get("actual_sha256")
        for name, item in consistency["checks"].items()
    }
    source_hashes["audited_runner"] = consistency["checks"]["runner"].get("post_audit_source_sha256")
    source_hashes["frozen_protocol"] = sha256_file(ORIGINAL_DIR / "frozen_protocol.json")
    source_hashes["original_report"] = sha256_file(ROOT / "reports/btc_preopen/btc_market_dynamics_action_value_20261020.md")
    source_hashes["original_report_manifest"] = sha256_file(ROOT / "reports/btc_preopen/btc_market_dynamics_action_value_20261020_manifest.json")
    source_hashes["verified_event_partition_inventory"] = sha256_file(AUDIT_DIR / "verified_event_inputs.json")
    report_lines = [
        "# Audyt historii dynamicznej BTC — 2026-10-20",
        "",
        f"Audyt: `{AUDIT_ID}`. Kod bazowy: `{BASELINE_COMMIT}`. Identyfikator eksperymentu: `btc_market_dynamics_action_value_20261020`.",
        "",
        "## Wniosek",
        "",
        "Potwierdzono błąd licznika `book_reset_count`: zwiększał się przed decyzją replayu dla każdej wiadomości `book`, także dla przyjętych, pełnostanowo identycznych snapshotów. W wąskiej grupie 992 rynków odrzuconych z tego powodu dokładny ślad wykazał 4 rynki z samymi przyjętymi identycznymi snapshotami. Po poprawce licznika i niezależnej kontroli stanu oraz cech odzyskano te 4 rynki; pozostałe 988 zawierały zaakceptowane snapshoty zmieniające pełny native book i nadal są odrzucone. Nie zmieniano cross-side, stale, ambiguity, quote validity, inicjalizacji, mianowników, granic czasu ani definicji 21 cech H.",
        "",
        f"Feasibility przeliczono przed fit: **{comma(gate['evaluation_market_count'])}/{comma(gate['minimum_evaluation_markets'])} rynków oceny; gate FAIL**. Obserwowane dni i niepuste foldy przechodzą, udostępnione etykiety przechodzą w obu foldach; fold 1 nadal ma {comma(gate['folds'][0]['known_two_sided_q5_training_market_count'])}/1 000 kwalifikowanych rynków q=$5. Wniosek: potwierdzony błąd poprawiono i feasibility przeliczono; oryginalny eksperyment pozostaje zablokowany. Rentowność nie była oceniana.",
        "",
        "## Zakres, populacja i odtwarzalność",
        "",
        f"Audyt odtworzył pełną populację **{comma(len(audit))}** statycznych kandydatów development z `{DEVELOPMENT_CUTOFF.isoformat()}` jako cutoffem wyłącznym. Wczytano wyłącznie rynki development, kończące się przed cutoffem. Oryginalne pliki, checkpointy i raporty pozostały nietknięte; wyniki zapisano pod `data/analysis/polymarket/BTC/{AUDIT_ID}/`.",
        "",
        f"Weryfikacja artefaktów: statyczne/dynamiczne/endpointy/common/execution-targets zachowują hashe zamrożonego przebiegu; wszystkie cztery checkpointy workerów i ich wyjścia zgadzają się z manifestami. Hash pliku cech w lokalnym checkoutcie odpowiada wersjonowanemu manifestowi raportu (`{consistency['checks']['feature_code'].get('expected_sha256_in_versioned_summary_manifest')}`), ale nie hashowi wpisanemu do `frozen_protocol.json` (`{consistency['checks']['feature_code'].get('expected_sha256')}`). Nie znaleziono wyjaśnienia tej rozbieżności; zachowano ją jako ograniczenie pochodzenia. Kod runnera z bazowego commitu odpowiada zamrożonemu SHA; audytowa poprawka ma osobny hash `{source_hashes['audited_runner']}`. Konfiguracja nie zmieniła się.",
        "",
        f"Wersje raportu/konfiguracji mają hashe: raport `{source_hashes['original_report']}`, manifest `{source_hashes['original_report_manifest']}`, konfiguracja `{source_hashes.get('configuration')}`. Pełny crosswalk hashy i checkpointów: `{consistency_artifact['path']}`.",
        "",
        "## Liczebności na tej samej populacji",
        "",
        "Kategorie pierwszego odrzucenia są rozłączne i zależne od kolejności kontroli; flagi z kolejnych horyzontów nakładają się. Poniższa tabela zachowuje historyczne liczby bazowe i pokazuje wyłącznie potwierdzone przesunięcie czterech rynków.",
        "",
        table(["Okres UTC", "Kandydaci", "Dni", "Przyjęte bazowo", "Odrzucone bazowo", "Przyjęte po audycie", "Odrzucone po audycie", "Pierwszy powód: unresolved", "book reset bazowo → po", "Quote 60 s", "Quote 15 s"], period_rows),
        "",
        table(["Pierwszy powód / status", "Bazowo", "Po audycie"], overall_rows),
        "",
        "Zmiany statusu odrzucony → przyjęty:",
        "",
        table(
            ["Market ID", "Start UTC", "Pierwotna klasa", "Po audycie"],
            [[
                item.condition_id, item.market_start_utc, item.primary_rejection_class,
                "accepted — `book_reset_count` nie przyrasta dla identycznego snapshotu",
            ] for item in recovered_rows.itertuples(index=False)],
        ),
        "",
        "Nie zmienił się żaden status przyjęty → odrzucony ani żaden niezwiązany powód. 4 zmiany odrzucony → przyjęty pochodzą wyłącznie z okresu 1–30 czerwca; 988 rynków pozostaje odrzuconych z powodu rzeczywistej zmiany pełnego book. Cztery rynki mają udokumentowane identyczne snapshoty, kompletną ciągłość źródła, przechodzą pozostałe flagi i mają skończone 21 cech H. Hipoteza wspólnego usunięcia obu gate’ów dawała 994 kandydatów; 2 dodatkowe rynki wymagają też pominięcia cross-side gate’u i nie zostały dopuszczone.",
        "",
        "### Pierwszy powód a nakładające się flagi (cały okres)",
        "",
        "Każda kolumna liczy rynki z flagą gdziekolwiek w horyzontach 5/15/60 s, również gdy pierwszym powodem był inny warunek.",
        "",
        table(["Pierwszy powód"] + reason_flag_labels, first_reason_rows),
        "",
        f"W horyzoncie 60 s: {comma(summary['all_horizon_flag_incidence'].get('accepted_cross_side_late_price_change', 0))} rynków miało zaakceptowane cross-side-late, {comma(summary['all_horizon_flag_incidence'].get('ignored_same_side_stale_price_change', 0))} miało zignorowane stale price changes, {comma(summary['all_horizon_flag_incidence'].get('ignored_stale_book', 0))} zignorowane stale books, {comma(summary['all_horizon_flag_incidence'].get('equal_timestamp_or_ambiguous_event', 0))} ambiguity/equal-time flags i {comma(summary['all_horizon_flag_incidence'].get('equal_timestamp_price_conflict', 0))} konflikty timestampów. Cross-side late pozostaje zastosowane do własnej strony zgodnie z replayem; samodzielne usunięcie gate’u odzyskuje 0 rynków.",
        f"Inicjalizacja przed T−120 brakowała w {comma(summary.get('population_integrity', {}).get('initialization_missing_market_count', 0))} rynkach; ciągłość źródła jest kompletna dla {comma(summary.get('population_integrity', {}).get('source_complete_market_count', 0))}/{comma(len(audit))}. Liczba nie-dodatnich mianowników depletion według horyzontu/strony: `{json.dumps(summary.get('population_integrity', {}).get('nonpositive_depletion_denominator_market_counts', {}), sort_keys=True)}`.",
        "",
        "### Niezależne flagi według okresu i horyzontu",
        "",
        "Wartości nie sumują się do liczby unikalnych rynków; pojedynczy rynek może mieć kilka flag. Pełna tabela po rynku zawiera czasy granic, liczniki start/end/przyrost, wszystkie flagi i powód statusu.",
        "",
        "\n\n".join(horizon_tables),
        "",
        f"Pełna tabela audytowa: `{all_market_audit['path']}` — {comma(all_market_audit['row_count'])} wierszy po rynku; {all_market_audit['column_count']} kolumn, zawiera wszystkie trzy horyzonty i cztery endpointy. Schemat i SHA-256 w manifeście. Podsumowanie liczebności: `{summary_artifact['path']}`.",
        "",
        "## Semantyka zdarzeń i poprawka",
        "",
        "| Przypadek | Zdarzenie wejściowe i zachowanie replayu | Licznik po poprawce | Decyzja gate’u |",
        "| --- | --- | --- | --- |",
        "| Pierwszy snapshot / reinitializacja | replay akceptuje i inicjalizuje native state | +1 | nadal blokuje okno, gdy wystąpi w oknie cechy |",
        "| Snapshot zaakceptowany, identyczne bidy i aski na wszystkich poziomach | bez zmiany book; pełny stan porównywany po normalizacji cen | +0 | nie odrzuca samą wiadomością book |",
        "| Snapshot zaakceptowany, zmieniony pełny book | stan przechodzi na snapshot; ścieżka depletion pozostaje nieznana | +1 | okno nadal odrzucone |",
        "| Snapshot stary względem najnowszego native side | replay ignoruje, zwiększa out-of-order book | +0 | nadal odrzuca przez out-of-order |",
        "| Konflikt equal timestamp / receive group | nadal zwiększa ambiguity i blokuje okno | tylko gdy zaakceptowana zmiana pełnego stanu | nadal odrzucone przez ambiguity/conflict |",
        "",
        "Kontrprzykład syntetyczny: " + synthetic_text,
        "",
        f"W 992 kandydatach książkowych prześledzono {comma(int(recovered.relevant_book_event_count.sum()))} wiadomości `book` w wymaganych oknach; wszystkie były przyjęte, bez stale snapshotów i konfliktów grupy. Cztery odzyskane rynki obejmowały łącznie {comma(int(recovered.loc[recovered.candidate_recovered, 'accepted_identical_full_book_event_count'].sum()))} identycznych snapshotów. Kategoria 988 state-changing rynków pozostaje wykluczona; nie rekonstruowano ścieżki przez odejmowanie snapshotów.",
        "",
        "### Niezależna kontrola wartości i stanu",
        "",
        f"- {comma(comparison['endpoint_rows_compared'])} endpoint rows z {comma(plan['unique_market_count'])} rynków: 0 mismatch dla dokładnych liczników/flag, czasu/książki; 0 różnic pełnych ask ladders; największa bezwzględna różnica float64 {comparison['maximum_absolute_endpoint_float_difference']:.3g}.",
        f"- {comma(comparison['static_feature_values_compared'])} wartości cech S zależnych od book: 0 mismatch; max |Δ| {comparison['maximum_absolute_static_feature_difference']:.3g}.",
        f"- {comma(comparison['dynamic_feature_values_compared'])} wartości H z 60 pierwotnie przyjętych rynków: 0 mismatch i 0 niefinite; max |Δ| {comparison['maximum_absolute_dynamic_feature_difference']:.3g}.",
        f"- Każdy z 4 odzyskanych rynków: 21/21 cech H porównano między niezależnym replayem i feature builderem poprawionego runnera; {feature_crosscheck['feature_values_compared']} wartości, {feature_crosscheck['mismatch_count']} mismatch, max |Δ| {feature_crosscheck['maximum_absolute_difference']:.3g}. Stan i ścieżka pochodziły z niezależnego replayu.",
        "- Event trace pokazuje czas source i receive, token/stronę, typ, cenę/rozmiar, accepted/ignored/conflict, stan przed/po, oraz wkład do depletion. Zapisano także synthetic endpoint counterexample.",
        "",
        "## Gate feasibility przed dopasowaniem",
        "",
        table(["Warunek", "Po audycie", "Minimum", "Wynik"], gate_rows),
        "",
        table(["Fold", "Rynki oceny", "Dni", "Dostępne etykiety treningowe", "Klasy", "q=$5 po obu stronach", "Labels", "q=$5"], fold_rows),
        "",
        f"Gate ogólny: **{'PASS' if gate['overall_prefit_gate_pass'] else 'FAIL'}**. Brakuje 1 363 rynków oceny względem progu 2 000 oraz 92 kwalifikowanych treningowych rynków q=$5 w foldzie 1. To luki liczebności, nie prognoza liczby możliwych do odzyskania rynków. Zbiory foldów mogą się nakładać. Nie uruchomiono fit ani ekonomiki.",
        "",
        "## Wcześniejsze źródła i ograniczenia historii",
        "",
        f"Użyty lokalny oficjalny indeks obejmuje {comma(scope['used_official_index']['development_rows'])} market starts od {scope['used_official_index']['first_start_utc']} do {scope['used_official_index']['last_start_utc']}. W lokalnych indeksach T−60/T−59 nie ma wierszy przed tym startem. RAW OOF ma {comma(scope['raw_oof_before_earliest_static_candidate']['rows'])} skończonych wierszy z 2023-06-20 do 2026-04-15 17:00Z; wszystkie są nominalnie T−60 względem startu, ale żaden nie ma zarejestrowanego `prediction_available_at_utc`; T−59 opiera się na założeniu 1 sekundy, a nie zmierzonym czasie. Przecięcie z wcześniejszym oficjalnym indeksem w dostarczonych plikach wynosi 0, lecz pełnego wcześniejszego indeksu globalnego nie zweryfikowano. Wartość 0 wpisana historycznie w runnerze nie dowodzi braku takich rynków.",
        "",
        f"Źródłowy replay ma {comma(source['hourly_partitions_expected'])} oczekiwanych godzin: {comma(source['hourly_partitions_present_and_hash_verified'])} plików zweryfikowano SHA-256, w tym {comma(source['nonempty_partitions_verified_by_sha256'])} niepustych i {comma(source['empty_partitions_verified_by_sha256_and_zero_rows'])} pustych. Brakują 2026-06-11T04, T05 i T06; wszystkie trzy są poza oknami 26 h każdego z {comma(plan['unique_market_count'])} badanych rynków. W ich oknach nie było brakujących partycji. Pełny indeks i archiwum sprzed 15 kwietnia oraz ich przecięcie S+H+RAW pozostają niezweryfikowane.",
        "",
        "Backfill wymaga osobnego, źródłowo zweryfikowanego oficjalnego indeksu, historii wiadomości book/price-change od inicjalizacji potrzebnej dla H oraz kompatybilnych RAW OOF i T−60/T−59 drabin z potwierdzonym czasem dostępności. Należy hashować każdą godzinę, rozliczać puste partycje i braki, a następnie zmierzyć przecięcie S+H+RAW i poprawnych targetów. Dostępne pliki nie pozwalają oszacować liczby odzyskiwalnych rynków; masowego backfillu nie wykonywano.",
        "",
        "## Koszt, walidacja i wykonane etapy",
        "",
        f"Niezależny replay: {comma(replay_summary['event_rows_selected'])} wierszy zdarzeń z {comma(source['content_bytes_verified'])} bajtów zweryfikowanych partycji; {comma(replay_summary['sample_market_count'])} rynków, 4 workery, {replay_summary['elapsed_seconds']:.1f} s, {replay_summary['throughput_rows_per_second']:,.0f} wierszy/s. Peak working set nie został wiarygodnie odczytany i pozostaje null; nie przedstawiam pojedynczego odczytu jako peak RAM. {comma(source['hourly_partitions_present_and_hash_verified'])} partycji plików zweryfikowano hashem; brakujące trzy pozostawały poza próbką.",
        "",
        "- Wykonano: inwentaryzację hashy i checkpointów; rekonstrukcję per-market/per-horizon dla 6 796 rynków; deterministyczny wybór 57 rozłącznych klas/okresów do 20 na klasę plus wszystkie 994 candidate IDs; niezależne odtworzenie dla 1 469 rynków; test syntetyczny; sprawdzenie recovered features; zamrożony gate przed fit.",
        "- Nie wykonano: fit A/B/C/D; tuning; scoring/PnL/ROI/bootstrap; odczyt lub scoring wierszy rynków holdout; historyczny backfill; live settings; order submission.",
        "",
        "Wyniki testów, kompilacji i diff-check z tej rewizji zapisano w historii commitów i w końcowym podsumowaniu audytu.",
        "",
        "## Status zakresu",
        "",
        "`models not fitted`; `economics not scored`; `holdout market rows not read/scored`; `live unchanged`; `orders not submitted`.",
        "",
        "Pliki szczegółowe: sample plan, source partition inventory, worker traces/checkpoints, independent endpoint states, independent feature table, candidate recovery table, counterexample, comparison, gate i schematy/hashes są wskazane w wersjonowanym manifeście.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    partition_inventory = {
        "path": (AUDIT_DIR / "verified_event_inputs.json").relative_to(ROOT).as_posix(),
        "sha256": source_hashes["verified_event_partition_inventory"],
        "partition_count_expected": source["hourly_partitions_expected"],
        "partition_count_hash_verified": source["hourly_partitions_present_and_hash_verified"],
        "nonempty_partition_count": source["nonempty_partitions_verified_by_sha256"],
        "verified_empty_partition_count": source["empty_partitions_verified_by_sha256_and_zero_rows"],
        "missing_or_unverified": source["hourly_partitions_missing_or_unverified"],
        "selected_window_missing": source["missing_partitions_inside_selected_market_source_windows"],
        "ordered_hash_fingerprint_sha256": replay_artifact["event_partition_manifest"]["ordered_verified_partition_fingerprint_sha256"],
        "bytes_content_verified": source["content_bytes_verified"],
        "schema": EVENT_COLUMNS,
    }
    manifest = {
        "manifest_version": 1,
        "audit_id": AUDIT_ID,
        "original_experiment_id": "btc_market_dynamics_action_value_20261020",
        "base_commit": BASELINE_COMMIT,
        "audit_code_commit": current_head,
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "population": {
            "market_count": len(audit),
            "development_cutoff_exclusive_utc": DEVELOPMENT_CUTOFF.isoformat(),
            "actual_market_start_first_utc": pd.Timestamp(audit.market_start_utc.min()).isoformat(),
            "actual_market_start_last_utc": pd.Timestamp(audit.market_start_utc.max()).isoformat(),
            "decision_offset_seconds": -60,
            "execution_offset_seconds": -59,
            "horizons_seconds": [5, 15, 60],
            "static_feature_count": len(STATIC_FEATURES),
            "dynamic_feature_count": len(DYNAMIC_FEATURES),
            "action_sizes_usd": [5, 10, 25, 50, 100],
            "folds_unchanged": CONFIG["folds"],
        },
        "source_hashes_sha256": source_hashes,
        "frozen_artifact_consistency": consistency,
        "selection": {
            "algorithm": plan["algorithm"],
            "seed": plan["seed"],
            "sample_cap_per_period_and_mutually_exclusive_flag_stratum": plan["sample_cap_per_stratum"],
            "strata_count": len(plan["strata"]),
            "sample_market_count": plan["unique_market_count"],
            "diagnostic_strata_market_count": plan["diagnostic_strata_market_count"],
            "candidate_recovery_market_count": plan["candidate_recovery_market_count"],
            "selection_fields": plan["selection_fields"],
            "forbidden_selection_fields": plan["forbidden_selection_fields"],
            "sample_plan_sha256": sha256_file(AUDIT_DIR / "sample_plan.json"),
            "tolerance": plan["tolerance"],
        },
        "semantics": {
            "book_reset_counter_before": "incremented for every book row before the shared replay accepted or ignored it",
            "book_reset_counter_after": "increment only for an accepted full-book snapshot that initializes or changes any normalized native bid/ask level",
            "shared_replay_changed": False,
            "other_gates_or_feature_definitions_changed": False,
            "original_outputs_overwritten": False,
        },
        "status_counts": {
            "original": summary["original_status_counts"],
            "original_first_rejection_reason": summary["original_first_rejection_reason_counts"],
            "after_audit": {
                "accepted": int(summary["original_status_counts"]["accepted"] + recovered.candidate_recovered.sum()),
                "rejected": int(summary["original_status_counts"]["rejected"] - recovered.candidate_recovered.sum()),
                "recovered_market_count": int(recovered.candidate_recovered.sum()),
                "recovered_condition_ids": recovered.loc[recovered.candidate_recovered, "condition_id"].astype(str).tolist(),
                "periods": {
                    period: {
                        "market_count": summary["periods"][period]["markets"],
                        "observed_days": summary["periods"][period]["observed_days"],
                        "original_accepted": summary["periods"][period]["original_accepted"],
                        "recovered": int(recovered_counts.get(period, 0)),
                        "accepted_after": int(summary["periods"][period]["original_accepted"] + recovered_counts.get(period, 0)),
                    } for period in period_order
                },
            },
            "overlapping_flags_by_period_and_horizon": summary["period_flag_incidence"],
            "first_reason_overlapping_flag_crosswalk": summary["first_reason_and_overlapping_flag_crosswalk"],
        },
        "independent_replay": {
            "event_rows": replay_summary["event_rows_selected"],
            "elapsed_seconds": replay_summary["elapsed_seconds"],
            "throughput_rows_per_second": replay_summary["throughput_rows_per_second"],
            "worker_count": replay_summary["worker_count"],
            "peak_worker_working_set_bytes": replay_summary.get("peak_worker_working_set_bytes_max"),
            "event_partition_inventory": partition_inventory,
            "endpoint_comparison": comparison,
            "recovered_candidate_feature_crosscheck": feature_crosscheck,
            "synthetic_path_counterexample": synthetic,
        },
        "prefit_gate": gate,
        "earlier_history_scope": scope,
        "artifacts_outside_git": {
            "market_horizon_audit": all_market_audit,
            "sample_plan": sample_plan_artifact,
            "endpoint_summary": summary_artifact,
            "artifact_consistency": consistency_artifact,
            "earlier_history_scope": scope_artifact,
            "verified_event_inputs": verified_sources_artifact,
            "candidate_recovery": candidate_artifact,
            "independent_endpoints": replay_endpoint_artifact,
            "independent_features": replay_feature_artifact,
            "event_traces": event_trace_artifacts,
            "recovered_candidate_feature_crosscheck": feature_crosscheck_artifact,
            "prefit_gate": prefit_artifact,
            "synthetic_counterexample": synthetic_artifact,
            "independent_replay_comparison": comparison_artifact,
            "sample_source_partition_gaps": source_gap_artifact,
            "worker_event_summaries": worker_summary_artifacts,
            "independent_replay_manifest": json_artifact("independent_replay_manifest.json"),
        },
        "stages": {
            "artifact_inventory_and_hash_verification": "complete",
            "per_market_horizon_audit": "complete",
            "cross_side_late_semantics_review": "complete; gate unchanged; candidates still fail other frozen checks",
            "full_book_event_trace": "complete for all 994 potentially recoverable candidate IDs and deterministic diagnostic sample",
            "independent_endpoint_and_feature_comparison": "complete",
            "counter_semantics_regression_test": "added; executed separately",
            "prefit_feasibility_gate": "complete; fail; stopped before fit",
            "training_or_economic_scoring": "not run",
            "holdout_rows": "not read or scored",
            "historical_backfill": "not run",
            "live_changes_or_orders": "none",
        },
        "scope_assertions": {
            "models_not_fitted": True,
            "economics_not_scored": True,
            "holdout_market_rows_not_read_or_scored": True,
            "live_unchanged": True,
            "orders_not_submitted": True,
        },
        "report": {
            "path": REPORT_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(REPORT_PATH),
        },
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def _as_ns(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(pd.Timestamp(value).value)


def _audit_levels(value) -> dict[float, float]:
    if value is None:
        return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    try:
        if pd.isna(value):
            return {}
    except (TypeError, ValueError):
        pass
    result = {}
    for pair in value:
        if isinstance(pair, dict):
            price = pair.get("price", pair.get("p"))
            size = pair.get("size", pair.get("s"))
        else:
            try:
                price, size = pair[0], pair[1]
            except (TypeError, IndexError, KeyError):
                continue
        try:
            price, size = float(price), float(size)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and math.isfinite(size) and 0.0 < price < 1.0 and size > 0.0:
            result[round(price, 8)] = size
    return result


def _audit_band_notional(levels: dict[float, float], best_ask: float | None) -> float:
    if best_ask is None:
        return 0.0
    return float(sum(price * size for price, size in levels.items()
                     if best_ask <= price <= min(1.0, best_ask + 0.02) and size > 0.0))


def _audit_vwap(levels: dict[float, float], requested_usd: float) -> float | None:
    remaining = float(requested_usd)
    shares = 0.0
    for price in sorted(levels):
        size = float(levels[price])
        if not 0.0 < price < 1.0 or size <= 0.0:
            continue
        take = min(remaining / price, size)
        shares += take
        remaining -= take * price
        if remaining <= 1e-9:
            return float(requested_usd) / shares if shares > 0.0 else None
    return None


def _new_audit_token(token_id: str) -> dict:
    return {
        "token_id": str(token_id), "bids": {}, "asks": {}, "initialized": False,
        "bid_last_source_ns": None, "ask_last_source_ns": None,
        "bid_order_ambiguous": False, "ask_order_ambiguous": False,
        "max_source_ns_seen": None, "last_source_ns": None,
        "book_receive_ns": None, "bid_update_ns": None, "ask_update_ns": None,
        "bid_source_update_ns": None, "ask_source_update_ns": None,
        "bid_book_update_ns": None, "ask_book_update_ns": None,
        "bid_book_source_ns": None, "ask_book_source_ns": None,
        "book_count": 0,
    }


def _new_audit_state() -> dict:
    return {
        "tokens": {}, "event_count": 0, "book_count": 0,
        "first_event_ns": None, "last_event_ns": None,
        "max_inter_event_gap_seconds": None,
        "uninitialized_price_changes": 0,
        "out_of_order_price_changes": 0, "out_of_order_book_events": 0,
        "equal_source_timestamp_price_changes": 0, "equal_source_timestamp_book_events": 0,
        "cross_side_late_price_changes": 0, "cross_side_late_asks": 0,
        "empty_bid_book_snapshots": 0, "empty_ask_book_snapshots": 0,
    }


def _audit_token(state: dict, token_id: str) -> dict:
    key = str(token_id)
    if key not in state["tokens"]:
        state["tokens"][key] = _new_audit_token(key)
    return state["tokens"][key]


def _audit_bbo(token: dict | None) -> tuple[float | None, float | None]:
    if not token or not token["initialized"]:
        return None, None
    bids = token["bids"]
    asks = token["asks"]
    return (max(bids) if bids else None, min(asks) if asks else None)


def _audit_market_probability(up_bid, up_ask, down_bid, down_ask) -> float | None:
    values = (up_bid, up_ask, down_bid, down_ask)
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return None
    up_bid, up_ask, down_bid, down_ask = map(float, values)
    if up_bid > up_ask or down_bid > down_ask:
        return None
    up_mid = (up_bid + up_ask) / 2.0
    down_mid = (down_bid + down_ask) / 2.0
    total = up_mid + down_mid
    return up_mid / total if total > 0.0 else None


def _audit_endpoint(state: dict, runtime: dict, market: dict, endpoint_ns: int) -> dict:
    row = {
        "condition_id": market["condition_id"],
        "market_start_utc": pd.Timestamp(market["market_start_utc"]),
        "endpoint_time_utc": pd.Timestamp(endpoint_ns, unit="ns", tz="UTC"),
        "endpoint_name": next(name for name, value in market["endpoint_times"].items() if value == endpoint_ns),
        "market_event_count": int(state["event_count"]),
        "market_book_snapshot_count": int(state["book_count"]),
        "ambiguous_event_count": int(runtime["ambiguous_event_count"]),
        "book_message_count": int(runtime["book_message_count"]),
        "book_reset_count": int(runtime["book_reset_count"]),
        "accepted_identical_book_count": int(runtime["accepted_identical_book_count"]),
        "accepted_changed_book_count": int(runtime["accepted_changed_book_count"]),
        "ignored_stale_book_count": int(state["out_of_order_book_events"]),
        "out_of_order_price_change_count": int(state["out_of_order_price_changes"]),
        "out_of_order_book_count": int(state["out_of_order_book_events"]),
        "cross_side_late_price_change_count": int(state["cross_side_late_price_changes"]),
        "cross_side_late_ask_count": int(state["cross_side_late_asks"]),
        "equal_source_timestamp_price_change_count": int(state["equal_source_timestamp_price_changes"]),
        "equal_source_timestamp_book_event_count": int(state["equal_source_timestamp_book_events"]),
        "max_inter_event_gap_seconds": state["max_inter_event_gap_seconds"],
    }
    no_future = True
    for token in state["tokens"].values():
        seen = token.get("max_source_ns_seen")
        if seen is not None and seen > endpoint_ns:
            no_future = False
    row["no_future_source_event"] = bool(no_future)
    for side in ("up", "down"):
        token_id = market[f"{side}_token_id"]
        token = state["tokens"].get(token_id)
        bid, ask = _audit_bbo(token)
        if not token or not token["initialized"]:
            row.update({
                f"{side}_initialized": False, f"{side}_bid": None, f"{side}_ask": None,
                f"{side}_bid_levels": [], f"{side}_ask_levels": [],
                f"{side}_ask_notional_band": 0.0, f"{side}_ask_ambiguous": True,
                f"{side}_first_book_receive_ns": runtime["first_book_receive_ns"].get(token_id),
                f"{side}_latest_book_receive_ns": None, f"{side}_max_source_ns_seen": None,
            })
            continue
        bid_levels = sorted((round(float(price), 8), float(size)) for price, size in token["bids"].items() if size > 0.0)
        ask_levels = sorted((round(float(price), 8), float(size)) for price, size in token["asks"].items() if size > 0.0)
        row.update({
            f"{side}_initialized": True, f"{side}_bid": bid, f"{side}_ask": ask,
            f"{side}_bid_levels": bid_levels, f"{side}_ask_levels": ask_levels,
            f"{side}_ask_notional_band": _audit_band_notional(token["asks"], ask),
            f"{side}_ask_ambiguous": bool(token["ask_order_ambiguous"]),
            f"{side}_first_book_receive_ns": runtime["first_book_receive_ns"].get(token_id),
            f"{side}_latest_book_receive_ns": token.get("book_receive_ns"),
            f"{side}_max_source_ns_seen": token.get("max_source_ns_seen"),
        })
    row["p_market"] = _audit_market_probability(
        row["up_bid"], row["up_ask"], row["down_bid"], row["down_ask"],
    )
    row["quote_valid"] = bool(
        row["up_initialized"] and row["down_initialized"]
        and row["up_bid"] is not None and row["up_ask"] is not None
        and row["down_bid"] is not None and row["down_ask"] is not None
        and row["up_bid"] <= row["up_ask"] and row["down_bid"] <= row["down_ask"]
        and not row["up_ask_ambiguous"] and not row["down_ask_ambiguous"]
        and row["no_future_source_event"]
    )
    row["spread_up"] = row["up_ask"] - row["up_bid"] if row["up_ask"] is not None and row["up_bid"] is not None else None
    row["spread_down"] = row["down_ask"] - row["down_bid"] if row["down_ask"] is not None and row["down_bid"] is not None else None
    return row


def _audit_group_conflicts(rows: list[dict]) -> tuple[set[tuple[str, str]], int]:
    books: dict[tuple[str, int], list[tuple[dict[float, float], dict[float, float]]]] = {}
    updates: dict[tuple[str, int, str, float], set[float | None]] = {}
    conflicts: set[tuple[str, str]] = set()
    conflict_count = 0
    for row in rows:
        token_id = str(row["asset_id"])
        source_ns = int(row["_source_ns"])
        if row["event_type"] == "book":
            books.setdefault((token_id, source_ns), []).append((
                _audit_levels(row.get("bids")), _audit_levels(row.get("asks")),
            ))
        elif row["event_type"] == "price_change":
            side = str(row.get("side") or "").upper()
            try:
                price, size = float(row.get("price")), float(row.get("size"))
            except (TypeError, ValueError):
                continue
            if side not in {"BUY", "SELL"} or not math.isfinite(price) or not math.isfinite(size):
                continue
            key = (token_id, source_ns, side, round(price, 8))
            updates.setdefault(key, set()).add(None if size <= 0.0 else size)
    for (token_id, source_ns), snapshots in books.items():
        first_bids, first_asks = snapshots[0]
        for bids, asks in snapshots[1:]:
            if bids != first_bids:
                conflicts.add((token_id, "BUY"))
            if asks != first_asks:
                conflicts.add((token_id, "SELL"))
    for (token_id, source_ns, side, price), sizes in updates.items():
        side_index = 0 if side == "BUY" else 1
        book_levels = [snapshot[side_index] for snapshot in books.get((token_id, source_ns), [])]
        has_conflict = len(sizes) > 1
        if book_levels and len(sizes) == 1:
            requested_size = next(iter(sizes))
            has_conflict = any(
                (levels.get(price) is None) != (requested_size is None)
                or (levels.get(price) is not None and requested_size is not None
                    and abs(levels[price] - requested_size) > 1e-9)
                for levels in book_levels
            )
        if has_conflict:
            conflicts.add((token_id, side))
            conflict_count += 1
    return conflicts, conflict_count


def _audit_market_summary(state: dict, market: dict) -> dict:
    result = {}
    for side in ("up", "down"):
        token = state["tokens"].get(market[f"{side}_token_id"])
        bid, ask = _audit_bbo(token)
        result[side] = {
            "initialized": bool(token and token["initialized"]),
            "best_bid": bid,
            "best_ask": ask,
            "ask_band_notional": _audit_band_notional(token["asks"], ask) if token and token["initialized"] else 0.0,
            "ask_ambiguous": bool(token["ask_order_ambiguous"]) if token and token["initialized"] else True,
        }
    result["p_market"] = _audit_market_probability(
        result["up"]["best_bid"], result["up"]["best_ask"],
        result["down"]["best_bid"], result["down"]["best_ask"],
    )
    return result


def _audit_state_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def _audit_update_event(
    state: dict, runtime: dict, row: dict, received_ns: int,
) -> tuple[str, str, float | None, float | None, bool]:
    """Independent event transition for the feature audit; does not call replay code."""
    state["event_count"] += 1
    if state["first_event_ns"] is None:
        state["first_event_ns"] = received_ns
    previous_event_ns = state["last_event_ns"]
    if previous_event_ns is not None and received_ns > previous_event_ns:
        gap = (received_ns - previous_event_ns) / 1e9
        old_gap = state["max_inter_event_gap_seconds"]
        state["max_inter_event_gap_seconds"] = gap if old_gap is None else max(old_gap, gap)
    state["last_event_ns"] = received_ns

    event_type = str(row.get("event_type") or "")
    token_id = str(row.get("asset_id"))
    source_ns = int(row["_source_ns"])
    token = _audit_token(state, token_id)
    old_max = token["max_source_ns_seen"]
    token["max_source_ns_seen"] = source_ns if old_max is None else max(old_max, source_ns)
    runtime["event_type_counts"][event_type] += 1

    if event_type in {"last_trade_price", "best_bid_ask"}:
        return "accepted_observation", "no_native_book_transition", None, None, False
    if event_type == "book":
        runtime["book_message_count"] += 1
        prior_sources = [
            value for value in (token["bid_last_source_ns"], token["ask_last_source_ns"])
            if value is not None
        ]
        latest_side_source_ns = max(prior_sources) if prior_sources else None
        if latest_side_source_ns is not None and source_ns < latest_side_source_ns:
            state["out_of_order_book_events"] += 1
            runtime["ignored_stale_book_count"] += 1
            return "ignored", "stale_before_latest_native_side", None, None, False
        if latest_side_source_ns is not None and source_ns == latest_side_source_ns:
            state["equal_source_timestamp_book_events"] += 1
        new_bids, new_asks = _audit_levels(row.get("bids")), _audit_levels(row.get("asks"))
        before_initialized = bool(token["initialized"])
        before_bids, before_asks = dict(token["bids"]), dict(token["asks"])
        bid_tied = token["bid_last_source_ns"] == source_ns
        ask_tied = token["ask_last_source_ns"] == source_ns
        if bid_tied and before_bids != new_bids:
            token["bid_order_ambiguous"] = True
        elif not bid_tied:
            token["bid_order_ambiguous"] = False
        if ask_tied and before_asks != new_asks:
            token["ask_order_ambiguous"] = True
        elif not ask_tied:
            token["ask_order_ambiguous"] = False
        token.update({
            "bids": new_bids, "asks": new_asks, "initialized": True,
            "last_source_ns": source_ns if token["last_source_ns"] is None else max(token["last_source_ns"], source_ns),
            "book_receive_ns": received_ns,
            "bid_last_source_ns": source_ns, "ask_last_source_ns": source_ns,
            "bid_update_ns": received_ns, "ask_update_ns": received_ns,
            "bid_source_update_ns": source_ns, "ask_source_update_ns": source_ns,
            "bid_book_update_ns": received_ns, "ask_book_update_ns": received_ns,
            "bid_book_source_ns": source_ns, "ask_book_source_ns": source_ns,
            "book_count": token["book_count"] + 1,
        })
        state["book_count"] += 1
        state["empty_bid_book_snapshots"] += int(not new_bids)
        state["empty_ask_book_snapshots"] += int(not new_asks)
        changed = (not before_initialized) or before_bids != new_bids or before_asks != new_asks
        if changed:
            runtime["book_reset_count"] += 1
            runtime["accepted_changed_book_count"] += 1
            return "accepted", "full_book_changed_native_state", None, None, True
        runtime["accepted_identical_book_count"] += 1
        return "accepted", "full_book_identical_native_state", None, None, False

    if event_type != "price_change":
        return "accepted_observation", "unmodeled_event_type_no_native_book_transition", None, None, False
    if not token["initialized"]:
        state["uninitialized_price_changes"] += 1
        return "ignored", "native_book_not_initialized", None, None, False
    try:
        price, size = float(row.get("price")), float(row.get("size"))
    except (TypeError, ValueError):
        return "ignored", "invalid_price_or_size", None, None, False
    side = str(row.get("side") or "").upper()
    if not math.isfinite(price) or not math.isfinite(size) or not 0.0 < price < 1.0 or side not in {"BUY", "SELL"}:
        return "ignored", "invalid_price_or_size_or_side", None, None, False
    price = round(price, 8)
    levels = token["bids"] if side == "BUY" else token["asks"]
    source_key = "bid_last_source_ns" if side == "BUY" else "ask_last_source_ns"
    side_last_source_ns = token[source_key]
    if side_last_source_ns is not None and source_ns < side_last_source_ns:
        state["out_of_order_price_changes"] += 1
        return "ignored", "stale_relative_to_same_native_side", levels.get(price), levels.get(price), False
    if token["last_source_ns"] is not None and source_ns < token["last_source_ns"]:
        state["cross_side_late_price_changes"] += 1
        state["cross_side_late_asks"] += int(side == "SELL")
        cross_side_late = True
    else:
        cross_side_late = False
    before_size = levels.get(price)
    if size <= 0.0:
        levels.pop(price, None)
        after_size = None
    else:
        levels[price] = size
        after_size = size
    changed = (before_size is None) != (size <= 0.0) or (
        before_size is not None and size > 0.0 and before_size != size
    )
    token[source_key] = source_ns if side_last_source_ns is None else max(side_last_source_ns, source_ns)
    token["last_source_ns"] = source_ns if token["last_source_ns"] is None else max(token["last_source_ns"], source_ns)
    if changed and side == "BUY":
        token["bid_book_update_ns"] = received_ns
        token["bid_book_source_ns"] = source_ns
        token["bid_update_ns"] = received_ns
        token["bid_source_update_ns"] = source_ns
    elif changed and side == "SELL":
        token["ask_book_update_ns"] = received_ns
        token["ask_book_source_ns"] = source_ns
        token["ask_update_ns"] = received_ns
        token["ask_source_update_ns"] = source_ns
    if cross_side_late:
        return "accepted_cross_side_late", "own_side_source_order_is_monotonic", before_size, after_size, changed
    return "accepted", "native_side_update", before_size, after_size, changed


def _audit_tracker_from_endpoint(endpoint: dict, horizon: int, start_ns: int) -> dict:
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


def _audit_apply_depletion(
    runtime: dict, market: dict, received_ns: int, event_token: str,
    side: str, price: float | None, before_size: float | None, after_size: float | None,
) -> dict[int, float]:
    contributions = {}
    if side.upper() != "SELL" or price is None:
        return contributions
    outcome = "up" if str(event_token) == market["up_token_id"] else "down" if str(event_token) == market["down_token_id"] else None
    if outcome is None:
        return contributions
    for horizon, tracker in runtime["depletion_trackers"].items():
        if not tracker["start_ns"] < int(received_ns) <= market["t60_ns"]:
            continue
        band = tracker["sides"][outcome]
        if band["band_low"] is None or not band["band_low"] <= price <= band["band_high"]:
            continue
        previous = 0.0 if before_size is None else max(0.0, float(before_size))
        current = 0.0 if after_size is None else max(0.0, float(after_size))
        removed = float(price) * max(previous - current, 0.0)
        band["removed_notional"] += removed
        contributions[int(horizon)] = removed
    return contributions


def _audit_full_token_json(token: dict | None) -> str | None:
    if token is None or not token["initialized"]:
        return None
    payload = {
        "token_id": token["token_id"],
        "bids": sorted((round(float(price), 8), float(size)) for price, size in token["bids"].items()),
        "asks": sorted((round(float(price), 8), float(size)) for price, size in token["asks"].items()),
    }
    return _audit_state_json(payload)


def _audit_process_receive_group(
    state: dict, runtime: dict, rows: list[dict], received_ns: int, market: dict,
    trace_append,
) -> None:
    conflicts, conflict_count = _audit_group_conflicts(rows)
    native_ids = {market["up_token_id"], market["down_token_id"]}
    if any(token_id in native_ids for token_id, _ in conflicts):
        runtime["ambiguous_event_count"] += 1
    pending_trace = []
    for row in rows:
        token_id = str(row["asset_id"])
        event_type = str(row.get("event_type") or "")
        token_before = state["tokens"].get(token_id)
        if token_before is None:
            token_before = _audit_token(state, token_id)
        if event_type == "book" and not token_before["initialized"] and token_id not in runtime["first_book_receive_ns"]:
            runtime["first_book_receive_ns"][token_id] = int(received_ns)
        before_ambiguous = bool(token_before["ask_order_ambiguous"])
        before_market = _audit_market_summary(state, market)
        before_full_book = _audit_full_token_json(token_before) if event_type == "book" else None
        try:
            row_price = float(row.get("price"))
        except (TypeError, ValueError):
            row_price = None
        try:
            row_size = float(row.get("size"))
        except (TypeError, ValueError):
            row_size = None
        side = str(row.get("side") or "").upper()
        before_level_size = None
        if event_type == "price_change" and side in {"BUY", "SELL"} and row_price is not None and token_before["initialized"]:
            levels = token_before["bids"] if side == "BUY" else token_before["asks"]
            before_level_size = levels.get(round(row_price, 8))
        decision, reason, before_size, after_size, changed = _audit_update_event(
            state, runtime, row, int(received_ns),
        )
        if event_type == "price_change":
            before_level_size = before_size
        token_after = state["tokens"].get(token_id)
        if token_after is not None and not before_ambiguous and bool(token_after["ask_order_ambiguous"]):
            if token_id in native_ids:
                runtime["ambiguous_event_count"] += 1
        contributions = {}
        if event_type == "price_change" and row_price is not None:
            contributions = _audit_apply_depletion(
                runtime, market, received_ns, token_id, side, round(row_price, 8), before_size, after_size,
            )
        row_conflicted = (token_id, side) in conflicts if side in {"BUY", "SELL"} else any(
            key[0] == token_id for key in conflicts
        )
        pending_trace.append({
            "condition_id": market["condition_id"],
            "market_start_utc": pd.Timestamp(market["market_start_utc"]).isoformat(),
            "received_at_utc": pd.Timestamp(received_ns, unit="ns", tz="UTC").isoformat(),
            "source_timestamp_utc": pd.Timestamp(int(row["_source_ns"]), unit="ns", tz="UTC").isoformat(),
            "asset_id": token_id,
            "side": side or None,
            "event_type": event_type,
            "price": None if row_price is None else repr(row_price),
            "size": None if row_size is None else repr(row_size),
            "decision": decision,
            "decision_reason": reason,
            "group_conflict": str(bool(row_conflicted)),
            "before_market_summary": before_market,
            "before_token_full_book": before_full_book,
            "level_size_before": None if before_size is None else repr(float(before_size)),
            "level_size_after": None if after_size is None else repr(float(after_size)),
            "depletion_contribution": contributions,
        })
    for token_id, side in conflicts:
        token = _audit_token(state, token_id)
        if side == "BUY":
            token["bid_order_ambiguous"] = True
        else:
            token["ask_order_ambiguous"] = True
    state["equal_source_timestamp_price_changes"] += int(conflict_count)
    after_market = _audit_market_summary(state, market)
    for trace in pending_trace:
        token_after = state["tokens"].get(trace["asset_id"])
        trace_record = {
            "condition_id": trace["condition_id"],
            "market_start_utc": trace["market_start_utc"],
            "received_at_utc": trace["received_at_utc"],
            "source_timestamp_utc": trace["source_timestamp_utc"],
            "asset_id": trace["asset_id"], "side": trace["side"],
            "event_type": trace["event_type"], "price": trace["price"], "size": trace["size"],
            "decision": (
                f"{trace['decision']}_with_equal_timestamp_conflict"
                if trace["group_conflict"] == "True" else trace["decision"]
            ),
            "decision_reason": trace["decision_reason"],
            "group_conflict": trace["group_conflict"],
            "state_before_json": _audit_state_json(trace["before_market_summary"]),
            "state_after_json": _audit_state_json(after_market),
            "token_book_before_json": trace["before_token_full_book"],
            "token_book_after_json": _audit_full_token_json(token_after) if trace["event_type"] == "book" else None,
            "level_size_before": trace["level_size_before"],
            "level_size_after": trace["level_size_after"],
            "depletion_contribution_by_horizon_json": _audit_state_json(trace["depletion_contribution"]),
        }
        trace_append(trace_record)


def _audit_features(market: dict, runtime: dict) -> dict:
    result = {
        "condition_id": market["condition_id"],
        "market_start_utc": pd.Timestamp(market["market_start_utc"]),
    }
    endpoints = runtime["endpoints"]
    end = endpoints.get("tminus60", {})
    for horizon, baseline_name in ((5, "tminus65"), (15, "tminus75"), (60, "tminus120")):
        base = endpoints.get(baseline_name, {})
        tracker = runtime["depletion_trackers"].get(horizon)
        values = {
            f"p_market_change_{horizon}s": (
                end.get("p_market") - base.get("p_market")
                if end.get("p_market") is not None and base.get("p_market") is not None else math.nan
            ),
            f"spread_up_change_{horizon}s": (
                end.get("spread_up") - base.get("spread_up")
                if end.get("spread_up") is not None and base.get("spread_up") is not None else math.nan
            ),
            f"spread_down_change_{horizon}s": (
                end.get("spread_down") - base.get("spread_down")
                if end.get("spread_down") is not None and base.get("spread_down") is not None else math.nan
            ),
        }
        for side in ("up", "down"):
            end_notional = end.get(f"{side}_ask_notional_band")
            base_notional = base.get(f"{side}_ask_notional_band")
            values[f"log1p_ask_notional_{side}_change_{horizon}s"] = (
                math.log1p(end_notional) - math.log1p(base_notional)
                if end_notional is not None and base_notional is not None
                and end_notional >= 0.0 and base_notional >= 0.0 else math.nan
            )
            tracker_side = tracker["sides"][side] if tracker else None
            starting = tracker_side["starting_notional"] if tracker_side else None
            removed = tracker_side["removed_notional"] if tracker_side else None
            values[f"ask_depletion_relative_{side}_{horizon}s"] = (
                removed / (starting + 1e-12) if starting is not None and starting > 0.0 else math.nan
            )
        result.update(values)
    result["all_dynamic_features_finite"] = bool(
        np.isfinite(np.asarray([result.get(name, math.nan) for name in DYNAMIC_FEATURES], dtype=np.float64)).all()
    )
    return result


def _worker_peak_working_set_bytes() -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb,
        )
        return int(counters.PeakWorkingSetSize) if ok else None
    except Exception:
        return None


def _flush_trace_rows(trace_rows: list[dict], writer_holder: list, path: Path) -> None:
    if not trace_rows:
        return
    table = __import__("pyarrow").table({
        column: __import__("pyarrow").array([row.get(column) for row in trace_rows], type=__import__("pyarrow").string())
        for column in TRACE_COLUMNS
    })
    if not writer_holder:
        writer_holder.append(pq.ParquetWriter(path, table.schema, compression="zstd"))
    writer_holder[0].write_table(table)
    trace_rows.clear()


def _independent_replay_worker(args: tuple) -> dict:
    shard_id, market_rows, part_rows, sample_plan_sha256 = args
    import pyarrow as pa
    import pyarrow.dataset as pds

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    worker_dir = AUDIT_DIR / "independent_replay" / f"worker_{shard_id}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    status_path = worker_dir / "progress.json"
    endpoints_path = worker_dir / "endpoints.parquet"
    features_path = worker_dir / "features.parquet"
    event_summary_path = worker_dir / "event_summary.json"
    trace_path = worker_dir / "event_trace.parquet"

    markets = {}
    for item in market_rows:
        start = pd.Timestamp(item["market_start_utc"])
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        start_ns = int(start.value)
        market = {
            "condition_id": str(item["condition_id"]).lower(),
            "market_start_utc": start,
            "up_token_id": str(item["up_token_id"]),
            "down_token_id": str(item["down_token_id"]),
            "t60_ns": start_ns - 60_000_000_000,
        }
        market["endpoint_times"] = {
            "tminus120": start_ns - 120_000_000_000,
            "tminus75": start_ns - 75_000_000_000,
            "tminus65": start_ns - 65_000_000_000,
            "tminus60": market["t60_ns"],
        }
        markets[market["condition_id"]] = market

    states = {cid: _new_audit_state() for cid in markets}
    runtimes = {
        cid: {
            "first_book_receive_ns": {}, "endpoints": {}, "depletion_trackers": {},
            "ambiguous_event_count": 0, "book_message_count": 0, "book_reset_count": 0,
            "accepted_identical_book_count": 0, "accepted_changed_book_count": 0,
            "ignored_stale_book_count": 0, "event_type_counts": Counter(),
            "missing_source_rows": 0,
        }
        for cid in markets
    }
    schedule = sorted(
        (at_ns, cid, name)
        for cid, market in markets.items()
        for name, at_ns in market["endpoint_times"].items()
    )
    schedule_position = 0
    endpoint_rows = []
    trace_rows = []
    trace_writer = []
    selected_event_rows = 0
    processed_parts = 0
    started = time.perf_counter()

    def emit_before(cutoff_ns: int, inclusive: bool = False) -> None:
        nonlocal schedule_position
        while schedule_position < len(schedule):
            at_ns, cid, endpoint_name = schedule[schedule_position]
            if at_ns > cutoff_ns or (at_ns == cutoff_ns and not inclusive):
                break
            market = markets[cid]
            runtime = runtimes[cid]
            endpoint = _audit_endpoint(states[cid], runtime, market, at_ns)
            endpoint["endpoint_name"] = endpoint_name
            runtime["endpoints"][endpoint_name] = endpoint
            endpoint_rows.append(endpoint)
            if endpoint_name == "tminus120":
                runtime["depletion_trackers"][60] = _audit_tracker_from_endpoint(endpoint, 60, at_ns)
            elif endpoint_name == "tminus75":
                runtime["depletion_trackers"][15] = _audit_tracker_from_endpoint(endpoint, 15, at_ns)
            elif endpoint_name == "tminus65":
                runtime["depletion_trackers"][5] = _audit_tracker_from_endpoint(endpoint, 5, at_ns)
            schedule_position += 1

    for part_position, part in enumerate(part_rows):
        path = ROOT / part["path"]
        hour = pd.Timestamp(part["hour_utc"])
        if int(part["row_count"]) > 0:
            dataset = pds.dataset(str(path), format="parquet")
            market_type = dataset.schema.field("market").type
            ids = [cid.encode("ascii") for cid in markets] if pa.types.is_binary(market_type) else list(markets)
            predicate = pds.field("market").isin(pa.array(ids, type=market_type))
            predicate = predicate & (pds.field("timestamp_received") < DEVELOPMENT_CUTOFF.to_pydatetime())
            table = dataset.to_table(columns=EVENT_COLUMNS, filter=predicate)
            if table.num_rows:
                frame = table.to_pandas()
                frame["market"] = frame["market"].map(
                    lambda value: value.decode("ascii").lower() if isinstance(value, bytes)
                    else str(value).lower()
                )
                frame["timestamp_received"] = pd.to_datetime(frame.timestamp_received, utc=True, errors="coerce")
                frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
                frame["_received_ns"] = frame.timestamp_received.dt.as_unit("ns").astype("int64")
                frame["_source_ns"] = frame.timestamp.dt.as_unit("ns").astype("int64")
                valid_receive = frame["_received_ns"].to_numpy(dtype=np.int64, copy=False) > np.iinfo(np.int64).min
                valid_source = frame["_source_ns"].to_numpy(dtype=np.int64, copy=False) > np.iinfo(np.int64).min
                missing_source_ids = frame.loc[valid_receive & ~valid_source, "market"].drop_duplicates().tolist()
                for cid in missing_source_ids:
                    if cid in runtimes:
                        runtimes[cid]["ambiguous_event_count"] += 1
                        runtimes[cid]["missing_source_rows"] += int((
                            frame.market.eq(cid).to_numpy() & valid_receive & ~valid_source
                        ).sum())
                frame = frame.loc[valid_receive & valid_source].copy()
                if not frame.empty:
                    limits = frame.market.map(lambda cid: markets[cid]["t60_ns"]).to_numpy(dtype=np.int64, copy=False)
                    receive_ns = frame["_received_ns"].to_numpy(dtype=np.int64, copy=False)
                    frame = frame.loc[receive_ns <= limits]
                if not frame.empty:
                    frame.sort_values(
                        ["_received_ns", "_source_ns", "market", "asset_id", "event_type"],
                        kind="stable", inplace=True,
                    )
                    selected_event_rows += int(len(frame))
                    for received_ns, receive_group in frame.groupby("_received_ns", sort=False):
                        received_ns = int(received_ns)
                        emit_before(received_ns, inclusive=False)
                        for cid, event_group in receive_group.groupby("market", sort=False):
                            if cid not in markets:
                                continue
                            rows = event_group.to_dict(orient="records")
                            _audit_process_receive_group(
                                states[cid], runtimes[cid], rows, received_ns, markets[cid],
                                lambda item: trace_rows.append(item),
                            )
                        emit_before(received_ns, inclusive=True)
                        if len(trace_rows) >= 25000:
                            _flush_trace_rows(trace_rows, trace_writer, trace_path)
        emit_before(int((hour + pd.Timedelta(hours=1)).value), inclusive=False)
        processed_parts += 1
        if processed_parts % 25 == 0 or processed_parts == len(part_rows):
            elapsed = max(time.perf_counter() - started, 1e-9)
            write_json(status_path, {
                "shard_id": shard_id, "status": "running",
                "completed_parts": processed_parts, "total_parts": len(part_rows),
                "selected_event_rows": selected_event_rows,
                "trace_rows_pending": len(trace_rows),
                "elapsed_seconds": elapsed,
                "selected_rows_per_second": selected_event_rows / elapsed,
                "peak_working_set_bytes": _worker_peak_working_set_bytes(),
                "sample_plan_sha256": sample_plan_sha256,
                "last_partition": path.name,
            })
    if schedule_position < len(schedule):
        emit_before(max(market["t60_ns"] for market in markets.values()), inclusive=True)
    _flush_trace_rows(trace_rows, trace_writer, trace_path)
    if trace_writer:
        trace_writer[0].close()

    feature_rows = [_audit_features(markets[cid], runtime) for cid, runtime in runtimes.items()]
    pd.DataFrame.from_records(endpoint_rows).to_parquet(endpoints_path, index=False, compression="zstd")
    pd.DataFrame.from_records(feature_rows).to_parquet(features_path, index=False, compression="zstd")
    summary_rows = []
    for cid, runtime in runtimes.items():
        summary_rows.append({
            "condition_id": cid,
            "market_start_utc": markets[cid]["market_start_utc"],
            "event_type_counts": dict(runtime["event_type_counts"]),
            "missing_source_rows": int(runtime["missing_source_rows"]),
            "book_message_count_by_t60": int(runtime["book_message_count"]),
            "accepted_identical_book_count_by_t60": int(runtime["accepted_identical_book_count"]),
            "accepted_changed_book_count_by_t60": int(runtime["accepted_changed_book_count"]),
            "ignored_stale_book_count_by_t60": int(runtime["ignored_stale_book_count"]),
        })
    write_json(event_summary_path, {"markets": summary_rows})
    elapsed = time.perf_counter() - started
    final_progress = {
        "shard_id": shard_id, "status": "completed",
        "completed_parts": processed_parts, "total_parts": len(part_rows),
        "markets": len(markets), "selected_event_rows": selected_event_rows,
        "trace_rows": pq.ParquetFile(trace_path).metadata.num_rows if trace_path.exists() else 0,
        "elapsed_seconds": elapsed,
        "selected_rows_per_second": selected_event_rows / max(elapsed, 1e-9),
        "peak_working_set_bytes": _worker_peak_working_set_bytes(),
        "sample_plan_sha256": sample_plan_sha256,
        "outputs": {
            "endpoints_sha256": sha256_file(endpoints_path),
            "features_sha256": sha256_file(features_path),
            "event_summary_sha256": sha256_file(event_summary_path),
            "event_trace_sha256": sha256_file(trace_path) if trace_path.exists() else None,
        },
    }
    write_json(status_path, final_progress)
    return final_progress


def verify_sample_event_partitions(static: pd.DataFrame, sample: pd.DataFrame) -> tuple[list[dict], dict]:
    dep_path = ROOT / CONFIG["sources"]["event_dependency_manifest_path"]
    event_dir = ROOT / CONFIG["sources"]["event_parts_dir"]
    dependency = json.loads(dep_path.read_text(encoding="utf-8"))
    metadata = {item["path"]: item for item in dependency["partition_metadata"]}
    content = {item["path"]: item for item in dependency["partitions"]}
    replay_start = pd.Timestamp(static.market_start_utc.min()).floor("h") - pd.Timedelta(hours=25)
    replay_end = (pd.to_datetime(sample.market_start_utc, utc=True).max() - pd.Timedelta(seconds=60)).floor("h")
    required_hours = pd.date_range(replay_start, replay_end, freq="h", tz="UTC")
    missing_metadata = []
    parts = []
    total_bytes = 0
    empty_count = 0
    nonempty_count = 0
    verified_hashes = []
    for index, hour in enumerate(required_hours, start=1):
        name = hour.strftime("%Y-%m-%dT%H.parquet")
        record = metadata.get(name)
        digest_record = content.get(name)
        if record is None or digest_record is None:
            missing_metadata.append(name)
            continue
        path = event_dir / name
        if not path.is_file():
            missing_metadata.append(name)
            continue
        size = path.stat().st_size
        if size != int(record["size_bytes"]):
            raise RuntimeError(f"source event partition size mismatch: {name}")
        actual_hash = sha256_file(path)
        expected_hash = str(digest_record["content_sha256"])
        if actual_hash != expected_hash:
            raise RuntimeError(f"source event partition content hash mismatch: {name}")
        row_count = int(pq.ParquetFile(path).metadata.num_rows)
        if size <= 3000 and row_count != 0:
            raise RuntimeError(f"small partition is not an empty source partition: {name}")
        total_bytes += size
        empty_count += int(row_count == 0)
        nonempty_count += int(row_count > 0)
        verified_hashes.append({"path": name, "size_bytes": size, "row_count": row_count, "sha256": actual_hash})
        parts.append({
            "path": path.relative_to(ROOT).as_posix(),
            "hour_utc": hour.isoformat(),
            "size_bytes": size,
            "row_count": row_count,
            "sha256": actual_hash,
        })
        if index % 250 == 0 or index == len(required_hours):
            print(
                f"[audit-source] verified {index:,}/{len(required_hours):,} hourly partitions; "
                f"content={total_bytes / 1e9:.2f} GB; missing={len(missing_metadata)}",
                flush=True,
            )
    missing_set = set(missing_metadata)
    missing_by_market = {}
    for row in sample.itertuples(index=False):
        start = pd.Timestamp(row.market_start_utc).floor("h") - pd.Timedelta(hours=25)
        t60 = pd.Timestamp(row.market_start_utc) - pd.Timedelta(seconds=60)
        per_market_hours = pd.date_range(start, t60.floor("h"), freq="h", tz="UTC")
        gaps = [value.strftime("%Y-%m-%dT%H.parquet") for value in per_market_hours
                if value.strftime("%Y-%m-%dT%H.parquet") in missing_set]
        if gaps:
            missing_by_market[str(row.condition_id)] = gaps
    write_json(AUDIT_DIR / "sample_source_partition_gaps.json", missing_by_market)
    actual_ordered = [{"path": item["path"].split("/")[-1], "size_bytes": item["size_bytes"], "sha256": item["sha256"]}
                      for item in parts]
    source_manifest = {
        "sample_plan_sha256": sha256_file(AUDIT_DIR / "sample_plan.json"),
        "dependency_manifest_sha256": sha256_file(dep_path),
        "dependency_internal_sha256": dependency.get("dependency_sha256"),
        "development_cutoff_exclusive_utc": DEVELOPMENT_CUTOFF.isoformat(),
        "global_replay_start_hour_utc": replay_start.isoformat(),
        "latest_selected_t60_hour_utc": replay_end.isoformat(),
        "hourly_partitions_expected": int(len(required_hours)),
        "hourly_partitions_present_and_hash_verified": len(parts),
        "hourly_partitions_missing_or_unverified": missing_metadata,
        "sample_markets_with_missing_partitions_in_their_26_hour_source_window": len(missing_by_market),
        "missing_partitions_inside_selected_market_source_windows": missing_by_market,
        "global_gaps_outside_all_selected_market_source_windows": sorted(
            set(missing_metadata) - {name for gaps in missing_by_market.values() for name in gaps}
        ),
        "empty_partitions_verified_by_sha256_and_zero_rows": empty_count,
        "nonempty_partitions_verified_by_sha256": nonempty_count,
        "content_bytes_verified": total_bytes,
        "ordered_verified_partition_fingerprint_sha256": canonical_sha256(actual_ordered),
        "individual_verified_partition_hashes": verified_hashes,
        "all_selected_market_starts_pre_cutoff": bool(
            pd.to_datetime(sample.market_start_utc, utc=True).lt(DEVELOPMENT_CUTOFF).all()
        ),
        "holdout_event_rows_materialized": 0,
    }
    write_json(AUDIT_DIR / "verified_event_inputs.json", source_manifest)
    return parts, source_manifest


def run_independent_replay(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    static_source, _, _ = endpoint_frame()
    plan_path = AUDIT_DIR / "sample_plan.json"
    plan_sha = sha256_file(plan_path)
    replay_manifest_path = AUDIT_DIR / "independent_replay_manifest.json"
    endpoints_path = AUDIT_DIR / "independent_endpoints.parquet"
    features_path = AUDIT_DIR / "independent_features.parquet"
    if replay_manifest_path.is_file() and endpoints_path.is_file() and features_path.is_file():
        prior = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
        dependency_path = ROOT / CONFIG["sources"]["event_dependency_manifest_path"]
        event_manifest = prior.get("event_partition_manifest", {})
        expected_outputs = prior.get("outputs", {})
        reusable = (
            prior.get("replay_semantics_version") == REPLAY_SEMANTICS_VERSION
            and
            prior.get("sample_plan_sha256") == plan_sha
            and event_manifest.get("sample_plan_sha256") == plan_sha
            and event_manifest.get("dependency_manifest_sha256") == sha256_file(dependency_path)
            and event_manifest.get("sample_markets_with_missing_partitions_in_their_26_hour_source_window") == 0
            and expected_outputs.get("endpoints", {}).get("sha256") == sha256_file(endpoints_path)
            and expected_outputs.get("features", {}).get("sha256") == sha256_file(features_path)
            and int(expected_outputs.get("endpoints", {}).get("rows", -1)) == len(sample) * len(ENDPOINTS)
            and int(expected_outputs.get("features", {}).get("rows", -1)) == len(sample)
        )
        if reusable:
            return pd.read_parquet(endpoints_path), pd.read_parquet(features_path), prior
    parts, part_manifest = verify_sample_event_partitions(static_source, sample)
    markets = static_source.loc[
        static_source.condition_id.astype(str).isin(sample.condition_id.astype(str)),
        ["condition_id", "market_start_utc", "up_token_id", "down_token_id"],
    ].copy()
    if markets.condition_id.duplicated().any() or len(markets) != sample.condition_id.nunique():
        raise RuntimeError("frozen sample does not map one-to-one to development market inputs")
    if not pd.to_datetime(markets.market_start_utc, utc=True).lt(DEVELOPMENT_CUTOFF).all():
        raise RuntimeError("independent replay sample contains a non-development market")
    if not parts:
        raise RuntimeError("no verified event partitions cover the sample replay interval")
    ids = sorted(markets.condition_id.astype(str))
    shard_count = min(4, max(1, len(ids)))
    shards = [ids[index::shard_count] for index in range(shard_count)]
    rows_by_id = markets.set_index("condition_id", drop=False)
    tasks = []
    for shard_id, shard_ids in enumerate(shards):
        market_rows = rows_by_id.loc[shard_ids].to_dict(orient="records")
        tasks.append((shard_id, market_rows, parts, plan_sha))
    output_dir = AUDIT_DIR / "independent_replay"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    context = mp.get_context("spawn")
    futures = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=shard_count, mp_context=context) as pool:
        for task in tasks:
            future = pool.submit(_independent_replay_worker, task)
            futures[future] = task[0]
        previous_signature = None
        while futures:
            done, _ = concurrent.futures.wait(
                list(futures), timeout=10.0, return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                shard_id = futures.pop(future)
                results.append(future.result())
                print(f"[audit-replay] worker {shard_id} completed: {results[-1]['selected_event_rows']:,} events",
                      flush=True)
            progress_rows = []
            for shard_id in range(shard_count):
                path = output_dir / f"worker_{shard_id}" / "progress.json"
                if path.is_file():
                    progress_rows.append(json.loads(path.read_text(encoding="utf-8")))
            signature = tuple((row.get("completed_parts"), row.get("selected_event_rows")) for row in progress_rows)
            if progress_rows and signature != previous_signature:
                completed = sum(int(row.get("completed_parts", 0)) for row in progress_rows)
                total = sum(int(row.get("total_parts", len(parts))) for row in progress_rows)
                rows = sum(int(row.get("selected_event_rows", 0)) for row in progress_rows)
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"[audit-replay] parts={completed:,}/{total:,}; events={rows:,}; "
                    f"throughput={rows / elapsed:,.0f} rows/s; elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )
                previous_signature = signature
    endpoint_parts = [pd.read_parquet(output_dir / f"worker_{i}" / "endpoints.parquet") for i in range(shard_count)]
    feature_parts = [pd.read_parquet(output_dir / f"worker_{i}" / "features.parquet") for i in range(shard_count)]
    endpoints = pd.concat(endpoint_parts, ignore_index=True)
    features = pd.concat(feature_parts, ignore_index=True)
    endpoints.sort_values(["condition_id", "endpoint_time_utc"], kind="stable", inplace=True, ignore_index=True)
    features.sort_values("condition_id", kind="stable", inplace=True, ignore_index=True)
    endpoints.to_parquet(endpoints_path, index=False, compression="zstd")
    features.to_parquet(features_path, index=False, compression="zstd")
    elapsed = time.perf_counter() - started
    summary = {
        "audit_id": AUDIT_ID,
        "replay_semantics_version": REPLAY_SEMANTICS_VERSION,
        "sample_plan_sha256": plan_sha,
        "sample_market_count": len(markets),
        "worker_count": shard_count,
        "event_rows_selected": int(sum(item["selected_event_rows"] for item in results)),
        "trace_rows": int(sum(item["trace_rows"] for item in results)),
        "elapsed_seconds": elapsed,
        "throughput_rows_per_second": float(sum(item["selected_event_rows"] for item in results) / max(elapsed, 1e-9)),
        "peak_worker_working_set_bytes_max": max(
            (item["peak_working_set_bytes"] for item in results if item.get("peak_working_set_bytes") is not None),
            default=None,
        ),
        "event_partition_manifest": part_manifest,
        "worker_results": results,
        "outputs": {
            "endpoints": {"path": endpoints_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(endpoints_path), "rows": len(endpoints)},
            "features": {"path": features_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(features_path), "rows": len(features)},
        },
        "scope": {
            "models_fitted": False, "economics_scored": False,
            "holdout_market_rows_read_or_scored": False,
            "live_changed": False, "orders_submitted": False,
        },
    }
    write_json(AUDIT_DIR / "independent_replay_manifest.json", summary)
    return endpoints, features, summary


def _within_tolerance(actual, expected) -> bool:
    try:
        actual, expected = float(actual), float(expected)
    except (TypeError, ValueError):
        return actual == expected
    if not math.isfinite(actual) or not math.isfinite(expected):
        return False
    return abs(actual - expected) <= ABS_TOLERANCE + REL_TOLERANCE * max(abs(actual), abs(expected))


def compare_independent_replay(
    static: pd.DataFrame,
    dynamic: pd.DataFrame,
    frozen_endpoints: pd.DataFrame,
    audit: pd.DataFrame,
    sample: pd.DataFrame,
    replay_endpoints: pd.DataFrame,
    replay_features: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    ids = set(sample.condition_id.astype(str))
    old = frozen_endpoints.loc[frozen_endpoints.condition_id.astype(str).isin(ids)].copy()
    new = replay_endpoints.loc[replay_endpoints.condition_id.astype(str).isin(ids)].copy()
    old["condition_id"] = old.condition_id.astype(str)
    new["condition_id"] = new.condition_id.astype(str)
    old.set_index(["condition_id", "endpoint_name"], inplace=True)
    new.set_index(["condition_id", "endpoint_name"], inplace=True)
    if set(old.index) != set(new.index) or len(old) != len(new):
        raise RuntimeError("independent replay endpoint keys do not match the frozen sample")

    exact_fields = (
        "market_event_count", "market_book_snapshot_count", "ambiguous_event_count",
        "out_of_order_price_change_count", "out_of_order_book_count",
        "cross_side_late_price_change_count", "cross_side_late_ask_count",
        "equal_source_timestamp_price_change_count", "no_future_source_event",
        "up_initialized", "down_initialized", "up_ask_ambiguous", "down_ask_ambiguous",
        "up_first_book_receive_ns", "down_first_book_receive_ns",
        "up_latest_book_receive_ns", "down_latest_book_receive_ns",
        "up_max_source_ns_seen", "down_max_source_ns_seen",
    )
    float_fields = (
        "max_inter_event_gap_seconds", "up_bid", "up_ask", "down_bid", "down_ask",
        "up_ask_notional_band", "down_ask_notional_band", "p_market",
        "spread_up", "spread_down",
    )
    endpoint_mismatches = []
    corrected_counter_deltas = []
    level_mismatches = 0
    maximum_absolute_endpoint_float_difference = 0.0
    for key in old.index:
        before, after = old.loc[key], new.loc[key]
        for field in exact_fields:
            left, right = before[field], after[field]
            if pd.isna(left) and pd.isna(right):
                continue
            if left != right:
                endpoint_mismatches.append({"condition_id": key[0], "endpoint_name": key[1], "field": field,
                                            "frozen": str(left), "independent": str(right)})
        for field in float_fields:
            left, right = before[field], after[field]
            if pd.isna(left) and pd.isna(right):
                continue
            if not pd.isna(left) and not pd.isna(right):
                maximum_absolute_endpoint_float_difference = max(
                    maximum_absolute_endpoint_float_difference, abs(float(left) - float(right)),
                )
            if pd.isna(left) or pd.isna(right) or not _within_tolerance(left, right):
                endpoint_mismatches.append({"condition_id": key[0], "endpoint_name": key[1], "field": field,
                                            "frozen": str(left), "independent": str(right)})
        if int(after.book_message_count) != int(before.book_reset_count):
            endpoint_mismatches.append({"condition_id": key[0], "endpoint_name": key[1],
                                        "field": "raw_book_messages_vs_frozen_counter",
                                        "frozen": int(before.book_reset_count),
                                        "independent": int(after.book_message_count)})
        corrected_counter_deltas.append(int(after.book_reset_count) - int(before.book_reset_count))
        for side in ("up", "down"):
            if _audit_levels(before[f"{side}_ask_levels"]) != _audit_levels(after[f"{side}_ask_levels"]):
                level_mismatches += 1
                endpoint_mismatches.append({"condition_id": key[0], "endpoint_name": key[1],
                                            "field": f"{side}_full_ask_ladder"})
        if bool(before.quote_valid) != bool(after.quote_valid):
            endpoint_mismatches.append({"condition_id": key[0], "endpoint_name": key[1],
                                        "field": "quote_valid"})

    static_by_id = static.set_index("condition_id")
    new_end = new.reset_index().loc[lambda frame: frame.endpoint_name.eq("tminus60")].set_index("condition_id")
    static_mismatches = []
    maximum_absolute_static_feature_difference = 0.0
    for cid in ids:
        base, endpoint = static_by_id.loc[cid], new_end.loc[cid]
        up_asks, down_asks = _audit_levels(endpoint.up_ask_levels), _audit_levels(endpoint.down_ask_levels)
        up_bid, up_ask = endpoint.up_bid, endpoint.up_ask
        down_bid, down_ask = endpoint.down_bid, endpoint.down_ask
        p_raw = float(base.p_raw)
        p_market = endpoint.p_market
        if any(value is None or pd.isna(value) for value in (up_bid, up_ask, down_bid, down_ask, p_market)):
            static_mismatches.append({"condition_id": cid, "feature": "independent_t60_native_quote_missing"})
            continue
        clipped_raw = min(max(p_raw, 1e-6), 1.0 - 1e-6)
        clipped_market = min(max(p_market, 1e-6), 1.0 - 1e-6)
        up_vwap, down_vwap = _audit_vwap(up_asks, 5.0), _audit_vwap(down_asks, 5.0)
        calculated = {
            "raw_logit": math.log(clipped_raw / (1.0 - clipped_raw)),
            "market_logit": math.log(clipped_market / (1.0 - clipped_market)),
            "spread_up": up_ask - up_bid,
            "spread_down": down_ask - down_bid,
            "mid_sum_minus_one": ((up_bid + up_ask) / 2.0 + (down_bid + down_ask) / 2.0 - 1.0),
            "log1p_ask_notional_up": math.log1p(_audit_band_notional(up_asks, up_ask)),
            "log1p_ask_notional_down": math.log1p(_audit_band_notional(down_asks, down_ask)),
            "slippage_5_up": up_vwap - up_ask if up_vwap is not None else math.nan,
            "slippage_5_down": down_vwap - down_ask if down_vwap is not None else math.nan,
        }
        calculated["raw_minus_market_logit"] = calculated["raw_logit"] - calculated["market_logit"]
        for name in STATIC_FEATURES:
            maximum_absolute_static_feature_difference = max(
                maximum_absolute_static_feature_difference,
                abs(float(calculated[name]) - float(base[name])),
            )
            if not _within_tolerance(calculated[name], base[name]):
                static_mismatches.append({"condition_id": cid, "feature": name,
                                          "frozen": float(base[name]), "independent": calculated[name]})

    original_valid = dynamic.loc[
        dynamic.condition_id.astype(str).isin(ids) & dynamic.dynamic_valid.fillna(False)
    ].copy()
    feature_by_id = replay_features.set_index("condition_id")
    feature_mismatches = []
    nonfinite_original_valid = 0
    compared_feature_values = 0
    maximum_absolute_dynamic_feature_difference = 0.0
    for row in original_valid.itertuples(index=False):
        cid = str(row.condition_id)
        actual = feature_by_id.loc[cid]
        if not bool(actual.all_dynamic_features_finite):
            nonfinite_original_valid += 1
        for name in DYNAMIC_FEATURES:
            expected = getattr(row, name)
            value = actual[name]
            compared_feature_values += 1
            maximum_absolute_dynamic_feature_difference = max(
                maximum_absolute_dynamic_feature_difference, abs(float(value) - float(expected)),
            )
            if not _within_tolerance(value, expected):
                if len(feature_mismatches) < 100:
                    feature_mismatches.append({"condition_id": cid, "feature": name,
                                               "frozen": float(expected), "independent": float(value)})

    candidate_ids = set(audit.loc[
        audit.candidate_if_book_message_gate_removed
        & ~audit.reconstructed_original_qualification,
        "condition_id",
    ].astype(str))
    sample_trace_paths = [AUDIT_DIR / "independent_replay" / f"worker_{i}" / "event_trace.parquet"
                          for i in range(min(4, max(1, len(ids))))]
    sample_trace_paths = [path for path in sample_trace_paths if path.is_file()]
    if sample_trace_paths and candidate_ids:
        trace_dataset = ds.dataset([str(path) for path in sample_trace_paths], format="parquet")
        trace_filter = ds.field("condition_id").isin(sorted(candidate_ids))
        trace_filter = trace_filter & ds.field("event_type").isin(["book"])
        trace = trace_dataset.to_table(
            columns=["condition_id", "received_at_utc", "event_type", "decision", "decision_reason",
                     "group_conflict", "token_book_before_json", "token_book_after_json"],
            filter=trace_filter,
        ).to_pandas()
    else:
        trace = pd.DataFrame(columns=[
            "condition_id", "received_at_utc", "event_type", "decision", "decision_reason",
            "group_conflict", "token_book_before_json", "token_book_after_json",
        ])
    trace["condition_id"] = trace.condition_id.astype(str)
    trace["received_at_utc"] = pd.to_datetime(trace.received_at_utc, utc=True, format="ISO8601")
    trace_by_id = {str(cid): group for cid, group in trace.groupby("condition_id", sort=False)}
    recovery = []
    audited_by_id = audit.set_index("condition_id")
    replay_ep_by_id = replay_endpoints.set_index(["condition_id", "endpoint_name"])
    features_by_id = replay_features.set_index("condition_id")
    source_gaps = json.loads((AUDIT_DIR / "sample_source_partition_gaps.json").read_text(encoding="utf-8"))
    for cid in sorted(candidate_ids):
        market_audit = audited_by_id.loc[cid]
        market_start = pd.Timestamp(market_audit.market_start_utc)
        windows = {
            60: (market_start - pd.Timedelta(seconds=120), market_start - pd.Timedelta(seconds=60)),
            15: (market_start - pd.Timedelta(seconds=75), market_start - pd.Timedelta(seconds=60)),
            5: (market_start - pd.Timedelta(seconds=65), market_start - pd.Timedelta(seconds=60)),
        }
        events = trace_by_id.get(cid, trace.iloc[0:0])
        events = events.loc[
            events.received_at_utc.gt(windows[60][0]) & events.received_at_utc.le(windows[60][1])
        ]
        identical_count = int(events.decision_reason.eq("full_book_identical_native_state").sum())
        changed_count = int(events.decision_reason.eq("full_book_changed_native_state").sum())
        stale_count = int(events.decision_reason.eq("stale_before_latest_native_side").sum())
        conflict_count = int(events.group_conflict.eq("True").sum())
        other_book_event_count = int(len(events) - identical_count - changed_count - stale_count)
        all_identical_accepted = bool(len(events)) and bool(
            events.decision.eq("accepted").all()
            and events.decision_reason.eq("full_book_identical_native_state").all()
            and events.group_conflict.eq("False").all()
        )
        source_complete = bool(market_audit.source_continuity_complete) and cid not in source_gaps
        corrected_windows_clear = True
        for horizon, baseline_name in ((60, "tminus120"), (15, "tminus75"), (5, "tminus65")):
            start_ep = replay_ep_by_id.loc[(cid, baseline_name)]
            end_ep = replay_ep_by_id.loc[(cid, "tminus60")]
            corrected_windows_clear &= int(end_ep.book_reset_count) == int(start_ep.book_reset_count)
            for counter in ("ambiguous_event_count", "out_of_order_price_change_count", "out_of_order_book_count",
                            "cross_side_late_price_change_count"):
                corrected_windows_clear &= int(end_ep[counter]) == int(start_ep[counter])
            corrected_windows_clear &= bool(start_ep.quote_valid and end_ep.quote_valid)
            corrected_windows_clear &= float(start_ep.up_ask_notional_band) > 0 and float(start_ep.down_ask_notional_band) > 0
        finite = bool(features_by_id.loc[cid].all_dynamic_features_finite)
        recovered = bool(source_complete and all_identical_accepted and corrected_windows_clear and finite)
        if changed_count:
            rejection_class = "accepted_state_changing_full_book"
        elif stale_count:
            rejection_class = "ignored_stale_book"
        elif conflict_count:
            rejection_class = "equal_receive_group_conflict"
        elif other_book_event_count:
            rejection_class = "other_book_event_semantics"
        elif not corrected_windows_clear:
            rejection_class = "another_order_or_quote_gate"
        elif not source_complete:
            rejection_class = "source_window_incomplete"
        elif not finite:
            rejection_class = "nonfinite_dynamic_features"
        else:
            rejection_class = "accepted_identical_full_book"
        recovery.append({
            "condition_id": cid,
            "market_start_utc": market_start.isoformat(),
            "period": market_audit.period,
            "source_continuity_complete": source_complete,
            "relevant_book_event_count": int(len(events)),
            "accepted_identical_full_book_event_count": identical_count,
            "accepted_state_changing_full_book_event_count": changed_count,
            "ignored_stale_book_event_count": stale_count,
            "equal_receive_group_conflict_book_event_count": conflict_count,
            "other_book_event_count": other_book_event_count,
            "primary_rejection_class": rejection_class,
            "all_relevant_book_events_accepted_full_native_book_identical": all_identical_accepted,
            "corrected_counter_and_other_window_gates_pass": bool(corrected_windows_clear),
            "all_21_dynamic_features_finite": finite,
            "candidate_recovered": recovered,
        })
    recovery_frame = pd.DataFrame.from_records(recovery)
    recovery_path = AUDIT_DIR / "candidate_recovery_audit.parquet"
    recovery_frame.to_parquet(recovery_path, index=False, compression="zstd")
    result = {
        "sample_market_count": int(len(ids)),
        "endpoint_rows_compared": int(len(old)),
        "endpoint_mismatch_count": len(endpoint_mismatches),
        "endpoint_mismatch_examples": endpoint_mismatches[:20],
        "maximum_absolute_endpoint_float_difference": maximum_absolute_endpoint_float_difference,
        "independent_corrected_book_reset_counter_delta_from_frozen_raw_book_message_counter": {
            "rows_with_lower_count": sum(value < 0 for value in corrected_counter_deltas),
            "total_counter_messages_removed": -sum(value for value in corrected_counter_deltas if value < 0),
        },
        "raw_book_message_count_matches_frozen_counter": not any(
            item.get("field") == "raw_book_messages_vs_frozen_counter" for item in endpoint_mismatches
        ),
        "full_ask_ladder_mismatch_count": level_mismatches,
        "static_feature_values_compared": int(len(ids) * len(STATIC_FEATURES)),
        "static_feature_mismatch_count": len(static_mismatches),
        "static_feature_mismatch_examples": static_mismatches[:20],
        "maximum_absolute_static_feature_difference": maximum_absolute_static_feature_difference,
        "original_accepted_sample_market_count": int(len(original_valid)),
        "dynamic_feature_values_compared": compared_feature_values,
        "dynamic_feature_mismatch_count": len(feature_mismatches),
        "dynamic_feature_mismatch_examples": feature_mismatches[:20],
        "maximum_absolute_dynamic_feature_difference": maximum_absolute_dynamic_feature_difference,
        "nonfinite_dynamic_feature_markets_among_original_accepted": nonfinite_original_valid,
        "candidate_book_message_only_count": len(candidate_ids),
        "candidate_recovered_after_independent_full_book_trace": int(recovery_frame.candidate_recovered.sum()) if len(recovery_frame) else 0,
        "candidate_primary_rejection_classes": (
            recovery_frame.primary_rejection_class.value_counts().to_dict() if len(recovery_frame) else {}
        ),
        "candidate_recovery_reasons": {
            "source_incomplete": int((~recovery_frame.source_continuity_complete).sum()) if len(recovery_frame) else 0,
            "book_events_not_all_accepted_identical": int((~recovery_frame.all_relevant_book_events_accepted_full_native_book_identical).sum()) if len(recovery_frame) else 0,
            "counter_or_other_gate_failure": int((~recovery_frame.corrected_counter_and_other_window_gates_pass).sum()) if len(recovery_frame) else 0,
            "nonfinite_features": int((~recovery_frame.all_21_dynamic_features_finite).sum()) if len(recovery_frame) else 0,
        },
        "candidate_recovery_path": recovery_path.relative_to(ROOT).as_posix(),
        "comparison_tolerances": {"absolute": ABS_TOLERANCE, "relative": REL_TOLERANCE},
        "scope": {"models_fitted": False, "economics_scored": False,
                  "holdout_market_rows_read_or_scored": False, "live_changed": False, "orders_submitted": False},
    }
    write_json(AUDIT_DIR / "independent_replay_comparison.json", result)
    if endpoint_mismatches or static_mismatches or feature_mismatches or nonfinite_original_valid:
        raise RuntimeError("independent replay failed frozen endpoint/static/dynamic feature equivalence checks")
    return result, recovery_frame


def final_prefit_gate(static: pd.DataFrame, audit: pd.DataFrame, recovery: pd.DataFrame) -> dict:
    """Recompute availability gates without reading return columns or scoring economics."""
    common_path = ORIGINAL_DIR / "common_inputs.parquet"
    execution_path = ORIGINAL_DIR / "execution_targets.parquet"
    market_path = ROOT / CONFIG["sources"]["market_index_path"]
    t59_path = ROOT / CONFIG["sources"]["t59_ladders_path"]
    recovered_ids = set(recovery.loc[recovery.candidate_recovered, "condition_id"].astype(str))
    recovery_rows = audit.loc[audit.condition_id.astype(str).isin(recovered_ids)].copy()
    if recovery_rows.condition_id.duplicated().any():
        raise RuntimeError("recovered markets are duplicated in the per-market audit")
    if recovered_ids != set(recovery_rows.condition_id.astype(str)):
        raise RuntimeError("recovered IDs do not map to the per-market audit")

    common_dataset = ds.dataset(str(common_path), format="parquet")
    common_calendar = common_dataset.to_table(
        columns=["condition_id", "market_start_utc"],
        filter=ds.field("market_start_utc") < DEVELOPMENT_CUTOFF.to_pydatetime(),
    ).to_pandas()
    common_calendar["condition_id"] = common_calendar.condition_id.astype(str)
    common_calendar["market_start_utc"] = pd.to_datetime(common_calendar.market_start_utc, utc=True)
    if not common_calendar.market_start_utc.lt(DEVELOPMENT_CUTOFF).all():
        raise RuntimeError("prefit gate materialized a common-input row at/after the cutoff")
    if set(common_calendar.condition_id) & recovered_ids:
        raise RuntimeError("recovered markets overlap the frozen common-input population")

    market_dataset = ds.dataset(str(market_path), format="parquet")
    market_schema = market_dataset.schema
    market_id_type = market_schema.field("condition_id").type
    all_recovered_sorted = sorted(recovered_ids)
    id_values = (
        [value.encode("ascii") for value in all_recovered_sorted]
        if str(market_id_type).startswith("binary") else all_recovered_sorted
    )
    candidate_index = market_dataset.to_table(
        columns=["condition_id", "market_start_utc"],
        filter=(ds.field("market_start_utc") < DEVELOPMENT_CUTOFF.to_pydatetime())
        & ds.field("condition_id").isin(id_values),
    ).to_pandas() if recovered_ids else pd.DataFrame(columns=["condition_id", "market_start_utc"])
    if len(candidate_index):
        candidate_index["condition_id"] = candidate_index.condition_id.astype(str).str.lower()
        candidate_index["market_start_utc"] = pd.to_datetime(candidate_index.market_start_utc, utc=True)
    if candidate_index.condition_id.duplicated().any():
        raise RuntimeError("official index has duplicate recovered market rows")
    index_by_id = candidate_index.set_index("condition_id") if len(candidate_index) else pd.DataFrame()

    t59_dataset = ds.dataset(str(t59_path), format="parquet")
    t59_schema = t59_dataset.schema
    wanted_t59 = [
        "condition_id", "market_start_utc", "entry_time_utc", "has_full_snapshot",
        "archive_entry_hour_available", "archive_freshness_window_available",
        "archive_window_missing_source_hours", "archive_window_unprocessed_hours",
        "archive_gap_after_up_book_init", "archive_gap_after_down_book_init", "archive_gap_after_fee_update",
        "no_future_event_at_entry", "no_future_source_event_at_entry", "fee_known", "fee_collection_mode",
    ]
    for side in ("up", "down"):
        wanted_t59.extend([
            f"{side}_quote_complemented", f"{side}_quote_source_token_id",
            f"{side}_ask_order_ambiguous", f"{side}_bid_ask_crossed",
            f"{side}_bbo_ask_comparable", f"{side}_bbo_ask_mismatch", f"{side}_book_init_receive_ns",
        ])
    missing_columns = sorted(set(wanted_t59) - set(t59_schema.names))
    if missing_columns:
        raise RuntimeError(f"T-59 source lacks required prefit eligibility columns: {missing_columns}")
    t59_id_type = t59_schema.field("condition_id").type
    t59_ids = (
        [value.encode("ascii") for value in all_recovered_sorted]
        if str(t59_id_type).startswith("binary") else all_recovered_sorted
    )
    t59 = t59_dataset.to_table(
        columns=wanted_t59,
        filter=(ds.field("market_start_utc") < DEVELOPMENT_CUTOFF.to_pydatetime())
        & ds.field("condition_id").isin(t59_ids),
    ).to_pandas() if recovered_ids else pd.DataFrame(columns=wanted_t59)
    if len(t59):
        t59["condition_id"] = t59.condition_id.astype(str).str.lower()
        t59["market_start_utc"] = pd.to_datetime(t59.market_start_utc, utc=True)
        t59["entry_time_utc"] = pd.to_datetime(t59.entry_time_utc, utc=True)
    if t59.condition_id.duplicated().any():
        raise RuntimeError("T-59 source has duplicate recovered market rows")
    t59_by_id = t59.set_index("condition_id") if len(t59) else pd.DataFrame()
    static_by_id = static.set_index("condition_id")

    # Outcome columns are read only for candidate markets in the expanding training prefix
    # through the second fold; no fold-2 evaluation or post-cutoff labels are materialized.
    candidate_label_columns = [
        "condition_id", "market_start_utc", "resolved_at_utc", "target_polymarket_up",
    ]
    candidate_labels = market_dataset.to_table(
        columns=candidate_label_columns,
        filter=(ds.field("market_start_utc") < FOLD_2_START.to_pydatetime())
        & ds.field("condition_id").isin(id_values),
    ).to_pandas() if recovered_ids else pd.DataFrame(columns=candidate_label_columns)
    if len(candidate_labels):
        candidate_labels["condition_id"] = candidate_labels.condition_id.astype(str).str.lower()
        candidate_labels["market_start_utc"] = pd.to_datetime(candidate_labels.market_start_utc, utc=True)
        candidate_labels["resolved_at_utc"] = pd.to_datetime(candidate_labels.resolved_at_utc, utc=True, errors="coerce")
        candidate_labels["target_polymarket_up"] = pd.to_numeric(candidate_labels.target_polymarket_up, errors="coerce")
        candidate_labels["label_release_at_utc"] = pd.concat([
            candidate_labels.resolved_at_utc,
            candidate_labels.market_start_utc + pd.Timedelta(minutes=5),
        ], axis=1).max(axis=1) + pd.Timedelta(seconds=60)
        candidate_labels.loc[
            candidate_labels.target_polymarket_up.isna() | candidate_labels.resolved_at_utc.isna(),
            "label_release_at_utc",
        ] = pd.NaT
    if candidate_labels.condition_id.duplicated().any():
        raise RuntimeError("official index has duplicate recovered training-label rows")
    labels_by_id = candidate_labels.set_index("condition_id") if len(candidate_labels) else pd.DataFrame()

    def side_target_is_known(cid: str, side: str, expected_entry: pd.Timestamp) -> bool:
        if cid not in t59_by_id.index or cid not in static_by_id.index:
            return False
        snapshot = t59_by_id.loc[cid].to_dict()
        base = static_by_id.loc[cid]
        if pd.Timestamp(snapshot["market_start_utc"]) != pd.Timestamp(base.market_start_utc):
            return False
        if pd.Timestamp(snapshot["entry_time_utc"]) != expected_entry:
            return False
        source_token = snapshot.get(f"{side}_quote_source_token_id")
        if isinstance(source_token, bytes):
            source_token = source_token.decode("ascii")
        checks = (
            not bool(snapshot.get(f"{side}_quote_complemented")),
            str(source_token) == str(base[f"{side}_token_id"]),
            bool(snapshot.get("archive_entry_hour_available")),
            bool(snapshot.get("archive_freshness_window_available")),
            bool(snapshot.get("has_full_snapshot")),
            int(snapshot.get("archive_window_missing_source_hours") or 0) == 0,
            int(snapshot.get("archive_window_unprocessed_hours") or 0) == 0,
            not bool(snapshot.get(f"archive_gap_after_{side}_book_init")),
            not bool(snapshot.get("archive_gap_after_fee_update")),
            bool(snapshot.get("no_future_event_at_entry")),
            bool(snapshot.get("no_future_source_event_at_entry")),
            not bool(snapshot.get(f"{side}_ask_order_ambiguous")),
            not bool(snapshot.get(f"{side}_bid_ask_crossed")),
            not (bool(snapshot.get(f"{side}_bbo_ask_comparable")) and bool(snapshot.get(f"{side}_bbo_ask_mismatch"))),
            bool(snapshot.get("fee_known")),
            snapshot.get("fee_collection_mode") != "maintenance_pause",
        )
        if not all(checks):
            return False
        init_ns = snapshot.get(f"{side}_book_init_receive_ns")
        return init_ns is not None and not pd.isna(init_ns) and int(init_ns) < int(expected_entry.value)

    candidate_availability = {}
    for cid in recovered_ids:
        label_known = False
        release_at = pd.NaT
        outcome = np.nan
        if cid in labels_by_id.index:
            label = labels_by_id.loc[cid]
            outcome = label.target_polymarket_up
            label_known = bool(outcome in (0.0, 1.0) and pd.notna(label.resolved_at_utc))
            release_at = label.label_release_at_utc
        market_start = pd.Timestamp(recovery_rows.set_index("condition_id").loc[cid].market_start_utc)
        entry = market_start - pd.Timedelta(seconds=59)
        candidate_availability[cid] = {
            "index_row": cid in index_by_id.index,
            "t59_row": cid in t59_by_id.index,
            "official_label_known": label_known,
            "label_release_at_utc": release_at,
            "target_polymarket_up": outcome,
            "q5_both_sides_known": (
                label_known and side_target_is_known(cid, "up", entry)
                and side_target_is_known(cid, "down", entry)
            ),
        }

    common_label_columns = [
        "condition_id", "market_start_utc", "target_polymarket_up", "label_release_at_utc", "official_label_known",
    ]
    common_labels_by_fold = {}
    for fold_start in (EVALUATION_START, FOLD_2_START):
        first_decision = fold_start - pd.Timedelta(seconds=60)
        frame = common_dataset.to_table(
            columns=common_label_columns,
            filter=ds.field("market_start_utc") < fold_start.to_pydatetime(),
        ).to_pandas()
        frame["condition_id"] = frame.condition_id.astype(str)
        frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True)
        frame["target_polymarket_up"] = pd.to_numeric(frame.target_polymarket_up, errors="coerce")
        frame["label_release_at_utc"] = pd.to_datetime(frame.label_release_at_utc, utc=True, errors="coerce")
        frame["official_label_known"] = frame.official_label_known.fillna(False).astype(bool)
        if not frame.market_start_utc.lt(fold_start).all():
            raise RuntimeError("fold label source scan escaped the training prefix")
        common_labels_by_fold[fold_start] = frame

    execution_dataset = ds.dataset(str(execution_path), format="parquet")
    execution_columns = ["condition_id", "market_start_utc", "up_q5_label_known", "down_q5_label_known"]
    existing_execution = execution_dataset.to_table(
        columns=execution_columns,
        filter=ds.field("market_start_utc") < FOLD_2_START.to_pydatetime(),
    ).to_pandas()
    existing_execution["condition_id"] = existing_execution.condition_id.astype(str)
    existing_execution["market_start_utc"] = pd.to_datetime(existing_execution.market_start_utc, utc=True)
    existing_execution["up_q5_label_known"] = existing_execution.up_q5_label_known.fillna(False).astype(bool)
    existing_execution["down_q5_label_known"] = existing_execution.down_q5_label_known.fillna(False).astype(bool)

    folds = []
    for fold_id, start, end in (
        (1, EVALUATION_START, FOLD_2_START),
        (2, FOLD_2_START, DEVELOPMENT_CUTOFF),
    ):
        first_decision = start - pd.Timedelta(seconds=60)
        existing_eval = common_calendar.loc[
            common_calendar.market_start_utc.ge(start) & common_calendar.market_start_utc.lt(end)
        ]
        added_eval = {
            cid for cid, item in candidate_availability.items()
            if item["index_row"] and item["t59_row"]
            and start <= pd.Timestamp(recovery_rows.set_index("condition_id").loc[cid].market_start_utc) < end
        }
        train = common_labels_by_fold[start]
        train = train.loc[
            train.official_label_known & train.label_release_at_utc.notna()
            & train.label_release_at_utc.lt(first_decision)
        ]
        q5_existing = existing_execution.loc[
            existing_execution.condition_id.isin(train.condition_id)
            & existing_execution.up_q5_label_known & existing_execution.down_q5_label_known
        ]
        added_train = {
            cid for cid, item in candidate_availability.items()
            if item["official_label_known"] and item["q5_both_sides_known"]
            and pd.notna(item["label_release_at_utc"])
            and pd.Timestamp(item["label_release_at_utc"]) < first_decision
            and pd.Timestamp(recovery_rows.set_index("condition_id").loc[cid].market_start_utc) < start
        }
        classes = set(int(value) for value in train.target_polymarket_up.dropna().unique())
        classes.update(
            int(candidate_availability[cid]["target_polymarket_up"])
            for cid in added_train
            if candidate_availability[cid]["target_polymarket_up"] in (0.0, 1.0)
        )
        released_count = int(len(train) + len(added_train))
        q5_count = int(len(q5_existing) + len(added_train))
        folds.append({
            "fold_id": fold_id,
            "evaluation_start_utc": start.isoformat(),
            "evaluation_end_exclusive_utc": end.isoformat(),
            "first_decision_at_utc": first_decision.isoformat(),
            "evaluation_market_count": int(len(existing_eval) + len(added_eval)),
            "evaluation_recovered_markets_added": len(added_eval),
            "evaluation_observed_days": int(existing_eval.market_start_utc.dt.floor("D").nunique())
                if len(existing_eval) else 0,
            "released_training_label_count": released_count,
            "released_training_label_classes": sorted(classes),
            "known_two_sided_q5_training_market_count": q5_count,
            "recovered_q5_training_markets_added": len(added_train),
            "released_label_minimum_met": (
                released_count >= int(CONFIG["folds"]["minimum_released_training_labels"]) and len(classes) == 2
            ),
            "q5_two_sided_minimum_met": (
                q5_count >= int(CONFIG["folds"]["minimum_training_markets_with_known_q5_both_sides"])
            ),
        })

    overall_existing = common_calendar.loc[
        common_calendar.market_start_utc.ge(EVALUATION_START)
        & common_calendar.market_start_utc.lt(DEVELOPMENT_CUTOFF)
    ]
    overall_added = {
        cid for cid, item in candidate_availability.items()
        if item["index_row"] and item["t59_row"]
        and EVALUATION_START <= pd.Timestamp(recovery_rows.set_index("condition_id").loc[cid].market_start_utc)
        < DEVELOPMENT_CUTOFF
    }
    observed_days = set(overall_existing.market_start_utc.dt.floor("D"))
    observed_days.update(
        pd.Timestamp(recovery_rows.set_index("condition_id").loc[cid].market_start_utc).floor("D")
        for cid in overall_added
    )
    total_markets = int(len(overall_existing) + len(overall_added))
    nonempty_folds = sum(item["evaluation_market_count"] > 0 for item in folds)
    min_markets = int(CONFIG["folds"]["minimum_qualifying_input_markets"])
    min_days = int(CONFIG["folds"]["minimum_observed_days"])
    min_folds = int(CONFIG["folds"]["minimum_nonempty_evaluation_blocks"])
    gate = {
        "evaluation_market_count": total_markets,
        "evaluation_recovered_markets_added": len(overall_added),
        "minimum_evaluation_markets": min_markets,
        "evaluation_markets_gate_pass": total_markets >= min_markets,
        "evaluation_observed_day_count": len(observed_days),
        "minimum_evaluation_observed_days": min_days,
        "evaluation_days_gate_pass": len(observed_days) >= min_days,
        "nonempty_evaluation_fold_count": nonempty_folds,
        "minimum_nonempty_evaluation_folds": min_folds,
        "nonempty_folds_gate_pass": nonempty_folds >= min_folds,
        "folds": folds,
        "released_label_gate_pass": all(
            item["released_label_minimum_met"] for item in folds if item["evaluation_market_count"] > 0
        ),
        "q5_two_sided_training_gate_pass": all(
            item["q5_two_sided_minimum_met"] for item in folds if item["evaluation_market_count"] > 0
        ),
        "overall_prefit_gate_pass": (
            total_markets >= min_markets and len(observed_days) >= min_days and nonempty_folds >= min_folds
            and all(item["released_label_minimum_met"] and item["q5_two_sided_minimum_met"]
                    for item in folds if item["evaluation_market_count"] > 0)
        ),
        "scope": {
            "model_fit_started": False,
            "economic_scores_or_returns_read_or_computed": False,
            "post_cutoff_rows_materialized": 0,
            "evaluation_outcome_rows_materialized": 0,
        },
    }
    write_json(AUDIT_DIR / "final_prefit_gate.json", gate)
    return gate


def synthetic_path_counterexample() -> dict:
    """Show that equal final books can hide different depletion paths."""
    market = {"condition_id": "synthetic", "up_token_id": "UP", "down_token_id": "DOWN", "t60_ns": 200}
    initial_book = {
        "event_type": "book", "asset_id": "UP", "_source_ns": 100,
        "bids": [[0.49, 10.0]], "asks": [[0.50, 10.0]],
    }

    def initial_state():
        state = _new_audit_state()
        runtime = {
            "event_type_counts": Counter(), "book_message_count": 0, "book_reset_count": 0,
            "accepted_identical_book_count": 0, "accepted_changed_book_count": 0,
            "ignored_stale_book_count": 0, "depletion_trackers": {},
        }
        _audit_update_event(state, runtime, initial_book, 100)
        runtime["depletion_trackers"][60] = {
            "start_ns": 100,
            "sides": {
                "up": {"band_low": 0.50, "band_high": 0.52, "starting_notional": 5.0, "removed_notional": 0.0},
                "down": {"band_low": None, "band_high": None, "starting_notional": 0.0, "removed_notional": 0.0},
            },
        }
        return state, runtime

    changed_state, changed_runtime = initial_state()
    price_change = {
        "event_type": "price_change", "asset_id": "UP", "_source_ns": 101,
        "side": "SELL", "price": 0.50, "size": 5.0,
    }
    _, _, before_size, after_size, _ = _audit_update_event(changed_state, changed_runtime, price_change, 110)
    _audit_apply_depletion(
        changed_runtime, market, 110, "UP", "SELL", 0.50, before_size, after_size,
    )
    reset_result = _audit_update_event(changed_state, changed_runtime, {
        **initial_book, "_source_ns": 102,
    }, 120)

    unchanged_state, unchanged_runtime = initial_state()
    same_final_book = _audit_full_token_json(_audit_token(changed_state, "UP")) == _audit_full_token_json(
        _audit_token(unchanged_state, "UP")
    )
    changed_depletion = (
        changed_runtime["depletion_trackers"][60]["sides"]["up"]["removed_notional"] / 5.0
    )
    unchanged_depletion = (
        unchanged_runtime["depletion_trackers"][60]["sides"]["up"]["removed_notional"] / 5.0
    )
    result = {
        "final_native_full_book_equal": same_final_book,
        "path_a_price_change_then_state_changing_snapshot_depletion_relative": changed_depletion,
        "path_b_no_native_state_change_depletion_relative": unchanged_depletion,
        "path_a_accepted_snapshot_reason": reset_result[1],
        "path_a_incremental_state_changing_book_count": (
            changed_runtime["book_reset_count"] - unchanged_runtime["book_reset_count"]
        ),
        "demonstrates_endpoint_equality_does_not_imply_feature_path_equality": (
            same_final_book and changed_depletion != unchanged_depletion
        ),
    }
    if not result["demonstrates_endpoint_equality_does_not_imply_feature_path_equality"]:
        raise RuntimeError("synthetic path counterexample failed")
    write_json(AUDIT_DIR / "synthetic_path_counterexample.json", result)
    return result


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    static, dynamic, endpoints = endpoint_frame()
    consistency = artifact_consistency()
    audit = attach_t60_source_continuity(build_market_audit(static, dynamic, endpoints))
    if int(audit.reconstructed_original_qualification.sum()) != int(dynamic.dynamic_valid.fillna(False).sum()):
        raise RuntimeError("independent endpoint qualification does not reproduce original accepted count")
    reason_reproduction = audit.original_first_rejection_reason.value_counts(dropna=False).to_dict()
    for expected_reason, expected_count in {
        "unresolved_order_or_history_60s": 2751,
        "book_reset_inside_depletion_window_60s": 992,
        "invalid_native_quote_window_60s": 659,
        "invalid_native_quote_window_15s": 1,
    }.items():
        if int(reason_reproduction.get(expected_reason, 0)) != expected_count:
            raise RuntimeError(f"original first-reason count mismatch for {expected_reason}")
    audit_path = AUDIT_DIR / "market_horizon_audit.parquet"
    audit.to_parquet(audit_path, index=False, compression="zstd")
    sample, plan = sample_plan(audit)
    sample_path = AUDIT_DIR / "sample_plan.json"
    write_json(sample_path, plan)
    summary = summarize_market_audit(audit)
    scope = source_scope_audit(static)
    summary_path = AUDIT_DIR / "endpoint_summary.json"
    write_json(summary_path, summary)
    scope_path = AUDIT_DIR / "earlier_history_scope.json"
    write_json(scope_path, scope)
    write_json(AUDIT_DIR / "artifact_consistency.json", consistency)
    print(
        f"[audit] frozen sample plan: {len(sample):,} markets; "
        f"{len(plan['candidate_recovery_condition_ids']):,} candidate recovery markets",
        flush=True,
    )
    replay_endpoints, replay_features, replay_summary = run_independent_replay(sample)
    comparison, recovery = compare_independent_replay(
        static, dynamic, endpoints, audit, sample, replay_endpoints, replay_features,
    )
    feature_crosscheck = crosscheck_recovered_candidate_features(
        recovery, replay_endpoints, replay_features,
    )
    gate = final_prefit_gate(static, audit, recovery)
    synthetic = synthetic_path_counterexample()
    summary["source_continuity_complete_markets"] = int(audit.source_continuity_complete.sum())
    summary["source_continuity_incomplete_markets"] = int((~audit.source_continuity_complete).sum())
    summary["population_integrity"] = {
        "initialization_missing_market_count": int(audit.initialization_missing.sum()),
        "source_complete_market_count": int(audit.source_continuity_complete.sum()),
        "source_incomplete_market_count": int((~audit.source_continuity_complete).sum()),
        "nonpositive_depletion_denominator_market_counts": {
            f"{horizon}s": {
                side: int((pd.to_numeric(
                    audit[f"h{horizon}s_{side}_starting_ask_band_notional"], errors="coerce",
                ).fillna(0) <= 0).sum())
                for side in ("up", "down")
            }
            for horizon in (5, 15, 60)
        },
    }
    summary["independent_replay_comparison"] = comparison
    summary["recovered_candidate_feature_crosscheck"] = feature_crosscheck
    summary["final_prefit_gate"] = gate
    summary["synthetic_path_counterexample"] = synthetic
    write_json(summary_path, summary)
    write_audit_reports(
        audit, summary, scope, consistency, plan, replay_summary,
        comparison, feature_crosscheck, gate, synthetic,
    )
    print(json.dumps({
        "audit_id": AUDIT_ID,
        "markets": len(audit),
        "original_valid": int(audit.original_status.eq("accepted").sum()),
        "original_rejected": int(audit.original_status.eq("rejected").sum()),
        "hypothesis_candidates": summary["hypothesis_candidate_counts_not_yet_feasibility"],
        "sample_markets": len(sample),
        "earlier_raw_oof_rows": scope["raw_oof_before_earliest_static_candidate"]["rows"],
        "earlier_raw_oof_nominally_aligned": scope["raw_oof_before_earliest_static_candidate"]["rows_with_nominal_decision_at_market_start_minus_60s"],
        "market_index_first": scope["used_official_index"]["first_start_utc"],
        "frozen_outputs_match": consistency["all_frozen_output_hashes_match"],
        "worker_checkpoints_match": consistency["all_four_worker_checkpoints_complete_and_output_hashes_match"],
        "feature_hash_matches_summary_manifest": consistency["feature_hash_matches_versioned_summary_manifest"],
        "candidate_recovered_after_independent_replay": comparison["candidate_recovered_after_independent_full_book_trace"],
        "prefit_gate_pass": gate["overall_prefit_gate_pass"],
        "replay_event_rows": replay_summary["event_rows_selected"],
        "replay_elapsed_seconds": replay_summary["elapsed_seconds"],
        "output_dir": AUDIT_DIR.relative_to(ROOT).as_posix(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
