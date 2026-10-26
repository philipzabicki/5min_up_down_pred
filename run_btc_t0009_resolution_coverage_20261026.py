"""T0009 source-only samples and development resolution-economics audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import run_btc_preopen_t59_reassessment as t59
import run_btc_preopen_economic_replay as economic_replay


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/research/btc_t0009_resolution_coverage_20261026.json"
RUN_MODE = "coverage"
OUTPUT_ROOT = ROOT / "data/analysis/polymarket/BTC"
GAMMA_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261007/gamma_markets.parquet"
PMXT_ROOT = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/pmxt_extended"
PMXT_PARTS = PMXT_ROOT / "event_parts"
PMXT_ARCHIVE_MANIFEST = PMXT_ROOT / "archive_manifest.json"
PMXT_MARKET_INDEX_PATH = PMXT_ROOT / "market_index.parquet"
MATCHES_PATH = ROOT / "data/analysis/polymarket/BTC/btc_candidate_polymarket_target_20261023/confirmation_market_matches.parquet"
OBADIAHA_ROOT = ROOT / "data/raw/polymarket/obadiaha/11793901f0ac89c5a6c51123a6ccd29a3aaf8f4c"
DEVELOPMENT_START_UTC = pd.Timestamp("2025-07-01T00:00:00Z")
DEVELOPMENT_END_UTC = pd.Timestamp("2026-07-01T00:00:00Z")
T60_ENTRY_CASE = "t60_decision_book"
T59_ENTRY_CASE = "t59_native_asks"
FALLBACK_MINIMUM_USD = 5.0
FILL_TOLERANCE_USD = 1e-6
SAMPLE_PMXT_ID = "0xfe21d75161d71bec075cedee0f0d6b2cd57beeebbba939ece415da2ef965f978"
SAMPLE_OBADIAHA_ID = "0x63aa6cc4782925cf7ea6631422ddcfc0acd1770fb4ae6c046bd4f9af53effa08"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol() -> tuple[dict, str]:
    payload = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(payload.decode("utf-8"))
    return protocol, hashlib.sha256(payload).hexdigest()


def _date_filter(dataset: ds.Dataset, field: str = "market_start_utc"):
    arrow_type = dataset.schema.field(field).type
    start = pa.scalar(DEVELOPMENT_START_UTC.to_pydatetime(), type=arrow_type)
    end = pa.scalar(DEVELOPMENT_END_UTC.to_pydatetime(), type=arrow_type)
    return (ds.field(field) >= start) & (ds.field(field) < end)


def _load_development_universe() -> tuple[pd.DataFrame, dict]:
    gamma_dataset = ds.dataset(GAMMA_PATH, format="parquet")
    gamma = gamma_dataset.to_table(
        columns=["condition_id", "market_slug", "market_start_utc", "up_token_id", "down_token_id"],
        filter=_date_filter(gamma_dataset),
    ).to_pandas()
    gamma["condition_id"] = gamma.condition_id.astype("string").str.lower()
    gamma["market_start_utc"] = pd.to_datetime(gamma.market_start_utc, utc=True)
    if gamma.condition_id.isna().any() or gamma.condition_id.duplicated().any():
        raise RuntimeError("Development Gamma market IDs are missing or duplicated")

    match_dataset = ds.dataset(MATCHES_PATH, format="parquet")
    candidate_columns = [
        "condition_id", "market_start_utc", "nominal_decision_time_utc",
        "ensemble_probability_A", "ensemble_probability_B",
        "ensemble_probability_C", "ensemble_probability_D", "official_label_valid",
    ]
    match_table = match_dataset.to_table(
        columns=candidate_columns,
        filter=_date_filter(match_dataset),
    )
    matches = match_table.to_pandas()
    matches["condition_id"] = matches.condition_id.astype("string").str.lower()
    no_id_rows = matches.loc[matches.condition_id.isna()].copy()
    candidates = matches.loc[matches.condition_id.notna()].copy()
    candidates["candidate_market_start_utc"] = pd.to_datetime(
        candidates.market_start_utc, utc=True,
    )
    candidates.drop(columns=["market_start_utc"], inplace=True)
    if candidates.condition_id.duplicated().any():
        raise RuntimeError("Saved candidate predictions have duplicate Gamma condition IDs")
    candidates["prediction_present"] = candidates[[
        "ensemble_probability_A", "ensemble_probability_B",
        "ensemble_probability_C", "ensemble_probability_D",
    ]].notna().all(axis=1)
    candidates["prediction_ready_timestamp_present"] = False

    universe = gamma.merge(candidates, on="condition_id", how="left", validate="one_to_one")
    market_time_match = universe.candidate_market_start_utc.isna() | universe.market_start_utc.eq(
        universe.candidate_market_start_utc
    )
    if not market_time_match.all():
        raise RuntimeError("Saved prediction market start differs from its exact Gamma market join")
    universe["prediction_present"] = universe.prediction_present.fillna(False).astype(bool)
    universe["prediction_ready_timestamp_present"] = False
    universe["official_label_valid"] = universe.official_label_valid.fillna(False).astype(bool)
    universe["token_mapping_present"] = (
        universe.up_token_id.notna() & universe.down_token_id.notna()
        & universe.up_token_id.astype("string").str.len().gt(0)
        & universe.down_token_id.astype("string").str.len().gt(0)
    )
    no_id_rows["month_utc"] = pd.to_datetime(no_id_rows.market_start_utc, utc=True).dt.strftime("%Y-%m")
    no_id_monthly = no_id_rows.groupby("month_utc", dropna=False).size().to_dict()
    identity = {
        "gamma_markets": len(gamma),
        "candidate_rows_with_condition_id": len(candidates),
        "candidate_rows_without_condition_id": len(no_id_rows),
        "candidate_condition_ids_not_in_gamma": int((~candidates.condition_id.isin(gamma.condition_id)).sum()),
        "gamma_markets_without_saved_candidate_row": int((~gamma.condition_id.isin(candidates.condition_id)).sum()),
        "candidate_probability_rows_missing_any_A_to_D": int((~candidates.prediction_present).sum()),
        "duplicate_gamma_condition_ids": 0,
        "duplicate_candidate_condition_ids": 0,
        "candidate_calendar_rows_without_gamma_id_by_month": {
            str(month): int(count) for month, count in sorted(no_id_monthly.items())
        },
    }
    if identity["candidate_condition_ids_not_in_gamma"]:
        raise RuntimeError("A saved candidate condition ID has no exact development Gamma row")
    return universe, identity


def _make_development_replay_index(
    universe: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    archive_dataset = ds.dataset(PMXT_MARKET_INDEX_PATH, format="parquet")
    archive_columns = ["condition_id", "archive_source"]
    archive = archive_dataset.to_table(
        columns=archive_columns,
        filter=_date_filter(archive_dataset),
    ).to_pandas()
    archive["condition_id"] = archive.condition_id.astype("string").str.lower()
    if archive.condition_id.duplicated().any():
        raise RuntimeError("Frozen PMXT index has duplicate development condition IDs")

    identity_columns = [
        "condition_id", "market_slug", "market_start_utc", "up_token_id", "down_token_id",
    ]
    replay_index = universe[identity_columns].merge(
        archive, on="condition_id", how="left", validate="one_to_one",
    )
    replay_index["archive_source"] = replay_index.archive_source.fillna("no_local_pmxt_index_row")
    replay_index["archive_entry_hour_available"] = replay_index.archive_entry_hour_available.fillna(False).astype(bool)
    replay_index["archive_freshness_window_available"] = replay_index.archive_freshness_window_available.fillna(False).astype(bool)
    replay_index["resolved_at_utc"] = pd.NaT
    replay_index["target_polymarket_up"] = None
    replay_index["p_model_raw"] = None
    replay_index["p_model_platt"] = None
    replay_index["p_candidate_raw"] = None
    replay_index["p_candidate_platt"] = None
    replay_index["up_token_id"] = replay_index.up_token_id.fillna("").astype(str)
    replay_index["down_token_id"] = replay_index.down_token_id.fillna("").astype(str)
    replay_index["archive_source"] = replay_index.archive_source.fillna("no_local_pmxt_index_row")
    replay_index.sort_values("market_start_utc", kind="stable", inplace=True)
    replay_index.reset_index(drop=True, inplace=True)

    part_dir = output_dir / "development_event_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    development_parts = [
        path for path in sorted(PMXT_PARTS.glob("*.parquet"))
        if path.stem[:10] < "2026-07-01"
    ]
    for source in development_parts:
        destination = part_dir / source.name
        if destination.exists():
            if not os.path.samefile(source, destination):
                raise RuntimeError(f"Development event part path is occupied: {destination}")
        else:
            os.link(source, destination)
    return replay_index, development_parts


def _replay_development_sources(
    replay_index: pd.DataFrame,
    part_dir: Path,
    output_dir: Path,
    archive_manifest_hash: str,
    protocol_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    replay_dir = output_dir / "pmxt_replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    old_out_dir, old_workers = t59.OUT_DIR, t59.T59_REPLAY_WORKERS
    t59.OUT_DIR = replay_dir
    t59.T59_REPLAY_WORKERS = 6
    try:
        snapshots_t59, replay_manifest = t59._replay_t59(
            replay_index,
            part_dir,
            archive_source_manifest_sha256=archive_manifest_hash,
        )
    finally:
        t59.OUT_DIR = old_out_dir
        t59.T59_REPLAY_WORKERS = old_workers

    snapshots_t60 = pd.read_parquet(replay_dir / "t60_decision_books.parquet")
    for frame in (snapshots_t60, snapshots_t59):
        frame.drop(columns=[
            "prediction_available_at_utc", "resolved_at_utc", "target_polymarket_up",
            "p_model_raw", "p_model_platt", "p_candidate_raw", "p_candidate_platt",
        ], errors="ignore", inplace=True)
    if len(snapshots_t60) != len(replay_index) or len(snapshots_t59) != len(replay_index):
        raise RuntimeError("PMXT replay did not capture both decision and execution rows for every Gamma market")
    snapshots_t60.to_parquet(replay_dir / "t60_decision_books.parquet", index=False, compression="zstd")
    snapshots_t59.to_parquet(replay_dir / "t59_ask_ladders.parquet", index=False, compression="zstd")
    replay_manifest["synthetic_nominal_prediction_available_at_column_removed"] = True
    replay_manifest["outcome_and_prediction_columns_read"] = False
    replay_manifest["holdout_event_partitions_read"] = 0
    replay_manifest["protocol_sha256"] = protocol_hash
    replay_manifest["decision_snapshot_sha256"] = _sha256(replay_dir / "t60_decision_books.parquet")
    replay_manifest["snapshot_sha256"] = _sha256(replay_dir / "t59_ask_ladders.parquet")
    (replay_dir / "t59_replay_manifest.json").write_text(
        json.dumps(replay_manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return snapshots_t60, snapshots_t59, replay_manifest


def _development_hour_sets(protocol: dict, development_parts: list[Path]) -> tuple[set[pd.Timestamp], set[pd.Timestamp]]:
    archive_manifest = json.loads(PMXT_ARCHIVE_MANIFEST.read_text(encoding="utf-8"))
    advertised = {
        pd.Timestamp(value).floor("h")
        for value in archive_manifest["pmxt_v2"]["available_hours_utc"]
        if pd.Timestamp(value) < DEVELOPMENT_END_UTC
    }
    processed = {
        pd.Timestamp(path.stem.replace("T", " "), tz="UTC")
        for path in development_parts
    }
    if not processed.issubset(advertised):
        raise RuntimeError("A local development PMXT part is absent from the frozen advertised-hour index")
    expected_missing = {
        pd.Timestamp(value).floor("h")
        for value in protocol["inputs"]["pmxt_archive"]["pmxt_v2_missing_hours"]
    }
    if not expected_missing.issubset(advertised - processed):
        raise RuntimeError("Frozen PMXT missing-hour record disagrees with development event parts")
    return advertised, processed


def _window_coverage(start: pd.Timestamp, entry: pd.Timestamp, advertised: set[pd.Timestamp], processed: set[pd.Timestamp]) -> dict:
    lower = start.floor("h") - pd.Timedelta(hours=25)
    expected = set(pd.date_range(lower, entry.floor("h"), freq="h"))
    source_count = len(expected & advertised)
    processed_count = len(expected & processed)
    recent = pd.date_range((entry - pd.Timedelta(seconds=30)).floor("h"), entry.floor("h"), freq="h")
    return {
        "archive_entry_hour_available": entry.floor("h") in processed,
        "archive_freshness_window_available": all(hour in processed for hour in recent),
        "archive_window_hours_expected": len(expected),
        "archive_window_hours_source_available": source_count,
        "archive_window_hours_processed": processed_count,
        "archive_window_missing_source_hours": len(expected) - source_count,
        "archive_window_unprocessed_hours": source_count - processed_count,
    }


def _gap_after_timestamp(value, entry: pd.Timestamp, processed: set[pd.Timestamp]) -> bool | None:
    if value is None or pd.isna(value):
        return None
    initialized_hour = pd.Timestamp(int(value), unit="ns", tz="UTC").floor("h")
    entry_hour = entry.floor("h")
    if initialized_hour > entry_hour:
        return True
    return any(hour not in processed for hour in pd.date_range(initialized_hour, entry_hour, freq="h"))


def _add_snapshot_archive_coverage(
    snapshots: pd.DataFrame,
    case: str,
    advertised: set[pd.Timestamp],
    processed: set[pd.Timestamp],
) -> pd.DataFrame:
    frame = snapshots.copy()
    offset = 60 if case == "t60" else 59
    start_values = pd.to_datetime(frame.market_start_utc, utc=True)
    entries = start_values - pd.Timedelta(seconds=offset)
    coverage_rows = []
    for position, row in enumerate(frame.itertuples(index=False)):
        start = pd.Timestamp(start_values.iloc[position])
        entry = pd.Timestamp(entries.iloc[position])
        window = _window_coverage(start, entry, advertised, processed)
        window[f"archive_gap_after_up_book_init"] = _gap_after_timestamp(
            getattr(row, "up_book_init_receive_ns", None), entry, processed,
        )
        window[f"archive_gap_after_down_book_init"] = _gap_after_timestamp(
            getattr(row, "down_book_init_receive_ns", None), entry, processed,
        )
        window["archive_gap_after_fee_update"] = _gap_after_timestamp(
            getattr(row, "fee_event_ns", None), entry, processed,
        )
        coverage_rows.append(window)
    coverage = pd.DataFrame(coverage_rows, index=frame.index).add_prefix(f"{case}_")
    for column in coverage.columns:
        frame[column] = coverage[column]
    return frame


def _side_statuses(snapshot: dict, market: dict, case: str, gross_usd: float = 5.0) -> dict:
    result = {}
    for side in ("up", "down"):
        token_id = str(market.get(f"{side}_token_id") or "")
        base = dict(snapshot)
        base["archive_entry_hour_available"] = snapshot.get(f"{case}_archive_entry_hour_available")
        base[f"archive_gap_after_{side}_book_init"] = snapshot.get(
            f"{case}_archive_gap_after_{side}_book_init"
        )
        base["archive_gap_after_fee_update"] = snapshot.get(f"{case}_archive_gap_after_fee_update")
        reason = t59._ask_base_reason(base, side, token_id)
        if case == "t60" and reason == "ask_valid_within_source_coverage" and not t59._full_depth_is_known(base, side):
            reason = "unknown_book_initialization_or_archive_gap"
        price = t59._float_or_none(snapshot.get(f"{side}_best_ask"))
        size = t59._float_or_none(snapshot.get(f"{side}_best_ask_size_shares"))
        levels = t59._ask_levels(snapshot.get(f"{side}_ask_levels"))
        fill = None
        sufficient = False
        if reason == "ask_valid_within_source_coverage" and price is not None and size is not None:
            ask_book = {price: size} if case == "t60" else levels
            if case == "t59" and not t59._full_depth_is_known(base, side):
                reason = "unknown_full_ask_depth"
            elif not ask_book:
                reason = "observed_empty_ask_book"
            else:
                fill = economic_replay._walk_asks(
                    ask_book,
                    t59._float_or_none(snapshot.get("fee_rate_bps")),
                    snapshot.get("fee_collection_mode"),
                    gross_order_usd=gross_usd,
                    entry_time_utc=pd.Timestamp(snapshot["entry_time_utc"]),
                    allow_partial=True,
                )
                sufficient = bool(
                    fill.get("depth_sufficient")
                    and float(fill.get("unfilled_gross_usd") or 0.0) <= FILL_TOLERANCE_USD
                )
                if not sufficient:
                    reason = "insufficient_5usd_ask_depth"
        result[side] = {
            "reason": reason,
            "base_reason": t59._ask_base_reason(base, side, token_id),
            "native_token": (
                snapshot.get(f"{side}_quote_source_token_id") is not None
                and str(snapshot.get(f"{side}_quote_source_token_id")) == token_id
                and not bool(snapshot.get(f"{side}_quote_complemented"))
            ),
            "fixed5_full_depth": sufficient,
            "fee_known": bool(snapshot.get("fee_known")),
            "displayed_size_shares": size,
            "best_ask": price,
            "gross_fill_usd": fill.get("gross_usd") if fill else None,
            "net_shares": fill.get("shares") if fill else None,
            "fee_usd": fill.get("fee_usd") if fill else None,
        }
    return result


def _source_coverage_manifest(
    universe: pd.DataFrame,
    snapshots_t60: pd.DataFrame,
    snapshots_t59: pd.DataFrame,
    advertised: set[pd.Timestamp],
    processed: set[pd.Timestamp],
) -> tuple[pd.DataFrame, dict]:
    t60 = _add_snapshot_archive_coverage(snapshots_t60, "t60", advertised, processed)
    t59_frame = _add_snapshot_archive_coverage(snapshots_t59, "t59", advertised, processed)
    source_columns = [
        "condition_id", "entry_time_utc", "archive_source", "fee_collection_mode", "fee_known", "fee_rate_bps",
        "fee_event_ns", "market_event_count", "market_book_snapshot_count", "no_future_event_at_entry",
        "no_future_source_event_at_entry", "up_best_bid", "up_best_ask", "up_best_ask_size_shares",
        "up_ask_age_seconds", "up_quote_source_token_id", "up_quote_complemented",
        "up_ask_order_ambiguous", "up_bbo_ask_comparable", "up_bbo_ask_mismatch", "up_bid_ask_crossed",
        "up_book_init_receive_ns", "down_best_bid", "down_best_ask", "down_best_ask_size_shares",
        "down_ask_age_seconds", "down_quote_source_token_id", "down_quote_complemented",
        "down_ask_order_ambiguous", "down_bbo_ask_comparable", "down_bbo_ask_mismatch", "down_bid_ask_crossed",
        "down_book_init_receive_ns", "t60_archive_entry_hour_available", "t60_archive_freshness_window_available",
        "t60_archive_window_hours_expected", "t60_archive_window_hours_source_available",
        "t60_archive_window_hours_processed", "t60_archive_window_missing_source_hours",
        "t60_archive_window_unprocessed_hours", "t60_archive_gap_after_up_book_init",
        "t60_archive_gap_after_down_book_init", "t60_archive_gap_after_fee_update",
    ]
    t60_frame = t60[[column for column in source_columns if column in t60.columns]].copy()
    t60_frame.rename(columns={
        column: f"t60_{column}"
        for column in t60_frame
        if column != "condition_id" and not column.startswith("t60_")
    }, inplace=True)
    t59_columns = [
        "condition_id", "entry_time_utc", "archive_source", "fee_collection_mode", "fee_known", "fee_rate_bps",
        "fee_event_ns", "market_event_count", "market_book_snapshot_count", "no_future_event_at_entry",
        "no_future_source_event_at_entry", "up_best_bid", "up_best_ask", "up_best_ask_size_shares",
        "up_ask_age_seconds", "up_quote_source_token_id", "up_quote_complemented",
        "up_ask_order_ambiguous", "up_bbo_ask_comparable", "up_bbo_ask_mismatch", "up_bid_ask_crossed",
        "up_book_init_receive_ns", "down_best_bid", "down_best_ask", "down_best_ask_size_shares",
        "down_ask_age_seconds", "down_quote_source_token_id", "down_quote_complemented",
        "down_ask_order_ambiguous", "down_bbo_ask_comparable", "down_bbo_ask_mismatch", "down_bid_ask_crossed",
        "down_book_init_receive_ns", "t59_archive_entry_hour_available", "t59_archive_freshness_window_available",
        "t59_archive_window_hours_expected", "t59_archive_window_hours_source_available",
        "t59_archive_window_hours_processed", "t59_archive_window_missing_source_hours",
        "t59_archive_window_unprocessed_hours", "t59_archive_gap_after_up_book_init",
        "t59_archive_gap_after_down_book_init", "t59_archive_gap_after_fee_update",
    ]
    t59_source = t59_frame[[column for column in t59_columns if column in t59_frame.columns]].copy()
    t59_source.rename(columns={
        column: f"t59_{column}"
        for column in t59_source
        if column != "condition_id" and not column.startswith("t59_")
    }, inplace=True)
    coverage = universe.merge(t60_frame, on="condition_id", how="left", validate="one_to_one")
    coverage = coverage.merge(t59_source, on="condition_id", how="left", validate="one_to_one")
    coverage["t60_source_status"] = "no_native_fixed5_side"
    coverage["t59_source_status"] = "no_native_fixed5_depth_side"
    coverage["native_token_status"] = "unverified"
    coverage["minimum_depth_status"] = "not_sufficient_on_either_side"
    additional_flags = []
    first_exclusions = []
    t60_side_rows = []
    t59_side_rows = []
    t60_by_id = {str(row.condition_id): row._asdict() for row in t60.itertuples(index=False)}
    t59_by_id = {str(row.condition_id): row._asdict() for row in t59_frame.itertuples(index=False)}
    for position, row in enumerate(coverage.itertuples(index=False)):
        market = {
            "market_start_utc": row.market_start_utc,
            "up_token_id": row.up_token_id,
            "down_token_id": row.down_token_id,
        }
        t60_row = t60_by_id[str(row.condition_id)]
        t59_row = t59_by_id[str(row.condition_id)]
        t60_sides = _side_statuses(t60_row, market, "t60")
        t59_sides = _side_statuses(t59_row, market, "t59")
        t60_side_rows.append(t60_sides)
        t59_side_rows.append(t59_sides)
        t60_usable = any(item["fixed5_full_depth"] for item in t60_sides.values())
        t59_usable = any(item["fixed5_full_depth"] for item in t59_sides.values())
        coverage.at[position, "t60_source_status"] = "at_least_one_native_fixed5_BBO" if t60_usable else "no_native_fixed5_side"
        coverage.at[position, "t59_source_status"] = "at_least_one_native_fixed5_depth" if t59_usable else "no_native_fixed5_depth_side"
        coverage.at[position, "native_token_status"] = (
            "both_sides_native" if all(item["native_token"] for item in t59_sides.values())
            else "partial_or_missing_native_mapping"
        )
        coverage.at[position, "minimum_depth_status"] = (
            "both_sides_fixed5_depth" if all(item["fixed5_full_depth"] for item in t59_sides.values())
            else "one_side_fixed5_depth" if t59_usable
            else "not_sufficient_on_either_side"
        )
        flags = []
        if not bool(row.prediction_ready_timestamp_present):
            flags.append("prediction_ready_timestamp_missing; readiness_unknown")
        if not bool(row.token_mapping_present):
            flags.append("gamma_native_token_mapping_missing")
        if t60_usable and not t59_usable:
            flags.append("t60_usable_but_no_t59_fixed5_depth")
        if t59_usable and not t60_usable:
            flags.append("t59_depth_without_usable_t60_decision_BBO")
        if not bool(t60_row.get("t60_archive_entry_hour_available")):
            flags.append("t60_archive_entry_hour_unavailable")
        if not bool(t59_row.get("t59_archive_entry_hour_available")):
            flags.append("t59_archive_entry_hour_unavailable")
        if any(t60_row.get(f"t60_archive_gap_after_{side}_book_init") is True for side in ("up", "down")):
            flags.append("t60_documented_gap_after_book_initialization")
        if any(t59_row.get(f"t59_archive_gap_after_{side}_book_init") is True for side in ("up", "down")):
            flags.append("t59_documented_gap_after_book_initialization")
        if t59_row.get("t59_archive_gap_after_fee_update") is True:
            flags.append("t59_documented_gap_after_fee_update")
        if not bool(t60_row.get("t60_archive_window_missing_source_hours", 0) == 0):
            flags.append("t60_25h_source_gap")
        if not bool(t59_row.get("t59_archive_window_missing_source_hours", 0) == 0):
            flags.append("t59_25h_source_gap")
        if bool(row.official_label_valid):
            label_status = True
        else:
            label_status = False
        if not bool(row.prediction_present):
            exclusion = "missing_saved_candidate_probabilities"
        elif not label_status:
            exclusion = "official_label_invalid_or_unavailable"
        elif not t60_usable:
            exclusion = "no_usable_native_t60_BBO_at_fixed5"
        elif not t59_usable:
            exclusion = "no_usable_native_t59_depth_at_fixed5"
        else:
            exclusion = "none_source_evaluable_under_retrospective_assumption"
        additional_flags.append(";".join(flags))
        first_exclusions.append(exclusion)
    coverage["t60_up_reason"] = [row["up"]["reason"] for row in t60_side_rows]
    coverage["t60_down_reason"] = [row["down"]["reason"] for row in t60_side_rows]
    coverage["t59_up_reason"] = [row["up"]["reason"] for row in t59_side_rows]
    coverage["t59_down_reason"] = [row["down"]["reason"] for row in t59_side_rows]
    coverage["t60_up_fixed5_BBO_fill"] = [row["up"]["fixed5_full_depth"] for row in t60_side_rows]
    coverage["t60_down_fixed5_BBO_fill"] = [row["down"]["fixed5_full_depth"] for row in t60_side_rows]
    coverage["t59_up_fixed5_depth"] = [row["up"]["fixed5_full_depth"] for row in t59_side_rows]
    coverage["t59_down_fixed5_depth"] = [row["down"]["fixed5_full_depth"] for row in t59_side_rows]
    coverage["first_exclusion_reason"] = first_exclusions
    coverage["additional_quality_flags"] = additional_flags
    coverage["source_joint_fixed5_coverage"] = (
        coverage.t60_source_status.eq("at_least_one_native_fixed5_BBO")
        & coverage.t59_source_status.eq("at_least_one_native_fixed5_depth")
    )
    coverage["source_gap_status"] = "no_observed_gap_after_state_initialization"
    entry_missing = (
        ~coverage.t60_archive_entry_hour_available.fillna(False)
        | ~coverage.t59_archive_entry_hour_available.fillna(False)
    )
    documented_gap = (
        coverage[[
            "t60_archive_gap_after_up_book_init", "t60_archive_gap_after_down_book_init",
            "t60_archive_gap_after_fee_update", "t59_archive_gap_after_up_book_init",
            "t59_archive_gap_after_down_book_init", "t59_archive_gap_after_fee_update",
        ]].fillna(False).astype(bool).any(axis=1)
    )
    unknown_init = (
        coverage[[
            "t60_archive_gap_after_up_book_init", "t60_archive_gap_after_down_book_init",
            "t60_archive_gap_after_fee_update", "t59_archive_gap_after_up_book_init",
            "t59_archive_gap_after_down_book_init", "t59_archive_gap_after_fee_update",
        ]].isna().any(axis=1)
    )
    coverage.loc[unknown_init, "source_gap_status"] = "book_or_fee_initialization_unobserved"
    coverage.loc[documented_gap, "source_gap_status"] = "documented_archive_gap_after_initialization"
    coverage.loc[entry_missing, "source_gap_status"] = "local_archive_missing_at_entry"
    coverage.loc[coverage.t60_source_status.eq("no_native_fixed5_side") & coverage.t59_source_status.eq("no_native_fixed5_depth_side"), "source_gap_status"] = "no_reconstructed_native_entry_source"
    coverage["monthly_utc"] = pd.to_datetime(coverage.market_start_utc, utc=True).dt.strftime("%Y-%m")
    monthly = {}
    for month, group in coverage.groupby("monthly_utc", sort=True):
        monthly[str(month)] = {
            "gamma_markets": int(len(group)),
            "official_label_valid": int(group.official_label_valid.sum()),
            "prediction_present": int(group.prediction_present.sum()),
            "official_label_and_prediction": int((group.official_label_valid & group.prediction_present).sum()),
            "t60_any_fixed5_native_side": int(group.t60_source_status.eq("at_least_one_native_fixed5_BBO").sum()),
            "t59_any_fixed5_native_depth": int(group.t59_source_status.eq("at_least_one_native_fixed5_depth").sum()),
            "joint_t60_t59_fixed5_source": int(group.source_joint_fixed5_coverage.sum()),
            "prediction_and_joint_source": int((group.prediction_present & group.source_joint_fixed5_coverage).sum()),
            "label_prediction_and_joint_source": int((group.official_label_valid & group.prediction_present & group.source_joint_fixed5_coverage).sum()),
            "first_exclusion_counts": {
                str(reason): int(count)
                for reason, count in group.first_exclusion_reason.value_counts(dropna=False).items()
                if reason is not None and not pd.isna(reason)
            },
        }
    summary = {
        "gamma_markets": int(len(coverage)),
        "official_label_valid": int(coverage.official_label_valid.sum()),
        "prediction_present": int(coverage.prediction_present.sum()),
        "official_label_and_prediction": int((coverage.official_label_valid & coverage.prediction_present).sum()),
        "prediction_ready_timestamp_present": int(coverage.prediction_ready_timestamp_present.sum()),
        "t60_any_fixed5_native_side": int(coverage.t60_source_status.eq("at_least_one_native_fixed5_BBO").sum()),
        "t59_any_fixed5_native_depth": int(coverage.t59_source_status.eq("at_least_one_native_fixed5_depth").sum()),
        "joint_t60_t59_fixed5_source": int(coverage.source_joint_fixed5_coverage.sum()),
        "prediction_and_joint_source": int((coverage.prediction_present & coverage.source_joint_fixed5_coverage).sum()),
        "label_prediction_and_joint_source": int((coverage.official_label_valid & coverage.prediction_present & coverage.source_joint_fixed5_coverage).sum()),
        "first_exclusion_counts": {
            str(reason): int(count)
            for reason, count in coverage.first_exclusion_reason.value_counts(dropna=False).items()
            if reason is not None and not pd.isna(reason)
        },
        "monthly": monthly,
        "outcomes_or_pnl_read": False,
    }
    return coverage, summary


def _jsonable(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (list, dict, tuple)):
        return json.loads(json.dumps(value, default=str))
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _parity_check(snapshots_t60: pd.DataFrame, snapshots_t59: pd.DataFrame) -> dict:
    ledger_dataset = ds.dataset(
        ROOT / "data/analysis/polymarket/BTC/btc_candidate_polymarket_target_20261023/economic_decision_ledger.parquet",
        format="parquet",
    )
    control_ids = ledger_dataset.to_table(columns=["condition_id"]).to_pandas().condition_id.astype(str).tolist()
    t60_old_dataset = ds.dataset(ROOT / "reports/btc_preopen/entry_snapshots.parquet", format="parquet")
    t60_old = t60_old_dataset.to_table(
        columns=[
            "condition_id", "entry_case", "entry_time_utc", "fee_collection_mode", "fee_rate_bps", "fee_known",
            "book_snapshot_token_count", "has_full_snapshot", "quote_valid", "bbo_at_entry_ask_mismatches",
            "up_best_ask", "up_best_ask_size_shares", "down_best_ask", "down_best_ask_size_shares",
            "up_quote_source_token_id", "down_quote_source_token_id", "up_quote_complemented", "down_quote_complemented",
            "no_future_event_at_entry", "no_future_source_event_at_entry",
        ],
        filter=ds.field("entry_case").eq("prestart_c0_o0") & ds.field("condition_id").isin(control_ids),
    ).to_pandas()
    t59_old_dataset = ds.dataset(
        ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/t59_ask_ladders.parquet",
        format="parquet",
    )
    t59_old = t59_old_dataset.to_table(
        columns=[
            "condition_id", "entry_case", "entry_time_utc", "fee_collection_mode", "fee_rate_bps", "fee_known",
            "book_snapshot_token_count", "has_full_snapshot", "quote_valid", "bbo_at_entry_ask_mismatches",
            "up_best_ask", "up_best_ask_size_shares", "down_best_ask", "down_best_ask_size_shares",
            "up_quote_source_token_id", "down_quote_source_token_id", "up_quote_complemented", "down_quote_complemented",
            "up_ask_levels", "down_ask_levels", "up_ask_order_ambiguous", "down_ask_order_ambiguous",
            "up_bid_ask_crossed", "down_bid_ask_crossed", "no_future_event_at_entry", "no_future_source_event_at_entry",
        ],
        filter=ds.field("entry_case").eq(T59_ENTRY_CASE) & ds.field("condition_id").isin(control_ids),
    ).to_pandas()
    if len(control_ids) != 1143 or len(set(control_ids)) != 1143:
        raise RuntimeError(f"Frozen E0036 ledger identity is not 1,143 unique markets: {len(set(control_ids))}")
    if len(t60_old) != len(control_ids) or len(t59_old) != len(control_ids):
        raise RuntimeError("Saved E0036 snapshots do not cover the frozen control ledger one-to-one")

    source_fields = {
        "t60": (
            "entry_time_utc", "fee_collection_mode", "fee_rate_bps", "fee_known", "book_snapshot_token_count",
            "has_full_snapshot", "quote_valid", "bbo_at_entry_ask_mismatches", "up_best_ask",
            "up_best_ask_size_shares", "down_best_ask", "down_best_ask_size_shares", "up_quote_source_token_id",
            "down_quote_source_token_id", "up_quote_complemented", "down_quote_complemented",
            "no_future_event_at_entry", "no_future_source_event_at_entry",
        ),
        "t59": (
            "entry_time_utc", "fee_collection_mode", "fee_rate_bps", "fee_known", "book_snapshot_token_count",
            "has_full_snapshot", "quote_valid", "bbo_at_entry_ask_mismatches", "up_best_ask",
            "up_best_ask_size_shares", "down_best_ask", "down_best_ask_size_shares", "up_quote_source_token_id",
            "down_quote_source_token_id", "up_quote_complemented", "down_quote_complemented", "up_ask_levels",
            "down_ask_levels", "up_ask_order_ambiguous", "down_ask_order_ambiguous", "up_bid_ask_crossed",
            "down_bid_ask_crossed", "no_future_event_at_entry", "no_future_source_event_at_entry",
        ),
    }
    results = {}
    for case, old in (("t60", t60_old), ("t59", t59_old)):
        current = snapshots_t60 if case == "t60" else snapshots_t59
        current = current.loc[current.condition_id.astype(str).isin(control_ids)].copy()
        if len(current) != len(control_ids):
            raise RuntimeError(f"New {case} replay does not contain all 1,143 E0036 controls")
        old.set_index("condition_id", inplace=True)
        current.set_index("condition_id", inplace=True)
        if not old.index.is_unique or not current.index.is_unique:
            raise RuntimeError(f"{case} parity snapshots contain duplicate condition IDs")
        mismatches = []
        compared = 0
        for condition_id in control_ids:
            left = current.loc[condition_id]
            right = old.loc[condition_id]
            for field in source_fields[case]:
                a, b = _jsonable(left.get(field)), _jsonable(right.get(field))
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    equal = math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-10)
                else:
                    equal = a == b
                compared += 1
                if not equal:
                    mismatches.append({"condition_id": condition_id, "field": field, "new": a, "saved": b})
                    if len(mismatches) >= 20:
                        break
            if len(mismatches) >= 20:
                break
        results[case] = {
            "control_markets": len(control_ids),
            "source_fields_compared": compared,
            "status": "passed" if not mismatches else "failed",
            "mismatches_first_20": mismatches,
        }
    return {
        "control_experiment_id": "E0036",
        "protocol_sha256": _protocol()[1],
        "outcomes_or_pnl_read": False,
        "t60": results["t60"],
        "t59": results["t59"],
        "status": "passed" if all(item["status"] == "passed" for item in results.values()) else "failed",
    }


def _obadiaha_development_coverage(universe: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    march_start = pd.Timestamp("2026-03-01T00:00:00Z")
    march_end = pd.Timestamp("2026-03-20T00:00:00Z")
    markets = universe.loc[
        universe.market_start_utc.ge(march_start) & universe.market_start_utc.lt(march_end),
        ["condition_id", "market_start_utc", "up_token_id", "down_token_id"],
    ].copy()
    token_map = {}
    for row in markets.itertuples(index=False):
        for side in ("up", "down"):
            token = str(getattr(row, f"{side}_token_id") or "")
            if token:
                if token in token_map:
                    raise RuntimeError("Obadiaha token map is not unique within the March Gamma cohort")
                token_map[token] = (str(row.condition_id), side)
    paths = sorted(OBADIAHA_ROOT.joinpath("orderbooks").glob("2026-03-*.parquet"))
    if len(paths) != 19:
        raise RuntimeError(f"Frozen Obadiaha March part count changed: {len(paths)}")
    source = ds.dataset(paths, format="parquet")
    frame = source.to_table(
        columns=["timestamp", "condition_id", "token_id", "best_ask", "ask_levels"],
        filter=ds.field("condition_id").isin(markets.condition_id.tolist()),
    ).to_pandas()
    frame["condition_id"] = frame.condition_id.astype(str).str.lower()
    frame["token_id"] = frame.token_id.astype(str)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    frame["mapped"] = frame.token_id.map(token_map)
    frame = frame.loc[frame.mapped.notna()].copy()
    frame["mapped_condition_id"] = frame.mapped.map(lambda value: value[0])
    frame["side"] = frame.mapped.map(lambda value: value[1])
    if not frame.condition_id.eq(frame.mapped_condition_id).all():
        raise RuntimeError("Obadiaha token and condition IDs disagree with native Gamma mapping")
    frame.sort_values(["token_id", "timestamp"], kind="stable", inplace=True)
    by_token = {
        token: group.reset_index(drop=True)
        for token, group in frame.groupby("token_id", sort=False)
    }
    rows = []
    for market in markets.itertuples(index=False):
        row = {
            "condition_id": str(market.condition_id),
            "market_start_utc": market.market_start_utc,
            "source_class": "10-second sampled proxy; no independent event/receive timestamps",
        }
        selections = {}
        for case, offset in (("t60", 60), ("t59", 59)):
            cutoff = pd.Timestamp(market.market_start_utc) - pd.Timedelta(seconds=offset)
            native = {}
            for side in ("up", "down"):
                token = str(getattr(market, f"{side}_token_id") or "")
                samples = by_token.get(token)
                match = None
                age = None
                if samples is not None:
                    times = pd.to_datetime(samples.timestamp, utc=True).astype("int64").to_numpy()
                    position = int(np.searchsorted(times, cutoff.value, side="right") - 1)
                    if position >= 0:
                        match = samples.iloc[position]
                        age = float((cutoff - match.timestamp).total_seconds())
                fresh = match is not None and 0.0 <= age <= 10.0
                levels = _ask_levels(match.ask_levels) if fresh else []
                best_ask = t59._float_or_none(match.best_ask) if fresh else None
                if fresh and levels and best_ask is not None and not math.isclose(
                    min(price for price, _ in levels), best_ask, rel_tol=0.0, abs_tol=1e-9,
                ):
                    raise RuntimeError("Obadiaha top-10 asks disagree with the saved best ask")
                depth_gross = sum(price * size for price, size in levels)
                native[side] = {
                    "sample_found": match is not None,
                    "sample_within_10s": bool(fresh),
                    "capture_time_utc": match.timestamp if match is not None else pd.NaT,
                    "age_at_cutoff_seconds": age,
                    "best_ask": best_ask,
                    "ask_level_count": len(levels),
                    "top10_gross_ask_depth_usd": depth_gross,
                    "top10_fixed5_gross_depth_sufficient": bool(fresh and depth_gross + FILL_TOLERANCE_USD >= 5.0),
                }
                for name, value in native[side].items():
                    row[f"{case}_{side}_{name}"] = value
            selections[case] = native
            row[f"{case}_both_native_samples_within_10s"] = all(
                native[side]["sample_within_10s"] for side in ("up", "down")
            )
        row["t59_both_sides_top10_fixed5_depth_sufficient"] = all(
            selections["t59"][side]["top10_fixed5_gross_depth_sufficient"] for side in ("up", "down")
        )
        row["joint_t60_t59_proxy_coverage"] = (
            row["t60_both_native_samples_within_10s"]
            and row["t59_both_native_samples_within_10s"]
        )
        if not row["t60_both_native_samples_within_10s"]:
            row["first_exclusion_reason"] = "missing_or_stale_t60_native_proxy_sample"
        elif not row["t59_both_native_samples_within_10s"]:
            row["first_exclusion_reason"] = "missing_or_stale_t59_native_proxy_sample"
        elif not row["t59_both_sides_top10_fixed5_depth_sufficient"]:
            row["first_exclusion_reason"] = "top10_depth_shortfall_or_unknown_beyond_top10"
        else:
            row["first_exclusion_reason"] = None
        rows.append(row)
    coverage = pd.DataFrame(rows)
    monthly = {}
    if not coverage.empty:
        coverage["month_utc"] = pd.to_datetime(coverage.market_start_utc, utc=True).dt.strftime("%Y-%m")
        for month, group in coverage.groupby("month_utc", sort=True):
            monthly[str(month)] = {
                "gamma_markets": int(len(group)),
                "t60_both_native_within_10s": int(group.t60_both_native_samples_within_10s.sum()),
                "t59_both_native_within_10s": int(group.t59_both_native_samples_within_10s.sum()),
                "joint_t60_t59_proxy": int(group.joint_t60_t59_proxy_coverage.sum()),
                "t59_top10_fixed5_depth_both_sides": int(group.t59_both_sides_top10_fixed5_depth_sufficient.sum()),
            }
    summary = {
        "source_class": "separate proxy; capture timestamp only, no event/receive timestamp",
        "march_market_start_window_utc": [march_start.isoformat(), march_end.isoformat()],
        "gamma_markets": int(len(coverage)),
        "source_parts": len(paths),
        "source_rows_mapped_to_gamma_native_tokens": int(len(frame)),
        "t60_both_native_within_10s": int(coverage.t60_both_native_samples_within_10s.sum()) if not coverage.empty else 0,
        "t59_both_native_within_10s": int(coverage.t59_both_native_samples_within_10s.sum()) if not coverage.empty else 0,
        "joint_t60_t59_proxy": int(coverage.joint_t60_t59_proxy_coverage.sum()) if not coverage.empty else 0,
        "t59_top10_fixed5_depth_both_sides": int(coverage.t59_both_sides_top10_fixed5_depth_sufficient.sum()) if not coverage.empty else 0,
        "monthly": monthly,
        "outcomes_or_pnl_read": False,
    }
    coverage.to_parquet(output_dir / "obadiaha_proxy_coverage.parquet", index=False, compression="zstd")
    return coverage, summary


def _kachoio_preopen_control(universe: pd.DataFrame) -> dict:
    source_dir = ROOT / "data/raw/polymarket/kachoio/42d917dc8e3205dde8ac909792af0cce2d715c9f"
    markets = pd.read_parquet(
        source_dir / "btc_markets.parquet",
        columns=["condition_id", "market_start", "token_up", "token_down"],
    )
    ticks = pd.read_parquet(
        source_dir / "btc_ticks.parquet",
        columns=["condition_id", "ts_utc"],
    )
    markets["condition_id"] = markets.condition_id.astype(str).str.lower()
    markets["market_start"] = pd.to_datetime(markets.market_start, utc=True)
    ticks["condition_id"] = ticks.condition_id.astype(str).str.lower()
    ticks["ts_utc"] = pd.to_datetime(ticks.ts_utc, utc=True)
    starts = universe[["condition_id", "market_start_utc"]].rename(
        columns={"market_start_utc": "gamma_market_start_utc"},
    )
    tick_markets = ticks.merge(starts, on="condition_id", how="inner", validate="many_to_one")
    if not tick_markets.empty:
        tick_markets["t60_cutoff_utc"] = tick_markets.gamma_market_start_utc - pd.Timedelta(seconds=60)
        tick_markets["t59_cutoff_utc"] = tick_markets.gamma_market_start_utc - pd.Timedelta(seconds=59)
        t60_rows = tick_markets.ts_utc.le(tick_markets.t60_cutoff_utc)
        t59_rows = tick_markets.ts_utc.le(tick_markets.t59_cutoff_utc)
    else:
        t60_rows = pd.Series(dtype=bool)
        t59_rows = pd.Series(dtype=bool)
    earliest = ticks.groupby("condition_id").ts_utc.min()
    source_starts = markets.set_index("condition_id").market_start
    comparable = source_starts.index.intersection(earliest.index)
    offset = earliest.loc[comparable].sub(source_starts.loc[comparable])
    return {
        "source_class": "1-second within-market BBO; no ask depth",
        "source_market_rows": int(len(markets)),
        "source_tick_rows": int(len(ticks)),
        "source_tick_start_utc": ticks.ts_utc.min().isoformat() if not ticks.empty else None,
        "source_tick_end_utc": ticks.ts_utc.max().isoformat() if not ticks.empty else None,
        "gamma_ids_with_source_ticks": int(len(comparable)),
        "markets_first_tick_at_or_after_start": int(offset.ge(pd.Timedelta(0)).sum()),
        "markets_first_tick_before_start": int(offset.lt(pd.Timedelta(0)).sum()),
        "matched_development_t60_preopen_tick_rows": int(t60_rows.sum()) if len(t60_rows) else 0,
        "matched_development_t59_preopen_tick_rows": int(t59_rows.sum()) if len(t59_rows) else 0,
        "matched_development_markets_with_any_t60_preopen_tick": int(
            tick_markets.loc[t60_rows, "condition_id"].nunique()
        ) if len(t60_rows) else 0,
        "matched_development_markets_with_any_t59_preopen_tick": int(
            tick_markets.loc[t59_rows, "condition_id"].nunique()
        ) if len(t59_rows) else 0,
        "t59_ask_depth_available": False,
        "outcomes_or_pnl_read": False,
    }


def run_coverage() -> None:
    protocol, protocol_hash = _protocol()
    if protocol["analysis_limits"]["holdout_rows_or_outcomes"]:
        raise RuntimeError("E0037 protocol permits holdout access")
    checked_hashes = _verify_frozen_sources(protocol)
    output_dir = OUTPUT_ROOT / f"btc_t0009_resolution_coverage_20261026_{protocol_hash[:12]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    universe, identity = _load_development_universe()
    replay_index, development_parts = _make_development_replay_index(universe, output_dir)
    advertised, processed = _development_hour_sets(protocol, development_parts)
    archive_manifest_hash = protocol["inputs"]["pmxt_archive"]["manifest_sha256"]
    t60, t59_snapshots, replay_manifest = _replay_development_sources(
        replay_index,
        output_dir / "development_event_parts",
        output_dir,
        archive_manifest_hash,
        protocol_hash,
    )
    coverage, coverage_summary = _source_coverage_manifest(
        universe, t60, t59_snapshots, advertised, processed,
    )
    probability_columns = [
        "ensemble_probability_A", "ensemble_probability_B", "ensemble_probability_C", "ensemble_probability_D",
        "nominal_decision_time_utc", "candidate_market_start_utc",
    ]
    coverage.drop(columns=[column for column in probability_columns if column in coverage.columns], inplace=True)
    coverage_path = output_dir / "coverage_manifest.parquet"
    coverage.to_parquet(coverage_path, index=False, compression="zstd")
    parity = _parity_check(t60, t59_snapshots)
    parity_path = output_dir / "e0036_source_parity.json"
    parity_path.write_text(json.dumps(parity, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if parity["status"] != "passed":
        raise RuntimeError("E0036 source parity failed; stop before economics and inspect the mismatch report")
    obadiaha_coverage, obadiaha_summary = _obadiaha_development_coverage(universe, output_dir)
    kachoio_summary = _kachoio_preopen_control(universe)
    source_window_summary = {
        "pmxt_advertised_development_hours": len(advertised),
        "pmxt_processed_development_hours": len(processed),
        "pmxt_unprocessed_advertised_development_hours": len(advertised - processed),
        "pmxt_unprocessed_hours": sorted(hour.isoformat() for hour in advertised - processed),
        "pmxt_local_development_parts": len(development_parts),
        "pmxt_local_development_bytes": int(sum(path.stat().st_size for path in development_parts)),
    }
    summary = {
        "experiment_id": "E0037",
        "status": "source_coverage_complete",
        "protocol_sha256": protocol_hash,
        "runner_sha256": _sha256(Path(__file__)),
        "input_hashes_verified": checked_hashes,
        "identity_audit": identity,
        "source_hours": source_window_summary,
        "pmxt_replay": replay_manifest,
        "pmxt_coverage": coverage_summary,
        "e0036_source_parity": parity,
        "obadiaha_proxy": obadiaha_summary,
        "kachoio_preopen_control": kachoio_summary,
        "coverage_manifest_path": coverage_path.relative_to(ROOT).as_posix(),
        "coverage_manifest_sha256": _sha256(coverage_path),
        "obadiaha_coverage_path": (output_dir / "obadiaha_proxy_coverage.parquet").relative_to(ROOT).as_posix(),
        "outcomes_or_pnl_read": False,
        "holdout_parts_read": 0,
        "prediction_ready_timestamps_invented": False,
        "economics_run": False,
    }
    summary_path = output_dir / "source_coverage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "protocol_sha256": protocol_hash,
        "runner_sha256": summary["runner_sha256"],
        "identity_audit": identity,
        "source_hours": source_window_summary,
        "pmxt_coverage": coverage_summary,
        "e0036_source_parity_status": parity["status"],
        "obadiaha_proxy": obadiaha_summary,
        "kachoio_preopen_control": kachoio_summary,
        "coverage_manifest_path": summary["coverage_manifest_path"],
        "summary_path": summary_path.relative_to(ROOT).as_posix(),
        "outcomes_or_pnl_read": False,
    }, indent=2, ensure_ascii=False, default=str))


def _verify_frozen_sources(protocol: dict) -> dict[str, str]:
    expected = {
        protocol["scope"]["candidate_source"]: protocol["scope"]["candidate_source_sha256"],
        protocol["candidate_identity"]["candidate_manifest_path"]: protocol["candidate_identity"]["candidate_manifest_sha256"],
    }
    for item in protocol["inputs"].values():
        if not isinstance(item, dict):
            continue
        for path_key, hash_key in (
            ("path", "sha256"),
            ("manifest_path", "manifest_sha256"),
            ("readme_path", "readme_sha256"),
            ("markets_path", "markets_sha256"),
            ("ticks_path", "ticks_sha256"),
            ("config_path", "config_sha256"),
            ("helper_path", "helper_sha256"),
            ("economic_replay_path", "economic_replay_sha256"),
            ("t59_replay_path", "t59_replay_sha256"),
            ("portfolio_path", "portfolio_sha256"),
            ("e0036_runner_path", "e0036_runner_sha256"),
        ):
            if path_key in item and hash_key in item:
                expected[item[path_key]] = item[hash_key]

    for source_name in ("obadiaha", "kachoio"):
        item = protocol["inputs"][source_name]
        expected[f"{item['root']}/manifest.json"] = item["manifest_sha256"]
        expected[f"{item['root']}/README.md"] = item["readme_sha256"]

    checked = {}
    for relative, expected_hash in expected.items():
        actual = _sha256(ROOT / relative)
        if actual != expected_hash:
            raise RuntimeError(f"Frozen source hash mismatch: {relative}")
        checked[relative] = actual

    archive = protocol["inputs"]["pmxt_archive"]
    part_paths = sorted(PMXT_PARTS.glob("*.parquet"))
    development_parts = [path for path in part_paths if path.stem[:10] < "2026-07-01"]
    records = []
    byte_count = 0
    source_rows = 0
    for path in development_parts:
        parquet = pq.ParquetFile(path)
        schema_hash = hashlib.sha256(str(parquet.schema_arrow).encode("utf-8")).hexdigest()
        records.append((
            path.name,
            _sha256(path),
            schema_hash,
            path.stat().st_size,
        ))
        byte_count += path.stat().st_size
        source_rows += parquet.metadata.num_rows
    fingerprint = hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        len(development_parts) != archive["development_parts"]
        or byte_count != archive["development_part_bytes"]
        or source_rows != archive["development_source_rows"]
        or fingerprint != archive["development_part_list_content_schema_fingerprint_sha256"]
    ):
        raise RuntimeError("Frozen PMXT development part set changed")
    checked["PMXT development event parts"] = fingerprint
    return checked


def _gamma_identity(condition_id: str) -> dict:
    columns = [
        "condition_id", "market_slug", "market_start_utc", "up_token_id", "down_token_id",
    ]
    dataset = ds.dataset(GAMMA_PATH, format="parquet")
    table = dataset.to_table(
        columns=columns,
        filter=ds.field("condition_id") == condition_id,
    )
    frame = table.to_pandas()
    if len(frame) != 1:
        raise RuntimeError(f"Expected one Gamma identity row for {condition_id}, found {len(frame)}")
    row = frame.iloc[0]
    return {
        "condition_id": str(row.condition_id).lower(),
        "market_slug": str(row.market_slug),
        "market_start_utc": pd.Timestamp(row.market_start_utc),
        "up_token_id": str(row.up_token_id),
        "down_token_id": str(row.down_token_id),
    }


def _pmxt_sample_replay(output_dir: Path) -> dict:
    market = _gamma_identity(SAMPLE_PMXT_ID)
    for field in ("resolved_at_utc", "target_polymarket_up", "p_model_raw", "p_model_platt", "p_candidate_raw", "p_candidate_platt"):
        market[field] = None
    index = pd.DataFrame([market])
    cutoff = market["market_start_utc"] - pd.Timedelta(seconds=59)
    cutoff_hour = cutoff.floor("h")
    subset_dir = output_dir / "pmxt_parts_through_sample_cutoff"
    subset_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        path for path in sorted(PMXT_PARTS.glob("*.parquet"))
        if path.stem <= cutoff_hour.strftime("%Y-%m-%dT%H")
    ]
    if not selected:
        raise RuntimeError("No PMXT development event parts exist before the frozen sample cutoff")
    for source in selected:
        destination = subset_dir / source.name
        if destination.exists():
            if not os.path.samefile(source, destination):
                raise RuntimeError(f"Sample partition path is occupied: {destination}")
        else:
            os.link(source, destination)

    replay_dir = output_dir / "pmxt_replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    old_out_dir, old_workers = t59.OUT_DIR, t59.T59_REPLAY_WORKERS
    t59.OUT_DIR = replay_dir
    t59.T59_REPLAY_WORKERS = 1
    try:
        snapshots_t59, replay_manifest = t59._replay_t59(
            index,
            subset_dir,
            archive_source_manifest_sha256=_sha256(PMXT_ROOT / "archive_manifest.json"),
        )
    finally:
        t59.OUT_DIR = old_out_dir
        t59.T59_REPLAY_WORKERS = old_workers

    snapshots_t60 = pd.read_parquet(replay_dir / "t60_decision_books.parquet")
    # The shared replay helper writes a nominal deadline in this field. Remove it
    # from persisted study snapshots; it is not historical prediction telemetry.
    snapshots_t59.drop(columns=["prediction_available_at_utc"], errors="ignore", inplace=True)
    snapshots_t60.drop(columns=["prediction_available_at_utc"], errors="ignore", inplace=True)
    snapshots_t59.to_parquet(replay_dir / "t59_ask_ladders.parquet", index=False, compression="zstd")
    snapshots_t60.to_parquet(replay_dir / "t60_decision_books.parquet", index=False, compression="zstd")
    replay_manifest["synthetic_nominal_prediction_available_at_column_removed"] = True
    replay_manifest["historical_prediction_readiness"] = "unknown; no row timestamp available"
    (replay_dir / "t59_replay_manifest.json").write_text(
        json.dumps(replay_manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    old_t60 = ds.dataset(ROOT / "reports/btc_preopen/entry_snapshots.parquet", format="parquet").to_table(
        filter=(ds.field("condition_id") == SAMPLE_PMXT_ID)
        & (ds.field("entry_case") == "prestart_c0_o0"),
    ).to_pandas()
    old_t59 = ds.dataset(
        ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/t59_ask_ladders.parquet",
        format="parquet",
    ).to_table(
        filter=(ds.field("condition_id") == SAMPLE_PMXT_ID)
        & (ds.field("entry_case") == "t59_native_asks"),
    ).to_pandas()
    if len(snapshots_t60) != 1 or len(snapshots_t59) != 1 or len(old_t60) != 1 or len(old_t59) != 1:
        raise RuntimeError("PMXT sample replay/saved quote join is not one-to-one")

    def compare(new: pd.Series, old: pd.Series, fields: tuple[str, ...]) -> dict:
        result = {}
        for field in fields:
            left, right = new[field], old[field]
            if field.endswith("_utc") or field == "entry_time_utc":
                equal = pd.Timestamp(left) == pd.Timestamp(right)
            elif isinstance(right, (float, int)) and not isinstance(right, bool):
                equal = left is not None and abs(float(left) - float(right)) <= 1e-10
            else:
                equal = left == right
            if not equal:
                raise RuntimeError(f"PMXT sample replay does not match saved field {field}: {left!r} != {right!r}")
            result[field] = True
        return result

    compare_fields = (
        "entry_time_utc", "up_best_ask", "up_best_ask_size_shares", "down_best_ask",
        "down_best_ask_size_shares", "up_quote_source_token_id", "down_quote_source_token_id",
        "up_quote_complemented", "down_quote_complemented", "no_future_event_at_entry",
        "no_future_source_event_at_entry", "bbo_at_entry_ask_mismatches", "quote_valid",
    )
    t60_checks = compare(snapshots_t60.iloc[0], old_t60.iloc[0], compare_fields)
    t59_checks = compare(snapshots_t59.iloc[0], old_t59.iloc[0], compare_fields)
    if snapshots_t60.iloc[0].entry_time_utc != market["market_start_utc"] - pd.Timedelta(seconds=60):
        raise RuntimeError("PMXT T-60 sample timestamp is not the frozen decision cutoff")
    if snapshots_t59.iloc[0].entry_time_utc != market["market_start_utc"] - pd.Timedelta(seconds=59):
        raise RuntimeError("PMXT T-59 sample timestamp is not the frozen execution cutoff")
    if not bool(snapshots_t60.iloc[0].no_future_event_at_entry) or not bool(snapshots_t59.iloc[0].no_future_event_at_entry):
        raise RuntimeError("PMXT sample contains a future receive-time event")
    if not bool(snapshots_t60.iloc[0].no_future_source_event_at_entry) or not bool(snapshots_t59.iloc[0].no_future_source_event_at_entry):
        raise RuntimeError("PMXT sample contains a future source-time event")
    for side in ("up", "down"):
        if str(snapshots_t59.iloc[0][f"{side}_quote_source_token_id"]) != market[f"{side}_token_id"]:
            raise RuntimeError(f"PMXT {side} quote is not the native Gamma token")

    source_quality = {
        side: t59._ask_base_reason(snapshots_t59.iloc[0].to_dict(), side, market[f"{side}_token_id"])
        for side in ("up", "down")
    }
    return {
        "condition_id": SAMPLE_PMXT_ID,
        "market_slug": market["market_slug"],
        "market_start_utc": market["market_start_utc"].isoformat(),
        "t60_entry_time_utc": snapshots_t60.iloc[0].entry_time_utc.isoformat(),
        "t59_entry_time_utc": snapshots_t59.iloc[0].entry_time_utc.isoformat(),
        "native_token_mapping": {
            side: str(snapshots_t59.iloc[0][f"{side}_quote_source_token_id"]) == market[f"{side}_token_id"]
            for side in ("up", "down")
        },
        "no_future_receive_or_source_events": True,
        "saved_quote_parity": {"t60": t60_checks, "t59": t59_checks},
        "saved_quality": {
            "t60_quote_valid": bool(old_t60.iloc[0].quote_valid),
            "t60_bbo_ask_mismatches": int(old_t60.iloc[0].bbo_at_entry_ask_mismatches),
            "t59_quote_valid": bool(old_t59.iloc[0].quote_valid),
            "t59_bbo_ask_mismatches": int(old_t59.iloc[0].bbo_at_entry_ask_mismatches),
        },
        "t59_side_source_quality_reasons": source_quality,
        "source_result": "Replay reproduces the saved source row; this sample is rejected by its saved BBO/source-quality checks and is not a trade.",
        "replay_parts": len(selected),
        "replay_event_rows_applied": int(replay_manifest["event_rows_applied"]),
        "prediction_readiness": "unknown; no historical prediction_available_at_utc is retained",
    }


def _ask_levels(value) -> list[tuple[float, float]]:
    if isinstance(value, str):
        value = json.loads(value)
    levels = []
    for row in value or []:
        if isinstance(row, dict):
            price, size = row.get("price"), row.get("size")
        else:
            try:
                price, size = row
            except (TypeError, ValueError):
                continue
        try:
            price, size = float(price), float(size)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and math.isfinite(size) and 0 < price < 1 and size > 0:
            levels.append((price, size))
    return sorted(levels, key=lambda item: item[0])


def _obadiaha_sample() -> dict:
    market = _gamma_identity(SAMPLE_OBADIAHA_ID)
    source_path = OBADIAHA_ROOT / "orderbooks/2026-03-01.parquet"
    frame = pd.read_parquet(
        source_path,
        columns=["timestamp", "condition_id", "token_id", "best_ask", "ask_levels"],
    )
    frame = frame.loc[frame.condition_id.astype(str).str.lower().eq(SAMPLE_OBADIAHA_ID)].copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    output = {}
    for name, offset in (("t60", 60), ("t59", 59)):
        cutoff = market["market_start_utc"] - pd.Timedelta(seconds=offset)
        cutoff_rows = []
        for side in ("up", "down"):
            token_id = market[f"{side}_token_id"]
            eligible = frame.loc[
                frame.token_id.astype(str).eq(token_id)
                & frame.timestamp.le(cutoff)
            ].sort_values("timestamp")
            if eligible.empty:
                raise RuntimeError(f"Obadiaha sample is missing its {side} native book before {name}")
            row = eligible.iloc[-1]
            age_seconds = float((cutoff - row.timestamp).total_seconds())
            if not 0 <= age_seconds <= 10:
                raise RuntimeError(f"Obadiaha sample {side} book is outside the frozen 10-second proxy window")
            levels = _ask_levels(row.ask_levels)
            if not levels or not math.isclose(min(price for price, _ in levels), float(row.best_ask), abs_tol=1e-9):
                raise RuntimeError(f"Obadiaha sample {side} top-10 ask levels do not match the reported best ask")
            cutoff_rows.append({
                "side": side,
                "token_id": token_id,
                "capture_time_utc": row.timestamp.isoformat(),
                "age_at_cutoff_seconds": age_seconds,
                "reported_best_ask": float(row.best_ask),
                "minimum_ask_level_price": min(price for price, _ in levels),
                "ask_level_count": len(levels),
                "ask_levels_ascending": levels,
                "top10_notional_usd": sum(price * size for price, size in levels),
            })
        output[name] = {"cutoff_utc": cutoff.isoformat(), "native_books": cutoff_rows}
    return {
        "condition_id": SAMPLE_OBADIAHA_ID,
        "market_slug": market["market_slug"],
        "market_start_utc": market["market_start_utc"].isoformat(),
        "sample_class": "10-second sampled proxy; no separate event/receive timestamp",
        "snapshots": output,
        "prediction_readiness": "unknown; source smoke test does not read predictions",
        "outcomes_or_pnl_read": False,
    }


def run_smoke() -> None:
    protocol, protocol_hash = _protocol()
    if protocol["analysis_limits"]["holdout_rows_or_outcomes"]:
        raise RuntimeError("E0037 protocol permits holdout access")
    checked_hashes = _verify_frozen_sources(protocol)
    output_dir = OUTPUT_ROOT / f"btc_t0009_resolution_coverage_20261026_{protocol_hash[:12]}" / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    pmxt_result = _pmxt_sample_replay(output_dir)
    obadiaha_result = _obadiaha_sample()
    summary = {
        "experiment_id": "E0037",
        "status": "source_only_smoke_passed",
        "protocol_sha256": protocol_hash,
        "runner_sha256": _sha256(Path(__file__)),
        "input_hashes_verified": checked_hashes,
        "samples": {"pmxt_exact": pmxt_result, "obadiaha_proxy": obadiaha_result},
        "holdout_parts_read": 0,
        "outcomes_or_pnl_read": False,
        "prediction_ready_timestamps_invented": False,
    }
    path = output_dir / "source_smoke_manifest.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "protocol_sha256": protocol_hash,
        "runner_sha256": summary["runner_sha256"],
        "pmxt": pmxt_result,
        "obadiaha": {
            "condition_id": obadiaha_result["condition_id"],
            "market_start_utc": obadiaha_result["market_start_utc"],
            "sample_class": obadiaha_result["sample_class"],
            "t60_age_seconds": [row["age_at_cutoff_seconds"] for row in obadiaha_result["snapshots"]["t60"]["native_books"]],
            "t59_age_seconds": [row["age_at_cutoff_seconds"] for row in obadiaha_result["snapshots"]["t59"]["native_books"]],
            "native_token_ids": [row["token_id"] for row in obadiaha_result["snapshots"]["t60"]["native_books"]],
            "asks": [row["reported_best_ask"] for row in obadiaha_result["snapshots"]["t60"]["native_books"]],
        },
        "output_manifest": path.relative_to(ROOT).as_posix(),
    }, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    if RUN_MODE == "smoke":
        run_smoke()
    elif RUN_MODE == "coverage":
        run_coverage()
    else:
        raise RuntimeError(f"Unsupported RUN_MODE: {RUN_MODE}")
