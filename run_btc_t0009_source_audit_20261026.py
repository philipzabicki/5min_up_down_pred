"""Source-only continuation audit for the frozen E0037 development cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from utils.btc_t0009_source_audit import (
    replay_native_book,
    source_quarantine_reasons,
    source_value_parity,
    validation_coverage,
)


ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / "data/analysis/polymarket/BTC/btc_t0009_resolution_coverage_20261026_38a6221f0d90"
T60_CACHE = CACHE_ROOT / "pmxt_replay/t60_decision_books.parquet"
T59_CACHE = CACHE_ROOT / "pmxt_replay/t59_ask_ladders.parquet"
COVERAGE_PATH = CACHE_ROOT / "coverage_manifest.parquet"
T60_OLD_PATH = ROOT / "reports/btc_preopen/entry_snapshots.parquet"
T59_OLD_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/t59_ask_ladders.parquet"
E0036_LEDGER_PATH = ROOT / "data/analysis/polymarket/BTC/btc_candidate_polymarket_target_20261023/economic_decision_ledger.parquet"
EVENT_PARTS = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/pmxt_extended/event_parts"
OUTPUT_ROOT = ROOT / "data/analysis/polymarket/BTC/btc_t0009_source_quality_continuation_20261026"
CONTROL_ENTRY_CASE = "prestart_c0_o0"
T59_ENTRY_CASE = "t59_native_asks"
T60_CACHE_CASE = "t60_decision_book"
T59_CACHE_CASE = "t59_native_asks"
EXPECTED_OLD_CONTROL_COUNT = 1143
DEVELOPMENT_CUTOFF = pd.Timestamp("2026-07-01T00:00:00Z")

T60_PARITY_FIELDS = (
    "entry_time_utc", "fee_collection_mode", "fee_rate_bps", "fee_known",
    "book_snapshot_token_count", "has_full_snapshot", "quote_valid",
    "up_best_ask", "up_best_ask_size_shares", "down_best_ask",
    "down_best_ask_size_shares", "up_quote_source_token_id",
    "down_quote_source_token_id", "up_quote_complemented", "down_quote_complemented",
    "no_future_event_at_entry", "no_future_source_event_at_entry",
)
T59_PARITY_FIELDS = (
    "entry_time_utc", "fee_collection_mode", "fee_rate_bps", "fee_known",
    "book_snapshot_token_count", "has_full_snapshot", "quote_valid",
    "up_best_ask", "up_best_ask_size_shares", "down_best_ask",
    "down_best_ask_size_shares", "up_quote_source_token_id",
    "down_quote_source_token_id", "up_quote_complemented", "down_quote_complemented",
    "up_ask_levels", "down_ask_levels", "up_ask_order_ambiguous",
    "down_ask_order_ambiguous", "up_bid_ask_crossed", "down_bid_ask_crossed",
    "no_future_event_at_entry", "no_future_source_event_at_entry",
)
RAW_TRACE_COLUMNS = [
    "timestamp_received", "timestamp", "market", "event_type", "asset_id",
    "bids", "asks", "price", "size", "side", "best_bid", "best_ask",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_control(path: Path, case: str, control_ids: list[str], columns: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    frame = dataset.to_table(
        columns=columns,
        filter=(ds.field("entry_case") == case) & ds.field("condition_id").isin(control_ids),
    ).to_pandas()
    frame["condition_id"] = frame.condition_id.astype(str).str.lower()
    if len(frame) != len(control_ids) or frame.condition_id.duplicated().any():
        raise RuntimeError(f"{path.name} does not cover the E0036 ID set one-to-one")
    return frame.set_index("condition_id").loc[control_ids]


def _read_trace_rows(condition_id: str, token_id: str, init_receive_ns: int, cutoff: pd.Timestamp) -> list[dict]:
    init_time = pd.Timestamp(init_receive_ns, unit="ns", tz="UTC").round("ms")
    entry_hour = cutoff.floor("h")
    rows = []
    for hour in pd.date_range(init_time.floor("h"), entry_hour, freq="h"):
        path = EVENT_PARTS / f"{hour.strftime('%Y-%m-%dT%H')}.parquet"
        if not path.is_file():
            continue
        parquet = pq.ParquetFile(path)
        row_group_offset = 0
        for row_group in range(parquet.num_row_groups):
            frame = parquet.read_row_group(row_group, columns=RAW_TRACE_COLUMNS).to_pandas()
            this_group_offset = row_group_offset
            row_group_offset += parquet.metadata.row_group(row_group).num_rows
            market_ids = frame.market.map(lambda value: value.decode() if isinstance(value, bytes) else str(value))
            mask = (
                market_ids.str.lower().eq(condition_id.lower())
                & frame.asset_id.astype(str).eq(token_id)
                & frame.timestamp_received.ge(init_time)
                & frame.timestamp_received.le(cutoff)
                & frame.timestamp.le(cutoff)
            )
            if not mask.any():
                continue
            positions = np.flatnonzero(mask.to_numpy())
            selected = frame.loc[mask]
            for position, (_, row) in zip(positions, selected.iterrows()):
                rows.append({
                    "file": path.name,
                    "row_group": row_group,
                    "row_in_group": int(position),
                    "timestamp_received_ns": int(row.timestamp_received.value),
                    "timestamp_ns": int(row.timestamp.value),
                    "asset_id": str(row.asset_id),
                    "event_type": str(row.event_type),
                    "bids": row["bids"],
                    "asks": row["asks"],
                    "price": row["price"],
                    "size": row["size"],
                    "side": row["side"],
                    "best_bid": row["best_bid"],
                    "best_ask": row["best_ask"],
                    "row_order": this_group_offset + int(position),
                })
    rows.sort(key=lambda row: (row["timestamp_received_ns"], row["row_order"]))
    return rows


def _trace_case(name: str, coverage_row: pd.Series, side: str, cutoff_name: str) -> dict:
    snapshot_path = T60_CACHE if cutoff_name == "t60" else T59_CACHE
    columns = [
        "condition_id", "entry_time_utc", f"{side}_book_init_receive_ns",
        f"{side}_best_ask", f"{side}_best_ask_size_shares", f"{side}_reported_best_ask",
        f"{side}_bbo_ask_comparable", f"{side}_bbo_ask_mismatch", f"{side}_ask_order_ambiguous",
    ]
    saved = pd.read_parquet(snapshot_path, columns=columns)
    condition_id = str(coverage_row.condition_id).lower()
    saved = saved.loc[saved.condition_id.astype(str).str.lower().eq(condition_id)]
    if len(saved) != 1:
        raise RuntimeError(f"Expected one cached {cutoff_name} row for {condition_id}")
    snapshot = saved.iloc[0]
    token_id = str(coverage_row[f"{side}_token_id"])
    cutoff = pd.Timestamp(snapshot.entry_time_utc)
    init_raw = snapshot[f"{side}_book_init_receive_ns"]
    init_ns = int(pd.Timestamp(int(init_raw), unit="ns", tz="UTC").round("ms").value)
    events = _read_trace_rows(condition_id, token_id, init_ns, cutoff)
    replay = replay_native_book(events, token_id, int(cutoff.value))
    reference_summary = {
        key: replay[key] for key in (
            "initialized", "best_bid", "best_ask", "best_ask_size",
            "reported_best_ask", "ambiguous_ask_prices", "events_seen", "events_applied",
        )
    }
    latest_book = next((row for row in reversed(events) if row["event_type"] == "book"), None)
    last_best_ask_update = None
    if latest_book is not None:
        ask = replay["best_ask"]
        ask_updates = [
            row for row in events
            if row["event_type"] == "price_change"
            and str(row["side"]).upper() == "SELL"
            and _number(row["price"]) == ask
        ]
        if ask_updates:
            last_best_ask_update = ask_updates[-1]
    later_best_ask_level_updates = 0
    if last_best_ask_update is not None:
        later_best_ask_level_updates = sum(
            row["row_order"] > last_best_ask_update["row_order"]
            and row["event_type"] == "price_change"
            and str(row["side"]).upper() == "SELL"
            and _number(row["price"]) == replay["best_ask"]
            for row in events
        )
    report_events = [row for row in events if _number(row["best_ask"]) is not None]
    latest_report = max(report_events, key=lambda row: (row["timestamp_ns"], row["row_order"])) if report_events else None
    latest_report_group = [
        row for row in events
        if latest_report is not None and row["timestamp_received_ns"] == latest_report["timestamp_received_ns"]
    ]
    ask_ambiguous_rows = []
    grouped = {}
    for row in events:
        if row["event_type"] != "price_change" or str(row["side"]).upper() != "SELL":
            continue
        key = (row["timestamp_received_ns"], row["timestamp_ns"], _number(row["price"]))
        grouped.setdefault(key, []).append(row)
    for group in grouped.values():
        if len({_number(row["size"]) for row in group}) > 1:
            ask_ambiguous_rows.extend(group)
    return {
        "name": name,
        "condition_id": condition_id,
        "market_start_utc": pd.Timestamp(coverage_row.market_start_utc).isoformat(),
        "side": side,
        "token_id": token_id,
        "cutoff": cutoff.isoformat(),
        "initialization_receive_utc": pd.Timestamp(init_ns, unit="ns", tz="UTC").isoformat(),
        "source_rows_from_initialization": len(events),
        "last_full_book": _trace_row(latest_book),
        "last_best_ask_level_update": _trace_row(last_best_ask_update),
        "later_updates_to_entry_best_ask_level": later_best_ask_level_updates,
        "latest_reported_bbo_event": _trace_row(latest_report),
        "latest_report_receive_group": [_trace_row(row) for row in latest_report_group],
        "reference_replay": reference_summary,
        "cached_reconstruction": {
            "best_ask": _number(snapshot[f"{side}_best_ask"]),
            "best_ask_size": _number(snapshot[f"{side}_best_ask_size_shares"]),
            "reported_best_ask": _number(snapshot[f"{side}_reported_best_ask"]),
            "bbo_ask_comparable": bool(snapshot[f"{side}_bbo_ask_comparable"]),
            "bbo_ask_mismatch": bool(snapshot[f"{side}_bbo_ask_mismatch"]),
            "ask_order_ambiguous": bool(snapshot[f"{side}_ask_order_ambiguous"]),
        },
        "ambiguous_ask_update_rows": [_trace_row(row) for row in ask_ambiguous_rows[:8]],
        "entry_bbo_disagrees_with_reference_book": (
            replay["best_ask"] is not None
            and replay["reported_best_ask"] is not None
            and not np.isclose(replay["best_ask"], replay["reported_best_ask"], rtol=0.0, atol=1e-9)
        ),
    }


def _trace_row(row):
    if row is None:
        return None
    return {
        "file": row["file"],
        "row_group": row["row_group"],
        "row_in_group": row["row_in_group"],
        "source_utc": pd.Timestamp(row["timestamp_ns"], unit="ns", tz="UTC").isoformat(),
        "receive_utc": pd.Timestamp(row["timestamp_received_ns"], unit="ns", tz="UTC").isoformat(),
        "event_type": row["event_type"],
        "side": row["side"],
        "price": _number(row["price"]),
        "size": _number(row["size"]),
        "best_bid": _number(row["best_bid"]),
        "best_ask": _number(row["best_ask"]),
    }


def _number(value):
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def build_source_audit() -> dict:
    coverage = pd.read_parquet(COVERAGE_PATH)
    base = coverage.loc[
        coverage.source_joint_fixed5_coverage
        & coverage.official_label_valid
        & coverage.prediction_present
    ].copy()
    base["condition_id"] = base.condition_id.astype(str).str.lower()
    base.sort_values(["market_start_utc", "condition_id"], kind="stable", inplace=True)
    reasons = [source_quarantine_reasons(row) for row in base.to_dict(orient="records")]
    base["source_quality_clean"] = [not item for item in reasons]
    base["source_quarantine_reasons"] = [json.dumps(item, ensure_ascii=False) for item in reasons]

    output_columns = [
        "condition_id", "market_start_utc", "monthly_utc", "up_token_id", "down_token_id",
        "t60_up_reason", "t60_down_reason", "t59_up_reason", "t59_down_reason",
        "t60_up_bbo_ask_comparable", "t60_down_bbo_ask_comparable",
        "t59_up_bbo_ask_comparable", "t59_down_bbo_ask_comparable",
        "t60_up_bbo_ask_mismatch", "t60_down_bbo_ask_mismatch",
        "t59_up_bbo_ask_mismatch", "t59_down_bbo_ask_mismatch",
        "source_quality_clean", "source_quarantine_reasons",
    ]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    quality_path = OUTPUT_ROOT / "source_quality_manifest.parquet"
    base.loc[:, output_columns].to_parquet(quality_path, index=False)

    ledger_dataset = ds.dataset(E0036_LEDGER_PATH, format="parquet")
    control_ids = ledger_dataset.to_table(columns=["condition_id"]).to_pandas().condition_id.astype(str).str.lower().tolist()
    if len(control_ids) != EXPECTED_OLD_CONTROL_COUNT or len(set(control_ids)) != EXPECTED_OLD_CONTROL_COUNT:
        raise RuntimeError("E0036 source control is not 1,143 unique condition IDs")
    old_t60_columns = ["condition_id", "entry_case", *T60_PARITY_FIELDS,
                       "bbo_at_entry_ask_checks", "bbo_at_entry_ask_mismatches"]
    old_t59_columns = ["condition_id", "entry_case", *T59_PARITY_FIELDS,
                       "bbo_at_entry_ask_checks", "bbo_at_entry_ask_mismatches"]
    old_t60 = _read_control(T60_OLD_PATH, CONTROL_ENTRY_CASE, control_ids, old_t60_columns)
    new_t60 = _read_control(T60_CACHE, T60_CACHE_CASE, control_ids,
                            ["condition_id", *T60_PARITY_FIELDS, "bbo_at_entry_ask_checks", "bbo_at_entry_ask_mismatches"])
    old_t59 = _read_control(T59_OLD_PATH, T59_ENTRY_CASE, control_ids, old_t59_columns)
    new_t59 = _read_control(T59_CACHE, T59_CACHE_CASE, control_ids,
                            ["condition_id", *T59_PARITY_FIELDS, "bbo_at_entry_ask_checks", "bbo_at_entry_ask_mismatches"])

    parity = {}
    checker_coverage = {}
    for name, old, new, fields in (
        ("t60", old_t60, new_t60, T60_PARITY_FIELDS),
        ("t59", old_t59, new_t59, T59_PARITY_FIELDS),
    ):
        mismatches = []
        compared = 0
        for condition_id in control_ids:
            result = source_value_parity(old.loc[condition_id], new.loc[condition_id], fields)
            compared += result["fields_compared"]
            if result["mismatches"]:
                mismatches.extend({"condition_id": condition_id, **item} for item in result["mismatches"])
        old_checks = old.bbo_at_entry_ask_checks.fillna(0).astype(int)
        new_checks = new.bbo_at_entry_ask_checks.fillna(0).astype(int)
        old_mismatch = old.bbo_at_entry_ask_mismatches.fillna(0).astype(int)
        new_mismatch = new.bbo_at_entry_ask_mismatches.fillna(0).astype(int)
        checker_coverage[name] = {
            "old_check_count_distribution": {str(k): int(v) for k, v in old_checks.value_counts().sort_index().items()},
            "new_check_count_distribution": {str(k): int(v) for k, v in new_checks.value_counts().sort_index().items()},
            "old_mismatch_count_distribution": {str(k): int(v) for k, v in old_mismatch.value_counts().sort_index().items()},
            "new_mismatch_count_distribution": {str(k): int(v) for k, v in new_mismatch.value_counts().sort_index().items()},
            "old_status_counts": _coverage_status_counts(old_checks, old_mismatch),
            "new_status_counts": _coverage_status_counts(new_checks, new_mismatch),
        }
        parity[name] = {
            "markets": len(control_ids),
            "source_fields_compared": compared,
            "status": "PASS" if not mismatches else "FAIL",
            "mismatch_count": len(mismatches),
            "first_mismatches": mismatches[:10],
        }

    control_id_set = set(control_ids)
    clean_control_count = int(base.loc[base.source_quality_clean, "condition_id"].isin(control_id_set).sum())
    t59_ambiguous = base.t59_up_reason.eq("ambiguous_equal_source_timestamp_order") | base.t59_down_reason.eq("ambiguous_equal_source_timestamp_order")
    t59_bbo_mismatch = base.t59_up_reason.eq("native_ask_bbo_reconciliation_mismatch") | base.t59_down_reason.eq("native_ask_bbo_reconciliation_mismatch")
    bbo_unchecked = pd.Series(False, index=base.index)
    for cutoff in ("t60", "t59"):
        for side in ("up", "down"):
            bbo_unchecked |= ~base[f"{cutoff}_{side}_bbo_ask_comparable"].fillna(False).astype(bool)
    source_invariants = {}
    for cutoff in ("t60", "t59"):
        source_invariants[cutoff] = {
            "valid_side_reason_up": int(base[f"{cutoff}_up_reason"].eq("ask_valid_within_source_coverage").sum()),
            "valid_side_reason_down": int(base[f"{cutoff}_down_reason"].eq("ask_valid_within_source_coverage").sum()),
            "no_future_receive_event": int(base[f"{cutoff}_no_future_event_at_entry"].fillna(False).astype(bool).sum()),
            "no_future_source_event": int(base[f"{cutoff}_no_future_source_event_at_entry"].fillna(False).astype(bool).sum()),
            "known_fee": int(base[f"{cutoff}_fee_known"].fillna(False).astype(bool).sum()),
            "entry_hour_available": int(base[f"{cutoff}_archive_entry_hour_available"].fillna(False).astype(bool).sum()),
            "freshness_window_available": int(base[f"{cutoff}_archive_freshness_window_available"].fillna(False).astype(bool).sum()),
            "bbo_ask_comparable_up": int(base[f"{cutoff}_up_bbo_ask_comparable"].fillna(False).astype(bool).sum()),
            "bbo_ask_comparable_down": int(base[f"{cutoff}_down_bbo_ask_comparable"].fillna(False).astype(bool).sum()),
            "bbo_ask_mismatch_up": int(base[f"{cutoff}_up_bbo_ask_mismatch"].fillna(False).astype(bool).sum()),
            "bbo_ask_mismatch_down": int(base[f"{cutoff}_down_bbo_ask_mismatch"].fillna(False).astype(bool).sum()),
            "ask_ambiguous_up": int(base[f"{cutoff}_up_ask_order_ambiguous"].fillna(True).astype(bool).sum()),
            "ask_ambiguous_down": int(base[f"{cutoff}_down_ask_order_ambiguous"].fillna(True).astype(bool).sum()),
            "crossed_up": int(base[f"{cutoff}_up_bid_ask_crossed"].fillna(True).astype(bool).sum()),
            "crossed_down": int(base[f"{cutoff}_down_bid_ask_crossed"].fillna(True).astype(bool).sum()),
            "post_initialization_or_fee_gaps": int(base[[
                f"{cutoff}_archive_gap_after_up_book_init",
                f"{cutoff}_archive_gap_after_down_book_init",
                f"{cutoff}_archive_gap_after_fee_update",
            ]].fillna(True).astype(bool).any(axis=1).sum()),
            "nonzero_unprocessed_window_hours": int(base[f"{cutoff}_archive_window_unprocessed_hours"].fillna(0).gt(0).sum()),
        }

    trace_target = coverage.loc[coverage.condition_id.astype(str).str.lower().eq(
        "0xe4d40fb41d0acedbcae8592f00ec3ac0f6e662f89260c11a65443385996537a3"
    )].iloc[0]
    trace_mismatch = base.loc[t59_bbo_mismatch].sort_values(["market_start_utc", "condition_id"], kind="stable").iloc[0]
    trace_ambiguous = base.loc[t59_ambiguous].sort_values(["market_start_utc", "condition_id"], kind="stable").iloc[0]
    traces = [
        _trace_case("instructed_t60_bbo_conflict", trace_target, "up", "t60"),
        _trace_case(
            "first_t59_bbo_mismatch_by_market_time", trace_mismatch,
            "up" if trace_mismatch.t59_up_reason == "native_ask_bbo_reconciliation_mismatch" else "down", "t59",
        ),
        _trace_case(
            "first_t59_equal_time_ambiguity_by_market_time", trace_ambiguous,
            "up" if trace_ambiguous.t59_up_reason == "ambiguous_equal_source_timestamp_order" else "down", "t59",
        ),
    ]
    return {
        "status": "source_audit_complete_economics_not_run",
        "source_inputs_only": True,
        "outcomes_or_pnl_read": False,
        "base_source_covered_labeled_predicted_markets": len(base),
        "source_quality_clean_markets": int(base.source_quality_clean.sum()),
        "quarantined_markets": int((~base.source_quality_clean).sum()),
        "quarantine_reason_counts": {
            "t59_equal_source_timestamp_ask_ambiguity_markets": int(t59_ambiguous.sum()),
            "t59_ask_bbo_mismatch_markets": int(t59_bbo_mismatch.sum()),
            "any_ask_bbo_not_comparable_markets": int(bbo_unchecked.sum()),
        },
        "e0036_common_source_quality_markets": clean_control_count,
        "e0036_controls_absent_from_base_source_coverage": len(control_id_set - set(base.condition_id)),
        "prediction_ready_timestamp_present": int(base.get("prediction_ready_timestamp_present", pd.Series(False, index=base.index)).fillna(False).sum()),
        "archive_window_unprocessed_hours_nonzero_markets": int((base.t60_archive_window_unprocessed_hours.gt(0) | base.t59_archive_window_unprocessed_hours.gt(0)).sum()),
        "archive_post_initialization_gap_markets": int((
            base[[
                "t60_archive_gap_after_up_book_init", "t60_archive_gap_after_down_book_init",
                "t60_archive_gap_after_fee_update", "t59_archive_gap_after_up_book_init",
                "t59_archive_gap_after_down_book_init", "t59_archive_gap_after_fee_update",
            ]].fillna(True).astype(bool).any(axis=1)
        ).sum()),
        "source_value_parity": parity,
        "checker_coverage": checker_coverage,
        "source_invariants": source_invariants,
        "e0037_frozen_gate_status": "FAILED",
        "source_quality_manifest_path": quality_path.relative_to(ROOT).as_posix(),
        "source_quality_manifest_sha256": sha256(quality_path),
        "instructed_market_source_coverage": {
            "condition_id": str(trace_target.condition_id),
            "in_5_684_base_intersection": bool(trace_target.condition_id in set(base.condition_id)),
            "source_joint_fixed5_coverage": bool(trace_target.source_joint_fixed5_coverage),
            "first_exclusion_reason": str(trace_target.first_exclusion_reason),
            "additional_quality_flags": str(trace_target.additional_quality_flags),
        },
        "coverage_funnel": {
            "manifest_rows": int(len(coverage)),
            "gamma_market_rows": int(coverage.condition_id.notna().sum()),
            "unmapped_candidate_calendar_rows": int(coverage.condition_id.isna().sum()),
            "official_label_valid_rows": int(coverage.official_label_valid.fillna(False).sum()),
            "prediction_present_rows": int(coverage.prediction_present.fillna(False).sum()),
            "label_prediction_joint_source_rows": int((
                coverage.official_label_valid.fillna(False)
                & coverage.prediction_present.fillna(False)
                & coverage.source_joint_fixed5_coverage.fillna(False)
            ).sum()),
            "first_exclusion_counts": {
                str(key): int(value)
                for key, value in coverage.first_exclusion_reason.value_counts(dropna=False).items()
            },
        },
        "source_traces": traces,
        "monthly_source_quality_clean_counts": {
            str(month): int(count)
            for month, count in base.loc[base.source_quality_clean].groupby("monthly_utc").size().items()
        },
        "scope_note": "Source-clean is conditional on the 5,684-row label/prediction/source intersection. Missing Gamma mappings, source gaps, and prediction-readiness unknowns are not strategy no-trades.",
    }


def _coverage_status_counts(checks: pd.Series, mismatches: pd.Series) -> dict[str, int]:
    statuses = [validation_coverage(check, mismatch, expected_checks=2)["status"]
                for check, mismatch in zip(checks.tolist(), mismatches.tolist())]
    return {status: statuses.count(status) for status in sorted(set(statuses))}


def main() -> None:
    result = build_source_audit()
    result["input_sha256"] = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in (COVERAGE_PATH, T60_CACHE, T59_CACHE, T60_OLD_PATH, T59_OLD_PATH, E0036_LEDGER_PATH)
    }
    result_path = OUTPUT_ROOT / "source_audit.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        key: result[key] for key in (
            "base_source_covered_labeled_predicted_markets", "source_quality_clean_markets",
            "quarantined_markets", "quarantine_reason_counts", "e0036_common_source_quality_markets",
            "e0036_controls_absent_from_base_source_coverage", "source_value_parity", "checker_coverage",
        )
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
