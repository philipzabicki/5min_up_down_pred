"""Regenerate PMXT availability diagnostics from saved outputs without replay."""
from __future__ import annotations

import hashlib
import json
import os
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as parquet

import run_btc_market_microstructure_development_20261019 as microstructure
from run_btc_market_microstructure_development_20261019 import ECONOMIC_VARIANTS
from run_btc_preopen_t59_reassessment import _fixed5_skip_reason


ROOT = Path(__file__).resolve().parent
BTC_DATA = ROOT / "data/analysis/polymarket/BTC"
MICRO_DIR = BTC_DATA / "btc_market_microstructure_development_20261019"
MICRO_REPORT = ROOT / "reports/btc_preopen/btc_market_microstructure_development_20261019.md"
MICRO_MANIFEST = ROOT / "reports/btc_preopen/btc_market_microstructure_development_20261019_manifest.json"
SKIP_REASON_PATHS = [
    BTC_DATA / "preopen_v1/t59_reassessment_20261008/coverage_calendar.parquet",
    BTC_DATA / "btc_base70_information_value_20261018/common_execution_calendar.parquet",
    BTC_DATA / "btc_base70_raw_holdout_20261018/common_execution_calendar.parquet",
]
PILOT_MANIFEST = (
    MICRO_DIR / "availability/pmxt_empty_partition_pilot_20261019/pilot_manifest.json"
)
HOURLY_PROVENANCE_CSV = ROOT / "reports/btc_preopen/btc_empty_partition_provenance_20261019_hourly.csv"
PNL_COLUMNS = [
    "day_utc", "C_net_pnl_usd", "X_net_pnl_usd", "MARKET_net_pnl_usd",
    "RAW_net_pnl_usd", "X_MINUS_C_net_pnl_usd",
]
DAILY_COLUMNS = [
    "day_utc", "market_count", "available_input_count", "snapshot_matched",
    "source_events_observed", "source_coverage_chronology", "native_both_books",
    "unambiguous_asks", "uncrossed", "p_market_valid", "builder_eligible",
    "C_decisions", "C_executions", "X_decisions", "X_executions",
    "MARKET_decisions", "MARKET_executions", "RAW_decisions", "RAW_executions", "data_status",
]
DAILY_HEADERS = {
    "day_utc": "Dzień UTC", "market_count": "Rynki", "available_input_count": "Dostępne wejścia",
    "snapshot_matched": "Snapshot T−60", "source_events_observed": "Zdarzenia",
    "source_coverage_chronology": "Pokrycie/chronologia", "native_both_books": "Dwie księgi",
    "unambiguous_asks": "Ask jednoznaczne", "uncrossed": "Nie crossed", "p_market_valid": "p_market",
    "builder_eligible": "Końcowe", "C_decisions": "Decyzje C", "C_executions": "Wykonania C",
    "X_decisions": "Decyzje X", "X_executions": "Wykonania X",
    "MARKET_decisions": "Decyzje MARKET", "MARKET_executions": "Wykonania MARKET",
    "RAW_decisions": "Decyzje RAW",
    "RAW_executions": "Wykonania RAW", "data_status": "Status dnia",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _regenerate_skip_reason(path: Path) -> dict:
    table = parquet.read_table(path)
    reason_index = table.schema.get_field_index("fixed5_skip_reason")
    calendar = table.select([
        "market_confirmed", "causal_prediction_available", "t59_snapshot_available",
        "snapshot_market_event_count", "archive_entry_hour_available",
        "fixed5_priceable_side_count", "fixed5_buy_eligible",
    ]).to_pandas()
    original = table.column(reason_index).to_pylist()
    corrected = _fixed5_skip_reason(calendar).tolist()
    changed_rows = sum(before != after for before, after in zip(original, corrected, strict=True))
    updated = table.set_column(
        reason_index, table.schema.field(reason_index), pa.array(corrected, type=table.schema.field(reason_index).type),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    parquet.write_table(updated, temporary, compression="zstd")
    os.replace(temporary, path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "rows": len(corrected),
        "changed_rows": changed_rows,
        "corrected_no_archive_events": corrected.count("no_archive_events_observed_by_t59"),
        "remaining_archive_hour_missing": corrected.count("archive_hour_missing_at_entry"),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _update_hourly_provenance() -> None:
    manifest = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
    fields = [
        "pilot_source_object_bytes", "pilot_source_rows_from_metadata", "pilot_row_groups_read",
        "pilot_market_matches_before_time", "pilot_after_received_cutoff",
        "pilot_after_25h_history_lower_bound", "pilot_after_source_timestamp_cutoff",
        "pilot_selected_events", "pilot_downloaded_at_utc", "pilot_etag",
        "pilot_last_modified", "pilot_full_object_sha256_computed",
        "pilot_run_payload_upper_bound_bytes",
    ]
    with HOURLY_PROVENANCE_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        existing_fields = list(reader.fieldnames or [])
        records = list(reader)
    for item in records:
        sample = manifest["files"].get(item["hour_utc"])
        if sample is None:
            for field in fields:
                item[field] = ""
            continue
        item.update({
            "pilot_source_object_bytes": str(sample["object_size_bytes"]),
            "pilot_source_rows_from_metadata": str(sample["row_count_from_parquet_metadata"]),
            "pilot_row_groups_read": f"{sample['row_groups_read']}/{sample['row_groups_total']}",
            "pilot_market_matches_before_time": str(sample["market_matches_before_time"]),
            "pilot_after_received_cutoff": str(sample["after_timestamp_received_cutoff"]),
            "pilot_after_25h_history_lower_bound": str(sample["after_market_history_lower_bound"]),
            "pilot_after_source_timestamp_cutoff": str(sample["after_source_timestamp_cutoff"]),
            "pilot_selected_events": str(sample["audit_event_rows"]),
            "pilot_downloaded_at_utc": sample["downloaded_at_utc"],
            "pilot_etag": sample["etag"],
            "pilot_last_modified": sample["last_modified"],
            "pilot_full_object_sha256_computed": "false",
            "pilot_run_payload_upper_bound_bytes": str(manifest["cumulative_network_payload_upper_bound_bytes"]),
        })
    output_fields = existing_fields + [field for field in fields if field not in existing_fields]
    temporary = HOURLY_PROVENANCE_CSV.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, HOURLY_PROVENANCE_CSV)


def _replace_daily_report(daily_funnel: pd.DataFrame, daily_pnl: pd.DataFrame) -> None:
    text = MICRO_REPORT.read_text(encoding="utf-8")
    marker = "Dzienny lejek dla pełnego kalendarza 15 kwietnia–30 czerwca znajduje się "
    table_start = text.index("|", text.index(marker))
    table_end = text.index("\n\nFlagi dostępności godziny", table_start)
    table = microstructure._markdown_table(daily_funnel, DAILY_COLUMNS, DAILY_HEADERS)
    text = text[:table_start] + "\n".join(table) + text[table_end:]

    source_update = (
        "Lokalne partycje `event_parts` dla 384 godzin od 13 do 28 czerwca pozostają puste po selekcji. "
        "Pilot odczytał wszystkie 132 grup i 137 377 951 wierszy z obecnych oryginalnych obiektów dla "
        "2026-06-13T12, 2026-06-20T12 i 2026-06-28T12. W każdym dokładne dopasowania `market` z pełnego "
        "indeksu wyniosły 0 jeszcze przed filtrami czasu; oryginalny selektor i niezależny filtr są zgodne. "
        "To wyjaśnia te trzy próbki, nie pozostałe godziny. Stary checkpoint nie zawiera wersji/hashy źródła, "
        "więc zmiany zdalnego obiektu nie da się wykazać. Sześć startów nadal nie ma rekordu Gamma. "
        "Szczegóły i ograniczenia są w `reports/btc_preopen/btc_empty_partition_provenance_20261019.md`."
    )
    stale_source = "Lokalne partycje `event_parts` dla 384 godzin od 13 do 28 czerwca"
    if stale_source in text:
        source_start = text.index(stale_source)
        source_end = text.index("\n\nKod rekonstrukcji", source_start)
        text = text[:source_start] + source_update + text[source_end:]
    elif source_update not in text:
        raise RuntimeError("Expected source-provenance text was not found in the development report")

    pnl_statuses = daily_pnl.data_status.value_counts().to_dict()
    unavailable_days = int(daily_pnl.available_input_count.eq(0).sum())
    no_market_days = int(daily_pnl.market_count.eq(0).sum())
    c_decisions = int(daily_pnl.C_decisions.sum())
    raw_executions = int(daily_pnl.RAW_executions.sum())
    decision_only_days = int(daily_pnl.C_data_status.eq("decisions_without_executions").sum())
    status_paragraph = (
        f"\n\nDzienny status danych: {no_market_days} dni bez rynków, {unavailable_days} dni z rynkami bez dostępnych "
        f"wejść, {int(pnl_statuses.get('decisions_without_executions', 0))} dni z decyzjami i bez wykonań oraz "
        f"{int(pnl_statuses.get('executions_present', 0))} dni z wykonaniami. C, X i MARKET mają po {c_decisions:,} "
        f"decyzji oraz 0 wykonań; dla C/X decyzje bez wykonań wystąpiły przez {decision_only_days} dni. "
        f"RAW ma {raw_executions:,} wykonań. Liczniki grupują po `market_start_utc`; wartości PnL zachowują "
        f"dotychczasowe grupowanie po czasie decyzji. Kolumny `*_data_status` i liczniki są w dziennym pliku PnL."
    )
    economics_marker = "\n\nKoncentracja dzienna jest zapisana w manifeście."
    if status_paragraph.strip() not in text:
        text = text.replace(economics_marker, status_paragraph + economics_marker, 1)
    MICRO_REPORT.write_text(text, encoding="utf-8", newline="\n")


def main() -> dict:
    skip_results = [_regenerate_skip_reason(path) for path in SKIP_REASON_PATHS]
    _update_hourly_provenance()

    availability_path = MICRO_DIR / "availability/daily_funnel.csv"
    pnl_path = MICRO_DIR / "economics/daily_pnl.csv"
    availability = pd.read_csv(availability_path, parse_dates=["day_utc"])
    old_pnl = pd.read_csv(pnl_path)
    decisions = {
        variant: pd.read_parquet(MICRO_DIR / f"economics/{variant}_decisions.parquet")
        for variant in ECONOMIC_VARIANTS
    }
    trades = {
        variant: pd.read_parquet(MICRO_DIR / f"economics/{variant}_trades.parquet")
        for variant in ECONOMIC_VARIANTS
    }
    availability = microstructure._attach_daily_policy_counts(availability, decisions, trades)
    rebuilt_pnl = microstructure._daily_pnl(
        trades, pd.DatetimeIndex(pd.to_datetime(old_pnl.day_utc, utc=True)), decisions, availability,
    )
    for column in PNL_COLUMNS[1:]:
        if not np.allclose(old_pnl[column].to_numpy(dtype=float), rebuilt_pnl[column].to_numpy(dtype=float), rtol=0, atol=1e-12):
            raise RuntimeError(f"Diagnostic regeneration would change saved PnL values in {column}")
    if not np.array_equal(
        pd.DatetimeIndex(pd.to_datetime(old_pnl.day_utc, utc=True)).asi8,
        pd.DatetimeIndex(pd.to_datetime(rebuilt_pnl.day_utc, utc=True)).asi8,
    ):
        raise RuntimeError("Diagnostic regeneration changed the daily PnL calendar")

    microstructure._atomic_csv(availability, availability_path)
    diagnostic_columns = [column for column in rebuilt_pnl.columns if column not in PNL_COLUMNS]
    updated_pnl = pd.concat([old_pnl[PNL_COLUMNS].copy(), rebuilt_pnl[diagnostic_columns]], axis=1)
    microstructure._atomic_csv(updated_pnl, pnl_path)
    saved_pnl = pd.read_csv(pnl_path)
    for column in PNL_COLUMNS[1:]:
        if not np.array_equal(old_pnl[column].to_numpy(dtype=float), saved_pnl[column].to_numpy(dtype=float)):
            raise RuntimeError(f"Saved PnL values changed while writing diagnostic columns: {column}")

    _replace_daily_report(availability, updated_pnl)

    t59_manifest_path = ROOT / "reports/btc_preopen/t59_reassessment_20261008/data_manifest.json"
    t59_manifest = json.loads(t59_manifest_path.read_text(encoding="utf-8"))
    for result in skip_results:
        path = result["path"]
        if path in t59_manifest.get("artifact_files", {}):
            t59_manifest["artifact_files"][path].update({"sha256": result["sha256"], "size_bytes": result["bytes"]})
    _atomic_json(t59_manifest_path, t59_manifest)

    for name in (
        "btc_base70_information_value_20261018_manifest.json",
        "btc_base70_raw_holdout_20261018_manifest.json",
    ):
        manifest_path = ROOT / "reports/btc_preopen" / name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for result in skip_results:
            for record in _walk_dicts(manifest):
                if record.get("path") == result["path"]:
                    record.update({"sha256": result["sha256"], "size_bytes": result["bytes"]})
        _atomic_json(manifest_path, manifest)

    micro_manifest = json.loads(MICRO_MANIFEST.read_text(encoding="utf-8"))
    for path in (availability_path, pnl_path):
        relative = path.relative_to(ROOT).as_posix()
        micro_manifest["artifacts"][relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    report_relative = MICRO_REPORT.relative_to(ROOT).as_posix()
    micro_manifest["report"] = {
        "bytes": MICRO_REPORT.stat().st_size,
        "path": report_relative,
        "sha256": _sha256(MICRO_REPORT),
    }
    micro_manifest["availability_diagnostic_refresh"] = {
        "source": "saved availability table and saved C/X/MARKET/RAW decision/trade ledgers",
        "pnl_columns_preserved": True,
        "daily_data_status_counts": {str(key): int(value) for key, value in updated_pnl.data_status.value_counts().items()},
        "markets_without_available_inputs_days": int(updated_pnl.available_input_count.eq(0).sum()),
        "no_market_data_days": int(updated_pnl.market_count.eq(0).sum()),
        "full_replay_or_training": False,
    }
    _atomic_json(MICRO_MANIFEST, micro_manifest)

    result = {
        "skip_reason_calendars": skip_results,
        "availability_days": len(availability),
        "economic_days": len(updated_pnl),
        "pnl_preserved": True,
        "daily_data_status_counts": micro_manifest["availability_diagnostic_refresh"]["daily_data_status_counts"],
        "markets_without_available_inputs_days": int(updated_pnl.available_input_count.eq(0).sum()),
        "no_market_data_days": int(updated_pnl.market_count.eq(0).sum()),
        "C_decisions": int(updated_pnl.C_decisions.sum()),
        "RAW_executions": int(updated_pnl.RAW_executions.sum()),
    }
    return result


def _walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
