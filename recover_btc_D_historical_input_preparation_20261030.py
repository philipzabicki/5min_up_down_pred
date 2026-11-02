"""Recover auditable E0048 outputs after its frozen wall-time guard fails.

This recovery path consumes completed local products and self-authenticating
checkpoints only. It checks PMXT file metadata and rebuilds saved snapshot
frames from a terminal checkpoint; it never reads event rows or evaluates.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pandas.testing as pdt

from features import btc_D_historical_input_preparation_20261030 as e0048

ROOT = Path(__file__).resolve().parent
RUNTIME = e0048.RUNTIME
RUNNER_PID = 43936
CONTINUATION_COMMAND = "continue_btc_D_historical_input_preparation_20261030.py"
RUN_STARTED_UTC = "2026-10-30T01:04:44Z"
WALL_BUDGET_SECONDS = 14_400
EXPECTED_BOOK_REPLAY_HOURS = 2_377
MANIFEST_PATH = RUNTIME / "input_preparation_recovery_manifest.json"
REPORT_PATH = ROOT / "reports/btc_preopen/btc_D_historical_input_preparation_20261030_recovery.md"
OUTPUT_NAMES = (
    "feature_rows.parquet",
    "t60_books.parquet",
    "t59_books.parquet",
    "slot_coverage.parquet",
    "holdout_input_rows.parquet",
    "slot_coverage_summary.csv",
    "regime_coverage_summary.csv",
)


def _assert_runner_stopped() -> dict:
    if os.name != "nt":
        raise RuntimeError("E0048 recovery is frozen to the Windows runner environment")
    continuation_pattern = re.escape(CONTINUATION_COMMAND)
    command = (
        f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={RUNNER_PID}'; "
        "$original=$null -ne $p -and $p.CommandLine -match 'run_btc_D_historical_input_preparation_20261030\\.py'; "
        "$active=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' -and "
        "($_.CommandLine -match 'run_btc_D_historical_input_preparation_20261030\\.py' -or "
        f"$_.CommandLine -match '{continuation_pattern}') }} | "
        "ForEach-Object { [string]$_.ProcessId + ':' + $_.Name }); "
        "if ($active.Count -gt 0) { [Console]::Write(($active -join ',')); exit 23 }; "
        "if ($original) { [Console]::Write('original-runner-active'); exit 23 }; "
        "if ($null -ne $p) { [Console]::Write('pid-reused') }"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    process_state = result.stdout.strip()
    if result.returncode == 23:
        raise RuntimeError(f"E0048 replay process is still active: {process_state}")
    if result.returncode != 0:
        raise RuntimeError("could not verify that frozen E0048 replay processes exited")
    return {
        "runner_pid": RUNNER_PID,
        "runner_process_absent": True,
        "continuation_process_absent": True,
        "pid_reused": process_state == "pid-reused",
    }


def _read_product_schemas() -> dict[str, dict]:
    result = {}
    for name in OUTPUT_NAMES:
        path = RUNTIME / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"completed E0048 product is missing or empty: {name}")
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            columns = list(parquet.schema_arrow.names)
            forbidden = e0048.FORBIDDEN_SOURCE_COLUMNS.intersection(column.lower() for column in columns)
            if forbidden:
                raise RuntimeError(f"forbidden evaluation fields are present in {name}: {sorted(forbidden)}")
            result[name] = {"rows": int(parquet.metadata.num_rows), "columns": columns}
    expected_rows = {
        "feature_rows.parquet": e0048.EXPECTED_FEATURE_ROWS,
        "t60_books.parquet": e0048.EXPECTED_MARKET_SLOTS,
        "t59_books.parquet": e0048.EXPECTED_MARKET_SLOTS,
        "slot_coverage.parquet": e0048.EXPECTED_MARKET_SLOTS,
        "holdout_input_rows.parquet": e0048.EXPECTED_MARKET_SLOTS,
    }
    for name, expected in expected_rows.items():
        if result[name]["rows"] != expected:
            raise RuntimeError(f"{name} has {result[name]['rows']} rows; expected {expected}")
    return result


def _coverage_checks() -> tuple[dict, list[dict], dict]:
    summary = pd.read_csv(RUNTIME / "slot_coverage_summary.csv")
    regimes = pd.read_csv(RUNTIME / "regime_coverage_summary.csv")
    if summary.empty or regimes.empty:
        raise RuntimeError("completed E0048 coverage summaries are empty")
    summary_records = summary.to_dict(orient="records")
    summary_counts = {str(row["metric"]): int(row["count"]) for row in summary_records}
    if summary_counts.get("calendar_slots") != e0048.EXPECTED_MARKET_SLOTS:
        raise RuntimeError("coverage summary calendar slot count differs from the frozen scope")

    coverage = pd.read_parquet(
        RUNTIME / "slot_coverage.parquet",
        columns=[
            "t59_archive_entry_hour_available",
            "t59_archive_freshness_window_available",
            "t59_archive_window_missing_source_hours",
        ],
    )
    entry = coverage["t59_archive_entry_hour_available"].fillna(False).astype(bool)
    freshness = coverage["t59_archive_freshness_window_available"].fillna(False).astype(bool)
    missing_window = pd.to_numeric(coverage["t59_archive_window_missing_source_hours"], errors="coerce").fillna(0)
    regime_records = []
    for record in regimes.to_dict(orient="records"):
        regime_records.append({
            key: int(value) if isinstance(value, np.integer) else value
            for key, value in record.items()
        })
    return summary_counts, regime_records, {
        "market_windows_affected_at_t59": int(missing_window.gt(0).sum()),
        "windows_without_entry_or_freshness": int((~entry | ~freshness).sum()),
        "windows_with_entry_but_missing_prior_history": int((entry & missing_window.gt(0)).sum()),
    }


def _verify_panel_and_features() -> dict:
    feature_path = RUNTIME / "feature_rows.parquet"
    panel_path = RUNTIME / "panel_repeat" / "feature_rows_20260815.parquet"
    if not panel_path.is_file():
        raise RuntimeError("saved E0048 deterministic panel output is missing")
    features = pd.read_parquet(feature_path)
    panel = pd.read_parquet(panel_path)
    if "flow_source_join_matched" in panel:
        panel = panel.drop(columns=["flow_source_join_matched"])
    if len(features) != e0048.EXPECTED_FEATURE_ROWS or len(panel) != 48:
        raise RuntimeError("saved feature or deterministic panel row count differs from E0048")
    if "Opened" not in features or "Opened" not in panel:
        raise RuntimeError("feature products do not contain the frozen Opened time key")
    features["Opened"] = pd.to_datetime(features["Opened"], utc=True, errors="raise")
    panel["Opened"] = pd.to_datetime(panel["Opened"], utc=True, errors="raise")
    reference = features.loc[features.Opened.isin(panel.Opened)].reset_index(drop=True)
    panel = panel.reset_index(drop=True)
    pdt.assert_frame_equal(panel, reference, check_exact=True, check_dtype=True, check_like=False)
    return {
        "rows": len(panel),
        "status": "PASS_EXACT",
        "date_utc": "2026-08-15",
        "panel_sha256": e0048.sha256_file(panel_path),
    }


def _checkpoint_head() -> dict:
    path = RUNTIME / "checkpoints" / "native_book_replay.json"
    with path.open("rb") as stream:
        head = stream.read(8192).decode("utf-8")
    result = {}
    for key in ("hour_cursor", "event_parts_read", "event_projected_rows_read", "event_rows_replayed"):
        match = re.search(r'"' + key + r'":([0-9]+)', head)
        if not match:
            raise RuntimeError(f"book checkpoint header is missing {key}")
        result[key] = int(match.group(1))
    return result


def _write_report(manifest: dict, summary_counts: dict, regimes: list[dict]) -> None:
    lines = [
        "# E0048: odzysk artefaktow po przekroczeniu budzetu",
        "",
        "**Wynik: produkty zmaterializowane; limit 4 godzin przekroczony. E0048 nie jest oznaczane jako zaliczone w ramach zamrozonego budzetu.**",
        "",
        f"Runner v4 wystartowal `{manifest['wall_time']['started_utc']}`. Dokladny czas zakonczenia procesu nie zostal zachowany; materializacja produktow nastapila co najmniej {manifest['wall_time']['minimum_observed_seconds']:.1f} s po starcie, przy budzecie {manifest['wall_time']['budget_seconds']} s.",
        "Odzysk sprawdzil kompletny checkpoint i gotowe produkty. Nie odczytywal ponownie wierszy zdarzen PMXT; sprawdzil tylko metadane Parquet potrzebne do weryfikacji tozsamosci checkpointu. Nie uruchomiono modelu ani ewaluatora.",
        "",
        "## Zakres i pokrycie",
        "",
        f"Cechy: {manifest['feature_products']['rows']:,} wierszy, {manifest['feature_products']['feature_count']:,} cech; panel powtorzony: {manifest['deterministic_panel']['status']} ({manifest['deterministic_panel']['rows']} wierszy).",
        f"Sloty rynku: {summary_counts['calendar_slots']:,}; T-60 obie poprawne strony: {summary_counts['t60_both_native_sides_valid']:,}; T-59: {summary_counts['t59_both_native_sides_valid']:,}; source-only kompletne: {summary_counts['source_only_components_complete']:,}.",
        f"Replay ksiazek: {manifest['book_replay']['cursor_hours']:,}/{manifest['book_replay']['expected_hours']:,} godzin; {manifest['book_replay']['event_rows_replayed']:,} zdarzen odtworzonych; {len(manifest['book_replay']['missing_source_hours'])} brakujacych godzin zrodlowych. Zachowano znana luke E0043: {manifest['known_gap_reconciliation']['known_gap_missing_hours_observed']}/68 godzin.",
        "",
        "| Miara pokrycia | Liczba |",
        "|---|---:|",
    ]
    for name, count in summary_counts.items():
        lines.append(f"| {name} | {count:,} |")
    lines.extend(["", "### Rezymy", "", "| Rezym | Sloty | Tozsamosc | Cechy gotowe | FLOW | T-60 | T-59 | Oplata T-60 | Oplata T-59 | Source-only |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in regimes:
        lines.append(
            f"| {row['regime']} | {int(row['calendar_slots']):,} | {int(row['market_identity_confirmed']):,} | {int(row['feature_ready']):,} | {int(row['flow_exact_key_join_matched']):,} | {int(row['t60_both_native_sides_valid']):,} | {int(row['t59_both_native_sides_valid']):,} | {int(row['t60_fee_evidence_known']):,} | {int(row['t59_fee_evidence_known']):,} | {int(row['source_only_components_complete']):,} |"
        )
    lines.extend([
        "",
        "## Integralnosc i granice uzycia",
        "",
        f"Checkpoint ksiazek ma zweryfikowana integralnosc i tozsamosc; hash: `{manifest['book_replay']['checkpoint_sha256']}`. Hashy wszystkich siedmiu produktow i checkpointu nie zmieniano w miejscu; manifest odzyskiwania: `{MANIFEST_PATH.relative_to(ROOT).as_posix()}`.",
        "Targety, etykiety, wyniki, rozstrzygniecia i payouty nie zostaly odczytane. Predykcje, EV, fille, transakcje, PnL i metryki jakosci: 0. `evaluation_authorized=false`, `activated=false`, `HOLDOUT_NOT_RUN`; E0037, T0009 i luka E0043 pozostaja bez zmian.",
        "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def recover() -> dict:
    process_check = _assert_runner_stopped()
    protocol, protocol_sha256 = e0048.load_protocol()
    if MANIFEST_PATH.exists() or REPORT_PATH.exists():
        raise RuntimeError("E0048 recovery artifacts already exist; preserve them and review before rerunning")

    output_schemas = _read_product_schemas()
    features = _verify_panel_and_features()
    summary_counts, regimes, gap_counts = _coverage_checks()

    input_path = RUNTIME / "holdout_input_rows.parquet"
    market_columns = ["market_start_utc", "condition_id", "up_token_id", "down_token_id", "market_identity_status"]
    market_rows = pd.read_parquet(input_path, columns=market_columns)
    valid = market_rows.loc[market_rows.market_identity_status.eq("CONFIRMED")].copy()
    valid["market_start_utc"] = pd.to_datetime(valid["market_start_utc"], utc=True, errors="raise")
    hours = pd.date_range(
        (valid.market_start_utc.dt.floor("h") - pd.Timedelta(hours=25)).min(),
        (valid.market_start_utc - pd.Timedelta(seconds=59)).dt.floor("h").max(),
        freq="h",
    )
    checkpoint_head = _checkpoint_head()
    if len(hours) != EXPECTED_BOOK_REPLAY_HOURS or checkpoint_head["hour_cursor"] != len(hours):
        raise RuntimeError(f"book replay checkpoint is incomplete: cursor {checkpoint_head['hour_cursor']} / {len(hours)}")

    # The frozen replay function performs identity validation and reconstructs
    # saved snapshots, but its event loop is empty at the verified terminal cursor.
    recovered_t60, recovered_t59, _quality, book_manifest = e0048.replay_native_books(
        protocol, market_rows, protocol_sha256,
    )
    if book_manifest["replayed_event_rows"] != checkpoint_head["event_rows_replayed"]:
        raise RuntimeError("terminal checkpoint event count changed during snapshot recovery")
    for prefix, recovered_frame in (("t60", recovered_t60), ("t59", recovered_t59)):
        saved_frame = pd.read_parquet(RUNTIME / f"{prefix}_books.parquet")
        pdt.assert_frame_equal(saved_frame, recovered_frame, check_exact=True, check_dtype=False, check_like=False)
    book_checkpoint_path = RUNTIME / "checkpoints" / "native_book_replay.json"
    cursor = checkpoint_head["hour_cursor"]
    missing_hours = book_manifest["missing_source_hours"]
    known_gap_start = pd.Timestamp("2026-08-15T10:00:00Z")
    known_gap_end = pd.Timestamp("2026-08-18T06:00:00Z")
    missing_hour_set = {pd.Timestamp(value) for value in missing_hours}
    expected_known_gap = set(pd.date_range(known_gap_start, known_gap_end - pd.Timedelta(hours=1), freq="h"))
    if not expected_known_gap.issubset(missing_hour_set):
        raise RuntimeError("recovered replay does not preserve E0043's frozen 68-hour PMXT gap")
    if gap_counts != {
        "market_windows_affected_at_t59": 1_115,
        "windows_without_entry_or_freshness": 816,
        "windows_with_entry_but_missing_prior_history": 299,
    }:
        raise RuntimeError("recovered T-59 coverage does not match E0043's frozen 816/299/1,115 counts")
    cp_mtime = datetime.fromtimestamp(book_checkpoint_path.stat().st_mtime, tz=timezone.utc)
    product_mtimes = [datetime.fromtimestamp((RUNTIME / name).stat().st_mtime, tz=timezone.utc) for name in OUTPUT_NAMES]
    products_completed_at = max(product_mtimes)
    started = datetime.fromisoformat(RUN_STARTED_UTC.replace("Z", "+00:00"))
    if cp_mtime < started or products_completed_at < started:
        raise RuntimeError("recovered checkpoint/products predate the frozen E0048 v4 runner")
    minimum_wall_seconds = max(0.0, (products_completed_at - started).total_seconds())
    if minimum_wall_seconds <= WALL_BUDGET_SECONDS:
        raise RuntimeError("saved product timestamps do not prove that the frozen wall-time budget was exceeded")

    output_hashes = {}
    for name in OUTPUT_NAMES:
        path = RUNTIME / name
        output_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = {
            "sha256": e0048.sha256_file(path),
            "size_bytes": path.stat().st_size,
            "rows": output_schemas.get(name, {}).get("rows"),
        }
    recovery_time = datetime.now(timezone.utc)
    panel_manifest = features | {"status": "PASS_EXACT"}
    book_manifest.update({
        "cursor_hours": cursor,
        "expected_hours": len(hours),
        "event_rows_replayed": book_manifest["replayed_event_rows"],
        "projected_event_rows_read_by_original_runner": checkpoint_head["event_projected_rows_read"],
        "checkpoint_last_modified_utc": cp_mtime.isoformat().replace("+00:00", "Z"),
    })
    gap_reconciliation = {
        "known_gap_start_utc": known_gap_start.isoformat().replace("+00:00", "Z"),
        "known_gap_end_exclusive_utc": known_gap_end.isoformat().replace("+00:00", "Z"),
        "known_gap_expected_hours": len(expected_known_gap),
        "known_gap_missing_hours_observed": len(expected_known_gap.intersection(missing_hour_set)),
        "known_gap_interval_fully_missing": expected_known_gap.issubset(missing_hour_set),
        **gap_counts,
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "E0048_POST_RUN_RECOVERY",
        "experiment_id": "E0048",
        "task_id": "T0017",
        "decision_id": "D0025",
        "status": "SOURCE_PRODUCTS_RECOVERED_BUDGET_EXCEEDED",
        "acceptance_status": "NOT_ACCEPTED_WITHIN_FROZEN_4_HOUR_BUDGET",
        "protocol_path": str(e0048.PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": protocol_sha256,
        "recovery_script_path": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "recovery_script_sha256": e0048.sha256_file(Path(__file__)),
        "authorization": protocol["authorization"],
        "scope": protocol["scope"],
        "wall_time": {
            "started_utc": RUN_STARTED_UTC,
            "products_completed_at_utc_lower_bound": products_completed_at.isoformat().replace("+00:00", "Z"),
            "minimum_observed_seconds": minimum_wall_seconds,
            "budget_seconds": WALL_BUDGET_SECONDS,
            "budget_exceeded": True,
            "exact_process_exit_time_recorded": False,
        },
        "runner_exit_check": process_check,
        "recovery_method": {
            "source": "completed local products and self-authenticating checkpoints",
            "pmxt_event_payload_rows_read_during_recovery": 0,
            "pmxt_parquet_metadata_files_read_for_identity": len(hours) - len(missing_hours),
            "new_event_replay_started": False,
            "model_loaded_or_invoked": False,
            "evaluator_invoked": False,
            "evaluation_authorized": False,
            "activated": False,
            "holdout_status": "HOLDOUT_NOT_RUN",
        },
        "feature_products": {
            "rows": output_schemas["feature_rows.parquet"]["rows"],
            "columns": len(output_schemas["feature_rows.parquet"]["columns"]),
            "feature_count": max(0, len(output_schemas["feature_rows.parquet"]["columns"]) - 1),
            "deterministic_panel": features,
        },
        "deterministic_panel": panel_manifest,
        "book_replay": book_manifest,
        "book_replay_peak_rss_bytes": book_manifest["peak_rss_bytes"],
        "coverage_summary": summary_counts,
        "regime_coverage_summary": regimes,
        "known_gap_reconciliation": gap_reconciliation,
        "output_hashes": output_hashes,
        "created_at_utc": recovery_time.isoformat().replace("+00:00", "Z"),
    }
    _write_report(manifest, summary_counts, regimes)
    manifest["report_path"] = str(REPORT_PATH.relative_to(ROOT)).replace("\\", "/")
    manifest["report_sha256"] = e0048.sha256_file(REPORT_PATH)
    manifest["recovery_manifest_path"] = str(MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/")
    e0048.atomic_json(MANIFEST_PATH, manifest)
    return manifest


if __name__ == "__main__":
    recovered = recover()
    print(json.dumps({
        "status": recovered["status"],
        "acceptance_status": recovered["acceptance_status"],
        "cursor_hours": recovered["book_replay"]["cursor_hours"],
        "expected_hours": recovered["book_replay"]["expected_hours"],
        "manifest": recovered["recovery_manifest_path"],
        "report": recovered["report_path"],
    }, ensure_ascii=False))
