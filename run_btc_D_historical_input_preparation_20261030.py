"""Run the frozen E0048 source-only historical input preparation."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["NUMBA_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from features import btc_D_historical_input_preparation_20261030 as e0048


ROOT = e0048.ROOT
RUNTIME = e0048.RUNTIME
REPORT_PATH = ROOT / "reports/btc_preopen/btc_D_historical_input_preparation_20261030.md"
WORKER_ENV = "BTC_D_E0048_FEATURE_WORKER"
REGIMES = (
    ("single", pd.Timestamp("2026-07-01T00:00:00Z"), pd.Timestamp("2026-08-07T00:00:00Z")),
    ("TWAP30", pd.Timestamp("2026-08-07T00:00:00Z"), pd.Timestamp("2026-08-14T00:00:00Z")),
    ("TWAP60", pd.Timestamp("2026-08-14T00:00:00Z"), pd.Timestamp("2026-10-07T00:00:00Z")),
)
_PEAK_RSS = 0


def _resources() -> dict:
    global _PEAK_RSS
    process_memory = e0048.process_memory_bytes()
    _PEAK_RSS = max(_PEAK_RSS, process_memory["peak_working_set_bytes"])
    if os.name == "nt":
        listing = subprocess.run(["tasklist.exe", "/FO", "CSV", "/NH"], capture_output=True, check=True, text=True, encoding="mbcs", errors="replace").stdout
        processes = []
        for row in csv.reader(listing.splitlines()):
            if len(row) < 5 or not row[0].lower().startswith("python"):
                continue
            memory_kib = "".join(character for character in row[4] if character.isdigit())
            processes.append({"pid": int(row[1]), "working_set_bytes": int(memory_kib) * 1024 if memory_kib else None})
    else:
        processes = [{"pid": os.getpid(), "working_set_bytes": process_memory["working_set_bytes"]}]
    return {
        "available_ram_bytes": e0048.available_ram_bytes(),
        "free_workspace_disk_bytes": int(shutil.disk_usage(str(ROOT)).free),
        "python_process_count_observed": len(processes),
        "other_python_working_sets_observed": processes,
        "this_process_working_set_bytes": process_memory["working_set_bytes"],
        "this_process_peak_working_set_bytes": int(_PEAK_RSS),
    }


def _resource_gate() -> dict:
    measured = _resources()
    minimum_ram = int(8 * 1024**3)
    minimum_disk = int(20 * 1024**3)
    if measured["available_ram_bytes"] < minimum_ram:
        raise RuntimeError("E0048 stopped at its frozen resource gate: available RAM is below 8 GiB")
    if measured["free_workspace_disk_bytes"] < minimum_disk:
        raise RuntimeError("E0048 stopped at its frozen resource gate: free workspace disk is below 20 GiB")
    return measured


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _checkpoint_payload(protocol_sha256: str, identity: str) -> dict:
    from features.btc_D_continuation_20261029 import load_checkpoint

    return load_checkpoint(RUNTIME / "checkpoints" / e0048.CHECKPOINT_NAME, identity_sha256=identity)


def _worker_loop() -> None:
    protocol, protocol_sha256 = e0048.load_protocol()
    _resource_gate()
    identity = e0048._feature_identity(protocol_sha256, protocol["parent"]["bundle_manifest_sha256"], protocol["parent"]["checkpoint_semantic_sha256"])
    checkpoint = RUNTIME / "checkpoints" / e0048.CHECKPOINT_NAME
    if not checkpoint.is_file():
        raise RuntimeError("fresh-process E0048 worker has no frozen-prefix checkpoint")
    while True:
        payload = _checkpoint_payload(protocol_sha256, identity)
        processed = int(payload["processed_feature_rows"])
        if processed >= e0048.EXPECTED_FEATURE_ROWS:
            break
        count = min(e0048.PARTITION_ROWS, e0048.EXPECTED_FEATURE_ROWS - processed)
        partition = processed // e0048.PARTITION_ROWS + 1
        e0048._materialize_feature_chunk(protocol, protocol_sha256, identity, count, output_name=f"feature_part_{partition:06d}")
        resource_state = _resource_gate()
        e0048.atomic_json(RUNTIME / "resource_samples" / f"feature_part_{partition:06d}.json", resource_state)


def _feature_output_names(bundle_manifest: dict) -> list[str]:
    names = ["Opened", *bundle_manifest["base_feature_order"], *bundle_manifest["memory_feature_order"]]
    esn_order = bundle_manifest["esn_feature_order_by_seed"]
    for seed in bundle_manifest["reservoir_seeds"]:
        names.extend(esn_order[str(seed)])
    if len(names) != 103 or len(set(names)) != 103:
        raise RuntimeError("frozen E0041 input width/order is not Opened plus 102 unique features")
    return names


def _feature_value_status(features: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, list[tuple[str, ...]], dict[str, int]]:
    complete = np.ones(len(features), dtype=bool)
    missing_by_row: dict[int, list[str]] = {}
    missing_by_feature = {}
    for name in feature_names:
        values = features[name].to_numpy(copy=False)
        if not np.issubdtype(values.dtype, np.number):
            raise RuntimeError(f"E0048 feature column is not numeric: {name}")
        valid = np.isfinite(values)
        missing = np.flatnonzero(~valid)
        missing_by_feature[name] = int(len(missing))
        complete &= valid
        for row_index in missing:
            missing_by_row.setdefault(int(row_index), []).append(name)
    missing_fields = [tuple(missing_by_row.get(row_index, ())) for row_index in range(len(features))]
    return complete, missing_fields, missing_by_feature


def _collect_feature_rows(protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    part_paths = sorted((RUNTIME / "minute_features").glob("feature_part_*.parquet"))
    frames = [pd.read_parquet(path) for path in part_paths]
    if not frames:
        raise RuntimeError("E0048 feature builder produced no minute partitions")
    internal = pd.concat(frames, ignore_index=True)
    internal["Opened"] = pd.to_datetime(internal["Opened"], utc=True, errors="raise")
    if len(internal) != e0048.EXPECTED_FEATURE_ROWS:
        raise RuntimeError("E0048 feature part row count differs from the frozen 141,116-minute range")
    expected = e0048.minute_keys()
    if not pd.DatetimeIndex(internal["Opened"]).equals(expected):
        raise RuntimeError("E0048 feature parts are not the exact contiguous frozen Opened sequence")
    flow_match = internal["flow_source_join_matched"].astype(bool).reset_index(drop=True)
    bundle_manifest = json.loads((ROOT / protocol["parent"]["bundle_manifest_path"]).read_text(encoding="utf-8"))
    feature_names = _feature_output_names(bundle_manifest)
    features = internal.loc[:, feature_names].copy()
    if len(features.columns) - 1 != 102:
        raise RuntimeError("E0048 feature output does not contain exactly 102 unique features")
    if any(column.lower() in e0048.FORBIDDEN_SOURCE_COLUMNS for column in features.columns):
        raise RuntimeError("forbidden target/evaluation column entered E0048 feature rows")
    complete, missing_fields, missing_by_feature = _feature_value_status(features, feature_names[1:])
    for name in bundle_manifest["base_feature_order"][64:]:
        if features[name].dtype != np.dtype("float32"):
            raise RuntimeError(f"E0041 historical FLOW column did not retain its float32 cast: {name}")
    _atomic_parquet(features, RUNTIME / "feature_rows.parquet")
    feature_meta = {
        "rows": len(features),
        "columns": list(features.columns),
        "feature_count": len(features.columns) - 1,
        "first_opened_utc": features.Opened.iloc[0].isoformat().replace("+00:00", "Z"),
        "last_opened_utc": features.Opened.iloc[-1].isoformat().replace("+00:00", "Z"),
        "rows_with_all_finite_features": int(complete.sum()),
        "rows_with_missing_or_nonfinite_features": int((~complete).sum()),
        "missing_or_nonfinite_by_feature": missing_by_feature,
        "column_dtypes": {name: str(dtype) for name, dtype in features.dtypes.items()},
        "flow_exact_join_matches": int(flow_match.sum()),
        "flow_exact_join_missing": int((~flow_match).sum()),
        "minute_partitions": [{"path": str(path.relative_to(ROOT)), "rows": int(pq.ParquetFile(path).metadata.num_rows), "sha256": e0048.sha256_file(path)} for path in part_paths],
    }
    feature_status = pd.DataFrame({
        "Opened": features.Opened,
        "feature_values_complete": complete,
        "feature_missing_fields": missing_fields,
        "flow_source_join_matched": flow_match,
    })
    return features, feature_status, feature_meta


def _deterministic_panel_repeat(protocol: dict, protocol_sha256: str, identity: str, features: pd.DataFrame, bundle_manifest: dict) -> dict:
    import pandas.testing as pdt
    from features.btc_D_continuation_20261029 import load_checkpoint

    c1_path = RUNTIME / "checkpoints" / "feature_part_000001_state.json"
    c1 = load_checkpoint(c1_path, identity_sha256=identity)
    next_opened = pd.Timestamp(c1["next_expected_opened_utc"])
    if next_opened.normalize() != pd.Timestamp("2026-08-15T00:00:00Z"):
        raise RuntimeError("E0048 first partition boundary is not the preregistered 2026-08-15 panel date")
    panel_count = 48
    panel_checkpoint = RUNTIME / "checkpoints" / "panel_repeat_state.json"
    panel_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copyfile(c1_path, panel_checkpoint)
    panel_path = RUNTIME / "panel_repeat" / "feature_rows_20260815.parquet"
    repeated, _ = e0048._materialize_feature_chunk(
        protocol, protocol_sha256, identity, panel_count,
        output_name="panel_repeat_20260815", checkpoint_path=panel_checkpoint, output_path=panel_path,
    )
    repeated = repeated.drop(columns=["flow_source_join_matched"])
    reference = features.loc[features.Opened.isin(repeated.Opened)].reset_index(drop=True)
    repeated = repeated.reset_index(drop=True)
    pdt.assert_frame_equal(repeated, reference, check_exact=True, check_dtype=True, check_like=False)
    return {
        "status": "PASS_EXACT",
        "date_utc": "2026-08-15",
        "rows": len(repeated),
        "first_opened_utc": repeated.Opened.iloc[0].isoformat().replace("+00:00", "Z"),
        "last_opened_utc": repeated.Opened.iloc[-1].isoformat().replace("+00:00", "Z"),
        "checkpoint_before_panel_sha256": e0048.sha256_file(c1_path),
        "panel_repeat_sha256": e0048.sha256_file(panel_path),
    }


def _regime_name(start: pd.Timestamp) -> str:
    for name, begin, end in REGIMES:
        if begin <= start < end:
            return name
    raise RuntimeError(f"market start is outside the frozen regime ranges: {start}")


def _source_only_input_rows(protocol: dict, market_rows: pd.DataFrame, features: pd.DataFrame, flow_status: pd.DataFrame, source_quality: pd.DataFrame) -> pd.DataFrame:
    calendar = market_rows.copy()
    calendar["regime"] = calendar.market_start_utc.map(_regime_name)
    features_for_join = features.rename(columns={"Opened": "feature_opened_utc"})
    input_rows = calendar.merge(features_for_join, on="feature_opened_utc", how="left", validate="one_to_one", indicator="_feature_join")
    feature_row_present = input_rows["_feature_join"].eq("both")
    input_rows.drop(columns=["_feature_join"], inplace=True)
    input_rows["feature_row_present"] = feature_row_present.astype(bool)
    if "condition_id" not in input_rows:
        raise RuntimeError("technical market identity was dropped from the source-only join")
    flow_status = flow_status.rename(columns={"Opened": "feature_opened_utc"})
    input_rows = input_rows.merge(flow_status, on="feature_opened_utc", how="left", validate="one_to_one")
    input_rows["feature_values_complete"] = input_rows["feature_values_complete"].fillna(False).astype(bool)
    input_rows["feature_missing_fields"] = input_rows["feature_missing_fields"].map(lambda value: tuple(value) if isinstance(value, (tuple, list)) else ())
    input_rows["flow_source_join_matched"] = input_rows["flow_source_join_matched"].fillna(False).astype(bool)
    input_rows["feature_ready"] = input_rows["feature_row_present"] & input_rows["feature_values_complete"]

    identity_columns = {"condition_id", "up_token_id", "down_token_id"}
    source_payload = source_quality.drop(columns=[name for name in identity_columns if name in source_quality.columns])
    input_rows = input_rows.merge(source_payload, on="market_start_utc", how="left", validate="one_to_one")
    aliases = protocol["evaluation_interface"]["source_field_aliases"]
    for target, source in aliases.items():
        if source not in input_rows:
            raise RuntimeError(f"source-only book join is missing evaluator source field {source}")
        input_rows[target] = input_rows[source]
    required_fields = set(protocol["evaluation_interface"]["evaluator_source_required_fields"])
    missing_fields = sorted(required_fields.difference(input_rows.columns))
    if missing_fields:
        raise RuntimeError(f"source-only input is missing evaluator-compatible fields: {missing_fields}")
    forbidden_fields = set(e0048.FORBIDDEN_SOURCE_COLUMNS)
    if forbidden_fields.intersection(input_rows.columns):
        raise RuntimeError("source-only input unexpectedly contains an evaluation target or resolution field")
    return input_rows


def _build_slot_products(protocol: dict, market_rows: pd.DataFrame, features: pd.DataFrame, flow_status: pd.DataFrame, t60: pd.DataFrame, t59: pd.DataFrame, source_quality: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    calendar = market_rows.copy()
    calendar["regime"] = calendar.market_start_utc.map(_regime_name)
    input_rows = _source_only_input_rows(protocol, calendar, features, flow_status, source_quality)
    if len(input_rows) != e0048.EXPECTED_MARKET_SLOTS:
        raise RuntimeError("source-only slot join changed the frozen market calendar row count")
    _atomic_parquet(input_rows, RUNTIME / "holdout_input_rows.parquet")

    t60 = t60.copy()
    t59 = t59.copy()
    _atomic_parquet(t60, RUNTIME / "t60_books.parquet")
    _atomic_parquet(t59, RUNTIME / "t59_books.parquet")
    coverage = calendar.merge(
        input_rows.loc[:, ["market_start_utc", "feature_row_present", "feature_values_complete", "feature_missing_fields", "feature_ready", "flow_source_join_matched"]],
        on="market_start_utc", how="left", validate="one_to_one",
    )
    quality_columns = [column for column in source_quality.columns if column not in {"condition_id", "up_token_id", "down_token_id"}]
    coverage = coverage.merge(source_quality.loc[:, quality_columns], on="market_start_utc", how="left", validate="one_to_one")
    coverage["feature_ready"] = coverage["feature_ready"].fillna(False).astype(bool)
    coverage["feature_row_present"] = coverage["feature_row_present"].fillna(False).astype(bool)
    coverage["feature_values_complete"] = coverage["feature_values_complete"].fillna(False).astype(bool)
    coverage["feature_missing_fields"] = coverage["feature_missing_fields"].map(lambda value: tuple(value) if isinstance(value, (tuple, list)) else ())
    coverage["flow_source_join_matched"] = coverage["flow_source_join_matched"].fillna(False).astype(bool)
    if "source_quality_clean" not in coverage:
        raise RuntimeError("exact E0043 source quality helper did not produce a coverage result")
    coverage["source_quality_clean"] = coverage["source_quality_clean"].fillna(False).astype(bool)
    from utils.btc_t0009_source_audit import source_quarantine_reasons

    reasons = []
    feature_reasons = []
    input_reasons = []
    for row in coverage.to_dict(orient="records"):
        source_reasons = source_quarantine_reasons(row)
        feature_row_reasons = []
        if not row["feature_row_present"]:
            feature_row_reasons.append("feature_row_missing")
        if not row["feature_values_complete"]:
            feature_row_reasons.append("feature_value_missing_or_nonfinite")
        if not row["flow_source_join_matched"]:
            feature_row_reasons.append("flow_exact_key_join_missing")
        market_reasons = [] if row["market_identity_status"] == "CONFIRMED" else ["market_identity_unknown_or_conflict"]
        reasons.append(source_reasons)
        feature_reasons.append(feature_row_reasons)
        input_reasons.append([*market_reasons, *feature_row_reasons, *source_reasons, "source_ready_receive_time_provenance_unknown"])
    coverage["source_quarantine_reasons"] = reasons
    coverage["feature_quarantine_reasons"] = feature_reasons
    coverage["source_only_quarantine_reasons"] = input_reasons
    coverage["market_identity_status"] = coverage["market_identity_status"].fillna("UNKNOWN")
    coverage["source_ready_receive_time_provenance"] = "UNKNOWN"
    coverage["source_only_components_complete"] = coverage["feature_ready"] & coverage["market_identity_status"].eq("CONFIRMED") & coverage["source_quality_clean"]
    for cutoff in ("t60", "t59"):
        side_validity = []
        for side in ("up", "down"):
            side_prefix = f"{cutoff}_{side}"
            expected_token = coverage[f"{side}_token_id"].astype("string")
            direct_token = coverage[f"{side_prefix}_quote_source_token_id"].astype("string").eq(expected_token).fillna(False)
            side_valid = (
                coverage[f"{side_prefix}_reason"].eq("ask_valid_within_source_coverage")
                & direct_token
                & coverage[f"{side_prefix}_quote_complemented"].eq(False).fillna(False)
                & coverage[f"{side_prefix}_bbo_ask_comparable"].eq(True).fillna(False)
                & coverage[f"{side_prefix}_bbo_ask_mismatch"].eq(False).fillna(False)
                & coverage[f"{side_prefix}_ask_order_ambiguous"].eq(False).fillna(False)
                & coverage[f"{side_prefix}_bid_ask_crossed"].eq(False).fillna(False)
            )
            side_validity.append(side_valid)
        coverage[f"{cutoff}_both_native_sides_valid"] = side_validity[0] & side_validity[1]
    _atomic_parquet(coverage, RUNTIME / "slot_coverage.parquet")

    summary_rows = []
    summary_metrics = {
        "calendar_slots": len(coverage),
        "market_identity_confirmed": int(coverage.market_identity_status.eq("CONFIRMED").sum()),
        "market_identity_unknown_or_conflict": int((~coverage.market_identity_status.eq("CONFIRMED")).sum()),
        "feature_rows_present": int(coverage.feature_row_present.sum()),
        "feature_values_complete": int(coverage.feature_values_complete.sum()),
        "feature_rows_ready": int(coverage.feature_ready.sum()),
        "flow_exact_key_join_matched": int(coverage.flow_source_join_matched.sum()),
        "t60_both_native_sides_valid": int(coverage.t60_both_native_sides_valid.sum()),
        "t59_both_native_sides_valid": int(coverage.t59_both_native_sides_valid.sum()),
        "t60_entry_hour_available": int(coverage.t60_archive_entry_hour_available.fillna(False).sum()),
        "t59_entry_hour_available": int(coverage.t59_archive_entry_hour_available.fillna(False).sum()),
        "t60_freshness_window_available": int(coverage.t60_archive_freshness_window_available.fillna(False).sum()),
        "t59_freshness_window_available": int(coverage.t59_archive_freshness_window_available.fillna(False).sum()),
        "t60_full_source_window": int((coverage.t60_archive_window_hours_processed == coverage.t60_archive_window_hours_expected).fillna(False).sum()),
        "t59_full_source_window": int((coverage.t59_archive_window_hours_processed == coverage.t59_archive_window_hours_expected).fillna(False).sum()),
        "t60_fee_evidence_known": int(coverage.t60_fee_known.fillna(False).sum()),
        "t59_fee_evidence_known": int(coverage.t59_fee_known.fillna(False).sum()),
        "source_quality_clean_both_cutoffs": int(coverage.source_quality_clean.sum()),
        "source_only_components_complete": int(coverage.source_only_components_complete.sum()),
    }
    for metric, value in summary_metrics.items():
        summary_rows.append({"metric": metric, "count": int(value), "denominator": len(coverage)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RUNTIME / "slot_coverage_summary.csv", index=False, encoding="utf-8")
    regime_rows = []
    for regime, _, _ in REGIMES:
        rows = coverage.loc[coverage.regime.eq(regime)]
        regime_rows.append({
            "regime": regime,
            "calendar_slots": len(rows),
            "market_identity_confirmed": int(rows.market_identity_status.eq("CONFIRMED").sum()),
            "feature_rows_present": int(rows.feature_row_present.sum()),
            "feature_values_complete": int(rows.feature_values_complete.sum()),
            "feature_ready": int(rows.feature_ready.sum()),
            "flow_exact_key_join_matched": int(rows.flow_source_join_matched.sum()),
            "t60_both_native_sides_valid": int(rows.t60_both_native_sides_valid.sum()),
            "t59_both_native_sides_valid": int(rows.t59_both_native_sides_valid.sum()),
            "t60_full_source_window": int((rows.t60_archive_window_hours_processed == rows.t60_archive_window_hours_expected).fillna(False).sum()),
            "t59_full_source_window": int((rows.t59_archive_window_hours_processed == rows.t59_archive_window_hours_expected).fillna(False).sum()),
            "t60_fee_evidence_known": int(rows.t60_fee_known.fillna(False).sum()),
            "t59_fee_evidence_known": int(rows.t59_fee_known.fillna(False).sum()),
            "source_quality_clean_both_cutoffs": int(rows.source_quality_clean.sum()),
            "source_only_components_complete": int(rows.source_only_components_complete.sum()),
        })
    regime_summary = pd.DataFrame(regime_rows)
    regime_summary.to_csv(RUNTIME / "regime_coverage_summary.csv", index=False, encoding="utf-8")
    products = {
        "holdout_input_rows": {"rows": len(input_rows), "path": str((RUNTIME / "holdout_input_rows.parquet").relative_to(ROOT))},
        "slot_coverage": {"rows": len(coverage), "path": str((RUNTIME / "slot_coverage.parquet").relative_to(ROOT))},
        "slot_summary": {"rows": len(summary), "path": str((RUNTIME / "slot_coverage_summary.csv").relative_to(ROOT))},
        "regime_summary": {"rows": len(regime_summary), "path": str((RUNTIME / "regime_coverage_summary.csv").relative_to(ROOT))},
        "feature_ready_count": summary_metrics["feature_rows_ready"],
        "source_only_components_complete_count": summary_metrics["source_only_components_complete"],
        "identity_confirmed_count": summary_metrics["market_identity_confirmed"],
        "source_quality_clean_count": summary_metrics["source_quality_clean_both_cutoffs"],
    }
    return input_rows, coverage, summary, products


def _output_hashes() -> dict:
    paths = [
        RUNTIME / "feature_rows.parquet", RUNTIME / "t60_books.parquet", RUNTIME / "t59_books.parquet",
        RUNTIME / "slot_coverage.parquet", RUNTIME / "holdout_input_rows.parquet",
        RUNTIME / "slot_coverage_summary.csv", RUNTIME / "regime_coverage_summary.csv",
    ]
    return {str(path.relative_to(ROOT)): {"sha256": e0048.sha256_file(path), "size_bytes": path.stat().st_size} for path in paths}


def _write_report(manifest: dict, summary: pd.DataFrame, regime_summary: pd.DataFrame) -> None:
    counts = {row.metric: int(row.count) for row in summary.itertuples(index=False)}
    gaps = manifest["book_replay"]["known_gap_reconciliation"]
    resource = manifest["resources"]
    product_bytes = sum(item["size_bytes"] for item in manifest["output_hashes"].values())
    lines = [
        "# E0048: historyczne wejscia D bez oceny",
        "",
        f"Minuty raw przetworzone: **{manifest['raw_source']['feature_payload_rows_read']:,}**. Sloty rynku: **{counts['calendar_slots']:,}**. Sloty z kompletnymi cechami: **{counts['feature_rows_ready']:,}**.",
        f"Obie natywne strony poprawne: T-60 **{counts['t60_both_native_sides_valid']:,}**, T-59 **{counts['t59_both_native_sides_valid']:,}**. Kompletne komponenty source-only: **{counts['source_only_components_complete']:,}**; unikalne sloty z brakami: **{counts['calendar_slots'] - counts['source_only_components_complete']:,}**.",
        f"Potwierdzone identyfikacje rynkow: **{counts['market_identity_confirmed']:,}**. Dokladne dopasowania FLOW: **{counts['flow_exact_key_join_matched']:,}**. Czysta kwalifikacja zrodlowa na obu cutoffach: **{counts['source_quality_clean_both_cutoffs']:,}**.",
        "",
        "## Zakres i produkty",
        "",
        "Cechy obejmuja `Opened` od `2026-06-30T23:58:00Z` do `2026-10-06T23:53:00Z`; sloty rynku to `[2026-07-01T00:00:00Z, 2026-10-07T00:00:00Z)`. Zachowano granice single/TWAP30/TWAP60 i klucz `Opened = market_start_utc - 2 min`.",
        "",
        "Lokalne produkty runtime: `feature_rows.parquet`, `t60_books.parquet`, `t59_books.parquet`, `slot_coverage.parquet`, `holdout_input_rows.parquet` oraz podsumowania CSV. Source-only join zachowuje wymagane pola techniczne E0041; `target_polymarket_up` i `resolved_at_utc` pozostaja nieprzygotowane, wiec produkt nie jest `evaluation_rows.parquet`. Checkpointy cech i replayu sa atomowe.",
        "",
        "## Pokrycie zrodel",
        "",
        "| Miara | Liczba | Mianownik |",
        "|---|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.metric} | {int(row.count):,} | {int(row.denominator):,} |")
    lines.extend([
        "",
        "### Rezymy",
        "",
        "| Rezym | Sloty | Tozsamosc | Cechy | FLOW | Dwie strony T-60 | Dwie strony T-59 | Pelne T-60 okno | Pelne T-59 okno | Oplata T-60 | Oplata T-59 | Source-only komponenty |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in regime_summary.itertuples(index=False):
        lines.append(f"| {row.regime} | {row.calendar_slots:,} | {row.market_identity_confirmed:,} | {row.feature_ready:,} | {row.flow_exact_key_join_matched:,} | {row.t60_both_native_sides_valid:,} | {row.t59_both_native_sides_valid:,} | {row.t60_full_source_window:,} | {row.t59_full_source_window:,} | {row.t60_fee_evidence_known:,} | {row.t59_fee_evidence_known:,} | {row.source_only_components_complete:,} |")
    missing = manifest["book_replay"]["missing_source_hours"]
    lines.extend([
        "",
        "## Braki i pochodzenie czasu",
        "",
        f"Luka PMXT `[2026-08-15T10:00Z, 2026-08-18T06:00Z)` obejmuje 68 godzin; potwierdzono {gaps['known_gap_missing_hours_observed']} brakujacych godzin. E0043 uzgodnienie: {gaps['market_windows_affected_at_t59']} okien dotknietych, {gaps['windows_without_entry_or_freshness']} bez godziny wejscia/freshness i {gaps['windows_with_entry_but_missing_prior_history']} z brakujaca wczesniejsza historia. W calym replayu brakuje {len(missing)} godzinowych partycji; zdarzen nie uzupelniano.",
        "Niepotwierdzone identyfikacje pozostaja UNKNOWN, bez sztucznych condition ID. `source_ready_receive_time_provenance=UNKNOWN`: archiwalne receive time nie dowodzi, kiedy dane byly dostepne w chwili decyzji. Kompletne komponenty source-only nie oznaczaja RAW PASS.",
        "",
        "## Weryfikacja i zasoby",
        "",
        f"Panel powtorzony w nowym procesie: **{manifest['deterministic_panel']['status']}**, {manifest['deterministic_panel']['rows']} wierszy identycznych bitowo. Szczyt RSS: {resource['peak_rss_gib']:.2f} GiB; czas: {resource['wall_seconds']:.1f} s; zaobserwowano {resource['python_process_count_observed']} procesow Python, bez zmiany ich stanu. Wolny dysk przed startem: {resource['resources_before']['free_workspace_disk_bytes'] / 1024**3:.1f} GiB; rozmiar glownych produktow: {product_bytes / 1024**3:.2f} GiB.",
        "Walidacja testow i `git diff --check` jest zapisana w canonical records.",
        "",
        "## Granice uzycia",
        "",
        "Etykiety, targety, outcomes, resolution i payouty nie byly odczytywane ani projektowane. Prognozy, sygnaly, EV, fille, transakcje, PnL i metryki jakosci: 0. Modeli nie zaladowano ani nie wywolano; evaluatora nie uruchomiono. `evaluation_authorized=false`, `activated=false`, `HOLDOUT_NOT_RUN`; E0041 i jego evaluator/manifest pozostaja bez zmian. E0037/T0009 nie sa zamykane, a luka E0043 pozostaje.",
        "",
        f"Manifest: `{RUNTIME.relative_to(ROOT).as_posix()}/input_preparation_manifest.json`.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_preparation() -> dict:
    started = time.perf_counter()
    protocol, protocol_sha256 = e0048.load_protocol()
    hybrid_record = protocol["source_allowlist"]["verified_hybrid_raw_input"]
    resources_before = _resource_gate()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "minute_features", "builder", "panel_repeat"):
        (RUNTIME / name).mkdir(parents=True, exist_ok=True)
    checkpoint = RUNTIME / "checkpoints" / e0048.CHECKPOINT_NAME
    identity = e0048._feature_identity(protocol_sha256, protocol["parent"]["bundle_manifest_sha256"], protocol["parent"]["checkpoint_semantic_sha256"])
    if not checkpoint.exists():
        e0048.initialize_runtime(protocol, protocol_sha256)
        e0048._materialize_feature_chunk(protocol, protocol_sha256, identity, e0048.PARTITION_ROWS, output_name="feature_part_000001")
        resource_state = _resource_gate()
        e0048.atomic_json(RUNTIME / "resource_samples" / "feature_part_000001.json", resource_state)
    remaining = dict(os.environ)
    remaining[WORKER_ENV] = "1"
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve())], cwd=str(ROOT), env=remaining, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"fresh-process E0048 feature continuation exited with {completed.returncode}")
    checkpoint_payload = _checkpoint_payload(protocol_sha256, identity)
    if int(checkpoint_payload["processed_feature_rows"]) != e0048.EXPECTED_FEATURE_ROWS:
        raise RuntimeError("fresh-process E0048 feature continuation did not reach its frozen end")

    _e47, bundle, _transforms, _state, _dynamic_state, _identity = e0048.restore_feature_state(protocol, protocol_sha256)
    features, flow_status, feature_meta = _collect_feature_rows(protocol)
    panel = _deterministic_panel_repeat(protocol, protocol_sha256, identity, features, bundle["manifest"])
    market_rows, market_sources = e0048._calendar_mapping_inputs(protocol)
    t60, t59, source_quality, book_manifest = e0048.replay_native_books(protocol, market_rows, protocol_sha256)
    input_rows, coverage, summary, products = _build_slot_products(protocol, market_rows, features, flow_status, t60, t59, source_quality)
    regime_summary = pd.read_csv(RUNTIME / "regime_coverage_summary.csv")
    e0043_gap = protocol["source_allowlist"]["pmxt_native_event_parts"]["known_missing_interval_utc"]
    gap_hours_expected = pd.date_range(pd.Timestamp(e0043_gap["start_inclusive"]), pd.Timestamp(e0043_gap["end_exclusive"]) - pd.Timedelta(hours=1), freq="h")
    actual_missing = {pd.Timestamp(value) for value in book_manifest["missing_source_hours"]}
    gap_reconciliation = {
        "known_gap_expected_hours": len(gap_hours_expected),
        "known_gap_missing_hours_observed": sum(hour in actual_missing for hour in gap_hours_expected),
        "known_gap_interval_fully_missing": all(hour in actual_missing for hour in gap_hours_expected),
        "market_windows_affected_at_t59": int(coverage.t59_archive_window_missing_source_hours.fillna(0).gt(0).sum()),
        "windows_without_entry_or_freshness": int((~coverage.t59_archive_entry_hour_available.fillna(False) | ~coverage.t59_archive_freshness_window_available.fillna(False)).sum()),
        "windows_with_entry_but_missing_prior_history": int((coverage.t59_archive_entry_hour_available.fillna(False) & coverage.t59_archive_window_missing_source_hours.fillna(0).gt(0)).sum()),
    }
    if not gap_reconciliation["known_gap_interval_fully_missing"]:
        raise RuntimeError("E0048 source replay disagrees with the pinned E0043 68-hour gap")
    if (gap_reconciliation["market_windows_affected_at_t59"] != 1_115
            or gap_reconciliation["windows_without_entry_or_freshness"] != 816
            or gap_reconciliation["windows_with_entry_but_missing_prior_history"] != 299):
        raise RuntimeError("E0048 T-59 window coverage does not exactly reconcile E0043's frozen 816/299/1,115 counts")

    resource_after = _resource_gate()
    wall = time.perf_counter() - started
    if wall > 14_400:
        raise RuntimeError("E0048 exceeded its frozen 4-hour wall-time budget")
    if _resources()["this_process_peak_working_set_bytes"] > 24 * 1024**3:
        raise RuntimeError("E0048 exceeded its frozen 24 GiB peak RSS limit")
    output_hashes = _output_hashes()
    feature_checkpoint_path = RUNTIME / "checkpoints" / e0048.CHECKPOINT_NAME
    flow_source_records = {}
    hybrid_source_audits = []
    for path in sorted((RUNTIME / "source_audit").glob("feature_part_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        flow_source_records.update(record["flow_sources_verified"])
        hybrid_source_audits.append(record)
    worker_peak = max([resource_after["this_process_peak_working_set_bytes"], *[
        int(json.loads(path.read_text(encoding="utf-8"))["this_process_peak_working_set_bytes"])
        for path in (RUNTIME / "resource_samples").glob("feature_part_*.json")
    ]])
    book_peak = int(book_manifest.get("peak_rss_bytes", 0))
    manifest = {
        "schema_version": 1,
        "experiment_id": "E0048",
        "task_id": "T0017",
        "decision_id": "D0025",
        "protocol_path": str(e0048.PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": protocol_sha256,
        "status": "SOURCE_INPUTS_MATERIALIZED",
        "scope": protocol["scope"],
        "authorization": protocol["authorization"],
        "evaluation_interface": protocol["evaluation_interface"],
        "parent_identity": {
            "bundle_manifest_sha256": protocol["parent"]["bundle_manifest_sha256"],
            "continuation_manifest_sha256": protocol["parent"]["continuation_manifest_sha256"],
            "checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
            "checkpoint_semantic_sha256": protocol["parent"]["checkpoint_semantic_sha256"],
        },
        "raw_source": {
            "path": hybrid_record["path"],
            "upstream_manifest_path": hybrid_record["manifest_path"],
            "upstream_manifest_sha256": hybrid_record["manifest_sha256"],
            "prior_manifest_file_sha256": hybrid_record["prior_manifest_file_sha256"],
            "whole_extended_file_hash_recomputed": False,
            "feature_payload_rows_read": e0048.EXPECTED_FEATURE_ROWS,
            "feature_payload_bytes_read": int(sum(record["raw_hybrid_payload_bytes_read_in_this_partition"] for record in hybrid_source_audits)),
            "cursor_prefix_bytes_scanned_for_timestamp_mapping": int(sum(record["raw_hybrid_cursor_prefix_bytes_scanned_in_this_partition"] for record in hybrid_source_audits)),
            "source_file_identity": checkpoint_payload["source_file_identity"],
            "next_byte_offset_after_last_required_row": int(checkpoint_payload["source_byte_offset"]),
            "panel_repeat_payload_rows_read_again": int(panel["rows"]),
            "first_opened_utc": feature_meta["first_opened_utc"],
            "last_opened_utc": feature_meta["last_opened_utc"],
            "rows_after_frozen_last_opened_parsed": 0,
            "source_ready_receive_time_provenance": "UNKNOWN",
        },
        "flow_raw_sources_verified": flow_source_records,
        "feature_generation": feature_meta,
        "feature_checkpoint": {
            "path": str(feature_checkpoint_path.relative_to(ROOT)),
            "sha256": e0048.sha256_file(feature_checkpoint_path),
            "processed_feature_rows": int(checkpoint_payload["processed_feature_rows"]),
            "last_processed_opened_utc": checkpoint_payload["last_processed_opened_utc"],
            "fresh_process_resume": True,
            "partition_rows": e0048.PARTITION_ROWS,
            "deterministic_panel": panel,
            "no_model_load_or_predict": True,
        },
        "market_technical_sources": market_sources,
        "book_replay": book_manifest | {"known_gap_reconciliation": gap_reconciliation},
        "products": products,
        "output_hashes": output_hashes,
        "resources": {
            "workers": 1,
            "native_threads_max": 4,
            "wall_seconds": wall,
            "peak_rss_gib": max(worker_peak, book_peak) / 1024**3,
            "python_process_count_observed": max(resources_before["python_process_count_observed"], resource_after["python_process_count_observed"]),
            "resources_before": resources_before,
            "resources_after": resource_after,
        },
        "predictions_generated": 0,
        "signals_generated": 0,
        "ev_generated": 0,
        "fills_or_trades_simulated": 0,
        "pnl_or_quality_metrics_generated": 0,
        "outcome_target_resolution_payout_fields_projected": False,
        "evaluator_invoked": False,
        "evaluation_authorized": False,
        "activated": False,
        "holdout_status": "HOLDOUT_NOT_RUN",
        "e0041_manifest_replaced": False,
        "e0041_evaluator_changed": False,
        "t0009_status_changed": False,
        "e0037_status_changed": False,
        "e0043_gap_repaired": False,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    manifest["report_path"] = str(REPORT_PATH.relative_to(ROOT))
    manifest["report_sha256"] = None
    _write_report(manifest, summary, regime_summary)
    manifest["report_sha256"] = e0048.sha256_file(REPORT_PATH)
    _atomic_json(RUNTIME / "input_preparation_manifest.json", manifest)
    return manifest


def main() -> None:
    if os.environ.get(WORKER_ENV) == "1":
        _worker_loop()
        return
    manifest = _run_preparation()
    print(json.dumps({"status": manifest["status"], "feature_rows": manifest["feature_generation"]["rows"], "calendar_slots": manifest["products"]["holdout_input_rows"]["rows"], "t60_captures": manifest["book_replay"]["t60_captures"], "t59_captures": manifest["book_replay"]["t59_captures"], "runtime": str(RUNTIME.relative_to(ROOT)), "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
