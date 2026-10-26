"""T0009 source-only samples and development resolution-economics audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import run_btc_preopen_t59_reassessment as t59


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/research/btc_t0009_resolution_coverage_20261026.json"
RUN_MODE = "smoke"
OUTPUT_ROOT = ROOT / "data/analysis/polymarket/BTC"
GAMMA_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261007/gamma_markets.parquet"
PMXT_ROOT = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/pmxt_extended"
PMXT_PARTS = PMXT_ROOT / "event_parts"
OBADIAHA_ROOT = ROOT / "data/raw/polymarket/obadiaha/11793901f0ac89c5a6c51123a6ccd29a3aaf8f4c"
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
    else:
        raise RuntimeError("Full replay mode is not enabled before source-only smoke validation")
