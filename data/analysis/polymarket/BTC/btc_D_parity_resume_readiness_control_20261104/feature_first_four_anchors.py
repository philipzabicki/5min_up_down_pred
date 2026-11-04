from __future__ import annotations

import json
import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from e0037_trace_control import ROOT, process_measurements

sys.path.insert(0, str(ROOT))


CONTROL = Path(__file__).resolve().parent
RUN_DIR = CONTROL / "runs" / "features_first_four"
RUNTIME = RUN_DIR / "runtime"
FULL_FEATURE_PRODUCT = ROOT / "data/analysis/polymarket/BTC/btc_D_historical_inputs_20261030_v4/holdout_input_rows.parquet"
FIRST_ROW_BYTE_OFFSET = 310_671_871
ANCHORS = (
    ("first_holdout_market", pd.Timestamp("2026-06-30T23:58:00Z")),
    ("regime_edge_aug07", pd.Timestamp("2026-08-06T23:58:00Z")),
    ("regime_edge_aug14", pd.Timestamp("2026-08-13T23:58:00Z")),
    ("post_gap_and_resume_edge", pd.Timestamp("2026-08-18T06:58:00Z")),
)


def _resource_snapshot(prep) -> dict:
    return {
        "available_ram_bytes": int(prep.available_ram_bytes()),
        "D_free_bytes": int(shutil.disk_usage(ROOT).free),
    }


def _frozen_product_row(anchor: pd.Timestamp, feature_names: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(FULL_FEATURE_PRODUCT, format="parquet")
    timestamp_type = dataset.schema.field("feature_opened_utc").type
    value = anchor.to_pydatetime()
    if timestamp_type.tz is None:
        value = value.replace(tzinfo=None)
    opened_scalar = pa.scalar(value, type=timestamp_type)
    columns = ["feature_opened_utc", *feature_names]
    table = dataset.to_table(columns=columns, filter=ds.field("feature_opened_utc") == opened_scalar)
    return table.to_pandas()


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path, result_path, exit_path = RUN_DIR / "run.log", RUN_DIR / "run.json", RUN_DIR / "exit_code.txt"
    t0, cpu0 = time.perf_counter(), time.process_time()
    read0, _ = process_measurements()
    prep = None
    record = {"status": "RUNNING", "anchors": [anchor.isoformat().replace("+00:00", "Z") for _, anchor in ANCHORS]}
    try:
        from features import btc_D_historical_input_preparation_20261030 as prep
        from features.btc_D_continuation_20261029 import atomic_checkpoint, load_checkpoint, state_digest

        protocol, protocol_sha = prep.load_protocol()
        prep.RUNTIME = RUNTIME
        resource_before = _resource_snapshot(prep)
        if resource_before["available_ram_bytes"] < 16 * 1024**3:
            raise RuntimeError("feature control stopped: available physical RAM below 16 GiB safety floor")
        _, identity = prep.initialize_runtime(protocol, protocol_sha)
        checkpoint_path = RUNTIME / "checkpoints" / prep.CHECKPOINT_NAME
        checkpoint = load_checkpoint(checkpoint_path, identity_sha256=identity)
        if (checkpoint["source_row_index"] != 3_186_145
                or checkpoint["next_expected_opened_utc"] != "2026-06-30T23:58:00Z"
                or checkpoint["source_byte_offset"] is not None):
            raise RuntimeError("fresh E0047-derived source cursor does not match the frozen feature start")
        state_sha_before = state_digest(checkpoint["state"])
        checkpoint["source_byte_offset"] = FIRST_ROW_BYTE_OFFSET
        atomic_checkpoint(checkpoint_path, checkpoint)
        checkpoint = load_checkpoint(checkpoint_path, identity_sha256=identity)
        if state_digest(checkpoint["state"]) != state_sha_before:
            raise RuntimeError("reusing the saved byte cursor changed frozen feature state")

        generated_rows = []
        stage_records = []
        feature_names = None
        for index, (name, anchor) in enumerate(ANCHORS, start=1):
            current = pd.Timestamp(checkpoint["next_expected_opened_utc"])
            delta_minutes = (anchor - current) / pd.Timedelta(minutes=1)
            if delta_minutes < 0 or not float(delta_minutes).is_integer():
                raise RuntimeError("anchor does not align to the next frozen minute cursor")
            count = int(delta_minutes) + 1
            features, checkpoint = prep._materialize_feature_chunk(
                protocol, protocol_sha, identity, count,
                output_name=f"prefix_anchor_{index}", checkpoint_path=checkpoint_path,
            )
            if pd.Timestamp(features["Opened"].iloc[-1]) != anchor:
                raise RuntimeError(f"generated prefix did not stop at {name}")
            if feature_names is None:
                feature_names = [column for column in features.columns if column not in {"Opened", "flow_source_join_matched"}]
                if len(feature_names) != 102:
                    raise RuntimeError("the frozen generator did not produce exactly 102 ordered features")
            anchor_row = features.iloc[-1:][["Opened", *feature_names]].copy()
            generated_rows.append(anchor_row)
            stage_records.append({
                "anchor_name": name,
                "anchor_opened_utc": anchor.isoformat().replace("+00:00", "Z"),
                "rows_generated_in_stage": count,
                "prefix_end_utc": pd.Timestamp(features["Opened"].iloc[-1]).isoformat().replace("+00:00", "Z"),
                "raw_source_audit": json.loads((RUNTIME / "source_audit" / f"prefix_anchor_{index}.json").read_text(encoding="utf-8")),
            })

        if feature_names is None:
            raise RuntimeError("no generated feature rows")
        product_feature_order = [name for name in ds.dataset(FULL_FEATURE_PRODUCT, format="parquet").schema.names if name in set(feature_names)]
        if product_feature_order != feature_names:
            raise RuntimeError("saved E0048 feature product order differs from the frozen 102-feature generator order")

        anchor_comparisons = []
        for (name, anchor), generated in zip(ANCHORS, generated_rows, strict=True):
            observed = _frozen_product_row(anchor, feature_names)
            if len(observed) != 1:
                raise RuntimeError(f"saved E0048 product does not contain exactly one technical row for {name}")
            generated_values = generated.loc[:, feature_names].to_numpy(dtype=np.float64, copy=False)
            observed_values = observed.loc[:, feature_names].to_numpy(dtype=np.float64, copy=False)
            nan_masks_equal = np.array_equal(np.isnan(generated_values), np.isnan(observed_values))
            values_equal = np.array_equal(generated_values, observed_values, equal_nan=True)
            anchor_comparisons.append({
                "anchor_name": name,
                "feature_opened_utc": anchor.isoformat().replace("+00:00", "Z"),
                "generated_rows": len(generated),
                "saved_rows": len(observed),
                "ordered_feature_count": len(feature_names),
                "ordered_feature_names_sha256": hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest(),
                "exact_values_equal": bool(values_equal),
                "exact_nan_masks_equal": bool(nan_masks_equal),
            })

        combined = pd.concat(generated_rows, ignore_index=True)
        combined.to_parquet(RUN_DIR / "generated_anchor_rows.parquet", index=False, compression="zstd")
        record.update({
            "status": "PASS" if all(row["exact_values_equal"] and row["exact_nan_masks_equal"] for row in anchor_comparisons) else "FAIL",
            "protocol_sha256": protocol_sha,
            "feature_order": feature_names,
            "source_cursor": {
                "row_index": 3_186_145,
                "byte_offset": FIRST_ROW_BYTE_OFFSET,
                "source_cursor_audit_path": "data/analysis/polymarket/BTC/btc_D_historical_inputs_20261030_v4/source_audit/feature_part_000001.json",
                "feature_state_sha256_unchanged": True,
                "hybrid_cursor_prefix_bytes_scanned": prep._IO_AUDIT["hybrid_cursor_prefix_bytes_scanned"],
            },
            "stage_records": stage_records,
            "anchor_comparisons": anchor_comparisons,
            "features_generated": int(sum(item["rows_generated_in_stage"] for item in stage_records)),
            "io_audit": dict(prep._IO_AUDIT),
            "source_audit_paths": sorted(str(path.relative_to(ROOT)) for path in (RUNTIME / "source_audit").glob("prefix_anchor_*.json")),
            "resource_before": resource_before,
            "measurement": {
                "wall_seconds": time.perf_counter() - t0,
                "cpu_seconds": time.process_time() - cpu0,
                "process_read_transfer_bytes": process_measurements()[0] - read0,
                "peak_working_set_bytes": process_measurements()[1],
                "resource_after": _resource_snapshot(prep),
                "generated_anchor_product_bytes": (RUN_DIR / "generated_anchor_rows.parquet").stat().st_size,
                "runtime_bytes": sum(path.stat().st_size for path in RUNTIME.rglob("*" ) if path.is_file()),
            },
            "exit_code": 0 if all(row["exact_values_equal"] and row["exact_nan_masks_equal"] for row in anchor_comparisons) else 1,
        })
        result_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        log_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        code = int(record["exit_code"])
        exit_path.write_text(f"{code}\n", encoding="utf-8")
        if code:
            raise AssertionError("one or more first-four feature anchor comparisons failed")
        return 0
    except Exception as exc:
        record.update({
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "io_audit": dict(prep._IO_AUDIT) if prep is not None else {},
            "measurement": {
                "wall_seconds": time.perf_counter() - t0,
                "cpu_seconds": time.process_time() - cpu0,
                "process_read_transfer_bytes": process_measurements()[0] - read0,
                "peak_working_set_bytes": process_measurements()[1],
                "resource_after": _resource_snapshot(prep) if prep is not None else None,
            },
            "exit_code": 1,
        })
        result_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        log_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        exit_path.write_text("1\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
