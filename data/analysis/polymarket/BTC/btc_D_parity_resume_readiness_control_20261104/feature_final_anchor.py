from __future__ import annotations

import json
import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from e0037_trace_control import ROOT, process_measurements

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_first_four_anchors import FULL_FEATURE_PRODUCT, _frozen_product_row


CONTROL = Path(__file__).resolve().parent
RUN_DIR = CONTROL / "runs" / "feature_final_anchor"
RUNTIME = RUN_DIR / "runtime"
SOURCE_CHECKPOINT = ROOT / "data/analysis/polymarket/BTC/btc_D_historical_inputs_20261030_v4/checkpoints/feature_part_000002_state.json"
ANCHOR = pd.Timestamp("2026-10-05T11:58:00Z")


def _resource_snapshot(prep) -> dict:
    return {
        "available_ram_bytes": int(prep.available_ram_bytes()),
        "D_free_bytes": int(shutil.disk_usage(ROOT).free),
    }


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path, result_path, exit_path = RUN_DIR / "run.log", RUN_DIR / "run.json", RUN_DIR / "exit_code.txt"
    t0, cpu0 = time.perf_counter(), time.process_time()
    read0, _ = process_measurements()
    prep = None
    record = {"status": "RUNNING", "anchor_opened_utc": ANCHOR.isoformat().replace("+00:00", "Z")}
    try:
        from features import btc_D_historical_input_preparation_20261030 as prep
        from features.btc_D_continuation_20261029 import load_checkpoint, state_digest

        protocol, protocol_sha = prep.load_protocol()
        prep.RUNTIME = RUNTIME
        resource_before = _resource_snapshot(prep)
        if resource_before["available_ram_bytes"] < 16 * 1024**3:
            raise RuntimeError("final feature control stopped: available physical RAM below 16 GiB safety floor")
        header = json.loads(SOURCE_CHECKPOINT.read_text(encoding="utf-8"))
        checkpoint = load_checkpoint(SOURCE_CHECKPOINT, identity_sha256=header["identity_sha256"])
        if checkpoint.get("protocol_sha256") != protocol_sha:
            raise RuntimeError("saved E0048 part-2 feature checkpoint uses a different frozen protocol")
        checkpoint_copy = RUNTIME / "checkpoints" / prep.CHECKPOINT_NAME
        checkpoint_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_CHECKPOINT, checkpoint_copy)
        start_opened = pd.Timestamp(checkpoint["next_expected_opened_utc"])
        delta = (ANCHOR - start_opened) / pd.Timedelta(minutes=1)
        if delta < 0 or not float(delta).is_integer():
            raise RuntimeError("final technical anchor is before or off the saved feature cursor")
        count = int(delta) + 1
        feature_names = None
        features, _next_checkpoint = prep._materialize_feature_chunk(
            protocol, protocol_sha, header["identity_sha256"], count,
            output_name="prefix_final_anchor", checkpoint_path=checkpoint_copy,
        )
        if pd.Timestamp(features["Opened"].iloc[-1]) != ANCHOR:
            raise RuntimeError("final generated prefix did not stop at the preregistered anchor")
        feature_names = [column for column in features.columns if column not in {"Opened", "flow_source_join_matched"}]
        if len(feature_names) != 102:
            raise RuntimeError("final feature prefix did not emit exactly 102 ordered features")
        product_feature_order = [name for name in pq.ParquetFile(FULL_FEATURE_PRODUCT).schema_arrow.names if name in set(feature_names)]
        if product_feature_order != feature_names:
            raise RuntimeError("saved E0048 feature product order differs from the frozen 102-feature generator order")
        generated_row = features.iloc[-1:][["Opened", *feature_names]].copy()
        observed = _frozen_product_row(ANCHOR, feature_names)
        if len(observed) != 1:
            raise RuntimeError("frozen E0048 product does not contain exactly one row for the final anchor")
        generated_values = generated_row.loc[:, feature_names].to_numpy(dtype=np.float64, copy=False)
        observed_values = observed.loc[:, feature_names].to_numpy(dtype=np.float64, copy=False)
        nan_equal = np.array_equal(np.isnan(generated_values), np.isnan(observed_values))
        values_equal = np.array_equal(generated_values, observed_values, equal_nan=True)
        generated_row.to_parquet(RUN_DIR / "generated_final_anchor.parquet", index=False, compression="zstd")
        result = {
            "status": "PASS" if values_equal and nan_equal else "FAIL",
            "protocol_sha256": protocol_sha,
            "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
            "source_checkpoint_identity_sha256": header["identity_sha256"],
            "source_checkpoint_state_digest": state_digest(checkpoint["state"]),
            "source_row_index": checkpoint["source_row_index"],
            "source_byte_offset": checkpoint["source_byte_offset"],
            "start_opened_utc": start_opened.isoformat().replace("+00:00", "Z"),
            "anchor_opened_utc": ANCHOR.isoformat().replace("+00:00", "Z"),
            "rows_generated": count,
            "ordered_feature_count": len(feature_names),
            "ordered_feature_names_sha256": hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest(),
            "exact_values_equal": bool(values_equal),
            "exact_nan_masks_equal": bool(nan_equal),
            "raw_source_audit": json.loads((RUNTIME / "source_audit" / "prefix_final_anchor.json").read_text(encoding="utf-8")),
            "io_audit": dict(prep._IO_AUDIT),
            "resource_before": resource_before,
            "measurement": {
                "wall_seconds": time.perf_counter() - t0,
                "cpu_seconds": time.process_time() - cpu0,
                "process_read_transfer_bytes": process_measurements()[0] - read0,
                "peak_working_set_bytes": process_measurements()[1],
                "resource_after": _resource_snapshot(prep),
                "generated_product_bytes": (RUN_DIR / "generated_final_anchor.parquet").stat().st_size,
                "runtime_bytes": sum(path.stat().st_size for path in RUNTIME.rglob("*") if path.is_file()),
            },
            "exit_code": 0 if values_equal and nan_equal else 1,
        }
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        log_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        code = int(result["exit_code"])
        exit_path.write_text(f"{code}\n", encoding="utf-8")
        if code:
            raise AssertionError("final feature anchor differs exactly from the frozen E0048 product")
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
