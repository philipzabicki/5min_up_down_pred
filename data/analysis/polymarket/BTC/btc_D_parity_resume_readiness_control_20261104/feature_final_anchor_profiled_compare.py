from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from e0037_trace_control import ROOT, process_measurements

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_first_four_anchors import _frozen_product_row


CONTROL = Path(__file__).resolve().parent
RUN_DIR = CONTROL / "runs" / "feature_final_anchor_profiled"
GENERATED = RUN_DIR / "runtime/minute_features/prefix_final_anchor_profiled.parquet"
PREVIOUS = CONTROL / "runs/feature_final_anchor/generated_final_anchor.parquet"
FULL_FEATURE_PRODUCT = ROOT / "data/analysis/polymarket/BTC/btc_D_historical_inputs_20261030_v4/holdout_input_rows.parquet"
ANCHOR = pd.Timestamp("2026-10-05T11:58:00Z")


def _differences(left: pd.DataFrame, right: pd.DataFrame, names: list[str]) -> list[dict]:
    a = left.loc[:, names].to_numpy(dtype=np.float64, copy=False)[0]
    b = right.loc[:, names].to_numpy(dtype=np.float64, copy=False)[0]
    result = []
    for index, name in enumerate(names):
        if np.isnan(a[index]) and np.isnan(b[index]):
            continue
        if a[index] != b[index]:
            result.append({"feature": name, "left": float(a[index]), "right": float(b[index]), "absolute_difference": abs(float(a[index] - b[index]))})
    return result


def main() -> int:
    t0, cpu0 = time.perf_counter(), time.process_time()
    read0, _ = process_measurements()
    generated_frame = pq.read_table(GENERATED).to_pandas()
    names = [column for column in generated_frame.columns if column not in {"Opened", "flow_source_join_matched"}]
    if len(names) != 102:
        raise RuntimeError("profiled final generator output does not contain 102 features")
    generated = generated_frame.loc[pd.to_datetime(generated_frame["Opened"], utc=True).eq(ANCHOR), ["Opened", *names]]
    if len(generated) != 1:
        raise RuntimeError("profiled final generator output lacks its unique anchor")
    product_order = [name for name in pq.ParquetFile(FULL_FEATURE_PRODUCT).schema_arrow.names if name in set(names)]
    if product_order != names:
        raise RuntimeError("profiled generator feature order differs from E0048 product order")
    frozen = _frozen_product_row(ANCHOR, names)
    previous = pq.read_table(PREVIOUS).to_pandas()
    if len(frozen) != 1 or len(previous) != 1:
        raise RuntimeError("final anchor comparison inputs must each contain one row")
    diff_frozen = _differences(generated, frozen, names)
    diff_previous = _differences(generated, previous, names)
    nan_equal = np.array_equal(
        np.isnan(generated.loc[:, names].to_numpy(dtype=np.float64, copy=False)),
        np.isnan(frozen.loc[:, names].to_numpy(dtype=np.float64, copy=False)),
    )
    generated.to_parquet(RUN_DIR / "generated_anchor_row.parquet", index=False, compression="zstd")
    read1, peak = process_measurements()
    result = {
        "status": "PASS" if not diff_frozen and nan_equal and not diff_previous else "FAIL_EXACT_PARITY",
        "anchor_opened_utc": ANCHOR.isoformat().replace("+00:00", "Z"),
        "feature_count": len(names),
        "feature_order_matches": True,
        "exact_nan_masks_equal_to_frozen_product": bool(nan_equal),
        "different_features_vs_frozen_product": diff_frozen[:10],
        "different_features_vs_previous_generated_row": diff_previous[:10],
        "measurement": {
            "wall_seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "process_read_transfer_bytes": read1 - read0,
            "peak_working_set_bytes": peak,
            "generated_chunk_bytes": GENERATED.stat().st_size,
        },
        "exit_code": 0 if not diff_frozen and nan_equal and not diff_previous else 1,
    }
    (RUN_DIR / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (RUN_DIR / "comparison.log").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (RUN_DIR / "comparison.exit_code").write_text(f"{result['exit_code']}\n", encoding="utf-8")
    if result["exit_code"]:
        raise AssertionError("profiled final feature output differs exactly from its frozen comparison")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        run = Path(__file__).resolve().parent / "runs/feature_final_anchor_profiled"
        (run / "comparison.log").write_text(f"{type(exc).__name__}: {exc}\nexit_code=1\n", encoding="utf-8")
        (run / "comparison.exit_code").write_text("1\n", encoding="utf-8")
        raise
