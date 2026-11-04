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
RUN_DIR = CONTROL / "runs" / "feature_final_anchor"
GENERATED = RUN_DIR / "runtime/minute_features/prefix_final_anchor.parquet"
FULL_FEATURE_PRODUCT = ROOT / "data/analysis/polymarket/BTC/btc_D_historical_inputs_20261030_v4/holdout_input_rows.parquet"
ANCHOR = pd.Timestamp("2026-10-05T11:58:00Z")


def main() -> int:
    t0, cpu0 = time.perf_counter(), time.process_time()
    read0, _ = process_measurements()
    generated_frame = pq.read_table(GENERATED).to_pandas()
    feature_names = [column for column in generated_frame.columns if column not in {"Opened", "flow_source_join_matched"}]
    if len(feature_names) != 102:
        raise RuntimeError("saved generated final prefix does not contain exactly 102 features")
    generated = generated_frame.loc[pd.to_datetime(generated_frame["Opened"], utc=True).eq(ANCHOR), ["Opened", *feature_names]]
    if len(generated) != 1:
        raise RuntimeError("generated final prefix has no unique row at the requested anchor")
    product_order = [name for name in pq.ParquetFile(FULL_FEATURE_PRODUCT).schema_arrow.names if name in set(feature_names)]
    if product_order != feature_names:
        raise RuntimeError("saved E0048 product feature order differs from the frozen generator order")
    observed = _frozen_product_row(ANCHOR, feature_names)
    if len(observed) != 1:
        raise RuntimeError("frozen E0048 product does not contain exactly one row for the final anchor")
    left = generated.loc[:, feature_names].to_numpy(dtype=np.float64, copy=False)[0]
    right = observed.loc[:, feature_names].to_numpy(dtype=np.float64, copy=False)[0]
    nan_equal = np.array_equal(np.isnan(left), np.isnan(right))
    different = []
    for index, name in enumerate(feature_names):
        if np.isnan(left[index]) and np.isnan(right[index]):
            continue
        if left[index] != right[index]:
            different.append({
                "feature": name,
                "generated": float(left[index]),
                "frozen_product": float(right[index]),
                "absolute_difference": abs(float(left[index] - right[index])),
            })
    generated.iloc[-1:][["Opened", *feature_names]].to_parquet(RUN_DIR / "generated_final_anchor.parquet", index=False, compression="zstd")
    read1, peak = process_measurements()
    result = {
        "status": "PASS" if not different and nan_equal else "FAIL_EXACT_PARITY",
        "anchor_opened_utc": ANCHOR.isoformat().replace("+00:00", "Z"),
        "feature_count": len(feature_names),
        "feature_order_matches": True,
        "exact_nan_masks_equal": bool(nan_equal),
        "exact_values_equal": not different,
        "different_feature_count": len(different),
        "first_differences": different[:10],
        "measurement": {
            "wall_seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "process_read_transfer_bytes": read1 - read0,
            "peak_working_set_bytes": peak,
            "generated_prefix_bytes": GENERATED.stat().st_size,
            "generated_anchor_row_bytes": (RUN_DIR / "generated_final_anchor.parquet").stat().st_size,
        },
        "exit_code": 0 if not different and nan_equal else 1,
    }
    (RUN_DIR / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (RUN_DIR / "comparison.log").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (RUN_DIR / "comparison.exit_code").write_text(f"{result['exit_code']}\n", encoding="utf-8")
    if result["exit_code"]:
        raise AssertionError("final feature anchor has an exact-value difference from the frozen E0048 product")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        (RUN_DIR / "comparison.log").write_text(f"{type(exc).__name__}: {exc}\nexit_code=1\n", encoding="utf-8")
        (RUN_DIR / "comparison.exit_code").write_text("1\n", encoding="utf-8")
        raise
