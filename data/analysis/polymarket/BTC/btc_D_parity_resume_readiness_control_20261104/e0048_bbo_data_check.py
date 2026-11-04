from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pyarrow.parquet as pq

from e0037_trace_control import ROOT, process_measurements


SOURCE = ROOT / "data/analysis/polymarket/BTC/btc_D_historical_inputs_20261030_v4/slot_coverage.parquet"
OUTPUT = Path(__file__).with_name("e0048_bbo_data_check.json")
LOG = Path(__file__).with_name("e0048_bbo_data_check.log")
EXIT = Path(__file__).with_name("e0048_bbo_data_check.exit_code")
COLUMNS = (
    "market_start_utc", "condition_id", "t60_both_native_sides_valid",
    "t60_up_bbo_ask_comparable", "t60_up_bbo_ask_mismatch",
    "t60_up_best_ask", "t60_up_reported_best_ask",
    "t60_down_bbo_ask_comparable", "t60_down_bbo_ask_mismatch",
    "t60_down_best_ask", "t60_down_reported_best_ask",
)


def main() -> int:
    t0, cpu0 = time.perf_counter(), time.process_time()
    read0, _ = process_measurements()
    parquet = pq.ParquetFile(SOURCE)
    projected_bytes = sum(
        parquet.metadata.row_group(rg).column(column).total_compressed_size
        for rg in range(parquet.metadata.num_row_groups)
        for column in range(parquet.metadata.row_group(rg).num_columns)
        if parquet.metadata.row_group(rg).column(column).path_in_schema in COLUMNS
    )
    rows = 0
    valid_pairs = 0
    comparable_sides = 0
    stored_mismatch_sides = 0
    recomputed_mismatch_sides = 0
    conflicts = []
    for batch in parquet.iter_batches(columns=list(COLUMNS), batch_size=4096):
        for row in batch.to_pylist():
            rows += 1
            if row["t60_both_native_sides_valid"] is not True:
                continue
            valid_pairs += 1
            for side in ("up", "down"):
                if row[f"t60_{side}_bbo_ask_comparable"] is not True:
                    continue
                comparable_sides += 1
                stored = row[f"t60_{side}_bbo_ask_mismatch"] is True
                stored_mismatch_sides += int(stored)
                native = row[f"t60_{side}_best_ask"]
                reported = row[f"t60_{side}_reported_best_ask"]
                observed = (
                    native is None or reported is None
                    or not math.isclose(float(native), float(reported), rel_tol=0.0, abs_tol=1e-9)
                )
                recomputed_mismatch_sides += int(observed)
                if stored or observed:
                    conflicts.append({
                        "market_start_utc": str(row["market_start_utc"]),
                        "condition_id": row["condition_id"],
                        "side": side,
                        "native_best_ask": native,
                        "reported_best_ask": reported,
                        "stored_mismatch": stored,
                        "recomputed_mismatch": observed,
                    })
    read1, peak = process_measurements()
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "projected_columns": list(COLUMNS),
        "source_file_bytes": SOURCE.stat().st_size,
        "projected_compressed_column_chunk_bytes": projected_bytes,
        "rows_read": rows,
        "valid_t60_native_pairs": valid_pairs,
        "comparable_bbo_sides_in_valid_pairs": comparable_sides,
        "stored_bbo_mismatch_sides": stored_mismatch_sides,
        "recomputed_bbo_mismatch_sides_abs_tol_1e-9": recomputed_mismatch_sides,
        "conflicts": conflicts,
        "measurement": {
            "wall_seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "process_read_transfer_bytes": read1 - read0,
            "peak_working_set_bytes": peak,
        },
        "exit_code": 0,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    LOG.write_text(
        f"source={result['source']}\nrows_read={rows}\nvalid_t60_native_pairs={valid_pairs}\n"
        f"comparable_bbo_sides={comparable_sides}\nstored_mismatches={stored_mismatch_sides}\n"
        f"recomputed_mismatches={recomputed_mismatch_sides}\nwall_seconds={result['measurement']['wall_seconds']:.6f}\n"
        f"cpu_seconds={result['measurement']['cpu_seconds']:.6f}\n"
        f"projected_compressed_column_chunk_bytes={projected_bytes}\n"
        f"process_read_transfer_bytes={result['measurement']['process_read_transfer_bytes']}\n"
        f"peak_working_set_bytes={peak}\nexit_code=0\n",
        encoding="utf-8",
    )
    EXIT.write_text("0\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOG.write_text(f"{type(exc).__name__}: {exc}\nexit_code=1\n", encoding="utf-8")
        EXIT.write_text("1\n", encoding="utf-8")
        raise
