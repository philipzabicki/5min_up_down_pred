from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from e0037_trace_control import ROOT, process_measurements


OUTPUT = Path(__file__).with_name("resume_source_preflight.json")
LOG = Path(__file__).with_name("resume_source_preflight.log")
EXIT = Path(__file__).with_name("resume_source_preflight.exit_code")
EVENT_COLUMNS = {
    "timestamp_received", "timestamp", "market", "event_type", "asset_id", "bids",
    "asks", "price", "size", "side", "best_bid", "best_ask", "fee_rate_bps",
}


def main() -> int:
    t0, cpu0 = time.perf_counter(), time.process_time()
    read0, _ = process_measurements()
    protocol = json.loads((ROOT / "configs/research/btc_D_historical_input_preparation_20261030_v4.json").read_text(encoding="utf-8"))
    parts_dir = ROOT / protocol["source_allowlist"]["pmxt_native_event_parts"]["path"]
    start = pd.Timestamp("2026-10-04T11:00:00Z")
    hours = pd.date_range(start, start + pd.Timedelta(hours=24), freq="h")
    expected = [f"{hour.strftime('%Y-%m-%dT%H')}.parquet" for hour in hours]
    paths = [parts_dir / name for name in expected if (parts_dir / name).is_file()]
    missing = [name for name in expected if not (parts_dir / name).is_file()]
    metadata = [(path, pq.ParquetFile(path)) for path in paths]
    projected_bytes = sum(
        parquet.metadata.row_group(rg).column(column).total_compressed_size
        for _path, parquet in metadata
        for rg in range(parquet.metadata.num_row_groups)
        for column in range(parquet.metadata.row_group(rg).num_columns)
        if parquet.metadata.row_group(rg).column(column).path_in_schema in EVENT_COLUMNS
    )
    read1, peak = process_measurements()
    result = {
        "source_parts_directory": str(parts_dir.relative_to(ROOT)),
        "expected_hours_utc": [hour.isoformat() for hour in hours],
        "parts_found": len(paths),
        "missing_parts": missing,
        "full_part_file_bytes": sum(path.stat().st_size for path in paths),
        "source_rows_in_full_parts": sum(parquet.metadata.num_rows for _path, parquet in metadata),
        "projected_event_column_chunk_bytes": projected_bytes,
        "parts": [
            {"name": path.name, "file_bytes": path.stat().st_size, "rows": parquet.metadata.num_rows}
            for path, parquet in metadata
        ],
        "measurement": {
            "wall_seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "process_read_transfer_bytes": read1 - read0,
            "peak_working_set_bytes": peak,
        },
        "exit_code": 0,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    LOG.write_text(
        f"parts_found={len(paths)}\nmissing_parts={missing}\n"
        f"full_part_file_bytes={result['full_part_file_bytes']}\n"
        f"source_rows_in_full_parts={result['source_rows_in_full_parts']}\n"
        f"projected_event_column_chunk_bytes={projected_bytes}\n"
        f"wall_seconds={result['measurement']['wall_seconds']:.6f}\n"
        f"cpu_seconds={result['measurement']['cpu_seconds']:.6f}\n"
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
