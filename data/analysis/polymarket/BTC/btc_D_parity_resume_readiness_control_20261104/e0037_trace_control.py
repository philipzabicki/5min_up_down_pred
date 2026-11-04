from __future__ import annotations

import ctypes
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from ctypes import wintypes


ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/pmxt_extended/event_parts/2026-04-30T06.parquet"
OUTPUT = Path(__file__).with_name("e0037_trace.json")
LOG = Path(__file__).with_name("e0037_trace.log")
EXIT = Path(__file__).with_name("e0037_trace.exit_code")
FIELDS = (
    "timestamp_received", "timestamp", "market", "event_type", "asset_id",
    "bids", "asks", "price", "size", "side", "best_bid", "best_ask", "fee_rate_bps",
)


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_measurements() -> tuple[int, int]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessIoCounters.argtypes = [wintypes.HANDLE, ctypes.POINTER(IO_COUNTERS)]
    kernel32.GetProcessIoCounters.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MEMORY_COUNTERS_EX), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    counters = IO_COUNTERS()
    if not kernel32.GetProcessIoCounters(handle, ctypes.byref(counters)):
        raise ctypes.WinError(ctypes.get_last_error())
    memory = MEMORY_COUNTERS_EX()
    memory.cb = ctypes.sizeof(memory)
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.ReadTransferCount), int(memory.PeakWorkingSetSize)


def main() -> int:
    t0, cpu0 = time.perf_counter(), time.process_time()
    read0, _ = process_measurements()
    table = pq.ParquetFile(SOURCE).read_row_group(0, columns=list(FIELDS)).slice(112555, 615)
    rows = table.to_pylist()
    target_asset = str(rows[0]["asset_id"])
    events = [
        {"row_group": 0, "row_index": 112555 + offset, **row}
        for offset, row in enumerate(rows)
        if str(row["asset_id"]) == target_asset
    ]
    read1, peak = process_measurements()
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "row_group": 0,
        "range_inclusive": [112555, 113169],
        "source_file_bytes": SOURCE.stat().st_size,
        "projected_event_fields": list(FIELDS),
        "rows_projected": len(rows),
        "target_asset_id": target_asset,
        "matching_asset_rows": len(events),
        "events": events,
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
        f"source={result['source']}\nrows_projected={len(rows)}\n"
        f"matching_asset_rows={len(events)}\nwall_seconds={result['measurement']['wall_seconds']:.6f}\n"
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
