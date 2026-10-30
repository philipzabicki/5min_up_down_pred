"""Source-only E0048 preparation of frozen E0041 D inputs."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import BinaryIO

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/research/btc_D_historical_input_preparation_20261030_v3.json"
RUNTIME = ROOT / "data/analysis/polymarket/BTC/btc_D_historical_inputs_20261030_v3"
RAW_COLUMNS = ("Opened", "Open", "High", "Low", "Close", "Volume", "UM_BTCUSDT_Open", "UM_BTCUSDT_High", "UM_BTCUSDT_Low", "UM_BTCUSDT_Close")
FLOW_RAW_COLUMNS = ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume")
FLOW_FEATURES = ("taker_Z_1m", "taker_V_1m", "taker_Z_5m", "taker_V_5m", "taker_Z_15m", "taker_V_15m")
FEATURE_START = pd.Timestamp("2026-06-30T23:58:00Z")
FEATURE_END = pd.Timestamp("2026-10-06T23:53:00Z")
MARKET_START = pd.Timestamp("2026-07-01T00:00:00Z")
MARKET_END = pd.Timestamp("2026-10-07T00:00:00Z")
EXPECTED_FEATURE_ROWS = 141_116
EXPECTED_MARKET_SLOTS = 28_224
PARTITION_ROWS = 65_536
BATCH_ROWS = 65_536
CHECKPOINT_NAME = "feature_state.json"
EVENT_COLUMNS = ("timestamp_received", "timestamp", "market", "event_type", "asset_id", "bids", "asks", "price", "size", "side", "best_bid", "best_ask", "fee_rate_bps")
FORBIDDEN_SOURCE_COLUMNS = frozenset({"outcome", "outcomes", "target", "target_polymarket_up", "resolved_at_utc", "resolution", "payout", "prediction", "probability", "ev", "pnl", "trade", "transaction_hash"})
_IO_AUDIT = {"hybrid_rows_read": 0, "hybrid_cursor_prefix_bytes_scanned": 0, "hybrid_raw_payload_bytes_read": 0, "flow_rows_read": 0, "flow_cursor_prefix_bytes_scanned": 0, "flow_cursor_prefix_lines_scanned": 0, "event_parts_read": 0, "event_projected_rows_read": 0, "event_rows_replayed": 0}
_SOURCE_AUDIT: dict[str, dict] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_hybrid_source_metadata(protocol: dict) -> dict[str, int]:
    record = protocol["source_allowlist"]["verified_hybrid_raw_input"]
    manifest = json.loads((ROOT / record["manifest_path"]).read_text(encoding="utf-8"))
    source = ROOT / record["path"]
    if (manifest.get("status") != "verified"
            or manifest.get("extended_file_path") != record["path"]
            or manifest.get("extended_file_sha256") != record["prior_manifest_file_sha256"]
            or manifest.get("extended_file_rows") != record["expected_extended_file_rows"]
            or manifest.get("last_opened_utc") != record["expected_last_opened_utc"]
            or manifest.get("cutoff_exclusive_utc") != "2026-10-07T00:00:00+00:00"):
        raise RuntimeError("pinned raw hybrid source manifest does not match E0048")
    stat = source.stat()
    if stat.st_size != record["expected_file_size_bytes"]:
        raise RuntimeError("pinned raw hybrid source file size differs from E0048")
    return {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def available_ram_bytes() -> int:
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(status.ullAvailPhys)
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        raise RuntimeError("E0048 cannot measure available system RAM")


def process_memory_bytes() -> dict[str, int]:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"working_set_bytes": int(counters.WorkingSetSize), "peak_working_set_bytes": int(counters.PeakWorkingSetSize)}
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.name != "nt" and __import__("sys").platform != "darwin":
        peak *= 1024
    return {"working_set_bytes": peak, "peak_working_set_bytes": peak}


def peak_working_set_bytes() -> int:
    return process_memory_bytes()["peak_working_set_bytes"]


def load_protocol() -> tuple[dict, str]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (protocol.get("experiment_id") != "E0048" or protocol.get("protocol_revision") != 3
            or protocol.get("status") != "FROZEN_BEFORE_REPLAY"):
        raise RuntimeError("E0048 protocol is not frozen before replay")
    previous = protocol["supersedes_protocol"]
    if sha256_file(ROOT / previous["path"]) != previous["sha256"]:
        raise RuntimeError("E0048 superseded preregistration hash changed")
    auth = protocol["authorization"]
    for key in ("source_payload_read_authorized", "frozen_feature_generation_authorized", "native_book_source_replay_authorized"):
        if auth.get(key) is not True:
            raise RuntimeError(f"E0048 source-only authorization is missing: {key}")
    for key in ("prediction_generation_authorized", "signal_or_ev_generation_authorized", "fill_or_trade_simulation_authorized", "pnl_or_quality_metrics_authorized", "outcome_label_resolution_or_payout_read_authorized", "evaluation_authorized", "activated", "training_selection_tuning_or_fit_authorized", "transform_pca_scaler_or_calibrator_fit_authorized", "live_shadow_or_orders_authorized", "paid_data_or_infrastructure_authorized"):
        if auth.get(key) is not False:
            raise RuntimeError(f"E0048 forbids this operation: {key}")
    if auth.get("holdout_status") != "HOLDOUT_NOT_RUN":
        raise RuntimeError("E0048 holdout status must remain HOLDOUT_NOT_RUN")
    attempt = protocol.get("previous_attempt", {})
    if (attempt.get("protocol_revision") != 2
            or attempt.get("status") != "FAILED_SOURCE_RANGE_GUARD"
            or attempt.get("protected_source_payload_rows_read") != 66_975
            or attempt.get("out_of_authorized_range_flow_records_csv_parsed") != 1_439
            or attempt.get("out_of_authorized_range_flow_records_skipped_by_loop") != 1_438
            or attempt.get("out_of_authorized_range_flow_records_time_guarded") != 1
            or attempt.get("out_of_range_record_persisted_or_used") is not False
            or attempt.get("outcomes_or_targets_read") is not False
            or attempt.get("evaluation_started") is not False):
        raise RuntimeError("E0048 v3 must retain the exact v2 source-range incident record")
    if protocol.get("runtime_artifact_root") != RUNTIME.relative_to(ROOT).as_posix():
        raise RuntimeError("E0048 runtime artifact root differs from the v3 isolated path")
    cursor_policy = protocol["source_allowlist"].get("flow_source_cursor_policy", {})
    if (cursor_policy.get("prefix_method") != "binary line-boundary scan"
            or cursor_policy.get("excluded_data_rows_csv_parsed") is not False
            or cursor_policy.get("excluded_data_row_fields_decoded_or_selected") is not False
            or cursor_policy.get("prefix_scan_bytes_and_lines_audited_separately") is not True):
        raise RuntimeError("E0048 FLOW source cursor policy must not parse excluded rows")
    for relative_path, expected_sha256 in protocol["implementation"]["source_sha256"].items():
        path = ROOT / relative_path
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise RuntimeError(f"E0048 frozen source code hash differs: {relative_path}")
    evaluator = protocol["evaluation_interface"]["evaluator_source"]
    if sha256_file(ROOT / evaluator["path"]) != evaluator["sha256"]:
        raise RuntimeError("pinned E0041 evaluator source hash changed")
    import pyarrow

    runtime = protocol["runtime"]
    observed_runtime = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
    }
    if observed_runtime != runtime["versions"]:
        raise RuntimeError(f"E0048 parser/runtime versions differ from the preregistration: {observed_runtime}")
    scope = protocol["scope"]
    if (scope["holdout_market_start_interval_utc"] != {"start_inclusive": "2026-07-01T00:00:00Z", "end_exclusive": "2026-10-07T00:00:00Z"}
            or int(scope["expected_market_slots"]) != EXPECTED_MARKET_SLOTS
            or scope["first_required_feature_opened_utc"] != FEATURE_START.isoformat().replace("+00:00", "Z")
            or scope["last_required_feature_opened_utc"] != FEATURE_END.isoformat().replace("+00:00", "Z")
            or int(scope["expected_contiguous_feature_minutes"]) != EXPECTED_FEATURE_ROWS):
        raise RuntimeError("E0048 frozen time range or expected row counts changed")
    parent = protocol["parent"]
    for record in (parent["bundle_manifest_path"], parent["continuation_manifest_path"], parent["checkpoint_path"], parent["checkpoint_identity_protocol_path"], parent["source_inventory_path"], parent["book_gap_source_record_path"]):
        path = ROOT / record
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned E0048 parent evidence: {record}")
    if sha256_file(ROOT / parent["bundle_manifest_path"]) != parent["bundle_manifest_sha256"]:
        raise RuntimeError("E0041 frozen bundle hash changed")
    if sha256_file(ROOT / parent["continuation_manifest_path"]) != parent["continuation_manifest_sha256"]:
        raise RuntimeError("E0047 frozen continuation manifest hash changed")
    if sha256_file(ROOT / parent["checkpoint_path"]) != parent["checkpoint_sha256"]:
        raise RuntimeError("E0047 continuation checkpoint hash changed")
    if sha256_file(ROOT / parent["checkpoint_identity_protocol_path"]) != parent["checkpoint_identity_protocol_sha256"]:
        raise RuntimeError("E0047 checkpoint creator protocol hash changed")
    if sha256_file(ROOT / parent["source_inventory_path"]) != parent["source_inventory_sha256"]:
        raise RuntimeError("E0042 source inventory hash changed")
    if sha256_file(ROOT / parent["book_gap_source_record_path"]) != parent["book_gap_source_record_sha256"]:
        raise RuntimeError("E0043 source record hash changed")
    flow_inventory = protocol["source_allowlist"]["flow_source_inventory"]
    if sha256_file(ROOT / flow_inventory["metadata_path"]) != flow_inventory["metadata_sha256"]:
        raise RuntimeError("pinned Binance FLOW source inventory hash changed")
    if sha256_file(ROOT / protocol["source_allowlist"]["verified_hybrid_raw_input"]["manifest_path"]) != protocol["source_allowlist"]["verified_hybrid_raw_input"]["manifest_sha256"]:
        raise RuntimeError("pinned raw hybrid source manifest hash changed")
    fee_rule = protocol["source_allowlist"]["historical_fee_rule"]
    if sha256_file(ROOT / fee_rule["path"]) != fee_rule["sha256"]:
        raise RuntimeError("frozen historical fee registry hash changed")
    verify_hybrid_source_metadata(protocol)
    digest = sha256_file(PROTOCOL_PATH)
    return protocol, digest


def minute_keys(start: pd.Timestamp = FEATURE_START, end: pd.Timestamp = FEATURE_END) -> pd.DatetimeIndex:
    keys = pd.date_range(start, end, freq="1min", tz="UTC")
    if start == FEATURE_START and end == FEATURE_END and len(keys) != EXPECTED_FEATURE_ROWS:
        raise RuntimeError("frozen required feature key count changed")
    return keys


def market_calendar() -> pd.DataFrame:
    starts = pd.date_range(MARKET_START, MARKET_END, freq="5min", inclusive="left", tz="UTC")
    if len(starts) != EXPECTED_MARKET_SLOTS:
        raise RuntimeError("frozen market calendar count changed")
    return pd.DataFrame({
        "market_start_utc": starts,
        "decision_at_utc": starts - pd.Timedelta(seconds=60),
        "feature_opened_utc": starts - pd.Timedelta(minutes=2),
        "t60_cutoff_utc": starts - pd.Timedelta(seconds=60),
        "t59_cutoff_utc": starts - pd.Timedelta(seconds=59),
    })


def safe_projection(path: Path, columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    if not set(columns).issubset(names):
        raise RuntimeError(f"safe technical projection is missing required columns in {path}")
    if set(columns) & FORBIDDEN_SOURCE_COLUMNS:
        raise RuntimeError("a forbidden outcome/evaluation field entered a technical projection")
    table = parquet.read(columns=list(columns))
    return table.to_pandas()


def merge_safe_market_indexes(gamma: pd.DataFrame, pmxt: pd.DataFrame, starts: pd.DatetimeIndex) -> pd.DataFrame:
    columns = ["condition_id", "market_slug", "market_start_utc", "up_token_id", "down_token_id"]
    records: dict[pd.Timestamp, tuple] = {}
    conflicts: set[pd.Timestamp] = set()
    for frame in (gamma, pmxt):
        projected = frame.loc[:, columns].copy()
        projected["market_start_utc"] = pd.to_datetime(projected["market_start_utc"], utc=True, errors="raise")
        for row in projected.itertuples(index=False, name=None):
            condition_id, slug, start, up, down = row
            start = pd.Timestamp(start)
            candidate = (None if pd.isna(condition_id) else str(condition_id).lower(), None if pd.isna(slug) else str(slug), start, None if pd.isna(up) else str(up), None if pd.isna(down) else str(down))
            prior = records.get(start)
            if prior is not None and prior != candidate:
                conflicts.add(start)
            else:
                records[start] = candidate
    wanted = pd.DatetimeIndex(starts).as_unit("ns")
    rows = []
    for start in wanted:
        candidate = records.get(pd.Timestamp(start))
        if candidate is None or pd.Timestamp(start) in conflicts:
            rows.append({"market_start_utc": start, "condition_id": None, "market_slug": None, "up_token_id": None, "down_token_id": None, "market_identity_status": "UNKNOWN" if candidate is None else "CONFLICT"})
        else:
            rows.append(dict(zip(columns, candidate)) | {"market_identity_status": "CONFIRMED" if all(candidate[i] for i in (0, 3, 4)) else "UNKNOWN"})
    return pd.DataFrame(rows)


def _valid_raw_frame(frame: pd.DataFrame, expected_opened: pd.DatetimeIndex) -> None:
    if tuple(frame.columns) != RAW_COLUMNS:
        raise RuntimeError("raw hybrid source columns/order differ from E0048 allowlist")
    opened = pd.DatetimeIndex(pd.to_datetime(frame["Opened"], utc=True, errors="raise")).as_unit("ns")
    if not opened.equals(pd.DatetimeIndex(expected_opened).as_unit("ns")):
        raise RuntimeError("raw hybrid source has a missing, duplicate, reordered, or out-of-range minute")
    numeric = frame.loc[:, RAW_COLUMNS[1:]].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(numeric).all():
        raise RuntimeError("approved raw hybrid feature rows contain missing/non-finite numeric fields")
    if (frame[["Open", "High", "Low", "Close", "UM_BTCUSDT_Open", "UM_BTCUSDT_High", "UM_BTCUSDT_Low", "UM_BTCUSDT_Close"]].to_numpy(dtype=np.float64) <= 0).any():
        raise RuntimeError("raw hybrid OHLC values must be positive")
    if (frame["Volume"].to_numpy(dtype=np.float64) < 0).any():
        raise RuntimeError("raw hybrid Volume must be non-negative")


def _read_hybrid_rows_at_cursor(source: Path, start_row: int, count: int, expected: pd.DatetimeIndex, start_byte_offset: int | None = None) -> tuple[pd.DataFrame, int, int, int]:
    """Map the E0047 row cursor once, then seek using E0048's saved byte cursor."""
    lines = []
    with Path(source).open("rb") as stream:
        header = stream.readline()
        try:
            parsed_header = next(csv.reader([header.decode("utf-8-sig")]))
        except (UnicodeDecodeError, StopIteration) as error:
            raise RuntimeError("raw hybrid CSV header cannot be decoded") from error
        if tuple(parsed_header) != RAW_COLUMNS:
            raise RuntimeError("raw hybrid CSV header differs from the frozen source schema")
        if start_byte_offset is None:
            for _ in range(start_row):
                if not stream.readline():
                    raise RuntimeError("raw hybrid source ended before the frozen E0047 row cursor")
        else:
            stream.seek(start_byte_offset)
        payload_start_offset = stream.tell()
        for _ in range(count):
            line = stream.readline()
            if not line:
                raise RuntimeError("raw hybrid source ended before the last required feature minute")
            try:
                values = next(csv.reader([line.decode("utf-8")]))
            except (UnicodeDecodeError, StopIteration) as error:
                raise RuntimeError("authorized raw hybrid row cannot be decoded") from error
            if len(values) != len(RAW_COLUMNS):
                raise RuntimeError("raw hybrid row width changed")
            lines.append(line)
        next_byte_offset = stream.tell()
    frame = pd.read_csv(io.BytesIO(header + b"".join(lines)), usecols=list(RAW_COLUMNS))
    frame["Opened"] = pd.to_datetime(frame["Opened"], utc=True, errors="raise")
    _valid_raw_frame(frame, expected)
    return frame, next_byte_offset, payload_start_offset - len(header) if start_byte_offset is None else 0, next_byte_offset - payload_start_offset


def read_hybrid_range(source: Path, start_row: int, count: int, start_byte_offset: int | None = None) -> tuple[pd.DataFrame, int, int, int]:
    """Read only authorized rows; the first call maps the row cursor by UTC time."""
    if start_row < 3_186_145 or count <= 0 or start_row + count > 3_186_145 + EXPECTED_FEATURE_ROWS:
        raise ValueError("hybrid source request is outside the E0048 authorized row interval")
    expected = minute_keys()[start_row - 3_186_145:start_row - 3_186_145 + count]
    result, next_byte_offset, cursor_scan_bytes, payload_bytes = _read_hybrid_rows_at_cursor(source, start_row, count, expected, start_byte_offset)
    _IO_AUDIT["hybrid_rows_read"] += len(result)
    _IO_AUDIT["hybrid_cursor_prefix_bytes_scanned"] += cursor_scan_bytes
    _IO_AUDIT["hybrid_raw_payload_bytes_read"] += payload_bytes
    return result, next_byte_offset, cursor_scan_bytes, payload_bytes


def _flow_source_path(day: pd.Timestamp, protocol: dict) -> tuple[Path, str]:
    root = ROOT / protocol["source_allowlist"]["binance_usd_m_btcusdt"]["local_cache_root"]
    date = day.strftime("%Y-%m-%d")
    month = day.strftime("%Y-%m")
    if day >= pd.Timestamp("2026-10-02", tz="UTC"):
        path = root / "days" / date / f"BTCUSDT-1m-{date}.zip"
        return path, "zip"
    if day >= pd.Timestamp("2026-10-01", tz="UTC"):
        path = root / "days" / date / f"BTCUSDT-1m-{date}.csv"
        return path, "csv"
    return root / "months" / month / f"BTCUSDT-1m-{month}.csv", "csv"


def _flow_source_period_start(day: pd.Timestamp) -> pd.Timestamp:
    """Return the first UTC minute represented by the selected daily or monthly archive."""
    if day < pd.Timestamp("2026-10-01", tz="UTC"):
        return pd.Timestamp(year=day.year, month=day.month, day=1, tz="UTC")
    return day.floor("D")


def _flow_source_period_end(day: pd.Timestamp) -> pd.Timestamp:
    period_start = _flow_source_period_start(day)
    if day < pd.Timestamp("2026-10-01", tz="UTC"):
        return period_start + pd.offsets.MonthBegin(1) - pd.Timedelta(minutes=1)
    return period_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)


def _flow_csv_reader_after_offset(stream: BinaryIO, path: Path, row_offset: int):
    """Skip archive prefixes by binary line boundary without parsing excluded data rows."""
    first_line = stream.readline()
    if not first_line:
        raise RuntimeError(f"USD-M source is empty: {path}")
    first_field = first_line.lstrip(b"\xef\xbb\xbf").split(b",", 1)[0].strip().lower()
    has_header = first_field in {b"open_time", b"open_time_ms"}
    prefix_bytes = len(first_line) if has_header or row_offset > 0 else 0
    prefix_lines = 1 if has_header or row_offset > 0 else 0
    remaining_rows = row_offset if has_header else max(0, row_offset - 1)
    first_fields = None
    if not has_header and row_offset == 0:
        first_fields = next(csv.reader([first_line.decode("utf-8-sig")]))
    for _ in range(remaining_rows):
        line = stream.readline()
        if not line:
            raise RuntimeError(f"USD-M source ended before the requested row in {path}")
        prefix_bytes += len(line)
        prefix_lines += 1
    text_stream = io.TextIOWrapper(stream, encoding="utf-8")
    reader = csv.reader(text_stream)
    if first_fields is not None:
        reader = chain((first_fields,), reader)
    return reader, text_stream, prefix_bytes, prefix_lines


def _verify_flow_source(path: Path, kind: str, day: pd.Timestamp, protocol: dict) -> tuple[BinaryIO, zipfile.ZipFile | None]:
    date = day.strftime("%Y-%m-%d")
    month = day.strftime("%Y-%m")
    records = protocol["source_allowlist"]["flow_source_inventory"]
    if kind == "csv":
        expected = records["csv_files_sha256"].get(month if day < pd.Timestamp("2026-10-01", tz="UTC") else date)
        actual = sha256_file(path)
        if expected is None or actual != expected:
            raise RuntimeError(f"pinned USD-M CSV source hash mismatch: {path}")
        _SOURCE_AUDIT[str(path.relative_to(ROOT))] = {"sha256": actual, "size_bytes": path.stat().st_size, "source_hash_type": "pinned extracted CSV hash"}
        return path.open("rb"), None
    expected = records["official_zip_sha256"].get(date)
    checksum_path = path.with_suffix(path.suffix + ".CHECKSUM")
    checksum_text = checksum_path.read_text(encoding="ascii").strip()
    if expected is None or expected not in checksum_text or sha256_file(path) != expected:
        raise RuntimeError(f"official USD-M archive checksum/hash mismatch: {path}")
    _SOURCE_AUDIT[str(path.relative_to(ROOT))] = {"sha256": expected, "size_bytes": path.stat().st_size, "official_checksum_text": checksum_text, "source_hash_type": "official Binance archive SHA-256"}
    archive = zipfile.ZipFile(path)
    names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if len(names) != 1:
        archive.close()
        raise RuntimeError(f"expected one CSV member in official Binance archive {path}")
    return archive.open(names[0], "r"), archive


def _validate_binance_minute_close(open_time_ms: int, close_time_ms: int) -> None:
    if int(close_time_ms) != int(open_time_ms) + 59_999:
        raise RuntimeError("USD-M source minute close_time differs from open_time + 59,999 ms")


def read_flow_range(start: pd.Timestamp, end_inclusive: pd.Timestamp, protocol: dict) -> pd.DataFrame:
    """Read exact USD-M raw FLOW rows, validating archive time indices and official hashes."""
    if start < FEATURE_START or end_inclusive > FEATURE_END or end_inclusive < start:
        raise ValueError("FLOW raw request is outside the E0048 feature interval")
    pieces = []
    day = start.floor("D")
    while day <= end_inclusive.floor("D"):
        source_period_start = _flow_source_period_start(day)
        source_period_end = _flow_source_period_end(day)
        segment_start = max(start, source_period_start)
        segment_end = min(end_inclusive, source_period_end)
        path, kind = _flow_source_path(day, protocol)
        binary_stream, archive = _verify_flow_source(path, kind, day, protocol)
        text_stream = None
        try:
            start_offset = int((segment_start - source_period_start) // pd.Timedelta(minutes=1))
            row_count = int((segment_end - segment_start) / pd.Timedelta(minutes=1)) + 1
            reader, text_stream, prefix_bytes, prefix_lines = _flow_csv_reader_after_offset(binary_stream, path, start_offset)
            _IO_AUDIT["flow_cursor_prefix_bytes_scanned"] += prefix_bytes
            _IO_AUDIT["flow_cursor_prefix_lines_scanned"] += prefix_lines
            for row_offset in range(row_count):
                try:
                    fields = next(reader)
                except StopIteration as error:
                    raise RuntimeError(f"USD-M source ended before {segment_end}") from error
                if len(fields) < 11:
                    raise RuntimeError(f"USD-M kline schema is incomplete in {path}")
                open_time_ms = int(fields[0])
                _validate_binance_minute_close(open_time_ms, int(fields[6]))
                opened = pd.Timestamp(open_time_ms, unit="ms", tz="UTC")
                expected_opened = segment_start + pd.Timedelta(minutes=row_offset)
                if opened != expected_opened:
                    raise RuntimeError(f"USD-M exact open_time mismatch: {opened} != {expected_opened}")
                pieces.append((opened, float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4]), float(fields[7]), float(fields[10])))
                _IO_AUDIT["flow_rows_read"] += 1
        finally:
            if text_stream is not None:
                text_stream.close()
            else:
                binary_stream.close()
            if archive is not None:
                archive.close()
        day = (source_period_end + pd.Timedelta(minutes=1)).floor("D")
    result = pd.DataFrame(pieces, columns=("Opened", *FLOW_RAW_COLUMNS))
    expected = pd.date_range(start, end_inclusive, freq="1min", tz="UTC")
    if not pd.DatetimeIndex(result["Opened"]).equals(expected):
        raise RuntimeError("USD-M FLOW source range contains missing, duplicate, or reordered minutes")
    values = result.loc[:, FLOW_RAW_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4:] < 0).any():
        raise RuntimeError("USD-M FLOW source contains invalid raw OHLC/quote-volume fields")
    if (values[:, 5] > values[:, 4] + np.maximum(1e-8, np.abs(values[:, 4]) * 1e-10)).any():
        raise RuntimeError("USD-M taker buy quote volume exceeds total quote volume")
    return result


def _build_base64_chunk(tail: pd.DataFrame, suffix: pd.DataFrame, base_order: list[str], e45_protocol: dict, work_dir: Path) -> pd.DataFrame:
    import create_modeling_dataset as dataset_builder
    from utils.data import load_modeling_dataset_settings, resolve_modeling_dataset_output_paths

    lineage = e45_protocol["allowed_inputs"]["base_feature_lineage"]
    metadata = json.loads((ROOT / lineage["feature_metadata"]["path"]).read_text(encoding="utf-8"))
    stage = json.loads((ROOT / lineage["indicator_stage"]["path"]).read_text(encoding="utf-8"))
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / "rolling_input.csv"
    rolling = pd.concat([tail, suffix], ignore_index=True) if len(tail) else suffix.copy()
    rolling.loc[:, RAW_COLUMNS].to_csv(raw_path, index=False)
    parsed = pd.read_csv(raw_path)
    if tuple(parsed.columns) != RAW_COLUMNS or len(parsed) != len(rolling):
        raise RuntimeError("BASE64 raw roundtrip changed schema or row count")
    if not pd.DatetimeIndex(pd.to_datetime(parsed["Opened"], utc=True, errors="raise")).equals(pd.DatetimeIndex(rolling["Opened"])):
        raise RuntimeError("BASE64 raw roundtrip changed exact Opened keys")
    for column in RAW_COLUMNS[1:]:
        left = rolling[column].to_numpy(dtype=np.float64, copy=False)
        right = parsed[column].to_numpy(dtype=np.float64, copy=False)
        if not np.array_equal(left, right, equal_nan=True):
            raise RuntimeError(f"BASE64 raw CSV roundtrip changed values in {column}")
    subset_path = work_dir / "selected_features.json"
    subset_path.write_text(json.dumps({"features": base_order[:64]}, separators=(",", ":")), encoding="utf-8")
    settings = load_modeling_dataset_settings(asset="BTC", dataset_profile_name="BTC", modeling_profile_name="BTC")
    settings.update({
        "active_asset": "BTC", "symbol": "BTCUSD", "interval": "1m",
        "raw_data_dir": raw_path.parent, "data_dir": raw_path.parent, "base_data_file": raw_path.name,
        "fit_results_dir": ROOT / stage["fit_results_dir"],
        "feature_subset_path": subset_path, "feature_subset_list_key": "features",
        "excluded_feature_names": (), "feature_intervals": metadata["feature_intervals"],
        "basis_premium_features": metadata["basis_premium_features"],
        "reaction_profile_fixed_grid": metadata["reaction_profile_fixed_grid"],
        "volume_profile_fixed_range": metadata["volume_profile_fixed_range"],
        "float_precision": "float64", "profile_state_dir": work_dir / "profile_states",
        "modeling_output_dir": work_dir / "feature_output", "output_suffix": "e0048_input",
        "preview_rows": 0,
    })
    settings.pop("preopen_target", None)
    output_path = resolve_modeling_dataset_output_paths(settings)["parquet"]
    built = dataset_builder.build_dataset_from_settings(settings, include_targets=False)
    if Path(built).resolve() != Path(output_path).resolve():
        raise RuntimeError("E0048 BASE64 builder wrote to an unexpected path")
    import pyarrow.parquet as pq

    result = pq.read_table(built, columns=["Opened", *base_order[:64]]).to_pandas()
    result["Opened"] = pd.to_datetime(result["Opened"], utc=True, errors="raise")
    if len(result) != len(rolling) or not pd.DatetimeIndex(result["Opened"]).equals(pd.DatetimeIndex(rolling["Opened"])):
        raise RuntimeError("E0048 BASE64 builder changed or reordered raw Opened keys")
    return result.iloc[len(tail):].reset_index(drop=True)


def _feature_identity(protocol_sha256: str, bundle_sha256: str, checkpoint_semantic_sha256: str) -> str:
    return canonical_sha256({"protocol_sha256": protocol_sha256, "bundle_sha256": bundle_sha256, "checkpoint_semantic_sha256": checkpoint_semantic_sha256})


def _compose_feature_rows(opened: pd.DatetimeIndex, base: pd.DataFrame, dynamic: dict, bundle_manifest: dict, flow_matched: np.ndarray) -> pd.DataFrame:
    base_order = list(bundle_manifest["base_feature_order"])
    memory_order = list(bundle_manifest["memory_feature_order"])
    esn_order = bundle_manifest["esn_feature_order_by_seed"]
    seeds = list(bundle_manifest["reservoir_seeds"])
    if base_order[-len(FLOW_FEATURES):] != list(FLOW_FEATURES):
        raise RuntimeError("frozen E0041 BASE70 order no longer ends with its six FLOW columns")
    if len(opened) != len(base) or len(opened) != len(flow_matched):
        raise RuntimeError("E0048 feature blocks do not share one exact row count")
    rows = {"Opened": pd.DatetimeIndex(opened)}
    for name in base_order:
        rows[name] = base[name].to_numpy(copy=False)
    for name in memory_order:
        rows[name] = dynamic["memory"][:, memory_order.index(name)]
    seed_by_key = {str(seed): f"esn_{seed}" for seed in seeds}
    for seed in seeds:
        names = esn_order[str(seed)]
        for index, name in enumerate(names):
            rows[name] = dynamic[seed_by_key[str(seed)]][:, index]
    ordered_names = ["Opened", *base_order, *memory_order]
    for seed in seeds:
        ordered_names.extend(esn_order[str(seed)])
    if len(ordered_names) != 103 or len(set(ordered_names)) != 103:
        raise RuntimeError("E0041 unique feature order is not 102 features plus Opened")
    features = pd.DataFrame(rows).loc[:, ordered_names]
    if any(features[name].dtype != np.dtype("float32") for name in FLOW_FEATURES):
        raise RuntimeError("E0041 historical FLOW join/cast must preserve all six float32 columns")
    features["flow_source_join_matched"] = np.asarray(flow_matched, dtype=bool)
    return features


def restore_feature_state(protocol: dict, protocol_sha256: str):
    import run_btc_D_continuation_parity_closure_20261029 as e47
    from features.btc_D_continuation_20261029 import load_checkpoint, state_digest
    from features.btc_D_input_generator_20261029 import restore_dynamic_state

    e47_protocol, _e45, bundle, transforms = e47.load_protocol()
    parent = protocol["parent"]
    checkpoint_path = ROOT / parent["checkpoint_path"]
    checkpoint_identity_protocol = json.loads((ROOT / parent["checkpoint_identity_protocol_path"]).read_text(encoding="utf-8"))
    checkpoint_identity = e47._continuation_identity(checkpoint_identity_protocol, bundle["identity"])
    state = load_checkpoint(checkpoint_path, identity_sha256=checkpoint_identity)
    if state_digest(state) != parent["checkpoint_semantic_sha256"]:
        raise RuntimeError("E0047 feature checkpoint semantic digest changed")
    if (state.get("source_sha256") != e47_protocol["scope"]["source_raw_sha256"]
            or state.get("last_processed_opened") != parent["checkpoint_last_opened_utc"].replace("Z", "+00:00")
            or state.get("next_expected_opened") != parent["checkpoint_next_opened_utc"].replace("Z", "+00:00")):
        raise RuntimeError("E0047 checkpoint source or time boundary differs from its pinned manifest")
    dynamic_state = restore_dynamic_state(state["dynamic_state"], transforms, temporary_root=RUNTIME / "restore_parts")
    identity = _feature_identity(protocol_sha256, protocol["parent"]["bundle_manifest_sha256"], protocol["parent"]["checkpoint_semantic_sha256"])
    return e47, bundle, transforms, state, dynamic_state, identity


def initialize_runtime(protocol: dict, protocol_sha256: str) -> tuple[dict, str]:
    e47, bundle, transforms, state, dynamic_state, identity = restore_feature_state(protocol, protocol_sha256)
    import features.btc_D_input_generator_20261029 as e45

    state["dynamic_state"] = e45._dynamic_checkpoint_payload(dynamic_state, temporary_root=RUNTIME / "checkpoint_parts")
    source_file_identity = verify_hybrid_source_metadata(protocol)
    state["source_byte_offset"] = None
    payload = {
        "schema_version": 1,
        "identity_sha256": identity,
        "protocol_sha256": protocol_sha256,
        "source_row_index": 3_186_145,
        "source_byte_offset": None,
        "source_file_identity": source_file_identity,
        "processed_feature_rows": 0,
        "last_processed_opened_utc": "2026-06-30T23:57:00Z",
        "next_expected_opened_utc": FEATURE_START.isoformat().replace("+00:00", "Z"),
        "state": state,
    }
    from features.btc_D_continuation_20261029 import atomic_checkpoint

    atomic_checkpoint(RUNTIME / "checkpoints" / CHECKPOINT_NAME, payload)
    return payload, identity


def _load_runtime_state(protocol: dict, protocol_sha256: str, identity: str, checkpoint_path: Path | None = None):
    from features.btc_D_continuation_20261029 import load_checkpoint
    import run_btc_D_continuation_parity_closure_20261029 as e47
    import features.btc_D_input_generator_20261029 as e45

    payload = load_checkpoint(checkpoint_path or (RUNTIME / "checkpoints" / CHECKPOINT_NAME), identity_sha256=identity)
    if payload.get("protocol_sha256") != protocol_sha256 or payload.get("schema_version") != 1:
        raise RuntimeError("E0048 feature checkpoint identity/schema mismatch")
    state = payload["state"]
    dynamic_state = e45.restore_dynamic_state(state["dynamic_state"], restore_feature_state(protocol, protocol_sha256)[2], temporary_root=RUNTIME / "restore_parts")
    return payload, state, dynamic_state


def _materialize_feature_chunk(protocol: dict, protocol_sha256: str, identity: str, count: int, *, output_name: str, checkpoint_path: Path | None = None, output_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    import run_btc_D_continuation_parity_closure_20261029 as e47
    from features import btc_D_continuation_20261029 as continuation
    from features.btc_D_input_generator_20261029 import generate_dynamic_block, _dynamic_checkpoint_payload

    checkpoint_path = checkpoint_path or (RUNTIME / "checkpoints" / CHECKPOINT_NAME)
    payload, state, dynamic_state = _load_runtime_state(protocol, protocol_sha256, identity, checkpoint_path)
    row_index = int(payload["source_row_index"])
    processed = int(payload["processed_feature_rows"])
    if processed + count > EXPECTED_FEATURE_ROWS:
        raise ValueError("feature chunk exceeds the frozen E0048 end boundary")
    hybrid_path = ROOT / protocol["source_allowlist"]["verified_hybrid_raw_input"]["path"]
    observed_source_identity = verify_hybrid_source_metadata(protocol)
    if payload.get("source_file_identity") != observed_source_identity:
        raise RuntimeError("pinned E0048 raw hybrid source file changed since the row cursor was mapped")
    suffix, next_source_byte_offset, cursor_scan_bytes, hybrid_payload_bytes = read_hybrid_range(hybrid_path, row_index, count, payload.get("source_byte_offset"))
    e47_protocol, e45_protocol, bundle, transforms = e47.load_protocol()
    base_order = bundle["manifest"]["base_feature_order"]
    raw_tail = state["raw_tail"]
    rolling_raw = pd.concat([raw_tail, suffix], ignore_index=True)
    if len(raw_tail) != continuation.TAIL_ROWS or not rolling_raw["Opened"].is_monotonic_increasing or rolling_raw["Opened"].duplicated().any():
        raise RuntimeError("E0048 BASE64 tail continuity invariant failed")
    base = _build_base64_chunk(raw_tail, suffix, base_order, e45_protocol, RUNTIME / "builder" / output_name)
    streak = continuation.advance_streak_states(state["streak_state"], suffix)
    profile = e47._profile_features_for_rows(state["profile_states"], suffix)
    for name, values in {**streak, **profile}.items():
        if name in base:
            base[name] = values
    recursive = e47._recursive_indicator_overrides(base, rolling_raw, len(suffix), state, state["indicator_params"])

    flow_start = pd.Timestamp(payload["next_expected_opened_utc"])
    flow_prefix_bytes_before = _IO_AUDIT["flow_cursor_prefix_bytes_scanned"]
    flow_prefix_lines_before = _IO_AUDIT["flow_cursor_prefix_lines_scanned"]
    flow_suffix = read_flow_range(flow_start, pd.Timestamp(suffix["Opened"].iloc[-1]), protocol)
    flow_rolling = pd.concat([state["flow_raw_tail"], flow_suffix], ignore_index=True)
    flow_computed = e47._compute_flow_block(flow_rolling)
    flow_suffix_features = flow_computed.iloc[-len(flow_suffix):].reset_index(drop=True)
    flow_cast, flow_matched = continuation.historical_flow_join_cast(
        pd.DatetimeIndex(suffix["Opened"]), pd.DatetimeIndex(flow_suffix_features["Opened"]),
        flow_suffix_features.loc[:, FLOW_FEATURES].to_numpy(dtype=np.float64, copy=False),
    )
    if flow_cast.shape != (len(suffix), len(FLOW_FEATURES)):
        raise RuntimeError("historical FLOW join returned an unexpected shape")
    atomic_json(RUNTIME / "source_audit" / f"{output_name}.json", {
        "flow_sources_verified": _SOURCE_AUDIT,
        "flow_rows_read_in_this_partition": int(len(flow_suffix)),
        "flow_cursor_prefix_bytes_scanned_in_this_partition": int(_IO_AUDIT["flow_cursor_prefix_bytes_scanned"] - flow_prefix_bytes_before),
        "flow_cursor_prefix_lines_scanned_in_this_partition": int(_IO_AUDIT["flow_cursor_prefix_lines_scanned"] - flow_prefix_lines_before),
        "opened_start_utc": flow_start.isoformat().replace("+00:00", "Z"),
        "opened_end_utc": suffix["Opened"].iloc[-1].isoformat().replace("+00:00", "Z"),
        "raw_hybrid_source_path": "data/datasets/raw/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007.csv",
        "raw_hybrid_rows_read_in_this_partition": int(len(suffix)),
        "raw_hybrid_cursor_prefix_bytes_scanned_in_this_partition": cursor_scan_bytes,
        "raw_hybrid_payload_bytes_read_in_this_partition": hybrid_payload_bytes,
        "raw_hybrid_next_byte_offset": next_source_byte_offset,
        "raw_hybrid_whole_file_hash_recomputed": False,
    })
    for index, name in enumerate(FLOW_FEATURES):
        base[base_order[64 + index]] = flow_cast[:, index]

    dynamic_rolling = pd.concat([state["dynamic_tail"], suffix], ignore_index=True)
    dynamic = generate_dynamic_block(dynamic_state, dynamic_rolling, new_rows_start=len(dynamic_rolling) - len(suffix))
    features = _compose_feature_rows(
        pd.DatetimeIndex(suffix["Opened"]), base, dynamic, bundle["manifest"], flow_matched,
    )

    state["raw_tail"] = rolling_raw.tail(continuation.TAIL_ROWS).reset_index(drop=True)
    state["flow_raw_tail"] = flow_rolling.tail(e47.FLOW_TAIL_ROWS).reset_index(drop=True)
    state["dynamic_tail"] = dynamic_rolling.tail(720).reset_index(drop=True)
    state["dynamic_state"] = _dynamic_checkpoint_payload(dynamic_state, temporary_root=RUNTIME / "checkpoint_parts")
    state["recursive_last_output"] = {key: np.asarray(value)[-1].item() for key, value in recursive.items()}
    state["source_row_index"] = row_index + len(suffix)
    state["source_byte_offset"] = next_source_byte_offset
    state["last_processed_opened"] = suffix["Opened"].iloc[-1].isoformat()
    state["next_expected_opened"] = (suffix["Opened"].iloc[-1] + pd.Timedelta(minutes=1)).isoformat()
    next_payload = {
        "schema_version": 1,
        "identity_sha256": identity,
        "protocol_sha256": protocol_sha256,
        "source_row_index": row_index + len(suffix),
        "source_byte_offset": next_source_byte_offset,
        "source_file_identity": observed_source_identity,
        "processed_feature_rows": processed + len(suffix),
        "last_processed_opened_utc": suffix["Opened"].iloc[-1].isoformat().replace("+00:00", "Z"),
        "next_expected_opened_utc": (suffix["Opened"].iloc[-1] + pd.Timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        "state": state,
    }
    part_path = output_path or (RUNTIME / "minute_features" / f"{output_name}.parquet")
    part_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(part_path, index=False, compression="zstd")
    continuation.atomic_checkpoint(checkpoint_path, next_payload)
    boundary_path = RUNTIME / "checkpoints" / f"{output_name}_state.json"
    continuation.atomic_checkpoint(boundary_path, next_payload)
    return features, next_payload


def _calendar_mapping_inputs(protocol: dict) -> tuple[pd.DataFrame, list[dict]]:
    safe_columns = tuple(protocol["source_allowlist"]["gamma_market_metadata"]["projected_fields"])
    gamma_path = ROOT / protocol["source_allowlist"]["gamma_market_metadata"]["path"]
    pmxt_cfg = protocol["source_allowlist"]["pmxt_market_metadata"]
    pmxt_path = ROOT / pmxt_cfg["path"]
    gamma = safe_projection(gamma_path, safe_columns)
    pmxt = safe_projection(pmxt_path, tuple(pmxt_cfg["projected_fields"]))
    calendar = market_calendar()
    identities = merge_safe_market_indexes(gamma, pmxt, pd.DatetimeIndex(calendar["market_start_utc"]))
    projection_records = []
    for path, frame in ((gamma_path, gamma), (pmxt_path, pmxt)):
        projection_records.append({
            "path": str(path.relative_to(ROOT)),
            "projection_sha256": canonical_sha256({"columns": list(frame.columns), "rows": frame.astype("string").fillna("<NA>").to_dict(orient="records")}),
            "rows_projected": int(len(frame)),
            "columns_projected": list(frame.columns),
            "whole_mixed_file_hash_read": False,
        })
    return calendar.merge(identities, on="market_start_utc", how="left", validate="one_to_one"), projection_records


def _event_hour(path: Path) -> pd.Timestamp:
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})T(\d{2})\.parquet", path.name)
    if not match:
        raise RuntimeError(f"unexpected PMXT event partition name: {path.name}")
    return pd.Timestamp(f"{match.group(1)}T{match.group(2)}:00:00Z")


def _archive_coverage(start: pd.Timestamp, entry: pd.Timestamp, processed_hours: set[pd.Timestamp]) -> dict:
    lower = start.floor("h") - pd.Timedelta(hours=25)
    expected = list(pd.date_range(lower, entry.floor("h"), freq="h"))
    recent = pd.date_range((entry - pd.Timedelta(seconds=30)).floor("h"), entry.floor("h"), freq="h")
    return {
        "archive_entry_hour_available": entry.floor("h") in processed_hours,
        "archive_freshness_window_available": all(hour in processed_hours for hour in recent),
        "archive_window_hours_expected": len(expected),
        "archive_window_hours_source_available": sum(hour in processed_hours for hour in expected),
        "archive_window_hours_processed": sum(hour in processed_hours for hour in expected),
        "archive_window_missing_source_hours": sum(hour not in processed_hours for hour in expected),
        "archive_window_unprocessed_hours": 0,
    }


def _gap_after(event_ns, entry: pd.Timestamp, processed_hours: set[pd.Timestamp]):
    if event_ns is None:
        return None
    event_hour = pd.Timestamp(int(event_ns), unit="ns", tz="UTC").floor("h")
    entry_hour = entry.floor("h")
    return event_hour > entry_hour or any(hour not in processed_hours for hour in pd.date_range(event_hour, entry_hour, freq="h"))


def _direct_book_snapshot(state: dict, market: dict, entry: pd.Timestamp, case: str, processed_hours: set[pd.Timestamp]) -> dict:
    import run_btc_preopen_economic_replay as replay
    import run_btc_preopen_t59_reassessment as t59

    entry_ns = int(entry.value)
    result = {
        "condition_id": market["condition_id"], "market_start_utc": market["market_start_utc"],
        "entry_time_utc": entry, "cutoff": case, "fee_rate_bps": None,
        "fee_collection_mode": replay._fee_collection_mode(entry),
        "fee_event_ns": state["fee_event_ns"], "fee_source_ns": state["fee_source_ns"],
        "fee_known": state["fee_rate_bps"] is not None and state["fee_event_ns"] is not None and state["fee_event_ns"] <= entry_ns,
        "no_future_event_at_entry": (state["last_event_ns"] is None or state["last_event_ns"] <= entry_ns) and not state.get("unknown_event_timing", False),
        "no_future_source_event_at_entry": all(token["max_source_ns_seen"] is None or token["max_source_ns_seen"] <= entry_ns for token in state["tokens"].values()) and (state["fee_source_ns"] is None or state["fee_source_ns"] <= entry_ns) and not state.get("unknown_event_timing", False),
        **_archive_coverage(pd.Timestamp(market["market_start_utc"]), entry, processed_hours),
    }
    if result["fee_known"]:
        result["fee_rate_bps"] = state["fee_rate_bps"]
    for side in ("up", "down"):
        token_id = str(market[f"{side}_token_id"])
        token = state["tokens"].get(token_id)
        initialized = bool(token and token["initialized"])
        bids = token["bids"] if initialized else {}
        asks = token["asks"] if initialized else {}
        best_bid = max(bids) if bids else None
        best_ask = min(asks) if asks else None
        reported_source_ns = token["reported_source_ns"] if token else None
        ask_comparable = bool(reported_source_ns is not None and token and token["ask_last_source_ns"] is not None and token["ask_last_source_ns"] <= reported_source_ns)
        ask_mismatch = bool(ask_comparable and (best_ask is None or abs(float(best_ask) - float(token["reported_best_ask"])) > 1e-6))
        init_ns = token["book_receive_ns"] if initialized else None
        age = (entry_ns - token["ask_book_source_ns"]) / 1e9 if token and token["ask_book_source_ns"] is not None else None
        gap = _gap_after(init_ns, entry, processed_hours)
        result.update({
            f"{side}_quote_source_token_id": token_id if initialized else None,
            f"{side}_quote_complemented": False,
            f"{side}_best_bid": best_bid, f"{side}_best_ask": best_ask,
            f"{side}_best_ask_size_shares": asks.get(best_ask) if best_ask is not None else None,
            f"{side}_ask_levels": sorted((float(price), float(size)) for price, size in asks.items()),
            f"{side}_ask_order_ambiguous": bool(token.get("ask_order_ambiguous", False)) if token else False,
            f"{side}_bbo_ask_comparable": ask_comparable,
            f"{side}_bbo_ask_mismatch": ask_mismatch,
            f"{side}_bid_ask_crossed": bool(best_bid is not None and best_ask is not None and best_bid > best_ask),
            f"{side}_ask_age_seconds": age,
            f"{side}_book_init_receive_ns": init_ns,
            f"archive_gap_after_{side}_book_init": gap,
            f"{side}_reported_best_bid": token.get("reported_best_bid") if token else None,
            f"{side}_reported_best_ask": token.get("reported_best_ask") if token else None,
        })
    result["archive_gap_after_fee_update"] = _gap_after(result["fee_event_ns"], entry, processed_hours)
    reason_input = dict(result)
    reason_input["archive_gap_after_fee_update"] = result["archive_gap_after_fee_update"]
    for side in ("up", "down"):
        reason_input[f"{side}_reason"] = "ask_valid_within_source_coverage"
        reason_input[f"{side}_quote_complemented"] = False
        reason_input[f"{side}_bbo_ask_comparable"] = result[f"{side}_bbo_ask_comparable"]
        reason_input[f"{side}_bbo_ask_mismatch"] = result[f"{side}_bbo_ask_mismatch"]
    for side in ("up", "down"):
        result[f"{side}_reason"] = t59._ask_base_reason(reason_input, side, str(market[f"{side}_token_id"]))
    return result


def _merge_source_quality(market_rows: pd.DataFrame, t60: pd.DataFrame, t59: pd.DataFrame) -> pd.DataFrame:
    from utils.btc_t0009_source_audit import source_quality_mask

    snapshots = t60.merge(t59, on=["market_start_utc", "condition_id"], how="outer", validate="one_to_one")
    identity = market_rows.loc[:, ["market_start_utc", "condition_id", "up_token_id", "down_token_id"]]
    source_quality = identity.merge(snapshots, on=["market_start_utc", "condition_id"], how="left", validate="one_to_one")
    source_quality["source_quality_clean"] = source_quality_mask(source_quality)
    return source_quality


def _as_event_frame(table, row_offset: int, active_ids: set[str], deadline_by_id: dict[str, int], token_by_id: dict[str, set[str]]) -> tuple[pd.DataFrame, set[str]]:
    frame = table.to_pandas()
    if frame.empty:
        return frame, set()
    if isinstance(frame["market"].iloc[0], bytes):
        frame["market"] = frame["market"].str.decode("ascii")
    frame["market"] = frame["market"].astype(str).str.lower()
    frame["_row_order"] = np.arange(row_offset, row_offset + len(frame), dtype=np.int64)
    frame = frame.loc[frame.market.isin(active_ids)].copy()
    if frame.empty:
        return frame, set()
    allowed = frame.apply(lambda row: str(row.asset_id) in token_by_id[str(row.market)], axis=1)
    frame = frame.loc[allowed].copy()
    if frame.empty:
        return frame, set()
    bad_receive = frame["timestamp_received"].isna() | frame["timestamp"].isna()
    unknown_timing_ids = set(frame.loc[bad_receive, "market"].astype(str))
    frame["_received_ns"] = pd.to_datetime(frame["timestamp_received"], utc=True, errors="coerce").array.as_unit("ns").asi8
    frame = frame.loc[~bad_receive].copy()
    if frame.empty:
        return frame, unknown_timing_ids
    row_deadline = frame.market.map(deadline_by_id)
    frame = frame.loc[row_deadline.notna() & frame._received_ns.le(row_deadline.fillna(-1))].copy()
    if frame.empty:
        return frame, unknown_timing_ids
    return frame, unknown_timing_ids


def _apply_native_receive_group(state: dict, rows, received_ns: int) -> None:
    """Apply one stable native receive-time group without constructing complements or fills."""
    import run_btc_preopen_economic_replay as replay

    records = list(rows)
    conflicts, conflict_count = replay._same_receive_group_conflicts(records)
    expected_bbo = {}
    for row in records:
        replay._update_event(state, row, int(received_ns), expected_bbo)
    for token_id, side in conflicts:
        token = replay._token_state(state, token_id)
        token["bid_order_ambiguous" if side == "BUY" else "ask_order_ambiguous"] = True
    state["equal_source_timestamp_price_changes"] += conflict_count


def replay_native_books(protocol: dict, market_rows: pd.DataFrame, protocol_sha256: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    import pyarrow.parquet as pq
    import run_btc_preopen_economic_replay as replay

    parts_dir = ROOT / protocol["source_allowlist"]["pmxt_native_event_parts"]["path"]
    all_parts = {_event_hour(path): path for path in parts_dir.glob("*.parquet")}
    valid = market_rows.loc[market_rows.market_identity_status.eq("CONFIRMED")].copy()
    valid["market_start_utc"] = pd.to_datetime(valid["market_start_utc"], utc=True, errors="raise")
    markets = {}
    lower_by_id, last_hour_by_id = {}, {}
    for row in valid.itertuples(index=False):
        record = row._asdict()
        cid = str(record["condition_id"])
        record["up_token_id"], record["down_token_id"] = str(record["up_token_id"]), str(record["down_token_id"])
        markets[cid] = record
        lower_by_id[cid] = record["market_start_utc"].floor("h") - pd.Timedelta(hours=25)
        last_hour_by_id[cid] = (record["market_start_utc"] - pd.Timedelta(seconds=59)).floor("h")
    start_hour = min(lower_by_id.values())
    end_hour = max(last_hour_by_id.values())
    hours = pd.date_range(start_hour, end_hour, freq="h")
    active_by_hour: dict[pd.Timestamp, set[str]] = defaultdict(set)
    for cid in markets:
        for hour in pd.date_range(lower_by_id[cid], last_hour_by_id[cid], freq="h"):
            active_by_hour[hour].add(cid)
    processed_hours: set[pd.Timestamp] = set()
    states: dict[str, dict] = {}
    t60, t59 = {}, {}
    t60_deadlines = sorted((int((market["market_start_utc"] - pd.Timedelta(seconds=60)).value), cid) for cid, market in markets.items())
    t59_deadlines = sorted((int((market["market_start_utc"] - pd.Timedelta(seconds=59)).value), cid) for cid, market in markets.items())
    i60 = i59 = 0
    unknown_timing_ids: set[str] = set()
    token_by_id = {cid: {record["up_token_id"], record["down_token_id"]} for cid, record in markets.items()}
    deadline_by_id = {cid: int((record["market_start_utc"] - pd.Timedelta(seconds=59)).value) for cid, record in markets.items()}
    part_schema_identity = []
    for hour in hours:
        path = all_parts.get(hour)
        if path is None:
            part_schema_identity.append((hour.isoformat(), None, None, None))
            continue
        schema_text = str(pq.ParquetFile(path).schema_arrow)
        stat = path.stat()
        part_schema_identity.append((hour.isoformat(), path.name, int(stat.st_size), int(stat.st_mtime_ns), hashlib.sha256(schema_text.encode("utf-8")).hexdigest()))
    market_identity = canonical_sha256(valid.loc[:, ["market_start_utc", "condition_id", "up_token_id", "down_token_id"]].astype("string").fillna("<NA>").to_dict(orient="records"))
    book_identity = canonical_sha256({"protocol_sha256": protocol_sha256, "part_schema_identity": part_schema_identity, "market_identity": market_identity})
    book_checkpoint_path = RUNTIME / "checkpoints" / "native_book_replay.json"
    from features.btc_D_continuation_20261029 import atomic_checkpoint, load_checkpoint
    hour_cursor = 0
    book_peak_rss = 0
    unknown_timing_ids: set[str] = set()
    if book_checkpoint_path.is_file():
        saved = load_checkpoint(book_checkpoint_path, identity_sha256=book_identity)
        hour_cursor = int(saved["hour_cursor"])
        if not 0 <= hour_cursor <= len(hours):
            raise RuntimeError("native book checkpoint hour cursor is invalid")
        states, t60, t59 = saved["states"], saved["t60"], saved["t59"]
        processed_hours = {pd.Timestamp(value) for value in saved["processed_hours"]}
        i60, i59 = int(saved["i60"]), int(saved["i59"])
        unknown_timing_ids = set(saved.get("unknown_timing_ids", ()))
        book_peak_rss = int(saved.get("peak_rss_bytes", 0))
        _IO_AUDIT["event_parts_read"] = int(saved.get("event_parts_read", 0))
        _IO_AUDIT["event_projected_rows_read"] = int(saved.get("event_projected_rows_read", 0))
        _IO_AUDIT["event_rows_replayed"] = int(saved.get("event_rows_replayed", 0))

    def capture(cutoff_ns: int, inclusive: bool) -> None:
        nonlocal i60, i59
        while i60 < len(t60_deadlines) and (t60_deadlines[i60][0] < cutoff_ns or (inclusive and t60_deadlines[i60][0] == cutoff_ns)):
            deadline, cid = t60_deadlines[i60]
            state = states.get(cid, replay._new_state())
            market = markets[cid]
            t60[cid] = _direct_book_snapshot(state, market, pd.Timestamp(deadline, unit="ns", tz="UTC"), "t60", processed_hours)
            i60 += 1
        while i59 < len(t59_deadlines) and (t59_deadlines[i59][0] < cutoff_ns or (inclusive and t59_deadlines[i59][0] == cutoff_ns)):
            deadline, cid = t59_deadlines[i59]
            state = states.get(cid, replay._new_state())
            market = markets[cid]
            t59[cid] = _direct_book_snapshot(state, market, pd.Timestamp(deadline, unit="ns", tz="UTC"), "t59", processed_hours)
            states.pop(cid, None)
            i59 += 1

    for hour_index in range(hour_cursor, len(hours)):
        hour = hours[hour_index]
        path = all_parts.get(hour)
        if path is not None:
            parquet = pq.ParquetFile(path)
            available = set(parquet.schema_arrow.names)
            if not set(EVENT_COLUMNS).issubset(available):
                raise RuntimeError(f"PMXT event schema is missing an allowlisted source column: {path.name}")
            processed_hours.add(hour)
            active_ids = active_by_hour.get(hour, set())
            if active_ids:
                deadline_active = {cid: deadline_by_id[cid] for cid in active_ids}
                token_active = {cid: token_by_id[cid] for cid in active_ids}
                row_offset = 0
                relevant = []
                for batch in parquet.iter_batches(columns=list(EVENT_COLUMNS), batch_size=BATCH_ROWS):
                    df, unknown_ids = _as_event_frame(batch, row_offset, active_ids, deadline_active, token_active)
                    unknown_timing_ids.update(unknown_ids)
                    if len(batch):
                        row_offset += len(batch)
                    if not df.empty:
                        relevant.append(df)
                _IO_AUDIT["event_parts_read"] += 1
                _IO_AUDIT["event_projected_rows_read"] += row_offset
                for cid in unknown_timing_ids & active_ids:
                    states.setdefault(cid, replay._new_state())["unknown_event_timing"] = True
                if relevant:
                    events = pd.concat(relevant, ignore_index=True)
                    events.sort_values(["_received_ns", "_row_order"], kind="stable", inplace=True)
                    for received_ns, receive_frame in events.groupby("_received_ns", sort=False):
                        received_ns = int(received_ns)
                        capture(received_ns, inclusive=False)
                        for cid, group in receive_frame.groupby("market", sort=False):
                            cid = str(cid)
                            market = markets[cid]
                            state = states.setdefault(cid, replay._new_state())
                            rows = list(group.itertuples(index=False))
                            _apply_native_receive_group(state, rows, received_ns)
                            _IO_AUDIT["event_rows_replayed"] += len(rows)
                        capture(received_ns, inclusive=True)
        capture(int((hour + pd.Timedelta(hours=1)).value), inclusive=False)
        if (hour_index + 1) % 12 == 0 or hour_index + 1 == len(hours):
            book_peak_rss = max(book_peak_rss, peak_working_set_bytes())
            if book_peak_rss > 24 * 1024**3 or available_ram_bytes() < 8 * 1024**3:
                raise RuntimeError("E0048 book replay exceeded its frozen memory safety gate")
            atomic_checkpoint(book_checkpoint_path, {
                "identity_sha256": book_identity,
                "hour_cursor": hour_index + 1,
                "states": states,
                "t60": t60,
                "t59": t59,
                "processed_hours": sorted(value.isoformat() for value in processed_hours),
                "i60": i60,
                "i59": i59,
                "event_parts_read": _IO_AUDIT["event_parts_read"],
                "event_projected_rows_read": _IO_AUDIT["event_projected_rows_read"],
                "event_rows_replayed": _IO_AUDIT["event_rows_replayed"],
                "unknown_timing_ids": sorted(unknown_timing_ids),
                "peak_rss_bytes": book_peak_rss,
            })
    capture(int((end_hour + pd.Timedelta(hours=1)).value), inclusive=False)
    if i60 != len(t60_deadlines) or i59 != len(t59_deadlines):
        raise RuntimeError("native book replay failed to capture every T-60/T-59 cutoff")

    def ordered_frame(mapping: dict[str, dict], prefix: str) -> pd.DataFrame:
        by_id = {str(item["condition_id"]): item for item in mapping.values()}
        rows = []
        for market in market_rows.itertuples(index=False):
            cid = getattr(market, "condition_id")
            item = by_id.get(str(cid)) if cid is not None and not pd.isna(cid) else None
            if item is None:
                rows.append({"market_start_utc": market.market_start_utc, "condition_id": cid, f"{prefix}_status": "UNKNOWN"})
                continue
            renamed = {f"{prefix}_{key}": value for key, value in item.items() if key not in {"condition_id", "market_start_utc", "entry_time_utc", "cutoff"}}
            rows.append({"market_start_utc": market.market_start_utc, "condition_id": cid, f"{prefix}_entry_time_utc": item["entry_time_utc"], f"{prefix}_status": "CAPTURED", **renamed})
        return pd.DataFrame(rows)

    t60_frame = ordered_frame(t60, "t60")
    t59_frame = ordered_frame(t59, "t59")
    source_quality_input = _merge_source_quality(market_rows, t60_frame, t59_frame)
    manifest = {
        "processed_source_hours": len(processed_hours),
        "missing_source_hours": [hour.isoformat().replace("+00:00", "Z") for hour in hours if hour not in processed_hours],
        "replayed_event_rows": _IO_AUDIT["event_rows_replayed"],
        "markets_with_unknown_event_timing": len(unknown_timing_ids),
        "peak_rss_bytes": book_peak_rss,
        "parts_read": _IO_AUDIT["event_parts_read"],
        "t60_captures": len(t60), "t59_captures": len(t59),
        "source_quality_clean_rows": int(source_quality_input.source_quality_clean.sum()),
        "source_quality_quarantined_rows": int((~source_quality_input.source_quality_clean).sum()),
        "quality_function": "utils.btc_t0009_source_audit.source_quality_mask (unchanged)",
        "checkpoint_path": str(book_checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(book_checkpoint_path) if book_checkpoint_path.is_file() else None,
        "source_part_schema_identity_sha256": canonical_sha256(part_schema_identity),
        "market_technical_identity_sha256": market_identity,
        "source_fields_projected": list(EVENT_COLUMNS),
        "transaction_hash_projected": False,
    }
    return t60_frame, t59_frame, source_quality_input, manifest
