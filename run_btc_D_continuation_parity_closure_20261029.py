"""Frozen development-only E0047 continuation parity closure."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_thread_variable] = "4"

import numpy as np
import pandas as pd

from features import btc_D_continuation_20261029 as continuation


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/research/btc_D_continuation_parity_closure_20261029_v5.json"
E0045_PROTOCOL_PATH = ROOT / "configs/research/btc_D_input_generator_final_bundle_fix_20261029_v2.json"
E0045_MANIFEST_PATH = ROOT / "reports/btc_preopen/btc_D_input_generator_final_bundle_fix_20261029_manifest.json"
RUNTIME = ROOT / "data/analysis/polymarket/BTC/btc_D_continuation_parity_closure_20261029_v5"
RAW_COLUMNS = ("Opened", "Open", "High", "Low", "Close", "Volume", "UM_BTCUSDT_Open", "UM_BTCUSDT_High", "UM_BTCUSDT_Low", "UM_BTCUSDT_Close")
FLOW_RAW_COLUMNS = ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume")
FLOW_COLUMNS = ("taker_Z_1m", "taker_V_1m", "taker_Z_5m", "taker_V_5m", "taker_Z_15m", "taker_V_15m")
FLOW_TAIL_ROWS = 10_095
PREFIX_CHUNK_ROWS = 65_536
MODE_ENV = "BTC_D_E0047_MODE"
IO_COUNTS = {"flow_archive_sha256_bytes": 0, "flow_archive_uncompressed_stream_bytes": 0}
CSV_ROUNDTRIP_AUDIT = []
NON_FAILURE_PARITY_STATUSES = frozenset({"MEASURED_FLOAT32_PRECISION_LOSS", "EXACT", "INFO"})


def _count_parity_failures(statuses) -> int:
    return sum(status != "PASS" and status not in NON_FAILURE_PARITY_STATUSES for status in statuses)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _require_hash(relative: str, expected: str, label: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"frozen {label} hash mismatch: expected {expected}, got {actual}: {path}")
    return path


def load_protocol() -> tuple[dict, dict, dict, dict]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (protocol.get("status") != "FROZEN_BEFORE_REPLAY" or protocol.get("experiment_id") != "E0047"
            or protocol.get("protocol_revision") != 5):
        raise RuntimeError("E0047 protocol is not frozen for replay")
    if (protocol["continuation"]["runtime_root"].replace("\\", "/")
            != str(RUNTIME.relative_to(ROOT)).replace("\\", "/")):
        raise RuntimeError("E0047 runtime identity differs from its pinned protocol")
    if protocol["continuation"]["comparison_rules"].get("panel_timestamp_identity") != (
            "Normalize both UTC Opened key sequences to datetime64[ns] before exact instant and order comparison."):
        raise RuntimeError("E0047 panel timestamp comparison rule changed")
    if protocol["continuation"]["comparison_rules"].get("semantic_dataframe_state") != (
            "Compare exact DataFrame class, shape, ordered columns, index class/dtype/name/values, column dtypes, and exact per-column values with matching NaN masks; no numeric tolerance."):
        raise RuntimeError("E0047 semantic DataFrame checkpoint comparison rule changed")
    scope = protocol["scope"]
    if (scope["source_raw_opened_exclusive_utc"] != "2026-06-30T23:58:00Z"
            or scope["last_allowed_raw_opened_utc"] != "2026-06-30T23:57:00Z"
            or scope["last_allowed_model_feature_opened_utc"] != "2026-06-30T23:53:00Z"
            or int(scope["safe_raw_rows"]) != 3_186_145
            or scope["holdout_market_start_utc"] != "[2026-07-01T00:00:00Z,2026-10-07T00:00:00Z)"):
        raise RuntimeError("E0047 development or protected-data scope changed")
    _require_hash(protocol["previous_protocol"]["path"], protocol["previous_protocol"]["sha256"],
                  f"preserved E0047 v{protocol['protocol_revision'] - 1} preregistration")
    if int(protocol["previous_attempt"]["protocol_revision"]) != protocol["protocol_revision"] - 1:
        raise RuntimeError("E0047 previous attempt revision is not the immediately preceding protocol")
    for record in protocol["previous_attempt"]["files"]:
        _require_hash(record["path"], record["sha256"],
                      f"preserved E0047 v{protocol['previous_attempt']['protocol_revision']} attempt artifact {record['path']}")
    for record in protocol["previous_attempt"].get("report_artifacts", []):
        _require_hash(record["path"], record["sha256"], f"preserved prior E0047 report {record['path']}")
    execution = protocol["execution"]
    source_products = execution["source_products"]
    if (execution.get("mode") != "REVALIDATE_HASH_PINNED_FRESH_PROCESS_PRODUCTS"
            or execution.get("source_runtime_root") != source_products["runtime_root"]):
        raise RuntimeError("E0047 v5 must revalidate its separately pinned fresh-process products")
    _require_hash(source_products["protocol_path"], source_products["protocol_sha256"],
                  f"preserved E0047 v{source_products['protocol_revision']} source preregistration")
    for record in source_products["files"]:
        _require_hash(record["path"], record["sha256"], f"pinned source E0047 runtime artifact {record['path']}")
    for record in source_products["report_artifacts"]:
        _require_hash(record["path"], record["sha256"], f"pinned source E0047 report {record['path']}")
    for item in execution["child_results"]:
        _require_hash(item["path"], item["sha256"], f"prior fresh-process execution record {item['path']}")
    for record in protocol["parent"]["preserved_E0046_artifacts"]:
        _require_hash(record["path"], record["sha256"], f"immutable E0046 artifact {record['path']}")
    if protocol["scope"]["evaluation_authorized"] or protocol["scope"]["activated"] or protocol["scope"]["holdout_status"] != "HOLDOUT_NOT_RUN":
        raise RuntimeError("E0047 development-only and holdout guards changed")
    for relative, expected in protocol["required_code_hashes"].items():
        _require_hash(relative, expected, f"frozen implementation {relative}")
    _require_hash(protocol["parent"]["bundle_manifest_path"], protocol["parent"]["bundle_manifest_sha256"], "final E0041 bundle")
    _require_hash(protocol["parent"]["E0045_manifest_path"], protocol["parent"]["E0045_manifest_sha256"], "E0045 manifest")
    _require_hash(protocol["parent"]["E0045_protocol_path"], protocol["parent"]["E0045_protocol_sha256"], "E0045 frozen protocol")
    _require_hash(protocol["scope"]["source_raw_path"], protocol["scope"]["source_raw_sha256"], "safe development raw prefix")
    e45_manifest = json.loads((ROOT / protocol["parent"]["E0045_manifest_path"]).read_text(encoding="utf-8"))
    if e45_manifest.get("protocol_sha256") != protocol["parent"]["E0045_protocol_sha256"]:
        raise RuntimeError("E0045 execution manifest does not identify its frozen v2 protocol")
    e45_protocol_path = ROOT / protocol["parent"]["E0045_protocol_path"]
    if e45_protocol_path.resolve() != E0045_PROTOCOL_PATH.resolve():
        raise RuntimeError("E0047 E0045 protocol path differs from the pinned runner input")
    e45 = json.loads(e45_protocol_path.read_text(encoding="utf-8"))
    from features import btc_D_input_generator_20261029 as e45_code

    e45_protocol = e45_code.load_protocol()
    safe_source = e45_protocol["allowed_inputs"]["safe_index_ohlcv_prefix"]
    if (e45_protocol["scope"]["source_raw_opened_exclusive_utc"] != protocol["scope"]["source_raw_opened_exclusive_utc"]
            or safe_source["path"] != protocol["scope"]["source_raw_path"]
            or safe_source["sha256"] != protocol["scope"]["source_raw_sha256"]
            or int(safe_source["rows"]) != int(protocol["scope"]["safe_raw_rows"])):
        raise RuntimeError("E0047 safe source or exclusive cutoff differs from the frozen E0045 v2 input")
    lineage = e45_protocol["allowed_inputs"]["base_feature_lineage"]
    for name in ("selection_pipeline_manifest", "selection_result", "final_feature_dataset", "feature_metadata",
                 "indicator_source_run_manifest", "indicator_stage", "effective_config",
                 "reaction_profile_config", "volume_profile_config"):
        record = lineage[name]
        _require_hash(record["path"], record["sha256"], f"E0041 {name}")
    for record in lineage["indicator_fit_results"]:
        _require_hash(record["path"], record["sha256"], "E0041 saved indicator fit result")
    for family in ("reaction_profile_state", "volume_profile_state"):
        record = lineage[family]
        _require_hash(record["npz_path"], record["npz_sha256"], f"E0041 {family} arrays")
        _require_hash(record["json_path"], record["json_sha256"], f"E0041 {family} metadata")
    for path, digest in e45_protocol["allowed_inputs"]["builder_configuration_hashes"].items():
        _require_hash(path, digest, "E0041 feature-builder configuration")
    bundle_identity, d_manifest, transforms = e45_code.verify_bundle(e45_protocol)
    if bundle_identity["bundle_manifest_sha256"] != protocol["parent"]["bundle_manifest_sha256"]:
        raise RuntimeError("E0047 final E0041 bundle identity differs from the preregistration")
    if d_manifest.get("holdout_rows_read") != 0 or d_manifest.get("holdout_predictions_generated") != 0:
        raise RuntimeError("final E0041 bundle no longer records zero holdout reads and predictions")
    return protocol, e45, {"identity": bundle_identity, "manifest": d_manifest}, transforms


def _memory_available_gib() -> float:
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
        raise OSError("GlobalMemoryStatusEx failed")
    return status.ullAvailPhys / 1024**3


def _peak_rss_gib() -> float:
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
    return counters.PeakWorkingSetSize / 1024**3


def _read_raw_exact(path: Path, row_count: int, *, start_offset: int = 0) -> tuple[pd.DataFrame, int]:
    """Read exactly row_count lines; the parser never requests a later CSV row."""
    pieces = []
    with path.open("rb") as stream:
        header = stream.readline()
        if start_offset:
            stream.seek(start_offset)
        remaining = int(row_count)
        while remaining:
            lines = []
            for _ in range(min(PREFIX_CHUNK_ROWS, remaining)):
                line = stream.readline()
                if not line:
                    break
                lines.append(line)
            if not lines:
                break
            part = pd.read_csv(io.BytesIO(header + b"".join(lines)), usecols=list(RAW_COLUMNS))
            pieces.append(part)
            remaining -= len(lines)
            if len(lines) == 0:
                break
        offset = stream.tell()
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=RAW_COLUMNS)
    frame["Opened"] = pd.to_datetime(frame["Opened"], utc=True, errors="raise")
    return frame, offset


def _verify_raw_sequence(frame: pd.DataFrame, *, expected_first=None, expected_last=None) -> None:
    if frame.empty:
        raise RuntimeError("development raw range unexpectedly empty")
    opened = pd.DatetimeIndex(frame["Opened"])
    if opened.has_duplicates or not opened.is_monotonic_increasing:
        raise RuntimeError("development raw Opened values are not unique and ordered")
    if expected_first and opened[0] != pd.Timestamp(expected_first):
        raise RuntimeError(f"raw prefix begins at {opened[0]}, expected {expected_first}")
    if expected_last and opened[-1] != pd.Timestamp(expected_last):
        raise RuntimeError(f"raw prefix ends at {opened[-1]}, expected {expected_last}")
    cutoff = pd.Timestamp("2026-06-30T23:58:00Z")
    if opened[-1] >= cutoff:
        raise RuntimeError("raw reader crossed the exclusive safe development cutoff")


def _archive_flow_range(e45_manifest: dict, start_inclusive: pd.Timestamp,
                        end_inclusive: pd.Timestamp) -> pd.DataFrame:
    """Read only the requested records from hash-pinned local archive CSV members."""
    if start_inclusive > end_inclusive:
        return pd.DataFrame(columns=["Opened", *FLOW_RAW_COLUMNS])
    archives = {record["month"]: record for record in e45_manifest["flow_source"]["archives"]}
    month_cursor = start_inclusive.strftime("%Y-%m")
    end_month = end_inclusive.strftime("%Y-%m")
    frames = []
    while month_cursor <= end_month:
        record = archives[month_cursor]
        archive_path = _require_hash(record["path"], record["expected_archive_sha256"], f"E0045 source archive {month_cursor}")
        IO_COUNTS["flow_archive_sha256_bytes"] += archive_path.stat().st_size
        month_start = pd.Timestamp(f"{month_cursor}-01T00:00:00Z")
        month_stop = month_start + pd.offsets.MonthBegin(1)
        slice_start = max(start_inclusive, month_start)
        slice_end = min(end_inclusive, month_stop - pd.Timedelta(minutes=1))
        first_index = int((slice_start - month_start) / pd.Timedelta(minutes=1))
        last_index = int((slice_end - month_start) / pd.Timedelta(minutes=1))
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise RuntimeError(f"expected one monthly raw CSV in {archive_path.name}")
            with archive.open(members[0], "r") as binary:
                header_line = binary.readline()
                header = next(csv.reader([header_line.decode("utf-8-sig").rstrip("\r\n")]))
                positions = {name: header.index(name) for name in ("open_time", *FLOW_RAW_COLUMNS)}
                for _ in range(first_index):
                    skipped = binary.readline()
                    IO_COUNTS["flow_archive_uncompressed_stream_bytes"] += len(skipped)
                    if not skipped:
                        raise RuntimeError(f"source archive {month_cursor} ended before requested raw range")
                lines = []
                for row_index in range(first_index, last_index + 1):
                    line = binary.readline()
                    IO_COUNTS["flow_archive_uncompressed_stream_bytes"] += len(line)
                    if not line:
                        raise RuntimeError(f"source archive {month_cursor} ended at row {row_index}")
                    fields = next(csv.reader([line.decode("utf-8").rstrip("\r\n")]))
                    opened = pd.Timestamp(int(fields[positions["open_time"]]), unit="ms", tz="UTC")
                    expected = month_start + pd.Timedelta(minutes=row_index)
                    if opened != expected:
                        raise RuntimeError(f"source archive time index mismatch in {month_cursor}: {opened} != {expected}")
                    lines.append({"Opened": opened, **{name: float(fields[positions[name]]) for name in FLOW_RAW_COLUMNS}})
        frames.append(pd.DataFrame(lines))
        next_month = month_stop
        month_cursor = next_month.strftime("%Y-%m")
    result = pd.concat(frames, ignore_index=True)
    if len(result) != int((end_inclusive - start_inclusive) / pd.Timedelta(minutes=1)) + 1:
        raise RuntimeError("hash-pinned raw FLOW archive range contains missing or duplicated minutes")
    return result


def _stream_dynamic_prefix(frame: pd.DataFrame, transforms: dict, *, start_state=None) -> tuple[dict, pd.DataFrame]:
    from features.btc_D_input_generator_20261029 import generate_dynamic_block, new_dynamic_state

    state = start_state if start_state is not None else new_dynamic_state(transforms)
    tail = pd.DataFrame(columns=RAW_COLUMNS)
    for start in range(0, len(frame), PREFIX_CHUNK_ROWS):
        current = frame.iloc[start:start + PREFIX_CHUNK_ROWS].loc[:, RAW_COLUMNS].copy()
        rolling = current if tail.empty else pd.concat([tail, current], ignore_index=True)
        generate_dynamic_block(state, rolling, new_rows_start=len(rolling) - len(current))
        tail = rolling.tail(720).reset_index(drop=True)
    return state, tail


def _seed_profile_states(prefix: pd.DataFrame, e45_protocol: dict) -> dict:
    lineage = e45_protocol["allowed_inputs"]["base_feature_lineage"]
    metadata = json.loads((ROOT / lineage["feature_metadata"]["path"]).read_text(encoding="utf-8"))
    from features.reaction_profile_fixed_grid import bootstrap_state_from_history as rp_bootstrap
    from features.volume_profile_fixed_range import bootstrap_state_from_history as vp_bootstrap

    rp_cfg = metadata["reaction_profile_fixed_grid"]
    vp_cfg = metadata["volume_profile_fixed_range"]
    return {"reaction_profile": rp_bootstrap(prefix, rp_cfg), "volume_profile": vp_bootstrap(prefix, vp_cfg)}


def _indicator_params(e45_protocol: dict) -> dict[str, dict]:
    import create_modeling_dataset as dataset_builder

    lineage = e45_protocol["allowed_inputs"]["base_feature_lineage"]
    stage = json.loads((ROOT / lineage["indicator_stage"]["path"]).read_text(encoding="utf-8"))
    configs = dataset_builder.parse_fit_results(ROOT / stage["fit_results_dir"])
    selected = {record["indicator"]: record["params"] for record in configs
                if record["indicator"] in {"ADX", "ChaikinOsc", "BollingerBands"}}
    if set(selected) != {"ADX", "ChaikinOsc", "BollingerBands"}:
        raise RuntimeError("saved E0041 ADX/Chaikin/Bollinger fit records are incomplete")
    return selected


def _profile_features_for_rows(states: dict, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    from features.reaction_profile_fixed_grid import update_state_with_candle as rp_update, _extract_feature_row_array as rp_extract

    rp_state = states["reaction_profile"]
    vp_state = states["volume_profile"]
    rp_cols = list(rp_state["feature_columns"])
    vp_cols = list(vp_state["feature_columns"])
    rp = np.empty((len(rows), len(rp_cols)), dtype=np.float64)
    vp = continuation.advance_volume_profile_state(vp_state, rows)
    for idx, row in enumerate(rows.itertuples(index=False)):
        rp_update(rp_state, open=row.Open, high=row.High, low=row.Low, close=row.Close)
        rp[idx] = rp_extract(rp_state, close=row.Close)
    if len(rows):
        last_time = pd.Timestamp(rows["Opened"].iloc[-1]).strftime("%Y-%m-%dT%H:%M:%S")
        rp_state["last_candle_time"] = last_time
        vp_state["last_candle_time"] = last_time
    return {**{name: rp[:, index] for index, name in enumerate(rp_cols)},
            **{name: vp[:, index] for index, name in enumerate(vp_cols)}}


def _build_base64_from_tail(tail: pd.DataFrame, suffix: pd.DataFrame, *, base_order: list[str],
                            e45_protocol: dict, work_dir: Path) -> pd.DataFrame:
    """Run the frozen E0041 batch feature builders on the sufficient finite tail plus new raw."""
    import create_modeling_dataset as dataset_builder
    from utils.data import load_modeling_dataset_settings, resolve_modeling_dataset_output_paths

    lineage = e45_protocol["allowed_inputs"]["base_feature_lineage"]
    metadata = json.loads((ROOT / lineage["feature_metadata"]["path"]).read_text(encoding="utf-8"))
    indicator_stage = json.loads((ROOT / lineage["indicator_stage"]["path"]).read_text(encoding="utf-8"))
    raw_path = work_dir / "rolling_input.csv"
    work_dir.mkdir(parents=True, exist_ok=True)
    rolling = pd.concat([tail, suffix], ignore_index=True) if len(tail) else suffix.copy()
    rolling.loc[:, RAW_COLUMNS].to_csv(raw_path, index=False)
    parsed_raw = pd.read_csv(raw_path)
    if len(parsed_raw) != len(rolling) or tuple(parsed_raw.columns) != RAW_COLUMNS:
        raise RuntimeError("temporary CSV parser changed raw row count or column order")
    opened_roundtrip = pd.to_datetime(parsed_raw["Opened"], utc=True, errors="raise")
    if not pd.DatetimeIndex(opened_roundtrip).equals(pd.DatetimeIndex(rolling["Opened"])):
        raise RuntimeError("temporary CSV roundtrip changed exact raw Opened keys")
    numeric_mismatch_rows = 0
    max_abs_error = 0.0
    for column in RAW_COLUMNS[1:]:
        expected_values = rolling[column].to_numpy(dtype=np.float64, copy=False)
        parsed_values = parsed_raw[column].to_numpy(dtype=np.float64, copy=False)
        mismatch = ~(parsed_values == expected_values) & ~(np.isnan(parsed_values) & np.isnan(expected_values))
        numeric_mismatch_rows += int(mismatch.sum())
        if mismatch.any():
            finite = np.isfinite(parsed_values[mismatch]) & np.isfinite(expected_values[mismatch])
            if finite.any():
                max_abs_error = max(max_abs_error, float(np.max(np.abs(parsed_values[mismatch][finite] - expected_values[mismatch][finite]))))
    CSV_ROUNDTRIP_AUDIT.append({"rows": len(rolling), "numeric_mismatch_rows": numeric_mismatch_rows,
                                "max_abs_error": max_abs_error,
                                "dtypes": {column: str(parsed_raw[column].dtype) for column in RAW_COLUMNS},
                                "parser": "pandas.read_csv default C parser; same call as create_modeling_dataset.py"})
    if numeric_mismatch_rows:
        raise RuntimeError(f"temporary CSV parser changed {numeric_mismatch_rows} raw numeric values")
    subset_path = work_dir / "selected_features.json"
    subset_path.write_text(json.dumps({"features": base_order[:64]}, separators=(",", ":")), encoding="utf-8")
    settings = load_modeling_dataset_settings(asset="BTC", dataset_profile_name="BTC", modeling_profile_name="BTC")
    settings.update({
        "active_asset": "BTC", "symbol": "BTCUSD", "interval": "1m",
        "raw_data_dir": raw_path.parent, "data_dir": raw_path.parent, "base_data_file": raw_path.name,
        "fit_results_dir": ROOT / indicator_stage["fit_results_dir"],
        "feature_subset_path": subset_path, "feature_subset_list_key": "features",
        "excluded_feature_names": (), "feature_intervals": metadata["feature_intervals"],
        "basis_premium_features": metadata["basis_premium_features"],
        "reaction_profile_fixed_grid": metadata["reaction_profile_fixed_grid"],
        "volume_profile_fixed_range": metadata["volume_profile_fixed_range"],
        "float_precision": "float64", "profile_state_dir": work_dir / "profile_states",
        "modeling_output_dir": work_dir / "feature_output", "output_suffix": "e0047_continuation",
        "preview_rows": 0,
        "preopen_target": {
            "development_cutoff_exclusive_utc": "2026-06-30T23:58:00Z",
            "development_input_path": str(raw_path),
            "source_raw_data_path": lineage["source_raw_path_identity_only"],
        },
    })
    output_path = resolve_modeling_dataset_output_paths(settings)["parquet"]
    built_path = dataset_builder.build_dataset_from_settings(settings, include_targets=False)
    if Path(built_path).resolve() != Path(output_path).resolve():
        raise RuntimeError("E0047 rolling BASE64 builder wrote to an unexpected path")
    import pyarrow.parquet as pq

    result = pq.read_table(built_path, columns=["Opened", *base_order[:64]]).to_pandas()
    result["Opened"] = pd.to_datetime(result["Opened"], utc=True, errors="raise")
    if len(result) != len(rolling) or not pd.DatetimeIndex(result["Opened"]).equals(pd.DatetimeIndex(rolling["Opened"])):
        raise RuntimeError("rolling E0041 builder changed or reordered exact raw Opened keys")
    return result.iloc[len(tail):].reset_index(drop=True)


def _compute_flow_block(flow_rows: pd.DataFrame) -> pd.DataFrame:
    from features.btc_taker_flow import compute_taker_flow_features

    source = flow_rows.loc[:, ["Opened", *FLOW_RAW_COLUMNS]].copy()
    source = source.set_index("Opened")
    source.columns = ["open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]
    return compute_taker_flow_features(source).reset_index(names="Opened")


def _recursive_indicator_overrides(base: pd.DataFrame, rolling_raw: pd.DataFrame, suffix_count: int,
                                   state: dict, params: dict) -> dict:
    from features.ADX import custom_adx
    from features.ta_tools import get_1d_ma, get_source_from_ohlcv, bollinger_channel_signal

    opened = pd.DatetimeIndex(rolling_raw["Opened"])
    target_opened = pd.DatetimeIndex(base["Opened"])
    if not target_opened.equals(opened[-suffix_count:]):
        raise RuntimeError("recursive-indicator output keys do not equal the raw suffix")
    ohlcv = rolling_raw[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=np.float64, copy=False)
    _, plus_di, minus_di = custom_adx(
        ohlcv,
        atr_period=int(params["ADX"]["atr_period"]),
        posDM_period=int(params["ADX"]["posDM_period"]),
        negDM_period=int(params["ADX"]["negDM_period"]),
        adx_period=int(params["ADX"]["adx_period"]),
        ma_type_atr=str(params["ADX"]["ma_type_atr"]),
        ma_type_posDM=str(params["ADX"]["ma_type_posDM"]),
        ma_type_negDM=str(params["ADX"]["ma_type_negDM"]),
        ma_type_adx=str(params["ADX"]["ma_type_adx"]),
    )
    adx = continuation.advance_adx_tema(state["adx_tema"], plus_di[-suffix_count:], minus_di[-suffix_count:])
    adx_name = next(name for name in base.columns if name.startswith("ADX_fit_"))
    # get_adx_values returns raw ADX; the deprecated threshold is not applied.
    base[adx_name] = adx

    # Continue Bollinger's frozen LMA accumulator and finite NWMA variance window.
    bb = params["BollingerBands"]
    center_source = get_source_from_ohlcv(ohlcv[-suffix_count:], str(bb["ma_source"]))
    center = continuation.advance_lma_state(state["bollinger_lma"], center_source)
    variance_source = get_source_from_ohlcv(ohlcv[-suffix_count:], str(bb["std_source"]))
    diff2 = (variance_source - center) ** 2
    variance_input = np.concatenate((state["bollinger_diff2_tail"], diff2))
    variance_smooth = get_1d_ma(variance_input, str(bb["std_ma_type"]), int(bb["std_ma_period"]))[-suffix_count:]
    std = np.sqrt(np.where(variance_smooth >= 0.0, variance_smooth, np.nan))
    bb_name = next(name for name in base.columns if name.startswith("BollingerBands_fit_"))
    base[bb_name] = bollinger_channel_signal(
        np_close=ohlcv[-suffix_count:, 3], np_xMA=center, np_STD=std, std_multi=1.0,
    )
    state["bollinger_diff2_tail"] = variance_input[-(int(bb["std_ma_period"]) - 1):].copy()

    # Continue ADL and every TA-Lib WMA accumulator used by the frozen HMA.
    adl_suffix = continuation.advance_chaikin_adl(state["adl"], rolling_raw.iloc[-suffix_count:])
    adl_tail = np.asarray(state["adl_tail"], dtype=np.float64)
    chaikin = params["ChaikinOsc"]
    filters = state["chaikin_wma"]
    half = continuation.advance_talib_wma_state(filters["half"], adl_suffix)
    full = continuation.advance_talib_wma_state(filters["full"], adl_suffix)
    raw_hma = 2 * half - full
    slow = continuation.advance_talib_wma_state(filters["final"], raw_hma)
    fast_period = int(chaikin["fast_period"])
    fast_tail = adl_tail[-(fast_period - 1):]
    fast_input = np.concatenate((fast_tail, adl_suffix))
    fast = get_1d_ma(fast_input, str(chaikin["fast_ma_type"]), fast_period)[-suffix_count:]
    chosc = fast - slow
    previous = np.concatenate(([float(filters["last_oscillator"])], chosc[:-1]))
    chaikin_name = next(name for name in base.columns if name.startswith("ChaikinOsc_fit_"))
    base[chaikin_name] = chosc
    diff_name = f"{chaikin_name}_diff"
    if diff_name in base.columns:
        base[diff_name] = chosc - previous
    if len(chosc):
        filters["last_oscillator"] = float(chosc[-1])
    state["adl_tail"] = np.concatenate((adl_tail, adl_suffix))[-continuation.TAIL_ROWS:].copy()
    return {"adx": adx, "bollinger": base[bb_name].to_numpy(copy=False), "chaikin": chosc}


def _seed_checkpoint(prefix: pd.DataFrame, *, byte_offset: int, row_index: int,
                     protocol: dict, e45_protocol: dict, e45_manifest: dict,
                     d_bundle: dict, transforms: dict, checkpoint_name: str) -> dict:
    from features import btc_D_input_generator_20261029 as e45_code
    from features.btc_D_input_generator_20261029 import _dynamic_checkpoint_payload

    scope_point = next(item for item in protocol["continuation"]["splits"] if item["checkpoint_after_opened_utc"] == prefix["Opened"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ"))
    expected_rows = int(scope_point["expected_prefix_rows"])
    if len(prefix) != expected_rows or row_index != expected_rows:
        raise RuntimeError(f"checkpoint {checkpoint_name} row count differs from its frozen prefix count")
    _verify_raw_sequence(prefix, expected_last=scope_point["checkpoint_after_opened_utc"])
    profile_states = _seed_profile_states(prefix, e45_protocol)
    intervals = tuple(e45_manifest["feature_contract"]["base70"]["base64"])
    streak_labels = tuple(name.removeprefix("candle_streak_") for name in intervals if name.startswith("candle_streak_"))
    streak = continuation.initialize_streak_states(prefix, streak_labels)
    indicator_params = _indicator_params(e45_protocol)
    adx_state = continuation.seed_adx_tema_state(prefix, indicator_params["ADX"])
    adl, adl_values = continuation.seed_chaikin_adl(prefix)
    from features.ta_tools import get_1d_ma, get_source_from_ohlcv
    bb = indicator_params["BollingerBands"]
    bb_source = get_source_from_ohlcv(
        prefix[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=np.float64, copy=False),
        str(bb["ma_source"]),
    )
    bollinger_lma, center_values = continuation.seed_lma_state(bb_source, int(bb["ma_period"]))
    variance_source = get_source_from_ohlcv(
        prefix[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=np.float64, copy=False),
        str(bb["std_source"]),
    )
    diff2 = (variance_source - center_values) ** 2
    bb_tail_rows = int(bb["std_ma_period"]) - 1
    bollinger_diff2_tail = diff2[-bb_tail_rows:].copy()
    chaikin = indicator_params["ChaikinOsc"]
    half_period = int(chaikin["slow_period"]) // 2
    full_period = int(chaikin["slow_period"])
    final_period = int(np.sqrt(full_period))
    half_state, half_values = continuation.seed_talib_wma_state(adl_values, half_period)
    full_state, full_values = continuation.seed_talib_wma_state(adl_values, full_period)
    raw_hma = 2 * half_values - full_values
    final_state, final_values = continuation.seed_talib_wma_state(raw_hma, final_period)
    fast_period = int(chaikin["fast_period"])
    fast_last = get_1d_ma(adl_values, str(chaikin["fast_ma_type"]), fast_period)[-1]
    last_oscillator = float(fast_last - final_values[-1])
    flow_end = pd.Timestamp(prefix["Opened"].iloc[-1])
    flow_tail = _archive_flow_range(
        e45_manifest, flow_end - pd.Timedelta(minutes=FLOW_TAIL_ROWS - 1), flow_end,
    )
    if len(flow_tail) != FLOW_TAIL_ROWS:
        raise RuntimeError("E0045 raw FLOW tail is shorter than the frozen 10,095-row requirement")
    dynamic_state, dynamic_tail = _stream_dynamic_prefix(prefix, transforms)
    if not dynamic_tail["Opened"].equals(prefix.tail(720)["Opened"].reset_index(drop=True)):
        raise RuntimeError("dynamic checkpoint raw tail does not match its exact prefix")
    tail = prefix.tail(continuation.TAIL_ROWS).reset_index(drop=True)
    identity = _continuation_identity(protocol, d_bundle["identity"])
    return {
        "schema_version": 3,
        "checkpoint_name": checkpoint_name,
        "identity_sha256": identity,
        "source_sha256": protocol["scope"]["source_raw_sha256"],
        "source_row_index": row_index,
        "source_byte_offset": byte_offset,
        "last_processed_opened": prefix["Opened"].iloc[-1].isoformat(),
        "next_expected_opened": (prefix["Opened"].iloc[-1] + pd.Timedelta(minutes=1)).isoformat(),
        "raw_tail": tail,
        "flow_raw_tail": flow_tail,
        "streak_state": streak,
        "profile_states": profile_states,
        "adx_tema": adx_state,
        "adl": adl,
        "adl_tail": np.asarray(adl_values[-continuation.TAIL_ROWS:], dtype=np.float64),
        "bollinger_lma": bollinger_lma,
        "bollinger_diff2_tail": bollinger_diff2_tail,
        "chaikin_wma": {"half": half_state, "full": full_state, "final": final_state,
                        "last_oscillator": last_oscillator},
        "dynamic_state": _dynamic_checkpoint_payload(dynamic_state, temporary_root=RUNTIME / "dynamic_checkpoint_parts"),
        "dynamic_tail": dynamic_tail,
        "indicator_params": indicator_params,
        "flow_tail_rows": int(len(flow_tail)),
        "raw_tail_rows": int(len(tail)),
        "created_prefix_rows": int(len(prefix)),
    }


def _continuation_identity(protocol: dict, bundle_identity: dict) -> str:
    return continuation.canonical_sha256({
        "protocol": protocol,
        "bundle_identity": bundle_identity,
        "code_hashes": protocol["required_code_hashes"],
        "source_sha256": protocol["scope"]["source_raw_sha256"],
    })


def _build_cp1(protocol: dict, e45_protocol: dict, e45_manifest: dict,
               d_bundle: dict, transforms: dict) -> dict:
    start = time.perf_counter()
    raw_path = ROOT / protocol["scope"]["source_raw_path"]
    point = protocol["continuation"]["splits"][0]
    raw_read_started = time.perf_counter()
    prefix, offset = _read_raw_exact(raw_path, int(point["expected_prefix_rows"]))
    raw_read_seconds = time.perf_counter() - raw_read_started
    checkpoint_build_started = time.perf_counter()
    state = _seed_checkpoint(prefix, byte_offset=offset, row_index=len(prefix), protocol=protocol,
                             e45_protocol=e45_protocol, e45_manifest=e45_manifest,
                             d_bundle=d_bundle, transforms=transforms, checkpoint_name="C1")
    checkpoint_build_seconds = time.perf_counter() - checkpoint_build_started
    checkpoint_path = RUNTIME / "checkpoints" / "C1.json"
    checkpoint_write_started = time.perf_counter()
    checkpoint_sha = continuation.atomic_checkpoint(checkpoint_path, state)
    checkpoint_write_seconds = time.perf_counter() - checkpoint_write_started
    return {"checkpoint": str(checkpoint_path.relative_to(ROOT)), "sha256": checkpoint_sha,
            "rows": len(prefix), "byte_offset": offset, "wall_seconds": time.perf_counter() - start,
            "raw_read_seconds": raw_read_seconds,
            "checkpoint_build_seconds": checkpoint_build_seconds,
            "checkpoint_write_seconds": checkpoint_write_seconds,
            "flow_archive_io": dict(IO_COUNTS),
            "peak_rss_gib": _peak_rss_gib(), "available_ram_after_gib": _memory_available_gib(),
            "checkpoint_bytes": checkpoint_path.stat().st_size}


def _restore_checkpoint(path: Path, protocol: dict, d_bundle: dict, transforms: dict) -> tuple[dict, dict]:
    from features.btc_D_input_generator_20261029 import restore_dynamic_state

    identity = _continuation_identity(protocol, d_bundle["identity"])
    state = continuation.load_checkpoint(path, identity_sha256=identity)
    if int(state.get("schema_version", -1)) != int(protocol["continuation"]["checkpoint_schema_version"]):
        raise RuntimeError("checkpoint schema differs from the frozen E0047 continuation version")
    if state["source_sha256"] != protocol["scope"]["source_raw_sha256"]:
        raise RuntimeError("continuation checkpoint is bound to a different raw source")
    dynamic = restore_dynamic_state(state["dynamic_state"], transforms, temporary_root=RUNTIME / "restore_parts")
    return state, dynamic


def _write_suffix_outputs(stage: str, base: pd.DataFrame, flow: pd.DataFrame, dynamic: dict,
                          opened: pd.DatetimeIndex) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = RUNTIME / "suffixes"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for label, frame in (("base64", base), ("flow6", flow)):
        path = out_dir / f"{stage}_{label}.parquet"
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")
        result[label] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "rows": len(frame), "bytes": path.stat().st_size}
    dyn_data = {"Opened": opened}
    for name, matrix in dynamic.items():
        if name == "memory":
            prefix = "memory"
        elif name.startswith("esn_"):
            prefix = name
        else:
            continue
        for index in range(matrix.shape[1]):
            dyn_data[f"{prefix}_{index:02d}"] = matrix[:, index]
    dyn = pd.DataFrame(dyn_data)
    path = out_dir / f"{stage}_dynamic.parquet"
    pq.write_table(pa.Table.from_pandas(dyn, preserve_index=False), path, compression="zstd")
    result["dynamic"] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "rows": len(dyn), "bytes": path.stat().st_size}
    return result


def _resume_suffix(checkpoint_name: str, stop_row: int, stage_name: str, *,
                   save_checkpoint_name: str | None, snapshot_rows: dict[int, str],
                   protocol: dict, e45_protocol: dict, e45_manifest: dict,
                   d_bundle: dict, transforms: dict) -> dict:
    from features.btc_D_input_generator_20261029 import generate_dynamic_block, _dynamic_checkpoint_payload

    started = time.perf_counter()
    checkpoint_path = RUNTIME / "checkpoints" / f"{checkpoint_name}.json"
    state_read_started = time.perf_counter()
    state, dynamic_state = _restore_checkpoint(checkpoint_path, protocol, d_bundle, transforms)
    state_read_seconds = time.perf_counter() - state_read_started
    source_path = ROOT / protocol["scope"]["source_raw_path"]
    row_index = int(state["source_row_index"])
    start_row = row_index
    start_offset = int(state["source_byte_offset"])
    if int(stop_row) <= row_index:
        raise RuntimeError("requested resume range is empty")
    base_order = d_bundle["manifest"]["base_feature_order"]
    boundaries = sorted({row for row in snapshot_rows if start_row < int(row) < int(stop_row)} | {int(stop_row)})
    base_parts, flow_parts, dynamic_parts, opened_parts = [], [], [], []
    stage_seconds = {"raw_suffix_read": 0.0, "base64_builder": 0.0, "streak_profile_and_indicators": 0.0,
                    "flow_formula": 0.0, "dynamic_update": 0.0, "checkpoint_write": 0.0,
                    "output_write": 0.0}
    base_builder_rows = 0
    flow_formula_rows = 0
    raw_tail_rows_recomputed = 0
    flow_tail_rows_recomputed = 0
    snapshot_records = []
    input_byte_offset = start_offset
    for boundary_index, boundary in enumerate(boundaries):
        count = int(boundary) - int(state["source_row_index"])
        raw_read_started = time.perf_counter()
        suffix, byte_offset = _read_raw_exact(source_path, count, start_offset=int(state["source_byte_offset"]))
        stage_seconds["raw_suffix_read"] += time.perf_counter() - raw_read_started
        expected_last = next((item["checkpoint_after_opened_utc"] for item in protocol["continuation"]["splits"]
                              if int(item["expected_prefix_rows"]) == int(boundary)), None)
        if int(boundary) == int(protocol["scope"]["safe_raw_rows"]):
            expected_last = protocol["scope"]["last_allowed_raw_opened_utc"]
        _verify_raw_sequence(suffix, expected_last=expected_last)
        expected_first = pd.Timestamp(state["next_expected_opened"])
        if suffix["Opened"].iloc[0] != expected_first:
            raise RuntimeError("fresh-process continuation first Opened differs from the checkpoint cursor")
        raw_tail = state["raw_tail"]
        rolling_raw = pd.concat([raw_tail, suffix], ignore_index=True)
        if len(raw_tail) != continuation.TAIL_ROWS:
            raise RuntimeError("BASE64 continuation lacks the preregistered 12,001-row tail")
        if not rolling_raw["Opened"].is_monotonic_increasing or rolling_raw["Opened"].duplicated().any():
            raise RuntimeError("rolling BASE64 raw rows duplicate or reorder checkpoint keys")
        base_build_started = time.perf_counter()
        base = _build_base64_from_tail(raw_tail, suffix, base_order=base_order, e45_protocol=e45_protocol,
                                       work_dir=RUNTIME / "builder" / stage_name / f"block_{boundary_index}")
        stage_seconds["base64_builder"] += time.perf_counter() - base_build_started
        base_builder_rows += len(rolling_raw)
        raw_tail_rows_recomputed += len(raw_tail)
        state_update_started = time.perf_counter()
        streak = continuation.advance_streak_states(state["streak_state"], suffix)
        profile = _profile_features_for_rows(state["profile_states"], suffix)
        for name, values in {**streak, **profile}.items():
            if name in base:
                base[name] = values
        recursive = _recursive_indicator_overrides(base, rolling_raw, len(suffix), state, state["indicator_params"])
        stage_seconds["streak_profile_and_indicators"] += time.perf_counter() - state_update_started

        flow_tail = state["flow_raw_tail"]
        flow_suffix = _archive_flow_range(
            e45_manifest,
            pd.Timestamp(state["last_processed_opened"]) + pd.Timedelta(minutes=1),
            pd.Timestamp(suffix["Opened"].iloc[-1]),
        )
        if not pd.DatetimeIndex(flow_suffix["Opened"]).equals(pd.DatetimeIndex(suffix["Opened"])):
            raise RuntimeError("raw FLOW archive suffix timestamps differ from the safe index/ohlcv source")
        if len(flow_suffix) != len(suffix):
            raise RuntimeError("FLOW raw suffix row count differs from index/ohlcv continuation")
        flow_rolling = pd.concat([flow_tail, flow_suffix], ignore_index=True)
        flow_started = time.perf_counter()
        flow_computed = _compute_flow_block(flow_rolling)
        stage_seconds["flow_formula"] += time.perf_counter() - flow_started
        flow_formula_rows += len(flow_rolling)
        flow_tail_rows_recomputed += len(flow_tail)
        flow_result = flow_computed.iloc[-len(suffix):].reset_index(drop=True)
        dynamic_rolling = pd.concat([state["dynamic_tail"], suffix], ignore_index=True)
        dynamic_started = time.perf_counter()
        dynamic = generate_dynamic_block(dynamic_state, dynamic_rolling,
                                         new_rows_start=len(dynamic_rolling) - len(suffix))
        stage_seconds["dynamic_update"] += time.perf_counter() - dynamic_started
        base_parts.append(base)
        flow_parts.append(flow_result)
        dynamic_parts.append(dynamic)
        opened_parts.append(pd.DatetimeIndex(suffix["Opened"]))

        state["source_row_index"] = int(boundary)
        state["source_byte_offset"] = byte_offset
        state["last_processed_opened"] = suffix["Opened"].iloc[-1].isoformat()
        state["next_expected_opened"] = (suffix["Opened"].iloc[-1] + pd.Timedelta(minutes=1)).isoformat()
        state["raw_tail"] = rolling_raw.tail(continuation.TAIL_ROWS).reset_index(drop=True)
        state["flow_raw_tail"] = flow_rolling.tail(FLOW_TAIL_ROWS).reset_index(drop=True)
        state["dynamic_tail"] = dynamic_rolling.tail(720).reset_index(drop=True)
        state["dynamic_state"] = _dynamic_checkpoint_payload(dynamic_state, temporary_root=RUNTIME / "dynamic_checkpoint_parts")
        state["recursive_last_output"] = {key: np.asarray(value)[-1].item() for key, value in recursive.items()}
        snapshot_name = snapshot_rows.get(int(boundary))
        if snapshot_name:
            snapshot_path = RUNTIME / "checkpoints" / f"{snapshot_name}.json"
            checkpoint_started = time.perf_counter()
            snapshot_records.append({
                "name": snapshot_name,
                "path": str(snapshot_path.relative_to(ROOT)),
                "sha256": continuation.atomic_checkpoint(snapshot_path, state),
                "semantic_sha256": continuation.state_digest(state),
                "bytes": snapshot_path.stat().st_size,
            })
            stage_seconds["checkpoint_write"] += time.perf_counter() - checkpoint_started

    base_all = pd.concat(base_parts, ignore_index=True)
    flow_all = pd.concat(flow_parts, ignore_index=True)
    dynamic_all = {name: np.concatenate([piece[name] for piece in dynamic_parts], axis=0)
                   for name in dynamic_parts[0]}
    opened_all = pd.DatetimeIndex(np.concatenate([part.as_unit("ns").asi8 for part in opened_parts])).as_unit("ns")
    output_started = time.perf_counter()
    outputs = _write_suffix_outputs(stage_name, base_all, flow_all, dynamic_all, opened_all)
    stage_seconds["output_write"] += time.perf_counter() - output_started
    if save_checkpoint_name:
        target_path = RUNTIME / "checkpoints" / f"{save_checkpoint_name}.json"
        checkpoint_started = time.perf_counter()
        outputs["saved_checkpoint"] = {"path": str(target_path.relative_to(ROOT)),
                                        "sha256": continuation.atomic_checkpoint(target_path, state),
                                        "bytes": target_path.stat().st_size,
                                        "semantic_sha256": continuation.state_digest(state)}
        stage_seconds["checkpoint_write"] += time.perf_counter() - checkpoint_started
    outputs["intermediate_checkpoints"] = snapshot_records
    outputs.update({"source_row_start": start_row, "source_row_end": stop_row,
                    "rows_recomputed": int(stop_row) - start_row, "first_suffix_opened": opened_all[0].isoformat(),
                    "last_suffix_opened": opened_all[-1].isoformat(),
                    "raw_bytes_read": int(state["source_byte_offset"]) - input_byte_offset,
                    "state_read_seconds": state_read_seconds,
                    "stage_seconds": stage_seconds,
                    "base64_builder_input_rows": base_builder_rows,
                    "base64_raw_tail_rows_recomputed": raw_tail_rows_recomputed,
                    "flow_formula_input_rows": flow_formula_rows,
                    "flow_raw_tail_rows_recomputed": flow_tail_rows_recomputed,
                    "flow_archive_io": dict(IO_COUNTS),
                    "csv_roundtrip_audit": list(CSV_ROUNDTRIP_AUDIT),
                    "wall_seconds": time.perf_counter() - started, "peak_rss_gib": _peak_rss_gib()})
    stats_path = RUNTIME / "suffixes" / f"{stage_name}_execution.json"
    _json_write(stats_path, outputs)
    return outputs


def _safe_parquet_columns(path: Path, columns: list[str], cutoff: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    opened_index = parquet.schema_arrow.get_field_index("Opened")
    if opened_index < 0 or not set(columns).issubset(parquet.schema_arrow.names):
        raise RuntimeError(f"required safe feature projection is missing from {path}")
    exclusive = pd.Timestamp(cutoff).tz_convert("UTC")
    for row_group in range(parquet.num_row_groups):
        stats = parquet.metadata.row_group(row_group).column(opened_index).statistics
        if stats is None or stats.max is None:
            raise RuntimeError(f"cannot prove the Opened cutoff for {path}")
        maximum = pd.Timestamp(stats.max)
        maximum = maximum.tz_localize("UTC") if maximum.tzinfo is None else maximum.tz_convert("UTC")
        if maximum >= exclusive:
            raise RuntimeError(f"Parquet row group crosses the exclusive safe cutoff: {path}")
    frame = pd.read_parquet(path, columns=columns)
    frame["Opened"] = pd.to_datetime(frame["Opened"], utc=True, errors="raise")
    return frame


def _load_flow_batch(e45_manifest: dict, cutoff: str) -> tuple[pd.DataFrame, list[str]]:
    records = e45_manifest["flow_source"]["batch_outputs"]
    frames, paths = [], []
    for record in records:
        path = _require_hash(record["path"], record["sha256"], f"E0045 FLOW6 batch {record['month']}")
        frame = _safe_parquet_columns(path, ["Opened", *FLOW_COLUMNS], cutoff)
        frames.append(frame)
        paths.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    flow = pd.concat(frames, ignore_index=True).sort_values("Opened", kind="stable", ignore_index=True)
    if flow["Opened"].duplicated().any() or not flow["Opened"].is_monotonic_increasing:
        raise RuntimeError("E0045 FLOW6 batch rows are not uniquely ordered")
    return flow, paths


def _mismatch_mask(actual, expected, *, atol: float, rtol: float) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    mismatch = (np.isnan(actual) != np.isnan(expected))
    mismatch |= (np.isposinf(actual) != np.isposinf(expected))
    mismatch |= (np.isneginf(actual) != np.isneginf(expected))
    finite = np.isfinite(actual) & np.isfinite(expected)
    mismatch[finite] |= np.abs(actual[finite] - expected[finite]) > (atol + rtol * np.abs(expected[finite]))
    return mismatch


def _parity_record(block: str, feature: str, actual, expected, opened,
                   *, atol: float, rtol: float, exact: bool = False) -> dict:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    opened = pd.DatetimeIndex(pd.to_datetime(opened, utc=True, errors="raise")).as_unit("ns")
    if actual.shape != expected.shape or actual.shape[0] != len(opened):
        return {"block": block, "feature": feature, "status": "FAIL_SHAPE", "rows": len(opened),
                "numeric_mismatch_rows": 0, "mask_mismatch_rows": 0, "first_mismatch_opened_utc": "",
                "max_abs_error": None, "max_rel_error": None, "atol": atol, "rtol": rtol}
    actual64 = actual.astype(np.float64, copy=False)
    expected64 = expected.astype(np.float64, copy=False)
    mismatches = _mismatch_mask(actual64, expected64, atol=atol, rtol=rtol)
    finite = np.isfinite(actual64) & np.isfinite(expected64)
    abs_error = np.zeros(actual64.shape, dtype=np.float64)
    abs_error[finite] = np.abs(actual64[finite] - expected64[finite])
    rel_error = np.zeros(actual64.shape, dtype=np.float64)
    denom = np.abs(expected64[finite])
    rel_error[finite] = np.divide(abs_error[finite], denom, out=np.zeros_like(denom), where=denom != 0.0)
    mask_mismatch = (np.isnan(actual64) != np.isnan(expected64)) | (np.isposinf(actual64) != np.isposinf(expected64)) | (np.isneginf(actual64) != np.isneginf(expected64))
    mismatch_positions = np.flatnonzero(np.any(mismatches.reshape((len(opened), -1)), axis=1)) if len(opened) else np.empty(0, dtype=np.int64)
    return {
        "block": block,
        "feature": feature,
        "status": "PASS" if not mismatches.any() else "FAIL",
        "rows": int(len(opened)),
        "numeric_mismatch_rows": int(np.count_nonzero(mismatches & finite)),
        "mask_mismatch_rows": int(mask_mismatch.sum()),
        "nan_mask_mismatch_rows": int((np.isnan(actual64) != np.isnan(expected64)).sum()),
        "posinf_mask_mismatch_rows": int((np.isposinf(actual64) != np.isposinf(expected64)).sum()),
        "neginf_mask_mismatch_rows": int((np.isneginf(actual64) != np.isneginf(expected64)).sum()),
        "first_mismatch_opened_utc": opened[int(mismatch_positions[0])].isoformat() if len(mismatch_positions) else "",
        "max_abs_error": float(abs_error[finite].max()) if finite.any() else 0.0,
        "max_rel_error": float(rel_error[finite].max()) if finite.any() else 0.0,
        "atol": 0.0 if exact else float(atol),
        "rtol": 0.0 if exact else float(rtol),
    }


def _compare_flow_representation(protocol: dict, e45_protocol: dict, e45_manifest: dict,
                                 flow_batch: pd.DataFrame) -> tuple[list[dict], dict]:
    reference_last = pd.Timestamp(e45_manifest["flow"]["last_reference_opened"]).tz_convert("UTC")
    canonical_record = e45_protocol["allowed_inputs"]["canonical_model_feature_reference"]
    canonical_path = _require_hash(canonical_record["path"], canonical_record["sha256"], "E0041 final BASE70 feature artifact")
    canonical = _safe_parquet_columns(canonical_path, ["Opened", *FLOW_COLUMNS], protocol["scope"]["source_raw_opened_exclusive_utc"])
    common = canonical.loc[canonical["Opened"] <= reference_last].reset_index(drop=True)
    raw_common = flow_batch.loc[flow_batch["Opened"].isin(common["Opened"])].reset_index(drop=True)
    if len(common) != len(raw_common) or not pd.DatetimeIndex(common["Opened"]).as_unit("ns").equals(pd.DatetimeIndex(raw_common["Opened"]).as_unit("ns")):
        raise RuntimeError("FLOW representation comparison keys are not a one-to-one exact timestamp join")
    cast, matched = continuation.historical_flow_join_cast(
        common["Opened"], flow_batch["Opened"], flow_batch.loc[:, FLOW_COLUMNS].to_numpy(dtype=np.float64),
    )
    if not matched.all():
        raise RuntimeError("historical FLOW cast unexpectedly left unmatched canonical development rows")
    canonical_values = common.loc[:, FLOW_COLUMNS].to_numpy()
    rows = []
    expected_key_ns = pd.DatetimeIndex(common["Opened"]).as_unit("ns").asi8
    key_digest = hashlib.sha256(expected_key_ns.tobytes()).hexdigest()
    raw_mismatches = 0
    raw_max_abs = 0.0
    raw_max_rel = 0.0
    for index, name in enumerate(FLOW_COLUMNS):
        actual64 = raw_common[name].to_numpy(dtype=np.float64)
        expected = canonical_values[:, index]
        after = cast[:, index]
        record = _parity_record("flow_representation_float32", name, after, expected, common["Opened"], atol=0.0, rtol=0.0, exact=True)
        record.update({"keys_first_opened_utc": common["Opened"].iloc[0].isoformat(),
                       "keys_last_opened_utc": common["Opened"].iloc[-1].isoformat(),
                       "keys_sha256": key_digest})
        rows.append(record)
        before = _parity_record("flow_representation_float64_diagnostic", name, actual64, expected, common["Opened"], atol=0.0, rtol=0.0, exact=True)
        before.update({"keys_first_opened_utc": common["Opened"].iloc[0].isoformat(),
                       "keys_last_opened_utc": common["Opened"].iloc[-1].isoformat(),
                       "keys_sha256": key_digest})
        before["status"] = "MEASURED_FLOAT32_PRECISION_LOSS" if before["numeric_mismatch_rows"] or before["mask_mismatch_rows"] else "EXACT"
        rows.append(before)
        raw_mismatches += before["numeric_mismatch_rows"] + before["mask_mismatch_rows"]
        raw_max_abs = max(raw_max_abs, float(before["max_abs_error"] or 0.0))
        raw_max_rel = max(raw_max_rel, float(before["max_rel_error"] or 0.0))
    last_allowed = pd.Timestamp(protocol["scope"]["last_allowed_raw_opened_utc"]).tz_convert("UTC")
    unmatched_after_reference = flow_batch.loc[(flow_batch["Opened"] > reference_last) & (flow_batch["Opened"] <= last_allowed)]
    cast_summary = _aggregate_flow_cast_records(rows, FLOW_COLUMNS, common["Opened"])
    diagnostic_rows = [row for row in rows if row["block"] == "flow_representation_float64_diagnostic"]
    return rows, {
        "common_reference_rows": int(len(common)),
        "reference_last_opened": reference_last.isoformat(),
        "last_allowed_raw_opened": last_allowed.isoformat(),
        "post_reference_flow_rows": int(len(unmatched_after_reference)),
        "post_reference_first_opened": unmatched_after_reference["Opened"].iloc[0].isoformat() if len(unmatched_after_reference) else "",
        "post_reference_last_opened": unmatched_after_reference["Opened"].iloc[-1].isoformat() if len(unmatched_after_reference) else "",
        "float64_diagnostic_differences": int(sum(row["numeric_mismatch_rows"] + row["mask_mismatch_rows"] for row in diagnostic_rows)),
        "float64_vs_final_product_mismatch_feature_rows": int(raw_mismatches),
        "float64_vs_final_product_max_abs_error": raw_max_abs,
        "float64_vs_final_product_max_rel_error": raw_max_rel,
        "float64_diagnostic_rows": len(diagnostic_rows),
        "float32_exact_mismatches": int(cast_summary["mismatch_rows"]),
        "float32_exact_status": cast_summary["status"],
        "float32_feature_coverage": cast_summary["feature_coverage"],
        "float32_rows_per_feature": cast_summary["rows_per_feature"],
        "canonical_flow_rows_in_feature_artifact": int(len(canonical)),
        "e0041_flow_formula_rows_reused_from_E0045": int(e45_manifest["flow"]["rows_compared"]),
    }


def _aggregate_flow_cast_records(rows: list[dict], expected_features, expected_opened) -> dict:
    expected_features = tuple(expected_features)
    expected_ns = pd.DatetimeIndex(expected_opened).as_unit("ns").asi8
    grouped = {name: [] for name in expected_features}
    unknown = []
    for row in rows:
        if row.get("block") != "flow_representation_float32":
            continue
        feature = row.get("feature")
        if feature not in grouped:
            unknown.append(feature)
        else:
            grouped[feature].append(row)
    valid = len(expected_ns) > 0 and not unknown
    mismatch_rows = 0
    row_totals = {}
    for feature in expected_features:
        records = grouped[feature]
        cursor = 0
        total = 0
        seen_ranges = set()
        if not records:
            valid = False
        for row in sorted(records, key=lambda item: item.get("keys_first_opened_utc", "")):
            try:
                start_ns = pd.Timestamp(row["keys_first_opened_utc"]).tz_convert("UTC").value
                end_ns = pd.Timestamp(row["keys_last_opened_utc"]).tz_convert("UTC").value
            except (KeyError, TypeError, ValueError):
                valid = False
                continue
            start = int(np.searchsorted(expected_ns, start_ns, side="left"))
            stop = int(np.searchsorted(expected_ns, end_ns, side="right"))
            identity = (start, stop)
            if identity in seen_ranges or start != cursor or stop <= start or start < 0 or stop > len(expected_ns):
                valid = False
                continue
            seen_ranges.add(identity)
            expected_slice = expected_ns[start:stop]
            expected_digest = hashlib.sha256(expected_slice.tobytes()).hexdigest()
            if (expected_ns[start] != start_ns or expected_ns[stop - 1] != end_ns
                    or len(expected_slice) != int(row.get("rows", -1))
                    or row.get("keys_sha256") != expected_digest):
                valid = False
            cursor = max(cursor, stop)
            total += int(row.get("rows", 0))
            mismatch_rows += int(row.get("numeric_mismatch_rows", 0)) + int(row.get("mask_mismatch_rows", 0))
            if row.get("status") != "PASS":
                valid = False
        row_totals[feature] = total
        if cursor != len(expected_ns) or total != len(expected_ns):
            valid = False
    return {"status": "PASS" if valid and mismatch_rows == 0 else "FAIL",
            "valid_coverage": bool(valid), "mismatch_rows": int(mismatch_rows),
            "feature_coverage": sorted(feature for feature, records in grouped.items() if records),
            "rows_per_feature": row_totals, "unknown_features": unknown}


def _reconcile_e0046_flow_csv(rows: pd.DataFrame, prior_manifest: dict) -> dict:
    expected_features = set(FLOW_COLUMNS)
    cast = rows.loc[rows["block"] == "flow_representation_float32"].copy()
    diagnostics = rows.loc[rows["block"] == "flow_representation_float64_diagnostic"].copy()
    features = cast["feature"].tolist()
    complete = len(features) == len(FLOW_COLUMNS) and set(features) == expected_features and len(set(features)) == len(FLOW_COLUMNS)
    if complete:
        complete = (cast["rows"].astype(int).eq(int(prior_manifest["flow_representation"]["common_reference_rows"])).all()
                    and cast["status"].eq("PASS").all())
    cast_mismatches = int((cast["numeric_mismatch_rows"].astype(int) + cast["mask_mismatch_rows"].astype(int)).sum())
    diagnostic_mismatches = int((diagnostics["numeric_mismatch_rows"].astype(int) + diagnostics["mask_mismatch_rows"].astype(int)).sum())
    prior_aggregate = int(prior_manifest["flow_representation"]["float32_exact_mismatches"])
    return {"status": "PASS" if complete else "FAIL",
            "postcast_mismatch_rows_recomputed_from_E0046_csv": cast_mismatches,
            "postcast_features": features,
            "postcast_rows_per_feature": cast["rows"].astype(int).tolist(),
            "diagnostic_float64_mismatch_rows": diagnostic_mismatches,
            "previously_reported_aggregate": prior_aggregate,
            "previous_aggregate_matches_diagnostic_only": prior_aggregate == diagnostic_mismatches,
            "csv_rows": int(len(rows)), "complete_six_feature_coverage": bool(complete)}


def _load_prior_e0046_flow_evidence(protocol: dict) -> dict:
    record = protocol["parent"]["E0046_parity_csv"]
    manifest_record = protocol["parent"]["E0046_manifest"]
    csv_path = _require_hash(record["path"], record["sha256"], "immutable E0046 parity CSV")
    manifest_path = _require_hash(manifest_record["path"], manifest_record["sha256"], "immutable E0046 manifest")
    prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (prior_manifest.get("experiment_id") != "E0046" or prior_manifest.get("status") != "DEV_GENERATOR_PARTIAL"
            or prior_manifest.get("evaluation_authorized") is not False
            or prior_manifest.get("activated") is not False
            or prior_manifest.get("holdout_status") != "HOLDOUT_NOT_RUN"):
        raise RuntimeError("preserved E0046 manifest identity or holdout guards changed")
    rows = pd.read_csv(csv_path)
    result = _reconcile_e0046_flow_csv(rows, prior_manifest)
    if not result["complete_six_feature_coverage"]:
        raise RuntimeError("hash-verified E0046 CSV does not contain a complete postcast FLOW block")
    return result


def _compare_suffix_stage(stage: str, execution: dict, e45_protocol: dict,
                          e45_manifest: dict, d_manifest: dict, flow_batch: pd.DataFrame) -> list[dict]:
    from features import btc_D_input_generator_20261029 as e45_code

    base_path = ROOT / execution["base64"]["path"]
    flow_path = ROOT / execution["flow6"]["path"]
    dyn_path = ROOT / execution["dynamic"]["path"]
    base_actual = pd.read_parquet(base_path)
    flow_actual = pd.read_parquet(flow_path)
    dynamic_actual = pd.read_parquet(dyn_path)
    base_order = d_manifest["base_feature_order"][:64]
    dynamic_reference = e45_protocol["allowed_inputs"]["canonical_dynamic_reference"]
    dynamic_path = _require_hash(dynamic_reference["path"], dynamic_reference["sha256"], "E0041 dynamic feature reference")
    base_reference_path = Path(e45_manifest["base64_build"]["output_path"])
    base_reference_path = _require_hash(str(base_reference_path), e45_manifest["base64_build"]["output_sha256"], "E0045 batch BASE64 reference")
    start_row = int(execution["source_row_start"])
    row_count = int(execution["rows_recomputed"])
    cutoff = e45_protocol["scope"]["source_raw_opened_exclusive_utc"]
    base_reader = e45_code._ParquetRowReader(base_reference_path, ["Opened", *base_order], cutoff, start_row=start_row)
    base_expected = base_reader.read(row_count)
    dynamic_names = list(d_manifest["memory_feature_order"])
    dynamic_names.extend(name for seed in d_manifest["reservoir_seeds"] for name in d_manifest["esn_feature_order_by_seed"][str(seed)])
    dynamic_reader = e45_code._ParquetRowReader(dynamic_path, ["Opened", *dynamic_names], cutoff, start_row=start_row)
    dynamic_expected = dynamic_reader.read(row_count)
    opened = pd.DatetimeIndex(pd.to_datetime(base_actual["Opened"], utc=True, errors="raise")).as_unit("ns")
    if not opened.equals(pd.DatetimeIndex(pd.to_datetime(base_expected["Opened"], utc=True, errors="raise")).as_unit("ns")):
        raise RuntimeError(f"BASE64 continuation exact row keys differ from batch reference in {stage}")
    if not opened.equals(pd.DatetimeIndex(pd.to_datetime(flow_actual["Opened"], utc=True, errors="raise")).as_unit("ns")):
        raise RuntimeError(f"FLOW6 continuation exact row keys differ from BASE64 rows in {stage}")
    if not opened.equals(pd.DatetimeIndex(pd.to_datetime(dynamic_actual["Opened"], utc=True, errors="raise")).as_unit("ns")):
        raise RuntimeError(f"dynamic continuation exact row keys differ from BASE64 rows in {stage}")
    rows = []
    for name in base_order:
        rows.append(_parity_record(f"base64_continuation_{stage}", name, base_actual[name].to_numpy(),
                                   base_expected[name].to_numpy(), opened, atol=1e-10, rtol=1e-10))
    flow_index = pd.DatetimeIndex(flow_batch["Opened"]).as_unit("ns")
    flow_lookup = np.searchsorted(flow_index.asi8, opened.asi8)
    flow_match = flow_lookup < len(flow_index)
    valid = np.flatnonzero(flow_match)
    flow_match[valid] = flow_index.asi8[flow_lookup[valid]] == opened.asi8[valid]
    if not flow_match.all():
        raise RuntimeError(f"E0045 FLOW6 batch comparison is missing continuation rows in {stage}")
    flow_expected = flow_batch.iloc[flow_lookup].reset_index(drop=True)
    for name in FLOW_COLUMNS:
        rows.append(_parity_record(f"flow6_continuation_{stage}", name, flow_actual[name].to_numpy(),
                                   flow_expected[name].to_numpy(), opened, atol=1e-10, rtol=1e-10))
    for name in d_manifest["memory_feature_order"]:
        rows.append(_parity_record(f"memory_continuation_{stage}", name, dynamic_actual[f"memory_{d_manifest['memory_feature_order'].index(name):02d}"].to_numpy(),
                                   dynamic_expected[name].to_numpy(), opened, atol=2e-6, rtol=2e-6))
    for seed in d_manifest["reservoir_seeds"]:
        for index, name in enumerate(d_manifest["esn_feature_order_by_seed"][str(seed)]):
            actual_name = f"esn_{seed}_{index:02d}"
            rows.append(_parity_record(f"esn_continuation_{stage}_r{seed}", name,
                                       dynamic_actual[actual_name].to_numpy(), dynamic_expected[name].to_numpy(),
                                       opened, atol=2e-6, rtol=2e-6))
    return rows


def _load_rows_by_key(path: Path, columns: list[str], opened: pd.DatetimeIndex,
                      cutoff: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    wanted = pd.DatetimeIndex(opened).as_unit("ns")
    opened_idx = parquet.schema_arrow.get_field_index("Opened")
    selected = []
    for group in range(parquet.num_row_groups):
        stats = parquet.metadata.row_group(group).column(opened_idx).statistics
        if stats is None or stats.min is None or stats.max is None:
            raise RuntimeError(f"cannot prove row-group time range for {path}")
        lo = pd.Timestamp(stats.min); hi = pd.Timestamp(stats.max)
        lo = lo.tz_localize("UTC") if lo.tzinfo is None else lo.tz_convert("UTC")
        hi = hi.tz_localize("UTC") if hi.tzinfo is None else hi.tz_convert("UTC")
        if hi >= pd.Timestamp(cutoff).tz_convert("UTC"):
            raise RuntimeError(f"row group crosses safe cutoff while reading {path}")
        if hi >= wanted[0] and lo <= wanted[-1]:
            table = parquet.read_row_group(group, columns=columns)
            selected.append(table.to_pandas())
    if not selected:
        return pd.DataFrame(columns=columns)
    result = pd.concat(selected, ignore_index=True)
    result["Opened"] = pd.to_datetime(result["Opened"], utc=True, errors="raise")
    result = result.loc[result["Opened"].isin(wanted)].sort_values("Opened", kind="stable", ignore_index=True)
    if not pd.DatetimeIndex(result["Opened"]).as_unit("ns").equals(wanted):
        raise RuntimeError(f"row-key selection did not return all exact requested rows from {path}")
    return result


def _compare_model_matrices_and_predictions(protocol: dict, e45_protocol: dict,
                                            e45_manifest: dict, d_bundle: dict,
                                            flow_batch: pd.DataFrame) -> tuple[list[dict], dict]:
    from features import btc_D_input_generator_20261029 as e45_code

    d_manifest = d_bundle["manifest"]
    base_order = list(d_manifest["base_feature_order"])
    raw_order = base_order[:64]
    memory_order = list(d_manifest["memory_feature_order"])
    reservoir_seeds = list(d_manifest["reservoir_seeds"])
    source = e45_protocol["allowed_inputs"]
    base_path = _require_hash(e45_manifest["base64_build"]["output_path"], e45_manifest["base64_build"]["output_sha256"], "E0045 raw BASE64 batch product")
    canonical_path = _require_hash(source["canonical_model_feature_reference"]["path"], source["canonical_model_feature_reference"]["sha256"], "E0041 canonical BASE70")
    dynamic_path = _require_hash(source["canonical_dynamic_reference"]["path"], source["canonical_dynamic_reference"]["sha256"], "E0041 dynamic feature reference")
    cutoff = protocol["scope"]["source_raw_opened_exclusive_utc"]
    base_reader = e45_code._ParquetRowReader(base_path, ["Opened", *raw_order], cutoff)
    canonical_reader = e45_code._ParquetRowReader(canonical_path, ["Opened", *base_order], cutoff)
    dynamic_names = [*memory_order]
    for seed in reservoir_seeds:
        dynamic_names.extend(d_manifest["esn_feature_order_by_seed"][str(seed)])
    dynamic_reader = e45_code._ParquetRowReader(dynamic_path, ["Opened", *dynamic_names], cutoff)
    model_cutoff = pd.Timestamp(protocol["scope"]["last_allowed_model_feature_opened_utc"]).tz_convert("UTC")
    expected_rows = int(protocol["scope"]["expected_common_model_rows"])
    chunk_rows = 65_536
    panel_path = _require_hash(protocol["parent"]["E0045_final_model_panel_path"], protocol["parent"]["E0045_final_model_panel_sha256"], "E0045 frozen development prediction panel")
    panel_doc = json.loads(panel_path.read_text(encoding="utf-8"))
    panel_rows = panel_doc["all_selected_rows"]
    panel_opened = pd.DatetimeIndex(pd.to_datetime([row["Opened"] for row in panel_rows], utc=True, errors="raise")).as_unit("ns")
    if len(panel_rows) != int(e45_manifest["prediction_status"]["panel_rows"] if "prediction_status" in e45_manifest else panel_doc["rows"]):
        if len(panel_rows) != int(panel_doc["rows"]):
            raise RuntimeError("E0045 frozen development panel row count changed")
    if any(pd.Timestamp(row["market_start_utc"]) >= pd.Timestamp("2026-07-01T00:00:00Z") for row in panel_rows):
        raise RuntimeError("E0045 development panel unexpectedly crosses the holdout market boundary")
    if (panel_opened > model_cutoff).any():
        raise RuntimeError("E0045 prediction panel includes post-cutoff model features")
    panel_ns = panel_opened.asi8
    flow_opened = pd.DatetimeIndex(flow_batch["Opened"]).as_unit("ns")
    flow_values = flow_batch.loc[:, FLOW_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    matrix_rows = []
    panel_raw_matrices = {seed: [] for seed in reservoir_seeds}
    panel_canonical_matrices = {seed: [] for seed in reservoir_seeds}
    panel_captured = 0
    processed = 0
    first_matrix_opened = None
    last_matrix_opened = None
    matrix_started = time.perf_counter()
    while processed < expected_rows:
        count = min(chunk_rows, expected_rows - processed)
        base = base_reader.read(count)
        canonical = canonical_reader.read(count)
        dynamic = dynamic_reader.read(count)
        if any(len(frame) != count for frame in (base, canonical, dynamic)):
            raise RuntimeError("canonical matrix inputs ended before the frozen common development row count")
        opened = pd.DatetimeIndex(pd.to_datetime(base["Opened"], utc=True, errors="raise")).as_unit("ns")
        for frame in (canonical, dynamic):
            other = pd.DatetimeIndex(pd.to_datetime(frame["Opened"], utc=True, errors="raise")).as_unit("ns")
            if not other.equals(opened):
                raise RuntimeError("canonical matrix sources have different exact Opened keys")
        if opened[-1] > model_cutoff:
            raise RuntimeError("model matrix comparison crossed the last permitted feature Opened")
        if first_matrix_opened is None:
            first_matrix_opened = opened[0]
        last_matrix_opened = opened[-1]
        raw_base = np.full((count, len(base_order)), np.nan, dtype=np.float32)
        raw_base[:, :64] = base.loc[:, raw_order].to_numpy(dtype=np.float32)
        aligned_flow, matched = continuation.historical_flow_join_cast(opened, flow_opened, flow_values)
        if not matched.all():
            raise RuntimeError("historically joined FLOW6 is missing a common model-matrix timestamp")
        raw_base[:, 64:] = aligned_flow
        canonical_base = canonical.loc[:, base_order].to_numpy(dtype=np.float32, copy=True)
        memory = dynamic.loc[:, memory_order].to_numpy(dtype=np.float32, copy=True)
        panel_index = np.searchsorted(panel_ns, opened.asi8)
        panel_match = panel_index < len(panel_ns)
        in_range = np.flatnonzero(panel_match)
        panel_match[in_range] = panel_ns[panel_index[in_range]] == opened.asi8[in_range]
        for seed in reservoir_seeds:
            esn_order = list(d_manifest["esn_feature_order_by_seed"][str(seed)])
            esn = dynamic.loc[:, esn_order].to_numpy(dtype=np.float32, copy=True)
            actual_matrix = np.concatenate((raw_base, memory, esn), axis=1).astype(np.float32, copy=False)
            expected_matrix = np.concatenate((canonical_base, memory, esn), axis=1).astype(np.float32, copy=False)
            actual_matrix[~np.isfinite(actual_matrix)] = np.nan
            expected_matrix[~np.isfinite(expected_matrix)] = np.nan
            if actual_matrix.shape[1] != 86 or expected_matrix.shape[1] != 86:
                raise RuntimeError("final E0041 D member matrix width changed from 86")
            for column, name in enumerate([*base_order, *memory_order, *esn_order]):
                matrix_row = _parity_record(f"model_matrix_r{seed}", name,
                                            actual_matrix[:, column], expected_matrix[:, column],
                                            opened, atol=0, rtol=0, exact=True)
                matrix_row.update({"key_start_index": int(processed), "key_stop_index": int(processed + count),
                                   "keys_sha256": hashlib.sha256(opened.asi8.tobytes()).hexdigest(),
                                   "keys_first_opened_utc": opened[0].isoformat(), "keys_last_opened_utc": opened[-1].isoformat()})
                matrix_rows.append(matrix_row)
            if panel_match.any():
                positions = np.flatnonzero(panel_match)
                panel_raw_matrices[seed].append(actual_matrix[positions].copy())
                panel_canonical_matrices[seed].append(expected_matrix[positions].copy())
                panel_captured += int(len(positions)) if seed == reservoir_seeds[0] else 0
        processed += count
    if processed != expected_rows or base_reader.row < expected_rows:
        raise RuntimeError("matrix comparison did not cover the frozen exact common row count")
    if panel_captured != len(panel_rows):
        raise RuntimeError(f"only {panel_captured} of {len(panel_rows)} exact E0045 panel keys were captured")
    expected_first_matrix_opened = pd.Timestamp("2020-06-09T09:33:00Z")
    if first_matrix_opened != expected_first_matrix_opened or last_matrix_opened != model_cutoff:
        raise RuntimeError("common model matrix Opened endpoints differ from the frozen safe development bounds")

    panel_raw = {seed: np.concatenate(parts, axis=0) for seed, parts in panel_raw_matrices.items()}
    panel_canonical = {seed: np.concatenate(parts, axis=0) for seed, parts in panel_canonical_matrices.items()}
    matrix_comparison_seconds = time.perf_counter() - matrix_started
    prediction_started = time.perf_counter()
    try:
        import lightgbm as lgb
        from scipy.special import expit
    except ImportError as exc:
        raise RuntimeError("frozen LightGBM/Platt prediction dependencies are unavailable") from exc
    raw_members, canonical_members = [], []
    prediction_rows = []
    for member in d_manifest["members"]:
        seed = int(member["reservoir_seed"])
        booster = lgb.Booster(model_file=str(member["model_path"]))
        if booster.num_feature() != 86:
            raise RuntimeError(f"saved E0041 member feature count changed: {member['model_path']}")
        raw_probability = np.asarray(booster.predict(panel_raw[seed]), dtype=np.float64)
        canonical_probability = np.asarray(booster.predict(panel_canonical[seed]), dtype=np.float64)
        raw_members.append(raw_probability)
        canonical_members.append(canonical_probability)
        expected_raw = np.asarray([row["raw_member_probabilities"][len(raw_members) - 1] for row in panel_rows], dtype=np.float64)
        expected_canonical = np.asarray([row["canonical_member_probabilities"][len(canonical_members) - 1] for row in panel_rows], dtype=np.float64)
        prediction_rows.append(_parity_record("final_D_member_prediction", f"r{seed}_l{member['lightgbm_seed']}",
                                              raw_probability, expected_raw, panel_opened, atol=2e-12, rtol=2e-12))
        prediction_rows.append(_parity_record("canonical_D_member_prediction", f"r{seed}_l{member['lightgbm_seed']}",
                                              canonical_probability, expected_canonical, panel_opened, atol=2e-12, rtol=2e-12))
    raw_mean = np.mean(np.vstack(raw_members), axis=0, dtype=np.float64)
    canonical_mean = np.mean(np.vstack(canonical_members), axis=0, dtype=np.float64)
    expected_raw_mean = np.asarray([row["raw_D_uniform_probability"] for row in panel_rows], dtype=np.float64)
    expected_canonical_mean = np.asarray([row["canonical_D_uniform_probability"] for row in panel_rows], dtype=np.float64)
    calibrator = d_manifest["calibrator_doc"]

    def apply_platt(values):
        clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped))
        return expit(float(calibrator["intercept"]) + float(calibrator["coefficient"]) * logits)

    raw_calibrated = apply_platt(raw_mean)
    canonical_calibrated = apply_platt(canonical_mean)
    expected_raw_cal = np.asarray([row["raw_D_terminal_Platt_probability"] for row in panel_rows], dtype=np.float64)
    expected_canonical_cal = np.asarray([row["canonical_D_terminal_Platt_probability"] for row in panel_rows], dtype=np.float64)
    prediction_rows.extend([
        _parity_record("final_D_uniform_mean", "uniform_1_over_9", raw_mean, expected_raw_mean, panel_opened, atol=2e-12, rtol=2e-12),
        _parity_record("canonical_D_uniform_mean", "uniform_1_over_9", canonical_mean, expected_canonical_mean, panel_opened, atol=2e-12, rtol=2e-12),
        _parity_record("final_D_terminal_Platt", "saved_Platt", raw_calibrated, expected_raw_cal, panel_opened, atol=2e-12, rtol=2e-12),
        _parity_record("canonical_D_terminal_Platt", "saved_Platt", canonical_calibrated, expected_canonical_cal, panel_opened, atol=2e-12, rtol=2e-12),
    ])
    matrix_rows.extend(prediction_rows)
    finalization_start = pd.Timestamp("2026-06-30T22:50:00Z")
    finalization_end = pd.Timestamp("2026-06-30T23:53:00Z")
    finalization_rows = int(((panel_opened >= finalization_start) & (panel_opened <= finalization_end)).sum())
    summary = {
        "rows_compared_per_seed": processed,
        "first_opened": first_matrix_opened.isoformat(),
        "last_opened": model_cutoff.isoformat(),
        "reservoir_seeds": reservoir_seeds,
        "feature_count_per_matrix": 86,
        "matrix_count": len(reservoir_seeds),
        "panel_rows": len(panel_rows),
        "decision_panel_rows": int(panel_doc["decision_rows_count"]),
        "finalization_panel_rows_2026_06_30_2250_to_2353": finalization_rows,
        "prediction_statuses": {row["feature"]: row["status"] for row in prediction_rows},
        "matrix_comparison_seconds": matrix_comparison_seconds,
        "prediction_seconds": time.perf_counter() - prediction_started,
    }
    coverage = _aggregate_partition_coverage(
        matrix_rows,
        {f"model_matrix_r{seed}": [*base_order, *memory_order, *d_manifest["esn_feature_order_by_seed"][str(seed)]]
         for seed in reservoir_seeds},
        expected_rows,
    )
    summary["coverage"] = coverage
    return matrix_rows, summary


def _resumed_matrix_parity_rows(actual_matrix: np.ndarray, canonical_matrix: np.ndarray,
                               opened: pd.DatetimeIndex, reservoir_seed: int,
                               stage: str, column_names: list[str]) -> list[dict]:
    if actual_matrix.shape != canonical_matrix.shape or actual_matrix.shape[1] != len(column_names):
        return [{"block": f"resumed_model_matrix_r{reservoir_seed}_{stage}", "feature": "matrix_shape",
                 "status": "FAIL_SHAPE", "rows": len(opened), "numeric_mismatch_rows": 0,
                 "mask_mismatch_rows": 0, "first_mismatch_opened_utc": "",
                 "max_abs_error": None, "max_rel_error": None, "atol": 0.0, "rtol": 0.0}]
    digest = hashlib.sha256(pd.DatetimeIndex(opened).as_unit("ns").asi8.tobytes()).hexdigest()
    rows = [_parity_record(f"resumed_model_matrix_r{reservoir_seed}_{stage}", name,
                           actual_matrix[:, index], canonical_matrix[:, index], opened,
                           atol=0.0, rtol=0.0, exact=True)
            for index, name in enumerate(column_names)]
    for row in rows:
        row.update({"key_start_index": 0, "key_stop_index": len(opened), "keys_sha256": digest,
                    "keys_first_opened_utc": opened[0].isoformat() if len(opened) else "",
                    "keys_last_opened_utc": opened[-1].isoformat() if len(opened) else ""})
    return rows


def _same_opened_keys(left, right) -> bool:
    left_opened = pd.DatetimeIndex(pd.to_datetime(left, utc=True, errors="raise")).as_unit("ns")
    right_opened = pd.DatetimeIndex(pd.to_datetime(right, utc=True, errors="raise")).as_unit("ns")
    return len(left_opened) == len(right_opened) and np.array_equal(left_opened.asi8, right_opened.asi8)


def _aggregate_partition_coverage(rows: list[dict], block_features: dict[str, list[str]], expected_rows: int) -> dict:
    valid = expected_rows > 0
    details = {}
    expected_blocks = set(block_features)
    found_blocks = {row.get("block") for row in rows if row.get("block") in expected_blocks}
    if found_blocks != expected_blocks:
        valid = False
    for block, expected_features in block_features.items():
        selected = [row for row in rows if row.get("block") == block]
        actual_features = {row.get("feature") for row in selected}
        block_valid = actual_features == set(expected_features) and len(selected) > 0
        for feature in expected_features:
            parts = [row for row in selected if row.get("feature") == feature]
            cursor = 0
            seen = set()
            for row in sorted(parts, key=lambda item: int(item.get("key_start_index", -1))):
                start = int(row.get("key_start_index", -1))
                stop = int(row.get("key_stop_index", -1))
                identity = (start, stop)
                if identity in seen or start != cursor or stop <= start or stop - start != int(row.get("rows", -1)):
                    block_valid = False
                seen.add(identity)
                if not row.get("keys_sha256") or not row.get("keys_first_opened_utc") or not row.get("keys_last_opened_utc"):
                    block_valid = False
                cursor = max(cursor, stop)
                if row.get("status") != "PASS":
                    block_valid = False
            if cursor != expected_rows or sum(int(row.get("rows", 0)) for row in parts) != expected_rows:
                block_valid = False
        partitions_by_feature = {}
        for feature in expected_features:
            partitions_by_feature[feature] = sorted(
                (int(row.get("key_start_index", -1)), int(row.get("key_stop_index", -1)), row.get("keys_sha256"))
                for row in selected if row.get("feature") == feature
            )
        if partitions_by_feature:
            reference_partitions = next(iter(partitions_by_feature.values()))
            if any(parts != reference_partitions for parts in partitions_by_feature.values()):
                block_valid = False
        details[block] = {"status": "PASS" if block_valid else "FAIL",
                          "feature_count": len(actual_features),
                          "records": len(selected), "rows_per_feature": {
                              feature: sum(int(row.get("rows", 0)) for row in selected if row.get("feature") == feature)
                              for feature in expected_features}}
        valid = valid and block_valid
    return {"status": "PASS" if valid else "FAIL", "valid": bool(valid), "expected_rows_per_feature": expected_rows,
            "blocks": details}


def _compare_resumed_suffix_matrices_and_predictions(protocol: dict, e45_protocol: dict,
                                                      e45_manifest: dict, d_bundle: dict,
                                                      executions: dict) -> tuple[list[dict], dict]:
    from features import btc_D_input_generator_20261029 as e45_code
    from scipy.special import expit
    import lightgbm as lgb

    d_manifest = d_bundle["manifest"]
    source = e45_protocol["allowed_inputs"]
    base_order = list(d_manifest["base_feature_order"])
    raw_order = base_order[:64]
    memory_order = list(d_manifest["memory_feature_order"])
    seeds = list(d_manifest["reservoir_seeds"])
    canonical_path = _require_hash(source["canonical_model_feature_reference"]["path"],
                                   source["canonical_model_feature_reference"]["sha256"],
                                   "E0041 canonical BASE70 resumed-matrix reference")
    dynamic_reference = source["canonical_dynamic_reference"]
    canonical_dynamic_path = _require_hash(dynamic_reference["path"], dynamic_reference["sha256"],
                                           "E0041 canonical dynamic resumed-matrix reference")
    panel_path = _require_hash(protocol["parent"]["E0045_final_model_panel_path"],
                               protocol["parent"]["E0045_final_model_panel_sha256"],
                               "E0045 frozen development prediction panel")
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    panel_rows = panel["all_selected_rows"]
    panel_ns = pd.DatetimeIndex(pd.to_datetime([row["Opened"] for row in panel_rows], utc=True, errors="raise")).as_unit("ns").asi8
    panel_lookup = {int(value): index for index, value in enumerate(panel_ns)}
    cutoff = pd.Timestamp(protocol["scope"]["last_allowed_model_feature_opened_utc"]).tz_convert("UTC")
    model_members = list(d_manifest["members"])
    model_hashes = {f"r{int(member['reservoir_seed'])}_l{member['lightgbm_seed']}":
                    sha256_file(Path(member["model_path"])) for member in model_members}
    calibrator_hash = continuation.canonical_sha256(d_manifest["calibrator_doc"])
    result_rows = []
    suffix_summary = {}

    for stage in ("C1_to_C2", "C1_to_END", "C2_to_END"):
        execution = executions[stage]
        block_paths = {name: ROOT / execution[name]["path"] for name in ("base64", "flow6", "dynamic")}
        block_hashes = {name: sha256_file(path) for name, path in block_paths.items()}
        for name in block_paths:
            if block_hashes[name] != execution[name]["sha256"]:
                raise RuntimeError(f"resumed {name} product changed after suffix execution for {stage}")
        actual_base = pd.read_parquet(block_paths["base64"])
        actual_flow = pd.read_parquet(block_paths["flow6"])
        actual_dynamic = pd.read_parquet(block_paths["dynamic"])
        actual_opened = pd.DatetimeIndex(pd.to_datetime(actual_base["Opened"], utc=True, errors="raise")).as_unit("ns")
        for name, frame in (("flow6", actual_flow), ("dynamic", actual_dynamic)):
            frame_opened = pd.DatetimeIndex(pd.to_datetime(frame["Opened"], utc=True, errors="raise")).as_unit("ns")
            if not frame_opened.equals(actual_opened):
                raise RuntimeError(f"resumed {name} keys differ from BASE64 in {stage}")
        if not actual_opened.is_monotonic_increasing or actual_opened.has_duplicates:
            raise RuntimeError(f"resumed {stage} products have duplicated or unordered Opened keys")
        model_mask = actual_opened <= cutoff
        opened = actual_opened[model_mask]
        actual_base = actual_base.loc[model_mask].reset_index(drop=True)
        actual_flow = actual_flow.loc[model_mask].reset_index(drop=True)
        actual_dynamic = actual_dynamic.loc[model_mask].reset_index(drop=True)
        expected_rows = int(protocol["continuation"]["model_rows_by_suffix"][stage])
        if len(opened) != expected_rows or not len(opened):
            raise RuntimeError(f"resumed model-range rows for {stage} differ from the frozen expected count")

        canonical = _load_rows_by_key(canonical_path, ["Opened", *base_order], opened,
                                      protocol["scope"]["source_raw_opened_exclusive_utc"])
        dynamic_names = list(memory_order)
        for seed in seeds:
            dynamic_names.extend(d_manifest["esn_feature_order_by_seed"][str(seed)])
        canonical_dynamic = _load_rows_by_key(canonical_dynamic_path, ["Opened", *dynamic_names], opened,
                                              protocol["scope"]["source_raw_opened_exclusive_utc"])
        for frame in (canonical, canonical_dynamic):
            frame_opened = pd.DatetimeIndex(pd.to_datetime(frame["Opened"], utc=True, errors="raise")).as_unit("ns")
            if not frame_opened.equals(opened):
                raise RuntimeError(f"canonical keys do not cover resumed {stage} model rows")

        raw_base = np.full((len(opened), len(base_order)), np.nan, dtype=np.float32)
        raw_base[:, :64] = actual_base.loc[:, raw_order].to_numpy(dtype=np.float32)
        flow_source = actual_flow.loc[:, FLOW_COLUMNS].to_numpy(dtype=np.float64, copy=False)
        flow_cast, flow_matched = continuation.historical_flow_join_cast(
            opened, actual_opened[model_mask], flow_source,
        )
        if not flow_matched.all():
            raise RuntimeError(f"resumed FLOW6 historical cast left unmatched rows in {stage}")
        raw_base[:, 64:] = flow_cast
        canonical_base = canonical.loc[:, base_order].to_numpy(dtype=np.float32, copy=True)
        memory_actual = actual_dynamic.loc[:, [f"memory_{index:02d}" for index in range(len(memory_order))]].to_numpy(dtype=np.float32, copy=True)
        memory_canonical = canonical_dynamic.loc[:, memory_order].to_numpy(dtype=np.float32, copy=True)
        panel_positions = np.array([panel_lookup.get(int(value), -1) for value in opened.asi8], dtype=np.int64)
        panel_mask = panel_positions >= 0
        panel_positions = panel_positions[panel_mask]
        stage_panel_ns = opened.asi8[panel_mask]
        if len(panel_positions) != len(np.unique(stage_panel_ns)):
            raise RuntimeError(f"resumed {stage} panel intersection contains duplicate keys")
        matrix_started = time.perf_counter()
        panel_actual_by_seed = {}
        panel_canonical_by_seed = {}
        stage_matrix_rows = []
        for seed in seeds:
            esn_order = list(d_manifest["esn_feature_order_by_seed"][str(seed)])
            actual_esn = actual_dynamic.loc[:, [f"esn_{seed}_{index:02d}" for index in range(len(esn_order))]].to_numpy(dtype=np.float32, copy=True)
            canonical_esn = canonical_dynamic.loc[:, esn_order].to_numpy(dtype=np.float32, copy=True)
            actual_matrix = np.concatenate((raw_base, memory_actual, actual_esn), axis=1).astype(np.float32, copy=False)
            expected_matrix = np.concatenate((canonical_base, memory_canonical, canonical_esn), axis=1).astype(np.float32, copy=False)
            actual_matrix[~np.isfinite(actual_matrix)] = np.nan
            expected_matrix[~np.isfinite(expected_matrix)] = np.nan
            names = [*base_order, *memory_order, *esn_order]
            if len(names) != 86 or actual_matrix.shape[1] != 86:
                raise RuntimeError("resumed E0041 matrix feature order/width is not 86")
            matrix_records = _resumed_matrix_parity_rows(actual_matrix, expected_matrix, opened, seed, stage, names)
            for record in matrix_records:
                record.update({"source_base64_sha256": block_hashes["base64"],
                               "source_flow6_sha256": block_hashes["flow6"],
                               "source_dynamic_sha256": block_hashes["dynamic"],
                               "canonical_base70_sha256": source["canonical_model_feature_reference"]["sha256"],
                               "canonical_dynamic_sha256": dynamic_reference["sha256"]})
            stage_matrix_rows.extend(matrix_records)
            panel_actual_by_seed[seed] = actual_matrix[panel_mask].copy()
            panel_canonical_by_seed[seed] = expected_matrix[panel_mask].copy()
        matrix_seconds = time.perf_counter() - matrix_started
        matrix_coverage = _aggregate_partition_coverage(
            stage_matrix_rows,
            {f"resumed_model_matrix_r{seed}_{stage}":
             [*base_order, *memory_order, *d_manifest["esn_feature_order_by_seed"][str(seed)]] for seed in seeds},
            len(opened),
        )
        result_rows.extend(stage_matrix_rows)

        prediction_rows = []
        raw_members = []
        canonical_members = []
        prediction_started = time.perf_counter()
        selected_panel = [panel_rows[index] for index in panel_positions]
        selected_panel_opened = pd.DatetimeIndex(pd.to_datetime(
            [row["Opened"] for row in selected_panel], utc=True, errors="raise")).as_unit("ns")
        if len(selected_panel) != len(stage_panel_ns) or not _same_opened_keys(selected_panel_opened, stage_panel_ns):
            raise RuntimeError(f"panel intersection order differs from resumed {stage}")
        for index, member in enumerate(model_members):
            seed = int(member["reservoir_seed"])
            booster = lgb.Booster(model_file=str(member["model_path"]))
            if booster.num_feature() != 86:
                raise RuntimeError(f"saved E0041 model width changed for resumed {stage}")
            raw_probability = np.asarray(booster.predict(panel_actual_by_seed[seed]), dtype=np.float64)
            canonical_probability = np.asarray(booster.predict(panel_canonical_by_seed[seed]), dtype=np.float64)
            raw_members.append(raw_probability)
            canonical_members.append(canonical_probability)
            expected_panel_raw = np.asarray([row["raw_member_probabilities"][index] for row in selected_panel], dtype=np.float64)
            expected_panel_canonical = np.asarray([row["canonical_member_probabilities"][index] for row in selected_panel], dtype=np.float64)
            prediction_rows.extend([
                _parity_record(f"resumed_D_member_prediction_{stage}", f"r{seed}_l{member['lightgbm_seed']}",
                               raw_probability, canonical_probability, selected_panel_opened, atol=2e-12, rtol=2e-12),
                _parity_record(f"canonical_D_member_panel_check_{stage}", f"r{seed}_l{member['lightgbm_seed']}",
                               canonical_probability, expected_panel_canonical, selected_panel_opened, atol=2e-12, rtol=2e-12),
                _parity_record(f"resumed_D_member_panel_check_{stage}", f"r{seed}_l{member['lightgbm_seed']}",
                               raw_probability, expected_panel_raw, selected_panel_opened, atol=2e-12, rtol=2e-12),
            ])
            for row in prediction_rows[-3:]:
                row["model_sha256"] = model_hashes[f"r{seed}_l{member['lightgbm_seed']}"]
        raw_mean = np.mean(np.vstack(raw_members), axis=0, dtype=np.float64)
        canonical_mean = np.mean(np.vstack(canonical_members), axis=0, dtype=np.float64)
        calibrator = d_manifest["calibrator_doc"]
        def apply_platt(values):
            clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
            logits = np.log(clipped / (1.0 - clipped))
            return expit(float(calibrator["intercept"]) + float(calibrator["coefficient"]) * logits)
        raw_calibrated = apply_platt(raw_mean)
        canonical_calibrated = apply_platt(canonical_mean)
        expected_raw_mean = np.asarray([row["raw_D_uniform_probability"] for row in selected_panel], dtype=np.float64)
        expected_canonical_mean = np.asarray([row["canonical_D_uniform_probability"] for row in selected_panel], dtype=np.float64)
        expected_raw_cal = np.asarray([row["raw_D_terminal_Platt_probability"] for row in selected_panel], dtype=np.float64)
        expected_canonical_cal = np.asarray([row["canonical_D_terminal_Platt_probability"] for row in selected_panel], dtype=np.float64)
        prediction_rows.extend([
            _parity_record(f"resumed_D_uniform_mean_{stage}", "uniform_1_over_9", raw_mean, canonical_mean, selected_panel_opened, atol=2e-12, rtol=2e-12),
            _parity_record(f"canonical_D_uniform_panel_check_{stage}", "uniform_1_over_9", canonical_mean, expected_canonical_mean, selected_panel_opened, atol=2e-12, rtol=2e-12),
            _parity_record(f"resumed_D_uniform_panel_check_{stage}", "uniform_1_over_9", raw_mean, expected_raw_mean, selected_panel_opened, atol=2e-12, rtol=2e-12),
            _parity_record(f"resumed_D_terminal_Platt_{stage}", "saved_Platt", raw_calibrated, canonical_calibrated, selected_panel_opened, atol=2e-12, rtol=2e-12),
            _parity_record(f"canonical_D_Platt_panel_check_{stage}", "saved_Platt", canonical_calibrated, expected_canonical_cal, selected_panel_opened, atol=2e-12, rtol=2e-12),
            _parity_record(f"resumed_D_Platt_panel_check_{stage}", "saved_Platt", raw_calibrated, expected_raw_cal, selected_panel_opened, atol=2e-12, rtol=2e-12),
        ])
        for row in prediction_rows:
            row.update({"source_base64_sha256": block_hashes["base64"], "source_flow6_sha256": block_hashes["flow6"],
                        "source_dynamic_sha256": block_hashes["dynamic"],
                        "canonical_base70_sha256": source["canonical_model_feature_reference"]["sha256"],
                        "canonical_dynamic_sha256": dynamic_reference["sha256"],
                        "final_bundle_sha256": protocol["parent"]["bundle_manifest_sha256"],
                        "all_model_hashes_sha256": continuation.canonical_sha256(model_hashes),
                        "saved_calibrator_sha256": calibrator_hash})
        result_rows.extend(prediction_rows)
        final_start = pd.Timestamp("2026-06-30T22:50:00Z")
        final_end = pd.Timestamp("2026-06-30T23:53:00Z")
        finalization_expected = [row["Opened"] for row in panel_rows
                                 if final_start <= pd.Timestamp(row["Opened"]).tz_convert("UTC") <= final_end
                                 and pd.Timestamp(row["Opened"]).tz_convert("UTC") >= actual_opened[0]]
        finalization_compared = [stamp for stamp in selected_panel_opened
                                 if final_start <= stamp.tz_convert("UTC") <= final_end]
        if stage in {"C1_to_END", "C2_to_END"} and len(finalization_compared) != len(finalization_expected):
            raise RuntimeError(f"finalization panel intersection is incomplete for {stage}")
        suffix_summary[stage] = {
            "suffix_rows": int(len(actual_opened)), "model_rows": int(len(opened)),
            "first_model_opened": opened[0].isoformat(), "last_model_opened": opened[-1].isoformat(),
            "panel_expected_keys": [row["Opened"] for row in panel_rows
                                    if int(pd.Timestamp(row["Opened"]).value) in set(opened.asi8)],
            "panel_compared_keys": [stamp.isoformat() for stamp in selected_panel_opened],
            "finalization_panel_compared_rows": int(len(finalization_compared)),
            "finalization_panel_complete": stage not in {"C1_to_END", "C2_to_END"} or len(finalization_compared) == len(finalization_expected),
            "matrix_coverage": matrix_coverage,
            "model_hashes": model_hashes,
            "saved_calibrator_sha256": calibrator_hash,
            "matrix_seconds": matrix_seconds, "prediction_seconds": time.perf_counter() - prediction_started,
            "product_hashes": block_hashes,
        }
    return result_rows, {"suffixes": suffix_summary, "matrix_records": sum(row["block"].startswith("resumed_model_matrix_") for row in result_rows),
                         "prediction_records": sum("_prediction_" in row.get("block", "") or "_panel_check_" in row.get("block", "") or "_panel_" in row.get("block", "") for row in result_rows),
                         "feature_count_per_matrix": 86, "matrix_count": len(seeds)}


def _semantic_diff(left, right, path: str = "$") -> list[dict]:
    diffs = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                diffs.append({"path": child, "category": "state_field_presence", "status": "FAIL",
                              "left_type": type(left.get(key)).__name__ if key in left else "MISSING",
                              "right_type": type(right.get(key)).__name__ if key in right else "MISSING"})
            elif key in {"checkpoint_name", "created_prefix_rows"}:
                if left[key] != right[key]:
                    diffs.append({"path": child, "category": "checkpoint_metadata", "status": "INFO",
                                  "left_type": type(left[key]).__name__, "right_type": type(right[key]).__name__,
                                  "left_value": str(left[key]), "right_value": str(right[key])})
            else:
                diffs.extend(_semantic_diff(left[key], right[key], child))
        return diffs
    if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
        if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
            return [{"path": path, "category": "dataframe_type", "status": "FAIL",
                     "left_type": type(left).__name__, "right_type": type(right).__name__}]
        if type(left) is not type(right):
            diffs.append({"path": path, "category": "dataframe_representation", "status": "FAIL",
                          "left_type": type(left).__name__, "right_type": type(right).__name__})
        if left.shape != right.shape:
            diffs.append({"path": path, "category": "dataframe_shape", "status": "FAIL",
                          "left_type": type(left).__name__, "right_type": type(right).__name__,
                          "left_shape": list(left.shape), "right_shape": list(right.shape)})
        if not left.columns.equals(right.columns):
            diffs.append({"path": f"{path}.columns", "category": "dataframe_column_order_or_names", "status": "FAIL",
                          "left_type": type(left.columns).__name__, "right_type": type(right.columns).__name__,
                          "left_value": list(left.columns), "right_value": list(right.columns)})
        if (type(left.index) is not type(right.index) or left.index.dtype != right.index.dtype
                or not left.index.equals(right.index) or left.index.name != right.index.name):
            diffs.append({"path": f"{path}.index", "category": "dataframe_index", "status": "FAIL",
                          "left_type": type(left.index).__name__, "right_type": type(right.index).__name__,
                          "left_dtype": str(left.index.dtype), "right_dtype": str(right.index.dtype),
                          "left_shape": [len(left.index)], "right_shape": [len(right.index)],
                          "left_value": str(left.index[:3]), "right_value": str(right.index[:3])})
        for column in sorted(set(left.columns) | set(right.columns), key=str):
            child = f"{path}.{column}"
            if column not in left.columns or column not in right.columns:
                diffs.append({"path": child, "category": "dataframe_column_presence", "status": "FAIL",
                              "left_type": str(left[column].dtype) if column in left.columns else "MISSING",
                              "right_type": str(right[column].dtype) if column in right.columns else "MISSING"})
                continue
            left_series, right_series = left[column], right[column]
            if left_series.dtype != right_series.dtype:
                diffs.append({"path": child, "category": "dataframe_column_dtype", "status": "FAIL",
                              "left_type": str(left_series.dtype), "right_type": str(right_series.dtype),
                              "left_shape": [len(left_series)], "right_shape": [len(right_series)]})
            diffs.extend(_semantic_diff(left_series.to_numpy(), right_series.to_numpy(), child))
        return diffs
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if len(left) != len(right):
            return [{"path": path, "category": "ordered_sequence_length", "status": "FAIL",
                     "left_type": type(left).__name__, "right_type": type(right).__name__,
                     "left_shape": [len(left)], "right_shape": [len(right)],
                     "mismatch_count": abs(len(left) - len(right))}]
        if type(left) is not type(right):
            diffs.append({"path": path, "category": "ordered_sequence_representation", "status": "INFO",
                          "left_type": type(left).__name__, "right_type": type(right).__name__,
                          "left_shape": [len(left)], "right_shape": [len(right)]})
        for index, (a, b) in enumerate(zip(left, right)):
            diffs.extend(_semantic_diff(a, b, f"{path}[{index}]"))
        return diffs
    if path.endswith(".last_candle_time") and isinstance(left, str) and isinstance(right, str):
        try:
            if pd.Timestamp(left).value == pd.Timestamp(right).value:
                if left != right:
                    return [{"path": path, "category": "same_instant_time_representation", "status": "INFO",
                             "left_type": "str", "right_type": "str", "left_value": left, "right_value": right}]
                return []
        except (TypeError, ValueError):
            pass
    if isinstance(left, (np.ndarray, list, tuple)) or isinstance(right, (np.ndarray, list, tuple)):
        a = np.asarray(left)
        b = np.asarray(right)
        if a.shape != b.shape or a.dtype != b.dtype:
            return [{"path": path, "category": "array_structure", "status": "FAIL",
                     "left_type": type(left).__name__, "right_type": type(right).__name__,
                     "left_dtype": str(a.dtype), "right_dtype": str(b.dtype),
                     "left_shape": list(a.shape), "right_shape": list(b.shape)}]
        try:
            equal = np.equal(a, b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind in "fc" else np.equal(a, b)
            mismatch = ~equal
            count = int(np.count_nonzero(mismatch))
            if not count:
                return []
            numeric = a.dtype.kind in "fiu" and b.dtype.kind in "fiu"
            if numeric:
                finite = np.isfinite(a) & np.isfinite(b)
                absolute = np.zeros(a.shape, dtype=np.float64)
                absolute[finite] = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
                denominator = np.abs(b[finite].astype(np.float64))
                relative = np.divide(absolute[finite], denominator, out=np.zeros_like(denominator), where=denominator != 0)
                max_abs = float(absolute[finite].max()) if finite.any() else 0.0
                max_rel = float(relative.max()) if len(relative) else 0.0
            else:
                max_abs = max_rel = None
            first = np.unravel_index(int(np.flatnonzero(mismatch)[0]), mismatch.shape)
            return [{"path": path, "category": "computational_array", "status": "FAIL",
                     "left_type": type(left).__name__, "right_type": type(right).__name__,
                     "left_dtype": str(a.dtype), "right_dtype": str(b.dtype),
                     "left_shape": list(a.shape), "right_shape": list(b.shape),
                     "mismatch_count": count, "first_mismatch_position": list(first),
                     "max_abs_error": max_abs, "max_rel_error": max_rel}]
        except (TypeError, ValueError):
            return [{"path": path, "category": "array_comparison_error", "status": "FAIL",
                     "left_type": type(left).__name__, "right_type": type(right).__name__,
                     "left_dtype": str(a.dtype), "right_dtype": str(b.dtype),
                     "left_shape": list(a.shape), "right_shape": list(b.shape)}]
    if isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)):
        equal = (np.isnan(left) and np.isnan(right)) or float(left) == float(right)
        if equal:
            return []
        absolute = abs(float(left) - float(right))
        relative = absolute / abs(float(right)) if float(right) != 0.0 else (0.0 if absolute == 0.0 else None)
        return [{"path": path, "category": "computational_scalar", "status": "FAIL",
                 "left_type": type(left).__name__, "right_type": type(right).__name__,
                 "left_value": float(left), "right_value": float(right),
                 "max_abs_error": absolute, "max_rel_error": relative}]
    try:
        equal = type(left) is type(right) and bool(left == right)
    except (TypeError, ValueError):
        equal = False
    if equal:
        return []
    return [{"path": path, "category": "computational_value", "status": "FAIL",
             "left_type": type(left).__name__, "right_type": type(right).__name__,
             "left_value": str(left)[:160], "right_value": str(right)[:160]}]


def _semantic_equal(left, right) -> bool:
    return not any(item["status"] == "FAIL" for item in _semantic_diff(left, right))


def _compare_continuation_paths(primary_execution: dict, full_execution: dict,
                                primary_checkpoint: dict, full_checkpoint: dict,
                                stage: str) -> list[dict]:
    rows = []
    common = min(int(primary_execution["rows_recomputed"]), int(full_execution["rows_recomputed"]))
    first_index = int(primary_execution["source_row_start"])
    if int(full_execution["source_row_start"]) != first_index or int(full_execution["rows_recomputed"]) < common:
        raise RuntimeError(f"independent {stage} continuation paths do not cover the same row keys")
    for block in ("base64", "flow6", "dynamic"):
        p_path = ROOT / primary_execution[block]["path"]
        f_path = ROOT / full_execution[block]["path"]
        p = pd.read_parquet(p_path).iloc[:common].reset_index(drop=True)
        f = pd.read_parquet(f_path).iloc[:common].reset_index(drop=True)
        p_opened = pd.DatetimeIndex(pd.to_datetime(p["Opened"], utc=True, errors="raise")).as_unit("ns")
        f_opened = pd.DatetimeIndex(pd.to_datetime(f["Opened"], utc=True, errors="raise")).as_unit("ns")
        if not p_opened.equals(f_opened):
            raise RuntimeError(f"independent {stage} continuation timestamps differ")
        for name in p.columns:
            if name == "Opened":
                continue
            if block == "dynamic":
                atol = rtol = 2e-6
            elif block == "base64":
                atol = rtol = 1e-10
            else:
                atol = rtol = 1e-10
            rows.append(_parity_record(f"independent_continuation_{stage}_{block}", name,
                                       p[name].to_numpy(), f[name].to_numpy(), p_opened,
                                       atol=atol, rtol=rtol))
    state_diffs = _semantic_diff(primary_checkpoint, full_checkpoint)
    states_equal = not any(item["status"] == "FAIL" for item in state_diffs)
    rows.append({"block": f"semantic_checkpoint_{stage}", "feature": "all_generator_state", "status": "PASS" if states_equal else "FAIL",
                 "rows": common, "numeric_mismatch_rows": 0 if states_equal else 1, "mask_mismatch_rows": 0,
                 "first_mismatch_opened_utc": "", "max_abs_error": 0.0 if states_equal else None,
                 "max_rel_error": 0.0 if states_equal else None, "atol": 0.0, "rtol": 0.0})
    rows.extend({"block": f"semantic_checkpoint_detail_{stage}", "feature": "all_generator_state",
                 "comparison": stage, **detail} for detail in state_diffs)
    return rows


def _compare_final_states(protocol: dict, e45_protocol: dict, e45_manifest: dict,
                          end_from_c1: dict, end_from_c2: dict) -> list[dict]:
    from features.reaction_profile_fixed_grid import load_state as load_rp_state
    from features.volume_profile_fixed_range import load_state as load_vp_state
    from features import btc_D_input_generator_20261029 as e45_code

    rows = []
    expected_rows = int(protocol["scope"]["safe_raw_rows"])
    expected_last = pd.Timestamp(protocol["scope"]["last_allowed_raw_opened_utc"]).tz_convert("UTC")
    for label, checkpoint in (("C1_to_END", end_from_c1), ("C2_to_END", end_from_c2)):
        if int(checkpoint.get("source_row_index", -1)) != expected_rows:
            raise RuntimeError(f"{label} final checkpoint row count differs from the frozen safe raw count")
        if pd.Timestamp(checkpoint["last_processed_opened"]).tz_convert("UTC") != expected_last:
            raise RuntimeError(f"{label} final checkpoint instant differs from the frozen safe raw endpoint")
    lineage = e45_protocol["allowed_inputs"]["base_feature_lineage"]
    for family, loader in (("reaction_profile", load_rp_state), ("volume_profile", load_vp_state)):
        record = lineage[f"{family}_state"]
        expected = loader(ROOT / record["base_path"])
        left = end_from_c1["profile_states"][family]
        right = end_from_c2["profile_states"][family]
        reference_diffs = _semantic_diff(left, expected)
        paths_diffs = _semantic_diff(left, right)
        equal_reference = not any(item["status"] == "FAIL" for item in reference_diffs)
        equal_paths = not any(item["status"] == "FAIL" for item in paths_diffs)
        rows.append({"block": "final_profile_state", "feature": family,
                     "status": "PASS" if equal_reference and equal_paths else "FAIL",
                     "rows": expected_rows,
                     "numeric_mismatch_rows": 0 if equal_reference and equal_paths else 1,
                     "mask_mismatch_rows": 0, "first_mismatch_opened_utc": "",
                     "max_abs_error": 0.0 if equal_reference and equal_paths else None,
                     "max_rel_error": 0.0 if equal_reference and equal_paths else None,
                     "atol": 0.0, "rtol": 0.0,
                     "matches_E0041_final_state": bool(equal_reference),
                     "matches_independent_resume_path": bool(equal_paths),
                     "reference_diff_fields": [item["path"] for item in reference_diffs],
                     "resume_diff_fields": [item["path"] for item in paths_diffs]})
        for comparison, details in (("E0041_reference", reference_diffs), ("independent_resume", paths_diffs)):
            for detail in details:
                rows.append({"block": "final_profile_state_detail", "feature": family,
                             "comparison": comparison, **detail})

    final_checkpoint = e45_code._load_checkpoint(
        _require_hash(e45_manifest["dynamic"]["checkpoint_path"], e45_manifest["dynamic"]["checkpoint_sha256"], "E0045 completed dynamic checkpoint"),
        identity=e45_manifest["run_identity_sha256"],
    )
    e45_dynamic = final_checkpoint["dynamic_stage"]["dynamic_state"]
    dynamic_equal = _semantic_equal(end_from_c1["dynamic_state"], e45_dynamic)
    dynamic_paths_equal = _semantic_equal(end_from_c1["dynamic_state"], end_from_c2["dynamic_state"])
    rows.append({"block": "final_dynamic_state", "feature": "memory_and_all_ESN_seeds",
                 "status": "PASS" if dynamic_equal and dynamic_paths_equal else "FAIL",
                 "rows": int(protocol["scope"]["safe_raw_rows"]),
                 "numeric_mismatch_rows": 0 if dynamic_equal and dynamic_paths_equal else 1,
                 "mask_mismatch_rows": 0, "first_mismatch_opened_utc": "",
                 "max_abs_error": 0.0 if dynamic_equal and dynamic_paths_equal else None,
                 "max_rel_error": 0.0 if dynamic_equal and dynamic_paths_equal else None,
                 "atol": 0.0, "rtol": 0.0,
                 "matches_E0045_completed_checkpoint": bool(dynamic_equal),
                 "matches_independent_resume_path": bool(dynamic_paths_equal)})
    return rows


def _run_child(mode: str) -> dict:
    log_dir = RUNTIME / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (hashlib.sha256(mode.encode("utf-8")).hexdigest()[:12] + ".log")
    env = os.environ.copy()
    env[MODE_ENV] = mode
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[variable] = "4"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve())], cwd=ROOT,
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        return_code = process.wait()
    if return_code != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(f"E0047 child failed ({mode}, exit {return_code}); log={log_path}\n{tail}")
    output_path = RUNTIME / f"child_{mode.replace('|', '_')}.json"
    if not output_path.is_file():
        raise RuntimeError(f"E0047 child did not write its result: {output_path}")
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["log_path"] = str(log_path.relative_to(ROOT)).replace("\\", "/")
    return result


def _feature_contract(e45_contract: dict, base_order: list[str]) -> dict:
    metadata = {item["name"]: item for item in e45_contract["base70"]["features"]}
    features = []
    for name in base_order[:64]:
        item = dict(metadata[name])
        if name.startswith("candle_streak_"):
            recurrence = "Serialized signed streak count plus current in-flight calendar bucket open/close/count/last timestamp; incomplete buckets do not emit."
            tail = "12,001 raw rows retained for aggregate boundary context; the streak itself is unbounded and is not reconstructed from that tail."
        elif name.startswith("rp_"):
            recurrence = "Existing reaction-profile state arrays, scales, counters, config signature and last candle timestamp; update then extract for each row."
            tail = "No profile history is inferred from tail; state is bootstrapped from the exact causal prefix."
        elif name.startswith("vp_"):
            recurrence = "Existing volume-profile state arrays, scales, counters, config signature and last candle timestamp; update then extract for each row."
            tail = "No profile history is inferred from tail; state is bootstrapped from the exact causal prefix."
        elif name.startswith("ADX_fit_"):
            recurrence = "Three saved TA-Lib EMA values for TEMA(640), warmup/seen counter, and frozen params; finite ATR/DM input uses rolling raw tail."
            tail = "12,001 raw rows; exceeds every selected finite ADX/DI input period (maximum 1,756)."
        elif name.startswith("BollingerBands_fit_"):
            recurrence = "Exact tindicators LMA accumulator and ordered p-1 raw buffer are checkpointed; NWMA variance uses its exact finite p-1 diff2 buffer."
            tail = "Stateful center LMA(8) plus exactly 593 prior variance inputs for NWMA_EPAN(594)."
        elif name.startswith("ChaikinOsc_fit_"):
            recurrence = "TA-Lib ADL accumulator plus exact TA-Lib WMA accumulators and circular input windows for both HMA legs and final smoothing; fast NWMA keeps seven prior ADL values."
            tail = "State, not a truncated HMA tail, carries arithmetic through the checkpoint."
        elif name.startswith("session_"):
            recurrence = "Stateless function of exact UTC Opened timestamp, saved exchange timezone and session configuration."
            tail = "No historical price tail required."
        elif name.startswith("realized_volatility") or name.startswith("vov_"):
            recurrence = "Finite rolling close-return moments; no recursive EWMA state in the selected features."
            tail = "At most 241 closes cover the frozen maximum 240-return window; 12,001 retained."
        elif name.startswith("futures_index_basis_"):
            recurrence = "Frozen basis/premium formulas on exact index and futures close columns with calendar interval grouping."
            tail = "12,001 raw rows cover the maximum selected 1d aggregation/boundary and its finite lags."
        else:
            recurrence = "Existing frozen candle resample/lag builder; incomplete buckets and NaN behavior are preserved."
            tail = "12,001 raw rows cover 1d lag7 plus at least eight complete UTC daily buckets and the current boundary."
        item["continuation_state"] = recurrence
        item["sufficient_tail"] = tail
        features.append(item)
    return {
        "experiment_id": "E0047",
        "parent_experiment_id": "E0046",
        "final_bundle_sha256": "7715419ce5efbce8405db861fb3a67c481038c07f11eec019137f14528f0f97d",
        "base64_order": base_order[:64],
        "flow6_order": list(FLOW_COLUMNS),
        "base_tail_rows": continuation.TAIL_ROWS,
        "flow_raw_tail_rows": FLOW_TAIL_ROWS,
        "features": features,
        "checkpoint_state": ["exact raw CSV byte offset", "source row index", "last Opened", "next expected Opened",
                             "raw tail", "FLOW raw tail", "streak recurrence", "RP/VP native states",
                             "ADX TEMA stages", "Chaikin ADL state and tail", "memory state", "all ESN states",
                             "frozen source/config/transform/model identities", "integrity SHA-256"],
        "model_matrix_order": "BASE70 then memory PCA8 then one frozen seed-specific ESN PCA8; cast float32 and replace all non-finite values with NaN exactly as final E0041 matrix preparation.",
        "nonfinite_and_gap_semantics": "No timestamp repair, interpolation, forward fill, memory reset, or profile/streak reset is introduced by the adapter.",
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv as csv_module

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv_module.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _write_results(protocol: dict, e45_protocol: dict, e45_manifest: dict,
                   d_bundle: dict, flow_batch: pd.DataFrame, flow_batch_paths: list[str],
                   child_results: dict, parity_rows: list[dict], flow_representation_summary: dict,
                   matrix_summary: dict, flow_rows: list[dict], state_rows: list[dict],
                   resource_before: dict, started: float) -> dict:
    output = protocol["outputs"]
    report_path = ROOT / output["report_path"]
    manifest_path = ROOT / output["manifest_path"]
    parity_path = ROOT / output["parity_csv_path"]
    contract_path = ROOT / output["feature_contract_path"]
    all_rows = list(parity_rows) + list(flow_rows) + list(state_rows)
    formula_rows = e45_manifest["flow_parity"]
    formula_pass = len(formula_rows) == 6 and all(row["status"] == "PASS" for row in formula_rows)
    base_rows = [row for row in all_rows if row["block"].startswith("base64_continuation_")]
    flow_continuation_rows = [row for row in all_rows if row["block"].startswith("flow6_continuation_")]
    dynamic_continuation_rows = [row for row in all_rows if row["block"].startswith(("memory_continuation_", "esn_continuation_"))]
    matrix_rows = [row for row in all_rows if row["block"].startswith("model_matrix_")]
    batch_prediction_blocks = {"final_D_member_prediction", "canonical_D_member_prediction", "final_D_uniform_mean",
                              "canonical_D_uniform_mean", "final_D_terminal_Platt", "canonical_D_terminal_Platt"}
    prediction_rows = [row for row in all_rows if row["block"] in batch_prediction_blocks]
    resumed_matrix_rows = [row for row in all_rows if row["block"].startswith("resumed_model_matrix_")]
    resumed_prediction_rows = [row for row in all_rows if row["block"].startswith((
        "resumed_D_", "canonical_D_member_panel_check_", "canonical_D_uniform_panel_check_", "canonical_D_Platt_panel_check_"))]
    profile_summary_rows = [row for row in state_rows if row["block"] in {"final_profile_state", "final_dynamic_state"}]
    checkpoint_summary_rows = [row for row in all_rows if row.get("block") in {
        "semantic_checkpoint_C1_to_C2", "semantic_checkpoint_C2_to_END"}]
    base_summary_rows = [row for row in base_rows if row["status"] == "PASS"]
    bb_rows = [row for row in base_rows if row["feature"].startswith("BollingerBands_fit_")]
    chaikin_rows = [row for row in base_rows if row["feature"].startswith("ChaikinOsc_fit_")]
    resumed_matrix_status = all(
        matrix_summary["resumed_suffixes"][stage]["matrix_coverage"]["status"] == "PASS"
        for stage in ("C1_to_C2", "C1_to_END", "C2_to_END")
    )
    status_groups = {
        "FLOW_formula": formula_pass,
        "FLOW_representation": bool(flow_representation_summary["float32_exact_status"] == "PASS" and flow_representation_summary["float32_exact_mismatches"] == 0),
        "Bollinger_continuation": bool(len(bb_rows) == 3 and all(row["status"] == "PASS" for row in bb_rows)),
        "Chaikin_continuation": bool(len(chaikin_rows) == 3 and all(row["status"] == "PASS" for row in chaikin_rows)),
        "BASE64_continuation": bool(len(base_summary_rows) == 64 * 3 and all(row["status"] == "PASS" for row in base_rows) and all(row["status"] == "PASS" for row in all_rows if row["block"].startswith("independent_continuation_") and "base64" in row["block"])),
        "FLOW_continuation": bool(len(flow_continuation_rows) == 6 * 3 and all(row["status"] == "PASS" for row in flow_continuation_rows) and all(row["status"] == "PASS" for row in all_rows if row["block"].startswith("independent_continuation_") and "flow6" in row["block"])),
        "memory_ESN_continuation": bool(len(dynamic_continuation_rows) == 32 * 3 and all(row["status"] == "PASS" for row in dynamic_continuation_rows) and all(row["status"] == "PASS" for row in all_rows if row["block"].startswith("independent_continuation_") and "dynamic" in row["block"])),
        "batch_model_matrices": bool(matrix_summary["coverage"]["status"] == "PASS" and all(row["status"] == "PASS" for row in matrix_rows)),
        "batch_D_predictions": bool(len(prediction_rows) == 22 and all(row["status"] == "PASS" for row in prediction_rows)),
        "resumed_model_matrices": bool(resumed_matrix_status and len(resumed_matrix_rows) == 3 * 3 * 86 and all(row["status"] == "PASS" for row in resumed_matrix_rows)),
        "resumed_D_predictions": bool(len(resumed_prediction_rows) > 0 and all(row["status"] == "PASS" for row in resumed_prediction_rows)
                                       and all(item["finalization_panel_complete"] for item in matrix_summary["resumed_suffixes"].values())),
        "final_states": bool(len(profile_summary_rows) == 3 and all(row["status"] == "PASS" for row in profile_summary_rows)
                              and len(checkpoint_summary_rows) == 2
                              and all(row["status"] == "PASS" for row in checkpoint_summary_rows)),
        "all_parity_records": _count_parity_failures(row["status"] for row in all_rows) == 0,
    }
    overall = "DEV_GENERATOR_VERIFIED" if all(status_groups.values()) else "DEV_GENERATOR_PARTIAL"
    suffix_summary = {}
    for key in ("C1_to_C2", "C2_to_END", "C1_to_END"):
        result = child_results[key]
        suffix_summary[key] = {
            "rows": result["rows_recomputed"],
            "first_opened": result["first_suffix_opened"],
            "last_opened": result["last_suffix_opened"],
            "raw_csv_bytes_read": result["raw_bytes_read"],
            "state_read_seconds": result["state_read_seconds"],
            "stage_seconds": result["stage_seconds"],
            "wall_seconds": result["wall_seconds"],
            "peak_rss_gib": result["peak_rss_gib"],
            "base64": result["base64"], "flow6": result["flow6"], "dynamic": result["dynamic"],
            "saved_checkpoint": result.get("saved_checkpoint"),
            "intermediate_checkpoints": result.get("intermediate_checkpoints", []),
            "base64_builder_input_rows": result["base64_builder_input_rows"],
            "base64_raw_tail_rows_recomputed": result["base64_raw_tail_rows_recomputed"],
            "flow_formula_input_rows": result["flow_formula_input_rows"],
            "flow_raw_tail_rows_recomputed": result["flow_raw_tail_rows_recomputed"],
            "csv_roundtrip_audit": result.get("csv_roundtrip_audit", []),
        }
    cp1 = child_results["build_C1"]
    code_lines = {
        "join_function": {"path": "run_btc_preopen_experiment.py", "first_line": 1311, "last_line": 1375,
                          "function": "_join_causal_candidate_features",
                          "sha256": protocol["historical_flow_representation"]["join_implementation_sha256"]},
        "matrix_function": {"path": "run_btc_chronology_reconstruction_20261027.py", "function": "_family_matrix",
                            "sha256": protocol["historical_flow_representation"]["downstream_matrix_implementation_sha256"]},
        "final_cast": {"path": "run_btc_D_holdout_preparation_20261028.py", "approx_lines": "868-871",
                       "sha256": protocol["required_code_hashes"]["run_btc_D_holdout_preparation_20261028.py"],
                       "operation": "np.asarray(matrix, dtype=np.float32); non-finite values then set to NaN"},
    }
    input_sizes = {
        "safe_raw_csv_bytes": (ROOT / protocol["scope"]["source_raw_path"]).stat().st_size,
        "E0045_base64_bytes": (ROOT / protocol["parent"]["E0045_base64_path"]).stat().st_size,
        "flow6_batch_partition_bytes": int(sum((ROOT / path).stat().st_size for path in flow_batch_paths)),
        "canonical_base70_bytes": (ROOT / e45_protocol["allowed_inputs"]["canonical_model_feature_reference"]["path"]).stat().st_size,
        "canonical_dynamic_bytes": (ROOT / e45_protocol["allowed_inputs"]["canonical_dynamic_reference"]["path"]).stat().st_size,
    }
    flow_archive_hash_bytes = int(sum(value.get("flow_archive_io", {}).get("flow_archive_sha256_bytes", 0) for value in child_results.values()))
    flow_archive_stream_bytes = int(sum(value.get("flow_archive_io", {}).get("flow_archive_uncompressed_stream_bytes", 0) for value in child_results.values()))
    suffix_raw_bytes = int(sum(int(item.get("raw_bytes_read", 0)) for item in child_results.values() if isinstance(item, dict)))
    reused_execution = protocol["execution"]["mode"] == "REVALIDATE_HASH_PINNED_FRESH_PROCESS_PRODUCTS"
    source_fresh_process_wall_seconds = float(sum(float(item.get("wall_seconds", 0.0)) for item in child_results.values()))
    source_fresh_process_suffix_rows = int(sum(int(item.get("rows_recomputed", 0)) for item in child_results.values()))
    source_products = protocol["execution"]["source_products"]
    source_manifest_record = next((item for item in source_products["report_artifacts"]
                                   if item["path"].endswith("_manifest.json")), None)
    source_run_resources = {}
    if reused_execution:
        if source_manifest_record is None:
            raise RuntimeError("E0047 revalidation protocol lacks the source execution manifest")
        source_run_manifest = json.loads(_require_hash(source_manifest_record["path"], source_manifest_record["sha256"],
                                                       "pinned E0047 v3 execution manifest").read_text(encoding="utf-8"))
        source_run_resources = source_run_manifest["resource"]
        source_fresh_process_wall_seconds = float(source_run_resources["wall_seconds"])
    source_input_bytes_lower_bound = int(
        cp1["byte_offset"] + suffix_raw_bytes + flow_archive_hash_bytes + flow_archive_stream_bytes + sum(input_sizes.values())
    )
    if reused_execution:
        source_input_bytes_lower_bound = int(source_run_resources["measured_input_bytes_lower_bound"])
    report_manifest = {
        "schema_version": 1,
        "experiment_id": "E0047", "task_id": "T0016", "decision_id": "D0024",
        "status": overall, "statuses": status_groups,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "execution_provenance": {
            "mode": protocol["execution"]["mode"],
            "source_runtime_root": protocol["execution"]["source_runtime_root"],
            "source_protocol_sha256": source_products["protocol_sha256"],
            "source_report_manifest_sha256": source_manifest_record["sha256"] if source_manifest_record else None,
            "source_fresh_process_child_records": len(protocol["execution"]["child_results"]),
            "recomputed_suffix_rows_in_this_audit": 0 if reused_execution else source_fresh_process_suffix_rows,
            "source_process_products_hash_verified": True,
        },
        "parent_experiment_id": "E0046", "parent_manifest_sha256": protocol["parent"]["E0046_manifest_sha256"],
        "parent_E0045_manifest_sha256": sha256_file(E0045_MANIFEST_PATH),
        "final_E0041_bundle_sha256": "7715419ce5efbce8405db861fb3a67c481038c07f11eec019137f14528f0f97d",
        "source_raw_sha256": protocol["scope"]["source_raw_sha256"],
        "source_raw_last_opened": protocol["scope"]["last_allowed_raw_opened_utc"],
        "first_protected_opened_payload_read": False,
        "holdout_values_rows_features_states_predictions_labels_books_economics_read": False,
        "holdout_predictions_generated": False,
        "evaluation_authorized": False, "activated": False, "holdout_status": "HOLDOUT_NOT_RUN",
        "flow_formula_reused_E0045": {"status": "PASS" if formula_pass else "FAIL", "rows": formula_rows[0]["rows"] if formula_rows else 0,
                                       "features": [row["feature"] for row in formula_rows],
                                       "all_zero_numeric_mismatches": bool(formula_pass)},
        "flow_representation": flow_representation_summary,
        "E0046_flow_reconciliation": flow_representation_summary.get("prior_E0046_reconciliation"),
        "historical_conversion": code_lines,
        "suffixes": suffix_summary,
        "model_matrices": matrix_summary,
        "semantic_state_comparisons": state_rows,
        "parity_rows": len(all_rows),
        "parity_csv_path": str(parity_path.relative_to(ROOT)).replace("\\", "/"),
        "parity_csv_sha256": sha256_file(parity_path) if parity_path.exists() else None,
        "feature_contract_path": str(contract_path.relative_to(ROOT)).replace("\\", "/"),
        "feature_contract_sha256": sha256_file(contract_path) if contract_path.exists() else None,
        "resource": {
            "wall_seconds": time.perf_counter() - started,
            "available_ram_gib_before": resource_before["available_ram_gib"],
            "available_ram_gib_after": _memory_available_gib(),
            "free_disk_gib_before": resource_before["free_disk_gib"],
            "free_disk_gib_after": shutil.disk_usage(ROOT).free / 1024**3,
            "parent_peak_rss_gib": _peak_rss_gib(),
            "maximum_peak_rss_gib": max([_peak_rss_gib(), cp1.get("peak_rss_gib", 0.0), *[float(result.get("peak_rss_gib", 0.0)) for result in child_results.values() if isinstance(result, dict)]]),
            "prefix_raw_rows_read": 0 if reused_execution else int(cp1["rows"]),
            "prefix_raw_csv_data_bytes_parsed": 0 if reused_execution else int(cp1["byte_offset"]),
            "prefix_raw_parse_seconds": 0.0 if reused_execution else cp1["raw_read_seconds"],
            "C1_checkpoint_build_seconds": 0.0 if reused_execution else cp1["checkpoint_build_seconds"],
            "C1_checkpoint_write_seconds": 0.0 if reused_execution else cp1["checkpoint_write_seconds"],
            "suffix_raw_rows_recomputed_total": 0 if reused_execution else source_fresh_process_suffix_rows,
            "suffix_raw_csv_data_bytes_parsed_total": 0 if reused_execution else suffix_raw_bytes,
            "base64_builder_input_rows_total": 0 if reused_execution else int(sum(int(item.get("base64_builder_input_rows", 0)) for item in child_results.values() if isinstance(item, dict))),
            "base64_raw_tail_rows_recomputed_total": 0 if reused_execution else int(sum(int(item.get("base64_raw_tail_rows_recomputed", 0)) for item in child_results.values() if isinstance(item, dict))),
            "flow_formula_input_rows_total": 0 if reused_execution else int(sum(int(item.get("flow_formula_input_rows", 0)) for item in child_results.values() if isinstance(item, dict))),
            "flow_raw_tail_rows_recomputed_total": 0 if reused_execution else int(sum(int(item.get("flow_raw_tail_rows_recomputed", 0)) for item in child_results.values() if isinstance(item, dict))),
            "source_fresh_process_prefix_rows_read": int(cp1["rows"]),
            "source_fresh_process_suffix_rows_recomputed": source_fresh_process_suffix_rows,
            "source_fresh_process_wall_seconds": source_fresh_process_wall_seconds,
            "source_fresh_process_max_peak_rss_gib": float(source_run_resources.get(
                "maximum_peak_rss_gib", max([float(cp1.get("peak_rss_gib", 0.0)), *[float(result.get("peak_rss_gib", 0.0)) for result in child_results.values() if isinstance(result, dict)]]))),
            "state_read_seconds_total": float(sum(float(item.get("state_read_seconds", 0.0)) for item in child_results.values() if isinstance(item, dict))),
            "resume_stage_seconds": {name: float(sum(item.get("stage_seconds", {}).get(name, 0.0) for item in child_results.values() if isinstance(item, dict)))
                                     for name in {key for item in child_results.values() if isinstance(item, dict) for key in item.get("stage_seconds", {})}},
            "csv_roundtrip_numeric_mismatch_rows": int(sum(
                audit.get("numeric_mismatch_rows", 0)
                for item in child_results.values() if isinstance(item, dict)
                for audit in item.get("csv_roundtrip_audit", [])
            )),
            "csv_roundtrip_parser": "pandas.read_csv default C parser, same parser call as create_modeling_dataset.py",
            "csv_roundtrip_dtypes": [audit.get("dtypes", {}) for item in child_results.values() if isinstance(item, dict)
                                     for audit in item.get("csv_roundtrip_audit", [])],
            "flow_archive_sha256_bytes_consumed_by_formula_processing": 0 if reused_execution else flow_archive_hash_bytes,
            "flow_archive_stream_bytes_consumed_by_formula_processing": 0 if reused_execution else flow_archive_stream_bytes,
            "source_fresh_process_flow_archive_sha256_bytes": flow_archive_hash_bytes,
            "source_fresh_process_flow_archive_stream_bytes": flow_archive_stream_bytes,
            "comparison_reference_artifact_sizes_bytes": input_sizes,
            "source_fresh_process_measured_input_bytes_lower_bound": source_input_bytes_lower_bound,
            "previous_attempt_report_hash_verified_bytes": int(sum(item.get("bytes", 0) for item in protocol["previous_attempt"].get("report_artifacts", []))),
            "source_runtime_hash_verified_bytes": int(sum(item.get("bytes", 0) for item in source_products["files"])) if reused_execution else 0,
            "source_report_hash_verified_bytes": int(sum(item.get("bytes", 0) for item in source_products["report_artifacts"])) if reused_execution else 0,
            "prefix_raw_csv_hash_bytes_read_lower_bound": (ROOT / protocol["scope"]["source_raw_path"]).stat().st_size if reused_execution else 0,
            "checkpoint_bytes": {
                "C1": child_results["build_C1"]["checkpoint_bytes"],
                "C2_primary": child_results["C1_to_C2"]["saved_checkpoint"]["bytes"],
                "C2_from_C1_independent": child_results["C1_to_END"]["intermediate_checkpoints"][0]["bytes"],
                "END_from_C1": child_results["C1_to_END"]["saved_checkpoint"]["bytes"],
                "END_from_C2": child_results["C2_to_END"]["saved_checkpoint"]["bytes"],
            },
            "batch_matrix_comparison_seconds": matrix_summary["matrix_comparison_seconds"],
            "batch_prediction_seconds": matrix_summary["prediction_seconds"],
            "resumed_matrix_comparison_seconds": sum(item["matrix_seconds"] for item in matrix_summary["resumed_suffixes"].values()),
            "resumed_prediction_seconds": sum(item["prediction_seconds"] for item in matrix_summary["resumed_suffixes"].values()),
        },
        "fit_selection_training_metrics_or_economics_run": False,
        "source_archives_used": [{"month": item["month"], "sha256": item["expected_archive_sha256"]}
                                 for item in json.loads(E0045_MANIFEST_PATH.read_text(encoding="utf-8"))["flow_source"]["archives"]
                                 if item["month"] in {"2026-05", "2026-06"}],
    }
    _write_csv(parity_path, all_rows)
    report_manifest["parity_csv_sha256"] = sha256_file(parity_path)
    e45_contract = e45_manifest["feature_contract"]
    contract = _feature_contract(e45_contract, d_bundle["manifest"]["base_feature_order"])
    _json_write(contract_path, contract)
    report_manifest["feature_contract_sha256"] = sha256_file(contract_path)
    _json_write(manifest_path, report_manifest)

    def section_status(name):
        return "PASS" if status_groups[name] else "FAIL"

    formula_max = max(float(row.get("max_abs_error", 0.0)) for row in formula_rows) if formula_rows else 0.0
    summary_table = "\n".join(f"| {name} | {section_status(name)} |" for name in status_groups)
    flow_info = flow_representation_summary
    resumed_suffixes = matrix_summary["resumed_suffixes"]
    profile_info = [row for row in state_rows if row.get("block") == "final_profile_state_detail"]
    def diff_paths(family, comparison):
        return ", ".join(row["path"] for row in profile_info
                          if row.get("feature") == family and row.get("comparison") == comparison) or "brak roznic"
    def bool_text(value):
        return "PASS" if value else "FAIL"
    execution_note = (
        f"Produkty sufiksow powstaly w swiezym przebiegu v3; v5 weryfikuje je z hash-pinned plikow i nie parsuje ani nie przelicza wierszy surowego prefiksu/sufiksow. Kontrola SHA-256 czyta zapisane bajty plikow. Pelny przebieg v3 trwal {report_manifest['resource']['source_fresh_process_wall_seconds']:.1f} s, peak RSS {report_manifest['resource']['source_fresh_process_max_peak_rss_gib']:.2f} GiB; biezacy audyt porownawczy trwal {report_manifest['resource']['wall_seconds']:.1f} s. "
        if reused_execution else
        f"Wszystkie wznowienia zostaly wykonane w swiezych procesach. Czas przebiegu: {report_manifest['resource']['wall_seconds']:.1f} s. "
    )
    report = f"""# E0047 - Zamkniecie zgodnosci kontynuacji generatora D

## Wyniki najwazniejszych kontroli

- **FLOW po historycznym cast:** {flow_info['float32_exact_mismatches']:,} bledow reprezentacji na {flow_info['common_reference_rows']:,} kluczach x 6 cech; status {flow_info['float32_exact_status']}. Liczba **19 063 239** z E0046 byla blednie przypisana: agregat sumowal blok diagnostyczny float64 przed castem. Hash-sprawdzenie CSV E0046 wykazalo 0 bledow wylacznie w rekordach `flow_representation_float32`; E0047 agreguje tylko ten blok i wymaga pelnego, rozlacznego pokrycia kluczy.
- **BollingerBands:** E0046 roznil sie od batch na wszystkich 38 977 / 1 440 / 40 417 wierszach odpowiednich sufiksow, bez roznic masek; pierwszy blad `2026-06-02T22:21:00Z`, maks. abs `0.00012635984263198452`. Obciecie wejscia resetowalo rekurencyjny akumulator LMA(8), a ogon nie zachowywal jego ciaglego stanu. E0047 przenosi dokladne akumulatory LMA i 593 poprzednie `diff2` dla zamrozonego NWMA_EPAN(594). Wynik: {bool_text(status_groups['Bollinger_continuation'])}.
- **ChaikinOsc:** E0046 roznil sie na wszystkich 38 977 / 1 440 / 40 417 wierszach, bez roznic masek; pierwszy blad `2026-06-02T22:21:00Z`, maks. abs `0.006234453991055489`. ADL i jego przyrosty byly bitowo zgodne z TA-Lib; roznice wprowadzalo ponowne zainicjowanie WMA w HMA(893) na obcietym ogonie. E0047 przenosi stan trzech akumulatorow TA-Lib WMA oraz okna wejsciowe; NWMA_EPAN(8) zachowuje siedem poprzednich wartosci. Wynik: {bool_text(status_groups['Chaikin_continuation'])}.
- **RP/VP:** koncowe stany E0047 porownano dokladnie z E0041 oraz miedzy niezaleznymi wznowieniami. RP: {bool_text(next(row['status'] for row in state_rows if row.get('block') == 'final_profile_state' and row.get('feature') == 'reaction_profile'))}; roznice reprezentacyjne: E0041 `{diff_paths('reaction_profile', 'E0041_reference')}`, sciezki wznowien `{diff_paths('reaction_profile', 'independent_resume')}`. VP: {bool_text(next(row['status'] for row in state_rows if row.get('block') == 'final_profile_state' and row.get('feature') == 'volume_profile'))}; roznice reprezentacyjne: E0041 `{diff_paths('volume_profile', 'E0041_reference')}`, sciezki wznowien `{diff_paths('volume_profile', 'independent_resume')}`. W E0046 RP `horizon_names` i `feature_columns` mialy rownowazne tuple/list, a `last_candle_time` zapisywal ten sam UTC instant z/bez `+00:00`; te waskie roznice nie zmieniaja stanu. W E0046 VP `raw_profile` roznil sie numerycznie (18/11/15 elementow dla short/medium/long; maks. abs `7.275957614183426e-12`, `7.450580596923828e-09`, `3.725290298461914e-09`). Przyczyna byla inna kolejnosc dzialan kontynuacji wzgledem zamrozonego batch; E0047 odtwarza kolejnosc batch i nie normalizuje tablic liczbowych.
- **Macierze/predykcje po wznowieniu:** macierze {bool_text(status_groups['resumed_model_matrices'])}, predykcje techniczne {bool_text(status_groups['resumed_D_predictions'])}. Kazdy wiersz wszystkich trzech rzeczywistych sufiksow przeszedl przez wznowione BASE64/FLOW6/memory/ESN, historyczne konwersje i trzy macierze po 86 cech. Przeciecia zamrozonego panelu i hashe kazdego zrodla sa w manifeście/CSV. Panel finalizacyjny: C1->END {resumed_suffixes['C1_to_END']['finalization_panel_compared_rows']}/64, C2->END {resumed_suffixes['C2_to_END']['finalization_panel_compared_rows']}/64.
- **Checkpointy:** oczekiwana kompletnosc C1, C2 i END oraz zgodnosc C1->C2->END z niezaleznym C1->END: {bool_text(status_groups['final_states'])}. Zakres raw: 3 186 145 wierszy, `Opened < 2026-06-30T23:58:00Z`; ostatni raw `23:57Z`, ostatni modelowy `23:53Z`.

## Status blokow

| Blok | Wynik |
|---|---|
{summary_table}

V3 zwrocilo etykiete `DEV_GENERATOR_VERIFIED`, ale nie objelo ona dwoch porownan semantycznych checkpointow. Porownywarka nie obslugiwala ogonow DataFrame, a agregacja statusu pominela wiersze tych kontroli. Etykieta i status v3 sa niewazne. Proba v4 zatrzymala sie przed publikacja wyniku z powodu braku katalogu `suffixes`; nie przypisuje sie jej statusow. V5 tworzy wymagany katalog, ponownie sprawdza hash-pinned produkty, porownuje checkpointy kolumna po kolumnie i blokuje status VERIFIED, jesli dowolny rekord parity ma status FAIL.

## Przebiegi i zasoby

Zakresy wyprodukowane w v3: C1->C2 {suffix_summary['C1_to_C2']['rows']:,} wierszy ({suffix_summary['C1_to_C2']['first_opened']}..{suffix_summary['C1_to_C2']['last_opened']}); C1->END {suffix_summary['C1_to_END']['rows']:,} ({suffix_summary['C1_to_END']['first_opened']}..{suffix_summary['C1_to_END']['last_opened']}); C2->END {suffix_summary['C2_to_END']['rows']:,} ({suffix_summary['C2_to_END']['first_opened']}..{suffix_summary['C2_to_END']['last_opened']}). {execution_note} Szczyt RSS audytu i zrodlowego przebiegu {report_manifest['resource']['maximum_peak_rss_gib']:.2f} GiB. Dostepny RAM przed/po: {report_manifest['resource']['available_ram_gib_before']:.2f}/{report_manifest['resource']['available_ram_gib_after']:.2f} GiB; wolny dysk: {report_manifest['resource']['free_disk_gib_before']:.2f}/{report_manifest['resource']['free_disk_gib_after']:.2f} GiB. W audycie v5 sparsowano {report_manifest['resource']['prefix_raw_rows_read']:,} wierszy surowego prefiksu i przeliczono {report_manifest['resource']['suffix_raw_rows_recomputed_total']:,} surowych wierszy sufiksu; kontrole hash czytaja bajty plikow. Szczegoly I/O v3 i bajtow hash-zweryfikowanych w v5 sa w raporcie JSON.

Nie wykonano treningu, selekcji, fitow, metryk jakosci, analizy ekonomicznej ani evaluatora. Nie odczytano wartosci ani artefaktow holdoutu; nie wygenerowano nowych predykcji holdoutu. Pozostaja `evaluation_authorized=false`, `activated=false`, `HOLDOUT_NOT_RUN`; hash finalnego bundle E0041: `7715419ce5efbce8405db861fb3a67c481038c07f11eec019137f14528f0f97d`. E0046 i jego poczatkowe/korygowane artefakty zachowano bez zmian. E0037 pozostaje `FAILED`, T0009 `BLOCKED`, a T0046/T0048 pozostaja nietkniete.

`{overall}` oznacza wylacznie status developmentowego generatora. Kolejnym bezpiecznym etapem jest przygotowanie osobnego, metadata-only manifestu pokrycia historycznych zrodel Binance index/OHLCV i USD-M FLOW dla przyszlego wejscia; wymaga ono zachowania obecnych bramek i nie obejmuje payloadu holdoutu.

Artefakty: [{manifest_path.name}]({manifest_path.name}), [{parity_path.name}]({parity_path.name}), [{contract_path.name}]({contract_path.name}).
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    report_manifest["report_path"] = str(report_path.relative_to(ROOT)).replace("\\", "/")
    report_manifest["report_sha256"] = sha256_file(report_path)
    _json_write(manifest_path, report_manifest)
    return report_manifest


def run_audit() -> dict:
    started = time.perf_counter()
    protocol, e45_protocol, d_bundle, transforms = load_protocol()
    if RUNTIME.exists():
        raise RuntimeError(f"E0047 runtime identity already exists; refusing to reuse it: {RUNTIME}")
    prior_flow_evidence = _load_prior_e0046_flow_evidence(protocol)
    e45_manifest = json.loads(E0045_MANIFEST_PATH.read_text(encoding="utf-8"))
    ram_before = _memory_available_gib()
    disk_before = shutil.disk_usage(ROOT).free / 1024**3
    if ram_before < float(protocol["resource_limits"]["minimum_available_ram_gib_before_run"]):
        raise RuntimeError(f"E0047 RAM gate requires {protocol['resource_limits']['minimum_available_ram_gib_before_run']} GiB available; observed {ram_before:.2f} GiB")
    disk_gate_gib = float(protocol["resource_limits"]["minimum_free_workspace_disk_gib_before_run"])
    if disk_before < disk_gate_gib:
        raise RuntimeError(f"E0047 disk gate requires {disk_gate_gib} GiB free; observed {disk_before:.2f} GiB")
    try:
        import numba
        numba.set_num_threads(4)
    except Exception as exc:
        raise RuntimeError("could not enforce the frozen four-thread Numba limit") from exc
    resource_before = {"available_ram_gib": ram_before, "free_disk_gib": disk_before}
    RUNTIME.mkdir(parents=True, exist_ok=True)

    children = {}
    for item in protocol["execution"]["child_results"]:
        child_path = _require_hash(item["path"], item["sha256"], f"pinned suffix execution result {item['name']}")
        result = json.loads(child_path.read_text(encoding="utf-8"))
        result["log_path"] = item["log_path"]
        children[item["name"]] = result
    split2 = int(protocol["continuation"]["splits"][1]["expected_prefix_rows"])
    if children["C1_to_C2"]["rows_recomputed"] != split2 - protocol["continuation"]["splits"][0]["expected_prefix_rows"]:
        raise RuntimeError("C1→C2 raw suffix row count differs from the frozen split")
    if children["C2_to_END"]["rows_recomputed"] != int(protocol["continuation"]["splits"][1]["expected_nonempty_suffix_rows"]):
        raise RuntimeError("C2 raw suffix row count differs from the frozen split")
    if children["C1_to_END"]["rows_recomputed"] != int(protocol["continuation"]["splits"][0]["expected_nonempty_suffix_rows"]):
        raise RuntimeError("C1 raw suffix row count differs from the frozen split")

    flow_batch, flow_batch_paths = _load_flow_batch(e45_manifest, protocol["scope"]["source_raw_opened_exclusive_utc"])
    formula_rows = e45_manifest["flow_parity"]
    if len(formula_rows) != 6 or any(row["status"] != "PASS" for row in formula_rows):
        raise RuntimeError("E0045 raw FLOW formula proof is not six of six PASS")
    flow_rows, flow_representation_summary = _compare_flow_representation(protocol, e45_protocol, e45_manifest, flow_batch)
    flow_representation_summary["prior_E0046_reconciliation"] = prior_flow_evidence
    parity_rows = []
    for stage in ("C1_to_C2", "C2_to_END", "C1_to_END"):
        parity_rows.extend(_compare_suffix_stage(stage, children[stage], e45_protocol, e45_manifest,
                                                 d_bundle["manifest"], flow_batch))
    source_protocol_path = ROOT / protocol["execution"]["source_products"]["protocol_path"]
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    source_runtime = ROOT / protocol["execution"]["source_runtime_root"]
    source_checkpoint_identity = _continuation_identity(source_protocol, d_bundle["identity"])
    c1_to_c2_checkpoint = continuation.load_checkpoint(source_runtime / "checkpoints" / "C2_primary.json",
                                                        identity_sha256=source_checkpoint_identity)
    c1_end_c2_checkpoint = continuation.load_checkpoint(source_runtime / "checkpoints" / "C2_from_C1_full.json",
                                                          identity_sha256=source_checkpoint_identity)
    parity_rows.extend(_compare_continuation_paths(children["C1_to_C2"], children["C1_to_END"],
                                                   c1_to_c2_checkpoint, c1_end_c2_checkpoint, "C1_to_C2"))
    c2_overlap_count = int(protocol["continuation"]["splits"][1]["expected_nonempty_suffix_rows"])
    c1_to_end_tail = dict(children["C1_to_END"])
    c1_to_end_tail["source_row_start"] = split2
    c1_to_end_tail["rows_recomputed"] = c2_overlap_count
    c1_to_end_tail["first_suffix_opened"] = protocol["continuation"]["splits"][1]["first_suffix_opened_utc"]
    tail_directory = RUNTIME / "suffixes"
    tail_directory.mkdir(parents=True, exist_ok=True)
    for name in ("base64", "flow6", "dynamic"):
        frame = pd.read_parquet(ROOT / children["C1_to_END"][name]["path"])
        tail_path = tail_directory / f"C1_to_END_tail_C2_{name}.parquet"
        frame.iloc[-c2_overlap_count:].to_parquet(tail_path, index=False)
        c1_to_end_tail[name] = {"path": str(tail_path.relative_to(ROOT)), "rows": c2_overlap_count,
                                "sha256": sha256_file(tail_path), "bytes": tail_path.stat().st_size}
    c2_to_end_checkpoint = continuation.load_checkpoint(source_runtime / "checkpoints" / "END_from_C2.json",
                                                         identity_sha256=source_checkpoint_identity)
    c1_to_end_checkpoint = continuation.load_checkpoint(source_runtime / "checkpoints" / "END_from_C1.json",
                                                          identity_sha256=source_checkpoint_identity)
    parity_rows.extend(_compare_continuation_paths(children["C2_to_END"], c1_to_end_tail,
                                                   c2_to_end_checkpoint, c1_to_end_checkpoint, "C2_to_END"))
    batch_matrix_rows, matrix_summary = _compare_model_matrices_and_predictions(protocol, e45_protocol,
                                                                                 e45_manifest, d_bundle, flow_batch)
    resumed_matrix_rows, resumed_matrix_summary = _compare_resumed_suffix_matrices_and_predictions(
        protocol, e45_protocol, e45_manifest, d_bundle,
        {key: children[key] for key in ("C1_to_C2", "C1_to_END", "C2_to_END")},
    )
    matrix_summary["batch_coverage"] = matrix_summary["coverage"]
    matrix_summary["resumed_suffixes"] = resumed_matrix_summary["suffixes"]
    matrix_summary["resumed_matrix_records"] = resumed_matrix_summary["matrix_records"]
    matrix_summary["resumed_prediction_records"] = resumed_matrix_summary["prediction_records"]
    parity_rows.extend(batch_matrix_rows)
    parity_rows.extend(resumed_matrix_rows)
    state_rows = _compare_final_states(protocol, e45_protocol, e45_manifest,
                                       c1_to_end_checkpoint, c2_to_end_checkpoint)
    all_rows = parity_rows + flow_rows + state_rows
    statuses = [row["status"] for row in all_rows]
    failed = _count_parity_failures(statuses)
    if failed:
        print(f"E0047 completed with {failed} parity failures; see report and first-mismatch rows.")
    results = _write_results(protocol, e45_protocol, e45_manifest, d_bundle, flow_batch,
                             flow_batch_paths, children, parity_rows, flow_representation_summary,
                             matrix_summary, flow_rows, state_rows, resource_before, started)
    return results


def _child(mode: str) -> None:
    protocol, e45_protocol, d_bundle, transforms = load_protocol()
    e45_manifest = json.loads(E0045_MANIFEST_PATH.read_text(encoding="utf-8"))
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if mode == "build_C1":
        result = _build_cp1(protocol, e45_protocol, e45_manifest, d_bundle, transforms)
    elif mode.startswith("resume|"):
        _, checkpoint_name, stop_name, stage, save_name, snapshot_spec = mode.split("|", 5)
        point = {item["name"]: item for item in protocol["continuation"]["splits"]}
        stop_row = int(point[stop_name]["expected_prefix_rows"]) if stop_name in point else int(protocol["scope"]["safe_raw_rows"])
        snapshots = {}
        if snapshot_spec != "none":
            row, name = snapshot_spec.split("=", 1)
            snapshots[int(row)] = name
        result = _resume_suffix(checkpoint_name, stop_row, stage,
                                save_checkpoint_name=None if save_name == "none" else save_name,
                                snapshot_rows=snapshots,
                                protocol=protocol, e45_protocol=e45_protocol,
                                e45_manifest=e45_manifest, d_bundle=d_bundle, transforms=transforms)
    else:
        raise ValueError(f"unknown internal E0047 mode: {mode}")
    _json_write(RUNTIME / f"child_{mode.replace('|', '_')}.json", result)


if __name__ == "__main__":
    if os.environ.get(MODE_ENV):
        _child(os.environ[MODE_ENV])
    else:
        run_audit()
