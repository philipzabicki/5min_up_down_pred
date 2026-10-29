"""Frozen development-only E0046 FLOW representation and BASE64 continuation audit."""
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
PROTOCOL_PATH = ROOT / "configs/research/btc_D_flow_representation_base64_continuation_20261029_v1.json"
E0045_PROTOCOL_PATH = ROOT / "configs/research/btc_D_input_generator_final_bundle_fix_20261029_v2.json"
E0045_MANIFEST_PATH = ROOT / "reports/btc_preopen/btc_D_input_generator_final_bundle_fix_20261029_manifest.json"
RUNTIME = ROOT / "data/analysis/polymarket/BTC/btc_D_flow_representation_and_base64_continuation_20261029"
RAW_COLUMNS = ("Opened", "Open", "High", "Low", "Close", "Volume", "UM_BTCUSDT_Open", "UM_BTCUSDT_High", "UM_BTCUSDT_Low", "UM_BTCUSDT_Close")
FLOW_RAW_COLUMNS = ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume")
FLOW_COLUMNS = ("taker_Z_1m", "taker_V_1m", "taker_Z_5m", "taker_V_5m", "taker_Z_15m", "taker_V_15m")
FLOW_TAIL_ROWS = 10_095
PREFIX_CHUNK_ROWS = 65_536
MODE_ENV = "BTC_D_E0046_MODE"
IO_COUNTS = {"flow_archive_sha256_bytes": 0, "flow_archive_uncompressed_stream_bytes": 0}


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
    if protocol.get("status") != "FROZEN_BEFORE_REPLAY" or protocol.get("experiment_id") != "E0046":
        raise RuntimeError("E0046 protocol is not frozen for replay")
    if protocol["scope"]["evaluation_authorized"] or protocol["scope"]["activated"] or protocol["scope"]["holdout_status"] != "HOLDOUT_NOT_RUN":
        raise RuntimeError("E0046 development-only and holdout guards changed")
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
        raise RuntimeError("E0046 E0045 protocol path differs from the pinned runner input")
    e45 = json.loads(e45_protocol_path.read_text(encoding="utf-8"))
    from features import btc_D_input_generator_20261029 as e45_code

    e45_protocol = e45_code.load_protocol()
    safe_source = e45_protocol["allowed_inputs"]["safe_index_ohlcv_prefix"]
    if (e45_protocol["scope"]["source_raw_opened_exclusive_utc"] != protocol["scope"]["source_raw_opened_exclusive_utc"]
            or safe_source["path"] != protocol["scope"]["source_raw_path"]
            or safe_source["sha256"] != protocol["scope"]["source_raw_sha256"]
            or int(safe_source["rows"]) != int(protocol["scope"]["safe_raw_rows"])):
        raise RuntimeError("E0046 safe source or exclusive cutoff differs from the frozen E0045 v2 input")
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
        raise RuntimeError("E0046 final E0041 bundle identity differs from the preregistration")
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

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        return float("nan")
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
                if record["indicator"] in {"ADX", "ChaikinOsc"}}
    if set(selected) != {"ADX", "ChaikinOsc"}:
        raise RuntimeError("saved E0041 ADX/Chaikin fit records are incomplete")
    return selected


def _profile_features_for_rows(states: dict, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    from features.reaction_profile_fixed_grid import update_state_with_candle as rp_update, _extract_feature_row_array as rp_extract
    from features.volume_profile_fixed_range import update_state_with_candle as vp_update, _extract_feature_row_array as vp_extract

    rp_state = states["reaction_profile"]
    vp_state = states["volume_profile"]
    rp_cols = list(rp_state["feature_columns"])
    vp_cols = list(vp_state["feature_columns"])
    rp = np.empty((len(rows), len(rp_cols)), dtype=np.float64)
    vp = np.empty((len(rows), len(vp_cols)), dtype=np.float64)
    for idx, row in enumerate(rows.itertuples(index=False)):
        rp_update(rp_state, open=row.Open, high=row.High, low=row.Low, close=row.Close)
        rp[idx] = rp_extract(rp_state, close=row.Close)
        vp_update(vp_state, high=row.High, low=row.Low, volume=row.Volume)
        vp[idx] = vp_extract(vp_state, high=row.High, low=row.Low)
    if len(rows):
        last_time = pd.Timestamp(rows["Opened"].iloc[-1]).isoformat()
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
        "modeling_output_dir": work_dir / "feature_output", "output_suffix": "e0046_continuation",
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
        raise RuntimeError("E0046 rolling BASE64 builder wrote to an unexpected path")
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
    from features.ta_tools import get_1d_ma

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

    # Chaikin's AD line has an unbounded accumulator; apply saved ADL state,
    # then its frozen finite HMA/NWMA filters over the sufficient rolling tail.
    adl_suffix = continuation.advance_chaikin_adl(state["adl"], rolling_raw.iloc[-suffix_count:])
    adl_tail = np.asarray(state["adl_tail"], dtype=np.float64)
    adl_values = np.concatenate((adl_tail, adl_suffix))
    fast = get_1d_ma(adl_values, str(params["ChaikinOsc"]["fast_ma_type"]), int(params["ChaikinOsc"]["fast_period"]))
    slow = get_1d_ma(adl_values, str(params["ChaikinOsc"]["slow_ma_type"]), int(params["ChaikinOsc"]["slow_period"]))
    oscillator = fast - slow
    chosc = oscillator[-suffix_count:]
    previous = oscillator[-suffix_count - 1:-1]
    chaikin_name = next(name for name in base.columns if name.startswith("ChaikinOsc_fit_"))
    base[chaikin_name] = chosc
    diff_name = f"{chaikin_name}_diff"
    if diff_name in base.columns:
        base[diff_name] = chosc - previous
    state["adl_tail"] = adl_values[-continuation.TAIL_ROWS:].copy()
    return {"adx": adx, "chaikin": chosc}


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
        "schema_version": 2,
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
    raw_mismatches = 0
    raw_max_abs = 0.0
    raw_max_rel = 0.0
    for index, name in enumerate(FLOW_COLUMNS):
        actual64 = raw_common[name].to_numpy(dtype=np.float64)
        expected = canonical_values[:, index]
        after = cast[:, index]
        record = _parity_record("flow_representation_float32", name, after, expected, common["Opened"], atol=0.0, rtol=0.0, exact=True)
        rows.append(record)
        before = _parity_record("flow_representation_float64_diagnostic", name, actual64, expected, common["Opened"], atol=0.0, rtol=0.0, exact=True)
        before["status"] = "MEASURED_FLOAT32_PRECISION_LOSS" if before["numeric_mismatch_rows"] or before["mask_mismatch_rows"] else "EXACT"
        rows.append(before)
        raw_mismatches += before["numeric_mismatch_rows"] + before["mask_mismatch_rows"]
        raw_max_abs = max(raw_max_abs, float(before["max_abs_error"] or 0.0))
        raw_max_rel = max(raw_max_rel, float(before["max_rel_error"] or 0.0))
    last_allowed = pd.Timestamp(protocol["scope"]["last_allowed_raw_opened_utc"]).tz_convert("UTC")
    unmatched_after_reference = flow_batch.loc[(flow_batch["Opened"] > reference_last) & (flow_batch["Opened"] <= last_allowed)]
    return rows, {
        "common_reference_rows": int(len(common)),
        "reference_last_opened": reference_last.isoformat(),
        "last_allowed_raw_opened": last_allowed.isoformat(),
        "post_reference_flow_rows": int(len(unmatched_after_reference)),
        "post_reference_first_opened": unmatched_after_reference["Opened"].iloc[0].isoformat() if len(unmatched_after_reference) else "",
        "post_reference_last_opened": unmatched_after_reference["Opened"].iloc[-1].isoformat() if len(unmatched_after_reference) else "",
        "float64_vs_final_product_mismatch_feature_rows": int(raw_mismatches),
        "float64_vs_final_product_max_abs_error": raw_max_abs,
        "float64_vs_final_product_max_rel_error": raw_max_rel,
        "float64_diagnostic_rows": len(FLOW_COLUMNS),
        "float32_exact_mismatches": int(sum(row["numeric_mismatch_rows"] + row["mask_mismatch_rows"] for row in rows)),
        "canonical_flow_rows_in_feature_artifact": int(len(canonical)),
        "e0041_flow_formula_rows_reused_from_E0045": int(e45_manifest["flow"]["rows_compared"]),
    }


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
                matrix_rows.append(_parity_record(f"model_matrix_r{seed}", name,
                                                  actual_matrix[:, column], expected_matrix[:, column],
                                                  opened, atol=0, rtol=0, exact=True))
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
    return matrix_rows, summary


def _semantic_equal(left, right) -> bool:
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=True, check_like=False)
            return True
        except AssertionError:
            return False
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)
        except (TypeError, ValueError):
            return False
    if isinstance(left, dict) and isinstance(right, dict):
        ignored = {"checkpoint_name", "created_prefix_rows"}
        left_keys = set(left) - ignored
        right_keys = set(right) - ignored
        return left_keys == right_keys and all(_semantic_equal(left[key], right[key]) for key in left_keys)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_semantic_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)):
        return (np.isnan(left) and np.isnan(right)) or float(left) == float(right)
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


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
    states_equal = _semantic_equal(primary_checkpoint, full_checkpoint)
    rows.append({"block": f"semantic_checkpoint_{stage}", "feature": "all_generator_state", "status": "PASS" if states_equal else "FAIL",
                 "rows": common, "numeric_mismatch_rows": 0 if states_equal else 1, "mask_mismatch_rows": 0,
                 "first_mismatch_opened_utc": "", "max_abs_error": 0.0 if states_equal else None,
                 "max_rel_error": 0.0 if states_equal else None, "atol": 0.0, "rtol": 0.0})
    return rows


def _compare_final_states(protocol: dict, e45_protocol: dict, e45_manifest: dict,
                          end_from_c1: dict, end_from_c2: dict) -> list[dict]:
    from features.reaction_profile_fixed_grid import load_state as load_rp_state
    from features.volume_profile_fixed_range import load_state as load_vp_state
    from features import btc_D_input_generator_20261029 as e45_code

    rows = []
    lineage = e45_protocol["allowed_inputs"]["base_feature_lineage"]
    for family, loader in (("reaction_profile", load_rp_state), ("volume_profile", load_vp_state)):
        record = lineage[f"{family}_state"]
        expected = loader(ROOT / record["base_path"])
        left = end_from_c1["profile_states"][family]
        right = end_from_c2["profile_states"][family]
        equal_reference = _semantic_equal(left, expected)
        equal_paths = _semantic_equal(left, right)
        rows.append({"block": "final_profile_state", "feature": family,
                     "status": "PASS" if equal_reference and equal_paths else "FAIL",
                     "rows": int(protocol["scope"]["safe_raw_rows"]),
                     "numeric_mismatch_rows": 0 if equal_reference and equal_paths else 1,
                     "mask_mismatch_rows": 0, "first_mismatch_opened_utc": "",
                     "max_abs_error": 0.0 if equal_reference and equal_paths else None,
                     "max_rel_error": 0.0 if equal_reference and equal_paths else None,
                     "atol": 0.0, "rtol": 0.0,
                     "matches_E0041_final_state": bool(equal_reference),
                     "matches_independent_resume_path": bool(equal_paths)})

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
        raise RuntimeError(f"E0046 child failed ({mode}, exit {return_code}); log={log_path}\n{tail}")
    output_path = RUNTIME / f"child_{mode.replace('|', '_')}.json"
    if not output_path.is_file():
        raise RuntimeError(f"E0046 child did not write its result: {output_path}")
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
        elif name.startswith("ChaikinOsc_fit_"):
            recurrence = "Cumulative ADL value and seen count serialized; the actual ADL tail feeds frozen HMA(893)/NWMA(8) filters."
            tail = "12,001 ADL/raw rows; exceeds maximum selected finite Chaikin filter window."
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
        "experiment_id": "E0046",
        "parent_experiment_id": "E0045",
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
    prediction_rows = [row for row in all_rows if row["block"].startswith(("final_D_", "canonical_D_"))]
    status_groups = {
        "FLOW_formula": formula_pass,
        "FLOW_representation": bool(flow_representation_summary["float32_exact_mismatches"] == 0 and all(row["status"] == "PASS" for row in flow_rows if row["block"] == "flow_representation_float32")),
        "model_matrices": bool(matrix_summary["rows_compared_per_seed"] == protocol["scope"]["expected_common_model_rows"] and len(matrix_rows) == matrix_summary["matrix_count"] * 86 and all(row["status"] == "PASS" for row in matrix_rows)),
        "BASE64_continuation": bool(len(base_rows) == 64 * 3 and all(row["status"] == "PASS" for row in base_rows) and all(row["status"] == "PASS" for row in all_rows if row["block"].startswith("independent_continuation_") and "base64" in row["block"])),
        "FLOW_continuation": bool(len(flow_continuation_rows) == 6 * 3 and all(row["status"] == "PASS" for row in flow_continuation_rows) and all(row["status"] == "PASS" for row in all_rows if row["block"].startswith("independent_continuation_") and "flow6" in row["block"])),
        "memory_ESN_continuation": bool(len(dynamic_continuation_rows) == 32 * 3 and all(row["status"] == "PASS" for row in dynamic_continuation_rows) and all(row["status"] == "PASS" for row in all_rows if row["block"].startswith("independent_continuation_") and "dynamic" in row["block"])),
        "final_D_predictions": bool(len(prediction_rows) == 22 and all(row["status"] == "PASS" for row in prediction_rows)),
        "final_states": bool(len(state_rows) == 3 and all(row["status"] == "PASS" for row in state_rows)),
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
    report_manifest = {
        "schema_version": 1,
        "experiment_id": "E0046", "task_id": "T0016", "decision_id": "D0024",
        "status": overall, "statuses": status_groups,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "parent_experiment_id": "E0045", "parent_manifest_sha256": sha256_file(E0045_MANIFEST_PATH),
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
            "prefix_raw_rows_read": int(cp1["rows"]),
            "prefix_raw_csv_bytes_read": int(cp1["byte_offset"]),
            "prefix_raw_read_seconds": cp1["raw_read_seconds"],
            "C1_checkpoint_build_seconds": cp1["checkpoint_build_seconds"],
            "C1_checkpoint_write_seconds": cp1["checkpoint_write_seconds"],
            "suffix_raw_rows_recomputed_total": int(sum(item["rows_recomputed"] for item in child_results.values() if isinstance(item, dict) and "rows_recomputed" in item)),
            "suffix_raw_csv_bytes_read_total": int(sum(int(item.get("raw_bytes_read", 0)) for item in child_results.values() if isinstance(item, dict))),
            "base64_builder_input_rows_total": int(sum(int(item.get("base64_builder_input_rows", 0)) for item in child_results.values() if isinstance(item, dict))),
            "base64_raw_tail_rows_recomputed_total": int(sum(int(item.get("base64_raw_tail_rows_recomputed", 0)) for item in child_results.values() if isinstance(item, dict))),
            "flow_formula_input_rows_total": int(sum(int(item.get("flow_formula_input_rows", 0)) for item in child_results.values() if isinstance(item, dict))),
            "flow_raw_tail_rows_recomputed_total": int(sum(int(item.get("flow_raw_tail_rows_recomputed", 0)) for item in child_results.values() if isinstance(item, dict))),
            "state_read_seconds_total": float(sum(float(item.get("state_read_seconds", 0.0)) for item in child_results.values() if isinstance(item, dict))),
            "resume_stage_seconds": {name: float(sum(item.get("stage_seconds", {}).get(name, 0.0) for item in child_results.values() if isinstance(item, dict)))
                                     for name in {key for item in child_results.values() if isinstance(item, dict) for key in item.get("stage_seconds", {})}},
            "flow_archive_sha256_bytes_read_total": flow_archive_hash_bytes,
            "flow_archive_uncompressed_stream_bytes_read_total": flow_archive_stream_bytes,
            "comparison_parquet_input_bytes_lower_bound": input_sizes,
            "measured_input_bytes_lower_bound": int(cp1["byte_offset"] + suffix_raw_bytes + flow_archive_hash_bytes + flow_archive_stream_bytes + sum(input_sizes.values())),
            "checkpoint_bytes": {
                "C1": child_results["build_C1"]["checkpoint_bytes"],
                "C2_primary": child_results["C1_to_C2"]["saved_checkpoint"]["bytes"],
                "C2_from_C1_independent": child_results["C1_to_END"]["intermediate_checkpoints"][0]["bytes"],
                "END_from_C1": child_results["C1_to_END"]["saved_checkpoint"]["bytes"],
                "END_from_C2": child_results["C2_to_END"]["saved_checkpoint"]["bytes"],
            },
            "matrix_comparison_seconds": matrix_summary["matrix_comparison_seconds"],
            "prediction_seconds": matrix_summary["prediction_seconds"],
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
    report = f"""# E0046 - FLOW representation and stateful BASE64 continuation

1. Czy roznice FLOW w calosci wyjasnia historyczna konwersja? **{'Tak' if status_groups['FLOW_representation'] else 'Nie'}**. Surowe FLOW6 obliczono jako `float64`. E0041 wykonuje dokladny join `Opened` w [`run_btc_preopen_experiment.py`](../../run_btc_preopen_experiment.py), `_join_causal_candidate_features` (linie 1311-1375): wyznacza indeksy, dopasowuje wiersze, rzutuje dopasowane wartosci na `float32`, pozostawia niedopasowania jako NaN i zapisuje kolumny Arrow `float32`. Join nie imputuje ani nie transformuje FLOW. Macierze sa laczone w `_family_matrix`; finalizer rzutuje wejscie na `float32` przed LightGBM. Po wiernym cast/join pozostalo **{flow_representation_summary['float32_exact_mismatches']}** niezgodnosci w {flow_representation_summary['common_reference_rows']:,} wspolnych wierszach x 6 cech. Wczesniejsze porownanie float64 do produktu zapisalo {flow_representation_summary['float64_vs_final_product_mismatch_feature_rows']:,} roznych wartosci; max abs {flow_representation_summary['float64_vs_final_product_max_abs_error']:.9g}, max rel {flow_representation_summary['float64_vs_final_product_max_rel_error']:.9g}. E0045 zachowuje dowod formuly float64: 6/6 cech, {formula_rows[0]['rows']:,} wierszy, tolerancja 1e-10/1e-10, max abs {formula_max:.9g}. Cztery wiersze po ostatniej referencji FLOW ({flow_representation_summary['post_reference_first_opened']} - {flow_representation_summary['post_reference_last_opened']}) sa osobno oznaczone.

2. Czy wszystkie macierze wejsciowe finalnego D sa zgodne na calym dozwolonym wspolnym zakresie? **{'Tak' if status_groups['model_matrices'] else 'Nie'}**. Porownano {matrix_summary['matrix_count']} macierze x 86 uporzadkowanych cech x {matrix_summary['rows_compared_per_seed']:,} wierszy, po historycznym cast `float32`, na kluczach `Opened` od {matrix_summary['first_opened']} do {matrix_summary['last_opened']}. Nie uzyto etykiet.

3. Czy kazdy z 64 BASE komponentow wznowil prace w swiezym procesie i czy oba rzeczywiste sufiksy oraz stany koncowe sa zgodne? **{'Tak' if status_groups['BASE64_continuation'] and status_groups['final_states'] else 'Nie'}**. C1: {suffix_summary['C1_to_END']['rows']:,} wierszy, {suffix_summary['C1_to_END']['first_opened']} - {suffix_summary['C1_to_END']['last_opened']}; C2: {suffix_summary['C2_to_END']['rows']:,} wierszy, {suffix_summary['C2_to_END']['first_opened']} - {suffix_summary['C2_to_END']['last_opened']}. Kazdy checkpoint zapisano atomowo, a kazde wznowienie dzialalo w osobnym interpreterze. C1->C2->koniec oraz niezalezne C1->koniec uzgodnily stan semantyczny.

4. Czy dziewiec modeli, agregacja i Platt zachowaly zgodnosc na panelu? **{'Tak' if status_groups['final_D_predictions'] else 'Nie'}**. Sprawdzono {matrix_summary['panel_rows']} wierszy panelu E0045, w tym {matrix_summary['decision_panel_rows']} minut decyzyjnych i {matrix_summary['finalization_panel_rows_2026_06_30_2250_to_2353']} minut finalizacyjnych 2026-06-30 22:50-23:53; bez etykiet i metryk jakosci.

5. Jaki jest czas, peak RAM, I/O, ilosc ponownie przeliczonych danych i rozmiar checkpointu? Calkowity czas {report_manifest['resource']['wall_seconds']:.1f} s; maksymalny peak RSS parent/worker {report_manifest['resource']['maximum_peak_rss_gib']:.2f} GiB. Prefiks: {report_manifest['resource']['prefix_raw_rows_read']:,} raw wierszy i {report_manifest['resource']['prefix_raw_csv_bytes_read']:,} bajtow; nowe wiersze suffixow przetworzone we wszystkich wznowieniach: {report_manifest['resource']['suffix_raw_rows_recomputed_total']:,}. Builder BASE64 ponownie obliczyl {report_manifest['resource']['base64_raw_tail_rows_recomputed_total']:,} wierszy ogona, a FLOW6 {report_manifest['resource']['flow_raw_tail_rows_recomputed_total']:,}; wiersze wejsciowe buildera BASE64 lacznie: {report_manifest['resource']['base64_builder_input_rows_total']:,}; raw bajty odczytane przy resume: {report_manifest['resource']['suffix_raw_csv_bytes_read_total']:,}. Hashowanie archiwow i odczyt ich strumieni FLOW: {report_manifest['resource']['flow_archive_sha256_bytes_read_total'] + report_manifest['resource']['flow_archive_uncompressed_stream_bytes_read_total']:,} bajtow. Rozmiary checkpointow i czasy budowy, odczytu stanu, etapow wznowienia oraz zapisu sa zapisane w manifeście. Projekcje Parquet stanowia osobno raportowana dolna granice I/O.

## Statusy

| Blok | Wynik |
|---|---|
{summary_table}

## Zakres i reprodukcja

E0046 uzywa wylacznie bezpiecznego raw development do `Opened=2026-06-30T23:57Z`; nie odczytano payloadu `23:58Z`. Ogon BASE64 ma {continuation.TAIL_ROWS:,} wierszy. Streak, RP/VP, ADX TEMA, Chaikin ADL, memory i ESN zachowuja stany; FLOW6 utrzymuje 10,095 raw wierszy. Konfiguracje, transformaty i dziewiec modeli E0041 pozostaly zamrozone. Nie wykonano refitow, selekcji, treningu, metryk jakosci ani ekonomiki.

`{overall}` opisuje tylko generator developmentowy. Nie zamyka bramki zrodel holdoutu ani nie oznacza rentownosci. Jesli wszystkie bloki przejda, przyszly osobny etap wymaga surowych Binance index/ohlcv i USD-M FLOW, danych futures/index do basis, pelnych ksiazek ze znacznikami venue/receive i ciagloscia, a nastepnie projekcji BASE70+dynamic z zachowaniem warmup i stanowych transformow. E0043 pozostawia nierozwiazana luke 68 godzin `[2026-08-15T10:00Z,2026-08-18T06:00Z)` bez kwalifikujacych pelnych ksiazek, dwa niedopasowane lokalne starty (`2026-07-12T02:25Z`, `2026-09-03T15:20Z`) i 1,115 okien bez potwierdzonej kompletnej historii. Osobny guard evaluatora nadal musi bronic granicy danych i wymagac odrebnego upowaznienia. Historycznych wejsc holdoutu ani evaluatora tu nie uruchomiono.

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
    e45_manifest = json.loads(E0045_MANIFEST_PATH.read_text(encoding="utf-8"))
    ram_before = _memory_available_gib()
    disk_before = shutil.disk_usage(ROOT).free / 1024**3
    if ram_before < float(protocol["resource_limits"]["minimum_available_ram_gib_before_run"]):
        raise RuntimeError(f"E0046 RAM gate requires 24 GiB available; observed {ram_before:.2f} GiB")
    if disk_before < float(protocol["resource_limits"]["minimum_free_workspace_disk_gib"]):
        raise RuntimeError(f"E0046 disk gate requires 80 GiB free; observed {disk_before:.2f} GiB")
    try:
        import numba
        numba.set_num_threads(4)
    except Exception as exc:
        raise RuntimeError("could not enforce the frozen four-thread Numba limit") from exc
    resource_before = {"available_ram_gib": ram_before, "free_disk_gib": disk_before}
    RUNTIME.mkdir(parents=True, exist_ok=True)

    children = {}
    children["build_C1"] = _run_child("build_C1")
    split2 = int(protocol["continuation"]["splits"][1]["expected_prefix_rows"])
    children["C1_to_C2"] = _run_child(f"resume|C1|C2|C1_to_C2|C2_primary|none")
    children["C2_to_END"] = _run_child(f"resume|C2_primary|END|C2_to_END|END_from_C2|none")
    children["C1_to_END"] = _run_child(f"resume|C1|END|C1_to_END|END_from_C1|{split2}=C2_from_C1_full")
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
    parity_rows = []
    for stage in ("C1_to_C2", "C2_to_END", "C1_to_END"):
        parity_rows.extend(_compare_suffix_stage(stage, children[stage], e45_protocol, e45_manifest,
                                                 d_bundle["manifest"], flow_batch))
    c1_to_c2_checkpoint = continuation.load_checkpoint(RUNTIME / "checkpoints" / "C2_primary.json",
                                                        identity_sha256=_continuation_identity(protocol, d_bundle["identity"]))
    c1_end_c2_checkpoint = continuation.load_checkpoint(RUNTIME / "checkpoints" / "C2_from_C1_full.json",
                                                          identity_sha256=_continuation_identity(protocol, d_bundle["identity"]))
    parity_rows.extend(_compare_continuation_paths(children["C1_to_C2"], children["C1_to_END"],
                                                   c1_to_c2_checkpoint, c1_end_c2_checkpoint, "C1_to_C2"))
    c2_overlap_count = int(protocol["continuation"]["splits"][1]["expected_nonempty_suffix_rows"])
    c1_to_end_tail = dict(children["C1_to_END"])
    c1_to_end_tail["source_row_start"] = split2
    c1_to_end_tail["rows_recomputed"] = c2_overlap_count
    c1_to_end_tail["first_suffix_opened"] = protocol["continuation"]["splits"][1]["first_suffix_opened_utc"]
    for name in ("base64", "flow6", "dynamic"):
        frame = pd.read_parquet(ROOT / children["C1_to_END"][name]["path"])
        tail_path = RUNTIME / "suffixes" / f"C1_to_END_tail_C2_{name}.parquet"
        frame.iloc[-c2_overlap_count:].to_parquet(tail_path, index=False)
        c1_to_end_tail[name] = {"path": str(tail_path.relative_to(ROOT)), "rows": c2_overlap_count,
                                "sha256": sha256_file(tail_path), "bytes": tail_path.stat().st_size}
    c2_to_end_checkpoint = continuation.load_checkpoint(RUNTIME / "checkpoints" / "END_from_C2.json",
                                                         identity_sha256=_continuation_identity(protocol, d_bundle["identity"]))
    c1_to_end_checkpoint = continuation.load_checkpoint(RUNTIME / "checkpoints" / "END_from_C1.json",
                                                          identity_sha256=_continuation_identity(protocol, d_bundle["identity"]))
    parity_rows.extend(_compare_continuation_paths(children["C2_to_END"], c1_to_end_tail,
                                                   c2_to_end_checkpoint, c1_to_end_checkpoint, "C2_to_END"))
    matrix_rows, matrix_summary = _compare_model_matrices_and_predictions(protocol, e45_protocol,
                                                                            e45_manifest, d_bundle, flow_batch)
    parity_rows.extend(matrix_rows)
    state_rows = _compare_final_states(protocol, e45_protocol, e45_manifest,
                                       c1_to_end_checkpoint, c2_to_end_checkpoint)
    all_rows = parity_rows + flow_rows + state_rows
    diagnostic_statuses = {"MEASURED_FLOAT32_PRECISION_LOSS", "EXACT"}
    statuses = [row["status"] for row in all_rows]
    failed = sum(status != "PASS" and status not in diagnostic_statuses for status in statuses)
    if failed:
        print(f"E0046 completed with {failed} parity failures; see report and first-mismatch rows.")
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
        raise ValueError(f"unknown internal E0046 mode: {mode}")
    _json_write(RUNTIME / f"child_{mode.replace('|', '_')}.json", result)


if __name__ == "__main__":
    if os.environ.get(MODE_ENV):
        _child(os.environ[MODE_ENV])
    else:
        run_audit()
