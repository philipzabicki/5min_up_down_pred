"""Run or resume the frozen BTC taker-flow comparison without CLI arguments."""
from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import math
import os
import threading
import time
import uuid
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import requests
from sklearn.metrics import log_loss, roc_auc_score

import optimize_lgbm_hyperparameters as model_tuning
import run_btc_preopen_experiment as experiment
import run_btc_preopen_model_update as model_update
import run_btc_preopen_research as research
from features.btc_taker_flow import compute_taker_flow_features, feature_names
from utils.cv_objective import ObjectiveAlignedCVCallback, cv_objective_series
from utils.metrics import (
    make_lightgbm_binary_balanced_accuracy_eval,
    make_lightgbm_binary_brier_eval,
    make_lightgbm_binary_logloss_eval,
)
from utils.polymarket_policy import calibrate, fit_calibrator


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_preopen_taker_flow_20261018.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_directory"]
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
REPORT_PATH = ROOT / CONFIG["report_path"]
STATUS_PATH = ROOT / CONFIG["status_path"]
STATUS_BEGIN = "<!-- BTC_TAKER_FLOW_20261018_BEGIN -->"
STATUS_END = "<!-- BTC_TAKER_FLOW_20261018_END -->"
SOURCE_ROOT = ROOT / Path(CONFIG["source_run"]["run_manifest_path"]).parent
SOURCE_DATA_PATH = ROOT / CONFIG["source_run"]["feature_dataset_path"]
SOURCE_RUN_PATH = ROOT / CONFIG["source_run"]["run_manifest_path"]
SOURCE_CONFIG_PATH = ROOT / CONFIG["source_run"]["effective_config_path"]
SOURCE_SELECTION_PATH = ROOT / CONFIG["source_run"]["selected_features_path"]
SOURCE_TUNING_PATH = ROOT / CONFIG["source_run"]["model_tuning_result_path"]
SOURCE_TRAINING_PATH = ROOT / CONFIG["source_run"]["training_manifest_path"]
FEATURE_DATASET_METADATA_PATH = SOURCE_DATA_PATH.with_name(
    SOURCE_DATA_PATH.stem + "_metadata.json"
)
SOURCE_CALIBRATION_PATH = ROOT / SOURCE_ROOT.joinpath(
    "stages/calibration_oof_9643e6c800d1/development_calibration_oof.parquet"
)
MARKET_INDEX_PATH = ROOT / CONFIG["official_outcomes"]["source_path"]
END_UTC = pd.Timestamp(CONFIG["data_end_inclusive_utc"])
START_UTC = pd.Timestamp(CONFIG["development_start_utc"])
CUTOFF_UTC = pd.Timestamp(CONFIG["development_cutoff_exclusive_utc"])
HORIZONS = tuple(int(x) for x in CONFIG["flow_features"]["horizons_minutes"])
CONTROL_FLOW_FEATURES, EXTENSION_FEATURES = feature_names(HORIZONS)
FLOW_FEATURES = CONTROL_FLOW_FEATURES + EXTENSION_FEATURES
SOURCE_FEATURES: list[str] = []
SOURCE_HASHES: dict = {}
SOURCE_TRAINING: dict = {}
SOURCE_TUNING: dict = {}
SOURCE_WEIGHT_RESULT: dict = {}
SOURCE_FOLD_RECORDS: list[dict] = []
MODEL_PARAMETERS: dict = {}
MODEL_SEED = 0
DECISION_WEIGHT = 0.0
AUXILIARY_WEIGHT = 0.0
CV_FOLD_COUNT = 0
MAX_ESTIMATORS = 0
EARLY_STOPPING_ROUNDS = 0
THREADS = int(CONFIG["model_comparison"]["execution_threads"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values, dtype=None) -> str:
    array = np.asarray(values, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _record(path: Path) -> dict:
    return {"path": _relative(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _verify_records(records: list[dict]) -> bool:
    for record in records:
        path = ROOT / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or _sha256(path) != record["sha256"]
        ):
            return False
    return bool(records)


def _current_memory_bytes() -> int:
    if os.name == "nt":
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        process = kernel32.GetCurrentProcess()
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb,
        )
        if not ok:
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage * (1024 if os.name != "darwin" else 1))
    except ImportError:
        return 0


class PeakMemorySampler:
    def __init__(self, interval_seconds: float = 0.5):
        self.interval_seconds = float(interval_seconds)
        self.peak_bytes = _current_memory_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.wait(self.interval_seconds):
            try:
                self.peak_bytes = max(self.peak_bytes, _current_memory_bytes())
            except OSError:
                continue

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, _current_memory_bytes())


def _resource_snapshot() -> dict:
    snapshot = {
        "logical_processors": os.cpu_count(),
        "lightgbm_threads": THREADS,
        "pyarrow_cpu_threads": pa.cpu_count(),
        "pyarrow_io_threads": pa.io_thread_count(),
        "native_library_thread_limit": THREADS,
    }
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            snapshot["total_memory_gib"] = round(status.ullTotalPhys / 2**30, 2)
            snapshot["available_memory_gib"] = round(status.ullAvailPhys / 2**30, 2)
    return snapshot


def _status_update(manifest: dict) -> None:
    stages = manifest.get("stages", {})
    active = manifest.get("active_stage") or "none"
    completed = sorted(name for name, stage in stages.items() if stage.get("status") == "completed")
    lines = [
        STATUS_BEGIN,
        "### BTC taker-flow experiment - 2026-10-18",
        f"- Updated UTC: `{pd.Timestamp.now(tz='UTC').isoformat()}`.",
        f"- Status: `{manifest.get('status')}`; active stage: `{active}`.",
        f"- Completed stages: `{', '.join(completed) or 'none'}`.",
        f"- Next step: `{manifest.get('next_step', 'Resume the no-argument runner.')}`.",
        f"- Run manifest: `{_relative(MANIFEST_PATH)}`.",
        STATUS_END,
        "",
    ]
    text = STATUS_PATH.read_text(encoding="utf-8")
    block = "\n".join(lines)
    if STATUS_BEGIN in text and STATUS_END in text:
        before, tail = text.split(STATUS_BEGIN, 1)
        _, after = tail.split(STATUS_END, 1)
        text = before.rstrip() + "\n" + block + after.lstrip("\n")
    else:
        anchor = "# BTC research execution status"
        if anchor in text:
            text = text.replace(anchor, block + anchor, 1)
        else:
            text = block + text
    temporary = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(STATUS_PATH)


def _begin_stage(manifest: dict, name: str, next_step: str) -> None:
    manifest["status"] = "running"
    manifest["active_stage"] = name
    manifest["next_step"] = next_step
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    manifest.setdefault("stages", {})[name] = {
        "status": "running",
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write_json(MANIFEST_PATH, manifest)
    _status_update(manifest)


def _finish_stage(manifest: dict, name: str, **payload) -> None:
    current = manifest.setdefault("stages", {}).get(name, {})
    current.update(payload)
    current.update({
        "status": "completed",
        "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    })
    manifest["stages"][name] = current
    manifest["active_stage"] = None
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(MANIFEST_PATH, manifest)
    _status_update(manifest)


def _load_source_artifacts() -> dict:
    global SOURCE_FEATURES, SOURCE_TRAINING, SOURCE_TUNING, SOURCE_WEIGHT_RESULT
    global SOURCE_FOLD_RECORDS, MODEL_PARAMETERS, MODEL_SEED, DECISION_WEIGHT
    global AUXILIARY_WEIGHT, CV_FOLD_COUNT, MAX_ESTIMATORS, EARLY_STOPPING_ROUNDS
    global SOURCE_HASHES

    source_run = _read_json(SOURCE_RUN_PATH)
    effective = _read_json(SOURCE_CONFIG_PATH)
    selected = _read_json(SOURCE_SELECTION_PATH)
    best = _read_json(SOURCE_TUNING_PATH)
    SOURCE_TRAINING = _read_json(SOURCE_TRAINING_PATH)
    SOURCE_TUNING = source_run["stages"]["model_tuning"]["result"]
    SOURCE_WEIGHT_RESULT = source_run["stages"]["target_weights"]["result"]
    SOURCE_FOLD_RECORDS = source_run["stages"]["calibration_oof"]["result"]["folds"]

    if source_run.get("status") != "completed":
        raise RuntimeError("The selected full-development source run is not completed")
    if SOURCE_TRAINING.get("params", {}).get("objective") != "binary":
        raise RuntimeError("The selected source is not the completed binary classifier")
    if effective["contract"]["training_target"]["name"] != "target_preopen_5m_up_binance_index_proxy":
        raise RuntimeError("The source binary model target differs from the pre-open proxy target")
    if len(selected.get("selected_features", [])) != 64:
        raise RuntimeError("The source feature-selection artifact does not contain 64 features")
    SOURCE_FEATURES = list(selected["selected_features"])
    if SOURCE_TRAINING.get("feature_order") != SOURCE_FEATURES:
        raise RuntimeError("The selected feature artifact and completed model order differ")
    if best.get("best_params") != SOURCE_TUNING.get("best_params"):
        raise RuntimeError("The saved best-result artifact and completed run parameters differ")
    if SOURCE_TRAINING.get("params", {}).get("num_leaves") != best["best_params"].get("num_leaves"):
        raise RuntimeError("The final binary model does not match the saved best tuning parameters")

    profile = effective["profile"]
    configured_backend = CONFIG["model_comparison"]["compute_backend_from_source_effective_config"]
    if effective["profile"].get("compute_backend") != configured_backend or SOURCE_TRAINING["params"].get("device_type") != configured_backend:
        raise RuntimeError("The saved source model backend differs from the frozen experiment configuration")
    MODEL_PARAMETERS = dict(SOURCE_TUNING["best_params"])
    MODEL_SEED = int(SOURCE_TRAINING["params"]["seed"])
    DECISION_WEIGHT = float(SOURCE_WEIGHT_RESULT["decision_weight"])
    AUXILIARY_WEIGHT = float(SOURCE_WEIGHT_RESULT["auxiliary_row_weight"])
    CV_FOLD_COUNT = int(profile["model_cv_folds"])
    MAX_ESTIMATORS = int(profile["maximum_estimators"])
    EARLY_STOPPING_ROUNDS = int(profile["early_stopping_rounds"])
    expected_auxiliary = (1.0 - DECISION_WEIGHT) / 4.0
    if not np.isclose(AUXILIARY_WEIGHT, expected_auxiliary, rtol=0.0, atol=1e-15):
        raise RuntimeError("Saved auxiliary-row weight does not match the completed weight artifact")
    if CV_FOLD_COUNT != int(CONFIG["model_comparison"]["fold_count_from_source_effective_config"]):
        raise RuntimeError("Experiment config's fold count differs from the completed source run")

    pipeline_manifest = _read_json(SOURCE_RUN_PATH)
    dataset_outputs = pipeline_manifest["stages"]["final_feature_dataset"]["outputs"]
    dataset_record = next(item for item in dataset_outputs if item["path"].endswith(".parquet"))
    if Path(dataset_record["path"]).as_posix() != Path(CONFIG["source_run"]["feature_dataset_path"]).as_posix():
        raise RuntimeError("Configured model-ready cache is not the completed source dataset")
    dataset_metadata = _read_json(FEATURE_DATASET_METADATA_PATH)
    model_path = SOURCE_TRAINING_PATH.parent / "lgbm_model.txt"
    source_hashes = {
        "experiment_config_sha256": _sha256(CONFIG_PATH),
        "source_run_manifest_sha256": _sha256(SOURCE_RUN_PATH),
        "source_effective_config_sha256": _sha256(SOURCE_CONFIG_PATH),
        "selected_features_sha256": _sha256(SOURCE_SELECTION_PATH),
        "model_tuning_best_result_sha256": _sha256(SOURCE_TUNING_PATH),
        "source_training_manifest_sha256": _sha256(SOURCE_TRAINING_PATH),
        "source_model_sha256": _sha256(model_path),
        "source_feature_dataset_sha256": dataset_record["sha256"],
        "source_feature_dataset_metadata_sha256": _sha256(FEATURE_DATASET_METADATA_PATH),
        "source_calibration_oof_sha256": _sha256(SOURCE_CALIBRATION_PATH),
        "official_market_index_sha256": _sha256(MARKET_INDEX_PATH),
        "source_raw_data_sha256": effective["input_data"]["sha256"],
        "source_code_sha256": {
            path: _sha256(ROOT / path)
            for path in (
                "run_btc_preopen_experiment.py",
                "optimize_lgbm_hyperparameters.py",
                "train_lgbm.py",
                "utils/cv_objective.py",
                "features/btc_preopen_contract.py",
            )
        },
        "experiment_code_sha256": {
            "runner": _sha256(Path(__file__)),
            "feature_module": _sha256(ROOT / "features/btc_taker_flow.py"),
        },
    }
    model_outputs = source_run["stages"]["final_model"]["outputs"]
    model_record = next(item for item in model_outputs if item["path"].endswith("lgbm_model.txt"))
    if model_path.stat().st_size != int(model_record["size_bytes"]):
        raise RuntimeError("The source binary model size differs from its completed-run manifest")
    if _sha256(model_path) != SOURCE_TRAINING.get("model_sha256"):
        raise RuntimeError("The source binary model hash differs from its completed training manifest")
    if dataset_record["size_bytes"] != SOURCE_DATA_PATH.stat().st_size:
        raise RuntimeError("The source feature cache size differs from its completed-run manifest")
    if _sha256(SOURCE_DATA_PATH) != dataset_record["sha256"]:
        raise RuntimeError("The source feature cache hash differs from its completed-run manifest")
    if _sha256(SOURCE_CALIBRATION_PATH) != source_run["stages"]["calibration_oof"]["outputs"][0]["sha256"]:
        raise RuntimeError("The source development OOF artifact hash differs from its completed-run manifest")
    SOURCE_HASHES = source_hashes
    return {
        "run_manifest": source_run,
        "effective_config": effective,
        "dataset_metadata": dataset_metadata,
        "dataset_record": dataset_record,
        "source_hashes": source_hashes,
        "source_model_path": _relative(model_path),
    }


def _month_range(first_month: str, last_month: str) -> list[str]:
    start = pd.Period(first_month, freq="M")
    end = pd.Period(last_month, freq="M")
    return [str(period) for period in pd.period_range(start, end, freq="M")]


def _download_month(month: str, cache_dir: Path, manifest: dict) -> dict:
    name = f"BTCUSDT-1m-{month}.zip"
    zip_path = cache_dir / name
    checksum_path = cache_dir / f"{name}.CHECKSUM"
    base_url = CONFIG["binance_source"]["base_url"].rstrip("/")
    session = requests.Session()
    checksum_text = None
    for attempt in range(1, 4):
        try:
            if checksum_path.is_file():
                checksum_text = checksum_path.read_text(encoding="utf-8").strip()
            else:
                response = session.get(f"{base_url}/{name}.CHECKSUM", timeout=(15, 60))
                response.raise_for_status()
                checksum_text = response.text.strip()
                checksum_temp = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
                checksum_temp.write_text(checksum_text + "\n", encoding="utf-8")
                checksum_temp.replace(checksum_path)
            parts = checksum_text.split()
            if len(parts) < 2 or parts[-1].lstrip("*") != name:
                raise RuntimeError(f"Malformed Binance CHECKSUM for {name}: {checksum_text!r}")
            expected_hash = parts[0].lower()
            if zip_path.is_file():
                observed = _sha256(zip_path)
                if observed != expected_hash:
                    raise RuntimeError(
                        f"Cached Binance archive checksum changed for {month}; preserved existing ZIP"
                    )
            else:
                temporary = zip_path.with_suffix(zip_path.suffix + ".partial")
                if temporary.is_file() and _sha256(temporary) == expected_hash:
                    with zipfile.ZipFile(temporary) as archive:
                        if archive.testzip() is not None:
                            raise RuntimeError(f"Corrupt resumed ZIP member inside {name}")
                    temporary.replace(zip_path)
                    continue_download = False
                else:
                    continue_download = True
                if not continue_download:
                    observed = expected_hash
                else:
                    offset = temporary.stat().st_size if temporary.is_file() else 0
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    response = session.get(
                        f"{base_url}/{name}", headers=headers, stream=True, timeout=(15, 120),
                    )
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        temporary.unlink(missing_ok=True)
                        raise RuntimeError(f"Binance returned an invalid resume range for {month}")
                    if offset and not append:
                        offset = 0
                    digest = hashlib.sha256()
                    if append:
                        with temporary.open("rb") as existing:
                            for block in iter(lambda: existing.read(8 * 1024 * 1024), b""):
                                digest.update(block)
                    mode = "ab" if append else "wb"
                    with temporary.open(mode) as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                                digest.update(chunk)
                    observed = digest.hexdigest()
                    if observed != expected_hash:
                        temporary.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"Binance archive CHECKSUM mismatch for {month}: {observed} != {expected_hash}"
                        )
                    with zipfile.ZipFile(temporary) as archive:
                        csv_members = [item for item in archive.namelist() if item.lower().endswith(".csv")]
                        if len(csv_members) != 1:
                            raise RuntimeError(f"Expected one CSV inside Binance archive {name}")
                        if archive.testzip() is not None:
                            raise RuntimeError(f"Corrupt ZIP member inside {name}")
                    temporary.replace(zip_path)
            record = {
                "month": month,
                "zip_path": _relative(zip_path),
                "checksum_path": _relative(checksum_path),
                "sha256": expected_hash,
                "size_bytes": zip_path.stat().st_size,
                "checksum_source_url": f"{base_url}/{name}.CHECKSUM",
                "data_source_url": f"{base_url}/{name}",
            }
            manifest.setdefault("binance_month_archives", {})[month] = record
            _write_json(MANIFEST_PATH, manifest)
            return record
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * attempt)
    raise AssertionError("unreachable")


def _download_archives(manifest: dict) -> list[dict]:
    cache_dir = OUTPUT_DIR / CONFIG["binance_source"]["cache_subdirectory"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    months = _month_range(
        CONFIG["binance_source"]["first_month"],
        CONFIG["binance_source"]["last_month"],
    )
    records = []
    for index, month in enumerate(months, start=1):
        record = _download_month(month, cache_dir, manifest)
        records.append(record)
        if index % 6 == 0 or index == len(months):
            print(
                f"[btc-taker-flow] Binance monthly klines checked {index}/{len(months)} "
                f"through {month}", flush=True,
            )
    return records


def _load_and_validate_archives(records: list[dict]) -> tuple[pd.DataFrame, dict]:
    columns = [
        "open_time", "open", "high", "low", "close", "base_volume", "close_time",
        "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume",
        "ignore",
    ]
    selected = ["open_time", "open", "high", "low", "close", "close_time", "quote_volume", "taker_buy_quote_volume"]
    per_month = []
    month_audit = []
    invalid_samples = []
    for record in records:
        archive_path = ROOT / record["zip_path"]
        if _sha256(archive_path) != record["sha256"]:
            raise RuntimeError(f"Binance archive changed after checksum validation: {record['month']}")
        with zipfile.ZipFile(archive_path) as archive:
            member = next(item for item in archive.namelist() if item.lower().endswith(".csv"))
            with archive.open(member) as handle:
                frame = pd.read_csv(
                    handle,
                    header=None,
                    names=columns,
                    usecols=list(range(11)),
                )
        frame = frame.loc[:, selected]
        count = len(frame)
        open_time = pd.to_numeric(frame["open_time"], errors="coerce")
        close_time = pd.to_numeric(frame["close_time"], errors="coerce")
        timestamps = pd.to_datetime(open_time, unit="ms", utc=True, errors="coerce")
        timestamp_valid = timestamps.notna() & (open_time % 60_000 == 0)
        month_valid = timestamps.dt.strftime("%Y-%m").eq(record["month"])
        close_time_valid = close_time.eq(open_time + 59_999)
        frame["Opened"] = timestamps
        for name in ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"):
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        price_valid = (
            np.isfinite(frame["open"]) & np.isfinite(frame["high"])
            & np.isfinite(frame["low"]) & np.isfinite(frame["close"])
            & frame["open"].gt(0) & frame["high"].gt(0)
            & frame["low"].gt(0) & frame["close"].gt(0)
            & frame["high"].ge(np.maximum(frame["open"], frame["close"]))
            & frame["low"].le(np.minimum(frame["open"], frame["close"]))
            & frame["high"].ge(frame["low"])
        )
        volume_valid = (
            np.isfinite(frame["quote_volume"])
            & np.isfinite(frame["taker_buy_quote_volume"])
            & frame["quote_volume"].ge(0)
            & frame["taker_buy_quote_volume"].ge(0)
            & frame["taker_buy_quote_volume"].le(frame["quote_volume"])
        )
        valid = timestamp_valid & month_valid & close_time_valid
        timestamp_errors = int((~timestamp_valid).sum())
        month_errors = int((timestamp_valid & ~month_valid).sum())
        close_time_errors = int((timestamp_valid & month_valid & ~close_time_valid).sum())
        bad_price = valid & ~price_valid
        bad_volume = valid & ~volume_valid
        invalid_row = ~valid | ~price_valid | ~volume_valid
        invalid_samples.extend(
            frame.loc[bad_volume, ["Opened", "quote_volume", "taker_buy_quote_volume"]]
            .head(5).to_dict("records")
        )
        frame = frame.loc[valid, ["Opened", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]].copy()
        frame.loc[~price_valid.loc[valid].to_numpy(), ["open", "high", "low", "close"]] = np.nan
        frame.loc[~volume_valid.loc[valid].to_numpy(), ["quote_volume", "taker_buy_quote_volume"]] = np.nan
        frame.set_index("Opened", inplace=True)
        frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC")
        per_month.append(frame)
        month_audit.append({
            "month": record["month"],
            "rows": int(count),
            "timestamp_unit": "milliseconds",
            "timestamp_errors": timestamp_errors,
            "out_of_month_rows": month_errors,
            "close_time_errors": close_time_errors,
            "invalid_ohlc_rows": int(bad_price.sum()),
            "invalid_quote_volume_or_taker_relation_rows": int(bad_volume.sum()),
            "invalid_rows_excluded_or_blank_fields": int(invalid_row.sum()),
        })

    all_rows = pd.concat(per_month, axis=0).sort_index(kind="stable")
    duplicate_mask = all_rows.index.duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    duplicate_examples = [str(x) for x in all_rows.index[duplicate_mask][:10]]
    if duplicate_count:
        all_rows = all_rows.loc[~duplicate_mask]

    expected_start = pd.Timestamp(CONFIG["binance_source"]["first_month"] + "-01", tz="UTC")
    expected_last_period = pd.Period(CONFIG["binance_source"]["last_month"], freq="M")
    expected_end = (expected_last_period.end_time.floor("min")).tz_localize("UTC")
    full_index = pd.date_range(expected_start, expected_end, freq="min", tz="UTC")
    all_rows = all_rows.reindex(full_index)
    validation = {
        "source": "official Binance Public Data USD-M Futures monthly BTCUSDT 1m klines",
        "archive_count": len(records),
        "archive_months": [item["month"] for item in records],
        "timestamp_unit": "milliseconds since Unix epoch UTC; every valid open time aligned to a minute",
        "instrument_identity": {"market_type": "USD-M Futures", "symbol": "BTCUSDT", "interval": "1m"},
        "rows_after_month_and_timestamp_filter": int(sum(len(x) for x in per_month)),
        "duplicate_rows_excluded": duplicate_count,
        "duplicate_timestamp_examples": duplicate_examples,
        "expected_minute_grid_rows": int(len(full_index)),
        "observed_valid_bar_rows": int(all_rows["close"].notna().sum()),
        "missing_or_invalid_bar_minutes": int(all_rows["close"].isna().sum()),
        "invalid_taker_buy_quote_rows": int(sum(x["invalid_quote_volume_or_taker_relation_rows"] for x in month_audit)),
        "invalid_ohlc_rows": int(sum(x["invalid_ohlc_rows"] for x in month_audit)),
        "month_audit": month_audit,
        "invalid_taker_volume_examples": invalid_samples[:20],
        "invariant": "0 <= taker_buy_quote_volume <= quote_volume; failures are reported and set missing, never clipped",
    }
    return all_rows, validation


def _load_and_validate_daily_archives(
    records: list[dict], *, volume_abs_tolerance: float = 1e-10, volume_rel_tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, dict]:
    """Parse and audit daily USD-M kline ZIPs using the same 12-column schema as monthly archives."""
    columns = [
        "open_time", "open", "high", "low", "close", "base_volume", "close_time",
        "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    official_header = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ]
    per_day = []
    day_audit = []
    ordered = sorted(records, key=lambda item: item["day"])
    for record in ordered:
        archive_path = ROOT / record["zip_path"]
        if _sha256(archive_path) != record["sha256"]:
            raise RuntimeError(f"Binance daily archive SHA-256 changed after download: {record['day']}")
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise RuntimeError(f"Expected exactly one CSV in daily archive for {record['day']}")
            expected_member = f"BTCUSDT-1m-{record['day']}.csv"
            if Path(members[0]).name != expected_member:
                raise RuntimeError(f"Unexpected kline member in {record['day']} archive: {members[0]}")
            with archive.open(members[0]) as handle:
                first_line = handle.readline().decode("utf-8-sig").strip().split(",")
                header_present = [value.strip().lower() for value in first_line] == official_header
                if not header_present:
                    handle.seek(0)
                frame = pd.read_csv(handle, header=None, names=columns, usecols=list(range(11)))

        input_rows = len(frame)
        open_time = pd.to_numeric(frame["open_time"], errors="coerce")
        close_time = pd.to_numeric(frame["close_time"], errors="coerce")
        timestamps = pd.to_datetime(open_time, unit="ms", utc=True, errors="coerce")
        day_start = pd.Timestamp(record["day"], tz="UTC")
        day_end = day_start + pd.Timedelta(days=1)
        timestamp_valid = timestamps.notna() & open_time.mod(60_000).eq(0)
        in_day = timestamps.ge(day_start) & timestamps.lt(day_end)
        close_time_valid = close_time.eq(open_time + 59_999)
        frame["Opened"] = timestamps
        for name in ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"):
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        price_valid = (
            np.isfinite(frame["open"]) & np.isfinite(frame["high"])
            & np.isfinite(frame["low"]) & np.isfinite(frame["close"])
            & frame["open"].gt(0) & frame["high"].gt(0)
            & frame["low"].gt(0) & frame["close"].gt(0)
            & frame["high"].ge(np.maximum(frame["open"], frame["close"]))
            & frame["low"].le(np.minimum(frame["open"], frame["close"]))
            & frame["high"].ge(frame["low"])
        )
        quote = frame["quote_volume"]
        taker_buy = frame["taker_buy_quote_volume"]
        tolerance = np.maximum(volume_abs_tolerance, volume_rel_tolerance * np.maximum(quote.abs(), taker_buy.abs()))
        volume_valid = (
            np.isfinite(quote) & np.isfinite(taker_buy)
            & quote.ge(0) & taker_buy.ge(0)
            & taker_buy.le(quote + tolerance)
        )
        valid_timestamp = timestamp_valid & in_day & close_time_valid
        accepted = frame.loc[valid_timestamp, [
            "Opened", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume",
        ]].copy()
        accepted_price_valid = price_valid.loc[valid_timestamp].to_numpy()
        accepted_volume_valid = volume_valid.loc[valid_timestamp].to_numpy()
        accepted.loc[~accepted_price_valid, ["open", "high", "low", "close"]] = np.nan
        accepted.loc[~accepted_volume_valid, ["quote_volume", "taker_buy_quote_volume"]] = np.nan
        duplicate_mask = accepted["Opened"].duplicated(keep=False)
        duplicate_rows = int(duplicate_mask.sum())
        accepted = accepted.loc[~duplicate_mask].set_index("Opened").sort_index(kind="stable")
        day_grid = pd.date_range(day_start, day_end - pd.Timedelta(minutes=1), freq="min", tz="UTC")
        reindexed = accepted.reindex(day_grid)
        per_day.append(reindexed)
        valid_volume_index = volume_valid.loc[valid_timestamp]
        strict_relation = taker_buy.loc[valid_timestamp].gt(quote.loc[valid_timestamp])
        day_audit.append({
            "day": record["day"],
            "source_archive_sha256": record["sha256"],
            "header_row_present": header_present,
            "input_rows": int(input_rows),
            "expected_rows": 1440,
            "rows_with_valid_utc_open_and_close_times": int(valid_timestamp.sum()),
            "timestamp_errors_or_off_minute_rows": int((~timestamp_valid).sum()),
            "out_of_day_rows": int((timestamp_valid & ~in_day).sum()),
            "close_time_errors": int((timestamp_valid & in_day & ~close_time_valid).sum()),
            "duplicate_rows_excluded": duplicate_rows,
            "missing_minute_rows_after_grid_check": int(reindexed["close"].isna().sum()),
            "invalid_ohlc_rows": int((valid_timestamp & ~price_valid).sum()),
            "invalid_quote_or_taker_volume_rows_beyond_tolerance": int((valid_timestamp & ~volume_valid).sum()),
            "taker_buy_above_quote_within_tolerance_rows": int((strict_relation & valid_volume_index).sum()),
            "volume_abs_tolerance": volume_abs_tolerance,
            "volume_rel_tolerance": volume_rel_tolerance,
        })

    all_rows = pd.concat(per_day, axis=0).sort_index(kind="stable")
    duplicate_count = int(all_rows.index.duplicated(keep=False).sum())
    if duplicate_count:
        raise RuntimeError("Daily Binance archives have overlapping or duplicate timestamps")
    expected_index = pd.date_range(
        pd.Timestamp(ordered[0]["day"], tz="UTC"),
        pd.Timestamp(ordered[-1]["day"], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=1),
        freq="min", tz="UTC",
    )
    all_rows = all_rows.reindex(expected_index)
    validation = {
        "source": "official Binance Public Data USD-M Futures daily BTCUSDT 1m klines",
        "instrument_identity": {"market_type": "USD-M Futures", "symbol": "BTCUSDT", "interval": "1m"},
        "timestamp_unit": "milliseconds since Unix epoch UTC; valid open times aligned to a minute",
        "expected_days": [item["day"] for item in ordered],
        "expected_minute_grid_rows": int(len(expected_index)),
        "observed_valid_close_rows": int(all_rows["close"].notna().sum()),
        "missing_or_invalid_bar_minutes": int(all_rows["close"].isna().sum()),
        "duplicate_timestamps_across_archives": duplicate_count,
        "volume_invariant": "quote_volume >= 0; 0 <= taker_buy_quote_volume <= quote_volume + max(abs_tol, rel_tol * max(abs(quote_volume), abs(taker_buy_quote_volume)))",
        "values_clipped_or_imputed": False,
        "day_audit": day_audit,
    }
    return all_rows, validation


def _load_dataset_inputs() -> tuple[pd.DataFrame, dict]:
    source_run = _read_json(SOURCE_RUN_PATH)
    cols = SOURCE_FEATURES + [
        "Opened", "target_preopen_5m_up_binance_index_proxy", "target_label_available_at",
        "nominal_decision_at", "is_polymarket_preopen_decision",
    ]
    filters = [("Opened", "<=", END_UTC.strftime("%Y-%m-%d %H:%M:%S"))]
    frame = pd.read_parquet(SOURCE_DATA_PATH, columns=cols, filters=filters)
    frame["Opened"] = pd.to_datetime(frame["Opened"], utc=True, errors="raise")
    frame["nominal_decision_at"] = pd.to_datetime(frame["nominal_decision_at"], utc=True, errors="raise")
    frame["target_label_available_at"] = pd.to_datetime(
        frame["target_label_available_at"], utc=True, errors="raise",
    )
    if frame["Opened"].duplicated().any() or not frame["Opened"].is_monotonic_increasing:
        raise RuntimeError("The model-ready cache is not a unique chronological minute series")
    expected = source_run["stages"]["final_feature_dataset"]["result"]["rows"]
    if len(frame) > int(expected):
        raise RuntimeError("Development-only parquet filter returned more rows than the frozen source dataset")
    effective = _read_json(SOURCE_CONFIG_PATH)
    split = effective["contract"]["chronological_split_frozen_before_tuning"]
    fit_start = pd.Timestamp(split["feature_and_model_fit_start_utc"])
    fit_end = pd.Timestamp(split["fit_and_tuning_end_exclusive_utc"])
    eligible = (
        frame["Opened"].ge(fit_start)
        & frame["target_preopen_5m_up_binance_index_proxy"].isin((0, 1))
        & frame["target_label_available_at"].lt(fit_end)
    )
    frame = frame.loc[eligible].reset_index(drop=True)
    expected_fit_rows = int(SOURCE_TRAINING["training_rows"])
    if len(frame) != expected_fit_rows:
        raise RuntimeError(
            f"Recovered {len(frame):,} source fit rows, expected {expected_fit_rows:,}"
        )
    expected_decisions = int(SOURCE_TUNING["decision_rows"])
    actual_decisions = int(frame["is_polymarket_preopen_decision"].sum())
    if actual_decisions != expected_decisions:
        raise RuntimeError(
            f"Recovered {actual_decisions:,} source decision rows, expected {expected_decisions:,}"
        )
    return frame, {
        "source_rows_after_fit_label_filter": len(frame),
        "source_decision_rows": actual_decisions,
        "fit_start_utc": fit_start.isoformat(),
        "fit_end_exclusive_utc": fit_end.isoformat(),
        "latest_label_available_at_utc": frame["target_label_available_at"].max().isoformat(),
        "target": "target_preopen_5m_up_binance_index_proxy",
        "target_formula": effective["contract"]["training_target"]["formula"],
    }


def _prepare_common_rows(manifest: dict, flow_path: Path) -> tuple[Path, dict]:
    output_path = OUTPUT_DIR / "common_complete_training_rows.parquet"
    prior = manifest.get("stages", {}).get("common_rows", {})
    if prior.get("status") == "completed" and _verify_records(prior.get("outputs", [])):
        return output_path, prior["summary"]

    _begin_stage(manifest, "common_rows", "Run B0 fixed-parameter 10-fold CV.")
    started = time.perf_counter()
    with PeakMemorySampler() as memory:
        source, source_summary = _load_dataset_inputs()
        flow = pd.read_parquet(flow_path)
        if flow["Opened"].duplicated().any() or not flow["Opened"].is_monotonic_increasing:
            raise RuntimeError("The saved Binance flow feature table is not unique and chronological")
        joined = source.merge(flow, on="Opened", how="left", validate="one_to_one", sort=False)
        if len(joined) != len(source):
            raise RuntimeError("The Binance flow join changed the source model row count")
        base_values = joined.loc[:, SOURCE_FEATURES].to_numpy(dtype=np.float32, copy=False)
        flow_values = joined.loc[:, FLOW_FEATURES].to_numpy(dtype=np.float64, copy=False)
        finite_base = np.isfinite(base_values).all(axis=1)
        finite_flow = np.isfinite(flow_values).all(axis=1)
        common = finite_base & finite_flow
        common_rows = joined.loc[common].copy()
        if common_rows.empty:
            raise RuntimeError("No rows remain after the shared B0/B1 complete-feature mask")
        common_rows["_source_row_index"] = np.flatnonzero(common).astype(np.int32)

        decision_mask = source["is_polymarket_preopen_decision"].to_numpy(dtype=bool)
        model_tuning.CV_FOLDS = CV_FOLD_COUNT
        model_tuning.WF_TEST_TO_TRAIN_RATIO = float(
            CONFIG["model_comparison"]["test_to_train_ratio_from_source_model_runner"]
        )
        source_folds = experiment._purged_walk_forward_folds(
            model_tuning,
            source["Opened"],
            source["nominal_decision_at"],
            CV_FOLD_COUNT,
            model_tuning.WF_TEST_TO_TRAIN_RATIO,
            validation_mask=decision_mask,
        )
        original_fingerprints = experiment._fold_index_fingerprints(source_folds)
        expected_fingerprints = SOURCE_TUNING["fold_indices"]
        if original_fingerprints != expected_fingerprints:
            raise RuntimeError(
                "Recovered source chronological folds do not match the completed model-tuning fingerprints"
            )
        original_weights = experiment._preopen_sample_weights(decision_mask, DECISION_WEIGHT)
        original_weight_hash = _array_sha256(original_weights, dtype=np.float32)
        if original_weight_hash != SOURCE_TUNING["training_weight_sha256"]:
            raise RuntimeError("Recovered source decision/auxiliary sample weights do not match the completed tuning artifact")

        source_to_common = np.full(len(source), -1, dtype=np.int32)
        common_source_indices = np.flatnonzero(common)
        source_to_common[common_source_indices] = np.arange(len(common_source_indices), dtype=np.int32)
        common_folds = []
        folds_summary = []
        for fold in source_folds:
            train_common = source_to_common[np.asarray(fold["train_idx"], dtype=np.int64)]
            valid_common = source_to_common[np.asarray(fold["valid_idx"], dtype=np.int64)]
            train_common = train_common[train_common >= 0]
            valid_common = valid_common[valid_common >= 0]
            if not len(train_common) or not len(valid_common):
                raise RuntimeError(f"Shared completeness mask removed all rows from fold {fold['fold_id']}")
            common_folds.append({
                "fold_id": int(fold["fold_id"]),
                "train_idx": train_common.astype(np.int32, copy=False),
                "valid_idx": valid_common.astype(np.int32, copy=False),
                "source_first_validation_decision_at": source["nominal_decision_at"].iloc[int(fold["valid_idx"][0])].isoformat(),
            })
            folds_summary.append({
                "fold_id": int(fold["fold_id"]),
                "source_train_rows": int(len(fold["train_idx"])),
                "source_valid_rows": int(len(fold["valid_idx"])),
                "common_train_rows": int(len(train_common)),
                "common_valid_rows": int(len(valid_common)),
                "common_validation_start_utc": common_rows["nominal_decision_at"].iloc[int(valid_common[0])].isoformat(),
                "common_validation_last_utc": common_rows["nominal_decision_at"].iloc[int(valid_common[-1])].isoformat(),
            })
        fold_path = OUTPUT_DIR / "common_folds.npz"
        fold_arrays = {}
        for item in common_folds:
            fold_arrays[f"train_{item['fold_id']}"] = item["train_idx"]
            fold_arrays[f"valid_{item['fold_id']}"] = item["valid_idx"]
        np.savez_compressed(fold_path, **fold_arrays)

        columns = [
            "_source_row_index", "Opened", "nominal_decision_at", "target_label_available_at",
            "target_preopen_5m_up_binance_index_proxy", "is_polymarket_preopen_decision",
            *SOURCE_FEATURES, *FLOW_FEATURES,
        ]
        temporary = output_path.with_suffix(".parquet.tmp")
        common_rows.loc[:, columns].to_parquet(
            temporary, index=False, compression="zstd", row_group_size=65_536,
        )
        temporary.replace(output_path)
        summary = {
            **source_summary,
            "rows_after_shared_complete_feature_mask": int(len(common_rows)),
            "rows_excluded_by_shared_complete_feature_mask": int(len(source) - len(common_rows)),
            "decision_rows_after_shared_complete_feature_mask": int(common_rows["is_polymarket_preopen_decision"].sum()),
            "common_mask": "finite across all 64 source features and all 18 taker-flow features; applied once for both variants",
            "source_fold_fingerprints": original_fingerprints,
            "source_fold_fingerprints_expected": expected_fingerprints,
            "source_training_weight_sha256": original_weight_hash,
            "source_training_weight_sha256_expected": SOURCE_TUNING["training_weight_sha256"],
            "common_fold_fingerprints": experiment._fold_index_fingerprints(common_folds),
            "folds": folds_summary,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_working_set_gib": round(memory.peak_bytes / 2**30, 3),
            "input_flow_features_sha256": _sha256(flow_path),
        }
        _write_json(OUTPUT_DIR / "common_rows_summary.json", summary)
        del source, flow, joined, base_values, flow_values, common_rows
        gc.collect()
    outputs = [
        _record(output_path), _record(fold_path), _record(OUTPUT_DIR / "common_rows_summary.json"),
    ]
    _finish_stage(
        manifest, "common_rows", summary=summary, outputs=outputs,
        elapsed_seconds=time.perf_counter() - started,
        peak_process_working_set_gib=round(memory.peak_bytes / 2**30, 3),
    )
    return output_path, summary


def _load_common_arrays(variant: str, common_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str], list[tuple[np.ndarray, np.ndarray]]]:
    common_features = CONTROL_FLOW_FEATURES if variant == "B0" else FLOW_FEATURES
    names = SOURCE_FEATURES + common_features
    if len(names) != len(set(names)):
        raise RuntimeError(f"{variant} feature order contains duplicate names")
    metadata_columns = [
        "_source_row_index", "Opened", "nominal_decision_at", "target_label_available_at",
        "target_preopen_5m_up_binance_index_proxy", "is_polymarket_preopen_decision",
    ]
    frame = pd.read_parquet(common_path, columns=[*metadata_columns, *names])
    x = frame.loc[:, names].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(x).all():
        raise RuntimeError(f"{variant} comparison matrix is not complete and finite")
    y = frame["target_preopen_5m_up_binance_index_proxy"].to_numpy(dtype=np.int8, copy=True)
    if not np.isin(y, (0, 1)).all():
        raise RuntimeError("The common target contains values outside binary classes")
    decision = frame["is_polymarket_preopen_decision"].to_numpy(dtype=bool, copy=True)
    weights = experiment._preopen_sample_weights(decision, DECISION_WEIGHT)
    with np.load(OUTPUT_DIR / "common_folds.npz") as saved:
        folds = [
            (
                saved[f"train_{fold_id}"].astype(np.int32, copy=False),
                saved[f"valid_{fold_id}"].astype(np.int32, copy=False),
            )
            for fold_id in range(CV_FOLD_COUNT)
        ]
    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        if not len(train_idx) or not len(valid_idx):
            raise RuntimeError(f"{variant} fold {fold_id} has an empty training or validation set")
        if not decision[valid_idx].all():
            raise RuntimeError(f"{variant} fold {fold_id} contains a non-decision validation row")
        if frame["target_label_available_at"].iloc[train_idx].max() > frame["nominal_decision_at"].iloc[valid_idx].min():
            raise RuntimeError(f"{variant} fold {fold_id} includes a label unavailable at validation start")
    return frame, x, y, weights, names, folds


def _score_binary(y: np.ndarray, probability: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return {
        "rows": int(len(y)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(np.mean(np.square(probability - y))),
        "auc": float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else None,
    }


def _recover_variant_from_saved_cv(
    manifest: dict,
    variant: str,
    common_path: Path,
    prior: dict,
) -> dict | None:
    failure = manifest.get("failure", {})
    if not (
        variant == "B0"
        and prior.get("status") == "running"
        and failure.get("stage") == "cv_B0"
        and failure.get("error_type") == "NameError"
        and "decision" in failure.get("error", "")
    ):
        return None

    prediction_path = OUTPUT_DIR / f"{variant}_decision_oof.parquet"
    summary_path = OUTPUT_DIR / f"{variant}_cv_summary.json"
    history_path = OUTPUT_DIR / f"{variant}_cv_history.json"
    model_paths = [OUTPUT_DIR / "models" / variant / f"fold_{fold_id:02d}.txt" for fold_id in range(CV_FOLD_COUNT)]
    if summary_path.exists() or history_path.exists() or not prediction_path.is_file() or not all(path.is_file() for path in model_paths):
        return None

    frame, x, y, weights, names, folds = _load_common_arrays(variant, common_path)
    saved_oof = pd.read_parquet(prediction_path)
    validation_mask = np.zeros(len(frame), dtype=bool)
    fold_by_row = np.full(len(frame), -1, dtype=np.int16)
    for fold_id, (_, valid_idx) in enumerate(folds):
        validation_mask[valid_idx] = True
        fold_by_row[valid_idx] = fold_id
    predicted_rows = np.flatnonzero(validation_mask)
    expected_source_rows = frame["_source_row_index"].iloc[predicted_rows].to_numpy(dtype=np.int32)
    if len(saved_oof) != len(predicted_rows) or not np.array_equal(
        saved_oof["_source_row_index"].to_numpy(dtype=np.int32), expected_source_rows,
    ):
        raise RuntimeError("Saved B0 OOF rows do not align with the frozen common validation rows")
    expected_folds = fold_by_row[predicted_rows].astype(str)
    if not np.array_equal(saved_oof["fold_id"].astype(str).to_numpy(), expected_folds):
        raise RuntimeError("Saved B0 OOF fold ids differ from the frozen common validation folds")
    if not np.array_equal(
        saved_oof[experiment.TARGET_COL].to_numpy(dtype=np.int8), y[predicted_rows],
    ):
        raise RuntimeError("Saved B0 OOF target differs from the frozen common development target")

    best_iterations = []
    per_fold = []
    importance_rows = []
    model_outputs = []
    saved_probability = saved_oof["p_model_raw"].to_numpy(dtype=np.float64)
    oof_cursor = 0
    for fold_id, ((_, valid_idx), model_path) in enumerate(zip(folds, model_paths)):
        booster = lgb.Booster(model_file=str(model_path))
        if list(booster.feature_name()) != names:
            raise RuntimeError(f"Saved B0 fold {fold_id} model feature order changed")
        iteration = int(booster.current_iteration())
        best_iterations.append(iteration)
        fold_probability = booster.predict(x[valid_idx], num_iteration=iteration)
        fold_size = len(valid_idx)
        oof_probability = saved_probability[oof_cursor:oof_cursor + fold_size]
        if not np.allclose(fold_probability, oof_probability, rtol=1e-6, atol=1e-7):
            raise RuntimeError(f"Saved B0 fold {fold_id} model does not reproduce its OOF predictions")
        oof_cursor += fold_size
        per_fold.append({
            "fold_id": fold_id,
            "validation_start_utc": frame["nominal_decision_at"].iloc[valid_idx].min().isoformat(),
            "validation_end_utc": frame["nominal_decision_at"].iloc[valid_idx].max().isoformat(),
            **_score_binary(y[valid_idx], oof_probability),
        })
        split_importance = booster.feature_importance(importance_type="split", iteration=iteration)
        gain_importance = booster.feature_importance(importance_type="gain", iteration=iteration)
        importance_rows.append({
            name: {"split_count": int(split_importance[index]), "gain": float(gain_importance[index])}
            for index, name in enumerate(names)
        })
        model_outputs.append(_record(model_path))
    if len(set(best_iterations)) != 1:
        raise RuntimeError("Saved B0 fold boosters do not share the selected CV iteration")
    best_iteration = best_iterations[0]
    if oof_cursor != len(saved_oof):
        raise RuntimeError("Saved B0 OOF rows have an unexpected fold partition")

    losses = np.asarray([item["log_loss"] for item in per_fold], dtype=np.float64)
    mean_loss = float(np.mean(losses))
    loss_std = float(np.std(losses))
    objective = mean_loss + 0.5 * loss_std
    aggregate_split = np.sum(
        [np.asarray([row[name]["split_count"] for name in names]) for row in importance_rows], axis=0,
    )
    aggregate_gain = np.sum(
        [np.asarray([row[name]["gain"] for name in names]) for row in importance_rows], axis=0,
    )
    importance = [
        {
            "feature": name,
            "split_count_across_folds": int(aggregate_split[index]),
            "folds_with_nonzero_splits": int(sum(row[name]["split_count"] > 0 for row in importance_rows)),
            "gain_across_folds": float(aggregate_gain[index]),
        }
        for index, name in enumerate(names)
    ]
    fit_elapsed = None
    if prior.get("started_utc"):
        fit_elapsed = max(
            0.0,
            prediction_path.stat().st_mtime - pd.Timestamp(prior["started_utc"]).timestamp(),
        )
    summary = {
        "variant": variant,
        "rows": int(len(frame)),
        "decision_rows": int(frame["is_polymarket_preopen_decision"].sum()),
        "oof_rows": int(len(saved_oof)),
        "feature_count": len(names),
        "feature_order": names,
        "best_iteration": best_iteration,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "maximum_estimators": MAX_ESTIMATORS,
        "params": {
            "objective": "binary", "metric": "None", "boosting_type": "gbdt",
            "device_type": "gpu", "verbosity": -1, "num_threads": THREADS,
            "max_bin": 63, "feature_pre_filter": False, "gpu_use_dp": False,
            "seed": MODEL_SEED, "feature_fraction_seed": MODEL_SEED,
            "bagging_seed": MODEL_SEED, "data_random_seed": MODEL_SEED,
            **MODEL_PARAMETERS,
        },
        "training_weight_sha256": _array_sha256(weights, dtype=np.float32),
        "decision_weight": DECISION_WEIGHT,
        "auxiliary_row_weight": AUXILIARY_WEIGHT,
        "validation_weights": "removed; unweighted decision rows",
        "best_cv_objective": objective,
        "cv_binary_logloss_mean_at_best": mean_loss,
        "cv_binary_logloss_population_std_at_best": loss_std,
        "fold_metrics": per_fold,
        "aggregate_feature_importance": importance,
        "new_feature_use": None,
        "elapsed_seconds": fit_elapsed,
        "peak_process_working_set_gib": None,
        "observed_working_set_lower_bound_gib": 12.98,
        "recovered_from_saved_cv_boosters_without_refitting": True,
        "cv_iteration_history": "only selected iteration recovered from saved boosters and OOF; full history was not persisted before the post-fit NameError",
        "recovery_checks": {
            "fold_model_count": len(model_paths),
            "fold_iterations": best_iterations,
            "oof_predictions_reproduced_from_saved_models": True,
            "source_rows_and_folds_verified": True,
        },
    }
    history = {
        "metric_names": ["binary_logloss", "balanced_accuracy", "brier_score"],
        "best_iteration": best_iteration,
        "best_objective": objective,
        "recovered_from_saved_cv_boosters": True,
        "full_iteration_history_saved": False,
        "iterations": [{
            "iteration": best_iteration,
            "binary_logloss_mean": mean_loss,
            "binary_logloss_fold_std": loss_std,
            "objective": objective,
        }],
    }
    _write_json(summary_path, summary)
    _write_json(history_path, history)
    outputs = [_record(prediction_path), _record(summary_path), _record(history_path), *model_outputs]
    _finish_stage(
        manifest,
        f"cv_{variant}",
        summary=summary,
        outputs=outputs,
        elapsed_seconds=fit_elapsed,
        peak_process_working_set_gib=None,
        recovered_from_saved_cv_boosters_without_refitting=True,
    )
    manifest.setdefault("recovered_failures", []).append(manifest.pop("failure"))
    _write_json(MANIFEST_PATH, manifest)
    return summary


def _fit_variant(manifest: dict, variant: str, common_path: Path) -> dict:
    stage_name = f"cv_{variant}"
    prior = manifest.get("stages", {}).get(stage_name, {})
    if prior.get("status") == "completed" and _verify_records(prior.get("outputs", [])):
        return prior["summary"]
    recovered = _recover_variant_from_saved_cv(manifest, variant, common_path, prior)
    if recovered is not None:
        return recovered

    next_step = "Run B1 fixed-parameter 10-fold CV." if variant == "B0" else "Score proxy and official OOF, calibrate causally, and reuse existing snapshots."
    _begin_stage(manifest, stage_name, next_step)
    started = time.perf_counter()
    model_dir = OUTPUT_DIR / "models" / variant
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = OUTPUT_DIR / f"{variant}_decision_oof.parquet"
    summary_path = OUTPUT_DIR / f"{variant}_cv_summary.json"
    history_path = OUTPUT_DIR / f"{variant}_cv_history.json"
    with PeakMemorySampler() as memory:
        frame, x, y, weights, names, folds = _load_common_arrays(variant, common_path)
        decision = frame["is_polymarket_preopen_decision"].to_numpy(dtype=bool, copy=False)
        train_set = lgb.Dataset(
            x,
            label=y,
            weight=weights,
            feature_name=names,
            free_raw_data=False,
        )

        def unweighted_validation(train_data, valid_data, params):
            valid_data.set_weight(None)
            return train_data, valid_data, params

        params = {
            "objective": "binary",
            "metric": "None",
            "boosting_type": "gbdt",
            "device_type": "gpu",
            "verbosity": -1,
            "num_threads": THREADS,
            "max_bin": 63,
            "feature_pre_filter": False,
            "gpu_use_dp": False,
            "seed": MODEL_SEED,
            "feature_fraction_seed": MODEL_SEED,
            "bagging_seed": MODEL_SEED,
            "data_random_seed": MODEL_SEED,
            **MODEL_PARAMETERS,
        }
        cv_results = lgb.cv(
            params=params,
            train_set=train_set,
            num_boost_round=MAX_ESTIMATORS,
            folds=folds,
            fpreproc=unweighted_validation,
            stratified=False,
            shuffle=False,
            feval=[
                make_lightgbm_binary_logloss_eval("binary_logloss"),
                make_lightgbm_binary_balanced_accuracy_eval("balanced_accuracy"),
                make_lightgbm_binary_brier_eval("brier_score"),
            ],
            callbacks=[
                ObjectiveAlignedCVCallback(
                    "binary_logloss",
                    higher_is_better=False,
                    std_penalty=0.5,
                    stopping_rounds=EARLY_STOPPING_ROUNDS,
                ),
                lgb.log_evaluation(period=25),
            ],
            return_cvbooster=True,
            seed=MODEL_SEED,
        )
        cvbooster = cv_results.pop("cvbooster")
        metric_series = cv_objective_series(
            cv_results,
            "binary_logloss",
            higher_is_better=False,
            std_penalty=0.5,
        )
        objective_values = metric_series["objective"]
        best_index = int(np.argmin(objective_values))
        best_iteration = best_index + 1
        if len(cvbooster.boosters) != CV_FOLD_COUNT:
            raise RuntimeError(f"{variant} CV retained {len(cvbooster.boosters)} boosters, expected {CV_FOLD_COUNT}")
        reported_best_iteration = int(getattr(cvbooster, "best_iteration", 0) or 0)
        if reported_best_iteration > 0 and reported_best_iteration != best_iteration:
            raise RuntimeError(f"{variant} callback and objective-series best iterations differ")

        probability = np.full(len(frame), np.nan, dtype=np.float32)
        per_fold = []
        importance_rows = []
        permutation_rows = []
        added_indices = [names.index(name) for name in EXTENSION_FEATURES] if variant == "B1" else []
        rng = np.random.default_rng(int(CONFIG["bootstrap"]["seed"]))
        model_outputs = []
        for fold_id, ((_, valid_idx), booster) in enumerate(zip(folds, cvbooster.boosters)):
            fold_probability = np.asarray(
                booster.predict(x[valid_idx], num_iteration=best_iteration), dtype=np.float32,
            )
            if not np.isfinite(fold_probability).all() or np.any((fold_probability < 0) | (fold_probability > 1)):
                raise RuntimeError(f"{variant} fold {fold_id} produced invalid probabilities")
            probability[valid_idx] = fold_probability
            per_fold.append({
                "fold_id": fold_id,
                "validation_start_utc": frame["nominal_decision_at"].iloc[valid_idx].min().isoformat(),
                "validation_end_utc": frame["nominal_decision_at"].iloc[valid_idx].max().isoformat(),
                **_score_binary(y[valid_idx], fold_probability),
            })
            split_importance = booster.feature_importance(importance_type="split", iteration=best_iteration)
            gain_importance = booster.feature_importance(importance_type="gain", iteration=best_iteration)
            importance_rows.append({
                name: {"split_count": int(split_importance[index]), "gain": float(gain_importance[index])}
                for index, name in enumerate(names)
            })
            model_path = model_dir / f"fold_{fold_id:02d}.txt"
            temporary = model_path.with_suffix(".txt.tmp")
            booster.save_model(str(temporary), num_iteration=best_iteration)
            temporary.replace(model_path)
            model_outputs.append(_record(model_path))

            if variant == "B1":
                permuted = x[valid_idx].copy()
                permutation = rng.permutation(len(valid_idx))
                permuted[:, added_indices] = permuted[permutation][:, added_indices]
                permuted_probability = booster.predict(permuted, num_iteration=best_iteration)
                original_loss = _score_binary(y[valid_idx], fold_probability)["log_loss"]
                permuted_loss = _score_binary(y[valid_idx], permuted_probability)["log_loss"]
                permutation_rows.append({
                    "fold_id": fold_id,
                    "rows": int(len(valid_idx)),
                    "original_log_loss": original_loss,
                    "joint_new_feature_permuted_log_loss": permuted_loss,
                    "log_loss_increase": float(permuted_loss - original_loss),
                })
                del permuted, permuted_probability

        predicted_rows = np.flatnonzero(np.isfinite(probability))
        if not len(predicted_rows):
            raise RuntimeError(f"{variant} CV produced no out-of-fold predictions")
        prediction_frame = frame.iloc[predicted_rows][[
            "_source_row_index", "Opened", "nominal_decision_at", "target_label_available_at",
            "target_preopen_5m_up_binance_index_proxy",
        ]].copy()
        prediction_frame.rename(columns={
            "target_preopen_5m_up_binance_index_proxy": experiment.TARGET_COL,
        }, inplace=True)
        prediction_frame["market_start_utc"] = prediction_frame["Opened"] + pd.Timedelta(minutes=2)
        prediction_frame["fold_id"] = pd.Series(pd.NA, index=prediction_frame.index, dtype="string")
        for fold_id, (_, valid_idx) in enumerate(folds):
            prediction_frame.loc[prediction_frame.index.isin(valid_idx), "fold_id"] = str(fold_id)
        prediction_frame["p_model_raw"] = probability[predicted_rows]
        prediction_frame["prediction_available_at_utc"] = pd.NaT
        temporary_predictions = prediction_path.with_suffix(".parquet.tmp")
        prediction_frame.to_parquet(temporary_predictions, index=False, compression="zstd")
        temporary_predictions.replace(prediction_path)

        aggregate_split = np.sum(
            [np.asarray([row[name]["split_count"] for name in names]) for row in importance_rows],
            axis=0,
        )
        aggregate_gain = np.sum(
            [np.asarray([row[name]["gain"] for name in names]) for row in importance_rows],
            axis=0,
        )
        importance = [
            {
                "feature": name,
                "split_count_across_folds": int(aggregate_split[index]),
                "folds_with_nonzero_splits": int(sum(row[name]["split_count"] > 0 for row in importance_rows)),
                "gain_across_folds": float(aggregate_gain[index]),
            }
            for index, name in enumerate(names)
        ]
        feature_use = None
        if variant == "B1":
            new_importance = [row for row in importance if row["feature"] in EXTENSION_FEATURES]
            feature_use = {
                "new_feature_count": len(EXTENSION_FEATURES),
                "new_features_used_in_any_split": int(sum(row["split_count_across_folds"] > 0 for row in new_importance)),
                "new_features_used_in_all_folds": int(sum(row["folds_with_nonzero_splits"] == CV_FOLD_COUNT for row in new_importance)),
                "new_features": new_importance,
                "joint_block_permutation_by_fold": permutation_rows,
                "joint_block_permutation_mean_log_loss_increase": float(np.mean([row["log_loss_increase"] for row in permutation_rows])),
            }

        history = {
            "metric_names": ["binary_logloss", "balanced_accuracy", "brier_score"],
            "best_iteration": best_iteration,
            "best_objective": float(objective_values[best_index]),
            "iterations": [
                {
                    "iteration": index + 1,
                    "binary_logloss_mean": float(metric_series["mean"][index]),
                    "binary_logloss_fold_std": float(metric_series["std"][index]),
                    "objective": float(objective_values[index]),
                }
                for index in range(len(objective_values))
            ],
        }
        summary = {
            "variant": variant,
            "rows": int(len(frame)),
            "decision_rows": int(decision.sum()),
            "oof_rows": int(len(prediction_frame)),
            "feature_count": len(names),
            "feature_order": names,
            "best_iteration": best_iteration,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "maximum_estimators": MAX_ESTIMATORS,
            "params": params,
            "training_weight_sha256": _array_sha256(weights, dtype=np.float32),
            "decision_weight": DECISION_WEIGHT,
            "auxiliary_row_weight": AUXILIARY_WEIGHT,
            "validation_weights": "removed; unweighted decision rows",
            "best_cv_objective": float(objective_values[best_index]),
            "cv_binary_logloss_mean_at_best": float(metric_series["mean"][best_index]),
            "cv_binary_logloss_population_std_at_best": float(metric_series["std"][best_index]),
            "fold_metrics": per_fold,
            "aggregate_feature_importance": importance,
            "new_feature_use": feature_use,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_working_set_gib": round(memory.peak_bytes / 2**30, 3),
        }
        _write_json(summary_path, summary)
        _write_json(history_path, history)
        del train_set, cvbooster, cv_results, frame, x, y, weights, probability, prediction_frame
        gc.collect()

    outputs = [
        _record(prediction_path), _record(summary_path), _record(history_path), *model_outputs,
    ]
    _finish_stage(
        manifest,
        stage_name,
        summary=summary,
        outputs=outputs,
        elapsed_seconds=time.perf_counter() - started,
        peak_process_working_set_gib=round(memory.peak_bytes / 2**30, 3),
        model_paths=[item["path"] for item in model_outputs],
        prediction_path=_relative(prediction_path),
    )
    return summary


def _causal_daily_platt(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    scored = frame.copy()
    scored["p_model_platt"] = np.nan
    scored["official_label_release_at_utc"] = pd.NaT
    starts = pd.to_datetime(scored["market_start_utc"], utc=True)
    resolved = pd.to_datetime(scored["resolved_at_utc"], utc=True, errors="coerce")
    release = resolved.where(
        resolved.ge(starts + pd.Timedelta(minutes=5)),
        starts + pd.Timedelta(minutes=5),
    ) + pd.Timedelta(seconds=60)
    scored["official_label_release_at_utc"] = release
    valid_label = (
        scored["target_polymarket_up"].isin((0, 1))
        & scored["p_model_raw"].notna()
        & resolved.notna()
        & release.notna()
    )
    scored["_utc_day"] = starts.dt.floor("D")
    calibrations = []
    minimum_rows = int(CONFIG["official_outcomes"]["minimum_calibration_rows_from_source_calibrator"])
    for day, day_rows in scored.groupby("_utc_day", sort=True):
        first_decision = pd.to_datetime(day_rows["nominal_decision_at"], utc=True).min()
        past_mask = valid_label & release.lt(first_decision)
        past = scored.loc[past_mask, ["p_model_raw", "target_polymarket_up"]].copy()
        past.rename(columns={"p_model_raw": "p_model_up"}, inplace=True)
        past["resolved_at_utc"] = release.loc[past.index]
        if len(past) >= minimum_rows and past["target_polymarket_up"].nunique() == 2:
            calibrator = fit_calibrator(past, "platt", first_decision)
            day_index = day_rows.index
            scored.loc[day_index, "p_model_platt"] = calibrate(
                calibrator, scored.loc[day_index, "p_model_raw"].to_numpy(dtype=np.float64),
            )
            calibrations.append({
                "utc_day": pd.Timestamp(day).isoformat(),
                "first_decision_at_utc": first_decision.isoformat(),
                "training_rows": int(len(past)),
                "latest_training_label_release_at_utc": pd.Timestamp(past["resolved_at_utc"].max()).isoformat(),
                "coefficient": float(calibrator.coef_[0, 0]),
                "intercept": float(calibrator.intercept_[0]),
                "status": "fit",
            })
        else:
            calibrations.append({
                "utc_day": pd.Timestamp(day).isoformat(),
                "first_decision_at_utc": first_decision.isoformat(),
                "training_rows": int(len(past)),
                "latest_training_label_release_at_utc": (
                    pd.Timestamp(past["resolved_at_utc"].max()).isoformat() if len(past) else None
                ),
                "status": "warmup_insufficient_rows_or_classes",
            })
    scored.drop(columns=["_utc_day"], inplace=True)
    return scored, calibrations


def _paired_block_bootstrap(
    y: np.ndarray,
    probability_b0: np.ndarray,
    probability_b1: np.ndarray,
    days: pd.Series,
    *,
    replications: int,
    seed: int,
    block_days: int = 3,
) -> dict:
    y = np.asarray(y, dtype=np.int8)
    p0 = np.asarray(probability_b0, dtype=np.float64)
    p1 = np.asarray(probability_b1, dtype=np.float64)
    day_values = pd.to_datetime(days, utc=True).dt.floor("D").to_numpy()
    unique_days = np.unique(day_values)
    if len(unique_days) < block_days:
        raise RuntimeError("Paired block bootstrap requires at least one full UTC block")
    row_indices_by_day = [np.flatnonzero(day_values == day) for day in unique_days]
    blocks_per_replication = int(math.ceil(len(unique_days) / block_days))
    rng = np.random.default_rng(seed)
    deltas = {name: np.empty(replications, dtype=np.float64) for name in ("log_loss", "brier", "auc")}
    for replication in range(replications):
        starts = rng.integers(0, len(unique_days), size=blocks_per_replication)
        sample_parts = [
            row_indices_by_day[(int(start) + offset) % len(unique_days)]
            for start in starts
            for offset in range(block_days)
        ]
        rows = np.concatenate(sample_parts)
        score0 = _score_binary(y[rows], p0[rows])
        score1 = _score_binary(y[rows], p1[rows])
        for name in deltas:
            if score0[name] is not None and score1[name] is not None:
                deltas[name][replication] = score1[name] - score0[name]
            else:
                deltas[name][replication] = np.nan
    interval = {}
    for name, values in deltas.items():
        finite = values[np.isfinite(values)]
        interval[name] = {
            "b1_minus_b0_median": float(np.median(finite)),
            "b1_minus_b0_95pct": [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))],
            "replications_with_both_classes": int(len(finite)),
        }
    return {
        "method": "paired circular moving-block bootstrap over 3-day UTC blocks",
        "replications": int(replications),
        "seed": int(seed),
        "observed_utc_days": int(len(unique_days)),
        "observed_rows": int(len(y)),
        "block_days": int(block_days),
        "intervals": interval,
    }


def _economic_summary_from_artifacts(result: dict, output_root: Path) -> dict:
    selected = result["policies"]["fixed_5_usd_ev_1pct"]["summary"]
    trades_path = output_root / "fixed_5_usd_ev_1pct" / "trades.parquet"
    decisions_path = output_root / "fixed_5_usd_ev_1pct" / "decisions.parquet"
    trades = pd.read_parquet(trades_path) if trades_path.is_file() else pd.DataFrame()
    decisions = pd.read_parquet(decisions_path) if decisions_path.is_file() else pd.DataFrame()
    forecast_ev = (
        float(pd.to_numeric(trades["chosen_side_ev_usd"], errors="coerce").sum())
        if not trades.empty and "chosen_side_ev_usd" in trades else 0.0
    )
    realized_pnl = float(pd.to_numeric(trades["net_pnl_usd"], errors="coerce").sum()) if not trades.empty else 0.0
    fills = int(len(trades))
    settlements = int(trades["settled_at_utc"].notna().sum()) if not trades.empty and "settled_at_utc" in trades else 0
    return {
        "decision_count": int(len(decisions)),
        "fill_count": fills,
        "settlement_count": settlements,
        "trade_count": int(selected.get("trade_count", len(trades))),
        "gross_turnover_usd": float(selected.get("gross_turnover_usd", 0.0)),
        "estimated_fees_usd": float(selected.get("fees_paid_estimated_usd", 0.0)),
        "net_pnl_usd": float(selected.get("net_pnl_usd", 0.0)),
        "pnl_per_dollar_turnover": (
            float(selected["net_pnl_usd"] / selected["gross_turnover_usd"])
            if float(selected.get("gross_turnover_usd", 0.0)) else None
        ),
        "forecast_net_ev_at_t60_for_filled_trades_usd": forecast_ev,
        "realized_net_pnl_for_filled_trades_usd": realized_pnl,
        "forecast_ev_minus_realized_pnl_usd": forecast_ev - realized_pnl,
        "accounting_audit": selected.get("accounting_audit", {}).get("status"),
        "rejections_by_reason": selected.get("rejections_by_reason", {}),
    }


def _score_and_evaluate(manifest: dict, common_path: Path) -> dict:
    stage_name = "quality_and_economics"
    prior = manifest.get("stages", {}).get(stage_name, {})
    if prior.get("status") == "completed" and _verify_records(prior.get("outputs", [])):
        return prior["summary"]
    _begin_stage(manifest, stage_name, "Write the report and mark the experiment complete.")
    started = time.perf_counter()
    _, _, _, _, _, correction_manifest, _, _ = model_update._source_inputs()
    b0_path = OUTPUT_DIR / "B0_decision_oof.parquet"
    b1_path = OUTPUT_DIR / "B1_decision_oof.parquet"
    b0 = pd.read_parquet(b0_path)
    b1 = pd.read_parquet(b1_path)
    key = ["_source_row_index", "Opened", "market_start_utc", "nominal_decision_at", experiment.TARGET_COL, "fold_id"]
    merged = b0.merge(
        b1[key + ["p_model_raw"]].rename(columns={"p_model_raw": "p_B1_raw"}),
        on=key, how="inner", validate="one_to_one",
    ).rename(columns={"p_model_raw": "p_B0_raw"})
    if len(merged) != len(b0) or len(merged) != len(b1):
        raise RuntimeError("B0/B1 OOF predictions do not contain the same complete validation rows")
    target = merged[experiment.TARGET_COL].to_numpy(dtype=np.int8)
    merged["p_delta_B1_minus_B0"] = merged["p_B1_raw"] - merged["p_B0_raw"]
    delta = merged["p_delta_B1_minus_B0"].to_numpy(dtype=np.float64)
    proxy_metrics = {
        "B0": _score_binary(target, merged["p_B0_raw"].to_numpy()),
        "B1": _score_binary(target, merged["p_B1_raw"].to_numpy()),
        "by_fold": {},
    }
    for fold_id, rows in merged.groupby("fold_id", sort=True):
        idx = rows.index.to_numpy()
        proxy_metrics["by_fold"][str(int(fold_id))] = {
            "B0": _score_binary(target[idx], merged["p_B0_raw"].to_numpy()[idx]),
            "B1": _score_binary(target[idx], merged["p_B1_raw"].to_numpy()[idx]),
        }
    delta_summary = {
        "rows": int(len(delta)),
        "mean": float(np.mean(delta)),
        "standard_deviation": float(np.std(delta)),
        "mean_absolute": float(np.mean(np.abs(delta))),
        "minimum": float(np.min(delta)),
        "p01": float(np.quantile(delta, 0.01)),
        "p05": float(np.quantile(delta, 0.05)),
        "median": float(np.quantile(delta, 0.50)),
        "p95": float(np.quantile(delta, 0.95)),
        "p99": float(np.quantile(delta, 0.99)),
        "maximum": float(np.max(delta)),
        "fraction_positive": float(np.mean(delta > 0)),
    }
    bootstrap_config = CONFIG["bootstrap"]
    proxy_bootstrap = _paired_block_bootstrap(
        target,
        merged["p_B0_raw"].to_numpy(),
        merged["p_B1_raw"].to_numpy(),
        merged["market_start_utc"],
        replications=int(bootstrap_config["replications"]),
        seed=int(bootstrap_config["seed"]),
    )

    official_columns = ["market_start_utc", "resolved_at_utc", "target_polymarket_up"]
    official = pd.read_parquet(MARKET_INDEX_PATH, columns=official_columns)
    official["market_start_utc"] = pd.to_datetime(official["market_start_utc"], utc=True, errors="raise")
    official["resolved_at_utc"] = pd.to_datetime(official["resolved_at_utc"], utc=True, errors="coerce")
    economic_start = pd.Timestamp(CONFIG["economics"]["start_utc"])
    dev_cutoff = pd.Timestamp(CONFIG["economics"]["cutoff_exclusive_utc"])
    official = official.loc[
        official["market_start_utc"].ge(economic_start)
        & official["market_start_utc"].lt(dev_cutoff)
        & official["resolved_at_utc"].notna()
    ].copy()
    official = official.loc[official["target_polymarket_up"].isin((0, 1))].copy()
    official = official.drop_duplicates("market_start_utc", keep=False)
    official_rows = merged.merge(official, on="market_start_utc", how="inner", validate="one_to_one")
    if official_rows.empty:
        raise RuntimeError("No official development outcomes join to the common OOF forecasts")
    official_rows, calibration_log_b0 = _causal_daily_platt(
        official_rows.drop(columns=["p_model_platt"], errors="ignore").rename(columns={"p_B0_raw": "p_model_raw"})
    )
    official_rows_b0 = official_rows.rename(columns={"p_model_raw": "p_B0_raw", "p_model_platt": "p_B0_platt"})
    b1_input = official_rows_b0.drop(columns=["p_model_platt"], errors="ignore").copy()
    b1_input["p_model_raw"] = b1_input["p_B1_raw"]
    official_rows_b1, calibration_log_b1 = _causal_daily_platt(b1_input)
    official_rows_b1.rename(columns={"p_model_platt": "p_B1_platt"}, inplace=True)
    official_rows_b0["p_B1_platt"] = official_rows_b1["p_B1_platt"].to_numpy()
    official_rows_b0["p_B1_raw"] = official_rows_b0["p_B1_raw"].to_numpy()
    official_target = official_rows_b0["target_polymarket_up"].to_numpy(dtype=np.int8)
    official_metrics = {
        "eligible_official_development_rows": int(len(official_rows_b0)),
        "official_outcomes": int(len(official_target)),
        "proxy_vs_official_target_disagreements": int(
            official_rows_b0[experiment.TARGET_COL].ne(official_rows_b0["target_polymarket_up"]).sum()
        ),
        "proxy_vs_official_target_agreement_rate": float(
            official_rows_b0[experiment.TARGET_COL].eq(official_rows_b0["target_polymarket_up"]).mean()
        ),
        "B0_raw": _score_binary(official_target, official_rows_b0["p_B0_raw"].to_numpy()),
        "B1_raw": _score_binary(official_target, official_rows_b0["p_B1_raw"].to_numpy()),
        "B0_platt": None,
        "B1_platt": None,
        "calibration_days": {
            "B0": calibration_log_b0,
            "B1": calibration_log_b1,
        },
    }
    mask0 = official_rows_b0["p_B0_platt"].notna().to_numpy()
    mask1 = official_rows_b0["p_B1_platt"].notna().to_numpy()
    if mask0.any():
        official_metrics["B0_platt"] = _score_binary(official_target[mask0], official_rows_b0.loc[mask0, "p_B0_platt"].to_numpy())
    if mask1.any():
        official_metrics["B1_platt"] = _score_binary(official_target[mask1], official_rows_b0.loc[mask1, "p_B1_platt"].to_numpy())
    official_metrics["by_fold_raw"] = {}
    for fold_id, rows in official_rows_b0.groupby("fold_id", sort=True):
        idx = rows.index.to_numpy()
        official_metrics["by_fold_raw"][str(int(fold_id))] = {
            "B0": _score_binary(official_target[idx], official_rows_b0["p_B0_raw"].to_numpy()[idx]),
            "B1": _score_binary(official_target[idx], official_rows_b0["p_B1_raw"].to_numpy()[idx]),
        }
    official_bootstrap = _paired_block_bootstrap(
        official_target,
        official_rows_b0["p_B0_raw"].to_numpy(),
        official_rows_b0["p_B1_raw"].to_numpy(),
        official_rows_b0["market_start_utc"],
        replications=int(bootstrap_config["replications"]),
        seed=int(bootstrap_config["seed"]),
    ) if len(official_rows_b0["market_start_utc"].dt.floor("D").unique()) >= 3 else None

    prediction_output = OUTPUT_DIR / "B0_B1_oof_comparison.parquet"
    forecast_output = OUTPUT_DIR / "B0_B1_official_oof.parquet"
    merged.to_parquet(prediction_output, index=False, compression="zstd")
    official_rows_b0.to_parquet(forecast_output, index=False, compression="zstd")

    summaries = {}
    for variant, calibrated_rows, calibration_log in (
        ("B0", official_rows_b0, calibration_log_b0),
        ("B1", official_rows_b0, calibration_log_b1),
    ):
        predictions = calibrated_rows.copy()
        predictions["p_model_raw"] = predictions[f"p_{variant}_raw"]
        predictions["p_model_platt"] = predictions[f"p_{variant}_platt"]
        predictions["prediction_available_at_utc"] = pd.NaT
        predictions[experiment.TARGET_COL] = predictions[experiment.TARGET_COL].astype(np.int8)
        prediction_dir = OUTPUT_DIR / "economics" / variant
        period_result = research._evaluate_portfolio_period(
            correction_manifest,
            predictions,
            start=economic_start,
            cutoff=dev_cutoff,
            policies=[("fixed_5_usd_ev_1pct", "fixed_5_usd", float(CONFIG["economics"]["minimum_net_ev_fraction"]))],
            output_root=prediction_dir,
            period_name=variant,
            input_identity={
                "experiment_id": CONFIG["experiment_id"],
                "variant": variant,
                "oof_predictions_sha256": _sha256(OUTPUT_DIR / f"{variant}_decision_oof.parquet"),
                "official_market_index_sha256": _sha256(MARKET_INDEX_PATH),
                "t59_snapshot_sha256": correction_manifest["stages"]["causal_t60_t59_book_replay"]["t59_coverage_sha256"],
                "t60_snapshot_sha256": correction_manifest["stages"]["causal_t60_t59_book_replay"]["t60_coverage_sha256"],
                "period_start_utc": economic_start.isoformat(),
                "period_cutoff_exclusive_utc": dev_cutoff.isoformat(),
                "policy": "fixed_5_usd with 1% minimum net EV; existing T-60/T-59 engine",
                "calibration_schedule": "causal daily Platt using earlier official OOF labels only",
            },
        )
        summaries[variant] = _economic_summary_from_artifacts(period_result, prediction_dir)

    summary = {
        "shared_oof_rows": int(len(merged)),
        "proxy_oof_quality": proxy_metrics,
        "official_development_oof_quality": official_metrics,
        "paired_3day_block_bootstrap": {
            "proxy_oof": proxy_bootstrap,
            "official_raw_oof": official_bootstrap,
        },
        "raw_prediction_delta_B1_minus_B0": delta_summary,
        "economic_comparison": summaries,
        "economic_scope": {
            "period_start_inclusive_utc": economic_start.isoformat(),
            "period_cutoff_exclusive_utc": dev_cutoff.isoformat(),
            "markets": "only qualifying development markets with OOF forecasts and official outcomes",
            "stake_usd": float(CONFIG["economics"]["gross_stake_usd"]),
            "minimum_net_ev_fraction": float(CONFIG["economics"]["minimum_net_ev_fraction"]),
            "decision_snapshot": "T-60",
            "execution_snapshot": "T-59",
            "book_archive_replay": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = OUTPUT_DIR / "quality_and_economics.json"
    _write_json(summary_path, summary)
    output_files = [summary_path, prediction_output, forecast_output]
    for variant in ("B0", "B1"):
        output_files.extend((OUTPUT_DIR / "economics" / variant).rglob("*"))
    output_records = [_record(path) for path in output_files if path.is_file()]
    _finish_stage(
        manifest, stage_name, summary=summary, outputs=output_records,
        elapsed_seconds=time.perf_counter() - started,
    )
    return summary


def _fmt_metric(value, digits: int = 6) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{float(value):.{digits}f}"


def _write_final_report(
    manifest: dict,
    source: dict,
    archive_summary: dict,
    common_summary: dict,
    b0: dict,
    b1: dict,
    quality: dict,
) -> None:
    proxy = quality["proxy_oof_quality"]
    official = quality["official_development_oof_quality"]
    bootstrap = quality["paired_3day_block_bootstrap"]
    delta = quality["raw_prediction_delta_B1_minus_B0"]
    econ = quality["economic_comparison"]
    feature_use = b1.get("new_feature_use") or {}
    new_features = feature_use.get("new_features", [])
    flow_peak = manifest["stages"]["flow_features"].get("peak_process_working_set_gib")
    common_peak = manifest["stages"]["common_rows"].get("peak_process_working_set_gib")
    b0_peak = b0.get("peak_process_working_set_gib")
    b1_peak = b1.get("peak_process_working_set_gib")
    b0_memory = (
        f"peak {b0_peak:.2f} GiB"
        if b0_peak is not None else
        f"peak unavailable; observed at least {b0.get('observed_working_set_lower_bound_gib', 0.0):.2f} GiB during CV"
    )
    b1_memory = f"peak {b1_peak:.2f} GiB" if b1_peak is not None else "peak unavailable"
    flow_memory = f"peak {flow_peak:.2f} GiB" if flow_peak is not None else "peak unavailable"
    common_memory = f"peak {common_peak:.2f} GiB" if common_peak is not None else "peak unavailable"
    b0_recovery_note = (
        "B0's post-fit summary was recovered from its saved fold boosters and persisted OOF predictions; "
        "the boosters reproduced those predictions and no fit was repeated. Its full per-iteration history "
        "was not persisted, and its runtime is inferred from the OOF file timestamp."
        if b0.get("recovered_from_saved_cv_boosters_without_refitting") else ""
    )
    feature_lines = [
        f"| `{row['feature']}` | {row['split_count_across_folds']:,} | {row['folds_with_nonzero_splits']}/10 | {row['gain_across_folds']:.3f} |"
        for row in new_features
    ]
    fold_lines = []
    for fold_id in sorted(proxy["by_fold"], key=int):
        fold = proxy["by_fold"][fold_id]
        fold_lines.append(
            f"| {fold_id} | {fold['B0']['rows']:,} | {_fmt_metric(fold['B0']['log_loss'])} | {_fmt_metric(fold['B1']['log_loss'])} | "
            f"{_fmt_metric(fold['B0']['brier'])} | {_fmt_metric(fold['B1']['brier'])} | "
            f"{_fmt_metric(fold['B0']['auc'], 4)} | {_fmt_metric(fold['B1']['auc'], 4)} |"
        )
    official_fold_lines = []
    for fold_id in sorted(official["by_fold_raw"], key=int):
        fold = official["by_fold_raw"][fold_id]
        official_fold_lines.append(
            f"| {fold_id} | {fold['B0']['rows']:,} | {_fmt_metric(fold['B0']['log_loss'])} | {_fmt_metric(fold['B1']['log_loss'])} | "
            f"{_fmt_metric(fold['B0']['brier'])} | {_fmt_metric(fold['B1']['brier'])} | "
            f"{_fmt_metric(fold['B0']['auc'], 4)} | {_fmt_metric(fold['B1']['auc'], 4)} |"
        )
    bootstrap_lines = []
    for scope, item in (("Proxy OOF", bootstrap["proxy_oof"]), ("Official raw OOF", bootstrap["official_raw_oof"])):
        if item is None:
            bootstrap_lines.append(f"| {scope} | n/a | insufficient UTC days | n/a | n/a | n/a |")
            continue
        intervals = item["intervals"]
        bootstrap_lines.append(
            f"| {scope} | {item['observed_rows']:,} | "
            f"[{intervals['log_loss']['b1_minus_b0_95pct'][0]:+.6f}, {intervals['log_loss']['b1_minus_b0_95pct'][1]:+.6f}] | "
            f"[{intervals['brier']['b1_minus_b0_95pct'][0]:+.6f}, {intervals['brier']['b1_minus_b0_95pct'][1]:+.6f}] | "
            f"[{intervals['auc']['b1_minus_b0_95pct'][0]:+.5f}, {intervals['auc']['b1_minus_b0_95pct'][1]:+.5f}] | {item['replications']} |"
        )
    economic_lines = []
    for variant in ("B0", "B1"):
        row = econ[variant]
        economic_lines.append(
            f"| {variant} | {row['decision_count']:,} | {row['fill_count']:,} | {row['settlement_count']:,} | "
            f"${row['gross_turnover_usd']:,.2f} | ${row['estimated_fees_usd']:,.2f} | ${row['net_pnl_usd']:,.2f} | "
            f"{_fmt_metric(row['pnl_per_dollar_turnover'], 4)} | ${row['forecast_net_ev_at_t60_for_filled_trades_usd']:,.2f} | "
            f"${row['realized_net_pnl_for_filled_trades_usd']:,.2f} | {row['accounting_audit']} |"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# BTC pre-open taker-flow experiment — 2026-10-18

## Run and source

- Status: `completed`. The model comparison uses the completed binary full-development run `{source['run_manifest']['run_id']}` and its exact 64-feature order.
- Source training rows: {common_summary['source_rows_after_fit_label_filter']:,}; decision rows: {common_summary['source_decision_rows']:,}; common complete rows for both variants: {common_summary['rows_after_shared_complete_feature_mask']:,} ({common_summary['decision_rows_after_shared_complete_feature_mask']:,} decision rows).
- The source's 10 chronological fold fingerprints were recovered and matched exactly before applying the shared completeness mask. Validation rows remain decision minutes only, and training labels are purged at the source fold boundary.
- Saved target remains `{common_summary['target']}`: `{common_summary['target_formula']}`. For a noon market, the last closed candle opens at 11:58 UTC, the information boundary is 11:59, and the market window is 12:00–12:05 UTC. No features use an unclosed candle.
- Model parameters: `{json.dumps(MODEL_PARAMETERS, sort_keys=True)}`; full-development fit seed `{MODEL_SEED}`; fixed decision/auxiliary weights `{DECISION_WEIGHT:.2f}` / `{AUXILIARY_WEIGHT:.4f}`. Execution used {THREADS} LightGBM threads to preserve headroom; fold logic, weights, GPU backend, and selected hyperparameters were held fixed.
- The control variant B0 uses the 64 source features plus `taker_Z` and `taker_V` at 1, 5, and 15 minutes ({b0['feature_count']} total). B1 adds `taker_I`, `taker_U`, `taker_beta`, and `taker_A` at those horizons ({b1['feature_count']} total). Both variants use the same rows.

## Result

The taker-flow additions did not improve this frozen pipeline. Proxy OOF scores were nearly tied and their paired 95% bootstrap intervals include zero; on the 21,951 official development outcomes, B1 had worse log loss, Brier score, and AUC, with all three paired intervals favoring B0. In the fixed-$5 replay, net PnL was $115.67 for B1 versus $260.05 for B0. These results give no evidence to add the new features to the current model; the comparison used development data only and does not establish holdout performance.

## Binance USD-M Futures source

Data came from official Binance Public Data monthly BTCUSDT 1-minute USD-M Futures kline archives, June 2020 through June 2026. Monthly ZIP files were checked against their SHA-256 sidecars. We used only OHLC, quote volume, taker-buy quote volume, and source timestamps; no trades or order-book data were downloaded. Binance documents the monthly archive and kline fields in its [Public Data README](https://github.com/binance/binance-public-data).

- Monthly archives checked: {archive_summary['archive_count']}; valid rows on the expected monthly range: {archive_summary['observed_valid_bar_rows']:,}; missing or invalid minutes on the full grid: {archive_summary['missing_or_invalid_bar_minutes']:,}.
- Invalid OHLC rows: {archive_summary['invalid_ohlc_rows']:,}; rows where quote volume or taker-buy volume failed `0 ≤ taker_buy_quote_volume ≤ quote_volume`: {archive_summary['invalid_taker_buy_quote_rows']:,}; duplicate timestamp rows excluded: {archive_summary['duplicate_rows_excluded']:,}.
- Missing bars and invalid quote/taker values remain missing. No forward fill or clipping is applied.

For each horizon `h`, `Q=sum(quote_volume)`, `D=sum(2*taker_buy_quote_volume−quote_volume)`, `I=D/Q`, and `U=D/(h*qbar)`, with `qbar` the mean quote volume in the preceding 1,440 minutes. `Z=log(Close_end/Open_start)/(sqrt(h)*max(sigma,1e-6))`, where `sigma` is the preceding 60-minute standard deviation of BTCUSDT log returns. `V=log1p(Q/(h*qbar))`. `beta` is the rolling slope of `Z` on `U` over valid pairs in the previous 10,080 minutes (minimum 1,440 pairs), excluding the current horizon; `A` is the regression residual. Near-zero `U` variance sets beta to zero. The old `candle_signed_vol` feature is candle-body direction (`sign(close-open)`), and `session_close_impulse_BVC` is a separate proxy; neither contains Binance taker-buy quote volume.

## Development proxy OOF quality

One fixed-parameter 10-fold CV produced both the selected iteration and the fold boosters used for OOF; there were no Optuna trials and no second OOF fit.

| Variant | Features | Best iteration | CV log loss mean | CV fold SD | CV objective | OOF rows | OOF log loss | OOF Brier | OOF AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | {b0['feature_count']} | {b0['best_iteration']} | {_fmt_metric(b0['cv_binary_logloss_mean_at_best'])} | {_fmt_metric(b0['cv_binary_logloss_population_std_at_best'])} | {_fmt_metric(b0['best_cv_objective'])} | {proxy['B0']['rows']:,} | {_fmt_metric(proxy['B0']['log_loss'])} | {_fmt_metric(proxy['B0']['brier'])} | {_fmt_metric(proxy['B0']['auc'], 4)} |
| B1 | {b1['feature_count']} | {b1['best_iteration']} | {_fmt_metric(b1['cv_binary_logloss_mean_at_best'])} | {_fmt_metric(b1['cv_binary_logloss_population_std_at_best'])} | {_fmt_metric(b1['best_cv_objective'])} | {proxy['B1']['rows']:,} | {_fmt_metric(proxy['B1']['log_loss'])} | {_fmt_metric(proxy['B1']['brier'])} | {_fmt_metric(proxy['B1']['auc'], 4)} |

{b0_recovery_note}

| Fold | Rows | B0 log loss | B1 log loss | B0 Brier | B1 Brier | B0 AUC | B1 AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(fold_lines)}

## Official development OOF quality

These metrics use the official Polymarket labels available for development markets from {quality['economic_scope']['period_start_inclusive_utc']} to before {quality['economic_scope']['period_cutoff_exclusive_utc']}. Raw metrics score all joined official outcomes. Platt calibration is fitted separately for B0 and B1 once per UTC day, using only earlier OOF official labels whose release time is before that day's first decision; the calibrator stays unavailable until it has {CONFIG['official_outcomes']['minimum_calibration_rows_from_source_calibrator']:,} prior labels and both classes.

| Forecast | Rows | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|
| B0 raw | {official['B0_raw']['rows']:,} | {_fmt_metric(official['B0_raw']['log_loss'])} | {_fmt_metric(official['B0_raw']['brier'])} | {_fmt_metric(official['B0_raw']['auc'], 4)} |
| B1 raw | {official['B1_raw']['rows']:,} | {_fmt_metric(official['B1_raw']['log_loss'])} | {_fmt_metric(official['B1_raw']['brier'])} | {_fmt_metric(official['B1_raw']['auc'], 4)} |
| B0 daily Platt | {official['B0_platt']['rows'] if official['B0_platt'] else 0:,} | {_fmt_metric(official['B0_platt']['log_loss'] if official['B0_platt'] else None)} | {_fmt_metric(official['B0_platt']['brier'] if official['B0_platt'] else None)} | {_fmt_metric(official['B0_platt']['auc'] if official['B0_platt'] else None, 4)} |
| B1 daily Platt | {official['B1_platt']['rows'] if official['B1_platt'] else 0:,} | {_fmt_metric(official['B1_platt']['log_loss'] if official['B1_platt'] else None)} | {_fmt_metric(official['B1_platt']['brier'] if official['B1_platt'] else None)} | {_fmt_metric(official['B1_platt']['auc'] if official['B1_platt'] else None, 4)} |

The proxy label matched the official outcome on {official['eligible_official_development_rows'] - official['proxy_vs_official_target_disagreements']:,}/{official['eligible_official_development_rows']:,} joined development OOF markets ({official['proxy_vs_official_target_agreement_rate']:.2%}).

| Fold | Official rows | B0 log loss | B1 log loss | B0 Brier | B1 Brier | B0 AUC | B1 AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(official_fold_lines)}

## Paired uncertainty and forecast changes

Intervals below are for B1 minus B0 on identical rows, using {bootstrap['proxy_oof']['replications']:,} paired circular moving-block bootstrap replications over 3-day UTC blocks. Negative log-loss/Brier deltas favor B1; positive AUC deltas favor B1.

| Scope | Rows | Log-loss delta 95% interval | Brier delta 95% interval | AUC delta 95% interval | Replications |
|---|---:|---:|---:|---:|---:|
{chr(10).join(bootstrap_lines)}

Raw probability delta (`B1−B0`): n={delta['rows']:,}, mean={delta['mean']:+.6f}, SD={delta['standard_deviation']:.6f}, mean absolute={delta['mean_absolute']:.6f}, p01/p05/median/p95/p99={delta['p01']:+.6f}/{delta['p05']:+.6f}/{delta['median']:+.6f}/{delta['p95']:+.6f}/{delta['p99']:+.6f}; positive on {delta['fraction_positive']:.2%} of rows.

## Use of the additional features

B1 used {feature_use.get('new_features_used_in_any_split', 0)} of {feature_use.get('new_feature_count', 0)} added features in at least one tree split, and {feature_use.get('new_features_used_in_all_folds', 0)} in every fold. The joint permutation diagnostic permuted the 12 added columns together within each validation fold without refitting; mean log-loss increase was {_fmt_metric(feature_use.get('joint_block_permutation_mean_log_loss_increase'))}.

| Added feature | Split count across folds | Folds with splits | Gain across folds |
|---|---:|---:|---:|
{chr(10).join(feature_lines)}

## Fixed-$5 development economics

The existing causal portfolio engine independently recomputed each model over qualifying official markets, with $5 gross stake, a 1% T-60 net-EV hurdle, T-59 execution snapshots, existing fees/depth rules, and settlement cash release. The saved T-60/T-59 snapshots were hash-verified and reused; no book replay was run. Forecast EV sums T-60 net EV for filled trades; realized PnL uses T-59 fills and official settlements.

| Variant | Decisions | Fills | Settlements | Turnover | Fees | Net PnL | PnL / turnover | Forecast EV | Realized PnL | Audit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(economic_lines)}

## Runtime and artifacts

- Archive validation and features: {manifest['stages']['flow_features']['elapsed_seconds']:.1f}s; {flow_memory}.
- Binance ZIP acquisition: at least {manifest['stages']['download_binance'].get('observed_zip_file_timestamp_span_seconds', 0.0):.1f}s from the first archive file creation to the last ZIP write; the exact first-attempt stage timer was overwritten on resume. Later ZIP/checksum revalidation took {manifest['stages']['download_binance'].get('cumulative_elapsed_seconds', manifest['stages']['download_binance']['elapsed_seconds']):.1f}s across {manifest['stages']['download_binance']['archive_count']} monthly files.
- Shared rows and fold reconstruction: {manifest['stages']['common_rows']['elapsed_seconds']:.1f}s; {common_memory}.
- B0 CV: {b0['elapsed_seconds']:.1f}s; {b0_memory}. B1 CV: {b1['elapsed_seconds']:.1f}s; {b1_memory}. B1 estimate after B0 was {manifest.get('B1_runtime_estimate_after_B0', {}).get('estimated_seconds')} seconds.
- OOF, bootstrap, calibration, and economics: {manifest['stages']['quality_and_economics']['elapsed_seconds']:.1f}s; sampled working set reached at least {manifest['stages']['quality_and_economics'].get('observed_working_set_lower_bound_gib', 0.0):.2f} GiB, but continuous peak was not sampled.
- Total experiment elapsed from the first attempt through completion: {manifest.get('elapsed_seconds', (pd.Timestamp.now(tz='UTC') - pd.Timestamp(manifest['started_utc'])).total_seconds()):.1f}s.
- Run manifest: `{_relative(MANIFEST_PATH)}`; source data audit: `{_relative(OUTPUT_DIR / 'binance_source_validation.json')}`; quality and economics: `{_relative(OUTPUT_DIR / 'quality_and_economics.json')}`.
- CV models and OOF predictions are saved under `{_relative(OUTPUT_DIR)}`. The experiment uses no holdout outcomes or deployment refit.
- The comparison is conditional on the source run's already selected feature set and hyperparameters; it is a controlled feature addition within that frozen pipeline.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def _run_experiment() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = _load_source_artifacts()
    identity = _make_input_identity(source)
    manifest = _load_or_create_manifest(identity)
    try:
        previous_download = manifest.get("stages", {}).get("download_binance", {})
        previous_download_elapsed = (
            float(previous_download.get("cumulative_elapsed_seconds", previous_download.get("elapsed_seconds", 0.0)))
            if previous_download.get("status") == "completed" else 0.0
        )
        _begin_stage(manifest, "download_binance", "Compute causal taker-flow features from closed 1-minute candles.")
        download_started = time.perf_counter()
        archive_records = _download_archives(manifest)
        archive_stage = {
            "archive_count": len(archive_records),
            "months": [record["month"] for record in archive_records],
            "total_compressed_bytes": int(sum(record["size_bytes"] for record in archive_records)),
            "archive_sha256_by_month": {record["month"]: record["sha256"] for record in archive_records},
            "checksum_sidecars_verified": True,
            "elapsed_seconds": time.perf_counter() - download_started,
            "cumulative_elapsed_seconds": previous_download_elapsed + time.perf_counter() - download_started,
        }
        _finish_stage(manifest, "download_binance", **archive_stage)

        flow_path, archive_summary = _build_flow_features(manifest, archive_records)
        common_path, common_summary = _prepare_common_rows(manifest, flow_path)

        b0_summary = _fit_variant(manifest, "B0", common_path)
        b0_seconds = float(b0_summary["elapsed_seconds"])
        estimate = {
            "basis": "measured B0 70-feature CV wall time multiplied by 82/70 feature count",
            "estimated_seconds": round(b0_seconds * 82.0 / 70.0, 1),
            "measured_B0_seconds": b0_seconds,
        }
        manifest["B1_runtime_estimate_after_B0"] = estimate
        _write_json(MANIFEST_PATH, manifest)
        _status_update(manifest)
        print(
            f"[btc-taker-flow] B0 CV completed in {b0_seconds:.1f}s; "
            f"estimated B1 CV {estimate['estimated_seconds']:.1f}s; starting B1.",
            flush=True,
        )
        b1_summary = _fit_variant(manifest, "B1", common_path)
        quality = _score_and_evaluate(manifest, common_path)
        manifest["elapsed_seconds"] = (
            pd.Timestamp.now(tz="UTC") - pd.Timestamp(manifest["started_utc"])
        ).total_seconds()
        _write_json(MANIFEST_PATH, manifest)

        _begin_stage(manifest, "report", "Complete the manifest and status document.")
        _write_final_report(
            manifest, source, archive_summary, common_summary, b0_summary, b1_summary, quality,
        )
        _finish_stage(manifest, "report", outputs=[_record(REPORT_PATH)])
        manifest["status"] = "completed"
        manifest["active_stage"] = None
        manifest["next_step"] = "Experiment complete; review the report and saved artifacts."
        manifest["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        manifest["elapsed_seconds"] = (
            pd.Timestamp(manifest["finished_utc"]) - pd.Timestamp(manifest["started_utc"])
        ).total_seconds()
        manifest["resource_snapshot_at_completion"] = _resource_snapshot()
        _write_json(MANIFEST_PATH, manifest)
        _status_update(manifest)
        print(f"[btc-taker-flow] completed: {_relative(REPORT_PATH)}", flush=True)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        manifest["failure"] = {
            "stage": manifest.get("active_stage"),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        manifest["next_step"] = "Inspect the failed stage, then rerun this no-argument script to resume verified checkpoints."
        _write_json(MANIFEST_PATH, manifest)
        _status_update(manifest)
        raise


def _build_flow_features(manifest: dict, archive_records: list[dict]) -> tuple[Path, dict]:
    output_path = OUTPUT_DIR / "flow_features.parquet"
    source_validation_path = OUTPUT_DIR / "binance_source_validation.json"
    prior = manifest.get("stages", {}).get("flow_features", {})
    if prior.get("status") == "completed" and _verify_records(prior.get("outputs", [])):
        return output_path, _read_json(source_validation_path)

    _begin_stage(manifest, "flow_features", "Run B0 fixed-parameter 10-fold CV.")
    started = time.perf_counter()
    with PeakMemorySampler() as memory:
        source, validation = _load_and_validate_archives(archive_records)
        features = compute_taker_flow_features(
            source,
            horizons=HORIZONS,
            qbar_window=int(CONFIG["flow_features"]["qbar_window_minutes"]),
            sigma_window=int(CONFIG["flow_features"]["sigma_window_minutes"]),
            beta_window=int(CONFIG["flow_features"]["beta_window_minutes"]),
            minimum_beta_pairs=int(CONFIG["flow_features"]["minimum_beta_pairs"]),
            beta_variance_tolerance=float(CONFIG["flow_features"]["beta_variance_tolerance"]),
            sigma_floor=float(CONFIG["flow_features"]["sigma_floor"]),
            sigma_ddof=int(CONFIG["flow_features"]["sigma_ddof"]),
        )
        feature_frame = features.reset_index(names="Opened")
        temporary = output_path.with_suffix(".parquet.tmp")
        feature_frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(output_path)
        validation["feature_rows"] = int(len(feature_frame))
        validation["feature_columns"] = list(FLOW_FEATURES)
        validation["invalid_or_incomplete_flow_feature_rows"] = int(
            (~np.isfinite(feature_frame.loc[:, FLOW_FEATURES].to_numpy(dtype=np.float64)).all(axis=1)).sum()
        )
        validation["feature_config"] = CONFIG["flow_features"]
        validation["elapsed_seconds"] = time.perf_counter() - started
        validation["peak_process_working_set_gib"] = round(memory.peak_bytes / 2**30, 3)
        _write_json(source_validation_path, validation)
        del source, features, feature_frame
    outputs = [_record(output_path), _record(source_validation_path)]
    _finish_stage(
        manifest,
        "flow_features",
        elapsed_seconds=time.perf_counter() - started,
        peak_process_working_set_gib=round(memory.peak_bytes / 2**30, 3),
        outputs=outputs,
        row_count=validation["feature_rows"],
        invalid_source_rows=int(validation["invalid_taker_buy_quote_rows"] + validation["invalid_ohlc_rows"]),
    )
    return output_path, validation


def _make_input_identity(source: dict) -> dict:
    return {
        "experiment_config_sha256": source["source_hashes"]["experiment_config_sha256"],
        "source_run_manifest_sha256": source["source_hashes"]["source_run_manifest_sha256"],
        "source_effective_config_sha256": source["source_hashes"]["source_effective_config_sha256"],
        "source_feature_dataset_sha256": source["dataset_record"]["sha256"],
        "selected_features_sha256": source["source_hashes"]["selected_features_sha256"],
        "model_tuning_best_result_sha256": source["source_hashes"]["model_tuning_best_result_sha256"],
        "source_training_manifest_sha256": source["source_hashes"]["source_training_manifest_sha256"],
        "official_market_index_sha256": source["source_hashes"]["official_market_index_sha256"],
        "source_code_sha256": source["source_hashes"]["source_code_sha256"],
        "experiment_code_sha256": source["source_hashes"]["experiment_code_sha256"],
        "target": "saved target_preopen_5m_up_binance_index_proxy; training label availability <= 2026-06-30T23:59:00Z",
        "flow_feature_config": CONFIG["flow_features"],
        "binance_month_range": [
            CONFIG["binance_source"]["first_month"], CONFIG["binance_source"]["last_month"],
        ],
        "cv_method": CONFIG["model_comparison"],
        "official_and_economic_method": CONFIG["official_outcomes"] | CONFIG["economics"],
    }


def _load_or_create_manifest(identity: dict) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.is_file():
        manifest = _read_json(MANIFEST_PATH)
        previous_identity = manifest.get("input_identity")
        if previous_identity != identity:
            previous_comparable = json.loads(json.dumps(previous_identity))
            current_comparable = json.loads(json.dumps(identity))
            previous_runner_hash = previous_comparable.get("experiment_code_sha256", {}).pop("runner", None)
            current_runner_hash = current_comparable.get("experiment_code_sha256", {}).pop("runner", None)
            completed_stages = {
                name for name, stage in manifest.get("stages", {}).items()
                if stage.get("status") == "completed"
            }
            failure = manifest.get("failure", {})
            archive_only_revision = (
                completed_stages == {"download_binance"}
                and failure.get("stage") == "flow_features"
                and failure.get("error_type") == "OSError"
                and not (OUTPUT_DIR / "flow_features.parquet").exists()
                and not (OUTPUT_DIR / "binance_source_validation.json").exists()
            )
            saved_cv_recovery = (
                completed_stages == {"download_binance", "flow_features", "common_rows"}
                and failure.get("stage") == "cv_B0"
                and failure.get("error_type") == "NameError"
                and "decision" in failure.get("error", "")
                and all(
                    _verify_records(manifest.get("stages", {}).get(stage_name, {}).get("outputs", []))
                    for stage_name in ("flow_features", "common_rows")
                )
                and (OUTPUT_DIR / "B0_decision_oof.parquet").is_file()
                and all((OUTPUT_DIR / "models" / "B0" / f"fold_{fold_id:02d}.txt").is_file()
                        for fold_id in range(CV_FOLD_COUNT))
                and not (OUTPUT_DIR / "B0_cv_summary.json").exists()
                and not (OUTPUT_DIR / "B0_cv_history.json").exists()
            )
            precompute_runner_fix = (
                previous_comparable == current_comparable
                and previous_runner_hash is not None
                and current_runner_hash is not None
                and (archive_only_revision or saved_cv_recovery)
            )
            if not precompute_runner_fix:
                raise RuntimeError("Existing taker-flow checkpoint belongs to different inputs or code")
            manifest.setdefault("code_revision_history", []).append({
                "previous_runner_sha256": previous_runner_hash,
                "current_runner_sha256": current_runner_hash,
                "reason": (
                    "recovered complete saved B0 CV boosters and OOF after a post-fit NameError"
                    if saved_cv_recovery else
                    "fixed Windows process-memory API before any Binance flow feature computation"
                ),
                "completed_stages_preserved": sorted(completed_stages),
            })
            manifest["input_identity"] = identity
            manifest["source_provenance"]["source_hashes"]["experiment_code_sha256"] = SOURCE_HASHES[
                "experiment_code_sha256"
            ]
            manifest["source_provenance"].setdefault("runner_code_revision_history", []).append({
                "previous_runner_sha256": previous_runner_hash,
                "current_runner_sha256": current_runner_hash,
                "reason": (
                    "post-fit B0 output recovery without repeating CV"
                    if saved_cv_recovery else "memory sampler correction before compute"
                ),
            })
            _write_json(MANIFEST_PATH, manifest)
        return manifest
    manifest = {
        "schema_version": 1,
        "experiment_id": CONFIG["experiment_id"],
        "status": "running",
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_identity": identity,
        "source_provenance": {
            "source_report": CONFIG["source_run"]["report_path"],
            "source_run_manifest": _relative(SOURCE_RUN_PATH),
            "source_effective_config": _relative(SOURCE_CONFIG_PATH),
            "selected_feature_artifact": _relative(SOURCE_SELECTION_PATH),
            "model_tuning_artifact": _relative(SOURCE_TUNING_PATH),
            "source_training_manifest": _relative(SOURCE_TRAINING_PATH),
            "source_binary_model": SOURCE_TRAINING["model_path"],
            "source_hashes": SOURCE_HASHES,
            "full_development_feature_order": SOURCE_FEATURES,
            "effective_main_model_hyperparameters": MODEL_PARAMETERS,
            "source_model_best_iteration_recorded_but_not_reused": int(SOURCE_TUNING["best_iteration"]),
            "main_model_seed": MODEL_SEED,
            "decision_weight": DECISION_WEIGHT,
            "auxiliary_row_weight": AUXILIARY_WEIGHT,
            "weight_rule": "source _preopen_sample_weights: decision rows receive 0.59; all other eligible minutes receive (1-0.59)/4; no target_5m_weight column multiplier",
            "fold_count": CV_FOLD_COUNT,
            "fold_test_to_train_ratio": float(CONFIG["model_comparison"]["test_to_train_ratio_from_source_model_runner"]),
            "maximum_estimators": MAX_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "cv_objective": CONFIG["model_comparison"]["iteration_objective"],
            "execution_threads": THREADS,
            "execution_thread_note": "fixed at eight to bound CPU and native-library concurrency; source hyperparameters, weights, seeds, folds, and GPU backend are unchanged",
        },
        "resource_snapshot_before_run": _resource_snapshot(),
        "stages": {},
        "binance_month_archives": {},
    }
    _write_json(MANIFEST_PATH, manifest)
    return manifest


def run() -> dict:
    pa.set_cpu_count(THREADS)
    pa.set_io_thread_count(THREADS)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return _run_experiment()
    with threadpool_limits(limits=THREADS):
        return _run_experiment()


if __name__ == "__main__":
    run()
