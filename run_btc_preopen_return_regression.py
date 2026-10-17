"""Run one fixed-scope BTC return-regression experiment and its frozen evaluation."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from features.btc_preopen_contract import (
    DECISION_AVAILABLE_COL,
    DECISION_COL,
    RETURN_TARGET_COL,
    TARGET_AVAILABLE_COL,
    TARGET_COL,
)
from features.btc_preopen_return_target import (
    VOLATILITY_DENOMINATOR_FLOOR,
    build_preopen_sigma_5m,
    normalize_preopen_return,
)
from utils.cv_objective import ObjectiveAlignedCVCallback, cv_objective_series

import optimize_lgbm_hyperparameters as model_tuning
import run_btc_preopen_experiment as experiment
import run_btc_preopen_fee_correction as fee_correction
import run_btc_preopen_research as research
import run_btc_preopen_signal_diagnostic as diagnostic
import run_btc_preopen_t59_reassessment as replay


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_preopen_return_regression_20261017.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_dir"]
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
STATUS_PATH = ROOT / CONFIG["status_path"]
STATUS_BEGIN = "<!-- BTC_RETURN_REGRESSION_STATUS_BEGIN -->"
STATUS_END = "<!-- BTC_RETURN_REGRESSION_STATUS_END -->"
HUBER_METRIC_NAME = "decision_huber_alpha_0_9"
MIN_CALIBRATION_ROWS = int(CONFIG["calibration"]["minimum_prior_oof_rows"])
PROBABILITY_EPSILON = float(CONFIG["calibration"]["probability_clip_for_scoring"])


class TuningBudgetExpired(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _array_sha256(values, dtype) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _resource_snapshot() -> dict:
    result = {"logical_processors": os.cpu_count(), "python": platform.python_version()}
    try:
        memory = ctypes.Structure

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        result.update({
            "total_ram_bytes": int(status.ullTotalPhys),
            "available_ram_bytes": int(status.ullAvailPhys),
            "memory_load_percent": int(status.dwMemoryLoad),
        })
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), ctypes.wintypes.DWORD,
        ]
        get_process_memory_info.restype = ctypes.wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not get_process_memory_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        result["process_peak_working_set_bytes"] = int(counters.PeakWorkingSetSize)
        result["process_working_set_bytes"] = int(counters.WorkingSetSize)
    except Exception:
        result.setdefault("total_ram_bytes", None)
        result.setdefault("available_ram_bytes", None)
        result.setdefault("process_peak_working_set_bytes", None)
    try:
        import subprocess

        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        result["gpu_snapshot"] = completed.stdout.strip() if completed.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        result["gpu_snapshot"] = None
    result["lightgbm_version"] = lgb.__version__
    result["optuna_version"] = optuna.__version__
    return result


def _write_status(*, status: str, stage: str, completed: list[str], next_step: str) -> None:
    current = STATUS_PATH.read_text(encoding="utf-8") if STATUS_PATH.is_file() else ""
    block = "\n".join([
        STATUS_BEGIN,
        "### Return-regression experiment checkpoint",
        f"- Updated UTC: `{pd.Timestamp.now(tz='UTC').isoformat()}`.",
        f"- Status: `{status}`; active stage: `{stage}`.",
        f"- Completed stages: `{', '.join(completed) or 'none'}`.",
        f"- Next action: {next_step}",
        f"- Run manifest: `{MANIFEST_PATH.resolve().relative_to(ROOT).as_posix()}`.",
        STATUS_END,
    ])
    if STATUS_BEGIN in current and STATUS_END in current:
        before, remainder = current.split(STATUS_BEGIN, 1)
        _, after = remainder.split(STATUS_END, 1)
        updated = before + block + after
    else:
        updated = block + "\n\n" + current
    temporary = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(STATUS_PATH)


def _metric_score(y_true, probability) -> dict:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.clip(np.asarray(probability, dtype=np.float64), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    loss = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    bins = []
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        selected = (p >= edges[index]) & ((p < edges[index + 1]) if index < 9 else (p <= edges[index + 1]))
        if selected.any():
            bins.append({
                "bin": index + 1,
                "count": int(selected.sum()),
                "mean_probability": float(p[selected].mean()),
                "observed_up_rate": float(y[selected].mean()),
            })
    ece = float(sum(
        item["count"] / len(y) * abs(item["mean_probability"] - item["observed_up_rate"])
        for item in bins
    )) if len(y) else None
    return {
        "n": int(len(y)),
        "up_rate": float(y.mean()) if len(y) else None,
        "log_loss": float(loss.mean()) if len(y) else None,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "ece_10_equal_width": ece,
        "calibration_bins": bins,
    }


def _huber_loss(y_true, prediction, alpha: float = 0.9) -> float:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    absolute = np.abs(residual)
    return float(np.mean(np.where(absolute <= alpha, 0.5 * residual**2, alpha * (absolute - 0.5 * alpha))))


def _huber_eval(prediction, dataset):
    return HUBER_METRIC_NAME, _huber_loss(dataset.get_label(), prediction, float(CONFIG["model"]["alpha"])), False


def _base_lgb_params(feature_count: int) -> dict:
    seed = int(CONFIG["model"]["seed"])
    return {
        "objective": "huber",
        "alpha": float(CONFIG["model"]["alpha"]),
        "metric": "None",
        "boosting_type": "gbdt",
        "device_type": CONFIG["model"]["compute_backend"],
        "verbosity": -1,
        "num_threads": int(CONFIG["model"]["num_threads"]),
        "max_bin": int(CONFIG["model"]["max_bin"]),
        "feature_pre_filter": False,
        "gpu_use_dp": False,
        "deterministic": True,
        "force_col_wise": True,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
    }


class _DeadlineCallback:
    order = 40
    before_iteration = False

    def __init__(self, deadline: float):
        self.deadline = deadline

    def __call__(self, env):
        if time.perf_counter() >= self.deadline:
            raise TuningBudgetExpired("The 2,700-second tuning budget expired during a fold")


def _load_bundle_inputs() -> tuple[dict, dict, dict, dict]:
    bundle_path = ROOT / CONFIG["bundle_path"]
    if _sha256(bundle_path) != CONFIG["bundle_sha256"]:
        raise RuntimeError("The 64-feature benchmark bundle changed after the experiment was frozen")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    training_path = ROOT / CONFIG["binary_training_manifest_path"]
    tuning_path = ROOT / CONFIG["binary_tuning_result_path"]
    training = json.loads(training_path.read_text(encoding="utf-8"))
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    frozen_path = ROOT / "reports/btc_preopen/btc_preopen_full_development_fit_20261016/frozen_bundle_manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if len(bundle.get("feature_order", [])) != 64:
        raise RuntimeError("The primary benchmark does not have exactly 64 frozen features")
    if bundle["feature_order"] != training["feature_order"]:
        raise RuntimeError("The bundle and training manifest feature orders differ")
    if int(training["training_rows"]) != 3_186_140:
        raise RuntimeError("The binary benchmark eligible-row count differs from its report")
    return bundle, training, tuning, frozen


def _load_cached_data(feature_order: list[str], training: dict, tuning: dict) -> dict:
    path = ROOT / CONFIG["dataset_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    required = [
        "Opened", "Open", "Close", TARGET_COL,
        DECISION_COL, DECISION_AVAILABLE_COL, TARGET_AVAILABLE_COL,
        "target_window_start_at",
        "target_window_end_at", "feature_candle_close_at", *feature_order,
    ]
    started = time.perf_counter()
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(required)))
    load_seconds = time.perf_counter() - started
    opened = pd.DatetimeIndex(pd.to_datetime(frame["Opened"], utc=True, errors="raise"))
    decision_at = pd.DatetimeIndex(pd.to_datetime(frame[DECISION_AVAILABLE_COL], utc=True, errors="raise"))
    available_at = pd.DatetimeIndex(pd.to_datetime(frame[TARGET_AVAILABLE_COL], utc=True, errors="coerce"))
    if opened.has_duplicates or not opened.is_monotonic_increasing:
        raise RuntimeError("The cached model dataset is not a unique, sorted minute series")
    if not np.array_equal(decision_at.asi8, (opened + pd.Timedelta(minutes=1)).asi8):
        raise RuntimeError("Cached nominal decision timestamps differ from Opened + 1 minute")
    if not np.array_equal(
        pd.DatetimeIndex(pd.to_datetime(frame["feature_candle_close_at"], utc=True, errors="raise")).asi8,
        decision_at.asi8,
    ):
        raise RuntimeError("Feature candle availability differs from the nominal decision")
    if not np.array_equal(
        pd.DatetimeIndex(pd.to_datetime(frame["target_window_start_at"], utc=True, errors="raise")).asi8,
        (opened + pd.Timedelta(minutes=2)).asi8,
    ):
        raise RuntimeError("Cached return target does not start at the contract's Opened + 2 minutes")
    if not np.array_equal(
        pd.DatetimeIndex(pd.to_datetime(frame["target_window_end_at"], utc=True, errors="raise")).asi8,
        (opened + pd.Timedelta(minutes=7)).asi8,
    ):
        raise RuntimeError("Cached return target does not end at the contract's Opened + 7 minutes")

    label = pd.to_numeric(frame[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    open_values = pd.to_numeric(frame["Open"], errors="coerce").to_numpy(dtype=np.float64)
    close_values = pd.to_numeric(frame["Close"], errors="coerce").to_numpy(dtype=np.float64)
    start_rows = opened.get_indexer(opened + pd.Timedelta(minutes=2))
    end_rows = opened.get_indexer(opened + pd.Timedelta(minutes=6))
    contiguous = (start_rows >= 0) & (end_rows >= 0)
    for offset in range(5):
        contiguous &= opened.get_indexer(opened + pd.Timedelta(minutes=2 + offset)) >= 0
    start_price = np.full(len(frame), np.nan, dtype=np.float64)
    end_price = np.full(len(frame), np.nan, dtype=np.float64)
    start_price[contiguous] = open_values[start_rows[contiguous]]
    end_price[contiguous] = close_values[end_rows[contiguous]]
    valid_return = contiguous & np.isfinite(start_price) & (start_price > 0.0) & np.isfinite(end_price) & np.isfinite(label)
    actual_return = np.full(len(frame), np.nan, dtype=np.float64)
    actual_return[valid_return] = end_price[valid_return] / start_price[valid_return] - 1.0
    if not np.array_equal(np.isfinite(label), np.isfinite(actual_return)):
        raise RuntimeError("The cached binary target validity mask differs from the contract price window")
    if not np.array_equal(label[valid_return], (actual_return[valid_return] >= 0.0).astype(np.float64)):
        raise RuntimeError("The binary target no longer resolves a zero return as UP")

    volatility_started = time.perf_counter()
    close = pd.to_numeric(frame["Close"], errors="coerce").to_numpy(dtype=np.float64)
    sigma = build_preopen_sigma_5m(opened, close)
    normalized = normalize_preopen_return(actual_return, sigma)
    volatility_seconds = time.perf_counter() - volatility_started
    sigma_available_at = opened + pd.Timedelta(minutes=1)
    if np.any(np.isfinite(sigma) & (sigma_available_at.asi8 > decision_at.asi8)):
        raise RuntimeError("The volatility normalization uses a close unavailable at decision time")

    fit_cutoff = pd.Timestamp(CONFIG["fit_label_available_through_utc"])
    fit_cutoff = fit_cutoff.tz_convert("UTC") if fit_cutoff.tzinfo else fit_cutoff.tz_localize("UTC")
    training_mask = np.isfinite(label) & available_at.notna() & (available_at <= fit_cutoff)
    fit_rows = np.flatnonzero(training_mask)
    expected_rows = int(training["training_rows"])
    if len(fit_rows) != expected_rows:
        raise RuntimeError(f"Cached binary training rows differ: {len(fit_rows)} != {expected_rows}")
    fit_decision = frame[DECISION_COL].to_numpy(dtype=np.bool_, copy=False)[fit_rows]
    fit_opened = opened[fit_rows]
    fit_decision_at = decision_at[fit_rows]
    fit_available_at = available_at[fit_rows]
    weights_full = experiment._preopen_sample_weights(
        frame[DECISION_COL].to_numpy(dtype=np.bool_, copy=False),
        float(training["decision_weight"]),
    )
    weight_hash = _array_sha256(weights_full[fit_rows], "<f4")
    expected_weight_hash = tuning.get("training_weight_sha256")
    if expected_weight_hash and weight_hash != expected_weight_hash:
        raise RuntimeError("The cached sample weights differ from the completed binary study")

    fold_count = int(CONFIG["model"]["cv_folds"])
    folds = experiment._purged_walk_forward_folds(
        model_tuning,
        fit_opened,
        fit_decision_at,
        fold_count,
        float(CONFIG["model"]["test_to_train_ratio"]),
        validation_mask=fit_decision,
    )
    fold_fingerprints = experiment._fold_index_fingerprints(folds)
    if fold_fingerprints != tuning.get("fold_indices"):
        raise RuntimeError("Regenerated chronological fold indices differ from the saved binary study")

    baseline_oof_path = ROOT / CONFIG["binary_oof_predictions_path"]
    baseline_oof = pd.read_parquet(baseline_oof_path)
    baseline_oof["Opened"] = pd.to_datetime(baseline_oof["Opened"], utc=True, errors="raise")
    baseline_oof["nominal_decision_at"] = pd.to_datetime(baseline_oof["nominal_decision_at"], utc=True, errors="raise")
    baseline_oof[TARGET_AVAILABLE_COL] = pd.to_datetime(baseline_oof[TARGET_AVAILABLE_COL], utc=True, errors="raise")
    if baseline_oof.Opened.duplicated().any() or len(baseline_oof) != sum(len(fold["valid_idx"]) for fold in folds):
        raise RuntimeError("Saved binary OOF predictions do not cover the frozen internal validation folds")
    for fold in folds:
        fold_id = f"main_model_cv_{fold['fold_id']}"
        group = baseline_oof.loc[baseline_oof.fold_id.eq(fold_id)].sort_values("Opened", kind="stable")
        expected_opened = fit_opened[np.asarray(fold["valid_idx"], dtype=np.int64)]
        if len(group) != len(expected_opened) or not np.array_equal(group.Opened.array.asi8, expected_opened.asi8):
            raise RuntimeError(f"Saved binary OOF fold {fold_id} differs from the regenerated split")

    feature_values = frame.loc[:, feature_order].to_numpy(dtype=np.float32, copy=True)
    feature_values[~np.isfinite(feature_values)] = np.nan
    y_binary = label[fit_rows].astype(np.int8, copy=False)
    weights = weights_full[fit_rows]
    return {
        "frame": frame,
        "opened": opened,
        "decision_at": decision_at,
        "available_at": available_at,
        "return": actual_return,
        "sigma": sigma,
        "z": normalized,
        "label": label,
        "weights_full": weights_full,
        "feature_values": feature_values,
        "fit_rows": fit_rows,
        "fit_opened": fit_opened,
        "fit_decision_at": fit_decision_at,
        "fit_available_at": fit_available_at,
        "fit_decision": fit_decision,
        "fit_y_binary": y_binary,
        "fit_weights": weights,
        "folds": folds,
        "fold_fingerprints": fold_fingerprints,
        "baseline_oof": baseline_oof,
        "load_seconds": load_seconds,
        "volatility_seconds": volatility_seconds,
        "weight_sha256": weight_hash,
        "return_recomputed_from_contract_prices": True,
        "return_valid_count": int(valid_return.sum()),
        "sigma_missing_count_all": int((~np.isfinite(sigma)).sum()),
        "sigma_zero_count_all": int((np.isfinite(sigma) & (sigma == 0.0)).sum()),
        "minute_gap_count": int(np.sum((opened[1:] - opened[:-1]) != pd.Timedelta(minutes=1))),
    }


def _folds_for_regression(data: dict) -> list[dict]:
    fit_rows = data["fit_rows"]
    fit_z = data["z"][fit_rows]
    eligible = np.isfinite(fit_z)
    full_to_regression = np.full(len(fit_rows), -1, dtype=np.int64)
    full_to_regression[eligible] = np.arange(int(eligible.sum()), dtype=np.int64)
    folds = []
    for source in data["folds"]:
        train_source = np.asarray(source["train_idx"], dtype=np.int64)
        valid_source = np.asarray(source["valid_idx"], dtype=np.int64)
        train_positions = full_to_regression[train_source]
        valid_positions = full_to_regression[valid_source]
        train_positions = train_positions[train_positions >= 0]
        valid_positions = valid_positions[valid_positions >= 0]
        if not len(train_positions) or not len(valid_positions):
            raise RuntimeError(f"Regression fold {source['fold_id']} has no eligible target rows")
        first_decision = data["fit_decision_at"][valid_source].min()
        latest_label = data["fit_available_at"][train_source].max()
        if latest_label > first_decision:
            raise RuntimeError(f"Regression fold {source['fold_id']} violates the label-availability purge")
        folds.append({
            "fold_id": int(source["fold_id"]),
            "train_idx": train_positions.astype(np.int32, copy=False),
            "valid_idx": valid_positions.astype(np.int32, copy=False),
            "first_validation_decision_at": first_decision,
            "last_validation_decision_at": data["fit_decision_at"][valid_source].max(),
            "latest_training_label_available_at": latest_label,
            "training_rows": int(len(train_positions)),
            "validation_rows": int(len(valid_positions)),
        })
    return folds


def _target_summary(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0}
    q = np.quantile(array, [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999])
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "quantiles": {"0.1%": float(q[0]), "1%": float(q[1]), "25%": float(q[2]), "50%": float(q[3]), "75%": float(q[4]), "99%": float(q[5]), "99.9%": float(q[6])},
        "zero_count": int(np.count_nonzero(array == 0.0)),
    }


def _make_regression_arrays(data: dict, feature_order: list[str]) -> dict:
    fit_rows = data["fit_rows"]
    fit_z = data["z"][fit_rows]
    valid = np.isfinite(fit_z)
    rows = fit_rows[valid]
    x = np.ascontiguousarray(data["feature_values"][rows], dtype=np.float32)
    y = np.ascontiguousarray(fit_z[valid], dtype=np.float64)
    weights = np.ascontiguousarray(data["weights_full"][rows], dtype=np.float32)
    full_to_regression = np.full(len(fit_rows), -1, dtype=np.int64)
    full_to_regression[valid] = np.arange(len(rows), dtype=np.int64)
    folds = _folds_for_regression(data)
    summary = {
        "binary_eligible_training_rows": int(len(fit_rows)),
        "regression_eligible_training_rows": int(len(rows)),
        "excluded_for_unavailable_return_or_sigma": int(len(fit_rows) - len(rows)),
        "actual_decision_training_rows": int(np.sum(data["fit_decision"])),
        "actual_decision_regression_rows": int(np.sum(data["fit_decision"] & valid)),
        "training_weight_sha256_binary_rows": data["weight_sha256"],
        "feature_dtype": str(x.dtype),
        "target_dtype": str(y.dtype),
        "feature_count": int(x.shape[1]),
        "feature_order": list(feature_order),
        "return_target_development": _target_summary(data["return"][fit_rows]),
        "sigma_5m_development": _target_summary(data["sigma"][fit_rows]),
        "normalized_target_z_development": _target_summary(data["z"][fit_rows]),
        "sigma_missing_training_rows": int(np.sum(~np.isfinite(data["sigma"][fit_rows]))),
        "sigma_zero_training_rows": int(np.sum(np.isfinite(data["sigma"][fit_rows]) & (data["sigma"][fit_rows] == 0.0))),
        "minute_timestamp_gaps_in_source": int(data["minute_gap_count"]),
        "return_target_column": RETURN_TARGET_COL,
        "return_target_recomputed_from_contract_prices": data["return_recomputed_from_contract_prices"],
        "return_contract_valid_rows": data["return_valid_count"],
        "volatility_available_at_decision": True,
        "normalized_target_clipped": False,
    }
    fold_manifest = []
    for fold in folds:
        fold_manifest.append({key: value for key, value in fold.items() if key not in {"train_idx", "valid_idx"}})
    summary["folds"] = fold_manifest
    return {"x": x, "y": y, "weights": weights, "folds": folds, "full_to_regression": full_to_regression, "rows": rows, "summary": summary}


def _profile_regressor(arrays: dict, feature_order: list[str]) -> dict:
    fold = arrays["folds"][-1]
    params = {**_base_lgb_params(len(feature_order)), **json.loads((ROOT / CONFIG["binary_tuning_result_path"]).read_text(encoding="utf-8"))["best_params"]}
    train = lgb.Dataset(
        arrays["x"][fold["train_idx"]], label=arrays["y"][fold["train_idx"]],
        weight=arrays["weights"][fold["train_idx"]], feature_name=feature_order,
        free_raw_data=True,
    )
    valid = lgb.Dataset(
        arrays["x"][fold["valid_idx"]], label=arrays["y"][fold["valid_idx"]],
        reference=train, feature_name=feature_order, free_raw_data=True,
    )
    started = time.perf_counter()
    model = lgb.train(
        params, train, num_boost_round=int(CONFIG["model"]["maximum_estimators"]),
        valid_sets=[valid], valid_names=["validation"], feval=_huber_eval,
        callbacks=[lgb.early_stopping(int(CONFIG["model"]["early_stopping_rounds"]), verbose=False), lgb.log_evaluation(0)],
    )
    elapsed = time.perf_counter() - started
    best_iteration = int(model.best_iteration)
    predictions = model.predict(arrays["x"][fold["valid_idx"]], num_threads=int(CONFIG["model"]["num_threads"]))
    loss = _huber_loss(arrays["y"][fold["valid_idx"]], predictions)
    del model, train, valid
    per_fold_worst_case = elapsed * int(CONFIG["model"]["maximum_estimators"]) / max(1, best_iteration)
    projected_60 = per_fold_worst_case * len(arrays["folds"]) * int(CONFIG["model"]["trial_limit"]) * 1.25
    estimate = {
        "recorded_before_tuning": True,
        "profile_kind": "one full chronological validation fold; seed parameters from the completed binary model; Huber objective and early stopping",
        "profile_fold_id": int(fold["fold_id"]),
        "profile_training_rows": int(fold["training_rows"]),
        "profile_validation_rows": int(fold["validation_rows"]),
        "profile_elapsed_seconds": elapsed,
        "profile_best_iteration": best_iteration,
        "profile_validation_huber": loss,
        "one_fold_500_tree_linear_projection_seconds": per_fold_worst_case,
        "60_trial_10_fold_projection_seconds_with_25pct_overhead": projected_60,
        "authorized_tuning_cap_seconds": float(CONFIG["model"]["time_limit_seconds"]),
        "estimated_tuning_minutes_capped": min(projected_60, float(CONFIG["model"]["time_limit_seconds"])) / 60.0,
        "source_rows": int(arrays["x"].shape[0]),
        "features": int(arrays["x"].shape[1]),
        "folds": len(arrays["folds"]),
        "trials": int(CONFIG["model"]["trial_limit"]),
        "resource_snapshot": _resource_snapshot(),
    }
    _write_json(OUTPUT_DIR / "cost_estimate.json", estimate)
    _write_status(status="running", stage="huber_tuning", completed=["target_smoke", "cache_and_fold_validation", "measured_runtime_estimate"], next_step="Run the one Huber Optuna study, capped at 60 trials and 2,700 seconds.")
    return estimate


def _suggest_lgb_params(trial) -> dict:
    return model_tuning.suggest_lgbm_hyperparams(trial, CONFIG["model"]["search_space"])


def _run_tuning(arrays: dict, feature_order: list[str], identity: dict) -> dict:
    output = OUTPUT_DIR / "tuning"
    output.mkdir(parents=True, exist_ok=True)
    database = output / "study.sqlite3"
    storage = f"sqlite:///{database.resolve().as_posix()}"
    study = optuna.create_study(
        study_name=CONFIG["model"]["study_name"], storage=storage,
        direction="minimize", sampler=optuna.samplers.TPESampler(
            seed=int(CONFIG["model"]["seed"]), n_startup_trials=5,
        ), pruner=optuna.pruners.NopPruner(), load_if_exists=True,
    )
    expected_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    existing_identity = study.user_attrs.get("input_identity_sha256")
    if existing_identity and existing_identity != expected_identity:
        raise RuntimeError("The resumable regression study belongs to different frozen inputs")
    if not existing_identity:
        study.set_user_attr("input_identity_sha256", expected_identity)
        source_training = json.loads((ROOT / CONFIG["binary_training_manifest_path"]).read_text(encoding="utf-8"))
        study.enqueue_trial({key: source_training["params"][key] for key in CONFIG["model"]["search_space"]}, user_attrs={"seed_source": "latest completed 64-feature binary parameters; scores not reused"})
    stale_trials = [trial.number for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING]
    for trial_number in stale_trials:
        study.tell(trial_number, state=optuna.trial.TrialState.FAIL)

    deadline = time.perf_counter() + float(CONFIG["model"]["time_limit_seconds"])
    elapsed_start = time.perf_counter()
    base_params = _base_lgb_params(len(feature_order))
    training_set = lgb.Dataset(
        arrays["x"], label=arrays["y"], weight=arrays["weights"],
        feature_name=feature_order, free_raw_data=False,
    )
    fold_indices = [(fold["train_idx"], fold["valid_idx"]) for fold in arrays["folds"]]
    target_trials = int(CONFIG["model"]["trial_limit"])
    checkpoints = set(int(value) for value in CONFIG["model"]["checkpoint_trials"])
    expired = False

    def objective(trial):
        params = {**base_params, **_suggest_lgb_params(trial)}
        callbacks = [
            ObjectiveAlignedCVCallback(
                HUBER_METRIC_NAME, higher_is_better=False,
                std_penalty=float(CONFIG["model"]["cv_std_penalty"]),
                stopping_rounds=int(CONFIG["model"]["early_stopping_rounds"]),
                trial=trial, report_interval=10,
            ),
            _DeadlineCallback(deadline),
            lgb.log_evaluation(0),
        ]
        result = lgb.cv(
            params=params, train_set=training_set, folds=fold_indices,
            num_boost_round=int(CONFIG["model"]["maximum_estimators"]),
            feval=_huber_eval, callbacks=callbacks, stratified=False, shuffle=False,
            return_cvbooster=False, seed=int(CONFIG["model"]["seed"]),
        )
        summary = cv_objective_series(
            result, HUBER_METRIC_NAME, higher_is_better=False,
            std_penalty=float(CONFIG["model"]["cv_std_penalty"]),
        )
        best_index = int(np.argmin(summary["objective"]))
        best_value = float(summary["objective"][best_index])
        trial.set_user_attr("best_iteration", best_index + 1)
        trial.set_user_attr("cv_huber_mean", float(summary["mean"][best_index]))
        trial.set_user_attr("cv_huber_std_population", float(summary["std"][best_index]))
        trial.set_user_attr("cv_objective", best_value)
        return best_value

    if not study.trials:
        raise RuntimeError("The regression Optuna study was not initialized")
    try:
        while len(study.trials) < target_trials:
            if time.perf_counter() >= deadline:
                expired = True
                break
            study.optimize(objective, n_trials=1, catch=(lgb.basic.LightGBMError,))
            evaluated = len(study.trials)
            states = {state.name: sum(trial.state == state for trial in study.trials) for state in optuna.trial.TrialState}
            best = study.best_trial if any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials) else None
            progress = {
                "evaluated_trials": evaluated,
                "trial_limit": target_trials,
                "trial_counts": states,
                "elapsed_seconds": time.perf_counter() - elapsed_start,
                "time_limit_seconds": float(CONFIG["model"]["time_limit_seconds"]),
                "best_cv_criterion": float(best.value) if best is not None else None,
                "best_iteration": best.user_attrs.get("best_iteration") if best is not None else None,
                "best_params": best.params if best is not None else None,
            }
            _write_json(output / "trial_progress.json", progress)
            if evaluated in checkpoints:
                checkpoint = {**progress, "checkpoint_trial_count": evaluated, "study_name": study.study_name, "input_identity_sha256": expected_identity}
                _write_json(output / f"checkpoint_{evaluated}.json", checkpoint)
                print(f"[return-regression] checkpoint={evaluated} best={progress['best_cv_criterion']}", flush=True)
                _write_status(status="running", stage="huber_tuning", completed=["target_smoke", "cache_and_fold_validation", "measured_runtime_estimate", f"tuning_checkpoint_{evaluated}"], next_step="Continue the same Optuna study to the next registered checkpoint or the 2,700-second cap.")
            if time.perf_counter() >= deadline:
                expired = True
                break
    except TuningBudgetExpired:
        expired = True

    elapsed = time.perf_counter() - elapsed_start
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None]
    if not complete:
        raise RuntimeError("The Huber study produced no completed trial")
    best = study.best_trial
    state_counts = {state.name: sum(trial.state == state for trial in study.trials) for state in optuna.trial.TrialState}
    failure_notes = [f"trial {number}: a stale RUNNING trial from an interrupted earlier invocation was marked failed" for number in stale_trials]
    deadline_trial = next((trial.number for trial in reversed(study.trials) if trial.state == optuna.trial.TrialState.FAIL), None)
    if expired and deadline_trial is not None:
        failure_notes.append(f"trial {deadline_trial}: wall-clock deadline expired during a CV fold")
    selected = {
        "status": "time_budget_reached" if expired else "trial_budget_reached",
        "study_name": study.study_name,
        "study_database_path": database.resolve().relative_to(ROOT).as_posix(),
        "study_database_sha256": _sha256(database),
        "trials_evaluated": len(study.trials),
        "completed_trials": len(complete),
        "failed_trials": state_counts["FAIL"],
        "pruned_trials": sum(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials),
        "trial_state_counts": state_counts,
        "stale_trials_marked_failed_at_resume": stale_trials,
        "failure_notes": failure_notes,
        "elapsed_seconds": elapsed,
        "time_limit_seconds": float(CONFIG["model"]["time_limit_seconds"]),
        "trial_limit": target_trials,
        "objective": "mean unweighted decision-row Huber(alpha=0.9) + 0.5 * population standard deviation across the same 10 chronological folds; early stopping and ranking use this same objective",
        "best_trial_number": int(best.number),
        "best_cv_criterion": float(best.value),
        "best_cv_huber_mean": float(best.user_attrs["cv_huber_mean"]),
        "best_cv_huber_std_population": float(best.user_attrs["cv_huber_std_population"]),
        "best_iteration": int(best.user_attrs["best_iteration"]),
        "best_params": dict(best.params),
        "input_identity_sha256": expected_identity,
        "no_optuna_pruning": True,
    }
    _write_json(OUTPUT_DIR / "tuning_result.json", selected)
    if expired:
        _write_json(output / "budget_stop.json", {"elapsed_seconds": elapsed, "trial_count": len(study.trials), "reason": "2,700-second wall-clock tuning limit"})
    del training_set
    return selected


def _fit_regression_model(arrays: dict, feature_order: list[str], params: dict, iterations: int) -> lgb.Booster:
    training_set = lgb.Dataset(
        arrays["x"], label=arrays["y"], weight=arrays["weights"],
        feature_name=feature_order, free_raw_data=True,
    )
    model = lgb.train(
        {**_base_lgb_params(len(feature_order)), **params}, training_set,
        num_boost_round=int(iterations),
    )
    del training_set
    return model


def _generate_development_oof(data: dict, arrays: dict, feature_order: list[str], selection: dict) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    fit_rows = data["fit_rows"]
    records = []
    fold_audit = []
    for fold in arrays["folds"]:
        model = _fit_fold_model(arrays, feature_order, selection["best_params"], selection["best_iteration"], fold)
        prediction = model.predict(arrays["x"][fold["valid_idx"]], num_threads=int(CONFIG["model"]["num_threads"]))
        source_fold = data["folds"][int(fold["fold_id"])]
        original_valid = np.asarray(source_fold["valid_idx"], dtype=np.int64)
        regression_valid = arrays["full_to_regression"][original_valid]
        regression_valid = regression_valid[regression_valid >= 0]
        global_rows = fit_rows[original_valid]
        valid_z = data["z"][global_rows]
        valid_return = data["return"][global_rows]
        valid_sigma = data["sigma"][global_rows]
        valid_label = data["label"][global_rows].astype(np.int8)
        # The mapped regression rows and the retained original validation rows must align.
        keep = arrays["full_to_regression"][original_valid] >= 0
        if len(prediction) != int(keep.sum()) or len(regression_valid) != len(prediction):
            raise RuntimeError(f"Regression OOF fold {fold['fold_id']} has a row-alignment mismatch")
        prediction_times = data["fit_decision_at"][original_valid][keep]
        target_available = data["fit_available_at"][original_valid][keep]
        fold_frame = pd.DataFrame({
            "Opened": data["fit_opened"][original_valid][keep],
            "nominal_decision_at": prediction_times,
            "market_start_utc": prediction_times + pd.Timedelta(minutes=1),
            "target_label_available_at": target_available,
            "target_preopen_5m_up_binance_index_proxy": valid_label[keep],
            "return_target": valid_return[keep],
            "sigma_5m": valid_sigma[keep],
            "z_actual": valid_z[keep],
            "z_predicted": np.asarray(prediction, dtype=np.float64),
            "fold_id": f"main_model_cv_{fold['fold_id']}",
            "prediction_available_at_utc": prediction_times + pd.Timedelta(seconds=1),
            "prediction_provenance": "development_selected_configuration_chronological_oof_conditional",
            "fit_label_available_through_utc": fold["latest_training_label_available_at"],
        })
        records.append(fold_frame)
        fold_audit.append({
            "fold_id": str(fold["fold_id"]),
            "training_rows": int(fold["training_rows"]),
            "validation_rows": int(len(fold_frame)),
            "validation_start_utc": prediction_times[0].isoformat(),
            "validation_end_utc": prediction_times[-1].isoformat(),
            "latest_training_label_available_at_utc": fold["latest_training_label_available_at"].isoformat(),
            "best_iteration": int(selection["best_iteration"]),
            "huber": _huber_loss(fold_frame.z_actual, fold_frame.z_predicted),
            "mae": float(np.mean(np.abs(fold_frame.z_actual - fold_frame.z_predicted))),
        })
        del model
        print(f"[return-regression] OOF fold={fold['fold_id']} rows={len(fold_frame):,}", flush=True)
    oof = pd.concat(records, ignore_index=True).sort_values("nominal_decision_at", kind="stable").reset_index(drop=True)
    baseline = data["baseline_oof"].loc[:, [
        "Opened", "p_model_raw", TARGET_COL, TARGET_AVAILABLE_COL, "fold_id",
    ]].rename(columns={TARGET_COL: "binary_target", TARGET_AVAILABLE_COL: "binary_label_available_at"})
    joined = oof.merge(baseline, on=["Opened", "fold_id"], how="left", validate="one_to_one")
    if joined.p_model_raw.isna().any() or not np.array_equal(joined.target_preopen_5m_up_binance_index_proxy.to_numpy(), joined.binary_target.to_numpy()):
        raise RuntimeError("Regression OOF rows do not match the saved binary OOF predictions")
    joined = joined.drop(columns=["binary_target"])
    path = OUTPUT_DIR / "development_oof_predictions.parquet"
    temporary = path.with_suffix(path.suffix + ".tmp")
    joined.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)
    return joined, {
        "prediction_path": path.resolve().relative_to(ROOT).as_posix(),
        "prediction_sha256": _sha256(path),
        "rows": int(len(joined)),
        "fold_count": len(fold_audit),
        "folds": fold_audit,
        "elapsed_seconds": time.perf_counter() - started,
        "holdout_rows_included": False,
        "conditional_on_development_selection": True,
    }


def _fit_fold_model(arrays: dict, feature_order: list[str], params: dict, iterations: int, fold: dict) -> lgb.Booster:
    train_indices = np.asarray(fold["train_idx"], dtype=np.int64)
    train = lgb.Dataset(
        arrays["x"][train_indices], label=arrays["y"][train_indices],
        weight=arrays["weights"][train_indices], feature_name=feature_order,
        free_raw_data=True,
    )
    model = lgb.train(
        {**_base_lgb_params(len(feature_order)), **params}, train,
        num_boost_round=int(iterations),
    )
    del train
    return model


def _logit(probability) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return np.log(p / (1.0 - p))


def _fit_logistic(values, labels) -> dict:
    x = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels, dtype=np.int8)
    if len(y) < MIN_CALIBRATION_ROWS or np.unique(y).size != 2:
        raise RuntimeError(f"Calibrator requires at least {MIN_CALIBRATION_ROWS} earlier rows from both classes")
    model = LogisticRegression(
        C=float(CONFIG["calibration"]["C"]), solver=CONFIG["calibration"]["solver"],
        random_state=int(CONFIG["calibration"]["seed"]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)
    return {
        "intercept": float(model.intercept_[0]),
        "coefficient": float(model.coef_[0, 0]),
        "training_rows": int(len(y)),
        "training_up_rate": float(y.mean()),
        "warnings": [str(item.message) for item in caught],
    }


def _apply_logistic(values, fit: dict) -> np.ndarray:
    return np.clip(
        expit(float(fit["intercept"]) + float(fit["coefficient"]) * np.asarray(values, dtype=np.float64)),
        PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON,
    )


def _causal_development_probabilities(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    result = oof.copy()
    result["p_return_logistic_causal"] = np.nan
    result["p_binary_platt_causal"] = np.nan
    result["p_constant_prior_causal"] = np.nan
    calibrators = []
    for fold_id, group in result.groupby("fold_id", sort=True):
        first_decision = pd.Timestamp(group.nominal_decision_at.min())
        earlier = result.loc[
            result.nominal_decision_at.lt(first_decision)
            & result.target_label_available_at.le(first_decision)
        ].copy()
        if len(earlier) < MIN_CALIBRATION_ROWS or earlier.target_preopen_5m_up_binance_index_proxy.nunique() != 2:
            calibrators.append({"fold_id": str(fold_id), "status": "warmup_excluded", "prior_rows": int(len(earlier))})
            continue
        y_prior = earlier.target_preopen_5m_up_binance_index_proxy.to_numpy(dtype=np.int8)
        regression = _fit_logistic(earlier.z_predicted, y_prior)
        binary = _fit_logistic(_logit(earlier.p_model_raw), y_prior)
        prior_probability = float(y_prior.mean())
        rows = result.index[result.fold_id.eq(fold_id)]
        result.loc[rows, "p_return_logistic_causal"] = _apply_logistic(result.loc[rows, "z_predicted"], regression)
        result.loc[rows, "p_binary_platt_causal"] = _apply_logistic(_logit(result.loc[rows, "p_model_raw"]), binary)
        result.loc[rows, "p_constant_prior_causal"] = np.clip(prior_probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
        calibrators.append({
            "fold_id": str(fold_id), "status": "fitted_on_earlier_oof_only",
            "first_validation_decision_at": first_decision.isoformat(),
            "latest_calibration_label_available_at": earlier.target_label_available_at.max().isoformat(),
            "prior_rows": int(len(earlier)), "constant_probability": prior_probability,
            "return_score_logistic": regression, "binary_raw_logit_platt": binary,
        })
    scored = result.loc[result.p_return_logistic_causal.notna()].copy()
    final_fit_rows = result.loc[result.target_label_available_at.le(pd.Timestamp(CONFIG["fit_label_available_through_utc"]))].copy()
    if len(final_fit_rows) < MIN_CALIBRATION_ROWS:
        raise RuntimeError("Too few development OOF rows are available for the frozen final calibrator")
    labels = final_fit_rows.target_preopen_5m_up_binance_index_proxy.to_numpy(dtype=np.int8)
    return_fit = _fit_logistic(final_fit_rows.z_predicted, labels)
    binary_fit = _fit_logistic(_logit(final_fit_rows.p_model_raw), labels)
    final_calibrators = {
        "method": CONFIG["calibration"]["method"],
        "return_score_logistic": return_fit,
        "binary_raw_logit_platt_comparator": binary_fit,
        "training_rows": int(len(final_fit_rows)),
        "training_up_rate": float(labels.mean()),
        "training_start_decision_at": final_fit_rows.nominal_decision_at.min().isoformat(),
        "training_end_decision_at": final_fit_rows.nominal_decision_at.max().isoformat(),
        "latest_label_available_at": final_fit_rows.target_label_available_at.max().isoformat(),
        "holdout_rows_included": False,
        "causal_development_fold_calibrators": calibrators,
        "development_rows_scored_after_calibration_warmup": int(len(scored)),
        "development_warmup_excluded_rows": int(len(result) - len(scored)),
    }
    final_path = OUTPUT_DIR / "final_calibrator.json"
    _write_json(final_path, final_calibrators)
    return scored, final_calibrators


def _paired_block_interval(y, p_candidate, p_baseline, *, seed: int, block_size: int, replications: int) -> dict:
    y = np.asarray(y, dtype=np.int8)
    p_candidate = np.clip(np.asarray(p_candidate, dtype=np.float64), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    p_baseline = np.clip(np.asarray(p_baseline, dtype=np.float64), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    candidate_loss = -(y * np.log(p_candidate) + (1 - y) * np.log1p(-p_candidate))
    baseline_loss = -(y * np.log(p_baseline) + (1 - y) * np.log1p(-p_baseline))
    difference = candidate_loss - baseline_loss
    n = len(difference)
    if not n:
        return {"n": 0, "method": "paired circular moving-block bootstrap", "ci95": None}
    blocks = int(np.ceil(n / block_size))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(replications, blocks), dtype=np.int64)
    totals = np.zeros(replications, dtype=np.float64)
    remaining = n
    offsets = np.arange(block_size, dtype=np.int64)
    for block_index in range(blocks):
        size = min(block_size, remaining)
        indices = (starts[:, block_index, None] + offsets[:size][None, :]) % n
        totals += difference[indices].sum(axis=1)
        remaining -= size
    draws = totals / n
    return {
        "n": int(n),
        "method": "paired circular moving-block bootstrap",
        "block_days": int(CONFIG["uncertainty"]["block_days"]),
        "block_size_decisions": int(block_size),
        "replications": int(replications),
        "seed": int(seed),
        "candidate_minus_binary_log_loss": float(difference.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "negative_favors_return_regression": True,
    }


def _official_labels(path: Path, starts: pd.Series, cutoff: pd.Timestamp) -> pd.DataFrame:
    markets = pd.read_parquet(path, columns=["market_start_utc", "resolved_at_utc", "target_polymarket_up"])
    markets["market_start_utc"] = pd.to_datetime(markets.market_start_utc, utc=True, errors="raise")
    markets["resolved_at_utc"] = pd.to_datetime(markets.resolved_at_utc, utc=True, errors="coerce")
    start = markets.market_start_utc
    resolved = markets.resolved_at_utc
    release = resolved.where(resolved.ge(start + pd.Timedelta(minutes=5)), start + pd.Timedelta(minutes=5)) + pd.Timedelta(seconds=60)
    markets["official_outcome_available_by_cutoff"] = (
        pd.to_numeric(markets.target_polymarket_up, errors="coerce").isin((0.0, 1.0))
        & resolved.notna() & release.le(cutoff)
    )
    selected = markets.loc[markets.official_outcome_available_by_cutoff, ["market_start_utc", "target_polymarket_up", "resolved_at_utc"]]
    return starts.to_frame("market_start_utc").merge(selected, on="market_start_utc", how="left", validate="one_to_one")


def _score_periods(dev: pd.DataFrame, official_markets_path: Path) -> dict:
    metric_columns = {
        "return_regression_causal_logistic": "p_return_logistic_causal",
        "binary_causal_platt": "p_binary_platt_causal",
        "binary_raw": "p_model_raw",
        "constant_prior": "p_constant_prior_causal",
    }
    all_metrics = {}
    for target_name, target_column, cutoff in (
        ("proxy", "target_preopen_5m_up_binance_index_proxy", pd.Timestamp(CONFIG["development_cutoff_exclusive_utc"])),
        ("official", None, pd.Timestamp(CONFIG["development_cutoff_exclusive_utc"])),
    ):
        frame = dev.copy()
        if target_name == "official":
            outcomes = _official_labels(official_markets_path, frame.market_start_utc, cutoff)
            frame = frame.merge(outcomes.drop(columns=["resolved_at_utc"], errors="ignore"), on="market_start_utc", how="left", validate="one_to_one")
            target_column = "target_polymarket_up"
            frame = frame.loc[pd.to_numeric(frame[target_column], errors="coerce").isin((0.0, 1.0))].copy()
        result = {
            "rows": int(len(frame)),
            "start_market_utc": frame.market_start_utc.min().isoformat() if len(frame) else None,
            "end_market_utc": frame.market_start_utc.max().isoformat() if len(frame) else None,
            "models": {},
            "folds": {},
        }
        labels = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=np.int8)
        for name, column in metric_columns.items():
            usable = frame[column].notna().to_numpy()
            if usable.any():
                result["models"][name] = _metric_score(labels[usable], frame.loc[usable, column])
            else:
                result["models"][name] = {"n": 0}
            fold_rows = {}
            for fold_id, group in frame.groupby("fold_id", sort=True):
                use = group[column].notna().to_numpy()
                if use.any():
                    y_fold = pd.to_numeric(group.loc[use, target_column], errors="coerce").to_numpy(dtype=np.int8)
                    fold_rows[str(fold_id)] = _metric_score(y_fold, group.loc[use, column])
            result["folds"][name] = fold_rows
        result["paired_log_loss_return_minus_binary_causal"] = _paired_block_interval(
            labels[
                frame.p_return_logistic_causal.notna().to_numpy()
                & frame.p_binary_platt_causal.notna().to_numpy()
            ],
            frame.loc[frame.p_return_logistic_causal.notna() & frame.p_binary_platt_causal.notna(), "p_return_logistic_causal"],
            frame.loc[frame.p_return_logistic_causal.notna() & frame.p_binary_platt_causal.notna(), "p_binary_platt_causal"],
            seed=int(CONFIG["uncertainty"]["seed"]),
            block_size=int(CONFIG["uncertainty"]["block_days"]) * 288,
            replications=int(CONFIG["uncertainty"]["replications"]),
        )
        all_metrics[target_name] = result
    return all_metrics


def _regression_diagnostics(oof: pd.DataFrame) -> dict:
    y = oof.z_actual.to_numpy(dtype=np.float64)
    p = oof.z_predicted.to_numpy(dtype=np.float64)
    r = oof.return_target.to_numpy(dtype=np.float64)
    order = np.argsort(p, kind="stable")
    deciles = []
    for index, rows in enumerate(np.array_split(order, 10), start=1):
        if not len(rows):
            continue
        deciles.append({
            "predicted_z_decile": index,
            "n": int(len(rows)),
            "mean_predicted_z": float(np.mean(p[rows])),
            "mean_actual_z": float(np.mean(y[rows])),
            "mean_actual_return": float(np.mean(r[rows])),
            "up_frequency": float(np.mean(oof.target_preopen_5m_up_binance_index_proxy.to_numpy(dtype=np.int8)[rows])),
        })
    return {
        "rows": int(len(oof)),
        "huber_alpha_0_9": _huber_loss(y, p),
        "mae": float(np.mean(np.abs(y - p))),
        "predicted_z_mean": float(p.mean()),
        "predicted_z_std_population": float(p.std(ddof=0)),
        "predicted_z": _target_summary(p),
        "actual_z_predicted_z_pearson": float(pd.Series(y).corr(pd.Series(p), method="pearson")),
        "actual_z_predicted_z_spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman")),
        "actual_return_predicted_z_pearson": float(pd.Series(r).corr(pd.Series(p), method="pearson")),
        "predicted_z_deciles": deciles,
    }


def _write_development_results(oof: pd.DataFrame, scored: pd.DataFrame, calibration: dict, data: dict, arrays: dict, tuning: dict) -> dict:
    metrics = _score_periods(scored, ROOT / CONFIG["official_markets_path"])
    regression = _regression_diagnostics(oof)
    regression["fold_metrics"] = {
        str(fold_id): {
            "n": int(len(group)),
            "huber_alpha_0_9": _huber_loss(group.z_actual, group.z_predicted),
            "mae": float(np.mean(np.abs(group.z_actual - group.z_predicted))),
        }
        for fold_id, group in oof.groupby("fold_id", sort=True)
    }
    payload = {
        "status": "development_scored_before_holdout_freeze",
        "calibration": calibration,
        "probability_quality": metrics,
        "regression_quality": regression,
        "tuning": tuning,
        "target_and_dataset": arrays["summary"],
        "oof_prediction_sha256": _sha256(ROOT / "data/analysis/polymarket/BTC/btc_preopen_return_regression_20261017/development_oof_predictions.parquet"),
        "holdout_rows_used": False,
    }
    path = OUTPUT_DIR / "development_results.json"
    _write_json(path, payload)
    return payload


def _inference_benchmark(model_path: Path, x: np.ndarray, *, model_kind: str, calibrator: dict | None = None) -> dict:
    if model_kind == "binary":
        model = lgb.Booster(model_file=str(model_path))
    else:
        model = lgb.Booster(model_file=str(model_path))
    # Warm once, then time repeated model-only predictions on the same holdout matrix.
    raw = model.predict(x, num_threads=int(CONFIG["model"]["num_threads"]))
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        raw = model.predict(x, num_threads=int(CONFIG["model"]["num_threads"]))
        timings.append(time.perf_counter() - started)
    result = {
        "rows": int(len(x)),
        "feature_count": int(x.shape[1]),
        "warm_prediction_median_seconds": float(np.median(timings)),
        "warm_prediction_p95_seconds": float(np.quantile(timings, 0.95)),
        "warm_prediction_microseconds_per_row": float(np.median(timings) * 1e6 / max(1, len(x))),
        "measured_repeats": len(timings),
    }
    if calibrator is not None:
        started = time.perf_counter()
        probabilities = _apply_logistic(raw, calibrator)
        result["calibration_seconds"] = time.perf_counter() - started
        result["prediction_finite"] = bool(np.isfinite(probabilities).all())
    del model
    return result


def _holdout_predictions(data: dict, final_model: lgb.Booster, final_calibrator: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = data["frame"]
    decision = frame[DECISION_COL].to_numpy(dtype=np.bool_, copy=False)
    market_start = pd.to_datetime(frame["target_window_start_at"], utc=True, errors="raise")
    first = pd.Timestamp(CONFIG["holdout_first_market_start_utc"])
    last = pd.Timestamp(CONFIG["holdout_last_market_start_utc"])
    rows = np.flatnonzero(decision & market_start.ge(first).to_numpy() & market_start.le(last).to_numpy())
    if len(rows) != 28_224:
        raise RuntimeError(f"The frozen holdout prediction row count changed: {len(rows)}")
    x_holdout = np.ascontiguousarray(data["feature_values"][rows], dtype=np.float32)
    predicted_z = np.asarray(final_model.predict(x_holdout, num_threads=int(CONFIG["model"]["num_threads"])), dtype=np.float64)
    probability = _apply_logistic(predicted_z, final_calibrator)
    decision_at = data["decision_at"][rows]
    prediction_at = decision_at + pd.Timedelta(seconds=1)
    market_starts = market_start.iloc[rows].reset_index(drop=True)
    if not np.array_equal(prediction_at.asi8, (market_starts + pd.Timedelta(seconds=-59)).array.asi8):
        raise RuntimeError("Holdout predictions do not meet the frozen T-59 readiness timestamp")
    output = pd.DataFrame({
        "Opened": data["opened"][rows],
        "nominal_decision_at": decision_at,
        "market_start_utc": market_starts,
        TARGET_COL: data["label"][rows].astype(np.int8),
        "prediction_available_at_utc": prediction_at,
        "z_predicted": predicted_z,
        "p_model_raw": probability,
        "p_model_platt": probability,
        "fold_id": "evaluated_holdout",
        "prediction_provenance": "final_regressor_fit_on_full_permitted_development; development_oof_logistic_calibrator",
    })
    return output, x_holdout, predicted_z


def _holdout_quality(candidate: pd.DataFrame, baseline: pd.DataFrame, a_frozen: pd.DataFrame, data: dict) -> dict:
    start = pd.Timestamp(CONFIG["holdout_first_market_start_utc"])
    cutoff = pd.Timestamp(CONFIG["holdout_cutoff_exclusive_utc"])
    all_rows = candidate.merge(
        baseline[["market_start_utc", "p_model_raw", "p_model_platt"]].rename(columns={"p_model_raw": "binary_raw", "p_model_platt": "binary_platt"}),
        on="market_start_utc", how="inner", validate="one_to_one",
    ).merge(
        a_frozen[["market_start_utc", "p_model_raw", "p_model_platt"]].rename(columns={"p_model_raw": "a_frozen_raw", "p_model_platt": "a_frozen_platt"}),
        on="market_start_utc", how="inner", validate="one_to_one",
    )
    frame = data["frame"]
    decision = frame[DECISION_COL].to_numpy(dtype=np.bool_, copy=False)
    starts = pd.to_datetime(frame["target_window_start_at"], utc=True, errors="raise")
    proxy = pd.DataFrame({
        "market_start_utc": starts[decision],
        "target_proxy": pd.to_numeric(frame.loc[decision, TARGET_COL], errors="coerce").to_numpy(dtype=np.float64),
    })
    all_rows = all_rows.merge(proxy, on="market_start_utc", how="left", validate="one_to_one")
    official = _official_labels(ROOT / CONFIG["official_markets_path"], all_rows.market_start_utc, cutoff)
    all_rows = all_rows.merge(official.drop(columns=["resolved_at_utc"], errors="ignore"), on="market_start_utc", how="left", validate="one_to_one")
    models = {
        "return_regression": ("p_model_platt", None),
        "binary": ("binary_raw", "binary_platt"),
        "A_frozen": ("a_frozen_raw", "a_frozen_platt"),
    }
    results = {"holdout_period": {"start_inclusive_utc": start, "cutoff_exclusive_utc": cutoff}, "common_market_rows": int(len(all_rows)), "models": {}}
    for target_name, target_column in (("proxy", "target_proxy"), ("official", "target_polymarket_up")):
        scored = all_rows.loc[pd.to_numeric(all_rows[target_column], errors="coerce").isin((0.0, 1.0))].copy()
        target_result = {"rows": int(len(scored)), "positive_rate": float(scored[target_column].mean()) if len(scored) else None, "models": {}}
        for name, (raw_column, calibrated_column) in models.items():
            if calibrated_column is None:
                target_result["models"][name] = {"direction_probability": _metric_score(scored[target_column], scored[raw_column])}
            else:
                target_result["models"][name] = {
                    "raw": _metric_score(scored[target_column], scored[raw_column]),
                    "calibrated": _metric_score(scored[target_column], scored[calibrated_column]),
                }
        results["models"][target_name] = target_result
    results["official_outcomes_unavailable_by_cutoff"] = int(len(all_rows) - pd.to_numeric(all_rows.target_polymarket_up, errors="coerce").isin((0.0, 1.0)).sum())
    results["predictions_match_binary_common_sample"] = bool(len(all_rows) == 28_224)
    results["historically_exposed"] = True
    results["selection_from_holdout"] = False
    return results


def _holdout_economics(predictions_by_model: dict[str, pd.DataFrame]) -> dict:
    fee_correction._configure_output_paths()
    correction_manifest_path = fee_correction.CORRECTION_DIR / "research_manifest.json"
    correction_manifest = research._read_json(correction_manifest_path)
    t59, t60, book_record = research._load_enriched_book_snapshots(correction_manifest)
    expected_snapshot_hashes = {
        "t60_coverage_sha256": "38259fadf50c28e455f5d4523610cbac827886e35922076dbba5cb33bd55e0b1",
        "t59_coverage_sha256": "abee0fdaf31eaf1455b4343e985bf9d5ee38a84dd8eb7e06adf1e08fc0c380d0",
    }
    for key, expected in expected_snapshot_hashes.items():
        if book_record.get(key) != expected:
            raise RuntimeError(f"Frozen {key} differs from the 2026-10-16 experiment")
    t59_by_id = research._snapshot_map(t59)
    t60_by_id = research._snapshot_map(t60)
    results = {
        "policy": "A_BTC_calibrated_ev_1.0000%",
        "minimum_net_ev_fraction": float(CONFIG["economics"]["minimum_net_ev_fraction"]),
        "snapshot_hashes": expected_snapshot_hashes,
        "archived_book_replay_performed": False,
        "models": {},
    }
    for name, predictions in predictions_by_model.items():
        calendar, _ = research._build_market_calendar(predictions)
        calendar, _ = replay._market_statuses(calendar, t59, t60)
        market_by_id = research._snapshot_map(calendar.loc[calendar.condition_id.notna()])
        t59_evals = diagnostic._t59_execution_evaluations(
            market_by_id, t59_by_id, float(CONFIG["economics"]["independent_order_usd"]),
        )
        decisions = diagnostic._independent_decisions(
            calendar, market_by_id, t59_evals, t60_by_id,
            (float(CONFIG["economics"]["minimum_net_ev_fraction"]),),
        )
        start = pd.Timestamp(CONFIG["holdout_first_market_start_utc"])
        cutoff = pd.Timestamp(CONFIG["holdout_cutoff_exclusive_utc"])
        independent = diagnostic._aggregate_signals(
            decisions, "holdout", float(CONFIG["economics"]["minimum_net_ev_fraction"]),
            bootstrap_start=start, bootstrap_end=cutoff,
        )
        holdout_rows, period_summary = research._asof_market_rows(calendar, start, cutoff)
        portfolio, trades, portfolio_decisions, curve = diagnostic._run_holdout_portfolio(
            float(CONFIG["economics"]["portfolio_starting_cash_usd"]),
            holdout_rows, t59_by_id, market_by_id, t60_by_id,
            float(CONFIG["economics"]["minimum_net_ev_fraction"]),
        )
        skip_reasons = {}
        for item in portfolio_decisions:
            reason = item.get("skip_reason")
            if reason:
                skip_reasons[str(reason)] = skip_reasons.get(str(reason), 0) + 1
        output = OUTPUT_DIR / "economics" / name
        output.mkdir(parents=True, exist_ok=True)
        holdout_decisions = decisions.loc[decisions.period.eq("holdout")].copy()
        holdout_decisions.to_parquet(output / "independent_order_decisions.parquet", index=False, compression="zstd")
        if trades:
            pd.DataFrame(trades).to_parquet(output / "portfolio_trades.parquet", index=False, compression="zstd")
        _write_json(output / "portfolio_summary.json", portfolio)
        results["models"][name] = {
            "independent_fixed_5_usd": independent,
            "independent_decision_count": int(len(holdout_decisions)),
            "independent_rejection_reasons": independent.get("rejection_reasons", {}),
            "no_credit_5pct_free_cash_portfolio": portfolio,
            "portfolio_skip_reasons": skip_reasons,
            "market_period": period_summary,
            "independent_decisions_path": (output / "independent_order_decisions.parquet").resolve().relative_to(ROOT).as_posix(),
            "portfolio_trades_path": (output / "portfolio_trades.parquet").resolve().relative_to(ROOT).as_posix() if trades else None,
            "portfolio_curve_rows": int(len(curve)),
        }
        print(f"[return-regression] economics={name} independent_fills={independent.get('fills')} portfolio_trades={portfolio.get('trade_count')}", flush=True)
    result_path = OUTPUT_DIR / "holdout_economics.json"
    _write_json(result_path, results)
    return results


def _freeze_manifest(identity: dict, selection: dict, final_calibrator: dict, model_path: Path, dev_results_path: Path, data: dict, arrays: dict) -> dict:
    bundle = json.loads((ROOT / CONFIG["bundle_path"]).read_text(encoding="utf-8"))
    frozen = {
        "status": "frozen_before_single_holdout_evaluation",
        "experiment_id": CONFIG["experiment_id"],
        "frozen_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_identity": identity,
        "target": CONFIG["normalization"],
        "feature_order": bundle["feature_order"],
        "feature_order_sha256": hashlib.sha256(json.dumps(bundle["feature_order"], separators=(",", ":")).encode()).hexdigest(),
        "generator_configuration": {
            "volume_profile": json.loads((ROOT / "data/analysis/polymarket/BTC/btc_preopen_full_development_fit_20261016/pipeline/794a69e369a3f82d/stages/final_feature_dataset_ac0ca964ce62/states/volume_profile/BTCUSD_INDEXVOL_UM_BTCUSDT_1m_vp_fixed_range_v3_modeling_end.json").read_text(encoding="utf-8")),
            "reaction_profile": json.loads((ROOT / "data/analysis/polymarket/BTC/btc_preopen_full_development_fit_20261016/pipeline/794a69e369a3f82d/stages/final_feature_dataset_ac0ca964ce62/states/reaction_profile/BTCUSD_INDEXVOL_UM_BTCUSDT_1m_rp_fixed_grid_v1_modeling_end.json").read_text(encoding="utf-8")),
        },
        "selected_sample_weight_sha256_binary_rows": data["weight_sha256"],
        "cv_split_fingerprints": data["fold_fingerprints"],
        "regression_eligible_training_rows": int(arrays["x"].shape[0]),
        "parameters": selection,
        "regressor_path": model_path.resolve().relative_to(ROOT).as_posix(),
        "regressor_sha256": _sha256(model_path),
        "calibrator": final_calibrator,
        "development_results_path": dev_results_path.resolve().relative_to(ROOT).as_posix(),
        "development_results_sha256": _sha256(dev_results_path),
        "holdout_period": {
            "start_inclusive_utc": CONFIG["holdout_first_market_start_utc"],
            "last_market_start_inclusive_utc": CONFIG["holdout_last_market_start_utc"],
            "data_cutoff_exclusive_utc": CONFIG["holdout_cutoff_exclusive_utc"],
        },
        "holdout_selection": False,
    }
    path = OUTPUT_DIR / "frozen_manifest.json"
    _write_json(path, frozen)
    return frozen


def _write_final_report(manifest: dict, target_summary: dict, tuning: dict, oof_record: dict, dev_results: dict, holdout: dict, economics: dict, costs: dict) -> None:
    dev_proxy = dev_results["probability_quality"]["proxy"]
    dev_official = dev_results["probability_quality"]["official"]
    hold_proxy = holdout["models"]["proxy"]
    hold_official = holdout["models"]["official"]
    regres_dev = dev_results["regression_quality"]
    profile = json.loads((OUTPUT_DIR / "cost_estimate.json").read_text(encoding="utf-8"))
    lines = [
        "# BTC pre-open return regression experiment — 2026-10-17", "",
        "## Result", "",
        "One Huber return-regression challenger was fitted on the frozen 64-feature dataset. The development and historically exposed holdout scores are conditional on the already selected generators, weights, feature order, and regression hyperparameters.", "",
        f"- Main development comparison (proxy, same causal calibrated sample): regression log-loss `{dev_proxy['models']['return_regression_causal_logistic'].get('log_loss')}` vs binary Platt `{dev_proxy['models']['binary_causal_platt'].get('log_loss')}`; paired regression-minus-binary difference `{dev_proxy['paired_log_loss_return_minus_binary_causal']['candidate_minus_binary_log_loss']}` with 95% block interval `{dev_proxy['paired_log_loss_return_minus_binary_causal']['ci95']}`.",
        f"- Holdout proxy comparison: regression `{hold_proxy['models']['return_regression']['direction_probability'].get('log_loss')}` vs binary `{hold_proxy['models']['binary']['calibrated'].get('log_loss')}` log-loss on {hold_proxy['rows']:,} common rows.",
        f"- Holdout official comparison: regression `{hold_official['models']['return_regression']['direction_probability'].get('log_loss')}` vs binary `{hold_official['models']['binary']['calibrated'].get('log_loss')}` log-loss on {hold_official['rows']:,} outcomes released by the frozen cutoff.", "",
        "## Frozen target and data", "",
        f"For each 12:00–12:05 market, the decision remains 11:59:00 UTC (T−60), the prediction deadline remains 11:59:01 (T−59), and `{RETURN_TARGET_COL}` is `Close[12:04] / Open[12:00] − 1`, available at 12:05. The return was reconstructed from cached minute OHLC using the contract window; cached binary labels verify the direction and tie rule.", "",
        "At row `Opened`, sigma uses only close-to-close returns from adjacent one-minute candles ending at that row’s close, available at `Opened + 1 minute`, which equals the decision time:", "",
        f"```text\nr = {RETURN_TARGET_COL}\nsigma_5m = sqrt(5) * population_std(last 60 valid one-minute simple returns; ddof=0)\nz = r / max(sigma_5m, 0.000001)\n```", "",
        "The 60-return window is fixed. A timestamp gap or missing/nonpositive close invalidates affected windows until 60 contiguous returns are present. Missing targets remain missing, zero sigma uses the specified positive denominator floor, and z is not clipped.", "",
        f"- Binary-eligible development rows: {target_summary['binary_eligible_training_rows']:,}; regression-eligible rows: {target_summary['regression_eligible_training_rows']:,}; excluded for unavailable return or sigma: {target_summary['excluded_for_unavailable_return_or_sigma']:,}.",
        f"- Development return range: `{target_summary['return_target_development']['min']}` to `{target_summary['return_target_development']['max']}`; z range: `{target_summary['normalized_target_z_development']['min']}` to `{target_summary['normalized_target_z_development']['max']}`.",
        f"- Cached source-minute gaps: {target_summary['minute_timestamp_gaps_in_source']:,}; missing sigma training rows: {target_summary['sigma_missing_training_rows']:,}; zero-volatility training rows: {target_summary['sigma_zero_training_rows']:,}.",
        f"- Exact 64-feature order, selected VP/RP settings, sample weights, and 10 saved fold fingerprints are recorded in `{manifest['frozen_manifest_path']}`.", "",
        "## Development probability comparison", "",
        "| Target | Model | Rows | Log-loss | Brier | AUC | ECE (10 bins) |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for target_name, target_result in dev_results["probability_quality"].items():
        for model_name in ("return_regression_causal_logistic", "binary_causal_platt", "binary_raw", "constant_prior"):
            metric = target_result["models"][model_name]
            lines.append(f"| {target_name} | {model_name} | {metric.get('n', 0):,} | {metric.get('log_loss', float('nan')):.6f} | {metric.get('brier', float('nan')):.6f} | {metric.get('auc', float('nan')):.4f} | {metric.get('ece_10_equal_width', float('nan')):.6f} |")
    peak_memory = costs.get("peak_working_set_gib")
    peak_memory_text = f"{peak_memory:.2f} GiB" if isinstance(peak_memory, (int, float)) else "unavailable (the process counter returned no sample)"
    lines.extend([
        "", "The first internal OOF fold is excluded from calibrated probability scoring because no earlier OOF predictions exist to fit its calibrator. Later fold calibrators use only prior OOF predictions whose proxy labels were available by the first decision in that fold. The constant-probability comparator uses the same earlier available labels. Official Polymarket outcomes are scored separately; they did not fit either calibrator.", "",
        f"Regression OOF diagnostics on {regres_dev['rows']:,} actual decision rows: Huber(alpha=0.9) `{regres_dev['huber_alpha_0_9']:.6g}`, MAE `{regres_dev['mae']:.6g}`, predicted-z standard deviation `{regres_dev['predicted_z_std_population']:.6g}`, Pearson(predicted z, actual z) `{regres_dev['actual_z_predicted_z_pearson']:.5f}`, and Pearson(predicted z, actual return) `{regres_dev['actual_return_predicted_z_pearson']:.5f}`.", "",
        "Fold metrics, calibration bins, predicted-z quantiles, and actual return/UP frequency by predicted-z decile are in `development_results.json`.", "",
        "## Frozen single holdout evaluation", "",
        f"The fit, parameters, and final development OOF calibrator were frozen at `{manifest['frozen_utc']}` before scoring. Holdout was historically exposed in earlier repository work and is descriptive. Cutoff: `{CONFIG['holdout_cutoff_exclusive_utc']}`. No development or holdout result changed the model or policy.", "",
        "| Target | Forecast | Rows | Log-loss | Brier | AUC | ECE (10 bins) |", "|---|---|---:|---:|---:|---:|---:|",
    ])
    for target_name, target_result in holdout["models"].items():
        for name in ("return_regression", "binary", "A_frozen"):
            probability_kinds = ("direction_probability",) if name == "return_regression" else ("raw", "calibrated")
            for probability_kind in probability_kinds:
                metric = target_result["models"][name][probability_kind]
                lines.append(f"| {target_name} | {name} {probability_kind} | {metric.get('n', 0):,} | {metric.get('log_loss', float('nan')):.6f} | {metric.get('brier', float('nan')):.6f} | {metric.get('auc', float('nan')):.4f} | {metric.get('ece_10_equal_width', float('nan')):.6f} |")
    lines.extend(["", "## Economics and cost", "", "Both forecasts use the existing 1% minimum net-EV rule, T−60 decision and T−59 snapshots. Independent orders request $5 each; the separate portfolio starts at $100, stakes 5% of free cash, and does not borrow. No archived order-book replay was run.", ""])
    lines.extend(["| Forecast | Independent $5 signals | Filled / settled | Net PnL | 5%-free-cash portfolio trades | Ending cash | PnL | Minimum-order skips |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for name, item in economics["models"].items():
        independent = item["independent_fixed_5_usd"]
        portfolio = item["no_credit_5pct_free_cash_portfolio"]
        minimum = portfolio.get("minimum_order_diagnostic", {}).get("minimum_skip_count", 0)
        lines.append(f"| {name} | {independent.get('signals', 0):,} | {independent.get('fills', 0):,} / {independent.get('resolved_fills_for_pnl', 0):,} | ${independent.get('pnl_after_fee_usd', 0.0):,.2f} | {portfolio.get('trade_count', 0):,} | ${portfolio.get('ending_cash_after_settlement_usd', 0.0):,.2f} | ${portfolio.get('net_pnl_usd', 0.0):,.2f} | {minimum:,} |")
    lines.extend([
        "", "Decision skip reasons, execution fills, minimum-order diagnostics, accounting audits, and monthly portfolio rows are in `holdout_economics.json` and its per-model artifacts.", "",
        f"- Regression tuning: {tuning['trials_evaluated']} study records ({tuning['completed_trials']} complete, {tuning['failed_trials']} failed, {tuning.get('pruned_trials', 0)} pruned), {tuning['elapsed_seconds']:.1f} s; best criterion `{tuning['best_cv_criterion']:.8g}` = mean Huber `{tuning['best_cv_huber_mean']:.8g}` + 0.5 × fold SD `{tuning['best_cv_huber_std_population']:.8g}`; selected iteration {tuning['best_iteration']}. Failed/partial trials did not enter selection.",
        f"- Trial exclusions: {'; '.join(tuning.get('failure_notes', [])) or 'none'}.",
        f"- Regression final fit: {costs.get('final_fit_seconds', float('nan')):.2f} s; 10-fold development OOF fits: {oof_record['elapsed_seconds']:.2f} s; warm holdout inference is benchmarked in `costs.json`.",
        f"- Pre-run profile: one full CV fold took {profile['profile_elapsed_seconds']:.2f} s; the linear 60-trial, 10-fold projection with 25% overhead was {profile['60_trial_10_fold_projection_seconds_with_25pct_overhead']:.1f} s. Actual tuning took {costs.get('tuning_seconds', float('nan')):.1f} s and stopped at the authorized 2,700-second cap. Peak process working set: `{peak_memory_text}`.",
        "- The main binary model tuning took 1,343.5 s, its final fit 3.1 s, and its ten-fold OOF predictions 19.9 s in the completed 2026-10-16 run.",
        "- Limits: calibration and feature/config selection are conditional on development; official development labels cover only the market-index period; the historical holdout is exposed; return regression and volatility definition are one tested variant only.", "",
        "## Conclusion", "",
        f"On the primary development proxy comparison, regression log-loss was {dev_proxy['models']['return_regression_causal_logistic'].get('log_loss', float('nan')):.6f} versus {dev_proxy['models']['binary_causal_platt'].get('log_loss', float('nan')):.6f}; the paired interval spans zero, so the small point-estimate advantage is inconclusive. On the single holdout proxy assessment it was {hold_proxy['models']['return_regression']['direction_probability'].get('log_loss', float('nan')):.6f} versus {hold_proxy['models']['binary']['calibrated'].get('log_loss', float('nan')):.6f}. These results do not establish a material improvement or reject other continuous targets.", "",
        f"Reproduction from the repository root: `{CONFIG['reproduction']}`.",
        f"Frozen manifest: `{manifest['frozen_manifest_path']}`. Development predictions: `{oof_record['prediction_path']}`.",
        "",
    ])
    report_path = ROOT / CONFIG["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.is_file():
        previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if previous.get("experiment_id") != CONFIG["experiment_id"]:
            raise RuntimeError("The return-regression output directory belongs to a different experiment")
        if previous.get("status") == "completed":
            report_path = ROOT / CONFIG["report_path"]
            if report_path.is_file() and previous.get("report_sha256") == _sha256(report_path):
                print("[return-regression] verified completed output already exists", flush=True)
                return previous
            frozen_manifest = json.loads((ROOT / previous["frozen_manifest_path"]).read_text(encoding="utf-8"))
            previous["frozen_utc"] = frozen_manifest["frozen_utc"]
            _write_final_report(
                previous,
                json.loads((OUTPUT_DIR / "target_summary.json").read_text(encoding="utf-8")),
                json.loads((OUTPUT_DIR / "tuning_result.json").read_text(encoding="utf-8")),
                previous["development_oof"],
                json.loads((OUTPUT_DIR / "development_results.json").read_text(encoding="utf-8")),
                json.loads((OUTPUT_DIR / "holdout_results.json").read_text(encoding="utf-8")),
                json.loads((OUTPUT_DIR / "holdout_economics.json").read_text(encoding="utf-8")),
                json.loads((OUTPUT_DIR / "costs.json").read_text(encoding="utf-8")),
            )
            previous["report_path"] = CONFIG["report_path"]
            previous["report_sha256"] = _sha256(report_path)
            _write_json(MANIFEST_PATH, previous)
            _write_status(status="completed", stage="none", completed=["target_validation", "smoke", "cost_estimate", "single_huber_study", "final_fit", "causal_development_comparison", "frozen_holdout_quality", "snapshot_economics", "report"], next_step=f"Review `{CONFIG['report_path']}`, then run targeted tests, git diff --check, commit, and push.")
            print("[return-regression] rebuilt missing report from the already scored frozen outputs", flush=True)
            return previous
    start = time.perf_counter()
    bundle, binary_training, binary_tuning, frozen = _load_bundle_inputs()
    cache_path = ROOT / CONFIG["dataset_path"]
    baseline_model_path = ROOT / bundle["model_path"]
    identity = {
        "config_sha256": _sha256(CONFIG_PATH),
        "contract_sha256": _sha256(ROOT / CONFIG["source_contract"]),
        "dataset_sha256": _sha256(cache_path),
        "benchmark_bundle_sha256": _sha256(ROOT / CONFIG["bundle_path"]),
        "benchmark_model_sha256": _sha256(baseline_model_path),
        "benchmark_oof_sha256": _sha256(ROOT / CONFIG["binary_oof_predictions_path"]),
        "benchmark_holdout_predictions_sha256": _sha256(ROOT / CONFIG["binary_holdout_predictions_path"]),
        "a_frozen_holdout_predictions_sha256": _sha256(ROOT / CONFIG["a_frozen_holdout_predictions_path"]),
        "official_markets_sha256": _sha256(ROOT / CONFIG["official_markets_path"]),
    }
    if identity["a_frozen_holdout_predictions_sha256"] != CONFIG["a_frozen_holdout_predictions_sha256"]:
        raise RuntimeError("The supplemental A_frozen historical prediction artifact changed")
    _write_status(status="running", stage="target_and_cache_validation", completed=[], next_step="Validate the cached return target, causal sigma, 64-feature order, weights, and saved CV splits.")
    data = _load_cached_data(bundle["feature_order"], binary_training, binary_tuning)
    arrays = _make_regression_arrays(data, bundle["feature_order"])
    identity.update({
        "feature_order_sha256": hashlib.sha256(json.dumps(bundle["feature_order"], separators=(",", ":")).encode()).hexdigest(),
        "binary_training_weight_sha256": data["weight_sha256"],
        "cv_split_fingerprints": data["fold_fingerprints"],
    })
    input_record = {
        **identity,
        "feature_order": bundle["feature_order"],
        "feature_order_sha256": hashlib.sha256(json.dumps(bundle["feature_order"], separators=(",", ":")).encode()).hexdigest(),
        "binary_training_weight_sha256": data["weight_sha256"],
        "cv_split_fingerprints": data["fold_fingerprints"],
        "dataset_load_seconds": data["load_seconds"],
        "target_and_volatility_seconds": data["volatility_seconds"],
        "generator_settings_from_frozen_manifest": {
            "volume_profile_config": frozen.get("volume_profile_config"),
            "reaction_profile_config": frozen.get("reaction_profile_config"),
            "selected_feature_count": len(bundle["feature_order"]),
            "target_decision_weight": binary_training.get("decision_weight"),
        },
    }
    _write_json(OUTPUT_DIR / "input_manifest.json", input_record)
    _write_json(OUTPUT_DIR / "target_summary.json", arrays["summary"])
    print(f"[return-regression] loaded rows={len(data['opened']):,} dev={len(data['fit_rows']):,} features=64; cache_read={data['load_seconds']:.1f}s", flush=True)

    # A cache-backed smoke verifies the exact stored return and normalized target before tuning.
    decision_rows = np.flatnonzero(data["fit_decision"])
    smoke_rows = decision_rows[:20_000]
    smoke_valid = np.isfinite(data["z"][data["fit_rows"][smoke_rows]])
    if int(smoke_valid.sum()) < 1_000:
        raise RuntimeError("The cache-backed target smoke has too few normalized return rows")
    smoke_global = data["fit_rows"][smoke_rows[smoke_valid]]
    smoke_z = data["z"][smoke_global]
    smoke_x = data["feature_values"][smoke_global]
    smoke_y = data["label"][smoke_global].astype(np.int8)
    smoke_training = lgb.Dataset(smoke_x[: max(512, len(smoke_x) // 2)], label=smoke_z[: max(512, len(smoke_z) // 2)], feature_name=bundle["feature_order"], free_raw_data=True)
    smoke_params = {**_base_lgb_params(len(bundle["feature_order"])), **binary_tuning["best_params"]}
    smoke_params.update({"device_type": CONFIG["model"]["compute_backend"], "objective": "huber", "alpha": float(CONFIG["model"]["alpha"]), "metric": "None", "num_threads": 2})
    smoke_model = lgb.train(smoke_params, smoke_training, num_boost_round=20)
    smoke_prediction = smoke_model.predict(smoke_x[-2_000:], num_threads=2)
    smoke_calibrator = _fit_logistic(smoke_prediction, smoke_y[-2_000:])
    smoke_probability = _apply_logistic(smoke_prediction, smoke_calibrator)
    if not np.isfinite(smoke_z).all() or not np.isfinite(smoke_prediction).all() or not np.isfinite(smoke_probability).all():
        raise RuntimeError("Cache-backed target/regression/probability smoke produced non-finite values")
    del smoke_training, smoke_model
    _write_json(OUTPUT_DIR / "smoke_summary.json", {
        "status": "passed", "cache_rows": int(len(smoke_rows)),
        "normalized_target_rows": int(len(smoke_z)), "regression_prediction_rows": int(len(smoke_prediction)),
        "regression_huber_alpha": float(CONFIG["model"]["alpha"]),
        "probability_min": float(smoke_probability.min()), "probability_max": float(smoke_probability.max()),
        "backend": CONFIG["model"]["compute_backend"],
    })

    estimate = _profile_regressor(arrays, bundle["feature_order"])
    _write_status(status="running", stage="huber_tuning", completed=["cache_and_fold_validation", "target_smoke", "measured_runtime_estimate"], next_step="Run one Huber Optuna study with the frozen six-parameter space and the registered time/trial limits.")
    tuning = _run_tuning(arrays, bundle["feature_order"], identity)
    model_path = OUTPUT_DIR / "regressor.txt"
    final_started = time.perf_counter()
    final_model = _fit_regression_model(arrays, bundle["feature_order"], tuning["best_params"], tuning["best_iteration"])
    temporary_model = model_path.with_suffix(".txt.tmp")
    final_model.save_model(str(temporary_model))
    temporary_model.replace(model_path)
    final_fit_seconds = time.perf_counter() - final_started
    oof, oof_record = _generate_development_oof(data, arrays, bundle["feature_order"], tuning)
    causal_dev, final_calibrator = _causal_development_probabilities(oof)
    dev_results = _write_development_results(oof, causal_dev, final_calibrator, data, arrays, tuning)
    _write_status(status="running", stage="holdout_freeze", completed=["cache_and_fold_validation", "target_smoke", "measured_runtime_estimate", "huber_tuning", "final_fit", "development_oof", "causal_calibration", "development_comparison"], next_step="Freeze target, features, regressor, calibrator, and policy before the single holdout evaluation.")
    freeze = _freeze_manifest(identity, tuning, final_calibrator, model_path, OUTPUT_DIR / "development_results.json", data, arrays)
    manifest = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "holdout_evaluation_running",
        "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_identity": identity,
        "tuning": tuning,
        "target_summary_path": (OUTPUT_DIR / "target_summary.json").resolve().relative_to(ROOT).as_posix(),
        "development_results_path": (OUTPUT_DIR / "development_results.json").resolve().relative_to(ROOT).as_posix(),
        "development_oof": oof_record,
        "frozen_manifest_path": (OUTPUT_DIR / "frozen_manifest.json").resolve().relative_to(ROOT).as_posix(),
        "frozen_manifest_sha256": _sha256(OUTPUT_DIR / "frozen_manifest.json"),
        "resource_snapshot_before_holdout": _resource_snapshot(),
        "holdout_selection": False,
    }
    _write_json(MANIFEST_PATH, manifest)

    # No holdout target or official label is scored until the frozen manifest exists.
    _write_status(status="running", stage="single_holdout_evaluation", completed=["target_and_cache_validation", "smoke", "runtime_estimate", "one_huber_study", "final_fit", "development_oof_and_comparison", "freeze"], next_step="Run the single frozen common-sample holdout score and snapshot-based economics; do not revise any choices.")
    holdout_predictions, holdout_x, _ = _holdout_predictions(data, final_model, final_calibrator["return_score_logistic"])
    baseline_holdout = pd.read_parquet(ROOT / CONFIG["binary_holdout_predictions_path"])
    a_frozen_holdout = pd.read_parquet(ROOT / CONFIG["a_frozen_holdout_predictions_path"])
    holdout_quality = _holdout_quality(holdout_predictions, baseline_holdout, a_frozen_holdout, data)
    holdout_path = OUTPUT_DIR / "holdout_predictions.parquet"
    holdout_predictions.to_parquet(holdout_path.with_suffix(".parquet.tmp"), index=False, compression="zstd")
    holdout_path.with_suffix(".parquet.tmp").replace(holdout_path)
    probabilities_for_economics = {
        "return_regression": holdout_predictions,
        "binary": baseline_holdout,
    }
    economics = _holdout_economics(probabilities_for_economics)
    holdout_quality["holdout_prediction_path"] = holdout_path.resolve().relative_to(ROOT).as_posix()
    holdout_quality["holdout_prediction_sha256"] = _sha256(holdout_path)
    holdout_quality["economics_path"] = (OUTPUT_DIR / "holdout_economics.json").resolve().relative_to(ROOT).as_posix()
    _write_json(OUTPUT_DIR / "holdout_results.json", holdout_quality)

    baseline_latency = _inference_benchmark(baseline_model_path, holdout_x, model_kind="binary")
    regression_latency = _inference_benchmark(model_path, holdout_x, model_kind="regression", calibrator=final_calibrator["return_score_logistic"])
    costs = {
        "cost_estimate_seconds": estimate["estimated_tuning_minutes_capped"] * 60.0,
        "dataset_load_seconds": data["load_seconds"],
        "volatility_target_construction_seconds": data["volatility_seconds"],
        "tuning_seconds": tuning["elapsed_seconds"],
        "final_fit_seconds": final_fit_seconds,
        "development_oof_fits_seconds": oof_record["elapsed_seconds"],
        "inference": {"binary": baseline_latency, "return_regression": regression_latency},
        "resource_snapshot_at_completion": _resource_snapshot(),
    }
    process_peak = costs["resource_snapshot_at_completion"].get("process_peak_working_set_bytes")
    costs["peak_working_set_gib"] = float(process_peak / 1024**3) if process_peak else None
    _write_json(OUTPUT_DIR / "costs.json", costs)
    manifest.update({
        "status": "completed",
        "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "frozen_utc": freeze["frozen_utc"],
        "holdout_results_path": (OUTPUT_DIR / "holdout_results.json").resolve().relative_to(ROOT).as_posix(),
        "holdout_results_sha256": _sha256(OUTPUT_DIR / "holdout_results.json"),
        "holdout_economics_path": (OUTPUT_DIR / "holdout_economics.json").resolve().relative_to(ROOT).as_posix(),
        "holdout_economics_sha256": _sha256(OUTPUT_DIR / "holdout_economics.json"),
        "costs_path": (OUTPUT_DIR / "costs.json").resolve().relative_to(ROOT).as_posix(),
        "total_elapsed_seconds": time.perf_counter() - start,
        "historically_exposed": True,
        "holdout_selection": False,
    })
    _write_json(MANIFEST_PATH, manifest)
    _write_final_report(manifest, arrays["summary"], tuning, oof_record, dev_results, holdout_quality, economics, costs)
    _write_status(status="completed", stage="none", completed=["target_validation", "smoke", "cost_estimate", "single_huber_study", "final_fit", "causal_development_comparison", "frozen_holdout_quality", "snapshot_economics", "report"], next_step=f"Review `{CONFIG['report_path']}`, then run targeted tests, git diff --check, commit, and push.")
    manifest["report_path"] = CONFIG["report_path"]
    manifest["report_sha256"] = _sha256(ROOT / CONFIG["report_path"])
    _write_json(MANIFEST_PATH, manifest)
    print(f"[return-regression] complete elapsed={manifest['total_elapsed_seconds']:.1f}s report={CONFIG['report_path']}", flush=True)
    return manifest


if __name__ == "__main__":
    run()
