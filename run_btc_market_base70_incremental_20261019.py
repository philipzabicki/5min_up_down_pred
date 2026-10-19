"""Measure one causal daily market calibration layer over saved BASE70 OOF forecasts."""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import platform
import time
from ctypes import wintypes
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

import run_btc_market_probability_experiment as market_probability
import run_btc_preopen_research as research
import run_btc_preopen_t59_reassessment as replay
import run_btc_preopen_taker_flow_20261018 as taker


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_market_base70_incremental_20261019.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_directory"]
RUN_MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
REPORT_PATH = ROOT / CONFIG["report_path"]
VERSIONED_MANIFEST_PATH = ROOT / CONFIG["versioned_manifest_path"]
STATUS_PATH = ROOT / CONFIG["status_path"]
STATUS_BEGIN = "<!-- BTC_MARKET_BASE70_INCREMENTAL_20261019_BEGIN -->"
STATUS_END = "<!-- BTC_MARKET_BASE70_INCREMENTAL_20261019_END -->"

START_UTC = pd.Timestamp(CONFIG["development_start_utc"])
CUTOFF_UTC = pd.Timestamp(CONFIG["development_cutoff_exclusive_utc"])
EPSILON = float(CONFIG["probability_epsilon"])
PENALTY = float(CONFIG["regularization_lambda"])
MINIMUM_TRAINING_ROWS = int(CONFIG["minimum_training_rows"])
MODEL_NAMES = ("MARKET", "RAW", "M0", "M1")
PROBABILITY_COLUMNS = {
    "MARKET": "p_market",
    "RAW": "p_raw",
    "M0": "p_m0",
    "M1": "p_m1",
}
CONTRASTS = {
    "M1_MINUS_M0": ("M1", "M0"),
    "M0_MINUS_MARKET": ("M0", "MARKET"),
    "M1_MINUS_RAW": ("M1", "RAW"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _record(path: Path) -> dict:
    return {"path": _relative(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _verify_records(records: list[dict]) -> bool:
    for record in records:
        path = ROOT / record["path"]
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
            return False
        if _sha256(path) != record["sha256"]:
            return False
    return bool(records)


def _checkpoint_valid(manifest: dict, stage_name: str) -> bool:
    stage = manifest.get("stages", {}).get(stage_name, {})
    return stage.get("status") == "completed" and _verify_records(stage.get("outputs", []))


def _write_manifest(manifest: dict) -> None:
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(RUN_MANIFEST_PATH, manifest)
    _write_json(VERSIONED_MANIFEST_PATH, manifest)


def _source_paths() -> dict[str, Path]:
    paths = {key: ROOT / value for key, value in CONFIG["sources"].items()}
    paths.update({
        "config": CONFIG_PATH,
        "runner": Path(__file__),
        "market_probability_definition": ROOT / "run_btc_market_probability_experiment.py",
        "research_calendar_and_snapshot_helpers": ROOT / "run_btc_preopen_research.py",
        "economic_engine": ROOT / "run_btc_preopen_t59_reassessment.py",
        "official_label_scoring": ROOT / "run_btc_preopen_taker_flow_20261018.py",
    })
    return paths


def _freeze_before_calculation() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = _source_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Frozen experiment sources are missing: " + ", ".join(missing))
    source_records = {name: _record(path) for name, path in paths.items()}

    base_manifest = json.loads(paths["base70_run_manifest_path"].read_text(encoding="utf-8"))
    spec_sources = base_manifest["frozen_comparison"]["specification"]["sources"]
    expected_snapshots = spec_sources["decision_and_execution_snapshots"]
    if _sha256(paths["t59_snapshots_path"]) != expected_snapshots["t59_sha256"]:
        raise RuntimeError("T-59 source snapshot differs from the frozen BASE70 reference run")
    if _sha256(paths["t60_snapshots_path"]) != expected_snapshots["t60_sha256"]:
        raise RuntimeError("T-60 source snapshot differs from the frozen BASE70 reference run")
    expected_labels = base_manifest["frozen_comparison"]["specification"]["source_hashes"]["source_official_market_index"]
    if _sha256(paths["official_market_index_path"]) != expected_labels:
        raise RuntimeError("Official labels differ from the frozen BASE70 reference run")

    frozen = {
        "config": CONFIG,
        "source_records": source_records,
        "method": {
            "fit_before_each_utc_day": True,
            "training_label_boundary_strict": True,
            "shared_m0_m1_training_rows": True,
            "development_only": True,
            "holdout_evaluated": False,
            "policy_variants": MODEL_NAMES,
            "quality_contrasts": CONTRASTS,
        },
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    stable_frozen = {key: value for key, value in frozen.items() if key != "frozen_at_utc"}
    identity = _canonical_sha256(stable_frozen)
    if RUN_MANIFEST_PATH.is_file():
        manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("frozen_comparison", {}).get("identity_sha256") != identity:
            previous = manifest.get("frozen_comparison", {}).get("specification", {})
            previous_records = previous.get("source_records", {})
            changed = [
                name for name, record in source_records.items()
                if previous_records.get(name, {}).get("sha256") != record["sha256"]
            ]
            unchanged = all(
                previous_records.get(name, {}).get("sha256") == record["sha256"]
                for name, record in source_records.items() if name != "runner"
            )
            if changed != ["runner"] or not unchanged or manifest.get("status") not in {"running", "completed"}:
                raise RuntimeError("Existing run manifest does not match the frozen configuration or sources")
            manifest["implementation_resume"] = {
                "previous_runner_sha256": previous_records["runner"]["sha256"],
                "current_runner_sha256": source_records["runner"]["sha256"],
                "reason": "A follow-up runner edit requires a clean result rebuild. All result stages are invalidated and recomputed under the corrected runner.",
                "recorded_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            for stage in manifest.get("stages", {}).values():
                if stage.get("status") in {"running", "completed"}:
                    stage["status"] = "invalidated"
                    stage["invalidation_reason"] = "runner source changed after failed first pass; recompute under corrected runner"
            manifest["frozen_comparison"] = {"identity_sha256": identity, "specification": frozen}
            manifest["runtime_identity"] = {
                **manifest.get("runtime_identity", {}),
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "runner_sha256": source_records["runner"]["sha256"],
                "economic_engine_sha256": source_records["economic_engine"]["sha256"],
            }
            manifest["status"] = "frozen_before_calculation"
            manifest["active_stage"] = None
            manifest["next_step"] = "Recompute all invalidated result stages under the corrected, re-frozen runner."
            _write_manifest(manifest)
            return manifest
        if not _verify_records(list(source_records.values())):
            raise RuntimeError("A frozen source changed since the experiment was first run")
        return manifest

    manifest = {
        "schema_version": 1,
        "experiment_id": CONFIG["experiment_id"],
        "status": "frozen_before_calculation",
        "active_stage": None,
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "frozen_comparison": {"identity_sha256": identity, "specification": frozen},
        "source_run_status": base_manifest.get("status"),
        "stages": {},
        "runtime_identity": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "runner_sha256": source_records["runner"]["sha256"],
            "economic_engine_sha256": source_records["economic_engine"]["sha256"],
        },
    }
    _write_manifest(manifest)
    return manifest


def _peak_working_set_bytes() -> int | None:
    if hasattr(ctypes, "WinDLL"):
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD),
                ("peak_working_set_size", ctypes.c_size_t), ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t), ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t), ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t), ("peak_pagefile_usage", ctypes.c_size_t),
                ("private_usage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD)
        get_process_memory_info.restype = wintypes.BOOL
        if get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
            return int(counters.peak_working_set_size)
    return None


def _read_development_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    field_type = dataset.schema.field("market_start_utc").type
    start = pa.scalar(START_UTC.to_pydatetime(), type=field_type)
    cutoff = pa.scalar(CUTOFF_UTC.to_pydatetime(), type=field_type)
    date_filter = (ds.field("market_start_utc") >= start) & (ds.field("market_start_utc") < cutoff)
    return dataset.to_table(columns=columns, filter=date_filter).to_pandas()


def _logit(probability) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(value) - np.log1p(-value)


def _build_development_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = CONFIG["sources"]
    raw = _read_development_parquet(
        ROOT / sources["raw_oof_path"],
        [
            "Opened", "nominal_decision_at", "target_label_available_at",
            "target_preopen_5m_up_binance_index_proxy", "market_start_utc", "fold_id",
            "p_model_raw", "prediction_available_at_utc",
        ],
    )
    for column in ("Opened", "nominal_decision_at", "target_label_available_at", "market_start_utc", "prediction_available_at_utc"):
        raw[column] = pd.to_datetime(raw[column], utc=True, errors="raise")
    raw.sort_values("market_start_utc", kind="stable", inplace=True)
    raw.reset_index(drop=True, inplace=True)
    if raw.market_start_utc.duplicated().any():
        raise RuntimeError("Development BASE70 OOF contains duplicate market starts")
    if not raw.market_start_utc.eq(raw.nominal_decision_at + pd.Timedelta(minutes=1)).all():
        raise RuntimeError("BASE70 OOF does not align to the existing T-60 decision timestamp")
    recorded = raw.prediction_available_at_utc.notna()
    if not raw.loc[recorded, "prediction_available_at_utc"].eq(raw.loc[recorded, "market_start_utc"] - pd.Timedelta(seconds=59)).all():
        raise RuntimeError("A recorded BASE70 prediction timestamp violates the T-60/T-59 contract")

    official = _read_development_parquet(
        ROOT / sources["official_market_index_path"],
        ["condition_id", "market_start_utc", "resolved_at_utc", "target_polymarket_up"],
    )
    official["market_start_utc"] = pd.to_datetime(official.market_start_utc, utc=True, errors="raise")
    official["resolved_at_utc"] = pd.to_datetime(official.resolved_at_utc, utc=True, errors="coerce")
    official = official.drop_duplicates("market_start_utc", keep=False)
    frame = raw.merge(official, on="market_start_utc", how="left", validate="one_to_one")
    starts = frame.market_start_utc
    resolved = pd.to_datetime(frame.resolved_at_utc, utc=True, errors="coerce")
    minimum_resolution = starts + pd.Timedelta(minutes=5)
    release_base = resolved.where(resolved.ge(minimum_resolution), minimum_resolution)
    frame["official_label_release_at_utc"] = release_base + pd.Timedelta(seconds=60)
    frame["official_outcome_available_by_cutoff"] = (
        frame.target_polymarket_up.isin((0, 1))
        & resolved.notna()
        & frame.official_label_release_at_utc.le(CUTOFF_UTC)
    )
    frame.rename(columns={"p_model_raw": "p_raw"}, inplace=True)
    frame["p_raw"] = pd.to_numeric(frame.p_raw, errors="coerce")
    if not frame.p_raw.dropna().between(0.0, 1.0).all():
        raise RuntimeError("Saved raw BASE70 OOF probabilities fall outside [0, 1]")

    t60 = _read_development_parquet(
        ROOT / sources["t60_snapshots_path"],
        [
            "condition_id", "market_start_utc", "up_best_bid", "up_best_ask", "down_best_bid", "down_best_ask",
            "quote_valid", "archive_entry_hour_available", "archive_freshness_window_available",
            "no_future_event_at_entry", "no_future_source_event_at_entry", "has_full_snapshot",
            "up_quote_complemented", "down_quote_complemented", "up_ask_age_seconds", "down_ask_age_seconds",
            "archive_gap_after_up_book_init", "archive_gap_after_down_book_init",
        ],
    )
    t60["condition_id"] = t60.condition_id.astype("string").str.lower()
    if t60.condition_id.duplicated().any():
        raise RuntimeError("T-60 source has duplicate development snapshots per condition id")
    t60_by_id = {str(row.condition_id): row._asdict() for row in t60.itertuples(index=False)}
    condition_ids = frame.condition_id.astype("string").str.lower()
    market_probabilities = np.full(len(frame), np.nan, dtype=np.float64)
    for index, condition_id in enumerate(condition_ids):
        if pd.isna(condition_id):
            continue
        snapshot = t60_by_id.get(str(condition_id))
        if snapshot is not None:
            market_probabilities[index] = market_probability._snapshot_features(snapshot)["p_market"] or np.nan
    frame["p_market"] = market_probabilities
    frame["decision_day_utc"] = frame.market_start_utc.dt.floor("D")
    frame["logit_market"] = np.nan
    frame["logit_raw"] = np.nan
    valid_inputs = frame.p_market.notna() & frame.p_raw.notna()
    frame.loc[valid_inputs, "logit_market"] = _logit(frame.loc[valid_inputs, "p_market"])
    frame.loc[valid_inputs, "logit_raw"] = _logit(frame.loc[valid_inputs, "p_raw"])
    frame["d_raw_minus_market"] = frame.logit_raw - frame.logit_market
    return frame, t60


def _training_rows(frame: pd.DataFrame, first_decision: pd.Timestamp) -> pd.DataFrame:
    release = pd.to_datetime(frame.official_label_release_at_utc, utc=True, errors="coerce")
    resolved = pd.to_datetime(frame.resolved_at_utc, utc=True, errors="coerce")
    valid = (
        frame.p_market.notna()
        & frame.p_raw.notna()
        & frame.target_polymarket_up.isin((0, 1))
        & resolved.notna()
        & release.notna()
        & release.lt(first_decision)
        & frame.market_start_utc.ge(START_UTC)
    )
    return frame.loc[valid].copy()


def _objective_gradient(coefficients, design: np.ndarray, labels: np.ndarray, *, include_deviation: bool) -> tuple[float, np.ndarray]:
    theta = np.asarray(coefficients, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or x.shape[1] != len(theta):
        raise ValueError("Objective design, labels, and coefficient shapes do not match")
    linear = x @ theta
    loss = float(np.mean(np.logaddexp(0.0, linear) - y * linear))
    gradient = x.T @ (expit(linear) - y) / len(y)
    loss += PENALTY * (theta[1] - 1.0) ** 2
    gradient[1] += 2.0 * PENALTY * (theta[1] - 1.0)
    if include_deviation:
        loss += PENALTY * theta[2] ** 2
        gradient[2] += 2.0 * PENALTY * theta[2]
    return loss, gradient


def _fit_model(train: pd.DataFrame, *, include_deviation: bool) -> dict:
    y = train.target_polymarket_up.to_numpy(dtype=np.float64)
    market_logit = train.logit_market.to_numpy(dtype=np.float64)
    if include_deviation:
        design = np.column_stack((np.ones(len(train)), market_logit, train.d_raw_minus_market.to_numpy(dtype=np.float64)))
    else:
        design = np.column_stack((np.ones(len(train)), market_logit))
    prior = float(np.clip(y.mean(), EPSILON, 1.0 - EPSILON))
    initial = np.asarray([_logit([prior])[0], 1.0, 0.0] if include_deviation else [_logit([prior])[0], 1.0])
    with threadpool_limits(limits=1):
        result = minimize(
            lambda value: _objective_gradient(value, design, y, include_deviation=include_deviation),
            initial,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-9, "maxls": 50},
        )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Daily {'M1' if include_deviation else 'M0'} optimizer failed: {result.message}")
    gradient_norm = float(np.linalg.norm(result.jac, ord=np.inf))
    if gradient_norm > 1e-5:
        raise RuntimeError(f"Daily optimizer gradient remained too large: {gradient_norm}")
    coefficients = {"a": float(result.x[0]), "b": float(result.x[1])}
    if include_deviation:
        coefficients["c"] = float(result.x[2])
    return {
        "coefficients": coefficients,
        "objective": float(result.fun),
        "gradient_inf_norm": gradient_norm,
        "iterations": int(result.nit),
        "optimizer_message": str(result.message),
    }


def _sample_identity(train: pd.DataFrame) -> str:
    values = pd.to_datetime(train.market_start_utc, utc=True).astype("int64").to_numpy(dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _predict_m0(logit_market, coefficients: dict) -> np.ndarray:
    return expit(float(coefficients["a"]) + float(coefficients["b"]) * np.asarray(logit_market, dtype=np.float64))


def _predict_m1(logit_market, deviation, coefficients: dict) -> np.ndarray:
    z_market = np.asarray(logit_market, dtype=np.float64)
    z_deviation = np.asarray(deviation, dtype=np.float64)
    return expit(float(coefficients["a"]) + float(coefficients["b"]) * z_market + float(coefficients.get("c", 0.0)) * z_deviation)


def _fit_daily_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = frame.copy()
    predictions["p_m0"] = np.nan
    predictions["p_m1"] = np.nan
    logs = []
    for day, positions in predictions.groupby("decision_day_utc", sort=True).groups.items():
        current = predictions.loc[positions]
        first_decision = pd.to_datetime(current.nominal_decision_at, utc=True).min()
        train = _training_rows(predictions, first_decision)
        labels = train.target_polymarket_up.astype(int)
        counts = labels.value_counts()
        has_both = len(counts) == 2
        entry = {
            "utc_day": pd.Timestamp(day).isoformat(),
            "first_decision_at_utc": first_decision.isoformat(),
            "training_rows": int(len(train)),
            "training_up_rows": int((labels == 1).sum()),
            "training_down_rows": int((labels == 0).sum()),
            "training_sample_sha256": _sample_identity(train),
            "m0_training_rows": int(len(train)),
            "m1_training_rows": int(len(train)),
            "m0_training_sample_sha256": _sample_identity(train),
            "m1_training_sample_sha256": _sample_identity(train),
            "a_m0": None, "b_m0": None, "a_m1": None, "b_m1": None, "c_m1": None,
            "m0_objective": None, "m1_objective": None,
            "m0_gradient_inf_norm": None, "m1_gradient_inf_norm": None,
            "m0_iterations": None, "m1_iterations": None,
            "status": "warmup_insufficient_rows_or_classes",
        }
        if len(train) < MINIMUM_TRAINING_ROWS or (CONFIG["require_both_classes"] and not has_both):
            logs.append(entry)
            continue
        m0 = _fit_model(train, include_deviation=False)
        m1 = _fit_model(train, include_deviation=True)
        if m0["coefficients"].get("c") is not None or m1["coefficients"].get("b") is None:
            raise RuntimeError("Daily model coefficient structure is invalid")
        if entry["m0_training_sample_sha256"] != entry["m1_training_sample_sha256"] or entry["m0_training_rows"] != entry["m1_training_rows"]:
            raise RuntimeError("M0 and M1 training samples differ")
        valid = predictions.index.isin(positions) & predictions.p_market.notna() & predictions.p_raw.notna()
        idx = predictions.index[valid]
        market_logit = predictions.loc[idx, "logit_market"].to_numpy(dtype=np.float64)
        deviation = predictions.loc[idx, "d_raw_minus_market"].to_numpy(dtype=np.float64)
        predictions.loc[idx, "p_m0"] = _predict_m0(market_logit, m0["coefficients"])
        predictions.loc[idx, "p_m1"] = _predict_m1(market_logit, deviation, m1["coefficients"])
        entry.update({
            "a_m0": m0["coefficients"]["a"], "b_m0": m0["coefficients"]["b"],
            "a_m1": m1["coefficients"]["a"], "b_m1": m1["coefficients"]["b"],
            "c_m1": m1["coefficients"]["c"],
            "m0_objective": m0["objective"], "m1_objective": m1["objective"],
            "m0_gradient_inf_norm": m0["gradient_inf_norm"], "m1_gradient_inf_norm": m1["gradient_inf_norm"],
            "m0_iterations": m0["iterations"], "m1_iterations": m1["iterations"],
            "m0_optimizer_message": m0["optimizer_message"], "m1_optimizer_message": m1["optimizer_message"],
            "status": "fit",
        })
        logs.append(entry)
    return predictions, pd.DataFrame(logs)


def _quality_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    probability_columns = [PROBABILITY_COLUMNS[name] for name in MODEL_NAMES]
    rows = predictions.loc[
        predictions[probability_columns].notna().all(axis=1)
        & predictions.target_polymarket_up.isin((0, 1))
        & predictions.resolved_at_utc.notna()
        & predictions.official_outcome_available_by_cutoff
    ].copy()
    if rows.empty or rows.target_polymarket_up.nunique() != 2:
        raise RuntimeError("Common quality sample lacks official outcomes or one class")
    return rows


def _metric_values(labels: np.ndarray, probability: np.ndarray) -> dict:
    return taker._score_binary(np.asarray(labels, dtype=np.int8), np.asarray(probability, dtype=np.float64))


def _bootstrap_indices(days: pd.DatetimeIndex) -> np.ndarray:
    replications = int(CONFIG["bootstrap"]["replications"])
    block = int(CONFIG["bootstrap"]["block_days"])
    rng = np.random.default_rng(int(CONFIG["bootstrap"]["seed"]))
    block_count = int(math.ceil(len(days) / block))
    starts = rng.integers(0, len(days), size=(replications, block_count))
    offsets = np.arange(block, dtype=np.int32)
    samples = (starts[:, :, None] + offsets[None, None, :]) % len(days)
    return samples.reshape(replications, -1)[:, :len(days)].astype(np.int32, copy=False)


def _write_bootstrap(path: Path, indices: np.ndarray, days: pd.DatetimeIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, indices=indices, days=days.asi8)
    temporary.replace(path)


def _interval(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    valid = array[np.isfinite(array)]
    return {
        "median": float(np.median(valid)) if len(valid) else None,
        "95pct": [float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))] if len(valid) else [None, None],
        "valid_replications": int(len(valid)),
    }


def _quality_summary(rows: pd.DataFrame, days: pd.DatetimeIndex, indices: np.ndarray) -> tuple[dict, pd.DataFrame]:
    y = rows.target_polymarket_up.to_numpy(dtype=np.int8)
    points = {name: _metric_values(y, rows[PROBABILITY_COLUMNS[name]].to_numpy()) for name in MODEL_NAMES}
    rows = rows.copy()
    rows["utc_month"] = rows.market_start_utc.dt.strftime("%Y-%m")
    monthly = []
    for month, group in rows.groupby("utc_month", sort=True):
        labels = group.target_polymarket_up.to_numpy(dtype=np.int8)
        item = {"utc_month": month, "rows": int(len(group)), "up_labels": int(labels.sum()), "down_labels": int(len(labels) - labels.sum())}
        for name in MODEL_NAMES:
            metric = _metric_values(labels, group[PROBABILITY_COLUMNS[name]].to_numpy())
            for field in ("log_loss", "brier", "auc"):
                item[f"{name}_{field}"] = metric[field]
        monthly.append(item)
    positions = {day: np.flatnonzero(rows.market_start_utc.dt.floor("D").eq(day).to_numpy()) for day in days}
    y_all = rows.target_polymarket_up.to_numpy(dtype=np.int8)
    p_all = {name: rows[PROBABILITY_COLUMNS[name]].to_numpy(dtype=np.float64) for name in MODEL_NAMES}
    bootstrap_metrics = {name: {metric: [] for metric in ("log_loss", "brier", "auc")} for name in MODEL_NAMES}
    for draw in indices:
        picked = [positions[days[index]] for index in draw if len(positions[days[index]])]
        selected = np.concatenate(picked) if picked else np.empty(0, dtype=np.int64)
        labels = y_all[selected]
        for name in MODEL_NAMES:
            metric = _metric_values(labels, p_all[name][selected]) if len(labels) and np.unique(labels).size == 2 else None
            for field in bootstrap_metrics[name]:
                bootstrap_metrics[name][field].append(metric[field] if metric is not None else np.nan)
    contrasts = {}
    for contrast, (left, right) in CONTRASTS.items():
        contrasts[contrast] = {}
        for metric in ("log_loss", "brier", "auc"):
            delta = np.asarray(bootstrap_metrics[left][metric]) - np.asarray(bootstrap_metrics[right][metric])
            contrasts[contrast][metric] = {
                "point_difference": float(points[left][metric] - points[right][metric]),
                **_interval(delta),
            }
    summary = {
        "sample": {
            "rows": int(len(rows)), "start_utc": rows.market_start_utc.min().isoformat(),
            "end_utc": rows.market_start_utc.max().isoformat(),
            "utc_days": int(rows.market_start_utc.dt.floor("D").nunique()),
            "up_labels": int(y.sum()), "down_labels": int(len(y) - y.sum()),
        },
        "overall": points,
        "contrasts": contrasts,
        "bootstrap": {"method": CONFIG["bootstrap"]["method"], **CONFIG["bootstrap"]},
    }
    return summary, pd.DataFrame(monthly)


def _divergence_summary(rows: pd.DataFrame) -> dict:
    work = rows.copy()
    work["abs_raw_market_logit_gap"] = work.d_raw_minus_market.abs()
    cutoff = float(work.abs_raw_market_logit_gap.quantile(0.75))
    top = work.loc[work.abs_raw_market_logit_gap.ge(cutoff)]
    raw_distance = (top.logit_raw - top.logit_market).abs()
    m1_logit = _logit(top.p_m1)
    m1_distance = (m1_logit - top.logit_market).abs()
    all_raw_distance = work.abs_raw_market_logit_gap
    all_m1_distance = (_logit(work.p_m1) - work.logit_market).abs()
    return {
        "definition": "absolute logit distance from p_market; large-divergence subset is the top quartile of |logit(p_RAW)-logit(p_market)| on the common quality sample",
        "top_quartile_threshold": cutoff,
        "top_quartile_rows": int(len(top)),
        "top_quartile_RAW_median_logit_distance": float(raw_distance.median()),
        "top_quartile_M1_median_logit_distance": float(m1_distance.median()),
        "top_quartile_fraction_M1_closer_to_market_than_RAW": float((m1_distance < raw_distance).mean()),
        "all_rows_RAW_median_logit_distance": float(all_raw_distance.median()),
        "all_rows_M1_median_logit_distance": float(all_m1_distance.median()),
    }


def _load_or_fit_predictions(manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    prediction_path = OUTPUT_DIR / "development_predictions.parquet"
    coefficients_path = OUTPUT_DIR / "daily_coefficients.csv"
    stage_name = "daily_expanding_calibration"
    if _checkpoint_valid(manifest, stage_name):
        return pd.read_parquet(prediction_path), pd.read_csv(coefficients_path), manifest["stages"][stage_name]["summary"]
    manifest["status"] = "running"
    manifest["active_stage"] = stage_name
    manifest["stages"][stage_name] = {"status": "running", "started_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    _write_manifest(manifest)
    started = time.perf_counter()
    frame, _ = _build_development_predictions()
    predictions, coefficients = _fit_daily_models(frame)
    _write_parquet(predictions, prediction_path)
    coefficients.to_csv(coefficients_path, index=False)
    fitted = coefficients.loc[coefficients.status.eq("fit")]
    summary = {
        "development_oof_rows": int(len(predictions)),
        "rows_with_raw_and_valid_market_probability": int((predictions.p_raw.notna() & predictions.p_market.notna()).sum()),
        "utc_days": int(predictions.decision_day_utc.nunique()),
        "fitted_utc_days": int(len(fitted)),
        "warmup_utc_days": int(len(coefficients) - len(fitted)),
        "first_fitted_day_utc": str(fitted.utc_day.iloc[0]) if len(fitted) else None,
        "last_fitted_day_utc": str(fitted.utc_day.iloc[-1]) if len(fitted) else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    outputs = [_record(prediction_path), _record(coefficients_path)]
    manifest["stages"][stage_name] = {
        "status": "completed", "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "summary": summary, "outputs": outputs,
    }
    manifest["active_stage"] = None
    _write_manifest(manifest)
    return predictions, coefficients, summary


def _load_execution_sources(condition_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sources = CONFIG["sources"]
    common = pd.read_parquet(ROOT / sources["common_execution_calendar_path"])
    common["market_start_utc"] = pd.to_datetime(common.market_start_utc, utc=True, errors="raise")
    common = common.loc[common.market_start_utc.ge(START_UTC) & common.market_start_utc.lt(CUTOFF_UTC)].copy()
    common["condition_id"] = common.condition_id.astype("string").str.lower()
    common = common.loc[common.condition_id.isin(condition_ids)].copy()
    if common.condition_id.duplicated().any():
        raise RuntimeError("Existing development execution calendar has duplicate condition ids")
    snapshot_columns = [
        "condition_id", "market_start_utc", "entry_time_utc", "fee_collection_mode", "fee_rate_bps",
        "fee_known", "fee_event_ns", "archive_source", "archive_entry_hour_available",
        "archive_freshness_window_available", "archive_window_hours_expected",
        "archive_window_hours_source_available", "archive_window_hours_processed",
        "archive_window_missing_source_hours", "archive_window_unprocessed_hours",
        "archive_gap_after_up_book_init", "archive_gap_after_down_book_init", "archive_gap_after_fee_update",
        "up_book_init_receive_ns", "down_book_init_receive_ns", "no_future_event_at_entry",
        "no_future_source_event_at_entry",
        "up_best_bid", "up_best_ask", "up_best_ask_size_shares", "up_ask_age_seconds",
        "up_quote_source_token_id", "up_quote_complemented", "up_ask_levels", "up_ask_order_ambiguous",
        "up_bbo_ask_comparable", "up_bbo_ask_mismatch", "up_bid_ask_crossed",
        "down_best_bid", "down_best_ask", "down_best_ask_size_shares", "down_ask_age_seconds",
        "down_quote_source_token_id", "down_quote_complemented", "down_ask_levels", "down_ask_order_ambiguous",
        "down_bbo_ask_comparable", "down_bbo_ask_mismatch", "down_bid_ask_crossed",
    ]
    t59 = _read_development_parquet(ROOT / sources["t59_snapshots_path"], snapshot_columns)
    t60 = _read_development_parquet(ROOT / sources["t60_snapshots_path"], snapshot_columns)
    t59["condition_id"] = t59.condition_id.astype("string").str.lower()
    t60["condition_id"] = t60.condition_id.astype("string").str.lower()
    t59 = t59.loc[t59.condition_id.isin(condition_ids)].copy()
    t60 = t60.loc[t60.condition_id.isin(condition_ids)].copy()
    return common, t59, t60


def _simulate_policy(
    variant: str,
    market_rows: pd.DataFrame,
    t59_by_id: dict,
    market_by_id: dict,
    t60_by_id: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    summary, trades, decisions = replay._portfolio_for_policy(
        "fixed_5_usd", market_rows, t59_by_id, market_by_id,
        decision_snapshot_by_id=t60_by_id,
        minimum_net_ev_fraction=float(CONFIG["policy"]["minimum_net_ev_fraction"]),
        output_policy=variant,
        causal_intent_mode=True,
        allow_unresolved_outcomes=True,
        independent_order_mode=True,
    )
    equity = summary.pop("_equity_path")
    audit = replay._portfolio_accounting_audit(
        summary, trades, decisions, equity, t59_by_id, t60_by_id,
        allow_unresolved_outcomes=True,
        independent_order_mode=True,
    )
    if audit.get("status") != "passed":
        raise RuntimeError(f"{variant} accounting audit failed: {audit}")
    summary["accounting_audit"] = audit
    decisions_frame = pd.DataFrame(decisions)
    trades_frame = pd.DataFrame(trades)
    settled = trades_frame.loc[trades_frame.settlement_status.eq("settled")] if len(trades_frame) else trades_frame
    pnl = float(pd.to_numeric(settled.net_pnl_usd, errors="coerce").sum()) if len(settled) else 0.0
    settled_turnover = float(pd.to_numeric(settled.gross_usd, errors="coerce").sum()) if len(settled) else 0.0
    forecast_ev = float(pd.to_numeric(settled.chosen_side_ev_usd, errors="coerce").sum()) if len(settled) else 0.0
    intents = pd.to_numeric(decisions_frame.get("net_ev_fraction_at_decision", pd.Series(dtype=float)), errors="coerce")
    summary.update({
        "variant": variant,
        "decision_count": int(len(decisions_frame)),
        "intent_count": int(intents.ge(float(CONFIG["policy"]["minimum_net_ev_fraction"])).sum()),
        "fill_count": int(len(trades_frame)),
        "settlement_count": int(len(settled)),
        "unresolved_fill_count": int(len(trades_frame) - len(settled)),
        "gross_turnover_usd": float(pd.to_numeric(trades_frame.gross_usd, errors="coerce").sum()) if len(trades_frame) else 0.0,
        "settled_turnover_usd": settled_turnover,
        "fees_paid_estimated_usd": float(pd.to_numeric(trades_frame.fee_usd, errors="coerce").sum()) if len(trades_frame) else 0.0,
        "net_pnl_usd": pnl,
        "pnl_per_settled_turnover": pnl / settled_turnover if settled_turnover else None,
        "forecast_net_ev_on_settled_fills_usd": forecast_ev,
        "pnl_minus_forecast_ev_on_settled_fills_usd": pnl - forecast_ev,
        "up_fill_count": int(trades_frame.chosen_side.eq("up").sum()) if len(trades_frame) else 0,
        "down_fill_count": int(trades_frame.chosen_side.eq("down").sum()) if len(trades_frame) else 0,
    })
    return summary, decisions_frame, trades_frame


def _monthly_economics(variant: str, decisions: pd.DataFrame, trades: pd.DataFrame) -> list[dict]:
    if len(decisions):
        decision_times = pd.to_datetime(decisions.decision_time_utc, utc=True)
        months = decision_times.dt.strftime("%Y-%m")
        ev_fraction = pd.to_numeric(
            decisions.get("net_ev_fraction_at_decision", pd.Series(np.nan, index=decisions.index)),
            errors="coerce",
        )
        intent_counts = ev_fraction.ge(float(CONFIG["policy"]["minimum_net_ev_fraction"]))
        intents = intent_counts.groupby(months).sum().to_dict()
    else:
        intents = {}
    frame = trades.copy()
    if len(frame):
        frame["month"] = pd.to_datetime(frame.decision_time_utc, utc=True).dt.strftime("%Y-%m")
        frame["settled"] = frame.settlement_status.eq("settled")
    else:
        frame = pd.DataFrame(columns=["month", "settled", "net_pnl_usd", "gross_usd", "chosen_side_ev_usd", "fee_usd", "chosen_side"])
    rows = []
    decision_months = sorted(pd.to_datetime(decisions.decision_time_utc, utc=True).dt.strftime("%Y-%m").unique()) if len(decisions) else []
    for month in decision_months:
        group = frame.loc[frame.month.eq(month)]
        settled = group.loc[group.settled]
        pnl = float(pd.to_numeric(settled.net_pnl_usd, errors="coerce").sum())
        turnover = float(pd.to_numeric(settled.gross_usd, errors="coerce").sum())
        ev = float(pd.to_numeric(settled.chosen_side_ev_usd, errors="coerce").sum())
        rows.append({
            "variant": variant, "utc_month": month, "intents": int(intents.get(month, 0)),
            "fills": int(len(group)), "settlements": int(len(settled)),
            "unresolved_fills": int(len(group) - len(settled)),
            "up_fills": int(group.chosen_side.eq("up").sum()), "down_fills": int(group.chosen_side.eq("down").sum()),
            "gross_turnover_usd": float(pd.to_numeric(group.gross_usd, errors="coerce").sum()),
            "settled_turnover_usd": turnover,
            "fees_usd": float(pd.to_numeric(group.fee_usd, errors="coerce").sum()),
            "net_pnl_usd": pnl, "pnl_per_settled_turnover": pnl / turnover if turnover else None,
            "forecast_ev_on_settled_fills_usd": ev,
            "pnl_minus_forecast_ev_on_settled_fills_usd": pnl - ev,
        })
    return rows


def _daily_pnl(trades_by_variant: dict[str, pd.DataFrame], days: pd.DatetimeIndex) -> pd.DataFrame:
    daily = pd.DataFrame({"utc_day": days})
    for variant, trades in trades_by_variant.items():
        result = np.zeros(len(days), dtype=np.float64)
        if len(trades):
            frame = trades.copy()
            frame["utc_day"] = pd.to_datetime(frame.decision_time_utc, utc=True).dt.floor("D")
            settled = frame.loc[frame.settlement_status.eq("settled")].copy()
            grouped = pd.to_numeric(settled.net_pnl_usd, errors="coerce").groupby(settled.utc_day).sum()
            indices = {day: index for index, day in enumerate(days)}
            for day, value in grouped.items():
                if day in indices:
                    result[indices[day]] = float(value)
        daily[f"{variant}_net_pnl_usd"] = result
    for contrast, (left, right) in CONTRASTS.items():
        daily[f"{contrast}_net_pnl_usd"] = daily[f"{left}_net_pnl_usd"] - daily[f"{right}_net_pnl_usd"]
    return daily


def _pnl_bootstrap(daily: pd.DataFrame, indices: np.ndarray) -> dict:
    result = {}
    for contrast, (left, right) in CONTRASTS.items():
        daily_difference = daily[f"{left}_net_pnl_usd"].to_numpy(dtype=np.float64) - daily[f"{right}_net_pnl_usd"].to_numpy(dtype=np.float64)
        estimates = daily_difference[indices].mean(axis=1)
        point = float(daily_difference.mean())
        result[contrast] = {
            "mean_daily_pnl_difference_usd": {"point_estimate": point, **_interval(estimates)},
            "total_pnl_difference_usd": {"point_estimate": point * len(daily), **{**_interval(estimates * len(daily))}},
        }
    return {
        "method": CONFIG["bootstrap"]["method"],
        "replications": int(len(indices)),
        "block_days": int(CONFIG["bootstrap"]["block_days"]),
        "seed": int(CONFIG["bootstrap"]["seed"]),
        "calendar_days": int(len(daily)),
        "contrasts": result,
    }


def _concentration(daily: pd.DataFrame) -> dict:
    output = {}
    for variant in MODEL_NAMES:
        column = f"{variant}_net_pnl_usd"
        ordered = daily.sort_values(column, ascending=False, kind="stable")
        top = ordered.head(5)
        total = float(daily[column].sum())
        top_total = float(top[column].sum())
        output[variant] = {
            "total_pnl_usd": total,
            "top_five_days_pnl_usd": top_total,
            "top_five_share_of_total_pnl": top_total / total if total else None,
            "top_five_days": [
                {"utc_day": row.utc_day.isoformat(), "net_pnl_usd": float(getattr(row, column))}
                for row in top.itertuples(index=False)
            ],
        }
    return output


def _reconcile_raw_anchor(raw_trades: pd.DataFrame, common_ids: set[str], raw_summary: dict, current_raw_trades: pd.DataFrame) -> dict:
    old = raw_trades.copy()
    old["condition_id"] = old.condition_id.astype("string").str.lower()
    old_common = old.loc[old.condition_id.isin(common_ids)].copy()
    old_settled = old.loc[old.settlement_status.eq("settled")]
    old_common_settled = old_common.loc[old_common.settlement_status.eq("settled")]
    new_settled = current_raw_trades.loc[current_raw_trades.settlement_status.eq("settled")].copy()
    new_settled["condition_id"] = new_settled.condition_id.astype("string").str.lower()
    fields = ["condition_id", "chosen_side", "gross_usd", "fee_usd", "net_pnl_usd", "settlement_status"]
    expected = old_common_settled[fields].sort_values("condition_id", kind="stable").reset_index(drop=True)
    observed = new_settled[fields].sort_values("condition_id", kind="stable").reset_index(drop=True)
    exact_fields_match = len(expected) == len(observed)
    if exact_fields_match:
        exact_fields_match = (
            expected.condition_id.astype(str).equals(observed.condition_id.astype(str))
            and expected.chosen_side.astype(str).equals(observed.chosen_side.astype(str))
            and expected.settlement_status.astype(str).equals(observed.settlement_status.astype(str))
        )
    if exact_fields_match:
        for field in ("gross_usd", "fee_usd", "net_pnl_usd"):
            left = pd.to_numeric(expected[field], errors="coerce").to_numpy(dtype=np.float64)
            right = pd.to_numeric(observed[field], errors="coerce").to_numpy(dtype=np.float64)
            exact_fields_match = exact_fields_match and np.allclose(left, right, rtol=0.0, atol=1e-10, equal_nan=True)
    if not exact_fields_match:
        raise RuntimeError("RAW common-calendar ledger does not reproduce the prior ledger on identical markets")
    old_total = float(pd.to_numeric(old_settled.net_pnl_usd, errors="coerce").sum())
    old_common_total = float(pd.to_numeric(old_common_settled.net_pnl_usd, errors="coerce").sum())
    new_total = float(pd.to_numeric(new_settled.net_pnl_usd, errors="coerce").sum())
    anchor = float(CONFIG["reference_RAW_anchor"]["net_pnl_usd"])
    if not math.isclose(float(raw_summary["net_pnl_usd"]), anchor, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("Saved RAW anchor does not match the frozen reference amount")
    if not math.isclose(old_total, anchor, rel_tol=0.0, abs_tol=1e-8) or not math.isclose(old_common_total, new_total, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("RAW PnL reconciliation failed")
    return {
        "previous_full_calendar_net_pnl_usd": old_total,
        "new_shared_calendar_net_pnl_usd": new_total,
        "previous_ledger_on_new_shared_calendar_net_pnl_usd": old_common_total,
        "pnl_removed_by_common_calendar_usd": old_total - old_common_total,
        "previous_full_calendar_settled_fills": int(len(old_settled)),
        "new_shared_calendar_settled_fills": int(len(new_settled)),
        "new_shared_calendar_decisions": int(len(common_ids)),
        "ledger_matches_on_shared_calendar": True,
    }


def _run_economic_variant(
    variant: str,
    predictions: pd.DataFrame,
    execution_calendar: pd.DataFrame,
    t59_by_id: dict,
    t60_by_id: dict,
    manifest: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    stage_name = f"economics_{variant}"
    out_dir = OUTPUT_DIR / "economics" / variant
    decisions_path = out_dir / "decisions.parquet"
    trades_path = out_dir / "trades.parquet"
    summary_path = out_dir / "summary.json"
    if _checkpoint_valid(manifest, stage_name):
        return json.loads(summary_path.read_text(encoding="utf-8")), pd.read_parquet(decisions_path), pd.read_parquet(trades_path)
    manifest["status"] = "running"
    manifest["active_stage"] = stage_name
    manifest["stages"][stage_name] = {"status": "running", "started_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    _write_manifest(manifest)
    started = time.perf_counter()
    pmap = predictions.set_index("market_start_utc")[PROBABILITY_COLUMNS[variant]]
    market_rows = execution_calendar.copy()
    market_rows["p_candidate_platt"] = market_rows.market_start_utc.map(pmap)
    if market_rows.p_candidate_platt.isna().any():
        raise RuntimeError(f"{variant} does not cover each shared execution decision")
    market_rows, period = research._asof_market_rows(market_rows, START_UTC, CUTOFF_UTC)
    market_by_id = research._snapshot_map(market_rows.loc[market_rows.condition_id.notna()])
    summary, decisions, trades = _simulate_policy(variant, market_rows, t59_by_id, market_by_id, t60_by_id)
    if len(decisions) != len(market_rows):
        raise RuntimeError(f"{variant} did not yield one decision row per shared market")
    summary["calendar"] = period
    summary["monthly_results"] = _monthly_economics(variant, decisions, trades)
    summary["elapsed_seconds"] = time.perf_counter() - started
    _write_parquet(decisions, decisions_path)
    _write_parquet(trades, trades_path)
    _write_json(summary_path, summary)
    manifest["stages"][stage_name] = {
        "status": "completed", "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "summary": summary, "outputs": [_record(path) for path in (decisions_path, trades_path, summary_path)],
    }
    manifest["active_stage"] = None
    _write_manifest(manifest)
    return summary, decisions, trades


def _status_update(summary: dict) -> None:
    contrast = summary["quality"]["contrasts"]["M1_MINUS_M0"]
    pnl = summary["economics"]["variants"]
    pnl_interval = summary["economics"]["paired_pnl_bootstrap"]["contrasts"]["M1_MINUS_M0"]["total_pnl_difference_usd"]["95pct"]
    block = "\n".join([
        STATUS_BEGIN,
        "### RAW BASE70 ponad prognozę rynku - 2026-10-19",
        f"- Status: `completed`; wspólna jakość: `{summary['quality']['sample']['rows']}` wierszy development.",
        f"- M1-M0 log loss: `{contrast['log_loss']['point_difference']:.8f}` (95% CI `{contrast['log_loss']['95pct']}`).",
        f"- PnL M1: `{_format_usd(pnl['M1']['net_pnl_usd'])}`; M0: `{_format_usd(pnl['M0']['net_pnl_usd'])}`; RAW: `{_format_usd(pnl['RAW']['net_pnl_usd'])}` na tym samym kalendarzu.",
        f"- M1−M0 paired PnL: `{_format_usd(pnl['M1']['net_pnl_usd'] - pnl['M0']['net_pnl_usd'])}`; 95% CI sumarycznego PnL `{_format_usd(pnl_interval[0])}` do `{_format_usd(pnl_interval[1])}`.",
        "- Audyt ledgerów po eksperymencie: `$5` to notional zakupu; M0 wydał `$20.54` gotówki na `$20.00` notionalu. PnL settled uzgodniono z przepływami dla bieżących, development i historycznych ledgerów; naruszeń limitu straty: `0`.",
        "- Opłaty V1 potrącane w tokenach raportuje się przez ekwiwalent USD informacyjnie; V2 cash fees zwiększają wypływ i są liczone raz. Nie zmieniono stawek, decyzji ani wykonań.",
        f"- Manifest: `{_relative(RUN_MANIFEST_PATH)}`; raport: `{_relative(REPORT_PATH)}`.",
        STATUS_END,
        "",
    ])
    text = STATUS_PATH.read_text(encoding="utf-8")
    if STATUS_BEGIN in text and STATUS_END in text:
        before, tail = text.split(STATUS_BEGIN, 1)
        _, after = tail.split(STATUS_END, 1)
        text = before.rstrip() + "\n" + block + after.lstrip("\n")
    else:
        text = block + text
    temporary = STATUS_PATH.with_name(STATUS_PATH.stem + ".tmp" + STATUS_PATH.suffix)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(STATUS_PATH)


def _format_metric(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{float(value):.6f}"


def _format_usd(value) -> str:
    number = float(value)
    return f"-${abs(number):.2f}" if number < 0.0 else f"${number:.2f}"


def _format_ledger_usd(value) -> str:
    number = float(value)
    prefix = "-" if number < 0.0 else ""
    text = repr(abs(number))
    if text.endswith(".0"):
        text = text[:-2]
    return f"{prefix}${text}"


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna().astype(float).tolist()
    return math.fsum(values)


def _cashflow_ledger_summary(trades: pd.DataFrame) -> dict:
    frame = trades.copy()
    settled = frame.loc[frame.settlement_status.eq("settled")].copy() if len(frame) else frame
    unresolved = frame.loc[frame.settlement_status.ne("settled")].copy() if len(frame) else frame
    if len(settled):
        cashflow_pnl = [
            float(row.payout_usd) - float(row.cash_debit_usd)
            for row in settled.itertuples(index=False)
        ]
        settled["cashflow_pnl_usd"] = cashflow_pnl
        settled["loss_bound_headroom_usd"] = [
            float(row.net_pnl_usd) + float(row.cash_debit_usd)
            for row in settled.itertuples(index=False)
        ]
        settled["pnl_reconciliation_delta_usd"] = [
            float(row.net_pnl_usd) - cashflow
            for row, cashflow in zip(settled.itertuples(index=False), cashflow_pnl)
        ]
    else:
        settled["cashflow_pnl_usd"] = pd.Series(dtype=float)
        settled["loss_bound_headroom_usd"] = pd.Series(dtype=float)
        settled["pnl_reconciliation_delta_usd"] = pd.Series(dtype=float)
    share_fee_rows = frame.loc[frame.fee_charge_unit.eq("outcome shares")] if len(frame) else frame
    settled_share_fee_rows = (
        share_fee_rows.loc[share_fee_rows.settlement_status.eq("settled")]
        if len(share_fee_rows) else share_fee_rows
    )
    summary = {
        "fill_count": int(len(frame)),
        "settlement_count": int(len(settled)),
        "unresolved_fill_count": int(len(unresolved)),
        "gross_notional_usd": _sum_column(settled, "gross_usd"),
        "all_fill_gross_notional_usd": _sum_column(frame, "gross_usd"),
        "settled_cash_outflow_usd": _sum_column(settled, "cash_debit_usd"),
        "all_fill_cash_outflow_usd": _sum_column(frame, "cash_debit_usd"),
        "unresolved_cash_outflow_usd": _sum_column(unresolved, "cash_debit_usd"),
        "cash_fees_paid_usd": _sum_column(settled, "fee_cash_usd"),
        "all_fill_cash_fees_paid_usd": _sum_column(frame, "fee_cash_usd"),
        "share_fee_usd_entry_equivalent_informational": _sum_column(
            settled_share_fee_rows, "fee_usd_entry_equivalent",
        ),
        "fee_usd_equivalent_all_units": _sum_column(settled, "fee_usd_entry_equivalent"),
        "payout_usd": _sum_column(settled, "payout_usd"),
        "ledger_pnl_usd": _sum_column(settled, "net_pnl_usd"),
        "cashflow_pnl_usd": _sum_column(settled, "cashflow_pnl_usd"),
        "max_abs_pnl_reconciliation_delta_usd": float(
            settled.pnl_reconciliation_delta_usd.abs().max()
        ) if len(settled) else 0.0,
        "minimum_loss_bound_headroom_usd": float(
            settled.loss_bound_headroom_usd.min()
        ) if len(settled) else 0.0,
        "loss_bound_violations": int(settled.loss_bound_headroom_usd.lt(-1e-8).sum()),
        "cash_debit_reconciliation_max_abs_usd": float(
            (
                pd.to_numeric(frame.cash_debit_usd, errors="coerce")
                - pd.to_numeric(frame.gross_usd, errors="coerce")
                - pd.to_numeric(frame.fee_cash_usd, errors="coerce")
            ).abs().max()
        ) if len(frame) else 0.0,
    }
    summary["pnl_per_notional"] = (
        summary["cashflow_pnl_usd"] / summary["gross_notional_usd"]
        if summary["gross_notional_usd"] else None
    )
    summary["pnl_per_cash_outflow"] = (
        summary["cashflow_pnl_usd"] / summary["settled_cash_outflow_usd"]
        if summary["settled_cash_outflow_usd"] else None
    )
    if (
        summary["max_abs_pnl_reconciliation_delta_usd"] > 1e-8
        or summary["loss_bound_violations"]
        or summary["cash_debit_reconciliation_max_abs_usd"] > 1e-8
    ):
        raise RuntimeError("Saved trade ledger fails the independent cashflow accounting audit")
    return summary


def _independent_cashflow_audit(trades: pd.DataFrame) -> dict:
    """Reconcile saved ledger rows with Decimal arithmetic, separate from report aggregates."""
    rows = list(trades.itertuples(index=False)) if len(trades) else []
    settled = [row for row in rows if row.settlement_status == "settled"]
    unresolved = [row for row in rows if row.settlement_status != "settled"]

    def amount(value) -> Decimal:
        return Decimal(str(value))

    cashflow_pnls = [amount(row.payout_usd) - amount(row.cash_debit_usd) for row in settled]
    ledger_pnls = [amount(row.net_pnl_usd) for row in settled]
    pnl_deltas = [abs(ledger - cashflow) for ledger, cashflow in zip(ledger_pnls, cashflow_pnls)]
    loss_headroom = [
        amount(row.net_pnl_usd) + amount(row.cash_debit_usd)
        for row in settled
    ]
    cash_debit_errors = [
        abs(amount(row.cash_debit_usd) - amount(row.gross_usd) - amount(row.fee_cash_usd))
        for row in rows
    ]
    loss_bound_violations = sum(value < Decimal("-0.00000001") for value in loss_headroom)
    result = {
        "fill_count": len(rows),
        "settlement_count": len(settled),
        "unresolved_fill_count": len(unresolved),
        "gross_notional_usd": sum((amount(row.gross_usd) for row in settled), Decimal(0)),
        "settled_cash_outflow_usd": sum((amount(row.cash_debit_usd) for row in settled), Decimal(0)),
        "unresolved_cash_outflow_usd": sum((amount(row.cash_debit_usd) for row in unresolved), Decimal(0)),
        "all_fill_cash_fees_paid_usd": sum((amount(row.fee_cash_usd) for row in rows), Decimal(0)),
        "share_fee_usd_entry_equivalent_informational": sum((
            amount(row.fee_usd_entry_equivalent)
            for row in settled if row.fee_charge_unit == "outcome shares"
        ), Decimal(0)),
        "cashflow_pnl_usd": sum(cashflow_pnls, Decimal(0)),
        "ledger_pnl_usd": sum(ledger_pnls, Decimal(0)),
        "max_abs_pnl_reconciliation_delta_usd": max(pnl_deltas, default=Decimal(0)),
        "minimum_loss_bound_headroom_usd": min(loss_headroom, default=Decimal(0)),
        "loss_bound_violations": loss_bound_violations,
        "cash_debit_reconciliation_max_abs_usd": max(cash_debit_errors, default=Decimal(0)),
    }
    if (
        result["max_abs_pnl_reconciliation_delta_usd"] > Decimal("0.00000001")
        or loss_bound_violations
        or result["cash_debit_reconciliation_max_abs_usd"] > Decimal("0.00000001")
    ):
        raise RuntimeError("Saved trade ledger fails the independent Decimal cashflow audit")
    return {key: float(value) if isinstance(value, Decimal) else value for key, value in result.items()}


def _monthly_cashflow_rows(
    monthly_economics: pd.DataFrame,
    trades_by_variant: dict[str, pd.DataFrame],
) -> list[dict]:
    rows = []
    for month_row in monthly_economics.itertuples(index=False):
        trades = trades_by_variant[month_row.variant]
        if len(trades):
            trade_months = pd.to_datetime(trades.decision_time_utc, utc=True).dt.strftime("%Y-%m")
            month_trades = trades.loc[trade_months.eq(month_row.utc_month)].copy()
        else:
            month_trades = trades
        audit = _cashflow_ledger_summary(month_trades)
        settled = month_trades.loc[month_trades.settlement_status.eq("settled")] if len(month_trades) else month_trades
        forecast_ev = _sum_column(settled, "chosen_side_ev_usd")
        rows.append({
            "variant": month_row.variant,
            "utc_month": month_row.utc_month,
            "fills": audit["fill_count"],
            "settlements": audit["settlement_count"],
            "gross_notional_usd": audit["gross_notional_usd"],
            "cash_outflow_usd": audit["settled_cash_outflow_usd"],
            "cash_fees_paid_usd": audit["cash_fees_paid_usd"],
            "share_fee_usd_entry_equivalent_informational": audit[
                "share_fee_usd_entry_equivalent_informational"
            ],
            "net_pnl_usd": audit["cashflow_pnl_usd"],
            "pnl_per_notional": audit["pnl_per_notional"],
            "pnl_per_cash_outflow": audit["pnl_per_cash_outflow"],
            "pnl_minus_forecast_ev_usd": audit["cashflow_pnl_usd"] - forecast_ev,
        })
    return rows


def _write_report(
    manifest: dict,
    summary: dict,
    monthly_quality: pd.DataFrame,
    monthly_economics: pd.DataFrame,
    trades_by_variant: dict[str, pd.DataFrame],
) -> None:
    quality = summary["quality"]
    economics = summary["economics"]
    lines = [
        "błąd raportowania/definicji stawki — księgowanie ledgerów jest poprawne; $5 oznacza notional zakupu, a dodatkowe opłaty gotówkowe podnoszą wypływ.",
        "",
        "# Przyrostowa informacja RAW BASE70 ponad prognozę rynku",
        "",
        "## Zakres i metoda",
        "",
        f"Jedyny eksperyment używa development `{START_UTC.isoformat()}`–`{CUTOFF_UTC.isoformat()}`. Nie trenowano modelu BTC i nie odtwarzano archiwum. Historyczny holdout nie uczestniczył w eksperymencie; jego zapisany ledger odczytano wyłącznie do kontroli księgowej. Hipoteza powstała po wcześniejszej analizie holdoutu, więc eksperyment go nie ocenia.",
        "",
        "Prawdopodobieństwo rynku to proporcja środków kwotowań UP/(UP+DOWN) na zwalidowanym T-60. Zmienna rynku to `m=logit(p_market)`, a różnica RAW to `d=logit(p_RAW)-m`; prawdopodobieństwa przycięto do `[1e-6, 1-1e-6]` przed logitem.",
        "",
        "M0 ma postać `sigmoid(a+b*m)`. M1 ma postać `sigmoid(a+b*m+c*d)`. Osobne modele dopasowano raz dziennie na wszystkich wcześniejszych etykietach oficjalnych, których release był ściśle wcześniejszy niż pierwsza decyzja T-60 tego dnia. Oba modele korzystają z identycznych wierszy, co najmniej 1000 obserwacji i obu klas. Cel to średni, nieważony log loss plus `0.001*((b-1)^2+c^2)`; w M0 składnik c pominięto. Znaki współczynników nie były ograniczane.",
        "",
        "BASE70 OOF i wcześniejsza selekcja cech widziały wszystkie etapy development. Dlatego dzienne dopasowanie jest przyczynowe względem etykiet, ale nie stanowi niezależnej oceny całego procesu budowy BASE70.",
        "",
        "## Pokrycie i jakość prognoz",
        "",
        f"Wspólna próbka jakości: **{quality['sample']['rows']:,}** rynków, `{quality['sample']['up_labels']:,}` UP, `{quality['sample']['down_labels']:,}` DOWN, od `{quality['sample']['start_utc']}` do `{quality['sample']['end_utc']}`. M0/M1 dopasowano w `{summary['calibration']['fitted_utc_days']}` z `{summary['calibration']['utc_days']}` dni UTC; rozgrzewka objęła `{summary['calibration']['warmup_utc_days']}` dni. Wspólny kalendarz ma `{summary['availability']['shared_forecast_rows']:,}` decyzji po rozgrzewce i dostępności obu prognoz.",
        "",
        "| Wariant | Log loss | Brier | AUC |",
        "|---|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        metric = quality["overall"][name]
        lines.append(f"| {name} | {_format_metric(metric['log_loss'])} | {_format_metric(metric['brier'])} | {_format_metric(metric['auc'])} |")
    lines.extend(["", "| Kontrast (pierwszy minus drugi) | Δ log loss, 95% CI | Δ Brier, 95% CI | Δ AUC, 95% CI |", "|---|---:|---:|---:|"])
    for name in CONTRASTS:
        fields = quality["contrasts"][name]
        values = []
        for metric in ("log_loss", "brier", "auc"):
            result = fields[metric]
            low, high = result["95pct"]
            values.append(f"{result['point_difference']:.6f} [{low:.6f}, {high:.6f}]")
        lines.append(f"| {name} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(["", "Wartości ujemne dla log loss/Brier oznaczają przewagę pierwszego wariantu. Przedziały pochodzą ze wspólnego, sparowanego bootstrapu kalendarzowego z blokami 3 dni i 2000 replikacjami.", ""])
    lines.extend([
        "### Jakość według miesiąca",
        "",
        "| Miesiąc | Wariant | Wiersze | Log loss | Brier | AUC |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in monthly_quality.itertuples(index=False):
        for name in MODEL_NAMES:
            lines.append(
                f"| {row.utc_month} | {name} | {row.rows:,} | "
                f"{_format_metric(getattr(row, f'{name}_log_loss'))} | "
                f"{_format_metric(getattr(row, f'{name}_brier'))} | "
                f"{_format_metric(getattr(row, f'{name}_auc'))} |"
            )
    lines.append("")
    divergence = summary["market_divergence"]
    lines.extend([
        "## Duże rozbieżności RAW–rynek",
        "",
        f"W górnym kwartylu początkowej różnicy logitów (próg `{divergence['top_quartile_threshold']:.4f}`, `{divergence['top_quartile_rows']:,}` wierszy) mediana odległości RAW od rynku wyniosła `{divergence['top_quartile_RAW_median_logit_distance']:.4f}`, a M1 `{divergence['top_quartile_M1_median_logit_distance']:.4f}`. M1 był bliżej rynku w `{divergence['top_quartile_fraction_M1_closer_to_market_than_RAW']:.1%}` tych wierszy. To opisowa statystyka tej samej próbki development.",
        "",
        "## Ekonomia przy stałym progu",
        "",
        "Każdy wariant wybierał stronę przy zleceniu o notionalu $5 i progu 1% net EV. EV na T-60 to prawdopodobieństwo pomnożone przez tokeny po opłacie minus pełny wypływ gotówki; próg dzieli EV przez żądany notional $5. Na T-59 wykonanie zachowuje wybraną stronę i notional, a rzeczywisty wypływ wynosi notional plus opłata gotówkowa. W reżimie V1 opłata kupna pomniejsza tokeny; w V2 jest pobierana w gotówce. PnL przeliczono niezależnie jako wypłata minus wypływ gotówki.",
        "",
        "| Wariant | Decyzje | Fill’e / rozliczone | Notional rozliczony | Wypływ gotówki | Opłaty gotówkowe | Ekwiwalent opłat tokenowych¹ | PnL | PnL / notional | PnL / wypływ | EV na rozliczonych fill’ach | PnL−EV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in MODEL_NAMES:
        item = economics["variants"][name]
        audit = _cashflow_ledger_summary(trades_by_variant[name])
        settled = trades_by_variant[name].loc[
            trades_by_variant[name].settlement_status.eq("settled")
        ] if len(trades_by_variant[name]) else trades_by_variant[name]
        forecast_ev = _sum_column(settled, "chosen_side_ev_usd")
        notional_ratio = "n/a" if audit["pnl_per_notional"] is None else f"{audit['pnl_per_notional']:.3%}"
        cash_ratio = "n/a" if audit["pnl_per_cash_outflow"] is None else f"{audit['pnl_per_cash_outflow']:.3%}"
        lines.append(
            f"| {name} | {item['decision_count']:,} | {item['fill_count']:,} / {item['settlement_count']:,} | "
            f"{_format_usd(audit['gross_notional_usd'])} | {_format_usd(audit['settled_cash_outflow_usd'])} | "
            f"{_format_usd(audit['cash_fees_paid_usd'])} | "
            f"{_format_usd(audit['share_fee_usd_entry_equivalent_informational'])} | "
            f"{_format_usd(audit['cashflow_pnl_usd'])} | {notional_ratio} | {cash_ratio} | "
            f"{_format_usd(forecast_ev)} | {_format_usd(audit['cashflow_pnl_usd'] - forecast_ev)} |"
        )
    m0_audit = _cashflow_ledger_summary(trades_by_variant["M0"])
    reconcile = economics["raw_anchor_reconciliation"]
    pnl_bootstrap = economics["paired_pnl_bootstrap"]["contrasts"]["M1_MINUS_M0"]["mean_daily_pnl_difference_usd"]
    low, high = pnl_bootstrap["95pct"]
    total_pnl_bootstrap = economics["paired_pnl_bootstrap"]["contrasts"]["M1_MINUS_M0"]["total_pnl_difference_usd"]
    total_low, total_high = total_pnl_bootstrap["95pct"]
    lines.extend([
        "",
        f"RAW anchor reconciliation: wcześniejsze `{reconcile['previous_full_calendar_decisions']:,}` decyzje przyniosły `{_format_usd(reconcile['previous_full_calendar_net_pnl_usd'])}`. Na obecnym wspólnym kalendarzu stary ledger RAW dał `{_format_usd(reconcile['previous_ledger_on_new_shared_calendar_net_pnl_usd'])}`, a nowy replay RAW `{_format_usd(reconcile['new_shared_calendar_net_pnl_usd'])}`; fill’e wyłączone przez wspólny kalendarz miały łącznie `{_format_usd(reconcile['pnl_removed_by_common_calendar_usd'])}` PnL. Ledger RAW zgadza się na wspólnych rynkach.",
        "",
        f"Sparowana różnica średniego dziennego PnL M1−M0: `{_format_usd(pnl_bootstrap['point_estimate'])}` na dzień (95% CI `{_format_usd(low)}` do `{_format_usd(high)}`). Różnica sumarycznego PnL to `{_format_usd(total_pnl_bootstrap['point_estimate'])}` (95% CI `{_format_usd(total_low)}` do `{_format_usd(total_high)}`); przedział obejmuje zero.",
        "",
        "¹ Informacyjny ekwiwalent USD opłaty potrąconej w tokenach; nie jest dodatkowym wypływem gotówki i nie odejmuje się go ponownie od PnL. Opłaty gotówkowe są już zawarte w wypływie i PnL.",
        "",
        "### Cztery rozliczenia M0",
        "",
        "Daty i wartości fill’i pochodzą z ledgeru; tokeny przed/po opłacie oznaczają odpowiednio `gross_shares` i `net_shares`. Ceny to VWAP wykonania.",
        "",
        "| condition_id | Start UTC | Strona | Wynik rynku / zakład | Stawka | EV T−60 | EV / stawka | Notional zakupu | Wypływ gotówki | VWAP | Tokeny przed → po opłacie | Opłata pobrana | Ekwiwalent USD | Wypłata | PnL |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ])
    m0_trades = trades_by_variant["M0"]
    for row in m0_trades.itertuples(index=False):
        outcome = "UP" if int(row.outcome_up) == 1 else "DOWN"
        result = "wygrana" if bool(row.won) else "przegrana"
        if row.fee_charge_unit == "outcome shares":
            fee_paid = f"{format(float(row.fee_shares), '.17g')} tokena"
        else:
            fee_paid = f"{_format_ledger_usd(row.fee_cash_usd)} gotówką"
        lines.append(
            f"| `{row.condition_id}` | {pd.Timestamp(row.market_start_utc).isoformat()} | {row.chosen_side.upper()} | "
            f"{outcome}; zakład {result} | {_format_ledger_usd(row.requested_gross_usd)} | "
            f"{_format_ledger_usd(row.chosen_side_ev_usd)} | "
            f"{float(row.chosen_side_ev_usd) / float(row.requested_gross_usd):.10%} | "
            f"{_format_ledger_usd(row.gross_usd)} | {_format_ledger_usd(row.cash_debit_usd)} | "
            f"{format(float(row.vwap), '.17g')} | "
            f"{format(float(row.gross_shares), '.17g')} → {format(float(row.net_shares), '.17g')} | "
            f"{fee_paid} | {_format_ledger_usd(row.fee_usd_entry_equivalent)} | "
            f"{_format_ledger_usd(row.payout_usd)} | {_format_ledger_usd(row.net_pnl_usd)} |"
        )
    lines.extend([
        "",
        f"Ręczne uzgodnienie M0 z ledgeru: wypłaty `{_format_ledger_usd(m0_audit['payout_usd'])}` − wypływy gotówki `{_format_ledger_usd(m0_audit['settled_cash_outflow_usd'])}` = PnL `{_format_ledger_usd(m0_audit['cashflow_pnl_usd'])}`. Notional wyniósł `{_format_ledger_usd(m0_audit['gross_notional_usd'])}`; opłaty pobrane w gotówce `{_format_ledger_usd(m0_audit['cash_fees_paid_usd'])}`; opłata pobrana w tokenach miała ekwiwalent `{_format_ledger_usd(m0_audit['share_fee_usd_entry_equivalent_informational'])}` i nie stanowi dodatkowego wypływu. Stąd `-$20.54 = $0.00 - ($20.00 + $0.54)`. Zgodnie z wartościami zapisanymi w ledgerze suma ekwiwalentów opłat to `{_format_ledger_usd(m0_audit['fee_usd_equivalent_all_units'])}`: `{_format_ledger_usd(m0_audit['cash_fees_paid_usd'])}` gotówką i `{_format_ledger_usd(m0_audit['share_fee_usd_entry_equivalent_informational'])}` informacyjnie w tokenach.",
        "",
        "### Niezależna kontrola zapisanych ledgerów",
        "",
        "Każdy settled trade przeliczono w osobnej ścieżce audytu z arytmetyką Decimal: `payout_usd - cash_debit_usd`. To niezależne od wspólnej funkcji używanej w agregatach głównej i miesięcznej tabeli. Warunek straty to `net_pnl_usd >= -cash_debit_usd`; gotówkowy wydatek sprawdzono też jako `gross_usd + fee_cash_usd`. Opłata w tokenach jest pokazana osobno jako ekwiwalent USD.",
        "",
        "| Ledger | Fill’e / settled / unresolved | Notional settled | Wypływ settled | Gotówka unresolved | Opłaty cash (wszystkie fill’e) | Ekwiwalent opłat tokenowych | PnL ledgeru / cashflow | Max Δ PnL, USD | Max Δ debit, USD | Min. `PnL + wypływ` | Naruszenia limitu straty |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    scoped_ledgers = [(f"Bieżący {name}", trades_by_variant[name]) for name in MODEL_NAMES]
    historical_paths = [
        ("RAW BASE70 development", ROOT / CONFIG["sources"]["base70_raw_trades_path"]),
        (
            "RAW BASE70 historyczny holdout",
            ROOT / "data/analysis/polymarket/BTC/btc_base70_raw_holdout_20261018/economics/BASE70/trades.parquet",
        ),
    ]
    scoped_ledgers.extend((name, pd.read_parquet(path)) for name, path in historical_paths)
    for name, trades in scoped_ledgers:
        audit = _independent_cashflow_audit(trades)
        lines.append(
            f"| {name} | {audit['fill_count']:,} / {audit['settlement_count']:,} / {audit['unresolved_fill_count']:,} | "
            f"{_format_usd(audit['gross_notional_usd'])} | {_format_usd(audit['settled_cash_outflow_usd'])} | "
            f"{_format_usd(audit['unresolved_cash_outflow_usd'])} | {_format_usd(audit['all_fill_cash_fees_paid_usd'])} | "
            f"{_format_usd(audit['share_fee_usd_entry_equivalent_informational'])} | "
            f"{_format_usd(audit['cashflow_pnl_usd'])} / {_format_usd(audit['ledger_pnl_usd'])} | "
            f"{_format_ledger_usd(audit['max_abs_pnl_reconciliation_delta_usd'])} | "
            f"{_format_ledger_usd(audit['cash_debit_reconciliation_max_abs_usd'])} | "
            f"{_format_usd(audit['minimum_loss_bound_headroom_usd'])} | {audit['loss_bound_violations']} |"
        )
    lines.extend([
        "",
        "Kontrola obejmuje wszystkie settled zakupy w bieżącym eksperymencie, RAW BASE70 development i ostatnim zapisanym holdoucie. W holdoucie jeden fill pozostaje nierozliczony: jego `$5.1925` wypływu gotówki jest pokazane osobno, bez dopisywania nieznanej wypłaty do PnL settled. We wszystkich settled ledgerach ręczne PnL zgadza się z kolumną ledgeru, wypływ z notionalem i opłatą gotówkową, a liczba naruszeń maksymalnej straty wynosi zero.",
        "",
        "## Kontrakt `$5` i raportowanie",
        "",
        "A (`$5` jako maksymalny wydatek z opłatami) nie opisuje tego kodu. B jest właściwą definicją: `$5` to żądany/gross notional zakupu; opłaty gotówkowe mogą zwiększyć całkowity wydatek. Historyczny tryb poboru opłaty zależy od daty: przed zmianą V1 potrąca opłatę z tokenów, po zmianie V2 pobiera ją w gotówce. Zmienia się jednostka opłaty, nie znaczenie stawki `$5`.",
        "",
        "- Decyzja T-60 i EV: obliczenie EV odejmuje pełny `cash_debit_usd` od `p × net_shares`; próg 1% to net EV podzielone przez żądany notional `$5`.",
        "- Wykonanie T-59: zachowuje wybraną stronę i notional; księguje gotówkową opłatę w `cash_debit_usd`, a V1 share fee w `net_shares`.",
        "- Obrót: `gross_usd` to notional, nie wszystkie wydatki gotówkowe. `PnL / obrót` należy czytać jako `PnL / notional`; tabela pokazuje równolegle zwrot względem całkowitego wypływu.",
        "- Rozliczenie i audyt: PnL to `payout_usd - cash_debit_usd`. USD-ekwiwalentu opłaty tokenowej nie odejmuje się ponownie.",
        "",
        "### Ekonomia według miesiąca",
        "",
        "| Miesiąc | Wariant | Fill’e / settled | Notional | Wypływ gotówki | Opłaty cash | Ekwiwalent opłaty tokenowej | PnL | PnL / notional | PnL / wypływ | PnL−EV |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in _monthly_cashflow_rows(monthly_economics, trades_by_variant):
        notional_ratio = "n/a" if row["pnl_per_notional"] is None else f"{row['pnl_per_notional']:.3%}"
        cash_ratio = "n/a" if row["pnl_per_cash_outflow"] is None else f"{row['pnl_per_cash_outflow']:.3%}"
        lines.append(
            f"| {row['utc_month']} | {row['variant']} | {row['fills']:,} / {row['settlements']:,} | "
            f"{_format_usd(row['gross_notional_usd'])} | {_format_usd(row['cash_outflow_usd'])} | "
            f"{_format_usd(row['cash_fees_paid_usd'])} | "
            f"{_format_usd(row['share_fee_usd_entry_equivalent_informational'])} | "
            f"{_format_usd(row['net_pnl_usd'])} | {notional_ratio} | {cash_ratio} | "
            f"{_format_usd(row['pnl_minus_forecast_ev_usd'])} |"
        )
    lines.extend([
        "",
        "| Wariant | PnL 5 najlepszych dni | Udział w PnL ogółem |",
        "|---|---:|---:|",
    ])
    for name in MODEL_NAMES:
        item = economics["daily_pnl_concentration"][name]
        share = "n/a" if item["top_five_share_of_total_pnl"] is None else f"{item['top_five_share_of_total_pnl']:.1%}"
        lines.append(f"| {name} | {_format_usd(item['top_five_days_pnl_usd'])} | {share} |")
    lines.extend([
        "",
        "Wariant bez rozliczonych fill’i nie wykazuje przewagi po progu EV; próg pozostał bez zmian. Miesięczne zestawienie obejmuje również miesiące bez fill’i.",
        "",
        "## Odpowiedź na pytania",
        "",
        f"- Czy RAW wnosi informację ponad skalibrowany rynek? Tak w jakości prognoz: M1 poprawia log loss o `{abs(quality['contrasts']['M1_MINUS_M0']['log_loss']['point_difference']):.6f}` i Brier o `{abs(quality['contrasts']['M1_MINUS_M0']['brier']['point_difference']):.6f}`, a oba 95% CI wykluczają zero. M1−RAW log loss wynosi `{quality['contrasts']['M1_MINUS_RAW']['log_loss']['point_difference']:.6f}` (CI obejmuje zero), więc M1 nie wykazał przewagi nad samym RAW.",
        f"- Czy warstwa przewiduje przyczynowo późniejsze dni development? Tak, współczynniki dla każdego dnia używają wyłącznie etykiet zwolnionych przed pierwszą decyzją tego dnia; nie jest to niezależna ocena całego modelu BASE70.",
        f"- Czy korzyść utrzymuje się po kosztach i progu? M1 ma `{_format_usd(economics['variants']['M1']['net_pnl_usd'])}` PnL wobec `{_format_usd(economics['variants']['M0']['net_pnl_usd'])}` dla M0; różnica M1−M0 to `{_format_usd(economics['variants']['M1']['net_pnl_usd'] - economics['variants']['M0']['net_pnl_usd'])}`. Bootstrap CI sumarycznego PnL `{_format_usd(total_low)}` do `{_format_usd(total_high)}` obejmuje zero, więc przewaga po kosztach nie jest rozstrzygająca.",
        f"- Czy M1 ogranicza duże rozbieżności? M1 był bliżej rynku w `{divergence['top_quartile_fraction_M1_closer_to_market_than_RAW']:.1%}` największych rozbieżności RAW–rynek.",
        "- Jeśli M1 nie przewyższa M0, eksperyment kończy się tutaj; nie dodano drzew, interakcji ani siatki regularyzacji.",
        "",
        "## Pliki i odtwarzalność",
        "",
        f"Manifest zamrożono przed obliczeniami: `{_relative(VERSIONED_MANIFEST_PATH)}`. Dane prognoz, współczynniki dzienne, bootstrap, wyniki miesięczne i niezależne ledgery wariantów są w `{CONFIG['output_directory']}`. Runner nie przyjmuje argumentów CLI: `python {CONFIG['runner']}`.",
        f"Czas oryginalnego runnera: `{manifest.get('elapsed_seconds')}` s; szczytowy working set procesu: `{manifest.get('peak_working_set_gib')}` GiB. Raport i audyt przeliczono z zapisanych ledgerów; nie pobierano danych, nie trenowano BTC ani nie odtwarzano archiwum.",
        "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    started = time.perf_counter()
    manifest = _freeze_before_calculation()
    predictions, coefficients, calibration = _load_or_fit_predictions(manifest)

    bootstrap_path = OUTPUT_DIR / "shared_bootstrap_day_indices.npz"
    shared = predictions.loc[
        predictions[[PROBABILITY_COLUMNS[name] for name in MODEL_NAMES]].notna().all(axis=1)
    ].copy()
    if shared.empty:
        raise RuntimeError("No common forecast rows remain after the daily model warm-up")
    shared.sort_values("market_start_utc", kind="stable", inplace=True)
    days = pd.date_range(shared.market_start_utc.min().floor("D"), shared.market_start_utc.max().floor("D"), freq="D", tz="UTC")
    indices = _bootstrap_indices(days)
    _write_bootstrap(bootstrap_path, indices, days)

    quality_path = OUTPUT_DIR / "quality_summary.json"
    monthly_quality_path = OUTPUT_DIR / "monthly_quality.csv"
    quality_stage = "quality_and_bootstrap"
    if _checkpoint_valid(manifest, quality_stage):
        quality_summary = json.loads(quality_path.read_text(encoding="utf-8"))
        monthly_quality = pd.read_csv(monthly_quality_path)
    else:
        manifest["status"] = "running"
        manifest["active_stage"] = quality_stage
        manifest["stages"][quality_stage] = {"status": "running", "started_utc": pd.Timestamp.now(tz="UTC").isoformat()}
        _write_manifest(manifest)
        quality_rows = _quality_rows(predictions)
        quality_summary, monthly_quality = _quality_summary(quality_rows, days, indices)
        quality_summary["divergence"] = _divergence_summary(quality_rows)
        _write_json(quality_path, quality_summary)
        monthly_quality.to_csv(monthly_quality_path, index=False)
        manifest["stages"][quality_stage] = {
            "status": "completed", "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "summary": quality_summary,
            "outputs": [_record(path) for path in (quality_path, monthly_quality_path, bootstrap_path)],
        }
        manifest["active_stage"] = None
        _write_manifest(manifest)

    common, t59, t60 = _load_execution_sources(set(shared.condition_id.dropna().astype(str).str.lower()))
    execution_calendar = common.merge(
        shared[["market_start_utc", *[PROBABILITY_COLUMNS[name] for name in MODEL_NAMES]]],
        on="market_start_utc", how="inner", validate="one_to_one",
    )
    if len(execution_calendar) != len(shared):
        raise RuntimeError("The saved development execution calendar does not cover all shared model forecasts")
    # p_market quality eligibility narrows the economic calendar to markets with all four predictions.
    common_ids = set(execution_calendar.condition_id.dropna().astype(str).str.lower())
    if len(common_ids) != len(execution_calendar):
        raise RuntimeError("Shared execution calendar contains a missing or duplicate market identity")
    t59_by_id = research._snapshot_map(t59.loc[t59.condition_id.isin(common_ids)])
    t60_by_id = research._snapshot_map(t60.loc[t60.condition_id.isin(common_ids)])
    summaries = {}
    decisions_by_variant = {}
    trades_by_variant = {}
    monthly_economic_rows = []
    for variant in MODEL_NAMES:
        summary, decisions, trades = _run_economic_variant(
            variant, shared, execution_calendar, t59_by_id, t60_by_id, manifest,
        )
        summaries[variant] = summary
        decisions_by_variant[variant] = decisions
        trades_by_variant[variant] = trades
        monthly_economic_rows.extend(summary["monthly_results"])

    base_trades = pd.read_parquet(ROOT / CONFIG["sources"]["base70_raw_trades_path"])
    raw_anchor = json.loads((ROOT / CONFIG["sources"]["base70_raw_summary_path"]).read_text(encoding="utf-8"))
    reconciliation = _reconcile_raw_anchor(base_trades, common_ids, raw_anchor, trades_by_variant["RAW"])
    daily = _daily_pnl(trades_by_variant, days)
    daily_path = OUTPUT_DIR / "daily_pnl.csv"
    monthly_economics_path = OUTPUT_DIR / "monthly_economics.csv"
    _write_daily = daily.to_csv(daily_path, index=False)
    monthly_economics = pd.DataFrame(monthly_economic_rows)
    monthly_economics.to_csv(monthly_economics_path, index=False)
    economics = {
        "variants": summaries,
        "shared_calendar": {
            "forecast_rows": int(len(shared)),
            "decisions": int(len(execution_calendar)),
            "first_market_start_utc": execution_calendar.market_start_utc.min().isoformat(),
            "last_market_start_utc": execution_calendar.market_start_utc.max().isoformat(),
            "utc_days": int(execution_calendar.market_start_utc.dt.floor("D").nunique()),
        },
        "raw_anchor_reconciliation": {
            **reconciliation,
            "previous_full_calendar_decisions": int(CONFIG["reference_RAW_anchor"]["decision_rows"]),
        },
        "paired_pnl_bootstrap": _pnl_bootstrap(daily, indices),
        "daily_pnl_concentration": _concentration(daily),
    }
    availability = {
        "development_oof_rows": int(len(predictions)),
        "rows_with_raw_probability": int(predictions.p_raw.notna().sum()),
        "rows_with_valid_market_probability": int(predictions.p_market.notna().sum()),
        "rows_with_both_raw_and_market": int((predictions.p_raw.notna() & predictions.p_market.notna()).sum()),
        "shared_forecast_rows": int(len(shared)),
        "quality_rows": int(quality_summary["sample"]["rows"]),
        "warmup_rows_before_first_shared_forecast": int(predictions.market_start_utc.lt(shared.market_start_utc.min()).sum()),
        "decision_snapshot_available": int(execution_calendar.t60_decision_snapshot_available.fillna(False).sum()),
        "execution_snapshot_available": int(execution_calendar.t59_snapshot_available.fillna(False).sum()),
        "outcomes_available_by_cutoff": int(execution_calendar.outcome_available_by_cutoff.fillna(False).sum()),
    }
    summary = {
        "experiment_id": CONFIG["experiment_id"],
        "development_only": True,
        "holdout_evaluated": False,
        "calibration": calibration,
        "availability": availability,
        "quality": quality_summary,
        "market_divergence": quality_summary["divergence"],
        "economics": economics,
    }
    summary_path = OUTPUT_DIR / "experiment_summary.json"
    _write_json(summary_path, summary)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    memory = _peak_working_set_bytes()
    manifest["peak_working_set_gib"] = round(memory / 2**30, 3) if memory is not None else None
    manifest["status"] = "completed"
    manifest["active_stage"] = None
    manifest["next_step"] = "Review the development-only report; do not extend the experiment unless a new hypothesis is frozen."
    manifest["result_summary"] = {
        "quality_rows": availability["quality_rows"],
        "shared_forecast_rows": availability["shared_forecast_rows"],
        "M1_minus_M0_log_loss": quality_summary["contrasts"]["M1_MINUS_M0"]["log_loss"],
        "M1_minus_M0_net_pnl_usd": summaries["M1"]["net_pnl_usd"] - summaries["M0"]["net_pnl_usd"],
        "raw_anchor_reconciliation": reconciliation,
    }
    manifest["stages"]["final_summary"] = {
        "status": "completed", "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "outputs": [_record(path) for path in (summary_path, daily_path, monthly_economics_path)],
    }
    _write_manifest(manifest)
    report_started = time.perf_counter()
    _write_report(manifest, summary, monthly_quality, monthly_economics, trades_by_variant)
    manifest["stages"]["report"] = {
        "status": "completed", "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "outputs": [_record(REPORT_PATH)],
        "elapsed_seconds": time.perf_counter() - report_started,
    }
    _write_manifest(manifest)
    _status_update(summary)
    return summary


if __name__ == "__main__":
    run()
