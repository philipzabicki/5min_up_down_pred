"""Search and compare a causal ESN feature portfolio on BTC development data only."""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import timezone
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import pyarrow.dataset as ds
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from features.esn_reservoir import (
    ESNConfig,
    ESNEngine,
    FeatureScaler,
    FrozenProjection,
    INPUT_COLUMNS,
    INPUT_GROUPS,
    build_causal_esn_inputs,
    build_nvar_features,
    development_arrow_filter,
    estimate_warmup,
    exclusive_opened_cutoff,
    nominal_decision_ns,
    purged_training_mask,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_esn_portfolio_20261020.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUT = ROOT / CONFIG["output_directory"]
PREP = OUT / "prepared"
TRIALS = OUT / "trials"
MODELS = OUT / "models"
REPORT_PATH = ROOT / CONFIG["experiment_report_path"]
MANIFEST_PATH = ROOT / CONFIG["versioned_manifest_path"]

BASE_PATH = ROOT / CONFIG["baseline_matrix_path"]
SUMMARY_PATH = ROOT / CONFIG["baseline_summary_path"]
SOURCE_PATH = ROOT / CONFIG["source_feature_dataset_path"]
TARGET = "target_preopen_5m_up_binance_index_proxy"
DECISION = "is_polymarket_preopen_decision"
RAW_COLUMNS = ("Opened", "Open", "High", "Low", "Close", "Volume", "futures_index_basis_rel_1m")
SEEDS = tuple(int(seed) for seed in CONFIG["screening"]["seeds"])
FINAL_SEEDS = tuple(int(seed) for seed in CONFIG["final_comparison"]["seeds"])
DEVELOPMENT_START = pd.Timestamp(CONFIG["development_data_start_utc"])
DEVELOPMENT_CUTOFF = pd.Timestamp(CONFIG["development_cutoff_exclusive_utc"])
FIT_END = pd.Timestamp(CONFIG["split"]["fit_core_end_exclusive_utc"])
CORRECTION_START = pd.Timestamp(CONFIG["split"]["correction_fit_start_utc"])
CORRECTION_END = pd.Timestamp(CONFIG["split"]["correction_fit_end_exclusive_utc"])
SEARCH_END = pd.Timestamp(CONFIG["split"]["search_validation_end_exclusive_utc"])
CONFIRMATION_START = pd.Timestamp(CONFIG["split"]["confirmation_start_utc"])
NS_MINUTE = pd.Timedelta(minutes=1).value
WEIGHT_DECISION = float(CONFIG["baseline"]["training_weights"]["decision_rows"])
WEIGHT_OTHER = float(CONFIG["baseline"]["training_weights"]["other_minute_rows"])
BASE_FEATURES = tuple(json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))["feature_order"])
RUN_STARTED = None
RUN_PRIOR_SECONDS = 0.0
RUN_IDENTITY = None
THREAD_LIMITER = threadpool_limits(limits=8)
RECOVERABLE_RUNNER_SHA256 = "0068563f64d3e2e3a70e2f1a62e59d94850d7addc38c0b0a45f9c3e2dab4f1a8"
RECOVERABLE_ENGINE_SHA256 = "a71b271187d13d8cb29f0b7d403346e55d743af40de3f8c2e85160279df17326"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_id(path: Path) -> dict:
    stat = path.stat()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _identity() -> dict:
    return {
        "experiment_id": CONFIG["experiment_id"],
        "source_commit": CONFIG["source_commit"],
        "config_sha256": _sha256(CONFIG_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "engine_sha256": _sha256(ROOT / "features/esn_reservoir.py"),
        "source_data": _file_id(SOURCE_PATH),
        "baseline_data": _file_id(BASE_PATH),
        "baseline_summary_sha256": _sha256(SUMMARY_PATH),
        "development_cutoff_exclusive_utc": CONFIG["development_cutoff_exclusive_utc"],
    }


def _fit_ready_indices(raw_opened, ready_mask, core_end, stream_rows: int) -> np.ndarray:
    streamed_opened = np.asarray(raw_opened[:stream_rows], dtype=np.int64)
    ready_mask = np.asarray(ready_mask, dtype=bool)
    if len(streamed_opened) != len(ready_mask):
        raise ValueError("ready mask must align with the materialized stream prefix")
    return np.flatnonzero((streamed_opened < int(core_end)) & ready_mask)


def _code_recovery_allowed(current_identity: dict, previous_identity: dict) -> bool:
    if (
        previous_identity.get("runner_sha256") != RECOVERABLE_RUNNER_SHA256
        or previous_identity.get("engine_sha256") != RECOVERABLE_ENGINE_SHA256
    ):
        return False
    code_keys = {"runner_sha256", "engine_sha256"}
    current_comparable = {key: value for key, value in current_identity.items() if key not in code_keys}
    previous_comparable = {key: value for key, value in previous_identity.items() if key not in code_keys}
    return current_comparable == previous_comparable


def _recover_completed_search_identity(current_identity: dict, ledger: dict) -> dict | None:
    previous_identity = ledger.get("identity")
    if not isinstance(previous_identity, dict) or not _code_recovery_allowed(current_identity, previous_identity):
        return None
    required_artifacts = (
        OUT / "search_results.json",
        OUT / "portfolio_selection.json",
        OUT / "downstream_budget.json",
        PREP / "confirmation" / "prepared_manifest.json",
    )
    for path in required_artifacts:
        if not path.exists():
            raise RuntimeError(f"runner recovery requires completed frozen-stage artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != previous_identity:
            raise RuntimeError(f"runner recovery artifact identity mismatch: {path}")
    search = json.loads(required_artifacts[0].read_text(encoding="utf-8"))
    downstream = json.loads(required_artifacts[2].read_text(encoding="utf-8"))
    confirmation = json.loads(required_artifacts[3].read_text(encoding="utf-8"))
    if search.get("status") != "search_complete" or downstream.get("status") != "fixed_downstream_budget_complete":
        raise RuntimeError("runner recovery requires completed search and downstream checks")
    if confirmation.get("materialization_cutoff_utc") != DEVELOPMENT_CUTOFF.isoformat():
        raise RuntimeError("runner recovery confirmation data does not match the frozen development cutoff")
    return previous_identity


def _elapsed_run_seconds() -> float:
    return RUN_PRIOR_SECONDS + (0.0 if RUN_STARTED is None else float(time.monotonic() - RUN_STARTED))


def _runtime_total(ledger: dict) -> float:
    stages = ledger.get("stages_seconds", {})
    non_search = sum(float(value) for key, value in stages.items() if key != "optuna_search")
    search_seconds = max(
        float(stages.get("optuna_search", 0.0)),
        float(sum(ledger.get("optuna_trial_seconds", {}).values())),
    )
    return float(ledger.get("prior_failed_attempt_active_seconds", 0.0)) + non_search + search_seconds


def _record_stage_time(stage: str, elapsed_seconds: float) -> None:
    ledger_path = OUT / "runtime_ledger.json"
    ledger = {
        "identity": RUN_IDENTITY,
        "stages_seconds": {},
        "optuna_trial_seconds": {},
        "prior_failed_attempt_active_seconds": float(CONFIG["search"].get("prior_failed_attempt_active_seconds", 0.0)),
        "active_seconds": RUN_PRIOR_SECONDS,
    }
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("identity") != RUN_IDENTITY:
            raise RuntimeError("runtime ledger identity mismatch")
    if stage not in ledger["stages_seconds"]:
        ledger["stages_seconds"][stage] = float(elapsed_seconds)
    ledger["active_seconds"] = _runtime_total(ledger)
    _atomic_json(ledger_path, ledger)


def _record_optuna_trial_time(trial) -> None:
    if trial.duration is None:
        return
    ledger_path = OUT / "runtime_ledger.json"
    ledger = {
        "identity": RUN_IDENTITY, "stages_seconds": {}, "optuna_trial_seconds": {},
        "prior_failed_attempt_active_seconds": float(CONFIG["search"].get("prior_failed_attempt_active_seconds", 0.0)),
    }
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("identity") != RUN_IDENTITY:
            raise RuntimeError("runtime ledger identity mismatch")
    ledger.setdefault("optuna_trial_seconds", {})[str(trial.number)] = float(trial.duration.total_seconds())
    ledger["active_seconds"] = _runtime_total(ledger)
    _atomic_json(ledger_path, ledger)


def _check_compute_cap() -> None:
    cap = float(CONFIG["search"]["whole_experiment_compute_limit_seconds"])
    if _elapsed_run_seconds() > cap:
        raise TimeoutError(f"frozen whole-experiment compute cap of {cap:.0f} seconds was reached")


def _write_frozen_manifest(identity: dict) -> None:
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing.get("identity") == identity:
            return
        search_started = (OUT / "optuna.sqlite3").exists() or (PREP / "search" / "prepared_manifest.json").exists()
        if existing.get("status") != "frozen_before_search" or search_started:
            raise RuntimeError("frozen manifest identity differs after search-stage artifacts exist")
    _atomic_json(MANIFEST_PATH, {
        "experiment_id": CONFIG["experiment_id"],
        "status": "frozen_before_search",
        "identity": identity,
        "development_start_utc": CONFIG["development_data_start_utc"],
        "development_cutoff_exclusive_utc": CONFIG["development_cutoff_exclusive_utc"],
        "data_paths": {
            "source_feature_dataset": CONFIG["source_feature_dataset_path"],
            "baseline_common_matrix": CONFIG["baseline_matrix_path"],
            "baseline_summary": CONFIG["baseline_summary_path"],
        },
        "data_contract": CONFIG["baseline"],
        "split": CONFIG["split"],
        "search_budget": CONFIG["search"],
        "screening": CONFIG["screening"],
        "final_comparison": CONFIG["final_comparison"],
        "holdout_rows_materialized": 0,
        "confirmation_observations_materialized_before_portfolio_freeze": False,
        "holdout_evaluation": False,
        "polymarket_book_replay": False,
        "live_configuration_changes": False,
        "live_orders": False,
    })


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _save_npy(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(temp, path)


def _load_npy(name: str, *, mmap=True, data_dir: Path = PREP):
    return np.load(data_dir / f"{name}.npy", mmap_mode="r" if mmap else None, allow_pickle=False)


def _utc_ns(value) -> int:
    return int(pd.Timestamp(value).tz_convert("UTC").value)


def _decision_at_ns(opened_ns: np.ndarray) -> np.ndarray:
    return nominal_decision_ns(opened_ns, offset_ns=NS_MINUTE)


def _development_filter(opened_field, start, exclusive_cutoff):
    return development_arrow_filter(opened_field, start, exclusive_cutoff)


def _training_mask(opened_ns, label_available_ns, boundary_opened) -> np.ndarray:
    """Keep only earlier development rows whose labels exist by the block decision."""
    boundary = _utc_ns(boundary_opened)
    first_nominal_decision = boundary + NS_MINUTE
    return purged_training_mask(
        opened_ns, label_available_ns, first_nominal_decision,
        development_start_ns=_utc_ns(DEVELOPMENT_START),
    ) & (opened_ns < boundary)


def _valid_candles(candles: pd.DataFrame) -> np.ndarray:
    values = candles[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=np.float64)
    return (
        np.isfinite(values).all(axis=1)
        & (values[:, 0] > 0.0) & (values[:, 1] > 0.0)
        & (values[:, 2] > 0.0) & (values[:, 3] > 0.0)
        & (values[:, 1] >= values[:, 2]) & (values[:, 2] <= values[:, 3])
        & (values[:, 4] >= 0.0)
    )


def _static_ready_mask(timestamps_ns, valid, warmup_steps):
    valid = np.asarray(valid, dtype=bool)
    reset = ~valid.copy()
    if len(valid) > 1:
        reset[1:] |= (~valid[:-1]) | (np.diff(timestamps_ns) != NS_MINUTE)
    starts = np.maximum.accumulate(np.where(reset, np.arange(len(valid)), 0))
    run_length = np.arange(len(valid)) - starts + 1
    return valid & (run_length > warmup_steps)


def _load_arrays(data_dir: Path, identity: dict) -> dict:
    arrays = {name: _load_npy(name, data_dir=data_dir) for name in (
        "raw_opened", "raw_x", "raw_valid", "base_opened", "base_decision_at",
        "label_available", "target", "decision", "base_x", "base_to_raw",
    )}
    arrays["identity"] = identity
    return arrays


def _prepare_data(identity: dict, *, materialization_cutoff=SEARCH_END, prep_dir: Path | None = None) -> dict:
    prep_dir = prep_dir or (PREP / "search")
    prep_dir.mkdir(parents=True, exist_ok=True)
    materialization_cutoff = pd.Timestamp(materialization_cutoff).tz_convert("UTC")
    cutoff_ns = int(materialization_cutoff.value)
    cutoff_text = materialization_cutoff.strftime("%Y-%m-%d %H:%M:%S")
    raw_opened_cutoff = exclusive_opened_cutoff(materialization_cutoff)
    raw_opened_cutoff_text = raw_opened_cutoff.strftime("%Y-%m-%d %H:%M:%S")
    manifest_path = prep_dir / "prepared_manifest.json"
    required = ["base_opened", "base_decision_at", "label_available", "target", "decision", "base_x", "raw_opened", "raw_x", "raw_valid", "base_to_raw"]
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity or existing.get("materialization_cutoff_utc") != materialization_cutoff.isoformat():
            raise RuntimeError("prepared data identity differs from the frozen code/config/data; use a new experiment directory")
        if all((prep_dir / f"{name}.npy").exists() for name in required):
            return existing

    start_string = CONFIG["development_data_start_utc"].replace("T", " ").replace("Z", "")
    source = ds.dataset(SOURCE_PATH, format="parquet")
    opened_type = str(source.schema.field("Opened").type)
    if opened_type not in {"string", "large_string"}:
        raise RuntimeError(f"unexpected raw Opened type: {opened_type}")
    predicate = _development_filter(ds.field("Opened"), start_string, raw_opened_cutoff_text)
    scanner = source.scanner(columns=list(RAW_COLUMNS), filter=predicate, batch_size=65_536, use_threads=False)
    candle_parts = [batch.to_pandas() for batch in scanner.to_batches()]
    if not candle_parts:
        raise RuntimeError("development-only raw-candle predicate returned no rows")
    candles = pd.concat(candle_parts, ignore_index=True)
    candles["Opened"] = pd.to_datetime(candles["Opened"], utc=True, errors="raise")
    if candles["Opened"].duplicated().any() or not candles["Opened"].is_monotonic_increasing:
        raise RuntimeError("development candle stream has duplicate or unordered timestamps")
    raw_opened = pd.DatetimeIndex(candles["Opened"]).as_unit("ns").asi8.astype(np.int64, copy=True)
    raw_open_price = pd.to_numeric(candles["Open"], errors="raise").to_numpy(dtype=np.float64)
    raw_close_price = pd.to_numeric(candles["Close"], errors="raise").to_numpy(dtype=np.float64)
    if (raw_opened + NS_MINUTE >= cutoff_ns).any():
        raise RuntimeError("raw candle close-event timestamp reached the exclusive stage cutoff")
    raw_valid = _valid_candles(candles)
    raw_timestamps, input_frame = build_causal_esn_inputs(candles)
    if not np.array_equal(raw_timestamps.as_unit("ns").asi8, raw_opened):
        raise RuntimeError("causal input timestamps changed during feature construction")
    raw_x = input_frame.to_numpy(dtype=np.float32, copy=True)
    del input_frame, candle_parts, candles

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if len(BASE_FEATURES) != int(summary["feature_count"]) or len(set(BASE_FEATURES)) != len(BASE_FEATURES):
        raise RuntimeError("baseline feature order is invalid")
    base_columns = ["Opened", "nominal_decision_at", "target_label_available_at", TARGET, DECISION, *BASE_FEATURES]
    baseline_dataset = ds.dataset(BASE_PATH, format="parquet")
    cutoff_datetime = materialization_cutoff.to_pydatetime().replace(tzinfo=timezone.utc)
    baseline_filter = (
        _development_filter(ds.field("Opened"), DEVELOPMENT_START.to_pydatetime(), cutoff_datetime)
        & (ds.field("target_label_available_at") < cutoff_datetime)
    )
    nrows = baseline_dataset.count_rows(filter=baseline_filter)
    base_opened = np.empty(nrows, dtype=np.int64)
    base_decision_at = np.empty(nrows, dtype=np.int64)
    label_available = np.empty(nrows, dtype=np.int64)
    target = np.empty(nrows, dtype=np.int8)
    decision = np.empty(nrows, dtype=bool)
    base_x = np.empty((nrows, len(BASE_FEATURES)), dtype=np.float32)
    cursor = 0
    baseline_scanner = baseline_dataset.scanner(
        columns=base_columns, filter=baseline_filter, batch_size=65_536, use_threads=False,
    )
    for batch in baseline_scanner.to_batches():
        count = batch.num_rows
        stop = cursor + count
        base_opened[cursor:stop] = pd.to_datetime(batch.column("Opened").to_numpy(zero_copy_only=False), utc=True).as_unit("ns").asi8
        base_decision_at[cursor:stop] = pd.to_datetime(batch.column("nominal_decision_at").to_numpy(zero_copy_only=False), utc=True).as_unit("ns").asi8
        label_available[cursor:stop] = pd.to_datetime(batch.column("target_label_available_at").to_numpy(zero_copy_only=False), utc=True).as_unit("ns").asi8
        labels = batch.column(TARGET).to_numpy(zero_copy_only=False)
        flags = batch.column(DECISION).to_numpy(zero_copy_only=False)
        if np.isnan(labels.astype(np.float64)).any() or not np.isin(labels, (0, 1)).all():
            raise RuntimeError("baseline common matrix has missing or non-binary targets")
        target[cursor:stop] = labels.astype(np.int8, copy=False)
        decision[cursor:stop] = flags.astype(bool, copy=False)
        feature_values = np.column_stack([
            batch.column(name).to_numpy(zero_copy_only=False) for name in BASE_FEATURES
        ])
        if not np.isfinite(feature_values).all():
            raise RuntimeError("common baseline feature matrix contains non-finite values")
        base_x[cursor:stop] = feature_values.astype(np.float32, copy=False)
        cursor = stop
    if cursor != nrows or (len(base_opened) > 1 and np.any(np.diff(base_opened) <= 0)):
        raise RuntimeError("baseline development matrix count/order does not match parquet metadata")
    if (base_opened >= cutoff_ns).any() or (label_available >= cutoff_ns).any():
        raise RuntimeError("baseline stage materialized a row or label at/after its exclusive cutoff")
    raw_lookup = np.searchsorted(raw_opened, base_opened)
    if np.any(raw_lookup >= len(raw_opened)) or not np.array_equal(raw_opened[raw_lookup], base_opened):
        raise RuntimeError("baseline rows cannot be exactly aligned to development candle inputs")
    if not np.array_equal(base_decision_at, _decision_at_ns(base_opened)):
        raise RuntimeError("baseline nominal decision timestamps differ from Opened + 1 minute")
    if not np.array_equal(label_available, base_opened + 7 * NS_MINUTE):
        raise RuntimeError("baseline target availability differs from Opened + 7 minutes")
    open_plus_2 = np.searchsorted(raw_opened, base_opened + 2 * NS_MINUTE)
    close_plus_6 = np.searchsorted(raw_opened, base_opened + 6 * NS_MINUTE)
    if (open_plus_2 >= len(raw_opened)).any() or (close_plus_6 >= len(raw_opened)).any():
        raise RuntimeError("a baseline label extends outside the materialized development source")
    if not np.array_equal(raw_opened[open_plus_2], base_opened + 2 * NS_MINUTE) or not np.array_equal(raw_opened[close_plus_6], base_opened + 6 * NS_MINUTE):
        raise RuntimeError("a baseline target depends on a missing minute")
    derived_target = (raw_close_price[close_plus_6] >= raw_open_price[open_plus_2]).astype(np.int8)
    if not np.array_equal(target, derived_target):
        raise RuntimeError("stored target does not match Close[Opened+6m] >= Open[Opened+2m]")
    for name, value in (
        ("raw_opened", raw_opened), ("raw_x", raw_x), ("raw_valid", raw_valid),
        ("base_opened", base_opened), ("base_decision_at", base_decision_at),
        ("label_available", label_available), ("target", target),
        ("decision", decision), ("base_x", base_x), ("base_to_raw", raw_lookup.astype(np.int32)),
    ):
        _save_npy(prep_dir / f"{name}.npy", value)
    prepared = {
        "identity": identity,
        "status": "development_data_prepared",
        "materialization_cutoff_utc": materialization_cutoff.isoformat(),
        "source_filter_applied_before_materialization": f"Opened >= {start_string} and Opened < {raw_opened_cutoff_text} (equivalent to Opened+1m < {cutoff_text})",
        "baseline_filter_applied_before_materialization": f"Opened >= {CONFIG['development_data_start_utc']} and Opened < {cutoff_text} and target_label_available_at < {cutoff_text}",
        "source_opened_type": opened_type,
        "development_raw_rows": int(len(raw_opened)),
        "development_baseline_rows": int(nrows),
        "decision_rows": int(decision.sum()),
        "baseline_feature_count": len(BASE_FEATURES),
        "first_raw_opened_utc": pd.Timestamp(raw_opened[0], tz="UTC").isoformat(),
        "last_raw_opened_utc": pd.Timestamp(raw_opened[-1], tz="UTC").isoformat(),
        "holdout_rows_materialized": 0,
    }
    _atomic_json(manifest_path, prepared)
    return prepared


def _metric_pack(y, p) -> dict:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y = np.asarray(y, dtype=np.int8)
    return {
        "n": int(len(y)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
    }


def _lgbm_params(seed: int) -> dict:
    params = dict(json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))["params"])
    params.update({
        "seed": int(seed), "data_random_seed": int(seed),
        "feature_fraction_seed": int(seed), "bagging_seed": int(seed),
        "metric": "None", "verbosity": -1, "num_threads": 8,
    })
    return params


def _fit_lgbm(features, labels, weights, training_mask, seed, num_boost_round=None):
    rows = np.flatnonzero(training_mask)
    x_train = np.asarray(features[rows], dtype=np.float32)
    y_train = labels[rows].astype(np.int8, copy=False)
    w_train = weights[rows].astype(np.float32, copy=False)
    dataset = lgb.Dataset(x_train, label=y_train, weight=w_train, free_raw_data=True)
    booster = lgb.train(
        _lgbm_params(seed), dataset,
        num_boost_round=int(num_boost_round or CONFIG["baseline"]["lightgbm"]["boosting_rounds"]),
        callbacks=[lgb.log_evaluation(0)],
    )
    return booster, rows


def _load_cached(path: Path, identity: dict):
    meta = path.with_suffix(path.suffix + ".json")
    if not path.exists() or not meta.exists():
        return None
    if json.loads(meta.read_text(encoding="utf-8")) != identity:
        raise RuntimeError(f"cached stage identity mismatch: {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _save_stage_array(path: Path, values, identity: dict) -> None:
    _save_npy(path, values)
    _atomic_json(path.with_suffix(path.suffix + ".json"), identity)


def _screen_baseline(identity: dict, arrays: dict) -> dict:
    result_path = OUT / "screen_baseline.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("identity") != identity:
            raise RuntimeError("screen baseline artifact identity mismatch")
        prediction_path = OUT / "screen_baseline_raw_score.npy"
        prediction_meta = prediction_path.with_suffix(prediction_path.suffix + ".json")
        if not prediction_path.exists() or not prediction_meta.exists() or json.loads(prediction_meta.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("screen baseline prediction artifact is missing or has a different identity")
        return saved
    opened = arrays["base_opened"]
    labels = arrays["target"]
    decision = arrays["decision"]
    label_available = arrays["label_available"]
    weights = np.where(decision, WEIGHT_DECISION, WEIGHT_OTHER).astype(np.float32)
    fit = _training_mask(opened, label_available, FIT_END)
    prediction_mask = (
        (opened >= _utc_ns(CORRECTION_START))
        & (opened < _utc_ns(SEARCH_END))
    )
    if not fit.any() or not prediction_mask.any():
        raise RuntimeError("screen baseline split has no fit or OOS rows")
    booster, fit_rows = _fit_lgbm(arrays["base_x"], labels, weights, fit, 20261015)
    pred_path = OUT / "screen_baseline_raw_score.npy"
    pred = np.lib.format.open_memmap(pred_path, mode="w+", dtype=np.float32, shape=(len(opened),))
    pred[:] = np.nan
    pred[prediction_mask] = booster.predict(arrays["base_x"][prediction_mask], raw_score=True).astype(np.float32)
    pred.flush()
    _atomic_json(pred_path.with_suffix(".npy.json"), identity)
    corr = prediction_mask & (opened >= _utc_ns(CORRECTION_START)) & (opened < _utc_ns(CORRECTION_END)) & decision
    val = prediction_mask & (opened >= _utc_ns(CORRECTION_END)) & (opened < _utc_ns(SEARCH_END)) & decision
    value = {
        "identity": identity,
        "status": "oos_screen_baseline_complete",
        "fit_rows": int(len(fit_rows)),
        "fit_first_opened_utc": pd.Timestamp(opened[fit_rows[0]], tz="UTC").isoformat(),
        "fit_last_label_available_at_utc": pd.Timestamp(label_available[fit_rows].max(), tz="UTC").isoformat(),
        "correction_fit_decision_rows": int(corr.sum()),
        "search_validation_decision_rows": int(val.sum()),
        "lightgbm_seed": 20261015,
        "boosting_rounds": int(booster.current_iteration()),
        "feature_count": len(BASE_FEATURES),
        "training_weight_rule": {"decision": WEIGHT_DECISION, "other_minute": WEIGHT_OTHER},
        "prediction_artifact": pred_path.relative_to(ROOT).as_posix(),
        "holdout_rows_materialized": 0,
    }
    booster.save_model(str(MODELS / "screen_baseline.txt"))
    _atomic_json(result_path, value)
    return value


def _config_from_trial(trial) -> ESNConfig:
    groups = [group for group in INPUT_GROUPS if trial.suggest_categorical(f"group_{group}", [False, True])]
    if not groups:
        groups = [trial.suggest_categorical("fallback_group", list(INPUT_GROUPS))]
    architecture = trial.suggest_categorical("architecture", CONFIG["search"]["architecture_choices"])
    count = 1 if architecture == "single" else trial.suggest_categorical("reservoir_count", CONFIG["search"]["reservoir_count_choices"])
    topology = trial.suggest_categorical("topology", CONFIG["search"]["topology_choices"])
    connectivity = (
        trial.suggest_float("connectivity", *CONFIG["search"]["sparse_connectivity_range"])
        if topology == "sparse" else 0.15
    )
    return ESNConfig(
        architecture=architecture,
        reservoir_size=trial.suggest_categorical("reservoir_size", CONFIG["search"]["reservoir_size_choices"]),
        reservoir_count=count,
        leak_rate=trial.suggest_float("leak_rate", *CONFIG["search"]["leak_rate_range"]),
        input_scaling=trial.suggest_float("input_scaling", *CONFIG["search"]["input_scaling_range"]),
        bias_scaling=trial.suggest_float("bias_scaling", *CONFIG["search"]["bias_scaling_range"]),
        spectral_radius=trial.suggest_float("spectral_radius", *CONFIG["search"]["spectral_radius_range"]),
        topology=topology,
        connectivity=connectivity,
        input_groups=tuple(groups),
    )


def _config_record(config: ESNConfig) -> dict:
    return {**config.__dict__, "input_groups": list(config.input_groups), "selected_columns": list(config.selected_columns)}


def _stream_candidate(config: ESNConfig, seed: int, arrays: dict, *, cache_key: str):
    raw_opened = arrays["raw_opened"]
    raw_values = arrays["raw_x"]
    raw_valid = arrays["raw_valid"]
    search_input_cutoff = _utc_ns(exclusive_opened_cutoff(SEARCH_END))
    stream_rows = int(np.searchsorted(raw_opened, search_input_cutoff, side="left"))
    core_end = _utc_ns(FIT_END)
    indices = [INPUT_COLUMNS.index(name) for name in config.selected_columns]
    core_rows = np.flatnonzero((raw_opened < core_end) & raw_valid)
    if len(core_rows) < int(CONFIG["screening"]["pca_fit_rows"]):
        raise RuntimeError("not enough fit-core candle rows for scaler/PCA training")
    scaler_x = np.asarray(raw_values[core_rows][:, indices], dtype=np.float64)
    scaler = FeatureScaler(config.selected_columns).fit(scaler_x)
    del scaler_x
    warmup_input = np.asarray(raw_values[:min(len(raw_values), 200_000), :][:, indices], dtype=np.float64)
    warmup_mask = np.asarray(raw_valid[:len(warmup_input)], dtype=bool)
    warmup_steps, warmup_diagnostics = estimate_warmup(
        warmup_input, config, seed,
        relative_tolerance=CONFIG["screening"]["dynamic_diagnostics"]["warmup_relative_initial_state_tolerance"],
        stable_window=CONFIG["screening"]["dynamic_diagnostics"]["warmup_stable_window_minutes"],
        scaler=scaler,
        timestamps=pd.to_datetime(raw_opened[:len(warmup_input)], unit="ns", utc=True),
        valid_mask=warmup_mask,
        max_steps=CONFIG["screening"]["dynamic_diagnostics"]["warmup_max_steps_examined"],
    )
    if not warmup_diagnostics.get("stabilized", False):
        raise ValueError("washout diagnostic did not stabilize within the frozen training-only budget")
    ready_mask = _static_ready_mask(raw_opened[:stream_rows], raw_valid[:stream_rows], warmup_steps)
    fit_ready = _fit_ready_indices(raw_opened, ready_mask, core_end, stream_rows)
    sample_count = min(int(CONFIG["screening"]["pca_fit_rows"]), len(fit_ready))
    if sample_count < int(CONFIG["screening"]["pca_fit_rows"]):
        raise ValueError("dynamic warm-up leaves fewer fit-core PCA samples than frozen")
    sample_idx = fit_ready[np.linspace(0, len(fit_ready) - 1, sample_count, dtype=np.int64)]

    opened = arrays["base_opened"]
    decision = arrays["decision"]
    correction_idx = np.flatnonzero(
        (opened >= _utc_ns(CORRECTION_START)) & (opened < _utc_ns(CORRECTION_END)) & decision
    )
    validation_idx = np.flatnonzero(
        (opened >= _utc_ns(CORRECTION_END)) & (opened < _utc_ns(SEARCH_END)) & decision
    )
    capture_base_idx = np.r_[correction_idx, validation_idx]
    capture_raw_idx = np.asarray(arrays["base_to_raw"][capture_base_idx], dtype=np.int64)
    if np.any(np.diff(capture_raw_idx) <= 0):
        raise RuntimeError("OOS ESN capture indices are not strictly ordered")

    state_width = config.state_count * config.reservoir_size
    key = hashlib.sha256((cache_key + f"|seed={seed}").encode("utf-8")).hexdigest()[:20]
    work = TRIALS / f"{key}"
    work.mkdir(parents=True, exist_ok=True)
    capture_path = work / "capture_states.dat"
    sample_path = work / "pca_sample.dat"
    checkpoint_path = work / "engine_checkpoint.json"
    progress_path = work / "progress.json"
    capture_states = np.memmap(capture_path, mode="w+" if not progress_path.exists() else "r+", dtype=np.float32, shape=(len(capture_base_idx), state_width))
    sample_states = np.memmap(sample_path, mode="w+" if not progress_path.exists() else "r+", dtype=np.float32, shape=(sample_count, state_width))
    engine = ESNEngine(config, seed, scaler, warmup_steps=warmup_steps)
    start_row = 0
    sample_cursor = 0
    capture_cursor = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = {"identity": arrays["identity"], "config": _config_record(config), "seed": seed,
                    "sample_idx_sha256": hashlib.sha256(sample_idx.tobytes()).hexdigest(),
                    "capture_idx_sha256": hashlib.sha256(capture_raw_idx.tobytes()).hexdigest()}
        if any(progress.get(key) != value for key, value in expected.items()):
            raise RuntimeError("stream checkpoint identity mismatch; refusing to resume candidate")
        engine = ESNEngine.load_checkpoint(checkpoint_path, expected_config=config, expected_seed=seed)
        start_row = int(progress["next_row"])
        sample_cursor = int(progress["sample_cursor"])
        capture_cursor = int(progress["capture_cursor"])
    selected_values = raw_values[:stream_rows, indices]
    chunk_size = 65_536
    for start in range(start_row, stream_rows, chunk_size):
        end = min(stream_rows, start + chunk_size)
        chunk_times = pd.to_datetime(raw_opened[start:end], unit="ns", utc=True)
        states, ready = engine.update_many(
            chunk_times, selected_values[start:end], valid_mask=raw_valid[start:end],
        )
        sample_left = int(np.searchsorted(sample_idx, start, side="left"))
        sample_right = int(np.searchsorted(sample_idx, end, side="left"))
        if sample_right > sample_left:
            loc = sample_idx[sample_left:sample_right] - start
            if not ready[loc].all():
                raise RuntimeError("training PCA sample was selected before reservoir warm-up")
            sample_states[sample_cursor:sample_cursor + len(loc)] = states[loc]
            sample_cursor += len(loc)
        cap_left = int(np.searchsorted(capture_raw_idx, start, side="left"))
        cap_right = int(np.searchsorted(capture_raw_idx, end, side="left"))
        if cap_right > cap_left:
            loc = capture_raw_idx[cap_left:cap_right] - start
            capture_states[capture_cursor:capture_cursor + len(loc)] = states[loc]
            capture_cursor += len(loc)
        if end % 262_144 == 0 or end == stream_rows:
            sample_states.flush()
            capture_states.flush()
            engine.save_checkpoint(checkpoint_path)
            _atomic_json(progress_path, {
                "identity": arrays["identity"], "config": _config_record(config), "seed": seed,
                "sample_idx_sha256": hashlib.sha256(sample_idx.tobytes()).hexdigest(),
                "capture_idx_sha256": hashlib.sha256(capture_raw_idx.tobytes()).hexdigest(),
                "next_row": end, "sample_cursor": sample_cursor, "capture_cursor": capture_cursor,
            })
    sample_states.flush()
    capture_states.flush()
    if sample_cursor != sample_count or capture_cursor != len(capture_base_idx):
        raise RuntimeError("streaming ESN capture count did not match frozen indices")
    saturation = float(np.mean(np.abs(sample_states) >= 0.98))
    if saturation > CONFIG["screening"]["dynamic_diagnostics"]["reject_saturation_fraction_above"]:
        raise ValueError(f"reservoir saturation {saturation:.3f} exceeds frozen limit")
    pc_count = int(CONFIG["screening"]["state_components_per_reservoir"])
    pcas = []
    projected_sample = []
    projected_capture = []
    raw_sample = np.asarray(sample_states)
    raw_capture = np.asarray(capture_states)
    for reservoir in range(config.state_count):
        lower = reservoir * config.reservoir_size
        upper = lower + config.reservoir_size
        pca = PCA(n_components=pc_count, svd_solver="randomized", random_state=seed)
        sample_part = raw_sample[:, lower:upper]
        capture_part = raw_capture[:, lower:upper]
        pca.fit(sample_part)
        projected_sample.append(pca.transform(sample_part).astype(np.float32))
        projected_capture.append(pca.transform(capture_part).astype(np.float32))
        pcas.append(pca)
    sample_pc = np.concatenate(projected_sample, axis=1)
    capture_pc = np.concatenate(projected_capture, axis=1)
    del sample_part, capture_part, raw_sample, raw_capture
    sample_states._mmap.close()
    capture_states._mmap.close()
    del sample_states, capture_states
    try:
        capture_path.unlink()
        sample_path.unlink()
        checkpoint_path.unlink()
        progress_path.unlink()
    except FileNotFoundError:
        pass
    return {
        "sample_states": sample_pc,
        "correction_pc": capture_pc[:len(correction_idx)],
        "validation_pc": capture_pc[len(correction_idx):],
        "correction_base_idx": correction_idx,
        "validation_base_idx": validation_idx,
        "warmup_steps": int(warmup_steps),
        "warmup_diagnostics": warmup_diagnostics,
        "saturation_fraction": saturation,
        "scaler": scaler,
        "pca": pcas,
        "ready_mask": ready_mask,
        "config": config,
        "seed": seed,
        "identity": arrays["identity"],
    }


def _corrector_delta(y_fit, base_fit_logits, esn_fit, y_val, base_val_logits, esn_val, c_value, seed):
    control = LogisticRegression(C=c_value, max_iter=500, solver="lbfgs", random_state=seed)
    augmented = LogisticRegression(C=c_value, max_iter=500, solver="lbfgs", random_state=seed)
    base_fit = np.asarray(base_fit_logits, dtype=np.float64).reshape(-1, 1)
    base_val = np.asarray(base_val_logits, dtype=np.float64).reshape(-1, 1)
    control.fit(base_fit, y_fit)
    augmented.fit(np.column_stack((base_fit, esn_fit)), y_fit)
    p_control = control.predict_proba(base_val)[:, 1]
    p_augmented = augmented.predict_proba(np.column_stack((base_val, esn_val)))[:, 1]
    return float(log_loss(y_val, p_control, labels=[0, 1]) - log_loss(y_val, p_augmented, labels=[0, 1]))


def _trial_result(config: ESNConfig, arrays: dict, seed_data: list, screen_logits, y) -> dict:
    deltas = []
    corr_counts = []
    val_counts = []
    for item in seed_data:
        corr_idx = item["correction_base_idx"]
        val_idx = item["validation_base_idx"]
        corr_ok = (
            np.isfinite(screen_logits[corr_idx])
            & (arrays["label_available"][corr_idx] <= _utc_ns(CORRECTION_END) + NS_MINUTE)
            & item["ready_mask"][arrays["base_to_raw"][corr_idx]]
        )
        val_ok = np.isfinite(screen_logits[val_idx]) & item["ready_mask"][arrays["base_to_raw"][val_idx]]
        if corr_ok.sum() < 1000 or val_ok.sum() < 1000:
            raise ValueError("warm-up masks leave too few OOS decision rows for the frozen correction split")
        delta = _corrector_delta(
            y[corr_idx[corr_ok]], screen_logits[corr_idx[corr_ok]], item["correction_pc"][corr_ok],
            y[val_idx[val_ok]], screen_logits[val_idx[val_ok]], item["validation_pc"][val_ok],
            CONFIG["screening"]["calibration_regularization_C"], item["seed"],
        )
        deltas.append(delta)
        corr_counts.append(int(corr_ok.sum()))
        val_counts.append(int(val_ok.sum()))
    return {
        "config": _config_record(config),
        "seed_deltas": deltas,
        "mean_delta_logloss": float(np.mean(deltas)),
        "std_delta_logloss": float(np.std(deltas, ddof=0)),
        "positive_seed_count": int(np.sum(np.asarray(deltas) > 0.0)),
        "passes_screen": bool(np.mean(deltas) > 0.0 and np.sum(np.asarray(deltas) > 0.0) >= 2),
        "correction_fit_rows_by_seed": corr_counts,
        "search_validation_rows_by_seed": val_counts,
        "warmup_steps_by_seed": [int(item["warmup_steps"]) for item in seed_data],
        "warmup_diagnostics_by_seed": [item["warmup_diagnostics"] for item in seed_data],
        "saturation_fraction_by_seed": [float(item["saturation_fraction"]) for item in seed_data],
    }


def _run_optuna(identity: dict, arrays: dict) -> dict:
    result_path = OUT / "search_results.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("identity") == identity and saved.get("status") == "search_complete":
            return saved
    raw_logits = np.load(OUT / "screen_baseline_raw_score.npy", mmap_mode="r", allow_pickle=False)
    labels = arrays["target"]
    database = OUT / "optuna.sqlite3"
    study = optuna.create_study(
        study_name=CONFIG["experiment_id"], storage=f"sqlite:///{database.as_posix()}",
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=CONFIG["search"]["optuna_sampler_seed"]),
        load_if_exists=True,
    )
    saved_identity = study.user_attrs.get("experiment_identity")
    if saved_identity is not None and saved_identity != identity:
        raise RuntimeError("Optuna study identity differs from current data/code/config/split")
    if saved_identity is None:
        study.set_user_attr("experiment_identity", identity)
    # A killed process can leave RUNNING trials. Re-enqueue the exact parameters so the
    # identity-checked per-seed stream checkpoint can finish the same configuration.
    running = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING]
    for old in running:
        study.tell(old.number, state=optuna.trial.TrialState.FAIL)
        if old.params:
            study.enqueue_trial(old.params, user_attrs={"resumed_from_trial": old.number})

    max_trials = int(CONFIG["search"]["maximum_trials"])
    remaining_trials = max(0, max_trials - len(study.trials))
    total_remaining = float(CONFIG["search"]["whole_experiment_compute_limit_seconds"]) - _elapsed_run_seconds()
    search_allowance = min(
        float(CONFIG["search"]["time_limit_seconds"]),
        max(0.0, total_remaining - float(CONFIG["search"]["confirmation_and_final_fit_reserve_seconds"])),
    )

    def objective(trial):
        config = _config_from_trial(trial)
        trial.set_user_attr("config", _config_record(config))
        seed_data = []
        try:
            cache_key = hashlib.sha256(json.dumps(_config_record(config), sort_keys=True).encode()).hexdigest()
            for seed in SEEDS:
                seed_data.append(_stream_candidate(config, seed, arrays, cache_key=cache_key))
            outcome = _trial_result(config, arrays, seed_data, raw_logits, labels)
            trial.set_user_attr("outcome", outcome)
            return float(outcome["mean_delta_logloss"])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            trial.set_user_attr("rejected", str(exc))
            return -1.0

    search_start = time.monotonic()
    if remaining_trials and search_allowance > 0.0:
        study.optimize(
            objective, n_trials=remaining_trials, timeout=search_allowance,
            gc_after_trial=True, show_progress_bar=False,
            callbacks=[lambda _study, trial: _record_optuna_trial_time(trial)],
        )
    records = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        outcome = trial.user_attrs.get("outcome")
        if outcome is None:
            continue
        records.append({"trial_number": trial.number, "value": float(trial.value), **outcome})
    records.sort(key=lambda item: item["mean_delta_logloss"], reverse=True)
    result = {
        "identity": identity,
        "status": "search_complete",
        "study_name": CONFIG["experiment_id"],
        "study_db": database.relative_to(ROOT).as_posix(),
        "completed_trials": len(records),
        "attempted_trials": len(study.trials),
        "time_limit_seconds": CONFIG["search"]["time_limit_seconds"],
        "prior_failed_attempt_active_seconds": float(CONFIG["search"].get("prior_failed_attempt_active_seconds", 0.0)),
        "effective_allowance_seconds": search_allowance,
        "elapsed_seconds": time.monotonic() - search_start,
        "seed_is_hyperparameter": False,
        "seed_values": list(SEEDS),
        "candidate_results": records,
        "holdout_rows_materialized": 0,
    }
    _atomic_json(result_path, result)
    return result


def _evaluate_portfolio(configs: list[ESNConfig], arrays: dict, screen_logits, labels) -> dict:
    if not configs:
        return {"member_count": 0, "seed_deltas": [0.0] * len(SEEDS), "mean_delta_logloss": 0.0}
    seed_results = []
    for seed in SEEDS:
        generated = []
        for index, config in enumerate(configs):
            key = hashlib.sha256(json.dumps(_config_record(config), sort_keys=True).encode()).hexdigest()
            generated.append(_stream_candidate(config, seed, arrays, cache_key=f"portfolio-{key}-{index}"))
        corr_idx = generated[0]["correction_base_idx"]
        val_idx = generated[0]["validation_base_idx"]
        corr_features = np.column_stack([item["correction_pc"] for item in generated])
        val_features = np.column_stack([item["validation_pc"] for item in generated])
        corr_ready = np.logical_and.reduce([item["ready_mask"][arrays["base_to_raw"][corr_idx]] for item in generated])
        val_ready = np.logical_and.reduce([item["ready_mask"][arrays["base_to_raw"][val_idx]] for item in generated])
        corr_ok = (
            np.isfinite(screen_logits[corr_idx])
            & (arrays["label_available"][corr_idx] <= _utc_ns(CORRECTION_END) + NS_MINUTE)
            & corr_ready
        )
        val_ok = np.isfinite(screen_logits[val_idx]) & val_ready
        delta = _corrector_delta(
            labels[corr_idx[corr_ok]], screen_logits[corr_idx[corr_ok]], corr_features[corr_ok],
            labels[val_idx[val_ok]], screen_logits[val_idx[val_ok]], val_features[val_ok],
            CONFIG["screening"]["calibration_regularization_C"], seed,
        )
        seed_results.append({"seed": seed, "delta_logloss": delta, "fit_rows": int(corr_ok.sum()), "validation_rows": int(val_ok.sum())})
    deltas = [item["delta_logloss"] for item in seed_results]
    return {
        "member_count": len(configs), "seed_deltas": deltas,
        "mean_delta_logloss": float(np.mean(deltas)),
        "positive_seed_count": int(np.sum(np.asarray(deltas) > 0.0)),
        "per_seed": seed_results,
    }


def _select_portfolio(identity: dict, search: dict, arrays: dict) -> dict:
    path = OUT / "portfolio_selection.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("identity") == identity:
            return saved
    raw_logits = np.load(OUT / "screen_baseline_raw_score.npy", mmap_mode="r", allow_pickle=False)
    passing = [record for record in search["candidate_results"] if record["passes_screen"]]
    selected: list[ESNConfig] = []
    selected_records = []
    stages = []
    candidates = passing[:int(CONFIG["search"]["portfolio_shortlist_size"])]
    labels = arrays["target"]
    max_members = int(CONFIG["screening"]["portfolio_maximum_members"])
    for member_index in range(max_members):
        if member_index == 0:
            if not candidates:
                break
            first = candidates[0]
            config_data = dict(first["config"])
            config_data.pop("selected_columns", None)
            config_data["input_groups"] = tuple(config_data["input_groups"])
            selected.append(ESNConfig(**config_data))
            selected_records.append({
                "trial_number": first["trial_number"], "config": _config_record(selected[0]),
                "portfolio_result": {
                    "member_count": 1,
                    "seed_deltas": first["seed_deltas"],
                    "mean_delta_logloss": first["mean_delta_logloss"],
                    "positive_seed_count": first["positive_seed_count"],
                },
            })
            single_mean = first["mean_delta_logloss"]
            stages.append({"member_added": 1, "current_mean_delta_logloss": single_mean})
            continue
        best = None
        for record in candidates:
            if record["trial_number"] in {item["trial_number"] for item in selected_records}:
                continue
            config_data = dict(record["config"])
            config_data.pop("selected_columns", None)
            config_data["input_groups"] = tuple(config_data["input_groups"])
            candidate_config = ESNConfig(**config_data)
            result = _evaluate_portfolio(selected + [candidate_config], arrays, raw_logits, labels)
            result["trial_number"] = record["trial_number"]
            stages.append(result)
            if result["positive_seed_count"] < 2 or result["mean_delta_logloss"] <= 0.0:
                continue
            if best is None or result["mean_delta_logloss"] > best["mean_delta_logloss"]:
                best = result
                best_config = candidate_config
        if best is None:
            break
        selected.append(best_config)
        selected_records.append({"trial_number": best["trial_number"], "config": _config_record(best_config), "portfolio_result": best})
        current_mean = best["mean_delta_logloss"]
        stages.append({"member_added": member_index + 1, "current_mean_delta_logloss": current_mean})
        # Only add the next reservoir when the measured combined correction improves.
        if current_mean <= single_mean:
            selected.pop()
            selected_records.pop()
            break
    payload = {
        "identity": identity,
        "status": "portfolio_frozen_before_confirmation",
        "selection_source": "linear_incremental_screen" if selected else "none_passed_linear_screen",
        "shortlist_size": int(CONFIG["search"]["portfolio_shortlist_size"]),
        "candidate_trials_passing_screen": len(passing),
        "selected_members": selected_records,
        "member_count": len(selected),
        "portfolio_stages": stages,
        "portfolio_result": selected_records[-1]["portfolio_result"] if selected_records else _evaluate_portfolio([], arrays, raw_logits, labels),
        "selection_rule": "greedy addition on three common seeds; require positive mean and at least two positive seed deltas; stop when the combined correction does not improve",
        "holdout_rows_materialized": 0,
    }
    _atomic_json(path, payload)
    return payload


class _AugmentedMatrix:
    def __init__(self, baseline, extra):
        self.baseline = baseline
        self.extra = extra

    def __getitem__(self, rows):
        base = np.asarray(self.baseline[rows], dtype=np.float32)
        extra = np.asarray(self.extra[rows], dtype=np.float32)
        return np.column_stack((base, extra))


def _pca_payload(config: ESNConfig, seed: int, scaler: FeatureScaler, pcas: list,
                 warmup_steps: int, warmup_diagnostics: dict, identity: dict) -> dict:
    return {
        "engine_version": "btc_esn_v1",
        "experiment_id": CONFIG["experiment_id"],
        "identity": identity,
        "config": _config_record(config),
        "seed": int(seed),
        "fit_boundary_utc": CONFIG["split"]["fit_core_end_exclusive_utc"],
        "scaler": scaler.to_dict(),
        "pca": [
            {
                "mean": item.mean_.tolist(),
                "components": item.components_.tolist(),
                "explained_variance_ratio": item.explained_variance_ratio_.tolist(),
            }
            for item in pcas
        ],
        "warmup_steps": int(warmup_steps),
        "warmup_diagnostics": warmup_diagnostics,
        "component_count_per_reservoir": int(CONFIG["screening"]["state_components_per_reservoir"]),
        "feature_names": [
            f"esn_{hashlib.sha256(json.dumps(_config_record(config), sort_keys=True).encode()).hexdigest()[:10]}_seed_{seed}_r{reservoir}_pc{component}"
            for reservoir in range(config.state_count)
            for component in range(int(CONFIG["screening"]["state_components_per_reservoir"]))
        ],
        "feature_availability": "state update uses only closed event-time candles; actual receive-time latency is unavailable in source data",
    }


def _generate_full_pca_features(config: ESNConfig, seed: int, scaler: FeatureScaler, pcas: list,
                                warmup_steps: int, arrays: dict, output_path: Path, bundle_hash: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta = output_path.with_suffix(output_path.suffix + ".json")
    expected_meta = {"identity": arrays["identity"], "bundle_sha256": bundle_hash,
                     "config": _config_record(config), "seed": int(seed)}
    if output_path.exists() and output_meta.exists():
        if json.loads(output_meta.read_text(encoding="utf-8")) != expected_meta:
            raise RuntimeError("ESN feature matrix identity mismatch")
        return np.load(output_path, mmap_mode="r", allow_pickle=False)

    raw_opened = arrays["raw_opened"]
    raw_values = arrays["raw_x"]
    raw_valid = arrays["raw_valid"]
    base_to_raw = arrays["base_to_raw"]
    indices = [INPUT_COLUMNS.index(name) for name in config.selected_columns]
    selected_values = raw_values[:, indices]
    width = config.state_count * int(CONFIG["screening"]["state_components_per_reservoir"])
    if not output_path.exists():
        output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(len(base_to_raw), width))
        output[:] = np.nan
        output.flush()
    else:
        output = np.lib.format.open_memmap(output_path, mode="r+", dtype=np.float32)

    checkpoint_path = output_path.with_suffix(output_path.suffix + ".engine.json")
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")
    engine = ESNEngine(config, seed, scaler, warmup_steps=warmup_steps)
    start_row = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if any(progress.get(key) != value for key, value in expected_meta.items()):
            raise RuntimeError("final ESN stage checkpoint identity mismatch")
        engine = ESNEngine.load_checkpoint(checkpoint_path, expected_config=config, expected_seed=seed)
        start_row = int(progress["next_row"])
    else:
        output[:] = np.nan
        output.flush()

    for start in range(start_row, len(raw_opened), 65_536):
        end = min(len(raw_opened), start + 65_536)
        chunk_times = pd.to_datetime(raw_opened[start:end], unit="ns", utc=True)
        states, ready = engine.update_many(chunk_times, selected_values[start:end], valid_mask=raw_valid[start:end])
        left = int(np.searchsorted(base_to_raw, start, side="left"))
        right = int(np.searchsorted(base_to_raw, end, side="left"))
        if right > left:
            local = base_to_raw[left:right] - start
            keep = ready[local]
            if keep.any():
                selected = states[local[keep]]
                components = []
                for reservoir, pca in enumerate(pcas):
                    lower = reservoir * config.reservoir_size
                    upper = lower + config.reservoir_size
                    components.append(pca.transform(selected[:, lower:upper]).astype(np.float32))
                output[left:right][keep] = np.concatenate(components, axis=1)
        if end % 262_144 == 0 or end == len(raw_opened):
            output.flush()
            engine.save_checkpoint(checkpoint_path)
            _atomic_json(progress_path, {**expected_meta, "next_row": end})
    output.flush()
    del output
    _atomic_json(output_meta, expected_meta)
    checkpoint_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    return np.load(output_path, mmap_mode="r", allow_pickle=False)


def _save_esn_bundle(path: Path, config: ESNConfig, seed: int, generated: dict) -> str:
    payload = _pca_payload(config, seed, generated["scaler"], generated["pca"],
                           generated["warmup_steps"], generated["warmup_diagnostics"], generated["identity"])
    _atomic_json(path, payload)
    return _sha256(path)


def _verify_bundle_roundtrip(original_pcas: list, restored: list, config: ESNConfig, seed: int) -> None:
    rng = np.random.default_rng(seed ^ 0xB00D1E)
    for original, loaded in zip(original_pcas, restored):
        sample = rng.normal(size=(7, config.reservoir_size)).astype(np.float32)
        if not np.allclose(original.transform(sample), loaded.transform(sample), rtol=1e-6, atol=1e-6):
            raise RuntimeError("saved and restored ESN feature bundle produced different PCA coordinates")


def _load_esn_bundle(path: Path, *, expected_config: ESNConfig, expected_seed: int,
                     expected_identity: dict) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config_data = dict(payload["config"])
    config_data.pop("selected_columns", None)
    config_data["input_groups"] = tuple(config_data["input_groups"])
    config = ESNConfig(**config_data)
    if payload.get("identity") != expected_identity or config != expected_config or int(payload["seed"]) != int(expected_seed):
        raise RuntimeError("ESN feature bundle identity, configuration, or seed mismatch")
    scaler = FeatureScaler.from_dict(payload["scaler"])
    if tuple(scaler.columns) != config.selected_columns:
        raise RuntimeError("ESN feature bundle scaler columns differ from the selected inputs")
    projections = [FrozenProjection.from_dict(item) for item in payload["pca"]]
    if len(projections) != config.state_count:
        raise RuntimeError("ESN feature bundle projection count differs from reservoir count")
    return {
        "config": config, "seed": int(expected_seed), "scaler": scaler,
        "pca": projections, "warmup_steps": int(payload["warmup_steps"]),
        "warmup_diagnostics": payload["warmup_diagnostics"],
    }


def _downstream_checks(identity: dict, search: dict, portfolio: dict, arrays: dict) -> dict:
    result_path = OUT / "downstream_budget.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("identity") == identity:
            return saved
    selected_trials = {int(item["trial_number"]) for item in portfolio["selected_members"]}
    ranked = search["candidate_results"]
    outside = [item for item in ranked if int(item["trial_number"]) not in selected_trials]
    fixed_budget = int(CONFIG["search"]["downstream_candidate_budget"])
    candidates = outside[:fixed_budget]
    progress_path = OUT / "downstream_budget.progress.json"
    progress = {"identity": identity, "results": []}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise RuntimeError("downstream progress identity mismatch")
    labels = arrays["target"]
    decision = arrays["decision"]
    opened = arrays["base_opened"]
    label_available = arrays["label_available"]
    weights = np.where(decision, WEIGHT_DECISION, WEIGHT_OTHER).astype(np.float32)
    train_mask = _training_mask(opened, label_available, FIT_END)
    validation_mask = (
        decision & (opened >= _utc_ns(CORRECTION_END)) & (opened < _utc_ns(SEARCH_END))
    )
    rows = []
    for record in candidates:
        config_data = dict(record["config"])
        config_data.pop("selected_columns", None)
        config_data["input_groups"] = tuple(config_data["input_groups"])
        config = ESNConfig(**config_data)
        previous = next((item for item in progress["results"] if item["trial_number"] == int(record["trial_number"])), None)
        trial_result = previous or {"trial_number": int(record["trial_number"]), "config": _config_record(config), "seed_results": []}
        completed_seeds = {int(item["seed"]) for item in trial_result["seed_results"]}
        for seed in SEEDS:
            if seed in completed_seeds:
                continue
            member_key = f"downstream_trial_{record['trial_number']}_seed_{seed}"
            generated = _stream_candidate(config, seed, arrays, cache_key=member_key)
            bundle_path = OUT / "bundles" / f"{member_key}.json"
            bundle_hash = _save_esn_bundle(bundle_path, config, seed, generated)
            bundle = _load_esn_bundle(bundle_path, expected_config=config, expected_seed=seed, expected_identity=identity)
            _verify_bundle_roundtrip(generated["pca"], bundle["pca"], config, seed)
            feature_path = OUT / "downstream" / f"{member_key}.npy"
            esn = _generate_full_pca_features(
                config, seed, bundle["scaler"], bundle["pca"], bundle["warmup_steps"],
                arrays, feature_path, bundle_hash,
            )
            feature_ready = np.isfinite(esn).all(axis=1)
            shared_train = train_mask & feature_ready
            shared_validation = validation_mask & feature_ready
            if shared_train.sum() < 100_000 or shared_validation.sum() < 1_000:
                raise RuntimeError("downstream ESN feature mask leaves too few rows")
            baseline, _ = _fit_lgbm(arrays["base_x"], labels, weights, shared_train, seed)
            augmented, _ = _fit_lgbm(
                _AugmentedMatrix(arrays["base_x"], esn), labels, weights, shared_train, seed,
            )
            valid_idx = np.flatnonzero(shared_validation)
            p_base = baseline.predict(arrays["base_x"][valid_idx])
            p_aug = augmented.predict(_AugmentedMatrix(arrays["base_x"], esn)[valid_idx])
            trial_result["seed_results"].append({
                "seed": seed,
                "rows": int(len(valid_idx)),
                "baseline": _metric_pack(labels[valid_idx], p_base),
                "augmented": _metric_pack(labels[valid_idx], p_aug),
                "delta_logloss": float(log_loss(labels[valid_idx], p_base, labels=[0, 1]) - log_loss(labels[valid_idx], p_aug, labels=[0, 1])),
                "feature_count": int(len(BASE_FEATURES) + esn.shape[1]),
            })
            progress["results"] = [item for item in progress["results"] if item["trial_number"] != int(record["trial_number"])]
            progress["results"].append(trial_result)
            _atomic_json(progress_path, progress)
            del baseline, augmented, esn, generated
        trial_result["mean_delta_logloss"] = float(np.mean([item["delta_logloss"] for item in trial_result["seed_results"]]))
        rows.append(trial_result)
    payload = {
        "identity": identity,
        "status": "fixed_downstream_budget_complete",
        "budget": fixed_budget,
        "evaluated_candidates": len(rows),
        "candidate_selection": "highest individual screening ranks outside the frozen selected linear-screen portfolio",
        "results": rows,
        "holdout_rows_materialized": 0,
    }
    _atomic_json(result_path, payload)
    return payload


def _promote_downstream_candidate(identity: dict, portfolio: dict, downstream: dict) -> dict:
    if portfolio["member_count"]:
        return portfolio
    passing = [
        item for item in downstream["results"]
        if item["mean_delta_logloss"] > 0.0
        and sum(result["delta_logloss"] > 0.0 for result in item["seed_results"]) >= 2
    ]
    if not passing:
        return portfolio
    selected = max(passing, key=lambda item: item["mean_delta_logloss"])
    trial = selected
    promoted = dict(portfolio)
    promoted["selection_source"] = "fixed_downstream_lgbm_budget"
    promoted["member_count"] = 1
    promoted["selected_members"] = [{
        "trial_number": trial["trial_number"], "config": trial["config"],
        "downstream_validation_result": trial,
    }]
    promoted["downstream_promotion_rule"] = CONFIG["search"]["downstream_promotion_rule"]
    promoted["portfolio_result"] = {
        "member_count": 1,
        "mean_delta_logloss": trial["mean_delta_logloss"],
        "positive_seed_count": sum(result["delta_logloss"] > 0.0 for result in trial["seed_results"]),
        "selection_metric": "downstream LightGBM search-validation logloss",
    }
    _atomic_json(OUT / "portfolio_selection.json", promoted)
    return promoted


def _nvar_features(identity: dict, arrays: dict, training_boundary) -> tuple[np.ndarray, np.ndarray, dict]:
    path = OUT / "nvar_features.npy"
    meta_path = path.with_suffix(".npy.json")
    expected = {"identity": identity, "training_boundary": pd.Timestamp(training_boundary).isoformat(),
                "columns": list(CONFIG["nvar_control"]["input_columns"]),
                "max_lag": int(CONFIG["nvar_control"]["maximum_lag"])}
    valid_path = OUT / "nvar_valid.npy"
    bundle_path = OUT / "bundles" / "nvar.json"
    if path.exists() and meta_path.exists() and valid_path.exists() and bundle_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("NVAR feature artifact identity mismatch")
        return np.load(path, mmap_mode="r", allow_pickle=False), np.load(valid_path, mmap_mode="r", allow_pickle=False), json.loads(bundle_path.read_text(encoding="utf-8"))

    columns = tuple(CONFIG["nvar_control"]["input_columns"])
    indices = [INPUT_COLUMNS.index(name) for name in columns]
    raw = arrays["raw_x"]
    fit_mask = (arrays["raw_opened"] < _utc_ns(training_boundary)) & arrays["raw_valid"]
    scaler = FeatureScaler(columns).fit(np.asarray(raw[fit_mask][:, indices], dtype=np.float64))
    standardized = scaler.transform(np.asarray(raw[:, indices], dtype=np.float64))
    raw_times = pd.to_datetime(arrays["raw_opened"], unit="ns", utc=True)
    valid = np.asarray(arrays["raw_valid"], dtype=bool)
    nvar, names, nvar_valid = build_nvar_features(
        standardized, raw_times, input_columns=columns,
        max_lag=int(CONFIG["nvar_control"]["maximum_lag"]), valid_mask=valid,
    )
    if nvar.shape[1] != int(CONFIG["nvar_control"]["feature_count"]):
        raise RuntimeError("NVAR feature count differs from frozen 76-feature control")
    _save_npy(path, nvar)
    _save_npy(valid_path, nvar_valid)
    bundle = {
        "experiment_id": CONFIG["experiment_id"], "identity": identity,
        "fit_boundary_utc": pd.Timestamp(training_boundary).isoformat(),
        "input_columns": list(columns), "scaler": scaler.to_dict(),
        "maximum_lag": int(CONFIG["nvar_control"]["maximum_lag"]),
        "feature_names": list(names), "feature_count": len(names),
    }
    _atomic_json(bundle_path, bundle)
    _atomic_json(meta_path, expected)
    return np.load(path, mmap_mode="r", allow_pickle=False), np.load(valid_path, mmap_mode="r", allow_pickle=False), bundle


def _block_bootstrap_delta(y, pred_a, pred_b, decision_times_ns, *, seed: int, replicates: int, block_days: int = 7) -> dict:
    y = np.asarray(y, dtype=np.int8)
    a = np.clip(np.asarray(pred_a, dtype=np.float64), 1e-7, 1 - 1e-7)
    b = np.clip(np.asarray(pred_b, dtype=np.float64), 1e-7, 1 - 1e-7)
    dates = pd.to_datetime(decision_times_ns, unit="ns", utc=True).floor("D").as_unit("ns").asi8
    day_ns = pd.Timedelta(days=1).value
    unique_days = np.arange(dates.min(), dates.max() + day_ns, day_ns, dtype=np.int64)
    day_index = np.searchsorted(unique_days, dates)
    delta = -(y * np.log(a) + (1 - y) * np.log1p(-a)) + (y * np.log(b) + (1 - y) * np.log1p(-b))
    day_sum = np.bincount(day_index, weights=delta, minlength=len(unique_days))
    day_count = np.bincount(day_index, minlength=len(unique_days)).astype(np.float64)
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(len(unique_days) / block_days))
    estimates = np.empty(replicates, dtype=np.float64)
    offsets = np.arange(block_days)
    for draw in range(replicates):
        starts = rng.integers(0, len(unique_days), size=blocks)
        sampled = ((starts[:, None] + offsets[None, :]) % len(unique_days)).reshape(-1)
        estimates[draw] = day_sum[sampled].sum() / day_count[sampled].sum()
    return {
        "mean_delta_logloss_a_minus_b": float(delta.mean()),
        "ci_2_5": float(np.quantile(estimates, 0.025)),
        "ci_97_5": float(np.quantile(estimates, 0.975)),
        "replicates": int(replicates), "block_days": int(block_days),
        "interpretation": "conditional on the selected development procedure",
    }


def _final_comparison(identity: dict, portfolio: dict, downstream: dict, arrays: dict) -> dict:
    result_path = OUT / "final_comparison.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            saved.get("identity") == identity
            and saved.get("status") == "confirmation_complete"
            and saved.get("finalization_runner_sha256") == _sha256(Path(__file__))
            and saved.get("finalization_engine_sha256") == _sha256(ROOT / "features/esn_reservoir.py")
        ):
            return saved
    started = time.monotonic()
    labels = arrays["target"]
    opened = arrays["base_opened"]
    label_available = arrays["label_available"]
    decision = arrays["decision"]
    weights = np.where(decision, WEIGHT_DECISION, WEIGHT_OTHER).astype(np.float32)
    train_boundary = SEARCH_END
    train_mask = _training_mask(opened, label_available, train_boundary)
    confirmation_mask = decision & (opened >= _utc_ns(CONFIRMATION_START)) & (opened < _utc_ns(DEVELOPMENT_CUTOFF))

    selected_configs = []
    for item in portfolio["selected_members"]:
        config_data = dict(item["config"])
        config_data.pop("selected_columns", None)
        config_data["input_groups"] = tuple(config_data["input_groups"])
        selected_configs.append(ESNConfig(**config_data))
    final_esn = []
    for member_index, config in enumerate(selected_configs):
        seed_features = []
        for seed in FINAL_SEEDS:
            key = f"final_member_{member_index}_seed_{seed}"
            generated = _stream_candidate(config, seed, arrays, cache_key=key)
            bundle_path = OUT / "bundles" / f"{key}.json"
            bundle_hash = _save_esn_bundle(bundle_path, config, seed, generated)
            bundle = _load_esn_bundle(bundle_path, expected_config=config, expected_seed=seed, expected_identity=identity)
            _verify_bundle_roundtrip(generated["pca"], bundle["pca"], config, seed)
            feature_path = OUT / "final_features" / f"{key}.npy"
            seed_features.append(_generate_full_pca_features(
                config, seed, bundle["scaler"], bundle["pca"], bundle["warmup_steps"],
                arrays, feature_path, bundle_hash,
            ))
            del generated
        final_esn.append(seed_features)

    nvar, nvar_valid, nvar_bundle = _nvar_features(identity, arrays, train_boundary)
    nvar_base_valid = np.asarray(nvar_valid[arrays["base_to_raw"]], dtype=bool)
    if final_esn:
        esn_valid_by_seed = [np.logical_and.reduce([np.isfinite(item[:, :]).all(axis=1) for item in member]) for member in final_esn]
        esn_common_valid = np.logical_and.reduce(esn_valid_by_seed)
    else:
        esn_common_valid = np.ones(len(opened), dtype=bool)
    common_valid = nvar_base_valid & esn_common_valid
    common_train = train_mask & common_valid
    common_confirmation = confirmation_mask & common_valid
    confirmation_indices = np.flatnonzero(common_confirmation)
    if common_train.sum() < 100_000 or len(confirmation_indices) < 1_000:
        raise RuntimeError("common A/B/C mask leaves too few rows for frozen confirmation")

    seed_predictions = {variant: [] for variant in ("A", "B", "C")}
    per_seed = []
    prediction_dir = OUT / "final_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    confirmation_digest = hashlib.sha256(confirmation_indices.tobytes()).hexdigest()
    prediction_identity = {**identity, "confirmation_indices_sha256": confirmation_digest}
    for seed in FINAL_SEEDS:
        cached = {}
        for variant in ("A", "B", "C"):
            cached[variant] = _load_cached(prediction_dir / f"{variant}_seed_{seed}.npy", prediction_identity)
        if all(value is not None for value in cached.values()):
            for variant in ("A", "B", "C"):
                probability = np.asarray(cached[variant], dtype=np.float32)
                seed_predictions[variant].append(probability)
            per_seed.append({
                "seed": seed,
                **{variant: _metric_pack(labels[confirmation_indices], cached[variant]) for variant in ("A", "B", "C")},
            })
            continue
        baseline, _ = _fit_lgbm(arrays["base_x"], labels, weights, common_train, seed)
        pred_a = baseline.predict(arrays["base_x"][confirmation_indices])
        seed_predictions["A"].append(pred_a.astype(np.float32))
        if final_esn:
            esn_columns = [member[FINAL_SEEDS.index(seed)] for member in final_esn]
            esn_matrix = np.column_stack([item[confirmation_indices] for item in esn_columns])
            esn_full = np.column_stack([item[common_train] for item in esn_columns])
            # This wrapper maps full-row selections to already selected augmentation rows.
            class _AlignedAugmented:
                def __getitem__(self, rows):
                    base_rows = np.asarray(rows)
                    selected_rows = np.flatnonzero(common_train)[base_rows] if base_rows.dtype.kind in "iu" else np.flatnonzero(common_train)[np.flatnonzero(base_rows)]
                    base = np.asarray(arrays["base_x"][selected_rows], dtype=np.float32)
                    extra = np.asarray(esn_full[base_rows], dtype=np.float32) if base_rows.dtype.kind in "iu" else np.asarray(esn_full[np.flatnonzero(base_rows)], dtype=np.float32)
                    return np.column_stack((base, extra))
            augmented_b, _ = _fit_lgbm(_AlignedAugmented(), labels[common_train], weights[common_train], np.ones(int(common_train.sum()), dtype=bool), seed)
            pred_b = augmented_b.predict(np.column_stack((arrays["base_x"][confirmation_indices], esn_matrix)))
        else:
            augmented_b = baseline
            pred_b = pred_a.copy()
        seed_predictions["B"].append(pred_b.astype(np.float32))

        nvar_confirm = np.asarray(nvar[arrays["base_to_raw"][confirmation_indices]], dtype=np.float32)
        nvar_train = np.asarray(nvar[arrays["base_to_raw"][common_train]], dtype=np.float32)
        # Both matrices use the identical row masks; all NVAR values are finite on this mask.
        class _AlignedNvar:
            def __getitem__(self, rows):
                if np.asarray(rows).dtype.kind in "iu":
                    selected = np.asarray(rows, dtype=np.int64)
                    return np.column_stack((arrays["base_x"][np.flatnonzero(common_train)[selected]], nvar_train[selected]))
                selected = np.flatnonzero(rows)
                return np.column_stack((arrays["base_x"][np.flatnonzero(common_train)[selected]], nvar_train[selected]))
        augmented_c, _ = _fit_lgbm(_AlignedNvar(), labels[common_train], weights[common_train], np.ones(int(common_train.sum()), dtype=bool), seed)
        pred_c = augmented_c.predict(np.column_stack((arrays["base_x"][confirmation_indices], nvar_confirm)))
        seed_predictions["C"].append(pred_c.astype(np.float32))
        per_seed.append({
            "seed": seed,
            "A": _metric_pack(labels[confirmation_indices], pred_a),
            "B": _metric_pack(labels[confirmation_indices], pred_b),
            "C": _metric_pack(labels[confirmation_indices], pred_c),
        })
        final_dir = OUT / "models"
        baseline.save_model(str(final_dir / f"confirmation_A_seed_{seed}.txt"))
        if final_esn:
            augmented_b.save_model(str(final_dir / f"confirmation_B_seed_{seed}.txt"))
        augmented_c.save_model(str(final_dir / f"confirmation_C_seed_{seed}.txt"))
        for variant, probability in (("A", pred_a), ("B", pred_b), ("C", pred_c)):
            _save_stage_array(prediction_dir / f"{variant}_seed_{seed}.npy", np.asarray(probability, dtype=np.float32), prediction_identity)
        del baseline, augmented_b, augmented_c

    predictions = {variant: np.mean(np.stack(values), axis=0) for variant, values in seed_predictions.items()}
    confirmation_y = labels[confirmation_indices]
    avg_metrics = {variant: _metric_pack(confirmation_y, probability) for variant, probability in predictions.items()}
    bootstrap_seed = int(CONFIG["final_comparison"]["bootstrap_seed"])
    reps = 2_000
    bootstrap = {
        "B_vs_A": _block_bootstrap_delta(
            confirmation_y, predictions["A"], predictions["B"], arrays["base_decision_at"][confirmation_indices],
            seed=bootstrap_seed, replicates=reps,
        ),
        "C_vs_A": _block_bootstrap_delta(
            confirmation_y, predictions["A"], predictions["C"], arrays["base_decision_at"][confirmation_indices],
            seed=bootstrap_seed + 1, replicates=reps,
        ),
    }
    payload = {
        "identity": identity,
        "finalization_runner_sha256": _sha256(Path(__file__)),
        "finalization_engine_sha256": _sha256(ROOT / "features/esn_reservoir.py"),
        "status": "confirmation_complete",
        "confirmation_is_development_only": True,
        "selection_and_hyperparameter_caveat": CONFIG["baseline"]["selection_independence"],
        "common_training_rows": int(common_train.sum()),
        "common_confirmation_decision_rows": int(len(confirmation_indices)),
        "training_label_available_by_utc": CONFIG["split"]["confirmation_start_utc"],
        "confirmation_start_utc": CONFIG["split"]["confirmation_start_utc"],
        "confirmation_end_exclusive_utc": CONFIG["split"]["confirmation_end_exclusive_utc"],
        "seed_results": per_seed,
        "metrics_mean_seed_prediction": avg_metrics,
        "paired_block_bootstrap": bootstrap,
        "feature_counts": {
            "A": len(BASE_FEATURES),
            "B": len(BASE_FEATURES) + (int(final_esn[0][0].shape[1]) * len(final_esn) if final_esn else 0),
            "C": len(BASE_FEATURES) + len(nvar_bundle["feature_names"]),
        },
        "portfolio_member_count": len(selected_configs),
        "nvar_feature_count": len(nvar_bundle["feature_names"]),
        "lightgbm": {"rounds": CONFIG["baseline"]["lightgbm"]["boosting_rounds"], "common_seeds": list(FINAL_SEEDS)},
        "elapsed_seconds": float(time.monotonic() - started),
        "holdout_rows_materialized": 0,
        "holdout_evaluation": False,
        "polymarket_replay": False,
    }
    for variant, values in seed_predictions.items():
        _save_stage_array(prediction_dir / f"{variant}_seed_mean.npy", np.mean(np.stack(values), axis=0), prediction_identity)
    _atomic_json(result_path, payload)
    return payload


def _write_report(identity: dict, prepared: dict, confirmation_prepared: dict,
                  search: dict, portfolio: dict, downstream: dict, final: dict,
                  total_seconds: float) -> None:
    def fmt(metric):
        return f"{metric['logloss']:.6f} / {metric['brier']:.6f} / {metric['auc']:.5f}"

    lines = [
        "# BTC ESN reservoir portfolio — development evaluation (2026-10-20)",
        "",
        "## Verdict",
        "",
    ]
    b_delta = final["metrics_mean_seed_prediction"]["A"]["logloss"] - final["metrics_mean_seed_prediction"]["B"]["logloss"]
    c_delta = final["metrics_mean_seed_prediction"]["A"]["logloss"] - final["metrics_mean_seed_prediction"]["C"]["logloss"]
    b_ci = final["paired_block_bootstrap"]["B_vs_A"]
    c_ci = final["paired_block_bootstrap"]["C_vs_A"]
    if portfolio["member_count"] == 0:
        lines.append("No ESN candidate passed the frozen three-seed incremental screen; B equals the BASE70 baseline.")
    elif b_delta > 0:
        if b_ci["ci_2_5"] <= 0.0 <= b_ci["ci_97_5"]:
            lines.append(f"The selected ESN portfolio reduced confirmation log loss by {b_delta:.6f} versus A, but its 95% paired block-bootstrap interval includes zero.")
        else:
            lines.append(f"The selected ESN portfolio reduced confirmation log loss by {b_delta:.6f} versus A on the frozen development confirmation interval.")
    else:
        lines.append(f"The selected ESN portfolio did not reduce confirmation log loss versus A (A−B delta {b_delta:.6f}).")
    if c_ci["ci_2_5"] <= 0.0 <= c_ci["ci_97_5"]:
        lines.append(f"NVAR confirmation A−C log-loss delta was {c_delta:.6f}; its 95% paired block-bootstrap interval includes zero.")
    else:
        lines.append(f"NVAR confirmation A−C log-loss delta was {c_delta:.6f}.")
    member_details = [
        f"trial {item['trial_number']} {item['config']['architecture']}/{item['config']['topology']} "
        f"{item['config']['reservoir_count']}×{item['config']['reservoir_size']}, "
        f"inputs: {', '.join(item['config']['selected_columns'])}"
        for item in portfolio["selected_members"]
    ]
    lines += [
        "",
        "The confirmation scores are conditional on the pre-existing BASE70 feature selection and LightGBM tuning, both of which used the full development history through 2026-06-30. They are not independent out-of-sample estimates for the complete baseline selection procedure. No trading profitability claim is made.",
        "",
        "## A/B/C confirmation",
        "",
        "Log loss / Brier / AUC; means over common LightGBM seeds on identical decision rows.",
        "",
        "| Variant | Features | Log loss / Brier / AUC | Paired 7-day block bootstrap delta log loss (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for variant, label in (("A", "BASE70"), ("B", "BASE70 + selected ESN portfolio"), ("C", "BASE70 + NVAR lags/quadratics")):
        pair = None if variant == "A" else final["paired_block_bootstrap"][f"{variant}_vs_A"]
        interval = "—" if pair is None else f"{pair['mean_delta_logloss_a_minus_b']:.6f} [{pair['ci_2_5']:.6f}, {pair['ci_97_5']:.6f}]"
        lines.append(f"| {label} | {final['feature_counts'][variant]} | {fmt(final['metrics_mean_seed_prediction'][variant])} | {interval} |")
    lines += [
        "",
        f"Common confirmation decision rows: {final['common_confirmation_decision_rows']:,}; common training rows: {final['common_training_rows']:,}; seeds: {', '.join(map(str, FINAL_SEEDS))}.",
        "",
        "## Frozen split and data contract",
        "",
        f"- Development range: `{CONFIG['development_data_start_utc']}` to exclusive `{CONFIG['development_cutoff_exclusive_utc']}`.",
        f"- Fit core: `{CONFIG['split']['fit_core_start_utc']}` to `{CONFIG['split']['fit_core_end_exclusive_utc']}`.",
        f"- Correction fit: `{CONFIG['split']['correction_fit_start_utc']}` to `{CONFIG['split']['correction_fit_end_exclusive_utc']}`.",
        f"- Search validation: `{CONFIG['split']['search_validation_start_utc']}` to `{CONFIG['split']['search_validation_end_exclusive_utc']}`.",
        f"- Confirmation: `{CONFIG['split']['confirmation_start_utc']}` to exclusive `{CONFIG['split']['confirmation_end_exclusive_utc']}`.",
        f"- Target verified row by row: `{CONFIG['baseline']['target_definition']}`; nominal decision is `Opened+1m`; label availability is `Opened+7m`.",
        "- Training purge uses label availability `<=` the first nominal decision in each next block. The source has no actual receive timestamps, so delivery by T−60 cannot be empirically verified.",
        f"- Training weights: decision rows {WEIGHT_DECISION}; other eligible rows {WEIGHT_OTHER}. Validation uses unweighted decision rows.",
        f"- Search-stage data stopped at `{prepared['materialization_cutoff_utc']}` and label availability before that boundary. Confirmation rows were materialized only after portfolio selection. Full development rows then loaded: {confirmation_prepared['development_raw_rows']:,} raw and {confirmation_prepared['development_baseline_rows']:,} common BASE70; holdout rows materialized: 0.",
        "- No project holdout observations, labels, predictions, scores, or book data were read for this experiment. No holdout evaluation, live change, order, or Polymarket replay was run.",
        "",
        "## Search and compute",
        "",
        f"- Optuna trials attempted/eligible: {search['attempted_trials']}/{search['completed_trials']} (limit {CONFIG['search']['maximum_trials']}; one trial was rejected by the frozen reservoir-saturation check). Search timeout: {CONFIG['search']['time_limit_seconds']} s.",
        f"- Two earlier pre-score attempts used {CONFIG['search'].get('prior_failed_attempt_active_seconds', 0.0):.1f} s combined; neither produced a completed candidate, and both are included in the four-hour compute cap.",
        "- Two finalization attempts stopped on code errors and were resumed from frozen search and confirmation artifacts. The completed-stage elapsed time below excludes those failed calls, so it is a lower bound on total active runtime.",
        f"- Screen objective: `logloss(control) - logloss(augmented)`; mean positive across three common reservoir seeds and positive for at least two seeds to pass.",
        f"- Selected portfolio size: {portfolio['member_count']} ({portfolio.get('selection_source', 'unknown')}); members: {'; '.join(member_details) if member_details else 'none'}; fixed downstream tree checks: {downstream['evaluated_candidates']}/{downstream['budget']}.",
        f"- Total measured active elapsed time: {total_seconds:.1f} s of the frozen {CONFIG['search']['whole_experiment_compute_limit_seconds']} s cap.",
        "- State generation used CPU with block processing. Benchmark: [runtime report](btc_esn_portfolio_20261020_benchmark.md). First-call JIT and restore timings are reported there.",
        "- The confirmation bootstrap uses 2,000 paired circular seven-day UTC blocks. Its interval is conditional on the selected procedure and does not adjust for all search attempts.",
        "",
        "## Artifacts and reproduction",
        "",
        f"- Configuration: `configs/research/btc_esn_portfolio_20261020.json`.",
        f"- Run manifest: `reports/btc_preopen/btc_esn_portfolio_20261020_manifest.json`.",
        f"- Results and resumable Optuna database: `{OUT.relative_to(ROOT).as_posix()}/`.",
        f"- ESN feature bundle and model files are under `{OUT.relative_to(ROOT).as_posix()}/bundles` and `models`.",
        "- Run or resume with: `python run_btc_esn_portfolio_20261020.py`.",
        f"- Identity: runner SHA-256 `{identity['runner_sha256']}`; source data size {identity['source_data']['bytes']:,} bytes; baseline data size {identity['baseline_data']['bytes']:,} bytes.",
        f"- Finalization runner SHA-256: `{final.get('finalization_runner_sha256', identity['runner_sha256'])}`.",
        f"- Finalization ESN engine SHA-256: `{final.get('finalization_engine_sha256', identity['engine_sha256'])}`.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    global RUN_STARTED, RUN_PRIOR_SECONDS, RUN_IDENTITY
    RUN_STARTED = time.monotonic()
    RUN_PRIOR_SECONDS = float(CONFIG["search"].get("prior_failed_attempt_active_seconds", 0.0))
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    TRIALS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    identity = _identity()
    RUN_IDENTITY = identity
    runner_recovery = None
    ledger_path = OUT / "runtime_ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("identity") != identity:
            recovered_identity = _recover_completed_search_identity(identity, ledger)
            if recovered_identity is None:
                raise RuntimeError("runtime ledger identity mismatch")
            identity = recovered_identity
            RUN_IDENTITY = identity
            runner_recovery = {
                "recovered_search_runner_sha256": identity["runner_sha256"],
                "recovered_search_engine_sha256": identity["engine_sha256"],
                "finalization_runner_sha256": _sha256(Path(__file__)),
                "finalization_engine_sha256": _sha256(ROOT / "features/esn_reservoir.py"),
                "reason": "resume finalization after fixing stream-prefix alignment and read-only NVAR validity-mask errors",
            }
        RUN_PRIOR_SECONDS = _runtime_total(ledger)
    _write_frozen_manifest(identity)
    stage_start = time.monotonic()
    prepared = _prepare_data(identity, materialization_cutoff=SEARCH_END, prep_dir=PREP / "search")
    _record_stage_time("prepare_development_data", time.monotonic() - stage_start)
    arrays = _load_arrays(PREP / "search", identity)
    stage_start = time.monotonic()
    screen_baseline = _screen_baseline(identity, arrays)
    _record_stage_time("screen_baseline", time.monotonic() - stage_start)
    stage_start = time.monotonic()
    search = _run_optuna(identity, arrays)
    _record_stage_time("optuna_search", time.monotonic() - stage_start)
    _check_compute_cap()
    stage_start = time.monotonic()
    portfolio = _select_portfolio(identity, search, arrays)
    _record_stage_time("portfolio_selection", time.monotonic() - stage_start)
    _check_compute_cap()
    stage_start = time.monotonic()
    downstream = _downstream_checks(identity, search, portfolio, arrays)
    _record_stage_time("downstream_budget", time.monotonic() - stage_start)
    portfolio = _promote_downstream_candidate(identity, portfolio, downstream)
    _check_compute_cap()
    stage_start = time.monotonic()
    confirmation_prepared = _prepare_data(
        identity, materialization_cutoff=DEVELOPMENT_CUTOFF,
        prep_dir=PREP / "confirmation",
    )
    _record_stage_time("confirmation_data_materialization", time.monotonic() - stage_start)
    final_arrays = _load_arrays(PREP / "confirmation", identity)
    stage_start = time.monotonic()
    try:
        final = _final_comparison(identity, portfolio, downstream, final_arrays)
    finally:
        _record_stage_time("confirmation_and_final_fit", time.monotonic() - stage_start)
    total_seconds = _elapsed_run_seconds()
    stage_start = time.monotonic()
    _write_report(identity, prepared, confirmation_prepared, search, portfolio, downstream, final, total_seconds)
    manifest = {
        "experiment_id": CONFIG["experiment_id"], "status": "complete_development_only",
        "identity": identity,
        "search_data_materialization": prepared,
        "development": confirmation_prepared,
        "data_contract": {
            "target": CONFIG["baseline"]["target_definition"],
            "nominal_decision_at": CONFIG["baseline"]["decision_definition"],
            "feature_availability": CONFIG["baseline"]["feature_availability"],
            "training_weights": CONFIG["baseline"]["training_weights"],
            "baseline_feature_order": list(BASE_FEATURES),
            "baseline_feature_origin": CONFIG["baseline"]["source_selection"],
            "baseline_selection_independence": CONFIG["baseline"]["selection_independence"],
        },
        "split": CONFIG["split"],
        "search": {key: value for key, value in search.items() if key != "candidate_results"},
        "portfolio": portfolio,
        "downstream_budget": downstream,
        "confirmation": final,
        "performance": {
            "benchmark_report": CONFIG["benchmark_report_path"],
            "total_active_elapsed_seconds": total_seconds,
            "compute_cap_seconds": CONFIG["search"]["whole_experiment_compute_limit_seconds"],
            "compute_cap_exceeded": total_seconds > float(CONFIG["search"]["whole_experiment_compute_limit_seconds"]),
        },
        "holdout_rows_materialized": 0, "holdout_evaluation": False,
        "polymarket_book_replay": False, "live_configuration_changes": False, "live_orders": False,
        "finalization_runner_sha256": final.get("finalization_runner_sha256", identity["runner_sha256"]),
        "finalization_engine_sha256": final.get("finalization_engine_sha256", identity["engine_sha256"]),
        "runner_code_recovery": runner_recovery,
        "report_path": CONFIG["experiment_report_path"],
    }
    _atomic_json(MANIFEST_PATH, manifest)
    _record_stage_time("reports_and_manifest", time.monotonic() - stage_start)
    state = {
        "identity": identity,
        "status": "search_and_portfolio_complete",
        "search_data_materialization": prepared,
        "development": confirmation_prepared,
        "screen_baseline": screen_baseline,
        "search": {key: value for key, value in search.items() if key != "candidate_results"},
        "portfolio": portfolio,
        "downstream": downstream,
        "final": final,
        "runner_code_recovery": runner_recovery,
        "holdout_rows_materialized": 0,
    }
    _atomic_json(OUT / "run_state.json", state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
