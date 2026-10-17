"""Run the frozen BTC conditional market-offset experiment."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import run_btc_market_probability_experiment as probability
import run_btc_preopen_fee_correction as fee_correction
import run_btc_preopen_research as research
import run_btc_preopen_t59_reassessment as replay


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_conditional_market_offset_20261018.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / "data/analysis/polymarket/BTC/btc_conditional_market_offset_20261018"
REPORT_PATH = ROOT / "reports/btc_preopen/btc_conditional_market_offset_20261018.md"
RUN_PATH = OUTPUT_DIR / "run_manifest.json"
FREEZE_PATH = OUTPUT_DIR / "frozen_development_bundle.json"
EPSILON = float(CONFIG["sample"]["logit_epsilon"])
FEATURES = (
    "logit_p_market", "d_btc_minus_market", "btc_return_1m", "btc_return_5m",
    "btc_return_15m", "btc_sigma_5m", "futures_index_basis_rel_1m",
    "spread_up", "spread_down",
)
PREDICTION_COLUMNS = {"A": "p_A_market", "B": "p_B_market_only", "C": "p_C_market_plus_btc", "D": "p_D_conditional"}
DEV_CUTOFF = pd.Timestamp(CONFIG["sample"]["development_cutoff_exclusive_utc"])
HOLDOUT_START = pd.Timestamp(CONFIG["sample"]["holdout_start_inclusive_utc"])
HOLDOUT_CUTOFF = pd.Timestamp(CONFIG["sample"]["holdout_cutoff_exclusive_utc"])
DEV_START = pd.Timestamp(CONFIG["sample"]["development_start_inclusive_utc"])
EXPERIMENT_STARTED: float | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _peak_working_set_bytes() -> int | None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

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
        get_process_memory_info.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
        )
        get_process_memory_info.restype = wintypes.BOOL
        ok = get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb)
        return int(counters.peak_working_set_size) if ok else None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if os.name == "nt" else value * 1024
    except (ImportError, AttributeError, OSError):
        return None


def _check_budget() -> None:
    if EXPERIMENT_STARTED is None:
        return
    elapsed = time.perf_counter() - EXPERIMENT_STARTED
    if elapsed > float(CONFIG["budget"]["maximum_wall_seconds"]):
        raise RuntimeError(f"Experiment reached its preregistered wall-time limit at {elapsed:.1f}s")
    peak_working_set = _peak_working_set_bytes()
    if peak_working_set is None:
        raise RuntimeError("Cannot verify the preregistered working-memory cap on this platform")
    memory_cap = float(CONFIG["budget"]["maximum_working_memory_gib"]) * 1024**3
    if peak_working_set > memory_cap:
        raise RuntimeError(f"Experiment reached its preregistered peak working-memory cap: {peak_working_set / 1024**3:.2f} GiB")


def _logit(probability) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(values) - np.log1p(-values)


def _apply_offset_correction(p_market, correction) -> np.ndarray:
    return expit(_logit(p_market) + np.asarray(correction, dtype=np.float64))


def _brier_decomposition(y, p_market, p_new) -> dict:
    outcome = np.asarray(y, dtype=np.float64)
    market = np.asarray(p_market, dtype=np.float64)
    new = np.asarray(p_new, dtype=np.float64)
    delta = new - market
    error = outcome - market
    movement = float(np.mean(np.square(delta)))
    alignment = float(2.0 * np.mean(delta * error))
    difference = float(np.mean(np.square(new - outcome)) - np.mean(np.square(market - outcome)))
    if not math.isclose(difference, movement - alignment, rel_tol=1e-10, abs_tol=1e-12):
        raise AssertionError("Exact Brier correction decomposition failed")
    return {"delta_brier": difference, "movement_cost": movement, "error_alignment": alignment}


def _fit_logistic(train: pd.DataFrame, columns: tuple[str, ...]) -> LogisticRegression:
    model = LogisticRegression(C=10.0, penalty="l2", solver="lbfgs", fit_intercept=True, max_iter=2000)
    with threadpool_limits(limits=4):
        model.fit(train.loc[:, columns].to_numpy(dtype=np.float64), train.official_outcome_up.astype(int).to_numpy())
    return model


def _predict_logistic(model: LogisticRegression, frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    if isinstance(model, dict):
        x = frame.loc[:, columns].to_numpy(dtype=np.float64)
        coefficient = np.asarray(model["coefficients"], dtype=np.float64)
        return expit(float(model["intercept"]) + x @ coefficient)
    with threadpool_limits(limits=4):
        return model.predict_proba(frame.loc[:, columns].to_numpy(dtype=np.float64))[:, 1]


def _load_causal_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    fee_correction._configure_output_paths()
    _, _, source_holdout = fee_correction._verify_reusable_sources()
    source_research = research._read_json(fee_correction.SOURCE_RESEARCH_MANIFEST)
    predictions, prediction_identity = research._load_model_predictions(include_holdout=True)
    t59, t60, book_record = research._load_enriched_book_snapshots(source_research)
    if book_record.get("t60_coverage_sha256") != CONFIG["source_artifacts"]["t60_snapshot_sha256"]:
        raise RuntimeError("The frozen T-60 snapshot hash differs from the preregistered source")
    if book_record.get("t59_coverage_sha256") != CONFIG["source_artifacts"]["t59_snapshot_sha256"]:
        raise RuntimeError("The frozen T-59 snapshot hash differs from the preregistered source")
    if prediction_identity["walk_forward_manifest_sha256"] != CONFIG["source_artifacts"]["development_walk_forward_manifest_sha256"]:
        raise RuntimeError("The frozen A_frozen walk-forward manifest changed")
    for filename, expected in CONFIG["source_artifacts"]["development_oof_sha256"].items():
        path = fee_correction.SOURCE_RESEARCH_DIR / "oof" / filename
        if _sha256(path) != expected:
            raise RuntimeError(f"Frozen A_frozen development predictions changed: {filename}")
    if prediction_identity["prediction_sources"]["evaluated_holdout"]["sha256"] != CONFIG["source_artifacts"]["holdout_predictions_sha256"]:
        raise RuntimeError("The frozen A_frozen holdout prediction artifact changed")

    calendar, _ = research._build_market_calendar(predictions)
    calendar["market_start_utc"] = pd.to_datetime(calendar.market_start_utc, utc=True, errors="raise")
    data = probability._build_dataset(calendar, t60)
    data["label_available_at_utc"] = pd.to_datetime(data.label_available_at_utc, utc=True, errors="coerce")
    data["market_start_utc"] = pd.to_datetime(data.market_start_utc, utc=True, errors="raise")
    return data, t59, t60, prediction_identity, book_record, source_holdout


def _btc_features(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    relative_path = CONFIG["source_artifacts"]["btc_feature_dataset_path"]
    path = ROOT / relative_path
    if _sha256(path) != CONFIG["source_artifacts"]["btc_feature_dataset_sha256"]:
        raise RuntimeError("The cached BTC feature dataset changed after preregistration")
    starts = data.market_start_utc.loc[
        data.market_start_utc.ge(DEV_START) & data.market_start_utc.lt(HOLDOUT_CUTOFF)
    ]
    feature_open = starts - pd.Timedelta(minutes=2)
    min_open = feature_open.min() - pd.Timedelta(minutes=60)
    max_open = feature_open.max()
    parquet = ds.dataset(path, format="parquet")
    table = parquet.to_table(
        columns=["Opened", "Close", "feature_candle_close_at", "futures_index_basis_rel_1m"],
        filter=(ds.field("Opened") >= min_open.strftime("%Y-%m-%d %H:%M:%S"))
        & (ds.field("Opened") <= max_open.strftime("%Y-%m-%d %H:%M:%S")),
    )
    candles = table.to_pandas()
    candles["Opened"] = pd.to_datetime(candles.Opened, utc=True, errors="raise")
    candles["feature_candle_close_at"] = pd.to_datetime(candles.feature_candle_close_at, utc=True, errors="coerce")
    candles.sort_values("Opened", kind="stable", inplace=True)
    if candles.Opened.duplicated().any():
        raise RuntimeError("Cached BTC features contain duplicate one-minute candle opens")
    expected_close = candles.Opened + pd.Timedelta(minutes=1)
    if not candles.feature_candle_close_at.eq(expected_close).all():
        raise RuntimeError("Cached BTC feature candle close times do not align with the one-minute index")

    candles.set_index("Opened", inplace=True)
    minute_index = pd.date_range(min_open.floor("min"), max_open.ceil("min"), freq="min", tz="UTC")
    close = pd.to_numeric(candles.Close, errors="coerce").reindex(minute_index)
    positive = close.where(close.gt(0.0))
    one_minute = positive.pct_change(fill_method=None)
    features = pd.DataFrame(index=minute_index)
    features["btc_return_1m"] = one_minute
    for horizon in (5, 15):
        continuous = positive.notna().rolling(horizon + 1, min_periods=horizon + 1).sum().eq(horizon + 1)
        features[f"btc_return_{horizon}m"] = (positive / positive.shift(horizon) - 1.0).where(continuous)
    return_window = positive.notna().rolling(61, min_periods=61).sum().eq(61)
    volatility = one_minute.rolling(60, min_periods=60).std(ddof=0) * math.sqrt(5.0)
    features["btc_sigma_5m"] = volatility.where(return_window)
    features["futures_index_basis_rel_1m"] = pd.to_numeric(
        candles.futures_index_basis_rel_1m, errors="coerce",
    ).reindex(minute_index)
    features["btc_feature_close_at"] = pd.Series(minute_index + pd.Timedelta(minutes=1), index=minute_index)
    features.index.name = "btc_feature_opened"

    result = data.copy()
    result["btc_feature_opened"] = result.market_start_utc - pd.Timedelta(minutes=2)
    result = result.merge(features.reset_index(), on="btc_feature_opened", how="left", validate="many_to_one")
    decision_at = result.market_start_utc - pd.Timedelta(minutes=1)
    result["btc_feature_time_aligned"] = result.btc_feature_close_at.eq(decision_at)
    result["logit_p_market"] = _logit(pd.to_numeric(result.p_market, errors="coerce"))
    result["logit_p_btc"] = _logit(pd.to_numeric(result.p_btc_calibrated, errors="coerce"))
    result["d_btc_minus_market"] = result.logit_p_btc - result.logit_p_market
    result["spread_up"] = pd.to_numeric(result.t60_up_spread, errors="coerce")
    result["spread_down"] = pd.to_numeric(result.t60_down_spread, errors="coerce")
    coverage = {
        "cached_rows_read": int(len(candles)),
        "cache_read_start_utc": candles.index.min().isoformat() if len(candles) else None,
        "cache_read_end_open_utc": candles.index.max().isoformat() if len(candles) else None,
        "feature_rows_with_aligned_close": int(result.btc_feature_time_aligned.sum()),
        "feature_source_sha256": _sha256(path),
        "basis_feature": "futures_index_basis_rel_1m",
    }
    return result, coverage


def _sample_tables(data: pd.DataFrame, enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    starts = enriched.market_start_utc
    release = pd.to_datetime(enriched.label_available_at_utc, utc=True, errors="coerce")
    official = pd.to_numeric(enriched.target_polymarket_up, errors="coerce").isin((0.0, 1.0))
    within_cutoff = (starts.lt(DEV_CUTOFF) & release.le(DEV_CUTOFF)) | (
        starts.ge(HOLDOUT_START) & starts.lt(HOLDOUT_CUTOFF) & release.le(HOLDOUT_CUTOFF)
    )
    periods = starts.ge(DEV_START) & starts.lt(HOLDOUT_CUTOFF)
    base_filter = enriched.market_confirmed.fillna(False) & enriched.causal_prediction_available.fillna(False) & periods
    valid_quotes = enriched.t60_common_quote_valid.fillna(False) & enriched.p_market.between(0.0, 1.0)
    valid_btc = enriched.p_btc_calibrated.between(0.0, 1.0)
    common = base_filter & official & valid_quotes & valid_btc
    feature_valid = enriched.btc_feature_time_aligned.fillna(False)
    for column in FEATURES:
        feature_valid &= np.isfinite(pd.to_numeric(enriched[column], errors="coerce"))
    selected = enriched.loc[common & within_cutoff & feature_valid].copy()
    selected["official_outcome_up"] = pd.to_numeric(selected.target_polymarket_up, errors="raise").astype(np.int8)
    selected["market_start_utc"] = pd.to_datetime(selected.market_start_utc, utc=True, errors="raise")
    selected["month_utc"] = selected.market_start_utc.dt.strftime("%Y-%m")
    dev = selected.loc[selected.market_start_utc.lt(DEV_CUTOFF)].copy()
    holdout = selected.loc[selected.market_start_utc.ge(HOLDOUT_START) & selected.market_start_utc.lt(HOLDOUT_CUTOFF)].copy()
    if len(dev) != 10794 or len(holdout) != 24735:
        raise RuntimeError(f"Frozen common sample changed: development={len(dev)}, holdout={len(holdout)}")
    if not np.isfinite(dev.loc[:, FEATURES].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("A declared development input is missing or non-finite")
    if not np.isfinite(holdout.loc[:, FEATURES].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("A declared holdout input is missing or non-finite")
    if not enriched.loc[selected.index, "btc_feature_time_aligned"].all():
        raise RuntimeError("A BTC feature row is not available at T-60")

    funnel = {
        "confirmed_with_causal_prediction": int(base_filter.sum()),
        "confirmed_with_causal_prediction_and_official_label": int((base_filter & official).sum()),
        "valid_two_sided_t60_market_reference": int((base_filter & official & valid_quotes).sum()),
        "with_causal_btc_probability": int((base_filter & official & valid_btc).sum()),
        "common_quote_and_market_btc_probability": int(common.sum()),
        "common_rows_released_by_period_cutoff": int((common & within_cutoff).sum()),
        "common_rows_missing_or_misaligned_btc_features": int((common & ~feature_valid).sum()),
        "eligible_final_rows": int(len(selected)),
        "late_or_missing_label_release_rows": int((common & ~within_cutoff).sum()),
        "sample_independent_of_ev_and_t59_execution": True,
    }
    period_rows = []
    for label, frame in (("development", dev), ("holdout", holdout)):
        period_rows.append({
            "period": label, "markets": int(len(frame)),
            "days": int(frame.market_start_utc.dt.floor("D").nunique()),
            "first_market_utc": frame.market_start_utc.min().isoformat(),
            "last_market_utc": frame.market_start_utc.max().isoformat(),
            "latest_label_available_at_utc": frame.label_available_at_utc.max().isoformat(),
            "fold_counts": frame.fold_id.astype(str).value_counts().sort_index().to_dict(),
        })
    for fold in CONFIG["chronological_cv"]["folds"]:
        start = pd.Timestamp(fold["start_inclusive_utc"])
        end = pd.Timestamp(fold["end_exclusive_utc"])
        train = dev.loc[dev.market_start_utc.lt(start) & dev.label_available_at_utc.le(start)]
        valid = dev.loc[dev.market_start_utc.ge(start) & dev.market_start_utc.lt(end)]
        period_rows.append({
            "period": fold["fold_id"], "training_rows": int(len(train)), "validation_rows": int(len(valid)),
            "latest_training_label_available_at_utc": train.label_available_at_utc.max().isoformat(),
            "validation_start_utc": start.isoformat(), "validation_end_exclusive_utc": end.isoformat(),
        })
    return dev, holdout, {"funnel": funnel, "periods": period_rows, "feature_coverage": {name: float(enriched.loc[selected.index, name].notna().mean()) for name in FEATURES}}


def _quantile_edges(values, bins: int = 4) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    edges = np.unique(np.quantile(array, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        value = float(edges[0]) if len(edges) else 0.0
        edges = np.array([value - 1e-12, value + 1e-12])
    return [float(value) for value in edges]


def _apply_context_bins(frame: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    result = frame.copy()
    definitions = {"d_btc_minus_market": "d_bin", "p_market": "market_price_bin", "btc_sigma_5m": "volatility_bin"}
    edges_by_name = {}
    for source, target in definitions.items():
        edges = _quantile_edges(train[source])
        labels = [f"Q{i + 1}" for i in range(len(edges) - 1)]
        result[target] = pd.cut(result[source], bins=edges, labels=labels, include_lowest=True, duplicates="drop").astype("string")
        edges_by_name[target] = edges
    return result, edges_by_name


class _OffsetEarlyStop:
    order = 30
    before_iteration = False

    def __init__(self, valid_x, valid_offset, valid_y, patience: int, threads: int):
        self.valid_x = valid_x
        self.valid_offset = np.asarray(valid_offset, dtype=np.float64)
        self.valid_y = np.asarray(valid_y, dtype=np.int8)
        self.patience = patience
        self.threads = threads
        self.best_loss = math.inf
        self.best_iteration = 0
        self.bad_rounds = 0

    def __call__(self, env):
        _check_budget()
        raw_correction = env.model.predict(
            self.valid_x, raw_score=True, num_iteration=env.iteration + 1, num_threads=self.threads,
        )
        probability = expit(self.valid_offset + np.asarray(raw_correction, dtype=np.float64))
        probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
        loss = float(-np.mean(self.valid_y * np.log(probability) + (1 - self.valid_y) * np.log1p(-probability)))
        if loss < self.best_loss - 1e-12:
            self.best_loss = loss
            self.best_iteration = int(env.iteration + 1)
            self.bad_rounds = 0
        else:
            self.bad_rounds += 1
        if self.bad_rounds >= self.patience:
            raise lgb.callback.EarlyStopException(
                self.best_iteration - 1,
                [("validation", "offset_logloss", self.best_loss, False)],
            )


def _tree_params(num_leaves: int, lambda_l2: float, train_rows: int) -> dict:
    return {
        "objective": "binary", "metric": "None", "boost_from_average": False,
        "learning_rate": 0.03, "num_leaves": int(num_leaves),
        "max_depth": {2: 1, 4: 2, 8: 3}[int(num_leaves)],
        "min_data_in_leaf": max(100, math.ceil(0.02 * int(train_rows))),
        "lambda_l2": float(lambda_l2), "verbosity": -1, "seed": 20261018,
        "feature_fraction_seed": 20261018, "bagging_seed": 20261018,
        "data_random_seed": 20261018, "deterministic": True, "force_col_wise": True,
        "num_threads": 4, "device_type": "cpu", "feature_pre_filter": False,
    }


def _fit_tree(train: pd.DataFrame, valid: pd.DataFrame, num_leaves: int, lambda_l2: float) -> tuple[lgb.Booster, dict]:
    x_train = train.loc[:, FEATURES].to_numpy(dtype=np.float64)
    x_valid = valid.loc[:, FEATURES].to_numpy(dtype=np.float64)
    offset_train = train.logit_p_market.to_numpy(dtype=np.float64)
    offset_valid = valid.logit_p_market.to_numpy(dtype=np.float64)
    y_train = train.official_outcome_up.to_numpy(dtype=np.int8)
    y_valid = valid.official_outcome_up.to_numpy(dtype=np.int8)
    train_set = lgb.Dataset(
        x_train, label=y_train, init_score=offset_train, feature_name=list(FEATURES), free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        x_valid, label=y_valid, init_score=offset_valid, reference=train_set,
        feature_name=list(FEATURES), free_raw_data=False,
    )
    if not np.array_equal(train_set.get_init_score(), offset_train) or not np.array_equal(valid_set.get_init_score(), offset_valid):
        raise AssertionError("LightGBM Dataset init_score is not row-aligned to the market offset")
    stop = _OffsetEarlyStop(
        x_valid, offset_valid, y_valid,
        patience=int(CONFIG["conditional_correction"]["early_stopping_rounds"]), threads=4,
    )
    fit_started = time.perf_counter()
    booster = lgb.train(
        _tree_params(num_leaves, lambda_l2, len(train)), train_set,
        num_boost_round=int(CONFIG["conditional_correction"]["max_estimators"]),
        valid_sets=[valid_set], valid_names=["validation"], callbacks=[stop],
    )
    best_iteration = stop.best_iteration or int(booster.current_iteration())
    raw = np.asarray(booster.predict(x_valid, raw_score=True, num_iteration=best_iteration, num_threads=4), dtype=np.float64)
    probability = expit(offset_valid + raw)
    observed = np.asarray(booster.predict(x_valid, raw_score=False, num_iteration=best_iteration, num_threads=4), dtype=np.float64)
    if not np.allclose(observed, expit(raw), rtol=1e-10, atol=1e-12):
        raise AssertionError("LightGBM prediction unexpectedly included the Dataset init_score")
    external_loss = float(-np.mean(y_valid * np.log(np.clip(probability, 1e-12, 1.0)) + (1-y_valid) * np.log(np.clip(1.0-probability, 1e-12, 1.0))))
    if not math.isclose(external_loss, stop.best_loss, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError("Offset early stopping and external prediction score disagree")
    return booster, {
        "best_iteration": int(best_iteration), "best_offset_logloss": external_loss,
        "elapsed_seconds": time.perf_counter() - fit_started,
        "train_init_score_rows": int(len(train_set.get_init_score())),
        "validation_init_score_rows": int(len(valid_set.get_init_score())),
        "external_offset_added_once": True,
        "booster_raw_score_excludes_dataset_init_score": True,
    }


def _fold_slices(dev: pd.DataFrame, fold_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(fold_config["start_inclusive_utc"])
    end = pd.Timestamp(fold_config["end_exclusive_utc"])
    train = dev.loc[dev.market_start_utc.lt(start) & dev.label_available_at_utc.le(start)].copy()
    valid = dev.loc[dev.market_start_utc.ge(start) & dev.market_start_utc.lt(end)].copy()
    if train.empty or valid.empty or train.official_outcome_up.nunique() != 2:
        raise RuntimeError(f"Chronological fold is unusable: {fold_config['fold_id']}")
    if not (train.label_available_at_utc <= start).all():
        raise AssertionError("A fold contains a label released after the validation start")
    return train, valid


def _diagnostics(frame: pd.DataFrame, name: str) -> dict:
    y = frame.official_outcome_up.to_numpy(dtype=np.float64)
    delta = frame.p_D_conditional.to_numpy(dtype=np.float64) - frame.p_A_market.to_numpy(dtype=np.float64)
    error = y - frame.p_A_market.to_numpy(dtype=np.float64)
    decomposition = _brier_decomposition(y, frame.p_A_market.to_numpy(dtype=np.float64), frame.p_D_conditional.to_numpy(dtype=np.float64))
    margin = frame.d_margin.to_numpy(dtype=np.float64)
    d = frame.d_btc_minus_market.to_numpy(dtype=np.float64)
    design = np.column_stack([np.ones(len(d)), d])
    coefficients = np.linalg.lstsq(design, margin, rcond=None)[0]
    fitted = design @ coefficients
    total = float(np.sum(np.square(margin - margin.mean())))
    r2 = 1.0 - float(np.sum(np.square(margin - fitted))) / total if total > 0.0 else 0.0
    nonzero = np.abs(delta) > 1e-15
    return {
        "period": name, "rows": int(len(frame)), "mean_delta_probability": float(delta.mean()),
        "mean_absolute_delta_probability": float(np.abs(delta).mean()),
        "p95_absolute_delta_probability": float(np.quantile(np.abs(delta), 0.95)),
        "mean_delta_margin": float(margin.mean()), "sd_delta_margin": float(margin.std(ddof=0)),
        "correction_alignment_2mean_delta_times_market_error": decomposition["error_alignment"],
        "correction_movement_cost_mean_delta_squared": decomposition["movement_cost"],
        "delta_brier_D_minus_A": decomposition["delta_brier"],
        "aligned_delta_sign_fraction": float(np.mean(np.sign(delta[nonzero]) == np.sign(error[nonzero]))) if nonzero.any() else None,
        "correction_margin_vs_d_linear_slope": float(coefficients[1]),
        "correction_margin_vs_d_linear_intercept": float(coefficients[0]),
        "correction_margin_vs_d_linear_r2": r2,
        "correction_margin_vs_d_residual_sd": float(np.sqrt(np.mean(np.square(margin - fitted)))),
    }


def _metric_tables(dev_cv: pd.DataFrame, holdout: pd.DataFrame) -> tuple[list[dict], list[dict], dict]:
    metric_rows, calibration_rows = [], []
    edges_by_model = {}
    monthly_quality = []
    for model, column in PREDICTION_COLUMNS.items():
        edges = probability._metric_edges(dev_cv[column].to_numpy(dtype=np.float64))
        edges_by_model[model] = edges
        row, bins = probability._metric_row(dev_cv, column, model, "development_cv_optimized", edges)
        metric_rows.append(row)
        calibration_rows.extend(bins)
        if model == "A":
            zero_row, zero_bins = probability._metric_row(dev_cv, column, "D_zero_correction", "development_cv_optimized", edges)
            metric_rows.append(zero_row)
            calibration_rows.extend(zero_bins)
            zero_row, zero_bins = probability._metric_row(holdout, column, "D_zero_correction", "holdout_historical_diagnostic", edges)
            metric_rows.append(zero_row)
            calibration_rows.extend(zero_bins)
        row, bins = probability._metric_row(holdout, column, model, "holdout_historical_diagnostic", edges)
        metric_rows.append(row)
        calibration_rows.extend(bins)
        for month, group in holdout.groupby("month_utc", sort=True):
            month_row, _ = probability._metric_row(group, column, model, f"holdout_{month}", edges)
            monthly_quality.append(month_row)
    paired = {
        f"D-{model}": probability._block_paired_differences(holdout, PREDICTION_COLUMNS["D"], PREDICTION_COLUMNS[model])
        for model in ("A", "B", "C")
    }
    paired_dev = {
        f"D-{model}": probability._block_paired_differences(dev_cv, PREDICTION_COLUMNS["D"], PREDICTION_COLUMNS[model])
        for model in ("A", "B", "C")
    }
    return metric_rows, calibration_rows, {
        "edges_by_model": edges_by_model, "paired_dev_optimized": paired_dev,
        "paired_holdout": paired, "monthly_quality": monthly_quality,
    }


def _economic_evaluation(frame: pd.DataFrame, t59: pd.DataFrame, t60: pd.DataFrame) -> dict:
    cutoff = HOLDOUT_CUTOFF
    t59_by_id, t60_by_id = research._snapshot_map(t59), research._snapshot_map(t60)
    enriched, _ = replay._market_statuses(frame, t59, t60)
    market_by_id = research._snapshot_map(enriched.loc[enriched.condition_id.notna()])
    model_results = {}
    for model, column in PREDICTION_COLUMNS.items():
        print(f"[conditional-offset] holdout economics {model}", flush=True)
        independent = []
        signal_count = 0
        rejects = {}
        for row in frame.itertuples(index=False):
            cid = str(row.condition_id).lower()
            market = {**market_by_id[cid], "p_candidate_platt": float(getattr(row, column))}
            t60_snapshot, t59_snapshot = t60_by_id.get(cid), t59_by_id.get(cid)
            if t60_snapshot is None or t59_snapshot is None:
                rejects["missing_t60_or_t59_snapshot"] = rejects.get("missing_t60_or_t59_snapshot", 0) + 1
                continue
            decision = replay._evaluate_market_sides(market, t60_snapshot, 5.0)
            side = decision["chosen_side"]
            if side is None:
                rejects["no_positive_net_ev_at_t60"] = rejects.get("no_positive_net_ev_at_t60", 0) + 1
                continue
            ev_fraction = float(decision["chosen_ev_usd"]) / 5.0
            if ev_fraction + 1e-12 < 0.01:
                rejects["below_1pct_net_ev"] = rejects.get("below_1pct_net_ev", 0) + 1
                continue
            signal_count += 1
            execution = replay._price_side(t59_snapshot, market, side, 5.0)
            fill = execution.get("fill") if execution["priceable"] else None
            if fill is None:
                reason = str(execution.get("reason", "t59_unpriceable"))
                rejects[reason] = rejects.get(reason, 0) + 1
                continue
            resolved = pd.Timestamp(row.resolved_at_utc) if pd.notna(row.resolved_at_utc) else pd.NaT
            release = max(resolved, pd.Timestamp(row.market_start_utc) + pd.Timedelta(minutes=5)) + pd.Timedelta(seconds=60) if pd.notna(resolved) else pd.NaT
            outcome = int(row.official_outcome_up)
            settled = pd.notna(release) and release <= cutoff
            won = outcome == int(side == "up")
            independent.append({
                "condition_id": cid, "model": model, "market_start_utc": row.market_start_utc,
                "side": side, "t60_ev_fraction": ev_fraction, "gross_usd": float(fill["gross_usd"]),
                "cash_debit_usd": float(fill["cash_debit_usd"]), "net_shares": float(fill["shares"]),
                "fee_usd": float(fill.get("fee_usd") or 0.0), "release_at_utc": release,
                "settled_by_cutoff": bool(settled), "net_pnl_usd": float(fill["shares"] if won else 0.0) - float(fill["cash_debit_usd"]) if settled else None,
                "depth_known": bool(fill.get("depth_known")), "fill_status": fill.get("fill_status"),
            })
        independent_frame = pd.DataFrame(independent)
        settled_independent = independent_frame.loc[independent_frame.settled_by_cutoff] if len(independent_frame) else independent_frame
        settled_turnover = float(settled_independent.gross_usd.sum()) if len(settled_independent) else 0.0

        # The existing simulator uses official outcomes only when their release event enters its settlement queue.
        # Its final all-settled summary is ignored; all cutoff results below come from trades released by cutoff.
        rows = frame.copy()
        rows["p_candidate_platt"] = rows[column].astype(float)
        rows["resolved_at_utc"] = pd.to_datetime(rows.resolved_at_utc, utc=True, errors="coerce")
        rows.loc[rows.resolved_at_utc.isna(), "resolved_at_utc"] = cutoff
        variant_market = {
            str(row.condition_id).lower(): {**market_by_id[str(row.condition_id).lower()], "p_candidate_platt": float(getattr(row, "p_candidate_platt"))}
            for row in rows.itertuples(index=False)
        }
        _, portfolio_trades, portfolio_decisions = replay._portfolio_for_policy(
            "free_cash_5pct", rows, t59_by_id, variant_market,
            decision_snapshot_by_id=t60_by_id, minimum_net_ev_fraction=0.01,
            output_policy=f"holdout_{model}_ev_1pct_cutoff_audit",
        )
        trade_frame = pd.DataFrame(portfolio_trades)
        if len(trade_frame):
            trade_frame["cash_release_at_utc"] = pd.to_datetime(trade_frame.cash_release_at_utc, utc=True, errors="raise")
            settled = trade_frame.loc[trade_frame.cash_release_at_utc.le(cutoff)].copy()
            unsettled = trade_frame.loc[trade_frame.cash_release_at_utc.gt(cutoff)].copy()
            debit_all = float(trade_frame.cash_debit_usd.sum())
            settled_payout = float(settled.payout_usd.sum())
            cash_at_cutoff = 100.0 - debit_all + settled_payout
            locked_at_cutoff = float(unsettled.gross_usd.sum())
            settled_pnl = float((settled.payout_usd - settled.cash_debit_usd).sum())
            portfolio_turnover = float(settled.gross_usd.sum())
            portfolio_fee = float(settled.fee_usd.sum())
        else:
            settled = unsettled = trade_frame
            cash_at_cutoff, locked_at_cutoff, settled_pnl, portfolio_turnover, portfolio_fee = 100.0, 0.0, 0.0, 0.0, 0.0
        rejection_counts = {}
        portfolio_signal_count = 0
        for item in portfolio_decisions:
            ev_fraction = item.get("net_ev_fraction_at_decision")
            if ev_fraction is not None and pd.notna(ev_fraction) and float(ev_fraction) + 1e-12 >= 0.01:
                portfolio_signal_count += 1
            reason = item.get("skip_reason")
            if reason:
                rejection_counts[str(reason)] = rejection_counts.get(str(reason), 0) + 1
        model_results[model] = {
            "independent_fixed_5_usd": {
                "signals_at_ev_1pct": signal_count,
                "fills": int(len(independent_frame)), "resolved_fills_by_cutoff": int(len(settled_independent)),
                "unresolved_fills_at_cutoff": int(len(independent_frame) - len(settled_independent)),
                "settled_net_pnl_usd": float(settled_independent.net_pnl_usd.sum()) if len(settled_independent) else 0.0,
                "settled_gross_turnover_usd": settled_turnover,
                "settled_net_over_settled_turnover": float(settled_independent.net_pnl_usd.sum() / settled_turnover) if settled_turnover else None,
                "fees_all_fills_usd": float(independent_frame.fee_usd.sum()) if len(independent_frame) else 0.0,
                "rejections": rejects,
            },
            "portfolio_100_usd_free_cash_5pct": {
                "decision_rows": int(len(portfolio_decisions)), "signals_at_ev_1pct": portfolio_signal_count,
                "entries_filled": int(len(trade_frame)), "resolved_fills_by_cutoff": int(len(settled)),
                "unresolved_fills_at_cutoff": int(len(unsettled)), "settled_net_pnl_usd": settled_pnl,
                "settled_gross_turnover_usd": portfolio_turnover,
                "settled_net_over_settled_turnover": settled_pnl / portfolio_turnover if portfolio_turnover else None,
                "fees_on_resolved_fills_usd": portfolio_fee,
                "cash_at_cutoff_usd": cash_at_cutoff, "locked_gross_cost_at_cutoff_usd": locked_at_cutoff,
                "rejections_by_reason": rejection_counts,
            },
        }
        if len(independent_frame):
            (OUTPUT_DIR / "economics" / model).mkdir(parents=True, exist_ok=True)
            independent_frame.to_parquet(OUTPUT_DIR / "economics" / model / "independent_fixed5.parquet", index=False, compression="zstd")
        if portfolio_trades:
            (OUTPUT_DIR / "economics" / model).mkdir(parents=True, exist_ok=True)
            trade_frame.to_parquet(OUTPUT_DIR / "economics" / model / "portfolio_trades.parquet", index=False, compression="zstd")
        if portfolio_decisions:
            (OUTPUT_DIR / "economics" / model).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(portfolio_decisions).to_parquet(OUTPUT_DIR / "economics" / model / "portfolio_decisions.parquet", index=False, compression="zstd")
    return {"cutoff_exclusive_utc": cutoff.isoformat(), "minimum_net_ev_fraction": 0.01, "models": model_results}


def _render_report(payload: dict) -> str:
    funnel = payload["sample"]["funnel"]
    btc_source = payload["sample"]["btc_source"]
    oof_sources = ", ".join(
        f"{fold}: {record['sha256']}"
        for fold, record in btc_source["prediction_sources"].items()
        if fold.startswith("dev_2026q2_")
    )
    holdout_source = btc_source["prediction_sources"]["evaluated_holdout"]
    rows = [
        "# BTC conditional market-logit correction - 2026-10-18", "",
        "> The July-October 2026 holdout was viewed earlier in this repository. Holdout results are descriptive diagnostics, not independent confirmation.", "",
        "## Question and frozen protocol", "",
        "Test whether a small regularized tree correction `p_D = sigmoid(logit(p_market) + f(X))` adds probability information beyond the raw market and global logistic benchmarks. The coefficient on the base market logit is fixed at 1. A zero correction reproduces `p_market` exactly within numerical precision.",
        "The raw `p_market` definition is unchanged: `mid_up=(up_best_bid+up_best_ask)/2`, `mid_down=(down_best_bid+down_best_ask)/2`, `p_market=mid_up/(mid_up+mid_down)`. Source fields come from valid causal T-60 snapshots. T-59 asks are used only for execution simulation.",
        "Labels are official Polymarket outcomes only. Label availability is `max(resolved_at, market_start+5 minutes)+60 seconds`; Binance proxy labels do not fill gaps. Quality sample inclusion does not depend on EV or execution.", "",
        f"BTC forecast source is A_frozen: chronological development OOF artifacts ({oof_sources}) and the frozen holdout prediction artifact ({holdout_source['sha256']}); walk-forward manifest SHA-256 `{btc_source['walk_forward_manifest_sha256']}`. No June-final BTC model predictions were substituted for April or May.", "",
        "## Pre-training sample and feature coverage", "",
        f"Funnel: {funnel['confirmed_with_causal_prediction']:,} confirmed markets with causal prediction; {funnel['confirmed_with_causal_prediction_and_official_label']:,} also have an official binary label; {funnel['valid_two_sided_t60_market_reference']:,} have a valid two-sided T-60 quote; {funnel['common_quote_and_market_btc_probability']:,} have both market and BTC probabilities; {funnel['common_rows_released_by_period_cutoff']:,} labels meet the period cutoff; {funnel['common_rows_missing_or_misaligned_btc_features']:,} lack an aligned BTC feature; final sample {funnel['eligible_final_rows']:,}.",
        "Inputs (fixed logit epsilon 1e-6): logit(p_market), d=logit(p_BTC)-logit(p_market), BTC close-to-close returns over 1/5/15 minutes, sigma from the last 60 contiguous one-minute returns (`sqrt(5) * population std`, ddof=0), existing `futures_index_basis_rel_1m`, and UP/DOWN bid-ask spreads. Every input is available at T-60.", "",
        "| Period / fold | Markets | Days | Range or rule |", "|---|---:|---:|---|",
    ]
    for item in payload["sample"]["periods"]:
        if "markets" in item:
            rows.append(f"| {item['period']} | {item['markets']:,} | {item['days']} | {item['first_market_utc']} to {item['last_market_utc']} |")
        else:
            rows.append(f"| {item['period']} | train {item['training_rows']:,}; valid {item['validation_rows']:,} | - | latest train label release {item['latest_training_label_available_at_utc']} |")
    rows += ["", "All nine features have 100% coverage on the common quality sample. April is the May fold training prefix. Each fold purges labels released after validation begins.", "",
             "## Models and selection", "",
             "A: raw p_market. B: L2 logistic regression, C=10, on logit(p_market). C: same regression on logit(p_market) and logit(p_BTC). D: LightGBM binary with row-aligned `init_score=logit(p_market)`, at most 300 trees, learning rate 0.03, and early stopping 30. Six fixed configurations: leaves 2/4/8 by lambda_l2 10/100; max_depth 1/2/3; min_data_in_leaf=max(100, ceil(2% of training rows)); seed 20261018.",
             "Validation log-loss is computed on `sigmoid(offset + raw tree correction)`. LightGBM prediction returns the tree correction; the offset is added exactly once. Selection minimizes `mean fold log-loss + 0.5 * population SD`. The development result is optimized over these six configurations.", "",
             "| D configuration | Leaves | L2 | Fold iterations | Fold log-loss | Mean + 0.5 SD |", "|---|---:|---:|---:|---|---:|"]
    for item in payload["selection"]:
        fold_loss_text = ", ".join(f"{value:.6f}" for value in item["fold_log_losses"])
        rows.append(f"| {item['config_id']} | {item['num_leaves']} | {item['lambda_l2']:.0f} | {item['best_iterations']} | {fold_loss_text} | {item['selection_score']:.6f} |")
    chosen = payload["chosen_config"]
    rows += ["", f"Selected `{chosen['config_id']}`; final D uses {payload['final_tree_count']} trees (floor of the median fold best iteration). Models and benchmarks were frozen at {payload['freeze_time_utc']} before holdout scoring.", "",
             "## Probability quality", "",
             "| Period | Model | n | Log-loss | Brier | AUC | Calibration intercept / slope | ECE |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for item in payload["metrics"]:
        rows.append(f"| {item['period']} | {item['forecast']} | {item['n']:,} | {item['log_loss']:.6f} | {item['brier']:.6f} | {item['auc']:.4f} | {item['calibration_intercept']:.3f} / {item['calibration_slope']:.3f} | {item['ece_frozen_dev_bins']:.5f} |")
    rows += ["", "Paired differences are D minus the named model; negative log-loss/Brier favors D. Intervals use paired moving 3-day UTC blocks, 1,000 replicates, seed 20261012.", "",
             "| Evaluation | Compare | Delta log-loss [95% CI] | Delta Brier [95% CI] | Delta AUC [95% CI] |", "|---|---|---:|---:|---:|"]
    for evaluation, key in (("Development OOF (optimized)", "paired_dev_optimized"), ("Historical holdout", "paired_holdout")):
        for name, item in payload[key].items():
            rows.append(f"| {evaluation} | {name} | {item['log_loss_difference_left_minus_right']:.6f} [{item['log_loss_difference_95pct'][0]:.6f}, {item['log_loss_difference_95pct'][1]:.6f}] | {item['brier_difference_left_minus_right']:.6f} [{item['brier_difference_95pct'][0]:.6f}, {item['brier_difference_95pct'][1]:.6f}] | {item['auc_difference_left_minus_right']:.4f} [{item['auc_difference_95pct'][0]:.4f}, {item['auc_difference_95pct'][1]:.4f}] |")
    rows += ["", "### Holdout probability quality by month", "", "| Month | Model | n | Log-loss | Brier | AUC |", "|---|---|---:|---:|---:|---:|"]
    for item in payload["monthly_quality"]:
        rows.append(f"| {item['period'].replace('holdout_', '')} | {item['forecast']} | {item['n']:,} | {item['log_loss']:.6f} | {item['brier']:.6f} | {item['auc']:.4f} |")
    rows += ["", "## Does the correction track market error?", "", "Exact identity: `Delta Brier = mean(delta^2) - 2*mean(delta*e)`, with `delta=p_D-p_market` and `e=y-p_market`. The first term is the correction-size cost; the second measures alignment with market error.", "",
             "| Period | Mean abs delta | P95 abs delta | Mean delta^2 cost | 2*mean(delta*e) | Delta Brier | Mean margin | SD margin | Margin vs d: slope / R2 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in payload["correction_diagnostics"]:
        rows.append(f"| {item['period']} | {item['mean_absolute_delta_probability']:.6f} | {item['p95_absolute_delta_probability']:.6f} | {item['correction_movement_cost_mean_delta_squared']:.7f} | {item['correction_alignment_2mean_delta_times_market_error']:.7f} | {item['delta_brier_D_minus_A']:.7f} | {item['mean_delta_margin']:.5f} | {item['sd_delta_margin']:.5f} | {item['correction_margin_vs_d_linear_slope']:.4f} / {item['correction_margin_vs_d_linear_r2']:.3f} |")
    rows += ["", "### By period", "", "| Period | n | Mean abs delta | Delta Brier | Correction alignment | Margin vs d: slope / R2 |", "|---|---:|---:|---:|---:|---:|"]
    for item in payload["period_diagnostics"]:
        rows.append(f"| {item['period']} | {item['rows']:,} | {item['mean_absolute_delta_probability']:.6f} | {item['delta_brier_D_minus_A']:.7f} | {item['correction_alignment_2mean_delta_times_market_error']:.7f} | {item['correction_margin_vs_d_linear_slope']:.4f} / {item['correction_margin_vs_d_linear_r2']:.3f} |")
    rows += ["", "### Training-defined context bins", "", "Quantile boundaries for d, market price, and volatility are learned from each fold training set; holdout bins use all eligible development rows.", "", "| Evaluation / context bin | n | Mean abs delta | Delta Brier | Correction alignment |", "|---|---:|---:|---:|---:|"]
    for item in payload["context_diagnostics"]:
        rows.append(f"| {item['period']} | {item['rows']:,} | {item['mean_absolute_delta_probability']:.6f} | {item['delta_brier_D_minus_A']:.7f} | {item['correction_alignment_2mean_delta_times_market_error']:.7f} |")
    rows += ["", "A near-zero mean margin with nontrivial spread indicates the correction is not a constant shift. The slope and R2 against d summarize whether it mostly scales the BTC-market probability gap; the context bins show its behavior across price and volatility states.", "",
             "## After-cost holdout economics", "", "Minimum net EV is 1%; decision T-60, execution T-59, existing fee and minimum-order rules. Independent $5 orders and the separate $100 portfolio use 5% of free cash with no borrowing. Entries/fills are separated from fills settled before cutoff. PnL and turnover in each return ratio use the same settled-fill set.", "",
             "| Model | $5 signals / fills / settled | Settled PnL | Settled turnover | $5 PnL/turnover | Portfolio signals / fills / settled | Portfolio PnL | Portfolio turnover | Cash / locked cost at cutoff |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model, item in payload["economics"]["models"].items():
        ind = item["independent_fixed_5_usd"]
        port = item["portfolio_100_usd_free_cash_5pct"]
        ind_ratio = f"{ind['settled_net_over_settled_turnover']:.2%}" if ind["settled_net_over_settled_turnover"] is not None else "n/a"
        rows.append(f"| {model} | {ind['signals_at_ev_1pct']:,} / {ind['fills']:,} / {ind['resolved_fills_by_cutoff']:,} | ${ind['settled_net_pnl_usd']:.2f} | ${ind['settled_gross_turnover_usd']:.2f} | {ind_ratio} | {port['signals_at_ev_1pct']:,} / {port['entries_filled']:,} / {port['resolved_fills_by_cutoff']:,} | ${port['settled_net_pnl_usd']:.2f} | ${port['settled_gross_turnover_usd']:.2f} | ${port['cash_at_cutoff_usd']:.2f} / ${port['locked_gross_cost_at_cutoff_usd']:.2f} |")
    rows += ["", "### Portfolio rejection reasons", "", "| Model | Rejections by reason |", "|---|---|"]
    for model, item in payload["economics"]["models"].items():
        reasons = json.dumps(item["portfolio_100_usd_free_cash_5pct"]["rejections_by_reason"], sort_keys=True)
        rows.append(f"| {model} | `{reasons}` |")
    rows += ["", "## Conclusion", "", payload["conclusion"], "", "Reproduce from the repository root: `python run_btc_conditional_market_offset_20261018.py`. Frozen protocol: `configs/research/btc_conditional_market_offset_20261018.json`. Predictions, checkpoints, frozen model, and detailed diagnostics are under `data/analysis/polymarket/BTC/btc_conditional_market_offset_20261018/`. No BTC training, new data fetch, or archive replay was run.", ""]
    cost = payload.get("cost")
    if cost:
        peak_working_set = (
            f"{cost['peak_working_set_gib']:.2f} GiB"
            if cost.get("peak_working_set_gib") is not None else "not recorded"
        )
        memory_note = (
            " The configured memory cap could not be verified for this run."
            if cost.get("peak_working_set_gib") is None else ""
        )
        rows[-1:-1] = [
            "", "## Runtime and local resource use", "",
            f"End-to-end elapsed time: {cost['elapsed_seconds']:.1f}s. Peak working set: {peak_working_set}. Generated files: {cost['generated_output_bytes']:,} bytes; configured caps were 1,800 seconds, 12 GiB, and 500,000,000 bytes.{memory_note}", "",
        ]
    return "\n".join(rows).rstrip() + "\n"


def main() -> dict:
    global EXPERIMENT_STARTED
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prior = json.loads(RUN_PATH.read_text(encoding="utf-8")) if RUN_PATH.exists() else {}
    resume_elapsed = float(prior.get("elapsed_seconds_before_holdout", 0.0))
    if prior.get("status") == "holdout_scored" and (OUTPUT_DIR / "holdout_economics.json").is_file():
        freeze_record = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        frozen_at = pd.Timestamp(freeze_record["frozen_at_utc"]).timestamp()
        economics_mtime = (OUTPUT_DIR / "holdout_economics.json").stat().st_mtime
        resume_elapsed += max(0.0, economics_mtime - frozen_at)
    EXPERIMENT_STARTED = started - resume_elapsed
    if prior.get("status") == "completed":
        print(f"Experiment already completed: {REPORT_PATH}", flush=True)
        return prior
    if prior and prior.get("status") not in ("development_frozen", "holdout_scored"):
        raise RuntimeError(f"An incomplete experiment checkpoint exists with status={prior.get('status')!r}; preserving it and refusing to repeat work or rescore holdout")
    _write_json(OUTPUT_DIR / "preregistered_config.json", CONFIG)
    data, t59, t60, prediction_identity, book_record, _ = _load_causal_inputs()
    enriched, feature_source = _btc_features(data)
    dev, holdout, sample = _sample_tables(data, enriched)
    sample["btc_source"] = prediction_identity
    sample["snapshot_hashes"] = {
        "t60": book_record["t60_coverage_sha256"], "t59": book_record["t59_coverage_sha256"],
    }
    _write_json(OUTPUT_DIR / "pretraining_sample_audit.json", {**sample, "btc_feature_source": feature_source})
    print(json.dumps(sample, indent=2, default=str), flush=True)
    _check_budget()

    fold_data = []
    for fold_config in CONFIG["chronological_cv"]["folds"]:
        train, valid = _fold_slices(dev, fold_config)
        fold_data.append((fold_config, train, valid))

    if prior.get("status") in ("development_frozen", "holdout_scored"):
        selection = json.loads((OUTPUT_DIR / "configuration_selection.json").read_text(encoding="utf-8"))
        config_results = selection["all_configurations"]
        chosen = selection["selected"]
        final_tree_count = int(selection["final_tree_count"])
        dev_cv_path = OUTPUT_DIR / "development_cv_predictions.parquet"
        freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        final_tree_path = ROOT / freeze["final_tree_path"]
        final_models_path = ROOT / freeze["final_logistic_models_path"]
        if _sha256(final_tree_path) != freeze["final_tree_sha256"]:
            raise RuntimeError("The frozen LightGBM correction changed before holdout scoring")
        if _sha256(final_models_path) != freeze["final_logistic_models_sha256"]:
            raise RuntimeError("The frozen logistic benchmarks changed before holdout scoring")
        if _sha256(dev_cv_path) != freeze["selected_development_cv_predictions_sha256"]:
            raise RuntimeError("The frozen development predictions changed before holdout scoring")
        dev_cv = pd.read_parquet(dev_cv_path)
        final_train = dev.loc[dev.label_available_at_utc.le(DEV_CUTOFF)].copy()
        final_tree = lgb.Booster(model_file=str(final_tree_path))
        final_models = json.loads(final_models_path.read_text(encoding="utf-8"))
        final_b, final_c = final_models["B"], final_models["C"]
        freeze_time = freeze["frozen_at_utc"]
        print("[conditional-offset] resuming from hash-verified frozen development models; no refit", flush=True)
    else:
        # Offset smoke is the first LightGBM fit and is reused as config_2_l2_10's first fold fit.
        first_config = {"config_id": "leaves2_l2_10", "num_leaves": 2, "lambda_l2": 10.0}
        smoke_train, smoke_valid = fold_data[0][1], fold_data[0][2]
        smoke_booster, smoke_check = _fit_tree(smoke_train, smoke_valid, 2, 10.0)
        smoke_valid = smoke_valid.copy()
        smoke_margin = np.asarray(smoke_booster.predict(
            smoke_valid.loc[:, FEATURES].to_numpy(dtype=np.float64), raw_score=True,
            num_iteration=smoke_check["best_iteration"], num_threads=4,
        ), dtype=np.float64)
        smoke_valid["p_D_conditional"] = expit(smoke_valid.logit_p_market.to_numpy(dtype=np.float64) + smoke_margin)
        smoke_valid["d_margin"] = smoke_margin
        smoke_valid, smoke_edges = _apply_context_bins(smoke_valid, smoke_train)
        smoke_check.update({"fold_id": "dev_2026q2_b", "num_leaves": 2, "lambda_l2": 10.0, "context_bin_edges": smoke_edges})
        smoke_path = OUTPUT_DIR / "checkpoints" / "leaves2_l2_10__dev_2026q2_b.parquet"
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_valid.to_parquet(smoke_path, index=False, compression="zstd")
        _write_json(smoke_path.with_suffix(".json"), smoke_check)
        _write_json(OUTPUT_DIR / "checkpoints" / "offset_smoke.json", {"status": "passed", **smoke_check})

        logistic_cv = {}
        for fold_config, train, valid in fold_data:
            b_model = _fit_logistic(train, ("logit_p_market",))
            c_model = _fit_logistic(train, ("logit_p_market", "logit_p_btc"))
            logistic_cv[fold_config["fold_id"]] = {
                "B": _predict_logistic(b_model, valid, ("logit_p_market",)),
                "C": _predict_logistic(c_model, valid, ("logit_p_market", "logit_p_btc")),
            }

        config_results = []
        cv_by_config = {}
        for leaves in (2, 4, 8):
            for l2 in (10.0, 100.0):
                _check_budget()
                config_id = f"leaves{leaves}_l2_{int(l2)}"
                fold_predictions, fold_scores, best_iterations, fold_fits = [], [], [], []
                for fold_index, (fold_config, train, valid) in enumerate(fold_data):
                    checkpoint = OUTPUT_DIR / "checkpoints" / f"{config_id}__{fold_config['fold_id']}.parquet"
                    fit_record_path = checkpoint.with_suffix(".json")
                    if fold_index == 0 and config_id == first_config["config_id"]:
                        current = pd.read_parquet(checkpoint)
                        fit_record = json.loads(fit_record_path.read_text(encoding="utf-8"))
                    elif checkpoint.exists() and fit_record_path.exists():
                        current = pd.read_parquet(checkpoint)
                        fit_record = json.loads(fit_record_path.read_text(encoding="utf-8"))
                    else:
                        booster, fit_record = _fit_tree(train, valid, leaves, l2)
                        correction = booster.predict(
                            valid.loc[:, FEATURES].to_numpy(dtype=np.float64), raw_score=True,
                            num_iteration=fit_record["best_iteration"], num_threads=4,
                        )
                        current = valid.copy()
                        current["p_D_conditional"] = expit(valid.logit_p_market.to_numpy(dtype=np.float64) + correction)
                        current["d_margin"] = np.asarray(correction, dtype=np.float64)
                        current, context_edges = _apply_context_bins(current, train)
                        fit_record.update({"fold_id": fold_config["fold_id"], "num_leaves": leaves, "lambda_l2": l2, "context_bin_edges": context_edges})
                        checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        current.to_parquet(checkpoint, index=False, compression="zstd")
                        _write_json(fit_record_path, fit_record)
                    if "d_margin" not in current:
                        correction = np.asarray(smoke_booster.predict(current.loc[:, FEATURES].to_numpy(dtype=np.float64), raw_score=True, num_iteration=fit_record["best_iteration"], num_threads=4), dtype=np.float64)
                        current["d_margin"] = correction
                        current["p_D_conditional"] = expit(current.logit_p_market.to_numpy(dtype=np.float64) + correction)
                        current, context_edges = _apply_context_bins(current, train)
                        fit_record["context_bin_edges"] = context_edges
                        current.to_parquet(checkpoint, index=False, compression="zstd")
                        _write_json(fit_record_path, fit_record)
                    current["p_A_market"] = current.p_market.astype(float)
                    current["p_B_market_only"] = logistic_cv[fold_config["fold_id"]]["B"]
                    current["p_C_market_plus_btc"] = logistic_cv[fold_config["fold_id"]]["C"]
                    fold_predictions.append(current)
                    best_iterations.append(int(fit_record["best_iteration"]))
                    fold_fits.append(fit_record)
                    fold_scores.append(float(-np.mean(
                        current.official_outcome_up.to_numpy(dtype=np.float64) * np.log(np.clip(current.p_D_conditional, 1e-12, 1.0))
                        + (1.0-current.official_outcome_up.to_numpy(dtype=np.float64)) * np.log(np.clip(1.0-current.p_D_conditional, 1e-12, 1.0))
                    )))
                combined = pd.concat(fold_predictions, ignore_index=True).sort_values("market_start_utc", kind="stable")
                mean_ll = float(np.mean(fold_scores))
                sd_ll = float(np.std(fold_scores, ddof=0))
                result = {
                    "config_id": config_id, "num_leaves": leaves, "lambda_l2": l2,
                    "best_iterations": best_iterations, "fold_log_losses": fold_scores,
                    "mean_fold_log_loss": mean_ll, "population_sd_fold_log_loss": sd_ll,
                    "selection_score": mean_ll + 0.5 * sd_ll,
                    "fit_seconds": float(sum(item.get("elapsed_seconds", 0.0) for item in fold_fits)),
                }
                config_results.append(result)
                cv_by_config[config_id] = combined
                combined.to_parquet(OUTPUT_DIR / "checkpoints" / f"{config_id}__development_cv.parquet", index=False, compression="zstd")
                _write_json(OUTPUT_DIR / "checkpoints" / f"{config_id}__completed.json", result)
                _write_json(OUTPUT_DIR / "progress.json", {"status": "fold_config_completed", "completed_configurations": len(config_results), "latest": result})
                print(f"[conditional-offset] {config_id}: score={result['selection_score']:.6f}, fold_ll={fold_scores}", flush=True)

        config_results.sort(key=lambda item: (item["selection_score"], item["num_leaves"], -item["lambda_l2"], int(np.median(item["best_iterations"]))))
        chosen = config_results[0]
        final_tree_count = max(1, int(np.floor(np.median(chosen["best_iterations"]))))
        dev_cv = cv_by_config[chosen["config_id"]].copy()
        for key in ("A", "B", "C", "D"):
            if key == "D":
                continue
            dev_cv[PREDICTION_COLUMNS[key]] = dev_cv[PREDICTION_COLUMNS[key]].astype(float)
        dev_cv.to_parquet(OUTPUT_DIR / "development_cv_predictions.parquet", index=False, compression="zstd")

        # Fit the final models on all and only cutoff-eligible development labels.
        final_train = dev.loc[dev.label_available_at_utc.le(DEV_CUTOFF)].copy()
        final_b = _fit_logistic(final_train, ("logit_p_market",))
        final_c = _fit_logistic(final_train, ("logit_p_market", "logit_p_btc"))
        final_leaves, final_l2 = chosen["num_leaves"], chosen["lambda_l2"]
        final_train_set = lgb.Dataset(
            final_train.loc[:, FEATURES].to_numpy(dtype=np.float64),
            label=final_train.official_outcome_up.to_numpy(dtype=np.int8),
            init_score=final_train.logit_p_market.to_numpy(dtype=np.float64),
            feature_name=list(FEATURES), free_raw_data=False,
        )
        _check_budget()
        final_tree = lgb.train(
            _tree_params(final_leaves, final_l2, len(final_train)), final_train_set,
            num_boost_round=final_tree_count,
        )
        final_tree_path = OUTPUT_DIR / "conditional_correction_lightgbm.txt"
        final_tree.save_model(str(final_tree_path), num_iteration=final_tree_count)
        saved_tree = lgb.Booster(model_file=str(final_tree_path))
        parity_x = final_train.loc[:, FEATURES].head(128).to_numpy(dtype=np.float64)
        if not np.allclose(
            final_tree.predict(parity_x, raw_score=True, num_threads=4),
            saved_tree.predict(parity_x, raw_score=True, num_threads=4), rtol=0.0, atol=1e-12,
        ):
            raise AssertionError("The saved conditional-offset model does not reproduce its in-memory correction")
        final_tree = saved_tree
        final_models_path = OUTPUT_DIR / "final_logistic_models.json"
        _write_json(final_models_path, {
            "B": {"features": ["logit_p_market"], "intercept": float(final_b.intercept_[0]), "coefficients": final_b.coef_[0].astype(float).tolist()},
            "C": {"features": ["logit_p_market", "logit_p_btc"], "intercept": float(final_c.intercept_[0]), "coefficients": final_c.coef_[0].astype(float).tolist()},
        })
        _, selection_edges = _apply_context_bins(dev_cv, dev)
        selection_path = OUTPUT_DIR / "configuration_selection.json"
        _write_json(selection_path, {"selected": chosen, "all_configurations": config_results, "criterion": CONFIG["chronological_cv"]["selection_score"], "final_tree_count": final_tree_count})
        freeze_time = pd.Timestamp.now(tz="UTC").isoformat()
        freeze = {
            "status": "development_models_frozen_before_holdout_scoring",
            "frozen_at_utc": freeze_time,
            "features": list(FEATURES), "epsilon": EPSILON, "selected_config": chosen,
            "final_tree_count": final_tree_count, "training_rows": int(len(final_train)),
            "latest_training_label_available_at_utc": final_train.label_available_at_utc.max().isoformat(),
            "final_tree_path": final_tree_path.resolve().relative_to(ROOT).as_posix(),
            "final_tree_sha256": _sha256(final_tree_path),
            "final_logistic_models_path": final_models_path.resolve().relative_to(ROOT).as_posix(),
            "final_logistic_models_sha256": _sha256(final_models_path),
            "selected_development_cv_predictions_sha256": _sha256(OUTPUT_DIR / "development_cv_predictions.parquet"),
            "selection_edges": selection_edges,
            "holdout_used_for_selection": False,
            "init_score_contract": "training and validation datasets get row-aligned logit_p_market init_score; prediction uses expit(logit_p_market + booster raw_score) once",
        }
        _write_json(FREEZE_PATH, freeze)
        _write_json(RUN_PATH, {"status": "development_frozen", "freeze_path": FREEZE_PATH.resolve().relative_to(ROOT).as_posix(), "started_at_utc": prior.get("started_at_utc", freeze_time), "elapsed_seconds_before_holdout": time.perf_counter() - started})

    best_cv = dev_cv.copy()
    holdout_universe = enriched.loc[
        enriched.market_start_utc.ge(HOLDOUT_START) & enriched.market_start_utc.lt(HOLDOUT_CUTOFF)
        & enriched.common_sample.fillna(False) & enriched.target_polymarket_up.isin((0.0, 1.0))
        & enriched.btc_feature_time_aligned.fillna(False)
    ].copy()
    holdout_universe["official_outcome_up"] = pd.to_numeric(holdout_universe.target_polymarket_up, errors="raise").astype(np.int8)
    holdout_universe["market_start_utc"] = pd.to_datetime(holdout_universe.market_start_utc, utc=True, errors="raise")
    holdout_universe["label_available_at_utc"] = pd.to_datetime(holdout_universe.label_available_at_utc, utc=True, errors="coerce")
    if not holdout.condition_id.isin(holdout_universe.condition_id).all() or len(holdout_universe) not in (len(holdout), len(holdout) + 1):
        raise RuntimeError("Holdout economic population differs from the preregistered quality sample unexpectedly")
    if prior.get("status") == "holdout_scored":
        prediction_path = OUTPUT_DIR / "holdout_predictions.parquet"
        expected_hash = prior.get("holdout_prediction_sha256")
        if not prediction_path.is_file() or _sha256(prediction_path) != expected_hash:
            raise RuntimeError("The once-scored holdout prediction checkpoint is missing or changed")
        holdout_predictions = pd.read_parquet(prediction_path)
        score_path = OUTPUT_DIR / "holdout_score.json"
        if not score_path.is_file():
            raise RuntimeError("The holdout prediction checkpoint has no matching score file")
        scored_payload = json.loads(score_path.read_text(encoding="utf-8"))
        print("[conditional-offset] resuming from the once-scored holdout checkpoint", flush=True)
    else:
        run_state = json.loads(RUN_PATH.read_text(encoding="utf-8")) if RUN_PATH.is_file() else {}
        _write_json(RUN_PATH, {**run_state, "status": "holdout_scoring_started", "freeze_path": FREEZE_PATH.resolve().relative_to(ROOT).as_posix(), "started_at_utc": run_state.get("started_at_utc", freeze_time), "holdout_scoring_started_at_utc": pd.Timestamp.now(tz="UTC").isoformat()})
        _check_budget()
        holdout_universe["p_A_market"] = holdout_universe.p_market.astype(float)
        holdout_universe["p_B_market_only"] = _predict_logistic(final_b, holdout_universe, ("logit_p_market",))
        holdout_universe["p_C_market_plus_btc"] = _predict_logistic(final_c, holdout_universe, ("logit_p_market", "logit_p_btc"))
        raw_d = np.asarray(final_tree.predict(holdout_universe.loc[:, FEATURES].to_numpy(dtype=np.float64), raw_score=True, num_threads=4), dtype=np.float64)
        holdout_universe["d_margin"] = raw_d
        holdout_universe["p_D_conditional"] = expit(holdout_universe.logit_p_market.to_numpy(dtype=np.float64) + raw_d)
        holdout_universe, holdout_edges = _apply_context_bins(holdout_universe, final_train)
        holdout_universe.to_parquet(OUTPUT_DIR / "holdout_predictions.parquet", index=False, compression="zstd")
        holdout_eval = holdout_universe.loc[holdout_universe.label_available_at_utc.le(HOLDOUT_CUTOFF)].copy()
        if len(holdout_eval) != len(holdout) or set(holdout_eval.condition_id) != set(holdout.condition_id):
            raise RuntimeError("Holdout cutoff evaluation rows do not match the pretraining sample audit")
        metrics, calibration, paired = _metric_tables(best_cv, holdout_eval)
        correction_diagnostics = [_diagnostics(best_cv, "development_cv_optimized"), _diagnostics(holdout_eval, "holdout_historical_diagnostic")]
        month_rows = []
        for month, group in pd.concat([best_cv.assign(eval_period="development_cv_optimized"), holdout_eval.assign(eval_period="holdout_historical_diagnostic")]).groupby(["eval_period", "month_utc"], sort=True):
            month_rows.append(_diagnostics(group, f"{month[0]}:{month[1]}"))
        # Training-defined context buckets for both outer development folds and holdout.
        context_rows = []
        for frame_name, frame in (("development_cv_optimized", best_cv), ("holdout_historical_diagnostic", holdout_eval)):
            for bucket in ("d_bin", "market_price_bin", "volatility_bin"):
                for label, group in frame.groupby(bucket, observed=True, sort=True):
                    context_rows.append(_diagnostics(group, f"{frame_name}:{bucket}:{label}"))
        pd.DataFrame(context_rows).to_csv(OUTPUT_DIR / "correction_context_diagnostics.csv", index=False)
        pd.DataFrame(month_rows).to_csv(OUTPUT_DIR / "correction_period_diagnostics.csv", index=False)
        scored_payload = {
            "metrics": metrics, "calibration_bins": calibration,
            "paired_dev_optimized": paired["paired_dev_optimized"],
            "paired_holdout": paired["paired_holdout"], "monthly_quality": paired["monthly_quality"],
            "correction_diagnostics": correction_diagnostics,
            "period_diagnostics": month_rows, "context_diagnostics_path": "correction_context_diagnostics.csv",
            "context_diagnostics": context_rows,
            "holdout_context_edges": holdout_edges,
            "holdout_prediction_universe_rows": int(len(holdout_universe)),
            "holdout_labels_available_by_cutoff_rows": int(len(holdout_eval)),
        }
        _write_json(OUTPUT_DIR / "holdout_score.json", scored_payload)
        run_state = json.loads(RUN_PATH.read_text(encoding="utf-8"))
        _write_json(RUN_PATH, {**run_state, "status": "holdout_scored", "holdout_prediction_sha256": _sha256(OUTPUT_DIR / "holdout_predictions.parquet")})
        holdout_predictions = holdout_universe


    # One descriptive economic pass on the already-scored historical holdout.
    economics_path = OUTPUT_DIR / "holdout_economics.json"
    if prior.get("status") == "holdout_scored" and economics_path.is_file():
        economics = json.loads(economics_path.read_text(encoding="utf-8"))
    else:
        economics = _economic_evaluation(holdout_predictions, t59, t60)
        _write_json(economics_path, economics)

    correction_diag = scored_payload["correction_diagnostics"]
    metrics = scored_payload["metrics"]
    dev_metric = {item["forecast"]: item for item in metrics if item["period"] == "development_cv_optimized"}
    paired = scored_payload["paired_holdout"]
    ind_a = economics["models"]["A"]["independent_fixed_5_usd"]
    ind_d = economics["models"]["D"]["independent_fixed_5_usd"]
    port_a = economics["models"]["A"]["portfolio_100_usd_free_cash_5pct"]
    port_d = economics["models"]["D"]["portfolio_100_usd_free_cash_5pct"]
    conclusion = (
        f"On development OOF, selected D log-loss was {dev_metric['D']['log_loss']:.6f} versus raw A {dev_metric['A']['log_loss']:.6f}. "
        f"On the historically exposed holdout, D-minus-A/B/C log-loss differences were "
        f"{paired['D-A']['log_loss_difference_left_minus_right']:+.6f}, "
        f"{paired['D-B']['log_loss_difference_left_minus_right']:+.6f}, and "
        f"{paired['D-C']['log_loss_difference_left_minus_right']:+.6f}; negative favors D. "
        f"The $5 independent-order settled PnL was ${ind_d['settled_net_pnl_usd']:.2f} for D and ${ind_a['settled_net_pnl_usd']:.2f} for A. "
        f"The free-cash portfolio cutoff cash was ${port_d['cash_at_cutoff_usd']:.2f} for D and ${port_a['cash_at_cutoff_usd']:.2f} for A; "
        "see the after-cost table and monthly/context diagnostics for stability. Holdout remains descriptive because it was viewed earlier; this result does not change the procedure and does not reject other conditional models."
    )
    payload = {
        "sample": sample, "selection": config_results, "chosen_config": chosen,
        "final_tree_count": final_tree_count, "freeze_time_utc": freeze_time,
        "metrics": metrics, "paired_dev_optimized": scored_payload["paired_dev_optimized"],
        "paired_holdout": scored_payload["paired_holdout"], "monthly_quality": scored_payload["monthly_quality"],
        "correction_diagnostics": correction_diag, "period_diagnostics": scored_payload["period_diagnostics"],
        "context_diagnostics": scored_payload["context_diagnostics"],
        "economics": economics, "conclusion": conclusion,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    elapsed = time.perf_counter() - EXPERIMENT_STARTED
    _check_budget()
    peak_working_set = _peak_working_set_bytes()
    payload["cost"] = {
        "elapsed_seconds": elapsed,
        "peak_working_set_gib": peak_working_set / 1024**3 if peak_working_set is not None else None,
        "generated_output_bytes": sum(path.stat().st_size for path in OUTPUT_DIR.rglob("*") if path.is_file()),
    }
    REPORT_PATH.write_text(_render_report(payload), encoding="utf-8")
    _write_json(OUTPUT_DIR / "final_results.json", payload)
    generated_bytes = sum(path.stat().st_size for path in OUTPUT_DIR.rglob("*") if path.is_file())
    if generated_bytes > int(CONFIG["budget"]["maximum_generated_checkpoint_bytes"]):
        raise RuntimeError(f"Generated experiment outputs exceed the preregistered cap: {generated_bytes} bytes")
    run = json.loads(RUN_PATH.read_text(encoding="utf-8"))
    run.update({"status": "completed", "elapsed_seconds": elapsed, "peak_working_set_bytes": peak_working_set, "generated_output_bytes": generated_bytes, "report_path": REPORT_PATH.resolve().relative_to(ROOT).as_posix(), "holdout_was_scored_once": True})
    _write_json(RUN_PATH, run)
    print(f"[conditional-offset] completed in {elapsed:.1f}s; report={REPORT_PATH}", flush=True)
    return payload


if __name__ == "__main__":
    main()
