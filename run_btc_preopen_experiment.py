"""Run the resumable BTC pre-open v1 training pipeline from its file profile."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.special import expit

from features.btc_preopen_baseline import (
    FEATURE_COLUMNS,
    build_causal_preopen_features,
    regularize_minute_candles,
)
from features.btc_preopen_contract import (
    DECISION_COL,
    RETURN_TARGET_COL,
    TARGET_AVAILABLE_COL,
    TARGET_COL,
    TARGET_END_PRICE_COL,
    TARGET_START_PRICE_COL,
    build_preopen_contract_frame,
    build_preopen_target_weights,
    last_closed_candle_opened_at,
    preopen_decision_mask,
    purge_unavailable_training_rows,
)
from utils.data import TARGET_WEIGHT_COL
from utils.polymarket_policy import calibration_metrics


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "configs/btc_preopen_v1.json"
RAW_CANDLES_PATH = ROOT / "data/datasets/raw/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m.csv"
OFFICIAL_MARKETS_PATH = (
    ROOT
    / "data/analysis/polymarket/BTC/new_model_comparison/runs/"
    / "d794ac2dea5a25a2/shared_market_evaluation.parquet"
)
DATA_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1"
MODEL_DIR = ROOT / "data/models/BTC/btc_preopen_v1"
FEATURE_CACHE_PATH = DATA_DIR / "causal_minute_features.parquet"
FEATURE_CACHE_META_PATH = DATA_DIR / "causal_minute_features.manifest.json"
RUN_MANIFEST_PATH = DATA_DIR / "run_manifest.json"
ITERATION_CHECKPOINT_PATH = DATA_DIR / "iteration_selection.json"
MODEL_CHECKPOINT_PATH = MODEL_DIR / "training.manifest.json"
METRICS_PATH = DATA_DIR / "evaluation.json"
PREDICTIONS_PATH = DATA_DIR / "external_test_predictions.parquet"
MODEL_PATH = MODEL_DIR / "lgbm_model.txt"
CALIBRATOR_PATH = MODEL_DIR / "platt_calibrator.json"
MODEL_META_PATH = MODEL_DIR / "lgbm_meta.json"
FEATURE_SOURCE_PATH = MODEL_DIR / "feature_sources.json"

FIT_END = pd.Timestamp("2026-01-01T00:00:00Z")
CALIBRATION_START = pd.Timestamp("2026-01-01T00:00:00Z")
CALIBRATION_END = pd.Timestamp("2026-04-15T17:04:00Z")
TEST_FIRST_DECISION = pd.Timestamp("2026-04-15T17:04:00Z")
TEST_MARKET_START = pd.Timestamp("2026-04-15T17:05:00Z")
TEST_LAST_MARKET_START = pd.Timestamp("2026-05-18T10:30:00Z")
VALIDATION_TRAIN_FRACTION = 0.80
BOOSTING_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 100
NUM_THREADS = 8
BOOTSTRAP_BLOCK_DAYS = 3
BOOTSTRAP_REPLICATIONS = 2000
BOOTSTRAP_SEED = 20261004
SEED = 20261004
SMOKE_TRAIN_ROWS = 20_000
SMOKE_VALIDATION_ROWS = 5_000
SMOKE_BOOSTING_ROUNDS = 20
_BOOTSTRAP_INDEX_CACHE = None

PRICE_TARGET_NAME = TARGET_COL
OFFICIAL_TARGET_COL = "target_polymarket_up"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _feature_cache_identity():
    return {
        "raw_candles_sha256": _sha256(RAW_CANDLES_PATH),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "feature_code_sha256": _sha256(
            ROOT / "features/btc_preopen_baseline.py"
        ),
        "target_code_sha256": _sha256(ROOT / "features/btc_preopen_contract.py"),
        "target": PRICE_TARGET_NAME,
        "feature_order": list(FEATURE_COLUMNS),
    }


def _load_or_build_feature_cache(identity):
    if FEATURE_CACHE_PATH.is_file() and FEATURE_CACHE_META_PATH.is_file():
        cached_identity = json.loads(
            FEATURE_CACHE_META_PATH.read_text(encoding="utf-8")
        )
        if cached_identity == identity:
            print(f"[preopen] loading verified causal feature cache: {FEATURE_CACHE_PATH}", flush=True)
            return pd.read_parquet(FEATURE_CACHE_PATH)

    print("[preopen] reading raw BTC index candles", flush=True)
    columns = [
        "Opened",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "UM_BTCUSDT_Close",
    ]
    raw = pd.read_csv(RAW_CANDLES_PATH, usecols=columns)
    raw["Opened"] = pd.to_datetime(raw["Opened"], utc=True, errors="raise")
    candles = regularize_minute_candles(raw)
    del raw

    print(f"[preopen] building causal features and exact rolling target ({len(candles):,} minutes)", flush=True)
    features = build_causal_preopen_features(candles)
    contract = build_preopen_contract_frame(candles)
    dataset = features
    dataset[TARGET_COL] = contract[TARGET_COL].to_numpy(dtype=np.float64, copy=False)
    dataset[DECISION_COL] = contract[DECISION_COL].to_numpy(dtype=np.bool_, copy=False)
    dataset["nominal_decision_at"] = contract["nominal_decision_at"]
    dataset[TARGET_AVAILABLE_COL] = contract[TARGET_AVAILABLE_COL]
    dataset["market_start_utc"] = contract["target_window_start_at"]
    dataset[TARGET_START_PRICE_COL] = contract[TARGET_START_PRICE_COL]
    dataset[TARGET_END_PRICE_COL] = contract[TARGET_END_PRICE_COL]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = FEATURE_CACHE_PATH.with_suffix(".parquet.tmp")
    dataset.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(FEATURE_CACHE_PATH)
    _write_json(FEATURE_CACHE_META_PATH, identity)
    return dataset


def _model_params():
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 1000,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "verbosity": -1,
        "device_type": "cpu",
        "num_threads": NUM_THREADS,
        "deterministic": True,
        "force_col_wise": True,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
    }


def _valid_matrix(frame):
    x = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    y = frame[TARGET_COL].to_numpy(dtype=np.float64, copy=False)
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    target = np.zeros(len(y), dtype=np.int8)
    finite_y = np.isfinite(y)
    target[finite_y] = y[finite_y].astype(np.int8, copy=False)
    return x, target, mask


def _run_smoke(frame, x, y, valid):
    fit_rows = np.flatnonzero(
        valid & frame[TARGET_AVAILABLE_COL].le(FIT_END).to_numpy()
    )
    validation_begin = int(len(fit_rows) * 0.80)
    validation_rows = fit_rows[
        validation_begin: validation_begin + SMOKE_VALIDATION_ROWS
    ]
    if len(validation_rows) != SMOKE_VALIDATION_ROWS:
        raise RuntimeError("Smoke validation does not have the requested development rows")
    cutoff = pd.Timestamp(frame["nominal_decision_at"].iloc[validation_rows[0]])
    train_candidates = fit_rows[:validation_begin]
    train_candidates = train_candidates[-SMOKE_TRAIN_ROWS:]
    train_rows = train_candidates[
        frame[TARGET_AVAILABLE_COL].iloc[train_candidates].to_numpy() <= cutoff
    ]
    if len(train_rows) == 0:
        raise RuntimeError("Smoke training rows were all unavailable by validation time")
    if frame[TARGET_AVAILABLE_COL].iloc[train_rows].max() > cutoff:
        raise RuntimeError("Smoke run failed timestamp-based training-label purge")
    if frame["nominal_decision_at"].iloc[validation_rows].ge(FIT_END).any():
        raise RuntimeError("Smoke validation crossed the frozen development boundary")
    train_set = lgb.Dataset(x[train_rows], label=y[train_rows], feature_name=list(FEATURE_COLUMNS))
    valid_set = lgb.Dataset(
        x[validation_rows],
        label=y[validation_rows],
        reference=train_set,
        feature_name=list(FEATURE_COLUMNS),
    )
    smoke_model = lgb.train(
        _model_params(),
        train_set,
        num_boost_round=SMOKE_BOOSTING_ROUNDS,
        valid_sets=[valid_set],
        valid_names=["development_smoke_validation"],
        callbacks=[lgb.log_evaluation(period=SMOKE_BOOSTING_ROUNDS)],
    )
    probability = smoke_model.predict(x[validation_rows])
    if not np.isfinite(probability).all():
        raise RuntimeError("Smoke model produced non-finite validation probabilities")
    return {
        "status": "passed",
        "target": TARGET_COL,
        "features": len(FEATURE_COLUMNS),
        "training_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_rows)),
        "first_validation_nominal_decision_at": cutoff,
        "latest_training_label_available_at": frame[TARGET_AVAILABLE_COL].iloc[train_rows].max(),
        "external_test_feature_rows_in_training": 0,
        "external_test_rows_in_iteration_or_calibration_fit": 0,
        "boosting_rounds": SMOKE_BOOSTING_ROUNDS,
        "validation_log_loss": float(
            smoke_model.best_score["development_smoke_validation"]["binary_logloss"]
        ),
    }


def _fit_iteration_count(frame, x, y, valid):
    fit_rows = np.flatnonzero(
        valid & frame[TARGET_AVAILABLE_COL].le(FIT_END).to_numpy()
    )
    if len(fit_rows) < 100_000:
        raise RuntimeError(f"Too few point-in-time development rows: {len(fit_rows)}")
    split = int(len(fit_rows) * VALIDATION_TRAIN_FRACTION)
    validation_start_idx = int(fit_rows[split])
    validation_decision_at = pd.Timestamp(
        frame["nominal_decision_at"].iloc[validation_start_idx]
    )
    train_rows = fit_rows[:split]
    known_at_validation = (
        frame[TARGET_AVAILABLE_COL].iloc[train_rows].to_numpy()
        <= validation_decision_at
    )
    purged_rows = train_rows[known_at_validation]
    validation_rows = fit_rows[split:]
    if len(purged_rows) == 0 or len(validation_rows) == 0:
        raise RuntimeError("Chronological development split produced an empty partition")

    train_set = lgb.Dataset(x[purged_rows], label=y[purged_rows], feature_name=list(FEATURE_COLUMNS))
    valid_set = lgb.Dataset(
        x[validation_rows],
        label=y[validation_rows],
        reference=train_set,
        feature_name=list(FEATURE_COLUMNS),
    )
    print(
        "[preopen] iteration selection | "
        f"train={len(purged_rows):,} validation={len(validation_rows):,} "
        f"purged={len(train_rows) - len(purged_rows):,} "
        f"validation_decision={validation_decision_at.isoformat()} threads={NUM_THREADS}",
        flush=True,
    )
    model = lgb.train(
        _model_params(),
        train_set,
        num_boost_round=BOOSTING_ROUNDS,
        valid_sets=[valid_set],
        valid_names=["development_validation"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    result = {
        "best_iteration": int(model.best_iteration),
        "validation_decision_at": validation_decision_at,
        "validation_start_opened": frame["Opened"].iloc[validation_rows[0]],
        "validation_end_opened": frame["Opened"].iloc[validation_rows[-1]],
        "training_rows": int(len(purged_rows)),
        "validation_rows": int(len(validation_rows)),
        "purged_rows": int(len(train_rows) - len(purged_rows)),
        "best_validation_log_loss": float(
            model.best_score["development_validation"]["binary_logloss"]
        ),
    }
    del train_set, valid_set, model
    return result


def _fit_final_model(frame, x, y, valid, best_iteration):
    fit_mask = valid & frame[TARGET_AVAILABLE_COL].le(FIT_END).to_numpy()
    fit_rows = np.flatnonzero(fit_mask)
    if not len(fit_rows):
        raise RuntimeError("No final training rows remain after label-availability cutoff")
    training_set = lgb.Dataset(x[fit_rows], label=y[fit_rows], feature_name=list(FEATURE_COLUMNS))
    print(
        f"[preopen] final fit | all eligible minute rows={len(fit_rows):,} "
        f"label_available_through={FIT_END.isoformat()} iterations={best_iteration}",
        flush=True,
    )
    model = lgb.train(
        _model_params(),
        training_set,
        num_boost_round=int(best_iteration),
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix(".txt.tmp")
    model.save_model(str(temporary))
    temporary.replace(MODEL_PATH)
    del training_set
    return model, int(len(fit_rows))


def _score(y, probability):
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    log_loss = -np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))
    result = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "log_loss": float(log_loss),
        "brier": float(np.mean((p - y) ** 2)),
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "calibration": calibration_metrics(y, p),
    }
    return result


def _paired_block_interval(y, first_p, second_p):
    global _BOOTSTRAP_INDEX_CACHE
    y = np.asarray(y, dtype=np.int8)
    first_p = np.clip(np.asarray(first_p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    second_p = np.clip(np.asarray(second_p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    n = len(y)
    if _BOOTSTRAP_INDEX_CACHE is None or _BOOTSTRAP_INDEX_CACHE.shape[1] != n:
        block_size = 288 * BOOTSTRAP_BLOCK_DAYS
        blocks = int(math.ceil(n / block_size))
        offsets = np.arange(block_size, dtype=np.int64)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        starts = rng.integers(0, n, size=(BOOTSTRAP_REPLICATIONS, blocks))
        _BOOTSTRAP_INDEX_CACHE = (
            (starts[:, :, None] + offsets[None, None, :]) % n
        ).reshape(BOOTSTRAP_REPLICATIONS, -1)[:, :n]
    indices = _BOOTSTRAP_INDEX_CACHE
    first_loss = -(y * np.log(first_p) + (1 - y) * np.log1p(-first_p))
    second_loss = -(y * np.log(second_p) + (1 - y) * np.log1p(-second_p))
    first_brier = (first_p - y) ** 2
    second_brier = (second_p - y) ** 2
    deltas = np.column_stack(
        (
            (first_loss - second_loss)[indices].mean(axis=1),
            (first_brier - second_brier)[indices].mean(axis=1),
        )
    )
    return {
        "method": "paired circular moving-block bootstrap",
        "block_days": BOOTSTRAP_BLOCK_DAYS,
        "replications": BOOTSTRAP_REPLICATIONS,
        "seed": BOOTSTRAP_SEED,
        "delta_first_minus_second_log_loss_ci95": np.quantile(
            deltas[:, 0], [0.025, 0.975]
        ).tolist(),
        "delta_first_minus_second_brier_ci95": np.quantile(
            deltas[:, 1], [0.025, 0.975]
        ).tolist(),
    }


def _fit_calibrator(model, frame, x, y, identity, *, minimum_rows=1000):
    if CALIBRATOR_PATH.is_file():
        existing = json.loads(CALIBRATOR_PATH.read_text(encoding="utf-8"))
        if existing.get("input_identity") == identity:
            print("[preopen] resuming verified Platt calibration checkpoint", flush=True)
            return existing

    target = y.astype(np.float64, copy=False)
    decision = frame["nominal_decision_at"]
    mask = (
        frame[DECISION_COL].to_numpy(dtype=np.bool_, copy=False)
        & np.isfinite(target)
        & np.isfinite(x).all(axis=1)
        & decision.ge(CALIBRATION_START).to_numpy()
        & decision.lt(CALIBRATION_END).to_numpy()
        & frame[TARGET_AVAILABLE_COL].le(TEST_FIRST_DECISION).to_numpy()
    )
    rows = np.flatnonzero(mask)
    if len(rows) < int(minimum_rows):
        raise RuntimeError(
            f"Too few point-in-time calibration decisions: {len(rows)}; "
            f"requires {int(minimum_rows)}"
        )
    if np.unique(target[rows]).size != 2:
        raise RuntimeError("Calibration period must contain both proxy target classes")
    raw_probability = np.clip(model.predict(x[rows]), 1e-6, 1.0 - 1e-6)
    logit_probability = np.log(raw_probability / (1.0 - raw_probability)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=SEED)
    calibrator.fit(logit_probability, target[rows].astype(np.int8))
    payload = {
        "method": "Platt logistic calibration of raw model logit",
        "input_identity": identity,
        "label_source": "Binance COIN-M BTCUSD index price proxy",
        "training_rows": int(len(rows)),
        "calibration_start_decision_at": decision.iloc[rows[0]],
        "calibration_end_decision_at": decision.iloc[rows[-1]],
        "latest_label_available_at": frame[TARGET_AVAILABLE_COL].iloc[rows].max(),
        "first_external_test_decision_at": TEST_FIRST_DECISION,
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
    }
    _write_json(CALIBRATOR_PATH, payload)
    return payload


def _apply_calibrator(raw_probability, calibration):
    raw = np.clip(np.asarray(raw_probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(raw / (1.0 - raw))
    return expit(
        float(calibration["intercept"])
        + float(calibration["coefficient"]) * logit
    )


def _evaluate_external_test(model, calibration, frame, x, y_proxy, valid, prior_up):
    mask = (
        valid
        & frame[DECISION_COL].to_numpy(dtype=np.bool_, copy=False)
        & frame["market_start_utc"].ge(TEST_MARKET_START).to_numpy()
        & frame["market_start_utc"].le(TEST_LAST_MARKET_START).to_numpy()
    )
    row_indices = np.flatnonzero(mask)
    sample = frame.iloc[row_indices].copy()
    if len(sample) == 0:
        raise RuntimeError("No complete actual-decision rows in the external test window")
    raw_probability = np.clip(model.predict(x[row_indices]), 1e-6, 1.0 - 1e-6)
    sample["p_model_raw"] = raw_probability
    sample["p_model_platt"] = _apply_calibrator(raw_probability, calibration)
    sample["p_constant_0_5"] = 0.5
    sample["p_development_proxy_prevalence"] = float(prior_up)
    sample["prediction_available_at_utc"] = None
    sample["prediction_timestamp_basis"] = "nominal only; historical compute latency unavailable"

    official = pd.read_parquet(
        OFFICIAL_MARKETS_PATH,
        columns=[
            "condition_id",
            "market_slug",
            "market_start_utc",
            OFFICIAL_TARGET_COL,
            "up_best_ask",
            "down_best_ask",
            "up_ask_size",
            "down_ask_size",
            "quote_delay_ms",
        ],
    )
    sample["market_start_utc"] = pd.to_datetime(sample["market_start_utc"], utc=True)
    official["market_start_utc"] = pd.to_datetime(official["market_start_utc"], utc=True)
    official = official.loc[
        official["market_start_utc"].ge(TEST_MARKET_START)
        & official["market_start_utc"].le(TEST_LAST_MARKET_START)
    ].copy()
    joined = sample.merge(
        official,
        on="market_start_utc",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_official"),
    )
    if len(joined) != len(official):
        raise RuntimeError(
            "External price-proxy predictions did not join one-to-one to official outcomes: "
            f"official={len(official)} joined={len(joined)}"
        )

    metric_probabilities = {
        "constant_0_5": joined["p_constant_0_5"].to_numpy(),
        "development_proxy_prevalence": joined["p_development_proxy_prevalence"].to_numpy(),
        "raw_model": joined["p_model_raw"].to_numpy(),
        "platt_model": joined["p_model_platt"].to_numpy(),
    }
    targets = {
        "official_polymarket": joined[OFFICIAL_TARGET_COL].to_numpy(dtype=np.int8),
        "binance_index_proxy": joined[TARGET_COL].to_numpy(dtype=np.int8),
    }
    metrics = {}
    paired = {}
    for target_name, target in targets.items():
        metrics[target_name] = {
            name: _score(target, probability)
            for name, probability in metric_probabilities.items()
        }
        paired[target_name] = {
            f"{model_name}_minus_{baseline_name}": _paired_block_interval(
                target,
                metric_probabilities[model_name],
                metric_probabilities[baseline_name],
            )
            for model_name in ("raw_model", "platt_model")
            for baseline_name in ("constant_0_5", "development_proxy_prevalence")
        }

    test = {
        "rows": int(len(joined)),
        "complete_price_proxy_decision_rows": int(len(sample)),
        "price_proxy_rows_without_official_market_record": int(len(sample) - len(joined)),
        "market_start_first": joined["market_start_utc"].min(),
        "market_start_last": joined["market_start_utc"].max(),
        "model_proxy_vs_official_disagreements": int(
            joined[TARGET_COL].ne(joined[OFFICIAL_TARGET_COL]).sum()
        ),
        "model_proxy_vs_official_disagreement_rate": float(
            joined[TARGET_COL].ne(joined[OFFICIAL_TARGET_COL]).mean()
        ),
        "paired_uncertainty": paired,
        "prestart_quote_rows": 0,
        "economic_replay": "not estimable; retained snapshots are at market start, not one minute before",
    }
    joined.to_parquet(PREDICTIONS_PATH, index=False, compression="zstd")
    return metrics, test, joined


def run():
    run_training_pipeline(profile_name="nightly_v1")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.streams[0], "errors", "strict")

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return bool(self.streams and self.streams[0].isatty())

    def fileno(self):
        return self.streams[0].fileno()


def _signature(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    ).hexdigest()


def _file_identity(path):
    path = Path(path)
    return {"path": path.resolve().as_posix(), "sha256": _sha256(path), "size": path.stat().st_size}


def _code_fingerprints(paths):
    return {
        Path(path).resolve().relative_to(ROOT).as_posix(): _sha256(Path(path))
        for path in paths
    }


def _array_sha256(values, dtype=None):
    array = np.asarray(values, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _preopen_sample_weights(decision_mask, decision_weight):
    decision_weight = float(decision_weight)
    if not 0.0 < decision_weight < 1.0:
        raise ValueError("decision_weight must be between zero and one")
    return np.where(
        np.asarray(decision_mask, dtype=bool),
        decision_weight,
        (1.0 - decision_weight) / 4.0,
    ).astype(np.float32, copy=False)


def _fold_index_fingerprints(folds):
    fingerprints = {}
    for key in ("train_idx", "valid_idx"):
        digest = hashlib.sha256()
        for fold in folds:
            digest.update(int(fold["fold_id"]).to_bytes(4, "little", signed=True))
            indices = np.asarray(fold[key], dtype="<i4")
            digest.update(len(indices).to_bytes(8, "little", signed=False))
            digest.update(np.ascontiguousarray(indices).view(np.uint8))
        fingerprints[key] = digest.hexdigest()
    return fingerprints


def _lock_run(run_root):
    lock_path = run_root / ".writer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Another process is already writing this pre-open run: {run_root}") from exc
    return handle


def _unlock_run(handle):
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _artifact_records(paths):
    records = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"Required stage artifact was not written: {path}")
        records.append(
            {
                "path": path.resolve().relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _artifacts_still_match(records):
    for record in records:
        path = ROOT / record["path"]
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
            return False
        if _sha256(path) != record["sha256"]:
            return False
    return bool(records)


def _run_stage(
        manifest,
        manifest_path,
        run_root,
        name,
        input_payload,
        action,
        *,
        must_run=None,
        deadline_monotonic=None,
):
    stage_signature = _signature(input_payload)
    previous = manifest.setdefault("stages", {}).get(name, {})
    if (
        previous.get("status") == "completed"
        and previous.get("input_signature") == stage_signature
        and _artifacts_still_match(previous.get("outputs", []))
        and not (must_run and must_run(previous.get("result") or {}))
    ):
        print(f"[preopen] stage={name} status=skipped (verified resume)", flush=True)
        return previous["result"], previous["artifact_signature"], previous["outputs"]

    stage_dir = run_root / "stages" / f"{name}_{stage_signature[:12]}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    entry = {
        "status": "running",
        "input_signature": stage_signature,
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "attempt": int(previous.get("attempt", 0)) + 1,
    }
    manifest["stages"][name] = entry
    manifest["status"] = "running"
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(manifest_path, manifest)
    print(f"[preopen] stage={name} status=started", flush=True)
    try:
        result, outputs = action(stage_dir)
        records = _artifact_records(outputs)
        artifact_signature = _signature(
            [(record["path"], record["sha256"]) for record in records]
        )
        entry.update(
            {
                "status": "completed",
                "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "result": result,
                "outputs": records,
                "artifact_signature": artifact_signature,
            }
        )
        manifest["updated_utc"] = entry["finished_utc"]
        _write_json(manifest_path, manifest)
        print(
            f"[preopen] stage={name} status=completed elapsed={entry['elapsed_seconds']:.1f}s",
            flush=True,
        )
        return result, artifact_signature, records
    except Exception as exc:
        entry.update(
            {
                "status": "failed",
                "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {str(exc)[:1600]}",
            }
        )
        manifest["status"] = "failed"
        manifest["updated_utc"] = entry["finished_utc"]
        _write_json(manifest_path, manifest)
        print(f"[preopen] stage={name} status=failed error={entry['error']}", flush=True)
        raise


def _purged_walk_forward_folds(
        module, opened_values, decision_values, n_folds, ratio, *, validation_mask=None
):
    opened = pd.DatetimeIndex(pd.to_datetime(opened_values, utc=True, errors="raise"))
    decision = pd.DatetimeIndex(pd.to_datetime(decision_values, utc=True, errors="raise"))
    fold_parameters = inspect.signature(module.make_walk_forward_folds).parameters
    fold_count_key = "n_splits" if "n_splits" in fold_parameters else "n_folds"
    folds = module.make_walk_forward_folds(
        n_rows=len(opened),
        **{fold_count_key: int(n_folds)},
        test_to_train_ratio=float(ratio),
    )
    label_available = opened + pd.Timedelta(minutes=7)
    validation_mask = (
        np.ones(len(opened), dtype=bool)
        if validation_mask is None
        else np.asarray(validation_mask, dtype=bool)
    )
    if validation_mask.shape != (len(opened),):
        raise ValueError("validation_mask must contain one value per opened row")
    for fold in folds:
        valid_candidates = (
            np.asarray(fold["valid_idx"], dtype=np.int64)
            if "valid_idx" in fold
            else np.arange(int(fold["test_start"]), int(fold["test_end"]), dtype=np.int64)
        )
        valid_indices = valid_candidates[validation_mask[valid_candidates]]
        if valid_indices.size == 0:
            raise RuntimeError(f"Fold {fold['fold_id']} has no actual decision validation rows")
        first_valid = int(valid_indices[0])
        if "train_idx" in fold:
            candidates = np.asarray(fold["train_idx"], dtype=np.int64)
        else:
            candidates = np.arange(
                int(fold["train_start"]), int(fold["train_end"]), dtype=np.int64
            )
        cutoff = decision[first_valid]
        legal = purge_unavailable_training_rows(
            opened,
            cutoff,
            train_indices=candidates,
        )
        if legal.size == 0:
            raise RuntimeError(f"Fold {fold['fold_id']} has no rows after label-availability purge")
        if "train_idx" not in fold:
            expected = np.arange(int(fold["train_start"]), int(fold["train_end"]), dtype=np.int64)
            if not np.array_equal(legal, expected[: len(legal)]):
                raise RuntimeError("Chronological label purge did not produce a prefix fold")
            fold["train_end"] = int(fold["train_start"] + len(legal))
        fold["train_idx"] = legal.astype(np.int32, copy=False)
        fold["valid_idx"] = valid_indices.astype(np.int32, copy=False)
        if label_available[legal].max() > cutoff:
            raise RuntimeError(f"Fold {fold['fold_id']} retained an unavailable training label")
    return folds


def _configure_training_backend(threads, device_type):
    import fit_reaction_profile as reaction
    import fit_volume_profile as volume
    import optimize_lgbm_hyperparameters as model_tuning
    import optimize_target_weights as weight_search
    import select_features as selector
    import train_lgbm

    threads = int(threads)
    device_type = str(device_type).strip().lower()
    if device_type not in {"cpu", "gpu"}:
        raise ValueError("compute_backend must be 'cpu' or 'gpu'.")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    train_lgbm.N_JOBS = threads
    train_lgbm.LGBM_DEVICE_TYPE = device_type
    train_lgbm._ACTIVE_MONOTONE_CONSTRAINTS_BY_FEATURE = {}
    for module in (reaction, volume):
        module.LGBM_DEVICE_TYPE = device_type
        module.LGBM_NUM_THREADS = threads
        module.OPTUNA_OPTIMIZE_N_JOBS = 1
        module.ENABLE_FOLD_RECENCY_WEIGHTING = False
    model_tuning.LGBM_DEVICE_TYPE = device_type
    model_tuning.LGBM_NUM_THREADS = threads
    model_tuning.OPTUNA_OPTIMIZE_N_JOBS = 1
    model_tuning.ENABLE_FOLD_RECENCY_WEIGHTING = False
    weight_search.DEVICE_TYPE = device_type
    weight_search.LGBM_N_JOBS = threads
    weight_search.ENABLE_FOLD_RECENCY_WEIGHTING = False
    selector.LGBM_DEVICE_TYPE = device_type
    selector.MODEL_PARAMS["device_type"] = device_type
    selector.MODEL_PARAMS["n_jobs"] = threads
    selector.ENABLE_FOLD_RECENCY_WEIGHTING = False
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=threads)
    except ImportError:
        pass
    return reaction, volume, model_tuning, weight_search, selector, train_lgbm


def _run_optuna_stage(
        *,
        optuna,
        stage_name,
        stage_dir,
        objective,
        target_trials,
        minimum_successful_trials,
        stage_deadline,
        overall_deadline,
        seed=20261004,
        catch=(),
):
    database = stage_dir / "study.sqlite3"
    result_path = stage_dir / "best_result.json"
    database.parent.mkdir(parents=True, exist_ok=True)
    study_name = f"btc_preopen_{stage_name}_{stage_dir.name.rsplit('_', 1)[-1]}"
    storage = "sqlite:///" + database.resolve().as_posix()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(seed), n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10),
        load_if_exists=True,
    )
    all_trials = study.get_trials(deepcopy=False)
    successful = [
        trial for trial in all_trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and np.isfinite(float(trial.value))
        and trial.user_attrs.get("trial_status") != "crash_penalty"
    ]
    trial_limit = int(target_trials)
    remaining_trials = max(0, trial_limit - len(all_trials))
    remaining_seconds = max(
        0.0,
        min(float(stage_deadline), float(overall_deadline)) - time.perf_counter(),
    )
    if remaining_trials and remaining_seconds > 0.0:
        print(
            f"[preopen] stage={stage_name} optuna_trials={len(all_trials)}/"
            f"{target_trials} remaining={remaining_trials} seconds={remaining_seconds:.0f}",
            flush=True,
        )
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=remaining_seconds,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=False,
            catch=tuple(catch),
        )
    all_trials = study.get_trials(deepcopy=False)
    successful = [
        trial for trial in all_trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and np.isfinite(float(trial.value))
        and trial.user_attrs.get("trial_status") != "crash_penalty"
    ]
    retry_limit = int(target_trials) + int(minimum_successful_trials)
    retry_count = max(0, retry_limit - len(all_trials)) if len(successful) < int(minimum_successful_trials) else 0
    retry_seconds = max(
        0.0,
        min(float(stage_deadline), float(overall_deadline)) - time.perf_counter(),
    )
    if retry_count and retry_seconds > 0.0:
        study.optimize(
            objective,
            n_trials=retry_count,
            timeout=retry_seconds,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=False,
            catch=tuple(catch),
        )
        all_trials = study.get_trials(deepcopy=False)
        successful = [
            trial for trial in all_trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and np.isfinite(float(trial.value))
            and trial.user_attrs.get("trial_status") != "crash_penalty"
        ]
    if len(successful) < int(minimum_successful_trials):
        raise RuntimeError(
            f"{stage_name} has {len(successful)} successful Optuna trials; "
            f"requires {minimum_successful_trials} before the time budget ended. "
            f"Study checkpoint is saved at {database}"
        )
    best = min(successful, key=lambda trial: float(trial.value))
    payload = {
        "stage": stage_name,
        "study_name": study_name,
        "study_database": database.resolve().relative_to(ROOT).as_posix(),
        "trials_total": len(all_trials),
        "successful_trials": len(successful),
        "target_trials": int(target_trials),
        "minimum_successful_trials": int(minimum_successful_trials),
        "search_complete": len(all_trials) >= int(target_trials),
        "stopped_by": (
            "target_trials"
            if len(all_trials) >= int(target_trials)
            else "time_budget"
        ),
        "best_trial_number": int(best.number),
        "best_value": float(best.value),
        "best_params": dict(best.params),
        "best_iteration": best.user_attrs.get("best_iteration"),
        "best_config": best.user_attrs.get("normalized_config"),
    }
    _write_json(result_path, payload)
    return payload, [database, result_path]


def _model_feature_frame(dataset):
    reserved = {
        "Opened",
        TARGET_COL,
        RETURN_TARGET_COL,
        TARGET_WEIGHT_COL,
        DECISION_COL,
        TARGET_START_PRICE_COL,
        TARGET_END_PRICE_COL,
        "target_window_start_at",
        "target_window_end_at",
        TARGET_AVAILABLE_COL,
        "feature_candle_close_at",
        "nominal_decision_at",
        "nominal_prediction_available_at",
        "actual_prediction_available_at",
        "prediction_deadline_at",
    }
    reserved.update(
        column for column in dataset.columns
        if str(column).startswith("target_") or str(column).endswith("_at")
    )
    reserved.update(name for name in ("Open", "High", "Low", "Close", "Volume") if name in dataset)
    raw = dataset.select_dtypes(include=[np.number]).drop(
        columns=[column for column in reserved if column in dataset.columns],
        errors="ignore",
    )
    if raw.empty:
        raise RuntimeError("Modeling dataset contains no numeric feature columns")
    return raw.replace([np.inf, -np.inf], np.nan)


def _evaluate_preopen_external(
        model, calibration, frame, x, valid, prior_up, official_path,
        test_history_status,
):
    mask = (
        valid
        & frame[DECISION_COL].to_numpy(dtype=np.bool_, copy=False)
        & frame["market_start_utc"].ge(TEST_MARKET_START).to_numpy()
        & frame["market_start_utc"].le(TEST_LAST_MARKET_START).to_numpy()
    )
    row_indices = np.flatnonzero(mask)
    sample = frame.iloc[row_indices].copy()
    if len(sample) == 0:
        raise RuntimeError("No complete actual-decision rows in the external test window")
    raw_probability = np.clip(model.predict(x[row_indices]), 1e-6, 1.0 - 1e-6)
    sample["p_model_raw"] = raw_probability
    sample["p_model_platt"] = _apply_calibrator(raw_probability, calibration)
    sample["p_constant_0_5"] = 0.5
    sample["p_development_proxy_prevalence"] = float(prior_up)
    sample["prediction_available_at_utc"] = (
        pd.to_datetime(sample["nominal_decision_at"], utc=True)
        + pd.to_timedelta(float(PREDICTION_COMPUTE_BUDGET_SECONDS), unit="s")
    )
    sample["prediction_timestamp_basis"] = "nominal candle close + configured compute budget; historical receive latency unavailable"
    y_proxy = sample[TARGET_COL].to_numpy(dtype=np.int8, copy=False)
    probabilities = {
        "constant_0_5": sample["p_constant_0_5"].to_numpy(dtype=np.float64),
        "development_proxy_prevalence": sample["p_development_proxy_prevalence"].to_numpy(dtype=np.float64),
        "raw_model": sample["p_model_raw"].to_numpy(dtype=np.float64),
        "platt_model": sample["p_model_platt"].to_numpy(dtype=np.float64),
    }
    metrics = {
        "label_source": "Binance COIN-M BTCUSD index price proxy; not official Polymarket settlement",
        "price_proxy": {name: _score(y_proxy, value) for name, value in probabilities.items()},
        "paired_uncertainty": {
            f"{model_name}_minus_{baseline_name}": _paired_block_interval(
                y_proxy, probabilities[model_name], probabilities[baseline_name]
            )
            for model_name in ("raw_model", "platt_model")
            for baseline_name in ("constant_0_5", "development_proxy_prevalence")
        },
    }
    official_summary = {
        "status": "unavailable",
        "reason": "official market outcomes and pre-start quotes were not present",
    }
    if Path(official_path).is_file():
        try:
            columns = [
                "condition_id", "market_slug", "market_start_utc", OFFICIAL_TARGET_COL,
                "up_best_ask", "down_best_ask", "up_ask_size", "down_ask_size", "quote_delay_ms",
            ]
            import pyarrow.parquet as pq

            available = set(pq.ParquetFile(official_path).schema_arrow.names)
            columns = [column for column in columns if column in available]
            if "market_start_utc" in columns and OFFICIAL_TARGET_COL in columns:
                official = pd.read_parquet(official_path, columns=columns)
                official["market_start_utc"] = pd.to_datetime(official["market_start_utc"], utc=True)
                official = official.merge(
                    sample[["market_start_utc", "p_constant_0_5", "p_development_proxy_prevalence", "p_model_raw", "p_model_platt", TARGET_COL]],
                    on="market_start_utc", how="inner", validate="one_to_one",
                )
                official = official.loc[official[OFFICIAL_TARGET_COL].notna()].copy()
                if official.empty:
                    raise LookupError("no complete official outcomes overlap the requested test rows")
                target = official[OFFICIAL_TARGET_COL].to_numpy(dtype=np.int8, copy=False)
                official_summary = {
                    "status": "available",
                    "rows": int(len(official)),
                    "metrics": {
                        name: _score(target, official[prediction_column].to_numpy(dtype=np.float64))
                        for name, prediction_column in {
                            "constant_0_5": "p_constant_0_5",
                            "development_proxy_prevalence": "p_development_proxy_prevalence",
                            "raw_model": "p_model_raw",
                            "platt_model": "p_model_platt",
                        }.items()
                    },
                    "price_proxy_disagreements": int(
                        official[TARGET_COL].ne(official[OFFICIAL_TARGET_COL]).sum()
                    ),
                    "economic_replay": "not estimable without pre-start quote snapshots; market-start quotes are excluded",
                }
            else:
                official_summary["reason"] = "official parquet lacks market_start_utc or official target"
        except Exception as exc:
            official_summary["reason"] = f"official parquet could not be joined: {type(exc).__name__}: {str(exc)[:300]}"
    metrics["official_market_data"] = official_summary
    sample.to_parquet(PREDICTIONS_PATH, index=False, compression="zstd")
    external = {
        "rows": int(len(sample)),
        "market_start_first": sample["market_start_utc"].min(),
        "market_start_last": sample["market_start_utc"].max(),
        "historically_exposed": str(test_history_status).startswith("historically exposed"),
        "economic_replay": "not estimable without pre-start quotes; predictive evaluation retained",
    }
    return metrics, external, sample


def load_prediction_bundle(bundle_path, *, require_verified=True):
    bundle_path = Path(bundle_path)
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if require_verified and payload.get("status") != "verified":
        raise RuntimeError("Prediction bundle has not passed its raw-data replay check")
    model_path = ROOT / payload["model_path"]
    calibrator_path = ROOT / payload["calibrator_path"]
    feature_order_path = ROOT / payload["feature_order_path"]
    feature_sources_path = ROOT / payload["feature_sources_path"]
    required_paths = [
        model_path,
        calibrator_path,
        feature_order_path,
        feature_sources_path,
        ROOT / payload["model_meta_path"],
        ROOT / payload["prediction_reference_path"],
        ROOT / payload["reaction_profile_config_path"],
        ROOT / payload["volume_profile_config_path"],
        *(ROOT / path for path in payload.get("profile_state_artifacts", [])),
        *(ROOT / path for path in payload["indicator_fit_result_paths"]),
    ]
    if require_verified:
        required_paths.append(ROOT / payload["verification_path"])
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Prediction bundle is missing required artifacts: {missing[:5]}")
    if _sha256(model_path) != payload.get("model_sha256"):
        raise RuntimeError("Prediction bundle model checksum does not match its manifest")
    if _sha256(calibrator_path) != payload.get("calibrator_sha256"):
        raise RuntimeError("Prediction bundle calibrator checksum does not match its manifest")
    model = lgb.Booster(model_file=str(model_path))
    calibrator = json.loads(calibrator_path.read_text(encoding="utf-8"))
    feature_order = json.loads(feature_order_path.read_text(encoding="utf-8")).get("feature_order")
    if list(model.feature_name()) != list(payload["feature_order"]) or feature_order != payload["feature_order"]:
        raise RuntimeError("Prediction bundle feature order differs from the LightGBM model")
    feature_sources = json.loads(feature_sources_path.read_text(encoding="utf-8"))
    if feature_sources.get("feature_order") != payload["feature_order"]:
        raise RuntimeError("Prediction bundle feature sources have a different feature order")
    state_dir = ROOT / feature_sources.get("profile_state_dir", "")
    if not state_dir.is_dir():
        raise RuntimeError(f"Prediction bundle generator state directory is missing: {state_dir}")
    model_meta = json.loads((ROOT / payload["model_meta_path"]).read_text(encoding="utf-8"))
    if (
        model_meta.get("feature_columns") != payload["feature_order"]
        or model_meta.get("model_sha256") != payload["model_sha256"]
        or model_meta.get("calibration_path") != payload["calibrator_path"]
    ):
        raise RuntimeError("Prediction bundle model metadata does not match its dependencies")
    if not (ROOT / payload["indicator_fit_results_dir"]).is_dir():
        raise RuntimeError("Prediction bundle indicator result directory is missing")
    if require_verified:
        verification = json.loads((ROOT / payload["verification_path"]).read_text(encoding="utf-8"))
        if verification.get("status") != "verified" or verification != payload.get("raw_replay_verification"):
            raise RuntimeError("Prediction bundle raw replay verification is missing or inconsistent")
    return payload, model, calibrator


def _read_batch_feature_row(dataset_path, reference_opened, feature_order):
    """Read one saved batch feature row, using row-group timestamp statistics."""
    import pyarrow.parquet as parquet

    reader = parquet.ParquetFile(dataset_path)
    try:
        opened_column = reader.schema_arrow.get_field_index("Opened")
        if opened_column < 0:
            raise RuntimeError("Batch feature dataset does not contain Opened")
        row_groups = []
        for row_group_index in range(reader.metadata.num_row_groups):
            stats = reader.metadata.row_group(row_group_index).column(opened_column).statistics
            if stats is None or not stats.has_min_max:
                row_groups.append(row_group_index)
                continue
            group_min = pd.Timestamp(stats.min)
            group_max = pd.Timestamp(stats.max)
            if group_min.tzinfo is None:
                group_min = group_min.tz_localize("UTC")
            else:
                group_min = group_min.tz_convert("UTC")
            if group_max.tzinfo is None:
                group_max = group_max.tz_localize("UTC")
            else:
                group_max = group_max.tz_convert("UTC")
            if group_min <= reference_opened <= group_max:
                row_groups.append(row_group_index)
        for row_group_index in row_groups:
            batch = reader.read_row_group(row_group_index, columns=["Opened", *feature_order])
            frame = batch.to_pandas()
            opened = pd.DatetimeIndex(pd.to_datetime(frame["Opened"], utc=True, errors="raise"))
            matches = np.flatnonzero(opened == reference_opened)
            if matches.size:
                return frame.iloc[int(matches[0])].copy(deep=True)
    finally:
        reader.close()
    raise RuntimeError(f"Batch feature dataset does not contain {reference_opened.isoformat()}")


def _verify_prediction_bundle_from_raw(
    bundle_path, prediction_path, verification_path, reference_opened=None
):
    """Rebuild through the last available input candle and compare a saved batch prediction."""
    import tempfile

    import pyarrow.parquet as parquet
    from utils.data import load_modeling_dataset_settings

    bundle_path = Path(bundle_path)
    prediction_path = Path(prediction_path)
    verification_path = Path(verification_path)
    payload, model, calibrator = load_prediction_bundle(bundle_path, require_verified=False)
    raw_value = Path(payload["raw_candle_path"])
    raw_path = raw_value if raw_value.is_absolute() else ROOT / raw_value
    if not raw_path.is_file() or _sha256(raw_path) != payload["raw_candle_sha256"]:
        raise RuntimeError("Raw candle source is missing or differs from the bundle manifest")
    prediction = pd.read_parquet(prediction_path, columns=["Opened", "p_model_raw", "p_model_platt"])
    if prediction.empty:
        raise RuntimeError("Training evaluation contains no prediction for bundle replay")
    if reference_opened is None:
        reference_opened = prediction.iloc[0]["Opened"]
    reference_opened = pd.Timestamp(reference_opened)
    if reference_opened.tzinfo is None:
        reference_opened = reference_opened.tz_localize("UTC")
    else:
        reference_opened = reference_opened.tz_convert("UTC")
    reference_rows = prediction.loc[
        pd.to_datetime(prediction["Opened"], utc=True, errors="raise").eq(reference_opened)
    ]
    if len(reference_rows) != 1:
        raise RuntimeError(
            f"Expected one batch prediction for {reference_opened.isoformat()}, got {len(reference_rows)}"
        )
    reference = reference_rows.iloc[0]
    decision_at = reference_opened + pd.Timedelta(minutes=1)
    raw_cutoff = last_closed_candle_opened_at(decision_at)
    if raw_cutoff != reference_opened:
        raise RuntimeError("Reference row is not the last candle closed at its nominal decision")

    with tempfile.TemporaryDirectory(prefix="btc_preopen_bundle_replay_", dir=bundle_path.parent) as temp_name:
        work_dir = Path(temp_name)
        raw_prefix_path = work_dir / raw_path.name
        wrote_header = False
        previous_opened = None
        for chunk in pd.read_csv(raw_path, chunksize=200_000):
            if "Opened" not in chunk:
                raise RuntimeError("Raw candles do not contain the Opened timestamp")
            opened = pd.DatetimeIndex(pd.to_datetime(chunk["Opened"], utc=True, errors="raise"))
            if not opened.is_monotonic_increasing or (
                previous_opened is not None and opened[0] < previous_opened
            ):
                raise RuntimeError("Raw candle source must be chronological for causal replay")
            previous_opened = opened[-1]
            keep = opened <= raw_cutoff
            if keep.any():
                chunk.loc[keep].to_csv(
                    raw_prefix_path,
                    mode="a" if wrote_header else "w",
                    header=not wrote_header,
                    index=False,
                )
                wrote_header = True
            if not keep.all():
                break
        if not wrote_header or previous_opened is None or previous_opened < raw_cutoff:
            raise RuntimeError("Raw candle source does not reach the last closed reference candle")

        settings = load_modeling_dataset_settings(
            asset="BTC",
            dataset_profile_name=payload["dataset_profile"],
            modeling_profile_name=payload["modeling_profile"],
        )
        settings.update(
            {
                "raw_data_dir": work_dir,
                "base_data_file": raw_prefix_path.name,
                "feature_subset_path": None,
                "feature_subset_list_key": None,
                "excluded_feature_names": [],
                "fit_results_dir": ROOT / payload["indicator_fit_results_dir"],
                "modeling_output_dir": work_dir / "dataset_output",
                "output_suffix": "_preopen_v1",
                "profile_state_dir": work_dir / "states",
                "reaction_profile_fixed_grid": json.loads(
                    (ROOT / payload["reaction_profile_config_path"]).read_text(encoding="utf-8")
                ),
                "volume_profile_fixed_range": json.loads(
                    (ROOT / payload["volume_profile_config_path"]).read_text(encoding="utf-8")
                ),
                "preopen_target": {
                    "prediction_compute_budget_seconds": float(
                        payload["prediction_compute_budget_seconds"]
                    )
                },
            }
        )
        import create_modeling_dataset

        replay_dataset_path = create_modeling_dataset.build_dataset_from_settings(settings)
        batch_reader = parquet.ParquetFile(replay_dataset_path)
        feature_order = list(payload["feature_order"])
        replay_row = None
        try:
            for batch in batch_reader.iter_batches(
                columns=["Opened", TARGET_COL, *feature_order], batch_size=65_536
            ):
                batch_frame = batch.to_pandas()
                opened = pd.DatetimeIndex(pd.to_datetime(batch_frame["Opened"], utc=True, errors="raise"))
                matches = np.flatnonzero(opened == reference_opened)
                if matches.size:
                    replay_row = batch_frame.iloc[int(matches[0])].copy(deep=True)
                    break
        finally:
            batch_reader.close()
            del batch_reader
            if "batch" in locals():
                del batch
            if "batch_frame" in locals():
                del batch_frame
            import gc

            gc.collect()
        if replay_row is None:
            raise RuntimeError(f"Raw replay dataset does not contain {reference_opened.isoformat()}")
        if not pd.isna(replay_row.get(TARGET_COL)):
            raise RuntimeError("Point-in-time replay unexpectedly has a future-dependent target")
        feature_row = pd.to_numeric(replay_row[feature_order], errors="coerce").to_numpy(
            dtype=np.float32, copy=True
        ).reshape(1, -1)
        feature_row[~np.isfinite(feature_row)] = np.nan
        batch_dataset_path = None
        for stage in json.loads((bundle_path.parent.parent.parent / "run_manifest.json").read_text(
            encoding="utf-8"
        )).get("stages", {}).values():
            result = stage.get("result", {}) if isinstance(stage, dict) else {}
            candidate = result.get("dataset_path") if isinstance(result, dict) else None
            if candidate:
                batch_dataset_path = Path(candidate)
                if not batch_dataset_path.is_absolute():
                    batch_dataset_path = ROOT / batch_dataset_path
                break
        if batch_dataset_path is None or not batch_dataset_path.is_file():
            raise RuntimeError("Run manifest does not identify the saved batch feature dataset")
        batch_row = _read_batch_feature_row(batch_dataset_path, reference_opened, feature_order)
        batch_features = pd.to_numeric(batch_row[feature_order], errors="coerce").to_numpy(
            dtype=np.float32, copy=True
        )
        batch_features[~np.isfinite(batch_features)] = np.nan
        feature_equal = np.isclose(feature_row[0], batch_features, rtol=0.0, atol=1e-6, equal_nan=True)
        feature_delta = np.abs(feature_row[0].astype(np.float64) - batch_features.astype(np.float64))
        finite_deltas = feature_delta[np.isfinite(feature_delta)]
        raw_probability = float(model.predict(feature_row)[0])
        calibrated_probability = float(_apply_calibrator(np.asarray([raw_probability]), calibrator)[0])

    tolerance = float(payload["verification_tolerance_abs"])
    raw_delta = abs(raw_probability - float(reference["p_model_raw"]))
    calibrated_delta = abs(calibrated_probability - float(reference["p_model_platt"]))
    if not feature_equal.all() or raw_delta > tolerance or calibrated_delta > tolerance:
        raise RuntimeError(
            "Point-in-time bundle replay differs from batch features or prediction: "
            f"feature_mismatches={(~feature_equal).sum()}, "
            f"raw_delta={raw_delta:.9g}, calibrated_delta={calibrated_delta:.9g}, "
            f"tolerance={tolerance:.9g}"
        )
    _write_json(
        verification_path,
        {
            "status": "verified",
            "reference_opened_utc": reference_opened,
            "reference_prediction_path": prediction_path.resolve().relative_to(ROOT).as_posix(),
            "last_closed_candle_opened_utc": reference_opened,
            "decision_at_utc": decision_at,
            "target_available_for_prediction": False,
            "compared_features": len(feature_order),
            "feature_mismatch_count": int((~feature_equal).sum()),
            "feature_max_abs_delta": float(finite_deltas.max()) if finite_deltas.size else 0.0,
            "raw_probability_delta": raw_delta,
            "calibrated_probability_delta": calibrated_delta,
            "absolute_tolerance": tolerance,
            "raw_history_cutoff_utc": raw_cutoff,
            "generator_state_source": "rebuilt from raw prefix in a fresh process",
        },
    )


def _run_bundle_raw_replay_fresh_process(bundle_path, prediction_path, verification_path):
    child_code = (
        "import run_btc_preopen_experiment as runner\n"
        f"runner._verify_prediction_bundle_from_raw({str(bundle_path)!r}, "
        f"{str(prediction_path)!r}, {str(verification_path)!r})\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Fresh-process raw bundle replay failed: "
            f"{completed.stderr[-3000:] or completed.stdout[-3000:]}"
        )
    if not Path(verification_path).is_file():
        raise RuntimeError("Fresh-process bundle replay did not write its verification result")
    return json.loads(Path(verification_path).read_text(encoding="utf-8"))


def run_training_pipeline(*, profile_name="nightly_v1", profile_override=None):
    global SEED
    import contextlib
    import importlib
    import optuna
    from utils.data import (
        load_modeling_dataset_settings,
        resolve_modeling_dataset_output_paths,
    )
    from utils.project_config import build_indicator_fit_config

    config_path = ROOT / "configs/btc_preopen_training.json"
    training_config = json.loads(config_path.read_text(encoding="utf-8"))
    if profile_override is None:
        profile = training_config["profiles"][profile_name]
    else:
        profile = dict(profile_override)
    profile = dict(profile)
    profile.setdefault("seed", SEED)
    contract_path = ROOT / profile["contract_path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    split = contract["chronological_split_frozen_before_tuning"]
    raw_path = Path(profile.get("raw_data_path", RAW_CANDLES_PATH))
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    if not raw_path.is_file():
        raise FileNotFoundError(f"BTC raw candle input not found: {raw_path}")
    raw_identity = _file_identity(raw_path)
    contract_identity = _file_identity(contract_path)
    task_identity = _signature(
        {
            "contract": contract,
            "raw_data_sha256": raw_identity["sha256"],
            "dataset_profile": profile["dataset_profile"],
            "modeling_profile": profile["modeling_profile"],
            "indicator_fit_profile": profile["indicator_fit_profile"],
        }
    )
    base_output = ROOT / profile["output_dir"]
    run_root = base_output / task_identity[:16]
    run_root.mkdir(parents=True, exist_ok=True)
    lock = _lock_run(run_root)
    manifest_path = run_root / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"schema_version": 1, "run_id": task_identity, "stages": {}}
    )
    log_path = run_root / "run.log"
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    tee = _Tee(original_stdout, log_handle)
    sys.stdout = tee
    sys.stderr = tee
    started = time.perf_counter()
    runtime_budget = int(profile["runtime_budget_seconds"])
    overall_deadline = started + runtime_budget
    try:
        global DATA_DIR, MODEL_DIR, FIT_END, CALIBRATION_START, CALIBRATION_END
        global TEST_FIRST_DECISION, TEST_MARKET_START, TEST_LAST_MARKET_START
        global PREDICTION_COMPUTE_BUDGET_SECONDS, FEATURE_CACHE_PATH
        global FEATURE_CACHE_META_PATH, RUN_MANIFEST_PATH, ITERATION_CHECKPOINT_PATH
        global MODEL_CHECKPOINT_PATH, METRICS_PATH, PREDICTIONS_PATH, MODEL_PATH
        global CALIBRATOR_PATH, MODEL_META_PATH, FEATURE_SOURCE_PATH, NUM_THREADS

        DATA_DIR = run_root
        MODEL_DIR = run_root / "prediction_bundle"
        FIT_END = pd.Timestamp(split["fit_and_tuning_end_exclusive_utc"])
        CALIBRATION_START = pd.Timestamp(split["calibration_start_utc"])
        CALIBRATION_END = pd.Timestamp(split["calibration_end_exclusive_utc"])
        TEST_FIRST_DECISION = pd.Timestamp(split["first_external_test_nominal_decision_utc"])
        TEST_MARKET_START = pd.Timestamp(split["first_external_test_market_start_utc"])
        TEST_LAST_MARKET_START = pd.Timestamp(
            split["external_test_market_start_last_inclusive_utc"]
        )
        PREDICTION_COMPUTE_BUDGET_SECONDS = float(
            profile["prediction_compute_budget_seconds"]
        )
        NUM_THREADS = int(profile["threads"])
        SEED = int(profile["seed"])
        FEATURE_CACHE_PATH = run_root / "causal_minute_features.parquet"
        FEATURE_CACHE_META_PATH = run_root / "causal_minute_features.manifest.json"
        RUN_MANIFEST_PATH = manifest_path
        ITERATION_CHECKPOINT_PATH = run_root / "iteration_selection.json"
        MODEL_CHECKPOINT_PATH = MODEL_DIR / "training.manifest.json"
        METRICS_PATH = run_root / "evaluation.json"
        PREDICTIONS_PATH = run_root / "external_test_predictions.parquet"
        MODEL_PATH = MODEL_DIR / "lgbm_model.txt"
        CALIBRATOR_PATH = MODEL_DIR / "platt_calibrator.json"
        MODEL_META_PATH = MODEL_DIR / "lgbm_meta.json"
        FEATURE_SOURCE_PATH = MODEL_DIR / "feature_sources.json"

        compute_backend = str(profile.get("compute_backend", "gpu")).strip().lower()
        if compute_backend not in {"cpu", "gpu"}:
            raise ValueError("profile.compute_backend must be 'cpu' or 'gpu'.")
        _configure_training_backend(NUM_THREADS, compute_backend)
        effective = {
            "profile_name": profile_name,
            "profile": profile,
            "contract": contract,
            "input_data": raw_identity,
            "contract_file": contract_identity,
            "task_identity": task_identity,
            "compute_backend": f"LightGBM {compute_backend.upper()}",
            "threads": NUM_THREADS,
        }
        _write_json(run_root / "effective_config.json", effective)
        manifest.update(
            {
                "schema_version": 1,
                "run_id": task_identity,
                "run_kind": profile_name,
                "status": "running",
                "effective_config_path": (run_root / "effective_config.json").resolve().relative_to(ROOT).as_posix(),
                "log_path": log_path.resolve().relative_to(ROOT).as_posix(),
                "input_data_sha256": raw_identity["sha256"],
                "contract_sha256": contract_identity["sha256"],
                "started_utc": manifest.get("started_utc", pd.Timestamp.now(tz="UTC").isoformat()),
                "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }
        )
        _write_json(manifest_path, manifest)
        print(
            f"[preopen] run={profile_name} rows_source={raw_path.name} "
            f"threads={NUM_THREADS} total_budget={runtime_budget}s output={run_root}",
            flush=True,
        )

        dataset_settings = load_modeling_dataset_settings(
            asset="BTC",
            dataset_profile_name=profile["dataset_profile"],
            modeling_profile_name=profile["modeling_profile"],
        )
        dataset_settings["raw_data_dir"] = raw_path.parent
        dataset_settings["base_data_file"] = raw_path.name
        dataset_settings["feature_subset_path"] = None
        dataset_settings["feature_subset_list_key"] = None
        dataset_settings["excluded_feature_names"] = []
        dataset_settings["preopen_target"] = {
            "prediction_compute_budget_seconds": PREDICTION_COMPUTE_BUDGET_SECONDS
        }

        stages = profile.get("stage_runtime_budgets_seconds", {})

        def deadline_for(stage_name):
            return min(
                overall_deadline,
                time.perf_counter() + float(stages.get(stage_name, runtime_budget)),
            )

        indicator_cfg = build_indicator_fit_config(
            asset="BTC",
            dataset_profile_name=profile["dataset_profile"],
            indicator_fit_profile_name=profile["indicator_fit_profile"],
        )
        indicator_cfg["run_budget"] = {
            "population_size": int(profile["indicator_fit"]["population_size"]),
            "seed": int(profile["seed"]),
            "minimum_generations": int(
                profile["indicator_fit"].get(
                    "minimum_generations", 1
                )
            ),
        }
        for pair_name, pair_cfg in indicator_cfg["pairs"].items():
            pair_cfg["proxy_target_mode"] = "preopen_v1"
            pair_cfg["proxy_target_horizonts"] = [5]
            interval_cfg = next(iter(pair_cfg["intervals"].values()))
            interval_cfg["data_path"] = raw_path.parent.as_posix()
            interval_cfg["data_file"] = raw_path.name
            interval_cfg["proxy_target_mode"] = "preopen_v1"
            interval_cfg["proxy_target_horizonts"] = [5]
            interval_cfg["prediction_compute_budget_seconds"] = (
                PREDICTION_COMPUTE_BUDGET_SECONDS
            )
            interval_cfg["target_fit_start_utc"] = split["feature_and_model_fit_start_utc"]
            interval_cfg["target_label_available_through_utc"] = split[
                "fit_and_tuning_end_exclusive_utc"
            ]
            pair_cfg["metric_gap"] = int(profile.get("indicator_metric_gap", pair_cfg["metric_gap"]))
            pair_cfg["metric_segments_count"] = int(
                profile.get("indicator_metric_segments_count", pair_cfg["metric_segments_count"])
            )
            pair_cfg["min_bucket_size"] = int(
                profile.get("indicator_metric_min_bucket_size", pair_cfg["min_bucket_size"])
            )
            pair_cfg["min_valid_segments"] = int(
                profile.get("indicator_metric_min_valid_segments", pair_cfg["min_valid_segments"])
            )

        indicator_code_paths = [
            ROOT / path for path in (
                "fit_indicators.py",
                "features/ADX.py",
                "features/BollingerBands.py",
                "features/ChaikinOsc.py",
                "features/KeltnerChannel.py",
                "features/MACD.py",
                "features/StochOsc.py",
                "features/common_utils.py",
                "features/btc_preopen_contract.py",
                "utils/metrics.py",
                "utils/data.py",
                "utils/project_config.py",
            )
        ]
        indicator_input = {
            "effective_fit_config": indicator_cfg,
            "generation_budget": int(profile["indicator_fit"]["generations"]),
            "worker_count": int(profile["indicator_fit"]["workers"]),
            "data_sha256": raw_identity["sha256"],
            "contract_sha256": contract_identity["sha256"],
            "code": _code_fingerprints([*indicator_code_paths, Path(__file__)]),
        }

        indicator_stage_deadline = deadline_for("indicators")

        def fit_indicators_action(stage_dir):
            import fit_indicators

            results_root = stage_dir / "results"
            cfg = json.loads(json.dumps(indicator_cfg))
            try:
                fit_indicators.main(
                    config=cfg,
                    results_root=results_root,
                    generation_budget=int(profile["indicator_fit"]["generations"]),
                    deadline_monotonic=indicator_stage_deadline,
                    worker_count=int(profile["indicator_fit"]["workers"]),
                    write_applied_config=False,
                )
            except TimeoutError:
                print(
                    "[preopen] indicators stopped between fits/generations; "
                    "checking whether every configured fit met its minimum",
                    flush=True,
                )
            results = list(results_root.rglob("*.json"))
            fit_jsons = [
                path for path in results
                if path.name not in {"fit_indicators_applied_config.json", "fit_indicators_config.json"}
            ]
            required_generations = int(indicator_cfg["run_budget"]["minimum_generations"])
            incomplete = [
                path for path in fit_jsons
                if int(json.loads(path.read_text(encoding="utf-8"))["best"]["generations_completed"])
                < required_generations
            ]
            if incomplete:
                raise TimeoutError(
                    f"Indicator stage has {len(incomplete)} fit(s) below the minimum "
                    f"{required_generations} completed generation(s); "
                    "resume the run to complete them."
                )
            parsed = importlib.import_module("create_modeling_dataset").parse_fit_results(results_root)
            expected = sum(
                len(pair_cfg["intervals"]) * len(pair_cfg["intervals"][next(iter(pair_cfg["intervals"]))]["indicators"])
                * len(pair_cfg["proxy_target_horizonts"]) * len(pair_cfg["quantile_pairs"])
                for pair_cfg in cfg["pairs"].values()
            )
            if len(parsed) != expected:
                raise RuntimeError(f"Indicator fits produced {len(parsed)} parsed configs; expected {expected}")
            result_file = stage_dir / "indicator_stage.json"
            _write_json(
                result_file,
                {
                    "config_count": len(parsed),
                    "fit_results_dir": (fit_jsons[0].parent if fit_jsons else results_root).resolve().relative_to(ROOT).as_posix(),
                    "result_files": [path.resolve().relative_to(ROOT).as_posix() for path in fit_jsons],
                    "target_mode": "preopen_v1",
                    "label_available_before_fit_end": True,
                },
            )
            return {
                "config_count": len(parsed),
                "fit_results_dir": (fit_jsons[0].parent if fit_jsons else results_root).resolve().relative_to(ROOT).as_posix(),
            }, [result_file, *fit_jsons]

        indicator_result, indicator_sig, indicator_outputs = _run_stage(
            manifest,
            manifest_path,
            run_root,
            "indicators",
            indicator_input,
            fit_indicators_action,
            deadline_monotonic=indicator_stage_deadline,
        )
        indicator_results_dir = ROOT / indicator_result["fit_results_dir"]
        if not indicator_results_dir.is_dir():
            raise RuntimeError(f"Indicator fit results directory is missing: {indicator_results_dir}")

        rp_module = importlib.import_module("fit_reaction_profile")
        vp_module = importlib.import_module("fit_volume_profile")
        reaction_base = rp_module.normalize_reaction_profile_config(None)
        volume_base = vp_module.normalize_volume_profile_config(None)
        rp_data = {
            "open_np": None,
            "high_np": None,
            "low_np": None,
            "close_np": None,
            "keep_mask": None,
            "y_filtered": None,
            "sample_weight_filtered": None,
        }
        volume_data = {
            "high_np": None,
            "low_np": None,
            "volume_np": None,
            "keep_mask": None,
            "y_filtered": None,
            "sample_weight_filtered": None,
        }

        # Build all stateful features from the fitted generator configs. The
        # generator objectives below receive only development decision labels.
        dataset_stage_payload = {
            "indicator_artifact_signature": indicator_sig,
            "reaction_base": reaction_base,
            "volume_base": volume_base,
            "dataset_settings": {
                key: value for key, value in dataset_settings.items()
                if key not in {"raw_data_dir", "modeling_output_dir", "fit_results_dir"}
            },
            "prediction_compute_budget_seconds": PREDICTION_COMPUTE_BUDGET_SECONDS,
        "code": _code_fingerprints([
            ROOT / "create_modeling_dataset.py",
            ROOT / "features/btc_preopen_contract.py",
            ROOT / "features/reaction_profile_fixed_grid.py",
            ROOT / "features/volume_profile_fixed_range.py",
            ROOT / "features/ADX.py",
            ROOT / "features/BollingerBands.py",
            ROOT / "features/ChaikinOsc.py",
            ROOT / "features/KeltnerChannel.py",
            ROOT / "features/MACD.py",
            ROOT / "features/StochOsc.py",
            ROOT / "features/common_utils.py",
            ROOT / "utils/metrics.py",
            ROOT / "utils/data.py",
            ROOT / "utils/project_config.py",
            ROOT / "fit_indicators.py",
            ROOT / "train_lgbm.py",
            Path(__file__),
        ]),
        }
        dataset_stage_signature = _signature(dataset_stage_payload)
        dataset_dir = run_root / "datasets" / dataset_stage_signature[:16]
        dataset_settings["fit_results_dir"] = indicator_results_dir
        dataset_settings["modeling_output_dir"] = dataset_dir
        dataset_settings["output_suffix"] = "_preopen_v1"
        dataset_settings["profile_state_dir"] = dataset_dir / "states"
        dataset_settings["reaction_profile_fixed_grid"] = reaction_base
        dataset_settings["volume_profile_fixed_range"] = volume_base

        def build_feature_dataset_action(stage_dir):
            import create_modeling_dataset

            dataset_settings["modeling_output_dir"] = stage_dir / "dataset"
            dataset_settings["profile_state_dir"] = stage_dir / "states"
            output_path = create_modeling_dataset.build_dataset_from_settings(dataset_settings)
            metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
            if not metadata_path.is_file():
                raise RuntimeError("Modeling dataset metadata file is missing")
            result_path = stage_dir / "feature_dataset_stage.json"
            _write_json(
                result_path,
                {
                    "dataset_path": output_path.resolve().relative_to(ROOT).as_posix(),
                    "metadata_path": metadata_path.resolve().relative_to(ROOT).as_posix(),
                    "rows": int(pd.read_parquet(output_path, columns=["Opened"]).shape[0]),
                    "feature_families": ["indicators", "reaction_profile", "volume_profile", "candle", "session", "realized_volatility", "basis_premium"],
                },
            )
            state_files = list((stage_dir / "states").rglob("*.npz"))
            if len(state_files) != 2:
                raise RuntimeError(
                    f"Expected both generator state files, found {len(state_files)} "
                    f"in {stage_dir / 'states'}"
                )
            return {
                "dataset_path": output_path.resolve().relative_to(ROOT).as_posix(),
                "metadata_path": metadata_path.resolve().relative_to(ROOT).as_posix(),
                "profile_state_dir": (stage_dir / "states").resolve().relative_to(ROOT).as_posix(),
                "rows": int(pd.read_parquet(output_path, columns=["Opened"]).shape[0]),
            }, [output_path, metadata_path, result_path, *state_files]

        raw_values = pd.read_csv(
            raw_path,
            usecols=["Opened", "Open", "High", "Low", "Close", "Volume"],
        )
        raw_values["Opened"] = pd.to_datetime(raw_values["Opened"], utc=True, errors="raise")
        contract_full = build_preopen_contract_frame(
            raw_values,
            prediction_compute_budget_seconds=PREDICTION_COMPUTE_BUDGET_SECONDS,
        )
        contract_cols = [
            "Opened", TARGET_COL, TARGET_AVAILABLE_COL, DECISION_COL,
            "feature_candle_close_at", "nominal_decision_at",
            "nominal_prediction_available_at", "actual_prediction_available_at",
            "prediction_deadline_at", "target_window_start_at",
            "target_window_end_at", TARGET_START_PRICE_COL, TARGET_END_PRICE_COL,
        ]
        contract_frame = contract_full.loc[:, [column for column in contract_cols if column in contract_full]].copy()
        contract_frame["market_start_utc"] = contract_frame["target_window_start_at"]
        del contract_full
        computed_decision = preopen_decision_mask(contract_frame)
        stored_decision = contract_frame[DECISION_COL].to_numpy(dtype=np.bool_, copy=False)
        if not np.array_equal(computed_decision, stored_decision):
            raise RuntimeError("Stored decision mask differs from point-in-time contract readiness")
        decision_mask = computed_decision
        opened = pd.DatetimeIndex(pd.to_datetime(contract_frame["Opened"], utc=True, errors="raise"))
        label_available = pd.DatetimeIndex(pd.to_datetime(contract_frame[TARGET_AVAILABLE_COL], utc=True, errors="coerce"))
        decision_at = pd.DatetimeIndex(pd.to_datetime(contract_frame["nominal_decision_at"], utc=True, errors="raise"))
        y_full = pd.to_numeric(contract_frame[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        valid_labels = np.isfinite(y_full) & label_available.notna()
        fit_start = pd.Timestamp(split["feature_and_model_fit_start_utc"])
        fit_mask = (
            valid_labels
            & (opened >= fit_start)
            & (label_available < FIT_END)
        )
        if int(fit_mask.sum()) < 100:
            raise RuntimeError(f"Only {int(fit_mask.sum())} rows remain in the allowed fitting interval")
        frame = contract_frame.copy()
        frame["market_start_utc"] = pd.to_datetime(frame["target_window_start_at"], utc=True)
        frame[TARGET_AVAILABLE_COL] = label_available
        frame["nominal_decision_at"] = decision_at
        y = y_full.copy()
        target_weight = build_preopen_target_weights(
            opened,
            decision_weight=0.4625,
            auxiliary_total_weight=0.5375,
            decision_mask=decision_mask,
        )

        # A profile only contains the task-specific overrides; generator search
        # spaces stay in their original fitter modules.
        def make_generator_arrays(decision_only=True):
            target_rows = fit_mask & (decision_mask if decision_only else True)
            if np.unique(y[target_rows]).size < 2:
                raise RuntimeError("Generator fit rows do not contain both target classes")
            return target_rows, y[target_rows], target_weight[target_rows].astype(np.float32, copy=False)

        rp_keep, rp_y, rp_weights = make_generator_arrays()
        vp_keep, vp_y, vp_weights = make_generator_arrays()
        price_arrays = {
            name.lower() + "_np": pd.to_numeric(raw_values[name], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            for name in ("Open", "High", "Low", "Close", "Volume")
        }
        rp_data.update({key: price_arrays[key] for key in ("open_np", "high_np", "low_np", "close_np")})
        rp_data.update({"keep_mask": rp_keep, "y_filtered": rp_y, "sample_weight_filtered": rp_weights})
        volume_data.update({key: price_arrays[key] for key in ("high_np", "low_np", "volume_np")})
        volume_data.update({"keep_mask": vp_keep, "y_filtered": vp_y, "sample_weight_filtered": vp_weights})

        reaction, volume, model_tuning, weight_search, selector, train_lgbm = _configure_training_backend(
            NUM_THREADS, compute_backend
        )
        cv_folds = int(profile.get("cv_folds", 10))
        model_params = {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 128,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 5.0,
            "lambda_l1": 0.0,
            "verbosity": -1,
            "device_type": compute_backend,
            "n_jobs": NUM_THREADS,
            "random_state": int(profile["seed"]),
        }

        def run_generator_stage(stage_name, module, base_config, base_data, search_space, trial_budget):
            rows = np.flatnonzero(base_data["keep_mask"])
            label_times = decision_at[rows]
            folds = _purged_walk_forward_folds(
                module,
                opened[rows],
                label_times,
                int(profile.get("generator_cv_folds", cv_folds)),
                0.1,
            )
            fold_indices = module.build_fold_indices(folds)
            fold_weights = module.build_fold_recency_weights(folds)
            signature_payload = {
                "stage": stage_name,
                "target": contract["training_target"],
                "fit_start": fit_start.isoformat(),
                "fit_end_exclusive": FIT_END.isoformat(),
                "data_sha256": raw_identity["sha256"],
                "base_config": base_config,
                "search_space": search_space,
                "folds": [(f["train_start"], f["train_end"], f["test_start"], f["test_end"]) for f in folds],
                "fold_indices": _fold_index_fingerprints(folds),
                "minimum_successful_trials": int(profile["minimum_successful_trials"]),
                "initial_weights": {"decision": 0.4625, "auxiliary_total": 0.5375},
                "seed": int(profile["seed"]),
                "target_contract_code_sha256": _sha256(ROOT / "features/btc_preopen_contract.py"),
                "compute_backend": compute_backend,
                "code": _code_fingerprints([
                    Path(module.__file__),
                    ROOT / (
                        "features/reaction_profile_fixed_grid.py"
                        if stage_name == "reaction_profile"
                        else "features/volume_profile_fixed_range.py"
                    ),
                    ROOT / "features/common_utils.py",
                    ROOT / "features/btc_preopen_contract.py",
                    ROOT / "utils/metrics.py",
                    ROOT / "utils/data.py",
                    ROOT / "train_lgbm.py",
                    Path(__file__),
                ]),
            }
            stage_deadline = deadline_for(stage_name)
            def action(stage_dir):
                objective = module.make_objective(
                    base_data=base_data,
                    folds=folds,
                    fold_indices=fold_indices,
                    fold_weight_by_id=fold_weights,
                    **({"base_rp_config": base_config} if stage_name == "reaction_profile" else {"base_vp_config": base_config}),
                    search_space=search_space,
                )
                payload, paths = _run_optuna_stage(
                    optuna=optuna,
                    stage_name=stage_name,
                    stage_dir=stage_dir,
                    objective=objective,
                    target_trials=int(trial_budget),
                    minimum_successful_trials=int(profile["minimum_successful_trials"]),
                    stage_deadline=stage_deadline,
                    overall_deadline=overall_deadline,
                    seed=int(profile["seed"]),
                    catch=(lgb.basic.LightGBMError, OSError),
                )
                if not isinstance(payload.get("best_config"), dict):
                    raise RuntimeError(f"{stage_name} best trial did not retain its normalized generator config")
                config_path = stage_dir / "fitted_generator_config.json"
                _write_json(config_path, payload["best_config"])
                payload["config_path"] = config_path.resolve().relative_to(ROOT).as_posix()
                _write_json(paths[1], payload)
                return payload, [*paths, config_path]
            minimum_successful = int(profile["minimum_successful_trials"])
            return _run_stage(
                manifest, manifest_path, run_root, stage_name, signature_payload, action,
                must_run=lambda result: (
                    int(result.get("target_trials", 0)) < int(trial_budget)
                    or int(result.get("minimum_successful_trials", 0)) < minimum_successful
                    or int(result.get("successful_trials", 0)) < minimum_successful
                ),
                deadline_monotonic=stage_deadline,
            )

        rp_result, rp_sig, rp_outputs = run_generator_stage(
            "reaction_profile", reaction, reaction_base, rp_data,
            reaction.REACTION_PROFILE_OPTUNA_SEARCH_SPACE,
            int(profile["reaction_profile_trials"]),
        )
        vp_result, vp_sig, vp_outputs = run_generator_stage(
            "volume_profile", volume, volume_base, volume_data,
            volume.VOLUME_PROFILE_OPTUNA_SEARCH_SPACE,
            int(profile["volume_profile_trials"]),
        )
        for stage_result, config_key, config_label in (
            (rp_result, "reaction_profile_fixed_grid", "reaction profile"),
            (vp_result, "volume_profile_fixed_range", "volume profile"),
        ):
            config_path = ROOT / stage_result.get("config_path", "")
            if not config_path.is_file():
                raise RuntimeError(f"Completed {config_label} config is missing: {config_path}")
            persisted_config = json.loads(config_path.read_text(encoding="utf-8"))
            if persisted_config != stage_result.get("best_config"):
                raise RuntimeError(f"Persisted {config_label} config differs from its stage result")
            dataset_settings[config_key] = persisted_config
        reaction_config = dataset_settings["reaction_profile_fixed_grid"]
        volume_config = dataset_settings["volume_profile_fixed_range"]
        dataset_stage_payload.update(
            {"reaction_artifact_signature": rp_sig, "volume_artifact_signature": vp_sig}
        )
        dataset_stage_payload["dataset_settings"] = {
            key: value for key, value in dataset_settings.items()
            if key not in {"raw_data_dir", "modeling_output_dir", "fit_results_dir"}
        }
        dataset_stage_signature = _signature(dataset_stage_payload)
        dataset_dir = run_root / "datasets" / dataset_stage_signature[:16]
        dataset_result, dataset_sig, dataset_outputs = _run_stage(
            manifest,
            manifest_path,
            run_root,
            "final_feature_dataset",
            dataset_stage_payload,
            build_feature_dataset_action,
            deadline_monotonic=overall_deadline,
        )
        dataset_path = ROOT / dataset_result["dataset_path"]
        dataset_metadata_path = ROOT / dataset_result["metadata_path"]
        dataset_settings["profile_state_dir"] = ROOT / dataset_result["profile_state_dir"]
        if not dataset_path.is_file() or not dataset_metadata_path.is_file():
            raise RuntimeError("Completed feature dataset or its metadata is missing")
        if not dataset_settings["profile_state_dir"].is_dir():
            raise RuntimeError("Completed feature dataset generator state directory is missing")
        if int(dataset_result.get("rows", 0)) < 1:
            raise RuntimeError("Completed feature dataset has no rows")
        feature_dataset = pd.read_parquet(dataset_path)
        contract_frame = feature_dataset.loc[:, [column for column in contract_cols if column in feature_dataset]].copy()
        contract_frame["market_start_utc"] = pd.to_datetime(contract_frame["target_window_start_at"], utc=True)
        computed_decision = preopen_decision_mask(contract_frame)
        if not np.array_equal(computed_decision, contract_frame[DECISION_COL].to_numpy(dtype=np.bool_, copy=False)):
            raise RuntimeError("Final generator feature build changed the pre-open decision mask")
        opened = pd.DatetimeIndex(pd.to_datetime(contract_frame["Opened"], utc=True, errors="raise"))
        label_available = pd.DatetimeIndex(pd.to_datetime(contract_frame[TARGET_AVAILABLE_COL], utc=True, errors="coerce"))
        decision_at = pd.DatetimeIndex(pd.to_datetime(contract_frame["nominal_decision_at"], utc=True, errors="raise"))
        decision_mask = computed_decision
        y_full = pd.to_numeric(contract_frame[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        valid_labels = np.isfinite(y_full) & label_available.notna()
        fit_mask = valid_labels & (opened >= fit_start) & (label_available < FIT_END)
        frame = contract_frame.copy()
        frame["market_start_utc"] = pd.to_datetime(frame["target_window_start_at"], utc=True)
        frame[TARGET_AVAILABLE_COL] = label_available
        frame["nominal_decision_at"] = decision_at
        y = y_full.copy()
        features = _model_feature_frame(feature_dataset)
        feature_names = list(features.columns)
        del feature_dataset
        development_rows = np.flatnonzero(fit_mask)
        development_features = features.iloc[development_rows].reset_index(drop=True)
        development_y = pd.Series(y[development_rows], dtype=np.int8)
        development_decision = decision_mask[development_rows]
        development_opened = opened[development_rows]
        development_decision_at = decision_at[development_rows]

        # Refit no generators after weight selection. The fixed matrix above is
        # used by the weight, feature, and model stages.
        tw_x = development_features.reset_index(drop=True)
        tw_y = development_y.reset_index(drop=True)
        tw_opened = development_opened
        weight_search.CV_FOLDS = cv_folds
        weight_search.SEARCH_CV_FOLDS = min(cv_folds, int(profile.get("weight_cv_folds", cv_folds)))
        weight_search.TEST_TO_TRAIN_RATIO = 0.1
        weight_search.SEARCH_REFINEMENT_ROUNDS = 0
        weight_search.SEARCH_N_ESTIMATORS = int(profile.get("weight_search_estimators", 100))
        weight_search.SEARCH_EARLY_STOPPING_ROUNDS = int(profile.get("early_stopping_rounds", 50))
        weight_search.SEED = int(profile["seed"])
        weight_search.OBJECTIVE_STD_PENALTY = 1.0
        weight_search.DECISION_WEIGHT_LOW = 0.05
        weight_search.DECISION_WEIGHT_HIGH = 0.95
        weight_search.INITIAL_WEIGHT_GRID = tuple(
            np.linspace(0.05, 0.95, int(profile["target_weight_candidates"]))
        )
        weight_search.TARGET_WEIGHT_DECISION_VALUE = 0.4625
        weight_search.DEVICE_TYPE = compute_backend
        weight_search.LGBM_N_JOBS = NUM_THREADS
        weight_folds = _purged_walk_forward_folds(
            weight_search, tw_opened, development_decision_at,
            int(profile.get("weight_cv_folds", cv_folds)), 0.1,
            validation_mask=development_decision,
        )
        weight_fold_weights = weight_search.build_fold_recency_weights(weight_folds)
        weight_params = dict(model_params)
        weight_params.pop("random_state", None)
        weight_params["n_jobs"] = NUM_THREADS
        weight_payload = {
            "data_sha256": raw_identity["sha256"],
            "dataset_signature": dataset_sig,
            "target": contract["training_target"],
            "fit_start": fit_start.isoformat(),
            "fit_end_exclusive": FIT_END.isoformat(),
            "candidate_count": int(profile["target_weight_candidates"]),
            "minimum_candidates": min(3, int(profile["target_weight_candidates"])),
            "training_rows": int(len(tw_y)),
            "decision_rows": int(development_decision.sum()),
            "decision_mask_sha256": _array_sha256(development_decision, dtype=np.bool_),
            "cv_folds": int(profile.get("weight_cv_folds", cv_folds)),
            "folds": [
                (fold["train_start"], fold["train_end"], fold["test_start"], fold["test_end"])
                for fold in weight_folds
            ],
            "fold_indices": _fold_index_fingerprints(weight_folds),
            "seed": int(weight_search.SEED),
            "compute_backend": compute_backend,
            "search_code_sha256": _sha256(ROOT / "optimize_target_weights.py"),
            "code": _code_fingerprints([
                ROOT / "optimize_target_weights.py",
                ROOT / "features/btc_preopen_contract.py",
                ROOT / "utils/metrics.py",
                ROOT / "train_lgbm.py",
                Path(__file__),
            ]),
            "observation_weights": "decision_weight; auxiliary=(1-decision_weight)/4",
            "validation": "actual decision rows; unweighted",
            "model_params": weight_params,
            "decision_mask_code_sha256": _sha256(ROOT / "features/btc_preopen_contract.py"),
        }

        weight_stage_deadline = deadline_for("target_weights")

        def tune_weights_action(stage_dir):
            result = weight_search.run_proxy_weight_search_for_subset(
                feature_subset_candidate={
                    "id": "all_features",
                    "label": "all features",
                    "path": dataset_path.resolve().relative_to(ROOT).as_posix(),
                    "feature_count": int(tw_x.shape[1]),
                    "is_active": True,
                },
                x=tw_x,
                y=tw_y,
                decision_mask=development_decision,
                folds=weight_folds,
                fold_weight_by_id=weight_fold_weights,
                param_overrides=weight_params,
                float_dtype=np.float32,
                deadline_monotonic=weight_stage_deadline,
                checkpoint_path=stage_dir / "candidate_metrics.parquet",
                minimum_candidates=min(3, int(profile["target_weight_candidates"])),
            )
            candidates = [row for row in result["search_rows"] if row.get("decision_weight") is not None]
            if len(candidates) < min(3, int(profile["target_weight_candidates"])):
                raise RuntimeError(f"Weight stage has only {len(candidates)} valid candidates")
            best = max(candidates, key=lambda row: float(row["objective_value"]))
            payload = {
                "objective": weight_search.resolve_decision_objective_name(),
                "metric_scope": "unweighted actual decision rows only",
                "decision_weight": float(best["decision_weight"]),
                "auxiliary_row_weight": float(weight_search.build_weight_config(best["decision_weight"])["other_weight"]),
                "objective_value": float(best["objective_value"]),
                "candidates_evaluated": len(candidates),
                "target_candidates": int(profile["target_weight_candidates"]),
                "minimum_candidates": min(3, int(profile["target_weight_candidates"])),
                "stopped_by": result["stopped_by"],
                "rows": int(len(tw_y)),
                "decision_rows": int(development_decision.sum()),
                "baseline_oof_metrics": {
                    "balanced_accuracy": (result.get("baseline_row") or {}).get(
                        "decision_rows_oof_balanced_accuracy"
                    ),
                    "binary_logloss": (result.get("baseline_row") or {}).get(
                        "decision_rows_oof_logloss"
                    ),
                    "brier_score": (result.get("baseline_row") or {}).get(
                        "decision_rows_oof_brier"
                    ),
                },
            }
            result_path = stage_dir / "target_weight_result.json"
            _write_json(result_path, payload)
            return payload, [result_path, stage_dir / "candidate_metrics.parquet", stage_dir / "candidate_fold_metrics.parquet"]

        weight_result, weight_sig, weight_outputs = _run_stage(
            manifest, manifest_path, run_root, "target_weights", weight_payload,
            tune_weights_action,
            must_run=lambda result: (
                int(result.get("target_candidates", 0)) < int(profile["target_weight_candidates"])
                or int(result.get("minimum_candidates", 0)) < min(3, int(profile["target_weight_candidates"]))
                or int(result.get("candidates_evaluated", 0)) < min(3, int(profile["target_weight_candidates"]))
            ),
            deadline_monotonic=weight_stage_deadline,
        )

        # Training uses every eligible development minute with the selected
        # observation weights; all selector validation remains decision-only.
        decision_weight = float(weight_result["decision_weight"])
        development_weights = _preopen_sample_weights(development_decision, decision_weight)
        selector_x = development_features.reset_index(drop=True)
        selector_y = development_y.reset_index(drop=True)
        selector_opened = development_opened
        selector_decision_at = development_decision_at
        selector_train_weights = pd.Series(development_weights, dtype=np.float32)
        selector.EXCLUDE_COLS = []
        selector.RANKING_N_SPLITS = min(cv_folds, int(profile.get("selector_cv_folds", cv_folds)))
        selector.PERMUTATION_N_SPLITS = selector.RANKING_N_SPLITS
        selector.TOPK_N_SPLITS = selector.RANKING_N_SPLITS
        selector.WF_TEST_TO_TRAIN_RATIO = 0.1
        selector.N_ESTIMATORS = int(profile.get("selector_estimators", 100))
        selector.EARLY_STOPPING_ROUNDS = int(profile.get("early_stopping_rounds", 50))
        selector.MAX_SWEEP_EVALUATIONS = int(profile["feature_selection"]["max_sweep_evaluations"])
        selector.MAX_REFINEMENT_ROUNDS = int(profile["feature_selection"].get("max_refinement_rounds", 3))
        selector.PERMUTATION_FEATURE_FRACTION = float(profile["feature_selection"]["permutation_feature_fraction"])
        selector.PERMUTATION_N_REPEATS = int(profile["feature_selection"]["permutation_repeats"])
        selector.MIN_NONZERO_IMPORTANCE_FOLDS = min(6, selector.RANKING_N_SPLITS)
        selection_folds = _purged_walk_forward_folds(
            selector, selector_opened, selector_decision_at,
            selector.RANKING_N_SPLITS, 0.1,
            validation_mask=development_decision,
        )
        selector_fold_weights = selector.build_fold_recency_weights(selection_folds)
        selection_payload = {
            "dataset_signature": dataset_sig,
            "weight_artifact_signature": weight_sig,
            "feature_count": int(selector_x.shape[1]),
            "training_rows": int(len(selector_y)),
            "validation_rows": int(development_decision.sum()),
            "training_weight_sha256": _array_sha256(development_weights, dtype=np.float32),
            "weight_scheme": {
                "decision_weight": decision_weight,
                "auxiliary_weight": float((1.0 - decision_weight) / 4.0),
            },
            "decision_mask_sha256": _array_sha256(development_decision, dtype=np.bool_),
            "fold_indices": _fold_index_fingerprints(selection_folds),
            "folds": selector.RANKING_N_SPLITS,
            "seed": [int(value) for value in selector.RANDOM_SEEDS],
            "compute_backend": compute_backend,
            "settings": profile["feature_selection"],
            "fit_settings": {
                "n_estimators": int(selector.N_ESTIMATORS),
                "early_stopping_rounds": int(selector.EARLY_STOPPING_ROUNDS),
                "model_params": dict(selector.MODEL_PARAMS),
            },
            "code": _code_fingerprints([
                ROOT / "select_features.py",
                ROOT / "features/btc_preopen_contract.py",
                ROOT / "utils/metrics.py",
                ROOT / "train_lgbm.py",
                Path(__file__),
            ]),
            "metric": "binary_logloss on unweighted actual decision rows",
        }
        selection_deadline = deadline_for("feature_selection")

        def select_features_action(stage_dir):
            start = time.perf_counter()
            x_prefilter, filter_report, duplicate_map, corr_map = selector.prefilter_features(
                selector_x.replace([np.inf, -np.inf], np.nan),
            )
            rank_path = stage_dir / "feature_ranking.parquet"
            topk_path = stage_dir / "topk_sweep.parquet"
            prescreen_folds = selection_folds
            permutation_folds = selection_folds
            topk_folds = selection_folds
            if rank_path.is_file():
                fold_ranking = pd.read_parquet(rank_path)
                if (
                    "feature" not in fold_ranking
                    or len(fold_ranking) != x_prefilter.shape[1]
                    or set(fold_ranking["feature"]) != set(x_prefilter.columns)
                ):
                    raise RuntimeError(f"Feature-ranking checkpoint is invalid: {rank_path}")
                print(f"[preopen] loading verified feature-ranking checkpoint: {rank_path}", flush=True)
            else:
                fold_ranking, _, fold_metadata = selector.run_feature_ranking(
                    x=x_prefilter,
                    y=selector_y,
                    sample_weight=selector_train_weights,
                    prescreen_folds=prescreen_folds,
                    prescreen_fold_weight_by_id=selector_fold_weights,
                    permutation_folds=permutation_folds,
                    permutation_fold_weight_by_id=selector_fold_weights,
                    unweighted_evaluation=True,
                    deadline_monotonic=selection_deadline,
                )
                temporary_rank_path = rank_path.with_name(rank_path.stem + ".tmp.parquet")
                fold_ranking.to_parquet(temporary_rank_path, index=False, compression="zstd")
                temporary_rank_path.replace(rank_path)
            topk = selector.run_topk_sweep(
                x=x_prefilter,
                y=selector_y,
                sample_weight=selector_train_weights,
                folds=topk_folds,
                fold_weight_by_id=selector_fold_weights,
                global_feature_order=fold_ranking["feature"].tolist(),
                unweighted_evaluation=True,
                checkpoint_path=topk_path,
                deadline_monotonic=selection_deadline,
            )
            best_row, recommended_row = selector.choose_recommended_row(topk)
            selected = fold_ranking["feature"].tolist()[: int(recommended_row["k"])]
            if not selected:
                raise RuntimeError("Feature selector returned an empty feature order")
            result_path = stage_dir / "selected_features.json"
            filter_report.to_parquet(stage_dir / "prefilter_report.parquet", index=False, compression="zstd")
            _write_json(
                result_path,
                {
                    "selected_features": selected,
                    "feature_count": len(selected),
                    "input_feature_count": int(selector_x.shape[1]),
                    "prefilter_feature_count": int(x_prefilter.shape[1]),
                    "recommended_k": int(recommended_row["k"]),
                    "best_k": int(best_row["k"]),
                    "recommendation_score": float(recommended_row["selection_score"]),
                    "ranking_method": "existing gain prescreen plus permutation reranking",
                    "validation_rows_are_actual_decisions": True,
                    "training_rows_include_auxiliary_minutes": True,
                    "training_rows": int(len(selector_y)),
                    "validation_rows": int(development_decision.sum()),
                    "permutation_features_scored": int(
                        fold_ranking["permutation_mean_delta_logloss"].notna().sum()
                    ),
                    "training_weight_scheme": selection_payload["weight_scheme"],
                    "topk_evaluations": int(topk.attrs["sweep_metadata"]["total_k_evaluations"]),
                    "minimum_topk_evaluations": int(topk.attrs["sweep_metadata"]["minimum_k_evaluations"]),
                    "stopped_by": topk.attrs["sweep_metadata"]["stopped_by"],
                    "elapsed_seconds": time.perf_counter() - start,
                },
            )
            return {
                "selected_features": selected,
                "feature_count": len(selected),
                "training_rows": int(len(selector_y)),
                "validation_rows": int(development_decision.sum()),
                "training_weight_sha256": _array_sha256(development_weights, dtype=np.float32),
                "permutation_features_scored": int(
                    fold_ranking["permutation_mean_delta_logloss"].notna().sum()
                ),
                "topk_evaluations": int(topk.attrs["sweep_metadata"]["total_k_evaluations"]),
                "minimum_topk_evaluations": int(topk.attrs["sweep_metadata"]["minimum_k_evaluations"]),
            }, [rank_path, topk_path, stage_dir / "prefilter_report.parquet", result_path]

        selection_result, selection_sig, selection_outputs = _run_stage(
            manifest, manifest_path, run_root, "feature_selection", selection_payload,
            select_features_action,
            deadline_monotonic=selection_deadline,
        )
        selected_features = selection_result["selected_features"]

        model_tuning.CV_FOLDS = cv_folds
        model_tuning.WF_TEST_TO_TRAIN_RATIO = 0.1
        model_tuning.MAX_N_ESTIMATORS = int(profile["maximum_estimators"])
        model_tuning.EARLY_STOPPING_ROUNDS = int(profile["early_stopping_rounds"])
        model_tuning.PRUNING_REPORT_EVERY_N_ITER = max(1, int(profile["early_stopping_rounds"]))
        model_tuning.N_TRIALS = int(profile["model_trials"])
        model_tuning.ENABLE_FOLD_RECENCY_WEIGHTING = False
        model_search_space = {
            key: value for key, value in model_tuning.LGBM_OPTUNA_SEARCH_SPACE.items()
            if key not in {"monotone_constraints_method", "monotone_penalty"}
        }
        if not isinstance(selected_features, list) or not selected_features:
            raise RuntimeError("Feature-selection result has no selected feature order")
        if len(set(selected_features)) != len(selected_features) or not set(selected_features).issubset(development_features.columns):
            raise RuntimeError("Feature-selection result contains missing or duplicate features")
        tune_x = development_features.loc[:, selected_features].reset_index(drop=True)
        tune_y = development_y.reset_index(drop=True)
        tune_opened = development_opened
        tune_decision_at = development_decision_at
        tune_folds = _purged_walk_forward_folds(
            model_tuning, tune_opened, tune_decision_at,
            int(profile.get("model_cv_folds", cv_folds)), 0.1,
            validation_mask=development_decision,
        )
        tune_fold_indices = model_tuning.build_fold_indices(tune_folds)
        tune_fold_weights = model_tuning.build_fold_recency_weights(tune_folds)
        tune_y_np = tune_y.to_numpy(dtype=np.int8, copy=False)
        tune_x_np = tune_x.to_numpy(dtype=np.float32, copy=False)
        tune_weights = development_weights
        tune_train_set = lgb.Dataset(
            tune_x_np,
            label=tune_y_np,
            weight=tune_weights,
            feature_name=selected_features,
            free_raw_data=False,
        )
        model_search_signature_payload = {
            "target": contract["training_target"],
            "data_sha256": raw_identity["sha256"],
            "fit_start": fit_start.isoformat(),
            "fit_end_exclusive": FIT_END.isoformat(),
            "weight_artifact_signature": weight_sig,
            "training_rows": int(len(tune_y_np)),
            "decision_rows": int(development_decision.sum()),
            "training_weight_sha256": _array_sha256(tune_weights, dtype=np.float32),
            "decision_mask_sha256": _array_sha256(development_decision, dtype=np.bool_),
            "fold_indices": _fold_index_fingerprints(tune_folds),
            "weight_scheme": {
                "decision_weight": decision_weight,
                "auxiliary_weight": float((1.0 - decision_weight) / 4.0),
            },
            "selection_artifact_signature": selection_sig,
            "feature_order": selected_features,
            "search_space": model_search_space,
            "max_estimators": int(profile["maximum_estimators"]),
            "early_stopping_rounds": int(profile["early_stopping_rounds"]),
            "cv_folds": int(profile.get("model_cv_folds", cv_folds)),
            "code": _code_fingerprints([
                ROOT / "optimize_lgbm_hyperparameters.py",
                ROOT / "features/btc_preopen_contract.py",
                ROOT / "utils/metrics.py",
                ROOT / "train_lgbm.py",
                Path(__file__),
            ]),
            "objective": "unweighted decision-row binary_logloss with chronological OOF",
            "seed": int(profile["seed"]),
            "compute_backend": compute_backend,
        }
        model_stage_deadline = deadline_for("model_tuning_and_fit")
        model_tuning_deadline = max(
            time.perf_counter(),
            model_stage_deadline - min(
                300.0,
                float(stages.get("model_tuning_and_fit", 0)) * 0.1,
            ),
        )

        def tune_model_action(stage_dir):
            objective = model_tuning.make_objective(
                train_set=tune_train_set,
                feature_names=selected_features,
                x_np=tune_x_np,
                y_np=tune_y_np,
                sample_weight_np=tune_weights,
                folds=tune_folds,
                fold_indices=tune_fold_indices,
                fold_weight_by_id=tune_fold_weights,
                search_space=model_search_space,
                unweighted_validation=True,
            )
            result, outputs = _run_optuna_stage(
                optuna=optuna,
                stage_name="model_tuning",
                stage_dir=stage_dir,
                objective=objective,
                target_trials=int(profile["model_trials"]),
                minimum_successful_trials=int(profile["minimum_successful_trials"]),
                stage_deadline=model_tuning_deadline,
                overall_deadline=overall_deadline,
                seed=int(profile["seed"]),
                catch=(lgb.basic.LightGBMError, OSError),
            )
            result.update(
                {
                    "training_rows": int(len(tune_y_np)),
                    "decision_rows": int(development_decision.sum()),
                    "training_weight_sha256": _array_sha256(tune_weights, dtype=np.float32),
                    "validation_weight": "no observation weights on actual decision rows",
                    "fold_indices": _fold_index_fingerprints(tune_folds),
                }
            )
            _write_json(outputs[1], result)
            return result, outputs

        model_tuning_result, model_tuning_sig, model_tuning_outputs = _run_stage(
            manifest, manifest_path, run_root, "model_tuning", model_search_signature_payload,
            tune_model_action,
            must_run=lambda result: (
                int(result.get("target_trials", 0)) < int(profile["model_trials"])
                or int(result.get("minimum_successful_trials", 0)) < int(profile["minimum_successful_trials"])
                or int(result.get("successful_trials", 0)) < int(profile["minimum_successful_trials"])
            ),
            deadline_monotonic=model_tuning_deadline,
        )
        if time.perf_counter() >= overall_deadline:
            raise TimeoutError("Overall runtime budget expired after model tuning")
        if not isinstance(model_tuning_result.get("best_iteration"), int) or model_tuning_result["best_iteration"] < 1:
            raise RuntimeError("Model tuning did not produce a positive best iteration count")
        tuned_params = dict(model_tuning_result["best_params"])
        final_rows = np.flatnonzero(fit_mask)
        final_x = features.iloc[final_rows][selected_features].to_numpy(dtype=np.float32, copy=True)
        final_x[~np.isfinite(final_x)] = np.nan
        final_y = y[final_rows]
        decision_weight = float(weight_result["decision_weight"])
        final_weights = _preopen_sample_weights(decision_mask[final_rows], decision_weight)
        final_model_signature = {
            "tuning_artifact_signature": model_tuning_sig,
            "weight_artifact_signature": weight_sig,
            "selected_feature_signature": selection_sig,
            "fit_rows": int(len(final_rows)),
            "training_weight_sha256": _array_sha256(final_weights, dtype=np.float32),
            "fit_end_exclusive": FIT_END,
            "best_iteration": int(model_tuning_result["best_iteration"]),
            "final_params": tuned_params,
            "seed": int(profile["seed"]),
            "compute_backend": compute_backend,
            "code": _code_fingerprints([
                ROOT / "train_lgbm.py",
                ROOT / "features/btc_preopen_contract.py",
                ROOT / "utils/metrics.py",
                Path(__file__),
            ]),
        }

        def fit_final_model_action(stage_dir):
            if time.perf_counter() >= model_stage_deadline:
                raise TimeoutError("Model tuning/fit budget expired before the final fit")
            if not np.array_equal(final_weights, development_weights):
                raise RuntimeError("Final model weights differ from the validated development weights")
            parameters = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "device_type": compute_backend,
                "num_threads": NUM_THREADS,
                "max_bin": 63,
                "feature_pre_filter": False,
                "deterministic": True,
                "force_col_wise": True,
                "seed": int(profile["seed"]),
                "feature_fraction_seed": int(profile["seed"]),
                "bagging_seed": int(profile["seed"]),
                "data_random_seed": int(profile["seed"]),
                **tuned_params,
            }
            training_set = lgb.Dataset(
                final_x,
                label=final_y,
                weight=final_weights,
                feature_name=selected_features,
                free_raw_data=True,
            )
            model = lgb.train(
                parameters,
                training_set,
                num_boost_round=int(model_tuning_result["best_iteration"]),
            )
            model_path = stage_dir / "lgbm_model.txt"
            temporary = model_path.with_suffix(".txt.tmp")
            model.save_model(str(temporary))
            temporary.replace(model_path)
            result_path = stage_dir / "training_manifest.json"
            _write_json(
                result_path,
                {
                    "model_path": model_path.resolve().relative_to(ROOT).as_posix(),
                    "model_sha256": _sha256(model_path),
                    "feature_order": selected_features,
                    "training_rows": int(len(final_rows)),
                    "latest_label_available_at": label_available[final_rows].max(),
                    "fit_end_exclusive": FIT_END,
                    "decision_weight": decision_weight,
                    "auxiliary_row_weight": float((1.0 - decision_weight) / 4.0),
                    "best_iteration": int(model_tuning_result["best_iteration"]),
                    "params": parameters,
                },
            )
            return {"model_path": model_path.resolve().relative_to(ROOT).as_posix(), "training_rows": int(len(final_rows)), "model_sha256": _sha256(model_path)}, [model_path, result_path]

        final_model_result, final_model_sig, final_model_outputs = _run_stage(
            manifest, manifest_path, run_root, "final_model", final_model_signature,
            fit_final_model_action,
            deadline_monotonic=model_stage_deadline,
        )
        final_model_path = ROOT / final_model_result["model_path"]
        if not final_model_path.is_file() or _sha256(final_model_path) != final_model_result.get("model_sha256"):
            raise RuntimeError("Completed final model is missing or has a checksum mismatch")
        final_model = lgb.Booster(model_file=str(final_model_path))
        bundle_dir = final_model_path.parent
        CALIBRATOR_PATH = bundle_dir / "platt_calibrator.json"
        MODEL_PATH = final_model_path
        PREDICTIONS_PATH = bundle_dir / "external_test_predictions.parquet"
        METRICS_PATH = bundle_dir / "evaluation.json"
        MODEL_META_PATH = bundle_dir / "lgbm_meta.json"
        FEATURE_SOURCE_PATH = bundle_dir / "feature_sources.json"
        calibration_identity = {
            "model_sha256": final_model_result["model_sha256"],
            "seed": int(profile["seed"]),
            "feature_order": selected_features,
            "data_sha256": raw_identity["sha256"],
            "calibration_start": CALIBRATION_START.isoformat(),
            "calibration_end_exclusive": CALIBRATION_END.isoformat(),
            "first_external_test_decision": TEST_FIRST_DECISION.isoformat(),
            "contract_sha256": contract_identity["sha256"],
            "minimum_rows": int(profile.get("minimum_calibration_rows", 1000)),
        }
        calibration_evaluation_deadline = deadline_for("calibration_evaluation_bundle")

        def calibration_action(stage_dir):
            global CALIBRATOR_PATH
            if time.perf_counter() >= calibration_evaluation_deadline:
                raise TimeoutError("Calibration/evaluation budget expired before calibration fit")
            CALIBRATOR_PATH = stage_dir / "platt_calibrator.json"
            calibration = _fit_calibrator(
                final_model, frame, features.loc[:, selected_features].to_numpy(dtype=np.float32, copy=False),
                y, calibration_identity,
                minimum_rows=int(profile.get("minimum_calibration_rows", 1000)),
            )
            return {"training_rows": int(calibration["training_rows"]), "calibrator_path": CALIBRATOR_PATH.resolve().relative_to(ROOT).as_posix()}, [CALIBRATOR_PATH]

        calibration_payload, calibration_sig, calibration_outputs = _run_stage(
            manifest, manifest_path, run_root, "calibration", {
                "final_model_signature": final_model_sig,
                "calibration_identity": calibration_identity,
                "code_sha256": _sha256(ROOT / "run_btc_preopen_experiment.py"),
            }, calibration_action,
            deadline_monotonic=calibration_evaluation_deadline,
        )
        CALIBRATOR_PATH = ROOT / calibration_payload["calibrator_path"]
        if not CALIBRATOR_PATH.is_file():
            raise RuntimeError("Completed calibrator is missing")
        calibration = json.loads(CALIBRATOR_PATH.read_text(encoding="utf-8"))
        prediction_matrix = features.loc[:, selected_features].to_numpy(dtype=np.float32, copy=False)
        evaluation_identity = {
            "final_model_signature": final_model_sig,
            "calibration_signature": calibration_sig,
            "code_sha256": _sha256(ROOT / "run_btc_preopen_experiment.py"),
            "test_start": TEST_MARKET_START,
            "test_end": TEST_LAST_MARKET_START,
            "official_markets_sha256": _sha256(OFFICIAL_MARKETS_PATH) if OFFICIAL_MARKETS_PATH.is_file() else None,
        }

        def evaluation_action(stage_dir):
            global PREDICTIONS_PATH, METRICS_PATH
            if time.perf_counter() >= calibration_evaluation_deadline:
                raise TimeoutError("Calibration/evaluation budget expired before external evaluation")
            PREDICTIONS_PATH = stage_dir / "external_test_predictions.parquet"
            METRICS_PATH = stage_dir / "evaluation.json"
            prior_mask = fit_mask & decision_mask
            prior_up = float(y[prior_mask].mean())
            metrics, external, prediction_frame = _evaluate_preopen_external(
                final_model,
                calibration,
                frame,
                prediction_matrix,
                valid_labels,
                prior_up,
                OFFICIAL_MARKETS_PATH,
                split["test_history_status"],
            )
            evaluation = {
                "status": "preopen_v1_training_pipeline_completed",
                "target_source": "Binance COIN-M BTCUSD index price proxy",
                "official_labels_mixed": False,
                "chronological_split": split,
                "test_history_status": split["test_history_status"],
                "fit_label_latest_available_at": label_available[fit_mask].max(),
                "calibration_label_latest_available_at": calibration["latest_label_available_at"],
                "development_proxy_prevalence_on_decisions": prior_up,
                "external_test": external,
                "metrics": metrics,
                "rows_predicted": int(len(prediction_frame)),
                "model_sha256": final_model_result["model_sha256"],
                "feature_count": len(selected_features),
                "elapsed_seconds": time.perf_counter() - started,
            }
            _write_json(METRICS_PATH, evaluation)
            return {"prediction_rows": int(len(prediction_frame)), "metrics_path": METRICS_PATH.resolve().relative_to(ROOT).as_posix()}, [PREDICTIONS_PATH, METRICS_PATH]

        evaluation_result, evaluation_sig, evaluation_outputs = _run_stage(
            manifest, manifest_path, run_root, "external_evaluation", evaluation_identity,
            evaluation_action,
            deadline_monotonic=calibration_evaluation_deadline,
        )
        PREDICTIONS_PATH = ROOT / next(record["path"] for record in evaluation_outputs if record["path"].endswith(".parquet"))
        METRICS_PATH = ROOT / evaluation_result["metrics_path"]
        final_features_path = bundle_dir / "feature_order.json"
        feature_sources_path = bundle_dir / "feature_sources.json"
        model_meta_path = bundle_dir / "lgbm_meta.json"
        bundle_manifest_path = bundle_dir / "model_bundle.json"
        if time.perf_counter() >= calibration_evaluation_deadline:
            raise TimeoutError("Calibration, evaluation, and bundle runtime budget expired before bundle writing")
        if final_features_path.is_file():
            persisted_order = json.loads(final_features_path.read_text(encoding="utf-8")).get("feature_order")
            if persisted_order != selected_features:
                raise RuntimeError("Persisted final feature order differs from feature-selection result")
        else:
            _write_json(final_features_path, {"feature_order": selected_features})
        _write_json(
            feature_sources_path,
            {
                "feature_semantics_version": "btc_preopen_v1_causal_modeling_dataset",
                "dataset_path": dataset_path.resolve().relative_to(ROOT).as_posix(),
                "dataset_profile": profile["dataset_profile"],
                "modeling_profile": profile["modeling_profile"],
                "raw_candle_path": raw_path.resolve().as_posix(),
                "raw_candle_sha256": raw_identity["sha256"],
                "indicator_fit_results_dir": indicator_result["fit_results_dir"],
                "feature_order": selected_features,
                "history_requirement": "continuous 1-minute BTC index OHLCV history from fit start; preserve all earlier rows as warmup",
                "reaction_profile_config": rp_result["best_config"],
                "volume_profile_config": vp_result["best_config"],
                "indicator_config_paths": [record["path"] for record in indicator_outputs if record["path"].endswith(".json") and "indicator_stage.json" not in record["path"]],
                "profile_state_dir": dataset_settings["profile_state_dir"].resolve().relative_to(ROOT).as_posix(),
                "profile_state_artifacts": [
                    record["path"] for record in dataset_outputs
                    if record["path"].endswith(".npz")
                ],
                "reaction_profile_config_path": rp_result["config_path"],
                "volume_profile_config_path": vp_result["config_path"],
            },
        )
        _write_json(
            model_meta_path,
            {
                "model_type": "LightGBM binary classifier with Platt calibration",
                "target_col": TARGET_COL,
                "target_source": "Binance COIN-M BTCUSD index price proxy; not official settlement",
                "feature_columns": selected_features,
                "training_rows": int(final_model_result["training_rows"]),
                "training_label_available_before_utc": FIT_END,
                "training_decision_weight": decision_weight,
                "training_auxiliary_weight": float((1.0 - decision_weight) / 4.0),
                "iteration_source": "development-only chronological OOF hyperparameter tuning",
                "calibration_path": CALIBRATOR_PATH.resolve().relative_to(ROOT).as_posix(),
                "model_sha256": final_model_result["model_sha256"],
            },
        )
        bundle_payload = {
            "contract_version": contract["contract_version"],
            "contract_path": contract_path.resolve().relative_to(ROOT).as_posix(),
            "target": contract["training_target"],
            "label_source_is_proxy": True,
            "chronological_split": split,
            "model_path": final_model_path.resolve().relative_to(ROOT).as_posix(),
            "model_sha256": final_model_result["model_sha256"],
            "calibrator_path": CALIBRATOR_PATH.resolve().relative_to(ROOT).as_posix(),
            "calibrator_sha256": _sha256(CALIBRATOR_PATH),
            "feature_order_path": final_features_path.resolve().relative_to(ROOT).as_posix(),
            "feature_sources_path": feature_sources_path.resolve().relative_to(ROOT).as_posix(),
            "model_meta_path": model_meta_path.resolve().relative_to(ROOT).as_posix(),
            "feature_order": selected_features,
            "reaction_profile_config_path": rp_result["config_path"],
            "volume_profile_config_path": vp_result["config_path"],
            "indicator_fit_result_paths": [record["path"] for record in indicator_outputs if record["path"].endswith(".json") and "indicator_stage.json" not in record["path"]],
            "history_requirement": "1-minute BTC index OHLCV; preserve causal warmup history before fit start",
            "raw_candle_path": raw_path.resolve().as_posix(),
            "raw_candle_sha256": raw_identity["sha256"],
            "dataset_profile": profile["dataset_profile"],
            "modeling_profile": profile["modeling_profile"],
            "indicator_fit_results_dir": indicator_result["fit_results_dir"],
            "prediction_reference_path": PREDICTIONS_PATH.resolve().relative_to(ROOT).as_posix(),
            "profile_state_artifacts": [
                record["path"] for record in dataset_outputs
                if record["path"].endswith(".npz")
            ],
            "verification_tolerance_abs": 1e-6,
            "status": "verification_pending",
            "prediction_compute_budget_seconds": PREDICTION_COMPUTE_BUDGET_SECONDS,
            "activated": False,
        }
        _write_json(bundle_manifest_path, bundle_payload)
        verification_path = bundle_dir / "raw_replay_verification.json"
        verification_input = {
            "bundle_payload": bundle_payload,
            "evaluation_artifact_signature": evaluation_sig,
            "raw_candle_sha256": raw_identity["sha256"],
            "code": _code_fingerprints([
                ROOT / "create_modeling_dataset.py",
                ROOT / "features/reaction_profile_fixed_grid.py",
                ROOT / "features/volume_profile_fixed_range.py",
                ROOT / "features/common_utils.py",
                ROOT / "features/ADX.py",
                ROOT / "features/BollingerBands.py",
                ROOT / "features/ChaikinOsc.py",
                ROOT / "features/KeltnerChannel.py",
                ROOT / "features/MACD.py",
                ROOT / "features/StochOsc.py",
                ROOT / "utils/data.py",
                Path(__file__),
            ]),
        }

        def verify_bundle_action(stage_dir):
            if time.perf_counter() >= calibration_evaluation_deadline:
                raise TimeoutError("Calibration/evaluation budget expired before raw bundle replay")
            replay = _run_bundle_raw_replay_fresh_process(
                bundle_manifest_path,
                PREDICTIONS_PATH,
                verification_path,
            )
            return replay, [verification_path]

        raw_replay_result, verification_sig, verification_outputs = _run_stage(
            manifest,
            manifest_path,
            run_root,
            "bundle_verification",
            verification_input,
            verify_bundle_action,
        )
        bundle_payload["status"] = "verified"
        bundle_payload["verification_path"] = verification_path.resolve().relative_to(ROOT).as_posix()
        bundle_payload["raw_replay_verification"] = raw_replay_result
        _write_json(bundle_manifest_path, bundle_payload)
        load_prediction_bundle(bundle_manifest_path)
        report_path = run_root / "final_report.md"
        evaluation = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        report_path.write_text(
            "# BTC pre-open v1 training run\n\n"
            f"- Status: `{evaluation['status']}`\n"
            f"- Label: Binance COIN-M BTCUSD index price proxy; {evaluation['test_history_status']}.\n"
            f"- Training rows: {final_model_result['training_rows']:,}; selected features: {len(selected_features)}.\n"
            f"- Calibration rows: {calibration_payload['training_rows']:,}.\n"
            f"- External decision predictions: {evaluation_result['prediction_rows']:,}.\n"
            f"- Model bundle: `{bundle_manifest_path.resolve().relative_to(ROOT).as_posix()}`\n"
            f"- Official outcomes: `{evaluation['metrics']['official_market_data']['status']}`; economic replay requires pre-start quotes.\n",
            encoding="utf-8",
        )
        manifest.update(
            {
                "status": "completed",
                "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "bundle_path": bundle_manifest_path.resolve().relative_to(ROOT).as_posix(),
                "report_path": report_path.resolve().relative_to(ROOT).as_posix(),
                "evaluation_path": METRICS_PATH.resolve().relative_to(ROOT).as_posix(),
            }
        )
        _write_json(manifest_path, manifest)
        print(
            f"[preopen] completed elapsed={manifest['elapsed_seconds']:.1f}s "
            f"bundle={bundle_manifest_path}",
            flush=True,
        )
        return {"run_root": run_root, "bundle_path": bundle_manifest_path, "report_path": report_path}
    except Exception:
        manifest["status"] = "failed"
        manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        _write_json(manifest_path, manifest)
        raise
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log_handle.close()
        _unlock_run(lock)


def run_integration_smoke():
    """Run every pipeline family on a small, isolated candle fixture."""
    import tempfile

    fixture_root = ROOT / "data/analysis/polymarket/BTC/preopen_v1/integration_smoke_fixture"
    fixture_root.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_CANDLES_PATH, nrows=8_000)
    if len(raw) < 7_500:
        raise RuntimeError("Raw BTC source is too short for the pre-open integration fixture")
    start = pd.Timestamp("2020-06-09T09:33:00Z")
    raw["Opened"] = pd.date_range(start, periods=len(raw), freq="min", tz="UTC")
    raw_path = fixture_root / "BTCUSD_INDEXVOL_UM_BTCUSDT1m.csv"
    raw.to_csv(raw_path, index=False)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    fit_end = start + pd.Timedelta(minutes=4_500)
    calibration_end = start + pd.Timedelta(minutes=6_000)
    split = contract["chronological_split_frozen_before_tuning"]
    split.update(
        {
            "feature_and_model_fit_start_utc": start.isoformat(),
            "fit_and_tuning_end_exclusive_utc": fit_end.isoformat(),
            "calibration_start_utc": fit_end.isoformat(),
            "calibration_end_exclusive_utc": calibration_end.isoformat(),
            "first_external_test_nominal_decision_utc": calibration_end.isoformat(),
            "first_external_test_market_start_utc": (calibration_end + pd.Timedelta(minutes=1)).isoformat(),
            "external_test_market_start_last_inclusive_utc": (start + pd.Timedelta(minutes=7_990)).isoformat(),
            "external_test_nominal_decision_last_utc": (start + pd.Timedelta(minutes=7_989)).isoformat(),
            "test_history_status": "historically exposed synthetic integration fixture; not a strategy result",
        }
    )
    smoke_contract_path = fixture_root / "btc_preopen_v1.json"
    smoke_contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    smoke_profile = {
        "contract_path": smoke_contract_path.resolve().relative_to(ROOT).as_posix(),
        "dataset_profile": "BTC",
        "modeling_profile": "BTC",
        "indicator_fit_profile": "candle_up_5m_qe20_qm10_center",
        "prediction_compute_budget_seconds": 45,
        "runtime_budget_seconds": 7_200,
        "stage_runtime_budgets_seconds": {
            "indicators": 1_200,
            "reaction_profile": 1_000,
            "volume_profile": 1_000,
            "target_weights": 700,
            "feature_selection": 1_000,
            "model_tuning_and_fit": 1_000,
            "calibration_evaluation_bundle": 500,
        },
        "threads": 2,
        "raw_data_path": raw_path.resolve().as_posix(),
        "indicator_fit": {
            "population_size": 8,
            "generations": 2,
            "minimum_generations": 2,
            "workers": 2,
        },
        "indicator_metric_gap": 1,
        "indicator_metric_segments_count": 3,
        "indicator_metric_min_bucket_size": 5,
        "indicator_metric_min_valid_segments": 2,
        "reaction_profile_trials": 1,
        "volume_profile_trials": 1,
        "target_weight_candidates": 2,
        "feature_selection": {
            "max_sweep_evaluations": 10,
            "max_refinement_rounds": 0,
            "permutation_feature_fraction": 0.5,
            "permutation_repeats": 1,
        },
        "model_trials": 1,
        "minimum_successful_trials": 1,
        "maximum_estimators": 20,
        "early_stopping_rounds": 5,
        "minimum_calibration_rows": 30,
        "weight_search_estimators": 20,
        "selector_estimators": 20,
        "cv_folds": 2,
        "generator_cv_folds": 2,
        "weight_cv_folds": 2,
        "selector_cv_folds": 2,
        "model_cv_folds": 2,
        "output_dir": "data/analysis/polymarket/BTC/preopen_v1/integration_smoke",
    }
    first_result = run_training_pipeline(
        profile_name="integration_smoke",
        profile_override=smoke_profile,
    )
    first_manifest_path = first_result["run_root"] / "run_manifest.json"
    first_manifest_payload = json.loads(first_manifest_path.read_text(encoding="utf-8"))
    result = run_training_pipeline(
        profile_name="integration_smoke",
        profile_override=smoke_profile,
    )
    run_manifest = result["run_root"] / "run_manifest.json"
    manifest_payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    if manifest_payload.get("status") != "completed":
        raise RuntimeError("Integration resume did not leave a completed run manifest")
    rerun_stages = [
        name for name, stage in manifest_payload.get("stages", {}).items()
        if (
            stage.get("status") != "completed"
            or int(stage.get("attempt", 0))
            != int(first_manifest_payload.get("stages", {}).get(name, {}).get("attempt", -1))
        )
    ]
    if rerun_stages:
        raise RuntimeError(f"Integration resume reran or failed completed stages: {rerun_stages}")
    bundle_payload, _, _ = load_prediction_bundle(result["bundle_path"])
    if bundle_payload.get("raw_replay_verification", {}).get("status") != "verified":
        raise RuntimeError("Integration resume lost its raw-data bundle verification")
    selection_result = manifest_payload["stages"]["feature_selection"]["result"]
    if int(selection_result.get("permutation_features_scored", 0)) < 1:
        raise RuntimeError("Integration smoke did not exercise decision-row permutation scoring")
    _write_json(
        fixture_root / "integration_result.json",
        {
            "first_run_root": str(first_result["run_root"]),
            "resumed_run_root": str(result["run_root"]),
            "run_manifest": run_manifest.resolve().relative_to(ROOT).as_posix(),
            "bundle_path": result["bundle_path"].resolve().relative_to(ROOT).as_posix(),
            "resume_skipped_completed_stages": True,
        },
    )
    return result


if __name__ == "__main__":
    run()
