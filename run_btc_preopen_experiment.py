"""Run the first causal BTC pre-open baseline; all settings are file constants."""
from __future__ import annotations

import hashlib
import json
import math
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
    TARGET_AVAILABLE_COL,
    TARGET_COL,
    TARGET_END_PRICE_COL,
    TARGET_START_PRICE_COL,
    build_preopen_contract_frame,
)
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


def _fit_calibrator(model, frame, x, y, identity):
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
    if len(rows) < 1000:
        raise RuntimeError(f"Too few point-in-time calibration decisions: {len(rows)}")
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
    started = time.perf_counter()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    identity = _feature_cache_identity()
    frame = _load_or_build_feature_cache(identity)
    x, y, valid = _valid_matrix(frame)
    print(
        f"[preopen] rows={len(frame):,} cache={FEATURE_CACHE_PATH.name} "
        f"features={len(FEATURE_COLUMNS)}", flush=True
    )

    if (
        MODEL_PATH.is_file()
        and METRICS_PATH.is_file()
        and PREDICTIONS_PATH.is_file()
        and RUN_MANIFEST_PATH.is_file()
    ):
        prior_run = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
        if prior_run.get("input_identity") == identity and prior_run.get("status") == "completed":
            print(f"[preopen] matching completed run already exists: {MODEL_PATH}", flush=True)
            return

    smoke = _run_smoke(frame, x, y, valid)
    _write_json(DATA_DIR / "smoke_run.json", smoke)
    print("[preopen] end-to-end timestamp and isolation smoke passed", flush=True)
    iteration_selection = None
    if ITERATION_CHECKPOINT_PATH.is_file():
        saved_iteration = json.loads(
            ITERATION_CHECKPOINT_PATH.read_text(encoding="utf-8")
        )
        if saved_iteration.get("input_identity") == identity:
            iteration_selection = saved_iteration
            print("[preopen] resuming verified iteration-selection checkpoint", flush=True)
    if iteration_selection is None:
        iteration_selection = _fit_iteration_count(frame, x, y, valid)
        iteration_selection["input_identity"] = identity
        _write_json(ITERATION_CHECKPOINT_PATH, iteration_selection)

    model = None
    training_rows = None
    if MODEL_PATH.is_file() and MODEL_CHECKPOINT_PATH.is_file():
        model_checkpoint = json.loads(
            MODEL_CHECKPOINT_PATH.read_text(encoding="utf-8")
        )
        if (
            model_checkpoint.get("input_identity") == identity
            and int(model_checkpoint.get("best_iteration", 0))
            == int(iteration_selection["best_iteration"])
            and model_checkpoint.get("model_sha256") == _sha256(MODEL_PATH)
        ):
            model = lgb.Booster(model_file=str(MODEL_PATH))
            training_rows = int(model_checkpoint["training_rows"])
            print("[preopen] resuming verified final-model checkpoint", flush=True)
    if model is None:
        model, training_rows = _fit_final_model(
            frame, x, y, valid, iteration_selection["best_iteration"]
        )
        _write_json(
            MODEL_CHECKPOINT_PATH,
            {
                "input_identity": identity,
                "best_iteration": int(iteration_selection["best_iteration"]),
                "training_rows": training_rows,
                "model_sha256": _sha256(MODEL_PATH),
            },
        )
    train_prevalence_mask = (
        valid
        & frame[DECISION_COL].to_numpy(dtype=np.bool_, copy=False)
        & frame[TARGET_AVAILABLE_COL].le(FIT_END).to_numpy()
    )
    prior_up = float(y[train_prevalence_mask].mean())
    calibration = _fit_calibrator(model, frame, x, y, identity)
    metrics, external_test, predictions = _evaluate_external_test(
        model, calibration, frame, x, y, valid, prior_up
    )

    model_hash = _sha256(MODEL_PATH)
    model_meta = {
        "experiment_id": "btc_preopen_v1_20261004",
        "model_type": "causal raw-candle LightGBM baseline; tuning layers not yet fitted",
        "model_path": "data/models/BTC/btc_preopen_v1/lgbm_model.txt",
        "model_sha256": model_hash,
        "target_col": TARGET_COL,
        "target_source": "Binance COIN-M BTCUSD index proxy",
        "feature_columns": list(FEATURE_COLUMNS),
        "training_rows": training_rows,
        "training_label_available_through_utc": FIT_END,
        "training_weights": "unit weight for every eligible minute row",
        "iteration_selection": iteration_selection,
        "smoke_run": smoke,
        "calibration": calibration,
        "feature_input_cutoff": "current row candle close; no later candle values",
        "external_test_metrics": "stored in data/analysis/polymarket/BTC/preopen_v1/evaluation.json",
    }
    _write_json(MODEL_META_PATH, model_meta)
    _write_json(
        FEATURE_SOURCE_PATH,
        {
            "feature_semantics_version": "btc_preopen_raw_causal_v1",
            "source_columns": [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "UM_BTCUSDT_Close",
                "Opened",
            ],
            "availability": "all source values from Opened candle, after candle close; past minute lags only",
            "feature_order": list(FEATURE_COLUMNS),
            "code_sha256": _sha256(ROOT / "features/btc_preopen_baseline.py"),
        },
    )
    evaluation = {
        "experiment_id": "btc_preopen_v1_20261004",
        "status": "causal_raw_feature_baseline_completed; full target-specific tuning remains pending",
        "input_identity": identity,
        "contract": json.loads(CONTRACT_PATH.read_text(encoding="utf-8")),
        "hardware_budget": {
            "backend": "LightGBM CPU",
            "threads": NUM_THREADS,
            "available_logical_processors_at_planning": 20,
            "boosting_round_budget": BOOSTING_ROUNDS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "minute_rows_used_for_final_fit": training_rows,
        },
        "iteration_selection": iteration_selection,
        "training_proxy_up_prevalence_on_real_decisions": prior_up,
        "calibration": calibration,
        "external_test": external_test,
        "metrics": metrics,
        "model_sha256": model_hash,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "end_to_end_elapsed_seconds": None,
        "elapsed_time_note": (
            "This execution resumed verified artifacts; cumulative wall time from the "
            "earlier feature-cache and fit invocations was not retained."
        ),
    }
    _write_json(METRICS_PATH, evaluation)
    run_manifest = {
        "experiment_id": "btc_preopen_v1_20261004",
        "status": "completed",
        "input_identity": identity,
        "model_sha256": model_hash,
        "model_meta_path": MODEL_META_PATH.relative_to(ROOT).as_posix(),
        "feature_source_path": FEATURE_SOURCE_PATH.relative_to(ROOT).as_posix(),
        "evaluation_path": METRICS_PATH.relative_to(ROOT).as_posix(),
        "predictions_path": PREDICTIONS_PATH.relative_to(ROOT).as_posix(),
        "last_invocation_elapsed_seconds": time.perf_counter() - started,
        "end_to_end_elapsed_seconds": None,
        "elapsed_time_note": (
            "This execution resumed verified artifacts; cumulative wall time from the "
            "earlier feature-cache and fit invocations was not retained."
        ),
    }
    _write_json(RUN_MANIFEST_PATH, run_manifest)
    print(
        "[preopen] completed resumed invocation in "
        f"{run_manifest['last_invocation_elapsed_seconds']:.1f}s "
        "(full-run elapsed unavailable)",
        flush=True,
    )


if __name__ == "__main__":
    run()
