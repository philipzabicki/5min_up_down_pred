"""Compare BTC-market disagreement corrections with and without context features."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_btc_conditional_market_offset_20261018 as source_experiment
import run_btc_market_probability_experiment as probability
import run_btc_preopen_fee_correction as fee_correction
import run_btc_preopen_research as research
import run_btc_preopen_t59_reassessment as replay


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_market_basis_vol_interaction_20261017.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / "data/analysis/polymarket/BTC/btc_market_basis_vol_interaction_20261017"
REPORT_DIR = ROOT / "reports/btc_preopen/btc_market_basis_vol_interaction_20261017"
REPORT_PATH = ROOT / "reports/btc_preopen/btc_market_basis_vol_interaction_20261017.md"
STATUS_PATH = ROOT / "docs/BTC_RESEARCH_STATUS_20261010.md"
STATUS_MARKER = "## BTC market-basis-volatility interaction experiment — 2026-10-17"
M0_FEATURES = tuple(CONFIG["features"]["M0"])
M1_FEATURES = tuple(CONFIG["features"]["M1"])
DEV_CUTOFF = pd.Timestamp(CONFIG["sample"]["development_cutoff_exclusive_utc"])
HOLDOUT_CUTOFF = pd.Timestamp(CONFIG["sample"]["holdout_cutoff_exclusive_utc"])
HOLDOUT_START = pd.Timestamp(CONFIG["sample"]["holdout_start_inclusive_utc"])
EXPECTED_DEV_OOF_ROWS = 8675
EXPECTED_HOLDOUT_ROWS = 24735


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    frame.to_parquet(temp, index=False, compression="zstd")
    temp.replace(path)


def _logit(values) -> np.ndarray:
    return source_experiment._logit(values)


def _frame_sha256(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    ordered = frame.loc[:, columns].reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(ordered, index=False, categorize=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _sample_sha256(frame: pd.DataFrame, features: tuple[str, ...]) -> str:
    columns = ("market_start_utc", "condition_id", "label_available_at_utc", "official_outcome_up", *features)
    return _frame_sha256(frame, columns)


def _fit_configuration(model_name: str, config: dict) -> dict:
    features = M0_FEATURES if model_name == "M0" else M1_FEATURES
    config_id = f"leaves{config['num_leaves']}_depth{config['max_depth']}_l2_{int(config['lambda_l2'])}"
    return {"model": model_name, "config_id": config_id, "features": list(features), **config}


def _fit_signature(identity: dict, model_name: str, config: dict, train: pd.DataFrame, valid: pd.DataFrame | None, features: tuple[str, ...]) -> dict:
    return {
        "identity_sha256": identity["identity_sha256"],
        "model": model_name,
        "configuration": config,
        "features": list(features),
        "train_rows": len(train),
        "train_sha256": _sample_sha256(train, features),
        "valid_rows": len(valid) if valid is not None else None,
        "valid_sha256": _sample_sha256(valid, features) if valid is not None else None,
    }


def _fit_tree(train: pd.DataFrame, features: tuple[str, ...], config: dict) -> tuple[lgb.Booster, dict]:
    x_train = train.loc[:, features].to_numpy(dtype=np.float64)
    y_train = train.official_outcome_up.to_numpy(dtype=np.int8)
    offset_train = train.logit_p_market.to_numpy(dtype=np.float64)
    if not np.isfinite(x_train).all() or not np.isfinite(offset_train).all():
        raise RuntimeError("Training inputs or the market offset contain non-finite values")
    if len(offset_train) != len(y_train):
        raise AssertionError("The offset and target lost row alignment")
    min_leaf = max(100, int(math.ceil(0.02 * len(train))))
    dataset = lgb.Dataset(
        x_train, label=y_train, init_score=offset_train, feature_name=list(features),
        free_raw_data=False,
    )
    if not np.array_equal(dataset.get_init_score(), offset_train):
        raise AssertionError("LightGBM init_score is not aligned with the training market offset")
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": float(CONFIG["training"]["learning_rate"]),
        "num_leaves": int(config["num_leaves"]),
        "max_depth": int(config["max_depth"]),
        "lambda_l2": float(config["lambda_l2"]),
        "min_data_in_leaf": min_leaf,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "seed": int(CONFIG["training"]["seed"]),
        "feature_fraction_seed": int(CONFIG["training"]["seed"]),
        "bagging_seed": int(CONFIG["training"]["seed"]),
        "data_random_seed": int(CONFIG["training"]["seed"]),
        "num_threads": int(CONFIG["training"]["num_threads"]),
        "device_type": "cpu",
        "boost_from_average": False,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    started = time.perf_counter()
    with threadpool_limits(limits=int(CONFIG["training"]["num_threads"])):
        booster = lgb.train(params, dataset, num_boost_round=int(CONFIG["training"]["iterations"]))
    elapsed = time.perf_counter() - started
    if booster.current_iteration() != int(CONFIG["training"]["iterations"]):
        raise AssertionError(f"Expected exactly 100 iterations, got {booster.current_iteration()}")
    return booster, {
        "elapsed_seconds": elapsed,
        "min_data_in_leaf": min_leaf,
        "iterations": int(booster.current_iteration()),
        "init_score_train_row_aligned": True,
        "sample_weights_used": False,
        "monotonic_constraints_used": False,
        "early_stopping_used": False,
    }


def _predict_offset(booster: lgb.Booster, frame: pd.DataFrame, features: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    x = frame.loc[:, features].to_numpy(dtype=np.float64)
    offset = frame.logit_p_market.to_numpy(dtype=np.float64)
    raw_correction = np.asarray(booster.predict(x, raw_score=True, num_threads=4), dtype=np.float64)
    checked_offset = _logit(frame.p_market)
    if not np.array_equal(offset, checked_offset):
        raise AssertionError("The saved market logit differs from the verified market offset")
    probability_values = source_experiment._apply_offset_correction(frame.p_market, raw_correction)
    return raw_correction, probability_values


def _checkpoint_paths(key: str) -> tuple[Path, Path, Path]:
    directory = OUTPUT_DIR / "checkpoints"
    return directory / f"{key}.txt", directory / f"{key}.parquet", directory / f"{key}.json"


def _load_fit_checkpoint(signature: dict, key: str, expected_rows: int) -> tuple[lgb.Booster, pd.DataFrame, dict] | None:
    model_path, predictions_path, record_path = _checkpoint_paths(key)
    if not all(path.is_file() for path in (model_path, predictions_path, record_path)):
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("fit_signature") != signature:
        return None
    if record.get("model_sha256") != _sha256(model_path) or record.get("predictions_sha256") != _sha256(predictions_path):
        return None
    predictions = pd.read_parquet(predictions_path)
    if len(predictions) != expected_rows:
        return None
    return lgb.Booster(model_file=str(model_path)), predictions, record["fit"]


def _save_fit_checkpoint(signature: dict, key: str, booster: lgb.Booster, predictions: pd.DataFrame, fit: dict) -> None:
    model_path, predictions_path, record_path = _checkpoint_paths(key)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_model = model_path.with_name(model_path.stem + ".tmp.txt")
    booster.save_model(str(temp_model))
    temp_model.replace(model_path)
    _write_parquet(predictions_path, predictions)
    _write_json(record_path, {
        "fit_signature": signature,
        "fit": fit,
        "model_sha256": _sha256(model_path),
        "predictions_sha256": _sha256(predictions_path),
        "completed": True,
    })


def _run_fit(
    identity: dict,
    model_name: str,
    config: dict,
    fold_id: str,
    train: pd.DataFrame,
    valid: pd.DataFrame | None,
) -> tuple[lgb.Booster, pd.DataFrame | None, dict]:
    features = M0_FEATURES if model_name == "M0" else M1_FEATURES
    signature = _fit_signature(identity, model_name, config, train, valid, features)
    key = f"{model_name}__{config['config_id']}__{fold_id}"
    expected_rows = len(valid) if valid is not None else 0
    cached = _load_fit_checkpoint(signature, key, expected_rows) if valid is not None else None
    if cached is not None:
        print(f"[interaction] reuse {key}", flush=True)
        return cached[0], cached[1], cached[2]
    cached_final = _load_fit_checkpoint(signature, key, 0) if valid is None else None
    if cached_final is not None:
        print(f"[interaction] reuse {key}", flush=True)
        return cached_final[0], None, cached_final[2]

    print(f"[interaction] fit {key}: train={len(train):,}, valid={expected_rows:,}", flush=True)
    booster, fit = _fit_tree(train, features, config)
    peak = source_experiment._peak_working_set_bytes()
    fit["peak_working_set_bytes_after_fit"] = peak
    predictions = None
    if valid is not None:
        raw, p = _predict_offset(booster, valid, features)
        predictions = valid.loc[:, ["market_start_utc", "condition_id", "official_outcome_up", "p_market"]].copy()
        predictions["raw_correction"] = raw
        predictions[f"p_{model_name}"] = p
        loss = _row_log_loss(predictions.official_outcome_up.to_numpy(), p)
        fit["validation_log_loss"] = loss
        fit["offset_added_once"] = True
        _save_fit_checkpoint(signature, key, booster, predictions, fit)
    else:
        _save_fit_checkpoint(signature, key, booster, pd.DataFrame(columns=["condition_id", "market_start_utc"]), fit)
    print(f"[interaction] done {key}: {fit['elapsed_seconds']:.3f}s, peak={peak}", flush=True)
    return booster, predictions, fit


def _row_log_loss(y, p) -> float:
    target = np.asarray(y, dtype=np.float64)
    probability_values = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(target * np.log(probability_values) + (1.0 - target) * np.log1p(-probability_values)))


def _fold_slices(dev: pd.DataFrame, fold_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(fold_config["start_inclusive_utc"])
    end = pd.Timestamp(fold_config["end_exclusive_utc"])
    train = dev.loc[dev.market_start_utc.lt(start) & dev.label_available_at_utc.le(start)].copy()
    valid = dev.loc[dev.market_start_utc.ge(start) & dev.market_start_utc.lt(end)].copy()
    if train.empty or valid.empty or train.official_outcome_up.nunique() != 2:
        raise RuntimeError(f"Chronological fold is unusable: {fold_config['fold_id']}")
    _assert_fold_boundaries(train, valid, start, end)
    return train, valid


def _assert_fold_boundaries(train: pd.DataFrame, valid: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    if not train.market_start_utc.lt(start).all() or not train.label_available_at_utc.le(start).all():
        raise AssertionError("A training row violates the chronological or label-release purge")
    if not valid.market_start_utc.ge(start).all() or not valid.market_start_utc.lt(end).all():
        raise AssertionError("A validation row is outside its frozen UTC interval")
    expected_decision = valid.market_start_utc - pd.Timedelta(minutes=1)
    if not valid.btc_feature_close_at.eq(expected_decision).all():
        raise AssertionError("A BTC feature was not available by the T-60 decision")


def _load_previous_predictions(dev: pd.DataFrame, holdout: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    old_dev_path = source_experiment.OUTPUT_DIR / "development_cv_predictions.parquet"
    old_holdout_path = source_experiment.OUTPUT_DIR / "holdout_predictions.parquet"
    if not old_dev_path.is_file() or not old_holdout_path.is_file():
        raise FileNotFoundError("The frozen previous D prediction tables are missing")
    old_dev = pd.read_parquet(old_dev_path, columns=[
        "market_start_utc", "condition_id", "official_outcome_up", "p_A_market", "p_D_conditional",
    ])
    old_holdout = pd.read_parquet(old_holdout_path, columns=[
        "market_start_utc", "condition_id", "official_outcome_up", "p_A_market", "p_D_conditional",
    ])
    old_freeze = json.loads(source_experiment.FREEZE_PATH.read_text(encoding="utf-8"))
    old_run = json.loads(source_experiment.RUN_PATH.read_text(encoding="utf-8"))
    dev_sha256, holdout_sha256 = _sha256(old_dev_path), _sha256(old_holdout_path)
    if dev_sha256 != old_freeze["selected_development_cv_predictions_sha256"]:
        raise RuntimeError("The saved previous D OOF table does not match its frozen development bundle")
    if holdout_sha256 != old_run["holdout_prediction_sha256"]:
        raise RuntimeError("The saved previous D holdout table does not match its completed run manifest")
    key = ["market_start_utc", "condition_id"]
    dev_common = dev.merge(old_dev, on=key, how="inner", suffixes=("", "_previous"), validate="one_to_one")
    holdout_common = holdout.merge(old_holdout, on=key, how="inner", suffixes=("", "_previous"), validate="one_to_one")
    if len(dev_common) != EXPECTED_DEV_OOF_ROWS or len(holdout_common) != EXPECTED_HOLDOUT_ROWS:
        raise RuntimeError(f"Frozen comparison rows changed: development={len(dev_common)}, holdout={len(holdout_common)}")
    for frame in (dev_common, holdout_common):
        if not np.array_equal(frame.official_outcome_up.to_numpy(), frame.official_outcome_up_previous.to_numpy()):
            raise RuntimeError("The previous frozen predictions do not match the official labels by market")
        if not np.allclose(frame.p_market.to_numpy(), frame.p_A_market.to_numpy(), rtol=0.0, atol=0.0):
            raise RuntimeError("The previous frozen predictions use a different market reference")
    if dev_common.market_start_utc.duplicated().any() or holdout_common.market_start_utc.duplicated().any():
        raise RuntimeError("A previous prediction table has duplicate comparison markets")
    identity = {
        "development_oof_predictions_path": str(old_dev_path.relative_to(ROOT)),
        "development_oof_predictions_sha256": dev_sha256,
        "holdout_predictions_path": str(old_holdout_path.relative_to(ROOT)),
        "holdout_predictions_sha256": holdout_sha256,
        "previous_frozen_development_bundle_sha256": _sha256(source_experiment.FREEZE_PATH),
        "previous_completed_run_manifest_sha256": _sha256(source_experiment.RUN_PATH),
        "development_common_rows": len(dev_common),
        "holdout_common_rows": len(holdout_common),
    }
    return dev_common, holdout_common, identity


def _source_identity(sample_dev: pd.DataFrame, sample_holdout: pd.DataFrame, previous: dict) -> dict:
    supporting_code = {
        "runner": Path(__file__),
        "config": CONFIG_PATH,
        "prior_experiment_runner": Path(source_experiment.__file__),
        "probability_dataset_helpers": Path(probability.__file__),
        "research_source_loader": Path(research.__file__),
        "fee_source_verifier": Path(fee_correction.__file__),
        "t59_execution_engine": Path(replay.__file__),
    }
    code_hashes = {name: _sha256(path) for name, path in supporting_code.items()}
    source_hashes = {
        "source_experiment_config": _sha256(source_experiment.CONFIG_PATH),
        "btc_feature_dataset": source_experiment.CONFIG["source_artifacts"]["btc_feature_dataset_sha256"],
        "official_market_index": source_experiment.CONFIG["source_artifacts"]["official_market_index_sha256"],
        "t60_snapshot_coverage": source_experiment.CONFIG["source_artifacts"]["t60_snapshot_sha256"],
        "t59_snapshot_coverage": source_experiment.CONFIG["source_artifacts"]["t59_snapshot_sha256"],
        "walk_forward_prediction_manifest": source_experiment.CONFIG["source_artifacts"]["development_walk_forward_manifest_sha256"],
        "development_A_frozen_oof": source_experiment.CONFIG["source_artifacts"]["development_oof_sha256"],
        "holdout_A_frozen_predictions": source_experiment.CONFIG["source_artifacts"]["holdout_predictions_sha256"],
        **previous,
    }
    sample_hashes = {
        "development_all_fit_rows": _sample_sha256(sample_dev, M1_FEATURES),
        "holdout_common_rows": _sample_sha256(sample_holdout, M1_FEATURES),
    }
    identity = {
        "experiment_id": CONFIG["experiment_id"],
        "config_sha256": code_hashes["config"],
        "code_sha256": code_hashes,
        "source_sha256": source_hashes,
        "sample_sha256": sample_hashes,
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def _update_status(status: str, next_step: str) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = STATUS_PATH.read_text(encoding="utf-8")
    start = text.find(STATUS_MARKER)
    section = f"{STATUS_MARKER}\n\n- Status: {status}\n- Next resumable step: {next_step}\n"
    if start < 0:
        updated = text.rstrip() + "\n\n" + section
    else:
        next_heading = text.find("\n## ", start + len(STATUS_MARKER))
        end = next_heading if next_heading >= 0 else len(text)
        updated = text[:start] + section.rstrip() + text[end:]
    STATUS_PATH.write_text(updated, encoding="utf-8")


def _select_configurations(identity: dict, dev: pd.DataFrame) -> tuple[dict, pd.DataFrame, list[dict]]:
    folds = []
    for fold_config in CONFIG["folds"]:
        train, valid = _fold_slices(dev, fold_config)
        folds.append((fold_config, train, valid))
    all_results = []
    winner_predictions = {}
    winners = {}
    for model_name in ("M0", "M1"):
        features = M0_FEATURES if model_name == "M0" else M1_FEATURES
        scored = []
        for config in CONFIG["configurations"]:
            fitted = _fit_configuration(model_name, config)
            fold_records, fold_predictions = [], []
            for fold_config, train, valid in folds:
                booster, predictions, fit = _run_fit(
                    identity, model_name, fitted, fold_config["fold_id"], train, valid,
                )
                _ = booster
                fold_loss = _row_log_loss(predictions.official_outcome_up.to_numpy(), predictions[f"p_{model_name}"].to_numpy())
                fold_records.append({
                    "fold_id": fold_config["fold_id"], "training_rows": len(train),
                    "validation_rows": len(valid), "log_loss": fold_loss,
                    "latest_training_label_available_at_utc": train.label_available_at_utc.max().isoformat(),
                    "fit": fit,
                })
                fold_predictions.append(predictions)
            losses = [item["log_loss"] for item in fold_records]
            mean_loss = float(np.mean(losses))
            fold_sd = float(np.std(losses, ddof=0))
            result = {
                **fitted,
                "folds": fold_records,
                "fold_log_losses": losses,
                "mean_fold_log_loss": mean_loss,
                "population_sd_fold_log_loss": fold_sd,
                "selection_score": mean_loss + 0.5 * fold_sd,
            }
            scored.append((result, fold_predictions))
        scored.sort(key=lambda item: (
            item[0]["selection_score"], item[0]["num_leaves"], -item[0]["lambda_l2"],
        ))
        winner = scored[0][0]
        winners[model_name] = winner
        winner_predictions[model_name] = pd.concat(scored[0][1], ignore_index=True).sort_values(
            "market_start_utc", kind="stable",
        )
        all_results.extend(sorted((item[0] for item in scored), key=lambda row: (row["num_leaves"], -row["lambda_l2"])))
        _write_json(OUTPUT_DIR / "configuration_results.json", {
            "identity_sha256": identity["identity_sha256"],
            "completed_models": list(winners), "winners": winners, "configurations": all_results,
        })
        _write_parquet(OUTPUT_DIR / "development_oof_predictions.parquet", _merge_winner_predictions(winner_predictions))
        if len(winners) < 2:
            status = f"Fitting complete for {model_name}; {len(winners)}/2 families selected, all fit checkpoints are resumable."
            next_step = "Finish the remaining family/configurations, fit selected full-development models, freeze the manifest, then score the holdout once."
        else:
            status = "All 16 validation fits are complete and both families have development-selected configurations."
            next_step = "Write the pre-holdout manifest, fit each winner on all qualifying development rows, then score the holdout once."
        _update_status(status, next_step)
        print(f"[interaction] {model_name} selected {winner['config_id']} with score={winner['selection_score']:.8f}", flush=True)
    return winners, _merge_winner_predictions(winner_predictions), all_results


def _merge_winner_predictions(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame()
    names = list(predictions)
    merged = predictions[names[0]].rename(columns={"raw_correction": f"raw_{names[0]}_correction"})
    for name in names[1:]:
        other = predictions[name].rename(columns={"raw_correction": f"raw_{name}_correction"})
        merged = merged.merge(
            other[["market_start_utc", "condition_id", "raw_correction" if "raw_correction" in other else f"raw_{name}_correction", f"p_{name}"]],
            on=["market_start_utc", "condition_id"], how="inner", validate="one_to_one",
        )
        if "raw_correction" in merged.columns:
            merged = merged.rename(columns={"raw_correction": f"raw_{name}_correction"})
    return merged


def _load_fit_or_train_final(identity: dict, model_name: str, winner: dict, dev: pd.DataFrame) -> tuple[lgb.Booster, dict]:
    config = {key: winner[key] for key in ("num_leaves", "max_depth", "lambda_l2")}
    config.update({"config_id": winner["config_id"]})
    booster, _, fit = _run_fit(identity, model_name, config, "final_development", dev, None)
    return booster, fit


def _paired_block_differences(frame: pd.DataFrame, left_column: str, right_column: str) -> dict:
    ordered = frame.sort_values("market_start_utc", kind="stable").reset_index(drop=True)
    y = ordered.official_outcome_up.to_numpy(dtype=np.int8)
    left = np.clip(ordered[left_column].to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
    right = np.clip(ordered[right_column].to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
    left_loss = -(y * np.log(left) + (1 - y) * np.log1p(-left))
    right_loss = -(y * np.log(right) + (1 - y) * np.log1p(-right))
    left_brier = np.square(left - y)
    right_brier = np.square(right - y)
    day_series = ordered.market_start_utc.dt.floor("D")
    days = pd.date_range(day_series.min(), day_series.max(), freq="D")
    positions = [np.flatnonzero(day_series.eq(day).to_numpy()) for day in days]
    block_length = min(3, len(days))
    blocks_per_sample = int(math.ceil(len(days) / block_length))
    rng = np.random.default_rng(int(CONFIG["quality_uncertainty"]["seed"]))
    ll_draws, brier_draws = [], []
    empty_resamples = 0
    attempts = 0
    replications = int(CONFIG["quality_uncertainty"]["replications"])
    while len(ll_draws) < replications:
        attempts += 1
        if attempts > replications * 100:
            raise RuntimeError("Too many empty-only UTC block-bootstrap samples")
        max_start = max(len(days) - block_length + 1, 1)
        starts = rng.integers(0, max_start, size=blocks_per_sample)
        sampled_days = [
            day_index
            for start in starts
            for day_index in range(int(start), min(int(start) + block_length, len(days)))
        ][:len(days)]
        sample_index = np.concatenate([positions[index] for index in sampled_days])
        if sample_index.size == 0:
            empty_resamples += 1
            continue
        ll_draws.append(float((left_loss[sample_index] - right_loss[sample_index]).mean()))
        brier_draws.append(float((left_brier[sample_index] - right_brier[sample_index]).mean()))
    return {
        "n": len(ordered),
        "log_loss_difference_left_minus_right": float((left_loss - right_loss).mean()),
        "log_loss_difference_95pct": np.quantile(ll_draws, [0.025, 0.975]).tolist(),
        "brier_difference_left_minus_right": float((left_brier - right_brier).mean()),
        "brier_difference_95pct": np.quantile(brier_draws, [0.025, 0.975]).tolist(),
        "valid_bootstrap_replicates": len(ll_draws), "empty_only_resamples_discarded": empty_resamples,
        "method": "paired moving 3-day UTC blocks; 2,000 replicates; seed 20261017",
    }


def _quality_metrics(frame: pd.DataFrame, columns: dict[str, str]) -> tuple[list[dict], dict]:
    metrics = []
    for name, column in columns.items():
        p = np.clip(frame[column].to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
        y = frame.official_outcome_up.to_numpy(dtype=np.int8)
        metrics.append({
            "model": name, "n": len(frame), "log_loss": _row_log_loss(y, p),
            "brier": float(np.mean(np.square(y - p))),
        })
    comparisons = {}
    for left, right in (("M1", "M0"), ("M1", "previous_D"), ("M1", "p_market")):
        comparisons[f"{left}-{right}"] = _paired_block_differences(frame, columns[left], columns[right])
    return metrics, comparisons


def _attach_candidate_predictions(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions[["market_start_utc", "condition_id", "p_M0", "p_M1"]]
    result = frame.merge(selected, on=["market_start_utc", "condition_id"], how="inner", validate="one_to_one")
    if len(result) != len(predictions):
        raise RuntimeError("M0 and M1 validation predictions do not share the same comparison rows")
    return result


def _adverse_ask_snapshot(snapshot: dict, side: str, bps: float) -> dict:
    adjusted = dict(snapshot)
    factor = 1.0 + float(bps) / 10000.0
    levels = replay._ask_levels(snapshot.get(f"{side}_ask_levels"))
    bumped = {}
    for price, size in levels.items():
        adverse_price = round(float(price) * factor, 8)
        if 0.0 < adverse_price < 1.0:
            bumped[adverse_price] = bumped.get(adverse_price, 0.0) + float(size)
    adjusted[f"{side}_ask_levels"] = list(bumped.items())
    best_ask = pd.to_numeric(pd.Series([snapshot.get(f"{side}_best_ask")]), errors="coerce").iloc[0]
    if pd.notna(best_ask):
        adjusted[f"{side}_best_ask"] = float(best_ask) * factor
    return adjusted


def _economic_evaluation(frame: pd.DataFrame, t59: pd.DataFrame, t60: pd.DataFrame, cutoff: pd.Timestamp, period: str) -> dict:
    t59_by_id, t60_by_id = research._snapshot_map(t59), research._snapshot_map(t60)
    enriched, _ = replay._market_statuses(frame, t59, t60)
    market_by_id = research._snapshot_map(enriched.loc[enriched.condition_id.notna()])
    model_columns = {"p_market": "p_market", "previous_D": "p_D_conditional", "M0": "p_M0", "M1": "p_M1"}
    results = {}
    decisions_by_model: dict[str, list[dict]] = {}
    slippage_bps = (0, 5, 10, 25)
    for model_name, column in model_columns.items():
        independent, decisions = [], []
        slippage = {
            bps: {"fills": 0, "misses": 0, "turnover": 0.0, "fees": 0.0, "pnl": 0.0, "full_book_fills": 0, "proxy_fills": 0}
            for bps in slippage_bps
        }
        priceable_markets = positive_ev_markets = qualified_markets = positive_ev_sides = 0
        missing_t60 = missing_t59_for_qualified = 0
        for row in frame.itertuples(index=False):
            cid = str(row.condition_id).lower()
            market = {**market_by_id[cid], "p_candidate_platt": float(getattr(row, column))}
            t60_snapshot, t59_snapshot = t60_by_id.get(cid), t59_by_id.get(cid)
            if t60_snapshot is None:
                missing_t60 += 1
                decisions.append({"condition_id": cid, "market_start_utc": row.market_start_utc, "model": model_name, "priceable": False, "positive_ev": False, "ev_over_1pct": False, "fill": False, "reason": "missing_t60_snapshot"})
                continue
            decision = replay._evaluate_market_sides(market, t60_snapshot, 5.0)
            priceable = int(decision["priceable_side_count"]) > 0
            positive = int(decision["positive_ev_side_count"]) > 0
            priceable_markets += int(priceable)
            positive_ev_markets += int(positive)
            positive_ev_sides += int(decision["positive_ev_side_count"])
            side = decision["chosen_side"]
            ev_fraction = float(decision["chosen_ev_usd"]) / 5.0 if side is not None else None
            qualified = ev_fraction is not None and ev_fraction + 1e-12 >= 0.01
            qualified_markets += int(qualified)
            fill = None
            reason = "no_positive_net_ev_at_t60" if side is None else "below_1pct_net_ev" if not qualified else None
            if qualified:
                if t59_snapshot is None:
                    missing_t59_for_qualified += 1
                    reason = "missing_t59_snapshot"
                else:
                    execution = replay._price_side(t59_snapshot, market, side, 5.0)
                    fill = execution.get("fill") if execution["priceable"] else None
                    if fill is None:
                        reason = str(execution.get("reason", "t59_unpriceable"))
                if fill is not None:
                    label_release = pd.Timestamp(row.label_available_at_utc)
                    settled = label_release <= cutoff
                    won = int(row.official_outcome_up) == int(side == "up")
                    pnl = float(fill["shares"] if won else 0.0) - float(fill["cash_debit_usd"]) if settled else None
                    independent.append({
                        "condition_id": cid, "market_start_utc": row.market_start_utc,
                        "model": model_name, "side": side, "t60_net_ev_fraction": ev_fraction,
                        "requested_gross_usd": 5.0, "gross_usd": float(fill["gross_usd"]),
                        "cash_debit_usd": float(fill["cash_debit_usd"]), "net_shares": float(fill["shares"]),
                        "fee_usd": float(fill.get("fee_usd") or 0.0), "settled_by_cutoff": bool(settled),
                        "net_pnl_usd": pnl, "depth_known": bool(fill.get("depth_known")),
                        "execution_source": fill.get("execution_source"),
                        "execution_assumption": fill.get("execution_assumption"),
                        "fill_status": fill.get("fill_status"),
                    })
                    for bps in slippage_bps:
                        bumped_snapshot = t59_snapshot if bps == 0 else _adverse_ask_snapshot(t59_snapshot, side, bps)
                        stressed = replay._price_side(bumped_snapshot, market, side, 5.0)
                        stressed_fill = stressed.get("fill") if stressed["priceable"] else None
                        bucket = slippage[bps]
                        if stressed_fill is None:
                            bucket["misses"] += 1
                            continue
                        bucket["fills"] += 1
                        bucket["turnover"] += float(stressed_fill["gross_usd"])
                        bucket["fees"] += float(stressed_fill.get("fee_usd") or 0.0)
                        if bool(stressed_fill.get("depth_known")):
                            bucket["full_book_fills"] += 1
                        else:
                            bucket["proxy_fills"] += 1
                        if settled:
                            bucket["pnl"] += float(stressed_fill["shares"] if won else 0.0) - float(stressed_fill["cash_debit_usd"])
            decisions.append({
                "condition_id": cid, "market_start_utc": row.market_start_utc, "model": model_name,
                "priceable": priceable, "positive_ev": positive, "ev_over_1pct": qualified,
                "fill": fill is not None, "chosen_side": side, "ev_fraction": ev_fraction, "reason": reason,
            })
        ledger = pd.DataFrame(independent)
        settled = ledger.loc[ledger.settled_by_cutoff] if len(ledger) else ledger
        turnover = float(settled.gross_usd.sum()) if len(settled) else 0.0
        pnl = float(settled.net_pnl_usd.sum()) if len(settled) else 0.0
        depth_groups = {}
        if len(settled):
            for depth_known, group in settled.groupby("depth_known", dropna=False):
                depth_groups["full_book" if bool(depth_known) else "top_of_book_proxy"] = {
                    "resolved_fills": int(len(group)),
                    "gross_turnover_usd": float(group.gross_usd.sum()),
                    "filled_shares": float(group.net_shares.sum()),
                    "fees_usd": float(group.fee_usd.sum()),
                    "net_pnl_usd": float(group.net_pnl_usd.sum()),
                    "wins": int(group.net_pnl_usd.gt(0.0).sum()),
                    "losses": int(group.net_pnl_usd.le(0.0).sum()),
                }
        results[model_name] = {
            "quality_rows": len(frame), "priceable_markets_at_t60": priceable_markets,
            "positive_ev_markets_at_t60": positive_ev_markets,
            "decisions_meeting_1pct_net_ev": qualified_markets,
            "fills_at_t59": int(len(ledger)),
            "resolved_fills_by_cutoff": int(len(settled)),
            "unresolved_fills_at_cutoff": int(len(ledger) - len(settled)),
            "gross_turnover_usd": turnover, "fees_usd": float(settled.fee_usd.sum()) if len(settled) else 0.0,
            "net_pnl_usd": pnl, "net_pnl_over_turnover": pnl / turnover if turnover else None,
            "profitable_after_costs": pnl > 0.0, "depth_breakdown": depth_groups,
            "incremental_adverse_cost_to_zero_usd": max(pnl, 0.0),
            "incremental_adverse_cost_to_zero_bps_of_turnover": max(pnl / turnover, 0.0) * 10000.0 if turnover else None,
            "additional_cost_sensitivity": [
                {"adverse_cost_bps_of_turnover": bps, "stressed_net_pnl_usd": pnl - turnover * bps / 10000.0}
                for bps in (0, 5, 10, 25)
            ],
            "adverse_price_sensitivity": [
                {"adverse_price_bps": bps, "fills": values["fills"], "misses": values["misses"],
                 "gross_turnover_usd": values["turnover"], "fees_usd": values["fees"],
                 "net_pnl_usd": values["pnl"], "net_pnl_over_turnover": values["pnl"] / values["turnover"] if values["turnover"] else None,
                 "full_book_fills": values["full_book_fills"], "top_of_book_proxy_fills": values["proxy_fills"]}
                for bps, values in slippage.items()
            ],
            "positive_ev_sides_at_t60": positive_ev_sides,
            "missing_t60_snapshots": missing_t60,
            "qualified_decisions_missing_t59": missing_t59_for_qualified,
            "fills_with_unknown_depth": int((~ledger.depth_known).sum()) if len(ledger) else 0,
            "ledger_path": str((OUTPUT_DIR / "economics" / period / f"{model_name}_fixed5.parquet").relative_to(ROOT)),
        }
        if len(ledger):
            _write_parquet(OUTPUT_DIR / "economics" / period / f"{model_name}_fixed5.parquet", ledger)
        decisions_by_model[model_name] = decisions

    day_start = frame.market_start_utc.min().floor("D")
    day_end = frame.market_start_utc.max().floor("D")
    calendar = pd.date_range(day_start, day_end, freq="D")
    daily = {name: pd.Series(0.0, index=calendar) for name in model_columns}
    for name in model_columns:
        path = OUTPUT_DIR / "economics" / period / f"{name}_fixed5.parquet"
        ledger = pd.read_parquet(path) if path.is_file() else pd.DataFrame()
        if len(ledger):
            settled = ledger.loc[ledger.settled_by_cutoff & ledger.net_pnl_usd.notna()].copy()
            if len(settled):
                settled["day_utc"] = pd.to_datetime(settled.market_start_utc, utc=True).dt.floor("D")
                daily[name] = settled.groupby("day_utc").net_pnl_usd.sum().reindex(calendar, fill_value=0.0)
    daily_differences = {}
    for period_name in ("M1-M0", "M1-previous_D", "M1-p_market"):
        left, right = period_name.split("-")
        diff = daily[left] - daily[right]
        daily_differences[period_name] = {
            "calendar_days_including_no_trade_days": len(calendar),
            "total_pnl_difference_usd": float(diff.sum()),
            "mean_daily_pnl_difference_usd": float(diff.mean()),
            "days_with_any_fill_in_either_policy": int((daily[left].ne(0.0) | daily[right].ne(0.0)).sum()),
        }
    _write_json(OUTPUT_DIR / "economics" / period / "decisions_summary.json", {
        "period": period, "cutoff_exclusive_utc": cutoff.isoformat(), "decision_threshold_net_ev_fraction": 0.01,
        "decisions": {name: {"rows": len(value), "positive_ev_market_count": results[name]["positive_ev_markets_at_t60"],
                             "qualifying_market_count": results[name]["decisions_meeting_1pct_net_ev"]}
                      for name, value in decisions_by_model.items()},
    })
    return {
        "period": period, "settlement_cutoff_exclusive_utc": cutoff.isoformat(),
        "stake_usd": 5.0, "minimum_net_ev_fraction": 0.01, "models": results,
        "paired_daily_pnl_differences": daily_differences,
        "daily_pnl_zero_filled": True,
        "note": "T-60 net EV already includes the existing fee treatment; fees are not subtracted from EV again.",
    }


def _portfolio_control(frame: pd.DataFrame, t59: pd.DataFrame, t60: pd.DataFrame, cutoff: pd.Timestamp, period: str) -> dict:
    t59_by_id, t60_by_id = research._snapshot_map(t59), research._snapshot_map(t60)
    enriched, _ = replay._market_statuses(frame, t59, t60)
    market_by_id = research._snapshot_map(enriched.loc[enriched.condition_id.notna()])
    models = {"p_market": "p_market", "previous_D": "p_D_conditional", "M0": "p_M0", "M1": "p_M1"}
    result = {}
    for name, column in models.items():
        rows = frame.copy()
        rows["p_candidate_platt"] = rows[column].astype(float)
        rows["resolved_at_utc"] = pd.to_datetime(rows.resolved_at_utc, utc=True, errors="coerce")
        rows.loc[rows.resolved_at_utc.isna(), "resolved_at_utc"] = cutoff
        variant_market = {
            str(row.condition_id).lower(): {**market_by_id[str(row.condition_id).lower()], "p_candidate_platt": float(row.p_candidate_platt)}
            for row in rows.itertuples(index=False)
        }
        _, trades, decisions = replay._portfolio_for_policy(
            "free_cash_5pct", rows, t59_by_id, variant_market,
            decision_snapshot_by_id=t60_by_id, minimum_net_ev_fraction=0.01,
            output_policy=f"{period}_{name}_interaction_control",
        )
        trade_frame = pd.DataFrame(trades)
        if len(trade_frame):
            trade_frame["cash_release_at_utc"] = pd.to_datetime(trade_frame.cash_release_at_utc, utc=True, errors="raise")
            settled = trade_frame.loc[trade_frame.cash_release_at_utc.le(cutoff)]
            unsettled = trade_frame.loc[trade_frame.cash_release_at_utc.gt(cutoff)]
            turnover = float(settled.gross_usd.sum())
            pnl = float((settled.payout_usd - settled.cash_debit_usd).sum())
            fees = float(settled.fee_usd.sum())
            cash_at_cutoff = 100.0 - float(trade_frame.cash_debit_usd.sum()) + float(settled.payout_usd.sum())
            locked_gross = float(unsettled.gross_usd.sum())
        else:
            settled, turnover, pnl, fees, cash_at_cutoff, locked_gross = trade_frame, 0.0, 0.0, 0.0, 100.0, 0.0
        count_qualified = sum(
            item.get("net_ev_fraction_at_decision") is not None
            and pd.notna(item.get("net_ev_fraction_at_decision"))
            and float(item["net_ev_fraction_at_decision"]) + 1e-12 >= 0.01
            for item in decisions
        )
        result[name] = {
            "initial_cash_usd": 100.0, "stake_rule": "5% of free cash at T-60", "decision_rows": len(decisions),
            "decisions_meeting_1pct_net_ev": int(count_qualified), "entries_filled": len(trade_frame),
            "settled_fills_by_cutoff": len(settled), "gross_turnover_usd": turnover,
            "fees_usd": fees, "net_pnl_usd": pnl,
            "net_pnl_over_turnover": pnl / turnover if turnover else None,
            "cash_at_cutoff_usd": cash_at_cutoff,
            "unsettled_locked_gross_usd": locked_gross,
            "cash_plus_locked_gross_usd": cash_at_cutoff + locked_gross,
        }
        if len(trade_frame):
            _write_parquet(OUTPUT_DIR / "economics" / period / f"{name}_free_cash_5pct_trades.parquet", trade_frame)
        if decisions:
            _write_parquet(OUTPUT_DIR / "economics" / period / f"{name}_free_cash_5pct_decisions.parquet", pd.DataFrame(decisions))
    return {"period": period, "cutoff_exclusive_utc": cutoff.isoformat(), "models": result}


def _tree_path_diagnostics(booster: lgb.Booster) -> dict:
    names = list(M1_FEATURES)
    dump = booster.dump_model()
    path_results = {"d_and_basis": {"paths": 0, "trees": set()}, "d_and_sigma": {"paths": 0, "trees": set()}}
    for tree_index, tree in enumerate(dump["tree_info"]):
        def visit(node: dict, seen: tuple[str, ...]) -> None:
            if "split_feature" not in node:
                features = set(seen)
                for key, pair in (("d_and_basis", {"d_btc_minus_market", "futures_index_basis_rel_1m"}),
                                  ("d_and_sigma", {"d_btc_minus_market", "btc_sigma_5m"})):
                    if pair.issubset(features):
                        path_results[key]["paths"] += 1
                        path_results[key]["trees"].add(tree_index)
                return
            feature = names[int(node["split_feature"])]
            next_seen = (*seen, feature)
            visit(node["left_child"], next_seen)
            visit(node["right_child"], next_seen)
        visit(tree["tree_structure"], ())
    for value in path_results.values():
        value["tree_count"] = len(value.pop("trees"))
    gains = booster.feature_importance(importance_type="gain").astype(float)
    splits = booster.feature_importance(importance_type="split").astype(int)
    feature_gain = [
        {"feature": feature, "split_count": int(split), "total_gain": float(gain),
         "gain_share": float(gain / gains.sum()) if gains.sum() else 0.0}
        for feature, split, gain in zip(names, splits, gains)
    ]
    return {"tree_count": len(dump["tree_info"]), "feature_importance": feature_gain, "root_to_leaf_interactions": path_results}


def _interaction_grid_definition(dev: pd.DataFrame) -> dict:
    points = [0.05, 0.25, 0.50, 0.75, 0.95]
    quantiles = {
        feature: [float(x) for x in np.quantile(dev[feature].to_numpy(dtype=np.float64), points)]
        for feature in ("d_btc_minus_market", "futures_index_basis_rel_1m", "btc_sigma_5m")
    }
    fixed = {
        "logit_p_market": float(dev.logit_p_market.median()),
        "spread_up": float(dev.spread_up.median()),
        "spread_down": float(dev.spread_down.median()),
    }
    return {"quantile_levels": points, "development_quantiles": quantiles, "fixed_other_features": fixed}


def _interaction_grids(booster: lgb.Booster, definition: dict) -> pd.DataFrame:
    quantiles = definition["development_quantiles"]
    fixed = definition["fixed_other_features"]
    rows = []
    for basis in quantiles["futures_index_basis_rel_1m"]:
        for d in quantiles["d_btc_minus_market"]:
            row = {**fixed, "d_btc_minus_market": d, "futures_index_basis_rel_1m": basis,
                   "btc_sigma_5m": quantiles["btc_sigma_5m"][2]}
            correction = float(booster.predict(pd.DataFrame([row], columns=M1_FEATURES).to_numpy(dtype=np.float64), raw_score=True)[0])
            rows.append({"surface": "d_by_basis_at_median_sigma", **row, "predicted_logit_correction": correction})
    for sigma in quantiles["btc_sigma_5m"]:
        for d in quantiles["d_btc_minus_market"]:
            row = {**fixed, "d_btc_minus_market": d,
                   "futures_index_basis_rel_1m": quantiles["futures_index_basis_rel_1m"][2], "btc_sigma_5m": sigma}
            correction = float(booster.predict(pd.DataFrame([row], columns=M1_FEATURES).to_numpy(dtype=np.float64), raw_score=True)[0])
            rows.append({"surface": "d_by_sigma_at_median_basis", **row, "predicted_logit_correction": correction})
    return pd.DataFrame(rows)


def _pre_holdout_manifest(identity: dict, winners: dict, dev_predictions: pd.DataFrame, grid_definition: dict) -> dict:
    return {
        "schema_version": 1,
        "status": "frozen_before_holdout_scoring",
        "experiment_id": CONFIG["experiment_id"],
        "identity": identity,
        "winner_configurations": {
            name: {key: winners[name][key] for key in ("config_id", "num_leaves", "max_depth", "lambda_l2", "fold_log_losses", "selection_score")}
            for name in ("M0", "M1")
        },
        "feature_lists": {"M0": list(M0_FEATURES), "M1": list(M1_FEATURES)},
        "selection_criterion": CONFIG["selection"],
        "training": CONFIG["training"],
        "economic_rules": CONFIG["economics"],
        "development_cutoff_exclusive_utc": DEV_CUTOFF.isoformat(),
        "holdout_cutoff_exclusive_utc": HOLDOUT_CUTOFF.isoformat(),
        "holdout_historically_exposed": True,
        "development_interaction_grid_definition": grid_definition,
        "development_oof_predictions_sha256": _sha256(OUTPUT_DIR / "development_oof_predictions.parquet"),
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def _metric_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {"p_market": "p_market", "previous_D": "p_D_conditional", "M0": "p_M0", "M1": "p_M1"}


def _write_grid_and_report(results: dict, grids: pd.DataFrame, grid_definition: dict) -> None:
    grid_path = REPORT_DIR / "interaction_grids.csv"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    grids.to_csv(grid_path, index=False, float_format="%.12g")
    results["interaction_grid_definition"] = grid_definition
    results["interaction_grid_path"] = str(grid_path.relative_to(ROOT))
    _write_json(OUTPUT_DIR / "results.json", results)
    report = _render_report(results, grid_path)
    REPORT_PATH.write_text(report, encoding="utf-8")


def _render_report(payload: dict, grid_path: Path) -> str:
    lines = [
        "# BTC-market interaction correction — 2026-10-17", "",
        "> The July–October holdout had been examined in earlier repository work. Its result here is descriptive and was not used for feature, configuration, or policy selection.", "",
        "## Question and protocol", "",
        "M0 is a market-offset correction using the market logit, BTC index-futures basis, BTC volatility, and both T-60 spreads. M0 therefore retains BTC information through basis and volatility. M1 adds only `d = logit(p_BTC_A_frozen) − logit(p_market)`. The primary comparison is M1−M0.", "",
        "Both models use `p = sigmoid(logit(p_market) + f(X))`, a row-aligned LightGBM `init_score`, official Polymarket outcomes, binary log-loss, 100 iterations, and the same complete rows and purged May/June folds. The fitted tree correction is raw margin; the market offset is added exactly once at prediction. No weights, monotonic constraints, early stopping, Optuna, generator rerun, BTC model/OOF refit, data fetch, or archive replay were used.", "",
        "## Inputs and freeze", "",
        f"Common prediction rows: {payload['sample']['development_rows']:,} development OOF markets and {payload['sample']['holdout_rows']:,} holdout markets. Development starts are before {DEV_CUTOFF.isoformat()}; holdout starts are before {HOLDOUT_CUTOFF.isoformat()}. All predictions and quality metrics use rows shared with the previous frozen D table.", "",
        f"The pre-holdout manifest was frozen at `{payload['pre_holdout_manifest']['frozen_at_utc']}` before holdout prediction scoring. Manifest SHA-256: `{payload['pre_holdout_manifest_sha256']}`. Input/code identities and winning settings are recorded in `data/analysis/polymarket/BTC/btc_market_basis_vol_interaction_20261017/pre_holdout_manifest.json`.", "",
        *([f"Reporting resumed at `{payload['reporting_runner_sha256']}` after the first report attempt hit empty-only June calendar bootstrap draws. Each empty-only draw was discarded; all comparisons then used 2,000 valid replicates. The frozen holdout predictions were reused from their one-time score at `{payload['holdout_scored_at_utc']}`, with the original final-model and prediction hashes intact. The frozen manifest retains the original runner hash.", ""] if payload.get("post_freeze_reporting_resume") else []),
        "BTC probabilities came from the existing causal A_frozen development OOF predictions and frozen holdout predictions, validated against the saved walk-forward manifest and hashes in the prior experiment config. They were not regenerated or replaced. The previous D comparator uses the saved development and holdout prediction tables; their hashes are recorded in the pre-holdout manifest.", "",
        "### Frozen periods and folds", "",
        "| Period/fold | Markets or train rows | Validation rows | First market | Last market | Latest eligible train-label release |", "|---|---:|---:|---|---|---|",
    ]
    for item in payload["sample"]["folds"]:
        if "training_rows" in item:
            lines.append(
                f"| {item['period']} | {item['training_rows']:,} | {item['validation_rows']:,} | "
                f"{item['validation_start_utc']} | {item['validation_end_exclusive_utc']} | "
                f"{item['latest_training_label_available_at_utc']} |"
            )
        else:
            lines.append(
                f"| {item['period']} | {item['markets']:,} | — | {item['first_market_utc']} | "
                f"{item['last_market_utc']} | {item['latest_label_available_at_utc']} |"
            )
    lines += ["", "The May and June validation folds use the frozen monthly boundaries. Training requires `market_start_utc < validation_start_utc` and `label_available_at_utc <= validation_start_utc`; all variants share the same complete-input mask.", "", "## Development configuration results", "", "Selection minimizes mean fold log-loss + 0.5 × population SD. Every configuration is shown; scores are development results after configuration selection.", "", "| Model | Leaves | Depth | L2 | May log-loss | June log-loss | Mean + 0.5 SD | Selected |", "|---|---:|---:|---:|---:|---:|---:|:---:|"]
    winners = payload["winner_configurations"]
    for item in payload["configuration_results"]:
        fold = {record["fold_id"]: record["log_loss"] for record in item["folds"]}
        selected = item["config_id"] == winners[item["model"]]["config_id"]
        lines.append(f"| {item['model']} | {item['num_leaves']} | {item['max_depth']} | {item['lambda_l2']:.0f} | {fold['dev_2026q2_b']:.8f} | {fold['dev_2026q2_c']:.8f} | {item['selection_score']:.8f} | {'yes' if selected else ''} |")
    lines += ["", "## Probability quality", "", "Negative paired differences favor the left-hand model. Intervals use paired moving 3-day UTC blocks, 2,000 replicates, seed 20261017. The selected development folds are not independent test results; their intervals are conditional on the selected predictions and do not remove selection optimism.", ""]
    for period_name, metrics in payload["quality_metrics"].items():
        lines += [f"### {period_name}", "", "| Model | n | Log-loss | Brier |", "|---|---:|---:|---:|"]
        for row in metrics["metrics"]:
            lines.append(f"| {row['model']} | {row['n']:,} | {row['log_loss']:.8f} | {row['brier']:.8f} |")
        lines += ["", "| Paired difference | Δ log-loss | 95% block CI | Δ Brier | 95% block CI |", "|---|---:|---:|---:|---:|"]
        for comparison, values in metrics["paired_differences"].items():
            lines.append(f"| {comparison} | {values['log_loss_difference_left_minus_right']:+.8f} | [{values['log_loss_difference_95pct'][0]:+.8f}, {values['log_loss_difference_95pct'][1]:+.8f}] | {values['brier_difference_left_minus_right']:+.8f} | [{values['brier_difference_95pct'][0]:+.8f}, {values['brier_difference_95pct'][1]:+.8f}] |")
        lines.append("")
    lines += ["## Fixed-$5 economics", "", "Decision and EV eligibility use the causal T-60 snapshot and existing fee logic. A qualified order is executed from the actual T-59 snapshot using existing depth/full-book or top-of-book proxy behavior. PnL uses fills settled by the period cutoff; fees are already included in cash debit and are not deducted from EV a second time.", ""]
    for period_name, section in payload["economics_fixed5"].items():
        lines += [f"### {period_name} (settlement cutoff {section['settlement_cutoff_exclusive_utc']})", "", "| Model | Priceable at T-60 | Positive-EV markets | Positive-EV sides | EV ≥ 1% | T-59 fills | Settled | Turnover | Fees | Net PnL | PnL/turnover |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, row in section["models"].items():
            ratio = "—" if row["net_pnl_over_turnover"] is None else f"{row['net_pnl_over_turnover']:.3%}"
            lines.append(f"| {name} | {row['priceable_markets_at_t60']:,} | {row['positive_ev_markets_at_t60']:,} | {row['positive_ev_sides_at_t60']:,} | {row['decisions_meeting_1pct_net_ev']:,} | {row['fills_at_t59']:,} | {row['resolved_fills_by_cutoff']:,} | ${row['gross_turnover_usd']:,.2f} | ${row['fees_usd']:,.4f} | ${row['net_pnl_usd']:+,.4f} | {ratio} |")
        lines += ["", "Resolved fills by execution depth:", "", "| Model | Depth source | Fills | Shares | Turnover | Fees | Wins | Losses | Net PnL |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, row in section["models"].items():
            for depth, values in row["depth_breakdown"].items():
                lines.append(f"| {name} | {depth} | {values['resolved_fills']} | {values['filled_shares']:.4f} | ${values['gross_turnover_usd']:.2f} | ${values['fees_usd']:.4f} | {values['wins']} | {values['losses']} | ${values['net_pnl_usd']:+.4f} |")
        lines += ["", "Incremental adverse execution cost stress, applied to settled gross turnover after the modeled fee treatment:", "", "| Model | 0 bps | 5 bps | 10 bps | 25 bps | Extra cost to erase positive PnL |", "|---|---:|---:|---:|---:|---:|"]
        for name, row in section["models"].items():
            stress = {item['adverse_cost_bps_of_turnover']: item['stressed_net_pnl_usd'] for item in row['additional_cost_sensitivity']}
            lines.append(f"| {name} | ${stress[0]:+.4f} | ${stress[5]:+.4f} | ${stress[10]:+.4f} | ${stress[25]:+.4f} | ${row['incremental_adverse_cost_to_zero_usd']:.4f} ({row['incremental_adverse_cost_to_zero_bps_of_turnover'] or 0.0:.2f} bps) |")
        lines += ["", "T-59 ask prices repriced adversely while keeping the side and $5 order intent selected at T-60:", "", "| Model | 0 bps PnL (fills) | +5 bps | +10 bps | +25 bps |", "|---|---:|---:|---:|---:|"]
        for name, row in section["models"].items():
            stressed = {item["adverse_price_bps"]: item for item in row["adverse_price_sensitivity"]}
            cells = []
            for bps in (0, 5, 10, 25):
                value = stressed[bps]
                cells.append(f"${value['net_pnl_usd']:+.4f} ({value['fills']} fills, {value['misses']} misses)")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", "M1−M0 paired daily net-PnL difference: " + f"${section['paired_daily_pnl_differences']['M1-M0']['total_pnl_difference_usd']:+,.4f} total across {section['paired_daily_pnl_differences']['M1-M0']['calendar_days_including_no_trade_days']} calendar days, including days without fills.", ""]
    lines += ["## Required free-cash portfolio control", "", "This separate control starts each period with $100 and stakes 5% of free cash at T-60, subject to the existing $5 minimum-order fallback and cutoff settlement accounting.", ""]
    for period_name, section in payload["economics_free_cash_5pct"].items():
        lines += [f"### {period_name}", "", "| Model | Qualified decisions | Filled entries | Settled fills | Turnover | Fees | Net PnL | PnL/turnover | Cash at cutoff | Unsettled locked gross |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, row in section["models"].items():
            ratio = "—" if row["net_pnl_over_turnover"] is None else f"{row['net_pnl_over_turnover']:.3%}"
            lines.append(f"| {name} | {row['decisions_meeting_1pct_net_ev']:,} | {row['entries_filled']:,} | {row['settled_fills_by_cutoff']:,} | ${row['gross_turnover_usd']:,.2f} | ${row['fees_usd']:,.4f} | ${row['net_pnl_usd']:+,.4f} | {ratio} | ${row['cash_at_cutoff_usd']:,.4f} | ${row['unsettled_locked_gross_usd']:,.4f} |")
        lines.append("")
    mechanism = payload["tree_diagnostics"]
    lines += ["## What the selected M1 learned", "", "Feature gains describe split use; a feature gain by itself does not prove interaction. Interaction counts below require both features to appear along one root-to-leaf path in the same final development tree.", "", "| Feature | Splits | Gain | Gain share |", "|---|---:|---:|---:|"]
    for row in mechanism["feature_importance"]:
        lines.append(f"| {row['feature']} | {row['split_count']} | {row['total_gain']:.6f} | {row['gain_share']:.1%} |")
    lines += ["", "| Interaction pair on the same path | Paths | Trees with ≥1 path |", "|---|---:|---:|"]
    for key, label in (("d_and_basis", "d and basis"), ("d_and_sigma", "d and sigma")):
        row = mechanism["root_to_leaf_interactions"][key]
        lines.append(f"| {label} | {row['paths']} | {row['tree_count']} |")
    grid_rel = grid_path.relative_to(ROOT).as_posix()
    if payload.get("post_freeze_reporting_resume"):
        runtime_summary = (
            f"Successful reporting-resume invocation: {payload['runtime_seconds']:.2f} s; peak process working set: "
            f"{payload['peak_working_set_bytes'] / 1024**3:.3f} GiB. It reused all {payload['fit_count']} verified fit artifacts "
            "and the one-time holdout prediction; no fits or holdout predictions were repeated. The initial fit/score invocation "
            "stopped during report serialization, so its elapsed time was not captured."
        )
    else:
        runtime_summary = (
            f"Full measured end-to-end runtime: {payload['runtime_seconds']:.2f} s. Peak process working set: "
            f"{payload['peak_working_set_bytes'] / 1024**3:.3f} GiB. Exactly {payload['fit_count']} fits were completed "
            "(16 validation and 2 final development fits), one at a time."
        )
    lines += ["", f"The final-refit M1 correction surfaces use five development quantiles per axis (25 points each), with remaining features fixed at development medians. Full values are in [{grid_path.name}]({grid_rel}). These are model mechanism views, not causal effects; their ranges were fixed before looking at holdout predictions.", "",
              "## Descriptive holdout note", "", "The holdout was scored once after the manifest freeze. It was historically exposed before this experiment, so it is descriptive and cannot validate a new edge or guide a change in this task.", "",
              runtime_summary, "",
              "## Conclusions", ""]
    lines.extend(f"{index}. {answer}" for index, answer in enumerate(payload["conclusions"], start=1))
    lines += ["", f"Reproduce without CLI arguments: `python run_btc_market_basis_vol_interaction_20261017.py`. Prediction checkpoints and run details are under `{OUTPUT_DIR.relative_to(ROOT).as_posix()}`.", ""]
    return "\n".join(lines)


def _conclusions(payload: dict) -> list[str]:
    dev = payload["quality_metrics"]["development_cv_selected"]["paired_differences"]["M1-M0"]
    may = payload["quality_metrics"]["dev_2026q2_b"]["paired_differences"]["M1-M0"]
    june = payload["quality_metrics"]["dev_2026q2_c"]["paired_differences"]["M1-M0"]
    holdout = payload["quality_metrics"]["holdout_descriptive"]["paired_differences"]["M1-M0"]
    dev_ci = dev["log_loss_difference_95pct"]
    direction = "both" if may["log_loss_difference_left_minus_right"] < 0 and june["log_loss_difference_left_minus_right"] < 0 else "neither" if may["log_loss_difference_left_minus_right"] >= 0 and june["log_loss_difference_left_minus_right"] >= 0 else "only one"
    quality_text = (
        f"Development-selected M1−M0 Δ log-loss is {dev['log_loss_difference_left_minus_right']:+.8f} "
        f"(95% block CI [{dev_ci[0]:+.8f}, {dev_ci[1]:+.8f}]); Δ Brier is {dev['brier_difference_left_minus_right']:+.8f}. "
        f"The direction favors M1 in {direction} validation periods (May {may['log_loss_difference_left_minus_right']:+.8f}; June {june['log_loss_difference_left_minus_right']:+.8f}). "
        + ("The interval excludes zero." if dev_ci[0] > 0 or dev_ci[1] < 0 else "The interval includes zero, so the point change is not statistically clear under this block bootstrap.")
    )
    interaction = payload["tree_diagnostics"]["root_to_leaf_interactions"]
    interaction_text = (
        f"The final M1 trees put d with basis on a root-to-leaf path in {interaction['d_and_basis']['paths']} paths "
        f"({interaction['d_and_basis']['tree_count']} trees), and d with sigma in {interaction['d_and_sigma']['paths']} paths "
        f"({interaction['d_and_sigma']['tree_count']} trees)."
    )
    dev_economics = payload["economics_fixed5"]["development_cv_selected"]["models"]
    holdout_economics = payload["economics_fixed5"]["holdout_descriptive"]["models"]
    dev_m0, dev_m1 = dev_economics["M0"], dev_economics["M1"]
    holdout_m0, holdout_m1 = holdout_economics["M0"], holdout_economics["M1"]
    economics_text = (
        f"Development fixed-$5 M0 had {dev_m0['decisions_meeting_1pct_net_ev']} qualifying decisions, {dev_m0['fills_at_t59']} fills, and net PnL ${dev_m0['net_pnl_usd']:+.4f}; "
        f"M1 had {dev_m1['decisions_meeting_1pct_net_ev']} qualifying decisions, {dev_m1['fills_at_t59']} fills, and ${dev_m1['net_pnl_usd']:+.4f}. "
        f"On holdout M0 had {holdout_m0['decisions_meeting_1pct_net_ev']} qualifying decisions, {holdout_m0['fills_at_t59']} fills, and ${holdout_m0['net_pnl_usd']:+.4f}; "
        f"M1 had {holdout_m1['decisions_meeting_1pct_net_ev']} qualifying decisions, {holdout_m1['fills_at_t59']} fills, and ${holdout_m1['net_pnl_usd']:+.4f}. "
        f"M1 was {'profitable' if holdout_m1['net_pnl_usd'] > 0 else 'not profitable'} after modeled costs on holdout; this is descriptive."
    )
    holdout_text = (
        f"Descriptive holdout M1−M0 Δ log-loss is {holdout['log_loss_difference_left_minus_right']:+.8f} "
        f"(95% block CI [{holdout['log_loss_difference_95pct'][0]:+.8f}, {holdout['log_loss_difference_95pct'][1]:+.8f}]); "
        f"Δ Brier is {holdout['brier_difference_left_minus_right']:+.8f}. The holdout was previously viewed and did not influence any choice."
    )
    return [quality_text, interaction_text, economics_text, holdout_text]


def main() -> dict:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[interaction] loading frozen predictions, snapshots, labels, and cached candle features", flush=True)
    data, t59, t60, prediction_identity, book_record, _ = source_experiment._load_causal_inputs()
    enriched, feature_coverage = source_experiment._btc_features(data)
    dev_all, holdout_all, sample = source_experiment._sample_tables(enriched, enriched)
    previous_dev, previous_holdout, previous_identity = _load_previous_predictions(dev_all, holdout_all)
    sample_dev = previous_dev.sort_values("market_start_utc", kind="stable").reset_index(drop=True)
    sample_holdout = previous_holdout.sort_values("market_start_utc", kind="stable").reset_index(drop=True)
    if set(M0_FEATURES) & {"d_btc_minus_market", "logit_p_btc"}:
        raise AssertionError("M0 must not include the BTC model prediction or its disagreement feature")
    if set(M1_FEATURES) - (set(M0_FEATURES) | {"d_btc_minus_market"}):
        raise AssertionError("M1 contains an unregistered feature")
    if set(M0_FEATURES) != set(CONFIG["features"]["M0"]) or set(M1_FEATURES) != set(CONFIG["features"]["M1"]):
        raise AssertionError("Feature lists differ from the frozen configuration")
    current_identity = _source_identity(dev_all, holdout_all, previous_identity)
    identity = current_identity
    post_freeze_reporting_resume = False
    existing_manifest_path = OUTPUT_DIR / "pre_holdout_manifest.json"
    holdout_path = OUTPUT_DIR / "holdout_predictions.parquet"
    holdout_record_path = OUTPUT_DIR / "holdout_predictions.json"
    if existing_manifest_path.is_file():
        old_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        frozen_identity = old_manifest["identity"]
        if frozen_identity["identity_sha256"] != current_identity["identity_sha256"]:
            current_comparable = dict(current_identity)
            frozen_comparable = dict(frozen_identity)
            current_code = dict(current_comparable.pop("code_sha256"))
            frozen_code = dict(frozen_comparable.pop("code_sha256"))
            frozen_runner_hash = frozen_code.get("runner")
            current_comparable.pop("identity_sha256", None)
            frozen_comparable.pop("identity_sha256", None)
            current_code.pop("runner", None)
            frozen_code.pop("runner", None)
            holdout_record = json.loads(holdout_record_path.read_text(encoding="utf-8")) if holdout_record_path.is_file() else {}
            frozen_manifest_sha = _sha256(existing_manifest_path)
            existing_holdout_is_frozen = (
                holdout_path.is_file()
                and holdout_record.get("identity_sha256") == frozen_identity["identity_sha256"]
                and holdout_record.get("manifest_sha256") == frozen_manifest_sha
                and holdout_record.get("predictions_sha256") == _sha256(holdout_path)
            )
            if (current_comparable != frozen_comparable or current_code != frozen_code or not existing_holdout_is_frozen):
                raise RuntimeError("This experiment has a different frozen identity; use a new experiment ID")
            identity = frozen_identity
            post_freeze_reporting_resume = True
            _write_json(OUTPUT_DIR / "post_freeze_reporting_identity.json", {
                "frozen_identity_sha256": frozen_identity["identity_sha256"],
                "frozen_runner_sha256": frozen_runner_hash,
                "reporting_runner_sha256": current_identity["code_sha256"]["runner"],
                "reason": "Resumed reporting after an empty-only bootstrap resample produced NaN; frozen holdout predictions and model hashes were retained.",
            })
        else:
            identity = frozen_identity
        _write_json(OUTPUT_DIR / "source_identity.json", identity)
        final_result = OUTPUT_DIR / "results.json"
        if final_result.is_file():
            cached = json.loads(final_result.read_text(encoding="utf-8"))
            if cached.get("identity_sha256") == identity["identity_sha256"] and cached.get("holdout_scored_once") and REPORT_PATH.is_file():
                print("[interaction] completed matching run found; reusing saved holdout result", flush=True)
                return cached
    else:
        _write_json(OUTPUT_DIR / "source_identity.json", identity)
    folds = []
    for fold_config in CONFIG["folds"]:
        folds.append((fold_config, *_fold_slices(dev_all, fold_config)))
    _update_status(
        "Frozen source inputs and previous D predictions verified; the fixed 18-fit experiment is starting. No model refit or holdout scoring has occurred yet.",
        "Run the 16 sequential validation fits, select M0/M1 on the two development folds, then perform the two full-development fits and freeze the pre-holdout manifest.",
    )
    winners, dev_oof, configuration_results = _select_configurations(identity, dev_all)
    if len(dev_oof) != EXPECTED_DEV_OOF_ROWS or dev_oof.condition_id.duplicated().any():
        raise RuntimeError("The selected M0/M1 OOF predictions do not cover the same frozen development rows")
    dev_eval = sample_dev.merge(dev_oof[["market_start_utc", "condition_id", "p_M0", "p_M1"]],
                               on=["market_start_utc", "condition_id"], how="inner", validate="one_to_one")
    if len(dev_eval) != EXPECTED_DEV_OOF_ROWS:
        raise RuntimeError("Development comparison rows differ across raw market, previous D, M0, and M1")
    dev_oof = _attach_candidate_predictions(sample_dev, dev_oof)
    _write_parquet(OUTPUT_DIR / "development_oof_predictions.parquet", dev_oof)
    grid_definition = _interaction_grid_definition(dev_all)
    pre_manifest = _pre_holdout_manifest(identity, winners, dev_oof, grid_definition)
    existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8")) if existing_manifest_path.is_file() else None
    if existing_manifest is not None:
        if existing_manifest.get("identity", {}).get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError("The holdout manifest identity changed after it was frozen")
        for key in ("winner_configurations", "feature_lists", "selection_criterion", "economic_rules", "development_interaction_grid_definition"):
            if existing_manifest.get(key) != pre_manifest.get(key):
                raise RuntimeError(f"Frozen pre-holdout manifest field changed: {key}")
        pre_manifest = existing_manifest
    else:
        _write_json(existing_manifest_path, pre_manifest)
    manifest_sha = _sha256(existing_manifest_path)
    _write_json(OUTPUT_DIR / "run_manifest.json", {
        "status": "pre_holdout_manifest_frozen_before_final_fits", "identity_sha256": identity["identity_sha256"],
        "pre_holdout_manifest_sha256": manifest_sha, "frozen_at_utc": pre_manifest["frozen_at_utc"],
        "fit_count_expected": 18, "validation_fit_count": 16, "final_fit_count": 0,
    })
    _update_status(
        "All 16 validation fits are complete and M0/M1 settings are selected on development. The pre-holdout manifest is frozen before the two final development fits; holdout scoring has not occurred.",
        "Fit the two selected configurations on all qualifying development rows, save model/grid artifacts, then score the existing holdout once.",
    )

    final_boosters, final_fits, final_model_hashes = {}, {}, {}
    for name in ("M0", "M1"):
        final_boosters[name], final_fits[name] = _load_fit_or_train_final(identity, name, winners[name], dev_all)
        model_path, _, record_path = _checkpoint_paths(f"{name}__{winners[name]['config_id']}__final_development")
        final_model_hashes[name] = _sha256(model_path)
        final_fits[name]["model_path"] = str(model_path.relative_to(ROOT))
        final_fits[name]["model_sha256"] = final_model_hashes[name]
        final_fits[name]["fit_record_path"] = str(record_path.relative_to(ROOT))
        final_fits[name]["fit_record_sha256"] = _sha256(record_path)
    tree_diagnostics = _tree_path_diagnostics(final_boosters["M1"])
    grids = _interaction_grids(final_boosters["M1"], grid_definition)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = REPORT_DIR / "interaction_grids.csv"
    grids.to_csv(grid_path, index=False, float_format="%.12g")
    grid_sha256 = _sha256(grid_path)
    _write_json(OUTPUT_DIR / "final_development_models.json", {
        "identity_sha256": identity["identity_sha256"], "pre_holdout_manifest_sha256": manifest_sha,
        "fit_records": final_fits, "M1_tree_diagnostics": tree_diagnostics,
        "development_interaction_grid_path": str(grid_path.relative_to(ROOT)),
        "development_interaction_grid_sha256": grid_sha256,
    })
    _write_json(OUTPUT_DIR / "run_manifest.json", {
        "status": "final_development_models_frozen_before_holdout_scoring", "identity_sha256": identity["identity_sha256"],
        "pre_holdout_manifest_sha256": manifest_sha, "frozen_at_utc": pre_manifest["frozen_at_utc"],
        "final_model_hashes": final_model_hashes, "interaction_grid_sha256": grid_sha256,
        "fit_count_expected": 18, "validation_fit_count": 16, "final_fit_count": 2,
    })
    _update_status(
        "The pre-holdout manifest was frozen before final fitting. Both selected models are now fit on all qualifying development rows and their hashes and development-only interaction surfaces are saved; holdout scoring is next.",
        "Score the existing holdout once from the frozen M0/M1 fits, then write and review the report and result tables.",
    )

    holdout_record = {}
    if holdout_path.is_file():
        holdout_predictions = pd.read_parquet(holdout_path)
        if holdout_record_path.is_file():
            record = json.loads(holdout_record_path.read_text(encoding="utf-8"))
            holdout_record = record
            if record.get("identity_sha256") != identity["identity_sha256"] or record.get("manifest_sha256") != manifest_sha:
                raise RuntimeError("Saved holdout predictions do not match the frozen manifest")
            if record.get("final_model_sha256") != final_model_hashes:
                raise RuntimeError("Saved holdout predictions were generated by different final development models")
            if record.get("predictions_sha256") != _sha256(holdout_path):
                raise RuntimeError("Saved holdout prediction checkpoint is incomplete or changed")
        else:
            expected_identity = holdout_predictions.get("prediction_identity_sha256", pd.Series(dtype="string")).dropna().unique().tolist()
            expected_manifest = holdout_predictions.get("pre_holdout_manifest_sha256", pd.Series(dtype="string")).dropna().unique().tolist()
            expected_m0 = holdout_predictions.get("final_M0_sha256", pd.Series(dtype="string")).dropna().unique().tolist()
            expected_m1 = holdout_predictions.get("final_M1_sha256", pd.Series(dtype="string")).dropna().unique().tolist()
            expected_keys = sample_holdout[["market_start_utc", "condition_id"]].reset_index(drop=True)
            saved_keys = holdout_predictions[["market_start_utc", "condition_id"]].reset_index(drop=True)
            if (expected_identity != [identity["identity_sha256"]] or expected_manifest != [manifest_sha]
                    or expected_m0 != [final_model_hashes["M0"]] or expected_m1 != [final_model_hashes["M1"]]
                    or not saved_keys.equals(expected_keys)):
                raise RuntimeError("An incomplete holdout checkpoint exists; refusing to score the holdout again")
            _write_json(holdout_record_path, {
                "identity_sha256": identity["identity_sha256"], "manifest_sha256": manifest_sha,
                "final_model_sha256": final_model_hashes, "predictions_sha256": _sha256(holdout_path),
                "holdout_scored_once": True, "recovered_from_atomic_prediction_checkpoint": True,
                "rows": len(holdout_predictions),
            })
            holdout_record = json.loads(holdout_record_path.read_text(encoding="utf-8"))
    elif holdout_record_path.is_file():
        raise RuntimeError("The holdout score record exists without its prediction artifact; refusing to score again")
    else:
        print("[interaction] scoring the historical holdout once from frozen full-development fits", flush=True)
        holdout_predictions = sample_holdout.loc[:, [
            "market_start_utc", "condition_id", "official_outcome_up", "p_market", "p_D_conditional",
            "label_available_at_utc", "resolved_at_utc", "up_token_id", "down_token_id",
        ]].copy()
        for name in ("M0", "M1"):
            _, holdout_predictions[f"p_{name}"] = _predict_offset(
                final_boosters[name], sample_holdout, M0_FEATURES if name == "M0" else M1_FEATURES,
            )
        holdout_predictions["prediction_identity_sha256"] = identity["identity_sha256"]
        holdout_predictions["pre_holdout_manifest_sha256"] = manifest_sha
        holdout_predictions["final_M0_sha256"] = final_model_hashes["M0"]
        holdout_predictions["final_M1_sha256"] = final_model_hashes["M1"]
        if len(holdout_predictions) != EXPECTED_HOLDOUT_ROWS:
            raise RuntimeError("Holdout scoring did not preserve the frozen common quality sample")
        _write_parquet(holdout_path, holdout_predictions)
        holdout_record = {
            "identity_sha256": identity["identity_sha256"], "manifest_sha256": manifest_sha,
            "final_model_sha256": final_model_hashes,
            "predictions_sha256": _sha256(holdout_path), "holdout_scored_once": True,
            "scored_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "rows": len(holdout_predictions),
        }
        _write_json(holdout_record_path, holdout_record)
    if len(holdout_predictions) != EXPECTED_HOLDOUT_ROWS:
        raise RuntimeError("Saved holdout rows changed")

    quality_metrics = {}
    for name, frame in (("development_cv_selected", dev_oof),
                        ("dev_2026q2_b", dev_oof.loc[dev_oof.market_start_utc.lt(pd.Timestamp("2026-06-01T00:00:00Z"))]),
                        ("dev_2026q2_c", dev_oof.loc[dev_oof.market_start_utc.ge(pd.Timestamp("2026-06-01T00:00:00Z"))]),
                        ("holdout_descriptive", holdout_predictions)):
        metrics, paired = _quality_metrics(frame, _metric_columns(frame))
        quality_metrics[name] = {"metrics": metrics, "paired_differences": paired}

    dev_predictions_for_economics = dev_oof.copy()
    dev_economics = _economic_evaluation(
        dev_predictions_for_economics, t59, t60, DEV_CUTOFF, "development_cv_selected",
    )
    holdout_for_economics = sample_holdout.merge(
        holdout_predictions[["market_start_utc", "condition_id", "p_M0", "p_M1"]],
        on=["market_start_utc", "condition_id"], how="inner", validate="one_to_one",
    )
    holdout_economics = _economic_evaluation(holdout_for_economics, t59, t60, HOLDOUT_CUTOFF, "holdout_descriptive")
    dev_portfolio = _portfolio_control(dev_predictions_for_economics, t59, t60, DEV_CUTOFF, "development_cv_selected")
    holdout_portfolio = _portfolio_control(holdout_for_economics, t59, t60, HOLDOUT_CUTOFF, "holdout_descriptive")
    peak = source_experiment._peak_working_set_bytes()
    if peak is None:
        raise RuntimeError("Could not measure the peak process working set")
    results = {
        "status": "completed", "experiment_id": CONFIG["experiment_id"],
        "identity_sha256": identity["identity_sha256"], "pre_holdout_manifest_sha256": manifest_sha,
        "pre_holdout_manifest": pre_manifest,
        "holdout_scored_once": True, "holdout_prediction_sha256": _sha256(holdout_path),
        "holdout_scored_at_utc": holdout_record.get("scored_at_utc"),
        "post_freeze_reporting_resume": post_freeze_reporting_resume,
        "runtime_scope": ("reporting_resume_reusing_frozen_fit_and_holdout_artifacts" if post_freeze_reporting_resume else "full_experiment_invocation"),
        "reporting_runner_sha256": _sha256(Path(__file__)),
        "sample": {"development_rows": len(dev_oof), "development_fit_rows": len(dev_all), "holdout_rows": len(holdout_predictions),
                   "folds": sample["periods"], "feature_coverage": feature_coverage,
                   "prediction_source_identity": prediction_identity, "snapshot_record": book_record,
                   "previous_D_identity": previous_identity},
        "winner_configurations": {name: winners[name] for name in ("M0", "M1")},
        "configuration_results": configuration_results,
        "final_development_fits": final_fits,
        "final_model_sha256": final_model_hashes,
        "interaction_grid_sha256": grid_sha256,
        "quality_metrics": quality_metrics,
        "economics_fixed5": {"development_cv_selected": dev_economics, "holdout_descriptive": holdout_economics},
        "economics_free_cash_5pct": {"development_cv_selected": dev_portfolio, "holdout_descriptive": holdout_portfolio},
        "tree_diagnostics": tree_diagnostics,
        "fit_count": 18, "runtime_seconds": time.perf_counter() - started,
        "peak_working_set_bytes": peak,
        "runtime_environment": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                                "numpy": np.__version__, "pandas": pd.__version__, "num_threads_per_fit": 4,
                                "concurrent_fits": 1},
    }
    results["conclusions"] = _conclusions(results)
    _write_grid_and_report(results, grids, grid_definition)
    _write_json(OUTPUT_DIR / "run_manifest.json", {
        "status": "completed", "identity_sha256": identity["identity_sha256"],
        "pre_holdout_manifest_sha256": manifest_sha, "holdout_prediction_sha256": _sha256(holdout_path),
        "final_model_hashes": final_model_hashes, "interaction_grid_sha256": grid_sha256,
        "holdout_scored_once": True, "fit_count": 18, "validation_fit_count": 16, "final_fit_count": 2,
        "runtime_scope": results["runtime_scope"], "reporting_runner_sha256": results["reporting_runner_sha256"],
        "runtime_seconds": results["runtime_seconds"], "peak_working_set_bytes": peak,
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
    })
    _update_status(
        "Completed: 18 sequential LightGBM fits, development selection, pre-holdout freeze, one descriptive holdout scoring pass, paired quality/economic results, interaction diagnostics, and report. Holdout was not used for any selection.",
        "No model/policy changes are planned from this exposed holdout; see the experiment report for the recorded result and reproduction command.",
    )
    print(f"[interaction] completed in {results['runtime_seconds']:.1f}s; report={REPORT_PATH}", flush=True)
    return results


if __name__ == "__main__":
    main()
