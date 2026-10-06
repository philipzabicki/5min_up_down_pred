"""Run a bounded, resumable BTC pre-open candidate study on the frozen v1 features."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from features.btc_preopen_contract import (
    DECISION_COL,
    TARGET_AVAILABLE_COL,
    TARGET_COL,
)


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/nightly_v1/06dc69ccb35a63b3"
REPORT_DIR = ROOT / "reports/btc_preopen"
STUDY_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/candidate_study_20261005"
FEATURE_DATASET = RUN_DIR / "stages/final_feature_dataset_1d21891c3f3f/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_preopen_v1.parquet"
BUNDLE_PATH = RUN_DIR / "stages/final_model_51d3adab0c53/model_bundle.json"
OFFICIAL_PATH = ROOT / "data/analysis/polymarket/BTC/new_model_comparison/runs/d794ac2dea5a25a2/shared_market_evaluation.parquet"
OUTPUT_DIR = ROOT / "data/models/BTC/btc_preopen_candidate_20261005"
FIT_END = pd.Timestamp("2026-01-01T00:00:00Z")
SEARCH_TRIALS = 16
SEED = 20261004
THREADS = 4
MAX_ROUNDS = 800
EARLY_STOPPING_ROUNDS = 50
CV_STD_PENALTY = 0.5
SEARCH_FOLDS = (
    (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2025-04-01T00:00:00Z")),
    (pd.Timestamp("2025-04-01T00:00:00Z"), pd.Timestamp("2025-07-01T00:00:00Z")),
)
DEVELOPMENT_SELECTION_FOLD = (
    pd.Timestamp("2025-07-01T00:00:00Z"),
    pd.Timestamp("2025-10-01T00:00:00Z"),
)
HISTORY_VARIANTS = ("all_history", "last_3y", "last_1y", "half_life_365d")
DECISION_WEIGHT_VARIANTS = (0.23, 0.5, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _new_parameters(trial=None):
    if trial is None:
        return {
            "learning_rate": 0.04,
            "num_leaves": 63,
            "max_depth": -1,
            "min_data_in_leaf": 1024,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 1.0,
            "lambda_l2": 10.0,
            "min_gain_to_split": 0.05,
            "extra_trees": False,
        }
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 127),
        "max_depth": -1,
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 128, 16_384, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.55, 0.95),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.65, 0.95),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 100.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
    }


def _weights(decision_mask, decision_weight):
    return np.where(
        decision_mask,
        float(decision_weight),
        (1.0 - float(decision_weight)) / 4.0,
    ).astype(np.float32)


def _history_train_indices(opened, train_indices, cutoff, history_mode):
    if history_mode == "all_history" or history_mode == "half_life_365d":
        return train_indices
    years = 3 if history_mode == "last_3y" else 1
    start = cutoff - pd.DateOffset(years=years)
    return train_indices[opened[train_indices] >= start]


def _train_weights(opened, decision, indices, cutoff, decision_weight, history_mode):
    weights = _weights(decision[indices], decision_weight)
    if history_mode == "half_life_365d":
        age_days = np.maximum(
            np.asarray((cutoff - opened[indices]).total_seconds(), dtype=np.float64) / 86_400.0,
            0.0,
        )
        base_total = float(weights.sum())
        weights *= np.exp(-math.log(2.0) * age_days / 365.0).astype(np.float32)
        if weights.sum() > 0.0:
            weights *= base_total / float(weights.sum())
    keep = weights > 0.0
    return indices[keep], weights[keep]


def _base_params():
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "device_type": "gpu",
        "num_threads": THREADS,
        "max_bin": 63,
        "feature_pre_filter": False,
        "deterministic": True,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
    }


def _fit_score(x, y, weight, train_indices, valid_indices, params, rounds=MAX_ROUNDS, early_stop=True):
    training = lgb.Dataset(
        x[train_indices], label=y[train_indices], weight=weight,
        free_raw_data=True,
    )
    validation = lgb.Dataset(
        x[valid_indices], label=y[valid_indices], reference=training,
        free_raw_data=True,
    )
    callbacks = [lgb.log_evaluation(0)]
    if early_stop:
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False))
    model = lgb.train(
        params,
        training,
        num_boost_round=int(rounds),
        valid_sets=[validation],
        valid_names=["decision_validation"],
        callbacks=callbacks,
    )
    iteration = int(model.best_iteration or rounds)
    probability = np.clip(model.predict(x[valid_indices], num_iteration=iteration), 1e-6, 1.0 - 1e-6)
    score = float(log_loss(y[valid_indices], probability, labels=[0, 1]))
    del validation, training
    return model, probability, score, iteration


def _indices_for_fold(opened, decision_at, labels_valid, valid_start, valid_end, label_available, y):
    valid = (
        labels_valid
        & (decision_at >= valid_start)
        & (decision_at < valid_end)
    )
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        raise RuntimeError(f"No decision labels in validation fold {valid_start}..{valid_end}")
    first_decision = decision_at[valid_indices[0]]
    train = labels_valid & (opened < valid_start) & (label_available <= first_decision)
    train_indices = np.flatnonzero(train)
    if train_indices.size == 0 or np.unique(y[train_indices]).size < 2:
        raise RuntimeError(f"Training labels before {first_decision} do not contain both classes")
    return train_indices, valid_indices


def _load_data():
    run = json.loads((RUN_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    feature_order = list(bundle["feature_order"])
    columns = ["Opened", "nominal_decision_at", DECISION_COL, TARGET_AVAILABLE_COL, TARGET_COL, *feature_order]
    frame = pd.read_parquet(FEATURE_DATASET, columns=columns)
    opened = pd.DatetimeIndex(pd.to_datetime(frame["Opened"], utc=True, errors="raise"))
    decision_at = pd.DatetimeIndex(pd.to_datetime(frame["nominal_decision_at"], utc=True, errors="raise"))
    label_available = pd.DatetimeIndex(pd.to_datetime(frame[TARGET_AVAILABLE_COL], utc=True, errors="coerce"))
    decision = frame[DECISION_COL].to_numpy(dtype=bool, copy=False)
    y_float = pd.to_numeric(frame[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    labels_valid = np.isfinite(y_float) & label_available.notna()
    x = frame.loc[:, feature_order].to_numpy(dtype=np.float32, copy=True)
    x[~np.isfinite(x)] = np.nan
    y = np.where(np.isfinite(y_float), y_float, 0).astype(np.int8)
    del frame
    return {
        "run": run,
        "bundle": bundle,
        "feature_order": feature_order,
        "opened": opened,
        "decision_at": decision_at,
        "label_available": label_available,
        "decision": decision,
        "labels_valid": labels_valid,
        "x": x,
        "y": y,
    }


def _candidate_manifest(data):
    return {
        "protocol": "binary logloss on unweighted actual-decision rows; two search quarters and one frozen development-selection quarter",
        "search_trials": SEARCH_TRIALS,
        "search_folds": [[a.isoformat(), b.isoformat()] for a, b in SEARCH_FOLDS],
        "development_selection_fold": [x.isoformat() for x in DEVELOPMENT_SELECTION_FOLD],
        "history_variants": list(HISTORY_VARIANTS),
        "decision_weight_variants": list(DECISION_WEIGHT_VARIANTS),
        "search_space": "depth fixed to -1; leaves 16-127; bagging fraction 0.65-0.95 with frequency 1; one feature sampling fraction; regularization and learning rate ranges in runner",
        "feature_order_sha256": hashlib.sha256("\n".join(data["feature_order"]).encode()).hexdigest(),
        "feature_dataset_sha256": sha256(FEATURE_DATASET),
        "original_model_sha256": data["bundle"]["model_sha256"],
        "fit_cutoff_exclusive": FIT_END.isoformat(),
        "backend": "LightGBM GPU, RTX 4060 Laptop, 4 LightGBM threads, sequential trials",
        "seed": SEED,
    }


def run_study():
    started = time.perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_data()
    identity = _candidate_manifest(data)
    identity_path = STUDY_DIR / "study_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise RuntimeError("Candidate study checkpoint identity differs from its input or protocol")
    write_json(identity_path, identity)
    opened = data["opened"]
    decision_at = data["decision_at"]
    label_available = data["label_available"]
    decision = data["decision"]
    labels_valid = data["labels_valid"]
    x = data["x"]
    y = data["y"]
    folds = [
        _indices_for_fold(opened, decision_at, labels_valid, a, b, label_available, y)
        for a, b in SEARCH_FOLDS
    ]

    def objective(trial):
        history_mode = trial.suggest_categorical("history_mode", list(HISTORY_VARIANTS))
        decision_weight = trial.suggest_categorical("decision_weight", list(DECISION_WEIGHT_VARIANTS))
        params = {**_base_params(), **_new_parameters(trial)}
        fold_scores, fold_iterations, fold_rows = [], [], []
        for train_indices, valid_indices in folds:
            valid_start = decision_at[valid_indices[0]]
            candidate_train = _history_train_indices(opened, train_indices, valid_start, history_mode)
            candidate_train, weights = _train_weights(
                opened, decision, candidate_train, valid_start, decision_weight, history_mode
            )
            model, _, score, iteration = _fit_score(
                x, y, weights, candidate_train, valid_indices, params,
            )
            fold_scores.append(score)
            fold_iterations.append(iteration)
            fold_rows.append({"logloss": score, "best_iteration": iteration, "train_rows": int(len(candidate_train)), "validation_rows": int(len(valid_indices))})
            del model
        objective_value = float(np.mean(fold_scores) + CV_STD_PENALTY * np.std(fold_scores))
        trial.set_user_attr("fold_results", fold_rows)
        trial.set_user_attr("mean_logloss", float(np.mean(fold_scores)))
        trial.set_user_attr("std_logloss", float(np.std(fold_scores)))
        trial.set_user_attr("median_best_iteration", int(np.median(fold_iterations)))
        return objective_value

    database = STUDY_DIR / "study.sqlite3"
    study = optuna.create_study(
        study_name="btc_preopen_candidates_20261005",
        storage="sqlite:///" + database.resolve().as_posix(),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6),
        load_if_exists=True,
    )
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if completed < SEARCH_TRIALS:
        study.optimize(objective, n_trials=SEARCH_TRIALS - completed, n_jobs=1, gc_after_trial=True)

    best = study.best_trial
    best_params = {**_base_params(), **{k: best.params[k] for k in _new_parameters_keys()}}
    best_history = str(best.params["history_mode"])
    best_decision_weight = float(best.params["decision_weight"])
    best_iteration = int(best.user_attrs["median_best_iteration"])
    pd.DataFrame(
        [
            {
                "trial": trial.number,
                "objective_mean_plus_0_5_std": trial.value,
                "mean_logloss": trial.user_attrs.get("mean_logloss"),
                "std_logloss": trial.user_attrs.get("std_logloss"),
                "history_mode": trial.params.get("history_mode"),
                "decision_weight": trial.params.get("decision_weight"),
                "median_best_iteration": trial.user_attrs.get("median_best_iteration"),
                **{f"param_{k}": trial.params.get(k) for k in _new_parameters_keys()},
            }
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
    ).to_csv(REPORT_DIR / "candidate_search_trials.csv", index=False)

    # Candidate list is frozen here and ranked only on the later 2025 development quarter.
    select_start, select_end = DEVELOPMENT_SELECTION_FOLD
    selection_train, selection_valid = _indices_for_fold(
        opened, decision_at, labels_valid, select_start, select_end, label_available, y
    )
    original_training = json.loads(
        (RUN_DIR / "stages/final_model_51d3adab0c53/training_manifest.json").read_text(encoding="utf-8")
    )
    original_params = dict(original_training["params"])
    original_params.update(_base_params())
    original_params["num_threads"] = THREADS
    original_weight = float(original_training["decision_weight"])
    selector_params = {
        "objective": "binary", "metric": "binary_logloss", "verbosity": -1,
        "device_type": "gpu", "num_threads": THREADS, "max_bin": 63,
        "learning_rate": 0.025, "num_leaves": 63, "max_depth": 6,
        "min_data_in_leaf": 512, "feature_fraction": 0.65,
        "bagging_fraction": 0.85, "bagging_freq": 1,
        "lambda_l2": 15.0, "lambda_l1": 2.0,
        "min_sum_hessian_in_leaf": 0.05, "min_gain_to_split": 0.03,
        "feature_fraction_bynode": 0.75, "path_smooth": 3.0,
        "extra_trees": False, "seed": 20261004,
    }
    candidate_specs = [
        ("original_v1_recipe", "all_history", original_weight, original_params, int(original_training["best_iteration"]), "original fitted params and 66-round schedule"),
        ("selector_baseline", "all_history", original_weight, selector_params, 81, "fixed selector params; 112 selected features"),
        ("tuned_search_winner", best_history, best_decision_weight, best_params, best_iteration, "history and decision weight selected by two earlier chronological logloss folds"),
        ("tuned_all_legacy_weight", "all_history", 0.23, best_params, best_iteration, "new search params; legacy 0.23 decision weight"),
        ("tuned_all_logloss_weight", "all_history", best_decision_weight, best_params, best_iteration, "new search params and logloss-selected decision weight"),
        ("tuned_last_3y", "last_3y", best_decision_weight, best_params, best_iteration, "new search params; three-year history"),
        ("tuned_last_1y", "last_1y", best_decision_weight, best_params, best_iteration, "new search params; one-year history"),
        ("tuned_half_life_365d", "half_life_365d", best_decision_weight, best_params, best_iteration, "new search params; 365-day exponential half-life"),
    ]
    write_json(STUDY_DIR / "frozen_candidate_list.json", {
        "selection_metric": "unweighted Binance proxy binary logloss on Q3 2025 actual decision rows",
        "selection_fold": [select_start.isoformat(), select_end.isoformat()],
        "candidates": [
            {"name": n, "history_mode": h, "decision_weight": w, "iterations": i, "description": d}
            for n, h, w, _, i, d in candidate_specs
        ],
        "created_before_external_test_scoring": True,
    })
    selection_path = STUDY_DIR / "selection_results.json"
    cached = {}
    if selection_path.exists():
        prior = json.loads(selection_path.read_text(encoding="utf-8"))
        if prior.get("candidate_list_sha256") != sha256(STUDY_DIR / "frozen_candidate_list.json"):
            raise RuntimeError("Frozen development candidate list changed after scoring began")
        cached = prior.get("results", {})
    selection_results = dict(cached)
    for name, history_mode, decision_weight, params, iterations, description in candidate_specs:
        if name in selection_results:
            continue
        valid_start = decision_at[selection_valid[0]]
        train = _history_train_indices(opened, selection_train, valid_start, history_mode)
        train, weights = _train_weights(opened, decision, train, valid_start, decision_weight, history_mode)
        model, probability, score, _ = _fit_score(
            x, y, weights, train, selection_valid, params,
            rounds=iterations, early_stop=False,
        )
        selection_results[name] = {
            "logloss": score,
            "rows": int(len(selection_valid)),
            "train_rows": int(len(train)),
            "description": description,
            "probability_sha256": hashlib.sha256(probability.astype("<f4").tobytes()).hexdigest(),
        }
        del model, probability
        write_json(selection_path, {
            "candidate_list_sha256": sha256(STUDY_DIR / "frozen_candidate_list.json"),
            "results": selection_results,
        })
    best_name = min(selection_results, key=lambda name: selection_results[name]["logloss"])
    selected_spec = next(spec for spec in candidate_specs if spec[0] == best_name)
    name, history_mode, decision_weight, params, iterations, description = selected_spec

    development_probabilities = {}
    valid_start = decision_at[selection_valid[0]]
    for needed in dict.fromkeys((best_name, "original_v1_recipe")):
        _, mode, weight, candidate_params, candidate_iterations, _ = next(
            spec for spec in candidate_specs if spec[0] == needed
        )
        train = _history_train_indices(opened, selection_train, valid_start, mode)
        train, weights = _train_weights(opened, decision, train, valid_start, weight, mode)
        model, probability, _, _ = _fit_score(
            x, y, weights, train, selection_valid, candidate_params,
            rounds=candidate_iterations, early_stop=False,
        )
        development_probabilities[needed] = probability.copy()
        del model, probability
    import run_btc_preopen_experiment as original_runner
    development_paired_ci = original_runner._paired_block_interval(
        y[selection_valid],
        development_probabilities[best_name],
        development_probabilities["original_v1_recipe"],
    )
    stable_development_improvement = bool(
        selection_results[best_name]["logloss"] < selection_results["original_v1_recipe"]["logloss"]
        and development_paired_ci["delta_first_minus_second_log_loss_ci95"][1] < 0.0
    )

    # Fit the development-selected candidate using only labels available before 2026-01-01.
    fit_indices = np.flatnonzero(labels_valid & (label_available < FIT_END))
    fit_indices = _history_train_indices(opened, fit_indices, FIT_END, history_mode)
    fit_indices, fit_weights = _train_weights(opened, decision, fit_indices, FIT_END, decision_weight, history_mode)
    final_train = lgb.Dataset(x[fit_indices], label=y[fit_indices], weight=fit_weights, free_raw_data=True)
    final_model = lgb.train(params, final_train, num_boost_round=iterations, callbacks=[lgb.log_evaluation(0)])
    model_path = OUTPUT_DIR / "candidate_model.txt"
    final_model.save_model(str(model_path), num_iteration=iterations)
    model_sha = sha256(model_path)
    del final_train

    calibration_mask = (
        labels_valid & decision
        & (decision_at >= pd.Timestamp("2026-01-01T00:00:00Z"))
        & (decision_at < pd.Timestamp("2026-04-15T17:04:00Z"))
        & (label_available <= pd.Timestamp("2026-04-15T17:04:00Z"))
    )
    calibration_indices = np.flatnonzero(calibration_mask)
    calibration_raw = np.clip(final_model.predict(x[calibration_indices], num_iteration=iterations), 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=SEED)
    calibrator.fit(np.log(calibration_raw / (1 - calibration_raw)).reshape(-1, 1), y[calibration_indices])
    cal_payload = {
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
        "rows": int(len(calibration_indices)),
        "latest_target_available_at": label_available[calibration_indices].max().isoformat(),
        "label_source": "Binance COIN-M BTCUSD index proxy",
    }
    write_json(OUTPUT_DIR / "candidate_calibrator.json", cal_payload)

    test_mask = (
        labels_valid & decision
        & (decision_at >= pd.Timestamp("2026-04-15T17:04:00Z"))
        & (decision_at <= pd.Timestamp("2026-05-18T10:29:00Z"))
    )
    test_indices = np.flatnonzero(test_mask)
    raw = np.clip(final_model.predict(x[test_indices], num_iteration=iterations), 1e-6, 1 - 1e-6)
    platt = 1.0 / (1.0 + np.exp(-(
        cal_payload["intercept"] + cal_payload["coefficient"] * np.log(raw / (1 - raw))
    )))
    test = pd.DataFrame({
        "Opened": opened[test_indices],
        "market_start_utc": opened[test_indices] + pd.Timedelta(minutes=2),
        "target_binance_proxy_up": y[test_indices],
        "p_candidate_raw": raw,
        "p_candidate_platt": platt,
    })
    official = pd.read_parquet(OFFICIAL_PATH, columns=["condition_id", "market_slug", "market_start_utc", "target_polymarket_up"])
    official["market_start_utc"] = pd.to_datetime(official["market_start_utc"], utc=True)
    test = test.merge(official, on="market_start_utc", how="left", validate="one_to_one")
    test.to_parquet(REPORT_DIR / "candidate_external_predictions.parquet", index=False)

    def metrics(target, probabilities):
        return {
            "n": int(len(target)),
            "positive_rate": float(np.mean(target)),
            "logloss": float(log_loss(target, probabilities, labels=[0, 1])),
            "brier": float(brier_score_loss(target, probabilities)),
            "auc": float(roc_auc_score(target, probabilities)),
        }

    external_metrics = {}
    for label_name, target_col in (("binance_proxy", "target_binance_proxy_up"), ("official_polymarket", "target_polymarket_up")):
        rows = test.loc[test[target_col].notna()]
        target = rows[target_col].to_numpy(dtype=np.int8)
        external_metrics[label_name] = {
            "raw": metrics(target, rows["p_candidate_raw"].to_numpy(dtype=np.float64)),
            "platt": metrics(target, rows["p_candidate_platt"].to_numpy(dtype=np.float64)),
        }

    original_predictions = pd.read_parquet(
        RUN_DIR / "stages/external_evaluation_7e5807168d4c/external_test_predictions.parquet",
        columns=["Opened", "p_model_raw", "p_model_platt"],
    )
    original_predictions["Opened"] = pd.to_datetime(original_predictions["Opened"], utc=True)
    aligned = test.merge(original_predictions, on="Opened", how="inner", validate="one_to_one")
    paired = {}
    for label_name, target_col in (("binance_proxy", "target_binance_proxy_up"), ("official_polymarket", "target_polymarket_up")):
        rows = aligned.loc[aligned[target_col].notna()]
        target = rows[target_col].to_numpy(dtype=np.int8)
        if len(target):
            paired[label_name] = {
                "candidate_raw_minus_original_raw": original_runner._paired_block_interval(
                    target, rows["p_candidate_raw"], rows["p_model_raw"]
                ),
                "candidate_platt_minus_original_platt": original_runner._paired_block_interval(
                    target, rows["p_candidate_platt"], rows["p_model_platt"]
                ),
            }

    comparison_rows = []
    for candidate_name, result in selection_results.items():
        comparison_rows.append({"candidate": candidate_name, "scope": "2025Q3 development selection", "logloss": result["logloss"], "rows": result["rows"], "train_rows": result["train_rows"], "decision_weight": next(s[2] for s in candidate_specs if s[0] == candidate_name), "history": next(s[1] for s in candidate_specs if s[0] == candidate_name)})
    for target_name, target_col in (("binance_proxy", "target_binance_proxy_up"), ("official_polymarket", "target_polymarket_up")):
        rows = aligned.loc[aligned[target_col].notna()]
        target = rows[target_col].to_numpy(dtype=np.int8)
        for model_name, probability_column in (("raw", "p_model_raw"), ("platt", "p_model_platt")):
            score = metrics(target, rows[probability_column].to_numpy(dtype=np.float64))
            comparison_rows.append({"candidate": f"original_v1_{model_name}", "scope": f"external {target_name}", **score, "rows": score["n"]})
        for model_name in ("raw", "platt"):
            score = external_metrics[target_name][model_name]
            comparison_rows.append({"candidate": f"selected_{best_name}_{model_name}", "scope": f"external {target_name}", **score, "rows": score["n"]})
    pd.DataFrame(comparison_rows).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    result = {
        "selected_candidate_by_development_logloss": best_name,
        "selected_candidate_description": description,
        "history_mode": history_mode,
        "decision_weight": decision_weight,
        "parameters": {k: v for k, v in params.items() if k not in {"seed", "feature_fraction_seed", "bagging_seed", "data_random_seed"}},
        "iterations": iterations,
        "q3_2025_development_selection": selection_results,
        "development_paired_bootstrap_vs_original": development_paired_ci,
        "stable_development_logloss_improvement": stable_development_improvement,
        "candidate_model_sha256": model_sha,
        "candidate_model_path": model_path.relative_to(ROOT).as_posix(),
        "calibration": cal_payload,
        "external_metrics": external_metrics,
        "paired_3day_bootstrap_vs_original": paired,
        "external_rows": int(len(test)),
        "external_official_rows": int(test["target_polymarket_up"].notna().sum()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(REPORT_DIR / "candidate_metrics.json", result)
    write_json(STUDY_DIR / "completed.json", {"selected_candidate": best_name, "model_sha256": model_sha, "elapsed_seconds": result["elapsed_seconds"]})
    print(json.dumps({"selected": best_name, "external_metrics": external_metrics, "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)


def _new_parameters_keys():
    return (
        "learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction",
        "bagging_fraction", "lambda_l1", "lambda_l2", "min_gain_to_split", "extra_trees",
    )


if __name__ == "__main__":
    run_study()
