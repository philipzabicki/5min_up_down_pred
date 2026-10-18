"""Compare the frozen BASE70 taker-flow control with its 64-feature BASE64 control."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

import run_btc_preopen_experiment as experiment
import run_btc_preopen_model_update as model_update
import run_btc_preopen_research as research
import run_btc_preopen_taker_flow_20261018 as taker
from utils.cv_objective import ObjectiveAlignedCVCallback, cv_objective_series
from utils.metrics import (
    make_lightgbm_binary_balanced_accuracy_eval,
    make_lightgbm_binary_brier_eval,
    make_lightgbm_binary_logloss_eval,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_preopen_zv_ablation_20261018.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_directory"]
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
REPORT_PATH = ROOT / CONFIG["report_path"]
VERSIONED_MANIFEST_PATH = ROOT / CONFIG["versioned_manifest_path"]
STATUS_PATH = ROOT / CONFIG["status_path"]
STATUS_BEGIN = "<!-- BTC_ZV_ABLATION_20261018_BEGIN -->"
STATUS_END = "<!-- BTC_ZV_ABLATION_20261018_END -->"

COMMON_DIR = taker.OUTPUT_DIR
COMMON_PATH = COMMON_DIR / "common_complete_training_rows.parquet"
FOLDS_PATH = COMMON_DIR / "common_folds.npz"
COMMON_SUMMARY_PATH = COMMON_DIR / "common_rows_summary.json"
BASE70_OOF_PATH = COMMON_DIR / "B0_decision_oof.parquet"
BASE70_CV_SUMMARY_PATH = COMMON_DIR / "B0_cv_summary.json"
BASE70_OFFICIAL_PATH = COMMON_DIR / "B0_B1_official_oof.parquet"
BASE70_QUALITY_PATH = COMMON_DIR / "quality_and_economics.json"
BASE70_ECON_DIR = COMMON_DIR / "economics/B0"
BASE64_MODEL_DIR = OUTPUT_DIR / "models/BASE64"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _record(path: Path) -> dict:
    return taker._record(path)


def _verify_records(records: list[dict]) -> bool:
    return taker._verify_records(records)


def _write_manifest(manifest: dict) -> None:
    _write_json(MANIFEST_PATH, manifest)
    _status_update(manifest)


def _status_update(manifest: dict) -> None:
    stages = manifest.get("stages", {})
    active = manifest.get("active_stage") or "none"
    completed = sorted(
        name for name, stage in stages.items() if stage.get("status") == "completed"
    )
    lines = [
        STATUS_BEGIN,
        "### BTC pre-open Z/V ablation - 2026-10-18",
        f"- Updated UTC: {pd.Timestamp.now(tz='UTC').isoformat()}.",
        f"- Status: {manifest.get('status')}; active stage: {active}.",
        f"- Completed stages: {', '.join(completed) or 'none'}.",
        f"- Next step: {manifest.get('next_step', 'Resume the no-argument runner.')}.",
        f"- Run manifest: {_relative(MANIFEST_PATH)}.",
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
        anchor = "<!-- BTC_RETURN_REGRESSION_STATUS_BEGIN -->"
        text = text.replace(anchor, block + anchor, 1) if anchor in text else block + text
    temporary = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(STATUS_PATH)


def _begin_stage(manifest: dict, name: str, next_step: str) -> None:
    manifest["status"] = "running"
    manifest["active_stage"] = name
    manifest["next_step"] = next_step
    manifest.setdefault("stages", {})[name] = {
        "status": "running",
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write_manifest(manifest)


def _finish_stage(manifest: dict, name: str, **payload) -> None:
    stage = manifest.setdefault("stages", {}).setdefault(name, {})
    stage.update(payload)
    stage["status"] = "completed"
    stage["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    manifest["active_stage"] = None
    _write_manifest(manifest)


def _load_common_metadata():
    base64_features = list(taker.SOURCE_FEATURES)
    base70_features = base64_features + list(taker.CONTROL_FLOW_FEATURES)
    if len(base64_features) != 64 or len(base70_features) != 70:
        raise RuntimeError("The frozen source or BASE70 feature count changed")
    if base70_features != taker._read_json(BASE70_CV_SUMMARY_PATH)["feature_order"]:
        raise RuntimeError("The existing B0 feature order does not match BASE70")
    if any(name.startswith("taker_") for name in base64_features):
        raise RuntimeError("BASE64 contains taker-flow features")
    if any(name.startswith(("taker_I_", "taker_U_", "taker_beta_", "taker_A_")) for name in base70_features):
        raise RuntimeError("BASE70 includes B1 taker-flow features")

    columns = [
        "_source_row_index", "Opened", "nominal_decision_at",
        "target_label_available_at", "target_preopen_5m_up_binance_index_proxy",
        "is_polymarket_preopen_decision",
    ]
    frame = pd.read_parquet(COMMON_PATH, columns=columns)
    if frame["_source_row_index"].duplicated().any() or not frame["Opened"].is_monotonic_increasing:
        raise RuntimeError("The frozen common sample is not unique and chronological")
    if not frame["target_preopen_5m_up_binance_index_proxy"].isin((0, 1)).all():
        raise RuntimeError("The frozen target contains non-binary labels")
    with np.load(FOLDS_PATH) as saved:
        folds = [
            (saved[f"train_{i}"].astype(np.int32, copy=True),
             saved[f"valid_{i}"].astype(np.int32, copy=True))
            for i in range(taker.CV_FOLD_COUNT)
        ]
    labels_available = pd.to_datetime(frame["target_label_available_at"], utc=True).array.asi8
    decisions_at = pd.to_datetime(frame["nominal_decision_at"], utc=True).array.asi8
    decision = frame["is_polymarket_preopen_decision"].to_numpy(dtype=bool)
    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        if not len(train_idx) or not len(valid_idx) or np.intersect1d(train_idx, valid_idx).size:
            raise RuntimeError(f"Frozen fold {fold_id} has invalid train/validation indices")
        if not decision[valid_idx].all():
            raise RuntimeError(f"Frozen fold {fold_id} validates non-decision rows")
        if labels_available[train_idx].max() > decisions_at[valid_idx].min():
            raise RuntimeError(f"Frozen fold {fold_id} violates label-availability purge")
    return frame, base64_features, base70_features, folds


def _freeze_comparison(source: dict):
    flow_manifest = taker._read_json(taker.MANIFEST_PATH)
    if flow_manifest.get("status") != "completed":
        raise RuntimeError("The source taker-flow experiment is not complete")
    for stage_name in ("flow_features", "common_rows", "cv_B0", "quality_and_economics"):
        if flow_manifest.get("stages", {}).get(stage_name, {}).get("status") != "completed":
            raise RuntimeError(f"Required BASE70 source stage is incomplete: {stage_name}")
    _, _, _, _, _, correction_manifest, _, correction_identity = model_update._source_inputs()
    for stage_name in ("flow_features", "common_rows", "cv_B0"):
        if not _verify_records(flow_manifest["stages"][stage_name]["outputs"]):
            raise RuntimeError(f"Saved BASE70 inputs failed verification: {stage_name}")

    frame, base64_features, base70_features, folds = _load_common_metadata()
    weights = experiment._preopen_sample_weights(
        frame["is_polymarket_preopen_decision"].to_numpy(dtype=bool),
        taker.DECISION_WEIGHT,
    )
    weight_hash = taker._array_sha256(weights, dtype=np.float32)
    if weight_hash != taker._read_json(BASE70_CV_SUMMARY_PATH)["training_weight_sha256"]:
        raise RuntimeError("Reconstructed common-row weights differ from BASE70")

    mask_path = OUTPUT_DIR / "shared_source_row_indices.npz"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    temp_mask = mask_path.with_suffix(".npz.tmp")
    with temp_mask.open("wb") as handle:
        np.savez_compressed(
            handle, source_row_index=frame["_source_row_index"].to_numpy(dtype=np.int32)
        )
    temp_mask.replace(mask_path)

    common_summary = taker._read_json(COMMON_SUMMARY_PATH)
    fold_records = []
    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        fold_records.append({
            "fold_id": fold_id,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(valid_idx)),
            "train_indices_sha256": taker._array_sha256(train_idx, dtype=np.int32),
            "validation_indices_sha256": taker._array_sha256(valid_idx, dtype=np.int32),
            "validation_first_decision_at_utc": frame["nominal_decision_at"].iloc[valid_idx].min().isoformat(),
            "validation_last_decision_at_utc": frame["nominal_decision_at"].iloc[valid_idx].max().isoformat(),
            "latest_train_label_available_at_utc": frame["target_label_available_at"].iloc[train_idx].max().isoformat(),
        })
    source_calibration = taker._read_json(taker.SOURCE_RUN_PATH)["stages"]["calibration_oof"]
    existing_base64_check = {
        "status": "incompatible; one common-sample BASE64 CV is required",
        "source_training_rows": int(taker.SOURCE_TRAINING["training_rows"]),
        "shared_training_rows": int(common_summary["rows_after_shared_complete_feature_mask"]),
        "source_decision_rows": int(taker.SOURCE_TUNING["decision_rows"]),
        "shared_decision_rows": int(common_summary["decision_rows_after_shared_complete_feature_mask"]),
        "saved_source_oof_rows": int(sum(
            int(fold["validation_rows"]) for fold in source_calibration["result"]["folds"]
        )),
        "saved_compatible_fold_boosters": False,
        "reasons": [
            "The source full-development model used the larger, unmasked 64-feature sample.",
            "The shared BASE70 completeness mask changes train and validation rows.",
            "The source run saved a full-development model and OOF forecasts, not fold boosters fitted on this shared sample.",
            "Its fixed source-selected iteration is not the required common-sample early-stopping selection.",
        ],
    }

    records = []
    source_paths = [
        taker.CONFIG_PATH, taker.MANIFEST_PATH, taker.SOURCE_RUN_PATH, taker.SOURCE_CONFIG_PATH,
        taker.SOURCE_SELECTION_PATH, taker.SOURCE_TUNING_PATH,
        taker.SOURCE_TRAINING_PATH, taker.SOURCE_DATA_PATH,
        taker.SOURCE_TRAINING_PATH.parent / "lgbm_model.txt",
        taker.FEATURE_DATASET_METADATA_PATH, taker.SOURCE_CALIBRATION_PATH,
        taker.MARKET_INDEX_PATH,
        COMMON_DIR / "binance_source_validation.json",
        Path(__file__), CONFIG_PATH,
        model_update.fee_correction.CORRECTION_DIR / "research_manifest.json",
    ]
    records.extend(_record(p) for p in source_paths)
    for name in ("flow_features", "common_rows", "cv_B0"):
        records.extend(flow_manifest["stages"][name]["outputs"])
    for item in flow_manifest["stages"]["quality_and_economics"]["outputs"]:
        path = item["path"]
        if path.endswith(("quality_and_economics.json", "B0_B1_official_oof.parquet")) or "/economics/B0/" in f"/{path}":
            records.append(item)
    unique = {r["path"]: r for r in records}
    records = list(unique.values())
    for item in records:
        path = ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["size_bytes"]) or taker._sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen input hash mismatch: {item['path']}")

    fold_record = _record(FOLDS_PATH)
    mask_record = _record(mask_path)
    specification = {
        "experiment_id": CONFIG["experiment_id"],
        "comparison": "BASE70 - BASE64",
        "variant_definitions": {
            "BASE64": "Exact ordered 64 source features.",
            "BASE70": "BASE64 plus taker_Z and taker_V at 1, 5, and 15 minutes; existing B0.",
        },
        "feature_lists": {"BASE64": base64_features, "BASE70": base70_features},
        "shared_completeness_mask": {
            "rule": "Use exactly saved common rows: finite across all 64 source and 18 saved taker-flow features.",
            "source_rows_before_mask": int(common_summary["source_rows_after_fit_label_filter"]),
            "rows_after_mask": int(len(frame)),
            "excluded_rows": int(common_summary["rows_excluded_by_shared_complete_feature_mask"]),
            "decision_rows": int(common_summary["decision_rows_after_shared_complete_feature_mask"]),
            "source_row_indices_artifact": mask_record,
            "source_row_indices_sha256": taker._array_sha256(
                frame["_source_row_index"].to_numpy(dtype=np.int32), dtype=np.int32
            ),
        },
        "folds": {
            "count": taker.CV_FOLD_COUNT,
            "indices_artifact": fold_record,
            "indices": fold_records,
            "validation_decision_rows_only": True,
            "purge": "Keep train rows only when target_label_available_at <= first retained validation nominal_decision_at.",
        },
        "target_and_weights": {
            "target": "target_preopen_5m_up_binance_index_proxy",
            "formula": source["effective_config"]["contract"]["training_target"]["formula"],
            "decision_time": "T-60; nominal_decision_at is one minute before market start.",
            "target_sha256": taker._array_sha256(
                frame["target_preopen_5m_up_binance_index_proxy"].to_numpy(dtype=np.int8),
                dtype=np.int8,
            ),
            "decision_weight": taker.DECISION_WEIGHT,
            "auxiliary_weight": taker.AUXILIARY_WEIGHT,
            "weight_rule": "decision_weight; auxiliary rows receive (1 - decision_weight) / 4.",
            "weights_sha256": weight_hash,
            "validation_metrics_unweighted": True,
        },
        "model_cv": {
            "parameters": taker.MODEL_PARAMETERS,
            "seed": taker.MODEL_SEED,
            "fold_count": taker.CV_FOLD_COUNT,
            "backend": taker.CONFIG["model_comparison"]["compute_backend_from_source_effective_config"],
            "threads": taker.THREADS,
            "maximum_estimators": taker.MAX_ESTIMATORS,
            "early_stopping_rounds": taker.EARLY_STOPPING_ROUNDS,
            "iteration_selection": "min(mean fold binary_logloss + 0.5 * population fold SD)",
            "optuna": False,
        },
        "calibration": {
            "method": "Separate daily Platt calibration per model.",
            "minimum_prior_official_labels": int(taker.CONFIG["official_outcomes"]["minimum_calibration_rows_from_source_calibrator"]),
            "requires_both_classes": True,
            "label_release_rule": taker.CONFIG["official_outcomes"]["label_release_rule"],
            "fit_time": "Before first T-60 decision of each UTC day, using only earlier released official labels.",
            "period_start_utc": taker.CONFIG["economics"]["start_utc"],
            "period_cutoff_exclusive_utc": taker.CONFIG["economics"]["cutoff_exclusive_utc"],
        },
        "economics": {
            "gross_stake_usd": taker.CONFIG["economics"]["gross_stake_usd"],
            "minimum_net_ev_fraction": taker.CONFIG["economics"]["minimum_net_ev_fraction"],
            "decision_offset_seconds": taker.CONFIG["economics"]["decision_offset_seconds"],
            "execution_offset_seconds": taker.CONFIG["economics"]["execution_offset_seconds"],
            "fees_depth_cash_release": taker.CONFIG["economics"]["fees_depth_and_accounting"],
            "book_archive_replay": False,
        },
        "bootstrap": {
            "method": "Paired circular moving-block bootstrap over 3-day UTC blocks.",
            "replications": int(taker.CONFIG["bootstrap"]["replications"]),
            "seed": int(taker.CONFIG["bootstrap"]["seed"]),
            "includes_zero_trade_days": True,
        },
        "source_BASE64_reuse_check": existing_base64_check,
        "input_artifacts": records,
        "source_binance_archive_sha256_by_month": flow_manifest["stages"][
            "download_binance"
        ]["archive_sha256_by_month"],
        "source_code_hashes": source["source_hashes"]["source_code_sha256"],
        "correction_source_identity": correction_identity,
        "resource_snapshot_before_fit": taker._resource_snapshot(),
        "environment": {
            "python": __import__("sys").version,
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    identity_bytes = json.dumps(
        specification, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    frozen = {
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
        "specification": specification,
    }
    if MANIFEST_PATH.exists():
        manifest = taker._read_json(MANIFEST_PATH)
        existing_frozen = manifest.get("frozen_comparison", {})
        existing_specification = existing_frozen.get("specification", {})
        existing_identity = hashlib.sha256(json.dumps(
            existing_specification, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")).hexdigest()
        if existing_identity != existing_frozen.get("identity_sha256"):
            raise RuntimeError("Saved frozen comparison failed its identity check")

        current_for_resume = json.loads(json.dumps(specification, default=str))
        existing_for_resume = json.loads(json.dumps(existing_specification, default=str))
        for resume_specification in (current_for_resume, existing_for_resume):
            resume_specification.pop("resource_snapshot_before_fit", None)
        runner_path = _relative(Path(__file__))
        current_runner = next(
            (item for item in current_for_resume["input_artifacts"] if item["path"] == runner_path),
            None,
        )
        existing_runner = next(
            (item for item in existing_for_resume["input_artifacts"] if item["path"] == runner_path),
            None,
        )
        if current_runner is None or existing_runner is None:
            raise RuntimeError("Frozen input records do not contain the comparison runner")
        current_runner.update(existing_runner)
        if current_for_resume != existing_for_resume:
            raise RuntimeError(
                "Existing checkpoints have a different frozen data/model identity; "
                "preserve them and choose a new output directory."
            )

        frozen = existing_frozen
        current_runner_sha256 = taker._sha256(Path(__file__))
        if current_runner_sha256 != existing_runner["sha256"]:
            continuations = manifest.setdefault("checkpoint_continuations", [])
            if not continuations or continuations[-1].get("runner_sha256") != current_runner_sha256:
                continuations.append({
                    "continued_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "runner_sha256": current_runner_sha256,
                    "reason": "Aligned monthly economics with required decision-day PnL attribution; verified the frozen data/model identity and completed checkpoints.",
                })
    else:
        manifest = {
            "experiment_id": CONFIG["experiment_id"],
            "status": "frozen_before_fit",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "active_stage": None,
            "frozen_comparison": frozen,
            "stages": {
                "freeze": {
                    "status": "completed",
                    "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "outputs": [_record(COMMON_PATH), fold_record, _record(COMMON_SUMMARY_PATH), mask_record],
                }
            },
            "next_step": "Run the single common-sample BASE64 10-fold CV.",
        }
    manifest["frozen_comparison"] = frozen
    manifest["source_run_manifest"] = _relative(taker.MANIFEST_PATH)
    manifest["source_run_manifest_sha256"] = taker._sha256(taker.MANIFEST_PATH)
    manifest["runner_sha256"] = taker._sha256(Path(__file__))
    _write_manifest(manifest)
    return manifest, frame, base64_features, base70_features, folds


def _load_base64_arrays(metadata: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = [
        "_source_row_index", "Opened", "nominal_decision_at",
        "target_label_available_at", "target_preopen_5m_up_binance_index_proxy",
        "is_polymarket_preopen_decision", *features,
    ]
    frame = pd.read_parquet(COMMON_PATH, columns=columns)
    if not frame["_source_row_index"].equals(metadata["_source_row_index"]):
        raise RuntimeError("BASE64 feature load changed the frozen row order")
    x = frame.loc[:, features].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(x).all():
        raise RuntimeError("BASE64 matrix is not finite on the frozen common rows")
    y = frame["target_preopen_5m_up_binance_index_proxy"].to_numpy(dtype=np.int8, copy=True)
    weights = experiment._preopen_sample_weights(
        frame["is_polymarket_preopen_decision"].to_numpy(dtype=bool),
        taker.DECISION_WEIGHT,
    )
    expected_weights = taker._read_json(BASE70_CV_SUMMARY_PATH)["training_weight_sha256"]
    if taker._array_sha256(weights, dtype=np.float32) != expected_weights:
        raise RuntimeError("BASE64 weights differ from the frozen BASE70 weights")
    return x, y, weights


def _fit_base64(manifest: dict, metadata: pd.DataFrame, features: list[str], folds) -> dict:
    stage_name = "cv_BASE64"
    prior = manifest.get("stages", {}).get(stage_name, {})
    if prior.get("status") == "completed" and _verify_records(prior.get("outputs", [])):
        return prior["summary"]

    started = time.perf_counter()
    _begin_stage(
        manifest,
        stage_name,
        "Score proxy and official quality, reproduce BASE70 calibration, and compare fixed-rule economics.",
    )
    BASE64_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history_path = OUTPUT_DIR / "BASE64_cv_history.json"
    checkpoint_path = OUTPUT_DIR / "BASE64_cv_fit_checkpoint.json"
    summary_path = OUTPUT_DIR / "BASE64_cv_summary.json"
    prediction_path = OUTPUT_DIR / "BASE64_decision_oof.parquet"
    frozen_identity = manifest["frozen_comparison"]["identity_sha256"]

    checkpoint = None
    if checkpoint_path.is_file() and history_path.is_file():
        candidate = taker._read_json(checkpoint_path)
        if (
            candidate.get("frozen_identity_sha256") == frozen_identity
            and _verify_records(candidate.get("outputs", []))
        ):
            checkpoint = candidate

    with taker.PeakMemorySampler() as memory:
        x, y, weights = _load_base64_arrays(metadata, features)
        decision = metadata["is_polymarket_preopen_decision"].to_numpy(dtype=bool)
        resumed_from_fit_checkpoint = checkpoint is not None
        if checkpoint is not None:
            history = taker._read_json(history_path)
            best_iteration = int(history["best_iteration"])
            boosters = [
                lgb.Booster(model_file=str(BASE64_MODEL_DIR / f"fold_{i:02d}.txt"))
                for i in range(taker.CV_FOLD_COUNT)
            ]
            if any(booster.current_iteration() != best_iteration for booster in boosters):
                raise RuntimeError("A BASE64 model checkpoint has an unexpected iteration count")
            if any(booster.feature_name() != features for booster in boosters):
                raise RuntimeError("A BASE64 model checkpoint has an unexpected feature order")
            params = checkpoint["params"]
            fit_elapsed = float(checkpoint["fit_elapsed_seconds"])
        else:
            train_set = lgb.Dataset(
                x, label=y, weight=weights, feature_name=features, free_raw_data=False
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
                "num_threads": taker.THREADS,
                "max_bin": 63,
                "feature_pre_filter": False,
                "gpu_use_dp": False,
                "seed": taker.MODEL_SEED,
                "feature_fraction_seed": taker.MODEL_SEED,
                "bagging_seed": taker.MODEL_SEED,
                "data_random_seed": taker.MODEL_SEED,
                **taker.MODEL_PARAMETERS,
            }
            fit_started = time.perf_counter()
            cv_results = lgb.cv(
                params=params,
                train_set=train_set,
                num_boost_round=taker.MAX_ESTIMATORS,
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
                        stopping_rounds=taker.EARLY_STOPPING_ROUNDS,
                    ),
                    lgb.log_evaluation(period=25),
                ],
                return_cvbooster=True,
                seed=taker.MODEL_SEED,
            )
            cvbooster = cv_results.pop("cvbooster")
            metric_series = cv_objective_series(
                cv_results, "binary_logloss", higher_is_better=False, std_penalty=0.5
            )
            objectives = metric_series["objective"]
            best_index = int(np.argmin(objectives))
            best_iteration = best_index + 1
            if len(cvbooster.boosters) != taker.CV_FOLD_COUNT:
                raise RuntimeError("BASE64 CV returned an unexpected fold count")
            callback_iteration = int(getattr(cvbooster, "best_iteration", 0) or 0)
            if callback_iteration > 0 and callback_iteration != best_iteration:
                raise RuntimeError("BASE64 callback and objective-series selection differ")
            history = {
                "metric_names": ["binary_logloss", "balanced_accuracy", "brier_score"],
                "selection_rule": "mean fold binary_logloss + 0.5 * population fold standard deviation",
                "best_iteration": best_iteration,
                "best_objective": float(objectives[best_index]),
                "iterations": [
                    {
                        "iteration": i + 1,
                        "binary_logloss_mean": float(metric_series["mean"][i]),
                        "binary_logloss_fold_std": float(metric_series["std"][i]),
                        "objective": float(objectives[i]),
                    }
                    for i in range(len(objectives))
                ],
            }
            taker._write_json(history_path, history)
            model_records = []
            for fold_id, booster in enumerate(cvbooster.boosters):
                model_path = BASE64_MODEL_DIR / f"fold_{fold_id:02d}.txt"
                temporary = model_path.with_suffix(".txt.tmp")
                booster.save_model(str(temporary), num_iteration=best_iteration)
                temporary.replace(model_path)
                model_records.append(_record(model_path))
            fit_elapsed = time.perf_counter() - fit_started
            checkpoint = {
                "frozen_identity_sha256": frozen_identity,
                "best_iteration": best_iteration,
                "params": params,
                "fit_elapsed_seconds": fit_elapsed,
                "outputs": [_record(history_path), *model_records],
            }
            _write_json(checkpoint_path, checkpoint)
            boosters = cvbooster.boosters
            del train_set, cv_results
            gc.collect()

        probabilities = np.full(len(metadata), np.nan, dtype=np.float32)
        fold_ids = np.full(len(metadata), -1, dtype=np.int8)
        fold_metrics = []
        for fold_id, ((_, valid_idx), booster) in enumerate(zip(folds, boosters)):
            p = np.asarray(booster.predict(x[valid_idx], num_iteration=best_iteration), dtype=np.float32)
            if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise RuntimeError(f"BASE64 fold {fold_id} produced invalid probabilities")
            probabilities[valid_idx] = p
            fold_ids[valid_idx] = fold_id
            fold_metrics.append({
                "fold_id": fold_id,
                "validation_start_utc": metadata["nominal_decision_at"].iloc[valid_idx].min().isoformat(),
                "validation_end_utc": metadata["nominal_decision_at"].iloc[valid_idx].max().isoformat(),
                **taker._score_binary(y[valid_idx], p),
            })
        predicted = np.flatnonzero(np.isfinite(probabilities))
        if not len(predicted) or not decision[predicted].all():
            raise RuntimeError("BASE64 OOF predictions are empty or include non-decision rows")
        output = metadata.iloc[predicted][[
            "_source_row_index", "Opened", "nominal_decision_at",
            "target_label_available_at", "target_preopen_5m_up_binance_index_proxy",
        ]].copy()
        output.rename(
            columns={"target_preopen_5m_up_binance_index_proxy": experiment.TARGET_COL},
            inplace=True,
        )
        output["market_start_utc"] = output["Opened"] + pd.Timedelta(minutes=2)
        output["fold_id"] = fold_ids[predicted].astype(str)
        output["p_model_raw"] = probabilities[predicted]
        output["prediction_available_at_utc"] = pd.NaT
        temporary_prediction = prediction_path.with_suffix(".parquet.tmp")
        output.to_parquet(temporary_prediction, index=False, compression="zstd")
        temporary_prediction.replace(prediction_path)

        selected = history["iterations"][best_iteration - 1]
        summary = {
            "variant": "BASE64",
            "feature_count": len(features),
            "feature_order": features,
            "rows": int(len(metadata)),
            "decision_rows": int(decision.sum()),
            "oof_rows": int(len(output)),
            "best_iteration": best_iteration,
            "early_stopping_rounds": taker.EARLY_STOPPING_ROUNDS,
            "maximum_estimators": taker.MAX_ESTIMATORS,
            "params": params,
            "training_weight_sha256": taker._array_sha256(weights, dtype=np.float32),
            "decision_weight": taker.DECISION_WEIGHT,
            "auxiliary_row_weight": taker.AUXILIARY_WEIGHT,
            "validation_weights": "removed; unweighted decision rows",
            "best_cv_objective": history["best_objective"],
            "cv_binary_logloss_mean_at_best": selected["binary_logloss_mean"],
            "cv_binary_logloss_population_std_at_best": selected["binary_logloss_fold_std"],
            "fold_metrics": fold_metrics,
            "fit_checkpoint_reused": resumed_from_fit_checkpoint,
            "fit_elapsed_seconds": fit_elapsed,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_working_set_gib": round(memory.peak_bytes / 2**30, 3),
        }
        taker._write_json(summary_path, summary)
        del x, y, weights, probabilities, output, boosters
        gc.collect()

    outputs = [
        _record(prediction_path), _record(summary_path), _record(history_path),
        _record(checkpoint_path), *checkpoint["outputs"][1:],
    ]
    _finish_stage(
        manifest, stage_name, summary=summary, outputs=outputs,
        elapsed_seconds=time.perf_counter() - started,
        peak_process_working_set_gib=round(memory.peak_bytes / 2**30, 3),
        model_paths=[item["path"] for item in outputs if item["path"].endswith(".txt")],
        prediction_path=_relative(prediction_path),
    )
    return summary


def _assert_same_keys(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> None:
    if len(left) != len(right):
        raise RuntimeError(f"OOF row counts differ: {len(left)} versus {len(right)}")
    a = left.sort_values(keys, kind="stable").reset_index(drop=True)
    b = right.sort_values(keys, kind="stable").reset_index(drop=True)
    for key in keys:
        if pd.api.types.is_datetime64_any_dtype(a[key]) or isinstance(
            a[key].dtype, pd.DatetimeTZDtype
        ):
            equal = np.array_equal(
                pd.to_datetime(a[key], utc=True).array.asi8,
                pd.to_datetime(b[key], utc=True).array.asi8,
            )
        elif pd.api.types.is_numeric_dtype(a[key]) and pd.api.types.is_numeric_dtype(b[key]):
            equal = np.array_equal(
                a[key].to_numpy(dtype=np.float64),
                b[key].to_numpy(dtype=np.float64),
                equal_nan=True,
            )
        else:
            equal = a[key].astype("string").equals(b[key].astype("string"))
        if not equal:
            raise RuntimeError(f"OOF validation rows differ in key {key}")


def _metric_delta(bootstrap: dict) -> dict:
    return {
        "method": bootstrap["method"],
        "replications": bootstrap["replications"],
        "seed": bootstrap["seed"],
        "block_days": bootstrap["block_days"],
        "observed_utc_days": bootstrap["observed_utc_days"],
        "observed_rows": bootstrap["observed_rows"],
        "BASE70_minus_BASE64": {
            metric: {
                "median": value["b1_minus_b0_median"],
                "95pct": value["b1_minus_b0_95pct"],
                "replications_with_both_classes": value["replications_with_both_classes"],
            }
            for metric, value in bootstrap["intervals"].items()
        },
    }


def _monthly_forecast_metrics(frame: pd.DataFrame, p64: str, p70: str, *, calibrated: bool) -> pd.DataFrame:
    rows = []
    period = frame["market_start_utc"].dt.strftime("%Y-%m")
    mask = frame[p64].notna() & frame[p70].notna() if calibrated else np.ones(len(frame), dtype=bool)
    scoped = frame.loc[mask].copy()
    scoped["_month"] = period.loc[mask].to_numpy()
    label_col = "target_polymarket_up" if "target_polymarket_up" in scoped else experiment.TARGET_COL
    for month, group in scoped.groupby("_month", sort=True):
        y = group[label_col].to_numpy(dtype=np.int8)
        score64 = taker._score_binary(y, group[p64].to_numpy())
        score70 = taker._score_binary(y, group[p70].to_numpy())
        rows.append({
            "utc_month": month,
            "rows": len(group),
            "BASE64_log_loss": score64["log_loss"],
            "BASE70_log_loss": score70["log_loss"],
            "BASE70_minus_BASE64_log_loss": score70["log_loss"] - score64["log_loss"],
            "BASE64_brier": score64["brier"],
            "BASE70_brier": score70["brier"],
            "BASE70_minus_BASE64_brier": score70["brier"] - score64["brier"],
            "BASE64_auc": score64["auc"],
            "BASE70_auc": score70["auc"],
            "BASE70_minus_BASE64_auc": (
                score70["auc"] - score64["auc"]
                if score64["auc"] is not None and score70["auc"] is not None else None
            ),
            "calibrated": calibrated,
        })
    return pd.DataFrame(rows)


def _score_comparison(manifest: dict) -> dict:
    stage_name = "quality_comparison"
    prior = manifest.get("stages", {}).get(stage_name, {})
    if prior.get("status") == "completed" and _verify_records(prior.get("outputs", [])):
        return prior["summary"]
    started = time.perf_counter()
    _begin_stage(
        manifest, stage_name,
        "Reuse the saved BASE70 economic ledger, simulate BASE64, and calculate paired daily uncertainty.",
    )
    base64 = pd.read_parquet(OUTPUT_DIR / "BASE64_decision_oof.parquet")
    base70 = pd.read_parquet(BASE70_OOF_PATH)
    keys = [
        "_source_row_index", "Opened", "market_start_utc", "nominal_decision_at",
        experiment.TARGET_COL, "fold_id",
    ]
    _assert_same_keys(base64, base70, keys)
    paired = base70[keys + ["p_model_raw"]].rename(
        columns={"p_model_raw": "p_BASE70_raw"}
    ).merge(
        base64[keys + ["p_model_raw"]].rename(
            columns={"p_model_raw": "p_BASE64_raw"}
        ),
        on=keys, how="inner", validate="one_to_one",
    )
    if len(paired) != len(base64) or len(paired) != len(base70):
        raise RuntimeError("BASE64 and BASE70 OOF rows are not identical")
    target = paired[experiment.TARGET_COL].to_numpy(dtype=np.int8)
    proxy = {
        "rows": int(len(paired)),
        "period_start_utc": paired["market_start_utc"].min().isoformat(),
        "period_end_utc": paired["market_start_utc"].max().isoformat(),
        "utc_days": int(paired["market_start_utc"].dt.floor("D").nunique()),
        "BASE64": taker._score_binary(target, paired["p_BASE64_raw"].to_numpy()),
        "BASE70": taker._score_binary(target, paired["p_BASE70_raw"].to_numpy()),
        "by_fold": {},
    }
    fold_rows = []
    for fold_id, rows in paired.groupby("fold_id", sort=True):
        y = rows[experiment.TARGET_COL].to_numpy(dtype=np.int8)
        score64 = taker._score_binary(y, rows["p_BASE64_raw"].to_numpy())
        score70 = taker._score_binary(y, rows["p_BASE70_raw"].to_numpy())
        entry = {"BASE64": score64, "BASE70": score70}
        proxy["by_fold"][str(fold_id)] = entry
        fold_rows.append({
            "fold_id": str(fold_id),
            "rows": len(rows),
            "BASE64_log_loss": score64["log_loss"],
            "BASE70_log_loss": score70["log_loss"],
            "BASE70_minus_BASE64_log_loss": score70["log_loss"] - score64["log_loss"],
            "BASE64_brier": score64["brier"],
            "BASE70_brier": score70["brier"],
            "BASE70_minus_BASE64_brier": score70["brier"] - score64["brier"],
            "BASE64_auc": score64["auc"],
            "BASE70_auc": score70["auc"],
            "BASE70_minus_BASE64_auc": score70["auc"] - score64["auc"],
        })
    proxy_boot = taker._paired_block_bootstrap(
        target,
        paired["p_BASE64_raw"].to_numpy(),
        paired["p_BASE70_raw"].to_numpy(),
        paired["market_start_utc"],
        replications=int(taker.CONFIG["bootstrap"]["replications"]),
        seed=int(taker.CONFIG["bootstrap"]["seed"]),
    )

    cached_official = pd.read_parquet(BASE70_OFFICIAL_PATH)
    _assert_same_keys(
        cached_official,
        base70.merge(cached_official[keys].drop_duplicates(), on=keys, how="inner"),
        keys,
    )
    official = cached_official.merge(
        base64[keys + ["p_model_raw"]].rename(
            columns={"p_model_raw": "p_BASE64_raw"}
        ),
        on=keys, how="inner", validate="one_to_one",
    )
    if len(official) != len(cached_official):
        raise RuntimeError("Official BASE64 and BASE70 development forecast rows differ")
    official["p_BASE70_raw"] = official["p_B0_raw"].astype(np.float64)

    def calibrate_column(rows: pd.DataFrame, probability_column: str):
        calibration_input = rows[[
            "market_start_utc", "nominal_decision_at", "resolved_at_utc",
            "target_polymarket_up", probability_column,
        ]].copy()
        calibration_input.rename(
            columns={probability_column: "p_model_raw"}, inplace=True
        )
        calibrated, log = taker._causal_daily_platt(calibration_input)
        return calibrated["p_model_platt"].to_numpy(dtype=np.float64), log

    p70_calibrated, calibration_log70 = calibrate_column(official, "p_BASE70_raw")
    cache_calibrated = official["p_B0_platt"].to_numpy(dtype=np.float64)
    same_cache_mask = np.isfinite(p70_calibrated) == np.isfinite(cache_calibrated)
    jointly_calibrated = np.isfinite(p70_calibrated) & np.isfinite(cache_calibrated)
    cache_abs_diff = (
        float(np.max(np.abs(p70_calibrated[jointly_calibrated] - cache_calibrated[jointly_calibrated])))
        if jointly_calibrated.any() else 0.0
    )
    if not same_cache_mask.all() or not np.allclose(
        p70_calibrated, cache_calibrated, rtol=0, atol=1e-12, equal_nan=True
    ):
        raise RuntimeError(
            f"Recreated BASE70 daily Platt values differ from the cache; max absolute difference={cache_abs_diff}"
        )
    p64_calibrated, calibration_log64 = calibrate_column(official, "p_BASE64_raw")
    official["p_BASE64_platt"] = p64_calibrated
    official["p_BASE70_platt"] = p70_calibrated
    mask_calibrated = np.isfinite(p64_calibrated) & np.isfinite(p70_calibrated)
    if not np.array_equal(np.isfinite(p64_calibrated), np.isfinite(p70_calibrated)):
        raise RuntimeError("The daily Platt availability masks differ between models")

    official_y = official["target_polymarket_up"].to_numpy(dtype=np.int8)
    official_result = {
        "rows": int(len(official)),
        "calibrated_rows": int(mask_calibrated.sum()),
        "period_start_utc": official["market_start_utc"].min().isoformat(),
        "period_end_utc": official["market_start_utc"].max().isoformat(),
        "utc_days": int(official["market_start_utc"].dt.floor("D").nunique()),
        "raw": {
            "BASE64": taker._score_binary(official_y, official["p_BASE64_raw"].to_numpy()),
            "BASE70": taker._score_binary(official_y, official["p_BASE70_raw"].to_numpy()),
        },
        "daily_platt": {
            "BASE64": taker._score_binary(
                official_y[mask_calibrated], p64_calibrated[mask_calibrated]
            ),
            "BASE70": taker._score_binary(
                official_y[mask_calibrated], p70_calibrated[mask_calibrated]
            ),
        },
        "base70_cache_reproduction": {
            "passed": True,
            "rows": int(len(cache_calibrated)),
            "calibrated_rows": int(np.isfinite(cache_calibrated).sum()),
            "maximum_absolute_probability_difference": cache_abs_diff,
            "calibration_days": len(calibration_log70),
        },
        "calibration_logs": {"BASE64": calibration_log64, "BASE70": calibration_log70},
        "by_fold_raw": {},
    }
    for fold_id, rows in official.groupby("fold_id", sort=True):
        y = rows["target_polymarket_up"].to_numpy(dtype=np.int8)
        official_result["by_fold_raw"][str(fold_id)] = {
            "BASE64": taker._score_binary(y, rows["p_BASE64_raw"].to_numpy()),
            "BASE70": taker._score_binary(y, rows["p_BASE70_raw"].to_numpy()),
        }
    official_boot_raw = taker._paired_block_bootstrap(
        official_y,
        official["p_BASE64_raw"].to_numpy(),
        official["p_BASE70_raw"].to_numpy(),
        official["market_start_utc"],
        replications=int(taker.CONFIG["bootstrap"]["replications"]),
        seed=int(taker.CONFIG["bootstrap"]["seed"]),
    )
    calibrated_boot = taker._paired_block_bootstrap(
        official_y[mask_calibrated],
        p64_calibrated[mask_calibrated],
        p70_calibrated[mask_calibrated],
        official.loc[mask_calibrated, "market_start_utc"],
        replications=int(taker.CONFIG["bootstrap"]["replications"]),
        seed=int(taker.CONFIG["bootstrap"]["seed"]),
    )
    official_result["paired_bootstrap"] = {
        "raw": _metric_delta(official_boot_raw),
        "daily_platt_common_mask": _metric_delta(calibrated_boot),
    }
    official_months = pd.concat([
        _monthly_forecast_metrics(
            official.rename(columns={
                "p_BASE64_raw": "p64", "p_BASE70_raw": "p70"
            }),
            "p64", "p70", calibrated=False,
        ),
        _monthly_forecast_metrics(
            official.rename(columns={
                "p_BASE64_platt": "p64", "p_BASE70_platt": "p70"
            }),
            "p64", "p70", calibrated=True,
        ),
    ], ignore_index=True)

    proxy_output = OUTPUT_DIR / "BASE64_BASE70_proxy_oof.parquet"
    official_output = OUTPUT_DIR / "BASE64_BASE70_official_oof.parquet"
    fold_csv = OUTPUT_DIR / "proxy_fold_metrics.csv"
    month_csv = OUTPUT_DIR / "official_month_metrics.csv"
    quality_path = OUTPUT_DIR / "quality_comparison.json"
    paired.to_parquet(proxy_output, index=False, compression="zstd")
    official.to_parquet(official_output, index=False, compression="zstd")
    pd.DataFrame(fold_rows).to_csv(fold_csv, index=False)
    official_months.to_csv(month_csv, index=False)
    summary = {
        "proxy_oof": {
            **proxy,
            "paired_bootstrap_BASE70_minus_BASE64": _metric_delta(proxy_boot),
        },
        "official_development": official_result,
        "quality_stage_elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(quality_path, summary)
    outputs = [
        _record(proxy_output), _record(official_output), _record(fold_csv),
        _record(month_csv), _record(quality_path),
    ]
    _finish_stage(
        manifest, stage_name, summary=summary, outputs=outputs,
        elapsed_seconds=time.perf_counter() - started,
    )
    return summary


def _build_daily_economics(
    trades: pd.DataFrame, start: pd.Timestamp, cutoff: pd.Timestamp
) -> pd.DataFrame:
    first_day = pd.Timestamp(start).tz_convert("UTC").floor("D")
    last_day = (pd.Timestamp(cutoff).tz_convert("UTC") - pd.Timedelta(nanoseconds=1)).floor("D")
    days = pd.date_range(first_day, last_day, freq="D", tz="UTC")
    daily = pd.DataFrame({"utc_day": days, "net_pnl_usd": 0.0, "turnover_usd": 0.0})
    if trades.empty:
        return daily
    settled = pd.to_datetime(trades["settled_at_utc"], utc=True, errors="coerce").notna()
    release = pd.to_datetime(trades["cash_release_at_utc"], utc=True, errors="coerce")
    eligible = settled & release.lt(pd.Timestamp(cutoff).tz_convert("UTC"))
    eligible &= pd.to_numeric(trades["net_pnl_usd"], errors="coerce").notna()
    settled_trades = trades.loc[eligible].copy()
    if settled_trades.empty:
        return daily
    settled_trades["_utc_day"] = pd.to_datetime(
        settled_trades["decision_time_utc"], utc=True
    ).dt.floor("D")
    totals = settled_trades.groupby("_utc_day", sort=True).agg(
        net_pnl_usd=("net_pnl_usd", "sum"),
        turnover_usd=("gross_usd", "sum"),
    )
    daily = daily.set_index("utc_day")
    daily.loc[totals.index, ["net_pnl_usd", "turnover_usd"]] = totals[
        ["net_pnl_usd", "turnover_usd"]
    ]
    return daily.reset_index()


def _economic_bootstrap(
    daily64: pd.DataFrame,
    daily70: pd.DataFrame,
    *,
    replications: int,
    seed: int,
    block_days: int = 3,
) -> dict:
    if not daily64["utc_day"].equals(daily70["utc_day"]):
        raise RuntimeError("Economic bootstrap series do not share the same UTC calendar")
    n_days = len(daily64)
    if n_days < block_days:
        raise RuntimeError("Economic bootstrap requires at least one full block")
    blocks = int(math.ceil(n_days / block_days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_days, size=(replications, blocks))
    offsets = np.arange(block_days, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n_days
    indices = indices.reshape(replications, -1)[:, :n_days]

    pnl64 = daily64["net_pnl_usd"].to_numpy(dtype=np.float64)
    pnl70 = daily70["net_pnl_usd"].to_numpy(dtype=np.float64)
    turnover64 = daily64["turnover_usd"].to_numpy(dtype=np.float64)
    turnover70 = daily70["turnover_usd"].to_numpy(dtype=np.float64)
    pnl64_sum = pnl64[indices].sum(axis=1)
    pnl70_sum = pnl70[indices].sum(axis=1)
    turnover64_sum = turnover64[indices].sum(axis=1)
    turnover70_sum = turnover70[indices].sum(axis=1)
    ratio64 = np.divide(
        pnl64_sum, turnover64_sum, out=np.full(replications, np.nan),
        where=turnover64_sum > 0,
    )
    ratio70 = np.divide(
        pnl70_sum, turnover70_sum, out=np.full(replications, np.nan),
        where=turnover70_sum > 0,
    )

    def interval(values: np.ndarray, *, scale: float = 1.0) -> dict:
        valid = values[np.isfinite(values)] / scale
        return {
            "median": float(np.median(valid)) if len(valid) else None,
            "95pct": (
                [float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))]
                if len(valid) else [None, None]
            ),
            "valid_replications": int(len(valid)),
        }

    delta_mean = (pnl70_sum - pnl64_sum) / n_days
    return {
        "method": "paired circular moving-block bootstrap over 3-day UTC blocks; resample jointly and truncate to the observed calendar length",
        "replications": int(replications),
        "seed": int(seed),
        "block_days": int(block_days),
        "calendar_days": int(n_days),
        "BASE64_mean_daily_pnl_usd": interval(pnl64_sum, scale=n_days),
        "BASE70_mean_daily_pnl_usd": interval(pnl70_sum, scale=n_days),
        "BASE70_minus_BASE64_mean_daily_pnl_usd": interval(delta_mean),
        "BASE64_pnl_per_turnover": interval(ratio64),
        "BASE70_pnl_per_turnover": interval(ratio70),
        "valid_ratio_replications": {
            "BASE64": int(np.isfinite(ratio64).sum()),
            "BASE70": int(np.isfinite(ratio70).sum()),
        },
    }


def _summarize_economics(
    variant: str, result: dict, output_root: Path, start: pd.Timestamp, cutoff: pd.Timestamp
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = taker._economic_summary_from_artifacts(result, output_root)
    summary["variant"] = variant
    policy_dir = output_root / "fixed_5_usd_ev_1pct"
    decisions = pd.read_parquet(policy_dir / "decisions.parquet")
    trades = pd.read_parquet(policy_dir / "trades.parquet")
    daily = _build_daily_economics(trades, start, cutoff)
    usable = trades.loc[
        pd.to_datetime(trades["settled_at_utc"], utc=True, errors="coerce").notna()
        & pd.to_datetime(trades["cash_release_at_utc"], utc=True, errors="coerce").lt(cutoff)
        & pd.to_numeric(trades["net_pnl_usd"], errors="coerce").notna()
    ].copy()
    ledger_pnl = float(pd.to_numeric(usable["net_pnl_usd"], errors="coerce").sum())
    if not np.isclose(ledger_pnl, float(summary["net_pnl_usd"]), rtol=0, atol=1e-7):
        raise RuntimeError(
            f"{variant} daily settlement ledger PnL {ledger_pnl:.10f} differs from simulator total {summary['net_pnl_usd']:.10f}"
        )
    if int(len(usable)) != int(summary["settlement_count"]):
        raise RuntimeError(
            f"{variant} settled execution count differs from the simulator summary"
        )
    ev = float(pd.to_numeric(usable.get("chosen_side_ev_usd"), errors="coerce").sum())
    summary["forecast_net_ev_at_t60_for_filled_trades_usd"] = ev
    summary["realized_net_pnl_for_filled_trades_usd"] = ledger_pnl
    summary["forecast_ev_minus_realized_pnl_usd"] = ev - ledger_pnl
    summary["decision_calendar_rows"] = int(len(decisions))
    summary["fills_outside_settlement_cutoff"] = int(len(trades) - len(usable))

    decisions["_utc_month"] = pd.to_datetime(decisions["decision_time_utc"], utc=True).dt.strftime("%Y-%m")
    usable["_utc_month"] = pd.to_datetime(usable["decision_time_utc"], utc=True).dt.strftime("%Y-%m")
    months = sorted(decisions["_utc_month"].unique())
    monthly = []
    for month in months:
        dm = decisions.loc[decisions["_utc_month"].eq(month)]
        tm = usable.loc[usable["_utc_month"].eq(month)]
        monthly.append({
            "variant": variant,
            "utc_month": month,
            "decisions": int(len(dm)),
            "fills": int(len(tm)),
            "settlements": int(pd.to_datetime(tm["settled_at_utc"], utc=True, errors="coerce").notna().sum()),
            "turnover_usd": float(pd.to_numeric(tm["gross_usd"], errors="coerce").sum()),
            "fees_usd": float(pd.to_numeric(tm["fee_usd"], errors="coerce").sum()),
            "net_pnl_usd": float(pd.to_numeric(tm["net_pnl_usd"], errors="coerce").sum()),
            "forecast_ev_usd": float(pd.to_numeric(tm["chosen_side_ev_usd"], errors="coerce").sum()),
        })
    daily["_utc_month"] = daily["utc_day"].dt.strftime("%Y-%m")
    daily_monthly = daily.groupby("_utc_month", sort=True).agg(
        net_pnl_usd=("net_pnl_usd", "sum"), turnover_usd=("turnover_usd", "sum")
    )
    for row in monthly:
        totals = daily_monthly.loc[row["utc_month"]]
        if not np.isclose(row["net_pnl_usd"], totals["net_pnl_usd"], rtol=0, atol=1e-7):
            raise RuntimeError(f"{variant} monthly PnL does not reconcile to the daily ledger")
    daily = daily.drop(columns="_utc_month")

    daily_profit = daily.groupby("utc_day", sort=True)["net_pnl_usd"].sum().sort_values(ascending=False)
    top5 = daily_profit.head(5)
    summary["top_five_decision_days"] = [
        {"utc_day": day.isoformat(), "net_pnl_usd": float(pnl)}
        for day, pnl in top5.items()
    ]
    summary["top_five_days_share_of_total_pnl"] = (
        float(top5.sum() / ledger_pnl) if ledger_pnl else None
    )
    reason_counts = (
        decisions.loc[decisions["decision"].eq("skip"), "skip_reason"]
        .value_counts(dropna=True).to_dict()
    )
    summary["skip_reasons"] = {str(key): int(value) for key, value in reason_counts.items()}
    summary["missing_required_data_statuses"] = {
        reason: count for reason, count in summary["skip_reasons"].items()
        if reason.startswith("missing_")
    }
    summary["daily_pnl_calendar_days"] = int(len(daily))
    summary["daily_pnl_start_utc"] = daily["utc_day"].min().isoformat()
    summary["daily_pnl_end_utc"] = daily["utc_day"].max().isoformat()
    summary["monthly_results"] = monthly
    return summary, decisions, usable, daily


def _run_economics(manifest: dict, quality: dict) -> dict:
    stage_name = "economics_and_uncertainty"
    prior = manifest.get("stages", {}).get(stage_name, {})
    if prior.get("status") == "completed" and _verify_records(prior.get("outputs", [])):
        monthly_path = OUTPUT_DIR / "monthly_economic_summary.csv"
        try:
            existing_months = pd.read_csv(monthly_path)
        except pd.errors.EmptyDataError:
            existing_months = pd.DataFrame()
        required_month_columns = {
            "variant", "utc_month", "decisions", "fills", "settlements",
            "turnover_usd", "fees_usd", "net_pnl_usd", "forecast_ev_usd",
        }
        if existing_months.empty or not required_month_columns.issubset(existing_months.columns):
            start = pd.Timestamp(taker.CONFIG["economics"]["start_utc"])
            cutoff = pd.Timestamp(taker.CONFIG["economics"]["cutoff_exclusive_utc"])
            output_root = OUTPUT_DIR / "economics/BASE64"
            base64_result = taker._read_json(output_root / "BASE64_portfolio_results.json")
            base70_result = taker._read_json(BASE70_ECON_DIR / "B0_portfolio_results.json")
            summary64, *_ = _summarize_economics(
                "BASE64", base64_result, output_root, start, cutoff
            )
            summary70, *_ = _summarize_economics(
                "BASE70", base70_result, BASE70_ECON_DIR, start, cutoff
            )
            month_rows = [
                *summary64["monthly_results"], *summary70["monthly_results"]
            ]
            if not month_rows:
                raise RuntimeError("Saved economics ledgers did not produce monthly summary rows")
            pd.DataFrame(month_rows).to_csv(monthly_path, index=False)
            monthly_record = _record(monthly_path)
            prior["outputs"] = [
                monthly_record if item["path"] == _relative(monthly_path) else item
                for item in prior["outputs"]
            ]
            _write_manifest(manifest)
        for variant in ("BASE64", "BASE70"):
            prior["summary"][variant]["variant"] = variant
        return prior["summary"]
    started = time.perf_counter()
    _begin_stage(manifest, stage_name, "Write the final report and complete the experiment.")
    _, _, _, _, _, correction_manifest, book_record, correction_identity = model_update._source_inputs()
    start = pd.Timestamp(taker.CONFIG["economics"]["start_utc"])
    cutoff = pd.Timestamp(taker.CONFIG["economics"]["cutoff_exclusive_utc"])
    policy_name = "fixed_5_usd_ev_1pct"

    base70_result = taker._read_json(BASE70_ECON_DIR / "B0_portfolio_results.json")
    base70_summary = taker._economic_summary_from_artifacts(base70_result, BASE70_ECON_DIR)
    base70_prior = taker._read_json(BASE70_QUALITY_PATH)["economic_comparison"]["B0"]
    for key, value in base70_summary.items():
        prior_value = base70_prior.get(key)
        if isinstance(value, (int, float)) and isinstance(prior_value, (int, float)):
            matches = np.isclose(float(value), float(prior_value), rtol=0, atol=1e-7)
        else:
            matches = value == prior_value
        if not matches:
            raise RuntimeError(f"Saved BASE70 economic summary changed at {key}")
    if (
        base70_summary["fill_count"] != 1452
        or not np.isclose(base70_summary["gross_turnover_usd"], 7260.0, rtol=0, atol=1e-8)
        or not np.isclose(base70_summary["net_pnl_usd"], 260.04642335283586, rtol=0, atol=0.01)
    ):
        raise RuntimeError("Saved BASE70 result does not match the previously reported anchor")

    prediction_frame = pd.read_parquet(OUTPUT_DIR / "BASE64_BASE70_official_oof.parquet")
    decision_predictions = prediction_frame[[
        "market_start_utc", "Opened", "nominal_decision_at", "resolved_at_utc", "fold_id",
        "target_polymarket_up", experiment.TARGET_COL,
        "p_BASE64_raw", "p_BASE64_platt",
    ]].copy()
    decision_predictions.rename(
        columns={"p_BASE64_raw": "p_model_raw", "p_BASE64_platt": "p_model_platt"},
        inplace=True,
    )
    decision_predictions["prediction_available_at_utc"] = pd.NaT
    output_root = OUTPUT_DIR / "economics/BASE64"
    result64 = research._evaluate_portfolio_period(
        correction_manifest,
        decision_predictions,
        start=start,
        cutoff=cutoff,
        policies=[(policy_name, "fixed_5_usd", float(taker.CONFIG["economics"]["minimum_net_ev_fraction"]))],
        output_root=output_root,
        period_name="BASE64",
        input_identity={
            "experiment_id": CONFIG["experiment_id"],
            "variant": "BASE64",
            "official_oof_predictions_sha256": taker._sha256(
                OUTPUT_DIR / "BASE64_BASE70_official_oof.parquet"
            ),
            "official_market_index_sha256": taker._sha256(taker.MARKET_INDEX_PATH),
            "t59_snapshot_sha256": book_record["t59_coverage_sha256"],
            "t60_snapshot_sha256": book_record["t60_coverage_sha256"],
            "period_start_utc": start.isoformat(),
            "period_cutoff_exclusive_utc": cutoff.isoformat(),
            "policy": "fixed_5_usd with 1% minimum net EV; existing T-60/T-59 engine",
            "calibration_schedule": "separate causal daily Platt using earlier official labels only",
        },
    )
    summary64, decisions64, trades64, daily64 = _summarize_economics(
        "BASE64", result64, output_root, start, cutoff
    )
    summary70, decisions70, trades70, daily70 = _summarize_economics(
        "BASE70", base70_result, BASE70_ECON_DIR, start, cutoff
    )
    decision_keys = ["market_start_utc", "decision_time_utc", "entry_time_utc"]
    _assert_same_keys(decisions64, decisions70, decision_keys)
    if not daily64["utc_day"].equals(daily70["utc_day"]):
        raise RuntimeError("BASE64 and BASE70 daily economics calendars differ")
    bootstrap = _economic_bootstrap(
        daily64,
        daily70,
        replications=int(taker.CONFIG["bootstrap"]["replications"]),
        seed=int(taker.CONFIG["bootstrap"]["seed"]),
        block_days=3,
    )
    daily = daily64[["utc_day", "net_pnl_usd", "turnover_usd"]].rename(
        columns={
            "net_pnl_usd": "BASE64_net_pnl_usd",
            "turnover_usd": "BASE64_turnover_usd",
        }
    ).merge(
        daily70[["utc_day", "net_pnl_usd", "turnover_usd"]].rename(
            columns={
                "net_pnl_usd": "BASE70_net_pnl_usd",
                "turnover_usd": "BASE70_turnover_usd",
            }
        ),
        on="utc_day", how="inner", validate="one_to_one",
    )
    daily["BASE70_minus_BASE64_net_pnl_usd"] = (
        daily["BASE70_net_pnl_usd"] - daily["BASE64_net_pnl_usd"]
    )
    daily["BASE70_minus_BASE64_turnover_usd"] = (
        daily["BASE70_turnover_usd"] - daily["BASE64_turnover_usd"]
    )
    if len(daily) != 77:
        raise RuntimeError(f"Expected the full 77-day development calendar, found {len(daily)}")

    summary = {
        "BASE64": summary64,
        "BASE70": summary70,
        "BASE70_previous_result_check": {
            "passed": True,
            "fills": int(summary70["fill_count"]),
            "turnover_usd": float(summary70["gross_turnover_usd"]),
            "net_pnl_usd": float(summary70["net_pnl_usd"]),
            "saved_calibration_cache_reproduced": quality["official_development"][
                "base70_cache_reproduction"
            ]["passed"],
            "saved_book_snapshot_identity": correction_identity,
            "economic_action": "Reused the hash-verified completed BASE70 simulation; did not replay it again.",
        },
        "paired_3day_economic_bootstrap": bootstrap,
        "calendar": {
            "start_utc": start.isoformat(),
            "cutoff_exclusive_utc": cutoff.isoformat(),
            "days": int(len(daily)),
            "decision_rows_each": int(len(decisions64)),
            "identical_decision_calendar": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    daily_path = OUTPUT_DIR / "daily_economic_pnl.csv"
    monthly_path = OUTPUT_DIR / "monthly_economic_summary.csv"
    summary_path = OUTPUT_DIR / "economic_comparison.json"
    daily.to_csv(daily_path, index=False)
    pd.concat([
        pd.DataFrame(summary64.pop("monthly_results", [])),
        pd.DataFrame(summary70.pop("monthly_results", [])),
    ], ignore_index=True).to_csv(monthly_path, index=False)
    _write_json(summary_path, summary)
    outputs = [
        _record(daily_path), _record(monthly_path), _record(summary_path),
        _record(output_root / policy_name / "decisions.parquet"),
        _record(output_root / policy_name / "trades.parquet"),
        _record(output_root / policy_name / "summary.json"),
        _record(output_root / policy_name / "slippage_sensitivity.json"),
        _record(output_root / "BASE64_portfolio_results.json"),
    ]
    _finish_stage(
        manifest, stage_name, summary=summary, outputs=outputs,
        elapsed_seconds=time.perf_counter() - started,
    )
    return summary


def _usd(value, digits: int = 2) -> str:
    return "$" + f"{float(value):,.{digits}f}"


def _write_final_report(manifest: dict) -> None:
    frozen = manifest["frozen_comparison"]["specification"]
    cv = manifest["stages"]["cv_BASE64"]["summary"]
    quality = manifest["stages"]["quality_comparison"]["summary"]
    economics = manifest["stages"]["economics_and_uncertainty"]["summary"]
    proxy = quality["proxy_oof"]
    official = quality["official_development"]
    econ64, econ70 = economics["BASE64"], economics["BASE70"]
    boot = economics["paired_3day_economic_bootstrap"]
    fold_table = pd.read_csv(OUTPUT_DIR / "proxy_fold_metrics.csv")
    month_quality = pd.read_csv(OUTPUT_DIR / "official_month_metrics.csv")
    month_econ = pd.read_csv(OUTPUT_DIR / "monthly_economic_summary.csv")
    folds_md = "\n".join(
        f"| {r.fold_id} | {int(r.rows):,} | {r.BASE64_log_loss:.6f} | {r.BASE70_log_loss:.6f} | "
        f"{r.BASE70_minus_BASE64_log_loss:+.6f} | {r.BASE64_brier:.6f} | {r.BASE70_brier:.6f} | "
        f"{r.BASE70_minus_BASE64_brier:+.6f} | {r.BASE64_auc:.4f} | {r.BASE70_auc:.4f} | "
        f"{r.BASE70_minus_BASE64_auc:+.4f} |"
        for r in fold_table.itertuples(index=False)
    )
    quality_md = []
    for calibrated, group in month_quality.groupby("calibrated", sort=False):
        kind = "Daily Platt" if bool(calibrated) else "Raw"
        for r in group.itertuples(index=False):
            quality_md.append(
                f"| {r.utc_month} | {kind} | {int(r.rows):,} | {r.BASE64_log_loss:.6f} | "
                f"{r.BASE70_log_loss:.6f} | {r.BASE70_minus_BASE64_log_loss:+.6f} | "
                f"{r.BASE64_brier:.6f} | {r.BASE70_brier:.6f} | "
                f"{r.BASE70_minus_BASE64_brier:+.6f} | {r.BASE64_auc:.4f} | "
                f"{r.BASE70_auc:.4f} | {r.BASE70_minus_BASE64_auc:+.4f} |"
            )
    econ_md = "\n".join(
        f"| {r.variant} | {int(r.decision_count):,} | {int(r.fill_count):,} | "
        f"{int(r.settlement_count):,} | {_usd(r.gross_turnover_usd)} | {_usd(r.estimated_fees_usd)} | "
        f"{_usd(r.net_pnl_usd)} | {r.pnl_per_dollar_turnover:.4%} | "
        f"{_usd(r.forecast_net_ev_at_t60_for_filled_trades_usd)} |"
        for r in pd.DataFrame([econ64, econ70]).itertuples(index=False)
    )
    monthly_md = "\n".join(
        f"| {r.utc_month} | {r.variant} | {int(r.decisions):,} | {int(r.fills):,} | "
        f"{_usd(r.turnover_usd)} | {_usd(r.fees_usd)} | {_usd(r.net_pnl_usd)} | "
        f"{_usd(r.forecast_ev_usd)} |"
        for r in month_econ.itertuples(index=False)
    )
    fold_wins = int((fold_table["BASE70_minus_BASE64_log_loss"] < 0).sum())
    raw_month = month_quality.loc[~month_quality["calibrated"].astype(bool)]
    calibrated_month = month_quality.loc[month_quality["calibrated"].astype(bool)]
    raw_wins = int((raw_month["BASE70_minus_BASE64_log_loss"] < 0).sum())
    calibrated_wins = int((calibrated_month["BASE70_minus_BASE64_log_loss"] < 0).sum())
    proxy_delta = proxy["paired_bootstrap_BASE70_minus_BASE64"]["BASE70_minus_BASE64"]
    raw_delta = official["paired_bootstrap"]["raw"]["BASE70_minus_BASE64"]
    cal_delta = official["paired_bootstrap"]["daily_platt_common_mask"]["BASE70_minus_BASE64"]
    daily_mean64 = econ64["net_pnl_usd"] / boot["calendar_days"]
    daily_mean70 = econ70["net_pnl_usd"] / boot["calendar_days"]

    report = f"""# BTC pre-open Z/V feature ablation — 2026-10-18

## Scope and frozen comparison

BASE64 is the exact ordered set of 64 source features. BASE70 is the saved B0 model: BASE64 plus taker_Z and taker_V at 1, 5, and 15 minutes. B1 is outside scope. This is a development-only comparison.

The comparison was frozen before fitting. It uses the completed taker-flow run's exact common mask and saved folds: {frozen['shared_completeness_mask']['rows_after_mask']:,} training rows, including {frozen['shared_completeness_mask']['decision_rows']:,} decision rows. The mask excludes {frozen['shared_completeness_mask']['excluded_rows']:,} otherwise eligible rows; BASE64 did not restore them. Proxy OOF: {proxy['rows']:,} paired forecasts, {proxy['utc_days']:,} UTC days, {proxy['period_start_utc']} through {proxy['period_end_utc']}. Official development: {official['rows']:,} resolved outcomes, {official['utc_days']:,} UTC days, {official['period_start_utc']} through {official['period_end_utc']}.

{"The completed BASE64 CV and quality checkpoints were reused after fixing the forecast projection and economics report assembly; their hashes matched the unchanged frozen data/model identity." if manifest.get('checkpoint_continuations') else ""}

The saved full-development BASE64 artifacts were incompatible because they used the larger unmasked sample and lacked fold boosters fitted on the shared rows. The only new fit was one fixed-parameter 10-fold BASE64 CV. Its selected iteration was {cv['best_iteration']}, using mean fold log loss + 0.5 × population fold standard deviation, {cv['early_stopping_rounds']} early-stopping rounds, and a {cv['maximum_estimators']}-round limit. No Optuna or extra configurations were used. BASE70 remains the original B0 at iteration {taker._read_json(BASE70_CV_SUMMARY_PATH)['best_iteration']}; its missing full iteration history was not reconstructed by retraining.

Target: {frozen['target_and_weights']['target']} ({frozen['target_and_weights']['formula']}); decision at T−60. The fixed training weights were {frozen['target_and_weights']['decision_weight']:.4f} for decision rows and {frozen['target_and_weights']['auxiliary_weight']:.4f} for auxiliary rows. Training labels were purged at each first validation decision.

## Proxy OOF quality

Lower log loss and Brier are better; higher AUC is better. Deltas are BASE70−BASE64.

| Fold | Rows | BASE64 LL | BASE70 LL | Δ LL | BASE64 Brier | BASE70 Brier | Δ Brier | BASE64 AUC | BASE70 AUC | Δ AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{folds_md}

Overall BASE64: LL {proxy['BASE64']['log_loss']:.6f}, Brier {proxy['BASE64']['brier']:.6f}, AUC {proxy['BASE64']['auc']:.4f}. BASE70: LL {proxy['BASE70']['log_loss']:.6f}, Brier {proxy['BASE70']['brier']:.6f}, AUC {proxy['BASE70']['auc']:.4f}.

Paired 3-day UTC block bootstrap, 2,000 replications (seed {taker.CONFIG['bootstrap']['seed']}), BASE70−BASE64 95% intervals: LL [{proxy_delta['log_loss']['95pct'][0]:+.6f}, {proxy_delta['log_loss']['95pct'][1]:+.6f}], Brier [{proxy_delta['brier']['95pct'][0]:+.6f}, {proxy_delta['brier']['95pct'][1]:+.6f}], AUC [{proxy_delta['auc']['95pct'][0]:+.4f}, {proxy_delta['auc']['95pct'][1]:+.4f}].

## Official development quality and calibration

| Score | Rows | BASE64 LL | BASE70 LL | BASE64 Brier | BASE70 Brier | BASE64 AUC | BASE70 AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | {official['rows']:,} | {official['raw']['BASE64']['log_loss']:.6f} | {official['raw']['BASE70']['log_loss']:.6f} | {official['raw']['BASE64']['brier']:.6f} | {official['raw']['BASE70']['brier']:.6f} | {official['raw']['BASE64']['auc']:.4f} | {official['raw']['BASE70']['auc']:.4f} |
| Daily Platt, common mask | {official['calibrated_rows']:,} | {official['daily_platt']['BASE64']['log_loss']:.6f} | {official['daily_platt']['BASE70']['log_loss']:.6f} | {official['daily_platt']['BASE64']['brier']:.6f} | {official['daily_platt']['BASE70']['brier']:.6f} | {official['daily_platt']['BASE64']['auc']:.4f} | {official['daily_platt']['BASE70']['auc']:.4f} |

Each model had its own daily Platt calibrator. It used only earlier official labels released before that UTC day's first decision, with at least 1,000 prior labels and both classes. Recreated BASE70 probabilities match the saved cache on {official['base70_cache_reproduction']['rows']:,} rows, with max absolute difference {official['base70_cache_reproduction']['maximum_absolute_probability_difference']:.3g}.

| UTC month | Score | Rows | BASE64 LL | BASE70 LL | Δ LL | BASE64 Brier | BASE70 Brier | Δ Brier | BASE64 AUC | BASE70 AUC | Δ AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(quality_md)}

Official raw bootstrap 95% intervals for BASE70−BASE64: LL [{raw_delta['log_loss']['95pct'][0]:+.6f}, {raw_delta['log_loss']['95pct'][1]:+.6f}], Brier [{raw_delta['brier']['95pct'][0]:+.6f}, {raw_delta['brier']['95pct'][1]:+.6f}], AUC [{raw_delta['auc']['95pct'][0]:+.4f}, {raw_delta['auc']['95pct'][1]:+.4f}]. Daily-Platt common-mask intervals: LL [{cal_delta['log_loss']['95pct'][0]:+.6f}, {cal_delta['log_loss']['95pct'][1]:+.6f}], Brier [{cal_delta['brier']['95pct'][0]:+.6f}, {cal_delta['brier']['95pct'][1]:+.6f}], AUC [{cal_delta['auc']['95pct'][0]:+.4f}, {cal_delta['auc']['95pct'][1]:+.4f}].

BASE70 had lower proxy log loss in {fold_wins}/10 folds, lower raw official log loss in {raw_wins}/{len(raw_month)} months, and lower calibrated official log loss in {calibrated_wins}/{len(calibrated_month)} months. Monthly results are descriptive and did not select thresholds or periods.

## Fixed-rule economics

Each model chose its own side and qualification on the same qualifying development calendar: independent $5 orders, 1% net-EV hurdle, T−60 decisions, T−59 execution, saved fees, depth, and settlement cash-release cutoff. Missing predictions and missing official outcome or settlement data remain separate statuses. These orders are not a portfolio return and are not scaled from $100.

| Model | Decisions | Fills | Settlements | Turnover | Fees | Net PnL | PnL / turnover | Forecast EV on settled fills |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{econ_md}

BASE70 reproduces the saved anchor: {economics['BASE70_previous_result_check']['fills']:,} fills, {_usd(economics['BASE70_previous_result_check']['turnover_usd'])} turnover, {_usd(economics['BASE70_previous_result_check']['net_pnl_usd'])} net PnL. Its hash-verified completed ledger was reused. BASE64 skip reasons and missing-data statuses are in economic_comparison.json.

| UTC decision month | Model | Decisions | Fills | Turnover | Fees | Net PnL | Forecast EV |
|---|---|---:|---:|---:|---:|---:|---:|
{monthly_md}

Five best decision days were {econ64['top_five_days_share_of_total_pnl']:.1%} of BASE64 net PnL and {econ70['top_five_days_share_of_total_pnl']:.1%} of BASE70 net PnL.

The paired economic bootstrap used all {boot['calendar_days']} UTC days, including days without trades, in shared circular 3-day blocks ({boot['replications']:,} replications; seed {boot['seed']}):

| Quantity | Estimate | 95% interval |
|---|---:|---:|
| BASE64 mean daily PnL | {_usd(daily_mean64, 4)} | {_usd(boot['BASE64_mean_daily_pnl_usd']['95pct'][0], 4)} to {_usd(boot['BASE64_mean_daily_pnl_usd']['95pct'][1], 4)} |
| BASE70 mean daily PnL | {_usd(daily_mean70, 4)} | {_usd(boot['BASE70_mean_daily_pnl_usd']['95pct'][0], 4)} to {_usd(boot['BASE70_mean_daily_pnl_usd']['95pct'][1], 4)} |
| BASE70−BASE64 mean daily PnL | {_usd(daily_mean70 - daily_mean64, 4)} | {_usd(boot['BASE70_minus_BASE64_mean_daily_pnl_usd']['95pct'][0], 4)} to {_usd(boot['BASE70_minus_BASE64_mean_daily_pnl_usd']['95pct'][1], 4)} |
| BASE64 PnL / turnover | {econ64['pnl_per_dollar_turnover']:.4%} | {boot['BASE64_pnl_per_turnover']['95pct'][0]:.4%} to {boot['BASE64_pnl_per_turnover']['95pct'][1]:.4%} |
| BASE70 PnL / turnover | {econ70['pnl_per_dollar_turnover']:.4%} | {boot['BASE70_pnl_per_turnover']['95pct'][0]:.4%} to {boot['BASE70_pnl_per_turnover']['95pct'][1]:.4%} |

Valid ratio replications: BASE64 {boot['valid_ratio_replications']['BASE64']:,}/{boot['replications']:,}; BASE70 {boot['valid_ratio_replications']['BASE70']:,}/{boot['replications']:,}. Replications with zero turnover have no ratio estimate.

## Conclusion and limits

1. BASE70 had slightly better aggregate point estimates for log loss, Brier, and AUC on proxy OOF and official development scores. The proxy AUC interval is positive; proxy log-loss and Brier intervals and all official intervals include zero. This is a limited proxy AUC signal, not broad evidence of a reliable quality improvement.
2. The effect was **{'distributed across multiple periods' if fold_wins >= 6 and calibrated_wins >= 2 else 'not consistently present across periods'}**: BASE70 won {fold_wins}/10 proxy folds and {calibrated_wins}/{len(calibrated_month)} calibrated official months by log loss.
3. BASE70 PnL minus BASE64 was {_usd(econ70['net_pnl_usd'] - econ64['net_pnl_usd'])}; the interval for the mean daily PnL difference was {_usd(boot['BASE70_minus_BASE64_mean_daily_pnl_usd']['95pct'][0], 4)} to {_usd(boot['BASE70_minus_BASE64_mean_daily_pnl_usd']['95pct'][1], 4)}.
4. BASE64 net PnL under the same sample and calibration was {_usd(econ64['net_pnl_usd'])}; a positive result was {'already present' if econ64['net_pnl_usd'] > 0 else 'not present'} without the additions.

The 64 selected features, hyperparameters, and iteration selection used development data. These intervals are conditional on that procedure and are not independent confirmation of generalization. Holdout labels and PnL were not evaluated; no deployment refit was run.

End-to-end time: {manifest.get('elapsed_seconds', 0):.1f}s. Sampled peak process working set: {f'{manifest.get("peak_run_working_set_gib"):.2f} GiB' if manifest.get('peak_run_working_set_gib') is not None else 'unavailable'}. BASE64 CV peak working set: {cv.get('peak_process_working_set_gib', 'n/a')} GiB. Hardware, stage times, and artifact hashes are in the run manifest.

Artifacts: {_relative(MANIFEST_PATH)}, {_relative(VERSIONED_MANIFEST_PATH)}, BASE64 fold models and CV history under {_relative(BASE64_MODEL_DIR)}, and OOF, tables, and daily PnL under {_relative(OUTPUT_DIR)}. The original BASE70 full iteration history was unavailable and was not recreated.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def _run() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = taker._load_source_artifacts()
    manifest, metadata, features64, _features70, folds = _freeze_comparison(source)
    fit = _fit_base64(manifest, metadata, features64, folds)
    del metadata, folds
    gc.collect()
    quality = _score_comparison(manifest)
    economics = _run_economics(manifest, quality)
    manifest["elapsed_seconds"] = (
        pd.Timestamp.now(tz="UTC") - pd.Timestamp(manifest["started_utc"])
    ).total_seconds()
    manifest["status"] = "completed"
    manifest["active_stage"] = None
    manifest["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    manifest["next_step"] = "Comparison complete; review the report and saved artifacts"
    manifest["summary"] = {
        "BASE64_best_iteration": fit["best_iteration"],
        "BASE64_proxy_oof": quality["proxy_oof"]["BASE64"],
        "BASE70_proxy_oof": quality["proxy_oof"]["BASE70"],
        "BASE64_official_raw": quality["official_development"]["raw"]["BASE64"],
        "BASE70_official_raw": quality["official_development"]["raw"]["BASE70"],
        "BASE64_official_platt": quality["official_development"]["daily_platt"]["BASE64"],
        "BASE70_official_platt": quality["official_development"]["daily_platt"]["BASE70"],
        "BASE64_net_pnl_usd": economics["BASE64"]["net_pnl_usd"],
        "BASE70_net_pnl_usd": economics["BASE70"]["net_pnl_usd"],
    }
    _write_final_report(manifest)
    manifest["stages"]["report"] = {
        "status": "completed",
        "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "outputs": [_record(REPORT_PATH)],
    }
    manifest["versioned_manifest_path"] = _relative(VERSIONED_MANIFEST_PATH)
    _write_manifest(manifest)
    _write_json(VERSIONED_MANIFEST_PATH, manifest)
    return manifest


def main() -> None:
    sampler = taker.PeakMemorySampler()
    sampler.__enter__()
    manifest = None
    try:
        manifest = _run()
        print(f"[btc-zv-ablation] completed: {_relative(REPORT_PATH)}", flush=True)
    except Exception as exc:
        if MANIFEST_PATH.is_file():
            manifest = taker._read_json(MANIFEST_PATH)
            manifest["status"] = "failed"
            manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
            manifest["failure"] = {
                "stage": manifest.get("active_stage"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            manifest["next_step"] = "Inspect the failed stage, then rerun this no-argument runner to resume verified checkpoints."
            _write_manifest(manifest)
        raise
    finally:
        sampler.__exit__(None, None, None)
        if manifest is not None:
            prior_peak = float(manifest.get("peak_run_working_set_gib", 0.0))
            cv_peak = float(
                manifest.get("stages", {}).get("cv_BASE64", {}).get("summary", {}).get(
                    "peak_process_working_set_gib", 0.0
                )
            )
            manifest["peak_run_working_set_gib"] = round(max(
                prior_peak, cv_peak, sampler.peak_bytes / 2**30
            ), 3)
            manifest["resource_snapshot_at_completion"] = taker._resource_snapshot()
            manifest["elapsed_seconds"] = (
                pd.Timestamp.now(tz="UTC") - pd.Timestamp(manifest["started_utc"])
            ).total_seconds()
            if manifest.get("status") == "completed":
                _write_final_report(manifest)
                manifest["stages"]["report"]["outputs"] = [_record(REPORT_PATH)]
                _write_json(VERSIONED_MANIFEST_PATH, manifest)
            _write_manifest(manifest)


if __name__ == "__main__":
    main()
