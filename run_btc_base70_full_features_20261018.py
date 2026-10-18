"""Run one frozen BASE70 feature_fraction comparison and its existing evaluations."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

import run_btc_base70_information_value_20261018 as base70
import run_btc_base70_seed_stability_20261018 as seed_stability
import run_btc_preopen_taker_flow_20261018 as taker


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_base70_full_features_20261018.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_directory"]
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
REPORT_PATH = ROOT / CONFIG["report_path"]
VERSIONED_MANIFEST_PATH = ROOT / CONFIG["versioned_manifest_path"]
STATUS_PATH = ROOT / CONFIG["status_path"]
STATUS_BEGIN = "<!-- BTC_BASE70_FULL_FEATURES_20261018_BEGIN -->"
STATUS_END = "<!-- BTC_BASE70_FULL_FEATURES_20261018_END -->"

SINGLE = "SINGLE"
FULL_FEATURES = "FULL_FEATURES"
ENSEMBLE = "ENSEMBLE"
VARIANTS = (SINGLE, FULL_FEATURES, ENSEMBLE)
CONTRASTS = {"FULL_FEATURES_MINUS_SINGLE": (FULL_FEATURES, SINGLE)}
PROBABILITY_COLUMNS = {
    SINGLE: "p_single_raw",
    FULL_FEATURES: "p_full_features_raw",
    ENSEMBLE: "p_ensemble_raw",
}
ROW_ID_COLUMNS = list(seed_stability.ROW_ID_COLUMNS)
TARGET_COLUMN = CONFIG["target"]
QUALITY_METRICS = ("log_loss", "brier", "auc")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _record(path: Path) -> dict:
    return {"path": _relative(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _verify_records(records: list[dict]) -> bool:
    return all(
        isinstance(item, dict)
        and {"path", "sha256", "size_bytes"}.issubset(item)
        and (ROOT / item["path"]).is_file()
        and (ROOT / item["path"]).stat().st_size == int(item["size_bytes"])
        and _sha256(ROOT / item["path"]) == item["sha256"]
        for item in records
    )


def _status_update(manifest: dict) -> None:
    completed = sorted(
        name for name, stage in manifest.get("stages", {}).items()
        if stage.get("status") == "completed"
    )
    completed_folds = sum(
        record.get("status") == "completed"
        for record in manifest.get("completed_fits", {}).values()
    )
    lines = [
        STATUS_BEGIN,
        "### BASE70 FULL_FEATURES experiment - 2026-10-18",
        f"- Updated UTC: `{pd.Timestamp.now(tz='UTC').isoformat()}`.",
        f"- Status: `{manifest.get('status')}`; active stage: `{manifest.get('active_stage') or 'none'}`.",
        f"- Completed stages: `{', '.join(completed) or 'none'}`; challenger folds: `{completed_folds}/10`.",
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
        text = text.replace(anchor, block + anchor, 1) if anchor in text else block + text
    temporary = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(STATUS_PATH)


def _write_manifest(manifest: dict) -> None:
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(MANIFEST_PATH, manifest)
    _status_update(manifest)


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


def _challenger_params(control_params: dict) -> dict:
    if float(control_params["feature_fraction"]) >= 1.0:
        raise RuntimeError("The saved SINGLE control does not use feature_fraction subsampling")
    challenger = copy.deepcopy(control_params)
    challenger["feature_fraction"] = 1.0
    differences = {
        key for key in set(control_params) | set(challenger)
        if control_params.get(key) != challenger.get(key)
    }
    if differences != {"feature_fraction"}:
        raise RuntimeError(f"Unexpected effective parameter differences: {sorted(differences)}")
    return challenger


def _effective_node_feature_fraction(params: dict, default: float = 1.0) -> float:
    aliases = ("feature_fraction_bynode", "sub_feature_bynode", "colsample_bynode")
    values = [float(params[name]) for name in aliases if name in params]
    if len(values) > 1 and any(value != values[0] for value in values[1:]):
        raise RuntimeError("Conflicting feature_fraction_bynode aliases are active")
    return values[0] if values else float(default)


def _load_sources() -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    flow_manifest, info_manifest, cv_summary, common_summary, source = seed_stability._source_inputs()
    seed_manifest_path = ROOT / CONFIG["source_seed_stability_run_manifest_path"]
    seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    seed_versioned = json.loads((ROOT / CONFIG["source_seed_stability_versioned_manifest_path"]).read_text(encoding="utf-8"))
    if seed_manifest.get("status") != "completed":
        raise RuntimeError("The saved BASE70 seed-stability run is incomplete")
    if seed_manifest.get("stages", {}).get("common_predictions", {}).get("status") != "completed":
        raise RuntimeError("The saved SINGLE/ENSEMBLE OOF predictions are incomplete")
    if not seed_stability._verify_records(seed_manifest["stages"]["common_predictions"].get("outputs", [])):
        raise RuntimeError("The saved SINGLE/ENSEMBLE OOF prediction hash differs")
    if seed_manifest.get("stages", {}).get("training", {}).get("summary", {}).get("all_fits_completed") is not True:
        raise RuntimeError("The source ensemble does not have all 40 completed component fits")
    local_identity = seed_manifest.get("frozen_comparison", {}).get("identity_sha256")
    versioned_identity = seed_versioned.get("frozen_comparison", {}).get("identity_sha256")
    if not local_identity or local_identity != versioned_identity:
        raise RuntimeError("The source seed-stability manifests have different frozen identities")
    if int(cv_summary["best_iteration"]) != int(CONFIG["boosting_iterations"]):
        raise RuntimeError("The saved SINGLE control does not use 31 iterations")
    if int(cv_summary["feature_count"]) != 70 or len(cv_summary["feature_order"]) != 70:
        raise RuntimeError("The saved control is not the exact 70-feature BASE70 matrix")
    if cv_summary["feature_order"] != seed_versioned["frozen_comparison"]["specification"]["data"]["feature_order"]:
        raise RuntimeError("The source seed-stability feature order differs from the saved CV control")
    saved_single_params = seed_versioned["frozen_comparison"]["specification"]["model_parameters_all_five"]["SINGLE"]["lightgbm_params_passed"]
    if saved_single_params != cv_summary["params"]:
        raise RuntimeError("The saved SINGLE effective parameters differ from the original CV summary")
    node_fraction = _effective_node_feature_fraction(cv_summary["params"])
    if node_fraction != 1.0:
        raise RuntimeError(
            "The SINGLE control has active feature_fraction_bynode sampling; it must be reviewed before any fit"
        )
    params = cv_summary["params"]
    if float(params.get("bagging_fraction", 1.0)) != 1.0 or int(params.get("bagging_freq", 0)) != 0:
        raise RuntimeError("The SINGLE control has active bagging; this experiment must not change it")
    if float(params.get("feature_fraction_bynode", 1.0)) != 1.0:
        raise RuntimeError("The SINGLE control has active node-level feature sampling")
    return flow_manifest, info_manifest, seed_manifest, seed_versioned, cv_summary, common_summary, source


def _source_artifacts(
    flow_manifest: dict,
    info_manifest: dict,
    seed_manifest: dict,
    source: dict,
) -> dict:
    source_outputs = source["source_records"]["source_outputs"]
    cv_records = {
        row["path"]: row
        for row in flow_manifest["stages"]["cv_B0"]["outputs"]
    }
    seed_predictions_record = seed_manifest["stages"]["common_predictions"]["outputs"][0]
    snapshot = info_manifest["frozen_comparison"]["specification"]["sources"]["decision_and_execution_snapshots"]
    snapshot_records = {
        name: {
            "path": snapshot[f"{name}_path"],
            "sha256": snapshot[f"{name}_sha256"],
            "size_bytes": (ROOT / snapshot[f"{name}_path"]).stat().st_size,
        }
        for name in ("t59", "t60")
    }
    models = [
        cv_records[_relative(seed_stability.BASE_MODEL_DIR / f"fold_{fold_id:02d}.txt")]
        for fold_id in range(10)
    ]
    source_documents = {
        "taker_flow_run_manifest": source["source_records"]["taker_flow_manifest"],
        "taker_flow_report": _record(ROOT / "reports/btc_preopen/btc_preopen_taker_flow_20261018.md"),
        "taker_flow_runner": source["source_records"]["taker_flow_runner"],
        "base70_information_run_manifest": source["source_records"]["base70_information_manifest"],
        "base70_information_versioned_manifest": source["source_records"]["base70_information_versioned_manifest"],
        "base70_information_report": source["source_records"]["base70_information_report"],
        "seed_stability_run_manifest": _record(ROOT / CONFIG["source_seed_stability_run_manifest_path"]),
        "seed_stability_versioned_manifest": _record(ROOT / CONFIG["source_seed_stability_versioned_manifest_path"]),
        "seed_stability_report": _record(ROOT / CONFIG["source_seed_stability_report_path"]),
        "seed_stability_runner": _record(Path(seed_stability.__file__)),
        "seed_stability_config": _record(seed_stability.CONFIG_PATH),
    }
    return {
        "matrix": source_outputs["matrix"],
        "folds": source_outputs["folds"],
        "single_oof": source_outputs["source_oof"],
        "single_cv_summary": source_outputs["source_cv_summary"],
        "single_boosters": models,
        "saved_single_and_ensemble_oof_bundle": seed_predictions_record,
        "common_execution_calendar": source_outputs["execution_calendar"],
        "single_raw_trades": source_outputs["source_trades"],
        "single_raw_decisions": source_outputs["source_decisions"],
        "single_raw_economic_summary": source_outputs["source_economic_summary"],
        "official_market_index": source["source_records"]["official_market_index"],
        "t59_snapshot": snapshot_records["t59"],
        "t60_snapshot": snapshot_records["t60"],
        "source_documents": source_documents,
    }


def _frozen_specification(
    info_manifest: dict,
    seed_manifest: dict,
    cv_summary: dict,
    common_summary: dict,
    artifacts: dict,
) -> dict:
    control_params = copy.deepcopy(cv_summary["params"])
    full_params = _challenger_params(control_params)
    source_single_record = seed_manifest["frozen_comparison"]["specification"]["model_parameters_all_five"]["SINGLE"]
    challenger_seed_roles = copy.deepcopy(source_single_record["seed_roles"])
    challenger_seed_roles["feature_fraction_seed"] = "inactive: feature_fraction=1.0; no per-tree feature subset is sampled"
    node_fraction = _effective_node_feature_fraction(control_params)
    return {
        "experiment_id": CONFIG["experiment_id"],
        "hypothesis": "With 70 selected features and weak signal, per-tree feature subsampling may reduce forecast stability; this is a hypothesis, not an established cause.",
        "contrast": "FULL_FEATURES minus SINGLE",
        "single": {
            "source_oof": artifacts["single_oof"],
            "source_seed_stability_oof_bundle": artifacts["saved_single_and_ensemble_oof_bundle"],
            "retrained": False,
            "iterations": int(CONFIG["boosting_iterations"]),
            "effective_lightgbm_params": control_params,
            "resolved_defaults_relevant_to_randomness": source_single_record["resolved_defaults_relevant_to_randomness"],
        },
        "challenger": {
            "name": FULL_FEATURES,
            "iterations": int(CONFIG["boosting_iterations"]),
            "effective_lightgbm_params": full_params,
            "only_intended_parameter_change": {
                "parameter": "feature_fraction",
                "control": float(control_params["feature_fraction"]),
                "challenger": 1.0,
            },
            "feature_fraction_bynode": {
                "effective_value": node_fraction,
                "explicitly_passed": any(key in control_params for key in ("feature_fraction_bynode", "sub_feature_bynode", "colsample_bynode")),
                "active_feature_subsampling": node_fraction < 1.0,
            },
            "seed_roles": challenger_seed_roles,
        },
        "data": {
            "matrix": artifacts["matrix"],
            "rows": int(cv_summary["rows"]),
            "feature_count": int(cv_summary["feature_count"]),
            "feature_order": list(cv_summary["feature_order"]),
            "target": TARGET_COLUMN,
            "folds": artifacts["folds"],
            "fold_count": 10,
            "fold_fingerprints": common_summary["common_fold_fingerprints"],
            "training_weight_sha256": cv_summary["training_weight_sha256"],
            "purge_rule": "reuse saved train/validation indices; label availability <= first validation nominal decision time",
            "dataset_construction": {
                "control_pattern": "lgb.Dataset(X, label, weight, feature_name, free_raw_data=False); lgb.cv applies params before construct; each fold uses Dataset.subset(sorted(train_idx))",
                "challenger_pattern": "same shared Dataset constructor, _update_params with challenger params, construct once, then Dataset.subset(sorted(train_idx)) per saved fold",
                "histogram_sampling_seed": int(control_params["data_random_seed"]),
                "dataset_loaded_once": True,
                "feature_generation_or_selection": False,
            },
        },
        "additional_context": {
            "ensemble_oof": artifacts["saved_single_and_ensemble_oof_bundle"],
            "rule": "reuse the saved equal-weight mean of all five historical component OOFs; do not retrain or select seeds",
        },
        "quality": {
            "proxy_oof_rows": 318560,
            "official_label_rows": 20718,
            "bootstrap": copy.deepcopy(CONFIG["bootstrap"]),
            "calibration": "none",
        },
        "economics": {
            **copy.deepcopy(CONFIG["economics"]),
            "calendar": artifacts["common_execution_calendar"],
            "snapshots": {"T59": artifacts["t59_snapshot"], "T60": artifacts["t60_snapshot"]},
            "single_ledger": artifacts["single_raw_trades"],
            "policy": "existing independent fixed-$5, net-EV >= 1%, T-60 decision and T-59 execution simulator, fees, depth, and settlement cutoff",
        },
        "source_documents": artifacts["source_documents"],
        "scope": {
            "challenger_fits": 10,
            "early_stopping": False,
            "optuna": False,
            "data_download": False,
            "archive_replay": False,
            "holdout_evaluation": False,
            "deployment_refit": False,
        },
    }


def _create_or_resume_manifest(spec: dict, artifacts: dict) -> dict:
    code_identity = {
        "runner_sha256": _sha256(Path(__file__)),
        "config_sha256": _sha256(CONFIG_PATH),
        "lightgbm_version": taker.lgb.__version__,
    }
    frozen_payload = {"specification": spec, "code_identity": code_identity}
    identity = _canonical_sha256(frozen_payload)
    previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else None
    completed_fits = {}
    stages = {}
    revisions = []
    single_anchor = None
    started_utc = pd.Timestamp.now(tz="UTC").isoformat()
    reused_frozen_identity = False
    if previous:
        previous_identity = previous.get("input_identity", {}).get("frozen_spec_sha256")
        if previous_identity != identity:
            if previous.get("completed_fits"):
                prior_spec = previous.get("frozen_comparison", {}).get("specification")
                prior_code = previous.get("input_identity", {})
                same_frozen_inputs = (
                    _canonical_sha256(prior_spec) == _canonical_sha256(spec)
                    and prior_code.get("config_sha256") == code_identity["config_sha256"]
                    and prior_code.get("lightgbm_version") == code_identity["lightgbm_version"]
                )
                if not same_frozen_inputs:
                    raise RuntimeError("Existing FULL_FEATURES checkpoints belong to different frozen inputs or parameters")
                completed_fits = previous.get("completed_fits", {})
                stages = previous.get("stages", {})
                single_anchor = previous.get("single_anchor_reproduction") or stages.get("economics_summary", {}).get("summary", {}).get("single_anchor_reproduction")
                revisions = list(previous.get("code_revision_history", []))
                if previous.get("input_identity", {}).get("active_runner_sha256") != code_identity["runner_sha256"]:
                    revisions.append({
                        "previous_runner_sha256": prior_code.get("active_runner_sha256", prior_code.get("runner_sha256")),
                        "current_runner_sha256": code_identity["runner_sha256"],
                        "reason": "Corrected the downstream saved OOF input path after all ten fits; the frozen model specification and fit implementation were unchanged. Existing fold checkpoints are revalidated before reuse.",
                        "completed_fit_count_preserved": len(completed_fits),
                    })
                started_utc = previous.get("started_utc", started_utc)
                identity = previous_identity
                reused_frozen_identity = True
            else:
                revisions = list(previous.get("code_revision_history", []))
                revisions.append({
                    "previous_frozen_sha256": previous_identity,
                    "current_frozen_sha256": identity,
                    "reason": "Frozen specification changed before any fold fit completed.",
                })
        else:
            completed_fits = previous.get("completed_fits", {})
            stages = previous.get("stages", {})
            single_anchor = previous.get("single_anchor_reproduction") or stages.get("economics_summary", {}).get("summary", {}).get("single_anchor_reproduction")
            revisions = previous.get("code_revision_history", [])
            started_utc = previous.get("started_utc", started_utc)
            reused_frozen_identity = True
    manifest = {
        "schema_version": 1,
        "experiment_id": CONFIG["experiment_id"],
        "status": "running",
        "active_stage": "frozen_preparation",
        "started_utc": started_utc,
        "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "frozen_at_utc": previous.get("frozen_at_utc") if previous and reused_frozen_identity else pd.Timestamp.now(tz="UTC").isoformat(),
        "next_step": "Verify the saved source artifacts and load the immutable BASE70 matrix once before the ten challenger fits.",
        "frozen_comparison": {"identity_sha256": identity, "specification": spec},
        "input_identity": {"frozen_spec_sha256": identity, **code_identity, "active_runner_sha256": code_identity["runner_sha256"]},
        "source_artifacts": artifacts,
        "code_revision_history": revisions,
        "single_anchor_reproduction": single_anchor,
        "completed_fits": completed_fits,
        "stages": stages,
    }
    _write_manifest(manifest)
    versioned = {
        "status": "frozen_before_fit",
        "frozen_at_utc": manifest["frozen_at_utc"],
        "frozen_comparison": manifest["frozen_comparison"],
        "input_identity": manifest["input_identity"],
        "source_artifacts": artifacts,
        "fit_count": 10,
        "completed_fit_count": len(completed_fits),
        "code_revision_history": revisions,
        "completion": None,
    }
    _write_json(VERSIONED_MANIFEST_PATH, versioned)
    return manifest


def _fold_paths(fold_id: int) -> tuple[Path, Path, Path]:
    folder = OUTPUT_DIR / "models" / "full_features"
    return (
        folder / f"fold_{fold_id:02d}.txt",
        folder / f"fold_{fold_id:02d}_oof.parquet",
        folder / f"fold_{fold_id:02d}_completion.json",
    )


def _completed_fold_valid(
    manifest: dict,
    fold_id: int,
    frame: pd.DataFrame,
    valid_idx: np.ndarray,
    feature_names: list[str],
) -> bool:
    record = manifest.get("completed_fits", {}).get(str(fold_id), {})
    model_path, oof_path, completion_path = _fold_paths(fold_id)
    identity = manifest["input_identity"]["frozen_spec_sha256"]
    if record.get("status") != "completed" or record.get("identity_sha256") != identity:
        return False
    if not _verify_records([record.get("model", {}), record.get("oof", {}), record.get("completion_manifest", {})]):
        return False
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("identity_sha256") != identity or completion.get("fold_id") != fold_id:
        return False
    if completion.get("model") != record.get("model") or completion.get("oof") != record.get("oof"):
        return False
    oof = pd.read_parquet(oof_path)
    expected_ids = frame["_source_row_index"].iloc[valid_idx].to_numpy(dtype=np.int32)
    actual_ids = oof["_source_row_index"].to_numpy(dtype=np.int32)
    if len(oof) != len(valid_idx) or not np.array_equal(actual_ids, expected_ids):
        return False
    if completion.get("validation_source_row_index_sha256") != taker._array_sha256(actual_ids, dtype=np.int32):
        return False
    if not oof["fold_id"].astype(str).eq(str(fold_id)).all() or not np.isfinite(oof["p_model_raw"]).all():
        return False
    booster = taker.lgb.Booster(model_file=str(model_path))
    valid = (
        booster.current_iteration() == int(CONFIG["boosting_iterations"])
        and list(booster.feature_name()) == feature_names
        and completion.get("effective_params") == manifest["frozen_comparison"]["specification"]["challenger"]["effective_lightgbm_params"]
    )
    del booster, oof
    return bool(valid)


def _train_full_features(
    manifest: dict,
    frame: pd.DataFrame,
    x: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    control_params: dict,
) -> dict:
    started = time.perf_counter()
    feature_names = list(manifest["frozen_comparison"]["specification"]["data"]["feature_order"])
    challenger_params = _challenger_params(control_params)
    expected_single_ids = pd.read_parquet(seed_stability.BASE_OOF_PATH, columns=["_source_row_index"])["_source_row_index"].to_numpy(dtype=np.int32)
    _begin_stage(manifest, "training", "Fit any remaining FULL_FEATURES folds; exactly ten folds x 31 rounds.")
    pending = [
        fold_id for fold_id, (_, valid_idx) in enumerate(folds)
        if not _completed_fold_valid(manifest, fold_id, frame, valid_idx, feature_names)
    ]
    dataset = None
    dataset_seconds = 0.0
    if pending:
        dataset_started = time.perf_counter()
        # lgb.cv applies params to the unparameterized Dataset before construct().
        dataset = taker.lgb.Dataset(
            x,
            label=target,
            weight=weights,
            feature_name=feature_names,
            free_raw_data=False,
        )
        dataset._update_params(challenger_params).construct()
        dataset_seconds = time.perf_counter() - dataset_started
    fold_seconds = []
    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        model_path, oof_path, completion_path = _fold_paths(fold_id)
        if fold_id not in pending:
            fold_seconds.append(float(manifest["completed_fits"][str(fold_id)].get("elapsed_seconds", 0.0)))
            print(f"Reused verified FULL_FEATURES fold {fold_id + 1}/10.")
            continue
        fold_started = time.perf_counter()
        train_set = dataset.subset(np.sort(train_idx))
        booster = taker.lgb.train(
            params=challenger_params,
            train_set=train_set,
            num_boost_round=int(CONFIG["boosting_iterations"]),
            keep_training_booster=False,
        )
        if booster.current_iteration() != int(CONFIG["boosting_iterations"]):
            raise RuntimeError(f"FULL_FEATURES fold {fold_id} did not fit exactly 31 iterations")
        if list(booster.feature_name()) != feature_names:
            raise RuntimeError(f"FULL_FEATURES fold {fold_id} feature order differs from the frozen BASE70 order")
        prediction = np.asarray(
            booster.predict(x[valid_idx], num_iteration=int(CONFIG["boosting_iterations"])),
            dtype=np.float32,
        )
        if not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 1.0)):
            raise RuntimeError(f"FULL_FEATURES fold {fold_id} produced invalid OOF probabilities")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_model = model_path.with_suffix(".txt.tmp")
        booster.save_model(str(temporary_model), num_iteration=int(CONFIG["boosting_iterations"]))
        temporary_model.replace(model_path)
        fold_oof = seed_stability._write_fold_oof(frame, valid_idx, fold_id, prediction)
        _write_parquet(fold_oof, oof_path)
        elapsed = time.perf_counter() - fold_started
        completion = {
            "identity_sha256": manifest["input_identity"]["frozen_spec_sha256"],
            "variant": FULL_FEATURES,
            "fold_id": fold_id,
            "iteration_count": int(booster.current_iteration()),
            "effective_params": challenger_params,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(valid_idx)),
            "training_label_available_at_utc": frame["target_label_available_at"].iloc[train_idx].max().isoformat(),
            "validation_first_decision_at_utc": frame["nominal_decision_at"].iloc[valid_idx].min().isoformat(),
            "validation_last_decision_at_utc": frame["nominal_decision_at"].iloc[valid_idx].max().isoformat(),
            "validation_source_row_index_sha256": taker._array_sha256(
                fold_oof["_source_row_index"].to_numpy(dtype=np.int32), dtype=np.int32,
            ),
            "elapsed_seconds": elapsed,
            "model": _record(model_path),
            "oof": _record(oof_path),
        }
        _write_json(completion_path, completion)
        fit_record = {
            "status": "completed",
            "identity_sha256": completion["identity_sha256"],
            "model": completion["model"],
            "oof": completion["oof"],
            "completion_manifest": _record(completion_path),
            "iteration_count": completion["iteration_count"],
            "elapsed_seconds": elapsed,
        }
        manifest.setdefault("completed_fits", {})[str(fold_id)] = fit_record
        fold_seconds.append(elapsed)
        print(f"Completed FULL_FEATURES fold {fold_id + 1}/10 in {elapsed:.1f}s.")
        _write_manifest(manifest)
        del train_set, booster, prediction, fold_oof
    if dataset is not None:
        del dataset

    fold_outputs = [pd.read_parquet(_fold_paths(fold_id)[1]) for fold_id in range(10)]
    combined = pd.concat(fold_outputs, ignore_index=True)
    combined.sort_values("_source_row_index", kind="stable", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    actual_ids = combined["_source_row_index"].to_numpy(dtype=np.int32)
    if len(combined) != len(expected_single_ids) or not np.array_equal(actual_ids, expected_single_ids):
        raise RuntimeError("FULL_FEATURES combined OOF does not match the saved SINGLE validation rows")
    if combined["_source_row_index"].duplicated().any() or not np.isfinite(combined["p_model_raw"]).all():
        raise RuntimeError("FULL_FEATURES combined OOF has duplicate rows or invalid probabilities")
    combined_path = OUTPUT_DIR / "oof_full_features.parquet"
    _write_parquet(combined, combined_path)
    training_summary = {
        "variant": FULL_FEATURES,
        "fold_count": 10,
        "newly_fitted_fold_count": len(pending),
        "reused_checkpoint_count": 10 - len(pending),
        "iterations_per_fit": int(CONFIG["boosting_iterations"]),
        "all_fits_completed": all(
            manifest.get("completed_fits", {}).get(str(fold_id), {}).get("iteration_count") == int(CONFIG["boosting_iterations"])
            for fold_id in range(10)
        ),
        "fit_seconds": float(sum(fold_seconds)),
        "dataset_construction_seconds": float(dataset_seconds),
        "fold_seconds": fold_seconds,
        "oof_rows": int(len(combined)),
        "elapsed_seconds": time.perf_counter() - started,
        "oof": _record(combined_path),
    }
    if not training_summary["all_fits_completed"]:
        raise RuntimeError("The FULL_FEATURES training stage is incomplete")
    outputs = [training_summary["oof"]]
    for fold_id in range(10):
        record = manifest["completed_fits"][str(fold_id)]
        outputs.extend([record["model"], record["oof"], record["completion_manifest"]])
    summary_path = OUTPUT_DIR / "training_summary.json"
    _write_json(summary_path, training_summary)
    outputs.append(_record(summary_path))
    _finish_stage(
        manifest,
        "training",
        summary=training_summary,
        outputs=outputs,
        elapsed_seconds=training_summary["elapsed_seconds"],
        completed_fits=10,
    )
    return training_summary


def _build_predictions(manifest: dict) -> pd.DataFrame:
    output_path = OUTPUT_DIR / "predictions.parquet"
    stage = manifest.get("stages", {}).get("common_predictions", {})
    if stage.get("status") == "completed" and _verify_records(stage.get("outputs", [])):
        return pd.read_parquet(output_path)
    started = time.perf_counter()
    _begin_stage(manifest, "common_predictions", "Join the new OOF with saved SINGLE and ENSEMBLE OOF predictions.")
    single = pd.read_parquet(seed_stability.BASE_OOF_PATH)
    saved = pd.read_parquet(seed_stability.OUTPUT_DIR / "predictions.parquet")
    full = pd.read_parquet(OUTPUT_DIR / "oof_full_features.parquet")
    for frame in (single, saved, full):
        frame["fold_id"] = frame["fold_id"].astype(str)
    if len(single) != 318560 or len(saved) != 318560 or len(full) != 318560:
        raise RuntimeError("The SINGLE, ENSEMBLE, and FULL_FEATURES OOF row counts must all be 318,560")
    if not single[ROW_ID_COLUMNS].equals(saved[ROW_ID_COLUMNS]) or not single[ROW_ID_COLUMNS].equals(full[ROW_ID_COLUMNS]):
        raise RuntimeError("The three OOF variants do not have exactly identical row and fold identities")
    component_columns = [seed_stability.PROBABILITY_COLUMNS[name] for name in seed_stability.COMPONENT_NAMES]
    component_values = saved[component_columns].to_numpy(dtype=np.float64)
    expected_ensemble = seed_stability._ensemble_probability(component_values)
    stored_ensemble = saved[seed_stability.PROBABILITY_COLUMNS[ENSEMBLE]].to_numpy(dtype=np.float64)
    if not np.array_equal(stored_ensemble, expected_ensemble):
        raise RuntimeError("Saved ENSEMBLE OOF is not the exact arithmetic mean of the five historical components")
    single_probability = single["p_model_raw"].to_numpy(dtype=np.float32)
    if not np.array_equal(single_probability, saved[seed_stability.PROBABILITY_COLUMNS[SINGLE]].to_numpy(dtype=np.float32)):
        raise RuntimeError("Saved seed-stability SINGLE OOF differs from the original SINGLE OOF")
    result = single[ROW_ID_COLUMNS].copy()
    result[PROBABILITY_COLUMNS[SINGLE]] = single_probability
    result[PROBABILITY_COLUMNS[FULL_FEATURES]] = full["p_model_raw"].to_numpy(dtype=np.float32)
    result[PROBABILITY_COLUMNS[ENSEMBLE]] = stored_ensemble
    if not np.isfinite(result[list(PROBABILITY_COLUMNS.values())]).all().all():
        raise RuntimeError("Joined common OOF probabilities contain non-finite values")
    _write_parquet(result, output_path)
    summary = {
        "oof_rows": int(len(result)),
        "variants": list(VARIANTS),
        "single_oof_reused": True,
        "ensemble_oof_reused": True,
        "ensemble_equal_weight_mean_verified": True,
        "identical_row_and_fold_indices_verified": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _finish_stage(manifest, "common_predictions", summary=summary, outputs=[_record(output_path)], elapsed_seconds=summary["elapsed_seconds"])
    return result


def _score(labels: np.ndarray, probability: np.ndarray) -> dict:
    return taker._score_binary(np.asarray(labels, dtype=np.int8), np.asarray(probability, dtype=np.float64))


def _paired_quality_bootstrap(rows: pd.DataFrame, label_column: str, seed: int) -> dict:
    labels = rows[label_column].to_numpy(dtype=np.int8)
    days = pd.date_range(
        rows["market_start_utc"].min().floor("D"),
        rows["market_start_utc"].max().floor("D"),
        freq="D",
        tz="UTC",
    )
    draws = seed_stability._block_indices(
        days,
        int(CONFIG["bootstrap"]["replications"]),
        int(CONFIG["bootstrap"]["block_days"]),
        seed,
    )
    row_days = seed_stability._row_day_index(rows["market_start_utc"], days)
    day_counts = np.stack([np.bincount(draw, minlength=len(days)) for draw in draws], axis=0)
    row_counts = np.bincount(row_days, minlength=len(days)).astype(np.float64)
    probabilities = {
        variant: rows[PROBABILITY_COLUMNS[variant]].to_numpy(dtype=np.float64)
        for variant in (SINGLE, FULL_FEATURES)
    }
    point = {
        variant: _score(labels, probabilities[variant])
        for variant in (SINGLE, FULL_FEATURES)
    }
    clips = {variant: np.clip(p, 1e-6, 1.0 - 1e-6) for variant, p in probabilities.items()}
    losses = {
        variant: -(labels * np.log(clips[variant]) + (1 - labels) * np.log1p(-clips[variant]))
        for variant in (SINGLE, FULL_FEATURES)
    }
    brier = {variant: np.square(probabilities[variant] - labels) for variant in (SINGLE, FULL_FEATURES)}
    row_loss = {variant: np.bincount(row_days, weights=losses[variant], minlength=len(days)) for variant in losses}
    row_brier = {variant: np.bincount(row_days, weights=brier[variant], minlength=len(days)) for variant in brier}
    denominator = day_counts @ row_counts
    bootstrap = {}
    auc_values = {
        variant: seed_stability._weighted_auc_bootstrap(
            labels,
            probabilities[variant],
            row_days,
            day_counts,
        )
        for variant in (SINGLE, FULL_FEATURES)
    }
    deltas = {
        "log_loss": (day_counts @ row_loss[FULL_FEATURES] - day_counts @ row_loss[SINGLE]) / denominator,
        "brier": (day_counts @ row_brier[FULL_FEATURES] - day_counts @ row_brier[SINGLE]) / denominator,
        "auc": auc_values[FULL_FEATURES] - auc_values[SINGLE],
    }
    for metric, values in deltas.items():
        valid = values[np.isfinite(values)]
        bootstrap[metric] = {
            "point_difference_FULL_FEATURES_minus_SINGLE": float(point[FULL_FEATURES][metric] - point[SINGLE][metric]),
            "95pct": [float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))] if len(valid) else [None, None],
            "valid_replications": int(len(valid)),
            "replications": int(len(draws)),
        }
    return {
        "method": CONFIG["bootstrap"]["method"],
        "sample_rows": int(len(rows)),
        "calendar_days": int(len(days)),
        "seed": int(seed),
        "block_days": int(CONFIG["bootstrap"]["block_days"]),
        "replications": int(len(draws)),
        "contrasts": bootstrap,
    }


def _probability_difference(rows: pd.DataFrame) -> dict:
    delta = (
        rows[PROBABILITY_COLUMNS[FULL_FEATURES]].to_numpy(dtype=np.float64)
        - rows[PROBABILITY_COLUMNS[SINGLE]].to_numpy(dtype=np.float64)
    )
    return {
        "rows": int(len(delta)),
        "mean": float(delta.mean()),
        "standard_deviation": float(delta.std(ddof=0)),
        "mean_absolute": float(np.abs(delta).mean()),
        "positive_fraction": float((delta > 0).mean()),
        "quantiles": {f"p{q}": float(np.quantile(delta, q / 100)) for q in (1, 5, 25, 50, 75, 95, 99)},
    }


def _quality(manifest: dict, predictions: pd.DataFrame, info_manifest: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    stage = manifest.get("stages", {}).get("quality", {})
    summary_path = OUTPUT_DIR / "quality_summary.json"
    proxy_folds_path = OUTPUT_DIR / "proxy_fold_quality.csv"
    official_monthly_path = OUTPUT_DIR / "official_monthly_quality.csv"
    official_rows_path = OUTPUT_DIR / "official_quality_rows.parquet"
    if stage.get("status") == "completed" and _verify_records(stage.get("outputs", [])):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return summary, pd.read_csv(proxy_folds_path), pd.read_csv(official_monthly_path)
    started = time.perf_counter()
    _begin_stage(manifest, "quality", "Score the common proxy and official-label samples, then run paired 3-day block bootstraps.")
    proxy = predictions.copy()
    proxy["fold_id"] = proxy["fold_id"].astype(str)
    proxy["market_start_utc"] = pd.to_datetime(proxy["market_start_utc"], utc=True, errors="raise")
    prior_variants = seed_stability.VARIANT_NAMES
    prior_probability_columns = seed_stability.PROBABILITY_COLUMNS
    try:
        seed_stability.VARIANT_NAMES = VARIANTS
        seed_stability.PROBABILITY_COLUMNS = PROBABILITY_COLUMNS
        proxy_overall, proxy_fold_rows = seed_stability._quality_table(proxy, TARGET_COLUMN, "fold_id")
    finally:
        seed_stability.VARIANT_NAMES = prior_variants
        seed_stability.PROBABILITY_COLUMNS = prior_probability_columns

    source_info = pd.read_parquet(seed_stability.BASE_PREDICTIONS_PATH)
    official_filter = (
        source_info["comparison_forecast_available"].astype(bool)
        & source_info["official_outcome_available_by_cutoff"].astype(bool)
        & pd.to_datetime(source_info["market_start_utc"], utc=True).lt(pd.Timestamp(CONFIG["economics"]["cutoff_exclusive_utc"]))
    )
    official_base = source_info.loc[official_filter, ["market_start_utc", "target_polymarket_up"]].copy()
    official_base["market_start_utc"] = pd.to_datetime(official_base["market_start_utc"], utc=True, errors="raise")
    official = official_base.merge(
        predictions[["market_start_utc", *PROBABILITY_COLUMNS.values()]],
        on="market_start_utc",
        how="left",
        validate="one_to_one",
    )
    if len(official) != 20718 or official.target_polymarket_up.nunique() != 2:
        raise RuntimeError("The official sample does not reproduce the same 20,718 labeled development outcomes")
    if official[list(PROBABILITY_COLUMNS.values())].isna().any().any():
        raise RuntimeError("A model is missing one or more common official-label probabilities")
    original_raw = source_info.loc[official_filter].sort_values("market_start_utc", kind="stable")["p_raw"].to_numpy(dtype=np.float64)
    reused_single = official.sort_values("market_start_utc", kind="stable")[PROBABILITY_COLUMNS[SINGLE]].to_numpy(dtype=np.float64)
    if not np.allclose(original_raw, reused_single, rtol=0, atol=1e-12):
        raise RuntimeError("Reused SINGLE probabilities differ from the saved official-quality sample")

    official_overall = {
        variant: _score(
            official["target_polymarket_up"].to_numpy(dtype=np.int8),
            official[PROBABILITY_COLUMNS[variant]].to_numpy(dtype=np.float64),
        )
        for variant in VARIANTS
    }
    official["utc_month"] = official.market_start_utc.dt.strftime("%Y-%m")
    monthly_rows = []
    for month, rows in official.groupby("utc_month", sort=True):
        labels = rows.target_polymarket_up.to_numpy(dtype=np.int8)
        for variant in VARIANTS:
            monthly_rows.append({
                "utc_month": month,
                "variant": variant,
                "rows": int(len(rows)),
                **_score(labels, rows[PROBABILITY_COLUMNS[variant]].to_numpy(dtype=np.float64)),
            })
    proxy_bootstrap = _paired_quality_bootstrap(proxy, TARGET_COLUMN, int(CONFIG["bootstrap"]["seed"]))
    official_bootstrap = _paired_quality_bootstrap(official, "target_polymarket_up", int(CONFIG["bootstrap"]["seed"]))
    proxy_delta = _probability_difference(proxy)
    official_delta = _probability_difference(official)
    proxy_quality = {
        "sample_rows": int(len(proxy)),
        "overall": proxy_overall,
        "fold_metrics": proxy_fold_rows,
        "paired_bootstrap": proxy_bootstrap,
        "probability_difference_FULL_FEATURES_minus_SINGLE": proxy_delta,
    }
    official_quality = {
        "sample_rows": int(len(official)),
        "up_labels": int(official.target_polymarket_up.sum()),
        "down_labels": int(len(official) - official.target_polymarket_up.sum()),
        "start_utc": official.market_start_utc.min().isoformat(),
        "end_utc": official.market_start_utc.max().isoformat(),
        "overall": official_overall,
        "monthly_metrics": monthly_rows,
        "paired_bootstrap": official_bootstrap,
        "probability_difference_FULL_FEATURES_minus_SINGLE": official_delta,
    }
    summary = {
        "proxy_oof": proxy_quality,
        "official": official_quality,
        "single_oof_reused": True,
        "ensemble_oof_reused": True,
        "calibration_applied": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    proxy_folds = pd.DataFrame(proxy_fold_rows)
    monthly = pd.DataFrame(monthly_rows)
    _write_json(summary_path, summary)
    proxy_folds.to_csv(proxy_folds_path, index=False)
    monthly.to_csv(official_monthly_path, index=False)
    _write_parquet(official, official_rows_path)
    records = [_record(path) for path in (summary_path, proxy_folds_path, official_monthly_path, official_rows_path)]
    _finish_stage(manifest, "quality", summary=summary, outputs=records, elapsed_seconds=summary["elapsed_seconds"])
    return summary, proxy_folds, monthly


def _load_economic_inputs(info_manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    started = time.perf_counter()
    calendar = pd.read_parquet(seed_stability.EXECUTION_CALENDAR_PATH)
    if len(calendar) != int(CONFIG["economics"]["calendar_rows"]):
        raise RuntimeError("The saved economic calendar does not contain exactly 20,735 decisions")
    snapshot = info_manifest["frozen_comparison"]["specification"]["sources"]["decision_and_execution_snapshots"]
    t60 = pd.read_parquet(ROOT / snapshot["t60_path"])
    t59 = pd.read_parquet(ROOT / snapshot["t59_path"])
    summary = {
        "calendar_rows": int(len(calendar)),
        "first_market_start_utc": pd.to_datetime(calendar.market_start_utc, utc=True).min().isoformat(),
        "last_market_start_utc": pd.to_datetime(calendar.market_start_utc, utc=True).max().isoformat(),
        "t60_snapshot_rows": int(len(t60)),
        "t59_snapshot_rows": int(len(t59)),
        "snapshot_hashes_verified": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return calendar, t59, t60, summary


def _configure_existing_economic_helpers() -> None:
    base70.OUTPUT_DIR = OUTPUT_DIR
    base70.MANIFEST_PATH = MANIFEST_PATH
    base70.CONFIG = {
        "policy": {
            "gross_stake_usd": float(CONFIG["economics"]["gross_stake_usd"]),
            "minimum_net_ev_fraction": float(CONFIG["economics"]["minimum_net_ev_fraction"]),
            "decision_offset_seconds": int(CONFIG["economics"]["decision_offset_seconds"]),
            "execution_offset_seconds": int(CONFIG["economics"]["execution_offset_seconds"]),
        },
        "bootstrap": copy.deepcopy(CONFIG["bootstrap"]),
    }
    base70.VARIANTS = VARIANTS
    base70.PROBABILITY_COLUMNS = PROBABILITY_COLUMNS
    base70.CONTRASTS = CONTRASTS
    base70.START_UTC = pd.Timestamp(CONFIG["economics"]["start_utc"])
    base70.CUTOFF_UTC = pd.Timestamp(CONFIG["economics"]["cutoff_exclusive_utc"])
    base70._status_update = lambda _manifest: None


def _economics(
    manifest: dict,
    predictions: pd.DataFrame,
    calendar: pd.DataFrame,
    t59: pd.DataFrame,
    t60: pd.DataFrame,
) -> dict:
    path = OUTPUT_DIR / "economics_summary.json"
    stage = manifest.get("stages", {}).get("economics_summary", {})
    if stage.get("status") == "completed" and _verify_records(stage.get("outputs", [])):
        summary = json.loads(path.read_text(encoding="utf-8"))
        manifest["single_anchor_reproduction"] = summary["single_anchor_reproduction"]
        return summary
    started = time.perf_counter()
    _configure_existing_economic_helpers()
    summaries = {}
    trades = {}
    decisions = {}
    for variant in VARIANTS:
        summaries[variant], decisions[variant], trades[variant] = (None, None, None)
        summary, decision_frame, trade_frame = base70._run_variant(
            variant,
            predictions,
            calendar,
            t59,
            t60,
            manifest,
        )
        if not base70.summary_accounting_passed(summary):
            raise RuntimeError(f"{variant} failed the existing portfolio accounting audit")
        summaries[variant] = summary
        trades[variant] = trade_frame
        decisions[variant] = decision_frame
        if variant == SINGLE:
            manifest["single_anchor_reproduction"] = seed_stability._verify_single_economics(trade_frame, summary)
        _write_manifest(manifest)

    days = pd.date_range(
        pd.to_datetime(calendar.market_start_utc, utc=True).min().floor("D"),
        pd.to_datetime(calendar.market_start_utc, utc=True).max().floor("D"),
        freq="D",
        tz="UTC",
    )
    daily = base70._build_daily_pnl(trades, days)
    draws = seed_stability._block_indices(
        days,
        int(CONFIG["bootstrap"]["replications"]),
        int(CONFIG["bootstrap"]["block_days"]),
        int(CONFIG["bootstrap"]["seed"]),
    )
    bootstrap = base70._economic_bootstrap(daily, draws)
    full_turnover = float(daily[f"{FULL_FEATURES}_settled_turnover_usd"].sum())
    single_turnover = float(daily[f"{SINGLE}_settled_turnover_usd"].sum())
    full_pnl = float(daily[f"{FULL_FEATURES}_net_pnl_usd"].sum())
    single_pnl = float(daily[f"{SINGLE}_net_pnl_usd"].sum())
    bootstrap["contrasts"]["FULL_FEATURES_MINUS_SINGLE"]["pnl_per_settled_turnover_difference"]["point_estimate"] = (
        full_pnl / full_turnover - single_pnl / single_turnover
        if full_turnover > 0 and single_turnover > 0 else None
    )
    decomposition = base70._decompose_pnl(
        trades[FULL_FEATURES], trades[SINGLE], FULL_FEATURES, SINGLE,
    )
    if not decomposition["sum_reconciles"]:
        raise RuntimeError("FULL_FEATURES minus SINGLE PnL decomposition does not reconcile")
    summary = {
        "variants": summaries,
        "availability": {
            "decision_rows": int(len(calendar)),
            "calendar_days": int(len(days)),
            "first_market_start_utc": pd.to_datetime(calendar.market_start_utc, utc=True).min().isoformat(),
            "last_market_start_utc": pd.to_datetime(calendar.market_start_utc, utc=True).max().isoformat(),
        },
        "bootstrap": bootstrap,
        "concentration": base70._concentration(daily),
        "pnl_decomposition_FULL_FEATURES_minus_SINGLE": decomposition,
        "up_down_economics": seed_stability._side_economics(trades),
        "single_anchor_reproduction": manifest["single_anchor_reproduction"],
        "accounting_audits_passed": True,
        "daily_calendar_rows_including_no_fill_days": int(len(daily)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    daily_path = OUTPUT_DIR / "daily_economics.csv"
    decomposition_path = OUTPUT_DIR / "pnl_decomposition.json"
    _write_json(path, summary)
    daily.to_csv(daily_path, index=False)
    _write_json(decomposition_path, decomposition)
    variant_outputs = [
        record
        for variant in VARIANTS
        for record in manifest.get("stages", {}).get(f"economics_{variant}", {}).get("outputs", [])
    ]
    outputs = [*variant_outputs, *[_record(item) for item in (path, daily_path, decomposition_path)]]
    _finish_stage(
        manifest,
        "economics_summary",
        summary=summary,
        outputs=outputs,
        elapsed_seconds=summary["elapsed_seconds"],
    )
    return summary


def _fmt(value, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):,.{digits}f}"


def _fmt_pct(value, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}%}"


def _write_report(manifest: dict) -> None:
    quality = manifest["stages"]["quality"]["summary"]
    economics = manifest["stages"]["economics_summary"]["summary"]
    spec = manifest["frozen_comparison"]["specification"]
    source_artifacts = manifest["source_artifacts"]
    params = spec["single"]["effective_lightgbm_params"]
    proxy = quality["proxy_oof"]
    official = quality["official"]
    proxy_delta = proxy["probability_difference_FULL_FEATURES_minus_SINGLE"]
    official_delta = official["probability_difference_FULL_FEATURES_minus_SINGLE"]
    econ_variants = economics["variants"]
    primary_ci = economics["bootstrap"]["contrasts"]["FULL_FEATURES_MINUS_SINGLE"]
    training_summary = manifest["stages"]["training"]["summary"]
    fold_fit_seconds = [
        float(manifest["completed_fits"][str(fold_id)]["elapsed_seconds"])
        for fold_id in range(10)
    ]
    completed_stage_seconds = {
        "quality_and_bootstrap": float(manifest["stages"]["quality"]["elapsed_seconds"]),
        "economics_and_audits_all_variants": float(
            manifest["stages"]["economics_summary"]["elapsed_seconds"]
        ),
    }
    ll = proxy["paired_bootstrap"]["contrasts"]["log_loss"]
    proxy_folds = pd.DataFrame(proxy["fold_metrics"])
    fold_lines = ["| Fold | Wiersze | Δ log-loss | Δ Brier | Δ AUC |", "|---:|---:|---:|---:|---:|"]
    fold_wins = {"log_loss": 0, "brier": 0, "auc": 0}
    for fold_id, part in proxy_folds.groupby("group", sort=True):
        rows = int(part.iloc[0]["rows"])
        by_variant = {row.variant: row for row in part.itertuples(index=False)}
        deltas = {
            "log_loss": float(by_variant[FULL_FEATURES].log_loss - by_variant[SINGLE].log_loss),
            "brier": float(by_variant[FULL_FEATURES].brier - by_variant[SINGLE].brier),
            "auc": float(by_variant[FULL_FEATURES].auc - by_variant[SINGLE].auc),
        }
        fold_wins["log_loss"] += deltas["log_loss"] < 0
        fold_wins["brier"] += deltas["brier"] < 0
        fold_wins["auc"] += deltas["auc"] > 0
        fold_lines.append(
            f"| {fold_id} | {rows:,} | {_fmt(deltas['log_loss'])} | {_fmt(deltas['brier'])} | {_fmt(deltas['auc'])} |"
        )

    monthly_lines = ["| Miesiąc UTC | Wariant | N | Log-loss | Brier | AUC |", "|---|---|---:|---:|---:|---:|"]
    for row in official["monthly_metrics"]:
        monthly_lines.append(
            f"| {row['utc_month']} | {row['variant']} | {row['rows']:,} | {_fmt(row['log_loss'])} | {_fmt(row['brier'])} | {_fmt(row['auc'], 5)} |"
        )

    overall_lines = ["| Wariant | Fills | Obrót | Opłaty | PnL netto | PnL/obrót | Prognozowane EV | PnL − EV |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        row = econ_variants[variant]
        overall_lines.append(
            f"| {variant} | {row['fill_count']:,} | ${row['gross_turnover_usd']:,.2f} | ${row['fees_paid_estimated_usd']:,.2f} | ${row['net_pnl_usd']:,.2f} | {_fmt_pct(row['pnl_per_settled_turnover'])} | ${row['forecast_net_ev_on_settled_fills_usd']:,.2f} | ${row['pnl_minus_forecast_ev_on_settled_fills_usd']:,.2f} |"
        )
    monthly_econ_lines = ["| Miesiąc UTC | Wariant | Wypełnienia | UP / DOWN | Obrót | Opłaty | PnL netto | PnL/obrót | EV | PnL − EV |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        for row in econ_variants[variant]["monthly_results"]:
            monthly_econ_lines.append(
                f"| {row['utc_month']} | {variant} | {row['fills']:,} | {row['up_fills']} / {row['down_fills']} | ${row['gross_turnover_usd']:,.2f} | ${row['fees_usd']:,.2f} | ${row['net_pnl_usd']:,.2f} | {_fmt_pct(row['pnl_per_settled_turnover'])} | ${row['forecast_ev_on_settled_fills_usd']:,.2f} | ${row['pnl_minus_forecast_ev_on_settled_fills_usd']:,.2f} |"
            )
    side_lines = ["| Wariant | Strona | Wypełnienia | Obrót | Opłaty | PnL | EV | PnL − EV |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in economics["up_down_economics"]:
        side_lines.append(
            f"| {row['variant']} | {row['side']} | {row['fills']:,} | ${row['turnover_usd']:,.2f} | ${row['fees_usd']:,.2f} | ${row['net_pnl_usd']:,.2f} | ${row['forecast_ev_usd']:,.2f} | ${row['pnl_minus_ev_usd']:,.2f} |"
        )
    concentration_lines = ["| Wariant | Pięć najlepszych dni | Udział w PnL | PnL pozostałych dni |", "|---|---:|---:|---:|"]
    for variant in VARIANTS:
        row = economics["concentration"][variant]
        best_days = ", ".join(f"{pd.Timestamp(item['utc_day']).date()}: ${item['net_pnl_usd']:.2f}" for item in row["top_five_days"])
        concentration_lines.append(
            f"| {variant} | ${row['top_five_pnl_usd']:,.2f} ({best_days}) | {_fmt_pct(row['top_five_share_of_total_pnl'])} | ${row['remaining_days_pnl_usd']:,.2f} |"
        )
    decomp = economics["pnl_decomposition_FULL_FEATURES_minus_SINGLE"]
    decomp_lines = [
        "| Składnik Δ PnL (FULL_FEATURES − SINGLE) | USD | Rynki |",
        "|---|---:|---:|",
    ]
    labels = {
        "settled_fills_same_side": "Wypełnienia wspólne, ta sama strona",
        "settled_fills_only_one_variant": "Wypełnienia tylko jednego wariantu",
        "settled_fills_opposite_sides": "Wypełnienia wspólne, przeciwne strony",
    }
    for key, label in labels.items():
        decomp_lines.append(f"| {label} | ${decomp['components_usd'][key]:,.2f} | {decomp['market_counts'][key]:,} |")

    param_lines = ["| Parametr | SINGLE | FULL_FEATURES |", "|---|---:|---:|"]
    for name in sorted(params):
        left = params[name]
        right = spec["challenger"]["effective_lightgbm_params"][name]
        param_lines.append(f"| `{name}` | `{left}` | `{right}` |")
    full_minus_single = {
        metric: proxy["overall"][FULL_FEATURES][metric] - proxy["overall"][SINGLE][metric]
        for metric in QUALITY_METRICS
    }
    official_metric_delta = {
        metric: official["overall"][FULL_FEATURES][metric] - official["overall"][SINGLE][metric]
        for metric in QUALITY_METRICS
    }
    quality_signal = (
        f"Na proxy OOF FULL_FEATURES zmienił log-loss o {_fmt(full_minus_single['log_loss'])} "
        f"(sparowany 95% CI {_fmt(ll['95pct'][0])} do {_fmt(ll['95pct'][1])}); poprawił log-loss w {fold_wins['log_loss']}/10 foldów, "
        f"Brier w {fold_wins['brier']}/10, a AUC w {fold_wins['auc']}/10. "
        f"Na oficjalnych etykietach różnice wyniosły Δ log-loss {_fmt(official_metric_delta['log_loss'])}, "
        f"Δ Brier {_fmt(official_metric_delta['brier'])}, Δ AUC {_fmt(official_metric_delta['auc'])}."
    )
    econ_full = econ_variants[FULL_FEATURES]
    econ_single = econ_variants[SINGLE]
    econ_signal = (
        f"FULL_FEATURES wykonał {econ_full['fill_count']:,} zleceń i osiągnął ${econ_full['net_pnl_usd']:,.2f} PnL "
        f"wobec {econ_single['fill_count']:,} i ${econ_single['net_pnl_usd']:,.2f} dla SINGLE. "
        f"95% CI dla różnicy średniego dziennego PnL to "
        f"${primary_ci['mean_daily_pnl_difference_usd']['95pct'][0]:,.2f} do ${primary_ci['mean_daily_pnl_difference_usd']['95pct'][1]:,.2f}; "
        f"dla PnL/obrót: {_fmt_pct(primary_ci['pnl_per_settled_turnover_difference']['95pct'][0])} do "
        f"{_fmt_pct(primary_ci['pnl_per_settled_turnover_difference']['95pct'][1])}."
    )
    proxy_ci = proxy["paired_bootstrap"]["contrasts"]
    official_ci = official["paired_bootstrap"]["contrasts"]
    quality_clear = (
        proxy_ci["log_loss"]["95pct"][1] < 0
        and proxy_ci["brier"]["95pct"][1] < 0
        and proxy_ci["auc"]["95pct"][0] > 0
        and official_ci["log_loss"]["95pct"][1] < 0
        and official_ci["brier"]["95pct"][1] < 0
        and official_ci["auc"]["95pct"][0] > 0
    )
    quality_regression = (
        proxy_ci["log_loss"]["95pct"][0] > 0
        and proxy_ci["brier"]["95pct"][0] > 0
        and proxy_ci["auc"]["95pct"][1] < 0
    )
    economic_clear = (
        primary_ci["mean_daily_pnl_difference_usd"]["95pct"][0] > 0
        and primary_ci["pnl_per_settled_turnover_difference"]["95pct"][0] > 0
    )
    economic_regression = (
        primary_ci["mean_daily_pnl_difference_usd"]["95pct"][1] < 0
        and primary_ci["pnl_per_settled_turnover_difference"]["95pct"][1] < 0
    )
    if quality_clear and economic_clear:
        followup = "Tak; zarówno jakość, jak i oba ekonomiczne przedziały wspierają dalszą, osobno zamrożoną weryfikację FULL_FEATURES. Ten wynik nie wybiera modelu wdrożeniowego ani nie uzasadnia refitu."
    elif quality_regression and economic_regression:
        followup = "Nie; jakość proxy i oba ekonomiczne przedziały przemawiają przeciw tej zmianie. Nie uruchamiaj rozszerzenia tego eksperymentu na podstawie punktowego PnL."
    elif quality_clear or economic_clear:
        followup = "Warunkowo; jedna część wyniku wspiera FULL_FEATURES, ale druga nie potwierdza poprawy. Można rozważyć tylko nowy, ograniczony test z osobnym zamrożeniem; ten runner nie rozszerza eksperymentu."
    else:
        followup = "Nie ma spójnych przedziałowo dowodów poprawy jakości i ekonomii. Wynik nie uzasadnia priorytetu FULL_FEATURES ani refitu; ewentualny dalszy test wymaga osobnej hipotezy i zamrożonego protokołu."
    report = f"""# BASE70 FULL_FEATURES: feature_fraction=1.0 — 2026-10-18

## Cel i zakres

Hipoteza zamrożona przed fitami: przy słabym sygnale losowe pomijanie części z 70 wybranych cech może obniżać stabilność prognoz istotnych dla kwalifikacji transakcji. To hipoteza do sprawdzenia, nie ustalona przyczyna wcześniejszych wyników.

Główny kontrast to FULL_FEATURES − SINGLE. ENSEMBLE jest dodatkowym kontekstem i pochodzi z istniejących predykcji pięciu komponentów o równych wagach; żaden seed nie został wybrany po wyniku. Nie trenowano ponownie SINGLE ani ENSEMBLE.

## Zamrożone porównanie i źródła

- Macierz: {spec['data']['rows']:,} wierszy i 70 cech w niezmienionej kolejności; target `{TARGET_COLUMN}`; te same 10 foldów, purging i sample weights. Hash wag: `{spec['data']['training_weight_sha256']}`.
- Dataset i indeksy treningowe/walidacyjne zostały zweryfikowane względem istniejących źródeł. Macierz wczytano raz; cechy, profile VP/RP, wskaźniki ani selekcja nie były generowane ponownie.
- Parametry budowania Dataset: kontrola używała `lgb.Dataset(X, label, weight, feature_name, free_raw_data=False)` przekazanego do `lgb.cv`; FULL_FEATURES zastosował challenger params przez `_update_params()` przed `construct()` i `Dataset.subset(sorted(train_idx))` na zapisanych foldach. Seed próbkowania histogramów pozostał `data_random_seed=20261015`.
- Oryginalny OOF SINGLE, jego 10 boosterów oraz zapisany OOF ENSEMBLE zweryfikowano i wykorzystano bez refitu. Hash bundle OOF SINGLE/ENSEMBLE: `{source_artifacts['saved_single_and_ensemble_oof_bundle']['sha256']}`.
- Macierz BASE70 SHA-256: `{source_artifacts['matrix']['sha256']}`; foldy SHA-256: `{source_artifacts['folds']['sha256']}`; OOF SINGLE SHA-256: `{source_artifacts['single_oof']['sha256']}`.
- Kalendarz ekonomiczny 20 735 decyzji, snapshoty T−60/T−59, etykiety oficjalne i ledger źródłowy zostały zweryfikowane względem hashy istniejących manifestów.

### Efektywne parametry

| Parametr | SINGLE | FULL_FEATURES |
|---|---:|---:|
{chr(10).join(param_lines[2:])}

- Oryginalne seedy zostały odczytane z zapisanego CV: `seed={params['seed']}`, `feature_fraction_seed={params['feature_fraction_seed']}`, `data_random_seed={params['data_random_seed']}`, `bagging_seed={params['bagging_seed']}`. Dataset histogramów użył tego samego `data_random_seed` i tego samego wzorca konstrukcji co kontrolne CV.
- `feature_fraction_bynode` nie jest jawnie ustawione w parametrach kontroli; efektywny default LightGBM 4.6.0 wynosi `1.0`, zatem nie ma dodatkowego losowania cech per node. Nie zmieniono go w challengerze. [Dokumentacja parametrów LightGBM 4.6.0](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#feature_fraction_bynode) opisuje losowanie na poziomie węzła tylko dla wartości poniżej 1.
- Bagging pozostaje wyłączony (`bagging_fraction=1.0`, `bagging_freq=0`). Nie zastosowano early stoppingu, Optuny ani kalibracji. Oba warianty mają dokładnie 31 iteracji.
- `feature_fraction=1.0` nie gwarantuje pełnej deterministyczności GPU ani nie usuwa innych źródeł wrażliwości. LightGBM opisuje GPU jako używające domyślnie sumowania FP32; opcja `deterministic` dotyczy CPU. [Dokumentacja LightGBM 4.6.0](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#core-parameters).

## Jakość probabilistyczna

### Wspólne proxy OOF

| Wariant | Wiersze | Log-loss | Brier | AUC |
|---|---:|---:|---:|---:|
{chr(10).join(f"| {v} | {proxy['sample_rows']:,} | {_fmt(proxy['overall'][v]['log_loss'])} | {_fmt(proxy['overall'][v]['brier'])} | {_fmt(proxy['overall'][v]['auc'], 5)} |" for v in VARIANTS)}

Δ log-loss/Brier poniżej zera oznacza poprawę FULL_FEATURES; Δ AUC powyżej zera oznacza poprawę.

| Fold | Wiersze | Δ log-loss | Δ Brier | Δ AUC |
|---:|---:|---:|---:|---:|
{chr(10).join(fold_lines[2:])}

Bootstrap sparowany: 2 000 replikacji, kołowe bloki UTC po 3 dni, seed `{CONFIG['bootstrap']['seed']}`.

| Miara | Punkt FULL_FEATURES − SINGLE | 95% przedział | Replikacje |
|---|---:|---:|---:|
{chr(10).join(f"| {metric} | {_fmt(proxy['paired_bootstrap']['contrasts'][metric]['point_difference_FULL_FEATURES_minus_SINGLE'])} | {_fmt(proxy['paired_bootstrap']['contrasts'][metric]['95pct'][0])} do {_fmt(proxy['paired_bootstrap']['contrasts'][metric]['95pct'][1])} | {proxy['paired_bootstrap']['contrasts'][metric]['valid_replications']:,} |" for metric in QUALITY_METRICS)}

### Oficjalne etykiety

Użyto tych samych 20 718 etykiet rozwojowych; bez holdoutu i bez kalibracji.

| Wariant | Wiersze | Log-loss | Brier | AUC |
|---|---:|---:|---:|---:|
{chr(10).join(f"| {v} | {official['sample_rows']:,} | {_fmt(official['overall'][v]['log_loss'])} | {_fmt(official['overall'][v]['brier'])} | {_fmt(official['overall'][v]['auc'], 5)} |" for v in VARIANTS)}

| Miesiąc UTC | Wariant | Wiersze | Log-loss | Brier | AUC |
|---|---|---:|---:|---:|---:|
{chr(10).join(monthly_lines[2:])}

Sparowany bootstrap oficjalnych etykiet używa tych samych 2 000 replikacji i bloków 3-dniowych.

| Miara | Punkt FULL_FEATURES − SINGLE | 95% przedział | Replikacje |
|---|---:|---:|---:|
{chr(10).join(f"| {metric} | {_fmt(official['paired_bootstrap']['contrasts'][metric]['point_difference_FULL_FEATURES_minus_SINGLE'])} | {_fmt(official['paired_bootstrap']['contrasts'][metric]['95pct'][0])} do {_fmt(official['paired_bootstrap']['contrasts'][metric]['95pct'][1])} | {official['paired_bootstrap']['contrasts'][metric]['valid_replications']:,} |" for metric in QUALITY_METRICS)}

Rozkład różnicy prawdopodobieństw `p_FULL_FEATURES − p_SINGLE`:

| Próbka | N | Średnia | SD | Średnia bezwzględna | Udział dodatnich | p01 | p05 | Mediana | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Proxy OOF | {proxy_delta['rows']:,} | {_fmt(proxy_delta['mean'])} | {_fmt(proxy_delta['standard_deviation'])} | {_fmt(proxy_delta['mean_absolute'])} | {_fmt_pct(proxy_delta['positive_fraction'])} | {_fmt(proxy_delta['quantiles']['p1'])} | {_fmt(proxy_delta['quantiles']['p5'])} | {_fmt(proxy_delta['quantiles']['p50'])} | {_fmt(proxy_delta['quantiles']['p95'])} | {_fmt(proxy_delta['quantiles']['p99'])} |
| Oficjalna | {official_delta['rows']:,} | {_fmt(official_delta['mean'])} | {_fmt(official_delta['standard_deviation'])} | {_fmt(official_delta['mean_absolute'])} | {_fmt_pct(official_delta['positive_fraction'])} | {_fmt(official_delta['quantiles']['p1'])} | {_fmt(official_delta['quantiles']['p5'])} | {_fmt(official_delta['quantiles']['p50'])} | {_fmt(official_delta['quantiles']['p95'])} | {_fmt(official_delta['quantiles']['p99'])} |

## Ekonomia po kosztach

Wspólny kalendarz obejmuje {economics['availability']['decision_rows']:,} decyzji. Każdy wariant samodzielnie wybiera stronę i kwalifikację z progiem net EV 1%, niezależnym zleceniem $5, decyzją T−60 i wykonaniem T−59. Snapshoty, opłaty, głębokość, cutoff i księgowanie pozostały bez zmian.

Kontrola odtworzyła historyczny ledger SINGLE: {manifest['single_anchor_reproduction']['actual_fills']:,} rozliczonych wykonań, ${manifest['single_anchor_reproduction']['actual_turnover_usd']:,.2f} obrotu i ${manifest['single_anchor_reproduction']['actual_net_pnl_usd']:,.2f} PnL. Zgodność z oryginalnym ledgerem: zaliczona.

| Wariant | Wypełnienia | Obrót | Opłaty | PnL netto | PnL/obrót | Prognozowane EV | PnL − EV |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(overall_lines[2:])}

Miesięcznie:

{chr(10).join(monthly_econ_lines)}

Podział UP/DOWN:

{chr(10).join(side_lines)}

Koncentracja PnL w pięciu najlepszych dniach (dni bez transakcji pozostają w kalendarzu bootstrapu):

{chr(10).join(concentration_lines)}

Sparowane 95% przedziały różnic względem SINGLE, 2 000 bootstrapów bloków 3-dniowych, z zachowaniem pełnego kalendarza 72 dni:

| Miara | Punkt FULL_FEATURES − SINGLE | 95% przedział | Replikacje |
|---|---:|---:|---:|
| Średni dzienny PnL | ${primary_ci['mean_daily_pnl_difference_usd']['point_estimate']:,.4f} | ${primary_ci['mean_daily_pnl_difference_usd']['95pct'][0]:,.4f} do ${primary_ci['mean_daily_pnl_difference_usd']['95pct'][1]:,.4f} | {CONFIG['bootstrap']['replications']:,} |
| PnL/obrót | {_fmt_pct(primary_ci['pnl_per_settled_turnover_difference']['point_estimate'])} | {_fmt_pct(primary_ci['pnl_per_settled_turnover_difference']['95pct'][0])} do {_fmt_pct(primary_ci['pnl_per_settled_turnover_difference']['95pct'][1])} | {primary_ci['pnl_per_settled_turnover_difference']['valid_replications']:,} |

Decompozycja różnicy rozliczonego PnL; suma sprawdzona: **${decomp['realized_pnl_difference_usd']:,.2f}**. Zgodność: `{decomp['sum_reconciles']}`.

{chr(10).join(decomp_lines)}

## Runtime, zasoby i ograniczenia

- Dziesięć fitów po 31 iteracji zajęło łącznie {sum(fold_fit_seconds):.2f}s (zakres na fold {min(fold_fit_seconds):.2f}–{max(fold_fit_seconds):.2f}s). Pełny etap jakości trwał {completed_stage_seconds['quality_and_bootstrap']:.2f}s, a ekonomia i audyty trzech wariantów {completed_stage_seconds['economics_and_audits_all_variants']:.2f}s.
- Ostatni przebieg runnera trwał {manifest.get('elapsed_seconds', 0.0):.2f}s i wznowił {training_summary.get('reused_checkpoint_count', 0)} zgodnych checkpointów; to czas wznowienia z cache, nie łączny czas wcześniejszych obliczeń.
- Szczytowy working set procesu: {manifest.get('peak_process_working_set_gib', 0.0):.2f} GiB. Środowisko i współbieżne obciążenie zapisano w run manifeście.
- Zapisano 10 boosterów FULL_FEATURES i OOF po każdym foldzie. Runner bez argumentów CLI pomija tylko checkpointy o zgodnej tożsamości, hashu, indeksach walidacyjnych, kolejności cech i liczbie iteracji.
- Eksperyment jest warunkowy względem wcześniej wybranego BASE70, parametrów i development sample. Nie oceniano holdoutu, nie pobierano danych, nie odtwarzano archiwum ani nie wykonywano refitu wdrożeniowego.

## Wniosek

- **Jakość probabilistyczna:** {quality_signal}
- **Decyzje po kosztach:** {econ_signal}
- **Czy wyniki uzasadniają dalszą pracę z FULL_FEATURES:** {followup} Punktowy PnL sam nie rozstrzyga. Ten eksperyment nie wybiera zwycięzcy ani seedu. SINGLE pozostaje historycznym punktem odniesienia, nie potwierdzonym najlepszym modelem. Nie uruchomiono kolejnego eksperymentu.

Runner: `python {CONFIG['experiment_id'].replace('btc_', 'run_btc_')}.py` (plik: `run_btc_base70_full_features_20261018.py`); bez argumentów CLI. Manifest: `{_relative(VERSIONED_MANIFEST_PATH)}`.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def _finalize_versioned_manifest(manifest: dict) -> None:
    completed_records = [
        manifest["completed_fits"][str(fold_id)]
        for fold_id in range(10)
    ]
    outputs = []
    for stage in manifest.get("stages", {}).values():
        outputs.extend(stage.get("outputs", []))
    versioned = {
        "status": manifest["status"],
        "frozen_at_utc": manifest["frozen_at_utc"],
        "finished_utc": manifest.get("finished_utc"),
        "frozen_comparison": manifest["frozen_comparison"],
        "input_identity": manifest["input_identity"],
        "source_artifacts": manifest["source_artifacts"],
        "code_revision_history": manifest.get("code_revision_history", []),
        "fit_count": 10,
        "completed_fit_count": len(completed_records),
        "fit_checkpoint_records": completed_records,
        "stages": {
            name: {key: value for key, value in stage.items() if key != "outputs"}
            for name, stage in manifest.get("stages", {}).items()
        },
        "outputs": outputs,
        "result_summary": {
            "quality": manifest.get("stages", {}).get("quality", {}).get("summary"),
            "economics": manifest.get("stages", {}).get("economics_summary", {}).get("summary"),
            "single_anchor_reproduction": manifest.get("single_anchor_reproduction"),
        },
        "performance": {
            "elapsed_seconds": manifest.get("elapsed_seconds"),
            "last_orchestration_elapsed_seconds": manifest.get("elapsed_seconds"),
            "completed_fit_wall_seconds_sum": sum(
                float(record.get("elapsed_seconds", 0.0)) for record in completed_records
            ),
            "last_invocation_stage_elapsed_seconds": manifest.get("stage_elapsed_seconds"),
            "stage_elapsed_seconds": manifest.get("stage_elapsed_seconds"),
            "peak_process_working_set_gib": manifest.get("peak_process_working_set_gib"),
            "resource_snapshot_before_run": manifest.get("resource_snapshot_before_run"),
            "resource_snapshot_at_completion": manifest.get("resource_snapshot_at_completion"),
        },
        "completion": manifest.get("completion"),
    }
    _write_json(VERSIONED_MANIFEST_PATH, versioned)


def run() -> dict:
    run_started = time.perf_counter()
    stages_elapsed = {}
    resource_before = taker._resource_snapshot()
    source_started = time.perf_counter()
    flow_manifest, info_manifest, seed_manifest, seed_versioned, cv_summary, common_summary, source = _load_sources()
    artifacts = _source_artifacts(flow_manifest, info_manifest, seed_manifest, source)
    spec = _frozen_specification(info_manifest, seed_manifest, cv_summary, common_summary, artifacts)
    stages_elapsed["source_manifest_and_hash_verification"] = time.perf_counter() - source_started
    frame = x = target = weights = folds = None
    with taker.PeakMemorySampler() as memory:
        try:
            verify_started = time.perf_counter()
            frame, x, target, weights, folds, single_check = seed_stability._load_training_inputs(
                cv_summary,
                common_summary,
                verify_single=True,
            )
            weight_hash = taker._array_sha256(weights, dtype=np.float32)
            if weight_hash != cv_summary["training_weight_sha256"]:
                raise RuntimeError("Reconstructed target weights differ from the frozen source weight hash")
            stages_elapsed["source_verification_and_matrix_load"] = time.perf_counter() - verify_started

            manifest = _create_or_resume_manifest(spec, artifacts)
            manifest["resource_snapshot_before_run"] = resource_before
            manifest["source_verification"] = {
                "single_oof_and_all_ten_boosters_reproduced": single_check,
                "weights_sha256_recomputed": weight_hash,
                "fold_fingerprints": common_summary["common_fold_fingerprints"],
                "matrix_rows": int(len(frame)),
                "feature_order_identical": True,
                "dataset_loaded_once": True,
                "source_experiments_completed": True,
                "ensemble_oof_bundle_sha256": artifacts["saved_single_and_ensemble_oof_bundle"]["sha256"],
            }
            _finish_stage(
                manifest,
                "source_verification",
                summary=manifest["source_verification"],
                outputs=[artifacts["matrix"], artifacts["folds"], artifacts["single_oof"], artifacts["saved_single_and_ensemble_oof_bundle"]],
                elapsed_seconds=stages_elapsed["source_verification_and_matrix_load"],
            )
            prior_training_seconds = float(seed_manifest["stages"]["training"]["summary"]["elapsed_seconds"])
            estimated_training = prior_training_seconds * 10 / int(seed_manifest["stages"]["training"]["summary"]["new_fits"])
            manifest["training_runtime_estimate_seconds"] = estimated_training
            _write_manifest(manifest)
            print(
                f"Frozen before fit: 10 FULL_FEATURES fits x 31 rounds; only feature_fraction changes "
                f"({params_value(spec, 'feature_fraction')}). Estimated challenger training {estimated_training:.1f}s "
                f"from the preceding 40-fit seed-stability run; "
                f"RAM available before run: {resource_before.get('available_memory_gib', 'unknown')} GiB."
            )

            training_started = time.perf_counter()
            _train_full_features(manifest, frame, x, target, weights, folds, cv_summary["params"])
            stages_elapsed["training"] = time.perf_counter() - training_started
            predictions_started = time.perf_counter()
            predictions = _build_predictions(manifest)
            stages_elapsed["common_predictions"] = time.perf_counter() - predictions_started

            quality_started = time.perf_counter()
            quality, proxy_folds, monthly_quality = _quality(manifest, predictions, info_manifest)
            stages_elapsed["quality_and_bootstrap"] = time.perf_counter() - quality_started

            economics_started = time.perf_counter()
            calendar, t59, t60, execution_inputs = _load_economic_inputs(info_manifest)
            input_path = OUTPUT_DIR / "execution_input_summary.json"
            _write_json(input_path, execution_inputs)
            _finish_stage(manifest, "execution_inputs", summary=execution_inputs, outputs=[_record(input_path)], elapsed_seconds=execution_inputs["elapsed_seconds"])
            stages_elapsed["execution_input_load"] = execution_inputs["elapsed_seconds"]
            economics = _economics(manifest, predictions, calendar, t59, t60)
            stages_elapsed["economics_and_audits"] = time.perf_counter() - economics_started

            manifest["stage_elapsed_seconds"] = stages_elapsed
            manifest["peak_process_working_set_gib"] = round(memory.peak_bytes / 2**30, 3)
            manifest["resource_snapshot_at_completion"] = taker._resource_snapshot()
            manifest["status"] = "completed"
            manifest["active_stage"] = "report"
            manifest["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
            manifest["elapsed_seconds"] = time.perf_counter() - run_started
            manifest["next_step"] = "Experiment complete; review the separate probability-quality and after-cost economic results. Do not start another experiment automatically."
            _write_manifest(manifest)
            report_started = time.perf_counter()
            _write_report(manifest)
            _finish_stage(manifest, "report", outputs=[_record(REPORT_PATH)], elapsed_seconds=time.perf_counter() - report_started)
            manifest["completion"] = {
                "quality_reported_separately_from_economics": True,
                "only_intended_model_change_feature_fraction": True,
                "all_ten_fits_31_iterations": True,
                "complete_oof_rows": int(len(predictions)),
                "official_rows": int(quality["official"]["sample_rows"]),
                "economic_calendar_rows": int(economics["availability"]["decision_rows"]),
                "accounting_audits_passed": economics["accounting_audits_passed"],
                "pnl_decomposition_reconciles": economics["pnl_decomposition_FULL_FEATURES_minus_SINGLE"]["sum_reconciles"],
                "single_anchor_reproduced": manifest["single_anchor_reproduction"]["passed"],
            }
            manifest["active_stage"] = None
            manifest["elapsed_seconds"] = time.perf_counter() - run_started
            _write_manifest(manifest)
            _finalize_versioned_manifest(manifest)
            return manifest
        except Exception as exc:
            manifest_path = MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
                "experiment_id": CONFIG["experiment_id"], "stages": {}, "completed_fits": {},
            }
            manifest["status"] = "failed"
            manifest["failure"] = {
                "stage": manifest.get("active_stage"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            manifest["active_stage"] = None
            manifest["next_step"] = "Inspect the recorded failure, then rerun this no-argument script to resume only identity-matching checkpoints."
            if "stages_elapsed_seconds" not in manifest:
                manifest["stage_elapsed_seconds"] = stages_elapsed
            _write_manifest(manifest)
            raise


def params_value(spec: dict, name: str) -> str:
    return f"{spec['single']['effective_lightgbm_params'][name]} -> {spec['challenger']['effective_lightgbm_params'][name]}"


if __name__ == "__main__":
    run()
