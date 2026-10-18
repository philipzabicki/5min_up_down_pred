"""Compare the saved RAW BASE70 model with four fixed-round seed variants."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

import run_btc_base70_information_value_20261018 as base70
import run_btc_preopen_model_update as model_update
import run_btc_preopen_research as research
import run_btc_preopen_taker_flow_20261018 as taker


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_base70_seed_stability_20261018.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_directory"]
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
REPORT_PATH = ROOT / CONFIG["report_path"]
VERSIONED_MANIFEST_PATH = ROOT / CONFIG["versioned_manifest_path"]
STATUS_PATH = ROOT / CONFIG["status_path"]
STATUS_BEGIN = "<!-- BTC_BASE70_SEED_STABILITY_20261018_BEGIN -->"
STATUS_END = "<!-- BTC_BASE70_SEED_STABILITY_20261018_END -->"

FLOW_DIR = taker.OUTPUT_DIR
COMMON_MATRIX_PATH = FLOW_DIR / "common_complete_training_rows.parquet"
COMMON_FOLDS_PATH = FLOW_DIR / "common_folds.npz"
COMMON_SUMMARY_PATH = FLOW_DIR / "common_rows_summary.json"
BASE_OOF_PATH = FLOW_DIR / "B0_decision_oof.parquet"
BASE_CV_SUMMARY_PATH = FLOW_DIR / "B0_cv_summary.json"
BASE_MODEL_DIR = FLOW_DIR / "models/B0"
BASE_INFO_DIR = base70.OUTPUT_DIR
BASE_PREDICTIONS_PATH = BASE_INFO_DIR / "predictions.parquet"
EXECUTION_CALENDAR_PATH = BASE_INFO_DIR / "common_execution_calendar.parquet"
BASE_RAW_TRADES_PATH = BASE_INFO_DIR / "economics/RAW/trades.parquet"
BASE_RAW_DECISIONS_PATH = BASE_INFO_DIR / "economics/RAW/decisions.parquet"
BASE_RAW_SUMMARY_PATH = BASE_INFO_DIR / "economics/RAW/summary.json"
MARKET_INDEX_PATH = ROOT / CONFIG["market_index_path"]

FEATURE_METADATA_COLUMNS = [
    "_source_row_index", "Opened", "nominal_decision_at", "target_label_available_at",
    "target_preopen_5m_up_binance_index_proxy", "is_polymarket_preopen_decision",
]
ROW_ID_COLUMNS = [
    "_source_row_index", "Opened", "nominal_decision_at", "target_label_available_at",
    "target_preopen_5m_up_binance_index_proxy", "market_start_utc", "fold_id",
]
BOOTSTRAP_METRICS = ("log_loss", "brier", "auc")
SINGLE_NAME = "SINGLE"
SEED_NAMES = tuple(f"SEED_{seed}" for seed in CONFIG["new_seeds"])
COMPONENT_NAMES = (SINGLE_NAME, *SEED_NAMES)
VARIANT_NAMES = (*COMPONENT_NAMES, "ENSEMBLE")
PROBABILITY_COLUMNS = {name: f"p_{name}" for name in VARIANT_NAMES}


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
    for row in records:
        if not isinstance(row, dict) or not {"path", "sha256", "size_bytes"}.issubset(row):
            return False
        path = ROOT / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]) or _sha256(path) != row["sha256"]:
            return False
    return True


def _status_update(manifest: dict) -> None:
    stages = manifest.get("stages", {})
    completed = sorted(name for name, stage in stages.items() if stage.get("status") == "completed")
    fits = sum(
        item.get("status") == "completed"
        for seed_folds in manifest.get("completed_fits", {}).values()
        for item in seed_folds.values()
    )
    lines = [
        STATUS_BEGIN,
        "### BASE70 seed stability and equal-weight ensemble - 2026-10-18",
        f"- Updated UTC: `{pd.Timestamp.now(tz='UTC').isoformat()}`.",
        f"- Status: `{manifest.get('status')}`; active stage: `{manifest.get('active_stage') or 'none'}`.",
        f"- Completed stages: `{', '.join(completed) or 'none'}`; new fits: `{fits}/40`.",
        f"- Next step: `{manifest.get('next_step', 'Resume the no-argument runner.')}`.",
        f"- Run manifest: `{_relative(MANIFEST_PATH)}`.",
        STATUS_END,
        "",
    ]
    text = STATUS_PATH.read_text(encoding="utf-8")
    if STATUS_BEGIN in text and STATUS_END in text:
        before, tail = text.split(STATUS_BEGIN, 1)
        _, after = tail.split(STATUS_END, 1)
        text = before.rstrip() + "\n" + "\n".join(lines) + after.lstrip("\n")
    else:
        anchor = "# BTC research execution status"
        text = text.replace(anchor, "\n".join(lines) + anchor, 1)
    temporary = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(STATUS_PATH)


def _write_manifest(manifest: dict) -> None:
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(MANIFEST_PATH, manifest)
    _write_json(VERSIONED_MANIFEST_PATH, manifest)
    _status_update(manifest)


def _begin_stage(manifest: dict, stage: str, next_step: str) -> None:
    manifest["status"] = "running"
    manifest["active_stage"] = stage
    manifest["next_step"] = next_step
    manifest.setdefault("stages", {})[stage] = {
        "status": "running", "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write_manifest(manifest)


def _finish_stage(manifest: dict, stage: str, **payload) -> None:
    value = manifest.setdefault("stages", {}).setdefault(stage, {})
    value.update(payload)
    value["status"] = "completed"
    value["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    manifest["active_stage"] = None
    _write_manifest(manifest)


def _effective_params(base_params: dict, seed: int | None) -> dict:
    params = dict(base_params)
    if seed is not None:
        for name in ("seed", "feature_fraction_seed", "data_random_seed", "bagging_seed"):
            params[name] = int(seed)
    return params


def _ensemble_probability(component_probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(component_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(COMPONENT_NAMES):
        raise ValueError("ENSEMBLE requires one saved probability column for each of the five models")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Component probabilities must be finite and lie in [0, 1]")
    return values.mean(axis=1, dtype=np.float64)


def _effective_parameter_record(base_params: dict, seed: int | None) -> dict:
    params = _effective_params(base_params, seed)
    return {
        "lightgbm_params_passed": params,
        "num_boost_round": int(CONFIG["boosting_iterations"]),
        "early_stopping": False,
        "resolved_defaults_relevant_to_randomness": {
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "feature_fraction_bynode": 1.0,
            "extra_trees": False,
            "boosting_type": params["boosting_type"],
            "drop_seed": 4,
            "objective_seed": 5,
        },
        "seed_roles": {
            "feature_fraction_seed": "active: feature_fraction is below 1 and samples features for each tree",
            "data_random_seed": "active: Dataset construction samples rows to construct histogram bins",
            "bagging_seed": "inactive: bagging_fraction=1 and bagging_freq=0",
            "seed": "explicit root seed; explicit child seed parameters take precedence",
            "drop_seed": "inactive: gbdt does not use DART dropout",
            "objective_seed": "inactive: binary objective needs no random sampling",
            "extra_seed": "inactive: extra_trees is false",
        },
    }


def _source_inputs() -> tuple[dict, dict, dict, dict, dict]:
    flow_manifest = json.loads((ROOT / CONFIG["source_taker_flow_run_manifest_path"]).read_text(encoding="utf-8"))
    zv_manifest_path = ROOT / CONFIG["source_zv_run_manifest_path"]
    zv_manifest = json.loads(zv_manifest_path.read_text(encoding="utf-8"))
    zv_versioned = json.loads((ROOT / CONFIG["source_zv_versioned_manifest_path"]).read_text(encoding="utf-8"))
    info_manifest = json.loads((ROOT / CONFIG["source_base70_information_run_manifest_path"]).read_text(encoding="utf-8"))
    info_versioned = json.loads((ROOT / CONFIG["source_base70_information_versioned_manifest_path"]).read_text(encoding="utf-8"))
    if any(item.get("status") != "completed" for item in (flow_manifest, zv_manifest, info_manifest)):
        raise RuntimeError("All three source experiments must be completed before seed stability can start")
    if zv_manifest["frozen_comparison"]["identity_sha256"] != zv_versioned["frozen_comparison"]["identity_sha256"]:
        raise RuntimeError("Taker-flow Z/V versioned and local manifests have different frozen identities")
    if info_manifest["frozen_comparison"]["identity_sha256"] != info_versioned["frozen_comparison"]["identity_sha256"]:
        raise RuntimeError("BASE70 information-value versioned and local manifests have different frozen identities")

    flow_rows = flow_manifest.get("stages", {}).get("common_rows", {})
    flow_cv = flow_manifest.get("stages", {}).get("cv_B0", {})
    if flow_rows.get("status") != "completed" or flow_cv.get("status") != "completed":
        raise RuntimeError("Saved BASE70 shared-row matrix or CV stage is incomplete")
    if not taker._verify_records(flow_rows.get("outputs", [])) or not taker._verify_records(flow_cv.get("outputs", [])):
        raise RuntimeError("A saved BASE70 matrix, fold, OOF, or model artifact differs from its frozen hash")

    required_info_stages = ("common_predictions", "execution_calendar", "economics_RAW", "economics_summary")
    for stage_name in required_info_stages:
        stage = info_manifest.get("stages", {}).get(stage_name, {})
        if stage.get("status") != "completed" or not _verify_records(stage.get("outputs", [])):
            raise RuntimeError(f"Completed BASE70 information-value stage is missing or changed: {stage_name}")

    source_hashes = dict(info_manifest["input_identity"]["source_hashes"])
    for path_name, hash_name in (("economic_engine", "economic_engine"), ("research_engine", "research_engine")):
        if _sha256(ROOT / f"run_btc_preopen_{'t59_reassessment' if path_name == 'economic_engine' else 'research'}.py") != source_hashes[path_name]:
            raise RuntimeError(f"The frozen BASE70 {path_name} has changed since its source experiment")
    for path_name, hash_name in (("source_taker_flow_manifest", "taker_flow_manifest"),):
        if source_hashes.get(path_name) != _sha256(ROOT / CONFIG["source_taker_flow_run_manifest_path"]):
            raise RuntimeError(f"The frozen BASE70 source manifest hash differs: {path_name}")

    snapshot_identity = info_manifest["frozen_comparison"]["specification"]["sources"]["decision_and_execution_snapshots"]
    if _sha256(ROOT / snapshot_identity["t59_path"]) != snapshot_identity["t59_sha256"]:
        raise RuntimeError("The saved T-59 snapshot differs from the completed BASE70 comparison")
    if _sha256(ROOT / snapshot_identity["t60_path"]) != snapshot_identity["t60_sha256"]:
        raise RuntimeError("The saved T-60 snapshot differs from the completed BASE70 comparison")
    if _sha256(MARKET_INDEX_PATH) != source_hashes["source_official_market_index"]:
        raise RuntimeError("The official market index changed since the BASE70 information-value comparison")

    cv_summary = json.loads(BASE_CV_SUMMARY_PATH.read_text(encoding="utf-8"))
    common_summary = json.loads(COMMON_SUMMARY_PATH.read_text(encoding="utf-8"))
    if int(cv_summary["best_iteration"]) != int(CONFIG["boosting_iterations"]):
        raise RuntimeError("Saved BASE70 control does not use the frozen 31-iteration model")
    if int(cv_summary["feature_count"]) != 70 or len(cv_summary["feature_order"]) != 70:
        raise RuntimeError("Saved BASE70 feature order does not contain exactly 70 features")
    if int(cv_summary["rows"]) != int(common_summary["rows_after_shared_complete_feature_mask"]):
        raise RuntimeError("Saved BASE70 CV row count differs from its shared matrix")
    base_params = cv_summary["params"]
    if any(int(base_params[name]) != int(CONFIG["single_seed"]) for name in ("seed", "feature_fraction_seed", "data_random_seed", "bagging_seed")):
        raise RuntimeError("The source BASE70 seed parameters differ from the requested SINGLE control")
    if float(base_params["feature_fraction"]) >= 1.0:
        raise RuntimeError("The frozen BASE70 model has no feature-fraction randomness")

    source_records = base70._source_records(flow_manifest)
    common_records = {row["path"]: row for row in flow_rows["outputs"]}
    info_records = {
        row["path"]: row
        for stage_name in required_info_stages
        for row in info_manifest["stages"][stage_name].get("outputs", [])
    }
    required = {
        "matrix": _relative(COMMON_MATRIX_PATH), "folds": _relative(COMMON_FOLDS_PATH),
        "source_oof": _relative(BASE_OOF_PATH), "source_cv_summary": _relative(BASE_CV_SUMMARY_PATH),
        "source_predictions": _relative(BASE_PREDICTIONS_PATH),
        "execution_calendar": _relative(EXECUTION_CALENDAR_PATH),
        "source_trades": _relative(BASE_RAW_TRADES_PATH),
        "source_decisions": _relative(BASE_RAW_DECISIONS_PATH),
        "source_economic_summary": _relative(BASE_RAW_SUMMARY_PATH),
    }
    source_output_records = {
        "matrix": common_records.get(required["matrix"]),
        "folds": common_records.get(required["folds"]),
        "source_oof": source_records.get(required["source_oof"]),
        "source_cv_summary": source_records.get(required["source_cv_summary"]),
        "source_predictions": info_records.get(required["source_predictions"]),
        "execution_calendar": info_records.get(required["execution_calendar"]),
        "source_trades": info_records.get(required["source_trades"]),
        "source_decisions": info_records.get(required["source_decisions"]),
        "source_economic_summary": info_records.get(required["source_economic_summary"]),
    }
    if any(value is None for value in source_output_records.values()):
        raise RuntimeError("A required frozen source artifact is absent from its source manifest")

    records = {
        "config": _record(CONFIG_PATH),
        "runner": _record(Path(__file__)),
        "taker_flow_manifest": _record(ROOT / CONFIG["source_taker_flow_run_manifest_path"]),
        "zv_manifest": _record(ROOT / CONFIG["source_zv_run_manifest_path"]),
        "zv_versioned_manifest": _record(ROOT / CONFIG["source_zv_versioned_manifest_path"]),
        "zv_report": _record(ROOT / "reports/btc_preopen/btc_preopen_zv_ablation_20261018.md"),
        "base70_information_manifest": _record(ROOT / CONFIG["source_base70_information_run_manifest_path"]),
        "base70_information_versioned_manifest": _record(ROOT / CONFIG["source_base70_information_versioned_manifest_path"]),
        "base70_information_report": _record(ROOT / CONFIG["source_base70_information_report_path"]),
        "official_market_index": _record(MARKET_INDEX_PATH),
        "economic_engine": _record(ROOT / "run_btc_preopen_t59_reassessment.py"),
        "research_engine": _record(ROOT / "run_btc_preopen_research.py"),
        "taker_flow_runner": _record(ROOT / "run_btc_preopen_taker_flow_20261018.py"),
        "source_outputs": source_output_records,
        "snapshots": snapshot_identity,
    }
    return flow_manifest, info_manifest, cv_summary, common_summary, {"source_records": records, "base_params": base_params}


def _frozen_model_parameters(base_params: dict) -> dict:
    models = {
        SINGLE_NAME: _effective_parameter_record(base_params, None),
    }
    for name, seed in zip(SEED_NAMES, CONFIG["new_seeds"], strict=True):
        models[name] = _effective_parameter_record(base_params, int(seed))
    return models


def _create_or_resume_manifest(
    source: dict,
    flow_manifest: dict,
    info_manifest: dict,
    cv_summary: dict,
    common_summary: dict,
) -> dict:
    model_parameters = _frozen_model_parameters(source["base_params"])
    snapshot_identity = info_manifest["frozen_comparison"]["specification"]["sources"]["decision_and_execution_snapshots"]
    frozen = {
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "single_model": {
            "source": _relative(BASE_OOF_PATH), "seed": int(CONFIG["single_seed"]),
            "iterations": int(CONFIG["boosting_iterations"]), "retrained": False,
        },
        "new_seeds": [int(value) for value in CONFIG["new_seeds"]],
        "model_parameters_all_five": model_parameters,
        "ensemble": CONFIG["ensemble_rule"],
        "data": {
            "matrix": _relative(COMMON_MATRIX_PATH),
            "rows": int(cv_summary["rows"]),
            "feature_order": cv_summary["feature_order"],
            "features": len(cv_summary["feature_order"]),
            "common_row_mask": "reuse the saved common BASE70/B1 complete-feature mask; no regeneration",
            "folds": _relative(COMMON_FOLDS_PATH),
            "fold_fingerprints": common_summary["common_fold_fingerprints"],
            "weights_sha256": cv_summary["training_weight_sha256"],
            "target": "target_preopen_5m_up_binance_index_proxy",
            "purge": "saved train indices; target_label_available_at <= first validation nominal_decision_at",
        },
        "economics": {
            **CONFIG["economics"],
            "calendar": "reuse existing BASE70 common_execution_calendar.parquet exactly",
            "snapshot_identity": snapshot_identity,
            "prediction_calibration": "none; raw probabilities only",
            "fills": "each variant independently selects side and execution; no common-fill filter",
        },
        "bootstrap": CONFIG["bootstrap"],
        "boundaries": {
            "development_start_utc": base70.START_UTC,
            "development_cutoff_exclusive_utc": base70.CUTOFF_UTC,
            "official_quality_sample": "saved BASE70 comparison mask and official labels; requires both to match exactly",
        },
        "source_statuses": {
            "taker_flow": flow_manifest["status"],
            "zv_ablation": "completed",
            "base70_information_value": info_manifest["status"],
        },
        "source_hashes": source["source_records"],
    }
    stable_frozen = {key: value for key, value in frozen.items() if key != "frozen_at_utc"}
    identity = {"frozen_spec_sha256": _canonical_sha256(stable_frozen), "runner_sha256": source["source_records"]["runner"]["sha256"]}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.is_file():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("input_identity") != identity:
            no_fit_completed = not any(
                record.get("status") == "completed"
                for folds in manifest.get("completed_fits", {}).values()
                for record in folds.values()
            )
            old_comparable = json.loads(json.dumps(manifest.get("frozen_comparison", {}).get("specification", {})))
            new_comparable = json.loads(json.dumps(frozen, default=str))
            for value in (old_comparable, new_comparable):
                value.pop("frozen_at_utc", None)
                value.get("source_hashes", {}).pop("runner", None)
            same_sources_except_runner = (
                old_comparable == new_comparable
            )
            if not same_sources_except_runner:
                raise RuntimeError("Existing seed-stability checkpoints belong to different inputs, parameters, or code")
            fit_migration = None
            if not no_fit_completed:
                fit_migration = _migrate_completed_fit_identity(
                    manifest, model_parameters, identity, cv_summary,
                )
            manifest.setdefault("code_revision_history", []).append({
                "previous_runner_sha256": manifest.get("input_identity", {}).get("runner_sha256"),
                "current_runner_sha256": identity["runner_sha256"],
                "reason": (
                    "Runner changed after all 40 completed fit checkpoints were verified against unchanged sources and exact effective parameters; downstream stages were invalidated and fitted artifacts were retained."
                    if fit_migration else
                    "Runner changed before any seed/fold fit completed; the frozen parameter/source manifest was refreshed before training."
                ),
            })
            manifest["input_identity"] = identity
            manifest["frozen_comparison"] = {"identity_sha256": identity["frozen_spec_sha256"], "specification": frozen}
            manifest["stages"] = fit_migration or {}
            manifest["active_stage"] = None
            manifest["status"] = "ready_to_resume_after_code_update" if fit_migration else "frozen_before_training"
            manifest["next_step"] = (
                "Resume downstream quality and economic analysis using the 40 verified saved fits."
                if fit_migration else
                "Verify the saved matrix and SINGLE control, then fit four new seeds across ten frozen folds."
            )
            manifest.pop("failure", None)
            manifest["runtime_identity"]["runner_sha256"] = identity["runner_sha256"]
            _write_manifest(manifest)
            return manifest
        old_frozen = {key: value for key, value in manifest.get("frozen_comparison", {}).get("specification", {}).items() if key != "frozen_at_utc"}
        if _canonical_sha256(old_frozen) != identity["frozen_spec_sha256"]:
            raise RuntimeError("Existing frozen seed-stability specification differs from the current run")
        return manifest
    manifest = {
        "schema_version": 1,
        "experiment_id": CONFIG["experiment_id"],
        "status": "frozen_before_training",
        "active_stage": None,
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "next_step": "Verify the saved matrix and SINGLE control, then fit four new seeds across ten frozen folds.",
        "input_identity": identity,
        "frozen_comparison": {"identity_sha256": identity["frozen_spec_sha256"], "specification": frozen},
        "source_baseline_economic_anchor": info_manifest["stages"]["economics_summary"]["summary"]["variants"]["RAW"],
        "runtime_identity": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "lightgbm": taker.lgb.__version__, "pyarrow": pa.__version__,
            "runner_sha256": identity["runner_sha256"],
        },
        "resource_snapshot_before_run": taker._resource_snapshot(),
        "completed_fits": {},
        "stages": {},
    }
    _write_manifest(manifest)
    _write_json(VERSIONED_MANIFEST_PATH, manifest)
    return manifest


def _migrate_completed_fit_identity(
    manifest: dict,
    model_parameters: dict,
    identity: dict,
    cv_summary: dict,
) -> dict:
    expected_count = len(CONFIG["new_seeds"]) * 10
    actual_count = sum(
        record.get("status") == "completed"
        for folds in manifest.get("completed_fits", {}).values()
        for record in folds.values()
    )
    if actual_count != expected_count:
        raise RuntimeError("Runner code changed after only some fits completed; refusing to migrate partial training checkpoints")
    with np.load(COMMON_FOLDS_PATH) as saved:
        folds = [(saved[f"train_{i}"].astype(np.int32, copy=False), saved[f"valid_{i}"].astype(np.int32, copy=False)) for i in range(10)]
    feature_names = list(cv_summary["feature_order"])
    training_stage = manifest.get("stages", {}).get("training", {})
    if training_stage.get("status") != "completed" or not _verify_records(training_stage.get("outputs", [])):
        raise RuntimeError("All fit checkpoints exist but the training stage was not completed and verified; refusing code migration")
    migration_plan = []
    completion_paths = set()
    for seed, name in zip(CONFIG["new_seeds"], SEED_NAMES, strict=True):
        seed = int(seed)
        expected_params = model_parameters[name]["lightgbm_params_passed"]
        for fold_id, (_, valid_idx) in enumerate(folds):
            if not _completed_fold_valid(manifest, seed, fold_id, None, valid_idx, feature_names):
                raise RuntimeError(f"Cannot migrate unverified completed checkpoint for seed {seed}, fold {fold_id}")
            _, _, completion_path = _fold_paths(seed, fold_id)
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if (
                completion.get("effective_params") != expected_params
                or completion.get("iteration_count") != int(CONFIG["boosting_iterations"])
                or completion.get("validation_rows") != len(valid_idx)
            ):
                raise RuntimeError(f"Completed checkpoint parameters differ for seed {seed}, fold {fold_id}")
            completion_paths.add(_relative(completion_path))
            migration_plan.append((seed, fold_id, completion_path, completion))
    stage_output_paths = {record.get("path") for record in training_stage.get("outputs", [])}
    if len(completion_paths) != expected_count or not completion_paths.issubset(stage_output_paths):
        raise RuntimeError("Training stage does not record every completed fit manifest")

    updated_completion_records = {}
    for seed, fold_id, completion_path, completion in migration_plan:
        completion["identity_sha256"] = identity["frozen_spec_sha256"]
        _write_json(completion_path, completion)
        fit_record = manifest["completed_fits"][str(seed)][str(fold_id)]
        fit_record["identity_sha256"] = identity["frozen_spec_sha256"]
        fit_record["completion_manifest"] = _record(completion_path)
        updated_completion_records[_relative(completion_path)] = fit_record["completion_manifest"]

    training_stage["outputs"] = [
        updated_completion_records.get(record.get("path"), record)
        for record in training_stage.get("outputs", [])
    ]
    if not _verify_records(training_stage["outputs"]):
        raise RuntimeError("Migrated training-stage output records do not verify")
    preserved_stages = {}
    for name in ("source_verification", "training"):
        stage = manifest.get("stages", {}).get(name)
        if stage and stage.get("status") == "completed" and _verify_records(stage.get("outputs", [])):
            preserved_stages[name] = stage
    return preserved_stages


def _load_training_inputs(cv_summary: dict, common_summary: dict, verify_single: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]], dict | None]:
    started = time.perf_counter()
    feature_names = list(cv_summary["feature_order"])
    if len(feature_names) != 70 or len(set(feature_names)) != 70:
        raise RuntimeError("Frozen BASE70 feature order is not exactly 70 unique features")
    frame = pd.read_parquet(COMMON_MATRIX_PATH, columns=[*FEATURE_METADATA_COLUMNS, *feature_names])
    if len(frame) != int(cv_summary["rows"]):
        raise RuntimeError("Loaded BASE70 common matrix row count differs from its saved CV summary")
    if frame["_source_row_index"].duplicated().any() or not frame["_source_row_index"].is_monotonic_increasing:
        raise RuntimeError("The common feature matrix lost its frozen chronological row identity")
    frame["Opened"] = pd.to_datetime(frame["Opened"], utc=True, errors="raise")
    frame["nominal_decision_at"] = pd.to_datetime(frame["nominal_decision_at"], utc=True, errors="raise")
    frame["target_label_available_at"] = pd.to_datetime(frame["target_label_available_at"], utc=True, errors="raise")
    frame["market_start_utc"] = frame["Opened"] + pd.Timedelta(minutes=2)
    x = frame.loc[:, feature_names].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(x).all():
        raise RuntimeError("The frozen BASE70 feature matrix contains non-finite values")
    target = frame["target_preopen_5m_up_binance_index_proxy"].to_numpy(dtype=np.int8, copy=False)
    if not np.isin(target, (0, 1)).all():
        raise RuntimeError("The frozen BASE70 target contains non-binary labels")
    decision = frame["is_polymarket_preopen_decision"].to_numpy(dtype=bool, copy=False)
    weights = taker.experiment._preopen_sample_weights(decision, float(cv_summary["decision_weight"]))
    if taker._array_sha256(weights, dtype=np.float32) != cv_summary["training_weight_sha256"]:
        raise RuntimeError("Reconstructed BASE70 sample weights differ from the saved weight hash")

    with np.load(COMMON_FOLDS_PATH) as saved:
        folds = [(saved[f"train_{i}"].astype(np.int32, copy=False), saved[f"valid_{i}"].astype(np.int32, copy=False)) for i in range(10)]
    fold_records = [{"fold_id": i, "train_idx": train, "valid_idx": valid} for i, (train, valid) in enumerate(folds)]
    actual_fingerprints = taker.experiment._fold_index_fingerprints(fold_records)
    if actual_fingerprints != common_summary["common_fold_fingerprints"]:
        raise RuntimeError("Loaded frozen fold indices do not match the common-row fingerprint")
    availability = frame["target_label_available_at"].array.asi8
    decisions_ns = frame["nominal_decision_at"].array.asi8
    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        if not len(train_idx) or not len(valid_idx):
            raise RuntimeError(f"Frozen fold {fold_id} is empty")
        if not decision[valid_idx].all():
            raise RuntimeError(f"Frozen fold {fold_id} validates non-decision rows")
        if availability[train_idx].max() > decisions_ns[valid_idx].min():
            raise RuntimeError(f"Frozen fold {fold_id} violates the target label-availability purge")
        if not np.array_equal(train_idx, np.arange(train_idx[0], train_idx[-1] + 1, dtype=np.int32)):
            raise RuntimeError(f"Frozen fold {fold_id} training indices are not a contiguous saved range")
    single_check = _verify_saved_single(frame, x, folds, feature_names) if verify_single else None
    print(
        f"Loaded {len(frame):,} rows x {len(feature_names)} BASE70 features once; "
        f"fold/weight/purge validation passed in {time.perf_counter() - started:.1f}s."
    )
    return frame, x, target, weights, folds, single_check


def _verify_saved_single(
    frame: pd.DataFrame,
    x: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_names: list[str],
) -> dict:
    saved = pd.read_parquet(BASE_OOF_PATH)
    expected_mask = np.zeros(len(frame), dtype=bool)
    expected_fold = np.full(len(frame), -1, dtype=np.int8)
    for fold_id, (_, valid_idx) in enumerate(folds):
        expected_mask[valid_idx] = True
        expected_fold[valid_idx] = fold_id
    expected_rows = np.flatnonzero(expected_mask)
    if len(saved) != len(expected_rows):
        raise RuntimeError("Saved SINGLE OOF row count differs from the ten frozen validation folds")
    if not np.array_equal(saved["_source_row_index"].to_numpy(dtype=np.int32), frame["_source_row_index"].iloc[expected_rows].to_numpy(dtype=np.int32)):
        raise RuntimeError("Saved SINGLE OOF rows differ from the frozen validation rows")
    if not np.array_equal(saved["fold_id"].astype(str).to_numpy(), expected_fold[expected_rows].astype(str)):
        raise RuntimeError("Saved SINGLE OOF fold ids differ from the frozen folds")
    target_name = "target_preopen_5m_up_binance_index_proxy"
    if not np.array_equal(saved[target_name].to_numpy(dtype=np.int8), frame[target_name].iloc[expected_rows].to_numpy(dtype=np.int8)):
        raise RuntimeError("Saved SINGLE OOF target differs from the frozen common matrix")
    saved_probability = saved["p_model_raw"].to_numpy(dtype=np.float32)
    for fold_id, (_, valid_idx) in enumerate(folds):
        model_path = BASE_MODEL_DIR / f"fold_{fold_id:02d}.txt"
        booster = taker.lgb.Booster(model_file=str(model_path))
        if booster.current_iteration() != int(CONFIG["boosting_iterations"]):
            raise RuntimeError(f"Frozen SINGLE fold {fold_id} is not exactly 31 iterations")
        if list(booster.feature_name()) != feature_names:
            raise RuntimeError(f"Frozen SINGLE fold {fold_id} feature order changed")
        prediction = np.asarray(booster.predict(x[valid_idx], num_iteration=int(CONFIG["boosting_iterations"])), dtype=np.float32)
        if not np.allclose(prediction, saved_probability[expected_fold[expected_rows] == fold_id], rtol=1e-6, atol=1e-7):
            raise RuntimeError(f"Frozen SINGLE fold {fold_id} booster does not reproduce its saved OOF predictions")
        del booster, prediction
    return {"oof_rows": int(len(saved)), "folds_reproduced": len(folds), "iteration_count": int(CONFIG["boosting_iterations"]), "passed": True}


def _write_fold_oof(frame: pd.DataFrame, valid_idx: np.ndarray, fold_id: int, probability: np.ndarray) -> pd.DataFrame:
    target_name = "target_preopen_5m_up_binance_index_proxy"
    result = frame.iloc[valid_idx][[
        "_source_row_index", "Opened", "nominal_decision_at", "target_label_available_at",
        target_name, "market_start_utc",
    ]].copy()
    result["fold_id"] = str(fold_id)
    result["p_model_raw"] = np.asarray(probability, dtype=np.float32)
    result["prediction_available_at_utc"] = pd.NaT
    return result


def _fold_paths(seed: int, fold_id: int) -> tuple[Path, Path, Path]:
    folder = OUTPUT_DIR / "models" / f"seed_{seed}"
    return (
        folder / f"fold_{fold_id:02d}.txt",
        folder / f"fold_{fold_id:02d}_oof.parquet",
        folder / f"fold_{fold_id:02d}_completion.json",
    )


def _completed_fold_valid(
    manifest: dict,
    seed: int,
    fold_id: int,
    frame: pd.DataFrame | None,
    valid_idx: np.ndarray,
    feature_names: list[str],
) -> bool:
    record = manifest.get("completed_fits", {}).get(str(seed), {}).get(str(fold_id), {})
    model_path, oof_path, completion_path = _fold_paths(seed, fold_id)
    if record.get("status") != "completed" or record.get("identity_sha256") != manifest["input_identity"]["frozen_spec_sha256"]:
        return False
    if record.get("completion_manifest", {}).get("path") != _relative(completion_path):
        return False
    if not _verify_records([
        record.get("model", {}), record.get("oof", {}), record.get("completion_manifest", {}),
    ]):
        return False
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("seed") != seed or completion.get("fold_id") != fold_id or completion.get("identity_sha256") != record["identity_sha256"]:
        return False
    if completion.get("model") != record.get("model") or completion.get("oof") != record.get("oof"):
        return False
    oof = pd.read_parquet(oof_path)
    source_indices = oof["_source_row_index"].to_numpy(dtype=np.int32)
    if len(oof) != len(valid_idx) or completion.get("validation_source_row_index_sha256") != taker._array_sha256(source_indices, dtype=np.int32):
        return False
    if frame is not None and not np.array_equal(source_indices, frame["_source_row_index"].iloc[valid_idx].to_numpy(dtype=np.int32)):
        return False
    if not oof["fold_id"].astype(str).eq(str(fold_id)).all() or not np.isfinite(oof["p_model_raw"]).all():
        return False
    booster = taker.lgb.Booster(model_file=str(model_path))
    valid = booster.current_iteration() == int(CONFIG["boosting_iterations"]) and list(booster.feature_name()) == feature_names
    del booster, oof
    return bool(valid)


def _pending_folds(
    manifest: dict,
    seed: int,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_names: list[str],
    frame: pd.DataFrame | None = None,
) -> list[int]:
    return [
        fold_id for fold_id, (_, valid_idx) in enumerate(folds)
        if not _completed_fold_valid(manifest, seed, fold_id, frame, valid_idx, feature_names)
    ]


def _train_new_models(
    manifest: dict,
    frame: pd.DataFrame | None,
    x: np.ndarray | None,
    target: np.ndarray | None,
    weights: np.ndarray | None,
    folds: list[tuple[np.ndarray, np.ndarray]],
    base_params: dict,
) -> dict:
    stage_started = time.perf_counter()
    _begin_stage(manifest, "training", "Fit and checkpoint each remaining seed/fold; then build OOF and the fixed equal-weight ensemble.")
    feature_names = list(json.loads(BASE_CV_SUMMARY_PATH.read_text(encoding="utf-8"))["feature_order"])
    output_records = []
    expected_single_ids = pd.read_parquet(BASE_OOF_PATH, columns=["_source_row_index"])["_source_row_index"].to_numpy(dtype=np.int32)
    with taker.PeakMemorySampler() as memory:
        for seed in CONFIG["new_seeds"]:
            seed = int(seed)
            params = _effective_params(base_params, seed)
            missing = _pending_folds(manifest, seed, folds, feature_names, frame)
            dataset = None
            if missing:
                if frame is None or x is None or target is None or weights is None:
                    raise RuntimeError("A missing fold needs the shared matrix loaded once before fitting")
                dataset = taker.lgb.Dataset(
                    x, label=target, weight=weights, feature_name=feature_names,
                    free_raw_data=False, params=params,
                )
                dataset.construct()
            fold_outputs = []
            for fold_id, (train_idx, valid_idx) in enumerate(folds):
                model_path, oof_path, completion_path = _fold_paths(seed, fold_id)
                if fold_id not in missing:
                    fold_outputs.append(pd.read_parquet(oof_path))
                    continue
                fold_started = time.perf_counter()
                train_set = dataset.subset(train_idx, params=params)
                booster = taker.lgb.train(
                    params=params,
                    train_set=train_set,
                    num_boost_round=int(CONFIG["boosting_iterations"]),
                    keep_training_booster=False,
                )
                if booster.current_iteration() != int(CONFIG["boosting_iterations"]):
                    raise RuntimeError(f"Seed {seed} fold {fold_id} did not fit exactly 31 iterations")
                prediction = np.asarray(
                    booster.predict(x[valid_idx], num_iteration=int(CONFIG["boosting_iterations"])),
                    dtype=np.float32,
                )
                if not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 1.0)):
                    raise RuntimeError(f"Seed {seed} fold {fold_id} produced invalid OOF probabilities")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_model = model_path.with_suffix(".txt.tmp")
                booster.save_model(str(temporary_model), num_iteration=int(CONFIG["boosting_iterations"]))
                temporary_model.replace(model_path)
                fold_oof = _write_fold_oof(frame, valid_idx, fold_id, prediction)
                _write_parquet(fold_oof, oof_path)
                completion = {
                    "identity_sha256": manifest["input_identity"]["frozen_spec_sha256"],
                    "seed": seed,
                    "fold_id": fold_id,
                    "iteration_count": int(booster.current_iteration()),
                    "effective_params": params,
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(valid_idx)),
                    "training_label_available_at_utc": frame["target_label_available_at"].iloc[train_idx].max().isoformat(),
                    "validation_first_decision_at_utc": frame["nominal_decision_at"].iloc[valid_idx].min().isoformat(),
                    "validation_last_decision_at_utc": frame["nominal_decision_at"].iloc[valid_idx].max().isoformat(),
                    "validation_source_row_index_sha256": taker._array_sha256(fold_oof["_source_row_index"].to_numpy(dtype=np.int32), dtype=np.int32),
                    "elapsed_seconds": time.perf_counter() - fold_started,
                    "model": _record(model_path),
                    "oof": _record(oof_path),
                }
                _write_json(completion_path, completion)
                fit_record = {
                    "status": "completed",
                    "identity_sha256": manifest["input_identity"]["frozen_spec_sha256"],
                    "model": completion["model"],
                    "oof": completion["oof"],
                    "completion_manifest": _record(completion_path),
                    "iteration_count": completion["iteration_count"],
                    "elapsed_seconds": completion["elapsed_seconds"],
                }
                manifest.setdefault("completed_fits", {}).setdefault(str(seed), {})[str(fold_id)] = fit_record
                fold_outputs.append(fold_oof)
                print(f"Completed seed {seed}, fold {fold_id + 1}/10 in {fit_record['elapsed_seconds']:.1f}s.")
                _write_manifest(manifest)
                del train_set, booster, prediction, fold_oof, completion
                gc.collect()
            if dataset is not None:
                del dataset
                gc.collect()
            combined = pd.concat(fold_outputs, ignore_index=True)
            combined.sort_values("_source_row_index", kind="stable", inplace=True)
            combined.reset_index(drop=True, inplace=True)
            if len(combined) != len(expected_single_ids) or not np.array_equal(combined["_source_row_index"].to_numpy(dtype=np.int32), expected_single_ids):
                raise RuntimeError(f"Seed {seed} OOF rows do not match the common frozen validation set")
            if combined["_source_row_index"].duplicated().any():
                raise RuntimeError(f"Seed {seed} has duplicate OOF rows")
            combined_path = OUTPUT_DIR / f"oof_seed_{seed}.parquet"
            _write_parquet(combined, combined_path)
            output_records.append(_record(combined_path))
            del combined, fold_outputs
            gc.collect()
        all_fit_seconds = [
            float(record.get("elapsed_seconds", 0.0))
            for seed_folds in manifest.get("completed_fits", {}).values()
            for record in seed_folds.values()
            if record.get("status") == "completed"
        ]
        summary = {
            "new_seeds": [int(seed) for seed in CONFIG["new_seeds"]],
            "new_fits": 40,
            "iterations_per_fit": int(CONFIG["boosting_iterations"]),
            "fit_seconds": float(sum(all_fit_seconds)),
            "mean_seconds_per_new_fit": float(np.mean(all_fit_seconds)) if all_fit_seconds else None,
            "seed_fold_fit_seconds": all_fit_seconds,
            "all_fits_completed": all(
                manifest.get("completed_fits", {}).get(str(seed), {}).get(str(fold_id), {}).get("iteration_count") == int(CONFIG["boosting_iterations"])
                for seed in CONFIG["new_seeds"] for fold_id in range(10)
            ),
        }
        if not summary["all_fits_completed"]:
            raise RuntimeError("The four-seed training stage is incomplete")
        for seed in CONFIG["new_seeds"]:
            for fold_id in range(10):
                record = manifest["completed_fits"][str(seed)][str(fold_id)]
                output_records.extend([record["model"], record["oof"], record["completion_manifest"]])
        summary["elapsed_seconds"] = time.perf_counter() - stage_started
        summary["peak_process_working_set_gib"] = round(memory.peak_bytes / 2**30, 3)
        summary_path = OUTPUT_DIR / "training_summary.json"
        _write_json(summary_path, summary)
        output_records.append(_record(summary_path))
    _finish_stage(
        manifest, "training", summary=summary, outputs=output_records,
        elapsed_seconds=time.perf_counter() - stage_started,
        peak_process_working_set_gib=round(memory.peak_bytes / 2**30, 3),
        completed_new_fits=40,
    )
    return summary


def _build_predictions(manifest: dict) -> pd.DataFrame:
    path = OUTPUT_DIR / "predictions.parquet"
    stage = manifest.get("stages", {}).get("common_predictions", {})
    if stage.get("status") == "completed" and _verify_records(stage.get("outputs", [])):
        return pd.read_parquet(path)
    started = time.perf_counter()
    _begin_stage(manifest, "common_predictions", "Score the full proxy OOF and the frozen 20,718-label official sample.")
    single = pd.read_parquet(BASE_OOF_PATH)
    key_columns = ROW_ID_COLUMNS
    single = single.loc[:, [*key_columns, "p_model_raw", "prediction_available_at_utc"]].copy()
    single["fold_id"] = single["fold_id"].astype(str)
    single.rename(columns={"p_model_raw": PROBABILITY_COLUMNS[SINGLE_NAME]}, inplace=True)
    if not np.isfinite(single[PROBABILITY_COLUMNS[SINGLE_NAME]]).all():
        raise RuntimeError("The reused SINGLE BASE70 OOF contains invalid probabilities")
    result = single
    for seed, name in zip(CONFIG["new_seeds"], SEED_NAMES, strict=True):
        fold_paths = [_fold_paths(int(seed), fold_id)[1] for fold_id in range(10)]
        folds = [pd.read_parquet(item) for item in fold_paths]
        current = pd.concat(folds, ignore_index=True).sort_values("_source_row_index", kind="stable").reset_index(drop=True)
        if len(current) != len(single) or not current[key_columns].equals(single[key_columns]):
            raise RuntimeError(f"Seed {seed} OOF identity differs from the reused SINGLE OOF")
        result[PROBABILITY_COLUMNS[name]] = current["p_model_raw"].to_numpy(dtype=np.float32)
        del current, folds
    component_values = result[[PROBABILITY_COLUMNS[name] for name in COMPONENT_NAMES]].to_numpy(dtype=np.float64)
    result[PROBABILITY_COLUMNS["ENSEMBLE"]] = _ensemble_probability(component_values)
    expected = _ensemble_probability(component_values)
    if not np.array_equal(result[PROBABILITY_COLUMNS["ENSEMBLE"]].to_numpy(dtype=np.float64), expected):
        raise RuntimeError("ENSEMBLE is not the exact arithmetic mean of the five saved component probabilities")
    if result["_source_row_index"].duplicated().any() or not np.isfinite(result[[PROBABILITY_COLUMNS[name] for name in VARIANT_NAMES]]).all().all():
        raise RuntimeError("Common OOF predictions contain duplicate rows or non-finite probabilities")
    _write_parquet(result, path)
    record = _record(path)
    summary = {
        "proxy_oof_rows": int(len(result)),
        "ensemble_rule": CONFIG["ensemble_rule"],
        "ensemble_arithmetic_mean_verified": True,
        "single_source_oof_reused": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _finish_stage(manifest, "common_predictions", summary=summary, outputs=[record], elapsed_seconds=summary["elapsed_seconds"])
    return result


def _score(labels: np.ndarray, probability: np.ndarray) -> dict:
    return taker._score_binary(np.asarray(labels, dtype=np.int8), np.asarray(probability, dtype=np.float64))


def _quality_table(rows: pd.DataFrame, label_column: str, group_column: str | None = None) -> tuple[dict, list[dict]]:
    labels = rows[label_column].to_numpy(dtype=np.int8)
    overall = {
        name: _score(labels, rows[PROBABILITY_COLUMNS[name]].to_numpy(dtype=np.float64))
        for name in VARIANT_NAMES
    }
    groups = []
    if group_column:
        grouped = rows.groupby(group_column, sort=True, dropna=False)
    else:
        grouped = [("overall", rows)]
    for group_id, part in grouped:
        part_labels = part[label_column].to_numpy(dtype=np.int8)
        for name in VARIANT_NAMES:
            groups.append({
                "group": str(group_id), "variant": name,
                **_score(part_labels, part[PROBABILITY_COLUMNS[name]].to_numpy(dtype=np.float64)),
            })
    return overall, groups


def _block_indices(days: pd.DatetimeIndex, replications: int, block_days: int, seed: int) -> np.ndarray:
    if len(days) == 0 or block_days <= 0 or replications <= 0:
        raise ValueError("Bootstrap calendar and dimensions must be non-empty and positive")
    block_count = int(math.ceil(len(days) / block_days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(days), size=(replications, block_count))
    offsets = np.arange(block_days, dtype=np.int32)
    draws = (starts[:, :, None] + offsets[None, None, :]) % len(days)
    return draws.reshape(replications, -1)[:, :len(days)].astype(np.int32, copy=False)


def _row_day_index(starts: pd.Series, days: pd.DatetimeIndex) -> np.ndarray:
    utc_days_ns = pd.to_datetime(starts, utc=True).dt.floor("D").to_numpy(dtype="datetime64[ns]")
    values = utc_days_ns.view(np.int64)
    first = days[0].to_datetime64().astype("datetime64[ns]").view(np.int64)
    day_index = ((values - first) // pd.Timedelta(days=1).value).astype(np.int32)
    if np.any(day_index < 0) or np.any(day_index >= len(days)):
        raise RuntimeError("A row lies outside its frozen bootstrap day grid")
    return day_index


def _weighted_auc_bootstrap(
    labels: np.ndarray,
    probability: np.ndarray,
    row_day_index: np.ndarray,
    day_counts: np.ndarray,
) -> np.ndarray:
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_labels = labels[order]
    sorted_days = row_day_index[order]
    group_starts = np.flatnonzero(np.r_[True, sorted_probability[1:] != sorted_probability[:-1]])
    group_id = np.repeat(np.arange(len(group_starts), dtype=np.int32), np.diff(np.r_[group_starts, len(order)]))
    results = np.full(len(day_counts), np.nan, dtype=np.float64)
    for row, counts in enumerate(day_counts):
        sample_weight = counts[sorted_days]
        positive = np.bincount(group_id, weights=sample_weight * sorted_labels, minlength=len(group_starts))
        negative = np.bincount(group_id, weights=sample_weight * (1 - sorted_labels), minlength=len(group_starts))
        total_positive = float(positive.sum())
        total_negative = float(negative.sum())
        if total_positive == 0 or total_negative == 0:
            continue
        negative_before = np.cumsum(negative) - negative
        numerator = np.sum(positive * (negative_before + 0.5 * negative))
        results[row] = numerator / (total_positive * total_negative)
    return results


def _paired_quality_bootstrap(rows: pd.DataFrame, label_column: str, days: pd.DatetimeIndex, draws: np.ndarray) -> dict:
    labels = rows[label_column].to_numpy(dtype=np.int8)
    row_days = _row_day_index(rows["market_start_utc"], days)
    day_counts = np.stack([np.bincount(draw, minlength=len(days)) for draw in draws], axis=0)
    row_counts = np.bincount(row_days, minlength=len(days)).astype(np.float64)
    samples = {name: {metric: np.full(len(draws), np.nan, dtype=np.float64) for metric in BOOTSTRAP_METRICS} for name in (SINGLE_NAME, "ENSEMBLE")}
    probabilities = {}
    for name in (SINGLE_NAME, "ENSEMBLE"):
        p = rows[PROBABILITY_COLUMNS[name]].to_numpy(dtype=np.float64)
        probabilities[name] = p
        clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
        losses = -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
        brier = np.square(p - labels)
        day_losses = np.bincount(row_days, weights=losses, minlength=len(days))
        day_brier = np.bincount(row_days, weights=brier, minlength=len(days))
        denominators = day_counts @ row_counts
        samples[name]["log_loss"] = (day_counts @ day_losses) / denominators
        samples[name]["brier"] = (day_counts @ day_brier) / denominators
        samples[name]["auc"] = _weighted_auc_bootstrap(labels, p, row_days, day_counts)
    output = {}
    for metric in BOOTSTRAP_METRICS:
        contrast = samples["ENSEMBLE"][metric] - samples[SINGLE_NAME][metric]
        valid = contrast[np.isfinite(contrast)]
        points = {
            name: _score(labels, probabilities[name])[metric]
            for name in (SINGLE_NAME, "ENSEMBLE")
        }
        output[metric] = {
            "point_difference_ENSEMBLE_minus_SINGLE": float(points["ENSEMBLE"] - points[SINGLE_NAME]),
            "95pct": [float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))] if len(valid) else [None, None],
            "valid_replications": int(len(valid)),
            "replications": int(len(draws)),
        }
    return {
        "method": "paired circular moving-block bootstrap over shared 3-day UTC blocks",
        "sample_rows": int(len(rows)),
        "calendar_days": int(len(days)),
        "seed": int(CONFIG["bootstrap"]["seed"]),
        "block_days": int(CONFIG["bootstrap"]["block_days"]),
        "replications": int(len(draws)),
        "contrasts": output,
    }


def _component_average_quality(overall: dict) -> dict:
    output = {}
    for metric in BOOTSTRAP_METRICS:
        values = [overall[name][metric] for name in COMPONENT_NAMES]
        output[metric] = {
            "component_mean": float(np.mean(values)),
            "ensemble": float(overall["ENSEMBLE"][metric]),
            "ensemble_minus_component_mean": float(overall["ENSEMBLE"][metric] - np.mean(values)),
        }
    return output


def _dispersion(rows: pd.DataFrame) -> dict:
    probabilities = rows[[PROBABILITY_COLUMNS[name] for name in COMPONENT_NAMES]].to_numpy(dtype=np.float64)
    dispersion = np.std(probabilities, axis=1, ddof=0)
    return {
        "rows": int(len(rows)),
        "mean_standard_deviation": float(dispersion.mean()),
        "quantiles": {
            f"p{quantile}": float(np.quantile(dispersion, quantile / 100.0))
            for quantile in (0, 25, 50, 75, 90, 95, 99, 100)
        },
    }


def _quality(manifest: dict, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage = manifest.get("stages", {}).get("quality", {})
    quality_path = OUTPUT_DIR / "quality_summary.json"
    monthly_path = OUTPUT_DIR / "official_monthly_quality.csv"
    folds_path = OUTPUT_DIR / "fold_quality.csv"
    if stage.get("status") == "completed" and _verify_records(stage.get("outputs", [])):
        return pd.read_parquet(OUTPUT_DIR / "official_quality_rows.parquet"), pd.read_csv(monthly_path), pd.read_csv(folds_path)
    started = time.perf_counter()
    _begin_stage(manifest, "quality", "Compute all-model proxy and official metrics, seed stability, correlations, and paired bootstrap.")
    info = pd.read_parquet(BASE_PREDICTIONS_PATH)
    official_filter = info["comparison_forecast_available"].astype(bool) & info["official_outcome_available_by_cutoff"].astype(bool)
    official_base = info.loc[official_filter, ["market_start_utc", "target_polymarket_up", "fold_id"]].copy()
    official_base["market_start_utc"] = pd.to_datetime(official_base["market_start_utc"], utc=True, errors="raise")
    official = official_base.merge(
        predictions[["market_start_utc", *[PROBABILITY_COLUMNS[name] for name in VARIANT_NAMES]]],
        on="market_start_utc", how="left", validate="one_to_one",
    )
    if len(official) != 20_718 or official.target_polymarket_up.nunique() != 2 or official[[PROBABILITY_COLUMNS[name] for name in VARIANT_NAMES]].isna().any().any():
        raise RuntimeError("The official quality rows do not reproduce the prior 20,718-label sample")
    source_raw = info.loc[official_filter].sort_values("market_start_utc", kind="stable")["p_raw"].to_numpy(dtype=np.float64)
    merged_raw = official.sort_values("market_start_utc", kind="stable")[PROBABILITY_COLUMNS["SINGLE"]].to_numpy(dtype=np.float64)
    if not np.allclose(source_raw, merged_raw, rtol=0, atol=1e-12):
        raise RuntimeError("Reused SINGLE does not reproduce the prior RAW BASE70 official predictions")

    proxy = predictions.copy()
    proxy["fold_id"] = proxy["fold_id"].astype(str)
    official["fold_id"] = official["fold_id"].astype(str)
    proxy_overall, proxy_fold = _quality_table(proxy, "target_preopen_5m_up_binance_index_proxy", "fold_id")
    official_overall, official_fold = _quality_table(official, "target_polymarket_up", "fold_id")
    monthly_rows = []
    official["utc_month"] = official.market_start_utc.dt.strftime("%Y-%m")
    for month, rows in official.groupby("utc_month", sort=True):
        labels = rows.target_polymarket_up.to_numpy(dtype=np.int8)
        for name in VARIANT_NAMES:
            monthly_rows.append({"utc_month": month, "variant": name, **_score(labels, rows[PROBABILITY_COLUMNS[name]].to_numpy(dtype=np.float64))})

    correlation = np.corrcoef(proxy[[PROBABILITY_COLUMNS[name] for name in COMPONENT_NAMES]].to_numpy(dtype=np.float64), rowvar=False)
    correlation_rows = [
        {"seed": left, **{right: float(correlation[i, j]) for j, right in enumerate(COMPONENT_NAMES)}}
        for i, left in enumerate(COMPONENT_NAMES)
    ]
    sample_days = pd.date_range(
        proxy.market_start_utc.min().floor("D"), proxy.market_start_utc.max().floor("D"), freq="D", tz="UTC",
    )
    official_days = pd.date_range(
        official.market_start_utc.min().floor("D"), official.market_start_utc.max().floor("D"), freq="D", tz="UTC",
    )
    draws_proxy = _block_indices(sample_days, int(CONFIG["bootstrap"]["replications"]), int(CONFIG["bootstrap"]["block_days"]), int(CONFIG["bootstrap"]["seed"]))
    draws_official = _block_indices(official_days, int(CONFIG["bootstrap"]["replications"]), int(CONFIG["bootstrap"]["block_days"]), int(CONFIG["bootstrap"]["seed"]))
    quality_bootstrap = {
        "proxy_oof": _paired_quality_bootstrap(proxy, "target_preopen_5m_up_binance_index_proxy", sample_days, draws_proxy),
        "official": _paired_quality_bootstrap(official, "target_polymarket_up", official_days, draws_official),
    }
    quality_summary = {
        "sample": {
            "proxy_oof_rows": int(len(proxy)),
            "proxy_oof_start_utc": proxy.market_start_utc.min().isoformat(),
            "proxy_oof_end_utc": proxy.market_start_utc.max().isoformat(),
            "proxy_oof_days": int(len(sample_days)),
            "official_rows": int(len(official)),
            "official_up_labels": int(official.target_polymarket_up.sum()),
            "official_down_labels": int(len(official) - official.target_polymarket_up.sum()),
            "official_days": int(len(official_days)),
        },
        "overall": {"proxy_oof": proxy_overall, "official": official_overall},
        "fold_metrics": {"proxy_oof": proxy_fold, "official": official_fold},
        "official_monthly": monthly_rows,
        "seed_prediction_correlations_proxy_oof": correlation_rows,
        "prediction_dispersion": {
            "proxy_oof": _dispersion(proxy), "official": _dispersion(official),
        },
        "ensemble_minus_mean_component_score": {
            "proxy_oof": _component_average_quality(proxy_overall),
            "official": _component_average_quality(official_overall),
        },
        "single_seed_stability": {
            "proxy_oof": _seed_ranges(proxy_overall),
            "official": _seed_ranges(official_overall),
        },
        "paired_bootstrap_ensemble_minus_single": quality_bootstrap,
        "interpretation": "Seed dispersion measures training sensitivity, not uncertainty in the true probability; it was not used to filter trades.",
    }
    official_path = OUTPUT_DIR / "official_quality_rows.parquet"
    draw_path = OUTPUT_DIR / "shared_bootstrap_day_indices.npz"
    with draw_path.with_suffix(draw_path.suffix + ".tmp").open("wb") as handle:
        np.savez_compressed(
            handle, proxy_indices=draws_proxy, proxy_days=sample_days.as_unit("ns").asi8,
            official_indices=draws_official, official_days=official_days.as_unit("ns").asi8,
        )
    draw_path.with_suffix(draw_path.suffix + ".tmp").replace(draw_path)
    _write_parquet(official, official_path)
    _write_json(quality_path, quality_summary)
    pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)
    fold_rows = [
        {"sample": sample, **row}
        for sample, rows in (("proxy_oof", proxy_fold), ("official", official_fold))
        for row in rows
    ]
    pd.DataFrame(fold_rows).to_csv(folds_path, index=False)
    pd.DataFrame(correlation_rows).to_csv(OUTPUT_DIR / "seed_prediction_correlation.csv", index=False)
    records = [_record(path) for path in (quality_path, official_path, monthly_path, folds_path, draw_path, OUTPUT_DIR / "seed_prediction_correlation.csv")]
    _finish_stage(manifest, "quality", summary=quality_summary, outputs=records, elapsed_seconds=time.perf_counter() - started)
    return official, pd.DataFrame(monthly_rows), pd.DataFrame(fold_rows)


def _seed_ranges(overall: dict) -> dict:
    result = {}
    for metric in BOOTSTRAP_METRICS:
        values = {name: float(overall[name][metric]) for name in COMPONENT_NAMES}
        result[metric] = {
            "single": values[SINGLE_NAME],
            "min": float(min(values.values())),
            "median": float(np.median(list(values.values()))),
            "max": float(max(values.values())),
            "single_within_observed_range": min(values.values()) <= values[SINGLE_NAME] <= max(values.values()),
            "by_model": values,
        }
    return result


def _monthly_economics(name: str, trades: pd.DataFrame, decisions: pd.DataFrame) -> list[dict]:
    return base70._monthly_economics(name, trades, decisions)


def _run_economic_variant(
    name: str,
    predictions: pd.DataFrame,
    market_rows: pd.DataFrame,
    period_summary: dict,
    snapshot_map: dict,
    decision_map: dict,
    manifest: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    stage_name = f"economics_{name}"
    output_dir = OUTPUT_DIR / "economics" / name
    decision_path = output_dir / "decisions.parquet"
    trade_path = output_dir / "trades.parquet"
    summary_path = output_dir / "summary.json"
    stage = manifest.get("stages", {}).get(stage_name, {})
    if stage.get("status") == "completed" and _verify_records(stage.get("outputs", [])):
        return json.loads(summary_path.read_text(encoding="utf-8")), pd.read_parquet(decision_path), pd.read_parquet(trade_path)
    started = time.perf_counter()
    _begin_stage(manifest, stage_name, f"Run the frozen independent-order replay for {name} on the same 20,735-row calendar.")
    pcol = PROBABILITY_COLUMNS[name]
    probability_map = predictions.set_index("market_start_utc")[pcol]
    rows = market_rows.copy()
    rows["p_candidate_platt"] = rows.market_start_utc.map(probability_map)
    if rows.p_candidate_platt.isna().any():
        raise RuntimeError(f"{name} does not cover every common economic calendar row")
    variant_market_map = research._snapshot_map(rows.loc[rows.condition_id.notna()])
    summary, trades, decisions = base70.replay._portfolio_for_policy(
        "fixed_5_usd", rows, snapshot_map, variant_market_map,
        decision_snapshot_by_id=decision_map,
        minimum_net_ev_fraction=float(CONFIG["economics"]["minimum_net_ev_fraction"]),
        output_policy=name,
        causal_intent_mode=True,
        allow_unresolved_outcomes=True,
        independent_order_mode=True,
    )
    equity_path = summary.pop("_equity_path")
    summary["monthly_diagnostics"] = []
    summary["accounting_audit"] = base70.replay._portfolio_accounting_audit(
        summary, trades, decisions, equity_path, snapshot_map, decision_map,
        allow_unresolved_outcomes=True, independent_order_mode=True,
    )
    decision_frame = pd.DataFrame(decisions)
    trade_frame = pd.DataFrame(trades)
    settled = trade_frame.loc[trade_frame.settlement_status.eq("settled")] if not trade_frame.empty else trade_frame
    rejected = decision_frame.loc[decision_frame.decision.eq("skip"), "skip_reason"].value_counts(dropna=True).to_dict()
    intent_values = pd.to_numeric(decision_frame.get("net_ev_fraction_at_decision", pd.Series(np.nan, index=decision_frame.index)), errors="coerce")
    intents = int(intent_values.ge(float(CONFIG["economics"]["minimum_net_ev_fraction"])).sum())
    pnl = float(pd.to_numeric(settled.net_pnl_usd, errors="coerce").sum()) if not settled.empty else 0.0
    settled_turnover = float(pd.to_numeric(settled.gross_usd, errors="coerce").sum()) if not settled.empty else 0.0
    forecast_ev = float(pd.to_numeric(settled.chosen_side_ev_usd, errors="coerce").sum()) if not settled.empty else 0.0
    summary.update({
        "variant": name,
        "decision_count": int(len(decision_frame)),
        "intent_count": intents,
        "fill_count": int(len(trade_frame)),
        "settlement_count": int(len(settled)),
        "unresolved_fill_count": int(len(trade_frame) - len(settled)),
        "gross_turnover_usd": float(pd.to_numeric(trade_frame.gross_usd, errors="coerce").sum()) if not trade_frame.empty else 0.0,
        "settled_turnover_usd": settled_turnover,
        "fees_paid_estimated_usd": float(pd.to_numeric(trade_frame.fee_usd, errors="coerce").sum()) if not trade_frame.empty else 0.0,
        "net_pnl_usd": pnl,
        "pnl_per_settled_turnover": pnl / settled_turnover if settled_turnover else None,
        "forecast_net_ev_on_settled_fills_usd": forecast_ev,
        "pnl_minus_forecast_ev_on_settled_fills_usd": pnl - forecast_ev,
        "up_fill_count": int(trade_frame.chosen_side.eq("up").sum()) if not trade_frame.empty else 0,
        "down_fill_count": int(trade_frame.chosen_side.eq("down").sum()) if not trade_frame.empty else 0,
        "up_settlement_count": int(settled.chosen_side.eq("up").sum()) if not settled.empty else 0,
        "down_settlement_count": int(settled.chosen_side.eq("down").sum()) if not settled.empty else 0,
        "rejections_by_reason": {str(key): int(value) for key, value in rejected.items()},
        "calendar": period_summary,
        "monthly_results": _monthly_economics(name, trade_frame, decision_frame),
        "elapsed_seconds": time.perf_counter() - started,
    })
    _write_parquet(decision_frame, decision_path)
    _write_parquet(trade_frame, trade_path)
    _write_json(summary_path, summary)
    records = [_record(path) for path in (decision_path, trade_path, summary_path)]
    _finish_stage(
        manifest, stage_name, summary=summary, outputs=records,
        elapsed_seconds=time.perf_counter() - started,
        decision_rows=int(len(decision_frame)), fill_rows=int(len(trade_frame)),
    )
    return summary, trade_frame, decision_frame


def _verify_single_economics(trades: pd.DataFrame, summary: dict) -> dict:
    expected = (1038, 5190.0, 267.439395238626)
    actual_fills = int(summary["fill_count"])
    actual_turnover = float(summary["gross_turnover_usd"])
    actual_pnl = float(summary["net_pnl_usd"])
    checks = {
        "fills": actual_fills == expected[0],
        "turnover": math.isclose(actual_turnover, expected[1], rel_tol=0, abs_tol=1e-8),
        "net_pnl": math.isclose(actual_pnl, expected[2], rel_tol=0, abs_tol=1e-7),
    }
    source = pd.read_parquet(BASE_RAW_TRADES_PATH)
    source = source.loc[source.settlement_status.eq("settled")].copy()
    actual = trades.loc[trades.settlement_status.eq("settled")].copy()
    keys = ["condition_id", "chosen_side", "gross_usd", "net_pnl_usd"]
    source_keys = source[keys].sort_values("condition_id", kind="stable").reset_index(drop=True)
    actual_keys = actual[keys].sort_values("condition_id", kind="stable").reset_index(drop=True)
    ledger_matches = len(source_keys) == len(actual_keys) and source_keys.equals(actual_keys)
    checks["settled_ledger_matches_previous_RAW"] = ledger_matches
    if not all(checks.values()):
        diagnostic = {
            "expected": {"fills": expected[0], "turnover_usd": expected[1], "net_pnl_usd": expected[2]},
            "actual": {"fills": actual_fills, "turnover_usd": actual_turnover, "net_pnl_usd": actual_pnl},
            "checks": checks,
            "source_ledger_rows": int(len(source_keys)),
            "new_ledger_rows": int(len(actual_keys)),
            "source_only_condition_ids": sorted(set(source_keys.condition_id) - set(actual_keys.condition_id))[:25],
            "new_only_condition_ids": sorted(set(actual_keys.condition_id) - set(source_keys.condition_id))[:25],
        }
        _write_json(OUTPUT_DIR / "single_anchor_mismatch.json", diagnostic)
        raise RuntimeError(f"SINGLE did not reproduce the frozen RAW economic anchor: {checks}")
    return {"expected_fills": expected[0], "expected_turnover_usd": expected[1], "expected_net_pnl_usd": expected[2], "actual_fills": actual_fills, "actual_turnover_usd": actual_turnover, "actual_net_pnl_usd": actual_pnl, "checks": checks, "passed": True}


def _daily_economics(trades: dict[str, pd.DataFrame], days: pd.DatetimeIndex) -> pd.DataFrame:
    daily = pd.DataFrame({"utc_day": days})
    for name, frame in trades.items():
        values = np.zeros((len(days), 4), dtype=np.float64)
        if not frame.empty:
            rows = frame.copy()
            rows["decision_time_utc"] = pd.to_datetime(rows.decision_time_utc, utc=True, errors="raise")
            rows["utc_day"] = rows.decision_time_utc.dt.floor("D")
            rows["settled"] = rows.settlement_status.eq("settled")
            day_to_index = {day: index for index, day in enumerate(days)}
            rows["day_index"] = rows.utc_day.map(day_to_index).astype(np.int32)
            settled = rows.loc[rows.settled]
            values[:, 0] = np.bincount(settled.day_index, weights=pd.to_numeric(settled.net_pnl_usd, errors="coerce"), minlength=len(days))
            values[:, 1] = np.bincount(settled.day_index, weights=pd.to_numeric(settled.gross_usd, errors="coerce"), minlength=len(days))
            values[:, 2] = np.bincount(rows.day_index, weights=pd.to_numeric(rows.gross_usd, errors="coerce"), minlength=len(days))
            values[:, 3] = np.bincount(rows.day_index, weights=pd.to_numeric(rows.fee_usd, errors="coerce"), minlength=len(days))
        for column, index in (("net_pnl_usd", 0), ("settled_turnover_usd", 1), ("filled_turnover_usd", 2), ("fees_usd", 3)):
            daily[f"{name}_{column}"] = values[:, index]
    daily["ENSEMBLE_MINUS_SINGLE_net_pnl_usd"] = daily["ENSEMBLE_net_pnl_usd"] - daily["SINGLE_net_pnl_usd"]
    return daily


def _economic_bootstrap(daily: pd.DataFrame, draws: np.ndarray) -> dict:
    n_days = len(daily)
    left_pnl = daily["ENSEMBLE_net_pnl_usd"].to_numpy(dtype=np.float64)
    right_pnl = daily["SINGLE_net_pnl_usd"].to_numpy(dtype=np.float64)
    left_turnover = daily["ENSEMBLE_settled_turnover_usd"].to_numpy(dtype=np.float64)
    right_turnover = daily["SINGLE_settled_turnover_usd"].to_numpy(dtype=np.float64)
    pnl_differences = np.empty(len(draws), dtype=np.float64)
    ratio_differences = np.full(len(draws), np.nan, dtype=np.float64)
    for index, draw in enumerate(draws):
        l_pnl = float(left_pnl[draw].sum())
        r_pnl = float(right_pnl[draw].sum())
        l_turn = float(left_turnover[draw].sum())
        r_turn = float(right_turnover[draw].sum())
        pnl_differences[index] = (l_pnl - r_pnl) / n_days
        if l_turn > 0 and r_turn > 0:
            ratio_differences[index] = l_pnl / l_turn - r_pnl / r_turn
    return {
        "method": "paired circular moving-block bootstrap over the complete UTC calendar, including days without fills",
        "replications": int(len(draws)), "block_days": int(CONFIG["bootstrap"]["block_days"]),
        "seed": int(CONFIG["bootstrap"]["seed"]), "calendar_days": n_days,
        "mean_daily_pnl_difference_usd": {
            "point_estimate": float((left_pnl - right_pnl).mean()),
            "95pct": [float(np.quantile(pnl_differences, 0.025)), float(np.quantile(pnl_differences, 0.975))],
        },
        "pnl_per_settled_turnover_difference": {
            "point_estimate": float(left_pnl.sum() / left_turnover.sum() - right_pnl.sum() / right_turnover.sum()),
            "95pct": [float(np.nanquantile(ratio_differences, 0.025)), float(np.nanquantile(ratio_differences, 0.975))],
            "valid_replications": int(np.isfinite(ratio_differences).sum()),
        },
    }


def _side_economics(trades: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for name, frame in trades.items():
        settled_frame = frame.loc[frame.settlement_status.eq("settled")] if not frame.empty else frame
        for side in ("up", "down"):
            side_all = frame.loc[frame.chosen_side.eq(side)] if not frame.empty else frame
            side_settled = settled_frame.loc[settled_frame.chosen_side.eq(side)] if not settled_frame.empty else settled_frame
            pnl = float(pd.to_numeric(side_settled.net_pnl_usd, errors="coerce").sum()) if not side_settled.empty else 0.0
            turnover = float(pd.to_numeric(side_settled.gross_usd, errors="coerce").sum()) if not side_settled.empty else 0.0
            ev = float(pd.to_numeric(side_settled.chosen_side_ev_usd, errors="coerce").sum()) if not side_settled.empty else 0.0
            rows.append({
                "variant": name, "side": side.upper(), "fills": int(len(side_all)),
                "settlements": int(len(side_settled)),
                "turnover_usd": float(pd.to_numeric(side_all.gross_usd, errors="coerce").sum()) if not side_all.empty else 0.0,
                "fees_usd": float(pd.to_numeric(side_all.fee_usd, errors="coerce").sum()) if not side_all.empty else 0.0,
                "net_pnl_usd": pnl, "pnl_per_settled_turnover": pnl / turnover if turnover else None,
                "forecast_ev_usd": ev, "pnl_minus_ev_usd": pnl - ev,
            })
    return rows


def _concentration_by_variant(daily: pd.DataFrame) -> dict:
    output = {}
    for name in VARIANT_NAMES:
        column = f"{name}_net_pnl_usd"
        frame = daily[["utc_day", column]].sort_values(column, ascending=False, kind="stable")
        total = float(frame[column].sum())
        best = frame.head(5)
        best_sum = float(best[column].sum())
        output[name] = {
            "top_five_days": [
                {"utc_day": row.utc_day.isoformat(), "net_pnl_usd": float(getattr(row, column))}
                for row in best.itertuples(index=False)
            ],
            "top_five_pnl_usd": best_sum,
            "top_five_share_of_total_pnl": best_sum / total if total != 0 else None,
            "remaining_days_pnl_usd": total - best_sum,
            "total_pnl_usd": total,
        }
    return output


def _economic_summary(manifest: dict, predictions: pd.DataFrame) -> dict:
    stage = manifest.get("stages", {}).get("economics_summary", {})
    summary_path = OUTPUT_DIR / "economic_summary.json"
    daily_path = OUTPUT_DIR / "daily_pnl.csv"
    monthly_path = OUTPUT_DIR / "monthly_economics.csv"
    decomposition_path = OUTPUT_DIR / "pnl_decomposition.json"
    if stage.get("status") == "completed" and _verify_records(stage.get("outputs", [])):
        return stage["summary"]
    started = time.perf_counter()
    _begin_stage(manifest, "execution_inputs", "Load the frozen T-60/T-59 snapshots once and prepare independent-order replay.")
    model_update.fee_correction._configure_output_paths()
    correction_manifest = research._read_json(model_update.fee_correction.CORRECTION_DIR / "research_manifest.json")
    with taker.PeakMemorySampler() as snapshot_memory:
        t59, t60, book_record = research._load_enriched_book_snapshots(correction_manifest)
    snapshot_peak = round(snapshot_memory.peak_bytes / 2**30, 3)
    frozen_snapshots = manifest["frozen_comparison"]["specification"]["economics"]["snapshot_identity"]
    if book_record["t59_coverage_sha256"] != frozen_snapshots["t59_sha256"] or book_record["t60_coverage_sha256"] != frozen_snapshots["t60_sha256"]:
        raise RuntimeError("Loaded T-60/T-59 snapshots differ from the pre-training frozen identity")
    execution_calendar = pd.read_parquet(EXECUTION_CALENDAR_PATH)
    if len(execution_calendar) != 20_735 or execution_calendar.market_start_utc.duplicated().any():
        raise RuntimeError("The saved common execution calendar is not the exact 20,735-row source calendar")
    execution_calendar["market_start_utc"] = pd.to_datetime(execution_calendar.market_start_utc, utc=True, errors="raise")
    market_rows, period_summary = research._asof_market_rows(execution_calendar, base70.START_UTC, base70.CUTOFF_UTC)
    if len(market_rows) != 20_735:
        raise RuntimeError("The frozen economic comparison calendar changed while preparing replay")
    snapshot_map = research._snapshot_map(t59)
    decision_map = research._snapshot_map(t60)
    _finish_stage(
        manifest, "execution_inputs",
        summary={"calendar_rows": int(len(market_rows)), "decision_snapshot_rows": int(execution_calendar.t60_decision_snapshot_available.sum()), "execution_snapshot_rows": int(execution_calendar.t59_snapshot_available.sum()), "snapshot_identity": frozen_snapshots, "peak_process_working_set_gib": snapshot_peak},
        outputs=[_record(EXECUTION_CALENDAR_PATH)], elapsed_seconds=time.perf_counter() - started,
        peak_process_working_set_gib=snapshot_peak,
    )

    summaries = {}
    trades = {}
    decisions = {}
    economics_started = time.perf_counter()
    for name in VARIANT_NAMES:
        summary, trade_frame, decision_frame = _run_economic_variant(
            name, predictions, market_rows, period_summary, snapshot_map, decision_map, manifest,
        )
        summaries[name] = summary
        trades[name] = trade_frame
        decisions[name] = decision_frame
        if name == SINGLE_NAME:
            anchor = _verify_single_economics(trade_frame, summary)
            manifest["single_anchor_reproduction"] = anchor
            _write_manifest(manifest)
    for name in VARIANT_NAMES:
        if len(decisions[name]) != 20_735 or not base70.summary_accounting_passed(summaries[name]):
            raise RuntimeError(f"{name} replay failed the full-calendar or accounting audit")

    days = pd.date_range(
        execution_calendar.market_start_utc.min().floor("D"),
        execution_calendar.market_start_utc.max().floor("D"),
        freq="D", tz="UTC",
    )
    daily = _daily_economics(trades, days)
    daily_draws = _block_indices(days, int(CONFIG["bootstrap"]["replications"]), int(CONFIG["bootstrap"]["block_days"]), int(CONFIG["bootstrap"]["seed"]))
    bootstrap = _economic_bootstrap(daily, daily_draws)
    decompositions = {
        "ENSEMBLE_MINUS_SINGLE": base70._decompose_pnl(trades["ENSEMBLE"], trades["SINGLE"], "ENSEMBLE", "SINGLE"),
    }
    concentration = _concentration_by_variant(daily)
    economic_average = {}
    for metric, key in (("net_pnl_usd", "net_pnl_usd"), ("pnl_per_settled_turnover", "pnl_per_settled_turnover"), ("fill_count", "fill_count")):
        component_average = float(np.mean([summaries[name][key] for name in COMPONENT_NAMES]))
        ensemble_value = float(summaries["ENSEMBLE"][key])
        economic_average[metric] = {"component_mean": component_average, "ensemble": ensemble_value, "ensemble_minus_component_mean": ensemble_value - component_average}
    seed_economics = {}
    for metric in ("fill_count", "gross_turnover_usd", "fees_paid_estimated_usd", "net_pnl_usd", "pnl_per_settled_turnover", "forecast_net_ev_on_settled_fills_usd"):
        values = {name: summaries[name][metric] for name in COMPONENT_NAMES}
        seed_economics[metric] = {"min": float(min(values.values())), "median": float(np.median(list(values.values()))), "max": float(max(values.values())), "single": values["SINGLE"], "by_model": values}
    monthly = [row for name in VARIANT_NAMES for row in summaries[name]["monthly_results"]]
    sides = _side_economics(trades)
    output = {
        "variants": summaries,
        "availability": {"decision_rows": 20_735, "calendar_days": int(len(days)), "first_market_start_utc": execution_calendar.market_start_utc.min().isoformat(), "last_market_start_utc": execution_calendar.market_start_utc.max().isoformat()},
        "bootstrap_ensemble_minus_single": bootstrap,
        "concentration": concentration,
        "pnl_decomposition": decompositions,
        "up_down_economics": sides,
        "ensemble_minus_mean_component_economics": economic_average,
        "single_seed_stability": seed_economics,
        "single_anchor_reproduction": manifest["single_anchor_reproduction"],
        "accounting_audits_passed": True,
        "daily_calendar_rows": int(len(daily)),
        "elapsed_seconds": time.perf_counter() - economics_started,
    }
    if not decompositions["ENSEMBLE_MINUS_SINGLE"]["sum_reconciles"]:
        raise RuntimeError("ENSEMBLE-SINGLE PnL components do not reconcile")
    _write_json(summary_path, output)
    daily.to_csv(daily_path, index=False)
    pd.DataFrame(monthly).to_csv(monthly_path, index=False)
    _write_json(decomposition_path, decompositions)
    variant_records = [
        record
        for name in VARIANT_NAMES
        for record in manifest.get("stages", {}).get(f"economics_{name}", {}).get("outputs", [])
    ]
    records = [*variant_records, *[_record(path) for path in (summary_path, daily_path, monthly_path, decomposition_path)]]
    _finish_stage(manifest, "economics_summary", summary=output, outputs=records, elapsed_seconds=output["elapsed_seconds"])
    return output


def _markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _fmt(value, digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _report(manifest: dict) -> None:
    quality = manifest["stages"]["quality"]["summary"]
    economics = manifest["stages"]["economics_summary"]["summary"]
    overall_rows = []
    for sample, values in quality["overall"].items():
        for name in VARIANT_NAMES:
            score = values[name]
            overall_rows.append([sample, name, score["rows"], _fmt(score["log_loss"]), _fmt(score["brier"]), _fmt(score["auc"], 4)])
    fold_rows = [
        [sample, row["group"], row["variant"], row["rows"], _fmt(row["log_loss"]), _fmt(row["brier"]), _fmt(row["auc"], 4)]
        for sample, metrics in quality["fold_metrics"].items()
        for row in metrics
    ]
    month_rows = [[row["utc_month"], row["variant"], row["rows"], _fmt(row["log_loss"]), _fmt(row["brier"]), _fmt(row["auc"], 4)] for row in quality["official_monthly"]]
    econ_rows = []
    for name in VARIANT_NAMES:
        row = economics["variants"][name]
        econ_rows.append([
            name, row["intent_count"], row["fill_count"], row["settlement_count"],
            f"${row['gross_turnover_usd']:,.2f}", f"${row['fees_paid_estimated_usd']:,.2f}",
            f"${row['net_pnl_usd']:,.2f}", _fmt(row["pnl_per_settled_turnover"] * 100, 2) + "%",
            f"${row['forecast_net_ev_on_settled_fills_usd']:,.2f}", f"${row['pnl_minus_forecast_ev_on_settled_fills_usd']:,.2f}",
        ])
    monthly_rows = [[row["utc_month"], row["variant"], row["fills"], f"${row['gross_turnover_usd']:,.2f}", f"${row['fees_usd']:,.2f}", f"${row['net_pnl_usd']:,.2f}", _fmt(row["pnl_per_settled_turnover"] * 100 if row["pnl_per_settled_turnover"] is not None else None, 2) + "%", f"${row['forecast_ev_on_settled_fills_usd']:,.2f}", f"${row['pnl_minus_forecast_ev_on_settled_fills_usd']:,.2f}"] for row in [entry for name in VARIANT_NAMES for entry in economics["variants"][name]["monthly_results"]]]
    side_rows = [[row["variant"], row["side"], row["fills"], row["settlements"], f"${row['turnover_usd']:,.2f}", f"${row['fees_usd']:,.2f}", f"${row['net_pnl_usd']:,.2f}", f"${row['forecast_ev_usd']:,.2f}", f"${row['pnl_minus_ev_usd']:,.2f}"] for row in economics["up_down_economics"]]
    top_rows = [[name, f"${value['top_five_pnl_usd']:,.2f}", _fmt(value["top_five_share_of_total_pnl"] * 100 if value["top_five_share_of_total_pnl"] is not None else None, 1) + "%", f"${value['remaining_days_pnl_usd']:,.2f}"] for name, value in economics["concentration"].items()]
    corr_rows = [[row["seed"], *[_fmt(row[name], 5) for name in COMPONENT_NAMES]] for row in quality["seed_prediction_correlations_proxy_oof"]]
    dispersion_rows = []
    for sample, result in quality["prediction_dispersion"].items():
        dispersion_rows.append([sample, result["rows"], _fmt(result["mean_standard_deviation"], 6), *[_fmt(result["quantiles"][f"p{q}"], 6) for q in (25, 50, 75, 90, 95, 99)]])
    stability_rows = []
    for sample, results in quality["single_seed_stability"].items():
        for metric, value in results.items():
            stability_rows.append([sample, metric, _fmt(value["single"]), _fmt(value["min"]), _fmt(value["median"]), _fmt(value["max"])])
    contrast_rows = []
    for sample, result in quality["paired_bootstrap_ensemble_minus_single"].items():
        for metric, ci in result["contrasts"].items():
            contrast_rows.append([sample, metric, _fmt(ci["point_difference_ENSEMBLE_minus_SINGLE"]), f"{_fmt(ci['95pct'][0])} to {_fmt(ci['95pct'][1])}", ci["valid_replications"]])
    component_gap_rows = []
    for sample, results in quality["ensemble_minus_mean_component_score"].items():
        for metric, value in results.items():
            component_gap_rows.append([sample, metric, _fmt(value["component_mean"]), _fmt(value["ensemble"]), _fmt(value["ensemble_minus_component_mean"])])

    quality_official = quality["overall"]["official"]
    quality_ci = quality["paired_bootstrap_ensemble_minus_single"]["official"]["contrasts"]
    econ_ci = economics["bootstrap_ensemble_minus_single"]
    probability_improvement = (
        quality_official["ENSEMBLE"]["log_loss"] < quality_official["SINGLE"]["log_loss"]
        and quality_official["ENSEMBLE"]["brier"] < quality_official["SINGLE"]["brier"]
        and quality_official["ENSEMBLE"]["auc"] >= quality_official["SINGLE"]["auc"]
        and quality_ci["log_loss"]["95pct"][1] < 0
        and quality_ci["brier"]["95pct"][1] < 0
    )
    economic_improvement = (
        economics["variants"]["ENSEMBLE"]["net_pnl_usd"] > economics["variants"]["SINGLE"]["net_pnl_usd"]
        and economics["variants"]["ENSEMBLE"]["pnl_per_settled_turnover"] > economics["variants"]["SINGLE"]["pnl_per_settled_turnover"]
        and econ_ci["mean_daily_pnl_difference_usd"]["95pct"][0] > 0
        and econ_ci["pnl_per_settled_turnover_difference"]["95pct"][0] > 0
    )
    conclusion = "Develop ENSEMBLE as the next research candidate" if probability_improvement and economic_improvement else "Keep SINGLE as the research baseline"
    quality_verdict = "Official log loss and Brier improve with intervals excluding zero, while AUC does not fall." if probability_improvement else "The official sample does not show a clear, interval-supported improvement across probability scores."
    economics_verdict = "ENSEMBLE improves both realized PnL and PnL/turnover with paired intervals above zero." if economic_improvement else "The after-cost evidence does not clear a paired, directional improvement for both daily PnL and PnL/turnover."

    content = f"""# RAW BASE70 seed stability and five-model ensemble - 2026-10-18

## Conclusion

**{conclusion}.**

- Probability quality: {quality_verdict}
- After-cost economics: {economics_verdict}
- SINGLE was retained exactly from its original 20261015 OOF and fold boosters. The four new seeds used 31 rounds on the same ten folds, common rows, target, sample weights, purge, and feature order.
- The ensemble is the arithmetic mean of the five saved raw probabilities. It uses no calibration, learned weights, seed selection, or prediction-dispersion filter.
- This is conditional on the previously selected features, hyperparameters, and 31 rounds; it is not independent validation of the whole model-selection process. Historical development outcomes have been examined previously.

## Frozen comparison and seed behavior

- Shared BASE70 matrix: {manifest['frozen_comparison']['specification']['data']['rows']:,} rows, 70 features, ten saved chronological folds. All 40 new fits were exactly 31 iterations with no early stopping.
- Target: `target_preopen_5m_up_binance_index_proxy`; decision-time validation rows and label-availability purge match the source folds. Sample weights hash: `{manifest['frozen_comparison']['specification']['data']['weights_sha256']}`.
- Original SINGLE parameters and boosters were reused. For new fits, `feature_fraction_seed` controls per-tree feature subsampling; `data_random_seed` controls the dataset's histogram-bin sampling and the dataset was rebuilt once per new seed with that seed. `bagging_seed` is set but inactive because bagging remains off (`bagging_fraction=1`, `bagging_freq=0`). The explicit child seeds override the root `seed` for their respective sources. `extra_seed`, `drop_seed`, and `objective_seed` remain inactive under the unchanged gbdt/binary settings. These LightGBM 4.6.0 definitions are documented in the [official parameter reference](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html).
- The full effective fit parameters for SINGLE and each new seed were frozen in the versioned manifest before training. The source control was not retrained to normalize parameters.

### Single-seed quality range

The observed seed range is sensitivity to training randomness, not a confidence interval for true probabilities.

{_markdown_table(['Sample', 'Metric', 'SINGLE', 'Min', 'Median', 'Max'], stability_rows)}

### Probability quality: overall

Lower log loss and Brier are better; higher AUC is better. The official sample exactly matches the prior 20,718 labeled rows. Proxy OOF contains every common validation row across all ten folds.

{_markdown_table(['Sample', 'Model', 'Rows', 'Log loss', 'Brier', 'AUC'], overall_rows)}

### Probability quality: fold metrics

The table contains every model's log loss, Brier, and AUC for each non-empty fold in the proxy and official samples. The complete machine-readable table is saved locally in `fold_quality.csv`.

{_markdown_table(['Sample', 'Fold', 'Model', 'Rows', 'Log loss', 'Brier', 'AUC'], fold_rows)}

### Official monthly quality

{_markdown_table(['UTC month', 'Model', 'Rows', 'Log loss', 'Brier', 'AUC'], month_rows)}

### Paired quality bootstrap: ENSEMBLE minus SINGLE

Negative log loss/Brier differences favor ENSEMBLE; positive AUC differences favor ENSEMBLE. Draws use shared circular 3-day UTC blocks, 2,000 replications, seed {CONFIG['bootstrap']['seed']}.

{_markdown_table(['Sample', 'Metric', 'Point difference', '95% interval', 'Valid draws'], contrast_rows)}

### Seed predictions and dispersion

Pearson correlations use all common proxy OOF rows.

{_markdown_table(['Seed', *COMPONENT_NAMES], corr_rows)}

{_markdown_table(['Sample', 'Rows', 'Mean SD', 'SD p25', 'SD p50', 'SD p75', 'SD p90', 'SD p95', 'SD p99'], dispersion_rows)}

The standard deviation is across the five raw component probabilities for each market. Ensemble quality relative to the arithmetic mean of the five component scores:

{_markdown_table(['Sample', 'Metric', 'Mean component score', 'Ensemble score', 'Ensemble minus component mean'], component_gap_rows)}

## Independent-order economics

The same 20,735 decisions over 72 UTC days were replayed after the prior Platt warm-up. Each model independently selected its side and orders at the same $5 gross amount, 1% net-EV hurdle, T-60 decision and T-59 execution, with the same snapshots, fees, depth, outcome cutoff, and accounting engine. Raw predictions were used without Platt. No common-fill filter was applied; a lower trade count alone is not treated as an improvement.

SINGLE reproduced the previous RAW anchor: {economics['single_anchor_reproduction']['actual_fills']:,} fills, ${economics['single_anchor_reproduction']['actual_turnover_usd']:,.2f} turnover, and ${economics['single_anchor_reproduction']['actual_net_pnl_usd']:,.2f} net PnL. All six accounting audits passed.

{_markdown_table(['Variant', 'Intents', 'Fills', 'Settled', 'Turnover', 'Fees', 'Net PnL', 'PnL/turnover', 'EV on settled fills', 'PnL minus EV'], econ_rows)}

### Monthly economics

{_markdown_table(['UTC month', 'Variant', 'Fills', 'Turnover', 'Fees', 'Net PnL', 'PnL/turnover', 'EV', 'PnL minus EV'], monthly_rows)}

### UP and DOWN outcomes

{_markdown_table(['Variant', 'Side', 'Fills', 'Settled', 'Turnover', 'Fees', 'Net PnL', 'EV', 'PnL minus EV'], side_rows)}

### Concentration and paired economic uncertainty

{_markdown_table(['Variant', 'Top-five-day PnL', 'Share of total PnL', 'Other days PnL'], top_rows)}

- ENSEMBLE minus SINGLE mean daily PnL: ${econ_ci['mean_daily_pnl_difference_usd']['point_estimate']:,.4f}; 95% paired interval ${econ_ci['mean_daily_pnl_difference_usd']['95pct'][0]:,.4f} to ${econ_ci['mean_daily_pnl_difference_usd']['95pct'][1]:,.4f}.
- ENSEMBLE minus SINGLE PnL/settled-turnover: {econ_ci['pnl_per_settled_turnover_difference']['point_estimate']:.2%}; 95% paired interval {econ_ci['pnl_per_settled_turnover_difference']['95pct'][0]:.2%} to {econ_ci['pnl_per_settled_turnover_difference']['95pct'][1]:.2%} ({econ_ci['pnl_per_settled_turnover_difference']['valid_replications']}/{econ_ci['replications']} valid draws).
- PnL difference decomposition: same-side fills ${economics['pnl_decomposition']['ENSEMBLE_MINUS_SINGLE']['components_usd']['settled_fills_same_side']:,.2f}; fills present in one variant ${economics['pnl_decomposition']['ENSEMBLE_MINUS_SINGLE']['components_usd']['settled_fills_only_one_variant']:,.2f}; opposite sides ${economics['pnl_decomposition']['ENSEMBLE_MINUS_SINGLE']['components_usd']['settled_fills_opposite_sides']:,.2f}. Components reconcile to the full settled PnL difference.

## Runtime, artifacts, and limits

- Elapsed seconds by stage: `{json.dumps({name: round(value.get('elapsed_seconds', 0.0), 2) for name, value in manifest['stages'].items()}, sort_keys=True)}`.
- Peak process working set: {manifest.get('peak_run_working_set_gib', 'not recorded')} GiB. New-fit cumulative seconds: {manifest['stages']['training']['summary']['fit_seconds']:.1f}; quality/bootstrap: {manifest['stages']['quality']['elapsed_seconds']:.1f}; replay: {manifest['stages']['economics_summary']['elapsed_seconds']:.1f}.
- Matrix and folds were loaded once for this training pass. Per-fold models, OOF checkpoints, trades and decisions are kept locally under `{CONFIG['output_directory']}`; only the compact config, report, versioned manifest, runner and tests are versioned.
- The development sample and feature/model settings were selected previously. These results do not establish future or live performance. No holdout was scored, no deployment refit was run, and no live artifacts were changed.

The runner is `python {CONFIG['runner']}`; no command-line arguments are required.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(REPORT_PATH)


def run() -> dict:
    started = time.perf_counter()
    flow_manifest, info_manifest, cv_summary, common_summary, source = _source_inputs()
    manifest = _create_or_resume_manifest(source, flow_manifest, info_manifest, cv_summary, common_summary)
    if manifest.get("status") == "completed" and all(
        stage.get("status") == "completed" and _verify_records(stage.get("outputs", []))
        for stage in manifest.get("stages", {}).values()
    ):
        return manifest
    pa.set_cpu_count(8)
    pa.set_io_thread_count(8)
    resource_before = taker._resource_snapshot()
    manifest["resource_snapshot_before_training"] = resource_before
    expected_seconds = float(flow_manifest["stages"]["cv_B0"]["elapsed_seconds"]) * len(CONFIG["new_seeds"])
    manifest["training_runtime_estimate_seconds"] = expected_seconds
    _write_manifest(manifest)
    print(
        f"Frozen run: 40 fits x {CONFIG['boosting_iterations']} rounds; "
        f"source CV measured {flow_manifest['stages']['cv_B0']['elapsed_seconds']:.1f}s/10 folds, "
        f"initial estimate {expected_seconds:.1f}s plus quality and replay. "
        f"Available RAM: {resource_before.get('available_memory_gib', 'unknown')} GiB."
    )
    try:
        manifest.pop("failure", None)
        frame = x = target = weights = folds = None
        source_stage = manifest.get("stages", {}).get("source_verification", {})
        source_verified = source_stage.get("status") == "completed" and _verify_records(source_stage.get("outputs", []))
        if not source_verified:
            _begin_stage(manifest, "source_verification", "Load the common matrix once, verify the unchanged SINGLE OOF, folds, purge, and weights.")
            verify_started = time.perf_counter()
            with taker.PeakMemorySampler() as verification_memory:
                frame, x, target, weights, folds, single_check = _load_training_inputs(cv_summary, common_summary, verify_single=True)
            verification = {
                "rows": int(len(frame)), "features": int(x.shape[1]), "fold_count": len(folds),
                "single_control": single_check,
                "weights_sha256": taker._array_sha256(weights, dtype=np.float32),
                "purge_verified": True,
                "matrix_loaded_once": True,
                "peak_process_working_set_gib": round(verification_memory.peak_bytes / 2**30, 3),
            }
            verification_path = OUTPUT_DIR / "source_verification.json"
            _write_json(verification_path, verification)
            _finish_stage(manifest, "source_verification", summary=verification, outputs=[_record(verification_path)], elapsed_seconds=time.perf_counter() - verify_started)
        training_stage = manifest.get("stages", {}).get("training", {})
        if folds is None:
            with np.load(COMMON_FOLDS_PATH) as saved:
                folds = [(saved[f"train_{i}"].astype(np.int32, copy=False), saved[f"valid_{i}"].astype(np.int32, copy=False)) for i in range(10)]
        feature_names = list(cv_summary["feature_order"])
        checkpoints_valid = all(
            _completed_fold_valid(manifest, int(seed), fold_id, None, valid_idx, feature_names)
            for seed in CONFIG["new_seeds"] for fold_id, (_, valid_idx) in enumerate(folds)
        )
        training_outputs_valid = training_stage.get("status") == "completed" and _verify_records(training_stage.get("outputs", []))
        if not checkpoints_valid or not training_outputs_valid:
            if not checkpoints_valid and x is None:
                frame, x, target, weights, folds, _ = _load_training_inputs(cv_summary, common_summary, verify_single=False)
            _train_new_models(manifest, frame, x, target, weights, folds, cv_summary["params"])
        if frame is not None:
            del frame, x, target, weights, folds
            gc.collect()

        predictions = _build_predictions(manifest)
        _quality(manifest, predictions)
        economics = _economic_summary(manifest, predictions)
        stage_peak_values = [
            stage.get("peak_process_working_set_gib", stage.get("summary", {}).get("peak_process_working_set_gib"))
            for stage in manifest["stages"].values()
        ]
        manifest["peak_run_working_set_gib"] = max(
            [float(value) for value in stage_peak_values if value is not None] + [0.0],
        )
        manifest["status"] = "completed"
        manifest["active_stage"] = "report"
        manifest["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        manifest["next_step"] = "Experiment complete; do not add seeds or run a deployment refit automatically."
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["resource_snapshot_at_completion"] = taker._resource_snapshot()
        _write_manifest(manifest)
        report_started = time.perf_counter()
        _report(manifest)
        _finish_stage(manifest, "report", outputs=[_record(REPORT_PATH)], elapsed_seconds=time.perf_counter() - report_started)
        manifest["active_stage"] = None
        manifest["elapsed_seconds"] = time.perf_counter() - started
        _write_manifest(manifest)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "stage": manifest.get("active_stage"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        manifest["active_stage"] = None
        manifest["next_step"] = "Inspect the recorded failure and resume only after correcting its cause."
        _write_manifest(manifest)
        raise


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "status": result["status"],
        "report": _relative(REPORT_PATH),
        "quality_rows": result.get("stages", {}).get("quality", {}).get("summary", {}).get("sample", {}),
        "single_anchor": result.get("single_anchor_reproduction"),
        "ensemble_minus_single_pnl": result.get("stages", {}).get("economics_summary", {}).get("summary", {}).get("bootstrap_ensemble_minus_single", {}).get("mean_daily_pnl_difference_usd"),
        "peak_run_working_set_gib": result.get("peak_run_working_set_gib"),
    }, indent=2))
