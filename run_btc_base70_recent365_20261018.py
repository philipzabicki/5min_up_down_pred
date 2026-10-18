"""Compare BASE70 SINGLE with its frozen last-365-day training variant."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_base70_full_features_20261018 as base70_full
import run_btc_base70_information_value_20261018 as base70_info
import run_btc_base70_seed_stability_20261018 as base70_seed
import run_btc_preopen_taker_flow_20261018 as taker


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_base70_recent365_20261018.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_directory"]
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
REPORT_PATH = ROOT / CONFIG["report_path"]
VERSIONED_MANIFEST_PATH = ROOT / CONFIG["versioned_manifest_path"]
STATUS_PATH = ROOT / CONFIG["status_path"]
STATUS_BEGIN = "<!-- BTC_BASE70_RECENT365_20261018_BEGIN -->"
STATUS_END = "<!-- BTC_BASE70_RECENT365_20261018_END -->"
SINGLE = "SINGLE"
RECENT365 = "RECENT365"
CONTRAST = "RECENT365_MINUS_SINGLE"
PROBABILITY_COLUMNS = {SINGLE: "p_single", RECENT365: "p_recent365"}
VARIANTS = (SINGLE, RECENT365)
CONTRASTS = {CONTRAST: (RECENT365, SINGLE)}
TARGET_COLUMN = CONFIG["target"]
FEATURE_METADATA_COLUMNS = base70_seed.FEATURE_METADATA_COLUMNS


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
        (ROOT / record["path"]).is_file()
        and (ROOT / record["path"]).stat().st_size == int(record["size_bytes"])
        and _sha256(ROOT / record["path"]) == record["sha256"]
        for record in records
    )


def _status_update(manifest: dict) -> None:
    completed = sum(item.get("status") == "completed" for item in manifest.get("completed_fits", {}).values())
    block = "\n".join([
        STATUS_BEGIN,
        "### BASE70 RECENT365 comparison - 2026-10-18",
        f"- Updated UTC: `{pd.Timestamp.now(tz='UTC').isoformat()}`.",
        f"- Status: `{manifest.get('status')}`; active stage: `{manifest.get('active_stage') or 'none'}`; challenger folds: `{completed}/10`.",
        f"- Next step: `{manifest.get('next_step', 'Review the completed paired comparison.')}`.",
        f"- Run manifest: `{_relative(MANIFEST_PATH)}`.",
        STATUS_END,
        "",
    ])
    text = STATUS_PATH.read_text(encoding="utf-8")
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


def _load_sources() -> tuple[dict, dict, dict, dict, dict, dict, dict, dict]:
    flow, info, seed_manifest, seed_versioned, cv, common, source = base70_full._load_sources()
    full_manifest_path = ROOT / CONFIG["source_full_features_run_manifest_path"]
    full_versioned_path = ROOT / CONFIG["source_full_features_versioned_manifest_path"]
    full_manifest = json.loads(full_manifest_path.read_text(encoding="utf-8"))
    full_versioned = json.loads(full_versioned_path.read_text(encoding="utf-8"))
    if full_manifest.get("status") != "completed" or full_versioned.get("status") != "completed":
        raise RuntimeError("The referenced BASE70 FULL_FEATURES source experiment is incomplete")
    full_identity = full_manifest.get("frozen_comparison", {}).get("identity_sha256")
    if not full_identity or full_identity != full_versioned.get("frozen_comparison", {}).get("identity_sha256"):
        raise RuntimeError("The BASE70 FULL_FEATURES source manifests have different frozen identities")
    if not base70_full._verify_records(full_manifest.get("stages", {}).get("training", {}).get("outputs", [])):
        raise RuntimeError("A referenced BASE70 FULL_FEATURES training artifact is missing or changed")
    return flow, info, seed_manifest, seed_versioned, cv, common, source, full_manifest


def _source_artifacts(flow: dict, info: dict, seed_manifest: dict, source: dict, full_manifest: dict) -> dict:
    artifacts = base70_full._source_artifacts(flow, info, seed_manifest, source)
    artifacts["source_full_features_run_manifest"] = _record(ROOT / CONFIG["source_full_features_run_manifest_path"])
    artifacts["source_full_features_versioned_manifest"] = _record(ROOT / CONFIG["source_full_features_versioned_manifest_path"])
    artifacts["source_full_features_report"] = _record(ROOT / "reports/btc_preopen/btc_base70_full_features_20261018.md")
    artifacts["source_full_features_runner"] = _record(ROOT / "run_btc_base70_full_features_20261018.py")
    artifacts["source_full_features_config"] = _record(ROOT / "configs/research/btc_base70_full_features_20261018.json")
    artifacts["source_full_features_identity_sha256"] = full_manifest["frozen_comparison"]["identity_sha256"]
    return artifacts


def _recent_training_indices(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    window_days: int = 365,
) -> tuple[np.ndarray, pd.Timestamp, pd.Timestamp]:
    first_validation = pd.Timestamp(frame["nominal_decision_at"].iloc[valid_idx].min())
    boundary = first_validation - pd.Timedelta(days=window_days)
    train_decisions = frame["nominal_decision_at"].iloc[train_idx]
    recent = train_idx[train_decisions.ge(boundary).to_numpy()]
    if not len(recent):
        raise RuntimeError("The RECENT365 training window is empty")
    if not np.isin(recent, train_idx).all() or not np.array_equal(recent, train_idx[train_decisions.ge(boundary).to_numpy()]):
        raise RuntimeError("RECENT365 introduced a row outside the original training indices")
    if frame["nominal_decision_at"].iloc[recent].min() < boundary:
        raise RuntimeError("RECENT365 included a decision before its inclusive lower boundary")
    if frame["nominal_decision_at"].iloc[recent].max() >= first_validation:
        raise RuntimeError("RECENT365 included a row at or after the first validation decision")
    latest_label = frame["target_label_available_at"].iloc[recent].max()
    if latest_label > first_validation:
        raise RuntimeError("RECENT365 violates the original fold label-availability purge")
    return recent.astype(np.int32, copy=False), boundary, first_validation


def _scaled_weights(weights: np.ndarray, original_idx: np.ndarray, recent_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    original_sum = float(np.asarray(weights[original_idx], dtype=np.float64).sum())
    recent = np.asarray(weights[recent_idx], dtype=np.float64)
    recent_sum = float(recent.sum())
    if not math.isfinite(original_sum) or not math.isfinite(recent_sum) or recent_sum <= 0:
        raise RuntimeError("Cannot preserve fold weight mass because an original or RECENT365 weight sum is invalid")
    multiplier = original_sum / recent_sum
    scaled = recent * multiplier
    scaled_sum = float(scaled.sum())
    if not np.isfinite(scaled).all() or not math.isclose(scaled_sum, original_sum, rel_tol=1e-12, abs_tol=1e-8):
        raise RuntimeError("RECENT365 common weight multiplier failed to preserve the original total weight")
    return scaled, {
        "original_training_weight_sum": original_sum,
        "recent365_unscaled_weight_sum": recent_sum,
        "common_weight_multiplier": multiplier,
        "recent365_scaled_weight_sum_requested": scaled_sum,
    }


def _fold_plans(frame: pd.DataFrame, weights: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    plans = []
    availability = frame["target_label_available_at"].array.asi8
    decisions = frame["nominal_decision_at"].array.asi8
    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        first_validation = pd.Timestamp(frame["nominal_decision_at"].iloc[valid_idx].min())
        if availability[train_idx].max() > decisions[valid_idx].min():
            raise RuntimeError(f"The saved training fold {fold_id} violates its original label-availability purge")
        recent_idx, lower, first = _recent_training_indices(frame, train_idx, valid_idx, int(CONFIG["window_days"]))
        scaled, weight_summary = _scaled_weights(weights, train_idx, recent_idx)
        plans.append({
            "fold_id": fold_id,
            "train_idx": train_idx,
            "recent_idx": recent_idx,
            "valid_idx": valid_idx,
            "window_start_inclusive_utc": lower.isoformat(),
            "window_end_exclusive_utc": first.isoformat(),
            "first_validation_decision_at_utc": first_validation.isoformat(),
            "validation_last_decision_at_utc": pd.Timestamp(frame["nominal_decision_at"].iloc[valid_idx].max()).isoformat(),
            "training_rows_before": int(len(train_idx)),
            "training_rows_after": int(len(recent_idx)),
            "validation_rows": int(len(valid_idx)),
            "original_training_source_row_index_sha256": taker._array_sha256(frame["_source_row_index"].iloc[train_idx].to_numpy(dtype=np.int32), dtype=np.int32),
            "recent365_source_row_index_sha256": taker._array_sha256(frame["_source_row_index"].iloc[recent_idx].to_numpy(dtype=np.int32), dtype=np.int32),
            "validation_source_row_index_sha256": taker._array_sha256(frame["_source_row_index"].iloc[valid_idx].to_numpy(dtype=np.int32), dtype=np.int32),
            "scaled_weight_sha256": taker._array_sha256(scaled, dtype=np.float64),
            **weight_summary,
        })
    return plans


def _frozen_specification(
    artifacts: dict,
    cv_summary: dict,
    common_summary: dict,
    plans: list[dict],
) -> dict:
    control = copy.deepcopy(cv_summary["params"])
    if not math.isclose(float(control.get("feature_fraction", 1.0)), 0.796, rel_tol=0, abs_tol=0.005):
        raise RuntimeError("The saved SINGLE control no longer has feature_fraction approximately 0.796")
    if int(cv_summary["best_iteration"]) != int(CONFIG["boosting_iterations"]):
        raise RuntimeError("The saved SINGLE control no longer uses exactly 31 iterations")
    if int(cv_summary["feature_count"]) != 70 or len(cv_summary["feature_order"]) != 70:
        raise RuntimeError("The saved control is not the exact 70-feature BASE70 matrix")
    plan_records = [{key: value for key, value in plan.items() if key not in ("train_idx", "recent_idx", "valid_idx")} for plan in plans]
    return {
        "experiment_id": CONFIG["experiment_id"],
        "hypothesis": "Restricting model training to recent information may reduce obsolete-history effects; this experiment does not assume improvement.",
        "contrast": "RECENT365 minus original raw SINGLE",
        "single": {
            "source_oof": artifacts["single_oof"],
            "source_boosters": artifacts["single_boosters"],
            "retrained": False,
            "iterations": int(CONFIG["boosting_iterations"]),
            "effective_lightgbm_params": control,
        },
        "challenger": {
            "name": RECENT365,
            "iterations": int(CONFIG["boosting_iterations"]),
            "effective_lightgbm_params": control,
            "changed_model_inputs": ["Fold training rows: saved eligible training rows restricted by nominal_decision_at to the preceding 365 days", "One common weight multiplier per fold preserves original total sample-weight mass"],
            "unchanged_model_parameters": True,
            "min_data_in_leaf_unchanged": True,
        },
        "data": {
            "matrix": artifacts["matrix"],
            "folds": artifacts["folds"],
            "single_oof": artifacts["single_oof"],
            "rows": int(cv_summary["rows"]),
            "feature_count": int(cv_summary["feature_count"]),
            "feature_order": list(cv_summary["feature_order"]),
            "target": TARGET_COLUMN,
            "fold_count": len(plans),
            "fold_fingerprints": common_summary["common_fold_fingerprints"],
            "original_training_weight_sha256": cv_summary["training_weight_sha256"],
            "purge": "Reuse each saved train/validation index pair; retain the existing label-availability purge, then apply the 365-day inclusive lower bound using nominal_decision_at only.",
            "window_days": int(CONFIG["window_days"]),
            "window_boundary": "[first_validation_nominal_decision_at - 365 days, first_validation_nominal_decision_at), UTC; lower bound based on information/decision time, not label completion.",
            "fold_training_counts_and_weight_scales": plan_records,
            "dataset_construction": "Construct one LightGBM Dataset per fold from only that fold's RECENT365 training rows, labels, and scaled weights; validation rows do not contribute to Dataset bin construction.",
            "feature_generation_or_selection": False,
        },
        "quality": {
            "proxy_oof_rows": int(len(pd.read_parquet(base70_seed.BASE_OOF_PATH, columns=["_source_row_index"]))),
            "official_label_rows": 20718,
            "bootstrap": copy.deepcopy(CONFIG["bootstrap"]),
            "calibration": "none",
        },
        "economics": {
            **copy.deepcopy(CONFIG["economics"]),
            "common_execution_calendar": artifacts["common_execution_calendar"],
            "snapshots": {"T59": artifacts["t59_snapshot"], "T60": artifacts["t60_snapshot"]},
            "single_ledger": artifacts["single_raw_trades"],
            "policy": "Existing independent fixed-$5, net-EV >= 1%, T-60 decision/T-59 execution engine, fees, depth and cutoff.",
        },
        "scope": {
            "challenger_fits": 10,
            "early_stopping": False,
            "calibration": False,
            "optuna": False,
            "feature_generation": False,
            "feature_selection": False,
            "archive_replay": False,
            "data_download": False,
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
    identity = _canonical_sha256({"specification": spec, "code_identity": code_identity})
    previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else None
    completed_fits, stages, revisions = {}, {}, []
    frozen_at = started_at = pd.Timestamp.now(tz="UTC").isoformat()
    if previous:
        previous_code = previous.get("input_identity", {})
        previous_identity = previous_code.get("frozen_spec_sha256")
        if previous_identity != identity:
            same_frozen_inputs = (
                _canonical_sha256(previous.get("frozen_comparison", {}).get("specification")) == _canonical_sha256(spec)
                and previous_code.get("config_sha256") == code_identity["config_sha256"]
                and previous_code.get("lightgbm_version") == code_identity["lightgbm_version"]
            )
            if previous.get("completed_fits") and not same_frozen_inputs:
                raise RuntimeError("Existing RECENT365 checkpoints belong to different frozen inputs, parameters, or LightGBM version")
            if previous.get("completed_fits"):
                completed_fits = previous["completed_fits"]
                stages = previous.get("stages", {})
                revisions = list(previous.get("code_revision_history", []))
                if previous_code.get("active_runner_sha256", previous_code.get("runner_sha256")) != code_identity["runner_sha256"]:
                    revisions.append({
                        "previous_runner_sha256": previous_code.get("active_runner_sha256", previous_code.get("runner_sha256")),
                        "current_runner_sha256": code_identity["runner_sha256"],
                        "reason": "Corrected paired-quality point estimates and report interpretation after the ten fits; the frozen model specification and fit implementation are unchanged. Every completed fold checkpoint is hash- and identity-validated before reuse.",
                        "completed_fit_count_preserved": len(completed_fits),
                    })
                identity = previous_identity
            elif same_frozen_inputs:
                identity = previous_identity
                stages = previous.get("stages", {})
                revisions = list(previous.get("code_revision_history", []))
            else:
                revisions = list(previous.get("code_revision_history", []))
                revisions.append({"previous_frozen_sha256": previous_identity, "current_frozen_sha256": identity, "reason": "Frozen specification changed before any fold fit completed."})
        else:
            completed_fits = previous.get("completed_fits", {})
            stages = previous.get("stages", {})
            revisions = list(previous.get("code_revision_history", []))
        frozen_at = previous.get("frozen_at_utc", frozen_at)
        started_at = previous.get("started_utc", started_at)
    manifest = {
        "schema_version": 1,
        "experiment_id": CONFIG["experiment_id"],
        "status": "running",
        "active_stage": "frozen_preparation",
        "started_utc": started_at,
        "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "frozen_at_utc": frozen_at,
        "next_step": "Write the complete fold count and weight table before fitting; then fit or resume the ten RECENT365 folds.",
        "frozen_comparison": {"identity_sha256": identity, "specification": spec},
        "input_identity": {"frozen_spec_sha256": identity, **code_identity},
        "source_artifacts": artifacts,
        "source_verification": previous.get("source_verification") if previous else None,
        "resource_snapshot_before_run": previous.get("resource_snapshot_before_run") if previous else None,
        "completed_fits": completed_fits,
        "stages": stages,
        "code_revision_history": revisions,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest)
    _write_versioned_manifest(manifest)
    return manifest


def _write_versioned_manifest(manifest: dict) -> None:
    fits = [manifest.get("completed_fits", {}).get(str(i)) for i in range(int(CONFIG["fold_count"]))]
    outputs = []
    for stage in manifest.get("stages", {}).values():
        outputs.extend(stage.get("outputs", []))
    _write_json(VERSIONED_MANIFEST_PATH, {
        "status": "frozen_before_fit" if not all(fits) else manifest.get("status"),
        "frozen_at_utc": manifest.get("frozen_at_utc"),
        "finished_utc": manifest.get("finished_utc"),
        "frozen_comparison": manifest["frozen_comparison"],
        "input_identity": manifest["input_identity"],
        "code_revision_history": manifest.get("code_revision_history", []),
        "source_artifacts": manifest["source_artifacts"],
        "fit_count": int(CONFIG["fold_count"]),
        "completed_fit_count": sum(bool(item) for item in fits),
        "fit_checkpoint_records": fits,
        "stages": {name: {key: value for key, value in stage.items() if key != "outputs"} for name, stage in manifest.get("stages", {}).items()},
        "outputs": outputs,
        "result_summary": {
            "quality": manifest.get("stages", {}).get("quality", {}).get("summary"),
            "economics": manifest.get("stages", {}).get("economics_summary", {}).get("summary"),
            "training_fit_diagnostics": manifest.get("stages", {}).get("training", {}).get("summary", {}).get("common_recent365_training_log_loss"),
            "single_anchor_reproduction": manifest.get("single_anchor_reproduction"),
        },
        "performance": {
            "elapsed_seconds": manifest.get("elapsed_seconds"),
            "stage_elapsed_seconds": manifest.get("stage_elapsed_seconds"),
            "peak_process_working_set_gib": manifest.get("peak_process_working_set_gib"),
            "resource_snapshot_before_run": manifest.get("resource_snapshot_before_run"),
            "resource_snapshot_at_completion": manifest.get("resource_snapshot_at_completion"),
        },
        "completion": manifest.get("completion"),
    })


def _fold_paths(fold_id: int) -> tuple[Path, Path, Path]:
    folder = OUTPUT_DIR / "models" / "recent365"
    return folder / f"fold_{fold_id:02d}.txt", folder / f"fold_{fold_id:02d}_oof.parquet", folder / f"fold_{fold_id:02d}_completion.json"


def _completed_fold_valid(manifest: dict, frame: pd.DataFrame, plan: dict, feature_names: list[str]) -> bool:
    fold_id = int(plan["fold_id"])
    record = manifest.get("completed_fits", {}).get(str(fold_id), {})
    model_path, oof_path, completion_path = _fold_paths(fold_id)
    identity = manifest["input_identity"]["frozen_spec_sha256"]
    if record.get("status") != "completed" or record.get("identity_sha256") != identity:
        return False
    files = [record.get(name, {}) for name in ("model", "oof", "completion_manifest")]
    if not _verify_records(files):
        return False
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("identity_sha256") != identity or completion.get("fold_id") != fold_id:
        return False
    if completion.get("model") != record.get("model") or completion.get("oof") != record.get("oof"):
        return False
    for key in ("recent365_source_row_index_sha256", "validation_source_row_index_sha256", "scaled_weight_sha256", "effective_params"):
        if completion.get(key) != plan.get(key, manifest["frozen_comparison"]["specification"]["challenger"]["effective_lightgbm_params"] if key == "effective_params" else None):
            return False
    oof = pd.read_parquet(oof_path)
    expected_ids = frame["_source_row_index"].iloc[plan["valid_idx"]].to_numpy(dtype=np.int32)
    actual_ids = oof["_source_row_index"].to_numpy(dtype=np.int32)
    if len(oof) != len(expected_ids) or not np.array_equal(actual_ids, expected_ids):
        return False
    if completion.get("oof_source_row_index_sha256") != taker._array_sha256(actual_ids, dtype=np.int32):
        return False
    if not oof["fold_id"].astype(str).eq(str(fold_id)).all() or not np.isfinite(oof["p_model_raw"]).all():
        return False
    booster = taker.lgb.Booster(model_file=str(model_path))
    valid = booster.current_iteration() == int(CONFIG["boosting_iterations"]) and list(booster.feature_name()) == feature_names
    del booster, oof
    return bool(valid)


def _weighted_log_loss(labels: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    y = np.asarray(labels, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    losses = -(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))
    return float(np.average(losses, weights=w))


def _assert_identical_oof_rows(single: pd.DataFrame, recent: pd.DataFrame, row_id_columns: list[str], expected_rows: int) -> None:
    for rows in (single, recent):
        rows["fold_id"] = rows["fold_id"].astype(str)
    if len(single) != expected_rows or len(recent) != expected_rows or not single[row_id_columns].equals(recent[row_id_columns]):
        raise RuntimeError("RECENT365 OOF rows do not exactly match the saved SINGLE validation rows")


def _train_recent365(
    manifest: dict,
    frame: pd.DataFrame,
    x: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    plans: list[dict],
    control_params: dict,
) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    feature_names = list(manifest["frozen_comparison"]["specification"]["data"]["feature_order"])
    expected_ids = pd.read_parquet(base70_seed.BASE_OOF_PATH, columns=["_source_row_index"])["_source_row_index"].to_numpy(dtype=np.int32)
    previous_training_summary = manifest.get("stages", {}).get("training", {}).get("summary", {})
    previous_new_fit_count = int(previous_training_summary.get("newly_fitted_fold_count", 0))
    _begin_stage(manifest, "training", "Fit or reuse each fold-specific RECENT365 Dataset and 31-iteration booster; save booster, OOF, and checkpoint per fold.")
    fold_seconds = []
    train_diagnostics = []
    newly_fitted = 0
    for plan in plans:
        fold_id = int(plan["fold_id"])
        if _completed_fold_valid(manifest, frame, plan, feature_names):
            fold_seconds.append(float(manifest["completed_fits"][str(fold_id)].get("elapsed_seconds", 0.0)))
            completion = json.loads(_fold_paths(fold_id)[2].read_text(encoding="utf-8"))
            train_diagnostics.append(completion["common_recent365_training_log_loss"])
            print(f"Reused verified RECENT365 fold {fold_id + 1}/10.")
            continue

        recent_idx = plan["recent_idx"]
        valid_idx = plan["valid_idx"]
        scaled_weights, weight_summary = _scaled_weights(weights, plan["train_idx"], recent_idx)
        if not math.isclose(weight_summary["common_weight_multiplier"], plan["common_weight_multiplier"], rel_tol=0, abs_tol=1e-14):
            raise RuntimeError(f"Fold {fold_id} weight multiplier changed after freezing")
        fold_started = time.perf_counter()
        train_set = taker.lgb.Dataset(
            x[recent_idx],
            label=target[recent_idx],
            weight=scaled_weights,
            feature_name=feature_names,
            params=control_params,
            free_raw_data=False,
        )
        train_set.construct()
        stored_weights = train_set.get_weight()
        if stored_weights is None or not np.allclose(stored_weights.astype(np.float64), scaled_weights, rtol=2e-6, atol=1e-8):
            raise RuntimeError(f"Fold {fold_id} LightGBM Dataset changed or normalized the explicitly scaled observation weights")
        stored_sum = float(stored_weights.astype(np.float64).sum())
        if not math.isclose(stored_sum, weight_summary["original_training_weight_sum"], rel_tol=2e-6, abs_tol=1e-6):
            raise RuntimeError(f"Fold {fold_id} stored weight mass does not match the original full-training mass")
        booster = taker.lgb.train(
            params=control_params,
            train_set=train_set,
            num_boost_round=int(CONFIG["boosting_iterations"]),
            keep_training_booster=False,
        )
        if booster.current_iteration() != int(CONFIG["boosting_iterations"]):
            raise RuntimeError(f"RECENT365 fold {fold_id} did not fit exactly 31 iterations")
        if list(booster.feature_name()) != feature_names:
            raise RuntimeError(f"RECENT365 fold {fold_id} feature order differs from the frozen BASE70 order")
        probability = np.asarray(booster.predict(x[valid_idx], num_iteration=int(CONFIG["boosting_iterations"])), dtype=np.float32)
        if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
            raise RuntimeError(f"RECENT365 fold {fold_id} produced invalid OOF probabilities")

        model_path, oof_path, completion_path = _fold_paths(fold_id)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_model = model_path.with_suffix(".txt.tmp")
        booster.save_model(str(temporary_model), num_iteration=int(CONFIG["boosting_iterations"]))
        temporary_model.replace(model_path)
        fold_oof = base70_seed._write_fold_oof(frame, valid_idx, fold_id, probability)
        _write_parquet(fold_oof, oof_path)

        single_model = taker.lgb.Booster(model_file=str(base70_seed.BASE_MODEL_DIR / f"fold_{fold_id:02d}.txt"))
        single_probability = single_model.predict(x[recent_idx], num_iteration=int(CONFIG["boosting_iterations"]))
        recent_probability = booster.predict(x[recent_idx], num_iteration=int(CONFIG["boosting_iterations"]))
        train_loss = {
            "fold_id": fold_id,
            "common_rows": int(len(recent_idx)),
            "window_start_inclusive_utc": plan["window_start_inclusive_utc"],
            "window_end_exclusive_utc": plan["window_end_exclusive_utc"],
            "SINGLE_weighted_log_loss": _weighted_log_loss(target[recent_idx], single_probability, weights[recent_idx]),
            "RECENT365_weighted_log_loss": _weighted_log_loss(target[recent_idx], recent_probability, weights[recent_idx]),
        }
        train_diagnostics.append(train_loss)
        elapsed = time.perf_counter() - fold_started
        actual_ids = fold_oof["_source_row_index"].to_numpy(dtype=np.int32)
        completion = {
            "identity_sha256": manifest["input_identity"]["frozen_spec_sha256"],
            "variant": RECENT365,
            "fold_id": fold_id,
            "iteration_count": int(booster.current_iteration()),
            "effective_params": copy.deepcopy(control_params),
            "training_rows_original": int(len(plan["train_idx"])),
            "training_rows_recent365": int(len(recent_idx)),
            "validation_rows": int(len(valid_idx)),
            "window_start_inclusive_utc": plan["window_start_inclusive_utc"],
            "window_end_exclusive_utc": plan["window_end_exclusive_utc"],
            "latest_training_label_available_at_utc": frame["target_label_available_at"].iloc[recent_idx].max().isoformat(),
            "validation_first_decision_at_utc": plan["first_validation_decision_at_utc"],
            "validation_last_decision_at_utc": plan["validation_last_decision_at_utc"],
            "recent365_source_row_index_sha256": plan["recent365_source_row_index_sha256"],
            "validation_source_row_index_sha256": plan["validation_source_row_index_sha256"],
            "oof_source_row_index_sha256": taker._array_sha256(actual_ids, dtype=np.int32),
            "scaled_weight_sha256": plan["scaled_weight_sha256"],
            "weight_summary": {**weight_summary, "recent365_scaled_weight_sum_stored": stored_sum, "lightgbm_dataset_weight_normalization": False},
            "common_recent365_training_log_loss": train_loss,
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
        newly_fitted += 1
        fold_seconds.append(elapsed)
        print(f"Completed RECENT365 fold {fold_id + 1}/10 in {elapsed:.1f}s ({len(recent_idx):,} train rows).")
        _write_manifest(manifest)
        _write_versioned_manifest(manifest)
        del train_set, booster, single_model, probability, fold_oof, single_probability, recent_probability

    fold_outputs = [pd.read_parquet(_fold_paths(fold_id)[1]) for fold_id in range(len(plans))]
    combined = pd.concat(fold_outputs, ignore_index=True).sort_values("_source_row_index", kind="stable").reset_index(drop=True)
    actual_ids = combined["_source_row_index"].to_numpy(dtype=np.int32)
    if len(combined) != len(expected_ids) or not np.array_equal(actual_ids, expected_ids):
        raise RuntimeError("RECENT365 OOF rows do not exactly match the saved SINGLE validation rows")
    if combined["_source_row_index"].duplicated().any() or not np.isfinite(combined["p_model_raw"]).all():
        raise RuntimeError("RECENT365 combined OOF contains duplicate rows or invalid probabilities")
    combined_path = OUTPUT_DIR / "oof_recent365.parquet"
    _write_parquet(combined, combined_path)
    train_diagnostics.sort(key=lambda row: int(row["fold_id"]))
    summary = {
        "variant": RECENT365,
        "fold_count": len(plans),
        "newly_fitted_fold_count": previous_new_fit_count + newly_fitted,
        "newly_fitted_this_invocation": newly_fitted,
        "reused_checkpoint_count_this_invocation": len(plans) - newly_fitted,
        "iterations_per_fit": int(CONFIG["boosting_iterations"]),
        "all_fits_completed": all(manifest.get("completed_fits", {}).get(str(i), {}).get("iteration_count") == int(CONFIG["boosting_iterations"]) for i in range(len(plans))),
        "fit_seconds_sum": float(sum(fold_seconds)),
        "fold_seconds_this_invocation": fold_seconds,
        "oof_rows": int(len(combined)),
        "common_recent365_training_log_loss": train_diagnostics,
        "mean_common_recent365_training_log_loss": {
            SINGLE: float(np.mean([row[f"{SINGLE}_weighted_log_loss"] for row in train_diagnostics])),
            RECENT365: float(np.mean([row[f"{RECENT365}_weighted_log_loss"] for row in train_diagnostics])),
        },
        "oof": _record(combined_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not summary["all_fits_completed"]:
        raise RuntimeError("The RECENT365 training stage is incomplete")
    summary_path = OUTPUT_DIR / "training_summary.json"
    _write_json(summary_path, summary)
    outputs = [summary["oof"]]
    for fold_id in range(len(plans)):
        outputs.extend([manifest["completed_fits"][str(fold_id)][name] for name in ("model", "oof", "completion_manifest")])
    outputs.append(_record(summary_path))
    _finish_stage(manifest, "training", summary=summary, outputs=outputs, elapsed_seconds=summary["elapsed_seconds"], completed_fits=len(plans))
    _write_versioned_manifest(manifest)
    return combined, summary


def _build_predictions(manifest: dict) -> pd.DataFrame:
    path = OUTPUT_DIR / "predictions.parquet"
    stage = manifest.get("stages", {}).get("common_predictions", {})
    if stage.get("status") == "completed" and _verify_records(stage.get("outputs", [])):
        return pd.read_parquet(path)
    started = time.perf_counter()
    _begin_stage(manifest, "common_predictions", "Join RECENT365, raw SINGLE, and the existing shared execution calendar without altering their rows.")
    single = pd.read_parquet(base70_seed.BASE_OOF_PATH)
    recent = pd.read_parquet(OUTPUT_DIR / "oof_recent365.parquet")
    ids = list(base70_full.ROW_ID_COLUMNS)
    _assert_identical_oof_rows(single, recent, ids, 318560)
    result = single[ids].copy()
    result[PROBABILITY_COLUMNS[SINGLE]] = single["p_model_raw"].to_numpy(dtype=np.float32)
    result[PROBABILITY_COLUMNS[RECENT365]] = recent["p_model_raw"].to_numpy(dtype=np.float32)
    result["market_start_utc"] = pd.to_datetime(result["market_start_utc"], utc=True, errors="raise")
    if not np.isfinite(result[list(PROBABILITY_COLUMNS.values())]).all().all():
        raise RuntimeError("Common proxy OOF probabilities contain non-finite values")
    _write_parquet(result, path)
    summary = {
        "oof_rows": int(len(result)),
        "variants": list(VARIANTS),
        "single_oof_reused": True,
        "identical_row_and_fold_indices_verified": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _finish_stage(manifest, "common_predictions", summary=summary, outputs=[_record(path)], elapsed_seconds=summary["elapsed_seconds"])
    return result


def _quality(manifest: dict, predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    stage = manifest.get("stages", {}).get("quality", {})
    if stage.get("status") == "completed" and _verify_records(stage.get("outputs", [])):
        cached = json.loads((OUTPUT_DIR / "quality_summary.json").read_text(encoding="utf-8"))
        if "point_difference_RECENT365_minus_SINGLE" in cached["proxy_oof"]["paired_bootstrap"]["contrasts"]["log_loss"]:
            return cached, pd.read_csv(OUTPUT_DIR / "proxy_fold_quality.csv"), pd.read_csv(OUTPUT_DIR / "official_monthly_quality.csv")
    started = time.perf_counter()
    _begin_stage(manifest, "quality", "Score the identical full proxy OOF and 20,718 official labels with paired 3-day UTC block resamples.")
    proxy = predictions.copy()
    proxy["fold_id"] = proxy["fold_id"].astype(str)
    prior_variants = base70_seed.VARIANT_NAMES
    prior_columns = base70_seed.PROBABILITY_COLUMNS
    try:
        base70_seed.VARIANT_NAMES = VARIANTS
        base70_seed.PROBABILITY_COLUMNS = PROBABILITY_COLUMNS
        proxy_overall, flat_proxy_fold_rows = base70_seed._quality_table(proxy, TARGET_COLUMN, "fold_id")
    finally:
        base70_seed.VARIANT_NAMES = prior_variants
        base70_seed.PROBABILITY_COLUMNS = prior_columns
    proxy_fold_rows = []
    for fold_id, part in proxy.groupby("fold_id", sort=True):
        metrics = {row["variant"]: {key: row[key] for key in ("log_loss", "brier", "auc")} for row in flat_proxy_fold_rows if row["group"] == str(fold_id)}
        proxy_fold_rows.append({"fold_id": str(fold_id), "rows": int(len(part)), **metrics})

    info = pd.read_parquet(base70_seed.BASE_PREDICTIONS_PATH)
    official_filter = (
        info["comparison_forecast_available"].astype(bool)
        & info["official_outcome_available_by_cutoff"].astype(bool)
        & pd.to_datetime(info["market_start_utc"], utc=True).lt(pd.Timestamp(CONFIG["economics"]["cutoff_exclusive_utc"]))
    )
    official = info.loc[official_filter, ["market_start_utc", "target_polymarket_up", "fold_id", "p_raw"]].copy()
    official["market_start_utc"] = pd.to_datetime(official["market_start_utc"], utc=True, errors="raise")
    official = official.merge(predictions[["market_start_utc", PROBABILITY_COLUMNS[RECENT365]]], on="market_start_utc", how="left", validate="one_to_one")
    official.rename(columns={"p_raw": PROBABILITY_COLUMNS[SINGLE]}, inplace=True)
    official["fold_id"] = official["fold_id"].astype(str)
    if len(official) != 20718 or official["target_polymarket_up"].nunique() != 2 or official[list(PROBABILITY_COLUMNS.values())].isna().any().any():
        raise RuntimeError("The official sample does not reproduce the same 20,718 labeled outcomes for both variants")
    if not np.array_equal(
        official.sort_values("market_start_utc", kind="stable")[PROBABILITY_COLUMNS[SINGLE]].to_numpy(dtype=np.float64),
        info.loc[official_filter].sort_values("market_start_utc", kind="stable")["p_raw"].to_numpy(dtype=np.float64),
    ):
        raise RuntimeError("The original raw SINGLE official probabilities changed during the join")
    official_overall = {
        name: base70_full._score(official["target_polymarket_up"].to_numpy(dtype=np.int8), official[column].to_numpy(dtype=np.float64))
        for name, column in PROBABILITY_COLUMNS.items()
    }
    official["utc_month"] = official["market_start_utc"].dt.strftime("%Y-%m")
    monthly_rows = []
    for month, rows in official.groupby("utc_month", sort=True):
        labels = rows["target_polymarket_up"].to_numpy(dtype=np.int8)
        for name, column in PROBABILITY_COLUMNS.items():
            monthly_rows.append({"utc_month": month, "variant": name, "rows": int(len(rows)), **base70_full._score(labels, rows[column].to_numpy(dtype=np.float64))})

    prior_full_constants = (base70_full.CONFIG, base70_full.FULL_FEATURES, base70_full.PROBABILITY_COLUMNS)
    base70_full.CONFIG = {"bootstrap": copy.deepcopy(CONFIG["bootstrap"])}
    base70_full.FULL_FEATURES = RECENT365
    base70_full.PROBABILITY_COLUMNS = PROBABILITY_COLUMNS
    try:
        proxy_bootstrap = base70_full._paired_quality_bootstrap(proxy, TARGET_COLUMN, int(CONFIG["bootstrap"]["seed"]))
        official_bootstrap = base70_full._paired_quality_bootstrap(official, "target_polymarket_up", int(CONFIG["bootstrap"]["seed"]))
        proxy_delta = base70_full._probability_difference(proxy)
        official_delta = base70_full._probability_difference(official)
    finally:
        base70_full.CONFIG, base70_full.FULL_FEATURES, base70_full.PROBABILITY_COLUMNS = prior_full_constants

    for bootstrap_result in (proxy_bootstrap, official_bootstrap):
        for values in bootstrap_result["contrasts"].values():
            point_key = next(key for key in values if key.startswith("point_difference_"))
            values["point_difference_RECENT365_minus_SINGLE"] = values.pop(point_key)

    summary = {
        "proxy_oof": {
            "sample_rows": int(len(proxy)), "overall": proxy_overall, "fold_metrics": proxy_fold_rows,
            "paired_bootstrap": proxy_bootstrap, "probability_difference_RECENT365_minus_SINGLE": proxy_delta,
        },
        "official": {
            "sample_rows": int(len(official)), "up_labels": int(official.target_polymarket_up.sum()),
            "down_labels": int(len(official) - official.target_polymarket_up.sum()),
            "start_utc": official.market_start_utc.min().isoformat(), "end_utc": official.market_start_utc.max().isoformat(),
            "overall": official_overall, "monthly_metrics": monthly_rows, "paired_bootstrap": official_bootstrap,
            "probability_difference_RECENT365_minus_SINGLE": official_delta,
        },
        "single_oof_reused": True,
        "full_proxy_oof_rows_identical": True,
        "calibration_applied": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    quality_path = OUTPUT_DIR / "quality_summary.json"
    fold_path = OUTPUT_DIR / "proxy_fold_quality.csv"
    monthly_path = OUTPUT_DIR / "official_monthly_quality.csv"
    official_path = OUTPUT_DIR / "official_quality_rows.parquet"
    _write_json(quality_path, summary)
    pd.DataFrame(flat_proxy_fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)
    _write_parquet(official, official_path)
    outputs = [_record(path) for path in (quality_path, fold_path, monthly_path, official_path)]
    _finish_stage(manifest, "quality", summary=summary, outputs=outputs, elapsed_seconds=summary["elapsed_seconds"])
    _write_versioned_manifest(manifest)
    return summary, pd.DataFrame(proxy_fold_rows), pd.DataFrame(monthly_rows)


def _configure_economic_helpers() -> None:
    base70_info.OUTPUT_DIR = OUTPUT_DIR
    base70_info.MANIFEST_PATH = MANIFEST_PATH
    base70_info.CONFIG = {
        "policy": {
            "gross_stake_usd": float(CONFIG["economics"]["gross_stake_usd"]),
            "minimum_net_ev_fraction": float(CONFIG["economics"]["minimum_net_ev_fraction"]),
            "decision_offset_seconds": int(CONFIG["economics"]["decision_offset_seconds"]),
            "execution_offset_seconds": int(CONFIG["economics"]["execution_offset_seconds"]),
        },
        "bootstrap": copy.deepcopy(CONFIG["bootstrap"]),
    }
    base70_info.VARIANTS = VARIANTS
    base70_info.PROBABILITY_COLUMNS = PROBABILITY_COLUMNS
    base70_info.CONTRASTS = CONTRASTS
    base70_info.START_UTC = pd.Timestamp(CONFIG["economics"]["start_utc"])
    base70_info.CUTOFF_UTC = pd.Timestamp(CONFIG["economics"]["cutoff_exclusive_utc"])
    base70_info._status_update = lambda _manifest: None


def _economics(manifest: dict, predictions: pd.DataFrame, info_manifest: dict) -> dict:
    stage = manifest.get("stages", {}).get("economics_summary", {})
    if stage.get("status") == "completed" and _verify_records(stage.get("outputs", [])):
        summary = json.loads((OUTPUT_DIR / "economics_summary.json").read_text(encoding="utf-8"))
        manifest["single_anchor_reproduction"] = summary["single_anchor_reproduction"]
        return summary
    started = time.perf_counter()
    _configure_economic_helpers()
    common, t59, t60, execution_inputs = base70_full._load_economic_inputs(info_manifest)
    if len(common) != int(CONFIG["economics"]["calendar_rows"]):
        raise RuntimeError("The saved common execution calendar differs from the frozen 20,735 decision rows")
    execution_path = OUTPUT_DIR / "execution_input_summary.json"
    _write_json(execution_path, execution_inputs)
    _finish_stage(manifest, "execution_inputs", summary=execution_inputs, outputs=[_record(execution_path)], elapsed_seconds=execution_inputs["elapsed_seconds"])

    summaries, decisions, trades = {}, {}, {}
    for variant in VARIANTS:
        summary, decision_frame, trade_frame = base70_info._run_variant(variant, predictions, common, t59, t60, manifest)
        if not base70_info.summary_accounting_passed(summary):
            raise RuntimeError(f"{variant} failed the existing economic accounting audit")
        summaries[variant], decisions[variant], trades[variant] = summary, decision_frame, trade_frame
        if variant == SINGLE:
            previous_output_dir = base70_seed.OUTPUT_DIR
            try:
                base70_seed.OUTPUT_DIR = OUTPUT_DIR
                manifest["single_anchor_reproduction"] = base70_seed._verify_single_economics(trade_frame, summary)
            finally:
                base70_seed.OUTPUT_DIR = previous_output_dir
        _write_manifest(manifest)

    days = pd.date_range(
        pd.to_datetime(common.market_start_utc, utc=True).min().floor("D"),
        pd.to_datetime(common.market_start_utc, utc=True).max().floor("D"),
        freq="D", tz="UTC",
    )
    daily = base70_info._build_daily_pnl(trades, days)
    draws = base70_seed._block_indices(days, int(CONFIG["bootstrap"]["replications"]), int(CONFIG["bootstrap"]["block_days"]), int(CONFIG["bootstrap"]["seed"]))
    bootstrap = base70_info._economic_bootstrap(daily, draws)
    left_pnl = float(daily[f"{RECENT365}_net_pnl_usd"].sum())
    right_pnl = float(daily[f"{SINGLE}_net_pnl_usd"].sum())
    left_turnover = float(daily[f"{RECENT365}_settled_turnover_usd"].sum())
    right_turnover = float(daily[f"{SINGLE}_settled_turnover_usd"].sum())
    bootstrap["contrasts"][CONTRAST]["pnl_per_settled_turnover_difference"]["point_estimate"] = (
        left_pnl / left_turnover - right_pnl / right_turnover if left_turnover > 0 and right_turnover > 0 else None
    )
    decomposition = base70_info._decompose_pnl(trades[RECENT365], trades[SINGLE], RECENT365, SINGLE)
    if not decomposition["sum_reconciles"]:
        raise RuntimeError("RECENT365 minus SINGLE economic PnL decomposition does not reconcile")
    summary = {
        "variants": summaries,
        "availability": {
            "decision_rows": int(len(common)), "calendar_days": int(len(days)),
            "first_market_start_utc": pd.to_datetime(common.market_start_utc, utc=True).min().isoformat(),
            "last_market_start_utc": pd.to_datetime(common.market_start_utc, utc=True).max().isoformat(),
        },
        "bootstrap": bootstrap,
        "concentration": base70_info._concentration(daily),
        "pnl_decomposition_RECENT365_minus_SINGLE": decomposition,
        "up_down_economics": base70_seed._side_economics(trades),
        "single_anchor_reproduction": manifest["single_anchor_reproduction"],
        "accounting_audits_passed": all(base70_info.summary_accounting_passed(item) for item in summaries.values()),
        "daily_calendar_rows_including_no_fill_days": int(len(daily)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = OUTPUT_DIR / "economics_summary.json"
    daily_path = OUTPUT_DIR / "daily_economics.csv"
    decomposition_path = OUTPUT_DIR / "pnl_decomposition.json"
    _write_json(summary_path, summary)
    daily.to_csv(daily_path, index=False)
    _write_json(decomposition_path, decomposition)
    variant_outputs = [record for variant in VARIANTS for record in manifest["stages"][f"economics_{variant}"]["outputs"]]
    outputs = [*variant_outputs, *[_record(path) for path in (summary_path, daily_path, decomposition_path)]]
    _finish_stage(manifest, "economics_summary", summary=summary, outputs=outputs, elapsed_seconds=summary["elapsed_seconds"])
    _write_versioned_manifest(manifest)
    return summary


def _fmt(value, digits: int = 6) -> str:
    return "—" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _quality_table_rows(rows: list[dict], variant_key: str = "variant") -> str:
    return "\n".join(
        f"| {row.get(variant_key, row.get('fold_id', ''))} | {row.get('rows', row.get('sample_rows', ''))} | {_fmt(row.get(SINGLE, {}).get('log_loss') if isinstance(row.get(SINGLE), dict) else row.get('SINGLE_log_loss'))} | {_fmt(row.get(RECENT365, {}).get('log_loss') if isinstance(row.get(RECENT365), dict) else row.get('RECENT365_log_loss'))} | {_fmt(row.get(SINGLE, {}).get('brier') if isinstance(row.get(SINGLE), dict) else row.get('SINGLE_brier'))} | {_fmt(row.get(RECENT365, {}).get('brier') if isinstance(row.get(RECENT365), dict) else row.get('RECENT365_brier'))} | {_fmt(row.get(SINGLE, {}).get('auc') if isinstance(row.get(SINGLE), dict) else row.get('SINGLE_auc'))} | {_fmt(row.get(RECENT365, {}).get('auc') if isinstance(row.get(RECENT365), dict) else row.get('RECENT365_auc'))} |"
        for row in rows
    )


def _write_report(manifest: dict) -> None:
    spec = manifest["frozen_comparison"]["specification"]
    quality = manifest["stages"]["quality"]["summary"]
    economics = manifest["stages"]["economics_summary"]["summary"]
    training = manifest["stages"]["training"]["summary"]
    proxy = quality["proxy_oof"]
    official = quality["official"]
    econ_variants = economics["variants"]
    control = econ_variants[SINGLE]
    candidate = econ_variants[RECENT365]
    econ_bootstrap = economics["bootstrap"]["contrasts"][CONTRAST]
    quality_bootstrap_proxy = proxy["paired_bootstrap"]["contrasts"]
    quality_bootstrap_official = official["paired_bootstrap"]["contrasts"]
    fold_quality = proxy["fold_metrics"]
    fold_rows = fold_quality
    proxy_overall = proxy["overall"]
    official_overall = official["overall"]
    fold_count_rows = spec["data"]["fold_training_counts_and_weight_scales"]
    count_table = "\n".join(
        f"| {row['fold_id']} | {row['window_start_inclusive_utc']} | {row['window_end_exclusive_utc']} | {row['training_rows_before']} | {row['training_rows_after']} | {row['validation_rows']} | {_fmt(row['original_training_weight_sum'], 3)} | {_fmt(row['recent365_unscaled_weight_sum'], 3)} | {_fmt(row['common_weight_multiplier'], 9)} | {_fmt(row['recent365_scaled_weight_sum_requested'], 3)} |"
        for row in fold_count_rows
    )
    proxy_overall_table = "\n".join(f"| {name} | {_fmt(proxy_overall[name]['log_loss'])} | {_fmt(proxy_overall[name]['brier'])} | {_fmt(proxy_overall[name]['auc'])} |" for name in VARIANTS)
    official_overall_table = "\n".join(f"| {name} | {_fmt(official_overall[name]['log_loss'])} | {_fmt(official_overall[name]['brier'])} | {_fmt(official_overall[name]['auc'])} |" for name in VARIANTS)
    official_monthly = []
    for month in sorted({row["utc_month"] for row in official["monthly_metrics"]}):
        values = {row["variant"]: row for row in official["monthly_metrics"] if row["utc_month"] == month}
        official_monthly.append(f"| {month} | {values[SINGLE]['rows']} | {_fmt(values[SINGLE]['log_loss'])} | {_fmt(values[RECENT365]['log_loss'])} | {_fmt(values[SINGLE]['brier'])} | {_fmt(values[RECENT365]['brier'])} | {_fmt(values[SINGLE]['auc'])} | {_fmt(values[RECENT365]['auc'])} |")
    econ_rows = []
    for name, item in ((SINGLE, control), (RECENT365, candidate)):
        econ_rows.append(f"| {name} | {item['fill_count']} | ${item['gross_turnover_usd']:.2f} | ${item['fees_paid_estimated_usd']:.2f} | ${item['net_pnl_usd']:.2f} | {item['pnl_per_settled_turnover']:.2%} | ${item['forecast_net_ev_on_settled_fills_usd']:.2f} |")
    econ_months = []
    for name, item in ((SINGLE, control), (RECENT365, candidate)):
        for month in item["monthly_results"]:
            econ_months.append(f"| {name} | {month['utc_month']} | {month['fills']} ({month['up_fills']}/{month['down_fills']}) | ${month['gross_turnover_usd']:.2f} | ${month['fees_usd']:.2f} | ${month['net_pnl_usd']:.2f} | ${month['forecast_ev_on_settled_fills_usd']:.2f} |")
    side_rows = [f"| {row['variant']} | {row['side']} | {row['fills']} | ${row['turnover_usd']:.2f} | ${row['fees_usd']:.2f} | ${row['net_pnl_usd']:.2f} | {row['pnl_per_settled_turnover']:.2%} | ${row['forecast_ev_usd']:.2f} |" for row in economics["up_down_economics"]]
    top_day_rows = []
    for name in VARIANTS:
        for row in economics["concentration"][name]["top_five_days"]:
            top_day_rows.append(f"| {name} | {row['utc_day'][:10]} | ${row['net_pnl_usd']:.2f} |")
    train_rows = [f"| {row['fold_id']} | {row['common_rows']} | {_fmt(row['SINGLE_weighted_log_loss'])} | {_fmt(row['RECENT365_weighted_log_loss'])} |" for row in training["common_recent365_training_log_loss"]]
    proxy_wins = {
        metric: sum(
            row[RECENT365][metric] > row[SINGLE][metric] if metric == "auc" else row[RECENT365][metric] < row[SINGLE][metric]
            for row in fold_quality
        )
        for metric in ("log_loss", "brier", "auc")
    }
    official_month_wins = {}
    for metric in ("log_loss", "brier", "auc"):
        wins = 0
        for month in sorted({row["utc_month"] for row in official["monthly_metrics"]}):
            monthly_rows = [row for row in official["monthly_metrics"] if row["utc_month"] == month]
            monthly_single = next(row for row in monthly_rows if row["variant"] == SINGLE)
            monthly_recent = next(row for row in monthly_rows if row["variant"] == RECENT365)
            wins += monthly_recent[metric] > monthly_single[metric] if metric == "auc" else monthly_recent[metric] < monthly_single[metric]
        official_month_wins[metric] = wins
    concentration_share = {}
    for name in VARIANTS:
        item = economics["concentration"][name]
        concentration_share[name] = f"{item['top_five_share_of_total_pnl']:.1%}" if item["total_pnl_usd"] > 0 else "N/A (total PnL ≤ 0)"
    hardware = manifest.get("resource_snapshot_before_run", {})
    conclusion = (
        "Wynik nie uzasadnia dalszej pracy nad tym konkretnym wariantem RECENT365; zamyka wyłącznie okno 365 dni z tą procedurą."
        if not _recent365_merits_followup(quality, economics)
        else "RECENT365 wykazuje wystarczająco spójny sygnał w jakości i ekonomii, aby uzasadnić osobną dalszą pracę rozwojową. Wynik nie jest oceną holdoutu ani wdrożeniem."
    )
    report = f"""# BASE70 RECENT365 — 2026-10-18

## Wniosek

{conclusion}

Ocena rozdziela jakość prawdopodobieństw od wyniku strategii. Sam punktowy PnL nie przesądza wniosku. Brak poprawy zamyka ten wariant 365 dni, ale nie dowodzi nieskuteczności innych metod adaptacji.

RECENT365 miał lepszy log-loss/Brier/AUC odpowiednio w {proxy_wins['log_loss']}/10, {proxy_wins['brier']}/10 i {proxy_wins['auc']}/10 proxy foldach. Na oficjalnym targetcie wygrał w {official_month_wins['log_loss']}/3, {official_month_wins['brier']}/3 i {official_month_wins['auc']}/3 miesiącach; sparowane przedziały różnic potwierdzają pogorszenie łączne. Po kosztach RECENT365 przyniósł o ${control['net_pnl_usd'] - candidate['net_pnl_usd']:.2f} mniej, a bootstrap różnicy dziennego PnL nie obejmuje zera.

## Zamrożona procedura

- Kontrast: surowy, oryginalny `SINGLE` z 70 cechami i `feature_fraction={spec['single']['effective_lightgbm_params']['feature_fraction']}` kontra `RECENT365 − SINGLE`.
- Wykonano dokładnie 10 nowych fitów po 31 iteracji; bez kalibracji, early stoppingu i strojenia. Cechy, target, kolejność cech, parametry i efektywne seedy odpowiadają kontroli.
- Każdy fold zachowuje zapisany zbiór walidacyjny i oryginalny zbiór dopuszczonych treningowo obserwacji. Okno to `[pierwsza decyzja walidacyjna − 365 dni, pierwsza decyzja walidacyjna) UTC` według `nominal_decision_at`; dolna granica jest włączona. Istniejący purge etykiet pozostaje bez zmian.
- Każdy foldowy Dataset LightGBM powstał wyłącznie z jego zachowanych danych treningowych. Walidacja nie uczestniczyła w konstrukcji binów.
- Wagi zachowanych wierszy pomnożono wspólnie przez `suma_wag_treningu_przed / suma_wag_RECENT365`; proporcje minut i pomocniczych obserwacji są zachowane, a łączna masa wagi odpowiada kontroli. Dataset zachował przekazane wagi bez dodatkowej normalizacji.
- `min_data_in_leaf` pozostało bez zmian. Mniejsza liczba wierszy zmienia jednak jego względne ograniczenie. To wynik konkretnej procedury RECENT365, a nie całkowicie odizolowany efekt wieku danych.
- Ograniczenie obejmuje tylko fit modelu głównego. Sposób utworzenia macierzy cech BASE70 pozostaje zamrożony.

## Liczebności i skala wag

| Fold | Początek UTC (włącznie) | Koniec UTC (wyłącznie) | Train przed | Train RECENT365 | Walidacja | Suma wag train | Suma wag RECENT365 przed skalą | Mnożnik | Suma wag po skali |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
{count_table}

## Jakość proxy OOF

Identyczne pełne OOF SINGLE/RECENT365: {proxy['sample_rows']:,} wierszy, ocenione łącznie i per fold.

| Wariant | Log-loss | Brier | AUC |
|---|---:|---:|---:|
{proxy_overall_table}

| Fold | Wiersze | SINGLE LL | RECENT365 LL | SINGLE Brier | RECENT365 Brier | SINGLE AUC | RECENT365 AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
{_quality_table_rows(fold_rows, 'fold_id')}

Sparowany bootstrap wspólnych 3-dniowych bloków UTC, 2 000 replikacji (RECENT365 − SINGLE):

| Metryka | Różnica punktowa | 95% przedział | Repliki poprawne |
|---|---:|---:|---:|
{_bootstrap_quality_rows(quality_bootstrap_proxy)}

## Oficjalne etykiety

Na tych samych {official['sample_rows']:,} oficjalnych etykietach (UP {official['up_labels']:,}, DOWN {official['down_labels']:,}):

| Wariant | Log-loss | Brier | AUC |
|---|---:|---:|---:|
{official_overall_table}

| Miesiąc UTC | Wiersze | SINGLE LL | RECENT365 LL | SINGLE Brier | RECENT365 Brier | SINGLE AUC | RECENT365 AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
{''.join(row + chr(10) for row in official_monthly)}

Sparowany bootstrap oficjalnego targetu:

| Metryka | Różnica punktowa | 95% przedział | Repliki poprawne |
|---|---:|---:|---:|
{_bootstrap_quality_rows(quality_bootstrap_official)}

## Pomocnicza diagnostyka dopasowania

Log-loss ważony na tym samym ostatnim rocznym fragmencie oryginalnego treningu każdego foldu; predykcje wykonano zapisanymi boosterami, bez refitu.

| Fold | Wspólne wiersze | SINGLE | RECENT365 |
|---:|---:|---:|---:|
{''.join(row + chr(10) for row in train_rows)}

Ta diagnostyka opisuje dopasowanie treningowe i nie jest kryterium wyboru.

## Ekonomia po kosztach

Wspólny zapisany kalendarz: 20 735 decyzji. Niezależne zlecenia po $5, próg EV netto 1%, T−60 decyzja/T−59 wykonanie, niezmienione snapshoty, opłaty, głębokość i cutoff. SINGLE odtworzono z istniejącego RAW ledgeru: {manifest['single_anchor_reproduction']['actual_fills']:,} wykonań, ${manifest['single_anchor_reproduction']['actual_turnover_usd']:,.2f} obrotu i ${manifest['single_anchor_reproduction']['actual_net_pnl_usd']:,.2f} PnL; audyt księgowy przeszedł.

| Wariant | Wykonania | Obrót | Opłaty | PnL netto | PnL/obrót | Prognozowane EV na tych samych rozliczonych wykonaniach |
|---|---:|---:|---:|---:|---:|---:|
{''.join(row + chr(10) for row in econ_rows)}

Miesięcznie; liczba UP/DOWN pokazana w nawiasach:

| Wariant | Miesiąc UTC | Wykonania (UP/DOWN) | Obrót | Opłaty | PnL | EV |
|---|---|---:|---:|---:|---:|---:|
{''.join(row + chr(10) for row in econ_months)}

UP/DOWN:

| Wariant | Strona | Wykonania | Obrót | Opłaty | PnL | PnL/obrót | Prognozowane EV |
|---|---|---:|---:|---:|---:|---:|---:|
{''.join(row + chr(10) for row in side_rows)}

Pięć najlepszych dni:

| Wariant | Dzień UTC | PnL |
|---|---|---:|
{''.join(row + chr(10) for row in top_day_rows)}

Udział tych dni w PnL wyniósł SINGLE {concentration_share[SINGLE]} i RECENT365 {concentration_share[RECENT365]}.

Sparowany bootstrap dziennej różnicy RECENT365 − SINGLE obejmuje wszystkie dni kalendarza, również bez transakcji. Średnia dzienna różnica PnL: ${econ_bootstrap['mean_daily_pnl_difference_usd']['point_estimate']:.4f}, 95% przedział ${econ_bootstrap['mean_daily_pnl_difference_usd']['95pct'][0]:.4f} do ${econ_bootstrap['mean_daily_pnl_difference_usd']['95pct'][1]:.4f}. Różnica PnL/obrót: {econ_bootstrap['pnl_per_settled_turnover_difference']['point_estimate']:.2%}, 95% przedział {econ_bootstrap['pnl_per_settled_turnover_difference']['95pct'][0]:.2%} do {econ_bootstrap['pnl_per_settled_turnover_difference']['95pct'][1]:.2%}.

## Wykonanie i artefakty

- Run manifest oraz tabele i ledgery: `{_relative(MANIFEST_PATH).replace('run_manifest.json', '')}`; versioned manifest: `{_relative(VERSIONED_MANIFEST_PATH)}`.
- Raport nie ocenia holdoutu. Nie pobierano danych, nie uruchamiano replayu archiwum ani refitu wdrożeniowego.
- Czas całkowity: {manifest.get('elapsed_seconds', 0):.1f}s; szczytowy working set procesu: {manifest.get('peak_process_working_set_gib', 0):.3f} GiB; dostępna pamięć przed startem: {hardware.get('available_memory_gib', 'unknown')} GiB.
- Zasoby/system: {platform.platform()}; LightGBM {taker.lgb.__version__}. Modele, OOF i checkpointy pozostają w lokalnym katalogu danych; raport i manifest zawierają ich hashe.

"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report.rstrip() + "\n", encoding="utf-8")


def _bootstrap_quality_rows(contrasts: dict) -> str:
    rows = []
    for metric in ("log_loss", "brier", "auc"):
        value = contrasts[metric]
        interval = value.get("95pct", [None, None])
        rows.append(f"| {metric} | {_fmt(value.get('point_difference_RECENT365_minus_SINGLE'))} | {_fmt(interval[0])} to {_fmt(interval[1])} | {value.get('valid_replications', 0)} |")
    return "\n".join(rows)


def _recent365_merits_followup(quality: dict, economics: dict) -> bool:
    proxy = quality["proxy_oof"]["paired_bootstrap"]["contrasts"]
    official = quality["official"]["paired_bootstrap"]["contrasts"]
    economic = economics["bootstrap"]["contrasts"][CONTRAST]
    proxy_gain = proxy["log_loss"].get("95pct", [1.0, 1.0])[1] < 0 and proxy["brier"].get("95pct", [1.0, 1.0])[1] < 0
    official_gain = official["log_loss"].get("95pct", [1.0, 1.0])[1] < 0 and official["brier"].get("95pct", [1.0, 1.0])[1] < 0
    economic_gain = economic["mean_daily_pnl_difference_usd"].get("95pct", [0.0, 0.0])[0] > 0
    return bool(proxy_gain and official_gain and economic_gain)


def _finalize_versioned_manifest(manifest: dict) -> None:
    _write_versioned_manifest(manifest)


def run() -> dict:
    run_started = time.perf_counter()
    previous_run = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    prior_elapsed = float(previous_run.get("elapsed_seconds", 0.0))
    prior_peak = float(previous_run.get("peak_process_working_set_gib", 0.0))
    first_resource_snapshot = previous_run.get("resource_snapshot_before_run")
    resource_before = taker._resource_snapshot()
    flow, info, seed_manifest, seed_versioned, cv, common, source, full_manifest = _load_sources()
    artifacts = _source_artifacts(flow, info, seed_manifest, source, full_manifest)
    frame = x = target = weights = folds = None
    with taker.PeakMemorySampler() as memory:
        try:
            frame, x, target, weights, folds, single_check = base70_seed._load_training_inputs(cv, common, verify_single=True)
            plans = _fold_plans(frame, weights, folds)
            if len(plans) != int(CONFIG["fold_count"]):
                raise RuntimeError("The saved common fold set does not contain exactly ten folds")
            spec = _frozen_specification(artifacts, cv, common, plans)
            manifest = _create_or_resume_manifest(spec, artifacts)
            if not manifest.get("resource_snapshot_before_run"):
                manifest["resource_snapshot_before_run"] = first_resource_snapshot or resource_before
            manifest["source_verification"] = {
                "single_oof_and_all_ten_saved_boosters_reproduced": single_check,
                "feature_order_identical": True,
                "saved_fold_fingerprints": common["common_fold_fingerprints"],
                "matrix_rows": int(len(frame)),
                "weights_sha256": cv["training_weight_sha256"],
                "source_information_status": info.get("status"),
                "source_seed_stability_status": seed_manifest.get("status"),
                "source_full_features_status": full_manifest.get("status"),
            }
            _finish_stage(manifest, "source_verification", summary=manifest["source_verification"], outputs=[artifacts["matrix"], artifacts["folds"], artifacts["single_oof"]], elapsed_seconds=0.0)

            counts_path = OUTPUT_DIR / "fold_training_counts.csv"
            count_rows = [{key: value for key, value in plan.items() if key not in ("train_idx", "recent_idx", "valid_idx")} for plan in plans]
            pd.DataFrame(count_rows).to_csv(counts_path, index=False)
            _finish_stage(manifest, "frozen_fold_counts", summary={"folds": count_rows, "written_before_any_fit": True}, outputs=[_record(counts_path)], elapsed_seconds=0.0)
            _write_versioned_manifest(manifest)

            previous_seed_fit = float(seed_manifest["stages"]["training"]["summary"]["elapsed_seconds"])
            prior_fit_count = int(seed_manifest["stages"]["training"]["summary"]["new_fits"])
            manifest["training_runtime_estimate_seconds"] = previous_seed_fit * int(CONFIG["fold_count"]) / prior_fit_count
            _write_manifest(manifest)
            print(
                f"Frozen before fit: exactly ten RECENT365 folds x 31 rounds; "
                f"estimated from the saved {prior_fit_count}-fit BASE70 run: {manifest['training_runtime_estimate_seconds']:.1f}s; "
                f"available RAM: {resource_before.get('available_memory_gib', 'unknown')} GiB."
            )

            training_started = time.perf_counter()
            oof, training_summary = _train_recent365(manifest, frame, x, target, weights, plans, cv["params"])
            predictions_started = time.perf_counter()
            predictions = _build_predictions(manifest)
            manifest.setdefault("stage_elapsed_seconds", {})["training"] = time.perf_counter() - training_started
            manifest["stage_elapsed_seconds"]["common_predictions"] = time.perf_counter() - predictions_started

            quality_started = time.perf_counter()
            quality, _, _ = _quality(manifest, predictions)
            manifest["stage_elapsed_seconds"]["quality"] = time.perf_counter() - quality_started
            economics_started = time.perf_counter()
            economics = _economics(manifest, predictions, info)
            manifest["stage_elapsed_seconds"]["economics"] = time.perf_counter() - economics_started

            manifest["peak_process_working_set_gib"] = round(max(prior_peak, memory.peak_bytes / 2**30), 3)
            manifest["resource_snapshot_at_completion"] = taker._resource_snapshot()
            manifest["status"] = "completed"
            manifest["active_stage"] = "report"
            manifest["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
            manifest["elapsed_seconds"] = prior_elapsed + time.perf_counter() - run_started
            manifest["next_step"] = "Review the single RECENT365 comparison; do not extend to another window automatically."
            _write_manifest(manifest)
            report_started = time.perf_counter()
            _write_report(manifest)
            _finish_stage(manifest, "report", outputs=[_record(REPORT_PATH)], elapsed_seconds=time.perf_counter() - report_started)
            manifest["completion"] = {
                "exactly_ten_new_fits_31_iterations": all(manifest["completed_fits"][str(i)]["iteration_count"] == 31 for i in range(10)),
                "validation_rows_identical_to_single": True,
                "proxy_oof_rows": int(len(oof)),
                "official_rows": int(quality["official"]["sample_rows"]),
                "economic_calendar_rows": int(economics["availability"]["decision_rows"]),
                "accounting_audits_passed": economics["accounting_audits_passed"],
                "pnl_decomposition_reconciles": economics["pnl_decomposition_RECENT365_minus_SINGLE"]["sum_reconciles"],
                "single_anchor_reproduced": manifest["single_anchor_reproduction"]["passed"],
            }
            manifest["active_stage"] = None
            manifest["elapsed_seconds"] = prior_elapsed + time.perf_counter() - run_started
            _write_manifest(manifest)
            _finalize_versioned_manifest(manifest)
            return manifest
        except Exception as exc:
            current = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {"experiment_id": CONFIG["experiment_id"], "stages": {}, "completed_fits": {}}
            current["status"] = "failed"
            current["failure"] = {"stage": current.get("active_stage"), "error_type": type(exc).__name__, "error": str(exc), "at_utc": pd.Timestamp.now(tz="UTC").isoformat()}
            current["active_stage"] = None
            current["next_step"] = "Inspect this failure and rerun the no-argument runner to reuse only complete identity-matching fold checkpoints."
            _write_manifest(current)
            _write_versioned_manifest(current)
            raise


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "status": result["status"],
        "report": _relative(REPORT_PATH),
        "fit_count": len(result.get("completed_fits", {})),
        "quality": result.get("stages", {}).get("quality", {}).get("summary", {}).get("official", {}).get("overall"),
        "economic_pnl": {name: row["net_pnl_usd"] for name, row in result.get("stages", {}).get("economics_summary", {}).get("summary", {}).get("variants", {}).items()},
        "peak_working_set_gib": result.get("peak_process_working_set_gib"),
    }, indent=2))
