"""Prepare a separate, resumable full-history BTC deployment-candidate refit."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_preopen_experiment as experiment


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_preopen_deployment_refit_20261011.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_dir"]
MANIFEST_PATH = OUTPUT_DIR / "deployment_refit_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _completed_stage_output(run_manifest: dict, stage_name: str, suffix: str) -> Path:
    stage = run_manifest.get("stages", {}).get(stage_name, {})
    if stage.get("status") != "completed":
        raise RuntimeError(f"Evaluated bundle stage is incomplete: {stage_name}")
    for record in stage.get("outputs", []):
        path = ROOT / record["path"]
        if str(record["path"]).endswith(suffix) and path.is_file() and _sha256(path) == record.get("sha256"):
            return path
    raise RuntimeError(f"No verified {suffix} output exists for evaluated stage {stage_name}")


def _copy_frozen_file(source: Path, target: Path, expected_sha256: str) -> str:
    if not source.is_file() or _sha256(source) != expected_sha256:
        raise RuntimeError(f"Frozen source artifact changed: {source}")
    if target.is_file():
        if _sha256(target) != expected_sha256:
            raise RuntimeError(f"Existing deployment candidate artifact differs: {target}")
    else:
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    return _sha256(target)


def run_deployment_refit() -> dict:
    """Fit the later candidate only when explicitly invoked; this script is not called by research."""
    research_manifest_path = ROOT / CONFIG["research_manifest_path"]
    holdout_manifest_path = ROOT / CONFIG["holdout_manifest_path"]
    if not research_manifest_path.is_file() or not holdout_manifest_path.is_file():
        raise RuntimeError("Complete the evaluated BTC research and holdout before deployment refit")
    research_manifest = _read_json(research_manifest_path)
    holdout = _read_json(holdout_manifest_path)
    if research_manifest.get("status") != "completed" or holdout.get("status") != "completed":
        raise RuntimeError("Deployment refit requires completed, frozen research and holdout manifests")
    if holdout.get("holdout_rows_used_for_training_or_selection") is not False:
        raise RuntimeError("The evaluated holdout manifest does not certify holdout exclusion")
    run_manifest_path = ROOT / holdout["run_manifest_path"]
    evaluation_run = _read_json(run_manifest_path)
    if evaluation_run.get("status") != "completed":
        raise RuntimeError("Evaluated bundle training run is not complete")
    bundle_path = ROOT / holdout["evaluated_bundle_path"]
    freeze_path = ROOT / holdout["evaluated_bundle_freeze_path"]
    bundle = _read_json(bundle_path)
    freeze = _read_json(freeze_path)
    if (
        _sha256(bundle_path) != holdout.get("evaluated_bundle_sha256")
        or _sha256(freeze_path) != holdout.get("evaluated_bundle_freeze_sha256")
        or freeze.get("policy_sha256") != _sha256(ROOT / freeze["policy_path"])
    ):
        raise RuntimeError("Evaluated bundle or its development policy freeze has changed")
    policy_source = ROOT / freeze["policy_path"]
    if _sha256(policy_source) != freeze["policy_sha256"]:
        raise RuntimeError("Frozen development policy hash mismatch")

    raw_path = ROOT / CONFIG["research_config_path"]
    research_config = _read_json(raw_path)
    candles_path = ROOT / research_config["raw_data_path"]
    dataset_result = evaluation_run["stages"]["final_feature_dataset"]["result"]
    dataset_path = ROOT / dataset_result["dataset_path"]
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    source_training = _completed_stage_output(evaluation_run, "final_model", "training_manifest.json")
    training = _read_json(source_training)
    calibration_source = ROOT / bundle["calibrator_path"]
    if _sha256(calibration_source) != freeze["calibrator_sha256"]:
        raise RuntimeError("Evaluated calibrator differs from the frozen evaluated bundle")
    policy_identity = _sha256(policy_source)
    data_cutoff = pd.Timestamp(CONFIG["fit_label_availability_cutoff_inclusive_utc"])
    if data_cutoff.tzinfo is None:
        data_cutoff = data_cutoff.tz_localize("UTC")
    else:
        data_cutoff = data_cutoff.tz_convert("UTC")
    feature_order = list(bundle["feature_order"])
    if training.get("feature_order") != feature_order:
        raise RuntimeError("Frozen training manifest and evaluated bundle feature orders differ")
    identity = {
        "candidate_id": CONFIG["candidate_id"],
        "config_sha256": _sha256(CONFIG_PATH),
        "research_manifest_sha256": _sha256(research_manifest_path),
        "holdout_manifest_sha256": _sha256(holdout_manifest_path),
        "evaluated_bundle_sha256": _sha256(bundle_path),
        "evaluated_bundle_freeze_sha256": _sha256(freeze_path),
        "evaluated_training_manifest_sha256": _sha256(source_training),
        "raw_data_sha256": _sha256(candles_path),
        "feature_dataset_sha256": _sha256(dataset_path),
        "development_oof_calibration_sha256": freeze["oof_calibration_predictions_sha256"],
        "development_policy_sha256": policy_identity,
        "deployment_refit_code_sha256": _sha256(Path(__file__)),
        "fit_label_availability_cutoff_inclusive_utc": data_cutoff,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(MANIFEST_PATH) if MANIFEST_PATH.is_file() else {"schema_version": 1, "stages": {}}
    if manifest.get("input_identity") not in (None, identity):
        raise RuntimeError("Existing deployment candidate belongs to different frozen source artifacts")
    if manifest.get("status") == "completed":
        model = OUTPUT_DIR / manifest.get("model_path", "__missing__")
        if model.is_file() and _sha256(model) == manifest.get("model_sha256"):
            print("[deployment-refit] status=skipped (verified candidate checkpoint)", flush=True)
            return manifest
        raise RuntimeError("Completed deployment candidate model is missing or has changed")
    manifest.update({
        "schema_version": 1,
        "candidate_id": CONFIG["candidate_id"],
        "status": "running",
        "input_identity": identity,
        "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "activated": False,
        "live_orders_enabled": False,
        "independent_backtest": False,
    })
    _write_json(MANIFEST_PATH, manifest)

    columns = list(dict.fromkeys([
        "Opened", experiment.TARGET_COL, experiment.TARGET_AVAILABLE_COL,
        experiment.DECISION_COL, *feature_order,
    ]))
    frame = pd.read_parquet(dataset_path, columns=columns)
    opened = pd.to_datetime(frame["Opened"], utc=True, errors="raise")
    label_available = pd.to_datetime(frame[experiment.TARGET_AVAILABLE_COL], utc=True, errors="coerce")
    labels = pd.to_numeric(frame[experiment.TARGET_COL], errors="coerce")
    fit_start = pd.Timestamp(CONFIG["initial_history_start_utc"])
    if fit_start.tzinfo is None:
        fit_start = fit_start.tz_localize("UTC")
    eligible = (
        opened.ge(fit_start)
        & label_available.notna()
        & label_available.le(data_cutoff)
        & labels.notna()
    )
    fit_rows = np.flatnonzero(eligible.to_numpy())
    if not len(fit_rows):
        raise RuntimeError("No deployment rows have labels available by the frozen cutoff")
    decision_mask = frame[experiment.DECISION_COL].to_numpy(dtype=np.bool_, copy=False)[fit_rows]
    decision_weight = float(training["decision_weight"])
    sample_weight = experiment._preopen_sample_weights(decision_mask, decision_weight)
    x = frame.iloc[fit_rows].loc[:, feature_order].to_numpy(dtype=np.float32, copy=True)
    x[~np.isfinite(x)] = np.nan
    y = labels.iloc[fit_rows].to_numpy(dtype=np.int8, copy=True)
    model_path = OUTPUT_DIR / "deployment_candidate_model.txt"
    if model_path.is_file():
        checkpoint = manifest.get("stages", {}).get("model_refit", {})
        if checkpoint.get("status") == "completed":
            if _sha256(model_path) != checkpoint.get("model_sha256"):
                raise RuntimeError("Checkpointed deployment model checksum mismatch")
        else:
            recovered_model = experiment.lgb.Booster(model_file=str(model_path))
            if (
                recovered_model.feature_name() != feature_order
                or recovered_model.current_iteration() != int(training["best_iteration"])
            ):
                raise RuntimeError("Uncheckpointed deployment model is not a complete matching refit")
    else:
        model = experiment.lgb.train(
            dict(training["params"]),
            experiment.lgb.Dataset(
                x, label=y, weight=sample_weight,
                feature_name=feature_order, free_raw_data=True,
            ),
            num_boost_round=int(training["best_iteration"]),
        )
        temporary_model = model_path.with_suffix(".txt.tmp")
        model.save_model(str(temporary_model))
        temporary_model.replace(model_path)
    model_sha256 = _sha256(model_path)
    manifest["stages"]["model_refit"] = {
        "status": "completed",
        "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_path": model_path.resolve().relative_to(ROOT).as_posix(),
        "model_sha256": model_sha256,
        "training_rows": int(len(fit_rows)),
        "decision_rows": int(decision_mask.sum()),
        "latest_label_available_at_utc": label_available.iloc[fit_rows].max().isoformat(),
        "fit_label_availability_cutoff_inclusive_utc": data_cutoff,
        "frozen_best_iteration": int(training["best_iteration"]),
        "frozen_params_sha256": hashlib.sha256(
            json.dumps(training["params"], sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "frozen_feature_count": len(feature_order),
    }
    _write_json(MANIFEST_PATH, manifest)
    del frame, x, y, sample_weight

    candidate_calibrator = OUTPUT_DIR / "deployment_candidate_calibrator.json"
    candidate_policy = OUTPUT_DIR / "deployment_candidate_policy.json"
    calibration_sha256 = _copy_frozen_file(
        calibration_source, candidate_calibrator, freeze["calibrator_sha256"],
    )
    policy_sha256 = _copy_frozen_file(policy_source, candidate_policy, freeze["policy_sha256"])
    feature_order_path = OUTPUT_DIR / "feature_order.json"
    _write_json(feature_order_path, {"feature_order": feature_order})
    meta_path = OUTPUT_DIR / "deployment_candidate_meta.json"
    _write_json(meta_path, {
        "model_type": "LightGBM binary classifier with development walk-forward Platt calibration",
        "training_rows": manifest["stages"]["model_refit"]["training_rows"],
        "latest_label_available_at_utc": manifest["stages"]["model_refit"]["latest_label_available_at_utc"],
        "feature_count": len(feature_order),
        "model_sha256": model_sha256,
        "calibrator_sha256": calibration_sha256,
        "development_policy_sha256": policy_sha256,
        "holdout_included_in_final_model_fit": True,
        "holdout_included_in_calibration_or_policy_selection": False,
        "activated": False,
        "live_orders_enabled": False,
    })
    bundle_path = OUTPUT_DIR / "deployment_candidate_bundle.json"
    bundle = {
        "candidate_id": CONFIG["candidate_id"],
        "status": "deployment_candidate_not_activated",
        "model_path": model_path.resolve().relative_to(ROOT).as_posix(),
        "model_sha256": model_sha256,
        "calibrator_path": candidate_calibrator.resolve().relative_to(ROOT).as_posix(),
        "calibrator_sha256": calibration_sha256,
        "policy_path": candidate_policy.resolve().relative_to(ROOT).as_posix(),
        "policy_sha256": policy_sha256,
        "feature_order_path": feature_order_path.resolve().relative_to(ROOT).as_posix(),
        "feature_order": feature_order,
        "feature_sources_path": bundle["feature_sources_path"],
        "generator_settings_and_states_source_bundle": holdout["evaluated_bundle_path"],
        "evaluated_bundle_sha256": holdout["evaluated_bundle_sha256"],
        "full_history_fit_label_cutoff_inclusive_utc": data_cutoff,
        "fit_start_utc": fit_start,
        "development_oof_calibration_sha256": freeze["oof_calibration_predictions_sha256"],
        "development_policy_sha256": policy_sha256,
        "holdout_is_in_final_model_fit": True,
        "calibrator_or_policy_fit_from_in_sample_predictions": False,
        "activated": False,
        "live_orders_enabled": False,
    }
    _write_json(bundle_path, bundle)
    manifest.update({
        "status": "completed",
        "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_path": model_path.resolve().relative_to(ROOT).as_posix(),
        "model_sha256": model_sha256,
        "bundle_path": bundle_path.resolve().relative_to(ROOT).as_posix(),
        "bundle_sha256": _sha256(bundle_path),
        "activated": False,
        "live_orders_enabled": False,
    })
    _write_json(MANIFEST_PATH, manifest)
    print(f"[deployment-refit] completed candidate={bundle_path}", flush=True)
    return manifest


if __name__ == "__main__":
    run_deployment_refit()
