"""Freeze and prepare the one-time D holdout bundle using development-only data."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/research/btc_D_holdout_preparation_20261028.json"
PARENT_PROTOCOL_PATH = ROOT / "configs/research/btc_chronology_reconstruction_20261027_v7.json"
OUTPUT_ROOT = ROOT / "data/analysis/polymarket/BTC/btc_D_holdout_preparation_20261028"
HOLDOUT_START_UTC = "2026-07-01T00:00:00Z"
SAFE_OPENED_END_UTC = "2026-06-30T23:58:00Z"
SELECTION_CUTOFF_UTC = "2025-10-01T00:00:00Z"
FIT_CUTOFF_UTC = "2026-06-30T23:59:00Z"
EVALUATION_AUTHORIZED = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_frozen_code_hashes(protocol: dict) -> dict:
    paths = {
        "preparation_runner_sha256": Path(__file__),
        "evaluation_runner_sha256": ROOT / "run_btc_D_holdout_evaluation_20261028.py",
        "focused_test_sha256": ROOT / "tests/test_btc_D_holdout_preparation_20261028.py",
    }
    verified = {}
    for key, path in paths.items():
        expected = protocol["outputs"].get(key)
        actual = sha256_file(path)
        if not expected or actual != expected:
            raise RuntimeError(f"Frozen E0041 code hash differs: {path.relative_to(ROOT).as_posix()}")
        verified[path.relative_to(ROOT).as_posix()] = actual
    return verified


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _copy_safe_raw_prefix(source: Path, destination: Path) -> dict:
    """Copy only development candles; inspect only the timestamp at the stop row."""
    import pandas as pd

    stop = pd.Timestamp(SAFE_OPENED_END_UTC)
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_opened = None
    first_excluded_opened = None
    rows = 0
    # Unbuffered reads prevent the runtime from prefetching protected OHLCV bytes.
    with source.open("rb", buffering=0) as reader, destination.open("wb") as writer:
        header = reader.readline()
        if not header or header.split(b",", 1)[0].strip().lstrip(b"\xef\xbb\xbf") != b"Opened":
            raise RuntimeError("Frozen raw prefix must have Opened as its first CSV column")
        writer.write(header)
        while True:
            first_field = bytearray()
            while True:
                value = reader.read(1)
                if not value:
                    break
                if value == b",":
                    break
                if value in (b"\n", b"\r"):
                    raise RuntimeError("Raw-prefix row ended before its Opened delimiter")
                first_field.extend(value)
            if not first_field:
                break
            opened = pd.Timestamp(first_field.decode("ascii").strip())
            opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC")
            if last_opened is not None and opened <= last_opened:
                raise RuntimeError("Allowed raw-prefix Opened timestamps are not strictly increasing")
            if opened >= stop:
                first_excluded_opened = opened.isoformat()
                break
            remainder = reader.readline()
            if not remainder:
                raise RuntimeError("Allowed raw-prefix row is truncated")
            writer.write(bytes(first_field) + b"," + remainder)
            last_opened = opened
            rows += 1
    if rows <= 0 or last_opened is None:
        raise RuntimeError("Safe development raw prefix is empty")
    if first_excluded_opened is None:
        raise RuntimeError("Raw-prefix metadata did not contain the required development/holdout boundary")
    if pd.Timestamp(first_excluded_opened) < stop:
        raise RuntimeError("Raw-prefix ordering failed at the first excluded timestamp")
    return {
        "path": destination.resolve().relative_to(ROOT).as_posix(),
        "sha256": sha256_file(destination),
        "rows_copied": rows,
        "last_copied_opened_utc": last_opened.isoformat(),
        "first_excluded_opened_timestamp_only_utc": first_excluded_opened,
        "stop_opened_exclusive_utc": SAFE_OPENED_END_UTC,
        "holdout_ohlcv_rows_read": 0,
        "reader_policy": "Copied allowed rows only; read only the Opened field at the first excluded row and stopped before its OHLCV payload.",
    }


def _verify_parent_inputs_without_protected_raw(parent: dict) -> dict:
    """Verify the frozen E0039 implementation and safe inputs, skipping mixed F3 CSV bytes."""
    verified = {}
    f3_raw = parent["outer_folds"][-1]["raw_prefix_path"]
    for relative, expected in parent["frozen_inputs"].items():
        if relative == f3_raw:
            verified[relative] = {
                "sha256": expected,
                "verification": "previously frozen metadata; payload not rehashed because its final rows cross the market-start split",
            }
            continue
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen E0039 input is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen E0039 input hash differs: {relative}")
        verified[relative] = {"sha256": actual, "verification": "recomputed"}
    for relative, expected in parent["implementation_hashes"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"E0039 implementation hash differs from v7: {relative}")
        verified[relative] = {"sha256": actual, "verification": "recomputed"}
    return verified


def _assert_development_only(arrays: dict) -> None:
    import pandas as pd

    boundary = pd.Timestamp(HOLDOUT_START_UTC)
    if (arrays["market_start"] >= boundary).any():
        raise RuntimeError("A protected holdout market-start row entered the preparation arrays")


def _fit_final_D(arrays: dict, selection: dict, memory_names: list[str],
                 esn_names: dict[int, list[str]], model_root: Path,
                 *, deadline_monotonic: float) -> tuple:
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import run_btc_chronology_reconstruction_20261027 as chronology

    cutoff = pd.Timestamp(FIT_CUTOFF_UTC)
    train_mask = chronology.strict_training_mask(
        arrays["opened"], arrays["target"], arrays["target_available"], cutoff,
    )
    train_rows = np.flatnonzero(train_mask)
    if not len(train_rows) or np.unique(arrays["target"][train_rows]).size != 2:
        raise RuntimeError("Final D fit has insufficient eligible binary development labels")
    if not (arrays["target_available"][train_mask] < cutoff).all():
        raise RuntimeError("Final D training labels are not strictly before the exclusive cutoff")
    if not (arrays["opened"][train_mask] < cutoff).all():
        raise RuntimeError("Final D training timestamps reach the exclusive cutoff")
    # This is only a serialization-parity sample. It is development data and no score is calculated.
    parity_start = cutoff - pd.Timedelta(hours=24)
    parity_end = pd.Timestamp(HOLDOUT_START_UTC)
    expected, bundle = chronology.fit_and_predict_ensemble(
        arrays,
        procedure="D",
        fit_cutoff=cutoff,
        prediction_start=parity_start,
        prediction_end=parity_end,
        selection=selection,
        memory_names=memory_names,
        esn_names=esn_names,
        model_root=model_root,
        backend="gpu",
        threads=20,
        deadline_monotonic=deadline_monotonic,
        prediction_provenance="development_serialization_parity_only",
    )
    if expected.empty or not (expected.market_start_utc < pd.Timestamp(HOLDOUT_START_UTC)).all():
        raise RuntimeError("Final D model parity sample is empty or crosses into holdout")

    position = pd.Index(arrays["opened"]).get_indexer(pd.DatetimeIndex(expected.Opened))
    if (position < 0).any():
        raise RuntimeError("Final D parity timestamps do not align with development features")
    reloaded_members = {}
    for record in bundle["members"]:
        name = record["member"]
        reservoir_seed = int(name.split("_r", 1)[1].split("_", 1)[0])
        matrix = chronology._family_matrix(arrays, "D", reservoir_seed)[position]
        matrix = np.asarray(matrix, dtype=np.float32)
        matrix[~np.isfinite(matrix)] = np.nan
        model_path = ROOT / record["model_path"]
        if sha256_file(model_path) != record["model_sha256"]:
            raise RuntimeError(f"Final D member hash changed after fit: {name}")
        booster = lgb.Booster(model_file=str(model_path))
        if booster.num_feature() != matrix.shape[1]:
            raise RuntimeError(f"Final D saved member schema changed: {name}")
        reloaded_members[name] = np.asarray(booster.predict(matrix), dtype=np.float64)
    reloaded_mean = np.mean(list(reloaded_members.values()), axis=0)
    fitted_mean = expected.ensemble_probability_raw.to_numpy(dtype=np.float64)
    max_abs_difference = float(np.max(np.abs(fitted_mean - reloaded_mean)))
    if not np.allclose(fitted_mean, reloaded_mean, rtol=0.0, atol=2e-12):
        raise RuntimeError("Final D ensemble predictions changed after save/reload")
    return bundle, {
        "development_rows": int(len(expected)),
        "protected_market_rows": 0,
        "max_abs_prediction_difference_after_reload": max_abs_difference,
        "predictions_persisted": False,
        "predictions_scored": False,
    }


def run() -> dict:
    import pandas as pd
    import run_btc_chronology_reconstruction_20261027 as chronology

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["evaluation"]["evaluation_authorized"] is not False or EVALUATION_AUTHORIZED:
        raise RuntimeError("Preparation may not authorize the holdout evaluation")
    code_hashes = _verify_frozen_code_hashes(protocol)
    parent = json.loads(PARENT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if sha256_file(PARENT_PROTOCOL_PATH) != protocol["parent_e0039"]["protocol_sha256"]:
        raise RuntimeError("Frozen E0039 v7 protocol hash changed")
    started = time.perf_counter()
    resource = chronology._resource_snapshot(20.0, require_gpu=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    prefit_path = OUTPUT_ROOT / "pre_fit_gate.json"
    gate = {
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "frozen_code_hashes": code_hashes,
        "resources": resource,
        "gate_elapsed_seconds": time.perf_counter() - started,
        "required_free_workspace_disk_gb": 80.0,
        "holdout_rows_read": 0,
        "holdout_predictions_generated": 0,
        "status": "blocked_by_resource_gate" if resource["gate_failures"] else "passed",
        "checked_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _atomic_json(prefit_path, gate)
    if resource["gate_failures"]:
        return gate

    verification = _verify_parent_inputs_without_protected_raw(parent)
    source_fold = dict(parent["outer_folds"][-1])
    safe_raw_path = OUTPUT_ROOT / "development_prefix" / "raw_before_holdout.csv"
    safe_raw = _copy_safe_raw_prefix(ROOT / source_fold["raw_prefix_path"], safe_raw_path)
    source_fold["raw_prefix_path"] = safe_raw["path"]
    source_fold["raw_prefix_sha256"] = safe_raw["sha256"]
    source_fold["selection_cutoff_exclusive_utc"] = SELECTION_CUTOFF_UTC
    source_fold["id"] = "D_FINAL_SELECTION_20251001"

    flow_path = ROOT / source_fold["flow_prefix_path"]
    if sha256_file(flow_path) != source_fold["flow_prefix_sha256"]:
        raise RuntimeError("Frozen development flow prefix hash changed")
    if source_fold["market_start_test_end_exclusive_utc"] != HOLDOUT_START_UTC:
        raise RuntimeError("Final preparation must end at the fixed first holdout market start")

    run_identity = _canonical_hash({
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "parent_run_identity_sha256": protocol["parent_e0039"]["run_identity_sha256"],
        "safe_raw_prefix_sha256": safe_raw["sha256"],
        "flow_prefix_sha256": source_fold["flow_prefix_sha256"],
        "selection_cutoff_exclusive_utc": SELECTION_CUTOFF_UTC,
        "fit_cutoff_exclusive_utc": FIT_CUTOFF_UTC,
        "preparation_runner_sha256": sha256_file(Path(__file__)),
    })
    run_root = OUTPUT_ROOT / "runs" / run_identity[:16]
    run_root.mkdir(parents=True, exist_ok=True)
    fold_root = run_root / "D_FINAL_SELECTION_20251001"
    fold_identity = _canonical_hash({
        "run_identity_sha256": run_identity,
        "fold": source_fold,
        "parent_v7_sha256": protocol["parent_e0039"]["protocol_sha256"],
    })
    started_utc = pd.Timestamp.now(tz="UTC").isoformat()
    manifest_path = run_root / "preparation_manifest.json"
    manifest = {
        **gate,
        "status": "running",
        "run_identity_sha256": run_identity,
        "started_utc": started_utc,
        "frozen_input_verification": verification,
        "safe_development_raw_prefix": safe_raw,
        "flow_prefix_sha256": source_fold["flow_prefix_sha256"],
        "selection_cutoff_exclusive_utc": SELECTION_CUTOFF_UTC,
        "fit_cutoff_exclusive_utc": FIT_CUTOFF_UTC,
        "holdout_start_utc": HOLDOUT_START_UTC,
        "holdout_rows_materialized": 0,
    }
    _atomic_json(manifest_path, manifest)
    overall_deadline = time.monotonic() + int(protocol["budgets"]["maximum_wall_seconds"])
    try:
        remaining = max(1, int(overall_deadline - time.monotonic()))
        selection = chronology._run_fold_selection(
            parent, source_fold, fold_root, fold_identity,
            remaining_seconds=remaining,
        )
        dynamic_path, memory_names, esn_names = chronology._build_dynamic_feature_artifact(
            parent, source_fold, fold_root, fold_identity,
        )
        arrays = chronology._load_fold_arrays(selection, dynamic_path, memory_names, esn_names)
        _assert_development_only(arrays)
        if (arrays["target_available"] >= pd.Timestamp(FIT_CUTOFF_UTC)).all():
            raise RuntimeError("No eligible labels remain before the final fit cutoff")

        oof_root = run_root / "calibration_oof_D"
        oof_root.mkdir(parents=True, exist_ok=True)
        oof_predictions, _ = chronology.fit_and_predict_ensemble(
            arrays,
            procedure="D",
            fit_cutoff=SELECTION_CUTOFF_UTC,
            prediction_start=SELECTION_CUTOFF_UTC,
            prediction_end=HOLDOUT_START_UTC,
            selection=selection,
            memory_names=memory_names,
            esn_names=esn_names,
            model_root=oof_root / "models",
            backend="gpu",
            threads=20,
            deadline_monotonic=min(overall_deadline, time.monotonic() + 7200.0),
            prediction_provenance="fold_local_preselection_chronological_oof",
        )
        if (oof_predictions.market_start_utc >= pd.Timestamp(HOLDOUT_START_UTC)).any():
            raise RuntimeError("D calibration OOF predictions contain a holdout market")
        chronology.validate_oof_provenance(
            oof_predictions,
            fit_cutoff=SELECTION_CUTOFF_UTC,
            prediction_start=SELECTION_CUTOFF_UTC,
            prediction_end=HOLDOUT_START_UTC,
            expected_provenance="fold_local_preselection_chronological_oof",
        )
        if not (pd.to_datetime(oof_predictions.latest_training_label_available_at_utc, utc=True)
                < pd.Timestamp(SELECTION_CUTOFF_UTC)).all():
            raise RuntimeError("D calibration OOF training labels cross the selection cutoff")
        oof_path = oof_root / "predictions.parquet"
        oof_predictions.to_parquet(oof_path, index=False, compression="zstd")

        gamma = chronology.load_official_gamma(parent)
        if (gamma.market_start_utc >= pd.Timestamp(HOLDOUT_START_UTC)).any():
            raise RuntimeError("Development calibration reader returned a holdout Gamma row")
        calibration_history = chronology._join_gamma_outcomes(
            oof_predictions, gamma, source="fold_local_preselection_chronological_oof",
        )
        calibrator, calibration_status = chronology.fit_official_platt(
            calibration_history,
            as_of_utc=FIT_CUTOFF_UTC,
            minimum_rows=int(protocol["calibration"]["minimum_rows"]),
        )
        calibration_path = run_root / "official_platt_D.json"
        if calibrator is not None:
            _atomic_json(calibration_path, calibrator)

        final_model, parity = _fit_final_D(
            arrays, selection, memory_names, esn_names,
            run_root / "final_D_bundle" / "models",
            deadline_monotonic=min(overall_deadline, time.monotonic() + 17100.0),
        )
        model_files = []
        member_metadata_files = []
        for member in final_model["members"]:
            member_path = ROOT / member["model_path"]
            sidecar = member_path.with_suffix(".json")
            model_files.append({"path": member["model_path"], "sha256": sha256_file(member_path)})
            member_metadata_files.append({
                "path": sidecar.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(sidecar),
            })
        bundle_manifest = {
            "schema_version": 1,
            "experiment_id": "E0041",
            "run_identity_sha256": run_identity,
            "preparation_protocol_sha256": sha256_file(PROTOCOL_PATH),
            "preparation_runner_sha256": sha256_file(Path(__file__)),
            "status": "D_BUNDLE_READY" if calibrator is not None else "D_BUNDLE_READY_CALIBRATION_MISSING",
            "procedure": "D = BASE70 + memory PCA8 + ESN PCA8",
            "selection_cutoff_exclusive_utc": SELECTION_CUTOFF_UTC,
            "fit_cutoff_exclusive_utc": FIT_CUTOFF_UTC,
            "training_rows": final_model["training_rows"],
            "latest_training_label_available_at_utc": final_model["latest_training_label_available_at_utc"],
            "best_params": selection["best_params"],
            "decision_weight": selection["decision_weight"],
            "best_iteration": selection["best_iteration"],
            "base_feature_order": selection["model_feature_order"],
            "memory_feature_order": memory_names,
            "esn_feature_order_by_seed": {str(seed): names for seed, names in esn_names.items()},
            "lightgbm_seeds": [20261101, 20261102, 20261103],
            "reservoir_seeds": [20261023, 20261024, 20261025],
            "ensemble_identity_sha256": final_model["ensemble_identity_sha256"],
            "member_files": model_files,
            "member_metadata_files": member_metadata_files,
            "calibration": {
                "status": calibration_status["status"],
                "rows": calibration_status["rows"],
                "required_rows": calibration_status["required"],
                "calibrator": calibrator,
                "calibrator_path": calibration_path.resolve().relative_to(ROOT).as_posix() if calibrator else None,
                "calibrator_sha256": sha256_file(calibration_path) if calibrator else None,
                "history_path": oof_path.resolve().relative_to(ROOT).as_posix(),
                "history_sha256": sha256_file(oof_path),
                "official_gamma_development_rows": int(len(gamma)),
                "fallback": None,
                "frozen_through_holdout": True,
            },
            "source_hashes": {
                "parent_raw_prefix_registered_sha256": protocol["development_inputs"]["raw_prefix_sha256"],
                "safe_raw_prefix_sha256": safe_raw["sha256"],
                "flow_prefix_sha256": source_fold["flow_prefix_sha256"],
                "memory_transform_sha256": parent["transform_hashes"]["memory"],
                "esn_transform_sha256": parent["transform_hashes"]["esn"],
                "parent_v7_protocol_sha256": protocol["parent_e0039"]["protocol_sha256"],
            },
            "development_state": {
                "dynamic_features_path": dynamic_path.resolve().relative_to(ROOT).as_posix(),
                "dynamic_features_sha256": sha256_file(dynamic_path),
                "last_materialized_market_start_utc": pd.Timestamp(arrays["market_start"].max()).isoformat(),
                "holdout_rows_materialized": 0,
                "rebuild_from_development_prefix_only": True,
            },
            "save_load_parity": parity,
            "activated": False,
            "holdout_predictions_generated": 0,
            "evaluation_authorized": False,
        }
        bundle_path = run_root / "final_D_bundle" / "bundle_manifest.json"
        _atomic_json(bundle_path, bundle_manifest)
        _atomic_json(PREPARATION_ROOT / "final_D_bundle_manifest.json", bundle_manifest)
        manifest.update({
            "status": bundle_manifest["status"],
            "selection": selection,
            "calibration_status": calibration_status,
            "final_model": final_model,
            "save_load_parity": parity,
            "bundle_manifest_path": bundle_path.resolve().relative_to(ROOT).as_posix(),
            "bundle_manifest_sha256": sha256_file(bundle_path),
            "elapsed_seconds": time.perf_counter() - started,
            "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        })
        _atomic_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest.update({
            "status": "interrupted_or_failed_with_checkpoints",
            "failure": {"type": type(exc).__name__, "detail": str(exc)},
            "elapsed_seconds": time.perf_counter() - started,
            "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        })
        _atomic_json(manifest_path, manifest)
        raise


def main() -> None:
    result = run()
    print(json.dumps({
        "status": result["status"],
        "experiment_id": result["experiment_id"],
        "run_identity_sha256": result.get("run_identity_sha256"),
        "holdout_rows_materialized": result.get("holdout_rows_materialized", 0),
        "elapsed_seconds": result.get("elapsed_seconds", result.get("gate_elapsed_seconds")),
        "gate_failures": result.get("resources", {}).get("gate_failures", []),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
