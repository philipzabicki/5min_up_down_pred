"""Run the resumable, full-pipeline BTC development walk-forward folds."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd

import run_btc_preopen_experiment as experiment


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_preopen_walk_forward_20261011.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_dir"]
CONTRACT_DIR = OUTPUT_DIR / "contracts"
OOF_DIR = OUTPUT_DIR / "oof"
RUN_MANIFEST_PATH = OUTPUT_DIR / "walk_forward_manifest.json"
SEED_FOLD_ID = "calibration_seed"
DEVELOPMENT_FOLD_IDS = ("dev_2026q2_a", "dev_2026q2_b", "dev_2026q2_c")
TRAINING_CODE_PATHS = (
    "run_btc_preopen_experiment.py",
    "fit_indicators.py",
    "fit_reaction_profile.py",
    "fit_volume_profile.py",
    "optimize_target_weights.py",
    "select_features.py",
    "optimize_lgbm_hyperparameters.py",
    "train_lgbm.py",
    "create_modeling_dataset.py",
    "features/btc_preopen_contract.py",
    "features/btc_preopen_baseline.py",
    "utils/data.py",
    "utils/project_config.py",
    "utils/metrics.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc(value) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


def _fold_contract(base_contract: dict, fold: dict) -> dict:
    contract = copy.deepcopy(base_contract)
    split = contract["chronological_split_frozen_before_tuning"]
    first_market = _utc(fold["market_start_first_utc"])
    last_market = _utc(fold["market_start_last_inclusive_utc"])
    first_decision = first_market - pd.Timedelta(seconds=60)
    last_decision = last_market - pd.Timedelta(seconds=60)
    if "fit_end_exclusive_utc" in fold:
        fit_end = _utc(fold["fit_end_exclusive_utc"])
        calibration_start = _utc(fold["calibration_start_utc"])
    else:
        fit_end = first_decision + pd.Timedelta(nanoseconds=1)
        calibration_start = first_decision
    split.update({
        "fit_and_tuning_end_exclusive_utc": fit_end.isoformat().replace("+00:00", "Z"),
        "fit_label_available_through_inclusive_utc": (fit_end - pd.Timedelta(nanoseconds=1)).isoformat().replace("+00:00", "Z"),
        "calibration_start_utc": calibration_start.isoformat().replace("+00:00", "Z"),
        "calibration_end_exclusive_utc": first_decision.isoformat().replace("+00:00", "Z"),
        "first_external_test_nominal_decision_utc": first_decision.isoformat().replace("+00:00", "Z"),
        "first_external_test_market_start_utc": first_market.isoformat().replace("+00:00", "Z"),
        "external_test_market_start_last_inclusive_utc": last_market.isoformat().replace("+00:00", "Z"),
        "external_test_nominal_decision_last_utc": last_decision.isoformat().replace("+00:00", "Z"),
        "data_cutoff_exclusive_utc": CONFIG["data_cutoff_exclusive_utc"],
        "development_cutoff_exclusive_utc": CONFIG["development_cutoff_exclusive_utc"],
        "fold_id": fold["fold_id"],
        "calibration_source": fold["calibration_source"],
        "policy_selection_fold": bool(fold.get("policy_selection_fold", False)),
        "test_history_status": (
            "historically exposed: this official-market period and overlapping BTC observations were used in prior repository experiments; "
            "new fitted artifacts exclude outcomes from model fitting and do not use this period to revise parameters"
        ),
    })
    contract["experiment_id"] = f"{CONFIG['experiment_id']}_{fold['fold_id']}"
    contract["contract_version"] = f"btc_preopen_wf_{fold['fold_id']}"
    contract["research_fold"] = {
        "fold_id": fold["fold_id"],
        "market_start_first_utc": split["first_external_test_market_start_utc"],
        "market_start_last_inclusive_utc": split["external_test_market_start_last_inclusive_utc"],
        "latest_training_label_available_at_utc": split["fit_label_available_through_inclusive_utc"],
        "calibration_source": fold["calibration_source"],
    }
    return contract


def _oof_columns(frame: pd.DataFrame, fold_id: str, fit_label_available_through_utc) -> pd.DataFrame:
    required = [
        "Opened", "nominal_decision_at", "market_start_utc", experiment.TARGET_COL,
        experiment.TARGET_AVAILABLE_COL, "p_model_raw", "p_model_platt",
        "prediction_available_at_utc",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(f"Fold {fold_id} prediction output is missing columns: {missing}")
    result = frame[required].copy()
    result["fold_id"] = fold_id
    result["prediction_provenance"] = "full_pipeline_chronological_out_of_fold"
    result["fit_label_available_through_utc"] = _utc(fit_label_available_through_utc)
    for column in (
        "nominal_decision_at", "market_start_utc", experiment.TARGET_AVAILABLE_COL,
        "prediction_available_at_utc", "fit_label_available_through_utc",
    ):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    if result["fit_label_available_through_utc"].gt(result["nominal_decision_at"]).any():
        raise RuntimeError(f"Fold {fold_id} used labels that were unavailable at prediction time")
    latency = result["prediction_available_at_utc"] - result["nominal_decision_at"]
    if not latency.eq(pd.Timedelta(seconds=1)).all():
        raise RuntimeError(f"Fold {fold_id} violates the one-second decision-to-execution budget")
    return result.sort_values("nominal_decision_at", kind="stable").reset_index(drop=True)


def _completed_predictions(run_root: Path) -> tuple[Path, dict]:
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError(f"Training pipeline has not completed: {manifest_path}")
    outputs = manifest.get("stages", {}).get("external_evaluation", {}).get("outputs", [])
    prediction = next((ROOT / item["path"] for item in outputs if str(item["path"]).endswith(".parquet")), None)
    if prediction is None or not prediction.is_file():
        raise RuntimeError(f"Completed training run has no verified external predictions: {manifest_path}")
    record = next(item for item in outputs if str(item["path"]).endswith(".parquet"))
    if _sha256(prediction) != record["sha256"]:
        raise RuntimeError(f"Prediction checksum mismatch: {prediction}")
    return prediction, manifest


def _write_calibration_pool(previous_fold_paths: list[Path], target_path: Path, cutoff: pd.Timestamp) -> Path:
    if not previous_fold_paths:
        raise RuntimeError("An OOF calibration pool requires at least one completed earlier fold")
    frames = [pd.read_parquet(path) for path in previous_fold_paths]
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.loc[
        pooled["nominal_decision_at"].lt(cutoff)
        & pd.to_datetime(pooled[experiment.TARGET_AVAILABLE_COL], utc=True).le(cutoff)
    ].copy()
    if pooled.empty:
        raise RuntimeError(f"No earlier OOF labels are available by {cutoff}")
    if pooled["fit_label_available_through_utc"].gt(pooled["nominal_decision_at"]).any():
        raise RuntimeError("An OOF calibrator pool contains a model trained beyond its prediction decision")
    if pooled["fold_id"].astype(str).eq("evaluated_holdout").any():
        raise RuntimeError("Holdout predictions cannot enter a calibration pool")
    pooled.sort_values(["nominal_decision_at", "fold_id"], kind="stable", inplace=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_suffix(".parquet.tmp")
    pooled.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(target_path)
    return target_path


def run_development_walk_forward() -> dict:
    raw_path = ROOT / CONFIG["raw_data_path"]
    market_path = ROOT / CONFIG["official_markets_path"]
    source_contract = ROOT / CONFIG["source_contract_path"]
    for path in (raw_path, market_path, source_contract):
        if not path.is_file():
            raise FileNotFoundError(path)
    identity = {
        "experiment_id": CONFIG["experiment_id"],
        "config_sha256": _sha256(CONFIG_PATH),
        "raw_data_sha256": _sha256(raw_path),
        "official_markets_sha256": _sha256(market_path),
        "source_contract_sha256": _sha256(source_contract),
        "training_config_sha256": _sha256(ROOT / "configs/btc_preopen_training.json"),
        "training_code_sha256": {
            path: _sha256(ROOT / path) for path in TRAINING_CODE_PATHS
        },
        "walk_forward_code_sha256": _sha256(Path(__file__)),
        "data_cutoff_exclusive_utc": CONFIG["data_cutoff_exclusive_utc"],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    manifest = (
        json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
        if RUN_MANIFEST_PATH.is_file() else {"schema_version": 1, "folds": {}}
    )
    if manifest.get("input_identity") not in (None, identity):
        raise RuntimeError("Existing walk-forward output belongs to different frozen inputs")
    manifest["input_identity"] = identity
    manifest["status"] = "running"
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(RUN_MANIFEST_PATH, manifest)
    base_contract = json.loads(source_contract.read_text(encoding="utf-8"))
    base_profile = json.loads((ROOT / "configs/btc_preopen_training.json").read_text(encoding="utf-8"))["profiles"]["nightly_v1"]
    raw_relative = raw_path.resolve().relative_to(ROOT).as_posix()
    market_relative = market_path.resolve().relative_to(ROOT).as_posix()
    completed_fold_paths: list[Path] = []
    prediction_paths: dict[str, Path] = {}

    for fold in CONFIG["folds"]:
        fold_id = str(fold["fold_id"])
        if fold_id == "evaluated_holdout":
            continue
        contract = _fold_contract(base_contract, fold)
        contract_path = CONTRACT_DIR / f"{fold_id}.json"
        _write_json(contract_path, contract)
        split = contract["chronological_split_frozen_before_tuning"]
        first_decision = _utc(split["first_external_test_nominal_decision_utc"])
        oof_pool_path = OOF_DIR / f"calibration_before_{fold_id}.parquet"
        calibration_path = None
        previous_paths = list(completed_fold_paths)
        if fold_id != SEED_FOLD_ID:
            if not previous_paths:
                # Existing completed folds may be loaded from the manifest after a restart.
                for prior in [SEED_FOLD_ID, *DEVELOPMENT_FOLD_IDS]:
                    record = manifest.get("folds", {}).get(prior, {})
                    oof_path = ROOT / record.get("oof_path", "__missing__")
                    if prior != fold_id and record.get("status") == "completed" and oof_path.is_file():
                        previous_paths.append(oof_path)
            calibration_path = _write_calibration_pool(previous_paths, oof_pool_path, first_decision)
        contract_relative = contract_path.resolve().relative_to(ROOT).as_posix()
        profile = copy.deepcopy(base_profile)
        profile.update({
            "contract_path": contract_relative,
            "raw_data_path": raw_relative,
            "official_markets_path": market_relative,
            "prediction_compute_budget_seconds": 1,
            "output_dir": f"{CONFIG['output_dir']}/runs/{fold_id}",
        })
        if calibration_path is not None:
            profile["oof_calibration_predictions_path"] = calibration_path.resolve().relative_to(ROOT).as_posix()
        existing = manifest.get("folds", {}).get(fold_id, {})
        cached_run_root = ROOT / existing.get("run_root", "__missing__")
        cached_oof = ROOT / existing.get("oof_path", "__missing__")
        if (
            existing.get("status") == "completed"
            and existing.get("contract_sha256") == _sha256(contract_path)
            and existing.get("oof_sha256") == (_sha256(cached_oof) if cached_oof.is_file() else None)
            and cached_oof.is_file()
        ):
            prediction_path = cached_oof
            run_root = cached_run_root
            inner_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
            print(f"[walk-forward] fold={fold_id} status=skipped (verified outer checkpoint)", flush=True)
        else:
            result = experiment.run_training_pipeline(profile_override=profile)
            run_root = result["run_root"]
            source_predictions, inner_manifest = _completed_predictions(run_root)
            source_frame = pd.read_parquet(source_predictions)
            prediction_path = OOF_DIR / f"{fold_id}.parquet"
            normalized = _oof_columns(
                source_frame,
                fold_id,
                split["fit_label_available_through_inclusive_utc"],
            )
            normalized.to_parquet(prediction_path.with_suffix(".parquet.tmp"), index=False, compression="zstd")
            prediction_path.with_suffix(".parquet.tmp").replace(prediction_path)
            market_start = pd.to_datetime(normalized["market_start_utc"], utc=True)
            if market_start.empty or market_start.min() != _utc(fold["market_start_first_utc"]) or market_start.max() != _utc(fold["market_start_last_inclusive_utc"]):
                raise RuntimeError(f"Fold {fold_id} predictions do not cover the exact frozen interval")
            if calibration_path is not None:
                calibration_pool = pd.read_parquet(calibration_path)
                if calibration_pool["nominal_decision_at"].ge(first_decision).any():
                    raise RuntimeError(f"Fold {fold_id} calibration pool contains current/future predictions")
                if pd.to_datetime(
                    calibration_pool[experiment.TARGET_AVAILABLE_COL], utc=True,
                ).gt(first_decision).any():
                    raise RuntimeError(f"Fold {fold_id} calibration pool contains labels unavailable at its first decision")
        prediction_paths[fold_id] = prediction_path
        if fold_id not in completed_fold_paths:
            completed_fold_paths.append(prediction_path)
        manifest.setdefault("folds", {})[fold_id] = {
            "status": "completed",
            "contract_path": contract_path.resolve().relative_to(ROOT).as_posix(),
            "contract_sha256": _sha256(contract_path),
            "run_root": run_root.resolve().relative_to(ROOT).as_posix(),
            "run_status": inner_manifest.get("status"),
            "run_manifest_path": (run_root / "run_manifest.json").resolve().relative_to(ROOT).as_posix(),
            "prediction_source_path": inner_manifest.get("stages", {}).get("external_evaluation", {}).get("outputs", [{}])[0].get("path"),
            "oof_path": prediction_path.resolve().relative_to(ROOT).as_posix(),
            "oof_sha256": _sha256(prediction_path),
            "calibration_source": fold["calibration_source"],
            "calibration_pool_path": calibration_path.resolve().relative_to(ROOT).as_posix() if calibration_path else None,
            "calibration_pool_sha256": _sha256(calibration_path) if calibration_path else None,
            "latest_training_label_available_at_utc": contract["chronological_split_frozen_before_tuning"]["fit_label_available_through_inclusive_utc"],
        }
        manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        _write_json(RUN_MANIFEST_PATH, manifest)
        print(
            f"[walk-forward] fold={fold_id} rows={len(pd.read_parquet(prediction_path, columns=['market_start_utc'])):,} "
            f"oof_sha256={manifest['folds'][fold_id]['oof_sha256']}",
            flush=True,
        )

    dev_paths = [prediction_paths[SEED_FOLD_ID], *(prediction_paths[key] for key in DEVELOPMENT_FOLD_IDS)]
    holdout_first_decision = _utc("2026-07-01T00:00:00Z") - pd.Timedelta(seconds=60)
    final_pool_path = OOF_DIR / "calibration_before_evaluated_holdout.parquet"
    _write_calibration_pool(dev_paths, final_pool_path, holdout_first_decision)
    manifest["calibration_before_evaluated_holdout"] = {
        "path": final_pool_path.resolve().relative_to(ROOT).as_posix(),
        "sha256": _sha256(final_pool_path),
        "rows": int(pd.read_parquet(final_pool_path, columns=["p_model_raw"]).shape[0]),
        "latest_nominal_decision_at_utc": pd.to_datetime(pd.read_parquet(final_pool_path, columns=["nominal_decision_at"]).nominal_decision_at, utc=True).max().isoformat(),
        "latest_label_available_at_utc": pd.to_datetime(pd.read_parquet(final_pool_path, columns=[experiment.TARGET_AVAILABLE_COL])[experiment.TARGET_AVAILABLE_COL], utc=True).max().isoformat(),
        "holdout_rows_included": False,
    }
    manifest["status"] = "development_walk_forward_completed"
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(RUN_MANIFEST_PATH, manifest)
    return manifest


if __name__ == "__main__":
    run_development_walk_forward()
