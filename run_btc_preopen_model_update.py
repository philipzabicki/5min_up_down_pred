"""Run the frozen BTC pre-open feature and bounded model-search comparison."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_market_probability_experiment as probability
import run_btc_market_probability_nested_experiment as nested
import run_btc_preopen_experiment as experiment
import run_btc_preopen_fee_correction as fee_correction
import run_btc_preopen_research as research
import run_btc_preopen_signal_diagnostic as diagnostic
import run_btc_preopen_walk_forward as walk_forward


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_preopen_model_update_20261012.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_dir"]
CONTRACT_DIR = OUTPUT_DIR / "contracts"
VARIANT_DIR = OUTPUT_DIR / "variants"
MANIFEST_PATH = OUTPUT_DIR / "model_update_manifest.json"
FREEZE_PATH = OUTPUT_DIR / "development_model_freeze.json"
REPORT_PATH = ROOT / "reports/btc_preopen/btc_preopen_model_update_20261012.md"
TABLE_DIR = ROOT / "reports/btc_preopen/btc_preopen_model_update_20261012"
NESTED_FROZEN_PATH = ROOT / "reports/btc_preopen/btc_market_probability_nested_frozen_20261012.json"
PREVIOUS_ECONOMICS_PATH = ROOT / "reports/btc_preopen/btc_market_probability_nested_20261012/independent_order_economics.csv"
DEV_CUTOFF_UTC = pd.Timestamp(CONFIG["development_cutoff_exclusive_utc"])
DATA_CUTOFF_UTC = pd.Timestamp(CONFIG["data_cutoff_exclusive_utc"])
VARIANTS = tuple(item["name"] for item in CONFIG["variants"])
FORECAST_NAMES = ("A_frozen", *VARIANTS)
PROBABILITY_COLUMNS = {
    "A_frozen": "p_btc_calibrated",
    **{name: f"p_{name}" for name in VARIANTS},
}
PORTFOLIO_CAPITALS = tuple(float(value) for value in CONFIG["economic_evaluation"]["holdout_portfolio_starting_cash_usd"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_convert("UTC") if stamp.tzinfo else stamp.tz_localize("UTC")


def _resource_snapshot() -> dict:
    result = {"logical_processors": os.cpu_count()}
    try:
        import psutil

        memory = psutil.virtual_memory()
        result.update({
            "available_ram_bytes": int(memory.available),
            "total_ram_bytes": int(memory.total),
        })
    except ImportError:
        result["available_ram_bytes"] = None
        result["total_ram_bytes"] = None
    return result


def _validate_frozen_config() -> None:
    source_folds = walk_forward.CONFIG["folds"]
    if [item["fold_id"] for item in source_folds] != list(CONFIG["outer_folds"]):
        raise RuntimeError("Frozen model-update folds differ from the source walk-forward folds")
    if walk_forward.CONFIG["development_cutoff_exclusive_utc"] != CONFIG["development_cutoff_exclusive_utc"]:
        raise RuntimeError("The frozen development cutoff differs from the source walk-forward contract")
    if walk_forward.CONFIG["data_cutoff_exclusive_utc"] != CONFIG["data_cutoff_exclusive_utc"]:
        raise RuntimeError("The frozen data cutoff differs from the source walk-forward contract")
    search = CONFIG["model_search"]
    requested = int(search["requested_trials_per_variant_per_fold"])
    expected_total = requested * len(VARIANTS) * len(source_folds)
    if expected_total != int(search["total_requested_trials"]):
        raise RuntimeError("Frozen total trial budget does not match variants and outer folds")
    if int(search["minimum_successful_trials_per_variant_per_fold"]) > requested:
        raise RuntimeError("Frozen minimum successful trials exceeds the requested trial count")
    import optimize_lgbm_hyperparameters as model_tuning

    model_tuning.validate_lgbm_search_space(search["search_space"])
    if int(search["tpe_startup_trials"]) != 5:
        raise RuntimeError("The current model-tuning runner is frozen to five TPE startup trials")
    seed = search["seed_trial_params"]
    model_tuning.validate_seed_trial_params(seed, search["search_space"])
    frozen_selection = research._read_json(NESTED_FROZEN_PATH)
    chosen = frozen_selection["policy_selection"]["selected_policy"]
    expected = CONFIG["economic_evaluation"]["selected_policy"]
    if chosen["policy"] != expected or not np.isclose(float(chosen["minimum_net_ev_fraction"]), 0.01):
        raise RuntimeError("The selected fixed-$5 A/EV 1% control differs from the prior freeze")


def _source_inputs() -> tuple[dict, dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    fee_correction._configure_output_paths()
    source_research, source_walk, source_holdout = fee_correction._verify_reusable_sources()
    bundle_path = ROOT / CONFIG["frozen_baseline"]["bundle_path"]
    if not bundle_path.is_file() or _sha256(bundle_path) != CONFIG["frozen_baseline"]["bundle_sha256"]:
        raise RuntimeError("The frozen A bundle does not match the pre-registered model-update baseline")
    if source_walk.get("input_identity", {}).get("data_cutoff_exclusive_utc") != CONFIG["data_cutoff_exclusive_utc"]:
        raise RuntimeError("The frozen A walk-forward data cutoff differs from this experiment")
    correction_manifest = research._read_json(fee_correction.CORRECTION_DIR / "research_manifest.json")
    if correction_manifest.get("status") != "completed":
        raise RuntimeError("The accepted fee-corrected research inputs are incomplete")
    t59, t60, book_record = research._load_enriched_book_snapshots(correction_manifest)
    source_identity = {
        "configuration_sha256": _sha256(CONFIG_PATH),
        "source_contract_sha256": _sha256(ROOT / CONFIG["source_contract"]),
        "raw_data_sha256": _sha256(ROOT / CONFIG["raw_data_path"]),
        "official_market_index_sha256": _sha256(ROOT / CONFIG["official_markets_path"]),
        "baseline_bundle_sha256": _sha256(bundle_path),
        "source_walk_forward_manifest_sha256": _sha256(fee_correction.SOURCE_WALK_FORWARD_MANIFEST),
        "source_holdout_manifest_sha256": _sha256(fee_correction.SOURCE_HOLDOUT_MANIFEST),
        "corrected_t60_snapshot_sha256": book_record["t60_coverage_sha256"],
        "corrected_t59_snapshot_sha256": book_record["t59_coverage_sha256"],
        "development_cutoff_exclusive_utc": CONFIG["development_cutoff_exclusive_utc"],
        "data_cutoff_exclusive_utc": CONFIG["data_cutoff_exclusive_utc"],
        "training_code_sha256": {
            name: _sha256(ROOT / name) for name in (
                "run_btc_preopen_experiment.py",
                "run_btc_preopen_model_update.py",
                "run_btc_preopen_walk_forward.py",
                "run_btc_market_probability_experiment.py",
                "run_btc_market_probability_nested_experiment.py",
                "run_btc_preopen_signal_diagnostic.py",
                "run_btc_preopen_research.py",
                "run_btc_preopen_t59_reassessment.py",
                "optimize_lgbm_hyperparameters.py",
                "create_modeling_dataset.py",
                "features/btc_preopen_baseline.py",
                "features/btc_preopen_contract.py",
                "features/candle_features.py",
            )
        },
        "training_configuration_sha256": {
            name: _sha256(ROOT / name) for name in (
                "configs/btc_preopen_training.json",
                "configs/modeling.json",
            )
        },
    }
    return source_research, source_walk, source_holdout, t59, t60, correction_manifest, book_record, source_identity


def _baseline_model_audit(source_holdout: dict) -> dict:
    run_manifest_path = ROOT / source_holdout["run_manifest_path"]
    run_manifest = research._read_json(run_manifest_path)
    stages = run_manifest.get("stages", {})
    selection = stages.get("feature_selection", {}).get("result", {})
    tuning = stages.get("model_tuning", {}).get("result", {})
    dataset = stages.get("final_feature_dataset", {}).get("result", {})
    bundle_path = ROOT / source_holdout["evaluated_bundle_path"]
    bundle = research._read_json(bundle_path)
    training_manifest_path = ROOT / bundle["model_path"]
    training_manifest_path = training_manifest_path.parent / "training_manifest.json"
    training = research._read_json(training_manifest_path)
    return {
        "bundle_path": bundle_path.resolve().relative_to(ROOT).as_posix(),
        "bundle_sha256": _sha256(bundle_path),
        "model_feature_count": len(bundle["feature_order"]),
        "dataset_rows_before_fit_filters": int(dataset.get("rows", 0)),
        "training_rows": int(training["training_rows"]),
        "decision_rows": int(tuning.get("decision_rows", 0)),
        "best_iteration": int(training["best_iteration"]),
        "selector_candidate_count": int(selection.get("permutation_features_scored", 0)),
        "selector_topk_evaluations": int(selection.get("topk_evaluations", 0)),
        "selector_training_rows": int(selection.get("training_rows", 0)),
        "tuning_requested_trials": int(tuning.get("target_trials", 0)),
        "tuning_successful_trials": int(tuning.get("successful_trials", 0)),
        "tuning_best_iteration": int(tuning.get("best_iteration", 0)),
    }


def _expected_fold_rows(fold: dict) -> int:
    first = _utc(fold["market_start_first_utc"])
    last = _utc(fold["market_start_last_inclusive_utc"])
    span = last - first
    if span < pd.Timedelta(0) or span % pd.Timedelta(minutes=5):
        raise RuntimeError(f"Fold {fold['fold_id']} is not an inclusive five-minute market interval")
    return int(span / pd.Timedelta(minutes=5)) + 1


def _validate_fold_predictions(frame: pd.DataFrame, fold: dict) -> None:
    fold_id = str(fold["fold_id"])
    if frame.empty:
        raise RuntimeError(f"Fold {fold_id} produced no predictions")
    starts = pd.to_datetime(frame.market_start_utc, utc=True, errors="raise")
    first = _utc(fold["market_start_first_utc"])
    last = _utc(fold["market_start_last_inclusive_utc"])
    if starts.min() != first or starts.max() != last or starts.duplicated().any():
        raise RuntimeError(f"Fold {fold_id} does not exactly cover the frozen market-start range")
    if len(frame) != _expected_fold_rows(fold):
        raise RuntimeError(f"Fold {fold_id} prediction count differs from its five-minute calendar")
    decision = pd.to_datetime(frame.nominal_decision_at, utc=True, errors="raise")
    available = pd.to_datetime(frame["prediction_available_at_utc"], utc=True, errors="raise")
    label_available = pd.to_datetime(frame[experiment.TARGET_AVAILABLE_COL], utc=True, errors="raise")
    fit_label_available = pd.to_datetime(frame.fit_label_available_through_utc, utc=True, errors="raise")
    if not starts.eq(decision + pd.Timedelta(minutes=1)).all():
        raise RuntimeError(f"Fold {fold_id} market starts are not aligned to T-60 decisions")
    if not available.eq(decision + pd.Timedelta(seconds=1)).all():
        raise RuntimeError(f"Fold {fold_id} predictions do not meet the one-second T-59 deadline")
    if not label_available.eq(starts + pd.Timedelta(minutes=5)).all():
        raise RuntimeError(f"Fold {fold_id} proxy labels do not end at the five-minute market boundary")
    if fit_label_available.gt(decision).any():
        raise RuntimeError(f"Fold {fold_id} model fit boundary includes labels unavailable at its decision")
    if not np.isfinite(pd.to_numeric(frame.p_model_platt, errors="coerce")).all():
        raise RuntimeError(f"Fold {fold_id} calibrated predictions contain missing or non-finite values")


def _write_development_calibration_pool(previous_paths: list[Path], target_path: Path, cutoff: pd.Timestamp) -> Path:
    if not previous_paths:
        raise RuntimeError("A later fold requires earlier candidate OOF rows for calibration")
    result = walk_forward._write_calibration_pool(previous_paths, target_path, cutoff)
    pooled = pd.read_parquet(result)
    if pooled["fold_id"].astype(str).eq("evaluated_holdout").any():
        raise RuntimeError("Holdout predictions entered a development calibration pool")
    return result


def _run_variant_fold(
    *,
    variant: dict,
    fold: dict,
    base_contract: dict,
    base_profile: dict,
    raw_relative: str,
    markets_relative: str,
    previous_paths: list[Path],
    manifest: dict,
) -> dict:
    variant_name = str(variant["name"])
    fold_id = str(fold["fold_id"])
    output_path = VARIANT_DIR / variant_name / "oof" / f"{fold_id}.parquet"
    existing = manifest.get("folds", {}).get(fold_id, {}).get(variant_name, {})
    if (
        existing.get("status") == "completed"
        and output_path.is_file()
        and _sha256(output_path) == existing.get("prediction_sha256")
    ):
        return existing

    contract = walk_forward._fold_contract(base_contract, fold)
    contract_path = CONTRACT_DIR / f"{fold_id}.json"
    walk_forward._write_json(contract_path, contract)
    split = contract["chronological_split_frozen_before_tuning"]
    first_decision = _utc(split["first_external_test_nominal_decision_utc"])
    calibration_path = None
    if fold_id != walk_forward.SEED_FOLD_ID:
        calibration_path = _write_development_calibration_pool(
            previous_paths,
            VARIANT_DIR / variant_name / "calibration" / f"before_{fold_id}.parquet",
            first_decision,
        )

    profile = copy.deepcopy(base_profile)
    feature_budgets = CONFIG["feature_generator_budgets"]
    indicator_fit = dict(profile["indicator_fit"])
    indicator_fit.update({
        "population_size": int(feature_budgets["indicator_population_size"]),
        "generations": int(feature_budgets["indicator_generations"]),
        "workers": int(CONFIG["runtime_profile"]["threads"]),
    })
    selector_config = dict(profile["feature_selection"])
    selector_config.update({
        "max_sweep_evaluations": int(feature_budgets["selector_max_sweep_evaluations"]),
        "max_refinement_rounds": int(feature_budgets["selector_max_refinement_rounds"]),
        "permutation_feature_fraction": float(feature_budgets["selector_permutation_feature_fraction"]),
        "permutation_repeats": int(feature_budgets["selector_permutation_repeats"]),
    })
    stage_budgets = dict(profile["stage_runtime_budgets_seconds"])
    stage_budgets["model_tuning_and_fit"] = int(
        CONFIG["runtime_profile"]["per_model_tuning_and_fit_budget_seconds"]
    )
    profile.update({
        "contract_path": contract_path.resolve().relative_to(ROOT).as_posix(),
        "raw_data_path": raw_relative,
        "official_markets_path": markets_relative,
        "prediction_compute_budget_seconds": 1,
        "output_dir": f"{CONFIG['output_dir']}/shared_runs/{fold_id}",
        "threads": int(CONFIG["runtime_profile"]["threads"]),
        "compute_backend": CONFIG["runtime_profile"]["compute_backend"],
        "runtime_budget_seconds": int(CONFIG["runtime_profile"]["per_pipeline_runtime_budget_seconds"]),
        "model_trials": int(CONFIG["model_search"]["requested_trials_per_variant_per_fold"]),
        "minimum_successful_trials": int(CONFIG["model_search"]["minimum_successful_trials_per_variant_per_fold"]),
        "maximum_estimators": int(CONFIG["model_search"]["maximum_estimators"]),
        "early_stopping_rounds": int(CONFIG["model_search"]["early_stopping_rounds"]),
        "model_cv_folds": int(CONFIG["model_search"]["model_cv_folds"]),
        "seed": int(CONFIG["model_search"]["random_seed"]),
        "model_search_space_override": CONFIG["model_search"]["search_space"],
        "model_seed_trial_params_override": [CONFIG["model_search"]["seed_trial_params"]],
        "model_feature_candidate_features": list(variant["candidate_features"]),
        "indicator_fit": indicator_fit,
        "reaction_profile_trials": int(feature_budgets["reaction_profile_trials"]),
        "volume_profile_trials": int(feature_budgets["volume_profile_trials"]),
        "target_weight_candidates": int(feature_budgets["target_weight_candidates"]),
        "feature_selection": selector_config,
        "stage_runtime_budgets_seconds": stage_budgets,
    })
    if calibration_path is not None:
        profile["oof_calibration_predictions_path"] = calibration_path.resolve().relative_to(ROOT).as_posix()

    started = time.perf_counter()
    result = experiment.run_training_pipeline(profile_name=variant_name, profile_override=profile)
    prediction_source, pipeline_manifest = walk_forward._completed_predictions(result["run_root"])
    source_frame = pd.read_parquet(prediction_source)
    normalized = walk_forward._oof_columns(
        source_frame,
        fold_id,
        split["fit_label_available_through_inclusive_utc"],
    )
    _validate_fold_predictions(normalized, fold)
    if calibration_path is not None:
        calibration_pool = pd.read_parquet(calibration_path)
        if calibration_pool.nominal_decision_at.ge(first_decision).any():
            raise RuntimeError(f"Fold {fold_id} calibration contains current or future predictions")
        if pd.to_datetime(calibration_pool[experiment.TARGET_AVAILABLE_COL], utc=True).gt(first_decision).any():
            raise RuntimeError(f"Fold {fold_id} calibration contains labels unavailable at its first decision")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.tmp")
    normalized.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output_path)
    model_bundle_path = result["bundle_path"]
    bundle = json.loads(model_bundle_path.read_text(encoding="utf-8"))
    training_manifest_path = ROOT / bundle["model_path"]
    training_manifest_path = training_manifest_path.parent / "training_manifest.json"
    training_record = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    calibration_record = json.loads((ROOT / bundle["calibrator_path"]).read_text(encoding="utf-8"))
    latest_training_label = _utc(training_record["latest_label_available_at"])
    if latest_training_label > _utc(split["fit_label_available_through_inclusive_utc"]):
        raise RuntimeError(f"Fold {fold_id} final model used a label beyond its frozen fit boundary")
    tuning_record = pipeline_manifest.get("stages", {}).get("model_tuning", {}).get("result", {})
    compact_tuning = {
        key: tuning_record.get(key)
        for key in ("target_trials", "successful_trials", "best_trial_number", "best_iteration", "best_value", "best_params", "training_rows", "decision_rows")
        if key in tuning_record
    }
    record = {
        "status": "completed",
        "fold_id": fold_id,
        "variant": variant_name,
        "prediction_path": output_path.resolve().relative_to(ROOT).as_posix(),
        "prediction_sha256": _sha256(output_path),
        "prediction_source_path": prediction_source.resolve().relative_to(ROOT).as_posix(),
        "prediction_source_sha256": _sha256(prediction_source),
        "bundle_path": model_bundle_path.resolve().relative_to(ROOT).as_posix(),
        "bundle_sha256": _sha256(model_bundle_path),
        "contract_path": contract_path.resolve().relative_to(ROOT).as_posix(),
        "contract_sha256": _sha256(contract_path),
        "generator_fit_start_utc": split["feature_and_model_fit_start_utc"],
        "generator_fit_end_exclusive_utc": split["fit_and_tuning_end_exclusive_utc"],
        "model_fit_start_utc": split["feature_and_model_fit_start_utc"],
        "model_fit_end_exclusive_utc": split["fit_and_tuning_end_exclusive_utc"],
        "prediction_market_start_first_utc": split["first_external_test_market_start_utc"],
        "prediction_market_start_last_inclusive_utc": split["external_test_market_start_last_inclusive_utc"],
        "calibration_start_decision_at_utc": calibration_record.get("calibration_start_decision_at"),
        "calibration_end_decision_at_utc": calibration_record.get("calibration_end_decision_at"),
        "calibration_source": calibration_record.get("calibration_source"),
        "calibration_training_rows": int(calibration_record.get("training_rows", 0)),
        "calibration_latest_label_available_at_utc": calibration_record.get("latest_label_available_at"),
        "dataset_rows_before_fit_filters": int(
            pipeline_manifest.get("stages", {}).get("final_feature_dataset", {}).get("result", {}).get("rows", 0)
        ),
        "training_manifest_path": training_manifest_path.resolve().relative_to(ROOT).as_posix(),
        "fit_and_tuning_end_exclusive_utc": split["fit_and_tuning_end_exclusive_utc"],
        "latest_training_label_available_at_utc": latest_training_label.isoformat(),
        "fit_rows": int(training_record["training_rows"]),
        "decision_rows": int(tuning_record.get("decision_rows", 0)),
        "auxiliary_fit_rows": int(training_record["training_rows"] - tuning_record.get("decision_rows", 0)),
        "model_feature_count": int(len(bundle["feature_order"])),
        "model_feature_order": bundle["feature_order"],
        "selector_feature_count": int(len(training_record.get("selector_feature_order", bundle["feature_order"]))),
        "added_candidate_features": training_record.get("added_candidate_features", []),
        "model_tuning": compact_tuning,
        "calibration_pool_path": calibration_path.resolve().relative_to(ROOT).as_posix() if calibration_path else None,
        "calibration_pool_sha256": _sha256(calibration_path) if calibration_path else None,
        "run_root": result["run_root"].resolve().relative_to(ROOT).as_posix(),
        "pipeline_status": pipeline_manifest.get("status"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest.setdefault("folds", {}).setdefault(fold_id, {})[variant_name] = record
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(MANIFEST_PATH, manifest)
    print(
        f"[btc-model-update] fold={fold_id} variant={variant_name} rows={len(normalized):,} "
        f"features={record['model_feature_count']} trials={compact_tuning.get('successful_trials')} "
        f"elapsed={record['elapsed_seconds'] / 60.0:.1f}m",
        flush=True,
    )
    return record


def _load_candidate_predictions(manifest: dict, variant_name: str, fold_ids: list[str]) -> pd.DataFrame:
    frames = []
    for fold_id in fold_ids:
        record = manifest.get("folds", {}).get(fold_id, {}).get(variant_name, {})
        path = ROOT / record.get("prediction_path", "__missing__")
        if record.get("status") != "completed" or not path.is_file() or _sha256(path) != record.get("prediction_sha256"):
            raise RuntimeError(f"Missing or changed {variant_name}/{fold_id} OOF predictions")
        frames.append(pd.read_parquet(path))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["market_start_utc"] = pd.to_datetime(combined.market_start_utc, utc=True, errors="raise")
    if combined.market_start_utc.duplicated().any():
        raise RuntimeError(f"{variant_name} prediction folds overlap")
    return combined.sort_values("market_start_utc", kind="stable").reset_index(drop=True)


def _merge_predictions(base_data: pd.DataFrame, candidate_predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    result = base_data.copy()
    result["market_start_utc"] = pd.to_datetime(result.market_start_utc, utc=True, errors="raise")
    if experiment.TARGET_AVAILABLE_COL not in result:
        result[experiment.TARGET_AVAILABLE_COL] = result.market_start_utc + pd.Timedelta(minutes=5)
    for name, predictions in candidate_predictions.items():
        columns = [
            "market_start_utc", "nominal_decision_at", experiment.TARGET_COL,
            experiment.TARGET_AVAILABLE_COL, "p_model_raw", "p_model_platt", "fold_id",
        ]
        attach = predictions.loc[:, columns].copy()
        attach.rename(columns={
            "nominal_decision_at": f"decision_{name}",
            experiment.TARGET_COL: f"target_{name}",
            experiment.TARGET_AVAILABLE_COL: f"target_available_{name}",
            "p_model_raw": f"p_raw_{name}",
            "p_model_platt": f"p_{name}",
            "fold_id": f"fold_{name}",
        }, inplace=True)
        result = result.merge(attach, on="market_start_utc", how="left", validate="one_to_one")
        if result.loc[result.common_sample, f"p_{name}"].isna().any():
            raise RuntimeError(f"{name} predictions do not cover the frozen common market sample")
        if not result.loc[result.common_sample, "nominal_decision_at"].eq(result.loc[result.common_sample, f"decision_{name}"]).all():
            raise RuntimeError(f"{name} decision timestamps do not align with frozen A")
        if not result.loc[result.common_sample, "target_binance_proxy_up"].eq(
            result.loc[result.common_sample, f"target_{name}"]
        ).all():
            raise RuntimeError(f"{name} training-target labels differ from frozen A on the common sample")
        if not result.loc[result.common_sample, f"target_available_{name}"].eq(
            result.loc[result.common_sample, experiment.TARGET_AVAILABLE_COL]
        ).all():
            raise RuntimeError(f"{name} target-label availability differs from frozen A")
        if not result.loc[result.common_sample, "fold_id"].astype(str).eq(
            result.loc[result.common_sample, f"fold_{name}"].astype(str)
        ).all():
            raise RuntimeError(f"{name} OOF fold provenance differs from frozen A")
    return result


def _metric_rows(frame: pd.DataFrame, period: str, model_name: str, probability_column: str, edges: list[float]) -> tuple[list[dict], list[dict]]:
    rows, bins = [], []
    for target_name, target_column in (
        ("official_polymarket_outcome", "official_outcome_up"),
        ("binance_index_price_proxy", "target_binance_proxy_up"),
    ):
        sample = frame.loc[
            frame[target_column].isin((0.0, 1.0))
            & np.isfinite(pd.to_numeric(frame[probability_column], errors="coerce"))
        ].copy()
        if sample.empty:
            continue
        scored = sample.copy()
        scored["official_outcome_up"] = pd.to_numeric(scored[target_column], errors="raise").astype(np.int8)
        metric, bin_rows = probability._metric_row(scored, probability_column, model_name, period, edges)
        metric["target"] = target_name
        for item in bin_rows:
            item["target"] = target_name
        rows.append(metric)
        bins.extend(bin_rows)
    return rows, bins


def _target_alignment(frame: pd.DataFrame) -> dict:
    paired = frame.loc[
        frame.official_outcome_up.isin((0.0, 1.0))
        & frame.target_binance_proxy_up.isin((0.0, 1.0))
    ]
    matches = int(paired.official_outcome_up.eq(paired.target_binance_proxy_up).sum())
    count = int(len(paired))
    return {
        "n": count,
        "matches": matches,
        "disagreements": count - matches,
        "match_rate": float(matches / count) if count else None,
    }


def _paired_block_interval(frame: pd.DataFrame, left_column: str, right_column: str, *, seed: int, replicates: int, alpha: float = 0.05) -> dict:
    ordered = frame.sort_values("market_start_utc", kind="stable")
    y = ordered.official_outcome_up.to_numpy(dtype=np.float64)
    left = np.clip(ordered[left_column].to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
    right = np.clip(ordered[right_column].to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
    deltas = {
        "log_loss": (
            -(y * np.log(left) + (1.0 - y) * np.log1p(-left))
            + (y * np.log(right) + (1.0 - y) * np.log1p(-right))
        ),
        "brier": np.square(left - y) - np.square(right - y),
    }
    days = pd.to_datetime(ordered.market_start_utc, utc=True).dt.floor("D")
    full_days = pd.date_range(days.min(), days.max(), freq="D", tz="UTC")
    day_count = len(full_days)
    block_count = int(np.ceil(day_count / 3.0))
    daily_count = np.asarray([int(days.eq(day).sum()) for day in full_days], dtype=np.float64)
    daily_sums = {}
    for metric, values in deltas.items():
        by_day = pd.DataFrame({"day": days.to_numpy(), "delta": values}).groupby("day", sort=True).delta.sum()
        daily_sums[metric] = np.asarray([float(by_day.get(day, 0.0)) for day in full_days], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    draws = {metric: np.empty(int(replicates), dtype=np.float64) for metric in deltas}
    for index in range(int(replicates)):
        starts = rng.integers(0, day_count, size=block_count)
        positions = np.concatenate([(np.arange(start, start + 3) % day_count) for start in starts])[:day_count]
        denominator = float(daily_count[positions].sum())
        if denominator <= 0.0:
            raise RuntimeError("Paired bootstrap sampled no development markets")
        for metric in deltas:
            draws[metric][index] = float(daily_sums[metric][positions].sum() / denominator)
    confidence = 1.0 - float(alpha)
    lower_q, upper_q = (1.0 - confidence) / 2.0, 1.0 - (1.0 - confidence) / 2.0
    return {
        "n": int(len(ordered)),
        "log_loss_difference_left_minus_right": float(deltas["log_loss"].mean()),
        "log_loss_difference_95pct": np.quantile(draws["log_loss"], [0.025, 0.975]).tolist(),
        "log_loss_difference_adjusted_interval": np.quantile(draws["log_loss"], [lower_q, upper_q]).tolist(),
        "brier_difference_left_minus_right": float(deltas["brier"].mean()),
        "brier_difference_95pct": np.quantile(draws["brier"], [0.025, 0.975]).tolist(),
        "confidence_level": confidence,
        "method": f"paired circular 3-day UTC blocks; {int(replicates)} replicates; seed {int(seed)}",
    }


def _development_freeze(dev: pd.DataFrame, baseline_identity: dict, candidate_predictions: dict[str, pd.DataFrame]) -> dict:
    config = CONFIG["development_model_freeze"]
    scored = dev.loc[
        dev.common_sample
        & dev.official_outcome_up.isin((0.0, 1.0))
        & dev.market_start_utc.lt(DEV_CUTOFF_UTC)
    ].copy()
    if scored.empty or not scored.fold_id.notna().all():
        raise RuntimeError("Development model freeze lacks complete causal OOF rows")
    for name in VARIANTS:
        if scored[f"p_{name}"].isna().any():
            raise RuntimeError(f"Development freeze is missing {name} probabilities")
    edges = probability._metric_edges(scored.p_btc_calibrated.to_numpy(dtype=np.float64))
    ranking = []
    bins = []
    for name in FORECAST_NAMES:
        column = PROBABILITY_COLUMNS[name]
        metrics, model_bins = _metric_rows(scored, "development_oof", name, column, edges)
        official = next(row for row in metrics if row["target"] == "official_polymarket_outcome")
        ranking.append({"forecast": name, **official})
        bins.extend(model_bins)
    comparisons = {}
    for name in VARIANTS:
        comparisons[f"{name}-A_frozen"] = _paired_block_interval(
            scored,
            PROBABILITY_COLUMNS[name],
            PROBABILITY_COLUMNS["A_frozen"],
            seed=int(config["bootstrap"].split("seed ")[-1]),
            replicates=2000,
            alpha=0.025,
        )
    candidate_ranking = sorted(
        (row for row in ranking if row["forecast"] in VARIANTS),
        key=lambda row: (row["log_loss"], row["forecast"]),
    )
    best_candidate = candidate_ranking[0]
    best_delta = comparisons[f"{best_candidate['forecast']}-A_frozen"]
    selected = best_candidate["forecast"] if best_delta["log_loss_difference_adjusted_interval"][1] < 0.0 else "A_frozen"
    payload = {
        "status": "frozen_before_model_update_holdout_evaluation",
        "experiment_id": CONFIG["experiment_id"],
        "configuration_sha256": _sha256(CONFIG_PATH),
        "baseline_prediction_identity": baseline_identity,
        "development_cutoff_exclusive_utc": CONFIG["development_cutoff_exclusive_utc"],
        "development_rows_common": int(len(scored)),
        "development_start_utc": scored.market_start_utc.min().isoformat(),
        "development_end_utc": scored.market_start_utc.max().isoformat(),
        "development_fold_rows": {
            str(fold_id): int(len(group)) for fold_id, group in scored.groupby("fold_id", sort=True)
        },
        "quality_ranking_official_outcome": ranking,
        "paired_comparisons_against_frozen_A": comparisons,
        "selection_rule": config["development_model_freeze"],
        "selected_model": selected,
        "selected_policy": {
            "name": CONFIG["economic_evaluation"]["selected_policy"],
            "hurdle_fraction": float(CONFIG["economic_evaluation"]["portfolio_policy_hurdle"]),
            "source": "previously frozen nested development policy; not reselected here",
        },
        "holdout_used_for_model_selection": False,
        "holdout_used_for_policy_selection": False,
    }
    existing = research._read_json(FREEZE_PATH) if FREEZE_PATH.is_file() else None
    if existing is not None:
        for key in ("configuration_sha256", "baseline_prediction_identity", "selected_model", "development_rows_common"):
            if existing.get(key) != payload.get(key):
                raise RuntimeError("Existing development freeze differs from the verified frozen inputs")
        payload = existing
    else:
        _write_json(FREEZE_PATH, payload)
    return payload


def _holdout_portfolios(
    frame: pd.DataFrame,
    selected_names: tuple[str, ...],
    t59_by_id: dict,
    t60_by_id: dict,
    market_by_id: dict,
    hurdle: float,
) -> list[dict]:
    output = []
    holdout = frame.loc[
        frame.common_sample
        & frame.official_outcome_up.isin((0.0, 1.0))
        & frame.market_start_utc.ge(DEV_CUTOFF_UTC)
        & frame.market_start_utc.lt(DATA_CUTOFF_UTC)
    ].copy()
    for name in selected_names:
        candidate_rows = holdout.copy()
        candidate_rows["p_candidate_raw"] = candidate_rows[f"p_raw_{name}"] if name != "A_frozen" else candidate_rows.p_btc_raw
        candidate_rows["p_candidate_platt"] = candidate_rows[PROBABILITY_COLUMNS[name]]
        candidate_markets = {
            str(row.condition_id).lower(): {
                **market_by_id[str(row.condition_id).lower()],
                "p_candidate_raw": float(row.p_candidate_raw),
                "p_candidate_platt": float(row.p_candidate_platt),
            }
            for row in candidate_rows.itertuples(index=False)
        }
        for capital in PORTFOLIO_CAPITALS:
            summary, trades, decisions, curve = diagnostic._run_holdout_portfolio(
                capital,
                candidate_rows,
                t59_by_id,
                candidate_markets,
                t60_by_id,
                hurdle,
            )
            candidate_dir = OUTPUT_DIR / "holdout_portfolios" / name / str(int(capital))
            candidate_dir.mkdir(parents=True, exist_ok=True)
            if trades:
                pd.DataFrame(trades).to_parquet(candidate_dir / "trades.parquet", index=False, compression="zstd")
            if decisions:
                pd.DataFrame(decisions).to_parquet(candidate_dir / "decisions.parquet", index=False, compression="zstd")
            pd.DataFrame(curve).to_csv(candidate_dir / "equity_path.csv", index=False)
            _write_json(candidate_dir / "summary.json", summary)
            output.append({
                "forecast": name,
                "starting_capital_usd": capital,
                "trade_count": int(summary.get("trade_count", 0)),
                "gross_turnover_usd": float(summary.get("gross_turnover_usd", 0.0)),
                "fees_paid_estimated_usd": float(summary.get("fees_paid_estimated_usd", 0.0)),
                "net_pnl_usd": float(summary.get("net_pnl_usd", 0.0)),
                "ending_cash_after_settlement_usd": float(summary.get("ending_cash_after_settlement_usd", capital)),
                "max_drawdown_cost_basis_equity": float(summary.get("max_drawdown_cost_basis_equity", 0.0)),
                "accounting_audit_status": summary.get("accounting_audit", {}).get("status"),
                "summary_path": (candidate_dir / "summary.json").resolve().relative_to(ROOT).as_posix(),
            })
    return output


def _verify_previous_selected_policy_economics(holdout_economics: list[dict]) -> dict:
    if not PREVIOUS_ECONOMICS_PATH.is_file():
        raise FileNotFoundError(PREVIOUS_ECONOMICS_PATH)
    previous = pd.read_csv(PREVIOUS_ECONOMICS_PATH)
    existing = previous.loc[
        previous.period.eq("holdout_diagnostic")
        & previous.forecast.eq("A_BTC_calibrated")
        & np.isclose(previous.ev_hurdle_fraction.astype(float), 0.01)
    ]
    if len(existing) != 1:
        raise RuntimeError("The previously reported A/EV 1% holdout row is missing or ambiguous")
    current = next(
        row for row in holdout_economics
        if row["forecast"] == "A_frozen" and np.isclose(float(row["ev_hurdle_fraction"]), 0.01)
    )
    original = existing.iloc[0].to_dict()
    checks = ("signals", "fills", "active_utc_days")
    for key in checks:
        if int(original[key]) != int(current[key]):
            raise RuntimeError(f"Reused A/EV 1% holdout result differs in {key}")
    value_checks = (
        ("gross_turnover_usd", "gross_turnover_usd"),
        ("fee_equivalent_usd", "fee_equivalent_usd"),
        ("expected_net_t59_usd", "expected_net_t59_usd"),
        ("realized_net_pnl_usd", "realized_net_pnl_usd"),
    )
    for old_key, new_key in value_checks:
        if not np.isclose(float(original[old_key]), float(current[new_key]), rtol=0.0, atol=1e-7):
            raise RuntimeError(f"Reused A/EV 1% holdout result differs in {old_key}")
    return {
        "status": "verified_match",
        "source_path": PREVIOUS_ECONOMICS_PATH.resolve().relative_to(ROOT).as_posix(),
        "source_sha256": _sha256(PREVIOUS_ECONOMICS_PATH),
        "previous_row": {key: original[key] for key, _ in value_checks} | {
            key: int(original[key]) for key in checks
        },
        "recomputed_row": {
            key: current[key] for _, key in value_checks
        } | {key: int(current[key]) for key in checks},
    }


def _write_report(payload: dict) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "probability_quality.csv": payload["probability_quality"],
        "paired_probability_differences.json": payload["paired_probability_differences"],
        "fold_probability_quality.csv": payload["fold_probability_quality"],
        "monthly_probability_quality.csv": payload["monthly_probability_quality"],
        "independent_order_economics.csv": payload["independent_order_economics"],
        "monthly_independent_order_economics.csv": payload["monthly_independent_order_economics"],
        "development_policy_fold_stability.csv": payload["development_policy_fold_stability"],
        "holdout_portfolios.csv": payload["holdout_portfolios"],
        "model_fold_training.csv": payload["model_fold_training"],
        "fold_periods.csv": payload["fold_periods"],
    }
    for name, rows in tables.items():
        destination = TABLE_DIR / name
        if name.endswith(".json"):
            _write_json(destination, rows)
        else:
            pd.DataFrame(rows).to_csv(destination, index=False)
    model_rows = payload["model_fold_training"]
    fold_lines = ["| Variant | Fold | Dataset rows | Fit rows | Decision rows | Features selected/model | Added candidates | Trials completed | Best iteration | Latest training label |", "|---|---|---:|---:|---:|---:|---|---:|---:|---|"]
    for row in model_rows:
        fold_lines.append(
            f"| {row['variant']} | {row['fold_id']} | {row['dataset_rows_before_fit_filters']:,} | {row['fit_rows']:,} "
            f"| {row['decision_rows']:,} | {row['selector_feature_count']}/{row['model_feature_count']} "
            f"| {', '.join(row['added_candidate_features']) or 'none'} "
            f"| {row.get('successful_trials', 0)} / {row.get('target_trials', 0)} | {row.get('best_iteration', 'n/a')} "
            f"| {row['latest_training_label_available_at_utc']} |"
        )
    period_lines = ["| Variant | Fold | Generator/model fit interval UTC | Calibration decision interval UTC | Calibration source / rows | Prediction market interval UTC |", "|---|---|---|---|---|---|"]
    for row in payload["fold_periods"]:
        period_lines.append(
            f"| {row['variant']} | {row['fold_id']} | {row['fit_start_utc']} to < {row['fit_end_exclusive_utc']} "
            f"| {row['calibration_start_decision_at_utc']} to {row['calibration_end_decision_at_utc']} "
            f"| {row['calibration_source']} / {row['calibration_training_rows']:,} "
            f"| {row['prediction_market_start_first_utc']} to {row['prediction_market_start_last_inclusive_utc']} |"
        )
    quality = payload["probability_quality"]
    dev_quality = [row for row in quality if row["period"] == "development_oof" and row["target"] == "official_polymarket_outcome"]
    hold_quality = [row for row in quality if row["period"] == "holdout_diagnostic" and row["target"] == "official_polymarket_outcome"]
    quality_lines = ["| Period | Forecast | n | Log loss | Brier | AUC | ECE |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in [*dev_quality, *hold_quality]:
        quality_lines.append(
            f"| {row['period']} | {row['forecast']} | {row['n']:,} | {row['log_loss']:.6f} "
            f"| {row['brier']:.6f} | {row['auc']:.4f} | {row['ece_frozen_dev_bins']:.4f} |"
        )
    hold_economics = [
        row for row in payload["independent_order_economics"]
        if row["period"] == "holdout_diagnostic" and np.isclose(float(row["ev_hurdle_fraction"]), 0.01)
    ]
    economics_lines = ["| Forecast | Opportunities | Signals | No-trade markets | Fills | Active days | Turnover | Fee equivalent | Gross PnL | Net PnL | Net / turnover | Top-3 absolute-PnL day share | 95% block CI |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in hold_economics:
        net_turnover = "n/a" if row["net_per_turnover"] is None else f"{row['net_per_turnover']:.3%}"
        economics_lines.append(
            f"| {row['forecast']} | {row['opportunity_markets']:,} | {row['signals']:,} "
            f"| {row['no_trade_markets']:,} ({row['no_trade_rate']:.1%}) | {row['fills']:,} | {row['active_utc_days']} "
            f"| ${row['gross_turnover_usd']:,.2f} | ${row['fee_equivalent_usd']:,.2f} "
            f"| ${row['gross_pnl_usd']:,.2f} | ${row['realized_net_pnl_usd']:,.2f} "
            f"| {net_turnover} | {row['top_3_absolute_pnl_day_share']:.1%} "
            f"| ${row['total_pnl_95pct'][0]:,.2f} to ${row['total_pnl_95pct'][1]:,.2f} |"
        )
    monthly = [
        row for row in payload["monthly_independent_order_economics"]
        if row["period"] == "holdout_diagnostic" and np.isclose(float(row["ev_hurdle_fraction"]), 0.01)
    ]
    monthly_lines = ["| Forecast | Month UTC | Opportunities | Signals | No-trade markets | Fills | Active days | Turnover | Fee equivalent | Gross PnL | Net PnL |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in monthly:
        monthly_lines.append(
            f"| {row['forecast']} | {row['month_utc']} | {row['opportunity_markets']:,} | {row['signals']:,} "
            f"| {row['no_trade_markets']:,} | {row['fills']:,} | {row['active_utc_days']} "
            f"| ${row['gross_turnover_usd']:,.2f} | ${row['fee_equivalent_usd']:,.2f} "
            f"| ${row['gross_pnl_usd']:,.2f} | ${row['realized_net_pnl_usd']:,.2f} |"
        )
    portfolio_lines = ["| Forecast | Starting cash | Trades | Turnover | Fees | Net PnL | Ending cash | Max drawdown | Audit |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in payload["holdout_portfolios"]:
        portfolio_lines.append(
            f"| {row['forecast']} | ${row['starting_capital_usd']:,.0f} | {row['trade_count']:,} "
            f"| ${row['gross_turnover_usd']:,.2f} | ${row['fees_paid_estimated_usd']:,.2f} "
            f"| ${row['net_pnl_usd']:,.2f} | ${row['ending_cash_after_settlement_usd']:,.2f} "
            f"| {row['max_drawdown_cost_basis_equity']:.2%} | {row['accounting_audit_status']} |"
        )
    selected_model = payload["development_freeze"]["selected_model"]
    comparisons = payload["development_freeze"]["paired_comparisons_against_frozen_A"]
    lines = [
        "# BTC pre-open: bounded model update",
        "",
        "> The pre-registered comparison was run with all available BTC price data before 2026-07-01 UTC for fitting. July–October is a historical diagnostic period because these outcomes were viewed in earlier repository research.",
        "",
        "## Frozen procedure",
        "",
        f"- Selected model: `{selected_model}`. It was selected on {payload['development_freeze']['development_rows_common']:,} common development markets; the frozen rule uses official-outcome log loss and Bonferroni-adjusted paired 3-day block intervals.",
        "- Two new candidate procedures were compared. The development intervals adjust the two candidate-versus-A comparisons; uncertainty remains conditional on this frozen search, and historical holdout diagnostics are not independent proof because those outcomes were viewed earlier.",
        f"- Existing selected policy: `{CONFIG['economic_evaluation']['selected_policy']}`; fixed $5 gross order, EV hurdle 1%, side chosen at T-60 and executed at T-59. Policy selection was not repeated.",
        f"- Trial budget: {CONFIG['model_search']['requested_trials_per_variant_per_fold']} requested trials per candidate and fold, {CONFIG['model_search']['minimum_successful_trials_per_variant_per_fold']} minimum successful, {CONFIG['model_search']['total_requested_trials']} requested total. Generator/selector history and budgets were preserved.",
        "- The proxy target remains Binance COIN-M BTCUSD index data: `Close[Opened+6m] >= Open[Opened+2m]`; the five-minute label becomes available at `Opened+7m`. Nominal inference is at `Opened+1m` (T-60), with the execution snapshot at T-59. Official Polymarket outcomes remain a separate evaluation target, and price-history training does not require a Polymarket quote.",
        "- The six added features are candle return (1m, 5m lag 2), realized volatility (1m), candle range (1m), candle-direction signed volume (1m), and 5m up/down-volume ratio (lag 1). The raw source has no trade-level aggressor volume, so the signed features are candle proxies.",
        "- Feature timing audit: candle/return/range/volatility/volume and index/perpetual features use closed one-minute inputs and trailing lags available by the nominal decision; session features use the timestamp; reaction/volume profiles, indicator choices, feature selection, and model parameters are fitted inside each earlier fold prefix, then evaluated with causal features. Up/down-volume and signed-volume features are candle-direction proxies because trade-level aggressor volume is unavailable.",
        "- Baseline A bundle: " + CONFIG["frozen_baseline"]["bundle_sha256"] + ". Corrected T-60/T-59 snapshots were hash-verified and reused; no book-event replay was run.",
        f"- Baseline A audit: {payload['frozen_baseline_audit']['model_feature_count']} model features from {payload['frozen_baseline_audit']['selector_candidate_count']} scored candidates and {payload['frozen_baseline_audit']['selector_topk_evaluations']} top-k evaluations; best iteration {payload['frozen_baseline_audit']['best_iteration']}; tuning completed {payload['frozen_baseline_audit']['tuning_successful_trials']} of {payload['frozen_baseline_audit']['tuning_requested_trials']} requested trials.",
        f"- Baseline A sample counts: {payload['frozen_baseline_audit']['dataset_rows_before_fit_filters']:,} generated rows before fit filters, {payload['frozen_baseline_audit']['training_rows']:,} fit rows, and {payload['frozen_baseline_audit']['decision_rows']:,} actual decision rows.",
        "",
        "## Probability quality",
        "",
        *quality_lines,
        "",
        "## Target alignment",
        "",
        f"The Binance proxy label matched the official Polymarket outcome on {payload['target_alignment']['development']['matches']:,}/{payload['target_alignment']['development']['n']:,} common development markets ({payload['target_alignment']['development']['match_rate']:.2%}) and {payload['target_alignment']['holdout_diagnostic']['matches']:,}/{payload['target_alignment']['holdout_diagnostic']['n']:,} historical holdout markets ({payload['target_alignment']['holdout_diagnostic']['match_rate']:.2%}). This is an outcome-agreement audit; the proxy target remains distinct from the official settlement source.",
        "",
        "Development paired log-loss deltas against frozen A (negative favors the candidate):",
        "",
        *[
            f"- `{name}`: {row['log_loss_difference_left_minus_right']:+.6f}; 95% CI {row['log_loss_difference_95pct'][0]:+.6f} to {row['log_loss_difference_95pct'][1]:+.6f}; adjusted interval {row['log_loss_difference_adjusted_interval'][0]:+.6f} to {row['log_loss_difference_adjusted_interval'][1]:+.6f}."
            for name, row in comparisons.items()
        ],
        "",
        "## Model fit audit",
        "",
        *fold_lines,
        "",
        "Learned generator and model state uses the fit interval shown below. Calibration uses earlier post-fit labels in the seed fold and earlier-fold OOF predictions thereafter.",
        "",
        *period_lines,
        "",
        "## Fixed-$5 holdout economics at the frozen 1% EV hurdle",
        "",
        "Historical diagnostics only. Turnover is filled gross stake; gross PnL is before fee effects, and net PnL uses accepted T-59 fees and share/cash accounting.",
        f"The frozen A/EV 1% row was recomputed against the prior report: `{payload['previous_selected_policy_holdout_verification']['status']}`.",
        "",
        *economics_lines,
        "",
        "Monthly results:",
        "",
        *monthly_lines,
        "",
        "## Separate 5%-of-free-cash portfolios",
        "",
        "Each account starts from its own cash balance. Minimum-order skips are retained; stake is not boosted. These are feasibility checks, not the signal-quality assessment.",
        "",
        *portfolio_lines,
        "",
        "## Inputs and reproducibility",
        "",
        f"- Run manifest: `{(MANIFEST_PATH.resolve().relative_to(ROOT)).as_posix()}`.",
        f"- Development freeze: `{FREEZE_PATH.resolve().relative_to(ROOT).as_posix()}`.",
        f"- Prediction and portfolio ledgers: `{OUTPUT_DIR.resolve().relative_to(ROOT).as_posix()}` (local research data).",
        f"- Summary tables: [`{TABLE_DIR.resolve().relative_to(ROOT).as_posix()}`]({TABLE_DIR.name}/).",
        "- Reproduce or resume with `python run_btc_preopen_model_update.py` (no command-line arguments).",
        "- Holdout results did not revise features, hyperparameters, calibration, or policy. Live trading remains off.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_model_update() -> dict:
    started = time.perf_counter()
    _validate_frozen_config()
    (
        source_research,
        source_walk,
        source_holdout,
        t59,
        t60,
        correction_manifest,
        book_record,
        source_identity,
    ) = _source_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = research._read_json(MANIFEST_PATH) if MANIFEST_PATH.is_file() else {
        "schema_version": 1,
        "experiment_id": CONFIG["experiment_id"],
        "folds": {},
    }
    if manifest.get("source_identity") not in (None, source_identity):
        raise RuntimeError("Existing model-update outputs belong to different frozen inputs or code")
    manifest["source_identity"] = source_identity
    manifest["status"] = "running"
    manifest["started_utc"] = manifest.get("started_utc", pd.Timestamp.now(tz="UTC").isoformat())
    manifest["resource_snapshot_at_start"] = _resource_snapshot()
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(MANIFEST_PATH, manifest)

    source_contract = json.loads((ROOT / CONFIG["source_contract"]).read_text(encoding="utf-8"))
    base_profile = json.loads((ROOT / "configs/btc_preopen_training.json").read_text(encoding="utf-8"))["profiles"]["nightly_v1"]
    raw_relative = (ROOT / CONFIG["raw_data_path"]).resolve().relative_to(ROOT).as_posix()
    markets_relative = (ROOT / CONFIG["official_markets_path"]).resolve().relative_to(ROOT).as_posix()
    folds = {str(item["fold_id"]): item for item in walk_forward.CONFIG["folds"]}
    development_fold_ids = [
        str(item["fold_id"]) for item in walk_forward.CONFIG["folds"]
        if str(item["fold_id"]) != "evaluated_holdout"
    ]
    for fold_id in development_fold_ids:
        fold = folds[fold_id]
        contract_path = CONTRACT_DIR / f"{fold_id}.json"
        walk_forward._write_json(contract_path, walk_forward._fold_contract(source_contract, fold))
        for variant in CONFIG["variants"]:
            previous_paths = []
            for prior_id in development_fold_ids:
                if prior_id == fold_id:
                    break
                record = manifest.get("folds", {}).get(prior_id, {}).get(variant["name"], {})
                prior_path = ROOT / record.get("prediction_path", "__missing__")
                if record.get("status") != "completed" or not prior_path.is_file() or _sha256(prior_path) != record.get("prediction_sha256"):
                    raise RuntimeError(f"Missing earlier {variant['name']} OOF rows before fold {fold_id}")
                previous_paths.append(prior_path)
            record = _run_variant_fold(
                variant=variant,
                fold=fold,
                base_contract=source_contract,
                base_profile=base_profile,
                raw_relative=raw_relative,
                markets_relative=markets_relative,
                previous_paths=previous_paths,
                manifest=manifest,
            )
            manifest.setdefault("folds", {}).setdefault(fold_id, {})[variant["name"]] = record
            manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
            _write_json(MANIFEST_PATH, manifest)
        print(f"[btc-model-update] development checkpoint complete fold={fold_id}", flush=True)

    dev_prediction_frames = {
        name: _load_candidate_predictions(manifest, name, development_fold_ids)
        for name in VARIANTS
    }
    dev_predictions, baseline_dev_identity = research._load_model_predictions(include_holdout=False)
    dev_calendar, _ = research._build_market_calendar(dev_predictions)
    dev_calendar["market_start_utc"] = pd.to_datetime(dev_calendar.market_start_utc, utc=True, errors="raise")
    dev_calendar = dev_calendar.loc[dev_calendar.market_start_utc.lt(DEV_CUTOFF_UTC)].copy()
    _, t60_dev, _ = research._load_enriched_book_snapshots(correction_manifest)
    dev_base_data = probability._build_dataset(dev_calendar, t60_dev)
    dev_data = _merge_predictions(dev_base_data, dev_prediction_frames)
    dev_common = dev_data.loc[
        dev_data.common_sample
        & dev_data.official_outcome_up.isin((0.0, 1.0))
        & dev_data.market_start_utc.lt(DEV_CUTOFF_UTC)
    ].copy()
    baseline_dev_identity = {
        **baseline_dev_identity,
        "t60_snapshot_sha256": book_record["t60_coverage_sha256"],
        "source_holdout_manifest_sha256": _sha256(fee_correction.SOURCE_HOLDOUT_MANIFEST),
        "candidate_prediction_sha256": {
            name: {fold_id: manifest["folds"][fold_id][name]["prediction_sha256"] for fold_id in development_fold_ids}
            for name in VARIANTS
        },
    }
    development_freeze = _development_freeze(dev_common, baseline_dev_identity, dev_prediction_frames)
    manifest["development_freeze_path"] = FREEZE_PATH.resolve().relative_to(ROOT).as_posix()
    manifest["development_freeze_sha256"] = _sha256(FREEZE_PATH)
    manifest["development_freeze"] = {
        "selected_model": development_freeze["selected_model"],
        "common_rows": development_freeze["development_rows_common"],
        "holdout_used_for_selection": False,
    }
    manifest["status"] = "development_selection_frozen"
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(MANIFEST_PATH, manifest)
    print(
        f"[btc-model-update] development selection frozen model={development_freeze['selected_model']} "
        f"rows={development_freeze['development_rows_common']:,}; holdout evaluation follows",
        flush=True,
    )

    holdout_fold = folds["evaluated_holdout"]
    holdout_previous_paths_by_variant = {}
    for variant in CONFIG["variants"]:
        name = str(variant["name"])
        holdout_previous_paths_by_variant[name] = [
            ROOT / manifest["folds"][fold_id][name]["prediction_path"]
            for fold_id in development_fold_ids
        ]
        holdout_record = _run_variant_fold(
            variant=variant,
            fold=holdout_fold,
            base_contract=source_contract,
            base_profile=base_profile,
            raw_relative=raw_relative,
            markets_relative=markets_relative,
            previous_paths=holdout_previous_paths_by_variant[name],
            manifest=manifest,
        )
        manifest.setdefault("folds", {}).setdefault("evaluated_holdout", {})[name] = holdout_record
        manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        _write_json(MANIFEST_PATH, manifest)

    all_fold_ids = [*development_fold_ids, "evaluated_holdout"]
    candidate_predictions = {
        name: _load_candidate_predictions(manifest, name, all_fold_ids)
        for name in VARIANTS
    }
    all_predictions, baseline_identity = research._load_model_predictions(include_holdout=True)
    calendar, calendar_coverage = research._build_market_calendar(all_predictions)
    calendar["market_start_utc"] = pd.to_datetime(calendar.market_start_utc, utc=True, errors="raise")
    base_data = probability._build_dataset(calendar, t60)
    evaluation_data = _merge_predictions(base_data, candidate_predictions)
    dev = evaluation_data.loc[
        evaluation_data.common_sample
        & evaluation_data.official_outcome_up.isin((0.0, 1.0))
        & evaluation_data.market_start_utc.lt(DEV_CUTOFF_UTC)
    ].copy()
    holdout = evaluation_data.loc[
        evaluation_data.common_sample
        & evaluation_data.official_outcome_up.isin((0.0, 1.0))
        & evaluation_data.market_start_utc.ge(DEV_CUTOFF_UTC)
        & evaluation_data.market_start_utc.lt(DATA_CUTOFF_UTC)
    ].copy()
    if not set(dev.fold_id.astype(str).unique()).issubset(set(development_fold_ids)):
        raise RuntimeError("Development economic/quality sample contains a non-development fold")
    if not holdout.fold_id.astype(str).eq("evaluated_holdout").all():
        raise RuntimeError("Holdout diagnostic sample contains a development prediction")
    edges = probability._metric_edges(dev.p_btc_calibrated.to_numpy(dtype=np.float64))
    quality_rows, calibration_bins, fold_quality, monthly_quality = [], [], [], []
    for period, frame in (("development_oof", dev), ("holdout_diagnostic", holdout)):
        for forecast in FORECAST_NAMES:
            column = PROBABILITY_COLUMNS[forecast]
            rows, bins = _metric_rows(frame, period, forecast, column, edges)
            quality_rows.extend(rows)
            calibration_bins.extend(bins)
            if period == "development_oof":
                for fold_id, fold_frame in frame.groupby("fold_id", sort=True):
                    fold_rows, _ = _metric_rows(fold_frame, str(fold_id), forecast, column, edges)
                    fold_quality.extend(fold_rows)
            else:
                for month, month_frame in frame.groupby("month_utc", sort=True):
                    month_rows, _ = _metric_rows(month_frame, period, forecast, column, edges)
                    for row in month_rows:
                        row["month_utc"] = month
                    monthly_quality.extend(month_rows)

    paired = {}
    for left, right in (
        ("A_reoptimized", "A_frozen"),
        ("B_dynamic_price_volume", "A_frozen"),
        ("B_dynamic_price_volume", "A_reoptimized"),
    ):
        key = f"{left}-{right}"
        paired[key] = _paired_block_interval(
            dev,
            PROBABILITY_COLUMNS[left],
            PROBABILITY_COLUMNS[right],
            seed=20261012,
            replicates=2000,
        )
    t59_by_id = research._snapshot_map(t59)
    t60_by_id = research._snapshot_map(t60)
    market_by_id = research._snapshot_map(calendar.loc[calendar.condition_id.notna()])
    hurdles = tuple(float(value) for value in CONFIG["economic_evaluation"]["ev_hurdles"])
    policy_config = json.loads(nested.CONFIG_PATH.read_text(encoding="utf-8"))
    dev_economic_frame = dev.copy()
    dev_economic_frame["outer_fold_id"] = dev_economic_frame.fold_id.astype(str)
    dev_details = nested._economic_detail(
        dev_economic_frame,
        FORECAST_NAMES,
        t59_by_id,
        t60_by_id,
        market_by_id,
        "development_oof",
        PROBABILITY_COLUMNS,
    )
    dev_economics, dev_fold_economics, dev_monthly_economics = nested._policy_summaries(
        dev_details, dev_economic_frame, hurdles, policy_config, model_names=FORECAST_NAMES,
    )
    hold_details = nested._economic_detail(
        holdout,
        FORECAST_NAMES,
        t59_by_id,
        t60_by_id,
        market_by_id,
        "holdout_diagnostic",
        PROBABILITY_COLUMNS,
    )
    hold_economics, _, hold_monthly_economics = nested._policy_summaries(
        hold_details, holdout, hurdles, policy_config, model_names=FORECAST_NAMES,
    )
    previous_policy_verification = _verify_previous_selected_policy_economics(hold_economics)
    baseline_model_audit = _baseline_model_audit(source_holdout)
    target_alignment = {
        "development": _target_alignment(dev),
        "holdout_diagnostic": _target_alignment(holdout),
    }
    selected_model = str(development_freeze["selected_model"])
    portfolios = _holdout_portfolios(
        evaluation_data,
        FORECAST_NAMES,
        t59_by_id,
        t60_by_id,
        market_by_id,
        float(CONFIG["economic_evaluation"]["portfolio_policy_hurdle"]),
    )
    model_fold_training = [
        {
            "variant": name,
            "fold_id": fold_id,
            "dataset_rows_before_fit_filters": manifest["folds"][fold_id][name]["dataset_rows_before_fit_filters"],
            "fit_rows": manifest["folds"][fold_id][name]["fit_rows"],
            "decision_rows": manifest["folds"][fold_id][name]["decision_rows"],
            "auxiliary_fit_rows": manifest["folds"][fold_id][name]["auxiliary_fit_rows"],
            "model_feature_count": manifest["folds"][fold_id][name]["model_feature_count"],
            "selector_feature_count": manifest["folds"][fold_id][name]["selector_feature_count"],
            "added_candidate_features": manifest["folds"][fold_id][name]["added_candidate_features"],
            "generator_fit_start_utc": manifest["folds"][fold_id][name]["generator_fit_start_utc"],
            "generator_fit_end_exclusive_utc": manifest["folds"][fold_id][name]["generator_fit_end_exclusive_utc"],
            "model_fit_start_utc": manifest["folds"][fold_id][name]["model_fit_start_utc"],
            "model_fit_end_exclusive_utc": manifest["folds"][fold_id][name]["model_fit_end_exclusive_utc"],
            "calibration_start_decision_at_utc": manifest["folds"][fold_id][name]["calibration_start_decision_at_utc"],
            "calibration_end_decision_at_utc": manifest["folds"][fold_id][name]["calibration_end_decision_at_utc"],
            "calibration_source": manifest["folds"][fold_id][name]["calibration_source"],
            "calibration_training_rows": manifest["folds"][fold_id][name]["calibration_training_rows"],
            "calibration_latest_label_available_at_utc": manifest["folds"][fold_id][name]["calibration_latest_label_available_at_utc"],
            "prediction_market_start_first_utc": manifest["folds"][fold_id][name]["prediction_market_start_first_utc"],
            "prediction_market_start_last_inclusive_utc": manifest["folds"][fold_id][name]["prediction_market_start_last_inclusive_utc"],
            "latest_training_label_available_at_utc": manifest["folds"][fold_id][name]["latest_training_label_available_at_utc"],
            **manifest["folds"][fold_id][name]["model_tuning"],
            "elapsed_seconds": manifest["folds"][fold_id][name]["elapsed_seconds"],
        }
        for fold_id in all_fold_ids for name in VARIANTS
    ]
    payload = {
        "experiment_id": CONFIG["experiment_id"],
        "configuration_sha256": _sha256(CONFIG_PATH),
        "source_identity": source_identity,
        "baseline_prediction_identity": baseline_identity,
        "book_snapshot_identity": {
            "t60_sha256": book_record["t60_coverage_sha256"],
            "t59_sha256": book_record["t59_coverage_sha256"],
            "replay_performed": False,
        },
        "calendar_coverage": calendar_coverage,
        "development_freeze": development_freeze,
        "frozen_baseline_audit": baseline_model_audit,
        "development_rows_common": int(len(dev)),
        "holdout_rows_common": int(len(holdout)),
        "target_alignment": target_alignment,
        "probability_quality": quality_rows,
        "calibration_bins": calibration_bins,
        "fold_probability_quality": fold_quality,
        "monthly_probability_quality": monthly_quality,
        "paired_probability_differences": paired,
        "development_independent_order_economics": dev_economics,
        "development_policy_fold_stability": dev_fold_economics,
        "development_monthly_independent_order_economics": dev_monthly_economics,
        "holdout_independent_order_economics": hold_economics,
        "previous_selected_policy_holdout_verification": previous_policy_verification,
        "holdout_monthly_independent_order_economics": hold_monthly_economics,
        "independent_order_economics": dev_economics + hold_economics,
        "monthly_independent_order_economics": dev_monthly_economics + hold_monthly_economics,
        "holdout_portfolios": portfolios,
        "model_fold_training": model_fold_training,
        "fold_periods": [
            {
                "variant": row["variant"],
                "fold_id": row["fold_id"],
                "fit_start_utc": row["generator_fit_start_utc"],
                "fit_end_exclusive_utc": row["generator_fit_end_exclusive_utc"],
                "calibration_start_decision_at_utc": row["calibration_start_decision_at_utc"],
                "calibration_end_decision_at_utc": row["calibration_end_decision_at_utc"],
                "calibration_source": row["calibration_source"],
                "calibration_training_rows": row["calibration_training_rows"],
                "calibration_latest_label_available_at_utc": row["calibration_latest_label_available_at_utc"],
                "prediction_market_start_first_utc": row["prediction_market_start_first_utc"],
                "prediction_market_start_last_inclusive_utc": row["prediction_market_start_last_inclusive_utc"],
            }
            for row in model_fold_training
        ],
        "runtime_seconds": time.perf_counter() - started,
        "resource_snapshot_at_completion": _resource_snapshot(),
        "holdout_used_for_model_or_policy_selection": False,
        "live_enabled": False,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    probability_frame = pd.concat([dev, holdout], ignore_index=True, sort=False)
    probability_frame.to_parquet(OUTPUT_DIR / "model_quality_predictions.parquet", index=False, compression="zstd")
    pd.concat([dev_details, hold_details], ignore_index=True, sort=False).to_parquet(
        OUTPUT_DIR / "independent_order_detail.parquet", index=False, compression="zstd",
    )
    _write_json(OUTPUT_DIR / "evaluation_summary.json", payload)
    _write_report(payload)
    manifest["status"] = "completed"
    manifest["evaluation_summary_path"] = (OUTPUT_DIR / "evaluation_summary.json").resolve().relative_to(ROOT).as_posix()
    manifest["evaluation_summary_sha256"] = _sha256(OUTPUT_DIR / "evaluation_summary.json")
    manifest["report_path"] = REPORT_PATH.resolve().relative_to(ROOT).as_posix()
    manifest["report_sha256"] = _sha256(REPORT_PATH)
    manifest["holdout_used_for_model_or_policy_selection"] = False
    manifest["selected_model"] = selected_model
    manifest["runtime_seconds"] = payload["runtime_seconds"]
    manifest["resource_snapshot_at_completion"] = payload["resource_snapshot_at_completion"]
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(MANIFEST_PATH, manifest)
    print(
        f"[btc-model-update] completed selected_model={selected_model} "
        f"development={len(dev):,} holdout_diagnostic={len(holdout):,} "
        f"runtime={payload['runtime_seconds'] / 3600.0:.2f}h",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run_model_update()
