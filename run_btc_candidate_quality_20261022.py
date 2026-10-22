"""Freeze BTC BASE70, causal-memory, and ESN procedures on existing development data."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
import numba
import threadpoolctl

import run_btc_esn_focused_20261022 as focused
import run_btc_esn_expansion_20261021 as expansion
from features.causal_memory_control import (
    CausalMemoryState,
    MEMORY_HALFLIVES,
    MEMORY_LAGS,
    MEMORY_MAX_LAG,
    memory_feature_names,
)
from features.esn_reservoir import ESNEngine, FeatureScaler


ROOT = Path(__file__).resolve().parent
FREEZE_CONFIG_PATH = ROOT / "configs/research/btc_candidate_quality_freeze_20261022.json"
FREEZE_CONFIG = json.loads(FREEZE_CONFIG_PATH.read_text(encoding="utf-8"))
SOURCE_OUT = ROOT / focused.CONFIG["output_directory"]
OUT = ROOT / "data/analysis/polymarket/BTC/btc_candidate_quality_freeze_20261022"
BUNDLE_DIR = OUT / "bundles"
REPORT_PATH = ROOT / "reports/btc_preopen/btc_candidate_quality_freeze_20261022.md"
ACTIVE_LIMIT_SECONDS = int(FREEZE_CONFIG["scope"]["new_active_compute_limit_seconds"])
BOOTSTRAP_REPLICATES = int(FREEZE_CONFIG["uncertainty"]["replicates"])
BOOTSTRAP_BLOCK_DAYS = int(FREEZE_CONFIG["uncertainty"]["block_length_days"])
BOOTSTRAP_SEED = int(FREEZE_CONFIG["uncertainty"]["random_seed"])
RUN_STARTED = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_array(values) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _check_active_limit() -> None:
    if time.monotonic() - RUN_STARTED > ACTIVE_LIMIT_SECONDS:
        raise TimeoutError(f"new candidate-quality work exceeded its frozen {ACTIVE_LIMIT_SECONDS}s cap")


def _configure_source_run() -> tuple[dict, dict]:
    prepared_path = ROOT / focused.CONFIG["reused_search_cache_directory"] / "prepared_manifest.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if (
        prepared.get("materialization_cutoff_utc") != "2025-07-01T00:00:00+00:00"
        or prepared.get("holdout_rows_materialized") != 0
    ):
        raise RuntimeError("source cache does not match the development-only frozen cutoff")
    identity = focused._focused_identity(prepared)
    focused.RUN_IDENTITY = identity
    focused.OUT = SOURCE_OUT
    focused._configure_engines()
    eng = expansion.engine_runner
    eng.OUT = SOURCE_OUT
    # The source runner's cap callback writes its old runtime ledger; this task owns its own cap.
    eng._check_compute_cap = _check_active_limit
    arrays = eng._load_search_data(identity)
    eng._SCREEN_ARRAYS = arrays
    if int(arrays["base_x"].shape[1]) != 70:
        raise RuntimeError("BASE70 width differs from the frozen candidate protocol")
    if int(arrays["raw_opened"][-1]) >= eng.SEARCH_END_NS:
        raise RuntimeError("the causal raw-feature stream crosses the search-validation cutoff")
    if int(arrays["base_opened"][-1]) >= eng.SEARCH_END_NS:
        raise RuntimeError("the development cache crosses the search-validation cutoff")
    if int(np.max(arrays["label_available"])) >= eng.SEARCH_END_NS:
        raise RuntimeError("the development cache includes a label unavailable at the cutoff")
    if not np.array_equal(arrays["base_decision_at"], arrays["base_opened"] + eng.MINUTE_NS):
        raise RuntimeError("candidate rows do not use the frozen Opened+1m decision timestamp")
    if not np.array_equal(arrays["label_available"], arrays["base_opened"] + eng.LABEL_HORIZON_NS):
        raise RuntimeError("candidate labels do not use the frozen Opened+7m availability time")
    return identity, arrays


def _load_frozen_inputs(arrays: dict) -> tuple[str, dict, np.ndarray, dict[int, np.ndarray], dict]:
    eng = expansion.engine_runner
    finalists = json.loads((SOURCE_OUT / "focused_finalist_pool.json").read_text(encoding="utf-8"))["finalists"]
    candidate = next(row for row in finalists if row.get("role") == "best_local_preliminary")
    candidate_id = candidate["candidate_id"]
    if candidate_id != FREEZE_CONFIG["procedures"]["C"]["candidate_id"]:
        raise RuntimeError("the selected focused ESN differs from the preregistered candidate")
    esn_config = eng._config_from_record(candidate["effective_config"])
    memory = focused._memory_feature_matrix(arrays)
    esn_features = {
        int(seed): focused._load_final_feature(arrays, candidate_id, int(seed))
        for seed in FREEZE_CONFIG["model_fit"]["reservoir_seeds"]
    }
    return candidate_id, candidate, memory, esn_features, {"config": esn_config, "finalists": finalists}


def _verify_transform_fit_scope(arrays: dict, memory_bundle: dict,
                                esn_config, esn_bundles: dict[int, dict]) -> dict:
    eng = expansion.engine_runner
    memory_identity = memory_bundle["identity"]
    expected_memory_inputs = list(memory_identity["input_columns"])
    if expected_memory_inputs != list(esn_config.selected_columns):
        raise RuntimeError("memory and ESN raw inputs differ from the frozen common 14-column order")
    if memory_identity.get("scope_end_exclusive_utc") != "2025-07-01T00:00:00Z":
        raise RuntimeError("memory projection bundle has an unexpected data scope")
    if int(memory_identity.get("pca_fit_rows", 0)) != int(focused.CONFIG["memory_control"]["pca_fit_rows"]):
        raise RuntimeError("memory PCA fit-row count differs from its frozen configuration")
    if len(memory_bundle.get("fit_sample_indices_sha256", "")) != 64:
        raise RuntimeError("memory PCA bundle is missing its fit-core sample-index checksum")
    source_run_identity = memory_identity.get("run_identity", {})
    if (
        source_run_identity.get("confirmation_read") is not False
        or source_run_identity.get("holdout_rows_materialized") != 0
    ):
        raise RuntimeError("memory transform provenance does not prove development-only scope")

    # Refit only the deterministic input scaler on fit-core and compare it byte-for-byte.
    expected_scaler = eng._fit_scaler(esn_config, arrays).to_dict()
    expected_scaler_json = json.dumps(expected_scaler, sort_keys=True, separators=(",", ":"))
    memory_scaler_json = json.dumps(memory_bundle["input_scaler"], sort_keys=True, separators=(",", ":"))
    if memory_scaler_json != expected_scaler_json:
        raise RuntimeError("memory input scaler does not equal the scaler fit on the frozen fit-core rows")

    expected_pca_rows = int(focused.CONFIG["screening"]["pca_fit_rows"])
    source_bundle_hashes = {}
    for seed, bundle in esn_bundles.items():
        if int(bundle["seed"]) != int(seed):
            raise RuntimeError("ESN reservoir bundle seed does not match its frozen seed slot")
        if bundle["identity"].get("holdout_rows_materialized") != 0 or bundle["identity"].get("confirmation_read") is not False:
            raise RuntimeError("ESN bundle provenance does not prove development-only scope")
        scaler_json = json.dumps(bundle["scaler"], sort_keys=True, separators=(",", ":"))
        if scaler_json != expected_scaler_json:
            raise RuntimeError(f"ESN input scaler for seed {seed} differs from the fit-core scaler")
        components = np.asarray(bundle["projection"]["components"], dtype=np.float64)
        if components.shape[0] != 8:
            raise RuntimeError(f"ESN PCA for seed {seed} does not contain exactly eight components")
        source_bundle_hashes[str(seed)] = {
            "pca_fit_rows_from_frozen_source_config": expected_pca_rows,
            "pca_scope": "source implementation samples only warm-up-ready states with raw_opened < FIT_END_NS",
            "projection_seed": int(bundle["projection_seed"]),
            "bundle_identity_source_commit": bundle["identity"].get("frozen_source_commit"),
        }
    return {
        "memory_pca_fit_rows": int(memory_identity["pca_fit_rows"]),
        "memory_fit_sample_indices_sha256": memory_bundle["fit_sample_indices_sha256"],
        "esn_pca_fit_rows_by_seed": source_bundle_hashes,
        "input_scalers_match_independent_fit_core_reconstruction": True,
        "scope_end_exclusive_utc": memory_identity["scope_end_exclusive_utc"],
        "transform_fit_scope_verified": True,
    }


def _load_feature_order() -> list[str]:
    source = ROOT / focused.CONFIG["baseline_feature_summary_path"]
    summary = json.loads(source.read_text(encoding="utf-8"))
    order = summary.get("feature_order")
    if not isinstance(order, list) or len(order) != 70:
        raise RuntimeError("the source BASE70 manifest does not contain its exact 70-column order")
    return [str(value) for value in order]


def _feature_definition_sources() -> dict:
    paths = (
        "run_btc_preopen_taker_flow_20261018.py",
        "run_btc_esn_focused_20261022.py",
        "run_btc_esn_expansion_20261021.py",
        "run_btc_esn_architecture_audit_20261020.py",
        "configs/research/btc_preopen_taker_flow_20261018.json",
        "configs/research/btc_esn_focused_20261022.json",
        "data/analysis/polymarket/BTC/btc_esn_focused_20261022/execution_hotfix.json",
        "features/btc_preopen_baseline.py",
        "features/btc_preopen_contract.py",
        "features/btc_preopen_return_target.py",
        "features/candle_features.py",
        "features/reaction_profile_fixed_grid.py",
        "features/volume_profile_fixed_range.py",
        "features/realized_volatility.py",
        "features/basis_premium_features.py",
        "features/session_open_features.py",
        "features/btc_taker_flow.py",
        "features/ADX.py",
        "features/BollingerBands.py",
        "features/KeltnerChannel.py",
        "features/StochOsc.py",
        "features/common_utils.py",
        "features/feature_intervals.py",
        "features/live_indicator_runtime.py",
        "features/live_indicator_runtime_entry_state_v1.py",
        "features/ta_custom_ma.py",
        "features/ta_tools.py",
        "features/esn_reservoir.py",
        "features/causal_memory_control.py",
        "utils/btc_oof_audit.py",
    )
    return {path: _sha256(ROOT / path) for path in paths}


def _base_feature_schema(base_feature_order: list[str]) -> list[dict]:
    families = (
        ("rp_", "reaction profile", "features/reaction_profile_fixed_grid.py"),
        ("vp_", "volume profile", "features/volume_profile_fixed_range.py"),
        ("realized_volatility", "realized volatility", "features/realized_volatility.py"),
        ("futures_index_basis", "futures/index basis", "features/basis_premium_features.py"),
        ("session_", "session open/close", "features/session_open_features.py"),
        ("taker_", "taker flow Z/V", "features/btc_taker_flow.py"),
        ("candle_", "candle shape/pressure/streak/return/volume", "features/candle_features.py"),
        ("KeltnerChannel_", "Keltner channel indicator; parameters are encoded in the name", "features/KeltnerChannel.py"),
        ("ADX_", "ADX indicator; parameters are encoded in the name", "features/ADX.py"),
        ("StochOsc_", "stochastic oscillator; parameters are encoded in the name", "features/StochOsc.py"),
        ("BollingerBands_", "Bollinger bands; parameters are encoded in the name", "features/BollingerBands.py"),
    )
    schema = []
    for name in base_feature_order:
        match = next((item for item in families if name.startswith(item[0])), None)
        if match is None:
            raise RuntimeError(f"no source-backed feature definition is registered for BASE70 column {name!r}")
        _, family, source = match
        schema.append({
            "name": name,
            "family": family,
            "definition_source": source,
            "definition_source_sha256": _sha256(ROOT / source),
        })
    return schema


def _common_masks(arrays: dict, candidate_id: str, memory: np.ndarray,
                  esn_features: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    eng = expansion.engine_runner
    finalists = json.loads((SOURCE_OUT / "focused_finalist_pool.json").read_text(encoding="utf-8"))["finalists"]
    finalist_ids = [row["candidate_id"] for row in finalists]
    if candidate_id not in finalist_ids:
        raise RuntimeError("the selected ESN is missing from the frozen common finalist grid")
    train_mask, validation_rows, _ = focused._common_final_masks(arrays, finalist_ids)
    if not np.isfinite(memory[validation_rows]).all():
        raise RuntimeError("the saved memory representation is incomplete on common validation decisions")
    if any(not np.isfinite(block[validation_rows]).all() for block in esn_features.values()):
        raise RuntimeError("a saved ESN representation is incomplete on common validation decisions")
    if int(validation_rows.size) != int(FREEZE_CONFIG["comparison"]["expected_common_validation_rows"]):
        raise RuntimeError(f"common decision mask changed: got {len(validation_rows)} rows")
    expected_fit_rows = int(FREEZE_CONFIG["comparison"]["expected_common_fit_rows"])
    if int(train_mask.sum()) != expected_fit_rows:
        raise RuntimeError(f"common fit mask changed: got {int(train_mask.sum())} rows")
    opened = arrays["base_opened"]
    label_available = arrays["label_available"]
    if np.any(opened[train_mask] >= eng.FIT_END_NS):
        raise RuntimeError("a downstream fit row crossed the fit-core boundary")
    if np.any(label_available[train_mask] > eng.FIT_END_NS + eng.DECISION_PURGE_NS):
        raise RuntimeError("a downstream fit row violates the label-availability purge")
    if np.any(label_available[validation_rows] >= eng.SEARCH_END_NS):
        raise RuntimeError("a validation target was unavailable by the frozen cutoff")
    return train_mask, validation_rows


def _load_stored_prediction(name: str, reservoir_seed: int | None, lightgbm_seed: int,
                            validation_rows: np.ndarray, feature_count: int) -> np.ndarray:
    return focused._load_prediction(name, reservoir_seed, lightgbm_seed, validation_rows, feature_count)


def _new_prediction_path(variant: str, reservoir_seed: int | None, lightgbm_seed: int) -> Path:
    key = "baseline" if reservoir_seed is None else f"r{int(reservoir_seed)}"
    token = hashlib.sha256(f"{variant}|{key}|l{int(lightgbm_seed)}".encode("utf-8")).hexdigest()[:24]
    return OUT / "predictions" / f"{token}.npy"


def _save_prediction(variant: str, reservoir_seed: int | None, lightgbm_seed: int,
                     probabilities, rows: np.ndarray, feature_count: int) -> tuple[Path, dict]:
    path = _new_prediction_path(variant, reservoir_seed, lightgbm_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(probabilities, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)
    metadata = {
        "variant": variant,
        "reservoir_seed": None if reservoir_seed is None else int(reservoir_seed),
        "lightgbm_seed": int(lightgbm_seed),
        "feature_count": int(feature_count),
        "validation_rows": int(len(rows)),
        "validation_row_indices_sha256": _digest_array(rows.astype(np.int64)),
        "probabilities_sha256": _sha256(path),
    }
    _atomic_json(path.with_suffix(".npy.json"), metadata)
    return path, metadata


def _metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return {
        "rows": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
    }


def _member_predictions(candidate_id: str, arrays: dict, rows: np.ndarray,
                        memory: np.ndarray, esn_features: dict[int, np.ndarray]) -> dict:
    output = {"A": {}, "B": {}, "C": {}}
    seeds = FREEZE_CONFIG["model_fit"]["lightgbm_seeds"]
    for lightgbm_seed in seeds:
        output["A"][str(lightgbm_seed)] = _load_stored_prediction(
            f"baseline_l{lightgbm_seed}", None, lightgbm_seed, rows, 70,
        )
        output["B"][str(lightgbm_seed)] = _load_stored_prediction(
            "simple_memory_pca8", None, lightgbm_seed, rows, 78,
        )
        for reservoir_seed in FREEZE_CONFIG["model_fit"]["reservoir_seeds"]:
            key = f"{reservoir_seed}:{lightgbm_seed}"
            output["C"][key] = _load_stored_prediction(
                candidate_id, reservoir_seed, lightgbm_seed, rows, 78,
            )
    output["D"] = {}
    for reservoir_seed in FREEZE_CONFIG["model_fit"]["reservoir_seeds"]:
        for lightgbm_seed in seeds:
            path = _new_prediction_path(
                "D_memory_plus_8f74a7951e10c11830baba75", reservoir_seed, lightgbm_seed,
            )
            if not path.exists() or not path.with_suffix(".npy.json").exists():
                raise RuntimeError(f"missing frozen D prediction for {reservoir_seed}/{lightgbm_seed}")
            metadata = json.loads(path.with_suffix(".npy.json").read_text(encoding="utf-8"))
            if metadata["validation_row_indices_sha256"] != _digest_array(rows.astype(np.int64)):
                raise RuntimeError("D prediction was produced on another decision mask")
            if metadata["probabilities_sha256"] != _sha256(path):
                raise RuntimeError("D prediction checksum mismatch")
            output["D"][f"{reservoir_seed}:{lightgbm_seed}"] = np.load(path, allow_pickle=False)
    return output


def _ensemble_predictions(members: dict) -> dict[str, np.ndarray]:
    lgbm_seeds = FREEZE_CONFIG["model_fit"]["lightgbm_seeds"]
    reservoir_seeds = FREEZE_CONFIG["model_fit"]["reservoir_seeds"]
    return {
        "A": np.mean([members["A"][str(seed)] for seed in lgbm_seeds], axis=0),
        "B": np.mean([members["B"][str(seed)] for seed in lgbm_seeds], axis=0),
        "C": np.mean([members["C"][f"{r}:{s}"] for r in reservoir_seeds for s in lgbm_seeds], axis=0),
        "D": np.mean([members["D"][f"{r}:{s}"] for r in reservoir_seeds for s in lgbm_seeds], axis=0),
    }


def _bootstrap_pairs(y: np.ndarray, dates_ns: np.ndarray, ensemble: dict[str, np.ndarray]) -> dict:
    from utils.btc_oof_audit import moving_calendar_block_multiplicity

    timestamps = pd.to_datetime(dates_ns, utc=True)
    if len(timestamps) == 0:
        raise RuntimeError("the preregistered block bootstrap requires validation observations")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = [("B", "A"), ("C", "A"), ("C", "B"), ("D", "B"), ("D", "C"), ("D", "A")]
    samples = {f"{right}-{left}": {metric: np.empty(BOOTSTRAP_REPLICATES) for metric in ("log_loss", "brier", "auc")}
               for left, right in comparisons}

    prepared = {}
    y_float = np.asarray(y, dtype=np.float64)
    for name, probabilities in ensemble.items():
        p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        order = np.argsort(p, kind="mergesort")
        sorted_p = p[order]
        prepared[name] = {
            "p": p,
            "order": order,
            "y_sorted": y_float[order],
            "score_starts": np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1],
            "log_loss_rows": -(y_float * np.log(p) + (1.0 - y_float) * np.log1p(-p)),
            "brier_rows": np.square(p - y_float),
        }

    def weighted_metrics(item: dict, multiplicity: np.ndarray) -> dict:
        total = float(multiplicity.sum())
        if total <= 0.0:
            raise RuntimeError("calendar bootstrap draw contains no validation rows")
        log_loss = float(np.dot(multiplicity, item["log_loss_rows"]) / total)
        brier = float(np.dot(multiplicity, item["brier_rows"]) / total)
        sorted_weights = multiplicity[item["order"]]
        positive_by_score = np.add.reduceat(sorted_weights * item["y_sorted"], item["score_starts"])
        negative_by_score = np.add.reduceat(sorted_weights * (1.0 - item["y_sorted"]), item["score_starts"])
        positive = float(positive_by_score.sum())
        negative = float(negative_by_score.sum())
        auc = float("nan")
        if positive > 0.0 and negative > 0.0:
            negative_before = np.cumsum(negative_by_score) - negative_by_score
            auc = float(np.dot(positive_by_score, negative_before + 0.5 * negative_by_score) / (positive * negative))
        return {"log_loss": log_loss, "brier": brier, "auc": auc}

    for replicate in range(BOOTSTRAP_REPLICATES):
        if replicate % 100 == 0:
            _check_active_limit()
            print(f"paired block bootstrap {replicate}/{BOOTSTRAP_REPLICATES}", flush=True)
        multiplicity = moving_calendar_block_multiplicity(
            timestamps, rng, block_days=BOOTSTRAP_BLOCK_DAYS,
        )
        metrics = {name: weighted_metrics(item, multiplicity) for name, item in prepared.items()}
        for left, right in comparisons:
            key = f"{right}-{left}"
            # Positive LL/Brier deltas and positive AUC deltas favor the right-hand procedure.
            for metric in ("log_loss", "brier"):
                samples[key][metric][replicate] = metrics[left][metric] - metrics[right][metric]
            samples[key]["auc"][replicate] = metrics[right]["auc"] - metrics[left]["auc"]
    return {
        "method": "paired moving calendar-day block bootstrap using utils.btc_oof_audit.moving_calendar_block_multiplicity",
        "block_length_days": BOOTSTRAP_BLOCK_DAYS,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "interval": "percentile 95%",
        "intervals": {
            pair: {
                metric: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
                for metric, values in metrics.items()
            }
            for pair, metrics in samples.items()
        },
    }


def _write_esn_matrix_bundle(esn_config, reservoir_seed: int,
                             source_bundle: dict, destination: Path) -> dict:
    engine = ESNEngine(esn_config, reservoir_seed, FeatureScaler.from_dict(source_bundle["scaler"]),
                       warmup_steps=int(source_bundle["warmup_steps"]))
    matrix_path = destination.with_suffix(".npz")
    np.savez_compressed(
        matrix_path,
        input_weights=engine._win,
        recurrent_indices=engine._rec_indices,
        recurrent_values=engine._rec_values,
        bias=engine._bias,
        input_indices=engine._input_indices,
        input_counts=engine._input_counts,
    )
    with np.load(matrix_path, allow_pickle=False) as saved:
        for name in ("input_weights", "recurrent_indices", "recurrent_values", "bias", "input_indices", "input_counts"):
            np.testing.assert_array_equal(saved[name], getattr(engine, {
                "input_weights": "_win",
                "recurrent_indices": "_rec_indices",
                "recurrent_values": "_rec_values",
                "bias": "_bias",
                "input_indices": "_input_indices",
                "input_counts": "_input_counts",
            }[name]))
    return {
        "source_bundle": str((SOURCE_OUT / "bundles" / f"{expansion.engine_runner._config_key(esn_config)}_seed_{reservoir_seed}.json").relative_to(ROOT)),
        "source_bundle_sha256": _sha256(SOURCE_OUT / "bundles" / f"{expansion.engine_runner._config_key(esn_config)}_seed_{reservoir_seed}.json"),
        "matrix_archive": str(matrix_path.relative_to(ROOT)),
        "matrix_archive_sha256": _sha256(matrix_path),
        "regeneration": "features.esn_reservoir.ESNEngine._make_weights via numpy.random.default_rng(reservoir_seed), with frozen config and dependency versions",
        "seed": int(reservoir_seed),
        "warmup_steps": int(source_bundle["warmup_steps"]),
        "feature_names": list(source_bundle["feature_names"]),
    }


def _fit_and_bundle_models(arrays: dict, candidate_id: str, candidate: dict,
                           memory: np.ndarray, esn_features: dict[int, np.ndarray],
                           train_mask: np.ndarray, rows: np.ndarray,
                           base_feature_order: list[str], source_identity: dict) -> dict:
    eng = expansion.engine_runner
    target = arrays["target"]
    weights = np.where(arrays["decision"], 0.59, 0.1025).astype(np.float32)
    seeds = FREEZE_CONFIG["model_fit"]["lightgbm_seeds"]
    reservoir_seeds = FREEZE_CONFIG["model_fit"]["reservoir_seeds"]
    model_objects = {"A": {}, "B": {}, "C": {}, "D": {}}
    procedures = {}
    fit_start = time.monotonic()
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    model_root = BUNDLE_DIR / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    transform_root = BUNDLE_DIR / "transforms"
    transform_root.mkdir(parents=True, exist_ok=True)
    memory_source = SOURCE_OUT / "memory_control" / "projection_bundle.json"
    memory_bundle = json.loads(memory_source.read_text(encoding="utf-8"))
    memory_copy = transform_root / "simple_memory_projection.json"
    shutil.copyfile(memory_source, memory_copy)
    candidate_record = candidate["effective_config"]
    esn_config = eng._config_from_record(candidate_record)
    esn_files = {}
    for reservoir_seed in reservoir_seeds:
        source_bundle_path = SOURCE_OUT / "bundles" / f"{eng._config_key(esn_config)}_seed_{reservoir_seed}.json"
        source_bundle = json.loads(source_bundle_path.read_text(encoding="utf-8"))
        destination = transform_root / f"esn_seed_{reservoir_seed}.json"
        shutil.copyfile(source_bundle_path, destination)
        esn_files[str(reservoir_seed)] = _write_esn_matrix_bundle(
            esn_config, reservoir_seed, source_bundle,
            transform_root / f"esn_matrices_seed_{reservoir_seed}",
        )
        esn_files[str(reservoir_seed)]["transform_bundle"] = str(destination.relative_to(ROOT))
        esn_files[str(reservoir_seed)]["transform_bundle_sha256"] = _sha256(destination)

    base_feature_schema = _base_feature_schema(base_feature_order)
    common_training = {
        "fit_rows": int(train_mask.sum()),
        "fit_row_indices_sha256": _digest_array(np.flatnonzero(train_mask).astype(np.int64)),
        "target_sha256": _digest_array(target[train_mask].astype(np.int8)),
        "weights_sha256": _digest_array(weights[train_mask].astype(np.float32)),
        "fit_start_utc": focused.CONFIG["split"]["fit_core_start_utc"],
        "fit_end_exclusive_utc": focused.CONFIG["split"]["fit_core_end_exclusive_utc"],
        "label_availability_cutoff_utc": "2024-07-01T00:01:00Z",
        "decision_weight": 0.59,
        "other_minute_weight": 0.1025,
        "early_stopping": False,
        "boosting_rounds": int(focused.CONFIG["baseline"]["lightgbm"]["boosting_rounds"]),
        "lgbm_parameters_by_seed": {str(seed): eng._lgbm_params(int(seed)) for seed in seeds},
        "feature_summary_sha256": _sha256(ROOT / focused.CONFIG["baseline_feature_summary_path"]),
        "common_baseline_seed_matching": True,
    }
    for variant in ("A", "B", "C", "D"):
        members = []
        if variant in {"A", "B"}:
            member_axis = [(None, seed) for seed in seeds]
        else:
            member_axis = [(reservoir_seed, seed) for reservoir_seed in reservoir_seeds for seed in seeds]
        for reservoir_seed, lightgbm_seed in member_axis:
            print(
                f"fit {variant}: reservoir={reservoir_seed} lightgbm={lightgbm_seed}",
                flush=True,
            )
            if variant == "A":
                matrix = eng._AugmentedMatrix(arrays["base_x"], [])
                feature_order = list(base_feature_order)
                added_features = 0
            elif variant == "B":
                matrix = eng._AugmentedMatrix(arrays["base_x"], [memory])
                feature_order = list(base_feature_order) + [f"simple_memory_PCA8_pc{index:02d}" for index in range(8)]
                added_features = 8
            elif variant == "C":
                matrix = eng._AugmentedMatrix(arrays["base_x"], [esn_features[int(reservoir_seed)]])
                feature_order = list(base_feature_order) + esn_files[str(reservoir_seed)]["feature_names"]
                added_features = 8
            else:
                matrix = eng._AugmentedMatrix(arrays["base_x"], [memory, esn_features[int(reservoir_seed)]])
                feature_order = (
                    list(base_feature_order)
                    + [f"simple_memory_PCA8_pc{index:02d}" for index in range(8)]
                    + esn_files[str(reservoir_seed)]["feature_names"]
                )
                added_features = 16
            fit_started = time.monotonic()
            model = eng._fit_lgbm(matrix, target, weights, train_mask, int(lightgbm_seed))
            fit_seconds = float(time.monotonic() - fit_started)
            expected_name = {
                "A": f"baseline_l{lightgbm_seed}",
                "B": "simple_memory_pca8",
                "C": candidate_id,
                "D": "D_memory_plus_8f74a7951e10c11830baba75",
            }[variant]
            prediction_started = time.monotonic()
            computed = np.asarray(model.predict(matrix[rows]), dtype=np.float32)
            prediction_seconds = float(time.monotonic() - prediction_started)
            if variant != "D":
                reference = _load_stored_prediction(
                    expected_name, reservoir_seed, int(lightgbm_seed), rows,
                    70 + added_features,
                )
                difference = np.abs(computed.astype(np.float64) - np.asarray(reference, dtype=np.float64))
                max_abs = float(difference.max(initial=0.0))
                if max_abs > 2e-5:
                    raise RuntimeError(f"{variant} saved-prediction replay differs by {max_abs:.8g}")
            else:
                prediction_path, prediction_metadata = _save_prediction(
                    expected_name, reservoir_seed, int(lightgbm_seed), computed, rows, 86,
                )
                max_abs = 0.0
            key = f"r{reservoir_seed}_l{lightgbm_seed}" if reservoir_seed is not None else f"l{lightgbm_seed}"
            model_path = model_root / variant / f"{key}.txt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(str(model_path))
            loaded_model = lgb.Booster(model_file=str(model_path))
            loaded_prediction = np.asarray(loaded_model.predict(matrix[rows]), dtype=np.float32)
            roundtrip_difference = float(np.max(np.abs(loaded_prediction.astype(np.float64) - computed.astype(np.float64))))
            if roundtrip_difference > 1e-7:
                raise RuntimeError(f"serialized {variant} model changed predictions by {roundtrip_difference:.8g}")
            model_objects[variant][key] = loaded_model
            member = {
                "reservoir_seed": reservoir_seed,
                "lightgbm_seed": int(lightgbm_seed),
                "model_path": str(model_path.relative_to(ROOT)),
                "model_sha256": _sha256(model_path),
                "feature_count": int(matrix.shape[1]),
                "feature_order_sha256": hashlib.sha256(json.dumps(feature_order, separators=(",", ":")).encode()).hexdigest(),
                "fit_seconds": fit_seconds,
                "validation_prediction_seconds": prediction_seconds,
                "saved_prediction_max_abs_error": max_abs,
                "bundle_roundtrip_max_abs_error": roundtrip_difference,
            }
            if variant == "D":
                member["prediction_path"] = str(prediction_path.relative_to(ROOT))
                member["prediction_metadata"] = prediction_metadata
            members.append(member)
            del model, matrix
            _check_active_limit()
        procedures[variant] = {
            "name": FREEZE_CONFIG["procedures"][variant]["name"],
            "candidate_id": FREEZE_CONFIG["procedures"][variant]["candidate_id"],
            "feature_count": 70 + (0 if variant == "A" else 8 if variant in {"B", "C"} else 16),
            "base_feature_order": base_feature_order,
            "base_feature_schema": base_feature_schema,
            "added_features": 0 if variant == "A" else 8 if variant in {"B", "C"} else 16,
            "members": members,
            "ensemble_weights": FREEZE_CONFIG["prediction_ensembles"][variant]["weights"],
            "training": common_training,
            "target": {
                "column": focused.CONFIG["baseline"]["target"],
                "definition": focused.CONFIG["baseline"]["target_definition"],
                "nominal_decision": "Opened+1m; Polymarket market start Opened+2m; T-60 contract",
                "label_available_at": "Opened+7m",
            },
            "input_availability": {
                "decision_at": "Opened+1m after the Opened candle has closed",
                "market_start_at": "Opened+2m (T-60 decision)",
                "raw_input_cutoff": "Only observations available by the decision timestamp; future rows are not consulted",
            },
            "preprocessing": {
                "baseline": "BASE70 source features are consumed in their frozen order; no extra scaling or imputation is applied by this comparison; LightGBM receives native missing values.",
                "invalid_dynamic_rows": "Memory and ESN state reset on an invalid row or any non-one-minute timestamp gap; no synthetic rows are inserted; common scoring rows require every dynamic representation to be finite.",
            },
        }
        if variant in {"B", "D"}:
            procedures[variant]["memory_transform"] = {
                "path": str(memory_copy.relative_to(ROOT)),
                "sha256": _sha256(memory_copy),
                "bundle": memory_bundle,
                "raw_input_columns": list(memory_bundle["identity"]["input_columns"]),
                "raw_feature_names": list(memory_feature_names(memory_bundle["identity"]["input_columns"])),
                "lags_minutes": list(MEMORY_LAGS),
                "ema_half_lives_minutes": list(MEMORY_HALFLIVES),
                "ema_alpha_definition": "alpha = 1 - 0.5 ** (1 / half_life_minutes); recursive EMA initialized to the first valid scaled input after reset",
                "deviation_features": ["current_scaled_input - EMA(60m)", "current_scaled_input - EMA(720m)"],
                "readiness_rule": f"first output after {MEMORY_MAX_LAG + 1} consecutive valid one-minute inputs following reset",
                "feature_names": [f"simple_memory_PCA8_pc{index:02d}" for index in range(8)],
                "gap_and_warmup": focused.CONFIG["memory_control"]["gap_rule"],
                "fit_scope": "input scaler and PCA parameters are fit-core-only and copied byte-for-byte from the frozen source bundle",
            }
        if variant in {"C", "D"}:
            procedures[variant]["esn"] = {
                "configuration": candidate_record,
                "input_columns": list(esn_config.selected_columns),
                "input_groups": list(esn_config.input_groups),
                "transform_scope": "input scaler and PCA8 are fit-core-only artifacts copied from the frozen focused-search bundles",
                "state_readiness_rule": "bundle warmup_steps plus the current valid row on one-minute contiguous input",
                "reservoir_seed_bundles": esn_files,
                "feature_order": "BASE70 + memory PCA8 (for D) + ESN PCA8; reservoir-specific names are frozen in each seed bundle",
                "state_reset": "reset on invalid row or any non-one-minute timestamp gap; no synthetic rows",
            }
        procedures[variant]["provenance"] = {
            "source_experiment_id": focused.CONFIG["experiment_id"],
            "source_commit": source_identity["frozen_source_commit"],
            "source_code_hashes": source_identity["source_hashes"],
            "source_data_identity": source_identity["prepared_search_identity"],
            "runtime_versions": {
                "python": os.sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "lightgbm": lgb.__version__,
                "scipy": scipy.__version__,
                "numba": numba.__version__,
                "threadpoolctl": threadpoolctl.__version__,
            },
            "feature_definition_source_hashes": _feature_definition_sources(),
            "base_feature_order_source": focused.CONFIG["baseline_feature_summary_path"],
            "base_feature_order_sha256": _sha256(ROOT / focused.CONFIG["baseline_feature_summary_path"]),
        }
        _atomic_json(BUNDLE_DIR / f"{variant}_manifest.json", procedures[variant])
    return {
        "procedures": procedures,
        "model_objects": model_objects,
        "model_fit_and_export_seconds": float(time.monotonic() - fit_start),
    }


def _build_comparison(arrays: dict, candidate_id: str, memory: np.ndarray,
                      esn_features: dict[int, np.ndarray], rows: np.ndarray) -> tuple[dict, dict]:
    y = np.asarray(arrays["target"][rows], dtype=np.int8)
    members = _member_predictions(candidate_id, arrays, rows, memory, esn_features)
    ensemble = _ensemble_predictions(members)
    individual = {}
    per_seed_metrics = {}
    for variant, predictions in members.items():
        individual[variant] = [_metrics(y, p) for p in predictions.values()]
        per_seed_metrics[variant] = [
            {"member": member, **_metrics(y, probability)}
            for member, probability in predictions.items()
        ]
    summary = {
        variant: {
            "mean_individual_model_metrics": {
                key: float(np.mean([row[key] for row in rows_for_seed]))
                for key in ("log_loss", "brier", "auc")
            },
            "uniform_ensemble_metrics": _metrics(y, ensemble[variant]),
            "member_count": len(members[variant]),
        }
        for variant, rows_for_seed in individual.items()
    }
    pair_metrics = {}
    pair_intervals = _bootstrap_pairs(y, arrays["base_decision_at"][rows], ensemble)
    for left, right in [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("A", "D")]:
        a = summary[left]["uniform_ensemble_metrics"]
        b = summary[right]["uniform_ensemble_metrics"]
        pair_metrics[f"{right}-{left}"] = {
            "delta_log_loss": a["log_loss"] - b["log_loss"],
            "delta_brier": a["brier"] - b["brier"],
            "delta_auc": b["auc"] - a["auc"],
            "intervals": pair_intervals["intervals"][f"{right}-{left}"],
        }
    seed_pairs = []
    for lightgbm_seed in FREEZE_CONFIG["model_fit"]["lightgbm_seeds"]:
        seed_metrics = {"lightgbm_seed": int(lightgbm_seed)}
        for variant in ("A", "B"):
            seed_metrics[variant] = _metrics(y, members[variant][str(lightgbm_seed)])
        for variant in ("C", "D"):
            same_lgbm = [members[variant][f"{reservoir_seed}:{lightgbm_seed}"]
                         for reservoir_seed in FREEZE_CONFIG["model_fit"]["reservoir_seeds"]]
            seed_metrics[variant] = _metrics(y, np.mean(same_lgbm, axis=0))
        seed_pairs.append(seed_metrics)
    results = {
        "rows": int(len(rows)),
        "decision_rows_sha256": _digest_array(arrays["base_decision_at"][rows].astype(np.int64)),
        "target_sha256": _digest_array(y),
        "target_positive_fraction": float(np.mean(y)),
        "procedures": summary,
        "per_seed_metrics": per_seed_metrics,
        "uniform_ensemble_pairwise_differences": pair_metrics,
        "matched_lightgbm_seed_comparisons": seed_pairs,
        "bootstrap": pair_intervals,
        "seed_metric_definition": "mean metric over individual fits; this is distinct from the metric computed on each procedure's uniformly averaged probabilities",
        "interpretation": FREEZE_CONFIG["D_interpretation"],
    }
    return results, members


def _verify_stream_and_checkpoint(arrays: dict, candidate_id: str, candidate: dict,
                                  memory_bundle: dict, esn_config, esn_bundles: dict[int, dict]) -> dict:
    eng = expansion.engine_runner
    raw_opened = np.asarray(arrays["raw_opened"])
    raw_x = arrays["raw_x"]
    raw_valid = np.asarray(arrays["raw_valid"], dtype=bool)
    validation_start = int(np.searchsorted(raw_opened, eng.VALIDATION_START_NS, side="left"))
    width = 128
    start = max(1, min(validation_start, len(raw_opened) - width - 1))
    # Batch and single-row updates use exactly the same ordered development observations.
    timestamps = pd.to_datetime(raw_opened[start:start + width], unit="ns", utc=True)
    columns = list(esn_config.selected_columns)
    input_positions = np.asarray([eng.INPUT_COLUMNS.index(name) for name in columns], dtype=np.int64)
    values = np.asarray(raw_x[start:start + width, :][:, input_positions], dtype=np.float32)
    valid = raw_valid[start:start + width].copy()
    if width > 30:
        valid[30] = False
    # Use the saved configuration and transform state; the source bundles are loaded in this fresh process.
    seed = int(FREEZE_CONFIG["model_fit"]["reservoir_seeds"][0])
    source_esn = esn_bundles[seed]
    scaler = FeatureScaler.from_dict(source_esn["scaler"])
    warmup = int(source_esn["warmup_steps"])
    batch_engine = ESNEngine(esn_config, seed, scaler, warmup_steps=warmup)
    batch_states, batch_ready = batch_engine.update_many(timestamps, values, valid_mask=valid)
    stream_engine = ESNEngine(esn_config, seed, scaler, warmup_steps=warmup)
    stream_states = np.full_like(batch_states, np.nan)
    stream_ready = np.zeros_like(batch_ready)
    for index, (timestamp, value, is_valid) in enumerate(zip(timestamps, values, valid)):
        state = stream_engine.update_one(timestamp, value, valid=bool(is_valid))
        if state is not None:
            stream_states[index] = state
            stream_ready[index] = True
    np.testing.assert_array_equal(batch_ready, stream_ready)
    np.testing.assert_allclose(batch_states[batch_ready], stream_states[stream_ready], rtol=1e-6, atol=1e-7)
    projection = eng._bundle_objects(source_esn)["projection"]
    batch_projected = projection.transform(batch_states[batch_ready])
    stream_projected = projection.transform(stream_states[stream_ready])
    np.testing.assert_allclose(batch_projected, stream_projected, rtol=0.0, atol=0.0)

    memory_width = len(memory_bundle["identity"]["input_columns"])
    memory_scaler = FeatureScaler.from_dict(memory_bundle["input_scaler"])
    memory_columns = np.asarray([eng.INPUT_COLUMNS.index(name) for name in memory_bundle["identity"]["input_columns"]], dtype=np.int64)
    memory_values = memory_scaler.transform(np.asarray(raw_x[start:start + width, :][:, memory_columns], dtype=np.float64))
    memory_batch = CausalMemoryState(memory_width)
    expected_memory, memory_ready = memory_batch.update_many(timestamps, memory_values, valid_mask=valid)
    memory_stream = CausalMemoryState(memory_width)
    actual_memory = np.full_like(expected_memory, np.nan)
    actual_memory_ready = np.zeros_like(memory_ready)
    for index, (timestamp, value, is_valid) in enumerate(zip(timestamps, memory_values, valid)):
        item = memory_stream.update_one(timestamp, value, valid=bool(is_valid))
        if item is not None:
            actual_memory[index] = item
            actual_memory_ready[index] = True
    np.testing.assert_array_equal(memory_ready, actual_memory_ready)
    np.testing.assert_allclose(expected_memory[memory_ready], actual_memory[memory_ready], rtol=1e-6, atol=1e-7)
    memory_mean = np.asarray(memory_bundle["pca_mean"], dtype=np.float64)
    memory_components = np.asarray(memory_bundle["pca_components"], dtype=np.float64)
    expected_memory_projected = (expected_memory[memory_ready].astype(np.float64) - memory_mean) @ memory_components.T
    actual_memory_projected = (actual_memory[actual_memory_ready].astype(np.float64) - memory_mean) @ memory_components.T
    np.testing.assert_allclose(expected_memory_projected, actual_memory_projected, rtol=0.0, atol=0.0)

    # Future perturbations must not change the already-emitted prefix.
    split = width // 2
    base_engine = ESNEngine(esn_config, seed, scaler, warmup_steps=warmup)
    prefix_states, prefix_ready = base_engine.update_many(timestamps[:split], values[:split], valid_mask=valid[:split])
    changed = values.copy()
    changed[split:] += np.float32(0.12345)
    future_engine = ESNEngine(esn_config, seed, scaler, warmup_steps=warmup)
    changed_prefix_states, changed_prefix_ready = future_engine.update_many(
        timestamps, changed, valid_mask=valid,
    )
    np.testing.assert_array_equal(prefix_ready, changed_prefix_ready[:split])
    np.testing.assert_allclose(prefix_states[prefix_ready], changed_prefix_states[:split][prefix_ready], rtol=0.0, atol=0.0)
    memory_prefix_state = CausalMemoryState(memory_width)
    memory_prefix, memory_prefix_ready = memory_prefix_state.update_many(
        timestamps[:split], memory_values[:split], valid_mask=valid[:split],
    )
    changed_memory_values = memory_values.copy()
    changed_memory_values[split:] += np.float32(0.12345)
    changed_memory_state = CausalMemoryState(memory_width)
    changed_memory, changed_memory_ready = changed_memory_state.update_many(
        timestamps, changed_memory_values, valid_mask=valid,
    )
    np.testing.assert_array_equal(memory_prefix_ready, changed_memory_ready[:split])
    np.testing.assert_allclose(memory_prefix[memory_prefix_ready], changed_memory[:split][memory_prefix_ready], rtol=0.0, atol=0.0)

    # The checkpoint must resume the stream at the exact saved timestamp and state.
    checkpoint_root = OUT / "verification" / "checkpoint"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    split = width // 2
    left_engine = ESNEngine(esn_config, seed, scaler, warmup_steps=warmup)
    left_engine.update_many(timestamps[:split], values[:split], valid_mask=valid[:split])
    checkpoint_path = checkpoint_root / "esn_state.json"
    left_engine.save_checkpoint(checkpoint_path)
    expected_tail, expected_ready = left_engine.update_many(timestamps[split:], values[split:], valid_mask=valid[split:])
    resumed = ESNEngine.load_checkpoint(checkpoint_path, expected_config=esn_config, expected_seed=seed)
    actual_tail, actual_ready = resumed.update_many(timestamps[split:], values[split:], valid_mask=valid[split:])
    np.testing.assert_array_equal(expected_ready, actual_ready)
    np.testing.assert_allclose(expected_tail[expected_ready], actual_tail[actual_ready], rtol=0.0, atol=0.0)
    memory_left = CausalMemoryState(memory_width)
    memory_left.update_many(timestamps[:split], memory_values[:split], valid_mask=valid[:split])
    memory_checkpoint = checkpoint_root / "memory_state.json"
    _atomic_json(memory_checkpoint, memory_left.to_dict())
    expected_memory_tail, expected_memory_ready = memory_left.update_many(
        timestamps[split:], memory_values[split:], valid_mask=valid[split:],
    )
    resumed_memory = CausalMemoryState.from_dict(json.loads(memory_checkpoint.read_text(encoding="utf-8")))
    actual_memory_tail, actual_memory_ready = resumed_memory.update_many(
        timestamps[split:], memory_values[split:], valid_mask=valid[split:],
    )
    np.testing.assert_array_equal(expected_memory_ready, actual_memory_ready)
    np.testing.assert_allclose(expected_memory_tail[expected_memory_ready], actual_memory_tail[actual_memory_ready], rtol=0.0, atol=0.0)
    return {
        "fresh_process": True,
        "stream_rows": int(width),
        "batch_stream_esn_equal": True,
        "batch_stream_esn_projection_equal": True,
        "batch_stream_memory_equal": True,
        "batch_stream_memory_projection_equal": True,
        "future_perturbation_prefix_invariant": True,
        "future_perturbation_memory_prefix_invariant": True,
        "esn_checkpoint_resume_equal": True,
        "memory_checkpoint_resume_equal": True,
        "valid_reset_case_included": True,
        "feature_order": list(esn_config.selected_columns),
    }


def _verify_full_representation_replay(arrays: dict, memory_bundle: dict, esn_config,
                                       esn_bundles: dict[int, dict],
                                       memory_features: np.ndarray,
                                       esn_features: dict[int, np.ndarray]) -> dict:
    eng = expansion.engine_runner
    raw_opened = np.asarray(arrays["raw_opened"], dtype=np.int64)
    raw_valid = np.asarray(arrays["raw_valid"], dtype=bool)
    raw_x = arrays["raw_x"]
    base_to_raw = np.asarray(arrays["base_to_raw"], dtype=np.int64)
    stream_rows = int(np.searchsorted(raw_opened, eng.SEARCH_END_NS - eng.MINUTE_NS, side="left"))
    if not len(base_to_raw) or int(base_to_raw[-1]) >= stream_rows:
        raise RuntimeError("saved BASE70 rows extend beyond the frozen search-validation input cutoff")
    block_rows = 65_536

    memory_columns = list(memory_bundle["identity"]["input_columns"])
    memory_positions = np.asarray([eng.INPUT_COLUMNS.index(name) for name in memory_columns], dtype=np.int64)
    memory_scaler = FeatureScaler.from_dict(memory_bundle["input_scaler"])
    memory_mean = np.asarray(memory_bundle["pca_mean"], dtype=np.float64)
    memory_components = np.asarray(memory_bundle["pca_components"], dtype=np.float64)
    memory_state = CausalMemoryState(len(memory_columns))
    memory_max_error = 0.0
    memory_replayed_rows = 0
    for start in range(0, stream_rows, block_rows):
        if start % (block_rows * 8) == 0:
            print(f"bundle replay memory: {start}/{stream_rows} raw rows", flush=True)
        end = min(stream_rows, start + block_rows)
        scaled = memory_scaler.transform(np.asarray(raw_x[start:end, :][:, memory_positions], dtype=np.float64))
        times = pd.to_datetime(raw_opened[start:end], unit="ns", utc=True)
        feature_block, ready = memory_state.update_many(
            times, scaled, valid_mask=raw_valid[start:end],
        )
        left = int(np.searchsorted(base_to_raw, start, side="left"))
        right = int(np.searchsorted(base_to_raw, end, side="left"))
        if right > left:
            locations = base_to_raw[left:right] - start
            keep = ready[locations]
            if keep.any():
                row_indices = np.arange(left, right, dtype=np.int64)[keep]
                projected = ((feature_block[locations[keep]].astype(np.float64) - memory_mean)
                             @ memory_components.T).astype(np.float32)
                expected = np.asarray(memory_features[row_indices], dtype=np.float32)
                if not np.isfinite(expected).all():
                    raise RuntimeError("memory feature cache is non-finite at a bundle-ready row")
                memory_max_error = max(memory_max_error, float(np.max(np.abs(projected - expected))))
                memory_replayed_rows += len(row_indices)
        _check_active_limit()

    expected_memory_rows = int(np.isfinite(memory_features).all(axis=1).sum())
    if memory_replayed_rows != expected_memory_rows or memory_max_error > 2e-5:
        raise RuntimeError(
            f"memory raw-stream bundle replay differs from cache: rows {memory_replayed_rows}/{expected_memory_rows}, "
            f"max error {memory_max_error:.8g}"
        )

    esn_positions = np.asarray([eng.INPUT_COLUMNS.index(name) for name in esn_config.selected_columns], dtype=np.int64)
    esn_results = {}
    for seed in FREEZE_CONFIG["model_fit"]["reservoir_seeds"]:
        print(f"bundle replay ESN seed {seed}: {stream_rows} raw rows", flush=True)
        bundle = esn_bundles[int(seed)]
        scaler = FeatureScaler.from_dict(bundle["scaler"])
        projection = eng._bundle_objects(bundle)["projection"]
        state = ESNEngine(esn_config, int(seed), scaler, warmup_steps=int(bundle["warmup_steps"]))
        feature_matrix = esn_features[int(seed)]
        maximum_error = 0.0
        replayed_rows = 0
        for start in range(0, stream_rows, block_rows):
            if start % (block_rows * 8) == 0:
                print(f"bundle replay ESN {seed}: {start}/{stream_rows} raw rows", flush=True)
            end = min(stream_rows, start + block_rows)
            values = np.asarray(raw_x[start:end, :][:, esn_positions], dtype=np.float32)
            times = pd.to_datetime(raw_opened[start:end], unit="ns", utc=True)
            states, ready = state.update_many(times, values, valid_mask=raw_valid[start:end])
            left = int(np.searchsorted(base_to_raw, start, side="left"))
            right = int(np.searchsorted(base_to_raw, end, side="left"))
            if right > left:
                locations = base_to_raw[left:right] - start
                keep = ready[locations]
                if keep.any():
                    row_indices = np.arange(left, right, dtype=np.int64)[keep]
                    projected = projection.transform(states[locations[keep]]).astype(np.float32)
                    expected = np.asarray(feature_matrix[row_indices], dtype=np.float32)
                    if not np.isfinite(expected).all():
                        raise RuntimeError(f"ESN feature cache is non-finite at a bundle-ready row for seed {seed}")
                    maximum_error = max(maximum_error, float(np.max(np.abs(projected - expected))))
                    replayed_rows += len(row_indices)
            _check_active_limit()
        expected_rows = int(np.isfinite(feature_matrix).all(axis=1).sum())
        if replayed_rows != expected_rows or maximum_error > 2e-5:
            raise RuntimeError(
                f"ESN raw-stream bundle replay differs from cache for seed {seed}: rows {replayed_rows}/{expected_rows}, "
                f"max error {maximum_error:.8g}"
            )
        esn_results[str(seed)] = {
            "rows": replayed_rows,
            "max_abs_error": maximum_error,
            "warmup_steps": int(bundle["warmup_steps"]),
        }
    return {
        "scope_end_exclusive_utc": "2025-07-01T00:00:00Z",
        "raw_stream_rows": stream_rows,
        "memory": {"rows": memory_replayed_rows, "max_abs_error": memory_max_error},
        "esn_by_reservoir_seed": esn_results,
        "all_saved_representation_rows_reproduced": True,
    }


def _measure_cold_and_normal_feature_cost(arrays: dict, memory_bundle: dict,
                                          esn_config, esn_bundles: dict[int, dict]) -> dict:
    eng = expansion.engine_runner
    raw_opened = np.asarray(arrays["raw_opened"], dtype=np.int64)
    raw_valid = np.asarray(arrays["raw_valid"], dtype=bool)
    raw_x = arrays["raw_x"]
    validation_start = int(np.searchsorted(raw_opened, eng.VALIDATION_START_NS, side="left"))
    required_esn = max(int(esn_bundles[seed]["warmup_steps"]) + 257
                       for seed in FREEZE_CONFIG["model_fit"]["reservoir_seeds"])
    scoped_opened = raw_opened[:validation_start]
    scoped_valid = raw_valid[:validation_start]
    pairs = scoped_valid[1:] & scoped_valid[:-1] & (np.diff(scoped_opened) == eng.MINUTE_NS)
    breaks = np.flatnonzero(~pairs) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(scoped_opened)]
    segments = [(int(start), int(end)) for start, end in zip(starts, ends)
                if int(end) <= validation_start and int(end - start) >= required_esn]
    if not segments:
        raise RuntimeError("no pre-validation contiguous development history is long enough for frozen ESN warm-up")
    segment_start, segment_end = min(segments, key=lambda item: item[0])
    memory_warm = MEMORY_MAX_LAG + 1
    memory_start = segment_start
    memory_end = memory_start + memory_warm + 257
    if memory_end > segment_end:
        raise RuntimeError("simple-memory cold-start window does not fit a contiguous development segment")
    memory_columns = list(memory_bundle["identity"]["input_columns"])
    memory_positions = np.asarray([eng.INPUT_COLUMNS.index(name) for name in memory_columns], dtype=np.int64)
    memory_scaler = FeatureScaler.from_dict(memory_bundle["input_scaler"])
    memory_projection = np.asarray(memory_bundle["pca_components"], dtype=np.float64)
    memory_mean = np.asarray(memory_bundle["pca_mean"], dtype=np.float64)
    memory_raw = np.asarray(raw_x[memory_start:memory_end, :][:, memory_positions], dtype=np.float64)
    memory_times = pd.to_datetime(raw_opened[memory_start:memory_end], unit="ns", utc=True)
    memory_engine = CausalMemoryState(len(memory_columns))
    before = time.perf_counter()
    memory_values = memory_scaler.transform(memory_raw)
    cold_memory_features, memory_ready = memory_engine.update_many(
        memory_times[:memory_warm], memory_values[:memory_warm],
        valid_mask=raw_valid[memory_start:memory_start + memory_warm],
    )
    ready_block, _ = memory_engine.update_many(
        memory_times[memory_warm:], memory_values[memory_warm:],
        valid_mask=raw_valid[memory_start + memory_warm:memory_end],
    )
    combined_memory = np.concatenate((cold_memory_features, ready_block), axis=0)
    memory_ready_rows = np.flatnonzero(np.isfinite(combined_memory).all(axis=1))
    valid_memory_features = combined_memory[memory_ready_rows].astype(np.float64)
    projected_memory = (valid_memory_features - memory_mean) @ memory_projection.T
    if not memory_ready[-1]:
        raise RuntimeError("simple-memory warm-up did not reach the frozen ready state")
    memory_cold_seconds = float(time.perf_counter() - before)
    memory_normal_us = []
    memory_normal_state = CausalMemoryState(len(memory_columns))
    memory_normal_state.update_many(
        memory_times[:memory_warm], memory_values[:memory_warm],
        valid_mask=raw_valid[memory_start:memory_start + memory_warm],
    )
    for timestamp, raw_value, is_valid in zip(
        memory_times[memory_warm:], memory_raw[memory_warm:], raw_valid[memory_start + memory_warm:memory_end],
    ):
        before = time.perf_counter_ns()
        scaled_value = memory_scaler.transform(raw_value.reshape(1, -1))[0]
        feature = memory_normal_state.update_one(timestamp, scaled_value, valid=bool(is_valid))
        if feature is not None:
            (feature.astype(np.float64) - memory_mean) @ memory_projection.T
        memory_normal_us.append((time.perf_counter_ns() - before) / 1000.0)

    esn_costs = {}
    input_positions = np.asarray([eng.INPUT_COLUMNS.index(name) for name in esn_config.selected_columns], dtype=np.int64)
    for seed in FREEZE_CONFIG["model_fit"]["reservoir_seeds"]:
        bundle = esn_bundles[int(seed)]
        warmup_steps = int(bundle["warmup_steps"])
        start = segment_start
        end = start + warmup_steps + 257
        if end > segment_end:
            raise RuntimeError(f"pre-validation reset segment is too short for ESN seed {seed}")
        values = np.asarray(raw_x[start:end, :][:, input_positions], dtype=np.float32)
        timestamps = pd.to_datetime(raw_opened[start:end], unit="ns", utc=True)
        valid = raw_valid[start:end]
        scaler = FeatureScaler.from_dict(bundle["scaler"])
        projection = expansion.engine_runner._bundle_objects(bundle)["projection"]
        state = ESNEngine(esn_config, int(seed), scaler, warmup_steps=warmup_steps)
        before = time.perf_counter()
        states, ready = state.update_many(timestamps, values, valid_mask=valid)
        projected = projection.transform(states[ready]) if ready.any() else np.empty((0, 8), dtype=np.float32)
        cold_seconds = float(time.perf_counter() - before)
        normal_state = ESNEngine(esn_config, int(seed), scaler, warmup_steps=warmup_steps)
        normal_state.update_many(timestamps[:warmup_steps + 1], values[:warmup_steps + 1], valid_mask=valid[:warmup_steps + 1])
        normal_us = []
        for timestamp, value, is_valid in zip(timestamps[warmup_steps + 1:], values[warmup_steps + 1:], valid[warmup_steps + 1:]):
            before = time.perf_counter_ns()
            item = normal_state.update_one(timestamp, value, valid=bool(is_valid))
            if item is not None:
                projection.transform(item.reshape(1, -1))
            normal_us.append((time.perf_counter_ns() - before) / 1000.0)
        esn_costs[str(seed)] = {
            "cold_start_rows": int(len(values)),
            "warmup_rows_to_first_feature": warmup_steps + 1,
            "cold_start_seconds_after_runtime_initialization": cold_seconds,
            "features_emitted_during_cold_start": int(ready.sum()),
            "normal_update_and_projection_us": {
                "samples": len(normal_us),
                "p50": float(np.quantile(normal_us, 0.50)),
                "p95": float(np.quantile(normal_us, 0.95)),
                "p99": float(np.quantile(normal_us, 0.99)),
            },
        }
    return {
        "measurement_scope": "causal 14-column development feature stream through memory/ESN state and PCA; excludes upstream BASE70 raw-market feature construction",
        "pre_validation_contiguous_development_rows": int(segment_end - segment_start),
        "memory": {
            "cold_start_rows": int(memory_end - memory_start),
            "warmup_rows_to_first_feature": memory_warm,
            "cold_start_seconds_after_runtime_initialization": memory_cold_seconds,
            "features_emitted_during_cold_start": int(len(projected_memory)),
            "normal_update_and_projection_us": {
                "samples": len(memory_normal_us),
                "p50": float(np.quantile(memory_normal_us, 0.50)),
                "p95": float(np.quantile(memory_normal_us, 0.95)),
                "p99": float(np.quantile(memory_normal_us, 0.99)),
            },
        },
        "esn_by_reservoir_seed": esn_costs,
    }


def _write_report(results: dict, manifests: dict, runtime: dict, equivalence: dict,
                  comparability: dict) -> None:
    p = results["procedures"]
    lines = [
        "# BTC candidate quality freeze and development comparison — 2026-10-22",
        "",
        "> All scores below use the previously reused `search_validation` interval. They are development selection results, not independent confirmation.",
        "",
        "## Frozen procedures",
        "",
        "A is BASE70; B adds the fixed simple-memory PCA8 representation; C adds the focused-search ESN PCA8 candidate `8f74a7951e10c11830baba75`; D adds both frozen representations in the preregistered order. D was frozen in `configs/research/btc_candidate_quality_freeze_20261022.json` before fitting.",
        "",
        "The common target is the Binance index proxy `Close[Opened+6m] >= Open[Opened+2m]`, available at Opened+7m. Nominal decisions are at Opened+1m, the T−60 point. LightGBM uses fit-core only, weights 0.59 on decision rows and 0.1025 otherwise, the unchanged one-minute label-availability purge, 31 boosting rounds, no early stopping, and seeds 20261101–20261103. C/D cross those seeds with reservoir seeds 20261023–20261025.",
        "",
        "The complete ordered features, transforms, seed matrices, model files, source hashes, and fit provenance are in the local bundle manifests under `data/analysis/polymarket/BTC/btc_candidate_quality_freeze_20261022/bundles/`.",
        "",
        "### Frozen representation parameters",
        "",
        f"- B memory inputs: `{manifests['B']['memory_transform']['raw_input_columns']}`; lags `{manifests['B']['memory_transform']['lags_minutes']}` minutes; EWM half-lives `{manifests['B']['memory_transform']['ema_half_lives_minutes']}` minutes; deviations `{manifests['B']['memory_transform']['deviation_features']}`. Input scaling and PCA8 use {manifests['B']['memory_transform']['bundle']['identity']['pca_fit_rows']:,} fit-core rows, randomized PCA seed `{manifests['B']['memory_transform']['bundle']['identity']['projection_seed']}`, and the exact scaler/PCA arrays in the copied transform bundle.",
        f"- C selected ESN configuration: `{json.dumps(results['selected_esn_candidate'], sort_keys=True)}`; raw inputs `{manifests['C']['esn']['input_columns']}`. Reservoir-specific matrices, input scaler, PCA8, warm-up counts, and projection seeds are frozen for seeds `{FREEZE_CONFIG['model_fit']['reservoir_seeds']}` in the C manifest.",
        "- Every procedure manifest stores the exact BASE70 order and one source-backed definition/source hash per column, plus complete model parameters per LightGBM seed.",
        "",
        "## Common observations and metrics",
        "",
        f"The comparison uses {results['rows']:,} identical decision timestamps and labels. Decision SHA-256: `{results['decision_rows_sha256']}`; target SHA-256: `{results['target_sha256']}`. Feature readiness and label availability were intersected once before scoring.",
        "",
        "Mean individual-model metrics are kept separate from metrics on the uniform ensemble prediction.",
        "",
        "| Procedure | Fits | Mean individual LL | Mean individual Brier | Mean individual AUC | Uniform ensemble LL | Uniform ensemble Brier | Uniform ensemble AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("A", "B", "C", "D"):
        item = p[variant]
        mean = item["mean_individual_model_metrics"]
        ens = item["uniform_ensemble_metrics"]
        lines.append(
            f"| {variant} | {item['member_count']} | {mean['log_loss']:.9f} | {mean['brier']:.9f} | {mean['auc']:.6f} | {ens['log_loss']:.9f} | {ens['brier']:.9f} | {ens['auc']:.6f} |"
        )
    lines += [
        "",
        "Positive LL/Brier deltas and positive AUC deltas favor the procedure on the right.",
        "",
        "| Pair | ΔLL | Paired 95% block CI | ΔBrier | Paired 95% block CI | ΔAUC | Paired 95% block CI |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for pair in ("B-A", "C-A", "C-B", "D-B", "D-C", "D-A"):
        item = results["uniform_ensemble_pairwise_differences"][pair]
        fmt_ci = lambda metric: f"[{item['intervals'][metric][0]:+.9f}, {item['intervals'][metric][1]:+.9f}]"
        lines.append(
            f"| {pair} | {item['delta_log_loss']:+.9f} | {fmt_ci('log_loss')} | {item['delta_brier']:+.9f} | {fmt_ci('brier')} | {item['delta_auc']:+.6f} | {fmt_ci('auc')} |"
        )
    lines += [
        "",
        f"The paired intervals use {BOOTSTRAP_REPLICATES:,} shared non-circular moving calendar-day block draws via `utils.btc_oof_audit.moving_calendar_block_multiplicity`, block length {BOOTSTRAP_BLOCK_DAYS} calendar days, seed {BOOTSTRAP_SEED}. They describe variability within this reused interval and do not correct the earlier search-selection effect.",
        "",
        "### Matched LightGBM seed comparisons",
        "",
        "For C−B and D−B, the three ESN-seed predictions are averaged within each LightGBM seed; B appears once per LightGBM seed.",
        "",
        "| LightGBM seed | A LL | B LL | C LL (mean reservoir prediction) | D LL (mean reservoir prediction) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in results["matched_lightgbm_seed_comparisons"]:
        lines.append(
            f"| {row['lightgbm_seed']} | {row['A']['log_loss']:.9f} | {row['B']['log_loss']:.9f} | {row['C']['log_loss']:.9f} | {row['D']['log_loss']:.9f} |"
        )
    lines += [
        "",
        "### Individual fitted-model metrics",
        "",
        "| Procedure | Reservoir seed | LightGBM seed | LL | Brier | AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ("A", "B", "C", "D"):
        for row in results["per_seed_metrics"][variant]:
            member = row["member"]
            if ":" in member:
                reservoir_seed, lightgbm_seed = member.split(":", 1)
            elif member.startswith("r"):
                reservoir_seed, lightgbm_seed = member[1:].split("_l", 1)
            else:
                reservoir_seed, lightgbm_seed = "—", member.removeprefix("l")
            lines.append(
                f"| {variant} | {reservoir_seed} | {lightgbm_seed} | {row['log_loss']:.9f} | {row['brier']:.9f} | {row['auc']:.6f} |"
            )
    lines += [
        "",
        "## ESN incremental-information decision",
        "",
    ]
    d_minus_b = results["uniform_ensemble_pairwise_differences"]["D-B"]
    low, high = d_minus_b["intervals"]["log_loss"]
    if d_minus_b["delta_log_loss"] > 0 and low > 0:
        decision = "D shows a positive incremental development signal beyond B under the frozen point-and-interval rule. This remains conditional on the repeated search-validation period."
    elif d_minus_b["delta_log_loss"] > 0:
        decision = "D's point estimate is positive beyond B, but the paired interval includes zero; this development run does not establish incremental information beyond B."
    else:
        decision = "D does not improve ensemble log loss over B on this development comparison; it provides no positive incremental signal beyond B under the frozen rule."
    lines.append(decision)
    lines += [
        "",
        "## Comparability audit",
        "",
        f"- Exact common row IDs and labels: **{comparability['same_decision_ids_and_labels']}** ({comparability['rows']:,} rows; identical row and label hashes).",
        f"- Same input cutoff and BASE70 matrix: **{comparability['same_input_cutoff_and_baseline']}**; T−60 inputs only, no label available at or after the search cutoff.",
        f"- Common fit parameters, weights, seeds, and purge: **{comparability['same_training_contract']}**; all procedure models use the same fit-row hash.",
        f"- Scaler/PCA fit provenance: **{comparability['transform_fit_scope_verified']}**; memory scaler/PCA and ESN scaler/PCA are fit-core-only artifacts.",
        f"- Fit-core scaler reconstruction matches the frozen memory/ESN scalers: **{comparability['transform_fit_scope_audit']['input_scalers_match_independent_fit_core_reconstruction']}**; memory PCA sample rows: {comparability['transform_fit_scope_audit']['memory_pca_fit_rows']:,}; ESN PCA sample rows per seed: {FREEZE_CONFIG['model_fit']['reservoir_seeds']} × {focused.CONFIG['screening']['pca_fit_rows']:,}.",
        f"- Maximum A/B/C saved-probability replay error: `{comparability['maximum_saved_prediction_replay_error']:.9g}`; maximum model save/load replay error: `{comparability['maximum_model_bundle_roundtrip_error']:.9g}`.",
        "- Three orchestration fixes: the auxiliary matrix is reconstructed for every seed fit; monthly pair-trigger comparisons read each `delta_logloss` field; latency output maps candidate aliases to saved bundle keys. The first can affect auxiliary PCA4 and simple-memory fits, the second only the optional pair trigger, and the third only latency lookup. No conditional pair was part of this A/B/C comparison; the old PCA4 result remains a historical artifact outside it. Historical probability files do not carry a per-artifact runner hash, so their exact producing orchestration revision cannot be established from run identity alone. A/B/C are scored from those saved probabilities and replayed with fit-core-only model exports; the report records the maximum prediction deltas. The bundle records the exact current runner/source hashes used for its model exports.",
        "",
        "## Bundle replay and streaming checks",
        "",
        f"Fresh-process checks: `{json.dumps(equivalence, sort_keys=True)}`.",
        f"Fit-core model export time (A/B/C refits for bundle export plus D): `{runtime['model_fit_and_export_seconds']:.2f}s`; D-only fit time: `{runtime['D_fit_seconds']:.2f}s`.",
        f"Fresh full search-prefix bundle-to-feature replay time: `{runtime['fresh_bundle_representation_replay_seconds']:.2f}s`.",
        f"Main runner active time: `{runtime['active_seconds']:.2f}s`; frozen task cap: `{ACTIVE_LIMIT_SECONDS}s`.",
        f"Source-recorded full-stream representation generation: B `{runtime['source_B_full_generation_seconds']:.2f}s`; C `{runtime['source_C_full_generation_seconds']:.2f}s`; both `{runtime['full_representation_replay_seconds']:.2f}s`. Validation batch prediction time for 52,120 rows: `{runtime['validation_batch_prediction_seconds_by_procedure']}`.",
        f"Measured end-to-end feature-generation-plus-prediction totals from the existing BASE70 and raw 14-column arrays: `{runtime['full_path_seconds_by_procedure']}`.",
        f"Process peak working set: `{runtime['peak_working_set_bytes']}` bytes. Resource snapshot before fit: `{runtime['resource_snapshot_before_fit']}`; after replay: `{runtime['resource_snapshot_after_replay']}`.",
        f"Per-decision ensemble inference latency (256 rows): `{runtime['latency']}`. Cold-start and normal state-update/projection latency: `{runtime['cold_start']}`.",
        "Feature generation starts from the cached BASE70 and raw 14-column development arrays. The upstream raw-market-to-BASE70 build was not rerun or timed in this comparison; its complete source code, configuration, ordered feature definitions, and hashes are included in the bundle manifests.",
        "",
        "## Test outside search_validation",
        "",
        "The former one-path split designated 2025-07-01 through 2026-07-01 as confirmation. Its configurations and manifests show that it has already been used for prior model comparisons, so it must not be called a fresh test or reused to make this result look independent. The project holdout is 2026-07-01 through exclusive 2026-10-07. The project research contract records prior exposure of historical holdout results; it is also not unseen. This task did not read any confirmation or holdout inputs, predictions, labels, or scores.",
        "",
        "The only eligible existing external period for a one-time frozen transfer of these candidates is the protected 2026-07-01–2026-10-07 holdout. Keep it protected until an explicitly authorized run. At the test start, refit A/B/C/D once using all pre-test rows whose labels were available by 2026-07-01T00:00:00Z, preserving all feature, model, seed, ensemble, target, weight, and purge settings. Fit memory/ESN scalers and PCA only on those allowed training rows. Fix 31 iterations with no test early stopping. Carry only causal feature state into the first decision, warm it using pre-test history, and never use a later test observation for an earlier decision.",
        "",
        "Use one common mask of test decision timestamps with valid BASE70, memory, all ESN members, and a label released before the frozen exclusive cutoff. Log loss is primary; Brier and AUC are secondary. Report B−A, C−A, C−B, D−B, D−C, and D−A using equal-weight ensembles; use the same 3-calendar-day paired moving block bootstrap from `utils.btc_oof_audit.moving_calendar_block_multiplicity` and publish all exclusions. Execute once, save input/code/model hashes before scoring, do not revise settings after seeing results, and label the interval historical/exposed rather than fresh.",
        "",
        "Development comparison command: `python run_btc_candidate_quality_20261022.py`.",
        "Future protected-test gate command: `python run_btc_candidate_holdout_20261022.py`. Its separate config defaults to `allow_protected_data: false`; this gate reads only that config and fails before importing any data pipeline or opening a protected file. The protected scorer is intentionally not implemented or run here; it must be supplied as part of a separately authorized evaluation.",
        "",
        "## Provenance and limits",
        "",
        f"- Source code commit: `{runtime['source_commit']}`; focused-search original frozen source commit: `{runtime['focused_run_source_commit']}`.",
        f"- Runner code hash: `{runtime['runner_sha256']}`; freeze-config hash: `{runtime['freeze_config_sha256']}`.",
        f"- Prior development selection results remain +0.000136588 mean ΔLL for B and +0.000101431 mean ΔLL for C; both came from repeated use of search_validation.",
        f"- Output: `{OUT.relative_to(ROOT)}`; report: `{REPORT_PATH.relative_to(ROOT)}`.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global RUN_STARTED
    os.chdir(ROOT)
    RUN_STARTED = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    source_ledger_path = SOURCE_OUT / "runtime_ledger.json"
    source_ledger_sha256_before = _sha256(source_ledger_path)
    identity, arrays = _configure_source_run()
    candidate_id, candidate, memory, esn_features, loaded = _load_frozen_inputs(arrays)
    train_mask, rows = _common_masks(arrays, candidate_id, memory, esn_features)
    y = np.asarray(arrays["target"][rows], dtype=np.int8)
    decision_ids = np.asarray(arrays["base_decision_at"][rows], dtype=np.int64)
    if len(np.unique(decision_ids)) != len(decision_ids):
        raise RuntimeError("decision IDs are not unique on the common mask")
    base_order = _load_feature_order()
    source_identity = identity
    memory_bundle = json.loads((SOURCE_OUT / "memory_control" / "projection_bundle.json").read_text(encoding="utf-8"))
    esn_bundle_payloads = {
        int(seed): json.loads((SOURCE_OUT / "bundles" / f"{expansion.engine_runner._config_key(loaded['config'])}_seed_{seed}.json").read_text(encoding="utf-8"))
        for seed in FREEZE_CONFIG["model_fit"]["reservoir_seeds"]
    }
    transform_audit = _verify_transform_fit_scope(arrays, memory_bundle, loaded["config"], esn_bundle_payloads)
    print(json.dumps({"stage": "fit_scope_verified", **transform_audit}, sort_keys=True), flush=True)
    resource_start = expansion._resource_snapshot()
    if resource_start["available_memory_bytes"] < 8 * 2**30:
        raise RuntimeError("less than 8 GiB RAM is available; preserving headroom and stopping before model fitting")
    model_export = _fit_and_bundle_models(
        arrays, candidate_id, candidate, memory, esn_features, train_mask, rows,
        base_order, source_identity,
    )
    d_seconds = float(sum(
        item["fit_seconds"] for item in model_export["procedures"]["D"]["members"]
    ))
    print("computing common metrics and preregistered paired bootstrap", flush=True)
    results, _ = _build_comparison(arrays, candidate_id, memory, esn_features, rows)
    results["fit_rows"] = int(train_mask.sum())
    results["fit_row_indices_sha256"] = _digest_array(np.flatnonzero(train_mask).astype(np.int64))
    results["selected_esn_candidate"] = candidate["effective_config"]
    results["memory_projection_source_sha256"] = _sha256(SOURCE_OUT / "memory_control" / "projection_bundle.json")
    results["source_run_identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()
    _atomic_json(OUT / "candidate_comparison.json", results)
    _atomic_json(BUNDLE_DIR / "run_manifest.json", {
        "experiment_id": FREEZE_CONFIG["experiment_id"],
        "freeze_config_sha256": _sha256(FREEZE_CONFIG_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "source_commit": _git_commit(),
        "focused_run_source_commit": identity["frozen_source_commit"],
        "source_run_identity": identity,
        "source_artifact_hashes": identity["source_hashes"],
        "package_versions": {
            "python": os.sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
            "threadpoolctl": threadpoolctl.__version__,
        },
        "scope": FREEZE_CONFIG["scope"],
        "resource_snapshot_before_fit": resource_start,
        "confirmation_read": False,
        "holdout_read": False,
        "holdout_rows_materialized": 0,
        "source_focused_runtime_ledger_sha256": source_ledger_sha256_before,
        "fit_row_indices_sha256": results["fit_row_indices_sha256"],
        "decision_ids_sha256": results["decision_rows_sha256"],
        "target_sha256": results["target_sha256"],
        "data_provenance": arrays["prepared"]["identity"],
        "transform_fit_scope_audit": transform_audit,
    })
    replay_memory_bundle = json.loads((BUNDLE_DIR / "transforms" / "simple_memory_projection.json").read_text(encoding="utf-8"))
    replay_esn_bundles = {
        int(seed): json.loads((BUNDLE_DIR / "transforms" / f"esn_seed_{seed}.json").read_text(encoding="utf-8"))
        for seed in FREEZE_CONFIG["model_fit"]["reservoir_seeds"]
    }
    equivalence = _verify_stream_and_checkpoint(
        arrays, candidate_id, candidate, replay_memory_bundle, loaded["config"], replay_esn_bundles,
    )
    bundle_replay_started = time.monotonic()
    full_bundle_replay = _verify_full_representation_replay(
        arrays, replay_memory_bundle, loaded["config"], replay_esn_bundles,
        memory, esn_features,
    )
    bundle_replay_seconds = float(time.monotonic() - bundle_replay_started)
    equivalence["full_search_prefix_bundle_replay"] = full_bundle_replay
    cold_cost = _measure_cold_and_normal_feature_cost(
        arrays, replay_memory_bundle, loaded["config"], replay_esn_bundles,
    )
    # Cached feature matrices are already checksum-bound to the source run. Record their established full-stream costs.
    original_summary = json.loads((SOURCE_OUT / "run_summary.json").read_text(encoding="utf-8"))
    memory_ledger = json.loads((SOURCE_OUT / "runtime_ledger.json").read_text(encoding="utf-8"))
    c_key = candidate_id
    c_generation_by_candidate = original_summary["latency_summary"]["state_generation_seconds_by_candidate"]
    if c_key not in c_generation_by_candidate:
        raise RuntimeError("source run summary is missing the selected candidate's full-stream generation time")
    c_generation_seconds = float(c_generation_by_candidate[c_key])
    if "memory_control" not in memory_ledger.get("stages_seconds", {}):
        raise RuntimeError("source run ledger is missing the full-stream memory generation time")
    memory_generation_seconds = float(memory_ledger["stages_seconds"]["memory_control"])
    full_replay_seconds = c_generation_seconds + memory_generation_seconds
    verification_started = time.monotonic()
    # Prediction latency includes matrix construction for one decision and every equally weighted model member.
    prediction_latencies = {variant: [] for variant in ("A", "B", "C", "D")}
    row_sample = rows[:min(256, len(rows))]
    for row in row_sample:
        for variant in ("A", "B", "C", "D"):
            members = model_export["model_objects"][variant]
            if variant in {"A", "B"}:
                axes = [(None, seed) for seed in FREEZE_CONFIG["model_fit"]["lightgbm_seeds"]]
            else:
                axes = [(r, s) for r in FREEZE_CONFIG["model_fit"]["reservoir_seeds"]
                        for s in FREEZE_CONFIG["model_fit"]["lightgbm_seeds"]]
            before = time.perf_counter_ns()
            for reservoir_seed, lightgbm_seed in axes:
                key = f"r{reservoir_seed}_l{lightgbm_seed}" if reservoir_seed is not None else f"l{lightgbm_seed}"
                if variant == "A":
                    values = np.asarray(arrays["base_x"][row], dtype=np.float32).reshape(1, -1)
                elif variant == "B":
                    values = np.concatenate((np.asarray(arrays["base_x"][row], dtype=np.float32), np.asarray(memory[row], dtype=np.float32))).reshape(1, -1)
                elif variant == "C":
                    values = np.concatenate((np.asarray(arrays["base_x"][row], dtype=np.float32), np.asarray(esn_features[int(reservoir_seed)][row], dtype=np.float32))).reshape(1, -1)
                else:
                    values = np.concatenate((np.asarray(arrays["base_x"][row], dtype=np.float32), np.asarray(memory[row], dtype=np.float32), np.asarray(esn_features[int(reservoir_seed)][row], dtype=np.float32))).reshape(1, -1)
                members[key].predict(values)
            prediction_latencies[variant].append((time.perf_counter_ns() - before) / 1000.0)
    latency = {
        variant: {
            "samples": len(values),
            "p50_us": float(np.quantile(values, 0.50)),
            "p95_us": float(np.quantile(values, 0.95)),
            "p99_us": float(np.quantile(values, 0.99)),
        }
        for variant, values in prediction_latencies.items()
    }
    batch_prediction_seconds = {
        variant: float(sum(member["validation_prediction_seconds"] for member in procedure["members"]))
        for variant, procedure in model_export["procedures"].items()
    }
    resource = expansion._resource_snapshot()
    runtime = {
        "active_seconds": float(time.monotonic() - RUN_STARTED),
        "model_fit_and_export_seconds": model_export["model_fit_and_export_seconds"],
        "D_fit_seconds": d_seconds,
        "fresh_bundle_representation_replay_seconds": bundle_replay_seconds,
        "full_representation_replay_seconds": full_replay_seconds,
        "source_C_full_generation_seconds": c_generation_seconds,
        "source_B_full_generation_seconds": memory_generation_seconds,
        "validation_batch_prediction_seconds_by_procedure": batch_prediction_seconds,
        "full_path_seconds_by_procedure": {
            "A_BASE70": batch_prediction_seconds["A"],
            "B_BASE70_plus_memory": memory_generation_seconds + batch_prediction_seconds["B"],
            "C_BASE70_plus_ESN": c_generation_seconds + batch_prediction_seconds["C"],
            "D_BASE70_plus_memory_plus_ESN": memory_generation_seconds + c_generation_seconds + batch_prediction_seconds["D"],
        },
        "profile_bookkeeping_seconds": float(time.monotonic() - verification_started),
        "peak_working_set_bytes": expansion._peak_working_set_bytes(),
        "resource_snapshot": resource,
        "resource_snapshot_before_fit": resource_start,
        "resource_snapshot_after_replay": resource,
        "latency": latency,
        "cold_start": cold_cost,
        "source_commit": _git_commit(),
        "focused_run_source_commit": identity["frozen_source_commit"],
        "runner_sha256": _sha256(Path(__file__)),
        "freeze_config_sha256": _sha256(FREEZE_CONFIG_PATH),
    }
    comparability = {
        "rows": int(len(rows)),
        "same_decision_ids_and_labels": True,
        "same_input_cutoff_and_baseline": True,
        "same_training_contract": True,
        "transform_fit_scope_verified": transform_audit["transform_fit_scope_verified"],
        "transform_fit_scope_audit": transform_audit,
        "historical_prediction_runner_hash_available": False,
        "maximum_saved_prediction_replay_error": max(
            item["saved_prediction_max_abs_error"]
            for variant in ("A", "B", "C")
            for item in model_export["procedures"][variant]["members"]
        ),
        "maximum_model_bundle_roundtrip_error": max(
            item["bundle_roundtrip_max_abs_error"]
            for procedure in model_export["procedures"].values()
            for item in procedure["members"]
        ),
        "decision_ids_sha256": results["decision_rows_sha256"],
        "target_sha256": results["target_sha256"],
        "fit_row_indices_sha256": results["fit_row_indices_sha256"],
    }
    source_ledger_sha256_after = _sha256(source_ledger_path)
    if source_ledger_sha256_after != source_ledger_sha256_before:
        raise RuntimeError("the existing focused-search runtime ledger changed during the development comparison")
    runtime["source_focused_runtime_ledger_sha256"] = source_ledger_sha256_before
    _write_report(results, model_export["procedures"], runtime, equivalence, comparability)
    _atomic_json(OUT / "runtime.json", runtime)
    _atomic_json(OUT / "stream_verification.json", equivalence)
    print(json.dumps({"status": "complete_development_only", "rows": len(rows), "results": results["procedures"],
                      "runtime": runtime, "output": str(OUT.relative_to(ROOT))}, indent=2, sort_keys=True), flush=True)
    _check_active_limit()


if __name__ == "__main__":
    main()
