"""Score the frozen BTC candidates on the two previously exposed periods, fail closed by default."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_candidate_holdout_gate_20261022.json"
BUNDLE_DIR = ROOT / "data/analysis/polymarket/BTC/btc_candidate_quality_freeze_20261022/bundles"
OUT_DIR = ROOT / "data/analysis/polymarket/BTC/btc_candidate_holdout_20261022"
PROTECTED_WINDOWS = {
    "confirmation": ("2025-07-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    "holdout": ("2026-07-01T00:00:00Z", "2026-10-07T00:00:00Z"),
}
EXPECTED_PROCEDURES = ("A", "B", "C", "D")
ALL_PAIRS = (("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("A", "D"))
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BLOCK_DAYS = 3
BOOTSTRAP_SEED = 20261023


class GateBlocked(RuntimeError):
    """Raised before any protected source or data pipeline is opened."""


def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _datetime_ns(values):
    import pandas as pd

    return pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise")).as_unit("ns").asi8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def save_period_checkpoint(run_dir: Path, period_id: str, run_identity: str, result: dict) -> Path:
    if period_id not in PROTECTED_WINDOWS:
        raise ValueError(f"unknown scoring period: {period_id}")
    path = run_dir / f"period_{period_id}.json"
    record = {"period_id": period_id, "run_identity_sha256": run_identity, "result": result}
    record["result_sha256"] = _canonical_hash(result)
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if (
            prior.get("period_id") != period_id
            or prior.get("run_identity_sha256") != run_identity
            or prior.get("result_sha256") != _canonical_hash(prior.get("result"))
        ):
            raise RuntimeError(f"existing {period_id} checkpoint identity or checksum differs")
        if prior["result"] != result:
            raise RuntimeError(f"refusing to overwrite completed {period_id} result")
        return path
    _atomic_json(path, record)
    return path


def load_period_checkpoint(run_dir: Path, period_id: str, run_identity: str) -> dict | None:
    path = run_dir / f"period_{period_id}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("period_id") != period_id or record.get("run_identity_sha256") != run_identity:
        raise RuntimeError(f"existing {period_id} checkpoint belongs to another frozen run")
    if record.get("result_sha256") != _canonical_hash(record.get("result")):
        raise RuntimeError(f"existing {period_id} checkpoint failed its result checksum")
    return record["result"]


def record_attempt(run_dir: Path, status: str, *, detail: str | None = None) -> None:
    path = run_dir / "attempts.json"
    history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"attempts": []}
    history["attempts"].append({
        "attempt": len(history["attempts"]) + 1,
        "status": str(status),
        "detail": detail,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    })
    _atomic_json(path, history)


def finalize_completed_run(
    run_dir: Path,
    run_identity: str,
    supplemental_source_data: dict,
    timings_seconds: dict,
    *,
    recovery_detail: str | None = None,
) -> None:
    period_results = {
        period: load_period_checkpoint(run_dir, period, run_identity)
        for period in PROTECTED_WINDOWS
    }
    if any(result is None for result in period_results.values()):
        raise RuntimeError("cannot finalize the run before both mandatory period checkpoints exist")
    _atomic_json(run_dir / "run_manifest.json", {
        "status": "completed_historical_robustness_scoring",
        "run_identity_sha256": run_identity,
        "completed_periods": list(PROTECTED_WINDOWS),
        "protected_rows_scored": True,
        "supplemental_source_data": supplemental_source_data,
        "timings_seconds": timings_seconds,
        "period_artifacts": {
            period: period_results[period]["prediction_artifacts"]
            for period in PROTECTED_WINDOWS
        },
        "not_fresh_or_independent_validation": True,
        "not_a_profitability_test": True,
        **({"completion_recovery": recovery_detail} if recovery_detail else {}),
    })
    record_attempt(run_dir, "completed", detail=recovery_detail)


def validate_scope(config: dict) -> None:
    """Require explicit approval for both exact windows; no broad flag can widen scope."""
    if config.get("experiment_id") != "btc_candidate_quality_holdout_20261022":
        raise GateBlocked("BLOCKED: unexpected experiment identity; no protected files were opened.")
    periods = config.get("periods")
    if not isinstance(periods, list) or [item.get("id") for item in periods] != list(PROTECTED_WINDOWS):
        raise GateBlocked("BLOCKED: the two frozen period IDs or their order changed; no protected files were opened.")
    for item in periods:
        expected = PROTECTED_WINDOWS[item["id"]]
        if (item.get("market_start_inclusive_utc"), item.get("market_start_exclusive_utc")) != expected:
            raise GateBlocked(f"BLOCKED: {item['id']} boundaries differ from the frozen protocol; no protected files were opened.")
        if item.get("allow_read") is not True:
            raise GateBlocked(f"BLOCKED: explicit read approval is missing for {item['id']}; no protected files or data pipelines were opened.")
    if config.get("allow_protected_data") is not True:
        raise GateBlocked("BLOCKED: protected BTC data access is disabled. Both exact periods must be explicitly approved; no protected files or data pipelines were opened.")
    if config.get("approved_period_ids") != list(PROTECTED_WINDOWS):
        raise GateBlocked("BLOCKED: the approved period allowlist must contain exactly confirmation and holdout; no protected files were opened.")


def common_observation_mask(
    opened_ns,
    decision_ns,
    label_available_ns,
    decision_flag,
    target,
    base_x,
    memory_x,
    esn_by_seed,
    *,
    start_ns: int,
    end_ns: int,
    minute_ns: int = 60_000_000_000,
) -> tuple[object, dict]:
    """Build one strict common row mask using market-start and label-release semantics."""
    import numpy as np

    opened = np.asarray(opened_ns, dtype=np.int64)
    decision = np.asarray(decision_ns, dtype=np.int64)
    available = np.asarray(label_available_ns, dtype=np.int64)
    decisions = np.asarray(decision_flag, dtype=bool)
    y = np.asarray(target)
    base = np.asarray(base_x)
    memory = np.asarray(memory_x)
    esn = {int(seed): np.asarray(values) for seed, values in esn_by_seed.items()}
    n = len(opened)
    if any(len(values) != n for values in (decision, available, decisions, y, base, memory, *esn.values())):
        raise ValueError("scoring inputs have different row counts")
    if base.ndim != 2 or base.shape[1] != 70 or memory.ndim != 2 or memory.shape[1] != 8:
        raise ValueError("BASE70 and memory PCA8 matrices must have widths 70 and 8")
    if set(esn) != {20261023, 20261024, 20261025} or any(x.ndim != 2 or x.shape[1] != 8 for x in esn.values()):
        raise ValueError("all three frozen ESN PCA8 matrices are required")
    if not np.array_equal(decision, opened + minute_ns) or not np.array_equal(available, opened + 7 * minute_ns):
        raise ValueError("T-60 decision or Opened+7m label-availability contract changed")
    market_start = opened + 2 * minute_ns
    in_scope = (market_start >= int(start_ns)) & (market_start < int(end_ns))
    binary = np.isin(y, (0, 1))
    label_released = available < int(end_ns)
    valid_base = np.isfinite(base).all(axis=1)
    valid_memory = np.isfinite(memory).all(axis=1)
    valid_esn = np.logical_and.reduce([np.isfinite(values).all(axis=1) for values in esn.values()])
    mask = in_scope & decisions & binary & label_released & valid_base & valid_memory & valid_esn
    return mask, {
        "source_rows": int(n),
        "market_start_in_scope": int(in_scope.sum()),
        "excluded_market_start_outside_period": int((~in_scope).sum()),
        "excluded_not_preopen_decision": int((in_scope & ~decisions).sum()),
        "excluded_missing_or_nonbinary_target": int((in_scope & decisions & ~binary).sum()),
        "excluded_label_not_released_before_period_end": int((in_scope & decisions & binary & ~label_released).sum()),
        "excluded_invalid_BASE70": int((in_scope & decisions & binary & label_released & ~valid_base).sum()),
        "excluded_invalid_memory": int((in_scope & decisions & binary & label_released & valid_base & ~valid_memory).sum()),
        "excluded_invalid_ESN_after_valid_memory": int((in_scope & decisions & binary & label_released & valid_base & valid_memory & ~valid_esn).sum()),
        "excluded_invalid_memory_or_ESN": int((in_scope & decisions & binary & label_released & valid_base & ~(valid_memory & valid_esn)).sum()),
        "expected_decisions": int((in_scope & decisions).sum()),
        "decisions_with_complete_features": int((in_scope & decisions & valid_base & valid_memory & valid_esn).sum()),
        "common_scoring_rows": int(mask.sum()),
    }


def predict_members(base_x, memory_x, esn_by_seed, boosters, *, lightgbm_seeds, reservoir_seeds):
    """Run the frozen models on causal representations; no fitting or model selection occurs here."""
    import numpy as np

    base = np.asarray(base_x, dtype=np.float32)
    memory = np.asarray(memory_x, dtype=np.float32)
    esn = {int(seed): np.asarray(values, dtype=np.float32) for seed, values in esn_by_seed.items()}
    expected_keys = set()
    members = {name: {} for name in EXPECTED_PROCEDURES}
    for lgb_seed in map(int, lightgbm_seeds):
        expected_keys.add(("A", None, lgb_seed))
        expected_keys.add(("B", None, lgb_seed))
        x_a = base
        x_b = np.column_stack((base, memory))
        members["A"][str(lgb_seed)] = np.asarray(boosters[("A", None, lgb_seed)].predict(x_a), dtype=np.float64)
        members["B"][str(lgb_seed)] = np.asarray(boosters[("B", None, lgb_seed)].predict(x_b), dtype=np.float64)
        for reservoir_seed in map(int, reservoir_seeds):
            expected_keys.update((("C", reservoir_seed, lgb_seed), ("D", reservoir_seed, lgb_seed)))
            x_c = np.column_stack((base, esn[reservoir_seed]))
            x_d = np.column_stack((base, memory, esn[reservoir_seed]))
            members["C"][f"{reservoir_seed}:{lgb_seed}"] = np.asarray(
                boosters[("C", reservoir_seed, lgb_seed)].predict(x_c), dtype=np.float64,
            )
            members["D"][f"{reservoir_seed}:{lgb_seed}"] = np.asarray(
                boosters[("D", reservoir_seed, lgb_seed)].predict(x_d), dtype=np.float64,
            )
    if set(boosters) != expected_keys:
        raise ValueError("loaded booster slots differ from the frozen seed grid")
    return members


def ensemble_predictions(members, lightgbm_seeds, reservoir_seeds):
    import numpy as np

    return {
        "A": np.mean([members["A"][str(seed)] for seed in lightgbm_seeds], axis=0),
        "B": np.mean([members["B"][str(seed)] for seed in lightgbm_seeds], axis=0),
        "C": np.mean([members["C"][f"{r}:{s}"] for r in reservoir_seeds for s in lightgbm_seeds], axis=0),
        "D": np.mean([members["D"][f"{r}:{s}"] for r in reservoir_seeds for s in lightgbm_seeds], axis=0),
    }


def score_metrics(target, probabilities) -> dict:
    import numpy as np
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    y = np.asarray(target, dtype=np.int8)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    if len(y) == 0 or not np.isin(y, (0, 1)).all():
        raise ValueError("metrics require non-empty binary targets")
    return {
        "rows": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    }


def _member_feature_order(procedure: str, base_order: list[str], *, reservoir_seed: int | None = None, esn_names: list[str] | None = None) -> list[str]:
    """Reconstruct the frozen per-member feature order consumed by LightGBM."""
    if procedure == "A":
        if reservoir_seed is not None:
            raise ValueError("A members cannot have a reservoir seed")
        return list(base_order)
    memory_names = [f"simple_memory_PCA8_pc{index:02d}" for index in range(8)]
    if procedure == "B":
        if reservoir_seed is not None:
            raise ValueError("B members cannot have a reservoir seed")
        return list(base_order) + memory_names
    if procedure not in {"C", "D"} or reservoir_seed is None:
        raise ValueError(f"invalid frozen member slot for {procedure}")
    if not isinstance(esn_names, list) or len(esn_names) != 8 or len(set(esn_names)) != 8:
        raise ValueError(f"ESN PCA8 feature names are invalid for reservoir {reservoir_seed}")
    order = list(base_order)
    if procedure == "D":
        order += memory_names
    return order + esn_names


def _bundle_identity() -> dict:
    """Verify model, transform, feature, training, and ensemble identities before data access."""
    manifests = {}
    paths = [BUNDLE_DIR / f"{name}_manifest.json" for name in EXPECTED_PROCEDURES]
    for path in paths:
        manifests[path.stem[0]] = json.loads(path.read_text(encoding="utf-8"))
    base_order = manifests["A"].get("base_feature_order")
    if not isinstance(base_order, list) or len(base_order) != 70:
        raise RuntimeError("frozen A manifest does not provide the exact BASE70 order")
    model_records = []
    shared_training = manifests["A"].get("training")
    for name, manifest in manifests.items():
        if manifest.get("base_feature_order") != base_order:
            raise RuntimeError(f"{name} BASE70 order differs from the frozen A order")
        training = manifest.get("training", {})
        if training != shared_training:
            raise RuntimeError(f"{name} training scope, seeds, weights, or LightGBM parameters differ from A")
        if (training.get("fit_rows"), training.get("fit_row_indices_sha256"), training.get("boosting_rounds"), training.get("early_stopping")) != (
            2_088_350, "edb48c5727350a96e02c4d94a97efba60ea0ed91cc319b2550479316a38e25ef", 31, False,
        ):
            raise RuntimeError(f"{name} fit provenance or fixed 31-round model differs from the freeze")
        expected_features = {"A": 70, "B": 78, "C": 78, "D": 86}[name]
        if manifest.get("feature_count") != expected_features:
            raise RuntimeError(f"{name} feature width differs from the frozen procedure")
        weights = manifest.get("ensemble_weights")
        expected_weight = "uniform 1/3" if name in {"A", "B"} else "uniform 1/9"
        if weights != expected_weight:
            raise RuntimeError(f"{name} ensemble is not the frozen uniform ensemble")
        for member in manifest.get("members", []):
            model_path = ROOT / Path(member["model_path"])
            observed = _sha256(model_path)
            if observed != member.get("model_sha256"):
                raise RuntimeError(f"frozen {name} model hash mismatch: {model_path.name}")
            model_records.append({
                "procedure": name,
                "lightgbm_seed": int(member["lightgbm_seed"]),
                "reservoir_seed": None if member.get("reservoir_seed") is None else int(member["reservoir_seed"]),
                "path": model_path.relative_to(ROOT).as_posix(),
                "sha256": observed,
                "feature_count": int(member["feature_count"]),
                "feature_order_sha256": member["feature_order_sha256"],
            })
    memory_transform_path = BUNDLE_DIR / "transforms/simple_memory_projection.json"
    esn_json_paths = [BUNDLE_DIR / f"transforms/esn_seed_{seed}.json" for seed in (20261023, 20261024, 20261025)]
    esn_matrix_paths = [BUNDLE_DIR / f"transforms/esn_matrices_seed_{seed}.npz" for seed in (20261023, 20261024, 20261025)]
    transform_paths = [memory_transform_path, *esn_json_paths, *esn_matrix_paths]
    transform_hashes = {path.relative_to(ROOT).as_posix(): _sha256(path) for path in transform_paths}
    for path in esn_json_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = payload.get("identity", {})
        if identity.get("confirmation_read") is not False or identity.get("holdout_rows_materialized") != 0:
            raise RuntimeError(f"ESN transform provenance is not development-only: {path.name}")
    esn_feature_names = {}
    for path in esn_json_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        if seed in esn_feature_names:
            raise RuntimeError(f"duplicate frozen ESN transform seed: {seed}")
        esn_feature_names[seed] = payload.get("feature_names")
    memory_payload = json.loads(memory_transform_path.read_text(encoding="utf-8"))
    if memory_payload.get("identity", {}).get("scope_end_exclusive_utc") != "2025-07-01T00:00:00Z":
        raise RuntimeError("memory transform scope differs from the frozen development-only fit")
    for name, manifest in manifests.items():
        for member in manifest.get("members", []):
            reservoir_seed = member.get("reservoir_seed")
            expected_order = _member_feature_order(
                name,
                base_order,
                reservoir_seed=None if reservoir_seed is None else int(reservoir_seed),
                esn_names=None if reservoir_seed is None else esn_feature_names.get(int(reservoir_seed)),
            )
            expected_hash = hashlib.sha256(
                json.dumps(expected_order, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if member.get("feature_count") != len(expected_order) or member.get("feature_order_sha256") != expected_hash:
                raise RuntimeError(f"{name} frozen member feature order differs from the candidate definition")
    member_feature_orders = []
    for name, manifest in manifests.items():
        for member in manifest.get("members", []):
            reservoir_seed = member.get("reservoir_seed")
            member_feature_orders.append({
                "procedure": name,
                "lightgbm_seed": int(member["lightgbm_seed"]),
                "reservoir_seed": None if reservoir_seed is None else int(reservoir_seed),
                "feature_order": _member_feature_order(
                    name,
                    base_order,
                    reservoir_seed=None if reservoir_seed is None else int(reservoir_seed),
                    esn_names=None if reservoir_seed is None else esn_feature_names.get(int(reservoir_seed)),
                ),
            })
    return {
        "base_feature_order_sha256": _canonical_hash(base_order),
        "base_feature_order": base_order,
        "procedure_manifest_sha256": {name: _sha256(BUNDLE_DIR / f"{name}_manifest.json") for name in EXPECTED_PROCEDURES},
        "transform_sha256": transform_hashes,
        "model_sha256": model_records,
        "member_feature_orders": member_feature_orders,
        "member_count": len(model_records),
        "training_identity": manifests["A"]["training"],
        "lightgbm_seeds": sorted(int(key) for key in manifests["A"]["training"]["lgbm_parameters_by_seed"]),
        "reservoir_seeds": sorted({int(item["reservoir_seed"]) for item in manifests["C"]["members"]}),
        "uniform_ensemble_weights": {name: manifests[name]["ensemble_weights"] for name in EXPECTED_PROCEDURES},
        "transform_parameters": {
            "simple_memory": memory_payload.get("identity"),
            "esn_by_seed": {
                int(payload["seed"]): payload.get("identity")
                for path in esn_json_paths
                for payload in [json.loads(path.read_text(encoding="utf-8"))]
            },
        },
    }


def _write_preread_manifest(config: dict, identity: dict) -> Path:
    run_id = _canonical_hash(identity)
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "pre_read_manifest.json"
    record = {
        "experiment_id": config["experiment_id"],
        "status": "pre_read_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity_sha256": run_id,
        "config_sha256": _sha256(CONFIG_PATH),
        "protocol_sha256": _canonical_hash(config["protocol"]),
        "code_commit_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "runner_sha256": _sha256(Path(__file__)),
        "user_authorization": config["authorization_record"],
        "frozen_bundle_identity": identity["bundle"],
        "scope": [{
            "id": item["id"],
            "market_start_inclusive_utc": item["market_start_inclusive_utc"],
            "market_start_exclusive_utc": item["market_start_exclusive_utc"],
            "label_available_strictly_before_utc": item["market_start_exclusive_utc"],
            "prior_exposure": item["prior_exposure"],
        } for item in config["periods"]],
        "source_paths": config["inputs"],
        "pinned_source_data_identity": identity.get("source_data"),
        "pinned_development_cache_identity": identity.get("development_cache"),
        "pinned_source_code_sha256": identity.get("source_code_sha256"),
        "no_refit_no_early_stopping": True,
        "all_periods_mandatory": True,
    }
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior.get("run_identity_sha256") != run_id:
            raise RuntimeError("existing pre-read manifest identity differs")
    else:
        _atomic_json(path, record)
    return run_dir


def _load_frozen_boosters(bundle_identity: dict):
    import lightgbm as lgb

    boosters = {}
    for name in EXPECTED_PROCEDURES:
        manifest = json.loads((BUNDLE_DIR / f"{name}_manifest.json").read_text(encoding="utf-8"))
        for member in manifest["members"]:
            reservoir = member.get("reservoir_seed")
            seed = int(member["lightgbm_seed"])
            path = ROOT / Path(member["model_path"])
            if _sha256(path) != member["model_sha256"]:
                raise RuntimeError(f"model changed after pre-read hash: {path}")
            booster = lgb.Booster(model_file=str(path))
            if int(booster.num_trees()) != 31 or booster.num_feature() != int(member["feature_count"]):
                raise RuntimeError(f"frozen {name} model shape or round count mismatch")
            boosters[(name, None if reservoir is None else int(reservoir), seed)] = booster
    return boosters


def _manifest_output_hash(manifest_path: Path, output_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = output_path.relative_to(ROOT).as_posix()
    for stage in manifest.get("stages", {}).values():
        for output in stage.get("outputs", []):
            if Path(output.get("path", "")).as_posix() == relative:
                return str(output["sha256"])
    raise RuntimeError(f"source manifest does not pin expected output: {relative}")


def _source_artifact_identity() -> dict:
    configured = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["inputs"]
    dataset_path = ROOT / Path(configured["baseline_feature_dataset"])
    feature_run_manifest = ROOT / "data/analysis/polymarket/BTC/btc_preopen_full_development_fit_20261016/pipeline/794a69e369a3f82d/run_manifest.json"
    flow_path = ROOT / Path(configured["taker_flow_features"])
    flow_manifest_path = ROOT / "data/analysis/polymarket/BTC/btc_preopen_taker_flow_20261018/run_manifest.json"
    effective_config_path = ROOT / "data/analysis/polymarket/BTC/btc_preopen_full_development_fit_20261016/pipeline/794a69e369a3f82d/effective_config.json"
    effective_config = json.loads(effective_config_path.read_text(encoding="utf-8"))
    raw_record = effective_config["input_data"]
    raw_path = ROOT / Path(configured["raw_um_ohlcv_source"])
    if Path(raw_record["path"]).name != raw_path.name:
        raise RuntimeError("configured raw USD-M OHLCV source differs from the frozen feature-run input")
    return {
        "baseline_feature_dataset": {
            "path": dataset_path.relative_to(ROOT).as_posix(),
            "pinned_sha256": _manifest_output_hash(feature_run_manifest, dataset_path),
            "source_manifest_path": feature_run_manifest.relative_to(ROOT).as_posix(),
            "source_manifest_sha256": _sha256(feature_run_manifest),
        },
        "taker_flow_features": {
            "path": flow_path.relative_to(ROOT).as_posix(),
            "pinned_sha256": _manifest_output_hash(flow_manifest_path, flow_path),
            "source_manifest_path": flow_manifest_path.relative_to(ROOT).as_posix(),
            "source_manifest_sha256": _sha256(flow_manifest_path),
        },
        "raw_um_ohlcv_source": {
            "path": raw_path.relative_to(ROOT).as_posix(),
            "pinned_sha256": raw_record["sha256"],
            "size_bytes": int(raw_record["size"]),
            "effective_config_path": effective_config_path.relative_to(ROOT).as_posix(),
            "effective_config_sha256": _sha256(effective_config_path),
        },
    }


def _development_cache_identity(config: dict) -> dict:
    """Pin the already-permitted development stream and BASE70 arrays before test reads."""
    prepared_dir = ROOT / Path(config["inputs"]["development_raw_prefix"])
    manifest_path = prepared_dir / "prepared_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("holdout_rows_materialized") != 0 or manifest.get("materialization_cutoff_utc") != "2025-07-01T00:00:00+00:00":
        raise RuntimeError("development cache manifest is outside the frozen development-only scope")
    filenames = (
        "base_decision_at.npy", "base_opened.npy", "base_to_raw.npy", "base_x.npy",
        "decision.npy", "label_available.npy", "raw_opened.npy", "raw_valid.npy",
        "raw_x.npy", "target.npy",
    )
    arrays = {}
    for name in filenames:
        path = prepared_dir / name
        arrays[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "scope_end_exclusive_utc": manifest["materialization_cutoff_utc"],
        "holdout_rows_materialized": manifest["holdout_rows_materialized"],
        "arrays": arrays,
    }


def _verify_source_artifacts(source_identity: dict) -> dict:
    """Verify pinned protected-source hashes only after the frozen pre-read manifest exists."""
    observed = {}
    for name, record in source_identity.items():
        path = ROOT / record["path"]
        if "size_bytes" in record and int(record["size_bytes"]) != path.stat().st_size:
            raise RuntimeError(f"pinned source data size mismatch for {name}: {record['path']}")
        digest = _sha256(path)
        if digest != record["pinned_sha256"]:
            raise RuntimeError(f"pinned source data hash mismatch for {name}: {record['path']}")
        source_manifest_path = record.get("source_manifest_path") or record.get("effective_config_path")
        source_manifest_sha256 = record.get("source_manifest_sha256") or record.get("effective_config_sha256")
        if source_manifest_path and _sha256(ROOT / source_manifest_path) != source_manifest_sha256:
            raise RuntimeError(f"pinned source manifest hash mismatch for {name}: {source_manifest_path}")
        observed[name] = {"path": record["path"], "sha256": digest, "verified": True}
    return observed


def _flow_coverage_error(flow_paths: list[Path], required_opened_inclusive: str, required_opened_exclusive: str) -> str | None:
    """Inspect only Parquet footer statistics before opening any protected row values."""
    import pyarrow.parquet as pq
    import pandas as pd

    required = ("Opened", "taker_Z_1m", "taker_V_1m", "taker_Z_5m", "taker_V_5m", "taker_Z_15m", "taker_V_15m")
    minima, maxima = [], []
    for flow_path in flow_paths:
        if not flow_path.is_file():
            continue
        parquet = pq.ParquetFile(flow_path)
        names = parquet.schema_arrow.names
        missing = sorted(set(required) - set(names))
        if missing:
            return f"taker-flow input {flow_path.relative_to(ROOT).as_posix()} is missing BASE70 columns: {missing}"
        opened_column = names.index("Opened")
        for index in range(parquet.metadata.num_row_groups):
            stats = parquet.metadata.row_group(index).column(opened_column).statistics
            if stats is not None and stats.has_min_max:
                minima.append(pd.Timestamp(stats.min))
                maxima.append(pd.Timestamp(stats.max))
    if not minima or not maxima:
        return "no taker-flow source has timestamp statistics to verify coverage"
    earliest, latest = min(minima), max(maxima)
    earliest = earliest.tz_localize("UTC") if earliest.tzinfo is None else earliest.tz_convert("UTC")
    latest = latest.tz_localize("UTC") if latest.tzinfo is None else latest.tz_convert("UTC")
    required_first = pd.Timestamp(required_opened_inclusive, tz="UTC")
    required_last = pd.Timestamp(required_opened_exclusive, tz="UTC") - pd.Timedelta(minutes=1)
    if earliest > required_first or latest < required_last:
        return (
            f"taker-flow input coverage is {earliest.isoformat()} through {latest.isoformat()}; "
            f"required coverage is {required_first.isoformat()} through {required_last.isoformat()}. "
            "The missing exact source is Binance USD-M BTCUSDT 1m quote_volume and taker_buy_quote_volume; "
            "no protected row values were opened."
        )
    return None


def _source_completion_identity(config: dict) -> dict | None:
    """Verify the separately acquired, checksum-pinned Binance supplement after the pre-read freeze."""
    import pandas as pd

    manifest_path = ROOT / config["inputs"]["taker_flow_supplement_manifest"]
    feature_path = ROOT / config["inputs"]["taker_flow_feature_supplement"]
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or not feature_path.is_file():
        return None
    output = manifest.get("flow_feature_output", {})
    if output.get("path") != feature_path.relative_to(ROOT).as_posix():
        raise RuntimeError("supplemental flow-feature path differs from its manifest")
    if int(output.get("size_bytes", -1)) != feature_path.stat().st_size or output.get("sha256") != _sha256(feature_path):
        raise RuntimeError("supplemental flow-feature checksum or size differs from its manifest")
    if manifest.get("required_binance_volume_opened_inclusive_utc") != config["inputs"]["required_binance_volume_opened_inclusive_utc"]:
        raise RuntimeError("supplemental Binance range has a different inclusive start")
    if manifest.get("required_binance_volume_opened_exclusive_utc") != config["inputs"]["required_binance_volume_opened_exclusive_utc"]:
        raise RuntimeError("supplemental Binance range has a different exclusive end")
    frozen_identity = manifest.get("frozen_identity", {})
    if frozen_identity.get("config_sha256") != _sha256(CONFIG_PATH):
        raise RuntimeError("supplemental Binance source was prepared under a different frozen config")
    if frozen_identity.get("protocol_sha256") != _canonical_hash(config["protocol"]):
        raise RuntimeError("supplemental Binance source was prepared under a different frozen protocol")
    feature_config_path = ROOT / "configs/research/btc_preopen_taker_flow_20261018.json"
    feature_code_path = ROOT / "features/btc_taker_flow.py"
    if (
        frozen_identity.get("flow_feature_configuration_sha256") != _sha256(feature_config_path)
        or manifest.get("feature_configuration_sha256") != _sha256(feature_config_path)
    ):
        raise RuntimeError("supplemental flow-feature configuration changed after data preparation")
    if (
        frozen_identity.get("flow_feature_code_sha256") != _sha256(feature_code_path)
        or manifest.get("feature_code_sha256") != _sha256(feature_code_path)
    ):
        raise RuntimeError("supplemental flow feature code changed after data preparation")
    expected_days = pd.date_range(
        pd.Timestamp(config["inputs"]["required_binance_volume_opened_inclusive_utc"]).floor("D"),
        (pd.Timestamp(config["inputs"]["required_binance_volume_opened_exclusive_utc"]) - pd.Timedelta(nanoseconds=1)).floor("D"),
        freq="D",
    ).strftime("%Y-%m-%d").tolist()
    archives = manifest.get("downloaded_archives", [])
    if [item.get("day") for item in archives] != expected_days:
        raise RuntimeError("supplemental Binance source does not contain every exact requested UTC day")
    expected_volume_rows = len(pd.date_range(
        pd.Timestamp(config["inputs"]["required_binance_volume_opened_inclusive_utc"]),
        pd.Timestamp(config["inputs"]["required_binance_volume_opened_exclusive_utc"]),
        freq="min", inclusive="left",
    ))
    coverage = manifest.get("coverage", {})
    if (
        int(coverage.get("expected_volume_minutes", -1)) != expected_volume_rows
        or int(coverage.get("observed_volume_minutes", -1)) != expected_volume_rows
        or int(coverage.get("missing_volume_minutes", -1)) != 0
        or int(coverage.get("invalid_ohlc_rows", -1)) != 0
        or int(coverage.get("invalid_quote_or_taker_volume_rows", -1)) != 0
        or int(output.get("rows", -1)) != expected_volume_rows
    ):
        raise RuntimeError("supplemental source manifest does not certify complete valid Binance minute coverage")
    if any(
        int(item.get("mismatch_rows_at_1e-7_absolute_tolerance", -1)) != 0
        for item in coverage.get("overlap_ohlc_comparison", {}).values()
    ):
        raise RuntimeError("supplemental source manifest records an OHLC mismatch on overlapping rows")
    for item in archives:
        raw_path = ROOT / item["raw_payload_path"]
        if not raw_path.is_file():
            return None
        if raw_path.stat().st_size != int(item.get("payload_size_bytes", -1)) or _sha256(raw_path) != item.get("payload_sha256"):
            raise RuntimeError(f"preserved raw Binance payload checksum failed: {item.get('day')}")
    warmup = manifest.get("warmup_source", {})
    warmup_path = ROOT / config["inputs"]["taker_flow_warmup_archive"]
    if warmup.get("path") != warmup_path.relative_to(ROOT).as_posix() or _sha256(warmup_path) != warmup.get("sha256"):
        raise RuntimeError("local causal warm-up archive differs from its source-completion manifest")
    return {
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "feature_path": feature_path.relative_to(ROOT).as_posix(),
        "feature_sha256": output["sha256"],
        "feature_size_bytes": output["size_bytes"],
        "downloaded_archive_count": len(manifest.get("downloaded_archives", [])),
        "coverage": manifest.get("coverage"),
        "source_manifest": manifest,
    }


def _same_utc_timestamps(left, right) -> bool:
    import numpy as np
    import pandas as pd

    def as_nanoseconds(values):
        timestamps = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
        return timestamps.astype("datetime64[ns, UTC]").asi8

    return bool(np.array_equal(as_nanoseconds(left), as_nanoseconds(right)))


def _complete_taker_flow_source(config: dict, run_dir: Path, run_identity: str) -> dict:
    """Fetch the exact missing USD-M kline range and build the frozen flow-feature supplement."""
    import json
    import time
    import zipfile
    import numpy as np
    import pandas as pd
    from extend_btc_preopen_cutoff_data import PAGE_ROWS, UM_ENDPOINT, _fetch
    from features.btc_taker_flow import compute_taker_flow_features, feature_names

    inputs = config["inputs"]
    manifest_path = ROOT / inputs["taker_flow_supplement_manifest"]
    feature_path = ROOT / inputs["taker_flow_feature_supplement"]
    raw_dir = manifest_path.parent / "raw_by_day"
    download_start = pd.Timestamp(inputs["required_binance_volume_opened_inclusive_utc"])
    download_end = pd.Timestamp(inputs["required_binance_volume_opened_exclusive_utc"])
    warmup_start = pd.Timestamp(inputs["required_taker_flow_warmup_opened_inclusive_utc"])
    if download_start.tzinfo is None or download_end.tzinfo is None or warmup_start.tzinfo is None:
        raise RuntimeError("supplemental input ranges must have explicit UTC offsets")
    download_start, download_end, warmup_start = (
        value.tz_convert("UTC") for value in (download_start, download_end, warmup_start)
    )
    if download_end <= download_start or warmup_start >= download_start:
        raise RuntimeError("supplemental source range or warm-up boundary is invalid")

    frozen_identity = {
        "config_sha256": _sha256(CONFIG_PATH),
        "protocol_sha256": _canonical_hash(config["protocol"]),
        "required_binance_volume_opened_inclusive_utc": inputs["required_binance_volume_opened_inclusive_utc"],
        "required_binance_volume_opened_exclusive_utc": inputs["required_binance_volume_opened_exclusive_utc"],
        "required_flow_feature_opened_inclusive_utc": inputs["required_flow_feature_opened_inclusive_utc"],
        "required_flow_feature_opened_exclusive_utc": inputs["required_flow_feature_opened_exclusive_utc"],
        "required_taker_flow_warmup_opened_inclusive_utc": inputs["required_taker_flow_warmup_opened_inclusive_utc"],
        "flow_feature_configuration_sha256": _sha256(ROOT / "configs/research/btc_preopen_taker_flow_20261018.json"),
        "flow_feature_code_sha256": _sha256(ROOT / "features/btc_taker_flow.py"),
    }
    pre_read = json.loads((run_dir / "pre_read_manifest.json").read_text(encoding="utf-8"))
    this_execution = {
        "run_identity_sha256": run_identity,
        "pre_read_manifest_sha256": _sha256(run_dir / "pre_read_manifest.json"),
        "code_commit_sha": pre_read["code_commit_sha"],
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("frozen_identity") != frozen_identity:
            raise RuntimeError("existing source-completion checkpoint belongs to another frozen run")
        if manifest.get("status") == "completed":
            if feature_path.is_file():
                verified = _source_completion_identity(config)
                if verified is not None:
                    return verified
            manifest["recovery_reason"] = "completed supplement artifact or raw payload was missing; resuming from preserved checkpoints"
        executions = manifest.setdefault("pre_read_executions", [])
        if not any(item.get("run_identity_sha256") == run_identity for item in executions):
            executions.append(this_execution)
    else:
        manifest = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "status": "planned",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_identity": frozen_identity,
            "pre_read_executions": [this_execution],
            "source": {
                "market_type": "Binance USD-M Futures",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "endpoint": UM_ENDPOINT,
                "payload_fields": [
                    "open_time_ms", "open", "high", "low", "close", "base_volume",
                    "close_time_ms", "quote_volume", "trade_count", "taker_buy_base_volume",
                    "taker_buy_quote_volume", "ignore",
                ],
                "quote_volume_unit": "USDT quote asset amount per kline",
                "taker_buy_quote_volume_unit": "USDT quote asset amount per kline",
                "download_method": "existing exact-range Binance REST paginator from extend_btc_preopen_cutoff_data.py",
                "page_limit": PAGE_ROWS,
                "raw_payload_format": "canonical JSON arrays containing the unchanged Binance kline fields",
            },
            "downloaded_archives": [],
        }
    manifest["status"] = "downloading"
    manifest.setdefault("downloaded_archives", [])
    _atomic_json(manifest_path, manifest)

    by_day = {item["day"]: item for item in manifest["downloaded_archives"]}
    expected_index = pd.date_range(download_start, download_end, freq="min", inclusive="left")
    days = pd.date_range(download_start.floor("D"), (download_end - pd.Timedelta(nanoseconds=1)).floor("D"), freq="D")
    import requests

    download_elapsed = 0.0
    for day_start in days:
        day = day_start.strftime("%Y-%m-%d")
        partition_start = max(day_start, download_start)
        partition_end = min(day_start + pd.Timedelta(days=1), download_end)
        stop = partition_end - pd.Timedelta(milliseconds=1)
        expected_rows = len(pd.date_range(partition_start, partition_end, freq="min", inclusive="left"))
        raw_path = raw_dir / f"BTCUSDT-1m-{day}.json"
        prior = by_day.get(day)
        recover_missing_payload = False
        if prior and prior.get("status") == "completed":
            if raw_path.is_file() and _sha256(raw_path) != prior.get("payload_sha256"):
                raise RuntimeError(f"downloaded Binance payload changed after checkpoint: {day}")
            if raw_path.is_file():
                continue
            recover_missing_payload = True
        elif raw_path.exists():
            raise RuntimeError(f"unregistered raw Binance payload exists; refusing to overwrite it: {raw_path}")
        params = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "startTime": int(partition_start.timestamp() * 1000),
            "endTime": int(stop.timestamp() * 1000),
            "limit": PAGE_ROWS,
        }
        request_url = requests.Request("GET", UM_ENDPOINT, params=params).prepare().url
        started_utc = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        last_error = None
        rows = None
        for attempt in range(1, 4):
            try:
                rows = _fetch(UM_ENDPOINT, "symbol", "BTCUSDT", partition_start, stop)
                break
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(1.5 * attempt)
        if rows is None:
            manifest["status"] = "failed_download"
            manifest["failure"] = {"day": day, "error": f"{type(last_error).__name__}: {last_error}"}
            _atomic_json(manifest_path, manifest)
            raise RuntimeError(f"Binance USD-M REST download failed for {day}") from last_error
        if len(rows) != expected_rows:
            raise RuntimeError(f"Binance returned {len(rows)} rows for {day}; expected {expected_rows}")
        payload = json.dumps(rows, separators=(",", ":"))
        payload_text = payload + "\n"
        payload_sha256 = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
        if recover_missing_payload and payload_sha256 != prior.get("payload_sha256"):
            raise RuntimeError(f"re-fetched Binance payload differs from its preserved checkpoint: {day}")
        _atomic_text(raw_path, payload_text)
        completed_utc = datetime.now(timezone.utc).isoformat()
        item = {
            "day": day,
            "status": "completed",
            "endpoint": UM_ENDPOINT,
            "request_url": request_url,
            "market_type": "USD-M Futures",
            "symbol": "BTCUSDT",
            "interval": "1m",
            "opened_inclusive_utc": partition_start.isoformat().replace("+00:00", "Z"),
            "opened_exclusive_utc": partition_end.isoformat().replace("+00:00", "Z"),
            "download_started_utc": started_utc,
            "download_completed_utc": completed_utc,
            "elapsed_seconds": time.perf_counter() - started,
            "expected_rows": expected_rows,
            "observed_rows": len(rows),
            "raw_payload_path": raw_path.relative_to(ROOT).as_posix(),
            "payload_sha256": _sha256(raw_path),
            "payload_size_bytes": raw_path.stat().st_size,
            "checksum_kind": "SHA-256 of preserved canonical raw API row payload; Binance REST does not publish a sidecar checksum",
            "timestamp_grid_validation": "passed by the existing exact-range paginator",
        }
        if recover_missing_payload:
            item["recovered_missing_raw_payload_utc"] = completed_utc
            item["prior_download_completed_utc"] = prior.get("download_completed_utc")
        by_day[day] = item
        manifest["downloaded_archives"] = [by_day[key] for key in sorted(by_day)]
        download_elapsed += item["elapsed_seconds"]
        manifest["download_elapsed_seconds_this_attempt"] = download_elapsed
        _atomic_json(manifest_path, manifest)
        print(f"Downloaded exact Binance USD-M BTCUSDT 1m partition {day}: {expected_rows} rows", flush=True)

    if set(by_day) != {day.strftime("%Y-%m-%d") for day in days}:
        raise RuntimeError("the downloaded Binance day partitions do not cover the exact approved date range")
    manifest["downloaded_archives"] = [by_day[key] for key in sorted(by_day)]
    manifest["status"] = "validating_source_and_features"
    _atomic_json(manifest_path, manifest)
    validation_started = time.perf_counter()

    columns = [
        "open_time", "open", "high", "low", "close", "base_volume", "close_time",
        "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    api_frames = []
    day_audits = []
    for day, item in sorted(by_day.items()):
        rows = json.loads((ROOT / item["raw_payload_path"]).read_text(encoding="utf-8"))
        frame = pd.DataFrame(rows, columns=columns)
        opened_ms = pd.to_numeric(frame["open_time"], errors="coerce")
        close_ms = pd.to_numeric(frame["close_time"], errors="coerce")
        opened = pd.to_datetime(opened_ms, unit="ms", utc=True, errors="coerce")
        day_start = pd.Timestamp(day, tz="UTC")
        day_end = min(day_start + pd.Timedelta(days=1), download_end)
        expected_day = pd.date_range(max(day_start, download_start), day_end, freq="min", inclusive="left")
        if opened.isna().any() or not opened.is_monotonic_increasing or opened.duplicated().any():
            raise RuntimeError(f"Binance API timestamps are invalid, duplicated, or unordered on {day}")
        if not _same_utc_timestamps(opened, expected_day):
            raise RuntimeError(f"Binance API open timestamps do not exactly cover the requested minute partition {day}")
        if not close_ms.eq(opened_ms + 59_999).all():
            raise RuntimeError(f"Binance close-time milliseconds are invalid on {day}")
        for name in ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"):
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        price_valid = (
            np.isfinite(frame[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)).all(axis=1)
            & frame["open"].gt(0).to_numpy() & frame["high"].gt(0).to_numpy()
            & frame["low"].gt(0).to_numpy() & frame["close"].gt(0).to_numpy()
            & frame["high"].ge(np.maximum(frame["open"], frame["close"])).to_numpy()
            & frame["low"].le(np.minimum(frame["open"], frame["close"])).to_numpy()
            & frame["high"].ge(frame["low"]).to_numpy()
        )
        quote = frame["quote_volume"].to_numpy(dtype=np.float64)
        taker_buy = frame["taker_buy_quote_volume"].to_numpy(dtype=np.float64)
        volume_valid = (
            np.isfinite(quote) & np.isfinite(taker_buy) & (quote >= 0.0)
            & (taker_buy >= 0.0) & (taker_buy <= quote)
        )
        if not price_valid.all() or not volume_valid.all():
            raise RuntimeError(f"Binance OHLC or quote-volume invariants failed on {day}")
        frame["Opened"] = opened
        api_frames.append(frame.loc[:, ["Opened", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]].set_index("Opened"))
        day_audits.append({
            "day": day,
            "expected_rows": int(len(expected_day)),
            "observed_rows": int(len(frame)),
            "timestamp_unit": "milliseconds since Unix epoch UTC",
            "off_minute_or_duplicate_rows": 0,
            "close_time_errors": 0,
            "missing_minutes": 0,
            "invalid_ohlc_rows": 0,
            "invalid_quote_volume_rows": 0,
            "taker_buy_quote_volume_above_quote_volume": 0,
            "raw_payload_sha256": item["payload_sha256"],
        })
    api_raw = pd.concat(api_frames, axis=0).sort_index(kind="stable")
    if api_raw.index.has_duplicates or not api_raw.index.is_monotonic_increasing:
        raise RuntimeError("Binance USD-M source partitions overlap or are unordered")
    if not _same_utc_timestamps(api_raw.index, expected_index):
        raise RuntimeError("Binance USD-M source partitions have a missing or extra minute")

    warmup_path = ROOT / inputs["taker_flow_warmup_archive"]
    checksum_path = ROOT / inputs["taker_flow_warmup_archive_checksum"]
    old_flow_manifest_path = ROOT / "data/analysis/polymarket/BTC/btc_preopen_taker_flow_20261018/run_manifest.json"
    old_flow_manifest = json.loads(old_flow_manifest_path.read_text(encoding="utf-8"))
    warmup_record = old_flow_manifest.get("binance_month_archives", {}).get("2026-06")
    if not warmup_record or warmup_path.relative_to(ROOT).as_posix() != warmup_record.get("zip_path"):
        raise RuntimeError("local June warm-up archive is not the previously pinned Binance monthly source")
    checksum_parts = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(checksum_parts) < 2 or checksum_parts[-1].lstrip("*") != warmup_path.name:
        raise RuntimeError("local June warm-up Binance checksum sidecar is malformed")
    warmup_sha = _sha256(warmup_path)
    if warmup_sha != checksum_parts[0].lower() or warmup_sha != warmup_record.get("sha256"):
        raise RuntimeError("local June warm-up Binance archive failed its pinned checksum")
    with zipfile.ZipFile(warmup_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1 or Path(members[0]).name != "BTCUSDT-1m-2026-06.csv":
            raise RuntimeError("local June warm-up Binance archive has an unexpected CSV member")
        with archive.open(members[0]) as handle:
            warmup = pd.read_csv(handle, header=None, names=columns, usecols=list(range(11)))
    warmup_open_ms = pd.to_numeric(warmup["open_time"], errors="coerce")
    warmup["Opened"] = pd.to_datetime(warmup_open_ms, unit="ms", utc=True, errors="coerce")
    warmup = warmup.loc[warmup["Opened"].ge(warmup_start) & warmup["Opened"].lt(download_start)].copy()
    if not pd.to_numeric(warmup["close_time"], errors="coerce").eq(
        pd.to_numeric(warmup["open_time"], errors="coerce") + 59_999,
    ).all():
        raise RuntimeError("local June warm-up archive has invalid millisecond close times")
    for name in ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"):
        warmup[name] = pd.to_numeric(warmup[name], errors="coerce")
    warmup_expected = pd.date_range(warmup_start, download_start, freq="min", inclusive="left")
    if warmup["Opened"].isna().any() or warmup["Opened"].duplicated().any():
        raise RuntimeError("local June warm-up source has invalid or duplicate timestamps")
    warmup = warmup.set_index("Opened").sort_index(kind="stable")
    if not _same_utc_timestamps(warmup.index, warmup_expected):
        raise RuntimeError("local June warm-up archive has a gap in the required causal history")
    warmup_raw = warmup.loc[:, ["open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]]
    warmup_quote = warmup_raw["quote_volume"].to_numpy(dtype=np.float64)
    warmup_taker = warmup_raw["taker_buy_quote_volume"].to_numpy(dtype=np.float64)
    warmup_open = warmup_raw["open"].to_numpy(dtype=np.float64)
    warmup_high = warmup_raw["high"].to_numpy(dtype=np.float64)
    warmup_low = warmup_raw["low"].to_numpy(dtype=np.float64)
    warmup_close = warmup_raw["close"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(warmup_raw.to_numpy(dtype=np.float64)).all()
        or (warmup_open <= 0.0).any() or (warmup_high <= 0.0).any()
        or (warmup_low <= 0.0).any() or (warmup_close <= 0.0).any()
        or (warmup_high < np.maximum(warmup_open, warmup_close)).any()
        or (warmup_low > np.minimum(warmup_open, warmup_close)).any()
        or (warmup_high < warmup_low).any()
        or (warmup_quote < 0.0).any() or (warmup_taker < 0.0).any()
        or (warmup_taker > warmup_quote).any()
    ):
        raise RuntimeError("local June warm-up archive has invalid prices or quote-volume values")
    combined_raw = pd.concat([warmup_raw, api_raw], axis=0).sort_index(kind="stable")
    required_raw_grid = pd.date_range(warmup_start, download_end, freq="min", inclusive="left")
    if combined_raw.index.has_duplicates or not _same_utc_timestamps(combined_raw.index, required_raw_grid):
        raise RuntimeError("local warm-up and downloaded Binance source do not form the required continuous UTC minute grid")

    raw_source_path = ROOT / inputs["raw_um_ohlcv_source"]
    overlap_columns = ["UM_BTCUSDT_Open", "UM_BTCUSDT_High", "UM_BTCUSDT_Low", "UM_BTCUSDT_Close"]
    overlap_parts = []
    for chunk in pd.read_csv(raw_source_path, usecols=["Opened", *overlap_columns], chunksize=250_000):
        opened = pd.to_datetime(chunk["Opened"], utc=True, errors="coerce")
        keep = opened.ge(download_start) & opened.lt(download_end)
        if keep.any():
            part = chunk.loc[keep, overlap_columns].copy()
            part["Opened"] = opened.loc[keep].to_numpy()
            for name in overlap_columns:
                part[name] = pd.to_numeric(part[name], errors="coerce")
            overlap_parts.append(part)
    if not overlap_parts:
        raise RuntimeError("no overlapping USD-M OHLC rows were available for source consistency checks")
    existing_ohlc = pd.concat(overlap_parts, ignore_index=True)
    existing_ohlc.sort_values("Opened", kind="stable", inplace=True)
    if existing_ohlc["Opened"].duplicated().any():
        raise RuntimeError("existing local USD-M OHLC overlap contains duplicate timestamps")
    api_ohlc = api_raw.reset_index().rename(columns={
        "open": "new_open", "high": "new_high", "low": "new_low", "close": "new_close",
    })
    overlap = existing_ohlc.merge(api_ohlc, on="Opened", how="inner", validate="one_to_one")
    if overlap.empty:
        raise RuntimeError("existing local USD-M OHLC source had no matching rows with the downloaded Binance source")
    overlap_errors = {}
    for old_name, new_name in zip(overlap_columns, ("new_open", "new_high", "new_low", "new_close")):
        delta = np.abs(overlap[old_name].to_numpy(dtype=np.float64) - overlap[new_name].to_numpy(dtype=np.float64))
        overlap_errors[old_name] = {
            "rows_compared": int(len(delta)),
            "mismatch_rows_at_1e-7_absolute_tolerance": int((delta > 1e-7).sum()),
            "maximum_absolute_difference": float(delta.max()),
        }
        if not np.isfinite(delta).all() or (delta > 1e-7).any():
            raise RuntimeError(f"Binance USD-M overlap differs from the existing local source for {old_name}")
    validation_elapsed = time.perf_counter() - validation_started

    flow_config_path = ROOT / "configs/research/btc_preopen_taker_flow_20261018.json"
    flow_config = json.loads(flow_config_path.read_text(encoding="utf-8"))["flow_features"]
    flow_started = time.perf_counter()
    generated = compute_taker_flow_features(
        combined_raw,
        horizons=tuple(int(value) for value in flow_config["horizons_minutes"]),
        qbar_window=int(flow_config["qbar_window_minutes"]),
        sigma_window=int(flow_config["sigma_window_minutes"]),
        beta_window=int(flow_config["beta_window_minutes"]),
        minimum_beta_pairs=int(flow_config["minimum_beta_pairs"]),
        beta_variance_tolerance=float(flow_config["beta_variance_tolerance"]),
        sigma_floor=float(flow_config["sigma_floor"]),
        sigma_ddof=int(flow_config["sigma_ddof"]),
    )
    control_features, _ = feature_names(tuple(int(value) for value in flow_config["horizons_minutes"]))
    feature_mask = (generated.index >= download_start) & (generated.index < download_end)
    supplement = generated.loc[feature_mask, control_features].copy()
    supplement.index.name = "Opened"
    if not _same_utc_timestamps(supplement.index, expected_index):
        raise RuntimeError("computed frozen taker-flow controls do not cover the exact new Binance input range")
    supplement_frame = supplement.reset_index()
    feature_elapsed = time.perf_counter() - flow_started
    if feature_path.exists():
        existing_features = pd.read_parquet(feature_path)
        try:
            pd.testing.assert_frame_equal(existing_features, supplement_frame, check_exact=True)
        except AssertionError as exc:
            raise RuntimeError("existing partial flow-feature output differs from the frozen recomputation") from exc
    else:
        temporary = feature_path.with_suffix(feature_path.suffix + f".{os.getpid()}.tmp")
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        supplement_frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, feature_path)

    expected_volume_rows = int(len(expected_index))
    finite_feature_rows = int(np.isfinite(supplement_frame[control_features].to_numpy(dtype=np.float64)).all(axis=1).sum())
    manifest.update({
        "status": "completed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "required_binance_volume_opened_inclusive_utc": inputs["required_binance_volume_opened_inclusive_utc"],
        "required_binance_volume_opened_exclusive_utc": inputs["required_binance_volume_opened_exclusive_utc"],
        "required_flow_feature_opened_inclusive_utc": inputs["required_flow_feature_opened_inclusive_utc"],
        "required_flow_feature_opened_exclusive_utc": inputs["required_flow_feature_opened_exclusive_utc"],
        "downloaded_archives": [by_day[key] for key in sorted(by_day)],
        "warmup_source": {
            "path": warmup_path.relative_to(ROOT).as_posix(),
            "checksum_path": checksum_path.relative_to(ROOT).as_posix(),
            "sha256": warmup_sha,
            "source_url": warmup_record.get("data_source_url"),
            "checksum_source_url": warmup_record.get("checksum_source_url"),
            "opened_inclusive_utc": warmup_start.isoformat().replace("+00:00", "Z"),
            "opened_exclusive_utc": download_start.isoformat().replace("+00:00", "Z"),
            "rows": int(len(warmup_raw)),
            "already_available_locally": True,
        },
        "coverage": {
            "market_type": "USD-M Futures",
            "symbol": "BTCUSDT",
            "interval": "1m",
            "timestamp_unit": "milliseconds since Unix epoch UTC",
            "required_volume_first_opened_utc": expected_index[0].isoformat().replace("+00:00", "Z"),
            "required_volume_last_opened_utc": expected_index[-1].isoformat().replace("+00:00", "Z"),
            "expected_volume_minutes": expected_volume_rows,
            "observed_volume_minutes": int(len(api_raw)),
            "missing_volume_minutes": int(expected_volume_rows - len(api_raw)),
            "invalid_ohlc_rows": 0,
            "invalid_quote_or_taker_volume_rows": 0,
            "overlap_ohlc_comparison": overlap_errors,
            "overlap_rows_compared": int(len(overlap)),
            "values_imputed_or_approximated": False,
            "per_day_validation": day_audits,
            "flow_feature_rows": int(len(supplement_frame)),
            "flow_feature_rows_with_all_six_controls_finite": finite_feature_rows,
            "flow_feature_first_opened_utc": supplement_frame["Opened"].iloc[0].isoformat().replace("+00:00", "Z"),
            "flow_feature_last_opened_utc": supplement_frame["Opened"].iloc[-1].isoformat().replace("+00:00", "Z"),
        },
        "feature_configuration": flow_config,
        "feature_configuration_sha256": _sha256(flow_config_path),
        "feature_code_sha256": _sha256(ROOT / "features/btc_taker_flow.py"),
        "flow_feature_output": {
            "path": feature_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(feature_path),
            "size_bytes": feature_path.stat().st_size,
            "rows": int(len(supplement_frame)),
            "columns": ["Opened", *control_features],
        },
        "timings_seconds": {
            "download_seconds_total": float(sum(float(item.get("elapsed_seconds", 0.0)) for item in by_day.values())),
            "source_validation_and_overlap_seconds": float(validation_elapsed),
            "feature_generation_seconds": float(feature_elapsed),
        },
    })
    _atomic_json(manifest_path, manifest)
    verified = _source_completion_identity(config)
    if verified is None:
        raise RuntimeError("completed Binance flow supplement failed its final identity check")
    return verified


def _append_raw_validity(development_valid, source_frame, append_raw):
    """Extend source-candle validity without treating derived warm-up NaNs as resets."""
    import numpy as np
    from run_btc_esn_portfolio_20261020 import _valid_candles

    tail_valid = _valid_candles(source_frame)
    return np.concatenate((np.asarray(development_valid, dtype=bool), tail_valid[np.asarray(append_raw, dtype=bool)]))


def _load_approved_arrays(config: dict, base_order: list[str]):
    """Join the existing 64 source features to the six frozen taker controls and causal raw stream."""
    import numpy as np
    import pandas as pd
    import run_btc_candidate_quality_20261022 as quality
    from features.esn_reservoir import INPUT_COLUMNS, build_causal_esn_inputs

    _, development = quality._configure_source_run()
    source_path = ROOT / config["inputs"]["baseline_feature_dataset"]
    flow_paths = [
        ROOT / config["inputs"]["taker_flow_features"],
        ROOT / config["inputs"]["taker_flow_feature_supplement"],
    ]
    flow_names = ["taker_Z_1m", "taker_V_1m", "taker_Z_5m", "taker_V_5m", "taker_Z_15m", "taker_V_15m"]
    source_features = [name for name in base_order if name not in set(flow_names)]
    required_source = source_features + config["inputs"]["raw_input_schema"] + [
        "Opened", config["inputs"]["target"], config["inputs"]["label_available_at"],
        config["inputs"]["decision_at"], config["inputs"]["feature_available_at"],
        "is_polymarket_preopen_decision",
    ]
    raw_tail_start = pd.Timestamp(int(development["raw_opened"][-1]), unit="ns", tz="UTC") - pd.Timedelta(days=1)
    cutoff = pd.Timestamp(config["inputs"]["exclusive_input_opened_cutoff_utc"])
    source_start = raw_tail_start.strftime("%Y-%m-%d %H:%M:%S")
    source_end = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    flow_start = raw_tail_start.to_pydatetime()
    flow_end = cutoff.to_pydatetime()
    source_frame = pd.read_parquet(
        source_path,
        columns=list(dict.fromkeys(required_source)),
        filters=[("Opened", ">=", source_start), ("Opened", "<", source_end)],
    )
    source_frame["Opened"] = pd.to_datetime(source_frame["Opened"], utc=True, errors="raise")
    source_frame[config["inputs"]["decision_at"]] = pd.to_datetime(
        source_frame[config["inputs"]["decision_at"]], utc=True, errors="raise",
    )
    source_frame[config["inputs"]["label_available_at"]] = pd.to_datetime(
        source_frame[config["inputs"]["label_available_at"]], utc=True, errors="raise",
    )
    source_frame[config["inputs"]["feature_available_at"]] = pd.to_datetime(
        source_frame[config["inputs"]["feature_available_at"]], utc=True, errors="raise",
    )
    source_frame.sort_values("Opened", kind="stable", inplace=True)
    if source_frame["Opened"].duplicated().any() or not source_frame["Opened"].is_monotonic_increasing:
        raise RuntimeError("protected baseline feature input is not unique and chronological")
    if (source_frame["Opened"] >= cutoff.tz_convert("UTC")).any():
        raise RuntimeError("feature source returned an Opened timestamp at or beyond the label-safe cutoff")
    opened_ns = _datetime_ns(source_frame["Opened"])
    decision_ns = _datetime_ns(source_frame[config["inputs"]["decision_at"]])
    available_ns = _datetime_ns(source_frame[config["inputs"]["label_available_at"]])
    feature_ns = _datetime_ns(source_frame[config["inputs"]["feature_available_at"]])
    minute_ns = 60_000_000_000
    if not np.array_equal(decision_ns, opened_ns + minute_ns):
        raise RuntimeError("source decisions do not follow the frozen Opened+1m T-60 contract")
    if not np.array_equal(feature_ns, decision_ns):
        raise RuntimeError("a BASE70 feature is not recorded as available at the T-60 decision")
    if not np.array_equal(available_ns, opened_ns + 7 * minute_ns):
        raise RuntimeError("source labels do not follow the frozen Opened+7m availability contract")

    flow_parts = [
        pd.read_parquet(
            path, columns=["Opened", *flow_names],
            filters=[("Opened", ">=", flow_start), ("Opened", "<", flow_end)],
        )
        for path in flow_paths if path.is_file()
    ]
    flow_frame = pd.concat(flow_parts, ignore_index=True)
    flow_frame["Opened"] = pd.to_datetime(flow_frame["Opened"], utc=True, errors="raise")
    flow_frame.sort_values("Opened", kind="stable", inplace=True)
    if flow_frame["Opened"].duplicated().any():
        raise RuntimeError("original and supplemental taker-flow features overlap")
    source_frame = source_frame.merge(flow_frame, on="Opened", how="left", validate="one_to_one", sort=False)
    source_frame.sort_values("Opened", kind="stable", inplace=True)

    raw_opened, raw_features_frame = build_causal_esn_inputs(source_frame)
    raw_x_tail = raw_features_frame.loc[:, INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    raw_ns_tail = raw_opened.as_unit("ns").asi8
    append_raw = raw_ns_tail > int(development["raw_opened"][-1])
    if not append_raw.any():
        raise RuntimeError("protected source contains no raw minutes after the frozen development stream")
    raw_opened_all = np.concatenate((development["raw_opened"], raw_ns_tail[append_raw]))
    raw_x_all = np.concatenate((development["raw_x"], raw_x_tail[append_raw]), axis=0)
    if np.any(np.diff(raw_opened_all) <= 0):
        raise RuntimeError("development and protected raw streams overlap or are unordered")
    raw_valid_all = _append_raw_validity(development["raw_valid"], source_frame, append_raw)

    source_base = source_frame.loc[:, base_order].to_numpy(dtype=np.float32, copy=True)
    source_opened_ns = _datetime_ns(source_frame["Opened"])
    append_base = source_opened_ns > int(development["base_opened"][-1])
    if not append_base.any():
        raise RuntimeError("protected source contains no baseline rows after the frozen development cache")
    source_frame = source_frame.loc[append_base].reset_index(drop=True)
    base_x = source_base[append_base]
    opened_ns = _datetime_ns(source_frame["Opened"])
    raw_rows = np.searchsorted(raw_opened_all, opened_ns)
    if (raw_rows >= len(raw_opened_all)).any() or not np.array_equal(raw_opened_all[raw_rows], opened_ns):
        raise RuntimeError("BASE70 observations do not align to the causal raw stream")
    _, _, _, _, source_details = quality._load_frozen_inputs(development)
    return {
        "opened_ns": opened_ns,
        "decision_ns": _datetime_ns(source_frame[config["inputs"]["decision_at"]]),
        "label_available_ns": _datetime_ns(source_frame[config["inputs"]["label_available_at"]]),
        "decision_flag": source_frame["is_polymarket_preopen_decision"].to_numpy(dtype=bool),
        "target": source_frame[config["inputs"]["target"]].to_numpy(copy=True),
        "base_x": base_x,
        "raw_opened": raw_opened_all,
        "raw_x": raw_x_all,
        "raw_valid": raw_valid_all,
        "raw_rows": raw_rows,
        "development_arrays": development,
        "esn_config": source_details["config"],
    }


def _generate_dynamic_features(raw_opened, raw_x, raw_valid, memory_bundle, esn_config, esn_bundles):
    """Regenerate memory/ESN PCA8 features with the exact frozen project state engines."""
    import numpy as np
    import pandas as pd
    from features.causal_memory_control import CausalMemoryState
    from features.esn_reservoir import ESNEngine, FeatureScaler, INPUT_COLUMNS

    opened = np.asarray(raw_opened, dtype=np.int64)
    raw = np.asarray(raw_x, dtype=np.float32)
    valid = np.asarray(raw_valid, dtype=bool)
    if raw.ndim != 2 or len(raw) != len(opened) or len(valid) != len(opened):
        raise ValueError("raw causal stream arrays are not aligned")
    timestamps = pd.to_datetime(opened, unit="ns", utc=True)
    memory_columns = list(memory_bundle["identity"]["input_columns"])
    memory_indices = np.asarray([INPUT_COLUMNS.index(name) for name in memory_columns], dtype=np.int64)
    memory_scaler = FeatureScaler.from_dict(memory_bundle["input_scaler"])
    memory_mean = np.asarray(memory_bundle["pca_mean"], dtype=np.float64)
    memory_components = np.asarray(memory_bundle["pca_components"], dtype=np.float64)
    memory_state = CausalMemoryState(len(memory_columns))
    memory_out = np.full((len(opened), 8), np.nan, dtype=np.float32)
    esn_out = {int(seed): np.full((len(opened), 8), np.nan, dtype=np.float32) for seed in esn_bundles}
    esn_states = {}
    for seed, bundle in esn_bundles.items():
        seed = int(seed)
        if int(bundle["seed"]) != seed:
            raise RuntimeError("ESN bundle seed differs from its frozen seed slot")
        esn_states[seed] = ESNEngine(
            esn_config, seed, FeatureScaler.from_dict(bundle["scaler"]),
            warmup_steps=int(bundle["warmup_steps"]),
        )
    input_indices = np.asarray([INPUT_COLUMNS.index(name) for name in esn_config.selected_columns], dtype=np.int64)
    chunk_rows = 65_536
    for start in range(0, len(opened), chunk_rows):
        end = min(len(opened), start + chunk_rows)
        block_raw = raw[start:end]
        # Missing derived inputs are imputed to zero by the frozen FeatureScaler;
        # only the source candle-validity mask resets state in the project runner.
        block_valid = valid[start:end]
        memory_scaled = memory_scaler.transform(block_raw[:, memory_indices].astype(np.float64, copy=False))
        memory_features, memory_ready = memory_state.update_many(
            timestamps[start:end], memory_scaled, valid_mask=block_valid,
        )
        if memory_ready.any():
            projected = (memory_features[memory_ready].astype(np.float64) - memory_mean) @ memory_components.T
            memory_out[start:end][memory_ready] = projected.astype(np.float32)
        esn_input = block_raw[:, input_indices]
        for seed, engine in esn_states.items():
            states, ready = engine.update_many(timestamps[start:end], esn_input, valid_mask=block_valid)
            if ready.any():
                projection = esn_bundles[seed]["projection"]
                mean = np.asarray(projection["mean"], dtype=np.float64)
                components = np.asarray(projection["components"], dtype=np.float64)
                values = (states[ready].astype(np.float64) - mean) @ components.T
                esn_out[seed][start:end][ready] = values.astype(np.float32)
    return memory_out, esn_out


def development_replay_check() -> dict:
    """Exercise the production prediction path only on the already-permitted search cache."""
    import numpy as np
    import pandas as pd
    import run_btc_candidate_quality_20261022 as quality
    from run_btc_preopen_taker_flow_20261018 import PeakMemorySampler

    started = __import__("time").perf_counter()
    peak_sampler = PeakMemorySampler(interval_seconds=0.1)
    peak_sampler.__enter__()
    _, arrays = quality._configure_source_run()
    candidate_id, _, cached_memory, cached_esn, details = quality._load_frozen_inputs(arrays)
    memory_bundle = json.loads((BUNDLE_DIR / "transforms/simple_memory_projection.json").read_text(encoding="utf-8"))
    esn_bundles = {
        int(seed): json.loads((BUNDLE_DIR / f"transforms/esn_seed_{int(seed)}.json").read_text(encoding="utf-8"))
        for seed in quality.FREEZE_CONFIG["model_fit"]["reservoir_seeds"]
    }
    memory_generated, esn_generated = _generate_dynamic_features(
        arrays["raw_opened"], arrays["raw_x"], arrays["raw_valid"],
        memory_bundle, details["config"], esn_bundles,
    )
    raw_rows = np.asarray(arrays["base_to_raw"], dtype=np.int64)
    def finite_error(actual, expected):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        actual_finite = np.isfinite(actual)
        expected_finite = np.isfinite(expected)
        mismatch_rows = np.flatnonzero(np.any(actual_finite != expected_finite, axis=1))
        error = 0.0
        if actual_finite.any() and expected_finite.any():
            common_finite = actual_finite & expected_finite
            if common_finite.any():
                error = float(np.max(np.abs(actual[common_finite] - expected[common_finite])))
        return {
            "max_abs_error_common_finite_rows": error,
            "readiness_mask_mismatch_rows": int(len(mismatch_rows)),
            "first_mismatch_row_indices": mismatch_rows[:10].astype(int).tolist(),
        }

    memory_audit = finite_error(memory_generated[raw_rows], cached_memory)
    esn_error = {
        str(seed): finite_error(esn_generated[int(seed)][raw_rows], cached_esn[int(seed)])
        for seed in esn_generated
    }
    train_mask, validation_rows = quality._common_masks(arrays, candidate_id, cached_memory, cached_esn)
    boosters = _load_frozen_boosters({})
    members = predict_members(
        arrays["base_x"][validation_rows], memory_generated[raw_rows[validation_rows]],
        {seed: values[raw_rows[validation_rows]] for seed, values in esn_generated.items()},
        boosters,
        lightgbm_seeds=quality.FREEZE_CONFIG["model_fit"]["lightgbm_seeds"],
        reservoir_seeds=quality.FREEZE_CONFIG["model_fit"]["reservoir_seeds"],
    )
    prediction_errors = {}
    for seed in quality.FREEZE_CONFIG["model_fit"]["lightgbm_seeds"]:
        expected = {
            "A": quality._load_new_prediction("A_fitcore_common_mask", None, seed, validation_rows, 70),
            "B": quality._load_stored_prediction("simple_memory_pca8", None, seed, validation_rows, 78),
        }
        prediction_errors[f"A:{seed}"] = float(np.max(np.abs(members["A"][str(seed)] - expected["A"])))
        prediction_errors[f"B:{seed}"] = float(np.max(np.abs(members["B"][str(seed)] - expected["B"])))
        for reservoir_seed in quality.FREEZE_CONFIG["model_fit"]["reservoir_seeds"]:
            for variant, count in (("C", 78), ("D", 86)):
                prediction_key = f"{reservoir_seed}:{seed}"
                variant_name = (
                    "C_fitcore_common_mask" if variant == "C"
                    else f"D_memory_plus_{candidate_id}"
                )
                expected_p = quality._load_new_prediction(
                    variant_name, reservoir_seed, seed, validation_rows, count,
                )
                prediction_errors[f"{variant}:{reservoir_seed}:{seed}"] = float(
                    np.max(np.abs(members[variant][prediction_key] - expected_p))
                )
    max_prediction_error = max(prediction_errors.values())
    max_feature_error = max(
        [memory_audit["max_abs_error_common_finite_rows"]]
        + [item["max_abs_error_common_finite_rows"] for item in esn_error.values()]
    )
    if max_prediction_error > 2e-5 or max_feature_error > 2e-6:
        raise RuntimeError("development replay mismatch: " + json.dumps({
            "memory": memory_audit,
            "esn": esn_error,
            "max_prediction_error": max_prediction_error,
            "max_feature_error": max_feature_error,
        }, sort_keys=True))
    decision_hash = quality._digest_array(arrays["base_decision_at"][validation_rows].astype(np.int64))
    target_hash = quality._digest_array(arrays["target"][validation_rows].astype(np.int8))
    peak_sampler.__exit__(None, None, None)
    result = {
        "status": "passed_development_only_replay",
        "scope_end_exclusive_utc": "2025-07-01T00:00:00Z",
        "confirmation_read": False,
        "holdout_rows_materialized": 0,
        "common_validation_rows": int(len(validation_rows)),
        "fit_rows": int(train_mask.sum()),
        "decision_ids_sha256": decision_hash,
        "target_sha256": target_hash,
        "memory_feature_replay_vs_frozen_cache": memory_audit,
        "esn_feature_max_abs_error_by_seed_vs_frozen_cache": esn_error,
        "feature_readiness_masks_match": memory_audit["readiness_mask_mismatch_rows"] == 0 and all(
            item["readiness_mask_mismatch_rows"] == 0 for item in esn_error.values()
        ),
        "prediction_max_abs_error_by_member_vs_frozen_source": prediction_errors,
        "prediction_tolerance": 2e-5,
        "models_reloaded_from_frozen_fit_core_bundle": True,
        "refit_or_early_stopping": False,
        "uniform_ensemble_weights": True,
        "elapsed_seconds": float(__import__("time").perf_counter() - started),
        "peak_process_working_set_bytes": int(peak_sampler.peak_bytes),
        "input_description": "existing development BASE70 and raw14 search cache; raw-to-BASE70 was not part of this check",
    }
    return result


def _fixed_interval_results(target, dates_ns, ensemble) -> dict:
    import numpy as np
    import pandas as pd
    from utils.btc_oof_audit import moving_calendar_block_multiplicity

    timestamps = pd.to_datetime(dates_ns, utc=True)
    if len(timestamps) != len(target) or len(timestamps) == 0:
        raise ValueError("bootstrap timestamps must align with a non-empty common sample")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = {f"{right}-{left}": {metric: np.empty(BOOTSTRAP_REPLICATES) for metric in ("log_loss", "brier", "auc")} for left, right in ALL_PAIRS}
    prepared = {}
    y = np.asarray(target, dtype=np.float64)
    for name, values in ensemble.items():
        p = np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        order = np.argsort(p, kind="mergesort")
        sorted_p = p[order]
        prepared[name] = {
            "p": p,
            "order": order,
            "y_sorted": y[order],
            "score_starts": np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1],
            "ll": -(y * np.log(p) + (1.0 - y) * np.log1p(-p)),
            "brier": np.square(p - y),
        }

    def metrics(item, weights):
        total = float(weights.sum())
        ll = float(np.dot(weights, item["ll"]) / total)
        brier = float(np.dot(weights, item["brier"]) / total)
        sorted_weights = weights[item["order"]]
        pos_by = np.add.reduceat(sorted_weights * item["y_sorted"], item["score_starts"])
        neg_by = np.add.reduceat(sorted_weights * (1.0 - item["y_sorted"]), item["score_starts"])
        positives, negatives = float(pos_by.sum()), float(neg_by.sum())
        auc = float("nan")
        if positives and negatives:
            auc = float(np.dot(pos_by, np.cumsum(neg_by) - neg_by + 0.5 * neg_by) / (positives * negatives))
        return ll, brier, auc

    for rep in range(BOOTSTRAP_REPLICATES):
        weights = moving_calendar_block_multiplicity(timestamps, rng, block_days=BOOTSTRAP_BLOCK_DAYS)
        by_name = {name: metrics(item, weights) for name, item in prepared.items()}
        for left, right in ALL_PAIRS:
            key = f"{right}-{left}"
            differences[key]["log_loss"][rep] = by_name[left][0] - by_name[right][0]
            differences[key]["brier"][rep] = by_name[left][1] - by_name[right][1]
            differences[key]["auc"][rep] = by_name[right][2] - by_name[left][2]
    intervals = {
        pair: {metric: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] for metric, values in entries.items()}
        for pair, entries in differences.items()
    }
    corrected = {
        pair: {metric: [float(np.quantile(values, 0.0125)), float(np.quantile(values, 0.9875))] for metric, values in differences[pair].items()}
        for pair in ("B-A", "D-B")
    }
    return {
        "method": "paired non-circular moving 3-calendar-day block bootstrap; one shared set of 2,000 row multiplicity draws for all candidates and pairs",
        "block_length_calendar_days": BOOTSTRAP_BLOCK_DAYS,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "pairwise_95_percentile_intervals": intervals,
        "primary_bonferroni_97_5_percent_intervals": corrected,
        "primary_multiple_comparison_correction": "Bonferroni familywise alpha 0.05 across only the prespecified D-B and B-A primary comparisons; 97.5% two-sided percentile intervals per comparison. This does not correct prior search or historical exposure.",
    }


def score_period(
    target, dates_ns, members, *, lightgbm_seeds, reservoir_seeds,
    period_start_utc: str, period_end_utc: str,
) -> dict:
    import time
    import numpy as np

    score_started = time.perf_counter()
    ensemble = ensemble_predictions(members, lightgbm_seeds, reservoir_seeds)
    summary = {name: score_metrics(target, values) for name, values in ensemble.items()}
    member_metrics = {
        name: {key: score_metrics(target, values) for key, values in members[name].items()}
        for name in EXPECTED_PROCEDURES
    }
    mean_member_metrics = {
        name: {
            metric: float(np.mean([item[metric] for item in rows.values() if item[metric] is not None]))
            for metric in ("log_loss", "brier", "auc")
        }
        for name, rows in member_metrics.items()
    }
    pairwise = {}
    for left, right in ALL_PAIRS:
        pairwise[f"{right}-{left}"] = {
            "delta_log_loss": summary[left]["log_loss"] - summary[right]["log_loss"],
            "delta_brier": summary[left]["brier"] - summary[right]["brier"],
            "delta_auc": None if summary[left]["auc"] is None or summary[right]["auc"] is None else summary[right]["auc"] - summary[left]["auc"],
        }
    pd = __import__("pandas")
    month_metrics = {}
    timestamps = pd.to_datetime(dates_ns, utc=True)
    start = pd.Timestamp(period_start_utc).tz_convert("UTC")
    end = pd.Timestamp(period_end_utc).tz_convert("UTC")
    first_month = start.tz_localize(None).to_period("M")
    last_month = (end - pd.Timedelta(nanoseconds=1)).tz_localize(None).to_period("M")
    row_months = timestamps.strftime("%Y-%m").to_numpy()
    for period_month in pd.period_range(first_month, last_month, freq="M"):
        month = str(period_month)
        rows = row_months == month
        if rows.any():
            metrics_by_candidate = {name: score_metrics(np.asarray(target)[rows], p[rows]) for name, p in ensemble.items()}
        else:
            metrics_by_candidate = {
                name: {"rows": 0, "log_loss": None, "brier": None, "auc": None}
                for name in EXPECTED_PROCEDURES
            }
        month_pairs = {
            f"{right}-{left}": {
                "delta_log_loss": (
                    None if metrics_by_candidate[left]["log_loss"] is None
                    else metrics_by_candidate[left]["log_loss"] - metrics_by_candidate[right]["log_loss"]
                ),
                "delta_brier": (
                    None if metrics_by_candidate[left]["brier"] is None
                    else metrics_by_candidate[left]["brier"] - metrics_by_candidate[right]["brier"]
                ),
                "delta_auc": (
                    None if metrics_by_candidate[left]["auc"] is None or metrics_by_candidate[right]["auc"] is None
                    else metrics_by_candidate[right]["auc"] - metrics_by_candidate[left]["auc"]
                ),
            }
            for left, right in ALL_PAIRS
        }
        month_start = max(start, period_month.start_time.tz_localize("UTC"))
        month_end = min(end, (period_month + 1).start_time.tz_localize("UTC"))
        month_metrics[month] = {
            "market_start_inclusive_utc": month_start.isoformat().replace("+00:00", "Z"),
            "market_start_exclusive_utc": month_end.isoformat().replace("+00:00", "Z"),
            "common_rows": int(rows.sum()),
            "candidate_metrics": metrics_by_candidate,
            "pairwise_differences": month_pairs,
        }
    bootstrap_started = time.perf_counter()
    bootstrap = _fixed_interval_results(target, dates_ns, ensemble)
    bootstrap_seconds = time.perf_counter() - bootstrap_started
    return {
        "rows": int(len(target)),
        "target_positive_fraction": float(np.mean(target)),
        "individual_member_metrics": member_metrics,
        "mean_individual_member_metrics": mean_member_metrics,
        "member_counts": {name: len(rows) for name, rows in member_metrics.items()},
        "uniform_ensemble_metrics": summary,
        "uniform_ensemble_pairwise_differences": pairwise,
        "fixed_utc_calendar_month_metrics": month_metrics,
        "bootstrap": bootstrap,
        "timings_seconds": {
            "metrics_and_subperiods_seconds": max(0.0, time.perf_counter() - score_started - bootstrap_seconds),
            "bootstrap_seconds": bootstrap_seconds,
            "scoring_total_seconds": time.perf_counter() - score_started,
        },
        "not_a_profitability_test": True,
    }


def save_period_prediction_artifacts(
    run_dir: Path, run_identity: str, period_id: str, start_ns: int, end_ns: int,
    inputs: dict, common_mask, memory, esn, rows, members, ensemble,
) -> dict:
    """Persist common-sample predictions and the full expected-decision eligibility mask."""
    import numpy as np
    import pandas as pd

    market_start_ns = inputs["opened_ns"] + 2 * 60_000_000_000
    in_scope = (market_start_ns >= int(start_ns)) & (market_start_ns < int(end_ns))
    expected = in_scope & inputs["decision_flag"]
    label_released = inputs["label_available_ns"] < int(end_ns)
    target = np.asarray(inputs["target"])
    eligibility = pd.DataFrame({
        "Opened": pd.to_datetime(inputs["opened_ns"][expected], utc=True),
        "market_start_utc": pd.to_datetime(market_start_ns[expected], utc=True),
        "label_available_at_utc": pd.to_datetime(inputs["label_available_ns"][expected], utc=True),
        "target": target[expected],
        "binary_target": np.isin(target[expected], (0, 1)),
        "label_released_strictly_before_period_end": label_released[expected],
        "BASE70_complete": np.isfinite(inputs["base_x"][expected]).all(axis=1),
        "memory_PCA8_complete": np.isfinite(memory[expected]).all(axis=1),
        "all_ESN_PCA8_complete": np.logical_and.reduce([
            np.isfinite(values[expected]).all(axis=1) for values in esn.values()
        ]),
        "common_evaluation_mask": np.asarray(common_mask, dtype=bool)[expected],
    })
    for seed, values in sorted(esn.items()):
        eligibility[f"ESN_{seed}_PCA8_complete"] = np.isfinite(values[expected]).all(axis=1)

    predictions = pd.DataFrame({
        "Opened": pd.to_datetime(inputs["opened_ns"][rows], utc=True),
        "market_start_utc": pd.to_datetime(market_start_ns[rows], utc=True),
        "target": target[rows].astype(np.int8),
        "common_evaluation_mask": True,
    })
    for name in EXPECTED_PROCEDURES:
        predictions[f"ensemble_probability_{name}"] = ensemble[name]
        for member_key, values in sorted(members[name].items()):
            predictions[f"member_probability_{name}_{member_key.replace(':', '_')}"] = values

    artifact_manifest_path = run_dir / f"period_{period_id}_artifacts.json"
    common_mask_hash = hashlib.sha256(np.asarray(common_mask, dtype=np.uint8).tobytes()).hexdigest()
    if artifact_manifest_path.is_file():
        prior = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
        if prior.get("run_identity_sha256") != run_identity or prior.get("common_mask_sha256") != common_mask_hash:
            raise RuntimeError(f"existing {period_id} artifact manifest belongs to a different common mask or run")
        for record in prior.get("artifacts", {}).values():
            prior_path = ROOT / record["path"]
            if not prior_path.is_file() or _sha256(prior_path) != record.get("sha256"):
                raise RuntimeError(f"existing {period_id} prediction artifact failed its checksum")
        prior["manifest_path"] = artifact_manifest_path.relative_to(ROOT).as_posix()
        prior["manifest_sha256"] = _sha256(artifact_manifest_path)
        return prior

    artifacts = {}
    for kind, frame in (("predictions", predictions), ("eligibility", eligibility)):
        path = run_dir / f"period_{period_id}_{kind}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            try:
                pd.testing.assert_frame_equal(existing, frame, check_exact=True)
            except AssertionError as exc:
                raise RuntimeError(f"existing partial {kind} artifact differs from the frozen rerun: {path.name}") from exc
        else:
            temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
            frame.to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, path)
        artifacts[kind] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "rows": int(len(frame)),
        }
    artifact_manifest = {
        "period_id": period_id,
        "run_identity_sha256": run_identity,
        "common_mask_sha256": common_mask_hash,
        "artifacts": artifacts,
    }
    _atomic_json(artifact_manifest_path, artifact_manifest)
    artifact_manifest["manifest_path"] = artifact_manifest_path.relative_to(ROOT).as_posix()
    artifact_manifest["manifest_sha256"] = _sha256(artifact_manifest_path)
    return artifact_manifest


def main() -> None:
    import time

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    try:
        validate_scope(config)
    except GateBlocked as exc:
        raise SystemExit(str(exc)) from exc

    # Imports of data/model code occur only after both exact period approvals.
    bundle = _bundle_identity()
    identity = {
        "experiment_id": config["experiment_id"],
        "config_sha256": _sha256(CONFIG_PATH),
        "protocol_sha256": _canonical_hash(config["protocol"]),
        "bundle": bundle,
        "source_data": _source_artifact_identity(),
        "development_cache": _development_cache_identity(config),
        "inputs": config["inputs"],
        "source_code_sha256": {
            path: _sha256(ROOT / path) for path in config["protocol"]["source_code_paths"]
        },
    }
    run_dir = _write_preread_manifest(config, identity)
    run_identity = json.loads((run_dir / "pre_read_manifest.json").read_text(encoding="utf-8"))["run_identity_sha256"]
    execution_started = time.perf_counter()
    stage_timings = {}
    record_attempt(run_dir, "started")
    try:
        source_hash_started = time.perf_counter()
        source_verification = _verify_source_artifacts(identity["source_data"])
        stage_timings["pinned_source_hash_verification_seconds"] = time.perf_counter() - source_hash_started
        _atomic_json(run_dir / "source_verification.json", {
            "run_identity_sha256": run_identity,
            "artifacts": source_verification,
        })
    except Exception as exc:
        record_attempt(run_dir, "failed_source_hash_verification", detail=f"{type(exc).__name__}: {exc}")
        _atomic_json(run_dir / "run_manifest.json", {
            "status": "failed_source_hash_verification",
            "run_identity_sha256": run_identity,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "protected_rows_scored": 0,
        })
        raise
    flow_paths = [
        ROOT / config["inputs"]["taker_flow_features"],
        ROOT / config["inputs"]["taker_flow_feature_supplement"],
    ]
    try:
        supplement_identity = _source_completion_identity(config)
        source_verification_path = run_dir / "source_verification.json"
        verification = json.loads(source_verification_path.read_text(encoding="utf-8"))
        verification["supplemental_source_data"] = supplement_identity
        _atomic_json(source_verification_path, verification)
        missing_input = _flow_coverage_error(
            flow_paths,
            config["inputs"]["required_flow_feature_opened_inclusive_utc"],
            config["inputs"]["required_flow_feature_opened_exclusive_utc"],
        )
        if supplement_identity is None:
            print("Required taker-flow supplement is missing or incomplete; starting/resuming the approved exact-range Binance source completion.", flush=True)
            supplement_identity = _complete_taker_flow_source(config, run_dir, run_identity)
            missing_input = _flow_coverage_error(
                flow_paths,
                config["inputs"]["required_flow_feature_opened_inclusive_utc"],
                config["inputs"]["required_flow_feature_opened_exclusive_utc"],
            )
            verification = json.loads(source_verification_path.read_text(encoding="utf-8"))
            verification["supplemental_source_data"] = supplement_identity
            _atomic_json(source_verification_path, verification)
        if missing_input is None and supplement_identity is None:
            missing_input = "supplemental Binance flow-source provenance manifest is missing"
    except Exception as exc:
        record_attempt(run_dir, "failed_input_preflight", detail=f"{type(exc).__name__}: {exc}")
        _atomic_json(run_dir / "run_manifest.json", {
            "status": "failed_input_preflight",
            "run_identity_sha256": run_identity,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "protected_rows_scored": 0,
        })
        raise
    if missing_input:
        record_attempt(run_dir, "blocked_missing_input", detail=missing_input)
        _atomic_json(run_dir / "run_manifest.json", {
            "status": "blocked_missing_input",
            "run_identity_sha256": run_identity,
            "error": missing_input,
            "protected_rows_scored": 0,
        })
        raise SystemExit("BLOCKED_INPUT: " + missing_input)

    try:
        input_load_started = time.perf_counter()
        inputs = _load_approved_arrays(config, bundle["base_feature_order"])
        stage_timings["protected_input_read_and_join_seconds"] = time.perf_counter() - input_load_started
        memory_bundle = json.loads((BUNDLE_DIR / "transforms/simple_memory_projection.json").read_text(encoding="utf-8"))
        esn_bundles = {
            seed: json.loads((BUNDLE_DIR / f"transforms/esn_seed_{seed}.json").read_text(encoding="utf-8"))
            for seed in bundle["reservoir_seeds"]
        }
        feature_started = time.perf_counter()
        memory, esn = _generate_dynamic_features(
            inputs["raw_opened"], inputs["raw_x"], inputs["raw_valid"],
            memory_bundle, inputs["esn_config"], esn_bundles,
        )
        stage_timings["memory_and_ESN_feature_replay_seconds"] = time.perf_counter() - feature_started
        raw_rows = inputs["raw_rows"]
        selected_memory = memory[raw_rows]
        selected_esn = {seed: values[raw_rows] for seed, values in esn.items()}
        boosters = _load_frozen_boosters(bundle)
        completed = []
        prediction_seconds = 0.0
        for period in config["periods"]:
            period_id = period["id"]
            prior = load_period_checkpoint(run_dir, period_id, run_identity)
            if prior is not None:
                completed.append(period_id)
                continue
            start_ns = int(__import__("pandas").Timestamp(period["market_start_inclusive_utc"]).value)
            end_ns = int(__import__("pandas").Timestamp(period["market_start_exclusive_utc"]).value)
            mask, exclusions = common_observation_mask(
                inputs["opened_ns"], inputs["decision_ns"], inputs["label_available_ns"],
                inputs["decision_flag"], inputs["target"], inputs["base_x"], selected_memory,
                selected_esn, start_ns=start_ns, end_ns=end_ns,
            )
            rows = __import__("numpy").flatnonzero(mask)
            if not len(rows):
                raise RuntimeError(f"no common scoring rows remain for mandatory period {period_id}")
            prediction_started = time.perf_counter()
            members = predict_members(
                inputs["base_x"][rows], selected_memory[rows],
                {seed: values[rows] for seed, values in selected_esn.items()}, boosters,
                lightgbm_seeds=bundle["lightgbm_seeds"], reservoir_seeds=bundle["reservoir_seeds"],
            )
            period_prediction_seconds = time.perf_counter() - prediction_started
            prediction_seconds += period_prediction_seconds
            market_start_ns = inputs["opened_ns"][rows] + 2 * 60_000_000_000
            result = score_period(
                inputs["target"][rows], market_start_ns, members,
                lightgbm_seeds=bundle["lightgbm_seeds"], reservoir_seeds=bundle["reservoir_seeds"],
                period_start_utc=period["market_start_inclusive_utc"],
                period_end_utc=period["market_start_exclusive_utc"],
            )
            ensemble = ensemble_predictions(members, bundle["lightgbm_seeds"], bundle["reservoir_seeds"])
            period_artifacts = save_period_prediction_artifacts(
                run_dir, run_identity, period_id, start_ns, end_ns,
                inputs, mask, selected_memory, selected_esn, rows, members, ensemble,
            )
            result.update({
                "period_id": period_id,
                "market_start_inclusive_utc": period["market_start_inclusive_utc"],
                "market_start_exclusive_utc": period["market_start_exclusive_utc"],
                "label_available_strictly_before_utc": period["market_start_exclusive_utc"],
                "prior_exposure": period["prior_exposure"],
                "exclusions": exclusions,
                "decision_ids_sha256": hashlib.sha256(inputs["decision_ns"][rows].astype("<i8", copy=False).tobytes()).hexdigest(),
                "target_sha256": hashlib.sha256(inputs["target"][rows].astype("i1", copy=False).tobytes()).hexdigest(),
                "expected_decisions": exclusions["expected_decisions"],
                "decisions_with_complete_features": exclusions["decisions_with_complete_features"],
                "prediction_artifacts": period_artifacts,
                "timings_seconds": result["timings_seconds"] | {"member_prediction_seconds": period_prediction_seconds},
                "score_source": "frozen fit-core model bundles; no refit, tuning, calibration, or test early stopping",
            })
            save_period_checkpoint(run_dir, period_id, run_identity, result)
            completed.append(period_id)
        if completed != list(PROTECTED_WINDOWS):
            raise RuntimeError("both mandatory periods were not completed in the frozen order")
        finalize_completed_run(
            run_dir,
            run_identity,
            supplement_identity,
            stage_timings | {
                "member_prediction_total_seconds": prediction_seconds,
                "end_to_end_after_pre_read_seconds": time.perf_counter() - execution_started,
            },
        )
    except Exception as exc:
        record_attempt(run_dir, "failed", detail=f"{type(exc).__name__}: {exc}")
        _atomic_json(run_dir / "run_manifest.json", {
            "status": "failed_resumable",
            "run_identity_sha256": run_identity,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_periods": [
                period for period in PROTECTED_WINDOWS
                if load_period_checkpoint(run_dir, period, run_identity) is not None
            ],
        })
        raise


if __name__ == "__main__":
    main()
