"""Score the frozen BTC candidates on the two previously exposed periods, fail closed by default."""
from __future__ import annotations

import hashlib
import json
import os
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
        "excluded_invalid_memory_or_ESN": int((in_scope & decisions & binary & label_released & valid_base & ~(valid_memory & valid_esn)).sum()),
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
            model_records.append({"procedure": name, "path": model_path.relative_to(ROOT).as_posix(), "sha256": observed})
    transform_paths = [
        BUNDLE_DIR / "transforms/simple_memory_projection.json",
        *(BUNDLE_DIR / f"transforms/esn_seed_{seed}.json" for seed in (20261023, 20261024, 20261025)),
    ]
    transform_hashes = {path.relative_to(ROOT).as_posix(): _sha256(path) for path in transform_paths}
    for path in transform_paths[1:]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = payload.get("identity", {})
        if identity.get("confirmation_read") is not False or identity.get("holdout_rows_materialized") != 0:
            raise RuntimeError(f"ESN transform provenance is not development-only: {path.name}")
    esn_feature_names = {}
    for path in transform_paths[1:]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        if seed in esn_feature_names:
            raise RuntimeError(f"duplicate frozen ESN transform seed: {seed}")
        esn_feature_names[seed] = payload.get("feature_names")
    memory_payload = json.loads(transform_paths[0].read_text(encoding="utf-8"))
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
    return {
        "base_feature_order_sha256": _canonical_hash(base_order),
        "base_feature_order": base_order,
        "procedure_manifest_sha256": {name: _sha256(BUNDLE_DIR / f"{name}_manifest.json") for name in EXPECTED_PROCEDURES},
        "transform_sha256": transform_hashes,
        "model_sha256": model_records,
        "training_identity": manifests["A"]["training"],
        "lightgbm_seeds": sorted(int(key) for key in manifests["A"]["training"]["lgbm_parameters_by_seed"]),
        "reservoir_seeds": sorted({int(item["reservoir_seed"]) for item in manifests["C"]["members"]}),
        "uniform_ensemble_weights": {name: manifests[name]["ensemble_weights"] for name in EXPECTED_PROCEDURES},
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
        "runner_sha256": _sha256(Path(__file__)),
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
    return {
        "baseline_feature_dataset": {
            "path": dataset_path.relative_to(ROOT).as_posix(),
            "pinned_sha256": _manifest_output_hash(feature_run_manifest, dataset_path),
            "source_manifest_sha256": _sha256(feature_run_manifest),
        },
        "taker_flow_features": {
            "path": flow_path.relative_to(ROOT).as_posix(),
            "pinned_sha256": _manifest_output_hash(flow_manifest_path, flow_path),
            "source_manifest_sha256": _sha256(flow_manifest_path),
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
        digest = _sha256(path)
        if digest != record["pinned_sha256"]:
            raise RuntimeError(f"pinned source data hash mismatch for {name}: {record['path']}")
        observed[name] = {"path": record["path"], "sha256": digest, "verified": True}
    return observed


def _flow_coverage_error(flow_path: Path, required_opened_exclusive: str) -> str | None:
    """Inspect only Parquet footer statistics before opening any protected row values."""
    import pyarrow.parquet as pq
    import pandas as pd

    if not flow_path.is_file():
        return f"missing taker-flow feature source: {flow_path.relative_to(ROOT).as_posix()}"
    parquet = pq.ParquetFile(flow_path)
    names = parquet.schema_arrow.names
    required = ("Opened", "taker_Z_1m", "taker_V_1m", "taker_Z_5m", "taker_V_5m", "taker_Z_15m", "taker_V_15m")
    missing = sorted(set(required) - set(names))
    if missing:
        return f"taker-flow input is missing BASE70 columns: {missing}"
    opened_column = names.index("Opened")
    maxima = []
    for index in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(index).column(opened_column).statistics
        if stats is not None and stats.has_min_max:
            maxima.append(pd.Timestamp(stats.max))
    if not maxima:
        return "taker-flow source has no timestamp statistics to verify coverage"
    latest = max(maxima)
    if latest.tzinfo is None:
        latest = latest.tz_localize("UTC")
    else:
        latest = latest.tz_convert("UTC")
    required_last = pd.Timestamp(required_opened_exclusive, tz="UTC") - pd.Timedelta(minutes=1)
    if latest < required_last:
        return (
            "taker-flow inputs stop at " + latest.isoformat()
            + "; the exact missing input is Binance USD-M BTCUSDT 1m quote-volume and taker-buy-quote-volume "
            + "history (needed to produce taker_Z/taker_V at BASE70 horizons 1m, 5m, and 15m) from "
            + "2026-07-01 through 2026-10-06 23:52 UTC. No protected row values were opened."
        )
    return None


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
    flow_path = ROOT / config["inputs"]["taker_flow_features"]
    flow_names = ["taker_Z_1m", "taker_V_1m", "taker_Z_5m", "taker_V_5m", "taker_Z_15m", "taker_V_15m"]
    source_features = [name for name in base_order if name not in set(flow_names)]
    required_source = source_features + config["inputs"]["raw_input_schema"] + [
        "Opened", config["inputs"]["target"], config["inputs"]["label_available_at"],
        config["inputs"]["decision_at"], config["inputs"]["feature_available_at"],
        "is_polymarket_preopen_decision",
    ]
    raw_tail_start = pd.Timestamp(int(development["raw_opened"][-1]), unit="ns", tz="UTC") - pd.Timedelta(days=1)
    cutoff = pd.Timestamp(config["inputs"]["exclusive_input_opened_cutoff_utc"])
    source_frame = pd.read_parquet(
        source_path,
        columns=list(dict.fromkeys(required_source)),
        filters=[("Opened", ">=", raw_tail_start.strftime("%Y-%m-%d %H:%M:%S")),
                 ("Opened", "<", cutoff.strftime("%Y-%m-%d %H:%M:%S"))],
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

    flow_start = raw_tail_start.strftime("%Y-%m-%d %H:%M:%S")
    flow_frame = pd.read_parquet(
        flow_path, columns=["Opened", *flow_names],
        filters=[("Opened", ">=", flow_start), ("Opened", "<", cutoff.strftime("%Y-%m-%d %H:%M:%S"))],
    )
    flow_frame["Opened"] = pd.to_datetime(flow_frame["Opened"], utc=True, errors="raise")
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
    import numpy as np

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
    return {
        "rows": int(len(target)),
        "target_positive_fraction": float(np.mean(target)),
        "individual_member_metrics": member_metrics,
        "mean_individual_member_metrics": mean_member_metrics,
        "member_counts": {name: len(rows) for name, rows in member_metrics.items()},
        "uniform_ensemble_metrics": summary,
        "uniform_ensemble_pairwise_differences": pairwise,
        "fixed_utc_calendar_month_metrics": month_metrics,
        "bootstrap": _fixed_interval_results(target, dates_ns, ensemble),
        "not_a_profitability_test": True,
    }


def main() -> None:
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
    record_attempt(run_dir, "started")
    try:
        source_verification = _verify_source_artifacts(identity["source_data"])
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
    flow_path = ROOT / config["inputs"]["taker_flow_features"]
    try:
        missing_input = _flow_coverage_error(flow_path, config["inputs"]["exclusive_input_opened_cutoff_utc"])
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
        inputs = _load_approved_arrays(config, bundle["base_feature_order"])
        memory_bundle = json.loads((BUNDLE_DIR / "transforms/simple_memory_projection.json").read_text(encoding="utf-8"))
        esn_bundles = {
            seed: json.loads((BUNDLE_DIR / f"transforms/esn_seed_{seed}.json").read_text(encoding="utf-8"))
            for seed in bundle["reservoir_seeds"]
        }
        memory, esn = _generate_dynamic_features(
            inputs["raw_opened"], inputs["raw_x"], inputs["raw_valid"],
            memory_bundle, inputs["esn_config"], esn_bundles,
        )
        raw_rows = inputs["raw_rows"]
        selected_memory = memory[raw_rows]
        selected_esn = {seed: values[raw_rows] for seed, values in esn.items()}
        boosters = _load_frozen_boosters(bundle)
        completed = []
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
            members = predict_members(
                inputs["base_x"][rows], selected_memory[rows],
                {seed: values[rows] for seed, values in selected_esn.items()}, boosters,
                lightgbm_seeds=bundle["lightgbm_seeds"], reservoir_seeds=bundle["reservoir_seeds"],
            )
            market_start_ns = inputs["opened_ns"][rows] + 2 * 60_000_000_000
            result = score_period(
                inputs["target"][rows], market_start_ns, members,
                lightgbm_seeds=bundle["lightgbm_seeds"], reservoir_seeds=bundle["reservoir_seeds"],
                period_start_utc=period["market_start_inclusive_utc"],
                period_end_utc=period["market_start_exclusive_utc"],
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
                "score_source": "frozen fit-core model bundles; no refit, tuning, calibration, or test early stopping",
            })
            save_period_checkpoint(run_dir, period_id, run_identity, result)
            completed.append(period_id)
        if completed != list(PROTECTED_WINDOWS):
            raise RuntimeError("both mandatory periods were not completed in the frozen order")
        _atomic_json(run_dir / "run_manifest.json", {
            "status": "completed_historical_robustness_scoring",
            "run_identity_sha256": run_identity,
            "completed_periods": completed,
            "protected_rows_scored": True,
            "not_fresh_or_independent_validation": True,
            "not_a_profitability_test": True,
        })
        record_attempt(run_dir, "completed")
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
