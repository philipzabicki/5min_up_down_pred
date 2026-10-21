"""Causal development-only ESN architecture comparison on frozen search validation."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from features.esn_reservoir import (
    ESNConfig,
    ESNEngine,
    FeatureScaler,
    FrozenProjection,
    INPUT_COLUMNS,
    INPUT_GROUPS,
    MINUTE_NS,
    estimate_warmup,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_esn_architecture_audit_20261020.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUT = ROOT / CONFIG["output_directory"]
PREP = ROOT / CONFIG["reused_search_cache_directory"]
BASELINE_LOGITS = ROOT / CONFIG["reused_baseline_logits_path"]
BASELINE_LOGITS_META = ROOT / CONFIG["reused_baseline_metadata_path"]
BASELINE_META = ROOT / CONFIG["reused_baseline_summary_path"]
FEATURE_SUMMARY = ROOT / CONFIG["baseline_feature_summary_path"]
REPORT_PATH = ROOT / CONFIG["report_path"]
TRIAL_EXPORT_DIR = ROOT / CONFIG["historical_trial_export_directory"]
FIT_END = pd.Timestamp(CONFIG["split"]["fit_core_end_exclusive_utc"])
CORRECTION_START = pd.Timestamp(CONFIG["split"]["correction_fit_start_utc"])
CORRECTION_END = pd.Timestamp(CONFIG["split"]["correction_fit_end_exclusive_utc"])
VALIDATION_START = pd.Timestamp(CONFIG["split"]["search_validation_start_utc"])
SEARCH_END = pd.Timestamp(CONFIG["split"]["search_validation_end_exclusive_utc"])
FIT_END_NS = int(FIT_END.value)
CORRECTION_START_NS = int(CORRECTION_START.value)
CORRECTION_END_NS = int(CORRECTION_END.value)
VALIDATION_START_NS = int(VALIDATION_START.value)
SEARCH_END_NS = int(SEARCH_END.value)
DECISION_PURGE_NS = int(pd.Timedelta(minutes=1).value)
LABEL_HORIZON_NS = int(pd.Timedelta(minutes=7).value)
SEEDS = tuple(int(value) for value in CONFIG["screening"]["reservoir_seeds"])
LGBM_SEEDS = tuple(int(value) for value in CONFIG["downstream"]["lightgbm_seeds"])
THREAD_LIMITER = threadpool_limits(limits=8)
RUN_STARTED = None
RUN_PRIOR_SECONDS = 0.0
RUN_IDENTITY = None
SCALER_CACHE: dict[tuple[str, ...], FeatureScaler] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2 ** attempt))


def _identity(prepared: dict) -> dict:
    return {
        "experiment_id": CONFIG["experiment_id"],
        "source_commit": CONFIG["source_commit"],
        "config_sha256": _sha256(CONFIG_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "engine_sha256": _sha256(ROOT / "features/esn_reservoir.py"),
        "prepared_search_identity": prepared["identity"],
        "baseline_logits_sha256": _sha256(BASELINE_LOGITS),
        "baseline_logits_metadata_sha256": _sha256(BASELINE_LOGITS_META),
        "search_validation_end_exclusive_utc": CONFIG["split"]["search_validation_end_exclusive_utc"],
    }


def _elapsed_seconds() -> float:
    return RUN_PRIOR_SECONDS + (0.0 if RUN_STARTED is None else time.monotonic() - RUN_STARTED)


def _runtime_ledger() -> dict:
    path = OUT / "runtime_ledger.json"
    if not path.exists():
        return {"identity": RUN_IDENTITY, "stages_seconds": {}, "trial_seconds": {}, "resume_count": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity") != RUN_IDENTITY:
        raise RuntimeError("runtime ledger identity differs from the frozen experiment")
    return payload


def _save_runtime_ledger(payload: dict) -> None:
    completed_stages = float(sum(payload.get("stages_seconds", {}).values()))
    current_elapsed = 0.0 if RUN_STARTED is None else _elapsed_seconds()
    payload["active_seconds"] = max(
        float(payload.get("active_seconds", 0.0)),
        completed_stages,
        current_elapsed,
    )
    _atomic_json(OUT / "runtime_ledger.json", payload)


def _record_stage_time(stage: str, seconds: float) -> None:
    ledger = _runtime_ledger()
    ledger.setdefault("stages_seconds", {})[stage] = float(
        ledger["stages_seconds"].get(stage, 0.0) + max(0.0, seconds)
    )
    _save_runtime_ledger(ledger)


def _record_trial_time(study_name: str, trial) -> None:
    if trial.duration is None:
        return
    ledger = _runtime_ledger()
    ledger.setdefault("trial_seconds", {})[f"{study_name}:{trial.number}"] = float(trial.duration.total_seconds())
    _save_runtime_ledger(ledger)


def _check_compute_cap() -> None:
    _save_runtime_ledger(_runtime_ledger())
    cap = float(CONFIG["search"]["whole_experiment_compute_limit_seconds"])
    if _elapsed_seconds() > cap:
        raise TimeoutError(f"frozen four-hour compute cap of {cap:.0f} seconds reached")


def _load_search_data(identity: dict) -> dict:
    prepared = json.loads((PREP / "prepared_manifest.json").read_text(encoding="utf-8"))
    cutoff = pd.Timestamp(prepared["materialization_cutoff_utc"])
    if cutoff != SEARCH_END.tz_convert("UTC"):
        raise RuntimeError("reused prepared data does not stop at the frozen search-validation cutoff")
    if int(prepared.get("holdout_rows_materialized", -1)) != 0:
        raise RuntimeError("search cache declares rows outside the permitted development stage")
    baseline_meta = json.loads(BASELINE_LOGITS_META.read_text(encoding="utf-8"))
    baseline_summary = json.loads(BASELINE_META.read_text(encoding="utf-8"))
    if baseline_meta != prepared["identity"]:
        raise RuntimeError("baseline prediction artifact is not aligned to the reused search cache")
    if baseline_summary.get("identity") != prepared["identity"]:
        raise RuntimeError("baseline training artifact has a different frozen data identity")
    if baseline_summary.get("lightgbm_seed") != CONFIG["baseline"]["raw_logit_lightgbm_seed"]:
        raise RuntimeError("baseline OOS logit seed differs from its frozen provenance")

    arrays = {
        name: np.load(PREP / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in (
            "raw_opened", "raw_x", "raw_valid", "base_opened", "base_decision_at",
            "label_available", "target", "decision", "base_x", "base_to_raw",
        )
    }
    arrays["baseline_logits"] = np.load(BASELINE_LOGITS, mmap_mode="r", allow_pickle=False)
    arrays["prepared"] = prepared
    arrays["identity"] = identity
    raw_opened = arrays["raw_opened"]
    base_opened = arrays["base_opened"]
    label_available = arrays["label_available"]
    if len(raw_opened) == 0 or int(raw_opened[-1]) + MINUTE_NS >= SEARCH_END_NS:
        raise RuntimeError("raw inputs include a close event at or after the exclusive search cutoff")
    if len(base_opened) == 0 or int(base_opened[-1]) >= SEARCH_END_NS:
        raise RuntimeError("search cache includes a target row at or after the exclusive cutoff")
    if int(label_available.max()) >= SEARCH_END_NS:
        raise RuntimeError("search cache includes a label unavailable at the exclusive cutoff")
    if arrays["baseline_logits"].shape != arrays["target"].shape:
        raise RuntimeError("baseline OOS logits do not align to search target rows")
    if arrays["base_x"].shape[1] != int(CONFIG["baseline"]["feature_count"]):
        raise RuntimeError("baseline feature matrix differs from the frozen BASE70 width")
    if not np.array_equal(raw_opened[arrays["base_to_raw"]], base_opened):
        raise RuntimeError("reused feature rows do not align exactly with the causal raw candle stream")
    if not np.array_equal(arrays["base_decision_at"], base_opened + MINUTE_NS):
        raise RuntimeError("search rows do not use the frozen Opened+1m nominal decision time")
    if not np.array_equal(label_available, base_opened + LABEL_HORIZON_NS):
        raise RuntimeError("search labels do not match the frozen Opened+7m availability time")
    return arrays


def _utc_ns(value) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("experiment timestamps must be timezone-aware")
    return int(timestamp.tz_convert("UTC").value)


def _scope_mask(opened_ns, label_available_ns) -> np.ndarray:
    opened = np.asarray(opened_ns, dtype=np.int64)
    available = np.asarray(label_available_ns, dtype=np.int64)
    if opened.shape != available.shape:
        raise ValueError("opened and label-availability arrays must align")
    return (opened < SEARCH_END_NS) & (available < SEARCH_END_NS)


def _fit_scaler(config: ESNConfig, arrays: dict) -> FeatureScaler:
    columns = config.selected_columns
    cached = SCALER_CACHE.get(columns)
    if cached is not None:
        return cached
    selected = np.asarray([INPUT_COLUMNS.index(name) for name in columns], dtype=np.int64)
    raw_opened = arrays["raw_opened"]
    raw_valid = arrays["raw_valid"]
    raw_x = arrays["raw_x"]
    core_rows = np.flatnonzero((raw_opened < FIT_END_NS) & raw_valid)
    training_values = np.asarray(raw_x[core_rows, :][:, selected], dtype=np.float64)
    scaler = FeatureScaler(columns).fit(training_values)
    SCALER_CACHE[columns] = scaler
    return scaler


def _ready_mask(timestamps_ns, valid, warmup_steps: int) -> np.ndarray:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    ready = np.zeros(len(timestamps), dtype=bool)
    if not len(timestamps):
        return ready
    consecutive = np.diff(timestamps) == MINUTE_NS
    starts = np.flatnonzero(valid & np.r_[True, (~valid[:-1] | ~consecutive)])
    ends = np.flatnonzero(valid & np.r_[~consecutive | ~valid[1:], True]) + 1
    for start, end in zip(starts, ends):
        ready[start:end] = np.arange(1, end - start + 1) > int(warmup_steps)
    return ready


def _config_record(config: ESNConfig) -> dict:
    count = int(config.state_count)
    components = []
    for index, size in enumerate(config.unit_sizes):
        components.append({
            "index": index,
            "source_component": int(config.source_components[index]),
            "units": int(size),
            "topology": config.topologies[index],
            "leak_rate": float(config.leak_rates[index]),
            "spectral_radius": float(config.spectral_radii[index]),
            "input_scaling": float(config.input_scalings[index]),
            "bias_scaling": float(config.bias_scalings[index]),
            "sparse_degree": int(config.sparse_degrees[index]) if config.topologies[index] in {"sparse", "block_sparse"} else None,
            "input_groups": list(config.input_groups_by_component[index]),
            "input_columns": list(config.component_columns(index)),
        })
    return {
        "architecture": config.architecture,
        "topology": config.topology,
        "total_units": int(config.total_units),
        "n_reservoirs": count,
        "layers": count if config.architecture == "deep" else None,
        "branch_count": int(config.branch_count),
        "layer_count": int(config.layer_count),
        "reservoir_size_per_layer_or_reservoir": int(config.reservoir_size),
        "leak_rate": float(config.leak_rate),
        "spectral_radius": float(config.spectral_radius),
        "input_scaling": float(config.input_scaling),
        "bias_scaling": float(config.bias_scaling),
        "connectivity": float(config.connectivity) if config.topology == "sparse" else None,
        "input_groups": list(config.input_groups),
        "selected_columns": list(config.selected_columns),
        "components": components,
        "component_sizes": list(config.unit_sizes),
        "component_input_groups": [list(groups) for groups in config.input_groups_by_component],
        "component_topologies": list(config.topologies),
        "component_leak_rates": list(config.leak_rates),
        "component_spectral_radii": list(config.spectral_radii),
        "component_input_scalings": list(config.input_scalings),
        "component_bias_scalings": list(config.bias_scalings),
        "component_sparse_degrees": list(config.sparse_degrees),
    }


def _config_from_record(record: dict) -> ESNConfig:
    components = record.get("components") or []
    return ESNConfig(
        architecture=record["architecture"],
        topology=record["topology"],
        reservoir_size=int(record["reservoir_size_per_layer_or_reservoir"]),
        reservoir_count=int(record["n_reservoirs"]),
        leak_rate=float(record["leak_rate"]),
        spectral_radius=float(record["spectral_radius"]),
        input_scaling=float(record["input_scaling"]),
        bias_scaling=float(record["bias_scaling"]),
        connectivity=0.15 if record.get("connectivity") is None else float(record["connectivity"]),
        input_groups=tuple(record["input_groups"]),
        component_sizes=tuple(int(value) for value in record.get("component_sizes", ())) or None,
        component_input_groups=tuple(tuple(groups) for groups in record.get("component_input_groups", ())) or None,
        component_topologies=tuple(record.get("component_topologies", ())) or None,
        component_leak_rates=tuple(float(value) for value in record.get("component_leak_rates", ())) or None,
        component_spectral_radii=tuple(float(value) for value in record.get("component_spectral_radii", ())) or None,
        component_input_scalings=tuple(float(value) for value in record.get("component_input_scalings", ())) or None,
        component_bias_scalings=tuple(float(value) for value in record.get("component_bias_scalings", ())) or None,
        component_sparse_degrees=tuple(int(value) for value in record.get("component_sparse_degrees", ())) or None,
        branch_count=int(record.get("branch_count", 1)),
        layer_count=int(record.get("layer_count", 1)),
        sparse_degree=(None if not components else next((item["sparse_degree"] for item in components if item.get("sparse_degree") is not None), None)),
    )


def _trial_config(trial, stage: str, input_bases: dict[str, dict] | None = None) -> ESNConfig:
    if stage in {"inputs", "input_search", "input_group_correction"}:
        slot = trial.suggest_categorical("family_slot", sorted(input_bases))
        base = _config_from_record(input_bases[slot])
        groups = [
            group for group in CONFIG["screening"]["input_groups"]
            if trial.suggest_categorical(f"group_{group}", [False, True])
        ]
        if not groups:
            groups = [trial.suggest_categorical("fallback_group", CONFIG["screening"]["input_groups"])]
        return ESNConfig(
            architecture=base.architecture,
            topology=base.topology,
            reservoir_size=base.reservoir_size,
            reservoir_count=base.reservoir_count,
            leak_rate=base.leak_rate,
            spectral_radius=base.spectral_radius,
            input_scaling=base.input_scaling,
            bias_scaling=base.bias_scaling,
            connectivity=base.connectivity,
            input_groups=tuple(groups),
        )

    architecture = trial.suggest_categorical("architecture", CONFIG["search"]["architectures"])
    topology = trial.suggest_categorical("topology", CONFIG["search"]["topologies"])
    total_units = int(trial.suggest_categorical("total_units", CONFIG["search"]["total_unit_budgets"]))
    if architecture == "single":
        count = 1
    elif architecture == "parallel":
        count = int(trial.suggest_categorical("n_reservoirs", CONFIG["search"]["parallel_reservoir_count_choices"]))
    else:
        count = int(trial.suggest_categorical("layers", CONFIG["search"]["deep_layer_count_choices"]))
    leak_rate = trial.suggest_float("leak_rate", *CONFIG["search"]["leak_rate_range"])
    spectral_radius = trial.suggest_float("spectral_radius", *CONFIG["search"]["spectral_radius_range"])
    input_scaling = trial.suggest_float("input_scaling", *CONFIG["search"]["input_scaling_range"])
    bias_scaling = trial.suggest_float("bias_scaling", *CONFIG["search"]["bias_scaling_range"])
    connectivity = (
        trial.suggest_float("connectivity", *CONFIG["search"]["sparse_connectivity_range"])
        if topology == "sparse" else 0.15
    )
    if total_units % count:
        raise ValueError("total unit budget must divide evenly among layers or reservoirs")
    return ESNConfig(
        architecture=architecture,
        topology=topology,
        reservoir_size=total_units // count,
        reservoir_count=count,
        leak_rate=leak_rate,
        spectral_radius=spectral_radius,
        input_scaling=input_scaling,
        bias_scaling=bias_scaling,
        connectivity=connectivity,
        input_groups=tuple(CONFIG["screening"]["common_input_groups"]),
    )


def _config_key(config: ESNConfig) -> str:
    raw = json.dumps(_config_record(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _anchor_dynamics() -> dict:
    return CONFIG["search"]["common_anchor"]


def _architecture_anchors() -> list[tuple[str, dict, dict]]:
    anchor = _anchor_dynamics()
    result = []
    for architecture in CONFIG["search"]["architectures"]:
        for topology in CONFIG["search"]["topologies"]:
            params = {
                "architecture": architecture,
                "topology": topology,
                "total_units": int(anchor["total_units"]),
                "leak_rate": float(anchor["leak_rate"]),
                "spectral_radius": float(anchor["spectral_radius"]),
                "input_scaling": float(anchor["input_scaling"]),
                "bias_scaling": float(anchor["bias_scaling"]),
            }
            if architecture == "parallel":
                params["n_reservoirs"] = int(anchor["parallel_reservoir_count"])
            elif architecture == "deep":
                params["layers"] = int(anchor["deep_layer_count"])
            if topology == "sparse":
                params["connectivity"] = float(anchor["sparse_connectivity"])
            anchor_id = f"{architecture}_{topology}_common_inputs"
            result.append((anchor_id, params, {"anchor_id": anchor_id}))
    return result


def _input_anchors(input_bases: dict[str, dict]) -> list[tuple[str, dict, dict]]:
    result = []
    for slot in sorted(input_bases):
        for group in CONFIG["screening"]["input_groups"]:
            params = {"family_slot": slot}
            params.update({f"group_{name}": name == group for name in CONFIG["screening"]["input_groups"]})
            anchor_id = f"{slot}_{group}"
            result.append((anchor_id, params, {"anchor_id": anchor_id}))
    return result


def _stream_screen_features(config: ESNConfig, seed: int, arrays: dict, *, capture_probe: bool = True) -> dict:
    started = time.monotonic()
    key = f"{_config_key(config)}_seed_{seed}"
    feature_path = OUT / "screen_features" / f"{key}.npz"
    feature_meta_path = feature_path.with_suffix(".json")
    bundle_path = OUT / "bundles" / f"{key}.json"
    expected_meta = {
        "identity": RUN_IDENTITY,
        "config": _config_record(config),
        "seed": int(seed),
        "capture_probe": bool(capture_probe),
    }
    if feature_path.exists() and feature_meta_path.exists() and bundle_path.exists():
        if json.loads(feature_meta_path.read_text(encoding="utf-8")) != expected_meta:
            raise RuntimeError("screen-feature cache identity mismatch")
        with np.load(feature_path, allow_pickle=False) as cached:
            loaded = {name: cached[name] for name in cached.files}
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle.get("identity") != RUN_IDENTITY or bundle.get("config") != _config_record(config) or int(bundle["seed"]) != int(seed):
            raise RuntimeError("screen bundle identity mismatch")
        loaded.update(_bundle_objects(bundle))
        loaded["generation_seconds"] = 0.0
        return loaded

    scaler = _fit_scaler(config, arrays)
    raw_opened = arrays["raw_opened"]
    raw_x = arrays["raw_x"]
    raw_valid = arrays["raw_valid"]
    stream_cutoff = SEARCH_END_NS - MINUTE_NS
    stream_rows = int(np.searchsorted(raw_opened, stream_cutoff, side="left"))
    selected = np.asarray([INPUT_COLUMNS.index(name) for name in config.selected_columns], dtype=np.int64)
    warmup_count = min(stream_rows, 200_000)
    warmup_values = np.asarray(raw_x[:warmup_count, :][:, selected], dtype=np.float64)
    warmup_times = pd.to_datetime(raw_opened[:warmup_count], unit="ns", utc=True)
    warmup_steps, warmup_diagnostics = estimate_warmup(
        warmup_values, config, seed,
        relative_tolerance=CONFIG["screening"]["warmup_relative_initial_state_tolerance"],
        stable_window=CONFIG["screening"]["warmup_stable_window_minutes"],
        scaler=scaler,
        timestamps=warmup_times,
        valid_mask=np.asarray(raw_valid[:warmup_count], dtype=bool),
        max_steps=CONFIG["screening"]["warmup_max_steps_examined"],
    )
    if not warmup_diagnostics.get("stabilized", False):
        raise ValueError("training-only initial-state washout did not stabilize within its frozen budget")
    raw_ready = _ready_mask(raw_opened[:stream_rows], raw_valid[:stream_rows], warmup_steps)
    fit_ready = np.flatnonzero((raw_opened[:stream_rows] < FIT_END_NS) & raw_ready)
    sample_count = int(CONFIG["screening"]["pca_fit_rows"])
    if len(fit_ready) < sample_count:
        raise ValueError("dynamic warm-up leaves too few fit-core states for the frozen PCA sample")
    sample_indices = fit_ready[np.linspace(0, len(fit_ready) - 1, sample_count, dtype=np.int64)]

    base_opened = arrays["base_opened"]
    decision = arrays["decision"]
    if capture_probe:
        correction_idx = np.flatnonzero(
            (base_opened >= CORRECTION_START_NS) & (base_opened < CORRECTION_END_NS) & decision
        )
        validation_idx = np.flatnonzero(
            (base_opened >= VALIDATION_START_NS) & (base_opened < SEARCH_END_NS) & decision
        )
        capture_base_idx = np.r_[correction_idx, validation_idx]
        capture_raw_idx = np.asarray(arrays["base_to_raw"][capture_base_idx], dtype=np.int64)
        if np.any(np.diff(capture_raw_idx) <= 0) or int(capture_raw_idx[-1]) >= stream_rows:
            raise RuntimeError("frozen correction and validation rows do not align to the permitted raw prefix")
        if not _scope_mask(base_opened[capture_base_idx], arrays["label_available"][capture_base_idx]).all():
            raise RuntimeError("screen candidate contains a row or label beyond search validation")
    else:
        correction_idx = validation_idx = capture_raw_idx = np.empty(0, dtype=np.int64)

    work = OUT / "work" / key
    work.mkdir(parents=True, exist_ok=True)
    progress_path = work / "progress.json"
    checkpoint_path = work / "engine_checkpoint.json"
    unit_count = config.total_units
    sample_path = work / "pca_sample.dat"
    capture_path = work / "capture_states.dat"
    ready_path = work / "capture_ready.dat"
    mode = "r+" if progress_path.exists() else "w+"
    sample_states = np.memmap(sample_path, mode=mode, dtype=np.float32, shape=(sample_count, unit_count))
    capture_states = (
        np.memmap(capture_path, mode=mode, dtype=np.float32, shape=(len(capture_raw_idx), unit_count))
        if capture_probe else None
    )
    capture_ready = (
        np.memmap(ready_path, mode=mode, dtype=np.bool_, shape=(len(capture_raw_idx),))
        if capture_probe else None
    )
    engine = ESNEngine(config, seed, scaler, warmup_steps=warmup_steps)
    start_row = sample_cursor = capture_cursor = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected_progress = {
            **expected_meta,
            "sample_indices_sha256": hashlib.sha256(sample_indices.tobytes()).hexdigest(),
            "capture_indices_sha256": hashlib.sha256(capture_raw_idx.tobytes()).hexdigest(),
        }
        if any(progress.get(name) != value for name, value in expected_progress.items()):
            raise RuntimeError("state-generation resume identity differs; refusing to reuse the checkpoint")
        engine = ESNEngine.load_checkpoint(checkpoint_path, expected_config=config, expected_seed=seed)
        start_row = int(progress["next_row"])
        sample_cursor = int(progress["sample_cursor"])
        capture_cursor = int(progress["capture_cursor"])
    values = raw_x[:stream_rows]
    for start in range(start_row, stream_rows, 65_536):
        end = min(stream_rows, start + 65_536)
        timestamps = pd.to_datetime(raw_opened[start:end], unit="ns", utc=True)
        observations = np.asarray(values[start:end, :][:, selected], dtype=np.float32)
        states, ready = engine.update_many(
            timestamps, observations, valid_mask=np.asarray(raw_valid[start:end], dtype=bool),
        )
        left = int(np.searchsorted(sample_indices, start, side="left"))
        right = int(np.searchsorted(sample_indices, end, side="left"))
        if right > left:
            locations = sample_indices[left:right] - start
            if not ready[locations].all():
                raise RuntimeError("a fit-core PCA sample precedes the measured washout")
            sample_states[sample_cursor:sample_cursor + len(locations)] = states[locations]
            sample_cursor += len(locations)
        if capture_probe:
            left = int(np.searchsorted(capture_raw_idx, start, side="left"))
            right = int(np.searchsorted(capture_raw_idx, end, side="left"))
            if right > left:
                locations = capture_raw_idx[left:right] - start
                capture_states[capture_cursor:capture_cursor + len(locations)] = states[locations]
                capture_ready[capture_cursor:capture_cursor + len(locations)] = ready[locations]
                capture_cursor += len(locations)
        if end % 262_144 == 0 or end == stream_rows:
            sample_states.flush()
            if capture_probe:
                capture_states.flush()
                capture_ready.flush()
            engine.save_checkpoint(checkpoint_path)
            _atomic_json(progress_path, {
                **expected_meta,
                "sample_indices_sha256": hashlib.sha256(sample_indices.tobytes()).hexdigest(),
                "capture_indices_sha256": hashlib.sha256(capture_raw_idx.tobytes()).hexdigest(),
                "next_row": end,
                "sample_cursor": sample_cursor,
                "capture_cursor": capture_cursor,
            })
        _check_compute_cap()
    sample_states.flush()
    if capture_probe:
        capture_states.flush()
        capture_ready.flush()
    if sample_cursor != sample_count or (capture_probe and capture_cursor != len(capture_raw_idx)):
        raise RuntimeError("screen state capture did not reach the frozen row counts")
    saturation = float(np.mean(np.abs(sample_states) >= 0.98))
    if saturation > CONFIG["screening"]["reject_saturation_fraction_above"]:
        raise ValueError(f"reservoir saturation {saturation:.3f} exceeds the frozen limit")

    projection_seed = int(seed + CONFIG["screening"]["projection_seed_offset"])
    pca = PCA(
        n_components=int(CONFIG["screening"]["state_components_total"]),
        svd_solver="randomized", random_state=projection_seed,
    )
    pca.fit(np.asarray(sample_states))
    projection = FrozenProjection(pca.mean_, pca.components_, pca.explained_variance_ratio_)
    correction_count = len(correction_idx)
    if capture_probe:
        correction_features = projection.transform(np.asarray(capture_states[:correction_count])).astype(np.float32)
        validation_features = projection.transform(np.asarray(capture_states[correction_count:])).astype(np.float32)
        ready_values = np.array(capture_ready, copy=True)
        correction_ready = ready_values[:correction_count]
        validation_ready = ready_values[correction_count:]
    else:
        correction_features = np.empty((0, int(CONFIG["screening"]["state_components_total"])), dtype=np.float32)
        validation_features = np.empty_like(correction_features)
        correction_ready = validation_ready = np.empty(0, dtype=bool)
    output = {
        "correction_features": correction_features,
        "validation_features": validation_features,
        "correction_ready": correction_ready,
        "validation_ready": validation_ready,
        "correction_indices": correction_idx,
        "validation_indices": validation_idx,
        "warmup_steps": int(warmup_steps),
        "warmup_diagnostics": warmup_diagnostics,
        "saturation_fraction": saturation,
        "actual_recurrent_edges": list(engine.actual_recurrent_edges),
        "projection_seed": projection_seed,
        "config": config,
        "seed": int(seed),
        "scaler": scaler,
        "projection": projection,
        "generation_seconds": float(time.monotonic() - started),
    }
    _write_screen_cache(key, expected_meta, output)
    del sample_states
    if capture_probe:
        del capture_states, capture_ready, ready_values
    shutil.rmtree(work)
    return output


def _bundle_objects(bundle: dict) -> dict:
    scaler = FeatureScaler.from_dict(bundle["scaler"])
    projection = FrozenProjection.from_dict(bundle["projection"])
    config = _config_from_record(bundle["config"])
    return {
        "config": config,
        "seed": int(bundle["seed"]),
        "scaler": scaler,
        "projection": projection,
        "warmup_steps": int(bundle["warmup_steps"]),
        "warmup_diagnostics": bundle["warmup_diagnostics"],
        "saturation_fraction": float(bundle["saturation_fraction"]),
        "actual_recurrent_edges": tuple(int(value) for value in bundle.get("actual_recurrent_edges", ())),
        "projection_seed": int(bundle["projection_seed"]),
    }


def _write_screen_cache(key: str, expected_meta: dict, output: dict) -> None:
    screen_dir = OUT / "screen_features"
    screen_dir.mkdir(parents=True, exist_ok=True)
    feature_path = screen_dir / f"{key}.npz"
    temporary = screen_dir / f"{key}.tmp.npz"
    np.savez(
        temporary,
        correction_features=output["correction_features"],
        validation_features=output["validation_features"],
        correction_ready=output["correction_ready"],
        validation_ready=output["validation_ready"],
        correction_indices=output["correction_indices"],
        validation_indices=output["validation_indices"],
    )
    os.replace(temporary, feature_path)
    _atomic_json(feature_path.with_suffix(".json"), expected_meta)
    bundle = {
        "identity": RUN_IDENTITY,
        "config": _config_record(output["config"]),
        "seed": int(output["seed"]),
        "scaler": output["scaler"].to_dict(),
        "projection": output["projection"].to_dict(),
        "warmup_steps": int(output["warmup_steps"]),
        "warmup_diagnostics": output["warmup_diagnostics"],
        "saturation_fraction": float(output["saturation_fraction"]),
        "actual_recurrent_edges": list(output["actual_recurrent_edges"]),
        "projection_seed": int(output["projection_seed"]),
        "feature_names": [
            f"esn_{_config_key(output['config'])}_seed_{output['seed']}_pc{index:02d}"
            for index in range(int(CONFIG["screening"]["state_components_total"]))
        ],
    }
    _atomic_json(OUT / "bundles" / f"{key}.json", bundle)


def _logistic_delta(y_fit, baseline_fit_logits, esn_fit, y_validation,
                    baseline_validation_logits, esn_validation) -> float:
    c_value = float(CONFIG["screening"]["calibration_regularization_C"])
    control = LogisticRegression(C=c_value, max_iter=500, solver="lbfgs", random_state=0)
    augmented = LogisticRegression(C=c_value, max_iter=500, solver="lbfgs", random_state=0)
    fit_base = np.asarray(baseline_fit_logits, dtype=np.float64).reshape(-1, 1)
    validation_base = np.asarray(baseline_validation_logits, dtype=np.float64).reshape(-1, 1)
    control.fit(fit_base, y_fit)
    augmented.fit(np.column_stack((fit_base, esn_fit)), y_fit)
    p_control = control.predict_proba(validation_base)[:, 1]
    p_augmented = augmented.predict_proba(np.column_stack((validation_base, esn_validation)))[:, 1]
    return float(log_loss(y_validation, p_control, labels=[0, 1]) - log_loss(y_validation, p_augmented, labels=[0, 1]))


def _score_candidate(config: ESNConfig, arrays: dict) -> dict:
    started = time.monotonic()
    raw_logits = arrays["baseline_logits"]
    labels = arrays["target"]
    label_available = arrays["label_available"]
    generated = []
    generation_seconds = []
    deltas = []
    fit_counts = []
    validation_counts = []
    for seed in SEEDS:
        _check_compute_cap()
        item = _stream_screen_features(config, seed, arrays)
        generated.append(item)
        generation_seconds.append(float(item["generation_seconds"]))
        corr_idx = item["correction_indices"]
        val_idx = item["validation_indices"]
        corr_ok = (
            np.isfinite(raw_logits[corr_idx])
            & (label_available[corr_idx] <= CORRECTION_END_NS + DECISION_PURGE_NS)
            & item["correction_ready"]
        )
        val_ok = np.isfinite(raw_logits[val_idx]) & item["validation_ready"]
        if corr_ok.sum() < 1000 or val_ok.sum() < 1000:
            raise ValueError("washout leaves too few common correction or search-validation rows")
        probe_start = time.monotonic()
        delta = _logistic_delta(
            labels[corr_idx[corr_ok]], raw_logits[corr_idx[corr_ok]], item["correction_features"][corr_ok],
            labels[val_idx[val_ok]], raw_logits[val_idx[val_ok]], item["validation_features"][val_ok],
        )
        deltas.append(delta)
        fit_counts.append(int(corr_ok.sum()))
        validation_counts.append(int(val_ok.sum()))
        item["probe_seconds"] = float(time.monotonic() - probe_start)
        _check_compute_cap()
    outcome = {
        "config": _config_record(config),
        "reservoir_seeds": list(SEEDS),
        "seed_deltas": deltas,
        "mean_delta_logloss": float(np.mean(deltas)),
        "std_delta_logloss": float(np.std(deltas, ddof=0)),
        "positive_seed_count": int(np.sum(np.asarray(deltas) > 0.0)),
        "passes_screen": bool(
            np.mean(deltas) > 0.0
            and np.sum(np.asarray(deltas) > 0.0) >= int(CONFIG["screening"]["minimum_positive_reservoir_seeds"])
        ),
        "correction_fit_rows_by_seed": fit_counts,
        "search_validation_rows_by_seed": validation_counts,
        "warmup_steps_by_seed": [int(item["warmup_steps"]) for item in generated],
        "warmup_diagnostics_by_seed": [item["warmup_diagnostics"] for item in generated],
        "saturation_fraction_by_seed": [float(item["saturation_fraction"]) for item in generated],
        "projection_seeds": [int(item["projection_seed"]) for item in generated],
        "state_generation_seconds_by_seed": generation_seconds,
        "probe_seconds_by_seed": [float(item["probe_seconds"]) for item in generated],
        "elapsed_seconds": float(time.monotonic() - started),
    }
    return outcome


def _study_records(study, stage: str) -> list[dict]:
    rows = []
    for trial in study.trials:
        config = trial.user_attrs.get("effective_config")
        outcome = trial.user_attrs.get("outcome")
        status = "completed_scored" if trial.state == optuna.trial.TrialState.COMPLETE and outcome else (
            "rejected" if trial.user_attrs.get("rejection_reason") else trial.state.name.lower()
        )
        rows.append({
            "stage": stage,
            "trial_number": int(trial.number),
            "state": trial.state.name,
            "status": status,
            "value": None if trial.value is None else float(trial.value),
            "duration_seconds": None if trial.duration is None else float(trial.duration.total_seconds()),
            "anchor_id": trial.user_attrs.get("anchor_id"),
            "resumed_from_trial": trial.user_attrs.get("resumed_from_trial"),
            "params": trial.params,
            "effective_config": config,
            "outcome": outcome,
            "rejection_reason": trial.user_attrs.get("rejection_reason"),
        })
    return rows


def _make_objective(stage: str, arrays: dict, input_bases: dict[str, dict] | None = None):
    def objective(trial):
        config = _trial_config(trial, stage, input_bases)
        trial.set_user_attr("effective_config", _config_record(config))
        if trial.user_attrs.get("anchor_id"):
            trial.set_user_attr("anchor_id", trial.user_attrs["anchor_id"])
        try:
            outcome = _score_candidate(config, arrays)
            trial.set_user_attr("outcome", outcome)
            return float(outcome["mean_delta_logloss"])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            trial.set_user_attr("rejection_reason", str(exc))
            return -1.0
    return objective


def _ensure_anchors(study, anchors: list[tuple[str, dict, dict]]) -> None:
    if not study.trials:
        for anchor_id, params, attrs in anchors:
            study.enqueue_trial(params, user_attrs={**attrs, "anchor_id": anchor_id})
        return
    for old in [trial for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING]:
        user_attrs = dict(old.user_attrs)
        user_attrs["resumed_from_trial"] = int(old.number)
        study.tell(old.number, state=optuna.trial.TrialState.FAIL)
        if old.params:
            study.enqueue_trial(old.params, user_attrs=user_attrs)
    trials = study.trials
    completed_anchors = {
        trial.user_attrs.get("anchor_id") for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.user_attrs.get("anchor_id")
    }
    waiting_anchors = {
        trial.user_attrs.get("anchor_id") for trial in trials
        if trial.state == optuna.trial.TrialState.WAITING and trial.user_attrs.get("anchor_id")
    }
    for anchor_id, params, attrs in anchors:
        if anchor_id not in completed_anchors and anchor_id not in waiting_anchors:
            study.enqueue_trial(params, user_attrs={**attrs, "anchor_id": anchor_id})


def _run_study(stage: str, study_name: str, sampler_seed: int, anchors: list[tuple[str, dict, dict]],
               time_limit: float, arrays: dict, input_bases: dict[str, dict] | None = None) -> dict:
    result_path = OUT / f"{stage}_search.json"
    input_bases_sha256 = None if input_bases is None else hashlib.sha256(
        json.dumps(input_bases, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            saved.get("identity") == RUN_IDENTITY
            and saved.get("status") == "complete"
            and saved.get("input_bases_sha256") == input_bases_sha256
        ):
            return saved
    database = OUT / "optuna.sqlite3"
    storage = f"sqlite:///{database.as_posix()}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(sampler_seed)),
        load_if_exists=True,
    )
    expected_identity = study.user_attrs.get("experiment_identity")
    if expected_identity is not None and expected_identity != RUN_IDENTITY:
        raise RuntimeError(f"Optuna study {study_name} belongs to another data/code/config identity")
    if expected_identity is None:
        study.set_user_attr("experiment_identity", RUN_IDENTITY)
    prior_bases_sha256 = study.user_attrs.get("input_bases_sha256")
    if prior_bases_sha256 is not None and prior_bases_sha256 != input_bases_sha256:
        raise RuntimeError("input-group search bases differ from the frozen architecture finalists")
    if prior_bases_sha256 is None and input_bases is not None:
        study.set_user_attr("input_bases_sha256", input_bases_sha256)
    _ensure_anchors(study, anchors)
    stage_started = time.monotonic()
    objective = _make_objective(stage, arrays, input_bases)
    remaining = float(time_limit)
    pending_anchors = sum(
        1 for trial in study.trials
        if trial.state == optuna.trial.TrialState.WAITING and trial.user_attrs.get("anchor_id")
    )
    if pending_anchors:
        study.optimize(
            objective, n_trials=int(pending_anchors), timeout=remaining,
            gc_after_trial=True, show_progress_bar=False,
            callbacks=[lambda _study, trial: _record_trial_time(study_name, trial)],
        )
    finished_anchor_ids = {
        trial.user_attrs.get("anchor_id") for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.user_attrs.get("anchor_id")
    }
    all_anchor_ids = {anchor_id for anchor_id, _, _ in anchors}
    anchors_complete = all_anchor_ids <= finished_anchor_ids
    remaining = max(0.0, float(time_limit) - (time.monotonic() - stage_started))
    if anchors_complete and remaining > 0.0:
        study.optimize(
            objective, n_trials=None, timeout=remaining,
            gc_after_trial=True, show_progress_bar=False,
            callbacks=[lambda _study, trial: _record_trial_time(study_name, trial)],
        )
    records = _study_records(study, stage)
    scored = [row for row in records if row["status"] == "completed_scored"]
    result = {
        "status": "complete" if anchors_complete else "anchors_incomplete",
        "stage": stage,
        "study_name": study_name,
        "input_bases_sha256": input_bases_sha256,
        "anchors_expected": len(anchors),
        "anchors_completed": len(finished_anchor_ids & all_anchor_ids),
        "anchors_complete": bool(anchors_complete),
        "attempted_trials": len(records),
        "completed_scored_trials": len(scored),
        "rejected_trials": sum(row["status"] == "rejected" for row in records),
        "elapsed_seconds_this_run": float(time.monotonic() - stage_started),
        "time_limit_seconds": float(time_limit),
        "trials": records,
    }
    _atomic_json(result_path, {"identity": RUN_IDENTITY, **result})
    return result


def _export_historical_trials() -> dict:
    source = ROOT / "data/analysis/polymarket/BTC/btc_esn_portfolio_20261020_r3/optuna.sqlite3"
    study = optuna.load_study(
        study_name="btc_esn_portfolio_20261020",
        storage=f"sqlite:///{source.as_posix()}",
    )
    rows = []
    for trial in study.trials:
        outcome = trial.user_attrs.get("outcome") or {}
        config = trial.user_attrs.get("config") or {}
        rows.append({
            "trial_number": int(trial.number),
            "optuna_state": trial.state.name,
            "status": "rejected" if trial.user_attrs.get("rejected") else "completed_scored",
            "score": None if trial.value is None else float(trial.value),
            "duration_seconds": None if trial.duration is None else float(trial.duration.total_seconds()),
            "effective_config": config,
            "parameters": trial.params,
            "seed_deltas": outcome.get("seed_deltas"),
            "mean_delta_logloss": outcome.get("mean_delta_logloss"),
            "std_delta_logloss": outcome.get("std_delta_logloss"),
            "positive_seed_count": outcome.get("positive_seed_count"),
            "passes_screen": outcome.get("passes_screen"),
            "warmup_steps_by_seed": outcome.get("warmup_steps_by_seed"),
            "warmup_diagnostics_by_seed": outcome.get("warmup_diagnostics_by_seed"),
            "saturation_fraction_by_seed": outcome.get("saturation_fraction_by_seed"),
            "outcome": outcome,
            "rejection_reason": trial.user_attrs.get("rejected"),
        })
    rows.sort(key=lambda item: item["trial_number"])
    if len(rows) != 48:
        raise RuntimeError(f"expected exactly 48 historical trials, found {len(rows)}")
    TRIAL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = TRIAL_EXPORT_DIR / "initial_48_trials.json"
    csv_path = TRIAL_EXPORT_DIR / "initial_48_trials.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    fields = (
        "trial_number", "optuna_state", "status", "score", "duration_seconds",
        "architecture", "topology", "reservoir_count", "reservoir_size", "total_units",
        "input_groups", "selected_columns", "leak_rate", "spectral_radius",
        "input_scaling", "bias_scaling", "connectivity", "seed_deltas",
        "mean_delta_logloss", "std_delta_logloss", "positive_seed_count",
        "passes_screen", "rejection_reason", "full_effective_config_json", "full_outcome_json",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            config = row["effective_config"]
            writer.writerow({
                "trial_number": row["trial_number"],
                "optuna_state": row["optuna_state"],
                "status": row["status"],
                "score": row["score"],
                "duration_seconds": row["duration_seconds"],
                "architecture": config.get("architecture"),
                "topology": config.get("topology"),
                "reservoir_count": config.get("reservoir_count"),
                "reservoir_size": config.get("reservoir_size"),
                "total_units": config.get("reservoir_count", 1) * config.get("reservoir_size", 0),
                "input_groups": json.dumps(config.get("input_groups"), ensure_ascii=False),
                "selected_columns": json.dumps(config.get("selected_columns"), ensure_ascii=False),
                "leak_rate": config.get("leak_rate"),
                "spectral_radius": config.get("spectral_radius"),
                "input_scaling": config.get("input_scaling"),
                "bias_scaling": config.get("bias_scaling"),
                "connectivity": config.get("connectivity"),
                "seed_deltas": json.dumps(row["seed_deltas"]),
                "mean_delta_logloss": row["mean_delta_logloss"],
                "std_delta_logloss": row["std_delta_logloss"],
                "positive_seed_count": row["positive_seed_count"],
                "passes_screen": row["passes_screen"],
                "rejection_reason": row["rejection_reason"],
                "full_effective_config_json": json.dumps(config, sort_keys=True),
                "full_outcome_json": json.dumps(row["outcome"], sort_keys=True),
            })
    counts = {}
    for row in rows:
        family = row["effective_config"].get("architecture", "unknown")
        counts.setdefault(family, {"attempted": 0, "scored": 0, "rejected": 0})
        counts[family]["attempted"] += 1
        counts[family]["rejected" if row["status"] == "rejected" else "scored"] += 1
    return {
        "trial_count": len(rows),
        "family_counts": counts,
        "rejected_trials": [row["trial_number"] for row in rows if row["status"] == "rejected"],
        "csv_path": csv_path.relative_to(ROOT).as_posix(),
        "json_path": json_path.relative_to(ROOT).as_posix(),
    }


def _scored_trials(search: dict) -> list[dict]:
    return [
        row for row in search["trials"]
        if row["status"] == "completed_scored" and row.get("effective_config") and row.get("outcome")
    ]


def _family_leaders(architecture_search: dict, input_search: dict) -> tuple[dict, dict, list[dict]]:
    common = {}
    for row in _scored_trials(architecture_search):
        config = row["effective_config"]
        key = (config["architecture"], config["topology"])
        current = common.get(key)
        if current is None or row["outcome"]["mean_delta_logloss"] > current["outcome"]["mean_delta_logloss"]:
            common[key] = row
    input_best = {}
    for row in _scored_trials(input_search):
        architecture = row["effective_config"]["architecture"]
        current = input_best.get(architecture)
        if current is None or row["outcome"]["mean_delta_logloss"] > current["outcome"]["mean_delta_logloss"]:
            input_best[architecture] = row
    finalists = []
    seen = set()
    for row in list(common.values()) + list(input_best.values()):
        config = _config_from_record(row["effective_config"])
        key = _config_key(config)
        if key in seen:
            continue
        seen.add(key)
        finalists.append({
            "candidate_id": key,
            "source_stage": row["stage"],
            "source_trial_number": row["trial_number"],
            "effective_config": _config_record(config),
            "screen_outcome": row["outcome"],
        })
    return common, input_best, finalists


def _load_screen_cache(config: ESNConfig, seed: int, arrays: dict, *, capture_probe: bool = True) -> dict:
    key = f"{_config_key(config)}_seed_{seed}"
    feature_path = OUT / "screen_features" / f"{key}.npz"
    feature_meta_path = feature_path.with_suffix(".json")
    bundle_path = OUT / "bundles" / f"{key}.json"
    expected_meta = {
        "identity": RUN_IDENTITY,
        "config": _config_record(config),
        "seed": int(seed),
        "capture_probe": bool(capture_probe),
    }
    if not (feature_path.exists() and feature_meta_path.exists() and bundle_path.exists()):
        return _stream_screen_features(config, seed, arrays, capture_probe=capture_probe)
    if json.loads(feature_meta_path.read_text(encoding="utf-8")) != expected_meta:
        raise RuntimeError("screen-feature cache belongs to another experiment identity")
    with np.load(feature_path, allow_pickle=False) as cached:
        loaded = {name: cached[name] for name in cached.files}
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("identity") != RUN_IDENTITY or bundle.get("config") != _config_record(config) or int(bundle["seed"]) != int(seed):
        raise RuntimeError("screen bundle belongs to another experiment identity")
    loaded.update(_bundle_objects(bundle))
    loaded["generation_seconds"] = 0.0
    return loaded


def _portfolio_score(configs: list[ESNConfig], generated: dict, arrays: dict) -> dict:
    if not configs:
        return {"mean_delta_logloss": 0.0, "seed_deltas": [0.0] * len(SEEDS), "positive_seed_count": 0}
    raw_logits = arrays["baseline_logits"]
    labels = arrays["target"]
    availability = arrays["label_available"]
    per_seed = []
    for seed in SEEDS:
        _check_compute_cap()
        items = [generated[_config_key(config)][seed] for config in configs]
        correction_idx = items[0]["correction_indices"]
        validation_idx = items[0]["validation_indices"]
        correction_ready = np.logical_and.reduce([item["correction_ready"] for item in items])
        validation_ready = np.logical_and.reduce([item["validation_ready"] for item in items])
        correction_ok = (
            np.isfinite(raw_logits[correction_idx])
            & (availability[correction_idx] <= CORRECTION_END_NS + DECISION_PURGE_NS)
            & correction_ready
        )
        validation_ok = np.isfinite(raw_logits[validation_idx]) & validation_ready
        if correction_ok.sum() < 1000 or validation_ok.sum() < 1000:
            raise ValueError("portfolio warm-up leaves too few common search rows")
        correction_features = np.column_stack([item["correction_features"] for item in items])
        validation_features = np.column_stack([item["validation_features"] for item in items])
        started = time.monotonic()
        delta = _logistic_delta(
            labels[correction_idx[correction_ok]], raw_logits[correction_idx[correction_ok]], correction_features[correction_ok],
            labels[validation_idx[validation_ok]], raw_logits[validation_idx[validation_ok]], validation_features[validation_ok],
        )
        per_seed.append({
            "reservoir_seed": seed,
            "delta_logloss": delta,
            "correction_rows": int(correction_ok.sum()),
            "validation_rows": int(validation_ok.sum()),
            "probe_seconds": float(time.monotonic() - started),
        })
        _check_compute_cap()
    deltas = [row["delta_logloss"] for row in per_seed]
    return {
        "member_count": len(configs),
        "seed_deltas": deltas,
        "mean_delta_logloss": float(np.mean(deltas)),
        "std_delta_logloss": float(np.std(deltas, ddof=0)),
        "positive_seed_count": int(np.sum(np.asarray(deltas) > 0.0)),
        "per_seed": per_seed,
    }


def _select_portfolio(finalists: list[dict], arrays: dict) -> dict:
    result_path = OUT / "portfolio_selection.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("identity") == RUN_IDENTITY and saved.get("status") == "portfolio_search_validation_complete":
            return saved
    started = time.monotonic()
    configs = {
        item["candidate_id"]: _config_from_record(item["effective_config"])
        for item in finalists
    }
    generated = {}
    generation_seconds = {}
    for key, config in configs.items():
        generated[key] = {}
        generation_seconds[key] = 0.0
        for seed in SEEDS:
            screen = _load_screen_cache(config, seed, arrays)
            generated[key][seed] = screen
            generation_seconds[key] += float(screen.get("generation_seconds", 0.0))
    individual = {key: _portfolio_score([config], generated, arrays) for key, config in configs.items()}
    trace = []
    minimum_positive = int(CONFIG["screening"]["minimum_positive_reservoir_seeds"])
    passing = [
        key for key, score in individual.items()
        if score["mean_delta_logloss"] > 0.0 and score["positive_seed_count"] >= minimum_positive
    ]
    selected = []
    current_score = {"mean_delta_logloss": 0.0, "seed_deltas": [0.0] * len(SEEDS), "positive_seed_count": 0}
    if passing:
        first = max(passing, key=lambda key: individual[key]["mean_delta_logloss"])
        for key, config in configs.items():
            score = individual[key]
            record = _config_record(config)
            trace.append({
                "current_set": [],
                "added_candidate": key,
                "candidate_config": record,
                "score_before": 0.0,
                "score_after": score["mean_delta_logloss"],
                "marginal_delta_logloss": score["mean_delta_logloss"],
                "candidate_total_units": record["total_units"],
                "generation_seconds": generation_seconds[key],
                "probe_seconds": float(sum(row["probe_seconds"] for row in score.get("per_seed", []))),
                "decision": "selected_initial_candidate" if key == first else (
                    "not_selected_lower_score" if key in passing else "rejected_single_candidate_rule"
                ),
                "seed_deltas": score["seed_deltas"],
            })
        selected = [first]
        current_score = individual[first]

    while selected and len(selected) < int(CONFIG["search"]["portfolio_maximum_members"]):
        before = float(current_score["mean_delta_logloss"])
        tested = []
        for key, config in configs.items():
            if key in selected:
                continue
            set_configs = [configs[member] for member in selected] + [config]
            score = _portfolio_score(set_configs, generated, arrays)
            marginal = float(score["mean_delta_logloss"] - before)
            qualifies = (
                score["mean_delta_logloss"] > 0.0
                and score["positive_seed_count"] >= minimum_positive
                and marginal > 0.0
            )
            probe_seconds = float(sum(row["probe_seconds"] for row in score.get("per_seed", [])))
            tested.append((key, score, marginal, qualifies, probe_seconds))
        if not tested:
            break
        eligible = [item for item in tested if item[3]]
        best_key = max(eligible, key=lambda item: item[1]["mean_delta_logloss"])[0] if eligible else None
        for key, score, marginal, qualifies, probe_seconds in tested:
            record = _config_record(configs[key])
            trace.append({
                "current_set": list(selected),
                "added_candidate": key,
                "candidate_config": record,
                "score_before": before,
                "score_after": score["mean_delta_logloss"],
                "marginal_delta_logloss": marginal,
                "candidate_total_units": record["total_units"],
                "combined_total_units_after_add": sum(_config_record(configs[item])["total_units"] for item in selected) + record["total_units"],
                "generation_seconds": generation_seconds[key],
                "probe_seconds": probe_seconds,
                "decision": "selected_best_positive_marginal" if key == best_key else (
                    "positive_but_not_best" if qualifies else "rejected_nonpositive_marginal_or_seed_rule"
                ),
                "seed_deltas": score["seed_deltas"],
            })
        if best_key is None:
            break
        selected.append(best_key)
        current_score = next(item[1] for item in tested if item[0] == best_key)

    payload = {
        "status": "portfolio_search_validation_complete",
        "candidate_count": len(configs),
        "candidate_scores": individual,
        "selected_candidate_ids": selected,
        "selected_configs": [_config_record(configs[key]) for key in selected],
        "portfolio_score": current_score,
        "selection_rule": "greedy addition by incremental full-set logistic-correction delta_logloss; accept only a positive marginal and a positive combined mean on at least two common reservoir seeds",
        "selection_trace": trace,
        "state_generation_seconds": float(sum(generation_seconds.values())),
        "probe_seconds": float(sum(item["probe_seconds"] for item in trace)),
        "elapsed_seconds": float(time.monotonic() - started),
    }
    _atomic_json(OUT / "portfolio_selection.json", {"identity": RUN_IDENTITY, **payload})
    return payload


def _bundle_for(config: ESNConfig, seed: int) -> dict:
    key = f"{_config_key(config)}_seed_{seed}"
    path = OUT / "bundles" / f"{key}.json"
    if not path.exists():
        _stream_screen_features(config, seed, _load_arrays_for_screen(), capture_probe=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity") != RUN_IDENTITY or payload.get("config") != _config_record(config) or int(payload["seed"]) != int(seed):
        raise RuntimeError("finalist feature bundle identity mismatch")
    return _bundle_objects(payload)


def _generate_full_features(config: ESNConfig, seed: int, arrays: dict) -> np.ndarray:
    key = f"{_config_key(config)}_seed_{seed}"
    output_path = OUT / "final_features" / f"{key}.npy"
    meta_path = output_path.with_suffix(".npy.json")
    bundle_path = OUT / "bundles" / f"{key}.json"
    if not bundle_path.exists():
        _stream_screen_features(config, seed, arrays, capture_probe=False)
    expected_meta = {
        "identity": RUN_IDENTITY,
        "config": _config_record(config),
        "seed": int(seed),
        "bundle_sha256": _sha256(bundle_path),
    }
    if output_path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != expected_meta:
            raise RuntimeError("finalist feature matrix identity mismatch")
        return np.load(output_path, mmap_mode="r", allow_pickle=False)

    bundle = _bundle_for(config, seed)
    raw_opened = arrays["raw_opened"]
    raw_x = arrays["raw_x"]
    raw_valid = arrays["raw_valid"]
    base_to_raw = arrays["base_to_raw"]
    stream_rows = int(np.searchsorted(raw_opened, SEARCH_END_NS - MINUTE_NS, side="left"))
    if int(base_to_raw[-1]) >= stream_rows:
        raise RuntimeError("finalist features extend outside the frozen search prefix")
    selected = np.asarray([INPUT_COLUMNS.index(name) for name in config.selected_columns], dtype=np.int64)
    width = int(CONFIG["screening"]["state_components_total"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(".progress.json")
    checkpoint_path = output_path.with_suffix(".engine.json")
    if output_path.exists():
        output = np.lib.format.open_memmap(output_path, mode="r+")
    else:
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32, shape=(len(base_to_raw), width),
        )
        output[:] = np.nan
        output.flush()
    engine = ESNEngine(config, seed, bundle["scaler"], warmup_steps=bundle["warmup_steps"])
    start_row = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if any(progress.get(name) != value for name, value in expected_meta.items()):
            raise RuntimeError("finalist feature checkpoint identity mismatch")
        engine = ESNEngine.load_checkpoint(checkpoint_path, expected_config=config, expected_seed=seed)
        start_row = int(progress["next_row"])
    else:
        output[:] = np.nan
        output.flush()
    projection: FrozenProjection = bundle["projection"]
    for start in range(start_row, stream_rows, 65_536):
        end = min(stream_rows, start + 65_536)
        timestamps = pd.to_datetime(raw_opened[start:end], unit="ns", utc=True)
        observations = np.asarray(raw_x[start:end, :][:, selected], dtype=np.float32)
        states, ready = engine.update_many(
            timestamps, observations, valid_mask=np.asarray(raw_valid[start:end], dtype=bool),
        )
        left = int(np.searchsorted(base_to_raw, start, side="left"))
        right = int(np.searchsorted(base_to_raw, end, side="left"))
        if right > left:
            locations = base_to_raw[left:right] - start
            keep = ready[locations]
            if keep.any():
                output[left:right][keep] = projection.transform(states[locations[keep]]).astype(np.float32)
        if end % 262_144 == 0 or end == stream_rows:
            output.flush()
            engine.save_checkpoint(checkpoint_path)
            _atomic_json(progress_path, {**expected_meta, "next_row": end})
        _check_compute_cap()
    output.flush()
    del output
    _atomic_json(meta_path, expected_meta)
    checkpoint_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    return np.load(output_path, mmap_mode="r", allow_pickle=False)


class _AugmentedMatrix:
    def __init__(self, baseline, extras: list[np.ndarray]):
        self.baseline = baseline
        self.extras = extras
        self.shape = (len(baseline), baseline.shape[1] + sum(item.shape[1] for item in extras))

    def __getitem__(self, rows):
        pieces = [np.asarray(self.baseline[rows], dtype=np.float32)]
        pieces.extend(np.asarray(item[rows], dtype=np.float32) for item in self.extras)
        return np.column_stack(pieces)


def _lgbm_params(seed: int) -> dict:
    summary = json.loads(FEATURE_SUMMARY.read_text(encoding="utf-8"))
    params = dict(summary["params"])
    params.update({
        "seed": int(seed),
        "data_random_seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "metric": "None",
        "verbosity": -1,
        "num_threads": int(CONFIG["baseline"]["lightgbm"]["num_threads"]),
    })
    return params


def _fit_lgbm(features, labels, weights, training_mask, seed: int):
    _check_compute_cap()
    rows = np.flatnonzero(training_mask)
    dataset = lgb.Dataset(
        np.asarray(features[rows], dtype=np.float32),
        label=np.asarray(labels[rows], dtype=np.int8),
        weight=np.asarray(weights[rows], dtype=np.float32),
        free_raw_data=True,
    )
    model = lgb.train(
        _lgbm_params(seed), dataset,
        num_boost_round=int(CONFIG["baseline"]["lightgbm"]["boosting_rounds"]),
        callbacks=[lgb.log_evaluation(0)],
    )
    _check_compute_cap()
    return model


def _metric_pack(target, probability) -> dict:
    y = np.asarray(target, dtype=np.int8)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return {
        "n": int(len(y)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
    }


def _downstream_search_validation(finalists: list[dict], portfolio: dict, arrays: dict,
                                  *, stage_name: str = "search_validation") -> dict:
    result_path = OUT / f"downstream_{stage_name}.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("identity") == RUN_IDENTITY and saved.get("status") == "search_validation_downstream_complete":
            return saved
    started = time.monotonic()
    configs = {
        item["candidate_id"]: _config_from_record(item["effective_config"])
        for item in finalists
    }
    selected_ids = list(portfolio["selected_candidate_ids"])
    variants = {candidate_id: [candidate_id] for candidate_id in configs}
    if selected_ids:
        variants["selected_portfolio"] = selected_ids
    for name, members in portfolio.get("portfolio_variants", {}).items():
        if name in variants or not members or any(member not in configs for member in members):
            raise ValueError(f"invalid downstream variant {name!r}")
        variants[name] = list(members)
    progress_path = OUT / f"downstream_{stage_name}_progress.json"
    progress_identity = {
        "identity": RUN_IDENTITY,
        "candidate_ids": list(configs),
        "variants": variants,
        "reservoir_seeds": list(SEEDS),
        "lightgbm_seeds": list(LGBM_SEEDS),
    }
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("progress_identity") != progress_identity:
            raise RuntimeError("downstream checkpoint identity mismatch")
    else:
        progress = {
            "progress_identity": progress_identity,
            "baseline_metrics": {},
            "candidate_records": {name: [] for name in variants},
            "feature_generation_seconds": {},
            "lightgbm_fit_seconds": 0.0,
        }
    feature_arrays = {}
    feature_generation_seconds = dict(progress.get("feature_generation_seconds", {}))
    for candidate_id, config in configs.items():
        feature_arrays[candidate_id] = {}
        feature_generation_seconds.setdefault(candidate_id, 0.0)
        for seed in SEEDS:
            before = time.monotonic()
            feature_arrays[candidate_id][seed] = _generate_full_features(config, seed, arrays)
            feature_generation_seconds[candidate_id] += float(time.monotonic() - before)
            progress["feature_generation_seconds"] = feature_generation_seconds
            _atomic_json(progress_path, progress)

    common_ready = np.ones(len(arrays["target"]), dtype=bool)
    for features_by_seed in feature_arrays.values():
        for features in features_by_seed.values():
            common_ready &= np.isfinite(features).all(axis=1)
    opened = arrays["base_opened"]
    available = arrays["label_available"]
    decision = arrays["decision"]
    train_mask = (
        (opened < FIT_END_NS)
        & (available <= FIT_END_NS + DECISION_PURGE_NS)
        & common_ready
    )
    validation_mask = (
        decision
        & (opened >= VALIDATION_START_NS)
        & (opened < SEARCH_END_NS)
        & (available < SEARCH_END_NS)
        & common_ready
    )
    if train_mask.sum() < 100_000 or validation_mask.sum() < 1_000:
        raise RuntimeError("common finalist masks leave too few fit or search-validation rows")
    weights = np.where(decision, 0.59, 0.1025).astype(np.float32)
    target = arrays["target"]
    validation_rows = np.flatnonzero(validation_mask)
    baseline_metrics = dict(progress.get("baseline_metrics", {}))
    candidate_records = {name: list(rows) for name, rows in progress.get("candidate_records", {}).items()}
    for name in variants:
        candidate_records.setdefault(name, [])
    fit_seconds = float(progress.get("lightgbm_fit_seconds", 0.0))
    for lgbm_seed in LGBM_SEEDS:
        seed_key = str(lgbm_seed)
        if seed_key not in baseline_metrics:
            before = time.monotonic()
            baseline = _fit_lgbm(arrays["base_x"], target, weights, train_mask, lgbm_seed)
            fit_seconds += time.monotonic() - before
            baseline_probability = baseline.predict(np.asarray(arrays["base_x"][validation_rows], dtype=np.float32))
            baseline_metrics[seed_key] = _metric_pack(target[validation_rows], baseline_probability)
            progress.update({
                "baseline_metrics": baseline_metrics,
                "lightgbm_fit_seconds": fit_seconds,
            })
            _atomic_json(progress_path, progress)
            del baseline
        for name, members in variants.items():
            for reservoir_seed in SEEDS:
                if any(
                    int(row["reservoir_seed"]) == int(reservoir_seed)
                    and int(row["lightgbm_seed"]) == int(lgbm_seed)
                    for row in candidate_records[name]
                ):
                    continue
                extras = [feature_arrays[member][reservoir_seed] for member in members]
                matrix = _AugmentedMatrix(arrays["base_x"], extras)
                before = time.monotonic()
                model = _fit_lgbm(matrix, target, weights, train_mask, lgbm_seed)
                fit_seconds += time.monotonic() - before
                probability = model.predict(matrix[validation_rows])
                metric = _metric_pack(target[validation_rows], probability)
                latency = None
                latency_count = int(CONFIG.get("downstream", {}).get("latency_sample_count", 0))
                if latency_count > 0 and stage_name.startswith("finalists"):
                    latency_values = np.empty(min(latency_count, len(validation_rows)), dtype=np.int64)
                    for latency_index, row_index in enumerate(validation_rows[:len(latency_values)]):
                        one_row = matrix[int(row_index):int(row_index) + 1]
                        latency_start = time.perf_counter_ns()
                        model.predict(one_row)
                        latency_values[latency_index] = time.perf_counter_ns() - latency_start
                    latency = {
                        "p50_us": float(np.quantile(latency_values, 0.50) / 1_000.0),
                        "p95_us": float(np.quantile(latency_values, 0.95) / 1_000.0),
                        "p99_us": float(np.quantile(latency_values, 0.99) / 1_000.0),
                        "sample_count": int(len(latency_values)),
                    }
                candidate_records[name].append({
                    "reservoir_seed": int(reservoir_seed),
                    "lightgbm_seed": int(lgbm_seed),
                    "metrics": metric,
                    "metrics_level": "downstream_lightgbm_search_validation",
                    "delta_logloss": baseline_metrics[seed_key]["logloss"] - metric["logloss"],
                    "delta_brier": baseline_metrics[seed_key]["brier"] - metric["brier"],
                    "delta_auc": metric["auc"] - baseline_metrics[seed_key]["auc"],
                    "feature_count": int(matrix.shape[1]),
                    "single_row_latency": latency,
                })
                progress.update({
                    "baseline_metrics": baseline_metrics,
                    "candidate_records": candidate_records,
                    "feature_generation_seconds": feature_generation_seconds,
                    "lightgbm_fit_seconds": fit_seconds,
                })
                _atomic_json(progress_path, progress)
                del model, matrix
        del baseline

    summaries = {}
    for name, rows in candidate_records.items():
        deltas = np.asarray([row["delta_logloss"] for row in rows], dtype=np.float64)
        per_reservoir = {}
        for seed in SEEDS:
            values = [row["delta_logloss"] for row in rows if row["reservoir_seed"] == seed]
            per_reservoir[str(seed)] = {
                "mean_delta_logloss": float(np.mean(values)),
                "std_over_lightgbm_seeds": float(np.std(values, ddof=0)),
            }
        summaries[name] = {
            "mean_delta_logloss": float(np.mean(deltas)),
            "std_delta_logloss_all_pairs": float(np.std(deltas, ddof=0)),
            "positive_pair_count": int(np.sum(deltas > 0.0)),
            "negative_pair_count": int(np.sum(deltas < 0.0)),
            "mean_delta_brier": float(np.mean([row["delta_brier"] for row in rows])),
            "mean_delta_auc": float(np.mean([row["delta_auc"] for row in rows])),
            "per_reservoir_seed": per_reservoir,
            "pair_count": len(rows),
            "rows": rows,
        }
    result = {
        "status": "search_validation_downstream_complete",
        "evaluation_period": {
            "start_utc": CONFIG["split"]["search_validation_start_utc"],
            "end_exclusive_utc": CONFIG["split"]["search_validation_end_exclusive_utc"],
        },
        "common_fit_rows": int(train_mask.sum()),
        "common_validation_decision_rows": int(validation_mask.sum()),
        "reservoir_seeds": list(SEEDS),
        "lightgbm_seeds": list(LGBM_SEEDS),
        "baseline_metrics_by_lightgbm_seed": baseline_metrics,
        "candidate_configs": {key: _config_record(value) for key, value in configs.items()},
        "portfolio_candidate_ids": selected_ids,
        "candidate_results": summaries,
        "state_generation_seconds_by_candidate": feature_generation_seconds,
        "lightgbm_fit_seconds": float(fit_seconds),
        "elapsed_seconds": float(time.monotonic() - started),
    }
    _atomic_json(result_path, {"identity": RUN_IDENTITY, **result})
    progress_path.unlink(missing_ok=True)
    return result


def _write_report(historical_export: dict, architecture_search: dict, input_search: dict,
                  common_leaders: dict, input_leaders: dict, finalists: list[dict],
                  portfolio: dict, downstream: dict, arrays: dict, total_seconds: float) -> None:
    frozen_manifest = json.loads((OUT / "audit_manifest.json").read_text(encoding="utf-8"))
    historical = json.loads((TRIAL_EXPORT_DIR / "initial_48_trials.json").read_text(encoding="utf-8"))
    old_portfolio = json.loads((ROOT / "data/analysis/polymarket/BTC/btc_esn_portfolio_20261020_r3/portfolio_selection.json").read_text(encoding="utf-8"))
    old_downstream = json.loads((ROOT / "data/analysis/polymarket/BTC/btc_esn_portfolio_20261020_r3/downstream_budget.json").read_text(encoding="utf-8"))
    old_runtime = json.loads((ROOT / "data/analysis/polymarket/BTC/btc_esn_portfolio_20261020_r3/runtime_ledger.json").read_text(encoding="utf-8"))
    old_winner = next(row for row in historical if row["trial_number"] == 44)
    old_winner_config = old_winner["effective_config"]
    old_outcome = old_winner["outcome"]
    old_feature_order = json.loads(FEATURE_SUMMARY.read_text(encoding="utf-8"))["feature_order"]
    old_feature_hash = hashlib.sha256(json.dumps(old_winner_config, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    rejected_trial = next(row for row in historical if row["status"] == "rejected")
    best_input_row = max(
        input_leaders.values(), key=lambda row: row["outcome"]["mean_delta_logloss"],
    )
    selected_portfolio_result = downstream.get("candidate_results", {}).get("selected_portfolio")
    selected_portfolio_delta = (
        None if selected_portfolio_result is None
        else float(selected_portfolio_result["mean_delta_logloss"])
    )

    old_counts = {}
    topology_counts = {}
    for row in historical:
        config = row["effective_config"]
        family = config.get("architecture", "unknown")
        topology = config.get("topology", "unknown")
        old_counts.setdefault(family, {"attempted": 0, "scored": 0, "rejected": 0})
        old_counts[family]["attempted"] += 1
        old_counts[family]["rejected" if row["status"] == "rejected" else "scored"] += 1
        topology_counts.setdefault(topology, {"attempted": 0, "scored": 0, "rejected": 0})
        topology_counts[topology]["attempted"] += 1
        topology_counts[topology]["rejected" if row["status"] == "rejected" else "scored"] += 1
    rows = arrays["prepared"]["development_baseline_rows"]

    lines = [
        "# Audyt ESN: przestrzeń architektur i selekcja portfela",
        "",
        "## Odpowiedź",
        "",
        "**Pierwotny eksperyment nie wykazał, że `ring 1×128` wygrał rzetelne porównanie rodzin i portfeli.** Silnik implementował single, parallel i DeepESN, a każda rodzina pojawiła się w 48 trialach. Przeszukiwanie nie gwarantowało jednak minimalnej liczby prób na rodzinę, a liczba jednostek w trialach nie była kontrolowana. Końcowy ranking skupił się na `single/ring/128`; zwycięski trial 44 nie należał do dwóch kandydatów sprawdzonych downstream w LightGBM. Portfel testował cztery dodatki z krótkiej listy i prawidłowo pozostał przy jednym członku według własnej reguły probe, ale to nie porównywało portfeli między architekturami.",
        "",
        "Nowy, osobno wersjonowany search r4 porównuje trzy rodziny przy 128 lub 256 łącznych jednostkach, zaczyna od sześciu kontrolowanych konfiguracji z tymi samymi wejściami i sprawdza wkład kandydatów do pełnego zestawu. Wszystkie nowe wyniki w tym raporcie pochodzą wyłącznie z `search_validation` i nie stanowią niezależnego potwierdzenia poprawy.",
        "",
        f"Poprawiony dobór grup wejściowych dał najlepszy probe dla `{best_input_row['effective_config']['architecture']}` z `{', '.join(best_input_row['effective_config']['input_groups'])}` ({best_input_row['outcome']['mean_delta_logloss']:+.9f}). Greedy screen wybrał {len(portfolio['selected_configs'])} członków z probe score `{portfolio['portfolio_score']['mean_delta_logloss']:+.9f}`; downstream LightGBM dla tego zestawu ma Δ log-loss `{selected_portfolio_delta if selected_portfolio_delta is not None else float('nan'):+.9f}`. To wskazuje, że dodatni probe screen nie wystarcza do wykazania poprawy downstream.",
        "",
        "## Zgodność pierwotnego eksperymentu",
        "",
        "| Element planu | Zaimplementowany | Dostępny w searchu | Faktycznie oceniony | Dowód |",
        "|---|---|---|---|---|",
        "| Pojedynczy leaky ESN | Tak | Tak | Tak | `ESNConfig` i triale `single`; liczba w tabeli poniżej |",
        "| Parallel ESN | Tak | Tak | Tak | Niezależne stany w `ESNEngine`; triale `parallel`; pełna liczba w eksporcie |",
        "| DeepESN | Tak | Tak | Tak | Kolejna warstwa dostaje `next_states[r-1]`; triale `deep` |",
        "| Ring i random sparse | Tak | Tak | Tak | Oba topologie w trialach; sparse ma losową maskę połączeń |",
        "| Dobór wejść | Tak | Tak | Tak | Sześć binarnych grup wejściowych; zwycięzca użył grupy `volume` |",
        "| Rozmiary reservoirów | Tak | 32, 64, 128 | Tak | Efektywne rozmiary w eksporcie; porównania nie wyrównywały sumy jednostek |",
        "| Leak, recurrent/input scaling, bias | Tak | Tak | Tak | Ciągłe parametry Optuny; trzy wspólne seedy reservoiru |",
        "| Connectivity | Tak | Dla sparse | Tak | Bernoulli dla połączeń poza diagonalą; diagonalę wymusza kod; ring ma 2 sąsiadów |",
        "| Kompresja stanów | Tak | Stała | Tak | PCA 8 składowych na reservoir; 128 stanów ringa dawało 8 cech probe |",
        "| Ocena na różnych seedach | Tak | 3 seedy reservoiru | Tak | Wspólne seedy `20261020–20261022`; osobne od seedów nie było w downstream |",
        "| Selekcja portfela | Częściowo | Maks. 2 członków | Tak, ograniczona | Greedy probe testował 4 dodatki spośród 5 najlepszych; brak kosztu per dodatek i brak porównania portfeli rodzin downstream |",
        "",
        "### Faktyczne pokrycie pierwszych 48 triali",
        "",
        "| Rodzina/topologia | Próby | Ocenione probe | Odrzucone |",
        "|---|---:|---:|---:|",
    ]
    for family in ("single", "parallel", "deep"):
        count = old_counts.get(family, {"attempted": 0, "scored": 0, "rejected": 0})
        lines.append(f"| `{family}` | {count['attempted']} | {count['scored']} | {count['rejected']} |")
    for topology in ("ring", "sparse", "dense"):
        count = topology_counts.get(topology, {"attempted": 0, "scored": 0, "rejected": 0})
        lines.append(f"| `{topology}` | {count['attempted']} | {count['scored']} | {count['rejected']} |")
    lines += [
        "",
        f"Wszystkie 48 rekordów Optuny mają stan `COMPLETE`; 47 miało wynik probe, a trial {rejected_trial['trial_number']} został odrzucony: `{rejected_trial['rejection_reason']}`. TPE nie wymuszało pokrycia rodzin. Pierwsze próby zawierały single, parallel i deep, lecz porównania np. 1×128 z 3×128 miały różne budżety. W próbach 19–44 dominował `single/ring/128`; 48 prób nie było z góry reprezentatywną pulą.",
        "",
        f"Pełne efektywne konfiguracje, parametry Optuny, statusy, wynik, czas, delty seedów, warm-up i przyczyny odrzucenia: [{Path(historical_export['csv_path']).name}](btc_esn_architecture_audit_20261020/{Path(historical_export['csv_path']).name}) oraz [{Path(historical_export['json_path']).name}](btc_esn_architecture_audit_20261020/{Path(historical_export['json_path']).name}). Eksport obejmuje 48/48 prób.",
        "",
        "### Dokładna konstrukcja zwycięskiego `ring 1×128`",
        "",
        f"Trial 44: `architecture=single`, `topology=ring`, 128 jednostek, leak `{old_winner_config['leak_rate']:.15g}`, recurrent spectral radius `{old_winner_config['spectral_radius']:.15g}`, input scaling `{old_winner_config['input_scaling']:.15g}`, bias scaling `{old_winner_config['bias_scaling']:.15g}`. Wartość connectivity `{old_winner_config.get('connectivity')}` jest nieużywana przez ring. Aktywacja to `tanh`; stan startuje od zera i jest resetowany po luce lub niepoprawnej świecy.",
        "",
        "Dla jednostki `i` z indeksem modulo 128 aktualizacja ma postać:",
        "",
        "```text",
        "h_i(t) = (1 − leak) h_i(t−1) + leak · tanh(W_in,i · u(t) + b_i + W_rec,i,i−1 h_i−1(t−1) + W_rec,i,i+1 h_i+1(t−1))",
        "```",
        "",
        "Wiersz `i` przyjmuje sygnał z `i−1` i `i+1`; dwa wagi są losowane niezależnie z `Normal(0,1)`, po czym cała macierz jest skalowana przez `spectral_radius / max(abs(eigvals(W_rec)))`. Nie ma self-connections. Weights wejściowe i bias są losowane jednostajnie z `[-input_scaling,+input_scaling]` i `[-bias_scaling,+bias_scaling]`. Struktura sąsiedztwa ringa jest deterministyczna; wagi recurrent, input weights i bias zależą od seedu. PCA także zależy od seedu reservoiru.",
        "",
        "Dwie surowe wejściowe cechy to `log_volume = log1p(Volume)` oraz `volume_ratio_to_prior_30m_median = Volume / median(poprzednich 30 wolumenów)`. Druga cecha wyklucza bieżącą świecę; luki resetują historię. Obie cechy są standaryzowane średnią i odchyleniem z fit-core; wartości niefinitywne są zastępowane zerem po transformacji.",
        "",
        f"Trzy seedy reservoiru: `{', '.join(map(str, SEEDS))}`. Screen delta log-loss (control minus augmented) dla trialu 44: `{', '.join(f'{value:.9f}' for value in old_outcome['seed_deltas'])}`; średnia `{old_outcome['mean_delta_logloss']:.9f}`, odchylenie populacyjne `{old_outcome['std_delta_logloss']:.9f}`. Warm-up (liczba minut po resecie) per seed: `{', '.join(map(str, old_outcome['warmup_steps_by_seed']))}`; szczegóły diagnostyki są w JSON eksporcie.",
        "",
        f"128 surowych stanów było kompresowanych przez PCA fitowane na 20 000 stanów z fit-core; probe dostawał 8 składowych. Model downstream używał 70 kolumn BASE70 w kolejności z `B0_cv_summary.json`, a po nich 8 składowych `r0_pc0–r0_pc7` (nazwy `esn_{old_feature_hash}_seed_<reservoir_seed>_r0_pc0` … `_r0_pc7`). Czyli winner nie przekazywał 128 cech LightGBM.",
        "",
        "BASE70 order: `" + "`, `".join(old_feature_order) + "`.",
        "",
        "### Sposób wyboru i portfel w r3",
        "",
        "Optuna maksymalizowała `logloss(control) − logloss(augmented)`: logistyczna korekta bazowego logitu kontra ta sama korekta rozszerzona o PCA ESN. To była poprawa ponad baseline, nie samodzielna jakość probe. Predykcje bazowe powstały z LightGBM fitowanego na fit-core, z seedem 20261015; korekta uczyła się na OOS correction-fit, a ocenę dawała na późniejszym `search_validation`. Zestaw feature’ów BASE70 i hiperparametry były wcześniej wybrane z pełnego development, więc wniosek jest warunkowy względem tego baseline’u.",
        "",
        "Kontrola i augmented logistic używały tego samego `C=0.1`, solvera i wierszy. Finaliści probe byli oceniani na trzech wspólnych seedach reservoiru; nie były to trzy niezależne realizacje LightGBM. Downstream w LightGBM sprawdził triale 42 i 43, czyli najwyżej ocenione konfiguracje poza wybranym trialem 44; wybrany winner 44 nie miał bezpośredniego downstream check na `search_validation`. W starym downstream seed reservoiru i LightGBM był tym samym numerem, więc brakowało pełnego rozdzielenia tych źródeł wariancji.",
        "",
        "| Zestaw przed | Dodany kandydat | Wynik przed | Wynik po | Przyrost | Koszt | Decyzja |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    old_single = float(old_portfolio["portfolio_result"]["mean_delta_logloss"])
    selected_trial = int(old_portfolio["selected_members"][0]["trial_number"])
    lines.append(f"| `[]` | trial {selected_trial} (`ring 1×128`) | 0.000000000 | {old_single:.9f} | {old_single:.9f} | nie zapisano | wybrano jako single-screen leader |")
    old_trial_map = {row["trial_number"]: row for row in historical}
    tested_additions = []
    for stage in old_portfolio["portfolio_stages"]:
        if "trial_number" in stage:
            tested_additions.append(stage)
    for stage in tested_additions:
        trial_id = int(stage["trial_number"])
        after = float(stage["mean_delta_logloss"])
        candidate = old_trial_map[trial_id]["effective_config"]
        label = f"trial {trial_id} `{candidate['architecture']}/{candidate['topology']} {candidate['reservoir_count']}×{candidate['reservoir_size']}`"
        decision = "odrzucono: dodanie obniżyło wynik pełnego zestawu" if after <= old_single else "nie wybrano: inny dodatek miał wyższy wynik"
        lines.append(f"| `trial {selected_trial}` | {label} | {old_single:.9f} | {after:.9f} | {after - old_single:+.9f} | nie zapisano | {decision} |")
    lines += [
        "",
        f"Reguła testowała cztery dodatki (trial 42, 43, 45 i 4); najlepszy zestaw dwuelementowy uzyskał `{max(float(row['mean_delta_logloss']) for row in tested_additions):.9f}`, poniżej wyniku single `{old_single:.9f}`. Portfel faktycznie sprawdzono w tej krótkiej liście i odrzucono dodatki na podstawie wyniku korekty, nie korelacji stanów. Brak zapisanego kosztu per dodatek ogranicza audyt koszt/korzyść.",
        "",
        "| Downstream `search_validation` r3 | Średnia poprawa log-loss | Seedy LightGBM (w tej wersji zrównane z reservoir seeds) |",
        "|---|---:|---|",
    ]
    for row in old_downstream["results"]:
        deltas = [item["delta_logloss"] for item in row["seed_results"]]
        lines.append(f"| trial {row['trial_number']} | {row['mean_delta_logloss']:+.9f} | `{', '.join(f'{value:+.9f}' for value in deltas)}` |")

    lines += [
        "",
        "## Poprawiony search r4 — wyłącznie `search_validation`",
        "",
        f"Podział zachowuje fit-core do `{FIT_END.isoformat()}`, correction-fit do `{CORRECTION_END.isoformat()}` i walidację `{VALIDATION_START.isoformat()}–{SEARCH_END.isoformat()}`. Cache zawiera {arrays['prepared']['development_raw_rows']:,} raw rows i {rows:,} BASE70 rows, materializowane przed końcem walidacji; runner sprawdza każdą etykietę `< {SEARCH_END.isoformat()}`. Poza tym cutoff runner nie czyta danych ani etykiet.",
        "",
        "Początkowe sześć prób porównuje single 1×128, parallel 2×64 i deep 2×64 dla ring oraz sparse, z tymi samymi 128 jednostkami, wszystkimi grupami wejść i wspólnymi parametrami dynamiki. Kolejne próby Optuny mają warunkowe `n_reservoirs`/`layers`, budżety 128/256 jednostek i dobierają dynamikę. Drugi etap, po zamrożeniu lidera każdej rodziny, osobno dobiera grupy wejściowe. Ring i sparse są równoważnymi wariantami topologii; sparse connectivity jest losowanym prawdopodobieństwem połączenia poza wymuszoną diagonalą.",
        "",
        "| Architektura/topologia — wspólna pula | Trial | Δ log-loss probe | Rozrzut między 3 reservoir seeds |",
        "|---|---:|---:|---:|",
    ]
    anchor_rows = {
        row.get("anchor_id"): row
        for row in architecture_search["trials"]
        if row.get("anchor_id") and row.get("outcome")
    }
    for architecture in CONFIG["search"]["architectures"]:
        for topology in CONFIG["search"]["topologies"]:
            anchor_id = f"{architecture}_{topology}_common_inputs"
            row = anchor_rows.get(anchor_id)
            if row:
                outcome = row["outcome"]
                lines.append(f"| `{architecture}/{topology}` | {row['trial_number']} | {outcome['mean_delta_logloss']:+.9f} | {outcome['std_delta_logloss']:.9f} |")
            else:
                lines.append(f"| `{architecture}/{topology}` | — | — | anchor incompleto |")
    lines += [
        "",
        "### Najlepsi kandydaci według rodziny i topologii",
        "",
        "| Rodzina/topologia | Trial | Jednostki | Wejścia | Probe Δ log-loss | SD między reservoir seeds |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for (architecture, topology), row in sorted(common_leaders.items()):
        config = row["effective_config"]
        outcome = row["outcome"]
        lines.append(
            f"| `{architecture}/{topology}` | {row['trial_number']} | {config['total_units']} | `{', '.join(config['input_groups'])}` | "
            f"{outcome['mean_delta_logloss']:+.9f} | {outcome['std_delta_logloss']:.9f} |"
        )
    lines += [
        "",
        "### Osobna optymalizacja grup wejściowych",
        "",
        "| Rodzina | Trial | Wybrane grupy | Probe Δ log-loss | SD reservoir seeds |",
        "|---|---:|---|---:|---:|",
    ]
    for architecture, row in sorted(input_leaders.items()):
        config = row["effective_config"]
        outcome = row["outcome"]
        lines.append(
            f"| `{architecture}` | {row['trial_number']} | `{', '.join(config['input_groups'])}` | "
            f"{outcome['mean_delta_logloss']:+.9f} | {outcome['std_delta_logloss']:.9f} |"
        )
    lines += [
        "",
        "Etap wejść optymalizował grupy osobno po kontrolowanym porównaniu architektur. Probe i downstream używają tych samych 8 składowych PCA na kandydata. Kontrola i augmented logistic mają jednakowe `C=0.1`, a wszyscy kandydaci portfela są oceniani na tym samym zestawie korekta/walidacja dla danego wspólnego reservoir seed.",
        "",
        "### LightGBM downstream i portfel",
        "",
        "Poniższe delty są obliczone wyłącznie na `search_validation`. Δ log-loss to baseline minus augmented, więc wartość dodatnia oznacza mniejszy log-loss augmented. Każdy finalist ma ten sam podzbiór wierszy, model baseline jest dopasowywany na fit-core, a reservoir seeds (`20261020–20261022`) są krzyżowane z odrębnymi seedami LightGBM (`20261101–20261103`). Wartości są wynikiem selekcji na tej samej walidacji, więc nie są niezależnym potwierdzeniem.",
        "",
        "| Kandydat | Konfiguracja | Średnia Δ log-loss | SD wszystkich 9 par | Średnia Δ Brier | Δ AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for candidate_id, result in downstream["candidate_results"].items():
        config = downstream["candidate_configs"].get(candidate_id)
        label = "selected portfolio" if candidate_id == "selected_portfolio" else candidate_id
        if config:
            details = f"{config['architecture']}/{config['topology']} {config['total_units']}u, {','.join(config['input_groups'])}"
        else:
            details = "; ".join(
                f"{item['architecture']}/{item['topology']} {item['total_units']}u"
                for item in portfolio["selected_configs"]
            )
        lines.append(
            f"| `{label}` | {details} | {result['mean_delta_logloss']:+.9f} | {result['std_delta_logloss_all_pairs']:.9f} | "
            f"{result['mean_delta_brier']:+.9f} | {result['mean_delta_auc']:+.7f} |"
        )
    lines += [
        "",
        "| Bieżący zestaw | Dodany kandydat | Wynik przed | Wynik po | Przyrost | Koszt (sekundy) | Decyzja |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in portfolio["selection_trace"]:
        config = item["candidate_config"]
        candidate = f"{config['architecture']}/{config['topology']} {config['total_units']}u ({item['added_candidate']})"
        current = ",".join(item["current_set"]) or "[]"
        cost = item.get("generation_seconds", 0.0) + item.get("probe_seconds", 0.0)
        lines.append(
            f"| `{current}` | `{candidate}` | {item['score_before']:+.9f} | {item['score_after']:+.9f} | "
            f"{item['marginal_delta_logloss']:+.9f} | {cost:.3f} | {item['decision']} |"
        )
    lines += [
        "",
        f"Wybrany screen portfolio: {len(portfolio['selected_configs'])} członków, score `{portfolio['portfolio_score']['mean_delta_logloss']:+.9f}`; końcową wartość sprawdzono dodatkowo w LightGBM na wspólnym `search_validation`.",
        "",
        "## Koszt, wznowienie i artefakty",
        "",
        f"Nowy search ma twardy limit {CONFIG['search']['whole_experiment_compute_limit_seconds']:,} s (4 h), z limitem do {CONFIG['search']['architecture_search_time_limit_seconds']:,} s na architektury, {CONFIG['search']['input_group_search_time_limit_seconds']:,} s na wejścia i rezerwą {CONFIG['search']['finalist_and_portfolio_reserve_seconds']:,} s na finalistów/portfel. Łącznie faktycznie zapisano {total_seconds:.1f} s.",
        "",
        "| Etap | Sekundy / liczba |",
        "|---|---:|",
    ]
    ledger = _runtime_ledger()
    for stage, seconds in sorted(ledger.get("stages_seconds", {}).items()):
        lines.append(f"| `{stage}` | {seconds:.3f} |")
    search_trials = architecture_search["trials"] + input_search["trials"]
    scored_outcomes = [row["outcome"] for row in search_trials if row.get("outcome")]
    search_generation_seconds = sum(
        sum(outcome.get("state_generation_seconds_by_seed", [])) for outcome in scored_outcomes
    )
    search_probe_seconds = sum(
        sum(outcome.get("probe_seconds_by_seed", [])) for outcome in scored_outcomes
    )
    search_trial_seconds = sum(
        float(row.get("duration_seconds") or 0.0) for row in search_trials
    )
    input_stage_key = (
        "input_group_correction"
        if "input_group_correction" in ledger.get("stages_seconds", {})
        else "input_search"
    )
    search_stage_seconds = float(ledger.get("stages_seconds", {}).get("architecture_search", 0.0))
    search_stage_seconds += float(ledger.get("stages_seconds", {}).get(input_stage_key, 0.0))
    search_stage_overrun_seconds = max(0.0, search_trial_seconds - search_stage_seconds)
    unique_scored_configs = {
        json.dumps(row["effective_config"], sort_keys=True, separators=(",", ":"))
        for row in search_trials
        if row.get("status") == "completed_scored" and row.get("effective_config")
    }
    failed_trials = sum(row["state"] == "FAIL" for row in search_trials)
    failed_trial_seconds = sum(
        float(row.get("duration_seconds") or 0.0)
        for row in search_trials if row["state"] == "FAIL"
    )
    rejected_trials = sum(row["status"] == "rejected" for row in search_trials)
    lines += [
        f"| `search state generation / warm-up / PCA` | {search_generation_seconds:.3f} |",
        f"| `search logistic probes` | {search_probe_seconds:.3f} |",
        f"| `search stage time outside trial durations` | {max(0.0, search_stage_seconds - search_trial_seconds):.3f} |",
        f"| `trial runtime beyond search timeouts` | {search_stage_overrun_seconds:.3f} |",
        f"| `unique scored search configs / repeated trials` | {len(unique_scored_configs)} / {sum(row.get('status') == 'completed_scored' for row in search_trials) - len(unique_scored_configs)} |",
        f"| `portfolio state generation` | {portfolio.get('state_generation_seconds', 0.0):.3f} |",
        f"| `portfolio full-set probes` | {portfolio.get('probe_seconds', 0.0):.3f} |",
        f"| `portfolio orchestration / I/O` | {max(0.0, portfolio['elapsed_seconds'] - portfolio.get('state_generation_seconds', 0.0) - portfolio.get('probe_seconds', 0.0)):.3f} |",
        f"| `finalist feature generation` | {sum(downstream['state_generation_seconds_by_candidate'].values()):.3f} |",
        f"| `LightGBM fits` | {downstream['lightgbm_fit_seconds']:.3f} |",
        f"| `downstream prediction / I/O / orchestration` | {max(0.0, downstream['elapsed_seconds'] - sum(downstream['state_generation_seconds_by_candidate'].values()) - downstream['lightgbm_fit_seconds']):.3f} |",
        f"| `Optuna rejected / failed / resumed` | {rejected_trials} / {failed_trials} / {ledger.get('resume_count', 0)} |",
        f"| `failed Optuna trial time` | {failed_trial_seconds:.3f} |",
        f"| `pretrial failed starts / seconds` | {ledger.get('pretrial_failed_attempts', 0)} / {ledger.get('pretrial_failed_attempt_active_seconds', 0.0):.3f} |",
        f"| `historical export` | {ledger.get('stages_seconds', {}).get('historical_trial_export', 0.0):.3f} |",
        f"| `report + manifest finalization` | {ledger.get('stages_seconds', {}).get('report_and_manifest', 0.0):.3f} |",
        "",
        "### Historia wznowień przed oceną poprawionej puli",
        "",
    ]
    for index, revision in enumerate(frozen_manifest.get("pretrial_runner_revisions", []), start=1):
        lines.append(
            f"{index}. `{revision.get('reason', 'runner correction')}` — "
            f"{revision.get('completed_trials', 0)} ukończonych i {revision.get('queued_anchors_only', 0)} oczekujących kotwic; "
            f"szczegóły: `{revision.get('error', 'brak błędu próby')}`."
        )
    lines += [
        "",
        f"Pierwotny runtime ledger rozdzielał search `{old_runtime['stages_seconds'].get('optuna_search', 0.0):.1f}s`, probe/state generation łącznie w tej samej puli, portfel `{old_runtime['stages_seconds'].get('portfolio_selection', 0.0):.1f}s`, downstream `{old_runtime['stages_seconds'].get('downstream_budget', 0.0):.1f}s`, finalne etapy `{old_runtime['stages_seconds'].get('confirmation_and_final_fit', 0.0) + old_runtime['stages_seconds'].get('confirmation_data_materialization', 0.0):.1f}s` oraz wcześniejsze nieudane próby `{old_runtime.get('prior_failed_attempt_active_seconds', 0.0):.1f}s`. Stary runner nie rozdzielał kosztu generowania stanów i probe per trial ani kosztu każdego dodatku do portfela.",
        "",
        f"Polecenie uruchomienia i wznowienia (bez argumentów CLI): `python {Path(__file__).name}`. Nowy cache: `{OUT.relative_to(ROOT).as_posix()}/`; manifest: `{(OUT / 'audit_manifest.json').relative_to(ROOT).as_posix()}`. Eksport pierwszych 48 triali: `{historical_export['csv_path']}` i `{historical_export['json_path']}`.",
        "",
        "Holdout, późniejsze dane/etykiety/predykcje, wcześniejsza ocena po search-validation, live model i replay Polymarket nie są wejściem nowego runnera. Wyniki screeningu i finalistów na `search_validation` są selekcyjne.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "complete_search_validation_only",
        "identity": RUN_IDENTITY,
        "split": CONFIG["split"],
        "reused_search_cache": CONFIG["reused_search_cache_directory"],
        "search_validation_cutoff_exclusive_utc": CONFIG["excluded_data"]["materialization_cutoff_exclusive_utc"],
        "holdout_rows_materialized": 0,
        "read_after_search_validation_cutoff": False,
        "holdout_evaluation": False,
        "polymarket_book_replay": False,
        "live_configuration_changes": False,
        "live_orders": False,
        "historical_48_trial_export": historical_export,
        "pretrial_runner_revisions": frozen_manifest.get("pretrial_runner_revisions", []),
        "architecture_search": {key: value for key, value in architecture_search.items() if key != "trials"},
        "input_search": {key: value for key, value in input_search.items() if key != "trials"},
        "family_finalists": finalists,
        "portfolio_selection": portfolio,
        "downstream_search_validation": downstream,
        "runtime_ledger": ledger,
        "total_active_seconds": total_seconds,
        "report_path": CONFIG["report_path"],
    }
    _atomic_json(OUT / "audit_manifest.json", manifest)


def _write_frozen_manifest(identity: dict, prepared: dict) -> None:
    path = OUT / "audit_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise RuntimeError("frozen audit manifest identity changed; use a new versioned directory")
        return
    _atomic_json(path, {
        "experiment_id": CONFIG["experiment_id"],
        "status": "frozen_before_search",
        "identity": identity,
        "split": CONFIG["split"],
        "search_validation_cutoff_exclusive_utc": CONFIG["excluded_data"]["materialization_cutoff_exclusive_utc"],
        "prepared_search_manifest": {
            key: value for key, value in prepared.items()
            if key not in {"identity"}
        },
        "holdout_rows_materialized": 0,
        "read_after_search_validation_cutoff": False,
        "holdout_evaluation": False,
        "polymarket_book_replay": False,
        "live_configuration_changes": False,
        "live_orders": False,
    })


def _save_run_state(stage: str, next_step: str, **extra) -> None:
    _atomic_json(OUT / "run_state.json", {
        "identity": RUN_IDENTITY,
        "status": stage,
        "next_resumable_step": next_step,
        "active_seconds": _elapsed_seconds(),
        "holdout_rows_materialized": 0,
        **extra,
    })


def _append_research_status(stage: str, next_step: str, detail: str) -> None:
    path = ROOT / "docs/BTC_RESEARCH_STATUS_20261010.md"
    timestamp = datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M:%S %Z")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            f"\n## ESN audit r4 checkpoint — {timestamp}\n\n"
            f"- Stage: `{stage}`.\n"
            f"- Next resumable step: `{next_step}`.\n"
            f"- Detail: {detail}\n"
            f"- Scope: fit-core, correction-fit, and search-validation ending `2025-07-01T00:00:00Z`; no reads beyond that cutoff.\n"
        )


def main() -> None:
    global RUN_STARTED, RUN_PRIOR_SECONDS, RUN_IDENTITY, _SCREEN_ARRAYS
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_STARTED = time.monotonic()
    prepared = json.loads((PREP / "prepared_manifest.json").read_text(encoding="utf-8"))
    RUN_IDENTITY = _identity(prepared)
    ledger_path = OUT / "runtime_ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("identity") != RUN_IDENTITY:
            raise RuntimeError("runtime ledger identity changed; refusing to resume another experiment")
        RUN_PRIOR_SECONDS = float(ledger.get("active_seconds", 0.0))
        ledger["resume_count"] = int(ledger.get("resume_count", 0)) + 1
        _save_runtime_ledger(ledger)
        if ledger.get("pretrial_failed_attempts") and ledger["resume_count"] == 1:
            _append_research_status(
                "pretrial runner correction and safe resume",
                "architecture search anchors",
                f"the anchor counter failed before scoring; identity migration verified all {len(_architecture_anchors())} anchors were still WAITING and zero trials had outcomes; preserved {ledger.get('pretrial_failed_attempt_active_seconds', 0.0):.3f}s in the four-hour cap.",
            )
    else:
        RUN_PRIOR_SECONDS = 0.0
        _save_runtime_ledger(_runtime_ledger())

    _write_frozen_manifest(RUN_IDENTITY, prepared)
    arrays = _load_search_data(RUN_IDENTITY)
    _SCREEN_ARRAYS = arrays
    _save_run_state("search_inputs_loaded", "export_historical_trials")

    stage_start = time.monotonic()
    historical_export = _export_historical_trials()
    _record_stage_time("historical_trial_export", time.monotonic() - stage_start)
    if historical_export["trial_count"] != 48:
        raise RuntimeError("historical trial export does not contain all 48 trials")
    previous_runtime = json.loads((
        ROOT / "data/analysis/polymarket/BTC/btc_esn_portfolio_20261020_r3/runtime_ledger.json"
    ).read_text(encoding="utf-8"))
    _append_research_status(
        "frozen r4 preflight complete",
        "architecture search anchors, then adaptive search",
        f"loaded {arrays['prepared']['development_raw_rows']:,} permitted raw rows and {arrays['prepared']['development_baseline_rows']:,} BASE70 rows; exported {historical_export['trial_count']} original trials; old measured search was {previous_runtime['stages_seconds']['optuna_search']:.1f}s for 48 trials and old downstream was {previous_runtime['stages_seconds']['downstream_budget']:.1f}s for two candidates; new runner cap is 14,400s.",
    )

    ledger = _runtime_ledger()
    search_settings = CONFIG["search"]
    cap = float(search_settings["whole_experiment_compute_limit_seconds"])
    finalist_reserve = float(search_settings["finalist_and_portfolio_reserve_seconds"])
    arch_used = float(ledger.get("stages_seconds", {}).get("architecture_search", 0.0))
    arch_limit = max(0.0, float(search_settings["architecture_search_time_limit_seconds"]) - arch_used)
    input_reserved = float(search_settings["input_group_search_time_limit_seconds"])
    available_for_search = max(0.0, cap - _elapsed_seconds() - finalist_reserve)
    arch_allowance = min(arch_limit, max(0.0, available_for_search - input_reserved))
    stage_start = time.monotonic()
    architecture_search = _run_study(
        "architecture", f"{CONFIG['experiment_id']}_architecture",
        int(search_settings["tpe_sampler_seed_architecture"]),
        _architecture_anchors(), arch_allowance, arrays,
    )
    _record_stage_time("architecture_search", time.monotonic() - stage_start)
    _append_research_status(
        "architecture search checkpoint",
        "input-group anchors and adaptive input search",
        f"{architecture_search['attempted_trials']} attempted architecture trials; {architecture_search['anchors_completed']}/6 required family/topology anchors complete; active runner time {_elapsed_seconds():.1f}s.",
    )
    _save_run_state(
        "architecture_search_complete" if architecture_search["anchors_complete"] else "architecture_anchors_incomplete",
        "input_group_search" if architecture_search["anchors_complete"] else "resume_architecture_search",
        architecture_trials=architecture_search["attempted_trials"],
        architecture_anchors_completed=architecture_search["anchors_completed"],
    )
    if not architecture_search["anchors_complete"]:
        raise RuntimeError("the six predeclared architecture/topology anchors did not complete; resume the runner")

    stage1_by_arch = {}
    for row in _scored_trials(architecture_search):
        architecture = row["effective_config"]["architecture"]
        current = stage1_by_arch.get(architecture)
        if current is None or row["outcome"]["mean_delta_logloss"] > current["outcome"]["mean_delta_logloss"]:
            stage1_by_arch[architecture] = row
    if set(stage1_by_arch) != set(CONFIG["search"]["architectures"]):
        raise RuntimeError("architecture search has no scored representative for every family")
    input_bases = {
        architecture: row["effective_config"]
        for architecture, row in stage1_by_arch.items()
    }
    ledger = _runtime_ledger()
    input_used = float(ledger.get("stages_seconds", {}).get("input_search", 0.0))
    input_limit = max(0.0, float(search_settings["input_group_search_time_limit_seconds"]) - input_used)
    input_allowance = min(input_limit, max(0.0, cap - _elapsed_seconds() - finalist_reserve))
    stage_start = time.monotonic()
    input_search = _run_study(
        "input_search", f"{CONFIG['experiment_id']}_input_groups",
        int(search_settings["tpe_sampler_seed_inputs"]),
        _input_anchors(input_bases), input_allowance, arrays, input_bases,
    )
    _record_stage_time("input_search", time.monotonic() - stage_start)
    _append_research_status(
        "input-group search checkpoint",
        "family finalists and full-set portfolio probe",
        f"{input_search['attempted_trials']} attempted input trials; {input_search['anchors_completed']}/18 required family/group anchors complete; active runner time {_elapsed_seconds():.1f}s.",
    )
    _save_run_state(
        "input_search_complete" if input_search["anchors_complete"] else "input_anchors_incomplete",
        "portfolio_selection" if input_search["anchors_complete"] else "resume_input_search",
        input_trials=input_search["attempted_trials"],
        input_anchors_completed=input_search["anchors_completed"],
    )
    if not input_search["anchors_complete"]:
        raise RuntimeError("the declared per-family input-group anchors did not complete; resume the runner")

    common_leaders, input_leaders, finalists = _family_leaders(architecture_search, input_search)
    if len(common_leaders) != len(CONFIG["search"]["architectures"]) * len(CONFIG["search"]["topologies"]):
        raise RuntimeError("the architecture search did not score every family/topology pair")
    if len(finalists) < len(CONFIG["search"]["architectures"]):
        raise RuntimeError("too few unique finalists remain for a family comparison")

    _check_compute_cap()
    stage_start = time.monotonic()
    portfolio = _select_portfolio(finalists, arrays)
    _record_stage_time("portfolio_selection", time.monotonic() - stage_start)
    _append_research_status(
        "portfolio selection checkpoint",
        "search-validation LightGBM finalist evaluation",
        f"tested {len(finalists)} family finalists; selected {len(portfolio['selected_configs'])} portfolio members after incremental full-set probes; active runner time {_elapsed_seconds():.1f}s.",
    )
    _save_run_state("portfolio_selection_complete", "search_validation_downstream")
    _check_compute_cap()

    stage_start = time.monotonic()
    downstream = _downstream_search_validation(finalists, portfolio, arrays)
    _record_stage_time("downstream_lightgbm", time.monotonic() - stage_start)
    _append_research_status(
        "search-validation LightGBM checkpoint",
        "write report and manifest",
        f"evaluated {len(downstream['candidate_results'])} variants on {downstream['common_validation_decision_rows']:,} common validation decisions with {len(SEEDS)} reservoir seeds crossed with {len(LGBM_SEEDS)} LightGBM seeds; active runner time {_elapsed_seconds():.1f}s.",
    )
    _save_run_state("downstream_complete", "write_report")
    _check_compute_cap()

    report_start = time.monotonic()
    total_seconds = _elapsed_seconds()
    _write_report(
        historical_export, architecture_search, input_search, common_leaders,
        input_leaders, finalists, portfolio, downstream, arrays, total_seconds,
    )
    _record_stage_time("report_and_manifest", time.monotonic() - report_start)
    total_seconds = _elapsed_seconds()
    _write_report(
        historical_export, architecture_search, input_search, common_leaders,
        input_leaders, finalists, portfolio, downstream, arrays, total_seconds,
    )
    _save_run_state(
        "complete_search_validation_only", "none",
        architecture_trials=architecture_search["attempted_trials"],
        input_trials=input_search["attempted_trials"],
        finalist_count=len(finalists),
        portfolio_member_count=len(portfolio["selected_configs"]),
        downstream_candidates=len(downstream["candidate_results"]),
        active_seconds=total_seconds,
    )
    _append_research_status(
        "r4 complete on search-validation only",
        "none",
        f"report `{REPORT_PATH.relative_to(ROOT).as_posix()}`; manifest `{(OUT / 'audit_manifest.json').relative_to(ROOT).as_posix()}`; active runner time {total_seconds:.1f}s of 14,400s.",
    )
    print(json.dumps({
        "status": "complete_search_validation_only",
        "active_seconds": total_seconds,
        "architecture_trials": architecture_search["attempted_trials"],
        "input_trials": input_search["attempted_trials"],
        "finalist_count": len(finalists),
        "portfolio_member_count": len(portfolio["selected_configs"]),
        "downstream_candidate_count": len(downstream["candidate_results"]),
        "report_path": REPORT_PATH.relative_to(ROOT).as_posix(),
        "manifest_path": (OUT / "audit_manifest.json").relative_to(ROOT).as_posix(),
        "holdout_rows_materialized": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
