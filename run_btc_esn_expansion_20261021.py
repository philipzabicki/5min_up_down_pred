"""Expanded, development-only ESN search with downstream LightGBM selection."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import time
import ctypes
from ctypes import wintypes
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

import run_btc_esn_architecture_audit_20261020 as engine_runner
from features.esn_reservoir import ESNConfig, INPUT_GROUPS


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_esn_expansion_20261021.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUT = ROOT / CONFIG["output_directory"]
REPORT_PATH = ROOT / CONFIG["report_path"]
FAMILIES = tuple(CONFIG["search"]["architecture_families"])
TOPOLOGIES = tuple(CONFIG["search"]["topologies"])
UNIT_BUDGETS = tuple(int(value) for value in CONFIG["search"]["total_unit_budgets"])
UNIT_BUDGETS_BY_FAMILY = {
    family: tuple(int(value) for value in budgets)
    for family, budgets in CONFIG["search"]["unit_budgets_by_family"].items()
}
SCREEN_SEED = int(CONFIG["screening"]["reservoir_seed"])
RUN_STARTED = None
RUN_PRIOR_SECONDS = 0.0
RUN_IDENTITY = None
RUN_STAGE_DEADLINE = None
RUN_STAGE_NAME = None


class StageBudgetExceeded(TimeoutError):
    pass


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def _resource_snapshot() -> dict:
    class MEMORYSTATUSEX(__import__("ctypes").Structure):
        _fields_ = [
            ("dwLength", __import__("ctypes").c_ulong),
            ("dwMemoryLoad", __import__("ctypes").c_ulong),
            ("ullTotalPhys", __import__("ctypes").c_ulonglong),
            ("ullAvailPhys", __import__("ctypes").c_ulonglong),
            ("ullTotalPageFile", __import__("ctypes").c_ulonglong),
            ("ullAvailPageFile", __import__("ctypes").c_ulonglong),
            ("ullTotalVirtual", __import__("ctypes").c_ulonglong),
            ("ullAvailVirtual", __import__("ctypes").c_ulonglong),
            ("ullAvailExtendedVirtual", __import__("ctypes").c_ulonglong),
        ]
    import ctypes
    memory = MEMORYSTATUSEX()
    memory.dwLength = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("GlobalMemoryStatusEx failed")
    disk = shutil.disk_usage(ROOT)
    try:
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            cwd=ROOT, capture_output=True, text=True, timeout=5, check=False,
        )
        gpu = gpu_result.stdout.strip() if gpu_result.returncode == 0 else "unavailable to nvidia-smi"
    except (OSError, subprocess.TimeoutExpired):
        gpu = "unavailable to nvidia-smi"
    return {
        "cpu_logical_processors": os.cpu_count(),
        "cpu_model": platform.processor(),
        "platform": platform.platform(),
        "physical_memory_bytes": int(memory.ullTotalPhys),
        "available_memory_bytes": int(memory.ullAvailPhys),
        "disk_free_bytes": int(disk.free),
        "gpu_detected": gpu,
        "reservoir_gpu_used": False,
        "workers": 1,
        "native_threads": 4,
    }


def _peak_working_set_bytes():
    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    get_handle = ctypes.windll.kernel32.GetCurrentProcess
    get_handle.restype = wintypes.HANDLE
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    get_memory.restype = wintypes.BOOL
    return int(counters.PeakWorkingSetSize) if get_memory(get_handle(), ctypes.byref(counters), counters.cb) else None


def _elapsed() -> float:
    return RUN_PRIOR_SECONDS + (0.0 if RUN_STARTED is None else time.monotonic() - RUN_STARTED)


def _ledger() -> dict:
    path = OUT / "runtime_ledger.json"
    if not path.exists():
        return {"identity": RUN_IDENTITY, "stages_seconds": {}, "resume_count": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity") != RUN_IDENTITY:
        raise RuntimeError("runtime ledger belongs to another frozen search")
    return payload


def _save_ledger(payload: dict) -> None:
    payload["active_seconds"] = max(float(payload.get("active_seconds", 0.0)), _elapsed())
    _atomic_json(OUT / "runtime_ledger.json", payload)


def _check_total_cap() -> None:
    ledger = _ledger()
    _save_ledger(ledger)
    cap = int(CONFIG["search"]["whole_experiment_compute_limit_seconds"])
    if _elapsed() > cap:
        raise TimeoutError(f"frozen 12-hour active-runner cap of {cap} seconds reached")


def _check_compute_cap() -> None:
    _check_total_cap()
    if RUN_STAGE_DEADLINE is not None and _elapsed() >= RUN_STAGE_DEADLINE:
        raise StageBudgetExceeded(f"frozen {RUN_STAGE_NAME} stage budget reached")


def _set_stage_budget(stage_name: str, cap_seconds: int, ledger_stages) -> None:
    global RUN_STAGE_DEADLINE, RUN_STAGE_NAME
    used = sum(float(_ledger().get("stages_seconds", {}).get(stage, 0.0)) for stage in ledger_stages)
    remaining = max(0.0, float(cap_seconds) - used)
    RUN_STAGE_NAME = stage_name
    RUN_STAGE_DEADLINE = _elapsed() + remaining


def _record_time(stage: str, seconds: float) -> None:
    ledger = _ledger()
    active = ledger.pop("in_progress_stage", None)
    if active is not None and active.get("name") != stage:
        raise RuntimeError("runtime ledger has a different in-progress stage")
    stages = ledger.setdefault("stages_seconds", {})
    stages[stage] = float(stages.get(stage, 0.0) + max(0.0, seconds))
    _save_ledger(ledger)


def _begin_stage_segment(stage: str) -> None:
    ledger = _ledger()
    active = ledger.get("in_progress_stage")
    if active is not None:
        raise RuntimeError("runtime ledger still has an unfinished stage segment")
    ledger["in_progress_stage"] = {"name": stage, "started_active_seconds": _elapsed()}
    _save_ledger(ledger)


def _recover_interrupted_stage() -> None:
    ledger = _ledger()
    active = ledger.pop("in_progress_stage", None)
    if active is None:
        return
    elapsed = max(0.0, float(ledger.get("active_seconds", 0.0)) - float(active["started_active_seconds"]))
    stages = ledger.setdefault("stages_seconds", {})
    stage = active["name"]
    stages[stage] = float(stages.get(stage, 0.0) + elapsed)
    _save_ledger(ledger)


def _append_status(stage: str, next_step: str, detail: str) -> None:
    timestamp = datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M:%S %Z")
    with (ROOT / "docs/BTC_RESEARCH_STATUS_20261010.md").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            f"\n## ESN expanded search — {timestamp}\n\n"
            f"- Stage: `{stage}`; next resumable step: `{next_step}`.\n"
            f"- Detail: {detail}\n"
            "- Scope: fit through 2024-07-01 and search validation through 2025-07-01; no confirmation or holdout reads.\n"
        )


def _identity(prepared: dict) -> dict:
    return {
        "experiment_id": CONFIG["experiment_id"],
        "runtime_source_commit": _git_commit(),
        "config_sha256": _sha256(CONFIG_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "engine_sha256": _sha256(ROOT / "features/esn_reservoir.py"),
        "downstream_helper_sha256": _sha256(ROOT / "run_btc_esn_architecture_audit_20261020.py"),
        "benchmark_sha256": _sha256(ROOT / CONFIG["benchmark_path"]),
        "prepared_search_identity": prepared["identity"],
        "baseline_logits_sha256": _sha256(ROOT / CONFIG["reused_baseline_logits_path"]),
        "baseline_logits_metadata_sha256": _sha256(ROOT / CONFIG["reused_baseline_metadata_path"]),
        "search_validation_end_exclusive_utc": CONFIG["split"]["search_validation_end_exclusive_utc"],
        "confirmation_read": False,
        "holdout_evaluation": False,
    }


def _log_draw(rng, bounds, *, parent_value=None, mutation_sd=0.45):
    lower, upper = (math.log10(float(value)) for value in bounds)
    if parent_value is None:
        return float(10.0 ** rng.uniform(lower, upper))
    value = math.log10(float(parent_value)) + float(rng.normal(0.0, mutation_sd))
    return float(10.0 ** np.clip(value, lower, upper))


def _sample_groups(rng, previous=None):
    names = tuple(INPUT_GROUPS)
    if previous is not None:
        chosen = set(previous)
        for name in names:
            if rng.random() < 0.25:
                if name in chosen:
                    chosen.remove(name)
                else:
                    chosen.add(name)
        if chosen:
            return tuple(name for name in names if name in chosen)
    chosen = {name for name in names if rng.random() < 0.48}
    if not chosen:
        chosen.add(str(rng.choice(names)))
    return tuple(name for name in names if name in chosen)


def _topology_for(rng, family, total_units, requested, parent=None):
    allowed = ["ring", "sparse", "orthogonal", "block_sparse"]
    if family == "single" and total_units <= int(CONFIG["search"]["legacy_dense_total_unit_limit"]):
        allowed.append("dense")
    if total_units > int(CONFIG["search"]["orthogonal_total_unit_limit"]):
        allowed.remove("orthogonal")
    if parent is not None and rng.random() < 0.65:
        prior = parent.topologies[0]
        if prior in allowed:
            return prior
    return requested if requested in allowed else str(rng.choice(allowed))


def _structure(rng, family, total_units, parent=None):
    if parent is not None and rng.random() < 0.65:
        if family == "single":
            return 1, 1, 1
        if family in {"parallel", "deep"}:
            return parent.reservoir_count, 1, 1
        return parent.reservoir_count, parent.branch_count, parent.layer_count
    if family == "single":
        return 1, 1, 1
    if family == "parallel":
        options = [int(value) for value in CONFIG["search"]["parallel_branch_choices"] if int(value) * 2 <= total_units]
        count = int(rng.choice(options))
        return count, count, 1
    if family == "deep":
        options = [int(value) for value in CONFIG["search"]["deep_layer_choices"] if int(value) * 2 <= total_units]
        count = int(rng.choice(options))
        return count, 1, count
    options = [
        (branches, layers)
        for branches in CONFIG["search"]["parallel_deep_branch_choices"]
        for layers in CONFIG["search"]["parallel_deep_layer_choices"]
        if branches * layers <= min(24, total_units // 2)
    ]
    branches, layers = options[int(rng.integers(0, len(options)))]
    return int(branches * layers), int(branches), int(layers)


def _sample_config(rng, family, unit_budget, topology, dynamics_mode, *, parent=None):
    if parent is not None and rng.random() < 0.45:
        family = parent.architecture
    unit_choices = UNIT_BUDGETS_BY_FAMILY[family]
    if parent is not None and rng.random() < 0.72:
        parent_units = parent.total_units
        neighbor = int(rng.choice([0.5, 1.0, 2.0]))
        target_units = max(min(unit_choices), min(max(unit_choices), parent_units * neighbor))
        unit_budget = min(unit_choices, key=lambda value: abs(math.log2(value) - math.log2(target_units)))
    else:
        unit_budget = min(unit_choices, key=lambda value: abs(math.log2(value) - math.log2(int(unit_budget))))
    unit_budget = int(unit_budget)
    actual_topology = _topology_for(rng, family, unit_budget, topology, parent)
    count, branches, layers = _structure(rng, family, unit_budget, parent)
    base_size, remainder = divmod(unit_budget, count)
    sizes = tuple(base_size + int(index < remainder) for index in range(count))
    if min(sizes) < 2:
        raise ValueError("sampled architecture has fewer than two units per component")

    branch_inputs = []
    previous_inputs = None if parent is None else [groups for groups in parent.input_groups_by_component if groups]
    for branch in range(branches):
        previous = previous_inputs[branch % len(previous_inputs)] if previous_inputs else None
        branch_inputs.append(_sample_groups(rng, previous))
    component_groups = []
    if family == "single":
        component_groups = [branch_inputs[0]]
    elif family == "deep":
        component_groups = [branch_inputs[0]] + [()] * (count - 1)
    elif family == "parallel":
        component_groups = list(branch_inputs)
    else:
        component_groups = [groups if layer == 0 else () for groups in branch_inputs for layer in range(layers)]
    all_groups = tuple(group for group in INPUT_GROUPS if any(group in item for item in component_groups))

    topology_mode = "independent" if rng.random() < 0.18 and count > 1 else "shared"
    if topology_mode == "independent":
        component_topologies = tuple(
            _topology_for(rng, family, unit_budget, str(rng.choice(TOPOLOGIES)))
            for _ in range(count)
        )
    else:
        component_topologies = (actual_topology,) * count

    if parent is not None:
        first = parent
        parent_leak = first.leak_rates[0]
        parent_radius = first.spectral_radii[0]
        parent_input = first.input_scalings[0]
        parent_bias = first.bias_scalings[0]
    else:
        parent_leak = parent_radius = parent_input = parent_bias = None
    leak = _log_draw(rng, CONFIG["search"]["leak_rate_range"], parent_value=parent_leak)
    radius = _log_draw(rng, CONFIG["search"]["spectral_radius_range"], parent_value=parent_radius)
    input_scale = _log_draw(rng, CONFIG["search"]["input_scaling_range"], parent_value=parent_input)
    positive_bias = CONFIG["search"]["positive_bias_scaling_range"]
    if parent is not None and parent_bias == 0.0 and rng.random() < 0.35:
        bias = 0.0
    else:
        bias = 0.0 if rng.random() < 0.2 else _log_draw(rng, positive_bias, parent_value=parent_bias if parent_bias else None)
    component_leaks = component_radii = component_inputs = component_biases = None
    progression = None
    if dynamics_mode == "progression":
        multiplier = float(10.0 ** rng.uniform(-0.5, 0.5))
        layer_slots = [index % layers if family in {"deep", "parallel_deep"} else index for index in range(count)]
        component_leaks = tuple(float(np.clip(leak * multiplier ** slot, 1e-4, 1.0)) for slot in layer_slots)
        progression = {"initial_leak": leak, "multiplier": multiplier}
    elif dynamics_mode == "independent":
        component_leaks = tuple(_log_draw(rng, CONFIG["search"]["leak_rate_range"], parent_value=leak) for _ in range(count))
        component_radii = tuple(_log_draw(rng, CONFIG["search"]["spectral_radius_range"], parent_value=radius) for _ in range(count))
        component_inputs = tuple(_log_draw(rng, CONFIG["search"]["input_scaling_range"], parent_value=input_scale) for _ in range(count))
        component_biases = tuple(0.0 if rng.random() < 0.2 else _log_draw(rng, positive_bias, parent_value=bias if bias else None) for _ in range(count))

    degree = int(rng.choice(CONFIG["search"]["sparse_average_degree_choices"]))
    config = ESNConfig(
        architecture=family,
        reservoir_size=min(sizes),
        reservoir_count=count,
        branch_count=branches if family == "parallel_deep" else 1,
        layer_count=layers if family == "parallel_deep" else 1,
        leak_rate=leak,
        spectral_radius=radius,
        input_scaling=input_scale,
        bias_scaling=bias,
        topology=actual_topology,
        connectivity=min(1.0, degree / max(sizes)),
        input_groups=all_groups,
        component_sizes=sizes,
        component_input_groups=tuple(tuple(groups) for groups in component_groups),
        component_topologies=component_topologies,
        component_leak_rates=component_leaks,
        component_spectral_radii=component_radii,
        component_input_scalings=component_inputs,
        component_bias_scalings=component_biases,
        sparse_degree=degree,
    )
    return config, {
        "architecture_family": family,
        "unit_budget_requested": unit_budget,
        "unit_budget_actual": config.total_units,
        "unit_stratum": _unit_stratum(config.total_units),
        "topology_mode": topology_mode,
        "dynamics_mode": dynamics_mode,
        "leak_progression": progression,
        "input_groups_by_component": [list(groups) for groups in config.input_groups_by_component],
    }


def _unit_stratum(units):
    if units < 64:
        return "32-63"
    if units < 256:
        return "64-255"
    if units < 1024:
        return "256-1023"
    return "1024-2048"


def _candidate(config, phase, ordinal, metadata):
    return {
        "candidate_id": engine_runner._config_key(config),
        "phase": phase,
        "ordinal": int(ordinal),
        "sampling": metadata,
        "effective_config": engine_runner._config_record(config),
    }


def _initial_pool():
    path = OUT / "candidate_pool_initial.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != RUN_IDENTITY:
            raise RuntimeError("initial candidate pool belongs to another frozen search")
        return payload["candidates"]
    rng = np.random.default_rng(int(CONFIG["search"]["initial_candidate_sampler_seed"]))
    candidates = []
    seen = set()
    for family_index, family in enumerate(FAMILIES):
        family_budgets = UNIT_BUDGETS_BY_FAMILY[family]
        for ordinal in range(int(CONFIG["screening"]["initial_configs_per_family"])):
            for attempt in range(100):
                budget = family_budgets[(ordinal + 2 * family_index) % len(family_budgets)]
                topology = TOPOLOGIES[(ordinal + family_index) % len(TOPOLOGIES)]
                mode = ("shared", "progression", "independent")[(ordinal + family_index) % 3]
                config, metadata = _sample_config(rng, family, budget, topology, mode)
                candidate_id = engine_runner._config_key(config)
                if candidate_id not in seen:
                    break
            else:
                raise RuntimeError("could not generate a unique stratified initial candidate")
            seen.add(candidate_id)
            metadata.update({"initial_family_quota_index": ordinal, "topology_cycle": topology})
            candidates.append(_candidate(config, "initial_stratified", len(candidates), metadata))
    _atomic_json(path, {"identity": RUN_IDENTITY, "candidates": candidates})
    return candidates


def _read_screen_results():
    folder = OUT / "screen_results"
    results = []
    if not folder.exists():
        return results
    for path in folder.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("identity") != RUN_IDENTITY:
            raise RuntimeError("screen result belongs to another frozen search")
        results.append(row)
    return results


def _screen_one(candidate, arrays):
    output_path = OUT / "screen_results" / f"{candidate['candidate_id']}.json"
    if output_path.exists():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        if saved.get("identity") != RUN_IDENTITY:
            raise RuntimeError("screen result identity mismatch")
        return saved
    config = engine_runner._config_from_record(candidate["effective_config"])
    started = time.monotonic()
    try:
        generated = engine_runner._stream_screen_features(config, SCREEN_SEED, arrays, capture_probe=True)
        raw_logits = arrays["baseline_logits"]
        labels = arrays["target"]
        availability = arrays["label_available"]
        corr_idx = generated["correction_indices"]
        val_idx = generated["validation_indices"]
        corr_ok = (
            np.isfinite(raw_logits[corr_idx])
            & (availability[corr_idx] <= engine_runner.CORRECTION_END_NS + engine_runner.DECISION_PURGE_NS)
            & generated["correction_ready"]
        )
        val_ok = np.isfinite(raw_logits[val_idx]) & generated["validation_ready"]
        if corr_ok.sum() < 1_000 or val_ok.sum() < 1_000:
            raise ValueError("warm-up leaves too few common correction or search-validation rows")
        probe_start = time.monotonic()
        delta = engine_runner._logistic_delta(
            labels[corr_idx[corr_ok]], raw_logits[corr_idx[corr_ok]], generated["correction_features"][corr_ok],
            labels[val_idx[val_ok]], raw_logits[val_idx[val_ok]], generated["validation_features"][val_ok],
        )
        row = {
            "candidate_id": candidate["candidate_id"],
            "phase": candidate["phase"],
            "sampling": candidate["sampling"],
            "effective_config": candidate["effective_config"],
            "status": "screened",
            "screen_level": "probe_logistic_search_validation",
            "probe_delta_logloss": float(delta),
            "probe_fit_rows": int(corr_ok.sum()),
            "probe_validation_rows": int(val_ok.sum()),
            "reservoir_seed": SCREEN_SEED,
            "warmup_steps": int(generated["warmup_steps"]),
            "warmup_diagnostics": generated["warmup_diagnostics"],
            "saturation_fraction": float(generated["saturation_fraction"]),
            "actual_recurrent_edges": generated["actual_recurrent_edges"],
            "probe_fit_seconds": float(time.monotonic() - probe_start),
            "elapsed_seconds": float(time.monotonic() - started),
        }
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        row = {
            "candidate_id": candidate["candidate_id"],
            "phase": candidate["phase"],
            "sampling": candidate["sampling"],
            "effective_config": candidate["effective_config"],
            "status": "infeasible_or_unstable",
            "screen_level": "feasibility_stability_probe",
            "rejection_reason": str(exc),
            "elapsed_seconds": float(time.monotonic() - started),
        }
    _atomic_json(output_path, {"identity": RUN_IDENTITY, **row})
    return row


def _screen_pool(candidates, arrays, *, phase_name, target_valid=None):
    screened = _read_screen_results()
    known = {row["candidate_id"] for row in screened}
    for index, candidate in enumerate(candidates):
        if candidate["candidate_id"] in known:
            continue
        ledger = _ledger()
        screen_time = float(ledger.get("stages_seconds", {}).get("screening", 0.0))
        if screen_time >= int(CONFIG["search"]["screening_time_limit_seconds"]):
            return False
        if target_valid is not None and sum(row["status"] == "screened" for row in screened) >= target_valid:
            return True
        before = time.monotonic()
        _begin_stage_segment("screening")
        try:
            result = _screen_one(candidate, arrays)
        finally:
            _record_time("screening", time.monotonic() - before)
        screened.append(result)
        known.add(candidate["candidate_id"])
        _check_total_cap()
        if len(known) % 10 == 0:
            successes = sum(row["status"] == "screened" for row in screened)
            _append_status(
                phase_name, "resume_screening_or_downstream",
                f"{len(known)} unique configs attempted; {successes} valid screens; active runner time {_elapsed():.1f}s.",
            )
            print(json.dumps({
                "event": "screen_progress", "phase": phase_name,
                "attempted": len(known), "valid": successes,
                "active_seconds": _elapsed(), "last_seconds": duration,
            }), flush=True)
    return True


def _adaptive_pool(initial_results):
    path = OUT / "candidate_pool_adaptive.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != RUN_IDENTITY:
            raise RuntimeError("adaptive candidate pool belongs to another frozen search")
        return payload["candidates"]
    rng = np.random.default_rng(int(CONFIG["search"]["adaptive_candidate_sampler_seed"]))
    valid = [row for row in initial_results if row["status"] == "screened"]
    elite_count = max(1, int(math.ceil(0.25 * len(valid))))
    parents = sorted(valid, key=lambda row: row["probe_delta_logloss"], reverse=True)[:elite_count]
    if not parents:
        raise RuntimeError("initial stratified phase produced no valid configuration to adapt from")
    seen = {row["candidate_id"] for row in _read_screen_results()}
    maximum = int(CONFIG["screening"]["maximum_unique_proposals"]) - len(seen)
    exploitation_fraction = float(CONFIG["screening"]["adaptive_exploitation_fraction"])
    candidates = []
    for ordinal in range(maximum):
        exploit = ordinal < int(maximum * exploitation_fraction)
        parent_row = parents[int(rng.integers(0, len(parents)))] if exploit else None
        parent_config = None if parent_row is None else engine_runner._config_from_record(parent_row["effective_config"])
        family = parent_config.architecture if parent_config is not None else FAMILIES[ordinal % len(FAMILIES)]
        family_budgets = UNIT_BUDGETS_BY_FAMILY[family]
        budget = family_budgets[int(rng.integers(0, len(family_budgets)))]
        topology = str(rng.choice(TOPOLOGIES)) if parent_config is None else parent_config.topologies[0]
        mode = str(rng.choice(("shared", "progression", "independent")))
        for _ in range(100):
            config, metadata = _sample_config(rng, family, budget, topology, mode, parent=parent_config)
            candidate_id = engine_runner._config_key(config)
            if candidate_id not in seen:
                break
        else:
            raise RuntimeError("could not generate a unique adaptive candidate")
        seen.add(candidate_id)
        metadata.update({
            "adaptive_source": "probe_elite_mutation" if exploit else "family_stratified_random",
            "parent_candidate_id": None if parent_row is None else parent_row["candidate_id"],
            "adaptive_exploitation_fraction": exploitation_fraction,
        })
        candidates.append(_candidate(config, "adaptive", ordinal, metadata))
    _atomic_json(path, {"identity": RUN_IDENTITY, "candidates": candidates})
    return candidates


def _exploration_ids(screened, target_count):
    target_fraction = float(CONFIG["downstream"]["probe_independent_exploration_fraction"])
    quota = min(target_count, int(math.ceil(target_count * target_fraction)))
    rng = np.random.default_rng(int(CONFIG["downstream"]["exploration_sampler_seed"]))
    families = [family for family in FAMILIES if any(row["sampling"]["architecture_family"] == family for row in screened)]
    selected = []
    remaining = {family: [row for row in screened if row["sampling"]["architecture_family"] == family] for family in families}
    for rows in remaining.values():
        rng.shuffle(rows)
    while len(selected) < quota and any(remaining.values()):
        for family in families:
            if len(selected) >= quota:
                break
            rows = remaining[family]
            if not rows:
                continue
            bucket_counts = {}
            for selected_id in selected:
                prior = next(row for row in screened if row["candidate_id"] == selected_id)
                key = (prior["sampling"]["unit_stratum"], prior["sampling"]["dynamics_mode"])
                bucket_counts[key] = bucket_counts.get(key, 0) + 1
            key = min(
                {(row["sampling"]["unit_stratum"], row["sampling"]["dynamics_mode"]) for row in rows},
                key=lambda value: bucket_counts.get(value, 0),
            )
            candidate = next(row for row in rows if (row["sampling"]["unit_stratum"], row["sampling"]["dynamics_mode"]) == key)
            rows.remove(candidate)
            selected.append(candidate["candidate_id"])
    return selected


def _downstream_pool(screened):
    target = min(int(CONFIG["downstream"]["target_distinct_configs"]), len(screened))
    explore_ids = _exploration_ids(screened, target)
    remaining = [row for row in screened if row["candidate_id"] not in set(explore_ids)]
    remaining.sort(key=lambda row: row["probe_delta_logloss"], reverse=True)
    selected_ids = list(explore_ids)
    selected_ids.extend(row["candidate_id"] for row in remaining[:target - len(selected_ids)])
    by_id = {row["candidate_id"]: row for row in screened}
    selected = [by_id[candidate_id] for candidate_id in selected_ids]
    return selected, {
        "target_count": int(target),
        "selected_count": len(selected),
        "probe_independent_count": len(explore_ids),
        "probe_independent_fraction": len(explore_ids) / max(1, len(selected)),
        "probe_independent_candidate_ids": explore_ids,
        "screen_ranked_candidate_ids": [row["candidate_id"] for row in remaining[:target - len(explore_ids)]],
        "exploration_rule": "stratified random by architecture family, unit-size stratum, and dynamics mode; probe score is not read in this branch",
    }


def _exploratory_result_rows(selected, downstream):
    by_id = {row["candidate_id"]: row for row in selected}
    return [
        {
            **by_id[candidate_id],
            "downstream": score,
            "downstream_delta_logloss": float(score["mean_delta_logloss"]),
        }
        for candidate_id, score in downstream["candidate_results"].items()
        if candidate_id in by_id
    ]


def _partial_downstream_result(stage_name):
    progress_path = OUT / f"downstream_{stage_name}_progress.json"
    if not progress_path.exists():
        return None
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("progress_identity", {}).get("identity") != RUN_IDENTITY:
        raise RuntimeError("partial downstream checkpoint belongs to another frozen search")
    expected_pairs = len(engine_runner.SEEDS) * len(engine_runner.LGBM_SEEDS)
    summaries = {}
    for name, rows in progress.get("candidate_records", {}).items():
        pairs = {(int(row["reservoir_seed"]), int(row["lightgbm_seed"])) for row in rows}
        if len(rows) != expected_pairs or len(pairs) != expected_pairs:
            continue
        deltas = np.asarray([row["delta_logloss"] for row in rows], dtype=np.float64)
        summaries[name] = {
            "mean_delta_logloss": float(np.mean(deltas)),
            "std_delta_logloss_all_pairs": float(np.std(deltas, ddof=0)),
            "positive_pair_count": int(np.sum(deltas > 0.0)),
            "negative_pair_count": int(np.sum(deltas < 0.0)),
            "pair_count": len(rows),
            "rows": rows,
        }
    return {
        "status": "search_validation_downstream_partial_stage_budget",
        "candidate_results": summaries,
        "baseline_metrics_by_lightgbm_seed": progress.get("baseline_metrics", {}),
        "completed_candidate_count": len(summaries),
    }


def _historical_ring_candidate():
    report_row = {
        "candidate_id": "historical_single_ring_1x128_44e399a127401eb4e7ff0fa6",
        "phase": "historical_control",
        "ordinal": 0,
        "sampling": {"architecture_family": "single", "unit_stratum": "64-255", "dynamics_mode": "shared"},
        "effective_config": engine_runner._config_record(ESNConfig(
            architecture="single", reservoir_size=128, reservoir_count=1,
            topology="ring", leak_rate=0.12875637661749686,
            spectral_radius=1.1513008725745195,
            input_scaling=0.2580663119097347,
            bias_scaling=0.11590666009961913,
            input_groups=tuple(INPUT_GROUPS),
        )),
    }
    report_row["candidate_id"] = engine_runner._config_key(
        engine_runner._config_from_record(report_row["effective_config"]),
    )
    report_row["historical_candidate_id"] = "44e399a127401eb4e7ff0fa6"
    report_row["historical_trial_number"] = 61
    return report_row


def _finalists(exploratory_rows):
    best_by_family = {}
    for row in exploratory_rows:
        family = row["effective_config"]["architecture"]
        current = best_by_family.get(family)
        if current is None or row["downstream_delta_logloss"] > current["downstream_delta_logloss"]:
            best_by_family[family] = row
    selected = [best_by_family[family] for family in FAMILIES if family in best_by_family]
    control = _historical_ring_candidate()
    if all(row["candidate_id"] != control["candidate_id"] for row in selected):
        selected.append(control)
    return selected


def _pair_variants(finalists, exploratory_rows):
    by_family = {}
    for row in exploratory_rows:
        family = row["effective_config"]["architecture"]
        current = by_family.get(family)
        if current is None or row["downstream_delta_logloss"] > current["downstream_delta_logloss"]:
            by_family[family] = row
    finalists_by_id = {row["candidate_id"]: row for row in finalists}
    singles = [row for row in finalists if row["effective_config"]["architecture"] == "single"]
    single = max(singles, key=lambda row: next((x["downstream_delta_logloss"] for x in exploratory_rows if x["candidate_id"] == row["candidate_id"]), float("-inf")))
    pairs = []
    for family in ("parallel", "deep", "parallel_deep"):
        other = by_family.get(family)
        if other is not None and other["candidate_id"] in finalists_by_id:
            pairs.append((f"pair_single_{family}", [single["candidate_id"], other["candidate_id"]]))
    if len(pairs) < int(CONFIG["downstream"]["maximum_portfolio_pairs"]):
        parallel, deep = by_family.get("parallel"), by_family.get("deep")
        if parallel is not None and deep is not None and parallel["candidate_id"] != deep["candidate_id"]:
            if parallel["candidate_id"] in finalists_by_id and deep["candidate_id"] in finalists_by_id:
                pairs.append(("pair_parallel_deep", [parallel["candidate_id"], deep["candidate_id"]]))
    return pairs[:int(CONFIG["downstream"]["maximum_portfolio_pairs"])]


def _portfolio_trace(first, extension, pairs, finalists):
    scores = first["candidate_results"]
    best_single = max(
        (row["candidate_id"] for row in finalists if row["effective_config"]["architecture"] == "single"),
        key=lambda key: scores[key]["mean_delta_logloss"],
    )
    best_single_score = scores[best_single]
    trace = []
    pair_rows = []
    for name, members in pairs:
        result = scores[name]
        marginal = float(result["mean_delta_logloss"] - best_single_score["mean_delta_logloss"])
        qualifies = result["mean_delta_logloss"] > 0 and marginal > 0
        row = {
            "variant": name,
            "members": members,
            "score_before": best_single_score["mean_delta_logloss"],
            "score_after": result["mean_delta_logloss"],
            "marginal_delta_logloss": marginal,
            "std_over_seed_pairs": result["std_delta_logloss_all_pairs"],
            "positive_seed_pairs": result["positive_pair_count"],
            "feature_count": result["rows"][0]["feature_count"],
            "decision": "eligible_pair_start" if qualifies else "rejected_pair_not_better_than_single_or_baseline",
        }
        trace.append(row)
        pair_rows.append(row)
    eligible_pairs = [row for row in pair_rows if row["decision"] == "eligible_pair_start"]
    best_pair = max(eligible_pairs, key=lambda row: row["score_after"]) if eligible_pairs else None
    start_members = [best_single] if best_single_score["mean_delta_logloss"] > 0 else []
    start_score = best_single_score["mean_delta_logloss"] if start_members else 0.0
    if best_pair is not None:
        start_members = list(best_pair["members"])
        start_score = float(best_pair["score_after"])
        trace.append({**best_pair, "decision": "selected_pair_start"})

    extension_scores = {} if extension is None else extension["candidate_results"]
    best_extension = None
    if best_pair is not None:
        for name in extension_scores:
            if name.startswith("portfolio_3_member_"):
                result = extension_scores[name]
                marginal = float(result["mean_delta_logloss"] - start_score)
                qualifies = result["mean_delta_logloss"] > 0 and marginal > 0
                row = {
                    "variant": name,
                    "members": result.get("members", []),
                    "score_before": start_score,
                    "score_after": result["mean_delta_logloss"],
                    "marginal_delta_logloss": marginal,
                    "std_over_seed_pairs": result["std_delta_logloss_all_pairs"],
                    "positive_seed_pairs": result["positive_pair_count"],
                    "feature_count": result["rows"][0]["feature_count"],
                    "decision": "eligible_positive_downstream_addition" if qualifies else "rejected_nonpositive_downstream_marginal",
                }
                trace.append(row)
                if qualifies and (best_extension is None or row["score_after"] > best_extension["score_after"]):
                    best_extension = row
        if best_extension is not None:
            best_extension["decision"] = "selected_full_set_downstream_addition"
            start_members = list(best_extension["members"])
            start_score = float(best_extension["score_after"])

    choices = [{"name": "baseline", "members": [], "score": 0.0}]
    choices.append({"name": "best_single", "members": [best_single], "score": float(best_single_score["mean_delta_logloss"])})
    for row in pair_rows:
        choices.append({"name": row["variant"], "members": row["members"], "score": float(row["score_after"])})
    if best_extension is not None:
        choices.append({"name": best_extension["variant"], "members": best_extension["members"], "score": float(best_extension["score_after"])})
    selected = max(choices, key=lambda row: row["score"])
    return {
        "best_single_id": best_single,
        "best_single_score": float(best_single_score["mean_delta_logloss"]),
        "trace": trace,
        "choices": choices,
        "selected": selected,
        "selection_rule": "maximize full downstream LightGBM mean paired delta versus the matched baseline; baseline score is zero; a pair/addition must improve the whole set and have positive combined delta",
    }


def _measure_finalist_latencies(finalists, arrays):
    measurements = {}
    rows = min(int(CONFIG["downstream"]["latency_sample_count"]), len(arrays["raw_opened"]))
    timestamps = pd.to_datetime(np.asarray(arrays["raw_opened"][:rows]), unit="ns", utc=True)
    valid = np.asarray(arrays["raw_valid"][:rows], dtype=bool)
    for finalist in finalists:
        config = engine_runner._config_from_record(finalist["effective_config"])
        selected = np.asarray([engine_runner.INPUT_COLUMNS.index(name) for name in config.selected_columns], dtype=np.int64)
        values = np.asarray(arrays["raw_x"][:rows, :][:, selected], dtype=np.float32)
        measurements[finalist["candidate_id"]] = {}
        for seed in CONFIG["downstream"]["finalist_reservoir_seeds"]:
            bundle_path = OUT / "bundles" / f"{finalist['candidate_id']}_seed_{seed}.json"
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            objects = engine_runner._bundle_objects(bundle)
            model = engine_runner.ESNEngine(config, int(seed), objects["scaler"])
            elapsed_ns = []
            projected_ns = []
            for timestamp, observation, is_valid in zip(timestamps, values, valid):
                before = time.perf_counter_ns()
                state = model.update_one(timestamp, observation, valid=bool(is_valid))
                duration = time.perf_counter_ns() - before
                if state is not None:
                    elapsed_ns.append(duration)
                    before_projection = time.perf_counter_ns()
                    objects["projection"].transform(state.reshape(1, -1))
                    projected_ns.append(time.perf_counter_ns() - before_projection)
            measurements[finalist["candidate_id"]][str(seed)] = {
                "esn_update_us": {
                    "p50": float(np.quantile(elapsed_ns, 0.50) / 1_000.0),
                    "p95": float(np.quantile(elapsed_ns, 0.95) / 1_000.0),
                    "p99": float(np.quantile(elapsed_ns, 0.99) / 1_000.0),
                },
                "pca_projection_us": {
                    "p50": float(np.quantile(projected_ns, 0.50) / 1_000.0),
                    "p95": float(np.quantile(projected_ns, 0.95) / 1_000.0),
                    "p99": float(np.quantile(projected_ns, 0.99) / 1_000.0),
                },
                "valid_updates": len(elapsed_ns),
                "fragment_rows": rows,
            }
    return measurements


def _boundary_summary(rows, level, probe_top_ids=None):
    values = [row for row in rows if row.get("status") == "screened"]
    if level == "downstream":
        values = [row for row in rows if row.get("downstream") is not None]
    score_key = "probe_delta_logloss" if level == "probe" else "downstream_delta_logloss"
    top_count = max(1, int(math.ceil(0.1 * len(values)))) if values else 0
    ranked = sorted(values, key=lambda row: row.get(score_key, float("-inf")), reverse=True)
    top_ids = {row["candidate_id"] for row in ranked[:top_count]} if probe_top_ids is None else probe_top_ids
    ranges = {
        "leak_rate": CONFIG["search"]["leak_rate_range"],
        "spectral_radius": CONFIG["search"]["spectral_radius_range"],
        "input_scaling": CONFIG["search"]["input_scaling_range"],
        "positive_bias_scaling": CONFIG["search"]["positive_bias_scaling_range"],
    }
    def count_boundary(group, name, bounds):
        lower = upper = top_lower = top_upper = zeros = 0
        for row in group:
            components = row["effective_config"].get("components", [])
            component_values = [
                float(component["bias_scaling"] if name == "positive_bias_scaling" else component[name])
                for component in components
            ]
            near_lower = near_upper = False
            for value in component_values:
                if name == "positive_bias_scaling" and value == 0.0:
                    continue
                fraction = (math.log(value) - math.log(bounds[0])) / (math.log(bounds[1]) - math.log(bounds[0]))
                near_lower |= fraction <= 0.05
                near_upper |= fraction >= 0.95
            is_top = row["candidate_id"] in top_ids
            lower += int(near_lower)
            upper += int(near_upper)
            top_lower += int(is_top and near_lower)
            top_upper += int(is_top and near_upper)
            zeros += int(name == "positive_bias_scaling" and any(value == 0.0 for value in component_values))
        count = len(group)
        top_count = sum(row["candidate_id"] in top_ids for row in group)
        return {
            "count": count,
            "near_lower": lower,
            "near_lower_rate": None if not count else lower / count,
            "near_upper": upper,
            "near_upper_rate": None if not count else upper / count,
            "top_decile_count": top_count,
            "top_decile_near_lower": top_lower,
            "top_decile_near_lower_rate": None if not top_count else top_lower / top_count,
            "top_decile_near_upper": top_upper,
            "top_decile_near_upper_rate": None if not top_count else top_upper / top_count,
            "positive_bias_zero_count": zeros if name == "positive_bias_scaling" else None,
        }

    boundary = {}
    for name, bounds in ranges.items():
        boundary[name] = {
            "range": bounds,
            "boundary_band_fraction_of_log_range": 0.05,
            **count_boundary(values, name, bounds),
            "by_family": {
                family: count_boundary(
                    [row for row in values if row["effective_config"]["architecture"] == family], name, bounds,
                )
                for family in FAMILIES
            },
            "by_total_units": {
                str(units): count_boundary(
                    [row for row in values if int(row["effective_config"]["total_units"]) == units], name, bounds,
                )
                for units in UNIT_BUDGETS
            },
        }
    return {
        "level": level,
        "count": len(values),
        "top_decile_count": top_count,
        "families": {
            family: sum(row["effective_config"]["architecture"] == family for row in values)
            for family in FAMILIES
        },
        "unit_budgets": {
            str(units): sum(int(row["effective_config"]["total_units"]) == units for row in values)
            for units in UNIT_BUDGETS
        },
        "continuous_parameter_boundaries": boundary,
    }


def _artifact_sizes():
    output = {}
    for folder_name in ("screen_features", "bundles", "final_features", "screen_results"):
        folder = OUT / folder_name
        output[folder_name] = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()) if folder.exists() else 0
    output["total_bytes"] = sum(output.values())
    return output


def _write_report(screen_rows, selected_pool, pool_selection, broad, finalist_rows,
                  first_final, extension, portfolio, finalist_meta, benchmark, resources):
    exploratory_rows = _exploratory_result_rows(selected_pool, broad)
    all_screened = [row for row in screen_rows if row["status"] == "screened"]
    family_counts = {}
    for family in FAMILIES:
        members = [row for row in all_screened if row["effective_config"]["architecture"] == family]
        eval_rows = [row for row in exploratory_rows if row["effective_config"]["architecture"] == family]
        family_counts[family] = {
            "screened": len(members),
            "infeasible_or_unstable": sum(row["effective_config"]["architecture"] == family and row["status"] != "screened" for row in screen_rows),
            "downstream_evaluated": len(eval_rows),
            "downstream_mean_delta_logloss": None if not eval_rows else float(np.mean([row["downstream_delta_logloss"] for row in eval_rows])),
            "downstream_positive_candidates": sum(row["downstream_delta_logloss"] > 0 for row in eval_rows),
        }
    probe_by_id = {row["candidate_id"]: row for row in all_screened}
    paired_ranks = [
        (probe_by_id[row["candidate_id"]]["probe_delta_logloss"], row["downstream_delta_logloss"])
        for row in exploratory_rows if row["candidate_id"] in probe_by_id
    ]
    rho = None if len(paired_ranks) < 3 else float(spearmanr(*zip(*paired_ranks)).statistic)
    boundary_probe = _boundary_summary(screen_rows, "probe")
    boundary_downstream = _boundary_summary(exploratory_rows, "downstream")
    selected = portfolio["selected"]
    selected_name = selected["name"]
    score_results = dict(first_final["candidate_results"])
    if extension is not None:
        score_results.update(extension["candidate_results"])
    final_result_rows = []
    for name, result in score_results.items():
        final_result_rows.append({
            "name": name,
            "mean_delta_logloss": result["mean_delta_logloss"],
            "std_delta_logloss": result["std_delta_logloss_all_pairs"],
            "positive_seed_pairs": result["positive_pair_count"],
            "pair_count": result["pair_count"],
            "feature_count": result["rows"][0]["feature_count"],
            "per_seed_rows": result["rows"],
        })
    baseline_rows = first_final["baseline_metrics_by_lightgbm_seed"]
    for row in finalist_rows:
        for seed in CONFIG["downstream"]["finalist_reservoir_seeds"]:
            bundle_path = OUT / "bundles" / f"{row['candidate_id']}_seed_{seed}.json"
            if bundle_path.exists():
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                row.setdefault("final_seed_diagnostics", {})[str(seed)] = {
                    "warmup_steps": bundle["warmup_steps"],
                    "warmup": bundle["warmup_diagnostics"],
                    "actual_recurrent_edges": bundle["actual_recurrent_edges"],
                    "saturation_fraction": bundle["saturation_fraction"],
                }
    artifact_sizes = _artifact_sizes()
    active_seconds = _elapsed()

    lines = [
        "# Expanded ESN search — 2026-10-21",
        "",
        "> Development search result only. It is not an independent confirmation of ESN advantage.",
        "",
        "## Frozen scope and budget",
        "",
        f"- Fit-core: `{CONFIG['split']['fit_core_start_utc']}` through exclusive `{CONFIG['split']['fit_core_end_exclusive_utc']}`; correction-fit: `{CONFIG['split']['correction_fit_start_utc']}` through exclusive `{CONFIG['split']['correction_fit_end_exclusive_utc']}`; search validation: `{CONFIG['split']['search_validation_start_utc']}` through exclusive `{CONFIG['split']['search_validation_end_exclusive_utc']}`.",
        "- Existing prepared cache and baseline logits were reused. No rows, predictions, or results after the frozen search-validation cutoff were read. No confirmation, holdout, live configuration, or Polymarket replay was used.",
        f"- Active runner time: **{active_seconds:.1f}s** of the frozen {CONFIG['search']['whole_experiment_compute_limit_seconds']}s cap; resume count `{_ledger().get('resume_count', 0)}`.",
        f"- Resources at preflight: `{resources['cpu_model']}` on `{resources['platform']}`, {resources['cpu_logical_processors']} logical CPUs, {resources['physical_memory_bytes'] / 2**30:.1f} GiB RAM, {resources['available_memory_bytes'] / 2**30:.1f} GiB free; detected GPU `{resources['gpu_detected']}`; reservoir generation on CPU, LightGBM device `{CONFIG['baseline']['lightgbm']['device_type']}`; one worker, four native threads.",
        f"- Process peak working set: `{_peak_working_set_bytes()}` bytes; artifact sizes: `{json.dumps(artifact_sizes, sort_keys=True)}`.",
        "",
        "## Coverage",
        "",
        f"- Valid unique screens: **{len(all_screened)}**; infeasible/unstable attempts: **{sum(row['status'] != 'screened' for row in screen_rows)}**.",
        f"- Downstream screening candidates: **{len(exploratory_rows)}**; probe independent: **{pool_selection['probe_independent_count']} ({pool_selection['probe_independent_fraction']:.1%})**.",
        f"- Finalist configurations: **{len(finalist_rows)}**; final candidates use 3 reservoir seeds × 3 LightGBM seeds.",
        "",
        "| Family | Eligible total-unit budgets |",
        "|---|---|",
        *[
            f"| `{family}` | {', '.join(map(str, UNIT_BUDGETS_BY_FAMILY[family]))} |"
            for family in FAMILIES
        ],
        "",
        "| Family | Valid screens | Downstream candidates | Mean downstream ΔLL | Positive candidates |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, count in family_counts.items():
        mean_value = count["downstream_mean_delta_logloss"]
        lines.append(f"| `{family}` | {count['screened']} | {count['downstream_evaluated']} | {'—' if mean_value is None else f'{mean_value:+.9f}'} | {count['downstream_positive_candidates']} |")
    lines.extend([
        "",
        "Screen feasibility/stability and the logistic probe are separate from downstream LightGBM quality. The probe did not reject candidates from downstream; a stratified random 30% of downstream slots was chosen without consulting probe ranks.",
        "",
        "## Baseline and final downstream comparison",
        "",
        "Primary metric is paired `LL(BASE70) − LL(BASE70 + ESN)` on the same search-validation decisions and target. Positive values favor ESN. The baseline is allowed to win.",
        "",
        "| Variant | Mean paired ΔLL | SD across 9 seed pairs | Positive pairs | Features |",
        "|---|---:|---:|---:|---:|",
    ])
    lines.append(f"| `BASE70` | +0.000000000 | 0.000000000 | — | 70 |")
    for row in sorted(final_result_rows, key=lambda item: item["mean_delta_logloss"], reverse=True):
        lines.append(f"| `{row['name']}` | {row['mean_delta_logloss']:+.9f} | {row['std_delta_logloss']:.9f} | {row['positive_seed_pairs']}/{row['pair_count']} | {row['feature_count']} |")
    lines.extend([
        "",
        f"Selected under the frozen full-downstream rule: **`{selected_name}`**, mean ΔLL `{selected['score']:+.9f}`. A portfolio was not retained when it lost to the best single or baseline.",
        "",
        "### Finalists and exact components",
        "",
    ])
    for row in finalist_rows:
        cfg = row["effective_config"]
        lines.append(f"- `{row['candidate_id']}` — `{cfg['architecture']}` / {cfg['total_units']} units; components `{json.dumps(cfg['components'], ensure_ascii=False, separators=(',', ':'))}`.")
        lines.append(f"  Input columns by component: `{json.dumps([item['input_columns'] for item in cfg['components']], ensure_ascii=False)}`; final representation is 8 PCA features per candidate.")
        if row.get("final_seed_diagnostics"):
            lines.append(f"  Final seed warm-up, saturation, and actual recurrent edge counts: `{json.dumps(row['final_seed_diagnostics'], sort_keys=True)}`.")
    lines.extend([
        "",
        "### Paired per-seed results",
        "",
        "| Variant | Reservoir seed | LightGBM seed | Baseline LL | Augmented LL | Paired ΔLL |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for variant in final_result_rows:
        for row in variant["per_seed_rows"]:
            base_metric = baseline_rows[str(row["lightgbm_seed"])]
            lines.append(f"| `{variant['name']}` | {row['reservoir_seed']} | {row['lightgbm_seed']} | {base_metric['logloss']:.9f} | {row['metrics']['logloss']:.9f} | {row['delta_logloss']:+.9f} |")
    lines.extend(["", "### Finalist inference latency", ""])
    for candidate_id, seed_results in finalist_meta.get("engine_latency", {}).items():
        lines.append(f"- `{candidate_id}`: `{json.dumps(seed_results, sort_keys=True)}`")
    lines.extend([
        "",
        "Single-row LightGBM inference latency on the same 256 sampled validation decisions:",
        "",
        "| Variant | Reservoir seed | LightGBM seed | p50 (µs) | p95 (µs) | p99 (µs) | Samples |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in final_result_rows:
        for row in variant["per_seed_rows"]:
            latency = row.get("single_row_latency")
            if latency is not None:
                lines.append(
                    f"| `{variant['name']}` | {row['reservoir_seed']} | {row['lightgbm_seed']} | {latency['p50_us']:.2f} | {latency['p95_us']:.2f} | {latency['p99_us']:.2f} | {latency['sample_count']} |"
                )
    lines.extend([
        "",
        "## Portfolio selection trace",
        "",
        "Each pair and each proposed addition was scored as a complete BASE70+ESN LightGBM feature set with common seed pairs.",
        "",
        "| Variant | Members | Score before | Score after | Marginal ΔLL | Decision |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in portfolio["trace"]:
        lines.append(f"| `{row['variant']}` | `{','.join(row.get('members', []))}` | {row['score_before']:+.9f} | {row['score_after']:+.9f} | {row['marginal_delta_logloss']:+.9f} | `{row['decision']}` |")
    lines.extend([
        "",
        "## Screening quality and ranking agreement",
        "",
        f"Spearman correlation between probe and one-seed/three-LightGBM downstream ranks for the same {len(paired_ranks)} candidates: `{rho}`. This is descriptive only because objectives differ and candidates were development-selected.",
        "",
        "### Probe boundary coverage",
        "",
        f"`{json.dumps(boundary_probe, sort_keys=True)}`",
        "",
        "### Downstream boundary coverage",
        "",
        f"`{json.dumps(boundary_downstream, sort_keys=True)}`",
        "",
        "## Benchmark, runtime, and artifacts",
        "",
        f"Fit-only benchmark report: `{CONFIG['benchmark_report_path']}`. It includes per-family and unit-budget initialization, generation throughput, one-step latency, checkpoint size/restore, and projection/probe fit cost. Screening and downstream stage timings are in `runtime_ledger.json`; feature artifact sizes are in `artifact_sizes.json`.",
        "",
        "## Limits and next-step guidance",
        "",
        f"Dense recurrence was capped at {CONFIG['search']['legacy_dense_total_unit_limit']} total units and orthogonal recurrence at {CONFIG['search']['orthogonal_total_unit_limit']}. The benchmark measured 2,048-unit deep and parallel_deep state generation at 32.85–34.03s and 17.14–17.62s per 43,200-row fragment; extrapolating to the 2,660,546-row cache and three finalist seeds gives about 105 and 54 minutes for those candidates alone. Both multi-layer families are therefore capped at 1,024 units to protect the 3-hour finalist/portfolio reserve. Benchmark-only 2,048-unit deep variants are not search results.",
        "- Read boundary diagnostics by probe, downstream, family, and total-unit budget. Do not infer a family advantage from its single best candidate.",
        "- This search does not establish an ESN advantage outside the explored configurations or replace independent confirmation.",
        "- Runner and resume command: `python run_btc_esn_expansion_20261021.py`.",
        f"- Frozen manifest: `data/analysis/polymarket/BTC/btc_esn_expansion_20261021/frozen_run_manifest.json`; final commit is recorded in the repository status and Git history.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "family_counts": family_counts,
        "ranking_spearman": rho,
        "boundary_probe": boundary_probe,
        "boundary_downstream": boundary_downstream,
        "final_selection": selected,
        "artifact_sizes": artifact_sizes,
    }


def main():
    global RUN_STARTED, RUN_PRIOR_SECONDS, RUN_IDENTITY, RUN_STAGE_DEADLINE, RUN_STAGE_NAME
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    resources = _resource_snapshot()
    if resources["available_memory_bytes"] < 8 * 2**30:
        raise RuntimeError("less than 8 GiB RAM is available; preserving existing workloads and deferring the run")
    prepared_path = ROOT / CONFIG["reused_search_cache_directory"] / "prepared_manifest.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared.get("materialization_cutoff_utc") != "2025-07-01T00:00:00+00:00" or prepared.get("holdout_rows_materialized") != 0:
        raise RuntimeError("reused cache does not match the frozen development-only scope")
    RUN_IDENTITY = _identity(prepared)
    summary_path = OUT / "run_summary.json"
    if summary_path.exists():
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        if completed.get("identity") == RUN_IDENTITY and completed.get("status") == "complete_search_validation_only":
            if not REPORT_PATH.exists():
                raise RuntimeError("completed-run summary exists but the final report is missing")
            print(json.dumps(completed, indent=2, sort_keys=True), flush=True)
            return
    engine_runner.CONFIG = CONFIG
    engine_runner.CONFIG_PATH = CONFIG_PATH
    engine_runner.OUT = OUT
    engine_runner.RUN_IDENTITY = RUN_IDENTITY
    engine_runner.SEEDS = tuple(int(value) for value in CONFIG["downstream"]["exploratory_reservoir_seeds"])
    engine_runner.LGBM_SEEDS = tuple(int(value) for value in CONFIG["downstream"]["lightgbm_seeds"])
    engine_runner.RUN_STARTED = None
    engine_runner.RUN_PRIOR_SECONDS = 0.0
    engine_runner.SCALER_CACHE = {}
    engine_runner.THREAD_LIMITER = threadpool_limits(limits=4)
    engine_runner._check_compute_cap = _check_compute_cap
    ledger_path = OUT / "runtime_ledger.json"
    if ledger_path.exists():
        prior = json.loads(ledger_path.read_text(encoding="utf-8"))
        if prior.get("identity") != RUN_IDENTITY:
            raise RuntimeError("runtime ledger identity mismatch; refusing unsafe resume")
        RUN_PRIOR_SECONDS = float(prior.get("active_seconds", 0.0))
        prior["resume_count"] = int(prior.get("resume_count", 0)) + 1
        _atomic_json(ledger_path, prior)
    else:
        RUN_PRIOR_SECONDS = 0.0
        _atomic_json(ledger_path, {"identity": RUN_IDENTITY, "stages_seconds": {}, "resume_count": 0})
    RUN_STARTED = time.monotonic()
    RUN_STAGE_DEADLINE = None
    RUN_STAGE_NAME = None
    engine_runner.RUN_STARTED = RUN_STARTED
    engine_runner.RUN_PRIOR_SECONDS = RUN_PRIOR_SECONDS
    _recover_interrupted_stage()
    _check_total_cap()

    frozen_path = OUT / "frozen_run_manifest.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen.get("identity") != RUN_IDENTITY:
            raise RuntimeError("frozen run manifest identity mismatch")
    else:
        frozen = {
            "identity": RUN_IDENTITY,
            "status": "frozen_before_any_new_trial",
            "frozen_at_warsaw": datetime.now(ZoneInfo("Europe/Warsaw")).isoformat(),
            "config": CONFIG,
            "benchmark_sha256": _sha256(ROOT / CONFIG["benchmark_path"]),
            "development_only": True,
            "confirmation_read": False,
            "holdout_rows_materialized": 0,
        }
        _atomic_json(frozen_path, frozen)
    _append_status(
        "frozen preflight complete", "initial stratified screening",
        f"resource snapshot {resources}; frozen protocol targets {CONFIG['screening']['target_valid_unique_configs']} valid screens and {CONFIG['downstream']['target_distinct_configs']} downstream candidates; active cap 43,200s.",
    )
    print(json.dumps({"event": "preflight", "identity": RUN_IDENTITY, "resources": resources}, indent=2), flush=True)
    arrays = engine_runner._load_search_data(RUN_IDENTITY)
    engine_runner._SCREEN_ARRAYS = arrays
    initial = _initial_pool()
    _set_stage_budget("screening", int(CONFIG["search"]["screening_time_limit_seconds"]), ("screening",))
    try:
        continue_search = _screen_pool(initial, arrays, phase_name="initial stratified screening")
    except StageBudgetExceeded:
        continue_search = False
        _append_status("screening budget reached", "exploratory downstream", f"Active screening time reached its frozen cap of {CONFIG['search']['screening_time_limit_seconds']}s; completed screen checkpoints are retained.")
    screen_rows = _read_screen_results()
    if continue_search:
        valid_initial = [row for row in screen_rows if row["phase"] == "initial_stratified" and row["status"] == "screened"]
        if valid_initial:
            adaptive = _adaptive_pool(valid_initial)
            _set_stage_budget("screening", int(CONFIG["search"]["screening_time_limit_seconds"]), ("screening",))
            try:
                _screen_pool(
                    adaptive, arrays, phase_name="adaptive probe-guided screening",
                    target_valid=int(CONFIG["screening"]["target_valid_unique_configs"]),
                )
            except StageBudgetExceeded:
                _append_status("screening budget reached", "exploratory downstream", f"Active screening time reached its frozen cap of {CONFIG['search']['screening_time_limit_seconds']}s; completed screen checkpoints are retained.")
        screen_rows = _read_screen_results()
    screened = [row for row in screen_rows if row["status"] == "screened"]
    _atomic_json(OUT / "screen_summary.json", {
        "identity": RUN_IDENTITY,
        "status": "screening_complete_or_budget_reached",
        "valid_unique_configs": len(screened),
        "infeasible_or_unstable_configs": sum(row["status"] != "screened" for row in screen_rows),
        "screening_target": CONFIG["screening"]["target_valid_unique_configs"],
        "screening_rows_by_family": {
            family: sum(row["effective_config"]["architecture"] == family for row in screened)
            for family in FAMILIES
        },
    })
    if not screened:
        raise RuntimeError("screening produced no feasible candidate; no downstream ranking is available")
    selected_pool, pool_selection = _downstream_pool(screened)
    _atomic_json(OUT / "downstream_pool_selection.json", {"identity": RUN_IDENTITY, **pool_selection})
    _append_status(
        "screening complete or budget reached", "exploratory LightGBM downstream",
        f"{len(screened)} valid unique configs; {pool_selection['selected_count']} downstream slots selected, including {pool_selection['probe_independent_count']} ({pool_selection['probe_independent_fraction']:.1%}) stratified independently of probe rank.",
    )
    _set_stage_budget("exploratory downstream", int(CONFIG["search"]["exploratory_downstream_time_limit_seconds"]), ("exploratory_downstream",))
    before = time.monotonic()
    _begin_stage_segment("exploratory_downstream")
    try:
        broad = engine_runner._downstream_search_validation(
            selected_pool,
            {"selected_candidate_ids": [], "portfolio_variants": {}},
            arrays,
            stage_name="exploratory",
        )
    except StageBudgetExceeded:
        broad = _partial_downstream_result("exploratory")
        if broad is None:
            raise RuntimeError("exploratory downstream stage reached its cap before a resumable progress checkpoint")
        _append_status("exploratory downstream budget reached", "finalist LightGBM evaluation", f"{broad['completed_candidate_count']} complete candidates were available at the frozen {CONFIG['search']['exploratory_downstream_time_limit_seconds']}s exploratory cap.")
    finally:
        _record_time("exploratory_downstream", time.monotonic() - before)
    exploratory_rows = _exploratory_result_rows(selected_pool, broad)
    _atomic_json(OUT / "exploratory_downstream_ranking.json", {
        "identity": RUN_IDENTITY,
        "screen_to_downstream": [
            {
                "candidate_id": row["candidate_id"],
                "architecture": row["effective_config"]["architecture"],
                "total_units": row["effective_config"]["total_units"],
                "probe_delta_logloss": next(item["probe_delta_logloss"] for item in selected_pool if item["candidate_id"] == row["candidate_id"]),
                "downstream_delta_logloss": row["downstream_delta_logloss"],
            }
            for row in exploratory_rows
        ],
    })
    finalist_rows = _finalists(exploratory_rows)
    if not exploratory_rows:
        raise RuntimeError("no complete downstream candidates were available to select development finalists")
    pairs = _pair_variants(finalist_rows, exploratory_rows)
    pair_map = {name: members for name, members in pairs}
    _atomic_json(OUT / "finalist_and_pair_protocol.json", {
        "identity": RUN_IDENTITY,
        "finalists": finalist_rows,
        "candidate_pairs": pairs,
        "maximum_pairs": CONFIG["downstream"]["maximum_portfolio_pairs"],
        "selection_basis": "one candidate per architecture family selected by exploratory downstream LightGBM, plus the prior ring 1x128 reference; pairs are limited and selected from these downstream-ranked candidates",
    })

    engine_runner.SEEDS = tuple(int(value) for value in CONFIG["downstream"]["finalist_reservoir_seeds"])
    engine_runner.LGBM_SEEDS = tuple(int(value) for value in CONFIG["downstream"]["finalist_lightgbm_seeds"])
    _set_stage_budget(
        "finalist and portfolio reserve",
        int(CONFIG["search"]["finalist_and_portfolio_reserve_seconds"]),
        ("finalist_downstream", "portfolio_extension"),
    )
    before = time.monotonic()
    _begin_stage_segment("finalist_downstream")
    try:
        first_final = engine_runner._downstream_search_validation(
            finalist_rows,
            {"selected_candidate_ids": [], "portfolio_variants": pair_map},
            arrays,
            stage_name="finalists",
        )
    finally:
        _record_time("finalist_downstream", time.monotonic() - before)

    extension = None
    first_scores = first_final["candidate_results"]
    singles = [row for row in finalist_rows if row["effective_config"]["architecture"] == "single"]
    best_single = max(singles, key=lambda row: first_scores[row["candidate_id"]]["mean_delta_logloss"])
    best_single_delta = first_scores[best_single["candidate_id"]]["mean_delta_logloss"]
    qualifying_pairs = [
        (name, members) for name, members in pairs
        if first_scores[name]["mean_delta_logloss"] > 0
        and first_scores[name]["mean_delta_logloss"] > best_single_delta
    ]
    if qualifying_pairs and len(finalist_rows) > 2:
        best_pair_name, best_pair_members = max(
            qualifying_pairs, key=lambda item: first_scores[item[0]]["mean_delta_logloss"],
        )
        additions = [row["candidate_id"] for row in finalist_rows if row["candidate_id"] not in set(best_pair_members)]
        max_additions = int(CONFIG["downstream"]["maximum_greedy_additions"])
        triplets = {
            f"portfolio_3_member_{index + 1}": best_pair_members + [candidate_id]
            for index, candidate_id in enumerate(additions[:max_additions])
        }
        if triplets:
            _set_stage_budget(
                "finalist and portfolio reserve",
                int(CONFIG["search"]["finalist_and_portfolio_reserve_seconds"]),
                ("finalist_downstream", "portfolio_extension"),
            )
            before = time.monotonic()
            _begin_stage_segment("portfolio_extension")
            try:
                extension = engine_runner._downstream_search_validation(
                    finalist_rows,
                    {"selected_candidate_ids": [], "portfolio_variants": triplets},
                    arrays,
                    stage_name="portfolio_extension",
                )
            finally:
                _record_time("portfolio_extension", time.monotonic() - before)
            for name, members in triplets.items():
                extension["candidate_results"][name]["members"] = members
            _atomic_json(OUT / "portfolio_extension_results.json", {"identity": RUN_IDENTITY, **extension})

    portfolio = _portfolio_trace(first_final, extension, pairs, finalist_rows)
    _atomic_json(OUT / "portfolio_selection.json", {"identity": RUN_IDENTITY, **portfolio})
    _atomic_json(OUT / "final_downstream_comparison.json", {
        "identity": RUN_IDENTITY,
        "evaluation_period": first_final["evaluation_period"],
        "common_fit_rows": first_final["common_fit_rows"],
        "common_validation_decision_rows": first_final["common_validation_decision_rows"],
        "reservoir_seeds": list(engine_runner.SEEDS),
        "lightgbm_seeds": list(engine_runner.LGBM_SEEDS),
        "baseline_metrics_by_lightgbm_seed": first_final["baseline_metrics_by_lightgbm_seed"],
        "candidate_results": first_final["candidate_results"],
        "portfolio_extension_results": {} if extension is None else extension["candidate_results"],
        "portfolio_selection": portfolio,
        "state_generation_seconds_by_candidate": first_final["state_generation_seconds_by_candidate"],
        "lightgbm_fit_seconds": first_final["lightgbm_fit_seconds"] + (0.0 if extension is None else extension["lightgbm_fit_seconds"]),
    })
    _record_time("reporting", 0.0)
    benchmark_path = ROOT / CONFIG["benchmark_path"]
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    engine_latency = _measure_finalist_latencies(finalist_rows, arrays)
    report_summary = _write_report(
        screen_rows, selected_pool, pool_selection, broad, finalist_rows,
        first_final, extension, portfolio, {"engine_latency": engine_latency}, benchmark, resources,
    )
    _atomic_json(OUT / "finalist_latency.json", {"identity": RUN_IDENTITY, "results": engine_latency})
    _atomic_json(OUT / "range_diagnostics.json", {
        "identity": RUN_IDENTITY,
        "probe": report_summary["boundary_probe"],
        "downstream": report_summary["boundary_downstream"],
        "family_counts": report_summary["family_counts"],
    })
    _atomic_json(OUT / "artifact_sizes.json", {"identity": RUN_IDENTITY, **report_summary["artifact_sizes"]})
    run_summary = {
        "identity": RUN_IDENTITY,
        "status": "complete_search_validation_only",
        "active_seconds": _elapsed(),
        "valid_unique_screens": len(screened),
        "downstream_candidates": len(exploratory_rows),
        "finalist_count": len(finalist_rows),
        "portfolio_pair_count": len(pairs),
        "selected_variant": portfolio["selected"],
        "holdout_rows_materialized": 0,
        "confirmation_read": False,
        "report_path": REPORT_PATH.relative_to(ROOT).as_posix(),
    }
    _atomic_json(OUT / "run_summary.json", run_summary)
    _append_status(
        "development search complete", "none",
        f"{len(screened)} valid unique screens, {len(exploratory_rows)} downstream configurations, {len(finalist_rows)} finalists; selected `{portfolio['selected']['name']}` at ΔLL {portfolio['selected']['score']:+.9f}; report `{REPORT_PATH.relative_to(ROOT).as_posix()}`.",
    )
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
