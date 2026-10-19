"""Frozen, resumable BASE70 capacity tuning on the saved ten-fold matrix."""
from __future__ import annotations

import ctypes
import gc
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

import run_btc_base70_information_value_20261018 as base70
import run_btc_base70_seed_stability_20261018 as seed_stability
import run_btc_preopen_model_update as model_update
import run_btc_preopen_research as research
import run_btc_preopen_taker_flow_20261018 as taker
from utils.cv_objective import ObjectiveAlignedCVCallback, cv_objective_series
from utils.metrics import make_lightgbm_binary_logloss_eval


ROOT = Path(__file__).resolve().parent
STUDY_ID = "btc_base70_capacity_tuning_20261019"
PROTOCOL_PATH = ROOT / "configs/research/btc_base70_capacity_tuning_20261019.json"
OUTPUT_DIR = ROOT / f"data/analysis/polymarket/BTC/{STUDY_ID}"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
STUDY_DB_PATH = OUTPUT_DIR / "study.sqlite3"
REPORT_PATH = ROOT / f"reports/btc_preopen/{STUDY_ID}.md"
VERSIONED_MANIFEST_PATH = ROOT / f"reports/btc_preopen/{STUDY_ID}_manifest.json"
STATUS_PATH = ROOT / "docs/BTC_RESEARCH_STATUS_20261010.md"
STATUS_BEGIN = "<!-- BTC_BASE70_CAPACITY_TUNING_20261019_BEGIN -->"
STATUS_END = "<!-- BTC_BASE70_CAPACITY_TUNING_20261019_END -->"

MAX_CONFIGURATIONS = 32
MAX_ESTIMATORS = 500
EARLY_STOPPING_ROUNDS = 50
STUDY_BUDGET_SECONDS = 1800.0
HARD_STOP_RESERVE_SECONDS = 120.0
SEARCH_SCHEDULE_SEED = 20261019
MODEL_SEEDS = (20261015, 20261016, 20261017)
PARAMETER_LEVELS = {
    "num_leaves": (15, 31, 63, 127),
    "min_data_in_leaf": (1024, 4096, 12089, 32768),
    "lambda_l1": (0.0, 1.0, 10.0, 30.0),
    "lambda_l2": (1.0, 10.0, 30.0, 100.0),
}
SEED_PARAMETER_NAMES = ("seed", "feature_fraction_seed", "data_random_seed", "bagging_seed")
TRAINING_FEATURE_METADATA = [
    "_source_row_index", "Opened", "nominal_decision_at", "target_label_available_at",
    "target_preopen_5m_up_binance_index_proxy", "is_polymarket_preopen_decision",
]
OOF_ID_COLUMNS = [
    "_source_row_index", "Opened", "nominal_decision_at", "target_label_available_at",
    "target_preopen_5m_up_binance_index_proxy", "market_start_utc", "fold_id",
]
OFFICIAL_CUTOFF = pd.Timestamp("2026-07-01T00:00:00Z")


class StudyBudgetExpired(RuntimeError):
    def __init__(self, iteration: int, elapsed_seconds: float):
        super().__init__(
            f"study budget safety boundary reached after CV iteration {iteration} "
            f"at {elapsed_seconds:.3f}s"
        )
        self.iteration = int(iteration)
        self.elapsed_seconds = float(elapsed_seconds)


class StudyClock:
    def __init__(self, elapsed_before_session: float):
        self.elapsed_before_session = float(elapsed_before_session)
        self.session_start = time.perf_counter()

    def elapsed(self) -> float:
        return self.elapsed_before_session + time.perf_counter() - self.session_start


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _record(path: Path) -> dict:
    return {"path": _relative(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _verify_record(record: dict) -> bool:
    if not isinstance(record, dict) or not {"path", "sha256", "size_bytes"}.issubset(record):
        return False
    path = ROOT / record["path"]
    return (
        path.is_file()
        and path.stat().st_size == int(record["size_bytes"])
        and _sha256(path) == record["sha256"]
    )


def _verify_records(records: list[dict]) -> bool:
    return bool(records) and all(_verify_record(record) for record in records)


def _gf4_multiply(left: int, right: int) -> int:
    """Multiply two GF(4) elements represented by polynomial bits, mod x^2+x+1."""
    if left == 0 or right == 0:
        return 0
    product = 0
    for bit in range(2):
        if right & (1 << bit):
            product ^= left << bit
    if product & 0b100:
        product ^= 0b111
    return product


def _balanced_candidate_schedule() -> list[dict]:
    values = [PARAMETER_LEVELS[name] for name in PARAMETER_LEVELS]
    rows = []
    # Two disjoint strength-2 OA(16, 4^4) arrays give 32 unique configurations.
    for shift in (0, 1):
        for a, b in itertools.product(range(4), repeat=2):
            symbols = (a ^ shift, b, a ^ b, a ^ _gf4_multiply(2, b))
            rows.append({name: values[i][symbols[i]] for i, name in enumerate(PARAMETER_LEVELS)})
    if len({tuple(row.values()) for row in rows}) != 32:
        raise RuntimeError("The frozen orthogonal-array schedule is not 32 unique configurations")
    rng = np.random.default_rng(SEARCH_SCHEDULE_SEED)
    order = rng.permutation(len(rows))
    selected = [rows[int(index)] for index in order[: MAX_CONFIGURATIONS - 1]]
    for name, levels in PARAMETER_LEVELS.items():
        counts = {value: sum(row[name] == value for row in selected) for value in levels}
        if min(counts.values()) < 7 or max(counts.values()) > 8:
            raise RuntimeError(f"The frozen 31-configuration schedule is unbalanced for {name}: {counts}")
    return selected


def _effective_params(base_params: dict, candidate: dict, seed: int) -> dict:
    result = dict(base_params)
    result.update(candidate)
    result["num_leaves"] = int(result["num_leaves"])
    result["min_data_in_leaf"] = int(result["min_data_in_leaf"])
    result["lambda_l1"] = float(result["lambda_l1"])
    result["lambda_l2"] = float(result["lambda_l2"])
    for name in SEED_PARAMETER_NAMES:
        result[name] = int(seed)
    return result


def _available_versions() -> dict:
    packages = ("lightgbm", "numpy", "pandas", "pyarrow", "scikit-learn", "optuna")
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def _memory_status() -> dict:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    if os.name != "nt":
        return {"platform": os.name, "physical_memory": "unavailable"}
    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {"platform": os.name, "physical_memory": "unavailable"}
    gib = float(2**30)
    return {
        "total_memory_gib": round(status.ullTotalPhys / gib, 3),
        "available_memory_gib": round(status.ullAvailPhys / gib, 3),
        "memory_load_pct": int(status.dwMemoryLoad),
    }


def _system_snapshot() -> dict:
    gpu = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        gpu = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "nvidia-smi unavailable"
    processes = []
    if os.name == "nt":
        command = (
            "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
            "Select-Object ProcessId,Name,WorkingSetSize,KernelModeTime,UserModeTime | ConvertTo-Json -Compress"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True, text=True, timeout=15, check=True,
            )
            raw = json.loads(result.stdout) if result.stdout.strip() else []
            if isinstance(raw, dict):
                raw = [raw]
            processes = [
                {key: item.get(key) for key in ("ProcessId", "Name", "WorkingSetSize", "KernelModeTime", "UserModeTime")}
                for item in raw
            ]
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            processes = [{"status": "process snapshot unavailable"}]
    return {
        "captured_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "cpu_logical_processors": os.cpu_count(),
        **_memory_status(),
        "free_disk_gib": round(shutil.disk_usage(ROOT).free / 2**30, 3),
        "gpu_query": gpu,
        "python_processes": processes,
        "lightgbm_threads": int(taker.THREADS),
        "parallel_training_workers": 1,
    }


def _source_paths() -> dict[str, Path]:
    return {
        "matrix": seed_stability.COMMON_MATRIX_PATH,
        "folds": seed_stability.COMMON_FOLDS_PATH,
        "common_summary": seed_stability.COMMON_SUMMARY_PATH,
        "single_cv_summary": seed_stability.BASE_CV_SUMMARY_PATH,
        "single_oof": seed_stability.BASE_OOF_PATH,
        "base70_information_manifest": base70.MANIFEST_PATH,
        "seed_stability_manifest": seed_stability.MANIFEST_PATH,
        "seed_stability_config": seed_stability.CONFIG_PATH,
        "taker_flow_manifest": seed_stability.FLOW_DIR / "run_manifest.json",
        "base70_information_predictions": seed_stability.BASE_PREDICTIONS_PATH,
        "execution_calendar": seed_stability.EXECUTION_CALENDAR_PATH,
        "runner_seed_stability": Path(seed_stability.__file__),
        "runner_taker_flow": Path(taker.__file__),
        "runner_base70_information": Path(base70.__file__),
        "runner_research": Path(research.__file__),
        "runner_model_update": Path(model_update.__file__),
        "cv_objective": ROOT / "utils/cv_objective.py",
        "metrics": ROOT / "utils/metrics.py",
    }


def _protocol_source_summary(cv_summary: dict, common_summary: dict) -> dict:
    feature_names = list(cv_summary["feature_order"])
    if len(feature_names) != 70 or len(set(feature_names)) != 70:
        raise RuntimeError("The source BASE70 feature order is not exactly 70 unique features")
    if int(cv_summary["rows"]) != 3_174_805 or int(common_summary["rows_after_shared_complete_feature_mask"]) != 3_174_805:
        raise RuntimeError("The frozen common BASE70 matrix row count changed")
    if not np.isfinite(float(cv_summary["decision_weight"])) or not np.isclose(
        float(cv_summary["auxiliary_row_weight"]), (1.0 - float(cv_summary["decision_weight"])) / 4.0,
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("The saved decision/auxiliary weight rule is invalid")
    if len(common_summary["common_fold_fingerprints"]) != 2:
        raise RuntimeError("The common fold fingerprints are incomplete")
    return {
        "rows": int(cv_summary["rows"]),
        "features": len(feature_names),
        "feature_order": feature_names,
        "target": "target_preopen_5m_up_binance_index_proxy",
        "decision_rows": int(cv_summary["decision_rows"]),
        "oof_rows_expected": int(cv_summary["oof_rows"]),
        "fold_count": 10,
        "folds_path": _relative(seed_stability.COMMON_FOLDS_PATH),
        "folds_sha256": _sha256(seed_stability.COMMON_FOLDS_PATH),
        "common_fold_fingerprints": common_summary["common_fold_fingerprints"],
        "weights_sha256": cv_summary["training_weight_sha256"],
        "pre_mask_source_weights_sha256": common_summary["source_training_weight_sha256_expected"],
        "decision_weight": float(cv_summary["decision_weight"]),
        "auxiliary_row_weight": float(cv_summary["auxiliary_row_weight"]),
        "purge": "reuse the ten saved folds; every training label availability is no later than the earliest validation nominal decision",
        "mask": "reuse common_complete_training_rows.parquet exactly; no feature or row regeneration",
    }


def _build_protocol() -> dict:
    paths = _source_paths()
    cv_summary = json.loads(seed_stability.BASE_CV_SUMMARY_PATH.read_text(encoding="utf-8"))
    common_summary = json.loads(seed_stability.COMMON_SUMMARY_PATH.read_text(encoding="utf-8"))
    source_manifest = json.loads(seed_stability.MANIFEST_PATH.read_text(encoding="utf-8"))
    full_features_manifest = json.loads(
        (ROOT / "reports/btc_preopen/btc_base70_full_features_20261018_manifest.json").read_text(encoding="utf-8")
    )
    zv_manifest = json.loads((ROOT / "reports/btc_preopen/btc_preopen_zv_ablation_20261018_manifest.json").read_text(encoding="utf-8"))
    base64_cv_path = ROOT / "data/analysis/polymarket/BTC/btc_preopen_zv_ablation_20261018/BASE64_cv_summary.json"
    base64_cv = json.loads(base64_cv_path.read_text(encoding="utf-8"))
    if base64_cv["params"] != cv_summary["params"]:
        raise RuntimeError("BASE64 and BASE70 did not inherit the same effective model parameters")
    if not int(cv_summary["best_iteration"]) == 31 or int(cv_summary["maximum_estimators"]) != MAX_ESTIMATORS:
        raise RuntimeError("Unexpected historical BASE70 iteration/maximum-estimator settings")
    if int(cv_summary["early_stopping_rounds"]) != EARLY_STOPPING_ROUNDS:
        raise RuntimeError("The inherited BASE70 early-stopping patience changed")
    expected_control_params = source_manifest["frozen_comparison"]["specification"]["model_parameters_all_five"]["SINGLE"]["lightgbm_params_passed"]
    if expected_control_params != cv_summary["params"]:
        raise RuntimeError("The source seed-stability control params differ from BASE70 B0")
    if isinstance(cv_summary.get("cv_iteration_history"), list):
        original_history_note = "full iteration history is present"
    else:
        original_history_note = str(cv_summary.get("cv_iteration_history", "missing"))
    candidate_schedule = _balanced_candidate_schedule()
    baseline_config = {
        name: cv_summary["params"][name]
        for name in PARAMETER_LEVELS
    }
    configurations = [{"trial_id": 0, "config_id": "control_original", "kind": "control", "parameters": baseline_config}]
    configurations.extend(
        {"trial_id": i, "config_id": f"capacity_{i:02d}", "kind": "candidate", "parameters": values}
        for i, values in enumerate(candidate_schedule, start=1)
    )
    effective = {
        config["config_id"]: {
            str(seed): _effective_params(cv_summary["params"], config["parameters"], seed)
            for seed in MODEL_SEEDS
        }
        for config in configurations
    }
    source_records = {name: _record(path) for name, path in paths.items()}
    source_records["base64_cv_summary"] = _record(base64_cv_path)
    source_records["full_features_manifest"] = _record(ROOT / "reports/btc_preopen/btc_base70_full_features_20261018_manifest.json")
    source_records["zv_ablation_manifest"] = _record(ROOT / "reports/btc_preopen/btc_preopen_zv_ablation_20261018_manifest.json")
    frozen_matrix = full_features_manifest["frozen_comparison"]["specification"]["data"]["matrix"]
    frozen_folds = full_features_manifest["frozen_comparison"]["specification"]["data"]["folds"]
    for name, record in (("matrix", frozen_matrix), ("folds", frozen_folds)):
        actual = source_records[name]
        if actual["sha256"] != record["sha256"] or actual["size_bytes"] != record["size_bytes"]:
            raise RuntimeError(f"The local {name} artifact differs from the frozen BASE70 identity")
    now = pd.Timestamp.now(tz="UTC").isoformat()
    protocol = {
        "experiment_id": STUDY_ID,
        "status": "frozen_before_fit",
        "frozen_at_utc": now,
        "code_identity": {
            "runner_path": _relative(Path(__file__)),
            "runner_sha256": _sha256(Path(__file__)),
        },
        "runtime_versions": _available_versions(),
        "study": {
            "direction": "minimize",
            "sampler": f"fixed balanced strength-2 4-level orthogonal-array subset; randomized order seed {SEARCH_SCHEDULE_SEED}; all 31 candidate configurations frozen before fitting",
            "pruner": "optuna.pruners.NopPruner; no cross-configuration pruning",
            "configurations_max": MAX_CONFIGURATIONS,
            "configurations": configurations,
            "budget_seconds": STUDY_BUDGET_SECONDS,
            "budget_hard_stop_reserve_seconds": HARD_STOP_RESERVE_SECONDS,
            "candidate_space_size": math.prod(len(values) for values in PARAMETER_LEVELS.values()),
            "candidate_configs": MAX_CONFIGURATIONS - 1,
            "model_seeds": list(MODEL_SEEDS),
            "seed_level_score": "minimum over iterations of mean ten-fold unweighted binary log-loss + 0.5 * population standard deviation across folds",
            "trial_score": "arithmetic mean of the three seed-level scores; no seed selection or probability averaging",
            "tie_break": "lowest trial number among equal scores",
            "maximum_estimators": MAX_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "objective_aligned_callback": "utils.cv_objective.ObjectiveAlignedCVCallback; trial=None; same callback controls stopping and ranking",
        },
        "model": {
            "inherited_BASE64_BASE70_params": cv_summary["params"],
            "parameters_varied": PARAMETER_LEVELS,
            "control_parameters": baseline_config,
            "effective_parameters_by_configuration_and_seed": effective,
            "fixed_training_settings": {
                "learning_rate": cv_summary["params"]["learning_rate"],
                "feature_fraction": cv_summary["params"]["feature_fraction"],
                "device_type": cv_summary["params"]["device_type"],
                "max_bin": cv_summary["params"]["max_bin"],
                "num_threads": cv_summary["params"]["num_threads"],
                "validation_metrics_unweighted": True,
                "training_weights_unchanged": True,
                "dataset_rebuilt_per_configuration_and_seed": True,
                "active_lightgbm_seed_parameters": list(SEED_PARAMETER_NAMES),
                "inactive_randomness_unchanged": ["extra_seed", "drop_seed", "objective_seed", "inactive because current GBDT/binary settings do not use them"],
            },
        },
        "data": _protocol_source_summary(cv_summary, common_summary),
        "historical_reuse_review": {
            "BASE70_specific_hyperparameter_search_found": False,
            "BASE64_and_BASE70_effective_params_match": True,
            "base70_b0_cv_summary_best_iteration": int(cv_summary["best_iteration"]),
            "base70_b0_cv_history": original_history_note,
            "base70_saved_boosters_iterations": [int(x) for x in source_manifest["stages"]["training"]["summary"].get("saved_single_fold_iterations", [31] * 10)],
            "saved_31_tree_models_reused_as_early_stopping_runs": False,
            "seed_stability_fits_compatible": False,
            "full_features_fits_compatible": False,
            "reason": "Existing seed-stability/FULL_FEATURES fits fixed 31 rounds without early stopping; B0 summary lacks full history after a post-fit NameError, so the saved 31-tree boosters cannot verify 500-round/patience-50 selection. The control is rerun for all three seeds.",
            "zv_ablation_optuna_used": bool(zv_manifest.get("frozen_comparison", {}).get("specification", {}).get("model_cv", {}).get("optuna", False)),
        },
        "source_records": source_records,
        "holdout_evaluation": False,
        "feature_or_target_generation": False,
        "feature_selection_or_weight_tuning": False,
        "archive_download_or_replay": False,
    }
    protocol = json.loads(json.dumps(protocol, sort_keys=True, default=str))
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        for key in ("experiment_id", "runtime_versions", "study", "model", "data", "source_records", "holdout_evaluation"):
            if existing.get(key) != protocol.get(key):
                raise RuntimeError(f"Existing frozen protocol differs at {key}; refusing to reuse study artifacts")
        if existing.get("code_identity") != protocol.get("code_identity"):
            completed_postfit = False
            if MANIFEST_PATH.is_file() and (OUTPUT_DIR / "selection.json").is_file():
                run_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
                completed_postfit = run_manifest.get("status") == "completed"
            if completed_postfit:
                protocol = existing
            elif MANIFEST_PATH.exists() or STUDY_DB_PATH.exists() or (OUTPUT_DIR / "trials").exists():
                raise RuntimeError("Runner identity changed after the study started; refusing to alter its frozen protocol")
            else:
                protocol["frozen_at_utc"] = existing["frozen_at_utc"]
                protocol["pre_fit_protocol_revision"] = {
                    "supersedes_protocol_sha256": _sha256(PROTOCOL_PATH),
                    "reason": "pre-fit serialization validation; no model fits or study database existed",
                    "revised_at_utc": now,
                }
                _atomic_json(PROTOCOL_PATH, protocol)
        else:
            protocol["frozen_at_utc"] = existing["frozen_at_utc"]
            if "pre_fit_protocol_revision" in existing:
                protocol["pre_fit_protocol_revision"] = existing["pre_fit_protocol_revision"]
            protocol = existing
    else:
        _atomic_json(PROTOCOL_PATH, protocol)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol_copy = OUTPUT_DIR / "frozen_protocol.json"
    if not protocol_copy.exists() or json.loads(protocol_copy.read_text(encoding="utf-8")) != protocol:
        _atomic_json(protocol_copy, protocol)
    return protocol


def _resource_snapshot() -> dict:
    return _system_snapshot()


def _new_manifest(protocol: dict, resources: dict) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": STUDY_ID,
        "status": "preflight_complete",
        "active_stage": None,
        "frozen_protocol_path": _relative(PROTOCOL_PATH),
        "frozen_protocol_sha256": _sha256(PROTOCOL_PATH),
        "frozen_identity_sha256": _canonical_sha256(protocol),
        "runtime_identity": protocol["runtime_versions"] | {"runner_sha256": protocol["code_identity"]["runner_sha256"]},
        "resource_snapshot_before_training": resources,
        "study_elapsed_seconds": 0.0,
        "study_budget_seconds": STUDY_BUDGET_SECONDS,
        "training_counters": {
            "started_configurations": 0,
            "completed_configurations": 0,
            "incomplete_configurations": 0,
            "started_seed_evaluations": 0,
            "completed_seed_evaluations": 0,
            "started_fold_fits": 0,
            "completed_fold_fits": 0,
            "reused_completed_seed_evaluations": 0,
        },
        "trial_results": {},
        "stages": {},
        "next_step": "Verify source matrix/folds and begin the pre-registered control trial.",
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def _write_status(manifest: dict) -> None:
    counts = manifest.get("training_counters", {})
    status = manifest.get("status", "not_started")
    text_block = [
        STATUS_BEGIN,
        "### BASE70 capacity tuning — 2026-10-19",
        f"- Status: `{status}`; completed configurations: `{counts.get('completed_configurations', 0)}`; started configurations: `{counts.get('started_configurations', 0)}/32`.",
        f"- CV work: `{counts.get('completed_fold_fits', 0)}` completed fold fits, `{counts.get('started_fold_fits', 0)}` started; study time `{manifest.get('study_elapsed_seconds', 0.0):.1f}/1800 seconds`.",
        f"- Next step: {manifest.get('next_step', 'See the run manifest.')}",
        f"- Report and manifest: `{_relative(REPORT_PATH)}`, `{_relative(VERSIONED_MANIFEST_PATH)}`.",
        STATUS_END,
        "",
    ]
    existing = STATUS_PATH.read_text(encoding="utf-8")
    block = "\n".join(text_block)
    if STATUS_BEGIN in existing and STATUS_END in existing:
        before, tail = existing.split(STATUS_BEGIN, 1)
        _, after = tail.split(STATUS_END, 1)
        updated = before.rstrip() + "\n" + block + after.lstrip("\n")
    else:
        anchor = "# BTC research execution status"
        if anchor not in existing:
            raise RuntimeError("Research status file is missing its expected heading")
        updated = existing.replace(anchor, anchor + "\n\n" + block.rstrip() + "\n", 1)
    temporary = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    os.replace(temporary, STATUS_PATH)


def _write_manifest(manifest: dict) -> None:
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _atomic_json(MANIFEST_PATH, manifest)
    _atomic_json(VERSIONED_MANIFEST_PATH, manifest)
    _write_status(manifest)


def _load_or_create_manifest(protocol: dict, resources: dict) -> dict:
    identity = _canonical_sha256(protocol)
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("frozen_identity_sha256") != identity:
            raise RuntimeError("Existing tuning manifest has a different frozen identity")
        return manifest
    manifest = _new_manifest(protocol, resources)
    manifest["stages"]["preflight"] = {
        "status": "completed",
        "summary": {
            "source_rows": protocol["data"]["rows"],
            "features": protocol["data"]["features"],
            "folds": protocol["data"]["fold_count"],
            "weight_hash": protocol["data"]["weights_sha256"],
            "purge": "verified by saved-fold validation before fitting",
        },
        "outputs": [protocol["source_records"]["matrix"], protocol["source_records"]["folds"]],
    }
    _write_manifest(manifest)
    return manifest


def _clock_path() -> Path:
    return OUTPUT_DIR / "study_clock.json"


def _write_clock(clock: StudyClock, **progress) -> None:
    value = {"elapsed_seconds": clock.elapsed(), "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(), **progress}
    _atomic_json(_clock_path(), value)


def _deadline_callback(clock: StudyClock):
    class Callback:
        order = 20
        before_iteration = False

        def __call__(self, env):
            elapsed = clock.elapsed()
            iteration = int(env.iteration + 1)
            if iteration % 10 == 0 or elapsed >= STUDY_BUDGET_SECONDS - HARD_STOP_RESERVE_SECONDS:
                _write_clock(clock, stage="cv", cv_iteration=iteration)
            if elapsed >= STUDY_BUDGET_SECONDS - HARD_STOP_RESERVE_SECONDS:
                raise StudyBudgetExpired(iteration, elapsed)

    return Callback()


def _distributions() -> dict:
    baseline = json.loads(seed_stability.BASE_CV_SUMMARY_PATH.read_text(encoding="utf-8"))["params"]
    return {
        "num_leaves": optuna.distributions.CategoricalDistribution([*PARAMETER_LEVELS["num_leaves"], int(baseline["num_leaves"])]),
        "min_data_in_leaf": optuna.distributions.CategoricalDistribution(list(PARAMETER_LEVELS["min_data_in_leaf"])),
        "lambda_l1": optuna.distributions.CategoricalDistribution([*PARAMETER_LEVELS["lambda_l1"], float(baseline["lambda_l1"])]),
        "lambda_l2": optuna.distributions.CategoricalDistribution([*PARAMETER_LEVELS["lambda_l2"], float(baseline["lambda_l2"])]),
    }


class _RecordingObjectiveCallback:
    """Record every evaluated iteration while delegating stopping to the shared callback."""

    def __init__(self, callback: ObjectiveAlignedCVCallback):
        self.callback = callback
        self.order = callback.order
        self.before_iteration = callback.before_iteration
        self.history = []

    def __call__(self, env):
        objective, mean, _base, std = self.callback._objective_from_evaluations(env.evaluation_result_list)
        self.history.append({
            "iteration": int(env.iteration + 1),
            "mean_fold_logloss": float(mean),
            "fold_logloss_population_std": float(std),
            "criterion": float(objective),
        })
        self.callback(env)


def _write_gzip_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(gzip.compress(value.encode("utf-8"), compresslevel=6, mtime=0))
    os.replace(temporary, path)


def _load_cv_checkpoint(path: Path, frozen_identity: str):
    metadata_path = path.with_name("cv_complete_checkpoint.json")
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("frozen_identity_sha256") != frozen_identity:
        raise RuntimeError(f"CV completion checkpoint identity differs: {metadata_path}")
    if not _verify_records(metadata.get("outputs", [])):
        raise RuntimeError(f"CV completion checkpoint outputs failed verification: {metadata_path}")
    model_bundle = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    cvbooster = lgb.CVBooster()
    cvbooster._from_dict(model_bundle)
    history_path = path.with_name("cv_objective_history.json")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return metadata, cvbooster, history


def _new_optuna_study(protocol: dict) -> optuna.study.Study:
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{STUDY_DB_PATH.as_posix()}")
    study = optuna.create_study(
        study_name=STUDY_ID,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=SEARCH_SCHEDULE_SEED),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )
    if not study.trials:
        for row in protocol["study"]["configurations"]:
            study.enqueue_trial(
                row["parameters"],
                user_attrs={"config_id": row["config_id"], "kind": row["kind"], "trial_id": row["trial_id"]},
            )
    return study


def _trial_configuration(trial: optuna.trial.Trial, protocol: dict) -> dict:
    parameters = {
        name: trial.suggest_categorical(name, distribution.choices)
        for name, distribution in _distributions().items()
    }
    config_id = trial.user_attrs.get("config_id")
    row = next((item for item in protocol["study"]["configurations"] if item["config_id"] == config_id), None)
    if row is None or row["parameters"] != parameters or int(trial.user_attrs.get("trial_id", -1)) != int(row["trial_id"]):
        raise RuntimeError("Optuna queued trial differs from the frozen configuration schedule")
    return row


def _seed_output_dir(config_id: str, seed: int) -> Path:
    return OUTPUT_DIR / "trials" / config_id / f"seed_{seed}"


def _verify_seed_result(path: Path, frozen_identity: str) -> dict | None:
    result_path = path / "seed_result.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("frozen_identity_sha256") != frozen_identity:
        raise RuntimeError(f"Seed checkpoint identity differs: {result_path}")
    if not _verify_records(result.get("outputs", [])):
        raise RuntimeError(f"Seed checkpoint outputs failed verification: {result_path}")
    return result


def _save_seed_result(
    frame: pd.DataFrame,
    x: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_names: list[str],
    base_params: dict,
    config: dict,
    seed: int,
    trial_number: int,
    clock: StudyClock,
    manifest: dict,
) -> dict:
    seed_dir = _seed_output_dir(config["config_id"], seed)
    history_path = seed_dir / "cv_objective_history.json"
    seed_oof_path = seed_dir / "oof.parquet"
    cv_bundle_path = seed_dir / "cv_complete_boosters.json.gz"
    params = _effective_params(base_params, config["parameters"], seed)
    checkpoint = _load_cv_checkpoint(cv_bundle_path, manifest["frozen_identity_sha256"])
    fit_elapsed_seconds = 0.0
    if checkpoint is None:
        if clock.elapsed() >= STUDY_BUDGET_SECONDS - HARD_STOP_RESERVE_SECONDS:
            raise StudyBudgetExpired(0, clock.elapsed())
        training_set = lgb.Dataset(
            x,
            label=target,
            weight=weights,
            feature_name=feature_names,
            free_raw_data=False,
        )

        def unweighted_validation(train_data, valid_data, cv_params):
            valid_data.set_weight(None)
            return train_data, valid_data, cv_params

        objective_callback = ObjectiveAlignedCVCallback(
            "binary_logloss",
            higher_is_better=False,
            std_penalty=0.5,
            stopping_rounds=EARLY_STOPPING_ROUNDS,
            trial=None,
        )
        recording_callback = _RecordingObjectiveCallback(objective_callback)
        callbacks = [_deadline_callback(clock), recording_callback, lgb.log_evaluation(period=25)]
        manifest["training_counters"]["started_seed_evaluations"] += 1
        manifest["training_counters"]["started_fold_fits"] += len(folds)
        manifest["active_seed_evaluation"] = {
            "trial_number": trial_number, "config_id": config["config_id"], "seed": seed,
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(), "folds_started": len(folds),
        }
        manifest["study_elapsed_seconds"] = clock.elapsed()
        manifest["next_step"] = f"Run seed {seed} of {config['config_id']} on the saved ten folds."
        _write_manifest(manifest)
        if clock.elapsed() >= STUDY_BUDGET_SECONDS - HARD_STOP_RESERVE_SECONDS:
            manifest["training_counters"]["started_seed_evaluations"] -= 1
            manifest["training_counters"]["started_fold_fits"] -= len(folds)
            manifest["active_seed_evaluation"] = None
            manifest["study_elapsed_seconds"] = clock.elapsed()
            _write_manifest(manifest)
            del training_set
            gc.collect()
            raise StudyBudgetExpired(0, clock.elapsed())
        fit_started = time.perf_counter()
        try:
            cv_results = lgb.cv(
                params=params,
                train_set=training_set,
                num_boost_round=MAX_ESTIMATORS,
                folds=folds,
                fpreproc=unweighted_validation,
                stratified=False,
                shuffle=False,
                feval=[make_lightgbm_binary_logloss_eval("binary_logloss")],
                callbacks=callbacks,
                return_cvbooster=True,
                seed=int(seed),
            )
        finally:
            fit_elapsed_seconds = time.perf_counter() - fit_started
            del training_set
            gc.collect()
        cvbooster = cv_results.pop("cvbooster")
        history = recording_callback.history
        if not history or not all(math.isfinite(row["criterion"]) for row in history):
            raise RuntimeError("CV returned an empty or non-finite objective history")
        best_index = int(np.argmin([row["criterion"] for row in history]))
        best_iteration = best_index + 1
        callback_iteration = int(objective_callback.best_iteration + 1)
        booster_iteration = int(getattr(cvbooster, "best_iteration", 0) or 0)
        if callback_iteration != best_iteration or booster_iteration not in (0, best_iteration):
            raise RuntimeError("The objective-aligned early-stop callback and full-history ranking differ")
        truncated_series = cv_objective_series(
            cv_results, "binary_logloss", higher_is_better=False, std_penalty=0.5,
        )
        if len(truncated_series["objective"]) not in (best_iteration, len(history)) or not np.allclose(
            truncated_series["objective"],
            [row["criterion"] for row in history[:len(truncated_series["objective"])]],
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError("Persisted full callback history differs from LightGBM's selected CV history")
        stopped_by_patience = len(truncated_series["objective"]) < len(history)
        seed_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(history_path, {
            "trial_number": trial_number,
            "config_id": config["config_id"],
            "seed": seed,
            "selection_rule": "mean ten-fold unweighted binary_logloss + 0.5 * population standard deviation",
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "maximum_estimators": MAX_ESTIMATORS,
            "best_iteration": best_iteration,
            "best_objective": history[best_index]["criterion"],
            "callback_best_iteration": callback_iteration,
            "history_length": len(history),
            "stopped_by_patience": stopped_by_patience,
            "iterations": history,
        })
        _write_gzip_text(cv_bundle_path, cvbooster.model_to_string(num_iteration=best_iteration))
        _atomic_json(seed_dir / "cv_complete_checkpoint.json", {
            "frozen_identity_sha256": manifest["frozen_identity_sha256"],
            "trial_number": trial_number,
            "config_id": config["config_id"],
            "seed": seed,
            "effective_params": params,
            "best_iteration": best_iteration,
            "evaluated_iterations": len(history),
            "fit_elapsed_seconds": fit_elapsed_seconds,
            "outputs": [_record(cv_bundle_path), _record(history_path)],
        })
        checkpoint = _load_cv_checkpoint(cv_bundle_path, manifest["frozen_identity_sha256"])
    else:
        checkpoint_meta, cvbooster, history_document = checkpoint
        if checkpoint_meta["effective_params"] != params or int(checkpoint_meta["seed"]) != seed:
            raise RuntimeError("The completed CV checkpoint parameters differ from the frozen seed configuration")
        best_iteration = int(checkpoint_meta["best_iteration"])
        fit_elapsed_seconds = float(checkpoint_meta.get("fit_elapsed_seconds", 0.0))
        history = history_document["iterations"]
        stopped_by_patience = bool(history_document["stopped_by_patience"])
    best_index = best_iteration - 1
    score_history = history
    if len(cvbooster.boosters) != len(folds):
        raise RuntimeError("LightGBM CV returned an unexpected fold count")
    if int(history[best_index]["iteration"]) != best_iteration:
        raise RuntimeError("The selected iteration does not match the persisted objective history")

    target_name = "target_preopen_5m_up_binance_index_proxy"
    probabilities = np.full(len(frame), np.nan, dtype=np.float32)
    fold_id_values = np.full(len(frame), -1, dtype=np.int8)
    fold_metrics = []
    fold_artifacts = []
    for fold_id, ((_, valid_idx), booster) in enumerate(zip(folds, cvbooster.boosters, strict=True)):
        model_path = seed_dir / "models" / f"fold_{fold_id:02d}.txt"
        oof_path = seed_dir / "fold_oof" / f"fold_{fold_id:02d}.parquet"
        completion_path = seed_dir / "fold_checkpoints" / f"fold_{fold_id:02d}_completion.json"
        reusable_fold = None
        if completion_path.is_file():
            reusable_fold = json.loads(completion_path.read_text(encoding="utf-8"))
            if (
                reusable_fold.get("frozen_identity_sha256") != manifest["frozen_identity_sha256"]
                or reusable_fold.get("effective_params") != params
                or int(reusable_fold.get("best_iteration", -1)) != best_iteration
                or not _verify_records([reusable_fold.get("model"), reusable_fold.get("oof")])
            ):
                raise RuntimeError(f"Fold checkpoint failed verification: {completion_path}")
            fold_oof = pd.read_parquet(oof_path)
            expected_indices = frame.iloc[valid_idx]["_source_row_index"].to_numpy(dtype=np.int32)
            if not np.array_equal(fold_oof["_source_row_index"].to_numpy(dtype=np.int32), expected_indices):
                raise RuntimeError(f"Fold {fold_id} OOF indices differ from the saved fold")
            probability = fold_oof["p_model_raw"].to_numpy(dtype=np.float32)
            fold_score = reusable_fold["metrics"]
        else:
            probability = np.asarray(booster.predict(x[valid_idx], num_iteration=best_iteration), dtype=np.float32)
            fold_score = None
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise RuntimeError(f"Seed {seed}, fold {fold_id}: invalid OOF probabilities")
        probabilities[valid_idx] = probability
        fold_id_values[valid_idx] = fold_id
        if reusable_fold is None:
            metrics = taker._score_binary(target[valid_idx], probability.astype(np.float64))
            fold_score = {
                "fold_id": fold_id,
                "train_rows": int(len(folds[fold_id][0])),
                "validation_rows": int(len(valid_idx)),
                "log_loss": float(metrics["log_loss"]),
                "brier": float(metrics["brier"]),
                "auc": float(metrics["auc"]),
            }
            temporary_model = model_path.with_suffix(".txt.tmp")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(temporary_model), num_iteration=best_iteration)
            os.replace(temporary_model, model_path)
            fold_oof = frame.iloc[valid_idx][[column for column in OOF_ID_COLUMNS if column in frame.columns]].copy()
            if "market_start_utc" not in fold_oof:
                fold_oof["market_start_utc"] = fold_oof["Opened"] + pd.Timedelta(minutes=2)
            fold_oof["fold_id"] = str(fold_id)
            fold_oof["p_model_raw"] = probability
            _atomic_parquet(fold_oof, oof_path)
            completion = {
                "frozen_identity_sha256": manifest["frozen_identity_sha256"],
                "trial_number": trial_number,
                "config_id": config["config_id"],
                "seed": seed,
                "fold_id": fold_id,
                "best_iteration": best_iteration,
                "effective_params": params,
                "validation_source_row_index_sha256": taker._array_sha256(
                    fold_oof["_source_row_index"].to_numpy(dtype=np.int32), dtype=np.int32,
                ),
                "metrics": fold_score,
                "model": _record(model_path),
                "oof": _record(oof_path),
            }
            _atomic_json(completion_path, completion)
            manifest["training_counters"]["completed_fold_fits"] += 1
        else:
            completion = reusable_fold
            if fold_id >= int(manifest["active_seed_evaluation"].get("completed_fold_checkpoints", 0)):
                manifest["training_counters"]["completed_fold_fits"] += 1
        fold_metrics.append(fold_score)
        fold_artifacts.append({
            "fold_id": fold_id,
            "model": completion["model"],
            "oof": completion["oof"],
            "completion": _record(completion_path),
            "metrics": fold_metrics[-1],
        })
        manifest["active_seed_evaluation"]["completed_fold_checkpoints"] = fold_id + 1
        manifest["study_elapsed_seconds"] = clock.elapsed()
        _write_manifest(manifest)

    valid_indices = np.flatnonzero(np.isfinite(probabilities))
    expected_oof_rows = int(json.loads(seed_stability.BASE_CV_SUMMARY_PATH.read_text(encoding="utf-8"))["oof_rows"])
    if len(valid_indices) != expected_oof_rows:
        raise RuntimeError("The CV OOF row count differs from the frozen source comparison")
    if np.any(fold_id_values[valid_indices] < 0) or len(np.unique(frame.iloc[valid_indices]["_source_row_index"])) != len(valid_indices):
        raise RuntimeError("The OOF predictions do not have unique source-row identities")
    seed_oof = frame.iloc[valid_indices][[
        "_source_row_index", "Opened", "nominal_decision_at", "target_label_available_at",
        target_name,
    ]].copy()
    seed_oof["market_start_utc"] = seed_oof["Opened"] + pd.Timedelta(minutes=2)
    seed_oof["fold_id"] = fold_id_values[valid_indices].astype(str)
    seed_oof["p_model_raw"] = probabilities[valid_indices]
    seed_oof = seed_oof.sort_values("_source_row_index", kind="stable").reset_index(drop=True)
    _atomic_parquet(seed_oof, seed_oof_path)

    original_oof = pd.read_parquet(seed_stability.BASE_OOF_PATH, columns=["_source_row_index", "fold_id"])
    if not np.array_equal(
        seed_oof["_source_row_index"].to_numpy(dtype=np.int32),
        original_oof["_source_row_index"].to_numpy(dtype=np.int32),
    ) or not np.array_equal(seed_oof["fold_id"].astype(str).to_numpy(), original_oof["fold_id"].astype(str).to_numpy()):
        raise RuntimeError("The new OOF source indices or fold IDs differ from the frozen BASE70 OOF index")

    score = float(history[best_index]["criterion"])
    outputs = [
        _record(history_path),
        _record(seed_oof_path),
        _record(seed_dir / "cv_complete_checkpoint.json"),
        _record(cv_bundle_path),
        *[
        record
        for item in fold_artifacts
        for record in (item["model"], item["oof"], item["completion"])
        ],
    ]
    result = {
        "frozen_identity_sha256": manifest["frozen_identity_sha256"],
        "trial_number": trial_number,
        "config_id": config["config_id"],
        "seed": seed,
        "effective_params": params,
        "best_iteration": best_iteration,
        "best_cv_objective": score,
        "best_mean_fold_logloss": history[best_index]["mean_fold_logloss"],
        "best_fold_logloss_population_std": history[best_index]["fold_logloss_population_std"],
        "history_length": len(history),
        "stopped_by_patience": stopped_by_patience,
        "fold_metrics": fold_metrics,
        "fold_artifacts": fold_artifacts,
        "history": _record(history_path),
        "oof": _record(seed_oof_path),
        "outputs": outputs,
        "fit_elapsed_seconds": fit_elapsed_seconds,
        "cv_completion_checkpoint": _record(seed_dir / "cv_complete_checkpoint.json"),
        "cv_booster_bundle": _record(cv_bundle_path),
    }
    result_path = seed_dir / "seed_result.json"
    _atomic_json(result_path, result)
    manifest["training_counters"]["completed_seed_evaluations"] += 1
    manifest["active_seed_evaluation"] = None
    manifest["study_elapsed_seconds"] = clock.elapsed()
    manifest["next_step"] = f"Seed {seed} of {config['config_id']} is checkpointed; continue remaining seeds."
    _write_manifest(manifest)
    return result


def _get_trial(study: optuna.study.Study, manifest: dict) -> optuna.trial.Trial | None:
    running = [trial for trial in study.get_trials(deepcopy=False) if trial.state == optuna.trial.TrialState.RUNNING]
    if len(running) > 1:
        raise RuntimeError("More than one Optuna trial is RUNNING; concurrent study execution is forbidden")
    if running:
        frozen = running[0]
        if manifest.get("active_trial_number") != frozen.number:
            raise RuntimeError("A RUNNING Optuna trial is not identified by the resumable manifest")
        return optuna.trial.Trial(study, frozen._trial_id)
    waiting = [trial for trial in study.get_trials(deepcopy=False) if trial.state == optuna.trial.TrialState.WAITING]
    if not waiting:
        return None
    return study.ask()


def _run_study(
    protocol: dict,
    manifest: dict,
    frame: pd.DataFrame,
    x: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_names: list[str],
    clock: StudyClock,
) -> tuple[optuna.study.Study, bool]:
    study = _new_optuna_study(protocol)
    frozen_identity = manifest["frozen_identity_sha256"]
    budget_exhausted = False
    while True:
        if clock.elapsed() >= STUDY_BUDGET_SECONDS - HARD_STOP_RESERVE_SECONDS:
            break
        trial = _get_trial(study, manifest)
        if trial is None:
            break
        config = _trial_configuration(trial, protocol)
        manifest["active_trial_number"] = int(trial.number)
        manifest["active_config_id"] = config["config_id"]
        started_now = sum(
            item.state != optuna.trial.TrialState.WAITING
            for item in study.get_trials(deepcopy=False)
        )
        manifest["training_counters"]["started_configurations"] = max(
            int(manifest["training_counters"]["started_configurations"]), started_now,
        )
        manifest["status"] = "running"
        manifest["active_stage"] = "cv_study"
        manifest["next_step"] = f"Complete three-seed CV for {config['config_id']}."
        manifest["trial_results"].setdefault(config["config_id"], {
            "trial_number": int(trial.number), "kind": config["kind"],
            "parameters": config["parameters"], "state": "running", "seed_results": {},
        })
        trial.set_user_attr("config_id", config["config_id"])
        trial.set_user_attr("kind", config["kind"])
        _write_manifest(manifest)
        trial_results = manifest["trial_results"][config["config_id"]]
        try:
            for seed in MODEL_SEEDS:
                if clock.elapsed() >= STUDY_BUDGET_SECONDS - HARD_STOP_RESERVE_SECONDS:
                    raise StudyBudgetExpired(0, clock.elapsed())
                seed_dir = _seed_output_dir(config["config_id"], seed)
                saved = _verify_seed_result(seed_dir, frozen_identity)
                if saved is not None:
                    already_recorded = str(seed) in trial_results.get("seed_results", {})
                    trial_results["seed_results"][str(seed)] = {
                        "path": _relative(seed_dir / "seed_result.json"),
                        "sha256": _sha256(seed_dir / "seed_result.json"),
                        "best_cv_objective": saved["best_cv_objective"],
                        "best_iteration": saved["best_iteration"],
                        "reused_checkpoint": True,
                    }
                    if not already_recorded:
                        manifest["training_counters"]["reused_completed_seed_evaluations"] += 1
                    _write_manifest(manifest)
                    continue
                saved = _save_seed_result(
                    frame, x, target, weights, folds, feature_names,
                    json.loads(seed_stability.BASE_CV_SUMMARY_PATH.read_text(encoding="utf-8"))["params"],
                    config, seed, int(trial.number), clock, manifest,
                )
                trial_results["seed_results"][str(seed)] = {
                    "path": _relative(seed_dir / "seed_result.json"),
                    "sha256": _sha256(seed_dir / "seed_result.json"),
                    "best_cv_objective": saved["best_cv_objective"],
                    "best_iteration": saved["best_iteration"],
                    "reused_checkpoint": False,
                }
                trial.set_user_attr("completed_seeds", sorted(int(x) for x in trial_results["seed_results"]))
                manifest["study_elapsed_seconds"] = clock.elapsed()
                _write_manifest(manifest)
            seed_records = [
                json.loads((ROOT / trial_results["seed_results"][str(seed)]["path"]).read_text(encoding="utf-8"))
                for seed in MODEL_SEEDS
            ]
            score = float(np.mean([row["best_cv_objective"] for row in seed_records]))
            trial_result = {
                "frozen_identity_sha256": frozen_identity,
                "trial_number": int(trial.number),
                "config_id": config["config_id"],
                "kind": config["kind"],
                "parameters": config["parameters"],
                "trial_objective": score,
                "seed_scores": {str(row["seed"]): row["best_cv_objective"] for row in seed_records},
                "seed_best_iterations": {str(row["seed"]): row["best_iteration"] for row in seed_records},
                "seed_result_records": [trial_results["seed_results"][str(seed)] for seed in MODEL_SEEDS],
                "selection_rule": "mean of the three seed-level CV objectives; no best-seed selection",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            trial_result_path = OUTPUT_DIR / "trials" / config["config_id"] / "trial_result.json"
            _atomic_json(trial_result_path, trial_result)
            trial_results.update({
                "state": "complete",
                "trial_objective": score,
                "trial_result": _record(trial_result_path),
                "completed_utc": trial_result["completed_utc"],
            })
            trial.set_user_attr("trial_result_path", _relative(trial_result_path))
            study.tell(trial.number, score)
            manifest["training_counters"]["completed_configurations"] += 1
            manifest["study_elapsed_seconds"] = clock.elapsed()
            manifest["active_trial_number"] = None
            manifest["active_config_id"] = None
            manifest["active_stage"] = None
            manifest["next_step"] = "Start the next frozen configuration if the time budget permits."
            _write_clock(clock, stage="trial_complete", config_id=config["config_id"])
            _write_manifest(manifest)
            print(
                f"[study] trial {trial.number + 1}/32 {config['config_id']} complete; "
                f"objective={score:.8f}; seeds={trial_result['seed_best_iterations']}; "
                f"elapsed={clock.elapsed():.1f}s",
                flush=True,
            )
        except StudyBudgetExpired as error:
            budget_exhausted = True
            manifest["training_counters"]["incomplete_configurations"] += 1
            trial_results["state"] = "incomplete_budget"
            trial_results["incomplete_reason"] = str(error)
            trial_results["seed_results"] = trial_results.get("seed_results", {})
            trial.set_user_attr("incomplete_reason", str(error))
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            manifest["status"] = "budget_reached"
            manifest["study_elapsed_seconds"] = clock.elapsed()
            manifest["active_trial_number"] = None
            manifest["active_config_id"] = None
            manifest["active_seed_evaluation"] = None
            manifest["active_stage"] = None
            manifest["next_step"] = "Do not start another CV fit; finalize selection from complete trials only."
            _write_clock(clock, stage="budget_reached", incomplete_config_id=config["config_id"], cv_iteration=error.iteration)
            _write_manifest(manifest)
            print(f"[study] hard stop: {error}; incomplete trial excluded from selection", flush=True)
            break
        except BaseException:
            manifest["status"] = "interrupted_resumable"
            manifest["study_elapsed_seconds"] = clock.elapsed()
            manifest["next_step"] = "Resume the active trial from completed seed/fold checkpoints; do not enqueue duplicate configurations."
            _write_clock(clock, stage="interrupted_resumable")
            _write_manifest(manifest)
            raise
    manifest["study_elapsed_seconds"] = clock.elapsed()
    if not budget_exhausted and manifest.get("status") == "running":
        manifest["status"] = "study_complete"
        manifest["active_stage"] = None
        manifest["active_trial_number"] = None
        manifest["active_config_id"] = None
        manifest["next_step"] = "Freeze the CV-only winner before opening official development outcomes."
    _write_clock(clock, stage="study_stopped", budget_exhausted=budget_exhausted)
    _write_manifest(manifest)
    return study, budget_exhausted


def _complete_trial_rows(study: optuna.study.Study, protocol: dict) -> list[dict]:
    by_id = {row["config_id"]: row for row in protocol["study"]["configurations"]}
    output = []
    for frozen in study.get_trials(deepcopy=False):
        config_id = frozen.user_attrs.get("config_id")
        if config_id not in by_id or frozen.state != optuna.trial.TrialState.COMPLETE:
            continue
        trial_path = OUTPUT_DIR / "trials" / config_id / "trial_result.json"
        trial_result = json.loads(trial_path.read_text(encoding="utf-8"))
        output.append({**trial_result, "parameters": by_id[config_id]["parameters"]})
    return sorted(output, key=lambda row: (row["trial_objective"], row["trial_number"]))


def _freeze_selection(study: optuna.study.Study, protocol: dict, manifest: dict) -> dict:
    complete = _complete_trial_rows(study, protocol)
    controls = [row for row in complete if row["kind"] == "control"]
    if len(controls) != 1:
        raise RuntimeError("The control CV configuration did not complete; no selection can be made")
    if not complete:
        raise RuntimeError("There are no complete CV trials to select")
    winner = complete[0]
    control = controls[0]
    selection = {
        "frozen_identity_sha256": manifest["frozen_identity_sha256"],
        "selected_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "selection_basis": "CV-only; mean of three seed-level mean-fold-logloss + 0.5 * population fold-SD scores",
        "official_target_or_economics_used_for_selection": False,
        "selected_trial_number": int(winner["trial_number"]),
        "selected_config_id": winner["config_id"],
        "selected_parameters": winner["parameters"],
        "selected_objective": winner["trial_objective"],
        "control_trial_number": int(control["trial_number"]),
        "control_objective": control["trial_objective"],
        "complete_trial_count": len(complete),
        "incomplete_trials_excluded": [
            {"config_id": key, "reason": value.get("incomplete_reason")}
            for key, value in manifest["trial_results"].items()
            if value.get("state") == "incomplete_budget"
        ],
        "winner_seed_scores": winner["seed_scores"],
        "control_seed_scores": control["seed_scores"],
    }
    path = OUTPUT_DIR / "selection.json"
    _atomic_json(path, selection)
    manifest["selection"] = {**selection, "record": _record(path)}
    manifest["status"] = "cv_selection_frozen"
    manifest["active_stage"] = "post_selection_development_evaluation"
    manifest["next_step"] = "Evaluate the frozen winner and control on official development and the saved T-60/T-59 calendar."
    _write_manifest(manifest)
    return selection


def _read_trial_seed(config_id: str, seed: int) -> tuple[dict, pd.DataFrame]:
    seed_dir = _seed_output_dir(config_id, seed)
    result = json.loads((seed_dir / "seed_result.json").read_text(encoding="utf-8"))
    oof = pd.read_parquet(seed_dir / "oof.parquet")
    return result, oof


def _development_quality(
    selected: dict,
    control: dict,
    historical_single: dict,
    cv_summary: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison = pd.read_parquet(seed_stability.BASE_PREDICTIONS_PATH)
    comparison["market_start_utc"] = pd.to_datetime(comparison["market_start_utc"], utc=True, errors="raise")
    if comparison.market_start_utc.max() >= OFFICIAL_CUTOFF:
        raise RuntimeError("The saved official evaluation source includes holdout starts")
    mask = (
        comparison["comparison_forecast_available"].astype(bool)
        & comparison["official_outcome_available_by_cutoff"].astype(bool)
        & comparison.market_start_utc.lt(OFFICIAL_CUTOFF)
    )
    official_base = comparison.loc[mask, ["market_start_utc", "target_polymarket_up", "fold_id", "p_raw"]].copy()
    if len(official_base) != 20_718:
        raise RuntimeError("The official development sample differs from the frozen 20,718-row sample")

    proxy_rows = []
    official_rows = []
    fold_rows = []
    prediction_bundle = None
    for role, config in (("control", control), ("selected", selected)):
        for seed in MODEL_SEEDS:
            result, oof = _read_trial_seed(config["config_id"], seed)
            if prediction_bundle is None:
                prediction_bundle = oof[[
                    "_source_row_index", "Opened", "nominal_decision_at", "target_label_available_at",
                    "target_preopen_5m_up_binance_index_proxy", "market_start_utc", "fold_id",
                ]].copy()
            elif not prediction_bundle[OOF_ID_COLUMNS].equals(oof[OOF_ID_COLUMNS]):
                raise RuntimeError("Control/winner OOF market indices differ")
            pcol = f"p_{role}_{seed}"
            prediction_bundle[pcol] = oof.p_model_raw.to_numpy(dtype=np.float32)
            proxy_score = taker._score_binary(
                oof.target_preopen_5m_up_binance_index_proxy.to_numpy(dtype=np.int8),
                oof.p_model_raw.to_numpy(dtype=np.float64),
            )
            proxy_rows.append({
                "role": role, "seed": seed, "config_id": config["config_id"],
                "best_iteration": result["best_iteration"],
                "cv_selection_objective": result["best_cv_objective"],
                **proxy_score,
            })
            official = official_base.merge(
                oof[["market_start_utc", "fold_id", "p_model_raw"]],
                on="market_start_utc", how="left", validate="one_to_one",
            )
            if official.p_model_raw.isna().any() or len(official) != 20_718:
                raise RuntimeError(f"{role} seed {seed} does not cover the exact official development sample")
            official_score = taker._score_binary(
                official.target_polymarket_up.to_numpy(dtype=np.int8),
                official.p_model_raw.to_numpy(dtype=np.float64),
            )
            official_rows.append({
                "role": role, "seed": seed, "config_id": config["config_id"],
                "best_iteration": result["best_iteration"], **official_score,
            })
            for fold in result["fold_metrics"]:
                fold_rows.append({"role": role, "seed": seed, "best_iteration": result["best_iteration"], **fold})

    # Historical SINGLE at 31 trees remains a separate saved comparator.
    historical_oof = pd.read_parquet(seed_stability.BASE_OOF_PATH)
    historical_score = taker._score_binary(
        historical_oof.target_preopen_5m_up_binance_index_proxy.to_numpy(dtype=np.int8),
        historical_oof.p_model_raw.to_numpy(dtype=np.float64),
    )
    proxy_rows.append({
        "role": "historical_SINGLE_31", "seed": 20261015, "config_id": "saved_B0_31_tree",
        "best_iteration": 31, "cv_selection_objective": None, **historical_score,
    })
    official_historical = taker._score_binary(
        official_base.target_polymarket_up.to_numpy(dtype=np.int8),
        official_base.p_raw.to_numpy(dtype=np.float64),
    )
    official_rows.append({
        "role": "historical_SINGLE_31", "seed": 20261015, "config_id": "saved_B0_31_tree",
        "best_iteration": 31, **official_historical,
    })
    bundle_path = OUTPUT_DIR / "selected_control_oof.parquet"
    _atomic_parquet(prediction_bundle, bundle_path)
    proxy_frame = pd.DataFrame(proxy_rows)
    official_frame = pd.DataFrame(official_rows)
    fold_frame = pd.DataFrame(fold_rows)
    proxy_frame.to_csv(OUTPUT_DIR / "proxy_quality_by_seed.csv", index=False)
    official_frame.to_csv(OUTPUT_DIR / "official_development_quality_by_seed.csv", index=False)
    fold_frame.to_csv(OUTPUT_DIR / "proxy_fold_quality_control_selected.csv", index=False)
    return proxy_frame, official_frame, fold_frame


def _prepare_economic_inputs(info_manifest: dict) -> tuple[pd.DataFrame, dict, dict, dict, dict]:
    model_update.fee_correction._configure_output_paths()
    correction_manifest = research._read_json(
        model_update.fee_correction.CORRECTION_DIR / "research_manifest.json"
    )
    t59, t60, book_record = research._load_enriched_book_snapshots(correction_manifest)
    frozen_snapshots = info_manifest["frozen_comparison"]["specification"]["sources"]["decision_and_execution_snapshots"]
    if (
        book_record["t59_coverage_sha256"] != frozen_snapshots["t59_sha256"]
        or book_record["t60_coverage_sha256"] != frozen_snapshots["t60_sha256"]
    ):
        raise RuntimeError("The saved decision/execution snapshots differ from the frozen BASE70 comparison")
    calendar = pd.read_parquet(seed_stability.EXECUTION_CALENDAR_PATH)
    if len(calendar) != 20_735 or calendar.market_start_utc.duplicated().any():
        raise RuntimeError("The frozen economic calendar is not the exact 20,735-row common calendar")
    calendar["market_start_utc"] = pd.to_datetime(calendar.market_start_utc, utc=True, errors="raise")
    market_rows, period_summary = research._asof_market_rows(calendar, base70.START_UTC, base70.CUTOFF_UTC)
    if len(market_rows) != 20_735 or market_rows.market_start_utc.max() >= OFFICIAL_CUTOFF:
        raise RuntimeError("The saved economic calendar is outside development or has changed")
    return market_rows, period_summary, research._snapshot_map(t59), research._snapshot_map(t60), frozen_snapshots


def _run_one_policy(name: str, predictions: pd.DataFrame, market_rows: pd.DataFrame, period_summary: dict, snapshot_map: dict, decision_map: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    probability_map = predictions.set_index("market_start_utc")[name]
    rows = market_rows.copy()
    rows["p_candidate_platt"] = rows.market_start_utc.map(probability_map)
    if rows.p_candidate_platt.isna().any():
        raise RuntimeError(f"{name} OOF does not cover every common development policy row")
    variant_market_map = research._snapshot_map(rows.loc[rows.condition_id.notna()])
    summary, trades, decisions = base70.replay._portfolio_for_policy(
        "fixed_5_usd",
        rows,
        snapshot_map,
        variant_market_map,
        decision_snapshot_by_id=decision_map,
        minimum_net_ev_fraction=0.01,
        output_policy=name,
        causal_intent_mode=True,
        allow_unresolved_outcomes=True,
        independent_order_mode=True,
    )
    equity_path = summary.pop("_equity_path")
    summary["accounting_audit"] = base70.replay._portfolio_accounting_audit(
        summary, trades, decisions, equity_path, snapshot_map, decision_map,
        allow_unresolved_outcomes=True, independent_order_mode=True,
    )
    trade_frame = pd.DataFrame(trades)
    decision_frame = pd.DataFrame(decisions)
    settled = trade_frame.loc[trade_frame.settlement_status.eq("settled")] if not trade_frame.empty else trade_frame
    pnl = float(pd.to_numeric(settled.net_pnl_usd, errors="coerce").sum()) if not settled.empty else 0.0
    turnover = float(pd.to_numeric(settled.gross_usd, errors="coerce").sum()) if not settled.empty else 0.0
    intents = int(pd.to_numeric(decision_frame.get("net_ev_fraction_at_decision", pd.Series(dtype=float)), errors="coerce").ge(0.01).sum())
    summary.update({
        "variant": name,
        "decision_count": int(len(decision_frame)),
        "intent_count": intents,
        "fill_count": int(len(trade_frame)),
        "settled_fill_count": int(len(settled)),
        "unresolved_fill_count": int(len(trade_frame) - len(settled)),
        "settled_turnover_usd": turnover,
        "net_pnl_usd": pnl,
        "fees_paid_estimated_usd": float(pd.to_numeric(trade_frame.fee_usd, errors="coerce").sum()) if not trade_frame.empty else 0.0,
        "pnl_per_settled_turnover": pnl / turnover if turnover else None,
        "period": period_summary,
    })
    if len(decision_frame) != 20_735 or not base70.summary_accounting_passed(summary):
        raise RuntimeError(f"Policy replay/accounting audit failed for {name}")
    return summary, trade_frame, decision_frame


def _economic_evaluation(selected: dict, control: dict, manifest: dict) -> pd.DataFrame:
    info_manifest = json.loads(seed_stability.BASE_INFO_DIR.joinpath("run_manifest.json").read_text(encoding="utf-8"))
    market_rows, period_summary, snapshot_map, decision_map, snapshot_identity = _prepare_economic_inputs(info_manifest)
    frames = []
    for role, config in (("control", control), ("selected", selected)):
        for seed in MODEL_SEEDS:
            result, oof = _read_trial_seed(config["config_id"], seed)
            name = f"{role}_seed_{seed}"
            predictions = oof[["market_start_utc", "p_model_raw"]].rename(columns={"p_model_raw": name})
            summary, trades, decisions = _run_one_policy(
                name, predictions, market_rows, period_summary, snapshot_map, decision_map,
            )
            variant_dir = OUTPUT_DIR / "economics" / name
            _atomic_parquet(decisions, variant_dir / "decisions.parquet")
            _atomic_parquet(trades, variant_dir / "trades.parquet")
            _atomic_json(variant_dir / "summary.json", summary)
            frames.append({
                "role": role, "seed": seed, "config_id": config["config_id"],
                "best_iteration": result["best_iteration"],
                "decision_count": summary["decision_count"], "intents": summary["intent_count"],
                "fills": summary["fill_count"], "settled_fills": summary["settled_fill_count"],
                "unresolved_fills": summary["unresolved_fill_count"],
                "turnover_usd": summary["settled_turnover_usd"],
                "fees_usd": summary["fees_paid_estimated_usd"], "net_pnl_usd": summary["net_pnl_usd"],
                "pnl_per_turnover": summary["pnl_per_settled_turnover"],
                "accounting_audit_passed": base70.summary_accounting_passed(summary),
                "decision_snapshot_hash": snapshot_identity["t60_sha256"],
                "execution_snapshot_hash": snapshot_identity["t59_sha256"],
            })
    result_frame = pd.DataFrame(frames)
    result_frame.to_csv(OUTPUT_DIR / "economics_by_seed.csv", index=False)
    manifest["stages"]["official_development_evaluation"] = {
        "status": "completed",
        "summary": {
            "policy_variants": len(result_frame),
            "calendar_rows_per_variant": 20_735,
            "snapshot_identity": snapshot_identity,
            "stake_usd": 5.0,
            "minimum_net_ev_fraction": 0.01,
            "decision_offset_seconds": -60,
            "execution_offset_seconds": -59,
            "all_accounting_audits_passed": bool(result_frame.accounting_audit_passed.all()),
        },
        "outputs": [
            _record(OUTPUT_DIR / "economics_by_seed.csv"),
            *[_record(path) for path in sorted((OUTPUT_DIR / "economics").glob("*/summary.json"))],
        ],
    }
    manifest["active_stage"] = None
    manifest["status"] = "completed"
    manifest["next_step"] = "No refit or holdout evaluation is authorized by this study."
    _write_manifest(manifest)
    return result_frame


def _historical_economics() -> dict:
    summary_path = seed_stability.OUTPUT_DIR / "economics" / "SINGLE" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "fills": int(summary["fill_count"]),
        "settled_fills": int(summary["settlement_count"]),
        "turnover_usd": float(summary["settled_turnover_usd"]),
        "fees_usd": float(summary["fees_paid_estimated_usd"]),
        "net_pnl_usd": float(summary["net_pnl_usd"]),
        "pnl_per_turnover": summary["pnl_per_settled_turnover"],
    }


def _write_report(protocol: dict, manifest: dict, study: optuna.study.Study, selection: dict, proxy: pd.DataFrame, official: pd.DataFrame, fold_quality: pd.DataFrame, economics: pd.DataFrame) -> None:
    complete = _complete_trial_rows(study, protocol)
    chosen = next(row for row in complete if row["trial_number"] == selection["selected_trial_number"])
    control = next(row for row in complete if row["trial_number"] == selection["control_trial_number"])
    alternatives = [row for row in complete if row["kind"] == "candidate"]
    best_alternative = min(alternatives, key=lambda row: (row["trial_objective"], row["trial_number"]))
    base_num_leaves = int(control["parameters"]["num_leaves"])
    lower_leaf_candidates = [row for row in alternatives if int(row["parameters"]["num_leaves"]) < base_num_leaves]
    higher_leaf_candidates = [row for row in alternatives if int(row["parameters"]["num_leaves"]) > base_num_leaves]
    best_lower_leaf = min(lower_leaf_candidates, key=lambda row: (row["trial_objective"], row["trial_number"]))
    best_higher_leaf = min(higher_leaf_candidates, key=lambda row: (row["trial_objective"], row["trial_number"]))
    selected_proxy = proxy.loc[proxy.role.eq("selected")].set_index("seed")
    control_proxy = proxy.loc[proxy.role.eq("control")].set_index("seed")
    selected_official = official.loc[official.role.eq("selected")].set_index("seed")
    control_official = official.loc[official.role.eq("control")].set_index("seed")
    selected_econ = economics.loc[economics.role.eq("selected")].set_index("seed")
    control_econ = economics.loc[economics.role.eq("control")].set_index("seed")
    seed_deltas = np.asarray([
        chosen["seed_scores"][str(seed)] - control["seed_scores"][str(seed)]
        for seed in MODEL_SEEDS
    ], dtype=np.float64)
    chosen_params = chosen["parameters"]
    boundary_hits = (
        "not applicable: selected setting is the original control outside candidate levels"
        if chosen["kind"] == "control"
        else {name: chosen_params[name] in (values[0], values[-1]) for name, values in PARAMETER_LEVELS.items()}
    )
    alternative_boundary_hits = {
        name: best_alternative["parameters"][name] in (values[0], values[-1])
        for name, values in PARAMETER_LEVELS.items()
    }
    alternative_seed_deltas = np.asarray([
        best_alternative["seed_scores"][str(seed)] - control["seed_scores"][str(seed)]
        for seed in MODEL_SEEDS
    ], dtype=np.float64)
    lower_leaf_deltas = np.asarray([
        best_lower_leaf["seed_scores"][str(seed)] - control["seed_scores"][str(seed)]
        for seed in MODEL_SEEDS
    ], dtype=np.float64)
    higher_leaf_deltas = np.asarray([
        best_higher_leaf["seed_scores"][str(seed)] - control["seed_scores"][str(seed)]
        for seed in MODEL_SEEDS
    ], dtype=np.float64)
    old = _historical_economics()
    hist_proxy = proxy.loc[proxy.role.eq("historical_SINGLE_31")].iloc[0]
    hist_official = official.loc[official.role.eq("historical_SINGLE_31")].iloc[0]
    trial_table = pd.DataFrame([
        {
            "trial": row["trial_number"], "config_id": row["config_id"],
            **row["parameters"], "objective": row["trial_objective"],
            "iterations_20261015": row["seed_best_iterations"]["20261015"],
            "iterations_20261016": row["seed_best_iterations"]["20261016"],
            "iterations_20261017": row["seed_best_iterations"]["20261017"],
        }
        for row in complete
    ]).sort_values(["objective", "trial"], kind="stable")
    trial_table.to_csv(OUTPUT_DIR / "completed_trial_summary.csv", index=False)
    proxy.to_csv(OUTPUT_DIR / "proxy_quality_by_seed.csv", index=False)
    official.to_csv(OUTPUT_DIR / "official_development_quality_by_seed.csv", index=False)
    fold_quality.to_csv(OUTPUT_DIR / "proxy_fold_quality_control_selected.csv", index=False)
    economics.to_csv(OUTPUT_DIR / "economics_by_seed.csv", index=False)
    report = [
        "# BASE70 main-model capacity tuning — 2026-10-19",
        "",
        "## Scope and result",
        "",
        "A single development-only CV study tested whether the inherited main-model settings restrict the existing 70-feature matrix. It changed only `num_leaves`, `min_data_in_leaf`, `lambda_l1`, and `lambda_l2`; the target, feature order, common mask, folds, purge, weights, learning rate, feature fraction, GPU backend, and `max_bin` stayed frozen. The candidate schedule was frozen before training. The study did not read holdout outcomes or use official labels/economics for selection.",
        "",
        f"BASE70 inherited the same effective parameter dictionary as the BASE64/main-model run (`{_relative(seed_stability.BASE_CV_SUMMARY_PATH)}`). The earlier BASE70 seed-stability fits used 31 fixed rounds without early stopping; FULL_FEATURES also fixed 31 rounds and changed `feature_fraction`. The original B0 summary records 500 maximum rounds and patience 50 but says its full history was not persisted after a post-fit NameError. Its saved 31-tree boosters were therefore not treated as completed early-stopping runs; the three-seed control was retrained.",
        "",
        f"The prior empty-partition audit is closed with evidence limited to **three UTC hours**; no remaining PMXT archives were downloaded. See [that report](btc_empty_partition_provenance_20261019.md).",
        "",
        f"Frozen matrix: {protocol['data']['rows']:,} rows × {protocol['data']['features']} features; shared OOF {protocol['data']['oof_rows_expected']:,} rows; ten chronological folds; weight SHA-256 `{protocol['data']['weights_sha256']}`. The trial score is the arithmetic mean over all three seed-specific minima of mean unweighted fold log-loss + 0.5 × population fold standard deviation. No best seed is chosen and no average-probability ensemble enters selection.",
        "",
        "## Frozen study and CV selection",
        "",
        f"The control was trial {control['trial_number']} with original parameters and reselected iterations: {control['parameters']}. The selected configuration was trial {chosen['trial_number']}: {chosen_params}; mean seed CV criterion {chosen['trial_objective']:.8f} versus control {control['trial_objective']:.8f}. Difference (selected − control) across seeds: mean {np.mean(seed_deltas):+.8f}, population SD {np.std(seed_deltas):.8f}, range [{np.min(seed_deltas):+.8f}, {np.max(seed_deltas):+.8f}]. Lower is better. Improvement in the seed criterion occurred in {int(np.sum(seed_deltas < 0))}/3 seeds.",
        "",
        f"Boundary hits for the selected setting: `{boundary_hits}`. Selected per-seed tree counts: `{chosen['seed_best_iterations']}`; control per-seed counts: `{control['seed_best_iterations']}`. Complete trials: {len(complete)}; started configurations: {manifest['training_counters']['started_configurations']}; incomplete trials excluded: {manifest['training_counters']['incomplete_configurations']}. Pruning was disabled. Study elapsed time: {manifest['study_elapsed_seconds']:.1f}s; peak process working set: {manifest.get('peak_process_working_set_gib', 'not recorded')} GiB.",
        "",
        f"Best completed candidate alternative: trial {best_alternative['trial_number']} `{best_alternative['config_id']}`, parameters {best_alternative['parameters']}, criterion {best_alternative['trial_objective']:.8f} (delta versus control {best_alternative['trial_objective'] - control['trial_objective']:+.8f}). Its seed scores {best_alternative['seed_scores']} versus control {control['seed_scores']}; mean selected-minus-control delta {np.mean(alternative_seed_deltas):+.8f}, population SD {np.std(alternative_seed_deltas):.8f}, improvement in {int(np.sum(alternative_seed_deltas < 0))}/3 seeds. No completed candidate beat the control. Candidate boundary hits: `{alternative_boundary_hits}`.",
        f"For the completed candidates below the inherited 81 leaves ({len(lower_leaf_candidates)} trials), the best used {best_lower_leaf['parameters']['num_leaves']} leaves and scored {best_lower_leaf['trial_objective']:.8f} (delta {best_lower_leaf['trial_objective'] - control['trial_objective']:+.8f}; seed deltas {lower_leaf_deltas.tolist()}, better in {int(np.sum(lower_leaf_deltas < 0))}/3). For the higher 127-leaf group ({len(higher_leaf_candidates)} trials), the best scored {best_higher_leaf['trial_objective']:.8f} (delta {best_higher_leaf['trial_objective'] - control['trial_objective']:+.8f}; seed deltas {higher_leaf_deltas.tolist()}, better in {int(np.sum(higher_leaf_deltas < 0))}/3). Other candidate parameters co-varied within the frozen grid.",
        "",
        "### Proxy OOF quality",
        "",
        "The table reports all three seeds for control and selected configuration; historical `SINGLE_31` is listed separately. Seed-level proxy metrics are computed on the common 318,560-row OOF sample.",
        "",
        "| Role | Seed | Trees | CV criterion | Proxy log-loss | Brier | AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in proxy.itertuples(index=False):
        crit = "—" if pd.isna(row.cv_selection_objective) else f"{row.cv_selection_objective:.8f}"
        report.append(f"| {row.role} | {row.seed} | {row.best_iteration} | {crit} | {row.log_loss:.8f} | {row.brier:.8f} | {row.auc:.6f} |")
    report.extend([
        "",
        "The full per-seed, per-fold log-loss/Brier/AUC table and every iteration’s mean, fold SD, and criterion are saved in the local trial checkpoints. The report manifest records their paths and hashes.",
        "",
        "### Paired proxy changes and fold results",
        "",
        "Changes are selected minus control on each same seed; negative log-loss/Brier and positive AUC are better.",
        "",
        "| Seed | CV criterion delta | Proxy log-loss delta | Brier delta | AUC delta |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for seed in MODEL_SEEDS:
        p_delta = {
            metric: float(selected_proxy.loc[seed, metric]) - float(control_proxy.loc[seed, metric])
            for metric in ("log_loss", "brier", "auc")
        }
        report.append(
            f"| {seed} | {seed_deltas[list(MODEL_SEEDS).index(seed)]:+.8f} | "
            f"{p_delta['log_loss']:+.8f} | {p_delta['brier']:+.8f} | {p_delta['auc']:+.6f} |"
        )
    proxy_delta_matrix = np.asarray([
        [
            selected_proxy.loc[seed, metric] - control_proxy.loc[seed, metric]
            for metric in ("log_loss", "brier", "auc")
        ]
        for seed in MODEL_SEEDS
    ], dtype=np.float64)
    report.extend([
        f"| Mean (population SD) | {np.mean(seed_deltas):+.8f} ({np.std(seed_deltas):.8f}) | "
        f"{np.mean(proxy_delta_matrix[:, 0]):+.8f} ({np.std(proxy_delta_matrix[:, 0]):.8f}) | "
        f"{np.mean(proxy_delta_matrix[:, 1]):+.8f} ({np.std(proxy_delta_matrix[:, 1]):.8f}) | "
        f"{np.mean(proxy_delta_matrix[:, 2]):+.6f} ({np.std(proxy_delta_matrix[:, 2]):.6f}) |",
        "",
        "Per-fold proxy metrics for the control and selected setting:",
        "",
        "| Role | Seed | Fold | Trees | Validation rows | Log-loss | Brier | AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in fold_quality.itertuples(index=False):
        report.append(
            f"| {row.role} | {row.seed} | {row.fold_id} | {row.best_iteration} | "
            f"{row.validation_rows} | {row.log_loss:.8f} | {row.brier:.8f} | {row.auc:.6f} |"
        )
    report.extend([
        "",
        "Every evaluated iteration's mean, fold SD, and criterion are saved in each local seed checkpoint, including rounds after the best iteration up to early stopping. The report manifest records their paths and hashes.",
        "",
        "## Official development and fixed policy, after selection",
        "",
        "Only after `selection.json` was written, the selected configuration and control were scored on the existing official development sample and replayed on the same saved 20,735-row calendar. Results use each seed separately and raw model probabilities with the existing $5 gross stake, 1% net-EV hurdle, T−60 decision, T−59 execution, snapshots, fee rules, and accounting audit. Holdout starts are excluded.",
        "",
        f"Frozen snapshot identities: decision T−60 `{selected_econ.iloc[0]['decision_snapshot_hash']}`; execution T−59 `{selected_econ.iloc[0]['execution_snapshot_hash']}`. The policy calendar has 20,735 rows; 20,718 had official outcomes and cash releases by cutoff, while 17 missing outcomes remained excluded.",
        "",
        "Missing source inputs remain explicit as rejections rather than fills:",
        "",
        "| Seed | Calendar rows | Outcome available | Missing official outcome | No Gamma market record | No priceable native ask |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for seed in MODEL_SEEDS:
        summary_path = OUTPUT_DIR / "economics" / f"selected_seed_{seed}" / "summary.json"
        replay_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report.append(
            f"| {seed} | {replay_summary['period']['market_rows_in_period']} | "
            f"{replay_summary['period']['official_outcomes_and_cash_releases_available_by_cutoff']} | "
            f"{replay_summary['period']['missing_official_outcome_rows']} | "
            f"{replay_summary['rejections_by_reason'].get('no_gamma_market_record', 0)} | "
            f"{replay_summary['rejections_by_reason'].get('no_priceable_native_ask_side', 0)} |"
        )
    report.extend([
        "",
        "| Role | Seed | Trees | Official LL | Brier | AUC | Fills | Settled | Fees USD | Net PnL USD | PnL / turnover | Audit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for seed in MODEL_SEEDS:
        for role, frame, econ in (("control", control_official, control_econ), ("selected", selected_official, selected_econ)):
            q = frame.loc[seed]
            e = econ.loc[seed]
            report.append(
                f"| {role} | {seed} | {int(q.best_iteration)} | {q.log_loss:.8f} | {q.brier:.8f} | {q.auc:.6f} | {int(e.fills)} | {int(e.settled_fills)} | {e.fees_usd:.2f} | {e.net_pnl_usd:.2f} | {e.pnl_per_turnover:.4%} | {bool(e.accounting_audit_passed)} |"
            )
    report.extend([
        "",
        "Paired changes (selected minus control) on the same official target and calendar:",
        "",
        "| Seed | Official LL delta | Brier delta | AUC delta | Fills delta | Net PnL delta USD | PnL/turnover delta pp |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for seed in MODEL_SEEDS:
        q_delta = {
            metric: float(selected_official.loc[seed, metric]) - float(control_official.loc[seed, metric])
            for metric in ("log_loss", "brier", "auc")
        }
        e_delta_fills = int(selected_econ.loc[seed, "fills"]) - int(control_econ.loc[seed, "fills"])
        e_delta_pnl = float(selected_econ.loc[seed, "net_pnl_usd"]) - float(control_econ.loc[seed, "net_pnl_usd"])
        e_delta_ppt = 100 * (
            float(selected_econ.loc[seed, "pnl_per_turnover"])
            - float(control_econ.loc[seed, "pnl_per_turnover"])
        )
        report.append(
            f"| {seed} | {q_delta['log_loss']:+.8f} | {q_delta['brier']:+.8f} | {q_delta['auc']:+.6f} | "
            f"{e_delta_fills:+d} | {e_delta_pnl:+.2f} | {e_delta_ppt:+.3f} |"
        )
    report.extend([
        "",
        f"| historical SINGLE_31 | 20261015 | 31 | {hist_official.log_loss:.8f} | {hist_official.brier:.8f} | {hist_official.auc:.6f} | {old['fills']} | {old['settled_fills']} | {old['fees_usd']:.2f} | {old['net_pnl_usd']:.2f} | {old['pnl_per_turnover']:.4%} | reused saved audited ledger |",
        "",
        "A proxy improvement is not sufficient if it does not carry to the official development target and policy. These development results are conditional on earlier feature selection and the current tuning study; they are not independent confirmation of an edge. No holdout evaluation or deployment refit was run.",
        "",
        "## Artifacts and limitations",
        "",
        f"Frozen protocol: `{_relative(PROTOCOL_PATH)}`. Run manifest: `{_relative(MANIFEST_PATH)}`. Optuna SQLite study: `{_relative(STUDY_DB_PATH)}`. CV-only selection record: `{_relative(OUTPUT_DIR / 'selection.json')}`. Saved fold models and OOF files are under `{_relative(OUTPUT_DIR / 'trials')}`; official development quality and replay outputs are under `{_relative(OUTPUT_DIR)}`.",
        "",
        f"Actual CV work: {manifest['training_counters']['started_fold_fits']} fold fits started; {manifest['training_counters']['completed_fold_fits']} fully completed; {manifest['training_counters']['started_configurations']} configurations started, {manifest['training_counters']['completed_configurations']} completed. Peak process working set was {manifest.get('peak_process_working_set_gib', 'not recorded')} GiB. The study was limited to its frozen candidate schedule and time budget; no space extension was run.",
        "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    manifest["stages"]["report"] = {"status": "completed", "outputs": [_record(REPORT_PATH)]}
    _write_manifest(manifest)


def main() -> dict:
    protocol = _build_protocol()
    resources = _resource_snapshot()
    manifest = _load_or_create_manifest(protocol, resources)
    current_runner_hash = _sha256(Path(__file__))
    frozen_runner_hash = protocol["code_identity"]["runner_sha256"]
    if current_runner_hash != frozen_runner_hash:
        manifest["postfit_report_generator_identity"] = {
            "path": _relative(Path(__file__)),
            "sha256": current_runner_hash,
            "scope": "numeric report generation only; CV, selection, and replay artifacts were already completed",
        }
        _write_manifest(manifest)
    if manifest.get("status") == "completed":
        return manifest
    previous_clock = json.loads(_clock_path().read_text(encoding="utf-8"))["elapsed_seconds"] if _clock_path().is_file() else float(manifest.get("study_elapsed_seconds", 0.0))
    clock = StudyClock(previous_clock)
    cv_summary = json.loads(seed_stability.BASE_CV_SUMMARY_PATH.read_text(encoding="utf-8"))
    common_summary = json.loads(seed_stability.COMMON_SUMMARY_PATH.read_text(encoding="utf-8"))
    with taker.PeakMemorySampler() as memory:
        frame, x, target, weights, folds, single_check = seed_stability._load_training_inputs(
            cv_summary, common_summary, verify_single=True,
        )
        if not single_check or not single_check.get("passed"):
            raise RuntimeError("The saved historical SINGLE OOF/fold models did not verify")
        manifest["stages"]["source_verification"] = {
            "status": "completed",
            "summary": {
                "rows": int(len(frame)), "features": int(x.shape[1]), "folds": len(folds),
                "weights_sha256": cv_summary["training_weight_sha256"],
                "purge_verified": True, "historical_single_31_reproduction": single_check,
            },
            "outputs": [protocol["source_records"][name] for name in ("matrix", "folds", "single_cv_summary", "single_oof")],
            "elapsed_seconds": 0.0,
        }
        manifest["peak_process_working_set_gib"] = round(memory.peak_bytes / 2**30, 3)
        _write_manifest(manifest)
        study, budget_exhausted = _run_study(
            protocol, manifest, frame, x, target, weights, folds,
            list(cv_summary["feature_order"]), clock,
        )
        manifest["peak_process_working_set_gib"] = round(memory.peak_bytes / 2**30, 3)
    if manifest.get("status") == "interrupted_resumable":
        return manifest
    selection_path = OUTPUT_DIR / "selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    else:
        selection = _freeze_selection(study, protocol, manifest)
    rows = _complete_trial_rows(study, protocol)
    selected = next(row for row in rows if row["config_id"] == selection["selected_config_id"])
    control = next(row for row in rows if row["config_id"] == "control_original")
    info_manifest = json.loads((seed_stability.BASE_INFO_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    proxy, official, fold_quality = _development_quality(selected, control, {}, cv_summary)
    economics = _economic_evaluation(selected, control, manifest)
    _write_report(protocol, manifest, study, selection, proxy, official, fold_quality, economics)
    manifest["budget_exhausted"] = bool(budget_exhausted)
    manifest["peak_process_working_set_gib"] = max(
        float(manifest.get("peak_process_working_set_gib", 0.0)),
    )
    manifest["stages"]["source_identities"] = {
        "status": "completed",
        "summary": {"official_development_cutoff_exclusive_utc": OFFICIAL_CUTOFF.isoformat(), "holdout_evaluated": False},
        "outputs": [protocol["source_records"]["base70_information_predictions"], protocol["source_records"]["execution_calendar"]],
    }
    _write_manifest(manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, default=str))
