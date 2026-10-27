"""Run the frozen chronological E0039 BTC A/B/C/D reconstruction."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/research/btc_chronology_reconstruction_20261027_v7.json"
PREVIOUS_PROTOCOL_PATH = ROOT / "configs/research/btc_chronology_reconstruction_20261027_v6.json"
OUTPUT_ROOT = ROOT / "data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027"
CANDIDATES = ("A", "B", "C", "D")
LIGHTGBM_SEEDS = (20261101, 20261102, 20261103)
RESERVOIR_SEEDS = (20261023, 20261024, 20261025)
MEMORY_NAMES = tuple(f"simple_memory_PCA8_pc{index:02d}" for index in range(8))
FLOW_NAMES = (
    "taker_Z_1m", "taker_V_1m", "taker_Z_5m", "taker_V_5m",
    "taker_Z_15m", "taker_V_15m",
)
MINUTE_NS = 60_000_000_000
GAMMA_COLUMNS = (
    "condition_id", "market_start_utc", "target_polymarket_up", "resolved_at_utc",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def strict_training_mask(opened, target, label_available, cutoff):
    """Return labels that were available strictly before an exclusive cutoff."""
    import numpy as np
    import pandas as pd

    opened = pd.DatetimeIndex(pd.to_datetime(opened, utc=True, errors="raise"))
    available = pd.DatetimeIndex(pd.to_datetime(label_available, utc=True, errors="coerce"))
    target = np.asarray(target, dtype=np.float64)
    cutoff = pd.Timestamp(cutoff)
    if (
        cutoff.tzinfo is None or target.ndim != 1
        or len(opened) != len(available) or len(opened) != len(target)
    ):
        raise ValueError("invalid training timestamps, labels, or exclusive cutoff")
    labeled = np.isfinite(target)
    if not np.isin(target[labeled], [0.0, 1.0]).all():
        raise ValueError("training targets must be binary when present")
    return (
        labeled & available.notna()
        & (opened < cutoff.tz_convert("UTC"))
        & (available < cutoff.tz_convert("UTC"))
    )


def validate_oof_provenance(
        frame, *, fit_cutoff, prediction_start, prediction_end,
        expected_provenance: str = "fold_local_preselection_chronological_oof",
):
    """Reject an OOF record whose pipeline or labels cross its forecast period."""
    import pandas as pd

    required = {
        "market_start_utc", "prediction_provenance", "fit_cutoff_exclusive_utc",
        "latest_training_label_available_at_utc", "model_identity_sha256",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OOF provenance columns missing: {sorted(missing)}")
    starts = pd.to_datetime(frame.market_start_utc, utc=True, errors="raise")
    cutoff = pd.Timestamp(fit_cutoff).tz_convert("UTC")
    pred_start = pd.Timestamp(prediction_start).tz_convert("UTC")
    pred_end = pd.Timestamp(prediction_end).tz_convert("UTC")
    fit_cutoffs = pd.to_datetime(frame.fit_cutoff_exclusive_utc, utc=True, errors="raise")
    latest_labels = pd.to_datetime(
        frame.latest_training_label_available_at_utc, utc=True, errors="raise",
    )
    if not starts.ge(pred_start).all() or not starts.lt(pred_end).all():
        raise ValueError("OOF predictions fall outside their frozen forecast interval")
    if not fit_cutoffs.eq(cutoff).all() or not fit_cutoffs.le(starts).all():
        raise ValueError("OOF model fit cutoff is not earlier than every prediction")
    if not latest_labels.lt(fit_cutoffs).all():
        raise ValueError("OOF model used a label unavailable before its fit cutoff")
    if not frame.prediction_provenance.eq(expected_provenance).all():
        raise ValueError("OOF predictions have an unapproved provenance")
    if frame.model_identity_sha256.isna().any():
        raise ValueError("OOF predictions are missing model identities")
    return True


def _resource_snapshot(minimum_free_memory_gb: float, *, require_gpu: bool) -> dict:
    """Report capacity and block when another substantial Python workload is active."""
    snapshot = {
        "logical_cpus": os.cpu_count(),
        "disk_free_gb": round(shutil.disk_usage(ROOT).free / (1024 ** 3), 2),
        "available_memory_gb": None,
        "total_memory_gb": None,
        "python_processes": [],
        "memory_blocking_python_pids": [],
        "gpu_query": "not_checked",
        "gate_failures": [],
    }
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            snapshot["available_memory_gb"] = round(status.ullAvailPhys / (1024 ** 3), 2)
            snapshot["total_memory_gb"] = round(status.ullTotalPhys / (1024 ** 3), 2)
    except (AttributeError, OSError):
        try:
            import psutil

            memory = psutil.virtual_memory()
            snapshot["available_memory_gb"] = round(memory.available / (1024 ** 3), 2)
            snapshot["total_memory_gb"] = round(memory.total / (1024 ** 3), 2)
        except ImportError:
            pass
    if os.name == "nt":
        command = (
            "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } "
            "| Select-Object ProcessId,WorkingSetSize | ConvertTo-Json -Compress"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            snapshot["gate_failures"].append("Cannot verify active Python workloads before a heavy fit")
        else:
            raw = result.stdout.strip()
            if raw:
                parsed = json.loads(raw)
                processes = parsed if isinstance(parsed, list) else [parsed]
                snapshot["python_processes"] = [
                    {
                        "pid": int(process["ProcessId"]),
                        "working_set_gb": round(int(process["WorkingSetSize"]) / (1024 ** 3), 2),
                    }
                    for process in processes if int(process["ProcessId"]) != os.getpid()
                ]
                snapshot["memory_blocking_python_pids"] = [
                    item["pid"] for item in snapshot["python_processes"]
                    if item["working_set_gb"] >= 1.0
                ]
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            cwd=ROOT, capture_output=True, text=True, check=False, timeout=10,
        )
        snapshot["gpu_query"] = gpu.stdout.strip() if gpu.returncode == 0 else "unavailable"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        snapshot["gpu_query"] = "unavailable"
    memory = snapshot["available_memory_gb"]
    if memory is None or memory < float(minimum_free_memory_gb):
        snapshot["gate_failures"].append(
            "E0039 fit blocked by available RAM: "
            f"{memory} GiB available; requires {minimum_free_memory_gb} GiB"
        )
    if snapshot["memory_blocking_python_pids"]:
        snapshot["gate_failures"].append(
            "E0039 fit blocked while other Python workloads use at least 1 GiB each: "
            f"PIDs={snapshot['memory_blocking_python_pids']}"
        )
    if snapshot["disk_free_gb"] < 80:
        snapshot["gate_failures"].append(
            f"E0039 fit blocked by low free disk: {snapshot['disk_free_gb']} GiB"
        )
    if require_gpu and snapshot["gpu_query"] == "unavailable":
        snapshot["gate_failures"].append(
            "E0039 fit blocked because the frozen GPU backend is unavailable"
        )
    return snapshot


def _verify_frozen_inputs(protocol: dict) -> dict:
    if protocol.get("experiment_id") != "E0039" or protocol.get("status") != "FROZEN_BEFORE_FIT":
        raise RuntimeError("E0039 v7 protocol is not frozen before fitting")
    previous = _sha256(PREVIOUS_PROTOCOL_PATH)
    if previous != protocol["previous_protocol_sha256"]:
        raise RuntimeError("E0039 v6 parent protocol hash changed")
    if protocol.get("exclusions", {}).get("holdout_read_for_E0039") is not False:
        raise RuntimeError("E0039 protocol does not exclude the protected holdout")
    verified = {"previous_protocol": previous}
    for path_text, expected in protocol["frozen_inputs"].items():
        path = ROOT / path_text
        if not path.is_file():
            raise FileNotFoundError(f"Frozen E0039 input not found: {path_text}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen E0039 input hash changed: {path_text}")
        verified[path_text] = actual
    for path_text, expected in protocol["implementation_hashes"].items():
        actual = _sha256(ROOT / path_text)
        if actual != expected:
            raise RuntimeError(f"E0039 implementation hash differs from v7 freeze: {path_text}")
        verified[path_text] = actual
    return verified


def _save_checkpoint(stage_path: Path, stage_identity: dict, outputs: list[Path], payload: dict) -> dict:
    records = []
    for path in outputs:
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(f"Cannot checkpoint missing stage output: {path}")
        records.append({"path": path.resolve().relative_to(ROOT).as_posix(), "sha256": _sha256(path)})
    checkpoint = {
        "identity": _canonical_hash(stage_identity),
        "identity_payload": stage_identity,
        "outputs": records,
        "payload": payload,
    }
    _atomic_json(stage_path, checkpoint)
    return checkpoint


def _load_checkpoint(stage_path: Path, stage_identity: dict) -> dict | None:
    if not stage_path.is_file():
        return None
    checkpoint = json.loads(stage_path.read_text(encoding="utf-8"))
    if checkpoint.get("identity") != _canonical_hash(stage_identity):
        raise RuntimeError(f"Checkpoint identity mismatch: {stage_path}")
    for record in checkpoint.get("outputs", []):
        path = ROOT / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Checkpoint output checksum mismatch: {record['path']}")
    return checkpoint


def _build_selection_contract(source_contract: dict, fold: dict) -> dict:
    import pandas as pd

    contract = json.loads(json.dumps(source_contract))
    split = contract["chronological_split_frozen_before_tuning"]
    start = pd.Timestamp(fold["market_start_test_start_inclusive_utc"])
    end = pd.Timestamp(fold["market_start_test_end_exclusive_utc"])
    selection_cutoff = pd.Timestamp(fold["selection_cutoff_exclusive_utc"])
    if pd.Timestamp(split["feature_and_model_fit_start_utc"]) != pd.Timestamp(fold["fit_data_start_utc"]):
        raise RuntimeError("Frozen contract fit-data start differs from the E0039 fold")
    split.update({
        "fit_and_tuning_end_exclusive_utc": selection_cutoff.isoformat(),
        "calibration_start_utc": selection_cutoff.isoformat(),
        "calibration_end_exclusive_utc": (start - pd.Timedelta(minutes=1)).isoformat(),
        "first_external_test_market_start_utc": start.isoformat(),
        "first_external_test_nominal_decision_utc": (start - pd.Timedelta(minutes=1)).isoformat(),
        "external_test_market_start_last_inclusive_utc": (end - pd.Timedelta(minutes=5)).isoformat(),
        "external_test_nominal_decision_last_utc": (end - pd.Timedelta(minutes=6)).isoformat(),
        "test_history_status": (
            "retrospective conditional chronology; procedure family was previously exposed "
            "through development ending 2025-07-01; not independent confirmation"
        ),
    })
    return contract


def _selection_profile(protocol: dict, fold: dict, fold_root: Path, *, remaining_seconds: float) -> dict:
    source = json.loads(
        (ROOT / protocol["procedure_budgets"]["source_config"]).read_text(encoding="utf-8")
    )
    training_profiles = json.loads(
        (ROOT / "configs/btc_preopen_training.json").read_text(encoding="utf-8")
    )["profiles"]
    profile = dict(training_profiles["nightly_v1"])
    original_contract = json.loads((ROOT / "configs/btc_preopen_v1.json").read_text(encoding="utf-8"))
    contract = _build_selection_contract(original_contract, fold)
    contract_path = fold_root / "config" / "btc_preopen_contract.json"
    _atomic_json(contract_path, contract)
    output_dir = fold_root / "selection_pipeline"
    raw_path = ROOT / fold["raw_prefix_path"]
    flow_path = ROOT / fold["flow_prefix_path"]
    main_search = source["main_model_search"]
    generator_search = source["generator_search"]
    runtime = source["runtime"]
    if (
        int(runtime["threads"]) != int(protocol["procedure_budgets"]["threads"])
        or str(runtime["compute_backend"]).lower()
        != str(protocol["procedure_budgets"]["compute_backend"]).lower()
    ):
        raise RuntimeError("Frozen E0039 execution backend/threads differ from the source procedure")
    profile.update({
        "contract_path": contract_path.resolve().relative_to(ROOT).as_posix(),
        "dataset_profile": "BTC",
        "modeling_profile": "BTC",
        "indicator_fit_profile": training_profiles["nightly_v1"]["indicator_fit_profile"],
        "prediction_compute_budget_seconds": 1,
        "runtime_budget_seconds": max(1, int(remaining_seconds)),
        "stage_runtime_budgets_seconds": {
            "indicators": int(protocol["procedure_budgets"]["indicator_stage_seconds"]),
            "volume_profile": int(protocol["procedure_budgets"]["volume_profile_stage_seconds"]),
            "reaction_profile": int(protocol["procedure_budgets"]["reaction_profile_stage_seconds"]),
            "target_weights": int(protocol["procedure_budgets"]["target_weight_stage_seconds"]),
            "feature_selection": int(protocol["procedure_budgets"]["feature_selection_stage_seconds"]),
            "model_tuning_and_fit": int(protocol["procedure_budgets"]["model_tuning_and_fit_stage_seconds"]),
            "calibration_evaluation_bundle": 1,
        },
        "threads": int(protocol["procedure_budgets"]["threads"]),
        "seed": int(main_search["random_seed"]),
        "indicator_seed": int(source["reuse"]["indicator_artifact_source"]["indicator_seed"]),
        "compute_backend": str(runtime["compute_backend"]),
        "indicator_fit": {
            "population_size": int(protocol["procedure_budgets"]["indicator_population"]),
            "generations": int(protocol["procedure_budgets"]["indicator_generations"]),
            "minimum_generations": int(protocol["procedure_budgets"]["indicator_generations"]),
            "workers": int(protocol["procedure_budgets"]["indicator_workers"]),
        },
        "reaction_profile_trials": int(protocol["procedure_budgets"]["reaction_profile_trials"]),
        "volume_profile_trials": int(protocol["procedure_budgets"]["volume_profile_trials"]),
        "generator_checkpoint_trials": [24, 48, 96],
        "generator_trial_limit_is_hard": True,
        "generator_tuning_time_limit_seconds": int(protocol["procedure_budgets"]["reaction_profile_stage_seconds"]),
        "target_weight_candidates": int(protocol["procedure_budgets"]["target_weight_candidates"]),
        "feature_selection": {
            "max_sweep_evaluations": int(protocol["procedure_budgets"]["feature_selection_max_evaluations"]),
            "max_refinement_rounds": int(protocol["procedure_budgets"]["feature_selection_max_refinement_rounds"]),
            "permutation_feature_fraction": 0.25,
            "permutation_repeats": 1,
        },
        "required_selected_feature_count": 64,
        "model_trials": int(protocol["procedure_budgets"]["model_search_trials"]),
        "minimum_successful_trials": int(main_search["minimum_successful_trials"]),
        "maximum_estimators": int(protocol["procedure_budgets"]["model_search_max_estimators"]),
        "early_stopping_rounds": int(protocol["procedure_budgets"]["model_search_early_stopping_rounds"]),
        "cv_folds": int(protocol["procedure_budgets"]["model_search_cv_folds"]),
        "model_cv_folds": int(protocol["procedure_budgets"]["model_search_cv_folds"]),
        "generator_cv_folds": int(protocol["procedure_budgets"]["model_search_cv_folds"]),
        "weight_cv_folds": int(protocol["procedure_budgets"]["model_search_cv_folds"]),
        "selector_cv_folds": int(protocol["procedure_budgets"]["model_search_cv_folds"]),
        "generator_minimum_successful_trials": int(generator_search["minimum_successful_trials"]),
        "main_model_tuning_time_limit_seconds": int(main_search["time_limit_seconds"]),
        "main_model_final_fit_reserve_seconds": int(main_search["final_fit_reserve_seconds"]),
        "model_checkpoint_trials": list(main_search["checkpoint_trials"]),
        "model_trial_limit_is_hard": True,
        "model_search_max_estimators": int(protocol["procedure_budgets"]["model_search_max_estimators"]),
        "model_search_space_override": dict(main_search["search_space"]),
        "model_feature_families": source["model_feature_families"],
        "model_feature_candidate_features": list(FLOW_NAMES),
        "exclusive_label_availability": True,
        "causal_candidate_feature_path": flow_path.resolve().relative_to(ROOT).as_posix(),
        "causal_candidate_feature_columns": list(FLOW_NAMES),
        "chronology_stage_only": True,
        "raw_data_path": raw_path.resolve().relative_to(ROOT).as_posix(),
        "development_input_path": raw_path.resolve().relative_to(ROOT).as_posix(),
        "source_raw_data_path": "data/datasets/raw/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m_development_cutoff_20260701.csv",
        "development_cutoff_exclusive_utc": fold["market_start_test_end_exclusive_utc"],
        "official_markets_path": (fold_root / "not_read_official_markets.parquet").resolve().relative_to(ROOT).as_posix(),
        "output_dir": output_dir.resolve().relative_to(ROOT).as_posix(),
    })
    return profile


def _run_fold_selection(
        protocol: dict, fold: dict, fold_root: Path, fold_identity: str,
        *, remaining_seconds: float,
) -> dict:
    import pandas as pd
    import run_btc_preopen_experiment as pipeline

    result_path = fold_root / "selection_result.json"
    identity = {
        "fold_identity_sha256": fold_identity,
        "selection_cutoff_exclusive_utc": fold["selection_cutoff_exclusive_utc"],
        "raw_prefix_sha256": fold["raw_prefix_sha256"],
        "flow_prefix_sha256": fold["flow_prefix_sha256"],
        "pipeline_sha256": _sha256(ROOT / "run_btc_preopen_experiment.py"),
        "contract_template_sha256": _sha256(ROOT / "configs/btc_preopen_v1.json"),
    }
    checkpoint = _load_checkpoint(result_path, identity)
    if checkpoint:
        return checkpoint["payload"]
    profile = _selection_profile(
        protocol, fold, fold_root,
        remaining_seconds=remaining_seconds,
    )
    selected = pipeline.run_training_pipeline(
        profile_name=f"E0039_{fold['id']}", profile_override=profile,
    )
    if selected.get("status") != "fold_local_selection_completed":
        raise RuntimeError("Existing training pipeline did not stop at the requested selection boundary")
    cutoff = pd.Timestamp(fold["selection_cutoff_exclusive_utc"])
    latest_label = pd.Timestamp(selected["latest_training_label_available_at_utc"])
    if not latest_label < cutoff or not selected.get("exclusive_label_availability"):
        raise RuntimeError("A fold selection stage used a label not released strictly before its cutoff")
    if any(
        pd.Timestamp(value) != cutoff
        for value in selected.get("stage_fit_end_exclusive_utc", {}).values()
    ):
        raise RuntimeError("A generator/weight/feature/model selection stage used a different cutoff")
    if len(selected.get("selected_base_features", [])) != 64:
        raise RuntimeError("Fold-local generator/selection pipeline did not select 64 base features")
    if selected.get("model_feature_order", []) != list(selected["selected_base_features"]) + list(FLOW_NAMES):
        raise RuntimeError("Selected BASE70 order differs from the frozen base64 plus six flow features")
    payload = {
        **selected,
        "pipeline_run_root": selected["run_root"],
        "selected_feature_count": len(selected["selected_base_features"]),
    }
    _save_checkpoint(result_path, identity, [ROOT / selected["run_root"] / "chronology_selection_result.json"], payload)
    return payload


def _transform_names(protocol: dict) -> tuple[list[str], dict[int, list[str]], dict]:
    from run_btc_esn_expansion_20261021 import engine_runner

    transform_root = ROOT / protocol["transforms"]["directory"]
    memory = json.loads((transform_root / "simple_memory_projection.json").read_text(encoding="utf-8"))
    esn_bundles = {
        seed: json.loads((transform_root / f"esn_seed_{seed}.json").read_text(encoding="utf-8"))
        for seed in RESERVOIR_SEEDS
    }
    configs = {
        _canonical_hash(engine_runner._config_record(engine_runner._config_from_record(bundle["config"])))
        for bundle in esn_bundles.values()
    }
    if len(configs) != 1:
        raise RuntimeError("Frozen ESN projections do not share a single reservoir configuration")
    memory_names = list(MEMORY_NAMES)
    esn_names = {seed: list(esn_bundles[seed]["feature_names"]) for seed in RESERVOIR_SEEDS}
    if any(len(names) != 8 or len(set(names)) != 8 for names in esn_names.values()):
        raise RuntimeError("Frozen ESN projection does not provide exactly eight named features")
    return memory_names, esn_names, {"memory": memory, "esn": esn_bundles}


def _build_dynamic_feature_artifact(
    protocol: dict, fold: dict, fold_root: Path, fold_identity: str,
) -> tuple[Path, list[str], dict[int, list[str]]]:
    import numpy as np
    import pandas as pd
    from features.esn_reservoir import build_causal_esn_inputs
    import run_btc_candidate_holdout_20261022 as frozen_transforms
    import run_btc_esn_portfolio_20261020 as esn_portfolio
    from run_btc_esn_expansion_20261021 import engine_runner

    memory_names, esn_names, bundles = _transform_names(protocol)
    raw_path = ROOT / fold["raw_prefix_path"]
    output_path = fold_root / "dynamic_features.parquet"
    checkpoint_path = fold_root / "dynamic_features.checkpoint.json"
    identity = {
        "fold_identity_sha256": fold_identity,
        "raw_prefix_sha256": fold["raw_prefix_sha256"],
        "memory_transform_sha256": protocol["transform_hashes"]["memory"],
        "esn_transform_sha256": protocol["transform_hashes"]["esn"],
        "feature_code_sha256": protocol["implementation_hashes"]["features/esn_reservoir.py"],
        "memory_state_code_sha256": protocol["implementation_hashes"]["features/causal_memory_control.py"],
        "transform_runner_sha256": protocol["implementation_hashes"]["run_btc_candidate_holdout_20261022.py"],
    }
    existing = _load_checkpoint(checkpoint_path, identity)
    if existing:
        return output_path, memory_names, esn_names

    header = pd.read_csv(raw_path, nrows=0).columns
    selected_raw_columns = [
        column for column in (
            "Opened", "Open", "High", "Low", "Close", "Volume",
            "UM_BTCUSDT_Close", "futures_index_basis_rel_1m",
        ) if column in header
    ]
    required = {"Opened", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(selected_raw_columns):
        raise RuntimeError("Frozen chronology raw prefix lacks the project OHLCV schema")
    raw = pd.read_csv(raw_path, usecols=selected_raw_columns)
    raw["Opened"] = pd.to_datetime(raw["Opened"], utc=True, errors="raise")
    opened, input_frame = build_causal_esn_inputs(raw)
    raw_x = input_frame.to_numpy(dtype=np.float32, copy=True)
    raw_valid = esn_portfolio._valid_candles(raw)
    configs = {
        seed: engine_runner._config_from_record(bundles["esn"][seed]["config"])
        for seed in RESERVOIR_SEEDS
    }
    esn_config = configs[RESERVOIR_SEEDS[0]]
    memory, esn, runtime_states = frozen_transforms._generate_dynamic_features(
        opened.as_unit("ns").asi8,
        raw_x,
        raw_valid,
        bundles["memory"],
        esn_config,
        bundles["esn"],
        return_states=True,
    )
    del raw_x, input_frame, raw, runtime_states
    result = pd.DataFrame({"Opened": opened})
    for index, name in enumerate(memory_names):
        result[name] = memory[:, index]
    for seed, values in esn.items():
        for index, name in enumerate(esn_names[seed]):
            result[name] = values[:, index]
    _atomic_parquet(result, output_path)
    _save_checkpoint(checkpoint_path, identity, [output_path], {
        "rows": int(len(result)),
        "feature_count": int(len(memory_names) + sum(map(len, esn_names.values()))),
        "memory_columns": memory_names,
        "esn_columns_by_seed": {str(seed): names for seed, names in esn_names.items()},
        "state_resets": "invalid OHLCV or non-one-minute timestamp gap",
    })
    return output_path, memory_names, esn_names


def _load_fold_arrays(selection: dict, dynamic_path: Path, memory_names: list[str],
                      esn_names: dict[int, list[str]]):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import run_btc_preopen_experiment as pipeline

    feature_path = ROOT / selection["dataset_path"]
    feature_columns = list(selection["model_feature_order"])
    target_columns = [
        pipeline.TARGET_COL, pipeline.TARGET_AVAILABLE_COL, pipeline.DECISION_COL,
    ]
    schema_names = set(pq.ParquetFile(feature_path).schema_arrow.names)
    wanted = ["Opened", *feature_columns, *target_columns]
    missing = sorted(set(wanted).difference(schema_names))
    if missing:
        raise RuntimeError(f"Generated fold dataset is missing required columns: {missing}")
    frame = pd.read_parquet(feature_path, columns=wanted)
    dynamic_columns = [*memory_names, *(name for names in esn_names.values() for name in names)]
    dynamic = pd.read_parquet(dynamic_path, columns=["Opened", *dynamic_columns])
    opened = pd.DatetimeIndex(pd.to_datetime(frame["Opened"], utc=True, errors="raise"))
    dynamic_opened = pd.DatetimeIndex(pd.to_datetime(dynamic["Opened"], utc=True, errors="raise"))
    if opened.has_duplicates or not opened.is_monotonic_increasing:
        raise RuntimeError("Generated model feature timestamps are not unique and chronological")
    if not opened.equals(dynamic_opened):
        raise RuntimeError("Fold dynamic state features do not align to generated Opened timestamps")
    target = pd.to_numeric(frame[pipeline.TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    label_available = pd.to_datetime(frame[pipeline.TARGET_AVAILABLE_COL], utc=True, errors="coerce")
    decision = frame[pipeline.DECISION_COL].to_numpy(dtype=bool)
    decision_at = pd.to_datetime(frame["Opened"], utc=True) + pd.Timedelta(minutes=1)
    market_start = pd.to_datetime(frame["Opened"], utc=True) + pd.Timedelta(minutes=2)
    base_order = list(selection["selected_base_features"]) + list(FLOW_NAMES)
    if feature_columns != base_order or len(feature_columns) != 70:
        raise RuntimeError("Selected model feature order is not the frozen BASE64 plus taker Z/V order")
    base_x = frame.loc[:, base_order].to_numpy(dtype=np.float32, copy=True)
    base_x[~np.isfinite(base_x)] = np.nan
    memory_x = dynamic.loc[:, memory_names].to_numpy(dtype=np.float32, copy=True)
    esn_x = {
        seed: dynamic.loc[:, esn_names[seed]].to_numpy(dtype=np.float32, copy=True)
        for seed in RESERVOIR_SEEDS
    }
    return {
        "opened": opened,
        "decision_at": pd.DatetimeIndex(decision_at),
        "market_start": pd.DatetimeIndex(market_start),
        "target_available": pd.DatetimeIndex(label_available),
        "target": target,
        "decision": decision,
        "base_order": base_order,
        "base_x": base_x,
        "memory_x": memory_x,
        "esn_x": esn_x,
    }


def _family_matrix(arrays: dict, procedure: str, reservoir_seed: int | None = None):
    import numpy as np

    blocks = [arrays["base_x"]]
    if procedure in {"B", "D"}:
        blocks.append(arrays["memory_x"])
    if procedure in {"C", "D"}:
        if reservoir_seed not in arrays["esn_x"]:
            raise ValueError(f"ESN reservoir seed is missing for {procedure}: {reservoir_seed}")
        blocks.append(arrays["esn_x"][reservoir_seed])
    return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)


def _expected_members(procedure: str) -> list[tuple[int | None, int]]:
    if procedure in {"A", "B"}:
        return [(None, seed) for seed in LIGHTGBM_SEEDS]
    if procedure in {"C", "D"}:
        return [(reservoir, seed) for reservoir in RESERVOIR_SEEDS for seed in LIGHTGBM_SEEDS]
    raise ValueError(f"unknown A/B/C/D procedure: {procedure}")


def _feature_order(base_order: list[str], memory_names: list[str], esn_names: dict[int, list[str]],
                   procedure: str, reservoir_seed: int | None = None) -> list[str]:
    order = list(base_order)
    if procedure in {"B", "D"}:
        order.extend(memory_names)
    if procedure in {"C", "D"}:
        if reservoir_seed is None:
            raise ValueError(f"{procedure} feature order requires an ESN reservoir seed")
        order.extend(esn_names[reservoir_seed])
    return order


def _common_prediction_mask(arrays: dict, *, start, end):
    import numpy as np
    import pandas as pd

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    base_complete = np.isfinite(arrays["base_x"]).all(axis=1)
    memory_complete = np.isfinite(arrays["memory_x"]).all(axis=1)
    esn_complete = np.logical_and.reduce([
        np.isfinite(arrays["esn_x"][seed]).all(axis=1) for seed in RESERVOIR_SEEDS
    ])
    return (
        arrays["decision"] & base_complete & memory_complete & esn_complete
        & (arrays["market_start"] >= start) & (arrays["market_start"] < end)
    )


def _prediction_eligibility_for_markets(arrays: dict, market_starts):
    """Explain common A/B/C/D prediction coverage for each frozen source market."""
    import numpy as np
    import pandas as pd

    wanted = pd.DatetimeIndex(pd.to_datetime(market_starts, utc=True, errors="raise"))
    source = arrays["market_start"].asi8
    indexer = np.searchsorted(source, wanted.asi8)
    present = indexer < len(source)
    present[present] &= source[indexer[present]] == wanted.asi8[present]
    decision = np.zeros(len(wanted), dtype=bool)
    base_complete = np.zeros(len(wanted), dtype=bool)
    flow_complete = np.zeros(len(wanted), dtype=bool)
    memory_complete = np.zeros(len(wanted), dtype=bool)
    esn_complete = np.zeros(len(wanted), dtype=bool)
    positions = np.flatnonzero(present)
    if len(positions):
        rows = indexer[positions]
        base = arrays["base_x"][rows]
        decision[positions] = arrays["decision"][rows]
        base_complete[positions] = np.isfinite(base[:, :64]).all(axis=1)
        flow_complete[positions] = np.isfinite(base[:, 64:]).all(axis=1)
        memory_complete[positions] = np.isfinite(arrays["memory_x"][rows]).all(axis=1)
        esn_complete[positions] = np.logical_and.reduce([
            np.isfinite(arrays["esn_x"][seed][rows]).all(axis=1)
            for seed in RESERVOIR_SEEDS
        ])
    eligible = present & decision & base_complete & flow_complete & memory_complete & esn_complete
    reasons = np.full(len(wanted), "ready_for_common_candidate_prediction", dtype=object)
    reasons[~esn_complete & present & decision & base_complete & flow_complete & memory_complete] = "incomplete_esn_state"
    reasons[~memory_complete & present & decision & base_complete & flow_complete] = "incomplete_memory_state"
    reasons[~flow_complete & present & decision & base_complete] = "incomplete_causal_flow_features"
    reasons[~base_complete & present & decision] = "incomplete_selected_base_features"
    reasons[present & ~decision] = "no_decision_row"
    reasons[~present] = "missing_fold_feature_row"
    return pd.DataFrame({
        "market_start_utc": wanted,
        "raw_feature_row_present": present,
        "decision_row_available": decision,
        "selected_base_features_complete": base_complete,
        "causal_flow_features_complete": flow_complete,
        "memory_state_complete": memory_complete,
        "esn_state_complete": esn_complete,
        "common_prediction_eligible": eligible,
        "prediction_eligibility_reason": reasons,
    })


def _member_parameters(best_params: dict, *, seed: int, backend: str, threads: int) -> dict:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "device_type": backend,
        "num_threads": int(threads),
        "max_bin": 63,
        "feature_pre_filter": False,
        "deterministic": True,
        "force_col_wise": backend == "cpu",
        "seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "data_random_seed": int(seed),
        **dict(best_params),
    }


def _fit_ensemble(
    arrays: dict,
    *,
    procedure: str,
    start_cutoff,
    prediction_start,
    prediction_end,
    decision_weight: float,
    best_params: dict,
    best_iteration: int,
    base_order: list[str],
    memory_names: list[str],
    esn_names: dict[int, list[str]],
    model_root: Path,
    backend: str,
    threads: int,
    deadline_monotonic: float,
    prediction_provenance: str,
) -> tuple[dict, dict]:
    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    started = time.perf_counter()
    fit_cutoff = pd.Timestamp(start_cutoff).tz_convert("UTC")
    fit_mask = strict_training_mask(
        arrays["opened"], arrays["target"], arrays["target_available"], fit_cutoff,
    )
    train_rows = np.flatnonzero(fit_mask)
    if not len(train_rows) or np.unique(arrays["target"][train_rows]).size != 2:
        raise RuntimeError(f"{procedure} has insufficient binary rows before {fit_cutoff}")
    weights = np.where(
        arrays["decision"], float(decision_weight), (1.0 - float(decision_weight)) / 4.0,
    ).astype(np.float32)
    eligible = _common_prediction_mask(
        arrays, start=pd.Timestamp(prediction_start), end=pd.Timestamp(prediction_end),
    )
    prediction_rows = np.flatnonzero(eligible)
    if not len(prediction_rows):
        raise RuntimeError(f"{procedure} has no common complete predictions in its frozen interval")
    latest_label = arrays["target_available"][fit_mask].max()
    if not latest_label < fit_cutoff:
        raise RuntimeError("Training-label availability is not strictly earlier than the fit cutoff")
    model_root.mkdir(parents=True, exist_ok=True)
    boosters = {}
    member_records = []
    family_reservoirs = (None,) if procedure in {"A", "B"} else RESERVOIR_SEEDS
    for reservoir_seed in family_reservoirs:
        x_all = _family_matrix(arrays, procedure, reservoir_seed)
        feature_order = _feature_order(
            base_order, memory_names, esn_names, procedure, reservoir_seed,
        )
        if x_all.shape[1] != len(feature_order) or len(set(feature_order)) != len(feature_order):
            raise RuntimeError(f"{procedure} matrix width/order is invalid")
        x_train = np.array(x_all[train_rows], dtype=np.float32, copy=True)
        x_train[~np.isfinite(x_train)] = np.nan
        y_train = arrays["target"][train_rows].astype(np.int8, copy=False)
        w_train = weights[train_rows]
        x_predict = np.array(x_all[prediction_rows], dtype=np.float32, copy=True)
        x_predict[~np.isfinite(x_predict)] = np.nan
        for lightgbm_seed in LIGHTGBM_SEEDS:
            if time.monotonic() >= deadline_monotonic:
                raise TimeoutError(f"{procedure} member fitting exceeded the frozen stage budget")
            member_key = (reservoir_seed, lightgbm_seed)
            member_name = (
                f"{procedure}_r{reservoir_seed}_l{lightgbm_seed}"
                if reservoir_seed is not None else f"{procedure}_l{lightgbm_seed}"
            )
            model_path = model_root / f"{member_name}.txt"
            metadata_path = model_root / f"{member_name}.json"
            identity = {
                "procedure": procedure,
                "reservoir_seed": reservoir_seed,
                "lightgbm_seed": lightgbm_seed,
                "fit_cutoff_exclusive_utc": fit_cutoff.isoformat(),
                "training_rows": int(len(train_rows)),
                "training_row_opened_sha256": _canonical_hash(arrays["opened"].asi8[train_rows].tolist()),
                "training_labels_sha256": _canonical_hash(y_train.tolist()),
                "training_weights_sha256": _canonical_hash(w_train.tolist()),
                "training_feature_matrix_sha256": hashlib.sha256(
                    memoryview(np.ascontiguousarray(x_train)).cast("B")
                ).hexdigest(),
                "feature_order": feature_order,
                "feature_order_sha256": _canonical_hash(feature_order),
                "best_params": best_params,
                "best_iteration": int(best_iteration),
                "compute_backend": backend,
                "threads": int(threads),
                "lightgbm_version": str(lgb.__version__),
                "numpy_version": str(np.__version__),
            }
            identity_hash = _canonical_hash(identity)
            if model_path.is_file() and metadata_path.is_file():
                saved = json.loads(metadata_path.read_text(encoding="utf-8"))
                if saved.get("identity_sha256") != identity_hash or saved.get("model_sha256") != _sha256(model_path):
                    raise RuntimeError(f"Saved ensemble member identity differs: {member_name}")
                booster = lgb.Booster(model_file=str(model_path))
            else:
                params = _member_parameters(
                    best_params, seed=lightgbm_seed, backend=backend, threads=threads,
                )
                train_set = lgb.Dataset(
                    x_train, label=y_train, weight=w_train,
                    feature_name=feature_order, params=params, free_raw_data=True,
                )
                booster = lgb.train(params, train_set, num_boost_round=int(best_iteration))
                del train_set
                booster.save_model(str(model_path))
                _atomic_json(metadata_path, {
                    "identity_sha256": identity_hash,
                    "identity": identity,
                    "model_sha256": _sha256(model_path),
                    "num_trees": int(booster.num_trees()),
                    "num_features": int(booster.num_feature()),
                })
            if int(booster.num_feature()) != len(feature_order):
                raise RuntimeError(f"Loaded {member_name} has a different feature width")
            boosters[member_key] = booster
            member_records.append({
                "member": member_name,
                "model_path": model_path.resolve().relative_to(ROOT).as_posix(),
                "model_sha256": _sha256(model_path),
                "feature_order_sha256": _canonical_hash(feature_order),
                "num_trees": int(booster.num_trees()),
            })
            if booster.num_trees() > int(best_iteration):
                raise RuntimeError(f"{member_name} exceeded its tuned boosting-round count")
            if reservoir_seed is None:
                boosters[member_key] = (booster, np.asarray(booster.predict(x_predict), dtype=np.float64))
            else:
                boosters[member_key] = (booster, np.asarray(booster.predict(x_predict), dtype=np.float64))
        del x_train, x_predict, x_all

    member_predictions = {name: {} for name in CANDIDATES}
    for (reservoir_seed, lightgbm_seed), (booster, values) in boosters.items():
        key = str(lightgbm_seed) if reservoir_seed is None else f"{reservoir_seed}:{lightgbm_seed}"
        member_predictions[procedure][key] = values
    if set(member_predictions[procedure]) != {
        str(seed) if reservoir is None else f"{reservoir}:{seed}"
        for reservoir, seed in _expected_members(procedure)
    }:
        raise RuntimeError(f"{procedure} ensemble is incomplete")
    raw = np.mean(list(member_predictions[procedure].values()), axis=0)
    if not np.isfinite(raw).all() or ((raw < 0.0) | (raw > 1.0)).any():
        raise RuntimeError(f"{procedure} produced invalid ensemble probabilities")
    fit_identity = _canonical_hash({
        "procedure": procedure,
        "fit_cutoff": fit_cutoff.isoformat(),
        "best_params": best_params,
        "best_iteration": int(best_iteration),
        "members": member_records,
    })
    prediction_rows_frame = pd.DataFrame({
        "Opened": arrays["opened"][prediction_rows],
        "decision_at_utc": arrays["decision_at"][prediction_rows],
        "market_start_utc": arrays["market_start"][prediction_rows],
        "ensemble_probability_raw": raw,
        "prediction_provenance": prediction_provenance,
        "fit_cutoff_exclusive_utc": fit_cutoff,
        "latest_training_label_available_at_utc": latest_label,
        "model_identity_sha256": fit_identity,
        "feature_version_sha256": _canonical_hash({
            "base": base_order,
            "memory": memory_names,
            "esn": esn_names,
            "procedure": procedure,
        }),
    })
    model_manifest = {
        "procedure": procedure,
        "fit_cutoff_exclusive_utc": fit_cutoff.isoformat(),
        "latest_training_label_available_at_utc": latest_label.isoformat(),
        "training_rows": int(len(train_rows)),
        "prediction_rows": int(len(prediction_rows)),
        "member_count": len(member_records),
        "ensemble_weights": "uniform",
        "ensemble_identity_sha256": fit_identity,
        "members": member_records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return prediction_rows_frame, model_manifest


def fit_and_predict_ensemble(
    arrays: dict,
    *,
    procedure: str,
    fit_cutoff,
    prediction_start,
    prediction_end,
    selection: dict,
    memory_names: list[str],
    esn_names: dict[int, list[str]],
    model_root: Path,
    backend: str,
    threads: int,
    deadline_monotonic: float,
    prediction_provenance: str,
) -> tuple:
    return _fit_ensemble(
        arrays,
        procedure=procedure,
        start_cutoff=fit_cutoff,
        prediction_start=prediction_start,
        prediction_end=prediction_end,
        decision_weight=float(selection["decision_weight"]),
        best_params=dict(selection["best_params"]),
        best_iteration=int(selection["best_iteration"]),
        base_order=list(selection["model_feature_order"]),
        memory_names=memory_names,
        esn_names=esn_names,
        model_root=model_root / procedure,
        backend=backend,
        threads=threads,
        deadline_monotonic=deadline_monotonic,
        prediction_provenance=prediction_provenance,
    )


def _read_arrow_development_rows(path: Path, columns: list[str] | tuple[str, ...], *, start, end,
                                condition_ids: list[str] | None = None):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    market_type = dataset.schema.field("market_start_utc").type
    start_ts = pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end).tz_convert("UTC")
    start_value = pa.scalar(start_ts.to_pydatetime(), type=market_type)
    end_value = pa.scalar(end_ts.to_pydatetime(), type=market_type)
    filter_expression = (
        (ds.field("market_start_utc") >= start_value)
        & (ds.field("market_start_utc") < end_value)
    )
    if condition_ids is not None:
        filter_expression &= ds.field("condition_id").isin(condition_ids)
    frame = dataset.to_table(columns=list(columns), filter=filter_expression).to_pandas()
    if "condition_id" in frame:
        frame["condition_id"] = frame.condition_id.astype(str).str.lower()
    frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True, errors="raise")
    if frame.market_start_utc.duplicated().any():
        raise RuntimeError(f"{Path(path).name} has duplicate development market starts")
    return frame


def _development_rows_sha256(path: Path, columns: list[str], *, start, end,
                             condition_ids: list[str] | None = None) -> str:
    frame = _read_arrow_development_rows(
        path, columns, start=start, end=end, condition_ids=condition_ids,
    )
    rows = frame.sort_values(
        [column for column in ("market_start_utc", "condition_id") if column in frame],
        kind="stable",
    ).to_dict(orient="records")
    return _canonical_hash(rows)


def verify_filtered_source_hashes(protocol: dict, whitelist, clean_ids_hash: str) -> dict:
    sources = protocol["economics_sources"]
    outer_start = protocol["outer_folds"][0]["market_start_test_start_inclusive_utc"]
    outer_end = protocol["outer_folds"][-1]["market_start_test_end_exclusive_utc"]
    all_ids = whitelist.condition_id.tolist()
    expectations = {
        "official_gamma": _development_rows_sha256(
            ROOT / protocol["official_gamma_source"]["path"], list(GAMMA_COLUMNS),
            start=protocol["official_gamma_source"]["read_start_utc"], end="2026-07-01T00:00:00Z",
        ),
        "economics_gamma": _development_rows_sha256(
            ROOT / sources["gamma_path"], [
                "condition_id", "market_start_utc", "target_polymarket_up", "resolved_at_utc",
                "up_token_id", "down_token_id",
            ], start=outer_start, end=outer_end, condition_ids=all_ids,
        ),
        "t60_books": _development_rows_sha256(
            ROOT / sources["t60_path"], [
                "condition_id", "market_start_utc", "entry_case", "entry_time_utc",
                "fee_collection_mode", "fee_rate_bps", "fee_known", "has_full_snapshot",
                "quote_valid", "no_future_event_at_entry", "no_future_source_event_at_entry",
                "up_best_ask", "up_best_ask_size_shares", "down_best_ask", "down_best_ask_size_shares",
            ], start=outer_start, end=outer_end, condition_ids=all_ids,
        ),
        "t59_asks": _development_rows_sha256(
            ROOT / sources["t59_path"], [
                "condition_id", "market_start_utc", "entry_case", "entry_time_utc",
                "fee_collection_mode", "fee_rate_bps", "fee_known", "has_full_snapshot",
                "quote_valid", "no_future_event_at_entry", "no_future_source_event_at_entry",
                "up_ask_levels", "down_ask_levels",
            ], start=outer_start, end=outer_end, condition_ids=all_ids,
        ),
    }
    if clean_ids_hash != sources["clean_ids_sha256"]:
        raise RuntimeError("Clean source-quality whitelist ID hash differs from the frozen protocol")
    expected_rows = protocol.get("filtered_source_row_hashes", {})
    if expected_rows:
        if set(expected_rows) != set(expectations):
            raise RuntimeError("Frozen filtered-source hash map has an incomplete key set")
        for key, actual in expectations.items():
            if actual != expected_rows[key]:
                raise RuntimeError(f"Filtered development source rows changed: {key}")
    return expectations


def load_official_gamma(protocol: dict):
    import numpy as np
    import pandas as pd

    source = protocol["official_gamma_source"]
    frame = _read_arrow_development_rows(
        ROOT / source["path"], GAMMA_COLUMNS,
        start=source["read_start_utc"], end="2026-07-01T00:00:00Z",
    )
    frame["condition_id"] = frame.condition_id.astype(str).str.lower()
    frame["resolved_at_utc"] = pd.to_datetime(frame.resolved_at_utc, utc=True, errors="coerce")
    outcomes = pd.to_numeric(frame.target_polymarket_up, errors="coerce")
    valid = outcomes.isin([0, 1]) & frame.resolved_at_utc.notna()
    frame = frame.loc[valid].copy()
    frame["target_polymarket_up"] = outcomes.loc[valid].astype(np.int8)
    frame["official_label_release_at_utc"] = pd.concat(
        [frame.resolved_at_utc, frame.market_start_utc + pd.Timedelta(minutes=5)], axis=1,
    ).max(axis=1) + pd.Timedelta(seconds=60)
    frame.sort_values("market_start_utc", kind="stable", inplace=True)
    return frame.reset_index(drop=True)


def _join_gamma_outcomes(predictions, gamma, *, source: str):
    import pandas as pd

    if predictions.market_start_utc.duplicated().any():
        raise RuntimeError(f"{source} predictions contain duplicate market starts")
    joined = predictions.merge(
        gamma[["market_start_utc", "condition_id", "target_polymarket_up",
               "resolved_at_utc", "official_label_release_at_utc"]],
        on="market_start_utc", how="left", validate="one_to_one",
    )
    joined["calibration_outcome_available"] = joined.target_polymarket_up.notna()
    joined["official_label_release_at_utc"] = pd.to_datetime(
        joined.official_label_release_at_utc, utc=True, errors="coerce",
    )
    joined["calibration_origin"] = source
    joined["calibration_label_source"] = "official_gamma"
    return joined


def fit_official_platt(calibration_rows, *, as_of_utc, minimum_rows: int = 1000):
    """Fit the frozen official Gamma Platt rule or return an explicit unavailable reason."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    as_of = pd.Timestamp(as_of_utc).tz_convert("UTC")
    available = calibration_rows.loc[
        calibration_rows.calibration_outcome_available
        & pd.to_datetime(calibration_rows.official_label_release_at_utc, utc=True).le(as_of)
        & pd.to_numeric(calibration_rows.ensemble_probability_raw, errors="coerce").notna()
    ].copy()
    if "market_start_utc" in available:
        available = available.loc[
            pd.to_datetime(available.market_start_utc, utc=True).lt(as_of)
        ]
    if available.market_start_utc.duplicated().any():
        raise RuntimeError("Official Platt history contains duplicate OOF market starts")
    accepted_provenance = {
        "fold_local_preselection_chronological_oof",
        "fold_external_out_of_sample",
    }
    if not available.prediction_provenance.isin(accepted_provenance).all():
        raise RuntimeError("Official Platt history contains in-sample or unapproved predictions")
    if not available.calibration_label_source.eq("official_gamma").all():
        raise RuntimeError("Official Platt history contains labels outside official Gamma")
    fit_cutoffs = pd.to_datetime(available.fit_cutoff_exclusive_utc, utc=True, errors="raise")
    latest_labels = pd.to_datetime(
        available.latest_training_label_available_at_utc, utc=True, errors="raise",
    )
    market_starts = pd.to_datetime(available.market_start_utc, utc=True, errors="raise")
    if (
        available.model_identity_sha256.isna().any()
        or not fit_cutoffs.le(market_starts).all()
        or not latest_labels.lt(fit_cutoffs).all()
    ):
        raise RuntimeError("Official Platt history has invalid model or training-label chronology")
    y = pd.to_numeric(available.target_polymarket_up, errors="coerce").to_numpy(dtype=np.float64)
    if len(available) < int(minimum_rows):
        return None, {"status": "minimum_not_met", "rows": int(len(available)), "required": int(minimum_rows)}
    if set(np.unique(y)) != {0.0, 1.0}:
        return None, {"status": "both_classes_not_present", "rows": int(len(available)), "required": int(minimum_rows)}
    p = np.clip(
        pd.to_numeric(available.ensemble_probability_raw, errors="raise").to_numpy(dtype=np.float64),
        1e-6, 1.0 - 1e-6,
    )
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", random_state=20261015)
    model.fit(logit, y.astype(np.int8))
    training_rows = available.sort_values("market_start_utc", kind="stable")
    record = {
        "status": "fitted",
        "method": "daily expanding Platt logistic regression on raw ensemble logit",
        "label_source": "official Gamma target_polymarket_up",
        "training_rows": int(len(available)),
        "training_first_market_start_utc": pd.Timestamp(available.market_start_utc.min()).isoformat(),
        "training_last_market_start_utc": pd.Timestamp(available.market_start_utc.max()).isoformat(),
        "latest_label_release_at_utc": pd.Timestamp(available.official_label_release_at_utc.max()).isoformat(),
        "as_of_utc": as_of.isoformat(),
        "training_rows_sha256": _canonical_hash(training_rows[[
            "market_start_utc", "ensemble_probability_raw", "target_polymarket_up",
            "official_label_release_at_utc", "model_identity_sha256",
        ]].to_dict(orient="records")),
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }
    record["calibrator_identity_sha256"] = _canonical_hash(record)
    return record, record


def apply_official_platt(probabilities, calibrator):
    import numpy as np
    from scipy.special import expit

    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p))
    return expit(float(calibrator["intercept"]) + float(calibrator["coefficient"]) * logits)


def calibrate_external_daily(predictions, history, gamma, *, minimum_rows: int):
    import numpy as np
    import pandas as pd

    predictions = _join_gamma_outcomes(predictions, gamma, source="fold_external_out_of_sample")
    history = history.copy()
    result = predictions.copy()
    result["ensemble_probability_calibrated"] = np.nan
    result["calibrator_identity_sha256"] = None
    result["calibration_status"] = "calibration_unavailable"
    daily_records = []
    day_values = pd.to_datetime(result.decision_at_utc, utc=True).dt.floor("D")
    for day in day_values.drop_duplicates().sort_values():
        day_mask = day_values.eq(day)
        day_first_decision = pd.to_datetime(result.loc[day_mask, "decision_at_utc"], utc=True).min()
        training = history.loc[
            pd.to_datetime(history.market_start_utc, utc=True).lt(day_first_decision)
            & pd.to_datetime(history.official_label_release_at_utc, utc=True, errors="coerce").le(day_first_decision)
            & history.calibration_outcome_available.fillna(False)
        ]
        if training.market_start_utc.duplicated().any():
            raise RuntimeError("Daily Platt history has duplicate OOF market starts")
        calibrator, status = fit_official_platt(
            training, as_of_utc=day_first_decision, minimum_rows=minimum_rows,
        )
        daily_records.append({
            "utc_day": pd.Timestamp(day).isoformat(),
            "first_decision_at_utc": pd.Timestamp(day_first_decision).isoformat(),
            **status,
        })
        if calibrator is None:
            result.loc[day_mask, "calibration_status"] = status["status"]
            continue
        selected = result.loc[day_mask, "ensemble_probability_raw"].to_numpy(dtype=np.float64)
        result.loc[day_mask, "ensemble_probability_calibrated"] = apply_official_platt(selected, calibrator)
        result.loc[day_mask, "calibrator_identity_sha256"] = calibrator["calibrator_identity_sha256"]
        result.loc[day_mask, "calibration_status"] = "fitted"
    return result, daily_records


def _load_clean_whitelist(protocol: dict):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds

    path = ROOT / protocol["economics_sources"]["quality_manifest_path"]
    dataset = ds.dataset(path, format="parquet")
    market_type = dataset.schema.field("market_start_utc").type
    start = pa.scalar(pd.Timestamp("2025-07-01T00:00:00Z").to_pydatetime(), type=market_type)
    end = pa.scalar(pd.Timestamp("2026-07-01T00:00:00Z").to_pydatetime(), type=market_type)
    frame = dataset.to_table(
        columns=["condition_id", "market_start_utc", "source_quality_clean", "up_token_id", "down_token_id"],
        filter=(ds.field("source_quality_clean") == True)
        & (ds.field("market_start_utc") >= start) & (ds.field("market_start_utc") < end),
    ).to_pandas()
    frame["condition_id"] = frame.condition_id.astype(str).str.lower()
    frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True, errors="raise")
    frame.sort_values(["market_start_utc", "condition_id"], kind="stable", inplace=True)
    expected = int(protocol["economics_sources"]["expected_clean_market_count"])
    if len(frame) != expected or frame.condition_id.duplicated().any():
        raise RuntimeError("E0038 clean-market whitelist count/uniqueness changed")
    if frame.market_start_utc.duplicated().any():
        raise RuntimeError("E0038 clean-market whitelist has duplicate market starts")
    ids = sorted(frame.condition_id.tolist())
    ids_hash = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    expected_hash = protocol["economics_sources"].get("clean_ids_sha256")
    if expected_hash and ids_hash != expected_hash:
        raise RuntimeError("E0038 clean ID allowlist hash changed")
    return frame.reset_index(drop=True), ids_hash


def load_economics_source_snapshots(protocol: dict, whitelist):
    import pandas as pd

    import run_btc_t0009_economic_continuation_20261026 as accounting

    first = protocol["outer_folds"][0]["market_start_test_start_inclusive_utc"]
    last = protocol["outer_folds"][-1]["market_start_test_end_exclusive_utc"]
    ids = whitelist.condition_id.tolist()
    sources = protocol["economics_sources"]
    gamma = _read_arrow_development_rows(
        ROOT / sources["gamma_path"], list(accounting.GAMMA_COLUMNS),
        start=first, end=last, condition_ids=ids,
    )
    t60 = _read_arrow_development_rows(
        ROOT / sources["t60_path"], list(accounting.T60_COLUMNS),
        start=first, end=last, condition_ids=ids,
    )
    t59 = _read_arrow_development_rows(
        ROOT / sources["t59_path"], list(accounting.T59_COLUMNS),
        start=first, end=last, condition_ids=ids,
    )
    allowed = whitelist.loc[
        whitelist.market_start_utc.ge(pd.Timestamp(first))
        & whitelist.market_start_utc.lt(pd.Timestamp(last))
    ].copy()
    for name, frame in (("Gamma", gamma), ("T-60", t60), ("T-59", t59)):
        if frame.condition_id.duplicated().any() or not set(frame.condition_id).issubset(set(allowed.condition_id)):
            raise RuntimeError(f"E0038 {name} snapshot contains duplicate or out-of-whitelist IDs")
    if (
        set(gamma.condition_id) != set(t60.condition_id)
        or set(gamma.condition_id) != set(t59.condition_id)
        or set(gamma.condition_id) != set(allowed.condition_id)
    ):
        raise RuntimeError("Gamma/T-60/T-59 snapshot coverage differs inside the frozen fold windows")
    expected_tokens = allowed.set_index("condition_id")[["up_token_id", "down_token_id"]]
    actual_tokens = gamma.set_index("condition_id")[["up_token_id", "down_token_id"]].astype(str)
    common_tokens = expected_tokens.loc[actual_tokens.index].astype(str)
    if not common_tokens.equals(actual_tokens):
        raise RuntimeError("Official Gamma tokens differ from the source-quality whitelist")
    gamma.drop(columns=["up_token_id", "down_token_id"], inplace=True)
    if gamma.target_polymarket_up.isna().any() or not gamma.target_polymarket_up.isin([0, 1, False, True]).all():
        raise RuntimeError("Official Gamma outcomes are not complete binary labels in the fold windows")
    source = allowed.merge(gamma, on=["condition_id", "market_start_utc"], validate="one_to_one")
    source = source.merge(t60, on=["condition_id", "market_start_utc"], suffixes=("", "_t60"), validate="one_to_one")
    source = source.merge(t59, on=["condition_id", "market_start_utc"], suffixes=("", "_t59"), validate="one_to_one")
    source["resolved_at_utc"] = pd.to_datetime(source.resolved_at_utc, utc=True, errors="raise")
    if not (source.entry_time_utc == source.market_start_utc - pd.Timedelta(seconds=60)).all():
        raise RuntimeError("E0038 T-60 book decision timestamp differs from the frozen rule")
    if not (source.entry_time_utc_t59 == source.market_start_utc - pd.Timedelta(seconds=59)).all():
        raise RuntimeError("E0038 T-59 ladder execution timestamp differs from the frozen rule")
    return source.sort_values(["market_start_utc", "condition_id"], kind="stable").reset_index(drop=True)


def _continuous_account_path(rows: list[dict], model: str, control: str, delay_seconds: int):
    import pandas as pd

    import run_btc_t0009_economic_continuation_20261026 as accounting
    from utils.btc_resolution_portfolio import ResolutionPortfolio

    if control not in {"fixed_5_usd", "free_cash_5pct"}:
        raise ValueError(control)
    account = ResolutionPortfolio(accounting.INITIAL_CASH_USD)
    decisions = []
    monthly = []
    boundaries = [
        (month, boundary) for month, boundary in accounting._month_boundaries()
        if "2026-04" <= month <= "2026-06"
    ]
    boundary_index = 0
    for row in rows:
        decision_time = pd.Timestamp(row["market_start_utc"]) - pd.Timedelta(seconds=60)
        while boundary_index < len(boundaries) and boundaries[boundary_index][1] < decision_time:
            month, boundary = boundaries[boundary_index]
            account.release_payouts_available_by(boundary)
            monthly.append(accounting._monthly_account_row(account, decisions, month, boundary))
            boundary_index += 1
        account.release_payouts_available_by(decision_time)
        if control == "fixed_5_usd":
            gross = accounting.FIXED_GROSS_USD
        else:
            gross = accounting.FREE_CASH_FRACTION * account.cash_usd
        decision = accounting._process_market(row, model, gross, delay_seconds, account=account)
        decisions.append(decision)
    while boundary_index < len(boundaries):
        month, boundary = boundaries[boundary_index]
        account.release_payouts_available_by(boundary)
        monthly.append(accounting._monthly_account_row(account, decisions, month, boundary))
        boundary_index += 1
    terminal_at = max(
        (pd.Timestamp(row["payout_available_at_utc"]) for row in decisions),
        default=pd.Timestamp("2026-07-01T00:00:00Z"),
    )
    account.release_payouts_available_by(terminal_at)
    terminal = account.summary()
    summary = {
        "eligible_calibrated_markets": len(decisions),
        "signals": sum(bool(row["signal"]) for row in decisions),
        "fills": sum(row["fill_status"] == "full" for row in decisions),
        "rejections": accounting._rejection_counts(decisions),
        "accepted_sides": {
            side: sum(row["fill_status"] == "full" and row["selected_side"] == side for row in decisions)
            for side in ("up", "down")
        },
        "settlement_delay_seconds": int(delay_seconds),
        "terminal_settlement_at_utc": pd.Timestamp(terminal_at).isoformat(),
        "terminal_account_after_all_payouts": terminal,
    }
    for item in monthly:
        month = item["month_utc"]
        item["monthly_realized_pnl_usd"] = sum(
            row["realized_pnl_usd"] for row in decisions
            if pd.Timestamp(row["decision_time_utc"]).strftime("%Y-%m") == month
        )
        item.update({"model": model, "control": control, "payout_delay_seconds": int(delay_seconds)})
    for item in decisions:
        item.update({"control": control, "payout_delay_seconds": int(delay_seconds)})
    return summary, decisions, monthly


def run_economics(
        protocol: dict, whitelist, snapshots, prediction_panel, eligibility_panel,
        output_dir: Path,
) -> dict:
    import numpy as np
    import pandas as pd

    import run_btc_t0009_economic_continuation_20261026 as accounting

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_start = pd.Timestamp(protocol["outer_folds"][0]["market_start_test_start_inclusive_utc"])
    fold_end = pd.Timestamp(protocol["outer_folds"][-1]["market_start_test_end_exclusive_utc"])
    coverage = whitelist[["condition_id", "market_start_utc"]].copy()
    coverage["inside_outer_fold_window"] = coverage.market_start_utc.ge(fold_start) & coverage.market_start_utc.lt(fold_end)
    if eligibility_panel.market_start_utc.duplicated().any():
        raise RuntimeError("Outer eligibility coverage contains duplicate source-market starts")
    coverage = coverage.merge(
        eligibility_panel, on="market_start_utc", how="left", validate="one_to_one",
    )
    for model in CANDIDATES:
        model_prediction = prediction_panel.loc[
            prediction_panel.model.eq(model),
            ["market_start_utc", "ensemble_probability_raw", "ensemble_probability_calibrated", "calibration_status", "model_identity_sha256", "calibrator_identity_sha256"],
        ]
        if model_prediction.market_start_utc.duplicated().any():
            raise RuntimeError(f"Duplicate outer predictions for {model}")
        coverage = coverage.merge(
            model_prediction.rename(columns={
                "ensemble_probability_raw": f"raw_probability_{model}",
                "ensemble_probability_calibrated": f"calibrated_probability_{model}",
                "calibration_status": f"calibration_status_{model}",
                "model_identity_sha256": f"model_identity_{model}",
                "calibrator_identity_sha256": f"calibrator_identity_{model}",
            }),
            on="market_start_utc", how="left", validate="one_to_one",
        )
        missing_in_fold = coverage.inside_outer_fold_window & coverage[f"raw_probability_{model}"].isna()
        coverage.loc[~coverage.inside_outer_fold_window, f"prediction_status_{model}"] = "outside_frozen_outer_fold_windows"
        eligibility_reason = coverage.prediction_eligibility_reason.fillna("missing_eligibility_record")
        coverage.loc[missing_in_fold, f"prediction_status_{model}"] = eligibility_reason.loc[missing_in_fold]
        unexpectedly_missing = missing_in_fold & coverage.common_prediction_eligible.fillna(False)
        coverage.loc[unexpectedly_missing, f"prediction_status_{model}"] = "eligible_but_raw_prediction_missing"
        has_raw = coverage[f"raw_probability_{model}"].notna()
        no_calibration = has_raw & coverage[f"calibrated_probability_{model}"].isna()
        coverage.loc[no_calibration, f"prediction_status_{model}"] = coverage.loc[
            no_calibration, f"calibration_status_{model}"
        ].fillna("calibration_unavailable")
        calibrated = coverage[f"calibrated_probability_{model}"].notna()
        coverage.loc[calibrated, f"prediction_status_{model}"] = "predicted_calibrated"

    coverage_path = output_dir / "prediction_coverage_by_clean_market.parquet"
    _atomic_parquet(coverage, coverage_path)
    coverage_summary = {}
    for model in CANDIDATES:
        counts = coverage[f"prediction_status_{model}"].value_counts(dropna=False).to_dict()
        coverage_summary[model] = {
            "source_clean_whitelist_markets": int(len(coverage)),
            "markets_in_outer_folds": int(coverage.inside_outer_fold_window.sum()),
            "raw_predictions": int(coverage[f"raw_probability_{model}"].notna().sum()),
            "officially_calibrated_predictions": int(coverage[f"calibrated_probability_{model}"].notna().sum()),
            "status_counts": {str(key): int(value) for key, value in counts.items()},
        }

    probabilities = prediction_panel.pivot(
        index="market_start_utc", columns="model", values="ensemble_probability_calibrated",
    ).rename(columns={model: f"ensemble_probability_{model}" for model in CANDIDATES}).reset_index()
    if probabilities.market_start_utc.duplicated().any():
        raise RuntimeError("Calibrated prediction panel is not one-to-one by market start")
    cohort = snapshots.merge(probabilities, on="market_start_utc", how="left", validate="one_to_one")
    fully_calibrated = np.logical_and.reduce([
        pd.to_numeric(cohort[f"ensemble_probability_{model}"], errors="coerce").notna().to_numpy()
        for model in CANDIDATES
    ])
    common = cohort.loc[fully_calibrated].sort_values(
        ["market_start_utc", "condition_id"], kind="stable",
    ).reset_index(drop=True)
    rows_by_model = {
        model: [
            accounting._row_for_model(row, model)
            for _, row in cohort.loc[
                pd.to_numeric(cohort[f"ensemble_probability_{model}"], errors="coerce").notna()
            ].sort_values(["market_start_utc", "condition_id"], kind="stable").iterrows()
        ]
        for model in CANDIDATES
    }

    paired_market_rows = {}
    common_market_records = common.sort_values(
        ["market_start_utc", "condition_id"], kind="stable",
    )
    for model in CANDIDATES:
        common_rows = [
            accounting._row_for_model(row, model)
            for _, row in common_market_records.iterrows()
        ]
        _, trades, _ = accounting._independent_diagnostic(
            common_rows, model, int(accounting.PAYOUT_DELAYS[0]),
        )
        paired_market_rows[model] = (
            pd.DataFrame(trades).set_index("market_start_utc")
            if trades else pd.DataFrame(columns=["realized_pnl_usd"])
        )

    paired_market_differences = []
    paired_monthly_differences = []
    for left_index, left in enumerate(CANDIDATES):
        for right in CANDIDATES[left_index + 1:]:
            left_trades = paired_market_rows[left]
            right_trades = paired_market_rows[right]
            market_starts = left_trades.index.to_numpy()
            paired = pd.DataFrame({
                "market_start_utc": market_starts,
                "left_model": np.repeat(left, len(market_starts)),
                "right_model": np.repeat(right, len(market_starts)),
                "left_realized_pnl_usd": left_trades.realized_pnl_usd.to_numpy(dtype=float),
                "right_realized_pnl_usd": right_trades.realized_pnl_usd.to_numpy(dtype=float),
            })
            if not np.array_equal(paired.market_start_utc.to_numpy(), right_trades.index.to_numpy()):
                raise RuntimeError("Paired economic diagnostics lost common-market row alignment")
            paired["left_minus_right_pnl_usd"] = (
                paired.left_realized_pnl_usd - paired.right_realized_pnl_usd
            )
            paired_market_differences.extend(paired.to_dict(orient="records"))
            paired["month_utc"] = pd.to_datetime(paired.market_start_utc, utc=True).dt.strftime("%Y-%m")
            for month, group in paired.groupby("month_utc", sort=True):
                difference = group.left_minus_right_pnl_usd.to_numpy(dtype=float)
                paired_monthly_differences.append({
                    "left_model": left,
                    "right_model": right,
                    "month_utc": month,
                    "common_markets": int(len(group)),
                    "left_realized_pnl_usd": float(group.left_realized_pnl_usd.sum()),
                    "right_realized_pnl_usd": float(group.right_realized_pnl_usd.sum()),
                    "left_minus_right_pnl_usd": float(difference.sum()),
                    "markets_positive_zero_negative": {
                        "positive": int((difference > 0.0).sum()),
                        "zero": int((difference == 0.0).sum()),
                        "negative": int((difference < 0.0).sum()),
                    },
                })

    paired_summary = {}
    paired_frame = pd.DataFrame(paired_market_differences, columns=[
        "market_start_utc", "left_model", "right_model",
        "left_realized_pnl_usd", "right_realized_pnl_usd", "left_minus_right_pnl_usd",
    ])
    for (left, right), group in paired_frame.groupby(["left_model", "right_model"], sort=True):
        difference = group.left_minus_right_pnl_usd.to_numpy(dtype=float)
        paired_summary[f"{left}-{right}"] = {
            "common_markets": int(len(group)),
            "left_realized_pnl_usd": float(group.left_realized_pnl_usd.sum()),
            "right_realized_pnl_usd": float(group.right_realized_pnl_usd.sum()),
            "left_minus_right_pnl_usd": float(difference.sum()),
            "markets_positive_zero_negative": {
                "positive": int((difference > 0.0).sum()),
                "zero": int((difference == 0.0).sum()),
                "negative": int((difference < 0.0).sum()),
            },
        }

    controls = {}
    decision_records = []
    monthly_records = []
    for model in CANDIDATES:
        controls[model] = {}
        model_rows = rows_by_model[model]
        for delay in accounting.PAYOUT_DELAYS:
            independent_summary, independent_trades, independent_monthly = accounting._independent_diagnostic(
                model_rows, model, int(delay),
            )
            fixed_summary, fixed_trades, fixed_monthly = _continuous_account_path(
                model_rows, model, "fixed_5_usd", int(delay),
            )
            variable_summary, variable_trades, variable_monthly = _continuous_account_path(
                model_rows, model, "free_cash_5pct", int(delay),
            )
            controls[model][str(delay)] = {
                "independent_fixed5": independent_summary,
                "account_fixed5": fixed_summary,
                "account_free_cash_5pct": variable_summary,
            }
            for control, trades in (
                ("independent_fixed5", independent_trades),
                ("fixed_5_usd", fixed_trades),
                ("free_cash_5pct", variable_trades),
            ):
                decision_records.extend(
                    {**trade, "model": model, "control": control, "payout_delay_seconds": int(delay)}
                    for trade in trades
                )
            if int(delay) == int(accounting.PAYOUT_DELAYS[0]):
                monthly_records.extend(
                    {**item, "model": model, "control": "independent_fixed5", "payout_delay_seconds": int(delay)}
                    for item in independent_monthly
                )
            monthly_records.extend(fixed_monthly)
            monthly_records.extend(variable_monthly)

    decisions_path = output_dir / "market_decisions.csv"
    monthly_path = output_dir / "monthly_paths.csv"
    paired_market_path = output_dir / "paired_market_differences.csv"
    paired_monthly_path = output_dir / "paired_monthly_differences.csv"
    pd.DataFrame(decision_records).to_csv(decisions_path, index=False)
    pd.DataFrame(monthly_records).to_csv(monthly_path, index=False)
    paired_frame.to_csv(paired_market_path, index=False)
    pd.DataFrame(paired_monthly_differences).to_csv(paired_monthly_path, index=False)
    return {
        "status": "completed_conditional_on_common_official_calibrated_predictions",
        "source_clean_whitelist_count": int(len(whitelist)),
        "outer_source_snapshot_count": int(len(snapshots)),
        "common_calibrated_prediction_market_count": int(len(common)),
        "coverage_by_model": coverage_summary,
        "accounting": {
            "strategy": "strictly positive net EV at T-60; buy with native T-59 asks; hold to official Gamma payout",
            "ev_threshold": "EV > 0.0 USD; exact zero rejected; tie selects UP",
            "entry_cases": [accounting.T60_ENTRY_CASE, accounting.T59_ENTRY_CASE],
            "payout_delays_seconds": list(accounting.PAYOUT_DELAYS),
            "controls": ["independent_fixed5", "fixed_5_usd", "free_cash_5pct"],
            "portfolio_path": "continuous one-time path across F1/F2/F3 for each model/control/delay",
            "minimum": "D0019 verified-USDC rule with frozen $5 fallback; no upward rounding",
            "terminal_settlement": "all positions released at terminal payout availability after the final fold",
            "unknown_prediction_rows_are_not_trades": True,
            "unknown_prediction_rows_are_not_labeled_no_trade": True,
        },
        "controls_results": controls,
        "paired_independent_fixed5_60s_common_coverage": paired_summary,
        "coverage_path": coverage_path.resolve().relative_to(ROOT).as_posix(),
        "market_decisions_path": decisions_path.resolve().relative_to(ROOT).as_posix(),
        "monthly_paths_path": monthly_path.resolve().relative_to(ROOT).as_posix(),
        "paired_market_differences_path": paired_market_path.resolve().relative_to(ROOT).as_posix(),
        "paired_monthly_differences_path": paired_monthly_path.resolve().relative_to(ROOT).as_posix(),
        "output_sha256": {
            path.name: _sha256(path) for path in (
                coverage_path, decisions_path, monthly_path,
                paired_market_path, paired_monthly_path,
            )
        },
        "all_source_clean_outer_markets_with_any_prediction": int(
            coverage.loc[coverage.inside_outer_fold_window, [f"raw_probability_{m}" for m in CANDIDATES]].notna().any(axis=1).sum()
        ),
        "all_source_clean_outer_markets_with_no_raw_prediction": int(
            coverage.loc[coverage.inside_outer_fold_window, [f"raw_probability_{m}" for m in CANDIDATES]].isna().all(axis=1).sum()
        ),
    }


def _memory_observation() -> dict:
    observation = {"available_memory_gb": None, "runner_working_set_gb": None}
    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            observation["available_memory_gb"] = status.ullAvailPhys / (1024 ** 3)
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb,
        ):
            observation["runner_working_set_gb"] = counters.WorkingSetSize / (1024 ** 3)
    else:
        try:
            import psutil

            observation["available_memory_gb"] = psutil.virtual_memory().available / (1024 ** 3)
            observation["runner_working_set_gb"] = psutil.Process().memory_info().rss / (1024 ** 3)
        except ImportError:
            pass
    return observation


class _RuntimeSampler:
    def __init__(self, interval_seconds: float = 2.0):
        import threading

        self.started = _memory_observation()
        self.peak_runner_working_set_gb = self.started["runner_working_set_gb"]
        self.minimum_available_memory_gb = self.started["available_memory_gb"]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, args=(interval_seconds,), daemon=True)
        self._thread.start()

    def _sample(self, interval_seconds: float) -> None:
        while not self._stop.wait(interval_seconds):
            observation = _memory_observation()
            working_set = observation["runner_working_set_gb"]
            available = observation["available_memory_gb"]
            if working_set is not None:
                self.peak_runner_working_set_gb = max(
                    self.peak_runner_working_set_gb or 0.0, working_set,
                )
            if available is not None:
                self.minimum_available_memory_gb = min(
                    self.minimum_available_memory_gb or available, available,
                )

    def finish(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=3.0)
        observation = _memory_observation()
        working_set = observation["runner_working_set_gb"]
        available = observation["available_memory_gb"]
        if working_set is not None:
            self.peak_runner_working_set_gb = max(
                self.peak_runner_working_set_gb or 0.0, working_set,
            )
        if available is not None:
            self.minimum_available_memory_gb = min(
                self.minimum_available_memory_gb or available, available,
            )
        return {
            "runner_peak_working_set_gb_sampled": (
                round(self.peak_runner_working_set_gb, 3)
                if self.peak_runner_working_set_gb is not None else None
            ),
            "system_minimum_available_memory_gb_sampled": (
                round(self.minimum_available_memory_gb, 3)
                if self.minimum_available_memory_gb is not None else None
            ),
            "sampling_interval_seconds": 2.0,
            "system_memory_includes_unrelated_processes": True,
        }


def _start_runtime_ledger(
        path: Path, identity: dict, budget_seconds: float,
) -> tuple[dict, float, _RuntimeSampler]:
    import datetime as dt

    path = Path(path)
    if path.is_file():
        ledger = json.loads(path.read_text(encoding="utf-8"))
    else:
        ledger = {
            "identity_sha256": _canonical_hash(identity),
            "identity": identity,
            "elapsed_seconds": 0.0,
            "attempts": [],
        }
    if ledger.get("identity_sha256") != _canonical_hash(identity):
        raise RuntimeError(f"Runtime ledger identity mismatch: {path}")
    now = dt.datetime.now(dt.timezone.utc)
    active = ledger.get("active_attempt")
    if active:
        started = dt.datetime.fromisoformat(active["started_utc"])
        if started.tzinfo is None:
            raise RuntimeError(f"Runtime ledger has a naive start time: {path}")
        recovered = max(0.0, (now - started).total_seconds())
        ledger["elapsed_seconds"] = float(ledger.get("elapsed_seconds", 0.0)) + recovered
        ledger["attempts"].append({
            "status": "recovered_after_interruption",
            "elapsed_seconds": recovered,
            "finished_utc": now.isoformat(),
        })
        ledger.pop("active_attempt", None)
    prior = float(ledger.get("elapsed_seconds", 0.0))
    remaining = float(budget_seconds) - prior
    if remaining <= 0:
        raise TimeoutError(f"Frozen runtime budget is exhausted for {path.name}")
    ledger["active_attempt"] = {"started_utc": now.isoformat()}
    _atomic_json(path, ledger)
    return ledger, time.monotonic() + remaining, _RuntimeSampler()


def _finish_runtime_ledger(
        path: Path, ledger: dict, status: str, *, detail: str | None = None,
        sampler: _RuntimeSampler | None = None,
) -> dict:
    import datetime as dt

    now = dt.datetime.now(dt.timezone.utc)
    active = ledger.pop("active_attempt", None)
    elapsed = 0.0
    if active:
        started = dt.datetime.fromisoformat(active["started_utc"])
        elapsed = max(0.0, (now - started).total_seconds())
        ledger["elapsed_seconds"] = float(ledger.get("elapsed_seconds", 0.0)) + elapsed
    attempt = {
        "status": status,
        "elapsed_seconds": elapsed,
        "finished_utc": now.isoformat(),
        "detail": detail,
    }
    if sampler is not None:
        attempt.update(sampler.finish())
    ledger.setdefault("attempts", []).append(attempt)
    _atomic_json(path, ledger)
    return attempt


def _empty_calibration_history():
    import pandas as pd

    return pd.DataFrame(columns=[
        "market_start_utc", "ensemble_probability_raw", "prediction_provenance",
        "fit_cutoff_exclusive_utc", "latest_training_label_available_at_utc",
        "model_identity_sha256", "calibration_outcome_available",
        "official_label_release_at_utc", "target_polymarket_up", "calibration_origin",
        "calibration_label_source",
    ])


def _load_or_fit_ensemble_stage(
        arrays, *, protocol, fold, fold_root, fold_identity, selection,
        memory_names, esn_names, stage_name, fit_cutoff, prediction_start,
        prediction_end, stage_deadline, fold_deadline,
) -> tuple:
    import pandas as pd

    stage_root = fold_root / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)
    prediction_path = stage_root / "predictions.parquet"
    models_path = stage_root / "model_bundles.json"
    checkpoint_path = stage_root / "stage.checkpoint.json"
    identity = {
        "fold_identity_sha256": fold_identity,
        "stage": stage_name,
        "fit_cutoff_exclusive_utc": str(fit_cutoff),
        "prediction_start_utc": str(prediction_start),
        "prediction_end_exclusive_utc": str(prediction_end),
        "selection_identity_sha256": _canonical_hash(selection),
        "base_feature_order_sha256": _canonical_hash(selection["model_feature_order"]),
        "memory_features_sha256": _canonical_hash(memory_names),
        "esn_features_sha256": _canonical_hash(esn_names),
        "runner_sha256": _sha256(Path(__file__)),
    }
    checkpoint = _load_checkpoint(checkpoint_path, identity)
    if checkpoint:
        predictions = pd.read_parquet(prediction_path)
        models = json.loads(models_path.read_text(encoding="utf-8"))
        return predictions, models
    all_predictions = []
    bundle_records = {}
    threads = int(protocol["procedure_budgets"]["threads"])
    backend = str(protocol["procedure_budgets"]["compute_backend"])
    common_deadline = min(float(stage_deadline), float(fold_deadline))
    for procedure in CANDIDATES:
        prediction, model_bundle = fit_and_predict_ensemble(
            arrays,
            procedure=procedure,
            fit_cutoff=fit_cutoff,
            prediction_start=prediction_start,
            prediction_end=prediction_end,
            selection=selection,
            memory_names=memory_names,
            esn_names=esn_names,
            model_root=stage_root / "models",
            backend=backend,
            threads=threads,
            deadline_monotonic=common_deadline,
            prediction_provenance=(
                "fold_local_preselection_chronological_oof"
                if stage_name == "inner_oof" else "fold_external_out_of_sample"
            ),
        )
        validate_oof_provenance(
            prediction,
            fit_cutoff=fit_cutoff,
            prediction_start=prediction_start,
            prediction_end=prediction_end,
            expected_provenance=(
                "fold_local_preselection_chronological_oof"
                if stage_name == "inner_oof" else "fold_external_out_of_sample"
            ),
        )
        prediction["model"] = procedure
        all_predictions.append(prediction)
        bundle_records[procedure] = model_bundle
        if time.monotonic() >= common_deadline:
            raise TimeoutError(f"{stage_name} exhausted its frozen fit/evaluation budget")
    predictions = pd.concat(all_predictions, ignore_index=True)
    _atomic_parquet(predictions, prediction_path)
    _atomic_json(models_path, {
        "stage": stage_name,
        "fold_identity_sha256": fold_identity,
        "procedures": bundle_records,
        "member_counts": {
            procedure: int(bundle_records[procedure]["member_count"])
            for procedure in CANDIDATES
        },
        "uniform_ensemble_weights": {"A": "1/3", "B": "1/3", "C": "1/9", "D": "1/9"},
    })
    model_outputs = [models_path, prediction_path]
    for records in bundle_records.values():
        for member in records["members"]:
            member_path = ROOT / member["model_path"]
            model_outputs.extend([member_path, member_path.with_suffix(".json")])
    _save_checkpoint(checkpoint_path, identity, model_outputs, {
        "procedures": list(CANDIDATES),
        "prediction_rows": int(len(predictions)),
        "bundle_counts": {key: value["member_count"] for key, value in bundle_records.items()},
        "elapsed_seconds": sum(float(value["elapsed_seconds"]) for value in bundle_records.values()),
    })
    if time.monotonic() > common_deadline:
        raise TimeoutError(f"{stage_name} completed after the frozen fit/evaluation budget")
    return predictions, json.loads(models_path.read_text(encoding="utf-8"))


def run_fold(protocol: dict, fold: dict, fold_index: int, fold_root: Path,
             fold_identity: str, gamma, history_by_model: dict, eligibility_markets):
    import pandas as pd

    fold_root.mkdir(parents=True, exist_ok=True)
    complete_path = fold_root / "fold.checkpoint.json"
    complete_identity = {
        "fold_identity_sha256": fold_identity,
        "fold_id": fold["id"],
        "runner_sha256": _sha256(Path(__file__)),
        "official_gamma_filtered_hash": protocol["official_gamma_source"]["filtered_rows_sha256"],
    }
    completed = _load_checkpoint(complete_path, complete_identity)
    if completed:
        outer = pd.read_parquet(fold_root / "outer_calibrated.parquet")
        eligibility = pd.read_parquet(fold_root / "outer_prediction_eligibility.parquet")
        outer_raw_official = {
            model: pd.read_parquet(fold_root / f"outer_history_{model}.parquet")
            for model in CANDIDATES
        }
        all_inner = pd.read_parquet(fold_root / "inner_oof_official.parquet") if fold_index == 0 else None
        for model in CANDIDATES:
            if fold_index == 0:
                inner = all_inner.loc[all_inner.model.eq(model)].copy()
                history_by_model[model] = pd.concat(
                    [inner, outer_raw_official[model]], ignore_index=True, sort=False,
                )
            else:
                history_by_model[model] = pd.concat(
                    [history_by_model[model], outer_raw_official[model]], ignore_index=True,
                )
        return completed["payload"], outer, eligibility

    budget = float(protocol["procedure_budgets"]["maximum_wall_seconds_per_outer_chain"])
    timer_path = fold_root / "fold_runtime.json"
    timer, fold_deadline, fold_sampler = _start_runtime_ledger(
        timer_path, complete_identity, budget,
    )
    fold_started = time.monotonic()
    try:
        remaining = max(1.0, fold_deadline - time.monotonic())
        selection = _run_fold_selection(
            protocol, fold, fold_root, fold_identity, remaining_seconds=remaining,
        )
        dynamic_path, memory_names, esn_names = _build_dynamic_feature_artifact(
            protocol, fold, fold_root, fold_identity,
        )
        arrays = _load_fold_arrays(selection, dynamic_path, memory_names, esn_names)
        selection_cutoff = fold["selection_cutoff_exclusive_utc"]
        outer_start = fold["market_start_test_start_inclusive_utc"]
        outer_end = fold["market_start_test_end_exclusive_utc"]
        eligibility = _prediction_eligibility_for_markets(
            arrays, eligibility_markets.market_start_utc,
        )
        _atomic_parquet(eligibility, fold_root / "outer_prediction_eligibility.parquet")
        shared_fit_budget = float(protocol["procedure_budgets"]["calibration_and_evaluation_bundle_stage_seconds"])
        fit_timer_path = fold_root / "ensemble_fit_runtime.json"
        fit_timer, fit_deadline, fit_sampler = _start_runtime_ledger(
            fit_timer_path,
            {"fold_identity_sha256": fold_identity, "stage": "inner_and_outer_ensemble_fit"},
            shared_fit_budget,
        )
        try:
            inner_predictions, inner_models = _load_or_fit_ensemble_stage(
                arrays, protocol=protocol, fold=fold, fold_root=fold_root,
                fold_identity=fold_identity, selection=selection,
                memory_names=memory_names, esn_names=esn_names,
                stage_name="inner_oof", fit_cutoff=selection_cutoff,
                prediction_start=selection_cutoff, prediction_end=outer_start,
                stage_deadline=fit_deadline, fold_deadline=fold_deadline,
            )
            outer_raw, outer_models = _load_or_fit_ensemble_stage(
                arrays, protocol=protocol, fold=fold, fold_root=fold_root,
                fold_identity=fold_identity, selection=selection,
                memory_names=memory_names, esn_names=esn_names,
                stage_name="outer_predictions_raw", fit_cutoff=fold["fit_cutoff_exclusive_utc"],
                prediction_start=outer_start, prediction_end=outer_end,
                stage_deadline=fit_deadline, fold_deadline=fold_deadline,
            )
        except Exception as exc:
            _finish_runtime_ledger(
                fit_timer_path, fit_timer, "failed", detail=repr(exc),
                sampler=fit_sampler,
            )
            raise

        inner_official_by_model = {}
        for model in CANDIDATES:
            model_oof = inner_predictions.loc[inner_predictions.model.eq(model)].copy()
            validate_oof_provenance(
                model_oof,
                fit_cutoff=selection_cutoff,
                prediction_start=selection_cutoff,
                prediction_end=outer_start,
            )
            inner_official_by_model[model] = _join_gamma_outcomes(
                model_oof, gamma, source="fold_local_preselection_chronological_oof",
            )
        outer_calibrated_by_model = []
        daily_calibrators = {}
        outer_history_by_model = {}
        for model in CANDIDATES:
            model_outer = outer_raw.loc[outer_raw.model.eq(model)].copy()
            validate_oof_provenance(
                model_outer,
                fit_cutoff=fold["fit_cutoff_exclusive_utc"],
                prediction_start=outer_start,
                prediction_end=outer_end,
                expected_provenance="fold_external_out_of_sample",
            )
            current_history = (
                inner_official_by_model[model]
                if fold_index == 0
                else history_by_model[model]
            )
            calibrated, daily = calibrate_external_daily(
                model_outer, current_history, gamma,
                minimum_rows=int(protocol["official_calibration"]["minimum_rows"]),
            )
            calibrated["model"] = model
            outer_calibrated_by_model.append(calibrated)
            daily_calibrators[model] = daily
            raw_official = _join_gamma_outcomes(
                model_outer, gamma, source="fold_external_out_of_sample",
            )
            outer_history_by_model[model] = raw_official
            history_by_model[model] = pd.concat(
                [current_history, raw_official], ignore_index=True, sort=False,
            )
            if history_by_model[model].market_start_utc.duplicated().any():
                raise RuntimeError(f"Calibration history overlaps by market for model {model}")
            history_by_model[model].sort_values("market_start_utc", kind="stable", inplace=True)
            history_by_model[model].reset_index(drop=True, inplace=True)
        outer_calibrated = pd.concat(outer_calibrated_by_model, ignore_index=True)
        inner_official = pd.concat(inner_official_by_model.values(), ignore_index=True)
        _atomic_parquet(inner_official, fold_root / "inner_oof_official.parquet")
        _atomic_parquet(outer_calibrated, fold_root / "outer_calibrated.parquet")
        for model, frame in outer_history_by_model.items():
            _atomic_parquet(frame, fold_root / f"outer_history_{model}.parquet")
        _atomic_json(fold_root / "daily_calibrators.json", daily_calibrators)
        if time.monotonic() > min(fit_deadline, fold_deadline):
            raise TimeoutError("Ensemble fitting and official calibration exceeded the shared fold budgets")
        summary = {
            "fold_id": fold["id"],
            "status": "chronological_fit_completed",
            "selection_cutoff_exclusive_utc": selection_cutoff,
            "outer_fit_cutoff_exclusive_utc": fold["fit_cutoff_exclusive_utc"],
            "outer_prediction_window_utc": [outer_start, outer_end],
            "selection_identity_sha256": _canonical_hash(selection),
            "selected_base_feature_count": len(selection["selected_base_features"]),
            "model_feature_order": selection["model_feature_order"],
            "decision_weight": selection["decision_weight"],
            "best_params": selection["best_params"],
            "best_iteration": selection["best_iteration"],
            "inner_oof_rows_by_model": {
                model: int((inner_predictions.model == model).sum()) for model in CANDIDATES
            },
            "outer_raw_prediction_rows_by_model": {
                model: int((outer_raw.model == model).sum()) for model in CANDIDATES
            },
            "outer_calibrated_prediction_rows_by_model": {
                model: int(outer_calibrated.loc[
                    outer_calibrated.model.eq(model), "ensemble_probability_calibrated",
                ].notna().sum()) for model in CANDIDATES
            },
            "inner_model_bundles": inner_models,
            "outer_model_bundles": outer_models,
            "calibration_history_rule": (
                "F1 uses its earlier-only inner OOF history; F2/F3 use the preceding continuous "
                "OOF stream plus prior external predictions, never overlapping inner OOF rows."
            ),
            "daily_calibrators": daily_calibrators,
            "elapsed_seconds_this_attempt": time.monotonic() - fold_started,
            "fold_runtime_ledger": timer_path.resolve().relative_to(ROOT).as_posix(),
            "holdout_rows_materialized": 0,
        }
        stage_outputs = [
            fold_root / "inner_oof/predictions.parquet",
            fold_root / "inner_oof/model_bundles.json",
            fold_root / "inner_oof_official.parquet",
            fold_root / "outer_predictions_raw/predictions.parquet",
            fold_root / "outer_predictions_raw/model_bundles.json",
            fold_root / "inner_oof_official.parquet",
            fold_root / "outer_calibrated.parquet",
            fold_root / "outer_prediction_eligibility.parquet",
            fold_root / "daily_calibrators.json",
            fold_root / "fold_summary.json",
            *(fold_root / f"outer_history_{model}.parquet" for model in CANDIDATES),
        ]
        model_paths = []
        for stage in (inner_models, outer_models):
            for procedure in CANDIDATES:
                for member in stage["procedures"][procedure]["members"]:
                    model_path = ROOT / member["model_path"]
                    model_paths.extend([model_path, model_path.with_suffix(".json")])
        stage_outputs.extend(model_paths)
        summary["ensemble_fit_resource_sample"] = _finish_runtime_ledger(
            fit_timer_path, fit_timer, "completed", sampler=fit_sampler,
        )
        summary["fold_resource_sample"] = _finish_runtime_ledger(
            timer_path, timer, "completed", sampler=fold_sampler,
        )
        _atomic_json(fold_root / "fold_summary.json", summary)
        _save_checkpoint(complete_path, complete_identity, stage_outputs, summary)
        return summary, outer_calibrated, eligibility
    except Exception as exc:
        if "fit_timer" in locals() and fit_timer.get("active_attempt"):
            _finish_runtime_ledger(
                fit_timer_path, fit_timer, "failed", detail=repr(exc),
                sampler=fit_sampler,
            )
        _finish_runtime_ledger(
            timer_path, timer, "failed", detail=repr(exc), sampler=fold_sampler,
        )
        raise


def _validate_fold_protocol(protocol: dict) -> None:
    import pandas as pd

    folds = protocol.get("outer_folds", [])
    expected = [
        ("F1_20260415_20260501", "2020-06-09T09:33:00Z", "2025-07-01T00:00:00Z", "2026-04-15T00:00:00Z", "2026-05-01T00:00:00Z"),
        ("F2_20260501_20260601", "2020-06-09T09:33:00Z", "2025-08-01T00:00:00Z", "2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"),
        ("F3_20260601_20260701", "2020-06-09T09:33:00Z", "2025-09-01T00:00:00Z", "2026-06-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    ]
    actual = [
        (
            fold.get("id"), fold.get("fit_data_start_utc"),
            fold.get("selection_cutoff_exclusive_utc"),
            fold.get("market_start_test_start_inclusive_utc"),
            fold.get("market_start_test_end_exclusive_utc"),
        )
        for fold in folds
    ]
    if actual != expected:
        raise RuntimeError("E0039 outer/inner fold dates or order differ from the frozen protocol")
    if any(fold.get("fit_cutoff_exclusive_utc") != fold.get("market_start_test_start_inclusive_utc") for fold in folds):
        raise RuntimeError("Each external fit cutoff must equal its frozen outer-window start")
    if protocol.get("official_calibration", {}).get("minimum_rows") != 1000:
        raise RuntimeError("E0039 official Gamma Platt minimum must remain 1,000 rows")
    if protocol.get("strategy_invariants", {}).get("ev_rule") != "strictly_positive_net_expected_value":
        raise RuntimeError("E0039 strategy must use the frozen strict-positive EV rule")
    for fold in folds:
        if not fold["raw_prefix_path"].startswith("data/analysis/btc_chronology_reconstruction_20261027/raw_prefixes/"):
            raise RuntimeError("Outer fold raw source escaped the frozen development prefix directory")
        if not fold["flow_prefix_path"].startswith("data/analysis/btc_chronology_reconstruction_20261027/flow_prefixes/"):
            raise RuntimeError("Outer fold flow source escaped the frozen causal prefix directory")
        end = pd.Timestamp(fold["market_start_test_end_exclusive_utc"])
        if end > pd.Timestamp("2026-07-01T00:00:00Z"):
            raise RuntimeError("Outer fold reaches the protected holdout")


def _write_run_report(path: Path, manifest: dict) -> None:
    folds = manifest.get("folds", {})
    lines = [
        "# E0039 BTC chronology reconstruction run",
        "",
        f"- Status: `{manifest.get('status')}`",
        f"- Run identity: `{manifest.get('run_identity_sha256')}`",
        f"- Protocol: `{manifest.get('protocol_path')}` (`{manifest.get('protocol_sha256')}`)",
        f"- Parent protocol: `{manifest.get('previous_protocol_path')}` (`{manifest.get('previous_protocol_sha256')}`)",
        f"- Runtime seconds: `{manifest.get('elapsed_seconds')}`",
        f"- Protected holdout rows materialized: `{manifest.get('holdout_rows_materialized', 0)}`",
        "",
        "## Fold evidence",
        "",
    ]
    for fold_id, result in folds.items():
        lines.extend([
            f"### {fold_id}",
            "",
            f"- Status: `{result.get('status')}`",
            f"- Selection cutoff: `{result.get('selection_cutoff_exclusive_utc')}`",
            f"- Outer fit cutoff: `{result.get('outer_fit_cutoff_exclusive_utc')}`",
            f"- Selected base features: `{result.get('selected_base_feature_count')}`",
            f"- OOF rows: `{result.get('inner_oof_rows_by_model')}`",
            f"- Outer raw rows: `{result.get('outer_raw_prediction_rows_by_model')}`",
            f"- Calibrated rows: `{result.get('outer_calibrated_prediction_rows_by_model')}`",
            f"- Fitting resource sample: `{json.dumps(result.get('ensemble_fit_resource_sample', {}), sort_keys=True)}`",
            f"- Fold resource sample: `{json.dumps(result.get('fold_resource_sample', {}), sort_keys=True)}`",
            "",
        ])
    if manifest.get("economics"):
        lines.extend([
            "## Source-clean economics",
            "",
            f"- Frozen E0038 whitelist: `{manifest['economics'].get('source_clean_whitelist_count')}`",
            f"- Common fully calibrated markets: `{manifest['economics'].get('common_calibrated_prediction_market_count')}`",
            f"- Coverage by model: `{json.dumps(manifest['economics'].get('coverage_by_model', {}), sort_keys=True)}`",
            f"- Paired independent-$5 differences on common calibrated coverage: `{json.dumps(manifest['economics'].get('paired_independent_fixed5_60s_common_coverage', {}), sort_keys=True)}`",
            f"- Monthly and paired ledgers: `{manifest['economics'].get('monthly_paths_path')}`, `{manifest['economics'].get('paired_monthly_differences_path')}`",
            "- Unknown coverage rows remain unknown and are not labelled `no_trade`.",
            "",
        ])
    lines.extend([
        "## Offline runtime",
        "",
        f"- Status: `{manifest.get('offline_runtime', {}).get('status', 'NOT_RUN')}`",
        f"- Detail: {manifest.get('offline_runtime', {}).get('detail', 'Not yet implemented.')}",
        "",
        "## Interpretation limits",
        "",
        "- This is a retrospective chronological reproduction conditional on the previously exposed A/B/C/D family search.",
        "- Historical T-59 readiness and network delay remain `UNKNOWN`.",
        "- E0037 remains historical `FAILED`; E0038 artifacts are unchanged.",
        "- No holdout, confirmation, PMXT replay, source acquisition, shadow run, or order was performed.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    import pandas as pd

    started = time.perf_counter()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _validate_fold_protocol(protocol)
    protocol_sha = _sha256(PROTOCOL_PATH)
    root_identity = _canonical_hash({
        "protocol_sha256": protocol_sha,
        "previous_protocol_sha256": protocol["previous_protocol_sha256"],
        "implementation_hashes": protocol["implementation_hashes"],
        "frozen_inputs": protocol["frozen_inputs"],
        "filtered_source_row_hashes": protocol["filtered_source_row_hashes"],
    })
    minimum_memory = float(protocol["resource_gate"]["minimum_free_memory_gb_before_fit"])
    try:
        require_gpu = str(protocol["procedure_budgets"]["compute_backend"]).lower() == "gpu"
        resources = _resource_snapshot(minimum_memory, require_gpu=require_gpu)
    except Exception as exc:
        resources = None
        resource_error = str(exc)
    else:
        resource_error = None
    if resources is None or resources["gate_failures"]:
        blocked_dir = OUTPUT_ROOT / "blocked_before_fit" / root_identity[:16]
        reason = resource_error or "; ".join(resources["gate_failures"])
        _atomic_json(blocked_dir / "pre_fit_gate.json", {
            "experiment_id": "E0039",
            "status": "BLOCKED_BEFORE_FIT",
            "reason": reason,
            "resource_snapshot": resources,
            "protocol_path": PROTOCOL_PATH.resolve().relative_to(ROOT).as_posix(),
            "protocol_sha256": protocol_sha,
            "minimum_free_memory_gb": minimum_memory,
            "data_or_label_reads_for_fit": False,
            "holdout_rows_materialized": 0,
            "next_action": "Resume this runner after active Python jobs exit and the frozen RAM/disk gate passes.",
        })
        return {
            "status": "BLOCKED_BEFORE_FIT",
            "reason": reason,
            "resource_snapshot": resources,
            "pre_fit_gate_path": (blocked_dir / "pre_fit_gate.json").resolve().relative_to(ROOT).as_posix(),
            "holdout_rows_materialized": 0,
        }

    verified = _verify_frozen_inputs(protocol)
    whitelist, clean_ids_hash = _load_clean_whitelist(protocol)
    filtered_hashes = verify_filtered_source_hashes(protocol, whitelist, clean_ids_hash)
    protocol["official_gamma_source"]["filtered_rows_sha256"] = filtered_hashes["official_gamma"]
    run_identity = _canonical_hash({
        "root_identity_sha256": root_identity,
        "verified_files": verified,
        "clean_ids_sha256": clean_ids_hash,
        "filtered_source_row_hashes": filtered_hashes,
    })
    run_root = OUTPUT_ROOT / "runs" / run_identity[:16]
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "run_manifest.json"
    manifest = {
        "experiment_id": "E0039",
        "status": "running",
        "protocol_path": PROTOCOL_PATH.resolve().relative_to(ROOT).as_posix(),
        "protocol_sha256": protocol_sha,
        "previous_protocol_path": PREVIOUS_PROTOCOL_PATH.resolve().relative_to(ROOT).as_posix(),
        "previous_protocol_sha256": protocol["previous_protocol_sha256"],
        "run_identity_sha256": run_identity,
        "verified_file_hashes": verified,
        "filtered_source_row_hashes": filtered_hashes,
        "clean_ids_sha256": clean_ids_hash,
        "resource_snapshot_before_fits": resources,
        "holdout_rows_materialized": 0,
        "confirmation_rows_materialized": 0,
        "source_clean_whitelist_count": int(len(whitelist)),
        "folds": {},
        "offline_runtime": {
            "status": "PARTIAL_NOT_IMPLEMENTED",
            "detail": "Batch and causal state construction is implemented. A full selected-generator/VP/RP streaming adapter and its frozen 300-window hot-path benchmark remain incomplete.",
        },
    }
    _atomic_json(manifest_path, manifest)
    try:
        gamma = load_official_gamma(protocol)
        histories = {model: _empty_calibration_history() for model in CANDIDATES}
        outer_predictions = []
        outer_eligibility = []
        for index, fold in enumerate(protocol["outer_folds"]):
            fold_start = pd.Timestamp(fold["market_start_test_start_inclusive_utc"])
            fold_end = pd.Timestamp(fold["market_start_test_end_exclusive_utc"])
            fold_markets = whitelist.loc[
                whitelist.market_start_utc.ge(fold_start)
                & whitelist.market_start_utc.lt(fold_end)
            ]
            fold_identity = _canonical_hash({
                "run_identity_sha256": run_identity,
                "fold": fold,
                "selection_cutoff_exclusive_utc": fold["selection_cutoff_exclusive_utc"],
                "allowed_code_hashes": protocol["implementation_hashes"],
            })
            fold_root = run_root / fold["id"] / fold_identity[:16]
            result, predictions, eligibility = run_fold(
                protocol, fold, index, fold_root, fold_identity, gamma, histories,
                fold_markets,
            )
            manifest["folds"][fold["id"]] = result
            outer_predictions.append(predictions)
            outer_eligibility.append(eligibility)
            manifest["last_completed_fold"] = fold["id"]
            manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
            _atomic_json(manifest_path, manifest)
        prediction_panel = pd.concat(outer_predictions, ignore_index=True)
        eligibility_panel = pd.concat(outer_eligibility, ignore_index=True)
        snapshots = load_economics_source_snapshots(protocol, whitelist)
        prediction_panel_path = run_root / "outer_predictions_all_folds.parquet"
        _atomic_parquet(prediction_panel, prediction_panel_path)
        economics = run_economics(
            protocol, whitelist, snapshots, prediction_panel, eligibility_panel,
            run_root / "economics",
        )
        manifest["economics"] = economics
        manifest["outer_predictions_path"] = prediction_panel_path.resolve().relative_to(ROOT).as_posix()
        manifest["outer_predictions_sha256"] = _sha256(prediction_panel_path)
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["status"] = "completed_folds_economics_runtime_partial"
        manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        report_path = ROOT / "reports/btc_preopen/btc_chronology_reconstruction_20261027_v7.md"
        manifest["report_path"] = report_path.resolve().relative_to(ROOT).as_posix()
        _atomic_json(manifest_path, manifest)
        _write_run_report(report_path, manifest)
        return manifest
    except Exception as exc:
        manifest["status"] = "interrupted_or_failed_with_checkpoints"
        manifest["failure"] = {"type": type(exc).__name__, "detail": str(exc)}
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        _atomic_json(manifest_path, manifest)
        report_path = ROOT / "reports/btc_preopen/btc_chronology_reconstruction_20261027_v7.md"
        manifest["report_path"] = report_path.resolve().relative_to(ROOT).as_posix()
        _write_run_report(report_path, manifest)
        raise


def main() -> None:
    result = run()
    print(json.dumps({
        "status": result.get("status"),
        "reason": result.get("reason"),
        "run_identity_sha256": result.get("run_identity_sha256"),
        "last_completed_fold": result.get("last_completed_fold"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "holdout_rows_materialized": result.get("holdout_rows_materialized", 0),
    }, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
