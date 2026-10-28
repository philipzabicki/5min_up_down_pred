"""Verify and package the saved E0041 D bundle without fitting models."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.btc_d_bundle import (
    bundle_status_from_calibration,
    calibration_manifest_fields,
    verify_file_hash,
)


FINALIZATION_PROTOCOL_PATH = ROOT / "configs/research/btc_D_bundle_finalization_20261028.json"
PREPARATION_PROTOCOL_PATH = ROOT / "configs/research/btc_D_holdout_preparation_20261028_v2.json"
PREPARATION_ROOT = ROOT / "data/analysis/polymarket/BTC/btc_D_holdout_preparation_20261028"
RUN_IDENTITY = "189049366841f6d025e3d9b080bb69dd0e441bb5638e451708f27c0635754c44"
RUN_ROOT = PREPARATION_ROOT / "runs" / RUN_IDENTITY[:16]
PREPARATION_MANIFEST_PATH = RUN_ROOT / "preparation_manifest.json"
BUNDLE_PATH = RUN_ROOT / "final_D_bundle/bundle_manifest.json"
BUNDLE_POINTER_PATH = PREPARATION_ROOT / "final_D_bundle_manifest.json"
FINALIZATION_RESULT_PATH = RUN_ROOT / "finalization_result.json"
FIT_CUTOFF_UTC = "2026-06-30T23:59:00Z"
HOLDOUT_START_UTC = "2026-07-01T00:00:00Z"
SAFE_OPENED_END_UTC = "2026-06-30T23:58:00Z"
LIGHTGBM_SEEDS = (20261101, 20261102, 20261103)
RESERVOIR_SEEDS = (20261023, 20261024, 20261025)
FLOW_NAMES = (
    "taker_Z_1m", "taker_V_1m", "taker_Z_5m", "taker_V_5m", "taker_Z_15m", "taker_V_15m",
)
TOLERANCE = 2e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n").encode("utf-8")


def _available_ram_bytes() -> int:
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed while checking finalizer memory requirements")
        return int(status.ullAvailPhys)
    return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Required E0041 metadata is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    os.replace(temporary, path)


def _preflight_json_output(path: Path, value: dict) -> None:
    if path.exists() and path.read_bytes() != _json_bytes(value):
        raise RuntimeError(f"Refusing to overwrite an unexpected existing finalization output: {path}")


def _verify_protocol_and_sources() -> tuple[dict, dict, dict, dict]:
    finalization = _read_json(FINALIZATION_PROTOCOL_PATH)
    preparation_protocol = _read_json(PREPARATION_PROTOCOL_PATH)
    if finalization.get("protocol_version") != 1 or finalization.get("experiment_id") != "E0041":
        raise RuntimeError("E0041 finalization protocol version or experiment identity is invalid")
    if (finalization.get("source_run_identity_sha256") != RUN_IDENTITY
            or finalization.get("source_run_root") != RUN_ROOT.relative_to(ROOT).as_posix()):
        raise RuntimeError("E0041 finalization protocol points to a different saved source run")
    expected_outputs = {
        "bundle_manifest": BUNDLE_PATH.relative_to(ROOT).as_posix(),
        "bundle_pointer": BUNDLE_POINTER_PATH.relative_to(ROOT).as_posix(),
        "finalization_result": FINALIZATION_RESULT_PATH.relative_to(ROOT).as_posix(),
    }
    if finalization.get("output_paths") != expected_outputs:
        raise RuntimeError("E0041 finalization output paths differ from the versioned protocol")
    if _sha256(PREPARATION_PROTOCOL_PATH) != finalization["preparation_protocol_sha256"]:
        raise RuntimeError("Frozen E0041 v2 preparation protocol hash changed")
    if preparation_protocol.get("protocol_version") != 2:
        raise RuntimeError("Finalization requires the frozen E0041 v2 protocol")
    if preparation_protocol["evaluation"]["evaluation_authorized"] is not False:
        raise RuntimeError("E0041 holdout authorization must remain false")
    if preparation_protocol["final_model"]["fit_cutoff_exclusive_utc"] != FIT_CUTOFF_UTC:
        raise RuntimeError("Frozen E0041 final-fit cutoff changed")
    if int(preparation_protocol["calibration"]["minimum_rows"]) != int(finalization["calibration_minimum_rows"]):
        raise RuntimeError("Finalization and frozen calibration minimum disagree")

    parent_path = ROOT / finalization["parent_protocol"]["path"]
    verify_file_hash(parent_path, finalization["parent_protocol"]["sha256"], "E0039 parent protocol")
    if preparation_protocol["parent_e0039"]["protocol_sha256"] != finalization["parent_protocol"]["sha256"]:
        raise RuntimeError("E0041 v2 does not identify the pinned E0039 parent protocol")

    code_hashes = finalization["code_hashes"]
    code_paths = {
        "finalizer_sha256": ROOT / "scripts/finalize_btc_D_bundle_20261028.py",
        "packaging_preparation_runner_sha256": ROOT / "run_btc_D_holdout_preparation_20261028.py",
        "evaluation_runner_sha256": ROOT / "run_btc_D_holdout_evaluation_20261028.py",
        "frozen_preparation_test_sha256": ROOT / "tests/test_btc_D_holdout_preparation_20261028.py",
        "finalization_test_sha256": ROOT / "tests/test_btc_D_bundle_finalization_20261028.py",
        "bundle_helper_sha256": ROOT / "utils/btc_d_bundle.py",
    }
    for key, path in code_paths.items():
        verify_file_hash(path, code_hashes[key], path.relative_to(ROOT).as_posix())
    if code_hashes["historical_preparation_runner_sha256"] != preparation_protocol["outputs"]["preparation_runner_sha256"]:
        raise RuntimeError("Finalization protocol changed the historical training-runner identity")
    if code_hashes["evaluation_runner_sha256"] != preparation_protocol["outputs"]["evaluation_runner_sha256"]:
        raise RuntimeError("Finalization protocol changed the frozen evaluator identity")
    if code_hashes["frozen_preparation_test_sha256"] != preparation_protocol["outputs"]["focused_test_sha256"]:
        raise RuntimeError("Finalization protocol changed the frozen preparation-test identity")

    preparation = _read_json(PREPARATION_MANIFEST_PATH)
    source_manifest_hash = finalization["source_preparation_manifest_sha256"]
    existing_finalization = preparation.get("finalization")
    if _sha256(PREPARATION_MANIFEST_PATH) != source_manifest_hash:
        if not isinstance(existing_finalization, dict) or existing_finalization.get("source_preparation_manifest_sha256") != source_manifest_hash:
            raise RuntimeError("Source preparation manifest changed outside this finalization")
    if preparation.get("run_identity_sha256") != RUN_IDENTITY:
        raise RuntimeError("Saved preparation manifest belongs to a different source run")
    if preparation.get("protocol_sha256") != finalization["preparation_protocol_sha256"]:
        raise RuntimeError("Saved preparation manifest points to a different preparation protocol")
    if preparation.get("runner_sha256") != code_hashes["historical_preparation_runner_sha256"]:
        raise RuntimeError("Saved model run does not match the historical preparation-runner hash")
    failure = preparation.get("failure")
    if not isinstance(failure, dict) or failure.get("type") != "KeyError" or failure.get("detail") != "'rows'":
        raise RuntimeError("Original E0041 packaging failure is missing or has been replaced")
    if preparation.get("evaluation_authorized") is not False:
        raise RuntimeError("Saved preparation manifest authorizes holdout evaluation")
    for key in ("holdout_rows_read", "holdout_rows_materialized", "holdout_predictions_generated"):
        if int(preparation.get(key, -1)) != 0:
            raise RuntimeError(f"Saved preparation manifest has nonzero holdout activity: {key}")

    for item in finalization["verified_files"]:
        verify_file_hash(ROOT / item["path"], item["sha256"], item["label"])
    return finalization, preparation_protocol, preparation, code_hashes


def _validate_selection_and_sidecars(finalization: dict, preparation_protocol: dict) -> tuple[dict, dict, dict]:
    selection = _read_json(ROOT / finalization["selection_result_path"])
    pipeline_manifest = _read_json(ROOT / finalization["selection_pipeline_manifest_path"])
    if _canonical_hash(selection.get("identity_payload")) != selection.get("identity"):
        raise RuntimeError("Saved selection checkpoint identity payload hash is invalid")
    if selection.get("identity") != finalization["selection_result_identity_sha256"]:
        raise RuntimeError("Saved selection checkpoint identity differs")
    selected = selection.get("payload")
    if not isinstance(selected, dict) or selected.get("status") != "fold_local_selection_completed":
        raise RuntimeError("E0041 development selection checkpoint is incomplete")
    if selected.get("selected_base_feature_count") != 64 or len(selected.get("selected_base_features", [])) != 64:
        raise RuntimeError("Saved D selection does not contain 64 ordered base features")
    base_order = list(selected.get("model_feature_order", []))
    if len(base_order) != 70 or base_order != list(selected["selected_base_features"]) + list(FLOW_NAMES):
        raise RuntimeError("Saved D base feature order is not BASE64 followed by the six frozen flow columns")
    if selected.get("selection_cutoff_exclusive_utc") not in {
        "2025-10-01T00:00:00Z", "2025-10-01 00:00:00+00:00",
    }:
        raise RuntimeError("Saved D selection cutoff changed")
    if selected.get("exclusive_label_availability") is not True:
        raise RuntimeError("Saved D selection does not attest strict label availability")
    if not selected.get("latest_training_label_available_at_utc"):
        raise RuntimeError("Saved D selection is missing its latest eligible training-label timestamp")

    stage = pipeline_manifest["stages"]["causal_candidate_feature_join"]
    dataset_path = ROOT / selected["dataset_path"]
    dataset_output = next((entry for entry in stage["outputs"] if entry["path"] == selected["dataset_path"]), None)
    if dataset_output is None or dataset_output["sha256"] != finalization["development_feature_dataset_sha256"]:
        raise RuntimeError("Development feature dataset is not pinned by its completed stage manifest")
    if pipeline_manifest.get("status") != "chronology_selection_completed":
        raise RuntimeError("E0041 chronology selection pipeline is not complete")

    dynamic_checkpoint = _read_json(ROOT / finalization["dynamic_features_checkpoint_path"])
    if _canonical_hash(dynamic_checkpoint.get("identity_payload")) != dynamic_checkpoint.get("identity"):
        raise RuntimeError("Saved causal dynamic-feature checkpoint identity hash is invalid")
    dynamic_record = next((entry for entry in dynamic_checkpoint["outputs"]
                           if entry["path"] == finalization["dynamic_features_path"]), None)
    if dynamic_record is None or dynamic_record["sha256"] != finalization["dynamic_features_sha256"]:
        raise RuntimeError("Saved causal dynamic-feature artifact is not pinned by its checkpoint")
    payload = dynamic_checkpoint.get("payload", {})
    memory_names = list(payload.get("memory_columns", []))
    esn_names = {int(seed): list(names) for seed, names in payload.get("esn_columns_by_seed", {}).items()}
    if len(memory_names) != 8 or sorted(esn_names) != list(RESERVOIR_SEEDS) or any(len(names) != 8 for names in esn_names.values()):
        raise RuntimeError("Saved causal transforms do not provide memory PCA8 and all three ESN PCA8 blocks")
    transform_hashes = dynamic_checkpoint["identity_payload"]
    if transform_hashes.get("memory_transform_sha256") != finalization["memory_transform_sha256"]:
        raise RuntimeError("Saved dynamic features use a different frozen memory transform")
    if transform_hashes.get("esn_transform_sha256") != finalization["esn_transform_sha256"]:
        raise RuntimeError("Saved dynamic features use a different frozen ESN transform")
    parent = _read_json(ROOT / finalization["parent_protocol"]["path"])
    historical_implementations = {
        "features/causal_memory_control.py": "memory_state_code_sha256",
        "features/esn_reservoir.py": "feature_code_sha256",
        "run_btc_candidate_holdout_20261022.py": "transform_runner_sha256",
        "run_btc_chronology_reconstruction_20261027.py": None,
    }
    for path, checkpoint_key in historical_implementations.items():
        expected_hash = parent["implementation_hashes"][path]
        if checkpoint_key is not None and transform_hashes.get(checkpoint_key) != expected_hash:
            raise RuntimeError(f"Saved dynamic features used a different frozen implementation: {path}")
        verify_file_hash(ROOT / path, expected_hash, f"Frozen E0039 implementation {path}")
    if int(payload.get("rows", -1)) != int(finalization["development_feature_rows"]):
        raise RuntimeError("Saved dynamic-feature row count differs from the development feature manifest")

    model_records = []
    reference_identity = None
    feature_orders = {}
    for expected in finalization["model_members"]:
        name = Path(expected["model_path"]).stem
        model_path = ROOT / expected["model_path"]
        sidecar_path = ROOT / expected["metadata_path"]
        verify_file_hash(model_path, expected["model_sha256"], f"Final D model {name}")
        verify_file_hash(sidecar_path, expected["metadata_sha256"], f"Final D sidecar {name}")
        sidecar = _read_json(sidecar_path)
        identity = sidecar.get("identity")
        if not isinstance(identity, dict) or _canonical_hash(identity) != sidecar.get("identity_sha256"):
            raise RuntimeError(f"Final D model sidecar identity is invalid: {name}")
        parts = name.split("_")
        if len(parts) != 3 or parts[0] != "D":
            raise RuntimeError(f"Final D model name is invalid: {name}")
        reservoir_seed = int(parts[1][1:])
        lightgbm_seed = int(parts[2][1:])
        expected_order = [*base_order, *memory_names, *esn_names[reservoir_seed]]
        if identity.get("procedure") != "D" or identity.get("reservoir_seed") != reservoir_seed or identity.get("lightgbm_seed") != lightgbm_seed:
            raise RuntimeError(f"Final D sidecar seed identity differs from filename: {name}")
        if identity.get("feature_order") != expected_order or identity.get("feature_order_sha256") != _canonical_hash(expected_order):
            raise RuntimeError(f"Final D sidecar feature order differs from its saved transforms: {name}")
        if int(sidecar.get("num_features", -1)) != 86 or len(expected_order) != 86:
            raise RuntimeError(f"Final D member does not have 86 features: {name}")
        if int(sidecar.get("num_trees", -1)) != int(identity.get("best_iteration", -2)):
            raise RuntimeError(f"Final D member tree count differs from best_iteration: {name}")
        if identity.get("fit_cutoff_exclusive_utc") != FIT_CUTOFF_UTC.replace("Z", "+00:00"):
            raise RuntimeError(f"Final D member fit cutoff differs from protocol: {name}")
        if int(identity.get("training_rows", -1)) != int(finalization["training_rows"]):
            raise RuntimeError(f"Final D member training-row count differs: {name}")
        for key in ("best_iteration", "best_params", "training_rows", "fit_cutoff_exclusive_utc",
                    "training_row_opened_sha256", "training_labels_sha256", "training_weights_sha256"):
            if reference_identity is None:
                continue
            if identity.get(key) != reference_identity.get(key):
                raise RuntimeError(f"Final D members disagree on shared training metadata field {key}")
        if reference_identity is None:
            reference_identity = identity
        feature_matrix_hashes = finalization["training_feature_matrix_sha256_by_reservoir"]
        if identity.get("training_feature_matrix_sha256") != feature_matrix_hashes[str(reservoir_seed)]:
            raise RuntimeError(f"Final D member training-feature identity differs for reservoir {reservoir_seed}")
        feature_orders[name] = expected_order
        model_records.append({
            "path": expected["model_path"],
            "sha256": expected["model_sha256"],
            "name": name,
            "reservoir_seed": reservoir_seed,
            "lightgbm_seed": lightgbm_seed,
            "metadata_path": expected["metadata_path"],
            "metadata_sha256": expected["metadata_sha256"],
            "feature_order": expected_order,
            "sidecar": sidecar,
        })
    grid = {(row["reservoir_seed"], row["lightgbm_seed"]) for row in model_records}
    expected_grid = {(r, l) for r in RESERVOIR_SEEDS for l in LIGHTGBM_SEEDS}
    if grid != expected_grid or len(model_records) != 9:
        raise RuntimeError("Final D bundle does not contain the unique 3-by-3 seed grid")
    if reference_identity is None or reference_identity.get("best_params") != selected.get("best_params"):
        raise RuntimeError("Final D model parameters differ from the frozen selection checkpoint")
    if int(reference_identity.get("best_iteration", -1)) != int(selected["best_iteration"]):
        raise RuntimeError("Final D model iteration differs from the frozen selection checkpoint")
    return selected, {"memory": memory_names, "esn": esn_names}, {
        "models": model_records,
        "reference_identity": reference_identity,
        "feature_orders": feature_orders,
        "selection_dataset_path": selected["dataset_path"],
    }


def _read_development_sample(finalization: dict, selected: dict, transforms: dict):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds

    dataset_path = ROOT / selected["dataset_path"]
    dynamic_path = ROOT / finalization["dynamic_features_path"]
    lower = pa.scalar("2026-06-30 00:00:00", type=ds.dataset(dataset_path, format="parquet").schema.field("Opened").type)
    upper = pa.scalar("2026-06-30 23:58:00", type=ds.dataset(dataset_path, format="parquet").schema.field("Opened").type)
    base = list(selected["model_feature_order"])
    memory = transforms["memory"]
    esn = transforms["esn"]
    all_dynamic = [*memory, *(name for seed in RESERVOIR_SEEDS for name in esn[seed])]
    feature_table = ds.dataset(dataset_path, format="parquet").to_table(
        columns=["Opened", *base],
        filter=(ds.field("Opened") >= lower) & (ds.field("Opened") < upper),
    )
    dynamic_dataset = ds.dataset(dynamic_path, format="parquet")
    dynamic_type = dynamic_dataset.schema.field("Opened").type
    dynamic_lower = pa.scalar(pd.Timestamp("2026-06-30T00:00:00Z").to_pydatetime(), type=dynamic_type)
    dynamic_upper = pa.scalar(pd.Timestamp(SAFE_OPENED_END_UTC.replace("Z", "+00:00")).to_pydatetime(), type=dynamic_type)
    dynamic_table = dynamic_dataset.to_table(
        columns=["Opened", *all_dynamic],
        filter=(ds.field("Opened") >= dynamic_lower) & (ds.field("Opened") < dynamic_upper),
    )
    feature_frame = feature_table.to_pandas()
    dynamic_frame = dynamic_table.to_pandas()
    feature_frame["Opened"] = pd.to_datetime(feature_frame["Opened"], utc=True, errors="raise")
    dynamic_frame["Opened"] = pd.to_datetime(dynamic_frame["Opened"], utc=True, errors="raise")
    if feature_frame["Opened"].duplicated().any() or dynamic_frame["Opened"].duplicated().any():
        raise RuntimeError("Development parity sample contains duplicate Opened timestamps")
    merged = feature_frame.merge(dynamic_frame, on="Opened", how="inner", validate="one_to_one", sort=True)
    if len(merged) != len(feature_frame) or len(merged) != len(dynamic_frame):
        raise RuntimeError("Saved development features and causal transform states do not align")
    if (merged["Opened"] >= pd.Timestamp(SAFE_OPENED_END_UTC)).any():
        raise RuntimeError("Development parity sample crossed the safe raw-prefix boundary")
    if (merged["Opened"] + pd.Timedelta(minutes=2) >= pd.Timestamp(HOLDOUT_START_UTC)).any():
        raise RuntimeError("Development parity sample contains a protected holdout market start")
    columns = [*base, *all_dynamic]
    matrix = merged[columns].to_numpy(dtype=np.float64)
    complete = np.isfinite(matrix).all(axis=1)
    merged = merged.loc[complete].tail(int(finalization["parity"]["sample_rows"])).copy()
    if len(merged) != int(finalization["parity"]["sample_rows"]):
        raise RuntimeError("Saved development cache has too few complete rows for the frozen parity sample")
    opened = pd.DatetimeIndex(merged["Opened"])
    sample = {
        "frame": merged,
        "opened_first_utc": opened.min().isoformat(),
        "opened_last_utc": opened.max().isoformat(),
        "opened_sha256": _canonical_hash(opened.asi8.tolist()),
        "rows": len(merged),
        "feature_dataset_path": selected["dataset_path"],
        "dynamic_features_path": finalization["dynamic_features_path"],
        "labels_read": False,
        "holdout_rows_read": 0,
    }
    return sample


def _latest_training_label_release(finalization: dict, selected: dict) -> str:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds

    dataset = ds.dataset(ROOT / selected["dataset_path"], format="parquet")
    opened_type = dataset.schema.field("Opened").type
    opened_lower = pa.scalar("2026-06-30 23:40:00", type=opened_type)
    opened_upper = pa.scalar("2026-06-30 23:58:00", type=opened_type)
    table = dataset.to_table(
        columns=["Opened", "target_label_available_at"],
        filter=(ds.field("Opened") >= opened_lower) & (ds.field("Opened") < opened_upper),
    )
    frame = table.to_pandas()
    opened = pd.to_datetime(frame["Opened"], utc=True, errors="raise")
    available = pd.to_datetime(frame["target_label_available_at"], utc=True, errors="raise")
    eligible = opened.lt(pd.Timestamp(FIT_CUTOFF_UTC)) & available.lt(pd.Timestamp(FIT_CUTOFF_UTC))
    if not eligible.any():
        raise RuntimeError("Saved development timestamps do not include eligible final-fit label metadata")
    latest = available.loc[eligible].max().isoformat()
    if latest != finalization["latest_training_label_available_at_utc"]:
        raise RuntimeError("Latest final-fit label availability differs from the frozen development checkpoint")
    return latest


def _validate_calibrator(finalization: dict, preparation_protocol: dict) -> tuple[dict, dict]:
    calibration_record = _read_json(ROOT / finalization["calibrator_path"])
    expected_identity = calibration_record.copy()
    identity_hash = expected_identity.pop("calibrator_identity_sha256", None)
    if identity_hash != _canonical_hash(expected_identity):
        raise RuntimeError("Saved official Platt calibrator identity hash is invalid")
    if calibration_record.get("status") != "fitted":
        raise RuntimeError("The saved official D calibrator is not fitted")
    if calibration_record.get("label_source") != "official Gamma target_polymarket_up":
        raise RuntimeError("The saved calibrator does not use official Gamma outcomes")
    if calibration_record.get("as_of_utc") != FIT_CUTOFF_UTC.replace("Z", "+00:00"):
        raise RuntimeError("The saved official calibrator uses a different as-of cutoff")
    latest_release = calibration_record.get("latest_label_release_at_utc")
    if latest_release is None or latest_release >= FIT_CUTOFF_UTC.replace("Z", "+00:00"):
        raise RuntimeError("Official calibration labels are not strictly available before the first holdout decision")
    metadata = calibration_manifest_fields(
        calibration_record,
        calibration_record,
        int(preparation_protocol["calibration"]["minimum_rows"]),
    )
    if metadata["rows"] != int(finalization["calibration_rows"]):
        raise RuntimeError("Saved official Platt training-row count differs from finalization protocol")
    return calibration_record, metadata


def _validate_oof_history(finalization: dict) -> dict:
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(ROOT / finalization["oof_path"])
    if len(frame) != int(finalization["oof_rows"]):
        raise RuntimeError("Saved D calibration OOF row count differs from finalization protocol")
    required = {
        "market_start_utc", "ensemble_probability_raw", "prediction_provenance",
        "fit_cutoff_exclusive_utc", "latest_training_label_available_at_utc",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError("Saved D calibration OOF schema is incomplete")
    market = pd.to_datetime(frame["market_start_utc"], utc=True, errors="raise")
    fit_cutoff = pd.to_datetime(frame["fit_cutoff_exclusive_utc"], utc=True, errors="raise")
    latest_label = pd.to_datetime(frame["latest_training_label_available_at_utc"], utc=True, errors="raise")
    probability = pd.to_numeric(frame["ensemble_probability_raw"], errors="coerce").to_numpy(dtype=np.float64)
    selection_cutoff = pd.Timestamp("2025-10-01T00:00:00Z")
    holdout_start = pd.Timestamp(HOLDOUT_START_UTC)
    if market.duplicated().any() or not (market >= selection_cutoff).all() or not (market < holdout_start).all():
        raise RuntimeError("Saved D calibration OOF has duplicate or out-of-interval development markets")
    if not frame["prediction_provenance"].eq("fold_local_preselection_chronological_oof").all():
        raise RuntimeError("Saved D calibration OOF provenance differs from the frozen procedure")
    if not (fit_cutoff <= market).all() or not (latest_label < fit_cutoff).all():
        raise RuntimeError("Saved D calibration OOF violates its causal fit/prediction chronology")
    if not np.isfinite(probability).all() or ((probability < 0.0) | (probability > 1.0)).any():
        raise RuntimeError("Saved D calibration OOF contains invalid raw probabilities")
    return {
        "rows": int(len(frame)),
        "first_market_start_utc": market.min().isoformat(),
        "last_market_start_utc": market.max().isoformat(),
        "provenance": "fold_local_preselection_chronological_oof",
        "market_starts_unique": True,
        "holdout_rows": 0,
        "label_values_read": False,
    }


def _predict_sidecar_members(frame, model_records: list[dict]) -> dict[str, object]:
    import lightgbm as lgb
    import numpy as np

    results = {}
    for record in model_records:
        name = record["name"]
        features = list(record["sidecar"]["identity"]["feature_order"])
        if features != record["feature_order"]:
            raise RuntimeError(f"Validated model feature order changed: {name}")
        values = frame.loc[:, features].to_numpy(dtype=np.float32, copy=True)
        values[~np.isfinite(values)] = np.nan
        booster = lgb.Booster(model_file=str(ROOT / record["path"]))
        if (booster.num_feature() != len(features) or booster.feature_name() != features
                or booster.num_trees() != int(record["sidecar"]["num_trees"])):
            raise RuntimeError(f"Loaded final D member feature schema differs: {name}")
        results[name] = np.asarray(booster.predict(values), dtype=np.float64)
    return results


def _predict_bundle_members(frame, bundle: dict) -> dict[str, object]:
    import lightgbm as lgb
    import numpy as np

    results = {}
    base = list(bundle["base_feature_order"])
    memory = list(bundle["memory_feature_order"])
    esn = {int(seed): list(names) for seed, names in bundle["esn_feature_order_by_seed"].items()}
    for record in bundle["member_files"]:
        path = Path(record["path"])
        name = path.stem
        parts = name.split("_")
        if len(parts) != 3 or parts[0] != "D":
            raise RuntimeError(f"Invalid final D bundle member name: {name}")
        reservoir_seed = int(parts[1][1:])
        features = [*base, *memory, *esn[reservoir_seed]]
        values = frame.loc[:, features].to_numpy(dtype=np.float32, copy=True)
        values[~np.isfinite(values)] = np.nan
        booster = lgb.Booster(model_file=str(ROOT / record["path"]))
        if booster.num_feature() != len(features) or booster.feature_name() != features:
            raise RuntimeError(f"Bundle-read final D member feature schema differs: {name}")
        results[name] = np.asarray(booster.predict(values), dtype=np.float64)
    return results


def _apply_platt(probabilities, calibrator: dict):
    import numpy as np
    from scipy.special import expit

    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return expit(float(calibrator["intercept"]) + float(calibrator["coefficient"]) * logits)


def _build_bundle(finalization: dict, preparation_protocol: dict, preparation: dict,
                  selected: dict, transforms: dict, model_data: dict,
                  calibration: dict, calibration_metadata: dict,
                  parity: dict, source_manifest_hash: str) -> dict:
    model_records = model_data["models"]
    member_files = [{"path": item["path"], "sha256": item["sha256"]} for item in model_records]
    member_metadata_files = [
        {"path": item["metadata_path"], "sha256": item["metadata_sha256"]}
        for item in model_records
    ]
    identity = model_data["reference_identity"]
    member_identity_records = [
        {
            "member": item["name"],
            "model_path": item["path"],
            "model_sha256": item["sha256"],
            "feature_order_sha256": _canonical_hash(item["feature_order"]),
            "num_trees": int(item["sidecar"]["num_trees"]),
        }
        for item in model_records
    ]
    ensemble_identity = _canonical_hash({
        "procedure": "D",
        "fit_cutoff": FIT_CUTOFF_UTC.replace("Z", "+00:00"),
        "best_params": selected["best_params"],
        "best_iteration": int(selected["best_iteration"]),
        "members": member_identity_records,
    })
    return {
        "schema_version": 1,
        "experiment_id": "E0041",
        "run_identity_sha256": RUN_IDENTITY,
        "status": bundle_status_from_calibration(calibration_metadata),
        "procedure": "D = BASE70 + memory PCA8 + ESN PCA8",
        "selection_cutoff_exclusive_utc": "2025-10-01T00:00:00Z",
        "fit_cutoff_exclusive_utc": FIT_CUTOFF_UTC,
        "training_rows": int(finalization["training_rows"]),
        "latest_training_label_available_at_utc": finalization["latest_training_label_available_at_utc"],
        "best_params": selected["best_params"],
        "decision_weight": selected["decision_weight"],
        "best_iteration": int(selected["best_iteration"]),
        "final_fit": {
            "training_rows": int(finalization["training_rows"]),
            "fit_cutoff_exclusive_utc": FIT_CUTOFF_UTC,
            "latest_training_label_available_at_utc": finalization["latest_training_label_available_at_utc"],
            "training_row_opened_sha256": identity["training_row_opened_sha256"],
            "training_labels_sha256": identity["training_labels_sha256"],
            "training_weights_sha256": identity["training_weights_sha256"],
            "training_feature_matrix_sha256_by_reservoir": finalization["training_feature_matrix_sha256_by_reservoir"],
            "identity_source": "common and per-reservoir fields reconstructed from all nine unchanged model sidecars",
            "refit_performed_during_finalization": False,
        },
        "base_feature_order": list(selected["model_feature_order"]),
        "memory_feature_order": transforms["memory"],
        "esn_feature_order_by_seed": {str(seed): transforms["esn"][seed] for seed in RESERVOIR_SEEDS},
        "lightgbm_seeds": list(LIGHTGBM_SEEDS),
        "reservoir_seeds": list(RESERVOIR_SEEDS),
        "ensemble_identity_sha256": ensemble_identity,
        "member_files": member_files,
        "member_metadata_files": member_metadata_files,
        "calibration": {
            "status": calibration_metadata["status"],
            "rows": calibration_metadata["rows"],
            "required_rows": calibration_metadata["required_rows"],
            "calibrator": calibration,
            "calibrator_path": finalization["calibrator_path"],
            "calibrator_sha256": finalization["calibrator_sha256"],
            "history_path": finalization["oof_path"],
            "history_sha256": finalization["oof_sha256"],
            "official_gamma_calibration_rows": calibration_metadata["rows"],
            "official_gamma_development_rows": "NOT_VERIFIED_IN_THIS_AUDIT",
            "fallback": None,
            "frozen_through_holdout": True,
        },
        "source_hashes": {
            "preparation_manifest_sha256": source_manifest_hash,
            "parent_raw_prefix_registered_sha256": preparation_protocol["development_inputs"]["raw_prefix_sha256"],
            "safe_raw_prefix_sha256": preparation["safe_development_raw_prefix"]["sha256"],
            "flow_prefix_sha256": preparation["flow_prefix_sha256"],
            "memory_transform_sha256": finalization["memory_transform_sha256"],
            "esn_transform_sha256": finalization["esn_transform_sha256"],
            "parent_v7_protocol_sha256": preparation_protocol["parent_e0039"]["protocol_sha256"],
            "development_feature_dataset_sha256": finalization["development_feature_dataset_sha256"],
            "dynamic_features_sha256": finalization["dynamic_features_sha256"],
        },
        "lineage": {
            "training": {
                "preparation_protocol_path": PREPARATION_PROTOCOL_PATH.relative_to(ROOT).as_posix(),
                "preparation_protocol_sha256": finalization["preparation_protocol_sha256"],
                "historical_preparation_runner_sha256": finalization["code_hashes"]["historical_preparation_runner_sha256"],
                "source_run_identity_sha256": RUN_IDENTITY,
                "source_preparation_manifest_sha256": source_manifest_hash,
            },
            "packaging": {
                "finalization_protocol_path": FINALIZATION_PROTOCOL_PATH.relative_to(ROOT).as_posix(),
                "finalization_protocol_sha256": _sha256(FINALIZATION_PROTOCOL_PATH),
                "finalizer_sha256": finalization["code_hashes"]["finalizer_sha256"],
                "packaging_preparation_runner_sha256": finalization["code_hashes"]["packaging_preparation_runner_sha256"],
                "method": "read-only verification and packaging; no training or calibration fit",
            },
        },
        "development_state": {
            "dynamic_features_path": finalization["dynamic_features_path"],
            "dynamic_features_sha256": finalization["dynamic_features_sha256"],
            "feature_dataset_path": selected["dataset_path"],
            "feature_dataset_signature": selected["dataset_signature"],
            "last_materialized_market_start_utc": finalization["last_development_market_start_utc"],
            "holdout_rows_materialized": 0,
            "rebuild_from_development_prefix_only": True,
        },
        "save_load_parity": parity,
        "activated": False,
        "holdout_rows_read": 0,
        "holdout_predictions_generated": 0,
        "holdout_status": "HOLDOUT_NOT_RUN",
        "evaluation_authorized": False,
    }


def finalize() -> dict:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import run_btc_D_holdout_evaluation_20261028 as evaluation
    import run_btc_chronology_reconstruction_20261027 as chronology

    finalization_config = _read_json(FINALIZATION_PROTOCOL_PATH)
    resource_snapshot = {
        "available_ram_bytes": _available_ram_bytes(),
        "free_workspace_disk_bytes": int(shutil.disk_usage(RUN_ROOT).free),
    }
    if resource_snapshot["available_ram_bytes"] < int(finalization_config["resource_checks"]["minimum_available_ram_bytes"]):
        raise RuntimeError("Insufficient available RAM for read-only nine-model parity verification")
    if resource_snapshot["free_workspace_disk_bytes"] < int(finalization_config["resource_checks"]["minimum_free_output_disk_bytes"]):
        raise RuntimeError("Insufficient free workspace disk for small atomic finalization outputs")

    finalization, preparation_protocol, preparation, _ = _verify_protocol_and_sources()
    source_manifest_hash = finalization["source_preparation_manifest_sha256"]
    selected, transforms, model_data = _validate_selection_and_sidecars(finalization, preparation_protocol)
    calibration, calibration_metadata = _validate_calibrator(finalization, preparation_protocol)
    oof_evidence = _validate_oof_history(finalization)

    dataset_path = ROOT / selected["dataset_path"]
    row_count = pq.ParquetFile(dataset_path).metadata.num_rows
    if row_count != int(finalization["development_feature_rows"]):
        raise RuntimeError("Saved development feature dataset row count differs from its stage manifest")
    sample = _read_development_sample(finalization, selected, transforms)
    frame = sample["frame"]
    latest_training_label = _latest_training_label_release(finalization, selected)
    if row_count - 6 != int(finalization["training_rows"]):
        raise RuntimeError("Final D training-row count is inconsistent with the six post-cutoff development rows")

    member_records = model_data["models"]
    bundle = _build_bundle(
        finalization, preparation_protocol, preparation, selected, transforms, model_data,
        calibration, calibration_metadata, {}, source_manifest_hash,
    )
    bundle["calibration"]["oof_validation"] = oof_evidence
    direct_predictions = _predict_sidecar_members(frame, member_records)
    bundle_member_predictions = _predict_bundle_members(frame, bundle)
    if set(direct_predictions) != set(bundle_member_predictions) or len(direct_predictions) != 9:
        raise RuntimeError("Independent and bundle model reads did not load all nine D members")
    member_differences = {
        name: float(np.max(np.abs(direct_predictions[name] - bundle_member_predictions[name])))
        for name in direct_predictions
    }
    max_member_difference = max(member_differences.values())
    if max_member_difference > TOLERANCE:
        raise RuntimeError("Direct sidecar and bundle member predictions differ")
    direct_mean = np.mean([direct_predictions[item["name"]] for item in member_records], axis=0)
    bundle_member_mean = np.mean([bundle_member_predictions[item["name"]] for item in member_records], axis=0)
    evaluator_mean = evaluation._predict_D(frame, bundle)
    max_mean_difference = max(
        float(np.max(np.abs(direct_mean - bundle_member_mean))),
        float(np.max(np.abs(direct_mean - evaluator_mean))),
    )
    if max_mean_difference > TOLERANCE:
        raise RuntimeError("Uniform nine-member mean differs between sidecars and the saved bundle reader")
    direct_calibrated = _apply_platt(direct_mean, calibration)
    bundle_calibrated = chronology.apply_official_platt(
        evaluator_mean, bundle["calibration"]["calibrator"],
    )
    max_calibrated_difference = float(np.max(np.abs(direct_calibrated - bundle_calibrated)))
    if max_calibrated_difference > TOLERANCE:
        raise RuntimeError("Official Platt probability differs between sidecar and bundle metadata")

    parity = {
        "sample_rows": int(sample["rows"]),
        "tolerance_absolute": TOLERANCE,
        "max_abs_member_prediction_difference": max_member_difference,
        "max_abs_uniform_ensemble_difference": max_mean_difference,
        "max_abs_calibrated_probability_difference": max_calibrated_difference,
        "ensemble_weights": "uniform 1/9",
        "member_count_verified": 9,
        "feature_order_verified_per_member": True,
        "sample_provenance": {
            "feature_dataset_path": sample["feature_dataset_path"],
            "feature_dataset_sha256": finalization["development_feature_dataset_sha256"],
            "dynamic_features_path": sample["dynamic_features_path"],
            "dynamic_features_sha256": finalization["dynamic_features_sha256"],
            "rule": "last 64 chronological complete development rows from Opened 2026-06-30 00:00 through 23:57 UTC",
            "opened_first_utc": sample["opened_first_utc"],
            "opened_last_utc": sample["opened_last_utc"],
            "opened_sha256": sample["opened_sha256"],
            "labels_read": False,
            "holdout_rows_read": 0,
        },
        "predictions_persisted": False,
        "predictions_scored": False,
    }
    bundle["save_load_parity"] = parity
    output_manifest_sha256 = hashlib.sha256(_json_bytes(bundle)).hexdigest()
    result = {
        "schema_version": 1,
        "experiment_id": "E0041",
        "run_identity_sha256": RUN_IDENTITY,
        "status": bundle["status"],
        "bundle_manifest_path": BUNDLE_PATH.relative_to(ROOT).as_posix(),
        "bundle_manifest_sha256": output_manifest_sha256,
        "finalization_protocol_sha256": _sha256(FINALIZATION_PROTOCOL_PATH),
        "source_preparation_manifest_sha256": source_manifest_hash,
        "model_count": len(member_records),
        "feature_count_per_model": 86,
        "oof_rows": int(finalization["oof_rows"]),
        "calibration_rows": calibration_metadata["rows"],
        "latest_training_label_available_at_utc": latest_training_label,
        "latest_calibration_label_release_at_utc": calibration["latest_label_release_at_utc"],
        "oof_validation": oof_evidence,
        "save_load_parity": parity,
        "evaluation_authorized": False,
        "activated": False,
        "holdout_status": "HOLDOUT_NOT_RUN",
        "holdout_rows_read": 0,
        "holdout_predictions_generated": 0,
        "resource_checks": {
            "passed": True,
            "minimum_available_ram_bytes": int(finalization["resource_checks"]["minimum_available_ram_bytes"]),
            "minimum_free_output_disk_bytes": int(finalization["resource_checks"]["minimum_free_output_disk_bytes"]),
        },
    }

    _preflight_json_output(BUNDLE_PATH, bundle)
    _preflight_json_output(BUNDLE_POINTER_PATH, bundle)
    _preflight_json_output(FINALIZATION_RESULT_PATH, result)
    _write_json_atomic(BUNDLE_PATH, bundle)
    _write_json_atomic(BUNDLE_POINTER_PATH, bundle)
    _write_json_atomic(FINALIZATION_RESULT_PATH, result)

    if preparation.get("status") != "D_BUNDLE_READY":
        preparation["status"] = bundle["status"]
        preparation["bundle_manifest_path"] = BUNDLE_PATH.relative_to(ROOT).as_posix()
        preparation["bundle_manifest_sha256"] = output_manifest_sha256
        preparation["activated"] = False
        preparation["holdout_status"] = "HOLDOUT_NOT_RUN"
        preparation["finalization"] = {
            "status": bundle["status"],
            "source_preparation_manifest_sha256": source_manifest_hash,
            "finalization_protocol_sha256": result["finalization_protocol_sha256"],
            "bundle_manifest_path": result["bundle_manifest_path"],
            "bundle_manifest_sha256": output_manifest_sha256,
            "finalization_result_path": FINALIZATION_RESULT_PATH.relative_to(ROOT).as_posix(),
            "save_load_parity": parity,
            "evaluation_authorized": False,
            "holdout_rows_read": 0,
            "holdout_predictions_generated": 0,
            "holdout_status": "HOLDOUT_NOT_RUN",
            "activated": False,
        }
        _write_json_atomic(PREPARATION_MANIFEST_PATH, preparation)
    else:
        saved = preparation.get("finalization", {})
        if saved.get("bundle_manifest_sha256") != output_manifest_sha256 or saved.get("source_preparation_manifest_sha256") != source_manifest_hash:
            raise RuntimeError("Existing preparation manifest records a different finalization")
    return result


def main() -> None:
    print(json.dumps(finalize(), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
