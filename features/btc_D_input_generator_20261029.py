"""Development-only reconstruction helpers for the frozen E0041 D bundle.

This module deliberately exposes no holdout loader and does not import targets,
outcomes, or the holdout evaluation path.  BASE64 reconstruction is kept
separate because the project currently has no stateful chunk interface for the
frozen E0041 RP/VP feature stack.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import time
import zipfile
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/research/btc_D_input_generator_final_bundle_fix_20261029_v1.json"
RUNTIME_DIR = ROOT / "data/analysis/polymarket/BTC/btc_D_input_generator_final_bundle_fix_20261029"
CHECKPOINT_PATH = RUNTIME_DIR / "checkpoint_state.json"
RAW_COLUMNS = (
    "Opened", "Open", "High", "Low", "Close", "Volume",
    "UM_BTCUSDT_Open", "UM_BTCUSDT_High", "UM_BTCUSDT_Low", "UM_BTCUSDT_Close",
)
FLOW_RAW_COLUMNS = (
    "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume",
)
FLOW_COLUMNS = tuple(
    name for horizon in (1, 5, 15)
    for name in (f"taker_Z_{horizon}m", f"taker_V_{horizon}m")
)
MINUTE_NS = 60_000_000_000
FLOW_TAIL_MINUTES = 10_094


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _replace_atomic(source: Path, target: Path) -> None:
    """Retry transient Windows sharing/access denials without weakening atomicity."""
    for attempt in range(16):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 15:
                raise
            time.sleep(0.05 * (attempt + 1))


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_REPLAY" or protocol.get("experiment_id") != "E0045":
        raise RuntimeError("E0045 protocol status or identity is not frozen")
    scope = protocol["scope"]
    if scope["holdout_values_rows_predictions_labels_books_and_economics_authorized"]:
        raise RuntimeError("E0045 unexpectedly authorizes protected data")
    if scope["evaluation_authorized"] or scope["activated"]:
        raise RuntimeError("E0045 must remain development-only and inactive")
    return protocol


def _required_file_hash(path_value: str, expected_sha256: str, label: str) -> Path:
    path = ROOT / Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"frozen {label} hash mismatch: {path}")
    return path


def _feature_order_sha256(order: list[str]) -> str:
    return hashlib.sha256(json.dumps(order, separators=(",", ":")).encode("utf-8")).hexdigest()


def verify_bundle(protocol: dict) -> tuple[dict, dict, dict]:
    """Load only the final E0041 D manifest, its sidecars, transforms and calibrator."""
    parent = protocol["parent"]
    forbidden = {"legacy_d_manifest_path", "legacy_bundle_directory", "candidate_holdout_manifest_path"}
    if forbidden.intersection(protocol.get("allowed_inputs", {})):
        raise RuntimeError("E0045 protocol contains a legacy D specification pointer")
    for relative_path, expected_sha256 in protocol["allowed_inputs"]["feature_implementation_hashes"].items():
        _required_file_hash(relative_path, expected_sha256, f"E0045 feature implementation {relative_path}")
    manifest_path = _required_file_hash(
        parent["bundle_manifest_path"], parent["bundle_manifest_sha256"], "final E0041 D bundle manifest",
    )
    finalization_path = _required_file_hash(
        parent["finalization_manifest_path"], parent["finalization_manifest_sha256"], "finalization bundle manifest",
    )
    if manifest_path.read_bytes() != finalization_path.read_bytes():
        raise RuntimeError("final E0041 bundle pointer differs from the finalization manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "E0041" or manifest.get("status") != "D_BUNDLE_READY":
        raise RuntimeError("final bundle manifest is not the frozen E0041 D bundle")
    if manifest.get("evaluation_authorized") is not False or manifest.get("activated") is not False:
        raise RuntimeError("final E0041 bundle must remain unauthorized and inactive")
    if manifest.get("holdout_status") != "HOLDOUT_NOT_RUN" or manifest.get("holdout_rows_read") != 0 or manifest.get("holdout_predictions_generated") != 0:
        raise RuntimeError("final E0041 manifest holdout status changed")

    base_order = list(manifest["base_feature_order"])
    memory_order = list(manifest["memory_feature_order"])
    reservoir_seeds = [int(seed) for seed in manifest["reservoir_seeds"]]
    lightgbm_seeds = [int(seed) for seed in manifest["lightgbm_seeds"]]
    esn_order_by_seed = {str(seed): list(names) for seed, names in manifest["esn_feature_order_by_seed"].items()}
    if len(base_order) != 70 or base_order[-6:] != list(FLOW_COLUMNS) or len(set(base_order)) != 70:
        raise RuntimeError("final E0041 BASE70 order differs from its recorded 64+FLOW6 order")
    if len(memory_order) != 8 or list(esn_order_by_seed) != [str(seed) for seed in reservoir_seeds]:
        raise RuntimeError("final E0041 dynamic feature order or reservoir seed order is invalid")
    if any(len(esn_order_by_seed[str(seed)]) != 8 for seed in reservoir_seeds):
        raise RuntimeError("final E0041 ESN feature width differs from the manifest")

    transform_sources = protocol["allowed_inputs"]["frozen_transforms"]
    memory_path = _required_file_hash(
        transform_sources["memory"]["path"], transform_sources["memory"]["sha256"], "final E0041 memory transform",
    )
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    if manifest["source_hashes"]["memory_transform_sha256"] != transform_sources["memory"]["sha256"]:
        raise RuntimeError("final E0041 manifest does not identify the supplied memory transform")
    memory_width = len(memory["feature_names"])
    if (len(memory["identity"]["input_columns"]) != 14
            or len(memory["pca_mean"]) != memory_width
            or len(memory["pca_components"]) != 8
            or any(len(component) != memory_width for component in memory["pca_components"])):
        raise RuntimeError("final E0041 memory transform input/projection schema is invalid")
    memory_input_order_hash = hashlib.sha256(
        json.dumps(memory["identity"]["input_columns"], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if memory_input_order_hash != memory["identity"]["input_column_order_sha256"]:
        raise RuntimeError("final E0041 memory transform input order hash is invalid")
    expected_memory_order = [f"simple_memory_PCA8_pc{i:02d}" for i in range(8)]
    if memory_order != expected_memory_order:
        raise RuntimeError("final E0041 memory PCA output order differs from the bundle")
    esn = {}
    for seed in reservoir_seeds:
        source = transform_sources["esn"][str(seed)]
        transform_path = _required_file_hash(source["path"], source["sha256"], f"E0041 ESN transform {seed}")
        payload = json.loads(transform_path.read_text(encoding="utf-8"))
        if int(payload["seed"]) != seed or list(payload["feature_names"]) != esn_order_by_seed[str(seed)]:
            raise RuntimeError(f"E0041 ESN transform order or seed differs for {seed}")
        if manifest["source_hashes"]["esn_transform_sha256"] != protocol["allowed_inputs"]["frozen_transforms"]["combined_esn_sha256"]:
            raise RuntimeError("final E0041 manifest does not identify the supplied ESN transforms")
        esn[seed] = payload

    expected_grid = [(reservoir_seed, model_seed) for reservoir_seed in reservoir_seeds for model_seed in lightgbm_seeds]
    if len(manifest["member_files"]) != 9 or len(manifest["member_metadata_files"]) != 9:
        raise RuntimeError("final E0041 D bundle does not contain nine model/sidecar pairs")
    members = []
    for index, (model_record, sidecar_record) in enumerate(zip(manifest["member_files"], manifest["member_metadata_files"])):
        model_path = _required_file_hash(model_record["path"], model_record["sha256"], "final E0041 LightGBM member")
        sidecar_path = _required_file_hash(sidecar_record["path"], sidecar_record["sha256"], "final E0041 LightGBM sidecar")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        identity_doc = sidecar["identity"]
        seed_pair = (int(identity_doc["reservoir_seed"]), int(identity_doc["lightgbm_seed"]))
        if seed_pair != expected_grid[index]:
            raise RuntimeError("final E0041 member ordering or seed grid differs from the bundle manifest")
        expected_order = [*base_order, *memory_order, *esn_order_by_seed[str(seed_pair[0])]]
        if identity_doc["feature_order"] != expected_order:
            raise RuntimeError(f"final E0041 member feature names/order differ: {model_path.name}")
        order_hash = _feature_order_sha256(expected_order)
        if identity_doc["feature_order_sha256"] != order_hash or int(sidecar["num_features"]) != len(expected_order):
            raise RuntimeError(f"final E0041 member feature schema hash or width differs: {model_path.name}")
        if sidecar["model_sha256"] != model_record["sha256"]:
            raise RuntimeError(f"final E0041 member sidecar names another model: {model_path.name}")
        if sidecar.get("identity_sha256") != _canonical_sha256(identity_doc):
            raise RuntimeError(f"final E0041 member sidecar identity hash mismatch: {model_path.name}")
        if int(identity_doc["best_iteration"]) != int(manifest["best_iteration"]):
            raise RuntimeError(f"final E0041 member iteration metadata differs: {model_path.name}")
        if int(sidecar["num_trees"]) != int(identity_doc["best_iteration"]):
            raise RuntimeError(f"final E0041 sidecar tree count differs from its saved best iteration: {model_path.name}")
        members.append({
            "reservoir_seed": seed_pair[0],
            "lightgbm_seed": seed_pair[1],
            "model_path": model_path,
            "model_sha256": model_record["sha256"],
            "metadata_path": sidecar_path,
            "metadata_sha256": sidecar_record["sha256"],
            "num_features": int(sidecar["num_features"]),
            "num_trees": int(sidecar["num_trees"]),
            "feature_order": expected_order,
            "feature_order_sha256": order_hash,
            "sidecar": sidecar,
        })

    calibration = manifest["calibration"]
    calibrator_path = _required_file_hash(
        calibration["calibrator_path"], calibration["calibrator_sha256"], "final E0041 Platt calibrator",
    )
    if calibration["calibrator_sha256"] != protocol["allowed_inputs"]["frozen_calibrator"]["sha256"]:
        raise RuntimeError("final E0041 calibrator hash differs from the protocol")
    calibrator_doc = json.loads(calibrator_path.read_text(encoding="utf-8"))
    frozen_coefficients = calibration["calibrator"]
    if (float(calibrator_doc["intercept"]) != float(frozen_coefficients["intercept"])
            or float(calibrator_doc["coefficient"]) != float(frozen_coefficients["coefficient"])
            or calibrator_doc["calibrator_identity_sha256"] != frozen_coefficients["calibrator_identity_sha256"]):
        raise RuntimeError("final E0041 Platt coefficients or identity differ from the bundle manifest")
    if manifest["save_load_parity"]["ensemble_weights"] != "uniform 1/9":
        raise RuntimeError("final E0041 ensemble aggregation is not the frozen uniform mean")

    identity = {
        "bundle_manifest_sha256": parent["bundle_manifest_sha256"],
        "run_identity_sha256": manifest["run_identity_sha256"],
        "ensemble_identity_sha256": manifest["ensemble_identity_sha256"],
        "members": [{"seed": [m["reservoir_seed"], m["lightgbm_seed"]], "model_sha256": m["model_sha256"], "metadata_sha256": m["metadata_sha256"]} for m in members],
        "memory_transform_sha256": transform_sources["memory"]["sha256"],
        "esn_transform_sha256_by_seed": {str(seed): transform_sources["esn"][str(seed)]["sha256"] for seed in reservoir_seeds},
        "calibrator_sha256": calibration["calibrator_sha256"],
    }
    d_manifest = dict(manifest)
    d_manifest["members"] = members
    d_manifest["calibrator_doc"] = calibrator_doc
    return identity, d_manifest, {"memory": memory, "esn": esn}


def _verify_pipeline_output(pipeline_manifest: dict, path: str, sha256: str, label: str) -> None:
    matches = [
        output
        for stage in pipeline_manifest["stages"].values()
        for output in stage.get("outputs", [])
        if output.get("path") == path
    ]
    if len(matches) != 1 or matches[0].get("sha256") != sha256:
        raise RuntimeError(f"E0041 pipeline lineage does not verify {label}: {path}")


def _same_state_value(left, right) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)
        except (TypeError, ValueError):
            return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_state_value(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same_state_value(a, b) for a, b in zip(left, right))
    if isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)):
        return (np.isnan(left) and np.isnan(right)) or float(left) == float(right)
    try:
        return left == right
    except (TypeError, ValueError):
        return False


def _validate_base64_output(path: Path, base_order: list[str], rows: int,
                            opened_cutoff: str, expected_dtypes: dict[str, str]) -> dict:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    names = schema.names
    missing = [name for name in base_order if name not in names]
    if missing:
        raise RuntimeError(f"raw BASE64 output is missing exact final E0041 features: {missing}")
    selected_in_schema = [name for name in names if name in set(base_order)]
    if selected_in_schema != base_order:
        raise RuntimeError("raw BASE64 output selected feature order differs from final E0041")
    target_columns = [name for name in names if name.lower().startswith("target_") or name.lower() in {"label", "outcome", "payout"}]
    if target_columns:
        raise RuntimeError(f"feature-only raw BASE64 output unexpectedly contains target/outcome columns: {target_columns}")
    if parquet.metadata.num_rows != rows:
        raise RuntimeError("raw BASE64 output row count differs from the safe development raw prefix")
    dtypes = {name: str(schema.field(name).type) for name in base_order}
    if dtypes != expected_dtypes:
        raise RuntimeError("raw BASE64 output dtypes differ from the canonical final E0041 feature schema")
    reader = _ParquetRowReader(path, ["Opened"], opened_cutoff)
    if reader.total_rows != rows:
        raise RuntimeError("raw BASE64 Opened row count differs from the safe development raw prefix")
    first_opened = last_opened = None
    while reader.row < reader.total_rows:
        opened = validate_opened(reader.read(65536)["Opened"], opened_cutoff)
        first_opened = first_opened or opened[0].isoformat()
        last_opened = opened[-1].isoformat()
    return {
        "rows": int(parquet.metadata.num_rows),
        "feature_order": list(base_order),
        "feature_dtypes": dtypes,
        "first_opened_utc": first_opened,
        "last_opened_utc": last_opened,
        "opened_cutoff_exclusive_utc": opened_cutoff,
    }


def rebuild_base64_raw(protocol: dict, bundle_identity: dict, d_manifest: dict) -> dict:
    """Apply the saved E0041 feature pipeline to the safe raw prefix without targets."""
    import pyarrow.parquet as pq
    import create_modeling_dataset as dataset_builder
    from utils.data import load_modeling_dataset_settings, resolve_modeling_dataset_output_paths

    base_record = protocol["allowed_inputs"]["safe_index_ohlcv_prefix"]
    raw_path = _required_file_hash(base_record["path"], base_record["sha256"], "safe development raw prefix")
    if int(base_record["rows"]) <= 0:
        raise RuntimeError("safe E0041 development raw prefix is empty")

    lineage = protocol["allowed_inputs"]["base_feature_lineage"]
    pipeline_path = _required_file_hash(lineage["selection_pipeline_manifest"]["path"], lineage["selection_pipeline_manifest"]["sha256"], "E0041 selection pipeline manifest")
    pipeline_manifest = json.loads(pipeline_path.read_text(encoding="utf-8"))
    selection_path = _required_file_hash(lineage["selection_result"]["path"], lineage["selection_result"]["sha256"], "E0041 selection result")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["payload"]
    base_order = list(d_manifest["base_feature_order"][:64])
    if selection["model_feature_order"] != d_manifest["base_feature_order"]:
        raise RuntimeError("E0041 selection result order differs from the final D bundle")
    if list(selection["selected_base_features"]) != base_order:
        raise RuntimeError("E0041 selected BASE64 order differs from the final D bundle")
    reference = protocol["allowed_inputs"]["canonical_model_feature_reference"]
    reference_schema = pq.ParquetFile(ROOT / reference["path"]).schema_arrow
    expected_dtypes = {name: str(reference_schema.field(name).type) for name in base_order}
    if (selection["dataset_path"] != reference["path"]
            or selection["dataset_signature"] != d_manifest["development_state"]["feature_dataset_signature"]):
        raise RuntimeError("E0041 selected feature dataset lineage differs from the final bundle")
    _verify_pipeline_output(pipeline_manifest, reference["path"], reference["sha256"], "causal model feature dataset")

    raw_feature_data = lineage["final_feature_dataset"]
    _required_file_hash(raw_feature_data["path"], raw_feature_data["sha256"], "E0041 pre-flow feature dataset")
    _verify_pipeline_output(pipeline_manifest, raw_feature_data["path"], raw_feature_data["sha256"], "pre-flow feature dataset")
    metadata_path = _required_file_hash(lineage["feature_metadata"]["path"], lineage["feature_metadata"]["sha256"], "E0041 feature-generation metadata")
    _verify_pipeline_output(pipeline_manifest, lineage["feature_metadata"]["path"], lineage["feature_metadata"]["sha256"], "feature-generation metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    effective_path = _required_file_hash(
        lineage["effective_config"]["path"], lineage["effective_config"]["sha256"], "E0041 effective feature configuration",
    )
    effective_config = json.loads(effective_path.read_text(encoding="utf-8"))
    effective_profile = effective_config["profile"]
    if (effective_profile.get("development_input_path") != base_record["path"]
            or effective_profile.get("raw_data_path") != base_record["path"]
            or effective_profile.get("source_raw_data_path") != lineage["source_raw_path_identity_only"]):
        raise RuntimeError("E0041 effective raw-input path identities differ from the frozen E0045 protocol")

    indicator_stage_path = _required_file_hash(lineage["indicator_stage"]["path"], lineage["indicator_stage"]["sha256"], "E0041 indicator stage manifest")
    _verify_pipeline_output(pipeline_manifest, lineage["indicator_stage"]["path"], lineage["indicator_stage"]["sha256"], "indicator stage manifest")
    indicator_stage = json.loads(indicator_stage_path.read_text(encoding="utf-8"))
    indicator_source_manifest_path = _required_file_hash(
        lineage["indicator_source_run_manifest"]["path"],
        lineage["indicator_source_run_manifest"]["sha256"],
        "E0041 indicator source run manifest",
    )
    indicator_source_manifest = json.loads(indicator_source_manifest_path.read_text(encoding="utf-8"))
    _verify_pipeline_output(
        indicator_source_manifest,
        lineage["indicator_stage"]["path"],
        lineage["indicator_stage"]["sha256"],
        "saved E0041 indicator stage",
    )
    for record in lineage["indicator_fit_results"]:
        _required_file_hash(record["path"], record["sha256"], "E0041 saved indicator fit result")
        _verify_pipeline_output(
            indicator_source_manifest, record["path"], record["sha256"], "saved E0041 indicator fit result",
        )
    for record_name in ("reaction_profile_config", "volume_profile_config"):
        _required_file_hash(lineage[record_name]["path"], lineage[record_name]["sha256"], f"E0041 {record_name}")
        _verify_pipeline_output(
            pipeline_manifest, lineage[record_name]["path"], lineage[record_name]["sha256"], f"E0041 {record_name}",
        )
    for record_name in ("reaction_profile_state", "volume_profile_state"):
        _required_file_hash(lineage[record_name]["npz_path"], lineage[record_name]["npz_sha256"], f"E0041 {record_name} data")
        _required_file_hash(lineage[record_name]["json_path"], lineage[record_name]["json_sha256"], f"E0041 {record_name} metadata")
        _verify_pipeline_output(
            pipeline_manifest, lineage[record_name]["npz_path"], lineage[record_name]["npz_sha256"], f"E0041 {record_name} data",
        )

    expected_fit_names = {name for name in base_order if "_fit_" in name}
    indicator_configs = dataset_builder.parse_fit_results(ROOT / indicator_stage["fit_results_dir"])
    fit_by_name = {record["feature_col"]: record for record in indicator_configs}
    if not expected_fit_names.issubset(fit_by_name):
        raise RuntimeError(f"saved E0041 indicator fit parameters are missing: {sorted(expected_fit_names - set(fit_by_name))}")
    if {record["path"] for record in lineage["indicator_fit_results"]} != {
            fit_by_name[name]["json_path"].as_posix() for name in expected_fit_names}:
        observed = {str(fit_by_name[name]["json_path"]).replace("\\", "/") for name in expected_fit_names}
        expected = {record["path"].replace("\\", "/") for record in lineage["indicator_fit_results"]}
        if observed != expected:
            raise RuntimeError("E0041 indicator feature names resolve to different saved fit-result files")
    selected_indicator_configs = [fit_by_name[name] for name in base_order if name in fit_by_name]
    if {record["feature_col"] for record in selected_indicator_configs} != expected_fit_names:
        raise RuntimeError("E0041 selected indicator configuration set is incomplete")

    for family, config_key in (("reaction_profile", "reaction_profile_fixed_grid"), ("volume_profile", "volume_profile_fixed_range")):
        saved_fit = json.loads((ROOT / lineage[f"{family}_config"]["path"]).read_text(encoding="utf-8"))
        if saved_fit["config_signature"] != metadata[config_key]["config_signature"]:
            raise RuntimeError(f"E0041 saved {family} fit config differs from the feature dataset metadata")

    runtime = RUNTIME_DIR
    runtime.mkdir(parents=True, exist_ok=True)
    subset_path = runtime / "selected_base64_features.json"
    _write_small_json(subset_path, {"features": base_order})
    settings = load_modeling_dataset_settings(asset="BTC", dataset_profile_name="BTC", modeling_profile_name="BTC")
    builder_config_hashes = protocol["allowed_inputs"]["builder_configuration_hashes"]
    for config_path, expected_sha in builder_config_hashes.items():
        if sha256_file(ROOT / config_path) != expected_sha:
            raise RuntimeError(f"E0045 modeling feature-builder config changed: {config_path}")
    settings.update({
        "active_asset": "BTC",
        "symbol": "BTCUSD",
        "interval": "1m",
        "raw_data_dir": raw_path.parent,
        "data_dir": raw_path.parent,
        "base_data_file": raw_path.name,
        "fit_results_dir": ROOT / indicator_stage["fit_results_dir"],
        "feature_subset_path": subset_path,
        "feature_subset_list_key": "features",
        "excluded_feature_names": (),
        "feature_intervals": metadata["feature_intervals"],
        "basis_premium_features": metadata["basis_premium_features"],
        "reaction_profile_fixed_grid": metadata["reaction_profile_fixed_grid"],
        "volume_profile_fixed_range": metadata["volume_profile_fixed_range"],
        "float_precision": "float64",
        "profile_state_dir": runtime / "profile_states",
        "modeling_output_dir": runtime / "base_feature_build",
        "output_suffix": "e0045_base64_raw",
        "preview_rows": 4,
        "preopen_target": {
            "development_cutoff_exclusive_utc": protocol["scope"]["source_raw_opened_exclusive_utc"],
            "development_input_path": str(raw_path),
            "source_raw_data_path": lineage["source_raw_path_identity_only"],
        },
    })
    output_path = resolve_modeling_dataset_output_paths(settings)["parquet"]
    stage_identity = _canonical_sha256({
        "bundle_identity": bundle_identity,
        "raw_prefix_sha256": base_record["sha256"],
        "selection_pipeline_sha256": lineage["selection_pipeline_manifest"]["sha256"],
        "selection_result_sha256": lineage["selection_result"]["sha256"],
        "feature_metadata_sha256": lineage["feature_metadata"]["sha256"],
        "indicator_stage_sha256": lineage["indicator_stage"]["sha256"],
        "indicator_fit_results": lineage["indicator_fit_results"],
        "reaction_profile_config_sha256": lineage["reaction_profile_config"]["sha256"],
        "volume_profile_config_sha256": lineage["volume_profile_config"]["sha256"],
        "profile_state_refs": {
            key: lineage[key] for key in ("reaction_profile_state", "volume_profile_state")
        },
        "feature_order": base_order,
        "feature_builder_sha256": sha256_file(ROOT / "create_modeling_dataset.py"),
        "project_feature_implementations": protocol["allowed_inputs"]["feature_implementation_hashes"],
        "include_targets": False,
    })
    stage_path = runtime / "base_feature_stage_manifest.json"
    if stage_path.is_file():
        saved_stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if saved_stage.get("identity_sha256") != stage_identity:
            raise RuntimeError("raw BASE64 batch cache identity differs from its frozen inputs")
        output_path = ROOT / saved_stage["output_path"]
        if sha256_file(output_path) != saved_stage.get("output_sha256"):
            raise RuntimeError("raw BASE64 batch cache hash differs from its completed stage manifest")
        schema = _validate_base64_output(output_path, base_order, int(base_record["rows"]), protocol["scope"]["source_raw_opened_exclusive_utc"], expected_dtypes)
        from features.reaction_profile_fixed_grid import load_state as load_rp_state
        from features.volume_profile_fixed_range import load_state as load_vp_state
        for family, loader in (("reaction_profile", load_rp_state), ("volume_profile", load_vp_state)):
            reference_record = lineage[f"{family}_state"]
            cached_state = saved_stage["profile_states"][family]
            generated_npz = ROOT / cached_state["generated_npz_path"]
            generated_json = ROOT / cached_state["generated_json_path"]
            if (sha256_file(generated_npz) != cached_state["generated_npz_sha256"]
                    or sha256_file(generated_json) != cached_state["generated_json_sha256"]):
                raise RuntimeError(f"cached E0045 {family} state hash differs from its stage manifest")
            if not _same_state_value(loader(ROOT / reference_record["base_path"]), loader(generated_npz.with_suffix(""))):
                raise RuntimeError(f"cached raw-derived {family} final state differs from E0041")
        return {**saved_stage, "path": output_path, "schema": schema, "reused": True}

    minimum_free = int(protocol["resource_limits"]["minimum_free_workspace_disk_gib"] * 1024**3)
    if shutil.disk_usage(ROOT).free < minimum_free:
        raise RuntimeError("workspace free disk is below the frozen E0045 minimum")
    built_path = dataset_builder.build_dataset_from_settings(settings, include_targets=False)
    if built_path.resolve() != output_path.resolve():
        raise RuntimeError("feature-only builder wrote outside its frozen E0045 output path")
    schema = _validate_base64_output(output_path, base_order, int(base_record["rows"]), protocol["scope"]["source_raw_opened_exclusive_utc"], expected_dtypes)

    from features.reaction_profile_fixed_grid import load_state as load_rp_state
    from features.volume_profile_fixed_range import load_state as load_vp_state

    state_results = {}
    for family, loader in (("reaction_profile", load_rp_state), ("volume_profile", load_vp_state)):
        reference_record = lineage[f"{family}_state"]
        reference_state = loader(ROOT / reference_record["base_path"])
        generated_base = runtime / "profile_states" / family / Path(reference_record["base_path"]).name
        generated_state = loader(generated_base)
        state_equal = _same_state_value(reference_state, generated_state)
        state_results[family] = {
            "matches_final_E0041_state": state_equal,
            "reference_sha256": reference_record["npz_sha256"],
            "generated_npz_path": generated_base.with_suffix(".npz").relative_to(ROOT).as_posix(),
            "generated_npz_sha256": sha256_file(generated_base.with_suffix(".npz")),
            "generated_json_path": generated_base.with_suffix(".json").relative_to(ROOT).as_posix(),
            "generated_json_sha256": sha256_file(generated_base.with_suffix(".json")),
        }
        if not state_equal:
            raise RuntimeError(f"raw-derived {family} final state differs from the E0041 saved state")

    result = {
        "identity_sha256": stage_identity,
        "output_path": output_path.relative_to(ROOT).as_posix(),
        "output_sha256": sha256_file(output_path),
        "rows": schema["rows"],
        "feature_order": base_order,
        "feature_dtypes": schema["feature_dtypes"],
        "profile_states": state_results,
        "target_generation": False,
        "mode": "single full-prefix batch using E0041 saved generator parameters; not stateful BASE64 continuation",
    }
    _write_small_json(stage_path, result)
    return {**result, "path": output_path, "schema": schema, "reused": False}


def preopen_alignment(opened) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    opened = pd.Timestamp(opened)
    if opened.tzinfo is None:
        raise ValueError("Opened must be timezone-aware")
    opened = opened.tz_convert("UTC")
    available = opened + pd.Timedelta(minutes=1)
    market_start = opened + pd.Timedelta(minutes=2)
    return opened, available, market_start


def validate_opened(values, exclusive_cutoff: str) -> pd.DatetimeIndex:
    opened = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))
    if opened.has_duplicates or not opened.is_monotonic_increasing:
        raise ValueError("Opened timestamps must be unique and strictly increasing")
    if np.any(opened.second != 0) or np.any(opened.microsecond != 0) or np.any(opened.nanosecond != 0):
        raise ValueError("Opened timestamps must align to whole UTC minutes")
    cutoff = pd.Timestamp(exclusive_cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("exclusive Opened cutoff must be timezone-aware")
    if len(opened) and opened[-1] >= cutoff.tz_convert("UTC"):
        raise ValueError("input contains an Opened row at or beyond the frozen development cutoff")
    return opened


def _source_months(first: str, last: str) -> list[str]:
    return [value.strftime("%Y-%m") for value in pd.period_range(first, last, freq="M").to_timestamp()]


def _expected_checksum(checksum_path: Path) -> str:
    text = checksum_path.read_text(encoding="ascii", errors="strict")
    match = re.search(r"\b([0-9a-fA-F]{64})\b", text)
    if not match:
        raise ValueError(f"invalid SHA-256 checksum sidecar: {checksum_path.name}")
    return match.group(1).lower()


def _read_month_archive(path: Path, *, cutoff_ms: int | None):
    """Read the permitted Binance columns; at cutoff read only the 14-byte open_time field."""
    rows = []
    timestamps = []
    with zipfile.ZipFile(path) as archive:
        entries = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(entries) != 1:
            raise RuntimeError(f"expected one CSV entry in {path.name}")
        with archive.open(entries[0], "r") as stream:
            while True:
                prefix = stream.read(14)
                if not prefix:
                    break
                if prefix.lower().startswith(b"open_time,"):
                    stream.readline()
                    continue
                if len(prefix) != 14 or prefix[13:14] != b",":
                    raise RuntimeError(f"noncanonical Binance open_time row in {path.name}")
                open_time_ms = int(prefix[:13])
                if cutoff_ms is not None and open_time_ms >= cutoff_ms:
                    # Do not read or parse the remainder of this protected row.
                    break
                remainder = stream.readline().rstrip(b"\r\n").split(b",")
                if len(remainder) < 10:
                    raise RuntimeError(f"truncated Binance kline row in {path.name}")
                timestamps.append(open_time_ms)
                rows.append((
                    float(remainder[0]), float(remainder[1]), float(remainder[2]),
                    float(remainder[3]), float(remainder[6]), float(remainder[9]),
                ))
    if not timestamps:
        raise RuntimeError(f"no permitted flow rows in {path.name}")
    index = pd.to_datetime(np.asarray(timestamps, dtype=np.int64), unit="ms", utc=True)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise RuntimeError(f"Binance source timestamps are duplicated or unordered: {path.name}")
    frame = pd.DataFrame(rows, columns=FLOW_RAW_COLUMNS, index=index)
    frame.index.name = "Opened"
    return frame


def _flow_parameters(source_config: dict) -> dict:
    config = source_config["flow_features"]
    return {
        "horizons": tuple(int(x) for x in config["horizons_minutes"]),
        "qbar_window": int(config["qbar_window_minutes"]),
        "sigma_window": int(config["sigma_window_minutes"]),
        "beta_window": int(config["beta_window_minutes"]),
        "minimum_beta_pairs": int(config["minimum_beta_pairs"]),
        "beta_variance_tolerance": float(config["beta_variance_tolerance"]),
        "sigma_floor": float(config["sigma_floor"]),
        "sigma_ddof": int(config["sigma_ddof"]),
    }


def _encode_frame(frame: pd.DataFrame) -> dict:
    values = []
    for row in frame.itertuples(index=False, name=None):
        encoded = []
        for value in row:
            if pd.isna(value):
                encoded.append(None)
            elif isinstance(value, (pd.Timestamp, np.datetime64)):
                encoded.append(pd.Timestamp(value).isoformat())
            elif isinstance(value, (np.integer, int)):
                encoded.append(int(value))
            elif isinstance(value, (np.floating, float)):
                encoded.append(float(value))
            else:
                encoded.append(str(value))
        values.append(encoded)
    return {"columns": list(frame.columns), "rows": values}


def _decode_frame(payload: dict, *, datetime_column: str = "Opened") -> pd.DataFrame:
    frame = pd.DataFrame(payload["rows"], columns=payload["columns"])
    if datetime_column in frame:
        frame[datetime_column] = pd.to_datetime(frame[datetime_column], utc=True, errors="raise")
    for column in frame.columns:
        if column != datetime_column:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.pop("integrity_sha256", None)
    body["integrity_sha256"] = _canonical_sha256(body)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    _replace_atomic(temporary, path)


def _load_checkpoint(path: Path, *, identity: str) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    integrity = value.pop("integrity_sha256", None)
    if integrity != _canonical_sha256(value):
        raise RuntimeError("checkpoint integrity hash mismatch")
    if value.get("run_identity_sha256") != identity:
        raise RuntimeError("checkpoint source/config/transform identity mismatch")
    return value


def _archive_identity(directory: Path, months: list[str], cutoff_ms: int) -> tuple[str, list[dict]]:
    records = []
    for month in months:
        path = directory / f"BTCUSDT-1m-{month}.zip"
        checksum_path = Path(str(path) + ".CHECKSUM")
        if not path.is_file() or not checksum_path.is_file():
            raise FileNotFoundError(f"missing local Binance source archive or checksum: {path.name}")
        expected = _expected_checksum(checksum_path)
        # Hashing the June 2026 ZIP would read protected rows after the cutoff.
        # Its allowed prefix is instead identified by the values parsed below.
        observed = None if month == "2026-06" else sha256_file(path)
        if observed is not None and observed != expected:
            raise RuntimeError(f"Binance archive checksum mismatch: {path.name}")
        records.append({
            "month": month,
            "path": path.relative_to(ROOT).as_posix(),
            "expected_archive_sha256": expected,
            "archive_sha256_verified": observed is not None,
            "protected_suffix_prefix_hash_only": observed is None,
        })
    return _canonical_sha256(records), records


def _save_flow_month(frame: pd.DataFrame, path: Path) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary, compression="zstd")
    digest = sha256_file(temporary)
    if path.exists():
        if sha256_file(path) != digest:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"uncommitted monthly flow output differs: {path.name}")
        temporary.unlink(missing_ok=True)
    else:
        _replace_atomic(temporary, path)
    return sha256_file(path)


def rebuild_flow_archives(protocol: dict, run_identity: str) -> tuple[list[Path], list[Path], dict]:
    """Recreate the six E0041 control-flow columns month by month from local ZIPs."""
    import pyarrow.parquet as pq
    from features.btc_taker_flow import compute_taker_flow_features

    inputs = protocol["allowed_inputs"]
    source = inputs["taker_flow_raw_archives"]
    flow_root = ROOT / source["directory"]
    source_config_path = ROOT / source["source_config_path"]
    if sha256_file(source_config_path) != source["source_config_sha256"]:
        raise RuntimeError("frozen taker-flow source config hash mismatch")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    parameters = _flow_parameters(source_config)
    cutoff = pd.Timestamp(source["last_allowed_source_open_time_utc"])
    cutoff_ms = int(cutoff.timestamp() * 1000) + 60_000
    months = _source_months("2020-06", "2026-06")
    archive_id, archive_records = _archive_identity(flow_root, months, cutoff_ms)
    identity = _canonical_sha256({
        "run_identity": run_identity,
        "stage": "raw_flow",
        "generator_code_sha256": sha256_file(Path(__file__)),
        "archive_identity": archive_id,
        "source_config_sha256": source["source_config_sha256"],
        "flow_module_sha256": sha256_file(ROOT / "features/btc_taker_flow.py"),
        "parameters": parameters,
        "cutoff_ms_exclusive": cutoff_ms,
    })

    runtime = RUNTIME_DIR / "flow_rebuilt"
    runtime.mkdir(parents=True, exist_ok=True)
    state = _load_checkpoint(CHECKPOINT_PATH, identity=run_identity)
    flow_state = state.get("flow_stage") if state else None
    if flow_state and flow_state.get("identity_sha256") != identity:
        raise RuntimeError("flow checkpoint configuration or archive identity mismatch")
    start_month = int(flow_state.get("next_month_index", 0)) if flow_state else 0
    tail = _decode_frame(flow_state["raw_tail"]).set_index("Opened") if flow_state and flow_state.get("raw_tail") else pd.DataFrame(columns=FLOW_RAW_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="Opened"))
    last_processed = pd.Timestamp(flow_state["last_processed_opened"]) if flow_state and flow_state.get("last_processed_opened") else None
    outputs = list(flow_state.get("outputs", [])) if flow_state else []

    for month_index in range(start_month, len(months)):
        month = months[month_index]
        archive_path = flow_root / f"BTCUSDT-1m-{month}.zip"
        month_cutoff = cutoff_ms if month == "2026-06" else None
        raw_month = _read_month_archive(archive_path, cutoff_ms=month_cutoff)
        if last_processed is not None and raw_month.index[0] <= last_processed:
            raise RuntimeError("flow source moved backward or duplicated the checkpoint boundary")
        joined = pd.concat([tail, raw_month]) if len(tail) else raw_month
        if joined.index.has_duplicates or not joined.index.is_monotonic_increasing:
            raise RuntimeError(f"flow source timestamps duplicate or reorder at {month}")

        raw_output = raw_month.reset_index(names="Opened")
        raw_path = runtime / f"flow_source_{month}.parquet"
        raw_sha = _save_flow_month(raw_output, raw_path)
        outputs = [record for record in outputs if record["month"] != month]
        outputs.append({
            "month": month,
            "rows": int(len(raw_month)),
            "raw_path": raw_path.relative_to(ROOT).as_posix(),
            "raw_sha256": raw_sha,
            "raw_prefix_sha256": _feature_projection_hash(
                raw_month.reset_index(), ["Opened", *FLOW_RAW_COLUMNS],
            ),
            "source_archive_sha256": archive_records[month_index]["expected_archive_sha256"],
            "source_archive_sha256_verified": archive_records[month_index]["archive_sha256_verified"],
        })

        tail = joined.loc[joined.index >= joined.index[-1] - pd.Timedelta(minutes=FLOW_TAIL_MINUTES)].copy()
        last_processed = raw_month.index[-1]
        flow_state = {
            "identity_sha256": identity,
            "next_month_index": month_index + 1,
            "last_completed_month": month,
            "last_processed_opened": last_processed.isoformat(),
            "next_expected_opened": (last_processed + pd.Timedelta(minutes=1)).isoformat(),
            "source_row_count": int(sum(item["rows"] for item in outputs if item["month"] <= month)),
            "raw_tail": _encode_frame(tail.reset_index()),
            "rolling_tail_minutes": FLOW_TAIL_MINUTES + 1,
            "outputs": outputs,
            "archive_identity": archive_id,
        }
        _atomic_checkpoint(CHECKPOINT_PATH, {
            "schema_version": 1,
            "stage": "flow",
            "run_identity_sha256": run_identity,
            "flow_stage": flow_state,
        })

    raw_paths = [runtime / f"flow_source_{month}.parquet" for month in months]
    if any(not path.is_file() for path in raw_paths):
        raise RuntimeError("flow source replay ended without all monthly development outputs")
    batch_outputs = flow_state.get("batch_outputs", []) if flow_state else []
    reusable = (
        len(batch_outputs) == len(months)
        and all(
            (ROOT / item["path"]).is_file()
            and sha256_file(ROOT / item["path"]) == item["sha256"]
            for item in batch_outputs
        )
    )
    if reusable:
        feature_paths = [ROOT / item["path"] for item in batch_outputs]
    else:
        # The frozen reference was generated through one uninterrupted pandas
        # rolling pass. Re-run that same numerical kernel over only the six
        # projected raw fields; monthly state restarts alter compensated rolling
        # sums by ~1e-8 despite having sufficient mathematical history.
        raw_parts = [
            pd.read_parquet(path, columns=["Opened", *FLOW_RAW_COLUMNS])
            for path in raw_paths
        ]
        source_frame = pd.concat(raw_parts, ignore_index=True)
        del raw_parts
        source_frame["Opened"] = pd.to_datetime(source_frame["Opened"], utc=True, errors="raise")
        source_frame.set_index("Opened", inplace=True)
        if source_frame.index.has_duplicates or not source_frame.index.is_monotonic_increasing:
            raise RuntimeError("projected raw flow source is not strictly chronological")
        batch = compute_taker_flow_features(source_frame, **parameters)
        del source_frame
        batch_outputs = []
        feature_paths = []
        for month in months:
            month_start = pd.Timestamp(f"{month}-01T00:00:00Z")
            month_end = month_start + pd.offsets.MonthBegin(1)
            part = batch.loc[(batch.index >= month_start) & (batch.index < month_end), list(FLOW_COLUMNS)].copy()
            part.insert(0, "Opened", part.index)
            part.reset_index(drop=True, inplace=True)
            output_path = runtime / f"flow_batch_features_{month}.parquet"
            output_sha = _save_flow_month(part, output_path)
            feature_paths.append(output_path)
            batch_outputs.append({
                "month": month,
                "path": output_path.relative_to(ROOT).as_posix(),
                "sha256": output_sha,
                "rows": int(len(part)),
            })
        flow_state = dict(flow_state)
        flow_state["batch_outputs"] = batch_outputs
        flow_state["flow_compute_mode"] = "single chronological pass over projected raw flow fields; no fit"
        flow_state["batch_feature_rows"] = int(sum(item["rows"] for item in batch_outputs))
        _atomic_checkpoint(CHECKPOINT_PATH, {
            "schema_version": 1,
            "stage": "flow_batch_complete",
            "run_identity_sha256": run_identity,
            "flow_stage": flow_state,
        })
    return feature_paths, raw_paths, {
        "archive_identity_sha256": archive_id,
        "archives": archive_records,
        "parameters": parameters,
        "outputs": outputs,
        "batch_outputs": batch_outputs,
        "flow_compute_mode": flow_state.get("flow_compute_mode"),
        "last_processed_opened": None if last_processed is None else last_processed.isoformat(),
    }


class _ParquetRowReader:
    """Read only named columns and row groups proven to end before the cutoff."""

    def __init__(self, path: Path, columns: list[str], cutoff: str, start_row: int = 0):
        import pyarrow.parquet as pq

        self.path = path
        self.columns = list(columns)
        self.parquet = pq.ParquetFile(path)
        if not set(self.columns).issubset(self.parquet.schema.names):
            raise ValueError(f"requested projection is absent from {path.name}")
        self.group_starts = []
        row_start = 0
        opened_index = self.parquet.schema.names.index("Opened")
        cutoff_ts = pd.Timestamp(cutoff).tz_convert("UTC")
        for group_index in range(self.parquet.num_row_groups):
            metadata = self.parquet.metadata.row_group(group_index).column(opened_index).statistics
            if metadata is None or metadata.max is None:
                raise RuntimeError(f"cannot prove Opened cutoff from Parquet statistics: {path.name}")
            maximum = pd.Timestamp(metadata.max)
            maximum = maximum.tz_localize("UTC") if maximum.tzinfo is None else maximum.tz_convert("UTC")
            if maximum >= cutoff_ts:
                raise RuntimeError(f"Parquet row group may materialize a protected Opened row: {path.name}")
            self.group_starts.append(row_start)
            row_start += self.parquet.metadata.row_group(group_index).num_rows
        self.total_rows = row_start
        if start_row < 0 or start_row > self.total_rows:
            raise ValueError("Parquet cursor is outside the file")
        self.row = start_row
        self.group_index = 0
        self.table = None
        self.local_row = 0
        self._load_group_for_row()

    def _load_group_for_row(self):
        while self.group_index < len(self.group_starts):
            start = self.group_starts[self.group_index]
            count = self.parquet.metadata.row_group(self.group_index).num_rows
            if self.row < start + count:
                self.table = self.parquet.read_row_group(self.group_index, columns=self.columns)
                self.local_row = self.row - start
                return
            self.group_index += 1
        self.table = None
        self.local_row = 0

    def read(self, count: int) -> pd.DataFrame:
        import pyarrow as pa

        remaining = min(int(count), self.total_rows - self.row)
        pieces = []
        while remaining:
            if self.table is None:
                self._load_group_for_row()
                if self.table is None:
                    break
            available = self.table.num_rows - self.local_row
            take = min(remaining, available)
            pieces.append(self.table.slice(self.local_row, take))
            self.local_row += take
            self.row += take
            remaining -= take
            if self.local_row >= self.table.num_rows:
                self.group_index += 1
                self.table = None
        if not pieces:
            return pd.DataFrame(columns=self.columns)
        return pa.concat_tables(pieces).to_pandas()


def _raw_csv_block(stream, count: int, header: bytes) -> pd.DataFrame:
    lines = []
    for _ in range(int(count)):
        line = stream.readline()
        if not line:
            break
        lines.append(line)
    if not lines:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.read_csv(io.BytesIO(header + b"".join(lines)), usecols=list(RAW_COLUMNS))


def _valid_candles(frame: pd.DataFrame) -> np.ndarray:
    values = frame[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=np.float64)
    return (
        np.isfinite(values).all(axis=1)
        & (values[:, 0] > 0.0) & (values[:, 1] > 0.0)
        & (values[:, 2] > 0.0) & (values[:, 3] > 0.0)
        & (values[:, 1] >= values[:, 2]) & (values[:, 2] <= values[:, 3])
        & (values[:, 4] >= 0.0)
    )


def _esn_config(payload: dict):
    from features.esn_reservoir import ESNConfig

    raw = payload["config"]
    allowed = {item.name for item in fields(ESNConfig)}
    config = {key: value for key, value in raw.items() if key in allowed}
    for key in (
        "input_groups", "component_sizes", "component_topologies", "component_leak_rates",
        "component_spectral_radii", "component_input_scalings", "component_bias_scalings",
        "component_sparse_degrees",
    ):
        if config.get(key) is not None:
            config[key] = tuple(config[key])
    if config.get("component_input_groups") is not None:
        config["component_input_groups"] = tuple(tuple(value) for value in config["component_input_groups"])
    return ESNConfig(**config)


def new_dynamic_state(transforms: dict):
    from features.causal_memory_control import CausalMemoryState
    from features.esn_reservoir import ESNEngine, FeatureScaler

    memory = transforms["memory"]
    memory_input = list(memory["identity"]["input_columns"])
    memory_scaler = FeatureScaler.from_dict(memory["input_scaler"])
    state = {
        "memory_state": CausalMemoryState(len(memory_input)),
        "memory_scaler": memory_scaler,
        "memory_indices": np.asarray([memory_scaler.columns.index(name) for name in memory_input], dtype=np.int64),
        "esn_engines": {},
        "esn_bundles": transforms["esn"],
        "memory_bundle": memory,
    }
    for seed, payload in transforms["esn"].items():
        config = _esn_config(payload)
        scaler = FeatureScaler.from_dict(payload["scaler"])
        if tuple(config.selected_columns) != tuple(scaler.columns):
            raise RuntimeError(f"frozen ESN scaler input order mismatch for seed {seed}")
        state["esn_engines"][seed] = ESNEngine(
            config, seed, scaler, warmup_steps=int(payload["warmup_steps"]),
        )
    return state


def _dynamic_checkpoint_payload(state: dict, *, temporary_root: Path | None = None) -> dict:
    engines = {}
    temporary_root = Path(temporary_root) if temporary_root else (RUNTIME_DIR / "engine_checkpoint_parts")
    temporary_root.mkdir(parents=True, exist_ok=True)
    for seed, engine in state["esn_engines"].items():
        path = temporary_root / f"esn_{seed}.json"
        engine.save_checkpoint(path)
        engines[str(seed)] = json.loads(path.read_text(encoding="utf-8"))
        path.unlink(missing_ok=True)
    return {
        "memory_state": state["memory_state"].to_dict(),
        "esn_engines": engines,
    }


def restore_dynamic_state(payload: dict, transforms: dict, *, temporary_root: Path | None = None):
    from features.causal_memory_control import CausalMemoryState
    from features.esn_reservoir import ESNEngine, FeatureScaler

    state = new_dynamic_state(transforms)
    temporary_root = Path(temporary_root) if temporary_root else (RUNTIME_DIR / "engine_checkpoint_parts")
    state["memory_state"] = CausalMemoryState.from_dict(payload["memory_state"])
    memory = state["memory_state"]
    if not np.isfinite(memory.history).all() or not np.isfinite(memory.ema).all():
        raise RuntimeError("memory checkpoint contains non-finite state")
    engines = {}
    for seed, bundle in state["esn_bundles"].items():
        checkpoint = payload["esn_engines"].get(str(seed))
        if checkpoint is None:
            raise RuntimeError(f"ESN checkpoint is missing seed {seed}")
        temporary_root.mkdir(parents=True, exist_ok=True)
        path = temporary_root / f"restore_{seed}_{os.getpid()}.json"
        path.write_text(json.dumps(checkpoint, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        try:
            engines[seed] = ESNEngine.load_checkpoint(
                path, expected_config=_esn_config(bundle), expected_seed=seed,
            )
        finally:
            path.unlink(missing_ok=True)
    state["esn_engines"] = engines
    return state


def generate_dynamic_block(
    state: dict, candles: pd.DataFrame, *, new_rows_start: int = 0,
) -> dict[str, np.ndarray]:
    from features.esn_reservoir import INPUT_COLUMNS, build_causal_esn_inputs

    opened, raw_frame = build_causal_esn_inputs(candles)
    opened = opened[new_rows_start:]
    raw = raw_frame.iloc[new_rows_start:].to_numpy(dtype=np.float32, copy=False)
    candles = candles.iloc[new_rows_start:]
    valid = _valid_candles(candles)
    memory_bundle = state["esn_bundles"]
    memory_json = state["memory_bundle"] if "memory_bundle" in state else None
    if memory_json is None:
        raise RuntimeError("memory projection is not attached to the dynamic state")
    memory_columns = memory_json["identity"]["input_columns"]
    input_positions = {name: index for index, name in enumerate(INPUT_COLUMNS)}
    memory_indices = np.asarray([input_positions[name] for name in memory_columns], dtype=np.int64)
    scaled = state["memory_scaler"].transform(raw[:, memory_indices].astype(np.float64, copy=False))
    memory_features, memory_ready = state["memory_state"].update_many(
        opened, scaled, valid_mask=valid,
    )
    memory_out = np.full((len(opened), 8), np.nan, dtype=np.float32)
    if memory_ready.any():
        bundle = memory_json
        projection = (memory_features[memory_ready].astype(np.float64) - np.asarray(bundle["pca_mean"], dtype=np.float64)) @ np.asarray(bundle["pca_components"], dtype=np.float64).T
        memory_out[memory_ready] = projection.astype(np.float32)

    outputs = {"memory": memory_out}
    for seed, engine in state["esn_engines"].items():
        payload = state["esn_bundles"][seed]
        indices = np.asarray([input_positions[name] for name in engine.config.selected_columns], dtype=np.int64)
        states, ready = engine.update_many(opened, raw[:, indices], valid_mask=valid)
        result = np.full((len(opened), 8), np.nan, dtype=np.float32)
        if ready.any():
            projection = payload["projection"]
            values = (states[ready].astype(np.float64) - np.asarray(projection["mean"], dtype=np.float64)) @ np.asarray(projection["components"], dtype=np.float64).T
            result[ready] = values.astype(np.float32)
        outputs[f"esn_{seed}"] = result
    return outputs


def _feature_projection_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    for column in columns:
        values = frame[column]
        if column == "Opened":
            array = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise")).as_unit("ns").asi8
        else:
            array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        digest.update(column.encode("utf-8"))
        digest.update(np.ascontiguousarray(array).view(np.uint8))
        mask = np.ascontiguousarray(pd.isna(array).astype(np.uint8))
        digest.update(mask)
    return digest.hexdigest()


def _parity_update(summary: dict, name: str, actual, expected, opened, *, atol: float, rtol: float) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or actual.ndim != 2:
        raise ValueError(f"parity matrix shape mismatch for {name}")
    opened = pd.DatetimeIndex(pd.to_datetime(opened, utc=True, errors="raise"))
    record = summary.setdefault(name, {
        "rows": 0,
        "mask_mismatch_rows": 0,
        "nan_mask_mismatch_rows": 0,
        "inf_mask_mismatch_rows": 0,
        "numeric_mismatch_rows": 0,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "first_mismatch_opened": None,
    })
    actual_finite = np.isfinite(actual)
    expected_finite = np.isfinite(expected)
    nan_mismatch = np.isnan(actual) != np.isnan(expected)
    inf_mismatch = (np.isinf(actual) != np.isinf(expected)) | (
        np.isinf(actual) & np.isinf(expected) & (np.signbit(actual) != np.signbit(expected))
    )
    nan_rows = np.any(nan_mismatch, axis=1)
    inf_rows = np.any(inf_mismatch, axis=1)
    row_mismatch = nan_rows | inf_rows
    record["mask_mismatch_rows"] += int(row_mismatch.sum())
    record["nan_mask_mismatch_rows"] += int(nan_rows.sum())
    record["inf_mask_mismatch_rows"] += int(inf_rows.sum())
    both = actual_finite & expected_finite
    if both.any():
        delta = np.abs(actual[both].astype(np.float64) - expected[both].astype(np.float64))
        denom = np.maximum(np.abs(expected[both].astype(np.float64)), np.finfo(np.float64).tiny)
        record["max_abs_error"] = max(float(record["max_abs_error"]), float(delta.max(initial=0.0)))
        record["max_rel_error"] = max(float(record["max_rel_error"]), float((delta / denom).max(initial=0.0)))
        within = actual_finite & expected_finite
        errors = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        allowed = atol + rtol * np.abs(expected.astype(np.float64))
        value_mismatch = np.any(within & (errors > allowed), axis=1)
        record["numeric_mismatch_rows"] += int(value_mismatch.sum())
        mismatch = row_mismatch | value_mismatch
    else:
        mismatch = row_mismatch
    if mismatch.any() and record["first_mismatch_opened"] is None:
        record["first_mismatch_opened"] = opened[int(np.flatnonzero(mismatch)[0])].isoformat()
    record["rows"] += int(len(actual))


def _report_parity_row(rows: list[dict], block: str, name: str, record: dict, atol: float, rtol: float) -> None:
    passed = (
        record.get("mask_mismatch_rows", 0) == 0
        and record.get("numeric_mismatch_rows", 0) == 0
    )
    rows.append({
        "block": block,
        "feature": name,
        "rows": record.get("rows", 0),
        "mask_mismatch_rows": record.get("mask_mismatch_rows", 0),
        "nan_mask_mismatch_rows": record.get("nan_mask_mismatch_rows", 0),
        "inf_mask_mismatch_rows": record.get("inf_mask_mismatch_rows", 0),
        "numeric_mismatch_rows": record.get("numeric_mismatch_rows", 0),
        "max_abs_error": record.get("max_abs_error", ""),
        "max_rel_error": record.get("max_rel_error", ""),
        "first_mismatch_opened_utc": record.get("first_mismatch_opened") or "",
        "atol": atol,
        "rtol": rtol,
        "status": "PASS" if passed else "FAIL",
    })


def compare_rebuilt_flow(protocol: dict, feature_paths: list[Path]) -> tuple[pd.DataFrame, list[dict], dict]:
    reference_path = ROOT / protocol["allowed_inputs"]["canonical_flow_reference"]["path"]
    expected_sha = protocol["allowed_inputs"]["canonical_flow_reference"]["sha256"]
    if sha256_file(reference_path) != expected_sha:
        raise RuntimeError("canonical E0041 development flow reference hash mismatch")
    rebuilt_parts = [pd.read_parquet(path, columns=["Opened", *FLOW_COLUMNS]) for path in feature_paths]
    rebuilt = pd.concat(rebuilt_parts, ignore_index=True)
    rebuilt["Opened"] = pd.to_datetime(rebuilt["Opened"], utc=True, errors="raise")
    rebuilt.sort_values("Opened", kind="stable", inplace=True, ignore_index=True)
    if rebuilt["Opened"].duplicated().any() or not rebuilt["Opened"].is_monotonic_increasing:
        raise RuntimeError("rebuilt flow rows are not uniquely chronological")

    reader = _ParquetRowReader(
        reference_path, ["Opened", *FLOW_COLUMNS],
        protocol["scope"]["source_raw_opened_exclusive_utc"],
    )
    summary = {}
    first_opened = None
    last_opened = None
    rows = 0
    while reader.row < reader.total_rows:
        expected = reader.read(65536)
        opened = validate_opened(expected["Opened"], protocol["scope"]["source_raw_opened_exclusive_utc"])
        lookup = rebuilt["Opened"].searchsorted(opened)
        match = lookup < len(rebuilt)
        match[match] &= rebuilt["Opened"].to_numpy()[lookup[match]] == opened.to_numpy()[match]
        if not match.all():
            bad = int(np.flatnonzero(~match)[0])
            raise RuntimeError(f"rebuilt flow is missing reference row at {opened[bad]}")
        actual = rebuilt.iloc[lookup].loc[:, FLOW_COLUMNS].to_numpy(dtype=np.float64)
        target = expected.loc[:, FLOW_COLUMNS].to_numpy(dtype=np.float64)
        for index, name in enumerate(FLOW_COLUMNS):
            _parity_update(
                summary, name, actual[:, index:index + 1], target[:, index:index + 1],
                opened, atol=1e-10, rtol=1e-10,
            )
        first_opened = first_opened or opened[0].isoformat()
        last_opened = opened[-1].isoformat()
        rows += len(opened)

    parity_rows = []
    for name in FLOW_COLUMNS:
        _report_parity_row(parity_rows, "flow", name, summary[name], 1e-10, 1e-10)
    return rebuilt, parity_rows, {
        "rows_compared": rows,
        "first_reference_opened": first_opened,
        "last_reference_opened": last_opened,
        "rebuilt_rows": int(len(rebuilt)),
        "rebuilt_last_opened": rebuilt["Opened"].iloc[-1].isoformat(),
        "unmatched_suffix_rows_after_reference": int((rebuilt["Opened"] > pd.Timestamp(last_opened)).sum()),
    }

def _load_raw_tail(checkpoint: dict | None) -> pd.DataFrame:
    if not checkpoint or not checkpoint.get("dynamic_stage", {}).get("raw_tail"):
        return pd.DataFrame(columns=RAW_COLUMNS)
    return _decode_frame(checkpoint["dynamic_stage"]["raw_tail"])


def _apply_platt(probabilities, calibrator: dict) -> np.ndarray:
    from scipy.special import expit

    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return expit(float(calibrator["intercept"]) + float(calibrator["coefficient"]) * logits)


def _model_matrix(base: np.ndarray, memory: np.ndarray, esn: np.ndarray) -> np.ndarray:
    values = np.concatenate((base, memory, esn), axis=1).astype(np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _prediction_panel(base: np.ndarray, memory: np.ndarray, esn: dict,
                      canonical_base: np.ndarray, canonical_memory: np.ndarray,
                      canonical_esn: dict, opened: pd.DatetimeIndex,
                      d_members: list[dict], boosters: dict, calibrator: dict,
                      windows: list[dict], prediction_cutoff: str,
                      parity: dict) -> list[dict]:
    """Compare raw-derived and canonical-input predictions on every declared row."""
    selected = np.zeros(len(opened), dtype=bool)
    for window in windows:
        start = pd.Timestamp(window["start_inclusive"]).tz_convert("UTC")
        end = pd.Timestamp(window["end_exclusive"]).tz_convert("UTC")
        selected |= (opened >= start) & (opened < end)
    selected &= opened <= pd.Timestamp(prediction_cutoff).tz_convert("UTC")
    rows = np.flatnonzero(selected)
    if not len(rows):
        return []

    raw_member_values = []
    canonical_member_values = []
    for member in d_members:
        seed = int(member["reservoir_seed"])
        lightgbm_seed = int(member["lightgbm_seed"])
        raw_matrix = _model_matrix(base[rows], memory[rows], esn[seed][rows])
        canonical_matrix = _model_matrix(
            canonical_base[rows], canonical_memory[rows], canonical_esn[seed][rows],
        )
        booster = boosters[(seed, lightgbm_seed)]
        actual = np.asarray(booster.predict(raw_matrix), dtype=np.float64)
        expected = np.asarray(booster.predict(canonical_matrix), dtype=np.float64)
        raw_member_values.append(actual)
        canonical_member_values.append(expected)
        _parity_update(
            parity, f"member_r{seed}_l{lightgbm_seed}", actual[:, None], expected[:, None],
            opened[rows], atol=2e-12, rtol=2e-12,
        )

    raw_members = np.vstack(raw_member_values)
    canonical_members = np.vstack(canonical_member_values)
    raw_mean = raw_members.mean(axis=0, dtype=np.float64)
    canonical_mean = canonical_members.mean(axis=0, dtype=np.float64)
    raw_calibrated = _apply_platt(raw_mean, calibrator)
    canonical_calibrated = _apply_platt(canonical_mean, calibrator)
    _parity_update(parity, "D_uniform_mean", raw_mean[:, None], canonical_mean[:, None], opened[rows], atol=2e-12, rtol=2e-12)
    _parity_update(parity, "D_terminal_Platt", raw_calibrated[:, None], canonical_calibrated[:, None], opened[rows], atol=2e-12, rtol=2e-12)

    market_start = opened[rows] + pd.Timedelta(minutes=2)
    records = []
    for panel_index, row in enumerate(rows):
        records.append({
            "Opened": opened[row].isoformat(),
            "available_at_utc": (opened[row] + pd.Timedelta(minutes=1)).isoformat(),
            "market_start_utc": market_start[panel_index].isoformat(),
            "is_decision_row": bool(market_start[panel_index].minute % 5 == 0),
            "raw_nonfinite_feature_count": int((~np.isfinite(base[row])).sum() + (~np.isfinite(memory[row])).sum() + sum((~np.isfinite(values[row])).sum() for values in esn.values())),
            "canonical_nonfinite_feature_count": int((~np.isfinite(canonical_base[row])).sum() + (~np.isfinite(canonical_memory[row])).sum() + sum((~np.isfinite(values[row])).sum() for values in canonical_esn.values())),
            "raw_member_probabilities": [float(values[panel_index]) for values in raw_members],
            "canonical_member_probabilities": [float(values[panel_index]) for values in canonical_members],
            "raw_D_uniform_probability": float(raw_mean[panel_index]),
            "canonical_D_uniform_probability": float(canonical_mean[panel_index]),
            "raw_D_terminal_Platt_probability": float(raw_calibrated[panel_index]),
            "canonical_D_terminal_Platt_probability": float(canonical_calibrated[panel_index]),
        })
    return records


def _load_d_boosters(d_manifest: dict) -> dict:
    import lightgbm as lgb

    output = {}
    for member in d_manifest["members"]:
        seed = int(member["reservoir_seed"])
        lgb_seed = int(member["lightgbm_seed"])
        path = Path(member["model_path"])
        sidecar_path = Path(member["metadata_path"])
        if sha256_file(path) != member["model_sha256"] or sha256_file(sidecar_path) != member["metadata_sha256"]:
            raise RuntimeError(f"final E0041 D member or sidecar hash changed: {path.name}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        expected_order = list(member["feature_order"])
        if sidecar["identity"].get("feature_order") != expected_order:
            raise RuntimeError(f"final E0041 D member feature order changed: {path.name}")
        if (int(sidecar["num_features"]) != len(expected_order)
                or int(sidecar["num_trees"]) != int(member["num_trees"])):
            raise RuntimeError(f"final E0041 D sidecar model shape changed: {path.name}")
        if (int(sidecar["identity"]["reservoir_seed"]) != seed
                or int(sidecar["identity"]["lightgbm_seed"]) != lgb_seed):
            raise RuntimeError(f"final E0041 D member seed changed: {path.name}")
        booster = lgb.Booster(model_file=str(path))
        if (booster.num_feature() != len(expected_order)
                or booster.feature_name() != expected_order
                or booster.num_trees() != int(sidecar["num_trees"])):
            raise RuntimeError(f"loaded final E0041 D member differs from its verified sidecar: {path.name}")
        output[(seed, lgb_seed)] = booster
    return output


def _write_small_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _replace_atomic(temporary, path)


def replay_development(protocol: dict, run_identity: str, bundle_identity: dict,
                       d_manifest: dict, transforms: dict, base64_build: dict,
                       flow_paths: list[Path], flow_raw_paths: list[Path],
                       flow_rows: pd.DataFrame, flow_summary: dict,
                       *, stop_after_opened_utc: str | None = None) -> dict:
    """Replay the dynamic state stream and compare raw-derived final E0041 inputs."""
    import pyarrow.parquet as pq

    inputs = protocol["allowed_inputs"]
    raw_record = inputs["safe_index_ohlcv_prefix"]
    raw_path = _required_file_hash(raw_record["path"], raw_record["sha256"], "safe E0041 development raw prefix")
    if tuple(pd.read_csv(raw_path, nrows=0).columns) != tuple(raw_record["columns"]):
        raise RuntimeError("safe E0041 raw prefix column schema changed")
    for key in ("canonical_model_feature_reference", "canonical_dynamic_reference"):
        record = inputs[key]
        _required_file_hash(record["path"], record["sha256"], f"canonical E0041 {key}")

    cutoff = protocol["scope"]["source_raw_opened_exclusive_utc"]
    base_order = list(d_manifest["base_feature_order"])
    base64_order = base_order[:64]
    memory_names = list(d_manifest["memory_feature_order"])
    reservoir_seeds = [int(seed) for seed in d_manifest["reservoir_seeds"]]
    dynamic_names = memory_names + [
        name for seed in reservoir_seeds for name in transforms["esn"][seed]["feature_names"]
    ]
    base_path = Path(base64_build["path"])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    canonical_path = ROOT / inputs["canonical_model_feature_reference"]["path"]
    dynamic_path = ROOT / inputs["canonical_dynamic_reference"]["path"]
    canonical_schema = pq.ParquetFile(canonical_path).schema.names
    missing = [name for name in base_order if name not in canonical_schema]
    if missing:
        raise RuntimeError(f"final E0041 model-feature reference lacks manifest features: {missing}")
    if not set(dynamic_names).issubset(pq.ParquetFile(dynamic_path).schema.names):
        raise RuntimeError("final E0041 dynamic reference lacks manifest transform features")

    replay_started = time.perf_counter()
    checkpoint_io_seconds = 0.0
    base_comparison_seconds = 0.0
    dynamic_comparison_seconds = 0.0
    prediction_seconds = 0.0
    first_opened_this_call = None
    checkpoint_started = time.perf_counter()
    checkpoint = _load_checkpoint(CHECKPOINT_PATH, identity=run_identity)
    checkpoint_io_seconds += time.perf_counter() - checkpoint_started
    dynamic_checkpoint = checkpoint.get("dynamic_stage") if checkpoint else None
    source_identity = _canonical_sha256({
        "run_identity_sha256": run_identity,
        "bundle_identity": bundle_identity,
        "safe_raw_sha256": raw_record["sha256"],
        "raw_base64_sha256": base64_build["output_sha256"],
        "canonical_feature_reference": inputs["canonical_model_feature_reference"],
        "canonical_dynamic_reference": inputs["canonical_dynamic_reference"],
        "canonical_flow_reference": inputs["canonical_flow_reference"],
        "feature_order": base_order,
        "dynamic_order": dynamic_names,
        "transform_hashes": protocol["allowed_inputs"]["frozen_transforms"],
        "state_engine_hashes": protocol["allowed_inputs"]["feature_implementation_hashes"],
    })
    if dynamic_checkpoint and dynamic_checkpoint.get("identity_sha256") != source_identity:
        raise RuntimeError("dynamic checkpoint source or state-engine identity mismatch")
    start_row = int(dynamic_checkpoint.get("source_row_index", 0)) if dynamic_checkpoint else 0
    byte_offset = int(dynamic_checkpoint.get("source_byte_offset", 0)) if dynamic_checkpoint else 0
    state = restore_dynamic_state(dynamic_checkpoint["dynamic_state"], transforms) if dynamic_checkpoint else new_dynamic_state(transforms)
    checkpoint_io_seconds += time.perf_counter() - checkpoint_started - checkpoint_io_seconds
    state["memory_bundle"] = transforms["memory"]
    raw_tail = _load_raw_tail(checkpoint)
    flow_tail_rows = len(dynamic_checkpoint.get("flow_tail", {}).get("rows", [])) if dynamic_checkpoint else 0
    parity = dict(dynamic_checkpoint.get("parity", {})) if dynamic_checkpoint else {}
    panel = list(dynamic_checkpoint.get("prediction_panel", [])) if dynamic_checkpoint else []
    projection_hashes = list(dynamic_checkpoint.get("base_projection_block_hashes", [])) if dynamic_checkpoint else []
    dynamic_hashes = list(dynamic_checkpoint.get("dynamic_projection_block_hashes", [])) if dynamic_checkpoint else []
    gap_count = int(dynamic_checkpoint.get("timestamp_gaps", 0)) if dynamic_checkpoint else 0
    invalid_candles = int(dynamic_checkpoint.get("invalid_candles", 0)) if dynamic_checkpoint else 0

    expected_rows = int(raw_record["rows"])
    base_reader = _ParquetRowReader(base_path, ["Opened", *base64_order], cutoff, start_row=start_row)
    canonical_reader = _ParquetRowReader(canonical_path, ["Opened", *base_order], cutoff, start_row=start_row)
    dynamic_reader = _ParquetRowReader(dynamic_path, ["Opened", *dynamic_names], cutoff, start_row=start_row)
    if any(reader.total_rows != expected_rows for reader in (base_reader, canonical_reader, dynamic_reader)):
        raise RuntimeError("safe raw, rebuilt BASE64, final BASE70, and dynamic row counts differ")

    flow_opened = pd.DatetimeIndex(pd.to_datetime(flow_rows["Opened"], utc=True, errors="raise"))
    flow_opened_ns = flow_opened.as_unit("ns").asi8
    flow_matrix = flow_rows.loc[:, FLOW_COLUMNS].to_numpy(dtype=np.float64)
    flow_raw = pd.concat(
        [pd.read_parquet(path, columns=["Opened", *FLOW_RAW_COLUMNS]) for path in flow_raw_paths],
        ignore_index=True,
    )
    flow_raw["Opened"] = pd.to_datetime(flow_raw["Opened"], utc=True, errors="raise")
    flow_source_ns = pd.DatetimeIndex(flow_raw["Opened"]).as_unit("ns").asi8
    flow_last_reference = pd.Timestamp(flow_summary["last_reference_opened"]).tz_convert("UTC")
    flow_indices = np.asarray([base_order.index(name) for name in FLOW_COLUMNS], dtype=np.int64)

    boosters = _load_d_boosters(d_manifest)
    calibrator_doc = d_manifest["calibrator_doc"]
    calibrator = {
        "intercept": float(calibrator_doc["intercept"]),
        "coefficient": float(calibrator_doc["coefficient"]),
    }
    if not np.isfinite(list(calibrator.values())).all():
        raise RuntimeError("final E0041 terminal Platt coefficients are non-finite")
    windows = protocol["predeclared_validation"]["windows_utc"]
    chunk_rows = int(protocol["resource_limits"]["chunk_rows"])
    raw_open_ns_previous = None
    if start_row:
        if not byte_offset or raw_tail.empty:
            raise RuntimeError("dynamic checkpoint lacks raw source byte cursor or rolling candle tail")
        if dynamic_checkpoint.get("last_processed_opened") is None:
            raise RuntimeError("dynamic checkpoint is missing last processed Opened")
        raw_open_ns_previous = pd.Timestamp(dynamic_checkpoint["last_processed_opened"]).value

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with raw_path.open("rb") as stream:
        header = stream.readline()
        if byte_offset:
            stream.seek(byte_offset)
        for batch_start in range(start_row, expected_rows, chunk_rows):
            count = min(chunk_rows, expected_rows - batch_start)
            raw = _raw_csv_block(stream, count, header)
            if len(raw) != count:
                raise RuntimeError("safe raw prefix ended before its frozen row count")
            opened = validate_opened(raw["Opened"], cutoff)
            if first_opened_this_call is None:
                first_opened_this_call = opened[0].isoformat()
            opened_ns = opened.as_unit("ns").asi8
            if raw_open_ns_previous is not None:
                if opened_ns[0] <= raw_open_ns_previous:
                    raise RuntimeError("resume source cursor would duplicate or reverse the boundary minute")
                gap_count += int((opened_ns[0] - raw_open_ns_previous) != MINUTE_NS)
            if len(opened_ns) > 1:
                gap_count += int(np.count_nonzero(np.diff(opened_ns) != MINUTE_NS))
            raw_open_ns_previous = int(opened_ns[-1])

            rebuilt_base64 = base_reader.read(count)
            canonical = canonical_reader.read(count)
            reference = dynamic_reader.read(count)
            if any(len(frame) != count for frame in (rebuilt_base64, canonical, reference)):
                raise RuntimeError("projected E0041 development source ended before safe raw")
            opened_frames = [
                pd.DatetimeIndex(pd.to_datetime(frame["Opened"], utc=True, errors="raise"))
                for frame in (rebuilt_base64, canonical, reference)
            ]
            if not all(np.array_equal(item.asi8, opened.asi8) for item in opened_frames):
                raise RuntimeError("safe raw-to-E0041 feature timestamps are not one-to-one and identical")
            projection_hashes.append(_feature_projection_hash(rebuilt_base64, ["Opened", *base64_order]))
            dynamic_hashes.append(_feature_projection_hash(reference, ["Opened", *dynamic_names]))

            base64_values = rebuilt_base64.loc[:, base64_order].to_numpy(dtype=np.float64)
            canonical_base = canonical.loc[:, base_order].to_numpy(dtype=np.float64)
            compare_started = time.perf_counter()
            for index, name in enumerate(base64_order):
                _parity_update(
                    parity, name, base64_values[:, index:index + 1],
                    canonical_base[:, index:index + 1], opened, atol=1e-10, rtol=1e-10,
                )

            flow_lookup = np.searchsorted(flow_opened_ns, opened_ns)
            flow_match = flow_lookup < len(flow_opened_ns)
            flow_match[flow_match] &= flow_opened_ns[flow_match] == opened_ns[flow_match]
            base_values = np.full((count, len(base_order)), np.nan, dtype=np.float64)
            base_values[:, :64] = base64_values
            if flow_match.any():
                base_values[np.flatnonzero(flow_match)[:, None], flow_indices] = flow_matrix[flow_lookup[flow_match]]
            flow_reference_covered = opened <= flow_last_reference
            if flow_reference_covered.any():
                for index, name in enumerate(FLOW_COLUMNS):
                    col_index = flow_indices[index]
                    _parity_update(
                        parity, name,
                        base_values[flow_reference_covered, col_index:col_index + 1],
                        canonical_base[flow_reference_covered, col_index:col_index + 1],
                        opened[flow_reference_covered], atol=1e-10, rtol=1e-10,
                    )
            base_comparison_seconds += time.perf_counter() - compare_started

            current_raw = raw.loc[:, RAW_COLUMNS].copy()
            rolling = current_raw if raw_tail.empty else pd.concat([raw_tail, current_raw], ignore_index=True)
            rolling_opened = pd.DatetimeIndex(pd.to_datetime(rolling["Opened"], utc=True, errors="raise"))
            if rolling_opened.has_duplicates or not rolling_opened.is_monotonic_increasing:
                raise RuntimeError("rolling raw tail duplicates or reorders the next input chunk")
            dynamic_started = time.perf_counter()
            dynamic = generate_dynamic_block(state, rolling, new_rows_start=len(rolling) - count)
            expected_memory = reference.loc[:, memory_names].to_numpy(dtype=np.float32)
            for index, name in enumerate(memory_names):
                _parity_update(
                    parity, name, dynamic["memory"][:, index:index + 1],
                    expected_memory[:, index:index + 1], opened, atol=2e-6, rtol=2e-6,
                )
            generated_esn = {}
            canonical_esn = {}
            for seed in reservoir_seeds:
                names = list(transforms["esn"][seed]["feature_names"])
                expected_esn = reference.loc[:, names].to_numpy(dtype=np.float32)
                generated_esn[seed] = dynamic[f"esn_{seed}"]
                canonical_esn[seed] = expected_esn
                for index, name in enumerate(names):
                    _parity_update(
                        parity, name, dynamic[f"esn_{seed}"][:, index:index + 1],
                        expected_esn[:, index:index + 1], opened, atol=2e-6, rtol=2e-6,
                    )
            dynamic_comparison_seconds += time.perf_counter() - dynamic_started
            invalid_candles += int((~_valid_candles(current_raw)).sum())

            model_cutoff = protocol["scope"]["last_allowed_model_feature_opened_utc"]
            model_rows = opened <= pd.Timestamp(model_cutoff).tz_convert("UTC")
            if model_rows.any():
                canonical_memory = reference.loc[:, memory_names].to_numpy(dtype=np.float32)
                prediction_started = time.perf_counter()
                panel.extend(_prediction_panel(
                    base_values[model_rows], dynamic["memory"][model_rows],
                    {seed: values[model_rows] for seed, values in generated_esn.items()},
                    canonical_base[model_rows], canonical_memory[model_rows],
                    {seed: values[model_rows] for seed, values in canonical_esn.items()},
                    opened[model_rows], d_manifest["members"], boosters, calibrator,
                    windows, model_cutoff, parity,
                ))
                prediction_seconds += time.perf_counter() - prediction_started

            raw_tail = rolling.tail(720).reset_index(drop=True)
            flow_tail_end = int(np.searchsorted(flow_source_ns, opened_ns[-1], side="right"))
            flow_tail_begin = int(np.searchsorted(flow_source_ns, opened_ns[-1] - FLOW_TAIL_MINUTES * MINUTE_NS, side="left"))
            flow_tail = flow_raw.iloc[flow_tail_begin:flow_tail_end].reset_index(drop=True)
            flow_tail_rows = len(flow_tail)
            byte_offset = stream.tell()
            dynamic_checkpoint = {
                "identity_sha256": source_identity,
                "source_row_index": batch_start + count,
                "source_byte_offset": byte_offset,
                "last_processed_opened": opened[-1].isoformat(),
                "next_expected_opened": (opened[-1] + pd.Timedelta(minutes=1)).isoformat(),
                "timestamp_gaps": gap_count,
                "invalid_candles": invalid_candles,
                "dynamic_state": _dynamic_checkpoint_payload(state),
                "raw_tail": _encode_frame(raw_tail),
                "flow_tail": _encode_frame(flow_tail),
                "flow_tail_max_history_minutes": FLOW_TAIL_MINUTES + 1,
                "base64_mode": base64_build["mode"],
                "base64_output_sha256": base64_build["output_sha256"],
                "parity": parity,
                "prediction_panel": panel,
                "base_projection_block_hashes": projection_hashes,
                "dynamic_projection_block_hashes": dynamic_hashes,
                "source_projection_sha256": _canonical_sha256(projection_hashes),
                "complete": False,
            }
            checkpoint_started = time.perf_counter()
            _atomic_checkpoint(CHECKPOINT_PATH, {
                "schema_version": 1,
                "stage": "dynamic_partial",
                "run_identity_sha256": run_identity,
                "dynamic_stage": dynamic_checkpoint,
                "flow_stage": checkpoint.get("flow_stage") if checkpoint else None,
            })
            checkpoint_io_seconds += time.perf_counter() - checkpoint_started
            if stop_after_opened_utc and opened[-1] >= pd.Timestamp(stop_after_opened_utc).tz_convert("UTC"):
                break

    complete = int(dynamic_checkpoint["source_row_index"]) == expected_rows
    if complete:
        if any(reader.row != reader.total_rows for reader in (base_reader, canonical_reader, dynamic_reader)):
            raise RuntimeError("development replay did not consume the complete projected sources")
        last_opened = pd.Timestamp(raw_open_ns_previous, unit="ns", tz="UTC")
        if last_opened != pd.Timestamp(raw_record["last_opened_utc"]):
            raise RuntimeError("final raw timestamp differs from the frozen development prefix")
        dynamic_checkpoint["complete"] = True
        dynamic_checkpoint["final_checkpoint_state_sha256"] = _canonical_sha256(_dynamic_checkpoint_payload(state))

    panel_path = RUNTIME_DIR / "development_prediction_panel.json"
    _write_small_json(panel_path, {
        "all_selected_rows": panel,
        "decision_rows": [record for record in panel if record["is_decision_row"]],
        "rows": len(panel),
        "decision_rows_count": sum(record["is_decision_row"] for record in panel),
        "calibrator_coefficients_only": calibrator,
    })
    checkpoint_started = time.perf_counter()
    _atomic_checkpoint(CHECKPOINT_PATH, {
        "schema_version": 1,
        "stage": "development_complete" if complete else "dynamic_partial",
        "run_identity_sha256": run_identity,
        "dynamic_stage": dynamic_checkpoint,
        "flow_stage": checkpoint.get("flow_stage") if checkpoint else None,
    })
    checkpoint_io_seconds += time.perf_counter() - checkpoint_started

    parity_rows = []
    for name, record in parity.items():
        block = "prediction" if name.startswith("member_") or name.startswith("D_") else (
            "dynamic" if name.startswith("simple_memory") or name.startswith("esn_") else "base70"
        )
        tolerance = 2e-6 if block == "dynamic" else (2e-12 if block == "prediction" else 1e-10)
        _report_parity_row(parity_rows, block, name, record, tolerance, tolerance)
    return {
        "parity": parity_rows,
        "parity_summary": parity,
        "panel_path": panel_path.relative_to(ROOT).as_posix(),
        "panel_rows": len(panel),
        "decision_panel_rows": sum(record["is_decision_row"] for record in panel),
        "checkpoint_path": CHECKPOINT_PATH.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        "processed_rows": int(dynamic_checkpoint["source_row_index"]),
        "start_row": start_row,
        "first_opened_this_call": first_opened_this_call,
        "suffix_rows_processed": int(dynamic_checkpoint["source_row_index"]) - start_row,
        "complete": complete,
        "first_opened": raw_record["first_opened_utc"],
        "last_opened": dynamic_checkpoint["last_processed_opened"],
        "last_allowed_opened": raw_record["last_opened_utc"],
        "last_model_feature_opened": protocol["scope"]["last_allowed_model_feature_opened_utc"],
        "timestamp_gaps": gap_count,
        "invalid_candles": invalid_candles,
        "projected_base64_sha256": _canonical_sha256(projection_hashes),
        "projected_dynamic_reference_sha256": _canonical_sha256(dynamic_hashes),
        "flow_tail_rows": int(flow_tail_rows),
        "base_features_missing_from_reference": [],
        "prediction_status": "GENERATED_AND_COMPARED" if complete else "IN_PROGRESS",
        "timings": {
            "replay_seconds": time.perf_counter() - replay_started,
            "base_comparison_seconds": base_comparison_seconds,
            "dynamic_generation_and_comparison_seconds": dynamic_comparison_seconds,
            "prediction_seconds": prediction_seconds,
            "checkpoint_io_seconds": checkpoint_io_seconds,
        },
    }

def build_feature_contract(protocol: dict, d_manifest: dict, transforms: dict,
                           bundle_identity: dict, base64_build: dict,
                           dynamic_result: dict) -> dict:
    """Write the ordered, source-linked final E0041 development feature contract."""
    import create_modeling_dataset as dataset_builder

    base_order = list(d_manifest["base_feature_order"])
    base64_order = base_order[:64]
    memory_inputs = list(transforms["memory"]["identity"]["input_columns"])
    lineage = protocol["allowed_inputs"]["base_feature_lineage"]
    import pyarrow.parquet as pq
    canonical_schema = pq.ParquetFile(
        ROOT / protocol["allowed_inputs"]["canonical_model_feature_reference"]["path"]
    ).schema_arrow
    feature_metadata = json.loads(
        (ROOT / lineage["feature_metadata"]["path"]).read_text(encoding="utf-8")
    )
    fit_configs = dataset_builder.parse_fit_results(
        ROOT / json.loads((ROOT / lineage["indicator_stage"]["path"]).read_text(encoding="utf-8"))["fit_results_dir"]
    )
    fit_by_name = {item["feature_col"]: item for item in fit_configs}

    def feature_lineage(name: str, position: int) -> dict:
        if name in FLOW_COLUMNS:
            return {
                "position": position,
                "name": name,
                "block": "flow6",
                "dtype_from_rebuild": "float64",
                "dtype_in_canonical_reference": str(canonical_schema.field(name).type),
                "inference_dtype": "float32",
                "source_fields": list(FLOW_RAW_COLUMNS),
                "implementation": "features/btc_taker_flow.py",
                "configuration": protocol["allowed_inputs"]["taker_flow_raw_archives"],
                "history_and_missing_semantics": "Frozen rolling Q-bar, volatility, beta windows; missing minutes are retained without forward fill; invalid source fields invalidate their dependent windows.",
                "timestamp": "Opened equals Binance kline open_time.",
            }
        if name in fit_by_name:
            fit = fit_by_name[name]
            fit_path = fit["json_path"].as_posix()
            fit_record = next((record for record in lineage["indicator_fit_results"] if record["path"] == fit_path), None)
            fields_by_family = {
                "KeltnerChannel": ["High", "Low", "Close"],
                "ADX": ["High", "Low", "Close"],
                "BollingerBands": ["Close"],
                "ChaikinOsc": ["High", "Low", "Close", "Volume"],
                "MACD": ["Close"],
                "StochOsc": ["High", "Low", "Close"],
            }
            family = name.split("_fit_", 1)[0]
            return {
                "position": position,
                "name": name,
                "block": "base64",
                "dtype_from_rebuild": base64_build["feature_dtypes"][name],
                "dtype_in_canonical_reference": str(canonical_schema.field(name).type),
                "inference_dtype": "float32",
                "source_fields": fields_by_family.get(family, []),
                "implementation": f"create_modeling_dataset.add_indicator_values via features/{family}.py",
                "fit_result": fit_record,
                "saved_parameters": {key: value for key, value in fit.items() if key not in {"json_path"}},
                "history_and_missing_semantics": "Uses the exact saved E0041 fit-result parameters and implementation warmup/NaN behavior; no new fit.",
                "timestamp": "Opened.",
            }
        if name.startswith("futures_index_basis_"):
            source_fields = ["Close", "UM_BTCUSDT_Close"]
            implementation = "features/basis_premium_features.py"
            history = "Uses the configured interval endpoints, epsilon, and causal Opened-indexed price changes; source warmup gaps remain missing."
        elif name.startswith(("rp_", "vp_")):
            source_fields = ["Open", "High", "Low", "Close", "Volume"]
            implementation = "features/reaction_profile_fixed_grid.py" if name.startswith("rp_") else "features/volume_profile_fixed_range.py"
            history = "Uses the saved E0041 fitted generator configuration and serialized state; state is initialized from the safe prefix and no forward fill is applied."
        elif name.startswith("realized_volatility_"):
            source_fields = ["Close"]
            implementation = "features/realized_volatility.py"
            history = "Uses exact named intervals and rolling volatility windows; warmup values remain missing."
        elif name.startswith("session_"):
            source_fields = ["Opened", "Open", "High", "Low", "Close", "Volume"]
            implementation = "features/session_open_features.py"
            history = "Uses UTC session boundaries and causal session-to-date candles; unavailable session history remains missing."
        else:
            source_fields = ["Opened", "Open", "High", "Low", "Close", "Volume"]
            implementation = "features/candle_features.py"
            history = "Uses the configured candle intervals and lag suffix in the feature name; no interpolation or forward fill."
        return {
            "position": position,
            "name": name,
            "block": "base64",
            "dtype_from_rebuild": base64_build["feature_dtypes"][name],
            "dtype_in_canonical_reference": str(canonical_schema.field(name).type),
            "inference_dtype": "float32",
            "source_fields": source_fields,
            "implementation": implementation,
            "configuration": (
                feature_metadata["basis_premium_features"] if name.startswith("futures_index_basis_")
                else feature_metadata["feature_intervals"]
            ),
            "history_and_missing_semantics": history,
            "timestamp": "Opened.",
        }

    old_features = protocol["diagnostics"]["old_e0044_required_features"]
    old_actuals = protocol["diagnostics"]["old_feature_replacements"]
    legacy_diagnosis = [
        {
            "old_loader_feature": name,
            "required_by_final_E0041": name in base_order,
            "final_E0041_same_family_feature": old_actuals.get(name),
            "alias_or_zero_fill_used": False,
        }
        for name in old_features
    ]
    parity_summary = dynamic_result["parity_summary"]
    return {
        "schema_version": 1,
        "experiment_id": "E0045",
        "task_id": protocol["task_id"],
        "decision_id": protocol["decision_id"],
        "bundle_manifest_sha256": protocol["parent"]["bundle_manifest_sha256"],
        "bundle_identity_sha256": _canonical_sha256(bundle_identity),
        "holdout_values_authorized": False,
        "evaluation_authorized": False,
        "activated": False,
        "holdout_status": "HOLDOUT_NOT_RUN",
        "opened_cutoff_exclusive_utc": protocol["scope"]["source_raw_opened_exclusive_utc"],
        "time_mapping": {
            "available_at": "Opened + 1 minute",
            "market_start_utc": "Opened + 2 minutes",
            "decision": "T-60",
            "execution": "T-59",
            "last_model_feature_opened_utc": protocol["scope"]["last_allowed_model_feature_opened_utc"],
        },
        "legacy_loader_diagnosis": legacy_diagnosis,
        "actual_missing_final_E0041_features": [],
        "base70": {
            "order": base_order,
            "base64": base64_order,
            "flow6": base_order[64:],
            "features": [feature_lineage(name, index) for index, name in enumerate(base_order)],
            "raw_reconstruction": {
                "status": "PASS" if all(parity_summary.get(name, {}).get("numeric_mismatch_rows", 1) == 0 and parity_summary.get(name, {}).get("mask_mismatch_rows", 1) == 0 for name in base64_order) else "FAIL",
                "mode": base64_build["mode"],
                "rows": base64_build["rows"],
                "output_path": base64_build["output_path"],
                "output_sha256": base64_build["output_sha256"],
                "feature_dtypes": base64_build["feature_dtypes"],
                "profile_states": base64_build["profile_states"],
                "stateful_continuation": "NOT_VERIFIED: E0041 RP/VP path is a full-prefix batch builder without a streaming checkpoint interface.",
                "zero_padding_or_aliasing": False,
            },
            "canonical_reference": protocol["allowed_inputs"]["canonical_model_feature_reference"],
        },
        "profile_generators": {
            "reaction_profile": {
                "config": lineage["reaction_profile_config"],
                "final_state": lineage["reaction_profile_state"],
                "implementation": "features/reaction_profile_fixed_grid.py",
            },
            "volume_profile": {
                "config": lineage["volume_profile_config"],
                "final_state": lineage["volume_profile_state"],
                "implementation": "features/volume_profile_fixed_range.py",
            },
            "indicator_fit_results": lineage["indicator_fit_results"],
            "fit_or_transform_refit_performed": False,
        },
        "flow6": {
            "order": list(FLOW_COLUMNS),
            "source_columns": list(FLOW_RAW_COLUMNS),
            "source_code": {
                "path": "features/btc_taker_flow.py",
                "sha256": protocol["allowed_inputs"]["feature_implementation_hashes"]["features/btc_taker_flow.py"],
                "parameters": _flow_parameters(json.loads((ROOT / protocol["allowed_inputs"]["taker_flow_raw_archives"]["source_config_path"]).read_text(encoding="utf-8"))),
            },
            "raw_source": protocol["allowed_inputs"]["taker_flow_raw_archives"],
            "canonical_reference": protocol["allowed_inputs"]["canonical_flow_reference"],
            "unmatched_rebuilt_suffix_rows": protocol["diagnostics"]["unmatched_rebuilt_flow_suffix_rows"],
        },
        "dynamic_raw_inputs": {
            "order": memory_inputs,
            "raw_dependencies": ["Opened", "Open", "High", "Low", "Close", "Volume"],
            "raw_implementation": {
                "path": "features/esn_reservoir.py",
                "sha256": protocol["allowed_inputs"]["feature_implementation_hashes"]["features/esn_reservoir.py"],
                "function": "build_causal_esn_inputs",
            },
            "dtype_before_transform": "float64 raw inputs; float32 ESN stream; frozen memory scaler/state",
            "history_minutes": 720,
            "missing_policy": "Frozen FeatureScaler imputation; memory/ESN reset on invalid OHLCV or a non-one-minute gap; no interpolation or forward fill.",
            "memory_transform": protocol["allowed_inputs"]["frozen_transforms"]["memory"],
            "esn_transforms_by_seed": protocol["allowed_inputs"]["frozen_transforms"]["esn"],
            "canonical_reference": protocol["allowed_inputs"]["canonical_dynamic_reference"],
            "replay_status": "PASS" if all(row["status"] == "PASS" for row in dynamic_result["parity"] if row["block"] == "dynamic") else "FAIL",
        },
        "d_members": [
            {
                "order": index,
                "reservoir_seed": int(member["reservoir_seed"]),
                "lightgbm_seed": int(member["lightgbm_seed"]),
                "feature_count": int(member["num_features"]),
                "tree_count": int(member["num_trees"]),
                "feature_order": list(member["feature_order"]),
                "feature_order_sha256": member["feature_order_sha256"],
                "model_sha256": member["model_sha256"],
                "metadata_sha256": member["metadata_sha256"],
                "input_dtype": "float32; nonfinite entries are converted to NaN as in the E0041 finalizer",
            }
            for index, member in enumerate(d_manifest["members"])
        ],
        "prediction": {
            "aggregation": "uniform arithmetic mean in final manifest member order",
            "calibrator": protocol["allowed_inputs"]["frozen_calibrator"],
            "calibrator_application": "clip raw mean to [1e-6, 1-1e-6]; logit=log(p/(1-p)); calibrated=sigmoid(intercept + coefficient*logit)",
            "panel_rows": dynamic_result["panel_rows"],
            "decision_panel_rows": dynamic_result["decision_panel_rows"],
            "parity_status": "PASS" if all(row["status"] == "PASS" for row in dynamic_result["parity"] if row["block"] == "prediction") else "FAIL",
            "reference": "Same nine final E0041 models applied directly to the canonical E0041 BASE70 and dynamic feature rows.",
        },
        "continuation": {
            "dynamic_suffix": "VERIFIED" if dynamic_result["complete"] and dynamic_result["suffix_rows_processed"] > 0 else "NOT_VERIFIED",
            "base64": "BATCH_ONLY",
            "full_generator": "NOT_VERIFIED",
        },
        "implementation_hashes": protocol["allowed_inputs"]["feature_implementation_hashes"],
    }
