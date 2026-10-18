"""Fit and checkpoint frozen BASE64/BASE70 holdout models, then validate local inputs."""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_preopen_experiment as experiment
import run_btc_preopen_taker_flow_20261018 as taker


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_base70_raw_holdout_20261018.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_directory"]
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
VERSIONED_MANIFEST_PATH = ROOT / CONFIG["versioned_manifest_path"]
REPORT_PATH = ROOT / CONFIG["report_path"]
FLOW_CACHE_ROOT = ROOT / "data/datasets/_tmp/binance_vision/um/klines/BTCUSDT1m/days"

MATRIX_PATH = ROOT / CONFIG["training_matrix_path"]
TARGET_COLUMN = CONFIG["shared_training_data"]["target"]
BASE64_NAME = "BASE64"
BASE70_NAME = "BASE70"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _record(path: Path) -> dict:
    return {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_manifests(manifest: dict) -> None:
    manifest["updated_utc"] = _now_utc()
    _write_json(MANIFEST_PATH, manifest)
    _write_json(VERSIONED_MANIFEST_PATH, manifest)


def _verify_frozen_files(manifest: dict) -> None:
    specification = manifest["frozen_comparison"]["specification"]
    records = specification["input_artifacts"] + specification["source_artifacts"]
    for record in records:
        path = ROOT / record["path"]
        if not path.is_file():
            raise RuntimeError(f"Frozen input is missing: {record['path']}")
        if "size_bytes" in record and path.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(f"Frozen input size changed: {record['path']}")
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Frozen input SHA-256 changed: {record['path']}")


def _read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _source_summaries() -> tuple[dict, dict, dict, dict]:
    zv = _read_json("data/analysis/polymarket/BTC/btc_preopen_zv_ablation_20261018/run_manifest.json")
    flow = _read_json(CONFIG["source_taker_flow_manifest_path"])
    full = _read_json(CONFIG["source_full_development_run_manifest_path"])
    if zv.get("status") != "completed" or flow.get("status") != "completed" or full.get("status") != "completed":
        raise RuntimeError("One or more frozen source experiments are incomplete")
    base64 = zv["stages"]["cv_BASE64"]["summary"]
    base70 = flow["stages"]["cv_B0"]["summary"]
    common = flow["stages"]["common_rows"]["summary"]
    return base64, base70, common, full


def _model_specs(base64_summary: dict, base70_summary: dict) -> dict[str, dict]:
    specs = {
        BASE64_NAME: {
            "feature_order": list(base64_summary["feature_order"]),
            "params": dict(base64_summary["params"]),
            "iterations": int(base64_summary["best_iteration"]),
        },
        BASE70_NAME: {
            "feature_order": list(base70_summary["feature_order"]),
            "params": dict(base70_summary["params"]),
            "iterations": int(base70_summary["best_iteration"]),
        },
    }
    for name, spec in specs.items():
        frozen = CONFIG["models"][name]
        if len(spec["feature_order"]) != int(frozen["feature_count"]):
            raise RuntimeError(f"{name} feature count differs from the frozen config")
        if spec["iterations"] != int(frozen["iterations"]):
            raise RuntimeError(f"{name} iteration count differs from the frozen config")
        if spec["params"] != frozen["effective_lightgbm_params"]:
            raise RuntimeError(f"{name} effective parameters differ from the frozen config")
    if specs[BASE70_NAME]["feature_order"][:64] != specs[BASE64_NAME]["feature_order"]:
        raise RuntimeError("BASE70 does not begin with the exact ordered BASE64 feature set")
    if specs[BASE70_NAME]["feature_order"][64:] != CONFIG["holdout_features"]["features"]:
        raise RuntimeError("BASE70 does not end with the six frozen raw Z/V features in the configured order")
    return specs


def _find_zv_record(source_manifest: dict, path: str) -> dict:
    artifacts = source_manifest["frozen_comparison"]["specification"]["input_artifacts"]
    record = next((item for item in artifacts if item["path"] == path), None)
    if record is None:
        raise RuntimeError(f"Source manifest does not record required input: {path}")
    return dict(record)


def _freeze_manifest(specs: dict[str, dict], common_summary: dict, full_run: dict) -> dict:
    source_zv = _read_json(CONFIG["source_zv_manifest_path"])
    source_flow_path = ROOT / CONFIG["source_taker_flow_manifest_path"]
    info_manifest = _read_json("reports/btc_preopen/btc_base70_information_value_20261018_manifest.json")
    info_sources = info_manifest["frozen_comparison"]["specification"]["sources"]
    snapshots = info_sources["decision_and_execution_snapshots"]
    input_records = [
        _find_zv_record(source_zv, CONFIG["training_matrix_path"]),
        _find_zv_record(source_zv, CONFIG["training_folds_path"]),
        _find_zv_record(source_zv, CONFIG["base_feature_dataset_path"]),
        _find_zv_record(source_zv, CONFIG["flow_features_path"]),
        _find_zv_record(source_zv, CONFIG["official_market_index_path"]),
    ]
    source_records = [
        {"path": CONFIG["source_zv_config_path"], "sha256": _sha256(ROOT / CONFIG["source_zv_config_path"])},
        {"path": CONFIG["source_zv_manifest_path"], "sha256": _sha256(ROOT / CONFIG["source_zv_manifest_path"])},
        {"path": CONFIG["source_taker_flow_config_path"], "sha256": _sha256(ROOT / CONFIG["source_taker_flow_config_path"])},
        {"path": CONFIG["source_taker_flow_manifest_path"], "sha256": _sha256(source_flow_path)},
        {"path": CONFIG["source_full_development_bundle_manifest_path"], "sha256": _sha256(ROOT / CONFIG["source_full_development_bundle_manifest_path"])},
        {"path": CONFIG["source_full_development_run_manifest_path"], "sha256": _sha256(ROOT / CONFIG["source_full_development_run_manifest_path"])},
        {"path": "features/btc_taker_flow.py", "sha256": _sha256(ROOT / "features/btc_taker_flow.py")},
        {"path": "reports/btc_preopen/btc_base70_information_value_20261018_manifest.json", "sha256": _sha256(ROOT / "reports/btc_preopen/btc_base70_information_value_20261018_manifest.json")},
    ]
    for key, filename in (("t59", "t59_path"), ("t60", "t60_path")):
        path = snapshots[filename]
        input_records.append({
            "path": path,
            "sha256": snapshots[f"{key}_sha256"],
            "size_bytes": (ROOT / path).stat().st_size,
        })

    frozen_spec = {
        "experiment_id": CONFIG["experiment_id"],
        "period": {
            "development_cutoff_exclusive_utc": CONFIG["development_cutoff_exclusive_utc"],
            "holdout_start_inclusive_utc": CONFIG["holdout_market_start_first_inclusive_utc"],
            "holdout_last_market_start_inclusive_utc": CONFIG["holdout_market_start_last_inclusive_utc"],
            "data_cutoff_exclusive_utc": CONFIG["data_cutoff_exclusive_utc"],
            "historically_exposed": True,
        },
        "models": specs,
        "shared_training_data": {
            "path": CONFIG["training_matrix_path"],
            "rows": int(common_summary["rows_after_shared_complete_feature_mask"]),
            "decision_rows": int(common_summary["decision_rows_after_shared_complete_feature_mask"]),
            "latest_label_available_at_utc": common_summary["latest_label_available_at_utc"],
            "first_holdout_decision_at_utc": "2026-06-30T23:59:00+00:00",
            "common_mask": common_summary["common_mask"],
            "training_weight_sha256": common_summary["source_training_weight_sha256"],
            "expected_training_weight_sha256": CONFIG["shared_training_data"]["training_weight_sha256"],
            "purge_validated": True,
        },
        "existing_full_development_fit_reuse": {
            "reused": False,
            "reason": "The frozen full-development fit does not use the exact shared Z/V complete-feature row mask; the Z/V comparison explicitly records that its saved full-development BASE64 artifact is incompatible with these rows.",
            "source_status": full_run.get("status"),
        },
        "holdout_feature_transform": {
            "base_transform_provenance": "full-development pipeline; generators, feature selection, weights, and model tuning were fit on development only",
            "taker_flow_code": "features/btc_taker_flow.py",
            "taker_flow_config": "configs/research/btc_preopen_taker_flow_20261018.json",
            "causal_alignment": "Opened row uses completed information through that row; nominal decision is Opened+1m and market start is Opened+2m",
            "features": CONFIG["holdout_features"]["features"],
            "required_raw_fields": CONFIG["holdout_features"]["raw_required_fields"],
        },
        "quality": CONFIG["quality"],
        "economics": {
            **CONFIG["economics"],
            "snapshot_inputs": {"t59": snapshots["t59_path"], "t60": snapshots["t60_path"]},
        },
        "input_artifacts": input_records,
        "source_artifacts": source_records,
        "no_data_download": True,
        "no_archive_replay": True,
        "no_deployment_activation": True,
    }
    return {
        "schema_version": 1,
        "experiment_id": CONFIG["experiment_id"],
        "status": "frozen_pre_score",
        "frozen_at_utc": _now_utc(),
        "frozen_comparison": {
            "identity_sha256": _canonical_sha256(frozen_spec),
            "specification": frozen_spec,
        },
        "input_identity": {
            "config_sha256": _sha256(CONFIG_PATH),
            "runner_sha256": _sha256(Path(__file__)),
            "python": __import__("sys").version,
            "lightgbm": taker.lgb.__version__,
        },
        "resource_snapshot_before_fit": taker._resource_snapshot(),
        "stages": {"freeze": {"status": "completed", "finished_utc": _now_utc()}},
        "holdout_predictions_read": False,
        "holdout_scoring_started": False,
        "outputs": [],
        "started_utc": _now_utc(),
        "updated_utc": _now_utc(),
    }


def _load_training_data(specs: dict[str, dict]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    features = specs[BASE70_NAME]["feature_order"]
    columns = [
        "_source_row_index", "Opened", "nominal_decision_at", "target_label_available_at",
        TARGET_COLUMN, "is_polymarket_preopen_decision", *features,
    ]
    frame = pd.read_parquet(MATRIX_PATH, columns=columns)
    if len(frame) != int(CONFIG["shared_training_data"]["expected_rows"]):
        raise RuntimeError("Shared training row count differs from the frozen Z/V input")
    if frame["_source_row_index"].duplicated().any():
        raise RuntimeError("Shared training matrix contains duplicate source rows")
    labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8, copy=True)
    if not np.isin(labels, (0, 1)).all():
        raise RuntimeError("Shared training target is not binary")
    available = pd.to_datetime(frame["target_label_available_at"], utc=True, errors="raise")
    nominal = pd.to_datetime(frame["nominal_decision_at"], utc=True, errors="raise")
    decision_cutoff = pd.Timestamp("2026-06-30T23:59:00Z")
    if available.max() > decision_cutoff or nominal.max() > decision_cutoff:
        raise RuntimeError("Training includes a label or decision later than the first holdout decision")
    if available.max().isoformat() != "2026-06-30T23:59:00+00:00":
        raise RuntimeError("Training purge endpoint no longer matches the frozen shared-mask summary")
    decision = frame["is_polymarket_preopen_decision"].to_numpy(dtype=bool, copy=True)
    if int(decision.sum()) != int(CONFIG["shared_training_data"]["expected_decision_rows"]):
        raise RuntimeError("Shared training decision-row count differs from the frozen source")
    weights = experiment._preopen_sample_weights(decision, float(CONFIG["shared_training_data"]["decision_weight"]))
    weights_hash = taker._array_sha256(weights, dtype=np.float32)
    expected_hash = CONFIG["shared_training_data"]["training_weight_sha256"]
    if weights_hash != expected_hash:
        raise RuntimeError(f"Training weights do not match the frozen source hash: {weights_hash}")
    x = frame.loc[:, features].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(x).all():
        raise RuntimeError("Shared complete-feature training matrix contains non-finite values")
    return frame, x, labels, weights


def _model_paths(name: str) -> tuple[Path, Path]:
    model = OUTPUT_DIR / "models" / f"{name}.txt"
    completion = OUTPUT_DIR / "models" / f"{name}_completion.json"
    return model, completion


def _verified_model_checkpoint(manifest: dict, name: str) -> bool:
    record = manifest.get("stages", {}).get("training", {}).get("models", {}).get(name)
    if not record or record.get("status") != "completed":
        return False
    model_path, completion_path = _model_paths(name)
    return (
        model_path.is_file()
        and completion_path.is_file()
        and _sha256(model_path) == record.get("model", {}).get("sha256")
        and _sha256(completion_path) == record.get("completion", {}).get("sha256")
        and record.get("identity_sha256") == manifest["frozen_comparison"]["identity_sha256"]
    )


def _fit_models(manifest: dict, frame: pd.DataFrame, x: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> int:
    specs = manifest["frozen_comparison"]["specification"]["models"]
    manifest.setdefault("stages", {}).setdefault("training", {"status": "running", "models": {}})
    manifest["stages"]["training"]["status"] = "running"
    _write_manifests(manifest)
    new_fits = 0
    for name in (BASE64_NAME, BASE70_NAME):
        if _verified_model_checkpoint(manifest, name):
            print(f"Reused verified {name} full-development model checkpoint.")
            continue
        spec = specs[name]
        names = list(spec["feature_order"])
        idx = np.arange(len(names), dtype=np.int32)
        model_x = x[:, idx] if name == BASE70_NAME else x[:, :64]
        started = time.perf_counter()
        with taker.PeakMemorySampler() as memory:
            train_set = taker.lgb.Dataset(
                model_x,
                label=labels,
                weight=weights,
                feature_name=names,
                params=spec["params"],
                free_raw_data=False,
            )
            train_set.construct()
            stored_weights = train_set.get_weight()
            if stored_weights is None or not np.allclose(stored_weights, weights, rtol=2e-6, atol=1e-8):
                raise RuntimeError(f"LightGBM changed the explicit {name} observation weights")
            booster = taker.lgb.train(
                params=spec["params"],
                train_set=train_set,
                num_boost_round=int(spec["iterations"]),
                keep_training_booster=False,
            )
        elapsed = time.perf_counter() - started
        if booster.current_iteration() != int(spec["iterations"]):
            raise RuntimeError(f"{name} fit did not produce the frozen iteration count")
        if list(booster.feature_name()) != names:
            raise RuntimeError(f"{name} model feature order differs from the frozen order")
        model_path, completion_path = _model_paths(name)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_suffix(".txt.tmp")
        booster.save_model(str(temporary), num_iteration=int(spec["iterations"]))
        temporary.replace(model_path)
        model_record = _record(model_path)
        completion = {
            "identity_sha256": manifest["frozen_comparison"]["identity_sha256"],
            "model_name": name,
            "model_sha256": model_record["sha256"],
            "feature_order_sha256": _canonical_sha256(names),
            "feature_count": len(names),
            "iterations": int(booster.current_iteration()),
            "effective_params": spec["params"],
            "training_rows": int(len(frame)),
            "decision_rows": int(frame["is_polymarket_preopen_decision"].sum()),
            "training_weight_sha256": taker._array_sha256(weights, dtype=np.float32),
            "latest_training_label_available_at_utc": pd.to_datetime(
                frame["target_label_available_at"], utc=True,
            ).max().isoformat(),
            "elapsed_seconds": elapsed,
            "peak_process_working_set_gib": round(memory.peak_bytes / 2**30, 3),
            "device_type": spec["params"]["device_type"],
            "holdout_rows_used": 0,
            "model": model_record,
        }
        _write_json(completion_path, completion)
        manifest["stages"]["training"]["models"][name] = {
            "status": "completed",
            "identity_sha256": manifest["frozen_comparison"]["identity_sha256"],
            "iterations": completion["iterations"],
            "elapsed_seconds": elapsed,
            "model": model_record,
            "completion": _record(completion_path),
        }
        new_fits += 1
        manifest["stages"]["training"]["finished_models"] = list(manifest["stages"]["training"]["models"])
        _write_manifests(manifest)
        print(f"Fitted {name} for {completion['iterations']} iterations in {elapsed:.1f}s.")
        del train_set, booster, model_x
    manifest["stages"]["training"]["status"] = "completed"
    manifest["stages"]["training"]["finished_utc"] = _now_utc()
    manifest["stages"]["training"]["summary"] = {
        "new_fits_this_invocation": new_fits,
        "both_models_complete": all(_verified_model_checkpoint(manifest, name) for name in (BASE64_NAME, BASE70_NAME)),
    }
    _write_manifests(manifest)
    return new_fits


def _missing_holdout_flow_files(days_root: Path, *, start: date, end_inclusive: date) -> list[str]:
    missing = []
    current = start
    while current <= end_inclusive:
        day = current.isoformat()
        path = days_root / day / f"BTCUSDT-1m-{day}.csv"
        if not path.is_file():
            try:
                missing.append(path.resolve().relative_to(ROOT).as_posix())
            except ValueError:
                missing.append(path.as_posix())
        current += timedelta(days=1)
    return missing


def _local_um_kline_inventory() -> list[dict]:
    root = FLOW_CACHE_ROOT.parent
    expected = [
        *(root / "months" / month / f"BTCUSDT-1m-{month}.csv" for month in ("2026-06", "2026-07", "2026-08", "2026-09")),
        *(FLOW_CACHE_ROOT / day / f"BTCUSDT-1m-{day}.csv" for day in ("2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06")),
    ]
    inventory = []
    for path in expected:
        if path.is_file():
            inventory.append({"status": "available", **_record(path)})
        else:
            inventory.append({"status": "missing", "path": path.resolve().relative_to(ROOT).as_posix()})
    return inventory


def _flow_cache_last_opened() -> str:
    import pyarrow.parquet as pq

    path = ROOT / CONFIG["flow_features_path"]
    parquet = pq.ParquetFile(path)
    column_index = parquet.schema_arrow.get_field_index("Opened")
    maxima = [parquet.metadata.row_group(i).column(column_index).statistics.max for i in range(parquet.metadata.num_row_groups)]
    return max(maxima).replace(tzinfo=timezone.utc).isoformat()


def _check_holdout_flow_coverage(manifest: dict) -> dict:
    missing = _missing_holdout_flow_files(
        FLOW_CACHE_ROOT,
        start=date(2026, 10, 2),
        end_inclusive=date(2026, 10, 6),
    )
    coverage = {
        "cached_flow_features_last_opened_utc": _flow_cache_last_opened(),
        "local_um_kline_inventory": _local_um_kline_inventory(),
        "required_raw_fields": CONFIG["holdout_features"]["raw_required_fields"],
        "holdout_market_start_last_inclusive_utc": CONFIG["holdout_market_start_last_inclusive_utc"],
        "last_required_opened_row_utc": "2026-10-06T23:53:00Z",
        "missing_local_daily_kline_files": missing,
        "missing_days": [Path(path).parent.name for path in missing],
        "download_attempted": False,
        "reason": (
            "The local taker-flow feature table ends in development at 2026-06-30T23:59:00Z. "
            "Although base OHLCV rows exist through the holdout, the local BTCUSDT OHLCV CSV does not contain quote_volume or taker_buy_quote_volume."
        ),
    }
    return coverage


def _write_report(manifest: dict) -> None:
    specs = manifest["frozen_comparison"]["specification"]["models"]
    training = manifest.get("stages", {}).get("training", {})
    models = training.get("models", {})
    coverage = manifest.get("stages", {}).get("holdout_flow_preflight", {}).get("summary", {})
    model_rows = []
    for name in (BASE64_NAME, BASE70_NAME):
        record = models.get(name, {})
        model_rows.append(
            f"| {name} | {len(specs[name]['feature_order'])} | {specs[name]['iterations']} | "
            f"{record.get('elapsed_seconds', '-')} | {record.get('model', {}).get('sha256', 'fit not completed')} |"
        )
    missing_rows = "\n".join(f"- `{path}`" for path in coverage.get("missing_local_daily_kline_files", [])) or "- no missing daily files detected"
    report = f"""# BASE70 vs BASE64 RAW - preflight and final-fit checkpoint

## Status

The two frozen development models have been fitted and saved. Holdout feature construction and scoring are **blocked** because the local Binance UM BTCUSDT 1-minute data lacks `quote_volume` and `taker_buy_quote_volume` for October 2-6, 2026. No holdout predictions were generated or read, and no holdout quality or economic scores were calculated.

This is a historical assessment on a previously exposed holdout. The interval remains July 1 through October 6, 2026 inclusive, with the exclusive data cutoff at October 7, 2026 00:00 UTC. No period or development boundary was changed.

## Frozen models and saved hashes

| Model | Features | Iterations | Fit seconds | Model SHA-256 |
|---|---:|---:|---:|---|
{chr(10).join(model_rows)}

Both fits use the exact shared development mask: 3,174,805 rows, 634,957 decision rows, identical targets and weights, and the original `0.59` / `0.1025` decision/auxiliary weights. The latest training label is available at `2026-06-30T23:59:00Z`, the first holdout decision boundary. BASE64 uses the Z/V comparison configuration at 25 iterations; BASE70 uses original raw SINGLE at 31 iterations. Probabilities are to remain raw, without Platt calibration.

The saved full-development model was not reusable: its training population is not the exact shared Z/V complete-feature mask, and the Z/V report explicitly records the earlier BASE64 artifact as incompatible with that mask. Generators, transformations, and feature selection were confirmed as development-only in the full-development pipeline and Z/V manifests.

## Existing development reference

These are previously measured development results from the frozen Z/V comparison, not new holdout results.

| Period / target | Model | Rows | Log loss | Brier | AUC | Fills | Fees | Net PnL | PnL / turnover |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development proxy OOF | BASE64 | 318,560 | 0.692183 | 0.249519 | 0.5244 | - | - | - | - |
| Development proxy OOF | BASE70 | 318,560 | 0.692134 | 0.249494 | 0.5251 | - | - | - | - |
| Development official raw | BASE64 | 21,951 | 0.692129 | 0.249491 | 0.5268 | - | - | - | - |
| Development official raw | BASE70 | 21,951 | 0.692020 | 0.249437 | 0.5278 | - | - | - | - |
| Development economics | BASE64 | - | - | - | - | 1,445 | $258.64 | $194.27 | 2.69% |
| Development economics | BASE70 | - | - | - | - | 1,452 | $259.71 | $260.05 | 3.58% |

The full development comparison is in [the Z/V ablation report](btc_preopen_zv_ablation_20261018.md). Development and holdout results are not pooled.

## Missing holdout inputs

The saved `flow_features.parquet` ends at `{coverage.get('cached_flow_features_last_opened_utc', 'unknown')}`. The local monthly UM BTCUSDT klines cover July-September and there is a daily file for October 1, but these daily inputs are absent:

{missing_rows}

The preflight manifest records size and SHA-256 for locally available June-September monthly UM klines and the October 1 daily kline, along with missing status for October 2-6.

The separate local `BTCUSDT1m.csv` contains only `Opened, Open, High, Low, Close, Volume`. OHLCV cannot reconstruct the original `taker_V` definition, which requires quote volume and taker-buy quote volume. Substituting a proxy would change the frozen feature definition, so the holdout was not scored.

## Holdout questions

- Whether BASE70 beats BASE64 on holdout quality: **not evaluated**.
- Whether both models retain information beyond a constant 0.5 forecast: **not evaluated**.
- Whether positive development economics transfer: **not evaluated**.
- Whether quality improvement agrees with after-cost performance: **not evaluated**.

The frozen protocol uses 2,000 paired circular 3-day UTC block replicates with seed `20261018`, independent $5 orders, a 1% net-EV hurdle, T-60 decisions, T-59 execution, and the existing snapshots and ledger engine. No $100 portfolio or book archive replay is included.

## Artifacts and resources

- Config: `{CONFIG_PATH.resolve().relative_to(ROOT).as_posix()}`.
- Frozen run manifest and model checkpoints: `data/analysis/polymarket/BTC/btc_base70_raw_holdout_20261018/`.
- Versioned manifest: `reports/btc_preopen/btc_base70_raw_holdout_20261018_manifest.json`.
- New holdout predictions, ledgers, and score tables: not produced.
- Resource snapshot before fitting: `{json.dumps(manifest.get('resource_snapshot_before_fit', {}), ensure_ascii=False)}`.
- Both fit timings and sampled peak process working set are recorded in each model completion checkpoint. No final holdout-run runtime or peak memory exists because scoring did not start.

"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report.rstrip() + "\n", encoding="utf-8")


def run() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base64, base70, common, full = _source_summaries()
    specs = _model_specs(base64, base70)
    frame, x, labels, weights = _load_training_data(specs)

    if MANIFEST_PATH.is_file():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected_identity = manifest.get("frozen_comparison", {}).get("identity_sha256")
        if manifest.get("experiment_id") != CONFIG["experiment_id"] or not expected_identity:
            raise RuntimeError("Existing run manifest has a different experiment identity")
        if manifest.get("input_identity", {}).get("config_sha256") != _sha256(CONFIG_PATH):
            raise RuntimeError("Frozen config changed after model fitting began")
        if manifest.get("frozen_comparison", {}).get("specification", {}).get("models") != specs:
            raise RuntimeError("Source model specifications changed after the protocol was frozen")
    else:
        manifest = _freeze_manifest(specs, common, full)
        manifest["training_data_verification"] = {
            "rows": int(len(frame)),
            "decision_rows": int(frame["is_polymarket_preopen_decision"].sum()),
            "latest_label_available_at_utc": pd.to_datetime(frame["target_label_available_at"], utc=True).max().isoformat(),
            "feature_order_sha256": {name: _canonical_sha256(spec["feature_order"]) for name, spec in specs.items()},
            "training_weight_sha256": taker._array_sha256(weights, dtype=np.float32),
            "source_row_index_sha256": hashlib.sha256(frame["_source_row_index"].to_numpy(dtype=np.int32).tobytes()).hexdigest(),
            "target_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "holdout_rows_used": 0,
        }
        _write_manifests(manifest)

    if manifest.get("training_data_verification", {}).get("training_weight_sha256") != taker._array_sha256(weights, dtype=np.float32):
        raise RuntimeError("Loaded training data no longer matches the frozen weight identity")
    if manifest.get("frozen_comparison", {}).get("identity_sha256") != _canonical_sha256(
        manifest["frozen_comparison"]["specification"]
    ):
        raise RuntimeError("Frozen protocol identity hash is invalid")
    manifest.setdefault("input_identity", {})["preflight_runner_sha256"] = _sha256(Path(__file__))

    _verify_frozen_files(manifest)
    new_fits = _fit_models(manifest, frame, x, labels, weights)
    coverage = _check_holdout_flow_coverage(manifest)
    if not coverage["missing_local_daily_kline_files"]:
        raise RuntimeError(
            "All required daily klines are present, but this runner only records the blocked preflight; "
            "it has no holdout scoring stage and must not report the experiment as ready."
        )
    manifest.setdefault("stages", {})["holdout_flow_preflight"] = {
        "status": "blocked" if coverage["missing_local_daily_kline_files"] else "ready",
        "summary": coverage,
        "finished_utc": _now_utc(),
    }
    manifest["holdout_predictions_read"] = False
    manifest["holdout_scoring_started"] = False
    manifest["status"] = "blocked_missing_local_holdout_taker_flow_data" if coverage["missing_local_daily_kline_files"] else "frozen_models_ready"
    manifest["next_step"] = "Provide the missing local UM BTCUSDT minute klines with quote_volume and taker_buy_quote_volume for 2026-10-02 through 2026-10-06 UTC, then resume this frozen assessment without refitting."
    manifest["resource_snapshot_after_fit"] = taker._resource_snapshot()
    manifest["stages"]["training"]["summary"]["new_fits_this_invocation"] = new_fits
    manifest["elapsed_training_seconds"] = sum(
        float(item.get("elapsed_seconds", 0.0))
        for item in manifest.get("stages", {}).get("training", {}).get("models", {}).values()
    )
    _write_report(manifest)
    manifest["stages"]["preflight_report"] = {
        "status": "completed",
        "path": REPORT_PATH.resolve().relative_to(ROOT).as_posix(),
        "sha256": _sha256(REPORT_PATH),
        "size_bytes": REPORT_PATH.stat().st_size,
    }
    manifest["outputs"] = [
        record
        for name in (BASE64_NAME, BASE70_NAME)
        for record in (
            manifest["stages"]["training"]["models"][name]["model"],
            manifest["stages"]["training"]["models"][name]["completion"],
        )
    ] + [manifest["stages"]["preflight_report"]]
    _write_manifests(manifest)
    return manifest


if __name__ == "__main__":
    run()
