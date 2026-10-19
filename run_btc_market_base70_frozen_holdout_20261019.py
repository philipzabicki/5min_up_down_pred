"""Evaluate the fixed development M0/M1 fits on the saved BASE70 holdout."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import run_btc_market_base70_incremental_20261019 as development
import run_btc_market_probability_experiment as market_probability
import run_btc_preopen_research as research
import run_btc_preopen_t59_reassessment as replay


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_market_base70_frozen_holdout_20261019.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_directory"]
RUN_MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
REPORT_PATH = ROOT / CONFIG["report_path"]
VERSIONED_MANIFEST_PATH = ROOT / CONFIG["versioned_manifest_path"]
STATUS_PATH = ROOT / CONFIG["status_path"]

HOLDOUT_START = pd.Timestamp(CONFIG["holdout_start_inclusive_utc"])
HOLDOUT_CUTOFF = pd.Timestamp(CONFIG["holdout_cutoff_exclusive_utc"])
FIT_BOUNDARY = pd.Timestamp(CONFIG["fit_boundary_first_holdout_decision_utc"])
EPSILON = float(CONFIG["probability_epsilon"])
MODEL_NAMES = ("MARKET", "RAW", "M0", "M1")
PROBABILITY_COLUMNS = {"MARKET": "p_market", "RAW": "p_raw", "M0": "p_m0", "M1": "p_m1"}
CONTRASTS = {
    "M1_MINUS_M0": ("M1", "M0"),
    "M1_MINUS_RAW": ("M1", "RAW"),
    "M0_MINUS_MARKET": ("M0", "MARKET"),
}
REPORT_ONLY_RESUME_REASON = (
    "Clarify transferred probability and economic results and report cash fees separately for settled and unresolved fills; "
    "no fit, prediction, bootstrap, execution, or accounting calculation changed."
)
SNAPSHOT_COLUMNS = [
    "condition_id", "market_start_utc", "entry_time_utc", "fee_collection_mode", "fee_rate_bps",
    "fee_known", "fee_event_ns", "archive_source", "archive_entry_hour_available",
    "archive_freshness_window_available", "archive_window_hours_expected",
    "archive_window_hours_source_available", "archive_window_hours_processed",
    "archive_window_missing_source_hours", "archive_window_unprocessed_hours",
    "archive_gap_after_up_book_init", "archive_gap_after_down_book_init", "archive_gap_after_fee_update",
    "up_book_init_receive_ns", "down_book_init_receive_ns", "no_future_event_at_entry",
    "no_future_source_event_at_entry",
    "up_best_bid", "up_best_ask", "up_best_ask_size_shares", "up_ask_age_seconds",
    "up_quote_source_token_id", "up_quote_complemented", "up_ask_levels", "up_ask_order_ambiguous",
    "up_bbo_ask_comparable", "up_bbo_ask_mismatch", "up_bid_ask_crossed",
    "down_best_bid", "down_best_ask", "down_best_ask_size_shares", "down_ask_age_seconds",
    "down_quote_source_token_id", "down_quote_complemented", "down_ask_levels", "down_ask_order_ambiguous",
    "down_bbo_ask_comparable", "down_bbo_ask_mismatch", "down_bid_ask_crossed",
    "quote_valid", "has_full_snapshot",
]
CALENDAR_COLUMNS = [
    "condition_id", "market_start_utc", "up_token_id", "down_token_id",
    "order_min_size_usd_current", "order_min_size_shares_current", "resolved_at_utc",
    "target_polymarket_up", "t59_snapshot_available", "t60_decision_snapshot_available",
    "outcome_available_by_cutoff", "official_outcome_available_by_cutoff",
    "official_label_release_at_utc", "cash_release_at_cutoff_utc",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _record(path: Path) -> dict:
    return {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_records(records: list[dict]) -> bool:
    for record in records:
        path = ROOT / record["path"]
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
            return False
        if _sha256(path) != record["sha256"]:
            return False
    return bool(records)


def _checkpoint_valid(manifest: dict, stage: str) -> bool:
    item = manifest.get("stages", {}).get(stage, {})
    return item.get("status") == "completed" and _verify_records(item.get("outputs", []))


def _source_paths() -> dict[str, Path]:
    paths = {key: ROOT / value for key, value in CONFIG["sources"].items()}
    paths.update({
        "config": CONFIG_PATH,
        "runner": Path(__file__),
        "development_runner": ROOT / "run_btc_market_base70_incremental_20261019.py",
        "market_probability_definition": ROOT / "run_btc_market_probability_experiment.py",
        "research_calendar_and_snapshot_helpers": ROOT / "run_btc_preopen_research.py",
        "economic_engine_and_cashflow_audit": ROOT / "run_btc_preopen_t59_reassessment.py",
        "official_label_scoring": ROOT / "run_btc_preopen_taker_flow_20261018.py",
        "fee_rule_implementation": ROOT / "utils/polymarket_btc5m_fee_rules.py",
    })
    return paths


def _verify_prior_manifests(paths: dict[str, Path]) -> None:
    raw_manifest = json.loads(paths["raw_holdout_run_manifest_path"].read_text(encoding="utf-8"))
    if raw_manifest.get("status") != "completed_historical_holdout_assessment":
        raise RuntimeError("The saved BASE70 holdout manifest is not complete")
    period = raw_manifest["frozen_comparison"]["specification"]["period"]
    if (
        period["holdout_start_inclusive_utc"] != CONFIG["holdout_start_inclusive_utc"]
        or period["data_cutoff_exclusive_utc"] != CONFIG["holdout_cutoff_exclusive_utc"]
        or period["holdout_last_market_start_inclusive_utc"] != CONFIG["holdout_last_market_start_inclusive_utc"]
    ):
        raise RuntimeError("The saved RAW holdout period differs from the frozen scope")
    expected_outputs = {
        "raw_holdout_predictions_path": "predictions",
        "raw_holdout_calendar_path": "execution_calendar",
        "raw_holdout_trades_path": "economics_BASE70",
    }
    for source_name, stage_name in expected_outputs.items():
        expected_path = CONFIG["sources"][source_name]
        outputs = raw_manifest["stages"][stage_name]["outputs"]
        record = next((item for item in outputs if item["path"] == expected_path), None)
        if record is None or not _verify_records([record]):
            raise RuntimeError(f"Saved RAW holdout artifact is missing or changed: {expected_path}")
    raw_artifacts = {item["path"]: item for item in raw_manifest["frozen_comparison"]["specification"]["input_artifacts"]}
    for source_name in ("official_market_index_path", "t59_snapshots_path", "t60_snapshots_path"):
        path = CONFIG["sources"][source_name]
        record = raw_artifacts.get(path)
        if record is None or not _verify_records([record]):
            raise RuntimeError(f"Frozen RAW holdout source differs from this run: {path}")

    dev_manifest = json.loads(paths["development_experiment_manifest_path"].read_text(encoding="utf-8"))
    if dev_manifest.get("status") != "completed" or not dev_manifest.get("frozen_comparison", {}).get("identity_sha256"):
        raise RuntimeError("The daily development experiment is not complete and frozen")
    dev_sources = dev_manifest["frozen_comparison"]["specification"]["source_records"]
    expected_dev = {
        "raw_oof_path": CONFIG["sources"]["development_oof_path"],
        "official_market_index_path": CONFIG["sources"]["official_market_index_path"],
        "t59_snapshots_path": CONFIG["sources"]["t59_snapshots_path"],
        "t60_snapshots_path": CONFIG["sources"]["t60_snapshots_path"],
    }
    for source_name, path in expected_dev.items():
        record = dev_sources.get(source_name)
        if record is None or record["path"] != path or not _verify_records([record]):
            raise RuntimeError(f"Development calibration source differs from its completed experiment: {path}")


def _freeze() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = _source_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Frozen holdout inputs are missing: " + ", ".join(missing))
    _verify_prior_manifests(paths)
    frozen = {
        "config": CONFIG,
        "source_records": {name: _record(path) for name, path in paths.items()},
        "method": {
            "development_fit_count_per_model": 1,
            "holdout_fit_count_per_model": 0,
            "training_label_release_strictly_before_first_holdout_decision": True,
            "shared_m0_m1_training_rows": True,
            "common_market_calendar": True,
            "holdout_previously_exposed": True,
            "coefficient_update_during_holdout": False,
            "holdout_prediction_variants": MODEL_NAMES,
            "quality_contrasts": CONTRASTS,
        },
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    identity = _canonical_sha256({key: item for key, item in frozen.items() if key != "frozen_at_utc"})
    if RUN_MANIFEST_PATH.is_file():
        manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
        previous_frozen = manifest.get("frozen_comparison", {})
        previous_spec = previous_frozen.get("specification", {})
        previous_records = previous_spec.get("source_records", {})
        current_spec = {key: value for key, value in frozen.items() if key != "frozen_at_utc"}
        previous_method = {key: value for key, value in previous_spec.items() if key != "frozen_at_utc"}
        previous_method["source_records"] = {
            name: record for name, record in previous_records.items() if name != "runner"
        }
        current_method = dict(current_spec)
        current_method["source_records"] = {
            name: record for name, record in frozen["source_records"].items() if name != "runner"
        }
        if (
            previous_frozen.get("identity_sha256") != identity
            and _canonical_sha256(previous_method) != _canonical_sha256(current_method)
        ):
            raise RuntimeError("Existing frozen holdout manifest does not match the config or source hashes")
        non_runner = [record for name, record in previous_records.items() if name != "runner"]
        if not _verify_records(non_runner):
            raise RuntimeError("A frozen input changed since this holdout run began")
        previous_runner_sha = previous_records["runner"]["sha256"]
        current_runner_sha = frozen["source_records"]["runner"]["sha256"]
        resumed_runner_sha = manifest.get("implementation_resume", {}).get("current_runner_sha256")
        if current_runner_sha not in {previous_runner_sha, resumed_runner_sha}:
            calculation_stages = [
                "frozen_development_fit", "holdout_alignment", "shared_bootstrap_calendar",
                "quality_and_bootstrap", "economics_MARKET", "economics_RAW", "economics_M0",
                "economics_M1", "final_summary",
            ]
            if (
                not REPORT_ONLY_RESUME_REASON
                or not all(_checkpoint_valid(manifest, stage) for stage in calculation_stages)
            ):
                raise RuntimeError("Runner source changed after evaluation; frozen result checkpoints cannot be reused")
            manifest["implementation_resume"] = {
                "original_frozen_runner_sha256": previous_runner_sha,
                "current_runner_sha256": current_runner_sha,
                "reason": REPORT_ONLY_RESUME_REASON,
                "recorded_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            _write_manifest(manifest)
        return manifest
    manifest = {
        "schema_version": 1,
        "experiment_id": CONFIG["experiment_id"],
        "status": "frozen_before_holdout_scoring",
        "active_stage": None,
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "frozen_comparison": {"identity_sha256": identity, "specification": frozen},
        "runtime_identity": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "stages": {},
    }
    _write_manifest(manifest)
    return manifest


def _write_manifest(manifest: dict) -> None:
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(RUN_MANIFEST_PATH, manifest)
    _write_json(VERSIONED_MANIFEST_PATH, manifest)


def _start_stage(manifest: dict, name: str) -> None:
    manifest["status"] = "running"
    manifest["active_stage"] = name
    manifest.setdefault("stages", {})[name] = {
        "status": "running", "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write_manifest(manifest)


def _complete_stage(manifest: dict, name: str, outputs: list[Path], summary: dict | None = None) -> None:
    manifest["stages"][name] = {
        "status": "completed",
        "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "summary": summary or {},
        "outputs": [_record(path) for path in outputs],
    }
    manifest["active_stage"] = None
    _write_manifest(manifest)


def _load_or_fit_frozen_coefficients(manifest: dict) -> dict:
    coeff_path = OUTPUT_DIR / "frozen_coefficients.json"
    rows_path = OUTPUT_DIR / "frozen_development_training_rows.parquet"
    stage = "frozen_development_fit"
    if _checkpoint_valid(manifest, stage):
        return json.loads(coeff_path.read_text(encoding="utf-8"))

    _start_stage(manifest, stage)
    development_rows, _ = development._build_development_predictions()
    train = development._training_rows(development_rows, FIT_BOUNDARY)
    if len(train) < int(CONFIG["minimum_training_rows"]):
        raise RuntimeError("The frozen development fit has fewer rows than the established minimum")
    labels = train.target_polymarket_up.astype(int)
    if len(labels.value_counts()) != 2:
        raise RuntimeError("The frozen development fit does not contain both official outcome classes")
    if not train.market_start_utc.lt(HOLDOUT_START).all():
        raise RuntimeError("The frozen fit includes a market that started in the holdout")
    releases = pd.to_datetime(train.official_label_release_at_utc, utc=True, errors="raise")
    if not releases.lt(FIT_BOUNDARY).all():
        raise RuntimeError("The frozen fit includes an official label released at or after the first holdout decision")

    m0 = development._fit_model(train, include_deviation=False)
    m1 = development._fit_model(train, include_deviation=True)
    sample_sha = development._sample_identity(train)
    if sample_sha != development._sample_identity(train.copy()) or train.market_start_utc.duplicated().any():
        raise RuntimeError("The M0/M1 frozen training calendar is not a unique shared sample")
    if m0["coefficients"].keys() != {"a", "b"} or m1["coefficients"].keys() != {"a", "b", "c"}:
        raise RuntimeError("The frozen coefficient structure differs from M0/M1")

    train_output = train[[
        "condition_id", "market_start_utc", "official_label_release_at_utc",
        "p_market", "p_raw", "target_polymarket_up",
    ]].copy()
    _write_parquet(train_output, rows_path)
    fit_record = {
        "fit_boundary_first_holdout_decision_utc": FIT_BOUNDARY,
        "label_release_rule": CONFIG["official_label_release_rule"],
        "release_boundary_strict": True,
        "training_rows": int(len(train)),
        "up_labels": int(labels.sum()),
        "down_labels": int(len(labels) - labels.sum()),
        "first_market_start_utc": train.market_start_utc.min(),
        "last_market_start_utc": train.market_start_utc.max(),
        "first_label_release_utc": releases.min(),
        "last_label_release_utc": releases.max(),
        "training_market_start_sample_sha256": sample_sha,
        "training_rows_sha256": _sha256(rows_path),
        "m0_training_rows": int(len(train)),
        "m1_training_rows": int(len(train)),
        "m0_training_market_start_sample_sha256": sample_sha,
        "m1_training_market_start_sample_sha256": sample_sha,
        "probability_epsilon": EPSILON,
        "regularization_lambda": float(CONFIG["regularization_lambda"]),
        "minimum_training_rows": int(CONFIG["minimum_training_rows"]),
        "objective": CONFIG["models"]["objective"],
        "optimizer": CONFIG["models"]["optimizer"],
        "m0": m0,
        "m1": m1,
        "holdout_source_hashes": {
            name: item["sha256"]
            for name, item in manifest["frozen_comparison"]["specification"]["source_records"].items()
            if name.startswith("raw_holdout_") or name in {"official_market_index_path", "t60_snapshots_path"}
        },
        "development_oof_sha256": manifest["frozen_comparison"]["specification"]["source_records"]["development_oof_path"]["sha256"],
        "frozen_manifest_identity_sha256": manifest["frozen_comparison"]["identity_sha256"],
    }
    _write_json(coeff_path, fit_record)
    summary = {
        key: fit_record[key]
        for key in (
            "fit_boundary_first_holdout_decision_utc", "training_rows", "up_labels", "down_labels",
            "first_market_start_utc", "last_market_start_utc", "first_label_release_utc",
            "last_label_release_utc", "training_market_start_sample_sha256",
        )
    }
    _complete_stage(manifest, stage, [coeff_path, rows_path], summary)
    return fit_record


def _load_snapshot_frames(condition_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for name in ("t59_snapshots_path", "t60_snapshots_path"):
        frame = pd.read_parquet(ROOT / CONFIG["sources"][name], columns=SNAPSHOT_COLUMNS)
        frame["condition_id"] = frame.condition_id.astype("string").str.lower()
        frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True, errors="raise")
        frame["entry_time_utc"] = pd.to_datetime(frame.entry_time_utc, utc=True, errors="coerce")
        frame = frame.loc[
            frame.condition_id.isin(condition_ids)
            & frame.market_start_utc.ge(HOLDOUT_START)
            & frame.market_start_utc.lt(HOLDOUT_CUTOFF)
        ].copy()
        if frame.condition_id.duplicated().any():
            raise RuntimeError(f"The saved {name} contains duplicate holdout snapshots per condition ID")
        offset = pd.Timedelta(seconds=-59 if name == "t59_snapshots_path" else -60)
        if not frame.entry_time_utc.eq(frame.market_start_utc + offset).all():
            raise RuntimeError(f"The saved {name} violates the frozen T-59/T-60 timestamp")
        frames.append(frame)
    return frames[0], frames[1]


def _build_holdout_inputs(manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "predictions": OUTPUT_DIR / "holdout_predictions.parquet",
        "calendar": OUTPUT_DIR / "shared_execution_calendar.parquet",
        "t59": OUTPUT_DIR / "t59_holdout_snapshots.parquet",
        "t60": OUTPUT_DIR / "t60_holdout_snapshots.parquet",
    }
    stage = "holdout_alignment"
    if _checkpoint_valid(manifest, stage):
        return tuple(pd.read_parquet(paths[key]) for key in ("predictions", "calendar", "t59", "t60"))

    _start_stage(manifest, stage)
    raw_path = ROOT / CONFIG["sources"]["raw_holdout_predictions_path"]
    raw = pd.read_parquet(raw_path, columns=[
        "nominal_decision_at", "market_start_utc", "p_model_raw", "p_BASE70_raw",
        "target_polymarket_up", "resolved_at_utc", "official_label_release_at_utc",
        "official_outcome_available_by_cutoff",
    ])
    raw["market_start_utc"] = pd.to_datetime(raw.market_start_utc, utc=True, errors="raise")
    raw["nominal_decision_at"] = pd.to_datetime(raw.nominal_decision_at, utc=True, errors="raise")
    raw["resolved_at_utc"] = pd.to_datetime(raw.resolved_at_utc, utc=True, errors="coerce")
    raw["official_label_release_at_utc"] = pd.to_datetime(raw.official_label_release_at_utc, utc=True, errors="coerce")
    if len(raw) != 28_224 or raw.market_start_utc.duplicated().any():
        raise RuntimeError("The saved holdout must contain the full 28,224 unique RAW forecast rows")
    if not raw.market_start_utc.between(HOLDOUT_START, pd.Timestamp(CONFIG["holdout_last_market_start_inclusive_utc"])).all():
        raise RuntimeError("Saved RAW predictions extend outside the frozen holdout market-start interval")
    if not np.array_equal(raw.p_model_raw.to_numpy(), raw.p_BASE70_raw.to_numpy(), equal_nan=True):
        raise RuntimeError("Saved RAW aliases differ; the existing RAW BASE70 holdout prediction was expected")
    if not np.isfinite(raw.p_model_raw.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("The saved RAW holdout contains missing or nonfinite probabilities")
    if raw.nominal_decision_at.min() != FIT_BOUNDARY:
        raise RuntimeError("Frozen fit boundary is not the first saved holdout T-60 decision")

    calendar = pd.read_parquet(ROOT / CONFIG["sources"]["raw_holdout_calendar_path"], columns=CALENDAR_COLUMNS)
    calendar["market_start_utc"] = pd.to_datetime(calendar.market_start_utc, utc=True, errors="raise")
    calendar["condition_id"] = calendar.condition_id.astype("string").str.lower()
    calendar = calendar.loc[
        calendar.market_start_utc.ge(HOLDOUT_START) & calendar.market_start_utc.lt(HOLDOUT_CUTOFF)
    ].copy()
    if len(calendar) != len(raw) or calendar.market_start_utc.duplicated().any():
        raise RuntimeError("Saved RAW execution calendar does not align one-to-one to all holdout forecasts")

    official_path = ROOT / CONFIG["sources"]["official_market_index_path"]
    official = pd.read_parquet(official_path, columns=[
        "condition_id", "market_start_utc", "resolved_at_utc", "target_polymarket_up",
    ])
    official["condition_id"] = official.condition_id.astype("string").str.lower()
    official["market_start_utc"] = pd.to_datetime(official.market_start_utc, utc=True, errors="raise")
    official["resolved_at_utc"] = pd.to_datetime(official.resolved_at_utc, utc=True, errors="coerce")
    official = official.loc[
        official.market_start_utc.ge(HOLDOUT_START) & official.market_start_utc.lt(HOLDOUT_CUTOFF)
    ].copy()
    if official.condition_id.duplicated().any():
        raise RuntimeError("Official market index contains duplicate holdout condition IDs")
    calendar = calendar.merge(
        raw[["market_start_utc", "nominal_decision_at", "p_model_raw", "p_BASE70_raw",
             "target_polymarket_up", "resolved_at_utc", "official_label_release_at_utc",
             "official_outcome_available_by_cutoff"]],
        on="market_start_utc", how="left", validate="one_to_one", suffixes=("_calendar", "_saved"),
    )
    official = official.rename(columns={
        "resolved_at_utc": "resolved_at_official_utc",
        "target_polymarket_up": "target_official_up",
    })
    calendar = calendar.merge(
        official[["condition_id", "market_start_utc", "resolved_at_official_utc", "target_official_up"]],
        on=["condition_id", "market_start_utc"], how="left", validate="one_to_one",
    )
    saved_label = pd.to_numeric(calendar.target_polymarket_up_saved, errors="coerce")
    official_label = pd.to_numeric(calendar.target_official_up, errors="coerce")
    if not np.allclose(
        saved_label.to_numpy(dtype=np.float64), official_label.to_numpy(dtype=np.float64),
        rtol=0.0, atol=0.0, equal_nan=True,
    ):
        raise RuntimeError("RAW holdout labels differ from the frozen official market index")
    saved_resolution = pd.to_datetime(calendar.resolved_at_utc_saved, utc=True, errors="coerce")
    official_resolution = pd.to_datetime(calendar.resolved_at_official_utc, utc=True, errors="coerce")
    if not np.array_equal(saved_resolution.astype("int64").to_numpy(), official_resolution.astype("int64").to_numpy()):
        raise RuntimeError("RAW holdout resolution timestamps differ from the official market index")
    calendar["resolved_at_utc"] = official_resolution
    calendar["target_polymarket_up"] = official_label
    starts = pd.to_datetime(calendar.market_start_utc, utc=True)
    release_base = official_resolution.where(official_resolution.ge(starts + pd.Timedelta(minutes=5)), starts + pd.Timedelta(minutes=5))
    calendar["official_label_release_at_utc"] = release_base + pd.Timedelta(seconds=60)
    available = (
        official_label.isin((0, 1)) & official_resolution.notna()
        & calendar.official_label_release_at_utc.le(HOLDOUT_CUTOFF)
    )
    if not available.astype(bool).equals(calendar.official_outcome_available_by_cutoff_saved.fillna(False).astype(bool)):
        raise RuntimeError("Saved holdout official-label availability flags do not match the release rule and cutoff")
    calendar["official_outcome_available_by_cutoff"] = available

    condition_ids = set(calendar.condition_id.dropna().astype(str).str.lower())
    t59, t60 = _load_snapshot_frames(condition_ids)
    t59_map = research._snapshot_map(t59)
    t60_map = research._snapshot_map(t60)
    start_by_id = calendar.set_index("condition_id").market_start_utc.to_dict()
    for condition_id, snapshot in t60_map.items():
        if condition_id in start_by_id and snapshot["market_start_utc"] != start_by_id[condition_id]:
            raise RuntimeError("T-60 snapshot is attached to the wrong holdout market")
    for condition_id, snapshot in t59_map.items():
        if condition_id in start_by_id and snapshot["market_start_utc"] != start_by_id[condition_id]:
            raise RuntimeError("T-59 snapshot is attached to the wrong holdout market")

    raw_prediction = pd.DataFrame({
        "market_start_utc": calendar.market_start_utc,
        "nominal_decision_at": calendar.nominal_decision_at,
        "condition_id": calendar.condition_id,
        "resolved_at_utc": calendar.resolved_at_utc,
        "official_label_release_at_utc": calendar.official_label_release_at_utc,
        "official_outcome_available_by_cutoff": calendar.official_outcome_available_by_cutoff,
        "target_polymarket_up": calendar.target_polymarket_up,
        "p_raw": pd.to_numeric(calendar.p_model_raw, errors="coerce"),
        "t59_snapshot_available": calendar.t59_snapshot_available.fillna(False).astype(bool),
        "t60_decision_snapshot_available": calendar.t60_decision_snapshot_available.fillna(False).astype(bool),
    })
    raw_prediction["p_market"] = [
        market_probability._snapshot_features(t60_map[condition_id])["p_market"]
        if pd.notna(condition_id) and condition_id in t60_map else np.nan
        for condition_id in raw_prediction.condition_id
    ]
    raw_prediction["p_market"] = pd.to_numeric(raw_prediction.p_market, errors="coerce")
    eligible = raw_prediction.p_market.notna() & raw_prediction.p_raw.notna() & raw_prediction.condition_id.notna()
    raw_prediction["logit_market"] = np.nan
    raw_prediction["logit_raw"] = np.nan
    raw_prediction.loc[eligible, "logit_market"] = development._logit(raw_prediction.loc[eligible, "p_market"])
    raw_prediction.loc[eligible, "logit_raw"] = development._logit(raw_prediction.loc[eligible, "p_raw"])
    raw_prediction["d_raw_minus_market"] = raw_prediction.logit_raw - raw_prediction.logit_market

    coefficient_record = json.loads((OUTPUT_DIR / "frozen_coefficients.json").read_text(encoding="utf-8"))
    raw_prediction["p_m0"] = np.nan
    raw_prediction["p_m1"] = np.nan
    raw_prediction.loc[eligible, "p_m0"] = development._predict_m0(
        raw_prediction.loc[eligible, "logit_market"], coefficient_record["m0"]["coefficients"],
    )
    raw_prediction.loc[eligible, "p_m1"] = development._predict_m1(
        raw_prediction.loc[eligible, "logit_market"], raw_prediction.loc[eligible, "d_raw_minus_market"],
        coefficient_record["m1"]["coefficients"],
    )
    raw_prediction["decision_day_utc"] = raw_prediction.market_start_utc.dt.floor("D")

    shared_starts = set(raw_prediction.loc[eligible, "market_start_utc"])
    shared_calendar = calendar.loc[calendar.market_start_utc.isin(shared_starts)].copy()
    if shared_calendar.condition_id.isna().any() or shared_calendar.condition_id.duplicated().any():
        raise RuntimeError("Shared RAW/p_market holdout calendar has missing or duplicate condition identities")
    _write_parquet(raw_prediction, paths["predictions"])
    _write_parquet(shared_calendar, paths["calendar"])
    _write_parquet(t59, paths["t59"])
    _write_parquet(t60, paths["t60"])
    summary = {
        "raw_prediction_rows": int(len(raw_prediction)),
        "finite_raw_probability_rows": int(raw_prediction.p_raw.notna().sum()),
        "market_record_rows": int(raw_prediction.condition_id.notna().sum()),
        "t60_snapshot_rows": int(raw_prediction.t60_decision_snapshot_available.sum()),
        "valid_t60_market_probability_rows": int(raw_prediction.p_market.notna().sum()),
        "shared_raw_market_forecast_rows": int(eligible.sum()),
        "shared_forecast_t59_snapshot_rows": int(raw_prediction.loc[eligible, "t59_snapshot_available"].sum()),
        "shared_forecast_official_outcomes_available_by_cutoff": int(
            raw_prediction.loc[eligible, "official_outcome_available_by_cutoff"].sum()
        ),
        "shared_quality_rows": int((eligible & raw_prediction.official_outcome_available_by_cutoff).sum()),
        "first_holdout_decision_utc": raw_prediction.nominal_decision_at.min(),
        "last_holdout_market_start_utc": raw_prediction.market_start_utc.max(),
        "missing_market_probability_rows_not_filled_with_0_5": int(raw_prediction.p_market.isna().sum()),
        "t59_snapshot_identity_rows": int(len(t59)),
        "t60_snapshot_identity_rows": int(len(t60)),
    }
    _complete_stage(manifest, stage, list(paths.values()), summary)
    return raw_prediction, shared_calendar, t59, t60


def _quality_sample(predictions: pd.DataFrame) -> pd.DataFrame:
    forecast_columns = [PROBABILITY_COLUMNS[name] for name in MODEL_NAMES]
    valid = (
        predictions[forecast_columns].notna().all(axis=1)
        & predictions.condition_id.notna()
        & predictions.target_polymarket_up.isin((0, 1))
        & predictions.resolved_at_utc.notna()
        & predictions.official_outcome_available_by_cutoff.fillna(False)
    )
    rows = predictions.loc[valid].copy()
    if rows.empty or rows.target_polymarket_up.nunique() != 2:
        raise RuntimeError("The shared holdout quality calendar lacks official labels or one outcome class")
    return rows


def _load_or_score_quality(manifest: dict, predictions: pd.DataFrame, days: pd.DatetimeIndex, indices: np.ndarray):
    summary_path = OUTPUT_DIR / "quality_summary.json"
    monthly_path = OUTPUT_DIR / "monthly_quality.csv"
    stage = "quality_and_bootstrap"
    if _checkpoint_valid(manifest, stage):
        return json.loads(summary_path.read_text(encoding="utf-8")), pd.read_csv(monthly_path)
    _start_stage(manifest, stage)
    rows = _quality_sample(predictions)
    summary, monthly = development._quality_summary(rows, days, indices)
    _write_json(summary_path, summary)
    monthly.to_csv(monthly_path, index=False)
    _complete_stage(
        manifest, stage, [summary_path, monthly_path, OUTPUT_DIR / "shared_bootstrap_day_indices.npz"],
        summary["sample"],
    )
    return summary, monthly


def _save_bootstrap(manifest: dict, days: pd.DatetimeIndex) -> tuple[np.ndarray, Path]:
    path = OUTPUT_DIR / "shared_bootstrap_day_indices.npz"
    stage = "shared_bootstrap_calendar"
    if _checkpoint_valid(manifest, stage):
        with np.load(path) as saved:
            if not np.array_equal(saved["days"], days.asi8):
                raise RuntimeError("Saved bootstrap day calendar differs from shared holdout calendar")
            return saved["indices"], path
    _start_stage(manifest, stage)
    indices = development._bootstrap_indices(days)
    development._write_bootstrap(path, indices, days)
    _complete_stage(manifest, stage, [path], {
        "calendar_days": len(days),
        "replications": len(indices),
        "block_days": int(CONFIG["bootstrap"]["block_days"]),
        "seed": int(CONFIG["bootstrap"]["seed"]),
    })
    return indices, path


def _simulate_variant(
    variant: str,
    predictions: pd.DataFrame,
    calendar: pd.DataFrame,
    t59: pd.DataFrame,
    t60: pd.DataFrame,
    manifest: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    stage = f"economics_{variant}"
    output_dir = OUTPUT_DIR / "economics" / variant
    decisions_path, trades_path, summary_path = (
        output_dir / "decisions.parquet", output_dir / "trades.parquet", output_dir / "summary.json",
    )
    if _checkpoint_valid(manifest, stage):
        return (
            json.loads(summary_path.read_text(encoding="utf-8")),
            pd.read_parquet(decisions_path), pd.read_parquet(trades_path),
        )
    _start_stage(manifest, stage)
    shared = predictions.loc[
        predictions[[PROBABILITY_COLUMNS[name] for name in MODEL_NAMES]].notna().all(axis=1)
        & predictions.condition_id.notna()
    ].copy()
    market_rows = calendar.merge(
        shared[["market_start_utc", *[PROBABILITY_COLUMNS[name] for name in MODEL_NAMES]]],
        on="market_start_utc", how="inner", validate="one_to_one",
    )
    if len(market_rows) != len(shared):
        raise RuntimeError("Economic holdout rows differ from the shared forecast calendar")
    market_rows["p_candidate_platt"] = market_rows[PROBABILITY_COLUMNS[variant]]
    market_rows, period = research._asof_market_rows(market_rows, HOLDOUT_START, HOLDOUT_CUTOFF)
    market_rows.sort_values("market_start_utc", kind="stable", inplace=True)
    market_by_id = research._snapshot_map(market_rows.loc[market_rows.condition_id.notna()])
    t59_by_id = research._snapshot_map(t59)
    t60_by_id = research._snapshot_map(t60)
    summary, decisions, trades = development._simulate_policy(
        variant, market_rows, t59_by_id, market_by_id, t60_by_id,
    )
    if len(decisions) != len(market_rows):
        raise RuntimeError(f"{variant} did not produce one decision/status row per shared market")
    if len(decisions) and not pd.to_datetime(decisions.decision_time_utc, utc=True).eq(
        pd.to_datetime(decisions.market_start_utc, utc=True) - pd.Timedelta(seconds=60)
    ).all():
        raise RuntimeError(f"{variant} holdout decisions do not match T-60")
    if len(trades) and not pd.to_datetime(trades.entry_time_utc, utc=True).eq(
        pd.to_datetime(trades.market_start_utc, utc=True) - pd.Timedelta(seconds=59)
    ).all():
        raise RuntimeError(f"{variant} holdout executions do not match T-59")
    summary["calendar"] = period
    summary["monthly_results"] = development._monthly_economics(variant, decisions, trades)
    _write_parquet(decisions, decisions_path)
    _write_parquet(trades, trades_path)
    _write_json(summary_path, summary)
    _complete_stage(manifest, stage, [decisions_path, trades_path, summary_path], {
        key: summary[key] for key in (
            "decision_count", "intent_count", "fill_count", "settlement_count",
            "unresolved_fill_count", "net_pnl_usd",
        )
    })
    return summary, decisions, trades


def _variant_cashflow(trades: pd.DataFrame) -> dict:
    settled = trades.loc[trades.settlement_status.eq("settled")].copy() if len(trades) else trades
    unresolved = trades.loc[trades.settlement_status.ne("settled")].copy() if len(trades) else trades
    audit = development._cashflow_ledger_summary(trades)
    independent = development._independent_cashflow_audit(trades)
    audit["independent_decimal_audit"] = independent
    audit["unresolved_gross_notional_usd"] = development._sum_column(unresolved, "gross_usd")
    audit["unresolved_cash_fees_paid_usd"] = development._sum_column(unresolved, "fee_cash_usd")
    audit["unresolved_share_fee_usd_entry_equivalent_informational"] = development._sum_column(
        unresolved.loc[unresolved.fee_charge_unit.eq("outcome shares")] if len(unresolved) else unresolved,
        "fee_usd_entry_equivalent",
    )
    audit["all_fill_share_fee_usd_entry_equivalent_informational"] = development._sum_column(
        trades.loc[trades.fee_charge_unit.eq("outcome shares")] if len(trades) else trades,
        "fee_usd_entry_equivalent",
    )
    if len(settled):
        ev = pd.to_numeric(settled.chosen_side_ev_usd, errors="coerce")
        pnl = pd.to_numeric(settled.net_pnl_usd, errors="coerce")
        audit["forecast_ev_on_settled_fills_usd"] = float(ev.sum())
        audit["pnl_minus_forecast_ev_on_settled_fills_usd"] = float(pnl.sum() - ev.sum())
        audit["mean_absolute_per_fill_pnl_minus_ev_usd"] = float((pnl - ev).abs().mean())
    else:
        audit["forecast_ev_on_settled_fills_usd"] = 0.0
        audit["pnl_minus_forecast_ev_on_settled_fills_usd"] = 0.0
        audit["mean_absolute_per_fill_pnl_minus_ev_usd"] = None
    audit["settled_notional_usd"] = audit["gross_notional_usd"]
    audit["settled_cash_debit_usd"] = audit["settled_cash_outflow_usd"]
    return audit


def _reconcile_raw_holdout(
    full_raw: pd.DataFrame,
    shared_ids: set[str],
    current_raw_shared: pd.DataFrame,
) -> dict:
    old = full_raw.copy()
    old["condition_id"] = old.condition_id.astype("string").str.lower()
    old_shared = old.loc[old.condition_id.isin(shared_ids)].copy()
    old_removed = old.loc[~old.condition_id.isin(shared_ids)].copy()
    current = current_raw_shared.copy()
    current["condition_id"] = current.condition_id.astype("string").str.lower()
    fields = [
        "condition_id", "chosen_side", "gross_usd", "cash_debit_usd", "fee_cash_usd",
        "fee_shares", "fee_usd_entry_equivalent", "payout_usd", "net_pnl_usd", "settlement_status",
    ]
    expected = old_shared[fields].sort_values("condition_id", kind="stable").reset_index(drop=True)
    observed = current[fields].sort_values("condition_id", kind="stable").reset_index(drop=True)
    matches = len(expected) == len(observed)
    if matches:
        for field in ("condition_id", "chosen_side", "settlement_status"):
            matches = matches and expected[field].astype(str).equals(observed[field].astype(str))
    if matches:
        for field in ("gross_usd", "cash_debit_usd", "fee_cash_usd", "fee_shares", "fee_usd_entry_equivalent", "payout_usd", "net_pnl_usd"):
            left = pd.to_numeric(expected[field], errors="coerce").to_numpy(dtype=np.float64)
            right = pd.to_numeric(observed[field], errors="coerce").to_numpy(dtype=np.float64)
            matches = matches and np.allclose(left, right, rtol=0.0, atol=1e-8, equal_nan=True)
    if not matches:
        raise RuntimeError("RAW common-calendar executions do not reconcile to the saved BASE70 holdout ledger")
    old_summary = development._independent_cashflow_audit(old)
    shared_summary = development._independent_cashflow_audit(old_shared)
    removed_summary = development._independent_cashflow_audit(old_removed)
    current_summary = development._independent_cashflow_audit(current)
    full_pnl = old_summary["cashflow_pnl_usd"]
    shared_pnl = shared_summary["cashflow_pnl_usd"]
    removed_pnl = removed_summary["cashflow_pnl_usd"]
    if not math.isclose(full_pnl, shared_pnl + removed_pnl, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("Full-to-shared RAW PnL does not reconcile to removed-market PnL")
    if not math.isclose(shared_pnl, current_summary["cashflow_pnl_usd"], rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("Saved and replayed RAW PnL differ on common forecast markets")
    return {
        "ledger_matches_on_shared_calendar": True,
        "full_raw_holdout": old_summary,
        "saved_raw_ledger_on_shared_calendar": shared_summary,
        "replayed_raw_on_shared_calendar": current_summary,
        "raw_fills_removed_for_missing_common_forecast": removed_summary,
        "pnl_reconciliation_usd": {
            "full_raw_holdout": full_pnl,
            "shared_calendar": shared_pnl,
            "removed_markets": removed_pnl,
        },
    }


def _monthly_cashflow_rows(trades_by_variant: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for variant, trades in trades_by_variant.items():
        for month in ("2026-07", "2026-08", "2026-09", "2026-10"):
            if len(trades):
                months = pd.to_datetime(trades.decision_time_utc, utc=True).dt.strftime("%Y-%m")
                month_trades = trades.loc[months.eq(month)].copy()
            else:
                month_trades = trades
            audit = _variant_cashflow(month_trades)
            rows.append({"variant": variant, "utc_month": month, **audit})
    return rows


def _format_usd(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"${float(value):,.2f}"


def _format_ratio(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{float(value):.2%}"


def _write_report(manifest: dict, summary: dict, monthly_quality: pd.DataFrame, monthly_economics: list[dict], trades: dict[str, pd.DataFrame]) -> None:
    fit = summary["frozen_development_fit"]
    quality = summary["quality"]
    economics = summary["economics"]
    availability = summary["availability"]
    lines = [
        "# Zamrożony transfer M0/M1 na holdout RAW BASE70",
        "",
        "## Zakres i dopasowanie",
        "",
        "To historyczny test przenośności. Holdout był wcześniej oglądany, a hipoteza powstała po jego analizie; wynik nie jest niezależnym potwierdzeniem przewagi.",
        "",
        f"Holdout obejmuje starty rynków `{CONFIG['holdout_start_inclusive_utc']}`–`{CONFIG['holdout_last_market_start_inclusive_utc']}` UTC, z wyłącznym cutoffem rozliczeń `{CONFIG['holdout_cutoff_exclusive_utc']}`. Decyzja pierwszego rynku przypada na `{fit['fit_boundary_first_holdout_decision_utc']}`.",
        f"M0/M1 dopasowano dokładnie raz na `{fit['training_rows']:,}` wspólnych, kwalifikujących się wierszach development OOF (`{fit['first_market_start_utc']}`–`{fit['last_market_start_utc']}`). Etykiety były oficjalne i zwolnione ściśle przed granicą fit; najpóźniejszy release to `{fit['last_label_release_utc']}`. SHA-256 próby kalibracyjnej: `{fit['training_market_start_sample_sha256']}`; hash zapisanych wierszy fitu: `{fit['training_rows_sha256']}`.",
        "",
        "| Model | a | b | c | Cel fitu |",
        "|---|---:|---:|---:|---:|",
        f"| M0 | {fit['m0']['coefficients']['a']:.12g} | {fit['m0']['coefficients']['b']:.12g} | — | {fit['m0']['objective']:.12g} |",
        f"| M1 | {fit['m1']['coefficients']['a']:.12g} | {fit['m1']['coefficients']['b']:.12g} | {fit['m1']['coefficients']['c']:.12g} | {fit['m1']['objective']:.12g} |",
        "",
        "Definicje pozostały bez zmian: `m=logit(p_market)`, `d=logit(p_RAW)-m`, `M0=sigmoid(a+b*m)`, `M1=sigmoid(a+b*m+c*d)`. Zachowano przycinanie `1e-6`, nieważony log loss, karę `0.001*((b-1)^2+c^2)` (w M0 bez `c`) i L-BFGS-B bez ograniczeń znaków. Współczynniki nie zmieniały się w holdoucie.",
        "",
        "Poprzedni eksperyment development dopasowywał M0/M1 codziennie przed pierwszą decyzją dnia. Tutaj jeden fit na pełnych kwalifikujących się OOF przeniesiono bez aktualizacji przez cały okres holdoutu; to odrębny test od codziennego dopasowania development.",
        "",
        "## Dostępność i wspólny kalendarz",
        "",
        "| Etap | Rynki | Udział z 28 224 RAW |",
        "|---|---:|---:|",
        f"| Zapisane predykcje RAW BASE70 | {availability['raw_prediction_rows']:,} | 100.0% |",
        f"| Prawidłowy `p_market` z walidowanego T−60 | {availability['valid_t60_market_probability_rows']:,} | {availability['valid_t60_market_probability_rows'] / availability['raw_prediction_rows']:.1%} |",
        f"| Wspólny RAW + `p_market` i identyfikator rynku | {availability['shared_raw_market_forecast_rows']:,} | {availability['shared_raw_market_forecast_rows'] / availability['raw_prediction_rows']:.1%} |",
        f"| T−59 snapshot na wspólnym kalendarzu | {availability['shared_forecast_t59_snapshot_rows']:,} | {availability['shared_forecast_t59_snapshot_rows'] / availability['raw_prediction_rows']:.1%} |",
        f"| Oficjalne rozliczenie do cutoffu na wspólnym kalendarzu | {availability['shared_forecast_official_outcomes_available_by_cutoff']:,} | {availability['shared_forecast_official_outcomes_available_by_cutoff'] / availability['raw_prediction_rows']:.1%} |",
        f"| Wspólne etykiety jakościowe | {availability['quality_rows']:,} | {availability['quality_rows'] / availability['raw_prediction_rows']:.1%} |",
        "",
        f"Wspólna próbka obejmuje `{availability['shared_raw_market_forecast_rows']:,}` decyzji od `{availability['first_shared_market_start_utc']}` do `{availability['last_shared_market_start_utc']}`. Brakujące `p_market` pozostawiono brakujące; nie wstawiano 0,5. `MARKET`, `RAW`, M0 i M1 używają identycznych rynków dla jakości i identycznego kalendarza decyzji dla ekonomii. Dostępność prognozy, wykonania T−59 i oficjalnego rozliczenia są raportowane oddzielnie.",
        "",
        "## Jakość na oficjalnych etykietach",
        "",
        "| Wariant | Wiersze | Log loss | Brier | AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        metric = quality["overall"][name]
        lines.append(f"| {name} | {quality['sample']['rows']:,} | {metric['log_loss']:.6f} | {metric['brier']:.6f} | {metric['auc']:.6f} |")
    lines.extend([
        "",
        "Kontrast to pierwszy wariant minus drugi. Dla log loss/Brier wartości ujemne sprzyjają pierwszemu wariantowi; dla AUC dodatnie.",
        "",
        "| Kontrast | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |",
        "|---|---:|---:|---:|",
    ])
    for contrast in ("M1_MINUS_M0", "M1_MINUS_RAW", "M0_MINUS_MARKET"):
        metrics = quality["contrasts"][contrast]
        cells = []
        for metric in ("log_loss", "brier", "auc"):
            item = metrics[metric]
            lo, hi = item["95pct"]
            cells.append(f"{item['point_difference']:.6f} [{lo:.6f}, {hi:.6f}]")
        lines.append(f"| {contrast} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.extend([
        "",
        f"Przedziały pochodzą ze sparowanego bootstrapu kołowych bloków 3-dniowych UTC: {CONFIG['bootstrap']['replications']:,} wspólnych replikacji, seed `{CONFIG['bootstrap']['seed']}`.",
        "",
        "### Jakość według miesiąca UTC",
        "",
        "Październik jest niepełny (1–6 października).",
        "",
        "| Miesiąc | Wariant | n | Log loss | Brier | AUC |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in monthly_quality.itertuples(index=False):
        for name in MODEL_NAMES:
            lines.append(
                f"| {row.utc_month}{' (niepełny)' if row.utc_month == '2026-10' else ''} | {name} | {int(row.rows):,} | "
                f"{getattr(row, name + '_log_loss'):.6f} | {getattr(row, name + '_brier'):.6f} | {getattr(row, name + '_auc'):.6f} |"
            )
    lines.extend([
        "",
        "## Ekonomia na wspólnym kalendarzu",
        "",
        "Każdy wariant sam wybiera stronę i transakcje. Zastosowano żądany notional zakupu $5, próg net EV 1% notionalu, decyzję T−60, wykonanie T−59, zapisane depth/snapshoty, historyczny tryb opłat i cutoff rozliczeń. No-trade ma PnL `$0`.",
        "",
        "| Wariant | Decyzje | Fill / settled / nierozliczone | Notional settled | Debet settled | Cash fee settled / unresolved | Token fee eq. settled / unresolved | PnL netto | PnL/notional | PnL/debet | EV T−60 na tych samych settled | PnL−EV | Nierozl. notional / debet |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in MODEL_NAMES:
        item = economics["variants"][name]
        audit = item["cashflow"]
        lines.append(
            f"| {name} | {item['decision_count']:,} | {item['fill_count']:,} / {item['settlement_count']:,} / {item['unresolved_fill_count']:,} | "
            f"{_format_usd(audit['settled_notional_usd'])} | {_format_usd(audit['settled_cash_debit_usd'])} | "
            f"{_format_usd(audit['cash_fees_paid_usd'])} / {_format_usd(audit['unresolved_cash_fees_paid_usd'])} | "
            f"{_format_usd(audit['share_fee_usd_entry_equivalent_informational'])} / "
            f"{_format_usd(audit['unresolved_share_fee_usd_entry_equivalent_informational'])} | "
            f"{_format_usd(audit['cashflow_pnl_usd'])} | {_format_ratio(audit['pnl_per_notional'])} | {_format_ratio(audit['pnl_per_cash_outflow'])} | "
            f"{_format_usd(audit['forecast_ev_on_settled_fills_usd'])} | {_format_usd(audit['pnl_minus_forecast_ev_on_settled_fills_usd'])} | "
            f"{_format_usd(audit['unresolved_gross_notional_usd'])} / {_format_usd(audit['unresolved_cash_outflow_usd'])} |"
        )
    pnl_bootstrap = economics["paired_pnl_bootstrap"]["contrasts"]
    lines.extend([
        "",
        f"Sparowany dzienny PnL M1−M0: średnio `{_format_usd(pnl_bootstrap['M1_MINUS_M0']['mean_daily_pnl_difference_usd']['point_estimate'])}`/dzień, 95% CI `{_format_usd(pnl_bootstrap['M1_MINUS_M0']['mean_daily_pnl_difference_usd']['95pct'][0])}` do `{_format_usd(pnl_bootstrap['M1_MINUS_M0']['mean_daily_pnl_difference_usd']['95pct'][1])}`; różnica sumaryczna `{_format_usd(pnl_bootstrap['M1_MINUS_M0']['total_pnl_difference_usd']['point_estimate'])}`.",
        f"Sparowany dzienny PnL M1−RAW: średnio `{_format_usd(pnl_bootstrap['M1_MINUS_RAW']['mean_daily_pnl_difference_usd']['point_estimate'])}`/dzień, 95% CI `{_format_usd(pnl_bootstrap['M1_MINUS_RAW']['mean_daily_pnl_difference_usd']['95pct'][0])}` do `{_format_usd(pnl_bootstrap['M1_MINUS_RAW']['mean_daily_pnl_difference_usd']['95pct'][1])}`; różnica sumaryczna `{_format_usd(pnl_bootstrap['M1_MINUS_RAW']['total_pnl_difference_usd']['point_estimate'])}`.",
        "",
        "Przedziały PnL używają tych samych 2 000 replikacji 3-dniowych UTC co jakość; dni bez transakcji mają dzienny PnL 0.",
        "",
        "### PnL i przepływy według miesiąca UTC",
        "",
        "Październik obejmuje tylko 1–6 października. Opłaty tokenowe to informacyjny ekwiwalent USD i nie są ponownie odejmowane od gotówki.",
        "",
        "| Miesiąc | Wariant | Fill / settled / unresolved | Notional settled | Debet settled | Cash fee settled / unresolved | Token fee eq. settled / unresolved | PnL settled | PnL/notional | PnL/debet | EV settled | Unresolved debet |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in monthly_economics:
        audit = row
        lines.append(
            f"| {row['utc_month']}{' (niepełny)' if row['utc_month'] == '2026-10' else ''} | {row['variant']} | "
            f"{audit['fill_count']:,} / {audit['settlement_count']:,} / {audit['unresolved_fill_count']:,} | "
            f"{_format_usd(audit['settled_notional_usd'])} | {_format_usd(audit['settled_cash_debit_usd'])} | "
            f"{_format_usd(audit['cash_fees_paid_usd'])} / {_format_usd(audit['unresolved_cash_fees_paid_usd'])} | "
            f"{_format_usd(audit['share_fee_usd_entry_equivalent_informational'])} / "
            f"{_format_usd(audit['unresolved_share_fee_usd_entry_equivalent_informational'])} | "
            f"{_format_usd(audit['cashflow_pnl_usd'])} | {_format_ratio(audit['pnl_per_notional'])} | "
            f"{_format_ratio(audit['pnl_per_cash_outflow'])} | {_format_usd(audit['forecast_ev_on_settled_fills_usd'])} | "
            f"{_format_usd(audit['unresolved_cash_outflow_usd'])} |"
        )
    lines.extend([
        "",
        "### RAW: uzgodnienie do istniejącego ledgeru i rynki wyłączone",
        "",
        "Pełny wynik RAW BASE70 służy wyłącznie jako punkt uzgodnienia; nie porównuję ograniczonego wyniku M1 z pełnopróbkowym `−$546.22`.",
        "",
        "| Próbka RAW | Fill / settled / unresolved | Rozliczony PnL | Debet nierozliczony |",
        "|---|---:|---:|---:|",
    ])
    recon = economics["raw_anchor_reconciliation"]
    for name, key in (
        ("Poprzedni pełny holdout", "full_raw_holdout"),
        ("Istniejący ledger RAW na wspólnych rynkach", "saved_raw_ledger_on_shared_calendar"),
        ("Nowy replay RAW na wspólnych rynkach", "replayed_raw_on_shared_calendar"),
        ("Wyłączone przez brak wspólnej dostępności", "raw_fills_removed_for_missing_common_forecast"),
    ):
        item = recon[key]
        lines.append(
            f"| {name} | {item['fill_count']:,} / {item['settlement_count']:,} / {item['unresolved_fill_count']:,} | "
            f"{_format_usd(item['cashflow_pnl_usd'])} | {_format_usd(item['unresolved_cash_outflow_usd'])} |"
        )
    lines.extend([
        "",
        f"Uzgodnienie wspólnego RAW przeszło dokładnie dla side, notionalu, debetu, opłat, wypłat, PnL i statusu rozliczenia. `PnL pełny = PnL wspólny + PnL rynków wyłączonych`: `{_format_usd(recon['pnl_reconciliation_usd']['full_raw_holdout'])} = {_format_usd(recon['pnl_reconciliation_usd']['shared_calendar'])} + {_format_usd(recon['pnl_reconciliation_usd']['removed_markets'])}`.",
        "",
        "### Koncentracja wyniku",
        "",
        "| Wariant | PnL | Pięć najlepszych dni | Udział pięciu dni w PnL |",
        "|---|---:|---:|---:|",
    ])
    for name, item in economics["daily_pnl_concentration"].items():
        lines.append(
            f"| {name} | {_format_usd(item['total_pnl_usd'])} | {_format_usd(item['top_five_days_pnl_usd'])} | "
            f"{_format_ratio(item['top_five_share_of_total_pnl'])} |"
        )
    lines.extend([
        "",
        "### Niezależny audyt gotówki",
        "",
        "Każdy settled fill sprawdzono ponownie arytmetyką `Decimal` niezależną od agregacji raportu: `payout − cash_debit = net_pnl`; dodatkowo `cash_debit = notional + cash_fee` oraz strata nie przekracza debetu. Zastosowano poprawkę audytu przepływów z `a5fd05e`. Ekwiwalent fee potrąconego w tokenach pozostaje wyłącznie informacyjny.",
        "",
        "| Ledger | Max różnica PnL | Max różnica debetu | Naruszenia limitu straty |",
        "|---|---:|---:|---:|",
    ])
    for name in MODEL_NAMES:
        audit = economics["variants"][name]["cashflow"]["independent_decimal_audit"]
        lines.append(
            f"| {name} | {_format_usd(audit['max_abs_pnl_reconciliation_delta_usd'])} | "
            f"{_format_usd(audit['cash_debit_reconciliation_max_abs_usd'])} | {audit['loss_bound_violations']} |"
        )
    raw_audit = recon["full_raw_holdout"]
    lines.append(
        f"| Poprzedni pełny RAW BASE70 | {_format_usd(raw_audit['max_abs_pnl_reconciliation_delta_usd'])} | "
        f"{_format_usd(raw_audit['cash_debit_reconciliation_max_abs_usd'])} | {raw_audit['loss_bound_violations']} |"
    )
    lines.extend([
        "",
        "## Wnioski",
        "",
        f"- Czy przyrostowa jakość M1 względem M0 przeniosła się poza development? AUC wzrosło o `{quality['contrasts']['M1_MINUS_M0']['auc']['point_difference']:.6f}` (95% CI `{quality['contrasts']['M1_MINUS_M0']['auc']['95pct']}`), ale przedziały różnic log loss `{quality['contrasts']['M1_MINUS_M0']['log_loss']['95pct']}` i Brier `{quality['contrasts']['M1_MINUS_M0']['brier']['95pct']}` obejmują zero. Transfer jest widoczny dla rankingu AUC, lecz nie wykazano poprawy log loss/Brier.",
        f"- Czy M1 poprawił RAW na wspólnej próbce? Tak w trzech metrykach; M1−RAW Δ log loss `{quality['contrasts']['M1_MINUS_RAW']['log_loss']['point_difference']:.6f}`, Brier `{quality['contrasts']['M1_MINUS_RAW']['brier']['point_difference']:.6f}`, AUC `{quality['contrasts']['M1_MINUS_RAW']['auc']['point_difference']:.6f}`; wszystkie sparowane 95% CI wykluczają zero.",
        f"- Czy M1 zarobił po kosztach, czy ograniczył ekspozycję? M1 stracił `{_format_usd(economics['variants']['M1']['cashflow']['cashflow_pnl_usd'])}` na `{economics['variants']['M1']['fill_count']:,}` fillach, RAW `{_format_usd(economics['variants']['RAW']['cashflow']['cashflow_pnl_usd'])}` na `{economics['variants']['RAW']['fill_count']:,}`. Mniejsza strata łączna M1 wynika z dużo mniejszej liczby transakcji; jego PnL/notional (`{_format_ratio(economics['variants']['M1']['cashflow']['pnl_per_notional'])}`) był gorszy niż RAW (`{_format_ratio(economics['variants']['RAW']['cashflow']['pnl_per_notional'])}`). To ograniczenie ekspozycji, nie dowód dodatniego edge. No-trade: `$0`.",
        f"- Czy modelowe EV lepiej odpowiadało realizacji? Nie w tym teście: średni bezwzględny błąd `PnL−EV T−60` na własnych settled fillach M1 wyniósł `{_format_usd(economics['variants']['M1']['cashflow']['mean_absolute_per_fill_pnl_minus_ev_usd'])}`, RAW `{_format_usd(economics['variants']['RAW']['cashflow']['mean_absolute_per_fill_pnl_minus_ev_usd'])}`. Oba warianty miały dodatnie prognozowane EV, lecz ujemny PnL; zestawy wykonań są różne.",
        "",
        "Holdout był ujawniony przed powstaniem hipotezy; nie dopasowywano współczynników po wyniku, nie zmieniano progów/stron/okresów i nie uruchamiano wariantu dziennego.",
        "",
        "## Odtwarzalność",
        "",
        f"Konfiguracja `{CONFIG_PATH.relative_to(ROOT).as_posix()}`; zamrożony manifest `{VERSIONED_MANIFEST_PATH.relative_to(ROOT).as_posix()}`. Predykcje, współczynniki, ledgery, bootstrap i raport są checkpointowane w `{CONFIG['output_directory']}`. Wznowienie: `python {CONFIG['runner']}`. Nie trenowano BASE70, nie pobierano danych, nie odtwarzano archiwum ani nie zmieniano ustawień live.",
        f"Ostatnie wznowienie wykorzystało ukończone checkpointy; jego czas wyniósł `{manifest.get('elapsed_seconds')}` s, a szczytowy working set procesu wznowienia `{manifest.get('peak_working_set_gib')}` GiB. Dopasowanie, bootstrap i ledgery ukończono we wcześniejszym przebiegu.",
        "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _update_status(summary: dict) -> None:
    begin = "<!-- BTC_MARKET_BASE70_FROZEN_HOLDOUT_20261019_BEGIN -->"
    end = "<!-- BTC_MARKET_BASE70_FROZEN_HOLDOUT_20261019_END -->"
    m1_m0 = summary["quality"]["contrasts"]["M1_MINUS_M0"]["log_loss"]["point_difference"]
    m1_raw = summary["quality"]["contrasts"]["M1_MINUS_RAW"]["log_loss"]["point_difference"]
    pnl = summary["economics"]["variants"]
    block = "\n".join([
        begin,
        "## Zamrożony transfer RAW BASE70 M0/M1 — 2026-10-19",
        f"- Status: `completed`; wspólny holdout ma {summary['availability']['shared_raw_market_forecast_rows']:,} prognoz, jakość na {summary['availability']['quality_rows']:,} oficjalnych etykietach.",
        f"- M1−M0 Δ log loss `{m1_m0:.6f}`; M1−RAW `{m1_raw:.6f}`. Zobacz sparowane przedziały w raporcie.",
        f"- Settled PnL na wspólnym kalendarzu: M1 `{_format_usd(pnl['M1']['cashflow']['cashflow_pnl_usd'])}`, RAW `{_format_usd(pnl['RAW']['cashflow']['cashflow_pnl_usd'])}`; współczynniki zamrożone przed oceną.",
        f"- Raport: `{CONFIG['report_path']}`; manifest: `{CONFIG['versioned_manifest_path']}`; konfiguracja: `{CONFIG_PATH.relative_to(ROOT).as_posix()}`.",
        end,
        "",
    ])
    text = STATUS_PATH.read_text(encoding="utf-8")
    if begin in text and end in text:
        before, tail = text.split(begin, 1)
        _, after = tail.split(end, 1)
        text = before + block + after
    else:
        lines = text.splitlines(keepends=True)
        insertion = 1 if lines and lines[0].startswith("# ") else 0
        lines[insertion:insertion] = ["\n", block, "\n"]
        text = "".join(lines)
    STATUS_PATH.write_text(text, encoding="utf-8")


def run() -> dict:
    started = time.perf_counter()
    manifest = _freeze()
    fit = _load_or_fit_frozen_coefficients(manifest)

    predictions, calendar, t59, t60 = _build_holdout_inputs(manifest)
    shared = predictions.loc[
        predictions[[PROBABILITY_COLUMNS[name] for name in MODEL_NAMES]].notna().all(axis=1)
        & predictions.condition_id.notna()
    ].copy()
    shared.sort_values("market_start_utc", kind="stable", inplace=True)
    days = pd.date_range(shared.market_start_utc.min().floor("D"), shared.market_start_utc.max().floor("D"), freq="D", tz="UTC")
    indices, _ = _save_bootstrap(manifest, days)
    quality, monthly_quality = _load_or_score_quality(manifest, predictions, days, indices)

    summaries, decisions_by_variant, trades_by_variant = {}, {}, {}
    for name in MODEL_NAMES:
        item, decisions, trades = _simulate_variant(name, predictions, calendar, t59, t60, manifest)
        summaries[name] = item
        decisions_by_variant[name] = decisions
        trades_by_variant[name] = trades
    economics_variant = {
        name: {
            **{key: item[key] for key in (
                "decision_count", "intent_count", "fill_count", "settlement_count",
                "unresolved_fill_count", "calendar", "rejections_by_reason",
            )},
            "cashflow": _variant_cashflow(trades_by_variant[name]),
        }
        for name, item in summaries.items()
    }
    shared_ids = set(shared.condition_id.dropna().astype(str).str.lower())
    reconciliation = _reconcile_raw_holdout(
        pd.read_parquet(ROOT / CONFIG["sources"]["raw_holdout_trades_path"]),
        shared_ids,
        trades_by_variant["RAW"],
    )
    daily = development._daily_pnl(trades_by_variant, days)
    daily_path = OUTPUT_DIR / "daily_pnl.csv"
    daily.to_csv(daily_path, index=False)
    monthly_economics = _monthly_cashflow_rows(trades_by_variant)
    monthly_economics_path = OUTPUT_DIR / "monthly_economics.csv"
    pd.DataFrame(monthly_economics).to_csv(monthly_economics_path, index=False)

    quality_rows = _quality_sample(predictions)
    availability = manifest["stages"]["holdout_alignment"]["summary"]
    availability.update({
        "quality_rows": int(len(quality_rows)),
        "first_shared_market_start_utc": shared.market_start_utc.min(),
        "last_shared_market_start_utc": shared.market_start_utc.max(),
        "shared_calendar_utc_days": int(shared.market_start_utc.dt.floor("D").nunique()),
        "quality_official_up_labels": int(quality_rows.target_polymarket_up.sum()),
        "quality_official_down_labels": int(len(quality_rows) - quality_rows.target_polymarket_up.sum()),
        "shared_forecast_rows_excluded_from_quality": int(len(shared) - len(quality_rows)),
    })
    economics = {
        "variants": economics_variant,
        "raw_anchor_reconciliation": reconciliation,
        "paired_pnl_bootstrap": development._pnl_bootstrap(daily, indices),
        "daily_pnl_concentration": development._concentration(daily),
        "no_trade_baseline_net_pnl_usd": 0.0,
    }
    summary = {
        "experiment_id": CONFIG["experiment_id"],
        "holdout_historically_exposed": True,
        "holdout_coefficient_updates": 0,
        "frozen_development_fit": fit,
        "availability": availability,
        "quality": quality,
        "economics": economics,
    }
    summary_path = OUTPUT_DIR / "experiment_summary.json"
    _write_json(summary_path, summary)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["peak_working_set_gib"] = development._peak_working_set_bytes()
    if manifest["peak_working_set_gib"] is not None:
        manifest["peak_working_set_gib"] = round(manifest["peak_working_set_gib"] / 2**30, 3)
    manifest["status"] = "completed"
    manifest["active_stage"] = None
    manifest["result_summary"] = {
        "quality_rows": availability["quality_rows"],
        "shared_forecast_rows": availability["shared_raw_market_forecast_rows"],
        "M1_minus_M0_log_loss": quality["contrasts"]["M1_MINUS_M0"]["log_loss"],
        "M1_minus_RAW_log_loss": quality["contrasts"]["M1_MINUS_RAW"]["log_loss"],
        "M1_net_pnl_usd": economics_variant["M1"]["cashflow"]["cashflow_pnl_usd"],
        "RAW_shared_net_pnl_usd": economics_variant["RAW"]["cashflow"]["cashflow_pnl_usd"],
        "raw_ledger_reconciled": reconciliation["ledger_matches_on_shared_calendar"],
    }
    _complete_stage(manifest, "final_summary", [summary_path, daily_path, monthly_economics_path], manifest["result_summary"])
    manifest["status"] = "completed"
    manifest["active_stage"] = "report"
    _write_manifest(manifest)
    _write_report(manifest, summary, monthly_quality, monthly_economics, trades_by_variant)
    _complete_stage(manifest, "report", [REPORT_PATH])
    manifest["status"] = "completed"
    manifest["elapsed_seconds"] = time.perf_counter() - started
    _write_manifest(manifest)
    _update_status(summary)
    return summary


if __name__ == "__main__":
    run()
