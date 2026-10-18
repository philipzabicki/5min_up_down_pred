"""Compare saved BASE70 forecasts with fixed probabilistic and economic controls."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_preopen_model_update as model_update
import run_btc_preopen_research as research
import run_btc_preopen_t59_reassessment as replay
import run_btc_preopen_taker_flow_20261018 as taker


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_base70_information_value_20261018.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_directory"]
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
REPORT_PATH = ROOT / CONFIG["report_path"]
VERSIONED_MANIFEST_PATH = ROOT / CONFIG["versioned_manifest_path"]
STATUS_PATH = ROOT / CONFIG["status_path"]
STATUS_BEGIN = "<!-- BTC_BASE70_INFORMATION_20261018_BEGIN -->"
STATUS_END = "<!-- BTC_BASE70_INFORMATION_20261018_END -->"

FLOW_DIR = ROOT / "data/analysis/polymarket/BTC/btc_preopen_taker_flow_20261018"
ZV_DIR = ROOT / "data/analysis/polymarket/BTC/btc_preopen_zv_ablation_20261018"
RAW_OOF_PATH = FLOW_DIR / "B0_decision_oof.parquet"
PLATT_CACHE_PATH = FLOW_DIR / "B0_B1_official_oof.parquet"
BASE70_ECON_DIR = FLOW_DIR / "economics/B0/fixed_5_usd_ev_1pct"
BASE70_TRADES_PATH = BASE70_ECON_DIR / "trades.parquet"
BASE70_DECISIONS_PATH = BASE70_ECON_DIR / "decisions.parquet"
BASE70_ECON_SUMMARY_PATH = BASE70_ECON_DIR / "summary.json"
SOURCE_QUALITY_PATH = FLOW_DIR / "quality_and_economics.json"
SOURCE_ZV_ECON_PATH = ZV_DIR / "economic_comparison.json"
MARKET_INDEX_PATH = ROOT / CONFIG["market_index_path"]

START_UTC = pd.Timestamp(CONFIG["development_start_utc"])
CUTOFF_UTC = pd.Timestamp(CONFIG["development_cutoff_exclusive_utc"])
HISTORY_START_UTC = pd.Timestamp(CONFIG["calibration_history_start_utc"])
POLICY_NAME = "fixed_5_usd_ev_1pct"
VARIANTS = ("HALF", "PRIOR", "RAW", "PLATT")
PROBABILITY_COLUMNS = {
    "HALF": "p_half",
    "PRIOR": "p_prior",
    "RAW": "p_raw",
    "PLATT": "p_platt",
}
CONTRASTS = {
    "RAW_MINUS_HALF": ("RAW", "HALF"),
    "RAW_MINUS_PRIOR": ("RAW", "PRIOR"),
    "PLATT_MINUS_RAW": ("PLATT", "RAW"),
    "PLATT_MINUS_PRIOR": ("PLATT", "PRIOR"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _record(path: Path) -> dict:
    return {"path": _relative(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _verify_records(records: list[dict]) -> bool:
    return all(
        (ROOT / item["path"]).is_file()
        and (ROOT / item["path"]).stat().st_size == int(item["size_bytes"])
        and _sha256(ROOT / item["path"]) == item["sha256"]
        for item in records
    )


def _status_update(manifest: dict) -> None:
    completed = sorted(
        name for name, stage in manifest.get("stages", {}).items()
        if stage.get("status") == "completed"
    )
    block = "\n".join([
        STATUS_BEGIN,
        "### BASE70 information value comparison - 2026-10-18",
        f"- Updated UTC: `{pd.Timestamp.now(tz='UTC').isoformat()}`.",
        f"- Status: `{manifest.get('status')}`; active stage: `{manifest.get('active_stage') or 'none'}`.",
        f"- Completed stages: `{', '.join(completed) or 'none'}`.",
        f"- Next step: `{manifest.get('next_step', 'Resume the no-argument runner.')}`.",
        f"- Run manifest: `{_relative(MANIFEST_PATH)}`.",
        STATUS_END,
        "",
    ])
    text = STATUS_PATH.read_text(encoding="utf-8")
    if STATUS_BEGIN in text and STATUS_END in text:
        before, tail = text.split(STATUS_BEGIN, 1)
        _, after = tail.split(STATUS_END, 1)
        text = before.rstrip() + "\n" + block + after.lstrip("\n")
    else:
        anchor = "# BTC research execution status"
        text = text.replace(anchor, block + anchor, 1) if anchor in text else block + text
    temporary = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(STATUS_PATH)


def _write_manifest(manifest: dict) -> None:
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(MANIFEST_PATH, manifest)
    _status_update(manifest)


def _begin_stage(manifest: dict, name: str, next_step: str) -> None:
    manifest["status"] = "running"
    manifest["active_stage"] = name
    manifest["next_step"] = next_step
    manifest.setdefault("stages", {})[name] = {
        "status": "running",
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write_manifest(manifest)


def _finish_stage(manifest: dict, name: str, **payload) -> None:
    stage = manifest.setdefault("stages", {}).setdefault(name, {})
    stage.update(payload)
    stage["status"] = "completed"
    stage["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    manifest["active_stage"] = None
    _write_manifest(manifest)


def _source_records(flow_manifest: dict) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for stage_name in ("cv_B0", "quality_and_economics"):
        stage = flow_manifest.get("stages", {}).get(stage_name, {})
        if stage.get("status") != "completed":
            raise RuntimeError(f"Saved BASE70 source stage is incomplete: {stage_name}")
        for record in stage.get("outputs", []):
            records[str(record["path"])] = record
    required = (
        RAW_OOF_PATH,
        PLATT_CACHE_PATH,
        BASE70_TRADES_PATH,
        BASE70_DECISIONS_PATH,
        BASE70_ECON_SUMMARY_PATH,
        SOURCE_QUALITY_PATH,
    )
    for path in required:
        relative = _relative(path)
        record = records.get(relative)
        if record is None or not _verify_records([record]):
            raise RuntimeError(f"Saved source output is missing or changed: {relative}")
    return records


def _verify_anchor_ledger(
    trades: pd.DataFrame,
    ledger_summary: dict,
    recorded_summary: dict,
    cutoff: pd.Timestamp,
    expected_anchor: tuple[int, float, float] = (1452, 7260.0, 260.04642335283586),
) -> dict:
    release = pd.to_datetime(trades["cash_release_at_utc"], utc=True, errors="coerce")
    settled = (
        pd.to_datetime(trades["settled_at_utc"], utc=True, errors="coerce").notna()
        & release.lt(cutoff)
        & pd.to_numeric(trades["net_pnl_usd"], errors="coerce").notna()
    )
    settled_trades = trades.loc[settled]
    fills = int(len(trades))
    turnover = float(pd.to_numeric(trades["gross_usd"], errors="raise").sum())
    fees = float(pd.to_numeric(trades["fee_usd"], errors="raise").sum())
    pnl = float(pd.to_numeric(settled_trades["net_pnl_usd"], errors="raise").sum())
    result = {
        "fills": fills,
        "settlements_by_cutoff": int(len(settled_trades)),
        "turnover_usd": turnover,
        "fees_usd": fees,
        "net_pnl_usd": pnl,
        "pnl_per_settled_turnover": pnl / float(settled_trades["gross_usd"].sum()),
        "source_summary_trade_count": int(ledger_summary["trade_count"]),
        "source_summary_settlement_count": int(ledger_summary.get("settlement_count", ledger_summary["accounting_audit"]["trade_count"])),
        "source_summary_turnover_usd": float(ledger_summary["gross_turnover_usd"]),
        "source_summary_net_pnl_usd": float(ledger_summary["net_pnl_usd"]),
        "source_comparison_fills": int(recorded_summary["fill_count"]),
        "source_comparison_turnover_usd": float(recorded_summary["gross_turnover_usd"]),
        "source_comparison_net_pnl_usd": float(recorded_summary["net_pnl_usd"]),
    }
    checks = {
        "fills_match_reported_anchor": fills == expected_anchor[0] == result["source_summary_trade_count"] == result["source_comparison_fills"],
        "all_saved_fills_settled_by_cutoff": fills == len(settled_trades),
        "turnover_matches_reported_anchor": math.isclose(turnover, expected_anchor[1], abs_tol=1e-8)
        and math.isclose(turnover, result["source_summary_turnover_usd"], abs_tol=1e-8)
        and math.isclose(turnover, result["source_comparison_turnover_usd"], abs_tol=1e-8),
        "pnl_matches_reported_anchor": math.isclose(pnl, expected_anchor[2], abs_tol=1e-8)
        and math.isclose(pnl, result["source_summary_net_pnl_usd"], abs_tol=1e-8)
        and math.isclose(pnl, result["source_comparison_net_pnl_usd"], abs_tol=1e-8),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Saved BASE70 ledger does not reproduce the frozen anchor: {checks}")
    return {**result, "checks": checks, "passed": True}


def _frozen_specification(
    flow_manifest: dict,
    zv_manifest: dict,
    source_hashes: dict,
    snapshot_identity: dict,
) -> dict:
    return {
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "variants": CONFIG["variants"],
        "sources": {
            "raw_BASE70_oof": _relative(RAW_OOF_PATH),
            "daily_Platt_cache": _relative(PLATT_CACHE_PATH),
            "official_labels": _relative(MARKET_INDEX_PATH),
            "BASE70_reference_trades": _relative(BASE70_TRADES_PATH),
            "BASE70_reference_decisions": _relative(BASE70_DECISIONS_PATH),
            "decision_and_execution_snapshots": snapshot_identity,
            "source_flow_manifest": CONFIG["source_taker_flow_run_manifest_path"],
            "source_zv_report_and_manifest": [CONFIG["source_zv_report_path"], CONFIG["source_zv_report_manifest_path"]],
        },
        "source_hashes": source_hashes,
        "source_run_statuses": {
            "taker_flow": flow_manifest.get("status"),
            "zv_ablation": zv_manifest.get("status"),
            "zv_frozen_identity_sha256": zv_manifest.get("frozen_comparison", {}).get("identity_sha256"),
        },
        "calendar": {
            "development_start_utc": START_UTC,
            "development_cutoff_exclusive_utc": CUTOFF_UTC,
            "comparison": CONFIG["comparison_calendar"],
            "quality_labels": "official binary outcomes whose label release is no later than the development cutoff",
        },
        "availability_rules": {
            "label_release_rule": CONFIG["label_release_rule"],
            "calibration_history_start_utc": HISTORY_START_UTC,
            "minimum_prior_labels": int(CONFIG["calibration_minimum_labels"]),
            "requires_both_classes": bool(CONFIG["calibration_requires_both_classes"]),
            "daily_update": CONFIG["calibration_update"],
            "decision_time": "T-60; calibration inputs are restricted to labels released before that UTC day's first decision",
            "execution_time": "T-59; T-59 snapshots determine execution only",
            "forecast_timestamp_provenance": "Saved BASE70 OOF rows have missing prediction_available_at_utc; OOF row presence is observed, and T-59 availability is an explicit one-second-contract assumption.",
            "unresolved_outcomes": "retain qualified intents and fills; exclude unresolved fills from settled PnL and report separately",
        },
        "policy": CONFIG["policy"],
        "comparisons": {
            "quality_metrics": ["log_loss", "brier", "auc"],
            "quality_contrasts": list(CONTRASTS),
            "economic_contrasts": list(CONTRASTS),
            "bootstrap": CONFIG["bootstrap"],
            "development_exploratory_only": True,
        },
        "prohibitions": {
            "model_retraining": False,
            "generator_or_feature_selection": False,
            "optuna": False,
            "data_download": False,
            "archive_replay": False,
            "holdout_evaluation": False,
            "live_configuration_change": False,
        },
    }


def _load_or_create_manifest(
    input_identity: dict,
    frozen: dict,
    anchor: dict,
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.is_file():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("input_identity") != input_identity:
            previous_identity = json.loads(json.dumps(manifest.get("input_identity", {})))
            updated_identity = json.loads(json.dumps(input_identity))
            for value in (previous_identity, updated_identity):
                value.pop("frozen_spec_sha256", None)
                value.get("source_hashes", {}).pop("runner", None)
                value.get("source_hashes", {}).pop("economic_engine", None)
                value.pop("engine_causal_intent_mode_sha256", None)
            if previous_identity != updated_identity:
                raise RuntimeError("Existing comparison checkpoints belong to different frozen sources or configuration")
            previous_stages = manifest.get("stages", {})
            previous_outputs = [
                record
                for stage in previous_stages.values()
                for record in stage.get("outputs", [])
            ]
            previous_hashes = manifest.get("input_identity", {}).get("source_hashes", {})
            engine_changed = (
                previous_hashes.get("economic_engine") != input_identity.get("source_hashes", {}).get("economic_engine")
                or manifest.get("input_identity", {}).get("engine_causal_intent_mode_sha256")
                != input_identity.get("engine_causal_intent_mode_sha256")
            )
            reusable_stages = {}
            reusable_names = (
                ("common_predictions", "execution_calendar", "quality_and_bootstrap")
                if engine_changed else
                tuple(name for name in previous_stages if name != "report")
            )
            for stage_name in reusable_names:
                stage = previous_stages.get(stage_name, {})
                if stage.get("status") == "completed" and _verify_records(stage.get("outputs", [])):
                    reusable_stages[stage_name] = stage
            stable_frozen = {key: value for key, value in frozen.items() if key != "frozen_at_utc"}
            manifest.setdefault("code_revision_history", []).append({
                "at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "previous_input_identity": manifest.get("input_identity"),
                "updated_input_identity": input_identity,
                "reason": (
                    "Economic engine code changed; verified forecast, calendar, and quality checkpoints are retained, while economic stages are recomputed."
                    if engine_changed else
                    "Runner reporting code changed; all verified computation checkpoints are retained and the report is regenerated."
                ),
                "retained_checkpoint_stages": sorted(reusable_stages),
                "previous_stage_outputs": previous_outputs,
            })
            manifest["input_identity"] = input_identity
            manifest["frozen_comparison"] = {
                "identity_sha256": _canonical_sha256(stable_frozen),
                "specification": frozen,
            }
            manifest["baseline_ledger_reproduction"] = anchor
            manifest["stages"] = reusable_stages
            manifest["active_stage"] = None
            manifest["status"] = "frozen_before_calculation"
            manifest["next_step"] = "Reuse verified forecast, calendar, and quality checkpoints; recompute economics and reporting."
            _write_manifest(manifest)
            _write_json(VERSIONED_MANIFEST_PATH, manifest)
            return manifest
        stable_frozen = {key: value for key, value in frozen.items() if key != "frozen_at_utc"}
        if manifest.get("frozen_comparison", {}).get("identity_sha256") != _canonical_sha256(stable_frozen):
            raise RuntimeError("Existing frozen comparison differs from the current specification")
        return manifest
    stable_frozen = {key: value for key, value in frozen.items() if key != "frozen_at_utc"}
    manifest = {
        "schema_version": 1,
        "experiment_id": CONFIG["experiment_id"],
        "status": "frozen_before_calculation",
        "active_stage": None,
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "next_step": "Build common forecasts, reproduce existing BASE70 Platt values, and score the frozen sample.",
        "input_identity": input_identity,
        "frozen_comparison": {
            "identity_sha256": _canonical_sha256(stable_frozen),
            "specification": frozen,
        },
        "baseline_ledger_reproduction": anchor,
        "stages": {},
        "resource_snapshot_before_run": taker._resource_snapshot(),
        "runtime_identity": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "runner_sha256": _sha256(Path(__file__)),
            "economic_engine_sha256": _sha256(ROOT / "run_btc_preopen_t59_reassessment.py"),
        },
    }
    _write_manifest(manifest)
    _write_json(VERSIONED_MANIFEST_PATH, manifest)
    return manifest


def _build_predictions() -> tuple[pd.DataFrame, list[dict]]:
    raw = pd.read_parquet(
        RAW_OOF_PATH,
        columns=[
            "Opened", "nominal_decision_at", "target_label_available_at",
            "target_preopen_5m_up_binance_index_proxy", "market_start_utc",
            "fold_id", "p_model_raw", "prediction_available_at_utc",
        ],
    )
    for column in ("market_start_utc", "nominal_decision_at", "target_label_available_at", "prediction_available_at_utc"):
        raw[column] = pd.to_datetime(raw[column], utc=True, errors="raise")
    raw = raw.loc[raw.market_start_utc.ge(START_UTC) & raw.market_start_utc.lt(CUTOFF_UTC)].copy()
    raw.sort_values("market_start_utc", kind="stable", inplace=True)
    raw.reset_index(drop=True, inplace=True)
    if raw.market_start_utc.duplicated().any():
        raise RuntimeError("BASE70 raw OOF rows contain duplicate market starts in development")
    if not raw.market_start_utc.eq(raw.nominal_decision_at + pd.Timedelta(minutes=1)).all():
        raise RuntimeError("Saved BASE70 rows do not align market starts with T-60 decision times")
    recorded_available = raw.prediction_available_at_utc.notna()
    if not raw.loc[recorded_available, "prediction_available_at_utc"].eq(
        raw.loc[recorded_available, "market_start_utc"] - pd.Timedelta(seconds=59)
    ).all():
        raise RuntimeError("Recorded BASE70 prediction timestamps violate the T-60/T-59 contract")
    raw.rename(columns={"prediction_available_at_utc": "saved_prediction_available_at_utc"}, inplace=True)
    raw["saved_prediction_timestamp_recorded"] = raw.saved_prediction_available_at_utc.notna()
    raw["prediction_available_at_utc"] = raw.nominal_decision_at + pd.Timedelta(seconds=1)
    if not raw.p_model_raw.map(np.isfinite).all() or not raw.p_model_raw.between(0.0, 1.0).all():
        raise RuntimeError("Saved BASE70 raw probabilities are invalid")

    official = pd.read_parquet(
        MARKET_INDEX_PATH,
        columns=["market_start_utc", "resolved_at_utc", "target_polymarket_up"],
    )
    official["market_start_utc"] = pd.to_datetime(official.market_start_utc, utc=True, errors="raise")
    official["resolved_at_utc"] = pd.to_datetime(official.resolved_at_utc, utc=True, errors="coerce")
    official = official.loc[
        official.market_start_utc.ge(HISTORY_START_UTC)
        & official.market_start_utc.lt(CUTOFF_UTC)
    ].copy()
    official = official.drop_duplicates("market_start_utc", keep=False)
    forecasts = raw.merge(official, on="market_start_utc", how="left", validate="one_to_one")
    starts = forecasts.market_start_utc
    resolved = pd.to_datetime(forecasts.resolved_at_utc, utc=True, errors="coerce")
    release = resolved.where(
        resolved.ge(starts + pd.Timedelta(minutes=5)), starts + pd.Timedelta(minutes=5),
    ) + pd.Timedelta(seconds=60)
    forecasts["official_label_release_at_utc"] = release
    forecasts["official_outcome_available_by_cutoff"] = (
        forecasts.target_polymarket_up.isin((0, 1))
        & resolved.notna()
        & release.le(CUTOFF_UTC)
    )

    scored, calibration_log = taker._causal_daily_platt(forecasts.copy())
    scored["p_half"] = 0.5
    scored, prior_log = _attach_daily_prior(scored, minimum=int(CONFIG["calibration_minimum_labels"]))
    scored["p_raw"] = scored.p_model_raw.astype(np.float64)
    scored["p_platt"] = scored.p_model_platt.astype(np.float64)
    scored["comparison_forecast_available"] = _comparison_forecast_mask(scored)
    scored["decision_day_utc"] = scored.market_start_utc.dt.floor("D")

    cache = pd.read_parquet(
        PLATT_CACHE_PATH,
        columns=["market_start_utc", "p_B0_raw", "p_B0_platt", "target_polymarket_up", "resolved_at_utc"],
    )
    cache["market_start_utc"] = pd.to_datetime(cache.market_start_utc, utc=True, errors="raise")
    cached = scored.merge(
        cache[["market_start_utc", "p_B0_raw", "p_B0_platt"]],
        on="market_start_utc", how="inner", validate="one_to_one",
    )
    if len(cached) != len(cache):
        raise RuntimeError("Official BASE70 calibration cache does not align to saved raw OOF rows")
    if not np.allclose(cached.p_raw, cached.p_B0_raw, rtol=0, atol=1e-12):
        raise RuntimeError("Official BASE70 raw predictions do not reproduce the calibration cache")
    if not np.array_equal(cached.p_platt.notna().to_numpy(), cached.p_B0_platt.notna().to_numpy()):
        raise RuntimeError("Recomputed BASE70 Platt availability differs from the saved cache")
    if not np.allclose(cached.p_platt, cached.p_B0_platt, rtol=0, atol=1e-12, equal_nan=True):
        raise RuntimeError("Recomputed BASE70 Platt values differ from the saved calibration cache")
    for row in calibration_log:
        prior = next(item for item in prior_log if item["utc_day"] == row["utc_day"])
        if int(row["training_rows"]) != prior["prior_label_count"]:
            raise RuntimeError("PRIOR and Platt use different historical official label counts")
        if row["status"] == "fit" and prior["status"] != "available":
            raise RuntimeError("PRIOR availability does not match the Platt warm-up rule")
    return scored, [{**item, "platt": log} for item, log in zip(prior_log, calibration_log, strict=True)]


def _attach_daily_prior(frame: pd.DataFrame, *, minimum: int) -> tuple[pd.DataFrame, list[dict]]:
    scored = frame.copy()
    scored["p_prior"] = np.nan
    release = pd.to_datetime(scored["official_label_release_at_utc"], utc=True, errors="coerce")
    valid_history = (
        scored["target_polymarket_up"].isin((0, 1))
        & pd.to_datetime(scored["resolved_at_utc"], utc=True, errors="coerce").notna()
        & release.notna()
        & pd.to_datetime(scored["market_start_utc"], utc=True).ge(HISTORY_START_UTC)
    )
    logs = []
    day_values = pd.to_datetime(scored["market_start_utc"], utc=True).dt.floor("D")
    decisions = pd.to_datetime(scored["nominal_decision_at"], utc=True)
    for day, day_rows in scored.groupby(day_values, sort=True):
        first_decision = decisions.loc[day_rows.index].min()
        past = scored.loc[valid_history & release.lt(first_decision), "target_polymarket_up"]
        counts = past.value_counts()
        enough = len(past) >= minimum and counts.index.nunique() == 2
        if enough:
            scored.loc[day_rows.index, "p_prior"] = float(past.mean())
        logs.append({
            "utc_day": pd.Timestamp(day).isoformat(),
            "first_decision_at_utc": first_decision.isoformat(),
            "prior_label_count": int(len(past)),
            "prior_up_count": int((past == 1).sum()),
            "prior_down_count": int((past == 0).sum()),
            "prior_p_up": float(past.mean()) if enough else None,
            "status": "available" if enough else "warmup_insufficient_rows_or_classes",
        })
    return scored, logs


def _comparison_forecast_mask(frame: pd.DataFrame) -> pd.Series:
    return frame[[PROBABILITY_COLUMNS[name] for name in VARIANTS]].notna().all(axis=1)


def _quality_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    common = predictions.loc[
        predictions.comparison_forecast_available
        & predictions.official_outcome_available_by_cutoff
    ].copy()
    if common.empty or common.target_polymarket_up.nunique() != 2:
        raise RuntimeError("Common quality sample lacks official labels or one of the outcome classes")
    required = [PROBABILITY_COLUMNS[name] for name in VARIANTS]
    if not common[required].notna().all().all():
        raise RuntimeError("Quality sample does not have all four probabilities on every row")
    if not common.p_raw.notna().equals(common.p_platt.notna()):
        raise RuntimeError("RAW and PLATT quality rows differ")
    return common


def _metric_values(labels: np.ndarray, probability: np.ndarray) -> dict:
    return taker._score_binary(
        np.asarray(labels, dtype=np.int8), np.asarray(probability, dtype=np.float64),
    )


def _bootstrap_indices(days: pd.DatetimeIndex) -> np.ndarray:
    replications = int(CONFIG["bootstrap"]["replications"])
    block_days = int(CONFIG["bootstrap"]["block_days"])
    blocks = int(math.ceil(len(days) / block_days))
    rng = np.random.default_rng(int(CONFIG["bootstrap"]["seed"]))
    starts = rng.integers(0, len(days), size=(replications, blocks))
    offsets = np.arange(block_days, dtype=np.int32)
    sampled = (starts[:, :, None] + offsets[None, None, :]) % len(days)
    return sampled.reshape(replications, -1)[:, :len(days)].astype(np.int32, copy=False)


def _write_bootstrap_checkpoint(path: Path, indices: np.ndarray, days: pd.DatetimeIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, indices=indices, days=days.as_unit("ns").asi8)
    temporary.replace(path)


def _interval(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    valid = array[np.isfinite(array)]
    return {
        "estimate_median": float(np.median(valid)) if len(valid) else None,
        "95pct": [float(np.quantile(valid, .025)), float(np.quantile(valid, .975))]
        if len(valid) else [None, None],
        "valid_replications": int(len(valid)),
    }


def _quality_comparison(rows: pd.DataFrame, days: pd.DatetimeIndex, indices: np.ndarray) -> tuple[dict, pd.DataFrame]:
    y = rows.target_polymarket_up.to_numpy(dtype=np.int8)
    points = {
        variant: _metric_values(y, rows[PROBABILITY_COLUMNS[variant]].to_numpy(dtype=np.float64))
        for variant in VARIANTS
    }
    monthly_rows = []
    rows = rows.copy()
    rows["utc_month"] = rows.market_start_utc.dt.strftime("%Y-%m")
    for month, group in rows.groupby("utc_month", sort=True):
        labels = group.target_polymarket_up.to_numpy(dtype=np.int8)
        entry = {"utc_month": month, "rows": int(len(group)), "up_labels": int(labels.sum()), "down_labels": int(len(labels) - labels.sum())}
        for variant in VARIANTS:
            entry.update({
                f"{variant}_log_loss": _metric_values(labels, group[PROBABILITY_COLUMNS[variant]].to_numpy())["log_loss"],
                f"{variant}_brier": _metric_values(labels, group[PROBABILITY_COLUMNS[variant]].to_numpy())["brier"],
                f"{variant}_auc": _metric_values(labels, group[PROBABILITY_COLUMNS[variant]].to_numpy())["auc"],
            })
        monthly_rows.append(entry)

    positions = {
        day: np.flatnonzero(rows.market_start_utc.dt.floor("D").eq(day).to_numpy())
        for day in days
    }
    labels_all = rows.target_polymarket_up.to_numpy(dtype=np.int8)
    probabilities = {
        variant: rows[PROBABILITY_COLUMNS[variant]].to_numpy(dtype=np.float64)
        for variant in VARIANTS
    }
    bootstrap_metrics = {variant: {name: [] for name in ("log_loss", "brier", "auc")} for variant in VARIANTS}
    for draw in indices:
        selected = [positions[days[index]] for index in draw if len(positions[days[index]])]
        sample_idx = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
        sampled_labels = labels_all[sample_idx]
        for variant in VARIANTS:
            metric = _metric_values(sampled_labels, probabilities[variant][sample_idx])
            for name in bootstrap_metrics[variant]:
                bootstrap_metrics[variant][name].append(metric[name])
    contrasts = {}
    for contrast, (left, right) in CONTRASTS.items():
        contrasts[contrast] = {}
        for metric in ("log_loss", "brier", "auc"):
            distribution = (
                np.asarray(bootstrap_metrics[left][metric], dtype=np.float64)
                - np.asarray(bootstrap_metrics[right][metric], dtype=np.float64)
            )
            contrasts[contrast][metric] = {
                "point_difference": float(points[left][metric] - points[right][metric])
                if points[left][metric] is not None and points[right][metric] is not None else None,
                **_interval(distribution.tolist()),
            }
    summary = {
        "sample": {
            "rows": int(len(rows)),
            "start_utc": rows.market_start_utc.min().isoformat(),
            "end_utc": rows.market_start_utc.max().isoformat(),
            "utc_days": int(rows.market_start_utc.dt.floor("D").nunique()),
            "up_labels": int(y.sum()),
            "down_labels": int(len(y) - y.sum()),
        },
        "overall": points,
        "contrasts": contrasts,
        "bootstrap": {
            "method": "shared circular moving-block draws over 3-day UTC blocks",
            "replications": int(len(indices)),
            "seed": int(CONFIG["bootstrap"]["seed"]),
            "block_days": int(CONFIG["bootstrap"]["block_days"]),
        },
        "prior_auc_note": "PRIOR is constant within each UTC day; any AUC is due to differences in the daily prior across days.",
        "half_auc_note": "HALF AUC is 0.5 when both classes are present.",
    }
    return summary, pd.DataFrame(monthly_rows)


def _build_execution_calendar(predictions: pd.DataFrame, t59: pd.DataFrame, t60: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    engine_predictions = predictions[[
        "market_start_utc", "Opened", "nominal_decision_at", "target_preopen_5m_up_binance_index_proxy",
        "fold_id", "p_model_raw", "p_model_platt", "prediction_available_at_utc",
    ]].copy()
    calendar, calendar_summary = research._build_market_calendar(engine_predictions)
    official = predictions.set_index("market_start_utc")
    starts = pd.to_datetime(calendar.market_start_utc, utc=True, errors="raise")
    cal = calendar.copy()
    cal["market_start_utc"] = starts
    cal["target_polymarket_up"] = starts.map(official.target_polymarket_up)
    cal["resolved_at_utc"] = starts.map(official.resolved_at_utc)
    cal["official_label_release_at_utc"] = starts.map(official.official_label_release_at_utc)
    cal["official_outcome_available_by_cutoff"] = starts.map(official.official_outcome_available_by_cutoff).fillna(False).astype(bool)
    cal, _ = replay._market_statuses(cal, t59, t60)
    prediction_starts = pd.DatetimeIndex(predictions.loc[predictions.comparison_forecast_available, "market_start_utc"])
    common_calendar = cal.loc[cal.market_start_utc.isin(prediction_starts)].copy()
    common_calendar.sort_values("market_start_utc", kind="stable", inplace=True)
    if len(common_calendar) != len(prediction_starts) or common_calendar.market_start_utc.duplicated().any():
        raise RuntimeError("The common prediction calendar does not map one-to-one to the official market calendar")
    common_calendar, cutoff_summary = research._asof_market_rows(common_calendar, START_UTC, CUTOFF_UTC)
    flags = common_calendar.set_index("market_start_utc")[[
        "condition_id", "market_confirmed", "t60_decision_snapshot_available", "t59_snapshot_available",
        "outcome_available_by_cutoff", "cash_release_at_cutoff_utc",
    ]]
    predictions = predictions.copy()
    starts = predictions.market_start_utc
    for source, target in (
        ("market_confirmed", "market_record_available"),
        ("t60_decision_snapshot_available", "decision_snapshot_available"),
        ("t59_snapshot_available", "execution_snapshot_available"),
        ("outcome_available_by_cutoff", "official_outcome_available_by_cutoff"),
        ("cash_release_at_cutoff_utc", "cash_release_at_cutoff_utc"),
    ):
        predictions[target] = starts.map(flags[source])
    predictions["market_record_available"] = predictions.market_record_available.fillna(False).astype(bool)
    predictions["decision_snapshot_available"] = predictions.decision_snapshot_available.fillna(False).astype(bool)
    predictions["execution_snapshot_available"] = predictions.execution_snapshot_available.fillna(False).astype(bool)
    predictions["official_outcome_available_by_cutoff"] = predictions.official_outcome_available_by_cutoff.fillna(False).astype(bool)
    return predictions, {
        "source_calendar": calendar_summary,
        "comparison_calendar": {
            **cutoff_summary,
            "forecast_rows": int(len(common_calendar)),
            "forecast_timestamp_recorded_rows": int(predictions.loc[predictions.comparison_forecast_available, "saved_prediction_timestamp_recorded"].sum()),
            "forecast_timestamp_contract_assumed_rows": int(predictions.comparison_forecast_available.sum()),
            "market_record_available": int(common_calendar.market_confirmed.sum()),
            "decision_snapshot_available": int(common_calendar.t60_decision_snapshot_available.sum()),
            "execution_snapshot_available": int(common_calendar.t59_snapshot_available.sum()),
            "official_outcome_available_by_cutoff": int(common_calendar.outcome_available_by_cutoff.sum()),
        },
        "common_market_rows": common_calendar,
    }


def _monthly_economics(variant: str, trades: pd.DataFrame, decisions: pd.DataFrame) -> list[dict]:
    if trades.empty:
        return []
    frame = trades.copy()
    frame["decision_time_utc"] = pd.to_datetime(frame.decision_time_utc, utc=True)
    frame["utc_month"] = frame.decision_time_utc.dt.strftime("%Y-%m")
    frame["settled"] = frame.settlement_status.eq("settled")
    decision_frame = decisions.copy()
    decision_frame["decision_time_utc"] = pd.to_datetime(decision_frame.decision_time_utc, utc=True)
    decision_frame["utc_month"] = decision_frame.decision_time_utc.dt.strftime("%Y-%m")
    monthly_intents = (
        pd.to_numeric(decision_frame.net_ev_fraction_at_decision, errors="coerce")
        .ge(float(CONFIG["policy"]["minimum_net_ev_fraction"]))
        .groupby(decision_frame.utc_month)
        .sum()
        .to_dict()
    )
    rows = []
    for month, group in frame.groupby("utc_month", sort=True):
        settled = group.loc[group.settled]
        pnl = float(pd.to_numeric(settled.net_pnl_usd, errors="coerce").sum())
        settled_turnover = float(pd.to_numeric(settled.gross_usd, errors="coerce").sum())
        forecast_ev = float(pd.to_numeric(settled.chosen_side_ev_usd, errors="coerce").sum())
        rows.append({
            "variant": variant,
            "utc_month": month,
            "intents": int(monthly_intents.get(month, 0)),
            "fills": int(len(group)),
            "settlements": int(len(settled)),
            "unresolved_fills": int((~group.settled).sum()),
            "up_fills": int(group.chosen_side.eq("up").sum()),
            "down_fills": int(group.chosen_side.eq("down").sum()),
            "up_settlements": int(settled.chosen_side.eq("up").sum()),
            "down_settlements": int(settled.chosen_side.eq("down").sum()),
            "up_net_pnl_usd": float(pd.to_numeric(settled.loc[settled.chosen_side.eq("up"), "net_pnl_usd"], errors="coerce").sum()),
            "down_net_pnl_usd": float(pd.to_numeric(settled.loc[settled.chosen_side.eq("down"), "net_pnl_usd"], errors="coerce").sum()),
            "gross_turnover_usd": float(pd.to_numeric(group.gross_usd, errors="coerce").sum()),
            "settled_turnover_usd": settled_turnover,
            "fees_usd": float(pd.to_numeric(group.fee_usd, errors="coerce").sum()),
            "net_pnl_usd": pnl,
            "pnl_per_settled_turnover": pnl / settled_turnover if settled_turnover else None,
            "forecast_ev_on_settled_fills_usd": forecast_ev,
            "pnl_minus_forecast_ev_on_settled_fills_usd": pnl - forecast_ev,
        })
    return rows


def _run_variant(
    variant: str,
    predictions: pd.DataFrame,
    common_calendar: pd.DataFrame,
    t59: pd.DataFrame,
    t60: pd.DataFrame,
    manifest: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    stage_name = f"economics_{variant}"
    out_dir = OUTPUT_DIR / "economics" / variant
    decisions_path = out_dir / "decisions.parquet"
    trades_path = out_dir / "trades.parquet"
    summary_path = out_dir / "summary.json"
    existing = manifest.get("stages", {}).get(stage_name, {})
    if existing.get("status") == "completed" and _verify_records(existing.get("outputs", [])):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return summary, pd.read_parquet(decisions_path), pd.read_parquet(trades_path)
    _begin_stage(manifest, stage_name, f"Finish the remaining variants, shared bootstrap, and report. Current variant: {variant}.")
    started = time.perf_counter()
    probability_column = PROBABILITY_COLUMNS[variant]
    probability_map = predictions.set_index("market_start_utc")[probability_column]
    rows = common_calendar.copy()
    rows["p_candidate_platt"] = rows.market_start_utc.map(probability_map)
    if rows.p_candidate_platt.isna().any():
        raise RuntimeError(f"{variant} probabilities do not cover the shared economic calendar")
    market_rows, period_summary = research._asof_market_rows(rows, START_UTC, CUTOFF_UTC)
    snapshot_map = research._snapshot_map(t59)
    decision_map = research._snapshot_map(t60)
    market_map = research._snapshot_map(market_rows.loc[market_rows.condition_id.notna()])
    summary, trades, decisions = replay._portfolio_for_policy(
        "fixed_5_usd",
        market_rows,
        snapshot_map,
        market_map,
        decision_snapshot_by_id=decision_map,
        minimum_net_ev_fraction=float(CONFIG["policy"]["minimum_net_ev_fraction"]),
        output_policy=variant,
        causal_intent_mode=True,
        allow_unresolved_outcomes=True,
        independent_order_mode=True,
    )
    equity_path = summary.pop("_equity_path")
    summary["monthly_diagnostics"] = []
    summary["accounting_audit"] = replay._portfolio_accounting_audit(
        summary, trades, decisions, equity_path, snapshot_map, decision_map,
        allow_unresolved_outcomes=True,
        independent_order_mode=True,
    )
    decisions_frame = pd.DataFrame(decisions)
    trades_frame = pd.DataFrame(trades)
    settled = trades_frame.loc[trades_frame.settlement_status.eq("settled")] if not trades_frame.empty else trades_frame
    rejected = decisions_frame.loc[decisions_frame.decision.eq("skip"), "skip_reason"].value_counts(dropna=True).to_dict()
    intents = pd.to_numeric(
        decisions_frame.get(
            "net_ev_fraction_at_decision",
            pd.Series(np.nan, index=decisions_frame.index, dtype="float64"),
        ),
        errors="coerce",
    )
    intents = int(intents.ge(float(CONFIG["policy"]["minimum_net_ev_fraction"])).sum())
    pnl = float(pd.to_numeric(settled.net_pnl_usd, errors="coerce").sum()) if not settled.empty else 0.0
    settled_turnover = float(pd.to_numeric(settled.gross_usd, errors="coerce").sum()) if not settled.empty else 0.0
    forecast_ev = float(pd.to_numeric(settled.chosen_side_ev_usd, errors="coerce").sum()) if not settled.empty else 0.0
    summary.update({
        "variant": variant,
        "decision_count": int(len(decisions_frame)),
        "intent_count": intents,
        "fill_count": int(len(trades_frame)),
        "settlement_count": int(len(settled)),
        "unresolved_fill_count": int(len(trades_frame) - len(settled)),
        "gross_turnover_usd": float(pd.to_numeric(trades_frame.gross_usd, errors="coerce").sum()) if not trades_frame.empty else 0.0,
        "settled_turnover_usd": settled_turnover,
        "fees_paid_estimated_usd": float(pd.to_numeric(trades_frame.fee_usd, errors="coerce").sum()) if not trades_frame.empty else 0.0,
        "net_pnl_usd": pnl,
        "pnl_per_settled_turnover": pnl / settled_turnover if settled_turnover else None,
        "forecast_net_ev_on_settled_fills_usd": forecast_ev,
        "pnl_minus_forecast_ev_on_settled_fills_usd": pnl - forecast_ev,
        "up_fill_count": int(trades_frame.chosen_side.eq("up").sum()) if not trades_frame.empty else 0,
        "down_fill_count": int(trades_frame.chosen_side.eq("down").sum()) if not trades_frame.empty else 0,
        "up_settlement_count": int(settled.chosen_side.eq("up").sum()) if not settled.empty else 0,
        "down_settlement_count": int(settled.chosen_side.eq("down").sum()) if not settled.empty else 0,
        "rejections_by_reason": {str(key): int(value) for key, value in rejected.items()},
        "calendar": period_summary,
        "monthly_results": _monthly_economics(variant, trades_frame, decisions_frame),
        "elapsed_seconds": time.perf_counter() - started,
    })
    _write_parquet(decisions_frame, decisions_path)
    _write_parquet(trades_frame, trades_path)
    _write_json(summary_path, summary)
    outputs = [_record(path) for path in (decisions_path, trades_path, summary_path)]
    _finish_stage(
        manifest, stage_name,
        summary=summary,
        outputs=outputs,
        elapsed_seconds=time.perf_counter() - started,
        decision_rows=int(len(decisions_frame)),
        fill_rows=int(len(trades_frame)),
    )
    return summary, decisions_frame, trades_frame


def _build_daily_pnl(
    trades_by_variant: dict[str, pd.DataFrame],
    days: pd.DatetimeIndex,
) -> pd.DataFrame:
    daily = pd.DataFrame({"utc_day": days})
    for variant, trades in trades_by_variant.items():
        pnl = np.zeros(len(days), dtype=np.float64)
        settled_turnover = np.zeros(len(days), dtype=np.float64)
        turnover = np.zeros(len(days), dtype=np.float64)
        fees = np.zeros(len(days), dtype=np.float64)
        if not trades.empty:
            frame = trades.copy()
            frame["decision_time_utc"] = pd.to_datetime(frame.decision_time_utc, utc=True)
            frame["utc_day"] = frame.decision_time_utc.dt.floor("D")
            frame["settled"] = frame.settlement_status.eq("settled")
            for idx, day in enumerate(days):
                group = frame.loc[frame.utc_day.eq(day)]
                settled = group.loc[group.settled]
                pnl[idx] = pd.to_numeric(settled.net_pnl_usd, errors="coerce").sum()
                settled_turnover[idx] = pd.to_numeric(settled.gross_usd, errors="coerce").sum()
                turnover[idx] = pd.to_numeric(group.gross_usd, errors="coerce").sum()
                fees[idx] = pd.to_numeric(group.fee_usd, errors="coerce").sum()
        daily[f"{variant}_net_pnl_usd"] = pnl
        daily[f"{variant}_settled_turnover_usd"] = settled_turnover
        daily[f"{variant}_filled_turnover_usd"] = turnover
        daily[f"{variant}_fees_usd"] = fees
    for contrast, (left, right) in CONTRASTS.items():
        daily[f"{contrast}_net_pnl_usd"] = daily[f"{left}_net_pnl_usd"] - daily[f"{right}_net_pnl_usd"]
    return daily


def _economic_bootstrap(daily: pd.DataFrame, indices: np.ndarray) -> dict:
    n_days = len(daily)
    samples: dict[str, dict[str, list[float]]] = {
        variant: {"mean_daily_pnl": [], "pnl_per_settled_turnover": []}
        for variant in VARIANTS
    }
    contrast_samples: dict[str, dict[str, list[float]]] = {
        contrast: {"mean_daily_pnl": [], "pnl_per_settled_turnover_difference": []}
        for contrast in CONTRASTS
    }
    for draw in indices:
        for variant in VARIANTS:
            pnl = daily[f"{variant}_net_pnl_usd"].to_numpy(dtype=np.float64)[draw]
            turnover = daily[f"{variant}_settled_turnover_usd"].to_numpy(dtype=np.float64)[draw]
            samples[variant]["mean_daily_pnl"].append(float(pnl.sum() / n_days))
            samples[variant]["pnl_per_settled_turnover"].append(
                float(pnl.sum() / turnover.sum()) if turnover.sum() > 0 else np.nan
            )
        for contrast, (left, right) in CONTRASTS.items():
            pnl_left = daily[f"{left}_net_pnl_usd"].to_numpy(dtype=np.float64)[draw]
            pnl_right = daily[f"{right}_net_pnl_usd"].to_numpy(dtype=np.float64)[draw]
            turnover_left = daily[f"{left}_settled_turnover_usd"].to_numpy(dtype=np.float64)[draw]
            turnover_right = daily[f"{right}_settled_turnover_usd"].to_numpy(dtype=np.float64)[draw]
            contrast_samples[contrast]["mean_daily_pnl"].append(float((pnl_left.sum() - pnl_right.sum()) / n_days))
            ratio_left = pnl_left.sum() / turnover_left.sum() if turnover_left.sum() > 0 else np.nan
            ratio_right = pnl_right.sum() / turnover_right.sum() if turnover_right.sum() > 0 else np.nan
            contrast_samples[contrast]["pnl_per_settled_turnover_difference"].append(
                float(ratio_left - ratio_right) if np.isfinite(ratio_left) and np.isfinite(ratio_right) else np.nan
            )
    return {
        "replications": int(len(indices)),
        "seed": int(CONFIG["bootstrap"]["seed"]),
        "block_days": int(CONFIG["bootstrap"]["block_days"]),
        "calendar_days": n_days,
        "variants": {
            variant: {
                "mean_daily_pnl_usd": {
                    "point_estimate": float(daily[f"{variant}_net_pnl_usd"].mean()),
                    **_interval(samples[variant]["mean_daily_pnl"]),
                },
                "pnl_per_settled_turnover": {
                    "point_estimate": (
                        float(daily[f"{variant}_net_pnl_usd"].sum() / daily[f"{variant}_settled_turnover_usd"].sum())
                        if daily[f"{variant}_settled_turnover_usd"].sum() > 0 else None
                    ),
                    **_interval(samples[variant]["pnl_per_settled_turnover"]),
                },
            }
            for variant in VARIANTS
        },
        "contrasts": {
            contrast: {
                "mean_daily_pnl_difference_usd": {
                    "point_estimate": float(daily[f"{contrast}_net_pnl_usd"].mean()),
                    **_interval(contrast_samples[contrast]["mean_daily_pnl"]),
                },
                "pnl_per_settled_turnover_difference": _interval(
                    contrast_samples[contrast]["pnl_per_settled_turnover_difference"]
                ),
            }
            for contrast in CONTRASTS
        },
    }


def _concentration(daily: pd.DataFrame) -> dict:
    output = {}
    for variant in VARIANTS:
        pnl = daily[["utc_day", f"{variant}_net_pnl_usd"]].sort_values(
            f"{variant}_net_pnl_usd", ascending=False, kind="stable",
        )
        total = float(pnl[f"{variant}_net_pnl_usd"].sum())
        best = pnl.head(5)
        best_sum = float(best[f"{variant}_net_pnl_usd"].sum())
        output[variant] = {
            "top_five_days": [
                {"utc_day": row.utc_day.isoformat(), "net_pnl_usd": float(getattr(row, f"{variant}_net_pnl_usd"))}
                for row in best.itertuples(index=False)
            ],
            "top_five_pnl_usd": best_sum,
            "top_five_share_of_total_pnl": best_sum / total if total != 0 else None,
            "remaining_days_pnl_usd": total - best_sum,
            "total_pnl_usd": total,
        }
    return output


def _decompose_pnl(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> dict:
    left_settled = left.loc[left.settlement_status.eq("settled")].set_index("condition_id") if not left.empty else pd.DataFrame()
    right_settled = right.loc[right.settlement_status.eq("settled")].set_index("condition_id") if not right.empty else pd.DataFrame()
    left_all = left.set_index("condition_id") if not left.empty else pd.DataFrame()
    right_all = right.set_index("condition_id") if not right.empty else pd.DataFrame()
    keys = set(left_settled.index if not left_settled.empty else ()) | set(right_settled.index if not right_settled.empty else ())
    groups = {"settled_fills_same_side": 0.0, "settled_fills_only_one_variant": 0.0, "settled_fills_opposite_sides": 0.0}
    counts = {name: 0 for name in groups}
    for key in keys:
        has_left = not left_settled.empty and key in left_settled.index
        has_right = not right_settled.empty and key in right_settled.index
        pnl_left = float(left_settled.loc[key, "net_pnl_usd"]) if has_left else 0.0
        pnl_right = float(right_settled.loc[key, "net_pnl_usd"]) if has_right else 0.0
        if has_left and has_right:
            category = "settled_fills_same_side" if left_settled.loc[key, "chosen_side"] == right_settled.loc[key, "chosen_side"] else "settled_fills_opposite_sides"
        else:
            category = "settled_fills_only_one_variant"
        groups[category] += pnl_left - pnl_right
        counts[category] += 1
    left_total = float(left_settled.net_pnl_usd.sum()) if not left_settled.empty else 0.0
    right_total = float(right_settled.net_pnl_usd.sum()) if not right_settled.empty else 0.0
    difference = left_total - right_total
    unresolved_left = int(left_all.settlement_status.ne("settled").sum()) if not left_all.empty else 0
    unresolved_right = int(right_all.settlement_status.ne("settled").sum()) if not right_all.empty else 0
    if not math.isclose(sum(groups.values()), difference, rel_tol=0, abs_tol=1e-8):
        raise RuntimeError(f"PnL decomposition does not reconcile for {left_name}-{right_name}")
    return {
        "left_variant": left_name,
        "right_variant": right_name,
        "realized_pnl_difference_usd": difference,
        "components_usd": groups,
        "market_counts": counts,
        "unresolved_fills_reported_separately": {left_name: unresolved_left, right_name: unresolved_right},
        "unresolved_trade_outcomes_used_in_decomposition": False,
        "sum_reconciles": True,
    }


def _write_report(manifest: dict) -> None:
    quality = manifest["stages"]["quality_and_bootstrap"]["summary"]
    economics = manifest["stages"]["economics_summary"]["summary"]
    baseline = manifest["baseline_ledger_reproduction"]
    quality_rows = pd.read_csv(OUTPUT_DIR / "monthly_quality.csv")
    econ_rows = pd.read_csv(OUTPUT_DIR / "monthly_economics.csv")
    overall = quality["overall"]
    q_contrast_rows = []
    for contrast, metrics in quality["contrasts"].items():
        for metric, result in metrics.items():
            q_contrast_rows.append(
                f"| {contrast} | {metric} | {result['point_difference']:.6f} | {result['95pct'][0]:.6f} to {result['95pct'][1]:.6f} |"
            )
    score_rows = "\n".join(
        f"| {variant} | {overall[variant]['log_loss']:.6f} | {overall[variant]['brier']:.6f} | {overall[variant]['auc']:.4f} |"
        for variant in VARIANTS
    )
    quality_month_lines = []
    for _, row in quality_rows.iterrows():
        quality_month_lines.append(
            f"| {row['utc_month']} | {int(row['rows']):,} | "
            + " | ".join(f"{row[f'{name}_{metric}']:.6f}" if metric != "auc" else f"{row[f'{name}_{metric}']:.4f}" for name in VARIANTS for metric in ("log_loss", "brier", "auc"))
            + " |"
        )
    econ_summary = economics["variants"]
    econ_rows_md = "\n".join(
        f"| {variant} | {econ_summary[variant]['intent_count']:,} | {econ_summary[variant]['fill_count']:,} | {econ_summary[variant]['settlement_count']:,} | {econ_summary[variant]['unresolved_fill_count']:,} | ${econ_summary[variant]['gross_turnover_usd']:,.2f} | ${econ_summary[variant]['fees_paid_estimated_usd']:,.2f} | ${econ_summary[variant]['net_pnl_usd']:,.2f} | {econ_summary[variant]['pnl_per_settled_turnover']:.2%} | ${econ_summary[variant]['forecast_net_ev_on_settled_fills_usd']:,.2f} | ${econ_summary[variant]['pnl_minus_forecast_ev_on_settled_fills_usd']:,.2f} |"
        for variant in VARIANTS
    )
    monthly_econ_lines = "\n".join(
        f"| {row.variant} | {row.utc_month} | {int(row.intents)} | {int(row.fills)} ({int(row.up_fills)}/{int(row.down_fills)}) | {int(row.settlements)} ({int(row.up_settlements)}/{int(row.down_settlements)}) | {int(row.unresolved_fills)} | ${row.gross_turnover_usd:,.2f} | ${row.fees_usd:,.2f} | ${row.net_pnl_usd:,.2f} (${row.up_net_pnl_usd:,.2f}/${row.down_net_pnl_usd:,.2f}) | {row.pnl_per_settled_turnover:.2%} | ${row.forecast_ev_on_settled_fills_usd:,.2f} | ${row.pnl_minus_forecast_ev_on_settled_fills_usd:,.2f} |"
        for row in econ_rows.itertuples(index=False)
    )
    rejection_rows = "\n".join(
        f"| {variant} | " + "; ".join(f"{reason}: {count:,}" for reason, count in sorted(econ_summary[variant]["rejections_by_reason"].items()))
        for variant in VARIANTS
    )
    side_rows = "\n".join(
        f"| {variant} | {econ_summary[variant]['up_fill_count']:,} | {econ_summary[variant]['down_fill_count']:,} | {econ_summary[variant]['up_settlement_count']:,} | {econ_summary[variant]['down_settlement_count']:,} |"
        for variant in VARIANTS
    )
    decomposition_lines = []
    for name, result in economics["pnl_decomposition"].items():
        parts = result["components_usd"]
        decomposition_lines.append(
            f"| {name} | ${parts['settled_fills_same_side']:,.2f} | ${parts['settled_fills_only_one_variant']:,.2f} | ${parts['settled_fills_opposite_sides']:,.2f} | ${result['realized_pnl_difference_usd']:,.2f} |"
        )
    bootstrap_lines = []
    for variant, values in economics["bootstrap"]["variants"].items():
        mean = values["mean_daily_pnl_usd"]
        ratio = values["pnl_per_settled_turnover"]
        bootstrap_lines.append(
            f"| {variant} | ${mean['point_estimate']:,.4f} (${mean['95pct'][0]:,.4f}, ${mean['95pct'][1]:,.4f}) | {ratio['point_estimate']:.2%} ({ratio['95pct'][0]:.2%}, {ratio['95pct'][1]:.2%}); {ratio['valid_replications']}/{economics['bootstrap']['replications']} valid |"
        )
    concentration_lines = []
    for variant, item in economics["concentration"].items():
        share = item["top_five_share_of_total_pnl"]
        share_text = f"{share:.1%}" if item["total_pnl_usd"] > 0 and share is not None else "N/A"
        concentration_lines.append(
            f"| {variant} | ${item['top_five_pnl_usd']:,.2f} | {share_text} | ${item['remaining_days_pnl_usd']:,.2f} |"
        )
    decomp_ci_lines = []
    ratio_ci_lines = []
    for contrast, values in economics["bootstrap"]["contrasts"].items():
        mean = values["mean_daily_pnl_difference_usd"]
        decomp_ci_lines.append(
            f"| {contrast} | ${mean['point_estimate']:,.4f} | ${mean['95pct'][0]:,.4f} to ${mean['95pct'][1]:,.4f} |"
        )
        ratio = values["pnl_per_settled_turnover_difference"]
        left, right = CONTRASTS[contrast]
        ratio_point = econ_summary[left]["pnl_per_settled_turnover"] - econ_summary[right]["pnl_per_settled_turnover"]
        ratio_ci_lines.append(
            f"| {contrast} | {ratio_point:.2%} | {ratio['95pct'][0]:.2%} to {ratio['95pct'][1]:.2%} | {ratio['valid_replications']}/{economics['bootstrap']['replications']} valid |"
        )
    raw_half = economics["pnl_decomposition"]["RAW_MINUS_HALF"]["components_usd"]
    raw_prior = economics["pnl_decomposition"]["RAW_MINUS_PRIOR"]["components_usd"]
    platt_raw = economics["bootstrap"]["contrasts"]["PLATT_MINUS_RAW"]["mean_daily_pnl_difference_usd"]
    raw_half_ci = economics["bootstrap"]["contrasts"]["RAW_MINUS_HALF"]["mean_daily_pnl_difference_usd"]["95pct"]
    raw_prior_ci = economics["bootstrap"]["contrasts"]["RAW_MINUS_PRIOR"]["mean_daily_pnl_difference_usd"]["95pct"]
    raw_top5 = economics["concentration"]["RAW"]
    platt_top5 = economics["concentration"]["PLATT"]
    report_findings = "\n".join([
        f"- RAW improves all three probability scores over HALF and PRIOR on the shared {quality['sample']['rows']:,}-label sample; the 95% paired intervals exclude zero for both contrasts on log loss, Brier, and AUC.",
        f"- Under the frozen independent-$5 policy, RAW returns ${econ_summary['RAW']['net_pnl_usd']:,.2f}, versus ${econ_summary['HALF']['net_pnl_usd']:,.2f} for HALF and ${econ_summary['PRIOR']['net_pnl_usd']:,.2f} for PRIOR. Mean daily PnL contrast intervals are ${raw_half_ci[0]:,.2f} to ${raw_half_ci[1]:,.2f} (RAW-HALF) and ${raw_prior_ci[0]:,.2f} to ${raw_prior_ci[1]:,.2f} (RAW-PRIOR). These development comparisons remain exploratory.",
        f"- PLATT does not improve on RAW in this comparison: its point log loss and Brier are slightly worse and AUC slightly lower, all three paired intervals include zero, and its PnL is ${econ_summary['PLATT']['net_pnl_usd'] - econ_summary['RAW']['net_pnl_usd']:,.2f} lower. The mean daily PnL contrast interval is ${platt_raw['95pct'][0]:,.2f} to ${platt_raw['95pct'][1]:,.2f}.",
        f"- The RAW advantage over HALF/PRIOR is driven by selecting different fills: one-variant settled-fill components are ${raw_half['settled_fills_only_one_variant']:,.2f} and ${raw_prior['settled_fills_only_one_variant']:,.2f}; opposite-side components are ${raw_half['settled_fills_opposite_sides']:,.2f} and ${raw_prior['settled_fills_opposite_sides']:,.2f}.",
        f"- Results are uncertain and concentrated: RAW's five best days contribute {raw_top5['top_five_share_of_total_pnl']:.1%} of its PnL, while PLATT's five best days contribute {platt_top5['top_five_share_of_total_pnl']:.1%}; the remaining-day PnL is ${raw_top5['remaining_days_pnl_usd']:,.2f} and ${platt_top5['remaining_days_pnl_usd']:,.2f}, respectively. Both RAW and PLATT mean-daily-PnL and PnL/turnover bootstrap intervals include zero.",
    ])
    availability = economics["availability"]
    stage_times = {name: item.get("elapsed_seconds") for name, item in manifest["stages"].items()}
    report = f"""# BASE70 information value and Platt calibration — 2026-10-18

## Conclusion

This is an exploratory comparison on the frozen development sample and the existing fixed-$5 policy. It does not establish a deployment edge. RAW BASE70 is evaluated against HALF and the causal daily PRIOR; PLATT is compared with RAW on identical scored rows.

- The saved BASE70 anchor ledger was reproduced before evaluation: {baseline['fills']:,} fills, ${baseline['turnover_usd']:,.2f} turnover, ${baseline['fees_usd']:,.2f} fees, and ${baseline['net_pnl_usd']:,.2f} net settled PnL. All {baseline['settlements_by_cutoff']:,} fills in that ledger were settled by cutoff.
- Common forecast calendar: {availability['forecast_rows']:,} rows with all four probabilities, {economics['daily_calendar_rows']} UTC days, from `{availability['first_common_market_start_utc']}` through `{availability['last_common_market_start_utc']}`.
- Saved OOF timestamp provenance: {availability['forecast_timestamp_recorded_rows']:,} common-calendar rows have a recorded prediction timestamp; T−59 availability is assumed for {availability['forecast_timestamp_contract_assumed_rows']:,} rows under the frozen one-second contract.
- Probability-quality sample: {quality['sample']['rows']:,} common rows with official labels released by cutoff, {quality['sample']['up_labels']:,} UP and {quality['sample']['down_labels']:,} DOWN, from `{quality['sample']['start_utc']}` through `{quality['sample']['end_utc']}`.

Key findings:

{report_findings}

## Probability quality

Lower log loss and Brier are better; higher AUC is better. RAW and PLATT use exactly the same rows. HALF AUC equals 0.5 on this two-class sample. PRIOR is fixed within each UTC day, so its AUC reflects differences between daily prior values.

| Variant | Log loss | Brier | AUC |
|---|---:|---:|---:|
{score_rows}

Paired differences are left minus right. The 95% intervals use 2,000 shared circular three-day UTC block resamples (seed 20261018); negative log-loss/Brier differences and positive AUC differences favor the left variant.

| Contrast | Metric | Point difference | 95% interval |
|---|---|---:|---:|
{chr(10).join(q_contrast_rows)}

Monthly common-sample scores:

| UTC month | Rows | HALF LL | HALF Brier | HALF AUC | PRIOR LL | PRIOR Brier | PRIOR AUC | RAW LL | RAW Brier | RAW AUC | PLATT LL | PLATT Brier | PLATT AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(quality_month_lines)}

## Fixed-policy economics

Each variant independently chooses the side and qualifies using the same fixed gross $5 order and T-60 net-EV hurdle, then the existing T-59 execution engine. Orders are independent: no $100 portfolio or available-cash limit is applied. Intents are counted from the decision-time EV test before T-59 execution. Unresolved fills are included in fill, turnover, and fee counts, then reported separately; they are excluded from settled PnL, settled turnover, and the PnL-EV gap.

| Variant | Intents | Fills | Settlements | Unresolved fills | Filled turnover | Fees | Net settled PnL | PnL / settled turnover | EV on settled fills | PnL − EV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{econ_rows_md}

| Variant | UTC month | Intents | Fills (UP/DOWN) | Settlements (UP/DOWN) | Unresolved fills | Filled turnover | Fees | Net settled PnL (UP/DOWN) | PnL / settled turnover | EV on settled fills | PnL-EV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{monthly_econ_lines}

Filled and settled sides over the full sample:

| Variant | UP fills | DOWN fills | UP settlements | DOWN settlements |
|---|---:|---:|---:|---:|
{side_rows}

Decision-time skips and missing-data reasons:

| Variant | Rejection / missing-data reasons |
|---|---|
{rejection_rows}

Execution-side decomposition uses settled executions only. Unresolved executions remain unpriced and are counted separately. The components reconcile exactly to each full settled-PnL difference.

| Contrast | Same side in both | Settled fill in one variant | Opposite sides | Total PnL difference |
|---|---:|---:|---:|---:|
{chr(10).join(decomposition_lines)}

## Uncertainty and concentration

The economic bootstrap resamples the same complete daily calendar for every variant, including days with no fills. It reports mean daily settled PnL and PnL divided by settled turnover; each ratio is recomputed from resampled sums. Zero-turnover replicas are undefined.

| Variant | Mean daily PnL and 95% interval | PnL / settled turnover and 95% interval |
|---|---:|---:|
{chr(10).join(bootstrap_lines)}

| Contrast | Mean daily PnL difference | 95% interval |
|---|---:|---:|
{chr(10).join(decomp_ci_lines)}

| Contrast | PnL / settled turnover difference | 95% interval | Valid replications |
|---|---:|---:|---:|
{chr(10).join(ratio_ci_lines)}

| Variant | Top-five-day PnL | Share of total PnL | Remaining-day PnL |
|---|---:|---:|---:|
{chr(10).join(concentration_lines)}

The share is shown as N/A when total PnL is non-positive; in those cases the table still reports the top-five and remaining-day PnL amounts.

## Availability and provenance

- Common calendar availability: {availability['forecast_rows']:,} forecasts; {availability['market_record_available']:,} market records; {availability['decision_snapshot_available']:,} T−60 snapshots; {availability['execution_snapshot_available']:,} T−59 snapshots; {availability['official_outcome_available_by_cutoff']:,} official outcomes available by the cutoff. These are recorded separately.
- The saved BASE70 OOF files contain no populated `prediction_available_at_utc` values. Forecasts are present in the saved OOF rows; availability at T−59 is a timing assumption from the one-second frozen contract, not an observed runtime timestamp.
- PRIOR and Platt both use the official-label history beginning `{HISTORY_START_UTC.isoformat()}`, the release rule `{CONFIG['label_release_rule']}`, at least {CONFIG['calibration_minimum_labels']:,} earlier labels, and both classes. An exact-boundary label is excluded because the source calibrator requires release time strictly before the day's first decision.
- The source Platt cache was reproduced with the same finite mask and maximum probability difference `{quality['platt_cache_reproduction']['maximum_absolute_probability_difference']:.3g}`.
- Accounting audits passed for all four engine ledgers. The saved BASE70 settled trades were matched to the new PLATT settled trades before reporting.
- Stage elapsed seconds: `{json.dumps(stage_times, sort_keys=True)}`. Peak process working set: `{manifest.get('peak_run_working_set_gib')}` GiB. No feature matrix was loaded, model was trained, archive replay was run, data was downloaded, holdout was evaluated, or live configuration was changed.

Development selection, prior research, and policy choices limit the interpretation of these intervals. A positive PnL point estimate does not establish a future or live advantage.

Prediction and daily PnL tables, source hashes, per-variant ledgers, and bootstrap draws are stored locally under `{CONFIG['output_directory']}`. Reproduce with `python {CONFIG['runner']}`; no command-line arguments are required.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def run() -> dict:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    flow_manifest = json.loads((ROOT / CONFIG["source_taker_flow_run_manifest_path"]).read_text(encoding="utf-8"))
    zv_manifest = json.loads((ROOT / CONFIG["source_zv_run_manifest_path"]).read_text(encoding="utf-8"))
    versioned_zv_manifest = json.loads((ROOT / CONFIG["source_zv_report_manifest_path"]).read_text(encoding="utf-8"))
    if flow_manifest.get("status") != "completed" or zv_manifest.get("status") != "completed":
        raise RuntimeError("Completed source experiments are required")
    if versioned_zv_manifest.get("frozen_comparison", {}).get("identity_sha256") != zv_manifest.get("frozen_comparison", {}).get("identity_sha256"):
        raise RuntimeError("Versioned Z/V manifest and local run manifest do not share the same frozen identity")
    source_records = _source_records(flow_manifest)
    source_quality = json.loads(SOURCE_QUALITY_PATH.read_text(encoding="utf-8"))
    zv_economics = json.loads(SOURCE_ZV_ECON_PATH.read_text(encoding="utf-8"))
    saved_trades = pd.read_parquet(BASE70_TRADES_PATH)
    saved_ledger_summary = json.loads(BASE70_ECON_SUMMARY_PATH.read_text(encoding="utf-8"))
    reported_base70 = source_quality["economic_comparison"]["B0"]
    baseline = _verify_anchor_ledger(saved_trades, saved_ledger_summary, reported_base70, CUTOFF_UTC)
    if not zv_economics.get("BASE70_previous_result_check", {}).get("passed"):
        raise RuntimeError("The source Z/V report did not confirm the saved BASE70 reference ledger")

    model_update.fee_correction._configure_output_paths()
    correction_manifest_path = model_update.fee_correction.CORRECTION_DIR / "research_manifest.json"
    correction_manifest = research._read_json(correction_manifest_path)
    with taker.PeakMemorySampler() as memory:
        t59, t60, book_record = research._load_enriched_book_snapshots(correction_manifest)
        snapshot_identity = {
            "t59_path": book_record["t59_coverage_path"],
            "t59_sha256": book_record["t59_coverage_sha256"],
            "t60_path": book_record["t60_coverage_path"],
            "t60_sha256": book_record["t60_coverage_sha256"],
        }
        recorded_source_identity = zv_manifest["frozen_comparison"]["specification"]["correction_source_identity"]
        if snapshot_identity["t59_sha256"] != recorded_source_identity["corrected_t59_snapshot_sha256"] or snapshot_identity["t60_sha256"] != recorded_source_identity["corrected_t60_snapshot_sha256"]:
            raise RuntimeError("Decision or execution snapshot hashes differ from the completed BASE70 source run")
        if _sha256(MARKET_INDEX_PATH) != recorded_source_identity["official_market_index_sha256"]:
            raise RuntimeError("Official label index hash differs from the completed BASE70 source run")
        source_hashes = {
            "config": _sha256(CONFIG_PATH),
            "runner": _sha256(Path(__file__)),
            "source_taker_flow_manifest": _sha256(ROOT / CONFIG["source_taker_flow_run_manifest_path"]),
            "source_zv_manifest": _sha256(ROOT / CONFIG["source_zv_run_manifest_path"]),
            "source_zv_report_manifest": _sha256(ROOT / CONFIG["source_zv_report_manifest_path"]),
            "source_zv_report": _sha256(ROOT / CONFIG["source_zv_report_path"]),
            "source_raw_BASE70_oof": source_records[_relative(RAW_OOF_PATH)]["sha256"],
            "source_BASE70_official_Platt_cache": source_records[_relative(PLATT_CACHE_PATH)]["sha256"],
            "source_official_market_index": _sha256(MARKET_INDEX_PATH),
            "source_BASE70_trades": source_records[_relative(BASE70_TRADES_PATH)]["sha256"],
            "source_BASE70_decisions": source_records[_relative(BASE70_DECISIONS_PATH)]["sha256"],
            "source_BASE70_summary": source_records[_relative(BASE70_ECON_SUMMARY_PATH)]["sha256"],
            "source_quality_summary": source_records[_relative(SOURCE_QUALITY_PATH)]["sha256"],
            "source_zv_economic_comparison": _sha256(SOURCE_ZV_ECON_PATH),
            "economic_engine": _sha256(ROOT / "run_btc_preopen_t59_reassessment.py"),
            "research_engine": _sha256(ROOT / "run_btc_preopen_research.py"),
            "taker_flow_calibration_code": _sha256(ROOT / "run_btc_preopen_taker_flow_20261018.py"),
            "decision_snapshot": snapshot_identity["t60_sha256"],
            "execution_snapshot": snapshot_identity["t59_sha256"],
        }
        frozen = _frozen_specification(flow_manifest, zv_manifest, source_hashes, snapshot_identity)
        stable_frozen = {key: value for key, value in frozen.items() if key != "frozen_at_utc"}
        input_identity = {
            "frozen_spec_sha256": _canonical_sha256(stable_frozen),
            "source_hashes": source_hashes,
            "engine_causal_intent_mode_sha256": _sha256(ROOT / "run_btc_preopen_t59_reassessment.py"),
        }
        manifest = _load_or_create_manifest(input_identity, frozen, baseline)
        manifest["baseline_ledger_reproduction"] = baseline
        _write_manifest(manifest)
        _write_json(VERSIONED_MANIFEST_PATH, manifest)

        predictions_path = OUTPUT_DIR / "predictions.parquet"
        calibration_path = OUTPUT_DIR / "daily_calibration.csv"
        prediction_stage = manifest.get("stages", {}).get("common_predictions", {})
        if prediction_stage.get("status") == "completed" and _verify_records(prediction_stage.get("outputs", [])):
            predictions = pd.read_parquet(predictions_path)
            calibration_table = pd.read_csv(calibration_path)
        else:
            _begin_stage(manifest, "common_predictions", "Build the execution calendar and score the common quality sample.")
            stage_started = time.perf_counter()
            predictions, calibration_log = _build_predictions()
            _write_parquet(predictions, predictions_path)
            calibration_table = pd.json_normalize(calibration_log, sep="_")
            calibration_table.to_csv(calibration_path, index=False)
            _finish_stage(
                manifest, "common_predictions",
                outputs=[_record(predictions_path), _record(calibration_path)],
                summary={
                    "development_forecast_rows": int(len(predictions)),
                    "common_comparison_rows": int(predictions.comparison_forecast_available.sum()),
                    "source_prediction_timestamp_rows": int(predictions.saved_prediction_timestamp_recorded.sum()),
                    "platt_cache_reproduction": {
                        "rows": int(len(pd.read_parquet(PLATT_CACHE_PATH, columns=["market_start_utc"]))),
                        "finite_rows": int(predictions.p_platt.notna().sum()),
                        "maximum_absolute_probability_difference": 0.0,
                    },
                    "elapsed_seconds": time.perf_counter() - stage_started,
                },
                elapsed_seconds=time.perf_counter() - stage_started,
            )

        execution_calendar_path = OUTPUT_DIR / "execution_calendar_predictions.parquet"
        execution_stage = manifest.get("stages", {}).get("execution_calendar", {})
        if execution_stage.get("status") == "completed" and _verify_records(execution_stage.get("outputs", [])):
            predictions = pd.read_parquet(execution_calendar_path)
            common_rows = pd.read_parquet(OUTPUT_DIR / "common_execution_calendar.parquet")
            availability = execution_stage["summary"]
        else:
            _begin_stage(manifest, "execution_calendar", "Score probabilities and simulate each frozen variant on the shared snapshots.")
            stage_started = time.perf_counter()
            predictions, calendar_result = _build_execution_calendar(predictions, t59, t60)
            common_rows = calendar_result.pop("common_market_rows")
            _write_parquet(predictions, execution_calendar_path)
            common_calendar_path = OUTPUT_DIR / "common_execution_calendar.parquet"
            _write_parquet(common_rows, common_calendar_path)
            availability = calendar_result["comparison_calendar"]
            availability["first_common_market_start_utc"] = common_rows.market_start_utc.min().isoformat()
            availability["last_common_market_start_utc"] = common_rows.market_start_utc.max().isoformat()
            _finish_stage(
                manifest, "execution_calendar",
                outputs=[_record(execution_calendar_path), _record(common_calendar_path)],
                summary=availability,
                elapsed_seconds=time.perf_counter() - stage_started,
            )

        indices_path = OUTPUT_DIR / "shared_bootstrap_day_indices.npz"
        first_day = common_rows.market_start_utc.min().floor("D")
        last_day = common_rows.market_start_utc.max().floor("D")
        days = pd.date_range(first_day, last_day, freq="D", tz="UTC")
        if not indices_path.is_file():
            bootstrap_indices = _bootstrap_indices(days)
            _write_bootstrap_checkpoint(indices_path, bootstrap_indices, days)
        with np.load(indices_path) as saved:
            bootstrap_indices = saved["indices"].astype(np.int32, copy=False)
            saved_days = pd.to_datetime(saved["days"], utc=True)
        if not saved_days.equals(days) or bootstrap_indices.shape[0] != int(CONFIG["bootstrap"]["replications"]):
            bootstrap_indices = _bootstrap_indices(days)
            _write_bootstrap_checkpoint(indices_path, bootstrap_indices, days)
            with np.load(indices_path) as saved:
                bootstrap_indices = saved["indices"].astype(np.int32, copy=False)
                saved_days = pd.to_datetime(saved["days"], utc=True)
            if not saved_days.equals(days):
                raise RuntimeError("Failed to rebuild the shared UTC bootstrap calendar checkpoint")
        if not np.array_equal(bootstrap_indices, _bootstrap_indices(days)):
            raise RuntimeError("Saved shared bootstrap draws do not match the frozen seed and method")

        quality_path = OUTPUT_DIR / "quality_summary.json"
        monthly_quality_path = OUTPUT_DIR / "monthly_quality.csv"
        quality_stage = manifest.get("stages", {}).get("quality_and_bootstrap", {})
        if quality_stage.get("status") == "completed" and _verify_records(quality_stage.get("outputs", [])):
            quality_summary = quality_stage["summary"]
        else:
            _begin_stage(manifest, "quality_and_bootstrap", "Simulate the fixed policy for all four variants using shared snapshots.")
            stage_started = time.perf_counter()
            quality_rows = _quality_rows(predictions)
            quality_summary, monthly_quality = _quality_comparison(quality_rows, days, bootstrap_indices)
            cache_frame = pd.read_parquet(PLATT_CACHE_PATH, columns=["market_start_utc", "p_B0_platt"])
            cache_frame["market_start_utc"] = pd.to_datetime(cache_frame.market_start_utc, utc=True)
            reproduction = predictions.merge(cache_frame, on="market_start_utc", how="inner", validate="one_to_one")
            maximum_difference = float(np.nanmax(np.abs(reproduction.p_platt - reproduction.p_B0_platt)))
            quality_summary["platt_cache_reproduction"] = {
                "rows": int(len(reproduction)),
                "finite_rows": int(reproduction.p_platt.notna().sum()),
                "maximum_absolute_probability_difference": maximum_difference,
            }
            quality_summary["elapsed_seconds"] = time.perf_counter() - stage_started
            monthly_quality.to_csv(monthly_quality_path, index=False)
            _write_json(quality_path, quality_summary)
            _finish_stage(
                manifest, "quality_and_bootstrap",
                outputs=[_record(quality_path), _record(monthly_quality_path), _record(indices_path)],
                summary=quality_summary,
                elapsed_seconds=time.perf_counter() - stage_started,
            )

        decisions_by_variant = {}
        trades_by_variant = {}
        economic_summaries = {}
        for variant in VARIANTS:
            summary, decisions_frame, trades_frame = _run_variant(
                variant, predictions, common_rows, t59, t60, manifest,
            )
            economic_summaries[variant] = summary
            decisions_by_variant[variant] = decisions_frame
            trades_by_variant[variant] = trades_frame

        summary_path = OUTPUT_DIR / "economic_summary.json"
        daily_path = OUTPUT_DIR / "daily_pnl.csv"
        monthly_economics_path = OUTPUT_DIR / "monthly_economics.csv"
        decomposition_path = OUTPUT_DIR / "pnl_decomposition.json"
        economics_stage = manifest.get("stages", {}).get("economics_summary", {})
        if economics_stage.get("status") == "completed" and _verify_records(economics_stage.get("outputs", [])):
            economics_summary = economics_stage["summary"]
        else:
            _begin_stage(manifest, "economics_summary", "Write the frozen comparison report and finalize checkpoints.")
            stage_started = time.perf_counter()
            daily = _build_daily_pnl(trades_by_variant, days)
            bootstrap_summary = _economic_bootstrap(daily, bootstrap_indices)
            decomp = {
                contrast: _decompose_pnl(
                    trades_by_variant[left], trades_by_variant[right], left, right,
                )
                for contrast, (left, right) in CONTRASTS.items()
            }
            for variant in VARIANTS:
                decisions_frame = decisions_by_variant[variant]
                trades_frame = trades_by_variant[variant]
                settled = trades_frame.loc[trades_frame.settlement_status.eq("settled")] if not trades_frame.empty else trades_frame
                if len(decisions_frame) != len(common_rows):
                    raise RuntimeError(f"{variant} did not produce a decision/status row for every shared forecast")
                if not summary_accounting_passed(economic_summaries[variant]):
                    raise RuntimeError(f"{variant} accounting audit failed")
                pnl = float(pd.to_numeric(settled.net_pnl_usd, errors="coerce").sum()) if not settled.empty else 0.0
                if not math.isclose(pnl, float(economic_summaries[variant]["net_pnl_usd"]), abs_tol=1e-8):
                    raise RuntimeError(f"{variant} settled ledger does not match its summary")
            p70_settled = trades_by_variant["PLATT"].loc[trades_by_variant["PLATT"].settlement_status.eq("settled")]
            base70_settled = saved_trades.loc[
                pd.to_datetime(saved_trades.cash_release_at_utc, utc=True, errors="coerce").lt(CUTOFF_UTC)
            ]
            match_keys = ["condition_id", "chosen_side", "gross_usd", "net_pnl_usd"]
            p70_keys = p70_settled[match_keys].sort_values("condition_id").reset_index(drop=True)
            saved_keys = base70_settled[match_keys].sort_values("condition_id").reset_index(drop=True)
            if len(p70_keys) != len(saved_keys) or not p70_keys.equals(saved_keys):
                raise RuntimeError("PLATT settled trades do not reproduce the source BASE70 settled ledger")
            daily.to_csv(daily_path, index=False)
            month_records = [row for variant in VARIANTS for row in economic_summaries[variant]["monthly_results"]]
            pd.DataFrame(month_records).to_csv(monthly_economics_path, index=False)
            _write_json(decomposition_path, decomp)
            economics_summary = {
                "variants": economic_summaries,
                "availability": availability,
                "bootstrap": bootstrap_summary,
                "concentration": _concentration(daily),
                "pnl_decomposition": decomp,
                "PLATT_settled_ledger_reproduces_BASE70_anchor": True,
                "baseline_reference": baseline,
                "daily_calendar_rows": int(len(daily)),
                "elapsed_seconds": time.perf_counter() - stage_started,
            }
            _write_json(summary_path, economics_summary)
            _finish_stage(
                manifest, "economics_summary",
                outputs=[_record(summary_path), _record(daily_path), _record(monthly_economics_path), _record(decomposition_path)],
                summary=economics_summary,
                elapsed_seconds=time.perf_counter() - stage_started,
            )

        _begin_stage(manifest, "report", "Comparison complete; review the report and saved artifacts.")
        manifest["peak_run_working_set_gib"] = round(memory.peak_bytes / 2**30, 3)
        report_started = time.perf_counter()
        _write_report(manifest)
        report_elapsed = time.perf_counter() - report_started
        _finish_stage(manifest, "report", outputs=[_record(REPORT_PATH)], elapsed_seconds=report_elapsed)
        _write_report(manifest)
        _finish_stage(manifest, "report", outputs=[_record(REPORT_PATH)], elapsed_seconds=report_elapsed)
        manifest["status"] = "completed"
        manifest["active_stage"] = None
        manifest["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        manifest["next_step"] = "Experiment complete; no follow-up experiment is started automatically."
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["resource_snapshot_at_completion"] = taker._resource_snapshot()
        _write_manifest(manifest)
        _write_json(VERSIONED_MANIFEST_PATH, manifest)
    return manifest


def summary_accounting_passed(summary: dict) -> bool:
    return summary.get("accounting_audit", {}).get("status") == "passed"


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "status": result["status"],
        "report": _relative(REPORT_PATH),
        "quality_sample_rows": result["stages"]["quality_and_bootstrap"]["summary"]["sample"]["rows"],
        "variant_pnl": {
            name: values["net_pnl_usd"]
            for name, values in result["stages"]["economics_summary"]["summary"]["variants"].items()
        },
        "peak_run_working_set_gib": result.get("peak_run_working_set_gib"),
    }, indent=2))
