"""Run the frozen, resumable BTC pre-open research sequence without CLI flags."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_preopen_experiment as experiment
import run_btc_preopen_t59_reassessment as replay
import run_btc_preopen_walk_forward as walk_forward


ROOT = Path(__file__).resolve().parent
CONFIG = walk_forward.CONFIG
OUTPUT_DIR = ROOT / CONFIG["output_dir"]
RESEARCH_MANIFEST_PATH = OUTPUT_DIR / "research_manifest.json"
ARCHIVE_ROOT = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008"
ARCHIVE_DIR = ARCHIVE_ROOT / "pmxt_extended"
ARCHIVE_INDEX_PATH = ARCHIVE_DIR / "market_index.parquet"
ARCHIVE_MANIFEST_PATH = ARCHIVE_DIR / "archive_manifest.json"
BASE_MARKET_CALENDAR_PATH = (
    ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261007/coverage_calendar_before_pmxt.parquet"
)
EVENT_PARTS_DIR = ARCHIVE_DIR / "event_parts"
BASE_EVENT_PARTS_DIR = replay.SOURCE_OUT_DIR / "pmxt/event_parts"
BOOK_REPLAY_DIR = OUTPUT_DIR / "book_replay_t60_t59"
POLICY_OUTPUT_DIR = OUTPUT_DIR / "development_policy_selection"
POLICY_PATH = OUTPUT_DIR / "selected_development_policy.json"
HOLDOUT_OUTPUT_DIR = OUTPUT_DIR / "runs/evaluated_holdout"
HOLDOUT_MANIFEST_PATH = OUTPUT_DIR / "evaluated_holdout_manifest.json"
REPORT_PATH = ROOT / "reports/btc_preopen/btc_preopen_research_20261011.md"
SLIPPAGE_USD_PER_SHARE = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05)
DEVELOPMENT_START_UTC = pd.Timestamp("2026-04-15T17:05:00Z")
DEVELOPMENT_CUTOFF_UTC = pd.Timestamp(CONFIG["development_cutoff_exclusive_utc"])
DATA_CUTOFF_UTC = pd.Timestamp(CONFIG["data_cutoff_exclusive_utc"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_stage(manifest: dict, name: str, payload: dict) -> None:
    manifest.setdefault("stages", {})[name] = {
        "status": "completed",
        "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        **payload,
    }
    manifest["status"] = "running"
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(RESEARCH_MANIFEST_PATH, manifest)


def _json_normalize(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _load_research_manifest() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    identity = {
        "experiment_id": CONFIG["experiment_id"],
        "config_sha256": _sha256(walk_forward.CONFIG_PATH),
        "raw_data_sha256": _sha256(ROOT / CONFIG["raw_data_path"]),
        "official_markets_sha256": _sha256(ROOT / CONFIG["official_markets_path"]),
        "archive_manifest_sha256": _sha256(ARCHIVE_MANIFEST_PATH),
        "archive_market_index_sha256": _sha256(ARCHIVE_INDEX_PATH),
        "replay_engine_sha256": _sha256(ROOT / "run_btc_preopen_t59_reassessment.py"),
        "walk_forward_engine_sha256": _sha256(ROOT / "run_btc_preopen_walk_forward.py"),
        "research_code_sha256": _sha256(Path(__file__)),
        "data_cutoff_exclusive_utc": CONFIG["data_cutoff_exclusive_utc"],
    }
    manifest = _read_json(RESEARCH_MANIFEST_PATH) if RESEARCH_MANIFEST_PATH.is_file() else {
        "schema_version": 1,
        "stages": {},
    }
    if manifest.get("input_identity") not in (None, identity):
        raise RuntimeError("Existing BTC research manifest belongs to different frozen inputs or code")
    manifest["input_identity"] = identity
    manifest["status"] = "running"
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(RESEARCH_MANIFEST_PATH, manifest)
    return manifest


def _prepare_t60_t59_books(manifest: dict) -> dict:
    cached = manifest.get("stages", {}).get("causal_t60_t59_book_replay", {})
    if cached.get("status") == "completed":
        cached_t59 = ROOT / cached.get("t59_coverage_path", "__missing__")
        cached_t60 = ROOT / cached.get("t60_coverage_path", "__missing__")
        if (
            cached.get("archive_market_index_sha256") == _sha256(ARCHIVE_INDEX_PATH)
            and cached.get("archive_manifest_sha256") == _sha256(ARCHIVE_MANIFEST_PATH)
            and cached_t59.is_file() and cached_t60.is_file()
            and _sha256(cached_t59) == cached.get("t59_coverage_sha256")
            and _sha256(cached_t60) == cached.get("t60_coverage_sha256")
        ):
            print("[research] book replay status=skipped (verified checkpoint)", flush=True)
            return cached
    for path in (ARCHIVE_INDEX_PATH, ARCHIVE_MANIFEST_PATH, EVENT_PARTS_DIR, BASE_EVENT_PARTS_DIR):
        if not path.exists():
            raise FileNotFoundError(path)
    archive_index = pd.read_parquet(ARCHIVE_INDEX_PATH)
    archive_manifest = _read_json(ARCHIVE_MANIFEST_PATH)
    if len(archive_index) != 50_184 or archive_index.condition_id.duplicated().any():
        raise RuntimeError("The frozen PMXT market index no longer matches the verified 50,184-market source")
    index = archive_index.copy()
    index["p_candidate_raw"] = index["p_model_raw"]
    index["p_candidate_platt"] = index["p_model_platt"]
    index["condition_id"] = index.condition_id.astype(str).str.lower()
    archive_sha256 = _sha256(ARCHIVE_MANIFEST_PATH)
    replay.OUT_DIR = BOOK_REPLAY_DIR
    started = time.perf_counter()
    snapshots, replay_manifest = replay._replay_t59(
        index,
        EVENT_PARTS_DIR,
        archive_source_manifest_sha256=archive_sha256,
    )
    decision_snapshots = pd.read_parquet(BOOK_REPLAY_DIR / "t60_decision_books.parquet")
    snapshots = replay._add_archive_state_coverage_to_snapshots(
        snapshots, archive_index, archive_manifest, BASE_EVENT_PARTS_DIR,
    )
    decision_snapshots = replay._add_archive_state_coverage_to_snapshots(
        decision_snapshots, archive_index, archive_manifest, BASE_EVENT_PARTS_DIR,
    )
    t59_coverage_path = BOOK_REPLAY_DIR / "t59_ask_ladders_with_source_coverage.parquet"
    t60_coverage_path = BOOK_REPLAY_DIR / "t60_decision_books_with_source_coverage.parquet"
    snapshots.to_parquet(t59_coverage_path.with_suffix(".parquet.tmp"), index=False, compression="zstd")
    t59_coverage_path.with_suffix(".parquet.tmp").replace(t59_coverage_path)
    decision_snapshots.to_parquet(t60_coverage_path.with_suffix(".parquet.tmp"), index=False, compression="zstd")
    t60_coverage_path.with_suffix(".parquet.tmp").replace(t60_coverage_path)
    if len(snapshots) != len(archive_index) or len(decision_snapshots) != len(archive_index):
        raise RuntimeError("The causal book replay did not capture both T-60 and T-59 for every archived market")
    identity = {
        "market_count": len(archive_index),
        "archive_market_index_sha256": _sha256(ARCHIVE_INDEX_PATH),
        "archive_manifest_sha256": archive_sha256,
        "archive_event_rows": int(archive_manifest.get("pmxt_v2", {}).get("locally_processed_hour_count", 0)),
        "replay_manifest": replay_manifest,
        "t59_coverage_path": t59_coverage_path.resolve().relative_to(ROOT).as_posix(),
        "t59_coverage_sha256": _sha256(t59_coverage_path),
        "t60_coverage_path": t60_coverage_path.resolve().relative_to(ROOT).as_posix(),
        "t60_coverage_sha256": _sha256(t60_coverage_path),
        "elapsed_seconds_this_call": time.perf_counter() - started,
        "model_predictions_in_replay_identity": False,
    }
    _record_stage(manifest, "causal_t60_t59_book_replay", identity)
    return identity


def _load_model_predictions(*, include_holdout: bool) -> tuple[pd.DataFrame, dict]:
    walk_manifest_path = walk_forward.RUN_MANIFEST_PATH
    walk_manifest = _read_json(walk_manifest_path)
    if walk_manifest.get("status") != "development_walk_forward_completed":
        raise RuntimeError("Complete all chronological development pipeline folds before policy selection")
    frames = []
    sources = {}
    for fold_id in (walk_forward.SEED_FOLD_ID, *walk_forward.DEVELOPMENT_FOLD_IDS):
        record = walk_manifest["folds"][fold_id]
        path = ROOT / record["oof_path"]
        if record.get("status") != "completed" or _sha256(path) != record.get("oof_sha256"):
            raise RuntimeError(f"Walk-forward prediction artifact is missing or changed: {fold_id}")
        frame = pd.read_parquet(path)
        frame["fold_id"] = fold_id
        frames.append(frame)
        sources[fold_id] = {"path": record["oof_path"], "sha256": record["oof_sha256"]}
    if include_holdout:
        holdout = _read_json(HOLDOUT_MANIFEST_PATH)
        path = ROOT / holdout["predictions_path"]
        if holdout.get("status") != "completed" or _sha256(path) != holdout.get("predictions_sha256"):
            raise RuntimeError("The frozen holdout prediction artifact is missing or changed")
        frame = pd.read_parquet(path)
        frame["fold_id"] = "evaluated_holdout"
        frame["prediction_provenance"] = "frozen_evaluated_bundle_external_prediction"
        frames.append(frame)
        sources["evaluated_holdout"] = {
            "path": holdout["predictions_path"],
            "sha256": holdout["predictions_sha256"],
        }
    predictions = pd.concat(frames, ignore_index=True, sort=False)
    predictions["market_start_utc"] = pd.to_datetime(predictions["market_start_utc"], utc=True, errors="raise")
    predictions["nominal_decision_at"] = pd.to_datetime(predictions["nominal_decision_at"], utc=True, errors="raise")
    predictions["prediction_available_at_utc"] = pd.to_datetime(
        predictions["prediction_available_at_utc"], utc=True, errors="raise",
    )
    if predictions.market_start_utc.duplicated().any():
        duplicate_starts = predictions.loc[predictions.market_start_utc.duplicated(), "market_start_utc"].head(5).tolist()
        raise RuntimeError(f"Walk-forward/external predictions overlap: {duplicate_starts}")
    if not (predictions["prediction_available_at_utc"] == predictions["market_start_utc"] - pd.Timedelta(seconds=59)).all():
        raise RuntimeError("A model prediction was not available at T-59 under the one-second decision budget")
    return predictions.sort_values("market_start_utc", kind="stable").reset_index(drop=True), {
        "walk_forward_manifest_sha256": _sha256(walk_manifest_path),
        "prediction_sources": sources,
        "prediction_rows": int(len(predictions)),
        "prediction_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(
                predictions[["market_start_utc", "p_model_raw", "p_model_platt", "fold_id"]],
                index=False,
            ).to_numpy().tobytes()
        ).hexdigest(),
    }


def _build_market_calendar(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base = pd.read_parquet(BASE_MARKET_CALENDAR_PATH)
    remove_columns = {
        "Opened", "p_candidate_raw", "p_candidate_platt", "target_binance_proxy_up",
        "prediction_origin", "prediction_available_at_utc", "entry_time_utc",
        "market_confirmed", "outcome_available", "btc_input_available",
        "causal_prediction_available", "prediction_source", "native_up_token_confirmed",
        "native_down_token_confirmed",
    }
    base = base.drop(columns=[column for column in remove_columns if column in base], errors="ignore")
    archive_index = pd.read_parquet(ARCHIVE_INDEX_PATH)
    archive_columns = [
        "condition_id", "archive_source", "archive_entry_hour_available",
        "archive_freshness_window_available", "archive_window_hours_expected",
        "archive_window_hours_source_available", "archive_window_hours_processed",
        "archive_window_missing_source_hours", "archive_window_unprocessed_hours",
        "archive_history_complete",
    ]
    base["condition_id"] = base.condition_id.astype("string").str.lower()
    archive_index["condition_id"] = archive_index.condition_id.astype("string").str.lower()
    base = base.merge(
        archive_index.loc[:, archive_columns],
        on="condition_id", how="left", validate="many_to_one",
    )
    predictions = predictions.copy()
    predictions["market_start_utc"] = pd.to_datetime(predictions.market_start_utc, utc=True, errors="raise")
    attach = predictions.rename(columns={
        "p_model_raw": "p_candidate_raw",
        "p_model_platt": "p_candidate_platt",
        experiment.TARGET_COL: "target_binance_proxy_up",
    })
    if "fold_id" in attach:
        attach["prediction_origin"] = np.where(
            attach.fold_id.notna(),
            "full_pipeline_oof_" + attach.fold_id.fillna("").astype(str),
            "none",
        )
    else:
        attach["fold_id"] = pd.Series(pd.NA, index=attach.index, dtype="string")
        attach["prediction_origin"] = np.where(
            attach.p_candidate_platt.notna(), "evaluated_holdout_model", "none",
        )
    selected = [
        "market_start_utc", "Opened", "p_candidate_raw", "p_candidate_platt",
        "target_binance_proxy_up", "prediction_available_at_utc", "fold_id",
        "prediction_origin",
    ]
    base = base.merge(
        attach.loc[:, selected],
        on="market_start_utc", how="left", validate="one_to_one",
    )
    base["prediction_origin"] = base["prediction_origin"].fillna("none")
    base["prediction_source"] = np.where(
        base.fold_id.notna(),
        base.fold_id,
        np.where(base.p_candidate_platt.notna(), "evaluated_holdout_model", "none"),
    )
    base["causal_prediction_available"] = base.p_candidate_platt.notna()
    base["market_confirmed"] = base.condition_id.notna()
    base["outcome_available"] = base.target_polymarket_up.notna() & base.resolved_at_utc.notna()
    base["native_up_token_confirmed"] = base.up_token_id.notna()
    base["native_down_token_confirmed"] = base.down_token_id.notna()
    base["prediction_available_at_utc"] = pd.to_datetime(base.prediction_available_at_utc, utc=True, errors="coerce")
    return base, {
        "calendar_rows": int(len(base)),
        "confirmed_market_rows": int(base.market_confirmed.sum()),
        "confirmed_with_prediction_rows": int((base.market_confirmed & base.causal_prediction_available).sum()),
        "price_proxy_target_rows": int(pd.to_numeric(base.target_binance_proxy_up, errors="coerce").notna().sum()),
    }


def _asof_market_rows(calendar: pd.DataFrame, start: pd.Timestamp, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    rows = calendar.loc[
        pd.to_datetime(calendar.market_start_utc, utc=True).ge(start)
        & pd.to_datetime(calendar.market_start_utc, utc=True).lt(cutoff)
    ].copy()
    starts = pd.to_datetime(rows.market_start_utc, utc=True)
    resolved = pd.to_datetime(rows.resolved_at_utc, utc=True, errors="coerce")
    official_outcome_present = pd.to_numeric(rows.target_polymarket_up, errors="coerce").notna()
    release = resolved.where(resolved.ge(starts + pd.Timedelta(minutes=5)), starts + pd.Timedelta(minutes=5))
    release = release + pd.Timedelta(seconds=60)
    available = official_outcome_present & resolved.notna() & release.le(cutoff)
    rows["outcome_available_by_cutoff"] = available.to_numpy()
    rows["cash_release_at_cutoff_utc"] = release
    unavailable = ~available
    if unavailable.any():
        rows.loc[unavailable, "target_polymarket_up"] = pd.NA
        rows.loc[unavailable, "resolved_at_utc"] = pd.NaT
    return rows, {
        "market_start_first_utc": starts.min().isoformat() if len(rows) else None,
        "market_start_last_utc": starts.max().isoformat() if len(rows) else None,
        "market_rows_in_period": int(len(rows)),
        "official_outcomes_and_cash_releases_available_by_cutoff": int(available.sum()),
        "excluded_until_after_cutoff_or_missing_resolution_or_official_outcome": int(unavailable.sum()),
        "missing_official_outcome_rows": int((~official_outcome_present).sum()),
        "selection_or_evaluation_cutoff_utc": cutoff.isoformat(),
    }


def _snapshot_map(frame: pd.DataFrame) -> dict[str, dict]:
    return {
        str(row.condition_id).lower(): row._asdict()
        for row in frame.itertuples(index=False)
    }


def _simulate(
    sizing_policy: str,
    market_rows: pd.DataFrame,
    snapshots_by_id: dict[str, dict],
    market_by_id: dict[str, dict],
    decision_snapshots_by_id: dict[str, dict],
    *,
    minimum_net_ev_fraction: float = 0.0,
    output_policy: str,
    output_dir: Path | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    summary, trades, decisions = replay._portfolio_for_policy(
        sizing_policy,
        market_rows,
        snapshots_by_id,
        market_by_id,
        decision_snapshot_by_id=decision_snapshots_by_id,
        minimum_net_ev_fraction=minimum_net_ev_fraction,
        output_policy=output_policy,
    )
    equity_path = summary.pop("_equity_path")
    summary["monthly_diagnostics"] = replay._monthly_portfolio_diagnostics(trades, decisions, equity_path)
    summary["accounting_audit"] = replay._portfolio_accounting_audit(
        summary, trades, decisions, equity_path, snapshots_by_id, decision_snapshots_by_id,
    )
    summary["drawdown_contributors"] = replay._drawdown_contributors(trades, equity_path)
    summary["minimum_net_ev_fraction"] = float(minimum_net_ev_fraction)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if trades:
            pd.DataFrame(trades).to_parquet(output_dir / "trades.parquet", index=False, compression="zstd")
        if decisions:
            pd.DataFrame(decisions).to_parquet(output_dir / "decisions.parquet", index=False, compression="zstd")
        _write_json(output_dir / "summary.json", summary)
    return summary, trades, decisions


def _load_enriched_book_snapshots(manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    record = manifest.get("stages", {}).get("causal_t60_t59_book_replay", {})
    if record.get("status") != "completed":
        raise RuntimeError("The causal T-60/T-59 replay must complete before policy evaluation")
    t59_path = ROOT / record["t59_coverage_path"]
    t60_path = ROOT / record["t60_coverage_path"]
    if _sha256(t59_path) != record.get("t59_coverage_sha256") or _sha256(t60_path) != record.get("t60_coverage_sha256"):
        raise RuntimeError("A source-coverage-enriched book snapshot changed after replay")
    return pd.read_parquet(t59_path), pd.read_parquet(t60_path), record


def _threshold_name(value: float) -> str:
    return f"free_cash_5pct_ev_{float(value):.4%}".replace("%", "pct")


def _select_development_policy(manifest: dict) -> dict:
    cached_stage = manifest.get("stages", {}).get("development_policy_selection", {})
    cached_policy = _read_json(POLICY_PATH) if POLICY_PATH.is_file() else {}
    cached_candidate_path = POLICY_OUTPUT_DIR / "candidate_results.parquet"
    book_record = manifest.get("stages", {}).get("causal_t60_t59_book_replay", {})
    if (
        cached_stage.get("status") == "completed"
        and cached_policy.get("status") == "frozen_from_development_before_holdout_evaluation"
        and _sha256(POLICY_PATH) == cached_stage.get("policy_sha256")
        and cached_candidate_path.is_file()
        and _sha256(cached_candidate_path) == cached_stage.get("candidate_summaries_sha256")
        and book_record.get("status") == "completed"
        and _sha256(ROOT / book_record["t59_coverage_path"]) == book_record.get("t59_coverage_sha256")
        and _sha256(ROOT / book_record["t60_coverage_path"]) == book_record.get("t60_coverage_sha256")
    ):
        predictions, prediction_identity = _load_model_predictions(include_holdout=False)
        del predictions
        expected_identity = {
            "experiment_id": CONFIG["experiment_id"],
            "development_start_utc": DEVELOPMENT_START_UTC,
            "development_cutoff_exclusive_utc": DEVELOPMENT_CUTOFF_UTC,
            "policy_selection_config": CONFIG["policy_selection"],
            "prediction_identity": prediction_identity,
            "t59_coverage_sha256": book_record["t59_coverage_sha256"],
            "t60_coverage_sha256": book_record["t60_coverage_sha256"],
            "replay_dependency_sha256": book_record["replay_manifest"]["replay_dependency_sha256"],
            "research_code_sha256": _sha256(Path(__file__)),
            "replay_code_sha256": _sha256(ROOT / "run_btc_preopen_t59_reassessment.py"),
        }
        if _json_normalize(cached_policy.get("input_identity")) == _json_normalize(expected_identity):
            print("[research] development policy status=skipped (verified freeze)", flush=True)
            return cached_policy
    snapshots, decision_snapshots, book_record = _load_enriched_book_snapshots(manifest)
    predictions, prediction_identity = _load_model_predictions(include_holdout=False)
    market_calendar, calendar_summary = _build_market_calendar(predictions)
    calendar, _fixed5 = replay._market_statuses(market_calendar, snapshots, decision_snapshots)
    decision_map = _snapshot_map(decision_snapshots)
    snapshot_map = _snapshot_map(snapshots)
    market_map = _snapshot_map(calendar.loc[calendar.condition_id.notna()])
    market_rows, period_summary = _asof_market_rows(
        calendar, DEVELOPMENT_START_UTC, DEVELOPMENT_CUTOFF_UTC,
    )
    POLICY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    result_summaries = []

    fixed5, fixed5_trades, fixed5_decisions = _simulate(
        "fixed_5_usd", market_rows, snapshot_map, market_map, decision_map,
        output_policy="fixed_5_usd_control",
        output_dir=POLICY_OUTPUT_DIR / "fixed_5_usd_control",
    )
    results["fixed_5_usd_control"] = {
        "summary": fixed5, "trades": fixed5_trades, "decisions": fixed5_decisions,
    }
    result_summaries.append({"policy": "fixed_5_usd_control", **fixed5})

    free_cash, free_cash_trades, free_cash_decisions = _simulate(
        "free_cash_5pct", market_rows, snapshot_map, market_map, decision_map,
        output_policy="free_cash_5pct_control",
        output_dir=POLICY_OUTPUT_DIR / "free_cash_5pct_control",
    )
    results["free_cash_5pct_control"] = {
        "summary": free_cash, "trades": free_cash_trades, "decisions": free_cash_decisions,
    }
    result_summaries.append({"policy": "free_cash_5pct_control", **free_cash})

    candidates = []
    for threshold in CONFIG["policy_selection"]["minimum_net_ev_fraction_candidates"]:
        threshold = float(threshold)
        name = _threshold_name(threshold)
        if threshold == 0.0:
            summary, trades, decisions = free_cash, free_cash_trades, free_cash_decisions
        else:
            summary, trades, decisions = _simulate(
                "free_cash_5pct", market_rows, snapshot_map, market_map, decision_map,
                minimum_net_ev_fraction=threshold,
                output_policy=name,
                output_dir=POLICY_OUTPUT_DIR / name,
            )
        candidate = {
            "policy": name,
            "minimum_net_ev_fraction": threshold,
            "ending_cash_after_settlement_usd": float(summary["ending_cash_after_settlement_usd"]),
            "net_pnl_usd": float(summary["net_pnl_usd"]),
            "max_drawdown_cost_basis_equity": float(summary["max_drawdown_cost_basis_equity"]),
            "trade_count": int(summary["trade_count"]),
            "gross_turnover_usd": float(summary["gross_turnover_usd"]),
            "fees_paid_estimated_usd": float(summary["fees_paid_estimated_usd"]),
            "rejections_by_reason": summary["rejections_by_reason"],
        }
        candidates.append(candidate)
        results[name] = {"summary": summary, "trades": trades, "decisions": decisions}
        result_summaries.append({"policy": name, **summary})

    selected = max(
        candidates,
        key=lambda item: (
            item["ending_cash_after_settlement_usd"],
            -item["max_drawdown_cost_basis_equity"],
            -item["trade_count"],
        ),
    )
    freeze_identity = {
        "experiment_id": CONFIG["experiment_id"],
        "development_start_utc": DEVELOPMENT_START_UTC,
        "development_cutoff_exclusive_utc": DEVELOPMENT_CUTOFF_UTC,
        "policy_selection_config": CONFIG["policy_selection"],
        "prediction_identity": prediction_identity,
        "t59_coverage_sha256": book_record["t59_coverage_sha256"],
        "t60_coverage_sha256": book_record["t60_coverage_sha256"],
        "replay_dependency_sha256": book_record["replay_manifest"]["replay_dependency_sha256"],
        "research_code_sha256": _sha256(Path(__file__)),
        "replay_code_sha256": _sha256(ROOT / "run_btc_preopen_t59_reassessment.py"),
    }
    policy_payload = {
        "status": "frozen_from_development_before_holdout_evaluation",
        "input_identity": freeze_identity,
        "selected_policy": {
            "name": selected["policy"],
            "sizing_policy": "free_cash_5pct",
            "minimum_net_ev_fraction": selected["minimum_net_ev_fraction"],
            "definition": "5% of free cash at T-60, no dollar cap, hold to settlement; trade only when T-60 net EV divided by requested gross stake meets the frozen hurdle",
        },
        "selection_metric": CONFIG["policy_selection"]["selection_metric"],
        "development_period": period_summary,
        "development_calendar": calendar_summary,
        "required_controls": {
            "fixed_5_usd": fixed5,
            "free_cash_5pct": free_cash,
        },
        "candidate_results": candidates,
        "selected_candidate_result": selected,
        "development_execution_recovery": _execution_recovery_summary(
            results[selected["policy"]]["trades"],
        ),
        "holdout_inputs_used_for_selection": False,
    }
    if POLICY_PATH.is_file():
        old_policy = _read_json(POLICY_PATH)
        if old_policy.get("input_identity") != freeze_identity:
            raise RuntimeError("A frozen development policy already exists for different inputs")
        if old_policy.get("selected_policy") != policy_payload["selected_policy"]:
            raise RuntimeError("Existing frozen policy selection differs from recomputed development results")
        policy_payload = old_policy
    else:
        _write_json(POLICY_PATH, policy_payload)
    candidate_frame = pd.DataFrame(candidates)
    candidate_frame.to_parquet(POLICY_OUTPUT_DIR / "candidate_results.parquet", index=False, compression="zstd")
    result = {
        "policy_path": POLICY_PATH.resolve().relative_to(ROOT).as_posix(),
        "policy_sha256": _sha256(POLICY_PATH),
        "selected_policy": policy_payload["selected_policy"],
        "policy_sha256": _sha256(POLICY_PATH),
        "prediction_identity": prediction_identity,
        "candidate_count": len(candidates),
        "development_period": period_summary,
        "development_calendar": calendar_summary,
        "candidate_summaries_path": (POLICY_OUTPUT_DIR / "candidate_results.parquet").resolve().relative_to(ROOT).as_posix(),
        "candidate_summaries_sha256": _sha256(POLICY_OUTPUT_DIR / "candidate_results.parquet"),
    }
    _record_stage(manifest, "development_policy_selection", result)
    return policy_payload


def _execution_recovery_summary(trades: list[dict]) -> dict:
    age_one_second = 0
    age_legacy_30 = 0
    proxy_depth = 0
    full_depth = 0
    for trade in trades:
        side = str(trade.get("chosen_side", "up"))
        age = trade.get(f"{side}_decision_requested_ask_age_seconds")
        if age is not None and not pd.isna(age):
            age = float(age)
            age_one_second += int(age > 1.0)
            age_legacy_30 += int(age > replay.LEGACY_ASK_AGE_CONTROL_SECONDS)
        if trade.get("depth_known") is True:
            full_depth += 1
        else:
            proxy_depth += 1
    return {
        "trades_with_active_ask_older_than_1_second": age_one_second,
        "trades_with_active_ask_older_than_legacy_30_second_control": age_legacy_30,
        "known_full_book_depth_trades": full_depth,
        "unknown_depth_top_of_book_proxy_trades": proxy_depth,
    }


def _slipped_snapshot_map(snapshots_by_id: dict[str, dict], increment: float) -> dict[str, dict]:
    if increment <= 0.0:
        return snapshots_by_id
    adjusted = {}
    for condition_id, source in snapshots_by_id.items():
        item = dict(source)
        for side in ("up", "down"):
            key = f"{side}_ask_levels"
            levels = item.get(key)
            if levels is not None and len(levels):
                shifted = [
                    [float(price) + increment, float(size)]
                    for price, size in levels
                    if 0.0 < float(price) + increment < 1.0 and float(size) > 0.0
                ]
                shifted.sort(key=lambda level: level[0])
                item[key] = shifted
                item[f"{side}_best_ask"] = shifted[0][0] if shifted else 1.0 + increment
                item[f"{side}_best_ask_size_shares"] = shifted[0][1] if shifted else 0.0
            else:
                price = item.get(f"{side}_best_ask")
                if price is not None and not pd.isna(price):
                    item[f"{side}_best_ask"] = float(price) + increment
        adjusted[condition_id] = item
    return adjusted


def _break_even_slippage(points: list[dict]) -> dict:
    if not points:
        return {"status": "unavailable"}
    first = points[0]
    if float(first["net_pnl_usd"]) <= 0.0:
        return {"status": "no_positive_baseline_edge", "increment_usd_per_share": 0.0}
    for left, right in zip(points, points[1:]):
        left_pnl, right_pnl = float(left["net_pnl_usd"]), float(right["net_pnl_usd"])
        if left_pnl > 0.0 >= right_pnl:
            width = float(right["increment_usd_per_share"]) - float(left["increment_usd_per_share"])
            estimate = float(left["increment_usd_per_share"]) + width * left_pnl / (left_pnl - right_pnl)
            return {"status": "crossed_zero", "increment_usd_per_share": estimate}
    return {
        "status": "not_crossed_in_sensitivity_range",
        "tested_max_increment_usd_per_share": float(points[-1]["increment_usd_per_share"]),
    }


def _slippage_sensitivity(
    sizing_policy: str,
    market_rows: pd.DataFrame,
    snapshots_by_id: dict[str, dict],
    market_by_id: dict[str, dict],
    decision_snapshots_by_id: dict[str, dict],
    *,
    minimum_net_ev_fraction: float,
    output_policy: str,
) -> dict:
    points = []
    for increment in SLIPPAGE_USD_PER_SHARE:
        shifted = _slipped_snapshot_map(snapshots_by_id, increment)
        summary, _trades, _decisions = replay._portfolio_for_policy(
            sizing_policy,
            market_rows,
            shifted,
            market_by_id,
            decision_snapshot_by_id=decision_snapshots_by_id,
            minimum_net_ev_fraction=minimum_net_ev_fraction,
            output_policy=output_policy,
        )
        summary.pop("_equity_path", None)
        points.append({
            "increment_usd_per_share": float(increment),
            "net_pnl_usd": float(summary["net_pnl_usd"]),
            "ending_cash_after_settlement_usd": float(summary["ending_cash_after_settlement_usd"]),
            "trade_count": int(summary["trade_count"]),
            "gross_turnover_usd": float(summary["gross_turnover_usd"]),
            "fees_paid_estimated_usd": float(summary["fees_paid_estimated_usd"]),
            "rejections_by_reason": summary["rejections_by_reason"],
        })
    return {
        "assumption": "shift every executable T-59 ask level upward by a fixed price amount; preserve T-60 side, stake rule, and EV hurdle",
        "points": points,
        "break_even": _break_even_slippage(points),
    }


def _stage_result(run_manifest: dict, name: str) -> dict:
    stage = run_manifest.get("stages", {}).get(name, {})
    if stage.get("status") != "completed":
        raise RuntimeError(f"Required training stage is incomplete: {name}")
    return stage.get("result", {})


def _run_evaluated_holdout(manifest: dict, policy: dict) -> dict:
    walk_manifest = _read_json(walk_forward.RUN_MANIFEST_PATH)
    if walk_manifest.get("status") != "development_walk_forward_completed":
        raise RuntimeError("Development walk-forward must finish before the evaluated holdout fit")
    calibration_record = walk_manifest.get("calibration_before_evaluated_holdout", {})
    calibration_path = ROOT / calibration_record.get("path", "__missing__")
    if (
        not calibration_path.is_file()
        or _sha256(calibration_path) != calibration_record.get("sha256")
        or calibration_record.get("holdout_rows_included") is not False
    ):
        raise RuntimeError("The frozen development-only calibration pool is missing or invalid")
    if policy.get("status") != "frozen_from_development_before_holdout_evaluation":
        raise RuntimeError("A development policy must be frozen before holdout evaluation")
    if policy.get("holdout_inputs_used_for_selection") is not False:
        raise RuntimeError("The policy freeze does not certify holdout exclusion")
    fold = next(item for item in CONFIG["folds"] if item["fold_id"] == "evaluated_holdout")
    source_contract_path = ROOT / CONFIG["source_contract_path"]
    base_contract = _read_json(source_contract_path)
    contract = walk_forward._fold_contract(base_contract, fold)
    split = contract["chronological_split_frozen_before_tuning"]
    contract_path = walk_forward.CONTRACT_DIR / "evaluated_holdout.json"
    walk_forward._write_json(contract_path, contract)
    profile_path = ROOT / "configs/btc_preopen_training.json"
    base_profile = _read_json(profile_path)["profiles"]["nightly_v1"]
    profile = copy.deepcopy(base_profile)
    profile.update({
        "contract_path": contract_path.resolve().relative_to(ROOT).as_posix(),
        "raw_data_path": CONFIG["raw_data_path"],
        "official_markets_path": CONFIG["official_markets_path"],
        "prediction_compute_budget_seconds": 1,
        "output_dir": f"{CONFIG['output_dir']}/runs/evaluated_holdout",
        "oof_calibration_predictions_path": calibration_path.resolve().relative_to(ROOT).as_posix(),
        "frozen_policy_path": POLICY_PATH.resolve().relative_to(ROOT).as_posix(),
    })
    identity = {
        "contract_sha256": _sha256(contract_path),
        "raw_data_sha256": _sha256(ROOT / CONFIG["raw_data_path"]),
        "official_markets_sha256": _sha256(ROOT / CONFIG["official_markets_path"]),
        "calibration_oof_sha256": _sha256(calibration_path),
        "development_policy_sha256": _sha256(POLICY_PATH),
        "training_code_sha256": _sha256(ROOT / "run_btc_preopen_experiment.py"),
        "training_dependencies": {
            path: _sha256(ROOT / path) for path in walk_forward.TRAINING_CODE_PATHS
        },
        "walk_forward_code_sha256": _sha256(ROOT / "run_btc_preopen_walk_forward.py"),
        "profile": profile,
        "historically_exposed": True,
    }
    prior = _read_json(HOLDOUT_MANIFEST_PATH) if HOLDOUT_MANIFEST_PATH.is_file() else {}
    if prior.get("input_identity") not in (None, identity):
        raise RuntimeError("Existing evaluated holdout belongs to different frozen inputs/settings")
    if prior.get("status") == "completed":
        prediction_path = ROOT / prior.get("predictions_path", "__missing__")
        freeze_path = ROOT / prior.get("evaluated_bundle_freeze_path", "__missing__")
        bundle_path = ROOT / prior.get("evaluated_bundle_path", "__missing__")
        run_manifest_path = ROOT / prior.get("run_manifest_path", "__missing__")
        cached_bundle = _read_json(bundle_path) if bundle_path.is_file() else {}
        cached_freeze = _read_json(freeze_path) if freeze_path.is_file() else {}
        if (
            prediction_path.is_file() and freeze_path.is_file()
            and bundle_path.is_file() and run_manifest_path.is_file()
            and _sha256(prediction_path) == prior.get("predictions_sha256")
            and _sha256(freeze_path) == prior.get("evaluated_bundle_freeze_sha256")
            and _sha256(bundle_path) == prior.get("evaluated_bundle_sha256")
            and _sha256(run_manifest_path) == prior.get("run_manifest_sha256")
            and cached_bundle.get("evaluated_bundle_freeze_sha256") == prior.get("evaluated_bundle_freeze_sha256")
            and cached_freeze.get("policy_sha256") == _sha256(POLICY_PATH)
            and _read_json(run_manifest_path).get("status") == "completed"
        ):
            print("[research] evaluated holdout status=skipped (verified checkpoint)", flush=True)
            _record_stage(manifest, "frozen_evaluated_holdout_model", {
                "holdout_manifest_path": HOLDOUT_MANIFEST_PATH.resolve().relative_to(ROOT).as_posix(),
                "holdout_manifest_sha256": _sha256(HOLDOUT_MANIFEST_PATH),
                "evaluated_bundle_sha256": prior["evaluated_bundle_sha256"],
                "evaluated_bundle_freeze_sha256": prior["evaluated_bundle_freeze_sha256"],
                "prediction_rows": prior["prediction_rows"],
                "exclusive_holdout_cutoff_utc": CONFIG["data_cutoff_exclusive_utc"],
            })
            return prior
    prior.update({
        "schema_version": 1,
        "status": "running",
        "input_identity": identity,
        "contract_path": contract_path.resolve().relative_to(ROOT).as_posix(),
        "started_utc": prior.get("started_utc", pd.Timestamp.now(tz="UTC").isoformat()),
        "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "holdout_rows_used_for_training_or_selection": False,
    })
    _write_json(HOLDOUT_MANIFEST_PATH, prior)
    print("[research] evaluated_holdout status=running", flush=True)
    result = experiment.run_training_pipeline(profile_override=profile)
    run_root = Path(result["run_root"])
    inner_manifest_path = run_root / "run_manifest.json"
    inner_manifest = _read_json(inner_manifest_path)
    prediction_path, inner_manifest = walk_forward._completed_predictions(run_root)
    prediction_frame = pd.read_parquet(prediction_path)
    starts = pd.to_datetime(prediction_frame.market_start_utc, utc=True, errors="raise")
    expected_first = pd.Timestamp(fold["market_start_first_utc"])
    expected_last = pd.Timestamp(fold["market_start_last_inclusive_utc"])
    if starts.empty or starts.min() != expected_first or starts.max() != expected_last:
        raise RuntimeError("Frozen evaluated bundle did not predict the exact holdout interval")
    if starts.ge(DATA_CUTOFF_UTC).any():
        raise RuntimeError("Holdout predictions extend beyond the frozen exclusive data cutoff")
    if "prediction_available_at_utc" not in prediction_frame:
        raise RuntimeError("Holdout predictions lack their actual availability timestamps")
    available_at = pd.to_datetime(prediction_frame.prediction_available_at_utc, utc=True, errors="raise")
    if not available_at.eq(starts - pd.Timedelta(seconds=59)).all():
        raise RuntimeError("Holdout prediction rows violate the T-60/T-59 one-second budget")
    freeze_path = run_root / "evaluated_bundle_freeze.json"
    bundle_path = ROOT / inner_manifest["bundle_path"]
    bundle = _read_json(bundle_path)
    if (
        not freeze_path.is_file()
        or _sha256(freeze_path) != bundle.get("evaluated_bundle_freeze_sha256")
        or _read_json(freeze_path).get("policy_sha256") != _sha256(POLICY_PATH)
    ):
        raise RuntimeError("Evaluated bundle was not frozen against the development-only policy")
    prior.update({
        "status": "completed",
        "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "run_root": run_root.resolve().relative_to(ROOT).as_posix(),
        "run_manifest_path": inner_manifest_path.resolve().relative_to(ROOT).as_posix(),
        "run_manifest_sha256": _sha256(inner_manifest_path),
        "run_status": inner_manifest.get("status"),
        "predictions_path": prediction_path.resolve().relative_to(ROOT).as_posix(),
        "predictions_sha256": _sha256(prediction_path),
        "prediction_rows": int(len(prediction_frame)),
        "prediction_start_first_utc": starts.min().isoformat(),
        "prediction_start_last_utc": starts.max().isoformat(),
        "evaluated_bundle_path": bundle_path.resolve().relative_to(ROOT).as_posix(),
        "evaluated_bundle_sha256": _sha256(bundle_path),
        "evaluated_bundle_freeze_path": freeze_path.resolve().relative_to(ROOT).as_posix(),
        "evaluated_bundle_freeze_sha256": _sha256(freeze_path),
        "fitted_model_stage": _stage_result(inner_manifest, "final_model"),
        "calibration_stage": _stage_result(inner_manifest, "calibration"),
        "external_evaluation_stage": _stage_result(inner_manifest, "external_evaluation"),
        "holdout_rows_used_for_training_or_selection": False,
    })
    _write_json(HOLDOUT_MANIFEST_PATH, prior)
    _record_stage(manifest, "frozen_evaluated_holdout_model", {
        "holdout_manifest_path": HOLDOUT_MANIFEST_PATH.resolve().relative_to(ROOT).as_posix(),
        "holdout_manifest_sha256": _sha256(HOLDOUT_MANIFEST_PATH),
        "evaluated_bundle_sha256": prior["evaluated_bundle_sha256"],
        "evaluated_bundle_freeze_sha256": prior["evaluated_bundle_freeze_sha256"],
        "prediction_rows": prior["prediction_rows"],
        "exclusive_holdout_cutoff_utc": CONFIG["data_cutoff_exclusive_utc"],
    })
    return prior


def _evaluate_portfolio_period(
    manifest: dict,
    predictions: pd.DataFrame,
    *,
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
    policies: list[tuple[str, str, float]],
    output_root: Path,
    period_name: str,
    input_identity: dict,
) -> dict:
    snapshots, decision_snapshots, book_record = _load_enriched_book_snapshots(manifest)
    calendar, calendar_summary = _build_market_calendar(predictions)
    calendar, _ = replay._market_statuses(calendar, snapshots, decision_snapshots)
    decision_map = _snapshot_map(decision_snapshots)
    snapshot_map = _snapshot_map(snapshots)
    market_map = _snapshot_map(calendar.loc[calendar.condition_id.notna()])
    market_rows, period_summary = _asof_market_rows(calendar, start, cutoff)
    output_root.mkdir(parents=True, exist_ok=True)
    evaluated = {}
    for policy_name, sizing_policy, hurdle in policies:
        policy_dir = output_root / policy_name
        summary, trades, decisions = _simulate(
            sizing_policy, market_rows, snapshot_map, market_map, decision_map,
            minimum_net_ev_fraction=hurdle,
            output_policy=policy_name,
            output_dir=policy_dir,
        )
        sensitivity = _slippage_sensitivity(
            sizing_policy, market_rows, snapshot_map, market_map, decision_map,
            minimum_net_ev_fraction=hurdle,
            output_policy=policy_name,
        )
        _write_json(policy_dir / "slippage_sensitivity.json", sensitivity)
        decisions_frame = pd.DataFrame(decisions)
        trades_frame = pd.DataFrame(trades)
        if decisions_frame.empty:
            decision_metrics = {
                "markets_with_at_least_one_t60_priceable_side_at_requested_size": 0,
                "markets_with_t60_selected_side_priceable_at_t59": 0,
            }
        else:
            priceable_count = pd.to_numeric(
                decisions_frame.get(
                    "priceable_sides_at_requested_amount",
                    pd.Series(index=decisions_frame.index, dtype="float64"),
                ), errors="coerce",
            ).fillna(0).gt(0)
            execution_priceable = []
            for decision in decisions:
                side = decision.get("requested_amount_chosen_side") or decision.get("chosen_side")
                execution_priceable.append(
                    bool(decision.get(f"{side}_execution_priceable")) if side in ("up", "down") else False
                )
            decision_metrics = {
                "markets_with_at_least_one_t60_priceable_side_at_requested_size": int(priceable_count.sum()),
                "markets_with_t60_selected_side_priceable_at_t59": int(sum(execution_priceable)),
            }
        evaluated[policy_name] = {
            "summary": summary,
            "slippage_sensitivity": sensitivity,
            "trade_recovery": _execution_recovery_summary(trades),
            "decision_count": len(decisions),
            "decision_metrics": decision_metrics,
            "decision_skip_reasons": (
                decisions_frame.loc[decisions_frame.decision.eq("skip"), "skip_reason"]
                .value_counts(dropna=True).to_dict()
                if not decisions_frame.empty else {}
            ),
            "trades_by_depth_source": (
                trades_frame.groupby(["execution_source", "depth_known"], dropna=False)
                .size().rename("trade_count").reset_index().to_dict("records")
                if not trades_frame.empty else []
            ),
            "summary_path": (policy_dir / "summary.json").resolve().relative_to(ROOT).as_posix(),
            "summary_sha256": _sha256(policy_dir / "summary.json"),
            "trades_path": (policy_dir / "trades.parquet").resolve().relative_to(ROOT).as_posix() if trades else None,
            "trades_sha256": _sha256(policy_dir / "trades.parquet") if trades else None,
            "decisions_path": (policy_dir / "decisions.parquet").resolve().relative_to(ROOT).as_posix() if decisions else None,
            "decisions_sha256": _sha256(policy_dir / "decisions.parquet") if decisions else None,
            "slippage_path": (policy_dir / "slippage_sensitivity.json").resolve().relative_to(ROOT).as_posix(),
            "slippage_sha256": _sha256(policy_dir / "slippage_sensitivity.json"),
        }
    result = {
        "period_name": period_name,
        "input_identity": input_identity,
        "market_period": {
            "start_inclusive_utc": start,
            "cutoff_exclusive_utc": cutoff,
        },
        "calendar": calendar_summary,
        "available_market_rows": period_summary,
        "book_replay": {
            "t59_coverage_sha256": book_record["t59_coverage_sha256"],
            "t60_coverage_sha256": book_record["t60_coverage_sha256"],
        },
        "policies": evaluated,
    }
    _write_json(output_root / f"{period_name}_portfolio_results.json", result)
    return result


def _model_fit_report_data(holdout: dict) -> dict:
    run_root = ROOT / holdout["run_root"]
    run_manifest = _read_json(run_root / "run_manifest.json")
    bundle_path = ROOT / run_manifest["bundle_path"]
    bundle = _read_json(bundle_path)
    freeze = _read_json(ROOT / holdout["evaluated_bundle_freeze_path"])
    calibration = _read_json(ROOT / bundle["calibrator_path"])
    walk_manifest = _read_json(walk_forward.RUN_MANIFEST_PATH)
    calibration_pool = walk_manifest["calibration_before_evaluated_holdout"]
    return {
        "run_manifest": run_manifest,
        "bundle": bundle,
        "freeze": freeze,
        "calibration": calibration,
        "walk_forward": walk_manifest,
        "calibration_pool": calibration_pool,
    }


def _format_usd(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"${float(value):,.2f}"


def _write_final_report(manifest: dict, policy: dict, holdout: dict, holdout_result: dict) -> None:
    fit_data = _model_fit_report_data(holdout)
    run_manifest = fit_data["run_manifest"]
    stages = run_manifest["stages"]
    model_training = next(
        _read_json(ROOT / item["path"])
        for item in stages["final_model"]["outputs"]
        if str(item["path"]).endswith("training_manifest.json")
    )
    tune = _stage_result(run_manifest, "model_tuning")
    weights = _stage_result(run_manifest, "target_weights")
    feature_selection = _stage_result(run_manifest, "feature_selection")
    calibration = fit_data["calibration"]
    pool = fit_data["calibration_pool"]
    walk_manifest = fit_data["walk_forward"]
    walk_run_records = [*walk_manifest["folds"].values(), holdout]
    model_elapsed_seconds = sum(
        float(_read_json(ROOT / record["run_manifest_path"]).get("elapsed_seconds", 0.0))
        for record in walk_run_records
    )
    archive_manifest = _read_json(ARCHIVE_MANIFEST_PATH)
    full_venue_gap = archive_manifest.get("full_venue_gap_utc", {})
    policies = holdout_result["policies"]
    selected_name = policy["selected_policy"]["name"]
    selected = policies[selected_name]
    selected_summary = selected["summary"]
    decision_metrics = selected["decision_metrics"]
    development_controls = policy["required_controls"]
    development_candidate_rows = "\n".join(
        f"| `{item['policy']}` | {item['trade_count']} | {_format_usd(item['ending_cash_after_settlement_usd'])} | "
        f"{_format_usd(item['net_pnl_usd'])} | {_format_usd(item['gross_turnover_usd'])} | "
        f"{_format_usd(item['fees_paid_estimated_usd'])} | {item['max_drawdown_cost_basis_equity']:.2%} |"
        for item in policy["candidate_results"]
    )
    development_control_rows = "\n".join(
        f"| `{name}` | {result['trade_count']} | {_format_usd(result['ending_cash_after_settlement_usd'])} | "
        f"{_format_usd(result['net_pnl_usd'])} | {_format_usd(result['gross_turnover_usd'])} | "
        f"{_format_usd(result['fees_paid_estimated_usd'])} | {result['max_drawdown_cost_basis_equity']:.2%} |"
        for name, result in development_controls.items()
    )
    tables = []
    for name, result in policies.items():
        summary = result["summary"]
        tables.append(
            f"| `{name}` | {summary['trade_count']} | {_format_usd(summary['ending_cash_after_settlement_usd'])} | "
            f"{_format_usd(summary['net_pnl_usd'])} | {_format_usd(summary['gross_turnover_usd'])} | "
            f"{_format_usd(summary['fees_paid_estimated_usd'])} | {summary['max_drawdown_cost_basis_equity']:.2%} |"
        )
    monthly = selected_summary.get("monthly_diagnostics", [])
    monthly_rows = "\n".join(
        f"| {item['month_utc']} | {item['trade_count']} | {_format_usd(item['monthly_equity_pnl_usd'])} | "
        f"{_format_usd(item['closing_cost_basis_equity_usd'])} | {item['monthly_max_drawdown']:.2%} |"
        for item in monthly
    ) or "| n/a | 0 | $0.00 | $100.00 | 0.00% |"
    slippage_rows = "\n".join(
        f"| ${item['increment_usd_per_share']:.4f} | {item['trade_count']} | {_format_usd(item['net_pnl_usd'])} | "
        f"{_format_usd(item['ending_cash_after_settlement_usd'])} |"
        for item in selected["slippage_sensitivity"]["points"]
    )
    recovered = selected["trade_recovery"]
    dev_period = policy["development_period"]
    holdout_period = holdout_result["available_market_rows"]
    holdout_summary = selected_summary
    verdict = (
        "The selected historical policy did not produce positive net PnL after modeled fees."
        if float(holdout_summary["net_pnl_usd"]) <= 0.0
        else "The selected historical policy produced positive net PnL after modeled fees; this is not a live-liquidity forecast."
    )
    ledger_links = []
    if selected["decisions_path"]:
        ledger_links.append(f"[`decisions.parquet`](../../{selected['decisions_path']})")
    if selected["trades_path"]:
        ledger_links.append(f"[`trades.parquet`](../../{selected['trades_path']})")
    ledger_links = " and ".join(ledger_links) if ledger_links else "No ledger files were produced."
    report = f"""# BTC 5-minute pre-open research — {CONFIG['experiment_id']}

## Conclusion

{verdict} The holdout period was historically exposed in earlier repository experiments, and was excluded from the new model, calibration, and policy selection. No live model was activated and no live orders were submitted.

## Frozen boundaries and data

- Development policy selection: `{DEVELOPMENT_START_UTC.isoformat()}` through `{DEVELOPMENT_CUTOFF_UTC.isoformat()}` exclusive; starting account `$100`.
- Model fit cutoff: labels available no later than `{fit_data['bundle']['chronological_split']['fit_label_available_through_inclusive_utc']}`.
- Holdout market starts: `{holdout_result['market_period']['start_inclusive_utc']}` through `{holdout_result['market_period']['cutoff_exclusive_utc']}` exclusive; separate starting account `$100`.
- Price input SHA-256: `{manifest['input_identity']['raw_data_sha256']}`; official market index SHA-256: `{manifest['input_identity']['official_markets_sha256']}`.
- Official archive cutoff: `{CONFIG['data_cutoff_exclusive_utc']}`. The source had 50,184 confirmed markets, including 50,183 resolved outcomes; held-out prices and outcomes were not used to select settings.
- Source coverage: `{holdout_result['calendar']['confirmed_market_rows']:,}` confirmed calendar records; `{holdout_period['official_outcomes_and_cash_releases_available_by_cutoff']:,}` holdout outcomes and cash releases available by cutoff; `{holdout_period['excluded_until_after_cutoff_or_missing_resolution_or_official_outcome']:,}` not available by cutoff, including `{holdout_period['missing_official_outcome_rows']:,}` with no official outcome.

## Learned pipeline and walk-forward selection

- Indicator generators were fit from `{fit_data['bundle']['chronological_split']['feature_and_model_fit_start_utc']}` using labels available through `{fit_data['bundle']['chronological_split']['fit_label_available_through_inclusive_utc']}` on the full price history available in the frozen input; later target labels were purged. Reaction/volume profile, target weights, feature selection, model tuning, and final model fit used the chronological development prefix only.
- Final evaluated model: `{model_training['training_rows']:,}` labeled rows, `{len(fit_data['bundle']['feature_order'])}` features, best iteration `{model_training['best_iteration']}`; LightGBM settings `{json.dumps(tune.get('best_params', model_training.get('params', {})), sort_keys=True)}`.
- Decision weight: `{weights.get('decision_weight', model_training.get('decision_weight'))}`; selected feature count: `{feature_selection.get('feature_count', len(fit_data['bundle']['feature_order']))}`; selected feature order: `{json.dumps(fit_data['bundle']['feature_order'])}`.
- Reaction profile config: `{json.dumps(fit_data['freeze']['reaction_profile_config'], sort_keys=True)}`. Volume profile config: `{json.dumps(fit_data['freeze']['volume_profile_config'], sort_keys=True)}`. Indicator-fit artifact hashes: `{json.dumps(fit_data['freeze']['indicator_fit_result_sha256'], sort_keys=True)}`. The full frozen configuration and checksums are in [`evaluated_bundle_freeze.json`](../../{holdout['evaluated_bundle_freeze_path']}).
- Walk-forward folds: calibration seed `2026-01-01`–`2026-04-15T17:00Z`; development policy folds `2026-04-15T17:05Z`–`2026-04-30T23:55Z`, `2026-05-01`–`2026-05-31T23:55Z`, and `2026-06-01`–`2026-06-30T23:55Z`. Each fold reran the learned pipeline using an earlier prefix. Calibration pools used only prior-fold OOF raw predictions whose labels were available before each decision boundary.
- Holdout calibration pool: `{pool['rows']:,}` development OOF rows; latest nominal decision `{pool['latest_nominal_decision_at_utc']}`; latest label availability `{pool['latest_label_available_at_utc']}`; holdout rows included: `false`. Calibrator rows: `{calibration['training_rows']:,}`.
- Selected policy: `{json.dumps(policy['selected_policy'], sort_keys=True)}`. It was selected only on `{dev_period['official_outcomes_and_cash_releases_available_by_cutoff']:,}` development outcomes/cash releases available by the development cutoff; selection metric was final settled cash with drawdown and trade-count tie-breaks.
- Policy freeze SHA-256: `{_sha256(POLICY_PATH)}`; evaluated bundle freeze SHA-256: `{holdout['evaluated_bundle_freeze_sha256']}`.

Development policy candidates and required controls:

| Policy | Trades | Ending cash | Net PnL | Gross turnover | Estimated fees | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
{development_candidate_rows}
{development_control_rows}

## Market availability and quote/execution audit

- Holdout confirmed market starts with causal model predictions: `{holdout_result['calendar']['confirmed_with_prediction_rows']:,}`.
- Holdout rows considered by the portfolio: `{holdout_period['market_rows_in_period']:,}`; official outcome and cash release available by cutoff: `{holdout_period['official_outcomes_and_cash_releases_available_by_cutoff']:,}`.
- Selected-policy decisions: `{selected['decision_count']:,}`; at least one side was priceable at T-60 for the requested stake in `{decision_metrics['markets_with_at_least_one_t60_priceable_side_at_requested_size']:,}` rows; the T-60 selected side remained priceable at T-59 in `{decision_metrics['markets_with_t60_selected_side_priceable_at_t59']:,}` rows; actual fills: `{holdout_summary['trade_count']:,}`. Skip reasons: `{json.dumps(selected['decision_skip_reasons'], sort_keys=True)}`.
- Quote/depth corrections among selected-policy fills: `{recovered['trades_with_active_ask_older_than_1_second']:,}` used an active ask older than one second; `{recovered['trades_with_active_ask_older_than_legacy_30_second_control']:,}` were older than the legacy 30-second age control; `{recovered['unknown_depth_top_of_book_proxy_trades']:,}` used unknown-depth top-of-book proxy pricing; `{recovered['known_full_book_depth_trades']:,}` used known full-book depth.
- Depth-source counts: `{json.dumps(selected['trades_by_depth_source'], sort_keys=True)}`. The selected strategy holds to settlement; exit bid continuity was separately audited in the replay change. Historical recorded liquidity is not a live-liquidity forecast.
- Remaining holdout skip reasons and full decision/trade ledgers: {ledger_links}.

## Holdout portfolio comparison

All variants use a separate `$100` portfolio, no credit or deposits, the same accepted fee schedule (0.072 through `2026-05-06T00:00:00Z`, then 0.07, retaining the established V1/V2 split and price-dependent fee arithmetic), and hold positions to settlement. “5% free cash” means 5% of currently available cash, with no additional dollar cap.

| Policy | Trades | Ending cash | Net PnL | Gross turnover | Estimated fees | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(tables)}

Monthly selected-policy equity path:

| UTC month | Trades | Equity PnL | Closing cost-basis equity | Max drawdown |
|---|---:|---:|---:|---:|
{monthly_rows}

Accounting audit: `{json.dumps(holdout_summary['accounting_audit'], sort_keys=True)}`.

## Adverse execution-price sensitivity

Every executable T-59 ask level was shifted upward while preserving the T-60 chosen side, stake, and EV hurdle. This is a fixed price increment per share, not a fitted cost model.

| Added price/share | Trades | Net PnL | Ending cash |
|---:|---:|---:|---:|
{slippage_rows}

Estimated break-even adverse price increment: `{json.dumps(selected['slippage_sensitivity']['break_even'], sort_keys=True)}`.

## Runtime, cache, and limitations

- Full book replay: `{manifest['stages']['causal_t60_t59_book_replay']['replay_manifest']['event_rows_applied']:,}` source events across `{manifest['stages']['causal_t60_t59_book_replay']['replay_manifest']['partitions_scanned']:,}` partitions; replay elapsed `{manifest['stages']['causal_t60_t59_book_replay']['elapsed_seconds_this_call']:.1f}s`. The source archive records a `{full_venue_gap.get('hours', 'unknown')}`-hour full-venue gap (`{full_venue_gap.get('start_inclusive', 'unknown')}`–`{full_venue_gap.get('end_inclusive', 'unknown')}`). Detailed replay cache/reuse and market-specific gap counts are in [`research_manifest.json`](../../{RESEARCH_MANIFEST_PATH.resolve().relative_to(ROOT).as_posix()}) and the replay manifest.
- The five causal model pipeline runs accumulated `{model_elapsed_seconds / 3600:.2f}` hours of wall time; the holdout run took `{run_manifest.get('elapsed_seconds', 0.0):.1f}s`. Per-fold logs, resource use, and stage cache identities are in the walk-forward and [`run_manifest.json`](../../{holdout['run_manifest_path']}).
- The research machine's measured replay/training profile and planned work counters are recorded in [`BTC_RESEARCH_STATUS_20261010.md`](../../docs/BTC_RESEARCH_STATUS_20261010.md). Price target is a Binance index proxy, not the official settlement feed. Only archived markets and source coverage are scored.
- Exit execution is not part of this hold-to-settlement entry policy. Rebates remain outside primary policy selection.

## Reproduction and artifacts

- Resume research: `python run_btc_preopen_research.py` (no CLI arguments).
- Deployment refit preparation: `python run_btc_preopen_deployment_refit.py` is a separate procedure and is not run by this experiment.
- Evaluated bundle: [`model_bundle.json`](../../{run_manifest['bundle_path']}); policy freeze: [`selected_development_policy.json`](../../{POLICY_PATH.resolve().relative_to(ROOT).as_posix()}); holdout summary: [`holdout_portfolio_results.json`](../../{(HOLDOUT_OUTPUT_DIR / 'holdout_portfolio_results.json').resolve().relative_to(ROOT).as_posix()}).
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_btc_preopen_research() -> dict:
    manifest = _load_research_manifest()
    print("[research] stage=causal_t60_t59_book_replay", flush=True)
    _prepare_t60_t59_books(manifest)
    if manifest.get("stages", {}).get("development_walk_forward", {}).get("status") != "completed":
        print("[research] stage=development_walk_forward", flush=True)
        walk_manifest = walk_forward.run_development_walk_forward()
        _record_stage(manifest, "development_walk_forward", {
            "walk_forward_manifest_path": walk_forward.RUN_MANIFEST_PATH.resolve().relative_to(ROOT).as_posix(),
            "walk_forward_manifest_sha256": _sha256(walk_forward.RUN_MANIFEST_PATH),
            "folds_completed": list(walk_manifest["folds"]),
            "calibration_pool_sha256": walk_manifest["calibration_before_evaluated_holdout"]["sha256"],
        })
    else:
        walk_manifest = _read_json(walk_forward.RUN_MANIFEST_PATH)
        if walk_manifest.get("status") != "development_walk_forward_completed":
            raise RuntimeError("Research checkpoint says walk-forward completed but its manifest does not")
    policy = _select_development_policy(manifest)
    holdout = _run_evaluated_holdout(manifest, policy)
    prediction_path = ROOT / holdout["predictions_path"]
    holdout_predictions = pd.read_parquet(prediction_path)
    selected = policy["selected_policy"]
    result_path = HOLDOUT_OUTPUT_DIR / "holdout_portfolio_results.json"
    book_record = manifest["stages"]["causal_t60_t59_book_replay"]
    portfolio_identity = {
        "predictions_sha256": holdout["predictions_sha256"],
        "policy_sha256": _sha256(POLICY_PATH),
        "t59_coverage_sha256": book_record["t59_coverage_sha256"],
        "t60_coverage_sha256": book_record["t60_coverage_sha256"],
        "market_index_sha256": manifest["input_identity"]["official_markets_sha256"],
        "period_start_utc": CONFIG["development_cutoff_exclusive_utc"],
        "period_cutoff_exclusive_utc": CONFIG["data_cutoff_exclusive_utc"],
        "research_code_sha256": _sha256(Path(__file__)),
    }
    portfolio_stage = manifest.get("stages", {}).get("holdout_portfolio_and_sensitivity", {})
    holdout_result = None
    if (
        portfolio_stage.get("status") == "completed"
        and _json_normalize(portfolio_stage.get("input_identity")) == _json_normalize(portfolio_identity)
        and result_path.is_file()
        and _sha256(result_path) == portfolio_stage.get("results_sha256")
    ):
        cached_result = _read_json(result_path)
        artifacts_match = True
        for item in cached_result.get("policies", {}).values():
            for path_key, hash_key in (
                ("summary_path", "summary_sha256"), ("trades_path", "trades_sha256"),
                ("decisions_path", "decisions_sha256"), ("slippage_path", "slippage_sha256"),
            ):
                if item.get(path_key) is None:
                    continue
                artifact = ROOT / item[path_key]
                if not artifact.is_file() or _sha256(artifact) != item.get(hash_key):
                    artifacts_match = False
                    break
            if not artifacts_match:
                break
        if artifacts_match:
            holdout_result = cached_result
            print("[research] holdout portfolios status=skipped (verified checkpoint)", flush=True)
    if holdout_result is None:
        holdout_result = _evaluate_portfolio_period(
            manifest,
            holdout_predictions,
            start=pd.Timestamp(CONFIG["development_cutoff_exclusive_utc"]),
            cutoff=pd.Timestamp(CONFIG["data_cutoff_exclusive_utc"]),
            policies=[
                (selected["name"], selected["sizing_policy"], float(selected["minimum_net_ev_fraction"])),
                ("fixed_5_usd_control", "fixed_5_usd", 0.0),
                ("free_cash_5pct_control", "free_cash_5pct", 0.0),
            ],
            output_root=HOLDOUT_OUTPUT_DIR,
            period_name="holdout",
            input_identity=portfolio_identity,
        )
        _record_stage(manifest, "holdout_portfolio_and_sensitivity", {
            "results_path": result_path.resolve().relative_to(ROOT).as_posix(),
            "results_sha256": _sha256(result_path),
            "input_identity": portfolio_identity,
            "holdout_market_rows": holdout_result["available_market_rows"]["market_rows_in_period"],
            "policy_names": list(holdout_result["policies"]),
        })
    _write_final_report(manifest, policy, holdout, holdout_result)
    manifest["status"] = "completed"
    manifest["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    manifest["updated_utc"] = manifest["finished_utc"]
    manifest["report_path"] = REPORT_PATH.resolve().relative_to(ROOT).as_posix()
    manifest["report_sha256"] = _sha256(REPORT_PATH)
    _write_json(RESEARCH_MANIFEST_PATH, manifest)
    print(f"[research] completed report={REPORT_PATH}", flush=True)
    return manifest


if __name__ == "__main__":
    run_btc_preopen_research()
