"""Build a hash-verified partial report from existing BTC model-update artifacts."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import sqlite3
from pathlib import Path

import lightgbm
import numpy as np
import optuna
import pandas as pd
import sklearn

import run_btc_market_probability_experiment as probability
import run_btc_market_probability_nested_experiment as nested
import run_btc_preopen_fee_correction as fee_correction
import run_btc_preopen_model_update as model_update
import run_btc_preopen_research as research
import run_btc_preopen_signal_diagnostic as diagnostic
import run_btc_preopen_walk_forward as walk_forward
import select_features


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = model_update.OUTPUT_DIR
MANIFEST_PATH = model_update.MANIFEST_PATH
FREEZE_PATH = model_update.FREEZE_PATH
REPORT_PATH = ROOT / "reports/btc_preopen/btc_preopen_model_update_20261012_partial.md"
TABLE_DIR = ROOT / "reports/btc_preopen/btc_preopen_model_update_20261012_partial"
LEDGER_DIR = OUTPUT_DIR / "partial_report_ledgers"
DEV_FOLDS = ["dev_2026q2_a", "dev_2026q2_b", "dev_2026q2_c"]
HOLDOUT_FOLD = "evaluated_holdout"
MODEL_NAMES = ("A_frozen", "A_reoptimized", "B_dynamic_price_volume")
HOLDOUT_MODEL_NAMES = ("A_frozen", "A_reoptimized")
HOLDOUT_QUALITY_COLUMNS = {
    "A_frozen": "p_btc_calibrated",
    "A_reoptimized": "p_A_reoptimized",
}
DEV_START = pd.Timestamp("2026-04-15T17:05:00Z")
DEV_CUTOFF = pd.Timestamp(model_update.CONFIG["development_cutoff_exclusive_utc"])
DATA_CUTOFF = pd.Timestamp(model_update.CONFIG["data_cutoff_exclusive_utc"])
FROZEN_HURDLE = float(model_update.CONFIG["economic_evaluation"]["portfolio_policy_hurdle"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _verify_prediction_record(manifest: dict, fold_id: str, variant: str) -> Path:
    record = manifest.get("folds", {}).get(fold_id, {}).get(variant, {})
    path = ROOT / record.get("prediction_path", "__missing__")
    if record.get("status") != "completed" or not path.is_file():
        raise RuntimeError(f"Missing completed prediction artifact for {variant}/{fold_id}")
    if _sha256(path) != record.get("prediction_sha256"):
        raise RuntimeError(f"Prediction artifact hash changed for {variant}/{fold_id}")
    return path


def available_holdout_variants(manifest: dict) -> dict[str, Path]:
    """Return only hash-verified completed holdout variants; partial runs are valid."""
    available = {}
    for variant in ("A_reoptimized", "B_dynamic_price_volume"):
        record = manifest.get("folds", {}).get(HOLDOUT_FOLD, {}).get(variant, {})
        path = ROOT / record.get("prediction_path", "__missing__")
        if record.get("status") == "completed" and path.is_file():
            if _sha256(path) != record.get("prediction_sha256"):
                raise RuntimeError(f"Prediction artifact hash changed for {variant}/{HOLDOUT_FOLD}")
            available[variant] = path
    if "A_reoptimized" not in available:
        raise RuntimeError("A_reoptimized holdout predictions are required for the partial comparison")
    return available


def _verify_frozen_inputs(manifest: dict, freeze: dict) -> dict:
    config_hash = model_update._sha256(model_update.CONFIG_PATH)
    source_identity = manifest.get("source_identity", {})
    if config_hash != source_identity.get("configuration_sha256"):
        raise RuntimeError("Frozen config hash differs from the existing model-update manifest")
    if config_hash != freeze.get("configuration_sha256"):
        raise RuntimeError("Frozen config hash differs from the development freeze")
    if manifest.get("development_freeze_sha256") != _sha256(FREEZE_PATH):
        raise RuntimeError("Development freeze hash differs from the outer manifest")
    bundle_path = ROOT / model_update.CONFIG["frozen_baseline"]["bundle_path"]
    if _sha256(bundle_path) != model_update.CONFIG["frozen_baseline"]["bundle_sha256"]:
        raise RuntimeError("Frozen A model bundle hash changed")
    if source_identity.get("baseline_bundle_sha256") != _sha256(bundle_path):
        raise RuntimeError("Frozen A bundle differs from the model-update source identity")
    return {
        "configuration_sha256": config_hash,
        "outer_manifest_sha256": _sha256(MANIFEST_PATH),
        "development_freeze_sha256": _sha256(FREEZE_PATH),
        "baseline_bundle_sha256": _sha256(bundle_path),
        "source_identity": source_identity,
    }


def _paired_block_interval(
    frame: pd.DataFrame,
    target_column: str,
    left_column: str,
    right_column: str,
    *,
    seed: int = 20261012,
    replicates: int = 2000,
) -> dict:
    ordered = frame.loc[
        frame[target_column].isin((0.0, 1.0))
        & pd.to_numeric(frame[left_column], errors="coerce").notna()
        & pd.to_numeric(frame[right_column], errors="coerce").notna()
    ].sort_values("market_start_utc", kind="stable")
    if ordered.empty:
        raise RuntimeError(f"No paired observations for {target_column}")
    y = ordered[target_column].to_numpy(dtype=np.float64)
    left = np.clip(ordered[left_column].to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
    right = np.clip(ordered[right_column].to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
    log_loss_delta = (
        -(y * np.log(left) + (1.0 - y) * np.log1p(-left))
        + (y * np.log(right) + (1.0 - y) * np.log1p(-right))
    )
    brier_delta = np.square(left - y) - np.square(right - y)
    days = pd.to_datetime(ordered.market_start_utc, utc=True).dt.floor("D")
    full_days = pd.date_range(days.min(), days.max(), freq="D", tz="UTC")
    block_count = int(np.ceil(len(full_days) / 3.0))
    daily_count = np.asarray([int(days.eq(day).sum()) for day in full_days], dtype=np.float64)
    daily_values = {}
    for metric, values in (("log_loss", log_loss_delta), ("brier", brier_delta)):
        by_day = pd.DataFrame({"day": days.to_numpy(), "delta": values}).groupby("day").delta.sum()
        daily_values[metric] = np.asarray([float(by_day.get(day, 0.0)) for day in full_days])
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(replicates, dtype=np.float64) for name in daily_values}
    for draw in range(replicates):
        starts = rng.integers(0, len(full_days), size=block_count)
        positions = np.concatenate([(np.arange(start, start + 3) % len(full_days)) for start in starts])[:len(full_days)]
        denominator = float(daily_count[positions].sum())
        for name in daily_values:
            draws[name][draw] = float(daily_values[name][positions].sum() / denominator)
    return {
        "n": int(len(ordered)),
        "target": target_column,
        "log_loss_difference_left_minus_right": float(log_loss_delta.mean()),
        "log_loss_difference_95pct": np.quantile(draws["log_loss"], [0.025, 0.975]).tolist(),
        "brier_difference_left_minus_right": float(brier_delta.mean()),
        "brier_difference_95pct": np.quantile(draws["brier"], [0.025, 0.975]).tolist(),
        "method": f"paired circular moving 3-UTC-day blocks; {replicates} replicates; seed {seed}",
    }


def _quality_rows(frame: pd.DataFrame, names: tuple[str, ...], period: str, edges: list[float]) -> list[dict]:
    rows = []
    for name in names:
        column = model_update.PROBABILITY_COLUMNS[name]
        metrics, _ = model_update._metric_rows(frame, period, name, column, edges)
        rows.extend(metrics)
    return rows


def _load_snapshots() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fee_correction._configure_output_paths()
    correction_manifest_path = fee_correction.CORRECTION_DIR / "research_manifest.json"
    correction_manifest = research._read_json(correction_manifest_path)
    t59, t60, book_record = research._load_enriched_book_snapshots(correction_manifest)
    if len(t59) != 50184 or len(t60) != 50184:
        raise RuntimeError("The verified corrected T-60/T-59 snapshot set no longer has 50,184 markets")
    return t59, t60, book_record


def _feature_selection_rows(manifest: dict) -> list[dict]:
    rows = []
    for fold_id in DEV_FOLDS:
        shared_run = ROOT / manifest["folds"][fold_id]["A_reoptimized"]["run_root"]
        run_manifest = research._read_json(shared_run / "run_manifest.json")
        selection = run_manifest["stages"]["feature_selection"]
        result = selection["result"]
        outputs = {Path(row["path"]).name: row for row in selection.get("outputs", [])}
        sweep_artifact = outputs.get("topk_sweep.parquet")
        if sweep_artifact is None:
            raise RuntimeError(f"Missing verified top-k sweep for {fold_id}")
        sweep_path = ROOT / sweep_artifact["path"]
        if _sha256(sweep_path) != sweep_artifact["sha256"]:
            raise RuntimeError(f"Top-k sweep hash changed for {fold_id}")
        sweep = pd.read_parquet(sweep_path)
        best = int(sweep.loc[sweep.selection_score.idxmin(), "k"])
        recommended = int(result["feature_count"])
        savings = best - recommended
        for variant in ("A_reoptimized", "B_dynamic_price_volume"):
            record = manifest["folds"][fold_id][variant]
            rows.append({
                "fold_id": fold_id,
                "variant": variant,
                "best_k": best,
                "selector_recommended_features": recommended,
                "model_features_used": int(record["model_feature_count"]),
                "added_candidate_features": len(record.get("added_candidate_features", [])),
                "added_candidate_feature_names": ",".join(record.get("added_candidate_features", [])),
                "savings_from_best_k": savings,
                "minimum_required_savings": int(select_features.MIN_PLATEAU_FEATURE_SAVINGS),
                "savings_gate_met": bool(savings >= select_features.MIN_PLATEAU_FEATURE_SAVINGS),
                "topk_evaluations": int(result["topk_evaluations"]),
                "selector_artifact_sha256": sweep_artifact["sha256"],
            })
    return rows


def _trial_diagnostics(manifest: dict) -> tuple[list[dict], list[dict], dict]:
    record = manifest["folds"][HOLDOUT_FOLD]
    run_root = ROOT / record["A_reoptimized"]["run_root"]
    inner = research._read_json(run_root / "run_manifest.json")
    tuning = inner["stages"]["model_tuning"]
    stage_signature = str(tuning.get("input_signature", ""))
    if len(stage_signature) < 12 or tuning.get("status") != "failed":
        raise RuntimeError("B model-tuning failure manifest lacks its frozen stage signature")
    database = run_root / "stages" / f"model_tuning_{stage_signature[:12]}" / "study.sqlite3"
    if not database.is_file():
        raise RuntimeError("B model-tuning SQLite checkpoint is missing from its failed stage directory")
    study_artifact = {"path": database.resolve().relative_to(ROOT).as_posix(), "sha256": _sha256(database)}
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        study_id, study_name = connection.execute("SELECT study_id, study_name FROM studies").fetchone()
        states = dict(connection.execute(
            "SELECT state, COUNT(*) FROM trials WHERE study_id=? GROUP BY state", (study_id,),
        ).fetchall())
        trials = connection.execute(
            "SELECT trial_id, number, state, datetime_start, datetime_complete "
            "FROM trials WHERE study_id=? ORDER BY number", (study_id,),
        ).fetchall()
        param_rows = connection.execute(
            "SELECT trial_id, param_name, param_value FROM trial_params ORDER BY trial_id, param_name",
        ).fetchall()
        value_rows = connection.execute(
            "SELECT trial_id, value FROM trial_values WHERE objective=0",
        ).fetchall()
        attr_rows = connection.execute(
            "SELECT trial_id, key, value_json FROM trial_user_attributes",
        ).fetchall()
        curve_rows = connection.execute(
            "SELECT trial_id, step, intermediate_value FROM trial_intermediate_values "
            "WHERE intermediate_value_type='FINITE' ORDER BY trial_id, step",
        ).fetchall()
    finally:
        connection.close()
    params: dict[int, dict] = {}
    for trial_id, name, value in param_rows:
        params.setdefault(int(trial_id), {})[name] = float(value)
    values = {int(trial_id): float(value) for trial_id, value in value_rows}
    attrs: dict[int, dict] = {}
    for trial_id, name, value_json in attr_rows:
        attrs.setdefault(int(trial_id), {})[name] = json.loads(value_json)
    curves: dict[int, list] = {}
    for trial_id, step, value in curve_rows:
        curves.setdefault(int(trial_id), []).append({
            "step": int(step), "iteration": int(step) + 1, "value": float(value),
        })
    rows = []
    for trial_id, number, state, started, completed in trials:
        duration = None
        if started and completed:
            duration = (pd.Timestamp(completed) - pd.Timestamp(started)).total_seconds()
        trial_curve = curves.get(int(trial_id), [])
        rows.append({
            "trial_number": int(number),
            "state": state,
            "duration_seconds": duration,
            "objective_value": values.get(int(trial_id)),
            "best_iteration": attrs.get(int(trial_id), {}).get("best_iteration"),
            "params": json.dumps(params.get(int(trial_id), {}), sort_keys=True),
            "pruned_at_iteration": trial_curve[-1]["iteration"] if state == "PRUNED" and trial_curve else None,
            "intermediate_curve": json.dumps(trial_curve, separators=(",", ":")),
            "user_attributes": json.dumps(attrs.get(int(trial_id), {}), sort_keys=True, default=str),
        })
    completed = [row for row in rows if row["state"] == "COMPLETE" and row["duration_seconds"] is not None]
    pruned = [row for row in rows if row["state"] == "PRUNED" and row["duration_seconds"] is not None]
    complete_durations = [row["duration_seconds"] for row in completed]
    pruned_durations = [row["duration_seconds"] for row in pruned]
    latest_record = inner["stages"]["model_tuning"]
    log_text = (run_root / "run.log").read_text(encoding="utf-8", errors="replace")
    resume_remaining_match = re.search(
        r"\[preopen-resume\] continuing evaluated_holdout B model_tuning with ([0-9.]+)s remaining",
        log_text,
    )
    tuning_allowance = float(model_update.CONFIG["runtime_profile"]["per_model_tuning_and_fit_budget_seconds"])
    tuning_allowance -= min(300.0, tuning_allowance * 0.1)
    resume_remaining = float(resume_remaining_match.group(1)) if resume_remaining_match else None
    prior_attempt_seconds = tuning_allowance - resume_remaining if resume_remaining is not None else None
    latest_stage_seconds = float(latest_record.get("elapsed_seconds", 0.0))
    complete_best = [int(row["best_iteration"]) for row in completed if row["best_iteration"] is not None]
    historical_resume = (ROOT / "scripts/resume_btc_preopen_holdout_b_tuning_and_evaluation.py").read_text(encoding="utf-8")
    prior_trial_match = re.search(r"EXPECTED_PREVIOUS_TRIALS\s*=\s*(\d+)", historical_resume)
    prior_success_match = re.search(r"EXPECTED_PREVIOUS_SUCCESSES\s*=\s*(\d+)", historical_resume)
    candidates = sorted(pruned, key=lambda row: (json.loads(row["params"]).get("learning_rate", float("inf")), row["trial_number"]))
    candidate_groups = np.array_split(np.asarray(candidates, dtype=object), 5)
    test_candidates = []
    for band, group in enumerate(candidate_groups, start=1):
        ordered_group = list(group)
        chosen = ordered_group[len(ordered_group) // 2]
        test_candidates.append({
            "test_config_id": f"B_PRUNED_LR_Q{band}",
            "source_trial_number": chosen["trial_number"],
            "selection_rule": "median learning_rate-ranked PRUNED trial in quintile; ties by trial_number",
            "params": chosen["params"],
            "source_pruned_at_iteration": chosen["pruned_at_iteration"],
        })
    summary = {
        "study_name": study_name,
        "database_path": study_artifact["path"],
        "database_sha256": study_artifact["sha256"],
        "counts": {state: int(states.get(state, 0)) for state in ("COMPLETE", "PRUNED", "FAIL" )},
        "trial_count": len(rows),
        "median_complete_trial_seconds": float(np.median(complete_durations)) if complete_durations else None,
        "median_pruned_trial_seconds": float(np.median(pruned_durations)) if pruned_durations else None,
        "median_complete_best_iteration": float(np.median([row["best_iteration"] for row in completed if row["best_iteration"]])) if any(row["best_iteration"] for row in completed) else None,
        "prune_iterations": {
            str(int(key)): int(value)
            for key, value in sorted(pd.Series([row["pruned_at_iteration"] for row in pruned]).value_counts().to_dict().items())
        },
        "failed_stage_elapsed_seconds_last_attempt": latest_record.get("elapsed_seconds"),
        "run_manifest_stage_attempt_counter": latest_record.get("attempt"),
        "b_tuning_attempts_visible_in_log": int(log_text.count("[preopen-resume] continuing evaluated_holdout B model_tuning")) + 1,
        "continuation_preflight_trial_count_from_saved_helper": int(prior_trial_match.group(1)) if prior_trial_match else None,
        "continuation_preflight_success_count_from_saved_helper": int(prior_success_match.group(1)) if prior_success_match else None,
        "failed_stage_error": latest_record.get("error"),
        "tuning_allowance_seconds": tuning_allowance,
        "resume_remaining_seconds_from_log": resume_remaining,
        "prior_attempt_seconds_reconstructed_from_log": prior_attempt_seconds,
        "latest_stage_wall_seconds": latest_stage_seconds,
        "cumulative_b_stage_wall_seconds_estimate": (
            prior_attempt_seconds + latest_stage_seconds if prior_attempt_seconds is not None else None
        ),
        "resume_overhead_after_budget_deadline_estimate_seconds": (
            latest_stage_seconds - resume_remaining if resume_remaining is not None else None
        ),
        "historical_budget_estimate_uncertainty_seconds": 0.05 if resume_remaining is not None else None,
        "completed_best_iterations": complete_best,
        "completed_trials_best_after_first_checkpoint": int(sum(value > 25 for value in complete_best)),
        "median_complete_learning_rate": float(np.median([
            json.loads(row["params"])["learning_rate"] for row in completed
        ])) if completed else None,
        "median_pruned_learning_rate": float(np.median([
            json.loads(row["params"])["learning_rate"] for row in pruned
        ])) if pruned else None,
        "failed_b_final_fit_seconds": 0.0,
        "resume_helper_sha256": _sha256(ROOT / "scripts/resume_btc_preopen_holdout_b_tuning_and_evaluation.py"),
        "requested_successful_minimum": int(model_update.CONFIG["model_search"]["minimum_successful_trials_per_variant_per_fold"]),
        "development_test_cost_seconds_five_configs_across_three_folds": float(sum(
            5 * np.median([
                row["duration_seconds"] for row in _development_trial_durations(manifest, fold_id)
                if row["state"] == "COMPLETE" and row["duration_seconds"] is not None
            ])
            for fold_id in DEV_FOLDS
        )),
    }
    return rows, test_candidates, summary


def _development_trial_durations(manifest: dict, fold_id: str) -> list[dict]:
    run_root = ROOT / manifest["folds"][fold_id]["B_dynamic_price_volume"]["run_root"]
    run = research._read_json(run_root / "run_manifest.json")
    stage = run["stages"]["model_tuning"]
    artifact = next(row for row in stage["outputs"] if Path(row["path"]).name == "study.sqlite3")
    path = ROOT / artifact["path"]
    if _sha256(path) != artifact["sha256"]:
        raise RuntimeError(f"Development B Optuna database hash changed for {fold_id}")
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT state, (julianday(datetime_complete)-julianday(datetime_start))*86400 "
            "FROM trials WHERE datetime_complete IS NOT NULL",
        ).fetchall()
    finally:
        connection.close()
    return [{"state": state, "duration_seconds": float(seconds)} for state, seconds in rows]


def _set_model_probability_columns(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    if name == "A_frozen":
        result["p_candidate_raw"] = result["p_btc_raw"]
        result["p_candidate_platt"] = result["p_btc_calibrated"]
    else:
        result["p_candidate_raw"] = result[f"p_raw_{name}"]
        result["p_candidate_platt"] = result[f"p_{name}"]
    return result


def _market_map_for_frame(frame: pd.DataFrame) -> dict:
    rows = frame.loc[frame.condition_id.notna()].copy()
    return research._snapshot_map(rows)


def _decision_ledger(frame: pd.DataFrame, name: str, t59: pd.DataFrame, t60: pd.DataFrame) -> pd.DataFrame:
    model_frame = _set_model_probability_columns(frame, name)
    market_map = _market_map_for_frame(model_frame)
    t59_map = research._snapshot_map(t59)
    t60_map = research._snapshot_map(t60)
    t59_evaluations = diagnostic._t59_execution_evaluations(
        market_map, t59_map, diagnostic.FIXED_ORDER_USD,
    )
    decisions = diagnostic._independent_decisions(
        model_frame, market_map, t59_evaluations, t60_map, (FROZEN_HURDLE,),
    )
    decisions.insert(0, "forecast", name)
    return decisions


def _group_economics(ledger: pd.DataFrame, price_edges: list[float], ev_edges: list[float]) -> list[dict]:
    selected = ledger.loc[
        ledger.period.eq("holdout")
        & ledger.decision_status.eq("positive_t60_net_ev")
        & pd.to_numeric(ledger.t60_net_ev_fraction, errors="coerce").ge(FROZEN_HURDLE - 1e-12)
        & ledger.execution_status.eq("filled")
        & ledger.official_outcome_up.isin((0.0, 1.0))
    ].copy()
    selected["t60_price_bin"] = [diagnostic._bin_index(value, price_edges) for value in selected.t60_vwap]
    selected["t60_ev_bin"] = [diagnostic._bin_index(value, ev_edges) for value in selected.t60_net_ev_fraction]
    group_specs = [
        ("side", "chosen_side"),
        ("month_utc", "month_utc"),
        ("t60_price_decile", "t60_price_bin"),
        ("t60_ev_decile", "t60_ev_bin"),
    ]
    rows = []
    for label, column in group_specs:
        for (forecast, group_value), group in selected.groupby(["forecast", column], dropna=False, sort=True):
            resolved = group.loc[group.official_outcome_up.isin((0.0, 1.0))]
            full_depth = int(resolved.depth_known.fillna(False).sum())
            row = {
                "group_type": label,
                "forecast": forecast,
                "group": group_value + 1 if label in ("t60_price_decile", "t60_ev_decile") else group_value,
                "fills": int(len(group)),
                "win_rate": float(resolved.won.mean()) if len(resolved) else None,
                "mean_predicted_chosen_side_probability": float(resolved.p_chosen_side_calibrated.mean()) if len(resolved) else None,
                "mean_t60_ev_fraction": float(group.t60_net_ev_fraction.mean()),
                "mean_t60_price": float(group.t60_vwap.mean()),
                "mean_t59_execution_vwap": float(group.vwap.mean()),
                "expected_net_t60_usd": float(resolved.t60_expected_net_pnl_usd.sum()) if len(resolved) else 0.0,
                "expected_net_t59_usd": float(resolved.expected_net_t59_usd.sum()) if len(resolved) else 0.0,
                "realized_net_pnl_usd": float(resolved.net_pnl_usd.sum()) if len(resolved) else 0.0,
                "realized_fee_effect_usd": float(resolved.realized_fee_effect_usd.sum()) if len(resolved) else 0.0,
                "full_depth_fills": full_depth,
                "proxy_fills": int(len(resolved) - full_depth),
                "full_depth_share": full_depth / len(resolved) if len(resolved) else None,
                "proxy_share": (len(resolved) - full_depth) / len(resolved) if len(resolved) else None,
            }
            if label == "t60_price_decile":
                row["dev_t60_price_lower"] = price_edges[group_value - 1] if group_value > 0 else None
                row["dev_t60_price_upper"] = price_edges[group_value] if group_value < len(price_edges) else None
            if label == "t60_ev_decile":
                row["dev_t60_ev_lower"] = ev_edges[group_value - 1] if group_value > 0 else None
                row["dev_t60_ev_upper"] = ev_edges[group_value] if group_value < len(ev_edges) else None
            rows.append(row)
    return rows


def _compare_previous_a_ledger(current_detail: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    previous_path = nested.OUTPUT_DIR / "independent_order_detail.parquet"
    if not previous_path.is_file():
        return {"status": "previous_market_ledger_missing", "path": previous_path.as_posix()}, pd.DataFrame()
    previous = pd.read_parquet(previous_path)
    old = previous.loc[
        previous.period.eq("holdout_diagnostic")
        & previous.forecast.eq("A_BTC_calibrated")
        & previous.t60_ev_fraction.ge(FROZEN_HURDLE - 1e-12)
    ].copy()
    new = current_detail.loc[
        current_detail.period.eq("holdout_diagnostic")
        & current_detail.forecast.eq("A_frozen")
        & current_detail.t60_ev_fraction.ge(FROZEN_HURDLE - 1e-12)
    ].copy()
    old = old.set_index("condition_id", drop=False)
    new = new.set_index("condition_id", drop=False)
    old_ids, new_ids = set(old.index), set(new.index)
    common_ids = sorted(old_ids & new_ids)
    comparable = (
        ("chosen_side", "chosen_side"),
        ("execution_status", "execution_status"),
        ("execution_reason", "execution_reason"),
        ("filled_gross_usd", "filled_gross_usd"),
        ("vwap", "vwap"),
        ("expected_net_t59_usd", "expected_net_t59_usd"),
        ("net_pnl_usd", "net_pnl_usd"),
    )
    differences = []
    for condition_id in common_ids:
        left, right = old.loc[condition_id], new.loc[condition_id]
        if isinstance(left, pd.DataFrame):
            left = left.iloc[0]
        if isinstance(right, pd.DataFrame):
            right = right.iloc[0]
        changed = {}
        for old_key, new_key in comparable:
            old_value, new_value = left.get(old_key), right.get(new_key)
            if pd.isna(old_value) and pd.isna(new_value):
                continue
            if isinstance(old_value, (int, float, np.number)) and isinstance(new_value, (int, float, np.number)):
                differs = not np.isclose(float(old_value), float(new_value), rtol=0.0, atol=1e-9)
            else:
                differs = old_value != new_value
            if differs:
                changed[old_key] = (old_value, new_value)
        if changed:
            differences.append({
                "condition_id": condition_id,
                "old_side": left.get("chosen_side"),
                "new_side": right.get("chosen_side"),
                "old_execution_status": left.get("execution_status"),
                "new_execution_status": right.get("execution_status"),
                "changed_fields": json.dumps(changed, sort_keys=True, default=str),
            })
    diff_frame = pd.DataFrame(differences, columns=(
        "condition_id",
        "old_side",
        "new_side",
        "old_execution_status",
        "new_execution_status",
        "changed_fields",
    ))
    result = {
        "status": "exact_match" if old_ids == new_ids and not differences else "market_or_ledger_mismatch",
        "previous_ledger_path": previous_path.resolve().relative_to(ROOT).as_posix(),
        "previous_ledger_sha256": _sha256(previous_path),
        "previous_signals_at_hurdle": int(len(old)),
        "recomputed_signals_at_hurdle": int(len(new)),
        "previous_only_condition_ids": len(old_ids - new_ids),
        "recomputed_only_condition_ids": len(new_ids - old_ids),
        "common_condition_ids": len(common_ids),
        "common_condition_ids_with_changed_ledger_fields": len(differences),
        "mismatch_details_path": (TABLE_DIR / "previous_a_ledger_mismatches.csv").resolve().relative_to(ROOT).as_posix(),
    }
    return result, diff_frame


def _portfolio_rows(
    frame: pd.DataFrame,
    t59: pd.DataFrame,
    t60: pd.DataFrame,
    portfolio_source_identity: dict,
) -> list[dict]:
    holdout = frame.loc[
        frame.common_sample
        & frame.official_outcome_up.isin((0.0, 1.0))
        & frame.market_start_utc.ge(DEV_CUTOFF)
        & frame.market_start_utc.lt(DATA_CUTOFF)
    ].copy()
    t59_map = research._snapshot_map(t59)
    t60_map = research._snapshot_map(t60)
    rows = []
    for name in HOLDOUT_MODEL_NAMES:
        candidate = _set_model_probability_columns(holdout, name)
        markets = _market_map_for_frame(candidate)
        for capital in model_update.PORTFOLIO_CAPITALS:
            destination = LEDGER_DIR / "holdout_portfolios" / name / str(int(capital))
            cache_path = destination / "report_input_identity.json"
            summary_path = destination / "summary.json"
            equity_path = destination / "equity_path.csv"
            expected_identity = {**portfolio_source_identity, "forecast": name, "starting_capital_usd": capital}
            if cache_path.is_file() and summary_path.is_file() and equity_path.is_file():
                cached_identity = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                cached_equity = pd.read_csv(equity_path)
                if (
                    cached_identity == expected_identity
                    and cached_summary.get("accounting_audit", {}).get("status") == "passed"
                    and cached_summary.get("minimum_net_ev_fraction") == FROZEN_HURDLE
                    and not cached_equity.empty
                    and np.isclose(
                        float(cached_equity.iloc[-1]["cash_usd"]),
                        float(cached_summary["ending_cash_after_settlement_usd"]),
                        rtol=0.0, atol=1e-8,
                    )
                ):
                    summary = cached_summary
                    stop_diagnostics = (
                        _portfolio_stop_diagnostics(destination, summary)
                        if float(capital) == 100.0 else {}
                    )
                    rows.append({
                        "forecast": name,
                        "starting_capital_usd": capital,
                        "trade_count": int(summary.get("trade_count", 0)),
                        "gross_turnover_usd": float(summary.get("gross_turnover_usd", 0.0)),
                        "fees_paid_estimated_usd": float(summary.get("fees_paid_estimated_usd", 0.0)),
                        "net_pnl_usd": float(summary.get("net_pnl_usd", 0.0)),
                        "ending_cash_after_settlement_usd": float(summary.get("ending_cash_after_settlement_usd", capital)),
                        "max_drawdown_cost_basis_equity": float(summary.get("max_drawdown_cost_basis_equity", 0.0)),
                        "accounting_audit_status": summary.get("accounting_audit", {}).get("status"),
                        "monthly_diagnostics": json.dumps(summary.get("monthly_diagnostics", []), sort_keys=True, default=str),
                        **stop_diagnostics,
                        "ledger_path": destination.resolve().relative_to(ROOT).as_posix(),
                    })
                    continue
            summary, trades, decisions, curve = diagnostic._run_holdout_portfolio(
                capital, candidate, t59_map, markets, t60_map, FROZEN_HURDLE,
            )
            destination.mkdir(parents=True, exist_ok=True)
            if trades:
                pd.DataFrame(trades).to_parquet(destination / "trades.parquet", index=False, compression="zstd")
            if decisions:
                pd.DataFrame(decisions).to_parquet(destination / "decisions.parquet", index=False, compression="zstd")
            pd.DataFrame(curve).to_csv(destination / "equity_path.csv", index=False)
            _write_json(destination / "summary.json", summary)
            _write_json(cache_path, expected_identity)
            stop_diagnostics = (
                _portfolio_stop_diagnostics(destination, summary)
                if float(capital) == 100.0 else {}
            )
            rows.append({
                "forecast": name,
                "starting_capital_usd": capital,
                "trade_count": int(summary.get("trade_count", 0)),
                "gross_turnover_usd": float(summary.get("gross_turnover_usd", 0.0)),
                "fees_paid_estimated_usd": float(summary.get("fees_paid_estimated_usd", 0.0)),
                "net_pnl_usd": float(summary.get("net_pnl_usd", 0.0)),
                "ending_cash_after_settlement_usd": float(summary.get("ending_cash_after_settlement_usd", capital)),
                "max_drawdown_cost_basis_equity": float(summary.get("max_drawdown_cost_basis_equity", 0.0)),
                "accounting_audit_status": summary.get("accounting_audit", {}).get("status"),
                "monthly_diagnostics": json.dumps(summary.get("monthly_diagnostics", []), sort_keys=True, default=str),
                **stop_diagnostics,
                "ledger_path": destination.resolve().relative_to(ROOT).as_posix(),
            })
    return rows


def _portfolio_stop_diagnostics(destination: Path, summary: dict) -> dict:
    decisions = pd.read_parquet(
        destination / "decisions.parquet",
        columns=[
            "decision_time_utc", "market_start_utc", "skip_reason", "requested_gross_usd",
            "cash_before_usd", "locked_cost_before_usd",
        ],
    )
    trades_path = destination / "trades.parquet"
    trades = (
        pd.read_parquet(trades_path, columns=["entry_time_utc"])
        if trades_path.is_file() else pd.DataFrame(columns=["entry_time_utc"])
    )
    rejection_counts = {
        str(reason): int(count)
        for reason, count in summary.get("rejections_by_reason", {}).items()
    }
    minimum_count = rejection_counts.get("below_minimum_order_usd", 0)
    locked_cash_count = sum(
        count for reason, count in rejection_counts.items()
        if reason in {"locked_cash", "insufficient_free_cash", "cash_locked"}
    )
    minimum_decisions = decisions.loc[
        decisions["skip_reason"].eq("below_minimum_order_usd")
    ]
    if len(minimum_decisions) != minimum_count:
        raise RuntimeError("Portfolio summary minimum-skip count differs from its decision ledger")
    last_decision_index = pd.to_datetime(decisions["decision_time_utc"], utc=True).idxmax()
    last_decision = decisions.loc[last_decision_index]
    free_cash = float(last_decision["cash_before_usd"])
    equity = free_cash + float(last_decision["locked_cost_before_usd"])
    validated_minimum_skips = (
        minimum_decisions["requested_gross_usd"].lt(5.0)
        & np.isclose(
            minimum_decisions["requested_gross_usd"],
            0.05 * minimum_decisions["cash_before_usd"],
            rtol=0.0,
            atol=1e-8,
        )
    )
    all_minimum_skips_valid = bool(validated_minimum_skips.all())
    return {
        "below_minimum_order_rejections": minimum_count,
        "locked_cash_rejections": locked_cash_count,
        "other_portfolio_rejections": sum(rejection_counts.values()) - minimum_count - locked_cash_count,
        "rejections_by_reason": json.dumps(rejection_counts, sort_keys=True),
        "last_trade_entry_time_utc": (
            str(pd.to_datetime(trades["entry_time_utc"], utc=True).max()) if not trades.empty else None
        ),
        "last_decision_time_utc": pd.Timestamp(last_decision["decision_time_utc"]).isoformat(),
        "free_cash_at_last_decision_usd": free_cash,
        "equity_at_last_decision_usd": equity,
        "five_percent_stake_at_last_decision_usd": 0.05 * free_cash,
        "minimum_skip_ledger_check": "passed" if all_minimum_skips_valid else "failed",
        "minimum_skips_below_five_usd": int(validated_minimum_skips.sum()),
        "minimum_skip_count_for_validation": int(len(minimum_decisions)),
    }


def _format_table(rows: list[dict], columns: list[tuple[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(label for label, _ in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append("n/a" if value is None else str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_report() -> dict:
    manifest = research._read_json(MANIFEST_PATH)
    freeze = research._read_json(FREEZE_PATH)
    frozen_identity = _verify_frozen_inputs(manifest, freeze)
    t59, t60, book_record = _load_snapshots()
    source_identity = manifest.get("source_identity", {})
    if source_identity.get("corrected_t60_snapshot_sha256") != book_record["t60_coverage_sha256"]:
        raise RuntimeError("T-60 snapshot hash differs from the original model-update identity")
    if source_identity.get("corrected_t59_snapshot_sha256") != book_record["t59_coverage_sha256"]:
        raise RuntimeError("T-59 snapshot hash differs from the original model-update identity")

    dev_ids = [walk_forward.SEED_FOLD_ID, *DEV_FOLDS]
    candidate_dev = {
        name: model_update._load_candidate_predictions(manifest, name, dev_ids)
        for name in ("A_reoptimized", "B_dynamic_price_volume")
    }
    baseline_dev_predictions, baseline_identity = research._load_model_predictions(include_holdout=False)
    frozen_prediction_identity = freeze.get("baseline_prediction_identity", {})
    if any(frozen_prediction_identity.get(key) != baseline_identity.get(key) for key in (
        "walk_forward_manifest_sha256", "prediction_sources", "prediction_rows", "prediction_sha256",
    )):
        raise RuntimeError("Verified baseline prediction identity differs from the frozen selection record")
    dev_calendar, _ = research._build_market_calendar(baseline_dev_predictions)
    dev_calendar = model_update._ensure_nominal_decision_at(dev_calendar)
    development = model_update._merge_predictions(probability._build_dataset(dev_calendar, t60), candidate_dev)
    dev = development.loc[
        development.common_sample
        & development.official_outcome_up.isin((0.0, 1.0))
        & development.market_start_utc.lt(DEV_CUTOFF)
    ].copy()
    if len(dev) != int(freeze["development_rows_common"]):
        raise RuntimeError("Development common market count differs from the frozen selection record")

    holdout_candidates = available_holdout_variants(manifest)
    baseline_all, baseline_all_identity = research._load_model_predictions(include_holdout=True)
    holdout_calendar, _ = research._build_market_calendar(baseline_all)
    holdout_calendar = model_update._ensure_nominal_decision_at(holdout_calendar)
    holdout_all = model_update._merge_predictions(
        probability._build_dataset(holdout_calendar, t60),
        {"A_reoptimized": model_update._load_candidate_predictions(
            manifest, "A_reoptimized", [*dev_ids, HOLDOUT_FOLD],
        )},
    )
    holdout_quality = holdout_all.loc[
        holdout_all.market_start_utc.ge(DEV_CUTOFF)
        & holdout_all.market_start_utc.lt(DATA_CUTOFF)
        & holdout_all.p_btc_calibrated.notna()
        & holdout_all.p_A_reoptimized.notna()
    ].copy()
    holdout_economic = holdout_all.loc[
        holdout_all.common_sample
        & holdout_all.official_outcome_up.isin((0.0, 1.0))
        & holdout_all.market_start_utc.ge(DEV_CUTOFF)
        & holdout_all.market_start_utc.lt(DATA_CUTOFF)
    ].copy()

    calibration_edges = probability._metric_edges(dev.p_btc_calibrated.to_numpy(dtype=np.float64))
    development_quality = _quality_rows(dev, MODEL_NAMES, "development_oof", calibration_edges)
    development_fold_quality = []
    for fold_id, group in dev.groupby("fold_id", sort=True):
        development_fold_quality.extend(_quality_rows(group, MODEL_NAMES, str(fold_id), calibration_edges))
    holdout_quality_rows = _quality_rows(holdout_quality, HOLDOUT_MODEL_NAMES, "holdout_diagnostic", calibration_edges)
    holdout_paired = {
        target: _paired_block_interval(holdout_quality, target, "p_A_reoptimized", "p_btc_calibrated")
        for target in ("official_outcome_up", "target_binance_proxy_up")
    }

    dev_decisions = _decision_ledger(development, "A_frozen", t59, t60)
    dev_positive = dev_decisions.loc[dev_decisions.decision_status.eq("positive_t60_net_ev")]
    price_edges = diagnostic._development_quantile_edges(dev_positive.t60_vwap.to_numpy(dtype=float), 10)
    ev_edges = diagnostic._development_quantile_edges(dev_positive.t60_net_ev_fraction.to_numpy(dtype=float), 10)
    holdout_ledgers = pd.concat([
        _decision_ledger(holdout_all, name, t59, t60) for name in HOLDOUT_MODEL_NAMES
    ], ignore_index=True, sort=False)
    funnel_rows = []
    for name in HOLDOUT_MODEL_NAMES:
        model_ledger = holdout_ledgers.loc[holdout_ledgers.forecast.eq(name)]
        summary = diagnostic._aggregate_signals(
            model_ledger, "holdout", FROZEN_HURDLE,
            bootstrap_start=DEV_CUTOFF, bootstrap_end=DATA_CUTOFF,
        )
        funnel_rows.append({
            "forecast": name,
            "market_count": summary["market_count"],
            "prediction_count": summary["prediction_count"],
            "t60_priceable_decisions": summary["t60_priceable_decisions"],
            "signals_at_1pct_t60_ev": summary["signals"],
            "t59_fills": summary["fills"],
            "resolved_fills": summary["resolved_fills_for_pnl"],
            "non_execution_reasons": json.dumps(summary["non_execution_reasons"], sort_keys=True),
            "wins": summary["wins"],
            "losses": summary["losses"],
            "mean_predicted_chosen_side_probability": float(pd.to_numeric(
                model_ledger.loc[
                    model_ledger.period.eq("holdout")
                    & model_ledger.decision_status.eq("positive_t60_net_ev")
                    & model_ledger.t60_net_ev_fraction.ge(FROZEN_HURDLE - 1e-12)
                    & model_ledger.execution_status.eq("filled")
                    & model_ledger.official_outcome_up.isin((0.0, 1.0)),
                    "p_chosen_side_calibrated",
                ], errors="coerce",
            ).mean()),
            "observed_win_rate": summary["wins"] / summary["resolved_fills_for_pnl"] if summary["resolved_fills_for_pnl"] else None,
            "full_book_resolved_fills": summary["full_book_resolved_fills"],
            "proxy_resolved_fills": summary["proxy_resolved_fills"],
            "expected_net_t60_usd": summary["pnl_decomposition"]["expected_net_t60_usd"] if summary["pnl_decomposition"] else None,
            "expected_net_t59_usd": summary["pnl_decomposition"]["expected_net_t59_usd"] if summary["pnl_decomposition"] else None,
            "net_ev_change_t60_to_t59_usd": (
                summary["pnl_decomposition"]["expected_net_t59_usd"]
                - summary["pnl_decomposition"]["expected_net_t60_usd"]
            ) if summary["pnl_decomposition"] else None,
            "execution_t60_to_t59_usd": summary["pnl_decomposition"]["execution_t60_to_t59_usd"] if summary["pnl_decomposition"] else None,
            "prediction_payout_residual_usd": summary["pnl_decomposition"]["prediction_payout_residual_usd"] if summary["pnl_decomposition"] else None,
            "realized_fee_effect_usd": summary["pnl_decomposition"]["realized_fee_effect_usd"] if summary["pnl_decomposition"] else None,
            "realized_net_pnl_usd": summary["pnl_after_fee_usd"],
            "expected_minus_realized_net_usd": (
                summary["pnl_decomposition"]["expected_net_t59_usd"] - summary["pnl_after_fee_usd"]
            ) if summary["pnl_decomposition"] else None,
            "expected_minus_realized_net_per_fill_usd": summary["block_bootstrap"].get(
                "expected_minus_realized_net_pnl_per_fill_point"
            ),
            "expected_minus_realized_net_per_fill_95pct": summary["block_bootstrap"].get(
                "expected_minus_realized_net_pnl_per_fill_95pct"
            ),
            "block_bootstrap": json.dumps(summary["block_bootstrap"], sort_keys=True, default=str),
        })
    economic_groups = _group_economics(holdout_ledgers, price_edges, ev_edges)

    policy = json.loads(nested.CONFIG_PATH.read_text(encoding="utf-8"))
    holdout_economic_detail = nested._economic_detail(
        holdout_economic, HOLDOUT_MODEL_NAMES, research._snapshot_map(t59), research._snapshot_map(t60),
        _market_map_for_frame(holdout_economic), "holdout_diagnostic", HOLDOUT_QUALITY_COLUMNS,
    )
    holdout_economics, _, monthly_economics = nested._policy_summaries(
        holdout_economic_detail, holdout_economic, (FROZEN_HURDLE,), policy,
        model_names=HOLDOUT_MODEL_NAMES,
    )
    funnel_by_forecast = {row["forecast"]: row for row in funnel_rows}
    for row in holdout_economics:
        funnel = funnel_by_forecast[row["forecast"]]
        row["mean_predicted_chosen_side_probability"] = funnel["mean_predicted_chosen_side_probability"]
        row["observed_win_rate"] = funnel["observed_win_rate"]
        turnover = float(row["gross_turnover_usd"])
        row["net_pnl_over_turnover_pct"] = (
            100.0 * float(row["realized_net_pnl_usd"]) / turnover if turnover > 0.0 else 0.0
        )
    previous_ledger_comparison, previous_ledger_mismatches = _compare_previous_a_ledger(holdout_economic_detail)
    previous = pd.read_csv(model_update.PREVIOUS_ECONOMICS_PATH)
    previous_row_frame = previous.loc[
        previous.period.eq("holdout_diagnostic")
        & previous.forecast.eq("A_BTC_calibrated")
        & np.isclose(previous.ev_hurdle_fraction.astype(float), FROZEN_HURDLE)
    ]
    if len(previous_row_frame) != 1:
        raise RuntimeError("Expected exactly one previous A/EV 1% independent-order result")
    old_row = previous_row_frame.iloc[0].to_dict()
    new_a = next(row for row in holdout_economics if row["forecast"] == "A_frozen")
    comparison_fields = (
        "signals", "fills", "active_utc_days", "gross_turnover_usd", "fee_equivalent_usd",
        "expected_net_t59_usd", "realized_net_pnl_usd",
    )
    previous_comparison = {
        key: {
            "previous": float(old_row[key]),
            "recomputed": float(new_a[key]),
            "difference": float(new_a[key]) - float(old_row[key]),
        }
        for key in comparison_fields
    }
    previous_comparison["status"] = "exact_match" if all(
        abs(value["difference"]) <= (1e-7 if key not in ("signals", "fills", "active_utc_days") else 0.0)
        for key, value in previous_comparison.items() if key != "status"
    ) else "mismatch_to_explain"

    portfolio_source_identity = {
        "configuration_sha256": frozen_identity["configuration_sha256"],
        "outer_manifest_sha256": frozen_identity["outer_manifest_sha256"],
        "development_freeze_sha256": frozen_identity["development_freeze_sha256"],
        "t60_snapshot_sha256": book_record["t60_coverage_sha256"],
        "t59_snapshot_sha256": book_record["t59_coverage_sha256"],
        "baseline_prediction_sha256": baseline_all_identity["prediction_sha256"],
        "challenger_holdout_prediction_sha256": manifest["folds"][HOLDOUT_FOLD]["A_reoptimized"]["prediction_sha256"],
        "policy_hurdle_fraction": FROZEN_HURDLE,
    }
    portfolios = _portfolio_rows(holdout_all, t59, t60, portfolio_source_identity)
    portfolio_monthly_rows = []
    for row in portfolios:
        for month in json.loads(row["monthly_diagnostics"]):
            portfolio_monthly_rows.append({
                "forecast": row["forecast"],
                "starting_capital_usd": row["starting_capital_usd"],
                **month,
            })
    feature_rows = _feature_selection_rows(manifest)
    trial_rows, test_candidates, trial_summary = _trial_diagnostics(manifest)

    summary = {
        "experiment_id": model_update.CONFIG["experiment_id"],
        "status": "partial_report_completed_b_incomplete",
        "selection_status": freeze["status"],
        "selected_model": freeze["selected_model"],
        "development_common_rows": len(dev),
        "holdout_prediction_rows_common": len(holdout_quality),
        "holdout_economic_common_rows": len(holdout_economic),
        "holdout_variants_available": sorted(holdout_candidates),
        "holdout_variant_missing": "B_dynamic_price_volume",
        "quality_development": development_quality,
        "quality_development_by_fold": development_fold_quality,
        "quality_holdout": holdout_quality_rows,
        "holdout_paired_differences": holdout_paired,
        "holdout_economics": holdout_economics,
        "holdout_funnel": funnel_rows,
        "holdout_monthly_economics": monthly_economics,
        "holdout_economic_detail_path": (LEDGER_DIR / "independent_order_detail.parquet").resolve().relative_to(ROOT).as_posix(),
        "holdout_decision_ledger_path": (LEDGER_DIR / "holdout_decisions.parquet").resolve().relative_to(ROOT).as_posix(),
        "economic_group_rows": economic_groups,
        "development_ev_bin_edges": ev_edges,
        "development_t60_price_bin_edges": price_edges,
        "previous_economics_comparison": previous_comparison,
        "previous_a_ledger_comparison": previous_ledger_comparison,
        "holdout_portfolios": portfolios,
        "holdout_portfolio_monthly": portfolio_monthly_rows,
        "feature_selection": feature_rows,
        "b_tuning_diagnostics": trial_summary,
        "b_pruned_configs_proposed_for_development_test": test_candidates,
        "snapshot_coverage": {
            "t60_markets": int(len(t60)),
            "t60_sha256": book_record["t60_coverage_sha256"],
            "t59_markets": int(len(t59)),
            "t59_sha256": book_record["t59_coverage_sha256"],
            "event_replay_performed": False,
        },
        "frozen_artifact_identity": frozen_identity,
    }
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    holdout_economic_detail.to_parquet(LEDGER_DIR / "independent_order_detail.parquet", index=False, compression="zstd")
    holdout_ledgers.to_parquet(LEDGER_DIR / "holdout_decisions.parquet", index=False, compression="zstd")
    tables = {
        "probability_quality_development.csv": development_quality,
        "probability_quality_development_by_fold.csv": development_fold_quality,
        "probability_quality_holdout.csv": holdout_quality_rows,
        "holdout_paired_differences.json": holdout_paired,
        "feature_selection_audit.csv": feature_rows,
        "holdout_fixed_5_economics.csv": holdout_economics,
        "holdout_monthly_economics.csv": monthly_economics,
        "holdout_funnel.csv": funnel_rows,
        "holdout_economics_by_side_price_ev_month.csv": economic_groups,
        "b_holdout_trials.csv": trial_rows,
        "b_pruned_configs_for_development_test.csv": test_candidates,
        "holdout_portfolios.csv": portfolios,
        "holdout_portfolio_monthly.csv": portfolio_monthly_rows,
        "previous_a_ledger_mismatches.csv": previous_ledger_mismatches,
    }
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        path = TABLE_DIR / name
        if name.endswith(".json"):
            _write_json(path, rows)
        elif isinstance(rows, pd.DataFrame):
            rows.to_csv(path, index=False)
        else:
            pd.DataFrame(rows).to_csv(path, index=False)
    _write_json(LEDGER_DIR / "partial_report_summary.json", summary)
    provenance = {
        "report_schema_version": 1,
        "report_status": "partial_report_completed_b_incomplete",
        "generated_from_existing_artifacts": True,
        "training_or_archive_replay_performed": False,
        "frozen_input_identity": frozen_identity,
        "report_code_sha256": {
            name: _sha256(ROOT / name) for name in (
                "build_btc_preopen_model_update_report.py",
                "run_btc_preopen_experiment.py",
                "run_btc_preopen_model_update.py",
                "run_btc_preopen_signal_diagnostic.py",
                "run_btc_market_probability_nested_experiment.py",
                "run_btc_preopen_research.py",
                "run_btc_preopen_t59_reassessment.py",
                "run_btc_preopen_fee_correction.py",
                "select_features.py",
            )
        },
        "dependency_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "optuna": optuna.__version__,
        },
        "report_path": REPORT_PATH.resolve().relative_to(ROOT).as_posix(),
        "report_sha256": None,
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write_report(summary, freeze, previous_comparison, previous_ledger_comparison, test_candidates, trial_summary, feature_rows, holdout_paired, holdout_economics, funnel_rows, monthly_economics, portfolios, portfolio_monthly_rows, economic_groups)
    provenance["report_sha256"] = _sha256(REPORT_PATH)
    _write_json(LEDGER_DIR / "report_provenance.json", provenance)
    return summary


def _write_report(
    summary: dict,
    freeze: dict,
    previous_comparison: dict,
    previous_ledger_comparison: dict,
    test_candidates: list[dict],
    trial_summary: dict,
    feature_rows: list[dict],
    holdout_paired: dict,
    holdout_economics: list[dict],
    funnel_rows: list[dict],
    monthly_economics: list[dict],
    portfolios: list[dict],
    portfolio_monthly_rows: list[dict],
    economic_groups: list[dict],
) -> None:
    dev_quality = summary["quality_development"]
    fold_quality = summary["quality_development_by_fold"]
    hold_quality = summary["quality_holdout"]
    comparisons = freeze["paired_comparisons_against_frozen_A"]
    frozen_funnel = next(row for row in funnel_rows if row["forecast"] == "A_frozen")
    frozen_bootstrap = json.loads(frozen_funnel["block_bootstrap"])
    expected_realized_interval = frozen_bootstrap["expected_minus_realized_net_pnl_per_fill_95pct"]
    predicted_probability = frozen_funnel["mean_predicted_chosen_side_probability"]
    observed_win_rate = frozen_funnel["observed_win_rate"]
    expected_realized_gap = frozen_funnel["expected_minus_realized_net_usd"]
    expected_realized_gap_per_fill = frozen_funnel["expected_minus_realized_net_per_fill_usd"]
    net_ev_change = frozen_funnel["net_ev_change_t60_to_t59_usd"]
    lines = [
        "# BTC pre-open model update: partial report",
        "",
        "> This report was built from hash-verified predictions, model artifacts, and corrected T-60/T-59 snapshots already on disk. It did not run training, a model refit, an event replay, or live execution. The evaluated July–October period was previously inspected and is historical diagnostic evidence.",
        "",
        "## Status and frozen decision",
        "",
        f"- Development selected `{freeze['selected_model']}` on {freeze['development_rows_common']:,} common markets. `A_frozen` is the earlier reference; `A_reoptimized` is the new challenger; `B_dynamic_price_volume` adds causal price/volume candidates.",
        "- Frozen selection rule: " + freeze["selection_rule"]["candidate_gate"],
        "- The best new development point estimate had lower mean log loss than A_frozen, but neither candidate's adjusted interval cleared zero. Keeping A_frozen is the pre-registered uncertainty rule, not evidence that all new models have worse mean metrics.",
        "- B holdout is incomplete: its model tuning ended at 4/5 required successful trials. No B holdout prediction or score is reported.",
        "- The frozen comparison includes A_frozen and A_reoptimized on the same holdout prediction markets; quality metrics use official labels and Binance proxy labels separately.",
        "",
        "### Development quality on the frozen common sample",
        "",
        *_format_table(dev_quality, [("Forecast", "forecast"), ("Target", "target"), ("n", "n"), ("Log loss", "log_loss"), ("Brier", "brier"), ("AUC", "auc")]),
        "",
        "### Development fold metrics",
        "",
        *_format_table(fold_quality, [("Fold", "period"), ("Forecast", "forecast"), ("Target", "target"), ("n", "n"), ("Log loss", "log_loss"), ("Brier", "brier"), ("AUC", "auc")]),
        "",
        "### Paired development comparisons against A_frozen",
        "",
        *_format_table([
            {"forecast": name, **row}
            for name, row in comparisons.items()
        ], [("Candidate", "forecast"), ("n", "n"), ("Delta log loss", "log_loss_difference_left_minus_right"), ("Bonferroni-adjusted 97.5% interval", "log_loss_difference_adjusted_interval")]),
        "",
        "The intervals are the frozen paired circular moving 3-UTC-day bootstrap (2,000 replicates, seed 20261012), with the Bonferroni-adjusted 97.5% upper-bound gate. A candidate would replace A_frozen only if its adjusted upper delta-log-loss bound were below zero.",
        "",
        "### Feature selection audit",
        "",
        *_format_table(feature_rows, [("Fold", "fold_id"), ("Variant", "variant"), ("Best k", "best_k"), ("Selector recommendation", "selector_recommended_features"), ("Model features used", "model_features_used"), ("Added B candidates", "added_candidate_features"), ("Savings", "savings_from_best_k"), ("Required", "minimum_required_savings"), ("Gate", "savings_gate_met")]),
        "",
        "In each development fold, best k equalled recommended k, so the 20-feature plateau-savings requirement was not met. B then added candidate features that were not already selected. The selector was run once per fold and reused by both variants.",
        "",
        "### Holdout probability quality",
        "",
        *_format_table(hold_quality, [("Forecast", "forecast"), ("Target", "target"), ("n", "n"), ("Log loss", "log_loss"), ("Brier", "brier"), ("AUC", "auc")]),
        "",
        *_format_table([
            {"target": "official Polymarket outcome", **holdout_paired["official_outcome_up"]},
            {"target": "Binance proxy", **holdout_paired["target_binance_proxy_up"]},
        ], [("Target", "target"), ("n", "n"), ("Delta log loss: A_reoptimized minus A_frozen", "log_loss_difference_left_minus_right"), ("95% block CI", "log_loss_difference_95pct"), ("Delta Brier", "brier_difference_left_minus_right")]),
        "",
        "Holdout intervals use the same paired circular 3-day UTC block structure and 2,000 replicates as the development procedure; they are descriptive because this period was historically exposed.",
        "",
        "## Economic evaluation at the frozen policy",
        "",
        "Policy remains 1% net EV at T-60, same side executed at T-59, independent gross order size USD 5, with the accepted fee schedule, depth, partial-fill, and settlement accounting. No T-59 observation selects the side or changes the requested amount. The separate portfolios use 5% of free cash at T-60 and lock stake until settlement.",
        f"- Locally verified corrected snapshots: {summary['snapshot_coverage']['t60_markets']:,} T-60 and {summary['snapshot_coverage']['t59_markets']:,} T-59 markets; hashes {summary['snapshot_coverage']['t60_sha256']} and {summary['snapshot_coverage']['t59_sha256']}. Training's prediction-quality evaluator does not run economic replay; the snapshots do exist and were reused here. Missing quote coverage is reported per market in the funnel.",
        f"- The earlier A/EV 1% result comparison status is `{previous_comparison['status']}`. The exact row-by-row values are in `holdout_fixed_5_economics.csv`.",
        f"- Previous market-level ledger comparison: `{previous_ledger_comparison['status']}`; previous/recomputed qualified signals {previous_ledger_comparison.get('previous_signals_at_hurdle', 'n/a')}/{previous_ledger_comparison.get('recomputed_signals_at_hurdle', 'n/a')}, previous-only/recomputed-only market IDs {previous_ledger_comparison.get('previous_only_condition_ids', 'n/a')}/{previous_ledger_comparison.get('recomputed_only_condition_ids', 'n/a')}, changed common ledgers {previous_ledger_comparison.get('common_condition_ids_with_changed_ledger_fields', 'n/a')}. Any differing market rows are listed in `previous_a_ledger_mismatches.csv` before interpreting challenger results.",
        "",
        *_format_table(holdout_economics, [("Forecast", "forecast"), ("Opportunities", "opportunity_markets"), ("Signals", "signals"), ("Fills", "fills"), ("Mean forecast P(win)", "mean_predicted_chosen_side_probability"), ("Observed win rate", "observed_win_rate"), ("Turnover USD", "gross_turnover_usd"), ("Fee equivalent USD", "fee_equivalent_usd"), ("Expected net T-59 USD", "expected_net_t59_usd"), ("Realized net PnL USD", "realized_net_pnl_usd"), ("PnL/turnover %", "net_pnl_over_turnover_pct"), ("Wins", "positive_fills")]),
        "",
        "### Funnel and EV reconciliation",
        "",
        *_format_table(funnel_rows, [("Forecast", "forecast"), ("Confirmed markets", "market_count"), ("Predictions", "prediction_count"), ("T-60 priceable", "t60_priceable_decisions"), ("1% signals", "signals_at_1pct_t60_ev"), ("T-59 fills", "t59_fills"), ("Resolved", "resolved_fills"), ("Wins", "wins"), ("Losses", "losses"), ("Mean forecast P(win)", "mean_predicted_chosen_side_probability"), ("Observed win rate", "observed_win_rate"), ("Expected net T-60 USD", "expected_net_t60_usd"), ("Expected net T-59 USD", "expected_net_t59_usd"), ("T-60 to T-59 net EV change USD", "net_ev_change_t60_to_t59_usd"), ("Gross repricing USD", "execution_t60_to_t59_usd"), ("Payout residual USD", "prediction_payout_residual_usd"), ("Realized fee effect USD", "realized_fee_effect_usd"), ("Expected minus realized net USD", "expected_minus_realized_net_usd"), ("Expected minus realized USD/fill", "expected_minus_realized_net_per_fill_usd"), ("95% block CI USD/fill", "expected_minus_realized_net_per_fill_95pct"), ("Realized net PnL USD", "realized_net_pnl_usd"), ("Exclusions", "non_execution_reasons")]),
        "",
        f"For A_frozen, the 1% hurdle was a model-based forecast filter: mean chosen-side P(win) on resolved fills was {predicted_probability:.2%} versus a {observed_win_rate:.2%} realized win rate. Expected net at T-60 was USD {frozen_funnel['expected_net_t60_usd']:,.2f}; T-59 repricing changed it by USD {net_ev_change:,.2f} to USD {frozen_funnel['expected_net_t59_usd']:,.2f}. Realized net PnL was USD {frozen_funnel['realized_net_pnl_usd']:,.2f}, leaving an expected-minus-realized gap of USD {expected_realized_gap:,.2f} (USD {expected_realized_gap_per_fill:.3f}/fill; 95% moving-block interval USD {expected_realized_interval[0]:.3f} to {expected_realized_interval[1]:.3f}/fill). This is evidence of an outcome/forecast gap among the selected fills in this historical period; it does not identify one cause. The bridge separates decision-time EV from diagnostic T-59 expected EV, and the group ledger breaks results out by side, development-frozen T-60 price/EV bins, and month, including full-depth/proxy shares.",
        "",
        "### Monthly independent-order economics",
        "",
        *_format_table(monthly_economics, [("Forecast", "forecast"), ("Month UTC", "month_utc"), ("Signals", "signals"), ("Fills", "fills"), ("Turnover USD", "gross_turnover_usd"), ("Fees USD", "fee_equivalent_usd"), ("Expected net T-59 USD", "expected_net_t59_usd"), ("Realized net PnL USD", "realized_net_pnl_usd")]),
        "",
        "Development-frozen group boundaries and all grouped rows are in `holdout_economics_by_side_price_ev_month.csv`.",
        "",
        "### Separate continuous portfolios",
        "",
        *_format_table(portfolios, [("Forecast", "forecast"), ("Starting cash", "starting_capital_usd"), ("Trades", "trade_count"), ("Turnover USD", "gross_turnover_usd"), ("Fees USD", "fees_paid_estimated_usd"), ("Net PnL USD", "net_pnl_usd"), ("Ending cash", "ending_cash_after_settlement_usd"), ("Max drawdown", "max_drawdown_cost_basis_equity"), ("Audit", "accounting_audit_status")]),
        "",
        "Each account starts independently at USD 100/500/1,000. Its detailed monthly diagnostics, decisions, trades, equity path, and accounting audit are in the ignored local ledger directory.",
        "",
        "### USD 100 portfolio rejection and stop audit",
        "",
        *_format_table([row for row in portfolios if float(row["starting_capital_usd"]) == 100.0], [("Forecast", "forecast"), ("Minimum-order skips", "below_minimum_order_rejections"), ("Locked-cash skips", "locked_cash_rejections"), ("Other skips", "other_portfolio_rejections"), ("Last trade entry UTC", "last_trade_entry_time_utc"), ("Last decision UTC", "last_decision_time_utc"), ("Free cash USD", "free_cash_at_last_decision_usd"), ("Cost-basis equity USD", "equity_at_last_decision_usd"), ("5% stake USD", "five_percent_stake_at_last_decision_usd"), ("$5 minimum check", "minimum_skip_ledger_check")]),
        "",
        "Both USD 100 ledgers show that every below-minimum skip requested exactly 5% of pre-decision free cash and that each requested amount was below the accepted USD 5 fallback; there were no locked-cash skips. A_frozen's 1,630 other skips were 1,625 unpriceable native asks and 5 no-positive-net-EV decisions. A_reoptimized's 1,628 were 1,625 unpriceable asks, 2 no-positive-net-EV decisions, and 1 below-minimum-net-EV decision. Historical Gamma order minimum metadata is unavailable, so the USD 5 fallback is the auditable threshold. The small portfolio losses reflect a few trades followed by stakes below minimum, not evidence of strategy robustness.",
        "",
        *_format_table(portfolio_monthly_rows, [("Forecast", "forecast"), ("Starting cash", "starting_capital_usd"), ("Month UTC", "month_utc"), ("Trades", "trade_count"), ("Turnover USD", "gross_turnover_usd"), ("Fees USD", "fee_usd"), ("Monthly equity PnL USD", "monthly_equity_pnl_usd"), ("Closing equity USD", "closing_cost_basis_equity_usd")]),
        "",
        "## B pruning and tuning budget",
        "",
        f"- Holdout B study: {trial_summary['counts'].get('COMPLETE', 0)} COMPLETE, {trial_summary['counts'].get('PRUNED', 0)} PRUNED, {trial_summary['counts'].get('FAIL', 0)} FAIL. The saved continuation preflight had {trial_summary['continuation_preflight_trial_count_from_saved_helper']} trials and {trial_summary['continuation_preflight_success_count_from_saved_helper']} successes; the final study has {trial_summary['trial_count']} trials and 4/5 required successes. The latest tuning stage used {trial_summary['failed_stage_elapsed_seconds_last_attempt']:.1f}s and stopped at the frozen tuning deadline.",
        f"- Median observed duration was {trial_summary['median_complete_trial_seconds']:.1f}s per completed trial and {trial_summary['median_pruned_trial_seconds']:.1f}s per pruned trial. The first pruning checkpoint is iteration 25 (Optuna step 24); trial curves, all parameters, durations, best iterations, and checkpoint values are in `b_holdout_trials.csv`.",
        f"- A high pruned share alone does not prove harmful pruning. {trial_summary['completed_trials_best_after_first_checkpoint']} of {trial_summary['counts'].get('COMPLETE', 0)} completed trials recorded their best iteration after the first pruning checkpoint (iteration 25), so later gains are possible. Pruned curves end at the pruning checkpoint; their counterfactual later values are unknown. This is consistent with a possible early-pruning risk, not proof that a slower configuration was wrongly discarded.",
        f"- Budget reconstruction: the saved resume log shows {trial_summary['resume_remaining_seconds_from_log']:.1f}s remaining from the 2,430s tuning allowance, implying about {trial_summary['prior_attempt_seconds_reconstructed_from_log']:.1f}s in the first B attempt. The resumed stage manifest records {trial_summary['latest_stage_wall_seconds']:.3f}s wall time, so cumulative B-stage wall time is about {trial_summary['cumulative_b_stage_wall_seconds_estimate']:.1f}s (roughly {trial_summary['resume_overhead_after_budget_deadline_estimate_seconds']:.1f}s beyond the resumed tuning clock, with ±{trial_summary['historical_budget_estimate_uncertainty_seconds']:.2f}s rounding uncertainty). This separates tuning from resume/cleanup overhead; B final fit was not reached (0s). The 2,430s tuning allowance was exhausted before 5/5 successes; do not interpret the broader 2,700s tuning-plus-fit envelope as extra tuning budget.",
        "- The frozen runner requires at least 5 successful trials before final fit. If the budget ends after that minimum but before the requested 10 total trials, the current acceptance gate can use the best successful trial while recording `search_complete=false`; these are distinct outcomes. This B study stopped at 4/5, so it correctly produced no final model or holdout score.",
        "- The separately authorized development-only continuation diagnostic completed 15 fixed configurations across the three development B folds, without TPE suggestions, feature selection, generator optimization, or a holdout score. It records full available validation curves, both source and continued checkpoint values, and same-fold COMPLETE-study comparisons. No hypothetical MedianPruner decision or false-prune rate is claimed; see `reports/btc_preopen/b_pruner_development_diagnostic_20261014/pruner_diagnostic_report.md`. The diagnostic did not change model or policy selection.",
        "- Six of 15 continuations found their best iteration after the corresponding source trial's prune checkpoint; only one of those six beat its same-fold COMPLETE-study best. Development fold C / Q1 (source trial 57) reached 0.692240 at iteration 73, 0.000031 below the best COMPLETE objective. This is limited evidence that a configuration from a historically PRUNED final-fit-B trial can be competitive on a development fold. It is not evidence that a specific development-fold MedianPruner decision was wrong: the source decision came from another chronological fit, and no hypothetical pruning was replayed. The other 14 continuations did not beat their same-fold best.",
        "",
        *_format_table(test_candidates, [("Config", "test_config_id"), ("Source trial", "source_trial_number"), ("Pruned at iteration", "source_pruned_at_iteration"), ("Selection rule", "selection_rule"), ("Frozen params JSON", "params")]),
        "",
        "## Artifacts and reproduction",
        "",
        f"- Report tables: `{TABLE_DIR.resolve().relative_to(ROOT).as_posix()}`.",
        f"- Detailed decision and portfolio ledgers: `{LEDGER_DIR.resolve().relative_to(ROOT).as_posix()}` (local research data).",
        "- Report provenance is recorded separately at `partial_report_ledgers/report_provenance.json`; the frozen config, original model-update manifest, and development freeze were read-only inputs.",
        "- Rebuild only this report, without training or command-line arguments: `python build_btc_preopen_model_update_report.py`.",
        "- Live remains off. No deployment refit, order, or policy change was made.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = build_report()
    print(
        f"[btc-model-update-report] status={result['status']} "
        f"development={result['development_common_rows']:,} "
        f"holdout={result['holdout_prediction_rows_common']:,} B=INCOMPLETE",
        flush=True,
    )
