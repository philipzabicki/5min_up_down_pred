"""Explain the frozen BTC conditional market-offset experiment without refitting."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit

import run_btc_conditional_market_offset_20261018 as experiment
import run_btc_preopen_economic_replay as economic_replay
import run_btc_preopen_research as research
import run_btc_preopen_t59_reassessment as replay


ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROOT / "data/analysis/polymarket/BTC/btc_conditional_market_offset_20261018"
DIAGNOSTIC_DIR = EXPERIMENT_DIR / "diagnostics"
TABLE_DIR = ROOT / "reports/btc_preopen/btc_conditional_market_offset_20261018_diagnostics"
REPORT_PATH = ROOT / "reports/btc_preopen/btc_conditional_market_offset_20261018_execution_diagnostic.md"
CONFIG_PATH = ROOT / "configs/research/btc_conditional_market_offset_20261018.json"
QUALITY_CUTOFF = pd.Timestamp("2026-10-07T00:00:00Z")
FIXED_GROSS_USD = 5.0
MINIMUM_NET_EV_FRACTION = 0.01
FEATURES = experiment.FEATURES
SCENARIO_ORDER = (
    "normalized_p_market_reference",
    "native_midpoint",
    "t60_best_ask",
    "t60_executable_depth",
    "t59_same_intent",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_rows_hash(frames: list[pd.DataFrame]) -> str:
    columns = ["condition_id", "official_outcome_up", "resolved_at_utc", "label_available_at_utc"]
    combined = pd.concat([frame.loc[:, columns] for frame in frames], ignore_index=True)
    combined.sort_values("condition_id", kind="stable", inplace=True)
    records = []
    for row in combined.itertuples(index=False, name=None):
        condition_id, outcome, resolved, available = row
        records.append([
            str(condition_id).lower(), int(outcome),
            pd.Timestamp(resolved).isoformat(), pd.Timestamp(available).isoformat(),
        ])
    payload = json.dumps(records, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_saved_labels(frames: list[pd.DataFrame], official_index_path: Path) -> int:
    official = pd.read_parquet(
        official_index_path,
        columns=["condition_id", "market_start_utc", "resolved_at_utc", "target_polymarket_up"],
    )
    official["condition_id"] = official.condition_id.astype(str).str.lower()
    observed = pd.concat([
        frame[["condition_id", "market_start_utc", "resolved_at_utc", "label_available_at_utc", "official_outcome_up"]]
        for frame in frames
    ], ignore_index=True)
    observed["condition_id"] = observed.condition_id.astype(str).str.lower()
    if observed.condition_id.duplicated().any():
        raise RuntimeError("Saved prediction files contain duplicate official label IDs")
    joined = observed.merge(official, on="condition_id", how="left", validate="one_to_one", suffixes=("_saved", "_official"))
    if joined.target_polymarket_up.isna().any():
        raise RuntimeError("A saved official outcome is absent from the hash-verified archive market index")
    saved_start = pd.to_datetime(joined.market_start_utc_saved, utc=True)
    official_start = pd.to_datetime(joined.market_start_utc_official, utc=True)
    saved_resolved = pd.to_datetime(joined.resolved_at_utc_saved, utc=True)
    official_resolved = pd.to_datetime(joined.resolved_at_utc_official, utc=True)
    saved_release = pd.to_datetime(joined.label_available_at_utc, utc=True)
    release_rule = pd.concat([
        saved_resolved.rename("resolved"), (saved_start + pd.Timedelta(minutes=5)).rename("five_minutes"),
    ], axis=1).max(axis=1) + pd.Timedelta(seconds=60)
    outcomes_match = joined.official_outcome_up.to_numpy(dtype=np.int8) == joined.target_polymarket_up.to_numpy(dtype=np.int8)
    resolutions_match = saved_resolved.eq(official_resolved) | (saved_resolved.isna() & official_resolved.isna())
    if not outcomes_match.all() or not saved_start.eq(official_start).all() or not resolutions_match.all():
        raise RuntimeError("Saved labels, resolutions, or market starts differ from the hash-verified official index")
    if not saved_release.eq(release_rule).all():
        raise RuntimeError("A saved label-availability time differs from the frozen release-time rule")
    return int(len(joined))


def _load_inputs() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    freeze = json.loads((EXPERIMENT_DIR / "frozen_development_bundle.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((EXPERIMENT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    results = json.loads((EXPERIMENT_DIR / "final_results.json").read_text(encoding="utf-8"))
    if run_manifest.get("status") != "completed" or not run_manifest.get("holdout_was_scored_once"):
        raise RuntimeError("The frozen experiment is not in its completed, once-scored state")

    source_manifest = research._read_json(research.RESEARCH_MANIFEST_PATH)
    book_record = source_manifest["stages"]["causal_t60_t59_book_replay"]
    t59_path = ROOT / book_record["t59_coverage_path"]
    t60_path = ROOT / book_record["t60_coverage_path"]
    dev_path = EXPERIMENT_DIR / "development_cv_predictions.parquet"
    holdout_path = EXPERIMENT_DIR / "holdout_predictions.parquet"
    model_path = ROOT / freeze["final_tree_path"]
    label_path = research.ARCHIVE_INDEX_PATH
    original_report = ROOT / "reports/btc_preopen/btc_conditional_market_offset_20261018.md"
    old_decisions_path = EXPERIMENT_DIR / "economics/D/portfolio_decisions.parquet"
    inputs = {
        "development_cv_predictions": (dev_path, freeze["selected_development_cv_predictions_sha256"]),
        "holdout_predictions": (holdout_path, run_manifest["holdout_prediction_sha256"]),
        "conditional_model": (model_path, freeze["final_tree_sha256"]),
        "t59_snapshots": (t59_path, book_record["t59_coverage_sha256"]),
        "t60_snapshots": (t60_path, book_record["t60_coverage_sha256"]),
        "official_market_labels": (label_path, config["source_artifacts"]["official_market_index_sha256"]),
    }
    source_hashes = {}
    for name, (path, expected) in inputs.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen {name} hash mismatch: expected {expected}, got {actual}")
        source_hashes[name] = {
            "path": path.resolve().relative_to(ROOT).as_posix(),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    if source_manifest["input_identity"]["official_markets_sha256"] != source_hashes["official_market_labels"]["sha256"]:
        raise RuntimeError("The official label source differs from the research manifest")
    if book_record["t59_coverage_sha256"] != config["source_artifacts"]["t59_snapshot_sha256"]:
        raise RuntimeError("The T-59 snapshots differ from the frozen experiment config")
    if book_record["t60_coverage_sha256"] != config["source_artifacts"]["t60_snapshot_sha256"]:
        raise RuntimeError("The T-60 snapshots differ from the frozen experiment config")

    dev = pd.read_parquet(dev_path)
    holdout = pd.read_parquet(holdout_path)
    t59, t60, verified_book_record = research._load_enriched_book_snapshots(source_manifest)
    if verified_book_record["t59_coverage_sha256"] != book_record["t59_coverage_sha256"]:
        raise RuntimeError("The loaded T-59 book changed during diagnostic setup")
    if verified_book_record["t60_coverage_sha256"] != book_record["t60_coverage_sha256"]:
        raise RuntimeError("The loaded T-60 book changed during diagnostic setup")
    old_decisions = pd.read_parquet(
        old_decisions_path,
        columns=["condition_id", "requested_gross_usd", "chosen_side", "skip_reason", "cash_before_usd"],
    )

    booster = lgb.Booster(model_file=str(model_path))
    if booster.num_trees() != 43 or freeze["final_tree_count"] != 43:
        raise RuntimeError(f"Frozen D tree count changed: booster={booster.num_trees()}")
    if tuple(booster.feature_name()) != tuple(FEATURES):
        raise RuntimeError("Frozen D feature order differs from the experiment runner")
    # Development values are fold-specific OOF predictions; the frozen final
    # model is expected to reproduce only the saved holdout predictions.
    for period, frame in (("holdout", holdout),):
        features = frame.loc[:, FEATURES].to_numpy(dtype=np.float64)
        raw = np.asarray(booster.predict(features, raw_score=True, num_threads=4), dtype=np.float64)
        if not np.allclose(raw, frame.d_margin.to_numpy(dtype=np.float64), rtol=1e-10, atol=1e-11):
            raise RuntimeError(f"Frozen D model does not reproduce saved {period} margins")
        reproduced = expit(frame.logit_p_market.to_numpy(dtype=np.float64) + raw)
        if not np.allclose(reproduced, frame.p_D_conditional.to_numpy(dtype=np.float64), rtol=1e-10, atol=1e-11):
            raise RuntimeError(f"Frozen D model does not reproduce saved {period} probabilities")

    all_predictions = pd.concat([dev, holdout], ignore_index=True, sort=False)
    enriched, _ = replay._market_statuses(all_predictions, t59, t60)
    market_by_id = research._snapshot_map(enriched)
    t59_by_id = research._snapshot_map(t59)
    t60_by_id = research._snapshot_map(t60)
    if old_decisions.condition_id.astype(str).str.lower().duplicated().any():
        raise RuntimeError("Frozen D portfolio decisions contain duplicate condition IDs")
    old_decisions["condition_id"] = old_decisions.condition_id.astype(str).str.lower()

    source_hashes["frozen_protocol"] = {
        "path": CONFIG_PATH.relative_to(ROOT).as_posix(), "sha256": _sha256(CONFIG_PATH),
    }
    source_hashes["frozen_runner"] = {
        "path": (ROOT / "run_btc_conditional_market_offset_20261018.py").relative_to(ROOT).as_posix(),
        "sha256": _sha256(ROOT / "run_btc_conditional_market_offset_20261018.py"),
    }
    source_hashes["historical_experiment_report"] = {
        "path": original_report.relative_to(ROOT).as_posix(), "sha256": _sha256(original_report),
    }
    source_hashes["observed_label_rows"] = {
        "path": "saved development OOF and holdout prediction artifacts",
        "sha256": _label_rows_hash([dev, holdout]),
        "rows": int(len(dev) + len(holdout)),
        "matched_to_official_index_rows": _verify_saved_labels([dev, holdout], label_path),
    }
    return {
        "config": config, "freeze": freeze, "run_manifest": run_manifest, "results": results,
        "dev": dev, "holdout": holdout, "t59": t59, "t60": t60,
        "t59_by_id": t59_by_id, "t60_by_id": t60_by_id,
        "market_by_id": market_by_id,
        "old_decisions": old_decisions, "booster": booster, "source_hashes": source_hashes,
    }


def _finite_or_none(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fill_metrics(fill: dict | None, probability: float, requested_gross: float = FIXED_GROSS_USD) -> dict:
    fill = fill or {}
    gross_usd = _finite_or_none(fill.get("gross_usd"))
    gross_shares = _finite_or_none(fill.get("gross_shares"))
    net_shares = _finite_or_none(fill.get("shares"))
    cash_debit = _finite_or_none(fill.get("cash_debit_usd"))
    fee_cash = _finite_or_none(fill.get("fee_cash_usd"))
    fee_shares = _finite_or_none(fill.get("fee_shares"))
    fee_equivalent = _finite_or_none(fill.get("fee_usd_entry_equivalent", fill.get("fee_usd")))
    if net_shares is None or cash_debit is None or gross_usd is None:
        gross_ev = net_ev = break_even = None
        fee_effect = None
    else:
        gross_ev = probability * gross_shares - gross_usd
        net_ev = probability * net_shares - cash_debit
        break_even = cash_debit / net_shares if net_shares > 0.0 else None
        fee_effect = net_ev - gross_ev
    return {
        "requested_gross_usd": float(requested_gross), "filled_gross_usd": gross_usd,
        "unfilled_gross_usd": _finite_or_none(fill.get("unfilled_gross_usd")),
        "gross_shares": gross_shares, "net_shares": net_shares,
        "vwap": _finite_or_none(fill.get("vwap")), "cash_debit_usd": cash_debit,
        "fee_cash_usd": fee_cash, "fee_shares": fee_shares, "fee_usd_equivalent": fee_equivalent,
        "gross_ev_usd": gross_ev, "fee_effect_usd": fee_effect, "net_ev_usd": net_ev,
        "net_ev_fraction_requested": net_ev / requested_gross if net_ev is not None else None,
        "net_ev_fraction_filled_cost": net_ev / cash_debit if net_ev is not None and cash_debit else None,
        "break_even_probability": break_even,
        "probability_minus_break_even": probability - break_even if break_even is not None else None,
        "ev_hurdle_gap_usd": max(0.0, requested_gross * MINIMUM_NET_EV_FRACTION - net_ev) if net_ev is not None else None,
        "ev_hurdle_gap_fraction": net_ev / requested_gross - MINIMUM_NET_EV_FRACTION if net_ev is not None else None,
        "fill_status": fill.get("fill_status"), "partial_fill": bool(fill.get("partial_fill", False)),
        "depth_sufficient": fill.get("depth_sufficient"), "fee_rule_id": fill.get("fee_rule_id"),
        "fee_rate": _finite_or_none(fill.get("fee_rate")),
        "fee_collection_mode": fill.get("fee_collection_mode"),
        "fee_charge_unit": fill.get("fee_charge_unit"),
        "execution_source": fill.get("execution_source"),
        "execution_assumption": fill.get("execution_assumption"),
        "depth_known": fill.get("depth_known"),
    }


def _fill_at_single_price(price: float | None, market: dict, requested_gross: float) -> dict | None:
    price = _finite_or_none(price)
    if price is None or not 0.0 < price < 1.0:
        return None
    entry_time = pd.Timestamp(market["market_start_utc"]) + pd.Timedelta(seconds=replay.T59_ENTRY_OFFSET_SECONDS)
    levels = {price: requested_gross / price + 1e-8}
    return economic_replay._walk_asks(
        levels, None, None, gross_order_usd=requested_gross,
        entry_time_utc=entry_time, allow_partial=True,
    )


def _fixed_intent_at_t59(market: dict, snapshot: dict | None, decision: dict) -> dict:
    """Price only the T-60-selected side and size; never reselect at T-59."""
    side = decision.get("chosen_side")
    requested = decision.get("requested_gross_usd")
    if side not in ("up", "down") or requested is None or snapshot is None:
        return {"side": side, "requested_gross_usd": requested, "priceable": False, "reason": "no_t60_intent_or_t59_snapshot", "fill": None}
    execution = replay._price_side(snapshot, market, side, float(requested))
    return {
        "side": side, "requested_gross_usd": float(requested),
        "priceable": bool(execution["priceable"]), "reason": execution["reason"],
        "fill": execution.get("fill"), "depth_known": execution.get("depth_known"),
    }


def _before_exclusive_cutoff(release_time, cutoff: pd.Timestamp = QUALITY_CUTOFF) -> bool:
    if release_time is None or pd.isna(release_time):
        return False
    return pd.Timestamp(release_time) < cutoff


def _waterfall_increment(previous: dict | None, current: dict) -> dict:
    if previous is None or current.get("net_ev_usd") is None or previous.get("net_ev_usd") is None:
        return {
            "delta_gross_ev_from_previous_usd": None,
            "delta_fee_effect_from_previous_usd": None,
            "delta_net_ev_from_previous_usd": None,
        }
    result = {
        "delta_gross_ev_from_previous_usd": current["gross_ev_usd"] - previous["gross_ev_usd"],
        "delta_fee_effect_from_previous_usd": current["fee_effect_usd"] - previous["fee_effect_usd"],
        "delta_net_ev_from_previous_usd": current["net_ev_usd"] - previous["net_ev_usd"],
    }
    if not math.isclose(
        result["delta_net_ev_from_previous_usd"],
        result["delta_gross_ev_from_previous_usd"] + result["delta_fee_effect_from_previous_usd"],
        rel_tol=0.0, abs_tol=2e-10,
    ):
        raise AssertionError("A price waterfall increment does not reconcile to its pre-fee and fee components")
    return result


def _center_and_group_stump_contributions(
    contributions: np.ndarray, centers: np.ndarray, feature_indices: list[int], feature_count: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    centered = contributions - centers[None, :]
    grouped = np.zeros((len(contributions), feature_count), dtype=np.float64)
    for tree_number, feature_index in enumerate(feature_indices):
        grouped[:, feature_index] += centered[:, tree_number]
    return float(centers.sum()), centered, grouped


def _snapshot_side_prices(snapshot: dict | None, side: str) -> dict:
    if snapshot is None:
        return {"best_bid": None, "best_ask": None, "native_midpoint": None, "spread": None, "best_ask_source": "missing_snapshot"}
    bid = _finite_or_none(snapshot.get(f"{side}_best_bid"))
    ask = _finite_or_none(snapshot.get(f"{side}_best_ask"))
    source = "native_bbo"
    if ask is None or not 0.0 < ask < 1.0:
        levels = replay._ask_levels(snapshot.get(f"{side}_ask_levels"))
        ask = min(levels) if levels else None
        source = "lowest_saved_ask_level" if ask is not None else "unavailable"
    midpoint = (bid + ask) / 2.0 if bid is not None and ask is not None and 0.0 < bid <= ask < 1.0 else None
    spread = ask - bid if bid is not None and ask is not None and bid <= ask else None
    return {"best_bid": bid, "best_ask": ask, "native_midpoint": midpoint, "spread": spread, "best_ask_source": source}


def _status_for_evaluation(evaluation: dict) -> tuple[str, float | None]:
    side = evaluation.get("chosen_side")
    if evaluation.get("priceable_side_count", 0) == 0:
        return "no_priceable_t60_side", None
    if side is None:
        return "nonpositive_net_ev", None
    ev = _finite_or_none(evaluation.get("chosen_ev_usd"))
    fraction = ev / FIXED_GROSS_USD if ev is not None else None
    if fraction is None:
        return "nonpositive_net_ev", None
    if fraction + 1e-12 < MINIMUM_NET_EV_FRACTION:
        return "positive_ev_below_1pct", fraction
    return "at_or_above_1pct", fraction


def _waterfall_row(
    *, period: str, row, market: dict, t60_snapshot: dict | None, t59_snapshot: dict | None,
    chosen_side: str, chosen_result: dict, decision_ev_fraction: float,
) -> tuple[list[dict], dict]:
    p_up = float(row.p_D_conditional)
    p_selected = p_up if chosen_side == "up" else 1.0 - p_up
    p_market_up = float(row.p_market)
    p_reference = p_market_up if chosen_side == "up" else 1.0 - p_market_up
    quotes = _snapshot_side_prices(t60_snapshot, chosen_side)
    best_ask = quotes["best_ask"]
    midpoint = quotes["native_midpoint"]
    t60_fill = chosen_result.get("fill")
    t59_decision = {"chosen_side": chosen_side, "requested_gross_usd": FIXED_GROSS_USD}
    t59_result = _fixed_intent_at_t59(market, t59_snapshot, t59_decision)
    stage_fills = {
        "normalized_p_market_reference": _fill_at_single_price(p_reference, market, FIXED_GROSS_USD),
        "native_midpoint": _fill_at_single_price(midpoint, market, FIXED_GROSS_USD),
        "t60_best_ask": _fill_at_single_price(best_ask, market, FIXED_GROSS_USD),
        "t60_executable_depth": t60_fill,
        "t59_same_intent": t59_result.get("fill"),
    }
    stage_meta = {
        "normalized_p_market_reference": {"price": p_reference, "price_source": "normalized T60 p_market"},
        "native_midpoint": {"price": midpoint, "price_source": "native T60 BBO midpoint"},
        "t60_best_ask": {"price": best_ask, "price_source": quotes["best_ask_source"]},
        "t60_executable_depth": {"price": _finite_or_none((t60_fill or {}).get("vwap")), "price_source": (t60_fill or {}).get("execution_assumption")},
        "t59_same_intent": {"price": _finite_or_none((t59_result.get("fill") or {}).get("vwap")), "price_source": (t59_result.get("fill") or {}).get("execution_assumption")},
    }
    stage_metrics = {}
    for stage in SCENARIO_ORDER:
        stage_metrics[stage] = _fill_metrics(stage_fills[stage], p_selected)

    scenario_rows = []
    previous = None
    for stage in SCENARIO_ORDER:
        item = stage_metrics[stage]
        meta = stage_meta[stage]
        current = {
            "period": period, "condition_id": str(row.condition_id).lower(),
            "market_start_utc": pd.Timestamp(row.market_start_utc).isoformat(),
            "month_utc": pd.Timestamp(row.market_start_utc).strftime("%Y-%m"),
            "chosen_side": chosen_side, "p_D_selected_side": p_selected,
            "p_market_selected_side": p_reference,
            "model_probability_correction_selected_side": p_selected - p_reference,
            "decision_t60_net_ev_fraction": decision_ev_fraction,
            "scenario": stage, "scenario_price": meta["price"], "price_source": meta["price_source"],
            **item,
            "native_midpoint_t60": midpoint, "best_ask_t60": best_ask,
            "t60_spread_selected_side": quotes["spread"],
            "t59_priceable_same_intent": t59_result["priceable"],
            "t59_reason_same_intent": t59_result["reason"],
        }
        current.update(_waterfall_increment(previous, item))
        previous = item
        scenario_rows.append(current)

    net_values = [stage_metrics[stage]["net_ev_usd"] for stage in SCENARIO_ORDER]
    if all(value is not None for value in net_values):
        increments = [scenario_rows[i]["delta_net_ev_from_previous_usd"] for i in range(1, len(scenario_rows))]
        if not math.isclose(sum(increments), net_values[-1] - net_values[0], rel_tol=0.0, abs_tol=2e-10):
            raise AssertionError("The sequential EV waterfall does not sum to its end-to-end change")

    summary = {
        "period": period, "condition_id": str(row.condition_id).lower(),
        "market_start_utc": pd.Timestamp(row.market_start_utc).isoformat(),
        "month_utc": pd.Timestamp(row.market_start_utc).strftime("%Y-%m"),
        "chosen_side": chosen_side, "p_D_selected_side": p_selected,
        "p_market_selected_side": p_reference,
        "model_probability_correction_selected_side": p_selected - p_reference,
        "t60_ev_fraction": decision_ev_fraction,
        "t60_native_midpoint": midpoint, "t60_best_ask": best_ask,
        "t60_spread": quotes["spread"], "t60_best_ask_source": quotes["best_ask_source"],
        "t60_vwap": stage_metrics["t60_executable_depth"]["vwap"],
        "t60_gross_filled_usd": stage_metrics["t60_executable_depth"]["filled_gross_usd"],
        "t60_gross_shares": stage_metrics["t60_executable_depth"]["gross_shares"],
        "t60_net_shares": stage_metrics["t60_executable_depth"]["net_shares"],
        "t60_fee_shares": stage_metrics["t60_executable_depth"]["fee_shares"],
        "t60_fee_cash_usd": stage_metrics["t60_executable_depth"]["fee_cash_usd"],
        "t60_fee_usd_equivalent": stage_metrics["t60_executable_depth"]["fee_usd_equivalent"],
        "t60_cash_debit_usd": stage_metrics["t60_executable_depth"]["cash_debit_usd"],
        "t60_net_ev_usd": stage_metrics["t60_executable_depth"]["net_ev_usd"],
        "t60_net_ev_fraction_requested": stage_metrics["t60_executable_depth"]["net_ev_fraction_requested"],
        "t60_break_even_probability": stage_metrics["t60_executable_depth"]["break_even_probability"],
        "t60_probability_minus_break_even": stage_metrics["t60_executable_depth"]["probability_minus_break_even"],
        "t60_ev_hurdle_gap_usd": stage_metrics["t60_executable_depth"]["ev_hurdle_gap_usd"],
        "t60_ev_hurdle_gap_fraction": stage_metrics["t60_executable_depth"]["ev_hurdle_gap_fraction"],
        "t59_priceable": t59_result["priceable"], "t59_reason": t59_result["reason"],
        "t59_vwap": stage_metrics["t59_same_intent"]["vwap"],
        "t59_gross_filled_usd": stage_metrics["t59_same_intent"]["filled_gross_usd"],
        "t59_gross_shares": stage_metrics["t59_same_intent"]["gross_shares"],
        "t59_net_shares": stage_metrics["t59_same_intent"]["net_shares"],
        "t59_fee_cash_usd": stage_metrics["t59_same_intent"]["fee_cash_usd"],
        "t59_cash_debit_usd": stage_metrics["t59_same_intent"]["cash_debit_usd"],
        "t59_partial_fill": stage_metrics["t59_same_intent"]["partial_fill"],
        "t59_net_ev_usd": stage_metrics["t59_same_intent"]["net_ev_usd"],
        "t59_net_ev_fraction_requested": stage_metrics["t59_same_intent"]["net_ev_fraction_requested"],
        "t59_break_even_probability": stage_metrics["t59_same_intent"]["break_even_probability"],
        "t59_probability_minus_break_even": stage_metrics["t59_same_intent"]["probability_minus_break_even"],
        "t59_release_at_utc": None,
    }
    return scenario_rows, summary


def _evaluate_period(period: str, frame: pd.DataFrame, inputs: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    side_rows, market_rows, waterfall_rows, intent_rows = [], [], [], []
    market_by_id = inputs["market_by_id"]
    t59_by_id, t60_by_id = inputs["t59_by_id"], inputs["t60_by_id"]
    for row in frame.itertuples(index=False):
        condition_id = str(row.condition_id).lower()
        market = dict(market_by_id[condition_id])
        market["p_candidate_platt"] = float(row.p_D_conditional)
        t60_snapshot = t60_by_id.get(condition_id)
        t59_snapshot = t59_by_id.get(condition_id)
        if t60_snapshot is None:
            evaluation = {"sides": {}, "priceable_side_count": 0, "positive_ev_side_count": 0, "chosen_side": None, "chosen_ev_usd": None}
        else:
            evaluation = replay._evaluate_market_sides(market, t60_snapshot, FIXED_GROSS_USD)
        status, ev_fraction = _status_for_evaluation(evaluation)
        quality_row = {
            "period": period, "condition_id": condition_id,
            "market_start_utc": pd.Timestamp(row.market_start_utc),
            "month_utc": pd.Timestamp(row.market_start_utc).strftime("%Y-%m"),
            "economic_status": status,
            "priceable_side_count": int(evaluation.get("priceable_side_count", 0)),
            "positive_ev_side_count": int(evaluation.get("positive_ev_side_count", 0)),
            "chosen_side": evaluation.get("chosen_side"), "chosen_ev_usd": _finite_or_none(evaluation.get("chosen_ev_usd")),
            "chosen_ev_fraction_requested": ev_fraction,
            "p_market": float(row.p_market), "p_D_conditional": float(row.p_D_conditional),
            "delta_probability_up": float(row.p_D_conditional) - float(row.p_market),
            "abs_delta_probability_up": abs(float(row.p_D_conditional) - float(row.p_market)),
            "official_outcome_up": int(row.official_outcome_up),
            "label_available_at_utc": pd.Timestamp(row.label_available_at_utc),
            "d_bin": str(row.d_bin), "market_price_bin": str(row.market_price_bin),
            "volatility_bin": str(row.volatility_bin),
        }
        market_rows.append(quality_row)
        for side in ("up", "down"):
            result = evaluation.get("sides", {}).get(side, {})
            fill = result.get("fill") or {}
            p_side = float(row.p_D_conditional) if side == "up" else 1.0 - float(row.p_D_conditional)
            p_mkt_side = float(row.p_market) if side == "up" else 1.0 - float(row.p_market)
            quotes = _snapshot_side_prices(t60_snapshot, side)
            fill_metrics = _fill_metrics(fill, p_side)
            side_rows.append({
                **quality_row,
                "side": side, "p_D_side": p_side, "p_market_side": p_mkt_side,
                "model_probability_correction_side": p_side - p_mkt_side,
                "native_midpoint_t60": quotes["native_midpoint"], "best_bid_t60": quotes["best_bid"],
                "best_ask_t60": quotes["best_ask"], "best_ask_source": quotes["best_ask_source"],
                "spread_t60": quotes["spread"], "priceable_t60": bool(result.get("priceable", False)),
                "reason_t60": result.get("reason", "missing_t60_snapshot"),
                "chosen_side_by_positive_ev_rule": evaluation.get("chosen_side") == side,
                "ev_net_usd": _finite_or_none(result.get("ev_usd")),
                **{f"t60_{name}": value for name, value in fill_metrics.items()},
            })

        chosen_side = evaluation.get("chosen_side")
        if chosen_side is not None:
            chosen_result = evaluation["sides"][chosen_side]
            rows, intent = _waterfall_row(
                period=period, row=row, market=market, t60_snapshot=t60_snapshot,
                t59_snapshot=t59_snapshot, chosen_side=chosen_side,
                chosen_result=chosen_result, decision_ev_fraction=float(ev_fraction),
            )
            waterfall_rows.extend(rows)
            intent["economic_status"] = status
            intent["official_outcome_up"] = int(row.official_outcome_up)
            release = pd.Timestamp(row.label_available_at_utc) if pd.notna(row.label_available_at_utc) else pd.NaT
            intent["t59_release_at_utc"] = release.isoformat() if pd.notna(release) else None
            settled = _before_exclusive_cutoff(release)
            t59_last = rows[-1]
            intent["t59_priceable"] = bool(t59_last["t59_priceable_same_intent"])
            intent["t59_partial_fill"] = bool(t59_last["partial_fill"])
            intent["t59_fill_status"] = t59_last["fill_status"]
            intent["t59_settled_before_exclusive_cutoff"] = bool(settled)
            intent["t59_hypothetical_pnl_usd"] = (
                float(t59_last["net_shares"] if int(row.official_outcome_up) == int(chosen_side == "up") else 0.0) - float(t59_last["cash_debit_usd"])
                if settled and intent["t59_priceable"] and t59_last["net_shares"] is not None and t59_last["cash_debit_usd"] is not None
                else None
            )
            intent_rows.append(intent)

    markets = pd.DataFrame(market_rows)
    sides = pd.DataFrame(side_rows)
    waterfall = pd.DataFrame(waterfall_rows)
    intents = pd.DataFrame(intent_rows)
    return markets, sides, waterfall, intents


def _metric_summary(frame: pd.DataFrame) -> dict:
    y = frame.official_outcome_up.to_numpy(dtype=np.float64)
    p_a = np.clip(frame.p_A_market.to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
    p_d = np.clip(frame.p_D_conditional.to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
    ll_a = -(y * np.log(p_a) + (1.0 - y) * np.log1p(-p_a))
    ll_d = -(y * np.log(p_d) + (1.0 - y) * np.log1p(-p_d))
    brier_a, brier_d = np.square(p_a - y), np.square(p_d - y)
    delta = p_d - p_a
    correction = frame.p_D_conditional.to_numpy(dtype=np.float64) - frame.p_market.to_numpy(dtype=np.float64)
    return {
        "n": int(len(frame)), "log_loss_A": float(ll_a.mean()), "log_loss_D": float(ll_d.mean()),
        "delta_log_loss_D_minus_A": float((ll_d - ll_a).mean()),
        "brier_A": float(brier_a.mean()), "brier_D": float(brier_d.mean()),
        "delta_brier_D_minus_A": float((brier_d - brier_a).mean()),
        "mean_abs_probability_correction": float(np.abs(delta).mean()),
        "median_abs_probability_correction": float(np.median(np.abs(delta))),
        "p95_abs_probability_correction": float(np.quantile(np.abs(delta), 0.95)),
        "min_signed_probability_correction": float(correction.min()),
        "median_signed_probability_correction": float(np.median(correction)),
        "max_signed_probability_correction": float(correction.max()),
    }


def _quality_tables(period_frames: dict[str, pd.DataFrame], market_decisions: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows, month_rows, status_rows, context_rows = [], [], [], []
    for period, original in period_frames.items():
        if period == "development_oof":
            frame = original.copy()
        else:
            release = pd.to_datetime(original.label_available_at_utc, utc=True, errors="coerce")
            frame = original.loc[release.lt(QUALITY_CUTOFF)].copy()
        if frame.official_outcome_up.isna().any():
            raise RuntimeError(f"{period} contains missing official labels in its quality sample")
        frame["period"] = period
        overall_rows.append({"period": period, **_metric_summary(frame)})
        for month, group in frame.groupby("month_utc", sort=True):
            month_rows.append({"period": period, "month_utc": month, **_metric_summary(group)})

        decisions = market_decisions[period].set_index("condition_id")
        frame = frame.join(decisions[["economic_status"]], on="condition_id", rsuffix="_decision")
        for status, group in frame.groupby("economic_status", dropna=False, sort=True):
            status_rows.append({"period": period, "economic_status": status, **_metric_summary(group)})
        for feature, bucket in (
            ("d_btc_minus_market", "d_bin"),
            ("p_market", "market_price_bin"),
            ("btc_sigma_5m", "volatility_bin"),
        ):
            for label, group in frame.groupby(bucket, dropna=False, observed=True, sort=True):
                context_rows.append({"period": period, "context": feature, "training_defined_bin": str(label), **_metric_summary(group)})
    return pd.DataFrame(overall_rows), pd.DataFrame(month_rows), pd.DataFrame(status_rows), pd.DataFrame(context_rows)


def _quality_cost_associations(
    period_frames: dict[str, pd.DataFrame], side_quotes: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for period, original in period_frames.items():
        frame = original.copy()
        if period == "holdout_historical_diagnostic":
            release = pd.to_datetime(frame.label_available_at_utc, utc=True, errors="coerce")
            frame = frame.loc[release.lt(QUALITY_CUTOFF)].copy()
        y = frame.official_outcome_up.to_numpy(dtype=np.float64)
        p_a = np.clip(frame.p_A_market.to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
        p_d = np.clip(frame.p_D_conditional.to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
        frame["delta_log_loss_D_minus_A"] = (
            -(y * np.log(p_d) + (1.0 - y) * np.log1p(-p_d))
            + y * np.log(p_a) + (1.0 - y) * np.log1p(-p_a)
        )
        frame["delta_brier_D_minus_A"] = np.square(p_d - y) - np.square(p_a - y)
        frame["abs_probability_correction"] = np.abs(p_d - frame.p_market.to_numpy(dtype=np.float64))
        selected_quotes = side_quotes.loc[
            side_quotes.period.eq(period) & side_quotes.priceable_t60,
            ["condition_id", "spread_t60", "best_ask_t60", "p_market_side"],
        ].copy()
        selected_quotes["ask_premium_vs_normalized_p_market"] = selected_quotes.best_ask_t60 - selected_quotes.p_market_side
        market_costs = selected_quotes.groupby("condition_id", sort=False).agg(
            mean_priceable_side_spread=("spread_t60", "mean"),
            mean_priceable_side_ask_premium=("ask_premium_vs_normalized_p_market", "mean"),
        )
        frame = frame.join(market_costs, on="condition_id")
        for measure in ("mean_priceable_side_spread", "mean_priceable_side_ask_premium", "abs_probability_correction"):
            quality_measures = ("delta_log_loss_D_minus_A", "delta_brier_D_minus_A")
            if measure != "abs_probability_correction":
                quality_measures += ("abs_probability_correction",)
            for quality in quality_measures:
                valid = frame[[measure, quality]].dropna()
                correlation = float(valid[measure].corr(valid[quality])) if len(valid) > 1 else None
                if correlation is not None and not math.isfinite(correlation):
                    correlation = None
                rows.append({
                    "period": period, "associated_measure": measure, "quality_measure": quality,
                    "n": int(len(valid)), "pearson_correlation": correlation,
                })
    return pd.DataFrame(rows)


def _distribution(values: pd.Series | np.ndarray) -> dict:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=np.float64)
    if not len(array):
        return {"n": 0, "min": None, "p10": None, "p25": None, "median": None, "p75": None, "p90": None, "max": None}
    return {
        "n": int(len(array)), "min": float(np.min(array)), "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)), "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)), "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _scenario_summaries(waterfall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if waterfall.empty:
        return pd.DataFrame()
    for (period, scenario), group in waterfall.groupby(["period", "scenario"], sort=False):
        net = pd.to_numeric(group.net_ev_usd, errors="coerce")
        margin = pd.to_numeric(group.probability_minus_break_even, errors="coerce")
        fractions = pd.to_numeric(group.net_ev_fraction_requested, errors="coerce")
        distributions = {
            "net_ev_usd": _distribution(net),
            "probability_minus_break_even": _distribution(margin),
            "net_ev_fraction_requested": _distribution(fractions),
        }
        row = {
            "period": period, "scenario": scenario, "intent_rows": int(len(group)),
            "priceable_rows": int(net.notna().sum()),
            "positive_net_ev_rows": int((net > 0.0).sum()),
            "at_or_above_1pct_rows": int((fractions >= MINIMUM_NET_EV_FRACTION - 1e-12).sum()),
            "partial_fills": int(group.partial_fill.fillna(False).sum()),
            "median_vwap": float(pd.to_numeric(group.vwap, errors="coerce").median()) if pd.to_numeric(group.vwap, errors="coerce").notna().any() else None,
            "median_fee_usd_equivalent": float(pd.to_numeric(group.fee_usd_equivalent, errors="coerce").median()) if pd.to_numeric(group.fee_usd_equivalent, errors="coerce").notna().any() else None,
            "median_filled_gross_usd": float(pd.to_numeric(group.filled_gross_usd, errors="coerce").median()) if pd.to_numeric(group.filled_gross_usd, errors="coerce").notna().any() else None,
        }
        for measure, values in distributions.items():
            for quantile in ("min", "p10", "median", "p90", "max"):
                row[f"{measure}_{quantile}"] = values[quantile]
        rows.append(row)
    return pd.DataFrame(rows)


def _tree_explanation(booster: lgb.Booster, frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    model = booster.dump_model()
    tree_info = model["tree_info"]
    split_rows = []
    leaf_maps = []
    centers = []
    feature_indices = []
    for tree_number, tree in enumerate(tree_info):
        root = tree["tree_structure"]
        if "split_index" not in root or "split_index" in root.get("left_child", {}):
            raise RuntimeError(f"Tree {tree_number} is not a one-split stump")
        left, right = root["left_child"], root["right_child"]
        if "leaf_index" not in left or "leaf_index" not in right:
            raise RuntimeError(f"Tree {tree_number} contains more than one split")
        feature_index = int(root["split_feature"])
        left_value, right_value = float(left["leaf_value"]), float(right["leaf_value"])
        center = (left_value + right_value) / 2.0
        leaf_maps.append({int(left["leaf_index"]): left_value, int(right["leaf_index"]): right_value})
        centers.append(center)
        feature_indices.append(feature_index)
        feature = FEATURES[feature_index]
        split_rows.append({
            "tree_index": tree_number + 1, "feature_index": feature_index,
            "feature": feature, "threshold": float(root["threshold"]),
            "decision_type": root["decision_type"], "split_gain": float(root["split_gain"]),
            "left_leaf_index": int(left["leaf_index"]), "left_leaf_value": left_value,
            "right_leaf_index": int(right["leaf_index"]), "right_leaf_value": right_value,
            "structural_center": center,
        })
    constant = float(np.sum(centers))
    split_frame = pd.DataFrame(split_rows)
    contribution_rows = []
    reconstruct_summary = {"center_constant": constant, "centering": "per tree, subtract the unweighted mean of its two leaf values; sum the removed means as a separate constant"}
    for period, frame in frames.items():
        leaves = np.asarray(booster.predict(frame.loc[:, FEATURES].to_numpy(dtype=np.float64), pred_leaf=True, num_threads=4))
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        if leaves.shape != (len(frame), len(tree_info)):
            raise RuntimeError(f"Unexpected leaf-prediction shape for {period}: {leaves.shape}")
        contributions = np.empty(leaves.shape, dtype=np.float64)
        for tree_number, leaf_map in enumerate(leaf_maps):
            contributions[:, tree_number] = np.fromiter(
                (leaf_map[int(index)] for index in leaves[:, tree_number]),
                dtype=np.float64, count=len(frame),
            )
        raw = contributions.sum(axis=1)
        center_constant, centered, grouped = _center_and_group_stump_contributions(
            contributions, np.asarray(centers, dtype=np.float64), feature_indices, len(FEATURES),
        )
        reconstructed = center_constant + centered.sum(axis=1)
        error = np.abs(reconstructed - raw)
        max_error = float(error.max()) if len(error) else 0.0
        if max_error > 1e-10:
            raise AssertionError(f"Centered stump contributions fail to reconstruct {period} margins: {max_error}")
        denominator = float(np.abs(grouped).sum())
        for feature_index, feature in enumerate(FEATURES):
            values = grouped[:, feature_index]
            total_abs = float(np.abs(values).sum())
            contribution_rows.append({
                "period": period, "feature": feature,
                "tree_count": int(sum(index == feature_index for index in feature_indices)),
                "gain_sum": float(split_frame.loc[split_frame.feature_index.eq(feature_index), "split_gain"].sum()),
                "centered_contribution_mean": float(values.mean()),
                "centered_contribution_mean_abs": float(np.abs(values).mean()),
                "centered_contribution_rms": float(np.sqrt(np.mean(np.square(values)))),
                "aggregate_absolute_contribution": total_abs,
                "share_of_grouped_absolute_contribution": total_abs / denominator if denominator else 0.0,
                "raw_margin_min": float(raw.min()), "raw_margin_median": float(np.median(raw)),
                "raw_margin_max": float(raw.max()), "max_reconstruction_error": max_error,
            })
        reconstruct_summary[period] = {
            "rows": int(len(frame)), "raw_margin_min": float(raw.min()),
            "raw_margin_median": float(np.median(raw)), "raw_margin_max": float(raw.max()),
            "max_absolute_reconstruction_error": max_error,
            "mean_absolute_probability_correction": float(np.abs(frame.p_D_conditional - frame.p_market).mean()),
            "p95_absolute_probability_correction": float(np.quantile(np.abs(frame.p_D_conditional - frame.p_market), 0.95)),
            "min_signed_probability_correction": float((frame.p_D_conditional - frame.p_market).min()),
            "median_signed_probability_correction": float((frame.p_D_conditional - frame.p_market).median()),
            "max_signed_probability_correction": float((frame.p_D_conditional - frame.p_market).max()),
        }
    feature_rows = []
    total_gain = float(split_frame.split_gain.sum())
    for feature in FEATURES:
        used = split_frame.loc[split_frame.feature.eq(feature)]
        thresholds = "; ".join(f"{value:.10g}" for value in used.threshold.tolist())
        feature_rows.append({
            "feature": feature, "split_count": int(len(used)),
            "total_gain": float(used.split_gain.sum()),
            "gain_share": float(used.split_gain.sum() / total_gain) if total_gain else 0.0,
            "thresholds": thresholds,
        })
    return split_frame, pd.DataFrame(feature_rows), {"reconstruction": reconstruct_summary, "feature_contributions": contribution_rows}


def _assert_holdout_decision_reproduction(market_summary: pd.DataFrame, inputs: dict) -> dict:
    decisions = inputs["old_decisions"]
    holdout = market_summary.loc[market_summary.period.eq("holdout_historical_diagnostic")].set_index("condition_id")
    if len(decisions) != len(holdout):
        raise AssertionError(f"Frozen portfolio decision count differs: {len(decisions)} vs {len(holdout)}")
    if not np.allclose(decisions.requested_gross_usd.to_numpy(dtype=np.float64), FIXED_GROSS_USD, rtol=0.0, atol=1e-12):
        raise AssertionError("The frozen D portfolio decisions are not all fixed-$5 requests")
    if not np.allclose(decisions.cash_before_usd.to_numpy(dtype=np.float64), 100.0, rtol=0.0, atol=1e-12):
        raise AssertionError("The frozen D portfolio spent or locked cash despite having no accepted orders")
    expected_sides = holdout.chosen_side.astype("string").fillna("").sort_index()
    frozen_sides = decisions.set_index("condition_id").chosen_side.astype("string").fillna("").sort_index()
    if not expected_sides.equals(frozen_sides):
        raise AssertionError("Reconstructed T-60 chosen sides differ from the frozen portfolio decisions")
    original_counts = decisions.skip_reason.value_counts(dropna=False).to_dict()
    comparison = {
        "frozen_portfolio_decision_rows": int(len(decisions)),
        "frozen_skip_reason_counts": {str(k): int(v) for k, v in original_counts.items()},
        "reproduced_status_counts": {str(k): int(v) for k, v in holdout.economic_status.value_counts().items()},
    }
    below_ids = set(decisions.loc[decisions.skip_reason.eq("below_minimum_net_ev_fraction"), "condition_id"])
    reproduced_below = set(holdout.index[holdout.economic_status.eq("positive_ev_below_1pct")])
    if below_ids != reproduced_below:
        raise AssertionError("The 37 below-hurdle cases do not match the frozen D portfolio decisions")
    if not holdout.index.isin(decisions.condition_id).all():
        raise AssertionError("The frozen D decisions do not cover the holdout economic universe")
    comparison["below_1pct_condition_ids_match"] = True
    comparison["below_1pct_count"] = len(below_ids)
    return comparison


def _fee_semantics_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"item": "p_market", "rule": "At T-60: mid_up=(UP best bid+ask)/2; mid_down=(DOWN best bid+ask)/2; p_market=mid_up/(mid_up+mid_down). For DOWN use 1-p_market."},
        {"item": "probability change versus EV percent", "rule": "A probability-point correction is not percent EV on stake; EV also depends on executable shares, ask price, depth, and fees."},
        {"item": "gross request", "rule": "$5 gross USDC requested for the fixed-order comparison; denominator of the original EV fraction"},
        {"item": "side selection", "rule": "At T-60, price both native ask sides for $5; choose the side with the greatest strictly positive after-fee EV; exact tie favors UP"},
        {"item": "qualification", "rule": "net_ev_usd / requested_gross_usd >= 1%; original implementation allows 1e-12 numerical tolerance"},
        {"item": "price cap", "rule": "None in the existing engine: FAK-style marketable BUY walks saved asks at execution; no T-60-derived limit price is passed"},
        {"item": "book depth", "rule": "Walk complete saved ask ladder when depth is known; otherwise assume full requested fill at saved best ask; known insufficient depth may partially fill"},
        {"item": "minimum order", "rule": "Current Gamma orderMinSize when present, otherwise $5 USDC fallback; checked against requested gross amount, not the eventual partial fill; historical minimum metadata is unavailable"},
        {"item": "fee rate", "rule": "Date-assigned BTC5m schedule: 7.2% before 2026-05-06, 7.0% from 2026-05-06; exponent 1; transition date is a stated estimate"},
        {"item": "fee formula", "rule": "Per price level: shares * rate * price * (1-price); the fee engine uses Decimal and its documented rounding"},
        {"item": "V1 BUY fee", "rule": "Before the assigned V1/V2 collection change, fee is outcome shares, floored to 1e-6 per saved aggregate price level; cash debit is gross"},
        {"item": "V2 BUY fee", "rule": "Cash fee rounded half-up to 5 decimals per order with 0.00001 minimum; net shares equal gross shares; cash debit is gross plus fee"},
        {"item": "expected payout", "rule": "p_D(selected side) * net shares; a winning token pays $1 per net share"},
        {"item": "break-even", "rule": "cash_debit_usd / net_shares; the exact p_D at which net expected value is zero"},
        {"item": "T-59", "rule": "Use the T-60 chosen side and same requested stake; price its T-59 ask ladder only. T-59 cannot select another side or retroactively qualify the decision"},
    ])


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g", na_rep="")


def _md_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    selected = frame.loc[:, columns]
    if limit is not None:
        selected = selected.head(limit)
    if selected.empty:
        return "*(no rows)*"
    def cell(value) -> str:
        if value is None or pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in selected.columns]
    rows = [[cell(value) for value in row] for row in selected.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _build_report(summary: dict, quality: dict[str, pd.DataFrame], funnel: pd.DataFrame,
                  waterfall_summary: pd.DataFrame, fee_table: pd.DataFrame,
                  tree_features: pd.DataFrame, feature_contributions: pd.DataFrame,
                  example: pd.DataFrame, source_hashes: dict, runtime: dict) -> str:
    overall, monthly, statuses, contexts = quality["overall"], quality["monthly"], quality["statuses"], quality["contexts"]
    development_qualified = statuses.loc[
        statuses.period.eq("development_oof") & statuses.economic_status.eq("at_or_above_1pct")
    ]
    development_qualified = development_qualified.iloc[0] if not development_qualified.empty else None
    holdout_qualified_count = int(funnel.loc[
        funnel.period.eq("holdout_historical_diagnostic"), "at_or_above_1pct",
    ].iloc[0])
    tree_contributions = feature_contributions.loc[
        feature_contributions.period.eq("holdout_historical_diagnostic")
    ].set_index("feature")
    d_split = tree_features.loc[tree_features.feature.eq("d_btc_minus_market")].iloc[0]
    d_contribution = float(tree_contributions.loc["d_btc_minus_market", "share_of_grouped_absolute_contribution"])
    basis_contribution = float(tree_contributions.loc["futures_index_basis_rel_1m", "share_of_grouped_absolute_contribution"])
    sigma_contribution = float(tree_contributions.loc["btc_sigma_5m", "share_of_grouped_absolute_contribution"])
    holdout_stage = waterfall_summary.loc[
        waterfall_summary.period.eq("holdout_historical_diagnostic")
    ].set_index("scenario")
    reference_median_ev = float(holdout_stage.loc["normalized_p_market_reference", "net_ev_usd_median"])
    ask_median_ev = float(holdout_stage.loc["t60_best_ask", "net_ev_usd_median"])
    executable_median_ev = float(holdout_stage.loc["t60_executable_depth", "net_ev_usd_median"])
    lines = [
        "# BTC conditional market-offset execution diagnostic - 2026-10-18", "",
        "> This is a read-only diagnostic of the frozen experiment in `btc_conditional_market_offset_20261018.md`. The holdout was historically exposed. No training, tuning, new threshold, replay, data collection, or policy change was run.", "",
        "## Findings", "",
        f"The frozen holdout has {summary['holdout_economic_rows']:,} economic decision rows; {summary['holdout_quality_rows']:,} have official labels released strictly before the exclusive cutoff, {summary['holdout_exact_cutoff_label_releases']} release exactly at the cutoff, and one additional row releases after cutoff. Development has {summary['development_oof_rows']:,} saved OOF predictions; the 2,119-row April prefix trained the fold models and has no out-of-fold D prediction.",
        f"At fixed $5 and T-60, D has {summary['holdout_funnel'].get('no_priceable_t60_side', 0):,} markets with no priceable side, {summary['holdout_funnel'].get('nonpositive_net_ev', 0):,} with a priceable side but no positive net EV, {summary['holdout_funnel'].get('positive_ev_below_1pct', 0):,} with positive EV below 1%, and {summary['holdout_funnel'].get('at_or_above_1pct', 0):,} at or above the 1% hurdle. The 37 below-hurdle IDs exactly match the frozen portfolio decisions. The prior independent-order aggregate combined unpriceable markets with nonpositive EV under `no_positive_net_ev_at_t60`; its 24,699 count is 1,625 + 23,074.",
        f"For those 37 positive-EV intents, the 1% rule requires $0.05 net EV on the requested $5 gross amount. Their actual T-60 EV and distance to that hurdle are summarized below. None qualifies; therefore the frozen policy makes no T-59 order and reports zero realized trades for D. All 37 labels are cutoff-safe; a same-intent T-59 shadow is priceable for {summary['t59_positive_below_hurdle']['priceable_fills']} and has cutoff-safe outcomes for {summary['t59_positive_below_hurdle']['realized_fills_used']} fills, with descriptive shadow PnL ${summary['t59_positive_below_hurdle']['descriptive_realized_pnl_usd']:+.2f}. These are not executed trades or a new policy.",
        f"In the fixed-order waterfall for those 37 cases, median net EV at the normalized p_market reference is ${reference_median_ev:.4f}; at the T-60 best ask it is ${ask_median_ev:.4f}; and after walking saved T-60 depth it is ${executable_median_ev:.4f}. The observed median drop occurs at the ask, while the depth walk adds no further median drop in this sample.",
        f"On saved D predictions, development OOF delta log-loss is {overall.loc[overall.period.eq('development_oof'),'delta_log_loss_D_minus_A'].iloc[0]:+.6f} and delta Brier is {overall.loc[overall.period.eq('development_oof'),'delta_brier_D_minus_A'].iloc[0]:+.6f}; holdout deltas are {overall.loc[overall.period.eq('holdout_historical_diagnostic'),'delta_log_loss_D_minus_A'].iloc[0]:+.6f} and {overall.loc[overall.period.eq('holdout_historical_diagnostic'),'delta_brier_D_minus_A'].iloc[0]:+.6f}. Development OOF includes {int(development_qualified.n) if development_qualified is not None else 0} T-60 cases at the frozen 1% hurdle (delta log-loss {development_qualified.delta_log_loss_D_minus_A:+.6f}, Brier {development_qualified.delta_brier_D_minus_A:+.6f}); holdout has {holdout_qualified_count}. Thus development quality gains do occur in a small set that clears the fixed execution rule, while holdout has no trade-qualified quality sample. All months, including October, remain in the monthly table.",
        "The order engine has no explicit price limit. It fixes side and stake at T-60, then walks the T-59 asks as a FAK-style marketable BUY, including partial fills. The model's probability correction is small relative to the cost of paying the native ask plus fees for most rows; the exact price waterfall shows which changes account for the net EV.",
        "",
        "## Frozen input audit", "",
        f"The saved final D model has 43 trees and reproduces the holdout raw margins and probabilities exactly. Development quality uses {summary['development_oof_rows']:,} saved fold-specific OOF predictions; the final full-development model is not expected to recreate those fold-model outputs. All {source_hashes['observed_label_rows']['matched_to_official_index_rows']:,} saved official labels match the hash-verified market index for outcome, start, resolution (including matching missing resolutions), and release time. The observed label rows hash is `{source_hashes['observed_label_rows']['sha256']}`; the prior experiment report hash confirms it was not overwritten.", "",
        "| Source | SHA-256 | Bytes |", "|---|---|---:|",
    ]
    for key, item in source_hashes.items():
        lines.append(f"| {key} (`{item['path']}`) | `{item['sha256']}` | {item.get('bytes', '')} |")
    lines += ["", "## Sample and T-60 funnel", "", _md_table(funnel, ["period", "population", "markets", "quality_label_rows", "no_priceable_t60_side", "nonpositive_net_ev", "positive_ev_below_1pct", "at_or_above_1pct", "positive_intents_priced_at_t59", "t59_priceable", "t59_partial_fills"]), "",
              "Status is assigned without filtering on the outcome: a side is selected only by the frozen positive-net-EV rule; the 1% criterion is applied afterward. The T-60 native ask reasons are broken out in `t60_unpriceable_side_reasons.csv`. The holdout quality sample excludes one economics row whose official label availability is after cutoff; the economic decision universe retains it, matching the frozen run.", "",
              "The frozen portfolio decision table reports 24,736 rows: 1,625 `no_priceable_native_ask_side`, 23,074 `no_positive_net_expected_value`, 37 `below_minimum_net_ev_fraction`, and zero accepted orders. The independent fixed-$5 evaluator's `no_positive_net_ev_at_t60=24,699` merges the first two categories because it increments that reason whenever no side is selected. The matching 37 below-1% cases occur in both calculations because the portfolio never spends cash and therefore its 5% free-cash request remains $5.", "",
              "## EV and execution definitions", "", _md_table(fee_table, ["item", "rule"]), "",
              "The original percent-EV denominator is the requested gross stake, not the fee-inclusive cash debit and not the filled amount. For a full $5 scenario, `EV_fraction = (p_D * net_shares - cash_debit) / 5`. For a partial T-59 fill the diagnostic keeps that same $5 denominator and separately reports filled gross. The exact zero-EV threshold is `p_break_even = cash_debit / net_shares`; the zero-EV condition is algebraically equivalent to `p_D = p_break_even`.", "",
              "### Hand-reconciled example", ""]
    if not example.empty:
        ex = example.iloc[0]
        lines += [
              f"Median-by-T-60-EV holdout example (chosen before looking at outcome): `{ex['market_start_utc']}`, side `{ex['chosen_side']}`. D assigns `{ex['p_D_selected_side']:.8f}` to that side, versus normalized market reference `{ex['p_market_selected_side']:.8f}`. At executable T-60 depth the VWAP is `{ex['t60_vwap']:.8f}` for gross filled `${ex['t60_gross_filled_usd']:.6f}`; gross shares are `{ex['t60_gross_shares']:.8f}`, fee shares `{ex['t60_fee_shares'] or 0:.8f}`, net shares `{ex['t60_net_shares']:.8f}`, cash fee `${ex['t60_fee_cash_usd'] or 0:.8f}`, and cash debit `${ex['t60_cash_debit_usd']:.8f}` (gross plus fee). "
            f"Expected payout is `{ex['p_D_selected_side']:.8f} * {ex['t60_net_shares']:.8f} = ${ex['t60_expected_payout_usd']:.8f}`. Subtracting cash debit gives net EV `${ex['t60_net_ev_usd']:.8f}`; it is `{ex['t60_net_ev_fraction_requested']:.4%}` of the requested $5 gross stake and `{ex['t60_net_ev_fraction_filled_cost']:.4%}` of the fee-inclusive cash debit. Break-even is `${ex['t60_cash_debit_usd']:.8f} / {ex['t60_net_shares']:.8f} = {ex['t60_break_even_probability']:.8f}`. The gap to the 1% hurdle is `${ex['t60_ev_hurdle_gap_usd']:.8f}`, or `{(-ex['t60_ev_hurdle_gap_fraction'] * 100.0):.4f}` percentage points. T-59 same-side VWAP is `{ex['t59_vwap']}` and the hypothetical fill status is `{ex['t59_fill_status']}`; it does not change the T-60 decision.",
            "",
        ]
    distribution_rows = [
        {"measure": key, **values}
        for key, values in summary["t59_positive_below_hurdle"]["distributions"].items()
    ]
    lines += ["### EV waterfall for positive T-60 intents", "", _md_table(waterfall_summary, ["period", "scenario", "intent_rows", "priceable_rows", "positive_net_ev_rows", "at_or_above_1pct_rows", "partial_fills", "median_vwap", "median_fee_usd_equivalent", "median_filled_gross_usd", "net_ev_usd_p10", "net_ev_usd_median", "net_ev_usd_p90", "net_ev_usd_max", "probability_minus_break_even_p10", "probability_minus_break_even_median", "probability_minus_break_even_p90", "net_ev_fraction_requested_median", "net_ev_fraction_requested_max"]), "",
              "Each reported stage uses the unchanged T-60 D probability and selected side. The normalized p_market stage is theoretical; native midpoint is a non-executable reference; best ask isolates crossing the spread; full depth adds ask-ladder costs; T-59 adds the one-second quote/depth change. For each adjacent step, `change in net EV = change in pre-fee EV + change in fee effect`; the per-intent CSV contains both terms and the exact net-EV difference. If T-59 fills less than $5, the report shows the smaller filled gross and net shares separately; that difference is not labelled only as slippage. Missing stages stay missing.", "",
              "`positive_ev_below_1pct_cases.csv` lists all 37 holdout cases, with their timestamp, fixed side, costs, p_break_even, T-59 shadow execution, and cutoff-safe realized outcome. Realized outcomes are descriptive only and do not imply a strategy.", "",
              "### Distribution across the 37 below-hurdle cases", "", _md_table(pd.DataFrame(distribution_rows), ["measure", "n", "min", "p10", "p25", "median", "p75", "p90", "max"]), "",
              "## Where probability quality improves", "", _md_table(overall, ["period", "n", "delta_log_loss_D_minus_A", "delta_brier_D_minus_A", "min_signed_probability_correction", "median_signed_probability_correction", "max_signed_probability_correction", "mean_abs_probability_correction", "median_abs_probability_correction", "p95_abs_probability_correction"]), "",
              "### By existing training-defined economic status", "", _md_table(statuses, ["period", "economic_status", "n", "delta_log_loss_D_minus_A", "delta_brier_D_minus_A", "mean_abs_probability_correction"]), "",
              "Status is mechanically defined by the frozen $5 T-60 evaluator, not selected by outcomes. A zero count for the 1%-qualified holdout tier means there is no holdout evidence about quality specifically among trade-qualified cases.", "",
              "### All months", "", _md_table(monthly, ["period", "month_utc", "n", "delta_log_loss_D_minus_A", "delta_brier_D_minus_A", "mean_abs_probability_correction"]), "",
              "October is retained: read its delta columns with their signs; positive means D is worse than A. The full 3-context-by-period table in `quality_by_existing_training_bins.csv` preserves the saved d, price, and volatility bins; no new holdout quantiles or profitable subsets were searched.", "",
              "### Full-sample quality versus correction size, spread, and ask premium", "", _md_table(quality["cost_associations"], ["period", "associated_measure", "quality_measure", "n", "pearson_correlation"]), "",
              "These Pearson associations use every quality-labeled row in each period and average the available priceable-side quote measures at T-60; they do not create bins or select subgroups. A negative correlation for a D-minus-A loss means the score difference tends to favor D as the associated measure increases. Ask premium is native best ask minus normalized p_market on each side; it is descriptive, not a realized cost estimate. The full-sample loss associations with correction magnitude, spread, and ask premium are close to zero, so these data do not show that quality gains are concentrated among small or large corrections, or at high spread/cost.", "",
              "### Existing training-defined context buckets", "", _md_table(contexts, ["period", "context", "training_defined_bin", "n", "delta_log_loss_D_minus_A", "delta_brier_D_minus_A", "mean_abs_probability_correction"]), "",
              "## What the 43 stumps learned", "",
              f"The `d_btc_minus_market` feature supplies {int(d_split.split_count)} of 43 splits, {d_split.gain_share:.1%} of split gain, and {d_contribution:.1%} of grouped absolute centered contribution on the holdout final-model predictions. Basis contributes {basis_contribution:.1%} and `btc_sigma_5m` contributes {sigma_contribution:.1%} of that grouped absolute amount. No split uses raw market logit, BTC returns, or spreads. This is a piecewise correction mainly along the BTC-market difference, with basis and volatility context; because `d` includes market logit, it does not isolate incremental BTC information.", "",
              f"The frozen final model has {summary['tree_count']} one-split trees and {summary['split_count']} splits. Centering subtracts each stump's unweighted mean of its two leaf values; the sum of those removed values is the fixed constant `{summary['center_constant']:.12g}`. Grouped centered tree contributions plus this constant reproduce every holdout final-model raw margin; maximum observed reconstruction error is `{summary['max_tree_reconstruction_error']:.3g}`. Development's saved OOF probabilities came from fold-specific models and are used for quality metrics, not this final-model decomposition.", "",
              _md_table(tree_features, ["feature", "split_count", "total_gain", "gain_share", "thresholds"]), "",
              "Per-period grouped feature contribution shares are in `feature_contributions.csv`; the exact 43 thresholds, gains, and leaf values are in `tree_splits.csv`. Gain and additive contribution are descriptive, not causal. The feature `d = logit(p_BTC)-logit(p_market)` already contains the market probability, so its use does not isolate incremental BTC information.", "",
              "## Reproduction and limitations", "",
              "Run from the repository root: `python run_btc_conditional_market_offset_diagnostics_20261018.py`. It verifies frozen hashes, loads saved predictions/model/snapshots, reproduces the existing T-60 rule, and writes separate diagnostic outputs. It does not train, tune, refit, fetch data, replay archives, or alter the original experiment artifacts.", "",
              "The exact fee amount and V1/V2 cutover remain estimates in the accepted fee registry; ask depth is a saved historical snapshot, and incomplete-depth rows use the engine's top-of-book full-fill proxy. Current Gamma minimums are not historical. `btc_sigma_5m` is sqrt(5) times the population standard deviation of the prior 60 contiguous one-minute close returns. T-59 is a one-second diagnostic, not a live-liquidity forecast. Positive forecast EV is not evidence of realized profit.", "",
              f"Diagnostic runtime: {runtime['elapsed_seconds']:.1f}s. Detailed read-only market scenarios are under `data/analysis/polymarket/BTC/btc_conditional_market_offset_20261018/diagnostics/`. Summary tables are in `reports/btc_preopen/btc_conditional_market_offset_20261018_diagnostics/`.", ""]
    return "\n".join(lines).rstrip() + "\n"


def main() -> dict:
    started = time.perf_counter()
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    inputs = _load_inputs()
    dev = inputs["dev"]
    holdout_all = inputs["holdout"]
    cutoff = pd.Timestamp("2026-07-01T00:00:00Z")
    dev_quality = dev.loc[pd.to_datetime(dev.label_available_at_utc, utc=True).lt(cutoff)].copy()
    holdout_quality = holdout_all.loc[pd.to_datetime(holdout_all.label_available_at_utc, utc=True).lt(QUALITY_CUTOFF)].copy()
    exact_cutoff_labels = int(pd.to_datetime(holdout_all.label_available_at_utc, utc=True).eq(QUALITY_CUTOFF).sum())
    if len(dev_quality) != 8675 or len(holdout_quality) != 24735:
        raise RuntimeError(f"Saved quality sample sizes changed: development={len(dev_quality)}, holdout={len(holdout_quality)}")
    if len(holdout_all) != 24736:
        raise RuntimeError(f"Saved holdout economic universe changed: {len(holdout_all)}")

    dev_markets, dev_sides, dev_waterfall, dev_intents = _evaluate_period("development_oof", dev_quality, inputs)
    holdout_markets, holdout_sides, holdout_waterfall, holdout_intents = _evaluate_period("holdout_historical_diagnostic", holdout_all, inputs)
    market_summaries = pd.concat([dev_markets, holdout_markets], ignore_index=True)
    side_quotes = pd.concat([dev_sides, holdout_sides], ignore_index=True)
    waterfall = pd.concat([dev_waterfall, holdout_waterfall], ignore_index=True)
    intents = pd.concat([dev_intents, holdout_intents], ignore_index=True)

    decision_audit = _assert_holdout_decision_reproduction(market_summaries, inputs)
    expected_portfolio_reasons = inputs["results"]["economics"]["models"]["D"]["portfolio_100_usd_free_cash_5pct"]["rejections_by_reason"]
    holdout_counts = market_summaries.loc[market_summaries.period.eq("holdout_historical_diagnostic"), "economic_status"].value_counts().to_dict()
    expected_to_status = {
        "no_priceable_native_ask_side": int(holdout_counts.get("no_priceable_t60_side", 0)),
        "no_positive_net_expected_value": int(holdout_counts.get("nonpositive_net_ev", 0)),
        "below_minimum_net_ev_fraction": int(holdout_counts.get("positive_ev_below_1pct", 0)),
        "below_minimum_order_usd": int(side_quotes.loc[(side_quotes.period.eq("holdout_historical_diagnostic")) & side_quotes.reason_t60.eq("below_minimum_order_usd")].shape[0] // 2),
    }
    for reason in ("no_priceable_native_ask_side", "no_positive_net_expected_value", "below_minimum_net_ev_fraction"):
        if expected_portfolio_reasons.get(reason, 0) != expected_to_status[reason]:
            raise AssertionError(f"Frozen decision reason {reason} does not match reconstructed T-60 status")
    decision_audit["frozen_reason_reconciliation"] = expected_to_status
    decision_audit["frozen_reason_counts"] = expected_portfolio_reasons
    decision_audit["holdout_quality_rows"] = len(holdout_quality)
    decision_audit["holdout_economic_universe_rows"] = len(holdout_all)
    decision_audit["holdout_exact_cutoff_label_releases"] = exact_cutoff_labels
    decision_audit["cutoff_rule"] = "exclusive; labels/releases at exactly 2026-10-07T00:00:00Z are excluded"

    quality = _quality_tables(
        {"development_oof": dev_quality, "holdout_historical_diagnostic": holdout_quality},
        {"development_oof": dev_markets, "holdout_historical_diagnostic": holdout_markets.loc[
            pd.to_datetime(holdout_all.label_available_at_utc, utc=True).lt(QUALITY_CUTOFF).to_numpy()
        ].copy()},
    )
    quality_tables = dict(zip(("overall", "monthly", "statuses", "contexts"), quality))
    quality_tables["cost_associations"] = _quality_cost_associations(
        {"development_oof": dev_quality, "holdout_historical_diagnostic": holdout_quality}, side_quotes,
    )

    split_frame, tree_features, tree_info = _tree_explanation(inputs["booster"], {
        "holdout_historical_diagnostic": holdout_quality,
    })
    waterfall_summary = _scenario_summaries(waterfall)
    holdout_subthreshold = intents.loc[
        intents.period.eq("holdout_historical_diagnostic") & intents.economic_status.eq("positive_ev_below_1pct")
    ].copy()
    if len(holdout_subthreshold) != 37:
        raise AssertionError(f"Expected the frozen 37 below-hurdle cases, got {len(holdout_subthreshold)}")
    holdout_subthreshold.sort_values(["t60_ev_fraction", "market_start_utc", "condition_id"], kind="stable", inplace=True)
    example_id = str(holdout_subthreshold.iloc[len(holdout_subthreshold) // 2].condition_id)
    example_stages = waterfall.loc[waterfall.condition_id.eq(example_id)].copy()
    t60_example = example_stages.loc[example_stages.scenario.eq("t60_executable_depth")].iloc[0]
    t59_example = example_stages.loc[example_stages.scenario.eq("t59_same_intent")].iloc[0]
    example = {
        "market_start_utc": t60_example.market_start_utc, "chosen_side": t60_example.chosen_side,
        "p_D_selected_side": t60_example.p_D_selected_side,
        "p_market_selected_side": t60_example.p_market_selected_side,
        "t60_vwap": t60_example.vwap, "t60_gross_filled_usd": t60_example.filled_gross_usd,
        "t60_gross_shares": t60_example.gross_shares, "t60_net_shares": t60_example.net_shares,
        "t60_fee_shares": t60_example.fee_shares, "t60_fee_cash_usd": t60_example.fee_cash_usd,
        "t60_cash_debit_usd": t60_example.cash_debit_usd,
        "t60_expected_payout_usd": t60_example.p_D_selected_side * t60_example.net_shares,
        "t60_net_ev_usd": t60_example.net_ev_usd,
        "t60_net_ev_fraction_requested": t60_example.net_ev_fraction_requested,
        "t60_net_ev_fraction_filled_cost": t60_example.net_ev_fraction_filled_cost,
        "t60_break_even_probability": t60_example.break_even_probability,
        "t60_ev_hurdle_gap_usd": t60_example.ev_hurdle_gap_usd,
        "t60_ev_hurdle_gap_fraction": t60_example.ev_hurdle_gap_fraction,
        "t59_vwap": t59_example.vwap, "t59_fill_status": t59_example.fill_status,
    }
    example_frame = pd.DataFrame([example])
    # Attach cutoff-safe shadow T-59 outcomes to the 37 positive-but-below-hurdle cases.
    held_cases = holdout_subthreshold.copy()
    held_cases.to_csv(TABLE_DIR / "positive_ev_below_1pct_cases.csv", index=False, float_format="%.12g", na_rep="")

    # Full resolution time and PnL are descriptive for the 37, with no threshold search.
    realized = held_cases.loc[held_cases.t59_settled_before_exclusive_cutoff & held_cases.t59_hypothetical_pnl_usd.notna()]
    funnel_rows = []
    for period, markets, quality_count, intents_frame in (
        ("development_oof", dev_markets, len(dev_quality), dev_intents),
        ("holdout_historical_diagnostic", holdout_markets, len(holdout_quality), holdout_intents),
    ):
        counts = markets.economic_status.value_counts().to_dict()
        t59_intents = intents_frame.loc[intents_frame.t59_priceable]
        funnel_rows.append({
            "period": period, "population": len(markets), "markets": len(markets),
            "quality_label_rows": quality_count,
            "no_priceable_t60_side": int(counts.get("no_priceable_t60_side", 0)),
            "nonpositive_net_ev": int(counts.get("nonpositive_net_ev", 0)),
            "positive_ev_below_1pct": int(counts.get("positive_ev_below_1pct", 0)),
            "at_or_above_1pct": int(counts.get("at_or_above_1pct", 0)),
            "positive_intents_priced_at_t59": int(len(intents_frame)),
            "t59_priceable": int(intents_frame.t59_priceable.sum()) if len(intents_frame) else 0,
            "t59_partial_fills": int(intents_frame.t59_partial_fill.fillna(False).sum()) if len(intents_frame) else 0,
            "t59_resolved_by_exclusive_cutoff": int(intents_frame.t59_settled_before_exclusive_cutoff.sum()) if len(intents_frame) else 0,
        })
    funnel = pd.DataFrame(funnel_rows)

    side_reasons = side_quotes.loc[
        side_quotes.period.eq("holdout_historical_diagnostic") & ~side_quotes.priceable_t60
    ].groupby(["side", "reason_t60"], dropna=False).size().rename("side_rows").reset_index()
    side_reasons.to_csv(TABLE_DIR / "t60_unpriceable_side_reasons.csv", index=False)
    _write_csv(funnel, TABLE_DIR / "decision_funnel.csv")
    side_quotes.to_parquet(DIAGNOSTIC_DIR / "all_market_t60_two_sides.parquet", index=False, compression="zstd")
    market_summaries.to_parquet(DIAGNOSTIC_DIR / "market_decisions.parquet", index=False, compression="zstd")
    waterfall.to_parquet(DIAGNOSTIC_DIR / "positive_intent_price_waterfall.parquet", index=False, compression="zstd")
    _write_csv(waterfall_summary, TABLE_DIR / "waterfall_summary.csv")
    _write_csv(quality_tables["overall"], TABLE_DIR / "quality_overall.csv")
    _write_csv(quality_tables["monthly"], TABLE_DIR / "quality_by_month.csv")
    _write_csv(quality_tables["statuses"], TABLE_DIR / "quality_by_economic_status.csv")
    _write_csv(quality_tables["contexts"], TABLE_DIR / "quality_by_existing_training_bins.csv")
    _write_csv(quality_tables["cost_associations"], TABLE_DIR / "quality_cost_associations.csv")
    _write_csv(split_frame, TABLE_DIR / "tree_splits.csv")
    _write_csv(tree_features, TABLE_DIR / "tree_feature_gain.csv")
    _write_csv(pd.DataFrame(tree_info["feature_contributions"]), TABLE_DIR / "feature_contributions.csv")
    _write_csv(fee_table := _fee_semantics_table(), TABLE_DIR / "fee_and_order_semantics.csv")

    # Stages are kept as full records in the ignored parquet; this compact CSV supports exact review.
    waterfall.to_csv(TABLE_DIR / "positive_intent_waterfall.csv", index=False, float_format="%.12g", na_rep="")
    quality_summary = {
        "overall": quality_tables["overall"].to_dict(orient="records"),
        "by_month": quality_tables["monthly"].to_dict(orient="records"),
        "by_economic_status": quality_tables["statuses"].to_dict(orient="records"),
        "by_existing_training_bins": quality_tables["contexts"].to_dict(orient="records"),
    }
    summary = {
        "status": "completed_read_only_diagnostic", "source_hashes": inputs["source_hashes"],
        "development_oof_rows": int(len(dev_quality)), "development_full_quality_sample_rows": int(inputs["results"]["sample"]["periods"][0]["markets"]),
        "holdout_economic_rows": int(len(holdout_all)), "holdout_quality_rows": int(len(holdout_quality)),
        "holdout_exact_cutoff_label_releases": exact_cutoff_labels,
        "holdout_funnel": {str(k): int(v) for k, v in holdout_counts.items()},
        "decision_audit": decision_audit,
        "t59_positive_below_hurdle": {
            "intents": int(len(held_cases)), "priceable_fills": int(held_cases.t59_priceable.sum()),
            "partial_fills": int(held_cases.t59_partial_fill.fillna(False).sum()),
            "settled_before_exclusive_cutoff": int(held_cases.t59_settled_before_exclusive_cutoff.sum()),
            "descriptive_realized_pnl_usd": float(realized.t59_hypothetical_pnl_usd.sum()) if len(realized) else 0.0,
            "realized_fills_used": int(len(realized)),
            "month_counts": held_cases.month_utc.value_counts().sort_index().to_dict(),
            "ev_fraction_distribution": _distribution(held_cases.t60_ev_fraction),
            "net_ev_usd_distribution": _distribution(held_cases.t60_net_ev_usd),
            "probability_minus_break_even_distribution": _distribution(held_cases.t60_probability_minus_break_even),
            "distributions": {
                "selected_side_D_probability": _distribution(held_cases.p_D_selected_side),
                "p_break_even_T60": _distribution(held_cases.t60_break_even_probability),
                "p_D_minus_p_break_even_T60": _distribution(held_cases.t60_probability_minus_break_even),
                "T60_net_EV_USD": _distribution(held_cases.t60_net_ev_usd),
                "T60_EV_fraction_requested": _distribution(held_cases.t60_net_ev_fraction_requested),
                "T60_cash_debit_USD": _distribution(held_cases.t60_cash_debit_usd),
                "T60_fee_equivalent_USD": _distribution(held_cases.t60_fee_usd_equivalent),
                "T60_VWAP": _distribution(held_cases.t60_vwap),
                "T59_cash_debit_USD_shadow": _distribution(held_cases.t59_cash_debit_usd),
                "T59_VWAP_shadow": _distribution(held_cases.t59_vwap),
            },
        },
        "quality": quality_summary, "tree": tree_info["reconstruction"],
        "tree_count": len(split_frame), "split_count": len(split_frame),
        "center_constant": tree_info["reconstruction"]["center_constant"],
        "max_tree_reconstruction_error": max(item["max_absolute_reconstruction_error"] for key, item in tree_info["reconstruction"].items() if key != "center_constant" and key != "centering"),
        "reproduction": "python run_btc_conditional_market_offset_diagnostics_20261018.py",
    }
    summary["heldout_shadow_case_pnl"] = {
        "pnl": float(realized.t59_hypothetical_pnl_usd.sum()) if len(realized) else 0.0,
        "settled_fills": int(len(realized)), "not_a_strategy": True,
    }
    (DIAGNOSTIC_DIR / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")
    runtime = {"elapsed_seconds": time.perf_counter() - started}
    summary["runtime"] = runtime
    summary["generated_diagnostic_bytes"] = sum(path.stat().st_size for path in DIAGNOSTIC_DIR.rglob("*") if path.is_file())
    (DIAGNOSTIC_DIR / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")
    report = _build_report(
        summary, quality_tables, funnel, waterfall_summary, fee_table, tree_features,
        pd.DataFrame(tree_info["feature_contributions"]), example_frame,
        inputs["source_hashes"], runtime,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": summary["status"], "report": REPORT_PATH.resolve().relative_to(ROOT).as_posix(),
        "runtime_seconds": round(runtime["elapsed_seconds"], 1), "diagnostic_bytes": summary["generated_diagnostic_bytes"],
        "holdout_funnel": summary["holdout_funnel"],
        "positive_below_hurdle": summary["t59_positive_below_hurdle"],
        "tree_splits": tree_features.to_dict(orient="records"),
    }, indent=2, default=str), flush=True)
    return summary


if __name__ == "__main__":
    main()
