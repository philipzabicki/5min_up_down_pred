from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

import run_btc_preopen_advanced_policy as advanced
import run_btc_preopen_economic_replay as replay
import run_btc_preopen_policy_optimization as ledger


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports/btc_preopen"
CONFIG_PATH = ROOT / "configs/research/btc_preopen_policy_continuation_20261007.json"
SOURCE_PATH = REPORT_DIR / "advanced_policy_study_20261007.json"
STATE_V2_PATH = REPORT_DIR / "advanced_policy_state_v2_20261007.json"
V1_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/advanced_policy_20261007"
V2_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/advanced_policy_state_v2_20261007"
KACHO_DIR = ROOT / "data/raw/polymarket/kachoio/42d917dc8e3205dde8ac909792af0cce2d715c9f"
PMXT_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/pmxt_v2_archive_full_scenarios/event_parts"
ETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
DECISION_OFFSETS = tuple(range(-54, 0, 5)) + tuple(range(1, 300, 5))
PRESTART_OFFSETS = tuple(range(-54, 0, 5))
POSTSTART_OFFSETS = tuple(range(1, 300, 5))
FEE_EXPONENT = 1.0
MAX_BID_AGE_SECONDS = 30.0
INITIAL_CASH_USD = 100.0
FIXED_GROSS_USD = 5.0
MAX_GROSS_USD = 20.0


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summary_row(summary: dict, policy_id: str) -> dict:
    for row in summary["common_outer_comparison"]:
        if row["strategy"] == policy_id:
            return row
    raise KeyError(policy_id)


def _state_impact() -> dict:
    old = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    new = json.loads(STATE_V2_PATH.read_text(encoding="utf-8"))
    old_decisions = pd.read_parquet(V1_DIR / "decisions.parquet")
    new_decisions = pd.read_parquet(V2_DIR / "decisions.parquet")
    policy_ids = [row["strategy"] for row in old["common_outer_comparison"]]
    output = []
    for policy_id in policy_ids:
        before = _summary_row(old, policy_id)
        after = _summary_row(new, policy_id)
        a = old_decisions.loc[old_decisions.policy_id == policy_id].set_index("condition_id")
        b = new_decisions.loc[new_decisions.policy_id == policy_id].set_index("condition_id")
        common_ids = a.index.intersection(b.index)
        changed = 0
        for condition_id in common_ids:
            left, right = a.loc[condition_id], b.loc[condition_id]
            left_side, right_side = str(left.get("chosen_side")), str(right.get("chosen_side"))
            left_amount = float(left.get("feasible_gross_usd", 0.0) or 0.0)
            right_amount = float(right.get("feasible_gross_usd", 0.0) or 0.0)
            changed += int(left_side != right_side or abs(left_amount - right_amount) > 0.005)
        locked = new_decisions.loc[new_decisions.policy_id == policy_id]
        known = pd.to_numeric(locked.get("known_outcome_locked_position_count", 0), errors="coerce").fillna(0)
        rejected = 0
        if policy_id.startswith("rck_"):
            for col in ("up_rejection_reason", "down_rejection_reason"):
                if col in locked:
                    rejected += int(locked[col].astype(str).str.contains("risk_constraint", case=False, regex=False).sum())
        output.append({
            "policy_id": policy_id,
            "v1": {key: before[key] for key in ("pnl_usd", "max_drawdown_cost_basis", "trades", "max_open_cost_basis_usd")},
            "state_v2": {key: after[key] for key in ("pnl_usd", "max_drawdown_cost_basis", "trades", "max_open_cost_basis_usd")},
            "changed_decisions": changed,
            "decision_moments_with_known_locked_payout": int((known > 0).sum()),
            "known_locked_position_moments_total": int(known.sum()),
            "rck_candidate_rejections": rejected,
        })
    return {
        "status": "completed",
        "v1_report": "reports/btc_preopen/advanced_policy_study_20261007.json",
        "corrected_report": "reports/btc_preopen/advanced_policy_state_v2_20261007.json",
        "policies": output,
        "conclusion": "The corrected known-outcome state has zero decision moments with known-but-locked positions in this schedule; policy decisions, trades, PnL, drawdown and cost-basis exposure therefore match v1.",
    }


def _source_side(row: dict) -> str | None:
    p = float(row["p_candidate_platt"])
    up = ledger._exact_reference_fill(row, "up")
    down = ledger._exact_reference_fill(row, "down")
    ev_up = p * up["net_shares"] - up["cash_debit_usd"]
    ev_down = (1.0 - p) * down["net_shares"] - down["cash_debit_usd"]
    if max(ev_up, ev_down) <= 0.0:
        return None
    return "up" if ev_up >= ev_down else "down"


def _kelly_request(row: dict, side: str, probability: float, base_config: dict) -> float:
    payout, debit = ledger._net_unit_rates(row, side, base_config)
    edge, win_profit = probability * payout - debit, payout - debit
    if edge <= 0.0 or win_profit <= 0.0:
        return 0.0
    return min(edge / (debit * win_profit) * INITIAL_CASH_USD, MAX_GROSS_USD)


def _pair_method_order(
    row: dict, side: str, method: str, pred: dict, base_config: dict, config: dict,
    z_bank: np.ndarray, rho: float,
) -> tuple[dict, float]:
    state = ledger.Portfolio(cash=INITIAL_CASH_USD)
    fill5 = ledger._feasible_order(row, side, FIXED_GROSS_USD, state, base_config)
    if method == "fixed_5":
        return fill5, FIXED_GROSS_USD
    if method == "fractional_kelly":
        params = {"kelly_multiplier": 0.5, "max_cost_basis_equity_fraction": 0.1, "max_gross_stake_usd": MAX_GROSS_USD, "min_expected_return": 0.0}
        request, _ = ledger._sized_request(row, side, "fractional_kelly", params, state, base_config)
        request = min(float(request), MAX_GROSS_USD)
    else:
        calibration = "btc_market" if "btc_market" in method or method.startswith("rck") or method == "portfolio_kelly_independent" else "btc_only"
        p_up = float(pred[f"p_{calibration}"])
        p_low = min(float(pred[f"p_{calibration}_low"]), p_up)
        p_high = max(float(pred[f"p_{calibration}_high"]), p_up)
        robust = method.startswith("drk")
        lam = None
        dependent = True
        if method.startswith("rck"):
            parts = method.split("_")
            alpha = float(parts[2])
            profile = next(item for item in config["rck"]["profiles"] if abs(float(item["alpha"]) - alpha) < 1e-9)
            lam = float(profile["lambda"])
            dependent = parts[-1] != "independent"
        elif method == "portfolio_kelly_independent":
            dependent = False
        elif method == "point_kelly_btc_market_independent":
            dependent = False
            calibration = "btc_market"
            p_up = float(pred["p_btc_market"])
            p_low = min(float(pred["p_btc_market_low"]), p_up)
            p_high = max(float(pred["p_btc_market_high"]), p_up)
        candidate_row = {**row, "_p_up_high": p_high}
        choice = advanced._choose_candidate(
            candidate_row, side, state, base_config, [], p_up=p_up, p_up_low=p_low,
            robust=robust, dependent=dependent, rho=rho, z_bank=z_bank, lam=lam,
        )
        request = float(choice.get("requested_gross_usd", 0.0))
        if choice.get("skip_reason") is not None and float(choice.get("admitted_gross_usd", 0.0)) <= 0:
            return {"admitted_gross_usd": 0.0, "skip_reason": choice.get("skip_reason"), "limit_reasons": []}, request
    fill = ledger._feasible_order(row, side, request, state, base_config)
    return fill, float(request)


def _fill_pnl(row: dict, side: str, fill: dict) -> float | None:
    if float(fill.get("admitted_gross_usd", 0.0)) <= 0.0:
        return None
    won = int(row["target_polymarket_up"]) == int(side == "up")
    return (float(fill["net_shares"]) if won else 0.0) - float(fill["cash_debit_usd"])


def _paired_sizing(rows: list[dict], folds: list[dict], predictions: dict, fold_meta: list[dict], base_config: dict, config: dict, z_bank: np.ndarray) -> dict:
    pairs = [
        ("point_kelly_btc_only", "drk_btc_only"),
        ("point_kelly_btc_market", "drk_btc_market"),
        ("point_kelly_btc_market", "rck_alpha_0.5_correlated"),
        ("point_kelly_btc_market", "rck_alpha_0.7_correlated"),
        ("point_kelly_btc_market", "rck_alpha_0.8_correlated"),
        ("portfolio_kelly_independent", "rck_alpha_0.5_independent"),
        ("portfolio_kelly_independent", "rck_alpha_0.7_independent"),
        ("portfolio_kelly_independent", "rck_alpha_0.8_independent"),
        ("fractional_kelly", "fixed_5"),
    ]
    results = []
    outer_start = int(folds[0]["evaluation"]["start_index_in_eligible_order"])
    for first, second in pairs:
        pair_started = time.perf_counter()
        print(f"paired sizing {first} vs {second}: start", flush=True)
        records = []
        for fold_number, fold in enumerate(folds, start=1):
            start = int(fold["evaluation"]["start_index_in_eligible_order"])
            stop = int(fold["evaluation"]["end_index_exclusive"])
            rho = float(fold_meta[fold_number - 1]["residual_dependence"]["shrunk_lag_one_correlation"])
            for row in rows[start:stop]:
                side = _source_side(row)
                if side is None:
                    continue
                prediction = predictions[fold_number][row["condition_id"]]
                one = _pair_method_order(row, side, first, prediction, base_config, config, z_bank, rho)
                two = _pair_method_order(row, side, second, prediction, base_config, config, z_bank, rho)
                five = ledger._feasible_order(row, side, FIXED_GROSS_USD, ledger.Portfolio(cash=INITIAL_CASH_USD), base_config)
                pnl1, pnl2, pnl5 = _fill_pnl(row, side, one[0]), _fill_pnl(row, side, two[0]), _fill_pnl(row, side, five)
                records.append({
                    "condition_id": row["condition_id"], "market_start_utc": str(row["market_start_utc"]),
                    "side": side, "first_fill": one[0], "first_requested": one[1], "second_fill": two[0],
                    "second_requested": two[1], "fixed5_fill": five,
                    "first_pnl": pnl1, "second_pnl": pnl2, "fixed5_pnl": pnl5,
                })
        common = [record for record in records if record["first_pnl"] is not None and record["second_pnl"] is not None]
        only_first = [record for record in records if record["first_pnl"] is not None and record["second_pnl"] is None]
        only_second = [record for record in records if record["first_pnl"] is None and record["second_pnl"] is not None]
        neither = [record for record in records if record["first_pnl"] is None and record["second_pnl"] is None]
        def reasons(items: list[dict], field: str) -> dict:
            counts = Counter(str(item[field].get("skip_reason") or "not_feasible") for item in items)
            return dict(counts)
        common_first = float(sum(item["first_pnl"] for item in common))
        common_second = float(sum(item["second_pnl"] for item in common))
        results.append({
            "first": first, "second": second, "frozen_population_markets": len(rows) - outer_start,
            "candidate_positive_ev_market_side_times": len(records),
            "common_feasible": len(common), "first_only": len(only_first), "second_only": len(only_second),
            "rejected_by_both": len(neither),
            "common_first_pnl_usd": common_first, "common_second_pnl_usd": common_second,
            "paired_pnl_difference_first_minus_second_usd": common_first - common_second,
            "common_first_admitted_gross_usd": float(sum(float(x["first_fill"]["admitted_gross_usd"]) for x in common)),
            "common_second_admitted_gross_usd": float(sum(float(x["second_fill"]["admitted_gross_usd"]) for x in common)),
            "common_fixed5_feasible": sum(x["fixed5_pnl"] is not None for x in common),
            "common_fixed5_pnl_usd": float(sum(x["fixed5_pnl"] or 0.0 for x in common)),
            "common_fixed5_admitted_gross_usd": float(sum(float(x["fixed5_fill"].get("admitted_gross_usd", 0.0)) for x in common)),
            "first_only_pnl_usd": float(sum(x["first_pnl"] or 0.0 for x in only_first)),
            "second_only_pnl_usd": float(sum(x["second_pnl"] or 0.0 for x in only_second)),
            "first_only_rejections_second": reasons(only_first, "second_fill"),
            "second_only_rejections_first": reasons(only_second, "first_fill"),
            "both_rejection_reasons_first": reasons(neither, "first_fill"),
            "both_rejection_reasons_second": reasons(neither, "second_fill"),
            "independent_trade_capital_usd": INITIAL_CASH_USD,
            "capital_method": "Each order is independently financed from a fresh $100 state; no shared cash path or reinvestment. This diagnostic is not one $100 portfolio return.",
        })
        print(f"paired sizing {first} vs {second}: done in {time.perf_counter()-pair_started:.1f}s; common={len(common)}", flush=True)
    return {"status": "completed", "pairs": results}


def _internal_eta_predictions(rows: list[dict], fold: dict, config: dict, fold_number: int) -> tuple[list[dict], dict, dict]:
    valid_start = int(fold["selection_validation"]["start_index_in_eligible_order"])
    outer_start = int(fold["evaluation"]["start_index_in_eligible_order"])
    valid_start_ns = int(rows[valid_start]["_entry_ns"])
    refit_ns = int(pd.Timestamp(fold["refit_at_utc"]).value)
    train = advanced._known_before(rows[:valid_start], valid_start_ns)
    valid = advanced._known_before(rows[valid_start:outer_start], refit_ns)
    block_days = int(config["calibration"]["bootstrap"]["block_length_days"])
    min_blocks = int(config["calibration"]["bootstrap"]["minimum_nonempty_blocks"])
    replicates = int(config["calibration"]["bootstrap"]["replicates"])
    blocks = len(set(int(row["_market_ns"]) // (block_days * advanced.DAY_NS) for row in train))
    outputs = {}
    expansion = 0
    fallback = len(train) < 100 or len(valid) < 25 or blocks < min_blocks
    if fallback:
        for row in valid:
            p = float(row["p_candidate_platt"])
            outputs[row["condition_id"]] = {"p_btc_market": p, "p_btc_market_low": p, "p_btc_market_high": p}
        return valid, outputs, {"fallback": "point_only_insufficient_prior_history", "train_rows": len(train), "validation_rows": len(valid), "prior_three_day_blocks": blocks, "interval_expansions": 0, "bootstrap_models_completed": 0}
    x_train = advanced._feature_matrix(train, "btc_market")
    y_train = np.asarray([int(row["target_polymarket_up"]) for row in train], dtype=int)
    x_valid = advanced._feature_matrix(valid, "btc_market")
    fallback_p = np.asarray([float(row["p_candidate_platt"]) for row in valid], dtype=float)
    # C=1.0 is predeclared here; it is not selected on the same validation block used for eta.
    model = advanced._fit_logistic(x_train, y_train, 1.0)
    center = advanced._predict(model, x_valid, fallback_p)
    rng = np.random.default_rng(int(config["calibration"]["bootstrap"]["seed"]) + 1000 + fold_number)
    boot = []
    for _ in range(replicates):
        sample, _ = advanced._bootstrap_sample_indices(train, rng, block_days)
        fitted = advanced._fit_logistic(x_train[sample], y_train[sample], 1.0)
        if fitted is not None:
            boot.append(advanced._predict(fitted, x_valid, fallback_p))
    if boot:
        lower_q, upper_q = config["calibration"]["bootstrap"]["interval_quantiles"]
        low = np.quantile(np.asarray(boot), float(lower_q), axis=0)
        high = np.quantile(np.asarray(boot), float(upper_q), axis=0)
    else:
        low, high = center.copy(), center.copy()
    for index, row in enumerate(valid):
        p = float(center[index])
        raw_low, raw_high = float(low[index]), float(high[index])
        expansion += int(raw_low > p or raw_high < p)
        outputs[row["condition_id"]] = {
            "p_btc_market": p, "p_btc_market_low": min(raw_low, p), "p_btc_market_high": max(raw_high, p),
        }
    return valid, outputs, {"fallback": None, "train_rows": len(train), "validation_rows": len(valid), "prior_three_day_blocks": blocks, "interval_expansions": expansion, "bootstrap_models_completed": len(boot), "fixed_C": 1.0}


def _eta_predictions(source: dict, eta: float) -> dict:
    result = {}
    for condition_id, pred in source.items():
        center = float(pred["p_btc_market"])
        low = min(float(pred["p_btc_market_low"]), center)
        high = max(float(pred["p_btc_market_high"]), center)
        result[condition_id] = {
            **pred,
            "p_btc_market_low": center + eta * (low - center),
            "p_btc_market_high": center + eta * (high - center),
        }
    return result


def _eta_study(rows: list[dict], folds: list[dict], outer_predictions: dict, fold_meta: list[dict], base_config: dict, config: dict, z_bank: np.ndarray) -> dict:
    internal = []
    selected = {}
    for fold_number, fold in enumerate(folds, start=1):
        print(f"DRK eta fold {fold_number}/{len(folds)}: fitting causal internal calibrator", flush=True)
        validation_rows, predictions, meta = _internal_eta_predictions(rows, fold, config, fold_number)
        if not validation_rows:
            raise RuntimeError(f"Fold {fold_number} has no internal validation rows for eta selection")
        synthetic_fold = [{"fold_id": fold_number, "evaluation": {
            "start_index_in_eligible_order": 0, "end_index_exclusive": len(validation_rows),
            "decision_start_utc": validation_rows[0]["entry_time_utc"],
            "decision_end_utc": validation_rows[-1]["entry_time_utc"],
        }}]
        rho = advanced.estimate_residual_correlation(
            advanced._known_before(rows[:int(fold["selection_validation"]["start_index_in_eligible_order"])], int(rows[int(fold["selection_validation"]["start_index_in_eligible_order"])]["_entry_ns"])),
            float(config["scenario_model"]["correlation_shrinkage_n"]),
        )
        meta["residual_dependence"] = rho
        scores = []
        for eta in ETA_GRID:
            spec = {"policy_id": f"internal_drk_eta_{eta:g}", "kind": "drk", "calibration": "btc_market", "robust": True, "dependent": True, "lambda": None, "uncertainty_eta": eta}
            sim, _, _, _ = advanced._simulate_advanced(
                validation_rows, synthetic_fold, {1: _eta_predictions(predictions, eta)}, [meta],
                spec, base_config, config, z_bank,
            )
            scores.append({"eta": eta, "mean_daily_log_growth": sim["mean_daily_log_growth"], "pnl_usd": sim["net_pnl_usd"], "trades": sim["trade_count"]})
        best_score = max(float(item["mean_daily_log_growth"]) for item in scores)
        choice = min((item for item in scores if abs(float(item["mean_daily_log_growth"]) - best_score) <= 1e-12), key=lambda item: float(item["eta"]))
        selected[fold_number] = float(choice["eta"])
        internal.append({"fold_id": fold_number, "validation_start_utc": fold["selection_validation"]["decision_start_utc"], "validation_end_utc": fold["selection_validation"]["decision_end_utc"], **meta, "eta_scores": scores, "selected_eta": float(choice["eta"]), "tie_rule": "smallest eta"})
    spec_base = {"policy_id": "drk_eta", "kind": "drk", "calibration": "btc_market", "robust": True, "dependent": True, "lambda": None}
    outputs = {}
    for label, spec in (("point_eta0", {**spec_base, "uncertainty_eta": 0.0}), ("full_interval_eta1", {**spec_base, "uncertainty_eta": 1.0}), ("selected_eta", {**spec_base, "eta_by_fold": selected})):
        summary, decisions, trades, blocks = advanced._simulate_advanced(rows, folds, outer_predictions, fold_meta, spec, base_config, config, z_bank)
        outputs[label] = {"summary": summary, "decisions": decisions, "trades": trades, "folds": blocks}
    selected_expansions = []
    for number, fold in enumerate(folds, start=1):
        start = int(fold["evaluation"]["start_index_in_eligible_order"])
        stop = int(fold["evaluation"]["end_index_exclusive"])
        expansion = 0
        for row in rows[start:stop]:
            pred = outer_predictions[number][row["condition_id"]]
            p = float(pred["p_btc_market"])
            expansion += int(float(pred["p_btc_market_low"]) > p or float(pred["p_btc_market_high"]) < p)
        selected_expansions.append({"fold_id": number, "interval_expansions": expansion, "selected_eta": selected[number]})
    return {
        "status": "completed", "eta_grid": list(ETA_GRID), "internal_selection": internal,
        "selected_eta_by_fold": selected, "outer_interval_expansions": selected_expansions,
        "outer": {key: {"summary": value["summary"], "folds": value["folds"], "trade_count": len(value["trades"])} for key, value in outputs.items()},
        "outer_artifacts": {key: {"decisions": value["decisions"], "trades": value["trades"]} for key, value in outputs.items()},
        "interpretation": "Development-period walk-forward; each eta was selected only on the immediately preceding internal validation block. The internal calibrator used fixed C=1.0 and prior labels only. This is not an independent holdout.",
    }


def _seed_pmxt_state(row: dict) -> dict:
    state = replay._new_state()
    for side, token_key in (("up", "up_token_id"), ("down", "down_token_id")):
        token = replay._token_state(state, row[token_key])
        bid, ask = float(row[f"{side}_best_bid"]), float(row[f"{side}_best_ask"])
        token.update({
            # Keep the entry prices only to satisfy the replay book invariant. Zero
            # capacity ensures neither snapshot side becomes executable; fresh prices
            # and sizes come from timestamped PMXT events below.
            "bids": {round(bid, 8): 0.0}, "asks": {round(ask, 8): 0.0},
            "initialized": True, "book_source_ns": None,
            "last_source_ns": int(row["_entry_ns"]), "book_receive_ns": None,
            # The saved entry snapshot has no bid update timestamp or bid size. Do not
            # turn it into a fresh observation at later decision times.
            "bid_update_ns": None, "ask_update_ns": None,
            "bid_source_update_ns": None, "ask_source_update_ns": None,
            "bid_book_update_ns": None, "ask_book_update_ns": None,
            "bid_book_source_ns": None, "ask_book_source_ns": int(row["_entry_ns"]),
        })
    return state


def _pmxt_premarket_quotes(rows: list[dict]) -> dict[str, dict[int, dict]]:
    """Replay only PMXT receive events between the frozen T-59 entry and market start."""
    if not rows:
        return {}
    rows_by_id = {str(row["condition_id"]): row for row in rows}
    output: dict[str, dict[int, dict]] = {key: {} for key in rows_by_id}
    states = {key: _seed_pmxt_state(row) for key, row in rows_by_id.items()}
    next_offset = {key: 0 for key in rows_by_id}
    start_ns = min(int(row["_entry_ns"]) for row in rows)
    end_ns = max(int(row["_market_ns"]) for row in rows)
    first_hour = pd.Timestamp(start_ns, unit="ns", tz="UTC").floor("h")
    last_hour = pd.Timestamp(end_ns, unit="ns", tz="UTC").floor("h")
    hours = pd.date_range(first_hour, last_hour, freq="h", tz="UTC")
    start_records = sorted((int(row["_market_ns"]), key) for key, row in rows_by_id.items())
    ordered_starts = np.asarray([item[0] for item in start_records], dtype=np.int64)
    ordered_ids = [item[1] for item in start_records]
    use_cols = ["market", "timestamp_received", "timestamp", "event_type", "asset_id", "bids", "asks", "price", "size", "side", "best_bid", "best_ask", "fee_rate_bps"]

    def record_due(condition_id: str, through_ns: int) -> None:
        row = rows_by_id[condition_id]
        while next_offset[condition_id] < len(PRESTART_OFFSETS):
            offset = PRESTART_OFFSETS[next_offset[condition_id]]
            decision_ns = int(row["_market_ns"]) + offset * 1_000_000_000
            if decision_ns > through_ns:
                break
            sides = {}
            for side, token_col, opposite_col in (("up", "up_token_id", "down_token_id"), ("down", "down_token_id", "up_token_id")):
                token_id = str(row[token_col])
                opposite_id = str(row[opposite_col])
                book = replay._effective_book(states[condition_id], token_id, opposite_id)
                source = book.get("source_token_id") if book else None
                source_token = states[condition_id]["tokens"].get(str(source)) if source else None
                bid_age = None
                if source_token and source_token.get("reported_receive_ns") is not None:
                    bid_age = max(0.0, (decision_ns - int(source_token["reported_receive_ns"])) / 1e9)
                direct = source == token_id
                reported_bid = _num(source_token.get("reported_best_bid")) if source_token else None
                reported_ask = _num(source_token.get("reported_best_ask")) if source_token else None
                if reported_bid is not None and reported_ask is not None and direct:
                    bid = reported_bid
                    ask = reported_ask
                    bid_size = source_token["bids"].get(round(bid, 8)) if source_token and bid is not None else None
                    ask_size = source_token["asks"].get(round(ask, 8)) if source_token and ask is not None else None
                elif reported_bid is not None and reported_ask is not None:
                    bid = 1.0 - reported_ask if reported_ask is not None else None
                    ask = 1.0 - reported_bid if reported_bid is not None else None
                    bid_size = source_token["asks"].get(round(reported_ask, 8)) if source_token and reported_ask is not None else None
                    ask_size = source_token["bids"].get(round(reported_bid, 8)) if source_token and reported_bid is not None else None
                elif source_token and source_token.get("book_source_ns") is not None:
                    top = replay._book_top(book)
                    if top is None:
                        bid = ask = bid_size = ask_size = None
                    else:
                        bid, ask, bid_size, ask_size = top
                        bid_update_ns = book.get("bid_update_ns")
                        if bid_update_ns is not None:
                            bid_age = max(0.0, (decision_ns - int(bid_update_ns)) / 1e9)
                else:
                    bid = ask = bid_size = ask_size = None
                sides[side] = {
                    "bid": bid,
                    "ask": ask,
                    "bid_size": float(bid_size) if bid_size is not None else None,
                    "ask_size": float(ask_size) if ask_size is not None else None,
                    "bid_age_seconds": bid_age,
                    "quote_timestamp_ns": decision_ns,
                    "source": "pmxt_event_replay",
                }
            output[condition_id][offset] = sides
            next_offset[condition_id] += 1

    replay_started = time.perf_counter()
    for hour_index, hour in enumerate(hours, start=1):
        path = PMXT_DIR / f"{hour.strftime('%Y-%m-%dT%H')}.parquet"
        if not path.is_file():
            if hour_index % 12 == 0 or hour_index == len(hours):
                print(f"PMXT pre-start replay {hour_index}/{len(hours)} hours; missing hour file", flush=True)
            continue
        hour_ns = int(hour.value)
        id_start = int(np.searchsorted(ordered_starts, hour_ns, side="right"))
        id_stop = int(np.searchsorted(ordered_starts, hour_ns + 3600_000_000_000 + 59_000_000_000, side="left"))
        hour_ids = ordered_ids[id_start:id_stop]
        if not hour_ids:
            continue
        table = pq.read_table(path, columns=use_cols)
        market_type = table.schema.field("market").type
        values = pa.array([key.encode("ascii") for key in hour_ids], type=market_type)
        mask = pc.is_in(table.column("market"), value_set=values)
        table = table.filter(mask)
        if table.num_rows == 0:
            continue
        frame = table.to_pandas()
        frame["market"] = frame["market"].map(lambda value: value.decode("ascii") if isinstance(value, bytes) else str(value))
        frame["_received_ns"] = pd.to_datetime(frame["timestamp_received"], utc=True).dt.as_unit("ns").astype("int64")
        frame["_source_ns"] = pd.to_datetime(frame["timestamp"], utc=True).dt.as_unit("ns").astype("int64")
        frame["_entry_ns"] = frame["market"].map(lambda key: int(rows_by_id[key]["_entry_ns"]))
        frame["_start_ns"] = frame["market"].map(lambda key: int(rows_by_id[key]["_market_ns"]))
        frame = frame.loc[(frame["_received_ns"] > frame["_entry_ns"]) & (frame["_received_ns"] < frame["_start_ns"])]
        if frame.empty:
            continue
        frame.sort_values(["_received_ns", "_source_ns", "market", "asset_id", "event_type"], kind="stable", inplace=True)
        for condition_id, market_frame in frame.groupby("market", sort=False):
            row = rows_by_id[condition_id]
            up_token, down_token = str(row["up_token_id"]), str(row["down_token_id"])
            for received_ns, group in market_frame.groupby("_received_ns", sort=False):
                # A whole-second decision between receive events must use the state
                # before the next event, not that event's later sub-second update.
                record_due(condition_id, int(received_ns) - 1)
                replay._process_receive_group(states[condition_id], group, int(received_ns), up_token, down_token)
                record_due(condition_id, int(received_ns))
        if hour_index % 12 == 0 or hour_index == len(hours):
            elapsed = time.perf_counter() - replay_started
            remaining = elapsed / hour_index * (len(hours) - hour_index) if hour_index else 0.0
            print(f"PMXT pre-start replay {hour_index}/{len(hours)} hours; elapsed={elapsed:.1f}s; ETA={remaining:.1f}s", flush=True)
    for condition_id, row in rows_by_id.items():
        record_due(condition_id, int(row["_market_ns"]) - 1)
    return output


def _load_trajectories(rows: list[dict]) -> tuple[dict, dict]:
    ids = [str(row["condition_id"]) for row in rows]
    tick_path = KACHO_DIR / "btc_ticks.parquet"
    market_path = KACHO_DIR / "btc_markets.parquet"
    market_meta = pd.read_parquet(market_path, columns=["condition_id", "token_up", "token_down"])
    market_meta["condition_id"] = market_meta["condition_id"].astype(str)
    token_map = market_meta.set_index("condition_id")[["token_up", "token_down"]].to_dict("index")
    for row in rows:
        tokens = token_map.get(str(row["condition_id"]))
        if tokens is None:
            raise RuntimeError(f"Kacho token mapping is missing for {row['condition_id']}")
        row["up_token_id"], row["down_token_id"] = str(tokens["token_up"]), str(tokens["token_down"])
    columns = ["condition_id", "t", "bu", "au", "bd", "ad", "su", "sd", "sau", "sad"]
    # Read only the compact top-of-book columns once, then filter in memory. Passing
    # thousands of IDs as a Parquet predicate is pathologically slow on this file.
    ticks = pd.read_parquet(tick_path, columns=columns)
    ticks["condition_id"] = ticks["condition_id"].astype(str)
    row_map = {str(row["condition_id"]): row for row in rows}
    ticks = ticks.loc[ticks["condition_id"].isin(row_map)]
    ticks.sort_values(["condition_id", "t"], kind="stable", inplace=True)
    tick_groups = {key: part.set_index("t", drop=False) for key, part in ticks.groupby("condition_id", sort=False)}
    pre = _pmxt_premarket_quotes(rows)
    quotes, coverage = {}, []
    for condition_id, row in row_map.items():
        market_start = pd.Timestamp(row["market_start_utc"])
        start_s = int(market_start.timestamp())
        market_ticks = tick_groups.get(condition_id)
        item_quotes = {}
        tick_rows = 0 if market_ticks is None else int(len(market_ticks))
        if market_ticks is not None:
            for offset in range(300):
                stamp = start_s + offset
                if stamp not in market_ticks.index:
                    continue
                tick = market_ticks.loc[stamp]
                if isinstance(tick, pd.DataFrame):
                    tick = tick.iloc[-1]
                item_quotes[offset] = {
                    "up": {"bid": _num(tick["bu"]), "ask": _num(tick["au"]), "bid_size": _num(tick["su"]), "ask_size": _num(tick["sau"]), "bid_age_seconds": 0.0, "source": "kacho_1hz_sample"},
                    "down": {"bid": _num(tick["bd"]), "ask": _num(tick["ad"]), "bid_size": _num(tick["sd"]), "ask_size": _num(tick["sad"]), "bid_age_seconds": 0.0, "source": "kacho_1hz_sample"},
                }
        for offset, sides in pre.get(condition_id, {}).items():
            item_quotes[int(offset)] = sides
        quotes[condition_id] = item_quotes
        post_decisions = sum(offset in item_quotes for offset in POSTSTART_OFFSETS)
        fresh_two_sided_pre = 0
        executable_pre = 0
        reference_trade = None
        reference_side = _source_side(row)
        if reference_side is not None:
            reference_fill = ledger._exact_reference_fill(row, reference_side)
            reference_trade = _trade_for_row(row, reference_side, reference_fill)
        reference_exit_pre = 0
        reference_exit_post = 0
        for offset in PRESTART_OFFSETS:
            sides = pre.get(condition_id, {}).get(offset, {})
            valid_bbo = []
            for side in ("up", "down"):
                quote = sides.get(side)
                bid, ask, age = (_num(quote.get(key)) if quote else None for key in ("bid", "ask", "bid_age_seconds"))
                valid_bbo.append(bid is not None and ask is not None and age is not None and 0.0 < bid <= ask < 1.0 and age <= MAX_BID_AGE_SECONDS)
            fresh_two_sided_pre += int(all(valid_bbo))
            executable_pre += int(any(_valid_sale_quote(sides.get(side), 1.0) for side in ("up", "down")))
            if reference_trade is not None and _valid_sale_quote(sides.get(reference_side), reference_trade["net_shares"]):
                reference_exit_pre += 1
        if reference_trade is not None:
            reference_exit_post = sum(
                _valid_sale_quote(item_quotes.get(offset, {}).get(reference_side), reference_trade["net_shares"])
                for offset in POSTSTART_OFFSETS
            )
        coverage.append({
            "condition_id": condition_id, "market_start_utc": row["market_start_utc"],
            "kacho_tick_rows": tick_rows, "poststart_decision_quotes_of_60": post_decisions,
            "pmxt_premarket_fresh_two_sided_quotes_of_11": fresh_two_sided_pre,
            "pmxt_premarket_fresh_bid_size_times_of_11": executable_pre,
            "pmxt_premarket_full_reference_position_exit_quotes": reference_exit_pre,
            "poststart_full_reference_position_exit_quotes": reference_exit_post,
            "poststart_up_bid_size_quotes_at_least_1_share": sum(_valid_sale_quote(item_quotes.get(offset, {}).get("up"), 1.0) for offset in POSTSTART_OFFSETS),
            "poststart_down_bid_size_quotes_at_least_1_share": sum(_valid_sale_quote(item_quotes.get(offset, {}).get("down"), 1.0) for offset in POSTSTART_OFFSETS),
        })
    return quotes, {"markets": coverage, "tick_path": tick_path.relative_to(ROOT).as_posix(), "pmxt_parts": len(list(PMXT_DIR.glob("*.parquet")))}


def _num(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _valid_sale_quote(quote: dict | None, shares: float) -> bool:
    if not quote:
        return False
    bid, size, age = _num(quote.get("bid")), _num(quote.get("bid_size")), _num(quote.get("bid_age_seconds"))
    return bid is not None and 0.0 < bid < 1.0 and size is not None and size + 1e-9 >= float(shares) and age is not None and age <= MAX_BID_AGE_SECONDS


def _sale_proceeds(trade: dict, quote: dict) -> float:
    bid = float(quote["bid"])
    shares = float(trade["net_shares"])
    rate = float(trade.get("fee_rate_bps", 0.0)) / 10_000.0
    if trade.get("fee_collection_mode") == "maintenance_pause":
        fee = 0.0
    elif trade.get("fee_collection_mode") == "outcome_shares":
        fee = shares * rate * min(bid, 1.0 - bid)
    else:
        fee = round(shares * rate * (bid * (1.0 - bid)) ** FEE_EXPONENT, 5)
        if fee < 0.00001:
            fee = 0.0
    return max(0.0, shares * bid - fee)


def _trade_for_row(row: dict, side: str, fill: dict, fold_id: int | None = None) -> dict:
    return {
        "condition_id": str(row["condition_id"]), "side": side,
        "net_shares": float(fill["net_shares"]), "gross_purchase_usd": float(fill["gross_usd"]),
        "cash_debit_usd": float(fill["cash_debit_usd"]),
        "entry_price": float(row[f"{side}_best_ask"]), "entry_bid": float(row[f"{side}_best_bid"]),
        "p_win": float(row["p_candidate_platt"] if side == "up" else 1.0 - row["p_candidate_platt"]),
        "target_up": int(row["target_polymarket_up"]), "market_start_ns": int(row["_market_ns"]),
        "outcome_ns": int(row["_outcome_ns"]), "release_ns": int(row["_release_ns"]),
        "fee_rate_bps": float(row.get("fee_rate_bps", 0.0)),
        "fee_collection_mode": str(row.get("fee_collection_mode", "")),
        "fold_id": fold_id,
    }


FEATURE_NAMES = ("seconds_to_close", "p_side", "bid", "ask", "spread", "bid_move_from_entry", "bid_capacity_per_share", "ask_capacity_per_share", "bid_age_seconds", "entry_price", "stake_fraction", "side_sign")


def _features(trade: dict, quote: dict | None, offset: int) -> np.ndarray | None:
    if not quote:
        return None
    bid, ask, bid_size, ask_size = (_num(quote.get(name)) for name in ("bid", "ask", "bid_size", "ask_size"))
    age = _num(quote.get("bid_age_seconds"))
    if bid is None or ask is None or age is None or not 0.0 < bid < 1.0 or not 0.0 < ask < 1.0 or bid > ask or age > MAX_BID_AGE_SECONDS:
        return None
    shares = max(float(trade["net_shares"]), 1e-12)
    entry_bid = float(trade["entry_bid"])
    return np.asarray([
        max(0.0, 300.0 - offset), float(trade["p_win"]), bid, ask, ask - bid,
        bid - entry_bid, (bid_size or 0.0) / shares, (ask_size or 0.0) / shares,
        age, float(trade["entry_price"]), float(trade["gross_purchase_usd"]) / INITIAL_CASH_USD,
        1.0 if trade["side"] == "up" else -1.0,
    ], dtype=float)


def _wealth_log(trade: dict, payout: float) -> float:
    wealth = INITIAL_CASH_USD - float(trade["cash_debit_usd"]) + float(payout)
    return math.log(wealth) if wealth > 0.0 else float("-inf")


def _train_os_model(training: list[dict], quotes: dict, kind: str) -> dict:
    future = {trade["condition_id"]: _wealth_log(trade, trade["net_shares"] if trade["target_up"] == int(trade["side"] == "up") else 0.0) for trade in training}
    models = {}
    samples_by_offset = {}
    for offset in reversed(DECISION_OFFSETS):
        x, y, records = [], [], []
        for trade in training:
            quote = quotes.get(trade["condition_id"], {}).get(offset, {}).get(trade["side"])
            features = _features(trade, quote, offset)
            if features is not None and math.isfinite(future[trade["condition_id"]]):
                x.append(features)
                y.append(future[trade["condition_id"]])
                records.append((trade, quote, features))
        model = None
        if len(x) >= 25:
            if kind == "ridge":
                model = Ridge(alpha=10.0).fit(np.asarray(x), np.asarray(y))
            else:
                model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=25, random_state=20261007).fit(np.asarray(x), np.asarray(y))
        models[offset] = model
        samples_by_offset[offset] = len(x)
        for trade in training:
            cid = trade["condition_id"]
            quote = quotes.get(cid, {}).get(offset, {}).get(trade["side"])
            features = _features(trade, quote, offset)
            if features is None:
                continue
            expected_log = _wealth_log(trade, float(trade["p_win"]) * float(trade["net_shares"]))
            continuation = float(model.predict(features.reshape(1, -1))[0]) if model is not None else expected_log
            if _valid_sale_quote(quote, trade["net_shares"]):
                sale_value = _wealth_log(trade, _sale_proceeds(trade, quote))
                if sale_value >= continuation:
                    future[cid] = sale_value
    return {"kind": kind, "models": models, "samples_by_offset": samples_by_offset, "training_markets": len(training), "feature_names": list(FEATURE_NAMES)}


def _should_sell(strategy: str, trade: dict, quote: dict | None, offset: int, models: dict) -> bool:
    if not _valid_sale_quote(quote, trade["net_shares"]):
        return False
    proceeds = _sale_proceeds(trade, quote)
    if strategy == "simple_expected_value":
        return proceeds >= float(trade["p_win"]) * float(trade["net_shares"])
    if strategy == "hold_to_resolution":
        return False
    model_bundle = models.get(trade.get("fold_id"), {}).get(strategy)
    model = model_bundle["models"].get(offset) if model_bundle else None
    features = _features(trade, quote, offset)
    if features is None:
        return False
    continuation = float(model.predict(features.reshape(1, -1))[0]) if model is not None else _wealth_log(trade, float(trade["p_win"]) * float(trade["net_shares"]))
    return _wealth_log(trade, proceeds) >= continuation


def _entry_trades(rows: list[dict]) -> dict[str, dict]:
    result = {}
    fold_lookup = {}
    for fold in []:
        pass
    for row in rows:
        side = _source_side(row)
        if side is None:
            continue
        fill = ledger._exact_reference_fill(row, side)
        if float(fill["net_shares"]) <= 0.0:
            continue
        result[str(row["condition_id"])] = _trade_for_row(row, side, fill)
    return result


def _outer_trade_records(rows: list[dict]) -> list[dict]:
    trades = pd.read_parquet(V1_DIR / "trades.parquet")
    trades = trades.loc[trades.policy_id == "legacy_exact_fixed_5"].copy()
    rows_by_id = {str(row["condition_id"]): row for row in rows}
    records = []
    for item in trades.to_dict("records"):
        cid = str(item["condition_id"])
        row = rows_by_id[cid]
        fill = {"net_shares": item["net_shares"], "gross_usd": item["gross_purchase_usd"], "cash_debit_usd": item["cash_debit_usd"]}
        trade = _trade_for_row(row, str(item["chosen_side"]), fill, int(item["fold_id"]))
        trade["actual_cash_before_usd"] = float(item["cash_before_usd"])
        records.append(trade)
    return records


def _base_config() -> dict:
    return json.loads(advanced.BASE_CONFIG_PATH.read_text(encoding="utf-8"))


def _source_study() -> dict:
    return json.loads(advanced.SOURCE_STUDY_PATH.read_text(encoding="utf-8"))


def _train_models_by_fold(rows: list[dict], folds: list[dict], all_reference_trades: dict, quotes: dict) -> dict:
    models_by_fold = {}
    for fold_number, fold in enumerate(folds, start=1):
        first = int(fold["evaluation"]["start_index_in_eligible_order"])
        refit_ns = int(pd.Timestamp(fold["refit_at_utc"]).value)
        prior_ids = {str(row["condition_id"]) for row in rows[:first] if int(row["_outcome_ns"]) < refit_ns}
        training = [trade for cid, trade in all_reference_trades.items() if cid in prior_ids and int(trade["outcome_ns"]) < refit_ns and cid in quotes]
        models_by_fold[fold_number] = {
            "ridge": _train_os_model(training, quotes, "ridge"),
            "tree": _train_os_model(training, quotes, "tree"),
            "training_label_cutoff_utc": pd.Timestamp(refit_ns, unit="ns", tz="UTC").isoformat(),
            "training_rows": len(training),
        }
        print(f"OS fit fold {fold_number}/{len(folds)}: prior labeled markets={len(training)}", flush=True)
    return models_by_fold


def _evaluate_fixed_buys(trades: list[dict], quotes: dict, models: dict) -> dict:
    strategies = ("hold_to_resolution", "simple_expected_value", "ridge", "tree")
    rows = []
    per_fold = []
    for strategy in strategies:
        results = []
        for trade in trades:
            fold_id = int(trade["fold_id"])
            if strategy in {"ridge", "tree"}:
                strategy_key = strategy
                trade_models = {fold_id: models[fold_id]}
            else:
                strategy_key, trade_models = strategy, models
            sold = False
            exit_offset, proceeds = None, None
            observed = 0
            for offset in DECISION_OFFSETS:
                quote = quotes.get(trade["condition_id"], {}).get(offset, {}).get(trade["side"])
                if quote:
                    observed += 1
                if _should_sell(strategy_key, trade, quote, offset, trade_models):
                    sold = True
                    exit_offset = offset
                    proceeds = _sale_proceeds(trade, quote)
                    break
            payout = proceeds if sold else (float(trade["net_shares"]) if trade["target_up"] == int(trade["side"] == "up") else 0.0)
            results.append({
                "condition_id": trade["condition_id"], "fold_id": fold_id,
                "gross_purchase_usd": trade["gross_purchase_usd"], "cash_debit_usd": trade["cash_debit_usd"],
                "net_shares": trade["net_shares"], "sold": sold, "exit_offset_seconds": exit_offset,
                "sale_proceeds_usd": proceeds, "pnl_usd": float(payout) - float(trade["cash_debit_usd"]),
                "independent_starting_wealth_usd": INITIAL_CASH_USD, "quote_observations": observed,
            })
        pnl = np.asarray([item["pnl_usd"] for item in results], dtype=float)
        growth = np.asarray([math.log(max(INITIAL_CASH_USD + value, 1e-12) / INITIAL_CASH_USD) for value in pnl])
        rows.append({
            "strategy": strategy, "trades": len(results), "sold": sum(item["sold"] for item in results),
            "total_pnl_usd": float(pnl.sum()), "mean_independent_bet_log_return": float(growth.mean()) if len(growth) else None,
            "mean_pnl_per_trade_usd": float(pnl.mean()) if len(pnl) else None,
            "total_entry_gross_usd": float(sum(item["gross_purchase_usd"] for item in results)),
            "sale_proceeds_usd": float(sum(item["sale_proceeds_usd"] or 0.0 for item in results)),
            "capital_interpretation": "Each accepted purchase is independently funded from $100; summed PnL is a paired diagnostic, not a single-portfolio return.",
        })
        for fold_id in sorted({int(item["fold_id"]) for item in results}):
            subset = [item for item in results if int(item["fold_id"]) == fold_id]
            per_fold.append({"strategy": strategy, "fold_id": fold_id, "trades": len(subset), "sold": sum(item["sold"] for item in subset), "pnl_usd": float(sum(item["pnl_usd"] for item in subset)), "mean_pnl_per_trade_usd": float(np.mean([item["pnl_usd"] for item in subset])) if subset else None})
    return {"summary": rows, "folds": per_fold}


def _full_portfolio(strategy: str, rows: list[dict], quotes: dict, models: dict) -> dict:
    event_times = set()
    entries = defaultdict(list)
    exits = defaultdict(list)
    releases = defaultdict(list)
    row_by_id = {str(row["condition_id"]): row for row in rows}
    signals = {}
    for row in rows:
        cid = str(row["condition_id"])
        side = _source_side(row)
        if side is not None:
            fill = ledger._exact_reference_fill(row, side)
            trade = _trade_for_row(row, side, fill, int(row.get("fold_id", 0) or 0))
            signals[cid] = trade
        entry_ns = int(row["_entry_ns"])
        entries[entry_ns].append(cid)
        event_times.add(entry_ns)
        for offset in DECISION_OFFSETS:
            ns = int(row["_market_ns"]) + offset * 1_000_000_000
            exits[ns].append((cid, offset))
            event_times.add(ns)
    for cid, trade in signals.items():
        releases[int(trade["release_ns"])].append(cid)
        event_times.add(int(trade["release_ns"]))
    cash = INITIAL_CASH_USD
    open_positions = {}
    realized_sales = 0
    trade_count = 0
    fees_paid = 0.0
    peak = INITIAL_CASH_USD
    max_dd = 0.0
    underwater_start = None
    max_underwater = 0.0
    max_exposure = 0.0
    max_positions = 0
    path, liq_path = [], []
    rejected = Counter()
    sold_by_strategy = Counter()
    for now in sorted(event_times):
        # Settlements and releases at the boundary are available before a new order.
        for cid in releases.get(now, []):
            position = open_positions.pop(cid, None)
            if position is None:
                continue
            won = int(position["target_up"]) == int(position["side"] == "up")
            payout = float(position["net_shares"]) if won else 0.0
            cash += payout
        # Entry is processed before same-time exits; sale proceeds cannot fund it.
        for cid in entries.get(now, []):
            trade = signals.get(cid)
            if trade is None:
                continue
            row = row_by_id[cid]
            fill = ledger._exact_reference_fill(row, trade["side"])
            if cash + 1e-9 < float(fill["cash_debit_usd"]):
                rejected["insufficient_cash"] += 1
                continue
            cash -= float(fill["cash_debit_usd"])
            position = dict(trade)
            position["gross_purchase_usd"] = float(fill["gross_usd"])
            position["cash_debit_usd"] = float(fill["cash_debit_usd"])
            position["net_shares"] = float(fill["net_shares"])
            position["entry_fee_usd"] = float(fill.get("fees_usd", 0.0))
            open_positions[cid] = position
            trade_count += 1
            fees_paid += float(fill.get("fees_usd", 0.0))
        for cid, offset in exits.get(now, []):
            position = open_positions.get(cid)
            if position is None:
                continue
            quote = quotes.get(cid, {}).get(offset, {}).get(position["side"])
            model_key = {"ridge": "ridge", "tree": "tree"}.get(strategy)
            use_strategy = model_key if model_key else strategy
            if _should_sell(use_strategy, position, quote, offset, models):
                proceeds = _sale_proceeds(position, quote)
                cash += proceeds
                fees_paid += max(0.0, float(position["net_shares"]) * float(quote["bid"]) - proceeds)
                open_positions.pop(cid, None)
                realized_sales += 1
                sold_by_strategy[strategy] += 1
        cost_basis = sum(float(position["gross_purchase_usd"]) for position in open_positions.values())
        equity = cash + cost_basis
        exposure = cost_basis
        max_exposure = max(max_exposure, exposure)
        max_positions = max(max_positions, len(open_positions))
        if equity >= peak:
            peak, underwater_start = equity, None
        else:
            max_dd = max(max_dd, 1.0 - equity / peak)
            underwater_start = now if underwater_start is None else underwater_start
            max_underwater = max(max_underwater, (now - underwater_start) / 1e9)
        path.append({"timestamp_ns": now, "cash_usd": cash, "cost_basis_equity_usd": equity, "exposure_usd": exposure, "open_positions": len(open_positions)})
        liquidation = cash
        valid = True
        for cid, position in open_positions.items():
            if now >= int(position["outcome_ns"]):
                liquidation += float(position["net_shares"]) if int(position["target_up"]) == int(position["side"] == "up") else 0.0
                continue
            offset_float = (now - int(position["market_start_ns"])) / 1e9
            if not float(offset_float).is_integer():
                valid = False
                break
            quote = quotes.get(cid, {}).get(int(offset_float), {}).get(position["side"])
            if not _valid_sale_quote(quote, position["net_shares"]):
                valid = False
                break
            liquidation += _sale_proceeds(position, quote)
        if valid:
            liq_path.append({"timestamp_ns": now, "liquidation_equity_usd": liquidation})
    if open_positions:
        raise AssertionError("The full portfolio replay ended with unresolved cash-release records")
    end_equity = cash
    if path:
        dates = pd.to_datetime([item["timestamp_ns"] for item in path], unit="ns", utc=True).date
        daily = {}
        for date, item in zip(dates, path):
            daily[str(date)] = float(item["cost_basis_equity_usd"])
        daily_values = list(daily.values())
        daily_growth = [math.log(daily_values[i] / daily_values[i - 1]) for i in range(1, len(daily_values)) if daily_values[i] > 0 and daily_values[i - 1] > 0]
    else:
        daily_growth = []
    liq_dd = None
    if liq_path:
        liq_peak, liq_dd = liq_path[0]["liquidation_equity_usd"], 0.0
        for item in liq_path:
            value = item["liquidation_equity_usd"]
            liq_peak = max(liq_peak, value)
            if liq_peak > 0:
                liq_dd = max(liq_dd, 1.0 - value / liq_peak)
    return {
        "strategy": strategy, "trade_count": trade_count, "sold_positions": realized_sales,
        "net_pnl_usd": end_equity - INITIAL_CASH_USD,
        "mean_daily_log_growth": float(np.mean(daily_growth)) if daily_growth else 0.0,
        "max_drawdown_cost_basis": max_dd, "max_drawdown_liquidation": liq_dd,
        "liquidation_coverage_events": len(liq_path), "liquidation_total_events": len(path),
        "liquidation_coverage_fraction": len(liq_path) / len(path) if path else 0.0,
        "maximum_time_underwater_seconds": max_underwater,
        "maximum_concurrent_cost_basis_exposure_usd": max_exposure,
        "maximum_concurrent_positions": max_positions, "minimum_free_cash_usd": min((item["cash_usd"] for item in path), default=INITIAL_CASH_USD),
        "fees_paid_usd": fees_paid, "rejections_by_reason": dict(rejected),
        "sale_fees_assumption": "Sale is immediate; outcome-share mode fee = shares*rate*min(bid,1-bid); cash-collateral mode uses the existing exponent 1.0 and five-decimal rounding.",
    }


def _os_study(rows: list[dict], folds: list[dict], quotes: dict, trajectory_meta: dict) -> dict:
    fold_for_market = {}
    for fold in folds:
        for row in rows[int(fold["evaluation"]["start_index_in_eligible_order"]):int(fold["evaluation"]["end_index_exclusive"])]:
            fold_for_market[str(row["condition_id"])] = int(fold["fold_id"])
    for row in rows:
        row["fold_id"] = fold_for_market.get(str(row["condition_id"]), 0)
    reference = _entry_trades(rows)
    models = _train_models_by_fold(rows, folds, reference, quotes)
    actual_buys = [trade for trade in _outer_trade_records(rows) if trade["condition_id"] in quotes]
    isolated = _evaluate_fixed_buys(actual_buys, quotes, models)
    full = [_full_portfolio(strategy, rows[int(folds[0]["evaluation"]["start_index_in_eligible_order"]):], quotes, models) for strategy in ("hold_to_resolution", "simple_expected_value", "ridge", "tree")]
    coverage_rows = trajectory_meta["markets"]
    coverage = {
        "markets": len(coverage_rows),
        "full_300_second_kacho_markets": sum(row["kacho_tick_rows"] == 300 for row in coverage_rows),
        "poststart_decision_quotes_60_of_60": sum(row["poststart_decision_quotes_of_60"] == 60 for row in coverage_rows),
        "prestart_fresh_two_sided_quote_counts": dict(Counter(str(row["pmxt_premarket_fresh_two_sided_quotes_of_11"]) for row in coverage_rows)),
        "prestart_fresh_bid_size_counts": dict(Counter(str(row["pmxt_premarket_fresh_bid_size_times_of_11"]) for row in coverage_rows)),
        "prestart_full_reference_position_exit_quotes": sum(row["pmxt_premarket_full_reference_position_exit_quotes"] for row in coverage_rows),
        "poststart_full_reference_position_exit_quotes": sum(row["poststart_full_reference_position_exit_quotes"] for row in coverage_rows),
        "prestart_fresh_bid_size_observations_at_least_1_share": sum(row["pmxt_premarket_fresh_bid_size_times_of_11"] for row in coverage_rows),
        "quote_timing": "Kacho row time is a 1Hz sample-time proxy for receive time; no forward fill. PMXT pre-start event times use timestamp_received.",
        "prestart_missing_or_unexecutable": "Pre-start sale requires a known full bid size after the T-59 snapshot; an unknown size is not treated as executable.",
        "fee_rate": "Existing market-specific entry fee_rate_bps held fixed for the 5-minute sell model; no per-second fee history exists in Kacho.",
    }
    return {
        "status": "completed_on_common_legacy_exact5_buys_and_fixed_entry_rule",
        "coverage": coverage,
        "isolated_fixed_buys": isolated,
        "full_portfolio_with_reinvestment": full,
        "models": {number: {kind: {"training_markets": bundle[kind]["training_markets"], "samples_by_offset": bundle[kind]["samples_by_offset"]} for kind in ("ridge", "tree")} for number, bundle in models.items()},
        "full_portfolio_selection_rule": "At each outer market T-59, reproduce the legacy exact-$5 candidate_platt side chooser and take the cached exact $5 fill only when current cash covers its actual cash debit. New sale proceeds are available immediately, but same-time entries occur before sales.",
        "liquidation_marking": "Only timestamped exact quotes with sufficient displayed top-bid size or deterministic known outcomes are included. Missing quotes are excluded, never forward-filled; coverage is reported.",
        "prestart_execution": "The saved T-59 snapshot supplies zero-capacity placeholder levels only; its bid has no update timestamp or executable size. Pre-start quote prices use timestamped PMXT BBO reports, and displayed size is executable only after a corresponding PMXT level update; no snapshot is forward-filled.",
        "models_trained_only_on_prior_labeled_whole_markets": True,
        "evaluation_label": "Retrospective development period; outer folds use only prior markets with outcome labels available before fold refit.",
    }


def _load_outer_predictions(folds: list[dict]) -> tuple[dict, list[dict]]:
    predictions, metadata = {}, []
    for fold_number in range(1, len(folds) + 1):
        directory = advanced.PREDICTION_CACHE_DIR
        prediction_path = directory / f"fold_{fold_number}_predictions.parquet"
        metadata_path = directory / f"fold_{fold_number}_metadata.json"
        frame = pd.read_parquet(prediction_path)
        predictions[fold_number] = {
            str(item["condition_id"]): {key: float(value) for key, value in item.items() if key != "condition_id"}
            for item in frame.to_dict("records")
        }
        metadata.append(json.loads(metadata_path.read_text(encoding="utf-8"))["model_meta"])
    return predictions, metadata


def _write_state_report(state: dict) -> None:
    rows = ["| Policy | v1 PnL | v2 PnL | v1 max DD | v2 max DD | v1/v2 trades | changed decisions | known locked moments | RCK rejections | max exposure v1/v2 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in state["policies"]:
        rows.append(
            f"| {item['policy_id']} | ${item['v1']['pnl_usd']:.2f} | ${item['state_v2']['pnl_usd']:.2f} | {item['v1']['max_drawdown_cost_basis']:.3%} | {item['state_v2']['max_drawdown_cost_basis']:.3%} | {item['v1']['trades']}/{item['state_v2']['trades']} | {item['changed_decisions']} | {item['decision_moments_with_known_locked_payout']} | {item['rck_candidate_rejections']} | ${item['v1']['max_open_cost_basis_usd']:.2f}/${item['state_v2']['max_open_cost_basis_usd']:.2f} |"
        )
    text = "\n".join([
        "# Position state correction comparison — 2026-10-07", "",
        "The original report and files remain the v1 baseline. The v2 runner used a separate cache and output set. The outcome timestamp is `resolved_at_utc` copied into `outcome_available_at_utc`; it is an explicit proxy, not the recorded receipt time of the official result.", "",
        "| Policy | v1 PnL | v2 PnL | v1 max DD | v2 max DD | v1/v2 trades | changed decisions | known locked moments | RCK rejections | max exposure v1/v2 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|", *rows[2:], "",
        "No evaluated decision timestamp fell between result availability and cash release. Decisions are 300 seconds apart; result availability was 314–752 seconds after market start, while release is `max(resolved_at, market_start+300s)+60s`. At the +541-second decision, an early result had already released or a late result was not yet available; by +841 seconds even the latest release had occurred. Thus the state correction changes no decision, RCK candidate rejection, trade, PnL, drawdown, or exposure in this schedule.", "",
        "A known locked win contributes its deterministic payout once to terminal scenario wealth; a known loss contributes zero. Neither is current cash. The release event removes the position and adds payout to cash once.", "",
    ])
    (REPORT_DIR / "position_state_comparison_20261007.md").write_text(text, encoding="utf-8")


def _write_paired_report(result: dict) -> None:
    lines = ["# Paired sizing diagnostics — 2026-10-07", "", "Frozen side/time comes from the legacy candidate_platt exact-$5 positive expected-profit chooser. Each order gets a fresh $100 cash state; there is no shared cash path or reinvestment. The figures below are diagnostics, not returns achievable by one $100 portfolio.", "", "| First | Second | Shared feasible | First only | Second only | Both rejected | Paired PnL first | Paired PnL second | Difference | Fixed-$5 common PnL / admitted gross |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in result["pairs"]:
        lines.append(f"| {item['first']} | {item['second']} | {item['common_feasible']} | {item['first_only']} | {item['second_only']} | {item['rejected_by_both']} | ${item['common_first_pnl_usd']:.2f} | ${item['common_second_pnl_usd']:.2f} | ${item['paired_pnl_difference_first_minus_second_usd']:+.2f} | ${item['common_fixed5_pnl_usd']:.2f} / ${item['common_fixed5_admitted_gross_usd']:.2f} |")
    lines.extend(["", "The common-set method stakes are actual admitted gross: first and second are reported separately in JSON alongside the fixed-$5 accepted gross above.", "", "## One-sided feasible orders and rejections", "", "| Pair | First-only PnL | Rejected by second | Second-only PnL | Rejected by first | Both rejected | Reasons: first / second |", "|---|---:|---|---:|---|---:|---|"])
    for item in result["pairs"]:
        first_only_reasons = ", ".join(f"{key}: {value}" for key, value in item["first_only_rejections_second"].items()) or "none"
        second_only_reasons = ", ".join(f"{key}: {value}" for key, value in item["second_only_rejections_first"].items()) or "none"
        both_first = ", ".join(f"{key}: {value}" for key, value in item["both_rejection_reasons_first"].items()) or "none"
        both_second = ", ".join(f"{key}: {value}" for key, value in item["both_rejection_reasons_second"].items()) or "none"
        lines.append(f"| {item['first']} vs {item['second']} | ${item['first_only_pnl_usd']:.2f} | {first_only_reasons} | ${item['second_only_pnl_usd']:.2f} | {second_only_reasons} | {item['rejected_by_both']} | {both_first} / {both_second} |")
    lines.extend(["", "Dollar stakes are the actual admitted gross amounts after cash and top-of-book limits; fixed $5 uses the same execution function and its actual admitted amount. No shared cash path or reinvestment is used in this diagnostic.", ""])
    (REPORT_DIR / "paired_sizing_20261007.md").write_text("\n".join(lines), encoding="utf-8")


def _write_eta_report(result: dict) -> None:
    lines = ["# DRK intensity selection — 2026-10-07", "", "The frozen grid is η ∈ {0, 0.25, 0.5, 0.75, 1}. Intervals are expanded to include the point estimate before interpolation; DOWN uses the complement of the adjusted UP interval. η=0 is point Kelly and η=1 is the prior full 5–95% DRK rule.", "", "| Outer fold | Internal validation rows | Internal prior rows | Selected η |", "|---:|---:|---:|---:|"]
    for item in result["internal_selection"]:
        lines.append(f"| {item['fold_id']} | {item['validation_rows']} | {item['train_rows']} | {item['selected_eta']:.2f} |")
    lines.extend(["", "## Internal validation scores", "", "| Fold | η | Mean daily log growth | PnL | Trades |", "|---:|---:|---:|---:|---:|"])
    for item in result["internal_selection"]:
        for score in item["eta_scores"]:
            lines.append(f"| {item['fold_id']} | {score['eta']:.2f} | {score['mean_daily_log_growth']:.6f} | ${score['pnl_usd']:.2f} | {score['trades']} |")
    lines.extend(["", "## Outer comparison", "", "| Variant | PnL | Mean daily log growth | Cost-basis max DD | Trades | Max exposure |", "|---|---:|---:|---:|---:|---:|"])
    for name, value in result["outer"].items():
        summary = value["summary"]
        lines.append(f"| {name} | ${summary['net_pnl_usd']:.2f} | {summary['mean_daily_log_growth']:.6f} | {summary['max_drawdown_cost_basis']:.3%} | {summary['trade_count']} | ${summary['maximum_concurrent_cost_basis_exposure_usd']:.2f} |")
    lines.extend(["", "## Outer fold economic results", "", "| Variant | Fold | η | Trades | Mean admitted stake | Total admitted stake | Expected net PnL at entry | Realized PnL |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for label, eta_value in (("point_eta0", 0.0), ("selected_eta", None), ("full_interval_eta1", 1.0)):
        artifact_path = ROOT / result["outer"][label]["trades_path"]
        trades = pd.read_parquet(artifact_path)
        grouped = trades.groupby("fold_id", sort=True).agg(
            trades=("gross_purchase_usd", "size"),
            mean_stake=("gross_purchase_usd", "mean"),
            total_stake=("gross_purchase_usd", "sum"),
            expected_pnl=("expected_net_profit_at_entry_usd", "sum"),
            realized_pnl=("realized_net_profit_usd", "sum"),
        ) if not trades.empty else pd.DataFrame()
        for fold_id in range(1, len(result["internal_selection"]) + 1):
            if fold_id in grouped.index:
                block = grouped.loc[fold_id]
                count, mean_stake, total_stake = int(block["trades"]), float(block["mean_stake"]), float(block["total_stake"])
                expected_pnl, realized_pnl = float(block["expected_pnl"]), float(block["realized_pnl"])
            else:
                count, mean_stake, total_stake, expected_pnl, realized_pnl = 0, 0.0, 0.0, 0.0, 0.0
            selected_eta = result["selected_eta_by_fold"].get(fold_id, result["selected_eta_by_fold"].get(str(fold_id)))
            applied_eta = selected_eta if eta_value is None else eta_value
            lines.append(f"| {label} | {fold_id} | {applied_eta:.2f} | {count} | ${mean_stake:.2f} | ${total_stake:.2f} | ${expected_pnl:.2f} | ${realized_pnl:.2f} |")
    lines.extend(["", "The internal calibrator used fixed C=1.0 and only labels available before that fold's validation block. Bootstrap intervals use the unchanged 3-day block length and 500 replicates. This remains development-period research, not a new holdout.", ""])
    (REPORT_DIR / "drk_intensity_20261007.md").write_text("\n".join(lines), encoding="utf-8")


def _write_os_report(result: dict) -> None:
    lines = ["# Optimal stopping continuation — 2026-10-07", "", "The isolated comparison reuses the exact same accepted legacy $5 buys and stakes. Every purchase is independently funded from $100; summed PnL is not a portfolio return. The full portfolio separately replays the frozen legacy exact-$5 entry selector with limited cash and sale proceeds available for later entries.", "", f"Kacho full 300-second paths: {result['coverage']['full_300_second_kacho_markets']}/{result['coverage']['markets']}; exact post-start 5-second decisions: {result['coverage']['poststart_decision_quotes_60_of_60']}/{result['coverage']['markets']}. Pre-start PMXT fresh two-sided BBO counts: `{result['coverage']['prestart_fresh_two_sided_quote_counts']}`; fresh bid-size counts: `{result['coverage']['prestart_fresh_bid_size_counts']}`. Full-position exit quotes: pre-start {result['coverage']['prestart_full_reference_position_exit_quotes']}, post-start {result['coverage']['poststart_full_reference_position_exit_quotes']}. The stored T-59 bid has no saved update timestamp or size and is not forward-filled.", "", "## Same buys and stakes", "", "| Exit rule | Trades | Sold | Total PnL | Mean independent bet log return | Sale proceeds |", "|---|---:|---:|---:|---:|---:|"]
    for item in result["isolated_fixed_buys"]["summary"]:
        lines.append(f"| {item['strategy']} | {item['trades']} | {item['sold']} | ${item['total_pnl_usd']:.2f} | {item['mean_independent_bet_log_return']:.6f} | ${item['sale_proceeds_usd']:.2f} |")
    lines.extend(["", "## Fold consistency for identical buys", "", "The figures below are per-fold paired diagnostics; each trade is independently funded from $100, so fold PnL is not a single-portfolio return.", "", "| Exit rule | Fold | Trades | Sold | PnL | Mean PnL per trade |", "|---|---:|---:|---:|---:|---:|"])
    for item in result["isolated_fixed_buys"]["folds"]:
        mean_pnl = "n/a" if item["mean_pnl_per_trade_usd"] is None else f"${item['mean_pnl_per_trade_usd']:.2f}"
        lines.append(f"| {item['strategy']} | {item['fold_id']} | {item['trades']} | {item['sold']} | ${item['pnl_usd']:.2f} | {mean_pnl} |")
    lines.extend(["", "## Full portfolio with reinvestment", "", "| Exit rule | Trades | Sold | PnL | Mean daily log growth | Cost-basis max DD | Liquidation max DD | Liquidation coverage | Max exposure | Underwater seconds |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for item in result["full_portfolio_with_reinvestment"]:
        liq = "n/a" if item["max_drawdown_liquidation"] is None else f"{item['max_drawdown_liquidation']:.2%}"
        lines.append(f"| {item['strategy']} | {item['trade_count']} | {item['sold_positions']} | ${item['net_pnl_usd']:.2f} | {item['mean_daily_log_growth']:.6f} | {item['max_drawdown_cost_basis']:.2%} | {liq} | {item['liquidation_coverage_fraction']:.1%} | ${item['maximum_concurrent_cost_basis_exposure_usd']:.2f} | {item['maximum_time_underwater_seconds']:.0f} |")
    lines.extend(["", "Kacho provides one-second sampled top bid/ask and top sizes from market start through second 299. Its row timestamp is a sample-time proxy, not a WebSocket receive timestamp. No quotes are forward-filled. A sale requires a fresh bid and enough best-bid size for the full position. The continuation models are trained per isolated $100 position and then applied to each position in the shared portfolio; they do not optimize cross-position utility. Sale fees extend the existing date-based fee mode using the market entry fee rate held constant for five minutes; fees are an explicit model approximation. Sale proceeds are available immediately. At equal timestamps, settlement release is processed first, then a new entry, then exits, so an exit cannot finance a same-time entry.", "", "For liquidation drawdown, every open position must be valued at an executable current bid or a deterministic known payout. Unvalued timestamps are excluded rather than forward-filled; the table reports coverage. Cost-basis drawdown uses the original ledger convention.", ""])
    (REPORT_DIR / "optimal_stopping_20261007.md").write_text("\n".join(lines), encoding="utf-8")


def _write_final_report(state: dict, sizing: dict, eta: dict, os_result: dict) -> None:
    selected = eta["outer"]["selected_eta"]["summary"]
    point = eta["outer"]["point_eta0"]["summary"]
    full_os = {item["strategy"]: item for item in os_result["full_portfolio_with_reinvestment"]}
    rck_v2 = next(item for item in state["policies"] if item["policy_id"] == "rck_alpha_0.5_correlated")
    top_growth = max(os_result["full_portfolio_with_reinvestment"], key=lambda item: item["mean_daily_log_growth"])
    isolated_folds = os_result["isolated_fixed_buys"]["folds"]
    ridge_fold_pnl = [item["pnl_usd"] for item in isolated_folds if item["strategy"] == "ridge"]
    tree_fold_pnl = [item["pnl_usd"] for item in isolated_folds if item["strategy"] == "tree"]
    simple_fold_pnl = [item["pnl_usd"] for item in isolated_folds if item["strategy"] == "simple_expected_value"]
    lines = [
        "# BTC pre-open policy continuation — 2026-10-07", "",
        "All policies remain inactive. This retrospective development period has earlier selection exposure and is not an independent holdout.", "",
        "## Findings", "",
        f"- Position-state correction: {sum(item['changed_decisions'] for item in state['policies'])} changed decisions across {len(state['policies'])} policies; the corrected logs contain no decision moment with a known-but-locked payout. The event schedule explains why: resolutions arrive 314–752 seconds after start and cash releases at the later of resolution or +300 seconds, plus 60 seconds.",
        f"- DRK intensity: selected η by outer fold `{eta['selected_eta_by_fold']}`. The selected-η outer portfolio PnL was ${selected['net_pnl_usd']:.2f}, versus ${point['net_pnl_usd']:.2f} for η=0 point Kelly; this is a development comparison, not evidence of stable improvement.",
        f"- RCK: the corrected-state α=0.5 correlated comparison matches v1 at ${rck_v2['v1']['pnl_usd']:.2f}. Its wealth denominator remains cost-basis equity, an explicit approximation rather than mark-to-market value; paired sizing separates stake effects from constraint rejections.",
        f"- Optimal stopping: no exit rule dominates PnL and risk. `{top_growth['strategy']}` has the highest full-portfolio mean daily log growth ({top_growth['mean_daily_log_growth']:.6f}). On independently funded identical buys, ridge was positive in {sum(value > 0 for value in ridge_fold_pnl)}/3 folds, tree in {sum(value > 0 for value in tree_fold_pnl)}/3, and simple expected value in {sum(value > 0 for value in simple_fold_pnl)}/3; fold PnL is not a portfolio return. These are development results, not a holdout.",
        "- All full-portfolio exit policies use identical legacy exact-$5 entry selection and reinvest sale proceeds. The isolated analysis uses identical accepted buys/stakes but each trade is independently funded from $100.",
        "", "## Full-portfolio risk and growth", "",
        "| Exit rule | PnL | Mean daily log growth | Cost-basis DD | Liquidation DD | Liquidation coverage | Max exposure | Max underwater | Trades |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in full_os.items():
        liq = "n/a" if item["max_drawdown_liquidation"] is None else f"{item['max_drawdown_liquidation']:.2%}"
        lines.append(f"| {name} | ${item['net_pnl_usd']:.2f} | {item['mean_daily_log_growth']:.6f} | {item['max_drawdown_cost_basis']:.2%} | {liq} | {item['liquidation_coverage_fraction']:.1%} | ${item['maximum_concurrent_cost_basis_exposure_usd']:.2f} | {item['maximum_time_underwater_seconds']:.0f}s | {item['trade_count']} |")
    lines.extend([
        "", "## Separate reports", "",
        "- `position_state_comparison_20261007.md` — v1 against corrected position state.",
        "- `paired_sizing_20261007.md` — common/one-sided feasibility and paired PnL, including actual fixed-$5 admitted amounts.",
        "- `drk_intensity_20261007.md` — causal internal η selection and outer comparison.",
        "- `optimal_stopping_20261007.md` — identical-buy exit comparison, reinvested portfolio, and execution coverage.",
        "- `trajectory_coverage_os_20261007.csv` — per-market PMXT/Kacho coverage.",
        "- `audit.json` — local PMXT schema/event provenance and the audit of other locally available Parquet files.",
        "", "Outcome availability uses `resolved_at_utc` as an explicit proxy because the source does not record the first receipt time of the official label. Entry fee rate is held constant for post-entry sales because per-tick fee history is absent. The saved Kacho data required no network retrieval; added data cost was $0.", "",
    ])
    (REPORT_DIR / "policy_continuation_20261007.md").write_text("\n".join(lines), encoding="utf-8")


def _run() -> None:
    started = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    base_config = _base_config()
    study = _source_study()
    rows, folds, _ = ledger._load_rows(base_config, study)
    for row in rows:
        row["_market_ns"] = int(pd.Timestamp(row["market_start_utc"]).value)
    predictions, fold_meta = _load_outer_predictions(folds)
    z_bank = np.random.default_rng(int(config["scenario_seed"])).standard_normal((int(config["scenario_count"]), 16))

    state = _state_impact()
    _write_json(REPORT_DIR / "position_state_comparison_20261007.json", state)
    _write_state_report(state)
    print("state v1/v2 comparison complete", flush=True)

    sizing = _paired_sizing(rows, folds, predictions, fold_meta, base_config, json.loads(advanced.CONFIG_PATH.read_text(encoding="utf-8")), z_bank)
    _write_json(REPORT_DIR / "paired_sizing_20261007.json", sizing)
    _write_paired_report(sizing)
    print("paired sizing complete", flush=True)

    eta = _eta_study(rows, folds, predictions, fold_meta, base_config, json.loads(advanced.CONFIG_PATH.read_text(encoding="utf-8")), z_bank)
    eta_artifacts = eta.pop("outer_artifacts")
    eta_dir = ROOT / "data/analysis/polymarket/BTC/preopen_v1/drk_intensity_20261007"
    eta_dir.mkdir(parents=True, exist_ok=True)
    for label, artifacts in eta_artifacts.items():
        pd.DataFrame(artifacts["decisions"]).to_parquet(eta_dir / f"{label}_decisions.parquet", index=False, compression="zstd")
        pd.DataFrame(artifacts["trades"]).to_parquet(eta_dir / f"{label}_trades.parquet", index=False, compression="zstd")
        eta["outer"][label]["decisions_path"] = (eta_dir / f"{label}_decisions.parquet").relative_to(ROOT).as_posix()
        eta["outer"][label]["trades_path"] = (eta_dir / f"{label}_trades.parquet").relative_to(ROOT).as_posix()
    _write_json(REPORT_DIR / "drk_intensity_20261007.json", eta)
    _write_eta_report(eta)
    print("DRK intensity selection and outer replay complete", flush=True)

    print(f"Loading local PMXT pre-start events and Kacho 1Hz trajectories for {len(rows)} eligible markets.", flush=True)
    quotes, trajectory_meta = _load_trajectories(rows)
    os_result = _os_study(rows, folds, quotes, trajectory_meta)
    coverage_frame = pd.DataFrame(trajectory_meta["markets"])
    coverage_frame.to_csv(REPORT_DIR / "trajectory_coverage_os_20261007.csv", index=False)
    _write_json(REPORT_DIR / "optimal_stopping_20261007.json", os_result)
    _write_os_report(os_result)
    _write_final_report(state, sizing, eta, os_result)
    print(f"Completed continuation in {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    _run()
