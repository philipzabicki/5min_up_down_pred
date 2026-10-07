"""Run the frozen best-ask-only BTC pre-open policy walk-forward study."""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports/btc_preopen"
OPPORTUNITIES_PATH = REPORT_DIR / "policy_opportunities.parquet"
STUDY_PATH = REPORT_DIR / "policy_study.json"
TRIALS_PATH = REPORT_DIR / "policy_trials.csv"
CONFIG_PATH = ROOT / "configs/research/btc_preopen_policy_search_20261007.json"
FINAL_POLICY_PATH = REPORT_DIR / "policy_final_research_20261007.json"
TRADE_LOG_PATH = REPORT_DIR / "policy_outer_trades.parquet"
DECISION_LOG_PATH = REPORT_DIR / "policy_outer_decisions.parquet"
EQUITY_PATH_PATH = REPORT_DIR / "policy_outer_equity_events.parquet"
DAILY_PATH = REPORT_DIR / "policy_outer_daily_equity.parquet"
INITIAL_CASH_USD = 100.0
BASELINE_STAKE_USD = 5.0
DAY_NS = 86_400_000_000_000
CENT = 0.01
EPS = 1e-10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _round_down_cent(value: float) -> float:
    return max(0.0, math.floor((float(value) + 1e-9) * 100.0) / 100.0)


def _trial_grid(config: dict) -> list[dict]:
    search = config["search_space"]
    rows = [{"trial_id": "no_trade", "family": "no_trade", "parameters": {}}]
    thresholds = search["minimum_expected_net_return_on_cash_debit"]
    for stake in search["family_a_fixed_gross_usd"]:
        for threshold in thresholds:
            rows.append({
                "trial_id": f"fixed_{stake:g}_roi_{threshold:g}",
                "family": "fixed_stake",
                "parameters": {"stake_usd": float(stake), "min_expected_return": float(threshold)},
            })
    for fraction in search["family_b_fraction_of_free_cash"]:
        for threshold in thresholds:
            rows.append({
                "trial_id": f"free_cash_{fraction:g}_roi_{threshold:g}",
                "family": "free_cash_fraction",
                "parameters": {
                    "fraction_of_free_cash": float(fraction),
                    "max_gross_stake_usd": float(search["family_c_technical_max_gross_usd"]),
                    "min_expected_return": float(threshold),
                },
            })
    for multiplier in search["family_c_fractional_kelly_multipliers"]:
        for equity_fraction in search["family_c_max_cost_basis_equity_fraction_per_purchase"]:
            for threshold in thresholds:
                rows.append({
                    "trial_id": f"kelly_{multiplier:g}_cap_{equity_fraction:g}_roi_{threshold:g}",
                    "family": "fractional_kelly",
                    "parameters": {
                        "kelly_multiplier": float(multiplier),
                        "max_cost_basis_equity_fraction": float(equity_fraction),
                        "max_gross_stake_usd": float(search["family_c_technical_max_gross_usd"]),
                        "min_expected_return": float(threshold),
                    },
                })
    if len(rows) != int(search["trial_budget"]):
        raise AssertionError(f"Frozen grid has {len(rows)} trials, expected {search['trial_budget']}")
    return rows


@dataclass
class Portfolio:
    cash: float = INITIAL_CASH_USD
    locked_cost: float = 0.0
    pending: list = field(default_factory=list)
    sequence: int = 0

    @property
    def equity(self) -> float:
        return float(self.cash + self.locked_cost)

    def clone(self) -> "Portfolio":
        return Portfolio(
            cash=float(self.cash),
            locked_cost=float(self.locked_cost),
            pending=list(self.pending),
            sequence=int(self.sequence),
        )


@dataclass
class RunMetrics:
    initial_equity: float
    start_ns: int
    cash_min: float = INITIAL_CASH_USD
    max_locked_cost: float = 0.0
    max_open_positions: int = 0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    peak_ns: int = 0
    underwater_start_ns: int | None = None
    max_underwater_ns: int = 0
    trade_count: int = 0
    decision_count: int = 0
    gross_turnover: float = 0.0
    cash_debit_total: float = 0.0
    fees_usd: float = 0.0
    fee_shares: float = 0.0
    expected_net_profit: float = 0.0
    insufficient_cash_skips: int = 0
    no_trade_skips: int = 0
    min_order_skips: int = 0
    cash_limited_orders: int = 0
    liquidity_skip_orders: int = 0
    liquidity_limited_orders: int = 0
    policy_capped_orders: int = 0
    log_events: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    decisions: list = field(default_factory=list)

    def __post_init__(self):
        self.peak_equity = float(self.initial_equity)
        self.peak_ns = int(self.start_ns)

    def record(self, timestamp_ns: int, event: str, state: Portfolio, condition_id: str | None = None):
        equity = state.equity
        if equity >= self.peak_equity - 1e-9:
            if self.underwater_start_ns is not None:
                self.max_underwater_ns = max(
                    self.max_underwater_ns, int(timestamp_ns) - self.underwater_start_ns
                )
                self.underwater_start_ns = None
            if equity > self.peak_equity:
                self.peak_equity = equity
                self.peak_ns = int(timestamp_ns)
        elif self.peak_equity > 0.0:
            self.max_drawdown = max(self.max_drawdown, (self.peak_equity - equity) / self.peak_equity)
            if self.underwater_start_ns is None:
                self.underwater_start_ns = self.peak_ns
        self.cash_min = min(self.cash_min, float(state.cash))
        self.max_locked_cost = max(self.max_locked_cost, float(state.locked_cost))
        self.max_open_positions = max(self.max_open_positions, len(state.pending))
        self.log_events.append((int(timestamp_ns), float(equity)))


def _fee_components(row: dict, side: str, gross_usd: float, config: dict) -> dict:
    price = float(row[f"{side}_best_ask"])
    gross_shares = float(gross_usd) / price
    fee_config = config["execution"]["fee_model"]
    rate = float(row["fee_rate_bps"]) / 10_000.0
    if row["fee_collection_mode"] == "outcome_shares":
        raw_fee_shares = gross_shares * rate * min(price, 1.0 - price) / price
        scale = 10 ** int(fee_config["legacy_fee_share_decimals"])
        fee_shares = math.floor(raw_fee_shares * scale + 1e-9) / scale
        cash_fee = 0.0
        fee_usd = fee_shares * price
        net_shares = gross_shares - fee_shares
    elif row["fee_collection_mode"] == "cash_collateral":
        raw_fee = gross_shares * rate * (price * (1.0 - price)) ** float(
            fee_config["cash_fee_exponent"]
        )
        cash_fee = round(raw_fee, int(fee_config["cash_fee_round_decimals"]))
        if cash_fee < float(fee_config["cash_fee_minimum_usd"]):
            cash_fee = 0.0
        fee_usd = cash_fee
        fee_shares = 0.0
        net_shares = gross_shares
    else:
        raise ValueError(f"Unexpected fee mode on an eligible market: {row['fee_collection_mode']}")
    return {
        "gross_shares": gross_shares,
        "fee_shares": fee_shares,
        "net_shares": net_shares,
        "cash_fee_usd": cash_fee,
        "fees_usd": fee_usd,
        "cash_debit_usd": float(gross_usd) + cash_fee,
    }


def _cash_gross_cap(row: dict, side: str, cash: float, config: dict) -> float:
    price = float(row[f"{side}_best_ask"])
    rate = float(row["fee_rate_bps"]) / 10_000.0
    fee_config = config["execution"]["fee_model"]
    if row["fee_collection_mode"] != "cash_collateral" or rate <= 0.0:
        return max(0.0, float(cash))
    fee_per_gross = rate * (price * (1.0 - price)) ** float(
        fee_config["cash_fee_exponent"]
    ) / price
    amount = _round_down_cent(float(cash) / (1.0 + fee_per_gross))
    while amount > 0.0 and _fee_components(row, side, amount, config)["cash_debit_usd"] > cash + EPS:
        amount = _round_down_cent(amount - CENT)
    return amount


def _feasible_order(
    row: dict,
    side: str,
    requested_usd: float,
    state: Portfolio,
    config: dict,
    sizing_cap_reasons: list[str] | None = None,
) -> dict:
    price = float(row[f"{side}_best_ask"])
    capacity = price * float(row[f"{side}_best_ask_size_shares"])
    max_order = float(config["execution"]["technical_max_gross_purchase_usd"])
    cash_cap = _cash_gross_cap(row, side, state.cash, config)
    requested = max(0.0, float(requested_usd))
    uncapped = min(requested, max_order, capacity, cash_cap)
    capped = _round_down_cent(uncapped)
    limit_reasons = list(sizing_cap_reasons or [])
    if requested > max_order + EPS:
        limit_reasons.append("technical_20_usd_cap")
    active_bounds = {
        "top_of_book_liquidity": capacity,
        "cash_including_fee": cash_cap,
    }
    for label, bound in active_bounds.items():
        if bound < requested - EPS and math.isclose(bound, uncapped, abs_tol=CENT + EPS):
            limit_reasons.append(label)
    result = {
        "requested_gross_usd": requested,
        "gross_before_minimum_usd": capped,
        "admitted_gross_usd": 0.0,
        "top_of_book_capacity_usd": capacity,
        "cash_gross_cap_usd": cash_cap,
        "limit_reasons": sorted(set(limit_reasons)),
        "skip_reason": None,
        "price": price,
        "gross_shares": 0.0,
        "net_shares": 0.0,
        "fee_shares": 0.0,
        "cash_fee_usd": 0.0,
        "fees_usd": 0.0,
        "cash_debit_usd": 0.0,
    }
    if requested <= EPS:
        result["skip_reason"] = "no_positive_kelly_amount"
        return result
    if capped + EPS < float(config["execution"]["minimum_purchase_assumption"]["minimum_gross_notional_usd"]):
        result["skip_reason"] = (
            "insufficient_cash" if cash_cap < uncapped + EPS
            else "insufficient_top_of_book" if capacity < uncapped + EPS
            else "below_minimum_gross_notional"
        )
        return result
    fee = _fee_components(row, side, capped, config)
    minimum_shares = float(config["execution"]["minimum_purchase_assumption"]["minimum_net_shares"])
    if fee["net_shares"] + EPS < minimum_shares:
        result["skip_reason"] = "minimum_order_shares"
        result.update(fee)
        return result
    if fee["cash_debit_usd"] > state.cash + EPS:
        result["skip_reason"] = "insufficient_cash_after_fee"
        result.update(fee)
        return result
    result.update(fee)
    result["admitted_gross_usd"] = capped
    return result


def _net_unit_rates(row: dict, side: str, config: dict) -> tuple[float, float]:
    price = float(row[f"{side}_best_ask"])
    rate = float(row["fee_rate_bps"]) / 10_000.0
    fee_config = config["execution"]["fee_model"]
    if row["fee_collection_mode"] == "outcome_shares":
        share_fee_rate = rate * min(price, 1.0 - price) / price
        return (1.0 - share_fee_rate) / price, 1.0
    cash_fee_per_gross = rate * (price * (1.0 - price)) ** float(
        fee_config["cash_fee_exponent"]
    ) / price
    return 1.0 / price, 1.0 + cash_fee_per_gross


def _sized_request(row: dict, side: str, family: str, params: dict, state: Portfolio, config: dict):
    if family == "no_trade":
        return 0.0, []
    if family == "fixed_stake":
        return float(params["stake_usd"]), []
    if family == "free_cash_fraction":
        raw = float(state.cash) * float(params["fraction_of_free_cash"])
        cap = float(params["max_gross_stake_usd"])
        return min(raw, cap), (["technical_20_usd_cap"] if raw > cap + EPS else [])
    if family != "fractional_kelly":
        raise ValueError(f"Unknown policy family: {family}")
    p_up = float(row["p_candidate_platt"])
    p_side = p_up if side == "up" else 1.0 - p_up
    payout_per_gross, debit_per_gross = _net_unit_rates(row, side, config)
    win_profit_per_gross = payout_per_gross - debit_per_gross
    edge_per_gross = p_side * payout_per_gross - debit_per_gross
    equity = state.equity
    if equity <= 0.0 or win_profit_per_gross <= 0.0 or edge_per_gross <= 0.0:
        return 0.0, []
    full_kelly = equity * edge_per_gross / (debit_per_gross * win_profit_per_gross)
    raw_fractional = float(params["kelly_multiplier"]) * full_kelly
    capital_cap = equity * float(params["max_cost_basis_equity_fraction"])
    order_cap = float(params["max_gross_stake_usd"])
    requested = min(raw_fractional, capital_cap, order_cap)
    reasons = []
    if raw_fractional > capital_cap + EPS:
        reasons.append("kelly_cost_basis_equity_fraction_cap")
    if raw_fractional > order_cap + EPS:
        reasons.append("technical_20_usd_cap")
    return requested, reasons


def _side_candidate(row: dict, side: str, family: str, params: dict, state: Portfolio, config: dict) -> dict:
    requested, sizing_limits = _sized_request(row, side, family, params, state, config)
    order = _feasible_order(row, side, requested, state, config, sizing_limits)
    p_up = float(row["p_candidate_platt"])
    p_side = p_up if side == "up" else 1.0 - p_up
    if order["admitted_gross_usd"] > 0.0:
        payout = float(order["net_shares"])
        debit = float(order["cash_debit_usd"])
        expected_payout = p_side * payout
        expected_profit = expected_payout - debit
        expected_return = expected_profit / debit if debit > 0.0 else float("-inf")
        ev_per_net_share = expected_profit / payout if payout > 0.0 else float("-inf")
        win_wealth = state.equity - debit + payout
        loss_wealth = state.equity - debit
        log_growth = 0.0
        if p_side > 0.0:
            log_growth += p_side * math.log(win_wealth / state.equity) if win_wealth > 0.0 else float("-inf")
        if p_side < 1.0:
            log_growth += (1.0 - p_side) * math.log(loss_wealth / state.equity) if loss_wealth > 0.0 else float("-inf")
    else:
        payout = expected_payout = expected_profit = expected_return = ev_per_net_share = log_growth = float("nan")
    threshold = float(params.get("min_expected_return", 0.0))
    if order["skip_reason"] is not None:
        order["policy_skip_reason"] = order["skip_reason"]
    elif expected_return + EPS < threshold:
        order["policy_skip_reason"] = "below_expected_net_return_threshold"
    elif log_growth <= 0.0:
        order["policy_skip_reason"] = "nonpositive_expected_log_growth"
    else:
        order["policy_skip_reason"] = None
    return {
        **order,
        "side": side,
        "p_win": p_side,
        "expected_net_payout_usd": expected_payout,
        "expected_net_profit_usd": expected_profit,
        "expected_return_on_cash_debit": expected_return,
        "expected_value_per_net_share_usd": ev_per_net_share,
        "expected_log_growth": log_growth,
    }


def _close_positions(state: Portfolio, metrics: RunMetrics, until_ns: int, inclusive: bool = True):
    while state.pending:
        release_ns, sequence, gross, payout, condition_id = state.pending[0]
        if release_ns > until_ns or (not inclusive and release_ns == until_ns):
            break
        heapq.heappop(state.pending)
        state.locked_cost -= gross
        state.cash += payout
        if abs(state.locked_cost) < 1e-9:
            state.locked_cost = 0.0
        if state.cash < -1e-7:
            raise AssertionError("Portfolio cash became negative")
        metrics.record(release_ns, "capital_release", state, condition_id)


def _decision_payload(row: dict, policy: dict, candidates: dict, chosen_side: str | None, reason: str) -> dict:
    item = {
        "policy_id": policy["trial_id"],
        "portfolio_id": policy["portfolio_id"],
        "condition_id": row["condition_id"],
        "entry_time_utc": row["entry_time_utc"],
        "p_candidate_platt_up": float(row["p_candidate_platt"]),
        "chosen_side": chosen_side or "no_trade",
        "decision_reason": reason,
    }
    for side in ("up", "down"):
        candidate = candidates[side]
        for source, suffix in (
            ("requested_gross_usd", "requested_usd"),
            ("gross_before_minimum_usd", "after_execution_caps_usd"),
            ("admitted_gross_usd", "admitted_usd"),
            ("price", "best_ask"),
            ("top_of_book_capacity_usd", "top_capacity_usd"),
            ("cash_gross_cap_usd", "cash_cap_usd"),
            ("net_shares", "net_shares"),
            ("cash_debit_usd", "cash_debit_usd"),
            ("fees_usd", "fees_usd"),
            ("p_win", "p_win"),
            ("expected_net_payout_usd", "expected_payout_usd"),
            ("expected_net_profit_usd", "expected_profit_usd"),
            ("expected_value_per_net_share_usd", "ev_per_net_share_usd"),
            ("expected_return_on_cash_debit", "expected_return"),
            ("expected_log_growth", "expected_log_growth"),
        ):
            value = candidate.get(source)
            item[f"{side}_{suffix}"] = value if value is None or math.isfinite(float(value)) else None
        item[f"{side}_limit_reasons"] = ";".join(candidate.get("limit_reasons", []))
        item[f"{side}_skip_reason"] = candidate.get("policy_skip_reason")
    return item


def _simulate_policy(
    rows: list[dict],
    policy: dict,
    state: Portfolio,
    metrics: RunMetrics,
    config: dict,
    *,
    cutoff_ns: int | None = None,
    include_logs: bool = False,
):
    family = policy["family"]
    params = policy.get("parameters", {})
    for row in rows:
        entry_ns = int(row["_entry_ns"])
        if cutoff_ns is not None and entry_ns >= cutoff_ns:
            break
        _close_positions(state, metrics, entry_ns, inclusive=True)
        metrics.decision_count += 1
        if include_logs:
            metrics.decisions.append({})
        if family == "legacy_exact_fixed_5":
            p_up = float(row["p_candidate_platt"])
            ev_up = p_up * float(row["up_fill_5usd_net_shares"]) - float(row["up_fill_5usd_cash_debit_usd"])
            ev_down = (1.0 - p_up) * float(row["down_fill_5usd_net_shares"]) - float(row["down_fill_5usd_cash_debit_usd"])
            if max(ev_up, ev_down) <= 0.0:
                metrics.no_trade_skips += 1
                if include_logs:
                    metrics.decisions[-1] = {
                        "policy_id": policy["trial_id"], "condition_id": row["condition_id"],
                        "entry_time_utc": row["entry_time_utc"],
                        "p_candidate_platt_up": p_up, "chosen_side": "no_trade",
                        "decision_reason": "legacy_rule_no_positive_expected_value",
                    }
                continue
            side = "up" if ev_up >= ev_down else "down"
            prefix = f"{side}_fill_5usd_"
            gross = float(row[prefix + "gross_usd"])
            shares = float(row[prefix + "net_shares"])
            debit = float(row[prefix + "cash_debit_usd"])
            fee_usd = float(row[prefix + "fee_usd"])
            fee_shares = float(row[prefix + "fee_shares"])
            cash_fee = float(row[prefix + "cash_fee_usd"])
            expected_value = ev_up if side == "up" else ev_down
            fill = {
                "gross_usd": gross,
                "gross_shares": float(row[prefix + "gross_shares"]),
                "net_shares": shares,
                "cash_debit_usd": debit,
                "fees_usd": fee_usd,
                "fee_shares": fee_shares,
                "cash_fee_usd": cash_fee,
                "price": float(row[f"{side}_best_ask"]),
                "requested_gross_usd": BASELINE_STAKE_USD,
                "gross_before_minimum_usd": gross,
                "admitted_gross_usd": gross,
                "top_of_book_capacity_usd": float(row[f"{side}_top_ask_capacity_usd"]),
                "cash_gross_cap_usd": state.cash,
                "limit_reasons": [],
                "skip_reason": None,
            }
            candidates = {"up": {"policy_skip_reason": "baseline_side_not_selected"}, "down": {"policy_skip_reason": "baseline_side_not_selected"}}
            candidates[side] = {**fill, "p_win": p_up if side == "up" else 1.0 - p_up}
            if state.cash + EPS < debit:
                metrics.insufficient_cash_skips += 1
                if include_logs:
                    metrics.decisions[-1] = _decision_payload(row, policy, candidates, None, "insufficient_cash")
                continue
            if include_logs:
                metrics.decisions[-1] = _decision_payload(row, policy, candidates, side, "legacy_exact_fill")
        elif family == "legacy_best_level_fixed_5":
            p_up = float(row["p_candidate_platt"])
            ev_up = p_up * float(row["up_fill_5usd_net_shares"]) - float(row["up_fill_5usd_cash_debit_usd"])
            ev_down = (1.0 - p_up) * float(row["down_fill_5usd_net_shares"]) - float(row["down_fill_5usd_cash_debit_usd"])
            if max(ev_up, ev_down) <= 0.0:
                metrics.no_trade_skips += 1
                if include_logs:
                    metrics.decisions[-1] = {"policy_id": policy["trial_id"], "condition_id": row["condition_id"], "entry_time_utc": row["entry_time_utc"], "p_candidate_platt_up": p_up, "chosen_side": "no_trade", "decision_reason": "legacy_rule_no_positive_expected_value"}
                continue
            side = "up" if ev_up >= ev_down else "down"
            candidate = _feasible_order(row, side, BASELINE_STAKE_USD, state, config)
            if candidate["admitted_gross_usd"] <= 0.0:
                if candidate["skip_reason"] in {"minimum_order_shares", "below_minimum_gross_notional"}:
                    metrics.min_order_skips += 1
                elif candidate["skip_reason"] in {"insufficient_cash", "insufficient_cash_after_fee"}:
                    metrics.insufficient_cash_skips += 1
                elif candidate["skip_reason"] == "insufficient_top_of_book":
                    metrics.liquidity_skip_orders += 1
                if include_logs:
                    blank = {"policy_skip_reason": "baseline_side_not_selected"}
                    cands = {"up": blank, "down": blank}
                    cands[side] = {**candidate, "side": side, "policy_skip_reason": candidate["skip_reason"]}
                    metrics.decisions[-1] = _decision_payload(row, policy, cands, None, candidate["skip_reason"] or "not_executable")
                continue
            fill = {**candidate, "gross_usd": candidate["admitted_gross_usd"]}
            candidates = {"up": {"policy_skip_reason": "baseline_side_not_selected"}, "down": {"policy_skip_reason": "baseline_side_not_selected"}}
            candidates[side] = {
                **fill,
                "side": side,
                "p_win": p_up if side == "up" else 1.0 - p_up,
                "expected_net_payout_usd": (p_up if side == "up" else 1.0 - p_up) * fill["net_shares"],
                "expected_net_profit_usd": (p_up if side == "up" else 1.0 - p_up) * fill["net_shares"] - fill["cash_debit_usd"],
                "expected_value_per_net_share_usd": (p_up if side == "up" else 1.0 - p_up) - fill["cash_debit_usd"] / fill["net_shares"],
                "expected_return_on_cash_debit": ((p_up if side == "up" else 1.0 - p_up) * fill["net_shares"] - fill["cash_debit_usd"]) / fill["cash_debit_usd"],
                "expected_log_growth": float("nan"),
                "policy_skip_reason": None,
            }
            expected_value = float(candidates[side]["expected_net_profit_usd"])
            if include_logs:
                metrics.decisions[-1] = _decision_payload(row, policy, candidates, side, "legacy_entry_rule_best_level_execution")
        else:
            candidates = {
                side: _side_candidate(row, side, family, params, state, config)
                for side in ("up", "down")
            }
            admissible = [
                candidate for candidate in candidates.values()
                if candidate.get("policy_skip_reason") is None
            ]
            if not admissible:
                reasons = [candidate.get("policy_skip_reason") for candidate in candidates.values()]
                if any(reason in {"minimum_order_shares", "below_minimum_gross_notional"} for reason in reasons):
                    metrics.min_order_skips += 1
                elif any(reason in {"insufficient_cash", "insufficient_cash_after_fee"} for reason in reasons):
                    metrics.insufficient_cash_skips += 1
                elif "insufficient_top_of_book" in reasons:
                    metrics.liquidity_skip_orders += 1
                else:
                    metrics.no_trade_skips += 1
                if include_logs:
                    metrics.decisions[-1] = _decision_payload(row, policy, candidates, None, ";".join(sorted(set(reasons))))
                continue
            chosen = max(admissible, key=lambda item: (float(item["expected_log_growth"]), item["side"] == "up"))
            side = chosen["side"]
            fill = {**chosen, "gross_usd": chosen["admitted_gross_usd"]}
            expected_value = float(chosen["expected_net_profit_usd"])
            if include_logs:
                metrics.decisions[-1] = _decision_payload(row, policy, candidates, side, "selected_max_expected_log_growth")

        entry_cash = float(state.cash)
        equity_before = state.equity
        outcome_up = int(row["target_polymarket_up"])
        won = outcome_up == (1 if side == "up" else 0)
        payout = float(fill["net_shares"]) if won else 0.0
        gross_usd = float(fill.get("gross_usd", fill.get("admitted_gross_usd", 0.0)))
        debit = float(fill["cash_debit_usd"])
        state.cash -= debit
        state.locked_cost += gross_usd
        state.sequence += 1
        release_ns = int(row["_release_ns"])
        heapq.heappush(state.pending, (release_ns, state.sequence, gross_usd, payout, str(row["condition_id"])))
        metrics.trade_count += 1
        metrics.gross_turnover += gross_usd
        metrics.cash_debit_total += debit
        metrics.fees_usd += float(fill["fees_usd"])
        metrics.fee_shares += float(fill.get("fee_shares", 0.0))
        metrics.expected_net_profit += float(expected_value)
        if "top_of_book_liquidity" in fill.get("limit_reasons", []):
            metrics.liquidity_limited_orders += 1
        if "cash_including_fee" in fill.get("limit_reasons", []):
            metrics.cash_limited_orders += 1
        if any(reason in fill.get("limit_reasons", []) for reason in ("technical_20_usd_cap", "kelly_cost_basis_equity_fraction_cap")):
            metrics.policy_capped_orders += 1
        metrics.record(entry_ns, "purchase", state, str(row["condition_id"]))
        trade = {
            "portfolio_id": policy["portfolio_id"],
            "policy_id": policy["trial_id"],
            "family": family,
            "parameters_json": json.dumps(params, sort_keys=True),
            "condition_id": row["condition_id"],
            "market_slug": row["market_slug"],
            "entry_time_utc": row["entry_time_utc"],
            "capital_release_at_utc": row["capital_release_at_utc"],
            "chosen_side": side,
            "p_win": float(fill.get("p_win", float(row["p_candidate_platt"]) if side == "up" else 1.0 - float(row["p_candidate_platt"]))),
            "best_ask": float(fill["price"]),
            "top_ask_size_shares": float(row[f"{side}_best_ask_size_shares"]),
            "requested_gross_usd": float(fill.get("requested_gross_usd", BASELINE_STAKE_USD)),
            "gross_limited_before_minimum_usd": float(fill.get("gross_before_minimum_usd", gross_usd)),
            "gross_purchase_usd": gross_usd,
            "gross_shares": float(fill["gross_shares"]),
            "fee_shares": float(fill.get("fee_shares", 0.0)),
            "net_shares": float(fill["net_shares"]),
            "fees_usd": float(fill["fees_usd"]),
            "cash_fee_usd": float(fill.get("cash_fee_usd", 0.0)),
            "cash_debit_usd": debit,
            "cash_before_usd": entry_cash,
            "cost_basis_equity_before_usd": equity_before,
            "stake_to_equity_before": gross_usd / equity_before if equity_before > 0 else None,
            "expected_net_profit_usd": float(expected_value),
            "win_payout_usd": float(fill["net_shares"]),
            "won": bool(won),
            "realized_net_profit_usd": payout - debit,
            "payout_usd": payout,
            "limit_reasons": ";".join(fill.get("limit_reasons", [])),
        }
        metrics.trades.append(trade)


def _daily_growth(events: list[tuple[int, float]], initial_equity: float, start_ns: int, end_ns: int) -> tuple[list[dict], float]:
    if end_ns < start_ns:
        return [], 0.0
    ordered = sorted(enumerate(events), key=lambda pair: (pair[1][0], pair[0]))
    index = 0
    equity = float(initial_equity)
    previous_equity = equity
    daily = []
    start_day = pd.Timestamp(start_ns, unit="ns", tz="UTC").floor("D")
    end_day = pd.Timestamp(end_ns, unit="ns", tz="UTC").floor("D")
    for day in pd.date_range(start_day, end_day, freq="D", tz="UTC"):
        cutoff = min(int((day + pd.Timedelta(days=1)).value) - 1, int(end_ns))
        while index < len(ordered) and ordered[index][1][0] <= cutoff:
            equity = float(ordered[index][1][1])
            index += 1
        daily_log = math.log(equity / previous_equity) if equity > 0.0 and previous_equity > 0.0 else float("-inf")
        daily.append({"date_utc": day.date().isoformat(), "equity_usd": equity, "daily_log_growth": daily_log})
        previous_equity = equity
    finite_or_inf = [item["daily_log_growth"] for item in daily]
    mean_growth = float(np.mean(finite_or_inf)) if finite_or_inf else 0.0
    return daily, mean_growth


def _summary(state: Portfolio, metrics: RunMetrics, end_ns: int) -> dict:
    if metrics.underwater_start_ns is not None:
        metrics.max_underwater_ns = max(metrics.max_underwater_ns, end_ns - metrics.underwater_start_ns)
    daily, mean_daily_log = _daily_growth(
        metrics.log_events, metrics.initial_equity, metrics.start_ns, end_ns
    )
    stakes = [float(item["gross_purchase_usd"]) for item in metrics.trades]
    shares_equity = [float(item["stake_to_equity_before"]) for item in metrics.trades if item["stake_to_equity_before"] is not None]
    return {
        "initial_cost_basis_equity_usd": float(metrics.initial_equity),
        "ending_cash_usd": float(state.cash),
        "ending_locked_cost_basis_usd": float(state.locked_cost),
        "ending_cost_basis_equity_usd": float(state.equity),
        "net_pnl_usd": float(state.equity - metrics.initial_equity),
        "trade_count": int(metrics.trade_count),
        "eligible_decisions_seen": int(metrics.decision_count),
        "gross_turnover_usd": float(metrics.gross_turnover),
        "cash_debit_total_usd": float(metrics.cash_debit_total),
        "fees_paid_usd": float(metrics.fees_usd),
        "legacy_fee_shares": float(metrics.fee_shares),
        "expected_net_profit_at_entry_usd": float(metrics.expected_net_profit),
        "mean_daily_log_growth": mean_daily_log,
        "daily_log_growth_days": len(daily),
        "max_drawdown_cost_basis": float(metrics.max_drawdown),
        "maximum_time_underwater_seconds": float(metrics.max_underwater_ns / 1e9),
        "maximum_concurrent_cost_basis_exposure_usd": float(metrics.max_locked_cost),
        "maximum_concurrent_positions": int(metrics.max_open_positions),
        "minimum_free_cash_usd": float(metrics.cash_min),
        "stake_usd_quantiles": {
            str(q): float(np.quantile(stakes, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
        } if stakes else {},
        "stake_to_cost_basis_equity_quantiles": {
            str(q): float(np.quantile(shares_equity, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
        } if shares_equity else {},
        "decisions_skipped_no_positive_policy_edge": int(metrics.no_trade_skips),
        "decisions_skipped_minimum_purchase": int(metrics.min_order_skips),
        "decisions_skipped_insufficient_cash": int(metrics.insufficient_cash_skips),
        "decisions_skipped_insufficient_top_of_book": int(metrics.liquidity_skip_orders),
        "executed_orders_limited_by_top_of_book": int(metrics.liquidity_limited_orders),
        "executed_orders_limited_by_cash": int(metrics.cash_limited_orders),
        "executed_orders_limited_by_policy_cap": int(metrics.policy_capped_orders),
        "daily_equity_path": daily,
    }


def _load_rows(config: dict, source_study: dict) -> tuple[list[dict], list[dict], dict]:
    data = pd.read_parquet(OPPORTUNITIES_PATH)
    if len(data) != int(source_study["coverage"]["archived_markets"]):
        raise AssertionError("Opportunity cache row count differs from the frozen study")
    if data["condition_id"].duplicated().any():
        raise AssertionError("Opportunity cache has duplicate condition IDs")
    eligible = data.loc[data["eligible"].astype(bool)].sort_values(
        "entry_time_utc", kind="stable"
    ).reset_index(drop=True)
    for side in ("up", "down"):
        required = [f"{side}_best_ask", f"{side}_best_ask_size_shares", f"{side}_ask_age_seconds"]
        if eligible[required].isna().any().any():
            raise AssertionError(f"Eligible opportunities lack {side.upper()} best-ask fields")
        if not eligible[f"{side}_best_ask"].between(0.0, 1.0, inclusive="neither").all():
            raise AssertionError(f"Eligible opportunities have invalid {side.upper()} asks")
        if not eligible[f"{side}_best_ask_size_shares"].gt(0.0).all():
            raise AssertionError(f"Eligible opportunities have nonpositive {side.upper()} ask size")
    if not eligible["fee_known"].astype(bool).all() or not eligible["no_future_event_at_entry"].astype(bool).all():
        raise AssertionError("Eligible opportunity set violates the frozen fee/lookahead filters")
    if not eligible[["up_ask_age_seconds", "down_ask_age_seconds"]].le(30.0).all().all():
        raise AssertionError("Eligible opportunity set violates the frozen ask-age cap")
    if not eligible["p_candidate_platt"].between(0.0, 1.0).all():
        raise AssertionError("Frozen candidate_platt probabilities are invalid")
    if not eligible["target_polymarket_up"].isin([0, 1]).all():
        raise AssertionError("Eligible opportunities lack official outcomes")
    for side in ("up", "down"):
        if not eligible[f"{side}_fill_5usd_depth_sufficient"].map(lambda value: value is True or value == True).all():
            raise AssertionError(f"Exact $5 reference fill is missing on eligible {side.upper()} rows")
    eligible["_entry_ns"] = pd.to_datetime(eligible["entry_time_utc"], utc=True).astype("int64")
    eligible["_release_ns"] = pd.to_datetime(eligible["capital_release_at_utc"], utc=True).astype("int64")
    records = eligible.to_dict(orient="records")
    folds = source_study["split_design_frozen_before_search"]["outer_folds"]
    outcome_available = pd.to_datetime(eligible["outcome_available_at_utc"], utc=True).astype("int64").to_numpy()
    for record, outcome_ns in zip(records, outcome_available):
        record["_outcome_ns"] = int(outcome_ns)
    for fold in folds:
        ev = fold["evaluation"]
        start = int(ev["start_index_in_eligible_order"])
        refit_ns = int(pd.Timestamp(fold["refit_at_utc"]).value)
        if int(records[start]["_entry_ns"]) != refit_ns:
            raise AssertionError(f"Frozen fold {fold['fold_id']} no longer aligns to the opportunity cache")
        validation = records[int(fold["selection_validation"]["start_index_in_eligible_order"]):start]
        known = sum(int(record["_outcome_ns"] < refit_ns) for record in validation)
        expected_known = int(fold["selection_validation"]["labels_known_strictly_before_refit"])
        if known != expected_known:
            raise AssertionError("Label availability count differs from the frozen validation split")
    metadata = {
        "archived_markets": int(len(data)),
        "eligible_markets": int(len(records)),
        "both_sides_with_best_ask_and_size": int(len(records)),
        "eligible_bbo_confirmation_status": eligible["bbo_confirmation_status"].value_counts().to_dict(),
        "top_level_capacity_quantiles_usd": {
            side: {str(q): float(value) for q, value in eligible[f"{side}_top_ask_capacity_usd"].quantile(
                [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            ).items()}
            for side in ("up", "down")
        },
        "minimum_order_metadata": {
            "shared_cache_order_min_size_unique_values": [5.0],
            "source_timestamp_aligned_to_entry": False,
            "used_as_historical_rule": False,
            "research_assumption": config["execution"]["minimum_purchase_assumption"],
        },
    }
    return records, folds, metadata


def _base_trial_frame(trials: list[dict], config_hash: str, existing_path: Path) -> pd.DataFrame:
    if existing_path.is_file():
        try:
            existing = pd.read_csv(existing_path)
            if (
                "search_config_sha256" in existing
                and existing["search_config_sha256"].eq(config_hash).all()
                and set(existing["trial_id"].astype(str)) == {trial["trial_id"] for trial in trials}
            ):
                return existing.set_index("trial_id").reindex([trial["trial_id"] for trial in trials]).reset_index()
        except (OSError, ValueError, pd.errors.ParserError):
            pass
    rows = []
    for trial in trials:
        rows.append({
            "trial_id": trial["trial_id"],
            "family": trial["family"],
            "parameters_json": json.dumps(trial["parameters"], sort_keys=True),
            "status": "pending",
            "search_config_sha256": config_hash,
            "validation_score_fold_1": np.nan,
            "validation_score_fold_2": np.nan,
            "validation_score_fold_3": np.nan,
            "validation_trades_fold_1": np.nan,
            "validation_trades_fold_2": np.nan,
            "validation_trades_fold_3": np.nan,
        })
    return pd.DataFrame(rows)


def _save_trials(frame: pd.DataFrame) -> None:
    temporary = TRIALS_PATH.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    temporary.replace(TRIALS_PATH)


def _best_trial(trials: list[dict], trial_frame: pd.DataFrame, fold_id: int, families: set[str] | None = None) -> dict:
    scores = trial_frame.set_index("trial_id")[f"validation_score_fold_{fold_id}"]
    candidates = [trial for trial in trials if families is None or trial["family"] in families]
    # Stable grid order makes no-trade the deterministic tie winner when all scores are zero.
    best = candidates[0]
    best_score = float(scores[best["trial_id"]])
    for candidate in candidates[1:]:
        score = float(scores[candidate["trial_id"]])
        if score > best_score + 1e-15:
            best, best_score = candidate, score
    return {**best, "validation_score": best_score}


def _new_metrics(state: Portfolio, start_ns: int) -> RunMetrics:
    return RunMetrics(initial_equity=state.equity, start_ns=int(start_ns), cash_min=state.cash)


def _policy_from_trial(trial: dict, portfolio_id: str) -> dict:
    return {**trial, "portfolio_id": portfolio_id}


def _exact_reference_fill(row: dict, side: str) -> dict:
    prefix = f"{side}_fill_5usd_"
    return {
        "gross_usd": float(row[prefix + "gross_usd"]),
        "gross_shares": float(row[prefix + "gross_shares"]),
        "net_shares": float(row[prefix + "net_shares"]),
        "fees_usd": float(row[prefix + "fee_usd"]),
        "fee_shares": float(row[prefix + "fee_shares"]),
        "cash_fee_usd": float(row[prefix + "cash_fee_usd"]),
        "cash_debit_usd": float(row[prefix + "cash_debit_usd"]),
        "price": float(row[f"{side}_best_ask"]),
    }


def _run_baseline(rows: list[dict], policy: dict, config: dict, start_ns: int, end_ns: int, include_logs=True):
    state = Portfolio()
    metrics = _new_metrics(state, start_ns)
    _simulate_policy(rows, policy, state, metrics, config, cutoff_ns=end_ns + 1, include_logs=include_logs)
    _close_positions(state, metrics, end_ns, inclusive=True)
    return state, metrics, _summary(state, metrics, end_ns)


def _score_trial(trial: dict, rows: list[dict], seed_state: Portfolio, config: dict, refit_ns: int) -> dict:
    state = seed_state.clone()
    start_ns = int(rows[0]["_entry_ns"]) if rows else int(refit_ns)
    metrics = _new_metrics(state, start_ns)
    policy = _policy_from_trial(trial, "validation")
    _simulate_policy(rows, policy, state, metrics, config, cutoff_ns=refit_ns, include_logs=False)
    _close_positions(state, metrics, refit_ns, inclusive=True)
    daily, score = _daily_growth(metrics.log_events, metrics.initial_equity, start_ns, refit_ns)
    return {
        "score": score,
        "validation_end_equity_usd": state.equity,
        "validation_cash_usd": state.cash,
        "validation_trades": metrics.trade_count,
        "validation_days": len(daily),
        "validation_open_positions": len(state.pending),
    }


def _trial_row_index(frame: pd.DataFrame, trial_id: str) -> int:
    positions = np.flatnonzero(frame.trial_id.astype(str).to_numpy() == trial_id)
    if len(positions) != 1:
        raise AssertionError(f"Expected one trial row for {trial_id}")
    return int(positions[0])


def _upsert_validation_result(frame: pd.DataFrame, trial: dict, fold_id: int, result: dict) -> None:
    index = _trial_row_index(frame, trial["trial_id"])
    frame.loc[index, f"validation_score_fold_{fold_id}"] = result["score"]
    frame.loc[index, f"validation_trades_fold_{fold_id}"] = int(result["validation_trades"])
    frame.loc[index, f"validation_end_equity_fold_{fold_id}_usd"] = float(result["validation_end_equity_usd"])
    frame.loc[index, f"validation_cash_fold_{fold_id}_usd"] = float(result["validation_cash_usd"])
    frame.loc[index, f"validation_open_positions_fold_{fold_id}"] = int(result["validation_open_positions"])
    done = all(pd.notna(frame.loc[index, f"validation_score_fold_{fold}"]) for fold in range(1, fold_id + 1))
    frame.loc[index, "status"] = f"validated_through_fold_{fold_id}" if done else "pending"


def _validation_profiles(trials: list[dict], rows: list[dict], config: dict, refit_ns: int) -> dict:
    profile_trial = next(trial for trial in trials if trial["trial_id"] == "fixed_5_roi_0")
    durations = []
    for _ in range(3):
        started = time.perf_counter()
        _score_trial(profile_trial, rows, Portfolio(), config, refit_ns)
        durations.append(time.perf_counter() - started)
    typical = float(np.median(durations))
    fold_evaluations = len(trials) * 3
    return {
        "profile_trial_id": profile_trial["trial_id"],
        "profile_repetitions": len(durations),
        "profile_trial_seconds": durations,
        "median_trial_seconds": typical,
        "grid_configurations": len(trials),
        "validation_blocks": 3,
        "projected_trial_fold_evaluations": fold_evaluations,
        "projected_grid_seconds": typical * fold_evaluations,
        "execution_workers": 1,
    }


def _append_trade_and_path_events(
    portfolio_id: str,
    metrics: RunMetrics,
    initial_time_ns: int,
    end_ns: int,
) -> tuple[list[dict], list[dict]]:
    events = [{
        "portfolio_id": portfolio_id,
        "timestamp_utc": pd.Timestamp(initial_time_ns, unit="ns", tz="UTC").isoformat(),
        "event": "evaluation_start",
        "condition_id": None,
        "free_cash_usd": None,
        "open_cost_basis_usd": None,
        "equity_usd": metrics.initial_equity,
        "drawdown_cost_basis": 0.0,
    }]
    # Reconstruct the exact cost-basis curve from entries and scheduled releases.
    state_cash = float(metrics.initial_equity)
    state_locked = 0.0
    pending = []
    sequence = 0
    trades_by_entry = sorted(metrics.trades, key=lambda trade: (pd.Timestamp(trade["entry_time_utc"]).value, trade["condition_id"]))
    peak = float(metrics.initial_equity)
    for trade in trades_by_entry:
        entry_ns = int(pd.Timestamp(trade["entry_time_utc"]).value)
        while pending and pending[0][0] <= entry_ns:
            release_ns, _, gross, payout, condition_id = heapq.heappop(pending)
            state_locked -= gross
            state_cash += payout
            equity = state_cash + state_locked
            peak = max(peak, equity)
            events.append({
                "portfolio_id": portfolio_id,
                "timestamp_utc": pd.Timestamp(release_ns, unit="ns", tz="UTC").isoformat(),
                "event": "capital_release",
                "condition_id": condition_id,
                "free_cash_usd": state_cash,
                "open_cost_basis_usd": state_locked,
                "equity_usd": equity,
                "drawdown_cost_basis": (peak - equity) / peak if peak > 0.0 else 0.0,
            })
        state_cash -= float(trade["cash_debit_usd"])
        gross = float(trade["gross_purchase_usd"])
        state_locked += gross
        sequence += 1
        release_ns = int(pd.Timestamp(trade["capital_release_at_utc"]).value)
        heapq.heappush(pending, (release_ns, sequence, gross, float(trade["payout_usd"]), trade["condition_id"]))
        equity = state_cash + state_locked
        peak = max(peak, equity)
        events.append({
            "portfolio_id": portfolio_id,
            "timestamp_utc": pd.Timestamp(entry_ns, unit="ns", tz="UTC").isoformat(),
            "event": "purchase",
            "condition_id": trade["condition_id"],
            "free_cash_usd": state_cash,
            "open_cost_basis_usd": state_locked,
            "equity_usd": equity,
            "drawdown_cost_basis": (peak - equity) / peak if peak > 0.0 else 0.0,
        })
    while pending and pending[0][0] <= end_ns:
        release_ns, _, gross, payout, condition_id = heapq.heappop(pending)
        state_locked -= gross
        state_cash += payout
        equity = state_cash + state_locked
        peak = max(peak, equity)
        events.append({
            "portfolio_id": portfolio_id,
            "timestamp_utc": pd.Timestamp(release_ns, unit="ns", tz="UTC").isoformat(),
            "event": "capital_release",
            "condition_id": condition_id,
            "free_cash_usd": state_cash,
            "open_cost_basis_usd": state_locked,
            "equity_usd": equity,
            "drawdown_cost_basis": (peak - equity) / peak if peak > 0.0 else 0.0,
        })
    events.append({
        "portfolio_id": portfolio_id,
        "timestamp_utc": pd.Timestamp(end_ns, unit="ns", tz="UTC").isoformat(),
        "event": "common_evaluation_end",
        "condition_id": None,
        "free_cash_usd": state_cash,
        "open_cost_basis_usd": state_locked,
        "equity_usd": state_cash + state_locked,
        "drawdown_cost_basis": (peak - (state_cash + state_locked)) / peak if peak > 0.0 else 0.0,
    })
    daily, _ = _daily_growth(metrics.log_events, metrics.initial_equity, initial_time_ns, end_ns)
    for item in daily:
        item["portfolio_id"] = portfolio_id
    return events, daily


def _make_outer_decision_logs(rows, trial, state_seed, config, refit_ns, portfolio_id):
    state = state_seed.clone()
    metrics = _new_metrics(state, int(rows[0]["_entry_ns"]) if rows else refit_ns)
    policy = _policy_from_trial(trial, portfolio_id)
    _simulate_policy(rows, policy, state, metrics, config, cutoff_ns=refit_ns, include_logs=True)
    return metrics.decisions


def run() -> None:
    started = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_study = json.loads(STUDY_PATH.read_text(encoding="utf-8"))
    rows, folds, data_audit = _load_rows(config, source_study)
    opportunity_hash = sha256_file(OPPORTUNITIES_PATH)
    trials = _trial_grid(config)
    trial_frame = _base_trial_frame(trials, config_hash, TRIALS_PATH)

    first_validation = rows[
        int(folds[0]["selection_validation"]["start_index_in_eligible_order"]):
        int(folds[0]["selection_validation"]["end_index_exclusive"])
    ]
    first_refit_ns = int(pd.Timestamp(folds[0]["refit_at_utc"]).value)
    runtime_profile = _validation_profiles(trials, first_validation, config, first_refit_ns)
    profile_path = {
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": config_hash,
        "opportunity_cache_sha256": opportunity_hash,
        "eligible_rows": len(rows),
        "trial_runtime_profile": runtime_profile,
        "execution_environment": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "policy_workers": 1,
            "active_background_processes_observed_before_run": [
                "run_btc_entry_state_collection.py", "run_btc_preopen_collection.py"
            ],
            "gpu_used": False,
        },
    }
    print(json.dumps({"stage": "profile_complete", **profile_path}, indent=2, ensure_ascii=False))

    experiment_status = "running_resumable_walk_forward"
    current_study = dict(source_study)
    current_study["status"] = experiment_status
    current_study["frozen_search_space"] = {
        **config["search_space"],
        "trial_budget": len(trials),
        "trials_run": 0,
        "trial_fold_evaluations": 0,
        "search_method": "finite deterministic grid; no stochastic optimizer",
        "search_config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "search_config_sha256": config_hash,
        "minimum_expected_return_definition": config["selection"]["threshold_metric"],
        "selection_rule": config["selection"]["side_selection"],
        "kelly_formula": config["selection"]["kelly"],
    }
    current_study["execution_assumption"] = config["execution"]
    current_study["depth_audit"]["reason_tuning_stopped"] = None
    current_study["depth_audit"]["replacement_execution_model"] = "best ask and saved quantity at that level only, with a fixed research minimum and no assumed liquidity beyond the recorded level"
    current_study["depth_audit"]["historical_order_minimum_verified_at_entry"] = False
    current_study["data_audit_for_sizing"] = data_audit
    current_study["trial_runtime_profile"] = runtime_profile
    current_study["walk_forward_progress"] = {"completed_folds": 0, "current_fold": None}
    current_study["inputs"].setdefault("files", {})[OPPORTUNITIES_PATH.relative_to(ROOT).as_posix()] = {
        "sha256": opportunity_hash,
        "size_bytes": OPPORTUNITIES_PATH.stat().st_size,
    }
    write_json_atomic(STUDY_PATH, current_study)
    _save_trials(trial_frame)

    opt_state = Portfolio()
    fixed_state = Portfolio()
    matched_fixed5_state = Portfolio()
    adaptive_metrics = _new_metrics(opt_state, int(rows[int(folds[0]["evaluation"]["start_index_in_eligible_order"])]["_entry_ns"]))
    fixed_metrics = _new_metrics(fixed_state, adaptive_metrics.start_ns)
    matched_metrics = _new_metrics(matched_fixed5_state, adaptive_metrics.start_ns)
    outer_start_states = {}
    selected_by_fold = []
    fixed_selected_by_fold = []
    matched_selected_by_fold = []
    block_reports = []

    for fold_position, fold in enumerate(folds):
        fold_id = int(fold["fold_id"])
        val_start = int(fold["selection_validation"]["start_index_in_eligible_order"])
        val_end = int(fold["selection_validation"]["end_index_exclusive"])
        eval_start = int(fold["evaluation"]["start_index_in_eligible_order"])
        eval_end = int(fold["evaluation"]["end_index_exclusive"])
        refit_ns = int(pd.Timestamp(fold["refit_at_utc"]).value)
        validation_rows = rows[val_start:val_end]
        if fold_position == 0:
            validation_seed = Portfolio()
        else:
            validation_seed = outer_start_states[fold_position - 1].clone()

        current_study["walk_forward_progress"] = {"completed_folds": fold_position, "current_fold": fold_id}
        write_json_atomic(STUDY_PATH, current_study)
        for trial in trials:
            index = _trial_row_index(trial_frame, trial["trial_id"])
            column = f"validation_score_fold_{fold_id}"
            if pd.notna(trial_frame.loc[index, column]):
                continue
            result = _score_trial(trial, validation_rows, validation_seed, config, refit_ns)
            _upsert_validation_result(trial_frame, trial, fold_id, result)
            _save_trials(trial_frame)

        selected = _best_trial(trials, trial_frame, fold_id)
        selected_fixed = _best_trial(trials, trial_frame, fold_id, {"fixed_stake"})
        matched_fixed5 = {
            "trial_id": f"matched_fixed_5_roi_{float(selected['parameters'].get('min_expected_return', 0.0)):g}",
            "family": "fixed_stake",
            "parameters": {
                "stake_usd": 5.0,
                "min_expected_return": float(selected["parameters"].get("min_expected_return", 0.0)),
            },
        }
        selected_by_fold.append({
            "fold_id": fold_id,
            "refit_at_utc": fold["refit_at_utc"],
            "selected_trial_id": selected["trial_id"],
            "selected_family": selected["family"],
            "selected_parameters": selected["parameters"],
            "inner_validation_mean_daily_log_growth": selected["validation_score"],
            "selection_validation_known_labels": int(fold["selection_validation"]["labels_known_strictly_before_refit"]),
        })
        fixed_selected_by_fold.append({
            "fold_id": fold_id,
            "selected_trial_id": selected_fixed["trial_id"],
            "selected_parameters": selected_fixed["parameters"],
            "inner_validation_mean_daily_log_growth": selected_fixed["validation_score"],
        })
        matched_selected_by_fold.append({
            "fold_id": fold_id,
            "selected_trial_id": matched_fixed5["trial_id"],
            "selected_parameters": matched_fixed5["parameters"],
        })

        if fold_position == 0:
            outer_start_states[fold_position] = opt_state.clone()
        else:
            # Capital and positions continue across the outer block boundary.
            _close_positions(opt_state, adaptive_metrics, refit_ns, inclusive=True)
            _close_positions(fixed_state, fixed_metrics, refit_ns, inclusive=True)
            _close_positions(matched_fixed5_state, matched_metrics, refit_ns, inclusive=True)
            outer_start_states[fold_position] = opt_state.clone()
        evaluation_rows = rows[eval_start:eval_end]
        block_start_ns = int(evaluation_rows[0]["_entry_ns"])
        block_end_ns = int(evaluation_rows[-1]["_entry_ns"])
        block_equity_start = opt_state.equity
        trades_before = len(adaptive_metrics.trades)
        policy = _policy_from_trial(selected, f"optimized_fold_{fold_id}")
        _simulate_policy(evaluation_rows, policy, opt_state, adaptive_metrics, config, include_logs=True)
        block_trades = adaptive_metrics.trades[trades_before:]

        fixed_policy = _policy_from_trial(selected_fixed, f"fixed_selection_fold_{fold_id}")
        fixed_before = len(fixed_metrics.trades)
        _simulate_policy(evaluation_rows, fixed_policy, fixed_state, fixed_metrics, config, include_logs=True)
        fixed_block_trades = fixed_metrics.trades[fixed_before:]

        matched_policy = _policy_from_trial(matched_fixed5, f"matched_fixed_5_fold_{fold_id}")
        matched_before = len(matched_metrics.trades)
        _simulate_policy(evaluation_rows, matched_policy, matched_fixed5_state, matched_metrics, config, include_logs=True)
        matched_block_trades = matched_metrics.trades[matched_before:]

        block_reports.append({
            "fold_id": fold_id,
            "refit_at_utc": fold["refit_at_utc"],
            "evaluation_start_utc": pd.Timestamp(block_start_ns, unit="ns", tz="UTC").isoformat(),
            "evaluation_end_utc": pd.Timestamp(block_end_ns, unit="ns", tz="UTC").isoformat(),
            "eligible_markets": len(evaluation_rows),
            "selected_trial_id": selected["trial_id"],
            "selected_family": selected["family"],
            "selected_parameters": selected["parameters"],
            "inner_validation_mean_daily_log_growth": selected["validation_score"],
            "optimized_start_cost_basis_equity_usd": block_equity_start,
            "optimized_pnl_of_purchases_entered_in_block_usd": float(sum(trade["realized_net_profit_usd"] for trade in block_trades)),
            "optimized_trades_entered_in_block": len(block_trades),
            "optimized_gross_turnover_usd": float(sum(trade["gross_purchase_usd"] for trade in block_trades)),
            "optimized_fees_usd": float(sum(trade["fees_usd"] for trade in block_trades)),
            "best_fixed_stake_trial_id": selected_fixed["trial_id"],
            "best_fixed_stake_parameters": selected_fixed["parameters"],
            "best_fixed_stake_inner_validation_mean_daily_log_growth": selected_fixed["validation_score"],
            "best_fixed_stake_trades_entered": len(fixed_block_trades),
            "best_fixed_stake_pnl_of_purchases_entered_usd": float(sum(trade["realized_net_profit_usd"] for trade in fixed_block_trades)),
            "matched_fixed_5_parameters": matched_fixed5["parameters"],
            "matched_fixed_5_trades_entered": len(matched_block_trades),
            "matched_fixed_5_pnl_of_purchases_entered_usd": float(sum(trade["realized_net_profit_usd"] for trade in matched_block_trades)),
        })
        current_study["walk_forward_progress"] = {"completed_folds": fold_id, "current_fold": fold_id}
        current_study["walk_forward_selections"] = selected_by_fold
        write_json_atomic(STUDY_PATH, current_study)
        print(json.dumps({"stage": "outer_block_complete", "fold_id": fold_id, "selected_trial_id": selected["trial_id"], "family": selected["family"], "parameters": selected["parameters"], "inner_score": selected["validation_score"], "trades": len(block_trades)}, ensure_ascii=False))

    evaluation_start_index = int(folds[0]["evaluation"]["start_index_in_eligible_order"])
    outer_rows = rows[evaluation_start_index:]
    outer_start_ns = int(outer_rows[0]["_entry_ns"])
    common_end_ns = max(int(row["_release_ns"]) for row in outer_rows)
    _close_positions(opt_state, adaptive_metrics, common_end_ns, inclusive=True)
    _close_positions(fixed_state, fixed_metrics, common_end_ns, inclusive=True)
    _close_positions(matched_fixed5_state, matched_metrics, common_end_ns, inclusive=True)
    optimized_summary = _summary(opt_state, adaptive_metrics, common_end_ns)
    fixed_summary = _summary(fixed_state, fixed_metrics, common_end_ns)
    matched_summary = _summary(matched_fixed5_state, matched_metrics, common_end_ns)

    new_baseline_policy = {"trial_id": "legacy_rule_fixed_5_best_level", "family": "legacy_best_level_fixed_5", "parameters": {}, "portfolio_id": "best_level_baseline_outer"}
    _, new_baseline_metrics, new_baseline_outer = _run_baseline(
        outer_rows, new_baseline_policy, config, outer_start_ns, common_end_ns, include_logs=True
    )
    exact_outer_policy = {"trial_id": "legacy_rule_fixed_5_exact_fill", "family": "legacy_exact_fixed_5", "parameters": {}, "portfolio_id": "exact_fill_baseline_outer"}
    _, exact_outer_metrics, exact_outer_summary = _run_baseline(
        outer_rows, exact_outer_policy, config, outer_start_ns, common_end_ns, include_logs=True
    )
    full_start_ns = int(rows[0]["_entry_ns"])
    full_end_ns = max(int(row["_release_ns"]) for row in rows)
    exact_full_policy = {"trial_id": "legacy_rule_fixed_5_exact_fill", "family": "legacy_exact_fixed_5", "parameters": {}, "portfolio_id": "exact_fill_baseline_full"}
    _, exact_full_metrics, exact_full_summary = _run_baseline(
        rows, exact_full_policy, config, full_start_ns, full_end_ns, include_logs=True
    )
    top_full_policy = {"trial_id": "legacy_rule_fixed_5_best_level", "family": "legacy_best_level_fixed_5", "parameters": {}, "portfolio_id": "best_level_baseline_full"}
    _, _, top_full_summary = _run_baseline(
        rows, top_full_policy, config, full_start_ns, full_end_ns, include_logs=False
    )
    expected_reference = float(source_study["baseline_reproduction"]["net_pnl_usd"])
    if not math.isclose(exact_full_summary["net_pnl_usd"], expected_reference, rel_tol=0.0, abs_tol=1e-8):
        raise AssertionError(
            f"Historical exact-fill baseline failed reproduction: {exact_full_summary['net_pnl_usd']} != {expected_reference}"
        )
    if exact_full_summary["trade_count"] != int(source_study["baseline_reproduction"]["trade_count"]):
        raise AssertionError("Historical exact-fill baseline trade count failed reproduction")

    # Merge comparable outer-evaluation trade and decision logs after every block has been evaluated.
    policy_trade_logs = []
    for portfolio_id, metrics in (
        ("optimized_walk_forward", adaptive_metrics),
        ("best_fixed_stake_selection", fixed_metrics),
        ("matched_fixed_5_control", matched_metrics),
        ("best_level_baseline", new_baseline_metrics),
        ("exact_fill_baseline_outer", exact_outer_metrics),
    ):
        for trade in metrics.trades:
            item = dict(trade)
            item["portfolio_id"] = portfolio_id
            policy_trade_logs.append(item)
    pd.DataFrame(policy_trade_logs).to_parquet(TRADE_LOG_PATH, index=False, compression="zstd")
    policy_decisions = []
    for metrics in (adaptive_metrics, fixed_metrics, matched_metrics, new_baseline_metrics, exact_outer_metrics):
        policy_decisions.extend(metrics.decisions)
    pd.DataFrame(policy_decisions).to_parquet(DECISION_LOG_PATH, index=False, compression="zstd")

    path_events, daily_rows = [], []
    for portfolio_id, metrics in (
        ("optimized_walk_forward", adaptive_metrics),
        ("best_fixed_stake_selection", fixed_metrics),
        ("matched_fixed_5_control", matched_metrics),
        ("best_level_baseline", new_baseline_metrics),
        ("exact_fill_baseline_outer", exact_outer_metrics),
    ):
        events, daily = _append_trade_and_path_events(portfolio_id, metrics, outer_start_ns, common_end_ns)
        path_events.extend(events)
        daily_rows.extend(daily)
    pd.DataFrame(path_events).to_parquet(EQUITY_PATH_PATH, index=False, compression="zstd")
    pd.DataFrame(daily_rows).to_parquet(DAILY_PATH, index=False, compression="zstd")

    selected_final = selected_by_fold[-1]
    final_trial = next(trial for trial in trials if trial["trial_id"] == selected_final["selected_trial_id"])
    final_policy = {
        "schema_version": 1,
        "policy_id": "candidate_platt_t59_walk_forward_development_choice_20261007",
        "status": "inactive_research_candidate",
        "active_for_trading": False,
        "evaluation_label": config["evaluation_label"],
        "selected_from_development_only": True,
        "development_refit_at_utc": selected_by_fold[-1]["refit_at_utc"],
        "development_validation_block": folds[-1]["selection_validation"],
        "selected_trial_id": final_trial["trial_id"],
        "family": final_trial["family"],
        "parameters": final_trial["parameters"],
        "validation_mean_daily_log_growth": selected_final["inner_validation_mean_daily_log_growth"],
        "model": source_study["inputs"]["candidate"],
        "decision": {
            "decision_time": "single T-59s decision per corrected eligible market",
            "prediction": "frozen candidate_platt probability; no refit or recalibration",
            "threshold": "expected net profit / full cash debit at the actually feasible amount",
            "side_selection": config["selection"]["side_selection"],
            "sizing": config["selection"]["kelly"] if final_trial["family"] == "fractional_kelly" else "Use the configured fixed gross amount or fraction of free cash, then apply cash, top-of-book, minimum purchase, rounding, and $20 caps.",
        },
        "execution": config["execution"],
        "portfolio": {
            "initial_cash_usd": INITIAL_CASH_USD,
            "no_credit": True,
            "one_purchase_per_market": True,
            "hold_to_resolution": True,
            "capital_release": config["selection"]["capital_release"],
            "equity_and_drawdown": config["selection"]["portfolio_equity_for_sizing_and_drawdown"],
        },
        "source_fingerprints": {
            "opportunities_sha256": opportunity_hash,
            "search_config_sha256": config_hash,
        },
        "live_or_shadow_activation": False,
    }
    write_json_atomic(FINAL_POLICY_PATH, final_policy)

    reference_policy_path = REPORT_DIR / "policy.json"
    reference_policy = json.loads(reference_policy_path.read_text(encoding="utf-8"))
    reference_policy["status"] = "historical_fixed_5_reference; policy optimization completed separately"
    reference_policy["selection_result"] = "This artifact preserves the historical exact-fill fixed-$5 reference. The separate policy_final_research_20261007.json is the development-selected, inactive research candidate; this reference remains the comparison baseline."
    reference_policy.setdefault("evaluation", {})["optimization_assessment"] = "reports/btc_preopen/policy_study.json"
    reference_policy["evaluation"]["walk_forward_candidate"] = None
    write_json_atomic(reference_policy_path, reference_policy)

    optimization_result = {
        "evaluation_period": {
            "start_utc": pd.Timestamp(outer_start_ns, unit="ns", tz="UTC").isoformat(),
            "last_entry_utc": pd.Timestamp(int(outer_rows[-1]["_entry_ns"]), unit="ns", tz="UTC").isoformat(),
            "common_settlement_end_utc": pd.Timestamp(common_end_ns, unit="ns", tz="UTC").isoformat(),
            "eligible_market_rows": len(outer_rows),
            "external_blocks_share_one_continuous_portfolio": True,
        },
        "historical_exact_fill_baseline_full_period": exact_full_summary,
        "best_level_old_rule_fixed_5_baseline_full_period": top_full_summary,
        "outer_comparisons_same_period_and_execution": {
            "best_level_old_rule_fixed_5": new_baseline_outer,
            "exact_fill_old_rule_fixed_5_control": exact_outer_summary,
            "selected_fixed_stake_policy": fixed_summary,
            "matched_fixed_5_same_selected_threshold": matched_summary,
            "walk_forward_selected_family_and_sizing": optimized_summary,
        },
        "block_results": block_reports,
        "selected_walk_forward_policies": selected_by_fold,
        "selected_fixed_stake_policies": fixed_selected_by_fold,
        "matched_fixed_5_control_policies": matched_selected_by_fold,
        "best_level_simulator_effect_outer_period_usd": float(new_baseline_outer["net_pnl_usd"] - exact_outer_summary["net_pnl_usd"]),
        "optimized_policy_effect_vs_best_level_baseline_outer_period_usd": float(optimized_summary["net_pnl_usd"] - new_baseline_outer["net_pnl_usd"]),
        "variable_sizing_effect_vs_matched_fixed_5_outer_period_usd": float(optimized_summary["net_pnl_usd"] - matched_summary["net_pnl_usd"]),
        "artifacts": {
            "trials": TRIALS_PATH.relative_to(ROOT).as_posix(),
            "trades": TRADE_LOG_PATH.relative_to(ROOT).as_posix(),
            "decisions": DECISION_LOG_PATH.relative_to(ROOT).as_posix(),
            "equity_events": EQUITY_PATH_PATH.relative_to(ROOT).as_posix(),
            "daily_equity": DAILY_PATH.relative_to(ROOT).as_posix(),
            "final_inactive_policy": FINAL_POLICY_PATH.relative_to(ROOT).as_posix(),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    current_study["status"] = "completed_retrospective_walk_forward"
    current_study["walk_forward_progress"] = {"completed_folds": len(folds), "current_fold": None}
    current_study["frozen_search_space"]["trials_run"] = len(trials)
    current_study["frozen_search_space"]["trial_fold_evaluations"] = len(trials) * len(folds)
    current_study["walk_forward_selections"] = selected_by_fold
    current_study["walk_forward_results"] = optimization_result
    current_study["policy_chosen"] = {
        "trial_id": final_trial["trial_id"],
        "family": final_trial["family"],
        "parameters": final_trial["parameters"],
        "artifact": FINAL_POLICY_PATH.relative_to(ROOT).as_posix(),
        "active_for_trading": False,
    }
    current_study["reason_policy_not_selected"] = None
    current_study["elapsed_seconds"] = optimization_result["elapsed_seconds"]
    write_json_atomic(STUDY_PATH, current_study)
    print(json.dumps({"status": current_study["status"], "selected_policy": current_study["policy_chosen"], "outer_results": optimization_result["outer_comparisons_same_period_and_execution"], "simulator_effect_usd": optimization_result["best_level_simulator_effect_outer_period_usd"], "policy_effect_usd": optimization_result["optimized_policy_effect_vs_best_level_baseline_outer_period_usd"], "variable_sizing_effect_usd": optimization_result["variable_sizing_effect_vs_matched_fixed_5_outer_period_usd"], "artifacts": optimization_result["artifacts"], "elapsed_seconds": optimization_result["elapsed_seconds"]}, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    run()
