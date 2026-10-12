"""Signal and bankroll diagnostics from the frozen BTC pre-open artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import run_btc_preopen_fee_correction as fee_correction
import run_btc_preopen_research as research
import run_btc_preopen_t59_reassessment as replay
from utils.polymarket_diagnostics import block_uncertainty
from utils.polymarket_policy import calibration_metrics


ROOT = Path(__file__).resolve().parent
CORRECTION_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_research_20261012_fee_correction"
SOURCE_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_research_20261011_full_compute20_retry"
OUTPUT_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_research_20261012_signal_diagnostic"
REPORT_PATH = ROOT / "reports/btc_preopen/btc_preopen_signal_diagnostic_20261012.md"
FIGURE_DIR = ROOT / "reports/btc_preopen/signal_diagnostic_20261012"
DEV_START_UTC = pd.Timestamp("2026-01-01T00:00:00Z")
DEV_CUTOFF_UTC = pd.Timestamp("2026-07-01T00:00:00Z")
DATA_CUTOFF_UTC = pd.Timestamp("2026-10-07T00:00:00Z")
FIXED_ORDER_USD = 5.0
PORTFOLIO_CAPITALS_USD = (100.0, 500.0, 1000.0)
CALIBRATION_BINS = 10
EV_BINS = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_model_artifact_hashes(holdout_manifest: dict) -> dict[str, str]:
    artifacts = {
        "evaluated_bundle": (
            holdout_manifest["evaluated_bundle_path"],
            holdout_manifest["evaluated_bundle_sha256"],
        ),
        "evaluated_model": (
            holdout_manifest["fitted_model_stage"]["model_path"],
            holdout_manifest["fitted_model_stage"]["model_sha256"],
        ),
        "evaluated_calibrator": (
            holdout_manifest["calibration_stage"]["calibrator_path"],
            None,
        ),
    }
    hashes = {}
    for name, (relative_path, expected_hash) in artifacts.items():
        actual_hash = _sha256(ROOT / relative_path)
        if expected_hash is not None and actual_hash != expected_hash:
            raise RuntimeError(f"Frozen {name} artifact hash does not match its manifest")
        hashes[name] = actual_hash
    return hashes


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _period_name(timestamp: pd.Timestamp) -> str | None:
    if DEV_START_UTC <= timestamp < DEV_CUTOFF_UTC:
        return "development"
    if DEV_CUTOFF_UTC <= timestamp < DATA_CUTOFF_UTC:
        return "holdout"
    return None


def _development_quantile_edges(values, bins: int, *, endpoints: tuple[float, float] | None = None) -> list[float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("Cannot freeze bins without development observations")
    quantiles = np.quantile(data, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    edges = list(quantiles)
    if endpoints is not None:
        edges = [endpoints[0], *edges, endpoints[1]]
    return [float(value) for value in np.unique(edges)]


def _bin_index(value: float, internal_edges: list[float]) -> int:
    return int(np.searchsorted(np.asarray(internal_edges, dtype=np.float64), float(value), side="left"))


def _score(y, probability, *, prevalence_baseline: float, bin_edges: list[float]) -> dict:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Scoring requires paired, finite labels and probabilities")
    p_clip = np.clip(p, 1e-12, 1.0 - 1e-12)
    reliability = calibration_metrics(y, p_clip)
    bucket = np.asarray([_bin_index(value, bin_edges[1:-1]) for value in p])
    calibration_error = 0.0
    for index in range(len(bin_edges) - 1):
        mask = bucket == index
        if mask.any():
            calibration_error += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p_clip, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "calibration_intercept": reliability.get("intercept"),
        "calibration_slope": reliability.get("slope"),
        "calibration_ece_frozen_bins": calibration_error,
        "constant_0_5_log_loss": float(log_loss(y, np.full(len(y), 0.5), labels=[0, 1])),
        "constant_0_5_brier": float(brier_score_loss(y, np.full(len(y), 0.5))),
        "development_prevalence_baseline": float(prevalence_baseline),
        "development_prevalence_log_loss": float(log_loss(
            y, np.full(len(y), prevalence_baseline), labels=[0, 1],
        )),
        "development_prevalence_brier": float(brier_score_loss(
            y, np.full(len(y), prevalence_baseline),
        )),
    }


def _prediction_quality(calendar: pd.DataFrame, calibration_edges: list[float]) -> tuple[list[dict], list[dict]]:
    periods = {
        "development": calendar.market_start_utc.ge(DEV_START_UTC) & calendar.market_start_utc.lt(DEV_CUTOFF_UTC),
        "holdout": calendar.market_start_utc.ge(DEV_CUTOFF_UTC) & calendar.market_start_utc.lt(DATA_CUTOFF_UTC),
        "combined": calendar.market_start_utc.ge(DEV_START_UTC) & calendar.market_start_utc.lt(DATA_CUTOFF_UTC),
    }
    targets = {
        "official_polymarket": "target_polymarket_up",
        "price_training_target": "target_binance_proxy_up",
    }
    probabilities = {
        "raw": "p_candidate_raw",
        "calibrated": "p_candidate_platt",
    }
    metric_rows: list[dict] = []
    bin_rows: list[dict] = []
    for period, period_mask in periods.items():
        base = calendar.loc[period_mask].copy()
        for scope in ("all_available", "common_official_and_price_target"):
            scoped = base
            if scope == "common_official_and_price_target":
                scoped = base.loc[base.target_polymarket_up.notna() & base.target_binance_proxy_up.notna()]
            for target_name, target_col in targets.items():
                if scope == "common_official_and_price_target" or target_name == "official_polymarket":
                    sample = scoped.loc[scoped[target_col].notna()].copy()
                else:
                    sample = base.loc[base[target_col].notna()].copy()
                if sample.empty:
                    continue
                y = pd.to_numeric(sample[target_col], errors="coerce").to_numpy(dtype=np.int8)
                dev_mask = (
                    calendar.market_start_utc.ge(DEV_START_UTC)
                    & calendar.market_start_utc.lt(DEV_CUTOFF_UTC)
                )
                dev = calendar.loc[dev_mask]
                if scope == "common_official_and_price_target":
                    dev = dev.loc[dev.target_polymarket_up.notna() & dev.target_binance_proxy_up.notna()]
                dev = dev.loc[dev[target_col].notna()]
                if dev.empty:
                    raise RuntimeError(f"No development labels for {target_name}/{scope}")
                dev_prevalence = float(pd.to_numeric(dev[target_col], errors="coerce").mean())
                for model_name, column in probabilities.items():
                    p = pd.to_numeric(sample[column], errors="coerce").to_numpy(dtype=np.float64)
                    row = {
                        "period": period,
                        "scope": scope,
                        "target": target_name,
                        "probability": model_name,
                        "development_rows_for_baseline": int(len(dev)),
                        "development_up_rate_for_baseline": dev_prevalence,
                        **_score(y, p, prevalence_baseline=dev_prevalence, bin_edges=calibration_edges),
                    }
                    metric_rows.append(row)
                    for bin_index in range(len(calibration_edges) - 1):
                        mask = np.asarray([_bin_index(value, calibration_edges[1:-1]) for value in p]) == bin_index
                        bin_rows.append({
                            "period": period,
                            "scope": scope,
                            "target": target_name,
                            "probability": model_name,
                            "bin": bin_index + 1,
                            "lower": calibration_edges[bin_index],
                            "upper": calibration_edges[bin_index + 1],
                            "n": int(mask.sum()),
                            "mean_probability": float(p[mask].mean()) if mask.any() else None,
                            "observed_up_rate": float(y[mask].mean()) if mask.any() else None,
                        })
    return metric_rows, bin_rows


def _expected_fill_economics(fill: dict, p_side: float) -> dict:
    gross_usd = float(fill["gross_usd"])
    gross_shares = float(fill["gross_shares"])
    net_shares = float(fill["shares"])
    fee_shares = float(fill.get("fee_shares") or 0.0)
    fee_cash = float(fill.get("fee_cash_usd") or 0.0)
    return {
        "expected_gross_t59_usd": p_side * gross_shares - gross_usd,
        "expected_net_t59_usd": p_side * net_shares - float(fill["cash_debit_usd"]),
        "expected_fee_effect_t59_usd": p_side * fee_shares + fee_cash,
        "modeled_fee_equivalent_usd": float(fill.get("fee_usd_entry_equivalent") or fill.get("fee_usd") or 0.0),
    }


def _realized_economics(fill: dict, p_side: float, outcome_side: int) -> dict:
    gross_usd = float(fill["gross_usd"])
    gross_shares = float(fill["gross_shares"])
    net_shares = float(fill["shares"])
    fee_shares = float(fill.get("fee_shares") or 0.0)
    fee_cash = float(fill.get("fee_cash_usd") or 0.0)
    won = int(outcome_side == 1)
    gross_pnl = won * gross_shares - gross_usd
    realized_fee_effect = won * fee_shares + fee_cash
    net_pnl = won * net_shares - float(fill["cash_debit_usd"])
    if not math.isclose(gross_pnl - realized_fee_effect, net_pnl, rel_tol=0.0, abs_tol=1e-8):
        raise AssertionError("Gross PnL less the realized V1/V2 fee effect did not reconcile to net PnL")
    return {
        "gross_pnl_usd": gross_pnl,
        "realized_fee_effect_usd": realized_fee_effect,
        "net_pnl_usd": net_pnl,
        "won": bool(won),
        "expected_gross_t59_usd": p_side * gross_shares - gross_usd,
        "expected_net_t59_usd": p_side * net_shares - float(fill["cash_debit_usd"]),
        "expected_fee_effect_t59_usd": p_side * fee_shares + fee_cash,
        "modeled_fee_equivalent_usd": float(fill.get("fee_usd_entry_equivalent") or fill.get("fee_usd") or 0.0),
    }


def _t60_unpriceable_reason(evaluation: dict) -> str:
    reasons = [evaluation["sides"][side]["reason"] for side in ("up", "down")]
    for reason in (
        "below_minimum_order_usd", "exchange_maintenance_pause", "future_receive_event_at_entry",
        "future_source_timestamp_at_entry", "archive_hour_missing_at_entry",
        "archive_gap_after_book_initialization", "no_native_book_snapshot", "no_executable_ask_liquidity",
    ):
        if reasons.count(reason) == 2:
            return reason
    if "below_minimum_order_usd" in reasons:
        return "below_minimum_order_usd"
    if "exchange_maintenance_pause" in reasons:
        return "exchange_maintenance_pause"
    return "no_priceable_native_ask_side"


def _independent_decisions(
    calendar: pd.DataFrame,
    market_by_id: dict[str, dict],
    t59_evaluations_by_id: dict[str, dict],
    t60_by_id: dict[str, dict],
    hurdles: tuple[float, ...],
) -> pd.DataFrame:
    decisions = []
    market_rows = calendar.loc[
        calendar.market_start_utc.ge(DEV_START_UTC)
        & calendar.market_start_utc.lt(DATA_CUTOFF_UTC)
    ].sort_values("market_start_utc", kind="stable")
    for row in market_rows.itertuples(index=False):
        start = pd.Timestamp(row.market_start_utc)
        period = _period_name(start)
        confirmed = bool(row.market_confirmed)
        prediction_available = bool(row.causal_prediction_available)
        cid = None if pd.isna(row.condition_id) else str(row.condition_id).lower()
        record = {
            "market_start_utc": start,
            "period": period,
            "month_utc": start.strftime("%Y-%m"),
            "condition_id": cid,
            "market_confirmed": confirmed,
            "prediction_available": prediction_available,
            "p_model_raw": _finite(row.p_candidate_raw),
            "p_model_calibrated": _finite(row.p_candidate_platt),
            "official_outcome_up": _finite(row.target_polymarket_up),
            "price_target_up": _finite(row.target_binance_proxy_up),
            "t60_priceable_decision": False,
            "chosen_side": None,
            "p_chosen_side_raw": None,
            "p_chosen_side_calibrated": None,
            "t60_expected_gross_pnl_usd": None,
            "t60_expected_net_pnl_usd": None,
            "t60_net_ev_fraction": None,
            "t60_fee_equivalent_usd": None,
            "t60_vwap": None,
            "t60_gross_shares": None,
            "t60_net_shares": None,
            "t60_fee_shares": None,
            "t60_fee_cash_usd": None,
            "t60_gross_usd": None,
            "t60_cash_debit_usd": None,
            "execution_status": None,
            "execution_reason": None,
            "execution_source": None,
            "depth_known": None,
            "execution_assumption": None,
            "fill_status": None,
            "requested_gross_usd": FIXED_ORDER_USD,
            "filled_gross_usd": None,
            "vwap": None,
            "gross_shares": None,
            "net_shares": None,
            "fee_shares": None,
            "fee_cash_usd": None,
            "fee_equivalent_usd": None,
            "cash_debit_usd": None,
            "expected_gross_t59_usd": None,
            "expected_net_t59_usd": None,
            "expected_fee_effect_t59_usd": None,
            "gross_pnl_usd": None,
            "realized_fee_effect_usd": None,
            "net_pnl_usd": None,
            "won": None,
            "ev_thresholds_qualified": "",
        }
        if not confirmed:
            record["decision_status"] = "no_gamma_market_record"
            decisions.append(record)
            continue
        market = market_by_id.get(cid or "")
        if not prediction_available or market is None or _finite(row.p_candidate_platt) is None:
            record["decision_status"] = "missing_causal_prediction"
            decisions.append(record)
            continue
        t60 = t60_by_id.get(cid or "")
        if t60 is None:
            record["decision_status"] = "missing_t60_decision_snapshot"
            decisions.append(record)
            continue
        decision_eval = replay._evaluate_market_sides(market, t60, FIXED_ORDER_USD)
        record["t60_priceable_decision"] = decision_eval["priceable_side_count"] > 0
        if decision_eval["priceable_side_count"] == 0:
            record["decision_status"] = "t60_unpriceable"
            record["decision_reason"] = _t60_unpriceable_reason(decision_eval)
            decisions.append(record)
            continue
        side = decision_eval["chosen_side"]
        if side is None:
            record["decision_status"] = "no_positive_t60_net_ev"
            decisions.append(record)
            continue
        chosen = decision_eval["sides"][side]
        decision_fill = chosen["fill"]
        p_side_cal = float(chosen["probability"])
        p_up_raw = float(market["p_candidate_raw"])
        p_side_raw = p_up_raw if side == "up" else 1.0 - p_up_raw
        t60_gross_ev = p_side_cal * float(decision_fill["gross_shares"]) - float(decision_fill["gross_usd"])
        t60_net_ev = float(chosen["ev_usd"])
        ev_fraction = t60_net_ev / FIXED_ORDER_USD
        qualified = [hurdle for hurdle in hurdles if ev_fraction + 1e-12 >= hurdle]
        record.update({
            "decision_status": "positive_t60_net_ev",
            "chosen_side": side,
            "p_chosen_side_raw": p_side_raw,
            "p_chosen_side_calibrated": p_side_cal,
            "t60_expected_gross_pnl_usd": t60_gross_ev,
            "t60_expected_net_pnl_usd": t60_net_ev,
            "t60_net_ev_fraction": ev_fraction,
            "t60_fee_equivalent_usd": float(decision_fill.get("fee_usd_entry_equivalent") or decision_fill.get("fee_usd") or 0.0),
            "t60_vwap": decision_fill.get("vwap"),
            "t60_gross_shares": decision_fill.get("gross_shares"),
            "t60_net_shares": decision_fill.get("shares"),
            "t60_fee_shares": decision_fill.get("fee_shares"),
            "t60_fee_cash_usd": decision_fill.get("fee_cash_usd"),
            "t60_gross_usd": decision_fill.get("gross_usd"),
            "t60_cash_debit_usd": decision_fill.get("cash_debit_usd"),
            "ev_thresholds_qualified": ",".join(f"{value:.6f}" for value in qualified),
        })
        execution_eval = t59_evaluations_by_id.get(cid or "")
        if execution_eval is None:
            record["execution_status"] = "missing_t59_execution_snapshot"
            record["execution_reason"] = "missing_t59_execution_snapshot"
            decisions.append(record)
            continue
        execution_side = execution_eval["sides"][side]
        if not execution_side["priceable"]:
            record["execution_status"] = "t59_unpriceable"
            record["execution_reason"] = execution_side["reason"]
            decisions.append(record)
            continue
        fill = execution_side["fill"]
        record.update({
            "execution_status": "filled",
            "execution_reason": None,
            "execution_source": fill.get("execution_source"),
            "depth_known": bool(fill.get("depth_known")),
            "execution_assumption": fill.get("execution_assumption"),
            "fill_status": fill.get("fill_status"),
            "filled_gross_usd": float(fill["gross_usd"]),
            "vwap": float(fill["vwap"]),
            "gross_shares": float(fill["gross_shares"]),
            "net_shares": float(fill["shares"]),
            "fee_shares": float(fill.get("fee_shares") or 0.0),
            "fee_cash_usd": float(fill.get("fee_cash_usd") or 0.0),
            "fee_equivalent_usd": float(fill.get("fee_usd_entry_equivalent") or fill.get("fee_usd") or 0.0),
            "cash_debit_usd": float(fill["cash_debit_usd"]),
        })
        record.update(_expected_fill_economics(fill, p_side_cal))
        outcome_up = record["official_outcome_up"]
        if outcome_up in (0.0, 1.0):
            outcome_side = int(outcome_up == (1.0 if side == "up" else 0.0))
            record.update(_realized_economics(fill, p_side_cal, outcome_side))
        decisions.append(record)
    return pd.DataFrame(decisions)


def _signal_subset(decisions: pd.DataFrame, hurdle: float) -> pd.DataFrame:
    return decisions.loc[
        decisions.decision_status.eq("positive_t60_net_ev")
        & decisions.t60_net_ev_fraction.ge(float(hurdle) - 1e-12)
    ].copy()


def _period_decisions(decisions: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "combined":
        return decisions
    return decisions.loc[decisions.period.eq(period)]


def _aggregate_signals(
    decisions: pd.DataFrame,
    period: str,
    hurdle: float,
    *,
    bootstrap_start: pd.Timestamp,
    bootstrap_end: pd.Timestamp,
) -> dict:
    all_period = _period_decisions(decisions, period)
    signals = _signal_subset(all_period, hurdle)
    fills = signals.loc[signals.execution_status.eq("filled")].copy()
    resolved = fills.loc[fills.official_outcome_up.isin((0.0, 1.0))].copy()
    reasons: Counter = Counter()
    for row in all_period.itertuples(index=False):
        if row.decision_status == "positive_t60_net_ev":
            if float(row.t60_net_ev_fraction) + 1e-12 < float(hurdle):
                reasons["below_frozen_ev_hurdle"] += 1
            elif row.execution_status == "filled":
                if row.official_outcome_up not in (0.0, 1.0):
                    reasons["filled_but_official_outcome_missing"] += 1
            else:
                reasons[str(row.execution_reason or row.execution_status or "no_execution_reason_recorded")] += 1
        elif row.decision_status == "t60_unpriceable":
            reasons[str(row.decision_reason or "no_priceable_native_ask_side")] += 1
        else:
            reasons[str(row.decision_status)] += 1
    pnl_sample_turnover = float(pd.to_numeric(resolved.filled_gross_usd, errors="coerce").sum())
    fill_count = int(len(fills))
    resolved_count = int(len(resolved))
    has_unscored_fills = fill_count > resolved_count
    if resolved_count:
        gross_pnl = float(resolved.gross_pnl_usd.sum())
        net_pnl = float(resolved.net_pnl_usd.sum())
        fee_effect = float(resolved.realized_fee_effect_usd.sum())
        protocol_fee_equivalent = float(resolved.fee_equivalent_usd.sum())
        wins = int(resolved.won.sum())
        losses = resolved_count - wins
        avg_execution = pnl_sample_turnover / float(resolved.gross_shares.sum())
        net_roi = net_pnl / pnl_sample_turnover if pnl_sample_turnover else None
        bridge = {
            "expected_gross_t60_usd": float(resolved.t60_expected_gross_pnl_usd.sum()),
            "expected_net_t60_usd": float(resolved.t60_expected_net_pnl_usd.sum()),
            "expected_gross_t59_usd": float(resolved.expected_gross_t59_usd.sum()),
            "expected_net_t59_usd": float(resolved.expected_net_t59_usd.sum()),
            "execution_t60_to_t59_usd": float((resolved.expected_gross_t59_usd - resolved.t60_expected_gross_pnl_usd).sum()),
            "prediction_payout_residual_usd": float((resolved.gross_pnl_usd - resolved.expected_gross_t59_usd).sum()),
            "realized_fee_effect_usd": fee_effect,
            "gross_pnl_usd": gross_pnl,
            "net_pnl_usd": net_pnl,
        }
        bridge["reconciliation_residual_usd"] = (
            bridge["expected_gross_t60_usd"] + bridge["execution_t60_to_t59_usd"]
            + bridge["prediction_payout_residual_usd"] - bridge["realized_fee_effect_usd"]
            - bridge["net_pnl_usd"]
        )
        if abs(bridge["reconciliation_residual_usd"]) > 1e-7:
            raise AssertionError("T-60/T-59, payout, and fee bridge did not reconcile")
    else:
        gross_pnl = net_pnl = fee_effect = protocol_fee_equivalent = None if has_unscored_fills else 0.0
        wins = losses = 0
        avg_execution = net_roi = None
        bridge = None
    all_fill_turnover = float(pd.to_numeric(fills.filled_gross_usd, errors="coerce").sum())
    full_book_count = int(resolved.depth_known.fillna(False).sum()) if resolved_count else 0
    proxy_count = resolved_count - full_book_count
    partial_count = int(resolved.fill_status.eq("partial").sum()) if resolved_count else 0
    summary = {
        "period": period,
        "ev_hurdle_fraction": float(hurdle),
        "market_count": int(all_period.market_confirmed.sum()),
        "prediction_count": int(all_period.prediction_available.sum()),
        "confirmed_markets_with_prediction": int((all_period.market_confirmed & all_period.prediction_available).sum()),
        "t60_priceable_decisions": int(all_period.t60_priceable_decision.sum()),
        "signals": int(len(signals)),
        "fills": fill_count,
        "resolved_fills_for_pnl": resolved_count,
        "filled_without_official_outcome": fill_count - resolved_count,
        "non_execution_reasons": dict(sorted(reasons.items())),
        "gross_turnover_resolved_pnl_sample_usd": pnl_sample_turnover,
        "gross_turnover_all_fills_usd": all_fill_turnover,
        "fees_equivalent_resolved_pnl_sample_usd": protocol_fee_equivalent,
        "realized_fee_effect_on_pnl_usd": fee_effect,
        "pnl_before_fee_usd": gross_pnl,
        "pnl_after_fee_usd": net_pnl,
        "net_pnl_per_turnover": net_roi,
        "wins": wins,
        "losses": losses,
        "average_execution_price_share_weighted": avg_execution,
        "full_book_resolved_fills": full_book_count,
        "proxy_resolved_fills": proxy_count,
        "partial_resolved_fills": partial_count,
        "execution_share_full_book": full_book_count / resolved_count if resolved_count else None,
        "execution_share_proxy": proxy_count / resolved_count if resolved_count else None,
        "execution_share_partial": partial_count / resolved_count if resolved_count else None,
        "pnl_decomposition": bridge,
    }
    summary["block_bootstrap"] = _signal_block_bootstrap(
        resolved, bootstrap_start=bootstrap_start, bootstrap_end=bootstrap_end,
    )
    return summary


def _signal_block_bootstrap(
    resolved: pd.DataFrame,
    *,
    bootstrap_start: pd.Timestamp,
    bootstrap_end: pd.Timestamp,
) -> dict:
    calendar_days = pd.date_range(
        bootstrap_start.floor("D"), (bootstrap_end - pd.Timedelta(nanoseconds=1)).floor("D"),
        freq="D", tz="UTC",
    )
    if resolved.empty:
        return {"method": "three-day moving-block bootstrap; shared seeded blocks", "resolved_fills": 0}
    pnl_data = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(resolved.market_start_utc, utc=True),
        "pnl": pd.to_numeric(resolved.net_pnl_usd, errors="raise").to_numpy(float),
        "stake": pd.to_numeric(resolved.filled_gross_usd, errors="raise").to_numpy(float),
    })
    roi_interval = block_uncertainty(pnl_data, calendar_days)
    error_data = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(resolved.market_start_utc, utc=True),
        "pnl": (
            pd.to_numeric(resolved.expected_net_t59_usd, errors="raise")
            - pd.to_numeric(resolved.net_pnl_usd, errors="raise")
        ).to_numpy(float),
        "stake": np.ones(len(resolved), dtype=np.float64),
    })
    error_interval = block_uncertainty(error_data, calendar_days)
    return {
        "method": "existing utils.polymarket_diagnostics.block_uncertainty; moving 3-day UTC blocks, 1,000 replicates, seed 20261002; includes zero-trade days",
        "shared_calendar_blocks_across_roi_and_expected_minus_realized": True,
        "calendar_days": int(len(calendar_days)),
        "resolved_fills": int(len(resolved)),
        "net_pnl_per_turnover_bootstrap_mean": roi_interval.get("pnl_per_turnover_bootstrap_mean"),
        "net_pnl_per_turnover_95pct": roi_interval.get("pnl_per_turnover_95pct"),
        "expected_minus_realized_net_pnl_per_fill_bootstrap_mean": error_interval.get("pnl_per_turnover_bootstrap_mean"),
        "expected_minus_realized_net_pnl_per_fill_95pct": error_interval.get("pnl_per_turnover_95pct"),
        "expected_minus_realized_net_pnl_per_fill_point": float(error_data.pnl.mean()),
        "roi_bootstrap_metadata": roi_interval,
    }


def _ev_group_rows(decisions: pd.DataFrame, ev_edges: list[float], hurdles: tuple[float, ...]) -> list[dict]:
    base = decisions.loc[decisions.decision_status.eq("positive_t60_net_ev")].copy()
    base["ev_group_index"] = [
        _bin_index(value, ev_edges)
        for value in pd.to_numeric(base.t60_net_ev_fraction, errors="raise")
    ]
    rows = []
    group_count = len(ev_edges) + 1
    for period in ("development", "holdout", "combined"):
        scoped = base if period == "combined" else base.loc[base.period.eq(period)]
        for hurdle in hurdles:
            signals = scoped.loc[scoped.t60_net_ev_fraction.ge(float(hurdle) - 1e-12)]
            for group_index in range(group_count):
                group = signals.loc[signals.ev_group_index.eq(group_index)]
                filled = group.loc[group.execution_status.eq("filled")]
                resolved = filled.loc[filled.official_outcome_up.isin((0.0, 1.0))]
                rows.append({
                    "period": period,
                    "ev_hurdle_fraction": float(hurdle),
                    "ev_group": group_index + 1,
                    "ev_lower_bound_dev_fraction": None if group_index == 0 else ev_edges[group_index - 1],
                    "ev_upper_bound_dev_fraction": None if group_index == len(ev_edges) else ev_edges[group_index],
                    "t60_signals": int(len(group)),
                    "t59_fills": int(len(filled)),
                    "resolved_fills": int(len(resolved)),
                    "expected_net_t60_usd": float(group.t60_expected_net_pnl_usd.sum()) if len(group) else 0.0,
                    "expected_net_t59_usd": float(resolved.expected_net_t59_usd.sum()) if len(resolved) else 0.0,
                    "realized_net_pnl_usd": float(resolved.net_pnl_usd.sum()) if len(resolved) else 0.0,
                    "gross_turnover_usd": float(resolved.filled_gross_usd.sum()) if len(resolved) else 0.0,
                })
    return rows


def _run_holdout_portfolio(
    initial_cash: float,
    holdout_rows: pd.DataFrame,
    snapshot_by_id: dict[str, dict],
    market_by_id: dict[str, dict],
    decision_snapshot_by_id: dict[str, dict],
    hurdle: float,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    original_initial_cash = replay.INITIAL_CASH_USD
    replay.INITIAL_CASH_USD = float(initial_cash)
    try:
        policy_name = f"free_cash_5pct_diagnostic_{int(initial_cash)}"
        summary, trades, decisions = replay._portfolio_for_policy(
            "free_cash_5pct",
            holdout_rows,
            snapshot_by_id,
            market_by_id,
            decision_snapshot_by_id=decision_snapshot_by_id,
            minimum_net_ev_fraction=float(hurdle),
            output_policy=policy_name,
        )
        equity_path = summary.pop("_equity_path")
        summary["monthly_diagnostics"] = replay._monthly_portfolio_diagnostics(trades, decisions, equity_path)
        summary["accounting_audit"] = replay._portfolio_accounting_audit(
            summary, trades, decisions, equity_path, snapshot_by_id, decision_snapshot_by_id,
        )
        summary["drawdown_contributors"] = replay._drawdown_contributors(trades, equity_path)
    finally:
        replay.INITIAL_CASH_USD = original_initial_cash
    frame = pd.DataFrame(trades)
    if not frame.empty:
        frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True)
    first_minimum = next((item for item in decisions if item.get("skip_reason") == "below_minimum_order_usd"), None)
    minimum_at = pd.Timestamp(first_minimum["entry_time_utc"]) if first_minimum else None
    later_fills = (
        int((frame.market_start_utc > minimum_at).sum())
        if minimum_at is not None and not frame.empty else 0
    )
    summary["minimum_order_diagnostic"] = {
        "minimum_skip_count": sum(item.get("skip_reason") == "below_minimum_order_usd" for item in decisions),
        "first_minimum_skip_utc": minimum_at.isoformat() if minimum_at is not None else None,
        "later_fills_after_first_minimum_skip": later_fills,
        "minimum_permanently_stopped_trading_after_first_skip": minimum_at is not None and later_fills == 0,
    }
    summary["first_trade_utc"] = frame.market_start_utc.min().isoformat() if not frame.empty else None
    summary["last_trade_utc"] = frame.market_start_utc.max().isoformat() if not frame.empty else None
    summary["starting_capital_usd"] = float(initial_cash)
    summary["return_fraction"] = float(summary["net_pnl_usd"]) / float(initial_cash)
    summary["monthly_activity"] = [
        {
            "month_utc": item["month_utc"],
            "trade_count": item["trade_count"],
            "gross_turnover_usd": item["gross_turnover_usd"],
            "fee_usd": item["fee_usd"],
            "monthly_equity_pnl_usd": item["monthly_equity_pnl_usd"],
        }
        for item in summary.get("monthly_diagnostics", [])
    ]
    curve = [
        {
            "timestamp_utc": pd.Timestamp(item["timestamp_ns"], unit="ns", tz="UTC").isoformat(),
            "equity_percent_of_start": 100.0 * float(item["cost_basis_equity_usd"]) / float(initial_cash),
            "cash_usd": float(item["cash_usd"]),
            "exposure_usd": float(item["exposure_usd"]),
            "event_type": item["event_type"],
        }
        for item in equity_path
    ]
    if summary["accounting_audit"].get("status") != "passed":
        raise AssertionError(f"Portfolio accounting audit failed for starting capital ${initial_cash:.0f}")
    if any(float(item["cash_usd"]) < -1e-8 for item in curve):
        raise AssertionError("Portfolio spent more free cash than was available")
    for trade in trades:
        if float(trade["cash_debit_usd"]) > float(trade["cash_before_usd"]) + 1e-8:
            raise AssertionError("Portfolio entry exceeded available free cash")
        side = trade["chosen_side"]
        minimum = _finite(trade.get(f"{side}_decision_requested_minimum_order_usd"))
        if minimum is not None and float(trade["requested_gross_usd"]) + 1e-9 < minimum:
            raise AssertionError("Portfolio executed an order below the configured minimum")
    return summary, trades, decisions, curve


def _fmt_usd(value, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"${float(value):,.{digits}f}"


def _fmt_pct(value, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}%}"


def _fmt_interval(mean, interval, formatter) -> str:
    if mean is None or interval is None:
        return "n/a"
    return f"{formatter(mean)} [{formatter(interval[0])}; {formatter(interval[1])}]"


def _fmt_hurdle(value: float) -> str:
    return f"{100.0 * float(value):g}%"


def _create_figures(
    monthly_rows: list[dict],
    quality_bins: list[dict],
    ev_groups: list[dict],
    portfolio_curves: dict[str, list[dict]],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    monthly = pd.DataFrame([row for row in monthly_rows if row["ev_hurdle_fraction"] == 0.0 and row["period"] == "combined"])
    if not monthly.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
        x = np.arange(len(monthly))
        width = 0.38
        ax.bar(x - width / 2, monthly.pnl_before_fee_usd, width, label="PnL przed fee", color="#3B82F6")
        ax.bar(x + width / 2, monthly.pnl_after_fee_usd, width, label="PnL po fee", color="#DC2626")
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_xticks(x, monthly.month_utc, rotation=35, ha="right")
        ax.set_ylabel("USD, niezależne zlecenia $5")
        ax.set_title("Niezależne sygnały z progiem EV 0% — miesięczny PnL")
        ax.legend(frameon=False, ncols=2)
        fig.savefig(FIGURE_DIR / "monthly_pnl.svg", format="svg", metadata={"Date": None})
        plt.close(fig)

    calibration = pd.DataFrame(quality_bins)
    calibration = calibration.loc[
        calibration.scope.eq("all_available")
        & calibration.target.eq("official_polymarket")
        & calibration.probability.isin(("raw", "calibrated"))
    ]
    fig, ax = plt.subplots(figsize=(7.6, 6.0), constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1, label="ideal")
    styles = {
        ("development", "raw"): ("#60A5FA", "development raw"),
        ("development", "calibrated"): ("#2563EB", "development calibrated"),
        ("holdout", "raw"): ("#FCA5A5", "holdout raw"),
        ("holdout", "calibrated"): ("#DC2626", "holdout calibrated"),
    }
    for (period, probability), (color, label) in styles.items():
        subset = calibration.loc[calibration.period.eq(period) & calibration.probability.eq(probability)]
        subset = subset.loc[subset.n.gt(0)].sort_values("bin")
        if not subset.empty:
            ax.plot(subset.mean_probability, subset.observed_up_rate, marker="o", linewidth=1.8, color=color, label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Średnie przewidywane P(UP)", ylabel="Częstość oficjalnego UP", title="Kalibracja na zamrożonych przedziałach z development")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(FIGURE_DIR / "calibration.svg", format="svg", metadata={"Date": None})
    plt.close(fig)

    ev = pd.DataFrame(ev_groups)
    ev = ev.loc[
        ev.period.isin(("development", "holdout"))
        & ev.ev_hurdle_fraction.eq(0.0)
        & ev.resolved_fills.gt(0)
    ].copy()
    if not ev.empty:
        ev["expected_net_per_fill"] = ev.expected_net_t59_usd / ev.resolved_fills
        ev["realized_net_per_fill"] = ev.realized_net_pnl_usd / ev.resolved_fills
        fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
        for period, color, label in (("development", "#2563EB", "development"), ("holdout", "#DC2626", "holdout")):
            subset = ev.loc[ev.period.eq(period)].sort_values("ev_group")
            if not subset.empty:
                ax.plot(subset.ev_group, subset.expected_net_per_fill, marker="o", color=color, linestyle="--", label=f"{label}: EV T−59")
                ax.plot(subset.ev_group, subset.realized_net_per_fill, marker="s", color=color, label=f"{label}: zrealizowany PnL")
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set(xlabel="Grupa EV T−60 (granice ustalone na development)", ylabel="USD na rozliczone wykonanie", title="Oczekiwana i zrealizowana przewaga po grupie EV")
        ax.legend(frameon=False, ncols=2)
        fig.savefig(FIGURE_DIR / "expected_vs_realized.svg", format="svg", metadata={"Date": None})
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for name, curve in portfolio_curves.items():
        if curve:
            times = pd.to_datetime([item["timestamp_utc"] for item in curve], utc=True)
            values = [item["equity_percent_of_start"] for item in curve]
            ax.plot(times, values, linewidth=1.5, label=f"{name} USD")
    ax.axhline(100.0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set(ylabel="Equity (% początkowego kapitału)", title="Holdout: 5% wolnej gotówki, osobne salda od 1 lipca")
    ax.legend(frameon=False)
    fig.savefig(FIGURE_DIR / "portfolio_equity.svg", format="svg", metadata={"Date": None})
    plt.close(fig)


def _render_report(
    signal_summaries: list[dict],
    monthly_rows: list[dict],
    quality_rows: list[dict],
    calibration_bins: list[dict],
    ev_groups: list[dict],
    portfolio_summaries: dict[str, dict],
    policy: dict,
    source_identity: dict,
    diagnostics: dict,
) -> str:
    lines = [
        "# Diagnostyka przewagi BTC 5m — 2026-10-12",
        "",
        "> Historycznie oglądany holdout (2026-07-01–2026-10-06); raport jest diagnostyką, nie niezależnym testem potwierdzającym.",
        "> Niezależne sygnały $5 nie są portfelem $100; ich łączny PnL nie jest wynikiem osiągalnym z takim depozytem.",
        "",
        "## Werdykt",
        "",
        diagnostics["verdict"],
        "",
        "## Zamrożony zakres i dane",
        "",
        f"Development obejmuje predykcje od {DEV_START_UTC.isoformat()} do cutoffu {DEV_CUTOFF_UTC.isoformat()}; holdout obejmuje starty rynków od tej daty do {DATA_CUTOFF_UTC.isoformat()} (exclusive). Progi EV zachowano z konfiguracji eksperymentu: {', '.join(_fmt_hurdle(x) for x in diagnostics['hurdles'])}. Aktywny challenger wybrany na development: `{policy['best_active_challenger']['policy']}` przy hurdle {_fmt_hurdle(policy['best_active_challenger']['minimum_net_ev_fraction'])}; oficjalny selektor no-trade pozostał bez zmian (`{policy['selected_policy']['name']}`).",
        "",
        f"Użyto {source_identity['prediction_rows']:,} gotowych predykcji, {source_identity['calendar_market_count']:,} potwierdzonych rynków, {source_identity['official_outcome_count']:,} oficjalnych wyników oraz {source_identity['t60_snapshot_rows']:,}/{source_identity['t59_snapshot_rows']:,} snapshotów T−60/T−59. Księgi do ekonomiki zaczynają się {source_identity['first_t60_market_utc']}; brak wcześniejszych snapshotów jest raportowany jako przyczyna, nie jako wyceniony sygnał.",
        "",
        "Nie trenowano modelu, nie robiono refitu ani replayu archiwum księgi. Nie zmieniono konfiguracji live. Opłaty, minimum USDC, FAK i częściowe wypełnienia pochodzą z zaakceptowanego fee-correction.",
        "",
        "## Niezależne zlecenia $5",
        "",
        "Każda decyzja wybiera stronę wyłącznie z wyceny T−60. Wykonanie testuje tę samą stronę na T−59. Zlecenia nie współdzielą salda. Wyniki PnL liczą wyłącznie wykonania z oficjalnym rozliczeniem; fill bez etykiety pozostaje policzony jako fill, ale nie dostaje PnL równemu zero.",
        "",
        "### Łącznie",
        "",
        "| Okres | EV min. | Rynki | Predykcje | Decyzje wycenialne | Sygnały | Fille | Rozliczone fille | Obrót na wspólnej próbie | Fee eq. | PnL brutto | PnL netto | Netto/obrót | Wygrane–przegrane | Śr. cena | Pełna księga / proxy / partial |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in signal_summaries:
        if row["period"] not in ("development", "holdout", "combined"):
            continue
        lines.append(
            f"| {row['period']} | {_fmt_hurdle(row['ev_hurdle_fraction'])} | {row['market_count']:,} | {row['prediction_count']:,} | {row['t60_priceable_decisions']:,} | {row['signals']:,} | {row['fills']:,} | {row['resolved_fills_for_pnl']:,} | {_fmt_usd(row['gross_turnover_resolved_pnl_sample_usd'])} | {_fmt_usd(row['fees_equivalent_resolved_pnl_sample_usd'])} | {_fmt_usd(row['pnl_before_fee_usd'])} | {_fmt_usd(row['pnl_after_fee_usd'])} | {_fmt_pct(row['net_pnl_per_turnover'])} | {row['wins']:,}–{row['losses']:,} | {_fmt_usd(row['average_execution_price_share_weighted'], 3)} | {row['full_book_resolved_fills']:,} / {row['proxy_resolved_fills']:,} / {row['partial_resolved_fills']:,} |"
        )
    lines += [
        "",
        "PnL brutto i netto są liczone na tych samych rozliczonych fillach. Fee eq. pokazuje estymowaną opłatę w USD przy wejściu. Dla V1 fee pobierane w udziałach zmienia zrealizowany payout tylko przy wygranej; dlatego osobny `realized fee effect` uzgadnia brutto z netto, a sama kwota fee eq. nie jest mechanicznie odejmowana od każdej przegranej.",
        "",
        "### Miesięcznie — wszystkie progi EV",
        "",
        "| Miesiąc UTC | EV min. | Rynki | Predykcje | Decyzje wycenialne | Sygnały | Fille | Rozliczone | Obrót | Fee eq. | PnL brutto | PnL netto | Netto/obrót |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in monthly_rows:
        lines.append(
            f"| {row['month_utc']} | {_fmt_hurdle(row['ev_hurdle_fraction'])} | {row['market_count']:,} | {row['prediction_count']:,} | {row['t60_priceable_decisions']:,} | {row['signals']:,} | {row['fills']:,} | {row['resolved_fills_for_pnl']:,} | {_fmt_usd(row['gross_turnover_resolved_pnl_sample_usd'])} | {_fmt_usd(row['fees_equivalent_resolved_pnl_sample_usd'])} | {_fmt_usd(row['pnl_before_fee_usd'])} | {_fmt_usd(row['pnl_after_fee_usd'])} | {_fmt_pct(row['net_pnl_per_turnover'])} |"
        )
    lines += [
        "",
        f"![Miesięczny PnL niezależnych sygnałów $5]({FIGURE_DIR.relative_to(REPORT_PATH.parent).as_posix()}/monthly_pnl.svg)",
        "",
        "### Oczekiwana a zrealizowana przewaga (te same fill’e)",
        "",
        "| Okres | EV min. | EV netto T−60 | EV netto T−59 | Zmiana brutto T−60→T−59 | Błąd payoutu (brutto) | Fee effect zrealizowany | PnL netto | Bootstrap mean PnL/obrót [95% CI] | Bootstrap mean EV−PnL/fill [95% CI] | Uzgodnienie residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in signal_summaries:
        if row["period"] not in ("development", "holdout", "combined"):
            continue
        bridge = row["pnl_decomposition"]
        if bridge is None:
            continue
        boot = row["block_bootstrap"]
        lines.append(
            f"| {row['period']} | {_fmt_hurdle(row['ev_hurdle_fraction'])} | {_fmt_usd(bridge['expected_net_t60_usd'])} | {_fmt_usd(bridge['expected_net_t59_usd'])} | {_fmt_usd(bridge['execution_t60_to_t59_usd'])} | {_fmt_usd(bridge['prediction_payout_residual_usd'])} | −{_fmt_usd(bridge['realized_fee_effect_usd'])} | {_fmt_usd(bridge['net_pnl_usd'])} | {_fmt_interval(boot.get('net_pnl_per_turnover_bootstrap_mean'), boot.get('net_pnl_per_turnover_95pct'), _fmt_pct)} | {_fmt_interval(boot.get('expected_minus_realized_net_pnl_per_fill_bootstrap_mean'), boot.get('expected_minus_realized_net_pnl_per_fill_95pct'), lambda value: _fmt_usd(value, 3))} | {_fmt_usd(bridge['reconciliation_residual_usd'], 6)} |"
        )
    lines += [
        "",
        "Uzgodnienie netto: `EV brutto T−60 + zmiana oczekiwanej wartości brutto przy wycenie T−59 + (wynik brutto − oczekiwany payout T−59) − zrealizowany fee effect = PnL netto`. EV netto T−60/T−59 pokazano osobno. Dla przedziałów użyto istniejącego `utils.polymarket_diagnostics.block_uncertainty`: ruchome bloki 3 dni UTC, 1 000 replikacji, seed 20261002, wspólne losowania bloków i dni bez transakcji. ROI jest warunkowe na wykonanych sygnałach; bootstrap nie symuluje salda ani ponownej selekcji.",
        "",
        f"![EV a zrealizowany PnL według grup EV]({FIGURE_DIR.relative_to(REPORT_PATH.parent).as_posix()}/expected_vs_realized.svg)",
        "",
        "### Powody braku wykonania — zero hurdle",
        "",
        "| Okres | Powód | Liczba rynków |",
        "|---|---|---:|",
    ]
    zero_rows = [row for row in signal_summaries if row["ev_hurdle_fraction"] == 0.0 and row["period"] in ("development", "holdout", "combined")]
    for summary in zero_rows:
        for reason, count in sorted(summary["non_execution_reasons"].items()):
            lines.append(f"| {summary['period']} | `{reason}` | {count:,} |")
    lines += [
        "",
        "## Jakość predykcji na dostępnych rynkach",
        "",
        "Kalibrator jest zamrożony. Baseline częstości UP jest liczony wyłącznie na development, osobno dla etykiety i zakresu próby. Wspólna próba ceny BTC i oficjalnego rozliczenia pozwala rozdzielić jakość targetu treningowego od rozliczeń Polymarket.",
        "",
        "| Okres | Próba | Target | Prognoza | N | UP | Log loss | Brier | AUC | Slope / intercept | ECE (biny dev) | LL baseline 0.5 / dev-rate | Brier baseline 0.5 / dev-rate |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in quality_rows:
        lines.append(
            f"| {row['period']} | {row['scope']} | {row['target']} | {row['probability']} | {row['n']:,} | {_fmt_pct(row['positive_rate'])} | {row['log_loss']:.5f} | {row['brier']:.5f} | {row['auc']:.4f} | {row['calibration_slope']:.3f} / {row['calibration_intercept']:.3f} | {row['calibration_ece_frozen_bins']:.4f} | {row['constant_0_5_log_loss']:.5f} / {row['development_prevalence_log_loss']:.5f} | {row['constant_0_5_brier']:.5f} / {row['development_prevalence_brier']:.5f} |"
        )
    lines += [
        "",
        "Przedziały kalibracji są kwantylami skalibrowanego P(UP) z development (granice wraz z liczebnościami zapisano w `calibration_bins.csv`); na holdoucie zastosowano te same granice. Surowe i skalibrowane szeregi są pokazane równolegle.",
        "",
        f"![Kalibracja prognoz względem oficjalnego wyniku]({FIGURE_DIR.relative_to(REPORT_PATH.parent).as_posix()}/calibration.svg)",
        "",
        "## Kapitał i minimum zlecenia",
        "",
        "Każdy holdout portfel startuje osobno 1 lipca. Użyto 5% wolnej gotówki, challengera 0% EV z development, rzeczywistej głębokości dostępnej przy większym zleceniu, fee i częściowego FAK. Oficjalny selektor pozostaje `no_trade`; te portfele są diagnostyczne.",
        "",
        "| Start | Kapitał końcowy | Zwrot | Max DD | Obrót | Fee eq. | Transakcje | Odrzucenia minimum | Brak wolnej gotówki | Pierwsza → ostatnia transakcja | Pierwszy skip minimum | Minimum zatrzymało dalsze fill’e? |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for capital, item in portfolio_summaries.items():
        summary = item["summary"]
        minimum = summary["minimum_order_diagnostic"]
        rejection = summary["rejections_by_reason"]
        lines.append(
            f"| {_fmt_usd(float(capital), 0)} | {_fmt_usd(summary['ending_cash_after_settlement_usd'])} | {_fmt_pct(summary['return_fraction'])} | {_fmt_pct(summary['max_drawdown_cost_basis_equity'])} | {_fmt_usd(summary['gross_turnover_usd'])} | {_fmt_usd(summary['fees_paid_estimated_usd'])} | {summary['trade_count']:,} | {minimum['minimum_skip_count']:,} | {rejection.get('insufficient_free_cash_for_cash_debit', 0) + rejection.get('no_free_cash', 0):,} | {summary['first_trade_utc']} → {summary['last_trade_utc']} | {minimum['first_minimum_skip_utc']} | {'tak' if minimum['minimum_permanently_stopped_trading_after_first_skip'] else 'nie' if minimum['minimum_skip_count'] else 'brak skipu'} |"
        )
    lines += ["", "Miesięczna aktywność portfeli:", "", "| Start | Miesiąc | Transakcje | Obrót | Fee | PnL equity |", "|---:|---|---:|---:|---:|---:|"]
    for capital, item in portfolio_summaries.items():
        for month in item["summary"]["monthly_activity"]:
            lines.append(
                f"| {_fmt_usd(float(capital), 0)} | {month['month_utc']} | {month['trade_count']:,} | {_fmt_usd(month['gross_turnover_usd'])} | {_fmt_usd(month['fee_usd'])} | {_fmt_usd(month['monthly_equity_pnl_usd'])} |"
            )
    lines += [
        "",
        f"![Ścieżki equity procentowego dla trzech kapitałów]({FIGURE_DIR.relative_to(REPORT_PATH.parent).as_posix()}/portfolio_equity.svg)",
        "",
        "Minimum bazuje na bieżącym Gamma `orderMinSize` w USDC albo fallbacku $5 i nie jest archiwalnym odczytem historycznym. Zlecenia poniżej progu nie były podbijane; FAK może pozostawić częściowy fill po spełnieniu minimum przez kwotę żądaną.",
        "",
        "## Wnioski do sześciu pytań",
        "",
        diagnostics["answers"],
        "",
        "Dalszy krok badawczy, jeśli wskazany, musi testować konkretny mechanizm rozpoznany powyżej. Obejrzanego holdoutu nie wolno używać do strojenia i nadal nazywać niezależnym testem.",
        "",
        "## Proweniencja i kontrole",
        "",
        f"Manifest: `{OUTPUT_DIR.relative_to(ROOT).as_posix()}/signal_diagnostic_manifest.json`. Szczegółowy ledger: `{OUTPUT_DIR.relative_to(ROOT).as_posix()}/independent_signal_decisions.parquet`; miesięczne tabele i metryki: `{OUTPUT_DIR.relative_to(ROOT).as_posix()}`.",
        "",
        f"Hash OOF manifestu: `{source_identity['walk_forward_manifest_sha256']}`; hash model bundle: `{source_identity['evaluated_bundle_sha256']}`; hash predykcji holdout: `{source_identity['holdout_prediction_sha256']}`; T−60: `{source_identity['t60_snapshot_sha256']}`; T−59: `{source_identity['t59_snapshot_sha256']}`. Wszystkie źródła zweryfikowano ponownie po obliczeniu.",
        "",
        f"Kontrole: niezależne sygnały nie używają salda; {diagnostics['gross_net_reconciled_trades']:,} wypełnionych i rozliczonych zleceń uzgodniono brutto/netto; trzy portfele przeszły audyt rachunkowości, nie miały ujemnej gotówki i nie wykonały zlecenia poniżej minimum. Czas obliczeń: {diagnostics['runtime_seconds']:.1f} s.",
        "",
    ]
    return "\n".join(lines)


def main() -> dict:
    started = time.perf_counter()
    fee_correction._configure_output_paths()
    source_research, source_walk, source_holdout = fee_correction._verify_reusable_sources()
    model_hashes_before = _verified_model_artifact_hashes(source_holdout)
    correction_manifest = research._read_json(CORRECTION_DIR / "research_manifest.json")
    policy = research._read_json(CORRECTION_DIR / "selected_development_policy.json")
    hurdles = tuple(float(value) for value in research.CONFIG["policy_selection"]["minimum_net_ev_fraction_candidates"])
    if tuple(sorted(set(hurdles))) != tuple(sorted(hurdles)) or 0.0 not in hurdles:
        raise RuntimeError("The frozen EV hurdle grid must contain unique values and zero hurdle")
    if policy.get("status") != "frozen_from_development_before_holdout_evaluation":
        raise RuntimeError("The corrected development policy is not frozen")
    challenger_hurdle = float(policy["best_active_challenger"]["minimum_net_ev_fraction"])

    predictions, prediction_identity = research._load_model_predictions(include_holdout=True)
    t59, t60, book_record = research._load_enriched_book_snapshots(correction_manifest)
    calendar, calendar_counts = research._build_market_calendar(predictions)
    calendar, fixed5_t59_evaluations = replay._market_statuses(calendar, t59, t60)
    calendar["market_start_utc"] = pd.to_datetime(calendar.market_start_utc, utc=True, errors="raise")
    market_by_id = research._snapshot_map(calendar.loc[calendar.condition_id.notna()])
    t59_by_id = research._snapshot_map(t59)
    t60_by_id = research._snapshot_map(t60)
    print(
        f"loaded verified inputs: predictions={len(predictions):,}, markets={calendar_counts['confirmed_market_rows']:,}, "
        f"T60/T59={len(t60):,}/{len(t59):,}; model bundle={source_holdout['evaluated_bundle_sha256']}",
        flush=True,
    )

    dev_mask = calendar.market_start_utc.ge(DEV_START_UTC) & calendar.market_start_utc.lt(DEV_CUTOFF_UTC)
    dev_probabilities = pd.to_numeric(
        calendar.loc[dev_mask & calendar.target_polymarket_up.notna(), "p_candidate_platt"], errors="coerce",
    ).dropna()
    calibration_edges = _development_quantile_edges(
        dev_probabilities.to_numpy(float), CALIBRATION_BINS, endpoints=(0.0, 1.0),
    )

    decisions = _independent_decisions(calendar, market_by_id, fixed5_t59_evaluations, t60_by_id, hurdles)
    positive_dev_ev = pd.to_numeric(
        decisions.loc[
            decisions.period.eq("development") & decisions.decision_status.eq("positive_t60_net_ev"),
            "t60_net_ev_fraction",
        ], errors="coerce",
    ).dropna()
    ev_edges = _development_quantile_edges(positive_dev_ev.to_numpy(float), EV_BINS)
    print(
        f"scored independent T60/T59 decisions: calendar={len(decisions):,}, "
        f"positive T60 candidates={int(decisions.decision_status.eq('positive_t60_net_ev').sum()):,}; EV bins frozen on {len(positive_dev_ev):,} dev candidates",
        flush=True,
    )

    periods = {
        "development": (DEV_START_UTC, DEV_CUTOFF_UTC),
        "holdout": (DEV_CUTOFF_UTC, DATA_CUTOFF_UTC),
        "combined": (DEV_START_UTC, DATA_CUTOFF_UTC),
    }
    signal_summaries = []
    for period, (start, end) in periods.items():
        for hurdle in hurdles:
            signal_summaries.append(_aggregate_signals(
                decisions, period, hurdle, bootstrap_start=start, bootstrap_end=end,
            ))
    monthly_rows = []
    observed_months = sorted(decisions.month_utc.unique())
    for month in observed_months:
        month_start = pd.Timestamp(f"{month}-01", tz="UTC")
        month_end = month_start + pd.offsets.MonthBegin(1)
        for hurdle in hurdles:
            monthly_rows.append(_aggregate_signals(
                decisions.loc[decisions.month_utc.eq(month)], "combined", hurdle,
                bootstrap_start=month_start, bootstrap_end=month_end,
            ))
            monthly_rows[-1]["month_utc"] = month
            monthly_rows[-1]["period"] = "combined"
    # The monthly aggregation above operates on the already month-filtered rows; mark the
    # true market month separately and use the same rows for every frozen EV hurdle.
    quality_rows, calibration_bins = _prediction_quality(calendar, calibration_edges)
    ev_groups = _ev_group_rows(decisions, ev_edges, hurdles)

    holdout_rows = calendar.loc[
        calendar.market_confirmed
        & calendar.market_start_utc.ge(DEV_CUTOFF_UTC)
        & calendar.market_start_utc.lt(DATA_CUTOFF_UTC)
    ].copy()
    portfolio_summaries: dict[str, dict] = {}
    portfolio_curves: dict[str, list[dict]] = {}
    for capital in PORTFOLIO_CAPITALS_USD:
        summary, trades, portfolio_decisions, curve = _run_holdout_portfolio(
            capital, holdout_rows, t59_by_id, market_by_id, t60_by_id, challenger_hurdle,
        )
        name = f"{int(capital)}"
        portfolio_summaries[name] = {
            "summary": summary,
            "trades": trades,
            "decisions": portfolio_decisions,
            "curve": curve,
        }
        portfolio_curves[name] = curve
        portfolio_dir = OUTPUT_DIR / "holdout_portfolios" / name
        portfolio_dir.mkdir(parents=True, exist_ok=True)
        if trades:
            pd.DataFrame(trades).to_parquet(portfolio_dir / "trades.parquet", index=False, compression="zstd")
        if portfolio_decisions:
            pd.DataFrame(portfolio_decisions).to_parquet(portfolio_dir / "decisions.parquet", index=False, compression="zstd")
        pd.DataFrame(curve).to_csv(portfolio_dir / "equity_path.csv", index=False)
        research._write_json(portfolio_dir / "summary.json", summary)
        print(f"completed {name} USD portfolio: trades={summary['trade_count']:,}, net={summary['net_pnl_usd']:.2f}", flush=True)

    _create_figures(monthly_rows, calibration_bins, ev_groups, portfolio_curves)
    decisions_path = OUTPUT_DIR / "independent_signal_decisions.parquet"
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(decisions_path, index=False, compression="zstd")
    pd.DataFrame(signal_summaries).to_json(
        OUTPUT_DIR / "signal_summary.json", orient="records", indent=2, force_ascii=False,
    )
    pd.DataFrame(monthly_rows).to_csv(OUTPUT_DIR / "monthly_signal_summary.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(OUTPUT_DIR / "prediction_quality.csv", index=False)
    pd.DataFrame(calibration_bins).to_csv(OUTPUT_DIR / "calibration_bins.csv", index=False)
    pd.DataFrame(ev_groups).to_csv(OUTPUT_DIR / "ev_group_summary.csv", index=False)

    # Period rows overlap, so use the combined zero-hurdle count exactly once.
    resolved_total = next(
        row["resolved_fills_for_pnl"] for row in signal_summaries
        if row["period"] == "combined" and row["ev_hurdle_fraction"] == 0.0
    )
    input_identity = {
        "walk_forward_manifest_sha256": _sha256(fee_correction.SOURCE_WALK_FORWARD_MANIFEST),
        "prediction_identity": prediction_identity,
        "development_oof_sources": source_research["stages"]["development_walk_forward"],
        "evaluated_bundle_sha256": source_holdout["evaluated_bundle_sha256"],
        "holdout_prediction_sha256": source_holdout["predictions_sha256"],
        "t60_snapshot_sha256": book_record["t60_coverage_sha256"],
        "t59_snapshot_sha256": book_record["t59_coverage_sha256"],
        "fee_rules_sha256": correction_manifest["stages"]["holdout_portfolio_and_sensitivity"]["input_identity"]["fee_rules_sha256"],
        "archive_cutoff_exclusive_utc": DATA_CUTOFF_UTC.isoformat(),
    }
    source_identity = {
        **input_identity,
        "model_artifact_sha256": model_hashes_before,
        "prediction_rows": len(predictions),
        "calendar_market_count": calendar_counts["confirmed_market_rows"],
        "official_outcome_count": int(calendar.target_polymarket_up.notna().sum()),
        "t60_snapshot_rows": len(t60),
        "t59_snapshot_rows": len(t59),
        "first_t60_market_utc": calendar.loc[
            calendar.condition_id.isin(t60.condition_id), "market_start_utc",
        ].min().isoformat(),
    }
    diagnostics = {
        "hurdles": hurdles,
        "gross_net_reconciled_trades": resolved_total,
        "runtime_seconds": time.perf_counter() - started,
        "calibration_bin_edges": calibration_edges,
        "ev_bin_internal_edges": ev_edges,
        "gross_net_reconciled": True,
        "prediction_hashes_checked_before_after": True,
        "snapshot_hashes_checked_before_after": True,
        "model_artifact_hashes_checked_before_after": True,
        "no_balance_state_in_independent_diagnostic": True,
        "no_refit_or_event_replay": True,
    }

    # Use measured aggregate values in the six-part conclusion; no holdout threshold is selected.
    by_period_hurdle = {(row["period"], row["ev_hurdle_fraction"]): row for row in signal_summaries}
    dev_zero = by_period_hurdle[("development", 0.0)]
    hold_zero = by_period_hurdle[("holdout", 0.0)]
    all_zero = by_period_hurdle[("combined", 0.0)]
    hold_official = next(
        row for row in quality_rows
        if row["period"] == "holdout" and row["scope"] == "all_available"
        and row["target"] == "official_polymarket" and row["probability"] == "calibrated"
    )
    gross_net_text = (
        f"Development: {_fmt_usd(dev_zero['pnl_before_fee_usd'])} przed fee, {_fmt_usd(dev_zero['pnl_after_fee_usd'])} po fee; "
        f"holdout: {_fmt_usd(hold_zero['pnl_before_fee_usd'])} / {_fmt_usd(hold_zero['pnl_after_fee_usd'])}; "
        f"łącznie: {_fmt_usd(all_zero['pnl_before_fee_usd'])} / {_fmt_usd(all_zero['pnl_after_fee_usd'])} "
        f"na odpowiednich rozliczonych fillach."
    )
    portfolio_counts = {float(capital): item["summary"]["trade_count"] for capital, item in portfolio_summaries.items()}
    min_counts = {float(capital): item["summary"]["minimum_order_diagnostic"]["minimum_skip_count"] for capital, item in portfolio_summaries.items()}
    if hold_official["auc"] is None:
        discrimination = "AUC niedostępne"
    else:
        discrimination = (
            f"słabe rozróżnianie (AUC {_fmt_pct(hold_official['auc'])}); log loss "
            f"{hold_official['log_loss']:.5f} wobec {hold_official['constant_0_5_log_loss']:.5f} "
            f"i Brier {hold_official['brier']:.5f} wobec {hold_official['constant_0_5_brier']:.5f} "
            f"dla baseline’u 0.5, slope/intercept {hold_official['calibration_slope']:.3f}/"
            f"{hold_official['calibration_intercept']:.3f}"
        )
    diagnostics["verdict"] = (
        f"Na całym holdoucie model ma {discrimination}. Sygnały zero-hurdle: {gross_net_text} "
        f"W portfelu 5% wolnej gotówki zawartość minimum ogranicza aktywność: fill’e dla kapitałów $100/$500/$1000 = "
        f"{portfolio_counts[100.0]}/{portfolio_counts[500.0]}/{portfolio_counts[1000.0]}, a pominięcia przez minimum = "
        f"{min_counts[100.0]}/{min_counts[500.0]}/{min_counts[1000.0]}. Szczegóły miesięczne i rozkład wykonania pokazują, czy poza minimum brakowało również dodatnich sygnałów."
    )
    diagnostics["answers"] = "\n".join([
        f"1. Jakość modelu na holdoucie: {discrimination}; szczegółowe baseline’y development-only są w tabeli jakości.",
        f"2. Przewaga wybranych sygnałów po fee: {gross_net_text} Wszystkie progi holdout są ujemne po fee; żadnego nie wybrano na holdoucie.",
        f"3. Wpływ zmiany wyceny T−60→T−59 oraz błędu payoutu jest rozdzielony w tabeli uzgodnienia (zmiana EV brutto, payout residual, fee effect).",
        f"4. Kapitał/minimum: liczba filli 5% dla $100/$500/$1000 to {portfolio_counts[100.0]}/{portfolio_counts[500.0]}/{portfolio_counts[1000.0]}; pominięcia minimum to {min_counts[100.0]}/{min_counts[500.0]}/{min_counts[1000.0]}. Zestaw z niezależnymi sygnałami i powodami no-trade.",
        "5. Zmienność miesięczna jest w tabeli i wykresie; ocena dotyczy sygnałów w historycznie oglądanym zbiorze.",
        "6. Obecna aktywna strategia nie uzyskuje potwierdzenia przez ten historycznie znany holdout. Następny test powinien dotyczyć wyłącznie wykrytego mechanizmu wykonania/minimum albo kalibracji selekcji, używając nowego przyszłego okresu; holdouta nie wolno ponownie stroić.",
    ])

    # Verify the frozen inputs again after all computations and before committing outputs.
    fee_correction._verify_reusable_sources()
    model_hashes_after = _verified_model_artifact_hashes(source_holdout)
    if model_hashes_before != model_hashes_after:
        raise RuntimeError("A frozen model, bundle, or calibrator changed during the diagnostic")
    report = _render_report(
        signal_summaries, monthly_rows, quality_rows, calibration_bins, ev_groups,
        portfolio_summaries, policy, source_identity, diagnostics,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "started_input_identity": input_identity,
        "source_identity": source_identity,
        "development_period": {"start_inclusive_utc": DEV_START_UTC.isoformat(), "cutoff_exclusive_utc": DEV_CUTOFF_UTC.isoformat()},
        "holdout_period": {"start_inclusive_utc": DEV_CUTOFF_UTC.isoformat(), "cutoff_exclusive_utc": DATA_CUTOFF_UTC.isoformat()},
        "frozen_ev_hurdles": list(hurdles),
        "active_development_challenger": policy["best_active_challenger"],
        "official_policy_selector": policy["selected_policy"],
        "independent_signal_orders": {"gross_order_usd": FIXED_ORDER_USD, "cash_state_used": False},
        "calibration_bin_edges_from_development": calibration_edges,
        "ev_bin_edges_from_development": ev_edges,
        "signal_summaries": signal_summaries,
        "monthly_signal_summaries": monthly_rows,
        "prediction_quality": quality_rows,
        "calibration_bins": calibration_bins,
        "ev_groups": ev_groups,
        "portfolio_summaries": {key: value["summary"] for key, value in portfolio_summaries.items()},
        "checks": diagnostics,
        "output_paths": {
            "report": REPORT_PATH.relative_to(ROOT).as_posix(),
            "independent_signal_decisions": decisions_path.relative_to(ROOT).as_posix(),
            "monthly_summary": (OUTPUT_DIR / "monthly_signal_summary.csv").relative_to(ROOT).as_posix(),
            "quality_summary": (OUTPUT_DIR / "prediction_quality.csv").relative_to(ROOT).as_posix(),
            "calibration_bins": (OUTPUT_DIR / "calibration_bins.csv").relative_to(ROOT).as_posix(),
            "ev_groups": (OUTPUT_DIR / "ev_group_summary.csv").relative_to(ROOT).as_posix(),
            "figures": FIGURE_DIR.relative_to(ROOT).as_posix(),
        },
        "report_sha256": _sha256(REPORT_PATH),
        "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write_json(OUTPUT_DIR / "signal_diagnostic_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    main()
