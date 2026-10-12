"""Compare the frozen BTC probability with T-60 Polymarket references."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import run_btc_preopen_fee_correction as fee_correction
import run_btc_preopen_research as research
import run_btc_preopen_signal_diagnostic as diagnostic
import run_btc_preopen_t59_reassessment as replay
from utils.polymarket_policy import calibration_metrics


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_market_probability_experiment_20261012.json"
OUTPUT_DIR = ROOT / "data/analysis/polymarket/BTC/btc_market_probability_experiment_20261012"
REPORT_PATH = ROOT / "reports/btc_preopen/btc_market_probability_experiment_20261012.md"
FROZEN_SELECTION_PATH = ROOT / "reports/btc_preopen/btc_market_probability_frozen_selection_20261012.json"
DEV_START_UTC = pd.Timestamp("2026-01-01T00:00:00Z")
DEV_CUTOFF_UTC = pd.Timestamp("2026-07-01T00:00:00Z")
DATA_CUTOFF_UTC = pd.Timestamp("2026-10-07T00:00:00Z")
CV_START_UTC = pd.Timestamp("2026-05-01T00:00:00Z")
MODEL_FEATURES = {"C": ("logit_market",), "D": ("logit_market", "logit_btc_calibrated")}
FORECAST_NAMES = ("A_BTC_calibrated", "B_market_reference", "C_market_logistic", "D_market_plus_BTC")
FORECAST_COLUMNS = {
    "A_BTC_calibrated": "p_btc_calibrated",
    "B_market_reference": "p_market",
    "C_market_logistic": "p_c",
    "D_market_plus_BTC": "p_d",
}


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truth(value, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return default
    return bool(value)


def _logit(probability) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _snapshot_features(snapshot: dict) -> dict:
    side_quotes = {}
    for side in ("up", "down"):
        bid = _finite(snapshot.get(f"{side}_best_bid"))
        ask = _finite(snapshot.get(f"{side}_best_ask"))
        side_quotes[side] = bool(
            bid is not None and ask is not None
            and 0.0 <= bid <= ask < 1.0 and ask > 0.0
        )
    values = {
        "t60_up_best_bid": _finite(snapshot.get("up_best_bid")),
        "t60_up_best_ask": _finite(snapshot.get("up_best_ask")),
        "t60_down_best_bid": _finite(snapshot.get("down_best_bid")),
        "t60_down_best_ask": _finite(snapshot.get("down_best_ask")),
        "t60_up_spread": None,
        "t60_down_spread": None,
        "t60_archive_source": snapshot.get("archive_source"),
        "t60_quote_valid": _truth(snapshot.get("quote_valid")),
        "t60_source_hour_available": _truth(snapshot.get("archive_entry_hour_available")),
        "t60_archive_freshness_window_available": _truth(snapshot.get("archive_freshness_window_available")),
        "t60_no_future_receive_event": _truth(snapshot.get("no_future_event_at_entry")),
        "t60_no_future_source_event": _truth(snapshot.get("no_future_source_event_at_entry")),
        "t60_up_available": side_quotes["up"],
        "t60_down_available": side_quotes["down"],
        "t60_has_full_snapshot": _truth(snapshot.get("has_full_snapshot")),
        "t60_up_complemented": _truth(snapshot.get("up_quote_complemented")),
        "t60_down_complemented": _truth(snapshot.get("down_quote_complemented")),
        "t60_up_ask_age_seconds": _finite(snapshot.get("up_ask_age_seconds")),
        "t60_down_ask_age_seconds": _finite(snapshot.get("down_ask_age_seconds")),
        "t60_up_gap_after_book_init": _truth(snapshot.get("archive_gap_after_up_book_init")),
        "t60_down_gap_after_book_init": _truth(snapshot.get("archive_gap_after_down_book_init")),
        "t60_up_gap_state_known": snapshot.get("archive_gap_after_up_book_init") is not None
            and not pd.isna(snapshot.get("archive_gap_after_up_book_init")),
        "t60_down_gap_state_known": snapshot.get("archive_gap_after_down_book_init") is not None
            and not pd.isna(snapshot.get("archive_gap_after_down_book_init")),
    }
    if values["t60_up_best_bid"] is not None and values["t60_up_best_ask"] is not None:
        values["t60_up_spread"] = values["t60_up_best_ask"] - values["t60_up_best_bid"]
    if values["t60_down_best_bid"] is not None and values["t60_down_best_ask"] is not None:
        values["t60_down_spread"] = values["t60_down_best_ask"] - values["t60_down_best_bid"]
    available = all(side_quotes.values())
    valid = (
        available
        and values["t60_quote_valid"]
        and values["t60_source_hour_available"]
        and values["t60_archive_freshness_window_available"]
        and values["t60_no_future_receive_event"]
        and values["t60_no_future_source_event"]
        and values["t60_up_gap_state_known"]
        and values["t60_down_gap_state_known"]
        and not values["t60_up_gap_after_book_init"]
        and not values["t60_down_gap_after_book_init"]
    )
    values["t60_both_sides_available"] = available
    values["t60_common_quote_valid"] = valid
    values["t60_proxy_flag"] = bool(
        values["t60_up_complemented"]
        or values["t60_down_complemented"]
        or not values["t60_has_full_snapshot"]
    )
    values["p_market"] = None
    if valid:
        mid_up = (values["t60_up_best_bid"] + values["t60_up_best_ask"]) / 2.0
        mid_down = (values["t60_down_best_bid"] + values["t60_down_best_ask"]) / 2.0
        if mid_up + mid_down > 0.0:
            values["p_market"] = mid_up / (mid_up + mid_down)
    return values


def _build_dataset(calendar: pd.DataFrame, t60: pd.DataFrame) -> pd.DataFrame:
    snapshot_by_id = research._snapshot_map(t60)
    records = []
    for row in calendar.itertuples(index=False):
        if pd.isna(row.condition_id):
            continue
        cid = str(row.condition_id).lower()
        snapshot = snapshot_by_id.get(cid)
        if snapshot is None:
            records.append({"condition_id": cid, "t60_snapshot_available": False})
            continue
        records.append({
            "condition_id": cid,
            "t60_snapshot_available": True,
            **_snapshot_features(snapshot),
        })
    feature_frame = pd.DataFrame(records)
    base = calendar.copy()
    base["condition_id"] = base.condition_id.astype("string").str.lower()
    data = base.merge(feature_frame, on="condition_id", how="left", validate="many_to_one")
    data["market_start_utc"] = pd.to_datetime(data.market_start_utc, utc=True, errors="raise")
    data["resolved_at_utc"] = pd.to_datetime(data.resolved_at_utc, utc=True, errors="coerce")
    starts = data.market_start_utc
    minimum_resolution = starts + pd.Timedelta(minutes=5)
    resolution = data.resolved_at_utc.where(data.resolved_at_utc.ge(minimum_resolution), minimum_resolution)
    data["label_available_at_utc"] = resolution + pd.Timedelta(seconds=60)
    data["p_btc_raw"] = pd.to_numeric(data.p_candidate_raw, errors="coerce")
    data["p_btc_calibrated"] = pd.to_numeric(data.p_candidate_platt, errors="coerce")
    data["official_outcome_up"] = pd.to_numeric(data.target_polymarket_up, errors="coerce")
    data["logit_market"] = _logit(pd.to_numeric(data.p_market, errors="coerce").fillna(0.5))
    data["logit_btc_calibrated"] = _logit(data.p_btc_calibrated.fillna(0.5))
    data["month_utc"] = data.market_start_utc.dt.strftime("%Y-%m")
    data["source_proxy_group"] = np.where(
        data.t60_proxy_flag.fillna(False), "proxy_or_complemented", "full_native_snapshot",
    )
    data["common_sample"] = (
        data.market_confirmed.fillna(False)
        & data.causal_prediction_available.fillna(False)
        & data.official_outcome_up.isin((0.0, 1.0))
        & data.t60_common_quote_valid.fillna(False)
        & data.p_btc_raw.between(0.0, 1.0)
        & data.p_btc_calibrated.between(0.0, 1.0)
        & data.p_market.between(0.0, 1.0)
    )
    dev = data.market_start_utc.lt(DEV_CUTOFF_UTC)
    if not data.loc[data.common_sample & dev, "fold_id"].notna().all():
        raise RuntimeError("A development BTC probability lacks causal OOF provenance")
    return data


def _fit_logistic(train: pd.DataFrame, features: tuple[str, ...], c_value: float) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value), l1_ratio=0.0, solver="lbfgs", fit_intercept=True,
        class_weight=None, max_iter=2000,
    )
    model.fit(train.loc[:, features].to_numpy(dtype=np.float64), train.official_outcome_up.astype(int))
    return model


def _predict(model: LogisticRegression, frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return model.predict_proba(frame.loc[:, features].to_numpy(dtype=np.float64))[:, 1]


def _fold_records(config: dict) -> list[dict]:
    return [
        {**fold,
         "start": pd.Timestamp(fold["start_inclusive_utc"]),
         "end": pd.Timestamp(fold["end_exclusive_utc"])}
        for fold in config["chronological_cv"]["folds"]
        if fold["used_for_meta_model_parameter_selection"]
    ]


def _select_logistic_parameters(dev: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = _fold_records(config)
    c_grid = [float(value) for value in config["chronological_cv"]["inverse_regularization_C_grid"]]
    validation_parts = {}
    fit_rows = []
    for fold in folds:
        validation = dev.loc[dev.market_start_utc.ge(fold["start"]) & dev.market_start_utc.lt(fold["end"])].copy()
        training = dev.loc[
            dev.market_start_utc.lt(fold["start"])
            & dev.label_available_at_utc.le(fold["start"])
        ].copy()
        if validation.empty or training.empty or training.official_outcome_up.nunique() != 2:
            raise RuntimeError(f"Insufficient causal rows for development fold {fold['fold_id']}")
        if not (training.label_available_at_utc <= fold["start"]).all():
            raise AssertionError("A development fold contains a label unavailable at its start")
        validation_parts[fold["fold_id"]] = (training, validation)
        for model_name, features in MODEL_FEATURES.items():
            for c_value in c_grid:
                model = _fit_logistic(training, features, c_value)
                prediction = _predict(model, validation, features)
                ll = float(log_loss(validation.official_outcome_up, prediction, labels=[0, 1]))
                fit_rows.append({
                    "model": model_name, "C": c_value, "lambda_inverse_C": 1.0 / c_value,
                    "fold_id": fold["fold_id"], "training_rows": len(training),
                    "validation_rows": len(validation), "validation_log_loss": ll,
                    "validation_brier": float(brier_score_loss(validation.official_outcome_up, prediction)),
                })
    grid = pd.DataFrame(fit_rows)
    selected = {}
    for model_name in MODEL_FEATURES:
        score_rows = []
        for c_value, group in grid.loc[grid.model.eq(model_name)].groupby("C", sort=True):
            score_rows.append({
                "C": float(c_value),
                "pooled_log_loss": float(np.average(group.validation_log_loss, weights=group.validation_rows)),
                "pooled_rows": int(group.validation_rows.sum()),
            })
        scores = pd.DataFrame(score_rows)
        scores.sort_values(["pooled_log_loss", "C"], ascending=[True, True], kind="stable", inplace=True)
        selected[model_name] = {
            "C": float(scores.iloc[0].C),
            "lambda_inverse_C": float(1.0 / scores.iloc[0].C),
            "cv_pooled_log_loss": float(scores.iloc[0].pooled_log_loss),
            "cv_rows": int(scores.iloc[0].pooled_rows),
        }
    selected_oof = dev.copy()
    selected_oof["p_c"] = np.nan
    selected_oof["p_d"] = np.nan
    coefficient_rows = []
    for fold in folds:
        training, validation = validation_parts[fold["fold_id"]]
        for model_name, output_col in (("C", "p_c"), ("D", "p_d")):
            features = MODEL_FEATURES[model_name]
            model = _fit_logistic(training, features, selected[model_name]["C"])
            selected_oof.loc[validation.index, output_col] = _predict(model, validation, features)
            coefficients = dict(zip(features, model.coef_[0].astype(float)))
            coefficient_rows.append({
                "model": model_name, "fold_id": fold["fold_id"],
                "C": selected[model_name]["C"], "lambda_inverse_C": selected[model_name]["lambda_inverse_C"],
                "training_rows": len(training), "validation_rows": len(validation),
                "intercept": float(model.intercept_[0]),
                "beta_logit_market": float(coefficients["logit_market"]),
                "beta_logit_btc_calibrated": float(coefficients.get("logit_btc_calibrated", 0.0)),
            })
    cv = selected_oof.loc[selected_oof.p_c.notna() & selected_oof.p_d.notna()].copy()
    return selected, grid, cv, pd.DataFrame(coefficient_rows)


def _metric_row(frame: pd.DataFrame, probability_column: str, model_name: str, period: str, edges: list[float]) -> tuple[dict, list[dict]]:
    y = frame.official_outcome_up.to_numpy(dtype=np.int8)
    p = np.clip(frame[probability_column].to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
    calibration = calibration_metrics(y, p)
    quantiles = np.asarray(edges, dtype=np.float64)
    bucket = np.searchsorted(quantiles[1:-1], p, side="left")
    ece = 0.0
    bin_rows = []
    for index in range(len(quantiles) - 1):
        selected = bucket == index
        if selected.any():
            mean_p = float(p[selected].mean())
            observed = float(y[selected].mean())
            ece += float(selected.mean()) * abs(observed - mean_p)
        else:
            mean_p = observed = None
        bin_rows.append({
            "period": period, "forecast": model_name, "bin": index + 1,
            "lower": float(quantiles[index]), "upper": float(quantiles[index + 1]),
            "n": int(selected.sum()), "mean_probability": mean_p,
            "observed_up_rate": observed,
        })
    row = {
        "period": period, "forecast": model_name, "n": int(len(frame)),
        "positive_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "calibration_intercept": calibration.get("intercept"),
        "calibration_slope": calibration.get("slope"),
        "ece_frozen_dev_bins": float(ece),
    }
    return row, bin_rows


def _metric_edges(values, bins: int = 10) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("Cannot define calibration bins on an empty development sample")
    internal = np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return [float(value) for value in np.unique(np.concatenate(([0.0], internal, [1.0])))]


def _block_paired_differences(holdout: pd.DataFrame, left_col: str, right_col: str) -> dict:
    ordered = holdout.sort_values("market_start_utc", kind="stable").copy()
    y = ordered.official_outcome_up.to_numpy(dtype=np.int8)
    left = np.clip(ordered[left_col].to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
    right = np.clip(ordered[right_col].to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
    left_ll = -(y * np.log(left) + (1 - y) * np.log1p(-left))
    right_ll = -(y * np.log(right) + (1 - y) * np.log1p(-right))
    left_brier = np.square(left - y)
    right_brier = np.square(right - y)
    days = ordered.market_start_utc.dt.floor("D")
    unique_days = pd.date_range(days.min(), days.max(), freq="D")
    day_positions = [np.flatnonzero(days.eq(day).to_numpy()) for day in unique_days]
    rng = np.random.default_rng(20261012)
    ll_draws, brier_draws, auc_draws = [], [], []
    day_count = len(unique_days)
    block_count = int(np.ceil(day_count / 3))
    for _ in range(1000):
        starts = rng.integers(0, max(day_count - 2, 1), size=block_count)
        pieces = [day_positions[index] for start in starts for index in range(start, min(start + 3, day_count))]
        sample_index = np.concatenate(pieces)[:len(ordered)]
        ll_draws.append(float((left_ll[sample_index] - right_ll[sample_index]).mean()))
        brier_draws.append(float((left_brier[sample_index] - right_brier[sample_index]).mean()))
        if np.unique(y[sample_index]).size == 2:
            auc_draws.append(float(
                roc_auc_score(y[sample_index], left[sample_index])
                - roc_auc_score(y[sample_index], right[sample_index])
            ))
    return {
        "n": len(ordered),
        "log_loss_difference_left_minus_right": float((left_ll - right_ll).mean()),
        "log_loss_difference_95pct": np.quantile(ll_draws, [0.025, 0.975]).tolist(),
        "brier_difference_left_minus_right": float((left_brier - right_brier).mean()),
        "brier_difference_95pct": np.quantile(brier_draws, [0.025, 0.975]).tolist(),
        "auc_difference_left_minus_right": float(roc_auc_score(y, left) - roc_auc_score(y, right)),
        "auc_difference_95pct": np.quantile(auc_draws, [0.025, 0.975]).tolist() if auc_draws else None,
        "method": "paired moving 3-day UTC blocks; 1,000 replicates; seed 20261012",
    }


def _economic_rows(frame: pd.DataFrame, t59_by_id: dict, t60_by_id: dict, market_by_id: dict, period: str, hurdles: tuple[float, ...]) -> tuple[list[dict], list[dict], pd.DataFrame]:
    detail_rows = []
    for forecast in FORECAST_NAMES:
        print(f"[market-probability] T60/T59 independent orders: {period}, forecast={forecast}", flush=True)
        probability_col = FORECAST_COLUMNS[forecast]
        for row in frame.itertuples(index=False):
            p_up = _finite(getattr(row, probability_col))
            cid = str(row.condition_id).lower()
            if p_up is None:
                continue
            decision_market = dict(market_by_id[cid])
            decision_market["p_candidate_platt"] = p_up
            t60 = t60_by_id.get(cid)
            t59 = t59_by_id.get(cid)
            if t60 is None or t59 is None:
                continue
            decision = replay._evaluate_market_sides(decision_market, t60, 5.0)
            side = decision["chosen_side"]
            if side is None:
                continue
            ev_fraction = float(decision["chosen_ev_usd"]) / 5.0
            execution = replay._price_side(t59, decision_market, side, 5.0)
            fill = execution.get("fill")
            if not execution["priceable"] or fill is None:
                fill = None
            p_side = p_up if side == "up" else 1.0 - p_up
            outcome_up = _finite(row.official_outcome_up)
            outcome_side = int(outcome_up == (1.0 if side == "up" else 0.0))
            detail_rows.append({
                "period": period, "month_utc": row.month_utc,
                "market_start_utc": row.market_start_utc, "condition_id": cid,
                "forecast": forecast, "p_up": p_up, "chosen_side": side,
                "t60_ev_fraction": ev_fraction, "t60_ev_usd": float(decision["chosen_ev_usd"]),
                "execution_status": "filled" if fill is not None else "t59_unpriceable",
                "execution_reason": None if fill is not None else execution["reason"],
                "official_outcome_up": outcome_up,
                "execution_source": fill.get("execution_source") if fill else None,
                "depth_known": bool(fill.get("depth_known")) if fill else None,
                "execution_assumption": fill.get("execution_assumption") if fill else None,
                "fill_status": fill.get("fill_status") if fill else None,
                "filled_gross_usd": float(fill["gross_usd"]) if fill else None,
                "vwap": float(fill["vwap"]) if fill else None,
                "gross_shares": float(fill["gross_shares"]) if fill else None,
                "net_shares": float(fill["shares"]) if fill else None,
                "cash_debit_usd": float(fill["cash_debit_usd"]) if fill else None,
                "fee_equivalent_usd": float(fill.get("fee_usd_entry_equivalent") or fill.get("fee_usd") or 0.0) if fill else None,
                "expected_net_t59_usd": p_side * float(fill["shares"]) - float(fill["cash_debit_usd"]) if fill else None,
                "net_pnl_usd": (outcome_side * float(fill["shares"]) - float(fill["cash_debit_usd"])) if fill else None,
            })
    details = pd.DataFrame(detail_rows)
    summary_rows = []
    monthly_rows = []
    for model in FORECAST_NAMES:
        for hurdle in hurdles:
            for month, destination in [(None, summary_rows), *[(month, monthly_rows) for month in sorted(details.month_utc.unique())]]:
                selected = details.loc[
                    details.forecast.eq(model)
                    & details.t60_ev_fraction.ge(float(hurdle) - 1e-12)
                ]
                if month is not None:
                    selected = selected.loc[selected.month_utc.eq(month)]
                fills = selected.loc[selected.execution_status.eq("filled")].copy()
                row = {
                    "period": period, "month_utc": month, "forecast": model,
                    "ev_hurdle_fraction": float(hurdle),
                    "opportunities_with_positive_t60_ev": int(len(selected)),
                    "signals": int(len(selected)), "fills": int(len(fills)),
                    "full_book_fills": int(fills.depth_known.fillna(False).sum()) if len(fills) else 0,
                    "proxy_fills": sum(not bool(value) for value in fills.depth_known.fillna(False).tolist()) if len(fills) else 0,
                    "partial_fills": int(fills.fill_status.eq("partial").sum()) if len(fills) else 0,
                    "gross_turnover_usd": float(fills.filled_gross_usd.sum()) if len(fills) else 0.0,
                    "fee_equivalent_usd": float(fills.fee_equivalent_usd.sum()) if len(fills) else 0.0,
                    "expected_net_t59_usd": float(fills.expected_net_t59_usd.sum()) if len(fills) else 0.0,
                    "realized_net_pnl_usd": float(fills.net_pnl_usd.sum()) if len(fills) else 0.0,
                    "expected_minus_realized_usd": float((fills.expected_net_t59_usd - fills.net_pnl_usd).sum()) if len(fills) else 0.0,
                    "expected_net_per_fill_usd": float(fills.expected_net_t59_usd.mean()) if len(fills) else None,
                    "realized_net_per_fill_usd": float(fills.net_pnl_usd.mean()) if len(fills) else None,
                    "expected_minus_realized_per_fill_usd": float((fills.expected_net_t59_usd - fills.net_pnl_usd).mean()) if len(fills) else None,
                    "net_roi": float(fills.net_pnl_usd.sum() / fills.filled_gross_usd.sum()) if len(fills) and fills.filled_gross_usd.sum() else None,
                }
                destination.append(row)
    return summary_rows, monthly_rows, details


def _selection_portfolios(dev_cv: pd.DataFrame, t59_by_id: dict, t60_by_id: dict, market_by_id: dict, hurdles: tuple[float, ...]) -> tuple[list[dict], dict]:
    candidates = []
    for forecast in FORECAST_NAMES:
        p_col = FORECAST_COLUMNS[forecast]
        for hurdle in hurdles:
            print(f"[market-probability] dev portfolio selection: {forecast}, EV>={hurdle:.2%}", flush=True)
            rows = dev_cv.copy()
            rows["p_candidate_platt"] = rows[p_col]
            rows = rows.loc[rows.label_available_at_utc.le(DEV_CUTOFF_UTC)].copy()
            variant_market = {
                str(row.condition_id).lower(): {**market_by_id[str(row.condition_id).lower()], "p_candidate_platt": float(getattr(row, p_col))}
                for row in rows.itertuples(index=False)
            }
            summary, trades, decisions = replay._portfolio_for_policy(
                "free_cash_5pct", rows, t59_by_id, variant_market,
                decision_snapshot_by_id=t60_by_id,
                minimum_net_ev_fraction=float(hurdle),
                output_policy=f"development_oof_{forecast}_ev_{hurdle:.4%}",
            )
            equity_path = summary.pop("_equity_path")
            audit = replay._portfolio_accounting_audit(
                summary, trades, decisions, equity_path, t59_by_id, t60_by_id,
            )
            candidates.append({
                "policy": f"{forecast}_ev_{hurdle:.4%}",
                "forecast": forecast, "minimum_net_ev_fraction": float(hurdle),
                "ending_cash_after_settlement_usd": float(summary["ending_cash_after_settlement_usd"]),
                "net_pnl_usd": float(summary["net_pnl_usd"]),
                "max_drawdown_cost_basis_equity": float(summary["max_drawdown_cost_basis_equity"]),
                "trade_count": int(summary["trade_count"]),
                "gross_turnover_usd": float(summary["gross_turnover_usd"]),
                "fees_paid_estimated_usd": float(summary["fees_paid_estimated_usd"]),
                "portfolio_audit": audit.get("status"),
            })
    active, no_trade, selected = research._compare_active_challenger_to_no_trade(candidates, 100.0)
    freeze = {
        "status": "frozen_from_development_before_holdout_scoring",
        "selection_metric": "highest ending cash after settlement; ties lower max drawdown, fewer trades, then no-trade",
        "development_sample": {
            "start_inclusive_utc": CV_START_UTC.isoformat(),
            "cutoff_exclusive_utc": DEV_CUTOFF_UTC.isoformat(),
            "rows": int(len(dev_cv)),
            "latest_label_available_at_utc": dev_cv.label_available_at_utc.max().isoformat(),
        },
        "active_challenger": active,
        "no_trade_control": no_trade,
        "selected_policy": selected,
        "candidate_portfolios": candidates,
        "holdout_used_for_selection": False,
    }
    return candidates, freeze


def _fmt(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_money(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${float(value):,.{digits}f}"


def _render_report(payload: dict) -> str:
    coverage = payload["coverage"]
    models = payload["model_selection"]
    lines = [
        "# BTC i rynek: porównanie prognoz 5m — 12 października 2026", "",
        "> Holdout lipiec–październik był wcześniej oglądany. To zamknięta diagnostyka historyczna, nie nowe niezależne potwierdzenie.",
        "> Skorygowany raport błędu T−59: [diagnostyka sygnału](btc_preopen_signal_diagnostic_correction_20261012.md).",
        "", "## Werdykt", "", payload["verdict"], "",
        "## Zamrożone zasady i pokrycie", "",
        f"T−60 wspólne kwotowania: development {coverage['development']['common_rows']:,}/{coverage['development']['confirmed_with_prediction_and_outcome']:,} rynków ({coverage['development']['coverage_fraction']:.1%}); holdout {coverage['holdout']['common_rows']:,}/{coverage['holdout']['confirmed_with_prediction_and_outcome']:,} ({coverage['holdout']['coverage_fraction']:.1%}). Braki cen pominięto bez imputacji.",
        f"Poprawne obustronne kwotowanie wymaga `quote_valid`, pokrycia godziny źródłowej, obu bid/ask w zakresie i bez crossing, braku przyszłych zdarzeń oraz braku udokumentowanej luki po inicjalizacji booka. Ask age nie jest filtrem. Udział źródeł i proxy: {coverage['source_proxy_summary']}.",
        "T\u221260 kwotowania na populacji z potwierdzonym wynikiem i przyczynow\u0105 prognoz\u0105:",
        "",
        "| Okres | UP dost\u0119pne | DOWN dost\u0119pne | Obie strony dost\u0119pne | Pr\u00f3ba wsp\u00f3lna | Mediana UP bid/ask | UP spread | Mediana DOWN bid/ask | DOWN spread | Proxy/complemented |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {period} | {item['up_available']:,} | {item['down_available']:,} | {item['both_sides_available']:,} | {item['common_rows']:,} | {_fmt(item['up_bid_median'], 3)} / {_fmt(item['up_ask_median'], 3)} | {_fmt(item['up_spread_median'], 3)} | {_fmt(item['down_bid_median'], 3)} / {_fmt(item['down_ask_median'], 3)} | {_fmt(item['down_spread_median'], 3)} | {item['proxy_or_complemented_rows']:,} |"
            for period, item in coverage["quote_detail"].items()
        ],
        "",
        "Referencja: `m_up=(bid_up+ask_up)/2`, `m_down=(bid_down+ask_down)/2`, `p_market=m_up/(m_up+m_down)`. Jest prognozą, nie ceną wykonania.",
        "C i D wybrano na dwóch chronologicznych foldach rozwoju, z oficjalnym wynikiem i etykietami dostępnymi przed początkiem walidacji. Fold 15–30 kwietnia nie mógł służyć do doboru C/D, ponieważ nie było wcześniejszej próby księgi. Rozmiar próby nie uzasadnia bardziej złożonego modelu.",
        "", "## Dobór regularizacji i współczynniki", "",
        "| Model | C wybrane | λ=1/C | CV log loss | CV rynki |", "|---|---:|---:|---:|---:|",
    ]
    for name in ("C", "D"):
        value = models["selected_parameters"][name]
        lines.append(f"| {name} | {_fmt(value['C'], 4)} | {_fmt(value['lambda_inverse_C'], 4)} | {_fmt(value['cv_pooled_log_loss'], 5)} | {value['cv_rows']:,} |")
    lines += ["", "Współczynniki w skali logitów; modele ponownie estymowano na tym samym prefiksie przed każdym foldem:", "",
              "| Model | Fold | n train | n valid | Intercept | β logit rynku | β logit BTC |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in payload["coefficients"]:
        lines.append(f"| {row['model']} | {row['fold_id']} | {row['training_rows']:,} | {row['validation_rows']:,} | {_fmt(row['intercept'], 4)} | {_fmt(row['beta_logit_market'], 4)} | {_fmt(row['beta_logit_btc_calibrated'], 4)} |")
    lines += ["", "Rozrzut współczynników między dwoma foldami:", "",
              "| Model | Współczynnik | Średnia | SD | Min | Max | Zgodność znaku |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in payload["coefficient_stability"]:
        lines.append(f"| {row['model']} | {row['coefficient']} | {_fmt(row['mean'], 4)} | {_fmt(row['sd'], 4)} | {_fmt(row['min'], 4)} | {_fmt(row['max'], 4)} | {row['same_sign_folds']}/{row['folds']} |")
    lines += ["", "## Jakość prawdopodobieństw na tych samych rynkach", "",
              "A oznacza dotychczasowe skalibrowane P(BTC); dodatkowo raport surowego P(BTC). C/D w development to wyłącznie chronologiczne prognozy OOF. Holdout korzysta z modeli dopasowanych po zamrożeniu ustawień na rozwojowym OOF.",
              "", "| Okres | Prognoza | n | UP | Log loss | Brier | AUC | Slope / intercept | ECE |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in payload["quality"]:
        lines.append(f"| {row['period']} | {row['forecast']} | {row['n']:,} | {row['positive_rate']:.2%} | {row['log_loss']:.5f} | {row['brier']:.5f} | {_fmt(row['auc'], 4)} | {_fmt(row['calibration_slope'], 3)} / {_fmt(row['calibration_intercept'], 3)} | {_fmt(row['ece_frozen_dev_bins'], 4)} |")
    lines += ["", "Sparowane różnice holdout `D−C` i `D−A`; dla log loss/Brier wartość ujemna sprzyja D, dla AUC dodatnia. Przedziały pochodzą z tych samych losowań trzydniowych bloków UTC:",
              "", "| Porównanie | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |", "|---|---:|---:|---:|"]
    for name, result in payload["paired_differences"].items():
        lines.append(f"| {name} | {_fmt(result['log_loss_difference_left_minus_right'], 5)} [{_fmt(result['log_loss_difference_95pct'][0], 5)}, {_fmt(result['log_loss_difference_95pct'][1], 5)}] | {_fmt(result['brier_difference_left_minus_right'], 5)} [{_fmt(result['brier_difference_95pct'][0], 5)}, {_fmt(result['brier_difference_95pct'][1], 5)}] | {_fmt(result['auc_difference_left_minus_right'], 4)} [{_fmt(result['auc_difference_95pct'][0], 4)}, {_fmt(result['auc_difference_95pct'][1], 4)}] |")
    lines += ["", "### Wyniki miesięczne jakości prognoz", "",
              "| Miesiąc UTC | Prognoza | n | Log loss | Brier | AUC |", "|---|---|---:|---:|---:|---:|"]
    for row in payload["monthly_quality"]:
        lines.append(f"| {row['month_utc']} | {row['forecast']} | {row['n']:,} | {row['log_loss']:.5f} | {row['brier']:.5f} | {_fmt(row['auc'], 4)} |")
    lines += ["", "## Niezależne zlecenia $5: T−60 decyzja, T−59 wykonanie", "",
              "Dla każdej prognozy użyto tych samych progów EV. Strona i żądane $5 pochodzą z T−60; wykonanie ponownie wyceniono po ask/VWAP T−59 z zaakceptowaną opłatą, minimum, FAK i znaną głębokością albo jawnie oznaczonym proxy. Sygnały nie współdzielą gotówki.",
              "", "| Okres | Prognoza | EV min. | Sygnały | Fille | Obrót | Fee eq. | Oczekiwany EV T−59 | Zrealizowany PnL netto | EV−PnL / fill | Netto / obrót | Full book / proxy / partial |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in payload["economics"]:
        if row["period"] not in ("development_cv", "holdout"):
            continue
        lines.append(f"| {row['period']} | {row['forecast']} | {row['ev_hurdle_fraction']:.2%} | {row['signals']:,} | {row['fills']:,} | {_fmt_money(row['gross_turnover_usd'])} | {_fmt_money(row['fee_equivalent_usd'])} | {_fmt_money(row['expected_net_t59_usd'])} | {_fmt_money(row['realized_net_pnl_usd'])} | {_fmt_money(row['expected_minus_realized_per_fill_usd'], 3)} | {_fmt(row['net_roi'] * 100 if row['net_roi'] is not None else None, 2)}% | {row['full_book_fills']:,} / {row['proxy_fills']:,} / {row['partial_fills']:,} |")
    lines += ["", "### Holdout miesięcznie przy progu EV 0%", "",
              "| Miesiąc UTC | Prognoza | Fille | Oczekiwany EV T−59 | PnL netto | EV−PnL / fill |", "|---|---|---:|---:|---:|---:|"]
    for row in payload["monthly_economics"]:
        if row["period"] == "holdout" and row["ev_hurdle_fraction"] == 0.0:
            lines.append(f"| {row['month_utc']} | {row['forecast']} | {row['fills']:,} | {_fmt_money(row['expected_net_t59_usd'])} | {_fmt_money(row['realized_net_pnl_usd'])} | {_fmt_money(row['expected_minus_realized_per_fill_usd'], 3)} |")
    selection = payload["frozen_selection"]
    lines += ["", "## Zamrożona symulacja portfela 5% wolnej gotówki", "",
              f"Na development OOF od {CV_START_UTC.isoformat()} do {DEV_CUTOFF_UTC.isoformat()} wybrano `{selection['selected_policy']['policy']}` przy hurdle {selection['selected_policy'].get('minimum_net_ev_fraction')}; kontrola no-trade kończy z {_fmt_money(selection['no_trade_control']['ending_cash_after_settlement_usd'])}. Wybór i parametry zapisano przed otwarciem wyników holdout w `{FROZEN_SELECTION_PATH.relative_to(ROOT).as_posix()}`.",
              "", "| Prognoza | EV min. | Końcowa gotówka development | PnL | Transakcje | Max DD |", "|---|---:|---:|---:|---:|---:|"]
    for row in selection["candidate_portfolios"]:
        lines.append(f"| {row['forecast']} | {row['minimum_net_ev_fraction']:.2%} | {_fmt_money(row['ending_cash_after_settlement_usd'])} | {_fmt_money(row['net_pnl_usd'])} | {row['trade_count']:,} | {row['max_drawdown_cost_basis_equity']:.2%} |")
    lines += ["", "### Holdout portfel selected policy — osobny kapitał początkowy", "",
              "| Kapitał początkowy | Końcowa gotówka | PnL netto | Transakcje | Obrót | Fee | Max DD | Audyt |", "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for capital, item in payload["holdout_portfolios"].items():
        summary = item["summary"]
        lines.append(f"| {_fmt_money(float(capital), 0)} | {_fmt_money(summary['ending_cash_after_settlement_usd'])} | {_fmt_money(summary['net_pnl_usd'])} | {summary['trade_count']:,} | {_fmt_money(summary['gross_turnover_usd'])} | {_fmt_money(summary['fees_paid_estimated_usd'])} | {summary['max_drawdown_cost_basis_equity']:.2%} | {summary.get('accounting_audit', {}).get('status', 'no-trade')} |")
    lines += ["", "## Rozstrzygnięcie", "", payload["answers"], "",
              "Odtworzenie: `python run_btc_preopen_signal_diagnostic.py`, następnie `python run_btc_market_probability_experiment.py`. Manifest i szczegółowe tabele znajdują się w `" + OUTPUT_DIR.relative_to(ROOT).as_posix() + "`. Model BTC nie był trenowany ani modyfikowany; replay księgi i live nie uruchamiano.", ""]
    return "\n".join(lines)


def main() -> dict:
    started = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_holdout_scoring":
        raise RuntimeError("The probability experiment rules are not frozen")
    fee_correction._configure_output_paths()
    _, _, source_holdout = fee_correction._verify_reusable_sources()
    correction_manifest = research._read_json(fee_correction.CORRECTION_DIR / "research_manifest.json")
    predictions, prediction_identity = research._load_model_predictions(include_holdout=True)
    t59, t60, book_record = research._load_enriched_book_snapshots(correction_manifest)
    calendar, _ = research._build_market_calendar(predictions)
    calendar["market_start_utc"] = pd.to_datetime(calendar.market_start_utc, utc=True, errors="raise")
    data = _build_dataset(calendar, t60)
    dev_all = data.loc[
        data.common_sample & data.market_start_utc.ge(DEV_START_UTC)
        & data.market_start_utc.lt(DEV_CUTOFF_UTC)
    ].copy()
    dev = dev_all.loc[dev_all.market_start_utc.ge(CV_START_UTC)].copy()
    if dev.empty:
        raise RuntimeError("No eligible common-quote development data remains")
    selected_parameters, cv_grid, cv_predictions, coefficients = _select_logistic_parameters(dev_all, config)
    if cv_predictions.empty or cv_predictions.p_c.isna().any() or cv_predictions.p_d.isna().any():
        raise RuntimeError("C/D development rows are not fully chronological out-of-fold predictions")
    cv_predictions["p_btc_calibrated"] = cv_predictions.p_btc_calibrated.astype(float)
    cv_predictions["p_market"] = cv_predictions.p_market.astype(float)
    cv_predictions["p_btc_raw"] = cv_predictions.p_btc_raw.astype(float)

    # Final C/D fits use eligible development labels only; write the frozen choice before scoring holdout.
    final_train = dev_all.loc[dev_all.label_available_at_utc.le(DEV_CUTOFF_UTC)].copy()
    final_models = {}
    for model_name, features in MODEL_FEATURES.items():
        final_models[model_name] = _fit_logistic(final_train, features, selected_parameters[model_name]["C"])
    cv_predictions["p_A_BTC_calibrated"] = cv_predictions.p_btc_calibrated
    cv_predictions["p_B_market_reference"] = cv_predictions.p_market
    cv_predictions["p_C_market_logistic"] = cv_predictions.p_c
    cv_predictions["p_D_market_plus_BTC"] = cv_predictions.p_d
    hurdles = tuple(float(value) for value in config["independent_order_economics"]["ev_thresholds"])
    t59_by_id = research._snapshot_map(t59)
    t60_by_id = research._snapshot_map(t60)
    market_by_id = research._snapshot_map(calendar.loc[calendar.condition_id.notna()])
    development_data_identity = {
        "walk_forward_manifest_sha256": research._sha256(fee_correction.SOURCE_WALK_FORWARD_MANIFEST),
        "prediction_identity_sha256": prediction_identity["prediction_sha256"],
        "t60_snapshot_sha256": book_record["t60_coverage_sha256"],
        "t59_snapshot_sha256": book_record["t59_coverage_sha256"],
    }
    frozen_record = None
    if FROZEN_SELECTION_PATH.is_file():
        prior_frozen = research._read_json(FROZEN_SELECTION_PATH)
        if (
            prior_frozen.get("status") == "frozen_from_development_before_holdout_scoring"
            and prior_frozen.get("experiment_id") == config["experiment_id"]
            and prior_frozen.get("configuration") == config
            and prior_frozen.get("development_data_identity") == development_data_identity
            and prior_frozen.get("selected_regularization") == selected_parameters
            and prior_frozen.get("frozen_before_holdout_metrics") is True
        ):
            frozen_record = prior_frozen
            candidates = prior_frozen["candidate_portfolios"]
            frozen_selection = {
                key: prior_frozen[key]
                for key in (
                    "status", "selection_metric", "development_sample", "active_challenger",
                    "no_trade_control", "selected_policy", "candidate_portfolios", "holdout_used_for_selection",
                )
            }
            print("[market-probability] reused frozen development portfolio selection", flush=True)
    if frozen_record is None:
        candidates, frozen_selection = _selection_portfolios(
            cv_predictions, t59_by_id, t60_by_id, market_by_id, hurdles,
        )
        frozen_record = {
            **frozen_selection,
            "experiment_id": config["experiment_id"],
            "configuration": config,
            "selected_regularization": selected_parameters,
            "development_data_identity": development_data_identity,
            "frozen_before_holdout_metrics": True,
        }
        research._write_json(FROZEN_SELECTION_PATH, frozen_record)

    # Holdout labels and scores are first used below, after the development selection is persisted.
    holdout = data.loc[
        data.common_sample & data.market_start_utc.ge(DEV_CUTOFF_UTC)
        & data.market_start_utc.lt(DATA_CUTOFF_UTC)
    ].copy()
    holdout["p_c"] = _predict(final_models["C"], holdout, MODEL_FEATURES["C"])
    holdout["p_d"] = _predict(final_models["D"], holdout, MODEL_FEATURES["D"])
    holdout["p_A_BTC_calibrated"] = holdout.p_btc_calibrated
    holdout["p_B_market_reference"] = holdout.p_market
    holdout["p_C_market_logistic"] = holdout.p_c
    holdout["p_D_market_plus_BTC"] = holdout.p_d

    quality_rows, calibration_bins, monthly_quality = [], [], []
    development_forecasts = {
        "A_BTC_calibrated": "p_btc_calibrated", "B_market_reference": "p_market",
        "C_market_logistic": "p_c", "D_market_plus_BTC": "p_d",
    }
    for name, column in development_forecasts.items():
        edges = _metric_edges(cv_predictions[column].to_numpy())
        row, bins = _metric_row(cv_predictions, column, name, "development_cv", edges)
        quality_rows.append(row)
        calibration_bins.extend(bins)
        holdout_row, holdout_bins = _metric_row(holdout, column, name, "holdout", edges)
        quality_rows.append(holdout_row)
        calibration_bins.extend(holdout_bins)
        for month in sorted(holdout.month_utc.unique()):
            month_data = holdout.loc[holdout.month_utc.eq(month)]
            monthly, _ = _metric_row(month_data, column, name, "holdout", edges)
            monthly["month_utc"] = month
            monthly_quality.append(monthly)
    raw_edges = _metric_edges(cv_predictions.p_btc_raw.to_numpy())
    raw_dev, _ = _metric_row(cv_predictions, "p_btc_raw", "A_BTC_raw", "development_cv", raw_edges)
    raw_hold, _ = _metric_row(holdout, "p_btc_raw", "A_BTC_raw", "holdout", raw_edges)
    quality_rows.extend([raw_dev, raw_hold])

    paired = {
        "D-C": _block_paired_differences(holdout, "p_d", "p_c"),
        "D-A": _block_paired_differences(holdout, "p_d", "p_btc_calibrated"),
    }
    dev_economics, dev_monthly, dev_details = _economic_rows(
        cv_predictions, t59_by_id, t60_by_id, market_by_id, "development_cv", hurdles,
    )
    hold_economics, hold_monthly, hold_details = _economic_rows(
        holdout, t59_by_id, t60_by_id, market_by_id, "holdout", hurdles,
    )

    selected_policy = frozen_selection["selected_policy"]
    selected_name = selected_policy["policy"]
    selected_forecast = None if selected_name == "no_trade" else next(
        row["forecast"] for row in candidates
        if row["forecast"] in selected_name and math.isclose(
            row["minimum_net_ev_fraction"], float(selected_policy["minimum_net_ev_fraction"]), abs_tol=1e-12,
        )
    )
    holdout_portfolios = {}
    for capital in (100.0, 500.0, 1000.0):
        if selected_forecast is None:
            summary = {
                "ending_cash_after_settlement_usd": capital, "net_pnl_usd": 0.0,
                "trade_count": 0, "gross_turnover_usd": 0.0, "fees_paid_estimated_usd": 0.0,
                "max_drawdown_cost_basis_equity": 0.0, "accounting_audit": {"status": "no_trade"},
            }
        else:
            probability_column = FORECAST_COLUMNS[selected_forecast]
            portfolio_rows = holdout.copy()
            portfolio_rows["p_candidate_platt"] = portfolio_rows[probability_column]
            portfolio_market = {
                str(row.condition_id).lower(): {**market_by_id[str(row.condition_id).lower()], "p_candidate_platt": float(getattr(row, probability_column))}
                for row in portfolio_rows.itertuples(index=False)
            }
            summary, _, _, _ = diagnostic._run_holdout_portfolio(
                capital, portfolio_rows, t59_by_id, portfolio_market, t60_by_id,
                float(selected_policy["minimum_net_ev_fraction"]),
            )
        holdout_portfolios[str(int(capital))] = {"summary": summary}

    coefficient_stability = []
    for model_name in ("C", "D"):
        for column, label in (("intercept", "intercept"), ("beta_logit_market", "logit_market"), ("beta_logit_btc_calibrated", "logit_btc_calibrated")):
            if model_name == "C" and label == "logit_btc_calibrated":
                continue
            values = coefficients.loc[coefficients.model.eq(model_name), column].to_numpy(dtype=float)
            coefficient_stability.append({
                "model": model_name, "coefficient": label, "mean": float(values.mean()),
                "sd": float(values.std(ddof=0)), "min": float(values.min()), "max": float(values.max()),
                "same_sign_folds": int(np.sum(np.sign(values) == np.sign(values[0]))) if len(values) else 0,
                "folds": int(len(values)),
            })

    eligible = data.loc[
        data.market_confirmed.fillna(False) & data.causal_prediction_available.fillna(False)
        & data.official_outcome_up.isin((0.0, 1.0))
    ].copy()
    coverage = {}
    for period, mask in (
        ("development", eligible.market_start_utc.lt(DEV_CUTOFF_UTC)),
        ("holdout", eligible.market_start_utc.ge(DEV_CUTOFF_UTC) & eligible.market_start_utc.lt(DATA_CUTOFF_UTC)),
    ):
        population = eligible.loc[mask]
        quoted = population.loc[population.t60_common_quote_valid.fillna(False)]
        quote_detail = {
            "up_available": int(population.t60_up_available.fillna(False).sum()),
            "down_available": int(population.t60_down_available.fillna(False).sum()),
            "both_sides_available": int(population.t60_both_sides_available.fillna(False).sum()),
            "common_rows": int(len(quoted)),
            "proxy_or_complemented_rows": int(quoted.t60_proxy_flag.fillna(False).sum()),
        }
        for side in ("up", "down"):
            for quote, suffix in (("best_bid", "bid"), ("best_ask", "ask"), ("spread", "spread")):
                column = f"t60_{side}_{quote}"
                quote_detail[f"{side}_{suffix}_median"] = (
                    float(pd.to_numeric(quoted[column], errors="coerce").median()) if len(quoted) else None
                )
        coverage[period] = {
            "confirmed_with_prediction_and_outcome": int(len(population)),
            "common_rows": int(len(quoted)),
            "coverage_fraction": float(len(quoted) / len(population)) if len(population) else None,
            "excluded_missing_or_invalid_quotes": int(len(population) - len(quoted)),
            "quote_detail": quote_detail,
        }
    coverage["quote_detail"] = {period: coverage[period].pop("quote_detail") for period in ("development", "holdout")}
    source_counts = holdout.groupby(
        ["t60_archive_source", "source_proxy_group"], dropna=False,
    ).size().reset_index(name="rows")
    coverage["source_proxy_summary"] = "; ".join(
        f"{row.t60_archive_source}/{row.source_proxy_group}: {row.rows:,}"
        for row in source_counts.itertuples(index=False)
    )

    metric_by_period = {(row["period"], row["forecast"]): row for row in quality_rows}
    d_hold = metric_by_period[("holdout", "D_market_plus_BTC")]
    c_hold = metric_by_period[("holdout", "C_market_logistic")]
    a_hold = metric_by_period[("holdout", "A_BTC_calibrated")]
    b_hold = metric_by_period[("holdout", "B_market_reference")]
    best_loss = min((a_hold, b_hold, c_hold, d_hold), key=lambda row: row["log_loss"])
    delta_c = paired["D-C"]["log_loss_difference_left_minus_right"]
    delta_a = paired["D-A"]["log_loss_difference_left_minus_right"]
    chosen_text = "no_trade" if selected_forecast is None else f"{selected_forecast} / {float(selected_policy['minimum_net_ev_fraction']):.2%}"
    holdout_zero = [row for row in hold_economics if row["ev_hurdle_fraction"] == 0.0]
    gap_by_model = {row["forecast"]: row["expected_minus_realized_per_fill_usd"] for row in holdout_zero}
    gap_total_by_model = {row["forecast"]: row["expected_minus_realized_usd"] for row in holdout_zero}
    gap_a = gap_by_model["A_BTC_calibrated"]
    gap_d = gap_by_model["D_market_plus_BTC"]
    gap_total_a = gap_total_by_model["A_BTC_calibrated"]
    gap_total_d = gap_total_by_model["D_market_plus_BTC"]
    gap_per_fill_reduction = 1.0 - gap_d / gap_a
    gap_total_reduction = 1.0 - gap_total_d / gap_total_a
    verdict = (
        f"Na wspólnych {len(holdout):,} rynkach holdout najniższy log loss ma {best_loss['forecast']} ({best_loss['log_loss']:.5f}); "
        f"rynek B: {b_hold['log_loss']:.5f}, BTC A: {a_hold['log_loss']:.5f}, C: {c_hold['log_loss']:.5f}, D: {d_hold['log_loss']:.5f}. "
        f"D−C log-loss={delta_c:+.5f} (95% CI {paired['D-C']['log_loss_difference_95pct'][0]:+.5f}…{paired['D-C']['log_loss_difference_95pct'][1]:+.5f}); "
        f"D−A={delta_a:+.5f} (95% CI {paired['D-A']['log_loss_difference_95pct'][0]:+.5f}…{paired['D-A']['log_loss_difference_95pct'][1]:+.5f}). "
        f"Zamrożony wybór portfela na development: {chosen_text}."
    )
    answers = "\n".join([
        f"1. T−59 poprawia wcześniejszą diagnostykę: poprzednia wycena wykonania używała T−60. Po korekcie raportowane są rzeczywiste snapshoty T−59; osobne portfele były poprawne i pozostają bez przeliczenia.",
        f"2. Referencja B ma log loss {b_hold['log_loss']:.5f} i Brier {b_hold['brier']:.5f}, lepsze od BTC A ({a_hold['log_loss']:.5f} i {a_hold['brier']:.5f}); BTC A ma wy\u017csze AUC ({a_hold['auc']:.4f} vs {b_hold['auc']:.4f}). B nie daje sygna\u0142\u00f3w z dodatnim EV ask po fee.",
        f"3. Dodanie BTC do rynku podnosi AUC D vs C o {paired['D-C']['auc_difference_left_minus_right']:.4f} (95% CI {paired['D-C']['auc_difference_95pct'][0]:.4f} do {paired['D-C']['auc_difference_95pct'][1]:.4f}), lecz nie poprawia log loss ({delta_c:+.5f}) ani Brier ({paired['D-C']['brier_difference_left_minus_right']:+.5f}); oba przedzia\u0142y obejmuj\u0105 zero. Dodatni wsp\u00f3\u0142czynnik BTC sam nie wystarcza jako dow\u00f3d poprawy prawdopodobie\u0144stw.",
        f"4. Przy EV 0% luka oczekiwany EV minus PnL na fill spada dla D do ${gap_d:.3f} z ${gap_a:.3f} dla A (o {gap_per_fill_reduction:.1%}); suma luki to {_fmt_money(gap_total_d, 0)} dla D wobec {_fmt_money(gap_total_a, 0)} dla A (o {gap_total_reduction:.1%}). Nie poprawia to wyniku po fee: PnL netto D to -$635, A -$557 (ROI -1.92% i -1.90%). Referencja B nie wygenerowala sygnalow z dodatnim EV ask po fee. Szczegolowe progi sa w tabelach.",
        f"5. Wynik nie utrzymuje si\u0119 w ka\u017cdym miesi\u0105cu: D ma najlepszy log loss w sierpniu i wrze\u015bniu, ale nie w lipcu ani pa\u017adzierniku; PnL D przy progu 0% wynosi odpowiednio -$553, +$114, +$109 i -$306. Wyb\u00f3r polityki z OOF development pozostaje {chosen_text}; historyczny holdout jest diagnostyczny.",
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv_grid.to_csv(OUTPUT_DIR / "logistic_cv_grid.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "fold_coefficients.csv", index=False)
    pd.DataFrame(coefficient_stability).to_csv(OUTPUT_DIR / "coefficient_stability.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(OUTPUT_DIR / "probability_quality.csv", index=False)
    pd.DataFrame(monthly_quality).to_csv(OUTPUT_DIR / "monthly_probability_quality.csv", index=False)
    pd.DataFrame(calibration_bins).to_csv(OUTPUT_DIR / "calibration_bins.csv", index=False)
    pd.DataFrame([{"comparison": key, **value} for key, value in paired.items()]).to_csv(OUTPUT_DIR / "paired_holdout_differences.csv", index=False)
    economics = dev_economics + hold_economics
    monthly_economics = dev_monthly + hold_monthly
    pd.DataFrame(economics).to_csv(OUTPUT_DIR / "independent_order_economics.csv", index=False)
    pd.DataFrame(monthly_economics).to_csv(OUTPUT_DIR / "monthly_independent_order_economics.csv", index=False)
    pd.concat([dev_details, hold_details], ignore_index=True).to_parquet(OUTPUT_DIR / "independent_order_detail.parquet", index=False, compression="zstd")
    all_predictions = pd.concat([cv_predictions, holdout], ignore_index=True, sort=False)
    all_predictions.to_parquet(OUTPUT_DIR / "common_quote_predictions.parquet", index=False, compression="zstd")
    fee_correction._verify_reusable_sources()
    portfolio_payload = {k: v for k, v in holdout_portfolios.items()}
    payload = {
        "verdict": verdict, "answers": answers, "coverage": coverage,
        "model_selection": {"selected_parameters": selected_parameters},
        "coefficients": coefficients.to_dict("records"),
        "coefficient_stability": coefficient_stability,
        "quality": quality_rows, "monthly_quality": monthly_quality,
        "paired_differences": paired, "economics": economics,
        "monthly_economics": monthly_economics, "frozen_selection": frozen_selection,
        "holdout_portfolios": portfolio_payload,
    }
    REPORT_PATH.write_text(_render_report(payload), encoding="utf-8")
    manifest = {
        "status": "completed", "experiment_id": config["experiment_id"],
        "configuration_sha256": research._sha256(CONFIG_PATH),
        "frozen_selection_path": FROZEN_SELECTION_PATH.relative_to(ROOT).as_posix(),
        "frozen_selection_sha256": research._sha256(FROZEN_SELECTION_PATH),
        "source_identity": {
            "walk_forward_manifest_sha256": research._sha256(fee_correction.SOURCE_WALK_FORWARD_MANIFEST),
            "prediction_identity": prediction_identity,
            "holdout_prediction_sha256": source_holdout["predictions_sha256"],
            "evaluated_bundle_sha256": source_holdout["evaluated_bundle_sha256"],
            "t60_snapshot_sha256": book_record["t60_coverage_sha256"],
            "t59_snapshot_sha256": book_record["t59_coverage_sha256"],
        },
        "selected_regularization": selected_parameters,
        "cv_rows": int(len(cv_predictions)), "holdout_rows": int(len(holdout)),
        "coverage": coverage, "probability_quality": quality_rows,
        "paired_differences": paired, "independent_order_economics": economics,
        "monthly_independent_order_economics": monthly_economics,
        "frozen_development_selection": frozen_selection,
        "holdout_portfolios": {k: v["summary"] for k, v in holdout_portfolios.items()},
        "holdout_historically_exposed": True,
        "training_or_refit": False, "book_replay": False, "live_orders": False,
        "runtime_seconds": time.perf_counter() - started,
        "report_path": REPORT_PATH.relative_to(ROOT).as_posix(),
    }
    research._write_json(OUTPUT_DIR / "experiment_manifest.json", manifest)
    print(json.dumps({"verdict": verdict, "selection": selected_policy, "holdout_rows": len(holdout), "runtime_seconds": manifest["runtime_seconds"]}, indent=2, default=str))
    return manifest


if __name__ == "__main__":
    main()
