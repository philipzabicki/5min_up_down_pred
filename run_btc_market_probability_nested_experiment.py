"""Nested development comparison of BTC-market probability corrections."""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import run_btc_market_probability_experiment as previous
import run_btc_preopen_fee_correction as fee_correction
import run_btc_preopen_research as research
import run_btc_preopen_signal_diagnostic as diagnostic
import run_btc_preopen_t59_reassessment as replay
from utils.polymarket_policy import calibration_metrics


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_market_probability_nested_20261012.json"
OUTPUT_DIR = ROOT / "data/analysis/polymarket/BTC/btc_market_probability_nested_20261012"
REPORT_PATH = ROOT / "reports/btc_preopen/btc_market_probability_nested_20261012.md"
SUMMARY_TABLE_DIR = ROOT / "reports/btc_preopen/btc_market_probability_nested_20261012"
FROZEN_SELECTION_PATH = ROOT / "reports/btc_preopen/btc_market_probability_nested_frozen_20261012.json"
DEV_CUTOFF_UTC = pd.Timestamp("2026-07-01T00:00:00Z")
DATA_CUTOFF_UTC = pd.Timestamp("2026-10-07T00:00:00Z")
MODEL_NAMES = (
    "A_BTC_calibrated", "B_market_reference", "C_market_logistic",
    "D_market_plus_BTC", "E_probability_mixture", "F_regularized_logit_correction",
)
PROBABILITY_COLUMNS = {
    "A_BTC_calibrated": "p_A",
    "B_market_reference": "p_B",
    "C_market_logistic": "p_C",
    "D_market_plus_BTC": "p_D",
    "E_probability_mixture": "p_E",
    "F_regularized_logit_correction": "p_F",
}
META_FAMILIES = ("C", "D", "E", "F")


def _specs(family: str, config: dict) -> list[dict]:
    grids = config["parameter_grids"]
    if family in ("C", "D"):
        key = f"{family}_inverse_regularization_C"
        return [{"parameter": "C", "value": float(value)} for value in grids[key]]
    if family == "E":
        return [{"parameter": "w", "value": float(value)} for value in grids["E_btc_weights"]]
    if family == "F":
        return [{"parameter": "no_correction", "value": None}] + [
            {"parameter": "lambda", "value": float(value)}
            for value in grids["F_l2_penalty_on_a_and_b"]
        ]
    raise ValueError(f"Unknown model family: {family}")


def _spec_name(family: str, spec: dict) -> str:
    if spec["parameter"] == "no_correction":
        return "F_no_correction"
    return f"{family}_{spec['parameter']}_{float(spec['value']):g}"


def _training_validation_rows(frame: pd.DataFrame, fold: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(fold["start_inclusive_utc"])
    end = pd.Timestamp(fold["end_exclusive_utc"])
    train = frame.loc[
        frame.market_start_utc.lt(start) & frame.label_available_at_utc.le(start)
    ]
    valid = frame.loc[frame.market_start_utc.ge(start) & frame.market_start_utc.lt(end)]
    if train.empty or valid.empty or train.official_outcome_up.nunique() != 2:
        raise RuntimeError(
            f"Insufficient rows for {fold['fold_id']}: train={len(train)}, valid={len(valid)}"
        )
    if not train.label_available_at_utc.le(start).all():
        raise AssertionError(f"Unpurged labels in {fold['fold_id']}")
    return train, valid


def _fit_logistic(train: pd.DataFrame, family: str, c_value: float) -> LogisticRegression:
    columns = _logistic_features(family)
    model = LogisticRegression(
        C=float(c_value), l1_ratio=0.0, solver="lbfgs", fit_intercept=True,
        class_weight=None, max_iter=2000,
    )
    model.fit(train.loc[:, columns].to_numpy(dtype=np.float64), train.official_outcome_up.astype(int))
    return model


def _logistic_features(family: str) -> tuple[str, ...]:
    return ("logit_market",) if family == "C" else ("logit_market", "logit_btc_calibrated")


def _fit_regularized_offset(train: pd.DataFrame, penalty: float | None) -> dict:
    if penalty is None:
        return {"a": 0.0, "b": 0.0, "lambda": None, "no_correction": True}
    market = train.logit_market.to_numpy(dtype=np.float64)
    btc_difference = (
        train.logit_btc_calibrated.to_numpy(dtype=np.float64) - market
    )
    target = train.official_outcome_up.to_numpy(dtype=np.float64)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = parameters
        logits = market + a + b * btc_difference
        probabilities = expit(logits)
        loss = np.mean(np.logaddexp(0.0, logits) - target * logits)
        loss += float(penalty) * float(parameters @ parameters) / 2.0
        gradient = np.array([
            np.mean(probabilities - target),
            np.mean((probabilities - target) * btc_difference),
        ]) + float(penalty) * parameters
        return float(loss), gradient

    result = minimize(
        objective, np.zeros(2, dtype=np.float64), jac=True, method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Regularized offset fit failed: {result.message}")
    return {
        "a": float(result.x[0]), "b": float(result.x[1]),
        "lambda": float(penalty), "no_correction": False,
    }


def _fit_spec(train: pd.DataFrame, family: str, spec: dict):
    value = spec["value"]
    if family in ("C", "D"):
        return _fit_logistic(train, family, float(value))
    if family == "F":
        return _fit_regularized_offset(train, None if spec["parameter"] == "no_correction" else float(value))
    return None


def _predict_spec(frame: pd.DataFrame, family: str, spec: dict, model=None) -> np.ndarray:
    market = frame.p_market.to_numpy(dtype=np.float64)
    btc = frame.p_btc_calibrated.to_numpy(dtype=np.float64)
    if family == "A":
        return btc
    if family == "B":
        return market
    if family in ("C", "D"):
        columns = _logistic_features(family)
        return model.predict_proba(frame.loc[:, columns].to_numpy(dtype=np.float64))[:, 1]
    if family == "E":
        weight = float(spec["value"])
        return (1.0 - weight) * market + weight * btc
    if family == "F":
        if spec["parameter"] == "no_correction":
            return market.copy()
        logits = frame.logit_market.to_numpy(dtype=np.float64)
        difference = frame.logit_btc_calibrated.to_numpy(dtype=np.float64) - logits
        return expit(logits + model["a"] + model["b"] * difference)
    raise ValueError(f"Unknown model family: {family}")


def _tie_key(family: str, spec: dict) -> tuple:
    if family in ("C", "D"):
        return (float(spec["value"]),)
    if family == "E":
        return (float(spec["value"]),)
    if spec["parameter"] == "no_correction":
        return (0, 0.0)
    return (1, -float(spec["value"]))


def _tune_family(prefix: pd.DataFrame, folds: list[dict], family: str, config: dict) -> tuple[dict, list[dict]]:
    accumulated: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    rows = []
    specs = _specs(family, config)
    for fold in folds:
        train, valid = _training_validation_rows(prefix, fold)
        y = valid.official_outcome_up.to_numpy(dtype=np.int8)
        for spec in specs:
            model = _fit_spec(train, family, spec)
            prediction = np.clip(_predict_spec(valid, family, spec, model), 1e-12, 1.0 - 1e-12)
            key = _spec_name(family, spec)
            accumulated.setdefault(key, []).append((prediction, y))
            rows.append({
                "selection_stage": "inner", "family": family,
                "variant": key, "parameter": spec["parameter"],
                "value": spec["value"], "fold_id": fold["fold_id"],
                "train_rows": int(len(train)), "validation_rows": int(len(valid)),
                "latest_train_label_available_at_utc": train.label_available_at_utc.max().isoformat(),
                "validation_log_loss": float(log_loss(y, prediction, labels=[0, 1])),
                "validation_brier": float(brier_score_loss(y, prediction)),
            })
    summary = {}
    for spec in specs:
        key = _spec_name(family, spec)
        predictions = np.concatenate([pair[0] for pair in accumulated[key]])
        outcomes = np.concatenate([pair[1] for pair in accumulated[key]])
        summary[key] = {
            "spec": spec,
            "log_loss": float(log_loss(outcomes, predictions, labels=[0, 1])),
            "brier": float(brier_score_loss(outcomes, predictions)),
            "rows": int(len(outcomes)),
        }
    selected_key = min(
        summary,
        key=lambda key: (summary[key]["log_loss"], _tie_key(family, summary[key]["spec"])),
    )
    selected = dict(summary[selected_key]["spec"])
    selected.update({
        "variant": selected_key,
        "inner_log_loss": summary[selected_key]["log_loss"],
        "inner_brier": summary[selected_key]["brier"],
        "inner_rows": summary[selected_key]["rows"],
    })
    for row in rows:
        row["pooled_inner_log_loss"] = summary[row["variant"]]["log_loss"]
        row["selected"] = row["variant"] == selected_key
    return selected, rows


def _model_from_choice(train: pd.DataFrame, family: str, choice: dict):
    return _fit_spec(train, family, {"parameter": choice["parameter"], "value": choice["value"]})


def _choice_for_fit(choice: dict) -> dict:
    return {"parameter": choice["parameter"], "value": choice["value"]}


def _outer_predictions(dev: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    output = []
    choice_rows = []
    grid_rows = []
    for outer in config["development_outer_folds"]:
        outer_start = pd.Timestamp(outer["start_inclusive_utc"])
        outer_end = pd.Timestamp(outer["end_exclusive_utc"])
        prefix = dev.loc[
            dev.market_start_utc.lt(outer_start) & dev.label_available_at_utc.le(outer_start)
        ]
        valid = dev.loc[dev.market_start_utc.ge(outer_start) & dev.market_start_utc.lt(outer_end)].copy()
        if valid.empty:
            raise RuntimeError(f"No rows in outer fold {outer['fold_id']}")
        valid["p_A"] = valid.p_btc_calibrated
        valid["p_B"] = valid.p_market
        valid["outer_fold_id"] = outer["fold_id"]
        for family in META_FAMILIES:
            choice, grid = _tune_family(prefix, outer["inner_validation_folds"], family, config)
            model = _model_from_choice(prefix, family, choice)
            valid[f"p_{family}"] = _predict_spec(valid, family, _choice_for_fit(choice), model)
            choice_rows.append({
                "outer_fold_id": outer["fold_id"], "family": family,
                "variant": choice["variant"], "parameter": choice["parameter"],
                "value": choice["value"], "inner_log_loss": choice["inner_log_loss"],
                "inner_rows": choice["inner_rows"],
                "outer_train_rows": int(len(prefix)), "outer_valid_rows": int(len(valid)),
                "latest_outer_train_label_available_at_utc": prefix.label_available_at_utc.max().isoformat(),
            })
            for row in grid:
                grid_rows.append({"selection_scope": outer["fold_id"], **row})
            if family in ("C", "D"):
                coefficients = dict(zip(_logistic_features(family), model.coef_[0].astype(float)))
                choice_rows[-1].update({
                    "intercept": float(model.intercept_[0]),
                    "beta_logit_market": float(coefficients["logit_market"]),
                    "beta_logit_btc_calibrated": float(coefficients.get("logit_btc_calibrated", 0.0)),
                })
            elif family == "E":
                choice_rows[-1].update({"mixture_weight_w": float(choice["value"])})
            else:
                fitted = model or {"a": 0.0, "b": 0.0}
                choice_rows[-1].update({"offset_a": fitted["a"], "btc_difference_b": fitted["b"]})
        if valid["p_C"].isna().any() or valid["p_D"].isna().any() or valid["p_E"].isna().any() or valid["p_F"].isna().any():
            raise AssertionError(f"Incomplete outer probabilities in {outer['fold_id']}")
        output.append(valid)
    predictions = pd.concat(output, ignore_index=True, sort=False).sort_values("market_start_utc", kind="stable")
    if predictions.condition_id.duplicated().any():
        raise RuntimeError("Nested outer development folds overlap")
    return predictions.reset_index(drop=True), choice_rows, grid_rows


def _final_development_choices(dev: pd.DataFrame, config: dict) -> tuple[dict, list[dict], dict]:
    choices = {}
    grid_rows = []
    final_train = dev.loc[dev.label_available_at_utc.le(DEV_CUTOFF_UTC)]
    fit_rows = {}
    for family in META_FAMILIES:
        choice, grid = _tune_family(dev, config["final_development_selection_folds"], family, config)
        model = _model_from_choice(final_train, family, choice)
        choices[family] = {**choice, "fit_rows": int(len(final_train))}
        fit_rows[family] = model
        for row in grid:
            grid_rows.append({"selection_scope": "final_holdout_freeze", **row})
        if family in ("C", "D"):
            coefficients = dict(zip(_logistic_features(family), model.coef_[0].astype(float)))
            choices[family].update({
                "intercept": float(model.intercept_[0]),
                "beta_logit_market": float(coefficients["logit_market"]),
                "beta_logit_btc_calibrated": float(coefficients.get("logit_btc_calibrated", 0.0)),
            })
        elif family == "E":
            choices[family]["mixture_weight_w"] = float(choice["value"])
        else:
            fitted = model or {"a": 0.0, "b": 0.0}
            choices[family].update({"offset_a": fitted["a"], "btc_difference_b": fitted["b"]})
    return choices, grid_rows, fit_rows


def _score_metrics(frame: pd.DataFrame, probability_column: str, model_name: str, period: str, edges: list[float]) -> tuple[dict, list[dict]]:
    return previous._metric_row(frame, probability_column, model_name, period, edges)


def _paired_comparisons(frame: pd.DataFrame) -> dict:
    columns = {name: PROBABILITY_COLUMNS[name] for name in MODEL_NAMES}
    comparisons = (
        ("E-B", "E_probability_mixture", "B_market_reference"),
        ("E-D", "E_probability_mixture", "D_market_plus_BTC"),
        ("F-B", "F_regularized_logit_correction", "B_market_reference"),
        ("F-D", "F_regularized_logit_correction", "D_market_plus_BTC"),
        ("D-B", "D_market_plus_BTC", "B_market_reference"),
        ("C-B", "C_market_logistic", "B_market_reference"),
        ("D-C", "D_market_plus_BTC", "C_market_logistic"),
        ("B-A", "B_market_reference", "A_BTC_calibrated"),
    )
    return {
        name: previous._block_paired_differences(frame, columns[left], columns[right])
        for name, left, right in comparisons
    }


def _block_interval(daily_values: pd.Series, seed: int, block_days: int = 3, replicates: int = 1000) -> dict:
    values = daily_values.to_numpy(dtype=np.float64)
    if not len(values):
        return {"mean_daily_pnl": 0.0, "mean_daily_pnl_95pct": [0.0, 0.0], "total_pnl_95pct": [0.0, 0.0]}
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(len(values) / block_days))
    total_draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        starts = rng.integers(0, len(values), size=block_count)
        positions = np.concatenate([
            (np.arange(start, start + block_days) % len(values)) for start in starts
        ])[:len(values)]
        total_draws[index] = float(values[positions].sum())
    return {
        "mean_daily_pnl": float(values.mean()),
        "mean_daily_pnl_95pct": np.quantile(total_draws / len(values), [0.025, 0.975]).tolist(),
        "total_pnl_95pct": np.quantile(total_draws, [0.025, 0.975]).tolist(),
        "method": f"circular moving {block_days}-day UTC blocks, {replicates} replicates, seed {seed}",
    }


def _economic_detail(
    frame: pd.DataFrame,
    forecasts: tuple[str, ...],
    t59_by_id: dict,
    t60_by_id: dict,
    market_by_id: dict,
    period: str,
    probability_columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    probability_columns = probability_columns or PROBABILITY_COLUMNS
    rows = []
    for forecast in forecasts:
        probability_column = probability_columns[forecast]
        print(f"[nested-probability] T-60/T-59 independent $5 orders: {period}, {forecast}", flush=True)
        for item in frame.itertuples(index=False):
            cid = str(item.condition_id).lower()
            p_up = float(getattr(item, probability_column))
            market = dict(market_by_id[cid])
            market["p_candidate_platt"] = p_up
            t60 = t60_by_id.get(cid)
            t59 = t59_by_id.get(cid)
            if t60 is None or t59 is None:
                continue
            decision = replay._evaluate_market_sides(market, t60, 5.0)
            side = decision["chosen_side"]
            if side is None:
                continue
            execution = replay._price_side(t59, market, side, 5.0)
            fill = execution.get("fill") if execution.get("priceable") else None
            p_side = p_up if side == "up" else 1.0 - p_up
            outcome_side = int(float(item.official_outcome_up) == (1.0 if side == "up" else 0.0))
            rows.append({
                "period": period, "forecast": forecast,
                "outer_fold_id": getattr(item, "outer_fold_id", None),
                "month_utc": item.market_start_utc.strftime("%Y-%m"),
                "day_utc": item.market_start_utc.floor("D"),
                "market_start_utc": item.market_start_utc,
                "condition_id": cid, "chosen_side": side, "p_up": p_up,
                "p_side": p_side, "outcome_side": outcome_side,
                "t60_ev_usd": float(decision["chosen_ev_usd"]),
                "t60_ev_fraction": float(decision["chosen_ev_usd"]) / 5.0,
                "execution_status": "filled" if fill is not None else "t59_unpriceable",
                "execution_reason": None if fill is not None else execution.get("reason"),
                "execution_source": fill.get("execution_source") if fill else None,
                "depth_known": bool(fill.get("depth_known")) if fill else None,
                "execution_assumption": fill.get("execution_assumption") if fill else None,
                "fill_status": fill.get("fill_status") if fill else None,
                "filled_gross_usd": float(fill["gross_usd"]) if fill else None,
                "gross_shares": float(fill["gross_shares"]) if fill else None,
                "vwap": float(fill["vwap"]) if fill else None,
                "net_shares": float(fill["shares"]) if fill else None,
                "cash_debit_usd": float(fill["cash_debit_usd"]) if fill else None,
                "fee_equivalent_usd": float(fill.get("fee_usd_entry_equivalent") or fill.get("fee_usd") or 0.0) if fill else None,
                "expected_net_t59_usd": p_side * float(fill["shares"]) - float(fill["cash_debit_usd"]) if fill else None,
                "gross_pnl_usd": (outcome_side * float(fill["gross_shares"]) - float(fill["gross_usd"])) if fill else None,
                "net_pnl_usd": (outcome_side * float(fill["shares"]) - float(fill["cash_debit_usd"])) if fill else None,
                "realized_fee_effect_usd": (
                    outcome_side * float(fill["gross_shares"]) - float(fill["gross_usd"])
                    - (outcome_side * float(fill["shares"]) - float(fill["cash_debit_usd"]))
                ) if fill else None,
            })
    return pd.DataFrame(rows)


def _policy_summaries(
    details: pd.DataFrame,
    frame: pd.DataFrame,
    hurdles: tuple[float, ...],
    config: dict,
    model_names: tuple[str, ...] = MODEL_NAMES,
) -> tuple[list[dict], list[dict], list[dict]]:
    rows = []
    fold_rows = []
    monthly_rows = []
    first_day = frame.market_start_utc.min().floor("D")
    last_day = frame.market_start_utc.max().floor("D")
    all_days = pd.date_range(first_day, last_day, freq="D")
    seed = int(config["probability_evaluation"]["block_bootstrap"].split("seed ")[-1])
    for forecast in model_names:
        forecast_details = details.loc[details.forecast.eq(forecast)]
        for hurdle in hurdles:
            selected = forecast_details.loc[forecast_details.t60_ev_fraction.ge(hurdle - 1e-12)]
            fills = selected.loc[selected.execution_status.eq("filled")].copy()
            daily = pd.Series(0.0, index=all_days)
            if len(fills):
                daily_values = fills.groupby("day_utc").net_pnl_usd.sum()
                daily.loc[daily_values.index] = daily_values.to_numpy(dtype=np.float64)
            uncertainty = _block_interval(daily, seed)
            active_days = int(fills.day_utc.nunique()) if len(fills) else 0
            positive_total = float(fills.loc[fills.net_pnl_usd.gt(0), "net_pnl_usd"].sum()) if len(fills) else 0.0
            absolute_total = float(fills.net_pnl_usd.abs().sum()) if len(fills) else 0.0
            daily_abs = daily.abs().sort_values(ascending=False)
            daily_positive = daily.clip(lower=0.0).sort_values(ascending=False)
            row = {
                "period": "development_oof" if "outer_fold_id" in frame else "holdout_diagnostic",
                "policy": f"{forecast}_ev_{hurdle:.4%}",
                "forecast": forecast, "ev_hurdle_fraction": float(hurdle),
                "opportunity_markets": int(len(frame)),
                "signals": int(len(selected)), "fills": int(len(fills)),
                "no_trade_markets": int(len(frame) - len(selected)),
                "no_trade_rate": float((len(frame) - len(selected)) / len(frame)) if len(frame) else None,
                "unfilled_signals": int(len(selected) - len(fills)),
                "active_utc_days": active_days,
                "gross_turnover_usd": float(fills.filled_gross_usd.sum()) if len(fills) else 0.0,
                "fee_equivalent_usd": float(fills.fee_equivalent_usd.sum()) if len(fills) else 0.0,
                "gross_pnl_usd": float(fills.gross_pnl_usd.sum()) if len(fills) else 0.0,
                "realized_fee_effect_usd": float(fills.realized_fee_effect_usd.sum()) if len(fills) else 0.0,
                "expected_net_t59_usd": float(fills.expected_net_t59_usd.sum()) if len(fills) else 0.0,
                "realized_net_pnl_usd": float(fills.net_pnl_usd.sum()) if len(fills) else 0.0,
                "expected_minus_realized_usd": float((fills.expected_net_t59_usd - fills.net_pnl_usd).sum()) if len(fills) else 0.0,
                "expected_minus_realized_per_fill_usd": float((fills.expected_net_t59_usd - fills.net_pnl_usd).mean()) if len(fills) else None,
                "net_per_turnover": float(fills.net_pnl_usd.sum() / fills.filled_gross_usd.sum()) if len(fills) and fills.filled_gross_usd.sum() else None,
                "positive_fills": int(fills.net_pnl_usd.gt(0).sum()) if len(fills) else 0,
                "negative_fills": int(fills.net_pnl_usd.lt(0).sum()) if len(fills) else 0,
                "top_3_absolute_pnl_day_share": float(daily_abs.head(3).sum() / absolute_total) if absolute_total else 0.0,
                "top_3_positive_pnl_day_share": float(daily_positive.head(3).sum() / positive_total) if positive_total else 0.0,
                **uncertainty,
            }
            if "outer_fold_id" in frame:
                for fold_id in frame.outer_fold_id.dropna().unique():
                    fold_fills = fills.loc[fills.outer_fold_id.eq(fold_id)]
                    fold_days = int(fold_fills.day_utc.nunique()) if len(fold_fills) else 0
                    fold_pnl = float(fold_fills.net_pnl_usd.sum()) if len(fold_fills) else 0.0
                    fold_rows.append({
                        "forecast": forecast, "ev_hurdle_fraction": float(hurdle),
                        "outer_fold_id": fold_id,
                        "opportunity_markets": int(frame.outer_fold_id.eq(fold_id).sum()),
                        "signals": int(selected.outer_fold_id.eq(fold_id).sum()),
                        "no_trade_markets": int(
                            frame.outer_fold_id.eq(fold_id).sum() - selected.outer_fold_id.eq(fold_id).sum()
                        ),
                        "fills": int(len(fold_fills)), "active_utc_days": fold_days,
                        "gross_turnover_usd": float(fold_fills.filled_gross_usd.sum()) if len(fold_fills) else 0.0,
                        "fee_equivalent_usd": float(fold_fills.fee_equivalent_usd.sum()) if len(fold_fills) else 0.0,
                        "gross_pnl_usd": float(fold_fills.gross_pnl_usd.sum()) if len(fold_fills) else 0.0,
                        "realized_fee_effect_usd": float(fold_fills.realized_fee_effect_usd.sum()) if len(fold_fills) else 0.0,
                        "expected_net_t59_usd": float(fold_fills.expected_net_t59_usd.sum()) if len(fold_fills) else 0.0,
                        "realized_net_pnl_usd": fold_pnl,
                        "expected_minus_realized_usd": float((fold_fills.expected_net_t59_usd - fold_fills.net_pnl_usd).sum()) if len(fold_fills) else 0.0,
                    })
            rows.append(row)
            for month in frame.month_utc.dropna().sort_values().unique():
                month_signals = selected.loc[selected.month_utc.eq(month)]
                month_fills = fills.loc[fills.month_utc.eq(month)]
                monthly_rows.append({
                    "period": row["period"], "month_utc": month,
                    "forecast": forecast, "ev_hurdle_fraction": float(hurdle),
                    "opportunity_markets": int(frame.month_utc.eq(month).sum()),
                    "signals": int(len(month_signals)), "fills": int(len(month_fills)),
                    "no_trade_markets": int(frame.month_utc.eq(month).sum() - len(month_signals)),
                    "active_utc_days": int(month_fills.day_utc.nunique()) if len(month_fills) else 0,
                    "gross_turnover_usd": float(month_fills.filled_gross_usd.sum()) if len(month_fills) else 0.0,
                    "fee_equivalent_usd": float(month_fills.fee_equivalent_usd.sum()) if len(month_fills) else 0.0,
                    "gross_pnl_usd": float(month_fills.gross_pnl_usd.sum()) if len(month_fills) else 0.0,
                    "realized_fee_effect_usd": float(month_fills.realized_fee_effect_usd.sum()) if len(month_fills) else 0.0,
                    "expected_net_t59_usd": float(month_fills.expected_net_t59_usd.sum()) if len(month_fills) else 0.0,
                    "realized_net_pnl_usd": float(month_fills.net_pnl_usd.sum()) if len(month_fills) else 0.0,
                })
    return rows, fold_rows, monthly_rows


def _select_policy(candidates: list[dict], config: dict) -> dict:
    rule = config["independent_order_economics"]["policy_selection_rule"]
    fold_rows = config["development_outer_folds"]
    fold_ids = [fold["fold_id"] for fold in fold_rows]
    qualifying = []
    for candidate in candidates:
        details = candidate.get("fold_results", [])
        by_fold = {row["outer_fold_id"]: row for row in details}
        stable = all(
            by_fold.get(fold_id, {}).get("fills", 0) >= int(rule["minimum_fills_in_each_outer_fold"])
            and by_fold.get(fold_id, {}).get("realized_net_pnl_usd", 0.0) > 0.0
            for fold_id in fold_ids
        )
        if (
            candidate["fills"] >= int(rule["minimum_fills"])
            and candidate["active_utc_days"] >= int(rule["minimum_active_utc_days"])
            and stable
            and candidate["total_pnl_95pct"][0] > 0.0
        ):
            qualifying.append(candidate)
    no_trade = {
        "policy": "no_trade", "forecast": None, "minimum_net_ev_fraction": None,
        "signals": 0, "fills": 0, "active_utc_days": 0,
        "realized_net_pnl_usd": 0.0, "net_per_turnover": None,
        "total_pnl_95pct": [0.0, 0.0], "reason": None,
    }
    selected = None
    if qualifying:
        selected = max(qualifying, key=lambda row: (
            row["total_pnl_95pct"][0], row["mean_daily_pnl"], row["active_utc_days"],
        ))
    else:
        no_trade["reason"] = "No active candidate passed all predeclared uncertainty, activity, trade-count and fold-stability requirements."
    return {
        "selection_metric": rule["ranking"],
        "predeclared_rule": rule,
        "qualifying_active_candidates": [row["policy"] for row in qualifying],
        "selected_policy": ({
            "policy": selected["policy"], "forecast": selected["forecast"],
            "minimum_net_ev_fraction": selected["ev_hurdle_fraction"],
            "fills": selected["fills"], "active_utc_days": selected["active_utc_days"],
            "realized_net_pnl_usd": selected["realized_net_pnl_usd"],
            "total_pnl_95pct": selected["total_pnl_95pct"],
            "reason": "Passed the predeclared active-policy gate.",
        } if selected else no_trade),
        "no_trade_control": no_trade,
        "holdout_used_for_selection": False,
    }


def _fixed_execution_comparison(frame: pd.DataFrame, details: pd.DataFrame, period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_by_id = frame.set_index(frame.condition_id.astype(str).str.lower())
    output = []
    pair_rows = []
    for selector in ("A_BTC_calibrated", "D_market_plus_BTC"):
        selected = details.loc[
            details.forecast.eq(selector) & details.execution_status.eq("filled")
        ].copy()
        if selected.empty:
            continue
        for forecast in MODEL_NAMES:
            probabilities = []
            outcomes = []
            expected = []
            realized = []
            for item in selected.itertuples(index=False):
                source = prediction_by_id.loc[item.condition_id]
                p_up = float(source[PROBABILITY_COLUMNS[forecast]])
                p_side = p_up if item.chosen_side == "up" else 1.0 - p_up
                probabilities.append(p_side)
                outcomes.append(int(item.outcome_side))
                expected.append(p_side * float(item.net_shares) - float(item.cash_debit_usd))
                realized.append(float(item.net_pnl_usd))
                pair_rows.append({
                    "period": period, "basket_selector": selector,
                    "scored_forecast": forecast, "condition_id": item.condition_id,
                    "market_start_utc": item.market_start_utc, "chosen_side": item.chosen_side,
                    "p_side": p_side, "outcome_side": int(item.outcome_side),
                    "T59_execution_source": item.execution_source,
                    "depth_known": item.depth_known, "filled_gross_usd": item.filled_gross_usd,
                    "vwap": item.vwap, "fee_equivalent_usd": item.fee_equivalent_usd,
                    "net_shares": item.net_shares, "cash_debit_usd": item.cash_debit_usd,
                    "expected_net_t59_usd": expected[-1], "realized_net_pnl_usd": realized[-1],
                })
            y = np.asarray(outcomes, dtype=np.int8)
            p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
            cal = calibration_metrics(y, p)
            output.append({
                "period": period, "basket_selector": selector,
                "scored_forecast": forecast, "filled_market_side_pairs": int(len(selected)),
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "brier": float(brier_score_loss(y, p)),
                "mean_predicted_side_probability": float(p.mean()),
                "realized_side_win_rate": float(y.mean()),
                "calibration_intercept": cal.get("intercept"),
                "calibration_slope": cal.get("slope"),
                "expected_net_t59_usd": float(np.sum(expected)),
                "realized_net_pnl_usd": float(np.sum(realized)),
                "expected_minus_realized_usd": float(np.sum(expected) - np.sum(realized)),
                "expected_minus_realized_per_fill_usd": float(np.mean(np.asarray(expected) - np.asarray(realized))),
                "expected_minus_realized_95pct": _block_interval(
                    pd.Series(
                        np.asarray(expected, dtype=np.float64) - np.asarray(realized, dtype=np.float64),
                        index=pd.DatetimeIndex(selected.day_utc),
                    ).groupby(level=0).sum().reindex(
                        pd.date_range(pd.Timestamp(selected.day_utc.min()), pd.Timestamp(selected.day_utc.max()), freq="D"),
                        fill_value=0.0,
                    ),
                    20261012,
                )["total_pnl_95pct"],
            })
        basket_rows = [row for row in output if row["period"] == period and row["basket_selector"] == selector]
        if len({round(row["realized_net_pnl_usd"], 10) for row in basket_rows}) != 1:
            raise AssertionError(f"Rescoring changed realized payouts in {period}/{selector} basket")
    return pd.DataFrame(output), pd.DataFrame(pair_rows)


def _own_selected_trade_quality(details: pd.DataFrame, hurdles: tuple[float, ...]) -> list[dict]:
    rows = []
    for forecast in MODEL_NAMES:
        model_details = details.loc[
            details.forecast.eq(forecast) & details.execution_status.eq("filled")
        ]
        for hurdle in hurdles:
            selected = model_details.loc[model_details.t60_ev_fraction.ge(hurdle - 1e-12)]
            if selected.empty:
                rows.append({
                    "period": str(details.period.iloc[0]), "forecast": forecast,
                    "ev_hurdle_fraction": float(hurdle), "fills": 0,
                    "log_loss": None, "brier": None, "mean_predicted_side_probability": None,
                    "realized_side_win_rate": None,
                })
                continue
            y = selected.outcome_side.to_numpy(dtype=np.int8)
            p = np.clip(selected.p_side.to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
            rows.append({
                "period": str(details.period.iloc[0]), "forecast": forecast,
                "ev_hurdle_fraction": float(hurdle), "fills": int(len(selected)),
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "brier": float(brier_score_loss(y, p)),
                "mean_predicted_side_probability": float(p.mean()),
                "realized_side_win_rate": float(y.mean()),
            })
    return rows


def _coefficient_stability(choice_rows: list[dict]) -> list[dict]:
    result = []
    fields = {
        "C": ("intercept", "beta_logit_market"),
        "D": ("intercept", "beta_logit_market", "beta_logit_btc_calibrated"),
        "E": ("mixture_weight_w",),
        "F": ("offset_a", "btc_difference_b"),
    }
    for family, columns in fields.items():
        rows = [row for row in choice_rows if row["family"] == family]
        for column in columns:
            values = np.asarray([row[column] for row in rows if row.get(column) is not None], dtype=np.float64)
            if len(values):
                result.append({
                    "family": family, "parameter": column,
                    "mean": float(values.mean()), "sd": float(values.std(ddof=0)),
                    "min": float(values.min()), "max": float(values.max()),
                    "folds": int(len(values)),
                    "same_sign_folds": int(np.sum(np.sign(values) == np.sign(values[0]))),
                })
    return result


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _fmt(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_money(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${float(value):,.{digits}f}"


def _build_verdict(payload: dict) -> str:
    development = [row for row in payload["quality"] if row["period"] == "development_nested_oof"]
    best = min(development, key=lambda row: row["log_loss"])
    paired = payload["paired_differences"]["development_nested_oof"]
    policy = payload["frozen_selection"]["policy_selection"]["selected_policy"]
    return (
        f"Nested May-June OOF ({payload['cv_rows']:,} markets) ranks {best['forecast']} first by log-loss ({best['log_loss']:.5f}). "
        f"E-B={paired['E-B']['log_loss_difference_left_minus_right']:+.5f} "
        f"[{paired['E-B']['log_loss_difference_95pct'][0]:+.5f}, {paired['E-B']['log_loss_difference_95pct'][1]:+.5f}], "
        f"E-D={paired['E-D']['log_loss_difference_left_minus_right']:+.5f} "
        f"[{paired['E-D']['log_loss_difference_95pct'][0]:+.5f}, {paired['E-D']['log_loss_difference_95pct'][1]:+.5f}]. "
        f"Frozen policy: {policy['policy']}. July-October is historical diagnostics only."
    )


def _build_answers(payload: dict) -> str:
    quality = {(row["period"], row["forecast"]): row for row in payload["quality"]}
    dev = payload["paired_differences"]["development_nested_oof"]
    hold = payload["paired_differences"]["holdout_diagnostic"]
    model_a = quality[("development_nested_oof", "A_BTC_calibrated")]
    market_b = quality[("development_nested_oof", "B_market_reference")]
    model_c = quality[("development_nested_oof", "C_market_logistic")]
    model_d = quality[("development_nested_oof", "D_market_plus_BTC")]
    model_e = quality[("development_nested_oof", "E_probability_mixture")]
    brier_gap = dev["E-B"]["brier_difference_left_minus_right"]
    hold_b = quality[("holdout_diagnostic", "B_market_reference")]
    hold_e = quality[("holdout_diagnostic", "E_probability_mixture")]
    final_choices = payload["final_choices"]
    fixed = {
        (row["period"], row["basket_selector"], row["scored_forecast"]): row
        for row in payload["fixed_baskets"]
    }
    a_basket_d = fixed[("development_oof", "A_BTC_calibrated", "D_market_plus_BTC")]
    a_basket_e = fixed[("development_oof", "A_BTC_calibrated", "E_probability_mixture")]
    d_basket_d = fixed[("development_oof", "D_market_plus_BTC", "D_market_plus_BTC")]
    d_basket_e = fixed[("development_oof", "D_market_plus_BTC", "E_probability_mixture")]
    dev_economics = {
        (row["forecast"], float(row["ev_hurdle_fraction"])): row
        for row in payload["development_economics"]
    }
    e_policy = dev_economics[("E_probability_mixture", 0.01)]
    selected = payload["frozen_selection"]["policy_selection"]["selected_policy"]
    selected_economics = dev_economics.get(
        (selected.get("forecast"), float(selected.get("minimum_net_ev_fraction") or 0.0)),
    ) if selected.get("forecast") else None
    portfolio = payload["holdout_portfolio"]
    e_ci = dev["E-B"]["log_loss_difference_95pct"]
    e_d_ci = dev["E-D"]["log_loss_difference_95pct"]
    e_brier_ci = dev["E-B"]["brier_difference_95pct"]
    a_pnl_ci = selected_economics["total_pnl_95pct"] if selected_economics else [0.0, 0.0]
    fold_pnl = selected_economics.get("fold_pnl", {}) if selected_economics else {}
    return "\n".join([
        f"1. W rozwoju rynek B ma log-loss {market_b['log_loss']:.5f} wobec {model_a['log_loss']:.5f} dla BTC A; Brier to {market_b['brier']:.5f} wobec {model_a['brier']:.5f}. B ma niższy LL i ECE niż C ({dev['C-B']['log_loss_difference_left_minus_right']:+.5f} dla C−B; ECE C={model_c['ece_frozen_dev_bins']:.4f}) oraz niższe ECE niż D ({model_d['ece_frozen_dev_bins']:.4f}), ale D ma nieco niższy LL ({dev['D-B']['log_loss_difference_left_minus_right']:+.5f} dla D−B). A ma niższy LL niż B. E ma punktowo log-loss {model_e['log_loss']:.5f} i Brier {model_e['brier']:.5f}; E-B ΔLL {dev['E-B']['log_loss_difference_left_minus_right']:+.5f} [{e_ci[0]:+.5f}, {e_ci[1]:+.5f}] oraz ΔBrier {brier_gap:+.5f} [{e_brier_ci[0]:+.5f}, {e_brier_ci[1]:+.5f}]. Przedziały obejmują zero, a E ECE {model_e['ece_frozen_dev_bins']:.4f} jest gorsze od B. Zakotwiczenie rynkowe poprawia ECE względem pozostałych prognoz, ale nie daje stabilnego zysku w LL; efekt BTC również nie jest stabilny.",
        f"2. Nie ma wiarygodnego dowodu, że BTC daje poprawę ponad rynek: E-B CI obejmuje zero; E-D ΔLL {dev['E-D']['log_loss_difference_left_minus_right']:+.5f} [{e_d_ci[0]:+.5f}, {e_d_ci[1]:+.5f}] też obejmuje zero. F-B ma ΔLL {dev['F-B']['log_loss_difference_left_minus_right']:+.5f} [{dev['F-B']['log_loss_difference_95pct'][0]:+.5f}, {dev['F-B']['log_loss_difference_95pct'][1]:+.5f}], a końcowy wybór F to {final_choices['F']['variant']} (dokładnie B, bez korekty). Końcowy wybór E to {final_choices['E']['variant']} (dokładnie A, sama prognoza BTC). Brak podstaw do wyboru aktywnej korekty E/F; no-trade pozostaje kontrolą dla korekty ponad rynek.",
        f"3. Na koszyku wykonań D korekta E poprawia punktowo log-loss {d_basket_e['log_loss']:.5f} vs {d_basket_d['log_loss']:.5f}, a lukę EV-PnL zmienia z {_fmt_money(d_basket_d['expected_minus_realized_usd'])} do {_fmt_money(d_basket_e['expected_minus_realized_usd'])}; oba 95% CI luki obejmują zero. Na stałym koszyku A E jest punktowo gorsze od D ({a_basket_e['log_loss']:.5f} vs {a_basket_d['log_loss']:.5f}). Poprawa nie powtarza się na obu zestawach wykonań.",
        f"4. E przy hurdle 1% ma development PnL {_fmt_money(e_policy['realized_net_pnl_usd'])} po fee na {e_policy['fills']:,} fillach, ale 95% blokowe CI [{_fmt_money(e_policy['total_pnl_95pct'][0])}, {_fmt_money(e_policy['total_pnl_95pct'][1])}] obejmuje zero. Koszty są wliczone z rzeczywistego T-59. Wybrana development polityka to {selected['policy']}; na stałych $5 ma {_fmt_money(selected_economics['realized_net_pnl_usd'] if selected_economics else 0.0)} (CI [{_fmt_money(a_pnl_ci[0])}, {_fmt_money(a_pnl_ci[1])}]), ale jest to BTC-only A, a przedział nie koryguje selekcji spośród modeli i progów. Portfel 5% wolnej gotówki na historycznym holdoucie wykonał {portfolio['trade_count']} zlecenia i stracił {_fmt_money(abs(portfolio['net_pnl_usd']))}; wynik ten nie potwierdza poprawki E/F.",
        f"5. Wybrana A polityka w development miała {selected['fills']:,} filli w {selected['active_utc_days']} dniach; PnL foldów: maj {_fmt_money(fold_pnl.get('dev_2026q2_b', 0.0))}, czerwiec {_fmt_money(fold_pnl.get('dev_2026q2_c', 0.0))}. Trzy największe dni to {selected_economics['top_3_absolute_pnl_day_share']:.1%} bezwzględnego wyniku. Zatem development wynik A rozkłada się na wiele dni, ale holdout feasibility ograniczyło minimum do {portfolio['trade_count']} filli; po tym nie podbijano stawki. Holdout B ma LL {hold_b['log_loss']:.5f}, a E {hold_e['log_loss']:.5f}; to historyczna diagnostyka, bo okres był wcześniej oglądany.",
    ])


def _copy_summary_tables() -> None:
    filenames = (
        "nested_parameter_selection.csv", "outer_fold_model_choices.csv",
        "coefficient_stability.csv", "probability_quality.csv",
        "outer_fold_probability_quality.csv", "monthly_holdout_probability_quality.csv",
        "calibration_bins.csv", "paired_probability_differences.csv",
        "independent_order_economics.csv", "monthly_independent_order_economics.csv",
        "development_policy_fold_stability.csv", "own_selected_trade_probability_quality.csv",
        "fixed_execution_forecast_comparison.csv",
    )
    SUMMARY_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        shutil.copyfile(OUTPUT_DIR / filename, SUMMARY_TABLE_DIR / filename)


def _refresh_saved_report() -> dict:
    payload_path = OUTPUT_DIR / "report_payload.json"
    payload = research._read_json(payload_path)
    cv = pd.read_parquet(OUTPUT_DIR / "development_nested_oof_predictions.parquet")
    holdout = pd.read_parquet(OUTPUT_DIR / "holdout_diagnostic_predictions.parquet")
    payload["paired_differences"] = {
        "development_nested_oof": _paired_comparisons(cv),
        "holdout_diagnostic": _paired_comparisons(holdout),
    }
    payload["fold_quality"] = pd.read_csv(OUTPUT_DIR / "outer_fold_probability_quality.csv").to_dict("records")
    payload["monthly_quality"] = pd.read_csv(OUTPUT_DIR / "monthly_holdout_probability_quality.csv").to_dict("records")
    payload["verdict"] = _build_verdict(payload)
    payload["answers"] = _build_answers(payload)
    pd.DataFrame([
        {"period": period, "comparison": name, **row}
        for period, comparisons in payload["paired_differences"].items() for name, row in comparisons.items()
    ]).to_csv(OUTPUT_DIR / "paired_probability_differences.csv", index=False)
    _write_json(payload_path, payload)
    REPORT_PATH.write_text(_render_report(payload), encoding="utf-8")
    _copy_summary_tables()
    manifest_path = OUTPUT_DIR / "experiment_manifest.json"
    manifest = research._read_json(manifest_path)
    manifest["paired_probability_differences"] = payload["paired_differences"]
    manifest["answers"] = payload["answers"]
    manifest["evaluation_script_sha256"] = research._sha256(Path(__file__))
    manifest["report_path"] = REPORT_PATH.relative_to(ROOT).as_posix()
    _write_json(manifest_path, manifest)
    return {"verdict": payload["verdict"], "report_path": str(REPORT_PATH), "summary_table_dir": str(SUMMARY_TABLE_DIR)}


def _render_report(payload: dict) -> str:
    lines = [
        "# BTC i rynek: zagnieżdżony test korekt prawdopodobieństwa",
        "",
        "> Nowy eksperyment selekcji z 12 października 2026. Zmiana procedury wyboru modeli i polityki; nie poprawka księgowania.",
        "> Holdout lipiec–październik był wcześniej oglądany. Jego wyniki są historyczną diagnostyką, nie niezależnym potwierdzeniem.",
        "",
        "## Werdykt",
        "",
        payload["verdict"],
        "",
        "## Zamrożona procedura i pokrycie",
        "",
        f"Źródło wejściowe zawiera {payload['source_identity']['prediction_identity']['prediction_rows']:,} zapisanych predykcji BTC; eksperyment użył {payload['cv_rows']:,} wspólnych rynków OOF w rozwoju i {payload['holdout_rows']:,} rynków holdout. Wspólna próba wymaga oficjalnego wyniku, przyczynowej skalibrowanej prognozy BTC oraz poprawnych obu stron T−60. T−59 służy wyłącznie do wyceny wykonania.",
        "",
        "Jedynym wejściem BTC dla E/F jest istniejące skalibrowane p_BTC. Nie trenowano ani nie refitowano głównego modelu BTC. E używa `p=(1-w)*p_market+w*p_BTC` z w=[0, 0.1, 0.25, 0.5, 1]. F używa offsetu rynkowego `logit(p)=logit(p_market)+a+b*(logit(p_BTC)-logit(p_market))`, z `lambda*(a²+b²)/2` dodanym do średniej log-loss; lambda=[0.1, 1, 10]. Obie rodziny zawierają dokładną opcję bez korekty równą p_market.",
        "",
        "C i D pozostały kontrolami. W zewnętrznym foldu majowym parametry dopasowano tylko na wcześniejszych prefiksach kwietnia; w czerwcowym tylko na wcześniejszych prefiksach do maja. W każdym wewnętrznym foldzie trening odcina etykiety z `label_available_at_utc` późniejszym niż początek walidacji. OOF obejmuje wyłącznie maj i czerwiec.",
        "",
        "| Zewnętrzny fold | Prefiks treningowy | Walidacja wewnętrzna | Rynki outer train | Rynki outer valid |",
        "|---|---|---|---:|---:|",
    ]
    for fold in payload["outer_folds"]:
        inner = "; ".join(f"{item['start']}–{item['end']} ({item['valid_rows']:,})" for item in fold["inner_folds"])
        lines.append(f"| {fold['fold_id']} | < {fold['start']} z purge etykiet | {inner} | {fold['train_rows']:,} | {fold['valid_rows']:,} |")
    lines += [
        "",
        "Dobór końcowych parametrów do zamrożonej oceny holdout używa odrębnie zadeklarowanych foldów majowym i czerwcowym, wyłącznie przed cutoffem 1 lipca. Wyników tych foldów nie użyto do doboru parametrów predykcji outer OOF z tych samych miesięcy.",
        "",
        "### Parametry wybrane dla końcowych modeli holdout",
        "",
        "| Rodzina | Wariant | Parametr | Wartość | Log-loss wewnętrzny | n walidacji |",
        "|---|---|---|---:|---:|---:|",
    ]
    for family, choice in payload["final_choices"].items():
        lines.append(f"| {family} | {choice['variant']} | {choice['parameter']} | {_fmt(choice['value'], 4)} | {choice['inner_log_loss']:.5f} | {choice['inner_rows']:,} |")
    lines += [
        "",
        "## Prawdopodobieństwa na tych samych rynkach",
        "",
        "Dla C–F development prognozy pochodzą wyłącznie z outer OOF. Holdout pokazuje modele z parametrami zamrożonymi przed jego oceną. Ujemna sparowana różnica log-loss/Brier sprzyja pierwszemu modelowi; bootstrap używa wspólnych ruchomych bloków 3 dni UTC.",
        "",
        "| Okres | Model | n | Log-loss | Brier | AUC | Intercept kalibracji | Slope | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["quality"]:
        lines.append(f"| {row['period']} | {row['forecast']} | {row['n']:,} | {row['log_loss']:.5f} | {row['brier']:.5f} | {_fmt(row['auc'], 4)} | {_fmt(row['calibration_intercept'], 3)} | {_fmt(row['calibration_slope'], 3)} | {_fmt(row['ece_frozen_dev_bins'], 4)} |")
    lines += [
        "",
        "### Jakość OOF w zewnętrznych foldach",
        "",
        "| Fold | Model | n | Log-loss | Brier | Slope | ECE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["fold_quality"]:
        lines.append(f"| {row['period']} | {row['forecast']} | {row['n']:,} | {row['log_loss']:.5f} | {row['brier']:.5f} | {_fmt(row['calibration_slope'], 3)} | {_fmt(row['ece_frozen_dev_bins'], 4)} |")
    lines += [
        "",
        "| Okres | Porównanie | Δ log-loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |",
        "|---|---|---:|---:|---:|",
    ]
    for period, comparisons in payload["paired_differences"].items():
        for name, row in comparisons.items():
            ll = row["log_loss_difference_95pct"]
            br = row["brier_difference_95pct"]
            auc = row["auc_difference_95pct"]
            lines.append(f"| {period} | {name} | {row['log_loss_difference_left_minus_right']:+.5f} [{ll[0]:+.5f}, {ll[1]:+.5f}] | {row['brier_difference_left_minus_right']:+.5f} [{br[0]:+.5f}, {br[1]:+.5f}] | {row['auc_difference_left_minus_right']:+.4f} [{auc[0]:+.4f}, {auc[1]:+.4f}] |")
    lines += [
        "",
        "### Jakość prawdopodobieństw holdoutu według miesiąca",
        "",
        "| Miesiąc UTC | Model | n | Log-loss | Brier | AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["monthly_quality"]:
        lines.append(f"| {row['month_utc']} | {row['forecast']} | {row['n']:,} | {row['log_loss']:.5f} | {row['brier']:.5f} | {_fmt(row['auc'], 4)} |")
    lines += [
        "",
        "## OOF zlecenia niezależne po $5",
        "",
        "Wszystkie sygnały używają strony wybranej z T−60 i rzeczywistego T−59 do wykonania. Zlecenia nie dzielą gotówki. Próg to dotychczasowa siatka 0%, 0.25%, 0.5%, 1% i 2%. Przedziały są blokowym bootstrapem 3-dniowym dla łącznego PnL po fee. Koncentracja pokazuje udział trzech dni o największym bezwzględnym PnL.",
        "",
        "| Model | Próg EV | Sygnały | Fille | Aktywne dni | Fee eq. | Oczekiwane EV T−59 | PnL netto | CI 95% PnL | Net/obrót | Top 3 dni | Maj / czerwiec PnL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["development_economics"]:
        fold_pnl = row["fold_pnl"]
        ci = row["total_pnl_95pct"]
        lines.append(f"| {row['forecast']} | {row['ev_hurdle_fraction']:.2%} | {row['signals']:,} | {row['fills']:,} | {row['active_utc_days']:,} | {_fmt_money(row['fee_equivalent_usd'])} | {_fmt_money(row['expected_net_t59_usd'])} | {_fmt_money(row['realized_net_pnl_usd'])} | [{_fmt_money(ci[0])}, {_fmt_money(ci[1])}] | {_fmt(row['net_per_turnover'] * 100 if row['net_per_turnover'] is not None else None, 2)}% | {row['top_3_absolute_pnl_day_share']:.1%} | {_fmt_money(fold_pnl.get('dev_2026q2_b', 0.0))} / {_fmt_money(fold_pnl.get('dev_2026q2_c', 0.0))} |")
    selection_record = payload["frozen_selection"].get("policy_selection", payload["frozen_selection"])
    selected_policy = selection_record["selected_policy"]
    lines += [
        "",
        "Kryterium aktywnego wyboru było zadeklarowane przed wynikiem: co najmniej 50 filli, 15 aktywnych dni, co najmniej 10 filli i dodatni PnL w każdym z dwóch foldów oraz dolna granica 95% blokowego CI całkowitego PnL powyżej zera. Przedziały dotyczą stałego kandydata i nie korygują wielokrotnego porównania modeli/progów.",
        "",
        f"Zamrożona polityka: `{selected_policy['policy']}`. {selected_policy.get('reason', '')}",
        f"Kontrola no-trade: {selection_record['no_trade_control']['fills']} fille, PnL {_fmt_money(selection_record['no_trade_control']['realized_net_pnl_usd'])}, CI [{_fmt_money(selection_record['no_trade_control']['total_pnl_95pct'][0])}, {_fmt_money(selection_record['no_trade_control']['total_pnl_95pct'][1])}].",
        "",
        "## Prognozy na tych samych wykonaniach",
        "",
        "W każdym koszyku ustalono rynek i stronę przez A albo D, a następnie zachowano te same fille T−59, fee, liczbę udziałów, debet i wypłatę. Zmieniają się tylko prognozowane p(side), log-loss/Brier i oczekiwane EV.",
        "",
        "| Okres | Koszyk wybrany przez | Prognoza | Fille | Log-loss | Brier | Śr. p(side) / win rate | Oczekiwane EV T−59 | Zrealizowany PnL | EV−PnL |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["fixed_baskets"]:
        gap_ci = row["expected_minus_realized_95pct"]
        lines.append(f"| {row['period']} | {row['basket_selector']} | {row['scored_forecast']} | {row['filled_market_side_pairs']:,} | {row['log_loss']:.5f} | {row['brier']:.5f} | {row['mean_predicted_side_probability']:.3f} / {row['realized_side_win_rate']:.3f} | {_fmt_money(row['expected_net_t59_usd'])} | {_fmt_money(row['realized_net_pnl_usd'])} | {_fmt_money(row['expected_minus_realized_usd'])} [{_fmt_money(gap_ci[0])}, {_fmt_money(gap_ci[1])}] |")
    lines += [
        "",
        "### Jakość własnych selekcji dodatniego EV",
        "",
        "| Okres | Model wybierający | Próg EV | Fille | Log-loss side | Brier side | Śr. p(side) / win rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["own_selected_trade_quality"]:
        if row["ev_hurdle_fraction"] in (0.0, 0.005, 0.01):
            mean_probability = _fmt(row["mean_predicted_side_probability"], 3)
            win_rate = _fmt(row["realized_side_win_rate"], 3)
            lines.append(f"| {row['period']} | {row['forecast']} | {row['ev_hurdle_fraction']:.2%} | {row['fills']:,} | {_fmt(row['log_loss'], 5)} | {_fmt(row['brier'], 5)} | {mean_probability} / {win_rate} |")
    lines += [
        "",
        "## Holdout: diagnostyka po zamrożeniu",
        "",
        "| Model | Próg EV | Sygnały | Fille | Aktywne dni | Oczekiwane EV T−59 | PnL netto | Fee eq. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["holdout_economics"]:
        lines.append(f"| {row['forecast']} | {row['ev_hurdle_fraction']:.2%} | {row['signals']:,} | {row['fills']:,} | {row['active_utc_days']:,} | {_fmt_money(row['expected_net_t59_usd'])} | {_fmt_money(row['realized_net_pnl_usd'])} | {_fmt_money(row['fee_equivalent_usd'])} |")
    lines += [
        "",
        "### PnL miesięczny holdout przy hurdle 0%",
        "",
        "| Miesiąc UTC | Model | Fille | Aktywne dni | Oczekiwane EV T−59 | PnL netto |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["holdout_monthly_economics"]:
        if row["ev_hurdle_fraction"] == 0.0:
            lines.append(f"| {row['month_utc']} | {row['forecast']} | {row['fills']:,} | {row['active_utc_days']:,} | {_fmt_money(row['expected_net_t59_usd'])} | {_fmt_money(row['realized_net_pnl_usd'])} |")
    portfolio = payload["holdout_portfolio"]
    lines += [
        "",
        "### Wykonalność zamrożonej polityki przy 5% wolnej gotówki",
        "",
        f"Kapitał początkowy $100; stake 5% wolnej gotówki, minimum obowiązuje, bez podbijania stawki. Fille {portfolio['trade_count']:,}; obrót {_fmt_money(portfolio['gross_turnover_usd'])}; fee {_fmt_money(portfolio['fees_paid_estimated_usd'])}; PnL {_fmt_money(portfolio['net_pnl_usd'])}; saldo końcowe {_fmt_money(portfolio['ending_cash_after_settlement_usd'])}; max DD {portfolio['max_drawdown_cost_basis_equity']:.2%}; audyt `{portfolio['accounting_audit']['status']}`.",
        "",
        "## Odpowiedzi",
        "",
        payload["answers"],
        "",
        f"Skrócone tabele CSV: [`{SUMMARY_TABLE_DIR.relative_to(REPORT_PATH.parent).as_posix()}/`](btc_market_probability_nested_20261012/). Pełne predykcje i ledgery: `{OUTPUT_DIR.relative_to(ROOT).as_posix()}`. Konfigurację i freeze development zapisano przed metrykami holdout. Model BTC nie był trenowany, replay księgi nie był powtarzany, a konfiguracja live nie była zmieniana.",
        "",
    ]
    return "\n".join(lines)


def main() -> dict:
    started = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_holdout_scoring":
        raise RuntimeError("Nested probability experiment rules are not frozen")
    fee_correction._configure_output_paths()
    _, _, source_holdout = fee_correction._verify_reusable_sources()
    correction_manifest = research._read_json(fee_correction.CORRECTION_DIR / "research_manifest.json")
    predictions, prediction_identity = research._load_model_predictions(include_holdout=True)
    t59, t60, book_record = research._load_enriched_book_snapshots(correction_manifest)
    calendar, _ = research._build_market_calendar(predictions)
    calendar["market_start_utc"] = pd.to_datetime(calendar.market_start_utc, utc=True, errors="raise")
    data = previous._build_dataset(calendar, t60)
    dev = data.loc[
        data.common_sample & data.market_start_utc.lt(DEV_CUTOFF_UTC)
    ].copy()
    if dev.empty:
        raise RuntimeError("No common-quote development sample remains")
    print(f"[nested-probability] common development rows={len(dev):,}; predictions={len(predictions):,}; T60={len(t60):,}; T59={len(t59):,}", flush=True)

    cv, outer_choices, outer_grid = _outer_predictions(dev, config)
    final_choices, final_grid, final_models = _final_development_choices(dev, config)
    final_train = dev.loc[dev.label_available_at_utc.le(DEV_CUTOFF_UTC)]

    def add_family_predictions(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["p_A"] = result.p_btc_calibrated.astype(float)
        result["p_B"] = result.p_market.astype(float)
        for family in META_FAMILIES:
            result[f"p_{family}"] = _predict_spec(
                result, family, _choice_for_fit(final_choices[family]), final_models[family],
            )
        return result

    # Outer OOF rows already have nested C-F probabilities from their earlier prefixes.
    cv["p_A"] = cv.p_btc_calibrated.astype(float)
    cv["p_B"] = cv.p_market.astype(float)
    hurdles = tuple(float(value) for value in config["independent_order_economics"]["ev_thresholds"])
    t59_by_id = research._snapshot_map(t59)
    t60_by_id = research._snapshot_map(t60)
    market_by_id = research._snapshot_map(calendar.loc[calendar.condition_id.notna()])
    source_identity = {
        "walk_forward_manifest_sha256": research._sha256(fee_correction.SOURCE_WALK_FORWARD_MANIFEST),
        "prediction_identity": prediction_identity,
        "holdout_prediction_sha256": source_holdout["predictions_sha256"],
        "evaluated_bundle_sha256": source_holdout["evaluated_bundle_sha256"],
        "t60_snapshot_sha256": book_record["t60_coverage_sha256"],
        "t59_snapshot_sha256": book_record["t59_coverage_sha256"],
    }

    cv_quality, cv_bins, fold_quality = [], [], []
    for forecast in MODEL_NAMES:
        column = PROBABILITY_COLUMNS[forecast]
        edges = previous._metric_edges(cv[column].to_numpy())
        row, bins = _score_metrics(cv, column, forecast, "development_nested_oof", edges)
        cv_quality.append(row)
        cv_bins.extend(bins)
        for fold_id, fold_data in cv.groupby("outer_fold_id", sort=True):
            fold_row, _ = _score_metrics(fold_data, column, forecast, str(fold_id), edges)
            fold_quality.append(fold_row)

    cv_details = _economic_detail(cv, MODEL_NAMES, t59_by_id, t60_by_id, market_by_id, "development_oof")
    cv_economics, cv_fold_economics, cv_monthly_economics = _policy_summaries(cv_details, cv, hurdles, config)
    fold_by_policy: dict[tuple[str, float], list[dict]] = {}
    for row in cv_fold_economics:
        fold_by_policy.setdefault((row["forecast"], row["ev_hurdle_fraction"]), []).append(row)
    for row in cv_economics:
        row["fold_results"] = fold_by_policy.get((row["forecast"], row["ev_hurdle_fraction"]), [])
        row["fold_pnl"] = {item["outer_fold_id"]: item["realized_net_pnl_usd"] for item in row["fold_results"]}
    frozen_policy = _select_policy(cv_economics, config)
    for row in cv_economics:
        del row["fold_results"]
    model_ranking = sorted(cv_quality, key=lambda row: (row["log_loss"], row["forecast"]))
    frozen_development = {
        "status": "frozen_from_nested_development_before_holdout_scoring",
        "experiment_id": config["experiment_id"],
        "configuration_sha256": research._sha256(CONFIG_PATH),
        "selection_procedure_change": config["selection_procedure_change"],
        "development_data_identity": source_identity,
        "development_market_cutoff_utc": DEV_CUTOFF_UTC.isoformat(),
        "nested_oof_rows": int(len(cv)),
        "final_train_rows": int(len(final_train)),
        "final_choices": final_choices,
        "nested_oof_probability_ranking_by_log_loss": [
            {"forecast": row["forecast"], "log_loss": row["log_loss"], "brier": row["brier"], "n": row["n"]}
            for row in model_ranking
        ],
        "policy_selection": frozen_policy,
        "holdout_used_for_selection": False,
    }
    if FROZEN_SELECTION_PATH.exists():
        existing = research._read_json(FROZEN_SELECTION_PATH)
        if existing != frozen_development:
            raise RuntimeError("Existing frozen development selection does not match current inputs/configuration")
    else:
        _write_json(FROZEN_SELECTION_PATH, frozen_development)
    print(f"[nested-probability] frozen development policy={frozen_policy['selected_policy']['policy']}; holdout scoring starts now", flush=True)

    # Do not inspect holdout labels or metrics until the development freeze above is written.
    holdout = data.loc[
        data.common_sample & data.market_start_utc.ge(DEV_CUTOFF_UTC)
        & data.market_start_utc.lt(DATA_CUTOFF_UTC)
    ].copy()
    holdout = add_family_predictions(holdout)
    hold_quality, hold_bins, monthly_quality = [], [], []
    for forecast in MODEL_NAMES:
        column = PROBABILITY_COLUMNS[forecast]
        edges = previous._metric_edges(cv[column].to_numpy())
        row, bins = _score_metrics(holdout, column, forecast, "holdout_diagnostic", edges)
        hold_quality.append(row)
        cv_bins.extend(bins)
        for month, month_data in holdout.groupby("month_utc", sort=True):
            monthly, _ = _score_metrics(month_data, column, forecast, "holdout_diagnostic", edges)
            monthly["month_utc"] = month
            monthly_quality.append(monthly)
    paired = {
        "development_nested_oof": _paired_comparisons(cv),
        "holdout_diagnostic": _paired_comparisons(holdout),
    }
    hold_details = _economic_detail(holdout, MODEL_NAMES, t59_by_id, t60_by_id, market_by_id, "holdout_diagnostic")
    hold_economics, _, hold_monthly_economics = _policy_summaries(hold_details, holdout, hurdles, config)
    fixed_basket_rows, fixed_pair_rows = [], []
    for period, frame, details in (
        ("development_oof", cv, cv_details), ("holdout_diagnostic", holdout, hold_details),
    ):
        summary, pairs = _fixed_execution_comparison(frame, details, period)
        fixed_basket_rows.extend(summary.to_dict("records"))
        fixed_pair_rows.extend(pairs.to_dict("records"))

    selected_policy = frozen_policy["selected_policy"]
    if selected_policy["policy"] == "no_trade":
        holdout_portfolio = {
            "starting_capital_usd": 100.0, "ending_cash_after_settlement_usd": 100.0,
            "net_pnl_usd": 0.0, "trade_count": 0, "gross_turnover_usd": 0.0,
            "fees_paid_estimated_usd": 0.0, "max_drawdown_cost_basis_equity": 0.0,
            "accounting_audit": {"status": "no_trade"},
        }
    else:
        selected_forecast = selected_policy["forecast"]
        selected_column = PROBABILITY_COLUMNS[selected_forecast]
        portfolio_rows = holdout.copy()
        portfolio_rows["p_candidate_platt"] = portfolio_rows[selected_column]
        portfolio_market = {
            str(row.condition_id).lower(): {
                **market_by_id[str(row.condition_id).lower()],
                "p_candidate_platt": float(getattr(row, selected_column)),
            }
            for row in portfolio_rows.itertuples(index=False)
        }
        holdout_portfolio, _, _, _ = diagnostic._run_holdout_portfolio(
            100.0, portfolio_rows, t59_by_id, portfolio_market, t60_by_id,
            float(selected_policy["minimum_net_ev_fraction"]),
        )

    coverage = {
        "development_common_rows": int(len(dev)),
        "development_outer_oof_rows": int(len(cv)),
        "development_excluded_from_outer_oof_prefix_rows": int(len(dev) - len(cv)),
        "holdout_common_rows": int(len(holdout)),
        "holdout_start_min_utc": holdout.market_start_utc.min().isoformat(),
        "holdout_start_max_utc": holdout.market_start_utc.max().isoformat(),
        "t60_proxy_or_complemented_rows_development": int(dev.t60_proxy_flag.fillna(False).sum()),
        "t60_proxy_or_complemented_rows_holdout": int(holdout.t60_proxy_flag.fillna(False).sum()),
    }
    best = model_ranking[0]
    paired_dev = paired["development_nested_oof"]
    e_b = paired_dev["E-B"]["log_loss_difference_left_minus_right"]
    e_d = paired_dev["E-D"]["log_loss_difference_left_minus_right"]
    f_b = paired_dev["F-B"]["log_loss_difference_left_minus_right"]
    f_d = paired_dev["F-D"]["log_loss_difference_left_minus_right"]
    if selected_policy["policy"] == "no_trade":
        policy_conclusion = "Development does not support an active frozen policy after the predeclared block-uncertainty, activity, sample-size and fold-stability gates."
    else:
        policy_conclusion = f"The predeclared gates select {selected_policy['policy']} on development; its holdout portfolio is a historical feasibility diagnostic."
    verdict = (
        f"Nested May–June OOF ({len(cv):,} common markets) ranks {best['forecast']} first by log-loss ({best['log_loss']:.5f}); "
        f"E−B={e_b:+.5f}, E−D={e_d:+.5f}, F−B={f_b:+.5f}, F−D={f_d:+.5f} log-loss. "
        f"{policy_conclusion} Holdout is diagnostic only."
    )
    answers = "\n".join([
        f"1. Rynek B kotwiczy prognozę: w zagnieżdżonym OOF B ma log-loss {next(x['log_loss'] for x in cv_quality if x['forecast']=='B_market_reference'):.5f}, Brier {next(x['brier'] for x in cv_quality if x['forecast']=='B_market_reference'):.5f} i ECE {next(x['ece_frozen_dev_bins'] for x in cv_quality if x['forecast']=='B_market_reference'):.4f}. Kalibrację porównuje tabela metryk oraz sparowane bloki.",
        f"2. BTC wnosi użyteczną korektę ponad rynek tylko jeśli nested OOF E/F poprawiają B log-loss z blokowym przedziałem poniżej zera. Obecne sparowane różnice: E−B {e_b:+.5f} [{paired_dev['E-B']['log_loss_difference_95pct'][0]:+.5f}, {paired_dev['E-B']['log_loss_difference_95pct'][1]:+.5f}], F−B {f_b:+.5f} [{paired_dev['F-B']['log_loss_difference_95pct'][0]:+.5f}, {paired_dev['F-B']['log_loss_difference_95pct'][1]:+.5f}].",
        f"3. Względem swobodnej regresji D, E ma Δ log-loss {e_d:+.5f}, a F {f_d:+.5f}; poprawa jest wiarygodna tylko jeśli sparowany CI nie obejmuje zera i diagnostyka kalibracji się zgadza.",
        f"4. Koszty są naliczone na T−59. Wyniki na wybranych transakcjach własnych prognoz oraz na stałych koszykach A/D są pokazane osobno; stałe koszyki zachowują identyczny fee, payout i PnL przy zmianie prognozy. Zamrożona polityka: {selected_policy['policy']}; {policy_conclusion}",
        "5. Liczbę aktywnych dni, PnL w foldach, blokowe CI i udział trzech najbardziej skoncentrowanych dni podano dla wszystkich progów. Holdout pokazuje lipiec–październik jako diagnostykę i nie wpływa na wybór.",
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(outer_grid + final_grid)
    grid.to_csv(OUTPUT_DIR / "nested_parameter_selection.csv", index=False)
    pd.DataFrame(outer_choices).to_csv(OUTPUT_DIR / "outer_fold_model_choices.csv", index=False)
    pd.DataFrame(_coefficient_stability(outer_choices)).to_csv(OUTPUT_DIR / "coefficient_stability.csv", index=False)
    quality = cv_quality + hold_quality
    pd.DataFrame(quality).to_csv(OUTPUT_DIR / "probability_quality.csv", index=False)
    pd.DataFrame(cv_bins).to_csv(OUTPUT_DIR / "calibration_bins.csv", index=False)
    pd.DataFrame(fold_quality).to_csv(OUTPUT_DIR / "outer_fold_probability_quality.csv", index=False)
    pd.DataFrame(monthly_quality).to_csv(OUTPUT_DIR / "monthly_holdout_probability_quality.csv", index=False)
    pd.DataFrame([
        {"period": period, "comparison": name, **row}
        for period, comparisons in paired.items() for name, row in comparisons.items()
    ]).to_csv(OUTPUT_DIR / "paired_probability_differences.csv", index=False)
    economics = cv_economics + hold_economics
    pd.DataFrame(economics).to_csv(OUTPUT_DIR / "independent_order_economics.csv", index=False)
    monthly_economics = cv_monthly_economics + hold_monthly_economics
    pd.DataFrame(monthly_economics).to_csv(OUTPUT_DIR / "monthly_independent_order_economics.csv", index=False)
    pd.DataFrame(cv_fold_economics).to_csv(OUTPUT_DIR / "development_policy_fold_stability.csv", index=False)
    own_quality = _own_selected_trade_quality(cv_details, hurdles) + _own_selected_trade_quality(hold_details, hurdles)
    pd.DataFrame(own_quality).to_csv(OUTPUT_DIR / "own_selected_trade_probability_quality.csv", index=False)
    fixed_baskets = pd.DataFrame(fixed_basket_rows)
    fixed_baskets.to_csv(OUTPUT_DIR / "fixed_execution_forecast_comparison.csv", index=False)
    pd.DataFrame(fixed_pair_rows).to_parquet(OUTPUT_DIR / "fixed_execution_pairs.parquet", index=False, compression="zstd")
    pd.concat([cv_details, hold_details], ignore_index=True).to_parquet(
        OUTPUT_DIR / "independent_order_detail.parquet", index=False, compression="zstd",
    )
    cv.to_parquet(OUTPUT_DIR / "development_nested_oof_predictions.parquet", index=False, compression="zstd")
    holdout.to_parquet(OUTPUT_DIR / "holdout_diagnostic_predictions.parquet", index=False, compression="zstd")

    # Holdout calibration and economic results are written only after the frozen selection exists.
    payload = {
        "verdict": verdict, "answers": answers, "coverage": coverage,
        "source_identity": source_identity, "cv_rows": len(cv), "holdout_rows": len(holdout),
        "outer_folds": [
            {
                "fold_id": outer["fold_id"],
                "start": outer["start_inclusive_utc"],
                "end": outer["end_exclusive_utc"],
                "train_rows": int((dev.market_start_utc.lt(pd.Timestamp(outer["start_inclusive_utc"])) & dev.label_available_at_utc.le(pd.Timestamp(outer["start_inclusive_utc"]))).sum()),
                "valid_rows": int(((cv.outer_fold_id == outer["fold_id"]).sum())),
                "inner_folds": [
                    {
                        "fold_id": inner["fold_id"],
                        "start": inner["start_inclusive_utc"], "end": inner["end_exclusive_utc"],
                        "train_rows": int((dev.market_start_utc.lt(pd.Timestamp(inner["start_inclusive_utc"])) & dev.label_available_at_utc.le(pd.Timestamp(inner["start_inclusive_utc"]))).sum()),
                        "valid_rows": int((dev.market_start_utc.ge(pd.Timestamp(inner["start_inclusive_utc"])) & dev.market_start_utc.lt(pd.Timestamp(inner["end_exclusive_utc"]))).sum()),
                    }
                    for inner in outer["inner_validation_folds"]
                ],
            }
            for outer in config["development_outer_folds"]
        ],
        "final_choices": final_choices,
        "quality": quality, "fold_quality": fold_quality,
        "monthly_quality": monthly_quality, "paired_differences": paired,
        "development_economics": cv_economics, "holdout_economics": hold_economics,
        "development_monthly_economics": cv_monthly_economics,
        "holdout_monthly_economics": hold_monthly_economics,
        "own_selected_trade_quality": own_quality,
        "fixed_baskets": fixed_basket_rows,
        "frozen_selection": frozen_development,
        "holdout_portfolio": holdout_portfolio,
    }
    payload["verdict"] = _build_verdict(payload)
    payload["answers"] = _build_answers(payload)
    verdict = payload["verdict"]
    answers = payload["answers"]
    _write_json(OUTPUT_DIR / "report_payload.json", payload)
    REPORT_PATH.write_text(_render_report(payload), encoding="utf-8")
    _copy_summary_tables()
    manifest = {
        "status": "completed", "experiment_id": config["experiment_id"],
        "configuration_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "configuration_sha256": research._sha256(CONFIG_PATH),
        "frozen_selection_path": FROZEN_SELECTION_PATH.relative_to(ROOT).as_posix(),
        "frozen_selection_sha256": research._sha256(FROZEN_SELECTION_PATH),
        "source_identity": source_identity,
        "development_common_rows": len(dev), "development_nested_oof_rows": len(cv),
        "holdout_common_rows": len(holdout), "holdout_historically_exposed": True,
        "final_choices": final_choices, "probability_quality": quality,
        "paired_probability_differences": paired,
        "answers": answers,
        "development_policy_economics": cv_economics,
        "development_policy_fold_stability": cv_fold_economics,
        "fixed_execution_comparison_rows": len(fixed_basket_rows),
        "frozen_policy": frozen_policy,
        "holdout_portfolio": holdout_portfolio,
        "main_btc_model_training_or_refit": False, "meta_models_fitted": True,
        "book_replay": False, "live_configuration_changed": False,
        "runtime_seconds": time.perf_counter() - started,
        "evaluation_script_sha256": research._sha256(Path(__file__)),
        "report_path": REPORT_PATH.relative_to(ROOT).as_posix(),
    }
    _write_json(OUTPUT_DIR / "experiment_manifest.json", manifest)
    print(json.dumps({
        "verdict": verdict, "frozen_policy": selected_policy,
        "cv_rows": len(cv), "holdout_rows": len(holdout),
        "runtime_seconds": manifest["runtime_seconds"],
    }, indent=2, ensure_ascii=True, default=str))
    return manifest


if __name__ == "__main__":
    main()
