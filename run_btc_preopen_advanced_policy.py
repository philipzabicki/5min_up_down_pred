"""Run frozen DRK/RCK comparisons on the saved BTC pre-open opportunities."""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import run_btc_preopen_policy_optimization as ledger


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_preopen_advanced_policy_20261007.json"
BASE_CONFIG_PATH = ROOT / "configs/research/btc_preopen_policy_search_20261007.json"
SOURCE_STUDY_PATH = ROOT / "reports/btc_preopen/policy_study.json"
OPPORTUNITIES_PATH = ROOT / "reports/btc_preopen/policy_opportunities.parquet"
SUMMARY_PATH = ROOT / "reports/btc_preopen/advanced_policy_state_v2_20261007.json"
FOLDS_PATH = ROOT / "reports/btc_preopen/advanced_policy_state_v2_folds_20261007.csv"
TRIALS_PATH = ROOT / "reports/btc_preopen/advanced_policy_state_v2_trials_20261007.csv"
MANIFEST_PATH = ROOT / "reports/btc_preopen/advanced_policy_state_v2_manifest_20261007.json"
REPORT_PATH = ROOT / "reports/btc_preopen/advanced_policy_state_v2_20261007.md"
PREDICTION_CACHE_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/advanced_policy_20261007"
CACHE_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/advanced_policy_state_v2_20261007"
DECISIONS_PATH = CACHE_DIR / "decisions.parquet"
TRADES_PATH = CACHE_DIR / "trades.parquet"
CHECKPOINT_PATH = CACHE_DIR / "run_checkpoint.json"
KACHO_MARKETS_PATH = ROOT / "data/raw/polymarket/kachoio/42d917dc8e3205dde8ac909792af0cce2d715c9f/btc_markets.parquet"
KACHO_TICKS_PATH = ROOT / "data/raw/polymarket/kachoio/42d917dc8e3205dde8ac909792af0cce2d715c9f/btc_ticks.parquet"
ENTRY_SNAPSHOTS_PATH = ROOT / "reports/btc_preopen/entry_snapshots.parquet"
TRAJECTORY_COVERAGE_PATH = ROOT / "reports/btc_preopen/trajectory_coverage_per_market_20261007.csv"

EPS = 1e-10
INTERVAL_NS = 300_000_000_000
DAY_NS = 86_400_000_000_000
# The failed first wrapper completed every model and policy task before reaching the isolated $5
# selection summary. Only that summary and report were corrected, so those completed outputs remain valid.
COMPATIBLE_RUNNER_SHA256S = {
    "c684f33d9ceaa9a5495820ef811b371e7fddf0ed8cc26b975b021556ec66819d",
    "4a78c26d516a3154d31b5945118f0cf0d322bb74e5a815111e1c459a7561ef3e",
    # Existing non-RCK policy caches are still valid after the RCK audit-count correction.
    "c3c39642087a0e33ff43a1fb214730fad4a3e09dbbca14bb0ad6c20ad24580db",
    "85f1dfe2efb667e630650e01aec52d53590efb195eaed416fc33e364e00f49a7",
    "4a680dd38a455927d3c3b1558ced786226fc763769d89c64958980208ea5d8c9",
}
RCK_AUDIT_METRIC_RUNNER_SHA256S = {
    "c3c39642087a0e33ff43a1fb214730fad4a3e09dbbca14bb0ad6c20ad24580db",
    "85f1dfe2efb667e630650e01aec52d53590efb195eaed416fc33e364e00f49a7",
}


def _identity_matches(cached: dict, current: dict) -> bool:
    if cached == current:
        return True
    same_inputs = all(cached.get(key) == value for key, value in current.items() if key != "runner_sha256")
    return bool(
        same_inputs
        and cached.get("runner_sha256") in COMPATIBLE_RUNNER_SHA256S
        and current.get("runner_sha256") not in COMPATIBLE_RUNNER_SHA256S
    )


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False, default=ledger._json_default, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _json_safe(value):
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logit(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _feature_matrix(rows: list[dict], model_name: str) -> np.ndarray:
    p = np.asarray([row["p_candidate_platt"] for row in rows], dtype=float)
    if model_name == "btc_only":
        return _logit(p).reshape(-1, 1)
    if model_name != "btc_market":
        raise ValueError(f"Unknown calibration model: {model_name}")
    midpoint = np.asarray([
        (float(row["up_best_bid"]) + float(row["up_best_ask"])) / 2.0
        for row in rows
    ], dtype=float)
    return np.column_stack((_logit(p), _logit(midpoint)))


def _fit_logistic(x: np.ndarray, y: np.ndarray, c_value: float):
    if len(np.unique(y)) != 2:
        return None
    model = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=300, random_state=20261007)
    model.fit(x, y)
    return model


def _predict(model, x: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if model is None:
        return np.asarray(fallback, dtype=float).copy()
    return np.clip(model.predict_proba(x)[:, 1], 1e-6, 1.0 - 1e-6)


def _known_before(rows: list[dict], cutoff_ns: int) -> list[dict]:
    return [row for row in rows if int(row["_outcome_ns"]) < int(cutoff_ns)]


def _select_regularization(
    history: list[dict], validation: list[dict], model_name: str, c_grid: list[float]
) -> tuple[float, list[dict]]:
    if len(history) < 100 or len(validation) < 25:
        return float(c_grid[1]), []
    x_train = _feature_matrix(history, model_name)
    y_train = np.asarray([row["target_polymarket_up"] for row in history], dtype=int)
    x_valid = _feature_matrix(validation, model_name)
    y_valid = np.asarray([row["target_polymarket_up"] for row in validation], dtype=int)
    scores = []
    for c_value in c_grid:
        model = _fit_logistic(x_train, y_train, c_value)
        p_valid = _predict(model, x_valid, np.asarray([row["p_candidate_platt"] for row in validation]))
        scores.append({"model": model_name, "C": float(c_value), "brier": float(np.mean((p_valid - y_valid) ** 2))})
    selected = min(scores, key=lambda item: (item["brier"], item["C"]))
    return float(selected["C"]), scores


def _bootstrap_sample_indices(rows: list[dict], rng: np.random.Generator, block_days: int) -> tuple[np.ndarray, int]:
    stamps = np.asarray([int(row["_market_ns"]) for row in rows], dtype=np.int64)
    block_codes = stamps // (int(block_days) * DAY_NS)
    blocks = np.unique(block_codes)
    by_block = {int(code): np.flatnonzero(block_codes == code) for code in blocks}
    picks = []
    while sum(len(part) for part in picks) < len(rows):
        code = int(rng.choice(blocks))
        picks.append(by_block[code])
    return np.concatenate(picks)[:len(rows)], int(len(blocks))


def _fit_fold_models(
    rows: list[dict], fold: dict, config: dict, fold_number: int
) -> tuple[dict[str, dict[str, float]], dict]:
    refit_ns = int(pd.Timestamp(fold["refit_at_utc"]).value)
    fold_start = int(fold["evaluation"]["start_index_in_eligible_order"])
    validation_start = int(fold["selection_validation"]["start_index_in_eligible_order"])
    all_prior = _known_before(rows[:fold_start], refit_ns)
    inner_start_ns = int(rows[validation_start]["_entry_ns"])
    inner_train = _known_before(rows[:validation_start], inner_start_ns)
    inner_valid = _known_before(rows[validation_start:fold_start], refit_ns)
    c_grid = [float(value) for value in config["calibration"]["regularization_C_grid"]]
    bootstrap_config = config["calibration"]["bootstrap"]
    block_days = int(bootstrap_config["block_length_days"])
    replicates = int(bootstrap_config["replicates"])
    min_blocks = int(bootstrap_config["minimum_nonempty_blocks"])
    rng = np.random.default_rng(int(bootstrap_config["seed"]) + fold_number)
    outputs = {row["condition_id"]: {"p_btc_only": float(row["p_candidate_platt"]), "p_btc_only_low": float(row["p_candidate_platt"]), "p_btc_only_high": float(row["p_candidate_platt"]), "p_btc_market": float(row["p_candidate_platt"]), "p_btc_market_low": float(row["p_candidate_platt"]), "p_btc_market_high": float(row["p_candidate_platt"])} for row in rows[fold_start:int(fold["evaluation"]["end_index_exclusive"])]}
    model_meta = {
        "fold_id": int(fold["fold_id"]),
        "refit_at_utc": fold["refit_at_utc"],
        "inner_history_rows": len(inner_train),
        "inner_validation_rows": len(inner_valid),
        "all_prior_label_rows": len(all_prior),
        "distinct_three_day_blocks": int(len(np.unique([row["_market_ns"] // (block_days * DAY_NS) for row in all_prior]))) if all_prior else 0,
        "models": {},
        "bootstrap_models_requested_per_variant": replicates,
        "bootstrap_models_completed_per_variant": 0,
        "fallback": None,
        "inner_candidate_scores": [],
    }
    if model_meta["distinct_three_day_blocks"] < min_blocks:
        model_meta["fallback"] = "insufficient_history_candidate_platt_and_no_robust_orders"
        return outputs, model_meta

    eval_rows = rows[fold_start:int(fold["evaluation"]["end_index_exclusive"])]
    y_all = np.asarray([row["target_polymarket_up"] for row in all_prior], dtype=int)
    lower_q, upper_q = [float(v) for v in bootstrap_config["interval_quantiles"]]
    for model_name, out_prefix in (("btc_only", "btc_only"), ("btc_market", "btc_market")):
        c_value, scores = _select_regularization(inner_train, inner_valid, model_name, c_grid)
        model_meta["inner_candidate_scores"].extend(scores)
        x_all = _feature_matrix(all_prior, model_name)
        model = _fit_logistic(x_all, y_all, c_value)
        x_eval = _feature_matrix(eval_rows, model_name)
        p_fallback = np.asarray([row["p_candidate_platt"] for row in eval_rows], dtype=float)
        p_center = _predict(model, x_eval, p_fallback)
        boot_predictions = []
        if model is not None:
            for replicate in range(replicates):
                sample_idx, _ = _bootstrap_sample_indices(all_prior, rng, block_days)
                boot = _fit_logistic(x_all[sample_idx], y_all[sample_idx], c_value)
                if boot is not None:
                    boot_predictions.append(_predict(boot, x_eval, p_fallback))
                if (replicate + 1) % 100 == 0:
                    print(f"fold {fold_number}: {model_name} bootstrap {replicate + 1}/{replicates}", flush=True)
        if boot_predictions:
            boot_matrix = np.asarray(boot_predictions, dtype=float)
            p_low = np.quantile(boot_matrix, lower_q, axis=0)
            p_high = np.quantile(boot_matrix, upper_q, axis=0)
        else:
            p_low = p_center.copy()
            p_high = p_center.copy()
        for index, row in enumerate(eval_rows):
            target = outputs[row["condition_id"]]
            target[f"p_{out_prefix}"] = float(p_center[index])
            target[f"p_{out_prefix}_low"] = float(p_low[index])
            target[f"p_{out_prefix}_high"] = float(p_high[index])
        model_meta["models"][model_name] = {
            "selected_C": c_value,
            "inner_scores": scores,
            "bootstrap_models_completed": len(boot_predictions),
            "prediction_mean": float(np.mean(p_center)),
            "prediction_interval_mean_width": float(np.mean(p_high - p_low)),
        }
        model_meta["bootstrap_models_completed_per_variant"] = len(boot_predictions)
    return outputs, model_meta


def estimate_residual_correlation(rows: list[dict], shrinkage: float) -> dict:
    residual_by_market = {}
    for row in rows:
        p = float(row["p_candidate_platt"])
        y = float(row["target_polymarket_up"])
        residual_by_market[int(row["_market_ns"])] = (y - p) / math.sqrt(max(p * (1.0 - p), 1e-6))
    pairs = []
    ordered = sorted(residual_by_market)
    for first, second in zip(ordered, ordered[1:]):
        if second - first == INTERVAL_NS:
            pairs.append((residual_by_market[first], residual_by_market[second]))
    if len(pairs) < 3:
        return {"raw_lag_one_correlation": 0.0, "shrunk_lag_one_correlation": 0.0, "adjacent_pairs": len(pairs)}
    x = np.asarray(pairs, dtype=float)
    raw = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])
    if not math.isfinite(raw):
        raw = 0.0
    weight = len(pairs) / (len(pairs) + float(shrinkage))
    rho = float(np.clip(raw * weight, -0.8, 0.8))
    return {"raw_lag_one_correlation": raw, "shrunk_lag_one_correlation": rho, "adjacent_pairs": len(pairs)}


def scenario_outcomes(
    probabilities: list[float], timestamps_ns: list[int], *, dependent: bool,
    rho: float, z_bank: np.ndarray, side_signs: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(probabilities)
    if count == 0:
        return np.empty((len(z_bank), 0), dtype=np.int8), np.full(len(z_bank), 1.0 / len(z_bank))
    correlation = np.eye(count, dtype=float)
    signs = [1] * count if side_signs is None else [int(value) for value in side_signs]
    if len(signs) != count or any(value not in (-1, 1) for value in signs):
        raise ValueError("Scenario side signs must be +1/-1 and match the probabilities")
    if dependent and count > 1:
        for i in range(count):
            for j in range(i):
                ticks = max(1, int(round(abs(int(timestamps_ns[i]) - int(timestamps_ns[j])) / INTERVAL_NS)))
                correlation[i, j] = correlation[j, i] = (rho ** ticks) * signs[i] * signs[j]
    chol = np.linalg.cholesky(correlation + np.eye(count) * 1e-12)
    latent = z_bank[:, :count] @ chol.T
    outcomes = np.zeros((len(z_bank), count), dtype=np.int8)
    for column, probability in enumerate(probabilities):
        wins = int(np.rint(float(np.clip(probability, 0.0, 1.0)) * len(z_bank)))
        if wins:
            order = np.argsort(latent[:, column], kind="stable")
            outcomes[order[:wins], column] = 1
    weights = np.full(len(z_bank), 1.0 / len(z_bank), dtype=float)
    if not np.isclose(weights.sum(), 1.0) or np.any(weights < 0.0):
        raise AssertionError("Scenario weights must be nonnegative and normalized")
    return outcomes, weights


def scenario_probabilities_match(outcomes: np.ndarray, weights: np.ndarray, probabilities: list[float], tolerance: float) -> bool:
    if outcomes.shape[1] != len(probabilities):
        return False
    actual = weights @ outcomes.astype(float)
    return bool(np.all(np.abs(actual - np.asarray(probabilities, dtype=float)) <= tolerance + 1e-12))


def single_bet_log_growth(fraction: float, probability: float, price: float) -> float:
    fraction = float(fraction)
    probability = float(probability)
    price = float(price)
    win = 1.0 + fraction * (1.0 / price - 1.0)
    loss = 1.0 - fraction
    if win <= 0.0 or loss <= 0.0:
        return float("-inf")
    return probability * math.log(win) + (1.0 - probability) * math.log(loss)


def single_bet_kelly_fraction(probability: float, price: float) -> float:
    probability, price = float(probability), float(price)
    if not 0.0 < price < 1.0:
        raise ValueError("Binary contract price must lie strictly between 0 and 1")
    return max(0.0, (probability - price) / (1.0 - price))


def single_bet_drk_fraction(p_low: float, p_high: float, price: float) -> float:
    p_low, p_high = float(p_low), float(p_high)
    if not 0.0 <= p_low <= p_high <= 1.0:
        raise ValueError("DRK probability interval must be ordered and within [0, 1]")
    # Binary log-growth is monotone in win probability, so the lower endpoint is worst.
    return single_bet_kelly_fraction(p_low, price)


def single_bet_risk_moment(fraction: float, probability: float, price: float, lam: float) -> float:
    fraction = float(fraction)
    win = 1.0 + fraction * (1.0 / float(price) - 1.0)
    loss = 1.0 - fraction
    if win <= 0.0 or loss <= 0.0:
        return float("inf")
    return float(probability * win ** (-lam) + (1.0 - probability) * loss ** (-lam))


def single_bet_rck_fraction(probability: float, price: float, lam: float) -> float:
    probability, price, lam = float(probability), float(price), float(lam)
    if not 0.0 < price < 1.0 or probability <= price:
        return 0.0
    full_kelly = single_bet_kelly_fraction(probability, price)
    if single_bet_risk_moment(full_kelly, probability, price, lam) <= 1.0 + 1e-12:
        return full_kelly
    lo, hi = 0.0, min(1.0 - 1e-12, max(full_kelly, 1e-9))
    if single_bet_risk_moment(hi, probability, price, lam) <= 1.0:
        hi = 1.0 - 1e-12
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if single_bet_risk_moment(mid, probability, price, lam) <= 1.0:
            lo = mid
        else:
            hi = mid
    return min(full_kelly, lo)


def _positions_from_state(state: ledger.Portfolio, as_of_ns: int) -> tuple[list[dict], float, int]:
    uncertain = []
    known_locked_payout = 0.0
    known_locked_count = 0
    for release_ns, _, _, payout, _, position in sorted(state.pending, key=lambda item: (item[0], item[1])):
        if int(release_ns) <= int(as_of_ns):
            raise AssertionError("Released capital must be removed before scenario positions are read")
        outcome_available_ns = position.get("outcome_available_ns")
        if outcome_available_ns is None or int(outcome_available_ns) > int(as_of_ns):
            uncertain.append(position)
            continue
        known_locked_count += 1
        known_locked_payout += float(payout)
    return uncertain, known_locked_payout, known_locked_count


def _position_probability(position: dict, robust: bool) -> float:
    return float(position["p_low_win"] if robust else position["p_center_win"])


def _scenario_base_wealth(
    state: ledger.Portfolio, positions: list[dict], y_existing: np.ndarray,
    known_locked_payout: float = 0.0,
) -> np.ndarray:
    wealth = np.full(y_existing.shape[0], float(state.cash) + float(known_locked_payout), dtype=float)
    for column, position in enumerate(positions):
        wealth += float(position["net_shares"]) * y_existing[:, column]
    return wealth


def _continuous_choice(
    *, base_wealth: np.ndarray, y_new: np.ndarray, weight: np.ndarray,
    payout_per_gross: float, debit_per_gross: float, wealth_reference: float,
    max_gross: float, minimum_gross: float, lam: float | None,
) -> tuple[float, float, float, float]:
    base_valid = np.all(base_wealth > 0.0)
    base_log = float(np.sum(weight * np.log(base_wealth / wealth_reference))) if base_valid else float("-inf")
    if max_gross + EPS < minimum_gross or payout_per_gross <= 0.0 or debit_per_gross <= 0.0:
        return 0.0, base_log, base_log, float("nan")
    if lam is not None:
        if not base_valid:
            return 0.0, base_log, base_log, float("inf")
        base_moment = float(np.sum(weight * (base_wealth / wealth_reference) ** (-lam)))
        if base_moment > 1.0 + 1e-10:
            return 0.0, base_log, base_log, base_moment
        at_upper = base_wealth + max_gross * (-debit_per_gross + payout_per_gross * y_new)
        if np.any(at_upper <= 0.0):
            hi = max_gross
        else:
            upper_moment = float(np.sum(weight * (at_upper / wealth_reference) ** (-lam)))
            if upper_moment <= 1.0 + 1e-10:
                hi = max_gross
            else:
                lo = 0.0
                hi = max_gross
                for _ in range(60):
                    mid = (lo + hi) / 2.0
                    wealth_mid = base_wealth + mid * (-debit_per_gross + payout_per_gross * y_new)
                    moment_mid = float(np.sum(weight * (wealth_mid / wealth_reference) ** (-lam))) if np.all(wealth_mid > 0) else float("inf")
                    if moment_mid <= 1.0 + 1e-10:
                        lo = mid
                    else:
                        hi = mid
                hi = lo
        max_gross = hi
    if not base_valid:
        return 0.0, base_log, base_log, float("inf")
    delta = -float(debit_per_gross) + float(payout_per_gross) * y_new
    left, right = float(minimum_gross), float(max_gross)
    if right + EPS < left:
        return 0.0, base_log, base_log, float("nan")
    def derivative(amount: float) -> float:
        wealth = base_wealth + amount * delta
        if np.any(wealth <= 0.0):
            return float("-inf")
        return float(np.sum(weight * delta / wealth))
    if derivative(left) <= 0.0:
        chosen = left
    elif derivative(right) >= 0.0:
        chosen = right
    else:
        lo, hi = left, right
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if derivative(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        chosen = (lo + hi) / 2.0
    wealth = base_wealth + chosen * delta
    objective = float(np.sum(weight * np.log(wealth / wealth_reference))) if np.all(wealth > 0.0) else float("-inf")
    moment = float(np.sum(weight * (wealth / wealth_reference) ** (-lam))) if lam is not None and np.all(wealth > 0.0) else (float("inf") if lam is not None else float("nan"))
    return float(chosen), base_log, objective, moment


def _max_gross(row: dict, side: str, state: ledger.Portfolio, config: dict) -> float:
    ask = float(row[f"{side}_best_ask"])
    top_capacity = ask * float(row[f"{side}_best_ask_size_shares"])
    cash_cap = ledger._cash_gross_cap(row, side, state.cash, config)
    return max(0.0, min(float(config["execution"]["technical_max_gross_purchase_usd"]), top_capacity, cash_cap))


def _minimum_feasible_gross(row: dict, side: str, state: ledger.Portfolio, config: dict) -> float:
    minimum = float(config["execution"]["minimum_purchase_assumption"]["minimum_gross_notional_usd"])
    minimum_shares = float(config["execution"]["minimum_purchase_assumption"]["minimum_net_shares"])
    ask = float(row[f"{side}_best_ask"])
    guess = max(minimum, minimum_shares * ask)
    guess = math.ceil(guess * 100.0 - 1e-9) / 100.0
    for _ in range(10):
        if ledger._feasible_order(row, side, guess, state, config)["admitted_gross_usd"] > 0.0:
            return guess
        guess = round(guess + 0.01, 2)
    return float("inf")


def _evaluate_exact_fill(
    base_wealth: np.ndarray, y_new: np.ndarray, weight: np.ndarray,
    fill: dict, wealth_reference: float, lam: float | None,
) -> tuple[float, float]:
    wealth = base_wealth - float(fill["cash_debit_usd"]) + float(fill["net_shares"]) * y_new
    if np.any(wealth <= 0.0):
        return float("-inf"), float("inf")
    objective = float(np.sum(weight * np.log(wealth / wealth_reference)))
    moment = float(np.sum(weight * (wealth / wealth_reference) ** (-lam))) if lam is not None else float("nan")
    return objective, moment


def _best_unconstrained_fill(
    row: dict, side: str, state: ledger.Portfolio, config: dict,
    base_wealth: np.ndarray, y_new: np.ndarray, weight: np.ndarray,
    wealth_reference: float, request: float, baseline_objective: float,
) -> tuple[dict | None, float]:
    rounded = ledger._round_down_cent(request)
    best_fill = None
    best_objective = baseline_objective
    for amount in (rounded, round(rounded - 0.01, 2), round(rounded + 0.01, 2)):
        if amount <= 0.0 or amount > _max_gross(row, side, state, config) + 0.011:
            continue
        fill = ledger._feasible_order(row, side, amount, state, config)
        if fill["admitted_gross_usd"] <= 0.0:
            continue
        objective, _ = _evaluate_exact_fill(base_wealth, y_new, weight, fill, wealth_reference, None)
        if objective > best_objective + 1e-12:
            best_fill = fill
            best_objective = objective
    return best_fill, best_objective


def _decision_changed_by_risk_constraint(candidates: dict, chosen: dict | None, constrained: bool) -> bool:
    if not constrained:
        return False
    unconstrained = [
        {**candidate, "side": side}
        for side, candidate in candidates.items()
        if float(candidate.get("unconstrained_admitted_gross_usd", 0.0)) > 0.0
        and float(candidate.get("unconstrained_objective", float("-inf")))
        > float(candidate.get("baseline_objective", float("inf"))) + 1e-12
    ]
    if not unconstrained:
        return chosen is not None
    best_unconstrained = max(unconstrained, key=lambda item: (float(item["unconstrained_objective"]), item["side"] == "up"))
    if chosen is None or str(chosen["side"]) != str(best_unconstrained["side"]):
        return True
    return abs(float(chosen["admitted_gross_usd"]) - float(best_unconstrained["unconstrained_admitted_gross_usd"])) > 0.005


def _choose_candidate(
    row: dict, side: str, state: ledger.Portfolio, config: dict, positions: list[dict],
    *, p_up: float, p_up_low: float, robust: bool, dependent: bool, rho: float,
    z_bank: np.ndarray, lam: float | None, known_locked_payout: float = 0.0,
) -> dict:
    p_center_win = p_up if side == "up" else 1.0 - p_up
    p_low_win = p_up_low if side == "up" else 1.0 - float(row.get("_p_up_high", p_up))
    p_new = p_low_win if robust else p_center_win
    probabilities = [_position_probability(position, robust) for position in positions] + [float(p_new)]
    times = [int(position["market_start_ns"]) for position in positions] + [int(row["_market_ns"])]
    signs = [1 if position["side"] == "up" else -1 for position in positions] + [1 if side == "up" else -1]
    outcomes, weights = scenario_outcomes(probabilities, times, dependent=dependent, rho=rho, z_bank=z_bank, side_signs=signs)
    y_existing = outcomes[:, :len(positions)].astype(float)
    y_new = outcomes[:, -1].astype(float)
    base_wealth = _scenario_base_wealth(state, positions, y_existing, known_locked_payout)
    wealth_reference = float(state.equity)
    if wealth_reference <= 0.0:
        return {"requested_gross_usd": 0.0, "admitted_gross_usd": 0.0, "skip_reason": "nonpositive_wealth_reference", "objective": float("-inf"), "baseline_objective": float("-inf"), "risk_moment": float("inf"), "risk_changed": False}
    max_gross = _max_gross(row, side, state, config)
    minimum = _minimum_feasible_gross(row, side, state, config)
    payout_rate, debit_rate = ledger._net_unit_rates(row, side, config)
    requested, baseline_objective, _, continuous_moment = _continuous_choice(
        base_wealth=base_wealth, y_new=y_new, weight=weights,
        payout_per_gross=payout_rate, debit_per_gross=debit_rate,
        wealth_reference=wealth_reference, max_gross=max_gross,
        minimum_gross=minimum, lam=lam,
    )
    unconstrained_request = requested
    if lam is not None:
        unconstrained_request, _, _, _ = _continuous_choice(
            base_wealth=base_wealth, y_new=y_new, weight=weights,
            payout_per_gross=payout_rate, debit_per_gross=debit_rate,
            wealth_reference=wealth_reference, max_gross=max_gross,
            minimum_gross=minimum, lam=None,
        )
        unconstrained_fill, unconstrained_objective = _best_unconstrained_fill(
            row, side, state, config, base_wealth, y_new, weights,
            wealth_reference, unconstrained_request, baseline_objective,
        )
        unconstrained_admitted = 0.0 if unconstrained_fill is None else float(unconstrained_fill["admitted_gross_usd"])
    else:
        unconstrained_admitted = 0.0
        unconstrained_objective = baseline_objective
    unconstrained_fields = {
        "unconstrained_admitted_gross_usd": unconstrained_admitted,
        "unconstrained_objective": unconstrained_objective,
    }
    if lam is not None:
        no_trade_moment = float(np.sum(weights * (base_wealth / wealth_reference) ** (-lam))) if np.all(base_wealth > 0) else float("inf")
        if no_trade_moment > 1.0 + 1e-10:
            return {**unconstrained_fields, "requested_gross_usd": 0.0, "unconstrained_requested_gross_usd": unconstrained_request, "admitted_gross_usd": 0.0, "skip_reason": "existing_portfolio_risk_constraint_breach", "objective": baseline_objective, "baseline_objective": baseline_objective, "risk_moment": no_trade_moment, "risk_changed": unconstrained_admitted > 0.01}
    if requested <= EPS or not math.isfinite(requested):
        return {**unconstrained_fields, "requested_gross_usd": 0.0, "unconstrained_requested_gross_usd": unconstrained_request, "admitted_gross_usd": 0.0, "skip_reason": "no_positive_log_growth", "objective": baseline_objective, "baseline_objective": baseline_objective, "risk_moment": continuous_moment, "risk_changed": lam is not None and unconstrained_admitted > 0.01}

    rounded = ledger._round_down_cent(requested)
    chosen = None
    best_objective = baseline_objective
    best_moment = float("nan")
    best_fill = None
    for amount in (rounded, round(rounded - 0.01, 2), round(rounded + 0.01, 2)):
        if amount <= 0.0 or amount > max_gross + 0.011:
            continue
        fill = ledger._feasible_order(row, side, amount, state, config)
        if fill["admitted_gross_usd"] <= 0.0:
            continue
        objective, moment = _evaluate_exact_fill(base_wealth, y_new, weights, fill, wealth_reference, lam)
        if lam is not None and moment > 1.0 + 1e-8:
            continue
        if objective > best_objective + 1e-12:
            chosen, best_objective, best_moment, best_fill = amount, objective, moment, fill
    if best_fill is None:
        reason = "risk_constraint" if lam is not None else "rounded_order_not_beneficial_or_feasible"
        return {**unconstrained_fields, "requested_gross_usd": float(requested), "unconstrained_requested_gross_usd": unconstrained_request, "admitted_gross_usd": 0.0, "skip_reason": reason, "objective": baseline_objective, "baseline_objective": baseline_objective, "risk_moment": continuous_moment, "risk_changed": lam is not None and unconstrained_admitted > 0.01}
    return {
        **unconstrained_fields,
        **best_fill,
        "requested_gross_usd": float(requested),
        "unconstrained_requested_gross_usd": float(unconstrained_request),
        "admitted_gross_usd": float(best_fill["admitted_gross_usd"]),
        "skip_reason": None,
        "objective": float(best_objective),
        "baseline_objective": float(baseline_objective),
        "risk_moment": float(best_moment),
        "risk_changed": bool(lam is not None and (unconstrained_admitted - best_fill["admitted_gross_usd"]) > 0.01),
        "p_win_center": float(p_center_win),
        "p_win_robust": float(p_low_win),
        "limit_reasons": list(best_fill.get("limit_reasons", [])),
    }


def _rss_peak_bytes() -> int | None:
    if not hasattr(__import__("ctypes"), "WinDLL"):
        return None
    import ctypes
    from ctypes import wintypes

    class MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
    kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    get_process_memory_info.restype = wintypes.BOOL
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE
    ok = get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize) if ok else None


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.parquet")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.csv")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _load_or_fit_predictions(rows: list[dict], folds: list[dict], config: dict, identity: dict) -> tuple[dict[int, dict[str, dict[str, float]]], list[dict]]:
    predictions = {}
    fold_meta = []
    for fold_number, fold in enumerate(folds, start=1):
        prediction_path = PREDICTION_CACHE_DIR / f"fold_{fold_number}_predictions.parquet"
        metadata_path = PREDICTION_CACHE_DIR / f"fold_{fold_number}_metadata.json"
        cache_valid = False
        if prediction_path.is_file() and metadata_path.is_file():
            try:
                cached_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                cache_valid = _identity_matches(cached_meta.get("identity", {}), identity)
                if cache_valid:
                    data = pd.read_parquet(prediction_path)
                    predictions[fold_number] = {
                        str(record["condition_id"]): {key: float(value) for key, value in record.items() if key != "condition_id"}
                        for record in data.to_dict(orient="records")
                    }
                    fold_meta.append(cached_meta["model_meta"])
            except (OSError, ValueError, KeyError, pd.errors.ParserError):
                cache_valid = False
        if cache_valid:
            print(f"fold {fold_number}/{len(folds)}: reusing causal calibration and bootstrap cache", flush=True)
            continue
        prediction_path = CACHE_DIR / f"fold_{fold_number}_predictions.parquet"
        metadata_path = CACHE_DIR / f"fold_{fold_number}_metadata.json"
        print(f"fold {fold_number}/{len(folds)}: fitting calibration and {config['calibration']['bootstrap']['replicates']} block-bootstrap replicates per model", flush=True)
        started = time.perf_counter()
        fold_predictions, metadata = _fit_fold_models(rows, fold, config, fold_number)
        prior_rows = _known_before(rows[:int(fold["evaluation"]["start_index_in_eligible_order"])], int(pd.Timestamp(fold["refit_at_utc"]).value))
        rho = estimate_residual_correlation(prior_rows, float(config["scenario_model"]["correlation_shrinkage_n"]))
        metadata["residual_dependence"] = rho
        metadata["fit_elapsed_seconds"] = time.perf_counter() - started
        predictions[fold_number] = fold_predictions
        fold_meta.append(metadata)
        prediction_frame = pd.DataFrame([
            {"condition_id": condition_id, **values}
            for condition_id, values in fold_predictions.items()
        ])
        _write_parquet_atomic(prediction_frame, prediction_path)
        _write_json_atomic(metadata_path, {"identity": identity, "model_meta": metadata})
        print(f"fold {fold_number}: fitted in {metadata['fit_elapsed_seconds']:.1f}s; rho={rho['shrunk_lag_one_correlation']:.3f}; peak RSS={_rss_peak_bytes()}", flush=True)
    return predictions, fold_meta


def _close_advanced(state: ledger.Portfolio, metrics: ledger.RunMetrics, until_ns: int, *, inclusive: bool = True) -> None:
    while state.pending:
        release_ns, sequence, gross, _, condition_id, position = state.pending[0]
        if release_ns > until_ns or (not inclusive and release_ns == until_ns):
            break
        heapq.heappop(state.pending)
        won = int(position["target_up"]) == (1 if position["side"] == "up" else 0)
        payout = float(position["net_shares"]) if won else 0.0
        state.locked_cost -= float(gross)
        state.cash += payout
        if abs(state.locked_cost) < 1e-9:
            state.locked_cost = 0.0
        if state.cash < -1e-7:
            raise AssertionError("Portfolio cash became negative")
        metrics.record(int(release_ns), "capital_release", state, str(condition_id))


def _policy_specs(config: dict) -> list[dict]:
    rows = [
        {"policy_id": "point_kelly_btc_only", "kind": "portfolio_kelly", "calibration": "btc_only", "robust": False, "dependent": True, "lambda": None},
        {"policy_id": "drk_btc_only", "kind": "drk", "calibration": "btc_only", "robust": True, "dependent": True, "lambda": None},
        {"policy_id": "point_kelly_btc_market", "kind": "portfolio_kelly", "calibration": "btc_market", "robust": False, "dependent": True, "lambda": None},
        {"policy_id": "drk_btc_market", "kind": "drk", "calibration": "btc_market", "robust": True, "dependent": True, "lambda": None},
        {"policy_id": "portfolio_kelly_independent", "kind": "portfolio_kelly", "calibration": "btc_market", "robust": False, "dependent": False, "lambda": None},
    ]
    for profile in config["rck"]["profiles"]:
        alpha = float(profile["alpha"])
        lam = float(profile["lambda"])
        for dependent in (True, False):
            suffix = "correlated" if dependent else "independent"
            rows.append({
                "policy_id": f"rck_alpha_{alpha:.1f}_{suffix}", "kind": "rck",
                "calibration": "btc_market", "robust": False, "dependent": dependent,
                "lambda": lam, "alpha": alpha, "beta": float(profile["beta"]),
            })
    return rows


def _simulate_advanced(
    rows: list[dict], folds: list[dict], fold_predictions: dict[int, dict[str, dict[str, float]]],
    fold_meta: list[dict], spec: dict, base_config: dict, config: dict, z_bank: np.ndarray,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    state = ledger.Portfolio()
    start_ns = int(rows[int(folds[0]["evaluation"]["start_index_in_eligible_order"])]["_entry_ns"])
    metrics = ledger.RunMetrics(initial_equity=state.equity, start_ns=start_ns)
    decisions, trades, block_rows = [], [], []
    for fold_number, fold in enumerate(folds, start=1):
        first = int(fold["evaluation"]["start_index_in_eligible_order"])
        stop = int(fold["evaluation"]["end_index_exclusive"])
        rho = float(fold_meta[fold_number - 1]["residual_dependence"]["shrunk_lag_one_correlation"])
        fold_start_equity = state.equity
        fold_trade_start = len(trades)
        fold_decision_start = len(decisions)
        predictions = fold_predictions[fold_number]
        for row in rows[first:stop]:
            entry_ns = int(row["_entry_ns"])
            _close_advanced(state, metrics, entry_ns, inclusive=True)
            metrics.decision_count += 1
            pred = predictions[row["condition_id"]]
            p_up = float(pred[f"p_{spec['calibration']}"])
            p_low = float(pred[f"p_{spec['calibration']}_low"])
            p_high = float(pred[f"p_{spec['calibration']}_high"])
            eta = spec.get("eta_by_fold", {}).get(fold_number)
            if eta is None:
                eta = float(spec.get("uncertainty_eta", 1.0 if spec["robust"] else 0.0))
            p_low = min(p_low, p_up)
            p_high = max(p_high, p_up)
            adjusted_low = p_up + float(eta) * (p_low - p_up)
            adjusted_high = p_up + float(eta) * (p_high - p_up)
            action_row = {**row, "_p_up_high": adjusted_high}
            positions, known_locked_payout, known_locked_count = _positions_from_state(state, entry_ns)
            candidates = {}
            for side in ("up", "down"):
                candidates[side] = _choose_candidate(
                    action_row, side, state, base_config, positions,
                    p_up=p_up, p_up_low=adjusted_low, robust=bool(spec["robust"]),
                    dependent=bool(spec["dependent"]), rho=rho, z_bank=z_bank,
                    lam=spec["lambda"], known_locked_payout=known_locked_payout,
                )
            eligible = [
                {**candidate, "side": side}
                for side, candidate in candidates.items()
                if candidate.get("skip_reason") is None
            ]
            chosen = max(eligible, key=lambda item: (float(item["objective"]), item["side"] == "up")) if eligible else None
            risk_changed = _decision_changed_by_risk_constraint(candidates, chosen, spec["lambda"] is not None)
            cash_before = float(state.cash)
            locked_before = float(state.locked_cost)
            common = {
                "policy_id": spec["policy_id"],
                "fold_id": int(fold["fold_id"]),
                "condition_id": str(row["condition_id"]),
                "market_start_utc": row["market_start_utc"],
                "entry_time_utc": row["entry_time_utc"],
                "p_candidate_platt_up": float(row["p_candidate_platt"]),
                "p_calibrated_up": p_up,
                "p_bootstrap_low": p_low,
                "p_bootstrap_high": p_high,
                "correlation_rho": rho if spec["dependent"] else 0.0,
                "scenario_dependence": "ar1_gaussian_copula" if spec["dependent"] else "independent",
                "scenario_count": int(config["scenario_model"]["scenario_count"]),
                "cash_before_usd": cash_before,
                "locked_cost_before_usd": locked_before,
                "open_positions_before": len(state.pending),
                "available_cash_now_usd": cash_before,
                "known_outcome_locked_position_count": known_locked_count,
                "known_outcome_locked_payout_usd": known_locked_payout,
                "uncertain_outcome_locked_position_count": len(positions),
                "cost_basis_equity_reference_approx_usd": float(state.equity),
                "rck_lambda": spec["lambda"],
                "decision_changed_by_risk_constraint": risk_changed,
            }
            if chosen is None:
                reason_values = [candidate.get("skip_reason") for candidate in candidates.values()]
                reason = "no_positive_feasible_growth" if not any(reason == "existing_portfolio_risk_constraint_breach" for reason in reason_values) else "existing_portfolio_risk_constraint_breach"
                metrics.no_trade_skips += 1
                decisions.append({
                    **common, "chosen_side": "no_trade", "decision_reason": reason,
                    "up_requested_usd": float(candidates["up"].get("requested_gross_usd", 0.0)),
                    "up_feasible_usd": float(candidates["up"].get("admitted_gross_usd", 0.0)),
                    "up_rejection_reason": candidates["up"].get("skip_reason"),
                    "up_limit_reasons": ";".join(candidates["up"].get("limit_reasons", [])),
                    "up_risk_moment": candidates["up"].get("risk_moment"),
                    "down_requested_usd": float(candidates["down"].get("requested_gross_usd", 0.0)),
                    "down_feasible_usd": float(candidates["down"].get("admitted_gross_usd", 0.0)),
                    "down_rejection_reason": candidates["down"].get("skip_reason"),
                    "down_limit_reasons": ";".join(candidates["down"].get("limit_reasons", [])),
                    "down_risk_moment": candidates["down"].get("risk_moment"),
                    "cash_after_usd": float(state.cash), "locked_cost_after_usd": float(state.locked_cost),
                })
                continue
            side = str(chosen["side"])
            fill = chosen
            debit = float(fill["cash_debit_usd"])
            gross = float(fill["admitted_gross_usd"])
            shares = float(fill["net_shares"])
            if debit > state.cash + EPS:
                raise AssertionError("Advanced policy order would use credit")
            p_side = float(fill["p_win_center"])
            p_low_side = float(fill["p_win_robust"])
            target_up = int(row["target_polymarket_up"])
            won = target_up == (1 if side == "up" else 0)
            payout = shares if won else 0.0
            state.cash -= debit
            state.locked_cost += gross
            state.sequence += 1
            position = {
                "net_shares": shares,
                "p_center_win": p_side,
                "p_low_win": p_low_side,
                "market_start_ns": int(row["_market_ns"]),
                "side": side,
                "target_up": target_up,
                "outcome_available_ns": int(row["_outcome_ns"]),
                "fold_id": int(fold["fold_id"]),
            }
            heapq.heappush(state.pending, (
                int(row["_release_ns"]), int(state.sequence), gross, payout,
                str(row["condition_id"]), position,
            ))
            metrics.trade_count += 1
            metrics.gross_turnover += gross
            metrics.cash_debit_total += debit
            metrics.fees_usd += float(fill["fees_usd"])
            metrics.fee_shares += float(fill.get("fee_shares", 0.0))
            metrics.expected_net_profit += p_side * shares - debit
            if "top_of_book_liquidity" in fill.get("limit_reasons", []):
                metrics.liquidity_limited_orders += 1
            if "cash_including_fee" in fill.get("limit_reasons", []):
                metrics.cash_limited_orders += 1
            metrics.record(entry_ns, "purchase", state, str(row["condition_id"]))
            decision = {
                **common, "chosen_side": side, "decision_reason": "maximum_scenario_log_growth",
                "requested_gross_usd": float(fill["requested_gross_usd"]),
                "unconstrained_requested_gross_usd": float(fill.get("unconstrained_requested_gross_usd", fill["requested_gross_usd"])),
                "feasible_gross_usd": gross,
                "cash_debit_usd": debit, "fees_usd": float(fill["fees_usd"]),
                "top_of_book_capacity_usd": float(fill["top_of_book_capacity_usd"]),
                "cash_gross_cap_usd": float(fill["cash_gross_cap_usd"]),
                "limit_reasons": ";".join(fill.get("limit_reasons", [])),
                "risk_moment": fill.get("risk_moment"),
                "expected_log_growth_scenarios": float(fill["objective"]),
                "no_trade_log_growth_scenarios": float(fill["baseline_objective"]),
                "p_side_center": p_side, "p_side_robust_low": p_low_side,
                "cash_after_usd": float(state.cash), "locked_cost_after_usd": float(state.locked_cost),
                "open_positions_after": len(state.pending),
                "up_requested_usd": float(candidates["up"].get("requested_gross_usd", 0.0)),
                "up_feasible_usd": float(candidates["up"].get("admitted_gross_usd", 0.0)),
                "up_rejection_reason": candidates["up"].get("skip_reason"),
                "up_limit_reasons": ";".join(candidates["up"].get("limit_reasons", [])),
                "up_risk_moment": candidates["up"].get("risk_moment"),
                "down_requested_usd": float(candidates["down"].get("requested_gross_usd", 0.0)),
                "down_feasible_usd": float(candidates["down"].get("admitted_gross_usd", 0.0)),
                "down_rejection_reason": candidates["down"].get("skip_reason"),
                "down_limit_reasons": ";".join(candidates["down"].get("limit_reasons", [])),
                "down_risk_moment": candidates["down"].get("risk_moment"),
            }
            decisions.append(decision)
            trades.append({
                "policy_id": spec["policy_id"], "fold_id": int(fold["fold_id"]),
                "condition_id": str(row["condition_id"]), "market_slug": row["market_slug"],
                "entry_time_utc": row["entry_time_utc"], "release_time_utc": row["capital_release_at_utc"],
                "chosen_side": side, "p_up_base": float(row["p_candidate_platt"]),
                "p_up_calibrated": p_up, "p_up_low": p_low, "p_up_high": p_high,
                "requested_gross_usd": float(fill["requested_gross_usd"]),
                "gross_purchase_usd": gross, "cash_debit_usd": debit,
                "best_ask": float(fill["price"]), "top_ask_capacity_usd": float(fill["top_of_book_capacity_usd"]),
                "gross_shares": float(fill["gross_shares"]), "net_shares": shares,
                "fees_usd": float(fill["fees_usd"]), "won": bool(won), "payout_usd": payout,
                "realized_net_profit_usd": payout - debit,
                "expected_net_profit_at_entry_usd": p_side * shares - debit,
                "cash_before_usd": cash_before, "cash_after_entry_usd": float(state.cash),
                "open_positions_before": len(positions), "open_positions_after": len(state.pending),
                "risk_moment": fill.get("risk_moment"), "rck_lambda": spec["lambda"],
                "decision_changed_by_risk_constraint": risk_changed,
                "scenario_dependence": "ar1_gaussian_copula" if spec["dependent"] else "independent",
            })
        eval_end_ns = int(rows[stop - 1]["_entry_ns"])
        block_trades = trades[fold_trade_start:]
        block_rows.append({
            "policy_id": spec["policy_id"], "fold_id": int(fold["fold_id"]),
            "evaluation_start_utc": fold["evaluation"]["decision_start_utc"],
            "evaluation_end_utc": fold["evaluation"]["decision_end_utc"],
            "eligible_markets": stop - first,
            "decisions": len(decisions) - fold_decision_start,
            "trades_entered": len(block_trades),
            "realized_pnl_on_block_entries_usd": float(sum(t["realized_net_profit_usd"] for t in block_trades)),
            "mean_expected_log_growth_at_entry": float(np.mean([
                d["expected_log_growth_scenarios"] for d in decisions[fold_decision_start:]
                if d.get("policy_id") == spec["policy_id"] and d.get("fold_id") == int(fold["fold_id"]) and d.get("chosen_side") != "no_trade"
            ])) if any(d.get("fold_id") == int(fold["fold_id"]) and d.get("chosen_side") != "no_trade" for d in decisions[fold_decision_start:]) else None,
            "starting_cost_basis_equity_usd": float(fold_start_equity),
            "cost_basis_equity_after_last_decision_usd": float(state.equity),
            "cash_after_last_decision_usd": float(state.cash),
            "open_positions_after_last_decision": len(state.pending),
            "risk_constraint_changed_decisions": int(sum(bool(d.get("decision_changed_by_risk_constraint")) for d in decisions[fold_decision_start:])),
        })
    final_release = max([int(rows[-1]["_release_ns"])] + [int(item[0]) for item in state.pending])
    _close_advanced(state, metrics, final_release, inclusive=True)
    end_ns = max(final_release, int(rows[-1]["_entry_ns"]))
    summary = ledger._summary(state, metrics, end_ns)
    summary.update({
        "policy_id": spec["policy_id"], "family": spec["kind"],
        "calibration": spec["calibration"], "robust_probability_interval": bool(spec["robust"]),
        "scenario_dependence": "ar1_gaussian_copula" if spec["dependent"] else "independent",
        "rck_alpha": spec.get("alpha"), "rck_beta": spec.get("beta"), "rck_lambda": spec["lambda"],
        "risk_constraint_changed_decisions": int(sum(bool(row.get("decision_changed_by_risk_constraint")) for row in decisions)),
        "outer_markets": len(rows) - int(folds[0]["evaluation"]["start_index_in_eligible_order"]),
    })
    return summary, decisions, trades, block_rows


def _run_legacy_controls(rows: list[dict], folds: list[dict], base_config: dict, study: dict) -> tuple[dict, dict, list[dict], list[dict], list[dict]]:
    outer_start = int(study["split_design_frozen_before_search"]["outer_folds"][0]["evaluation"]["start_index_in_eligible_order"])
    outer_rows = rows[outer_start:]
    full_start = int(rows[0]["_entry_ns"])
    full_end = max(int(row["_release_ns"]) for row in rows)
    controls = {}
    full_exact = ledger._run_baseline(rows, {"trial_id": "legacy_exact_fixed_5", "portfolio_id": "development", "family": "legacy_exact_fixed_5", "parameters": {}}, base_config, full_start, full_end, False)[2]
    full_level = ledger._run_baseline(rows, {"trial_id": "legacy_best_level_fixed_5", "portfolio_id": "development", "family": "legacy_best_level_fixed_5", "parameters": {}}, base_config, full_start, full_end, False)[2]
    controls["legacy_exact_fixed_5_full_development"] = full_exact
    controls["legacy_best_level_fixed_5_full_development"] = full_level
    policies = [
        {"trial_id": "no_trade", "portfolio_id": "outer_no_trade", "family": "no_trade", "parameters": {}},
        {"trial_id": "legacy_best_level_fixed_5", "portfolio_id": "outer_legacy_5", "family": "legacy_best_level_fixed_5", "parameters": {}},
        {"trial_id": "legacy_exact_fixed_5", "portfolio_id": "outer_exact_5", "family": "legacy_exact_fixed_5", "parameters": {}},
        {"trial_id": "fractional_kelly_half_10pct", "portfolio_id": "outer_fractional_kelly", "family": "fractional_kelly", "parameters": {"kelly_multiplier": 0.5, "max_cost_basis_equity_fraction": 0.1, "max_gross_stake_usd": 20.0, "min_expected_return": 0.0}},
    ]
    baseline_decisions, baseline_trades, baseline_folds = [], [], []
    fold_by_market = {
        str(record["condition_id"]): int(fold["fold_id"])
        for fold in folds
        for record in rows[int(fold["evaluation"]["start_index_in_eligible_order"]):int(fold["evaluation"]["end_index_exclusive"])]
    }
    for policy in policies:
        state = ledger.Portfolio()
        metrics = ledger._new_metrics(state, int(outer_rows[0]["_entry_ns"]))
        policy_blocks = []
        for fold in folds:
            first = int(fold["evaluation"]["start_index_in_eligible_order"])
            stop = int(fold["evaluation"]["end_index_exclusive"])
            trade_start = len(metrics.trades)
            decision_start = len(metrics.decisions)
            ledger._simulate_policy(rows[first:stop], policy, state, metrics, base_config, include_logs=True)
            block_trades = metrics.trades[trade_start:]
            policy_blocks.append({
                "policy_id": policy["trial_id"], "fold_id": int(fold["fold_id"]),
                "evaluation_start_utc": fold["evaluation"]["decision_start_utc"],
                "evaluation_end_utc": fold["evaluation"]["decision_end_utc"],
                "eligible_markets": stop - first,
                "decisions": len(metrics.decisions) - decision_start,
                "trades_entered": len(block_trades),
                "realized_pnl_on_block_entries_usd": float(sum(item["realized_net_profit_usd"] for item in block_trades)),
                "cash_after_last_decision_usd": float(state.cash),
                "cost_basis_equity_after_last_decision_usd": float(state.equity),
                "open_positions_after_last_decision": len(state.pending),
            })
        outer_end_ns = max(int(row["_release_ns"]) for row in outer_rows)
        ledger._close_positions(state, metrics, outer_end_ns, inclusive=True)
        summary = ledger._summary(state, metrics, outer_end_ns)
        controls[policy["trial_id"]] = summary
        baseline_folds.extend(policy_blocks)
        for row in metrics.decisions:
            baseline_decisions.append({**row, "policy_id": policy["trial_id"], "fold_id": fold_by_market.get(str(row.get("condition_id")))})
        for row in metrics.trades:
            baseline_trades.append({**row, "policy_id": policy["trial_id"], "fold_id": fold_by_market.get(str(row.get("condition_id")))})
    return controls, {"outer_markets": len(outer_rows), "full_eligible_markets": len(rows)}, baseline_decisions, baseline_trades, baseline_folds


def _fixed_stake_selection(
    rows: list[dict], folds: list[dict], fold_predictions: dict[int, dict[str, dict[str, float]]],
    base_config: dict, z_bank: np.ndarray,
) -> list[dict]:
    variants = [
        ("candidate_platt", None, None),
        ("point_kelly_btc_only", "btc_only", False),
        ("drk_btc_only", "btc_only", True),
        ("point_kelly_btc_market", "btc_market", False),
        ("drk_btc_market", "btc_market", True),
    ]
    summaries = []
    for label, calibration, robust in variants:
        pnl, selected, feasible, wins, up_count, down_count, rejected = [], 0, 0, 0, 0, 0, {}
        for fold_number, fold in enumerate(folds, start=1):
            first = int(fold["evaluation"]["start_index_in_eligible_order"])
            stop = int(fold["evaluation"]["end_index_exclusive"])
            prediction = fold_predictions[fold_number]
            for row in rows[first:stop]:
                pred = prediction[row["condition_id"]]
                if calibration is None:
                    p_up = float(row["p_candidate_platt"])
                    p_low = p_high = p_up
                else:
                    p_up = float(pred[f"p_{calibration}"])
                    p_low = float(pred[f"p_{calibration}_low"])
                    p_high = float(pred[f"p_{calibration}_high"])
                candidates = []
                for side in ("up", "down"):
                    p_side = p_up if side == "up" else 1.0 - p_up
                    p_robust = p_low if side == "up" else 1.0 - p_high
                    fill = ledger._feasible_order(row, side, 5.0, ledger.Portfolio(), base_config)
                    if fill["admitted_gross_usd"] <= 0.0:
                        rejected[fill["skip_reason"]] = rejected.get(fill["skip_reason"], 0) + 1
                        continue
                    p_score = p_robust if robust else p_side
                    wealth = 100.0
                    win_wealth = wealth - fill["cash_debit_usd"] + fill["net_shares"]
                    loss_wealth = wealth - fill["cash_debit_usd"]
                    objective = p_score * math.log(win_wealth / wealth) + (1.0 - p_score) * math.log(loss_wealth / wealth)
                    if objective > 0.0:
                        candidates.append((objective, side, fill))
                if not candidates:
                    continue
                _, side, fill = max(candidates, key=lambda item: (item[0], item[1] == "up"))
                selected += 1
                actual_win = int(row["target_polymarket_up"]) == (1 if side == "up" else 0)
                profit = (float(fill["net_shares"]) if actual_win else 0.0) - float(fill["cash_debit_usd"])
                pnl.append(profit)
                feasible += 1
                wins += int(actual_win)
                up_count += int(side == "up")
                down_count += int(side == "down")
        summaries.append({
            "comparison": "selection_fixed_5_usd_independent_markets_no_reinvestment",
            "strategy": label, "markets": sum(int(fold["evaluation"]["markets"]) for fold in folds),
            "selected": selected, "feasible": feasible, "up_trades": up_count, "down_trades": down_count,
            "wins": wins, "hit_rate": wins / feasible if feasible else None,
            "gross_stake_usd": feasible * 5.0, "realized_pnl_usd": float(sum(pnl)),
            "mean_pnl_per_trade_usd": float(np.mean(pnl)) if pnl else None,
            "rejections_by_reason": rejected,
        })
    return summaries


def _frozen_sizing_comparison(
    rows: list[dict], folds: list[dict], fold_predictions: dict[int, dict[str, dict[str, float]]],
    base_config: dict, config: dict, z_bank: np.ndarray,
) -> list[dict]:
    output = []
    outer_start = int(folds[0]["evaluation"]["start_index_in_eligible_order"])
    names = [
        ("candidate_fractional_kelly", None, False, None),
        ("point_kelly_btc_only", "btc_only", False, None),
        ("drk_btc_only", "btc_only", True, None),
        ("point_kelly_btc_market", "btc_market", False, None),
        ("drk_btc_market", "btc_market", True, None),
        ("rck_alpha_0.5", "btc_market", False, float(config["rck"]["profiles"][0]["lambda"])),
        ("rck_alpha_0.7", "btc_market", False, float(config["rck"]["profiles"][1]["lambda"])),
        ("rck_alpha_0.8", "btc_market", False, float(config["rck"]["profiles"][2]["lambda"])),
    ]
    for policy_id, calibration, robust, lam in names:
        pnl, requested, admitted, feasible, skipped, capacities_limited = [], [], [], 0, {}, 0
        for fold_number, fold in enumerate(folds, start=1):
            first = int(fold["evaluation"]["start_index_in_eligible_order"])
            stop = int(fold["evaluation"]["end_index_exclusive"])
            for row in rows[first:stop]:
                p_base = float(row["p_candidate_platt"])
                evs = []
                for side in ("up", "down"):
                    fill5 = ledger._exact_reference_fill(row, side)
                    pside = p_base if side == "up" else 1.0 - p_base
                    expected = pside * fill5["net_shares"] - fill5["cash_debit_usd"]
                    evs.append((expected, side))
                if max(evs)[0] <= 0.0:
                    continue
                side = max(evs, key=lambda item: (item[0], item[1] == "up"))[1]
                pred = fold_predictions[fold_number][row["condition_id"]]
                if calibration is None:
                    p_up, p_low, p_high = p_base, p_base, p_base
                else:
                    p_up = float(pred[f"p_{calibration}"])
                    p_low = float(pred[f"p_{calibration}_low"])
                    p_high = float(pred[f"p_{calibration}_high"])
                pside = p_up if side == "up" else 1.0 - p_up
                p_robust = p_low if side == "up" else 1.0 - p_high
                one_trade_state = ledger.Portfolio()
                if policy_id == "candidate_fractional_kelly":
                    params = {"kelly_multiplier": 0.5, "max_cost_basis_equity_fraction": 0.1, "max_gross_stake_usd": 20.0, "min_expected_return": 0.0}
                    row_for_request = {**row, "p_candidate_platt": p_base}
                    raw, caps = ledger._sized_request(row_for_request, side, "fractional_kelly", params, one_trade_state, base_config)
                    req = min(raw, 20.0)
                else:
                    probability = p_robust if robust else pside
                    payout_rate, debit_rate = ledger._net_unit_rates(row, side, base_config)
                    win_profit = payout_rate - debit_rate
                    edge = probability * payout_rate - debit_rate
                    req = min(max(0.0, edge / (debit_rate * win_profit) * one_trade_state.equity) if edge > 0.0 and win_profit > 0.0 else 0.0, 20.0)
                    caps = []
                    if lam is not None and req > 0.0:
                        row_for_sizing = {**row, "_p_up_high": p_high}
                        candidate = _choose_candidate(
                            row_for_sizing, side, one_trade_state, base_config, [],
                            p_up=p_up, p_up_low=p_low, robust=False, dependent=False,
                            rho=0.0, z_bank=z_bank, lam=lam,
                        )
                        req = float(candidate.get("requested_gross_usd", 0.0))
                requested.append(float(req))
                fill = ledger._feasible_order(row, side, req, one_trade_state, base_config)
                if fill["admitted_gross_usd"] <= 0.0:
                    reason = fill["skip_reason"] or "not_feasible"
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue
                if "top_of_book_liquidity" in fill["limit_reasons"]:
                    capacities_limited += 1
                amount = float(fill["admitted_gross_usd"])
                admitted.append(amount)
                feasible += 1
                won = int(row["target_polymarket_up"]) == (1 if side == "up" else 0)
                pnl.append((float(fill["net_shares"]) if won else 0.0) - float(fill["cash_debit_usd"]))
        output.append({
            "comparison": "sizing_on_frozen_legacy_market_side_time",
            "strategy": policy_id, "frozen_population_markets": len(rows) - outer_start,
            "requested_orders": len(requested), "feasible_orders": feasible,
            "feasibility_rate": feasible / len(requested) if requested else None,
            "mean_requested_usd": float(np.mean(requested)) if requested else None,
            "mean_admitted_usd": float(np.mean(admitted)) if admitted else None,
            "realized_pnl_on_feasible_orders_usd": float(sum(pnl)),
            "top_of_book_limited_orders": capacities_limited,
            "rejections_by_reason": skipped,
            "side_and_time_frozen_from": "candidate_platt legacy exact-$5 expected-profit side rule",
        })
    return output


def _safe_policy_file(policy_id: str, suffix: str) -> Path:
    return CACHE_DIR / f"policy_{policy_id}_{suffix}"


def _load_policy_cache(policy_id: str, identity: dict):
    summary_path = _safe_policy_file(policy_id, "summary.json")
    decisions_path = _safe_policy_file(policy_id, "decisions.parquet")
    trades_path = _safe_policy_file(policy_id, "trades.parquet")
    folds_path = _safe_policy_file(policy_id, "folds.csv")
    if not all(path.is_file() for path in (summary_path, decisions_path, trades_path, folds_path)):
        return None
    try:
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if policy_id.startswith("rck_alpha_") and cached.get("identity", {}).get("runner_sha256") in RCK_AUDIT_METRIC_RUNNER_SHA256S:
            return None
        if not _identity_matches(cached.get("identity", {}), identity):
            return None
        result = (
            cached["summary"],
            pd.read_parquet(decisions_path).to_dict(orient="records"),
            pd.read_parquet(trades_path).to_dict(orient="records"),
            pd.read_csv(folds_path).to_dict(orient="records"),
        )
        if cached.get("identity") != identity:
            _write_json_atomic(summary_path, {"identity": identity, "summary": cached["summary"]})
        return result
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return None


def _save_policy_cache(policy_id: str, identity: dict, result) -> None:
    summary, decisions, trades, fold_rows = result
    _write_json_atomic(_safe_policy_file(policy_id, "summary.json"), {"identity": identity, "summary": summary})
    _write_parquet_atomic(pd.DataFrame(decisions), _safe_policy_file(policy_id, "decisions.parquet"))
    _write_parquet_atomic(pd.DataFrame(trades), _safe_policy_file(policy_id, "trades.parquet"))
    _write_csv_atomic(pd.DataFrame(fold_rows), _safe_policy_file(policy_id, "folds.csv"))


def _trajectory_audit() -> dict:
    archive_root = ROOT / "data/analysis/polymarket/BTC/preopen_v1/pmxt_v2_archive_full_scenarios"
    extraction_path = archive_root / "extraction_summary.json"
    replay_path = ROOT / "reports/btc_preopen/event_replay_summary.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    local_parts = list((archive_root / "event_parts").glob("*.parquet"))
    raw_root = ROOT / "data/raw/polymarket"
    raw_files = list(raw_root.rglob("*.parquet")) if raw_root.exists() else []
    opportunities = pd.read_parquet(OPPORTUNITIES_PATH, columns=["condition_id", "eligible", "market_start_utc", "target_polymarket_up"])
    eligible = opportunities.loc[opportunities["eligible"].astype(bool)].copy()
    eligible["market_start_utc"] = pd.to_datetime(eligible["market_start_utc"], utc=True)
    market_meta = pd.read_parquet(
        KACHO_MARKETS_PATH,
        columns=["condition_id", "market_start", "market_end", "recorded_at", "token_up", "token_down", "outcome", "n_ticks"],
    )
    token_index = pd.read_parquet(
        archive_root / "market_index.parquet",
        columns=["condition_id", "up_token_id", "down_token_id"],
    )
    targets = eligible.merge(market_meta, on="condition_id", how="left", validate="one_to_one")
    targets = targets.merge(token_index, on="condition_id", how="left", validate="one_to_one")
    targets["token_ids_match"] = (
        targets["token_up"].astype("string") == targets["up_token_id"].astype("string")
    ) & (
        targets["token_down"].astype("string") == targets["down_token_id"].astype("string")
    )
    ticks = pd.read_parquet(
        KACHO_TICKS_PATH,
        columns=["condition_id", "t", "bu", "au", "bd", "ad", "su", "sd", "sau", "sad"],
    )
    target_ids = set(eligible["condition_id"].astype(str))
    ticks = ticks.loc[ticks["condition_id"].isin(target_ids)].copy()
    start_seconds = targets.set_index("condition_id")["market_start"].astype("int64") // 1_000_000_000
    ticks["market_start_epoch_s"] = ticks["condition_id"].map(start_seconds)
    ticks["offset_seconds"] = ticks["t"] - ticks["market_start_epoch_s"]
    ticks.sort_values(["condition_id", "t"], kind="stable", inplace=True)
    ticks["gap_from_previous_seconds"] = ticks.groupby("condition_id", sort=False)["t"].diff()
    coverage = ticks.groupby("condition_id", sort=False).agg(
        trajectory_rows=("t", "size"),
        unique_sample_seconds=("t", "nunique"),
        first_offset_seconds=("offset_seconds", "min"),
        last_offset_seconds=("offset_seconds", "max"),
        internal_gaps=("gap_from_previous_seconds", lambda values: int((values > 1).sum())),
        duplicate_seconds=("t", lambda values: int(values.duplicated().sum())),
        up_bid_size_missing=("su", lambda values: int(values.isna().sum())),
        down_bid_size_missing=("sd", lambda values: int(values.isna().sum())),
        up_bid_price_missing=("bu", lambda values: int(values.isna().sum())),
        down_bid_price_missing=("bd", lambda values: int(values.isna().sum())),
    ).reset_index()
    coverage = eligible[["condition_id", "market_start_utc"]].merge(coverage, on="condition_id", how="left", validate="one_to_one")
    coverage = coverage.merge(targets[["condition_id", "market_start", "market_end", "recorded_at", "n_ticks", "token_ids_match", "outcome", "target_polymarket_up"]], on="condition_id", how="left", validate="one_to_one")
    coverage["expected_300s_coverage"] = (
        coverage["trajectory_rows"].eq(300)
        & coverage["unique_sample_seconds"].eq(300)
        & coverage["first_offset_seconds"].eq(0)
        & coverage["last_offset_seconds"].eq(299)
        & coverage["internal_gaps"].eq(0)
        & coverage["duplicate_seconds"].eq(0)
    )
    coverage.to_csv(TRAJECTORY_COVERAGE_PATH, index=False)
    complete = coverage["expected_300s_coverage"].fillna(False)
    nonnull_outcome = coverage["outcome"].notna()
    outcome_match = coverage.loc[nonnull_outcome, "outcome"].str.lower().eq(
        coverage.loc[nonnull_outcome, "target_polymarket_up"].map({1: "up", 0: "down"})
    )
    pmxt_check = {}
    if ENTRY_SNAPSHOTS_PATH.is_file():
        entry_quotes = pd.read_parquet(
            ENTRY_SNAPSHOTS_PATH,
            columns=["condition_id", "entry_case", "up_best_bid", "up_best_ask", "down_best_bid", "down_best_ask"],
        )
        entry_quotes = entry_quotes.loc[
            entry_quotes["entry_case"].eq("market_start_c45_o5") & entry_quotes["condition_id"].isin(target_ids)
        ].drop(columns="entry_case")
        at_five = ticks.loc[ticks["offset_seconds"].eq(5), ["condition_id", "bu", "au", "bd", "ad"]]
        compared = entry_quotes.merge(at_five, on="condition_id", how="inner", validate="one_to_one")
        pmxt_check = {"markets_compared_at_market_start_plus_5s": int(len(compared)), "median_absolute_price_difference_by_side": {}}
        for local, sampled in (("up_best_bid", "bu"), ("up_best_ask", "au"), ("down_best_bid", "bd"), ("down_best_ask", "ad")):
            difference = (compared[local].astype(float) - compared[sampled].astype(float)).abs()
            pmxt_check["median_absolute_price_difference_by_side"][local] = float(difference.median()) if len(difference) else None
    orderbook_files = [path for path in raw_files if "orderbooks" in path.parts]
    resolution_files = [path for path in raw_files if "resolutions" in path.parts]
    eligible_token_index = token_index.loc[token_index["condition_id"].isin(target_ids)]
    target_token_ids = set(eligible_token_index["up_token_id"].astype(str)) | set(eligible_token_index["down_token_id"].astype(str))
    raw_orderbook_audit = []
    for path in orderbook_files:
        frame = pd.read_parquet(path, columns=["condition_id", "token_id", "timestamp"])
        ids = set(frame["condition_id"].dropna().astype(str)) | set(frame["token_id"].dropna().astype(str))
        raw_orderbook_audit.append({
            "path": path.relative_to(ROOT).as_posix(),
            "rows": int(len(frame)),
            "first_timestamp_utc": pd.to_datetime(frame["timestamp"], utc=True).min().isoformat(),
            "last_timestamp_utc": pd.to_datetime(frame["timestamp"], utc=True).max().isoformat(),
            "target_condition_or_token_ids": int(len(ids & (target_ids | target_token_ids))),
        })
    raw_resolution_ids = set()
    for path in resolution_files:
        resolution_ids = pd.read_parquet(path, columns=["condition_id"])["condition_id"].dropna().astype(str)
        raw_resolution_ids.update(resolution_ids)
    raw_orderbook_time_overlap_count = sum(
        pd.Timestamp(item["first_timestamp_utc"]) <= eligible["market_start_utc"].max()
        and pd.Timestamp(item["last_timestamp_utc"]) >= eligible["market_start_utc"].min()
        for item in raw_orderbook_audit
    )
    target_range = {
        "start_utc": eligible["market_start_utc"].min().isoformat(),
        "end_utc": eligible["market_start_utc"].max().isoformat(),
    }
    return {
        "local_event_partitions": len(local_parts),
        "local_event_partition_bytes": int(sum(path.stat().st_size for path in local_parts)),
        "cached_event_rows": int(replay["event_rows"]),
        "covered_market_states": int(replay["market_states_with_events"]),
        "snapshot_rows": int(replay["snapshot_rows"]),
        "local_cutoff": extraction["row_group_filter"],
        "full_hourly_files_downloaded": bool(extraction["full_hourly_files_downloaded"]),
        "eligible_markets_requiring_trajectory": int(len(eligible)),
        "target_market_start_range": target_range,
        "post_entry_source": "local Kacho 1-second top-of-book observations from market start through market end; existing PMXT event replay supplies pre-start exit decisions",
        "post_entry_trajectory_available": bool(len(eligible) and complete.all()),
        "post_entry_market_rows": int(coverage["trajectory_rows"].fillna(0).sum()),
        "post_entry_markets_with_exact_300_second_coverage": int(complete.sum()),
        "post_entry_markets_with_gaps_or_incomplete_coverage": int((~complete).sum()),
        "post_entry_observation_seconds": {"first": 0, "last": 299, "nominal_market_end_offset": 300, "decision_grid_seconds": 5},
        "markets_with_up_bid_size_missing_samples": int(coverage["up_bid_size_missing"].fillna(0).gt(0).sum()),
        "markets_with_down_bid_size_missing_samples": int(coverage["down_bid_size_missing"].fillna(0).gt(0).sum()),
        "up_bid_size_missing_samples": int(coverage["up_bid_size_missing"].fillna(0).sum()),
        "down_bid_size_missing_samples": int(coverage["down_bid_size_missing"].fillna(0).sum()),
        "kacho_market_ids_matched": int(targets["market_start"].notna().sum()),
        "kacho_start_matches": int((targets["market_start"] == targets["market_start_utc"]).sum()),
        "kacho_token_ids_match_pmxt": int(targets["token_ids_match"].fillna(False).sum()),
        "kacho_market_end_duration_seconds": sorted((targets["market_end"] - targets["market_start"]).dropna().dt.total_seconds().unique().tolist()),
        "kacho_recorded_outcome_nonnull": int(nonnull_outcome.sum()),
        "kacho_recorded_outcome_matches_official_label": int(outcome_match.sum()),
        "kacho_recorded_outcome_used_for_pnl": False,
        "quote_source_crosscheck": pmxt_check,
        "per_market_coverage_csv": TRAJECTORY_COVERAGE_PATH.relative_to(ROOT).as_posix(),
        "other_local_parquet_files_found": len(raw_files),
        "other_local_orderbook_parquet_files": len(orderbook_files),
        "other_local_resolution_parquet_files": len(resolution_files),
        "other_local_orderbook_market_time_overlap_files": int(raw_orderbook_time_overlap_count),
        "other_local_orderbook_target_condition_or_token_ids": int(sum(item["target_condition_or_token_ids"] for item in raw_orderbook_audit)),
        "other_local_resolution_target_condition_ids": int(len(raw_resolution_ids & target_ids)),
        "other_local_orderbook_audit": raw_orderbook_audit,
        "recovery_source_pattern_in_existing_extractor": "https://r2v2.pmxt.dev/polymarket_orderbook_{hour}.parquet",
        "recovery_assessment": "selective PMXT HTTP range access is available, but the local 1-second Kacho trajectories cover all eligible post-entry windows; no additional data download was needed",
        "outcome_availability_proxy": "outcome_available_at_utc equals resolved_at_utc from shared_market_evaluation; it is a market-resolution timestamp, not an observed label-receipt timestamp",
        "trajectory_sampling_limitations": [
            "Kacho records one cached top-of-book sample per second; websocket receive timestamps are not included.",
            "Sale feasibility uses the sampled best bid and its displayed top-level quantity; deeper levels are not present in this source.",
            "The exact timestamp cross-check with the independent PMXT replay differs by a median of one to two price ticks; the exit study uses Kacho snapshots as a single consistent trajectory source after market start and does not forward-fill missing quotes.",
        ],
        "hold_to_resolution_available": True,
        "trained_or_simple_exit_policy_evaluable": bool(len(eligible) and complete.all()),
    }


def _comparison_row(name: str, summary: dict, category: str) -> dict:
    return {
        "strategy": name,
        "comparison": category,
        "trades": int(summary.get("trade_count", 0)),
        "pnl_usd": float(summary.get("net_pnl_usd", 0.0)),
        "mean_daily_log_growth": summary.get("mean_daily_log_growth"),
        "max_drawdown_cost_basis": summary.get("max_drawdown_cost_basis"),
        "max_underwater_seconds": summary.get("maximum_time_underwater_seconds"),
        "max_open_cost_basis_usd": summary.get("maximum_concurrent_cost_basis_exposure_usd"),
        "max_open_positions": summary.get("maximum_concurrent_positions"),
        "minimum_cash_usd": summary.get("minimum_free_cash_usd"),
        "fees_usd": float(summary.get("fees_paid_usd", 0.0)),
        "turnover_usd": float(summary.get("gross_turnover_usd", 0.0)),
    }


def _fmt(value, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _make_report(summary: dict) -> str:
    rows = summary["common_outer_comparison"]
    header = "| Polityka | Transakcje | PnL | Śr. dzienny log wzrostu | Max DD kosztowy | Czas pod wodą (dni) | Max ekspozycja | Min. gotówka | Opłaty |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    table = [header, sep]
    for row in rows:
        table.append(
            f"| {row['strategy']} | {row['trades']} | ${_fmt(row['pnl_usd'])} | {_fmt(100 * row['mean_daily_log_growth'], 3) if row['mean_daily_log_growth'] is not None else 'n/a'}% | {_fmt(100 * row['max_drawdown_cost_basis'], 2) if row['max_drawdown_cost_basis'] is not None else 'n/a'}% | {_fmt(row['max_underwater_seconds'] / 86400, 2) if row['max_underwater_seconds'] is not None else 'n/a'} | ${_fmt(row['max_open_cost_basis_usd'])} | ${_fmt(row['minimum_cash_usd'])} | ${_fmt(row['fees_usd'], 4)} |"
        )
    full = summary["development_baselines"]
    exact = full["legacy_exact_fixed_5_full_development"]["net_pnl_usd"]
    level = full["legacy_best_level_fixed_5_full_development"]["net_pnl_usd"]
    delta = level - exact
    outer = summary["common_outer_comparison"]
    by_name = {row["strategy"]: row for row in outer}
    drk_btc = by_name["drk_btc_market"]
    point_btc = by_name["point_kelly_btc_market"]
    rck = [row for row in outer if row["strategy"].startswith("rck_alpha_")]
    frac = by_name["fractional_kelly_half_10pct"]
    best_rck = max(rck, key=lambda row: row["pnl_usd"]) if rck else None
    if best_rck and best_rck["pnl_usd"] > frac["pnl_usd"] and best_rck["max_drawdown_cost_basis"] < frac["max_drawdown_cost_basis"]:
        rck_conclusion = f"W tym okresie {best_rck['strategy']} miał wyższy PnL i niższy drawdown niż kontrola fractional Kelly; wynik jest rozwojowy i wymaga potwierdzenia poza tym historycznym zakresem."
    else:
        rck_conclusion = "Nie ma stabilnej poprawy RCK nad fractional Kelly w dostępnych blokach; profil ryzyka należy czytać jako lokalne ograniczenie scenariuszowe, nie gwarancję drawdownu."
    if drk_btc["pnl_usd"] > point_btc["pnl_usd"]:
        drk_conclusion = "Wariant DRK z ceną rynku zakończył z wyższym PnL niż odpowiadający mu punktowy Kelly; bootstrapowa niepewność nie wyjaśnia jednak sama poprawy kalibracji i wynik pozostaje rozwojowy."
    else:
        drk_conclusion = "Wariant DRK z ceną rynku nie poprawił PnL względem odpowiadającego mu punktowego Kelly; sama odporność na niepewność nie wykazała stabilnej korzyści."
    trajectory = summary["optimal_stopping_data_audit"]
    fold_table = ["| Polityka | Blok | Transakcje | PnL transakcji z bloku | Equity po decyzjach | Zmiany przez RCK |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary["fold_results"]:
        fold_table.append(f"| {row['policy_id']} | {row['fold_id']} | {row.get('trades_entered', 0)} | ${_fmt(row.get('realized_pnl_on_block_entries_usd'))} | ${_fmt(row.get('cost_basis_equity_after_last_decision_usd', row.get('cost_basis_equity_after_last_decision_usd', 100.0)))} | {row.get('risk_constraint_changed_decisions', 0)} |")
    selection_table = [
        "| Reguła wyboru | Rynki | Wybrane | Wykonalne $5 | UP / DOWN | Win rate | PnL | Śr. PnL / trade |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["selection_comparison"]:
        selection_table.append(
            f"| {row['strategy']} | {row['markets']} | {row['selected']} | {row['feasible']} | {row['up_trades']} / {row['down_trades']} | {_fmt(100 * row['hit_rate'], 2) if row['hit_rate'] is not None else 'n/a'}% | ${_fmt(row['realized_pnl_usd'])} | ${_fmt(row['mean_pnl_per_trade_usd'], 4)} |"
        )
    sizing_table = [
        "| Sizing | Zamrożone wejścia | Wykonalne | Wykonalność | Śr. żądana | Śr. przyjęta | PnL na wykonalnych | Ograniczone top-level |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["sizing_comparison"]:
        sizing_table.append(
            f"| {row['strategy']} | {row['requested_orders']} | {row['feasible_orders']} | {_fmt(100 * row['feasibility_rate'], 2) if row['feasibility_rate'] is not None else 'n/a'}% | ${_fmt(row['mean_requested_usd'])} | ${_fmt(row['mean_admitted_usd'])} | ${_fmt(row['realized_pnl_on_feasible_orders_usd'])} | {row['top_of_book_limited_orders']} |"
        )
    calibration_table = ["| Blok | Model | Wybrane C | Szerokość przedziału p (średnia) | ρ AR(1) po shrinkage |", "|---:|---|---:|---:|---:|"]
    for meta in summary["calibration_by_fold"]:
        rho = meta.get("residual_dependence", {}).get("shrunk_lag_one_correlation")
        for model_name, model in meta.get("models", {}).items():
            calibration_table.append(f"| {meta['fold_id']} | {model_name} | {model['selected_C']:g} | {_fmt(model['prediction_interval_mean_width'], 4)} | {_fmt(rho, 4)} |")
    return "\n".join([
        "# Badanie DRK, RCK i OS — 7 października 2026",
        "",
        "Badanie używa zamrożonego `candidate_platt`, wejścia T−59 s, kapitału $100, braku kredytu i limitu zakupu $20. To retrospektywna walidacja rozwojowa na okresie wcześniej używanym do selekcji, a nie nowy niezależny holdout.",
        "",
        "## Rekonstrukcja baseline’u",
        "",
        f"Odtworzony dokładny fill $5: **${exact:+.6f}**. Baseline starej reguły na zapisanym best level: **${level:+.6f}** (różnica ${delta:+.6f}, wynik modelu wykonania). Obie wartości są dla pełnego okresu rozwojowego. Wspólny zewnętrzny okres obejmuje {summary['coverage']['outer_markets']} rynków w trzech zamrożonych blokach.",
        "",
        "## Wspólna tabela polityk na zewnętrznych blokach",
        "",
        *table,
        "",
        "PnL, drawdown i czas pod wodą pochodzą z tego samego chronologicznego ledgeru. Ekspozycja to koszt brutto nierozliczonych pozycji; nie jest wyceną likwidacyjną. Wypełnienie zakupu przyjęto na zapisanym best ask i ilości, a historyczne opłaty liczy wspólny ledger.",
        "",
        "## DRK",
        "",
        drk_conclusion,
        f"Punktowa kalibracja BTC+midpoint rynku: PnL ${_fmt(point_btc['pnl_usd'])}; DRK na tym samym modelu z przedziałem bootstrapowym: ${_fmt(drk_btc['pnl_usd'])}. Porównanie BTC-only i BTC+market jest w tabeli oraz `advanced_policy_study_20261007.json`; C wybierano wyłącznie na poprzedzającym bloku wewnętrznym, a bootstrap używał 500 replik 3-dniowych bloków.",
        "",
        "## RCK",
        "",
        rck_conclusion,
        "RCK używa wspólnych scenariuszy AR(1) copula oraz wariantu niezależnego, a także trzech intensywności α=0.5/0.7/0.8 przy β=0.1. Mianownikiem jest cost-basis equity zamrożone przed decyzją; to jawne przybliżenie wartości majątku, a nie wycena mark-to-market. Środki z otwartych pozycji wchodzą do terminalnych wypłat scenariuszowych raz, lecz nie finansują bieżącego zakupu. Gdy scenariuszowy portfel już narusza lokalny limit, dodatkowy zakup jest blokowany.",
        "",
        "## 1. Pełne polityki na wspólnej populacji",
        "",
        "Tabela wyżej pokazuje ciągły portfel na trzech outer blokach, z oddzielnym przebiegiem dla każdego wariantu i początkowym kapitałem $100.",
        "",
        "## 2. Selekcja przy wspólnej stawce $5",
        "",
        *selection_table,
        "",
        "Każdy rynek jest oceniany osobno bez reinwestowania i bez wspólnego ograniczenia gotówkowego; wykonanie i minima pozostają wspólne. To izoluje wybór strony od sizingu.",
        "",
        "## 3. Sizing na zamrożonym market/side/time",
        "",
        *sizing_table,
        "",
        "Lista market/side/time pochodzi ze starej reguły wyboru strony na dokładnym fillu $5. Tabela pokazuje PnL tylko dla wykonalnych zleceń, a osobno liczbę odrzuceń i ograniczeń ilości.",
        "",
        "## 4. Wyjścia na identycznych zakupach i portfel",
        "",
        "| Reguła wyjścia | Wynik |",
        "|---|---|",
        f"| Hold-to-resolution | dostępna kontrola; zakupy i rozliczenia z ledgeru |",
        "| Prosta reguła sprzedaży | oceniona w osobnym raporcie `optimal_stopping_20261007.md` |",
        "| Regresyjne OS | oceniono w osobnym raporcie `optimal_stopping_20261007.md` |",
        "",
        f"Audyt znalazł {trajectory['post_entry_markets_with_exact_300_second_coverage']}/{trajectory['eligible_markets_requiring_trajectory']} rynków z 300 próbkami jedn-sekundowymi i bez luk. Dane Kacho zawierają bid, ask i ilość na najlepszym poziomie; brak ilości oznacza brak wykonalnej sprzedaży na tym poziomie. `resolved_at_utc` jest jawnym przybliżeniem dostępności wyniku, a nie zarejestrowanym czasem odbioru etykiety. Per-market coverage: `{trajectory['per_market_coverage_csv']}`. Dodatkowe pobranie PMXT nie było potrzebne; szczegóły OS są w osobnym raporcie kontynuacji.",
        "",
        "## Kalibracja i zależność scenariuszy",
        "",
        *calibration_table,
        "",
        "C wybierano wyłącznie na poprzedzającym bloku wewnętrznym. Przedziały p są kwantylami 5–95% z 500 bootstrapów 3-dniowych bloków, nie gwarantowanymi przedziałami prawdziwego prawdopodobieństwa.",
        "",
        "## Wyniki między blokami",
        "",
        *fold_table,
        "",
        "Licznik zmian RCK porownuje najlepsza wykonalna akcje bez ograniczenia (strone i stake) z ostateczna akcja po ograniczeniu; liczy tylko zmiane strony lub przyjetej kwoty.",
        "",
        "Kalibracja i estymacja zależności były dopasowane przed każdym blokiem, a etykiety po czasie refit były purge’owane przez `outcome_available_at_utc`. Wyniki są historyczne i mają prior development exposure.",
        "",
        "## Artefakty i ograniczenia",
        "",
        "- Zamrożona konfiguracja: `configs/research/btc_preopen_advanced_policy_20261007.json`.",
        "- Tabela wyników i manifest: `advanced_policy_study_20261007.json`, `advanced_policy_folds_20261007.csv`, `advanced_policy_trials_20261007.csv`, `advanced_policy_manifest_20261007.json`.",
        "- Pełne decyzje i transakcje pozostają w ignorowanym `data/analysis/polymarket/BTC/preopen_v1/advanced_policy_20261007/`.",
        "- Brak potwierdzonych offline filli ani pełnej drabinki; zapisany best-level jest badawczym modelem wykonania, nie rzeczywistym potwierdzeniem fillu.",
        "- Wynik okresu historycznego nie zastępuje niezależnego okresu po zamrożeniu procedury.",
        "",
    ])


def _run() -> None:
    started = time.perf_counter()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    base_config = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    study = json.loads(SOURCE_STUDY_PATH.read_text(encoding="utf-8"))
    rows, folds, row_metadata = ledger._load_rows(base_config, study)
    opportunity_times = pd.read_parquet(OPPORTUNITIES_PATH, columns=["eligible", "resolved_at_utc", "outcome_available_at_utc"])
    eligible_times = opportunity_times.loc[opportunity_times["eligible"].astype(bool)]
    row_metadata["outcome_availability"] = {
        "field": "outcome_available_at_utc",
        "source_field": "resolved_at_utc in shared_market_evaluation.parquet",
        "eligible_rows_equal_to_resolved_at": int((eligible_times["outcome_available_at_utc"] == eligible_times["resolved_at_utc"]).sum()),
        "eligible_rows": int(len(eligible_times)),
        "observed_label_receipt_timestamp": False,
        "interpretation": "resolved_at_utc is an explicit proxy for result availability; the source does not record when the strategy first observed the label",
        "purge_rule": "labels strictly before fit/refit; an open position becomes deterministic when the proxy timestamp is at or before a decision",
    }
    for row in rows:
        row["_market_ns"] = int(pd.Timestamp(row["market_start_utc"]).value)
    if int(config["entry"]["relative_to_market_start"].replace("T-", "").replace("s", "")) != 59:
        raise AssertionError("The frozen entry point must remain T-59s")
    candidate_model_path = ROOT / "data/models/BTC/btc_preopen_candidate_20261005/candidate_model.txt"
    identity = {
        "runner_sha256": _sha256(Path(__file__)),
        "advanced_config_sha256": _sha256(CONFIG_PATH),
        "base_policy_config_sha256": _sha256(BASE_CONFIG_PATH),
        "source_study_sha256": _sha256(SOURCE_STUDY_PATH),
        "opportunities_sha256": _sha256(OPPORTUNITIES_PATH),
        "candidate_model_sha256": _sha256(candidate_model_path),
    }
    print(f"Loaded {len(rows)} eligible opportunities; outer={sum(int(f['evaluation']['markets']) for f in folds)}; rows reused without feature/model retraining.", flush=True)
    controls, coverage, baseline_decisions, baseline_trades, baseline_folds = _run_legacy_controls(rows, folds, base_config, study)
    reproduced = float(controls["legacy_exact_fixed_5_full_development"]["net_pnl_usd"])
    print(f"Recreated exact-fill $5 baseline: ${reproduced:+.6f}; best-level=${controls['legacy_best_level_fixed_5_full_development']['net_pnl_usd']:+.6f}", flush=True)
    if abs(reproduced - 475.840156) > 0.00001:
        raise RuntimeError(f"Exact-fill baseline differs from documented +$475.840156: {reproduced:+.9f}; investigate before comparisons")

    prediction_identity = {**identity, "calibration": config["calibration"], "scenario_model": config["scenario_model"]}
    fold_predictions, fold_meta = _load_or_fit_predictions(rows, folds, config, prediction_identity)
    outer_rows = rows[int(folds[0]["evaluation"]["start_index_in_eligible_order"]):]
    z_bank = np.random.default_rng(int(config["scenario_model"]["seed"])).standard_normal((int(config["scenario_model"]["scenario_count"]), 16))
    # Validate normalized, nonnegative scenario weights and probability margins before outer portfolio replay.
    outcomes, weights = scenario_outcomes([0.2, 0.6, 0.8], [0, INTERVAL_NS, 2 * INTERVAL_NS], dependent=True, rho=0.55, z_bank=z_bank)
    if not scenario_probabilities_match(outcomes, weights, [0.2, 0.6, 0.8], float(config["scenario_model"]["marginal_check_tolerance"])):
        raise AssertionError("Scenario bank does not reproduce its requested probability margins")

    checkpoint = {}
    if CHECKPOINT_PATH.is_file():
        try:
            checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            checkpoint = {}
    completed = set(checkpoint.get("completed_policy_ids", [])) if _identity_matches(checkpoint.get("identity", {}), identity) else set()
    custom_summaries, decisions, trades, fold_rows = {}, list(baseline_decisions), list(baseline_trades), list(baseline_folds)
    policy_specs = _policy_specs(config)
    policy_started = time.perf_counter()
    for index, spec in enumerate(policy_specs, start=1):
        policy_id = spec["policy_id"]
        cached = _load_policy_cache(policy_id, identity) if policy_id in completed else None
        if cached is not None:
            result = cached
            _write_json_atomic(CHECKPOINT_PATH, {"identity": identity, "completed_policy_ids": sorted(completed), "last_completed_policy": policy_id, "elapsed_seconds": time.perf_counter() - started})
            print(f"policy {index}/{len(policy_specs)} {policy_id}: checkpoint reused", flush=True)
        else:
            print(f"policy {index}/{len(policy_specs)} {policy_id}: replay starts ({coverage['outer_markets']} markets)", flush=True)
            result = _simulate_advanced(rows, folds, fold_predictions, fold_meta, spec, base_config, config, z_bank)
            _save_policy_cache(policy_id, identity, result)
            completed.add(policy_id)
            _write_json_atomic(CHECKPOINT_PATH, {"identity": identity, "completed_policy_ids": sorted(completed), "last_completed_policy": policy_id, "elapsed_seconds": time.perf_counter() - started})
            elapsed = time.perf_counter() - policy_started
            eta = elapsed / index * (len(policy_specs) - index)
            print(f"policy {index}/{len(policy_specs)} {policy_id}: done in {time.perf_counter()-policy_started:.1f}s; replay ETA {eta:.1f}s; PnL=${result[0]['net_pnl_usd']:+.2f}; trades={result[0]['trade_count']}", flush=True)
        custom_summaries[policy_id] = result[0]
        decisions.extend(result[1])
        trades.extend(result[2])
        fold_rows.extend(result[3])

    selection_comparison = _fixed_stake_selection(rows, folds, fold_predictions, base_config, z_bank)
    sizing_comparison = _frozen_sizing_comparison(rows, folds, fold_predictions, base_config, config, z_bank)
    common = []
    for policy_id in ("no_trade", "legacy_best_level_fixed_5", "legacy_exact_fixed_5", "fractional_kelly_half_10pct"):
        common.append(_comparison_row(policy_id, controls[policy_id], "outer_walk_forward"))
    for policy_id, result in custom_summaries.items():
        common.append(_comparison_row(policy_id, result, "outer_walk_forward"))
    calibration_trials = []
    for meta in fold_meta:
        for row in meta["inner_candidate_scores"]:
            calibration_trials.append({"fold_id": meta["fold_id"], "trial_type": "calibration_C", **row})
        for model_name, model in meta.get("models", {}).items():
            calibration_trials.append({
                "fold_id": meta["fold_id"], "trial_type": "selected_calibration_model",
                "model": model_name, "selected_C": model["selected_C"],
                "bootstrap_models_completed": model["bootstrap_models_completed"],
                "prediction_interval_mean_width": model["prediction_interval_mean_width"],
            })
    for spec in policy_specs:
        calibration_trials.append({"trial_type": "frozen_outer_policy", **spec})
    trial_frame = pd.DataFrame(calibration_trials)
    fold_frame = pd.DataFrame(fold_rows)
    decisions_frame = pd.DataFrame(decisions)
    trades_frame = pd.DataFrame(trades)
    _write_parquet_atomic(decisions_frame, DECISIONS_PATH)
    _write_parquet_atomic(trades_frame, TRADES_PATH)
    _write_csv_atomic(fold_frame, FOLDS_PATH)
    _write_csv_atomic(trial_frame, TRIALS_PATH)

    trajectory = _trajectory_audit()
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "corrected_position_state_evaluated; complete_local_post_entry_trajectory_available",
        "evaluation_label": config["evaluation_label"],
        "coverage": {**coverage, **row_metadata, "outer_start_utc": folds[0]["evaluation"]["decision_start_utc"], "outer_end_utc": folds[-1]["evaluation"]["decision_end_utc"]},
        "frozen_config": config,
        "input_identity": identity,
        "development_baselines": {key: controls[key] for key in ("legacy_exact_fixed_5_full_development", "legacy_best_level_fixed_5_full_development")},
        "outer_controls": {key: controls[key] for key in ("no_trade", "legacy_best_level_fixed_5", "legacy_exact_fixed_5", "fractional_kelly_half_10pct")},
        "calibration_by_fold": fold_meta,
        "policies": custom_summaries,
        "common_outer_comparison": common,
        "selection_comparison": selection_comparison,
        "sizing_comparison": sizing_comparison,
        "exit_comparison": {
            "hold_to_resolution": "available control; same settlement and +60s release as shared ledger",
            "simple_exit_rule": "trajectory data audited; evaluated in the separate exit-policy continuation run",
            "regression_optimal_stopping": "trajectory data audited; evaluated in the separate exit-policy continuation run",
            "reason": "The local 1-second Kacho top-of-book file covers every eligible market for its full 300-second window; existing PMXT event replay supplies the pre-start interval.",
        },
        "optimal_stopping_data_audit": trajectory,
        "fold_results": fold_rows,
        "artifacts": {
            "config": "configs/research/btc_preopen_advanced_policy_20261007.json",
            "folds": FOLDS_PATH.relative_to(ROOT).as_posix(),
            "trials": TRIALS_PATH.relative_to(ROOT).as_posix(),
            "manifest": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "decisions": DECISIONS_PATH.relative_to(ROOT).as_posix(),
            "trades": TRADES_PATH.relative_to(ROOT).as_posix(),
        },
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "policy_replay_seconds": time.perf_counter() - policy_started,
            "peak_working_set_bytes": _rss_peak_bytes(),
            "cpu_count": __import__("os").cpu_count(),
            "bootstrap_fit_count_requested": sum(int(meta["bootstrap_models_requested_per_fold"]) if "bootstrap_models_requested_per_fold" in meta else int(config["calibration"]["bootstrap"]["replicates"]) * 2 for meta in fold_meta),
            "bootstrap_fit_count_completed": sum(sum(int(model["bootstrap_models_completed"]) for model in meta.get("models", {}).values()) for meta in fold_meta),
            "policies_completed": len(custom_summaries),
            "decision_rows_logged": len(decisions_frame),
            "trade_rows_logged": len(trades_frame),
        },
    }
    _write_json_atomic(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(_make_report(summary), encoding="utf-8")
    try:
        import subprocess
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    artifacts = [CONFIG_PATH, SUMMARY_PATH, FOLDS_PATH, TRIALS_PATH, REPORT_PATH, OPPORTUNITIES_PATH, SOURCE_STUDY_PATH, BASE_CONFIG_PATH]
    manifest = {
        "experiment_id": config["experiment_id"],
        "git_commit_at_run": git_commit,
        "input_identity": identity,
        "python_version": __import__("sys").version,
        "platform": __import__("platform").platform(),
        "elapsed_seconds": summary["execution"]["elapsed_seconds"],
        "peak_working_set_bytes": summary["execution"]["peak_working_set_bytes"],
        "artifacts": [{"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": _sha256(path)} for path in artifacts],
        "external_logs": {
            "decisions_path": DECISIONS_PATH.relative_to(ROOT).as_posix(), "decisions_sha256": _sha256(DECISIONS_PATH),
            "trades_path": TRADES_PATH.relative_to(ROOT).as_posix(), "trades_sha256": _sha256(TRADES_PATH),
            "policy_checkpoint_path": CHECKPOINT_PATH.relative_to(ROOT).as_posix(),
        },
    }
    _write_json_atomic(MANIFEST_PATH, manifest)
    print(f"Completed: {len(custom_summaries)} advanced policies, {len(fold_rows)} fold rows, {len(decisions_frame)} decisions; elapsed={summary['execution']['elapsed_seconds']:.1f}s; peak RSS={summary['execution']['peak_working_set_bytes']}", flush=True)
    for row in common:
        print(f"{row['strategy']}: PnL ${row['pnl_usd']:+.2f}, trades={row['trades']}, maxDD={row['max_drawdown_cost_basis']:.3f}", flush=True)


if __name__ == "__main__":
    _run()
