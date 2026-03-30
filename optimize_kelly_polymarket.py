import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from numba import njit

from kelly_utils import adjust_probability_for_kelly
from modeling_dataset_utils import (
    load_modeling_dataset_settings,
    resolve_oof_prediction_output_paths,
)
from project_config import load_runtime_artifact_paths

optuna.logging.set_verbosity(optuna.logging.INFO)

RUNTIME_ARTIFACT_PATHS = load_runtime_artifact_paths()
MODELING_DATASET_SETTINGS = load_modeling_dataset_settings()


def resolve_kelly_input_path():
    candidate_paths = []
    model_meta_path = Path(RUNTIME_ARTIFACT_PATHS["model_meta_path"])
    if model_meta_path.exists():
        payload = json.loads(model_meta_path.read_text(encoding="utf-8"))
        artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
        if isinstance(artifacts, dict):
            raw_artifact_path = str(artifacts.get("oof_predictions_path", "")).strip()
            if raw_artifact_path:
                artifact_path = Path(raw_artifact_path)
                candidate_paths.append(artifact_path)
                candidate_paths.append(
                    Path(MODELING_DATASET_SETTINGS["modeling_output_dir"])
                    / artifact_path.name
                )
        oof_section = payload.get("oof_predictions") if isinstance(payload, dict) else None
        if isinstance(oof_section, dict):
            raw_meta_path = str(oof_section.get("path", "")).strip()
            if raw_meta_path:
                meta_path = Path(raw_meta_path)
                candidate_paths.append(meta_path)
                candidate_paths.append(
                    Path(MODELING_DATASET_SETTINGS["modeling_output_dir"])
                    / meta_path.name
                )

    candidate_paths.append(
        resolve_oof_prediction_output_paths(
            MODELING_DATASET_SETTINGS,
            preview_rows=1000,
        )["parquet"]
    )

    seen = set()
    for candidate in candidate_paths:
        candidate = Path(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        "Could not resolve Kelly input parquet from runtime model metadata or "
        f"active modeling settings. Searched: {searched}"
    )


# Artifact paths
INPUT_PATH = resolve_kelly_input_path()
RUNTIME_CONFIG_PATH = Path(RUNTIME_ARTIFACT_PATHS["kelly_runtime_config_path"])
OPTUNA_STORAGE = "sqlite:///data/optuna/databases/kelly_polymarket.db"
OPTUNA_OUTPUT_DIR = Path("data/optuna/kelly_polymarket")
STUDY_NAME_PREFIX = "kelly_polymarket_opt_v2"
SIMULATIONS_DIR = Path("data/simulations")

# Input schema
REQUIRED_COLUMNS = [
    "Opened",
    "target_5m_candle_up",
    "oof_pred_proba_up",
    "Close",
]

# Decision cadence and sanity checks
DECISION_INTERVAL_MINUTES = 5
DECISION_MINUTE_OFFSET = 4
TARGET_ALIGNMENT_MIN_MATCH_RATIO = 0.995
MIN_WALK_FORWARD_FOLDS = 10

# Fee, bankroll and probability clipping
FEE_RATE = 0.25
FEE_EXPONENT = 2
FEE_ROUND_DECIMALS = 4
MIN_FEE = 0.0001
MIN_STAKE_USDC = 1.0
PROBABILITY_MIN_CLIP = 1e-6
SIMULATION_START_BANKROLL_USDC = 1000.0

# Optuna search space
TRIAL_PARAM_BOUNDS = {
    "fractional_kelly": (0.01, 1.0),
    "cap": (0.001, 1.0),
    "min_edge": (0.0, 0.15),
    "prob_shrink": (0.0, 1.0),
}

# CV / holdout settings
RANDOM_SEED = 37
N_TRIALS = 2_000
OPTUNA_TPE_STARTUP_TRIALS = int(N_TRIALS * 0.2)
HOLDOUT_FRACTION = 0.05
EXECUTION_SCENARIO_SEEDS = [101, 202, 303, 404, 505]
CV_TARGET_FOLD_DAYS = 35
HOLDOUT_WINDOW_DAYS = 7
HOLDOUT_WINDOW_RUNS = 50
HOLDOUT_WINDOW_SCENARIO_SEEDS = [
    10_000 + idx for idx in range(1, HOLDOUT_WINDOW_RUNS + 1)
]
FULL_HOLDOUT_SCENARIO_SEED = 20_001

# Scenario scoring
SCENARIO_SCORE_WEIGHTS = {
    "mean": 0.6,
    "q10": 0.2,
    "min": 0.1,
    "q90_drawdown_penalty": 0.1,
}

# Manual live-vs-modeling probability error fit used only inside Kelly optimization.
# Update these by hand from the accepted parity audit snapshot.
MODEL_PROBA_ERROR_SIM_ENABLED = False
MODEL_PROBA_ERROR_ABS_DIFF_MEAN = 0.003136730568813267
MODEL_PROBA_ERROR_ABS_DIFF_MAX = 0.026809411202558198
MODEL_PROBA_ERROR_ABS_DIFF_STD = MODEL_PROBA_ERROR_ABS_DIFF_MAX / 3.0
MODEL_PROBA_ERROR_POLICY = (
    "signed_error = random_sign * max(normal(mean_abs_diff, max_abs_diff/3), 0.0); "
    "p_simulated = clip(p_raw + signed_error); prob_shrink applied after simulated model error"
)

SCORING_FORMULA = (
    "fold_score = log(final_bankroll / start_bankroll), "
    "scenario_score = "
    "0.45 * mean(fold_scores) + 0.30 * q10(fold_scores) + 0.15 * min(fold_scores) "
    "- 0.10 * q90(fold_max_drawdowns), "
    "trial_score = "
    "0.45 * mean(scenario_scores) + 0.30 * q10(scenario_scores) + 0.15 * min(scenario_scores), "
    "if total scenario trades == 0: scenario_score = -1e9"
)

# Neutral execution model
NEUTRAL_PRICE_SIM_MODEL = "neutral_conservative_fixed"
NEUTRAL_ORDER_PRICE_CAP = 0.5
NEUTRAL_ORDER_MIN_SIZE = 1.0
NEUTRAL_FIXED_ASK_PRICE = NEUTRAL_ORDER_PRICE_CAP
NEUTRAL_PRICE_POLICY = (
    "market-neutral symmetric ask; both sides always execute at the same "
    "pessimistic fixed price"
)

STEP_LOG_COLUMNS = [
    "Opened",
    "traded",
    "side",
    "price",
    "edge",
    "fraction_f",
    "stake",
    "fee",
    "payout",
    "pnl_usdc",
    "r",
    "g_t",
    "win",
    "win_rate_resolved",
    "win_rate_traded",
    "bankroll_before",
    "bankroll_after",
    "drawdown",
]


def load_oof_frame():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(f"Missing input parquet: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH, columns=REQUIRED_COLUMNS)

    raw_rows = len(df)

    df["Opened"] = pd.to_datetime(df["Opened"], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)

    rows_after_dropna = len(df)

    if rows_after_dropna == 0:

        raise ValueError("No rows left after dropping NA in required columns.")

    df = df.sort_values("Opened").reset_index(drop=True)

    if not df["Opened"].is_monotonic_increasing:

        raise ValueError("Opened must be monotonic increasing after sorting.")

    target_values = df["target_5m_candle_up"].to_numpy(dtype=np.float64, copy=False)

    unique_targets = np.unique(target_values)

    if not np.isin(unique_targets, [0.0, 1.0]).all():

        raise ValueError(
            "target_5m_candle_up must be binary 0/1. "
            f"Found unique values: {unique_targets.tolist()[:20]}"
        )

    p_raw = df["oof_pred_proba_up"].to_numpy(dtype=np.float64, copy=False)

    p_min = float(np.min(p_raw))

    p_max = float(np.max(p_raw))

    if p_min < 0.0 or p_max > 1.0:

        raise ValueError(
            "oof_pred_proba_up must be in [0,1]. "
            f"Found range: min={p_min:.6f}, max={p_max:.6f}"
        )

    print(
        f"load data | path={INPUT_PATH} raw_rows={raw_rows} "
        f"rows_after_dropna={rows_after_dropna}"
    )

    return df


def build_decision_frame(df):

    decision_mask = (df["Opened"].dt.minute % DECISION_INTERVAL_MINUTES) == (
        DECISION_MINUTE_OFFSET
    )

    decision_rows = int(decision_mask.sum())

    df5 = df.loc[decision_mask].reset_index(drop=True)

    if len(df5) < 2:

        raise ValueError(
            "Not enough decision rows after decision-cadence mapping. "
            "Need at least 2 rows in df5."
        )

    print(f"rows | df={len(df)} df5={len(df5)}")

    print(
        "decision mask | "
        f"condition=Opened.minute%{DECISION_INTERVAL_MINUTES}=={DECISION_MINUTE_OFFSET} "
        f"rows_matching={decision_rows}"
    )

    return df5


def sanity_check_target_alignment(df5):

    y_calc = (df5["Close"].shift(-1) >= df5["Close"]).astype(np.int8)

    y_ref = df5["target_5m_candle_up"].astype(np.int8)

    match = y_calc.iloc[:-1].to_numpy() == y_ref.iloc[:-1].to_numpy()

    match_ratio = float(np.mean(match))

    print(
        f"target sanity | match_ratio={match_ratio:.6f} "
        f"threshold={TARGET_ALIGNMENT_MIN_MATCH_RATIO:.3f}"
    )

    if match_ratio < TARGET_ALIGNMENT_MIN_MATCH_RATIO:

        raise ValueError(
            "Target sanity-check failed for 1m->5m boundary mapping: "
            f"match_ratio={match_ratio:.6f} < {TARGET_ALIGNMENT_MIN_MATCH_RATIO:.3f}. "
            "Definitions are inconsistent with boundary-based tie-is-up target."
        )

    return match_ratio


def make_walk_forward_folds(n_obs, n_folds):

    if n_folds < MIN_WALK_FORWARD_FOLDS:

        raise ValueError(
            f"n_folds must be >={MIN_WALK_FORWARD_FOLDS}, got {n_folds}"
        )

    if n_obs <= n_folds:

        raise ValueError(
            f"Too few observations for folds: n_obs={n_obs}, n_folds={n_folds}"
        )

    boundaries = np.linspace(0, n_obs, n_folds + 1, dtype=np.int64)

    folds = []

    for i in range(n_folds):

        start_idx = int(boundaries[i])

        end_idx = int(boundaries[i + 1])

        if end_idx <= start_idx:

            raise ValueError(
                f"Invalid fold range at fold={i}: start={start_idx}, end={end_idx}"
            )

        folds.append((start_idx, end_idx))

    return folds


def infer_n_folds_for_target_days(
    n_obs,
    target_fold_days,
):

    if target_fold_days <= 0:

        raise ValueError(f"target_fold_days must be > 0, got {target_fold_days}")

    decision_rows_per_day = (24 * 60) // DECISION_INTERVAL_MINUTES

    target_rows_per_fold = max(int(target_fold_days * decision_rows_per_day), 1)

    n_folds = max(MIN_WALK_FORWARD_FOLDS, int(round(n_obs / target_rows_per_fold)))

    n_folds = min(n_folds, n_obs - 1)

    approx_fold_days = n_obs / n_folds / decision_rows_per_day

    return n_folds, float(approx_fold_days)


def build_neutral_price_simulation_model():
    ask_price = float(NEUTRAL_FIXED_ASK_PRICE)
    if ask_price <= 0.0 or ask_price >= 1.0:
        raise ValueError(
            "Neutral execution ask price must stay in (0,1). "
            f"Received {ask_price:.6f}."
        )
    return {
        "model": NEUTRAL_PRICE_SIM_MODEL,
        "ask_price": float(ask_price),
        "order_price_cap": float(NEUTRAL_ORDER_PRICE_CAP),
        "order_min_size": float(NEUTRAL_ORDER_MIN_SIZE),
        "policy": str(NEUTRAL_PRICE_POLICY),
    }


def sample_neutral_orderbook_arrays(
    p_raw,
    price_sim_model,
):
    p_raw = np.asarray(p_raw, dtype=np.float64)
    n_rows = len(p_raw)
    ask_price = float(price_sim_model["ask_price"])
    up_ask = np.full(n_rows, ask_price, dtype=np.float64)
    down_ask = np.full(n_rows, ask_price, dtype=np.float64)

    return {
        "up_ask": up_ask,
        "down_ask": down_ask,
        "order_price_cap": np.full(
            n_rows,
            float(price_sim_model["order_price_cap"]),
            dtype=np.float64,
        ),
        "order_min_size": np.full(
            n_rows,
            float(price_sim_model["order_min_size"]),
            dtype=np.float64,
        ),
    }


def sample_model_proba_error_components(
    rng,
    n_rows,
):

    if not MODEL_PROBA_ERROR_SIM_ENABLED:

        return (
            np.zeros(n_rows, dtype=np.float64),
            np.zeros(n_rows, dtype=np.int8),
        )

    model_error_abs_z = rng.standard_normal(n_rows).astype(np.float64, copy=False)

    model_error_sign_bits = rng.integers(
        0,
        2,
        size=n_rows,
        dtype=np.int8,
    )

    return model_error_abs_z, model_error_sign_bits


def build_trial_static_arrays(
    p_raw,
    price_sim_model,
    model_error_abs_z,
    model_error_sign_bits,
):
    sampled_orderbook = sample_neutral_orderbook_arrays(
        p_raw=p_raw,
        price_sim_model=price_sim_model,
    )

    up_price = sampled_orderbook["up_ask"]

    down_price = sampled_orderbook["down_ask"]

    order_price_cap = sampled_orderbook["order_price_cap"]

    order_min_size = sampled_orderbook["order_min_size"]

    up_fee_coef = FEE_RATE * np.power(up_price * (1.0 - up_price), FEE_EXPONENT)

    down_fee_coef = FEE_RATE * np.power(
        down_price * (1.0 - down_price),
        FEE_EXPONENT,
    )

    up_valid = up_fee_coef < 0.99

    down_valid = down_fee_coef < 0.99

    up_c_eff = np.empty_like(up_price, dtype=np.float64)

    down_c_eff = np.empty_like(down_price, dtype=np.float64)

    up_c_eff[up_valid] = up_price[up_valid] / (1.0 - up_fee_coef[up_valid])

    up_c_eff[~up_valid] = np.nan

    down_c_eff[down_valid] = down_price[down_valid] / (
        1.0 - down_fee_coef[down_valid]
    )

    down_c_eff[~down_valid] = np.nan

    if MODEL_PROBA_ERROR_SIM_ENABLED:

        model_error_abs = np.maximum(
            MODEL_PROBA_ERROR_ABS_DIFF_MEAN
            + MODEL_PROBA_ERROR_ABS_DIFF_STD * model_error_abs_z,
            0.0,
        )

        model_error_sign = np.where(model_error_sign_bits == 0, -1.0, 1.0)

        model_proba_error = model_error_sign * model_error_abs

    else:

        model_proba_error = np.zeros_like(p_raw, dtype=np.float64)

    return {
        "up_price": up_price,
        "down_price": down_price,
        "order_price_cap": order_price_cap,
        "order_min_size": order_min_size,
        "up_fee_coef": up_fee_coef,
        "down_fee_coef": down_fee_coef,
        "up_valid": up_valid,
        "down_valid": down_valid,
        "up_c_eff": up_c_eff,
        "down_c_eff": down_c_eff,
        "model_proba_error": model_proba_error,
    }


def build_execution_scenario_static_arrays(
    p_raw_tune,
    p_raw_holdout,
    scenario_seeds,
    price_sim_model,
):

    scenarios = []

    for scenario_seed in scenario_seeds:

        rng = np.random.default_rng(int(scenario_seed))

        model_error_abs_z_tune, model_error_sign_bits_tune = (
            sample_model_proba_error_components(rng, len(p_raw_tune))
        )

        model_error_abs_z_holdout, model_error_sign_bits_holdout = (
            sample_model_proba_error_components(rng, len(p_raw_holdout))
        )

        scenarios.append(
            {
                "seed": int(scenario_seed),
                "tune_static_arrays": build_trial_static_arrays(
                    p_raw_tune,
                    price_sim_model,
                    model_error_abs_z_tune,
                    model_error_sign_bits_tune,
                ),
                "holdout_static_arrays": build_trial_static_arrays(
                    p_raw_holdout,
                    price_sim_model,
                    model_error_abs_z_holdout,
                    model_error_sign_bits_holdout,
                ),
            }
        )

    return scenarios


def build_single_execution_static_arrays(
    p_raw,
    scenario_seed,
    price_sim_model,
):

    rng = np.random.default_rng(int(scenario_seed))

    model_error_abs_z, model_error_sign_bits = sample_model_proba_error_components(
        rng,
        len(p_raw),
    )

    return build_trial_static_arrays(
        p_raw,
        price_sim_model,
        model_error_abs_z,
        model_error_sign_bits,
    )


def build_trial_arrays(
    p_raw,
    static_arrays,
    fractional_kelly,
    cap,
    min_edge,
    prob_shrink,
):

    p_simulated = np.clip(
        p_raw + static_arrays["model_proba_error"],
        PROBABILITY_MIN_CLIP,
        1.0 - PROBABILITY_MIN_CLIP,
    )

    p = adjust_probability_for_kelly(
        p_simulated,
        prob_shrink=prob_shrink,
        min_clip=PROBABILITY_MIN_CLIP,
    )

    up_price = static_arrays["up_price"]

    down_price = static_arrays["down_price"]

    up_fee_coef = static_arrays["up_fee_coef"]

    down_fee_coef = static_arrays["down_fee_coef"]

    up_valid = static_arrays["up_valid"]

    down_valid = static_arrays["down_valid"]

    up_c_eff = static_arrays["up_c_eff"]

    down_c_eff = static_arrays["down_c_eff"]

    edge_up = np.where(up_valid, p - up_c_eff, -np.inf)

    edge_down = np.where(down_valid, (1.0 - p) - down_c_eff, -np.inf)

    choose_up = edge_up >= edge_down

    selected_edge = np.where(choose_up, edge_up, edge_down)

    selected_valid = np.where(choose_up, up_valid, down_valid)

    selected_price = np.where(choose_up, up_price, down_price)

    selected_fee_coef = np.where(choose_up, up_fee_coef, down_fee_coef)

    selected_c_eff = np.where(choose_up, up_c_eff, down_c_eff)

    selected_order_min_size = static_arrays["order_min_size"]

    selected_order_price_cap = static_arrays["order_price_cap"]

    can_trade = (
        selected_valid
        & (selected_edge >= min_edge)
        & np.isfinite(selected_price)
        & np.isfinite(selected_order_price_cap)
        & (selected_price > 0.0)
        & (selected_price < 1.0)
        & (selected_price <= selected_order_price_cap)
    )

    p_side = np.where(choose_up, p, 1.0 - p)

    with np.errstate(divide="ignore", invalid="ignore"):

        f_star = (p_side - selected_c_eff) / (1.0 - selected_c_eff)

    f_star = np.where(can_trade, np.maximum(f_star, 0.0), 0.0)

    f = np.minimum(cap, fractional_kelly * f_star)

    f = np.where(np.isfinite(f), np.maximum(f, 0.0), 0.0)

    return {
        "price": selected_price,
        "fee_coef": selected_fee_coef,
        "order_min_size": selected_order_min_size,
        "can_trade": can_trade,
        "choose_up": choose_up,
        "edge": selected_edge,
        "f": f,
        "up_price": up_price,
        "down_price": down_price,
    }


@njit(cache=True, nogil=True)
def _simulate_segment_fast_numba(
    target,
    price,
    fee_coef,
    order_min_size_arr,
    can_trade,
    choose_up,
    f_arr,
    start_idx,
    end_idx,
    min_stake_usdc,
    start_bankroll,
):

    bankroll = float(start_bankroll)

    equity_peak = bankroll

    max_drawdown = 0.0

    trades = 0

    sum_g = 0.0

    sum_fraction = 0.0

    n_steps = end_idx - start_idx

    for i in range(start_idx, end_idx):

        bankroll_before = bankroll

        if can_trade[i] and f_arr[i] > 0.0:

            stake = bankroll_before * f_arr[i]

            if stake >= min_stake_usdc:

                fee = float(np.round(stake * fee_coef[i], FEE_ROUND_DECIMALS))

                if fee < MIN_FEE:

                    fee = 0.0

                if fee < stake:

                    shares_net = (stake - fee) / price[i]

                    if shares_net < order_min_size_arr[i]:

                        continue

                    if choose_up[i]:

                        win = target[i] == 1

                    else:

                        win = target[i] == 0

                    if win:

                        payout = (stake - fee) / price[i]

                    else:

                        payout = 0.0

                    bankroll = bankroll_before - stake + payout

                    if (not np.isfinite(bankroll)) or bankroll <= 0.0:

                        return 1, 0.0, 0, 0.0, 0.0, 0.0, 0

                    trades += 1

                    sum_g += np.log(bankroll / bankroll_before)

                    sum_fraction += f_arr[i]

        if bankroll > equity_peak:

            equity_peak = bankroll

        elif equity_peak > 0.0:

            drawdown = (equity_peak - bankroll) / equity_peak

            if drawdown > max_drawdown:

                max_drawdown = drawdown

    return 0, bankroll, trades, max_drawdown, sum_g, sum_fraction, n_steps


def simulate_segment_fast(
    target,
    price,
    fee_coef,
    order_min_size_arr,
    can_trade,
    choose_up,
    f_arr,
    start_idx,
    end_idx,
    min_stake_usdc,
    start_bankroll=SIMULATION_START_BANKROLL_USDC,
):

    overflowed, final_bankroll, trades, max_drawdown, sum_g, sum_fraction, n_steps = (
        _simulate_segment_fast_numba(
            target=target,
            price=price,
            fee_coef=fee_coef,
            order_min_size_arr=order_min_size_arr,
            can_trade=can_trade,
            choose_up=choose_up,
            f_arr=f_arr,
            start_idx=start_idx,
            end_idx=end_idx,
            min_stake_usdc=min_stake_usdc,
            start_bankroll=start_bankroll,
        )
    )

    if overflowed:

        raise FloatingPointError("Non-finite bankroll encountered during simulation.")

    return {
        "final_bankroll": float(final_bankroll),
        "n_trades": int(trades),
        "max_drawdown": float(max_drawdown),
        "sum_g": float(sum_g),
        "avg_fraction": float(sum_fraction / trades) if trades > 0 else 0.0,
        "n_steps": int(n_steps),
    }


def simulate_segment_trace(
    target,
    trial_arrays,
    start_idx,
    end_idx,
    min_stake_usdc,
    opened,
    scenario_seed,
    start_bankroll=SIMULATION_START_BANKROLL_USDC,
):

    price = trial_arrays["price"]

    fee_coef = trial_arrays["fee_coef"]

    can_trade = trial_arrays["can_trade"]

    choose_up = trial_arrays["choose_up"]

    order_min_size_arr = trial_arrays["order_min_size"]

    edge = trial_arrays["edge"]

    f_arr = trial_arrays["f"]

    bankroll = float(start_bankroll)

    equity_peak = bankroll

    max_drawdown = 0.0

    trades = 0

    wins = 0

    resolved = 0

    resolved_wins = 0

    traded_wins = 0

    sum_g = 0.0

    sum_edge = 0.0

    sum_stake = 0.0

    sum_fraction = 0.0

    n_steps = end_idx - start_idx

    step_log = {col: [] for col in STEP_LOG_COLUMNS}

    for i in range(start_idx, end_idx):

        bankroll_before = float(bankroll)

        price_i = float(price[i])

        edge_i = float(edge[i])

        if not np.isfinite(edge_i):

            edge_i = float("nan")

        f_i = float(f_arr[i])

        if not np.isfinite(f_i):

            f_i = float("nan")

        traded = False

        side = "NONE"

        stake = 0.0

        fee = 0.0

        payout = 0.0

        pnl_usdc = 0.0

        r = None

        g_t = 0.0

        win = None

        # Track directional accuracy on every resolved holdout row, not just trades.

        signal_up = bool(choose_up[i])

        resolved_win = bool(target[i] == 1) if signal_up else bool(target[i] == 0)

        if can_trade[i] and f_i > 0.0:

            stake = bankroll_before * f_i

            if stake >= min_stake_usdc:

                fee = round(stake * float(fee_coef[i]), FEE_ROUND_DECIMALS)

                if fee < MIN_FEE:

                    fee = 0.0

                if fee < stake:

                    shares_net = (stake - fee) / price_i

                    if shares_net >= float(order_min_size_arr[i]):

                        traded = True

                        side = "UP" if signal_up else "DOWN"

                        win = resolved_win

                        payout = shares_net if win else 0.0

                        bankroll = bankroll_before - stake + payout

                        if (not np.isfinite(bankroll)) or bankroll <= 0.0:

                            raise FloatingPointError(
                                "Non-finite bankroll encountered during simulation."
                            )

                        pnl_usdc = payout - stake

                        r = pnl_usdc / stake

                        g_t = float(np.log(bankroll / bankroll_before))

                        trades += 1

                        wins += int(win)

                        traded_wins += int(win)

                        sum_g += g_t

                        sum_edge += edge_i

                        sum_stake += stake

                        sum_fraction += f_i

        resolved += 1

        resolved_wins += int(resolved_win)

        win_rate_resolved = float(resolved_wins / resolved)

        win_rate_traded = float(traded_wins / trades) if trades > 0 else float("nan")

        if bankroll > equity_peak:

            equity_peak = bankroll

        drawdown = 0.0

        if equity_peak > 0.0:

            drawdown = (equity_peak - bankroll) / equity_peak

            if drawdown > max_drawdown:

                max_drawdown = drawdown

        step_log["Opened"].append(pd.Timestamp(opened[i]))

        step_log["traded"].append(bool(traded))

        step_log["side"].append(side)

        step_log["price"].append(float(price_i))

        step_log["edge"].append(float(edge_i))

        step_log["fraction_f"].append(float(f_i))

        step_log["stake"].append(float(stake))

        step_log["fee"].append(float(fee))

        step_log["payout"].append(float(payout))

        step_log["pnl_usdc"].append(float(pnl_usdc))

        step_log["r"].append(None if r is None else float(r))

        step_log["g_t"].append(float(g_t))

        step_log["win"].append(win)

        step_log["win_rate_resolved"].append(float(win_rate_resolved))

        step_log["win_rate_traded"].append(float(win_rate_traded))

        step_log["bankroll_before"].append(float(bankroll_before))

        step_log["bankroll_after"].append(float(bankroll))

        step_log["drawdown"].append(float(drawdown))

    if trades > 0:

        hit_rate = wins / trades

        avg_edge = sum_edge / trades

        avg_stake = sum_stake / trades

    else:

        hit_rate = 0.0

        avg_edge = 0.0

        avg_stake = 0.0

    avg_fraction = sum_fraction / trades if trades > 0 else 0.0

    mean_g = sum_g / n_steps if n_steps > 0 else 0.0

    fold_score = float(sum_g)

    result = {
        "final_bankroll": float(bankroll),
        "n_steps": int(n_steps),
        "trades": int(trades),
        "wins": int(wins),
        "hit_rate": float(hit_rate),
        "sum_g": float(sum_g),
        "mean_g": float(mean_g),
        "fold_score": float(fold_score),
        "avg_edge": float(avg_edge),
        "avg_stake": float(avg_stake),
        "avg_fraction": float(avg_fraction),
        "max_drawdown": float(max_drawdown),
        "scenario_seed": int(scenario_seed),
        "step_log": step_log,
    }

    return result


def evaluate_cv_folds_for_scenario(
    target,
    trial_arrays,
    folds,
    min_stake_usdc,
):

    fold_scores = []

    fold_trades = []

    fold_log_growth = []

    fold_max_drawdown = []

    fold_avg_fraction = []

    price = trial_arrays["price"]

    fee_coef = trial_arrays["fee_coef"]

    order_min_size_arr = trial_arrays["order_min_size"]

    can_trade = trial_arrays["can_trade"]

    choose_up = trial_arrays["choose_up"]

    f_arr = trial_arrays["f"]

    for start_idx, end_idx in folds:

        segment_result = simulate_segment_fast(
            target=target,
            price=price,
            fee_coef=fee_coef,
            order_min_size_arr=order_min_size_arr,
            can_trade=can_trade,
            choose_up=choose_up,
            f_arr=f_arr,
            start_idx=start_idx,
            end_idx=end_idx,
            min_stake_usdc=min_stake_usdc,
            start_bankroll=SIMULATION_START_BANKROLL_USDC,
        )

        mean_g = float(segment_result["sum_g"]) / int(segment_result["n_steps"])

        fold_score = float(segment_result["sum_g"])

        fold_scores.append(fold_score)

        fold_trades.append(int(segment_result["n_trades"]))

        fold_log_growth.append(mean_g)

        fold_max_drawdown.append(float(segment_result["max_drawdown"]))

        fold_avg_fraction.append(float(segment_result["avg_fraction"]))

    fold_scores_arr = np.asarray(fold_scores, dtype=np.float64)

    fold_avg_fraction_arr = np.asarray(fold_avg_fraction, dtype=np.float64)

    scenario_score = float(
        SCENARIO_SCORE_WEIGHTS["mean"] * np.mean(fold_scores_arr)
        + SCENARIO_SCORE_WEIGHTS["q10"] * np.quantile(fold_scores_arr, 0.10)
        + SCENARIO_SCORE_WEIGHTS["min"] * np.min(fold_scores_arr)
        - SCENARIO_SCORE_WEIGHTS["q90_drawdown_penalty"]
        * np.quantile(np.asarray(fold_max_drawdown, dtype=np.float64), 0.90)
    )

    if sum(fold_trades) == 0:

        scenario_score = -1e9

    return {
        "score": scenario_score,
        "fold_scores": fold_scores,
        "fold_trades": fold_trades,
        "fold_log_growth": fold_log_growth,
        "fold_max_drawdown": fold_max_drawdown,
        "fold_avg_fraction": fold_avg_fraction,
        "mean_fold_avg_fraction": float(np.mean(fold_avg_fraction_arr)),
        "q90_fold_avg_fraction": float(np.quantile(fold_avg_fraction_arr, 0.90)),
    }


def evaluate_trial_across_execution_scenarios(
    target,
    p_raw,
    scenario_static_arrays,
    folds,
    fractional_kelly,
    cap,
    min_edge,
    prob_shrink,
    min_stake_usdc,
):

    scenario_scores = []

    fold_scores = []

    fold_trades = []

    fold_log_growth = []

    fold_max_drawdown = []

    fold_avg_fraction = []

    for scenario in scenario_static_arrays:

        trial_arrays = build_trial_arrays(
            p_raw=p_raw,
            static_arrays=scenario["tune_static_arrays"],
            fractional_kelly=fractional_kelly,
            cap=cap,
            min_edge=min_edge,
            prob_shrink=prob_shrink,
        )

        scenario_result = evaluate_cv_folds_for_scenario(
            target=target,
            trial_arrays=trial_arrays,
            folds=folds,
            min_stake_usdc=min_stake_usdc,
        )

        scenario_scores.append(float(scenario_result["score"]))

        fold_scores.extend(scenario_result["fold_scores"])

        fold_trades.extend(scenario_result["fold_trades"])

        fold_log_growth.extend(scenario_result["fold_log_growth"])

        fold_max_drawdown.extend(scenario_result["fold_max_drawdown"])

        fold_avg_fraction.extend(scenario_result["fold_avg_fraction"])

    scenario_scores_arr = np.asarray(scenario_scores, dtype=np.float64)

    fold_scores_arr = np.asarray(fold_scores, dtype=np.float64)

    fold_trades_arr = np.asarray(fold_trades, dtype=np.float64)

    fold_log_growth_arr = np.asarray(fold_log_growth, dtype=np.float64)

    fold_max_drawdown_arr = np.asarray(fold_max_drawdown, dtype=np.float64)

    fold_avg_fraction_arr = np.asarray(fold_avg_fraction, dtype=np.float64)

    return {
        "score": float(
            SCENARIO_SCORE_WEIGHTS["mean"] * np.mean(scenario_scores_arr)
            + SCENARIO_SCORE_WEIGHTS["q10"]
            * np.quantile(scenario_scores_arr, 0.10)
            + SCENARIO_SCORE_WEIGHTS["min"] * np.min(scenario_scores_arr)
        ),
        "mean_scenario_score": float(np.mean(scenario_scores_arr)),
        "q10_scenario_score": float(np.quantile(scenario_scores_arr, 0.10)),
        "min_scenario_score": float(np.min(scenario_scores_arr)),
        "mean_fold_score": float(np.mean(fold_scores_arr)),
        "q10_fold_score": float(np.quantile(fold_scores_arr, 0.10)),
        "min_fold_score": float(np.min(fold_scores_arr)),
        "q25_fold_score": float(np.quantile(fold_scores_arr, 0.25)),
        "mean_fold_trades": float(np.mean(fold_trades_arr)),
        "mean_fold_log_growth": float(np.mean(fold_log_growth_arr)),
        "mean_fold_max_drawdown": float(np.mean(fold_max_drawdown_arr)),
        "q90_fold_max_drawdown": float(np.quantile(fold_max_drawdown_arr, 0.90)),
        "mean_fold_avg_fraction": float(np.mean(fold_avg_fraction_arr)),
        "q90_fold_avg_fraction": float(np.quantile(fold_avg_fraction_arr, 0.90)),
    }


def suggest_trial_params(trial):
    return {
        name: float(trial.suggest_float(name, bounds[0], bounds[1]))
        for name, bounds in TRIAL_PARAM_BOUNDS.items()
    }


def build_runtime_config(
    fractional_kelly,
    cap,
    min_edge,
    prob_shrink,
    price_sim_model,
):

    return {
        "fractional_kelly": float(fractional_kelly),
        "cap": float(cap),
        "min_edge": float(min_edge),
        "prob_shrink": float(prob_shrink),
        "min_stake_usdc": float(MIN_STAKE_USDC),
        "fee_model": {
            "feeRate": float(FEE_RATE),
            "exponent": float(FEE_EXPONENT),
            "fee_round_decimals": int(FEE_ROUND_DECIMALS),
            "min_fee": float(MIN_FEE),
        },
        "price_sim": {
            "model": str(price_sim_model["model"]),
            "ask_price": float(price_sim_model["ask_price"]),
            "order_price_cap": float(price_sim_model["order_price_cap"]),
            "order_min_size": float(price_sim_model["order_min_size"]),
            "policy": str(price_sim_model["policy"]),
        },
        "model_proba_error_sim": {
            "enabled": bool(MODEL_PROBA_ERROR_SIM_ENABLED),
            "abs_diff_mean": float(MODEL_PROBA_ERROR_ABS_DIFF_MEAN),
            "abs_diff_max": float(MODEL_PROBA_ERROR_ABS_DIFF_MAX),
            "abs_diff_std": float(MODEL_PROBA_ERROR_ABS_DIFF_STD),
            "policy": (
                MODEL_PROBA_ERROR_POLICY
                if MODEL_PROBA_ERROR_SIM_ENABLED
                else "disabled"
            ),
            "prob_min_clip": float(PROBABILITY_MIN_CLIP),
        },
        "cv_meta": {
            "seed": int(RANDOM_SEED),
            "scoring_formula": SCORING_FORMULA,
        },
    }


def save_json(path, payload):

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:

        json.dump(payload, f, indent=2)


def build_holdout_window_specs(
    opened_holdout,
    window_days,
    n_runs,
    scenario_seeds,
):

    if len(scenario_seeds) != n_runs:

        raise ValueError(
            "Holdout window scenario seeds count must match holdout window run count."
        )

    if opened_holdout.ndim != 1 or len(opened_holdout) == 0:

        raise ValueError("Holdout opened array must be a non-empty 1D array.")

    window_delta = np.timedelta64(int(window_days), "D")

    latest_start_time = opened_holdout[-1] - window_delta

    max_start_idx = int(
        np.searchsorted(opened_holdout, latest_start_time, side="right") - 1
    )

    if max_start_idx < 0:

        raise ValueError(
            f"Holdout too short for {window_days} day windows. "
            f"holdout_start={pd.Timestamp(opened_holdout[0]).isoformat()} "
            f"holdout_end={pd.Timestamp(opened_holdout[-1]).isoformat()}"
        )

    if max_start_idx + 1 < n_runs:

        raise ValueError(
            f"Not enough distinct holdout starts for {n_runs} windows of {window_days} days. "
            f"available_starts={max_start_idx + 1}"
        )

    raw_start_positions = np.linspace(0, max_start_idx, num=n_runs)

    start_indices = []

    next_min_idx = 0

    for run_idx, raw_position in enumerate(raw_start_positions):

        remaining_runs = n_runs - run_idx - 1

        max_allowed_idx = max_start_idx - remaining_runs

        start_idx = int(round(float(raw_position)))

        start_idx = max(start_idx, next_min_idx)

        start_idx = min(start_idx, max_allowed_idx)

        start_indices.append(start_idx)

        next_min_idx = start_idx + 1

    specs = []

    for run_idx, start_idx in enumerate(start_indices, start=1):

        start_time = opened_holdout[start_idx]

        end_idx = int(
            np.searchsorted(opened_holdout, start_time + window_delta, side="left")
        )

        if end_idx <= start_idx:

            raise ValueError(
                f"Invalid holdout window for run={run_idx}: start_idx={start_idx}, end_idx={end_idx}"
            )

        specs.append(
            {
                "window_id": int(run_idx),
                "scenario_seed": int(scenario_seeds[run_idx - 1]),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "start_opened": pd.Timestamp(opened_holdout[start_idx]).isoformat(),
                "end_opened": pd.Timestamp(opened_holdout[end_idx - 1]).isoformat(),
            }
        )

    return specs


def summarize_holdout_results(results):

    scores = np.asarray(
        [float(result["fold_score"]) for result in results], dtype=np.float64
    )

    final_bankrolls = np.asarray(
        [float(result["final_bankroll"]) for result in results],
        dtype=np.float64,
    )

    trades = np.asarray([int(result["trades"]) for result in results], dtype=np.int64)

    hit_rates = np.asarray(
        [float(result["hit_rate"]) for result in results], dtype=np.float64
    )

    avg_edges = np.asarray(
        [float(result["avg_edge"]) for result in results], dtype=np.float64
    )

    avg_stakes = np.asarray(
        [float(result["avg_stake"]) for result in results], dtype=np.float64
    )

    avg_fractions = np.asarray(
        [float(result["avg_fraction"]) for result in results], dtype=np.float64
    )

    mean_g = np.asarray(
        [float(result["mean_g"]) for result in results], dtype=np.float64
    )

    max_drawdowns = np.asarray(
        [float(result["max_drawdown"]) for result in results],
        dtype=np.float64,
    )

    return {
        "score": float(0.75 * np.mean(scores) + 0.25 * np.min(scores)),
        "mean_final_bankroll": float(np.mean(final_bankrolls)),
        "worst_final_bankroll": float(np.min(final_bankrolls)),
        "total_pnl": float(
            np.mean(final_bankrolls) - SIMULATION_START_BANKROLL_USDC
        ),
        "mean_n_trades": float(np.mean(trades)),
        "min_n_trades": int(np.min(trades)),
        "mean_hit_rate": float(np.mean(hit_rates)),
        "mean_avg_edge": float(np.mean(avg_edges)),
        "mean_avg_stake": float(np.mean(avg_stakes)),
        "mean_avg_fraction": float(np.mean(avg_fractions)),
        "mean_log_growth": float(np.mean(mean_g)),
        "worst_log_growth": float(np.min(mean_g)),
        "mean_max_drawdown": float(np.mean(max_drawdowns)),
        "worst_max_drawdown": float(np.max(max_drawdowns)),
    }


def main():

    print("optimize kelly polymarket | start")

    run_started_at_utc = datetime.now(timezone.utc)

    run_timestamp = run_started_at_utc.strftime("%Y%m%d_%H%M%S")

    holdout_trace_dir = SIMULATIONS_DIR / f"holdout_trace_{run_timestamp}"

    trials_csv_path = OPTUNA_OUTPUT_DIR / f"kelly_polymarket_trials_{run_timestamp}.csv"

    run_summary_path = OPTUNA_OUTPUT_DIR / f"kelly_polymarket_run_{run_timestamp}.json"

    df = load_oof_frame()

    df5 = build_decision_frame(df)

    match_ratio = sanity_check_target_alignment(df5)

    p_raw_stats = df5["oof_pred_proba_up"].to_numpy(dtype=np.float64, copy=False)

    print(
        "pred stats | p_raw "
        f"min={float(np.min(p_raw_stats)):.6f} "
        f"mean={float(np.mean(p_raw_stats)):.6f} "
        f"max={float(np.max(p_raw_stats)):.6f}"
    )

    print(
        "model error sim | "
        f"enabled={MODEL_PROBA_ERROR_SIM_ENABLED} "
        f"abs_mean={MODEL_PROBA_ERROR_ABS_DIFF_MEAN:.6f} "
        f"abs_max={MODEL_PROBA_ERROR_ABS_DIFF_MAX:.6f} "
        f"abs_std={MODEL_PROBA_ERROR_ABS_DIFF_STD:.6f} "
        f"policy={MODEL_PROBA_ERROR_POLICY if MODEL_PROBA_ERROR_SIM_ENABLED else 'disabled'}"
    )

    price_sim_model = build_neutral_price_simulation_model()

    print(
        "price sim | "
        f"model={price_sim_model['model']} "
        f"ask_price={price_sim_model['ask_price']:.6f} "
        f"order_price_cap={price_sim_model['order_price_cap']:.3f} "
        f"order_min_size={price_sim_model['order_min_size']:.4f} "
        f"policy={price_sim_model['policy']}"
    )

    n_decision_rows = len(df5)

    n_trades = n_decision_rows - 1

    if n_trades <= 0:

        raise ValueError("Not enough decision rows to create trades.")

    target = df5["target_5m_candle_up"].to_numpy(dtype=np.int8, copy=False)[:-1]

    p_raw = p_raw_stats[:-1]

    opened_trade = df5["Opened"].iloc[:-1].to_numpy(dtype="datetime64[ns]", copy=False)

    split_idx = int(len(target) * (1.0 - HOLDOUT_FRACTION))

    if split_idx <= 0 or split_idx >= len(target):

        raise ValueError(
            f"Invalid holdout split: split_idx={split_idx}, total_rows={len(target)}"
        )

    target_tune = target[:split_idx]

    p_raw_tune = p_raw[:split_idx]

    target_holdout = target[split_idx:]

    p_raw_holdout = p_raw[split_idx:]

    opened_holdout = opened_trade[split_idx:]

    execution_scenarios = build_execution_scenario_static_arrays(
        p_raw_tune=p_raw_tune,
        p_raw_holdout=p_raw_holdout,
        scenario_seeds=EXECUTION_SCENARIO_SEEDS,
        price_sim_model=price_sim_model,
    )

    n_folds, approx_fold_days = infer_n_folds_for_target_days(
        n_obs=len(target_tune),
        target_fold_days=CV_TARGET_FOLD_DAYS,
    )

    folds = make_walk_forward_folds(n_obs=len(target_tune), n_folds=n_folds)

    print(
        f"cv setup | n_folds={n_folds} approx_fold_days={approx_fold_days:.2f} "
        f"target_fold_days={CV_TARGET_FOLD_DAYS} n_trials={N_TRIALS} "
        f"n_execution_scenarios={len(execution_scenarios)} "
        f"n_trades_tune={len(target_tune)} "
        f"first_fold=[{folds[0][0]}:{folds[0][1]}]"
    )

    print(
        f"data split | holdout_frac={HOLDOUT_FRACTION:.3f} split_idx={split_idx} "
        f"tune_rows={len(target_tune)} holdout_rows={len(target_holdout)} "
        f"holdout_start_opened={pd.Timestamp(opened_holdout[0]).isoformat()} "
        f"holdout_end_opened={pd.Timestamp(opened_holdout[-1]).isoformat()}"
    )

    print(f"execution scenarios | seeds={EXECUTION_SCENARIO_SEEDS}")

    def objective(trial):
        trial_params = suggest_trial_params(trial)

        cv_result = evaluate_trial_across_execution_scenarios(
            target=target_tune,
            p_raw=p_raw_tune,
            scenario_static_arrays=execution_scenarios,
            folds=folds,
            fractional_kelly=trial_params["fractional_kelly"],
            cap=trial_params["cap"],
            min_edge=trial_params["min_edge"],
            prob_shrink=trial_params["prob_shrink"],
            min_stake_usdc=MIN_STAKE_USDC,
        )

        return float(cv_result["score"])

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_SEED,
        n_startup_trials=OPTUNA_TPE_STARTUP_TRIALS,
    )

    study_name = f"{STUDY_NAME_PREFIX}_{run_timestamp}"

    print(
        "optuna setup | "
        f"n_trials={N_TRIALS} folds={len(folds)} "
        f"n_execution_scenarios={len(execution_scenarios)} "
        f"study={study_name} storage={OPTUNA_STORAGE}"
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=OPTUNA_STORAGE,
        load_if_exists=False,
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        catch=(FloatingPointError, OverflowError),
    )

    best_trial = study.best_trial

    best_params = best_trial.params

    print(
        f"optuna done | best_score={best_trial.value:.6f} "
        f"trial={best_trial.number} params={best_params}"
    )

    best_fractional_kelly = float(best_params["fractional_kelly"])

    best_cap = float(best_params["cap"])

    best_min_edge = float(best_params["min_edge"])

    best_prob_shrink = float(best_params["prob_shrink"])

    best_cv_result = evaluate_trial_across_execution_scenarios(
        target=target_tune,
        p_raw=p_raw_tune,
        scenario_static_arrays=execution_scenarios,
        folds=folds,
        fractional_kelly=best_fractional_kelly,
        cap=best_cap,
        min_edge=best_min_edge,
        prob_shrink=best_prob_shrink,
        min_stake_usdc=MIN_STAKE_USDC,
    )

    holdout_window_specs = build_holdout_window_specs(
        opened_holdout=opened_holdout,
        window_days=HOLDOUT_WINDOW_DAYS,
        n_runs=HOLDOUT_WINDOW_RUNS,
        scenario_seeds=HOLDOUT_WINDOW_SCENARIO_SEEDS,
    )

    print(
        "holdout setup | "
        f"window_days={HOLDOUT_WINDOW_DAYS} "
        f"window_runs={HOLDOUT_WINDOW_RUNS} "
        f"full_run_seed={FULL_HOLDOUT_SCENARIO_SEED}"
    )

    min_price = float("inf")

    max_price = float("-inf")

    holdout_window_results = []

    holdout_window_runs_output = []

    holdout_trace_dir.mkdir(parents=True, exist_ok=True)

    holdout_static_arrays_by_seed = {}

    for scenario_seed in [*HOLDOUT_WINDOW_SCENARIO_SEEDS, FULL_HOLDOUT_SCENARIO_SEED]:

        static_arrays = build_single_execution_static_arrays(
            p_raw=p_raw_holdout,
            scenario_seed=int(scenario_seed),
            price_sim_model=price_sim_model,
        )

        holdout_static_arrays_by_seed[int(scenario_seed)] = static_arrays

        min_price = min(
            min_price,
            float(np.min(static_arrays["up_price"])),
            float(np.min(static_arrays["down_price"])),
        )

        max_price = max(
            max_price,
            float(np.max(static_arrays["up_price"])),
            float(np.max(static_arrays["down_price"])),
        )

    holdout_trace_csv_by_window_id = {}

    for window_spec in holdout_window_specs:

        scenario_seed = int(window_spec["scenario_seed"])

        best_arrays_holdout = build_trial_arrays(
            p_raw=p_raw_holdout,
            static_arrays=holdout_static_arrays_by_seed[scenario_seed],
            fractional_kelly=best_fractional_kelly,
            cap=best_cap,
            min_edge=best_min_edge,
            prob_shrink=best_prob_shrink,
        )

        holdout_result = simulate_segment_trace(
            target=target_holdout,
            trial_arrays=best_arrays_holdout,
            start_idx=int(window_spec["start_idx"]),
            end_idx=int(window_spec["end_idx"]),
            min_stake_usdc=MIN_STAKE_USDC,
            opened=opened_holdout,
            scenario_seed=scenario_seed,
            start_bankroll=SIMULATION_START_BANKROLL_USDC,
        )

        holdout_trace_path = holdout_trace_dir / (
            f"window_{int(window_spec['window_id']):02d}_seed_{scenario_seed}.csv"
        )

        pd.DataFrame(holdout_result["step_log"]).to_csv(
            holdout_trace_path,
            index=False,
            columns=STEP_LOG_COLUMNS,
        )

        holdout_trace_csv_by_window_id[str(int(window_spec["window_id"]))] = str(
            holdout_trace_path
        )

        holdout_window_results.append(holdout_result)

        holdout_window_runs_output.append(
            {
                "window_id": int(window_spec["window_id"]),
                "window_days": int(HOLDOUT_WINDOW_DAYS),
                "scenario_seed": scenario_seed,
                "start_idx": int(window_spec["start_idx"]),
                "end_idx": int(window_spec["end_idx"]),
                "start_opened": str(window_spec["start_opened"]),
                "end_opened": str(window_spec["end_opened"]),
                "score": float(holdout_result["fold_score"]),
                "final_bankroll": float(holdout_result["final_bankroll"]),
                "n_steps": int(holdout_result["n_steps"]),
                "n_trades": int(holdout_result["trades"]),
                "mean_log_growth": float(holdout_result["mean_g"]),
                "max_drawdown": float(holdout_result["max_drawdown"]),
                "hit_rate": float(holdout_result["hit_rate"]),
                "avg_edge": float(holdout_result["avg_edge"]),
                "avg_stake": float(holdout_result["avg_stake"]),
                "avg_fraction": float(holdout_result["avg_fraction"]),
                "holdout_trace_csv": str(holdout_trace_path),
            }
        )

    full_holdout_arrays = build_trial_arrays(
        p_raw=p_raw_holdout,
        static_arrays=holdout_static_arrays_by_seed[FULL_HOLDOUT_SCENARIO_SEED],
        fractional_kelly=best_fractional_kelly,
        cap=best_cap,
        min_edge=best_min_edge,
        prob_shrink=best_prob_shrink,
    )

    full_holdout_result = simulate_segment_trace(
        target=target_holdout,
        trial_arrays=full_holdout_arrays,
        start_idx=0,
        end_idx=len(target_holdout),
        min_stake_usdc=MIN_STAKE_USDC,
        opened=opened_holdout,
        scenario_seed=FULL_HOLDOUT_SCENARIO_SEED,
        start_bankroll=SIMULATION_START_BANKROLL_USDC,
    )

    full_holdout_trace_path = (
        holdout_trace_dir / f"full_holdout_seed_{FULL_HOLDOUT_SCENARIO_SEED}.csv"
    )

    pd.DataFrame(full_holdout_result["step_log"]).to_csv(
        full_holdout_trace_path,
        index=False,
        columns=STEP_LOG_COLUMNS,
    )

    print(
        "price sanity | "
        f"min_price={min_price:.6f} "
        f"max_price={max_price:.6f} "
        f"price_sim_model={price_sim_model['model']}"
    )

    if min_price <= 0.0 or max_price >= 1.0:

        raise ValueError(
            "Invalid price construction: empirical sampled ask prices must stay in (0,1)."
        )

    holdout_window_summary = summarize_holdout_results(holdout_window_results)

    full_holdout_output = {
        "scenario_seed": int(FULL_HOLDOUT_SCENARIO_SEED),
        "start_idx": 0,
        "end_idx": len(target_holdout),
        "start_opened": pd.Timestamp(opened_holdout[0]).isoformat(),
        "end_opened": pd.Timestamp(opened_holdout[-1]).isoformat(),
        "score": float(full_holdout_result["fold_score"]),
        "final_bankroll": float(full_holdout_result["final_bankroll"]),
        "total_pnl": float(
            full_holdout_result["final_bankroll"] - SIMULATION_START_BANKROLL_USDC
        ),
        "n_steps": int(full_holdout_result["n_steps"]),
        "n_trades": int(full_holdout_result["trades"]),
        "mean_log_growth": float(full_holdout_result["mean_g"]),
        "max_drawdown": float(full_holdout_result["max_drawdown"]),
        "hit_rate": float(full_holdout_result["hit_rate"]),
        "avg_edge": float(full_holdout_result["avg_edge"]),
        "avg_stake": float(full_holdout_result["avg_stake"]),
        "avg_fraction": float(full_holdout_result["avg_fraction"]),
        "holdout_trace_csv": str(full_holdout_trace_path),
    }

    runtime_config = build_runtime_config(
        fractional_kelly=best_fractional_kelly,
        cap=best_cap,
        min_edge=best_min_edge,
        prob_shrink=best_prob_shrink,
        price_sim_model=price_sim_model,
    )

    study.trials_dataframe().to_csv(trials_csv_path, index=False)

    run_output = {
        "generated_at_utc": run_started_at_utc.isoformat(),
        "input_path": str(INPUT_PATH),
        "study_name": study_name,
        "storage": OPTUNA_STORAGE,
        "runtime_config": runtime_config,
        "cv_meta": {
            "model_proba_error_enabled": bool(MODEL_PROBA_ERROR_SIM_ENABLED),
            "model_proba_abs_diff_mean": float(MODEL_PROBA_ERROR_ABS_DIFF_MEAN),
            "model_proba_abs_diff_max": float(MODEL_PROBA_ERROR_ABS_DIFF_MAX),
            "model_proba_abs_diff_std": float(MODEL_PROBA_ERROR_ABS_DIFF_STD),
            "model_proba_error_policy": (
                MODEL_PROBA_ERROR_POLICY
                if MODEL_PROBA_ERROR_SIM_ENABLED
                else "disabled"
            ),
            "n_folds": int(n_folds),
            "target_fold_days": int(CV_TARGET_FOLD_DAYS),
            "approx_fold_days": float(approx_fold_days),
            "n_trials": N_TRIALS,
            "seed": RANDOM_SEED,
            "trial_param_bounds": {
                name: [float(bounds[0]), float(bounds[1])]
                for name, bounds in TRIAL_PARAM_BOUNDS.items()
            },
            "execution_scenario_seeds": EXECUTION_SCENARIO_SEEDS,
            "price_sim_model": str(price_sim_model["model"]),
            "price_sim_ask_price": float(price_sim_model["ask_price"]),
            "price_sim_order_price_cap": float(price_sim_model["order_price_cap"]),
            "price_sim_order_min_size": float(price_sim_model["order_min_size"]),
            "price_sim_policy": str(price_sim_model["policy"]),
            "scoring_formula": SCORING_FORMULA,
            "best_trial_number": int(best_trial.number),
            "best_trial_score": float(best_trial.value),
            "mean_scenario_score": float(best_cv_result["mean_scenario_score"]),
            "q10_scenario_score": float(best_cv_result["q10_scenario_score"]),
            "min_scenario_score": float(best_cv_result["min_scenario_score"]),
            "mean_fold_score": float(best_cv_result["mean_fold_score"]),
            "q10_fold_score": float(best_cv_result["q10_fold_score"]),
            "min_fold_score": float(best_cv_result["min_fold_score"]),
            "q25_fold_score": float(best_cv_result["q25_fold_score"]),
            "mean_fold_trades": float(best_cv_result["mean_fold_trades"]),
            "mean_fold_log_growth": float(best_cv_result["mean_fold_log_growth"]),
            "mean_fold_max_drawdown": float(best_cv_result["mean_fold_max_drawdown"]),
            "q90_fold_max_drawdown": float(best_cv_result["q90_fold_max_drawdown"]),
            "mean_fold_avg_fraction": float(best_cv_result["mean_fold_avg_fraction"]),
            "q90_fold_avg_fraction": float(best_cv_result["q90_fold_avg_fraction"]),
        },
        "summary": {
            "holdout_window_score": float(holdout_window_summary["score"]),
            "holdout_window_mean_final_bankroll": float(
                holdout_window_summary["mean_final_bankroll"]
            ),
            "holdout_window_worst_final_bankroll": float(
                holdout_window_summary["worst_final_bankroll"]
            ),
            "holdout_window_mean_n_trades": float(
                holdout_window_summary["mean_n_trades"]
            ),
            "holdout_window_min_n_trades": int(holdout_window_summary["min_n_trades"]),
            "holdout_window_mean_avg_fraction": float(
                holdout_window_summary["mean_avg_fraction"]
            ),
            "full_holdout_score": float(full_holdout_output["score"]),
            "full_holdout_final_bankroll": float(full_holdout_output["final_bankroll"]),
            "full_holdout_n_trades": int(full_holdout_output["n_trades"]),
            "full_holdout_max_drawdown": float(full_holdout_output["max_drawdown"]),
            "full_holdout_avg_fraction": float(full_holdout_output["avg_fraction"]),
        },
        "holdout": {
            "window_days": int(HOLDOUT_WINDOW_DAYS),
            "window_run_count": int(HOLDOUT_WINDOW_RUNS),
            "window_summary": holdout_window_summary,
            "window_runs": holdout_window_runs_output,
            "full_run": full_holdout_output,
        },
        "data_split": {
            "holdout_frac": HOLDOUT_FRACTION,
            "split_idx": int(split_idx),
            "tune_rows": len(target_tune),
            "holdout_rows": len(target_holdout),
            "holdout_start_opened": pd.Timestamp(opened_holdout[0]).isoformat(),
            "holdout_end_opened": pd.Timestamp(opened_holdout[-1]).isoformat(),
        },
        "artifacts": {
            "trials_csv": str(trials_csv_path),
            "holdout_trace_dir": str(holdout_trace_dir),
            "holdout_trace_csv_by_window_id": holdout_trace_csv_by_window_id,
            "full_holdout_trace_csv": str(full_holdout_trace_path),
        },
        "sanity": {
            "target_match_ratio": float(match_ratio),
            "decision_rows": int(n_decision_rows),
            "trade_rows": int(n_trades),
        },
    }

    save_json(RUNTIME_CONFIG_PATH, runtime_config)

    save_json(run_summary_path, run_output)

    print(f"saved runtime config | path={RUNTIME_CONFIG_PATH}")

    print(f"saved optuna run | path={run_summary_path}")

    print(f"saved optuna trials | path={trials_csv_path}")

    print(
        f"saved holdout traces | dir={holdout_trace_dir} "
        f"n_files={len(holdout_window_runs_output) + 1}"
    )

    print(
        "summary | "
        f"window_score={float(holdout_window_summary['score']):.6f} "
        f"window_mean_final_bankroll={float(holdout_window_summary['mean_final_bankroll']):.4f} "
        f"full_holdout_score={float(full_holdout_output['score']):.6f} "
        f"full_holdout_final_bankroll={float(full_holdout_output['final_bankroll']):.4f}"
    )


if __name__ == "__main__":

    main()
