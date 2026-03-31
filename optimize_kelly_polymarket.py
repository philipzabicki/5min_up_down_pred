import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from numba import njit

from common_config_utils import load_json_object
from kelly_utils import adjust_probability_for_kelly
from modeling_dataset_utils import (
    load_modeling_dataset_settings,
    resolve_oof_prediction_output_paths,
)
from project_config import load_runtime_artifact_paths

optuna.logging.set_verbosity(optuna.logging.INFO)

RUNTIME_ARTIFACT_PATHS = load_runtime_artifact_paths()
MODELING_DATASET_SETTINGS = load_modeling_dataset_settings()
OPTIMIZER_CONFIG_PATH = Path("configs/kelly_optimizer_config.json")


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


def _require_optimizer_object(payload, key):
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(
            f"Optimizer config '{OPTIMIZER_CONFIG_PATH}' must define '{key}' as an object."
        )
    return value


def _require_optimizer_bounds(payload, key):
    raw_bounds = _require_optimizer_object(payload, key)
    bounds = {}
    for name, value in raw_bounds.items():
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(
                f"Optimizer config '{OPTIMIZER_CONFIG_PATH}' key '{key}.{name}' "
                "must be a 2-item array."
            )
        bounds[str(name)] = (float(value[0]), float(value[1]))
    return bounds


OPTIMIZER_CONFIG = load_json_object(OPTIMIZER_CONFIG_PATH)
OPTUNA_CONFIG = _require_optimizer_object(OPTIMIZER_CONFIG, "optuna")
TRIAL_PARAM_BOUNDS = _require_optimizer_bounds(OPTIMIZER_CONFIG, "trial_param_bounds")
DATA_SPLIT_CONFIG = _require_optimizer_object(OPTIMIZER_CONFIG, "data_split")
CV_CONFIG = _require_optimizer_object(OPTIMIZER_CONFIG, "cv")
HOLDOUT_CONFIG = _require_optimizer_object(OPTIMIZER_CONFIG, "holdout")
MARKET_PRICE_SIM_CONFIG = _require_optimizer_object(
    OPTIMIZER_CONFIG,
    "market_price_sim",
)
MODEL_PROBA_ERROR_SIM_CONFIG = _require_optimizer_object(
    OPTIMIZER_CONFIG,
    "model_proba_error_sim",
)

EXPECTED_TRIAL_PARAMS = {"fractional_kelly", "cap", "min_edge"}
if set(TRIAL_PARAM_BOUNDS.keys()) != EXPECTED_TRIAL_PARAMS:
    raise ValueError(
        "Optimizer config trial_param_bounds must define exactly "
        f"{sorted(EXPECTED_TRIAL_PARAMS)}. "
        f"Found {sorted(TRIAL_PARAM_BOUNDS.keys())}."
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
PROBABILITY_MIN_CLIP = float(
    MODEL_PROBA_ERROR_SIM_CONFIG.get("prob_min_clip", 1e-6)
)
SIMULATION_START_BANKROLL_USDC = 1000.0

# CV / holdout settings
RANDOM_SEED = int(OPTUNA_CONFIG["random_seed"])
N_TRIALS = int(OPTUNA_CONFIG["n_trials"])
OPTUNA_TPE_STARTUP_TRIALS = int(
    OPTUNA_CONFIG.get("tpe_startup_trials", int(N_TRIALS * 0.2))
)
HOLDOUT_FRACTION = float(DATA_SPLIT_CONFIG["holdout_fraction"])
CV_TARGET_FOLD_DAYS = int(CV_CONFIG["target_fold_days"])
CV_MARKET_SIM_SEEDS = [int(seed) for seed in CV_CONFIG["market_sim_seeds"]]
HOLDOUT_WINDOW_DAYS = int(HOLDOUT_CONFIG["window_days"])
HOLDOUT_WINDOW_RUNS = int(HOLDOUT_CONFIG["window_runs"])
HOLDOUT_MARKET_SIM_SEEDS = [
    int(seed) for seed in HOLDOUT_CONFIG["market_sim_seeds"]
]
FULL_HOLDOUT_MARKET_SIM_SEEDS = [
    int(seed) for seed in HOLDOUT_CONFIG["full_market_sim_seeds"]
]

# Scenario scoring
SEED_SCORE_WEIGHTS = {
    "mean": 0.6,
    "q10": 0.20,
    "min": 0.1,
    "q90_drawdown_penalty": 0.1,
}
AGGREGATE_SCORE_WEIGHTS = {
    "mean": 0.45,
    "q10": 0.30,
    "min": 0.15,
}

# Manual live-vs-modeling probability error fit used only inside Kelly optimization.
# Update these by hand from the accepted parity audit snapshot.
MODEL_PROBA_ERROR_SIM_ENABLED = bool(MODEL_PROBA_ERROR_SIM_CONFIG["enabled"])
MODEL_PROBA_ERROR_ABS_DIFF_MEAN = float(
    MODEL_PROBA_ERROR_SIM_CONFIG["abs_diff_mean"]
)
MODEL_PROBA_ERROR_ABS_DIFF_MAX = float(MODEL_PROBA_ERROR_SIM_CONFIG["abs_diff_max"])
MODEL_PROBA_ERROR_ABS_DIFF_STD = float(
    MODEL_PROBA_ERROR_SIM_CONFIG.get(
        "abs_diff_std",
        MODEL_PROBA_ERROR_ABS_DIFF_MAX / 3.0,
    )
)
MODEL_PROBA_ERROR_POLICY = str(MODEL_PROBA_ERROR_SIM_CONFIG["policy"])
MODEL_PROBA_ERROR_SEED_OFFSET = 1_000_000

SCORING_FORMULA = (
    "fold_score = log(final_bankroll / start_bankroll), "
    "seed_score = "
    "0.45 * mean(fold_scores) + 0.30 * q10(fold_scores) + 0.15 * min(fold_scores) "
    "- 0.10 * q90(fold_max_drawdowns), "
    "trial_score = "
    "0.45 * mean(seed_scores) + 0.30 * q10(seed_scores) + 0.15 * min(seed_scores), "
    "if total seed trades == 0: seed_score = -1e9"
)

# Optimizer-only oracle market simulator. Live runtime still uses real orderbook
# quotes from Polymarket and must not inherit these offline execution assumptions.
PRICE_SIM_MODEL = str(MARKET_PRICE_SIM_CONFIG["model"])
if PRICE_SIM_MODEL != "oracle_direction_normal_magnitude":
    raise ValueError(
        "Optimizer config market_price_sim.model must be "
        "'oracle_direction_normal_magnitude'. "
        f"Found '{PRICE_SIM_MODEL}'."
    )
PRICE_SIM_MARKET_DIRECTION_ACCURACY = float(
    MARKET_PRICE_SIM_CONFIG["market_direction_accuracy"]
)
if not 0.0 <= PRICE_SIM_MARKET_DIRECTION_ACCURACY <= 1.0:
    raise ValueError(
        "Optimizer config market_price_sim.market_direction_accuracy must be in [0,1]. "
        f"Found {PRICE_SIM_MARKET_DIRECTION_ACCURACY:.6f}."
    )
PRICE_SIM_MU_DELTA = float(MARKET_PRICE_SIM_CONFIG["mu_delta"])
PRICE_SIM_SIGMA_DELTA = float(MARKET_PRICE_SIM_CONFIG["sigma_delta"])
PRICE_SIM_DELTA_MAX = float(MARKET_PRICE_SIM_CONFIG["delta_max"])
PRICE_SIM_ASK_OVERROUND = float(MARKET_PRICE_SIM_CONFIG["ask_overround"])
PRICE_SIM_EPS = float(MARKET_PRICE_SIM_CONFIG["eps"])
PRICE_SIM_ORDER_MIN_SIZE = float(MARKET_PRICE_SIM_CONFIG["order_min_size"])
PRICE_SIM_POLICY = str(MARKET_PRICE_SIM_CONFIG["policy"])
PRICE_SIM_PARAMS = {
    "market_direction_accuracy": float(PRICE_SIM_MARKET_DIRECTION_ACCURACY),
    "mu_delta": float(PRICE_SIM_MU_DELTA),
    "sigma_delta": float(PRICE_SIM_SIGMA_DELTA),
    "delta_max": float(PRICE_SIM_DELTA_MAX),
    "ask_overround": float(PRICE_SIM_ASK_OVERROUND),
    "eps": float(PRICE_SIM_EPS),
    "order_min_size": float(PRICE_SIM_ORDER_MIN_SIZE),
    "policy": str(PRICE_SIM_POLICY),
}

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


def sample_oracle_direction_orderbook_arrays(
    target,
    scenario_seed,
    price_sim_config,
):
    target = np.asarray(target, dtype=np.int8)
    if target.ndim != 1:
        raise ValueError("target must be a 1D array.")
    if len(target) == 0:
        raise ValueError("target must be non-empty.")
    if not np.isin(np.unique(target), [0, 1]).all():
        raise ValueError("target must contain only 0/1 values.")

    rng = np.random.default_rng(int(scenario_seed))
    mu_delta = float(price_sim_config["mu_delta"])
    sigma_delta = float(price_sim_config["sigma_delta"])
    delta_max = float(price_sim_config["delta_max"])
    ask_overround = float(price_sim_config["ask_overround"])
    eps = float(price_sim_config["eps"])
    order_min_size = float(price_sim_config["order_min_size"])
    market_direction_accuracy = float(price_sim_config["market_direction_accuracy"])

    z = rng.standard_normal(len(target)).astype(np.float64, copy=False)
    delta_mag = np.maximum(mu_delta + sigma_delta * z, 0.0)
    delta_mag = np.minimum(delta_mag, delta_max)
    direction_is_correct = (
        rng.random(len(target)).astype(np.float64, copy=False)
        < market_direction_accuracy
    )
    direction_sign = np.where(direction_is_correct, 1.0, -1.0)
    signed_delta = delta_mag * direction_sign

    up_mid = np.where(target == 1, 0.5 + signed_delta, 0.5 - signed_delta)
    down_mid = np.where(target == 1, 0.5 - signed_delta, 0.5 + signed_delta)

    half_overround = ask_overround / 2.0
    up_ask = np.clip(up_mid + half_overround, eps, 1.0 - eps)
    down_ask = np.clip(down_mid + half_overround, eps, 1.0 - eps)

    return {
        "up_mid": up_mid,
        "down_mid": down_mid,
        "up_ask": up_ask,
        "down_ask": down_ask,
        "delta_mag": delta_mag,
        "direction_is_correct": direction_is_correct,
        "order_min_size": np.full(len(target), order_min_size, dtype=np.float64),
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
    target,
    market_sim_seed,
    price_sim_config,
):
    sampled_orderbook = sample_oracle_direction_orderbook_arrays(
        target=target,
        scenario_seed=market_sim_seed,
        price_sim_config=price_sim_config,
    )

    up_ask = sampled_orderbook["up_ask"]
    down_ask = sampled_orderbook["down_ask"]
    order_min_size = sampled_orderbook["order_min_size"]

    up_fee_coef = FEE_RATE * np.power(up_ask * (1.0 - up_ask), FEE_EXPONENT)
    down_fee_coef = FEE_RATE * np.power(down_ask * (1.0 - down_ask), FEE_EXPONENT)

    up_valid = (
        np.isfinite(up_ask)
        & (up_ask > 0.0)
        & (up_ask < 1.0)
        & np.isfinite(up_fee_coef)
        & (up_fee_coef < 0.99)
    )
    down_valid = (
        np.isfinite(down_ask)
        & (down_ask > 0.0)
        & (down_ask < 1.0)
        & np.isfinite(down_fee_coef)
        & (down_fee_coef < 0.99)
    )

    up_c_eff = np.full(len(up_ask), np.nan, dtype=np.float64)
    down_c_eff = np.full(len(down_ask), np.nan, dtype=np.float64)
    up_c_eff[up_valid] = up_ask[up_valid] / (1.0 - up_fee_coef[up_valid])
    down_c_eff[down_valid] = down_ask[down_valid] / (1.0 - down_fee_coef[down_valid])

    model_error_seed = None
    if MODEL_PROBA_ERROR_SIM_ENABLED:
        model_error_seed = int(market_sim_seed) + MODEL_PROBA_ERROR_SEED_OFFSET
        model_error_rng = np.random.default_rng(model_error_seed)
        model_error_abs_z, model_error_sign_bits = sample_model_proba_error_components(
            model_error_rng,
            len(target),
        )
        model_error_abs = np.maximum(
            MODEL_PROBA_ERROR_ABS_DIFF_MEAN
            + MODEL_PROBA_ERROR_ABS_DIFF_STD * model_error_abs_z,
            0.0,
        )
        model_error_abs = np.minimum(model_error_abs, MODEL_PROBA_ERROR_ABS_DIFF_MAX)
        model_error_sign = np.where(model_error_sign_bits == 0, -1.0, 1.0)
        model_proba_error = model_error_sign * model_error_abs
    else:
        model_proba_error = np.zeros(len(target), dtype=np.float64)

    market_diagnostics = {
        "direction_correct_rate": float(
            np.mean(sampled_orderbook["direction_is_correct"].astype(np.float64))
        ),
        "direction_wrong_rate": float(
            1.0
            - np.mean(sampled_orderbook["direction_is_correct"].astype(np.float64))
        ),
        "mean_delta_mag": float(np.mean(sampled_orderbook["delta_mag"])),
        "q90_delta_mag": float(np.quantile(sampled_orderbook["delta_mag"], 0.90)),
        "max_delta_mag": float(np.max(sampled_orderbook["delta_mag"])),
        "mean_up_ask": float(np.mean(up_ask)),
        "mean_down_ask": float(np.mean(down_ask)),
        "min_up_ask": float(np.min(up_ask)),
        "max_up_ask": float(np.max(up_ask)),
        "min_down_ask": float(np.min(down_ask)),
        "max_down_ask": float(np.max(down_ask)),
    }

    return {
        "up_ask": up_ask,
        "down_ask": down_ask,
        "order_min_size": order_min_size,
        "up_fee_coef": up_fee_coef,
        "down_fee_coef": down_fee_coef,
        "up_valid": up_valid,
        "down_valid": down_valid,
        "up_c_eff": up_c_eff,
        "down_c_eff": down_c_eff,
        "model_proba_error": model_proba_error,
        "market_sim_seed": int(market_sim_seed),
        "model_error_seed": model_error_seed,
        "market_diagnostics": market_diagnostics,
    }


def slice_static_arrays(static_arrays, start_idx, end_idx):
    sliced = {}
    for key, value in static_arrays.items():
        if isinstance(value, np.ndarray):
            sliced[key] = value[start_idx:end_idx]
        elif isinstance(value, dict):
            sliced[key] = dict(value)
        else:
            sliced[key] = value
    return sliced


def build_market_sim_scenarios(
    target,
    split_idx,
    market_sim_seeds,
    price_sim_config,
):
    scenarios = []

    for market_sim_seed in market_sim_seeds:
        full_static_arrays = build_trial_static_arrays(
            target=target,
            market_sim_seed=market_sim_seed,
            price_sim_config=price_sim_config,
        )
        scenarios.append(
            {
                "market_sim_seed": int(market_sim_seed),
                "tune_static_arrays": slice_static_arrays(
                    full_static_arrays,
                    0,
                    split_idx,
                ),
                "holdout_static_arrays": slice_static_arrays(
                    full_static_arrays,
                    split_idx,
                    len(target),
                ),
            }
        )

    return scenarios


def build_holdout_static_arrays_by_seed(
    target_holdout,
    market_sim_seeds,
    price_sim_config,
):
    return {
        int(market_sim_seed): build_trial_static_arrays(
            target=target_holdout,
            market_sim_seed=market_sim_seed,
            price_sim_config=price_sim_config,
        )
        for market_sim_seed in market_sim_seeds
    }


def build_trial_arrays(
    p_raw,
    static_arrays,
    fractional_kelly,
    cap,
    min_edge,
):
    p_simulated = np.clip(
        p_raw + static_arrays["model_proba_error"],
        PROBABILITY_MIN_CLIP,
        1.0 - PROBABILITY_MIN_CLIP,
    )

    p = adjust_probability_for_kelly(
        p_simulated,
        min_clip=PROBABILITY_MIN_CLIP,
    )

    up_ask = static_arrays["up_ask"]

    down_ask = static_arrays["down_ask"]

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

    selected_price = np.where(choose_up, up_ask, down_ask)

    selected_fee_coef = np.where(choose_up, up_fee_coef, down_fee_coef)

    selected_c_eff = np.where(choose_up, up_c_eff, down_c_eff)

    selected_order_min_size = static_arrays["order_min_size"]

    can_trade = (
        selected_valid
        & (selected_edge >= min_edge)
        & np.isfinite(selected_price)
        & (selected_price > 0.0)
        & (selected_price < 1.0)
        & np.isfinite(selected_c_eff)
        & (selected_c_eff > 0.0)
        & (selected_c_eff < 1.0)
        & np.isfinite(selected_order_min_size)
        & (selected_order_min_size > 0.0)
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
        "up_ask": up_ask,
        "down_ask": down_ask,
        "p_simulated": p_simulated,
        "p_kelly": p,
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
    market_sim_seed,
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
        "market_sim_seed": int(market_sim_seed),
        "step_log": step_log,
    }

    return result


def evaluate_cv_folds_for_market_seed(
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
        fold_scores.append(float(segment_result["sum_g"]))
        fold_trades.append(int(segment_result["n_trades"]))
        fold_log_growth.append(
            float(segment_result["sum_g"]) / int(segment_result["n_steps"])
        )
        fold_max_drawdown.append(float(segment_result["max_drawdown"]))
        fold_avg_fraction.append(float(segment_result["avg_fraction"]))

    fold_scores_arr = np.asarray(fold_scores, dtype=np.float64)
    fold_trades_arr = np.asarray(fold_trades, dtype=np.float64)
    fold_log_growth_arr = np.asarray(fold_log_growth, dtype=np.float64)
    fold_max_drawdown_arr = np.asarray(fold_max_drawdown, dtype=np.float64)
    fold_avg_fraction_arr = np.asarray(fold_avg_fraction, dtype=np.float64)

    seed_score = float(
        SEED_SCORE_WEIGHTS["mean"] * np.mean(fold_scores_arr)
        + SEED_SCORE_WEIGHTS["q10"] * np.quantile(fold_scores_arr, 0.10)
        + SEED_SCORE_WEIGHTS["min"] * np.min(fold_scores_arr)
        - SEED_SCORE_WEIGHTS["q90_drawdown_penalty"]
        * np.quantile(fold_max_drawdown_arr, 0.90)
    )

    if int(np.sum(fold_trades_arr)) == 0:
        seed_score = -1e9

    return {
        "score": float(seed_score),
        "fold_scores": fold_scores,
        "fold_trades": fold_trades,
        "fold_log_growth": fold_log_growth,
        "fold_max_drawdown": fold_max_drawdown,
        "fold_avg_fraction": fold_avg_fraction,
        "mean_fold_score": float(np.mean(fold_scores_arr)),
        "q10_fold_score": float(np.quantile(fold_scores_arr, 0.10)),
        "min_fold_score": float(np.min(fold_scores_arr)),
        "mean_fold_trades": float(np.mean(fold_trades_arr)),
        "mean_fold_log_growth": float(np.mean(fold_log_growth_arr)),
        "mean_fold_max_drawdown": float(np.mean(fold_max_drawdown_arr)),
        "q90_fold_max_drawdown": float(np.quantile(fold_max_drawdown_arr, 0.90)),
        "mean_fold_avg_fraction": float(np.mean(fold_avg_fraction_arr)),
        "q90_fold_avg_fraction": float(np.quantile(fold_avg_fraction_arr, 0.90)),
    }


def summarize_weighted_score(values, weights):
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.ndim != 1 or len(values_arr) == 0:
        raise ValueError("Score summary expects a non-empty 1D array.")

    return {
        "score": float(
            weights["mean"] * np.mean(values_arr)
            + weights["q10"] * np.quantile(values_arr, 0.10)
            + weights["min"] * np.min(values_arr)
        ),
        "mean": float(np.mean(values_arr)),
        "q10": float(np.quantile(values_arr, 0.10)),
        "min": float(np.min(values_arr)),
    }


def evaluate_trial_across_market_sim_scenarios(
    target,
    p_raw,
    market_sim_scenarios,
    folds,
    fractional_kelly,
    cap,
    min_edge,
    min_stake_usdc,
):
    seed_results = []
    fold_scores = []
    fold_trades = []
    fold_log_growth = []
    fold_max_drawdown = []
    fold_avg_fraction = []

    for market_sim_scenario in market_sim_scenarios:
        static_arrays = market_sim_scenario["tune_static_arrays"]
        trial_arrays = build_trial_arrays(
            p_raw=p_raw,
            static_arrays=static_arrays,
            fractional_kelly=fractional_kelly,
            cap=cap,
            min_edge=min_edge,
        )

        seed_result = evaluate_cv_folds_for_market_seed(
            target=target,
            trial_arrays=trial_arrays,
            folds=folds,
            min_stake_usdc=min_stake_usdc,
        )

        diagnostics = static_arrays["market_diagnostics"]
        seed_output = {
            "market_sim_seed": int(market_sim_scenario["market_sim_seed"]),
            "model_error_seed": static_arrays["model_error_seed"],
            "score": float(seed_result["score"]),
            "mean_fold_score": float(seed_result["mean_fold_score"]),
            "q10_fold_score": float(seed_result["q10_fold_score"]),
            "min_fold_score": float(seed_result["min_fold_score"]),
            "mean_fold_trades": float(seed_result["mean_fold_trades"]),
            "mean_fold_log_growth": float(seed_result["mean_fold_log_growth"]),
            "mean_fold_max_drawdown": float(seed_result["mean_fold_max_drawdown"]),
            "q90_fold_max_drawdown": float(seed_result["q90_fold_max_drawdown"]),
            "mean_fold_avg_fraction": float(seed_result["mean_fold_avg_fraction"]),
            "q90_fold_avg_fraction": float(seed_result["q90_fold_avg_fraction"]),
            "direction_correct_rate": float(diagnostics["direction_correct_rate"]),
            "direction_wrong_rate": float(diagnostics["direction_wrong_rate"]),
            "mean_delta_mag": float(diagnostics["mean_delta_mag"]),
            "q90_delta_mag": float(diagnostics["q90_delta_mag"]),
            "max_delta_mag": float(diagnostics["max_delta_mag"]),
            "mean_up_ask": float(diagnostics["mean_up_ask"]),
            "mean_down_ask": float(diagnostics["mean_down_ask"]),
            "fold_scores": list(seed_result["fold_scores"]),
            "fold_trades": list(seed_result["fold_trades"]),
            "fold_log_growth": list(seed_result["fold_log_growth"]),
            "fold_max_drawdown": list(seed_result["fold_max_drawdown"]),
            "fold_avg_fraction": list(seed_result["fold_avg_fraction"]),
        }
        seed_results.append(seed_output)

        fold_scores.extend(seed_output["fold_scores"])
        fold_trades.extend(seed_output["fold_trades"])
        fold_log_growth.extend(seed_output["fold_log_growth"])
        fold_max_drawdown.extend(seed_output["fold_max_drawdown"])
        fold_avg_fraction.extend(seed_output["fold_avg_fraction"])

    seed_score_summary = summarize_weighted_score(
        [float(item["score"]) for item in seed_results],
        AGGREGATE_SCORE_WEIGHTS,
    )

    fold_scores_arr = np.asarray(fold_scores, dtype=np.float64)
    fold_trades_arr = np.asarray(fold_trades, dtype=np.float64)
    fold_log_growth_arr = np.asarray(fold_log_growth, dtype=np.float64)
    fold_max_drawdown_arr = np.asarray(fold_max_drawdown, dtype=np.float64)
    fold_avg_fraction_arr = np.asarray(fold_avg_fraction, dtype=np.float64)

    return {
        "score": float(seed_score_summary["score"]),
        "mean_seed_score": float(seed_score_summary["mean"]),
        "q10_seed_score": float(seed_score_summary["q10"]),
        "min_seed_score": float(seed_score_summary["min"]),
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
        "seed_results": seed_results,
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
):
    return {
        "fractional_kelly": float(fractional_kelly),
        "cap": float(cap),
        "min_edge": float(min_edge),
        "min_stake_usdc": float(MIN_STAKE_USDC),
        "fee_model": {
            "feeRate": float(FEE_RATE),
            "exponent": float(FEE_EXPONENT),
            "fee_round_decimals": int(FEE_ROUND_DECIMALS),
            "min_fee": float(MIN_FEE),
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
):
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
                f"Invalid holdout window for run={run_idx}: "
                f"start_idx={start_idx}, end_idx={end_idx}"
            )

        specs.append(
            {
                "window_id": int(run_idx),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "start_opened": pd.Timestamp(opened_holdout[start_idx]).isoformat(),
                "end_opened": pd.Timestamp(opened_holdout[end_idx - 1]).isoformat(),
            }
        )

    return specs


def summarize_seed_run_results(results):
    if not results:
        raise ValueError("At least one seed result is required.")

    scores = np.asarray(
        [float(result.get("score", result.get("fold_score"))) for result in results],
        dtype=np.float64,
    )
    final_bankrolls = np.asarray(
        [float(result["final_bankroll"]) for result in results],
        dtype=np.float64,
    )
    trades = np.asarray(
        [int(result.get("n_trades", result.get("trades", 0))) for result in results],
        dtype=np.int64,
    )
    hit_rates = np.asarray(
        [float(result.get("hit_rate", 0.0)) for result in results],
        dtype=np.float64,
    )
    avg_edges = np.asarray(
        [float(result.get("avg_edge", 0.0)) for result in results],
        dtype=np.float64,
    )
    avg_stakes = np.asarray(
        [float(result.get("avg_stake", 0.0)) for result in results],
        dtype=np.float64,
    )
    avg_fractions = np.asarray(
        [float(result.get("avg_fraction", 0.0)) for result in results],
        dtype=np.float64,
    )
    mean_g = np.asarray(
        [
            float(result.get("mean_log_growth", result.get("mean_g", 0.0)))
            for result in results
        ],
        dtype=np.float64,
    )
    max_drawdowns = np.asarray(
        [float(result["max_drawdown"]) for result in results],
        dtype=np.float64,
    )
    seed_scores = summarize_weighted_score(scores, AGGREGATE_SCORE_WEIGHTS)
    unique_market_sim_seeds = sorted(
        {
            int(result["market_sim_seed"])
            for result in results
            if result.get("market_sim_seed") is not None
        }
    )

    return {
        "score": float(seed_scores["score"]),
        "score_mean": float(seed_scores["mean"]),
        "score_q10": float(seed_scores["q10"]),
        "score_min": float(seed_scores["min"]),
        "final_bankroll_mean": float(np.mean(final_bankrolls)),
        "final_bankroll_worst": float(np.min(final_bankrolls)),
        "total_pnl_mean": float(
            np.mean(final_bankrolls) - SIMULATION_START_BANKROLL_USDC
        ),
        "n_trades_mean": float(np.mean(trades)),
        "n_trades_worst": int(np.min(trades)),
        "hit_rate_mean": float(np.mean(hit_rates)),
        "avg_edge_mean": float(np.mean(avg_edges)),
        "avg_stake_mean": float(np.mean(avg_stakes)),
        "avg_fraction_mean": float(np.mean(avg_fractions)),
        "mean_log_growth_mean": float(np.mean(mean_g)),
        "mean_log_growth_worst": float(np.min(mean_g)),
        "max_drawdown_mean": float(np.mean(max_drawdowns)),
        "max_drawdown_worst": float(np.max(max_drawdowns)),
        "run_count": int(len(results)),
        "market_sim_seed_count": int(len(unique_market_sim_seeds)),
        "market_sim_seeds": unique_market_sim_seeds,
    }


def summarize_static_price_range(static_arrays_collection):
    min_ask = float("inf")
    max_ask = float("-inf")

    for static_arrays in static_arrays_collection:
        min_ask = min(
            min_ask,
            float(np.min(static_arrays["up_ask"])),
            float(np.min(static_arrays["down_ask"])),
        )
        max_ask = max(
            max_ask,
            float(np.max(static_arrays["up_ask"])),
            float(np.max(static_arrays["down_ask"])),
        )

    if not np.isfinite(min_ask) or not np.isfinite(max_ask):
        raise ValueError("Price sanity failed: non-finite sampled asks.")
    if min_ask <= 0.0 or max_ask >= 1.0:
        raise ValueError(
            "Invalid price construction: sampled ask prices must stay in (0,1). "
            f"Observed min_ask={min_ask:.6f}, max_ask={max_ask:.6f}."
        )

    return {
        "min_ask": float(min_ask),
        "max_ask": float(max_ask),
    }


def evaluate_holdout_for_seeds(
    target_holdout,
    p_raw_holdout,
    opened_holdout,
    holdout_static_arrays_by_seed,
    market_sim_seeds,
    fractional_kelly,
    cap,
    min_edge,
    min_stake_usdc,
    start_idx,
    end_idx,
    trace_dir,
    trace_name_prefix,
):
    seed_results = []
    trace_paths = {}

    for market_sim_seed in market_sim_seeds:
        static_arrays = holdout_static_arrays_by_seed[int(market_sim_seed)]
        trial_arrays = build_trial_arrays(
            p_raw=p_raw_holdout,
            static_arrays=static_arrays,
            fractional_kelly=fractional_kelly,
            cap=cap,
            min_edge=min_edge,
        )

        holdout_result = simulate_segment_trace(
            target=target_holdout,
            trial_arrays=trial_arrays,
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            min_stake_usdc=min_stake_usdc,
            opened=opened_holdout,
            market_sim_seed=int(market_sim_seed),
            start_bankroll=SIMULATION_START_BANKROLL_USDC,
        )

        trace_path = trace_dir / (
            f"{trace_name_prefix}_market_seed_{int(market_sim_seed)}.csv"
        )
        pd.DataFrame(holdout_result["step_log"]).to_csv(
            trace_path,
            index=False,
            columns=STEP_LOG_COLUMNS,
        )

        diagnostics = static_arrays["market_diagnostics"]
        seed_results.append(
            {
                "market_sim_seed": int(market_sim_seed),
                "model_error_seed": static_arrays["model_error_seed"],
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "start_opened": pd.Timestamp(opened_holdout[start_idx]).isoformat(),
                "end_opened": pd.Timestamp(opened_holdout[end_idx - 1]).isoformat(),
                "score": float(holdout_result["fold_score"]),
                "final_bankroll": float(holdout_result["final_bankroll"]),
                "total_pnl": float(
                    holdout_result["final_bankroll"] - SIMULATION_START_BANKROLL_USDC
                ),
                "n_steps": int(holdout_result["n_steps"]),
                "n_trades": int(holdout_result["trades"]),
                "mean_log_growth": float(holdout_result["mean_g"]),
                "max_drawdown": float(holdout_result["max_drawdown"]),
                "hit_rate": float(holdout_result["hit_rate"]),
                "avg_edge": float(holdout_result["avg_edge"]),
                "avg_stake": float(holdout_result["avg_stake"]),
                "avg_fraction": float(holdout_result["avg_fraction"]),
                "direction_correct_rate": float(diagnostics["direction_correct_rate"]),
                "direction_wrong_rate": float(diagnostics["direction_wrong_rate"]),
                "mean_delta_mag": float(diagnostics["mean_delta_mag"]),
                "q90_delta_mag": float(diagnostics["q90_delta_mag"]),
                "max_delta_mag": float(diagnostics["max_delta_mag"]),
                "mean_up_ask": float(diagnostics["mean_up_ask"]),
                "mean_down_ask": float(diagnostics["mean_down_ask"]),
                "holdout_trace_csv": str(trace_path),
            }
        )
        trace_paths[str(int(market_sim_seed))] = str(trace_path)

    return {
        "summary": summarize_seed_run_results(seed_results),
        "seed_results": seed_results,
        "trace_paths": trace_paths,
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
        "market sim | "
        f"model={PRICE_SIM_MODEL} "
        f"market_direction_accuracy={PRICE_SIM_MARKET_DIRECTION_ACCURACY:.3f} "
        f"mu_delta={PRICE_SIM_MU_DELTA:.6f} "
        f"sigma_delta={PRICE_SIM_SIGMA_DELTA:.6f} "
        f"delta_max={PRICE_SIM_DELTA_MAX:.6f} "
        f"ask_overround={PRICE_SIM_ASK_OVERROUND:.6f} "
        f"eps={PRICE_SIM_EPS:.8f} "
        f"order_min_size={PRICE_SIM_ORDER_MIN_SIZE:.4f}"
    )

    print(
        "model error sim | "
        f"enabled={MODEL_PROBA_ERROR_SIM_ENABLED} "
        f"abs_mean={MODEL_PROBA_ERROR_ABS_DIFF_MEAN:.6f} "
        f"abs_max={MODEL_PROBA_ERROR_ABS_DIFF_MAX:.6f} "
        f"abs_std={MODEL_PROBA_ERROR_ABS_DIFF_STD:.6f} "
        f"policy={MODEL_PROBA_ERROR_POLICY if MODEL_PROBA_ERROR_SIM_ENABLED else 'disabled'}"
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

    market_sim_scenarios = build_market_sim_scenarios(
        target=target,
        split_idx=split_idx,
        market_sim_seeds=CV_MARKET_SIM_SEEDS,
        price_sim_config=PRICE_SIM_PARAMS,
    )

    n_folds, approx_fold_days = infer_n_folds_for_target_days(
        n_obs=len(target_tune),
        target_fold_days=CV_TARGET_FOLD_DAYS,
    )

    folds = make_walk_forward_folds(n_obs=len(target_tune), n_folds=n_folds)
    cv_price_sanity = summarize_static_price_range(
        [scenario["tune_static_arrays"] for scenario in market_sim_scenarios]
        + [scenario["holdout_static_arrays"] for scenario in market_sim_scenarios]
    )

    print(
        f"cv setup | n_folds={n_folds} approx_fold_days={approx_fold_days:.2f} "
        f"target_fold_days={CV_TARGET_FOLD_DAYS} n_trials={N_TRIALS} "
        f"n_market_sim_seeds={len(CV_MARKET_SIM_SEEDS)} "
        f"n_trades_tune={len(target_tune)} "
        f"first_fold=[{folds[0][0]}:{folds[0][1]}]"
    )

    print(
        f"data split | holdout_frac={HOLDOUT_FRACTION:.3f} split_idx={split_idx} "
        f"tune_rows={len(target_tune)} holdout_rows={len(target_holdout)} "
        f"holdout_start_opened={pd.Timestamp(opened_holdout[0]).isoformat()} "
        f"holdout_end_opened={pd.Timestamp(opened_holdout[-1]).isoformat()}"
    )

    print(
        "market sim seeds | "
        f"cv={CV_MARKET_SIM_SEEDS} "
        f"holdout={HOLDOUT_MARKET_SIM_SEEDS} "
        f"full_holdout={FULL_HOLDOUT_MARKET_SIM_SEEDS}"
    )
    print(
        "price sanity | "
        f"cv_min_ask={cv_price_sanity['min_ask']:.6f} "
        f"cv_max_ask={cv_price_sanity['max_ask']:.6f}"
    )

    def objective(trial):
        trial_params = suggest_trial_params(trial)

        cv_result = evaluate_trial_across_market_sim_scenarios(
            target=target_tune,
            p_raw=p_raw_tune,
            market_sim_scenarios=market_sim_scenarios,
            folds=folds,
            fractional_kelly=trial_params["fractional_kelly"],
            cap=trial_params["cap"],
            min_edge=trial_params["min_edge"],
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
        f"n_market_sim_seeds={len(CV_MARKET_SIM_SEEDS)} "
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

    best_cv_result = evaluate_trial_across_market_sim_scenarios(
        target=target_tune,
        p_raw=p_raw_tune,
        market_sim_scenarios=market_sim_scenarios,
        folds=folds,
        fractional_kelly=best_fractional_kelly,
        cap=best_cap,
        min_edge=best_min_edge,
        min_stake_usdc=MIN_STAKE_USDC,
    )

    holdout_window_specs = build_holdout_window_specs(
        opened_holdout=opened_holdout,
        window_days=HOLDOUT_WINDOW_DAYS,
        n_runs=HOLDOUT_WINDOW_RUNS,
    )
    holdout_seed_union = sorted(
        set(HOLDOUT_MARKET_SIM_SEEDS).union(FULL_HOLDOUT_MARKET_SIM_SEEDS)
    )
    holdout_static_arrays_by_seed = build_holdout_static_arrays_by_seed(
        target_holdout=target_holdout,
        market_sim_seeds=holdout_seed_union,
        price_sim_config=PRICE_SIM_PARAMS,
    )
    holdout_price_sanity = summarize_static_price_range(
        list(holdout_static_arrays_by_seed.values())
    )

    print(
        "holdout setup | "
        f"window_days={HOLDOUT_WINDOW_DAYS} "
        f"window_runs={HOLDOUT_WINDOW_RUNS} "
        f"holdout_market_sim_seed_count={len(HOLDOUT_MARKET_SIM_SEEDS)} "
        f"full_holdout_market_sim_seed_count={len(FULL_HOLDOUT_MARKET_SIM_SEEDS)}"
    )
    print(
        "price sanity | "
        f"holdout_min_ask={holdout_price_sanity['min_ask']:.6f} "
        f"holdout_max_ask={holdout_price_sanity['max_ask']:.6f}"
    )

    holdout_trace_dir.mkdir(parents=True, exist_ok=True)
    holdout_trace_csv_by_window_id = {}
    holdout_window_runs_output = []
    all_window_seed_results = []

    for window_spec in holdout_window_specs:
        window_eval = evaluate_holdout_for_seeds(
            target_holdout=target_holdout,
            p_raw_holdout=p_raw_holdout,
            opened_holdout=opened_holdout,
            holdout_static_arrays_by_seed=holdout_static_arrays_by_seed,
            market_sim_seeds=HOLDOUT_MARKET_SIM_SEEDS,
            fractional_kelly=best_fractional_kelly,
            cap=best_cap,
            min_edge=best_min_edge,
            min_stake_usdc=MIN_STAKE_USDC,
            start_idx=int(window_spec["start_idx"]),
            end_idx=int(window_spec["end_idx"]),
            trace_dir=holdout_trace_dir,
            trace_name_prefix=f"window_{int(window_spec['window_id']):02d}",
        )
        holdout_trace_csv_by_window_id[str(int(window_spec["window_id"]))] = (
            window_eval["trace_paths"]
        )
        all_window_seed_results.extend(window_eval["seed_results"])
        holdout_window_runs_output.append(
            {
                "window_id": int(window_spec["window_id"]),
                "window_days": int(HOLDOUT_WINDOW_DAYS),
                "start_idx": int(window_spec["start_idx"]),
                "end_idx": int(window_spec["end_idx"]),
                "start_opened": str(window_spec["start_opened"]),
                "end_opened": str(window_spec["end_opened"]),
                **window_eval["summary"],
                "seed_results": window_eval["seed_results"],
            }
        )

    holdout_window_summary = summarize_seed_run_results(all_window_seed_results)
    holdout_window_summary.update(
        {
            "window_days": int(HOLDOUT_WINDOW_DAYS),
            "window_run_count": int(HOLDOUT_WINDOW_RUNS),
            "total_seed_runs": int(len(all_window_seed_results)),
        }
    )

    full_holdout_eval = evaluate_holdout_for_seeds(
        target_holdout=target_holdout,
        p_raw_holdout=p_raw_holdout,
        opened_holdout=opened_holdout,
        holdout_static_arrays_by_seed=holdout_static_arrays_by_seed,
        market_sim_seeds=FULL_HOLDOUT_MARKET_SIM_SEEDS,
        fractional_kelly=best_fractional_kelly,
        cap=best_cap,
        min_edge=best_min_edge,
        min_stake_usdc=MIN_STAKE_USDC,
        start_idx=0,
        end_idx=len(target_holdout),
        trace_dir=holdout_trace_dir,
        trace_name_prefix="full_holdout",
    )
    full_holdout_output = {
        "start_idx": 0,
        "end_idx": len(target_holdout),
        "start_opened": pd.Timestamp(opened_holdout[0]).isoformat(),
        "end_opened": pd.Timestamp(opened_holdout[-1]).isoformat(),
        **full_holdout_eval["summary"],
        "seed_results": full_holdout_eval["seed_results"],
    }

    runtime_config = build_runtime_config(
        fractional_kelly=best_fractional_kelly,
        cap=best_cap,
        min_edge=best_min_edge,
    )

    study.trials_dataframe().to_csv(trials_csv_path, index=False)

    run_output = {
        "generated_at_utc": run_started_at_utc.isoformat(),
        "input_path": str(INPUT_PATH),
        "optimizer_config_path": str(OPTIMIZER_CONFIG_PATH),
        "optimizer_config": OPTIMIZER_CONFIG,
        "study_name": study_name,
        "storage": OPTUNA_STORAGE,
        "runtime_config": runtime_config,
        "price_sim_model": PRICE_SIM_MODEL,
        "price_sim_params": PRICE_SIM_PARAMS,
        "cv_market_sim_seeds": CV_MARKET_SIM_SEEDS,
        "holdout_market_sim_seeds": HOLDOUT_MARKET_SIM_SEEDS,
        "full_holdout_market_sim_seeds": FULL_HOLDOUT_MARKET_SIM_SEEDS,
        "model_proba_error_sim": {
            "enabled": bool(MODEL_PROBA_ERROR_SIM_ENABLED),
            "abs_diff_mean": float(MODEL_PROBA_ERROR_ABS_DIFF_MEAN),
            "abs_diff_max": float(MODEL_PROBA_ERROR_ABS_DIFF_MAX),
            "abs_diff_std": float(MODEL_PROBA_ERROR_ABS_DIFF_STD),
            "prob_min_clip": float(PROBABILITY_MIN_CLIP),
            "policy": (
                MODEL_PROBA_ERROR_POLICY if MODEL_PROBA_ERROR_SIM_ENABLED else "disabled"
            ),
        },
        "cv_meta": {
            "n_folds": int(n_folds),
            "target_fold_days": int(CV_TARGET_FOLD_DAYS),
            "approx_fold_days": float(approx_fold_days),
            "n_trials": N_TRIALS,
            "seed": RANDOM_SEED,
            "trial_param_bounds": {
                name: [float(bounds[0]), float(bounds[1])]
                for name, bounds in TRIAL_PARAM_BOUNDS.items()
            },
            "scoring_formula": SCORING_FORMULA,
            "best_trial_number": int(best_trial.number),
            "best_trial_score": float(best_trial.value),
            "mean_seed_score": float(best_cv_result["mean_seed_score"]),
            "q10_seed_score": float(best_cv_result["q10_seed_score"]),
            "min_seed_score": float(best_cv_result["min_seed_score"]),
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
        "best_trial_seed_results": best_cv_result["seed_results"],
        "summary": {
            "best_trial_number": int(best_trial.number),
            "best_trial_score": float(best_trial.value),
            "best_fractional_kelly": float(best_fractional_kelly),
            "best_cap": float(best_cap),
            "best_min_edge": float(best_min_edge),
            "cv_score": float(best_cv_result["score"]),
            "cv_mean_seed_score": float(best_cv_result["mean_seed_score"]),
            "cv_q10_seed_score": float(best_cv_result["q10_seed_score"]),
            "cv_min_seed_score": float(best_cv_result["min_seed_score"]),
            "holdout_window_score": float(holdout_window_summary["score"]),
            "holdout_window_score_mean": float(holdout_window_summary["score_mean"]),
            "holdout_window_score_q10": float(holdout_window_summary["score_q10"]),
            "holdout_window_score_min": float(holdout_window_summary["score_min"]),
            "holdout_window_final_bankroll_mean": float(
                holdout_window_summary["final_bankroll_mean"]
            ),
            "holdout_window_final_bankroll_worst": float(
                holdout_window_summary["final_bankroll_worst"]
            ),
            "holdout_window_n_trades_mean": float(
                holdout_window_summary["n_trades_mean"]
            ),
            "holdout_window_n_trades_worst": int(
                holdout_window_summary["n_trades_worst"]
            ),
            "holdout_window_max_drawdown_mean": float(
                holdout_window_summary["max_drawdown_mean"]
            ),
            "holdout_window_max_drawdown_worst": float(
                holdout_window_summary["max_drawdown_worst"]
            ),
            "full_holdout_score": float(full_holdout_output["score"]),
            "full_holdout_score_mean": float(full_holdout_output["score_mean"]),
            "full_holdout_score_q10": float(full_holdout_output["score_q10"]),
            "full_holdout_score_min": float(full_holdout_output["score_min"]),
            "full_holdout_final_bankroll_mean": float(
                full_holdout_output["final_bankroll_mean"]
            ),
            "full_holdout_final_bankroll_worst": float(
                full_holdout_output["final_bankroll_worst"]
            ),
            "full_holdout_n_trades_mean": float(full_holdout_output["n_trades_mean"]),
            "full_holdout_n_trades_worst": int(full_holdout_output["n_trades_worst"]),
            "full_holdout_max_drawdown_mean": float(
                full_holdout_output["max_drawdown_mean"]
            ),
            "full_holdout_max_drawdown_worst": float(
                full_holdout_output["max_drawdown_worst"]
            ),
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
            "full_holdout_trace_csv_by_market_seed": full_holdout_eval["trace_paths"],
        },
        "sanity": {
            "target_match_ratio": float(match_ratio),
            "decision_rows": int(n_decision_rows),
            "trade_rows": int(n_trades),
            "cv_min_ask": float(cv_price_sanity["min_ask"]),
            "cv_max_ask": float(cv_price_sanity["max_ask"]),
            "holdout_min_ask": float(holdout_price_sanity["min_ask"]),
            "holdout_max_ask": float(holdout_price_sanity["max_ask"]),
        },
    }

    save_json(RUNTIME_CONFIG_PATH, runtime_config)

    save_json(run_summary_path, run_output)

    print(f"saved runtime config | path={RUNTIME_CONFIG_PATH}")

    print(f"saved optuna run | path={run_summary_path}")

    print(f"saved optuna trials | path={trials_csv_path}")

    print(
        f"saved holdout traces | dir={holdout_trace_dir} "
        f"n_files={len(all_window_seed_results) + len(FULL_HOLDOUT_MARKET_SIM_SEEDS)}"
    )

    print(
        "summary | "
        f"window_score={float(holdout_window_summary['score']):.6f} "
        f"window_final_bankroll_mean={float(holdout_window_summary['final_bankroll_mean']):.4f} "
        f"full_holdout_score={float(full_holdout_output['score']):.6f} "
        f"full_holdout_final_bankroll_mean={float(full_holdout_output['final_bankroll_mean']):.4f}"
    )


if __name__ == "__main__":

    main()
