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
from polymarket_fee_utils import normalize_polymarket_fee_model
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


def _require_optimizer_object(payload, key, *, context=None):
    value = payload.get(key)
    if not isinstance(value, dict):
        context = str(OPTIMIZER_CONFIG_PATH) if context is None else str(context)
        raise ValueError(
            f"Optimizer config '{context}' must define '{key}' as an object."
        )
    return value


def _require_optimizer_float(payload, key, *, context=None):
    if key not in payload:
        context = str(OPTIMIZER_CONFIG_PATH) if context is None else str(context)
        raise KeyError(
            f"Optimizer config '{context}' must define '{key}'."
        )
    return float(payload[key])


def _require_optimizer_bounds(payload, key, *, context=None):
    context = str(OPTIMIZER_CONFIG_PATH) if context is None else str(context)
    raw_bounds = _require_optimizer_object(payload, key, context=context)
    bounds = {}
    for name, value in raw_bounds.items():
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(
                f"Optimizer config '{context}' key '{key}.{name}' "
                "must be a 2-item array."
            )
        bounds[str(name)] = (float(value[0]), float(value[1]))
    return bounds


def _require_optimizer_weight_map(payload, key, *, expected_keys, context=None):
    context = str(OPTIMIZER_CONFIG_PATH) if context is None else str(context)
    weights = _require_optimizer_object(payload, key, context=context)
    if set(weights.keys()) != set(expected_keys):
        raise ValueError(
            f"Optimizer config '{context}' key '{key}' must define exactly "
            f"{sorted(expected_keys)}. Found {sorted(weights.keys())}."
        )
    return {name: float(weights[name]) for name in expected_keys}


def build_market_price_sim_params(market_price_sim_config):
    model = str(market_price_sim_config["model"])
    if model != "latent_conviction_directional":
        raise ValueError(
            "Optimizer config market_price_sim.model must be "
            "'latent_conviction_directional'. "
            f"Found '{model}'."
        )

    params = {
        "model": model,
        "conviction_beta_alpha": _require_optimizer_float(
            market_price_sim_config,
            "conviction_beta_alpha",
        ),
        "conviction_beta_beta": _require_optimizer_float(
            market_price_sim_config,
            "conviction_beta_beta",
        ),
        "gap_min": _require_optimizer_float(market_price_sim_config, "gap_min"),
        "gap_max": _require_optimizer_float(market_price_sim_config, "gap_max"),
        "gap_gamma": _require_optimizer_float(market_price_sim_config, "gap_gamma"),
        "p_correct_min": _require_optimizer_float(
            market_price_sim_config,
            "p_correct_min",
        ),
        "p_correct_max": _require_optimizer_float(
            market_price_sim_config,
            "p_correct_max",
        ),
        "overround_min": _require_optimizer_float(
            market_price_sim_config,
            "overround_min",
        ),
        "overround_max": _require_optimizer_float(
            market_price_sim_config,
            "overround_max",
        ),
        "overround_gamma": _require_optimizer_float(
            market_price_sim_config,
            "overround_gamma",
        ),
        "tick_size": _require_optimizer_float(market_price_sim_config, "tick_size"),
        "eps": _require_optimizer_float(market_price_sim_config, "eps"),
        "sim_order_min_size_shares": _require_optimizer_float(
            market_price_sim_config,
            "sim_order_min_size_shares",
        ),
        "policy": str(market_price_sim_config["policy"]),
    }

    if params["conviction_beta_alpha"] <= 0.0 or params["conviction_beta_beta"] <= 0.0:
        raise ValueError("market_price_sim conviction beta params must be > 0.")
    if params["gap_min"] < 0.0 or params["gap_max"] < params["gap_min"]:
        raise ValueError("market_price_sim gap bounds must satisfy 0 <= gap_min <= gap_max.")
    if params["gap_gamma"] <= 0.0:
        raise ValueError("market_price_sim gap_gamma must be > 0.")
    if not (
        0.0 <= params["p_correct_min"] <= 1.0
        and 0.0 <= params["p_correct_max"] <= 1.0
    ):
        raise ValueError(
            "market_price_sim p_correct bounds must satisfy "
            "0 <= p_correct_min <= 1 and 0 <= p_correct_max <= 1."
        )
    if (
        params["overround_min"] < 0.0
        or params["overround_max"] < params["overround_min"]
    ):
        raise ValueError(
            "market_price_sim overround bounds must satisfy "
            "0 <= overround_min <= overround_max."
        )
    if params["overround_gamma"] <= 0.0:
        raise ValueError("market_price_sim overround_gamma must be > 0.")
    if params["tick_size"] <= 0.0:
        raise ValueError("market_price_sim tick_size must be > 0.")
    if not 0.0 < params["eps"] < 0.5:
        raise ValueError("market_price_sim eps must be in (0, 0.5).")
    if params["sim_order_min_size_shares"] <= 0.0:
        raise ValueError("market_price_sim sim_order_min_size_shares must be > 0.")

    return params


EXPECTED_TRIAL_PARAMS = {"fractional_kelly", "cap", "min_edge"}
EXPECTED_SEED_SCORE_WEIGHT_KEYS = {"mean", "q10", "min", "q90_drawdown_penalty"}
EXPECTED_AGGREGATE_SCORE_WEIGHT_KEYS = {"mean", "q10", "min"}


def _format_score_weight(value):
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def build_scoring_formula(score_weights):
    seed_weights = score_weights["seed"]
    aggregate_weights = score_weights["aggregate"]
    return (
        "fold_score = log(final_bankroll / start_bankroll), "
        "seed_score = "
        f"{_format_score_weight(seed_weights['mean'])} * mean(fold_scores) + "
        f"{_format_score_weight(seed_weights['q10'])} * q10(fold_scores) + "
        f"{_format_score_weight(seed_weights['min'])} * min(fold_scores) - "
        f"{_format_score_weight(seed_weights['q90_drawdown_penalty'])} * "
        "q90(fold_max_drawdowns), "
        "trial_score = "
        f"{_format_score_weight(aggregate_weights['mean'])} * mean(seed_scores) + "
        f"{_format_score_weight(aggregate_weights['q10'])} * q10(seed_scores) + "
        f"{_format_score_weight(aggregate_weights['min'])} * min(seed_scores), "
        "if total seed trades == 0: seed_score = -1e9"
    )


def load_optimizer_settings(config_path):
    config_path = Path(config_path)
    optimizer_config = load_json_object(config_path)
    context = str(config_path)

    optuna_settings = _require_optimizer_object(
        optimizer_config,
        "optuna",
        context=context,
    )
    trial_param_bounds = _require_optimizer_bounds(
        optimizer_config,
        "trial_param_bounds",
        context=context,
    )
    data_split_settings = _require_optimizer_object(
        optimizer_config,
        "data_split",
        context=context,
    )
    cv_settings = _require_optimizer_object(
        optimizer_config,
        "cv",
        context=context,
    )
    holdout_settings = _require_optimizer_object(
        optimizer_config,
        "holdout",
        context=context,
    )
    fee_model_config = _require_optimizer_object(
        optimizer_config,
        "fee_model",
        context=context,
    )
    simulation_settings = _require_optimizer_object(
        optimizer_config,
        "simulation",
        context=context,
    )
    score_weight_config = _require_optimizer_object(
        optimizer_config,
        "score_weights",
        context=context,
    )
    market_price_sim_config = _require_optimizer_object(
        optimizer_config,
        "market_price_sim",
        context=context,
    )
    model_proba_error_sim_config = _require_optimizer_object(
        optimizer_config,
        "model_proba_error_sim",
        context=context,
    )

    if set(trial_param_bounds.keys()) != EXPECTED_TRIAL_PARAMS:
        raise ValueError(
            "Optimizer config trial_param_bounds must define exactly "
            f"{sorted(EXPECTED_TRIAL_PARAMS)}. "
            f"Found {sorted(trial_param_bounds.keys())}."
        )

    fee_model = normalize_polymarket_fee_model(
        fee_model_config,
        context=f"Optimizer config '{context}' fee_model",
    )
    fee_model["policy"] = str(fee_model_config.get("policy", ""))

    seed_score_weights = _require_optimizer_weight_map(
        score_weight_config,
        "seed",
        expected_keys=EXPECTED_SEED_SCORE_WEIGHT_KEYS,
        context=f"{context} score_weights",
    )
    aggregate_score_weights = _require_optimizer_weight_map(
        score_weight_config,
        "aggregate",
        expected_keys=EXPECTED_AGGREGATE_SCORE_WEIGHT_KEYS,
        context=f"{context} score_weights",
    )

    kelly_min_stake_usdc = _require_optimizer_float(
        optimizer_config,
        "kelly_min_stake_usdc",
        context=context,
    )
    if kelly_min_stake_usdc <= 0.0:
        raise ValueError("Optimizer config kelly_min_stake_usdc must be > 0.")

    start_bankroll_usdc = _require_optimizer_float(
        simulation_settings,
        "start_bankroll_usdc",
        context=f"{context} simulation",
    )
    if start_bankroll_usdc <= 0.0:
        raise ValueError("Optimizer config simulation.start_bankroll_usdc must be > 0.")

    prob_min_clip = float(model_proba_error_sim_config.get("prob_min_clip", 1e-6))
    if not 0.0 < prob_min_clip < 0.5:
        raise ValueError("Optimizer config model_proba_error_sim.prob_min_clip must be in (0, 0.5).")

    return optimizer_config, {
        "optuna": {
            "random_seed": int(optuna_settings["random_seed"]),
            "n_trials": int(optuna_settings["n_trials"]),
            "tpe_startup_trials": int(
                optuna_settings.get(
                    "tpe_startup_trials",
                    int(int(optuna_settings["n_trials"]) * 0.2),
                )
            ),
        },
        "trial_param_bounds": trial_param_bounds,
        "kelly_min_stake_usdc": float(kelly_min_stake_usdc),
        "data_split": {
            "holdout_fraction": float(data_split_settings["holdout_fraction"]),
        },
        "cv": {
            "target_fold_days": int(cv_settings["target_fold_days"]),
            "market_sim_seeds": [int(seed) for seed in cv_settings["market_sim_seeds"]],
        },
        "holdout": {
            "window_days": int(holdout_settings["window_days"]),
            "window_runs": int(holdout_settings["window_runs"]),
            "market_sim_seeds": [
                int(seed) for seed in holdout_settings["market_sim_seeds"]
            ],
            "full_market_sim_seeds": [
                int(seed) for seed in holdout_settings["full_market_sim_seeds"]
            ],
        },
        "fee_model": fee_model,
        "simulation": {
            "start_bankroll_usdc": float(start_bankroll_usdc),
        },
        "score_weights": {
            "seed": seed_score_weights,
            "aggregate": aggregate_score_weights,
        },
        "market_price_sim": build_market_price_sim_params(market_price_sim_config),
        "model_proba_error_sim": {
            "enabled": bool(model_proba_error_sim_config["enabled"]),
            "abs_diff_mean": float(model_proba_error_sim_config["abs_diff_mean"]),
            "abs_diff_max": float(model_proba_error_sim_config["abs_diff_max"]),
            "abs_diff_std": float(
                model_proba_error_sim_config.get(
                    "abs_diff_std",
                    float(model_proba_error_sim_config["abs_diff_max"]) / 3.0,
                )
            ),
            "prob_min_clip": float(prob_min_clip),
            "policy": str(model_proba_error_sim_config["policy"]),
        },
    }


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

optimizer_config, optimizer_settings = load_optimizer_settings(OPTIMIZER_CONFIG_PATH)
optuna_settings = optimizer_settings["optuna"]
trial_param_bounds = optimizer_settings["trial_param_bounds"]
data_split_settings = optimizer_settings["data_split"]
cv_settings = optimizer_settings["cv"]
holdout_settings = optimizer_settings["holdout"]
fee_model = optimizer_settings["fee_model"]
simulation_settings = optimizer_settings["simulation"]
score_weights = optimizer_settings["score_weights"]
price_sim_params = optimizer_settings["market_price_sim"]
model_proba_error_sim = optimizer_settings["model_proba_error_sim"]

# Manual live-vs-modeling probability error fit used only inside Kelly optimization.
# Update these by hand from the accepted parity audit snapshot.
MODEL_PROBA_ERROR_SEED_OFFSET = 1_000_000

scoring_formula = build_scoring_formula(score_weights)

# Optimizer-only market simulator. Live runtime still uses real orderbook quotes
# from Polymarket and must not inherit these offline execution assumptions.
price_sim_model = str(price_sim_params["model"])

TRADE_TRACE_COLUMNS = [
    "Opened",
    "trade_number",
    "side",
    "price",
    "edge",
    "fraction_f",
    "stake",
    "fee",
    "payout",
    "pnl_usdc",
    "win",
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


def sample_market_orderbook_arrays(
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
    n_rows = len(target)
    conviction_beta_alpha = float(price_sim_config["conviction_beta_alpha"])
    conviction_beta_beta = float(price_sim_config["conviction_beta_beta"])
    gap_min = float(price_sim_config["gap_min"])
    gap_max = float(price_sim_config["gap_max"])
    gap_gamma = float(price_sim_config["gap_gamma"])
    p_correct_min = float(price_sim_config["p_correct_min"])
    p_correct_max = float(price_sim_config["p_correct_max"])
    overround_min = float(price_sim_config["overround_min"])
    overround_max = float(price_sim_config["overround_max"])
    overround_gamma = float(price_sim_config["overround_gamma"])
    tick_size = float(price_sim_config["tick_size"])
    eps = float(price_sim_config["eps"])
    sim_order_min_size_shares = float(price_sim_config["sim_order_min_size_shares"])

    conviction = rng.beta(
        conviction_beta_alpha,
        conviction_beta_beta,
        size=n_rows,
    ).astype(np.float64, copy=False)
    abs_gap = gap_min + np.power(conviction, gap_gamma) * (gap_max - gap_min)
    # Allow either slope sign so directional market skill can rise or fall with
    # conviction while gap/overround remain conviction-linked.
    p_correct = p_correct_min + conviction * (p_correct_max - p_correct_min)
    overround = overround_min + np.power(conviction, overround_gamma) * (
        overround_max - overround_min
    )
    direction_is_correct = rng.random(n_rows).astype(np.float64, copy=False) < p_correct
    direction_sign = np.where(direction_is_correct, 1.0, -1.0)

    winner_ask = 0.5 + abs_gap / 2.0 + overround / 2.0
    loser_ask = 0.5 - abs_gap / 2.0 + overround / 2.0
    winner_is_up = (
        ((target == 1) & (direction_sign > 0.0))
        | ((target == 0) & (direction_sign < 0.0))
    )
    up_ask = np.where(winner_is_up, winner_ask, loser_ask)
    down_ask = np.where(winner_is_up, loser_ask, winner_ask)
    up_ask = np.clip(up_ask, eps, 1.0 - eps)
    down_ask = np.clip(down_ask, eps, 1.0 - eps)
    up_ask = np.round(up_ask / tick_size) * tick_size
    down_ask = np.round(down_ask / tick_size) * tick_size
    up_ask = np.clip(up_ask, eps, 1.0 - eps)
    down_ask = np.clip(down_ask, eps, 1.0 - eps)

    return {
        "conviction": conviction,
        "abs_gap": abs_gap,
        "p_correct": p_correct,
        "overround": overround,
        "winner_ask": winner_ask,
        "loser_ask": loser_ask,
        "up_ask": up_ask,
        "down_ask": down_ask,
        "direction_is_correct": direction_is_correct,
        "direction_sign": direction_sign,
        "sim_order_min_size_shares": np.full(
            n_rows,
            sim_order_min_size_shares,
            dtype=np.float64,
        ),
    }


def sample_model_proba_error_components(
    rng,
    n_rows,
):
    if not model_proba_error_sim["enabled"]:
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
    sampled_orderbook = sample_market_orderbook_arrays(
        target=target,
        scenario_seed=market_sim_seed,
        price_sim_config=price_sim_config,
    )

    up_ask = sampled_orderbook["up_ask"]
    down_ask = sampled_orderbook["down_ask"]
    order_min_size = sampled_orderbook["sim_order_min_size_shares"]

    up_fee_coef = float(fee_model["rate"]) * np.power(
        up_ask * (1.0 - up_ask),
        float(fee_model["exponent"]),
    ) / up_ask
    down_fee_coef = float(fee_model["rate"]) * np.power(
        down_ask * (1.0 - down_ask),
        float(fee_model["exponent"]),
    ) / down_ask

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
    if model_proba_error_sim["enabled"]:
        model_error_seed = int(market_sim_seed) + MODEL_PROBA_ERROR_SEED_OFFSET
        model_error_rng = np.random.default_rng(model_error_seed)
        model_error_abs_z, model_error_sign_bits = sample_model_proba_error_components(
            model_error_rng,
            len(target),
        )
        model_error_abs = np.maximum(
            float(model_proba_error_sim["abs_diff_mean"])
            + float(model_proba_error_sim["abs_diff_std"]) * model_error_abs_z,
            0.0,
        )
        model_error_abs = np.minimum(
            model_error_abs,
            float(model_proba_error_sim["abs_diff_max"]),
        )
        model_error_sign = np.where(model_error_sign_bits == 0, -1.0, 1.0)
        model_proba_error = model_error_sign * model_error_abs
    else:
        model_proba_error = np.zeros(len(target), dtype=np.float64)

    market_pick_up = np.where(
        up_ask > down_ask,
        1.0,
        np.where(down_ask > up_ask, 0.0, np.nan),
    )
    direction_correct_mask = np.where(
        np.isfinite(market_pick_up),
        market_pick_up == target.astype(np.float64, copy=False),
        np.nan,
    )
    valid_direction_rows = np.isfinite(direction_correct_mask)
    if np.any(valid_direction_rows):
        direction_correct_rate = float(np.mean(direction_correct_mask[valid_direction_rows]))
        direction_wrong_rate = float(
            np.mean(1.0 - direction_correct_mask[valid_direction_rows])
        )
    else:
        direction_correct_rate = float("nan")
        direction_wrong_rate = float("nan")
    market_diagnostics = {
        "direction_correct_rate": direction_correct_rate,
        "direction_wrong_rate": direction_wrong_rate,
        "direction_tie_rate": float(np.mean(~np.isfinite(market_pick_up))),
        "mean_abs_gap": float(np.mean(np.abs(up_ask - down_ask))),
        "q90_abs_gap": float(np.quantile(np.abs(up_ask - down_ask), 0.90)),
        "max_abs_gap": float(np.max(np.abs(up_ask - down_ask))),
        "mean_overround": float(np.mean(up_ask + down_ask - 1.0)),
        "q90_overround": float(np.quantile(up_ask + down_ask - 1.0, 0.90)),
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
        float(model_proba_error_sim["prob_min_clip"]),
        1.0 - float(model_proba_error_sim["prob_min_clip"]),
    )

    p = adjust_probability_for_kelly(
        p_simulated,
        min_clip=float(model_proba_error_sim["prob_min_clip"]),
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
    fee_round_decimals,
    min_fee,
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

                fee = float(np.round(stake * fee_coef[i], fee_round_decimals))

                if fee < min_fee:

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
    start_bankroll=None,
):
    if start_bankroll is None:
        start_bankroll = float(simulation_settings["start_bankroll_usdc"])

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
            fee_round_decimals=int(fee_model["fee_round_decimals"]),
            min_fee=float(fee_model["min_fee"]),
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
    start_bankroll=None,
):
    if start_bankroll is None:
        start_bankroll = float(simulation_settings["start_bankroll_usdc"])

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

    sum_g = 0.0

    sum_edge = 0.0

    sum_stake = 0.0

    sum_fraction = 0.0

    n_steps = end_idx - start_idx

    trade_log = {col: [] for col in TRADE_TRACE_COLUMNS}

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

        stake = 0.0

        fee = 0.0

        payout = 0.0

        pnl_usdc = 0.0

        g_t = 0.0

        win = None

        signal_up = bool(choose_up[i])

        resolved_win = bool(target[i] == 1) if signal_up else bool(target[i] == 0)

        if can_trade[i] and f_i > 0.0:

            stake = bankroll_before * f_i

            if stake >= min_stake_usdc:

                fee = round(
                    stake * float(fee_coef[i]),
                    int(fee_model["fee_round_decimals"]),
                )

                if fee < float(fee_model["min_fee"]):

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

                        g_t = float(np.log(bankroll / bankroll_before))

                        trades += 1

                        wins += int(win)

                        sum_g += g_t

                        sum_edge += edge_i

                        sum_stake += stake

                        sum_fraction += f_i

        if bankroll > equity_peak:

            equity_peak = bankroll

        drawdown = 0.0

        if equity_peak > 0.0:

            drawdown = (equity_peak - bankroll) / equity_peak

            if drawdown > max_drawdown:

                max_drawdown = drawdown

        if traded:
            trade_log["Opened"].append(pd.Timestamp(opened[i]))
            trade_log["trade_number"].append(int(trades))
            trade_log["side"].append("UP" if signal_up else "DOWN")
            trade_log["price"].append(float(price_i))
            trade_log["edge"].append(float(edge_i))
            trade_log["fraction_f"].append(float(f_i))
            trade_log["stake"].append(float(stake))
            trade_log["fee"].append(float(fee))
            trade_log["payout"].append(float(payout))
            trade_log["pnl_usdc"].append(float(pnl_usdc))
            trade_log["win"].append(bool(win))
            trade_log["bankroll_before"].append(float(bankroll_before))
            trade_log["bankroll_after"].append(float(bankroll))
            trade_log["drawdown"].append(float(drawdown))

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
        "trade_log": trade_log,
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
            start_bankroll=float(simulation_settings["start_bankroll_usdc"]),
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
        score_weights["seed"]["mean"] * np.mean(fold_scores_arr)
        + score_weights["seed"]["q10"] * np.quantile(fold_scores_arr, 0.10)
        + score_weights["seed"]["min"] * np.min(fold_scores_arr)
        - score_weights["seed"]["q90_drawdown_penalty"]
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
            "q90_fold_max_drawdown": float(seed_result["q90_fold_max_drawdown"]),
            "mean_fold_avg_fraction": float(seed_result["mean_fold_avg_fraction"]),
            "direction_correct_rate": float(diagnostics["direction_correct_rate"]),
            "direction_tie_rate": float(diagnostics["direction_tie_rate"]),
            "mean_abs_gap": float(diagnostics["mean_abs_gap"]),
            "mean_overround": float(diagnostics["mean_overround"]),
        }
        seed_results.append(compact_cv_seed_result(seed_output))

        fold_scores.extend(seed_result["fold_scores"])
        fold_trades.extend(seed_result["fold_trades"])
        fold_log_growth.extend(seed_result["fold_log_growth"])
        fold_max_drawdown.extend(seed_result["fold_max_drawdown"])
        fold_avg_fraction.extend(seed_result["fold_avg_fraction"])

    seed_score_summary = summarize_weighted_score(
        [float(item["score"]) for item in seed_results],
        score_weights["aggregate"],
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
        for name, bounds in trial_param_bounds.items()
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
        "kelly_min_stake_usdc": float(optimizer_settings["kelly_min_stake_usdc"]),
        "fee_model": {
            "source": str(fee_model["source"]),
            "rate": float(fee_model["rate"]),
            "exponent": float(fee_model["exponent"]),
            "fee_round_decimals": int(fee_model["fee_round_decimals"]),
            "min_fee": float(fee_model["min_fee"]),
        },
    }


def save_json(path, payload):

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:

        json.dump(payload, f, indent=2)


def compact_cv_seed_result(seed_result):
    return {
        "market_sim_seed": int(seed_result["market_sim_seed"]),
        "model_error_seed": seed_result["model_error_seed"],
        "score": float(seed_result["score"]),
        "mean_fold_score": float(seed_result["mean_fold_score"]),
        "q10_fold_score": float(seed_result["q10_fold_score"]),
        "min_fold_score": float(seed_result["min_fold_score"]),
        "mean_fold_trades": float(seed_result["mean_fold_trades"]),
        "mean_fold_log_growth": float(seed_result["mean_fold_log_growth"]),
        "q90_fold_max_drawdown": float(seed_result["q90_fold_max_drawdown"]),
        "mean_fold_avg_fraction": float(seed_result["mean_fold_avg_fraction"]),
        "direction_correct_rate": float(seed_result["direction_correct_rate"]),
        "direction_tie_rate": float(seed_result["direction_tie_rate"]),
        "mean_abs_gap": float(seed_result["mean_abs_gap"]),
        "mean_overround": float(seed_result["mean_overround"]),
    }


def compact_holdout_seed_result(seed_result):
    compact = {
        "market_sim_seed": int(seed_result["market_sim_seed"]),
        "model_error_seed": seed_result["model_error_seed"],
        "score": float(seed_result["score"]),
        "final_bankroll": float(seed_result["final_bankroll"]),
        "total_pnl": float(seed_result["total_pnl"]),
        "n_trades": int(seed_result["n_trades"]),
        "mean_log_growth": float(seed_result["mean_log_growth"]),
        "hit_rate": float(seed_result["hit_rate"]),
        "avg_edge": float(seed_result["avg_edge"]),
        "avg_stake": float(seed_result["avg_stake"]),
        "avg_fraction": float(seed_result["avg_fraction"]),
        "max_drawdown": float(seed_result["max_drawdown"]),
        "direction_correct_rate": float(seed_result["direction_correct_rate"]),
        "direction_tie_rate": float(seed_result["direction_tie_rate"]),
        "mean_abs_gap": float(seed_result["mean_abs_gap"]),
        "mean_overround": float(seed_result["mean_overround"]),
    }
    trace_path = seed_result.get("holdout_trade_trace_csv")
    if trace_path:
        compact["holdout_trade_trace_csv"] = str(trace_path)
    return compact


def build_holdout_window_output(window_spec, window_summary):
    return {
        "window_id": int(window_spec["window_id"]),
        "window_days": int(holdout_settings["window_days"]),
        "start_idx": int(window_spec["start_idx"]),
        "end_idx": int(window_spec["end_idx"]),
        "start_opened": str(window_spec["start_opened"]),
        "end_opened": str(window_spec["end_opened"]),
        "score": float(window_summary["score"]),
        "score_mean": float(window_summary["score_mean"]),
        "score_q10": float(window_summary["score_q10"]),
        "score_min": float(window_summary["score_min"]),
        "final_bankroll_mean": float(window_summary["final_bankroll_mean"]),
        "final_bankroll_worst": float(window_summary["final_bankroll_worst"]),
        "n_trades_mean": float(window_summary["n_trades_mean"]),
        "n_trades_worst": int(window_summary["n_trades_worst"]),
        "max_drawdown_mean": float(window_summary["max_drawdown_mean"]),
        "max_drawdown_worst": float(window_summary["max_drawdown_worst"]),
    }


def build_compact_trials_frame(study):
    rows = []

    for trial in study.trials:
        row = {
            "trial_number": int(trial.number),
            "score": float(trial.value) if trial.value is not None else np.nan,
            "fractional_kelly": float(trial.params["fractional_kelly"])
            if "fractional_kelly" in trial.params
            else np.nan,
            "cap": float(trial.params["cap"]) if "cap" in trial.params else np.nan,
            "min_edge": float(trial.params["min_edge"])
            if "min_edge" in trial.params
            else np.nan,
            "state": str(trial.state.name),
        }
        rows.append(row)

    trials_df = pd.DataFrame(rows)
    if trials_df.empty:
        return trials_df

    trials_df = trials_df.sort_values(
        by=["score", "trial_number"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    trials_df.insert(0, "rank", np.arange(1, len(trials_df) + 1, dtype=np.int64))
    return trials_df


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
    seed_scores = summarize_weighted_score(scores, score_weights["aggregate"])
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
            np.mean(final_bankrolls) - float(simulation_settings["start_bankroll_usdc"])
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
    trace_dir=None,
    trace_name_prefix=None,
    save_trade_traces=False,
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
            start_bankroll=float(simulation_settings["start_bankroll_usdc"]),
        )

        trace_path = None
        if save_trade_traces:
            if trace_dir is None or trace_name_prefix is None:
                raise ValueError(
                    "trace_dir and trace_name_prefix are required when save_trade_traces=True."
                )
            trace_path = trace_dir / (
                f"{trace_name_prefix}_market_seed_{int(market_sim_seed)}.csv"
            )
            pd.DataFrame(holdout_result["trade_log"]).to_csv(
                trace_path,
                index=False,
                columns=TRADE_TRACE_COLUMNS,
            )

        diagnostics = static_arrays["market_diagnostics"]
        seed_result = {
            "market_sim_seed": int(market_sim_seed),
            "model_error_seed": static_arrays["model_error_seed"],
            "score": float(holdout_result["fold_score"]),
            "final_bankroll": float(holdout_result["final_bankroll"]),
            "total_pnl": float(
                holdout_result["final_bankroll"]
                - float(simulation_settings["start_bankroll_usdc"])
            ),
            "n_trades": int(holdout_result["trades"]),
            "mean_log_growth": float(holdout_result["mean_g"]),
            "hit_rate": float(holdout_result["hit_rate"]),
            "avg_edge": float(holdout_result["avg_edge"]),
            "avg_stake": float(holdout_result["avg_stake"]),
            "avg_fraction": float(holdout_result["avg_fraction"]),
            "max_drawdown": float(holdout_result["max_drawdown"]),
            "direction_correct_rate": float(diagnostics["direction_correct_rate"]),
            "direction_tie_rate": float(diagnostics["direction_tie_rate"]),
            "mean_abs_gap": float(diagnostics["mean_abs_gap"]),
            "mean_overround": float(diagnostics["mean_overround"]),
        }
        if trace_path is not None:
            seed_result["holdout_trade_trace_csv"] = str(trace_path)
            trace_paths[str(int(market_sim_seed))] = str(trace_path)

        seed_results.append(compact_holdout_seed_result(seed_result))

    return {
        "summary": summarize_seed_run_results(seed_results),
        "seed_results": seed_results,
        "trace_paths": trace_paths,
    }


def main():

    print("optimize kelly polymarket | start")

    run_started_at_utc = datetime.now(timezone.utc)

    run_timestamp = run_started_at_utc.strftime("%Y%m%d_%H%M%S")

    full_holdout_trace_dir = SIMULATIONS_DIR / f"full_holdout_trade_trace_{run_timestamp}"

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
        f"model={price_sim_model} "
        f"conviction_beta_alpha={price_sim_params['conviction_beta_alpha']:.3f} "
        f"conviction_beta_beta={price_sim_params['conviction_beta_beta']:.3f} "
        f"gap=[{price_sim_params['gap_min']:.6f},{price_sim_params['gap_max']:.6f}] "
        f"gap_gamma={price_sim_params['gap_gamma']:.3f} "
        f"p_correct=[{price_sim_params['p_correct_min']:.3f},{price_sim_params['p_correct_max']:.3f}] "
        f"overround=[{price_sim_params['overround_min']:.6f},{price_sim_params['overround_max']:.6f}] "
        f"overround_gamma={price_sim_params['overround_gamma']:.3f} "
        f"tick_size={price_sim_params['tick_size']:.4f} "
        f"eps={price_sim_params['eps']:.8f} "
        f"sim_order_min_size_shares={price_sim_params['sim_order_min_size_shares']:.4f}"
    )

    print(
        "model error sim | "
        f"enabled={model_proba_error_sim['enabled']} "
        f"abs_mean={model_proba_error_sim['abs_diff_mean']:.6f} "
        f"abs_max={model_proba_error_sim['abs_diff_max']:.6f} "
        f"abs_std={model_proba_error_sim['abs_diff_std']:.6f} "
        f"policy={model_proba_error_sim['policy'] if model_proba_error_sim['enabled'] else 'disabled'}"
    )

    n_decision_rows = len(df5)

    n_trades = n_decision_rows - 1

    if n_trades <= 0:

        raise ValueError("Not enough decision rows to create trades.")

    target = df5["target_5m_candle_up"].to_numpy(dtype=np.int8, copy=False)[:-1]

    p_raw = p_raw_stats[:-1]

    opened_trade = df5["Opened"].iloc[:-1].to_numpy(dtype="datetime64[ns]", copy=False)

    split_idx = int(len(target) * (1.0 - float(data_split_settings["holdout_fraction"])))

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
        market_sim_seeds=cv_settings["market_sim_seeds"],
        price_sim_config=price_sim_params,
    )

    n_folds, approx_fold_days = infer_n_folds_for_target_days(
        n_obs=len(target_tune),
        target_fold_days=int(cv_settings["target_fold_days"]),
    )

    folds = make_walk_forward_folds(n_obs=len(target_tune), n_folds=n_folds)
    cv_price_sanity = summarize_static_price_range(
        [scenario["tune_static_arrays"] for scenario in market_sim_scenarios]
        + [scenario["holdout_static_arrays"] for scenario in market_sim_scenarios]
    )

    print(
        f"cv setup | n_folds={n_folds} approx_fold_days={approx_fold_days:.2f} "
        f"target_fold_days={cv_settings['target_fold_days']} "
        f"n_trials={optuna_settings['n_trials']} "
        f"n_market_sim_seeds={len(cv_settings['market_sim_seeds'])} "
        f"n_trades_tune={len(target_tune)} "
        f"first_fold=[{folds[0][0]}:{folds[0][1]}]"
    )

    print(
        f"data split | holdout_frac={data_split_settings['holdout_fraction']:.3f} "
        f"split_idx={split_idx} "
        f"tune_rows={len(target_tune)} holdout_rows={len(target_holdout)} "
        f"holdout_start_opened={pd.Timestamp(opened_holdout[0]).isoformat()} "
        f"holdout_end_opened={pd.Timestamp(opened_holdout[-1]).isoformat()}"
    )

    print(
        "market sim seeds | "
        f"cv={cv_settings['market_sim_seeds']} "
        f"holdout={holdout_settings['market_sim_seeds']} "
        f"full_holdout={holdout_settings['full_market_sim_seeds']}"
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
            min_stake_usdc=float(optimizer_settings["kelly_min_stake_usdc"]),
        )

        return float(cv_result["score"])

    sampler = optuna.samplers.TPESampler(
        seed=int(optuna_settings["random_seed"]),
        n_startup_trials=int(optuna_settings["tpe_startup_trials"]),
    )

    study_name = f"{STUDY_NAME_PREFIX}_{run_timestamp}"

    print(
        "optuna setup | "
        f"n_trials={optuna_settings['n_trials']} folds={len(folds)} "
        f"n_market_sim_seeds={len(cv_settings['market_sim_seeds'])} "
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
        n_trials=int(optuna_settings["n_trials"]),
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
        min_stake_usdc=float(optimizer_settings["kelly_min_stake_usdc"]),
    )

    holdout_window_specs = build_holdout_window_specs(
        opened_holdout=opened_holdout,
        window_days=int(holdout_settings["window_days"]),
        n_runs=int(holdout_settings["window_runs"]),
    )
    holdout_seed_union = sorted(
        set(holdout_settings["market_sim_seeds"]).union(
            holdout_settings["full_market_sim_seeds"]
        )
    )
    holdout_static_arrays_by_seed = build_holdout_static_arrays_by_seed(
        target_holdout=target_holdout,
        market_sim_seeds=holdout_seed_union,
        price_sim_config=price_sim_params,
    )
    holdout_price_sanity = summarize_static_price_range(
        list(holdout_static_arrays_by_seed.values())
    )

    print(
        "holdout setup | "
        f"window_days={holdout_settings['window_days']} "
        f"window_runs={holdout_settings['window_runs']} "
        f"holdout_market_sim_seed_count={len(holdout_settings['market_sim_seeds'])} "
        f"full_holdout_market_sim_seed_count={len(holdout_settings['full_market_sim_seeds'])}"
    )
    print(
        "price sanity | "
        f"holdout_min_ask={holdout_price_sanity['min_ask']:.6f} "
        f"holdout_max_ask={holdout_price_sanity['max_ask']:.6f}"
    )

    holdout_window_runs_output = []
    all_window_seed_results = []

    for window_spec in holdout_window_specs:
        window_eval = evaluate_holdout_for_seeds(
            target_holdout=target_holdout,
            p_raw_holdout=p_raw_holdout,
            opened_holdout=opened_holdout,
            holdout_static_arrays_by_seed=holdout_static_arrays_by_seed,
            market_sim_seeds=holdout_settings["market_sim_seeds"],
            fractional_kelly=best_fractional_kelly,
            cap=best_cap,
            min_edge=best_min_edge,
            min_stake_usdc=float(optimizer_settings["kelly_min_stake_usdc"]),
            start_idx=int(window_spec["start_idx"]),
            end_idx=int(window_spec["end_idx"]),
        )
        all_window_seed_results.extend(window_eval["seed_results"])
        holdout_window_runs_output.append(
            build_holdout_window_output(window_spec, window_eval["summary"])
        )

    holdout_window_summary = summarize_seed_run_results(all_window_seed_results)
    holdout_window_summary.update(
        {
            "window_days": int(holdout_settings["window_days"]),
            "window_run_count": int(holdout_settings["window_runs"]),
            "total_seed_runs": int(len(all_window_seed_results)),
        }
    )

    full_holdout_trace_dir.mkdir(parents=True, exist_ok=True)
    full_holdout_eval = evaluate_holdout_for_seeds(
        target_holdout=target_holdout,
        p_raw_holdout=p_raw_holdout,
        opened_holdout=opened_holdout,
        holdout_static_arrays_by_seed=holdout_static_arrays_by_seed,
        market_sim_seeds=holdout_settings["full_market_sim_seeds"],
        fractional_kelly=best_fractional_kelly,
        cap=best_cap,
        min_edge=best_min_edge,
        min_stake_usdc=float(optimizer_settings["kelly_min_stake_usdc"]),
        start_idx=0,
        end_idx=len(target_holdout),
        trace_dir=full_holdout_trace_dir,
        trace_name_prefix="full_holdout",
        save_trade_traces=True,
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

    build_compact_trials_frame(study).to_csv(
        trials_csv_path,
        index=False,
        float_format="%.6f",
    )

    run_output = {
        "generated_at_utc": run_started_at_utc.isoformat(),
        "input_path": str(INPUT_PATH),
        "optimizer_config_path": str(OPTIMIZER_CONFIG_PATH),
        "optimizer_config": optimizer_config,
        "study_name": study_name,
        "storage": OPTUNA_STORAGE,
        "runtime_config": runtime_config,
        "price_sim_model": price_sim_model,
        "price_sim_params": price_sim_params,
        "cv_market_sim_seeds": cv_settings["market_sim_seeds"],
        "holdout_market_sim_seeds": holdout_settings["market_sim_seeds"],
        "full_holdout_market_sim_seeds": holdout_settings["full_market_sim_seeds"],
        "model_proba_error_sim": model_proba_error_sim,
        "cv_meta": {
            "n_folds": int(n_folds),
            "target_fold_days": int(cv_settings["target_fold_days"]),
            "approx_fold_days": float(approx_fold_days),
            "n_trials": int(optuna_settings["n_trials"]),
            "seed": int(optuna_settings["random_seed"]),
            "trial_param_bounds": {
                name: [float(bounds[0]), float(bounds[1])]
                for name, bounds in trial_param_bounds.items()
            },
            "scoring_formula": scoring_formula,
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
            "window_days": int(holdout_settings["window_days"]),
            "window_run_count": int(holdout_settings["window_runs"]),
            "window_summary": holdout_window_summary,
            "window_runs": holdout_window_runs_output,
            "full_run": full_holdout_output,
        },
        "data_split": {
            "holdout_frac": float(data_split_settings["holdout_fraction"]),
            "split_idx": int(split_idx),
            "tune_rows": len(target_tune),
            "holdout_rows": len(target_holdout),
            "holdout_start_opened": pd.Timestamp(opened_holdout[0]).isoformat(),
            "holdout_end_opened": pd.Timestamp(opened_holdout[-1]).isoformat(),
        },
        "artifacts": {
            "trials_csv": str(trials_csv_path),
            "full_holdout_trade_trace_dir": str(full_holdout_trace_dir),
            "full_holdout_trade_trace_csv_by_market_seed": full_holdout_eval[
                "trace_paths"
            ],
            "trade_trace_columns": TRADE_TRACE_COLUMNS,
            "trade_trace_policy": (
                "Only executed trades from full holdout are persisted to CSV; "
                "holdout windows are summarized in JSON only."
            ),
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
        f"saved full holdout trade traces | dir={full_holdout_trace_dir} "
        f"n_files={len(holdout_settings['full_market_sim_seeds'])}"
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
