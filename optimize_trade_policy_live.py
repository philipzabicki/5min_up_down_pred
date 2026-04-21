import json
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from common_config_utils import load_json_object
from market_price_sim import load_live_market_empirical_frame, sample_market_orderbook_arrays
from polymarket_fee_utils import (
    normalize_polymarket_fee_model,
)
from project_config import load_runtime_artifact_paths
from trade_policy import (
    build_trade_intent,
    decide_trade_from_ev,
    load_trade_policy_runtime_config,
    resolve_fee_fractions_from_quotes,
)

optuna.logging.set_verbosity(optuna.logging.INFO)

CONFIG_PATH = Path("configs/trade_policy_optimizer_config.json")
RUNTIME_CONFIG_PATH = Path(
    load_runtime_artifact_paths()["trade_policy_runtime_config_path"]
)
OPTUNA_OUTPUT_DIR = Path("data/optuna/trade_policy_live")
OPTUNA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_STORAGE = "sqlite:///data/optuna/databases/trade_policy_live.db"
OBJECTIVE_VERSION = "ev_policy_activity_v4"
DEFAULT_STUDY_NAME = f"trade_policy_live_{OBJECTIVE_VERSION}"


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _require_object(payload, key):
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a JSON object.")
    return dict(value)


def _optional_object(payload, key):
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a JSON object when provided.")
    return dict(value)


def _optional_list(payload, key):
    value = payload.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Config key '{key}' must be a JSON array when provided.")
    return list(value)


def _require_bounds(payload, key):
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Config key '{key}' must be a [lo, hi] array.")
    lo = float(value[0])
    hi = float(value[1])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
        raise ValueError(f"Config key '{key}' has invalid bounds: {value!r}")
    return lo, hi


def resolve_study_name(optuna_settings):
    raw_study_name = str(optuna_settings.get("study_name", "")).strip()
    if not raw_study_name:
        return DEFAULT_STUDY_NAME
    if raw_study_name == DEFAULT_STUDY_NAME:
        return raw_study_name
    if raw_study_name.endswith(OBJECTIVE_VERSION):
        return raw_study_name
    return DEFAULT_STUDY_NAME


def resolve_trial_budget(optuna_settings, *, existing_complete_trials):
    trials_per_run = int(
        optuna_settings.get("trials_per_run", optuna_settings["n_trials"])
    )
    if trials_per_run <= 0:
        raise ValueError("optuna.trials_per_run must be > 0.")

    max_total_trials_raw = optuna_settings.get("max_total_trials")
    max_total_trials = (
        None if max_total_trials_raw is None else int(max_total_trials_raw)
    )
    if max_total_trials is not None and max_total_trials <= 0:
        raise ValueError("optuna.max_total_trials must be > 0 when provided.")

    if max_total_trials is None:
        return {
            "trials_this_run": int(trials_per_run),
            "trials_per_run": int(trials_per_run),
            "max_total_trials": None,
            "remaining_until_cap": None,
        }

    remaining_until_cap = max(0, int(max_total_trials) - int(existing_complete_trials))
    trials_this_run = min(int(trials_per_run), int(remaining_until_cap))
    return {
        "trials_this_run": int(trials_this_run),
        "trials_per_run": int(trials_per_run),
        "max_total_trials": int(max_total_trials),
        "remaining_until_cap": int(remaining_until_cap),
    }


def load_optimizer_settings(config_path=CONFIG_PATH):
    payload = load_json_object(config_path)
    optuna_settings = _require_object(payload, "optuna")
    replay_data = _require_object(payload, "replay_data")
    reporting = _require_object(payload, "reporting")
    simulation = _require_object(payload, "simulation")
    runtime_defaults = _require_object(payload, "runtime_defaults")
    if "stake_usdc" in runtime_defaults:
        raise ValueError(
            "trade_policy_optimizer.runtime_defaults.stake_usdc was removed; "
            "use stake_multiplier."
        )
    bounds = _require_object(payload, "trial_param_bounds")
    cv_settings = _optional_object(payload, "cv")
    cv_score_weights = _optional_object(cv_settings, "score_weights")
    market_price_sim = _optional_object(payload, "market_price_sim")
    seed_trials = _optional_list(payload, "seed_trials")

    study_name = resolve_study_name(optuna_settings)
    cv_weights = {
        "full_train": float(cv_score_weights.get("full_train", 0.5)),
        "fold_mean": float(cv_score_weights.get("fold_mean", 0.3)),
        "fold_min": float(cv_score_weights.get("fold_min", 0.2)),
    }
    cv_weight_total = sum(cv_weights.values())
    if cv_weight_total <= 0.0:
        raise ValueError("cv.score_weights must sum to a positive value.")
    cv_weights = {
        key: float(value / cv_weight_total) for key, value in cv_weights.items()
    }

    replay_settings = {
        "trade_csv_glob": replay_data.get("trade_csv_glob"),
        "shared_csv_path": str(replay_data["shared_csv_path"]),
        "timestamp_col": str(replay_data.get("timestamp_col", "prediction_time")),
        "default_order_min_size_shares": float(
            replay_data.get("default_order_min_size_shares", 0.0)
        ),
        "recent_resolved_rows": (
            None
            if replay_data.get("recent_resolved_rows") is None
            else int(replay_data["recent_resolved_rows"])
        ),
        "preferred_model_hash": (
            None
            if replay_data.get("preferred_model_hash") in (None, "")
            else str(replay_data["preferred_model_hash"]).strip()
        ),
        "max_prediction_delay_ms": (
            None
            if replay_data.get("max_prediction_delay_ms") is None
            else float(replay_data["max_prediction_delay_ms"])
        ),
        "max_decision_delay_ms": (
            None
            if replay_data.get("max_decision_delay_ms") is None
            else float(replay_data["max_decision_delay_ms"])
        ),
        "max_market_lookup_ms": (
            None
            if replay_data.get("max_market_lookup_ms") is None
            else float(replay_data["max_market_lookup_ms"])
        ),
        "max_submit_order_ms": (
            None
            if replay_data.get("max_submit_order_ms") is None
            else float(replay_data["max_submit_order_ms"])
        ),
        "max_execution_ms": (
            None
            if replay_data.get("max_execution_ms") is None
            else float(replay_data["max_execution_ms"])
        ),
    }

    market_price_sim_settings = {}
    if market_price_sim:
        market_price_sim_settings = {
            "enabled": bool(market_price_sim.get("enabled", True)),
            "scenario_count": int(market_price_sim.get("scenario_count", 4)),
            "scenario_seed_offset": int(
                market_price_sim.get("scenario_seed_offset", 100_000)
            ),
            "model": str(market_price_sim["model"]),
            "trade_csv_glob": market_price_sim.get(
                "trade_csv_glob",
                replay_settings["trade_csv_glob"],
            ),
            "shared_csv_path": str(
                market_price_sim.get(
                    "shared_csv_path",
                    replay_settings["shared_csv_path"],
                )
            ),
            "recent_resolved_rows": (
                replay_settings["recent_resolved_rows"]
                if market_price_sim.get("recent_resolved_rows") is None
                else int(market_price_sim["recent_resolved_rows"])
            ),
            "preferred_model_hash": (
                replay_settings["preferred_model_hash"]
                if market_price_sim.get("preferred_model_hash") in (None, "")
                else str(market_price_sim["preferred_model_hash"]).strip()
            ),
            "max_prediction_delay_ms": (
                replay_settings["max_prediction_delay_ms"]
                if market_price_sim.get("max_prediction_delay_ms") is None
                else float(market_price_sim["max_prediction_delay_ms"])
            ),
            "max_decision_delay_ms": (
                replay_settings["max_decision_delay_ms"]
                if market_price_sim.get("max_decision_delay_ms") is None
                else float(market_price_sim["max_decision_delay_ms"])
            ),
            "max_market_lookup_ms": (
                replay_settings["max_market_lookup_ms"]
                if market_price_sim.get("max_market_lookup_ms") is None
                else float(market_price_sim["max_market_lookup_ms"])
            ),
            "max_submit_order_ms": (
                replay_settings["max_submit_order_ms"]
                if market_price_sim.get("max_submit_order_ms") is None
                else float(market_price_sim["max_submit_order_ms"])
            ),
            "max_execution_ms": (
                replay_settings["max_execution_ms"]
                if market_price_sim.get("max_execution_ms") is None
                else float(market_price_sim["max_execution_ms"])
            ),
            "min_pool_rows": int(market_price_sim.get("min_pool_rows", 250)),
            "elapsed_quantile_bins": int(
                market_price_sim.get("elapsed_quantile_bins", 12)
            ),
            "tick_size": float(market_price_sim.get("tick_size", 0.01)),
            "eps": float(market_price_sim.get("eps", 1e-6)),
            "sim_order_min_size_shares": float(
                market_price_sim.get("sim_order_min_size_shares", 1.0)
            ),
        }
        if market_price_sim_settings["scenario_count"] <= 0:
            raise ValueError("market_price_sim.scenario_count must be > 0.")

    return {
        "optuna": {
            "random_seed": int(optuna_settings["random_seed"]),
            "n_trials": int(optuna_settings["n_trials"]),
            "trials_per_run": int(
                optuna_settings.get("trials_per_run", optuna_settings["n_trials"])
            ),
            "tpe_startup_trials": int(optuna_settings["tpe_startup_trials"]),
            "max_total_trials": (
                None
                if optuna_settings.get("max_total_trials") is None
                else int(optuna_settings["max_total_trials"])
            ),
            "study_name": study_name,
        },
        "replay_data": replay_settings,
        "reporting": {
            "top_n_candidates": int(reporting["top_n_candidates"]),
        },
        "cv": {
            "folds": int(cv_settings.get("folds", 4)),
            "min_rows_per_fold": int(cv_settings.get("min_rows_per_fold", 100)),
            "inactive_fold_score": float(cv_settings.get("inactive_fold_score", -0.35)),
            "score_weights": cv_weights,
        },
        "seed_trials": [dict(item) for item in seed_trials if isinstance(item, dict)],
        "simulation": {
            "start_bankroll_usdc": float(simulation["start_bankroll_usdc"]),
        },
        "runtime_defaults": {
            "extra_buffer": float(runtime_defaults.get("extra_buffer", 0.0)),
            "stake_multiplier": float(runtime_defaults.get("stake_multiplier", 1.0)),
            "fee_model": normalize_polymarket_fee_model(
                runtime_defaults["fee_model"],
                context="trade_policy_optimizer.runtime_defaults.fee_model",
            ),
        },
        "market_price_sim": market_price_sim_settings,
        "bounds": {key: _require_bounds(bounds, key) for key in bounds},
    }


def load_replay_rows(settings):
    replay_cfg = settings["replay_data"]
    frame = load_live_market_empirical_frame(
        trade_csv_glob=replay_cfg.get("trade_csv_glob"),
        shared_csv_path=replay_cfg.get("shared_csv_path"),
        recent_resolved_rows=replay_cfg.get("recent_resolved_rows"),
        preferred_model_hash=replay_cfg.get("preferred_model_hash"),
        max_prediction_delay_ms=replay_cfg.get("max_prediction_delay_ms"),
        max_decision_delay_ms=replay_cfg.get("max_decision_delay_ms"),
        max_market_lookup_ms=replay_cfg.get("max_market_lookup_ms"),
        max_submit_order_ms=replay_cfg.get("max_submit_order_ms"),
        max_execution_ms=replay_cfg.get("max_execution_ms"),
    )
    timestamp_col = replay_cfg["timestamp_col"]
    if timestamp_col in frame.columns:
        frame = frame.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("Replay CSV has no resolved rows with quotes.")
    return frame


def build_synthetic_replay_rows(rows, *, price_sim_config, scenario_seed):
    synthetic = rows.copy()
    sampled = sample_market_orderbook_arrays(
        target=rows["actual_up"].to_numpy(dtype=np.int8, copy=False),
        market_elapsed_ms=rows["market_elapsed_ms"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),
        scenario_seed=int(scenario_seed),
        price_sim_config=price_sim_config,
    )
    synthetic["pm_up_best_ask"] = sampled["up_ask"]
    synthetic["pm_down_best_ask"] = sampled["down_ask"]
    if "sim_order_min_size_shares" in sampled:
        synthetic["pm_order_min_size"] = sampled["sim_order_min_size_shares"]
    if "source_path" in synthetic.columns:
        synthetic["source_path"] = f"synthetic:{price_sim_config['model']}:{int(scenario_seed)}"
    return synthetic.reset_index(drop=True)


def evaluate_synthetic_objective(
    real_rows,
    cv_blocks,
    runtime_config,
    *,
    simulation_settings,
    cv_settings,
    market_price_sim_settings,
    default_order_min_size_shares,
    trial_number,
):
    if not market_price_sim_settings.get("enabled"):
        return None

    scenario_count = int(market_price_sim_settings["scenario_count"])
    scenario_seed_offset = int(market_price_sim_settings["scenario_seed_offset"])
    scenario_scores = []
    scenario_train_scores = []
    scenario_fold_scores = []

    for scenario_idx in range(scenario_count):
        base_seed = (
            scenario_seed_offset
            + int(trial_number) * 10_000
            + int(scenario_idx) * 100
        )
        synthetic_train_rows = build_synthetic_replay_rows(
            real_rows,
            price_sim_config=market_price_sim_settings,
            scenario_seed=base_seed,
        )
        synthetic_train_metrics = replay_policy(
            synthetic_train_rows,
            runtime_config,
            start_bankroll_usdc=simulation_settings["start_bankroll_usdc"],
            default_order_min_size_shares=default_order_min_size_shares,
        )
        synthetic_train_score = score_metrics(
            synthetic_train_metrics,
            start_bankroll_usdc=simulation_settings["start_bankroll_usdc"],
            no_trade_score=cv_settings["inactive_fold_score"],
        )
        synthetic_fold_metrics = []
        synthetic_fold_scores = []
        for block_idx, block in enumerate(cv_blocks):
            synthetic_block_rows = build_synthetic_replay_rows(
                block,
                price_sim_config=market_price_sim_settings,
                scenario_seed=base_seed + block_idx + 1,
            )
            metrics = replay_policy(
                synthetic_block_rows,
                runtime_config,
                start_bankroll_usdc=simulation_settings["start_bankroll_usdc"],
                default_order_min_size_shares=default_order_min_size_shares,
            )
            synthetic_fold_metrics.append(metrics)
            synthetic_fold_scores.append(
                score_metrics(
                    metrics,
                    start_bankroll_usdc=simulation_settings["start_bankroll_usdc"],
                    no_trade_score=cv_settings["inactive_fold_score"],
                )
            )
        scenario_score = aggregate_cv_scores(
            synthetic_train_score,
            synthetic_fold_scores,
            cv_settings["score_weights"],
        )
        scenario_scores.append(float(scenario_score))
        scenario_train_scores.append(float(synthetic_train_score))
        scenario_fold_scores.append([float(score) for score in synthetic_fold_scores])

    return {
        "enabled": True,
        "scenario_count": int(scenario_count),
        "mean_objective_score": float(np.mean(scenario_scores)),
        "mean_train_score": float(np.mean(scenario_train_scores)),
        "scenario_scores": [float(score) for score in scenario_scores],
        "scenario_fold_scores": scenario_fold_scores,
    }


def build_cv_blocks(frame, *, requested_folds, min_rows_per_fold):
    requested_folds = int(requested_folds)
    min_rows_per_fold = int(min_rows_per_fold)
    if requested_folds < 2:
        raise ValueError("cv.folds must be >= 2.")
    if min_rows_per_fold < 1:
        raise ValueError("cv.min_rows_per_fold must be >= 1.")
    if len(frame) < 2:
        raise ValueError("Need at least 2 rows to build cv blocks.")

    max_folds_by_min_rows = max(1, len(frame) // min_rows_per_fold)
    fold_count = min(requested_folds, max_folds_by_min_rows)
    if fold_count < 2:
        fold_count = min(requested_folds, len(frame))
    fold_count = max(2, min(fold_count, len(frame)))

    row_indexes = np.arange(len(frame))
    blocks = [
        frame.iloc[idxs].reset_index(drop=True)
        for idxs in np.array_split(row_indexes, fold_count)
        if len(idxs) > 0
    ]
    if len(blocks) < 2:
        raise ValueError("CV block builder produced fewer than 2 non-empty blocks.")
    return blocks


def build_optimization_signature(settings):
    payload = {
        "objective_version": OBJECTIVE_VERSION,
        "bounds": settings["bounds"],
        "cv": settings["cv"],
        "replay_data": {
            "trade_csv_glob": settings["replay_data"]["trade_csv_glob"],
            "shared_csv_path": settings["replay_data"]["shared_csv_path"],
            "timestamp_col": settings["replay_data"]["timestamp_col"],
            "default_order_min_size_shares": settings["replay_data"][
                "default_order_min_size_shares"
            ],
            "recent_resolved_rows": settings["replay_data"]["recent_resolved_rows"],
            "preferred_model_hash": settings["replay_data"]["preferred_model_hash"],
            "max_prediction_delay_ms": settings["replay_data"]["max_prediction_delay_ms"],
            "max_decision_delay_ms": settings["replay_data"]["max_decision_delay_ms"],
            "max_market_lookup_ms": settings["replay_data"]["max_market_lookup_ms"],
            "max_submit_order_ms": settings["replay_data"]["max_submit_order_ms"],
            "max_execution_ms": settings["replay_data"]["max_execution_ms"],
        },
        "runtime_defaults": settings["runtime_defaults"],
        "simulation": settings["simulation"],
        "market_price_sim": settings.get("market_price_sim", {}),
        "reporting": {"top_n_candidates": settings["reporting"]["top_n_candidates"]},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def trial_matches_signature(trial, *, optimization_signature):
    return (
        str(trial.user_attrs.get("objective_version", "")) == OBJECTIVE_VERSION
        and str(trial.user_attrs.get("optimization_signature", ""))
        == str(optimization_signature)
    )


def trial_params_match(trial, params):
    if set(trial.params.keys()) != set(params.keys()):
        return False
    for key, value in params.items():
        trial_value = trial.params.get(key)
        if isinstance(value, bool):
            if bool(trial_value) != bool(value):
                return False
            continue
        if not math.isclose(float(trial_value), float(value), rel_tol=0.0, abs_tol=1e-12):
            return False
    return True


def enqueue_seed_trials(study, seed_trials):
    if not seed_trials:
        return 0

    enqueued = 0
    existing_trials = list(study.trials)
    for seed_params in seed_trials:
        if any(trial_params_match(trial, seed_params) for trial in existing_trials):
            continue
        study.enqueue_trial(seed_params)
        existing_trials.append(type("TrialLike", (), {"params": dict(seed_params)})())
        enqueued += 1
    return enqueued


def fee_model_from_row(row, fallback_fee_model):
    try:
        payload = {
            "rate": float(row.pm_fee_rate),
            "exponent": float(row.pm_fee_exponent),
            "fee_round_decimals": int(row.pm_fee_round_decimals),
            "min_fee": float(row.pm_min_fee_usdc),
            "source": str(getattr(row, "pm_fee_source", "") or fallback_fee_model["source"]),
        }
        return normalize_polymarket_fee_model(payload, context="replay_row.fee_model")
    except Exception:
        return fallback_fee_model


def replay_policy(
    rows,
    runtime_config,
    *,
    start_bankroll_usdc,
    default_order_min_size_shares,
):
    bankroll = float(start_bankroll_usdc)
    peak = float(bankroll)
    max_drawdown = 0.0
    executed = 0
    wins = 0
    expected_pnl_adj = 0.0
    expected_pnl_raw = 0.0

    for row in rows.itertuples(index=False):
        fee_model = fee_model_from_row(row, runtime_config["fee_model"])
        order_min_size = float(
            getattr(row, "pm_order_min_size", default_order_min_size_shares)
            or default_order_min_size_shares
        )
        fee_fractions = resolve_fee_fractions_from_quotes(
            ask_yes=float(row.pm_up_best_ask),
            ask_no=float(row.pm_down_best_ask),
            fee_model=fee_model,
        )
        policy_result = decide_trade_from_ev(
            float(row.proba_up),
            float(row.pm_up_best_ask),
            float(row.pm_down_best_ask),
            float(fee_fractions["fee_yes"]),
            float(fee_fractions["fee_no"]),
            float(runtime_config["extra_buffer"]),
        )
        decision = build_trade_intent(
            policy_result=policy_result,
            bankroll=float(bankroll),
            stake_multiplier=float(runtime_config["stake_multiplier"]),
            fee_model=fee_model,
            order_min_size=order_min_size,
        )
        if decision.get("final_reason") != "ok":
            peak = max(peak, bankroll)
            if peak > 0.0:
                max_drawdown = max(max_drawdown, 1.0 - bankroll / peak)
            continue

        executed += 1
        trade_side = str(decision["trade_side"])
        prob_win = (
            float(row.proba_up) if trade_side == "yes" else float(1.0 - float(row.proba_up))
        )
        expected_pnl_adj += prob_win * float(decision["shares_net"]) - float(
            decision["bet_usdc"]
        )
        expected_pnl_raw += prob_win * float(decision["shares_net"]) - float(
            decision["bet_usdc"]
        )
        is_win = (trade_side == "yes" and int(row.actual_up) == 1) or (
            trade_side == "no" and int(row.actual_up) == 0
        )
        stake = float(decision["bet_usdc"])
        payout = float(decision["shares_net"]) if is_win else 0.0
        pnl = payout - stake
        bankroll += pnl
        wins += int(is_win)
        peak = max(peak, bankroll)
        if peak > 0.0:
            max_drawdown = max(max_drawdown, 1.0 - bankroll / peak)

    trade_rate = float(executed / len(rows)) if len(rows) else 0.0
    return {
        "final_bankroll": float(bankroll),
        "pnl": float(bankroll - float(start_bankroll_usdc)),
        "executed": int(executed),
        "trade_rate": float(trade_rate),
        "win_rate": float(wins / executed) if executed else float("nan"),
        "max_drawdown": float(max_drawdown),
        "expected_pnl_adj": float(expected_pnl_adj),
        "expected_pnl_raw": float(expected_pnl_raw),
    }


def expected_pnl_adj(metrics):
    return float(metrics.get("expected_pnl_adj", metrics.get("pnl", 0.0)))


def edge_capture_ratio(metrics):
    expected = expected_pnl_adj(metrics)
    if not math.isfinite(expected) or expected <= 0.0:
        return 0.0
    pnl = float(metrics.get("pnl", 0.0))
    if not math.isfinite(pnl):
        return 0.0
    return float(pnl / expected)


def clipped_positive_capture(capture, *, max_capture=1.0):
    if not math.isfinite(capture):
        return 0.0
    return float(min(max(float(capture), 0.0), float(max_capture)))


def score_metrics(metrics, *, start_bankroll_usdc, no_trade_score=-1e9):
    if metrics["executed"] <= 0:
        return float(no_trade_score)
    pnl_ratio = float(metrics["pnl"]) / float(start_bankroll_usdc)
    positive_expected_pnl = max(0.0, expected_pnl_adj(metrics))
    expected_pnl_ratio = float(positive_expected_pnl) / float(start_bankroll_usdc)
    captured_expected_ratio = float(
        clipped_positive_capture(edge_capture_ratio(metrics)) * expected_pnl_ratio
    )
    realization_gap_ratio = max(
        0.0,
        float(positive_expected_pnl) - max(0.0, float(metrics["pnl"])),
    ) / float(start_bankroll_usdc)
    return float(
        0.65 * pnl_ratio
        + 0.25 * captured_expected_ratio
        - 0.15 * realization_gap_ratio
        - 0.10 * float(metrics["max_drawdown"])
    )


def aggregate_cv_scores(full_train_score, fold_scores, score_weights):
    if not fold_scores:
        raise ValueError("fold_scores must be non-empty.")
    return float(
        float(score_weights["full_train"]) * float(full_train_score)
        + float(score_weights["fold_mean"]) * float(np.mean(fold_scores))
        + float(score_weights["fold_min"]) * float(np.min(fold_scores))
    )


def build_candidate_report_row(item):
    synthetic_objective = item.get("synthetic_objective") or {}
    return {
        "trial_number": int(item["trial_number"]),
        "objective_score": float(item["objective_score"]),
        "real_test_score": float(item["real_test_score"]),
        "real_test_final_bankroll": float(item["real_test_metrics"]["final_bankroll"]),
        "real_test_max_drawdown": float(item["real_test_metrics"]["max_drawdown"]),
        "real_test_pnl": float(item["real_test_metrics"]["pnl"]),
        "real_test_expected_pnl_adj": float(
            item["real_test_metrics"].get("expected_pnl_adj", 0.0)
        ),
        "real_test_expected_pnl_raw": float(
            item["real_test_metrics"].get("expected_pnl_raw", 0.0)
        ),
        "real_test_executed": int(item["real_test_metrics"]["executed"]),
        "synthetic_scenario_count": int(synthetic_objective.get("scenario_count", 0)),
        "synthetic_mean_objective_score": float(
            item.get("synthetic_objective_score") or 0.0
        ),
        "synthetic_mean_train_score": float(
            synthetic_objective.get("mean_train_score", 0.0)
        ),
        "runtime_config": json.dumps(item["runtime_config"], sort_keys=True),
    }


def summarize_candidate(item):
    return {
        "trial_number": int(item["trial_number"]),
        "objective_score": float(item["objective_score"]),
        "real_test_score": float(item["real_test_score"]),
        "real_test_metrics": item["real_test_metrics"],
        "synthetic_objective": item.get("synthetic_objective"),
        "runtime_config": item["runtime_config"],
    }


def build_runtime_config_from_trial(trial, settings):
    bounds = settings["bounds"]
    defaults = settings["runtime_defaults"]

    runtime = {
        "extra_buffer": trial.suggest_float("extra_buffer", *bounds["extra_buffer"]),
        "stake_multiplier": float(defaults["stake_multiplier"]),
        "fee_model": defaults["fee_model"],
    }
    return load_trade_policy_runtime_config_dict(runtime)


def load_trade_policy_runtime_config_dict(payload):
    temp_path = OPTUNA_OUTPUT_DIR / "_runtime_validation_tmp.json"
    temp_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        return load_trade_policy_runtime_config(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    print("optimize trade policy live | start")
    settings = load_optimizer_settings()
    rows = load_replay_rows(settings)
    replay_cfg = settings["replay_data"]
    latency_filters = {
        key: replay_cfg[key]
        for key in (
            "max_prediction_delay_ms",
            "max_decision_delay_ms",
            "max_market_lookup_ms",
            "max_submit_order_ms",
            "max_execution_ms",
        )
        if replay_cfg.get(key) is not None
    }
    print(
        "replay rows | "
        f"total={len(rows)} real_test={len(rows)} "
        f"trade_csv_glob={replay_cfg.get('trade_csv_glob')} "
        f"shared_csv={replay_cfg.get('shared_csv_path')} "
        f"source_files={rows['source_path'].nunique() if 'source_path' in rows.columns else 'na'} "
        f"latency_filters={latency_filters}"
    )
    cv_blocks = build_cv_blocks(
        rows,
        requested_folds=settings["cv"]["folds"],
        min_rows_per_fold=settings["cv"]["min_rows_per_fold"],
    )
    optimization_signature = build_optimization_signature(settings)
    print(
        "cv blocks | "
        f"count={len(cv_blocks)} "
        f"rows={','.join(str(len(block)) for block in cv_blocks)}"
    )
    if settings["market_price_sim"]:
        print(
            "market price sim | "
            f"enabled={settings['market_price_sim'].get('enabled', False)} "
            f"model={settings['market_price_sim'].get('model')} "
            f"scenario_count={settings['market_price_sim'].get('scenario_count')} "
            f"elapsed_quantile_bins={settings['market_price_sim'].get('elapsed_quantile_bins')}"
        )

    sim = settings["simulation"]

    def objective(trial):
        runtime_config = build_runtime_config_from_trial(trial, settings)
        synthetic_objective = evaluate_synthetic_objective(
            rows,
            cv_blocks,
            runtime_config,
            simulation_settings=sim,
            cv_settings=settings["cv"],
            market_price_sim_settings=settings["market_price_sim"],
            default_order_min_size_shares=settings["replay_data"][
                "default_order_min_size_shares"
            ],
            trial_number=trial.number,
        )
        if synthetic_objective is None:
            raise ValueError(
                "market_price_sim.enabled must be true to optimize on synthetic prices."
            )
        objective_score = float(synthetic_objective["mean_objective_score"])
        trial.set_user_attr("objective_version", OBJECTIVE_VERSION)
        trial.set_user_attr("optimization_signature", optimization_signature)
        trial.set_user_attr("runtime_config", runtime_config)
        trial.set_user_attr("synthetic_objective", synthetic_objective)
        trial.set_user_attr("synthetic_objective_score", float(objective_score))
        trial.set_user_attr("objective_score", float(objective_score))
        return float(objective_score)

    sampler = optuna.samplers.TPESampler(
        seed=int(settings["optuna"]["random_seed"]),
        n_startup_trials=int(settings["optuna"]["tpe_startup_trials"]),
    )
    study = optuna.create_study(
        study_name=settings["optuna"]["study_name"],
        direction="maximize",
        sampler=sampler,
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
    )
    enqueued_seed_trials = enqueue_seed_trials(study, settings["seed_trials"])
    existing_complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and trial_matches_signature(
            trial,
            optimization_signature=optimization_signature,
        )
    ]
    trial_budget = resolve_trial_budget(
        settings["optuna"],
        existing_complete_trials=len(existing_complete_trials),
    )
    max_total_trials_text = (
        "none"
        if trial_budget["max_total_trials"] is None
        else str(trial_budget["max_total_trials"])
    )
    remaining_until_cap_text = (
        "none"
        if trial_budget["remaining_until_cap"] is None
        else str(trial_budget["remaining_until_cap"])
    )
    print(
        "study state | "
        f"name={settings['optuna']['study_name']} "
        f"signature={optimization_signature} "
        f"enqueued_seed_trials={enqueued_seed_trials} "
        f"existing_complete={len(existing_complete_trials)} "
        f"trials_per_run={trial_budget['trials_per_run']} "
        f"trials_this_run={trial_budget['trials_this_run']} "
        f"max_total_trials={max_total_trials_text} "
        f"remaining_until_cap={remaining_until_cap_text}"
    )
    if trial_budget["trials_this_run"] > 0:
        study.optimize(objective, n_trials=trial_budget["trials_this_run"])

    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and trial_matches_signature(
            trial,
            optimization_signature=optimization_signature,
        )
    ]
    complete_trials.sort(key=lambda trial: float(trial.value), reverse=True)
    candidates = []
    for trial in complete_trials:
        runtime_config = trial.user_attrs["runtime_config"]
        real_test_metrics = replay_policy(
            rows,
            runtime_config,
            start_bankroll_usdc=sim["start_bankroll_usdc"],
            default_order_min_size_shares=settings["replay_data"]["default_order_min_size_shares"],
        )
        real_test_score = score_metrics(
            real_test_metrics,
            start_bankroll_usdc=sim["start_bankroll_usdc"],
        )
        candidates.append(
            {
                "trial_number": int(trial.number),
                "objective_score": float(trial.value),
                "real_test_score": float(real_test_score),
                "runtime_config": runtime_config,
                "real_test_metrics": real_test_metrics,
                "synthetic_objective": trial.user_attrs.get("synthetic_objective"),
                "synthetic_objective_score": float(
                    trial.user_attrs.get("synthetic_objective_score", trial.value)
                ),
            }
        )

    if not candidates:
        raise RuntimeError("No completed trials available for runtime selection.")
    candidates.sort(
        key=lambda item: (
            float(item["objective_score"]),
            float(item["real_test_score"]),
            float(item["real_test_metrics"]["pnl"]),
            int(item["real_test_metrics"]["executed"]),
            -float(item["real_test_metrics"]["max_drawdown"]),
        ),
        reverse=True,
    )
    report_top_n = min(int(settings["reporting"]["top_n_candidates"]), len(candidates))
    candidate_pool_summary = {
        "candidate_count": int(len(candidates)),
        "real_test_profitable_count": int(
            sum(float(item["real_test_metrics"]["pnl"]) > 0.0 for item in candidates)
        ),
        "real_test_expected_positive_count": int(
            sum(
                float(item["real_test_metrics"].get("expected_pnl_adj", 0.0)) > 0.0
                for item in candidates
            )
        ),
    }
    best_objective_candidate = summarize_candidate(candidates[0])
    best_real_test_candidate = max(
        candidates,
        key=lambda item: (
            float(item["real_test_score"]),
            float(item["real_test_metrics"]["pnl"]),
            int(item["real_test_metrics"]["executed"]),
            -float(item["real_test_metrics"]["max_drawdown"]),
        ),
    )
    selected = candidates[0]
    runtime_config = selected["runtime_config"]
    real_test_pnl = float(selected["real_test_metrics"]["pnl"])
    real_test_expected_pnl_adj = float(
        selected["real_test_metrics"].get("expected_pnl_adj", real_test_pnl)
    )
    should_save_runtime = True

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_summary = {
        "objective_version": OBJECTIVE_VERSION,
        "optimization_signature": optimization_signature,
        "study_name": settings["optuna"]["study_name"],
        "saved_runtime_config_path": str(RUNTIME_CONFIG_PATH),
        "runtime_config_saved": True,
        "candidate_pool_summary": candidate_pool_summary,
        "best_objective_candidate": best_objective_candidate,
        "best_real_test_candidate": summarize_candidate(best_real_test_candidate),
        "selected_candidate": summarize_candidate(selected),
        "selected_trial_number": int(selected["trial_number"]),
        "selected_objective_score": float(selected["objective_score"]),
        "selected_real_test_score": float(selected["real_test_score"]),
        "selected_real_test_metrics": selected["real_test_metrics"],
        "selected_synthetic_objective": selected.get("synthetic_objective"),
        "runtime_config": runtime_config,
        "top_objective_candidates": [
            summarize_candidate(item) for item in candidates[:report_top_n]
        ],
        "settings": settings,
    }
    run_summary_path = OPTUNA_OUTPUT_DIR / f"trade_policy_live_run_{timestamp}.json"
    trials_csv_path = OPTUNA_OUTPUT_DIR / f"trade_policy_live_trials_{timestamp}.csv"
    save_json(run_summary_path, run_summary)
    pd.DataFrame(
        [build_candidate_report_row(item) for item in candidates]
    ).to_csv(trials_csv_path, index=False)

    save_json(RUNTIME_CONFIG_PATH, runtime_config)

    print(
        "candidate summary | "
        f"total={candidate_pool_summary['candidate_count']} "
        f"real_test_profitable={candidate_pool_summary['real_test_profitable_count']} "
        "real_test_expected_positive="
        f"{candidate_pool_summary['real_test_expected_positive_count']}"
    )
    print(
        "selected runtime | "
        f"trial={selected['trial_number']} "
        f"objective_score={selected['objective_score']:.6f} "
        f"real_test_score={selected['real_test_score']:.6f} "
        f"real_test_pnl={real_test_pnl:.6f} "
        f"real_test_expected_pnl_adj={real_test_expected_pnl_adj:.6f}"
    )
    print(f"saved runtime config | path={RUNTIME_CONFIG_PATH}")
    print(f"saved optuna run | path={run_summary_path}")
    print(f"saved optuna trials | path={trials_csv_path}")


if __name__ == "__main__":
    main()
