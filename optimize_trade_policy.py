import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from utils.config import load_json_object, path_to_portable_str
from utils.polymarket import (
    polymarket_taker_fee_fraction_of_notional,
    polymarket_taker_fee_usdc_from_notional,
)
from utils.project_config import load_live_profile, load_runtime_artifact_paths
from utils.trading import decide_trade_from_ev, load_trade_policy_runtime_config


LIVE_DIR = Path("data/live")
OOF_PREDICTIONS_FALLBACK_PATH = Path(
    "data/datasets/modeling/BTCUSD_INDEXVOL_UM_BTCUSDT1m_oof_predictions.parquet"
)
OUTPUT_DIR = Path("data/optuna/trade_policy_live")
RUNTIME_CONFIG_FALLBACK_PATH = Path("configs/runtime/trade_policy_project.json")
STUDY_NAME = "trade_policy_live_oof_live_prices_v1"
N_TRIALS = 250
SEED = 37
TPE_STARTUP_TRIALS = 40
EXTRA_BUFFER_BOUNDS = (0.0, 0.05)
SUBMITTED_PRICE_SLIPPAGE_TICKS_BOUNDS = (0, 8)
TICK_SIZE_FALLBACK = 0.01
ORDER_PRICE_CAP = None
ORDER_PRICE_CAP_FALLBACK = 0.95
SAVE_RUNTIME_CONFIG = True
NO_TRADE_OBJECTIVE = -1_000_000.0
OBJECTIVE_DOWNSIDE_EPS = 1e-12

BUCKET_TIME_COL = "bucket_start"
OOF_TIME_COL = "Opened"
OOF_PRED_COL = "oof_pred_proba_up"

PRICE_COLUMN_GROUPS = {
    "btc_open": ("bucket_open_price", "btc_open"),
    "btc_close": ("bucket_close_price", "btc_close"),
    "ask_yes": ("pm_up_best_ask", "ask_yes", "policy_ask_yes", "up_ask_price"),
    "ask_no": ("pm_down_best_ask", "ask_no", "policy_ask_no", "down_ask_price"),
    "tick_size": ("pm_tick_size",),
}
PRICE_VALUE_COLUMNS = tuple(PRICE_COLUMN_GROUPS)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return path_to_portable_str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def resolve_runtime_config_path():
    try:
        return Path(load_runtime_artifact_paths()["trade_policy_path"])
    except Exception:
        return RUNTIME_CONFIG_FALLBACK_PATH


def resolve_oof_path():
    try:
        model_meta_path = Path(load_runtime_artifact_paths()["model_meta_path"])
        model_meta = load_json_object(model_meta_path)
        oof_info = model_meta.get("oof_predictions")
        if isinstance(oof_info, dict) and oof_info.get("path"):
            return Path(str(oof_info["path"]))
    except Exception:
        pass
    return OOF_PREDICTIONS_FALLBACK_PATH


def resolve_order_price_cap():
    if ORDER_PRICE_CAP is not None:
        cap = float(ORDER_PRICE_CAP)
    else:
        try:
            cap = float(load_live_profile()["polymarket_order_price_cap"])
        except Exception:
            cap = ORDER_PRICE_CAP_FALLBACK
    if not math.isfinite(cap) or not (0.0 < cap < 1.0):
        raise ValueError(f"order_price_cap must be in (0, 1), got: {cap!r}")
    return float(cap)


def _to_utc_minute(values):
    timestamps = pd.to_datetime(values, errors="coerce", utc=True)
    return timestamps.dt.tz_convert(None).dt.floor("min")


def _median_across_columns(frame, columns):
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    numeric = frame[present].apply(pd.to_numeric, errors="coerce")
    return numeric.median(axis=1, skipna=True)


def load_live_price_observations(live_dir=LIVE_DIR):
    live_dir = Path(live_dir)
    if not live_dir.exists():
        raise FileNotFoundError(f"Live price directory not found: {live_dir}")

    frames = []
    used_files = []
    skipped_files = []
    for csv_path in sorted(live_dir.rglob("*.csv")):
        try:
            columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
        except Exception as exc:
            skipped_files.append({"path": str(csv_path), "reason": str(exc)})
            continue
        if BUCKET_TIME_COL not in columns:
            continue

        price_columns = sorted(
            {
                column
                for group in PRICE_COLUMN_GROUPS.values()
                for column in group
                if column in columns
            }
        )
        if not price_columns:
            continue

        usecols = [BUCKET_TIME_COL] + price_columns
        try:
            raw = pd.read_csv(csv_path, usecols=usecols)
        except Exception as exc:
            skipped_files.append({"path": str(csv_path), "reason": str(exc)})
            continue

        observation = pd.DataFrame(
            {"bucket_time": _to_utc_minute(raw[BUCKET_TIME_COL])}
        )
        for output_col, candidate_cols in PRICE_COLUMN_GROUPS.items():
            observation[output_col] = _median_across_columns(raw, candidate_cols)
        observation = observation.dropna(subset=["bucket_time"])
        observation = observation.dropna(subset=list(PRICE_VALUE_COLUMNS), how="all")
        if observation.empty:
            continue

        frames.append(observation)
        used_files.append(str(csv_path))

    if not frames:
        raise ValueError(f"No usable price observations found under {live_dir}.")
    return pd.concat(frames, ignore_index=True), {
        "live_dir": str(live_dir),
        "used_file_count": int(len(used_files)),
        "used_files": used_files,
        "skipped_files": skipped_files,
    }


def aggregate_price_observations(observations):
    if observations.empty:
        raise ValueError("Price observation frame is empty.")
    missing = {"bucket_time"}.difference(observations.columns)
    if missing:
        raise ValueError(f"Price observations missing columns: {sorted(missing)}")

    value_cols = [column for column in PRICE_VALUE_COLUMNS if column in observations]
    if not value_cols:
        raise ValueError("Price observations contain no price value columns.")
    grouped = observations.groupby("bucket_time", as_index=False)[value_cols].median()
    counts = (
        observations.groupby("bucket_time", as_index=False)
        .size()
        .rename(columns={"size": "price_observation_count"})
    )
    return grouped.merge(counts, on="bucket_time", how="left")


def load_live_price_frame(live_dir=LIVE_DIR):
    observations, summary = load_live_price_observations(live_dir)
    prices = aggregate_price_observations(observations)
    duplicate_buckets = int((prices["price_observation_count"] > 1).sum())
    summary.update(
        {
            "raw_observation_count": int(len(observations)),
            "bucket_count": int(len(prices)),
            "duplicate_bucket_count": duplicate_buckets,
            "time_min": (
                prices["bucket_time"].min().isoformat() if not prices.empty else None
            ),
            "time_max": (
                prices["bucket_time"].max().isoformat() if not prices.empty else None
            ),
        }
    )
    return prices, summary


def load_oof_predictions(oof_path, *, time_col=OOF_TIME_COL, pred_col=OOF_PRED_COL):
    oof_path = Path(oof_path)
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {oof_path}")
    frame = pd.read_parquet(oof_path, columns=[time_col, pred_col])
    out = pd.DataFrame(
        {
            "bucket_time": _to_utc_minute(frame[time_col]),
            "proba_up": pd.to_numeric(frame[pred_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["bucket_time", "proba_up"])
    out = out[np.isfinite(out["proba_up"]) & out["proba_up"].between(0.0, 1.0)]
    if out.empty:
        raise ValueError(f"No usable OOF predictions in {oof_path}.")
    return (
        out.groupby("bucket_time", as_index=False)["proba_up"]
        .median()
        .sort_values("bucket_time")
        .reset_index(drop=True)
    )


def build_backtest_frame(oof_predictions, live_prices):
    merged = oof_predictions.merge(live_prices, on="bucket_time", how="inner")
    required = ["proba_up", "btc_open", "btc_close", "ask_yes", "ask_no"]
    merged = merged.dropna(subset=required).copy()
    for column in required + ["tick_size"]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")

    valid = (
        np.isfinite(merged["proba_up"])
        & merged["proba_up"].between(0.0, 1.0)
        & np.isfinite(merged["btc_open"])
        & np.isfinite(merged["btc_close"])
        & (merged["btc_open"] > 0.0)
        & (merged["btc_close"] > 0.0)
        & np.isfinite(merged["ask_yes"])
        & np.isfinite(merged["ask_no"])
        & merged["ask_yes"].between(0.0, 1.0, inclusive="neither")
        & merged["ask_no"].between(0.0, 1.0, inclusive="neither")
    )
    merged = merged.loc[valid].copy()
    if merged.empty:
        raise ValueError("OOF predictions and live prices have no usable aligned rows.")

    if "tick_size" not in merged.columns:
        merged["tick_size"] = TICK_SIZE_FALLBACK
    merged["tick_size"] = merged["tick_size"].where(
        np.isfinite(merged["tick_size"]) & (merged["tick_size"] > 0.0),
        TICK_SIZE_FALLBACK,
    )
    merged["actual_up"] = (merged["btc_close"] > merged["btc_open"]).astype(np.int8)
    keep_cols = [
        "bucket_time",
        "proba_up",
        "actual_up",
        "btc_open",
        "btc_close",
        "ask_yes",
        "ask_no",
        "tick_size",
    ]
    if "price_observation_count" in merged.columns:
        keep_cols.append("price_observation_count")
    return merged.loc[:, keep_cols].sort_values("bucket_time").reset_index(drop=True)


def resolve_submitted_buy_price(
    *,
    entry_price,
    order_price_cap,
    submitted_price_mode,
    tick_size,
    slippage_ticks,
):
    entry_price = float(entry_price)
    order_price_cap = float(order_price_cap)
    if not math.isfinite(entry_price) or not (0.0 < entry_price < 1.0):
        return float("nan"), "invalid_entry_price"

    mode = str(submitted_price_mode or "entry_price").strip().lower()
    if mode in {"", "entry_price"}:
        return float(entry_price), ""

    if not math.isfinite(order_price_cap) or not (0.0 < order_price_cap < 1.0):
        return float("nan"), "invalid_order_price_cap"
    if entry_price > order_price_cap + 1e-12:
        return float("nan"), "entry_price_above_order_price_cap"

    if mode == "entry_price_plus_ticks":
        tick_size = float(tick_size)
        ticks = int(slippage_ticks)
        if not math.isfinite(tick_size) or tick_size <= 0.0:
            return float("nan"), "invalid_tick_size"
        if ticks < 0:
            return float("nan"), "invalid_submitted_price_slippage_ticks"
        submitted_price = min(entry_price + ticks * tick_size, order_price_cap)
        submitted_price = round(float(submitted_price), 4)
        if not (0.0 < submitted_price < 1.0):
            return float("nan"), "invalid_submitted_price"
        return float(submitted_price), ""

    if mode == "order_price_cap":
        return float(order_price_cap), ""
    return float("nan"), f"unsupported_submitted_price_mode:{mode}"


def _fee_fraction(price, fee_model):
    value = polymarket_taker_fee_fraction_of_notional(float(price), fee_model)
    if value is None or not math.isfinite(float(value)):
        return float("nan")
    return float(value)


def replay_policy(backtest_frame, runtime_config, *, order_price_cap):
    if str(runtime_config.get("mode", "ev")).strip().lower() != "ev":
        raise ValueError("optimize_trade_policy.py currently optimizes EV mode only.")

    fee_model = runtime_config["fee_model"]
    extra_buffer = float(runtime_config["extra_buffer"])
    submitted_price_mode = runtime_config.get("submitted_price_mode", "entry_price")
    slippage_ticks = int(runtime_config.get("submitted_price_slippage_ticks", 0))

    pnls = []
    returns = []
    sides = []
    skipped_no_trade = 0
    skipped_submission = 0
    wins = 0
    total_stake = 0.0

    for row in backtest_frame.itertuples(index=False):
        fee_yes = _fee_fraction(row.ask_yes, fee_model)
        fee_no = _fee_fraction(row.ask_no, fee_model)
        policy_result = decide_trade_from_ev(
            float(row.proba_up),
            float(row.ask_yes),
            float(row.ask_no),
            fee_yes,
            fee_no,
            extra_buffer,
        )
        decision = str(policy_result.get("decision", "no_trade"))
        if decision == "no_trade":
            skipped_no_trade += 1
            continue

        if decision == "buy_yes":
            side = "yes"
            entry_price = float(row.ask_yes)
            is_win = int(row.actual_up) == 1
        elif decision == "buy_no":
            side = "no"
            entry_price = float(row.ask_no)
            is_win = int(row.actual_up) == 0
        else:
            skipped_no_trade += 1
            continue

        submitted_price, submitted_error = resolve_submitted_buy_price(
            entry_price=entry_price,
            order_price_cap=order_price_cap,
            submitted_price_mode=submitted_price_mode,
            tick_size=float(row.tick_size),
            slippage_ticks=slippage_ticks,
        )
        if submitted_error or not math.isfinite(submitted_price):
            skipped_submission += 1
            continue

        stake = float(submitted_price)
        fee_result = polymarket_taker_fee_usdc_from_notional(
            stake,
            submitted_price,
            fee_model,
        )
        fee_usdc = float(fee_result["fee_usdc"])
        shares_net = (stake - fee_usdc) / submitted_price
        pnl = (shares_net if is_win else 0.0) - stake

        pnls.append(float(pnl))
        returns.append(float(pnl / stake))
        sides.append(side)
        wins += int(is_win)
        total_stake += float(stake)

    executed = len(pnls)
    row_count = int(len(backtest_frame))
    if executed == 0:
        return {
            "objective_score": NO_TRADE_OBJECTIVE,
            "row_count": row_count,
            "executed": 0,
            "trade_rate": 0.0,
            "win_rate": float("nan"),
            "total_profit_usdc": 0.0,
            "total_stake_usdc": 0.0,
            "mean_pnl_usdc": float("nan"),
            "std_pnl_usdc": float("nan"),
            "mean_return": float("nan"),
            "std_return": float("nan"),
            "downside_deviation": float("nan"),
            "objective_downside_denominator": float("nan"),
            "max_drawdown_usdc": 0.0,
            "yes_trades": 0,
            "no_trades": 0,
            "skipped_no_trade": int(skipped_no_trade),
            "skipped_submission": int(skipped_submission),
        }

    pnl_array = np.asarray(pnls, dtype=np.float64)
    return_array = np.asarray(returns, dtype=np.float64)
    cumulative = np.cumsum(pnl_array)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:]
    max_drawdown = float(np.max(peak - cumulative)) if cumulative.size else 0.0
    std_pnl = float(np.std(pnl_array, ddof=1)) if executed > 1 else 0.0
    total_profit = float(np.sum(pnl_array))
    mean_return = float(np.mean(return_array))
    downside_returns = np.minimum(return_array, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside_returns**2)))
    objective_denominator = max(abs(downside_deviation), OBJECTIVE_DOWNSIDE_EPS)
    objective = float(total_profit / objective_denominator)

    return {
        "objective_score": float(objective),
        "row_count": row_count,
        "executed": int(executed),
        "trade_rate": float(executed / row_count) if row_count else 0.0,
        "win_rate": float(wins / executed),
        "total_profit_usdc": total_profit,
        "total_stake_usdc": float(total_stake),
        "mean_pnl_usdc": float(np.mean(pnl_array)),
        "std_pnl_usdc": std_pnl,
        "mean_return": mean_return,
        "std_return": float(np.std(return_array, ddof=1)) if executed > 1 else 0.0,
        "downside_deviation": downside_deviation,
        "objective_downside_denominator": float(objective_denominator),
        "max_drawdown_usdc": max_drawdown,
        "yes_trades": int(sum(side == "yes" for side in sides)),
        "no_trades": int(sum(side == "no" for side in sides)),
        "skipped_no_trade": int(skipped_no_trade),
        "skipped_submission": int(skipped_submission),
    }


def build_trial_runtime_config(base_runtime_config, *, extra_buffer, slippage_ticks):
    runtime = dict(base_runtime_config)
    runtime["extra_buffer"] = float(extra_buffer)
    runtime["submitted_price_slippage_ticks"] = int(slippage_ticks)
    return runtime


def enqueue_seed_trials(study, base_runtime_config):
    extra_buffer_low, extra_buffer_high = EXTRA_BUFFER_BOUNDS
    slippage_ticks_low, slippage_ticks_high = SUBMITTED_PRICE_SLIPPAGE_TICKS_BOUNDS
    candidates = [
        {
            "extra_buffer": float(base_runtime_config["extra_buffer"]),
            "submitted_price_slippage_ticks": int(
                base_runtime_config.get("submitted_price_slippage_ticks", 0)
            ),
        },
        {"extra_buffer": 0.0, "submitted_price_slippage_ticks": 0},
        {"extra_buffer": 0.01, "submitted_price_slippage_ticks": 0},
        {"extra_buffer": 0.01, "submitted_price_slippage_ticks": 2},
        {"extra_buffer": 0.02, "submitted_price_slippage_ticks": 2},
    ]
    seen = set()
    enqueued = 0
    for params in candidates:
        extra_buffer = float(params["extra_buffer"])
        slippage_ticks = int(params["submitted_price_slippage_ticks"])
        if not (
            extra_buffer_low <= extra_buffer <= extra_buffer_high
            and slippage_ticks_low <= slippage_ticks <= slippage_ticks_high
        ):
            continue
        key = (round(extra_buffer, 12), slippage_ticks)
        if key in seen:
            continue
        study.enqueue_trial(
            {
                "extra_buffer": extra_buffer,
                "submitted_price_slippage_ticks": slippage_ticks,
            }
        )
        seen.add(key)
        enqueued += 1
    return enqueued


def trial_report_row(trial):
    metrics = trial.user_attrs.get("metrics", {})
    return {
        "trial_number": int(trial.number),
        "objective_score": float(trial.value),
        "extra_buffer": float(trial.params["extra_buffer"]),
        "submitted_price_slippage_ticks": int(
            trial.params["submitted_price_slippage_ticks"]
        ),
        **metrics,
    }


def build_selected_runtime_payload(raw_runtime_payload, best_params):
    payload = dict(raw_runtime_payload)
    payload["extra_buffer"] = float(best_params["extra_buffer"])
    payload["submitted_price_slippage_ticks"] = int(
        best_params["submitted_price_slippage_ticks"]
    )
    return payload


def validate_optimizer_settings(*, n_trials):
    extra_buffer_low, extra_buffer_high = EXTRA_BUFFER_BOUNDS
    slippage_ticks_low, slippage_ticks_high = SUBMITTED_PRICE_SLIPPAGE_TICKS_BOUNDS
    if int(n_trials) <= 0:
        raise ValueError("N_TRIALS must be > 0.")
    if extra_buffer_low < 0.0 or extra_buffer_low > extra_buffer_high:
        raise ValueError("Invalid EXTRA_BUFFER_BOUNDS.")
    if slippage_ticks_low < 0 or slippage_ticks_low > slippage_ticks_high:
        raise ValueError("Invalid SUBMITTED_PRICE_SLIPPAGE_TICKS_BOUNDS.")


def run_optimization(*, n_trials=None, save_runtime_config=None):
    n_trials = N_TRIALS if n_trials is None else int(n_trials)
    save_runtime_config = (
        SAVE_RUNTIME_CONFIG if save_runtime_config is None else bool(save_runtime_config)
    )
    validate_optimizer_settings(n_trials=n_trials)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    runtime_config_path = resolve_runtime_config_path()
    oof_path = resolve_oof_path()
    order_price_cap = resolve_order_price_cap()
    extra_buffer_low, extra_buffer_high = EXTRA_BUFFER_BOUNDS
    slippage_ticks_low, slippage_ticks_high = SUBMITTED_PRICE_SLIPPAGE_TICKS_BOUNDS

    raw_runtime_payload = load_json_object(runtime_config_path)
    base_runtime_config = load_trade_policy_runtime_config(runtime_config_path)
    live_prices, live_price_summary = load_live_price_frame(LIVE_DIR)
    oof_predictions = load_oof_predictions(oof_path)
    backtest_frame = build_backtest_frame(oof_predictions, live_prices)

    print(
        "trade policy optimizer | "
        f"runtime={runtime_config_path} "
        f"oof={oof_path} live_dir={LIVE_DIR}"
    )
    print(
        "aligned replay rows | "
        f"rows={len(backtest_frame)} "
        f"price_buckets={live_price_summary['bucket_count']} "
        f"price_files={live_price_summary['used_file_count']} "
        f"order_price_cap={order_price_cap:.4f}"
    )

    sampler = optuna.samplers.TPESampler(
        seed=int(SEED),
        n_startup_trials=int(TPE_STARTUP_TRIALS),
    )
    study = optuna.create_study(
        study_name=str(STUDY_NAME),
        direction="maximize",
        sampler=sampler,
    )
    enqueued_seed_trials = enqueue_seed_trials(study, base_runtime_config)

    def objective(trial):
        runtime_config = build_trial_runtime_config(
            base_runtime_config,
            extra_buffer=trial.suggest_float(
                "extra_buffer",
                float(extra_buffer_low),
                float(extra_buffer_high),
            ),
            slippage_ticks=trial.suggest_int(
                "submitted_price_slippage_ticks",
                int(slippage_ticks_low),
                int(slippage_ticks_high),
            ),
        )
        metrics = replay_policy(
            backtest_frame,
            runtime_config,
            order_price_cap=order_price_cap,
        )
        trial.set_user_attr("runtime_config", runtime_config)
        trial.set_user_attr("metrics", metrics)
        return float(metrics["objective_score"])

    study.optimize(objective, n_trials=int(n_trials))
    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not complete_trials:
        raise RuntimeError("No completed Optuna trials.")
    complete_trials.sort(key=lambda trial: float(trial.value), reverse=True)
    best_trial = complete_trials[0]
    selected_payload = build_selected_runtime_payload(
        raw_runtime_payload,
        best_trial.params,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_summary_path = OUTPUT_DIR / f"trade_policy_live_run_{timestamp}.json"
    trials_csv_path = OUTPUT_DIR / f"trade_policy_live_trials_{timestamp}.csv"

    trial_rows = [trial_report_row(trial) for trial in complete_trials]
    pd.DataFrame(trial_rows).to_csv(trials_csv_path, index=False)
    run_summary = {
        "study_name": str(STUDY_NAME),
        "runtime_config_path": runtime_config_path,
        "runtime_config_saved": bool(save_runtime_config),
        "oof_path": oof_path,
        "live_price_summary": live_price_summary,
        "aligned_replay_rows": int(len(backtest_frame)),
        "order_price_cap": float(order_price_cap),
        "objective_formula": "total_profit / max(abs(downside_deviation), 1e-12)",
        "n_trials": int(n_trials),
        "seed": int(SEED),
        "tpe_startup_trials": int(TPE_STARTUP_TRIALS),
        "enqueued_seed_trials": int(enqueued_seed_trials),
        "search_space": {
            "extra_buffer": [float(extra_buffer_low), float(extra_buffer_high)],
            "submitted_price_slippage_ticks": [
                int(slippage_ticks_low),
                int(slippage_ticks_high),
            ],
        },
        "base_runtime_config": base_runtime_config,
        "selected_params": dict(best_trial.params),
        "selected_metrics": best_trial.user_attrs["metrics"],
        "selected_runtime_payload": selected_payload,
        "top_trials": trial_rows[:20],
    }
    save_json(run_summary_path, run_summary)

    if save_runtime_config:
        save_json(runtime_config_path, selected_payload)

    metrics = best_trial.user_attrs["metrics"]
    print(
        "selected policy | "
        f"trial={best_trial.number} "
        f"objective={float(best_trial.value):.6f} "
        f"profit={metrics['total_profit_usdc']:.6f} "
        f"executed={metrics['executed']} "
        f"extra_buffer={float(best_trial.params['extra_buffer']):.6f} "
        "submitted_price_slippage_ticks="
        f"{int(best_trial.params['submitted_price_slippage_ticks'])}"
    )
    if save_runtime_config:
        print(f"saved runtime config | path={runtime_config_path}")
    else:
        print("runtime config not saved")
    print(f"saved optuna run | path={run_summary_path}")
    print(f"saved optuna trials | path={trials_csv_path}")
    return run_summary


def main():
    run_optimization()


if __name__ == "__main__":
    main()
