import json
from pathlib import Path

import numpy as np
import pandas as pd

from common_config_utils import load_json_object

LIVE_CSV_LIMIT = 10
LIVE_CSV_PATHS = sorted(
    Path("data/live/trade").glob("*.csv"),
    key=lambda path: path.stat().st_mtime,
)[-LIVE_CSV_LIMIT:]
OPTIMIZER_CONFIG_PATH = Path("configs/kelly_optimizer_config.json")
N_ROWS_SIMULATED = None
SEED = 37


def load_market_price_sim_config(config_path):
    payload = load_json_object(config_path)
    market_price_sim = payload.get("market_price_sim")
    if not isinstance(market_price_sim, dict):
        raise ValueError(f"Missing market_price_sim in {config_path}")
    if str(market_price_sim.get("model")) != "latent_conviction_directional":
        raise ValueError(
            "analyze_market_sim_vs_live.py expects "
            "market_price_sim.model='latent_conviction_directional'."
        )
    return market_price_sim


def load_live_trade_frame(csv_paths):
    if not csv_paths:
        raise ValueError("LIVE_CSV_PATHS is empty.")

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        rename_map = {}
        if "pm_up_best_ask" in frame.columns and "up_best_ask" not in frame.columns:
            rename_map["pm_up_best_ask"] = "up_best_ask"
        if "pm_down_best_ask" in frame.columns and "down_best_ask" not in frame.columns:
            rename_map["pm_down_best_ask"] = "down_best_ask"
        if "pm_tick_size" in frame.columns and "tick_size" not in frame.columns:
            rename_map["pm_tick_size"] = "tick_size"
        frame = frame.rename(columns=rename_map)

        required_columns = {"up_best_ask", "down_best_ask", "actual_up"}
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")

        keep_columns = ["up_best_ask", "down_best_ask", "actual_up"]
        if "tick_size" in frame.columns:
            keep_columns.append("tick_size")

        cleaned = frame.loc[:, keep_columns].copy()
        cleaned["source_csv"] = str(csv_path)
        frames.append(cleaned)

    live = pd.concat(frames, ignore_index=True)
    live = live.dropna(subset=["up_best_ask", "down_best_ask", "actual_up"]).copy()
    live["actual_up"] = pd.to_numeric(live["actual_up"], errors="coerce")
    live = live[live["actual_up"].isin([0.0, 1.0])].copy()
    if live.empty:
        raise ValueError("No resolved live rows with asks and actual_up.")

    live["actual_up"] = live["actual_up"].astype(np.int8)
    live["up_best_ask"] = pd.to_numeric(live["up_best_ask"], errors="coerce")
    live["down_best_ask"] = pd.to_numeric(live["down_best_ask"], errors="coerce")
    if "tick_size" in live.columns:
        live["tick_size"] = pd.to_numeric(live["tick_size"], errors="coerce")

    live = live.dropna(subset=["up_best_ask", "down_best_ask"]).reset_index(drop=True)
    if live.empty:
        raise ValueError("No usable live ask rows after numeric coercion.")

    return live


def sample_market_orderbook_arrays(target, scenario_seed, price_sim_config):
    target = np.asarray(target, dtype=np.int8)
    rng = np.random.default_rng(int(scenario_seed))
    n_rows = len(target)

    conviction = rng.beta(
        float(price_sim_config["conviction_beta_alpha"]),
        float(price_sim_config["conviction_beta_beta"]),
        size=n_rows,
    )
    abs_gap = float(price_sim_config["gap_min"]) + np.power(
        conviction,
        float(price_sim_config["gap_gamma"]),
    ) * (float(price_sim_config["gap_max"]) - float(price_sim_config["gap_min"]))
    p_correct = float(price_sim_config["p_correct_min"]) + conviction * (
        float(price_sim_config["p_correct_max"])
        - float(price_sim_config["p_correct_min"])
    )
    overround = float(price_sim_config["overround_min"]) + np.power(
        conviction,
        float(price_sim_config["overround_gamma"]),
    ) * (
        float(price_sim_config["overround_max"])
        - float(price_sim_config["overround_min"])
    )
    direction_is_correct = rng.random(n_rows) < p_correct
    direction_sign = np.where(direction_is_correct, 1.0, -1.0)

    winner_ask = 0.5 + abs_gap / 2.0 + overround / 2.0
    loser_ask = 0.5 - abs_gap / 2.0 + overround / 2.0
    winner_is_up = (
        ((target == 1) & (direction_sign > 0.0))
        | ((target == 0) & (direction_sign < 0.0))
    )
    up_ask = np.where(winner_is_up, winner_ask, loser_ask)
    down_ask = np.where(winner_is_up, loser_ask, winner_ask)

    eps = float(price_sim_config["eps"])
    tick_size = float(price_sim_config["tick_size"])
    up_ask = np.clip(up_ask, eps, 1.0 - eps)
    down_ask = np.clip(down_ask, eps, 1.0 - eps)
    up_ask = np.round(up_ask / tick_size) * tick_size
    down_ask = np.round(down_ask / tick_size) * tick_size
    up_ask = np.clip(up_ask, eps, 1.0 - eps)
    down_ask = np.clip(down_ask, eps, 1.0 - eps)

    return {
        "up_ask": up_ask,
        "down_ask": down_ask,
        "abs_gap": np.abs(up_ask - down_ask),
        "overround": up_ask + down_ask - 1.0,
    }


def summarize_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": np.nan, "median": np.nan, "q10": np.nan, "q90": np.nan}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q90": float(np.quantile(values, 0.90)),
    }


def summarize_direction_correctness(up_ask, down_ask, target):
    up_ask = np.asarray(up_ask, dtype=np.float64)
    down_ask = np.asarray(down_ask, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    picked_up = np.where(up_ask > down_ask, 1.0, np.where(down_ask > up_ask, 0.0, np.nan))
    valid = np.isfinite(picked_up)
    if not np.any(valid):
        return {"rate": np.nan, "tie_rate": 1.0, "n_valid": 0}
    return {
        "rate": float(np.mean(picked_up[valid] == target[valid])),
        "tie_rate": float(np.mean(~valid)),
        "n_valid": int(np.sum(valid)),
    }


def tick_size_hit_rate(prices, tick_size):
    prices = np.asarray(prices, dtype=np.float64)
    if np.isscalar(tick_size):
        tick_sizes = np.full(prices.shape, float(tick_size), dtype=np.float64)
    else:
        tick_sizes = np.asarray(tick_size, dtype=np.float64)
    valid = np.isfinite(prices) & np.isfinite(tick_sizes) & (tick_sizes > 0.0)
    if not np.any(valid):
        return np.nan
    scaled = prices[valid] / tick_sizes[valid]
    return float(np.mean(np.abs(scaled - np.round(scaled)) <= 1e-9))


def build_side_metrics(label, up_ask, down_ask, target, tick_size):
    abs_gap = np.abs(up_ask - down_ask)
    overround = up_ask + down_ask - 1.0
    both_prices = np.concatenate([up_ask, down_ask])
    if np.isscalar(tick_size):
        tick_size_for_both = float(tick_size)
    else:
        tick_arr = np.asarray(tick_size, dtype=np.float64)
        tick_size_for_both = np.concatenate([tick_arr, tick_arr])
    return {
        "label": label,
        "up_best_ask": summarize_distribution(up_ask),
        "down_best_ask": summarize_distribution(down_ask),
        "abs_gap": summarize_distribution(abs_gap),
        "overround": summarize_distribution(overround),
        "direction_correctness": summarize_direction_correctness(up_ask, down_ask, target),
        "tick_size_hit_rate": tick_size_hit_rate(both_prices, tick_size_for_both),
    }


def main():
    market_price_sim_config = load_market_price_sim_config(OPTIMIZER_CONFIG_PATH)
    live = load_live_trade_frame(LIVE_CSV_PATHS)
    n_rows = len(live) if N_ROWS_SIMULATED is None else min(int(N_ROWS_SIMULATED), len(live))
    live = live.iloc[:n_rows].copy()

    target = live["actual_up"].to_numpy(dtype=np.int8, copy=False)
    live_up_ask = live["up_best_ask"].to_numpy(dtype=np.float64, copy=False)
    live_down_ask = live["down_best_ask"].to_numpy(dtype=np.float64, copy=False)
    live_tick_size = (
        live["tick_size"].to_numpy(dtype=np.float64, copy=False)
        if "tick_size" in live.columns
        else float(market_price_sim_config["tick_size"])
    )

    simulated = sample_market_orderbook_arrays(
        target=target,
        scenario_seed=SEED,
        price_sim_config=market_price_sim_config,
    )

    summary = {
        "inputs": {
            "optimizer_config_path": str(OPTIMIZER_CONFIG_PATH),
            "live_csv_paths": [str(path) for path in LIVE_CSV_PATHS],
            "n_live_rows_used": int(len(live)),
            "seed": int(SEED),
            "market_sim_model": str(market_price_sim_config["model"]),
        },
        "live": build_side_metrics(
            label="live",
            up_ask=live_up_ask,
            down_ask=live_down_ask,
            target=target,
            tick_size=live_tick_size,
        ),
        "simulated": build_side_metrics(
            label="simulated",
            up_ask=simulated["up_ask"],
            down_ask=simulated["down_ask"],
            target=target,
            tick_size=float(market_price_sim_config["tick_size"]),
        ),
    }

    print(
        "market sim vs live | "
        f"rows={summary['inputs']['n_live_rows_used']} "
        f"csvs={len(summary['inputs']['live_csv_paths'])} "
        f"seed={summary['inputs']['seed']} "
        f"model={summary['inputs']['market_sim_model']}"
    )
    print(
        "direction correctness | "
        f"live={summary['live']['direction_correctness']['rate']:.4f} "
        f"sim={summary['simulated']['direction_correctness']['rate']:.4f}"
    )
    print(
        "tick grid hit rate | "
        f"live={summary['live']['tick_size_hit_rate']:.4f} "
        f"sim={summary['simulated']['tick_size_hit_rate']:.4f}"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
