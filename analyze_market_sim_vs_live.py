import json
from pathlib import Path

import numpy as np
import pandas as pd

from common_config_utils import load_json_object
from market_price_sim import (
    DEFAULT_SHARED_CSV_PATH,
    DEFAULT_TRADE_CSV_GLOB,
    load_live_market_empirical_frame,
    sample_market_orderbook_arrays as shared_sample_market_orderbook_arrays,
)

OPTIMIZER_CONFIG_PATH = Path("configs/trade_policy_optimizer_config.json")
LIVE_TRADE_CSV_GLOB = DEFAULT_TRADE_CSV_GLOB
LIVE_SHARED_CSV_PATH = DEFAULT_SHARED_CSV_PATH
N_ROWS_SIMULATED = None
SEED = 37


def load_market_price_sim_config(config_path):
    payload = load_json_object(config_path)
    market_price_sim = payload.get("market_price_sim")
    if not isinstance(market_price_sim, dict):
        raise ValueError(f"Missing market_price_sim in {config_path}")
    return market_price_sim


def load_live_trade_frame(csv_paths):
    if not csv_paths:
        raise ValueError("csv_paths is empty.")

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        required_columns = {"pm_up_best_ask", "pm_down_best_ask", "actual_up"}
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")

        keep_columns = ["pm_up_best_ask", "pm_down_best_ask", "actual_up"]
        if "pm_tick_size" in frame.columns:
            keep_columns.append("pm_tick_size")

        cleaned = frame.loc[:, keep_columns].copy()
        cleaned["source_csv"] = str(csv_path)
        frames.append(cleaned)

    live = pd.concat(frames, ignore_index=True)
    live = live.dropna(
        subset=["pm_up_best_ask", "pm_down_best_ask", "actual_up"]
    ).copy()
    live["actual_up"] = pd.to_numeric(live["actual_up"], errors="coerce")
    live = live[live["actual_up"].isin([0.0, 1.0])].copy()
    if live.empty:
        raise ValueError("No resolved live rows with asks and actual_up.")

    live["actual_up"] = live["actual_up"].astype(np.int8)
    live["pm_up_best_ask"] = pd.to_numeric(live["pm_up_best_ask"], errors="coerce")
    live["pm_down_best_ask"] = pd.to_numeric(
        live["pm_down_best_ask"], errors="coerce"
    )
    if "pm_tick_size" in live.columns:
        live["pm_tick_size"] = pd.to_numeric(live["pm_tick_size"], errors="coerce")

    live = live.dropna(subset=["pm_up_best_ask", "pm_down_best_ask"]).reset_index(
        drop=True
    )
    if live.empty:
        raise ValueError("No usable live ask rows after numeric coercion.")

    return live


def load_default_live_trade_frame(recent_resolved_rows=None):
    return load_live_market_empirical_frame(
        trade_csv_glob=LIVE_TRADE_CSV_GLOB,
        shared_csv_path=LIVE_SHARED_CSV_PATH,
        recent_resolved_rows=recent_resolved_rows,
    )


def sample_market_orderbook_arrays(
    target,
    scenario_seed,
    price_sim_config,
    market_elapsed_ms,
):
    simulated = shared_sample_market_orderbook_arrays(
        target=target,
        scenario_seed=scenario_seed,
        price_sim_config=price_sim_config,
        market_elapsed_ms=market_elapsed_ms,
    )
    up_ask = np.asarray(simulated["up_ask"], dtype=np.float64)
    down_ask = np.asarray(simulated["down_ask"], dtype=np.float64)
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


def build_reconstruction_metrics(live_up_ask, live_down_ask, sim_up_ask, sim_down_ask):
    live_up_ask = np.asarray(live_up_ask, dtype=np.float64)
    live_down_ask = np.asarray(live_down_ask, dtype=np.float64)
    sim_up_ask = np.asarray(sim_up_ask, dtype=np.float64)
    sim_down_ask = np.asarray(sim_down_ask, dtype=np.float64)
    return {
        "up_ask_mae": float(np.mean(np.abs(sim_up_ask - live_up_ask))),
        "down_ask_mae": float(np.mean(np.abs(sim_down_ask - live_down_ask))),
        "winner_ask_mae": float(
            np.mean(
                np.abs(
                    np.maximum(sim_up_ask, sim_down_ask)
                    - np.maximum(live_up_ask, live_down_ask)
                )
            )
        ),
        "loser_ask_mae": float(
            np.mean(
                np.abs(
                    np.minimum(sim_up_ask, sim_down_ask)
                    - np.minimum(live_up_ask, live_down_ask)
                )
            )
        ),
        "abs_gap_mae": float(
            np.mean(
                np.abs(
                    np.abs(sim_up_ask - sim_down_ask)
                    - np.abs(live_up_ask - live_down_ask)
                )
            )
        ),
        "overround_mae": float(
            np.mean(
                np.abs(
                    (sim_up_ask + sim_down_ask)
                    - (live_up_ask + live_down_ask)
                )
            )
        ),
    }


def main():
    market_price_sim_config = load_market_price_sim_config(OPTIMIZER_CONFIG_PATH)
    live = load_default_live_trade_frame(
        recent_resolved_rows=market_price_sim_config.get("recent_resolved_rows")
    )
    n_rows = len(live) if N_ROWS_SIMULATED is None else min(int(N_ROWS_SIMULATED), len(live))
    live = live.iloc[:n_rows].copy()

    target = live["actual_up"].to_numpy(dtype=np.int8, copy=False)
    market_elapsed_ms = live["market_elapsed_ms"].to_numpy(dtype=np.float64, copy=False)
    live_up_ask = live["pm_up_best_ask"].to_numpy(dtype=np.float64, copy=False)
    live_down_ask = live["pm_down_best_ask"].to_numpy(dtype=np.float64, copy=False)
    live_tick_size = (
        live["pm_tick_size"].to_numpy(dtype=np.float64, copy=False)
        if "pm_tick_size" in live.columns and live["pm_tick_size"].notna().any()
        else float(market_price_sim_config["tick_size"])
    )

    simulated = sample_market_orderbook_arrays(
        target=target,
        scenario_seed=SEED,
        price_sim_config=market_price_sim_config,
        market_elapsed_ms=market_elapsed_ms,
    )

    summary = {
        "inputs": {
            "optimizer_config_path": str(OPTIMIZER_CONFIG_PATH),
            "trade_csv_glob": LIVE_TRADE_CSV_GLOB,
            "shared_csv_path": LIVE_SHARED_CSV_PATH,
            "n_live_rows_used": int(len(live)),
            "n_unique_source_files": int(live["source_path"].nunique()),
            "recent_resolved_rows": market_price_sim_config.get("recent_resolved_rows"),
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
        "reconstruction": build_reconstruction_metrics(
            live_up_ask,
            live_down_ask,
            simulated["up_ask"],
            simulated["down_ask"],
        ),
    }

    print(
        "market sim vs live | "
        f"rows={summary['inputs']['n_live_rows_used']} "
        f"source_files={summary['inputs']['n_unique_source_files']} "
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
    print(
        "reconstruction mae | "
        f"up={summary['reconstruction']['up_ask_mae']:.4f} "
        f"down={summary['reconstruction']['down_ask_mae']:.4f} "
        f"gap={summary['reconstruction']['abs_gap_mae']:.4f} "
        f"overround={summary['reconstruction']['overround_mae']:.4f}"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
