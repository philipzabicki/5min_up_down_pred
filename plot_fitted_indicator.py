import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from talib import TRANGE

from features.ADX import custom_adx
from features.ChaikinOsc import custom_chaikin_oscillator
from features.MACD import custom_macd
from features.StochOsc import custom_stochastic_oscillator
from features.ta_tools import apply_ma, get_1d_ma, precompute_ohlcv_sources
from utils.data import drop_frozen_ohlc_blocks
from utils.project_config import load_modeling_settings


INDICATOR_CONFIG_PATH = Path(
    "data/features/indicators_fit/ETH/tuning/c206782593b2444c/ChaikinOsc_target_5m_candle_up_pop1024_qe0.2_qm0.1_tf0.8_stmc_sg10_rwlin1-1.5.json"
)
OUTPUT_DIR = Path("data/analysis/fitted_indicator_plots")
PLOT_TAIL_ROWS = 1500

PROJECT_ROOT = Path(__file__).resolve().parent
TIME_COL = "Opened"
INDICATOR_RE = re.compile(
    r"^(ADX|BollingerBands|ChaikinOsc|KeltnerChannel|MACD|StochOsc)_"
)
SKIP_JSON_NAMES = {
    "fit_indicators_config.json",
    "fit_indicators_applied_config.json",
}


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_params(payload, json_path):
    params = payload.get("params")
    if isinstance(params, dict):
        return params

    best = payload.get("best")
    if isinstance(best, dict) and isinstance(best.get("params"), dict):
        return best["params"]

    raise ValueError(f"Missing params or best.params in {json_path}")


def indicator_from_path(json_path):
    match = INDICATOR_RE.match(json_path.name)
    if not match:
        return None
    return match.group(1)


def find_config_files(config_path):
    config_path = project_path(config_path)

    if config_path.is_file():
        configs = [config_path]
    elif config_path.is_dir():
        configs = [
            path
            for path in sorted(config_path.glob("*.json"))
            if path.name not in SKIP_JSON_NAMES and indicator_from_path(path)
        ]
    else:
        raise FileNotFoundError(f"Config path does not exist: {config_path}")

    if not configs:
        raise FileNotFoundError(f"No indicator config files found in {config_path}")

    return configs


def first_applied_interval(applied_config_path):
    payload = json.loads(applied_config_path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs")
    if not isinstance(pairs, dict):
        raise ValueError(f"Malformed applied config: {applied_config_path}")

    for pair_cfg in pairs.values():
        if not isinstance(pair_cfg, dict):
            continue
        intervals = pair_cfg.get("intervals")
        if not isinstance(intervals, dict):
            continue
        for interval_cfg in intervals.values():
            if isinstance(interval_cfg, dict):
                return pair_cfg, interval_cfg

    raise ValueError(f"No pair interval found in {applied_config_path}")


def resolve_raw_dataset(config_root):
    applied_config_path = config_root / "fit_indicators_applied_config.json"
    if applied_config_path.exists():
        pair_cfg, interval_cfg = first_applied_interval(applied_config_path)
        data_path = project_path(interval_cfg["data_path"])
        data_file = interval_cfg.get("data_file", "dataset.csv")
        drop_frozen_cfg = interval_cfg.get(
            "drop_frozen_ohlc_blocks",
            pair_cfg.get("drop_frozen_ohlc_blocks"),
        )
        return data_path / data_file, drop_frozen_cfg

    settings = load_modeling_settings()
    return (
        project_path(settings["raw_data_dir"]) / str(settings["base_data_file"]),
        settings.get("drop_frozen_ohlc_blocks"),
    )


def infer_ohlcv_columns(columns):
    cols = [str(col) for col in columns]
    lower = [col.lower() for col in cols]
    required = ["open", "high", "low", "close", "volume"]
    if all(name in lower for name in required):
        return [cols[lower.index(name)] for name in required]
    if len(cols) >= 6:
        return cols[1:6]
    raise ValueError("Cannot infer OHLCV columns from raw dataset header.")


def resolve_time_column(columns, ohlcv_cols):
    cols = [str(col) for col in columns]
    lower = [col.lower() for col in cols]
    if TIME_COL.lower() in lower:
        return cols[lower.index(TIME_COL.lower())]
    for col in cols:
        if col not in ohlcv_cols:
            return col
    raise ValueError("Cannot infer time column from raw dataset header.")


def load_raw_ohlcv(raw_path, drop_frozen_cfg):
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw dataset: {raw_path}")

    header = pd.read_csv(raw_path, nrows=0)
    ohlcv_cols = infer_ohlcv_columns(header.columns)
    time_col = resolve_time_column(header.columns, ohlcv_cols)
    usecols = list(dict.fromkeys([time_col, *ohlcv_cols]))

    df = pd.read_csv(raw_path, usecols=usecols)
    if time_col != TIME_COL:
        df = df.rename(columns={time_col: TIME_COL})
    for col in ohlcv_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="raise")

    df, drop_summary = drop_frozen_ohlc_blocks(
        df,
        raw_config=drop_frozen_cfg,
        opened_col=TIME_COL,
        ohlc_cols=tuple(ohlcv_cols[:4]),
    )
    return df, ohlcv_cols, drop_summary


def slice_tail(df, series_map, tail_rows):
    if tail_rows is None or int(tail_rows) <= 0 or int(tail_rows) >= len(df):
        start = 0
    else:
        start = len(df) - int(tail_rows)
    out_df = df.iloc[start:].reset_index(drop=True)
    out_series = {
        name: np.asarray(values)[start:]
        for name, values in series_map.items()
    }
    return out_df, out_series


def bollinger_lines(params, ohlcv_np):
    sources = precompute_ohlcv_sources(ohlcv_np)
    volume = ohlcv_np[:, 4]
    center = apply_ma(
        sources[params["ma_source"]],
        params["ma_type"],
        params["ma_period"],
        volume,
    )
    src = sources[params["std_source"]]
    diff2 = (src - center) ** 2
    var_smooth = get_1d_ma(diff2, params["std_ma_type"], params["std_ma_period"])
    std = np.sqrt(np.where(var_smooth >= 0.0, var_smooth, np.nan))
    std_multi = float(params.get("std_multi", 1.0))
    return {
        "BB lower": center - std_multi * std,
        "BB middle": center,
        "BB upper": center + std_multi * std,
    }


def keltner_lines(params, ohlcv_np):
    sources = precompute_ohlcv_sources(ohlcv_np)
    volume = ohlcv_np[:, 4]
    center = apply_ma(
        sources[params["source"]],
        params["ma_type"],
        params["ma_period"],
        volume,
    )
    atr = get_1d_ma(
        TRANGE(*ohlcv_np[:, 1:4].T),
        params["atr_ma_type"],
        params["atr_period"],
    )
    atr_multi = float(params.get("atr_multi", 1.0))
    return {
        "KC lower": center - atr_multi * atr,
        "KC middle": center,
        "KC upper": center + atr_multi * atr,
    }


def compute_indicator_series(indicator, params, ohlcv_np):
    if indicator == "BollingerBands":
        return bollinger_lines(params, ohlcv_np)
    if indicator == "KeltnerChannel":
        return keltner_lines(params, ohlcv_np)
    if indicator == "ADX":
        adx, plus_di, minus_di = custom_adx(
            ohlcv_np,
            atr_period=params["atr_period"],
            posDM_period=params["posDM_period"],
            negDM_period=params["negDM_period"],
            adx_period=params["adx_period"],
            ma_type_atr=params["ma_type_atr"],
            ma_type_posDM=params["ma_type_posDM"],
            ma_type_negDM=params["ma_type_negDM"],
            ma_type_adx=params["ma_type_adx"],
        )
        return {"ADX": adx, "+DI": plus_di, "-DI": minus_di}
    if indicator == "ChaikinOsc":
        osc = custom_chaikin_oscillator(
            ohlcv_np,
            fast_period=params["fast_period"],
            slow_period=params["slow_period"],
            fast_ma_type=params["fast_ma_type"],
            slow_ma_type=params["slow_ma_type"],
        )
        return {"Chaikin Osc": osc}
    if indicator == "MACD":
        macd, signal = custom_macd(
            ohlcv_np,
            fast_source=params["fast_source"],
            slow_source=params["slow_source"],
            fast_period=params["fast_period"],
            slow_period=params["slow_period"],
            signal_period=params["signal_period"],
            fast_ma_type=params["fast_ma_type"],
            slow_ma_type=params["slow_ma_type"],
            signal_ma_type=params["signal_ma_type"],
        )
        return {"MACD": macd, "Signal": signal, "Histogram": macd - signal}
    if indicator == "StochOsc":
        slow_k, slow_d = custom_stochastic_oscillator(
            ohlcv_np,
            fastK_period=params["fastK_period"],
            slowK_period=params["slowK_period"],
            slowD_period=params["slowD_period"],
            slowK_ma_type=params["slowK_ma_type"],
            slowD_ma_type=params["slowD_ma_type"],
        )
        return {"SlowK": slow_k, "SlowD": slow_d, "K-D": slow_k - slow_d}
    raise ValueError(f"Unsupported indicator: {indicator}")


def draw_ohlc(ax, df, ohlcv_cols):
    open_col, high_col, low_col, close_col, _ = ohlcv_cols
    x = np.arange(len(df))
    open_ = df[open_col].to_numpy(dtype=float)
    high = df[high_col].to_numpy(dtype=float)
    low = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)
    up = close >= open_
    colors = np.where(up, "#16803c", "#c43d3d")
    body_bottom = np.minimum(open_, close)
    body_height = np.abs(close - open_)
    min_body = np.nanmedian(np.maximum(high - low, 0.0)) * 0.02
    if not np.isfinite(min_body) or min_body <= 0:
        min_body = 1e-9

    ax.vlines(x, low, high, color=colors, linewidth=0.7, alpha=0.75)
    ax.bar(
        x,
        np.maximum(body_height, min_body),
        bottom=body_bottom,
        width=0.65,
        color=colors,
        edgecolor=colors,
        linewidth=0.4,
        alpha=0.9,
    )
    ax.set_ylabel("OHLC")
    ax.grid(True, color="#e7e7e7", linewidth=0.6)


def add_time_ticks(ax, opened):
    count = len(opened)
    if count == 0:
        return
    tick_count = min(8, count)
    ticks = np.unique(np.linspace(0, count - 1, tick_count, dtype=int))
    labels = pd.Series(opened).iloc[ticks].dt.strftime("%Y-%m-%d\n%H:%M")
    ax.set_xlim(-1, count)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def plot_lines(ax, series_map, names, ylabel=None):
    x = np.arange(len(next(iter(series_map.values()))))
    for name in names:
        ax.plot(x, series_map[name], label=name, linewidth=1.0)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, color="#e7e7e7", linewidth=0.6)
    ax.legend(loc="upper left")


def plot_histogram(ax, values):
    x = np.arange(len(values))
    colors = np.where(values >= 0, "#16803c", "#c43d3d")
    ax.bar(x, values, color=colors, width=0.8, alpha=0.75)
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_ylabel("Hist")
    ax.grid(True, color="#e7e7e7", linewidth=0.6)


def plot_indicator(df, ohlcv_cols, indicator, series_map, json_path, output_path):
    if indicator == "MACD":
        fig, axes = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(16, 10),
            gridspec_kw={"height_ratios": [3.0, 1.2, 0.9]},
        )
    elif indicator in {"ADX", "ChaikinOsc", "StochOsc"}:
        fig, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(16, 9),
            gridspec_kw={"height_ratios": [3.0, 1.25]},
        )
    else:
        fig, axes = plt.subplots(1, 1, figsize=(16, 8))
        axes = [axes]

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    axes = list(np.ravel(axes))

    draw_ohlc(axes[0], df, ohlcv_cols)
    axes[0].set_title(f"{indicator} | {json_path.name}")

    if indicator == "BollingerBands":
        plot_lines(axes[0], series_map, ["BB lower", "BB middle", "BB upper"])
    elif indicator == "KeltnerChannel":
        plot_lines(axes[0], series_map, ["KC lower", "KC middle", "KC upper"])
    elif indicator == "ADX":
        plot_lines(axes[1], series_map, ["ADX", "+DI", "-DI"], ylabel="ADX")
    elif indicator == "ChaikinOsc":
        plot_lines(axes[1], series_map, ["Chaikin Osc"], ylabel="Chaikin")
        axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    elif indicator == "MACD":
        plot_lines(axes[1], series_map, ["MACD", "Signal"], ylabel="MACD")
        axes[1].axhline(0.0, color="#555555", linewidth=0.8)
        plot_histogram(axes[2], series_map["Histogram"])
    elif indicator == "StochOsc":
        plot_lines(axes[1], series_map, ["SlowK", "SlowD"], ylabel="Stoch")
        axes[1].axhline(80.0, color="#777777", linewidth=0.8, linestyle="--")
        axes[1].axhline(20.0, color="#777777", linewidth=0.8, linestyle="--")
    else:
        raise ValueError(f"Unsupported indicator: {indicator}")

    add_time_ticks(axes[-1], df[TIME_COL])
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main():
    config_files = find_config_files(INDICATOR_CONFIG_PATH)
    config_root = project_path(INDICATOR_CONFIG_PATH)
    config_root = config_root.parent if config_root.is_file() else config_root
    raw_path, drop_frozen_cfg = resolve_raw_dataset(config_root)

    print(f"raw dataset: {raw_path}")
    df, ohlcv_cols, drop_summary = load_raw_ohlcv(raw_path, drop_frozen_cfg)
    if drop_summary["enabled"]:
        print(
            "drop frozen OHLC blocks: "
            f"removed_rows={drop_summary['rows_removed']} "
            f"rows_after={drop_summary['rows_after']}"
        )
    ohlcv_np = df[ohlcv_cols].to_numpy(dtype=np.float64, copy=False)

    output_root = project_path(OUTPUT_DIR) / config_root.name
    saved_paths = []
    for json_path in config_files:
        indicator = indicator_from_path(json_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        params = extract_params(payload, json_path)
        print(f"plotting {indicator}: {json_path}")
        series_map = compute_indicator_series(indicator, params, ohlcv_np)
        plot_df, plot_series = slice_tail(df, series_map, PLOT_TAIL_ROWS)
        output_path = output_root / f"{json_path.stem}.png"
        plot_indicator(
            plot_df,
            ohlcv_cols,
            indicator,
            plot_series,
            json_path,
            output_path,
        )
        saved_paths.append(output_path)
        print(f"saved: {output_path}")

    print(f"done: {len(saved_paths)} chart(s)")


if __name__ == "__main__":
    main()
