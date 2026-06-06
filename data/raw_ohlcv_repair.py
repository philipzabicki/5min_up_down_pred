from pathlib import Path

import numpy as np
import pandas as pd

OHLCV_COLS = ["Opened", "Open", "High", "Low", "Close", "Volume"]
PRICE_COLS = ["Open", "High", "Low", "Close"]
_INSERTED_GAP_VOLUME_SENTINEL = -1.0
DEFAULT_RAW_OHLCV_REPAIR_CONFIG = {
    "enabled": True,
    "mode": "monte_carlo_histogram",
    "histogram_bins": 100,
    "gap_min_block_len": 3,
    "volume_range_bins": 1000,
    "random_seed": 42,
    "bridge_weight_power": 2.0,
    "save_gap_charts": True,
    "gap_chart_context_before": 8,
    "gap_chart_context_after": 8,
}


class HistogramSampler:
    def __init__(self, values, bins, rng):
        clean = np.asarray(values, dtype=np.float64)
        clean = clean[np.isfinite(clean)]
        if clean.size == 0:
            raise ValueError(
                "HistogramSampler requires at least one finite observation."
            )

        self._rng = rng
        self._constant = None
        if np.allclose(clean, clean[0], rtol=0.0, atol=0.0):
            self._constant = float(clean[0])
            self._left = None
            self._right = None
            self._probs = None
            return

        counts, edges = np.histogram(clean, bins=max(1, int(bins)))
        positive_mask = counts > 0
        if not np.any(positive_mask):
            self._constant = float(np.median(clean))
            self._left = None
            self._right = None
            self._probs = None
            return

        probs = counts[positive_mask].astype(np.float64)
        probs /= probs.sum()
        self._left = edges[:-1][positive_mask].astype(np.float64, copy=False)
        self._right = edges[1:][positive_mask].astype(np.float64, copy=False)
        self._probs = probs

    def sample(self):
        if self._constant is not None:
            return float(self._constant)

        bin_idx = int(self._rng.choice(len(self._probs), p=self._probs))
        left = float(self._left[bin_idx])
        right = float(self._right[bin_idx])
        if not np.isfinite(left) or not np.isfinite(right) or right <= left:
            return left
        return float(self._rng.uniform(left, right))


class VolumeRangeMeanMap:
    def __init__(self, candle_ranges, volumes, bins):
        clean_ranges = np.asarray(candle_ranges, dtype=np.float64)
        clean_volumes = np.asarray(volumes, dtype=np.float64)
        valid_mask = (
                np.isfinite(clean_ranges)
                & (clean_ranges >= 0.0)
                & np.isfinite(clean_volumes)
                & (clean_volumes > 0.0)
        )
        clean_ranges = clean_ranges[valid_mask]
        clean_volumes = clean_volumes[valid_mask]
        if clean_ranges.size == 0:
            raise ValueError(
                "VolumeRangeMeanMap requires at least one valid range/volume observation."
            )

        self.observations = int(clean_ranges.size)
        self.global_mean = float(np.mean(clean_volumes))
        if np.allclose(clean_ranges, clean_ranges[0], rtol=0.0, atol=0.0):
            self.constant_mean = self.global_mean
            self.edges = None
            self.means = None
            self.populated_indices = None
            self.populated_bins = 1
            return

        _, edges = np.histogram(clean_ranges, bins=max(1, int(bins)))
        bin_count = len(edges) - 1
        bin_idx = np.searchsorted(edges, clean_ranges, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, bin_count - 1)

        sums = np.bincount(
            bin_idx,
            weights=clean_volumes,
            minlength=bin_count,
        ).astype(np.float64, copy=False)
        counts = np.bincount(bin_idx, minlength=bin_count)
        means = np.full(bin_count, np.nan, dtype=np.float64)
        populated_mask = counts > 0
        means[populated_mask] = sums[populated_mask] / counts[populated_mask]

        self.constant_mean = None
        self.edges = edges.astype(np.float64, copy=False)
        self.means = means
        self.populated_indices = np.flatnonzero(populated_mask)
        self.populated_bins = int(self.populated_indices.size)

    def estimate(self, candle_range):
        if self.constant_mean is not None:
            return float(self.constant_mean), False
        if self.edges is None or self.means is None or self.populated_indices is None:
            return float(self.global_mean), True

        bin_idx = int(
            np.searchsorted(self.edges, float(candle_range), side="right") - 1
        )
        bin_idx = max(0, min(bin_idx, len(self.means) - 1))
        if np.isfinite(self.means[bin_idx]):
            return float(self.means[bin_idx]), False

        if self.populated_indices.size == 0:
            return float(self.global_mean), True

        nearest_idx = int(
            self.populated_indices[np.argmin(np.abs(self.populated_indices - bin_idx))]
        )
        return float(self.means[nearest_idx]), True


def normalize_raw_ohlcv_repair_config(raw_config):
    if raw_config is None:
        return dict(DEFAULT_RAW_OHLCV_REPAIR_CONFIG)
    if not isinstance(raw_config, dict):
        raise ValueError("raw_ohlcv_repair config must be a JSON object.")

    mode = (
        str(raw_config.get("mode", DEFAULT_RAW_OHLCV_REPAIR_CONFIG["mode"]))
        .strip()
        .lower()
    )
    if mode not in {"monte_carlo_histogram"}:
        raise ValueError(
            "raw_ohlcv_repair.mode must be 'monte_carlo_histogram'. " f"Got: {mode!r}"
        )

    histogram_bins = int(
        raw_config.get(
            "histogram_bins",
            DEFAULT_RAW_OHLCV_REPAIR_CONFIG["histogram_bins"],
        )
    )
    if histogram_bins < 1:
        raise ValueError(
            f"raw_ohlcv_repair.histogram_bins must be >= 1, got: {histogram_bins}"
        )

    gap_min_block_len = int(
        raw_config.get(
            "gap_min_block_len",
            DEFAULT_RAW_OHLCV_REPAIR_CONFIG["gap_min_block_len"],
        )
    )
    if gap_min_block_len < 3:
        raise ValueError(
            "raw_ohlcv_repair.gap_min_block_len must be >= 3 because a gap is "
            f"defined as more than 2 identical candles, got: {gap_min_block_len}"
        )

    volume_range_bins = int(
        raw_config.get(
            "volume_range_bins",
            raw_config.get(
                "volume_window_radius",
                DEFAULT_RAW_OHLCV_REPAIR_CONFIG["volume_range_bins"],
            ),
        )
    )
    if volume_range_bins < 1:
        raise ValueError(
            "raw_ohlcv_repair.volume_range_bins must be >= 1, "
            f"got: {volume_range_bins}"
        )

    random_seed_raw = raw_config.get(
        "random_seed",
        DEFAULT_RAW_OHLCV_REPAIR_CONFIG["random_seed"],
    )
    random_seed = None if random_seed_raw in ("", None) else int(random_seed_raw)

    bridge_weight_power = float(
        raw_config.get(
            "bridge_weight_power",
            DEFAULT_RAW_OHLCV_REPAIR_CONFIG["bridge_weight_power"],
        )
    )
    if bridge_weight_power <= 0.0:
        raise ValueError(
            "raw_ohlcv_repair.bridge_weight_power must be > 0, "
            f"got: {bridge_weight_power}"
        )

    gap_chart_context_before = int(
        raw_config.get(
            "gap_chart_context_before",
            DEFAULT_RAW_OHLCV_REPAIR_CONFIG["gap_chart_context_before"],
        )
    )
    if gap_chart_context_before < 0:
        raise ValueError(
            "raw_ohlcv_repair.gap_chart_context_before must be >= 0, "
            f"got: {gap_chart_context_before}"
        )

    gap_chart_context_after = int(
        raw_config.get(
            "gap_chart_context_after",
            DEFAULT_RAW_OHLCV_REPAIR_CONFIG["gap_chart_context_after"],
        )
    )
    if gap_chart_context_after < 0:
        raise ValueError(
            "raw_ohlcv_repair.gap_chart_context_after must be >= 0, "
            f"got: {gap_chart_context_after}"
        )

    return {
        "enabled": bool(
            raw_config.get("enabled", DEFAULT_RAW_OHLCV_REPAIR_CONFIG["enabled"])
        ),
        "mode": mode,
        "histogram_bins": histogram_bins,
        "gap_min_block_len": gap_min_block_len,
        "volume_range_bins": volume_range_bins,
        "random_seed": random_seed,
        "bridge_weight_power": bridge_weight_power,
        "save_gap_charts": bool(
            raw_config.get(
                "save_gap_charts",
                DEFAULT_RAW_OHLCV_REPAIR_CONFIG["save_gap_charts"],
            )
        ),
        "gap_chart_context_before": gap_chart_context_before,
        "gap_chart_context_after": gap_chart_context_after,
    }


def _normalize_decimal_places(value, field_name):
    if value in ("", None):
        return None
    out = int(value)
    if out < 0:
        raise ValueError(f"{field_name} must be >= 0, got: {out}")
    return out


def _interval_to_pandas_rule(interval):
    if interval.endswith("m"):
        return f"{int(interval[:-1])}min"
    if interval.endswith("h"):
        return f"{int(interval[:-1])}h"
    if interval.endswith("d"):
        return f"{int(interval[:-1])}d"
    if interval.endswith("w"):
        return f"{int(interval[:-1])}w"
    raise ValueError(f"Unsupported pandas rule for raw OHLCV repair: {interval}")


def _prepare_ohlcv_frame(df):
    out = df.copy()
    if len(out.columns) != len(OHLCV_COLS):
        out = out.iloc[:, : len(OHLCV_COLS)].copy()
    out.columns = OHLCV_COLS
    if out.empty:
        return out, {
            "rows_before": 0,
            "duplicates_removed": 0,
            "dropped_invalid_price_rows": 0,
        }
    out = out[~out.isin(OHLCV_COLS).any(axis=1)].copy()

    rows_before = len(out)
    out["Opened"] = pd.to_datetime(out["Opened"], errors="raise")
    out = out.sort_values("Opened").drop_duplicates(subset=["Opened"], keep="last")
    rows_after_dedup = len(out)

    for col in OHLCV_COLS[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Opened", *PRICE_COLS]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid OHLC rows remain after cleanup.")

    return out, {
        "rows_before": rows_before,
        "duplicates_removed": int(max(rows_before - rows_after_dedup, 0)),
        "dropped_invalid_price_rows": int(max(rows_after_dedup - len(out), 0)),
    }


def _reindex_with_gap_placeholders(df, interval):
    rule = _interval_to_pandas_rule(interval)
    full_index = pd.DataFrame(
        {
            "Opened": pd.date_range(
                df["Opened"].iloc[0], df["Opened"].iloc[-1], freq=rule
            )
        }
    )
    out = full_index.merge(df, on="Opened", how="left")

    inserted_mask = (
            out["Open"].isna()
            | out["High"].isna()
            | out["Low"].isna()
            | out["Close"].isna()
    )
    if bool(inserted_mask.any()):
        out[PRICE_COLS] = out[PRICE_COLS].ffill()
        if out.loc[inserted_mask, PRICE_COLS].isna().any(axis=None):
            raise ValueError(
                "Cannot forward-fill inserted gap prices at the beginning of the series."
            )
        out.loc[inserted_mask, "Volume"] = _INSERTED_GAP_VOLUME_SENTINEL

    return out.reset_index(drop=True), inserted_mask.to_numpy(dtype=bool, copy=True)


def _detect_gap_blocks(df, min_block_len):
    same_prev_mask = (
        df.loc[:, OHLCV_COLS[1:]].eq(df.loc[:, OHLCV_COLS[1:]].shift(1)).all(axis=1)
    )
    same_prev_mask = same_prev_mask.fillna(False)
    run_id = (~same_prev_mask).cumsum()
    run_sizes = run_id.groupby(run_id).transform("size")
    gap_mask = (run_sizes >= int(min_block_len)).to_numpy(dtype=bool, copy=True)

    blocks = []
    idx = 0
    row_count = len(df)
    while idx < row_count:
        if not gap_mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < row_count and gap_mask[idx]:
            idx += 1
        end = idx - 1
        blocks.append((start, end))
    return gap_mask, blocks


def _build_distribution_samplers(df, gap_mask, histogram_bins, rng):
    base = df.loc[~gap_mask, PRICE_COLS].copy()
    if base.empty:
        base = df.loc[:, PRICE_COLS].copy()
    if base.empty:
        raise ValueError(
            "Cannot fit Monte Carlo repair distributions because no price rows are available."
        )

    open_arr = base["Open"].to_numpy(dtype=np.float64, copy=False)
    high_arr = base["High"].to_numpy(dtype=np.float64, copy=False)
    low_arr = base["Low"].to_numpy(dtype=np.float64, copy=False)
    close_arr = base["Close"].to_numpy(dtype=np.float64, copy=False)

    returns = close_arr - open_arr
    bullish_mask = close_arr >= open_arr
    bearish_mask = ~bullish_mask

    high_wick = np.maximum(high_arr - np.maximum(open_arr, close_arr), 0.0)
    low_wick = np.maximum(np.minimum(open_arr, close_arr) - low_arr, 0.0)

    if not bullish_mask.any():
        bullish_mask = np.ones_like(bullish_mask, dtype=bool)
    if not bearish_mask.any():
        bearish_mask = np.ones_like(bearish_mask, dtype=bool)

    return {
        "returns": HistogramSampler(returns, bins=histogram_bins, rng=rng),
        "high_bull": HistogramSampler(
            high_wick[bullish_mask],
            bins=histogram_bins,
            rng=rng,
        ),
        "high_bear": HistogramSampler(
            high_wick[bearish_mask],
            bins=histogram_bins,
            rng=rng,
        ),
        "low_bull": HistogramSampler(
            low_wick[bullish_mask],
            bins=histogram_bins,
            rng=rng,
        ),
        "low_bear": HistogramSampler(
            low_wick[bearish_mask],
            bins=histogram_bins,
            rng=rng,
        ),
        "obs": {
            "returns": int(returns.size),
            "high_bull": int(np.sum(bullish_mask)),
            "high_bear": int(np.sum(bearish_mask)),
            "low_bull": int(np.sum(bullish_mask)),
            "low_bear": int(np.sum(bearish_mask)),
        },
    }


def _bridge_returns(raw_returns, target_sum, bridge_weight_power):
    adjusted = np.asarray(raw_returns, dtype=np.float64).copy()
    if adjusted.size == 0 or target_sum is None:
        return adjusted, 0.0

    weights = np.power(
        np.arange(1, adjusted.size + 1, dtype=np.float64),
        float(bridge_weight_power),
    )
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        weights = np.full(adjusted.size, 1.0 / adjusted.size, dtype=np.float64)
    else:
        weights /= weight_sum

    correction_total = float(target_sum - adjusted.sum())
    adjusted += weights * correction_total
    adjusted[-1] += float(target_sum - adjusted.sum())
    return adjusted, correction_total


def _simulate_gap_prices(df, blocks, samplers, bridge_weight_power):
    out = df.copy()
    repaired_rows = 0
    gap_records = []

    for gap_idx, (start, end) in enumerate(blocks, start=1):
        start_opened = pd.Timestamp(out.at[start, "Opened"])
        end_opened = pd.Timestamp(out.at[end, "Opened"])
        block_len = int(end - start + 1)

        if start > 0:
            left_anchor_close = float(out.at[start - 1, "Close"])
        else:
            left_anchor_close = float(out.at[start, "Open"])

        has_right_anchor = end + 1 < len(out)
        right_anchor_open = float(out.at[end + 1, "Open"]) if has_right_anchor else None

        raw_returns = np.asarray(
            [samplers["returns"].sample() for _ in range(block_len)],
            dtype=np.float64,
        )
        raw_return_sum = float(raw_returns.sum())
        target_return_sum = (
            float(right_anchor_open - left_anchor_close) if has_right_anchor else None
        )
        adjusted_returns, bridge_correction_total = _bridge_returns(
            raw_returns,
            target_sum=target_return_sum,
            bridge_weight_power=bridge_weight_power,
        )
        bridge_correction_per_candle = (
            float(bridge_correction_total / block_len) if block_len else 0.0
        )

        print(
            "[raw_ohlcv_repair] "
            f"repairing_gap start_opened={start_opened} "
            f"end_opened={end_opened} "
            f"rows={block_len} "
            f"right_anchor_open={right_anchor_open} "
            f"bridge_correction_total={bridge_correction_total:.6f} "
            f"bridge_correction_per_candle={bridge_correction_per_candle:.6f}"
        )

        prev_close = left_anchor_close
        for offset, row_idx in enumerate(range(start, end + 1)):
            open_price = prev_close
            close_price = open_price + float(adjusted_returns[offset])
            is_bullish = close_price >= open_price
            high_wick = (
                samplers["high_bull"].sample()
                if is_bullish
                else samplers["high_bear"].sample()
            )
            low_wick = (
                samplers["low_bull"].sample()
                if is_bullish
                else samplers["low_bear"].sample()
            )

            body_high = max(open_price, close_price)
            body_low = min(open_price, close_price)
            high_price = body_high + max(float(high_wick), 0.0)
            low_price = body_low - max(float(low_wick), 0.0)

            out.at[row_idx, "Open"] = float(open_price)
            out.at[row_idx, "Close"] = float(close_price)
            out.at[row_idx, "High"] = float(max(high_price, body_high))
            out.at[row_idx, "Low"] = float(min(low_price, body_low))
            out.at[row_idx, "Volume"] = np.nan
            prev_close = close_price
            repaired_rows += 1

        gap_records.append(
            {
                "gap_index": int(gap_idx),
                "start_idx": int(start),
                "end_idx": int(end),
                "rows": int(block_len),
                "start_opened": start_opened,
                "end_opened": end_opened,
                "left_anchor_close": float(left_anchor_close),
                "right_anchor_open": (
                    None if right_anchor_open is None else float(right_anchor_open)
                ),
                "has_right_anchor": bool(has_right_anchor),
                "raw_return_sum": float(raw_return_sum),
                "target_return_sum": (
                    None if target_return_sum is None else float(target_return_sum)
                ),
                "bridge_correction_total": float(bridge_correction_total),
                "bridge_correction_per_candle": float(bridge_correction_per_candle),
            }
        )

    return out, repaired_rows, gap_records


def _fill_invalid_volume(df, invalid_mask, range_bins):
    out = df.copy()
    candle_ranges = out["High"].to_numpy(dtype=np.float64, copy=False) - out[
        "Low"
    ].to_numpy(dtype=np.float64, copy=False)
    volumes = out["Volume"].to_numpy(dtype=np.float64, copy=True)
    original_valid_mask = (
            np.isfinite(candle_ranges)
            & (candle_ranges >= 0.0)
            & np.isfinite(volumes)
            & (volumes > 0.0)
            & (~invalid_mask)
    )
    invalid_indices = np.flatnonzero(invalid_mask)

    if invalid_indices.size == 0:
        return out, {
            "invalid_volume_rows": 0,
            "invalid_volume_filled": 0,
            "invalid_volume_unresolved": 0,
            "volume_range_map_observations": int(np.sum(original_valid_mask)),
            "volume_range_map_populated_bins": 0,
            "volume_range_map_fallbacks": 0,
        }

    if not bool(np.any(original_valid_mask)):
        return out, {
            "invalid_volume_rows": int(invalid_indices.size),
            "invalid_volume_filled": 0,
            "invalid_volume_unresolved": int(invalid_indices.size),
            "volume_range_map_observations": 0,
            "volume_range_map_populated_bins": 0,
            "volume_range_map_fallbacks": 0,
        }

    volume_map = VolumeRangeMeanMap(
        candle_ranges=candle_ranges[original_valid_mask],
        volumes=volumes[original_valid_mask],
        bins=range_bins,
    )

    unresolved = 0
    filled = 0
    fallback_count = 0
    for idx in invalid_indices:
        candle_range = float(candle_ranges[idx])
        if not np.isfinite(candle_range) or candle_range < 0.0:
            unresolved += 1
            continue

        fill_value, used_fallback = volume_map.estimate(candle_range)
        volumes[idx] = float(fill_value)
        filled += 1
        if used_fallback:
            fallback_count += 1

    out["Volume"] = volumes
    return out, {
        "invalid_volume_rows": int(invalid_indices.size),
        "invalid_volume_filled": int(filled),
        "invalid_volume_unresolved": int(unresolved),
        "volume_range_map_observations": int(volume_map.observations),
        "volume_range_map_populated_bins": int(volume_map.populated_bins),
        "volume_range_map_fallbacks": int(fallback_count),
    }


def _round_ohlcv_values(df, price_decimals=None, volume_decimals=None):
    out = df.copy()
    changed = False

    if price_decimals is not None and not out.empty:
        rounded_prices = out.loc[:, PRICE_COLS].round(int(price_decimals))
        changed = changed or not rounded_prices.equals(out.loc[:, PRICE_COLS])
        out.loc[:, PRICE_COLS] = rounded_prices

    if volume_decimals is not None and not out.empty:
        rounded_volume = out["Volume"].round(int(volume_decimals))
        changed = changed or not rounded_volume.equals(out["Volume"])
        out["Volume"] = rounded_volume

    return out, {"rounding_applied": bool(changed)}


def _sanitize_path_token(text):
    return (
        str(text)
        .replace(":", "")
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
    )


def _draw_candles(ax, df, gap_start, gap_end, title):
    from matplotlib.patches import Rectangle

    row_count = len(df)
    if row_count == 0:
        ax.set_title(title)
        return

    ax.axvspan(
        float(gap_start) - 0.5,
        float(gap_end) + 0.5,
        color="#f4c542",
        alpha=0.18,
    )

    for candle_idx, row in enumerate(df.itertuples(index=False)):
        open_price = float(row.Open)
        high_price = float(row.High)
        low_price = float(row.Low)
        close_price = float(row.Close)
        color = "#188038" if close_price >= open_price else "#d93025"

        ax.vlines(
            candle_idx,
            low_price,
            high_price,
            color=color,
            linewidth=1.1,
            zorder=2,
        )

        body_low = min(open_price, close_price)
        body_height = abs(close_price - open_price)
        if body_height <= 0.0:
            ax.hlines(
                open_price,
                candle_idx - 0.32,
                candle_idx + 0.32,
                color=color,
                linewidth=2.2,
                zorder=3,
            )
            continue

        ax.add_patch(
            Rectangle(
                (candle_idx - 0.32, body_low),
                0.64,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                zorder=3,
            )
        )

    y_min = float(df["Low"].min())
    y_max = float(df["High"].max())
    y_span = max(y_max - y_min, 1e-9)
    y_pad = y_span * 0.08
    ax.set_xlim(-0.8, row_count - 0.2)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.set_ylabel("Price")
    ax.set_title(title)

    tick_count = min(10, row_count)
    tick_positions = sorted(
        set(
            np.linspace(0, row_count - 1, num=tick_count, dtype=int).tolist()
            + [int(gap_start), int(gap_end)]
        )
    )
    tick_labels = [
        pd.Timestamp(df.iloc[pos]["Opened"]).strftime("%m-%d %H:%M")
        for pos in tick_positions
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")


def _write_gap_artifacts(csv_path, original_df, repaired_df, gap_records, config):
    if not gap_records or not bool(config["save_gap_charts"]):
        return {
            "gap_artifacts_dir": None,
            "gap_summary_csv": None,
            "gap_chart_count": 0,
        }

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[raw_ohlcv_repair] gap charts skipped: {exc}")
        return {
            "gap_artifacts_dir": None,
            "gap_summary_csv": None,
            "gap_chart_count": 0,
        }

    timestamp_dir = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = (
            Path(csv_path).parent
            / "analysis"
            / "raw_ohlcv_repair"
            / Path(csv_path).stem
            / timestamp_dir
    )
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    context_before = int(config["gap_chart_context_before"])
    context_after = int(config["gap_chart_context_after"])

    for gap_record in gap_records:
        start_idx = int(gap_record["start_idx"])
        end_idx = int(gap_record["end_idx"])
        view_start = max(0, start_idx - context_before)
        view_end = min(len(repaired_df) - 1, end_idx + context_after)
        rel_gap_start = start_idx - view_start
        rel_gap_end = end_idx - view_start

        original_slice = original_df.iloc[view_start: view_end + 1].reset_index(
            drop=True
        )
        repaired_slice = repaired_df.iloc[view_start: view_end + 1].reset_index(
            drop=True
        )

        chart_name = (
            f"gap_{int(gap_record['gap_index']):03d}_"
            f"{_sanitize_path_token(gap_record['start_opened'])}_"
            f"{int(gap_record['rows'])}rows.png"
        )
        chart_path = charts_dir / chart_name

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(16, 9),
            sharex=True,
            constrained_layout=True,
        )
        _draw_candles(
            axes[0],
            original_slice,
            rel_gap_start,
            rel_gap_end,
            title="Before repair",
        )
        _draw_candles(
            axes[1],
            repaired_slice,
            rel_gap_start,
            rel_gap_end,
            title="After repair",
        )
        figure_title = (
            f"{Path(csv_path).name} | gap #{int(gap_record['gap_index'])} | "
            f"{pd.Timestamp(gap_record['start_opened'])} -> {pd.Timestamp(gap_record['end_opened'])}\n"
            f"right_anchor_open={gap_record['right_anchor_open']} | "
            f"bridge_correction_total={gap_record['bridge_correction_total']:.6f} | "
            f"bridge_correction_per_candle={gap_record['bridge_correction_per_candle']:.6f}"
        )
        fig.suptitle(figure_title)
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)

        summary_rows.append(
            {
                "gap_index": int(gap_record["gap_index"]),
                "start_opened": str(pd.Timestamp(gap_record["start_opened"])),
                "end_opened": str(pd.Timestamp(gap_record["end_opened"])),
                "rows": int(gap_record["rows"]),
                "has_right_anchor": bool(gap_record["has_right_anchor"]),
                "left_anchor_close": float(gap_record["left_anchor_close"]),
                "right_anchor_open": gap_record["right_anchor_open"],
                "raw_return_sum": float(gap_record["raw_return_sum"]),
                "target_return_sum": gap_record["target_return_sum"],
                "bridge_correction_total": float(gap_record["bridge_correction_total"]),
                "bridge_correction_per_candle": float(
                    gap_record["bridge_correction_per_candle"]
                ),
                "chart_path": str(chart_path),
            }
        )

    summary_csv = run_dir / "gap_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(
        "[raw_ohlcv_repair] "
        f"gap_artifacts_dir={run_dir} "
        f"gap_chart_count={len(summary_rows)} "
        f"gap_summary_csv={summary_csv}"
    )
    return {
        "gap_artifacts_dir": str(run_dir),
        "gap_summary_csv": str(summary_csv),
        "gap_chart_count": len(summary_rows),
    }


def repair_raw_ohlcv_frame(
        df,
        interval,
        raw_config=None,
        price_decimals=None,
        volume_decimals=None,
        artifact_csv_path=None,
):
    config = normalize_raw_ohlcv_repair_config(raw_config)
    price_decimals = _normalize_decimal_places(price_decimals, "price_decimals")
    volume_decimals = _normalize_decimal_places(volume_decimals, "volume_decimals")

    prepared, cleanup_summary = _prepare_ohlcv_frame(df)
    summary = {
        **cleanup_summary,
        "enabled": bool(config["enabled"]),
        "interval": str(interval),
        "histogram_bins": int(config["histogram_bins"]),
        "gap_min_block_len": int(config["gap_min_block_len"]),
        "volume_range_bins": int(config["volume_range_bins"]),
        "random_seed": config["random_seed"],
        "bridge_weight_power": float(config["bridge_weight_power"]),
        "rows_after_cleanup": len(prepared),
        "rows_after_repair": len(prepared),
        "missing_intervals_inserted": 0,
        "gap_blocks_repaired": 0,
        "gap_rows_repaired": 0,
        "invalid_volume_rows": 0,
        "invalid_volume_filled": 0,
        "invalid_volume_unresolved": 0,
        "volume_range_map_observations": 0,
        "volume_range_map_populated_bins": 0,
        "volume_range_map_fallbacks": 0,
        "distribution_observations": {},
        "price_decimals": price_decimals,
        "volume_decimals": volume_decimals,
        "rounding_applied": False,
        "gap_artifacts_dir": None,
        "gap_summary_csv": None,
        "gap_chart_count": 0,
        "gap_records": [],
        "changed": False,
    }

    repaired = prepared.loc[:, OHLCV_COLS].copy()
    original_expanded_for_artifacts = None
    gap_records = []
    distribution_summary = {}

    if config["enabled"] and not prepared.empty:
        expanded, inserted_mask = _reindex_with_gap_placeholders(prepared, interval)
        original_expanded_for_artifacts = expanded.loc[:, OHLCV_COLS].reset_index(
            drop=True
        )
        repaired = expanded.loc[:, OHLCV_COLS].copy()
        gap_mask, gap_blocks = _detect_gap_blocks(
            expanded,
            min_block_len=config["gap_min_block_len"],
        )
        summary["missing_intervals_inserted"] = int(inserted_mask.sum())
        summary["gap_blocks_repaired"] = len(gap_blocks)

        if gap_blocks:
            rng = np.random.default_rng(config["random_seed"])
            samplers = _build_distribution_samplers(
                expanded,
                gap_mask=gap_mask,
                histogram_bins=config["histogram_bins"],
                rng=rng,
            )
            repaired, repaired_rows, gap_records = _simulate_gap_prices(
                expanded,
                gap_blocks,
                samplers,
                bridge_weight_power=config["bridge_weight_power"],
            )
            distribution_summary = samplers["obs"]
            summary["gap_rows_repaired"] = int(repaired_rows)

        invalid_volume_mask = (
                repaired["Volume"].isna().to_numpy(dtype=bool, copy=True)
                | (repaired["Volume"].to_numpy(dtype=np.float64, copy=False) <= 0.0)
                | inserted_mask
                | gap_mask
        )
        repaired, volume_summary = _fill_invalid_volume(
            repaired,
            invalid_mask=invalid_volume_mask,
            range_bins=config["volume_range_bins"],
        )
        summary.update(volume_summary)

    repaired = repaired.loc[:, OHLCV_COLS].reset_index(drop=True)
    if not repaired.empty:
        repaired["Opened"] = pd.to_datetime(repaired["Opened"], errors="raise")
        repaired = repaired.sort_values("Opened").reset_index(drop=True)

    repaired, rounding_summary = _round_ohlcv_values(
        repaired,
        price_decimals=price_decimals,
        volume_decimals=volume_decimals,
    )

    summary.update(
        {
            "rows_after_repair": len(repaired),
            "distribution_observations": distribution_summary,
            "rounding_applied": bool(rounding_summary["rounding_applied"]),
            "gap_records": gap_records,
        }
    )
    summary["changed"] = bool(
        summary["duplicates_removed"] > 0
        or summary["dropped_invalid_price_rows"] > 0
        or summary["missing_intervals_inserted"] > 0
        or summary["gap_blocks_repaired"] > 0
        or summary["invalid_volume_rows"] > 0
        or summary["rounding_applied"]
    )
    if artifact_csv_path is not None and summary["gap_records"]:
        if original_expanded_for_artifacts is None:
            original_expanded_for_artifacts = prepared.loc[:, OHLCV_COLS].reset_index(
                drop=True
            )
        artifact_summary = _write_gap_artifacts(
            csv_path=Path(artifact_csv_path),
            original_df=original_expanded_for_artifacts,
            repaired_df=repaired,
            gap_records=summary["gap_records"],
            config=config,
        )
        summary.update(artifact_summary)
    return repaired, summary


def repair_raw_ohlcv_csv(
        csv_path,
        interval,
        raw_config=None,
        price_decimals=None,
        volume_decimals=None,
):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw OHLCV CSV not found: {path}")

    config = normalize_raw_ohlcv_repair_config(raw_config)
    df = pd.read_csv(path, header=0)
    repaired_df, summary = repair_raw_ohlcv_frame(
        df,
        interval=interval,
        raw_config=config,
        price_decimals=price_decimals,
        volume_decimals=volume_decimals,
    )

    if summary["gap_records"] and bool(config["save_gap_charts"]):
        prepared_df, _cleanup_summary = _prepare_ohlcv_frame(df)
        if not prepared_df.empty:
            original_expanded_df, _inserted_mask = _reindex_with_gap_placeholders(
                prepared_df,
                interval=interval,
            )
            artifact_summary = _write_gap_artifacts(
                csv_path=path,
                original_df=original_expanded_df.loc[:, OHLCV_COLS].reset_index(
                    drop=True
                ),
                repaired_df=repaired_df,
                gap_records=summary["gap_records"],
                config=config,
            )
            summary.update(artifact_summary)

    if summary["changed"]:
        repaired_df.to_csv(path, index=False)

    print(
        "[raw_ohlcv_repair] "
        f"path={path} "
        f"enabled={summary['enabled']} "
        f"missing_intervals_inserted={summary['missing_intervals_inserted']} "
        f"gap_blocks_repaired={summary['gap_blocks_repaired']} "
        f"gap_rows_repaired={summary['gap_rows_repaired']} "
        f"invalid_volume_rows={summary['invalid_volume_rows']} "
        f"invalid_volume_filled={summary['invalid_volume_filled']} "
        f"invalid_volume_unresolved={summary['invalid_volume_unresolved']} "
        f"volume_range_map_observations={summary['volume_range_map_observations']} "
        f"volume_range_map_populated_bins={summary['volume_range_map_populated_bins']} "
        f"volume_range_map_fallbacks={summary['volume_range_map_fallbacks']} "
        f"rounding_applied={summary['rounding_applied']} "
        f"gap_chart_count={summary['gap_chart_count']} "
        f"rows={summary['rows_after_repair']}"
    )
    return repaired_df, summary


__all__ = [
    "DEFAULT_RAW_OHLCV_REPAIR_CONFIG",
    "normalize_raw_ohlcv_repair_config",
    "repair_raw_ohlcv_csv",
    "repair_raw_ohlcv_frame",
]
