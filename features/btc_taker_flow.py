"""Causal taker-flow and price-reaction features from Binance USD-M klines."""
from __future__ import annotations

import numpy as np
import pandas as pd


def feature_names(horizons: tuple[int, ...] = (1, 5, 15)) -> tuple[list[str], list[str]]:
    control = [name for h in horizons for name in (f"taker_Z_{h}m", f"taker_V_{h}m")]
    extension = [
        name
        for h in horizons
        for name in (
            f"taker_I_{h}m",
            f"taker_U_{h}m",
            f"taker_beta_{h}m",
            f"taker_A_{h}m",
        )
    ]
    return control, extension


def compute_taker_flow_features(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 5, 15),
    qbar_window: int = 1440,
    sigma_window: int = 60,
    beta_window: int = 10080,
    minimum_beta_pairs: int = 1440,
    beta_variance_tolerance: float = 1e-12,
    sigma_floor: float = 1e-6,
    sigma_ddof: int = 0,
) -> pd.DataFrame:
    """Compute features at each candle open using closed candles through that row."""
    required = {
        "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing Binance kline fields: {sorted(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Kline index must be timezone-aware UTC timestamps")
    if str(frame.index.tz) != "UTC":
        raise ValueError("Kline index must use UTC")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Kline timestamps must be unique and sorted")
    if any(h < 1 for h in horizons):
        raise ValueError("Horizons must be positive")
    if min(qbar_window, sigma_window, beta_window, minimum_beta_pairs) < 1:
        raise ValueError("Rolling windows and minimum observations must be positive")
    if beta_window < minimum_beta_pairs:
        raise ValueError("beta_window must be at least minimum_beta_pairs")
    if beta_variance_tolerance < 0 or sigma_floor <= 0:
        raise ValueError("Invalid numerical tolerance")

    start = frame.index.min().floor("min")
    end = frame.index.max().floor("min")
    grid = pd.date_range(start, end, freq="min", tz="UTC")
    data = frame.loc[:, sorted(required)].reindex(grid)
    open_ = pd.to_numeric(data["open"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    close = pd.to_numeric(data["close"], errors="coerce")
    quote = pd.to_numeric(data["quote_volume"], errors="coerce")
    taker_buy = pd.to_numeric(data["taker_buy_quote_volume"], errors="coerce")

    price_valid = (
        np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
        & open_.gt(0) & high.gt(0) & low.gt(0) & close.gt(0)
        & high.ge(np.maximum(open_, close))
        & low.le(np.minimum(open_, close)) & high.ge(low)
    )
    volume_valid = (
        np.isfinite(quote) & np.isfinite(taker_buy) & quote.ge(0)
        & taker_buy.ge(0) & taker_buy.le(quote)
    )
    quote = quote.where(volume_valid)
    taker_buy = taker_buy.where(volume_valid)
    signed = (2.0 * taker_buy - quote).where(volume_valid)
    close = close.where(price_valid)
    open_ = open_.where(price_valid)

    minute_return = np.log(close).diff()
    output: dict[str, pd.Series] = {}
    for h in horizons:
        q = quote.rolling(h, min_periods=h).sum()
        d = signed.rolling(h, min_periods=h).sum()
        qbar = quote.rolling(qbar_window, min_periods=qbar_window).mean().shift(h)
        sigma = (
            minute_return.rolling(sigma_window, min_periods=sigma_window)
            .std(ddof=sigma_ddof)
            .shift(h)
        )
        window_price_valid = price_valid.astype(np.int8).rolling(h, min_periods=h).sum().eq(h)
        window_open = open_.shift(h - 1)
        window_close = close
        price_move = np.log(window_close / window_open).where(window_price_valid)
        z = price_move / (np.sqrt(float(h)) * sigma.clip(lower=sigma_floor))

        normalizer = float(h) * qbar
        u = d / normalizer.where(normalizer.gt(0))
        imbalance = d / q.where(q.gt(0))
        volume_activity = np.log1p(q / normalizer.where(normalizer.gt(0)))

        pair_valid = np.isfinite(u) & np.isfinite(z)
        x = u.where(pair_valid)
        y = z.where(pair_valid)
        pair_count = pair_valid.astype(np.int32).rolling(
            beta_window, min_periods=minimum_beta_pairs,
        ).sum().shift(h)
        sum_x = x.rolling(beta_window, min_periods=minimum_beta_pairs).sum().shift(h)
        sum_y = y.rolling(beta_window, min_periods=minimum_beta_pairs).sum().shift(h)
        sum_xx = x.pow(2).rolling(beta_window, min_periods=minimum_beta_pairs).sum().shift(h)
        sum_xy = (x * y).rolling(beta_window, min_periods=minimum_beta_pairs).sum().shift(h)
        n = pair_count.where(pair_count.ge(minimum_beta_pairs))
        mean_x = sum_x / n
        mean_y = sum_y / n
        variance_x = (sum_xx / n - mean_x.pow(2)).clip(lower=0.0)
        covariance_xy = sum_xy / n - mean_x * mean_y
        near_zero = variance_x.le(beta_variance_tolerance)
        beta = (covariance_xy / variance_x.where(~near_zero)).where(n.notna())
        beta = beta.where(~near_zero, 0.0)
        intercept = (mean_y - beta * mean_x).where(n.notna())
        intercept = intercept.where(~near_zero, mean_y)
        residual = z - (intercept + beta * u)

        output[f"taker_Z_{h}m"] = z
        output[f"taker_V_{h}m"] = volume_activity
        output[f"taker_I_{h}m"] = imbalance
        output[f"taker_U_{h}m"] = u
        output[f"taker_beta_{h}m"] = beta
        output[f"taker_A_{h}m"] = residual

    return pd.DataFrame(output, index=grid)


def preopen_market_alignment(opened: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """Map the final closed candle start to the pre-open decision and market."""
    candle_open = pd.Timestamp(opened)
    if candle_open.tzinfo is None:
        raise ValueError("Opened must be timezone-aware")
    candle_open = candle_open.tz_convert("UTC")
    decision = candle_open + pd.Timedelta(minutes=1)
    market_start = candle_open + pd.Timedelta(minutes=2)
    return {
        "last_closed_candle_opened_at": candle_open,
        "information_boundary_at": decision,
        "market_start_at": market_start,
        "market_end_at": market_start + pd.Timedelta(minutes=5),
        "target_label_available_at": market_start + pd.Timedelta(minutes=5),
    }
