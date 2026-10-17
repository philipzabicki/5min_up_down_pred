"""Causal realized-volatility normalization for the BTC pre-open return target."""

from __future__ import annotations

import numpy as np
import pandas as pd


VOLATILITY_RETURN_COUNT = 60
VOLATILITY_TARGET_MINUTES = 5
VOLATILITY_DENOMINATOR_FLOOR = 1e-6


def build_preopen_sigma_5m(opened_values, close_values) -> np.ndarray:
    """Scale the population std of the last 60 closed 1m returns to five minutes.

    A return is present only when both adjacent closes are finite and positive and
    their candles are exactly one minute apart. The returned sigma is aligned to
    ``opened_values``; the last value uses the close of the candle opened at that
    timestamp, which is available at its close one minute later.
    """
    opened = pd.DatetimeIndex(pd.to_datetime(opened_values, utc=True, errors="raise"))
    close = np.asarray(close_values, dtype=np.float64)
    if close.ndim != 1 or len(close) != len(opened):
        raise ValueError("close_values must be a one-dimensional array aligned to opened_values")
    if opened.has_duplicates or not opened.is_monotonic_increasing:
        raise ValueError("opened_values must be unique and sorted")

    valid_close = np.isfinite(close) & (close > 0.0)
    minute_returns = np.full(len(close), np.nan, dtype=np.float64)
    if len(close) > 1:
        adjacent = np.asarray((opened[1:] - opened[:-1]) == pd.Timedelta(minutes=1))
        valid_pair = adjacent & valid_close[:-1] & valid_close[1:]
        current = np.flatnonzero(valid_pair) + 1
        minute_returns[current] = close[current] / close[current - 1] - 1.0
        invalid_return = ~np.isfinite(minute_returns[current])
        minute_returns[current[invalid_return]] = np.nan

    sigma = (
        pd.Series(minute_returns)
        .rolling(VOLATILITY_RETURN_COUNT, min_periods=VOLATILITY_RETURN_COUNT)
        .std(ddof=0)
        .to_numpy(dtype=np.float64, copy=False)
    )
    return sigma * np.sqrt(float(VOLATILITY_TARGET_MINUTES))


def normalize_preopen_return(return_values, sigma_5m_values) -> np.ndarray:
    """Compute z = r / max(sigma_5m, 1e-6), preserving unavailable rows as NaN."""
    returns = np.asarray(return_values, dtype=np.float64)
    sigma = np.asarray(sigma_5m_values, dtype=np.float64)
    if returns.ndim != 1 or sigma.ndim != 1 or len(returns) != len(sigma):
        raise ValueError("return and sigma inputs must be aligned one-dimensional arrays")
    if np.any(np.isfinite(sigma) & (sigma < 0.0)):
        raise ValueError("sigma_5m cannot be negative")
    valid = np.isfinite(returns) & np.isfinite(sigma)
    normalized = np.full(len(returns), np.nan, dtype=np.float64)
    denominator = np.maximum(sigma[valid], VOLATILITY_DENOMINATOR_FLOOR)
    normalized[valid] = returns[valid] / denominator
    return normalized
