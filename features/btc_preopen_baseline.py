"""Causal raw BTC features for the pre-open experiment's first model."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "log_return_1m",
    "log_return_2m",
    "log_return_3m",
    "log_return_5m",
    "log_return_10m",
    "log_return_15m",
    "log_return_30m",
    "log_return_60m",
    "log_return_240m",
    "candle_body_log_return",
    "candle_range_fraction",
    "candle_close_location",
    "candle_upper_wick_fraction",
    "candle_lower_wick_fraction",
    "realized_volatility_5m",
    "realized_volatility_15m",
    "realized_volatility_60m",
    "realized_volatility_240m",
    "log_volume",
    "volume_ratio_to_prior_30m_median",
    "index_futures_log_basis",
    "index_futures_basis_change_1m",
    "index_futures_basis_change_5m",
    "futures_log_return_1m",
    "futures_log_return_5m",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)


def regularize_minute_candles(candles: pd.DataFrame) -> pd.DataFrame:
    """Insert missing UTC minutes as null rows so all lags retain minute meaning."""
    if "Opened" not in candles.columns:
        raise ValueError("BTC pre-open candles require an Opened column")
    out = candles.copy()
    opened = pd.DatetimeIndex(pd.to_datetime(out.pop("Opened"), utc=True, errors="raise"))
    if opened.has_duplicates:
        raise ValueError("Duplicate Opened timestamps in BTC pre-open candles")
    order = np.argsort(opened.asi8, kind="stable")
    out = out.iloc[order].copy()
    opened = opened.take(order)
    minute_ns = pd.Timedelta(minutes=1).value
    if len(opened) <= 1 or np.all(np.diff(opened.asi8) == minute_ns):
        out.insert(0, "Opened", opened)
        return out.reset_index(drop=True)
    minute_index = pd.date_range(opened[0], opened[-1], freq="min", tz="UTC")
    out.index = opened
    out = out.reindex(minute_index)
    out.index.name = "Opened"
    return out.reset_index()


def build_causal_preopen_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Build causal features from each fully closed candle and its past only."""
    required = ("Opened", "Open", "High", "Low", "Close", "Volume")
    missing = [column for column in required if column not in candles.columns]
    if missing:
        raise ValueError(f"Missing BTC pre-open feature inputs: {missing}")

    frame = regularize_minute_candles(candles)
    opened = pd.DatetimeIndex(frame["Opened"])
    open_ = pd.to_numeric(frame["Open"], errors="coerce")
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce")
    log_close = np.log(close.where(close > 0.0))

    values = {
        f"log_return_{horizon}m": log_close - log_close.shift(horizon)
        for horizon in (1, 2, 3, 5, 10, 15, 30, 60, 240)
    }
    candle_range = (high - low).where((high - low) > 0.0)
    values.update(
        {
            "candle_body_log_return": np.log(close.where(close > 0.0))
            - np.log(open_.where(open_ > 0.0)),
            "candle_range_fraction": candle_range / close.where(close > 0.0),
            "candle_close_location": (close - low) / candle_range,
            "candle_upper_wick_fraction": (high - pd.concat([open_, close], axis=1).max(axis=1))
            / candle_range,
            "candle_lower_wick_fraction": (pd.concat([open_, close], axis=1).min(axis=1) - low)
            / candle_range,
        }
    )
    one_minute_returns = values["log_return_1m"]
    for window in (5, 15, 60, 240):
        values[f"realized_volatility_{window}m"] = one_minute_returns.rolling(
            window, min_periods=window
        ).std(ddof=1)
    values["log_volume"] = np.log1p(volume.where(volume >= 0.0))
    prior_median_volume = volume.shift(1).rolling(30, min_periods=30).median()
    values["volume_ratio_to_prior_30m_median"] = volume / prior_median_volume.where(
        prior_median_volume > 0.0
    )

    if "UM_BTCUSDT_Close" in frame.columns:
        futures_close = pd.to_numeric(frame["UM_BTCUSDT_Close"], errors="coerce")
        log_basis = np.log(futures_close.where(futures_close > 0.0)) - log_close
        values["index_futures_log_basis"] = log_basis
        values["index_futures_basis_change_1m"] = log_basis - log_basis.shift(1)
        values["index_futures_basis_change_5m"] = log_basis - log_basis.shift(5)
        log_futures = np.log(futures_close.where(futures_close > 0.0))
        values["futures_log_return_1m"] = log_futures - log_futures.shift(1)
        values["futures_log_return_5m"] = log_futures - log_futures.shift(5)
    else:
        for name in FEATURE_COLUMNS[20:25]:
            values[name] = pd.Series(np.nan, index=frame.index, dtype=np.float64)

    hour_phase = (opened.hour + opened.minute / 60.0) * (2.0 * np.pi / 24.0)
    weekday_phase = opened.dayofweek * (2.0 * np.pi / 7.0)
    values["hour_sin"] = np.sin(hour_phase)
    values["hour_cos"] = np.cos(hour_phase)
    values["weekday_sin"] = np.sin(weekday_phase)
    values["weekday_cos"] = np.cos(weekday_phase)

    result = pd.DataFrame({"Opened": opened})
    for feature in FEATURE_COLUMNS:
        result[feature] = np.asarray(values[feature], dtype=np.float64)
    return result
