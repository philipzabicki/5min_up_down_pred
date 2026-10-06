"""Timestamp contract for BTC five-minute pre-open predictions (variant 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd


CONTRACT_VERSION = "btc_preopen_v1"
TARGET_COL = "target_preopen_5m_up_binance_index_proxy"
RETURN_TARGET_COL = "target_preopen_5m_return_binance_index_proxy"
TARGET_WEIGHT_COL = "target_preopen_5m_weight"
DECISION_COL = "is_polymarket_preopen_decision"
DECISION_AVAILABLE_COL = "nominal_decision_at"
FEATURE_AVAILABLE_COL = "feature_candle_close_at"
TARGET_START_COL = "target_window_start_at"
TARGET_END_COL = "target_window_end_at"
TARGET_AVAILABLE_COL = "target_label_available_at"
TARGET_START_PRICE_COL = "target_window_start_open"
TARGET_END_PRICE_COL = "target_window_end_close"
PREDICTION_DEADLINE_COL = "prediction_deadline_at"


def build_preopen_contract_frame(
        candles: pd.DataFrame,
        *,
        opened_col: str = "Opened",
        open_col: str = "Open",
        close_col: str = "Close",
        prediction_compute_budget_seconds: float = 0.0,
) -> pd.DataFrame:
    """Attach variant-1 timestamps and the minute-forward five-minute price proxy.

    A row opened at ``t`` becomes feature-available at ``t+1m``. Its forecast
    window begins at ``t+2m`` and ends at ``t+7m``: the open of candle ``t+2m``
    compared with the close of candle ``t+6m``. Every candle in that five-minute
    window must be present. Equal prices resolve UP, matching Polymarket ties.
    """
    required = (opened_col, open_col, close_col)
    missing = [name for name in required if name not in candles.columns]
    if missing:
        raise ValueError(f"Missing BTC pre-open input columns: {missing}")

    out = candles.copy()
    opened = pd.DatetimeIndex(pd.to_datetime(out[opened_col], utc=True, errors="raise"))
    if opened.has_duplicates:
        raise ValueError("Duplicate Opened timestamps found in BTC pre-open input")
    if not opened.is_monotonic_increasing:
        raise ValueError("BTC pre-open input candles must be sorted by Opened")

    step = pd.Timedelta(minutes=1)
    nominal_decision = opened + step
    target_start = nominal_decision + step
    target_end = target_start + pd.Timedelta(minutes=5)
    target_last_candle = target_start + pd.Timedelta(minutes=4)

    open_values = pd.to_numeric(out[open_col], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    close_values = pd.to_numeric(out[close_col], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    open_by_time = pd.Series(open_values, index=opened)
    close_by_time = pd.Series(close_values, index=opened)
    start_prices = open_by_time.reindex(target_start).to_numpy(
        dtype=np.float64, copy=False
    )
    end_prices = close_by_time.reindex(target_last_candle).to_numpy(
        dtype=np.float64, copy=False
    )

    # Require all five future candles; endpoints alone can hide a data gap.
    contiguous = np.ones(len(opened), dtype=np.bool_)
    for offset in range(5):
        required_times = target_start + pd.Timedelta(minutes=offset)
        contiguous &= opened.get_indexer(required_times) >= 0

    valid = contiguous & np.isfinite(start_prices) & np.isfinite(end_prices)
    target = np.full(len(opened), np.nan, dtype=np.float64)
    target[valid] = (end_prices[valid] >= start_prices[valid]).astype(np.float64)

    out[opened_col] = opened
    out[FEATURE_AVAILABLE_COL] = nominal_decision
    out[DECISION_AVAILABLE_COL] = nominal_decision
    out[TARGET_START_COL] = target_start
    out[TARGET_END_COL] = target_end
    out[TARGET_AVAILABLE_COL] = target_end
    prediction_ready_at = nominal_decision + pd.to_timedelta(
        float(prediction_compute_budget_seconds), unit="s"
    )
    out["nominal_prediction_available_at"] = prediction_ready_at
    out[TARGET_START_PRICE_COL] = start_prices
    out[TARGET_END_PRICE_COL] = end_prices
    out[TARGET_COL] = target
    out[RETURN_TARGET_COL] = build_preopen_return_target(out)
    # A decision is valid when the last candle is closed before the market opens.
    # Derive schedule alignment from availability, not from the candle's Opened phase.
    out[PREDICTION_DEADLINE_COL] = target_start
    out["actual_prediction_available_at"] = pd.NaT
    out[DECISION_COL] = (
        ((target_start.minute % 5) == 0)
        & (prediction_ready_at < target_start)
    )
    return out


def build_preopen_return_target(contract_frame: pd.DataFrame) -> np.ndarray:
    """Return the price-proxy return over the exact pre-open contract window."""
    start = pd.to_numeric(
        contract_frame[TARGET_START_PRICE_COL], errors="coerce"
    ).to_numpy(dtype=np.float64, copy=False)
    end = pd.to_numeric(
        contract_frame[TARGET_END_PRICE_COL], errors="coerce"
    ).to_numpy(dtype=np.float64, copy=False)
    valid = (
        np.isfinite(start)
        & np.isfinite(end)
        & (start > 0.0)
        & np.isfinite(contract_frame[TARGET_COL].to_numpy(dtype=np.float64, copy=False))
    )
    result = np.full(len(contract_frame), np.nan, dtype=np.float64)
    result[valid] = end[valid] / start[valid] - 1.0
    return result


def preopen_decision_mask(contract_frame: pd.DataFrame) -> np.ndarray:
    """Return only scheduled rows whose features and prediction fit before entry."""
    opened = pd.DatetimeIndex(
        pd.to_datetime(contract_frame["Opened"], utc=True, errors="raise")
    )
    feature_available = pd.DatetimeIndex(
        pd.to_datetime(contract_frame[FEATURE_AVAILABLE_COL], utc=True, errors="raise")
    )
    deadline = pd.DatetimeIndex(
        pd.to_datetime(contract_frame[PREDICTION_DEADLINE_COL], utc=True, errors="raise")
    )
    actual_available = pd.DatetimeIndex(
        pd.to_datetime(
            contract_frame["actual_prediction_available_at"],
            utc=True,
            errors="coerce",
        )
    )
    nominal_ready = pd.DatetimeIndex(
        pd.to_datetime(
            contract_frame["nominal_prediction_available_at"],
            utc=True,
            errors="raise",
        )
    )
    ready_ns = np.where(actual_available.isna(), nominal_ready.asi8, actual_available.asi8)
    return (
        (opened == feature_available - pd.Timedelta(minutes=1))
        & (deadline.minute % 5 == 0)
        & (ready_ns < deadline.asi8)
    )


def build_preopen_target(
        opened_values,
        open_values,
        close_values,
) -> pd.DataFrame:
    """Return only target-contract fields for existing dataset builders."""
    return build_preopen_contract_frame(
        pd.DataFrame(
            {
                "Opened": opened_values,
                "Open": open_values,
                "Close": close_values,
            }
        )
    )


def purge_unavailable_training_rows(
        opened_values,
        validation_decision_at,
        train_indices=None,
) -> np.ndarray:
    """Keep training labels known by the first validation decision timestamp."""
    opened = pd.DatetimeIndex(pd.to_datetime(opened_values, utc=True, errors="raise"))
    cutoff = pd.Timestamp(validation_decision_at)
    if cutoff.tzinfo is None:
        raise ValueError("validation_decision_at must be timezone-aware")
    label_available = opened + pd.Timedelta(minutes=7)
    candidates = (
        np.arange(len(opened), dtype=np.int64)
        if train_indices is None
        else np.asarray(train_indices, dtype=np.int64)
    )
    if candidates.ndim != 1 or np.any(candidates < 0) or np.any(candidates >= len(opened)):
        raise ValueError("train_indices must be a one-dimensional in-range index array")
    return candidates[label_available[candidates] <= cutoff.tz_convert("UTC")]


def build_preopen_target_weights(
        opened_values,
        *,
        decision_weight: float = 0.4625,
        auxiliary_total_weight: float = 0.5375,
        decision_mask=None,
) -> np.ndarray:
    """Default equal-five-minute-mass weights, pending development-only tuning."""
    opened = pd.DatetimeIndex(pd.to_datetime(opened_values, utc=True, errors="raise"))
    if decision_weight <= 0.0 or auxiliary_total_weight <= 0.0:
        raise ValueError("Target weight masses must be positive")
    if decision_mask is None:
        decision_mask = ((opened + pd.Timedelta(minutes=2)).minute % 5) == 0
    else:
        decision_mask = np.asarray(decision_mask, dtype=np.bool_)
        if decision_mask.ndim != 1 or len(decision_mask) != len(opened):
            raise ValueError("decision_mask must be a 1D array matching opened_values")
    return np.where(
        decision_mask,
        float(decision_weight),
        float(auxiliary_total_weight) / 4.0,
    )


def scheduled_market_start_for_opened(opened_value):
    """Return the next five-minute market start for a pre-open input candle."""
    opened = pd.Timestamp(opened_value)
    if opened.tzinfo is None:
        raise ValueError("opened_value must be timezone-aware")
    opened = opened.tz_convert("UTC")
    market_start = opened + pd.Timedelta(minutes=2)
    if market_start.minute % 5 != 0 or opened.second != 0 or opened.microsecond != 0:
        return None
    return market_start


def last_closed_candle_opened_at(nominal_decision_at):
    """Return the start timestamp of the latest 1m candle closed by decision time."""
    decision_at = pd.Timestamp(nominal_decision_at)
    if decision_at.tzinfo is None:
        decision_at = decision_at.tz_localize("UTC")
    else:
        decision_at = decision_at.tz_convert("UTC")
    return decision_at - pd.Timedelta(minutes=1)
