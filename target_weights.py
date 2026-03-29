import numpy as np
import pandas as pd

TARGET_WEIGHT_COL = "target_5m_weight"
TARGET_WEIGHT_MINUTE_MODULO = 5
TARGET_WEIGHT_MINUTE_REMAINDER = 4
TARGET_WEIGHT_DECISION_VALUE = 0.80
TARGET_WEIGHT_OTHER_VALUE = 0.20 / 4


def _format_weight_key(value):
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def compute_decision_mask_from_opened(
    opened_values,
    minute_modulo=TARGET_WEIGHT_MINUTE_MODULO,
    minute_remainder=TARGET_WEIGHT_MINUTE_REMAINDER,
):
    opened_index = pd.DatetimeIndex(pd.to_datetime(opened_values, errors="raise"))
    opened_minute = opened_index.minute.to_numpy(dtype=np.int16, copy=False)
    return (opened_minute % int(minute_modulo)) == int(minute_remainder)


def compute_target_weights_from_opened(opened_values):
    decision_mask = compute_decision_mask_from_opened(opened_values)
    weights = np.where(
        decision_mask,
        TARGET_WEIGHT_DECISION_VALUE,
        TARGET_WEIGHT_OTHER_VALUE,
    )
    return weights.astype(np.float64, copy=False)


def compute_binary_close_target_from_opened(
    opened_values,
    close_values,
    horizon_minutes,
):
    horizon = int(horizon_minutes)
    if horizon <= 0:
        raise ValueError(f"horizon_minutes must be > 0, got: {horizon_minutes}")

    opened_index = pd.DatetimeIndex(pd.to_datetime(opened_values, errors="raise"))
    if opened_index.has_duplicates:
        dup_count = int(opened_index.duplicated().sum())
        raise ValueError(f"Duplicate Opened values found: {dup_count}")

    close_np = pd.to_numeric(close_values, errors="coerce").to_numpy(
        dtype=np.float64,
        copy=False,
    )
    close_series = pd.Series(close_np, index=opened_index)
    future_opened = opened_index + pd.Timedelta(minutes=horizon)
    future_close = close_series.reindex(future_opened).to_numpy(
        dtype=np.float64, copy=False
    )
    current_close = close_series.to_numpy(dtype=np.float64, copy=False)

    target = np.full(len(close_series), np.nan, dtype=np.float64)
    valid_mask = np.isfinite(current_close) & np.isfinite(future_close)
    if np.any(valid_mask):
        # Keep target semantics aligned with Polymarket settlement: ties resolve Up.
        target[valid_mask] = (
            future_close[valid_mask] >= current_close[valid_mask]
        ).astype(np.float64, copy=False)
    return target


def add_target_weights(df, opened_col="Opened", weight_col=TARGET_WEIGHT_COL):
    if opened_col not in df.columns:
        raise ValueError(
            f"Cannot build target weights without opened column '{opened_col}'."
        )

    out = df.copy()
    out[weight_col] = compute_target_weights_from_opened(out[opened_col])
    return out


def summarize_target_weights(weights):
    weights_np = np.asarray(weights, dtype=np.float64)
    if weights_np.ndim != 1:
        raise ValueError("Target weights summary expects a 1D array.")
    if weights_np.size == 0:
        raise ValueError("Cannot summarize empty target weights.")

    unique_weights, counts = np.unique(weights_np, return_counts=True)
    distribution = {
        _format_weight_key(weight): int(count)
        for weight, count in zip(unique_weights, counts)
    }
    return {
        "min": float(np.min(weights_np)),
        "max": float(np.max(weights_np)),
        "mean": float(np.mean(weights_np)),
        "sum": float(np.sum(weights_np)),
        "distribution": distribution,
    }
