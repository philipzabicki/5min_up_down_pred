import numpy as np
import pandas as pd


TARGET_WEIGHT_COL = "target_5m_weight"
TARGET_WEIGHT_MINUTE_MODULO = 5
TARGET_WEIGHT_MINUTE_REMAINDER = 4
TARGET_WEIGHT_DECISION_VALUE = 0.75
TARGET_WEIGHT_OTHER_VALUE = 0.25/4


def _format_weight_key(value):
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def compute_target_weights_from_opened(opened_values):
    opened_index = pd.DatetimeIndex(pd.to_datetime(opened_values, errors="raise"))
    opened_minute = opened_index.minute.to_numpy(dtype=np.int16, copy=False)
    weights = np.where(
        (opened_minute % TARGET_WEIGHT_MINUTE_MODULO) == TARGET_WEIGHT_MINUTE_REMAINDER,
        TARGET_WEIGHT_DECISION_VALUE,
        TARGET_WEIGHT_OTHER_VALUE,
    )
    return weights.astype(np.float32, copy=False)


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
