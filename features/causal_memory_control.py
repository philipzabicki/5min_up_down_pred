"""Small causal-history representation used as a control for ESN features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit


MEMORY_LAGS = (1, 5, 15, 60, 240, 720)
MEMORY_HALFLIVES = (15, 60, 240, 720)
MEMORY_MAX_LAG = max(MEMORY_LAGS)
_ALPHAS = np.asarray([1.0 - 0.5 ** (1.0 / value) for value in MEMORY_HALFLIVES], dtype=np.float64)


@njit(cache=True, nogil=True)
def _update_block(values, timestamp_ns, valid, history, ema, step_count, last_timestamp_ns,
                  has_last_timestamp, output, ready):
    capacity, width = history.shape
    for row in range(values.shape[0]):
        timestamp = timestamp_ns[row]
        if has_last_timestamp:
            gap = timestamp - last_timestamp_ns
            if gap <= 0:
                raise ValueError("memory-control timestamps must be strictly increasing")
            if gap != 60_000_000_000:
                history.fill(0.0)
                ema.fill(0.0)
                step_count = 0
        if not valid[row]:
            history.fill(0.0)
            ema.fill(0.0)
            step_count = 0
            last_timestamp_ns = timestamp
            has_last_timestamp = True
            continue

        slot = step_count % capacity
        for column in range(width):
            value = values[row, column]
            history[slot, column] = value
            if step_count == 0:
                for half_life in range(ema.shape[0]):
                    ema[half_life, column] = value
            else:
                for half_life in range(ema.shape[0]):
                    alpha = _ALPHAS[half_life]
                    ema[half_life, column] = alpha * value + (1.0 - alpha) * ema[half_life, column]

        step_count += 1
        if step_count > MEMORY_MAX_LAG:
            ready[row] = True
            for column in range(width):
                offset = column * (len(MEMORY_LAGS) + len(MEMORY_HALFLIVES) + 2)
                for lag_index, lag in enumerate(MEMORY_LAGS):
                    output[row, offset + lag_index] = history[(slot - lag) % capacity, column]
                base = offset + len(MEMORY_LAGS)
                for half_life in range(len(MEMORY_HALFLIVES)):
                    output[row, base + half_life] = ema[half_life, column]
                output[row, base + len(MEMORY_HALFLIVES)] = values[row, column] - ema[1, column]
                output[row, base + len(MEMORY_HALFLIVES) + 1] = values[row, column] - ema[3, column]
        last_timestamp_ns = timestamp
        has_last_timestamp = True
    return step_count, last_timestamp_ns, has_last_timestamp


def memory_feature_names(columns):
    names = []
    for column in columns:
        names.extend(f"{column}_lag_{lag}m" for lag in MEMORY_LAGS)
        names.extend(f"{column}_ema_hl_{half_life}m" for half_life in MEMORY_HALFLIVES)
        names.extend((f"{column}_minus_ema_hl_60m", f"{column}_minus_ema_hl_720m"))
    return tuple(names)


@dataclass
class CausalMemoryState:
    """Streaming state for fixed lags, exponential averages, and level deviations."""

    input_width: int

    def __post_init__(self):
        self.history = np.zeros((MEMORY_MAX_LAG + 1, int(self.input_width)), dtype=np.float64)
        self.ema = np.zeros((len(MEMORY_HALFLIVES), int(self.input_width)), dtype=np.float64)
        self.step_count = 0
        self.last_timestamp_ns = 0
        self.has_last_timestamp = False

    @property
    def output_width(self):
        return int(self.input_width) * (len(MEMORY_LAGS) + len(MEMORY_HALFLIVES) + 2)

    def update_many(self, timestamps, values, *, valid_mask=None):
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="raise"))
        if np.any(ts.second != 0) or np.any(ts.microsecond != 0) or np.any(ts.nanosecond != 0):
            raise ValueError("memory-control timestamps must align to whole minutes")
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.input_width or len(ts) != len(x):
            raise ValueError("memory-control inputs do not match timestamps and configured width")
        valid = np.ones(len(x), dtype=np.bool_) if valid_mask is None else np.asarray(valid_mask, dtype=np.bool_)
        if valid.ndim != 1 or len(valid) != len(x):
            raise ValueError("valid_mask must match memory-control inputs")
        output = np.full((len(x), self.output_width), np.nan, dtype=np.float32)
        ready = np.zeros(len(x), dtype=np.bool_)
        self.step_count, self.last_timestamp_ns, self.has_last_timestamp = _update_block(
            x, ts.as_unit("ns").asi8, valid, self.history, self.ema,
            self.step_count, self.last_timestamp_ns, self.has_last_timestamp,
            output, ready,
        )
        return output, ready

    def update_one(self, timestamp, values, *, valid=True):
        output, ready = self.update_many([timestamp], np.asarray(values).reshape(1, -1), valid_mask=[valid])
        return output[0] if ready[0] else None

    def to_dict(self):
        return {
            "input_width": int(self.input_width),
            "history": self.history.tolist(),
            "ema": self.ema.tolist(),
            "step_count": int(self.step_count),
            "last_timestamp_ns": int(self.last_timestamp_ns),
            "has_last_timestamp": bool(self.has_last_timestamp),
        }

    @classmethod
    def from_dict(cls, payload):
        state = cls(int(payload["input_width"]))
        state.history = np.asarray(payload["history"], dtype=np.float64)
        state.ema = np.asarray(payload["ema"], dtype=np.float64)
        if state.history.shape != (MEMORY_MAX_LAG + 1, state.input_width):
            raise ValueError("memory-control checkpoint history shape differs")
        if state.ema.shape != (len(MEMORY_HALFLIVES), state.input_width):
            raise ValueError("memory-control checkpoint EMA shape differs")
        state.step_count = int(payload["step_count"])
        state.last_timestamp_ns = int(payload["last_timestamp_ns"])
        state.has_last_timestamp = bool(payload["has_last_timestamp"])
        return state
