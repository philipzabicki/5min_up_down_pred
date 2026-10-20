"""Causal Echo State Network features with shared batch and streaming semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without the optional JIT
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


ENGINE_VERSION = "btc_esn_v1"
MINUTE_NS = pd.Timedelta(minutes=1).value


def development_arrow_filter(opened_field, start, exclusive_cutoff):
    """Build a predicate that can be applied before materializing development rows."""
    return (opened_field >= start) & (opened_field < exclusive_cutoff)


def nominal_decision_ns(opened_ns, *, offset_ns=MINUTE_NS):
    return np.asarray(opened_ns, dtype=np.int64) + int(offset_ns)


def exclusive_opened_cutoff(exclusive_event_cutoff, *, event_offset_ns=MINUTE_NS):
    cutoff = pd.Timestamp(exclusive_event_cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("exclusive event cutoff must be timezone-aware")
    return cutoff.tz_convert("UTC") - pd.Timedelta(int(event_offset_ns), unit="ns")


def purged_training_mask(opened_ns, label_available_ns, first_decision_ns, *, development_start_ns=None):
    opened = np.asarray(opened_ns, dtype=np.int64)
    available = np.asarray(label_available_ns, dtype=np.int64)
    if opened.ndim != 1 or available.ndim != 1 or len(opened) != len(available):
        raise ValueError("opened and label availability timestamps must be aligned vectors")
    eligible = (opened < int(first_decision_ns)) & (available <= int(first_decision_ns))
    if development_start_ns is not None:
        eligible &= opened >= int(development_start_ns)
    return eligible

INPUT_GROUPS = {
    "returns": (
        "log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m",
    ),
    "candle": (
        "candle_body_log_return", "candle_close_location",
        "candle_upper_wick_fraction", "candle_lower_wick_fraction",
    ),
    "range_volatility": (
        "candle_range_fraction", "realized_volatility_5m",
        "realized_volatility_15m", "realized_volatility_60m",
    ),
    "volume": ("log_volume", "volume_ratio_to_prior_30m_median"),
    "basis": (
        "index_futures_log_basis", "index_futures_basis_change_1m",
        "index_futures_basis_change_5m",
    ),
    "time": ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos"),
}
INPUT_COLUMNS = tuple(column for columns in INPUT_GROUPS.values() for column in columns)
INPUT_COLUMN_GROUP = {
    column: group for group, columns in INPUT_GROUPS.items() for column in columns
}


@dataclass(frozen=True)
class ESNConfig:
    architecture: str = "single"
    reservoir_size: int = 64
    reservoir_count: int = 1
    leak_rate: float = 0.25
    input_scaling: float = 0.5
    bias_scaling: float = 0.1
    spectral_radius: float = 0.9
    topology: str = "sparse"
    connectivity: float = 0.15
    input_groups: tuple[str, ...] = ("returns", "candle", "range_volatility", "volume", "basis", "time")

    def __post_init__(self):
        if self.architecture not in {"single", "parallel", "deep"}:
            raise ValueError("architecture must be single, parallel, or deep")
        if self.topology not in {"dense", "sparse", "ring"}:
            raise ValueError("topology must be dense, sparse, or ring")
        if self.reservoir_size < 2 or self.reservoir_count < 1:
            raise ValueError("reservoir_size and reservoir_count are out of range")
        if self.architecture == "single" and self.reservoir_count != 1:
            raise ValueError("single architecture requires reservoir_count=1")
        if not 0.0 < self.leak_rate <= 1.0:
            raise ValueError("leak_rate must be in (0, 1]")
        if self.input_scaling <= 0.0 or self.bias_scaling < 0.0:
            raise ValueError("input_scaling must be positive and bias_scaling nonnegative")
        if self.spectral_radius <= 0.0 or not 0.0 < self.connectivity <= 1.0:
            raise ValueError("spectral_radius and connectivity must be positive")
        if not self.input_groups or any(group not in INPUT_GROUPS for group in self.input_groups):
            raise ValueError("input_groups must contain known feature groups")

    @property
    def state_count(self) -> int:
        return 1 if self.architecture == "single" else self.reservoir_count

    @property
    def deep(self) -> bool:
        return self.architecture == "deep"

    @property
    def selected_columns(self) -> tuple[str, ...]:
        return tuple(
            column for group in INPUT_GROUPS for column in INPUT_GROUPS[group]
            if group in self.input_groups
        )


class FeatureScaler:
    """Finite-value standardizer whose fit rows are supplied explicitly."""

    def __init__(self, columns, mean=None, scale=None, finite_counts=None):
        self.columns = tuple(columns)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float64)
        self.finite_counts = None if finite_counts is None else np.asarray(finite_counts, dtype=np.int64)

    def fit(self, values):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.columns) or not len(x):
            raise ValueError("scaler input must be a nonempty matrix matching columns")
        finite = np.isfinite(x)
        counts = finite.sum(axis=0)
        if np.any(counts == 0):
            missing = [self.columns[i] for i in np.flatnonzero(counts == 0)]
            raise ValueError(f"no finite training values for scaler columns: {missing}")
        safe = np.where(finite, x, 0.0)
        mean = safe.sum(axis=0) / counts
        centered = np.where(finite, x - mean, 0.0)
        variance = np.square(centered).sum(axis=0) / counts
        scale = np.sqrt(variance)
        scale[scale <= np.finfo(np.float64).eps] = 1.0
        self.mean = mean
        self.scale = scale
        self.finite_counts = counts.astype(np.int64)
        return self

    def transform(self, values):
        if self.mean is None or self.scale is None:
            raise RuntimeError("FeatureScaler must be fit before transform")
        x = np.asarray(values, dtype=np.float64)
        if x.shape[-1] != len(self.columns):
            raise ValueError("scaler input width differs from fitted columns")
        z = (x - self.mean) / self.scale
        z[~np.isfinite(z)] = 0.0
        return np.asarray(z, dtype=np.float32)

    def to_dict(self):
        if self.mean is None or self.scale is None:
            raise RuntimeError("cannot serialize an unfitted FeatureScaler")
        return {
            "columns": list(self.columns),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "finite_counts": self.finite_counts.tolist(),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(payload["columns"], payload["mean"], payload["scale"], payload["finite_counts"])


class FrozenProjection:
    """Small persisted PCA projection used by a fitted ESN feature bundle."""

    def __init__(self, mean, components, explained_variance_ratio=None):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.components = np.asarray(components, dtype=np.float64)
        self.explained_variance_ratio = (
            None if explained_variance_ratio is None
            else np.asarray(explained_variance_ratio, dtype=np.float64)
        )
        if self.mean.ndim != 1 or self.components.ndim != 2 or self.components.shape[1] != len(self.mean):
            raise ValueError("frozen projection dimensions are inconsistent")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.components).all():
            raise ValueError("frozen projection contains non-finite values")

    def transform(self, values):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean):
            raise ValueError("projection input width differs from fitted components")
        return (x - self.mean) @ self.components.T

    def to_dict(self):
        return {
            "mean": self.mean.tolist(),
            "components": self.components.tolist(),
            "explained_variance_ratio": None if self.explained_variance_ratio is None else self.explained_variance_ratio.tolist(),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(payload["mean"], payload["components"], payload.get("explained_variance_ratio"))


@njit(cache=True, nogil=True)
def _advance(states, x, win, rec_indices, rec_values, bias, leak, deep):
    next_states = np.empty_like(states)
    n_reservoirs, size = states.shape
    for r in range(n_reservoirs):
        if deep and r > 0:
            source_len = size
        else:
            source_len = x.shape[0]
        for i in range(size):
            activation = bias[r, i]
            if deep and r > 0:
                for j in range(source_len):
                    activation += win[r, i, j] * next_states[r - 1, j]
            else:
                for j in range(source_len):
                    activation += win[r, i, j] * x[j]
            for edge in range(rec_indices.shape[2]):
                j = rec_indices[r, i, edge]
                if j >= 0:
                    activation += rec_values[r, i, edge] * states[r, j]
            candidate = np.tanh(activation)
            next_states[r, i] = (1.0 - leak) * states[r, i] + leak * candidate
    return next_states


@njit(cache=True, nogil=True)
def _run_segment(inputs, initial_states, win, rec_indices, rec_values, bias, leak, deep):
    result = np.empty((inputs.shape[0], initial_states.size), dtype=np.float32)
    states = initial_states.copy()
    for row in range(inputs.shape[0]):
        states = _advance(states, inputs[row], win, rec_indices, rec_values, bias, leak, deep)
        result[row] = states.reshape(states.size)
    return result, states


class ESNEngine:
    """Leaky single, parallel, or shallow DeepESN with a serializable stream state.

    A timestamp gap resets the dynamic state before the next real observation. Missing
    candles are never synthesized. Duplicate and decreasing timestamps raise.
    """

    def __init__(self, config: ESNConfig, seed: int, scaler: FeatureScaler, *, warmup_steps: int = 0):
        if tuple(scaler.columns) != config.selected_columns:
            raise ValueError("scaler columns do not match the ESN input-group selection")
        if scaler.mean is None or scaler.scale is None:
            raise ValueError("ESNEngine requires a fitted input scaler")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        self.config = config
        self.seed = int(seed)
        self.scaler = scaler
        self.warmup_steps = int(warmup_steps)
        self._state_count = config.state_count
        self._size = config.reservoir_size
        self.states = np.zeros((self._state_count, self._size), dtype=np.float32)
        self.steps_since_reset = 0
        self.last_timestamp = None
        self._win, self._rec_indices, self._rec_values, self._bias = self._make_weights()

    def _make_weights(self):
        rng = np.random.default_rng(self.seed)
        n, size, input_size = self._state_count, self._size, len(self.config.selected_columns)
        max_source = max(size, input_size)
        win = np.zeros((n, size, max_source), dtype=np.float32)
        wrec = np.zeros((n, size, size), dtype=np.float32)
        bias = np.zeros((n, size), dtype=np.float32)
        for reservoir in range(n):
            source_size = size if self.config.deep and reservoir > 0 else input_size
            win[reservoir, :, :source_size] = rng.uniform(
                -self.config.input_scaling, self.config.input_scaling,
                size=(size, source_size),
            )
            bias[reservoir] = rng.uniform(
                -self.config.bias_scaling, self.config.bias_scaling, size=size,
            )
            if self.config.topology == "ring":
                for row in range(size):
                    wrec[reservoir, row, (row - 1) % size] = rng.normal()
                    wrec[reservoir, row, (row + 1) % size] = rng.normal()
            else:
                recurrent = rng.normal(size=(size, size))
                if self.config.topology == "sparse":
                    mask = rng.random((size, size)) < self.config.connectivity
                    np.fill_diagonal(mask, True)
                    recurrent *= mask
                wrec[reservoir] = recurrent.astype(np.float32)
            radius = float(np.max(np.abs(np.linalg.eigvals(wrec[reservoir].astype(np.float64)))))
            if not np.isfinite(radius) or radius <= np.finfo(np.float64).eps:
                raise ValueError("recurrent matrix has invalid spectral radius")
            wrec[reservoir] *= np.float32(self.config.spectral_radius / radius)
        max_degree = max(int(np.count_nonzero(wrec[r, i])) for r in range(n) for i in range(size))
        rec_indices = np.full((n, size, max_degree), -1, dtype=np.int32)
        rec_values = np.zeros((n, size, max_degree), dtype=np.float32)
        for reservoir in range(n):
            for row in range(size):
                columns = np.flatnonzero(wrec[reservoir, row])
                rec_indices[reservoir, row, :len(columns)] = columns
                rec_values[reservoir, row, :len(columns)] = wrec[reservoir, row, columns]
        return win, rec_indices, rec_values, bias

    @property
    def ready(self):
        return self.steps_since_reset > self.warmup_steps

    @property
    def feature_names(self):
        return tuple(
            f"esn_r{reservoir}_state_{unit:03d}"
            for reservoir in range(self._state_count)
            for unit in range(self._size)
        )

    def get_features(self):
        if not self.ready:
            return None
        return self.states.reshape(-1).copy()

    def _reset(self):
        self.states.fill(0.0)
        self.steps_since_reset = 0

    def update_one(self, timestamp, observation, *, valid=True):
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            raise ValueError("ESN timestamps must be timezone-aware")
        ts = ts.tz_convert("UTC")
        if ts.second or ts.microsecond or ts.nanosecond:
            raise ValueError("ESN timestamps must be aligned to whole minutes")
        if self.last_timestamp is not None:
            delta = ts.value - self.last_timestamp.value
            if delta <= 0:
                raise ValueError("duplicate or unordered ESN timestamp")
            if delta != MINUTE_NS:
                self._reset()
        if not valid:
            self._reset()
            self.last_timestamp = ts
            return None
        raw = np.asarray(observation, dtype=np.float64).reshape(1, -1)
        scaled = self.scaler.transform(raw)[0]
        self.states = _advance(
            self.states, scaled, self._win, self._rec_indices, self._rec_values,
            self._bias, self.config.leak_rate, self.config.deep,
        )
        self.steps_since_reset += 1
        self.last_timestamp = ts
        return self.get_features()

    def update_many(self, timestamps, observations, *, valid_mask=None):
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="raise"))
        if ts.has_duplicates or not ts.is_monotonic_increasing:
            raise ValueError("batch timestamps must be strictly increasing")
        if np.any(ts.second != 0) or np.any(ts.microsecond != 0) or np.any(ts.nanosecond != 0):
            raise ValueError("ESN timestamps must be aligned to whole minutes")
        x = np.asarray(observations, dtype=np.float64)
        if x.ndim != 2 or len(x) != len(ts) or x.shape[1] != len(self.config.selected_columns):
            raise ValueError("batch observations must match timestamps and selected inputs")
        valid = np.ones(len(ts), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        if valid.ndim != 1 or len(valid) != len(ts):
            raise ValueError("valid_mask must match batch timestamps")
        if not len(ts):
            return np.empty((0, self.states.size), dtype=np.float32), np.empty(0, dtype=bool)
        scaled = self.scaler.transform(x)
        timestamp_ns = ts.as_unit("ns").asi8
        states_out = np.full((len(ts), self.states.size), np.nan, dtype=np.float32)
        ready_out = np.zeros(len(ts), dtype=bool)
        if self.last_timestamp is not None:
            delta = timestamp_ns[0] - self.last_timestamp.value
            if delta <= 0:
                raise ValueError("duplicate or unordered ESN timestamp")
            if delta != MINUTE_NS:
                self._reset()
        consecutive = np.diff(timestamp_ns) == MINUTE_NS
        starts = np.flatnonzero(valid & np.r_[True, (~valid[:-1] | ~consecutive)])
        ends = np.flatnonzero(valid & np.r_[~consecutive | ~valid[1:], True]) + 1
        for start_value, end_value in zip(starts, ends):
            start, end = int(start_value), int(end_value)
            if start > 0:
                self._reset()
            init_steps = self.steps_since_reset
            states, final_state = _run_segment(
                scaled[start:end], self.states, self._win, self._rec_indices,
                self._rec_values, self._bias, self.config.leak_rate, self.config.deep,
            )
            states_out[start:end] = states
            self.states = final_state
            self.steps_since_reset = end - start + init_steps
            ready_out[start:end] = (init_steps + np.arange(1, end - start + 1)) > self.warmup_steps
            self.last_timestamp = ts[end - 1]
        if not valid[-1]:
            self._reset()
            self.last_timestamp = ts[-1]
        return states_out, ready_out

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "engine_version": ENGINE_VERSION,
            "config": {**asdict(self.config), "input_groups": list(self.config.input_groups)},
            "seed": self.seed,
            "warmup_steps": self.warmup_steps,
            "scaler": self.scaler.to_dict(),
            "states": self.states.tolist(),
            "steps_since_reset": self.steps_since_reset,
            "last_timestamp": None if self.last_timestamp is None else self.last_timestamp.isoformat(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)

    @classmethod
    def load_checkpoint(cls, path, *, expected_config=None, expected_seed=None):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("engine_version") != ENGINE_VERSION:
            raise ValueError("checkpoint engine version mismatch")
        config_data = payload["config"]
        config_data["input_groups"] = tuple(config_data["input_groups"])
        config = ESNConfig(**config_data)
        seed = int(payload["seed"])
        if expected_config is not None and config != expected_config:
            raise ValueError("checkpoint ESN configuration mismatch")
        if expected_seed is not None and seed != int(expected_seed):
            raise ValueError("checkpoint seed mismatch")
        scaler_data = payload["scaler"]
        scaler = FeatureScaler(
            scaler_data["columns"], scaler_data["mean"], scaler_data["scale"],
            scaler_data["finite_counts"],
        )
        engine = cls(config, seed, scaler, warmup_steps=int(payload["warmup_steps"]))
        states = np.asarray(payload["states"], dtype=np.float32)
        if states.shape != engine.states.shape or not np.isfinite(states).all():
            raise ValueError("checkpoint state shape or values are invalid")
        engine.states = states
        engine.steps_since_reset = int(payload["steps_since_reset"])
        last = payload["last_timestamp"]
        engine.last_timestamp = None if last is None else pd.Timestamp(last).tz_convert("UTC")
        return engine


def build_causal_esn_inputs(candles: pd.DataFrame):
    """Build minute-causal ESN inputs without filling timestamp gaps."""
    required = ("Opened", "Open", "High", "Low", "Close", "Volume")
    missing = [column for column in required if column not in candles]
    if missing:
        raise ValueError(f"missing OHLCV columns for ESN inputs: {missing}")
    opened = pd.DatetimeIndex(pd.to_datetime(candles["Opened"], utc=True, errors="raise"))
    if opened.has_duplicates or not opened.is_monotonic_increasing:
        raise ValueError("ESN candle timestamps must be unique and ordered; gaps are allowed")
    if np.any(opened.second != 0) or np.any(opened.microsecond != 0) or np.any(opened.nanosecond != 0):
        raise ValueError("ESN candle timestamps must be aligned to whole minutes")
    n = len(candles)
    result = {name: np.full(n, np.nan, dtype=np.float64) for name in INPUT_COLUMNS}
    if n == 0:
        return opened, pd.DataFrame(result, columns=INPUT_COLUMNS)

    opened_ns = opened.as_unit("ns").asi8
    segment_starts = np.r_[0, np.flatnonzero(np.diff(opened_ns) != MINUTE_NS) + 1]
    segment_ends = np.r_[segment_starts[1:], n]
    open_ = pd.to_numeric(candles["Open"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(candles["High"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(candles["Low"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(candles["Close"], errors="coerce").to_numpy(dtype=np.float64)
    volume = pd.to_numeric(candles["Volume"], errors="coerce").to_numpy(dtype=np.float64)
    futures = (
        pd.to_numeric(candles["UM_BTCUSDT_Close"], errors="coerce").to_numpy(dtype=np.float64)
        if "UM_BTCUSDT_Close" in candles else np.full(n, np.nan, dtype=np.float64)
    )
    relative_basis = (
        pd.to_numeric(candles["futures_index_basis_rel_1m"], errors="coerce").to_numpy(dtype=np.float64)
        if "futures_index_basis_rel_1m" in candles else np.full(n, np.nan, dtype=np.float64)
    )
    phase_hour = (opened.hour + opened.minute / 60.0) * (2.0 * np.pi / 24.0)
    phase_weekday = opened.dayofweek * (2.0 * np.pi / 7.0)
    result["hour_sin"] = np.sin(phase_hour)
    result["hour_cos"] = np.cos(phase_hour)
    result["weekday_sin"] = np.sin(phase_weekday)
    result["weekday_cos"] = np.cos(phase_weekday)

    for start, end in zip(segment_starts, segment_ends):
        start, end = int(start), int(end)
        sl = slice(start, end)
        log_close = np.log(np.where(close[sl] > 0.0, close[sl], np.nan))
        log_open = np.log(np.where(open_[sl] > 0.0, open_[sl], np.nan))
        if "UM_BTCUSDT_Close" in candles:
            log_futures = np.log(np.where(futures[sl] > 0.0, futures[sl], np.nan))
            basis = log_futures - log_close
        else:
            rel = relative_basis[sl]
            basis = np.log1p(np.where(rel > -1.0, rel, np.nan))
            log_futures = np.full(end - start, np.nan, dtype=np.float64)
        ret1 = np.full(end - start, np.nan, dtype=np.float64)
        for horizon in (1, 5, 15, 60):
            ret = np.full(end - start, np.nan, dtype=np.float64)
            if horizon < len(ret):
                ret[horizon:] = log_close[horizon:] - log_close[:-horizon]
            result[f"log_return_{horizon}m"][sl] = ret
            if horizon == 1:
                ret1 = ret
        candle_range = high[sl] - low[sl]
        valid_range = np.isfinite(candle_range) & (candle_range > 0.0)
        body = log_close - log_open
        result["candle_body_log_return"][sl] = body
        result["candle_range_fraction"][sl] = np.divide(
            candle_range, close[sl], out=np.full(end - start, np.nan),
            where=valid_range & (close[sl] > 0.0),
        )
        result["candle_close_location"][sl] = np.divide(
            close[sl] - low[sl], candle_range, out=np.full(end - start, np.nan), where=valid_range,
        )
        result["candle_upper_wick_fraction"][sl] = np.divide(
            high[sl] - np.maximum(open_[sl], close[sl]), candle_range,
            out=np.full(end - start, np.nan), where=valid_range,
        )
        result["candle_lower_wick_fraction"][sl] = np.divide(
            np.minimum(open_[sl], close[sl]) - low[sl], candle_range,
            out=np.full(end - start, np.nan), where=valid_range,
        )
        for window in (5, 15, 60):
            result[f"realized_volatility_{window}m"][sl] = pd.Series(ret1).rolling(
                window, min_periods=window,
            ).std(ddof=1).to_numpy()
        result["log_volume"][sl] = np.log1p(np.where(volume[sl] >= 0.0, volume[sl], np.nan))
        prior_median = pd.Series(volume[sl]).shift(1).rolling(30, min_periods=30).median().to_numpy()
        result["volume_ratio_to_prior_30m_median"][sl] = np.divide(
            volume[sl], prior_median, out=np.full(end - start, np.nan), where=prior_median > 0.0,
        )
        result["index_futures_log_basis"][sl] = basis
        basis_change_1 = np.full(end - start, np.nan, dtype=np.float64)
        basis_change_5 = np.full(end - start, np.nan, dtype=np.float64)
        if end - start > 1:
            basis_change_1[1:] = basis[1:] - basis[:-1]
        if end - start > 5:
            basis_change_5[5:] = basis[5:] - basis[:-5]
        result["index_futures_basis_change_1m"][sl] = basis_change_1
        result["index_futures_basis_change_5m"][sl] = basis_change_5
    return opened, pd.DataFrame(result, columns=INPUT_COLUMNS)


def estimate_warmup(training_inputs, config: ESNConfig, seed: int, *,
                    relative_tolerance: float = 1e-3, stable_window: int = 60,
                    initial_perturbation: float = 0.25, scaler: FeatureScaler | None = None,
                    timestamps=None, valid_mask=None, max_steps: int = 200_000):
    """Measure decay of an initial-state perturbation on training inputs only."""
    x = np.asarray(training_inputs, dtype=np.float32)
    if x.ndim != 2 or not len(x):
        raise ValueError("warm-up estimation needs a nonempty training input matrix")
    columns = config.selected_columns
    if scaler is None:
        scaler = FeatureScaler(columns).fit(x)
    elif tuple(scaler.columns) != columns or scaler.mean is None:
        raise ValueError("warm-up scaler must be fitted on the selected training inputs")
    normalized = scaler.transform(x)
    if max_steps < stable_window or stable_window < 1:
        raise ValueError("warm-up step budget must cover at least one stable window")
    valid = np.ones(len(x), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 1 or len(valid) != len(x):
        raise ValueError("warm-up valid_mask must match training inputs")
    if timestamps is None:
        continuous = np.ones(max(0, len(x) - 1), dtype=bool)
    else:
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="raise"))
        if len(ts) != len(x) or ts.has_duplicates or not ts.is_monotonic_increasing:
            raise ValueError("warm-up timestamps must uniquely match ordered training inputs")
        continuous = np.diff(ts.as_unit("ns").asi8) == MINUTE_NS
    starts = np.flatnonzero(valid & np.r_[True, (~valid[:-1] | ~continuous)])
    ends = np.flatnonzero(valid & np.r_[~continuous | ~valid[1:], True]) + 1
    engine = ESNEngine(config, seed, scaler, warmup_steps=0)
    required_steps = []
    residuals = []
    examined = 0
    unstable_segment = False
    for segment_number, (start, end) in enumerate(zip(starts, ends)):
        if examined >= max_steps:
            break
        length = min(int(end - start), max_steps - examined)
        if length < stable_window:
            examined += length
            continue
        control = engine.states.copy()
        rng = np.random.default_rng((seed ^ 0x5EED) + segment_number)
        perturbed = rng.uniform(-initial_perturbation, initial_perturbation, size=control.shape).astype(np.float32)
        window_values = np.empty(stable_window, dtype=np.float32)
        window_above = np.zeros(stable_window, dtype=bool)
        above_count = 0
        # For the 95th percentile to remain below tolerance, the largest order
        # statistic used by NumPy's linear quantile must also be below it.
        allowed_above = stable_window - 1 - int(np.ceil(0.95 * (stable_window - 1)))
        found = None
        residual = None
        for offset in range(length):
            row = int(start + offset)
            control = _advance(
                control, normalized[row], engine._win, engine._rec_indices,
                engine._rec_values, engine._bias, config.leak_rate, config.deep,
            )
            perturbed = _advance(
                perturbed, normalized[row], engine._win, engine._rec_indices,
                engine._rec_values, engine._bias, config.leak_rate, config.deep,
            )
            difference = perturbed - control
            error = float(np.sqrt(np.mean(np.square(difference))))
            norm = float(np.sqrt(np.mean(np.square(control))))
            relative = error / max(norm, 1e-6)
            position = offset % stable_window
            if offset >= stable_window:
                above_count -= int(window_above[position])
            window_values[position] = relative
            window_above[position] = relative > relative_tolerance
            above_count += int(window_above[position])
            if offset + 1 >= stable_window and above_count <= allowed_above:
                found = offset
                first = (offset + 1) % stable_window
                ordered_window = np.r_[window_values[first:], window_values[:first]]
                residual = float(np.quantile(ordered_window, 0.95))
                break
        examined += (found + 1) if found is not None else length
        if found is None:
            unstable_segment = True
        else:
            required_steps.append(found)
            residuals.append(residual)
    stabilized = bool(required_steps) and not unstable_segment
    warmup_steps = max(required_steps) if stabilized else int(max(required_steps, default=0))
    return warmup_steps, {
        "relative_tolerance": relative_tolerance,
        "stable_window_minutes": stable_window,
        "initial_perturbation": initial_perturbation,
        "relative_difference_p95_after_warmup": max(residuals) if residuals else None,
        "training_steps_examined": examined,
        "segments_examined": len(required_steps) + int(unstable_segment),
        "stabilized": stabilized,
    }


def build_nvar_features(inputs, timestamps, *, input_columns, max_lag=5, valid_mask=None):
    """Compact NVAR control: selected input lags and current quadratic terms."""
    x = np.asarray(inputs, dtype=np.float32)
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="raise"))
    if x.ndim != 2 or len(x) != len(ts) or x.shape[1] != len(input_columns):
        raise ValueError("NVAR inputs, timestamps, and names must align")
    if ts.has_duplicates or not ts.is_monotonic_increasing:
        raise ValueError("NVAR timestamps must be unique and ordered")
    if max_lag < 1:
        raise ValueError("max_lag must be positive")
    if not np.isfinite(x).all():
        raise ValueError("NVAR inputs must be finite after training-fitted scaling")
    valid = np.ones(len(x), dtype=bool) if valid_mask is None else np.array(valid_mask, dtype=bool, copy=True)
    if valid.ndim != 1 or len(valid) != len(x):
        raise ValueError("NVAR valid_mask must match inputs")
    lag_blocks = [np.roll(x, lag, axis=0) for lag in range(max_lag + 1)]
    lagged = np.concatenate(lag_blocks, axis=1)
    timestamp_ns = ts.as_unit("ns").asi8
    segment_id = np.cumsum(np.r_[0, ((np.diff(timestamp_ns) != MINUTE_NS) | (~valid[:-1]) | (~valid[1:])).astype(np.int64)])
    for lag in range(1, max_lag + 1):
        valid[:lag] = False
        valid[lag:] &= segment_id[lag:] == segment_id[:-lag]
    pair_i, pair_j = np.triu_indices(x.shape[1])
    quadratic = x[:, pair_i] * x[:, pair_j]
    values = np.concatenate((lagged, quadratic), axis=1)
    values[~valid] = np.nan
    names = tuple(
        f"nvar_lag{lag}_{name}"
        for lag in range(max_lag + 1) for name in input_columns
    ) + tuple(f"nvar_quad_{input_columns[i]}__{input_columns[j]}" for i, j in zip(pair_i, pair_j))
    return values, names, valid
