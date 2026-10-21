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


ENGINE_VERSION = "btc_esn_v2"
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
    component_sizes: tuple[int, ...] | None = None
    component_input_groups: tuple[tuple[str, ...], ...] | None = None
    component_topologies: tuple[str, ...] | None = None
    component_leak_rates: tuple[float, ...] | None = None
    component_spectral_radii: tuple[float, ...] | None = None
    component_input_scalings: tuple[float, ...] | None = None
    component_bias_scalings: tuple[float, ...] | None = None
    sparse_degree: int | None = None
    component_sparse_degrees: tuple[int, ...] | None = None
    branch_count: int = 1
    layer_count: int = 1

    def __post_init__(self):
        if self.architecture not in {"single", "parallel", "deep", "parallel_deep"}:
            raise ValueError("architecture must be single, parallel, deep, or parallel_deep")
        allowed_topologies = {"dense", "sparse", "ring", "orthogonal", "block_sparse"}
        if self.topology not in allowed_topologies:
            raise ValueError("unknown recurrent topology")
        if self.reservoir_size < 2 or self.reservoir_count < 1:
            raise ValueError("reservoir_size and reservoir_count are out of range")
        if self.architecture == "single" and self.reservoir_count != 1:
            raise ValueError("single architecture requires reservoir_count=1")
        if self.architecture == "parallel_deep":
            if self.branch_count < 2 or self.layer_count < 2:
                raise ValueError("parallel_deep requires at least two branches and layers")
            if self.reservoir_count != self.branch_count * self.layer_count:
                raise ValueError("parallel_deep reservoir_count must equal branches × layers")
        elif self.branch_count != 1 or self.layer_count != 1:
            raise ValueError("branch_count/layer_count apply only to parallel_deep")
        if not 0.0 < self.leak_rate <= 1.0:
            raise ValueError("leak_rate must be in (0, 1]")
        if self.input_scaling <= 0.0 or self.bias_scaling < 0.0:
            raise ValueError("input_scaling must be positive and bias_scaling nonnegative")
        if self.spectral_radius <= 0.0 or not 0.0 < self.connectivity <= 1.0:
            raise ValueError("spectral_radius and connectivity must be positive")
        if not self.input_groups or any(group not in INPUT_GROUPS for group in self.input_groups):
            raise ValueError("input_groups must contain known feature groups")
        configured_sizes = self.component_sizes if self.component_sizes is not None else (self.reservoir_size,) * self.state_count
        if sum(configured_sizes) > 2048:
            raise ValueError("total ESN units must not exceed the frozen 2048-unit budget")
        sizes = self.component_sizes
        if sizes is not None and (len(sizes) != self.state_count or any(int(size) < 2 for size in sizes)):
            raise ValueError("component_sizes must provide at least two units per component")
        if self.component_input_groups is not None:
            if len(self.component_input_groups) != self.state_count:
                raise ValueError("component_input_groups must match the component count")
            if any(group not in INPUT_GROUPS for groups in self.component_input_groups for group in groups):
                raise ValueError("component_input_groups contains an unknown feature group")
        for name, values, lower, upper in (
            ("component_leak_rates", self.component_leak_rates, 0.0, 1.0),
            ("component_spectral_radii", self.component_spectral_radii, 0.0, float("inf")),
            ("component_input_scalings", self.component_input_scalings, 0.0, float("inf")),
        ):
            if values is not None and (len(values) != self.state_count or any(not lower < float(v) <= upper for v in values)):
                raise ValueError(f"{name} must contain one valid value per component")
        if self.component_bias_scalings is not None and (
            len(self.component_bias_scalings) != self.state_count
            or any(not np.isfinite(float(value)) or float(value) < 0.0 for value in self.component_bias_scalings)
        ):
            raise ValueError("component_bias_scalings must contain one nonnegative value per component")
        if self.component_topologies is not None and (
            len(self.component_topologies) != self.state_count
            or any(topology not in allowed_topologies for topology in self.component_topologies)
        ):
            raise ValueError("component_topologies must contain one supported topology per component")
        if self.sparse_degree is not None and self.sparse_degree < 1:
            raise ValueError("sparse_degree must be positive")
        if self.component_sparse_degrees is not None and (
            len(self.component_sparse_degrees) != self.state_count
            or any(int(degree) < 1 for degree in self.component_sparse_degrees)
        ):
            raise ValueError("component_sparse_degrees must contain one positive degree per component")

    @property
    def state_count(self) -> int:
        return 1 if self.architecture == "single" else self.reservoir_count

    @property
    def unit_sizes(self) -> tuple[int, ...]:
        sizes = self.component_sizes if self.component_sizes is not None else (self.reservoir_size,) * self.state_count
        return tuple(int(size) for size in sizes)

    @property
    def total_units(self) -> int:
        return sum(self.unit_sizes)

    @property
    def input_groups_by_component(self) -> tuple[tuple[str, ...], ...]:
        if self.component_input_groups is not None:
            return tuple(tuple(groups) for groups in self.component_input_groups)
        if self.architecture == "deep":
            return (tuple(self.input_groups),) + ((),) * (self.state_count - 1)
        if self.architecture == "parallel_deep":
            return tuple(tuple(self.input_groups) if layer == 0 else ()
                         for _branch in range(self.branch_count)
                         for layer in range(self.layer_count))
        return (tuple(self.input_groups),) * self.state_count

    @property
    def source_components(self) -> tuple[int, ...]:
        if self.architecture == "deep":
            return tuple(index - 1 if index else -1 for index in range(self.state_count))
        if self.architecture == "parallel_deep":
            return tuple(
                index - 1 if layer else -1
                for branch in range(self.branch_count)
                for layer in range(self.layer_count)
                for index in (branch * self.layer_count + layer,)
            )
        return (-1,) * self.state_count

    def _component_values(self, values, shared):
        return tuple(values) if values is not None else (shared,) * self.state_count

    @property
    def leak_rates(self) -> tuple[float, ...]:
        return self._component_values(self.component_leak_rates, self.leak_rate)

    @property
    def spectral_radii(self) -> tuple[float, ...]:
        return self._component_values(self.component_spectral_radii, self.spectral_radius)

    @property
    def input_scalings(self) -> tuple[float, ...]:
        return self._component_values(self.component_input_scalings, self.input_scaling)

    @property
    def bias_scalings(self) -> tuple[float, ...]:
        return self._component_values(self.component_bias_scalings, self.bias_scaling)

    @property
    def topologies(self) -> tuple[str, ...]:
        return self._component_values(self.component_topologies, self.topology)

    @property
    def sparse_degrees(self) -> tuple[int, ...]:
        if self.component_sparse_degrees is not None:
            return tuple(int(value) for value in self.component_sparse_degrees)
        return tuple(
            int(self.sparse_degree) if self.sparse_degree is not None
            else max(1, int(round(self.connectivity * size)))
            for size in self.unit_sizes
        )

    @property
    def deep(self) -> bool:
        return self.architecture in {"deep", "parallel_deep"}

    @property
    def selected_columns(self) -> tuple[str, ...]:
        groups = {group for component in self.input_groups_by_component for group in component}
        return tuple(
            column for group in INPUT_GROUPS for column in INPUT_GROUPS[group]
            if group in groups
        )

    def component_columns(self, index: int) -> tuple[str, ...]:
        groups = set(self.input_groups_by_component[index])
        return tuple(column for column in self.selected_columns if INPUT_COLUMN_GROUP[column] in groups)


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
def _advance(states, x, win, rec_indices, rec_values, bias, leak, component_sizes,
             source_components, input_indices, input_counts):
    next_states = np.empty_like(states)
    next_states.fill(0.0)
    n_reservoirs = states.shape[0]
    for r in range(n_reservoirs):
        source_component = source_components[r]
        for i in range(component_sizes[r]):
            activation = bias[r, i]
            if source_component >= 0:
                for j in range(component_sizes[source_component]):
                    activation += win[r, i, j] * next_states[source_component, j]
            else:
                for j in range(input_counts[r]):
                    activation += win[r, i, j] * x[input_indices[r, j]]
            for edge in range(rec_indices.shape[2]):
                j = rec_indices[r, i, edge]
                if j >= 0:
                    activation += rec_values[r, i, edge] * states[r, j]
            candidate = np.tanh(activation)
            next_states[r, i] = (1.0 - leak[r]) * states[r, i] + leak[r] * candidate
    return next_states


@njit(cache=True, nogil=True)
def _run_segment(inputs, initial_states, win, rec_indices, rec_values, bias, leak,
                 component_sizes, source_components, input_indices, input_counts):
    total_units = 0
    for size in component_sizes:
        total_units += size
    result = np.empty((inputs.shape[0], total_units), dtype=np.float32)
    states = initial_states.copy()
    for row in range(inputs.shape[0]):
        states = _advance(states, inputs[row], win, rec_indices, rec_values, bias, leak,
                          component_sizes, source_components, input_indices, input_counts)
        offset = 0
        for component in range(len(component_sizes)):
            size = component_sizes[component]
            result[row, offset:offset + size] = states[component, :size]
            offset += size
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
        self._sizes = np.asarray(config.unit_sizes, dtype=np.int32)
        self._size = int(self._sizes.max())
        self._total_units = config.total_units
        self._source_components = np.asarray(config.source_components, dtype=np.int32)
        self.states = np.zeros((self._state_count, self._size), dtype=np.float32)
        self.steps_since_reset = 0
        self.last_timestamp = None
        (self._win, self._rec_indices, self._rec_values, self._bias,
         self._input_indices, self._input_counts) = self._make_weights()

    def _make_weights(self):
        rng = np.random.default_rng(self.seed)
        n, max_size = self._state_count, self._size
        input_columns = self.config.selected_columns
        input_positions = {column: index for index, column in enumerate(input_columns)}
        component_columns = [self.config.component_columns(index) for index in range(n)]
        input_counts = np.asarray([len(columns) for columns in component_columns], dtype=np.int32)
        input_indices = np.full((n, max(1, int(input_counts.max(initial=0)))), -1, dtype=np.int32)
        for component, columns in enumerate(component_columns):
            if columns:
                input_indices[component, :len(columns)] = [input_positions[column] for column in columns]
        source_width = max(max_size, int(input_counts.max(initial=0)))
        win = np.zeros((n, max_size, source_width), dtype=np.float32)
        wrec = np.zeros((n, max_size, max_size), dtype=np.float32)
        bias = np.zeros((n, max_size), dtype=np.float32)
        actual_edges = []
        radius_estimates = []
        scaled_radii = []
        for component, size_value in enumerate(self._sizes):
            size = int(size_value)
            source_component = int(self._source_components[component])
            source_size = int(self._sizes[source_component]) if source_component >= 0 else int(input_counts[component])
            input_scale = self.config.input_scalings[component]
            bias_scale = self.config.bias_scalings[component]
            bias[component, :size] = rng.uniform(-bias_scale, bias_scale, size=size)
            if source_size:
                win[component, :size, :source_size] = rng.uniform(
                    -input_scale, input_scale, size=(size, source_size),
                )
            topology = self.config.topologies[component]
            degree = min(int(self.config.sparse_degrees[component]), size)
            if topology == "ring":
                for row in range(size):
                    wrec[component, row, (row - 1) % size] = rng.normal()
                    wrec[component, row, (row + 1) % size] = rng.normal()
            elif topology == "sparse":
                for row in range(size):
                    columns = rng.choice(size, size=degree, replace=False)
                    wrec[component, row, columns] = rng.normal(size=degree)
            elif topology == "block_sparse":
                for start in range(0, size, degree):
                    end = min(size, start + degree)
                    wrec[component, start:end, start:end] = rng.normal(size=(end - start, end - start))
            elif topology == "orthogonal":
                orthogonal, _ = np.linalg.qr(rng.normal(size=(size, size)))
                wrec[component, :size, :size] = orthogonal.astype(np.float32)
            else:
                wrec[component, :size, :size] = rng.normal(size=(size, size)).astype(np.float32)
            if topology == "orthogonal":
                radius = 1.0
            else:
                matrix = wrec[component, :size, :size].astype(np.float64)
                if size <= 3 or topology == "dense" or (topology == "ring" and size <= 32):
                    radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
                else:
                    from scipy.sparse import csr_matrix
                    from scipy.sparse.linalg import ArpackNoConvergence, eigs
                    start_vector = np.linspace(1.0, 2.0, size, dtype=np.float64)
                    try:
                        eigenvalues = eigs(
                            csr_matrix(matrix), k=1, which="LM", return_eigenvectors=False,
                            v0=start_vector, tol=1e-5, maxiter=max(1000, 10 * size),
                        )
                    except ArpackNoConvergence as exc:
                        eigenvalues = exc.eigenvalues
                        if eigenvalues is None or not len(eigenvalues):
                            raise ValueError("sparse recurrent spectral-radius estimate did not converge") from exc
                    radius = float(np.max(np.abs(eigenvalues)))
            if not np.isfinite(radius) or radius <= np.finfo(np.float64).eps:
                raise ValueError("recurrent matrix has invalid spectral radius")
            target_radius = float(self.config.spectral_radii[component])
            wrec[component, :size, :size] *= np.float32(target_radius / radius)
            radius_estimates.append(radius)
            scaled_radii.append(target_radius)
            actual_edges.append(int(np.count_nonzero(wrec[component, :size, :size])))

        max_degree = max(int(np.count_nonzero(wrec[c, row, :self._sizes[c]]))
                         for c in range(n) for row in range(int(self._sizes[c])))
        rec_indices = np.full((n, max_size, max_degree), -1, dtype=np.int32)
        rec_values = np.zeros((n, max_size, max_degree), dtype=np.float32)
        for component, size_value in enumerate(self._sizes):
            for row in range(int(size_value)):
                columns = np.flatnonzero(wrec[component, row, :int(size_value)])
                rec_indices[component, row, :len(columns)] = columns
                rec_values[component, row, :len(columns)] = wrec[component, row, columns]
        self.actual_recurrent_edges = tuple(actual_edges)
        self.recurrent_radius_estimates = tuple(radius_estimates)
        self.scaled_spectral_radii = tuple(scaled_radii)
        self.input_columns_by_component = tuple(tuple(columns) for columns in component_columns)
        return win, rec_indices, rec_values, bias, input_indices, input_counts

    @property
    def ready(self):
        return self.steps_since_reset > self.warmup_steps

    @property
    def feature_names(self):
        return tuple(
            f"esn_r{reservoir}_state_{unit:03d}"
            for reservoir in range(self._state_count)
            for unit in range(int(self._sizes[reservoir]))
        )

    def get_features(self):
        if not self.ready:
            return None
        return np.concatenate([self.states[index, :int(size)] for index, size in enumerate(self._sizes)])

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
            self._bias, np.asarray(self.config.leak_rates, dtype=np.float32), self._sizes,
            self._source_components, self._input_indices, self._input_counts,
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
            return np.empty((0, self._total_units), dtype=np.float32), np.empty(0, dtype=bool)
        scaled = self.scaler.transform(x)
        timestamp_ns = ts.as_unit("ns").asi8
        states_out = np.full((len(ts), self._total_units), np.nan, dtype=np.float32)
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
                self._rec_values, self._bias, np.asarray(self.config.leak_rates, dtype=np.float32),
                self._sizes, self._source_components, self._input_indices, self._input_counts,
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
            "config": {
                **asdict(self.config),
                "input_groups": list(self.config.input_groups),
                "component_sizes": None if self.config.component_sizes is None else list(self.config.component_sizes),
                "component_input_groups": None if self.config.component_input_groups is None else [list(groups) for groups in self.config.component_input_groups],
                "component_topologies": None if self.config.component_topologies is None else list(self.config.component_topologies),
                "component_leak_rates": None if self.config.component_leak_rates is None else list(self.config.component_leak_rates),
                "component_spectral_radii": None if self.config.component_spectral_radii is None else list(self.config.component_spectral_radii),
                "component_input_scalings": None if self.config.component_input_scalings is None else list(self.config.component_input_scalings),
                "component_bias_scalings": None if self.config.component_bias_scalings is None else list(self.config.component_bias_scalings),
                "component_sparse_degrees": None if self.config.component_sparse_degrees is None else list(self.config.component_sparse_degrees),
            },
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
        for name in ("component_sizes", "component_topologies", "component_leak_rates",
                     "component_spectral_radii", "component_input_scalings",
                     "component_bias_scalings", "component_sparse_degrees"):
            if config_data.get(name) is not None:
                config_data[name] = tuple(config_data[name])
        if config_data.get("component_input_groups") is not None:
            config_data["component_input_groups"] = tuple(tuple(groups) for groups in config_data["component_input_groups"])
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
                engine._rec_values, engine._bias, np.asarray(config.leak_rates, dtype=np.float32),
                engine._sizes, engine._source_components, engine._input_indices, engine._input_counts,
            )
            perturbed = _advance(
                perturbed, normalized[row], engine._win, engine._rec_indices,
                engine._rec_values, engine._bias, np.asarray(config.leak_rates, dtype=np.float32),
                engine._sizes, engine._source_components, engine._input_indices, engine._input_counts,
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
