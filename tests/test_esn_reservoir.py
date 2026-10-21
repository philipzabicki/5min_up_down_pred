from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from features.esn_reservoir import (
    ESNConfig,
    ESNEngine,
    FeatureScaler,
    FrozenProjection,
    INPUT_COLUMNS,
    build_causal_esn_inputs,
    build_nvar_features,
    estimate_warmup,
)


class ESNReservoirTests(unittest.TestCase):
    def setUp(self):
        self.timestamps = pd.date_range("2026-01-01", periods=12, freq="min", tz="UTC")
        self.x = np.random.default_rng(19).normal(size=(12, len(INPUT_COLUMNS)))
        self.columns = INPUT_COLUMNS
        self.scaler = FeatureScaler(self.columns).fit(self.x[:6])

    def _engine(self, architecture="single", topology="sparse", count=1, seed=7):
        config = ESNConfig(
            architecture=architecture,
            reservoir_size=12,
            reservoir_count=count,
            topology=topology,
            connectivity=0.25,
            input_groups=tuple(dict.fromkeys(
                group for group in ("returns", "candle", "range_volatility", "volume", "basis", "time")
            )),
        )
        return config, ESNEngine(config, seed, self.scaler, warmup_steps=2)

    def test_batch_and_stream_match_for_each_architecture_and_topology(self):
        for architecture, count in (("single", 1), ("parallel", 3), ("deep", 3)):
            for topology in ("dense", "sparse", "ring"):
                config, batch = self._engine(architecture, topology, count)
                batch_states, batch_ready = batch.update_many(self.timestamps, self.x)
                _, stream = self._engine(architecture, topology, count)
                stream_states = []
                stream_ready = []
                for timestamp, row in zip(self.timestamps, self.x):
                    value = stream.update_one(timestamp, row)
                    stream_ready.append(value is not None)
                    stream_states.append(stream.states.reshape(-1).copy())
                np.testing.assert_allclose(batch_states, np.asarray(stream_states), rtol=0.0, atol=1e-7)
                np.testing.assert_array_equal(batch_ready, np.asarray(stream_ready))

    def test_deep_layer_uses_the_previous_layer_state(self):
        config = ESNConfig(
            architecture="deep", reservoir_size=8, reservoir_count=2,
            topology="ring", input_groups=("returns",),
        )
        scaler = FeatureScaler(config.selected_columns).fit(self.x[:6, :4])
        coupled = ESNEngine(config, 41, scaler, warmup_steps=0)
        disconnected = ESNEngine(config, 41, scaler, warmup_steps=0)
        disconnected._win[1].fill(0.0)
        disconnected._bias[1].fill(0.0)
        coupled_state, _ = coupled.update_many(self.timestamps[:1], self.x[:1, :4])
        disconnected_state, _ = disconnected.update_many(self.timestamps[:1], self.x[:1, :4])
        np.testing.assert_array_equal(coupled_state[0, :8], disconnected_state[0, :8])
        self.assertGreater(float(np.linalg.norm(coupled_state[0, 8:] - disconnected_state[0, 8:])), 1e-6)

    def test_parallel_reservoirs_have_separate_states_without_cross_talk(self):
        config = ESNConfig(
            architecture="parallel", reservoir_size=8, reservoir_count=2,
            topology="sparse", connectivity=0.4, input_groups=("returns",),
        )
        scaler = FeatureScaler(config.selected_columns).fit(self.x[:6, :4])
        left = ESNEngine(config, 43, scaler, warmup_steps=0)
        right = ESNEngine(config, 43, scaler, warmup_steps=0)
        left.update_many(self.timestamps[:1], self.x[:1, :4])
        right.update_many(self.timestamps[:1], self.x[:1, :4])
        self.assertFalse(np.array_equal(left.states[0], left.states[1]))
        self.assertFalse(np.array_equal(left._rec_values[0], left._rec_values[1]))
        right.states[0, 0] += 0.25
        left_state, _ = left.update_many(self.timestamps[1:2], self.x[1:2, :4])
        right_state, _ = right.update_many(self.timestamps[1:2], self.x[1:2, :4])
        np.testing.assert_array_equal(left_state[0, 8:], right_state[0, 8:])
        self.assertFalse(np.array_equal(left_state[0, :8], right_state[0, :8]))

    def test_architecture_and_dynamics_parameters_change_computation(self):
        common = dict(reservoir_size=8, topology="sparse", connectivity=0.4, input_groups=("returns",))
        single = ESNConfig(architecture="single", reservoir_count=1, **common)
        parallel = ESNConfig(architecture="parallel", reservoir_count=2, **common)
        deep = ESNConfig(architecture="deep", reservoir_count=2, **common)
        self.assertEqual(single.state_count * single.reservoir_size, 8)
        self.assertEqual(parallel.state_count * parallel.reservoir_size, 16)
        self.assertEqual(deep.state_count * deep.reservoir_size, 16)

        low_radius = ESNConfig(architecture="single", reservoir_count=1, spectral_radius=0.1, **common)
        high_radius = ESNConfig(architecture="single", reservoir_count=1, spectral_radius=0.9, **common)
        scaler = FeatureScaler(low_radius.selected_columns).fit(self.x[:6, :4])
        low = ESNEngine(low_radius, 47, scaler, warmup_steps=0)
        high = ESNEngine(high_radius, 47, scaler, warmup_steps=0)
        low_states, _ = low.update_many(self.timestamps[:4], self.x[:4, :4])
        high_states, _ = high.update_many(self.timestamps[:4], self.x[:4, :4])
        self.assertFalse(np.allclose(low_states[-1], high_states[-1]))

    def test_ring_structure_is_seed_invariant_but_weights_are_seeded(self):
        config = ESNConfig(
            reservoir_size=12, topology="ring", input_groups=("returns",),
        )
        scaler = FeatureScaler(config.selected_columns).fit(self.x[:6, :4])
        first = ESNEngine(config, 53, scaler, warmup_steps=0)
        repeat = ESNEngine(config, 53, scaler, warmup_steps=0)
        other_seed = ESNEngine(config, 59, scaler, warmup_steps=0)
        np.testing.assert_array_equal(first._rec_indices, repeat._rec_indices)
        np.testing.assert_array_equal(first._rec_values, repeat._rec_values)
        np.testing.assert_array_equal(first._win, repeat._win)
        np.testing.assert_array_equal(first._bias, repeat._bias)
        np.testing.assert_array_equal(first._rec_indices, other_seed._rec_indices)
        self.assertFalse(np.array_equal(first._rec_values, other_seed._rec_values))
        self.assertFalse(np.array_equal(first._win, other_seed._win))
        self.assertFalse(np.array_equal(first._bias, other_seed._bias))
        for row in range(config.reservoir_size):
            connected = first._rec_indices[0, row]
            connected = connected[connected >= 0]
            self.assertEqual(set(connected.tolist()), {(row - 1) % 12, (row + 1) % 12})

    def test_sparse_connectivity_changes_the_recurrent_graph(self):
        low = ESNConfig(
            reservoir_size=24, topology="sparse", connectivity=0.05,
            input_groups=("returns",),
        )
        high = ESNConfig(
            reservoir_size=24, topology="sparse", connectivity=0.8,
            input_groups=("returns",),
        )
        scaler = FeatureScaler(low.selected_columns).fit(self.x[:6, :4])
        low_engine = ESNEngine(low, 61, scaler, warmup_steps=0)
        high_engine = ESNEngine(high, 61, scaler, warmup_steps=0)
        low_edges = int(np.count_nonzero(low_engine._rec_indices >= 0))
        high_edges = int(np.count_nonzero(high_engine._rec_indices >= 0))
        self.assertGreater(high_edges, low_edges)

    def test_chunk_boundary_and_gap_reset_match_stream(self):
        timestamps = pd.DatetimeIndex([
            self.timestamps[i] if i < 6 else self.timestamps[i] + pd.Timedelta(minutes=1)
            for i in range(len(self.timestamps))
        ])
        _, batch = self._engine()
        left, left_ready = batch.update_many(timestamps[:5], self.x[:5])
        right, right_ready = batch.update_many(timestamps[5:], self.x[5:])
        _, single = self._engine()
        full, ready = single.update_many(timestamps, self.x)
        np.testing.assert_allclose(np.vstack((left, right)), full, rtol=0.0, atol=1e-7)
        np.testing.assert_array_equal(np.r_[left_ready, right_ready], ready)

    def test_gap_resets_state_and_duplicate_or_unordered_timestamps_raise(self):
        times = pd.DatetimeIndex([self.timestamps[0], self.timestamps[1], self.timestamps[3]])
        _, engine = self._engine()
        states, ready = engine.update_many(times, self.x[:3])
        self.assertFalse(ready[2])
        _, fresh = self._engine()
        fresh.update_one(times[2], self.x[2])
        np.testing.assert_allclose(states[2], fresh.states.reshape(-1), rtol=0.0, atol=1e-7)
        with self.assertRaises(ValueError):
            engine.update_many([times[0], times[0]], self.x[:2])
        with self.assertRaises(ValueError):
            engine.update_many([times[1], times[0]], self.x[:2])

    def test_future_rows_do_not_change_earlier_features(self):
        _, full = self._engine()
        all_states, _ = full.update_many(self.timestamps, self.x)
        _, prefix = self._engine()
        prefix_states, _ = prefix.update_many(self.timestamps[:7], self.x[:7])
        np.testing.assert_array_equal(all_states[:7], prefix_states)

    def test_masking_after_state_update_changes_later_state(self):
        _, all_rows = self._engine()
        all_rows.update_many(self.timestamps, self.x)
        _, skipped = self._engine()
        skipped.update_many(self.timestamps[:4], self.x[:4])
        skipped.update_many(self.timestamps[5:], self.x[5:])
        self.assertFalse(np.allclose(all_rows.states, skipped.states))

    def test_invalid_observation_resets_without_synthetic_update(self):
        _, batch = self._engine()
        all_states, ready = batch.update_many(
            self.timestamps[:5], self.x[:5], valid_mask=[True, True, False, True, True],
        )
        self.assertFalse(ready[2])
        self.assertTrue(np.isnan(all_states[2]).all())
        _, fresh = self._engine()
        fresh.update_one(self.timestamps[3], self.x[3])
        np.testing.assert_allclose(all_states[3], fresh.states.reshape(-1), rtol=0.0, atol=1e-7)

    def test_checkpoint_resume_matches_uninterrupted_stream(self):
        config, uninterrupted = self._engine("parallel", "ring", 2, seed=23)
        expected, _ = uninterrupted.update_many(self.timestamps, self.x)
        _, partial = self._engine("parallel", "ring", 2, seed=23)
        first, _ = partial.update_many(self.timestamps[:7], self.x[:7])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            partial.save_checkpoint(path)
            resumed = ESNEngine.load_checkpoint(path, expected_config=config, expected_seed=23)
            rest, _ = resumed.update_many(self.timestamps[7:], self.x[7:])
        np.testing.assert_array_equal(np.vstack((first, rest)), expected)

    def test_checkpoint_rejects_changed_configuration(self):
        config, engine = self._engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            engine.save_checkpoint(path)
            changed = ESNConfig(
                reservoir_size=13,
                input_groups=config.input_groups,
            )
            with self.assertRaises(ValueError):
                ESNEngine.load_checkpoint(path, expected_config=changed)

    def test_saved_feature_bundle_transform_roundtrip(self):
        projection = FrozenProjection(
            mean=[0.2, -0.1, 0.5],
            components=[[0.7, 0.2, -0.1], [-0.3, 0.4, 0.8]],
            explained_variance_ratio=[0.6, 0.3],
        )
        restored_projection = FrozenProjection.from_dict(projection.to_dict())
        restored_scaler = FeatureScaler.from_dict(self.scaler.to_dict())
        sample = np.random.default_rng(24).normal(size=(9, 3))
        np.testing.assert_array_equal(
            projection.transform(sample), restored_projection.transform(sample),
        )
        np.testing.assert_array_equal(
            self.scaler.transform(self.x[:4]), restored_scaler.transform(self.x[:4]),
        )

    def test_warmup_is_measured_and_stops_when_stable(self):
        config = ESNConfig(
            reservoir_size=8, leak_rate=1.0, spectral_radius=0.1,
            input_groups=("returns",),
        )
        columns = config.selected_columns
        values = np.random.default_rng(29).normal(size=(1_000, len(columns)))
        scaler = FeatureScaler(columns).fit(values[:500])
        warmup, diagnostics = estimate_warmup(
            values, config, 17, scaler=scaler,
            timestamps=pd.date_range("2026-01-01", periods=len(values), freq="min", tz="UTC"),
            relative_tolerance=0.01, stable_window=12, max_steps=500,
        )
        self.assertTrue(diagnostics["stabilized"])
        self.assertGreater(warmup, 0)
        self.assertLess(diagnostics["training_steps_examined"], 500)

    def test_causal_input_builder_resets_lags_at_timestamp_gap(self):
        times = pd.DatetimeIndex([
            self.timestamps[0], self.timestamps[1], self.timestamps[4], self.timestamps[5],
        ])
        candles = pd.DataFrame({
            "Opened": times,
            "Open": [100, 101, 102, 103],
            "High": [102, 103, 104, 105],
            "Low": [99, 100, 101, 102],
            "Close": [101, 102, 103, 104],
            "Volume": [10, 11, 12, 13],
            "UM_BTCUSDT_Close": [100.5, 101.5, 102.5, 103.5],
        })
        _, features = build_causal_esn_inputs(candles)
        self.assertTrue(np.isfinite(features.loc[1, "log_return_1m"]))
        self.assertTrue(np.isnan(features.loc[2, "log_return_1m"]))
        self.assertTrue(np.isnan(features.loc[2, "index_futures_basis_change_1m"]))
        self.assertEqual(tuple(features.columns), INPUT_COLUMNS)

    def test_input_builder_rejects_duplicates_and_out_of_order_times(self):
        candles = pd.DataFrame({
            "Opened": [self.timestamps[1], self.timestamps[0]],
            "Open": [1, 1], "High": [2, 2], "Low": [0, 0],
            "Close": [1, 1], "Volume": [1, 1],
        })
        with self.assertRaises(ValueError):
            build_causal_esn_inputs(candles)

    def test_nvar_lags_and_quadratics_respect_gaps(self):
        times = pd.DatetimeIndex([
            self.timestamps[0], self.timestamps[1], self.timestamps[2],
            self.timestamps[5], self.timestamps[6],
        ])
        values = np.arange(15, dtype=np.float32).reshape(5, 3)
        features, names, valid = build_nvar_features(
            values, times, input_columns=("a", "b", "c"), max_lag=2,
        )
        self.assertEqual(features.shape, (5, 15))
        self.assertEqual(len(names), 15)
        np.testing.assert_array_equal(valid, [False, False, True, False, False])
        self.assertTrue(np.isnan(features[3:]).all())

    def test_nvar_invalid_candle_resets_lag_history(self):
        times = pd.date_range("2026-01-01", periods=6, freq="min", tz="UTC")
        values = np.arange(18, dtype=np.float32).reshape(6, 3)
        features, _, valid = build_nvar_features(
            values, times, input_columns=("a", "b", "c"), max_lag=2,
            valid_mask=[True, True, False, True, True, True],
        )
        np.testing.assert_array_equal(valid, [False, False, False, False, False, True])
        self.assertTrue(np.isnan(features[2:5]).all())

    def test_nvar_accepts_read_only_validity_masks_without_mutating_them(self):
        valid = np.ones(len(self.x), dtype=bool)
        valid.setflags(write=False)
        features, _, nvar_valid = build_nvar_features(
            self.x.astype(np.float32), self.timestamps,
            input_columns=self.columns, max_lag=4, valid_mask=valid,
        )
        np.testing.assert_array_equal(valid, np.ones(len(self.x), dtype=bool))
        np.testing.assert_array_equal(nvar_valid[:4], [False, False, False, False])
        self.assertTrue(np.isfinite(features[4:]).all())


if __name__ == "__main__":
    unittest.main()
