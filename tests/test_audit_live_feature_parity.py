import json
import math
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import audit_feature_readiness as audit
import run as live_runtime
from features.ChaikinOsc import get_chaikin_oscillator_values
from features.live_indicator_runtime import (
    ChaikinOscillatorRuntimeState,
    IndicatorFullHistoryScratch,
    IndicatorWindowScratch,
    get_chaikin_oscillator_latest_value_live,
)


class PseudoLiveAuditPredictorTests(unittest.TestCase):
    def test_chaikin_state_continues_batch_feature_across_restart(self):
        count = 12_000
        steps = np.arange(count, dtype=np.float64)
        center = 25_000.0 + 0.4 * steps + 30.0 * np.sin(steps / 47.0)
        ohlcv = np.column_stack(
            (
                center,
                center + 2.0,
                center - 2.0,
                center + np.sin(steps / 13.0),
                2_000.0 + 500.0 * np.sin(steps / 17.0) ** 2,
            )
        )
        params = {"fast_period": 13, "slow_period": 79}
        expected = get_chaikin_oscillator_values(
            {
                **params,
                "fast_ma_type": "EMA",
                "slow_ma_type": "SHMMA",
            },
            ohlcv,
        )
        split = 8_000
        state = ChaikinOscillatorRuntimeState.from_history(
            ohlcv[:split], **params
        )
        state = ChaikinOscillatorRuntimeState.from_dict(state.to_dict())
        actual = np.asarray(
            [state.update(row) for row in ohlcv[split:]],
            dtype=np.float64,
        )

        np.testing.assert_allclose(actual, expected[split:], rtol=1e-12, atol=1e-9)

    def test_live_chaikin_state_seed_loads_and_catches_up_contiguously(self):
        count = 300
        steps = np.arange(count + 2, dtype=np.float64)
        center = 1_000.0 + 0.2 * steps + 2.0 * np.sin(steps / 11.0)
        rows = np.column_stack(
            (
                center,
                center + 1.0,
                center - 1.0,
                center + 0.1 * np.cos(steps / 5.0),
                100.0 + 20.0 * np.sin(steps / 7.0) ** 2,
            )
        )
        seed_count = count
        params = {"fast_period": 13, "slow_period": 79}
        state = ChaikinOscillatorRuntimeState.from_history(
            rows[:seed_count], **params
        )
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        opened = pd.date_range(start, periods=count + 2, freq="min")
        feature_col = "ChaikinOsc_fit_test"
        with tempfile.TemporaryDirectory() as temp_dir:
            seed_path = Path(temp_dir) / "indicator_state.json"
            seed_path.write_text(
                json.dumps({
                    "state_version": 1,
                    "feature_col": feature_col,
                    "params": params,
                    "state_as_of_opened_utc": opened[seed_count - 1].isoformat(),
                    "model_sha256": "candidate-sha",
                    "state": state.to_dict(),
                }),
                encoding="utf-8",
            )
            catchup = pd.DataFrame(rows[seed_count:], columns=live_runtime.OHLCV_COLS)
            catchup.insert(0, "Opened", opened[seed_count:])
            predictor = live_runtime.LivePredictor.__new__(live_runtime.LivePredictor)
            predictor.indicator_specs = [SimpleNamespace(
                indicator="ChaikinOsc",
                feature_col=feature_col,
                params={
                    **params,
                    "fast_ma_type": "EMA",
                    "slow_ma_type": "SHMMA",
                },
            )]
            predictor.indicator_state_by_feature = {}
            predictor.model_hash = "candidate-sha"
            predictor.session = object()
            with mock.patch.object(
                live_runtime, "INDICATOR_STATE_SEED_PATH", seed_path
            ), mock.patch.object(
                live_runtime, "fetch_closed_ohlcv_range", return_value=catchup
            ):
                predictor._initialize_indicator_state(
                    pd.DataFrame({"Opened": [opened[-1]]})
                )

        expected = ChaikinOscillatorRuntimeState.from_history(rows, **params)
        self.assertAlmostEqual(
            predictor.indicator_state_by_feature[feature_col].value,
            expected.value,
            places=9,
        )

    def test_live_chaikin_state_startup_fails_clearly_when_seed_is_missing(self):
        predictor = live_runtime.LivePredictor.__new__(live_runtime.LivePredictor)
        predictor.indicator_specs = [SimpleNamespace(
            indicator="ChaikinOsc",
            feature_col="ChaikinOsc_fit_test",
            params={"fast_ma_type": "EMA", "slow_ma_type": "SHMMA"},
        )]
        with mock.patch.object(live_runtime, "INDICATOR_STATE_SEED_PATH", None):
            with self.assertRaisesRegex(
                RuntimeError,
                "missing artifacts.indicator_state_seed_path.*ChaikinOsc_fit_test",
            ):
                predictor._initialize_indicator_state(
                    pd.DataFrame({"Opened": [pd.Timestamp("2026-01-01T00:00:00Z")]})
                )

    def test_live_feature_parameter_override_requires_expected_fitted_value(self):
        spec = SimpleNamespace(
            feature_col="chaikin_feature",
            params={"slow_ma_type": "GMA", "slow_period": 60},
        )
        requirements = {
            "payload": {
                "live_feature_param_overrides": {
                    "chaikin_feature": {
                        "parameter": "slow_ma_type",
                        "expected": "GMA",
                        "replacement": "SMA",
                    }
                }
            }
        }

        live_runtime.apply_live_feature_parameter_overrides([spec], requirements)

        self.assertEqual(spec.params["slow_ma_type"], "SMA")
        self.assertEqual(spec.params["slow_period"], 60)

        spec.params["slow_ma_type"] = "EMA"
        with self.assertRaisesRegex(ValueError, "does not match the fitted feature"):
            live_runtime.apply_live_feature_parameter_overrides([spec], requirements)

    def test_chaikin_live_feature_matches_full_history_after_semantics_override(self):
        count = 5_000
        steps = np.arange(count, dtype=np.float64)
        center = 1_000.0 + 0.03 * steps + 3.0 * np.sin(steps / 31.0)
        high = center + 2.0
        low = center - 2.0
        close = high - 0.05
        close[:10] = low[:10] + 0.05
        ohlcv = np.column_stack(
            (center, high, low, close, 100.0 + 20.0 * np.sin(steps / 17.0) ** 2)
        )
        original_params = {
            "fast_period": 20,
            "slow_period": 60,
            "fast_ma_type": "T3",
            "slow_ma_type": "GMA",
        }
        stored_value = float(
            get_chaikin_oscillator_values(original_params, ohlcv)[-1]
        )
        scratch = IndicatorWindowScratch(
            IndicatorFullHistoryScratch(ohlcv),
            window_len=1_000,
        )
        current_live_value = get_chaikin_oscillator_latest_value_live(
            original_params,
            scratch,
        )
        spec = SimpleNamespace(
            feature_col="chaikin_feature",
            params=dict(original_params),
        )
        live_runtime.apply_live_feature_parameter_overrides(
            [spec],
            {
                "payload": {
                    "live_feature_param_overrides": {
                        "chaikin_feature": {
                            "parameter": "slow_ma_type",
                            "expected": "GMA",
                            "replacement": "SMA",
                        }
                    }
                }
            },
        )
        fixed_live_value = get_chaikin_oscillator_latest_value_live(
            spec.params,
            scratch,
        )

        self.assertGreater(abs(current_live_value - stored_value), 1.0)
        self.assertLess(abs(fixed_live_value - stored_value), 1e-6)

    def test_indicator_runtime_uses_global_retained_history(self):
        predictor = audit.LivePredictor.__new__(audit.LivePredictor)
        predictor.required_stable_window = 2047
        predictor.ohlcv_np = np.empty((21936, 5), dtype=np.float64)
        predictor.indicator_runtime_window_by_feature = {
            "short_recursive_indicator": 7166,
            "long_recursive_indicator": 18000,
        }

        self.assertEqual(
            predictor._resolve_indicator_window_len("short_recursive_indicator"),
            21936,
        )
        self.assertEqual(
            predictor._resolve_indicator_window_len("long_recursive_indicator"),
            21936,
        )

    def test_restart_from_retained_candles_reproduces_indicators_predictions_and_decisions(self):
        class Spec:
            feature_col = "recursive_indicator"
            params = None

            @staticmethod
            def latest_builder(_params, scratch):
                # A history-dependent feature makes a restart from the same
                # retained candles observable without relying on external state.
                return float(np.mean(scratch.get_adl()[-5:]))

        def predictor_from(frame):
            predictor = audit.LivePredictor.__new__(audit.LivePredictor)
            opened = pd.to_datetime(frame["Opened"], utc=True)
            predictor.opened_candles = deque(pd.Timestamp(value) for value in opened)
            predictor.opened_ns_np = opened.astype("int64").to_numpy()
            predictor.ohlcv_np = frame[["Open", "High", "Low", "Close", "Volume"]].to_numpy(
                dtype=np.float64,
                copy=True,
            )
            predictor.candle_open_close = {
                pd.Timestamp(row.Opened): (float(row.Open), float(row.Close))
                for row in frame.itertuples(index=False)
            }
            predictor.max_keep = 8
            predictor.required_stable_window = 8
            predictor.indicator_runtime_window_by_feature = {"recursive_indicator": 2}
            predictor.indicator_specs = [Spec()]
            predictor.feature_columns = ["Close", "recursive_indicator"]
            predictor.candle_derived_feature_columns = ()
            predictor.candle_pattern_feature_columns = ()
            predictor.streak_interval_to_rule = {}
            predictor.session_feature_columns = ()
            predictor.realized_volatility_state = None
            predictor.latest_realized_volatility_values = {}
            predictor.basis_premium_feature_columns = ()
            predictor.last_indicator_nan_cols = []
            return predictor

        opened = pd.date_range("2026-01-01T00:00:00Z", periods=12, freq="min")
        close = np.linspace(100.0, 111.0, len(opened))
        history = pd.DataFrame(
            {
                "Opened": opened,
                "Open": close - 0.2,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
                "Volume": np.linspace(10.0, 21.0, len(opened)),
            }
        )

        continuous = predictor_from(history.iloc[:8].copy())
        for row in history.iloc[8:10].itertuples(index=False):
            continuous._append_new_candle(
                row.Opened,
                (row.Open, row.High, row.Low, row.Close, row.Volume),
            )

        restarted = predictor_from(
            pd.DataFrame(
                {
                    "Opened": list(continuous.opened_candles),
                    "Open": continuous.ohlcv_np[:, 0],
                    "High": continuous.ohlcv_np[:, 1],
                    "Low": continuous.ohlcv_np[:, 2],
                    "Close": continuous.ohlcv_np[:, 3],
                    "Volume": continuous.ohlcv_np[:, 4],
                }
            )
        )

        for row in history.iloc[10:].itertuples(index=False):
            candle = (row.Open, row.High, row.Low, row.Close, row.Volume)
            continuous._append_new_candle(row.Opened, candle)
            restarted._append_new_candle(row.Opened, candle)
            live_vector = continuous._build_feature_vector()
            restart_vector = restarted._build_feature_vector()
            np.testing.assert_array_equal(live_vector, restart_vector)

            live_probability = 1.0 / (1.0 + np.exp(-live_vector[0, 1]))
            restart_probability = 1.0 / (1.0 + np.exp(-restart_vector[0, 1]))
            self.assertEqual(live_probability, restart_probability)
            self.assertEqual(live_probability >= 0.5, restart_probability >= 0.5)

    def test_rest_catchup_rejects_missing_candle_without_mutating_runtime_state(self):
        predictor = audit.LivePredictor.__new__(audit.LivePredictor)
        last_opened = pd.Timestamp("2026-01-01T00:00:00Z")
        predictor.opened_candles = deque([last_opened])
        predictor.ohlcv_np = np.array([[100.0, 101.0, 99.0, 100.5, 10.0]])
        predictor.opened_ns_np = np.array([last_opened.value], dtype=np.int64)
        predictor.candle_open_close = {last_opened: (100.0, 100.5)}
        predictor.basis_premium_feature_columns = ()
        predictor.realized_volatility_state = None
        predictor.max_keep = 10
        predictor.session = object()
        predictor.volume_profile_enabled = False
        predictor.reaction_profile_enabled = False
        predictor._save_runtime_volume_profile_state_async = lambda: None
        predictor._save_runtime_reaction_profile_state_async = lambda: None

        missing_row = pd.DataFrame(
            {
                "Opened": pd.to_datetime(
                    ["2026-01-01T00:01:00Z", "2026-01-01T00:03:00Z"]
                ),
                "Open": [100.5, 100.7],
                "High": [101.0, 101.2],
                "Low": [100.0, 100.2],
                "Close": [100.6, 100.8],
                "Volume": [11.0, 13.0],
            }
        )

        with mock.patch.object(
                live_runtime,
                "fetch_closed_ohlcv_range",
                return_value=missing_row,
        ):
            with self.assertRaisesRegex(RuntimeError, "missing or out-of-order candle"):
                predictor._sync_closed_candles_from_rest(
                    stop_before_opened=pd.Timestamp("2026-01-01T00:04:00Z")
                )

        self.assertEqual(list(predictor.opened_candles), [last_opened])
        self.assertEqual(predictor.ohlcv_np.shape, (1, 5))

    def test_rest_catchup_appends_a_complete_gap_in_order(self):
        predictor = audit.LivePredictor.__new__(audit.LivePredictor)
        last_opened = pd.Timestamp("2026-01-01T00:00:00Z")
        predictor.opened_candles = deque([last_opened])
        predictor.ohlcv_np = np.array([[100.0, 101.0, 99.0, 100.5, 10.0]])
        predictor.opened_ns_np = np.array([last_opened.value], dtype=np.int64)
        predictor.candle_open_close = {last_opened: (100.0, 100.5)}
        predictor.basis_premium_feature_columns = ()
        predictor.realized_volatility_state = None
        predictor.max_keep = 10
        predictor.session = object()
        predictor.volume_profile_enabled = False
        predictor.reaction_profile_enabled = False
        predictor._save_runtime_volume_profile_state_async = lambda: None
        predictor._save_runtime_reaction_profile_state_async = lambda: None
        catchup = pd.DataFrame(
            {
                "Opened": pd.date_range(
                    "2026-01-01T00:01:00Z",
                    periods=3,
                    freq="min",
                ),
                "Open": [100.5, 100.6, 100.7],
                "High": [101.0, 101.1, 101.2],
                "Low": [100.0, 100.1, 100.2],
                "Close": [100.6, 100.7, 100.8],
                "Volume": [11.0, 12.0, 13.0],
            }
        )

        with mock.patch.object(
                live_runtime,
                "fetch_closed_ohlcv_range",
                return_value=catchup,
        ):
            added = predictor._sync_closed_candles_from_rest(
                stop_before_opened=pd.Timestamp("2026-01-01T00:04:00Z")
            )

        self.assertEqual(added, 3)
        self.assertEqual(
            list(predictor.opened_candles),
            [last_opened, *list(catchup.Opened)],
        )
        self.assertEqual(predictor.ohlcv_np.shape, (4, 5))

    def test_matrix_audit_flags_small_finite_indicator_drift(self):
        class Model:
            @staticmethod
            def predict(matrix):
                return np.clip(matrix[:, 0], 0.0, 1.0)

        audit_df = pd.DataFrame(
            {
                "Opened": pd.date_range(
                    "2026-01-01 00:00:00", periods=2, freq="min", tz="UTC"
                ),
                "Open": [0.5, 0.5],
                "High": [0.5, 0.5],
                "Low": [0.5, 0.5],
                "Close": [0.5, 0.5],
                "Volume": [1.0, 1.0],
            }
        )
        feature_builder_frame = pd.DataFrame(
            {
                "feature": ["indicator_macd"],
                "builder_family": ["indicator"],
                "builder_name": ["MACD"],
                "builder_source": ["fixture"],
            }
        )

        report = audit.build_matrix_comparison_report(
            candidate_label="live",
            reference_label="stored",
            candidate_matrix=np.array([[0.5], [0.5001]]),
            reference_matrix=np.array([[0.5], [0.5]]),
            audit_df=audit_df,
            feature_columns=["indicator_macd"],
            feature_group_by_name={"indicator_macd": "indicator"},
            feature_builder_frame=feature_builder_frame,
            model=Model(),
        )

        self.assertEqual(report["summary"]["rows_with_proba_diff_gt_tol"], 1)
        self.assertEqual(report["summary"]["rows_with_signal_mismatch"], 0)
        self.assertEqual(report["summary"]["feature_parity"]["status"], "measured")
        self.assertEqual(report["summary"]["prediction_parity"]["status"], "measured")
        self.assertEqual(
            report["summary"]["decision_parity"]["status"],
            "not_verified_missing_quotes",
        )
        self.assertEqual(report["summary"]["decision_parity"]["verified_rows"], 0)
        drift = report["step_summary_df"].iloc[1]
        self.assertEqual(drift["worst_feature"], "indicator_macd")
        self.assertGreater(drift["proba_up_abs_diff"], audit.PREDICTION_DIFF_TOL)

    def test_unclassified_model_feature_fails_before_prediction(self):
        bootstrap_df = pd.DataFrame(
            {
                "Opened": pd.date_range(
                    "2026-01-01 00:00:00", periods=2, freq="min", tz="UTC"
                ),
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.5, 101.5],
                "Volume": [10.0, 11.0],
            }
        )
        feature_columns = ["Close", "Open", "not_available_live"]
        meta = {
            "feature_columns": feature_columns,
            "target_col": "target_5m_candle_up",
        }
        requirements = {
            "global_required_runtime_window": 1,
            "global_required_stable_window": 1,
            "stable_window_by_feature": {},
            "runtime_window_by_feature": {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            requirements_path = Path(tmpdir) / "requirements.json"
            requirements_path.write_text(
                json.dumps({"unstable_features": []}), encoding="utf-8"
            )
            with (
                mock.patch.object(
                    audit,
                    "INDICATOR_HISTORY_REQUIREMENTS_PATH",
                    requirements_path,
                ),
                mock.patch.object(
                    audit,
                    "load_model_and_meta",
                    return_value=(object(), meta),
                ),
                mock.patch.object(
                    audit,
                    "load_trade_policy_runtime_config",
                    return_value={},
                ),
                mock.patch.object(audit, "load_indicator_specs", return_value=[]),
                mock.patch.object(
                    audit,
                    "load_indicator_history_requirements",
                    return_value=requirements,
                ),
            ):
                with self.assertRaisesRegex(
                        ValueError,
                        "no live feature family.*not_available_live",
                ):
                    audit.PseudoLiveAuditPredictor(
                        bootstrap_df,
                        model_meta_path="unused.json",
                        max_keep=10,
                    )

    def test_basis_premium_features_are_replayed_from_futures_close(self):
        bootstrap_df = pd.DataFrame(
            {
                "Opened": pd.date_range(
                    "2026-01-01 00:00:00",
                    periods=2,
                    freq="min",
                    tz="UTC",
                ),
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.5, 101.5],
                "Volume": [10.0, 11.0],
                "UM_BTCUSDT_Close": [100.8, 101.7],
            }
        )
        meta = {
            "feature_columns": [
                "Open",
                "Close",
                "futures_index_basis_rel_1m",
            ],
            "target_col": "target_5m_candle_up",
        }
        requirements = {
            "global_required_runtime_window": 1,
            "global_required_stable_window": 1,
            "stable_window_by_feature": {},
            "runtime_window_by_feature": {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            requirements_path = Path(tmpdir) / "requirements.json"
            requirements_path.write_text(
                json.dumps({"unstable_features": []}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    audit,
                    "INDICATOR_HISTORY_REQUIREMENTS_PATH",
                    requirements_path,
                ),
                mock.patch.object(
                    audit,
                    "load_model_and_meta",
                    return_value=(object(), meta),
                ),
                mock.patch.object(
                    audit,
                    "load_trade_policy_runtime_config",
                    return_value={},
                ),
                mock.patch.object(audit, "load_indicator_specs", return_value=[]),
                mock.patch.object(
                    audit,
                    "load_indicator_history_requirements",
                    return_value=requirements,
                ),
            ):
                predictor = audit.PseudoLiveAuditPredictor(
                    bootstrap_df,
                    model_meta_path="unused.json",
                    max_keep=10,
                )

        self.assertIn("futures_index_basis_rel_1m", predictor.feature_columns)
        self.assertEqual(
            predictor.basis_premium_feature_columns,
            ("futures_index_basis_rel_1m",),
        )

        predictor._append_new_candle(
            pd.Timestamp("2026-01-01 00:02:00", tz="UTC"),
            (102.0, 103.0, 101.0, 102.5, 12.0),
            basis_futures_close=102.8,
        )

        self.assertIsNotNone(predictor.basis_futures_close_np)
        self.assertEqual(len(predictor.opened_candles), 3)
        self.assertEqual(len(predictor.basis_futures_close_np), 3)

    def test_reaction_profile_features_are_replayed_into_snapshot(self):
        bootstrap_df = pd.DataFrame(
            {
                "Opened": pd.date_range(
                    "2026-01-01 00:00:00",
                    periods=2,
                    freq="min",
                    tz="UTC",
                ),
                "Open": [100.0, 101.0],
                "High": [101.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [100.5, 102.5],
                "Volume": [10.0, 11.0],
            }
        )
        rp_cfg = {
            "enabled": True,
            "price_min": 0.0,
            "price_max": 200.0,
            "bin_size": 1.0,
            "neighbor_bins": 3.0,
            "eps": 1e-12,
            "min_reaction_strength": 0.0,
            "wick_power": 1.0,
            "distance_power": 1.0,
            "horizons": {
                "short": {"local_window": 8, "half_life_candles": 2},
                "medium": {"local_window": 8, "half_life_candles": 4},
                "long": {"local_window": 8, "half_life_candles": 8},
                "all": {"local_window": 8, "half_life_candles": None},
            },
        }
        meta = {
            "feature_columns": ["rp_short_support_below"],
            "target_col": "target_5m_candle_up",
            "reaction_profile_fixed_grid": rp_cfg,
        }
        requirements = {
            "global_required_runtime_window": 1,
            "global_required_stable_window": 1,
            "stable_window_by_feature": {},
            "runtime_window_by_feature": {},
        }
        active_settings = {
            **audit.MODELING_DATASET_SETTINGS,
            "volume_profile_fixed_range": {"enabled": False},
            "reaction_profile_fixed_grid": rp_cfg,
            "basis_premium_features": {"enabled": False},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            requirements_path = Path(tmpdir) / "requirements.json"
            requirements_path.write_text(
                json.dumps({"unstable_features": []}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(audit, "MODELING_DATASET_SETTINGS", active_settings),
                mock.patch.object(
                    audit,
                    "INDICATOR_HISTORY_REQUIREMENTS_PATH",
                    requirements_path,
                ),
                mock.patch.object(
                    audit,
                    "load_model_and_meta",
                    return_value=(object(), meta),
                ),
                mock.patch.object(
                    audit,
                    "load_trade_policy_runtime_config",
                    return_value={},
                ),
                mock.patch.object(audit, "load_indicator_specs", return_value=[]),
                mock.patch.object(
                    audit,
                    "load_indicator_history_requirements",
                    return_value=requirements,
                ),
            ):
                predictor = audit.PseudoLiveAuditPredictor(
                    bootstrap_df,
                    model_meta_path="unused.json",
                    max_keep=10,
                )

        predictor._append_new_candle(
            pd.Timestamp("2026-01-01 00:02:00", tz="UTC"),
            (102.0, 104.0, 101.0, 103.5, 12.0),
        )
        reaction_values = predictor._prepare_reaction_profile_features_for_latest_candle(
            pd.Timestamp("2026-01-01 00:02:00", tz="UTC"),
        )
        snapshot = predictor.build_feature_snapshot(
            reaction_profile_values=reaction_values,
        )

        self.assertIn("rp_short_support_below", reaction_values)
        self.assertTrue(math.isfinite(float(snapshot["vector"][0, 0])))
        self.assertEqual(snapshot["nonfinite_feature_indices"], ())


class LiveFeatureParityOutputTests(unittest.TestCase):
    def test_live_source_prefers_rest_replay_frame(self):
        expected_opened = pd.date_range(
            "2026-01-01 00:00:00",
            periods=2,
            freq="min",
            tz="UTC",
        )
        rest_frame = pd.DataFrame(
            {
                "Opened": expected_opened,
                "Open": [10.0, 11.0],
                "High": [12.0, 13.0],
                "Low": [9.0, 10.0],
                "Close": [11.0, 12.0],
                "Volume": [100.0, 101.0],
                "UM_BTCUSDT_Close": [11.2, 12.2],
            }
        )
        audit_window = audit.AuditWindow(
            bootstrap_start=expected_opened[0],
            audit_start=expected_opened[1],
            audit_end=expected_opened[-1],
            bootstrap_rows=1,
            audit_rows=1,
            requested_days_back=1,
            max_steps=1,
        )

        with mock.patch.object(
                audit,
                "fetch_live_closed_ohlcv_range",
                return_value=rest_frame,
        ):
            frame, metadata = audit.load_live_source_audit_frame(
                audit_window=audit_window,
                expected_opened=expected_opened,
                auxiliary_columns=("UM_BTCUSDT_Close",),
                use_rest=True,
            )

        self.assertEqual(metadata["live_ohlcv_source"], "live_rest_api")
        self.assertEqual(frame["Open"].tolist(), [10.0, 11.0])
        self.assertEqual(frame["UM_BTCUSDT_Close"].tolist(), [11.2, 12.2])

    def test_live_source_falls_back_to_raw_csv_aligned_by_opened(self):
        expected_opened = pd.date_range(
            "2026-01-01 00:00:00",
            periods=2,
            freq="min",
            tz="UTC",
        )
        audit_window = audit.AuditWindow(
            bootstrap_start=expected_opened[0],
            audit_start=expected_opened[1],
            audit_end=expected_opened[-1],
            bootstrap_rows=1,
            audit_rows=1,
            requested_days_back=1,
            max_steps=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw.csv"
            pd.DataFrame(
                {
                    "Opened": [
                        "2026-01-01 00:01:00",
                        "2026-01-01 00:00:00",
                    ],
                    "Open": [21.0, 20.0],
                    "High": [23.0, 22.0],
                    "Low": [19.0, 18.0],
                    "Close": [22.0, 21.0],
                    "Volume": [201.0, 200.0],
                    "UM_BTCUSDT_Close": [22.2, 21.2],
                }
            ).to_csv(raw_path, index=False)
            active_settings = {
                **audit.MODELING_DATASET_SETTINGS,
                "raw_data_dir": Path(tmpdir),
                "base_data_file": raw_path.name,
            }

            with (
                mock.patch.object(audit, "MODELING_DATASET_SETTINGS", active_settings),
                mock.patch.object(
                    audit,
                    "fetch_live_closed_ohlcv_range",
                    side_effect=RuntimeError("REST unavailable"),
                ),
            ):
                frame, metadata = audit.load_live_source_audit_frame(
                    audit_window=audit_window,
                    expected_opened=expected_opened,
                    auxiliary_columns=("UM_BTCUSDT_Close",),
                    use_rest=True,
                )

        self.assertEqual(
            metadata["live_ohlcv_source"],
            "raw_csv_fallback_after_rest_failure",
        )
        self.assertEqual(frame["Open"].tolist(), [20.0, 21.0])
        self.assertEqual(frame["UM_BTCUSDT_Close"].tolist(), [21.2, 22.2])

    def test_raw_auxiliary_columns_are_aligned_by_opened(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw.csv"
            pd.DataFrame(
                {
                    "Opened": [
                        "2026-01-01 00:02:00",
                        "2026-01-01 00:00:00",
                        "2026-01-01 00:01:00",
                    ],
                    "UM_BTCUSDT_Close": [102.8, 100.8, 101.7],
                }
            ).to_csv(raw_path, index=False)
            frame = pd.DataFrame(
                {
                    "Opened": pd.to_datetime(
                        ["2026-01-01 00:01:00", "2026-01-01 00:02:00"],
                        utc=True,
                    ),
                    "Close": [101.5, 102.5],
                }
            )
            active_settings = {
                **audit.MODELING_DATASET_SETTINGS,
                "raw_data_dir": Path(tmpdir),
                "base_data_file": raw_path.name,
            }

            with mock.patch.object(
                    audit,
                    "MODELING_DATASET_SETTINGS",
                    active_settings,
            ):
                result = audit._merge_raw_auxiliary_columns(
                    frame,
                    ("UM_BTCUSDT_Close",),
                )

        self.assertEqual(result["UM_BTCUSDT_Close"].tolist(), [101.7, 102.8])

    def test_artifact_validation_rejects_stale_dataset_precision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "dataset.parquet"
            metadata_path = Path(tmpdir) / "dataset_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "float_precision": "float32",
                        "parquet_path": str(parquet_path),
                    }
                ),
                encoding="utf-8",
            )
            model_meta = {
                "data_path": str(parquet_path),
                "numeric_precision": {
                    "configured_float_precision": "float64",
                    "parquet_float_columns": "float32",
                },
            }
            active_settings = {
                **audit.MODELING_DATASET_SETTINGS,
                "float_precision": "float64",
            }

            with (
                mock.patch.object(audit, "MODELING_DATASET_SETTINGS", active_settings),
                self.assertRaisesRegex(ValueError, "dataset artifact is stale"),
            ):
                audit.validate_modeling_artifacts_for_audit(
                    parquet_path=parquet_path,
                    model_meta_path=Path(tmpdir) / "model_meta.json",
                    model_meta=model_meta,
                )

    def test_artifact_validation_accepts_matching_precision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "dataset.parquet"
            metadata_path = Path(tmpdir) / "dataset_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "float_precision": "float64",
                        "parquet_path": str(parquet_path),
                    }
                ),
                encoding="utf-8",
            )
            model_meta = {
                "data_path": str(parquet_path),
                "numeric_precision": {
                    "configured_float_precision": "float64",
                    "parquet_float_columns": "float64",
                },
            }
            active_settings = {
                **audit.MODELING_DATASET_SETTINGS,
                "float_precision": "float64",
            }

            with mock.patch.object(
                    audit,
                    "MODELING_DATASET_SETTINGS",
                    active_settings,
            ):
                result = audit.validate_modeling_artifacts_for_audit(
                    parquet_path=parquet_path,
                    model_meta_path=Path(tmpdir) / "model_meta.json",
                    model_meta=model_meta,
                )

        self.assertEqual(result["active_float_precision"], "float64")
        self.assertEqual(result["dataset_float_precision"], "float64")

    def test_artifact_validation_rejects_model_dataset_precision_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "dataset.parquet"
            metadata_path = Path(tmpdir) / "dataset_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "float_precision": "float64",
                        "parquet_path": str(parquet_path),
                    }
                ),
                encoding="utf-8",
            )
            model_meta = {
                "data_path": str(parquet_path),
                "numeric_precision": {
                    "configured_float_precision": "float64",
                    "parquet_float_columns": "float32",
                },
            }
            active_settings = {
                **audit.MODELING_DATASET_SETTINGS,
                "float_precision": "float64",
            }

            with (
                mock.patch.object(audit, "MODELING_DATASET_SETTINGS", active_settings),
                self.assertRaisesRegex(ValueError, "parquet precision"),
            ):
                audit.validate_modeling_artifacts_for_audit(
                    parquet_path=parquet_path,
                    model_meta_path=Path(tmpdir) / "model_meta.json",
                    model_meta=model_meta,
                )

    def test_features_to_inspect_keeps_only_prediction_impact_columns(self):
        feature_summary_df = pd.DataFrame(
            {
                "feature": [
                    "signal_feature",
                    "drift_feature",
                    "medium_feature",
                    "raw_diff_only_feature",
                ],
                "rows_pred_shift_gt_tol_if_fixed": [3, 2, 1, 0],
                "mean_abs_proba_shift_on_shift_rows_if_fixed": [
                    0.020,
                    0.010,
                    0.005,
                    0.0,
                ],
                "max_abs_proba_shift_if_fixed": [0.040, 0.030, 0.006, 0.0],
                "rows_proba_diff_gt_tol_resolved_if_fixed": [1, 2, 0, 0],
                "rows_signal_mismatch_resolved_if_fixed": [1, 0, 0, 0],
                "max_abs_diff": [0.5, 0.4, 0.3, 9.9],
                "mean_abs_diff": [0.05, 0.04, 0.03, 0.99],
                "importance_gain": [10.0, 20.0, 30.0, 40.0],
                "builder": ["unused", "unused", "unused", "unused"],
                "group": ["unused", "unused", "unused", "unused"],
                "net_pred_gap_reduction": [1.0, 1.0, 1.0, 1.0],
            }
        )

        result = audit._build_features_to_inspect_df(
            feature_summary_df,
            decision_row_count=10,
        )

        self.assertEqual(
            list(result.columns),
            audit.FEATURES_TO_INSPECT_COLUMNS,
        )
        self.assertEqual(
            result["feature"].tolist(),
            ["signal_feature", "drift_feature", "medium_feature"],
        )
        self.assertEqual(result["severity"].tolist(), ["critical", "high", "medium"])
        self.assertAlmostEqual(float(result.loc[0, "pred_shift_rows_pct"]), 30.0)
        self.assertNotIn("builder", result.columns)
        self.assertNotIn("group", result.columns)
        self.assertNotIn("net_pred_gap_reduction", result.columns)

    def test_features_to_inspect_does_not_report_nan_diff_as_zero(self):
        feature_summary_df = pd.DataFrame(
            {
                "feature": ["nan_diff_feature"],
                "rows_pred_shift_gt_tol_if_fixed": [1],
                "mean_abs_proba_shift_on_shift_rows_if_fixed": [0.01],
                "max_abs_proba_shift_if_fixed": [0.02],
                "rows_proba_diff_gt_tol_resolved_if_fixed": [0],
                "rows_signal_mismatch_resolved_if_fixed": [0],
                "max_abs_diff": [float("nan")],
                "mean_abs_diff": [float("nan")],
                "importance_gain": [1.0],
            }
        )

        result = audit._build_features_to_inspect_df(
            feature_summary_df,
            decision_row_count=10,
        )

        self.assertEqual(result["feature"].tolist(), ["nan_diff_feature"])
        self.assertTrue(math.isnan(float(result.loc[0, "max_feature_abs_diff"])))
        self.assertTrue(math.isnan(float(result.loc[0, "mean_feature_abs_diff"])))

    def test_summary_payload_is_short_and_feature_focused(self):
        features_to_inspect_df = pd.DataFrame(
            {
                "rank": [1],
                "severity": ["high"],
                "feature": ["drift_feature"],
                "pred_shift_rows": [2],
                "pred_shift_rows_pct": [20.0],
                "mean_pred_shift": [0.01],
                "max_pred_shift": [0.03],
                "rows_where_prediction_diff_exceeds_tol_explained": [2],
                "rows_where_up_down_prediction_flips_explained": [0],
                "max_feature_abs_diff": [0.4],
                "mean_feature_abs_diff": [0.04],
                "importance_gain": [20.0],
            }
        )
        report = {
            "summary": pd.Series(
                {
                    "audit_start": "2026-05-09T09:19:00",
                    "audit_end": "2026-05-16T09:18:00",
                    "bootstrap_rows": 21600,
                    "audit_rows_total_1m": 10080,
                    "decision_row_count": 10,
                    "feature_count": 124,
                    "rows_with_proba_diff_gt_tol": 2,
                    "max_proba_up_abs_diff": 0.03,
                    "mean_proba_up_abs_diff": 0.001,
                    "rows_with_signal_mismatch": 0,
                    "rows_with_business_decision_mismatch": 0,
                    "rows_with_any_policy_mismatch": 0,
                }
            )
        }
        drift_reason_report = {
            "summary": pd.Series({"explanation_basis": "proba_diff_gt_tol"})
        }

        payload = audit._build_live_feature_parity_summary_payload(
            report,
            drift_reason_report,
            features_to_inspect_df,
        )

        self.assertEqual(payload["verdict"], "inspect")
        self.assertEqual(payload["features_to_inspect"], 1)
        self.assertEqual(payload["top_features"][0]["feature"], "drift_feature")
        self.assertNotIn("live_vs_stored", payload)
        self.assertNotIn("top10_feature_drop_candidates", payload)
        self.assertNotIn("feature_drop_candidate_thresholds", payload)

    def test_default_save_outputs_writes_only_main_csvs(self):
        feature_summary_df = pd.DataFrame(
            {
                "feature": ["drift_feature"],
                "rows_pred_shift_gt_tol_if_fixed": [2],
                "mean_abs_proba_shift_on_shift_rows_if_fixed": [0.01],
                "max_abs_proba_shift_if_fixed": [0.03],
                "rows_proba_diff_gt_tol_resolved_if_fixed": [2],
                "rows_signal_mismatch_resolved_if_fixed": [0],
                "max_abs_diff": [0.4],
                "mean_abs_diff": [0.04],
                "importance_gain": [20.0],
            }
        )
        step_summary_df = pd.DataFrame(
            {
                "Opened": ["2026-05-09T09:20:00"],
                "live_proba_up": [0.55],
                "stored_proba_up": [0.52],
                "proba_up_abs_diff": [0.03],
                "signal_mismatch": [0],
                "business_decision_mismatch": [0],
                "policy_decision_mismatch": [0],
                "top_prediction_impact_feature": ["drift_feature"],
                "top_prediction_impact_abs_proba_shift_if_fixed": [0.03],
                "top_prediction_impact_live_value": [1.2],
                "top_prediction_impact_stored_value": [1.1],
                "feature_max_abs_diff": [0.4],
                "feature_mean_abs_diff": [0.04],
            }
        )
        results = {
            "live_vs_stored_report": {
                "summary": pd.Series(
                    {
                        "audit_start": "2026-05-09T09:19:00",
                        "audit_end": "2026-05-16T09:18:00",
                        "bootstrap_rows": 21600,
                        "audit_rows_total_1m": 10080,
                        "decision_row_count": 1,
                        "feature_count": 1,
                        "rows_with_proba_diff_gt_tol": 1,
                        "max_proba_up_abs_diff": 0.03,
                        "mean_proba_up_abs_diff": 0.03,
                        "rows_with_signal_mismatch": 0,
                        "rows_with_business_decision_mismatch": 0,
                        "rows_with_any_policy_mismatch": 0,
                    }
                ),
                "feature_summary_df": feature_summary_df,
                "step_summary_df": step_summary_df,
            },
            "drift_reason_report": {
                "summary": pd.Series({"explanation_basis": "proba_diff_gt_tol"})
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            written = audit.save_audit_outputs(results, output_dir=Path(tmpdir))
            csv_names = sorted(path.name for path in Path(tmpdir).glob("*.csv"))
            summary = json.loads(
                (Path(tmpdir) / "live_vs_stored_summary.json").read_text()
            )

        self.assertEqual(
            csv_names,
            ["features_to_inspect.csv", "rows_to_inspect.csv"],
        )
        self.assertEqual(
            sorted(path.name for path in written.values() if path.suffix == ".csv"),
            ["features_to_inspect.csv", "rows_to_inspect.csv"],
        )
        self.assertEqual(summary["features_to_inspect"], 1)
        self.assertEqual(summary["top_features"][0]["feature"], "drift_feature")
        self.assertEqual(summary["feature_parity"]["status"], "measured")
        self.assertEqual(summary["prediction_parity"]["status"], "measured")
        self.assertEqual(
            summary["decision_parity"]["status"],
            "not_verified_missing_quotes",
        )


if __name__ == "__main__":
    unittest.main()
