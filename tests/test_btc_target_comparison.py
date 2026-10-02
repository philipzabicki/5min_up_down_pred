import unittest

import numpy as np
import pandas as pd

from run_btc_target_comparison import (
    _assert_portfolio_conservation,
    assert_identical_pair_rows,
    assert_meta_oos_rows,
    exact_binance_targets,
    make_outer_boundaries,
    split_outer_window,
)


class ExactWindowMappingTests(unittest.TestCase):
    def test_exact_five_minute_endpoint_and_availability(self):
        opened = pd.Timestamp("2026-04-01 12:04:00", tz="UTC")
        source = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.DatetimeIndex([opened, opened + pd.Timedelta(minutes=5)]),
        )
        current, future, target, valid, available = exact_binance_targets(source, [opened])
        self.assertTrue(valid[0])
        self.assertEqual(target[0], 1.0)
        self.assertEqual(current.iloc[0].Close, 100.0)
        self.assertEqual(future.iloc[0].Close, 101.0)
        self.assertEqual(available[0], opened + pd.Timedelta(minutes=6))

    def test_missing_exact_endpoint_is_not_filled_from_later_row(self):
        opened = pd.Timestamp("2026-04-01 12:04:00", tz="UTC")
        source = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.DatetimeIndex([opened, opened + pd.Timedelta(minutes=6)]),
        )
        _, future, target, valid, _ = exact_binance_targets(source, [opened])
        self.assertFalse(valid[0])
        self.assertTrue(pd.isna(future.iloc[0].Close))
        self.assertTrue(pd.isna(target[0]))


class PairedSplitTests(unittest.TestCase):
    def test_expected_outer_boundaries(self):
        self.assertEqual(make_outer_boundaries(15_677), [6_270, 9_406, 12_541, 15_677])

    def test_outer_test_never_enters_training_and_unavailable_labels_are_dropped(self):
        times = pd.date_range("2026-04-01", periods=10, freq="5min", tz="UTC")
        data = pd.DataFrame({
            "sample_position": np.arange(10),
            "condition_id": [f"m{i}" for i in range(10)],
            "decision_available_at": times,
            "both_labels_available_at": times - pd.Timedelta(seconds=1),
        })
        data.loc[3, "both_labels_available_at"] = times[4] + pd.Timedelta(seconds=1)
        past, test, cutoff, lo, hi = split_outer_window(data, 0)
        self.assertEqual((lo, hi), (4, 6))
        self.assertEqual(test.sample_position.tolist(), [4, 5])
        self.assertNotIn(3, past.sample_position.tolist())
        self.assertTrue((past.both_labels_available_at < cutoff).all())
        self.assertTrue((past.sample_position < lo).all())

    def test_pair_requires_same_rows_features_order_and_weights(self):
        assert_identical_pair_rows(
            ["a", "b"], ["a", "b"], ["f1", "f2"], ["f1", "f2"], [0.4, 0.5], [0.4, 0.5]
        )
        with self.assertRaises(AssertionError):
            assert_identical_pair_rows(
                ["a", "b"], ["a", "c"], ["f1", "f2"], ["f1", "f2"], [0.4, 0.5], [0.4, 0.5]
            )
        with self.assertRaises(AssertionError):
            assert_identical_pair_rows(
                ["a", "b"], ["a", "b"], ["f1", "f2"], ["f2", "f1"], [0.4, 0.5], [0.4, 0.5]
            )


class SecondLayerAndPortfolioGuardTests(unittest.TestCase):
    def _meta_rows(self):
        return pd.DataFrame({
            "sample_position": [10, 11],
            "base_model_train_last_sample_position": [8, 8],
            "base_prediction_is_oos": [True, True],
            "decision_available_at": pd.to_datetime(["2026-04-01T00:10Z", "2026-04-01T00:15Z"]),
            "base_model_train_label_available_at": pd.to_datetime(["2026-04-01T00:05Z", "2026-04-01T00:05Z"]),
            "both_labels_available_at": pd.to_datetime(["2026-04-01T00:06Z", "2026-04-01T00:11Z"]),
            "outer_cutoff": pd.to_datetime(["2026-04-01T00:20Z", "2026-04-01T00:20Z"]),
            "p_target_B": [0.48, 0.52],
            "p_target_P": [0.49, 0.51],
        })

    def test_meta_layer_requires_first_stage_oos_predictions_and_available_labels(self):
        assert_meta_oos_rows(self._meta_rows())
        bad = self._meta_rows()
        bad.loc[0, "base_prediction_is_oos"] = False
        with self.assertRaises(AssertionError):
            assert_meta_oos_rows(bad)
        bad = self._meta_rows()
        bad.loc[0, "both_labels_available_at"] = pd.Timestamp("2026-04-01T00:21Z")
        with self.assertRaises(AssertionError):
            assert_meta_oos_rows(bad)

    def test_continuous_portfolio_reconciliation_guard(self):
        _assert_portfolio_conservation({"initial_bankroll": 100.0, "final_balance": 103.0, "realized_pnl": 3.0})
        with self.assertRaises(AssertionError):
            _assert_portfolio_conservation({"initial_bankroll": 100.0, "final_balance": 102.0, "realized_pnl": 3.0})


if __name__ == "__main__":
    unittest.main()
