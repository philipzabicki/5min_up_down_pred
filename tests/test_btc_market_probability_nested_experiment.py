import json
import unittest

import numpy as np
import pandas as pd

import run_btc_market_probability_nested_experiment as experiment


class BtcMarketProbabilityNestedExperimentTests(unittest.TestCase):
    def test_mixture_zero_and_full_weights_are_exact_controls(self):
        frame = pd.DataFrame({
            "p_market": [0.2, 0.7],
            "p_btc_calibrated": [0.8, 0.3],
        })
        no_correction = experiment._predict_spec(
            frame, "E", {"parameter": "w", "value": 0.0},
        )
        btc_only = experiment._predict_spec(
            frame, "E", {"parameter": "w", "value": 1.0},
        )
        half_mix = experiment._predict_spec(
            frame, "E", {"parameter": "w", "value": 0.5},
        )
        np.testing.assert_array_equal(no_correction, frame.p_market.to_numpy())
        np.testing.assert_array_equal(btc_only, frame.p_btc_calibrated.to_numpy())
        np.testing.assert_allclose(half_mix, [0.5, 0.5])

    def test_logit_correction_has_exact_market_control_and_shrinks(self):
        market = np.full(80, 0.5)
        btc = np.tile([0.8, 0.2], 40)
        target = np.tile([1, 0], 40)
        frame = pd.DataFrame({
            "logit_market": np.zeros(80),
            "logit_btc_calibrated": np.log(btc / (1.0 - btc)),
            "official_outcome_up": target,
        })
        no_correction = experiment._fit_regularized_offset(frame, None)
        low_penalty = experiment._fit_regularized_offset(frame, 0.1)
        high_penalty = experiment._fit_regularized_offset(frame, 10.0)
        prediction = experiment._predict_spec(
            pd.DataFrame({"p_market": market, "p_btc_calibrated": btc}), "F",
            {"parameter": "no_correction", "value": None}, no_correction,
        )
        np.testing.assert_array_equal(prediction, market)
        self.assertLess(np.hypot(high_penalty["a"], high_penalty["b"]), np.hypot(low_penalty["a"], low_penalty["b"]))

    def test_training_fold_purges_labels_unavailable_at_boundary(self):
        frame = pd.DataFrame({
            "market_start_utc": pd.to_datetime([
                "2026-04-28T00:00:00Z", "2026-04-29T00:00:00Z",
                "2026-04-30T00:00:00Z", "2026-05-01T00:00:00Z",
            ], utc=True),
            "label_available_at_utc": pd.to_datetime([
                "2026-04-28T00:06:00Z", "2026-04-29T00:06:00Z",
                "2026-05-01T00:01:00Z", "2026-05-01T00:06:00Z",
            ], utc=True),
            "official_outcome_up": [0, 1, 0, 1],
        })
        train, valid = experiment._training_validation_rows(frame, {
            "fold_id": "purge-test",
            "start_inclusive_utc": "2026-05-01T00:00:00Z",
            "end_exclusive_utc": "2026-05-02T00:00:00Z",
        })
        self.assertEqual(len(train), 2)
        self.assertEqual(len(valid), 1)
        self.assertTrue(train.label_available_at_utc.le(pd.Timestamp("2026-05-01T00:00:00Z")).all())

    def test_policy_gate_requires_positive_block_interval_and_both_folds(self):
        config = json.loads(experiment.CONFIG_PATH.read_text(encoding="utf-8"))
        fold_ids = [fold["fold_id"] for fold in config["development_outer_folds"]]
        candidate = {
            "policy": "F_regularized_logit_correction_ev_0.0000%",
            "forecast": "F_regularized_logit_correction",
            "ev_hurdle_fraction": 0.0,
            "fills": 80,
            "active_utc_days": 20,
            "realized_net_pnl_usd": 3.5,
            "total_pnl_95pct": [1.0, 5.0],
            "mean_daily_pnl": 0.2,
            "fold_results": [
                {"outer_fold_id": fold_ids[0], "fills": 30, "realized_net_pnl_usd": 1.5},
                {"outer_fold_id": fold_ids[1], "fills": 50, "realized_net_pnl_usd": 2.0},
            ],
        }
        selected = experiment._select_policy([candidate], config)
        self.assertEqual(selected["selected_policy"]["policy"], candidate["policy"])
        candidate["total_pnl_95pct"] = [-1.0, 5.0]
        selected = experiment._select_policy([candidate], config)
        self.assertEqual(selected["selected_policy"]["policy"], "no_trade")
        self.assertEqual(selected["selected_policy"]["total_pnl_95pct"], [0.0, 0.0])

    def test_daily_block_interval_is_reproducible_and_no_trade_is_zero(self):
        daily = pd.Series([0.0] * 20, index=pd.date_range("2026-05-01", periods=20, tz="UTC"))
        first = experiment._block_interval(daily, seed=123)
        second = experiment._block_interval(daily, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(first["total_pnl_95pct"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
