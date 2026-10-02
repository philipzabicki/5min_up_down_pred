import unittest

import numpy as np
import pandas as pd

from utils.btc_oof_audit import (
    macro_average_fold_metrics,
    paired_block_bootstrap,
    recompute_close_target,
    validate_common_market_sample,
)


class BtcOofTargetAuditTests(unittest.TestCase):
    def test_target_uses_exact_five_minute_timestamp_and_keeps_gaps_missing(self):
        opened = pd.to_datetime([
            "2026-01-01 00:00:00",
            "2026-01-01 00:02:00",
            "2026-01-01 00:07:00",
            "2026-01-01 00:08:00",
        ])
        close = [100.0, 101.0, 101.0, 99.0]

        target = recompute_close_target(opened, close, horizon_minutes=5)

        self.assertTrue(np.isnan(target[0]))
        self.assertEqual(target[1], 1.0)  # Exact t+5 tie resolves UP.
        self.assertTrue(np.isnan(target[2]))
        self.assertTrue(np.isnan(target[3]))

    def test_common_market_sample_requires_same_ids_labels_and_raw_probabilities(self):
        evaluation = pd.DataFrame({
            "condition_id": ["a", "b"],
            "market_start_utc": ["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"],
            "target_polymarket_up": [1.0, 0.0],
            "p_model_up": [0.7, 0.3],
            "oof_raw": [0.7, 0.3],
        })
        common = pd.DataFrame({
            "condition_id": ["b", "a"],
            "market_start_utc": ["2026-01-01T00:05:00Z", "2026-01-01T00:00:00Z"],
            "target_binance_proxy_up": [0.0, 1.0],
            "target_polymarket_up": [0.0, 1.0],
            "polymarket_outcome_up": [0.0, 1.0],
            "p_model_up": [0.3, 0.7],
        })

        joined = validate_common_market_sample(evaluation, common, expected_rows=2)
        self.assertEqual(set(joined.condition_id), {"a", "b"})
        self.assertEqual(len(joined), 2)

        broken = common.copy()
        broken.loc[broken.condition_id.eq("a"), "p_model_up"] = 0.71
        with self.assertRaisesRegex(ValueError, "Raw OOF probabilities differ"):
            validate_common_market_sample(evaluation, broken, expected_rows=2)

    def test_training_cv_summary_is_macro_average_not_pooled_rows(self):
        folds = [
            {"log_loss": 0.2, "accuracy": 0.9},
            {"log_loss": 0.8, "accuracy": 0.1},
        ]

        summary = macro_average_fold_metrics(folds)

        self.assertEqual(summary["log_loss"], 0.5)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertNotEqual(summary["log_loss"], (0.2 * 90 + 0.8 * 10) / 100)

    def test_paired_block_bootstrap_is_reproducible_and_keeps_identical_targets_paired(self):
        timestamps = pd.date_range("2026-01-01", periods=16, freq="6h", tz="UTC")
        target = np.array([0.0, 1.0] * 8)
        probability = np.linspace(0.2, 0.8, len(target))
        kwargs = dict(
            targets={"Binance": target, "Polymarket": target.copy()},
            probability=probability,
            timestamps=timestamps,
            replications=40,
            block_days=3,
            seed=12,
        )

        first = paired_block_bootstrap(**kwargs)
        second = paired_block_bootstrap(**kwargs)

        self.assertEqual(first, second)
        for metric, bounds in first["Polymarket_minus_Binance"].items():
            self.assertEqual(bounds, [0.0, 0.0], metric)


if __name__ == "__main__":
    unittest.main()
