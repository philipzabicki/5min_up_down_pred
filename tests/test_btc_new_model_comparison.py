import unittest

import pandas as pd

from run_btc_new_model_comparison import (
    align_new_oof_to_markets,
    purge_training_by_label_availability,
)


class BtcNewModelComparisonTests(unittest.TestCase):
    def test_label_availability_purges_exactly_the_unavailable_boundary_rows(self):
        train_opened = pd.Series(
            pd.date_range("2026-01-01T11:53:00Z", periods=7, freq="min")
        )
        available = purge_training_by_label_availability(
            train_opened,
            pd.Timestamp("2026-01-01T12:01:00Z"),
        )

        self.assertEqual(int(available.sum()), 2)
        self.assertEqual(int((~available).sum()), 5)
        self.assertFalse(available.iloc[2])  # 11:55 label is available exactly at 12:01.
        self.assertTrue(available.iloc[1])

    def test_oof_alignment_uses_exact_opened_timestamps_not_row_position(self):
        opened = pd.date_range("2026-01-01T12:00:00Z", periods=3, freq="5min")
        common = pd.DataFrame(
            {
                "Opened": opened,
                "condition_id": ["market-a", "market-b", "market-c"],
                "market_start_utc": opened + pd.Timedelta(minutes=1),
                "decision_available_at": opened + pd.Timedelta(minutes=1),
                "timestamp_utc": opened + pd.Timedelta(minutes=1),
                "target_polymarket_up": [1, 0, 1],
                "target_binance_proxy_up": [1, 1, 0],
                "p_model_up": [0.6, 0.4, 0.7],
                "quote_valid": [True, True, True],
                "validation_status": ["valid", "valid", "valid"],
            }
        )
        new_oof = pd.DataFrame(
            {
                "Opened": [opened[2], opened[0]],
                "oof_pred_proba_up": [0.82, 0.21],
                "target_5m_candle_up": [0, 1],
            }
        )

        aligned, counts = align_new_oof_to_markets(common, new_oof)
        by_market = aligned.set_index("condition_id")

        self.assertEqual(counts["exact_timestamp_matches"], 2)
        self.assertEqual(counts["common_rows_without_new_oof"], 1)
        self.assertAlmostEqual(by_market.loc["market-a", "p_new_model_up"], 0.21)
        self.assertAlmostEqual(by_market.loc["market-c", "p_new_model_up"], 0.82)
        self.assertEqual(by_market.loc["market-b", "new_oof_join"], "left_only")
        self.assertTrue(pd.isna(by_market.loc["market-b", "p_new_model_up"]))

    def test_oof_alignment_rejects_a_shifted_market_start(self):
        opened = pd.Timestamp("2026-01-01T12:00:00Z")
        common = pd.DataFrame(
            {
                "Opened": [opened],
                "condition_id": ["market-a"],
                "market_start_utc": [opened + pd.Timedelta(minutes=2)],
                "decision_available_at": [opened + pd.Timedelta(minutes=1)],
                "timestamp_utc": [opened + pd.Timedelta(minutes=1)],
                "target_polymarket_up": [1],
                "target_binance_proxy_up": [1],
                "p_model_up": [0.7],
                "quote_valid": [True],
                "validation_status": ["valid"],
            }
        )
        new_oof = pd.DataFrame(
            {
                "Opened": [opened],
                "oof_pred_proba_up": [0.8],
                "target_5m_candle_up": [1],
            }
        )

        with self.assertRaisesRegex(ValueError, "Market start is not the exact decision timestamp"):
            align_new_oof_to_markets(common, new_oof)


if __name__ == "__main__":
    unittest.main()
