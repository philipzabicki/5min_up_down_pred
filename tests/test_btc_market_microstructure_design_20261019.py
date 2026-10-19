import json
import unittest
from types import SimpleNamespace

import pandas as pd

import build_btc_market_microstructure_design_20261019 as design


class MicrostructureDesignTests(unittest.TestCase):
    @staticmethod
    def _snapshot(start, **updates):
        snapshot = {
            "condition_id": "0xMarket",
            "market_start_utc": start,
            "entry_time_utc": start - pd.Timedelta(seconds=60),
            "up_best_bid": 0.48,
            "up_best_ask": 0.50,
            "up_best_ask_size_shares": 20.0,
            "down_best_bid": 0.49,
            "down_best_ask": 0.51,
            "down_best_ask_size_shares": 20.0,
            "up_ask_levels": [[0.50, 20.0]],
            "down_ask_levels": [[0.51, 20.0]],
            "quote_valid": True,
            "has_full_snapshot": True,
            "up_quote_complemented": False,
            "down_quote_complemented": False,
            "archive_entry_hour_available": True,
            "archive_freshness_window_available": True,
            "no_future_event_at_entry": True,
            "no_future_source_event_at_entry": True,
            "archive_gap_after_up_book_init": False,
            "archive_gap_after_down_book_init": False,
            "up_ask_order_ambiguous": False,
            "down_ask_order_ambiguous": False,
            "market_equal_source_timestamp_price_change_count": 0,
            "market_out_of_order_price_change_count": 0,
            "market_cross_side_late_price_change_count": 0,
            "up_bid_ask_crossed": False,
            "down_bid_ask_crossed": False,
            "up_ask_age_seconds": 2.0,
            "down_ask_age_seconds": 3.0,
            "up_ask_received_age_seconds": 2.1,
            "down_ask_received_age_seconds": 3.1,
        }
        snapshot.update(updates)
        return snapshot

    @staticmethod
    def _raw(start, probability=0.70):
        return SimpleNamespace(
            market_start_utc=start,
            nominal_decision_at=start - pd.Timedelta(seconds=60),
            p_model_raw=probability,
        )

    def test_vwap_walks_ask_levels_for_exact_gross_notional(self):
        vwap = design._vwap_for_gross_spend([(0.50, 2.0), (0.60, 20.0)])
        expected = 5.0 / (2.0 + 4.0 / 0.60)
        self.assertAlmostEqual(vwap, expected, places=12)
        self.assertAlmostEqual(vwap - 0.50, expected - 0.50, places=12)
        self.assertIsNone(design._vwap_for_gross_spend([(0.50, 2.0)]))

    def test_feature_formulas_use_native_t60_book_and_raw_oof(self):
        start = pd.Timestamp("2026-05-01T12:00:00Z")
        row = design._feature_record(self._raw(start), self._snapshot(start))
        self.assertTrue(row["microstructure_eligible"])
        self.assertEqual(row["condition_id"], "0xmarket")
        self.assertAlmostEqual(row["spread_up"], 0.02)
        self.assertAlmostEqual(row["spread_down"], 0.02)
        self.assertAlmostEqual(row["up_down_ask_depth_imbalance"], 0.0)
        self.assertAlmostEqual(row["slippage_5_up"], 0.0)
        self.assertAlmostEqual(row["slippage_5_down"], 0.0)
        self.assertAlmostEqual(row["mid_sum_minus_one"], -0.01)
        self.assertAlmostEqual(row["p_market"], 0.49 / 0.99)
        self.assertAlmostEqual(
            row["raw_minus_market_logit"],
            design.market_probability._logit([0.70])[0] - design.market_probability._logit([0.49 / 0.99])[0],
        )

    def test_t59_snapshot_and_ambiguous_or_gapped_state_are_not_features(self):
        start = pd.Timestamp("2026-05-01T12:00:00Z")
        with self.assertRaisesRegex(RuntimeError, "not T-60"):
            design._feature_record(
                self._raw(start),
                self._snapshot(start, entry_time_utc=start - pd.Timedelta(seconds=59)),
            )
        for updates in (
            {"up_ask_order_ambiguous": True},
            {"archive_gap_after_down_book_init": True},
            {"down_quote_complemented": True},
            {"no_future_source_event_at_entry": False},
        ):
            with self.subTest(updates=updates):
                row = design._feature_record(self._raw(start), self._snapshot(start, **updates))
                self.assertFalse(row["microstructure_eligible"])
                self.assertIsNone(row["spread_up"])
                self.assertIsNone(row["slippage_5_down"])

    def test_aggregate_event_conflict_count_is_audit_only_not_quote_age_or_current_ambiguity(self):
        start = pd.Timestamp("2026-05-01T12:00:00Z")
        row = design._feature_record(
            self._raw(start),
            self._snapshot(start, market_equal_source_timestamp_price_change_count=17),
        )
        self.assertTrue(row["microstructure_eligible"])
        self.assertEqual(row["equal_source_timestamp_conflict_count"], 17)
        self.assertEqual(row["up_ask_book_event_age_seconds"], 2.0)

    def test_config_and_input_columns_exclude_holdout_and_t59_features(self):
        config = json.loads(design.CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertLess(pd.Timestamp(config["development_start_utc"]), pd.Timestamp(config["development_cutoff_exclusive_utc"]))
        self.assertTrue(config["sources"]["t60_snapshot_path"].endswith("t60_decision_books_with_source_coverage.parquet"))
        self.assertFalse(any("target" in name or "outcome" in name or "label" in name for name in design.RAW_COLUMNS + design.SNAPSHOT_COLUMNS))
        self.assertNotIn("target_polymarket_up", design.RAW_COLUMNS + design.SNAPSHOT_COLUMNS)
        self.assertEqual(len(design.MICROSTRUCTURE_FEATURES), 6)


if __name__ == "__main__":
    unittest.main()
