import unittest

import numpy as np
import pandas as pd

from features.btc_preopen_contract import (
    DECISION_COL,
    TARGET_AVAILABLE_COL,
    TARGET_COL,
    TARGET_END_PRICE_COL,
    TARGET_START_COL,
    TARGET_START_PRICE_COL,
    build_preopen_contract_frame,
    purge_unavailable_training_rows,
    scheduled_market_start_for_opened,
)
from features.btc_preopen_baseline import (
    FEATURE_COLUMNS,
    build_causal_preopen_features,
)


def candles(start, periods, *, missing=()):
    opened = pd.date_range(start, periods=periods, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "Opened": opened,
            "Open": np.arange(periods, dtype=np.float64) + 100.0,
            "High": np.arange(periods, dtype=np.float64) + 101.0,
            "Low": np.arange(periods, dtype=np.float64) + 99.0,
            "Close": np.arange(periods, dtype=np.float64) + 100.5,
            "Volume": np.ones(periods, dtype=np.float64),
        }
    )
    return frame.loc[~frame["Opened"].isin(pd.DatetimeIndex(missing))].reset_index(drop=True)


class BtcPreopenContractTests(unittest.TestCase):
    def test_input_candle_and_exact_five_future_candles_define_row(self):
        frame = candles("2026-06-01 16:40", 14)
        row = build_preopen_contract_frame(frame).set_index("Opened").loc[
            pd.Timestamp("2026-06-01 16:43", tz="UTC")
        ]

        self.assertEqual(row["Open"], 103.0)
        self.assertEqual(row["Close"], 103.5)
        self.assertEqual(row[TARGET_START_COL], pd.Timestamp("2026-06-01 16:45", tz="UTC"))
        self.assertEqual(row[TARGET_START_PRICE_COL], 105.0)
        self.assertEqual(row[TARGET_END_PRICE_COL], 109.5)
        self.assertEqual(row[TARGET_COL], 1.0)
        self.assertEqual(
            row[TARGET_AVAILABLE_COL], pd.Timestamp("2026-06-01 16:50", tz="UTC")
        )

        # The old close-to-close six-minute return would use 103.5 -> 109.5.
        # Variant 1 uses the market window's 16:45 open -> 16:50 close instead.
        self.assertNotEqual(row[TARGET_START_PRICE_COL], row["Close"])

    def test_candle_close_is_available_at_nominal_decision_and_entry_is_marked(self):
        frame = build_preopen_contract_frame(candles("2026-06-01 16:40", 12))
        indexed = frame.set_index("Opened")
        row = indexed.loc[pd.Timestamp("2026-06-01 16:43", tz="UTC")]

        self.assertEqual(
            row["feature_candle_close_at"], pd.Timestamp("2026-06-01 16:44", tz="UTC")
        )
        self.assertEqual(
            row["nominal_decision_at"], pd.Timestamp("2026-06-01 16:44", tz="UTC")
        )
        self.assertTrue(row[DECISION_COL])
        self.assertFalse(
            indexed.loc[pd.Timestamp("2026-06-01 16:44", tz="UTC"), DECISION_COL]
        )

    def test_window_crosses_hour_and_utc_day_boundary(self):
        frame = candles("2026-12-31 23:56", 12)
        row = build_preopen_contract_frame(frame).set_index("Opened").loc[
            pd.Timestamp("2026-12-31 23:58", tz="UTC")
        ]

        self.assertEqual(row[TARGET_START_COL], pd.Timestamp("2027-01-01 00:00", tz="UTC"))
        self.assertEqual(row[TARGET_START_PRICE_COL], 104.0)
        self.assertEqual(row[TARGET_END_PRICE_COL], 108.5)
        self.assertEqual(row[TARGET_COL], 1.0)
        self.assertTrue(row[DECISION_COL])

    def test_missing_minute_invalidates_target_even_when_endpoints_exist(self):
        gap_time = pd.Timestamp("2026-06-01 16:47", tz="UTC")
        result = build_preopen_contract_frame(
            candles("2026-06-01 16:40", 14, missing=(gap_time,))
        ).set_index("Opened")

        row = result.loc[pd.Timestamp("2026-06-01 16:43", tz="UTC")]
        self.assertTrue(np.isnan(row[TARGET_COL]))

    def test_overlapping_minute_labels_are_distinct_and_purge_by_label_time(self):
        result = build_preopen_contract_frame(candles("2026-06-01 16:40", 18)).set_index(
            "Opened"
        )
        first = result.loc[pd.Timestamp("2026-06-01 16:43", tz="UTC")]
        second = result.loc[pd.Timestamp("2026-06-01 16:44", tz="UTC")]
        self.assertEqual(first[TARGET_START_COL], pd.Timestamp("2026-06-01 16:45", tz="UTC"))
        self.assertEqual(second[TARGET_START_COL], pd.Timestamp("2026-06-01 16:46", tz="UTC"))

        opened = pd.DatetimeIndex(
            [
                "2026-06-01 16:56+00:00",  # label available at 17:03
                "2026-06-01 16:57+00:00",  # label available at 17:04
                "2026-06-01 16:58+00:00",  # label available at 17:05
            ]
        )
        kept = purge_unavailable_training_rows(opened, "2026-06-01 17:04+00:00")
        np.testing.assert_array_equal(kept, np.array([0, 1], dtype=np.int64))

    def test_ties_resolve_up_and_missing_future_end_is_nan(self):
        frame = candles("2026-06-01 16:40", 12)
        frame.loc[frame["Opened"].eq(pd.Timestamp("2026-06-01 16:49", tz="UTC")), "Close"] = 105.0
        result = build_preopen_contract_frame(frame).set_index("Opened")
        tie = result.loc[pd.Timestamp("2026-06-01 16:43", tz="UTC")]
        self.assertEqual(tie[TARGET_COL], 1.0)

        truncated = build_preopen_contract_frame(frame.iloc[:-3]).set_index("Opened")
        self.assertTrue(
            np.isnan(truncated.loc[pd.Timestamp("2026-06-01 16:43", tz="UTC"), TARGET_COL])
        )

    def test_live_schedule_uses_completed_candle_one_minute_before_market(self):
        self.assertEqual(
            scheduled_market_start_for_opened("2026-06-01 16:43+00:00"),
            pd.Timestamp("2026-06-01 16:45", tz="UTC"),
        )
        self.assertIsNone(scheduled_market_start_for_opened("2026-06-01 16:44+00:00"))
        self.assertEqual(
            scheduled_market_start_for_opened("2026-12-31 23:58+00:00"),
            pd.Timestamp("2027-01-01 00:00", tz="UTC"),
        )

    def test_baseline_features_use_only_current_closed_candle_and_past(self):
        frame = candles("2026-06-01 15:00", 320)
        frame["UM_BTCUSDT_Close"] = frame["Close"] + 2.0
        first = build_causal_preopen_features(frame)
        cutoff = pd.Timestamp("2026-06-01 17:43", tz="UTC")
        changed = frame.copy()
        changed.loc[changed["Opened"] > cutoff, ["Open", "High", "Low", "Close", "Volume", "UM_BTCUSDT_Close"]] *= 5.0
        second = build_causal_preopen_features(changed)
        first_row = first.loc[first["Opened"].eq(cutoff), list(FEATURE_COLUMNS)].to_numpy()
        second_row = second.loc[second["Opened"].eq(cutoff), list(FEATURE_COLUMNS)].to_numpy()
        np.testing.assert_allclose(first_row, second_row, equal_nan=True)

    def test_baseline_minute_lags_do_not_jump_over_data_gaps(self):
        frame = candles(
            "2026-06-01 16:40",
            20,
            missing=(pd.Timestamp("2026-06-01 16:47", tz="UTC"),),
        )
        result = build_causal_preopen_features(frame).set_index("Opened")
        self.assertTrue(np.isnan(result.loc[pd.Timestamp("2026-06-01 16:48", tz="UTC"), "log_return_1m"]))


if __name__ == "__main__":
    unittest.main()
