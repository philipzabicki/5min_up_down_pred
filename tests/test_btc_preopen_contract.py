import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from features.btc_preopen_contract import (
    DECISION_COL,
    TARGET_AVAILABLE_COL,
    TARGET_COL,
    TARGET_END_PRICE_COL,
    RETURN_TARGET_COL,
    TARGET_START_COL,
    TARGET_START_PRICE_COL,
    build_preopen_contract_frame,
    build_preopen_return_target,
    chronological_fit_mask,
    last_closed_candle_opened_at,
    preopen_decision_mask,
    purge_unavailable_training_rows,
    scheduled_market_start_for_opened,
    validate_development_candle_input,
    validate_preopen_development_labels,
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
    def test_default_training_profile_uses_the_frozen_one_second_deadline(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "btc_preopen_training.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        self.assertEqual(
            config["profiles"]["nightly_v1"]["prediction_compute_budget_seconds"],
            1,
        )

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
        self.assertAlmostEqual(
            build_preopen_return_target(build_preopen_contract_frame(frame))[3],
            109.5 / 105.0 - 1.0,
        )
        self.assertEqual(
            row[TARGET_AVAILABLE_COL], pd.Timestamp("2026-06-01 16:50", tz="UTC")
        )

        # The old close-to-close six-minute return would use 103.5 -> 109.5.
        # Variant 1 uses the market window's 16:45 open -> 16:50 close instead.
        self.assertNotEqual(row[TARGET_START_PRICE_COL], row["Close"])

    def test_development_input_rejects_candles_at_exclusive_cutoff(self):
        cutoff = pd.Timestamp("2026-06-01 16:42:00Z")
        self.assertEqual(
            validate_development_candle_input(candles("2026-06-01 16:40", 2), cutoff),
            cutoff,
        )
        with self.assertRaisesRegex(ValueError, "at or after the exclusive cutoff"):
            validate_development_candle_input(candles("2026-06-01 16:40", 3), cutoff)

    def test_development_target_must_end_no_later_than_cutoff(self):
        row = build_preopen_contract_frame(
            candles("2026-06-01 16:40", 10)
        ).loc[lambda frame: frame["Opened"].eq(pd.Timestamp("2026-06-01 16:43Z"))]
        validate_preopen_development_labels(row, "2026-06-01 16:50:00Z")

        row = row.copy()
        row.loc[:, "target_window_end_at"] = pd.Timestamp("2026-06-01 16:50:01Z")
        with self.assertRaisesRegex(ValueError, "target reaches outside"):
            validate_preopen_development_labels(row, "2026-06-01 16:50:00Z")

    def test_candle_event_close_is_nominal_time_but_receive_time_is_unobserved_in_batch(self):
        frame = build_preopen_contract_frame(candles("2026-06-01 16:40", 12))
        indexed = frame.set_index("Opened")
        row = indexed.loc[pd.Timestamp("2026-06-01 16:43", tz="UTC")]

        self.assertEqual(
            row["feature_candle_close_at"], pd.Timestamp("2026-06-01 16:44", tz="UTC")
        )
        self.assertEqual(
            row["nominal_decision_at"], pd.Timestamp("2026-06-01 16:44", tz="UTC")
        )
        self.assertTrue(pd.isna(row["actual_prediction_available_at"]))
        self.assertTrue(row[DECISION_COL])
        self.assertFalse(
            indexed.loc[pd.Timestamp("2026-06-01 16:44", tz="UTC"), DECISION_COL]
        )
        self.assertTrue(preopen_decision_mask(frame)[3])

    def test_prediction_compute_budget_must_finish_by_t59(self):
        candles_frame = candles("2026-06-01 16:40", 12)
        on_time = build_preopen_contract_frame(candles_frame, prediction_compute_budget_seconds=1.0)
        late = build_preopen_contract_frame(candles_frame, prediction_compute_budget_seconds=1.001)
        legacy_window = build_preopen_contract_frame(
            candles_frame, prediction_compute_budget_seconds=59.0
        )
        self.assertTrue(on_time.loc[3, DECISION_COL])
        self.assertFalse(late.loc[3, DECISION_COL])
        self.assertFalse(preopen_decision_mask(late)[3])
        self.assertFalse(legacy_window.loc[3, DECISION_COL])

    def test_12_00_market_uses_12_00_open_and_12_04_close(self):
        frame = candles("2026-06-01 11:58", 8)
        frame.loc[frame["Opened"].eq(pd.Timestamp("2026-06-01 12:00", tz="UTC")), "Open"] = 101.0
        frame.loc[frame["Opened"].eq(pd.Timestamp("2026-06-01 12:04", tz="UTC")), "Close"] = 101.0
        row = build_preopen_contract_frame(
            frame, prediction_compute_budget_seconds=1.0
        ).set_index("Opened").loc[pd.Timestamp("2026-06-01 11:58", tz="UTC")]

        self.assertEqual(row["feature_candle_close_at"], pd.Timestamp("2026-06-01 11:59", tz="UTC"))
        self.assertEqual(row["nominal_decision_at"], pd.Timestamp("2026-06-01 11:59", tz="UTC"))
        self.assertEqual(row["prediction_deadline_at"], pd.Timestamp("2026-06-01 11:59:01", tz="UTC"))
        self.assertEqual(row[TARGET_START_COL], pd.Timestamp("2026-06-01 12:00", tz="UTC"))
        self.assertEqual(row[TARGET_AVAILABLE_COL], pd.Timestamp("2026-06-01 12:05", tz="UTC"))
        self.assertEqual(row[TARGET_START_PRICE_COL], 101.0)
        self.assertEqual(row[TARGET_END_PRICE_COL], 101.0)
        self.assertEqual(row[TARGET_COL], 1.0)  # tie resolves UP

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

    def test_chronological_fit_mask_excludes_labels_at_exclusive_cutoff(self):
        opened = pd.DatetimeIndex([
            "2026-04-14T23:52:00Z",
            "2026-04-14T23:53:00Z",
            "2026-04-14T23:54:00Z",
            "2026-04-15T00:00:00Z",
        ])
        target = np.asarray([0.0, 1.0, 0.0, 1.0])
        available = opened + pd.Timedelta(minutes=7)
        cutoff = pd.Timestamp("2026-04-15T00:00:00Z")
        mask = chronological_fit_mask(
            opened, target, available,
            "2020-06-09T09:33:00Z", cutoff,
        )
        np.testing.assert_array_equal(mask, [True, False, False, False])

    def test_future_rows_cannot_change_the_existing_chronological_fit_prefix(self):
        cutoff = pd.Timestamp("2026-04-15T00:00:00Z")
        opened = pd.date_range("2026-04-14T23:45:00Z", periods=10, freq="min")
        target = np.arange(10, dtype=np.float64) % 2
        available = opened + pd.Timedelta(minutes=7)
        start = pd.Timestamp("2020-06-09T09:33:00Z")
        baseline = chronological_fit_mask(opened, target, available, start, cutoff)

        later_opened = pd.date_range(cutoff, periods=5, freq="min")
        changed = chronological_fit_mask(
            opened.append(later_opened),
            np.concatenate([target, np.ones(len(later_opened))]),
            available.append(later_opened + pd.Timedelta(minutes=7)),
            start,
            cutoff,
        )
        np.testing.assert_array_equal(changed[:len(opened)], baseline)
        self.assertFalse(changed[len(opened):].any())

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

    def test_last_closed_candle_at_decision_hour_and_day_boundaries(self):
        self.assertEqual(
            last_closed_candle_opened_at("2026-04-15 17:04:00+00:00"),
            pd.Timestamp("2026-04-15 17:03:00", tz="UTC"),
        )
        self.assertEqual(
            last_closed_candle_opened_at("2026-04-16 00:00:00+00:00"),
            pd.Timestamp("2026-04-15 23:59:00", tz="UTC"),
        )
        self.assertEqual(
            last_closed_candle_opened_at("2026-04-16 00:01:00+00:00"),
            pd.Timestamp("2026-04-16 00:00:00", tz="UTC"),
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
