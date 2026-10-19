import unittest

import numpy as np
import pandas as pd

from audit_btc_pmxt_empty_partitions_20261019 import _reference_selection_masks
from run_btc_market_microstructure_development_20261019 import _attach_daily_policy_counts
from run_btc_preopen_t59_reassessment import _fixed5_skip_reason


class BtcPmxtEmptyPartitionAuditTests(unittest.TestCase):
    def test_reference_filter_uses_exact_market_bytes_and_inclusive_receive_cutoff(self):
        market_a = b"0x" + b"a" * 64
        market_b = b"0x" + b"b" * 64
        market_values = np.asarray(
            [market_a, market_b, market_a, market_a.upper()], dtype=object,
        )
        cutoff = np.datetime64("2026-06-13T12:00:00.000", "ms")
        received = np.asarray(
            [
                "2026-06-13T12:00:00.000",
                "2026-06-13T12:00:00.001",
                "2026-06-13T11:59:59.999",
                "2026-06-13T12:00:00.000",
            ],
            dtype="datetime64[ms]",
        )

        market_mask, selected_mask, positions, cutoffs = _reference_selection_masks(
            market_values,
            received,
            {market_a, market_b},
            {market_a: cutoff, market_b: cutoff},
        )

        np.testing.assert_array_equal(market_mask, [True, True, True, False])
        np.testing.assert_array_equal(selected_mask, [True, False, True, False])
        np.testing.assert_array_equal(positions, [0, 1, 2])
        np.testing.assert_array_equal(
            cutoffs,
            np.asarray([cutoff, cutoff, cutoff], dtype="datetime64[ms]"),
        )

    def test_empty_available_snapshot_has_its_own_skip_reason(self):
        calendar = pd.DataFrame({
            "market_confirmed": [True, True, True],
            "causal_prediction_available": [True, True, True],
            "t59_snapshot_available": [True, True, False],
            "snapshot_market_event_count": [0, 0, np.nan],
            "archive_entry_hour_available": [True, False, False],
            "fixed5_priceable_side_count": [0, 0, 0],
            "fixed5_buy_eligible": [False, False, False],
        })

        np.testing.assert_array_equal(
            _fixed5_skip_reason(calendar),
            [
                "no_archive_events_observed_by_t59",
                "no_archive_events_observed_by_t59",
                "archive_hour_missing_at_entry",
            ],
        )

    def test_daily_counts_distinguish_no_data_from_no_execution(self):
        days = pd.to_datetime(["2026-06-12", "2026-06-13"], utc=True)
        availability = pd.DataFrame({
            "day_utc": days,
            "raw_oof_rows": [0, 2],
            "builder_eligible": [0, 1],
            "market_count": [0, 2],
            "available_input_count": [0, 1],
        })
        decisions = {
            variant: pd.DataFrame(columns=["market_start_utc"])
            for variant in ("C", "X", "MARKET", "RAW")
        }
        decisions["C"] = pd.DataFrame({
            "market_start_utc": pd.to_datetime(["2026-06-13T00:05:00Z"], utc=True),
        })
        trades = {
            variant: pd.DataFrame(columns=["market_start_utc"])
            for variant in ("C", "X", "MARKET", "RAW")
        }

        result = _attach_daily_policy_counts(availability, decisions, trades)

        self.assertEqual(result.data_status.tolist(), ["no_market_data", "decisions_without_executions"])
        self.assertEqual(result.C_decisions.tolist(), [0, 1])
        self.assertEqual(result.C_executions.tolist(), [0, 0])
        self.assertEqual(result.C_data_status.tolist(), ["no_market_data", "decisions_without_executions"])


if __name__ == "__main__":
    unittest.main()
