import unittest

import numpy as np
import pandas as pd

from features.btc_sequence_ohlcv import (
    decode_sequence,
    known_streak_state,
    sequence_targets,
    window_contains_flag,
)
from features.btc_series_nplus import align_outcomes


START = pd.Timestamp("2026-06-30T23:30:00Z")


def aligned_from(pattern, lags=None, start=START):
    starts = pd.date_range(start, periods=len(pattern), freq="5min", tz="UTC")
    outcome = np.array([{"U": 1.0, "D": 0.0, "?": np.nan}[c] for c in pattern])
    lags = [360.0] * len(pattern) if lags is None else lags
    available = [pd.NaT if lag is None else s + pd.Timedelta(seconds=lag) for s, lag in zip(starts, lags)]
    return align_outcomes(starts, outcome, available, calendar_start=start,
                          cutoff_exclusive=starts[-1] + pd.Timedelta(minutes=5))


class SequenceTargetTests(unittest.TestCase):
    def test_alignment_example_12_00_n3(self):
        start = pd.Timestamp("2026-06-01T12:00:00Z")
        labels = sequence_targets(aligned_from("UDUU", start=start), 3)
        first = labels.iloc[0]
        self.assertEqual(first["market_start_utc"], start)
        self.assertEqual(first["decision_at"], start - pd.Timedelta(seconds=60))
        self.assertEqual(decode_sequence(first["target"], 3), "UDU")
        self.assertEqual(decode_sequence(labels.iloc[1]["target"], 3), "DUU")
        # Last two windows run past the calendar end.
        self.assertEqual(labels["target"].tolist()[2:], [-1, -1])

    def test_gap_not_bridged_and_n1(self):
        starts = pd.date_range(START, periods=5, freq="5min", tz="UTC").delete(2)
        aligned = align_outcomes(starts, np.ones(4), starts + pd.Timedelta(minutes=6),
                                 calendar_start=START, cutoff_exclusive=START + pd.Timedelta(minutes=25))
        self.assertEqual(sequence_targets(aligned, 2)["target"].tolist(), [3, -1, -1, 3, -1])
        self.assertEqual(sequence_targets(aligned, 1)["target"].tolist(), [1, 1, -1, 1, 1])

    def test_availability_is_max_and_unknown_is_nat(self):
        labels = sequence_targets(aligned_from("UUUU", lags=[300, 9000, 300, 300]), 2)
        self.assertEqual(labels["label_available_at"].iloc[0],
                         START + pd.Timedelta(minutes=5) + pd.Timedelta(seconds=9000))
        labels = sequence_targets(aligned_from("UUUU", lags=[300, None, 300, 300]), 2)
        self.assertTrue(pd.isna(labels["label_available_at"].iloc[0]))
        self.assertEqual(labels["target"].iloc[0], 3)

    def test_known_streak_uses_only_outcomes_available_at_decision(self):
        # Outcomes resolve 6 minutes after start: slot T-5m is not known at T-60s.
        states = known_streak_state(aligned_from("UUUDDD")).tolist()
        self.assertEqual(states[:3], ["none", "none", "U1"])
        self.assertEqual(states[3], "U2")
        self.assertEqual(states[5], "D1")
        # A very late outcome is skipped and the run stops at it.
        late = known_streak_state(aligned_from("UUUU", lags=[360, 360, 99999, 360])).tolist()
        self.assertEqual(late[3], "U2")

    def test_window_contains_flag(self):
        flag = np.array([False, False, True, False, False])
        self.assertEqual(window_contains_flag(flag, 2).tolist(), [False, True, True, False, True])


if __name__ == "__main__":
    unittest.main()
