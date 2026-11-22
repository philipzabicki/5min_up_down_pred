import unittest

import numpy as np
import pandas as pd

from features.btc_series_nplus import (
    STATUS_COMPLETE,
    STATUS_CROSSES_CUTOFF,
    STATUS_MISSING_OUTCOME,
    align_outcomes,
    binance_proxy_outcomes,
    build_series_labels,
    fit_eligible,
)


START = pd.Timestamp("2026-06-30T23:30:00Z")
CUTOFF = pd.Timestamp("2026-07-01T00:00:00Z")


def aligned_from(pattern, *, start=START, cutoff=None, lag_seconds=None):
    """'U'/'D' outcomes, '?' missing; one market per slot from ``start``."""
    starts = pd.date_range(start, periods=len(pattern), freq="5min", tz="UTC")
    outcome = np.array([{"U": 1.0, "D": 0.0, "?": np.nan}[c] for c in pattern])
    lags = [360.0] * len(pattern) if lag_seconds is None else lag_seconds
    available = [
        pd.NaT if lag is None else s + pd.Timedelta(seconds=lag)
        for s, lag in zip(starts, lags)
    ]
    end = starts[-1] + pd.Timedelta(minutes=5) if cutoff is None else cutoff
    return align_outcomes(
        starts, outcome, available, calendar_start=start, cutoff_exclusive=end
    )


def first_label(pattern, n):
    return build_series_labels(aligned_from(pattern), n)["label"].iloc[0]


class SeriesLabelTests(unittest.TestCase):
    def test_instruction_table(self):
        cases = {
            "UUUUUD": (1.0, 1.0),
            "DDDDDD": (1.0, 1.0),
            "UUUDUU": (1.0, 0.0),
            "UDUUUU": (0.0, 0.0),
            "UUUUU?": (1.0, 1.0),
        }
        for pattern, (y3, y5) in cases.items():
            with self.subTest(pattern=pattern):
                self.assertEqual(first_label(pattern, 3), y3)
                self.assertEqual(first_label(pattern, 5), y5)
        self.assertTrue(np.isnan(first_label("UU?UUU", 3)))
        self.assertTrue(np.isnan(first_label("UU?UUU", 5)))

    def test_exact_and_longer_series_both_directions(self):
        for side in "UD":
            other = "D" if side == "U" else "U"
            exact = side * 4 + other
            longer = side * 7
            self.assertEqual(first_label(exact, 4), 1.0)
            self.assertEqual(first_label(exact, 5), 0.0)
            labels = build_series_labels(aligned_from(longer), 4)
            self.assertEqual(labels["label"].iloc[:4].tolist(), [1.0] * 4)

    def test_gap_is_not_bridged(self):
        # Slot 2 has no market at all; the next existing rows must not shift in.
        starts = pd.date_range(START, periods=6, freq="5min", tz="UTC").delete(2)
        aligned = align_outcomes(
            starts,
            np.ones(5),
            starts + pd.Timedelta(minutes=6),
            calendar_start=START,
            cutoff_exclusive=START + pd.Timedelta(minutes=30),
        )
        labels = build_series_labels(aligned, 3)
        self.assertEqual(
            labels["status"].tolist()[:4],
            [STATUS_MISSING_OUTCOME] * 3 + [STATUS_COMPLETE],
        )
        self.assertTrue(labels["label"].iloc[:3].isna().all())

    def test_windows_crossing_cutoff_and_off_grid(self):
        labels = build_series_labels(aligned_from("U" * 6, cutoff=CUTOFF), 3)
        self.assertEqual(labels["market_start_utc"].iloc[-1], CUTOFF - pd.Timedelta(minutes=5))
        self.assertEqual(labels["status"].tolist()[-2:], [STATUS_CROSSES_CUTOFF] * 2)
        self.assertTrue((labels["decision_at"] == labels["market_start_utc"] - pd.Timedelta(seconds=60)).all())
        with self.assertRaises(ValueError):
            align_outcomes(
                [START + pd.Timedelta(minutes=1)], [1.0], [CUTOFF],
                calendar_start=START, cutoff_exclusive=CUTOFF,
            )

    def test_holdout_rows_are_dropped(self):
        starts = pd.date_range(CUTOFF - pd.Timedelta(minutes=10), periods=4, freq="5min", tz="UTC")
        aligned = align_outcomes(
            starts, np.ones(4), starts + pd.Timedelta(minutes=6),
            calendar_start=starts[0], cutoff_exclusive=CUTOFF,
        )
        self.assertEqual(len(aligned), 2)
        self.assertTrue((aligned.index < CUTOFF).all())

    def test_duplicate_start_is_ambiguous(self):
        starts = pd.DatetimeIndex([START, START, START + pd.Timedelta(minutes=5)])
        aligned = align_outcomes(
            starts, [1.0, 1.0, 1.0], starts + pd.Timedelta(minutes=6),
            calendar_start=START, cutoff_exclusive=START + pd.Timedelta(minutes=10),
        )
        self.assertEqual(aligned.attrs["ambiguous_starts"], 1)
        self.assertTrue(np.isnan(aligned["outcome_up"].iloc[0]))

    def test_availability_is_max_over_components_for_both_classes(self):
        lags = [360, 9000, 330, 330, 330, 330]
        for pattern in ("UUUUUU", "UUDUUU"):
            labels = build_series_labels(aligned_from(pattern, lag_seconds=lags), 3)
            expected = START + pd.Timedelta(minutes=5) + pd.Timedelta(seconds=9000)
            self.assertEqual(labels["label_available_at"].iloc[0], expected)

    def test_late_outcome_at_fold_boundary_is_purged(self):
        lags = [330, 330, 7200, 330, 330, 330]
        labels = build_series_labels(aligned_from("UUUUUU", lag_seconds=lags), 3)
        boundary = START + pd.Timedelta(minutes=40)
        eligible = fit_eligible(labels, boundary)
        # Windows containing slot 2 (23:40) become available only after 01:40.
        self.assertEqual(eligible[:4].tolist(), [False, False, False, True])

    def test_unknown_availability_is_never_eligible(self):
        lags = [330, None, 330, 330, 330, 330]
        labels = build_series_labels(aligned_from("UUUUUU", lag_seconds=lags), 2)
        self.assertEqual(labels["label"].iloc[0], 1.0)
        self.assertTrue(pd.isna(labels["label_available_at"].iloc[0]))
        self.assertFalse(fit_eligible(labels, CUTOFF + pd.Timedelta(days=1))[0])

    def test_longer_series_label_never_exceeds_shorter(self):
        rng = np.random.default_rng(7)
        pattern = "".join(rng.choice(list("UUDD?"), size=400))
        aligned = aligned_from(pattern)
        previous = build_series_labels(aligned, 2)["label"].to_numpy()
        for n in range(3, 8):
            current = build_series_labels(aligned, n)["label"].to_numpy()
            both = np.isfinite(previous) & np.isfinite(current)
            self.assertFalse((current[both] > previous[both]).any())
            previous = current

    def test_earlier_labels_unchanged_by_later_data(self):
        full = build_series_labels(aligned_from("UUDUUUDDDU"), 3)
        prefix = build_series_labels(aligned_from("UUDUUUD"), 3)
        done = prefix["status"] == STATUS_COMPLETE
        pd.testing.assert_frame_equal(prefix[done], full.iloc[: len(prefix)][done])

    def test_binance_proxy_uses_preopen_contract(self):
        opened = pd.date_range("2026-01-01T00:00:00Z", periods=20, freq="min", tz="UTC")
        close = np.array([100.0 + (i // 5 % 2) * (-1) ** (i // 5) for i in range(20)])
        candles = pd.DataFrame({"Opened": opened, "Open": 100.0, "Close": close})
        out = binance_proxy_outcomes(candles)
        self.assertTrue((out["market_start_utc"].dt.minute % 5 == 0).all())
        first = out.iloc[0]
        self.assertEqual(first["market_start_utc"], pd.Timestamp("2026-01-01T00:05:00Z"))
        # Close[00:09] = 99 < Open[00:05] = 100 -> DOWN; ties resolve UP elsewhere.
        self.assertEqual(first["outcome_up"], 0.0)
        self.assertEqual(first["outcome_available_at"], pd.Timestamp("2026-01-01T00:10:00Z"))


if __name__ == "__main__":
    unittest.main()
