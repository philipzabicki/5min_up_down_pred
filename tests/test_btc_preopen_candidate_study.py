import unittest

import numpy as np
import pandas as pd

from run_btc_preopen_candidate_study import (
    _history_train_indices,
    _new_parameters,
    _train_weights,
    _weights,
)


class BtcPreopenCandidateStudyTests(unittest.TestCase):
    def test_decision_and_auxiliary_weights_sum_to_one_per_five_minute_cycle(self):
        decision = np.array([False, False, True, False, False])
        weights = _weights(decision, 0.23)

        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertAlmostEqual(float(weights[2]), 0.23)
        self.assertTrue(np.allclose(weights[[0, 1, 3, 4]], 0.1925))

    def test_history_windows_and_half_life_keep_only_eligible_train_rows(self):
        opened = pd.DatetimeIndex(
            ["2021-01-01", "2022-01-01", "2023-01-01", "2024-07-01", "2024-12-01"],
            tz="UTC",
        )
        indices = np.arange(5, dtype=np.int64)
        cutoff = pd.Timestamp("2025-01-01", tz="UTC")
        decision = np.array([True, False, True, False, True])

        self.assertEqual(
            _history_train_indices(opened, indices, cutoff, "last_1y").tolist(),
            [3, 4],
        )
        weighted_indices, weights = _train_weights(
            opened, decision, indices, cutoff, 0.5, "half_life_365d"
        )
        self.assertEqual(weighted_indices.tolist(), indices.tolist())
        self.assertAlmostEqual(float(weights.sum()), float(_weights(decision, 0.5).sum()))
        self.assertGreater(weights[0], 0.0)
        self.assertGreater(weights[-1], weights[0])

    def test_new_search_space_keeps_bagging_active_and_avoids_depth_and_sampling_redundancy(self):
        parameters = _new_parameters()

        self.assertEqual(parameters["bagging_freq"], 1)
        self.assertLess(parameters["bagging_fraction"], 1.0)
        self.assertEqual(parameters["max_depth"], -1)
        self.assertNotIn("feature_fraction_bynode", parameters)


if __name__ == "__main__":
    unittest.main()
