import unittest

import numpy as np

from utils.metrics import (
    make_lightgbm_binary_logloss_eval,
    make_sklearn_binary_logloss_eval,
    weighted_binary_logloss,
)


class _DummyLightGBMDataset:
    def __init__(self, label, weight):
        self._label = np.asarray(label, dtype=np.float64)
        self._weight = np.asarray(weight, dtype=np.float64)

    def get_label(self):
        return self._label

    def get_weight(self):
        return self._weight


class BinaryLoglossMetricTests(unittest.TestCase):
    def test_weighted_binary_logloss_matches_manual_average(self):
        y_true = np.asarray([1, 0, 1], dtype=np.float64)
        y_pred = np.asarray([0.8, 0.25, 0.6], dtype=np.float64)
        weights = np.asarray([2.0, 1.0, 3.0], dtype=np.float64)
        manual_loss = -(
                y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
        )
        expected = float(np.average(manual_loss, weights=weights))

        actual = weighted_binary_logloss(y_true, y_pred, sample_weight=weights)

        self.assertAlmostEqual(actual, expected)

    def test_lightgbm_logloss_eval_is_minimized(self):
        metric_name, value, higher_is_better = make_lightgbm_binary_logloss_eval()(
            np.asarray([0.8, 0.25], dtype=np.float64),
            _DummyLightGBMDataset(label=[1, 0], weight=[1.0, 2.0]),
        )

        self.assertEqual(metric_name, "binary_logloss")
        self.assertGreater(value, 0.0)
        self.assertFalse(higher_is_better)

    def test_sklearn_logloss_eval_is_minimized(self):
        metric_name, value, higher_is_better = make_sklearn_binary_logloss_eval()(
            np.asarray([1, 0], dtype=np.float64),
            np.asarray([0.8, 0.25], dtype=np.float64),
            sample_weight=np.asarray([1.0, 2.0], dtype=np.float64),
        )

        self.assertEqual(metric_name, "binary_logloss")
        self.assertGreater(value, 0.0)
        self.assertFalse(higher_is_better)


if __name__ == "__main__":
    unittest.main()
