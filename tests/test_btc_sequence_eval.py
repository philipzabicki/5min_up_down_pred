import unittest

import numpy as np
import pandas as pd

from utils.btc_sequence_eval import (
    apply_temperature,
    block_bootstrap_mean_diff,
    conditional_benchmark,
    fit_temperature,
    global_benchmark,
    per_window_scores,
)


class SequenceEvalTests(unittest.TestCase):
    def test_scores(self):
        prob = np.array([[0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])
        s = per_window_scores(prob, np.array([0, 3]))
        np.testing.assert_allclose(s["log_loss"], [-np.log(0.7), -np.log(0.25)])
        np.testing.assert_allclose(s["brier"][0], 0.3 ** 2 + 3 * 0.01)
        self.assertEqual(s["hit"][0], 1.0)

    def test_temperature_recovers_overconfidence(self):
        rng = np.random.default_rng(1)
        true = rng.dirichlet(np.ones(4) * 2, size=20000)
        target = np.array([rng.choice(4, p=p) for p in true])
        sharpened = apply_temperature(true, 0.5)  # overconfident by temperature 0.5
        t = fit_temperature(sharpened, target)
        self.assertAlmostEqual(t, 2.0, delta=0.15)
        np.testing.assert_allclose(apply_temperature(true, 1.0), true, atol=1e-12)

    def test_benchmarks(self):
        g = global_benchmark(np.array([0, 0, 1]), 2)
        np.testing.assert_allclose(g, [3 / 5, 2 / 5])
        cond, table, prior = conditional_benchmark(np.array([0, 0, 1, 1]), np.array(["U1", "U1", "D1", "D1"]),
                                                   np.array(["U1", "new"]), 2, prior_strength=2.0)
        np.testing.assert_allclose(cond[1], prior)
        self.assertGreater(cond[0][0], 0.5)

    def test_bootstrap_paired_constant_difference(self):
        days = pd.date_range("2026-05-15", periods=17 * 288, freq="5min", tz="UTC")
        res = block_bootstrap_mean_diff(np.full(len(days), -0.01), days, block_days=3, reps=200, seed=3)
        self.assertAlmostEqual(res["ci_low"], -0.01)
        self.assertAlmostEqual(res["ci_high"], -0.01)
        self.assertEqual((res["days"], res["blocks_per_replicate"]), (17, 6))


if __name__ == "__main__":
    unittest.main()
