import unittest

import numpy as np

import run_btc_conditional_market_offset_20261018 as experiment


class ConditionalMarketOffsetTests(unittest.TestCase):
    def test_zero_correction_reproduces_market_probability(self):
        p_market = np.array([0.001, 0.08, 0.35, 0.5, 0.91, 0.999], dtype=np.float64)

        actual = experiment._apply_offset_correction(p_market, np.zeros_like(p_market))

        np.testing.assert_allclose(actual, p_market, rtol=0.0, atol=2e-16)

    def test_brier_correction_decomposition_is_exact(self):
        y = np.array([1, 0, 0, 1, 1, 0], dtype=np.int8)
        p_market = np.array([0.55, 0.47, 0.6, 0.42, 0.68, 0.3])
        p_new = np.array([0.61, 0.45, 0.52, 0.51, 0.63, 0.34])

        result = experiment._brier_decomposition(y, p_market, p_new)

        self.assertTrue(np.isclose(
            result["delta_brier"], result["movement_cost"] - result["error_alignment"],
            rtol=0.0, atol=1e-15,
        ))

    def test_fixed_epsilon_keeps_boundary_logits_finite(self):
        values = experiment._logit(np.array([0.0, 1.0]))

        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.allclose(values, -values[::-1], rtol=0.0, atol=1e-9))

    def test_peak_working_set_sampler_returns_positive_bytes(self):
        working_set = experiment._peak_working_set_bytes()

        self.assertIsNotNone(working_set)
        self.assertGreater(working_set, 0)


if __name__ == "__main__":
    unittest.main()
