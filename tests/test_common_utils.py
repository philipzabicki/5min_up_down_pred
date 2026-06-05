import unittest

import numpy as np

from features.common_utils import (
    build_linear_recency_weights,
    extremes_vs_mid_ir_oof,
)


class LinearRecencyWeightTests(unittest.TestCase):
    def test_build_linear_recency_weights_matches_fold_style(self):
        weights = build_linear_recency_weights(
            4,
            enabled=True,
            min_weight=1.0,
            max_weight=1.5,
        )

        np.testing.assert_allclose(
            weights,
            np.asarray([1.0, 1.1666666667, 1.3333333333, 1.5]),
        )

    def test_disabled_linear_recency_weights_are_unit_weights(self):
        weights = build_linear_recency_weights(
            4,
            enabled=False,
            min_weight=1.0,
            max_weight=1.5,
        )

        np.testing.assert_allclose(weights, np.ones(4, dtype=np.float64))

    def test_extremes_vs_mid_ir_uses_recency_weighted_mean(self):
        x_parts = []
        y_parts = []
        for amplitude in (1.0, 2.0, 3.0, 4.0):
            x_segment = np.tile(np.arange(50, dtype=np.float64), 2)
            y_segment = np.zeros(100, dtype=np.float64)
            test_x = x_segment[50:]
            test_y = y_segment[50:]
            test_y[test_x < 10.0] = -amplitude
            test_y[test_x > 39.0] = amplitude
            y_segment[50:] = test_y
            x_parts.append(x_segment)
            y_parts.append(y_segment)

        metric_kwargs = {
            "segments_count": 4,
            "train_frac": 0.5,
            "gap": 0,
            "q_ext": 0.2,
            "q_mid": 0.1,
            "stat": "mean_clip",
            "clip_q": 0.0,
            "min_bucket_size": 5,
            "min_valid_segments": 2,
        }
        unweighted = extremes_vs_mid_ir_oof(
            np.concatenate(x_parts),
            np.concatenate(y_parts),
            **metric_kwargs,
        )
        weighted = extremes_vs_mid_ir_oof(
            np.concatenate(x_parts),
            np.concatenate(y_parts),
            **metric_kwargs,
            recency_weighting_enabled=True,
            recency_weight_min=1.0,
            recency_weight_max=1.5,
        )

        self.assertGreater(weighted, unweighted)


if __name__ == "__main__":
    unittest.main()
