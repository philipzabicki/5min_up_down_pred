import unittest

from fit_indicators import _resolve_metric_configs


class FitIndicatorMetricConfigTests(unittest.TestCase):
    def test_quantile_pairs_are_not_expanded_as_cartesian_product(self):
        metrics = _resolve_metric_configs(
            {},
            {
                "quantile_pairs": [
                    {"q_ext": 0.10, "q_mid": 0.15},
                    {"q_ext": 0.20, "q_mid": 0.20},
                ],
            },
        )

        self.assertEqual(
            [(metric["q_ext"], metric["q_mid"]) for metric in metrics],
            [(0.10, 0.15), (0.20, 0.20)],
        )

    def test_rejects_overlapping_quantile_pair(self):
        with self.assertRaisesRegex(ValueError, "overlap extremes"):
            _resolve_metric_configs(
                {},
                {
                    "quantile_pairs": [
                        {"q_ext": 0.20, "q_mid": 0.30},
                    ],
                },
            )


if __name__ == "__main__":
    unittest.main()
