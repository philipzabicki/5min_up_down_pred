import unittest

import numpy as np

from features.candle_features import (
    SUPPORTED_CANDLE_FEATURE_COLS,
    build_candle_derived_features_from_series,
)


class CandleFeatureTests(unittest.TestCase):
    def test_up_down_volume_log_ratio_is_bounded_for_zero_down_volume(self):
        features = build_candle_derived_features_from_series(
            open_=np.array([10.0, 10.0, 10.0]),
            high=np.array([11.0, 10.0, 10.0]),
            low=np.array([10.0, 9.0, 10.0]),
            close=np.array([11.0, 9.0, 10.0]),
            volume=np.array([100.0, 25.0, 0.0]),
        )

        self.assertIn("candle_up_down_vol_log_ratio", features)
        self.assertNotIn("candle_up_down_vol_ratio", features)
        np.testing.assert_allclose(
            features["candle_up_down_vol_log_ratio"],
            np.array([np.log1p(100.0), -np.log1p(25.0), 0.0]),
        )

    def test_supported_feature_names_use_log_ratio_name(self):
        supported = set(SUPPORTED_CANDLE_FEATURE_COLS)

        self.assertIn("candle_up_down_vol_log_ratio_1m", supported)
        self.assertIn("candle_up_down_vol_log_ratio_5m_lag1", supported)
        self.assertNotIn("candle_up_down_vol_ratio_1m", supported)


if __name__ == "__main__":
    unittest.main()
