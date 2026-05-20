import unittest

import numpy as np

from features.candle_features import (
    SUPPORTED_CANDLE_FEATURE_COLS,
    build_candle_derived_features_from_series,
    is_deprecated_candle_feature_col,
)


class CandleFeatureTests(unittest.TestCase):
    def test_volume_dependent_candle_features_drop_absolute_volume_scale(self):
        features = build_candle_derived_features_from_series(
            open_=np.array([10.0, 10.0, 10.0, 10.0]),
            high=np.array([11.0, 10.0, 10.0, 11.0]),
            low=np.array([10.0, 9.0, 10.0, 10.0]),
            close=np.array([11.0, 9.0, 10.0, 11.0]),
            volume=np.array([100.0, 25.0, 999.0, 0.0]),
        )

        self.assertIn("candle_up_down_vol_log_ratio", features)
        self.assertNotIn("candle_up_down_vol_ratio", features)
        self.assertNotIn("candle_log_volume", features)
        np.testing.assert_allclose(
            features["candle_signed_vol"],
            np.array([1.0, -1.0, 0.0, 1.0]),
        )
        np.testing.assert_allclose(
            features["candle_up_down_vol_log_ratio"],
            np.array([1.0, -1.0, 0.0, 0.0]),
        )

    def test_up_down_volume_log_ratio_uses_supplied_volume_split_direction(self):
        features = build_candle_derived_features_from_series(
            open_=np.array([10.0, 10.0, 10.0]),
            high=np.array([11.0, 11.0, 11.0]),
            low=np.array([9.0, 9.0, 9.0]),
            close=np.array([11.0, 11.0, 9.0]),
            volume=np.array([30.0, 30.0, 30.0]),
            up_volume=np.array([20.0, 2.0, 5.0]),
            down_volume=np.array([10.0, 3.0, 5.0]),
        )

        np.testing.assert_allclose(
            features["candle_up_down_vol_log_ratio"],
            np.array([1.0, -1.0, 0.0]),
        )

    def test_supported_feature_names_use_log_ratio_name(self):
        supported = set(SUPPORTED_CANDLE_FEATURE_COLS)

        self.assertIn("candle_up_down_vol_log_ratio_1m", supported)
        self.assertIn("candle_up_down_vol_log_ratio_5m_lag1", supported)
        self.assertNotIn("candle_up_down_vol_ratio_1m", supported)
        self.assertNotIn("candle_log_volume_1m", supported)
        self.assertTrue(is_deprecated_candle_feature_col("candle_log_volume_1m"))
        self.assertTrue(is_deprecated_candle_feature_col("candle_log_volume_5m_lag2"))


if __name__ == "__main__":
    unittest.main()
