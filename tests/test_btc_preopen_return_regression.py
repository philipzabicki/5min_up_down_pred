import unittest

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from features.btc_preopen_contract import (
    RETURN_TARGET_COL,
    TARGET_COL,
    TARGET_END_PRICE_COL,
    TARGET_START_PRICE_COL,
    build_preopen_contract_frame,
)
from features.btc_preopen_return_target import (
    VOLATILITY_DENOMINATOR_FLOOR,
    build_preopen_sigma_5m,
    normalize_preopen_return,
)


class TestBtcPreopenReturnRegression(unittest.TestCase):
    def test_contract_return_uses_open_to_close_and_tie_is_up(self):
        opened = pd.date_range("2026-10-17T11:58:00Z", periods=10, freq="min")
        candles = pd.DataFrame({
            "Opened": opened,
            "Open": np.arange(100.0, 110.0),
            "Close": np.arange(100.0, 110.0),
        })
        frame = build_preopen_contract_frame(candles)
        row = frame.loc[frame.Opened.eq(pd.Timestamp("2026-10-17T11:58:00Z"))].iloc[0]
        self.assertEqual(row["nominal_decision_at"], pd.Timestamp("2026-10-17T11:59:00Z"))
        self.assertEqual(row["target_window_start_at"], pd.Timestamp("2026-10-17T12:00:00Z"))
        self.assertEqual(row["target_window_end_at"], pd.Timestamp("2026-10-17T12:05:00Z"))
        self.assertEqual(row[TARGET_START_PRICE_COL], 102.0)
        self.assertEqual(row[TARGET_END_PRICE_COL], 106.0)
        self.assertAlmostEqual(row[RETURN_TARGET_COL], 106.0 / 102.0 - 1.0)
        self.assertEqual(row[TARGET_COL], 1.0)

    def test_sigma_uses_only_trailing_closed_minute_returns_and_resets_on_gap(self):
        opened = pd.date_range("2026-10-17T00:00:00Z", periods=70, freq="min")
        close = 100.0 * np.power(1.001, np.arange(len(opened)))
        sigma = build_preopen_sigma_5m(opened, close)
        returns = close[1:61] / close[:60] - 1.0
        expected = np.std(returns, ddof=0) * np.sqrt(5.0)
        self.assertTrue(np.isnan(sigma[59]))
        self.assertAlmostEqual(sigma[60], expected, places=12)

        with_gap = opened.delete(40)
        gap_close = np.delete(close, 40)
        gap_sigma = build_preopen_sigma_5m(with_gap, gap_close)
        self.assertTrue(np.isnan(gap_sigma[60]))

    def test_zero_volatility_has_positive_denominator_and_missing_rows_stay_missing(self):
        result = normalize_preopen_return(
            np.array([0.0, 0.01, np.nan]), np.array([0.0, 0.0, 0.0])
        )
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[1], 0.01 / VOLATILITY_DENOMINATOR_FLOOR)
        self.assertTrue(np.isnan(result[2]))

    def test_huber_regression_and_probability_conversion_smoke(self):
        x = np.linspace(-2.0, 2.0, 256, dtype=np.float32).reshape(-1, 1)
        y = 0.4 * x[:, 0] + np.sin(x[:, 0]) * 0.02
        model = lgb.train(
            {
                "objective": "huber",
                "alpha": 0.9,
                "metric": "None",
                "verbosity": -1,
                "device_type": "cpu",
                "num_threads": 1,
                "seed": 7,
            },
            lgb.Dataset(x[:192], label=y[:192]),
            num_boost_round=20,
        )
        prediction = model.predict(x[64:192])
        labels = (y[64:192] >= 0.0).astype(np.int8)
        calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=7)
        calibrator.fit(prediction.reshape(-1, 1), labels)
        probability = calibrator.predict_proba(prediction.reshape(-1, 1))[:, 1]
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability >= 0.0) & (probability <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
