import unittest

import numpy as np
import pandas as pd

from features.btc_preopen_baseline import FEATURE_COLUMNS, build_causal_preopen_features
from run_btc_preopen_collection import HISTORY_ROWS, RawPreopenPredictor


class PreopenCollectionFeatureParityTests(unittest.TestCase):
    def test_live_snapshot_matches_batch_feature_builder_after_append(self):
        opened = pd.date_range("2026-01-01T00:00:00Z", periods=HISTORY_ROWS, freq="min")
        close = 100_000.0 + np.arange(HISTORY_ROWS, dtype=np.float64) * 0.25
        candles = pd.DataFrame(
            {
                "Opened": opened,
                "Open": close - 0.1,
                "High": close + 0.5,
                "Low": close - 0.4,
                "Close": close,
                "Volume": 10.0 + np.arange(HISTORY_ROWS, dtype=np.float64),
                "UM_BTCUSDT_Close": close + 1.0,
            }
        )
        predictor = RawPreopenPredictor(candles)
        expected = build_causal_preopen_features(candles).loc[:, FEATURE_COLUMNS].iloc[-1]
        np.testing.assert_allclose(
            predictor.build_feature_snapshot()["vector"][0],
            expected.to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )

        next_close = float(close[-1] + 0.25)
        predictor._append_new_candle(
            opened[-1] + pd.Timedelta(minutes=1),
            (next_close - 0.1, next_close + 0.5, next_close - 0.4, next_close, 300.0),
            next_close + 1.0,
        )
        expected_frame = pd.concat(
            [
                candles,
                pd.DataFrame(
                    [{
                        "Opened": opened[-1] + pd.Timedelta(minutes=1),
                        "Open": next_close - 0.1,
                        "High": next_close + 0.5,
                        "Low": next_close - 0.4,
                        "Close": next_close,
                        "Volume": 300.0,
                        "UM_BTCUSDT_Close": next_close + 1.0,
                    }]
                ),
            ],
            ignore_index=True,
        ).tail(HISTORY_ROWS)
        expected_latest = build_causal_preopen_features(expected_frame).loc[:, FEATURE_COLUMNS].iloc[-1]
        np.testing.assert_allclose(
            predictor.build_feature_snapshot()["vector"][0],
            expected_latest.to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
