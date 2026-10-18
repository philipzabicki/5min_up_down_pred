import numpy as np
import pandas as pd
import unittest

from features.btc_taker_flow import compute_taker_flow_features, preopen_market_alignment


def _klines(rows=9000):
    index = pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC")
    close = 50_000.0 * np.exp(np.cumsum(np.full(rows, 0.00001)))
    return pd.DataFrame(
        {
            "open": close / np.exp(0.00001),
            "high": close * 1.0001,
            "low": close * 0.9999,
            "close": close,
            "quote_volume": np.full(rows, 100_000.0),
            "taker_buy_quote_volume": np.full(rows, 55_000.0),
        },
        index=index,
    )


class TestBtcTakerFlow(unittest.TestCase):
    def test_same_ohlc_and_quote_volume_with_different_taker_buy_flow_changes_features(self):
        buy_heavy = _klines()
        sell_heavy = buy_heavy.copy()
        sell_heavy["taker_buy_quote_volume"] = 45_000.0
        first = compute_taker_flow_features(buy_heavy)
        second = compute_taker_flow_features(sell_heavy)
        at = first.index[-1]
        self.assertEqual(first.loc[at, "taker_V_1m"], second.loc[at, "taker_V_1m"])
        self.assertNotEqual(first.loc[at, "taker_I_1m"], second.loc[at, "taker_I_1m"])
        self.assertNotEqual(first.loc[at, "taker_U_1m"], second.loc[at, "taker_U_1m"])


    def test_changes_after_boundary_do_not_change_earlier_features(self):
        source = _klines()
        expected = compute_taker_flow_features(source)
        changed = source.copy()
        boundary = changed.index[7000]
        future = boundary + pd.Timedelta(minutes=1)
        changed.loc[future:, "close"] *= 1.25
        changed.loc[future:, "open"] *= 1.25
        changed.loc[future:, "high"] *= 1.25
        changed.loc[future:, "low"] *= 1.25
        changed.loc[future:, "taker_buy_quote_volume"] = 90_000.0
        actual = compute_taker_flow_features(changed)
        pd.testing.assert_frame_equal(expected.loc[:boundary], actual.loc[:boundary])


    def test_beta_excludes_the_current_horizon_window(self):
        source = _klines()
        changed = source.copy()
        row = 8000
        h = 15
        start = row - h + 1
        changed.iloc[start:row + 1, changed.columns.get_loc("taker_buy_quote_volume")] = 90_000.0
        first = compute_taker_flow_features(source)
        second = compute_taker_flow_features(changed)
        at = source.index[row]
        self.assertEqual(first.loc[at, "taker_beta_15m"], second.loc[at, "taker_beta_15m"])


    def test_gaps_and_zero_quote_volume_remain_missing_without_forward_fill(self):
        source = _klines()
        gap_at = source.index[5000]
        source = source.drop(index=gap_at)
        zero_at = source.index[8000]
        source.loc[zero_at, "quote_volume"] = 0.0
        source.loc[zero_at, "taker_buy_quote_volume"] = 0.0
        features = compute_taker_flow_features(source)
        self.assertTrue(features.loc[gap_at, [
            "taker_Z_1m", "taker_V_1m", "taker_I_1m", "taker_U_1m", "taker_A_1m",
        ]].isna().all())
        self.assertTrue(np.isnan(features.loc[zero_at, "taker_I_1m"]))
        self.assertEqual(features.loc[zero_at, "taker_V_1m"], 0.0)
        self.assertTrue(np.isnan(features.loc[gap_at + pd.Timedelta(minutes=1), "taker_U_5m"]))


    def test_noon_market_uses_1159_boundary_and_1200_to_1205_target_window(self):
        alignment = preopen_market_alignment(pd.Timestamp("2026-06-30T11:58:00Z"))
        self.assertEqual(alignment["information_boundary_at"], pd.Timestamp("2026-06-30T11:59:00Z"))
        self.assertEqual(alignment["market_start_at"], pd.Timestamp("2026-06-30T12:00:00Z"))
        self.assertEqual(alignment["market_end_at"], pd.Timestamp("2026-06-30T12:05:00Z"))
        self.assertEqual(alignment["target_label_available_at"], pd.Timestamp("2026-06-30T12:05:00Z"))
        prices = pd.DataFrame(
            {"Open": [100, 101, 102, 103, 104], "Close": [101, 102, 103, 104, 105]},
            index=pd.date_range("2026-06-30T12:00:00Z", periods=5, freq="min"),
        )
        target = prices.loc["2026-06-30T12:04:00Z", "Close"] >= prices.loc[
            "2026-06-30T12:00:00Z", "Open"
        ]
        self.assertTrue(bool(target))
