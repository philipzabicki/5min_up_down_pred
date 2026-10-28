import unittest

import numpy as np
import pandas as pd

from features.btc_taker_flow import compute_taker_flow_features
from run_btc_chronology_runtime_benchmark_20261028 import (
    FLOW_NAMES,
    FlowRuntimeState,
    _calibrate,
    assemble_model_matrix,
)


class BTCChronologyStreamingRuntimeTests(unittest.TestCase):
    @staticmethod
    def _flow_frame(row_count=3000, gap_after=None):
        rng = np.random.default_rng(20261028)
        returns = rng.normal(0.0, 0.0005, size=row_count)
        close = 60_000.0 * np.exp(np.cumsum(returns))
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.0002, size=row_count))
        low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.0002, size=row_count))
        quote = rng.lognormal(mean=8.0, sigma=0.35, size=row_count)
        taker = quote * rng.uniform(0.05, 0.95, size=row_count)
        opened = pd.date_range("2026-01-01", periods=row_count, freq="min", tz="UTC")
        if gap_after is not None:
            opened = opened.where(np.arange(row_count) < gap_after, opened + pd.Timedelta(minutes=1))
        return pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close,
            "quote_volume": quote, "taker_buy_quote_volume": taker,
        }, index=opened)

    def test_incremental_flow_matches_batch_including_gap_reset(self):
        frame = self._flow_frame(gap_after=1500)
        expected = compute_taker_flow_features(frame)
        state = FlowRuntimeState()
        rows = []
        for opened, row in frame.iterrows():
            values = state.update({"Opened": opened, **row.to_dict()})
            rows.append([opened, *[values[name] for name in FLOW_NAMES]])
        actual = pd.DataFrame(rows, columns=["Opened", *FLOW_NAMES]).set_index("Opened")
        np.testing.assert_allclose(
            actual.to_numpy(dtype=np.float64),
            expected.loc[actual.index, list(FLOW_NAMES)].to_numpy(dtype=np.float64),
            atol=2e-6,
            rtol=0.0,
            equal_nan=True,
        )
        self.assertTrue(actual.iloc[1500].loc[list(FLOW_NAMES)].isna().all())
        self.assertTrue(np.isfinite(actual.iloc[-1].loc[list(FLOW_NAMES)].to_numpy(dtype=np.float64)).all())

    def test_model_matrix_preserves_each_frozen_procedure_order(self):
        base_order = [f"base_{idx}" for idx in range(64)]
        memory_names = [f"memory_{idx}" for idx in range(8)]
        esn_names = {11: [f"esn11_{idx}" for idx in range(8)]}
        base = np.arange(64, dtype=np.float64)
        flow = {name: float(idx) for idx, name in enumerate(FLOW_NAMES)}
        memory = np.arange(8, dtype=np.float32)
        esn = {11: np.arange(8, dtype=np.float32) + 10}
        for model, seed, extra in (
            ("A", None, []),
            ("B", None, memory_names),
            ("C", 11, esn_names[11]),
            ("D", 11, memory_names + esn_names[11]),
        ):
            with self.subTest(model=model):
                matrix, order = assemble_model_matrix(
                    base, flow, memory, esn, base_order, memory_names, esn_names,
                    model, seed,
                )
                self.assertEqual(order, base_order + list(FLOW_NAMES) + extra)
                self.assertEqual(matrix.shape, (1, len(order)))

    def test_calibration_respects_frozen_no_fit_fallback(self):
        self.assertIsNone(_calibrate(0.6, {"status": "minimum_not_met"}))
        calibrated = _calibrate(0.5, {"status": "fitted", "intercept": 0.0, "coefficient": 2.0})
        self.assertAlmostEqual(calibrated, 0.5)


if __name__ == "__main__":
    unittest.main()
