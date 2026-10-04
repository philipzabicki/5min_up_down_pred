import unittest

import numpy as np

from features.ADX import get_adx_values
from features.live_indicator_runtime import (
    IndicatorFullHistoryScratch,
    IndicatorWindowScratch,
    get_adx_latest_value_live,
)
from features.live_indicator_runtime_entry_state_v1 import (
    get_adx_latest_value_full_history,
)


class EntryStateAdxRuntimeTests(unittest.TestCase):
    def test_mama_adx_uses_full_available_history(self):
        count = 5_000
        steps = np.arange(count, dtype=np.float64)
        center = 1_000.0 + 0.03 * steps + 3.0 * np.sin(steps / 31.0)
        high = center + 2.0
        low = center - 2.0
        close = high - 0.05
        close[:10] = low[:10] + 0.05
        ohlcv = np.column_stack(
            (center, high, low, close, 100.0 + 20.0 * np.sin(steps / 17.0) ** 2)
        )
        params = {
            "atr_period": 1_160,
            "posDM_period": 1_855,
            "negDM_period": 64,
            "adx_period": 142,
            "ma_type_atr": "DEMA",
            "ma_type_posDM": "FBA",
            "ma_type_negDM": "HMA",
            "ma_type_adx": "MAMA",
        }

        batch_value = float(get_adx_values(params, ohlcv)[-1])
        rolling_scratch = IndicatorWindowScratch(
            IndicatorFullHistoryScratch(ohlcv),
            window_len=1_000,
        )
        reset_value = get_adx_latest_value_live(params, rolling_scratch)
        fixed_value = get_adx_latest_value_full_history(params, ohlcv)

        self.assertGreater(abs(reset_value - batch_value), 1e-3)
        self.assertEqual(fixed_value, batch_value)


if __name__ == "__main__":
    unittest.main()
