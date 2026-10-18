import tempfile
import unittest
from datetime import date
from pathlib import Path

import run_btc_base70_raw_holdout_20261018 as experiment


class FrozenRawHoldoutTests(unittest.TestCase):
    def test_model_specs_preserve_frozen_feature_sets_and_iterations(self):
        base64_features = [f"base_{index}" for index in range(64)]
        base70_features = base64_features + [
            "taker_Z_1m", "taker_V_1m", "taker_Z_5m", "taker_V_5m", "taker_Z_15m", "taker_V_15m",
        ]
        params = experiment.CONFIG["models"]["BASE64"]["effective_lightgbm_params"]
        specs = experiment._model_specs(
            {"feature_order": base64_features, "params": params, "best_iteration": 25},
            {"feature_order": base70_features, "params": params, "best_iteration": 31},
        )
        self.assertEqual(specs["BASE64"]["feature_order"], base64_features)
        self.assertEqual(specs["BASE64"]["iterations"], 25)
        self.assertEqual(specs["BASE70"]["feature_order"], base70_features)
        self.assertEqual(specs["BASE70"]["iterations"], 31)

    def test_missing_daily_klines_are_reported_for_each_holdout_date(self):
        with tempfile.TemporaryDirectory() as temporary:
            days_root = Path(temporary)
            for day in ("2026-10-02", "2026-10-04"):
                directory = days_root / day
                directory.mkdir()
                (directory / f"BTCUSDT-1m-{day}.csv").touch()

            missing = experiment._missing_holdout_flow_files(
                days_root,
                start=date(2026, 10, 2),
                end_inclusive=date(2026, 10, 6),
            )

        self.assertEqual(
            missing,
            [
                (days_root / "2026-10-03" / "BTCUSDT-1m-2026-10-03.csv").as_posix(),
                (days_root / "2026-10-05" / "BTCUSDT-1m-2026-10-05.csv").as_posix(),
                (days_root / "2026-10-06" / "BTCUSDT-1m-2026-10-06.csv").as_posix(),
            ],
        )


if __name__ == "__main__":
    unittest.main()
