import unittest
from unittest import mock

import pandas as pd

import audit_feature_readiness as audit


class PseudoLiveAuditPredictorTests(unittest.TestCase):
    def test_basis_premium_features_are_stored_only(self):
        bootstrap_df = pd.DataFrame(
            {
                "Opened": pd.date_range(
                    "2026-01-01 00:00:00",
                    periods=2,
                    freq="min",
                    tz="UTC",
                ),
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.5, 101.5],
                "Volume": [10.0, 11.0],
            }
        )
        meta = {
            "feature_columns": [
                "Open",
                "Close",
                "futures_index_basis_rel_1m",
            ],
            "target_col": "target_5m_candle_up",
        }
        requirements = {
            "global_required_runtime_window": 1,
            "global_required_stable_window": 1,
            "stable_window_by_feature": {},
            "runtime_window_by_feature": {},
        }

        with (
            mock.patch.object(
                audit,
                "load_model_and_meta",
                return_value=(object(), meta),
            ),
            mock.patch.object(
                audit,
                "load_trade_policy_runtime_config",
                return_value={},
            ),
            mock.patch.object(audit, "load_indicator_specs", return_value=[]),
            mock.patch.object(
                audit,
                "load_indicator_history_requirements",
                return_value=requirements,
            ),
        ):
            predictor = audit.PseudoLiveAuditPredictor(
                bootstrap_df,
                model_meta_path="unused.json",
                max_keep=10,
            )

        self.assertIn("futures_index_basis_rel_1m", predictor.feature_columns)
        self.assertEqual(predictor.basis_premium_feature_columns, ())

        predictor._append_new_candle(
            pd.Timestamp("2026-01-01 00:02:00", tz="UTC"),
            (102.0, 103.0, 101.0, 102.5, 12.0),
        )

        self.assertIsNone(predictor.basis_futures_close_np)
        self.assertEqual(len(predictor.opened_candles), 3)


if __name__ == "__main__":
    unittest.main()
