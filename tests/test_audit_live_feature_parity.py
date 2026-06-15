import json
import tempfile
import unittest
from pathlib import Path
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

        with tempfile.TemporaryDirectory() as tmpdir:
            requirements_path = Path(tmpdir) / "requirements.json"
            requirements_path.write_text(
                json.dumps({"unstable_features": []}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    audit,
                    "INDICATOR_HISTORY_REQUIREMENTS_PATH",
                    requirements_path,
                ),
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


class LiveFeatureParityOutputTests(unittest.TestCase):
    def test_features_to_inspect_keeps_only_prediction_impact_columns(self):
        feature_summary_df = pd.DataFrame(
            {
                "feature": [
                    "signal_feature",
                    "drift_feature",
                    "medium_feature",
                    "raw_diff_only_feature",
                ],
                "rows_pred_shift_gt_tol_if_fixed": [3, 2, 1, 0],
                "mean_abs_proba_shift_on_shift_rows_if_fixed": [
                    0.020,
                    0.010,
                    0.005,
                    0.0,
                ],
                "max_abs_proba_shift_if_fixed": [0.040, 0.030, 0.006, 0.0],
                "rows_proba_diff_gt_tol_resolved_if_fixed": [1, 2, 0, 0],
                "rows_signal_mismatch_resolved_if_fixed": [1, 0, 0, 0],
                "max_abs_diff": [0.5, 0.4, 0.3, 9.9],
                "mean_abs_diff": [0.05, 0.04, 0.03, 0.99],
                "importance_gain": [10.0, 20.0, 30.0, 40.0],
                "builder": ["unused", "unused", "unused", "unused"],
                "group": ["unused", "unused", "unused", "unused"],
                "net_pred_gap_reduction": [1.0, 1.0, 1.0, 1.0],
            }
        )

        result = audit._build_features_to_inspect_df(
            feature_summary_df,
            decision_row_count=10,
        )

        self.assertEqual(
            list(result.columns),
            audit.FEATURES_TO_INSPECT_COLUMNS,
        )
        self.assertEqual(
            result["feature"].tolist(),
            ["signal_feature", "drift_feature", "medium_feature"],
        )
        self.assertEqual(result["severity"].tolist(), ["critical", "high", "medium"])
        self.assertAlmostEqual(float(result.loc[0, "pred_shift_rows_pct"]), 30.0)
        self.assertNotIn("builder", result.columns)
        self.assertNotIn("group", result.columns)
        self.assertNotIn("net_pred_gap_reduction", result.columns)

    def test_summary_payload_is_short_and_feature_focused(self):
        features_to_inspect_df = pd.DataFrame(
            {
                "rank": [1],
                "severity": ["high"],
                "feature": ["drift_feature"],
                "pred_shift_rows": [2],
                "pred_shift_rows_pct": [20.0],
                "mean_pred_shift": [0.01],
                "max_pred_shift": [0.03],
                "rows_where_prediction_diff_exceeds_tol_explained": [2],
                "rows_where_up_down_prediction_flips_explained": [0],
                "max_feature_abs_diff": [0.4],
                "mean_feature_abs_diff": [0.04],
                "importance_gain": [20.0],
            }
        )
        report = {
            "summary": pd.Series(
                {
                    "audit_start": "2026-05-09T09:19:00",
                    "audit_end": "2026-05-16T09:18:00",
                    "bootstrap_rows": 21600,
                    "audit_rows_total_1m": 10080,
                    "decision_row_count": 10,
                    "feature_count": 124,
                    "rows_with_proba_diff_gt_tol": 2,
                    "max_proba_up_abs_diff": 0.03,
                    "mean_proba_up_abs_diff": 0.001,
                    "rows_with_signal_mismatch": 0,
                    "rows_with_business_decision_mismatch": 0,
                    "rows_with_any_policy_mismatch": 0,
                }
            )
        }
        drift_reason_report = {
            "summary": pd.Series({"explanation_basis": "proba_diff_gt_tol"})
        }

        payload = audit._build_live_feature_parity_summary_payload(
            report,
            drift_reason_report,
            features_to_inspect_df,
        )

        self.assertEqual(payload["verdict"], "inspect")
        self.assertEqual(payload["features_to_inspect"], 1)
        self.assertEqual(payload["top_features"][0]["feature"], "drift_feature")
        self.assertNotIn("live_vs_stored", payload)
        self.assertNotIn("top10_feature_drop_candidates", payload)
        self.assertNotIn("feature_drop_candidate_thresholds", payload)

    def test_default_save_outputs_writes_only_main_csvs(self):
        feature_summary_df = pd.DataFrame(
            {
                "feature": ["drift_feature"],
                "rows_pred_shift_gt_tol_if_fixed": [2],
                "mean_abs_proba_shift_on_shift_rows_if_fixed": [0.01],
                "max_abs_proba_shift_if_fixed": [0.03],
                "rows_proba_diff_gt_tol_resolved_if_fixed": [2],
                "rows_signal_mismatch_resolved_if_fixed": [0],
                "max_abs_diff": [0.4],
                "mean_abs_diff": [0.04],
                "importance_gain": [20.0],
            }
        )
        step_summary_df = pd.DataFrame(
            {
                "Opened": ["2026-05-09T09:20:00"],
                "live_proba_up": [0.55],
                "stored_proba_up": [0.52],
                "proba_up_abs_diff": [0.03],
                "signal_mismatch": [0],
                "business_decision_mismatch": [0],
                "policy_decision_mismatch": [0],
                "top_prediction_impact_feature": ["drift_feature"],
                "top_prediction_impact_abs_proba_shift_if_fixed": [0.03],
                "top_prediction_impact_live_value": [1.2],
                "top_prediction_impact_stored_value": [1.1],
                "feature_max_abs_diff": [0.4],
                "feature_mean_abs_diff": [0.04],
            }
        )
        results = {
            "live_vs_stored_report": {
                "summary": pd.Series(
                    {
                        "audit_start": "2026-05-09T09:19:00",
                        "audit_end": "2026-05-16T09:18:00",
                        "bootstrap_rows": 21600,
                        "audit_rows_total_1m": 10080,
                        "decision_row_count": 1,
                        "feature_count": 1,
                        "rows_with_proba_diff_gt_tol": 1,
                        "max_proba_up_abs_diff": 0.03,
                        "mean_proba_up_abs_diff": 0.03,
                        "rows_with_signal_mismatch": 0,
                        "rows_with_business_decision_mismatch": 0,
                        "rows_with_any_policy_mismatch": 0,
                    }
                ),
                "feature_summary_df": feature_summary_df,
                "step_summary_df": step_summary_df,
            },
            "drift_reason_report": {
                "summary": pd.Series({"explanation_basis": "proba_diff_gt_tol"})
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            written = audit.save_audit_outputs(results, output_dir=Path(tmpdir))
            csv_names = sorted(path.name for path in Path(tmpdir).glob("*.csv"))
            summary = json.loads(
                (Path(tmpdir) / "live_vs_stored_summary.json").read_text()
            )

        self.assertEqual(
            csv_names,
            ["features_to_inspect.csv", "rows_to_inspect.csv"],
        )
        self.assertEqual(
            sorted(path.name for path in written.values() if path.suffix == ".csv"),
            ["features_to_inspect.csv", "rows_to_inspect.csv"],
        )
        self.assertEqual(summary["features_to_inspect"], 1)
        self.assertEqual(summary["top_features"][0]["feature"], "drift_feature")


if __name__ == "__main__":
    unittest.main()
