import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

import run_btc_market_basis_vol_interaction_20261017 as experiment


class OffsetOnceBooster:
    def __init__(self, correction):
        self.correction = np.asarray(correction, dtype=np.float64)
        self.calls = 0

    def predict(self, features, raw_score, num_threads):
        self.calls += 1
        if not raw_score or num_threads != 4:
            raise AssertionError("Prediction must request raw tree corrections")
        return self.correction


class BTCMarketBasisVolInteractionTests(unittest.TestCase):
    def test_market_offset_is_added_once_to_raw_tree_correction(self):
        market_logits = np.array([-1.25, 0.0, 2.0])
        frame = pd.DataFrame({feature: np.zeros(3) for feature in experiment.M0_FEATURES})
        frame["p_market"] = expit(market_logits)
        frame["logit_p_market"] = experiment._logit(frame.p_market)
        correction = np.array([0.4, -0.3, 0.15])
        booster = OffsetOnceBooster(correction)

        raw, probabilities = experiment._predict_offset(booster, frame, experiment.M0_FEATURES)

        np.testing.assert_array_equal(raw, correction)
        np.testing.assert_allclose(probabilities, expit(frame.logit_p_market.to_numpy() + correction), rtol=0.0, atol=0.0)
        self.assertEqual(booster.calls, 1)

    def test_m0_excludes_btc_model_prediction_and_disagreement(self):
        self.assertEqual(experiment.M0_FEATURES, (
            "logit_p_market", "futures_index_basis_rel_1m", "btc_sigma_5m", "spread_up", "spread_down",
        ))
        self.assertNotIn("d_btc_minus_market", experiment.M0_FEATURES)
        self.assertNotIn("logit_p_btc", experiment.M0_FEATURES)
        self.assertNotIn("p_btc_calibrated", experiment.M0_FEATURES)
        self.assertEqual(set(experiment.M1_FEATURES), set(experiment.M0_FEATURES) | {"d_btc_minus_market"})

    def test_comparisons_share_rows_and_respect_label_and_feature_boundaries(self):
        starts = pd.to_datetime([
            "2026-04-30T23:45:00Z", "2026-04-30T23:50:00Z",
            "2026-05-01T00:05:00Z", "2026-05-01T00:10:00Z",
        ], utc=True)
        fold_start = pd.Timestamp("2026-05-01T00:00:00Z")
        frame = pd.DataFrame({
            "market_start_utc": starts,
            "condition_id": ["a", "b", "c", "d"],
            "label_available_at_utc": pd.to_datetime([
                "2026-04-30T23:55:00Z", "2026-04-30T23:59:00Z",
                "2026-05-01T00:11:00Z", "2026-05-01T00:16:00Z",
            ], utc=True),
            "btc_feature_close_at": starts - pd.Timedelta(minutes=1),
            "official_outcome_up": [0, 1, 1, 0],
        })
        train, valid = experiment._fold_slices(frame, {
            "fold_id": "dev_2026q2_b",
            "start_inclusive_utc": fold_start.isoformat(),
            "end_exclusive_utc": "2026-06-01T00:00:00Z",
        })
        self.assertEqual(train.condition_id.tolist(), ["a", "b"])
        self.assertEqual(valid.condition_id.tolist(), ["c", "d"])
        self.assertTrue(valid.label_available_at_utc.ge(fold_start).all())

        predictions = pd.DataFrame({
            "market_start_utc": valid.market_start_utc,
            "condition_id": valid.condition_id,
            "p_M0": [0.51, 0.49],
            "p_M1": [0.52, 0.48],
        })
        compared = experiment._attach_candidate_predictions(valid, predictions)
        self.assertEqual(compared.condition_id.tolist(), ["c", "d"])
        self.assertEqual(len(compared), len(predictions))
        experiment._assert_fold_boundaries(train, valid, fold_start, pd.Timestamp("2026-06-01T00:00:00Z"))

    def test_block_bootstrap_discards_empty_only_calendar_resamples(self):
        starts = pd.date_range("2026-06-01T00:05:00Z", periods=14, freq="D").repeat(2)
        outcomes = np.tile(np.array([0, 1], dtype=np.int8), 14)
        frame = pd.DataFrame({
            "market_start_utc": starts,
            "official_outcome_up": outcomes,
            "p_left": np.where(outcomes == 1, 0.55, 0.45),
            "p_right": np.full(len(starts), 0.5),
        })

        result = experiment._paired_block_differences(frame, "p_left", "p_right")

        self.assertEqual(result["valid_bootstrap_replicates"], 2000)
        self.assertTrue(np.isfinite(result["log_loss_difference_95pct"]).all())
        self.assertTrue(np.isfinite(result["brier_difference_95pct"]).all())


if __name__ == "__main__":
    unittest.main()
