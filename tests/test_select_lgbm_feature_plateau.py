import unittest

import numpy as np

import select_lgbm_feature_plateau as plateau


class SelectLgbmFeaturePlateauMetricTests(unittest.TestCase):
    def test_default_scorer_is_binary_logloss_minimized(self):
        self.assertEqual(plateau.SCORER["name"], "binary_logloss")
        self.assertFalse(plateau.SCORER["greater_is_better"])

    def test_topk_selection_penalizes_logloss_std_upward(self):
        score = plateau.topk_selection_score(mean_score=0.25, std_score=0.02)

        self.assertAlmostEqual(
            score,
            0.25 + plateau.TOPK_SELECTION_STD_COEF * 0.02,
        )

    def test_score_predictions_uses_weighted_binary_logloss(self):
        y_true = np.asarray([1, 0, 1], dtype=np.int8)
        y_pred_proba = np.asarray(
            [
                [0.2, 0.8],
                [0.75, 0.25],
                [0.4, 0.6],
            ],
            dtype=np.float64,
        )
        sample_weight = np.asarray([2.0, 1.0, 3.0], dtype=np.float64)

        score = plateau.score_predictions(
            scorer_cfg=plateau.SCORER,
            y_true=y_true,
            y_pred=None,
            y_pred_proba=y_pred_proba,
            sample_weight=sample_weight,
        )

        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
