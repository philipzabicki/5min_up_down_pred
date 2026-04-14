import numpy as np
import pytest

import optimize_lgbm_optuna as lgbm_optuna


def test_build_fold_recency_weights_uses_linear_growth(monkeypatch):
    monkeypatch.setattr(lgbm_optuna, "ENABLE_FOLD_RECENCY_WEIGHTING", True)
    monkeypatch.setattr(lgbm_optuna, "FOLD_RECENCY_WEIGHTING_MODE", "linear")
    monkeypatch.setattr(lgbm_optuna, "FOLD_RECENCY_WEIGHT_MIN", 1.0)
    monkeypatch.setattr(lgbm_optuna, "FOLD_RECENCY_WEIGHT_MAX", 1.4)

    folds = [
        {"fold_id": 0},
        {"fold_id": 1},
        {"fold_id": 2},
    ]

    weights = lgbm_optuna.build_fold_recency_weights(folds)

    assert weights.tolist() == pytest.approx([1.0, 1.2, 1.4])


def test_summarize_cv_fold_scores_uses_weighted_mean_when_active(monkeypatch):
    monkeypatch.setattr(lgbm_optuna, "ENABLE_FOLD_RECENCY_WEIGHTING", True)
    monkeypatch.setattr(lgbm_optuna, "FOLD_RECENCY_WEIGHT_MIN", 1.0)
    monkeypatch.setattr(lgbm_optuna, "FOLD_RECENCY_WEIGHT_MAX", 1.4)

    folds = [
        {"fold_id": 0},
        {"fold_id": 1},
        {"fold_id": 2},
    ]
    fold_scores = np.array([0.50, 0.60, 0.90], dtype=np.float64)
    fold_weight_by_id = lgbm_optuna.build_fold_recency_weights(folds)

    summary = lgbm_optuna.summarize_cv_fold_scores(
        fold_scores=fold_scores,
        folds=folds,
        fold_weight_by_id=fold_weight_by_id,
        base_metric="balanced_accuracy",
        higher_is_better=True,
        std_penalty=0.5,
    )

    expected_weighted_mean = np.average(
        fold_scores,
        weights=np.array([1.0, 1.2, 1.4], dtype=np.float64),
    )
    expected_std = np.std(fold_scores)

    assert summary["cv_balanced_accuracy_mean"] == pytest.approx(np.mean(fold_scores))
    assert summary["cv_balanced_accuracy_weighted_mean"] == pytest.approx(
        expected_weighted_mean
    )
    assert summary["cv_balanced_accuracy_std"] == pytest.approx(expected_std)
    assert summary["objective_base_value"] == pytest.approx(expected_weighted_mean)
    assert summary["objective_value"] == pytest.approx(
        expected_weighted_mean - (0.5 * expected_std)
    )
