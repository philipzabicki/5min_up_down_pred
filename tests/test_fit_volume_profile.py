import numpy as np
import pytest

import fit_volume_profile as volume_profile_fit


def test_build_fold_recency_weights_uses_linear_growth(monkeypatch):
    monkeypatch.setattr(volume_profile_fit, "ENABLE_FOLD_RECENCY_WEIGHTING", True)
    monkeypatch.setattr(volume_profile_fit, "FOLD_RECENCY_WEIGHTING_MODE", "linear")
    monkeypatch.setattr(volume_profile_fit, "FOLD_RECENCY_WEIGHT_MIN", 1.0)
    monkeypatch.setattr(volume_profile_fit, "FOLD_RECENCY_WEIGHT_MAX", 1.2)

    folds = [
        {"fold_id": 0},
        {"fold_id": 1},
        {"fold_id": 2},
    ]

    weights = volume_profile_fit.build_fold_recency_weights(folds)

    assert weights.index.tolist() == [0, 1, 2]
    assert weights.tolist() == pytest.approx([1.0, 1.1, 1.2])


def test_build_fold_recency_weights_returns_ones_when_disabled(monkeypatch):
    monkeypatch.setattr(volume_profile_fit, "ENABLE_FOLD_RECENCY_WEIGHTING", False)
    monkeypatch.setattr(volume_profile_fit, "FOLD_RECENCY_WEIGHT_MIN", 1.0)
    monkeypatch.setattr(volume_profile_fit, "FOLD_RECENCY_WEIGHT_MAX", 1.2)

    folds = [
        {"fold_id": 3},
        {"fold_id": 4},
        {"fold_id": 5},
    ]

    weights = volume_profile_fit.build_fold_recency_weights(folds)

    assert weights.index.tolist() == [3, 4, 5]
    assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_summarize_cv_fold_scores_uses_weighted_mean_when_active(monkeypatch):
    monkeypatch.setattr(volume_profile_fit, "ENABLE_FOLD_RECENCY_WEIGHTING", True)
    monkeypatch.setattr(volume_profile_fit, "FOLD_RECENCY_WEIGHT_MIN", 1.0)
    monkeypatch.setattr(volume_profile_fit, "FOLD_RECENCY_WEIGHT_MAX", 1.2)

    folds = [
        {"fold_id": 0},
        {"fold_id": 1},
        {"fold_id": 2},
    ]
    fold_scores = np.array([0.50, 0.60, 0.90], dtype=np.float64)
    fold_weight_by_id = volume_profile_fit.build_fold_recency_weights(folds)

    summary = volume_profile_fit.summarize_cv_fold_scores(
        fold_scores=fold_scores,
        folds=folds,
        fold_weight_by_id=fold_weight_by_id,
        std_penalty=0.5,
    )

    expected_weighted_mean = np.average(
        fold_scores,
        weights=[1.0, 1.1, 1.2],
    )
    expected_std = np.std(fold_scores)

    assert summary["cv_binary_logloss_mean"] == pytest.approx(np.mean(fold_scores))
    assert summary["cv_binary_logloss_weighted_mean"] == pytest.approx(
        expected_weighted_mean
    )
    assert summary["cv_binary_logloss_std"] == pytest.approx(expected_std)
    assert summary["objective_base_value"] == pytest.approx(expected_weighted_mean)
    assert summary["objective_value"] == pytest.approx(
        expected_weighted_mean + (0.5 * expected_std)
    )


def test_summarize_cv_fold_scores_uses_plain_mean_when_weighting_disabled(
    monkeypatch,
):
    monkeypatch.setattr(volume_profile_fit, "ENABLE_FOLD_RECENCY_WEIGHTING", False)
    monkeypatch.setattr(volume_profile_fit, "FOLD_RECENCY_WEIGHT_MIN", 1.0)
    monkeypatch.setattr(volume_profile_fit, "FOLD_RECENCY_WEIGHT_MAX", 1.2)

    folds = [
        {"fold_id": 0},
        {"fold_id": 1},
        {"fold_id": 2},
    ]
    fold_scores = np.array([0.50, 0.60, 0.90], dtype=np.float64)
    fold_weight_by_id = volume_profile_fit.build_fold_recency_weights(folds)

    summary = volume_profile_fit.summarize_cv_fold_scores(
        fold_scores=fold_scores,
        folds=folds,
        fold_weight_by_id=fold_weight_by_id,
        std_penalty=0.5,
    )

    expected_mean = np.mean(fold_scores)
    expected_std = np.std(fold_scores)

    assert summary["cv_binary_logloss_weighted_mean"] == pytest.approx(expected_mean)
    assert summary["objective_base_value"] == pytest.approx(expected_mean)
    assert summary["objective_value"] == pytest.approx(
        expected_mean + (0.5 * expected_std)
    )
