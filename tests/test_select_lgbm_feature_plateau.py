import numpy as np
import pandas as pd
import pytest

import select_lgbm_feature_plateau as feature_plateau


def test_build_fold_recency_weights_uses_linear_growth(monkeypatch):
    monkeypatch.setattr(feature_plateau, "ENABLE_FOLD_RECENCY_WEIGHTING", True)
    monkeypatch.setattr(feature_plateau, "FOLD_RECENCY_WEIGHTING_MODE", "linear")
    monkeypatch.setattr(feature_plateau, "FOLD_RECENCY_WEIGHT_MIN", 1.0)
    monkeypatch.setattr(feature_plateau, "FOLD_RECENCY_WEIGHT_MAX", 1.2)

    folds = [
        {"fold_id": 0},
        {"fold_id": 1},
        {"fold_id": 2},
    ]

    weights = feature_plateau.build_fold_recency_weights(folds)

    assert weights.index.tolist() == [0, 1, 2]
    assert weights.tolist() == pytest.approx([1.0, 1.1, 1.2])


def test_build_fold_recency_weights_returns_ones_when_disabled(monkeypatch):
    monkeypatch.setattr(feature_plateau, "ENABLE_FOLD_RECENCY_WEIGHTING", False)
    monkeypatch.setattr(feature_plateau, "FOLD_RECENCY_WEIGHT_MIN", 1.0)
    monkeypatch.setattr(feature_plateau, "FOLD_RECENCY_WEIGHT_MAX", 1.2)

    folds = [
        {"fold_id": 3},
        {"fold_id": 4},
        {"fold_id": 5},
    ]

    weights = feature_plateau.build_fold_recency_weights(folds)

    assert weights.index.tolist() == [3, 4, 5]
    assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_weighted_mean_helpers_respect_fold_order():
    folds = [
        {"fold_id": 2},
        {"fold_id": 0},
        {"fold_id": 1},
    ]
    fold_weight_by_id = pd.Series(
        [1.2, 1.0, 1.1],
        index=[2, 0, 1],
        dtype=np.float64,
    )

    resolved_weights = feature_plateau.resolve_fold_weight_array(
        folds,
        fold_weight_by_id,
    )

    assert resolved_weights.tolist() == pytest.approx([1.2, 1.0, 1.1])
    assert feature_plateau.weighted_mean_vector(
        [10.0, 1.0, 4.0],
        resolved_weights,
    ) == pytest.approx(np.average([10.0, 1.0, 4.0], weights=resolved_weights))

    table = np.array(
        [
            [1.0, 2.0, 3.0],
            [5.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    expected = np.average(table, axis=1, weights=resolved_weights)

    assert feature_plateau.weighted_mean_rows(table, resolved_weights) == pytest.approx(
        expected
    )


def test_choose_recommended_row_keeps_best_when_only_one_feature_is_saved(
    monkeypatch,
):
    monkeypatch.setattr(feature_plateau, "TOLERANCE_MODE", "abs")
    monkeypatch.setattr(feature_plateau, "ABS_TOL", 0.000005)
    monkeypatch.setattr(feature_plateau, "MIN_PLATEAU_FEATURE_SAVINGS", 2)

    results_df = pd.DataFrame(
        [
            {
                "k": 279,
                "mean_score": 0.6913159557746157,
                "weighted_mean_score": 0.6913572499554237,
                "selection_base_score": 0.6913572499554237,
                "std_score": 0.0011108386743386412,
                "selection_score": 0.6924680886297623,
            },
            {
                "k": 280,
                "mean_score": 0.6913269623379718,
                "weighted_mean_score": 0.6913674495195363,
                "selection_base_score": 0.6913674495195363,
                "std_score": 0.001097477156073263,
                "selection_score": 0.6924649266756095,
            },
        ]
    )

    best_row, recommended_row = feature_plateau.choose_recommended_row(results_df)

    assert int(best_row["k"]) == 280
    assert int(recommended_row["k"]) == 280


def test_choose_recommended_row_can_still_pick_smaller_plateau_candidate(
    monkeypatch,
):
    monkeypatch.setattr(feature_plateau, "TOLERANCE_MODE", "abs")
    monkeypatch.setattr(feature_plateau, "ABS_TOL", 0.000005)
    monkeypatch.setattr(feature_plateau, "MIN_PLATEAU_FEATURE_SAVINGS", 2)

    results_df = pd.DataFrame(
        [
            {
                "k": 278,
                "mean_score": 0.6913200,
                "weighted_mean_score": 0.6913605,
                "selection_base_score": 0.6913605,
                "std_score": 0.0011040,
                "selection_score": 0.6924690,
            },
            {
                "k": 279,
                "mean_score": 0.6913210,
                "weighted_mean_score": 0.6913615,
                "selection_base_score": 0.6913615,
                "std_score": 0.0011040,
                "selection_score": 0.6924675,
            },
            {
                "k": 280,
                "mean_score": 0.6913269623379718,
                "weighted_mean_score": 0.6913674495195363,
                "selection_base_score": 0.6913674495195363,
                "std_score": 0.001097477156073263,
                "selection_score": 0.6924649266756095,
            },
        ]
    )

    best_row, recommended_row = feature_plateau.choose_recommended_row(results_df)

    assert int(best_row["k"]) == 280
    assert int(recommended_row["k"]) == 278
