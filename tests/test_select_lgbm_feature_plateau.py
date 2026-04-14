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


def test_score_topk_subset_uses_weighted_mean_for_selection_base_when_enabled(
    monkeypatch,
):
    monkeypatch.setattr(feature_plateau, "RANDOM_SEEDS", [37])
    monkeypatch.setattr(feature_plateau, "ENABLE_FOLD_RECENCY_WEIGHTING", True)
    monkeypatch.setattr(feature_plateau, "FOLD_RECENCY_WEIGHT_MIN", 1.0)
    monkeypatch.setattr(feature_plateau, "FOLD_RECENCY_WEIGHT_MAX", 1.4)
    monkeypatch.setattr(feature_plateau, "TOPK_SELECTION_MODE", "mean_only")

    x = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 3.0],
            "f2": [1.0, 0.0, 1.0, 0.0],
        }
    )
    y = pd.Series([0, 1, 0, 1], dtype=np.int8)
    sample_weight = pd.Series([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    folds = [
        {"fold_id": 0, "train_idx": np.array([0, 1]), "valid_idx": np.array([2])},
        {"fold_id": 1, "train_idx": np.array([0, 1, 2]), "valid_idx": np.array([3])},
    ]
    fold_weight_by_id = pd.Series([1.0, 2.0], index=[0, 1], dtype=np.float64)

    monkeypatch.setattr(
        feature_plateau,
        "prepare_fold_features",
        lambda x_train_raw, x_valid_raw: (x_train_raw, x_valid_raw, []),
    )

    class DummyModel:
        pass

    monkeypatch.setattr(feature_plateau, "make_estimator", lambda seed: DummyModel())
    monkeypatch.setattr(
        feature_plateau,
        "fit_model",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(feature_plateau, "get_best_iteration", lambda model: 7)

    predict_calls = iter(
        [
            (np.array([0], dtype=np.int8), np.array([[0.9, 0.1]], dtype=np.float64)),
            (np.array([1], dtype=np.int8), np.array([[0.1, 0.9]], dtype=np.float64)),
        ]
    )
    monkeypatch.setattr(
        feature_plateau,
        "predict_for_scoring",
        lambda model, x_valid, best_iteration: next(predict_calls),
    )
    monkeypatch.setattr(
        feature_plateau,
        "score_predictions",
        lambda scorer_cfg, y_true, y_pred, y_pred_proba, sample_weight: (
            0.4 if int(y_true[0]) == 0 else 0.8
        ),
    )

    result = feature_plateau.score_topk_subset(
        x=x,
        y=y,
        sample_weight=sample_weight,
        folds=folds,
        fold_weight_by_id=fold_weight_by_id,
        global_feature_order=["f1", "f2"],
        k=2,
        phase="test",
    )

    assert result["mean_score"] == pytest.approx(0.6)
    assert result["weighted_mean_score"] == pytest.approx((0.4 * 1.0 + 0.8 * 2.0) / 3.0)
    assert result["selection_base_score"] == pytest.approx(result["weighted_mean_score"])


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
                "selection_score": 0.6924620886297623,
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
                "selection_score": 0.6924608,
            },
            {
                "k": 279,
                "mean_score": 0.6913210,
                "weighted_mean_score": 0.6913615,
                "selection_base_score": 0.6913615,
                "std_score": 0.0011040,
                "selection_score": 0.6924590,
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
