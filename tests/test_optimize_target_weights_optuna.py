import numpy as np
import pandas as pd
import pytest

from optimize_target_weights_optuna import (
    build_fold_recency_weights,
    build_initial_weight_candidates,
    build_refined_weight_candidates,
    build_sample_weight_series,
    build_subset_summary_rows,
    build_weight_config,
    summarize_decision_fold_metric_scores,
)


def test_build_weight_config_matches_current_default_split():
    cfg = build_weight_config(0.8)

    assert cfg["decision_weight"] == pytest.approx(0.8)
    assert cfg["other_weight"] == pytest.approx(0.05)
    assert cfg["non_decision_total_weight"] == pytest.approx(0.2)
    assert cfg["total_block_weight"] == pytest.approx(1.0)


def test_build_sample_weight_series_applies_decision_mask():
    opened = pd.to_datetime(
        [
            "2026-04-01 12:13:00",
            "2026-04-01 12:14:00",
            "2026-04-01 12:15:00",
            "2026-04-01 12:19:00",
        ],
        utc=True,
    )
    decision_mask = np.array([False, True, False, True], dtype=bool)
    weights = build_sample_weight_series(
        decision_mask,
        build_weight_config(0.8),
        index=pd.RangeIndex(len(opened)),
        float_dtype=np.float64,
    )

    assert weights.tolist() == pytest.approx([0.05, 0.8, 0.05, 0.8])


def test_build_weight_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        build_weight_config(0.0)
    with pytest.raises(ValueError):
        build_weight_config(1.0)


def test_build_fold_recency_weights_uses_linear_growth(monkeypatch):
    import optimize_target_weights_optuna as target_optuna

    monkeypatch.setattr(target_optuna, "ENABLE_FOLD_RECENCY_WEIGHTING", True)
    monkeypatch.setattr(target_optuna, "FOLD_RECENCY_WEIGHTING_MODE", "linear")
    monkeypatch.setattr(target_optuna, "FOLD_RECENCY_WEIGHT_MIN", 1.0)
    monkeypatch.setattr(target_optuna, "FOLD_RECENCY_WEIGHT_MAX", 1.4)

    folds = [
        {"fold_id": 0},
        {"fold_id": 1},
        {"fold_id": 2},
    ]

    weights = build_fold_recency_weights(folds)

    assert weights.tolist() == pytest.approx([1.0, 1.2, 1.4])


def test_summarize_decision_fold_metric_scores_uses_weighted_mean_when_active(
    monkeypatch,
):
    import optimize_target_weights_optuna as target_optuna

    monkeypatch.setattr(target_optuna, "ENABLE_FOLD_RECENCY_WEIGHTING", True)
    monkeypatch.setattr(target_optuna, "FOLD_RECENCY_WEIGHT_MIN", 1.0)
    monkeypatch.setattr(target_optuna, "FOLD_RECENCY_WEIGHT_MAX", 1.4)

    folds = [
        {"fold_id": 0},
        {"fold_id": 1},
        {"fold_id": 2},
    ]
    metric_values = np.array([0.50, 0.60, 0.90], dtype=np.float64)
    fold_weight_by_id = build_fold_recency_weights(folds)

    summary = summarize_decision_fold_metric_scores(
        metric_values,
        folds=folds,
        fold_weight_by_id=fold_weight_by_id,
        std_penalty=1.0,
    )

    expected_weighted_mean = np.average(
        metric_values,
        weights=np.array([1.0, 1.2, 1.4], dtype=np.float64),
    )
    expected_std = np.std(metric_values)

    assert summary["decision_metric_mean"] == pytest.approx(np.mean(metric_values))
    assert summary["decision_metric_weighted_mean"] == pytest.approx(
        expected_weighted_mean
    )
    assert summary["decision_metric_std"] == pytest.approx(expected_std)
    assert summary["decision_metric_base_value"] == pytest.approx(
        expected_weighted_mean
    )
    assert summary["decision_metric_objective"] == pytest.approx(
        expected_weighted_mean - expected_std
    )


def test_build_initial_weight_candidates_is_sorted_and_deduped():
    weights = build_initial_weight_candidates()

    assert list(weights) == sorted(weights)
    assert len(weights) == len(set(weights))
    assert weights[0] >= 0.2
    assert weights[-1] <= 0.999
    assert 0.9 in weights


def test_build_refined_weight_candidates_adds_midpoints_around_best_weights():
    refined = build_refined_weight_candidates(
        evaluated_weights=[0.2, 0.5, 0.9],
        top_parent_weights=[0.5],
    )

    assert refined == pytest.approx((0.35, 0.7))


def test_build_subset_summary_rows_prefers_baseline_when_weighted_is_worse():
    final_results_df = pd.DataFrame(
        [
            {
                "feature_subset_id": "subset_00",
                "feature_subset_label": "active:foo",
                "feature_subset_path": "foo.json",
                "feature_count": 64,
                "is_active_feature_subset": True,
                "strategy_name": "decision_rows_only_baseline",
                "objective_value": 0.52,
                "decision_rows_oof_balanced_accuracy": 0.53,
                "decision_weight": np.nan,
                "decision_metric_weighted_mean": 0.52,
            },
            {
                "feature_subset_id": "subset_00",
                "feature_subset_label": "active:foo",
                "feature_subset_path": "foo.json",
                "feature_count": 64,
                "is_active_feature_subset": True,
                "strategy_name": "all_rows_weighted",
                "objective_value": 0.51,
                "decision_rows_oof_balanced_accuracy": 0.54,
                "decision_weight": 0.9,
                "decision_metric_weighted_mean": 0.51,
            },
        ]
    )

    summary_df = build_subset_summary_rows(final_results_df)

    assert len(summary_df) == 1
    assert summary_df.iloc[0]["recommended_strategy_name"] == "decision_rows_only_baseline"
    assert summary_df.iloc[0]["recommendation_reason"] == "subset_baseline_beats_weighted"
