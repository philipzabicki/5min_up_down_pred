import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from optimize_target_weights_optuna import (
    build_baseline_recommendation_summary,
    build_feature_view_candidates,
    build_fold_recency_weights,
    build_initial_weight_candidates,
    build_param_profile_candidates,
    build_refined_weight_candidates,
    build_sample_weight_series,
    build_subset_summary_rows,
    build_weight_config,
    enrich_aggregate_results_with_baseline_deltas,
    enrich_context_results_with_baseline_deltas,
    load_target_weight_search_settings,
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


def test_build_feature_view_candidates_includes_all_active_and_random():
    x_all = pd.DataFrame(
        np.arange(60, dtype=np.float64).reshape(10, 6),
        columns=["f0", "f1", "f2", "f3", "f4", "f5"],
    )
    active_subset = {
        "path": Path("data/analysis/feature_selector/example/recommended_features.json"),
        "features": ("f0", "f1", "f2"),
        "count": 3,
        "format": "json",
        "list_key": "final_feature_list",
        "created_utc": None,
        "source_data_path": None,
        "metadata": {},
        "source_count": 3,
        "excluded_feature_names": tuple(),
        "excluded_count": 0,
        "excluded_from_subset_count": 0,
    }

    feature_views = build_feature_view_candidates(
        x_all=x_all,
        active_subset=active_subset,
        include_all_features_view=True,
        include_active_feature_subset_view=True,
        random_feature_subsets=2,
        random_feature_subset_size=None,
        random_feature_subset_fraction=0.5,
        random_feature_subset_min_features=2,
    )

    assert [view["label"] for view in feature_views][:2] == [
        "all_features",
        "active:example",
    ]
    assert len(feature_views) == 4
    assert all(view["feature_count"] >= 2 for view in feature_views)
    assert {view["source_kind"] for view in feature_views} == {
        "all_features",
        "active_subset",
        "random_subset",
    }


def test_build_param_profile_candidates_uses_requested_profiles():
    param_profiles = build_param_profile_candidates(
        {
            "param_profiles": ("target_weight_robust",),
        }
    )

    assert [profile["name"] for profile in param_profiles] == ["target_weight_robust"]
    assert all("device_type" in profile["params"] for profile in param_profiles)


def test_load_target_weight_search_settings_defaults_to_multi_context():
    settings = load_target_weight_search_settings(
        {
            "modeling_output_dir": Path("data/modeling_datasets"),
            "output_suffix": "_model_ready",
            "base_data_file": "BTCUSDT1m.csv",
        }
    )

    assert settings["output_dir"] == Path("data/modeling_datasets")
    assert settings["output_suffix"] == "_model_ready"
    assert settings["param_profiles"] == ("target_weight_robust",)
    assert settings["include_all_features_view"] is False
    assert settings["include_active_feature_subset_view"] is False
    assert settings["random_feature_subsets"] == 10
    assert settings["random_feature_subset_size"] == 256
    assert settings["random_feature_subset_fraction"] is None
    assert settings["random_feature_subset_min_features"] is None
    assert settings["lookback_days"] == 365
    assert settings["context_std_penalty"] == pytest.approx(1.0)


def test_load_target_weight_search_settings_uses_dataset_output_location():
    settings = load_target_weight_search_settings(
        {
            "modeling_output_dir": Path("data/modeling_datasets/custom"),
            "output_suffix": "_custom_suffix",
            "base_data_file": "BTCUSDT1m.csv",
        }
    )

    assert settings["output_dir"] == Path("data/modeling_datasets/custom")
    assert settings["output_suffix"] == "_custom_suffix"
    assert settings["param_profiles"] == ("target_weight_robust",)
    assert settings["include_all_features_view"] is False
    assert settings["include_active_feature_subset_view"] is False
    assert settings["random_feature_subsets"] == 10
    assert settings["random_feature_subset_size"] == 256
    assert settings["random_feature_subset_fraction"] is None
    assert settings["random_feature_subset_min_features"] is None
    assert settings["lookback_days"] == 365
    assert settings["context_std_penalty"] == pytest.approx(1.0)


def test_enrich_context_results_with_baseline_deltas_compares_per_context():
    context_results_df = pd.DataFrame(
        [
            {
                "context_id": "context_00",
                "context_label": "random:00 | robust",
                "strategy_name": "decision_rows_only_baseline",
                "decision_weight": np.nan,
                "objective_value": 0.510,
                "decision_rows_oof_balanced_accuracy": 0.520,
            },
            {
                "context_id": "context_00",
                "context_label": "random:00 | robust",
                "strategy_name": "all_rows_weighted",
                "decision_weight": 0.35,
                "objective_value": 0.515,
                "decision_rows_oof_balanced_accuracy": 0.525,
            },
            {
                "context_id": "context_01",
                "context_label": "random:01 | robust",
                "strategy_name": "decision_rows_only_baseline",
                "decision_weight": np.nan,
                "objective_value": 0.530,
                "decision_rows_oof_balanced_accuracy": 0.540,
            },
            {
                "context_id": "context_01",
                "context_label": "random:01 | robust",
                "strategy_name": "all_rows_weighted",
                "decision_weight": 0.35,
                "objective_value": 0.520,
                "decision_rows_oof_balanced_accuracy": 0.535,
            },
        ]
    )

    enriched = enrich_context_results_with_baseline_deltas(context_results_df)
    weighted_row = enriched[
        (enriched["strategy_name"] == "all_rows_weighted")
        & (enriched["context_id"] == "context_00")
    ].iloc[0]
    baseline_row = enriched[
        (enriched["strategy_name"] == "decision_rows_only_baseline")
        & (enriched["context_id"] == "context_00")
    ].iloc[0]

    assert weighted_row["baseline_objective_value"] == pytest.approx(0.510)
    assert weighted_row["objective_delta_vs_context_baseline"] == pytest.approx(0.005)
    assert weighted_row["decision_bal_acc_delta_vs_context_baseline"] == pytest.approx(
        0.005
    )
    assert baseline_row["objective_delta_vs_context_baseline"] == pytest.approx(0.0)


def test_enrich_aggregate_results_with_baseline_deltas_counts_context_wins():
    aggregate_results_df = pd.DataFrame(
        [
            {
                "strategy_name": "decision_rows_only_baseline",
                "decision_weight": np.nan,
                "objective_value": 0.512,
                "context_objective_mean": 0.514,
                "context_objective_std": 0.002,
                "decision_rows_bal_acc_mean": 0.521,
            },
            {
                "strategy_name": "all_rows_weighted",
                "decision_weight": 0.35,
                "objective_value": 0.515,
                "context_objective_mean": 0.516,
                "context_objective_std": 0.001,
                "decision_rows_bal_acc_mean": 0.522,
            },
        ]
    )
    context_results_df = pd.DataFrame(
        [
            {
                "context_id": "context_00",
                "strategy_name": "decision_rows_only_baseline",
                "decision_weight": np.nan,
                "objective_value": 0.510,
                "decision_rows_oof_balanced_accuracy": 0.520,
            },
            {
                "context_id": "context_01",
                "strategy_name": "decision_rows_only_baseline",
                "decision_weight": np.nan,
                "objective_value": 0.514,
                "decision_rows_oof_balanced_accuracy": 0.522,
            },
            {
                "context_id": "context_00",
                "strategy_name": "all_rows_weighted",
                "decision_weight": 0.35,
                "objective_value": 0.515,
                "decision_rows_oof_balanced_accuracy": 0.525,
            },
            {
                "context_id": "context_01",
                "strategy_name": "all_rows_weighted",
                "decision_weight": 0.35,
                "objective_value": 0.513,
                "decision_rows_oof_balanced_accuracy": 0.521,
            },
        ]
    )

    enriched = enrich_aggregate_results_with_baseline_deltas(
        aggregate_results_df,
        context_results_df=context_results_df,
    )
    weighted_row = enriched[enriched["strategy_name"] == "all_rows_weighted"].iloc[0]
    baseline_row = enriched[
        enriched["strategy_name"] == "decision_rows_only_baseline"
    ].iloc[0]

    assert weighted_row["objective_delta_vs_baseline"] == pytest.approx(0.003)
    assert weighted_row["contexts_beating_baseline"] == pytest.approx(1)
    assert weighted_row["contexts_losing_to_baseline"] == pytest.approx(1)
    assert weighted_row["contexts_tying_baseline"] == pytest.approx(0)
    assert baseline_row["objective_delta_vs_baseline"] == pytest.approx(0.0)


def test_build_baseline_recommendation_summary_prefers_baseline_when_weighted_is_worse():
    final_results_df = pd.DataFrame(
        [
            {
                "strategy_name": "all_rows_weighted",
                "decision_weight": 0.35,
                "objective_value": 0.5135,
                "decision_rows_bal_acc_mean": 0.5203,
            },
            {
                "strategy_name": "decision_rows_only_baseline",
                "decision_weight": np.nan,
                "objective_value": 0.5142,
                "decision_rows_bal_acc_mean": 0.5208,
            },
        ]
    )

    recommendation = build_baseline_recommendation_summary(final_results_df)

    assert recommendation["recommended_strategy_name"] == "decision_rows_only_baseline"
    assert recommendation["recommended_decision_weight"] is None
    assert recommendation["recommendation_reason"] == "context_baseline_beats_weighted"
