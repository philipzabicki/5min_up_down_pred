import json

import numpy as np
import optuna
import pytest

import fit_volume_profile as volume_profile_fit
from features.volume_profile_fixed_range import normalize_config as normalize_vp_config


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

    assert summary["cv_balanced_accuracy_mean"] == pytest.approx(np.mean(fold_scores))
    assert summary["cv_balanced_accuracy_weighted_mean"] == pytest.approx(
        expected_weighted_mean
    )
    assert summary["cv_balanced_accuracy_std"] == pytest.approx(expected_std)
    assert summary["objective_base_value"] == pytest.approx(expected_weighted_mean)
    assert summary["objective_value"] == pytest.approx(
        expected_weighted_mean - (0.5 * expected_std)
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

    assert summary["cv_balanced_accuracy_weighted_mean"] == pytest.approx(expected_mean)
    assert summary["objective_base_value"] == pytest.approx(expected_mean)
    assert summary["objective_value"] == pytest.approx(
        expected_mean - (0.5 * expected_std)
    )


def test_build_volume_profile_config_from_params_builds_per_horizon_payload():
    base_config = normalize_vp_config(
        {
            "enabled": True,
            "price_min": 0.0,
            "price_max": 100.0,
            "neighbor_bins": 3,
            "eps": 1e-6,
            "horizons": {
                "short": {
                    "step": 2,
                    "local_window": 4,
                    "sigma_divisor": 5.0,
                    "min_sigma": 6.0,
                    "half_life_candles": 10,
                },
                "medium": {
                    "step": 3,
                    "local_window": 5,
                    "sigma_divisor": 6.0,
                    "min_sigma": 7.0,
                    "half_life_candles": 20,
                },
                "long": {
                    "step": 4,
                    "local_window": 6,
                    "sigma_divisor": 7.0,
                    "min_sigma": 8.0,
                    "half_life_candles": 30,
                },
                "all": {
                    "step": 5,
                    "local_window": 7,
                    "sigma_divisor": 8.0,
                    "min_sigma": 9.0,
                    "half_life_candles": None,
                },
            },
        }
    )

    cfg = volume_profile_fit.build_volume_profile_config_from_params(
        base_config,
        {
            "neighbor_bins": 11,
            "short_step": 10,
            "medium_local_window": 22,
            "long_sigma_divisor": 33.0,
            "all_min_sigma": 44.0,
            "short_half_life_candles": 55,
        },
    )

    assert cfg["neighbor_bins"] == 11
    assert cfg["horizons"]["short"]["step"] == 10
    assert cfg["horizons"]["medium"]["local_window"] == 22
    assert cfg["horizons"]["long"]["sigma_divisor"] == pytest.approx(33.0)
    assert cfg["horizons"]["all"]["min_sigma"] == pytest.approx(44.0)
    assert cfg["horizons"]["short"]["half_life_candles"] == 55
    assert cfg["horizons"]["all"]["half_life_candles"] is None


def test_compact_volume_profile_artifact_payload_surfaces_readable_best_config():
    base_config = normalize_vp_config(
        {
            "enabled": True,
            "price_min": 0.0,
            "price_max": 100.0,
            "neighbor_bins": 3,
            "eps": 1e-6,
            "horizons": {
                "short": {
                    "step": 2,
                    "local_window": 4,
                    "sigma_divisor": 5.0,
                    "min_sigma": 6.0,
                    "half_life_candles": 10,
                },
                "medium": {
                    "step": 3,
                    "local_window": 5,
                    "sigma_divisor": 6.0,
                    "min_sigma": 7.0,
                    "half_life_candles": 20,
                },
                "long": {
                    "step": 4,
                    "local_window": 6,
                    "sigma_divisor": 7.0,
                    "min_sigma": 8.0,
                    "half_life_candles": 30,
                },
                "all": {
                    "step": 5,
                    "local_window": 7,
                    "sigma_divisor": 8.0,
                    "min_sigma": 9.0,
                    "half_life_candles": None,
                },
            },
        }
    )
    best_config = volume_profile_fit.build_volume_profile_config_from_params(
        base_config,
        {
            "neighbor_bins": 11,
            "short_step": 10,
            "medium_local_window": 22,
            "long_sigma_divisor": 33.0,
            "all_min_sigma": 44.0,
            "short_half_life_candles": 55,
        },
    )
    objective_name = volume_profile_fit.resolve_cv_objective_name()

    payload = {
        "created_utc": "2026-04-13T18:28:21+00:00",
        "study_name": "vp_test",
        "study_name_source": "auto",
        "run_timestamp_utc": "20260413_182821",
        "base_data_path": "data/raw.csv",
        "target_col": "target_5m_candle_up",
        "sample_weight_col": "target_5m_weight",
        "sample_weight": {
            "used": True,
            "source": "dataset_column",
            "min": 0.8,
            "max": 0.8,
            "mean": 0.8,
            "sum": 8.0,
            "distribution": {"0.8": 10},
        },
        "class_distribution": {"0": 5, "1": 5},
        "weighted_class_distribution": {"0": 4.0, "1": 4.0},
        "rows_raw": 100,
        "rows_after_target_notna": 99,
        "decision_row_filter": {"enabled": True, "rows_after": 10},
        "feature_set": {
            "mode": "volume_profile_fixed_range_only",
            "base_feature_count": len(base_config["feature_columns"]),
            "best_feature_count": len(best_config["feature_columns"]),
        },
        "storage": "sqlite:///data/optuna/databases/volume_profile.db",
        "n_trials_requested": 200,
        "timeout_seconds": None,
        "cv_folds": 20,
        "walk_forward_test_to_train_ratio": 0.2,
        "fold_recency_weighting": {
            "enabled": True,
            "active": True,
            "mode": "linear",
            "min_weight": 1.0,
            "max_weight": 1.4,
            "std_score_aggregation": "unweighted",
            "fold_weights": [
                {"fold_id": 0, "weight": 1.0},
                {"fold_id": 19, "weight": 1.4},
            ],
        },
        "max_n_estimators": 300,
        "early_stopping_rounds": 40,
        "prune_report_every_n_iteration": 10,
        "cv_objective": {
            "name": objective_name,
            "base_metric": volume_profile_fit.CV_OBJECTIVE_BASE_METRIC,
            "base_score": volume_profile_fit.cv_objective_base_score_label(),
            "aggregation": "weighted_mean - std_penalty * cv_std",
            "std_penalty": 0.75,
        },
        "lgbm_params": {"objective": "binary"},
        "base_volume_profile_fixed_range": base_config,
        "volume_profile_optuna_search_space": {"neighbor_bins": {"type": "int"}},
        "optuna_seed_trial_params": [{"neighbor_bins": 3}],
        "best_trial": {
            "number": 7,
            f"objective_{objective_name}": 0.61,
            "cv_balanced_accuracy_mean": 0.62,
            "cv_balanced_accuracy_weighted_mean": 0.63,
            "cv_balanced_accuracy_std": 0.02,
            "best_iteration": 123,
            "feature_count": len(best_config["feature_columns"]),
            "params": {"neighbor_bins": 11, "short_step": 10},
            "volume_profile_fixed_range": best_config,
        },
        "artifacts": {
            "best_result_path": "data/optuna/volume_profile/best.json",
            "trials_csv_path": "data/optuna/volume_profile/trials.csv",
        },
    }

    compact = volume_profile_fit.compact_volume_profile_artifact_payload(payload)

    assert compact["best_trial"]["trial_number"] == 7
    assert compact["best_trial"]["objective_name"] == objective_name
    assert compact["best_trial"]["objective_value"] == pytest.approx(0.61)
    assert compact["best_params_flat"] == {"neighbor_bins": 11, "short_step": 10}
    assert compact["best_volume_profile_fixed_range"]["neighbor_bins"] == 11
    assert (
        compact["best_volume_profile_fixed_range"]["horizons"]["short"]["step"] == 10
    )
    assert compact["best_volume_profile_derived"]["feature_count"] == len(
        best_config["feature_columns"]
    )
    assert "feature_columns" not in compact["best_volume_profile_fixed_range"]
    assert "config_signature" not in json.dumps(compact)
    assert "distribution" not in compact["dataset"]["sample_weight"]
    assert "fold_weights" not in compact["optimization"]["fold_recency_weighting"]
    assert compact["optimization"]["fold_recency_weighting"]["fold_count"] == 2


def test_get_best_successful_trial_respects_study_direction():
    maximize_study = optuna.create_study(direction="maximize")
    maximize_study.add_trial(
        optuna.trial.create_trial(
            params={},
            distributions={},
            value=0.50,
            user_attrs={"trial_status": "ok"},
        )
    )
    maximize_study.add_trial(
        optuna.trial.create_trial(
            params={},
            distributions={},
            value=0.75,
            user_attrs={"trial_status": "ok"},
        )
    )

    best_max = volume_profile_fit.get_best_successful_trial(maximize_study)

    assert best_max.value == pytest.approx(0.75)

    minimize_study = optuna.create_study(direction="minimize")
    minimize_study.add_trial(
        optuna.trial.create_trial(
            params={},
            distributions={},
            value=0.50,
            user_attrs={"trial_status": "ok"},
        )
    )
    minimize_study.add_trial(
        optuna.trial.create_trial(
            params={},
            distributions={},
            value=0.75,
            user_attrs={"trial_status": "ok"},
        )
    )

    best_min = volume_profile_fit.get_best_successful_trial(minimize_study)

    assert best_min.value == pytest.approx(0.50)
