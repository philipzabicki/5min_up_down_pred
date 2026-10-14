import unittest
from unittest.mock import patch

import numpy as np
import optuna
import lightgbm as lgb
from lightgbm.callback import EarlyStopException
from types import SimpleNamespace

from utils.cv_objective import (
    ObjectiveAlignedCVCallback,
    build_fold_metric_signatures,
    cv_objective_series,
    make_fold_metric_eval,
)
from utils.metrics import weighted_binary_logloss


class RecordingTrial:
    def __init__(self, prune=False):
        self.values = []
        self.prune = prune

    def report(self, value, step):
        self.values.append((int(step), float(value)))

    def should_prune(self):
        return self.prune


def cv_env(iteration, mean, std, fold_scores=None):
    evaluations = [("valid", "binary_logloss", mean, False, std)]
    if fold_scores is not None:
        evaluations.extend(
            ("valid", f"binary_logloss_fold_{fold_id}", value, False, 0.0)
            for fold_id, value in enumerate(fold_scores)
        )
    return SimpleNamespace(
        iteration=iteration,
        evaluation_result_list=evaluations,
    )


class CvObjectiveTests(unittest.TestCase):
    def test_dispersion_penalty_can_improve_after_mean_stops_improving(self):
        result = cv_objective_series(
            {
                "valid binary_logloss-mean": [0.70, 0.72, 0.74],
                "valid binary_logloss-stdv": [0.10, 0.00, 0.00],
            },
            "binary_logloss",
            std_penalty=0.5,
        )
        self.assertGreater(result["mean"][1], result["mean"][0])
        self.assertLess(result["objective"][1], result["objective"][0])
        self.assertEqual(int(np.argmin(result["objective"])), 1)

    def test_recency_weights_can_change_best_iteration(self):
        result = cv_objective_series(
            {
                "valid binary_logloss_fold_0-mean": [0.70, 0.95],
                "valid binary_logloss_fold_1-mean": [0.70, 0.35],
            },
            "binary_logloss",
            fold_ids=[0, 1],
            fold_weights=[1.0, 3.0],
            std_penalty=0.5,
        )
        unweighted = cv_objective_series(
            {
                "valid binary_logloss-mean": [0.70, 0.65],
                "valid binary_logloss-stdv": [0.00, 0.30],
            },
            "binary_logloss",
            std_penalty=0.5,
        )
        self.assertEqual(int(np.argmin(unweighted["objective"])), 0)
        self.assertEqual(int(np.argmin(result["objective"])), 1)

    def test_stopping_ranking_and_pruner_receive_the_same_penalized_score(self):
        trial = RecordingTrial()
        callback = ObjectiveAlignedCVCallback(
            "binary_logloss",
            std_penalty=0.5,
            stopping_rounds=1,
            trial=trial,
            report_interval=1,
        )
        callback(cv_env(0, 0.70, 0.10))
        callback(cv_env(1, 0.72, 0.00))
        with self.assertRaises(optuna.TrialPruned):
            ObjectiveAlignedCVCallback(
                "binary_logloss",
                std_penalty=0.5,
                trial=RecordingTrial(prune=True),
                report_interval=1,
            )(cv_env(0, 0.70, 0.10))
        with self.assertRaises(EarlyStopException) as caught:
            callback(cv_env(2, 0.74, 0.00))

        self.assertEqual(trial.values, [(0, 0.75), (1, 0.72), (2, 0.74)])
        self.assertEqual(callback.best_iteration, 1)
        self.assertEqual(caught.exception.best_iteration, 1)
        self.assertEqual(caught.exception.best_score[0][2], 0.72)

    def test_recency_weighted_score_is_used_for_stopping_and_pruning(self):
        trial = RecordingTrial()
        callback = ObjectiveAlignedCVCallback(
            "binary_logloss",
            fold_ids=[0, 1],
            fold_weights=[1.0, 3.0],
            std_penalty=0.5,
            stopping_rounds=1,
            trial=trial,
            report_interval=1,
        )
        callback(cv_env(0, 0.70, 0.00, [0.70, 0.70]))
        callback(cv_env(1, 0.65, 0.30, [0.95, 0.35]))
        self.assertEqual(callback.best_iteration, 1)
        self.assertAlmostEqual(trial.values[0][1], 0.70)
        self.assertAlmostEqual(trial.values[1][1], 0.65)

    def test_lightgbm_cv_curve_and_callback_select_the_same_objective(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(180, 4))
        y = (x[:, 0] + 0.2 * x[:, 1] > 0).astype(np.float64)
        sample_weight = np.full(len(y), 0.4625)
        row_indices = np.arange(len(y))
        fold_specs = []
        fold_indices = []
        for fold_id, valid_indices in enumerate(np.array_split(row_indices, 3)):
            train_indices = np.setdiff1d(row_indices, valid_indices)
            fold_specs.append({"fold_id": fold_id, "valid_idx": valid_indices})
            fold_indices.append((train_indices, valid_indices))

        evaluator = make_fold_metric_eval(
            "binary_logloss",
            build_fold_metric_signatures(fold_specs, y, sample_weight),
            weighted_binary_logloss,
        )
        callback = ObjectiveAlignedCVCallback(
            "binary_logloss",
            fold_ids=[0, 1, 2],
            fold_weights=[1.0, 1.25, 1.5],
            std_penalty=0.5,
            stopping_rounds=3,
        )
        results = lgb.cv(
            {
                "objective": "binary",
                "metric": "None",
                "verbosity": -1,
                "num_threads": 1,
                "num_iterations": 12,
                "learning_rate": 0.1,
                "num_leaves": 7,
                "max_bin": 15,
            },
            lgb.Dataset(x, label=y, weight=sample_weight),
            folds=fold_indices,
            stratified=False,
            shuffle=False,
            feval=[evaluator],
            callbacks=[callback],
            seed=1,
        )
        objective = cv_objective_series(
            results,
            "binary_logloss",
            fold_ids=[0, 1, 2],
            fold_weights=[1.0, 1.25, 1.5],
            std_penalty=0.5,
        )["objective"]

        self.assertEqual(callback.best_iteration, int(np.argmin(objective)))
        self.assertAlmostEqual(callback.best_value, float(np.min(objective)))

    def test_profile_cv_runners_use_per_fold_metrics_when_recency_is_active(self):
        import fit_reaction_profile as reaction_profile
        import fit_volume_profile as volume_profile

        rng = np.random.default_rng(41)
        x = rng.normal(size=(300, 4)).astype(np.float32)
        y = (x[:, 0] - 0.3 * x[:, 1] > 0).astype(np.float64)
        sample_weight = np.full(len(y), 0.4625)
        validation_ranges = ((150, 200), (200, 250), (250, 300))
        folds = []
        fold_indices = []
        for fold_id, (valid_start, valid_end) in enumerate(validation_ranges):
            folds.append(
                {
                    "fold_id": fold_id,
                    "test_start": valid_start,
                    "test_end": valid_end,
                    "valid_idx": np.arange(valid_start, valid_end),
                    "train_start": 0,
                    "train_end": valid_start,
                }
            )
            fold_indices.append(
                (np.arange(valid_start), np.arange(valid_start, valid_end))
            )

        fold_weights = {0: 1.0, 1: 1.25, 2: 1.5}
        for module, prefix in (
            (volume_profile, "vp"),
            (reaction_profile, "rp"),
        ):
            with (
                patch.object(module, "LGBM_DEVICE_TYPE", "cpu"),
                patch.object(module, "LGBM_NUM_THREADS", 1),
                patch.object(module, "MAX_N_ESTIMATORS", 20),
                patch.object(module, "EARLY_STOPPING_ROUNDS", 4),
                patch.object(module, "GPU_MAX_BIN_LIMIT", 31),
                patch.object(
                    module,
                    "LGBM_DEFAULT_PARAMS",
                    {
                        **module.LGBM_DEFAULT_PARAMS,
                        "num_leaves": 7,
                        "min_data_in_leaf": 10,
                        "learning_rate": 0.1,
                        "max_depth": -1,
                    },
                ),
            ):
                result = module.run_lightgbm_cv(
                    x,
                    y,
                    sample_weight,
                    folds,
                    fold_indices,
                    fold_weights,
                    [f"{prefix}{i}" for i in range(x.shape[1])],
                    return_cvbooster=True,
                )

            self.assertEqual(len(result["cvbooster"].boosters), len(folds))
            self.assertAlmostEqual(
                result["objective_value"],
                result["objective_base_value"]
                + module.CV_STD_PENALTY * result[f"cv_{module.CV_OBJECTIVE_BASE_METRIC}_std"],
            )


if __name__ == "__main__":
    unittest.main()
