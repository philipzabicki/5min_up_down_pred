from types import SimpleNamespace
import unittest

import numpy as np
from lightgbm.callback import EarlyStopException

import run_btc_base70_capacity_tuning_20261019 as tuning
from utils.cv_objective import ObjectiveAlignedCVCallback


class Base70CapacityTuningTests(unittest.TestCase):
    def test_candidate_schedule_stays_within_the_frozen_space(self):
        schedule = tuning._balanced_candidate_schedule()

        self.assertEqual(len(schedule), 31)
        self.assertEqual(len({tuple(row.values()) for row in schedule}), 31)
        for row in schedule:
            for name, levels in tuning.PARAMETER_LEVELS.items():
                with self.subTest(parameter=name, value=row[name]):
                    self.assertIn(row[name], levels)


    def test_effective_seed_controls_only_lightgbm_seed_parameters(self):
        base = {
            "seed": 7,
            "feature_fraction_seed": 8,
            "data_random_seed": 9,
            "bagging_seed": 10,
            "learning_rate": 0.064,
            "feature_fraction": 0.79,
            "device_type": "gpu",
            "max_bin": 63,
        }
        candidate = {
            "num_leaves": 63,
            "min_data_in_leaf": 4096,
            "lambda_l1": 1,
            "lambda_l2": 30,
        }

        effective = tuning._effective_params(base, candidate, 20261016)

        self.assertEqual(
            {name: effective[name] for name in tuning.SEED_PARAMETER_NAMES},
            {name: 20261016 for name in tuning.SEED_PARAMETER_NAMES},
        )
        self.assertEqual(effective["learning_rate"], base["learning_rate"])
        self.assertEqual(effective["feature_fraction"], base["feature_fraction"])
        self.assertEqual(effective["device_type"], base["device_type"])
        self.assertEqual(effective["max_bin"], base["max_bin"])
        self.assertEqual(effective["num_leaves"], candidate["num_leaves"])


    def test_recorded_history_includes_patience_rounds_and_uses_the_shared_objective(self):
        objective = ObjectiveAlignedCVCallback(
            "binary_logloss",
            higher_is_better=False,
            std_penalty=0.5,
            stopping_rounds=2,
            trial=None,
        )
        recording = tuning._RecordingObjectiveCallback(objective)
        mean_and_std = [(0.60, 0.10), (0.55, 0.10), (0.53, 0.10), (0.54, 0.10), (0.56, 0.10)]

        for iteration, (mean, std) in enumerate(mean_and_std):
            env = SimpleNamespace(
                iteration=iteration,
                evaluation_result_list=[("valid", "binary_logloss", mean, False, std)],
            )
            if iteration == len(mean_and_std) - 1:
                with self.assertRaises(EarlyStopException):
                    recording(env)
            else:
                recording(env)

        objectives = np.asarray([mean + 0.5 * std for mean, std in mean_and_std])
        self.assertEqual(len(recording.history), len(mean_and_std))
        self.assertEqual(np.argmin([row["criterion"] for row in recording.history]), objective.best_iteration)
        self.assertTrue(np.allclose([row["criterion"] for row in recording.history], objectives))
        self.assertEqual(objective.best_iteration + 1, 3)


if __name__ == "__main__":
    unittest.main()
