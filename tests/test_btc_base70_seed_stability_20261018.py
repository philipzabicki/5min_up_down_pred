import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_btc_base70_seed_stability_20261018 as runner
import run_btc_preopen_taker_flow_20261018 as taker


class Base70SeedStabilityTests(unittest.TestCase):
    def test_new_seed_changes_only_the_explicit_seed_controls(self):
        base = {
            "feature_fraction": 0.8,
            "seed": 20261015,
            "feature_fraction_seed": 20261015,
            "data_random_seed": 20261015,
            "bagging_seed": 20261015,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "num_leaves": 81,
        }
        params = runner._effective_params(base, 20261018)
        self.assertEqual(base["seed"], 20261015)
        self.assertEqual({params[name] for name in ("seed", "feature_fraction_seed", "data_random_seed", "bagging_seed")}, {20261018})
        self.assertEqual(params["feature_fraction"], base["feature_fraction"])
        self.assertEqual(params["num_leaves"], base["num_leaves"])
        self.assertEqual(params["bagging_fraction"], 1.0)
        self.assertEqual(params["bagging_freq"], 0)

    def test_ensemble_is_the_equal_weight_arithmetic_mean_of_five_saved_values(self):
        values = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
        expected = values.astype(np.float64).mean(axis=1)
        np.testing.assert_array_equal(runner._ensemble_probability(values), expected)
        with self.assertRaises(ValueError):
            runner._ensemble_probability(np.zeros((2, 4)))

    def test_circular_three_day_block_draws_are_seeded_and_shared(self):
        days = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
        first = runner._block_indices(days, replications=10, block_days=3, seed=44)
        second = runner._block_indices(days, replications=10, block_days=3, seed=44)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (10, 8))
        for draw in first:
            for start in (0, 3, 6):
                block = draw[start:min(start + 3, len(draw))]
                self.assertTrue(np.all((np.diff(block) % len(days)) == 1))

    def test_row_day_index_handles_different_pandas_timestamp_resolutions(self):
        starts = pd.Series(pd.to_datetime(["2026-01-01T23:55:00Z", "2026-01-02T00:05:00Z"], utc=True).as_unit("us"))
        days = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC").as_unit("s")
        np.testing.assert_array_equal(runner._row_day_index(starts, days), np.asarray([0, 1], dtype=np.int32))

    def test_weighted_auc_bootstrap_matches_repeated_day_weights(self):
        labels = np.asarray([0, 1, 1, 0, 0, 1, 0, 1], dtype=np.int8)
        probability = np.asarray([0.2, 0.5, 0.5, 0.5, 0.1, 0.8, 0.8, 0.8])
        row_days = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        day_counts = np.asarray([[2, 1, 0, 1], [0, 2, 1, 0]], dtype=np.int32)
        actual = runner._weighted_auc_bootstrap(labels, probability, row_days, day_counts)
        for index, counts in enumerate(day_counts):
            weights = counts[row_days]
            expected = roc_auc_score(labels, probability, sample_weight=weights)
            self.assertAlmostEqual(actual[index], expected, places=14)

    def test_valid_completed_fold_is_not_scheduled_for_refit(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / "data/analysis/polymarket/BTC") as temporary:
            temporary_root = Path(temporary)
            with patch.object(runner, "OUTPUT_DIR", temporary_root):
                seed, fold_id = 20261016, 0
                model_path, oof_path, completion_path = runner._fold_paths(seed, fold_id)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_text("checkpoint", encoding="utf-8")
                oof = pd.DataFrame({"_source_row_index": np.asarray([31], dtype=np.int32), "fold_id": ["0"], "p_model_raw": [0.4]})
                runner._write_parquet(oof, oof_path)
                identity = "frozen-inputs"
                model_record = runner._record(model_path)
                oof_record = runner._record(oof_path)
                runner._write_json(completion_path, {
                    "identity_sha256": identity,
                    "seed": seed,
                    "fold_id": fold_id,
                    "validation_source_row_index_sha256": taker._array_sha256(np.asarray([31], dtype=np.int32), dtype=np.int32),
                    "model": model_record,
                    "oof": oof_record,
                })
                completion_record = runner._record(completion_path)
                manifest = {
                    "input_identity": {"frozen_spec_sha256": identity},
                    "completed_fits": {str(seed): {str(fold_id): {
                        "status": "completed", "identity_sha256": identity,
                        "model": model_record, "oof": oof_record,
                        "completion_manifest": completion_record,
                    }}},
                }
                booster = Mock()
                booster.current_iteration.return_value = 31
                booster.feature_name.return_value = ["x"]
                with patch.object(taker.lgb, "Booster", return_value=booster), patch.object(taker.lgb, "train") as train:
                    pending = runner._pending_folds(
                        manifest, seed, [(np.asarray([0], dtype=np.int32), np.asarray([0], dtype=np.int32))], ["x"],
                    )
                self.assertEqual(pending, [])
                train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
