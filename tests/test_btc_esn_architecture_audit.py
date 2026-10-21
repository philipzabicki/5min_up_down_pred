from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
import run_btc_esn_architecture_audit_20261020 as audit

from run_btc_esn_architecture_audit_20261020 import (
    CONFIG,
    SEARCH_END_NS,
    _architecture_anchors,
    _config_record,
    _config_from_record,
    _config_key,
    _input_anchors,
    _scope_mask,
    _trial_config,
)
from features.esn_reservoir import ESNConfig


class ESNArchitectureAuditTests(unittest.TestCase):
    def test_engine_atomic_json_retries_transient_windows_replace_denial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            actual_replace = audit.os.replace
            attempts = 0

            def transiently_denied(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("temporary sharing violation")
                actual_replace(source, destination)

            with patch.object(audit.os, "replace", side_effect=transiently_denied):
                audit._atomic_json(path, {"ok": True})

            self.assertEqual(attempts, 3)
            self.assertEqual(audit.json.loads(path.read_text(encoding="utf-8")), {"ok": True})
            self.assertEqual({entry.name for entry in path.parent.iterdir()}, {path.name})

    def test_downstream_resume_skips_cached_seed_without_deleting_missing_baseline(self):
        previous_out = audit.OUT
        previous_identity = audit.RUN_IDENTITY
        previous_seeds = audit.SEEDS
        previous_lgbm_seeds = audit.LGBM_SEEDS
        with tempfile.TemporaryDirectory() as directory:
            audit.OUT = Path(directory)
            audit.RUN_IDENTITY = {"test_identity": "resume"}
            audit.SEEDS = (1,)
            audit.LGBM_SEEDS = (10, 11)
            try:
                n = 101_000
                opened = np.full(n, audit.FIT_END_NS - 60_000_000_000, dtype=np.int64)
                available = np.full(n, audit.FIT_END_NS, dtype=np.int64)
                decision = np.ones(n, dtype=bool)
                opened[-1_000:] = audit.VALIDATION_START_NS
                available[-1_000:] = audit.VALIDATION_START_NS + audit.MINUTE_NS
                metrics = {"n": 1_000, "logloss": 0.69, "brier": 0.24, "auc": 0.5}
                config = ESNConfig(architecture="single", reservoir_size=4, reservoir_count=1)
                candidate_id = "cached-candidate"
                finalist = {
                    "candidate_id": candidate_id,
                    "effective_config": audit._config_record(config),
                }
                progress_identity = {
                    "identity": audit.RUN_IDENTITY,
                    "candidate_ids": [candidate_id],
                    "variants": {candidate_id: [candidate_id]},
                    "reservoir_seeds": [1],
                    "lightgbm_seeds": [10, 11],
                }
                progress_path = audit.OUT / "downstream_resume_test_progress.json"
                audit._atomic_json(progress_path, {
                    "progress_identity": progress_identity,
                    "baseline_metrics": {"10": metrics},
                    "candidate_records": {candidate_id: [{
                        "reservoir_seed": 1,
                        "lightgbm_seed": 10,
                        "metrics": metrics,
                        "delta_logloss": 0.0,
                        "delta_brier": 0.0,
                        "delta_auc": 0.0,
                        "feature_count": 4,
                        "single_row_latency": None,
                    }]},
                    "feature_generation_seconds": {candidate_id: 0.0},
                    "lightgbm_fit_seconds": 0.0,
                })

                class Model:
                    @staticmethod
                    def predict(features):
                        return np.full(len(features), 0.5, dtype=np.float32)

                arrays = {
                    "base_opened": opened,
                    "label_available": available,
                    "decision": decision,
                    "target": np.zeros(n, dtype=np.int8),
                    "base_x": np.zeros((n, 3), dtype=np.float32),
                }
                with (
                    patch.object(audit, "_generate_full_features", return_value=np.zeros((n, 1), dtype=np.float32)),
                    patch.object(audit, "_metric_pack", return_value=metrics),
                    patch.object(audit, "_fit_lgbm", return_value=Model()) as fit_lgbm,
                ):
                    result = audit._downstream_search_validation(
                        [finalist], {"selected_candidate_ids": [], "portfolio_variants": {}}, arrays,
                        stage_name="resume_test",
                    )

                self.assertEqual(result["candidate_results"][candidate_id]["pair_count"], 2)
                self.assertEqual(fit_lgbm.call_count, 2)
                self.assertEqual(
                    [call.args[-1] for call in fit_lgbm.call_args_list],
                    [11, 11],
                )
            finally:
                audit.OUT = previous_out
                audit.RUN_IDENTITY = previous_identity
                audit.SEEDS = previous_seeds
                audit.LGBM_SEEDS = previous_lgbm_seeds

    def test_controlled_anchor_pool_covers_every_family_and_topology_at_128_units(self):
        anchors = _architecture_anchors()
        self.assertEqual(len(anchors), 6)
        evaluated = {}
        for anchor_id, params, _ in anchors:
            config = _trial_config(optuna.trial.FixedTrial(params), "architecture")
            record = _config_record(config)
            evaluated[anchor_id] = record
            self.assertEqual(record["total_units"], 128)
            self.assertEqual(record["input_groups"], CONFIG["screening"]["common_input_groups"])
        self.assertEqual(
            {(row["architecture"], row["topology"]) for row in evaluated.values()},
            {(family, topology) for family in ("single", "parallel", "deep") for topology in ("ring", "sparse")},
        )
        self.assertEqual({row["leak_rate"] for row in evaluated.values()}, {0.5})
        self.assertEqual({row["spectral_radius"] for row in evaluated.values()}, {0.3})
        self.assertEqual({row["input_scaling"] for row in evaluated.values()}, {0.3})

    def test_input_group_stage_covers_each_family_and_each_group_separately(self):
        bases = {
            "single": _config_record(ESNConfig(architecture="single", reservoir_size=128, reservoir_count=1)),
            "parallel": _config_record(ESNConfig(architecture="parallel", reservoir_size=64, reservoir_count=2)),
            "deep": _config_record(ESNConfig(architecture="deep", reservoir_size=64, reservoir_count=2)),
        }
        anchors = _input_anchors(bases)
        self.assertEqual(len(anchors), 18)
        seen = set()
        for anchor_id, params, _ in anchors:
            config = _trial_config(optuna.trial.FixedTrial(params), "input_search", bases)
            self.assertEqual(len(config.input_groups), 1)
            seen.add((config.architecture, config.input_groups[0]))
            self.assertIn(anchor_id, f"{params['family_slot']}_{config.input_groups[0]}")
        self.assertEqual(
            seen,
            {(family, group) for family in ("single", "parallel", "deep") for group in CONFIG["screening"]["input_groups"]},
        )

    def test_search_scope_excludes_rows_and_labels_at_or_after_cutoff(self):
        opened = pd.to_datetime([
            "2025-06-30T23:50:00Z",
            "2025-07-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
        ]).asi8
        available = np.asarray([
            pd.Timestamp("2025-06-30T23:57:00Z").value,
            pd.Timestamp("2025-07-01T00:07:00Z").value,
            pd.Timestamp("2026-07-01T00:07:00Z").value,
        ], dtype=np.int64)
        np.testing.assert_array_equal(_scope_mask(opened, available), [True, False, False])
        self.assertEqual(SEARCH_END_NS, pd.Timestamp("2025-07-01T00:00:00Z").value)

    def test_effective_cache_identity_roundtrips_component_inputs_and_dynamics(self):
        config = ESNConfig(
            architecture="parallel_deep", reservoir_size=4, reservoir_count=4,
            branch_count=2, layer_count=2, component_sizes=(4, 5, 4, 5),
            component_input_groups=(("returns", "range_volatility"), (), ("volume", "basis"), ()),
            component_topologies=("ring", "sparse", "block_sparse", "orthogonal"),
            component_leak_rates=(0.2, 0.3, 0.4, 0.5),
            component_spectral_radii=(0.4, 0.6, 0.8, 1.1),
            component_input_scalings=(0.2, 0.3, 0.4, 0.5),
            component_bias_scalings=(0.0, 0.1, 0.2, 0.3),
            component_sparse_degrees=(2, 3, 4, 5),
            sparse_degree=3,
            input_groups=("returns", "range_volatility", "volume", "basis"),
        )
        record = _config_record(config)
        restored = _config_from_record(record)
        self.assertEqual(_config_record(restored), record)
        changed_groups = ESNConfig(
            **{
                **config.__dict__,
                "component_input_groups": (("volume", "basis"), (), ("returns", "range_volatility"), ()),
            },
        )
        self.assertNotEqual(_config_key(config), _config_key(changed_groups))
        self.assertEqual(record["total_units"], 18)
        self.assertEqual(record["components"][2]["input_columns"], ["log_volume", "volume_ratio_to_prior_30m_median", "index_futures_log_basis", "index_futures_basis_change_1m", "index_futures_basis_change_5m"])

    def test_runner_reads_only_the_frozen_search_cache_and_has_no_later_data_path(self):
        self.assertTrue(CONFIG["reused_search_cache_directory"].endswith("/prepared/search"))
        for key in ("reused_search_cache_directory", "reused_baseline_logits_path", "reused_baseline_metadata_path"):
            value = CONFIG[key].lower()
            self.assertNotIn("confirmation", value)
            self.assertNotIn("holdout", value)
        self.assertFalse(CONFIG["excluded_data"]["read_after_cutoff"])
        self.assertFalse(CONFIG["excluded_data"]["holdout_evaluation"])
        self.assertEqual(CONFIG["excluded_data"]["materialization_cutoff_exclusive_utc"], "2025-07-01T00:00:00Z")
        self.assertTrue(set(CONFIG["screening"]["reservoir_seeds"]).isdisjoint(CONFIG["downstream"]["lightgbm_seeds"]))


if __name__ == "__main__":
    unittest.main()
