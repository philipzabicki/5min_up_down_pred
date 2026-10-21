from __future__ import annotations

import unittest

import numpy as np
import optuna
import pandas as pd

from run_btc_esn_architecture_audit_20261020 import (
    CONFIG,
    SEARCH_END_NS,
    _architecture_anchors,
    _config_record,
    _input_anchors,
    _scope_mask,
    _trial_config,
)
from features.esn_reservoir import ESNConfig


class ESNArchitectureAuditTests(unittest.TestCase):
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
