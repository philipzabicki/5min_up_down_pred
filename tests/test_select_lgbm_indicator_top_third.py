from pathlib import Path

import pandas as pd
import pytest

import select_lgbm_indicator_top_third as selector


def test_keep_fraction_count_rounds_up():
    assert selector.keep_fraction_count(1) == 1
    assert selector.keep_fraction_count(3) == 1
    assert selector.keep_fraction_count(4) == 2
    assert selector.keep_fraction_count(651) == 217


def test_collect_prefilter_removed_features_preserves_first_seen_order():
    filter_report_df = pd.DataFrame(
        [
            {
                "removed_features_json": "[]",
            },
            {
                "removed_features_json": "[\"f2\", \"f3\"]",
            },
            {
                "removed_features_json": "[\"f1\", \"f2\"]",
            },
        ]
    )

    assert selector.collect_prefilter_removed_features(filter_report_df) == [
        "f2",
        "f3",
        "f1",
    ]


def test_partition_features_covers_entire_candidate_pool():
    candidate_features = ["f1", "f2", "f3", "f4", "f5", "f6"]
    ranked_features = ["f4", "f2", "f6", "f3"]

    important_features, less_important_features = selector.partition_features(
        candidate_features,
        ranked_features,
        keep_count=2,
    )

    assert important_features == ["f4", "f2"]
    assert less_important_features == ["f1", "f3", "f5", "f6"]


def test_apply_move_plan_moves_files_and_is_idempotent(tmp_path):
    src_dir = tmp_path / "all"
    dest_dir = tmp_path / "less_important"
    src_dir.mkdir()
    feature_path = src_dir / "feature_a.json"
    feature_path.write_text("{}", encoding="utf-8")

    move_plan = [
        {
            "feature": "feature_a",
            "src": str(feature_path),
            "dest": str(dest_dir / "feature_a.json"),
        }
    ]

    first = selector.apply_move_plan(move_plan, dry_run=False)
    second = selector.apply_move_plan(move_plan, dry_run=False)

    assert first["moved_count"] == 1
    assert second["already_moved_count"] == 1
    assert not feature_path.exists()
    assert (dest_dir / "feature_a.json").exists()


def test_build_move_plan_uses_original_file_name(tmp_path):
    json_path = tmp_path / "all" / "feature_x.json"
    configs_by_feature = {
        "feature_x": {
            "json_path": json_path,
        }
    }

    move_plan = selector.build_move_plan(
        configs_by_feature,
        ["feature_x"],
        tmp_path / "less_important",
    )

    assert move_plan == [
        {
            "feature": "feature_x",
            "src": str(json_path),
            "dest": str(tmp_path / "less_important" / "feature_x.json"),
        }
    ]


def test_build_indicator_dataset_settings_forces_float32_and_disables_vp(tmp_path):
    modeling_settings = {
        "fit_results_dir": Path("data/features/indicators_fit/all"),
        "feature_subset_path": None,
        "feature_subset_list_key": None,
        "excluded_feature_names": ("f1",),
        "output_suffix": "_model_ready",
        "float_precision": "float64",
        "volume_profile_fixed_range": {"enabled": True, "foo": "bar"},
    }

    dataset_settings = selector.build_indicator_dataset_settings(
        "run_1",
        tmp_path / "candidate.json",
        modeling_settings,
    )

    assert dataset_settings["fit_results_dir"] == selector.INDICATOR_ALL_DIR
    assert dataset_settings["feature_subset_list_key"] == "feature_columns"
    assert dataset_settings["excluded_feature_names"] == tuple()
    assert dataset_settings["output_suffix"] == "_indicator_top_third_selector_run_1"
    assert dataset_settings["float_precision"] == "float32"
    assert dataset_settings["volume_profile_fixed_range"]["enabled"] is False
