import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import select_lgbm_feature_plateau as feature_plateau
from create_modeling_dataset import build_dataset_from_settings, parse_fit_results
from modeling_dataset_utils import load_feature_subset, load_modeling_dataset_settings

INDICATOR_ALL_DIR = Path("data/features/indicators_fit/all")
LOW_IMPORTANCE_DIR = Path("data/features/indicators_fit/less_important")
OUTPUT_ROOT = Path("data/analysis/indicator_feature_selector")
KEEP_NUMERATOR = 1
KEEP_DENOMINATOR = 3
DATASET_OUTPUT_SUFFIX_PREFIX = "_indicator_top_third_selector_"
INDICATOR_SELECTOR_FLOAT_PRECISION = "float32"


def utc_now():
    return datetime.now(timezone.utc)


def default_run_id():
    return utc_now().strftime("%Y%m%d_%H%M%S")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_feature_text(path, features, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in (metadata or {}).items():
        lines.append(f"{key}={value}")
    if lines:
        lines.append("")
    lines.extend(str(feature) for feature in features)
    path.write_text("\n".join(lines), encoding="utf-8")


def keep_fraction_count(total_count, *, numerator=KEEP_NUMERATOR, denominator=KEEP_DENOMINATOR):
    total_count = int(total_count)
    numerator = int(numerator)
    denominator = int(denominator)
    if total_count <= 0:
        raise ValueError("total_count must be > 0.")
    if numerator <= 0 or denominator <= 0:
        raise ValueError("keep fraction must be positive.")
    if numerator > denominator:
        raise ValueError("keep fraction cannot exceed 1.")
    return int(math.ceil(total_count * numerator / denominator))


def collect_prefilter_removed_features(filter_report_df):
    removed_features = []
    seen = set()
    for _, row in filter_report_df.iterrows():
        payload = json.loads(row["removed_features_json"])
        for feature_name in payload:
            feature_name = str(feature_name)
            if feature_name in seen:
                continue
            seen.add(feature_name)
            removed_features.append(feature_name)
    return removed_features


def build_move_plan(configs_by_feature, feature_names, destination_dir):
    destination_dir = Path(destination_dir)
    move_plan = []
    for feature_name in feature_names:
        cfg = configs_by_feature.get(feature_name)
        if cfg is None:
            continue
        src_path = Path(cfg["json_path"])
        move_plan.append(
            {
                "feature": feature_name,
                "src": str(src_path),
                "dest": str(destination_dir / src_path.name),
            }
        )
    return move_plan


def apply_move_plan(move_plan, *, dry_run):
    moved_features = []
    already_moved_features = []
    missing_features = []

    for item in move_plan:
        src_path = Path(item["src"])
        dest_path = Path(item["dest"])

        if src_path.exists():
            if dry_run:
                moved_features.append(str(item["feature"]))
                continue
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if dest_path.exists():
                raise FileExistsError(
                    "Cannot move indicator config because destination already exists: "
                    f"{dest_path}"
                )
            shutil.move(str(src_path), str(dest_path))
            moved_features.append(str(item["feature"]))
            continue

        if dest_path.exists():
            already_moved_features.append(str(item["feature"]))
            continue

        missing_features.append(str(item["feature"]))

    if missing_features:
        preview = ", ".join(missing_features[:10])
        raise FileNotFoundError(
            "Some indicator config files were missing in both source and destination. "
            f"Missing_count={len(missing_features)} preview=[{preview}]"
        )

    return {
        "planned_count": len(move_plan),
        "moved_count": len(moved_features),
        "already_moved_count": len(already_moved_features),
        "moved_features": moved_features,
        "already_moved_features": already_moved_features,
    }


def partition_features(candidate_features, ranked_features, keep_count):
    keep_count = int(keep_count)
    candidate_features = [str(feature) for feature in candidate_features]
    ranked_features = [str(feature) for feature in ranked_features]
    if keep_count <= 0:
        raise ValueError("keep_count must be > 0.")
    if keep_count > len(ranked_features):
        raise ValueError(
            "keep_count cannot exceed the ranked feature pool size. "
            f"keep_count={keep_count} ranked={len(ranked_features)}"
        )

    important_features = ranked_features[:keep_count]
    important_feature_set = set(important_features)
    if len(important_feature_set) != len(important_features):
        raise ValueError("Top-ranked feature list contains duplicates.")

    candidate_feature_set = set(candidate_features)
    missing_ranked = [feature for feature in important_features if feature not in candidate_feature_set]
    if missing_ranked:
        preview = ", ".join(missing_ranked[:10])
        raise ValueError(
            "Ranked features contain items outside the candidate pool. "
            f"Missing_count={len(missing_ranked)} preview=[{preview}]"
        )

    less_important_features = [
        feature for feature in candidate_features if feature not in important_feature_set
    ]
    if len(important_features) + len(less_important_features) != len(candidate_features):
        raise ValueError("Feature partition size mismatch.")
    return important_features, less_important_features


def collect_indicator_configs(config_dir):
    configs = parse_fit_results(config_dir)
    return configs, {str(cfg["feature_col"]): cfg for cfg in configs}


def collect_optional_indicator_features(config_dir):
    config_dir = Path(config_dir)
    if not config_dir.exists():
        return []
    configs, _ = collect_indicator_configs(config_dir)
    return [str(cfg["feature_col"]) for cfg in configs]


def load_seed_sets(all_feature_names):
    settings = load_modeling_dataset_settings()
    all_feature_set = set(all_feature_names)
    less_important_feature_set = set(collect_optional_indicator_features(LOW_IMPORTANCE_DIR))
    excluded_feature_names = list(settings.get("excluded_feature_names") or ())
    seed_less_important = [
        feature_name
        for feature_name in excluded_feature_names
        if feature_name in all_feature_set or feature_name in less_important_feature_set
    ]

    feature_subset_path = settings.get("feature_subset_path")
    feature_subset_list_key = settings.get("feature_subset_list_key")
    seed_important = []
    if feature_subset_path:
        subset_info = load_feature_subset(
            feature_subset_path,
            list_key=feature_subset_list_key,
        )
        seed_important = [
            feature_name
            for feature_name in subset_info["features"]
            if feature_name in all_feature_set and "_fit_" in feature_name
        ]

    return {
        "modeling_settings": settings,
        "seed_less_important": seed_less_important,
        "seed_less_important_in_all": [
            feature_name for feature_name in seed_less_important if feature_name in all_feature_set
        ],
        "seed_less_important_already_moved": [
            feature_name
            for feature_name in seed_less_important
            if feature_name not in all_feature_set and feature_name in less_important_feature_set
        ],
        "seed_important": seed_important,
    }


def write_candidate_artifact(run_dir, candidate_features):
    artifact_path = Path(run_dir) / "candidate_indicator_features.json"
    payload = {
        "created_utc": utc_now().isoformat(),
        "source_fit_results_dir": str(INDICATOR_ALL_DIR),
        "feature_count": len(candidate_features),
        "feature_columns": list(candidate_features),
    }
    write_json(artifact_path, payload)
    write_feature_text(
        Path(run_dir) / "candidate_indicator_features.txt",
        candidate_features,
        metadata={
            "created_utc": payload["created_utc"],
            "feature_count": payload["feature_count"],
            "source_fit_results_dir": payload["source_fit_results_dir"],
        },
    )
    return artifact_path


def write_seed_artifacts(
    run_dir,
    *,
    all_feature_names,
    candidate_features,
    seed_important,
    seed_less_important,
    seed_less_important_in_all,
    seed_less_important_already_moved,
    seed_move_summary,
):
    keep_count = keep_fraction_count(len(candidate_features))
    payload = {
        "created_utc": utc_now().isoformat(),
        "source_fit_results_dir": str(INDICATOR_ALL_DIR),
        "less_important_dir": str(LOW_IMPORTANCE_DIR),
        "all_indicator_feature_count": len(all_feature_names),
        "candidate_indicator_feature_count": len(candidate_features),
        "target_keep_count": keep_count,
        "keep_fraction_numerator": KEEP_NUMERATOR,
        "keep_fraction_denominator": KEEP_DENOMINATOR,
        "seed_important_count": len(seed_important),
        "seed_less_important_count": len(seed_less_important),
        "seed_less_important_in_all_count": len(seed_less_important_in_all),
        "seed_less_important_already_moved_count": len(
            seed_less_important_already_moved
        ),
        "seed_move_summary": seed_move_summary,
        "seed_important_features": list(seed_important),
        "seed_less_important_features": list(seed_less_important),
        "seed_less_important_in_all": list(seed_less_important_in_all),
        "seed_less_important_already_moved": list(seed_less_important_already_moved),
    }
    write_json(Path(run_dir) / "seed_state.json", payload)
    write_json(
        Path(run_dir) / "seed_important_features.json",
        {
            "created_utc": payload["created_utc"],
            "feature_count": len(seed_important),
            "final_feature_list": list(seed_important),
        },
    )
    write_json(
        Path(run_dir) / "seed_less_important_features.json",
        {
            "created_utc": payload["created_utc"],
            "feature_count": len(seed_less_important),
            "feature_columns": list(seed_less_important),
        },
    )
    write_feature_text(
        Path(run_dir) / "seed_important_features.txt",
        seed_important,
        metadata={"feature_count": len(seed_important)},
    )
    write_feature_text(
        Path(run_dir) / "seed_less_important_features.txt",
        seed_less_important,
        metadata={"feature_count": len(seed_less_important)},
    )


def build_indicator_dataset_settings(run_id, candidate_artifact_path, modeling_settings):
    dataset_settings = dict(modeling_settings)
    volume_profile_cfg = dict(dataset_settings.get("volume_profile_fixed_range") or {})
    volume_profile_cfg["enabled"] = False
    dataset_settings.update(
        {
            "fit_results_dir": INDICATOR_ALL_DIR,
            "feature_subset_path": Path(candidate_artifact_path),
            "feature_subset_list_key": "feature_columns",
            "excluded_feature_names": tuple(),
            "output_suffix": f"{DATASET_OUTPUT_SUFFIX_PREFIX}{run_id}",
            "float_precision": INDICATOR_SELECTOR_FLOAT_PRECISION,
            "volume_profile_fixed_range": volume_profile_cfg,
        }
    )
    return dataset_settings


def build_indicator_only_dataset(run_dir, run_id, candidate_artifact_path, modeling_settings):
    dataset_settings = build_indicator_dataset_settings(
        run_id,
        candidate_artifact_path,
        modeling_settings,
    )
    write_json(
        Path(run_dir) / "indicator_dataset_settings_snapshot.json",
        {
            "created_utc": utc_now().isoformat(),
            "fit_results_dir": str(dataset_settings["fit_results_dir"]),
            "feature_subset_path": str(dataset_settings["feature_subset_path"]),
            "feature_subset_list_key": str(dataset_settings["feature_subset_list_key"]),
            "excluded_feature_names_count": len(dataset_settings["excluded_feature_names"]),
            "output_suffix": str(dataset_settings["output_suffix"]),
            "float_precision": str(dataset_settings["float_precision"]),
            "volume_profile_fixed_range": dataset_settings["volume_profile_fixed_range"],
        },
    )
    return build_dataset_from_settings(dataset_settings)


def summarize_fold_weighting(
    ranking_folds,
    permutation_folds,
    topk_folds,
    ranking_weights,
    permutation_weights,
    topk_weights,
):
    return {
        "enabled": bool(feature_plateau.ENABLE_FOLD_RECENCY_WEIGHTING),
        "active": bool(feature_plateau.is_nontrivial_fold_recency_weighting_enabled()),
        "mode": str(feature_plateau.FOLD_RECENCY_WEIGHTING_MODE),
        "min_weight": float(feature_plateau.FOLD_RECENCY_WEIGHT_MIN),
        "max_weight": float(feature_plateau.FOLD_RECENCY_WEIGHT_MAX),
        "topk_std_score_aggregation": "unweighted",
        "ranking_fold_weights": feature_plateau.fold_weight_items_for_summary(
            ranking_folds,
            ranking_weights,
        ),
        "permutation_fold_weights": feature_plateau.fold_weight_items_for_summary(
            permutation_folds,
            permutation_weights,
        ),
        "topk_fold_weights": feature_plateau.fold_weight_items_for_summary(
            topk_folds,
            topk_weights,
        ),
    }


def run_selection(
    *,
    run_dir,
    dataset_path,
    candidate_features,
    seed_important,
    seed_less_important,
):
    print(f"load indicator-only dataset | path={dataset_path}")
    df = feature_plateau.load_dataframe(dataset_path)
    if feature_plateau.TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column not found in indicator-only dataset: {feature_plateau.TARGET_COL}"
        )

    missing_candidate_features = [
        feature_name for feature_name in candidate_features if feature_name not in df.columns
    ]
    if missing_candidate_features:
        preview = ", ".join(missing_candidate_features[:10])
        raise ValueError(
            "Indicator-only dataset is missing requested candidate features. "
            f"Missing_count={len(missing_candidate_features)} preview=[{preview}]"
        )

    df = df[df[feature_plateau.TARGET_COL].notna()].copy()
    if df.empty:
        raise ValueError("No rows left after TARGET_COL non-null filtering.")

    (
        df,
        sample_weight,
        sample_weight_source,
        sample_weight_summary,
        row_filter_info,
    ) = feature_plateau.filter_rows_by_min_sample_weight(
        df,
        context_label="indicator selector load data",
    )

    x_raw = df[list(candidate_features)].replace([np.inf, -np.inf], np.nan)
    (
        x_prefilter,
        filter_report_df,
        duplicate_map,
        high_corr_drop_map,
    ) = feature_plateau.prefilter_features(x_raw)
    removed_in_prefilter = collect_prefilter_removed_features(filter_report_df)
    y, class_mapping = feature_plateau.prepare_binary_target(df[feature_plateau.TARGET_COL])

    ranking_folds = feature_plateau.make_walk_forward_folds(
        n_rows=len(df),
        n_splits=feature_plateau.RANKING_N_SPLITS,
        test_to_train_ratio=feature_plateau.WF_TEST_TO_TRAIN_RATIO,
    )
    permutation_folds = feature_plateau.make_walk_forward_folds(
        n_rows=len(df),
        n_splits=feature_plateau.PERMUTATION_N_SPLITS,
        test_to_train_ratio=feature_plateau.WF_TEST_TO_TRAIN_RATIO,
    )
    topk_folds = feature_plateau.make_walk_forward_folds(
        n_rows=len(df),
        n_splits=feature_plateau.TOPK_N_SPLITS,
        test_to_train_ratio=feature_plateau.WF_TEST_TO_TRAIN_RATIO,
    )
    ranking_weights = feature_plateau.build_fold_recency_weights(ranking_folds)
    permutation_weights = feature_plateau.build_fold_recency_weights(permutation_folds)
    topk_weights = feature_plateau.build_fold_recency_weights(topk_folds)

    print(
        f"indicator selector | candidate_features={len(candidate_features)} "
        f"after_prefilter={x_prefilter.shape[1]}"
    )
    feature_plateau.print_prefilter_report_cli(filter_report_df)

    ranking_df, _, fold_metadata = feature_plateau.run_feature_ranking(
        x=x_prefilter,
        y=y,
        sample_weight=sample_weight,
        prescreen_folds=ranking_folds,
        prescreen_fold_weight_by_id=ranking_weights,
        permutation_folds=permutation_folds,
        permutation_fold_weight_by_id=permutation_weights,
    )
    ranked_features = ranking_df["feature"].tolist()
    keep_count = keep_fraction_count(len(candidate_features))
    important_features, less_important_features = partition_features(
        candidate_features,
        ranked_features,
        keep_count,
    )
    important_feature_set = set(important_features)
    seed_important_selected = [
        feature_name for feature_name in seed_important if feature_name in important_feature_set
    ]

    evaluation_row = feature_plateau.score_topk_subset(
        x=x_prefilter,
        y=y,
        sample_weight=sample_weight,
        folds=topk_folds,
        fold_weight_by_id=topk_weights,
        global_feature_order=ranked_features,
        k=keep_count,
        phase="fixed_top_third",
    )

    summary_payload = {
        "created_utc": utc_now().isoformat(),
        "data_path": str(dataset_path),
        "target_col": feature_plateau.TARGET_COL,
        "scorer": feature_plateau.SCORER["name"],
        "ranking_method": (
            "prescreen_used_folds_recency_weighted_mean_gain_"
            "then_permutation_recency_weighted_delta_logloss"
        ),
        "selection_mode": "fixed_top_fraction",
        "selection_statement": (
            "zostawiamy dokladnie 1/3 indicator features po seed exclusions; "
            "kolejnosc bierze sie z tego samego rankingu co w select_lgbm_feature_plateau.py"
        ),
        "keep_fraction_numerator": KEEP_NUMERATOR,
        "keep_fraction_denominator": KEEP_DENOMINATOR,
        "input_feature_count": len(candidate_features),
        "prefilter_feature_count": int(x_prefilter.shape[1]),
        "prefilter_removed_feature_count": len(removed_in_prefilter),
        "prefilter_removed_features": removed_in_prefilter,
        "eligible_feature_count": int(ranking_df["eligible_for_selection"].sum()),
        "permutation_feature_fraction": float(
            feature_plateau.PERMUTATION_FEATURE_FRACTION
        ),
        "permutation_top_n": int(
            feature_plateau.resolve_permutation_feature_limit(
                ranking_df["eligible_for_selection"].sum()
            )
        ),
        "permutation_n_repeats": int(feature_plateau.PERMUTATION_N_REPEATS),
        "permutation_candidate_count": int(ranking_df["prescreen_candidate"].sum()),
        "permutation_ranked_feature_count": int(
            ranking_df["permutation_mean_delta_logloss"].notna().sum()
        ),
        "duplicate_column_map": duplicate_map,
        "high_corr_drop_map": high_corr_drop_map,
        "sample_weight": {
            "used": bool(feature_plateau.USE_SAMPLE_WEIGHTS),
            "source": sample_weight_source,
            **sample_weight_summary,
        },
        "row_filter": row_filter_info,
        "ranking_n_splits": len(ranking_folds),
        "permutation_n_splits": len(permutation_folds),
        "topk_n_splits": len(topk_folds),
        "walk_forward_test_to_train_ratio": float(feature_plateau.WF_TEST_TO_TRAIN_RATIO),
        "random_seeds": [int(seed) for seed in feature_plateau.RANDOM_SEEDS],
        "fold_recency_weighting": summarize_fold_weighting(
            ranking_folds,
            permutation_folds,
            topk_folds,
            ranking_weights,
            permutation_weights,
            topk_weights,
        ),
        "topk_selection_mode": feature_plateau.TOPK_SELECTION_MODE,
        "topk_selection_std_coef": float(feature_plateau.TOPK_SELECTION_STD_COEF),
        "topk_selection_formula": feature_plateau.topk_selection_formula(),
        "topk_selection_base_score": feature_plateau.topk_selection_base_score_label(),
        "selected_feature_count": len(important_features),
        "less_important_feature_count": len(less_important_features),
        "seed_important_count": len(seed_important),
        "seed_important_selected_count": len(seed_important_selected),
        "seed_important_selected_features": seed_important_selected,
        "seed_less_important_count": len(seed_less_important),
        "seed_less_important_features": list(seed_less_important),
        "topk_evaluation": evaluation_row,
        "final_feature_list": important_features,
        "less_important_feature_list": less_important_features,
        "class_mapping": class_mapping,
        "fold_ranking_metadata": fold_metadata,
    }

    filter_report_df.to_csv(Path(run_dir) / "feature_filter_report.csv", index=False)
    ranking_df.to_csv(Path(run_dir) / "indicator_feature_ranking.csv", index=False)
    write_json(Path(run_dir) / "selection_summary.json", summary_payload)
    write_json(
        Path(run_dir) / "top_third_selected_features.json",
        {
            "created_utc": summary_payload["created_utc"],
            "data_path": summary_payload["data_path"],
            "feature_count": len(important_features),
            "final_feature_list": important_features,
        },
    )
    write_json(
        Path(run_dir) / "less_important_features.json",
        {
            "created_utc": summary_payload["created_utc"],
            "data_path": summary_payload["data_path"],
            "feature_count": len(less_important_features),
            "feature_columns": less_important_features,
        },
    )
    write_feature_text(
        Path(run_dir) / "top_third_selected_features.txt",
        important_features,
        metadata={
            "data_path": summary_payload["data_path"],
            "feature_count": len(important_features),
        },
    )
    write_feature_text(
        Path(run_dir) / "less_important_features.txt",
        less_important_features,
        metadata={
            "data_path": summary_payload["data_path"],
            "feature_count": len(less_important_features),
        },
    )
    return {
        "summary_payload": summary_payload,
        "important_features": important_features,
        "less_important_features": less_important_features,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Seed and run an indicator-only feature selection workflow that keeps "
            "the top 1/3 of configs from data/features/indicators_fit/all."
        )
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Only seed the less-important directory from modeling.excluded_feature_names "
            "and build the indicator-only modeling dataset."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan config moves without touching files.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run identifier used under data/analysis/indicator_feature_selector.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = str(args.run_id).strip() or default_run_id()
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"indicator selector | run_dir={run_dir}")
    all_configs, all_configs_by_feature = collect_indicator_configs(INDICATOR_ALL_DIR)
    all_feature_names = [str(cfg["feature_col"]) for cfg in all_configs]
    seed_info = load_seed_sets(all_feature_names)
    seed_less_important = list(seed_info["seed_less_important"])
    seed_less_important_in_all = list(seed_info["seed_less_important_in_all"])
    seed_less_important_already_moved = list(
        seed_info["seed_less_important_already_moved"]
    )
    seed_important = list(seed_info["seed_important"])
    seed_move_plan = build_move_plan(
        all_configs_by_feature,
        seed_less_important_in_all,
        LOW_IMPORTANCE_DIR,
    )
    write_json(Path(run_dir) / "seed_move_plan.json", {"move_plan": seed_move_plan})
    seed_move_summary = apply_move_plan(seed_move_plan, dry_run=bool(args.dry_run))
    candidate_feature_names = [
        feature_name
        for feature_name in all_feature_names
        if feature_name not in set(seed_less_important)
    ]
    candidate_configs_by_feature = {
        feature_name: cfg
        for feature_name, cfg in all_configs_by_feature.items()
        if feature_name in set(candidate_feature_names)
    }

    write_seed_artifacts(
        run_dir,
        all_feature_names=all_feature_names,
        candidate_features=candidate_feature_names,
        seed_important=seed_important,
        seed_less_important=seed_less_important,
        seed_less_important_in_all=seed_less_important_in_all,
        seed_less_important_already_moved=seed_less_important_already_moved,
        seed_move_summary=seed_move_summary,
    )
    candidate_artifact_path = write_candidate_artifact(run_dir, candidate_feature_names)

    print(
        f"indicator selector | all={len(all_feature_names)} "
        f"seed_less_important={len(seed_less_important)} "
        f"candidate={len(candidate_feature_names)} "
        f"target_keep={keep_fraction_count(len(candidate_feature_names))}"
    )
    dataset_path = build_indicator_only_dataset(
        run_dir,
        run_id,
        candidate_artifact_path,
        seed_info["modeling_settings"],
    )
    print(f"indicator selector | indicator-only dataset ready -> {dataset_path}")

    if args.prepare_only:
        print("indicator selector | prepare-only complete")
        return

    selection_output = run_selection(
        run_dir=run_dir,
        dataset_path=dataset_path,
        candidate_features=candidate_feature_names,
        seed_important=seed_important,
        seed_less_important=seed_less_important,
    )

    final_move_plan = build_move_plan(
        candidate_configs_by_feature,
        selection_output["less_important_features"],
        LOW_IMPORTANCE_DIR,
    )
    write_json(Path(run_dir) / "final_move_plan.json", {"move_plan": final_move_plan})
    final_move_summary = apply_move_plan(final_move_plan, dry_run=bool(args.dry_run))
    write_json(Path(run_dir) / "final_move_summary.json", final_move_summary)

    print(
        f"indicator selector | selected={len(selection_output['important_features'])} "
        f"moved_to_less_important={final_move_summary['moved_count']} "
        f"already_moved={final_move_summary['already_moved_count']}"
    )


if __name__ == "__main__":
    main()
