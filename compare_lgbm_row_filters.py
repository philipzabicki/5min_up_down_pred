import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common_config_utils import path_to_portable_str
from modeling_dataset_utils import (
    load_excluded_feature_names_from_settings,
    load_feature_subset_from_settings,
    load_modeling_dataset_settings,
    resolve_modeling_dataset_output_paths,
    resolve_modeling_float_dtype,
    resolve_modeling_float_dtype_name,
    summarize_feature_subset,
)
from target_weights import TARGET_WEIGHT_COL, compute_decision_mask_from_opened
from train_lgbm import (
    CV_FOLDS as DEFAULT_CV_FOLDS,
    EARLY_STOPPING_EVAL_METRIC,
    EARLY_STOPPING_ROUNDS as DEFAULT_EARLY_STOPPING_ROUNDS,
    LGBM_DEFAULT_PARAMS,
    LGBM_OPTUNA_BEST_PARAMS,
    N_ESTIMATORS as DEFAULT_N_ESTIMATORS,
    OOF_EXPORT_BASE_COLS,
    OOF_PREVIEW_ROWS as DEFAULT_PREVIEW_ROWS,
    TARGET_COL,
    WF_TEST_TO_TRAIN_RATIO as DEFAULT_WF_TEST_TO_TRAIN_RATIO,
    build_lgbm_model,
    classification_metrics,
    clean_and_impute_fold,
    format_lgbm_monotone_constraint_summary,
    load_walk_forward_training_frame,
    make_walk_forward_folds,
    summarize_lgbm_monotone_constraints,
)

ROW_MODE_ALL_ROWS = "all_rows"
ROW_MODE_DECISION_ONLY = "decision_rows_only"
SUPPORTED_ROW_MODES = (ROW_MODE_ALL_ROWS, ROW_MODE_DECISION_ONLY)
LOWER_IS_BETTER_METRICS = {"brier_score", "binary_logloss"}
PRIMARY_SCOPE = "decision_rows"
PRIMARY_METRIC = "balanced_accuracy"
DEFAULT_OUTPUT_DIR = Path("data/analysis/lgbm_row_filter_compare")


def utc_now():
    return datetime.now(timezone.utc)


def make_run_timestamp():
    return utc_now().strftime("%Y%m%d_%H%M%S")


def build_cli_args():
    parser = argparse.ArgumentParser(
        description=(
            "Porownaj trening LGBM na wszystkich wierszach vs tylko na "
            "decision rows, bez zmian w feature engineeringu."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Katalog bazowy na artefakty porownania; skrypt utworzy podkatalog z timestampem.",
    )
    parser.add_argument(
        "--params-source",
        choices=("optuna", "default"),
        default="optuna",
        help="Ktory zestaw parametrow LGBM porownac dla obu trybow row filter.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help="Liczba foldow walk-forward.",
    )
    parser.add_argument(
        "--test-to-train-ratio",
        type=float,
        default=DEFAULT_WF_TEST_TO_TRAIN_RATIO,
        help="Relacja dlugosci okna testowego do treningowego w walk-forward.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help="Maksymalna liczba drzew przed early stopping.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_EARLY_STOPPING_ROUNDS,
        help="Early stopping rounds; 0 wylacza early stopping.",
    )
    parser.add_argument(
        "--device-type",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Device type przekazywany do LightGBM.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Opcjonalny override dla liczby watkow LGBM.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=DEFAULT_PREVIEW_ROWS,
        help="Liczba wierszy dla podgladowych CSV z OOF.",
    )
    return parser.parse_args()


def summarize_boolean_mask(mask):
    mask_np = np.asarray(mask, dtype=bool)
    total_rows = int(mask_np.size)
    matched_rows = int(mask_np.sum())
    return {
        "rows": matched_rows,
        "total_rows": total_rows,
        "share": float(matched_rows / total_rows) if total_rows > 0 else 0.0,
    }


def select_fold_row_indices(train_indices, test_indices, decision_mask, row_mode):
    train_indices = np.asarray(train_indices, dtype=np.int32)
    test_indices = np.asarray(test_indices, dtype=np.int32)
    decision_mask_np = np.asarray(decision_mask, dtype=bool)

    if row_mode == ROW_MODE_ALL_ROWS:
        train_used = train_indices
        eval_used = test_indices
    elif row_mode == ROW_MODE_DECISION_ONLY:
        train_used = train_indices[decision_mask_np[train_indices]]
        eval_used = test_indices[decision_mask_np[test_indices]]
    else:
        raise ValueError(f"Unsupported row_mode: {row_mode}")

    return {
        "train_indices": train_used,
        "eval_indices": eval_used,
        "predict_indices": test_indices,
    }


def evaluate_prediction_scopes(y_true, y_pred_proba, sample_weight, decision_mask):
    y_true_np = np.asarray(y_true, dtype=np.int8)
    y_pred_np = np.asarray(y_pred_proba, dtype=np.float64)
    sample_weight_np = np.asarray(sample_weight, dtype=np.float64)
    decision_mask_np = np.asarray(decision_mask, dtype=bool)

    if y_true_np.shape[0] != y_pred_np.shape[0]:
        raise ValueError("Prediction scope evaluation received mismatched y_true/y_pred.")
    if y_true_np.shape[0] != sample_weight_np.shape[0]:
        raise ValueError(
            "Prediction scope evaluation received mismatched y_true/sample_weight."
        )
    if y_true_np.shape[0] != decision_mask_np.shape[0]:
        raise ValueError(
            "Prediction scope evaluation received mismatched y_true/decision_mask."
        )

    scopes = {}
    scope_masks = {
        "all_rows": np.ones(y_true_np.shape[0], dtype=bool),
        "decision_rows": decision_mask_np,
    }
    for scope_name, scope_mask in scope_masks.items():
        rows = int(scope_mask.sum())
        if rows == 0:
            scopes[scope_name] = {
                "rows": 0,
                "weight_sum": 0.0,
                "positive_rate": None,
                "metrics": None,
            }
            continue

        y_scope = y_true_np[scope_mask]
        pred_scope = y_pred_np[scope_mask]
        weight_scope = sample_weight_np[scope_mask]
        scopes[scope_name] = {
            "rows": rows,
            "weight_sum": float(weight_scope.sum()),
            "positive_rate": float(np.average(y_scope, weights=weight_scope)),
            "metrics": classification_metrics(
                y_scope,
                pred_scope,
                sample_weight=weight_scope,
            ),
        }
    return scopes


def build_model_param_overrides(args):
    if args.params_source == "optuna":
        params = dict(LGBM_OPTUNA_BEST_PARAMS)
    else:
        params = dict(LGBM_DEFAULT_PARAMS)
    params["device_type"] = str(args.device_type)
    if args.n_jobs is not None:
        params["n_jobs"] = int(args.n_jobs)
    return params


def metric_summary_from_fold_df(fold_df, scope_name):
    summary = {}
    metric_prefix = f"{scope_name}_"
    metric_cols = [
        col
        for col in fold_df.columns
        if col.startswith(metric_prefix)
        and col
        not in {
            f"{scope_name}_rows",
            f"{scope_name}_weight_sum",
            f"{scope_name}_positive_rate",
        }
    ]
    for metric_col in metric_cols:
        metric_name = metric_col[len(metric_prefix) :]
        metric_values = pd.to_numeric(fold_df[metric_col], errors="coerce").dropna()
        if metric_values.empty:
            continue
        summary[metric_name] = {
            "mean": float(metric_values.mean()),
            "std": float(metric_values.std(ddof=0)),
        }
    return summary


def build_variant_summary_row(variant_name, variant_summary):
    row = {
        "variant": variant_name,
        "row_mode": variant_summary["row_filter"]["mode"],
        "oof_rows": int(variant_summary["oof_rows"]),
        "oof_coverage_ratio": float(variant_summary["oof_coverage_ratio"]),
        "mean_best_iteration": float(variant_summary["best_iteration"]["mean"]),
        "median_best_iteration": float(variant_summary["best_iteration"]["median"]),
    }
    for scope_name, scope_payload in variant_summary["oof_metrics"].items():
        row[f"{scope_name}_rows"] = int(scope_payload["rows"])
        row[f"{scope_name}_weight_sum"] = float(scope_payload["weight_sum"])
        row[f"{scope_name}_positive_rate"] = scope_payload["positive_rate"]
        metrics = scope_payload["metrics"] or {}
        for metric_name, metric_value in metrics.items():
            row[f"{scope_name}_{metric_name}"] = float(metric_value)
    return row


def compare_scope_metrics(base_variant, candidate_variant, scope_name):
    base_scope = base_variant["oof_metrics"][scope_name]
    candidate_scope = candidate_variant["oof_metrics"][scope_name]
    base_metrics = base_scope["metrics"] or {}
    candidate_metrics = candidate_scope["metrics"] or {}

    delta = {}
    winner_by_metric = {}
    for metric_name in sorted(set(base_metrics) & set(candidate_metrics)):
        diff_value = float(candidate_metrics[metric_name] - base_metrics[metric_name])
        delta[metric_name] = diff_value
        if abs(diff_value) <= 1e-15:
            winner_by_metric[metric_name] = "tie"
        elif metric_name in LOWER_IS_BETTER_METRICS:
            winner_by_metric[metric_name] = (
                ROW_MODE_DECISION_ONLY if diff_value < 0.0 else ROW_MODE_ALL_ROWS
            )
        else:
            winner_by_metric[metric_name] = (
                ROW_MODE_DECISION_ONLY if diff_value > 0.0 else ROW_MODE_ALL_ROWS
            )

    return {
        "base_variant": ROW_MODE_ALL_ROWS,
        "candidate_variant": ROW_MODE_DECISION_ONLY,
        "scope": scope_name,
        "delta_candidate_minus_base": delta,
        "winner_by_metric": winner_by_metric,
    }


def format_metric_value(scope_payload, metric_name):
    metrics = scope_payload.get("metrics") or {}
    if metric_name not in metrics:
        return "n/a"
    return f"{float(metrics[metric_name]):.6f}"


def evaluate_variant(
    *,
    variant_name,
    x,
    y,
    sample_weight,
    df,
    decision_mask,
    folds,
    param_overrides,
    float_dtype,
    n_estimators,
    early_stopping_rounds,
):
    y_np = y.to_numpy(dtype=np.int8, copy=False)
    sample_weight_np = sample_weight.to_numpy(dtype=np.float64, copy=False)
    decision_mask_np = np.asarray(decision_mask, dtype=bool)
    oof_pred = np.full(shape=len(df), fill_value=np.nan, dtype=np.float64)
    fold_records = []
    dropped_all_nan_train_features = set()
    best_iterations = []

    for fold in folds:
        train_indices = np.arange(fold["train_start"], fold["train_end"], dtype=np.int32)
        test_indices = np.arange(fold["test_start"], fold["test_end"], dtype=np.int32)
        selected = select_fold_row_indices(
            train_indices=train_indices,
            test_indices=test_indices,
            decision_mask=decision_mask_np,
            row_mode=variant_name,
        )
        train_used = selected["train_indices"]
        eval_used = selected["eval_indices"]
        predict_used = selected["predict_indices"]

        if train_used.size == 0:
            raise ValueError(
                f"Fold {fold['fold_id']} produced 0 training rows for row_mode={variant_name}."
            )
        if eval_used.size == 0:
            raise ValueError(
                f"Fold {fold['fold_id']} produced 0 validation rows for row_mode={variant_name}."
            )

        y_train = y_np[train_used]
        if np.unique(y_train).size < 2:
            raise ValueError(
                "Target has only one class inside a fold after row filtering. "
                f"fold_id={fold['fold_id']} row_mode={variant_name}"
            )

        x_train_raw = x.iloc[train_used]
        x_eval_raw = x.iloc[eval_used]
        x_predict_raw = x.iloc[predict_used]
        x_train, x_eval, _, fold_all_nan_train_features, _ = clean_and_impute_fold(
            x_train_raw,
            x_eval_raw,
            float_dtype=float_dtype,
        )
        x_predict = x_predict_raw.drop(
            columns=fold_all_nan_train_features,
            errors="ignore",
        ).astype(float_dtype, copy=False)

        dropped_all_nan_train_features.update(fold_all_nan_train_features)
        model = build_lgbm_model(
            n_estimators=n_estimators,
            param_overrides=param_overrides,
            feature_names=x_train.columns,
        )
        fit_kwargs = {
            "X": x_train,
            "y": y_train,
            "sample_weight": sample_weight_np[train_used],
            "eval_set": [(x_eval, y_np[eval_used])],
            "eval_sample_weight": [sample_weight_np[eval_used]],
            "eval_metric": EARLY_STOPPING_EVAL_METRIC,
        }
        if early_stopping_rounds > 0:
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(
                    stopping_rounds=int(early_stopping_rounds),
                    first_metric_only=True,
                    verbose=False,
                )
            ]
        model.fit(**fit_kwargs)

        best_iteration = int(model.best_iteration_ or n_estimators)
        best_iterations.append(best_iteration)
        fold_pred = model.predict_proba(
            x_predict,
            num_iteration=best_iteration,
        )[:, 1]
        oof_pred[predict_used] = fold_pred.astype(np.float64, copy=False)

        scoped_metrics = evaluate_prediction_scopes(
            y_true=y_np[predict_used],
            y_pred_proba=fold_pred,
            sample_weight=sample_weight_np[predict_used],
            decision_mask=decision_mask_np[predict_used],
        )
        fold_record = {
            "variant": variant_name,
            "fold_id": int(fold["fold_id"]),
            "train_rows_total_window": int(train_indices.size),
            "train_rows_used": int(train_used.size),
            "train_decision_rows_used": int(decision_mask_np[train_used].sum()),
            "eval_rows_used": int(eval_used.size),
            "predict_rows": int(predict_used.size),
            "predict_decision_rows": int(decision_mask_np[predict_used].sum()),
            "best_iteration": best_iteration,
            "train_opened_min": df.loc[train_used[0], "Opened"].isoformat(),
            "train_opened_max": df.loc[train_used[-1], "Opened"].isoformat(),
            "test_opened_min": df.loc[predict_used[0], "Opened"].isoformat(),
            "test_opened_max": df.loc[predict_used[-1], "Opened"].isoformat(),
        }
        for scope_name, scope_payload in scoped_metrics.items():
            fold_record[f"{scope_name}_rows"] = int(scope_payload["rows"])
            fold_record[f"{scope_name}_weight_sum"] = float(scope_payload["weight_sum"])
            fold_record[f"{scope_name}_positive_rate"] = scope_payload["positive_rate"]
            metrics = scope_payload["metrics"] or {}
            for metric_name, metric_value in metrics.items():
                fold_record[f"{scope_name}_{metric_name}"] = float(metric_value)
        fold_records.append(fold_record)

        print(
            f"variant={variant_name} fold={fold['fold_id']} "
            f"train_used={train_used.size}/{train_indices.size} "
            f"eval_used={eval_used.size} predict_rows={predict_used.size} "
            f"decision_test_rows={int(decision_mask_np[predict_used].sum())} "
            f"best_iteration={best_iteration} "
            f"decision_brier={format_metric_value(scoped_metrics['decision_rows'], 'brier_score')}"
        )

    fold_df = pd.DataFrame(fold_records)
    oof_mask = np.isfinite(oof_pred)
    if not np.any(oof_mask):
        raise RuntimeError(f"No OOF predictions produced for row_mode={variant_name}.")

    oof_scope_metrics = evaluate_prediction_scopes(
        y_true=y_np[oof_mask],
        y_pred_proba=oof_pred[oof_mask],
        sample_weight=sample_weight_np[oof_mask],
        decision_mask=decision_mask_np[oof_mask],
    )
    decision_oof_mask = oof_mask & decision_mask_np

    return {
        "variant_name": variant_name,
        "row_filter": {
            "mode": variant_name,
            "enabled": variant_name == ROW_MODE_DECISION_ONLY,
            "train_filter": (
                "Opened.minute % 5 == 4"
                if variant_name == ROW_MODE_DECISION_ONLY
                else None
            ),
            "eval_filter": (
                "Opened.minute % 5 == 4"
                if variant_name == ROW_MODE_DECISION_ONLY
                else None
            ),
            "prediction_scope": "all_test_rows",
        },
        "fold_count": int(len(fold_df)),
        "best_iteration": {
            "mean": float(np.mean(best_iterations)),
            "median": float(np.median(best_iterations)),
            "min": int(np.min(best_iterations)),
            "max": int(np.max(best_iterations)),
        },
        "rows_used": {
            "train_rows_used_mean": float(fold_df["train_rows_used"].mean()),
            "train_rows_used_min": int(fold_df["train_rows_used"].min()),
            "train_rows_used_max": int(fold_df["train_rows_used"].max()),
            "eval_rows_used_mean": float(fold_df["eval_rows_used"].mean()),
            "eval_rows_used_min": int(fold_df["eval_rows_used"].min()),
            "eval_rows_used_max": int(fold_df["eval_rows_used"].max()),
        },
        "oof_rows": int(oof_mask.sum()),
        "oof_coverage_ratio": float(oof_mask.mean()),
        "oof_decision_rows": int(decision_oof_mask.sum()),
        "oof_decision_coverage_ratio": (
            float(decision_oof_mask.sum() / decision_mask_np.sum())
            if int(decision_mask_np.sum()) > 0
            else 0.0
        ),
        "oof_metrics": oof_scope_metrics,
        "cv_scope_metrics": {
            scope_name: metric_summary_from_fold_df(fold_df, scope_name)
            for scope_name in ("all_rows", "decision_rows")
        },
        "fold_metrics": fold_df,
        "oof_pred": oof_pred,
        "dropped_all_nan_train_features": sorted(dropped_all_nan_train_features),
    }


def main():
    args = build_cli_args()
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")
    if args.n_estimators <= 0:
        raise ValueError("--n-estimators must be > 0.")
    if args.preview_rows <= 0:
        raise ValueError("--preview-rows must be > 0.")

    output_dir = Path(args.output_dir) / make_run_timestamp()
    output_dir.mkdir(parents=True, exist_ok=False)

    dataset_settings = load_modeling_dataset_settings()
    modeling_float_dtype = resolve_modeling_float_dtype(dataset_settings)
    modeling_float_dtype_name = resolve_modeling_float_dtype_name(dataset_settings)
    data_path = resolve_modeling_dataset_output_paths(dataset_settings)["parquet"]
    feature_subset = load_feature_subset_from_settings(dataset_settings)
    excluded_features = load_excluded_feature_names_from_settings(dataset_settings)

    training_data = load_walk_forward_training_frame(
        data_path=data_path,
        feature_subset=feature_subset,
        excluded_features=excluded_features,
        float_dtype=modeling_float_dtype,
    )
    df = training_data["df"].reset_index(drop=True)
    x = training_data["x"].reset_index(drop=True)
    y = training_data["y"].reset_index(drop=True)
    sample_weight = training_data["sample_weight"].reset_index(drop=True)
    sample_weight_source = training_data["sample_weight_source"]
    sample_weight_summary = training_data["sample_weight_summary"]
    class_distribution = training_data["class_distribution"]
    weighted_class_distribution = training_data["weighted_class_distribution"]

    decision_mask = compute_decision_mask_from_opened(df["Opened"])
    decision_summary = summarize_boolean_mask(decision_mask)
    folds = make_walk_forward_folds(
        n_rows=len(x),
        n_folds=int(args.cv_folds),
        test_to_train_ratio=float(args.test_to_train_ratio),
    )
    param_overrides = build_model_param_overrides(args)
    monotone_constraint_summary = summarize_lgbm_monotone_constraints(x.columns)

    print(
        f"dataset={data_path} rows={len(df)} features={x.shape[1]} "
        f"decision_rows={decision_summary['rows']} "
        f"decision_share={decision_summary['share']:.4f}"
    )
    print(
        f"sample_weight_source={sample_weight_source} "
        f"sample_weight_summary={sample_weight_summary}"
    )
    print(
        f"params_source={args.params_source} device_type={args.device_type} "
        f"cv_folds={args.cv_folds} test_to_train_ratio={args.test_to_train_ratio:.4f} "
        f"float_precision={modeling_float_dtype_name}"
    )
    print(
        "monotone_constraints="
        f"{format_lgbm_monotone_constraint_summary(monotone_constraint_summary)}"
    )

    variant_results = {}
    fold_metrics_frames = []
    for variant_name in SUPPORTED_ROW_MODES:
        variant_results[variant_name] = evaluate_variant(
            variant_name=variant_name,
            x=x,
            y=y,
            sample_weight=sample_weight,
            df=df,
            decision_mask=decision_mask,
            folds=folds,
            param_overrides=param_overrides,
            float_dtype=modeling_float_dtype,
            n_estimators=int(args.n_estimators),
            early_stopping_rounds=int(args.early_stopping_rounds),
        )
        fold_metrics_frames.append(variant_results[variant_name]["fold_metrics"])

    variant_summary_rows = pd.DataFrame(
        [
            build_variant_summary_row(variant_name, result)
            for variant_name, result in variant_results.items()
        ]
    )
    fold_metrics_df = pd.concat(fold_metrics_frames, ignore_index=True)

    comparison_payload = {
        scope_name: compare_scope_metrics(
            variant_results[ROW_MODE_ALL_ROWS],
            variant_results[ROW_MODE_DECISION_ONLY],
            scope_name=scope_name,
        )
        for scope_name in ("all_rows", "decision_rows")
    }
    primary_comparison = comparison_payload[PRIMARY_SCOPE]
    primary_delta = primary_comparison["delta_candidate_minus_base"].get(PRIMARY_METRIC)
    if primary_delta is None:
        recommended_variant = None
    elif abs(primary_delta) <= 1e-15:
        recommended_variant = "tie"
    elif PRIMARY_METRIC in LOWER_IS_BETTER_METRICS:
        recommended_variant = (
            ROW_MODE_DECISION_ONLY if primary_delta < 0.0 else ROW_MODE_ALL_ROWS
        )
    else:
        recommended_variant = (
            ROW_MODE_DECISION_ONLY if primary_delta > 0.0 else ROW_MODE_ALL_ROWS
        )

    oof_export = df.loc[:, [col for col in OOF_EXPORT_BASE_COLS if col in df.columns]].copy()
    oof_export["decision_row"] = np.asarray(decision_mask, dtype=np.int8)
    oof_export["oof_pred_proba_up_all_rows"] = variant_results[ROW_MODE_ALL_ROWS][
        "oof_pred"
    ]
    oof_export["oof_pred_proba_up_decision_rows_only"] = variant_results[
        ROW_MODE_DECISION_ONLY
    ]["oof_pred"]

    summary_payload = {
        "created_utc": utc_now().isoformat(),
        "output_dir": path_to_portable_str(output_dir),
        "data_path": path_to_portable_str(data_path),
        "target_col": TARGET_COL,
        "sample_weight_col": TARGET_WEIGHT_COL,
        "sample_weight": {
            "used": True,
            "source": sample_weight_source,
            **sample_weight_summary,
        },
        "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
        "weighted_class_distribution": weighted_class_distribution,
        "feature_selection": summarize_feature_subset(
            feature_subset,
            excluded_features=excluded_features,
        ),
        "monotone_constraints": monotone_constraint_summary,
        "decision_row_definition": {
            "opened_minute_modulo": 5,
            "decision_remainder": 4,
            "expression": "Opened.minute % 5 == 4",
            **decision_summary,
        },
        "walk_forward": {
            "fold_count": int(len(folds)),
            "test_to_train_ratio": float(args.test_to_train_ratio),
            "n_estimators": int(args.n_estimators),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "params_source": str(args.params_source),
            "device_type": str(args.device_type),
            "float_precision": modeling_float_dtype_name,
        },
        "variants": {
            variant_name: {
                key: value
                for key, value in variant_results[variant_name].items()
                if key not in {"fold_metrics", "oof_pred"}
            }
            for variant_name in SUPPORTED_ROW_MODES
        },
        "comparison": {
            "primary_scope": PRIMARY_SCOPE,
            "primary_metric": PRIMARY_METRIC,
            "recommended_variant": recommended_variant,
            "details": comparison_payload,
        },
        "artifacts": {
            "summary_json": path_to_portable_str(output_dir / "summary.json"),
            "variant_summary_csv": path_to_portable_str(
                output_dir / "variant_summary.csv"
            ),
            "fold_metrics_csv": path_to_portable_str(output_dir / "fold_metrics.csv"),
            "oof_predictions_parquet": path_to_portable_str(
                output_dir / "oof_predictions.parquet"
            ),
            "oof_predictions_head_csv": path_to_portable_str(
                output_dir / f"oof_predictions_head{int(args.preview_rows)}.csv"
            ),
            "oof_predictions_tail_csv": path_to_portable_str(
                output_dir / f"oof_predictions_tail{int(args.preview_rows)}.csv"
            ),
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    variant_summary_rows.to_csv(output_dir / "variant_summary.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    oof_export.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    oof_export.head(int(args.preview_rows)).to_csv(
        output_dir / f"oof_predictions_head{int(args.preview_rows)}.csv",
        index=False,
    )
    oof_export.tail(int(args.preview_rows)).to_csv(
        output_dir / f"oof_predictions_tail{int(args.preview_rows)}.csv",
        index=False,
    )

    for variant_name in SUPPORTED_ROW_MODES:
        decision_scope = variant_results[variant_name]["oof_metrics"][PRIMARY_SCOPE]
        all_rows_scope = variant_results[variant_name]["oof_metrics"]["all_rows"]
        print(
            f"summary variant={variant_name} "
            f"decision_bal_acc={format_metric_value(decision_scope, 'balanced_accuracy')} "
            f"decision_brier={format_metric_value(decision_scope, 'brier_score')} "
            f"decision_logloss={format_metric_value(decision_scope, 'binary_logloss')} "
            f"all_rows_brier={format_metric_value(all_rows_scope, 'brier_score')} "
            f"mean_best_iteration={variant_results[variant_name]['best_iteration']['mean']:.2f}"
        )
    print(
        f"recommendation scope={PRIMARY_SCOPE} metric={PRIMARY_METRIC} "
        f"winner={recommended_variant} output_dir={path_to_portable_str(output_dir)}"
    )


if __name__ == "__main__":
    main()
