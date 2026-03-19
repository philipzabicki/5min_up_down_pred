import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from features.candle_features import RAW_OHLCV_COLS
from metrics_utils import make_lightgbm_binary_brier_eval
from features.volume_profile_fixed_range import (
    build_volume_profile_feature_matrix_from_arrays,
    normalize_config as normalize_volume_profile_config,
)
from modeling_dataset_utils import load_modeling_dataset_settings
from target_weights import (
    TARGET_WEIGHT_COL,
    TARGET_WEIGHT_DECISION_VALUE,
    add_target_weights,
    summarize_target_weights,
)

TARGET_TIME_COL = "Opened"
TARGET_PRICE_COL = "Close"
TARGET_COL = "target_5m_candle_up"
TARGET_HORIZON_MINUTES = 5

BASE_DATA_PATH = Path("data/BTCUSDT1m.csv")
SEED = 37

CV_FOLDS = 10
WF_TEST_TO_TRAIN_RATIO = 0.2
MIN_SAMPLE_WEIGHT = float(TARGET_WEIGHT_DECISION_VALUE)

MAX_N_ESTIMATORS = 5000
EARLY_STOPPING_ROUNDS = 50
PRUNE_REPORT_EVERY_N_ITER = 10

LGBM_NUM_THREADS = 14
OPTUNA_OPTIMIZE_N_JOBS = 1
LGBM_DEVICE_TYPE = "gpu"
LGBM_VERBOSITY = -1
GPU_MAX_BIN_LIMIT = 63
LGBM_GPU_USE_DP = True

LGBM_DEFAULT_PARAMS = {
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "max_depth": -1,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "lambda_l2": 0.0,
    "lambda_l1": 0.0,
    "min_sum_hessian_in_leaf": 0.001,
    "min_gain_to_split": 0.0,
    "feature_fraction_bynode": 1.0,
    "path_smooth": 0.0,
    "extra_trees": False,
}

# Edit these ranges directly. Only keys listed here are optimized by Optuna.
# These ranges were widened after comparing the best regions from the broader
# v7/v8 searches with the later conservative pass. The goal is to reopen
# truncated edges without going back to the original fully loose space.
VOLUME_PROFILE_OPTUNA_SEARCH_SPACE = {
    "step": {"type": "int", "low": 1, "high": 136, "log": True},
    "neighbor_bins": {"type": "int", "low": 1, "high": 24},
    "local_window": {"type": "int", "low": 1, "high": 256, "log": True},
    "sigma_divisor": {"type": "float", "low": 0.25, "high": 50.0, "log": True},
    "min_sigma": {"type": "float", "low": 0.25, "high": 256.0, "log": True},
    "short_half_life_candles": {
        "type": "int",
        "low": 12,
        "high": 2_400,
        "log": True,
    },
    "medium_half_life_candles": {
        "type": "int",
        "low": 2_400,
        "high": 20_160,
        "log": True,
    },
    "long_half_life_candles": {
        "type": "int",
        "low": 14_400,
        "high": 86_400,
        "log": True,
    },
}

# Seed trials are injected before optimization starts.
OPTUNA_SEED_TRIAL_PARAMS = [
    {
      "step": 73,
      "neighbor_bins": 15,
      "local_window": 19,
      "sigma_divisor": 22.103834316321777,
      "min_sigma": 158.51656478094498,
      "short_half_life_candles": 60,
      "medium_half_life_candles": 4017,
      "long_half_life_candles": 22587
    },
    {
      "step": 54,
      "neighbor_bins": 15,
      "local_window": 128,
      "sigma_divisor": 13.808327590597779,
      "min_sigma": 37.674893963884564,
      "short_half_life_candles": 91,
      "medium_half_life_candles": 2942,
      "long_half_life_candles": 60205
    },
    {
    "step": 42,
    "neighbor_bins": 9,
    "local_window": 77,
    "sigma_divisor": 8.250626386620294,
    "min_sigma": 247.31082044719778,
    "short_half_life_candles": 67,
    "medium_half_life_candles": 5415,
    "long_half_life_candles": 19529
    }
]

N_TRIALS = 300
TIMEOUT_SECONDS = None
LOAD_IF_EXISTS = True
TPE_STARTUP_TRIALS = 10

CV_OBJECTIVE_NAME = "brier_score_mean_plus_std_penalty"
CV_BRIER_STD_PENALTY = 1.0
CRASH_PENALTY = 1.0
STUDY_NAME = "volume_profile_fixed_range_opt_brier_mean_std_v4"
STORAGE = "sqlite:///data/optuna/volume_profile_fixed_range_only_filtered_cv.db"
BEST_RESULT_PATH = Path(
    "data/models/volume_profile_fixed_range_only_filtered_cv_best_mean_std.json"
)
TRIALS_CSV_PATH = Path(
    "data/optuna/volume_profile_fixed_range_only_filtered_cv_trials_mean_std.csv"
)


def require_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def build_target_frame(df):
    out = df.copy()
    out[TARGET_TIME_COL] = pd.to_datetime(out[TARGET_TIME_COL], errors="raise")
    out = out.sort_values(TARGET_TIME_COL).reset_index(drop=True)
    future_close = out[TARGET_PRICE_COL].shift(-TARGET_HORIZON_MINUTES)
    target = (future_close > out[TARGET_PRICE_COL]).astype("float32")
    out[TARGET_COL] = target.where(future_close.notna())
    out = add_target_weights(out, opened_col=TARGET_TIME_COL, weight_col=TARGET_WEIGHT_COL)
    return out


def validate_sample_weight_array(sample_weight_np):
    if sample_weight_np.ndim != 1:
        raise ValueError("Sample weights must be a 1D array.")
    if sample_weight_np.size == 0:
        raise ValueError("Sample weights array is empty.")
    if not np.isfinite(sample_weight_np).all():
        raise ValueError("Sample weights contain non-finite values.")
    if np.any(sample_weight_np <= 0.0):
        raise ValueError("Sample weights must be strictly positive.")


def load_base_ohlcv_frame(data_path):
    if not data_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {data_path}")

    required_columns = [TARGET_TIME_COL, *RAW_OHLCV_COLS]
    print(f"load raw data | path={data_path}")
    df = pd.read_csv(data_path, usecols=required_columns)
    require_columns(df, required_columns)
    raw_rows = int(len(df))

    df = build_target_frame(df)
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    rows_after_target_notna = int(len(df))
    if rows_after_target_notna == 0:
        raise ValueError("No rows left after target construction.")

    sample_weight_full = pd.to_numeric(df[TARGET_WEIGHT_COL], errors="raise").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    validate_sample_weight_array(sample_weight_full)

    keep_mask = sample_weight_full >= float(MIN_SAMPLE_WEIGHT)
    filtered_rows = int(keep_mask.sum())
    if filtered_rows == 0:
        raise ValueError(
            f"No rows left after row filter {TARGET_WEIGHT_COL}>={MIN_SAMPLE_WEIGHT:.2f}."
        )

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        df = df.astype({col: np.float32 for col in numeric_columns}, copy=False)

    y_full = df[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
    y_filtered = y_full[keep_mask]
    sample_weight_filtered = sample_weight_full[keep_mask]
    class_distribution = {
        int(cls): int(count)
        for cls, count in zip(*np.unique(y_filtered.astype(np.int8), return_counts=True))
    }
    weighted_class_distribution = {
        str(int(class_id)): float(sample_weight_filtered[y_filtered == float(class_id)].sum())
        for class_id in sorted(class_distribution.keys())
    }
    row_filter_info = {
        "enabled": True,
        "weight_col": TARGET_WEIGHT_COL,
        "min_weight": float(MIN_SAMPLE_WEIGHT),
        "rows_before": int(rows_after_target_notna),
        "rows_after": int(filtered_rows),
        "rows_removed": int(rows_after_target_notna - filtered_rows),
    }
    high_np = df["High"].to_numpy(dtype=np.float64, copy=False)
    low_np = df["Low"].to_numpy(dtype=np.float64, copy=False)
    volume_np = df["Volume"].to_numpy(dtype=np.float64, copy=False)

    print(
        f"load raw data | raw_rows={raw_rows} rows_after_target_notna={rows_after_target_notna}"
    )
    print(
        f"load raw data | row_filter {TARGET_WEIGHT_COL}>={MIN_SAMPLE_WEIGHT:.2f} "
        f"removed={row_filter_info['rows_removed']} remaining={row_filter_info['rows_after']}"
    )
    print(
        f"load raw data | filtered_class_distribution={class_distribution} "
        f"filtered_weighted_class_distribution={weighted_class_distribution}"
    )

    return {
        "df": df,
        "keep_mask": keep_mask,
        "high_np": high_np,
        "low_np": low_np,
        "volume_np": volume_np,
        "y_filtered": y_filtered.astype(np.float32, copy=False),
        "raw_rows": raw_rows,
        "rows_after_target_notna": rows_after_target_notna,
        "sample_weight_full": sample_weight_full,
        "sample_weight_filtered": sample_weight_filtered,
        "sample_weight_summary": summarize_target_weights(sample_weight_filtered),
        "class_distribution": class_distribution,
        "weighted_class_distribution": weighted_class_distribution,
        "row_filter_info": row_filter_info,
    }


def make_walk_forward_folds(n_rows, n_folds, test_to_train_ratio):
    if n_rows < 100:
        raise ValueError(f"Dataset too small for walk-forward CV: {n_rows} rows.")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    if not (0.0 < test_to_train_ratio < 1.0):
        raise ValueError("test_to_train_ratio must be in (0, 1).")

    ratio_inv = 1.0 / test_to_train_ratio
    test_len = int(np.floor(n_rows / (n_folds + ratio_inv)))
    train_len = int(np.floor(test_len / test_to_train_ratio))

    if test_len <= 0 or train_len <= 0:
        raise ValueError(
            f"Cannot create valid folds for n_rows={n_rows}, "
            f"n_folds={n_folds}, ratio={test_to_train_ratio}."
        )

    folds = []
    for fold_id in range(n_folds):
        train_start = fold_id * test_len
        train_end = train_start + train_len
        test_start = train_end
        test_end = test_start + test_len
        if test_end > n_rows:
            break
        folds.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

    if len(folds) != n_folds:
        raise ValueError(
            f"Created {len(folds)} folds, expected {n_folds}. "
            "Increase dataset size or lower folds."
        )
    return folds


def build_fold_indices(folds):
    return [
        (
            np.arange(fold["train_start"], fold["train_end"], dtype=np.int32),
            np.arange(fold["test_start"], fold["test_end"], dtype=np.int32),
        )
        for fold in folds
    ]


class ObjectiveAlignedLightGBMPruningCallback:
    def __init__(
        self,
        trial,
        metric,
        std_penalty,
        valid_name="valid_0",
        report_interval=1,
    ):
        self._trial = trial
        self._valid_name = valid_name
        self._metric = metric
        self._std_penalty = std_penalty
        self._report_interval = report_interval

    def _find_evaluation_result(self, target_valid_names, env):
        evaluation_result_list = env.evaluation_result_list
        if evaluation_result_list is None:
            return None

        for evaluation_result in evaluation_result_list:
            valid_name, metric = evaluation_result[:2]
            if valid_name not in target_valid_names:
                continue
            if metric != self._metric and metric != f"valid {self._metric}":
                continue
            return evaluation_result

        return None

    def __call__(self, env):
        if (env.iteration + 1) % self._report_interval != 0:
            return

        evaluation_result_list = env.evaluation_result_list
        is_cv = (
            evaluation_result_list is not None
            and len(evaluation_result_list) > 0
            and len(evaluation_result_list[0]) == 5
        )
        target_valid_names = ("cv_agg", "valid") if is_cv else (self._valid_name,)
        evaluation_result = self._find_evaluation_result(target_valid_names, env)
        if evaluation_result is None:
            raise ValueError(
                'The entry associated with the validation names "{}" and the metric name "{}" '
                "is not found in the evaluation result list {}.".format(
                    ", ".join(target_valid_names),
                    self._metric,
                    str(env.evaluation_result_list),
                )
            )

        _, _, current_score, is_higher_better = evaluation_result[:4]
        if is_higher_better:
            if self._trial.study.direction != optuna.study.StudyDirection.MAXIMIZE:
                raise ValueError(
                    "The intermediate values are inconsistent with the objective values "
                    "in terms of study directions."
                )
        else:
            if self._trial.study.direction != optuna.study.StudyDirection.MINIMIZE:
                raise ValueError(
                    "The intermediate values are inconsistent with the objective values "
                    "in terms of study directions."
                )

        if is_cv:
            current_mean = float(evaluation_result[2])
            current_std = float(evaluation_result[4])
            current_objective = current_mean + (self._std_penalty * current_std)
        else:
            current_objective = float(current_score)

        self._trial.report(current_objective, step=env.iteration)

        if self._trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial was pruned at iteration {env.iteration}."
            )


def get_supported_volume_profile_param_names(normalized_base):
    param_names = {
        "step",
        "neighbor_bins",
        "local_window",
        "sigma_divisor",
        "min_sigma",
    }
    for horizon_name in normalized_base["horizon_names"]:
        if normalized_base["horizons"][horizon_name]["half_life_candles"] is not None:
            param_names.add(f"{horizon_name}_half_life_candles")
    return param_names


def validate_optuna_search_spec(name, spec):
    if not isinstance(spec, dict):
        raise ValueError(f"Search space spec for {name!r} must be a dict.")

    spec_type = str(spec.get("type", "")).strip().lower()
    if spec_type not in {"int", "float"}:
        raise ValueError(
            f"Search space spec for {name!r} must define type='int' or type='float'."
        )

    if "low" not in spec or "high" not in spec:
        raise ValueError(f"Search space spec for {name!r} must define low and high.")

    low = spec["low"]
    high = spec["high"]
    log = bool(spec.get("log", False))

    if spec_type == "int":
        low_i = int(low)
        high_i = int(high)
        step = int(spec.get("step", 1))
        if step <= 0:
            raise ValueError(f"Integer search space step must be > 0 for {name!r}.")
        if log and step != 1:
            raise ValueError(
                f"Integer log search space cannot use step != 1 for {name!r}."
            )
        if log and low_i < 1:
            raise ValueError(
                f"Integer log search space requires low >= 1 for {name!r}."
            )
        if high_i < low_i:
            raise ValueError(
                f"Integer search space requires high >= low for {name!r}."
            )
        return

    low_f = float(low)
    high_f = float(high)
    step = spec.get("step")
    if step is not None and float(step) <= 0.0:
        raise ValueError(f"Float search space step must be > 0 for {name!r}.")
    if log and step is not None:
        raise ValueError(f"Float log search space cannot use step for {name!r}.")
    if log and low_f <= 0.0:
        raise ValueError(f"Float log search space requires low > 0 for {name!r}.")
    if high_f < low_f:
        raise ValueError(f"Float search space requires high >= low for {name!r}.")


def validate_volume_profile_search_space(normalized_base, search_space):
    supported_param_names = get_supported_volume_profile_param_names(normalized_base)
    search_space_names = set(search_space)
    unknown_names = sorted(search_space_names - supported_param_names)
    if unknown_names:
        raise ValueError(
            "Unsupported volume profile Optuna search params: "
            f"{unknown_names}. Supported={sorted(supported_param_names)}"
        )

    for name, spec in search_space.items():
        validate_optuna_search_spec(name, spec)


def suggest_value_from_spec(trial, name, spec):
    spec_type = str(spec["type"]).strip().lower()
    log = bool(spec.get("log", False))
    if spec_type == "int":
        return int(
            trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=log,
            )
        )

    step = spec.get("step")
    return float(
        trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            step=float(step) if step is not None else None,
            log=log,
        )
    )


def validate_seed_trial_params(seed_params, search_space):
    unknown_names = sorted(set(seed_params) - set(search_space))
    if unknown_names:
        raise ValueError(
            f"Seed trial contains params not present in search space: {unknown_names}"
        )

    for name, value in seed_params.items():
        spec = search_space[name]
        spec_type = str(spec["type"]).strip().lower()
        if spec_type == "int":
            value_i = int(value)
            low_i = int(spec["low"])
            high_i = int(spec["high"])
            step_i = int(spec.get("step", 1))
            if value_i < low_i or value_i > high_i:
                raise ValueError(
                    f"Seed trial param {name!r}={value_i} is outside [{low_i}, {high_i}]."
                )
            if ((value_i - low_i) % step_i) != 0:
                raise ValueError(
                    f"Seed trial param {name!r}={value_i} does not match step={step_i}."
                )
            continue

        value_f = float(value)
        low_f = float(spec["low"])
        high_f = float(spec["high"])
        if value_f < low_f or value_f > high_f:
            raise ValueError(
                f"Seed trial param {name!r}={value_f} is outside [{low_f}, {high_f}]."
            )
        step_f = spec.get("step")
        if step_f is not None:
            scaled = (value_f - low_f) / float(step_f)
            if not np.isclose(scaled, round(scaled), atol=1e-9):
                raise ValueError(
                    f"Seed trial param {name!r}={value_f} does not match step={step_f}."
                )


def enqueue_seed_trials(study, seed_trial_params, normalized_base, search_space):
    validate_volume_profile_search_space(normalized_base, search_space)

    for seed_index, params in enumerate(seed_trial_params, start=1):
        validate_seed_trial_params(params, search_space)
        build_volume_profile_config_from_params(normalized_base, params)
        study.enqueue_trial(
            params=params,
            user_attrs={"seed_trial_index": int(seed_index)},
            skip_if_exists=True,
        )


def suggest_volume_profile_config(trial, base_config, search_space):
    normalized_base = normalize_volume_profile_config(base_config)
    validate_volume_profile_search_space(normalized_base, search_space)
    params = {
        name: suggest_value_from_spec(trial, name, spec)
        for name, spec in search_space.items()
    }
    return build_volume_profile_config_from_params(normalized_base, params)


def build_volume_profile_config_from_params(base_config, params):
    normalized_base = normalize_volume_profile_config(base_config)
    config = {
        "enabled": True,
        "price_min": float(normalized_base["price_min"]),
        "price_max": float(normalized_base["price_max"]),
        "step": int(params.get("step", normalized_base["step"])),
        "neighbor_bins": int(
            params.get("neighbor_bins", normalized_base["neighbor_bins"])
        ),
        "local_window": int(
            params.get("local_window", normalized_base["local_window"])
        ),
        "sigma_divisor": float(
            params.get("sigma_divisor", normalized_base["sigma_divisor"])
        ),
        "min_sigma": float(params.get("min_sigma", normalized_base["min_sigma"])),
        "eps": float(normalized_base["eps"]),
        "horizons": {},
    }

    for horizon_name in normalized_base["horizon_names"]:
        base_half_life = normalized_base["horizons"][horizon_name]["half_life_candles"]
        if base_half_life is None:
            half_life = None
        else:
            half_life = int(
                params.get(f"{horizon_name}_half_life_candles", base_half_life)
            )
        config["horizons"][horizon_name] = {"half_life_candles": half_life}

    return normalize_volume_profile_config(config)


def make_lgbm_cv_params():
    return {
        "objective": "binary",
        "metric": "None",
        "boosting_type": "gbdt",
        "device_type": LGBM_DEVICE_TYPE,
        "verbosity": LGBM_VERBOSITY,
        "num_threads": LGBM_NUM_THREADS,
        "max_bin": GPU_MAX_BIN_LIMIT,
        "gpu_use_dp": LGBM_GPU_USE_DP,
        "num_iterations": MAX_N_ESTIMATORS,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
        "feature_pre_filter": True,
        **LGBM_DEFAULT_PARAMS,
    }


def build_filtered_training_arrays(
    high_np,
    low_np,
    volume_np,
    keep_mask,
    y_filtered,
    sample_weight_filtered,
    normalized_vp_config,
):
    x_np, _ = build_volume_profile_feature_matrix_from_arrays(
        high=high_np,
        low=low_np,
        volume=volume_np,
        cfg=normalized_vp_config,
        keep_mask=keep_mask,
    )
    y_np = y_filtered
    sample_weight_np = sample_weight_filtered

    if x_np.shape[0] == 0:
        raise ValueError("No rows left in filtered training matrix.")
    if x_np.shape[1] == 0:
        raise ValueError("No features left after volume profile build.")
    validate_sample_weight_array(sample_weight_np)

    y_unique = np.unique(y_np)
    if len(y_unique) < 2:
        raise ValueError(
            "Target has only one class after weight filtering. "
            f"Found classes={y_unique.tolist()}"
        )

    return x_np, y_np, sample_weight_np


def run_lightgbm_cv(
    x_np,
    y_np,
    sample_weight_np,
    fold_indices,
    feature_names,
    trial=None,
    return_cvbooster=False,
):
    train_set = lgb.Dataset(
        data=x_np,
        label=y_np,
        weight=sample_weight_np,
        feature_name=list(feature_names),
        free_raw_data=True,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
    ]
    if trial is not None:
        callbacks.append(
            ObjectiveAlignedLightGBMPruningCallback(
                trial=trial,
                metric="brier_score",
                std_penalty=CV_BRIER_STD_PENALTY,
                report_interval=PRUNE_REPORT_EVERY_N_ITER,
            )
        )

    cv_results = lgb.cv(
        params=make_lgbm_cv_params(),
        train_set=train_set,
        folds=fold_indices,
        stratified=False,
        shuffle=False,
        feval=make_lightgbm_binary_brier_eval(),
        callbacks=callbacks,
        return_cvbooster=return_cvbooster,
        seed=SEED,
    )

    mean_series = np.asarray(cv_results["valid brier_score-mean"], dtype=np.float64)
    std_series = np.asarray(cv_results["valid brier_score-stdv"], dtype=np.float64)
    objective_series = mean_series + (CV_BRIER_STD_PENALTY * std_series)
    best_index = int(np.argmin(objective_series))

    result = {
        "best_iteration": int(best_index + 1),
        "cv_brier_score_mean": float(mean_series[best_index]),
        "cv_brier_score_std": float(std_series[best_index]),
        "objective_value": float(objective_series[best_index]),
    }
    if return_cvbooster:
        result["cvbooster"] = cv_results["cvbooster"]
    return result


def make_objective(base_data, fold_indices, base_vp_config, search_space):
    def objective(trial):
        normalized_vp_config = None
        x_np = None

        try:
            normalized_vp_config = suggest_volume_profile_config(
                trial=trial,
                base_config=base_vp_config,
                search_space=search_space,
            )
            x_np, y_np, sample_weight_np = build_filtered_training_arrays(
                high_np=base_data["high_np"],
                low_np=base_data["low_np"],
                volume_np=base_data["volume_np"],
                keep_mask=base_data["keep_mask"],
                y_filtered=base_data["y_filtered"],
                sample_weight_filtered=base_data["sample_weight_filtered"],
                normalized_vp_config=normalized_vp_config,
            )
            cv_result = run_lightgbm_cv(
                x_np=x_np,
                y_np=y_np,
                sample_weight_np=sample_weight_np,
                fold_indices=fold_indices,
                feature_names=normalized_vp_config["feature_columns"],
                trial=trial,
                return_cvbooster=False,
            )

            trial.set_user_attr("trial_status", "ok")
            trial.set_user_attr(
                "cv_brier_score_mean",
                cv_result["cv_brier_score_mean"],
            )
            trial.set_user_attr(
                "cv_brier_score_std",
                cv_result["cv_brier_score_std"],
            )
            trial.set_user_attr("best_iteration", cv_result["best_iteration"])
            trial.set_user_attr("feature_count", int(x_np.shape[1]))
            trial.set_user_attr(
                "config_signature",
                str(normalized_vp_config["config_signature"]),
            )
            return cv_result["objective_value"]
        except (lgb.basic.LightGBMError, OSError) as e:
            trial.set_user_attr("trial_status", "crash_penalty")
            trial.set_user_attr("crash_type", type(e).__name__)
            trial.set_user_attr("crash_message", str(e)[:1000])
            if normalized_vp_config is not None:
                trial.set_user_attr(
                    "config_signature",
                    str(normalized_vp_config["config_signature"]),
                )
            if x_np is not None:
                trial.set_user_attr("feature_count", int(x_np.shape[1]))
            return CRASH_PENALTY

    return objective


def run_optuna_optimization():
    optuna.logging.set_verbosity(optuna.logging.INFO)

    dataset_settings = load_modeling_dataset_settings()
    base_vp_config = dataset_settings.get("volume_profile_fixed_range") or {}
    normalized_base_vp_config = normalize_volume_profile_config(base_vp_config)

    base_data = load_base_ohlcv_frame(BASE_DATA_PATH)
    base_df = base_data["df"]
    filtered_rows = int(base_data["row_filter_info"]["rows_after"])
    folds = make_walk_forward_folds(
        n_rows=filtered_rows,
        n_folds=CV_FOLDS,
        test_to_train_ratio=WF_TEST_TO_TRAIN_RATIO,
    )
    fold_indices = build_fold_indices(folds)

    if STORAGE.startswith("sqlite:///"):
        Path(STORAGE.replace("sqlite:///", "", 1)).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=TPE_STARTUP_TRIALS,
        multivariate=True,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=100,
        max_resource=MAX_N_ESTIMATORS,
        reduction_factor=3,
        bootstrap_count=2,
    )

    print(
        f"start optimize | base_rows={len(base_df)} filtered_rows={filtered_rows} "
        f"folds={len(fold_indices)} trials={N_TRIALS} timeout={TIMEOUT_SECONDS} "
        f"objective={CV_OBJECTIVE_NAME} std_penalty={CV_BRIER_STD_PENALTY:.4f}"
    )
    print(
        "start optimize | "
        f"base_volume_profile_feature_count={len(normalized_base_vp_config['feature_columns'])} "
        f"row_filter_min_weight={MIN_SAMPLE_WEIGHT:.2f} "
        f"step={normalized_base_vp_config['step']} "
        f"neighbor_bins={normalized_base_vp_config['neighbor_bins']} "
        f"local_window={normalized_base_vp_config['local_window']}"
    )
    print(
        "start optimize | "
        f"search_params={sorted(VOLUME_PROFILE_OPTUNA_SEARCH_SPACE)} "
        f"seed_trials_configured={len(OPTUNA_SEED_TRIAL_PARAMS)}"
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=LOAD_IF_EXISTS,
    )
    enqueue_seed_trials(
        study=study,
        seed_trial_params=OPTUNA_SEED_TRIAL_PARAMS,
        normalized_base=normalized_base_vp_config,
        search_space=VOLUME_PROFILE_OPTUNA_SEARCH_SPACE,
    )

    study.optimize(
        make_objective(
            base_data=base_data,
            fold_indices=fold_indices,
            base_vp_config=base_vp_config,
            search_space=VOLUME_PROFILE_OPTUNA_SEARCH_SPACE,
        ),
        n_trials=N_TRIALS,
        timeout=TIMEOUT_SECONDS,
        n_jobs=OPTUNA_OPTIMIZE_N_JOBS,
        gc_after_trial=True,
        show_progress_bar=True,
        catch=(lgb.basic.LightGBMError, OSError),
    )

    best_trial = study.best_trial
    best_normalized_vp_config = build_volume_profile_config_from_params(
        base_config=base_vp_config,
        params=best_trial.params,
    )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_data_path": str(BASE_DATA_PATH),
        "target_col": TARGET_COL,
        "sample_weight_col": TARGET_WEIGHT_COL,
        "sample_weight": {
            "used": True,
            "source": "dataset_column",
            **base_data["sample_weight_summary"],
        },
        "class_distribution": base_data["class_distribution"],
        "weighted_class_distribution": base_data["weighted_class_distribution"],
        "rows_raw": int(base_data["raw_rows"]),
        "rows_after_target_notna": int(base_data["rows_after_target_notna"]),
        "decision_row_filter": base_data["row_filter_info"],
        "feature_set": {
            "mode": "volume_profile_fixed_range_only",
            "base_feature_count": int(len(normalized_base_vp_config["feature_columns"])),
            "best_feature_count": int(len(best_normalized_vp_config["feature_columns"])),
        },
        "study_name": STUDY_NAME,
        "storage": STORAGE,
        "n_trials_requested": int(N_TRIALS),
        "timeout_seconds": TIMEOUT_SECONDS,
        "cv_folds": CV_FOLDS,
        "walk_forward_test_to_train_ratio": WF_TEST_TO_TRAIN_RATIO,
        "max_n_estimators": MAX_N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "prune_report_every_n_iteration": PRUNE_REPORT_EVERY_N_ITER,
        "cv_objective": {
            "name": CV_OBJECTIVE_NAME,
            "base_metric": "brier_score",
            "aggregation": "cv_mean + std_penalty * cv_std",
            "std_penalty": float(CV_BRIER_STD_PENALTY),
        },
        "lgbm_params": make_lgbm_cv_params(),
        "base_volume_profile_fixed_range": normalized_base_vp_config,
        "volume_profile_optuna_search_space": VOLUME_PROFILE_OPTUNA_SEARCH_SPACE,
        "optuna_seed_trial_params": OPTUNA_SEED_TRIAL_PARAMS,
        "best_trial": {
            "number": int(best_trial.number),
            "objective_cv_brier_score_mean_plus_std_penalty": float(best_trial.value),
            "cv_brier_score_mean": float(
                best_trial.user_attrs.get("cv_brier_score_mean")
            ),
            "cv_brier_score_std": float(
                best_trial.user_attrs.get("cv_brier_score_std")
            ),
            "best_iteration": int(best_trial.user_attrs.get("best_iteration")),
            "feature_count": int(best_trial.user_attrs.get("feature_count")),
            "params": best_trial.params,
            "volume_profile_fixed_range": best_normalized_vp_config,
        },
    }

    BEST_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    TRIALS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(TRIALS_CSV_PATH, index=False)

    print(
        f"best trial | number={best_trial.number} objective={best_trial.value:.8f} "
        f"brier_mean={float(best_trial.user_attrs.get('cv_brier_score_mean')):.8f} "
        f"brier_std={float(best_trial.user_attrs.get('cv_brier_score_std')):.8f} "
        f"best_iteration={int(best_trial.user_attrs.get('best_iteration'))} "
        f"feature_count={int(best_trial.user_attrs.get('feature_count'))}"
    )
    print(f"saved best payload -> {BEST_RESULT_PATH}")
    print(f"saved trials csv -> {TRIALS_CSV_PATH}")


def main():
    run_optuna_optimization()


if __name__ == "__main__":
    main()
