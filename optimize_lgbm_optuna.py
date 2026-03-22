import json
import re
from datetime import datetime, timezone
from pathlib import Path
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from features.candle_features import RAW_OHLCV_COLS
from modeling_dataset_utils import (
    load_excluded_feature_names_from_settings,
    load_feature_subset_from_settings,
    load_modeling_dataset_settings,
    resolve_modeling_dataset_output_paths,
    summarize_feature_subset,
)
from target_weights import (
    TARGET_WEIGHT_COL,
    summarize_target_weights,
)
from train_lgbm import (
    CV_FOLDS as FINAL_CV_FOLDS,
    WF_TEST_TO_TRAIN_RATIO as FINAL_WF_TEST_TO_TRAIN_RATIO,
    evaluate_walk_forward_variant,
    load_walk_forward_training_frame,
    make_walk_forward_folds as make_final_walk_forward_folds,
)

TARGET_COL = "target_5m_candle_up"

CV_FOLDS = 10
WF_TEST_TO_TRAIN_RATIO = 0.2

MAX_N_ESTIMATORS = 5000
EARLY_STOPPING_ROUNDS = 50
PRUNE_REPORT_EVERY_N_ITER = 10

SEED = 37
LGBM_NUM_THREADS = 14
OPTUNA_OPTIMIZE_N_JOBS = 1
LGBM_DEVICE_TYPE = "gpu"
LGBM_VERBOSITY = -1
GPU_MAX_BIN_LIMIT = 63

LGBM_OPTUNA_SEARCH_SPACE = {
    "learning_rate": {"type": "float", "low": 0.001, "high": 0.5, "log": True},
    "num_leaves": {"type": "int", "low": 16, "high": 256},
    "min_data_in_leaf": {"type": "int", "low": 2, "high": 4096, "log": True},
    "max_depth": {"type": "int", "low": 2, "high": 128},
    "feature_fraction": {"type": "float", "low": 0.1, "high": 1.0},
    "bagging_fraction": {"type": "float", "low": 0.1, "high": 1.0},
    "bagging_freq": {"type": "int", "low": 0, "high": 25},
    "lambda_l2": {"type": "float", "low": 0.0, "high": 100.0},
    "lambda_l1": {"type": "float", "low": 0.0, "high": 100.0},
    "min_sum_hessian_in_leaf": {
        "type": "float",
        "low": 1e-4,
        "high": 100.0,
        "log": True,
    },
    "min_gain_to_split": {"type": "float", "low": 0.0, "high": 10.0},
    "feature_fraction_bynode": {"type": "float", "low": 0.1, "high": 1.0},
    "path_smooth": {"type": "float", "low": 0.0, "high": 100.0},
    "extra_trees": {"type": "categorical", "choices": [True, False]},
}

# Seed trials are injected before optimization starts.
OPTUNA_SEED_TRIAL_PARAMS = [
    {'learning_rate': 0.005345243845517257,
     'num_leaves': 249,
     'min_data_in_leaf': 1092,
     'max_depth': 47,
     'feature_fraction': 0.547406725512275,
     'bagging_fraction': 0.8720848738316145,
     'bagging_freq': 19,
     'lambda_l2': 11.793596403657679,
     'lambda_l1': 16.802466762568642,
     'min_sum_hessian_in_leaf': 28.814446384235985,
     'min_gain_to_split': 0.3523220198504663,
     'feature_fraction_bynode': 0.5741253197885922,
     'path_smooth': 29.535516337206555,
     'extra_trees': False},
     {
      "learning_rate": 0.0021574582075051204,
      "num_leaves": 196,
      "min_data_in_leaf": 21,
      "max_depth": 58,
      "feature_fraction": 0.8187932858940383,
      "bagging_fraction": 0.8714798148611362,
      "bagging_freq": 4,
      "lambda_l2": 15.442114704888553,
      "lambda_l1": 2.212521909423135,
      "min_sum_hessian_in_leaf": 0.13475540327517857,
      "min_gain_to_split": 1.5679522702774737,
      "feature_fraction_bynode": 0.6443195011832288,
      "path_smooth": 18.354650904317488,
      "extra_trees": False
    },
    {
        'learning_rate': 0.0032826894654068746,
        'num_leaves': 240,
        'min_data_in_leaf': 51,
        'max_depth': 76,
        'feature_fraction': 0.2600155012187151,
        'bagging_fraction': 0.9386274411409469,
        'bagging_freq': 4,
        'lambda_l2': 8.97624362058438,
        'lambda_l1': 5.935593621429428,
        'min_sum_hessian_in_leaf': 0.22629666928997544,
        'min_gain_to_split': 1.247365118812283,
        'feature_fraction_bynode': 0.98701845583249, 
        'path_smooth': 18.823257423983854,
        'extra_trees': False},
    {
        "learning_rate": 0.00149752979585742,
        "num_leaves": 128,
        "min_data_in_leaf": 31,
        "max_depth": 10,
        "feature_fraction": 0.3668497153192713,
        "bagging_fraction": 0.6842364646883217,
        "bagging_freq": 18,
        "lambda_l2": 16.915746933977747,
        "lambda_l1": 2.0712451188077488,
        "min_sum_hessian_in_leaf": 0.25380833829794036,
        "min_gain_to_split": 0.8504082880772725,
        "feature_fraction_bynode": 0.32836945948911733,
        "path_smooth": 68.45674322734372,
        "extra_trees": False
      },
      {
      "learning_rate": 0.0047168930397256115,
      "num_leaves": 203,
      "min_data_in_leaf": 6,
      "max_depth": 41,
      "feature_fraction": 0.3535849279738236,
      "bagging_fraction": 0.8060837855401768,
      "bagging_freq": 10,
      "lambda_l2": 6.467328293921802,
      "lambda_l1": 4.930249967778666,
      "min_sum_hessian_in_leaf": 0.014887735545909926,
      "min_gain_to_split": 1.031237481486823,
      "feature_fraction_bynode": 0.8821060783972539,
      "path_smooth": 3.743686767424939,
      "extra_trees": False
    }
]

N_TRIALS = 10
TIMEOUT_SECONDS = None
CV_OBJECTIVE_NAME = "binary_logloss_mean_plus_std_penalty"
CV_LOGLOSS_STD_PENALTY = 0.5
RECHECK_OBJECTIVE_BASE_METRIC = "brier_score"
RECHECK_STD_PENALTY = 0.5
RECHECK_OBJECTIVE_NAME = f"{RECHECK_OBJECTIVE_BASE_METRIC}_mean_plus_std_penalty"
STUDY_NAME = "de_besta_v8"
STORAGE = "sqlite:///data/optuna/databases/lgbm_generic_tpe_hyperband_gpu.db"
LOAD_IF_EXISTS = True
BEST_RESULT_PATH = Path(
    "data/optuna/lgbm/lgbm_generic_optuna_best_mean_std.json"
)
TRIALS_CSV_PATH = Path("data/optuna/lgbm/lgbm_generic_optuna_trials_mean_std.csv")
RUN_MODE = "optimize"  # "optimize" or "recheck-topn"
RECHECK_STUDY_NAME = STUDY_NAME
RECHECK_STORAGE = STORAGE
TOP_TRIALS_RECHECK_N = 10
TOP_TRIALS_RECHECK_OUTPUT_DIR = Path("data/optuna/lgbm/recheck")
TOP_TRIALS_RECHECK_OUTPUT_JSON_PATH = None
TOP_TRIALS_RECHECK_OUTPUT_CSV_PATH = None

PRUNER_MIN_RESOURCE = 100
PRUNER_REDUCTION_FACTOR = 3
PRUNER_BOOTSTRAP_COUNT = 0
TPE_STARTUP_TRIALS = int(N_TRIALS*0.1)

FEATURE_HORIZON_RE = re.compile(r"(?:_fit_|_target_)(\d+)m(?:_ahead_ret)?")
TARGET_HORIZON_RE = re.compile(r"target_(\d+)m")


def make_walk_forward_folds(
    n_rows,
    n_folds,
    test_to_train_ratio,
):
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


def load_generic_training_data(data_path, feature_subset=None, excluded_features=None):
    excluded_feature_names = (
        tuple(excluded_features["features"]) if excluded_features else tuple()
    )
    excluded_feature_set = set(excluded_feature_names)
    selected_feature_columns = (
        list(feature_subset["features"]) if feature_subset else None
    )
    parquet_columns = None
    if selected_feature_columns is not None:
        parquet_columns = list(
            dict.fromkeys([TARGET_COL, TARGET_WEIGHT_COL, *selected_feature_columns])
        )

    print(f"load data | path={data_path}")
    try:
        df = pd.read_parquet(data_path, columns=parquet_columns)
    except Exception as exc:
        if parquet_columns is None:
            raise
        preview = ", ".join(parquet_columns[:10])
        raise ValueError(
            "Dataset is missing columns required by optimize_lgbm_optuna.py. "
            "Rebuild it with create_modeling_dataset.py for the active feature subset. "
            f"Requested_count={len(parquet_columns)} preview=[{preview}]"
        ) from exc
    raw_rows, raw_cols = df.shape
    print(f"load data | raw_rows={raw_rows} raw_cols={raw_cols}")

    df = df[df[TARGET_COL].notna()]
    rows_after_target_notna = int(len(df))
    if rows_after_target_notna == 0:
        raise ValueError("No rows left after TARGET_COL non-null filtering.")
    if TARGET_WEIGHT_COL not in df.columns:
        raise ValueError(
            f"Dataset is missing required sample weight column '{TARGET_WEIGHT_COL}'. "
            "Rebuild it with create_modeling_dataset.py."
        )
    sample_weight_full = pd.to_numeric(df[TARGET_WEIGHT_COL], errors="raise").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    sample_weight_source = "dataset_column"
    print(f"load data | rows_after_target_notna={rows_after_target_notna}")

    dropped_raw_ohlcv_features = [col for col in RAW_OHLCV_COLS if col in df.columns]
    x = df.drop(
        columns=[TARGET_COL, TARGET_WEIGHT_COL, *dropped_raw_ohlcv_features],
        errors="ignore",
    )
    x = x.select_dtypes(include=[np.number])
    if selected_feature_columns is not None:
        missing_selected_features = [
            col for col in selected_feature_columns if col not in x.columns
        ]
        if missing_selected_features:
            preview = ", ".join(missing_selected_features[:10])
            raise ValueError(
                "Dataset is missing configured subset features for optimization. "
                f"Missing_count={len(missing_selected_features)} preview=[{preview}]"
            )
        x = x.loc[:, selected_feature_columns]
    if excluded_feature_set:
        excluded_present_features = [
            col for col in x.columns if col in excluded_feature_set
        ]
        excluded_missing_features = [
            col for col in excluded_feature_names if col not in x.columns
        ]
        if excluded_present_features:
            x = x.drop(columns=excluded_present_features)
        print(
            "load data | exclusions "
            f"dropped={len(excluded_present_features)} "
            f"missing_requested={len(excluded_missing_features)}"
        )
    if x.shape[1] == 0:
        raise ValueError("No numeric feature columns left after preprocessing.")
    print(
        f"load data | numeric_features={x.shape[1]} "
        f"dropped_raw_ohlcv={len(dropped_raw_ohlcv_features)}"
    )

    feature_cols = x.columns.tolist()
    feature_horizons = [
        int(m.group(1))
        for col in feature_cols
        for m in [FEATURE_HORIZON_RE.search(col)]
        if m is not None
    ]
    max_feature_horizon = max(feature_horizons) if feature_horizons else 0

    target_horizon_match = TARGET_HORIZON_RE.search(TARGET_COL)
    target_horizon = int(target_horizon_match.group(1)) if target_horizon_match else 0

    x_np_full = x.to_numpy(dtype=np.float32, copy=False)
    invalid_full = ~np.isfinite(x_np_full)
    n_rows, n_features = x_np_full.shape
    n_invalid_full = int(invalid_full.sum())

    max_leading_invalid = 0
    max_trailing_invalid = 0
    for j in range(n_features):
        col_invalid = invalid_full[:, j]
        if col_invalid[0]:
            if col_invalid.all():
                leading = n_rows
            else:
                leading = int(np.argmax(~col_invalid))
            if leading > max_leading_invalid:
                max_leading_invalid = leading

        if col_invalid[-1]:
            rev_invalid = col_invalid[::-1]
            if rev_invalid.all():
                trailing = n_rows
            else:
                trailing = int(np.argmax(~rev_invalid))
            if trailing > max_trailing_invalid:
                max_trailing_invalid = trailing

    head_trim = max_feature_horizon
    tail_trim = max(max_feature_horizon, target_horizon)
    end_idx = n_rows - tail_trim if tail_trim > 0 else n_rows
    if end_idx <= head_trim:
        raise ValueError(
            "No rows left after horizon trim. "
            f"rows={n_rows} head_trim={head_trim} tail_trim={tail_trim}"
        )

    x_np = np.asarray(x_np_full[head_trim:end_idx], dtype=np.float32)
    finite_by_col = np.isfinite(x_np).any(axis=0)
    dropped_all_invalid_feature_names = []
    if not finite_by_col.all():
        dropped_all_invalid_feature_names = [
            feature_cols[j] for j in range(n_features) if not finite_by_col[j]
        ]
        x_np = x_np[:, finite_by_col]
    if x_np.shape[1] == 0:
        raise ValueError("No usable features left after dropping fully invalid columns.")

    invalid = ~np.isfinite(x_np)
    invalid_after_trim = int(invalid.sum())
    x_np = np.where(np.isinf(x_np), np.nan, x_np).astype(np.float32, copy=False)
    nan_after_trim = int(np.isnan(x_np).sum())

    y_np_full = df[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
    y_np = np.asarray(y_np_full[head_trim:end_idx], dtype=np.float32)
    sample_weight_np = np.asarray(
        sample_weight_full[head_trim:end_idx], dtype=np.float32
    )

    print(
        f"data trim | head={head_trim} tail={tail_trim} "
        f"lead_invalid={max_leading_invalid} trail_invalid={max_trailing_invalid} "
        f"max_feature_horizon={max_feature_horizon} target_horizon={target_horizon}"
    )
    if dropped_all_invalid_feature_names:
        preview = ", ".join(dropped_all_invalid_feature_names[:5])
        print(
            f"load data | dropped_all_invalid_features={len(dropped_all_invalid_feature_names)} "
            f"preview=[{preview}]"
        )
    print(
        f"load data | invalid_full={n_invalid_full} invalid_after_trim={invalid_after_trim} "
        f"nan_after_trim={nan_after_trim}"
    )
    print(
        f"load data | final_rows={x_np.shape[0]} features={x_np.shape[1]} "
        f"dtypes(x/y)=({x_np.dtype}/{y_np.dtype})"
    )
    if x_np.shape[0] == 0:
        raise ValueError("No rows left in training matrix after preprocessing.")
    if sample_weight_np.shape[0] != x_np.shape[0]:
        raise ValueError(
            "Sample weights length mismatch after preprocessing: "
            f"{sample_weight_np.shape[0]} != {x_np.shape[0]}"
        )
    if not np.isfinite(sample_weight_np).all():
        raise ValueError("Sample weights contain non-finite values.")
    if np.any(sample_weight_np <= 0.0):
        raise ValueError("Sample weights must be strictly positive.")

    y_unique = np.unique(y_np)
    if len(y_unique) < 2:
        raise ValueError(
            "Target has only one class after preprocessing. "
            f"Found classes={y_unique.tolist()}"
        )

    return (
        x_np,
        y_np,
        sample_weight_np,
        rows_after_target_notna,
        sample_weight_source,
        summarize_target_weights(sample_weight_np),
    )


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
        if is_cv:
            # LightGBM 4.6.0 reports CV metrics under "valid".
            # Accept "cv_agg" as well for compatibility with older assumptions.
            target_valid_names = ("cv_agg", "valid")
        else:
            target_valid_names = (self._valid_name,)

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
                    "in terms of study directions. Please specify a metric to be minimized "
                    "for ObjectiveAlignedLightGBMPruningCallback."
                )
        else:
            if self._trial.study.direction != optuna.study.StudyDirection.MINIMIZE:
                raise ValueError(
                    "The intermediate values are inconsistent with the objective values "
                    "in terms of study directions. Please specify a metric to be "
                    "maximized for ObjectiveAlignedLightGBMPruningCallback."
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


def validate_optuna_search_spec(name, spec):
    if not isinstance(spec, dict):
        raise ValueError(f"Search space spec for {name!r} must be a dict.")

    spec_type = str(spec.get("type", "")).strip().lower()
    if spec_type not in {"int", "float", "categorical"}:
        raise ValueError(
            f"Search space spec for {name!r} must define type='int', 'float', or 'categorical'."
        )

    if spec_type == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, (list, tuple)) or len(choices) == 0:
            raise ValueError(
                f"Categorical search space spec for {name!r} must define non-empty choices."
            )
        return

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


def validate_lgbm_search_space(search_space):
    for name, spec in search_space.items():
        validate_optuna_search_spec(name, spec)


def suggest_value_from_spec(trial, name, spec):
    spec_type = str(spec["type"]).strip().lower()
    if spec_type == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))

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


def suggest_lgbm_hyperparams(trial, search_space):
    validate_lgbm_search_space(search_space)
    return {
        name: suggest_value_from_spec(trial, name, spec)
        for name, spec in search_space.items()
    }


def validate_seed_trial_params(seed_params, search_space):
    unknown_names = sorted(set(seed_params) - set(search_space))
    if unknown_names:
        raise ValueError(
            f"Seed trial contains params not present in search space: {unknown_names}"
        )

    for name, value in seed_params.items():
        spec = search_space[name]
        spec_type = str(spec["type"]).strip().lower()
        if spec_type == "categorical":
            choices = list(spec["choices"])
            if not any(value == choice for choice in choices):
                raise ValueError(
                    f"Seed trial param {name!r}={value!r} is not in choices={choices!r}."
                )
            continue

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


def enqueue_seed_trials(study, seed_trial_params, search_space):
    validate_lgbm_search_space(search_space)

    for seed_index, params in enumerate(seed_trial_params, start=1):
        validate_seed_trial_params(params, search_space)
        study.enqueue_trial(
            params=params,
            user_attrs={"seed_trial_index": int(seed_index)},
            skip_if_exists=True,
        )


def make_objective(
    train_set,
    fold_indices,
    search_space,
):
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "device_type": LGBM_DEVICE_TYPE,
            "verbosity": LGBM_VERBOSITY,
            "num_threads": LGBM_NUM_THREADS,
            "max_bin": GPU_MAX_BIN_LIMIT,
            "num_iterations": MAX_N_ESTIMATORS,
            "seed": SEED,
            "feature_fraction_seed": SEED,
            "bagging_seed": SEED,
            "data_random_seed": SEED,
            "feature_pre_filter": False,
            "gpu_use_dp": False,
            **suggest_lgbm_hyperparams(trial, search_space),
        }

        cv_results = lgb.cv(
            params=params,
            train_set=train_set,
            folds=fold_indices,
            stratified=False,
            shuffle=False,
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
                ObjectiveAlignedLightGBMPruningCallback(
                    trial=trial,
                    metric="binary_logloss",
                    std_penalty=CV_LOGLOSS_STD_PENALTY,
                    report_interval=PRUNE_REPORT_EVERY_N_ITER,
                ),
            ],
            return_cvbooster=False,
            seed=SEED,
        )

        mean_series = np.asarray(
            cv_results["valid binary_logloss-mean"], dtype=np.float64
        )
        std_series = np.asarray(
            cv_results["valid binary_logloss-stdv"], dtype=np.float64
        )
        objective_series = mean_series + (CV_LOGLOSS_STD_PENALTY * std_series)
        best_index = int(np.argmin(objective_series))
        best_iteration = best_index + 1
        cv_logloss_mean = float(mean_series[best_index])
        cv_logloss_std = float(std_series[best_index])
        objective_value = float(objective_series[best_index])
        trial.set_user_attr("best_iteration", best_iteration)
        trial.set_user_attr("cv_binary_logloss_mean", cv_logloss_mean)
        trial.set_user_attr("cv_binary_logloss_std", cv_logloss_std)
        return objective_value

    return objective


def make_top_trial_recheck_output_paths(
    study_name,
    top_n,
    output_json=None,
    output_csv=None,
):
    if output_json is not None and output_csv is not None:
        return output_json, output_csv

    safe_study_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", study_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"{safe_study_name}_top{top_n}_final_cv_{timestamp}"
    json_path = output_json or (TOP_TRIALS_RECHECK_OUTPUT_DIR / f"{stem}.json")
    csv_path = output_csv or (TOP_TRIALS_RECHECK_OUTPUT_DIR / f"{stem}.csv")
    return json_path, csv_path


def build_top_trial_recheck_summary_row(result):
    row = {
        "recheck_rank": int(result["recheck_rank"]),
        "optuna_rank": int(result["optuna_rank"]),
        "trial_number": int(result["trial_number"]),
        "optuna_objective": float(result["optuna_objective"]),
        "optuna_best_iteration": result["optuna_best_iteration"],
        "optuna_cv_binary_logloss_mean": result["optuna_cv_binary_logloss_mean"],
        "optuna_cv_binary_logloss_std": result["optuna_cv_binary_logloss_std"],
        "recheck_objective": float(result["recheck_objective"]),
        "recheck_mean_best_iteration": int(result["recheck_mean_best_iteration"]),
    }
    for metric_name, metric_value in result["recheck_cv_mean_metrics"].items():
        row[f"recheck_cv_mean_{metric_name}"] = float(metric_value)
    for metric_name, metric_value in result["recheck_cv_std_metrics"].items():
        row[f"recheck_cv_std_{metric_name}"] = float(metric_value)
    return row


def build_top_trial_recheck_best_trial(result):
    best_metric_mean = result["recheck_cv_mean_metrics"].get(
        RECHECK_OBJECTIVE_BASE_METRIC
    )
    best_metric_std = result["recheck_cv_std_metrics"].get(
        RECHECK_OBJECTIVE_BASE_METRIC
    )
    binary_logloss_mean = result["recheck_cv_mean_metrics"].get("binary_logloss")
    binary_logloss_std = result["recheck_cv_std_metrics"].get("binary_logloss")
    return {
        "recheck_rank": int(result["recheck_rank"]),
        "optuna_rank": int(result["optuna_rank"]),
        "number": int(result["trial_number"]),
        f"objective_{RECHECK_OBJECTIVE_NAME}": float(result["recheck_objective"]),
        f"cv_{RECHECK_OBJECTIVE_BASE_METRIC}_mean": (
            float(best_metric_mean) if best_metric_mean is not None else None
        ),
        f"cv_{RECHECK_OBJECTIVE_BASE_METRIC}_std": (
            float(best_metric_std) if best_metric_std is not None else None
        ),
        "cv_binary_logloss_mean": (
            float(binary_logloss_mean) if binary_logloss_mean is not None else None
        ),
        "cv_binary_logloss_std": (
            float(binary_logloss_std) if binary_logloss_std is not None else None
        ),
        "mean_best_iteration": int(result["recheck_mean_best_iteration"]),
        "params": result["params"],
    }


def run_top_trials_recheck(
    study_name,
    storage,
    top_n,
    output_json=None,
    output_csv=None,
):
    if top_n < 1:
        raise ValueError("top_n must be >= 1.")

    dataset_settings = load_modeling_dataset_settings()
    data_path = resolve_modeling_dataset_output_paths(dataset_settings)["parquet"]
    feature_subset = load_feature_subset_from_settings(dataset_settings)
    excluded_features = load_excluded_feature_names_from_settings(dataset_settings)
    training_data = load_walk_forward_training_frame(
        data_path=data_path,
        feature_subset=feature_subset,
        excluded_features=excluded_features,
    )
    x = training_data["x"]
    y = training_data["y"]
    sample_weight = training_data["sample_weight"]
    folds = make_final_walk_forward_folds(
        n_rows=len(x),
        n_folds=FINAL_CV_FOLDS,
        test_to_train_ratio=FINAL_WF_TEST_TO_TRAIN_RATIO,
    )

    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )
    completed_trials = [
        trial
        for trial in study.get_trials(
            deepcopy=False,
            states=(optuna.trial.TrialState.COMPLETE,),
        )
        if trial.value is not None
    ]
    completed_trials.sort(key=lambda trial: float(trial.value))
    selected_trials = completed_trials[:top_n]
    if not selected_trials:
        raise ValueError(
            f"No completed trials found for study_name={study_name!r} in storage={storage!r}."
        )

    print(
        f"start recheck | study_name={study_name} storage={storage} "
        f"completed_trials={len(completed_trials)} selected_trials={len(selected_trials)} "
        f"rows={len(x)} features={x.shape[1]} folds={len(folds)} "
        f"test/train={FINAL_WF_TEST_TO_TRAIN_RATIO:.3f}"
    )
    if feature_subset:
        print(
            "start recheck | "
            f"feature_subset_path={feature_subset['path']} count={feature_subset['count']}"
        )

    recheck_results = []
    for optuna_rank, trial in enumerate(selected_trials, start=1):
        print(
            f"recheck trial {optuna_rank}/{len(selected_trials)} | "
            f"trial_number={trial.number} optuna_objective={float(trial.value):.8f}"
        )
        cv_result, _, _ = evaluate_walk_forward_variant(
            x=x,
            y=y,
            sample_weight=sample_weight,
            folds=folds,
            param_overrides=trial.params,
            model_variant=f"trial_{trial.number}",
            collect_oof_predictions=False,
            collect_feature_importance=False,
            early_stopping_verbose=False,
        )
        recheck_metric_mean = float(
            cv_result["cv_mean_metrics"][RECHECK_OBJECTIVE_BASE_METRIC]
        )
        recheck_metric_std = float(
            cv_result["cv_std_metrics"][RECHECK_OBJECTIVE_BASE_METRIC]
        )
        recheck_objective = float(
            recheck_metric_mean + (RECHECK_STD_PENALTY * recheck_metric_std)
        )
        trial_result = {
            "optuna_rank": int(optuna_rank),
            "trial_number": int(trial.number),
            "optuna_objective": float(trial.value),
            "optuna_best_iteration": trial.user_attrs.get("best_iteration"),
            "optuna_cv_binary_logloss_mean": trial.user_attrs.get(
                "cv_binary_logloss_mean"
            ),
            "optuna_cv_binary_logloss_std": trial.user_attrs.get(
                "cv_binary_logloss_std"
            ),
            "params": trial.params,
            "recheck_objective": recheck_objective,
            "recheck_mean_best_iteration": int(cv_result["mean_best_iteration"]),
            "recheck_cv_mean_metrics": cv_result["cv_mean_metrics"],
            "recheck_cv_std_metrics": cv_result["cv_std_metrics"],
            "recheck_folds": cv_result["folds"],
        }
        recheck_results.append(trial_result)
        result_parts = [
            f"recheck result | trial_number={trial.number}",
            f"{RECHECK_OBJECTIVE_BASE_METRIC}={recheck_metric_mean:.8f}",
            f"{RECHECK_OBJECTIVE_BASE_METRIC}_std={recheck_metric_std:.8f}",
        ]
        if RECHECK_OBJECTIVE_BASE_METRIC != "binary_logloss":
            result_parts.extend(
                [
                    f"logloss={cv_result['cv_mean_metrics']['binary_logloss']:.8f}",
                    (
                        "logloss_std="
                        f"{cv_result['cv_std_metrics']['binary_logloss']:.8f}"
                    ),
                ]
            )
        result_parts.extend(
            [
                f"objective={recheck_objective:.8f}",
                f"mean_best_iteration={cv_result['mean_best_iteration']}",
            ]
        )
        print(" ".join(result_parts))

    recheck_results.sort(key=lambda item: float(item["recheck_objective"]))
    for recheck_rank, result in enumerate(recheck_results, start=1):
        result["recheck_rank"] = int(recheck_rank)

    json_path, csv_path = make_top_trial_recheck_output_paths(
        study_name=study_name,
        top_n=top_n,
        output_json=output_json,
        output_csv=output_csv,
    )
    summary_rows = [
        build_top_trial_recheck_summary_row(result) for result in recheck_results
    ]
    best_result = recheck_results[0]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_name": study_name,
        "storage": storage,
        "top_n_requested": int(top_n),
        "top_n_evaluated": int(len(recheck_results)),
        "data_path": str(data_path),
        "feature_selection": summarize_feature_subset(
            feature_subset,
            excluded_features=excluded_features,
        ),
        "sample_weight": {
            "used": True,
            "source": training_data["sample_weight_source"],
            **training_data["sample_weight_summary"],
        },
        "train_pipeline_alignment": {
            "source_script": "train_lgbm.py",
            "cv_folds": FINAL_CV_FOLDS,
            "walk_forward_test_to_train_ratio": FINAL_WF_TEST_TO_TRAIN_RATIO,
        },
        "recheck_objective": {
            "name": RECHECK_OBJECTIVE_NAME,
            "base_metric": RECHECK_OBJECTIVE_BASE_METRIC,
            "aggregation": "cv_mean + std_penalty * cv_std",
            "std_penalty": float(RECHECK_STD_PENALTY),
        },
        "best_trial": build_top_trial_recheck_best_trial(best_result),
        "best_params": best_result["params"],
        "results": recheck_results,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    print(
        f"best recheck | trial_number={best_result['trial_number']} "
        f"recheck_objective={best_result['recheck_objective']:.8f} "
        f"{RECHECK_OBJECTIVE_BASE_METRIC}="
        f"{best_result['recheck_cv_mean_metrics'][RECHECK_OBJECTIVE_BASE_METRIC]:.8f} "
        f"{RECHECK_OBJECTIVE_BASE_METRIC}_std="
        f"{best_result['recheck_cv_std_metrics'][RECHECK_OBJECTIVE_BASE_METRIC]:.8f}"
    )
    print(f"Saved recheck payload: {json_path}")
    print(f"Saved recheck csv: {csv_path}")


def run_optuna_optimization():
    optuna.logging.set_verbosity(optuna.logging.INFO)
    dataset_settings = load_modeling_dataset_settings()
    data_path = resolve_modeling_dataset_output_paths(dataset_settings)["parquet"]
    feature_subset = load_feature_subset_from_settings(dataset_settings)
    excluded_features = load_excluded_feature_names_from_settings(dataset_settings)

    (
        x_np,
        y_np,
        sample_weight_np,
        rows_after_target_notna,
        sample_weight_source,
        sample_weight_summary,
    ) = load_generic_training_data(
        data_path=data_path,
        feature_subset=feature_subset,
        excluded_features=excluded_features,
    )
    folds = make_walk_forward_folds(
        n_rows=len(x_np),
        n_folds=CV_FOLDS,
        test_to_train_ratio=WF_TEST_TO_TRAIN_RATIO,
    )
    fold_indices = build_fold_indices(folds)

    train_set = lgb.Dataset(
        data=x_np,
        label=y_np,
        weight=sample_weight_np,
        free_raw_data=True,
    )

    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=TPE_STARTUP_TRIALS,
        multivariate=True,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=PRUNER_MIN_RESOURCE,
        max_resource=MAX_N_ESTIMATORS,
        reduction_factor=PRUNER_REDUCTION_FACTOR,
        bootstrap_count=PRUNER_BOOTSTRAP_COUNT,
    )

    if STORAGE.startswith("sqlite:///"):
        Path(STORAGE.replace("sqlite:///", "", 1)).parent.mkdir(
            parents=True, exist_ok=True
        )

    print(
        f"start optimize | rows={len(x_np)} features={x_np.shape[1]} folds={len(fold_indices)} "
        f"trials={N_TRIALS} timeout={TIMEOUT_SECONDS} prune_every={PRUNE_REPORT_EVERY_N_ITER} "
        f"pruner_bootstrap_count={PRUNER_BOOTSTRAP_COUNT} "
        f"sample_weight_source={sample_weight_source} "
        f"objective={CV_OBJECTIVE_NAME} std_penalty={CV_LOGLOSS_STD_PENALTY:.4f}"
    )
    print(
        "start optimize | "
        f"search_params={sorted(LGBM_OPTUNA_SEARCH_SPACE)} "
        f"seed_trials_configured={len(OPTUNA_SEED_TRIAL_PARAMS)}"
    )
    if feature_subset:
        print(
            "start optimize | "
            f"feature_subset_path={feature_subset['path']} count={feature_subset['count']}"
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
        search_space=LGBM_OPTUNA_SEARCH_SPACE,
    )

    objective = make_objective(
        train_set=train_set,
        fold_indices=fold_indices,
        search_space=LGBM_OPTUNA_SEARCH_SPACE,
    )
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT_SECONDS,
        n_jobs=OPTUNA_OPTIMIZE_N_JOBS,
        gc_after_trial=False,
        show_progress_bar=True,
        catch=(lgb.basic.LightGBMError, OSError),
    )

    best = study.best_trial
    best_cv_logloss_mean = best.user_attrs.get("cv_binary_logloss_mean")
    best_cv_logloss_std = best.user_attrs.get("cv_binary_logloss_std")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "target_col": TARGET_COL,
        "sample_weight_col": TARGET_WEIGHT_COL,
        "sample_weight": {
            "used": True,
            "source": sample_weight_source,
            **sample_weight_summary,
        },
        "feature_selection": summarize_feature_subset(
            feature_subset,
            excluded_features=excluded_features,
        ),
        "rows_after_target_notna": int(rows_after_target_notna),
        "decision_row_filter": {
            "enabled": False,
        },
        "study_name": STUDY_NAME,
        "storage": STORAGE,
        "lgbm_optuna_search_space": LGBM_OPTUNA_SEARCH_SPACE,
        "optuna_seed_trial_params": OPTUNA_SEED_TRIAL_PARAMS,
        "cv_objective": {
            "name": CV_OBJECTIVE_NAME,
            "base_metric": "binary_logloss",
            "aggregation": "cv_mean + std_penalty * cv_std",
            "std_penalty": float(CV_LOGLOSS_STD_PENALTY),
        },
        "recommended_final_selection": {
            "name": RECHECK_OBJECTIVE_NAME,
            "base_metric": RECHECK_OBJECTIVE_BASE_METRIC,
            "aggregation": "cv_mean + std_penalty * cv_std",
            "std_penalty": float(RECHECK_STD_PENALTY),
            "workflow": "run_top_trials_recheck",
        },
        "n_trials_requested": int(N_TRIALS),
        "timeout_seconds": TIMEOUT_SECONDS,
        "cv_folds": CV_FOLDS,
        "walk_forward_test_to_train_ratio": WF_TEST_TO_TRAIN_RATIO,
        "max_n_estimators": MAX_N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "prune_report_every_n_iteration": PRUNE_REPORT_EVERY_N_ITER,
        "pruner_bootstrap_count": PRUNER_BOOTSTRAP_COUNT,
        "best_trial": {
            "number": int(best.number),
            "objective_cv_binary_logloss_mean_plus_std_penalty": float(best.value),
            "cv_binary_logloss_mean": (
                float(best_cv_logloss_mean)
                if best_cv_logloss_mean is not None
                else None
            ),
            "cv_binary_logloss_std": (
                float(best_cv_logloss_std)
                if best_cv_logloss_std is not None
                else None
            ),
            "best_iteration": best.user_attrs.get("best_iteration"),
            "params": best.params,
        },
    }

    BEST_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    TRIALS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(TRIALS_CSV_PATH, index=False)

    print(
        f"Best trial: #{best.number} "
        f"objective={best.value:.8f} "
        f"logloss_mean={float(best_cv_logloss_mean):.8f} "
        f"logloss_std={float(best_cv_logloss_std):.8f} "
        f"iter={best.user_attrs.get('best_iteration')}"
    )
    print(
        "Recommended final selection | "
        f"workflow=run_top_trials_recheck "
        f"objective={RECHECK_OBJECTIVE_NAME} "
        f"std_penalty={RECHECK_STD_PENALTY:.4f}"
    )
    print(f"Saved best payload: {BEST_RESULT_PATH}")
    print(f"Saved trials csv: {TRIALS_CSV_PATH}")


def main():
    if RUN_MODE == "recheck-topn":
        run_top_trials_recheck(
            study_name=RECHECK_STUDY_NAME,
            storage=RECHECK_STORAGE,
            top_n=TOP_TRIALS_RECHECK_N,
            output_json=TOP_TRIALS_RECHECK_OUTPUT_JSON_PATH,
            output_csv=TOP_TRIALS_RECHECK_OUTPUT_CSV_PATH,
        )
        return

    if RUN_MODE != "optimize":
        raise ValueError(
            f"Unsupported RUN_MODE={RUN_MODE!r}. Expected 'optimize' or 'recheck-topn'."
        )

    run_optuna_optimization()


if __name__ == "__main__":
    main()
