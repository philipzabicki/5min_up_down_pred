"""Shared fold-metric objective and LightGBM CV callbacks."""

from __future__ import annotations

import hashlib

import numpy as np
import optuna
from lightgbm.callback import EarlyStopException


def combine_mean_std(mean_value, std_value, *, higher_is_better, std_penalty):
    sign = -1.0 if bool(higher_is_better) else 1.0
    return float(mean_value) + sign * float(std_penalty) * float(std_value)


def fold_metric_signature(y_values, sample_weight_values):
    y_array = np.ascontiguousarray(np.asarray(y_values, dtype=np.float64).ravel())
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray(y_array.shape, dtype=np.int64).tobytes())
    digest.update(y_array.tobytes())
    if sample_weight_values is None:
        digest.update(np.asarray([-1], dtype=np.int64).tobytes())
    else:
        weight_array = np.ascontiguousarray(
            np.asarray(sample_weight_values, dtype=np.float32).ravel()
        )
        digest.update(np.asarray(weight_array.shape, dtype=np.int64).tobytes())
        digest.update(weight_array.tobytes())
    return digest.hexdigest()


def build_fold_metric_signatures(folds, y_values, sample_weight_values):
    signatures = {}
    for fold in folds:
        if "valid_idx" in fold:
            valid_indices = np.asarray(fold["valid_idx"], dtype=np.int64)
        else:
            valid_indices = np.arange(
                int(fold["test_start"]), int(fold["test_end"]), dtype=np.int64
            )
        weights = (
            None
            if sample_weight_values is None
            else np.asarray(sample_weight_values)[valid_indices]
        )
        signature = fold_metric_signature(np.asarray(y_values)[valid_indices], weights)
        if signature in signatures:
            raise ValueError(
                "Duplicate validation fold signature for fold metric extraction: "
                f"{signatures[signature]} and {fold['fold_id']}."
            )
        signatures[signature] = int(fold["fold_id"])
    return signatures


def make_fold_metric_eval(metric_name, fold_metric_signatures, scorer, *, higher_is_better=False):
    """Add each fold's score as a separate CV metric for weighted objectives."""
    metric_name = str(metric_name).strip()
    dataset_fold_ids = {}

    def _eval(predictions, valid_data):
        dataset_key = id(valid_data)
        fold_id = dataset_fold_ids.get(dataset_key)
        if fold_id is None:
            signature = fold_metric_signature(
                valid_data.get_label(), valid_data.get_weight()
            )
            if signature not in fold_metric_signatures:
                raise ValueError("Could not match LightGBM validation fold signature.")
            fold_id = int(fold_metric_signatures[signature])
            dataset_fold_ids[dataset_key] = fold_id

        score = float(
            scorer(
                valid_data.get_label(),
                predictions,
                valid_data.get_weight(),
            )
        )
        return [
            (metric_name, score, bool(higher_is_better)),
            (f"{metric_name}_fold_{fold_id}", score, bool(higher_is_better)),
        ]

    return _eval


def cv_objective_series(
        cv_results,
        metric_name,
        *,
        fold_ids=None,
        fold_weights=None,
        higher_is_better=False,
        std_penalty=0.0,
):
    """Build the same mean, dispersion, and objective series used by callbacks."""
    if fold_ids is not None and fold_weights is not None:
        fold_scores = np.vstack([
            np.asarray(
                cv_results[f"valid {metric_name}_fold_{int(fold_id)}-mean"],
                dtype=np.float64,
            )
            for fold_id in fold_ids
        ])
        weights = np.asarray(fold_weights, dtype=np.float64)
        if fold_scores.shape[0] != len(weights):
            raise ValueError("Fold score and weight counts differ.")
        mean_series = np.mean(fold_scores, axis=0)
        weighted_mean_series = np.average(fold_scores, axis=0, weights=weights)
        base_series = weighted_mean_series
        std_series = np.std(fold_scores, axis=0)
    else:
        mean_series = np.asarray(
            cv_results[f"valid {metric_name}-mean"], dtype=np.float64
        )
        weighted_mean_series = mean_series
        base_series = mean_series
        std_series = np.asarray(
            cv_results[f"valid {metric_name}-stdv"], dtype=np.float64
        )

    sign = -1.0 if bool(higher_is_better) else 1.0
    objective_series = base_series + sign * float(std_penalty) * std_series
    return {
        "mean": mean_series,
        "weighted_mean": weighted_mean_series,
        "std": std_series,
        "objective_base": base_series,
        "objective": objective_series,
    }


class ObjectiveAlignedCVCallback:
    """Use one fold objective for Optuna pruning and LightGBM early stopping."""

    order = 30
    before_iteration = False

    def __init__(
            self,
            metric_name,
            *,
            fold_ids=None,
            fold_weights=None,
            higher_is_better=False,
            std_penalty=0.0,
            stopping_rounds=0,
            trial=None,
            report_interval=1,
            min_delta=0.0,
    ):
        self.metric_name = str(metric_name).strip()
        self.fold_ids = None if fold_ids is None else tuple(int(x) for x in fold_ids)
        self.fold_weights = (
            None if fold_weights is None
            else np.asarray(fold_weights, dtype=np.float64)
        )
        if self.fold_ids is not None and self.fold_weights is not None:
            if len(self.fold_ids) != len(self.fold_weights):
                raise ValueError("Fold id and weight counts differ.")
            if not np.isfinite(self.fold_weights).all() or np.any(self.fold_weights <= 0):
                raise ValueError("Fold weights must be finite and positive.")
        self.higher_is_better = bool(higher_is_better)
        if trial is not None:
            study = getattr(trial, "study", None)
            direction = getattr(study, "direction", None)
            expected_direction = (
                optuna.study.StudyDirection.MAXIMIZE
                if self.higher_is_better
                else optuna.study.StudyDirection.MINIMIZE
            )
            if direction is not None and direction != expected_direction:
                raise ValueError(
                    "The CV objective direction differs from the Optuna study direction."
                )
        self.std_penalty = float(std_penalty)
        self.stopping_rounds = int(stopping_rounds)
        self.trial = trial
        self.report_interval = int(report_interval)
        self.min_delta = float(min_delta)
        if self.report_interval <= 0:
            raise ValueError("report_interval must be positive.")
        if self.stopping_rounds < 0 or self.min_delta < 0:
            raise ValueError("stopping_rounds and min_delta cannot be negative.")
        self.best_iteration = -1
        self.best_value = None
        self.best_score_list = None

    def _objective_from_evaluations(self, evaluation_result_list):
        target_names = {"valid", "cv_agg"}
        aggregate = None
        fold_scores = {}
        for entry in evaluation_result_list or ():
            dataset_name, name = entry[:2]
            if dataset_name not in target_names:
                continue
            if name == self.metric_name:
                aggregate = entry
            elif name.startswith(f"{self.metric_name}_fold_"):
                try:
                    fold_id = int(name.rsplit("_", 1)[1])
                except ValueError:
                    continue
                fold_scores[fold_id] = float(entry[2])

        if aggregate is None:
            raise ValueError(
                f"CV metric {self.metric_name!r} not found in evaluation results."
            )
        if bool(aggregate[3]) != self.higher_is_better:
            raise ValueError("CV metric direction differs from the frozen objective.")

        if self.fold_ids is not None and self.fold_weights is not None:
            missing = [fold_id for fold_id in self.fold_ids if fold_id not in fold_scores]
            if missing:
                raise ValueError(f"Missing fold metrics for objective: {missing}")
            scores = np.asarray([fold_scores[x] for x in self.fold_ids], dtype=np.float64)
            mean_value = float(np.mean(scores))
            base_value = float(np.average(scores, weights=self.fold_weights))
            std_value = float(np.std(scores))
        else:
            mean_value = float(aggregate[2])
            base_value = mean_value
            std_value = float(aggregate[4])
        value = combine_mean_std(
            base_value,
            std_value,
            higher_is_better=self.higher_is_better,
            std_penalty=self.std_penalty,
        )
        return value, mean_value, base_value, std_value

    def __call__(self, env):
        value, _mean, _base, _std = self._objective_from_evaluations(
            env.evaluation_result_list
        )
        if self.trial is not None and (env.iteration + 1) % self.report_interval == 0:
            self.trial.report(value, step=env.iteration)
            if self.trial.should_prune():
                raise optuna.TrialPruned(
                    f"Trial was pruned at iteration {env.iteration}."
                )

        improved = (
            self.best_value is None
            or (
                value > self.best_value + self.min_delta
                if self.higher_is_better
                else value < self.best_value - self.min_delta
            )
        )
        if improved:
            self.best_iteration = int(env.iteration)
            self.best_value = float(value)
            self.best_score_list = list(env.evaluation_result_list)
        elif (
            self.stopping_rounds > 0
            and env.iteration - self.best_iteration >= self.stopping_rounds
        ):
            raise EarlyStopException(self.best_iteration, self.best_score_list)
