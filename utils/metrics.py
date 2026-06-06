import numpy as np


def _resolve_sample_weight(sample_weight, n_obs):
    if sample_weight is None:
        return np.ones(int(n_obs), dtype=np.float64)

    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("Sample weights must be a 1D array.")
    if weights.shape[0] != int(n_obs):
        raise ValueError(
            "Sample weights length mismatch: " f"{weights.shape[0]} != {int(n_obs)}"
        )
    return weights


def _resolve_positive_class_proba(y_pred):
    pred = np.asarray(y_pred, dtype=np.float64)
    if pred.ndim == 1:
        return pred
    if pred.ndim == 2 and pred.shape[1] == 2:
        return pred[:, 1]
    raise ValueError(
        "Expected binary probabilities as a 1D positive-class array or a 2D array "
        f"with 2 columns, got shape={pred.shape}."
    )


def weighted_brier_score(y_true, y_pred_proba, sample_weight=None):
    y_true_f = np.asarray(y_true, dtype=np.float64)
    y_pred_proba_f = _resolve_positive_class_proba(y_pred_proba)
    if y_true_f.ndim != 1 or y_pred_proba_f.ndim != 1:
        raise ValueError("weighted_brier_score expects 1D arrays.")
    if y_true_f.shape[0] != y_pred_proba_f.shape[0]:
        raise ValueError(
            "weighted_brier_score length mismatch: "
            f"{y_true_f.shape[0]} != {y_pred_proba_f.shape[0]}"
        )

    weights = _resolve_sample_weight(sample_weight, len(y_true_f))
    return float(np.average((y_pred_proba_f - y_true_f) ** 2, weights=weights))


def weighted_binary_logloss(y_true, y_pred_proba, sample_weight=None):
    y_true_f = np.asarray(y_true, dtype=np.float64)
    y_pred_proba_f = _resolve_positive_class_proba(y_pred_proba)
    if y_true_f.ndim != 1 or y_pred_proba_f.ndim != 1:
        raise ValueError("weighted_binary_logloss expects 1D arrays.")
    if y_true_f.shape[0] != y_pred_proba_f.shape[0]:
        raise ValueError(
            "weighted_binary_logloss length mismatch: "
            f"{y_true_f.shape[0]} != {y_pred_proba_f.shape[0]}"
        )

    weights = _resolve_sample_weight(sample_weight, len(y_true_f))
    p = np.clip(y_pred_proba_f, 1e-15, 1.0 - 1e-15)
    loss = -(y_true_f * np.log(p) + (1.0 - y_true_f) * np.log(1.0 - p))
    return float(np.average(loss, weights=weights))


def weighted_balanced_accuracy_score(
    y_true,
    y_pred_proba,
    sample_weight=None,
    threshold=0.5,
):
    y_true_i = np.asarray(y_true, dtype=np.int8)
    y_pred_proba_f = _resolve_positive_class_proba(y_pred_proba)
    if y_true_i.ndim != 1 or y_pred_proba_f.ndim != 1:
        raise ValueError("weighted_balanced_accuracy_score expects 1D arrays.")
    if y_true_i.shape[0] != y_pred_proba_f.shape[0]:
        raise ValueError(
            "weighted_balanced_accuracy_score length mismatch: "
            f"{y_true_i.shape[0]} != {y_pred_proba_f.shape[0]}"
        )

    weights = _resolve_sample_weight(sample_weight, len(y_true_i))
    y_pred_i = (y_pred_proba_f >= float(threshold)).astype(np.int8)

    pos_mask = y_true_i == 1
    neg_mask = y_true_i == 0
    pos_weight = float(np.sum(weights[pos_mask]))
    neg_weight = float(np.sum(weights[neg_mask]))

    tpr = (
        float(np.sum(weights[pos_mask & (y_pred_i == 1)]) / pos_weight)
        if pos_weight > 0.0
        else 0.0
    )
    tnr = (
        float(np.sum(weights[neg_mask & (y_pred_i == 0)]) / neg_weight)
        if neg_weight > 0.0
        else 0.0
    )
    return float((tpr + tnr) / 2.0)


def make_lightgbm_binary_logloss_eval(metric_name="binary_logloss"):
    metric_name = str(metric_name).strip() or "binary_logloss"

    def _eval(preds, train_data):
        return (
            metric_name,
            weighted_binary_logloss(
                y_true=train_data.get_label(),
                y_pred_proba=preds,
                sample_weight=train_data.get_weight(),
            ),
            False,
        )

    return _eval


def make_lightgbm_binary_brier_eval(metric_name="brier_score"):
    metric_name = str(metric_name).strip() or "brier_score"

    def _eval(preds, train_data):
        return (
            metric_name,
            weighted_brier_score(
                y_true=train_data.get_label(),
                y_pred_proba=preds,
                sample_weight=train_data.get_weight(),
            ),
            False,
        )

    return _eval


def make_lightgbm_binary_balanced_accuracy_eval(
    metric_name="balanced_accuracy",
    threshold=0.5,
):
    metric_name = str(metric_name).strip() or "balanced_accuracy"
    threshold_value = float(threshold)

    def _eval(preds, train_data):
        return (
            metric_name,
            weighted_balanced_accuracy_score(
                y_true=train_data.get_label(),
                y_pred_proba=preds,
                sample_weight=train_data.get_weight(),
                threshold=threshold_value,
            ),
            True,
        )

    return _eval


def make_sklearn_binary_logloss_eval(metric_name="binary_logloss"):
    metric_name = str(metric_name).strip() or "binary_logloss"

    def _eval(y_true, y_pred, sample_weight=None):
        return (
            metric_name,
            weighted_binary_logloss(
                y_true=y_true,
                y_pred_proba=y_pred,
                sample_weight=sample_weight,
            ),
            False,
        )

    return _eval


def make_sklearn_binary_brier_eval(metric_name="brier_score"):
    metric_name = str(metric_name).strip() or "brier_score"

    def _eval(y_true, y_pred, sample_weight=None):
        return (
            metric_name,
            weighted_brier_score(
                y_true=y_true,
                y_pred_proba=y_pred,
                sample_weight=sample_weight,
            ),
            False,
        )

    return _eval


def make_sklearn_binary_balanced_accuracy_eval(
    metric_name="balanced_accuracy",
    threshold=0.5,
):
    metric_name = str(metric_name).strip() or "balanced_accuracy"
    threshold_value = float(threshold)

    def _eval(y_true, y_pred, sample_weight=None):
        return (
            metric_name,
            weighted_balanced_accuracy_score(
                y_true=y_true,
                y_pred_proba=y_pred,
                sample_weight=sample_weight,
                threshold=threshold_value,
            ),
            True,
        )

    return _eval
