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


def weighted_brier_score(y_true, y_pred_proba, sample_weight=None):
    y_true_f = np.asarray(y_true, dtype=np.float64)
    y_pred_proba_f = np.asarray(y_pred_proba, dtype=np.float64)
    if y_true_f.ndim != 1 or y_pred_proba_f.ndim != 1:
        raise ValueError("weighted_brier_score expects 1D arrays.")
    if y_true_f.shape[0] != y_pred_proba_f.shape[0]:
        raise ValueError(
            "weighted_brier_score length mismatch: "
            f"{y_true_f.shape[0]} != {y_pred_proba_f.shape[0]}"
        )

    weights = _resolve_sample_weight(sample_weight, len(y_true_f))
    return float(np.average((y_pred_proba_f - y_true_f) ** 2, weights=weights))


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
