"""Small, testable calculations used by the BTC OOF target audit."""

import numpy as np
import pandas as pd

from utils.data import compute_binary_close_target_from_opened


METRIC_NAMES = ("up_rate", "log_loss", "brier", "auc", "accuracy")


def score_probabilities(y_true, probability, sample_weight=None):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Labels and probabilities must be non-empty 1D arrays of equal length")
    if not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("Labels must be finite binary values")
    if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")

    if sample_weight is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)
        if w.shape != y.shape or not np.isfinite(w).all() or (w <= 0.0).any():
            raise ValueError("Sample weights must be finite, positive, and match labels")

    p_clip = np.clip(p, 1e-15, 1.0 - 1e-15)
    total = float(w.sum())
    positive = float(np.dot(w, y))
    negative = total - positive
    predicted_up = p >= 0.5
    true_positive_weight = float(w[predicted_up & (y == 1.0)].sum())
    true_negative_weight = float(w[(~predicted_up) & (y == 0.0)].sum())
    correct = (true_positive_weight + true_negative_weight) / total

    auc = float("nan")
    if positive > 0.0 and negative > 0.0:
        order = np.argsort(p, kind="mergesort")
        p_sorted = p[order]
        y_sorted = y[order]
        w_sorted = w[order]
        starts = np.r_[0, np.flatnonzero(p_sorted[1:] != p_sorted[:-1]) + 1]
        positive_by_score = np.add.reduceat(w_sorted * y_sorted, starts)
        negative_by_score = np.add.reduceat(w_sorted * (1.0 - y_sorted), starts)
        negative_before = np.cumsum(negative_by_score) - negative_by_score
        auc = float(
            np.dot(
                positive_by_score,
                negative_before + 0.5 * negative_by_score,
            )
            / (positive * negative)
        )

    log_loss = -float(
        np.dot(w, y * np.log(p_clip) + (1.0 - y) * np.log1p(-p_clip)) / total
    )
    brier = float(np.dot(w, np.square(p - y)) / total)
    return {
        "rows": int(len(y)),
        "weight_sum": total,
        "up_rate": positive / total,
        "log_loss": log_loss,
        "brier": brier,
        "auc": auc,
        "accuracy": correct,
    }


def training_classification_metrics(y_true, probability, sample_weight):
    """Reproduce train_lgbm.classification_metrics on its weighted sample."""
    base = score_probabilities(y_true, probability, sample_weight)
    y = np.asarray(y_true, dtype=np.int8)
    pred = (np.asarray(probability, dtype=np.float64) >= 0.5).astype(np.int8)
    w = np.asarray(sample_weight, dtype=np.float64)
    tp = float(w[(y == 1) & (pred == 1)].sum())
    tn = float(w[(y == 0) & (pred == 0)].sum())
    fp = float(w[(y == 0) & (pred == 1)].sum())
    fn = float(w[(y == 1) & (pred == 0)].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": base["accuracy"],
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": base["brier"],
        "binary_logloss": base["log_loss"],
    }


def recompute_close_target(opened, close, horizon_minutes=5):
    """Use exact UTC timestamp lookup; missing future timestamps stay missing."""
    return compute_binary_close_target_from_opened(
        opened_values=opened,
        close_values=pd.Series(close),
        horizon_minutes=horizon_minutes,
    )


def validate_common_market_sample(evaluation, common, expected_rows=9407):
    """Join by immutable market identity and enforce identical OOF observations."""
    required_eval = {
        "condition_id",
        "market_start_utc",
        "target_polymarket_up",
        "p_model_up",
        "oof_raw",
    }
    required_common = {
        "condition_id",
        "market_start_utc",
        "target_binance_proxy_up",
        "target_polymarket_up",
        "polymarket_outcome_up",
        "p_model_up",
    }
    if not required_eval.issubset(evaluation.columns):
        raise ValueError(f"Evaluation sample missing columns: {sorted(required_eval - set(evaluation.columns))}")
    if not required_common.issubset(common.columns):
        raise ValueError(f"Common markets missing columns: {sorted(required_common - set(common.columns))}")
    if len(evaluation) != int(expected_rows):
        raise ValueError(f"Expected {expected_rows} market-value rows, found {len(evaluation)}")
    if evaluation.condition_id.duplicated().any() or common.condition_id.duplicated().any():
        raise ValueError("condition_id must be unique in evaluation and common-market inputs")

    ids = set(evaluation.condition_id.astype(str))
    matched_common = common[common.condition_id.astype(str).isin(ids)].copy()
    if len(matched_common) != int(expected_rows) or set(matched_common.condition_id.astype(str)) != ids:
        raise ValueError("The common-market file does not contain the exact evaluated condition_id set")

    joined = evaluation.merge(
        matched_common,
        on="condition_id",
        how="inner",
        suffixes=("_evaluation", "_common"),
        validate="one_to_one",
    )
    checks = {
        "market_start_utc": ("market_start_utc_evaluation", "market_start_utc_common"),
        "target_polymarket_up": ("target_polymarket_up_evaluation", "target_polymarket_up_common"),
        "p_model_up": ("p_model_up_evaluation", "p_model_up_common"),
    }
    for name, (left, right) in checks.items():
        a, b = joined[left], joined[right]
        if name == "p_model_up":
            if not np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=0.0, atol=1e-14):
                raise ValueError("Raw OOF probabilities differ between market-value and common files")
        elif not a.equals(b):
            raise ValueError(f"{name} differs between market-value and common files")
    if not np.allclose(
            joined.target_polymarket_up_evaluation.to_numpy(float),
            joined.polymarket_outcome_up.to_numpy(float),
            rtol=0.0,
            atol=0.0,
    ):
        raise ValueError("Market-value settlement labels differ from official common-market outcomes")
    if not np.allclose(
            joined.p_model_up_evaluation.to_numpy(float),
            joined.oof_raw.to_numpy(float),
            rtol=0.0,
            atol=1e-14,
    ):
        raise ValueError("The market-value raw OOF column differs from its source OOF probabilities")
    return joined


def moving_calendar_block_multiplicity(
        timestamps,
        rng,
        *,
        block_days=3,
):
    """Draw moving calendar-day blocks and return one multiplicity per row."""
    times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="raise"))
    if times.isna().any() or len(times) == 0:
        raise ValueError("Bootstrap timestamps must be non-empty and valid")
    days = times.floor("D")
    calendar = pd.date_range(days.min(), days.max(), freq="D", tz="UTC")
    day_codes = ((days - calendar[0]) / pd.Timedelta(days=1)).to_numpy(dtype=np.int64)
    block_days = max(int(block_days), 1)
    n_blocks = int(np.ceil(len(calendar) / block_days))
    max_start = max(len(calendar) - block_days, 0)
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    drawn_days = np.concatenate(
        [np.arange(start, min(start + block_days, len(calendar))) for start in starts]
    )[:len(calendar)]
    day_multiplicity = np.bincount(drawn_days, minlength=len(calendar))
    return day_multiplicity[day_codes]


def paired_block_bootstrap(
        targets,
        probability,
        timestamps,
        *,
        sample_weight=None,
        replications=2000,
        block_days=3,
        seed=37,
):
    """Paired, same-row moving-block bootstrap for multiple target labels."""
    p = np.asarray(probability, dtype=np.float64)
    times = pd.to_datetime(timestamps, utc=True, errors="raise")
    if len(p) != len(times):
        raise ValueError("Probability and timestamp lengths differ")
    ys = {name: np.asarray(values, dtype=np.float64) for name, values in targets.items()}
    if len(ys) < 1 or any(len(y) != len(p) for y in ys.values()):
        raise ValueError("Each target must match the probability rows")
    base_weight = (
        np.ones(len(p), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    if base_weight.shape != p.shape:
        raise ValueError("Bootstrap sample weights differ in length")

    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    score_starts = np.r_[0, np.flatnonzero(p_sorted[1:] != p_sorted[:-1]) + 1]
    sorted_targets = {name: y[order] for name, y in ys.items()}
    draws = {name: {metric: [] for metric in METRIC_NAMES} for name in ys}
    names = list(ys)
    delta_draws = (
        {metric: [] for metric in METRIC_NAMES}
        if len(names) == 2
        else None
    )
    rng = np.random.default_rng(seed)

    def score_with_weight(y, row_weight, y_sorted):
        total = float(row_weight.sum())
        up_weight = float(np.dot(row_weight, y))
        p_clip = np.clip(p, 1e-15, 1.0 - 1e-15)
        log_loss = -float(
            np.dot(row_weight, y * np.log(p_clip) + (1.0 - y) * np.log1p(-p_clip))
            / total
        )
        brier = float(np.dot(row_weight, np.square(p - y)) / total)
        accuracy = float(np.dot(row_weight, ((p >= 0.5) == y)) / total)
        auc = float("nan")
        positive = up_weight
        negative = total - positive
        if positive > 0.0 and negative > 0.0:
            w_sorted = row_weight[order]
            positive_by_score = np.add.reduceat(w_sorted * y_sorted, score_starts)
            negative_by_score = np.add.reduceat(w_sorted * (1.0 - y_sorted), score_starts)
            negative_before = np.cumsum(negative_by_score) - negative_by_score
            auc = float(
                np.dot(positive_by_score, negative_before + 0.5 * negative_by_score)
                / (positive * negative)
            )
        return {
            "up_rate": up_weight / total,
            "log_loss": log_loss,
            "brier": brier,
            "auc": auc,
            "accuracy": accuracy,
        }

    for _ in range(int(replications)):
        multiplicity = moving_calendar_block_multiplicity(
            times,
            rng,
            block_days=block_days,
        )
        row_weight = base_weight * multiplicity
        scored = {}
        for name, y in ys.items():
            scored[name] = score_with_weight(y, row_weight, sorted_targets[name])
            for metric in METRIC_NAMES:
                draws[name][metric].append(scored[name][metric])
        if delta_draws is not None:
            for metric in METRIC_NAMES:
                delta_draws[metric].append(scored[names[1]][metric] - scored[names[0]][metric])

    intervals = {
        name: {
            metric: np.nanquantile(values, [0.025, 0.975]).tolist()
            for metric, values in metrics.items()
        }
        for name, metrics in draws.items()
    }
    if delta_draws is not None:
        intervals[f"{names[1]}_minus_{names[0]}"] = {
            metric: np.nanquantile(values, [0.025, 0.975]).tolist()
            for metric, values in delta_draws.items()
        }
    intervals["method"] = {
        "type": "paired moving calendar-day block bootstrap",
        "block_days": int(block_days),
        "replications": int(replications),
        "seed": int(seed),
    }
    return intervals


def macro_average_fold_metrics(fold_metrics):
    """The training report averages fold metrics, rather than pooling rows."""
    if not fold_metrics:
        raise ValueError("At least one fold is required")
    keys = tuple(fold_metrics[0])
    if any(tuple(row) != keys for row in fold_metrics):
        raise ValueError("Fold metric names do not match")
    return {
        key: float(np.mean([row[key] for row in fold_metrics]))
        for key in keys
    }
