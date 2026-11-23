"""Scoring, calibration, benchmarks and block bootstrap for E0051 sequence models."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-15


def per_window_scores(prob: np.ndarray, target: np.ndarray) -> dict:
    """Per-window multiclass log loss, Brier (sum over classes) and exact hit."""
    prob = np.asarray(prob, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    rows = np.arange(len(target))
    p_true = np.clip(prob[rows, target], EPS, 1.0)
    onehot = np.zeros_like(prob)
    onehot[rows, target] = 1.0
    return {
        "log_loss": -np.log(p_true),
        "brier": ((prob - onehot) ** 2).sum(axis=1),
        "hit": (prob.argmax(axis=1) == target).astype(np.float64),
    }


def apply_temperature(prob: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(prob, EPS, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def fit_temperature(prob: np.ndarray, target: np.ndarray, lo=0.2, hi=5.0, iters=80) -> float:
    """Golden-section search of the NLL-minimising temperature on [lo, hi]."""
    def nll(t):
        return per_window_scores(apply_temperature(prob, t), target)["log_loss"].mean()

    g = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = np.log(lo), np.log(hi)
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(np.exp(c)), nll(np.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(np.exp(d))
    return float(np.exp((a + b) / 2.0))


def global_benchmark(targets: np.ndarray, n_classes: int, alpha: float = 1.0) -> np.ndarray:
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=n_classes).astype(np.float64)
    return (counts + alpha) / (counts.sum() + alpha * n_classes)


def conditional_benchmark(train_targets, train_states, eval_states, n_classes, prior_strength=20.0):
    """Per-state class frequencies shrunk to the global benchmark."""
    prior = global_benchmark(train_targets, n_classes)
    train_targets = np.asarray(train_targets, dtype=np.int64)
    train_states = np.asarray(train_states, dtype=object)
    table = {}
    for state in np.unique(train_states):
        counts = np.bincount(train_targets[train_states == state], minlength=n_classes).astype(np.float64)
        table[state] = (counts + prior_strength * prior) / (counts.sum() + prior_strength)
    out = np.array([table.get(s, prior) for s in np.asarray(eval_states, dtype=object)])
    return out.reshape(-1, n_classes), {k: v.tolist() for k, v in table.items()}, prior


def block_bootstrap_mean_diff(diff: np.ndarray, days: np.ndarray, *, block_days=3, reps=2000, seed=0):
    """Paired circular moving-block bootstrap over UTC days of a per-window difference.

    Returns mean, 2.5/97.5 percentiles, number of days, blocks per replicate and
    observations. Days are the sorted unique UTC dates present.
    """
    diff = np.asarray(diff, dtype=np.float64)
    day_index = pd.DatetimeIndex(days).normalize()
    uniq = np.array(sorted(set(day_index.asi8)))
    pos = np.searchsorted(uniq, day_index.asi8)
    sums = np.bincount(pos, weights=diff, minlength=len(uniq))
    cnts = np.bincount(pos, minlength=len(uniq)).astype(np.float64)
    n_days = len(uniq)
    blocks = int(np.ceil(n_days / block_days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_days, size=(reps, blocks))
    offsets = np.arange(block_days)
    picked = (starts[:, :, None] + offsets[None, None, :]) % n_days
    picked = picked.reshape(reps, -1)[:, :n_days]
    stats = sums[picked].sum(axis=1) / cnts[picked].sum(axis=1)
    return {
        "mean": float(diff.mean()) if len(diff) else None,
        "ci_low": float(np.percentile(stats, 2.5)),
        "ci_high": float(np.percentile(stats, 97.5)),
        "days": int(n_days),
        "blocks_per_replicate": blocks,
        "observations": int(len(diff)),
    }


def block_bootstrap_stat(days, arrays: dict, stat_fn, *, block_days=3, reps=2000, seed=0):
    """Common-block circular moving-block bootstrap of a statistic of window sums.

    ``arrays`` maps names to per-window values on one calendar; their daily sums
    are resampled with the same day blocks in every replicate, so statistics
    that compare several methods use common blocks. ``stat_fn`` receives a dict
    of totals (arrays of shape (reps,) or scalars for the point estimate) and
    returns the statistic. Returns estimate, 2.5/97.5 percentiles (NaN
    replicates, e.g. empty selections, are dropped and counted), days and blocks.
    """
    day_index = pd.DatetimeIndex(days).normalize()
    uniq = np.array(sorted(set(day_index.asi8)))
    pos = np.searchsorted(uniq, day_index.asi8)
    n_days = len(uniq)
    daily = {k: np.bincount(pos, weights=np.asarray(v, dtype=np.float64), minlength=n_days)
             for k, v in arrays.items()}
    blocks = int(np.ceil(n_days / block_days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_days, size=(reps, blocks))
    picked = ((starts[:, :, None] + np.arange(block_days)[None, None, :]) % n_days).reshape(reps, -1)[:, :n_days]
    with np.errstate(invalid="ignore", divide="ignore"):
        estimate = float(stat_fn({k: v.sum() for k, v in daily.items()}))
        stats = np.asarray(stat_fn({k: v[picked].sum(axis=1) for k, v in daily.items()}), dtype=np.float64)
    ok = np.isfinite(stats)
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(stats[ok], 2.5)) if ok.any() else float("nan"),
        "ci_high": float(np.percentile(stats[ok], 97.5)) if ok.any() else float("nan"),
        "days": int(n_days),
        "block_days": int(block_days),
        "blocks_per_replicate": blocks,
        "undefined_replicates": int((~ok).sum()),
    }


def rolling_prior_quantiles(stream_ns, stream_scores, query_ns, *, window_ns, quantiles, min_count):
    """Quantiles of earlier stream scores for each query time (E0052 selector).

    For a query ``T`` the window holds stream scores with ``T - window_ns <= t < T``
    (strictly earlier decisions, labels never consulted). Quantiles use numpy's
    default linear interpolation. Queries with fewer than ``min_count`` scores
    get NaN thresholds. ``stream_ns`` must be sorted. Returns (thresholds of shape
    (len(query), len(quantiles)), window counts).
    """
    stream_ns = np.asarray(stream_ns, dtype=np.int64)
    scores = np.asarray(stream_scores, dtype=np.float64)
    query_ns = np.asarray(query_ns, dtype=np.int64)
    if len(stream_ns) > 1 and (np.diff(stream_ns) < 0).any():
        raise ValueError("stream_ns must be sorted")
    lo = np.searchsorted(stream_ns, query_ns - int(window_ns), side="left")
    hi = np.searchsorted(stream_ns, query_ns, side="left")
    out = np.full((len(query_ns), len(quantiles)), np.nan)
    counts = hi - lo
    for i in np.flatnonzero(counts >= min_count):
        out[i] = np.quantile(scores[lo[i]:hi[i]], quantiles)
    return out, counts
