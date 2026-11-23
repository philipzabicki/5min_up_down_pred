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
