import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from modeling_dataset_utils import (
    load_modeling_dataset_settings,
    resolve_oof_prediction_output_paths,
)


# Edit these constants directly. This script intentionally has no CLI.
OOF_PATH = None
OUTPUT_DIR = Path("data/analysis/lucky_run")

TIME_COL = "Opened"
PRED_COL = "oof_pred_proba_up"
TARGET_COL = "target_5m_candle_up"
PREDICTION_THRESHOLD = 0.5
FIXED_PRICE_CAP = 0.53
USE_POLYMARKET_PRICES = False
DECISION_ROWS_ONLY = True

OOF_ONLY_WARNING = (
    "This is OOF-only analysis with fixed worst-case price cap assumption; it does "
    "not model actual Polymarket price availability."
)

MIN_ROWS = 100
MIN_SURVIVAL_PROB = 0.5
HALF_LIFE_DAYS_GRID = [7, 14, 30, 60, 90, 180]
ROLLING_WINDOWS_DAYS = [7, 14, 30, 60, 90, 180]
STARTS_PER_DAY_GRID = [1, 2, 3, 4]
MANUAL_MAX_STEPS_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]
N_RANDOM_SEEDS = 100
SAVE_BY_DAY_MAX_ROWS = 250_000

SUMMARY_JSON = "lucky_run_summary.json"
WEIGHTED_SURVIVAL_CSV = "forward_streak_survival_weighted.csv"
ROLLING_SURVIVAL_CSV = "forward_streak_survival_rolling.csv"
COMPLETED_STREAK_CSV = "completed_streak_summary.csv"
SIMULATION_SUMMARY_CSV = "daily_start_simulation_summary.csv"
SIMULATION_BY_SEED_CSV = "daily_start_simulation_by_seed.csv"
SIMULATION_BY_DAY_CSV = "daily_start_simulation_by_day.csv"


def resolve_oof_path():
    if OOF_PATH is not None:
        path = Path(OOF_PATH)
        if not path.exists():
            raise FileNotFoundError(f"OOF_PATH does not exist: {path}")
        return path

    resolver_errors = []
    try:
        settings = load_modeling_dataset_settings()
        paths = resolve_oof_prediction_output_paths(
            settings,
            preview_rows=int(settings["preview_rows"]),
        )
        path = Path(paths["parquet"])
        if path.exists():
            return path
        resolver_errors.append(f"resolved path does not exist: {path}")
    except Exception as exc:
        resolver_errors.append(str(exc))

    details = "; ".join(resolver_errors) if resolver_errors else "no resolver detail"
    raise FileNotFoundError(
        "Could not find OOF predictions automatically. Set OOF_PATH at the top "
        f"of analyze_lucky_run_oof.py to a .parquet file. Details: {details}"
    )


def read_oof_frame(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported OOF file extension for {path}. Use .parquet.")


def require_columns(frame, required_cols, *, source_path):
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        available = ", ".join(str(col) for col in frame.columns)
        raise ValueError(
            f"OOF file is missing required columns {missing}: {source_path}. "
            f"Available columns: {available}"
        )


def load_analysis_frame(path):
    frame = read_oof_frame(path)
    required_cols = [TIME_COL, PRED_COL, TARGET_COL]
    require_columns(frame, required_cols, source_path=path)
    all_oof_rows_loaded = int(len(frame))

    out = frame.loc[:, required_cols].rename(
        columns={
            TIME_COL: "oof_time",
            PRED_COL: "oof_proba_up",
            TARGET_COL: "oof_target",
        }
    )
    out = out.copy()
    out["oof_time"] = pd.to_datetime(out["oof_time"], errors="coerce", utc=True)
    out["oof_time"] = out["oof_time"].dt.tz_convert(None)
    out["oof_proba_up"] = pd.to_numeric(out["oof_proba_up"], errors="coerce")
    out["oof_target"] = pd.to_numeric(out["oof_target"], errors="coerce")
    out = out.dropna(subset=["oof_time", "oof_proba_up", "oof_target"]).copy()
    usable_oof_rows = int(len(out))

    target_values = set(out["oof_target"].dropna().unique().tolist())
    if not target_values.issubset({0, 1, 0.0, 1.0}):
        preview = sorted(str(value) for value in target_values)[:20]
        raise ValueError(
            "Target must be binary after dropping NaN values. "
            f"Observed values preview: {preview}"
        )
    if not out["oof_proba_up"].between(0.0, 1.0).all():
        bad_count = int((~out["oof_proba_up"].between(0.0, 1.0)).sum())
        raise ValueError(f"OOF predictions must be in [0, 1]. Bad rows: {bad_count}")

    out["oof_target"] = out["oof_target"].astype(np.int8)
    out = out.sort_values("oof_time", kind="stable").reset_index(drop=True)
    if DECISION_ROWS_ONLY:
        bucket_start = out["oof_time"].dt.floor("5min")
        bucket_end = bucket_start + pd.Timedelta(minutes=4)
        out = out.loc[out["oof_time"] == bucket_end].copy()
        out["decision_time"] = bucket_start.loc[out.index] + pd.Timedelta(minutes=5)
        out = out.drop_duplicates(subset=["decision_time"], keep="last")
    else:
        out["decision_time"] = out["oof_time"]

    out = out.sort_values("decision_time", kind="stable").reset_index(drop=True)
    if len(out) < MIN_ROWS:
        raise ValueError(
            f"Not enough usable decision rows after filtering: {len(out)}. "
            f"MIN_ROWS={MIN_ROWS}."
        )

    out["event_time"] = out["decision_time"]
    out["pred_side"] = out["oof_proba_up"] >= float(PREDICTION_THRESHOLD)
    out["correct"] = (
        out["pred_side"].to_numpy(dtype=bool) == out["oof_target"].to_numpy(dtype=bool)
    )
    out["event_day"] = out["event_time"].dt.floor("D")
    out.attrs["all_oof_rows_loaded"] = all_oof_rows_loaded
    out.attrs["usable_oof_rows"] = usable_oof_rows
    out.attrs["decision_rows_count"] = int(len(out))
    return out


def compute_forward_streak_lengths(correct):
    correct = np.asarray(correct, dtype=bool)
    streaks = np.zeros(len(correct), dtype=np.int32)
    run_len = 0
    for idx in range(len(correct) - 1, -1, -1):
        if correct[idx]:
            run_len += 1
        else:
            run_len = 0
        streaks[idx] = run_len
    return streaks


def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=np.float64)
    weight_sq_sum = float(np.sum(weights * weights))
    if weight_sq_sum <= 0.0:
        return 0.0
    weight_sum = float(np.sum(weights))
    return float((weight_sum * weight_sum) / weight_sq_sum)


def choose_max_steps_from_survival(survival_frame, prob_col):
    if survival_frame.empty or prob_col not in survival_frame.columns:
        return 0
    eligible = survival_frame.loc[
        survival_frame[prob_col] >= float(MIN_SURVIVAL_PROB),
        "k",
    ]
    if eligible.empty:
        return 0
    return int(eligible.max())


def build_survival_table(streaks, *, weights=None):
    streaks = np.asarray(streaks, dtype=np.int32)
    max_k = int(streaks.max()) if len(streaks) else 0
    if weights is None:
        weights = np.ones(len(streaks), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    total = int(len(streaks))
    weight_total = float(np.sum(weights))
    columns = [
        "k",
        "raw_count",
        "raw_probability",
        "weighted_count",
        "weighted_probability",
    ]
    rows = []
    for k in range(1, max_k + 1):
        mask = streaks >= k
        raw_count = int(np.count_nonzero(mask))
        weighted_count = float(np.sum(weights[mask]))
        rows.append(
            {
                "k": int(k),
                "raw_count": raw_count,
                "raw_probability": float(raw_count / total) if total else float("nan"),
                "weighted_count": weighted_count,
                "weighted_probability": (
                    float(weighted_count / weight_total)
                    if weight_total > 0.0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compute_recency_weights(event_time, half_life_days):
    max_time = event_time.max()
    age_days = (max_time - event_time).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    return np.power(0.5, age_days / float(half_life_days)).astype(np.float64, copy=False)


def summarize_basic(frame):
    per_day = frame.groupby("event_day", sort=True).size()
    return {
        "all_oof_rows_loaded": int(frame.attrs.get("all_oof_rows_loaded", len(frame))),
        "usable_oof_rows": int(frame.attrs.get("usable_oof_rows", len(frame))),
        "decision_rows_only": bool(DECISION_ROWS_ONLY),
        "decision_rows_count": int(frame.attrs.get("decision_rows_count", len(frame))),
        "row_count": int(len(frame)),
        "decision_row_accuracy": float(frame["correct"].mean()),
        "accuracy": float(frame["correct"].mean()),
        "start_time": frame["event_time"].iloc[0].isoformat(),
        "end_time": frame["event_time"].iloc[-1].isoformat(),
        "day_count": int(per_day.size),
        "decision_rows_per_day_mean": float(per_day.mean()),
        "decision_rows_per_day_median": float(per_day.median()),
    }


def build_weighted_survival_outputs(frame, streaks):
    rows = []
    summary_rows = []
    target = frame["oof_target"].to_numpy(dtype=np.int8, copy=False)
    correct = frame["correct"].to_numpy(dtype=bool, copy=False)
    for half_life_days in HALF_LIFE_DAYS_GRID:
        weights = compute_recency_weights(frame["event_time"], half_life_days)
        survival = build_survival_table(streaks, weights=weights)
        chosen_max_steps = choose_max_steps_from_survival(
            survival,
            "weighted_probability",
        )
        ess = effective_sample_size(weights)
        weighted_accuracy = float(np.average(correct.astype(np.float64), weights=weights))
        weighted_event_rate = float(np.average(target.astype(np.float64), weights=weights))

        survival = survival.copy()
        survival.insert(0, "half_life_days", int(half_life_days))
        survival["chosen_max_steps"] = int(chosen_max_steps)
        survival["effective_sample_size"] = float(ess)
        rows.append(survival)
        summary_rows.append(
            {
                "half_life_days": int(half_life_days),
                "row_count": int(len(frame)),
                "effective_sample_size": float(ess),
                "weight_sum": float(np.sum(weights)),
                "accuracy": float(correct.mean()),
                "weighted_accuracy": weighted_accuracy,
                "target_event_rate": float(target.mean()),
                "weighted_target_event_rate": weighted_event_rate,
                "chosen_max_steps": int(chosen_max_steps),
            }
        )

    survival_frame = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    summary_frame = pd.DataFrame(summary_rows)
    return survival_frame, summary_frame


def build_rolling_survival_outputs(frame, streaks):
    rows = []
    summary_rows = []
    max_time = frame["event_time"].max()
    for window_days in ROLLING_WINDOWS_DAYS:
        cutoff = max_time - pd.Timedelta(days=int(window_days))
        mask = frame["event_time"] >= cutoff
        window_frame = frame.loc[mask].copy()
        window_streaks = streaks[mask.to_numpy()]
        survival = build_survival_table(window_streaks)
        chosen_max_steps = choose_max_steps_from_survival(
            survival,
            "raw_probability",
        )
        day_count = int(window_frame["event_day"].nunique())
        survival = survival.copy()
        survival.insert(0, "window_days", int(window_days))
        survival["row_count"] = int(len(window_frame))
        survival["day_count"] = day_count
        survival["chosen_max_steps"] = int(chosen_max_steps)
        rows.append(survival)
        summary_rows.append(
            {
                "window_days": int(window_days),
                "row_count": int(len(window_frame)),
                "day_count": day_count,
                "accuracy": (
                    float(window_frame["correct"].mean())
                    if len(window_frame)
                    else float("nan")
                ),
                "chosen_max_steps": int(chosen_max_steps),
            }
        )

    survival_frame = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    summary_frame = pd.DataFrame(summary_rows)
    return survival_frame, summary_frame


def compute_completed_streak_lengths(correct):
    lengths = []
    run_len = 0
    for is_correct in np.asarray(correct, dtype=bool):
        if is_correct:
            run_len += 1
            continue
        if run_len > 0:
            lengths.append(int(run_len))
            run_len = 0
    if run_len > 0:
        lengths.append(int(run_len))
    return lengths


def summarize_completed_streaks(correct):
    lengths = compute_completed_streak_lengths(correct)
    if not lengths:
        return pd.DataFrame(
            [
                {
                    "count": 0,
                    "mean": float("nan"),
                    "median": float("nan"),
                    "p75": float("nan"),
                    "p90": float("nan"),
                    "p95": float("nan"),
                    "p99": float("nan"),
                    "max": 0,
                }
            ]
        )

    values = np.asarray(lengths, dtype=np.float64)
    return pd.DataFrame(
        [
            {
                "count": int(len(values)),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p75": float(np.quantile(values, 0.75)),
                "p90": float(np.quantile(values, 0.90)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
                "max": int(values.max()),
            }
        ]
    )


def collect_candidate_max_steps(weighted_summary, rolling_summary):
    values = set(int(value) for value in MANUAL_MAX_STEPS_GRID if int(value) > 0)
    for frame in (weighted_summary, rolling_summary):
        if frame.empty:
            continue
        for value in frame["chosen_max_steps"].dropna().astype(int).tolist():
            if value > 0:
                values.add(int(value))
    return sorted(values)


def draw_start_indices(day_indices, starts_per_day, seed):
    rng = np.random.default_rng(int(seed))
    chosen_parts = []
    for indices in day_indices:
        if len(indices) == 0:
            continue
        sample_size = min(int(starts_per_day), len(indices))
        chosen_parts.append(rng.choice(indices, size=sample_size, replace=False))
    if not chosen_parts:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(chosen_parts).astype(np.int64, copy=False))


def theoretical_pnl_units(max_steps):
    price_cap = float(FIXED_PRICE_CAP)
    if not (0.0 < price_cap < 1.0):
        raise ValueError(f"FIXED_PRICE_CAP must be in (0, 1). Got {FIXED_PRICE_CAP!r}.")
    return float((1.0 / price_cap) ** int(max_steps) - 1.0)


def max_drawdown_from_daily_pnl(daily_pnl_units):
    cumulative = np.cumsum(np.asarray(daily_pnl_units, dtype=np.float64))
    if len(cumulative) == 0:
        return 0.0
    running_max = np.maximum.accumulate(np.maximum(cumulative, 0.0))
    drawdown = running_max - cumulative
    return float(np.max(drawdown))


def simulate_one_seed(streaks, day_pos_by_row, day_count, start_indices, max_steps):
    n_rows = len(streaks)
    max_steps = int(max_steps)
    completed_pnl_units = theoretical_pnl_units(max_steps)
    next_available_idx = 0
    run_count = 0
    win_steps = 0
    loss_ended_runs = 0
    max_step_ended_runs = 0
    data_end_ended_runs = 0
    total_steps = 0
    daily_runs = np.zeros(day_count, dtype=np.int32)
    daily_pnl_units = np.zeros(day_count, dtype=np.float64)
    run_lengths = []

    for start_idx in start_indices:
        start_idx = int(start_idx)
        if start_idx < next_available_idx:
            continue
        if start_idx >= n_rows or start_idx + max_steps > n_rows:
            continue

        forward_wins = int(streaks[start_idx])
        if forward_wins >= max_steps:
            wins = max_steps
            run_length = max_steps
            pnl_units = completed_pnl_units
            ended_by_loss = False
            ended_by_max = True
            ended_by_data_end = False
        else:
            wins = forward_wins
            run_length = forward_wins + 1
            pnl_units = -1.0
            ended_by_loss = True
            ended_by_max = False
            ended_by_data_end = False

        if run_length <= 0:
            continue

        day_pos = int(day_pos_by_row[start_idx])
        run_count += 1
        win_steps += int(wins)
        loss_ended_runs += int(ended_by_loss)
        max_step_ended_runs += int(ended_by_max)
        data_end_ended_runs += int(ended_by_data_end)
        total_steps += int(run_length)
        daily_runs[day_pos] += 1
        daily_pnl_units[day_pos] += pnl_units
        run_lengths.append(int(run_length))
        next_available_idx = start_idx + run_length

    total_pnl_units = float(np.sum(daily_pnl_units))
    daily_median = float(np.median(daily_pnl_units)) if day_count else float("nan")
    daily_p05 = float(np.quantile(daily_pnl_units, 0.05)) if day_count else float("nan")
    worst_daily = float(np.min(daily_pnl_units)) if day_count else float("nan")
    avg_steps_per_run = float(total_steps / run_count) if run_count else 0.0
    length_counter = Counter(run_lengths)
    return {
        "run_count": int(run_count),
        "win_steps": int(win_steps),
        "loss_ended_runs": int(loss_ended_runs),
        "max_step_ended_runs": int(max_step_ended_runs),
        "data_end_ended_runs": int(data_end_ended_runs),
        "total_steps": int(total_steps),
        "total_pnl_units": total_pnl_units,
        "mean_daily_pnl_units": (
            float(np.mean(daily_pnl_units)) if day_count else float("nan")
        ),
        "median_daily_pnl_units": daily_median,
        "p05_daily_pnl_units": daily_p05,
        "worst_daily_pnl_units": worst_daily,
        "share_losing_days": (
            float(np.mean(daily_pnl_units < 0.0)) if day_count else float("nan")
        ),
        "max_drawdown_units": max_drawdown_from_daily_pnl(daily_pnl_units),
        "average_runs_per_day": float(run_count / day_count) if day_count else 0.0,
        "average_steps_per_run": avg_steps_per_run,
        "share_runs_reaching_max_steps": (
            float(max_step_ended_runs / run_count) if run_count else 0.0
        ),
        "run_length_distribution_json": json.dumps(
            {str(k): int(v) for k, v in sorted(length_counter.items())},
            sort_keys=True,
        ),
        "daily_runs": daily_runs,
        "daily_pnl_units": daily_pnl_units,
    }


def summarize_run_length_distribution(distributions):
    total = Counter()
    for raw in distributions:
        if not raw:
            continue
        parsed = json.loads(raw)
        total.update({int(key): int(value) for key, value in parsed.items()})
    run_total = sum(total.values())
    if run_total <= 0:
        return "{}"
    shares = {str(k): float(v / run_total) for k, v in sorted(total.items())}
    return json.dumps(shares, sort_keys=True)


def build_simulation_outputs(frame, streaks, candidate_max_steps):
    days = pd.Index(sorted(frame["event_day"].unique()))
    day_count = int(len(days))
    day_to_pos = {day: idx for idx, day in enumerate(days)}
    day_pos_by_row = frame["event_day"].map(day_to_pos).to_numpy(dtype=np.int32)
    day_indices = [
        frame.index[frame["event_day"] == day].to_numpy(dtype=np.int64)
        for day in days
    ]

    combo_count = len(STARTS_PER_DAY_GRID) * len(candidate_max_steps)
    estimated_by_day_rows = combo_count * int(N_RANDOM_SEEDS) * day_count
    keep_by_day = estimated_by_day_rows <= int(SAVE_BY_DAY_MAX_ROWS)

    by_seed_records = []
    by_day_records = []
    day_labels = [pd.Timestamp(day).date().isoformat() for day in days]

    start_index_cache = {}
    for starts_per_day in STARTS_PER_DAY_GRID:
        for seed in range(int(N_RANDOM_SEEDS)):
            start_index_cache[(int(starts_per_day), int(seed))] = draw_start_indices(
                day_indices,
                starts_per_day,
                seed,
            )

    for starts_per_day in STARTS_PER_DAY_GRID:
        for max_steps in candidate_max_steps:
            for seed in range(int(N_RANDOM_SEEDS)):
                metrics = simulate_one_seed(
                    streaks,
                    day_pos_by_row,
                    day_count,
                    start_index_cache[(int(starts_per_day), int(seed))],
                    max_steps,
                )
                record = {
                    "starts_per_day": int(starts_per_day),
                    "max_steps": int(max_steps),
                    "fixed_price_cap_used": float(FIXED_PRICE_CAP),
                    "theoretical_pnl_units": theoretical_pnl_units(max_steps),
                    "seed": int(seed),
                    "total_pnl_units": metrics["total_pnl_units"],
                    "mean_daily_pnl_units": metrics["mean_daily_pnl_units"],
                    "median_daily_pnl_units": metrics["median_daily_pnl_units"],
                    "p05_daily_pnl_units": metrics["p05_daily_pnl_units"],
                    "worst_daily_pnl_units": metrics["worst_daily_pnl_units"],
                    "share_losing_days": metrics["share_losing_days"],
                    "max_drawdown_units": metrics["max_drawdown_units"],
                    "run_count": metrics["run_count"],
                    "win_steps": metrics["win_steps"],
                    "loss_ended_runs": metrics["loss_ended_runs"],
                    "max_step_ended_runs": metrics["max_step_ended_runs"],
                    "data_end_ended_runs": metrics["data_end_ended_runs"],
                    "total_steps": metrics["total_steps"],
                    "average_runs_per_day": metrics["average_runs_per_day"],
                    "average_steps_per_run": metrics["average_steps_per_run"],
                    "share_runs_reaching_max_steps": metrics[
                        "share_runs_reaching_max_steps"
                    ],
                    "run_length_distribution_json": metrics[
                        "run_length_distribution_json"
                    ],
                }
                by_seed_records.append(record)

                if keep_by_day:
                    for day_pos, day_label in enumerate(day_labels):
                        by_day_records.append(
                            {
                                "starts_per_day": int(starts_per_day),
                                "max_steps": int(max_steps),
                                "seed": int(seed),
                                "day": day_label,
                                "run_count": int(metrics["daily_runs"][day_pos]),
                                "pnl_units": float(metrics["daily_pnl_units"][day_pos]),
                            }
                        )

    by_seed = pd.DataFrame(by_seed_records)
    by_day = pd.DataFrame(by_day_records)
    summary = aggregate_simulation_summary(by_seed)
    return summary, by_seed, by_day, {
        "day_count": day_count,
        "estimated_by_day_rows": int(estimated_by_day_rows),
        "by_day_saved": bool(keep_by_day),
        "decision_rows_only": bool(DECISION_ROWS_ONLY),
        "use_polymarket_prices": bool(USE_POLYMARKET_PRICES),
        "fixed_price_cap_used": float(FIXED_PRICE_CAP),
        "pnl_rule": (
            "loss before max_steps => -1.0; reaching max_steps => "
            "(1 / FIXED_PRICE_CAP) ** max_steps - 1.0"
        ),
    }


def aggregate_simulation_summary(by_seed):
    rows = []
    grouped = by_seed.groupby(["starts_per_day", "max_steps"], sort=True)
    for (starts_per_day, max_steps), group in grouped:
        total_pnl = group["total_pnl_units"].to_numpy(dtype=np.float64)
        run_count = float(group["run_count"].sum())
        max_step_ended = float(group["max_step_ended_runs"].sum())
        total_steps = float(group["total_steps"].sum())
        rows.append(
            {
                "starts_per_day": int(starts_per_day),
                "max_steps": int(max_steps),
                "fixed_price_cap_used": float(FIXED_PRICE_CAP),
                "theoretical_pnl_units": theoretical_pnl_units(max_steps),
                "seed_count": int(len(group)),
                "mean_total_pnl_units": float(np.mean(total_pnl)),
                "median_total_pnl_units": float(np.median(total_pnl)),
                "p05_total_pnl_units": float(np.quantile(total_pnl, 0.05)),
                "p95_total_pnl_units": float(np.quantile(total_pnl, 0.95)),
                "worst_seed_pnl_units": float(np.min(total_pnl)),
                "mean_daily_pnl_units": float(group["mean_daily_pnl_units"].mean()),
                "median_daily_pnl_units": float(group["median_daily_pnl_units"].median()),
                "p05_daily_pnl_units": float(group["p05_daily_pnl_units"].mean()),
                "worst_daily_pnl_units": float(group["worst_daily_pnl_units"].min()),
                "share_losing_days": float(group["share_losing_days"].mean()),
                "mean_max_drawdown_units": float(group["max_drawdown_units"].mean()),
                "worst_max_drawdown_units": float(group["max_drawdown_units"].max()),
                "average_runs_per_day": float(group["average_runs_per_day"].mean()),
                "average_steps_per_run": (
                    float(total_steps / run_count) if run_count > 0.0 else 0.0
                ),
                "share_runs_reaching_max_steps": (
                    float(max_step_ended / run_count) if run_count > 0.0 else 0.0
                ),
                "mean_run_count": float(group["run_count"].mean()),
                "mean_win_steps": float(group["win_steps"].mean()),
                "mean_loss_ended_runs": float(group["loss_ended_runs"].mean()),
                "mean_max_step_ended_runs": float(group["max_step_ended_runs"].mean()),
                "mean_data_end_ended_runs": float(group["data_end_ended_runs"].mean()),
                "run_length_distribution_share_json": summarize_run_length_distribution(
                    group["run_length_distribution_json"].tolist()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    best_mean = float(summary["mean_total_pnl_units"].max())
    if best_mean > 0.0:
        result_fraction = summary["mean_total_pnl_units"] / best_mean
    else:
        result_fraction = np.zeros(len(summary), dtype=np.float64)
    summary["best_pnl_fraction"] = result_fraction
    summary["risk_reward_score"] = (
        summary["mean_total_pnl_units"]
        - 0.25 * summary["mean_max_drawdown_units"]
        - 20.0 * summary["share_losing_days"]
    )
    return summary.sort_values(
        ["risk_reward_score", "mean_total_pnl_units"],
        ascending=[False, False],
    ).reset_index(drop=True)


def lower_median(values):
    values = sorted(int(value) for value in values if int(value) > 0)
    if not values:
        return 1
    return int(values[(len(values) - 1) // 2])


def rounded_median(values):
    values = [int(value) for value in values if int(value) > 0]
    if not values:
        return 1
    return int(round(float(np.median(values))))


def recommend_max_steps(weighted_summary, rolling_summary):
    weighted_recent = weighted_summary.loc[
        weighted_summary["half_life_days"].isin([14, 30, 60]),
        "chosen_max_steps",
    ].tolist()
    rolling_recent = rolling_summary.loc[
        rolling_summary["window_days"].isin([14, 30, 60]),
        "chosen_max_steps",
    ].tolist()
    recent_values = [int(value) for value in [*weighted_recent, *rolling_recent] if int(value) > 0]
    conservative = lower_median(recent_values)
    balanced = rounded_median(recent_values)

    recent_roll = [int(value) for value in rolling_recent if int(value) > 0]
    if recent_roll and min(recent_roll) <= 1:
        conservative = min(conservative, 1)
    if recent_values and max(recent_values) - min(recent_values) >= 4:
        conservative = min(conservative, lower_median(recent_values[:]))
        balanced = min(balanced, max(1, int(np.quantile(recent_values, 0.75))))
    return int(max(1, conservative)), int(max(1, balanced))


def recommend_starts_per_day(sim_summary, max_steps, *, target_fraction):
    if sim_summary.empty:
        return 1
    subset = sim_summary.loc[sim_summary["max_steps"] == int(max_steps)].copy()
    if subset.empty:
        subset = sim_summary.copy()
    best_mean = float(subset["mean_total_pnl_units"].max())
    if best_mean <= 0.0:
        return int(subset.sort_values(["starts_per_day"]).iloc[0]["starts_per_day"])

    viable = subset.loc[
        subset["mean_total_pnl_units"] >= best_mean * float(target_fraction)
    ].copy()
    if viable.empty:
        viable = subset.copy()
    viable = viable.sort_values(
        [
            "starts_per_day",
            "mean_max_drawdown_units",
            "share_losing_days",
            "mean_total_pnl_units",
        ],
        ascending=[True, True, True, False],
    )
    return int(viable.iloc[0]["starts_per_day"])


def build_recommendations(weighted_summary, rolling_summary, sim_summary):
    max_steps_conservative, max_steps_balanced = recommend_max_steps(
        weighted_summary,
        rolling_summary,
    )
    starts_conservative = recommend_starts_per_day(
        sim_summary,
        max_steps_conservative,
        target_fraction=0.80,
    )
    starts_balanced = recommend_starts_per_day(
        sim_summary,
        max_steps_balanced,
        target_fraction=0.90,
    )
    return {
        "fixed_price_cap_used": float(FIXED_PRICE_CAP),
        "recommended_max_steps_conservative": int(max_steps_conservative),
        "recommended_max_steps_balanced": int(max_steps_balanced),
        "recommended_starts_per_day_conservative": int(starts_conservative),
        "recommended_starts_per_day_balanced": int(starts_balanced),
        "price_cap_note": "Price cap is a fixed user assumption, not an optimized output.",
    }


def dataframe_records(frame):
    return json.loads(frame.to_json(orient="records"))


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def write_outputs(
    *,
    output_dir,
    source_path,
    basic_stats,
    weighted_survival,
    weighted_summary,
    rolling_survival,
    rolling_summary,
    completed_summary,
    sim_summary,
    sim_by_seed,
    sim_by_day,
    simulation_meta,
    recommendations,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary_json": output_dir / SUMMARY_JSON,
        "weighted_survival_csv": output_dir / WEIGHTED_SURVIVAL_CSV,
        "rolling_survival_csv": output_dir / ROLLING_SURVIVAL_CSV,
        "completed_streak_csv": output_dir / COMPLETED_STREAK_CSV,
        "simulation_summary_csv": output_dir / SIMULATION_SUMMARY_CSV,
        "simulation_by_seed_csv": output_dir / SIMULATION_BY_SEED_CSV,
    }
    if simulation_meta["by_day_saved"]:
        paths["simulation_by_day_csv"] = output_dir / SIMULATION_BY_DAY_CSV

    weighted_survival.to_csv(paths["weighted_survival_csv"], index=False)
    rolling_survival.to_csv(paths["rolling_survival_csv"], index=False)
    completed_summary.to_csv(paths["completed_streak_csv"], index=False)
    sim_summary.to_csv(paths["simulation_summary_csv"], index=False)
    sim_by_seed.to_csv(paths["simulation_by_seed_csv"], index=False)
    if simulation_meta["by_day_saved"]:
        sim_by_day.to_csv(paths["simulation_by_day_csv"], index=False)

    candidate_max_steps = sorted(sim_summary["max_steps"].unique().astype(int).tolist())
    theoretical_pnl_by_max_steps = [
        {
            "max_steps": int(max_steps),
            "fixed_price_cap_used": float(FIXED_PRICE_CAP),
            "theoretical_pnl_units": theoretical_pnl_units(max_steps),
        }
        for max_steps in candidate_max_steps
    ]
    summary = {
        "source": {
            "oof_path": str(source_path),
            "time_col": TIME_COL,
            "pred_col": PRED_COL,
            "target_col": TARGET_COL,
            "prediction_source": "oof",
            "target_source": "oof",
            "prediction_threshold": float(PREDICTION_THRESHOLD),
        },
        "config": {
            "decision_rows_only": bool(DECISION_ROWS_ONLY),
            "use_polymarket_prices": bool(USE_POLYMARKET_PRICES),
            "fixed_price_cap": float(FIXED_PRICE_CAP),
            "min_survival_prob": float(MIN_SURVIVAL_PROB),
            "half_life_days_grid": list(HALF_LIFE_DAYS_GRID),
            "rolling_windows_days": list(ROLLING_WINDOWS_DAYS),
            "starts_per_day_grid": list(STARTS_PER_DAY_GRID),
            "manual_max_steps_grid": list(MANUAL_MAX_STEPS_GRID),
            "n_random_seeds": int(N_RANDOM_SEEDS),
        },
        "basic_stats": basic_stats,
        "all_oof_rows_loaded": basic_stats["all_oof_rows_loaded"],
        "decision_rows_count": basic_stats["decision_rows_count"],
        "decision_row_accuracy": basic_stats["decision_row_accuracy"],
        "forward_streak_survival_scope": "decision_rows",
        "weighted_half_life_summary": dataframe_records(weighted_summary),
        "rolling_window_summary": dataframe_records(rolling_summary),
        "completed_streak_scope": "decision_rows",
        "completed_streak_summary": dataframe_records(completed_summary)[0],
        "simulation": {
            "meta": simulation_meta,
            "candidate_max_steps": candidate_max_steps,
            "theoretical_pnl_by_max_steps": theoretical_pnl_by_max_steps,
            "summary_scope": "starts_per_day_and_max_steps",
            "best_variants_by_risk_reward": dataframe_records(
                sim_summary.head(10)
            ),
        },
        "recommendations": recommendations,
        "artifacts": {key: str(path) for key, path in paths.items()},
        "warning": OOF_ONLY_WARNING,
        "notes": [
            OOF_ONLY_WARNING,
            "Offline research only. Forward streaks use known OOF targets and are not live-available signals.",
            "Daily simulation PnL is assigned to the run start day.",
            "Price cap is a fixed user assumption, not an optimized script output.",
        ],
    }
    paths["summary_json"].write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    return paths


def print_console_summary(
    *,
    basic_stats,
    weighted_summary,
    rolling_summary,
    sim_summary,
    recommendations,
    paths,
):
    print(
        "lucky run OOF analysis | "
        f"all_oof_rows={basic_stats['all_oof_rows_loaded']} "
        f"decision_rows={basic_stats['decision_rows_count']} "
        f"range={basic_stats['start_time']}..{basic_stats['end_time']} "
        f"decision_accuracy={basic_stats['decision_row_accuracy']:.4f} "
        f"fixed_price_cap={FIXED_PRICE_CAP:.2f}"
    )
    print("warning | " + OOF_ONLY_WARNING)
    print(
        "recommendations | "
        f"max_steps_conservative={recommendations['recommended_max_steps_conservative']} "
        f"max_steps_balanced={recommendations['recommended_max_steps_balanced']} "
        f"starts_per_day_conservative={recommendations['recommended_starts_per_day_conservative']} "
        f"starts_per_day_balanced={recommendations['recommended_starts_per_day_balanced']}"
    )
    print("\nhalf-life sensitivity")
    print(
        weighted_summary.loc[
            :,
            [
                "half_life_days",
                "effective_sample_size",
                "weighted_accuracy",
                "chosen_max_steps",
            ],
        ].to_string(index=False)
    )
    print("\nrolling-window sensitivity")
    print(
        rolling_summary.loc[
            :,
            ["window_days", "row_count", "day_count", "accuracy", "chosen_max_steps"],
        ].to_string(index=False)
    )
    print("\ntop simulation variants")
    top_cols = [
        "starts_per_day",
        "max_steps",
        "fixed_price_cap_used",
        "theoretical_pnl_units",
        "mean_total_pnl_units",
        "p05_total_pnl_units",
        "mean_max_drawdown_units",
        "share_losing_days",
        "average_runs_per_day",
        "share_runs_reaching_max_steps",
    ]
    print(sim_summary.loc[:, top_cols].head(8).to_string(index=False))
    print("\nartifacts")
    for label, path in paths.items():
        print(f"{label}: {path}")


def main():
    source_path = resolve_oof_path()
    frame = load_analysis_frame(source_path)
    basic_stats = summarize_basic(frame)
    streaks = compute_forward_streak_lengths(frame["correct"].to_numpy(dtype=bool))

    weighted_survival, weighted_summary = build_weighted_survival_outputs(frame, streaks)
    rolling_survival, rolling_summary = build_rolling_survival_outputs(frame, streaks)
    completed_summary = summarize_completed_streaks(frame["correct"].to_numpy(dtype=bool))

    candidate_max_steps = collect_candidate_max_steps(weighted_summary, rolling_summary)
    sim_summary, sim_by_seed, sim_by_day, simulation_meta = build_simulation_outputs(
        frame,
        streaks,
        candidate_max_steps,
    )
    recommendations = build_recommendations(
        weighted_summary,
        rolling_summary,
        sim_summary,
    )

    paths = write_outputs(
        output_dir=OUTPUT_DIR,
        source_path=source_path,
        basic_stats=basic_stats,
        weighted_survival=weighted_survival,
        weighted_summary=weighted_summary,
        rolling_survival=rolling_survival,
        rolling_summary=rolling_summary,
        completed_summary=completed_summary,
        sim_summary=sim_summary,
        sim_by_seed=sim_by_seed,
        sim_by_day=sim_by_day,
        simulation_meta=simulation_meta,
        recommendations=recommendations,
    )
    print_console_summary(
        basic_stats=basic_stats,
        weighted_summary=weighted_summary,
        rolling_summary=rolling_summary,
        sim_summary=sim_summary,
        recommendations=recommendations,
        paths=paths,
    )


if __name__ == "__main__":
    main()
