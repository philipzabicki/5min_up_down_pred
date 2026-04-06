import glob
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from live_utils import parse_live_trade_records_path

DEFAULT_TRADE_CSV_GLOB = "data/live/trade/*.csv"
DEFAULT_SHARED_CSV_PATH = "data/live/polymarket_5m.csv"
LIVE_MARKET_REQUIRED_COLUMNS = (
    "proba_up",
    "pm_up_best_ask",
    "pm_down_best_ask",
    "actual_up",
)
LIVE_MARKET_OPTIONAL_COLUMNS = (
    "record_id",
    "pm_model_hash",
    "pm_run_started_at_utc",
    "pm_tick_size",
    "pm_order_min_size",
    "decision_delay_ms",
    "market_lookup_ms",
    "submit_order_ms",
    "execution_ms",
)
LIVE_MARKET_OPTIONAL_TIMESTAMP_COLUMNS = (
    "prediction_time",
    "resolved_at",
    "bucket_start",
    "bucket_end",
)
LIVE_MARKET_KEY_COLUMNS = (
    "pm_run_started_at_utc",
    "pm_model_hash",
    "record_id",
    "prediction_time",
    "bucket_start",
    "bucket_end",
)


def _coalesce_market_elapsed_ms(frame):
    if "market_elapsed_ms" in frame.columns:
        return pd.to_numeric(frame["market_elapsed_ms"], errors="coerce")
    if "decision_delay_ms" in frame.columns:
        elapsed = pd.to_numeric(frame["decision_delay_ms"], errors="coerce")
    else:
        elapsed = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    if "prediction_delay_ms" in frame.columns:
        prediction_delay = pd.to_numeric(frame["prediction_delay_ms"], errors="coerce")
        elapsed = elapsed.where(np.isfinite(elapsed), prediction_delay)
    return elapsed.astype(np.float64, copy=False)


def _rank01(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rank01 expects a 1D array.")
    n_rows = len(values)
    if n_rows == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n_rows, dtype=np.float64)
    ranks[order] = (np.arange(n_rows, dtype=np.float64) + 0.5) / float(n_rows)
    return ranks


def _normalize_model_hash(value):
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _read_live_market_csv(path):
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(set(LIVE_MARKET_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required live market columns: {missing}")

    keep_columns = list(LIVE_MARKET_REQUIRED_COLUMNS)
    keep_columns.extend(
        [col for col in LIVE_MARKET_OPTIONAL_COLUMNS if col in frame.columns]
    )
    keep_columns.extend(
        [col for col in LIVE_MARKET_OPTIONAL_TIMESTAMP_COLUMNS if col in frame.columns]
    )
    out = frame.loc[:, keep_columns].copy()
    parsed_path_meta = parse_live_trade_records_path(path)
    if parsed_path_meta is not None:
        if "pm_model_hash" not in out.columns:
            out["pm_model_hash"] = parsed_path_meta["model_hash"]
        else:
            out["pm_model_hash"] = out["pm_model_hash"].fillna(
                parsed_path_meta["model_hash"]
            )
        if "pm_run_started_at_utc" not in out.columns:
            out["pm_run_started_at_utc"] = parsed_path_meta["run_started_at_utc"]
        else:
            out["pm_run_started_at_utc"] = out["pm_run_started_at_utc"].fillna(
                parsed_path_meta["run_started_at_utc"]
            )
    out["source_path"] = str(path)
    return out


def load_live_market_empirical_frame(
    *,
    trade_csv_glob=DEFAULT_TRADE_CSV_GLOB,
    shared_csv_path=DEFAULT_SHARED_CSV_PATH,
    recent_resolved_rows=None,
    preferred_model_hash=None,
    max_prediction_delay_ms=None,
    max_decision_delay_ms=None,
    max_market_lookup_ms=None,
    max_submit_order_ms=None,
    max_execution_ms=None,
):
    preferred_model_hash = _normalize_model_hash(preferred_model_hash)
    frames = []

    if trade_csv_glob is not None:
        for raw_path in sorted(glob.glob(str(trade_csv_glob))):
            frames.append(_read_live_market_csv(Path(raw_path)))

    if shared_csv_path not in (None, ""):
        shared_path = Path(shared_csv_path)
        if shared_path.exists():
            frames.append(_read_live_market_csv(shared_path))

    if not frames:
        raise ValueError(
            "No live market CSV sources were found for market-price calibration."
        )

    frame = pd.concat(frames, ignore_index=True, sort=False)

    for col in ("proba_up", "pm_up_best_ask", "pm_down_best_ask", "actual_up"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "pm_order_min_size" in frame.columns:
        frame["pm_order_min_size"] = pd.to_numeric(
            frame["pm_order_min_size"],
            errors="coerce",
        )
    else:
        frame["pm_order_min_size"] = np.nan
    for col in (
        "decision_delay_ms",
        "market_lookup_ms",
        "submit_order_ms",
        "execution_ms",
    ):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in LIVE_MARKET_OPTIONAL_TIMESTAMP_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce", utc=True)

    if "prediction_time" in frame.columns and "bucket_start" in frame.columns:
        frame["prediction_delay_ms"] = (
            frame["prediction_time"] - frame["bucket_start"]
        ).dt.total_seconds() * 1000.0
    frame["market_elapsed_ms"] = _coalesce_market_elapsed_ms(frame)

    for col, limit in (
        ("prediction_delay_ms", max_prediction_delay_ms),
        ("decision_delay_ms", max_decision_delay_ms),
        ("market_lookup_ms", max_market_lookup_ms),
        ("submit_order_ms", max_submit_order_ms),
        ("execution_ms", max_execution_ms),
    ):
        if limit is None:
            continue
        limit = float(limit)
        if col not in frame.columns:
            frame[col] = np.nan
        frame = frame[np.isfinite(frame[col]) & (frame[col] <= limit)].copy()

    if preferred_model_hash is not None:
        if "pm_model_hash" not in frame.columns:
            raise ValueError(
                "preferred_model_hash was provided but live market rows do not contain "
                "pm_model_hash."
            )
        model_hash_series = frame["pm_model_hash"].fillna("").astype(str).str.strip()
        frame = frame.loc[model_hash_series == preferred_model_hash].copy()
        if frame.empty:
            raise ValueError(
                "No resolved live market rows matched preferred_model_hash="
                f"{preferred_model_hash!r}."
            )

    frame = frame[frame["actual_up"].isin([0.0, 1.0])].copy()
    frame["actual_up"] = frame["actual_up"].astype(np.int8)
    frame = frame.dropna(subset=["proba_up", "pm_up_best_ask", "pm_down_best_ask"])
    frame = frame[
        np.isfinite(frame["pm_up_best_ask"])
        & np.isfinite(frame["pm_down_best_ask"])
        & np.isfinite(frame["proba_up"])
        & (frame["pm_up_best_ask"] > 0.0)
        & (frame["pm_up_best_ask"] < 1.0)
        & (frame["pm_down_best_ask"] > 0.0)
        & (frame["pm_down_best_ask"] < 1.0)
    ].copy()

    sort_columns = [
        col
        for col in ("prediction_time", "resolved_at", "bucket_start", "record_id")
        if col in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(
            by=sort_columns,
            kind="stable",
            na_position="last",
        )

    preferred_dedupe_columns = [
        col
        for col in ("pm_run_started_at_utc", "pm_model_hash", "record_id")
        if col in frame.columns
    ]
    if "record_id" in preferred_dedupe_columns:
        dedupe_columns = preferred_dedupe_columns
    else:
        dedupe_columns = preferred_dedupe_columns + [
            col
            for col in ("prediction_time", "bucket_start", "bucket_end")
            if col in frame.columns
        ]
    if dedupe_columns:
        frame = frame.drop_duplicates(subset=dedupe_columns, keep="last")

    if recent_resolved_rows is not None:
        recent_resolved_rows = int(recent_resolved_rows)
        if recent_resolved_rows <= 0:
            raise ValueError("recent_resolved_rows must be > 0 when provided.")
        frame = frame.tail(recent_resolved_rows)

    frame = frame.reset_index(drop=True)
    frame["model_conf"] = (frame["proba_up"] - 0.5).abs()
    frame["winner_ask"] = frame[["pm_up_best_ask", "pm_down_best_ask"]].max(axis=1)
    frame["loser_ask"] = frame[["pm_up_best_ask", "pm_down_best_ask"]].min(axis=1)
    frame["actual_winner_ask"] = np.where(
        frame["actual_up"].to_numpy(dtype=np.int8, copy=False) == 1,
        frame["pm_up_best_ask"],
        frame["pm_down_best_ask"],
    )
    frame["actual_loser_ask"] = np.where(
        frame["actual_up"].to_numpy(dtype=np.int8, copy=False) == 1,
        frame["pm_down_best_ask"],
        frame["pm_up_best_ask"],
    )
    frame["market_pick_up"] = np.where(
        frame["pm_up_best_ask"] > frame["pm_down_best_ask"],
        1.0,
        np.where(frame["pm_down_best_ask"] > frame["pm_up_best_ask"], 0.0, np.nan),
    )
    frame["is_tie"] = ~np.isfinite(frame["market_pick_up"])
    frame["market_direction_correct"] = np.where(
        frame["is_tie"],
        np.nan,
        frame["market_pick_up"] == frame["actual_up"].astype(np.float64, copy=False),
    )
    frame["abs_gap"] = (frame["pm_up_best_ask"] - frame["pm_down_best_ask"]).abs()
    frame["overround"] = frame["pm_up_best_ask"] + frame["pm_down_best_ask"] - 1.0
    frame["up_residual"] = frame["pm_up_best_ask"] - frame["proba_up"]
    frame["down_residual"] = frame["pm_down_best_ask"] - (1.0 - frame["proba_up"])

    if frame.empty:
        raise ValueError("No resolved live rows with valid asks remained after cleanup.")

    return frame


@lru_cache(maxsize=8)
def load_elapsed_target_live_market_calibration(
    trade_csv_glob=DEFAULT_TRADE_CSV_GLOB,
    shared_csv_path=DEFAULT_SHARED_CSV_PATH,
    recent_resolved_rows=None,
    elapsed_quantile_bins=12,
    min_pool_rows=250,
    preferred_model_hash=None,
    max_prediction_delay_ms=None,
    max_decision_delay_ms=None,
    max_market_lookup_ms=None,
    max_submit_order_ms=None,
    max_execution_ms=None,
):
    frame = load_live_market_empirical_frame(
        trade_csv_glob=trade_csv_glob,
        shared_csv_path=shared_csv_path,
        recent_resolved_rows=recent_resolved_rows,
        preferred_model_hash=preferred_model_hash,
        max_prediction_delay_ms=max_prediction_delay_ms,
        max_decision_delay_ms=max_decision_delay_ms,
        max_market_lookup_ms=max_market_lookup_ms,
        max_submit_order_ms=max_submit_order_ms,
        max_execution_ms=max_execution_ms,
    )
    frame = frame[np.isfinite(frame["market_elapsed_ms"])].copy()
    frame = frame[frame["market_elapsed_ms"] >= 0.0].copy()

    elapsed_quantile_bins = int(elapsed_quantile_bins)
    min_pool_rows = int(min_pool_rows)
    if elapsed_quantile_bins < 2:
        raise ValueError("elapsed_quantile_bins must be >= 2.")
    if min_pool_rows <= 0:
        raise ValueError("min_pool_rows must be > 0.")
    if len(frame) < min_pool_rows:
        raise ValueError(
            "Elapsed-target market calibration pool is too small. "
            f"rows={len(frame)} required>={min_pool_rows}"
        )

    elapsed_ms = frame["market_elapsed_ms"].to_numpy(dtype=np.float64, copy=False)
    elapsed_edges = np.quantile(
        elapsed_ms,
        np.linspace(0.0, 1.0, elapsed_quantile_bins + 1),
    ).astype(np.float64, copy=False)
    elapsed_edges = np.maximum.accumulate(elapsed_edges)
    elapsed_bins = np.searchsorted(
        elapsed_edges[1:-1],
        elapsed_ms,
        side="right",
    ).astype(np.int64, copy=False)

    all_indices = np.arange(len(frame), dtype=np.int64)
    bin_indices = tuple(
        np.flatnonzero(elapsed_bins == bin_id).astype(np.int64, copy=False)
        for bin_id in range(elapsed_quantile_bins)
    )
    order_min_size = frame["pm_order_min_size"].to_numpy(dtype=np.float64, copy=False)
    valid_order_min_size = order_min_size[np.isfinite(order_min_size) & (order_min_size > 0.0)]
    fallback_order_min_size = (
        float(np.median(valid_order_min_size))
        if len(valid_order_min_size) > 0
        else float("nan")
    )

    return {
        "elapsed_edges": elapsed_edges,
        "bin_indices": bin_indices,
        "all_indices": all_indices,
        "actual_winner_ask": frame["actual_winner_ask"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),
        "actual_loser_ask": frame["actual_loser_ask"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),
        "order_min_size": order_min_size.astype(np.float64, copy=False),
        "fallback_order_min_size": float(fallback_order_min_size),
        "row_count": int(len(frame)),
    }


@lru_cache(maxsize=8)
def load_empirical_residual_live_market_calibration(
    trade_csv_glob=DEFAULT_TRADE_CSV_GLOB,
    shared_csv_path=DEFAULT_SHARED_CSV_PATH,
    recent_resolved_rows=None,
    confidence_quantile_bins=10,
    min_pool_rows=250,
    preferred_model_hash=None,
    max_prediction_delay_ms=None,
    max_decision_delay_ms=None,
    max_market_lookup_ms=None,
    max_submit_order_ms=None,
    max_execution_ms=None,
):
    confidence_quantile_bins = int(confidence_quantile_bins)
    min_pool_rows = int(min_pool_rows)
    preferred_model_hash = _normalize_model_hash(preferred_model_hash)

    if confidence_quantile_bins < 1:
        raise ValueError("confidence_quantile_bins must be >= 1.")
    if min_pool_rows <= 0:
        raise ValueError("min_pool_rows must be > 0.")

    fallback_reason = None
    selected_model_hash = preferred_model_hash
    preferred_frame = None
    if preferred_model_hash is not None:
        try:
            preferred_frame = load_live_market_empirical_frame(
                trade_csv_glob=trade_csv_glob,
                shared_csv_path=shared_csv_path,
                recent_resolved_rows=recent_resolved_rows,
                preferred_model_hash=preferred_model_hash,
                max_prediction_delay_ms=max_prediction_delay_ms,
                max_decision_delay_ms=max_decision_delay_ms,
                max_market_lookup_ms=max_market_lookup_ms,
                max_submit_order_ms=max_submit_order_ms,
                max_execution_ms=max_execution_ms,
            )
        except ValueError as exc:
            if "No resolved live market rows matched preferred_model_hash=" not in str(exc):
                raise
            fallback_reason = "no_matching_rows"
        else:
            if len(preferred_frame) >= min_pool_rows:
                frame = preferred_frame
            else:
                fallback_reason = f"insufficient_rows:{len(preferred_frame)}"
    if preferred_frame is None or len(preferred_frame) < min_pool_rows:
        frame = load_live_market_empirical_frame(
            trade_csv_glob=trade_csv_glob,
            shared_csv_path=shared_csv_path,
            recent_resolved_rows=recent_resolved_rows,
            max_prediction_delay_ms=max_prediction_delay_ms,
            max_decision_delay_ms=max_decision_delay_ms,
            max_market_lookup_ms=max_market_lookup_ms,
            max_submit_order_ms=max_submit_order_ms,
            max_execution_ms=max_execution_ms,
        )
        selected_model_hash = None
        if preferred_model_hash is not None:
            print(
                "market sim calibration fallback | "
                f"preferred_model_hash={preferred_model_hash} "
                f"reason={fallback_reason or 'unknown'} "
                f"fallback_rows={len(frame)}"
            )

    if len(frame) < min_pool_rows:
        raise ValueError(
            "Empirical residual market calibration pool is too small. "
            f"rows={len(frame)} required>={min_pool_rows}"
        )

    confidence = frame["model_conf"].to_numpy(dtype=np.float64, copy=False)
    quantile_levels = np.linspace(0.0, 1.0, confidence_quantile_bins + 1)
    confidence_edges = np.quantile(confidence, quantile_levels).astype(
        np.float64,
        copy=False,
    )
    confidence_edges = np.maximum.accumulate(confidence_edges)

    if confidence_quantile_bins == 1:
        calibration_bins = np.zeros(len(frame), dtype=np.int64)
    else:
        calibration_bins = np.searchsorted(
            confidence_edges[1:-1],
            confidence,
            side="right",
        ).astype(np.int64, copy=False)

    all_indices = np.arange(len(frame), dtype=np.int64)
    bin_indices = tuple(
        np.flatnonzero(calibration_bins == bin_id).astype(np.int64, copy=False)
        for bin_id in range(confidence_quantile_bins)
    )

    order_min_size = frame["pm_order_min_size"].to_numpy(dtype=np.float64, copy=False)
    valid_order_min_size = order_min_size[np.isfinite(order_min_size) & (order_min_size > 0.0)]
    fallback_order_min_size = (
        float(np.median(valid_order_min_size))
        if len(valid_order_min_size) > 0
        else float("nan")
    )

    return {
        "confidence_edges": confidence_edges,
        "bin_indices": bin_indices,
        "all_indices": all_indices,
        "up_residual": frame["up_residual"].to_numpy(dtype=np.float64, copy=False),
        "down_residual": frame["down_residual"].to_numpy(dtype=np.float64, copy=False),
        "order_min_size": order_min_size.astype(np.float64, copy=False),
        "fallback_order_min_size": float(fallback_order_min_size),
        "row_count": int(len(frame)),
        "preferred_model_hash": preferred_model_hash,
        "selected_model_hash": selected_model_hash,
        "fallback_reason": fallback_reason,
    }


@lru_cache(maxsize=8)
def load_constructive_live_market_calibration(
    trade_csv_glob=DEFAULT_TRADE_CSV_GLOB,
    shared_csv_path=DEFAULT_SHARED_CSV_PATH,
    recent_resolved_rows=None,
    confidence_quantile_bins=10,
    min_pool_rows=250,
    smoothing_passes=1,
    max_prediction_delay_ms=None,
    max_decision_delay_ms=None,
    max_market_lookup_ms=None,
    max_submit_order_ms=None,
    max_execution_ms=None,
):
    frame = load_live_market_empirical_frame(
        trade_csv_glob=trade_csv_glob,
        shared_csv_path=shared_csv_path,
        recent_resolved_rows=recent_resolved_rows,
        max_prediction_delay_ms=max_prediction_delay_ms,
        max_decision_delay_ms=max_decision_delay_ms,
        max_market_lookup_ms=max_market_lookup_ms,
        max_submit_order_ms=max_submit_order_ms,
        max_execution_ms=max_execution_ms,
    )
    confidence_quantile_bins = int(confidence_quantile_bins)
    min_pool_rows = int(min_pool_rows)
    smoothing_passes = int(smoothing_passes)

    if confidence_quantile_bins < 2:
        raise ValueError("confidence_quantile_bins must be >= 2.")
    if min_pool_rows <= 0:
        raise ValueError("min_pool_rows must be > 0.")
    if smoothing_passes < 0:
        raise ValueError("smoothing_passes must be >= 0.")
    if len(frame) < min_pool_rows:
        raise ValueError(
            "Constructive market calibration pool is too small. "
            f"rows={len(frame)} required>={min_pool_rows}"
        )

    conf_rank = _rank01(frame["model_conf"].to_numpy(dtype=np.float64, copy=False))
    calibration_bins = np.minimum(
        (conf_rank * confidence_quantile_bins).astype(np.int64),
        confidence_quantile_bins - 1,
    )
    calibration_frame = frame.copy()
    calibration_frame["confidence_bin"] = calibration_bins
    grouped = calibration_frame.groupby("confidence_bin", sort=True)

    def _reindex_stat(series_like, *, fallback):
        series = pd.Series(series_like, dtype=np.float64).reindex(
            range(confidence_quantile_bins)
        )
        series = series.interpolate(limit_direction="both")
        values = series.to_numpy(dtype=np.float64, copy=False)
        return np.where(np.isfinite(values), values, float(fallback))

    def _smooth_curve(values):
        smoothed = np.asarray(values, dtype=np.float64)
        for _ in range(smoothing_passes):
            if smoothed.size <= 1:
                break
            out = smoothed.copy()
            out[0] = 0.75 * smoothed[0] + 0.25 * smoothed[1]
            out[-1] = 0.25 * smoothed[-2] + 0.75 * smoothed[-1]
            if smoothed.size > 2:
                out[1:-1] = (
                    0.25 * smoothed[:-2]
                    + 0.50 * smoothed[1:-1]
                    + 0.25 * smoothed[2:]
                )
            smoothed = out
        return smoothed

    def _group_std(column):
        return grouped[column].agg(
            lambda values: float(
                np.std(np.asarray(values, dtype=np.float64), ddof=0)
            )
        )

    global_abs_gap_mean = float(calibration_frame["abs_gap"].mean())
    global_abs_gap_std = float(
        np.std(calibration_frame["abs_gap"].to_numpy(dtype=np.float64, copy=False), ddof=0)
    )
    global_overround_mean = float(calibration_frame["overround"].mean())
    global_overround_std = float(
        np.std(
            calibration_frame["overround"].to_numpy(dtype=np.float64, copy=False),
            ddof=0,
        )
    )
    global_tie_rate = float(calibration_frame["is_tie"].mean())
    global_p_correct = float(
        calibration_frame["market_direction_correct"].dropna().mean()
    )

    abs_gap_mean_curve = _smooth_curve(
        _reindex_stat(grouped["abs_gap"].mean(), fallback=global_abs_gap_mean)
    )
    abs_gap_std_curve = np.maximum(
        _smooth_curve(
            _reindex_stat(
                _group_std("abs_gap"),
                fallback=global_abs_gap_std,
            )
        ),
        0.0,
    )
    overround_mean_curve = np.maximum(
        _smooth_curve(
            _reindex_stat(
                grouped["overround"].mean(),
                fallback=global_overround_mean,
            )
        ),
        0.0,
    )
    overround_std_curve = np.maximum(
        _smooth_curve(
            _reindex_stat(
                _group_std("overround"),
                fallback=global_overround_std,
            )
        ),
        0.0,
    )
    tie_rate_curve = np.clip(
        _smooth_curve(
            _reindex_stat(grouped["is_tie"].mean(), fallback=global_tie_rate)
        ),
        0.0,
        1.0,
    )
    p_correct_curve = np.clip(
        _smooth_curve(
            _reindex_stat(
                grouped["market_direction_correct"].mean(),
                fallback=global_p_correct,
            )
        ),
        0.0,
        1.0,
    )

    gap_residual = (
        calibration_frame["abs_gap"].to_numpy(dtype=np.float64, copy=False)
        - abs_gap_mean_curve[calibration_bins]
    )
    overround_residual = (
        calibration_frame["overround"].to_numpy(dtype=np.float64, copy=False)
        - overround_mean_curve[calibration_bins]
    )
    residual_mask = np.isfinite(gap_residual) & np.isfinite(overround_residual)
    if np.any(residual_mask):
        gap_overround_correlation = float(
            np.corrcoef(gap_residual[residual_mask], overround_residual[residual_mask])[0, 1]
        )
    else:
        gap_overround_correlation = 0.0
    if not np.isfinite(gap_overround_correlation):
        gap_overround_correlation = 0.0
    gap_overround_correlation = float(
        np.clip(gap_overround_correlation, -0.95, 0.95)
    )

    return {
        "bin_centers": (
            (np.arange(confidence_quantile_bins, dtype=np.float64) + 0.5)
            / float(confidence_quantile_bins)
        ),
        "abs_gap_mean_curve": abs_gap_mean_curve.astype(np.float64, copy=False),
        "abs_gap_std_curve": abs_gap_std_curve.astype(np.float64, copy=False),
        "overround_mean_curve": overround_mean_curve.astype(np.float64, copy=False),
        "overround_std_curve": overround_std_curve.astype(np.float64, copy=False),
        "tie_rate_curve": tie_rate_curve.astype(np.float64, copy=False),
        "p_correct_curve": p_correct_curve.astype(np.float64, copy=False),
        "gap_overround_correlation": gap_overround_correlation,
        "confidence_quantile_bins": int(confidence_quantile_bins),
        "row_count": int(len(calibration_frame)),
    }


def sample_latent_conviction_orderbook_arrays(
    *,
    target,
    scenario_seed,
    price_sim_config,
):
    target = np.asarray(target, dtype=np.int8)
    rng = np.random.default_rng(int(scenario_seed))
    n_rows = len(target)

    conviction = rng.beta(
        float(price_sim_config["conviction_beta_alpha"]),
        float(price_sim_config["conviction_beta_beta"]),
        size=n_rows,
    ).astype(np.float64, copy=False)
    abs_gap = float(price_sim_config["gap_min"]) + np.power(
        conviction,
        float(price_sim_config["gap_gamma"]),
    ) * (float(price_sim_config["gap_max"]) - float(price_sim_config["gap_min"]))
    p_correct = float(price_sim_config["p_correct_min"]) + conviction * (
        float(price_sim_config["p_correct_max"])
        - float(price_sim_config["p_correct_min"])
    )
    overround = float(price_sim_config["overround_min"]) + np.power(
        conviction,
        float(price_sim_config["overround_gamma"]),
    ) * (
        float(price_sim_config["overround_max"])
        - float(price_sim_config["overround_min"])
    )
    direction_is_correct = rng.random(n_rows).astype(np.float64, copy=False) < p_correct

    winner_ask = 0.5 + abs_gap / 2.0 + overround / 2.0
    loser_ask = 0.5 - abs_gap / 2.0 + overround / 2.0
    winner_is_up = np.where(direction_is_correct, target == 1, target == 0)
    up_ask = np.where(winner_is_up, winner_ask, loser_ask)
    down_ask = np.where(winner_is_up, loser_ask, winner_ask)
    eps = float(price_sim_config["eps"])
    tick_size = float(price_sim_config["tick_size"])
    up_ask = np.clip(up_ask, eps, 1.0 - eps)
    down_ask = np.clip(down_ask, eps, 1.0 - eps)
    up_ask = np.round(up_ask / tick_size) * tick_size
    down_ask = np.round(down_ask / tick_size) * tick_size
    up_ask = np.clip(up_ask, eps, 1.0 - eps)
    down_ask = np.clip(down_ask, eps, 1.0 - eps)

    return {
        "up_ask": up_ask,
        "down_ask": down_ask,
        "sim_order_min_size_shares": np.full(
            n_rows,
            float(price_sim_config["sim_order_min_size_shares"]),
            dtype=np.float64,
        ),
    }


def sample_constructive_confidence_calibrated_orderbook_arrays(
    *,
    target,
    p_raw,
    scenario_seed,
    price_sim_config,
):
    target = np.asarray(target, dtype=np.int8)
    p_raw = np.asarray(p_raw, dtype=np.float64)
    if target.ndim != 1 or p_raw.ndim != 1 or len(target) != len(p_raw):
        raise ValueError("target and p_raw must be 1D arrays with equal length.")

    calibration = load_constructive_live_market_calibration(
        trade_csv_glob=str(price_sim_config["trade_csv_glob"]),
        shared_csv_path=str(price_sim_config["shared_csv_path"]),
        recent_resolved_rows=price_sim_config.get("recent_resolved_rows"),
        confidence_quantile_bins=int(price_sim_config["confidence_quantile_bins"]),
        min_pool_rows=int(price_sim_config["min_pool_rows"]),
        smoothing_passes=int(price_sim_config["smoothing_passes"]),
        max_prediction_delay_ms=price_sim_config.get("max_prediction_delay_ms"),
        max_decision_delay_ms=price_sim_config.get("max_decision_delay_ms"),
        max_market_lookup_ms=price_sim_config.get("max_market_lookup_ms"),
        max_submit_order_ms=price_sim_config.get("max_submit_order_ms"),
        max_execution_ms=price_sim_config.get("max_execution_ms"),
    )
    rng = np.random.default_rng(int(scenario_seed))
    model_conf_rank = _rank01(np.abs(p_raw - 0.5))
    bin_centers = calibration["bin_centers"]

    def _curve_values(name):
        curve = np.asarray(calibration[name], dtype=np.float64)
        return np.interp(
            model_conf_rank,
            bin_centers,
            curve,
            left=float(curve[0]),
            right=float(curve[-1]),
        )

    abs_gap_mean = _curve_values("abs_gap_mean_curve")
    abs_gap_std = _curve_values("abs_gap_std_curve")
    overround_mean = _curve_values("overround_mean_curve")
    overround_std = _curve_values("overround_std_curve")
    tie_rate = _curve_values("tie_rate_curve")
    p_correct = _curve_values("p_correct_curve")

    correlation = float(calibration["gap_overround_correlation"]) * float(
        price_sim_config["correlation_shrink"]
    )
    correlation = float(np.clip(correlation, -0.95, 0.95))

    z_gap = rng.standard_normal(len(target)).astype(np.float64, copy=False)
    z_overround = (
        correlation * z_gap
        + np.sqrt(max(1.0 - correlation * correlation, 0.0))
        * rng.standard_normal(len(target)).astype(np.float64, copy=False)
    )
    abs_gap = abs_gap_mean + float(price_sim_config["abs_gap_std_scale"]) * abs_gap_std * z_gap
    overround = (
        overround_mean
        + float(price_sim_config["overround_std_scale"]) * overround_std * z_overround
    )
    abs_gap = np.where(np.isfinite(abs_gap), abs_gap, abs_gap_mean)
    overround = np.where(np.isfinite(overround), overround, overround_mean)
    abs_gap = np.clip(abs_gap, 0.0, None)
    overround = np.clip(overround, 0.0, None)

    is_tie = rng.random(len(target)).astype(np.float64, copy=False) < np.clip(
        tie_rate * float(price_sim_config["tie_rate_scale"]),
        0.0,
        0.95,
    )
    min_gap = float(price_sim_config["min_gap_ticks"]) * float(price_sim_config["tick_size"])
    abs_gap = np.where(is_tie, 0.0, np.maximum(abs_gap, min_gap))
    eps = float(price_sim_config["eps"])
    max_gap = np.maximum(1.0 - overround - 2.0 * eps, 0.0)
    abs_gap = np.minimum(abs_gap, max_gap)

    market_direction_correct = (
        rng.random(len(target)).astype(np.float64, copy=False)
        < np.clip(p_correct, 0.0, 1.0)
    )

    winner_ask = 0.5 + abs_gap / 2.0 + overround / 2.0
    loser_ask = 0.5 - abs_gap / 2.0 + overround / 2.0
    tie_ask = 0.5 + overround / 2.0
    winner_is_up = np.where(market_direction_correct, target == 1, target == 0)
    up_ask = np.where(is_tie, tie_ask, np.where(winner_is_up, winner_ask, loser_ask))
    down_ask = np.where(is_tie, tie_ask, np.where(winner_is_up, loser_ask, winner_ask))
    up_ask = np.clip(up_ask, eps, 1.0 - eps)
    down_ask = np.clip(down_ask, eps, 1.0 - eps)
    tick_size = float(price_sim_config["tick_size"])
    up_ask = np.round(up_ask / tick_size) * tick_size
    down_ask = np.round(down_ask / tick_size) * tick_size
    up_ask = np.clip(up_ask, eps, 1.0 - eps)
    down_ask = np.clip(down_ask, eps, 1.0 - eps)

    return {
        "up_ask": up_ask.astype(np.float64, copy=False),
        "down_ask": down_ask.astype(np.float64, copy=False),
        "sim_order_min_size_shares": np.full(
            len(target),
            float(price_sim_config["sim_order_min_size_shares"]),
            dtype=np.float64,
        ),
    }


def sample_empirical_residual_orderbook_arrays(
    *,
    p_raw,
    scenario_seed,
    price_sim_config,
):
    p_raw = np.asarray(p_raw, dtype=np.float64)
    if p_raw.ndim != 1:
        raise ValueError("p_raw must be a 1D array.")

    calibration = load_empirical_residual_live_market_calibration(
        trade_csv_glob=str(price_sim_config["trade_csv_glob"]),
        shared_csv_path=str(price_sim_config["shared_csv_path"]),
        recent_resolved_rows=price_sim_config.get("recent_resolved_rows"),
        confidence_quantile_bins=int(price_sim_config["confidence_quantile_bins"]),
        min_pool_rows=int(price_sim_config["min_pool_rows"]),
        preferred_model_hash=price_sim_config.get("preferred_model_hash"),
        max_prediction_delay_ms=price_sim_config.get("max_prediction_delay_ms"),
        max_decision_delay_ms=price_sim_config.get("max_decision_delay_ms"),
        max_market_lookup_ms=price_sim_config.get("max_market_lookup_ms"),
        max_submit_order_ms=price_sim_config.get("max_submit_order_ms"),
        max_execution_ms=price_sim_config.get("max_execution_ms"),
    )
    rng = np.random.default_rng(int(scenario_seed))

    conf = np.abs(p_raw - 0.5)
    confidence_edges = np.asarray(calibration["confidence_edges"], dtype=np.float64)
    if len(confidence_edges) <= 2:
        query_bins = np.zeros(len(p_raw), dtype=np.int64)
    else:
        query_bins = np.searchsorted(
            confidence_edges[1:-1],
            conf,
            side="right",
        ).astype(np.int64, copy=False)

    sampled_indices = np.empty(len(p_raw), dtype=np.int64)
    global_indices = np.asarray(calibration["all_indices"], dtype=np.int64)
    for bin_id, pool_indices in enumerate(calibration["bin_indices"]):
        row_mask = query_bins == bin_id
        n_rows = int(np.count_nonzero(row_mask))
        if n_rows == 0:
            continue
        pool = np.asarray(pool_indices, dtype=np.int64)
        if len(pool) == 0:
            pool = global_indices
        sampled_indices[row_mask] = pool[
            rng.integers(0, len(pool), size=n_rows, dtype=np.int64)
        ]

    up_residual = np.asarray(calibration["up_residual"], dtype=np.float64)[sampled_indices]
    down_residual = np.asarray(calibration["down_residual"], dtype=np.float64)[sampled_indices]
    order_min_size = np.asarray(calibration["order_min_size"], dtype=np.float64)[
        sampled_indices
    ]

    up_ask = p_raw + up_residual
    down_ask = (1.0 - p_raw) + down_residual

    eps = float(price_sim_config["eps"])
    tick_size = float(price_sim_config["tick_size"])
    up_ask = np.clip(up_ask, eps, 1.0 - eps)
    down_ask = np.clip(down_ask, eps, 1.0 - eps)
    up_ask = np.round(up_ask / tick_size) * tick_size
    down_ask = np.round(down_ask / tick_size) * tick_size
    up_ask = np.clip(up_ask, eps, 1.0 - eps)
    down_ask = np.clip(down_ask, eps, 1.0 - eps)

    fallback_order_min_size = float(
        calibration.get(
            "fallback_order_min_size",
            float(price_sim_config["sim_order_min_size_shares"]),
        )
    )
    if not np.isfinite(fallback_order_min_size) or fallback_order_min_size <= 0.0:
        fallback_order_min_size = float(price_sim_config["sim_order_min_size_shares"])
    invalid_order_min_size = ~np.isfinite(order_min_size) | (order_min_size <= 0.0)
    if np.any(invalid_order_min_size):
        order_min_size = order_min_size.copy()
        order_min_size[invalid_order_min_size] = fallback_order_min_size

    return {
        "up_ask": up_ask.astype(np.float64, copy=False),
        "down_ask": down_ask.astype(np.float64, copy=False),
        "sim_order_min_size_shares": order_min_size.astype(np.float64, copy=False),
    }


def sample_market_orderbook_arrays(
    *,
    target,
    scenario_seed,
    price_sim_config,
    p_raw=None,
    market_elapsed_ms=None,
):
    model = str(price_sim_config["model"])
    if model == "elapsed_target_empirical":
        if market_elapsed_ms is None:
            raise ValueError("elapsed_target_empirical requires market_elapsed_ms.")
        return sample_elapsed_target_empirical_orderbook_arrays(
            target=target,
            market_elapsed_ms=market_elapsed_ms,
            scenario_seed=scenario_seed,
            price_sim_config=price_sim_config,
        )
    if model == "latent_conviction_directional":
        return sample_latent_conviction_orderbook_arrays(
            target=target,
            scenario_seed=scenario_seed,
            price_sim_config=price_sim_config,
        )
    if model == "constructive_confidence_calibrated":
        if p_raw is None:
            raise ValueError("constructive_confidence_calibrated requires p_raw.")
        return sample_constructive_confidence_calibrated_orderbook_arrays(
            target=target,
            p_raw=p_raw,
            scenario_seed=scenario_seed,
            price_sim_config=price_sim_config,
        )
    if model == "empirical_residual":
        if p_raw is None:
            raise ValueError("empirical_residual requires p_raw.")
        return sample_empirical_residual_orderbook_arrays(
            p_raw=p_raw,
            scenario_seed=scenario_seed,
            price_sim_config=price_sim_config,
        )
    raise ValueError(f"Unsupported market price sim model: {model}")


def sample_elapsed_target_empirical_orderbook_arrays(
    *,
    target,
    market_elapsed_ms,
    scenario_seed,
    price_sim_config,
):
    target = np.asarray(target, dtype=np.int8)
    market_elapsed_ms = np.asarray(market_elapsed_ms, dtype=np.float64)
    if target.ndim != 1 or market_elapsed_ms.ndim != 1 or len(target) != len(market_elapsed_ms):
        raise ValueError(
            "target and market_elapsed_ms must be 1D arrays with equal length."
        )

    calibration = load_elapsed_target_live_market_calibration(
        trade_csv_glob=str(price_sim_config["trade_csv_glob"]),
        shared_csv_path=str(price_sim_config["shared_csv_path"]),
        recent_resolved_rows=price_sim_config.get("recent_resolved_rows"),
        elapsed_quantile_bins=int(price_sim_config["elapsed_quantile_bins"]),
        min_pool_rows=int(price_sim_config["min_pool_rows"]),
        preferred_model_hash=price_sim_config.get("preferred_model_hash"),
        max_prediction_delay_ms=price_sim_config.get("max_prediction_delay_ms"),
        max_decision_delay_ms=price_sim_config.get("max_decision_delay_ms"),
        max_market_lookup_ms=price_sim_config.get("max_market_lookup_ms"),
        max_submit_order_ms=price_sim_config.get("max_submit_order_ms"),
        max_execution_ms=price_sim_config.get("max_execution_ms"),
    )
    rng = np.random.default_rng(int(scenario_seed))
    elapsed_edges = np.asarray(calibration["elapsed_edges"], dtype=np.float64)
    query_bins = np.searchsorted(
        elapsed_edges[1:-1],
        market_elapsed_ms,
        side="right",
    ).astype(np.int64, copy=False)

    sampled_indices = np.empty(len(target), dtype=np.int64)
    global_indices = np.asarray(calibration["all_indices"], dtype=np.int64)
    for bin_id, pool_indices in enumerate(calibration["bin_indices"]):
        row_mask = query_bins == bin_id
        n_rows = int(np.count_nonzero(row_mask))
        if n_rows == 0:
            continue
        pool = np.asarray(pool_indices, dtype=np.int64)
        if len(pool) == 0:
            pool = global_indices
        sampled_indices[row_mask] = pool[
            rng.integers(0, len(pool), size=n_rows, dtype=np.int64)
        ]

    winner_ask = np.asarray(calibration["actual_winner_ask"], dtype=np.float64)[
        sampled_indices
    ]
    loser_ask = np.asarray(calibration["actual_loser_ask"], dtype=np.float64)[
        sampled_indices
    ]
    order_min_size = np.asarray(calibration["order_min_size"], dtype=np.float64)[
        sampled_indices
    ]

    up_ask = np.where(target == 1, winner_ask, loser_ask)
    down_ask = np.where(target == 1, loser_ask, winner_ask)

    eps = float(price_sim_config["eps"])
    tick_size = float(price_sim_config["tick_size"])
    up_ask = np.clip(up_ask, eps, 1.0 - eps)
    down_ask = np.clip(down_ask, eps, 1.0 - eps)
    up_ask = np.round(up_ask / tick_size) * tick_size
    down_ask = np.round(down_ask / tick_size) * tick_size
    up_ask = np.clip(up_ask, eps, 1.0 - eps)
    down_ask = np.clip(down_ask, eps, 1.0 - eps)

    fallback_order_min_size = float(
        calibration.get(
            "fallback_order_min_size",
            float(price_sim_config["sim_order_min_size_shares"]),
        )
    )
    if not np.isfinite(fallback_order_min_size) or fallback_order_min_size <= 0.0:
        fallback_order_min_size = float(price_sim_config["sim_order_min_size_shares"])
    invalid_order_min_size = ~np.isfinite(order_min_size) | (order_min_size <= 0.0)
    if np.any(invalid_order_min_size):
        order_min_size = order_min_size.copy()
        order_min_size[invalid_order_min_size] = fallback_order_min_size

    return {
        "up_ask": up_ask.astype(np.float64, copy=False),
        "down_ask": down_ask.astype(np.float64, copy=False),
        "sim_order_min_size_shares": order_min_size.astype(np.float64, copy=False),
    }
