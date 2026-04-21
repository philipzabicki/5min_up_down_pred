import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common_config_utils import load_json_object
from market_price_sim import load_live_market_empirical_frame
from project_config import RUNTIME_ACTIVE_PATH, load_runtime_artifact_paths
from target_weights import compute_decision_mask_from_opened
from trade_policy import build_trade_intent, load_trade_policy_runtime_config

DEFAULT_SHARED_CSV_PATH = Path("data/live/polymarket_5m.csv")
DEFAULT_OUTPUT_DIR = Path("data/analysis/model_direction_margins")
DEFAULT_OOF_TIME_COL = "Opened"
DEFAULT_TARGET_COL = "target_5m_candle_up"
DEFAULT_PRED_COL = "oof_pred_proba_up"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MARGIN_MAX = 0.10
DEFAULT_MARGIN_STEP = 0.0001
DEFAULT_ORDER_MIN_SIZE_SHARES = 5.0
DEFAULT_TOP_CANDIDATES = 10
DEFAULT_FOLDS = 10
SIM_BANKROLL_USDC = 1_000_000_000.0
MARGIN_EPS = 1e-12


def _utcnow():
    return datetime.now(timezone.utc)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _as_float(value, *, default=None):
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value_f):
        return default
    return value_f


def _coerce_positive_float(value, *, label):
    value_f = _as_float(value)
    if value_f is None or value_f <= 0.0:
        raise ValueError(f"{label} must be finite and > 0, got: {value!r}")
    return float(value_f)


def _coerce_probability(value, *, label, default=None):
    value_f = _as_float(value, default=default)
    if value_f is None:
        raise ValueError(f"{label} must be finite, got: {value!r}")
    if value_f < 0.0 or value_f > 1.0:
        raise ValueError(f"{label} must be in [0, 1], got: {value!r}")
    return float(value_f)


def _coerce_margin_list(raw_value, *, default_max, default_step, label):
    if raw_value in (None, ""):
        default_max = _coerce_positive_float(default_max, label=f"{label}_max")
        default_step = _coerce_positive_float(default_step, label=f"{label}_step")
        values = np.arange(0.0, default_max + default_step * 0.5, default_step)
        return np.round(values.astype(np.float64, copy=False), 10)

    values = []
    for raw_part in str(raw_value).split(","):
        part = raw_part.strip()
        if not part:
            continue
        value = _coerce_probability(float(part), label=label)
        values.append(value)
    if not values:
        raise ValueError(f"{label} produced an empty grid.")
    return np.unique(np.asarray(values, dtype=np.float64))


def _load_model_meta(model_meta_path):
    payload = load_json_object(model_meta_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Model meta must be a JSON object: {model_meta_path}")
    return payload


def resolve_script_inputs(args):
    runtime_paths = load_runtime_artifact_paths(
        args.runtime_manifest_path or RUNTIME_ACTIVE_PATH
    )
    runtime_config_path = Path(
        args.runtime_config_path or runtime_paths["trade_policy_runtime_config_path"]
    )
    runtime_cfg = load_trade_policy_runtime_config(runtime_config_path)

    model_meta_path = Path(args.model_meta_path or runtime_paths["model_meta_path"])
    model_meta = _load_model_meta(model_meta_path)

    oof_info = model_meta.get("oof_predictions")
    if not isinstance(oof_info, dict):
        raise ValueError(f"Model meta missing oof_predictions object: {model_meta_path}")
    if not bool(oof_info.get("enabled", False)):
        raise ValueError(f"OOF predictions are disabled in model meta: {model_meta_path}")

    oof_path = Path(args.oof_path or oof_info.get("path") or "")
    if not str(oof_path).strip():
        raise ValueError(f"Model meta missing oof_predictions.path: {model_meta_path}")
    threshold = (
        args.threshold
        if args.threshold is not None
        else model_meta.get("prediction_threshold", DEFAULT_THRESHOLD)
    )

    return {
        "runtime_manifest_path": Path(args.runtime_manifest_path or RUNTIME_ACTIVE_PATH),
        "runtime_config_path": runtime_config_path,
        "runtime_cfg": runtime_cfg,
        "model_meta_path": model_meta_path,
        "model_meta": model_meta,
        "oof_path": oof_path,
        "oof_time_col": str(args.time_col or DEFAULT_OOF_TIME_COL),
        "oof_target_col": str(args.target_col or model_meta.get("target_col", DEFAULT_TARGET_COL)),
        "oof_pred_col": str(args.pred_col or oof_info.get("prediction_col", DEFAULT_PRED_COL)),
        "threshold": _coerce_probability(threshold, label="threshold"),
        "shared_csv_path": Path(args.shared_csv_path or DEFAULT_SHARED_CSV_PATH),
        "margin_grid_up": _coerce_margin_list(
            args.margin_grid_up,
            default_max=args.margin_max,
            default_step=args.margin_step,
            label="margin_grid_up",
        ),
        "margin_grid_down": _coerce_margin_list(
            args.margin_grid_down,
            default_max=args.margin_max,
            default_step=args.margin_step,
            label="margin_grid_down",
        ),
        "default_order_min_size_shares": _coerce_positive_float(
            args.default_order_min_size_shares,
            label="default_order_min_size_shares",
        ),
        "recent_aligned_rows": (
            None if args.recent_aligned_rows is None else int(args.recent_aligned_rows)
        ),
        "folds": max(int(args.folds), 1),
        "top_candidates": max(int(args.top_candidates), 1),
        "output_dir": Path(args.output_dir or DEFAULT_OUTPUT_DIR),
    }


def load_oof_decision_frame(
    parquet_path,
    *,
    time_col,
    pred_col,
    target_col,
):
    parquet_path = Path(parquet_path)
    columns = [time_col, pred_col, target_col]
    frame = pd.read_parquet(parquet_path, columns=columns).copy()
    frame = frame.rename(
        columns={
            time_col: "opened_time",
            pred_col: "proba_up",
            target_col: "oof_target_up",
        }
    )
    frame["opened_time"] = pd.to_datetime(frame["opened_time"], errors="coerce", utc=True)
    frame["proba_up"] = pd.to_numeric(frame["proba_up"], errors="coerce")
    frame["oof_target_up"] = pd.to_numeric(frame["oof_target_up"], errors="coerce")
    frame = frame.dropna(subset=["opened_time", "proba_up", "oof_target_up"]).copy()
    frame = frame[
        np.isfinite(frame["proba_up"])
        & (frame["proba_up"] >= 0.0)
        & (frame["proba_up"] <= 1.0)
        & frame["oof_target_up"].isin([0.0, 1.0])
    ].copy()
    if frame.empty:
        raise ValueError(f"No usable OOF rows remained after cleanup: {parquet_path}")

    decision_mask = compute_decision_mask_from_opened(frame["opened_time"])
    frame = frame.loc[decision_mask].copy()
    if frame.empty:
        raise ValueError(f"No OOF decision rows matched 5m decision mask: {parquet_path}")

    frame["oof_target_up"] = frame["oof_target_up"].astype(np.int8, copy=False)
    frame["bucket_start"] = frame["opened_time"] + pd.Timedelta(minutes=1)
    frame = frame.sort_values("bucket_start", kind="stable").reset_index(drop=True)
    return frame.loc[:, ["opened_time", "bucket_start", "proba_up", "oof_target_up"]]


def build_canonical_quote_frame(shared_csv_path):
    frame = load_live_market_empirical_frame(
        trade_csv_glob=None,
        shared_csv_path=shared_csv_path,
    ).copy()
    if "bucket_start" not in frame.columns:
        raise ValueError("Shared Polymarket CSV must contain bucket_start.")

    frame = frame.dropna(subset=["bucket_start"]).copy()
    if frame.empty:
        raise ValueError("No usable quote rows remained after dropping missing bucket_start.")

    actual_conflicts = (
        frame.groupby("bucket_start")["actual_up"].nunique(dropna=True).gt(1).sum()
    )
    if int(actual_conflicts) > 0:
        raise ValueError(
            "actual_up differs inside the same bucket_start in shared Polymarket CSV."
        )

    frame["quote_rows_in_bucket"] = (
        frame.groupby("bucket_start")["bucket_start"].transform("size").astype(np.int32)
    )
    sort_columns = ["bucket_start"]
    if "prediction_delay_ms" in frame.columns:
        sort_columns.append("prediction_delay_ms")
    elif "prediction_time" in frame.columns:
        sort_columns.append("prediction_time")
    frame = frame.sort_values(sort_columns, kind="stable", na_position="last")

    total_rows = int(len(frame))
    frame = frame.drop_duplicates(subset=["bucket_start"], keep="first").reset_index(drop=True)
    duplicates_dropped = total_rows - int(len(frame))
    if "prediction_time" not in frame.columns:
        frame["prediction_time"] = pd.NaT
    if "prediction_delay_ms" not in frame.columns:
        frame["prediction_delay_ms"] = np.nan

    return (
        frame.loc[
            :,
            [
                "bucket_start",
                "prediction_time",
                "prediction_delay_ms",
                "pm_up_best_ask",
                "pm_down_best_ask",
                "pm_order_min_size",
                "actual_up",
                "quote_rows_in_bucket",
            ],
        ].copy(),
        {
            "quote_rows_total": total_rows,
            "canonical_quote_rows": int(len(frame)),
            "bucket_duplicates_dropped": int(duplicates_dropped),
            "bucket_actual_up_conflicts": int(actual_conflicts),
        },
    )


def align_oof_with_quotes(oof_frame, quote_frame, *, threshold, recent_aligned_rows=None):
    aligned = oof_frame.merge(
        quote_frame,
        on="bucket_start",
        how="inner",
        validate="one_to_one",
    )
    aligned = aligned.sort_values("bucket_start", kind="stable").reset_index(drop=True)
    if recent_aligned_rows is not None:
        recent_aligned_rows = int(recent_aligned_rows)
        if recent_aligned_rows <= 0:
            raise ValueError("recent_aligned_rows must be > 0 when provided.")
        aligned = aligned.tail(recent_aligned_rows).reset_index(drop=True)
    if aligned.empty:
        raise ValueError("No overlapping rows between OOF decision rows and Polymarket quotes.")

    target_match_mask = aligned["oof_target_up"].isin([0, 1]) & aligned["actual_up"].isin([0, 1])
    target_mismatch_count = int(
        (
            aligned.loc[target_match_mask, "oof_target_up"].astype(np.int8, copy=False)
            != aligned.loc[target_match_mask, "actual_up"].astype(np.int8, copy=False)
        ).sum()
    )
    aligned["model_conf"] = (aligned["proba_up"] - float(threshold)).abs()

    return aligned, {
        "aligned_rows": int(len(aligned)),
        "aligned_bucket_start_min": aligned["bucket_start"].min().isoformat(),
        "aligned_bucket_start_max": aligned["bucket_start"].max().isoformat(),
        "oof_target_comparable_rows": int(target_match_mask.sum()),
        "oof_target_mismatch_count": int(target_mismatch_count),
        "oof_target_mismatch_rate": (
            float(target_mismatch_count / target_match_mask.sum())
            if int(target_match_mask.sum()) > 0
            else float("nan")
        ),
    }


def _resolve_order_min_size(value, *, default_order_min_size_shares):
    order_min_size = _as_float(value)
    if order_min_size is None or order_min_size <= 0.0:
        return float(default_order_min_size_shares)
    return float(order_min_size)


def precompute_trade_arrays(
    aligned_frame,
    *,
    stake_multiplier,
    fee_model,
    default_order_min_size_shares,
):
    n_rows = int(len(aligned_frame))
    yes_tradable = np.zeros(n_rows, dtype=bool)
    no_tradable = np.zeros(n_rows, dtype=bool)
    yes_bet_usdc = np.zeros(n_rows, dtype=np.float64)
    no_bet_usdc = np.zeros(n_rows, dtype=np.float64)
    yes_win_pnl_usdc = np.zeros(n_rows, dtype=np.float64)
    no_win_pnl_usdc = np.zeros(n_rows, dtype=np.float64)
    order_min_size_used = np.zeros(n_rows, dtype=np.float64)

    for idx, row in enumerate(aligned_frame.itertuples(index=False)):
        order_min_size = _resolve_order_min_size(
            getattr(row, "pm_order_min_size", np.nan),
            default_order_min_size_shares=default_order_min_size_shares,
        )
        order_min_size_used[idx] = float(order_min_size)

        base_policy = {
            "ask_yes": float(row.pm_up_best_ask),
            "ask_no": float(row.pm_down_best_ask),
        }
        yes_intent = build_trade_intent(
            policy_result={**base_policy, "decision": "buy_yes"},
            bankroll=SIM_BANKROLL_USDC,
            stake_multiplier=stake_multiplier,
            fee_model=fee_model,
            order_min_size=order_min_size,
        )
        if yes_intent.get("final_reason") == "ok":
            yes_tradable[idx] = True
            yes_bet_usdc[idx] = float(yes_intent["bet_usdc"])
            yes_win_pnl_usdc[idx] = float(yes_intent["shares_net"] - yes_intent["bet_usdc"])

        no_intent = build_trade_intent(
            policy_result={**base_policy, "decision": "buy_no"},
            bankroll=SIM_BANKROLL_USDC,
            stake_multiplier=stake_multiplier,
            fee_model=fee_model,
            order_min_size=order_min_size,
        )
        if no_intent.get("final_reason") == "ok":
            no_tradable[idx] = True
            no_bet_usdc[idx] = float(no_intent["bet_usdc"])
            no_win_pnl_usdc[idx] = float(no_intent["shares_net"] - no_intent["bet_usdc"])

    return {
        "yes_tradable": yes_tradable,
        "no_tradable": no_tradable,
        "yes_bet_usdc": yes_bet_usdc,
        "no_bet_usdc": no_bet_usdc,
        "yes_win_pnl_usdc": yes_win_pnl_usdc,
        "no_win_pnl_usdc": no_win_pnl_usdc,
        "order_min_size_used": order_min_size_used,
    }


def build_fold_ids(n_rows, folds):
    n_rows = int(n_rows)
    folds = int(folds)
    if n_rows <= 0:
        raise ValueError("Cannot build fold ids for an empty aligned frame.")
    if folds <= 1:
        return np.zeros(n_rows, dtype=np.int32), 1
    folds = min(folds, n_rows)
    fold_ids = np.floor(np.arange(n_rows, dtype=np.float64) * float(folds) / float(n_rows)).astype(
        np.int32
    )
    fold_ids = np.clip(fold_ids, 0, folds - 1)
    return fold_ids, folds


def evaluate_margin_grid(
    aligned_frame,
    precomputed,
    *,
    threshold,
    margin_grid_up,
    margin_grid_down,
    folds,
):
    proba_up = aligned_frame["proba_up"].to_numpy(dtype=np.float64, copy=False)
    actual_up = aligned_frame["actual_up"].to_numpy(dtype=np.int8, copy=False)
    yes_margin = proba_up - float(threshold)
    no_margin = float(threshold) - proba_up

    fold_ids, fold_count = build_fold_ids(len(aligned_frame), folds)
    rows = []
    for margin_up in np.asarray(margin_grid_up, dtype=np.float64):
        yes_selected = (
            (yes_margin + MARGIN_EPS >= float(margin_up))
            & (yes_margin >= -MARGIN_EPS)
            & precomputed["yes_tradable"]
        )
        for margin_down in np.asarray(margin_grid_down, dtype=np.float64):
            no_selected = (
                (no_margin + MARGIN_EPS >= float(margin_down))
                & (no_margin > MARGIN_EPS)
                & precomputed["no_tradable"]
            )
            trade_mask = yes_selected | no_selected
            trade_count = int(trade_mask.sum())

            pnl_usdc = np.zeros(len(aligned_frame), dtype=np.float64)
            yes_win_mask = yes_selected & (actual_up == 1)
            yes_loss_mask = yes_selected & (actual_up == 0)
            no_win_mask = no_selected & (actual_up == 0)
            no_loss_mask = no_selected & (actual_up == 1)

            pnl_usdc[yes_win_mask] = precomputed["yes_win_pnl_usdc"][yes_win_mask]
            pnl_usdc[yes_loss_mask] = -precomputed["yes_bet_usdc"][yes_loss_mask]
            pnl_usdc[no_win_mask] = precomputed["no_win_pnl_usdc"][no_win_mask]
            pnl_usdc[no_loss_mask] = -precomputed["no_bet_usdc"][no_loss_mask]

            fold_pnl = np.bincount(
                fold_ids,
                weights=pnl_usdc,
                minlength=fold_count,
            ).astype(np.float64, copy=False)

            win_count = int((yes_win_mask | no_win_mask).sum())
            rows.append(
                {
                    "min_decision_margin_up": float(margin_up),
                    "min_decision_margin_down": float(margin_down),
                    "trade_count": trade_count,
                    "buy_yes_count": int(yes_selected.sum()),
                    "buy_no_count": int(no_selected.sum()),
                    "sum_pnl_usdc": float(pnl_usdc.sum()),
                    "mean_pnl_usdc": (
                        float(pnl_usdc[trade_mask].mean()) if trade_count > 0 else float("nan")
                    ),
                    "win_rate": (
                        float(win_count / trade_count) if trade_count > 0 else float("nan")
                    ),
                    "robust_score_usdc": float(pnl_usdc.sum() + fold_pnl.min()),
                    "fold_mean_pnl_usdc": float(fold_pnl.mean()),
                    "fold_min_pnl_usdc": float(fold_pnl.min()),
                    "fold_std_pnl_usdc": float(fold_pnl.std()),
                }
            )
    return pd.DataFrame(rows)


def recommend_margin_candidate(grid_frame, *, aligned_rows, min_trade_fraction=0.05, min_trade_floor=25):
    if grid_frame.empty:
        raise ValueError("Cannot recommend margins from an empty candidate grid.")

    aligned_rows = max(int(aligned_rows), 1)
    min_trade_count = max(int(min_trade_floor), int(math.ceil(aligned_rows * float(min_trade_fraction))))
    candidate_pool = grid_frame.loc[grid_frame["trade_count"] >= min_trade_count].copy()
    selection_notes = [f"min_trade_count={min_trade_count}"]
    if candidate_pool.empty:
        candidate_pool = grid_frame.copy()
        selection_notes.append("min_trade_count_relaxed")

    stable_pool = candidate_pool.loc[candidate_pool["fold_min_pnl_usdc"] >= 0.0].copy()
    if stable_pool.empty:
        stable_pool = candidate_pool.copy()
        selection_notes.append("fold_min_pnl_constraint_relaxed")
        selection_notes.append("using_robust_score=sum_pnl_usdc+fold_min_pnl_usdc")
        stable_pool = stable_pool.sort_values(
            by=[
                "robust_score_usdc",
                "sum_pnl_usdc",
                "mean_pnl_usdc",
                "trade_count",
                "min_decision_margin_up",
                "min_decision_margin_down",
            ],
            ascending=[False, False, False, False, True, True],
            kind="stable",
        ).reset_index(drop=True)
    else:
        stable_pool = stable_pool.sort_values(
            by=[
                "sum_pnl_usdc",
                "fold_min_pnl_usdc",
                "mean_pnl_usdc",
                "trade_count",
                "min_decision_margin_up",
                "min_decision_margin_down",
            ],
            ascending=[False, False, False, False, True, True],
            kind="stable",
        ).reset_index(drop=True)
    recommended = stable_pool.iloc[0].to_dict()
    recommended["selection_notes"] = selection_notes
    return recommended, stable_pool


def _serialize_candidate_row(row):
    return {
        "min_decision_margin_up": float(row["min_decision_margin_up"]),
        "min_decision_margin_down": float(row["min_decision_margin_down"]),
        "trade_count": int(row["trade_count"]),
        "buy_yes_count": int(row["buy_yes_count"]),
        "buy_no_count": int(row["buy_no_count"]),
        "sum_pnl_usdc": float(row["sum_pnl_usdc"]),
        "mean_pnl_usdc": float(row["mean_pnl_usdc"]),
        "win_rate": float(row["win_rate"]),
        "robust_score_usdc": float(row["robust_score_usdc"]),
        "fold_mean_pnl_usdc": float(row["fold_mean_pnl_usdc"]),
        "fold_min_pnl_usdc": float(row["fold_min_pnl_usdc"]),
        "fold_std_pnl_usdc": float(row["fold_std_pnl_usdc"]),
    }


def build_report(inputs, *, oof_frame, quote_summary, aligned_frame, alignment_summary, grid_frame, recommended):
    runtime_cfg = inputs["runtime_cfg"]
    top_candidates = (
        grid_frame.sort_values(
            by=[
                "sum_pnl_usdc",
                "fold_min_pnl_usdc",
                "robust_score_usdc",
                "mean_pnl_usdc",
                "trade_count",
            ],
            ascending=[False, False, False, False, False],
            kind="stable",
        )
        .head(inputs["top_candidates"])
        .to_dict(orient="records")
    )
    return {
        "created_utc": _utcnow().isoformat(),
        "inputs": {
            "runtime_manifest_path": str(inputs["runtime_manifest_path"]),
            "runtime_config_path": str(inputs["runtime_config_path"]),
            "model_meta_path": str(inputs["model_meta_path"]),
            "oof_path": str(inputs["oof_path"]),
            "shared_csv_path": str(inputs["shared_csv_path"]),
            "oof_time_col": inputs["oof_time_col"],
            "oof_target_col": inputs["oof_target_col"],
            "oof_pred_col": inputs["oof_pred_col"],
            "threshold": float(inputs["threshold"]),
            "stake_multiplier": float(runtime_cfg["stake_multiplier"]),
            "default_order_min_size_shares": float(inputs["default_order_min_size_shares"]),
            "folds": int(inputs["folds"]),
            "recent_aligned_rows": inputs["recent_aligned_rows"],
        },
        "fee_model": runtime_cfg["fee_model"],
        "oof_summary": {
            "decision_rows": int(len(oof_frame)),
            "decision_bucket_start_min": oof_frame["bucket_start"].min().isoformat(),
            "decision_bucket_start_max": oof_frame["bucket_start"].max().isoformat(),
        },
        "quote_summary": quote_summary,
        "alignment_summary": alignment_summary,
        "recommendation": {
            "min_decision_margin_up": float(recommended["min_decision_margin_up"]),
            "min_decision_margin_down": float(recommended["min_decision_margin_down"]),
            "selection_notes": list(recommended["selection_notes"]),
            "metrics": _serialize_candidate_row(recommended),
            "config_snippet": {
                "min_decision_margin_up": float(recommended["min_decision_margin_up"]),
                "min_decision_margin_down": float(recommended["min_decision_margin_down"]),
            },
        },
        "top_candidates": [
            _serialize_candidate_row(candidate) for candidate in top_candidates
        ],
        "grid_row_count": int(len(grid_frame)),
        "aligned_rows": int(len(aligned_frame)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recommend min_decision_margin_up/down from model OOF predictions "
            "aligned to historical Polymarket quote snapshots."
        )
    )
    parser.add_argument("--runtime-manifest-path", default=str(RUNTIME_ACTIVE_PATH))
    parser.add_argument("--runtime-config-path")
    parser.add_argument("--model-meta-path")
    parser.add_argument("--oof-path")
    parser.add_argument("--shared-csv-path", default=str(DEFAULT_SHARED_CSV_PATH))
    parser.add_argument("--time-col")
    parser.add_argument("--target-col")
    parser.add_argument("--pred-col")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--margin-grid-up")
    parser.add_argument("--margin-grid-down")
    parser.add_argument("--margin-max", type=float, default=DEFAULT_MARGIN_MAX)
    parser.add_argument("--margin-step", type=float, default=DEFAULT_MARGIN_STEP)
    parser.add_argument(
        "--default-order-min-size-shares",
        type=float,
        default=DEFAULT_ORDER_MIN_SIZE_SHARES,
    )
    parser.add_argument("--recent-aligned-rows", type=int)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--top-candidates", type=int, default=DEFAULT_TOP_CANDIDATES)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    inputs = resolve_script_inputs(args)
    oof_frame = load_oof_decision_frame(
        inputs["oof_path"],
        time_col=inputs["oof_time_col"],
        pred_col=inputs["oof_pred_col"],
        target_col=inputs["oof_target_col"],
    )
    quote_frame, quote_summary = build_canonical_quote_frame(inputs["shared_csv_path"])
    aligned_frame, alignment_summary = align_oof_with_quotes(
        oof_frame,
        quote_frame,
        threshold=inputs["threshold"],
        recent_aligned_rows=inputs["recent_aligned_rows"],
    )
    precomputed = precompute_trade_arrays(
        aligned_frame,
        stake_multiplier=inputs["runtime_cfg"]["stake_multiplier"],
        fee_model=inputs["runtime_cfg"]["fee_model"],
        default_order_min_size_shares=inputs["default_order_min_size_shares"],
    )
    grid_frame = evaluate_margin_grid(
        aligned_frame,
        precomputed,
        threshold=inputs["threshold"],
        margin_grid_up=inputs["margin_grid_up"],
        margin_grid_down=inputs["margin_grid_down"],
        folds=inputs["folds"],
    )
    recommended, _ = recommend_margin_candidate(
        grid_frame,
        aligned_rows=len(aligned_frame),
    )
    grid_sorted = grid_frame.sort_values(
        by=[
            "sum_pnl_usdc",
            "fold_min_pnl_usdc",
            "robust_score_usdc",
            "mean_pnl_usdc",
            "trade_count",
            "min_decision_margin_up",
            "min_decision_margin_down",
        ],
        ascending=[False, False, False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    run_dir = inputs["output_dir"] / _utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = run_dir / "candidate_grid.csv"
    report_path = run_dir / "recommendation_report.json"
    grid_sorted.to_csv(ranked_path, index=False)
    report = build_report(
        inputs,
        oof_frame=oof_frame,
        quote_summary=quote_summary,
        aligned_frame=aligned_frame,
        alignment_summary=alignment_summary,
        grid_frame=grid_sorted,
        recommended=recommended,
    )
    save_json(report_path, report)

    print("recommended_config_snippet")
    print(json.dumps(report["recommendation"]["config_snippet"], indent=2))
    print(
        "summary "
        f"aligned_rows={report['alignment_summary']['aligned_rows']} "
        f"mismatch_rate={report['alignment_summary']['oof_target_mismatch_rate']:.4f} "
        f"candidate_grid={len(grid_sorted)}"
    )
    print(f"report_path={report_path}")
    print(f"grid_path={ranked_path}")


if __name__ == "__main__":
    main()
