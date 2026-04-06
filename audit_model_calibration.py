import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from metrics_utils import weighted_brier_score

DEFAULT_OOF_PATH = Path(
    "data/modeling_datasets/BTCUSD_INDEXVOL_UM_BTCUSDT1m_oof_predictions.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data/calibration/model_probability")
DEFAULT_TIME_COL = "Opened"
DEFAULT_TARGET_COL = "target_5m_candle_up"
DEFAULT_WEIGHT_COL = "target_5m_weight"
DEFAULT_PRED_COL = "oof_pred_proba_up"
DEFAULT_EVAL_FRACTION = 0.2
DEFAULT_N_BINS = 20
DEFAULT_MIN_ROWS_PER_BIN = 1
PROB_EPS = 1e-6


def _clip_probabilities(values, *, eps=PROB_EPS):
    values_f = np.asarray(values, dtype=np.float64)
    return np.clip(values_f, float(eps), 1.0 - float(eps))


def weighted_binary_log_loss(y_true, y_pred_proba, sample_weight=None):
    y_true_f = np.asarray(y_true, dtype=np.float64)
    y_pred_proba_f = _clip_probabilities(y_pred_proba)
    if y_true_f.ndim != 1 or y_pred_proba_f.ndim != 1:
        raise ValueError("weighted_binary_log_loss expects 1D arrays.")
    if y_true_f.shape[0] != y_pred_proba_f.shape[0]:
        raise ValueError(
            "weighted_binary_log_loss length mismatch: "
            f"{y_true_f.shape[0]} != {y_pred_proba_f.shape[0]}"
        )

    if sample_weight is None:
        weights = np.ones(len(y_true_f), dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.ndim != 1 or weights.shape[0] != len(y_true_f):
            raise ValueError("sample_weight must be 1D and match y_true length.")

    loss = -(y_true_f * np.log(y_pred_proba_f) + (1.0 - y_true_f) * np.log(1.0 - y_pred_proba_f))
    return float(np.average(loss, weights=weights))


def build_reliability_frame(
    y_true,
    y_pred_proba,
    *,
    sample_weight=None,
    n_bins=DEFAULT_N_BINS,
    min_rows_per_bin=DEFAULT_MIN_ROWS_PER_BIN,
):
    y_true_f = np.asarray(y_true, dtype=np.float64)
    y_pred_proba_f = np.asarray(y_pred_proba, dtype=np.float64)
    if y_true_f.ndim != 1 or y_pred_proba_f.ndim != 1:
        raise ValueError("build_reliability_frame expects 1D arrays.")
    if y_true_f.shape[0] != y_pred_proba_f.shape[0]:
        raise ValueError(
            "build_reliability_frame length mismatch: "
            f"{y_true_f.shape[0]} != {y_pred_proba_f.shape[0]}"
        )
    if sample_weight is None:
        weights = np.ones(len(y_true_f), dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.ndim != 1 or weights.shape[0] != len(y_true_f):
            raise ValueError("sample_weight must be 1D and match y_true length.")

    n_bins = int(n_bins)
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")

    probs = _clip_probabilities(y_pred_proba_f)
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    bin_ids = np.searchsorted(edges[1:-1], probs, side="right").astype(np.int64)

    rows = []
    total_weight = float(np.sum(weights))
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(np.count_nonzero(mask))
        if count < int(min_rows_per_bin):
            continue
        bin_weights = weights[mask]
        weight_sum = float(np.sum(bin_weights))
        if weight_sum <= 0.0:
            continue
        mean_pred = float(np.average(probs[mask], weights=bin_weights))
        event_rate = float(np.average(y_true_f[mask], weights=bin_weights))
        rows.append(
            {
                "bin_id": int(bin_id),
                "bin_left": float(edges[bin_id]),
                "bin_right": float(edges[bin_id + 1]),
                "count": int(count),
                "weight_sum": float(weight_sum),
                "weight_fraction": (
                    float(weight_sum / total_weight) if total_weight > 0.0 else float("nan")
                ),
                "mean_pred": float(mean_pred),
                "event_rate": float(event_rate),
                "abs_calibration_gap": float(abs(mean_pred - event_rate)),
            }
        )
    return pd.DataFrame(rows)


def summarize_reliability(reliability_frame):
    if reliability_frame.empty:
        return {
            "bin_count": 0,
            "ece": float("nan"),
            "mce": float("nan"),
        }
    gap = reliability_frame["abs_calibration_gap"].to_numpy(dtype=np.float64, copy=False)
    weight_fraction = reliability_frame["weight_fraction"].to_numpy(dtype=np.float64, copy=False)
    return {
        "bin_count": int(len(reliability_frame)),
        "ece": float(np.sum(gap * weight_fraction)),
        "mce": float(np.max(gap)),
    }


def evaluate_probability_predictions(
    y_true,
    y_pred_proba,
    *,
    sample_weight=None,
    n_bins=DEFAULT_N_BINS,
):
    reliability = build_reliability_frame(
        y_true,
        y_pred_proba,
        sample_weight=sample_weight,
        n_bins=n_bins,
    )
    calibration = summarize_reliability(reliability)
    return {
        "brier_score": float(
            weighted_brier_score(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                sample_weight=sample_weight,
            )
        ),
        "log_loss": float(
            weighted_binary_log_loss(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                sample_weight=sample_weight,
            )
        ),
        "mean_pred": float(np.average(y_pred_proba, weights=sample_weight)),
        "event_rate": float(np.average(y_true, weights=sample_weight)),
        "ece": float(calibration["ece"]),
        "mce": float(calibration["mce"]),
        "reliability": reliability,
    }


def _logit(values):
    probs = _clip_probabilities(values)
    return np.log(probs / (1.0 - probs))


def fit_logistic_calibrator(raw_pred_proba, y_true, *, sample_weight=None):
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )
    x = _logit(raw_pred_proba).reshape(-1, 1)
    model.fit(x, np.asarray(y_true, dtype=np.int8), sample_weight=sample_weight)
    return model


def apply_logistic_calibrator(model, raw_pred_proba):
    x = _logit(raw_pred_proba).reshape(-1, 1)
    return model.predict_proba(x)[:, 1].astype(np.float64, copy=False)


def fit_isotonic_calibrator(raw_pred_proba, y_true, *, sample_weight=None):
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(
        np.asarray(raw_pred_proba, dtype=np.float64),
        np.asarray(y_true, dtype=np.float64),
        sample_weight=sample_weight,
    )
    return model


def apply_isotonic_calibrator(model, raw_pred_proba):
    return np.asarray(model.predict(np.asarray(raw_pred_proba, dtype=np.float64)), dtype=np.float64)


def load_oof_predictions_frame(
    parquet_path,
    *,
    time_col=DEFAULT_TIME_COL,
    target_col=DEFAULT_TARGET_COL,
    pred_col=DEFAULT_PRED_COL,
    weight_col=DEFAULT_WEIGHT_COL,
):
    parquet_path = Path(parquet_path)
    frame = pd.read_parquet(
        parquet_path,
        columns=[time_col, target_col, pred_col, weight_col],
    )
    frame = frame.rename(
        columns={
            time_col: "event_time",
            target_col: "target",
            pred_col: "raw_proba",
            weight_col: "sample_weight",
        }
    ).copy()
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="coerce")
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce")
    frame["raw_proba"] = pd.to_numeric(frame["raw_proba"], errors="coerce")
    frame["sample_weight"] = pd.to_numeric(frame["sample_weight"], errors="coerce")
    frame = frame.dropna(subset=["event_time", "target", "raw_proba"]).copy()
    frame = frame[frame["target"].isin([0.0, 1.0])].copy()
    frame["target"] = frame["target"].astype(np.int8, copy=False)
    frame["raw_proba"] = _clip_probabilities(frame["raw_proba"].to_numpy(dtype=np.float64, copy=False))
    invalid_weights = ~np.isfinite(frame["sample_weight"]) | (frame["sample_weight"] <= 0.0)
    if np.any(invalid_weights):
        frame.loc[invalid_weights, "sample_weight"] = 1.0
    frame = frame.sort_values("event_time", kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No usable rows in {parquet_path}")
    return frame


def split_oof_frame(frame, *, eval_fraction=DEFAULT_EVAL_FRACTION):
    eval_fraction = float(eval_fraction)
    if eval_fraction <= 0.0 or eval_fraction >= 1.0:
        raise ValueError("eval_fraction must be in (0, 1).")
    split_idx = int(round(len(frame) * (1.0 - eval_fraction)))
    split_idx = max(1, min(split_idx, len(frame) - 1))
    fit_frame = frame.iloc[:split_idx].reset_index(drop=True)
    eval_frame = frame.iloc[split_idx:].reset_index(drop=True)
    return fit_frame, eval_frame


def _best_method_name(summary_by_method, *, metric_name):
    return min(summary_by_method, key=lambda name: float(summary_by_method[name][metric_name]))


def _serialize_metrics(metrics):
    return {
        key: float(value)
        for key, value in metrics.items()
        if key != "reliability"
    }


def run_calibration_audit(
    *,
    parquet_path=DEFAULT_OOF_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    eval_fraction=DEFAULT_EVAL_FRACTION,
    n_bins=DEFAULT_N_BINS,
):
    frame = load_oof_predictions_frame(parquet_path)
    fit_frame, eval_frame = split_oof_frame(frame, eval_fraction=eval_fraction)

    y_fit = fit_frame["target"].to_numpy(dtype=np.int8, copy=False)
    raw_fit = fit_frame["raw_proba"].to_numpy(dtype=np.float64, copy=False)
    w_fit = fit_frame["sample_weight"].to_numpy(dtype=np.float64, copy=False)

    y_eval = eval_frame["target"].to_numpy(dtype=np.int8, copy=False)
    raw_eval = eval_frame["raw_proba"].to_numpy(dtype=np.float64, copy=False)
    w_eval = eval_frame["sample_weight"].to_numpy(dtype=np.float64, copy=False)

    isotonic_model = fit_isotonic_calibrator(raw_fit, y_fit, sample_weight=w_fit)
    logistic_model = fit_logistic_calibrator(raw_fit, y_fit, sample_weight=w_fit)

    eval_predictions = {
        "raw": raw_eval,
        "isotonic": apply_isotonic_calibrator(isotonic_model, raw_eval),
        "logistic": apply_logistic_calibrator(logistic_model, raw_eval),
    }

    unweighted_metrics = {}
    weighted_metrics = {}
    reliability_frames = []
    for method_name, probs in eval_predictions.items():
        unweighted = evaluate_probability_predictions(
            y_eval,
            probs,
            sample_weight=None,
            n_bins=n_bins,
        )
        weighted = evaluate_probability_predictions(
            y_eval,
            probs,
            sample_weight=w_eval,
            n_bins=n_bins,
        )
        unweighted_metrics[method_name] = unweighted
        weighted_metrics[method_name] = weighted

        reliability = unweighted["reliability"].copy()
        reliability["method"] = method_name
        reliability_frames.append(reliability)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    reliability_path = output_dir / f"model_calibration_reliability_{timestamp}.csv"
    report_path = output_dir / f"model_calibration_report_{timestamp}.json"

    reliability_frame = pd.concat(reliability_frames, ignore_index=True, sort=False)
    reliability_frame.to_csv(reliability_path, index=False)

    report = {
        "source": {
            "oof_predictions_path": str(Path(parquet_path)),
            "event_time_col": DEFAULT_TIME_COL,
            "target_col": DEFAULT_TARGET_COL,
            "pred_col": DEFAULT_PRED_COL,
            "weight_col": DEFAULT_WEIGHT_COL,
        },
        "split": {
            "eval_fraction": float(eval_fraction),
            "fit_rows": int(len(fit_frame)),
            "eval_rows": int(len(eval_frame)),
            "fit_start": fit_frame["event_time"].iloc[0].isoformat(),
            "fit_end": fit_frame["event_time"].iloc[-1].isoformat(),
            "eval_start": eval_frame["event_time"].iloc[0].isoformat(),
            "eval_end": eval_frame["event_time"].iloc[-1].isoformat(),
        },
        "fit_summary": {
            "fit_event_rate_unweighted": float(np.mean(y_fit)),
            "fit_event_rate_weighted": float(np.average(y_fit, weights=w_fit)),
            "fit_raw_mean_pred": float(np.mean(raw_fit)),
        },
        "eval_summary": {
            "eval_event_rate_unweighted": float(np.mean(y_eval)),
            "eval_event_rate_weighted": float(np.average(y_eval, weights=w_eval)),
            "eval_raw_mean_pred": float(np.mean(raw_eval)),
        },
        "methods": {
            method_name: {
                "unweighted": _serialize_metrics(unweighted_metrics[method_name]),
                "weighted": _serialize_metrics(weighted_metrics[method_name]),
            }
            for method_name in eval_predictions
        },
        "best_method": {
            "unweighted_brier": _best_method_name(unweighted_metrics, metric_name="brier_score"),
            "unweighted_log_loss": _best_method_name(unweighted_metrics, metric_name="log_loss"),
            "weighted_brier": _best_method_name(weighted_metrics, metric_name="brier_score"),
            "weighted_log_loss": _best_method_name(weighted_metrics, metric_name="log_loss"),
        },
        "artifacts": {
            "report_path": str(report_path),
            "reliability_csv_path": str(reliability_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Audit model probability calibration on OOF predictions.")
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_OOF_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_calibration_audit(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        eval_fraction=args.eval_fraction,
        n_bins=args.n_bins,
    )
    raw_unweighted = report["methods"]["raw"]["unweighted"]
    isotonic_unweighted = report["methods"]["isotonic"]["unweighted"]
    logistic_unweighted = report["methods"]["logistic"]["unweighted"]
    print(
        "model calibration audit | "
        f"fit_rows={report['split']['fit_rows']} "
        f"eval_rows={report['split']['eval_rows']} "
        f"eval_range={report['split']['eval_start']}..{report['split']['eval_end']}"
    )
    print(
        "unweighted eval | "
        f"raw_brier={raw_unweighted['brier_score']:.6f} "
        f"isotonic_brier={isotonic_unweighted['brier_score']:.6f} "
        f"logistic_brier={logistic_unweighted['brier_score']:.6f} "
        f"best_brier={report['best_method']['unweighted_brier']}"
    )
    print(
        "unweighted eval | "
        f"raw_ece={raw_unweighted['ece']:.6f} "
        f"isotonic_ece={isotonic_unweighted['ece']:.6f} "
        f"logistic_ece={logistic_unweighted['ece']:.6f} "
        f"best_log_loss={report['best_method']['unweighted_log_loss']}"
    )
    print(f"saved report | path={report['artifacts']['report_path']}")
    print(f"saved reliability csv | path={report['artifacts']['reliability_csv_path']}")


if __name__ == "__main__":
    main()
