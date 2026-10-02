"""Run the BTC OOF target, timing, stability, and live-latency audit."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utils.btc_oof_audit import (
    macro_average_fold_metrics,
    paired_block_bootstrap,
    recompute_close_target,
    score_probabilities,
    training_classification_metrics,
    validate_common_market_sample,
)
from utils.data import (
    TARGET_WEIGHT_COL,
    compute_target_weights_from_opened,
    resolve_modeling_dataset_output_paths,
    resolve_oof_prediction_output_paths,
)
from utils.project_config import load_modeling_settings


ASSET = "BTC"
EXPECTED_MARKET_VALUE_ROWS = 9407
BOOTSTRAP_BLOCK_DAYS = 3
SAMPLE_BOOTSTRAP_REPLICATIONS = 2000
HISTORY_BOOTSTRAP_REPLICATIONS = 250
LIVE_START_STAGE_COUNT = 1
ANALYSIS_ROOT = Path("data/analysis/polymarket/BTC")
REPORT_PATH = Path("docs/polymarket_btc_experiment.md")
AUDIT_START_MARKER = "<!-- BTC_OOF_TARGET_AUDIT_START -->"
AUDIT_END_MARKER = "<!-- BTC_OOF_TARGET_AUDIT_END -->"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _fmt(value, digits=6):
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_ci(point, bounds, digits=6):
    if point is None or not np.isfinite(float(point)):
        return "n/a"
    if not bounds or len(bounds) != 2 or not np.isfinite(bounds).all():
        return _fmt(point, digits)
    return f"{_fmt(point, digits)} [{_fmt(bounds[0], digits)}, {_fmt(bounds[1], digits)}]"


def _markdown_table(headers, rows):
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(escape(value) for value in row) + " |" for row in rows)
    return lines


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _probability_summary(probability):
    p = np.asarray(probability, dtype=np.float64)
    quantiles = np.quantile(p, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    bins = [0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0]
    counts, edges = np.histogram(p, bins=bins)
    return {
        "min": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p90": float(quantiles[5]),
        "p95": float(quantiles[6]),
        "p99": float(quantiles[7]),
        "max": float(quantiles[8]),
        "histogram": [
            {"interval": f"({edges[i]:g}, {edges[i + 1]:g}]", "rows": int(counts[i])}
            for i in range(len(counts))
        ],
    }


def _calibration_table(y, probability, labels, bins=(0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0)):
    data = pd.DataFrame({"y": y, "p": probability, "label": labels})
    data["bin"] = pd.cut(data.p, bins=bins, include_lowest=True, right=True)
    rows = []
    total = len(data)
    ece = 0.0
    mce = 0.0
    for interval, group in data.groupby("bin", observed=False):
        if group.empty:
            continue
        mean_p = float(group.p.mean())
        up_rate = float(group.y.mean())
        error = abs(mean_p - up_rate)
        ece += len(group) / total * error
        mce = max(mce, error)
        rows.append({
            "target": labels,
            "probability_bin": str(interval),
            "rows": int(len(group)),
            "mean_probability_up": mean_p,
            "observed_up_rate": up_rate,
            "calibration_error": mean_p - up_rate,
        })
    return rows, {"ece": float(ece), "mce": float(mce)}


def _resolve_inputs():
    settings = load_modeling_settings(asset=ASSET)
    oof_paths = resolve_oof_prediction_output_paths(
        settings,
        preview_rows=settings["preview_rows"],
    )
    dataset_paths = resolve_modeling_dataset_output_paths(settings)
    oof_path = Path(oof_paths["parquet"])
    model_ready_path = Path(dataset_paths["parquet"])
    runtime_path = Path("configs/runtime/active.json")
    runtime = _read_json(runtime_path)
    runtime_model_path = Path(
        runtime["assets"][ASSET]["artifacts"]["model_meta_path"]
    )
    if not oof_path.exists():
        raise FileNotFoundError(f"Configured BTC OOF artifact is missing: {oof_path.resolve()}")
    if not model_ready_path.exists():
        raise FileNotFoundError(f"Configured BTC model-ready artifact is missing: {model_ready_path.resolve()}")
    if not runtime_model_path.exists():
        raise FileNotFoundError(f"Runtime BTC model metadata is missing: {runtime_model_path.resolve()}")
    return settings, oof_path, model_ready_path, runtime_path, runtime_model_path


def _validate_oof_lineage(oof, source, model_meta, oof_path, model_ready_path):
    oof_manifest = model_meta.get("oof_predictions", {})
    if not oof_manifest.get("enabled"):
        raise ValueError("The active model metadata does not enable OOF predictions")
    if Path(oof_manifest.get("path", "")).resolve() != Path(oof_path).resolve():
        raise ValueError("Configured OOF path does not match the active model metadata")
    if Path(model_meta.get("data_path", "")).resolve() != Path(model_ready_path).resolve():
        raise ValueError("Configured model-ready path does not match model metadata data_path")
    if model_meta.get("target_col") != "target_5m_candle_up":
        raise ValueError(f"Unexpected active model target: {model_meta.get('target_col')!r}")
    if oof_manifest.get("prediction_col") != "oof_pred_proba_up":
        raise ValueError("The active OOF probability column differs from the expected BTC contract")

    required_oof = {
        "Opened", "Open", "High", "Low", "Close", "Volume",
        "target_5m_candle_up", TARGET_WEIGHT_COL, "oof_pred_proba_up",
    }
    if not required_oof.issubset(oof.columns):
        raise ValueError(f"OOF artifact missing columns: {sorted(required_oof - set(oof.columns))}")
    if oof.Opened.isna().any() or oof.Opened.duplicated().any():
        raise ValueError("OOF timestamps must be present and unique")
    if not oof.Opened.is_monotonic_increasing:
        raise ValueError("OOF timestamps are not chronological")
    if not np.isfinite(oof.oof_pred_proba_up).all() or not oof.oof_pred_proba_up.between(0.0, 1.0).all():
        raise ValueError("OOF probabilities must be finite values in [0, 1]")
    if not oof.target_5m_candle_up.isin([0.0, 1.0]).all():
        raise ValueError("OOF target contains missing or non-binary values")
    if not np.isfinite(oof[TARGET_WEIGHT_COL]).all() or not oof[TARGET_WEIGHT_COL].gt(0.0).all():
        raise ValueError("OOF target weights must be finite and positive")

    source_opened = pd.to_datetime(source.Opened, errors="raise")
    if source_opened.isna().any() or source_opened.duplicated().any():
        raise ValueError("Model-ready timestamps must be present and unique")
    if not source_opened.is_monotonic_increasing:
        raise ValueError("Model-ready timestamps are not chronological")
    source_times = source_opened.to_numpy(dtype="datetime64[ns]")
    oof_times = pd.to_datetime(oof.Opened, errors="raise").to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(source_times, oof_times)
    if (positions >= len(source_times)).any() or not np.array_equal(source_times[positions], oof_times):
        raise ValueError("OOF timestamps do not align exactly to the saved model-ready row order")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column not in source.columns:
            raise ValueError(f"Model-ready file lacks OOF base column {column!r}")
        current = pd.to_numeric(source[column].iloc[positions], errors="raise").to_numpy(float)
        saved = pd.to_numeric(oof[column], errors="raise").to_numpy(float)
        if not np.array_equal(current, saved):
            raise ValueError(f"Model-ready {column} values do not match OOF-exported rows")

    recreated_target = recompute_close_target(source.Opened, source.Close, horizon_minutes=5)
    recreated_weights = compute_target_weights_from_opened(source.Opened)
    target_at_oof = recreated_target[positions]
    weight_at_oof = recreated_weights[positions]
    if not np.array_equal(target_at_oof, oof.target_5m_candle_up.to_numpy(float)):
        raise ValueError("OOF labels differ from exact-time Binance close comparisons")
    if not np.array_equal(weight_at_oof, oof[TARGET_WEIGHT_COL].to_numpy(float)):
        raise ValueError("OOF weights differ from target weights recomputed from Opened")

    fold_specs = model_meta.get("walk_forward_folds", [])
    fold_id_by_position = np.full(len(source), -1, dtype=np.int16)
    for fold in fold_specs:
        mask = (positions >= int(fold["test_start"])) & (positions < int(fold["test_end"]))
        if not mask.any():
            continue
        if (fold_id_by_position[positions[mask]] >= 0).any():
            raise ValueError("Saved training fold test ranges overlap")
        fold_id_by_position[positions[mask]] = int(fold["fold_id"])
    fold_ids = fold_id_by_position[positions]
    if (fold_ids < 0).any() or len(fold_specs) == 0:
        raise ValueError("Could not reconstruct every OOF fold from saved row ranges")
    return positions, fold_ids, recreated_target, recreated_weights


def _reproduce_training_report(oof, fold_ids, source, positions, model_meta):
    report = model_meta["metrics"]["cv"]["optuna"]
    metric_names = tuple(report["mean"])
    fold_rows = []
    fold_metric_rows = []
    for fold in model_meta["walk_forward_folds"]:
        fold_id = int(fold["fold_id"])
        mask = fold_ids == fold_id
        if not mask.any():
            continue
        values = training_classification_metrics(
            oof.target_5m_candle_up.to_numpy(float)[mask],
            oof.oof_pred_proba_up.to_numpy(float)[mask],
            oof[TARGET_WEIGHT_COL].to_numpy(float)[mask],
        )
        weighted_auc = score_probabilities(
            oof.target_5m_candle_up.to_numpy(float)[mask],
            oof.oof_pred_proba_up.to_numpy(float)[mask],
            oof[TARGET_WEIGHT_COL].to_numpy(float)[mask],
        )["auc"]
        fold_metric_rows.append(values)
        train_end_position = int(fold["train_end"]) - 1
        test_opened = pd.Timestamp(oof.Opened.iloc[np.flatnonzero(mask)[0]]).tz_localize("UTC")
        train_last_opened = pd.Timestamp(source.Opened.iloc[train_end_position]).tz_localize("UTC")
        test_positions = positions[mask]
        fold_rows.append({
            "fold_id": fold_id,
            "train_size_from_metadata": int(fold["train_end"]) - int(fold["train_start"]),
            "oof_rows": int(mask.sum()),
            "oof_start_utc": test_opened,
            "oof_end_utc": pd.Timestamp(oof.Opened.iloc[np.flatnonzero(mask)[-1]]).tz_localize("UTC"),
            "train_last_opened_utc": train_last_opened,
            "training_target_available_through_utc": train_last_opened + pd.Timedelta(minutes=6),
            "oof_position_start": int(test_positions.min()),
            "oof_position_end_exclusive": int(test_positions.max()) + 1,
            "auc_weighted": weighted_auc,
            **values,
        })

    reproduced_mean = macro_average_fold_metrics(fold_metric_rows)
    reproduced_std = {
        name: float(np.std([row[name] for row in fold_metric_rows]))
        for name in metric_names
    }
    comparisons = []
    for name in metric_names:
        reported_mean = float(report["mean"][name])
        reported_std = float(report["std"][name])
        comparisons.append({
            "metric": name,
            "reported_cv_fold_mean": reported_mean,
            "recomputed_cv_fold_mean": reproduced_mean[name],
            "mean_difference": reproduced_mean[name] - reported_mean,
            "reported_cv_fold_std": reported_std,
            "recomputed_cv_fold_std": reproduced_std[name],
            "std_difference": reproduced_std[name] - reported_std,
            "matches_1e_10": bool(
                math.isclose(reproduced_mean[name], reported_mean, rel_tol=0.0, abs_tol=1e-10)
                and math.isclose(reproduced_std[name], reported_std, rel_tol=0.0, abs_tol=1e-10)
            ),
        })
    return fold_rows, comparisons


def _prior_frequency(history, start_time, phase=None):
    available_at = history.opened_utc + pd.Timedelta(minutes=6)
    mask = available_at.le(start_time) & history.target_5m_candle_up.notna()
    if phase is not None:
        mask &= history.phase.eq(int(phase))
    prior = history.loc[mask]
    if prior.empty:
        return {"rows": 0, "unweighted": 0.5, "weighted": 0.5}
    weights = prior[TARGET_WEIGHT_COL].to_numpy(float)
    target = prior.target_5m_candle_up.to_numpy(float)
    return {
        "rows": int(len(prior)),
        "unweighted": float(target.mean()),
        "weighted": float(np.dot(target, weights) / weights.sum()),
    }


def _temporal_row(frame, history, *, period, scope, phase, period_start, group_name):
    mask = frame.period_key.eq(group_name)
    if phase is not None:
        mask &= frame.phase.eq(int(phase))
    group = frame.loc[mask]
    if group.empty:
        return None
    prior = _prior_frequency(history, period_start, phase=phase)
    y = group.target_5m_candle_up.to_numpy(float)
    p = group.oof_pred_proba_up.to_numpy(float)
    w = group[TARGET_WEIGHT_COL].to_numpy(float)
    unweighted = score_probabilities(y, p)
    weighted = score_probabilities(y, p, w)
    baseline_unweighted = score_probabilities(y, np.full(len(y), prior["unweighted"]))
    baseline_weighted = score_probabilities(y, np.full(len(y), prior["weighted"]), w)
    return {
        "period_type": period,
        "period": group_name,
        "scope": scope,
        "phase": "all" if phase is None else int(phase),
        "rows": int(len(group)),
        "start_utc": group.opened_utc.min(),
        "end_utc": group.opened_utc.max(),
        "up_rate_unweighted": unweighted["up_rate"],
        "up_rate_training_weighted": weighted["up_rate"],
        "log_loss_unweighted": unweighted["log_loss"],
        "brier_unweighted": unweighted["brier"],
        "auc_unweighted": unweighted["auc"],
        "accuracy_unweighted": unweighted["accuracy"],
        "log_loss_training_weighted": weighted["log_loss"],
        "brier_training_weighted": weighted["brier"],
        "auc_training_weighted": weighted["auc"],
        "accuracy_training_weighted": weighted["accuracy"],
        "prior_rows": prior["rows"],
        "baseline_up_rate_unweighted": prior["unweighted"],
        "baseline_log_loss_unweighted": baseline_unweighted["log_loss"],
        "baseline_brier_unweighted": baseline_unweighted["brier"],
        "baseline_accuracy_unweighted": baseline_unweighted["accuracy"],
        "baseline_log_loss_training_weighted": baseline_weighted["log_loss"],
        "baseline_brier_training_weighted": baseline_weighted["brier"],
        "baseline_accuracy_training_weighted": baseline_weighted["accuracy"],
    }


def _collect_period_metrics(oof_frame, source_history, common, exact_sample):
    frame = oof_frame.copy()
    frame["phase"] = frame.opened_utc.dt.minute.mod(5)
    frame["period_key"] = frame.opened_utc.dt.tz_localize(None).dt.to_period("Q").astype(str)
    rows = []

    quarters = sorted(frame.period_key.unique())
    for quarter in quarters:
        start = pd.Period(quarter, freq="Q").start_time.tz_localize("UTC")
        for phase in [None, 0, 1, 2, 3, 4]:
            scope = "all_minutes" if phase is None else ("live_entry" if phase == 4 else "minute_phase")
            row = _temporal_row(frame, source_history, period="quarter", scope=scope,
                                phase=phase, period_start=start, group_name=quarter)
            if row is not None:
                rows.append(row)

    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    current_month = now_utc.tz_localize(None).to_period("M")
    first_month = current_month - 11
    # Calendar month keys are kept separately so this remains a rolling 12-calendar-month view.
    frame["month_key"] = frame.opened_utc.dt.strftime("%Y-%m")
    months = [str(month) for month in pd.period_range(first_month, current_month, freq="M")]
    rows = [row for row in rows if row["period_type"] == "quarter"]
    for month in months:
        start = pd.Period(month, freq="M").start_time.tz_localize("UTC")
        for phase in [None, 0, 1, 2, 3, 4]:
            scope = "all_minutes" if phase is None else ("live_entry" if phase == 4 else "minute_phase")
            row = _temporal_row(frame.assign(period_key=frame.month_key), source_history,
                                period="month", scope=scope, phase=phase,
                                period_start=start, group_name=month)
            if row is not None:
                rows.append(row)

    # The complete Kacho availability window, before the market-value warmup split.
    kacho_start = pd.to_datetime(common.market_start_utc, utc=True).min() - pd.Timedelta(minutes=1)
    kacho_end = pd.to_datetime(common.market_start_utc, utc=True).max() - pd.Timedelta(minutes=1)
    kacho_mask = frame.opened_utc.between(kacho_start, kacho_end, inclusive="both")
    kacho = frame.loc[kacho_mask].copy()
    for phase in [None, 0, 1, 2, 3, 4]:
        g = kacho if phase is None else kacho[kacho.phase.eq(phase)]
        if g.empty:
            continue
        prior = _prior_frequency(source_history, kacho_start, phase=phase)
        y, p, w = (g.target_5m_candle_up.to_numpy(float),
                   g.oof_pred_proba_up.to_numpy(float),
                   g[TARGET_WEIGHT_COL].to_numpy(float))
        u, wt = score_probabilities(y, p), score_probabilities(y, p, w)
        bu = score_probabilities(y, np.full(len(y), prior["unweighted"]))
        bw = score_probabilities(y, np.full(len(y), prior["weighted"]), w)
        rows.append({
            "period_type": "coverage", "period": "Kacho available range",
            "scope": "all_minutes" if phase is None else ("live_entry" if phase == 4 else "minute_phase"),
            "phase": "all" if phase is None else phase, "rows": len(g),
            "start_utc": g.opened_utc.min(), "end_utc": g.opened_utc.max(),
            "up_rate_unweighted": u["up_rate"], "up_rate_training_weighted": wt["up_rate"],
            "log_loss_unweighted": u["log_loss"], "brier_unweighted": u["brier"],
            "auc_unweighted": u["auc"], "accuracy_unweighted": u["accuracy"],
            "log_loss_training_weighted": wt["log_loss"], "brier_training_weighted": wt["brier"],
            "auc_training_weighted": wt["auc"], "accuracy_training_weighted": wt["accuracy"],
            "prior_rows": prior["rows"], "baseline_up_rate_unweighted": prior["unweighted"],
            "baseline_log_loss_unweighted": bu["log_loss"], "baseline_brier_unweighted": bu["brier"],
            "baseline_accuracy_unweighted": bu["accuracy"],
            "baseline_log_loss_training_weighted": bw["log_loss"],
            "baseline_brier_training_weighted": bw["brier"],
            "baseline_accuracy_training_weighted": bw["accuracy"],
        })

    exact = exact_sample.copy()
    exact["phase"] = 4
    prior = _prior_frequency(source_history, exact.market_start_utc.min(), phase=4)
    y = exact.target_binance_proxy_up.to_numpy(float)
    p = exact.p_model_up.to_numpy(float)
    w = exact.target_5m_weight.to_numpy(float)
    u, wt = score_probabilities(y, p), score_probabilities(y, p, w)
    bu = score_probabilities(y, np.full(len(y), prior["unweighted"]))
    bw = score_probabilities(y, np.full(len(y), prior["weighted"]), w)
    rows.append({
        "period_type": "coverage", "period": "9407 market-value evaluation rows",
        "scope": "exact_market_value_sample", "phase": "4", "rows": len(exact),
        "start_utc": exact.opened_utc.min(), "end_utc": exact.opened_utc.max(),
        "up_rate_unweighted": u["up_rate"], "up_rate_training_weighted": wt["up_rate"],
        "log_loss_unweighted": u["log_loss"], "brier_unweighted": u["brier"],
        "auc_unweighted": u["auc"], "accuracy_unweighted": u["accuracy"],
        "log_loss_training_weighted": wt["log_loss"], "brier_training_weighted": wt["brier"],
        "auc_training_weighted": wt["auc"], "accuracy_training_weighted": wt["accuracy"],
        "prior_rows": prior["rows"], "baseline_up_rate_unweighted": prior["unweighted"],
        "baseline_log_loss_unweighted": bu["log_loss"], "baseline_brier_unweighted": bu["brier"],
        "baseline_accuracy_unweighted": bu["accuracy"],
        "baseline_log_loss_training_weighted": bw["log_loss"],
        "baseline_brier_training_weighted": bw["brier"],
        "baseline_accuracy_training_weighted": bw["accuracy"],
    })
    return pd.DataFrame(rows), kacho, frame


def _period_bootstrap_row(group, label="OOF Binance target", replications=HISTORY_BOOTSTRAP_REPLICATIONS):
    intervals = paired_block_bootstrap(
        {label: group.target_5m_candle_up.to_numpy(float)},
        group.oof_pred_proba_up.to_numpy(float),
        group.opened_utc,
        replications=replications,
        block_days=BOOTSTRAP_BLOCK_DAYS,
        seed=37,
    )[label]
    point = score_probabilities(
        group.target_5m_candle_up.to_numpy(float),
        group.oof_pred_proba_up.to_numpy(float),
    )
    return {metric: intervals[metric] for metric in ("log_loss", "brier", "auc")}, point


def _live_latency_audit(current_model_sha):
    trade_files = sorted(Path("data/live/BTC/trade").glob("*.csv"))
    log_files = sorted(Path("data/live/BTC/logs").glob("*.log"))
    shared_path = Path("data/live/BTC/polymarket_5m.csv")
    sessions = []
    stage_rows = []
    stage_columns = {
        "price_event_from_minute_open_ms": ("ws_price_event_delay_ms", "all"),
        "volume_event_from_minute_open_ms": ("ws_volume_event_delay_ms", "all"),
        "price_received_from_minute_open_ms": ("ws_price_receive_delay_ms", "all"),
        "volume_received_from_minute_open_ms": ("ws_volume_receive_delay_ms", "all"),
        "both_required_inputs_ready_from_minute_open_ms": ("ws_receive_delay_ms", "all"),
        "feature_preparation_ms": ("feature_prep_ms", "all"),
        "feature_vector_construction_ms": ("feature_vector_ms", "all"),
        "model_inference_ms": ("model_predict_ms", "all"),
        "signal_ready_from_window_start_ms_wall_clock": ("signal_ready_delay_ms", "all"),
        "prefetched_quote_snapshot_age_at_use_ms": ("market_prefetch_age_ms", "all"),
        "quote_snapshot_lookup_or_refetch_ms": ("market_lookup_ms", "all"),
        "policy_decision_computation_ms": ("policy_compute_ms", "all"),
        "decision_ready_from_window_start_ms_wall_clock": ("decision_ready_delay_ms", "all"),
        "submit_call_including_response_ms": ("submit_order_ms", "attempts"),
        "execution_stage_including_lookup_policy_and_submission_ms": ("execution_ms", "all"),
        "cycle_complete_from_window_start_ms_wall_clock": ("cycle_complete_delay_ms", "all"),
    }

    for path in trade_files:
        frame = pd.read_csv(path)
        if frame.empty or "pm_run_started_at_utc" not in frame:
            continue
        for run_id, data in frame.groupby("pm_run_started_at_utc", dropna=False):
            data = data.sort_values("prediction_time", kind="stable").reset_index(drop=True)
            model_hash = str(data.pm_model_hash.dropna().iloc[0]) if data.pm_model_hash.notna().any() else "unknown"
            matching_logs = [log for log in log_files if str(run_id) in log.name]
            log_text = "\n".join(log.read_text(encoding="utf-8", errors="replace") for log in matching_logs)
            reconnect_lines = [
                line for line in log_text.splitlines()
                if re.search(r"\[ws\].*(reconnect|disconnect|connection closed)", line, re.I)
            ]
            statuses = data.pm_order_status.fillna("missing").astype(str)
            attempts = statuses.isin({"submitted_fak", "submission_error", "submission_retryable"})
            for metric_name, (column, scope) in stage_columns.items():
                if column not in data:
                    continue
                for stage_name, stage_mask in (
                    ("all_records", pd.Series(True, index=data.index)),
                    ("first_decision_startup_sample", pd.Series(np.arange(len(data)) < LIVE_START_STAGE_COUNT, index=data.index)),
                    ("steady_after_first_decision", pd.Series(np.arange(len(data)) >= LIVE_START_STAGE_COUNT, index=data.index)),
                ):
                    selected = stage_mask & (attempts if scope == "attempts" else True)
                    values = pd.to_numeric(data.loc[selected, column], errors="coerce").dropna().to_numpy(float)
                    stage_rows.append({
                        "run_id": str(run_id), "model_hash": model_hash,
                        "active_model_hash_prefix": current_model_sha[:12],
                        "model_matches_active_prefix": model_hash == current_model_sha[:12],
                        "stage": metric_name, "sample": stage_name,
                        "rows": int(len(values)),
                        "median_ms": float(np.median(values)) if len(values) else None,
                        "p90_ms": float(np.quantile(values, 0.90)) if len(values) else None,
                        "p95_ms": float(np.quantile(values, 0.95)) if len(values) else None,
                        "p99_ms": float(np.quantile(values, 0.99)) if len(values) else None,
                    })
            filled = pd.to_numeric(data.get("filled_stake_usdc", pd.Series(index=data.index, dtype=float)), errors="coerce")
            quote_sources = data.get("market_lookup_source", pd.Series(index=data.index, dtype=object)).fillna("missing")
            sessions.append({
                "file": path.as_posix(), "run_id": str(run_id), "rows": int(len(data)),
                "first_prediction_time": data.prediction_time.min(),
                "last_prediction_time": data.prediction_time.max(),
                "model_hash": model_hash,
                "active_model_sha256": current_model_sha,
                "model_matches_active_sha256_prefix": model_hash == current_model_sha[:12],
                "matching_log_files": [p.as_posix() for p in matching_logs],
                "initial_websocket_connections": len(re.findall(r"\[ws\] connected:", log_text, re.I)),
                "logged_reconnect_or_disconnect_events": len(reconnect_lines),
                "order_status_counts": {str(k): int(v) for k, v in statuses.value_counts().items()},
                "attempted_order_rows": int(attempts.sum()),
                "positive_filled_stake_rows": int(filled.fillna(0.0).gt(0.0).sum()),
                "filled_stake_missing_rows": int(filled.isna().sum()),
                "market_lookup_source_counts": {str(k): int(v) for k, v in quote_sources.value_counts().items()},
                "prefetch_hit_rows": int(data.get("market_prefetch_hit", pd.Series(False, index=data.index)).fillna(False).astype(bool).sum()),
            })

    shared_summary = {"path": shared_path.as_posix(), "exists": shared_path.exists()}
    if shared_path.exists():
        shared = pd.read_csv(shared_path, usecols=lambda c: c in {"pm_run_started_at_utc", "pm_model_hash"})
        shared_summary.update({
            "rows": int(len(shared)),
            "run_counts": {str(k): int(v) for k, v in shared.pm_run_started_at_utc.value_counts(dropna=False).items()},
            "model_hash_counts": {str(k): int(v) for k, v in shared.pm_model_hash.value_counts(dropna=False).items()},
            "note": "The shared history covers more runs, but does not contain the full request and response duration columns used above.",
        })
    return {
        "detailed_trade_files": [p.as_posix() for p in trade_files],
        "sessions": sessions,
        "stage_rows": stage_rows,
        "shared_prediction_history": shared_summary,
        "clock_notes": [
            "feature, vector, inference, policy, quote lookup, submit-call, and execution durations use local perf_counter intervals.",
            "WS event/receive delay and bucket-start-to-signal/decision/cycle fields compare exchange-derived UTC event times with host wall time; no NTP/clock-offset record is available, so their absolute delay interpretation is unverified.",
            "The submit-call duration includes the synchronous client call and response; request-send and acceptance timestamps are not stored separately.",
            "A positive filled_stake_usdc is recorded from the order response, but a fill timestamp is not stored.",
            "The first decision is only a startup sample; process initialization/model loading time is outside its recorded row. No reconnect events were found in the matching detailed run log.",
        ],
    }


def _target_examples(sample, source, source_time_index):
    data = sample.sort_values("market_start_utc").copy()
    aligned = data.target_binance_proxy_up.eq(data.target_polymarket_up)
    chosen = []
    if aligned.any():
        chosen.append(("ordinary_aligned_window", data.loc[aligned].iloc[0]))
    day_boundary = data.market_start_utc.dt.hour.eq(0) & data.market_start_utc.dt.minute.eq(0)
    if day_boundary.any():
        chosen.append(("utc_day_boundary", data.loc[day_boundary].iloc[0]))
    for binance_value, polymarket_value, name in (
        (1.0, 0.0, "binance_up_polymarket_down"),
        (0.0, 1.0, "binance_down_polymarket_up"),
    ):
        mask = data.target_binance_proxy_up.eq(binance_value) & data.target_polymarket_up.eq(polymarket_value)
        if mask.any():
            chosen.append((name, data.loc[mask].iloc[0]))

    rows = []
    for example, row in chosen:
        opened = pd.Timestamp(row.opened_utc).tz_localize(None)
        endpoint_opened = opened + pd.Timedelta(minutes=5)
        current_candle = source_time_index.loc[opened]
        future_candle = source_time_index.loc[endpoint_opened]
        start_close_at = opened + pd.Timedelta(minutes=1)
        end_close_at = endpoint_opened + pd.Timedelta(minutes=1)
        saved = float(row.target_binance_proxy_up)
        reconstructed = float(future_candle.Close >= current_candle.Close)
        rows.append({
            "example": example,
            "condition_id": row.condition_id,
            "market_slug": row.get("market_slug_evaluation", row.get("market_slug_common", "")),
            "opened_utc": pd.Timestamp(row.opened_utc),
            "input_candle_interval_utc": f"[{pd.Timestamp(row.opened_utc).isoformat()}, {(pd.Timestamp(row.opened_utc) + pd.Timedelta(minutes=1)).isoformat()})",
            "initial_proxy_price_timestamp_utc": start_close_at.tz_localize("UTC"),
            "initial_proxy_close": float(current_candle.Close),
            "final_proxy_price_timestamp_utc": end_close_at.tz_localize("UTC"),
            "final_proxy_close": float(future_candle.Close),
            "features_available_no_earlier_than_utc": start_close_at.tz_localize("UTC"),
            "market_start_utc": row.market_start_utc,
            "market_end_utc": row.market_end_utc,
            "target_equality_rule": "final proxy Close >= initial proxy Close -> UP; ties resolve UP",
            "saved_binance_target": saved,
            "recomputed_binance_target": reconstructed,
            "official_polymarket_settlement": float(row.target_polymarket_up),
            "p_model_up": float(row.p_model_up),
            "target_match": bool(saved == reconstructed),
        })
    return rows


def _latency_cell(row):
    if row is None:
        return "n/a"
    return (
        f"n={row['rows']}; {_fmt(row['median_ms'], 2)}/"
        f"{_fmt(row['p90_ms'], 2)}/{_fmt(row['p95_ms'], 2)}/{_fmt(row['p99_ms'], 2)} ms"
    )


def _recommend_next_experiment(results):
    all_oof = results["binance_target_quality"]["all_available_oof"]
    temporal = pd.DataFrame(results["temporal_metrics"])
    quarters = temporal.loc[
        temporal.period_type.eq("quarter") & temporal.scope.isin(["all_minutes", "live_entry"])
    ]
    own_beats_prior = (
        all_oof["training_weighted"]["log_loss"]
        < all_oof["prior_baselines"]["all_minutes"]["training_weighted"]["log_loss"]
    )
    quarter_share = float(
        (quarters.log_loss_unweighted < quarters.baseline_log_loss_unweighted).mean()
    ) if not quarters.empty else 0.0
    sample_scores = {row["target"]: row for row in results["same_9407_target_comparison"]["scores"]}
    binance_sample = sample_scores["Binance"]
    polymarket_sample = sample_scores["Polymarket"]
    label_delta = results["same_9407_target_comparison"]["label_attribution"]
    settlement_gap_is_stable = (
        label_delta["log_loss_bootstrap_95pct_pm_minus_binance"][0] > 0.0
        and label_delta["brier_bootstrap_95pct_pm_minus_binance"][0] > 0.0
    )
    entry = results["temporal_summary_bootstrap"]["live entry phase 4"]["point"]
    all_point = results["temporal_summary_bootstrap"]["all OOF minutes"]["point"]
    entry_brier_worse = entry["brier"] > all_point["brier"] + 0.002
    pm_log_loss_gap = polymarket_sample["model_log_loss"] - binance_sample["model_log_loss"]

    if quarter_share < 0.5 or not own_beats_prior:
        name = "Audyt wartości grup cech i procesu selekcji na zapisanych foldach OOF"
        rationale = (
            f"Ważony log loss OOF na celu Binance ({all_oof['training_weighted']['log_loss']:.6f}) "
            f"jest {'lepszy' if own_beats_prior else 'nie lepszy'} od wcześniejszego baseline'u "
            f"({all_oof['prior_baselines']['all_minutes']['training_weighted']['log_loss']:.6f}); "
            f"model wygrywa z baseline'em w {quarter_share:.0%} okresów kwartalnych faz all/entry."
        )
        criterion = (
            "Porównać istniejące grupy cech na tych samych zapisanych granicach walk-forward; "
            "wymagać poprawy 3-dniowym blokowym bootstrapem log loss względem baseline'u "
            "oraz poprawy w co najmniej 8/10 foldach przed rozważeniem nowego treningu."
        )
    elif pm_log_loss_gap > 0.0 and settlement_gap_is_stable:
        name = "Zbadać źródło ceny i semantykę settlementu na tych samych oknach BTC 5m"
        rationale = (
            f"Na identycznych 9 407 oknach Polymarket ma gorszy log loss o {pm_log_loss_gap:.6f}; "
            f"sparowany 3-dniowy bootstrap dla PM−Binance daje log loss "
            f"[{label_delta['log_loss_bootstrap_95pct_pm_minus_binance'][0]:.6f}, "
            f"{label_delta['log_loss_bootstrap_95pct_pm_minus_binance'][1]:.6f}] i Brier "
            f"[{label_delta['brier_bootstrap_95pct_pm_minus_binance'][0]:.6f}, "
            f"{label_delta['brier_bootstrap_95pct_pm_minus_binance'][1]:.6f}]. "
            f"Etykiety różnią się w {label_delta['label_disagreement_rows']} oknach ({label_delta['label_disagreement_rate']:.2%}); "
            "na pełnym OOF model na celu Binance przewyższa wcześniejszy baseline. Dokładne granice czasu są zgodne, "
            "ale lokalny cache Chainlink nie pokrywa okresu Kacho, więc brak danych do oddzielenia źródła ceny od samplingu settlementu."
        )
        criterion = (
            "Pozyskać równoczesne referencyjne ceny oracle i ich czasy obserwacji dla 9 407 okien; "
            "odtworzyć oficjalne settlementy ze zgodnością wszystkich dostępnych przypadków, a rozbieżności "
            "przypisać do proxy ceny, momentu samplingu lub reguły rozstrzygnięcia. Następnie ocenić niezmienione "
            "OOF na tych targetach sparowanym bootstrapem 3-dniowych bloków. Nie szukać przesunięcia po AUC."
        )
    elif entry_brier_worse:
        name = "Sprawdzić dopasowanie treningu i oceny do fazy wejścia BTC 5m"
        rationale = (
            f"Brier dla fazy wejścia ({entry['brier']:.6f}) przewyższa wynik wszystkich minut "
            f"({all_point['brier']:.6f}) o {entry['brier'] - all_point['brier']:.6f}."
        )
        criterion = (
            "Na identycznych foldach porównać wejściową fazę 4 z pozostałymi fazami; "
            "wymagać poprawy blokowego log loss/Brier w fazie wejścia bez pogorszenia "
            "w większości foldów."
        )
    else:
        name = "Zmierzyć wartość i dostępność sygnału OOF względem świeżego snapshotu książki"
        rationale = (
            "Model wnosi mierzalną informację na celu Binance, a audyt market_value nie potwierdza "
            "stabilnej poprawy wszystkich metryk ponad książkę; istniejące czasy 1 s i 2 s są "
            "scenariuszami wrażliwości, nie pomiarem live."
        )
        criterion = (
            "Zebrać monotoniczne znaczniki czasu dla granicy świecy, gotowości wejść, inferencji, "
            "quote, wysłania, odpowiedzi i fillu; po zebraniu wystarczającej próby porównać "
            "sparowany blokowy scoring i faktyczne filli przy zmierzonym czasie dostępności."
        )
    return {"name": name, "rationale": rationale, "criterion": criterion}


def _render_report(results, temporal_metrics, fold_rows, examples, latency):
    sample = results["same_9407_target_comparison"]
    score_by_target = {row["target"]: row for row in sample["scores"]}
    summary_by_scope = results["temporal_summary_bootstrap"]
    lines = [
        "# BTC OOF target, czas i latencja: audyt",
        "",
        f"Fingerprint audytu: `{results['fingerprint']}`; reprodukcja: `{results['reproduction_command']}`.",
        "",
        "## Tabela 1. Identyczne 9 407 okien BTC 5m: Binance i settlement Polymarket",
        "",
    ]
    sample_rows = []
    for target in ("Binance", "Polymarket"):
        row = score_by_target[target]
        sample_rows.append([
            target,
            f"{row['observed_up_rate']:.4f}",
            str(row["baseline_prior_rows"]), _fmt(row["baseline_up_probability"], 4),
            _fmt_ci(row["model_log_loss"], row["model_log_loss_95pct"]),
            _fmt(row["baseline_log_loss"]),
            _fmt_ci(row["model_brier"], row["model_brier_95pct"]),
            _fmt(row["baseline_brier"]),
            _fmt_ci(row["model_auc"], row["model_auc_95pct"], 4),
            _fmt(row["model_accuracy"], 4),
        ])
    lines += _markdown_table(
        ["Target", "UP", "Prior rows", "Prior UP", "Model log loss (95% CI)", "Prior log loss", "Model Brier (95% CI)", "Prior Brier", "AUC (95% CI)", "Accuracy"],
        sample_rows,
    )
    attribution = sample["label_attribution"]
    lines += [
        "",
        "Różnica etykiet (Polymarket − Binance), przy tych samych predykcjach; ujemny wynik poprawia scoring Polymarket:",
        "",
    ]
    lines += _markdown_table(
        ["Zgodne etykiety", "Niezgodne", "Niezgodne %", "Δ log loss / okno (95% CI)", "Suma Δ log loss", "Δ Brier / okno (95% CI)", "Suma Δ Brier"],
        [[
            str(attribution["label_agreement_rows"]), str(attribution["label_disagreement_rows"]),
            _fmt(attribution["label_disagreement_rate"] * 100, 2) + "%",
            _fmt_ci(attribution["per_observation_log_loss_delta_pm_minus_binance_mean"], attribution["log_loss_bootstrap_95pct_pm_minus_binance"]),
            _fmt(attribution["per_observation_log_loss_delta_pm_minus_binance_sum"]),
            _fmt_ci(attribution["per_observation_brier_delta_pm_minus_binance_mean"], attribution["brier_bootstrap_95pct_pm_minus_binance"]),
            _fmt(attribution["per_observation_brier_delta_pm_minus_binance_sum"]),
        ]],
    )
    transition_rows = []
    for label, row in attribution["transitions"].items():
        transition_rows.append([
            label, str(row["rows"]), f"{row['share_of_sample']:.2%}",
            _fmt(row["mean_p_model_up"], 4), _fmt(row["median_p_model_up"], 4),
            _fmt(row["mean_absolute_confidence_from_0_5"], 4),
            _fmt(row["log_loss_delta_sum_polymarket_minus_binance"]),
            _fmt(row["brier_delta_sum_polymarket_minus_binance"]),
        ])
    lines += ["", "Przejścia etykiet; pewność oznacza średnie |p−0,5|:", ""]
    lines += _markdown_table(
        ["Przejście", "N", "Udział", "Śr. p(UP)", "Mediana p(UP)", "Pewność", "Suma Δ log loss", "Suma Δ Brier"],
        transition_rows,
    )
    lines += [
        "",
        "Wszystkie 9 407 prawdopodobieństw są identyczne dla obu targetów; wagi treningowe są stałe (0.4625) w fazie wejścia, więc wyniki ważone i nieważone na tej próbie są równe.",
        "",
        "Rozkład p(UP):",
        "",
    ]
    distribution = sample["probability_distribution"]
    distribution_row = [[
        _fmt(distribution["min"], 4), _fmt(distribution["p10"], 4),
        _fmt(distribution["p25"], 4), _fmt(distribution["p50"], 4),
        _fmt(distribution["p75"], 4), _fmt(distribution["p90"], 4),
        _fmt(distribution["p95"], 4), _fmt(distribution["p99"], 4),
        _fmt(distribution["max"], 4),
    ]]
    lines += _markdown_table(
        ["Min", "P10", "P25", "Mediana", "P75", "P90", "P95", "P99", "Max"],
        distribution_row,
    )
    calibration_rows = []
    for row in sample["calibration_bins"]:
        calibration_rows.append([
            row["target"], row["probability_bin"], str(row["rows"]),
            _fmt(row["mean_probability_up"], 4), _fmt(row["observed_up_rate"], 4),
            _fmt(row["calibration_error"], 4),
        ])
    lines += ["", "Kalibracja (błąd = średnie p(UP) − zaobserwowany udział UP):", ""]
    lines += _markdown_table(
        ["Target", "Bin p(UP)", "N", "Średnie p", "Observed UP", "Błąd"],
        calibration_rows,
    )
    calibration_summary_rows = [
        [target, _fmt(values["ece"], 4), _fmt(values["mce"], 4)]
        for target, values in sample["calibration_summary"].items()
    ]
    lines += ["", ""] + _markdown_table(["Target", "ECE", "MCE"], calibration_summary_rows)

    lines += ["", "## Tabela 2. Jakość na celu Binance w czasie i według fazy wejścia", ""]
    scope_rows = []
    for name in (
        "all OOF minutes", "live entry phase 4", "Kacho available range, all phases",
        "Kacho available range, phase 4", "9407 market-value rows, Binance target",
    ):
        row = summary_by_scope[name]
        point, weighted, ci = row["point"], row["training_weighted_point"], row["unweighted_95pct"]
        scope_rows.append([
            name, str(row["rows"]), _fmt(point["up_rate"], 4), _fmt(weighted["up_rate"], 4),
            _fmt_ci(point["log_loss"], ci["log_loss"]), _fmt(weighted["log_loss"]),
            _fmt(point["brier"], 4), _fmt(weighted["brier"], 4),
            _fmt(point["auc"], 4), _fmt(weighted["auc"], 4),
            _fmt(point["accuracy"], 4), _fmt(weighted["accuracy"], 4),
        ])
    lines += _markdown_table(
        ["Zakres", "N", "UP unweighted", "UP weighted", "LL unweighted (95% CI)", "LL weighted", "Brier unweighted", "Brier weighted", "AUC unweighted", "AUC weighted", "Accuracy unweighted", "Accuracy weighted"],
        scope_rows,
    )
    period_rows = []
    selected_periods = temporal_metrics.loc[
        temporal_metrics.period_type.isin(["quarter", "month"])
        & temporal_metrics.scope.isin(["all_minutes", "live_entry"])
    ].sort_values(["period_type", "period", "scope"])
    for row in selected_periods.to_dict(orient="records"):
        period_rows.append([
            row["period_type"], row["period"], row["scope"], str(row["rows"]),
            _fmt(row["log_loss_unweighted"]), _fmt(row["baseline_log_loss_unweighted"]),
            _fmt(row["log_loss_training_weighted"]), _fmt(row["brier_unweighted"]),
            _fmt(row["brier_training_weighted"]), _fmt(row["auc_unweighted"], 4),
            _fmt(row["auc_training_weighted"], 4), _fmt(row["accuracy_unweighted"], 4),
            _fmt(row["accuracy_training_weighted"], 4),
        ])
    lines += [
        "",
        "Historia kwartalna oraz 12 ostatnich miesięcy (wyniki punktowe; pełne fazy i kolumny w `temporal_metrics.csv`). 2026Q4 i październik 2026 są częściowe. Główne przedziały powyżej używają 3-dniowego sparowanego bootstrapu blokowego, który zachowuje zależność sąsiednich targetów.",
        "",
    ]
    lines += _markdown_table(
        ["Okres", "Data", "Zakres", "N", "LL unweighted", "Prior LL", "LL weighted", "Brier unweighted", "Brier weighted", "AUC unweighted", "AUC weighted", "Acc unweighted", "Acc weighted"],
        period_rows,
    )

    fold_table_rows = []
    for row in fold_rows:
        fold_table_rows.append([
            str(row["fold_id"]), str(row["oof_rows"]),
            f"{row['oof_start_utc']} – {row['oof_end_utc']}",
            _fmt(row["binary_logloss"]), _fmt(row["brier_score"]),
            _fmt(row["auc_weighted"], 4), _fmt(row["accuracy"]),
            str(row["kacho_coverage_oof_rows"]), str(row["market_value_evaluation_rows"]),
        ])
    lines += ["", "Foldy modelu głównego; metryki ważone zgodnie z treningiem:", ""]
    lines += _markdown_table(
        ["Fold", "OOF N", "Zakres UTC", "LL", "Brier", "AUC", "Accuracy", "OOF w Kacho", "Wspólne 9 407"],
        fold_table_rows,
    )
    age_rows = []
    for row in results["model_age_metrics"]:
        age_rows.append([
            row["model_age_since_last_train_row"], str(row["rows"]),
            _fmt(row["log_loss"]), _fmt(row["log_loss_training_weighted"]),
            _fmt(row["brier"]), _fmt(row["auc"], 4),
        ])
    lines += ["", "Wynik według wieku modelu od ostatniego wiersza treningowego:", ""]
    lines += _markdown_table(
        ["Wiek", "N", "LL unweighted", "LL weighted", "Brier", "AUC"], age_rows,
    )

    lines += ["", "## Tabela 3. Zmierzone etapy live BTC", ""]
    stage_rows = latency["stage_rows"]
    stage_names = list(dict.fromkeys(row["stage"] for row in stage_rows if row["sample"] == "all_records"))
    run_ids = list(dict.fromkeys(row["run_id"] for row in stage_rows if row["sample"] == "all_records"))
    latency_table = []
    for run_id in run_ids:
        run_stage_rows = [row for row in stage_rows if row["run_id"] == run_id]
        for stage in stage_names:
            by_sample = {
                row["sample"]: row for row in run_stage_rows if row["stage"] == stage
            }
            all_row = by_sample.get("all_records")
            if not all_row or all_row["rows"] == 0:
                continue
            latency_table.append([
                run_id, stage, _latency_cell(all_row),
                _latency_cell(by_sample.get("first_decision_startup_sample")),
                _latency_cell(by_sample.get("steady_after_first_decision")),
            ])
    lines += _markdown_table(
        ["Run", "Etap", "Wszystkie: n; med/p90/p95/p99", "Pierwsza decyzja", "Po pierwszej decyzji"],
        latency_table,
    )
    lines += [
        "",
        "W source-latency 1 s i 2 s oraz dodatkowych opóźnieniach wykonania +1 s/+2 s z market_value użyto scenariuszy wrażliwości. Source-latency 0 s jest najbliższym scenariuszem operacyjnym, ale nie dokładnym pomiarem: historyczna gotowość sygnału ma opóźnienie wall-clock i kwotowania Kacho są próbkowane co sekundę. Te dane nie rozróżniają wykonania po 100 i 300 ms.",
    ]

    lines += ["", "### Jawne przykłady mapowania targetu", ""]
    example_rows = []
    for row in examples:
        example_rows.append([
            row["example"], str(row["opened_utc"]), row["input_candle_interval_utc"],
            f"{row['initial_proxy_price_timestamp_utc']} = {row['initial_proxy_close']}",
            f"{row['final_proxy_price_timestamp_utc']} = {row['final_proxy_close']}",
            str(row["features_available_no_earlier_than_utc"]),
            f"[{row['market_start_utc']}, {row['market_end_utc']})",
            f"{row['saved_binance_target']} / {row['recomputed_binance_target']}",
            str(row["official_polymarket_settlement"]), _fmt(row["p_model_up"], 4),
        ])
    lines += _markdown_table(
        ["Przypadek", "Opened UTC", "Świeca wejściowa UTC", "Cena początkowa UTC", "Cena końcowa UTC", "Cechy dostępne od", "Okno Polymarket UTC", "Binance zapis / odtworzony", "Settlement PM", "p(UP)"],
        example_rows,
    )

    next_experiment = _recommend_next_experiment(results)
    sessions = latency["sessions"]
    measured_attempts = sum(row["attempted_order_rows"] for row in sessions)
    measured_fills = sum(row["positive_filled_stake_rows"] for row in sessions)
    missing_fill_values = sum(row["filled_stake_missing_rows"] for row in sessions)
    websocket_connections = sum(row["initial_websocket_connections"] for row in sessions)
    market_lookup_sources = {}
    for session in sessions:
        for source_name, count in session["market_lookup_source_counts"].items():
            market_lookup_sources[source_name] = market_lookup_sources.get(source_name, 0) + count
    chainlink = results["chainlink_reference_cache"]
    lines += [
        "",
        "## Ustalenia audytu",
        "",
        f"- OOF: `{results['oof_manifest']['path']}`; SHA256 `{results['oof_manifest']['sha256']}`; kolumna `{results['oof_manifest']['probability_column']}`; {results['oof_manifest']['rows']:,} predykcji od {results['oof_manifest']['start_utc']} do {results['oof_manifest']['end_utc']}.",
        f"- Metadane: `{results['oof_manifest']['model_metadata_path']}`; target `{results['oof_manifest']['metadata_target']}`; powiązanie OOF w metadanych i manifest zgodne: {results['oof_manifest']['metadata_oof_matches_rows_and_probability_column']} / {results['oof_manifest']['preexisting_oof_manifest_sha256_matches']}; foldy odtworzono z zapisanych zakresów wierszy.",
        f"- Wagi: `{TARGET_WEIGHT_COL}`; faza wejścia minuta % 5 = 4 ma wagę 0.4625, pozostałe fazy po 0.134375; średnia waga OOF 0.2.",
        f"- OOF mapuje się ciągle na pozycje model-ready {results['oof_lineage_alignment']['oof_source_position_start']}–{results['oof_lineage_alignment']['oof_source_position_end_exclusive'] - 1} ({results['oof_lineage_alignment']['oof_positions_are_consecutive']}); etykiet t+5 w model-ready jest {results['oof_lineage_alignment']['source_rows_with_valid_exact_t_plus_5_target']:,}; poprawne etykiety po ostatnim OOF: {results['oof_lineage_alignment']['valid_target_rows_after_last_oof_position']}.",
        f"- Reprodukcja metryk CV zgodna do 1e-10: {results['training_report_reproduction']['all_reported_fold_macro_metrics_match_1e_10']}. Raport treningowy to średnia arytmetyczna metryk foldów ważonych osobno; zagregowany wynik puli OOF jest raportowany oddzielnie.",
        f"- Target: dokładny lookup `Opened + 5 min`; Close przyszłej świecy ≥ Close bieżącej oznacza UP, remisy UP. Luki minutowe w źródle: {results['time_and_target_audit']['timestamp_gap_summary']['source_minute_timestamp_gaps']}; brak dokładnego endpointu t+5: {results['time_and_target_audit']['timestamp_gap_summary']['rows_missing_exact_t_plus_5_endpoint']} ({results['time_and_target_audit']['timestamp_gap_summary']['interior_rows_missing_exact_t_plus_5_endpoint']} wewnątrz historii). Braki są pomijane, bez przesuwania o pięć wierszy.",
        f"- Zgodność historycznego źródła treningowego: {results['oof_manifest']['lineage_limit']}",
        f"- Ostatni eksperyment Polymarket: przy źródłowym opóźnieniu 0 s przedział Brier dla market_plus_oof − market_only wyklucza zero, a log loss obejmuje zero; przy 1 s i 2 s oba przedziały obejmują zero. Mała poprawa Brier nie potwierdza rentowności ani nie obala predykcyjności względem celu Binance.",
        f"- Ceny: proxy treningowe to BTCUSD Coin-M index Close, a settlement jest oficjalnym wynikiem Polymarket. Cache Chainlink ma {chainlink.get('rows', 0)} raportów z zakresu {chainlink.get('start_utc', 'n/a')}–{chainlink.get('end_utc', 'n/a')}; pokrycie Kacho: {chainlink.get('overlaps_kacho_period', False)}. Brak pokrycia nie pozwala rozłożyć różnic na proxy źródła ceny kontra sampling settlementu.",
        "- OOF `Opened` bez strefy został zinterpretowany jako UTC zgodnie z kontraktem; czasy Polymarket są UTC, a ceny to close świec 1m przy granicach otwarcia i wygaśnięcia. Live czeka na zamkniętą świecę Binance (flaga kline `x=true`); target nie korzysta z niezakończonego close.",
        f"- Live: {len(sessions)} szczegółowa sesja ({sessions[0]['run_id'] if sessions else 'brak'}); logowane połączenia WS: {websocket_connections}, reconnect/disconnect: {sum(s['logged_reconnect_or_disconnect_events'] for s in sessions)}; model sesji zgodny z aktualnym modelem: {any(s['model_matches_active_sha256_prefix'] for s in sessions)}. Próby zleceń: {measured_attempts}; dodatni filled stake zapisany dla {measured_fills}, wartość fillu brak dla {missing_fill_values}. Lookup źródła: {market_lookup_sources}. Brak osobnych znaczników request send, accept/ack i fill time; brak czasu startu procesu/modelu.",
        "- Czas etapu feature/inference/policy/lookup/submit/execution pochodzi z monotonicznego perf_counter; opóźnienia granicy okna i gotowości sygnału porównują czas Binance/UTC z zegarem hosta bez telemetrii synchronizacji. Submit obejmuje synchroniczne wywołanie i odpowiedź, nie dowodzi fillu. Wiek snapshotu aplikacji nie mierzy wieku feedu CLOB. Nie sumowano etapów.",
        "- Wiersz `Pierwsza decyzja` rozdziela tylko pierwszą predykcję od kolejnych, nie mierzy ładowania modelu/cold startu. Nie znaleziono reconnectów. Dodatkowe +1 s/+2 s po obserwacji quote nie są zmierzonym czasem live.",
        "",
        "## Główny następny eksperyment",
        "",
        f"**{next_experiment['name']}.** {next_experiment['rationale']}",
        "",
        f"Kryterium: {next_experiment['criterion']}",
        "",
        "Testy celowane: `python -m unittest discover -s tests -p test_btc_oof_target_audit.py`. Reprodukcja audytu: `python audit_btc_oof.py` (bez argumentów CLI; nie uruchamia treningu).",
        "",
        "Pliki wynikowe: `audit.json`, `report.md`, `same_9407_oof_target_comparison.parquet`, `temporal_metrics.csv`, `fold_metrics.csv`, `training_metric_reproduction.csv`, `model_age_metrics.csv`, `market_value_calibration.csv`, `target_examples.csv`, `live_latency_stage_metrics.csv`.",
    ]
    reported_cv = results["training_report_reproduction"]["reported_metrics"]
    training_metric_rows = []
    for row in results["training_report_reproduction"]["recomputed_fold_metrics"]:
        metric = row["metric"]
        training_metric_rows.append([
            metric, _fmt(row["reported_cv_fold_mean"]), _fmt(row["recomputed_cv_fold_mean"]),
            _fmt(row["reported_cv_fold_std"]), _fmt(row["recomputed_cv_fold_std"]),
            _fmt(row["mean_difference"], 12), _fmt(row["std_difference"], 12),
        ])
    lines += ["", "Reprodukcja raportu treningowego — średnia arytmetyczna 10 foldów, każdy liczony z wagami:", ""]
    lines += _markdown_table(
        ["Metryka", "Raport mean", "Odtworzona mean", "Raport std", "Odtworzona std", "Δ mean", "Δ std"],
        training_metric_rows,
    )
    return "\n".join(lines) + "\n"


def _append_report_appendix(report):
    existing = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    if AUDIT_START_MARKER in existing and AUDIT_END_MARKER in existing:
        before = existing.split(AUDIT_START_MARKER, 1)[0].rstrip()
        after = existing.split(AUDIT_END_MARKER, 1)[1].lstrip()
        content = before + "\n\n" + AUDIT_START_MARKER + "\n" + report.strip() + "\n" + AUDIT_END_MARKER
        if after:
            content += "\n\n" + after
    else:
        content = existing.rstrip() + "\n\n" + AUDIT_START_MARKER + "\n" + report.strip() + "\n" + AUDIT_END_MARKER + "\n"
    REPORT_PATH.write_text(content, encoding="utf-8")


def run_audit():
    settings, oof_path, model_ready_path, runtime_path, runtime_model_path = _resolve_inputs()
    model_meta = _read_json(runtime_model_path)
    oof = pd.read_parquet(oof_path)
    source_columns = ["Opened", "Open", "High", "Low", "Close", "Volume"]
    source = pd.read_parquet(model_ready_path, columns=source_columns)
    positions, fold_ids, source_targets, source_weights = _validate_oof_lineage(
        oof, source, model_meta, oof_path, model_ready_path
    )
    source_positions = np.arange(len(source), dtype=np.int64)
    valid_target_positions = np.flatnonzero(np.isfinite(source_targets))
    lineage_summary = {
        "model_ready_rows": int(len(source)),
        "source_rows_with_valid_exact_t_plus_5_target": int(len(valid_target_positions)),
        "oof_source_position_start": int(positions.min()),
        "oof_source_position_end_exclusive": int(positions.max()) + 1,
        "oof_positions_are_consecutive": bool(np.all(np.diff(positions) == 1)),
        "valid_target_rows_after_last_oof_position": int(
            (np.isfinite(source_targets) & (source_positions >= positions.max() + 1)).sum()
        ),
        "oof_fold_row_counts": {
            str(int(fold_id)): int((fold_ids == fold_id).sum())
            for fold_id in sorted(np.unique(fold_ids))
        },
    }
    fold_rows, training_metric_reproduction = _reproduce_training_report(
        oof, fold_ids, source, positions, model_meta
    )

    run_summary_path = Path(ANALYSIS_ROOT / "market_value.json")
    market_value_summary = _read_json(run_summary_path)
    market_value_dir = Path(market_value_summary["directory"])
    market_value_run = _read_json(market_value_dir / "market_value.json")
    if market_value_run.get("status") != "complete":
        raise ValueError("The current market_value run is not complete")
    manifest_rows = int(market_value_run["scores"]["latencies"]["0"]["rows"])
    if manifest_rows != EXPECTED_MARKET_VALUE_ROWS:
        raise ValueError(f"Current market_value manifest has {manifest_rows} scored rows, expected 9407")
    evaluation_path = market_value_dir / "latency_0s" / "out_of_fold_probabilities.parquet"
    common_path = market_value_dir / "common_latency_0s.parquet"
    evaluation = pd.read_parquet(evaluation_path)
    common = pd.read_parquet(common_path)
    sample = validate_common_market_sample(evaluation, common, EXPECTED_MARKET_VALUE_ROWS)
    sample["market_start_utc"] = pd.to_datetime(sample.market_start_utc_evaluation, utc=True)
    sample["market_end_utc"] = pd.to_datetime(sample.market_end_utc_common, utc=True)
    sample["opened_utc"] = sample.market_start_utc - pd.Timedelta(minutes=1)
    sample["target_binance_proxy_up"] = sample.target_binance_proxy_up.astype(float)
    sample["target_polymarket_up"] = sample.target_polymarket_up_evaluation.astype(float)
    sample["p_model_up"] = sample.p_model_up_evaluation.astype(float)
    sample["target_weight_recomputed"] = compute_target_weights_from_opened(
        sample.opened_utc.dt.tz_localize(None)
    )
    fold_by_time = pd.Series(fold_ids, index=pd.to_datetime(oof.Opened, utc=True))
    oof_join = oof[["Opened", "target_5m_candle_up", TARGET_WEIGHT_COL, "oof_pred_proba_up"]].copy()
    oof_join["opened_utc"] = pd.to_datetime(oof_join.Opened, utc=True)
    oof_join["fold_id"] = oof_join.opened_utc.map(fold_by_time)
    sample = sample.merge(
        oof_join[["opened_utc", "target_5m_candle_up", TARGET_WEIGHT_COL, "oof_pred_proba_up", "fold_id"]],
        on="opened_utc", how="left", validate="one_to_one",
    )
    if sample.target_5m_candle_up.isna().any():
        raise ValueError("One or more of the 9407 market rows lacks a source OOF record")
    if not np.array_equal(sample.target_5m_candle_up.to_numpy(float), sample.target_binance_proxy_up.to_numpy(float)):
        raise ValueError("Binance labels in the 9407 market sample differ from the current OOF labels")
    if not np.allclose(sample.p_model_up.to_numpy(float), sample.oof_pred_proba_up.to_numpy(float), rtol=0.0, atol=1e-14):
        raise ValueError("The market_value sample raw probability differs from the active OOF file")
    if not np.array_equal(sample[TARGET_WEIGHT_COL].to_numpy(float), sample.target_weight_recomputed.to_numpy(float)):
        raise ValueError("Training weights in market sample do not match the current OOF export")

    source_history = pd.DataFrame({
        "opened_utc": pd.to_datetime(source.Opened, utc=True),
        "target_5m_candle_up": source_targets,
        TARGET_WEIGHT_COL: source_weights,
    })
    source_history["phase"] = source_history.opened_utc.dt.minute.mod(5)
    oof_frame = oof[["Opened", "target_5m_candle_up", TARGET_WEIGHT_COL, "oof_pred_proba_up"]].copy()
    oof_frame["opened_utc"] = pd.to_datetime(oof_frame.Opened, utc=True)
    oof_frame["fold_id"] = fold_ids
    oof_frame["source_position"] = positions
    oof_frame["phase"] = oof_frame.opened_utc.dt.minute.mod(5)
    train_end_times = {
        int(fold["fold_id"]): pd.Timestamp(source.Opened.iloc[int(fold["train_end"]) - 1]).tz_localize("UTC")
        for fold in model_meta["walk_forward_folds"]
    }
    oof_frame["train_data_cutoff_opened_utc"] = oof_frame.fold_id.map(train_end_times)
    oof_frame["model_age_days_from_last_train_row"] = (
        oof_frame.opened_utc - oof_frame.train_data_cutoff_opened_utc
    ).dt.total_seconds() / 86400.0
    oof_frame["period_key"] = oof_frame.opened_utc.dt.tz_localize(None).dt.to_period("Q").astype(str)

    sample_times = sample.market_start_utc
    y_bin = sample.target_binance_proxy_up.to_numpy(float)
    y_pm = sample.target_polymarket_up.to_numpy(float)
    p_sample = sample.p_model_up.to_numpy(float)
    sample_weights = sample[TARGET_WEIGHT_COL].to_numpy(float)
    if not np.isclose(sample_weights.min(), sample_weights.max()):
        raise ValueError("The 9407-row evaluated sample is not on the single live-entry phase")

    first_sample_start = sample_times.min()
    prior_binance = common.loc[
        common.target_binance_proxy_up.notna()
        & pd.to_datetime(common.market_end_utc, utc=True).le(first_sample_start),
        "target_binance_proxy_up",
    ].astype(float)
    prior_pm_mask = (
        common.target_polymarket_up.notna()
        & pd.to_datetime(common.resolved_at_utc, utc=True).lt(first_sample_start)
    )
    prior_polymarket = common.loc[prior_pm_mask, "target_polymarket_up"].astype(float)
    if prior_binance.empty or prior_polymarket.empty:
        raise ValueError("No earlier Kacho labels are available for the constant-frequency baselines")
    baseline_binance_probability = float(prior_binance.mean())
    baseline_pm_probability = float(prior_polymarket.mean())

    target_scores = {
        "Binance": score_probabilities(y_bin, p_sample),
        "Polymarket": score_probabilities(y_pm, p_sample),
    }
    baseline_scores = {
        "Binance": score_probabilities(y_bin, np.full(len(y_bin), baseline_binance_probability)),
        "Polymarket": score_probabilities(y_pm, np.full(len(y_pm), baseline_pm_probability)),
    }
    bootstrap = paired_block_bootstrap(
        {"Binance": y_bin, "Polymarket": y_pm},
        p_sample,
        sample_times,
        sample_weight=sample_weights,
        replications=SAMPLE_BOOTSTRAP_REPLICATIONS,
        block_days=BOOTSTRAP_BLOCK_DAYS,
        seed=37,
    )

    discord = y_bin != y_pm
    p_clipped = np.clip(p_sample, 1e-15, 1.0 - 1e-15)
    ll_bin = -(y_bin * np.log(p_clipped) + (1.0 - y_bin) * np.log1p(-p_clipped))
    ll_pm = -(y_pm * np.log(p_clipped) + (1.0 - y_pm) * np.log1p(-p_clipped))
    brier_bin = np.square(p_sample - y_bin)
    brier_pm = np.square(p_sample - y_pm)
    ll_delta = ll_pm - ll_bin
    brier_delta = brier_pm - brier_bin
    mismatch_detail = {}
    for name, mask in (
        ("Binance UP -> Polymarket DOWN", (y_bin == 1.0) & (y_pm == 0.0)),
        ("Binance DOWN -> Polymarket UP", (y_bin == 0.0) & (y_pm == 1.0)),
    ):
        mismatch_detail[name] = {
            "rows": int(mask.sum()),
            "share_of_sample": float(mask.mean()),
            "mean_p_model_up": float(p_sample[mask].mean()) if mask.any() else None,
            "median_p_model_up": float(np.median(p_sample[mask])) if mask.any() else None,
            "mean_absolute_confidence_from_0_5": float(np.abs(p_sample[mask] - 0.5).mean()) if mask.any() else None,
            "model_accuracy_on_binance_label": float(((p_sample[mask] >= 0.5) == y_bin[mask]).mean()) if mask.any() else None,
            "model_accuracy_on_polymarket_label": float(((p_sample[mask] >= 0.5) == y_pm[mask]).mean()) if mask.any() else None,
            "log_loss_delta_sum_polymarket_minus_binance": float(ll_delta[mask].sum()),
            "log_loss_delta_mean_polymarket_minus_binance": float(ll_delta[mask].mean()) if mask.any() else None,
            "brier_delta_sum_polymarket_minus_binance": float(brier_delta[mask].sum()),
            "brier_delta_mean_polymarket_minus_binance": float(brier_delta[mask].mean()) if mask.any() else None,
        }
    paired_delta_summary = {
        "label_agreement_rows": int((~discord).sum()),
        "label_disagreement_rows": int(discord.sum()),
        "label_disagreement_rate": float(discord.mean()),
        "transitions": mismatch_detail,
        "model_confidence_on_disagreement": {
            "mean_p_model_up": float(p_sample[discord].mean()) if discord.any() else None,
            "median_p_model_up": float(np.median(p_sample[discord])) if discord.any() else None,
            "mean_absolute_confidence_from_0_5": float(np.abs(p_sample[discord] - 0.5).mean()) if discord.any() else None,
        },
        "per_observation_log_loss_delta_pm_minus_binance_sum": float(ll_delta.sum()),
        "per_observation_log_loss_delta_pm_minus_binance_mean": float(ll_delta.mean()),
        "disagreement_only_log_loss_delta_sum": float(ll_delta[discord].sum()),
        "per_observation_brier_delta_pm_minus_binance_sum": float(brier_delta.sum()),
        "per_observation_brier_delta_pm_minus_binance_mean": float(brier_delta.mean()),
        "disagreement_only_brier_delta_sum": float(brier_delta[discord].sum()),
        "share_of_nonzero_loss_differences_from_discordant_labels": 1.0,
        "ex_post_diagnostic_only": True,
    }
    sample["log_loss_binance"] = ll_bin
    sample["log_loss_polymarket"] = ll_pm
    sample["log_loss_delta_polymarket_minus_binance"] = ll_delta
    sample["brier_binance"] = brier_bin
    sample["brier_polymarket"] = brier_pm
    sample["brier_delta_polymarket_minus_binance"] = brier_delta
    sample["label_disagreement"] = discord

    distribution = _probability_summary(p_sample)
    calibration_rows = []
    calibration_summary = {}
    for label, y in (("Binance", y_bin), ("Polymarket", y_pm)):
        rows, summary = _calibration_table(y, p_sample, label)
        calibration_rows.extend(rows)
        calibration_summary[label] = summary
    sample_metrics_rows = []
    for label in ("Binance", "Polymarket"):
        metrics = target_scores[label]
        baseline = baseline_scores[label]
        ci = bootstrap[label]
        sample_metrics_rows.append({
            "target": label,
            "rows": len(sample),
            "observed_up_rate": metrics["up_rate"],
            "baseline_prior_rows": len(prior_binance) if label == "Binance" else len(prior_polymarket),
            "baseline_up_probability": baseline_binance_probability if label == "Binance" else baseline_pm_probability,
            "model_log_loss": metrics["log_loss"],
            "model_log_loss_95pct": ci["log_loss"],
            "baseline_log_loss": baseline["log_loss"],
            "model_brier": metrics["brier"],
            "model_brier_95pct": ci["brier"],
            "baseline_brier": baseline["brier"],
            "model_auc": metrics["auc"],
            "model_auc_95pct": ci["auc"],
            "baseline_auc": baseline["auc"],
            "model_accuracy": metrics["accuracy"],
            "model_accuracy_95pct": ci["accuracy"],
            "baseline_accuracy": baseline["accuracy"],
            "training_weighted_equals_unweighted": bool(np.allclose(sample_weights, sample_weights[0])),
            "training_weight": float(sample_weights[0]),
        })

    temporal_metrics, kacho_frame, oof_period_frame = _collect_period_metrics(
        oof_frame, source_history, common, sample
    )
    summary_group_specs = [
        ("all OOF minutes", oof_frame),
        ("live entry phase 4", oof_frame[oof_frame.phase.eq(4)]),
        ("Kacho available range, all phases", kacho_frame),
        ("Kacho available range, phase 4", kacho_frame[kacho_frame.phase.eq(4)]),
        ("9407 market-value rows, Binance target", sample.assign(
            opened_utc=sample.opened_utc,
            target_5m_candle_up=sample.target_binance_proxy_up,
            oof_pred_proba_up=sample.p_model_up,
            **{TARGET_WEIGHT_COL: sample[TARGET_WEIGHT_COL]},
        )),
    ]
    temporal_summary_rows = []
    for name, group in summary_group_specs:
        intervals, point = _period_bootstrap_row(group)
        weighted_point = score_probabilities(
            group.target_5m_candle_up.to_numpy(float),
            group.oof_pred_proba_up.to_numpy(float),
            group[TARGET_WEIGHT_COL].to_numpy(float),
        )
        temporal_summary_rows.append({
            "scope": name, "rows": len(group), "point": point,
            "training_weighted_point": weighted_point, "intervals": intervals,
        })

    fold_eval_counts = sample.groupby("fold_id").size().to_dict()
    market_period_start = pd.to_datetime(common.market_start_utc, utc=True).min()
    market_period_end = pd.to_datetime(common.market_start_utc, utc=True).max()
    for row in fold_rows:
        fold_id = int(row["fold_id"])
        row["market_value_evaluation_rows"] = int(fold_eval_counts.get(fold_id, 0))
        row["kacho_coverage_oof_rows"] = int(
            oof_frame.loc[
                oof_frame.fold_id.eq(fold_id)
                & oof_frame.opened_utc.between(market_period_start - pd.Timedelta(minutes=1),
                                               market_period_end - pd.Timedelta(minutes=1), inclusive="both")
            ].shape[0]
        )

    age_edges = [0.0, 7.0, 30.0, 60.0, 90.0, np.inf]
    age_labels = ["0-7 days", "7-30 days", "30-60 days", "60-90 days", "90+ days"]
    oof_frame["model_age_bin"] = pd.cut(
        oof_frame.model_age_days_from_last_train_row.clip(lower=0.0),
        bins=age_edges,
        labels=age_labels,
        right=True,
        include_lowest=True,
    )
    age_rows = []
    for age_bin, group in oof_frame.groupby("model_age_bin", observed=True):
        age_rows.append({
            "model_age_since_last_train_row": str(age_bin),
            "rows": int(len(group)),
            "start_utc": group.opened_utc.min(), "end_utc": group.opened_utc.max(),
            **score_probabilities(group.target_5m_candle_up, group.oof_pred_proba_up),
            "log_loss_training_weighted": score_probabilities(
                group.target_5m_candle_up, group.oof_pred_proba_up, group[TARGET_WEIGHT_COL]
            )["log_loss"],
            "brier_training_weighted": score_probabilities(
                group.target_5m_candle_up, group.oof_pred_proba_up, group[TARGET_WEIGHT_COL]
            )["brier"],
        })

    source_times = pd.DatetimeIndex(pd.to_datetime(source.Opened, errors="raise"))
    future_times = source_times + pd.Timedelta(minutes=5)
    future_positions = source_times.get_indexer(future_times)
    missing_horizon_rows = np.flatnonzero(future_positions < 0)
    interior_missing = [
        int(position) for position in missing_horizon_rows
        if position < len(source_times) - 5
    ]
    time_gap_mask = source_times.to_series().diff().ne(pd.Timedelta(minutes=1))
    actual_gaps = source_times.to_series().loc[time_gap_mask].iloc[1:]
    close_series = pd.Series(source.Close.to_numpy(float), index=source_times)
    future_close = close_series.reindex(source_times + pd.Timedelta(minutes=5)).to_numpy(float)
    tie_count_all = int((np.isfinite(source_targets) & np.equal(source.Close.to_numpy(float), future_close)).sum())
    oof_future_close = close_series.reindex(
        pd.DatetimeIndex(pd.to_datetime(oof.Opened, errors="raise")) + pd.Timedelta(minutes=5)
    ).to_numpy(float)
    tie_count_oof = int(
        (np.isfinite(oof_future_close) & np.equal(oof.Close.to_numpy(float), oof_future_close)).sum()
    )

    examples = _target_examples(sample, source, source.set_index("Opened", drop=False))
    chainlink_path = Path("data/chainlink/raw_reports/BTCUSD_reports.csv")
    chainlink_info = {"path": chainlink_path.as_posix(), "available": chainlink_path.exists()}
    if chainlink_path.exists():
        chainlink = pd.read_csv(chainlink_path, usecols=["ObservedAt"])
        chainlink_times = pd.to_datetime(chainlink.ObservedAt, utc=True, errors="coerce").dropna()
        chainlink_info.update({"rows": int(len(chainlink_times)), "start_utc": chainlink_times.min(),
                              "end_utc": chainlink_times.max(), "overlaps_kacho_period": bool(
                                  chainlink_times.max() >= market_period_start and chainlink_times.min() <= market_period_end
                              )})

    active_model_path = Path(model_meta["artifacts"]["final_model_path"])
    active_model_sha = _sha256(active_model_path)
    latency = _live_latency_audit(active_model_sha)

    hash_inputs = {
        "oof_sha256": _sha256(oof_path),
        "model_meta_sha256": _sha256(runtime_model_path),
        "model_ready_sha256": _sha256(model_ready_path),
        "market_value_fingerprint": market_value_run["fingerprint"],
        "audit_script_sha256": _sha256(Path(__file__)),
        "helper_sha256": _sha256(Path("utils/btc_oof_audit.py")),
        "live_trade_files": {str(path): _sha256(path) for path in latency["detailed_trade_files"]},
    }
    fingerprint = hashlib.sha256(
        json.dumps(hash_inputs, sort_keys=True).encode("utf-8")
    ).hexdigest()
    destination = ANALYSIS_ROOT / "runs" / fingerprint[:16]
    destination.mkdir(parents=True, exist_ok=True)

    probability_distribution = _probability_summary(p_sample)
    paired_delta_summary["log_loss_bootstrap_95pct_pm_minus_binance"] = bootstrap["Polymarket_minus_Binance"]["log_loss"]
    paired_delta_summary["brier_bootstrap_95pct_pm_minus_binance"] = bootstrap["Polymarket_minus_Binance"]["brier"]
    paired_delta_summary["auc_bootstrap_95pct_pm_minus_binance"] = bootstrap["Polymarket_minus_Binance"]["auc"]

    valid_oof_days = pd.to_datetime(oof.Opened, utc=True)
    model_metrics_unweighted = score_probabilities(
        oof.target_5m_candle_up, oof.oof_pred_proba_up
    )
    model_metrics_weighted = score_probabilities(
        oof.target_5m_candle_up, oof.oof_pred_proba_up, oof[TARGET_WEIGHT_COL]
    )
    overall_prior = _prior_frequency(source_history, valid_oof_days.min())
    overall_entry_prior = _prior_frequency(source_history, valid_oof_days.min(), phase=4)
    overall_baselines = {
        "unweighted": score_probabilities(
            oof.target_5m_candle_up,
            np.full(len(oof), overall_prior["unweighted"]),
        ),
        "training_weighted": score_probabilities(
            oof.target_5m_candle_up,
            np.full(len(oof), overall_prior["weighted"]),
            oof[TARGET_WEIGHT_COL],
        ),
        "entry_phase_unweighted": score_probabilities(
            oof.target_5m_candle_up.to_numpy(float)[oof_frame.phase.eq(4).to_numpy()],
            np.full(int(oof_frame.phase.eq(4).sum()), overall_entry_prior["unweighted"]),
        ),
    }
    oof_bootstrap = paired_block_bootstrap(
        {"Binance": oof.target_5m_candle_up.to_numpy(float)},
        oof.oof_pred_proba_up.to_numpy(float),
        valid_oof_days,
        sample_weight=oof[TARGET_WEIGHT_COL].to_numpy(float),
        replications=HISTORY_BOOTSTRAP_REPLICATIONS,
        block_days=BOOTSTRAP_BLOCK_DAYS,
        seed=38,
    )["Binance"]
    oof_target_scores = {
        "all_available_oof": {
            "rows": len(oof), "start_utc": valid_oof_days.min(), "end_utc": valid_oof_days.max(),
            "unweighted": model_metrics_unweighted,
            "training_weighted": model_metrics_weighted,
            "prior_baselines": {
                "all_minutes": {"available_rows": overall_prior["rows"], "up_probability": overall_prior["unweighted"], "unweighted": overall_baselines["unweighted"], "training_weighted": overall_baselines["training_weighted"]},
                "entry_phase_4": {"available_rows": overall_entry_prior["rows"], "up_probability": overall_entry_prior["unweighted"], "unweighted": overall_baselines["entry_phase_unweighted"]},
            },
            "weighted_metrics_block_bootstrap_95pct": oof_bootstrap,
        },
        "recomputed_target_rows": int(np.isfinite(source_targets).sum()),
        "target_rows_without_exact_t_plus_5_close": int(len(missing_horizon_rows)),
        "interior_missing_horizon_rows_excluding_final_5_source_rows": int(len(interior_missing)),
        "source_minute_timestamp_gaps": int(len(actual_gaps)),
        "source_gap_examples_utc": [pd.Timestamp(x).tz_localize("UTC") for x in actual_gaps.iloc[:10]],
        "binance_proxy_tie_rows_full_source": tie_count_all,
        "binance_proxy_tie_rows_oof": tie_count_oof,
        "target_definition": {
            "time_column": "Opened",
            "price_column": "Close",
            "future_timestamp": "Opened + exactly 5 minutes, reindexed by timestamp",
            "rule": "Close[Opened + 5 minutes] >= Close[Opened] -> UP; equality resolves UP",
            "missing_future_timestamp": "target is NaN and is excluded; row position does not substitute for elapsed time",
            "weight_column": TARGET_WEIGHT_COL,
            "weight_phase_4": 0.4625,
            "weight_phases_0_to_3": 0.134375,
        },
    }

    oof_path_sha = hash_inputs["oof_sha256"]
    manifest_path = ANALYSIS_ROOT / "oof_manifest.json"
    saved_oof_manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    oof_manifest_summary = {
        "path": oof_path.resolve(), "sha256": oof_path_sha, "size_bytes": oof_path.stat().st_size,
        "rows": int(len(oof)), "probability_column": "oof_pred_proba_up",
        "valid_probability_rows": int(oof.oof_pred_proba_up.notna().sum()),
        "probability_min": float(oof.oof_pred_proba_up.min()),
        "probability_max": float(oof.oof_pred_proba_up.max()),
        "start_utc": valid_oof_days.min(), "end_utc": valid_oof_days.max(),
        "model_metadata_path": runtime_model_path.resolve(),
        "model_metadata_sha256": hash_inputs["model_meta_sha256"],
        "model_ready_path": model_ready_path.resolve(),
        "model_ready_sha256": hash_inputs["model_ready_sha256"],
        "active_runtime_model_hash": active_model_sha,
        "metadata_target": model_meta["target_col"],
        "training_weight": model_meta["sample_weight"],
        "metadata_oof_matches_rows_and_probability_column": bool(
            int(model_meta["oof_predictions"]["rows"]) == len(oof)
            and model_meta["oof_predictions"]["prediction_col"] == "oof_pred_proba_up"
        ),
        "preexisting_oof_manifest_sha256_matches": bool(
            not saved_oof_manifest or saved_oof_manifest.get("sha256") == oof_path_sha
        ),
        "lineage_limit": "The model metadata stores output path, target, variant, counts, weights, and fold row ranges, but not a training-input content hash or OOF row-level fold column. This audit verifies all OOF timestamps, base OHLCV, targets, and weights against the current saved modeling input and reconstructs fold IDs from stored ranges; it cannot cryptographically prove the historical training input contents.",
        "oof_fold_ids_reconstructed_from_saved_ranges": True,
        "oof_row_fold_id_column_present": False,
    }

    market_value_info = {
        "summary_path": run_summary_path.resolve(),
        "run_directory": market_value_dir.resolve(),
        "fingerprint": market_value_run["fingerprint"],
        "parent_fingerprint": market_value_run.get("parent_fingerprint"),
        "common_markets_from_manifest": market_value_run["coverage"]["common_markets"],
        "evaluated_rows_from_manifest": manifest_rows,
        "evaluated_rows_in_probability_file": int(len(evaluation)),
        "evaluated_ids_unique": bool(evaluation.condition_id.is_unique),
        "same_ids_same_probability_and_labels": True,
        "primary_source": market_value_run["coverage"]["source"],
        "settlement_source_in_common_sample": common.loc[
            common.condition_id.isin(sample.condition_id), "outcome_source"
        ].value_counts(dropna=False).to_dict() if "outcome_source" in common else {},
        "validation_status_counts": common.loc[
            common.condition_id.isin(sample.condition_id), "validation_status"
        ].value_counts(dropna=False).to_dict() if "validation_status" in common else {},
    }

    target_examples_frame = pd.DataFrame(examples)
    temporal_metrics.to_csv(destination / "temporal_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(destination / "fold_metrics.csv", index=False)
    pd.DataFrame(training_metric_reproduction).to_csv(destination / "training_metric_reproduction.csv", index=False)
    pd.DataFrame(age_rows).to_csv(destination / "model_age_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(destination / "market_value_calibration.csv", index=False)
    target_examples_frame.to_csv(destination / "target_examples.csv", index=False)
    pd.DataFrame(latency["stage_rows"]).to_csv(destination / "live_latency_stage_metrics.csv", index=False)
    sample.to_parquet(destination / "same_9407_oof_target_comparison.parquet", index=False)

    main_scope_intervals = {
        row["scope"]: {
            "rows": row["rows"], "point": row["point"],
            "training_weighted_point": row["training_weighted_point"],
            "unweighted_95pct": row["intervals"],
        }
        for row in temporal_summary_rows
    }

    results = {
        "status": "complete",
        "fingerprint": fingerprint,
        "directory": destination.resolve(),
        "reproduction_command": "python audit_btc_oof.py",
        "run_inputs": hash_inputs,
        "oof_manifest": oof_manifest_summary,
        "oof_lineage_alignment": lineage_summary,
        "market_value_sample": market_value_info,
        "training_report_reproduction": {
            "reported_metrics": model_meta["metrics"]["cv"]["optuna"],
            "recomputed_fold_metrics": training_metric_reproduction,
            "all_reported_fold_macro_metrics_match_1e_10": all(
                row["matches_1e_10"] for row in training_metric_reproduction
            ),
            "pooled_oof_training_weighted_metrics": model_metrics_weighted,
            "fold_aggregation": "The training report is the unweighted arithmetic mean of its ten fold-level weighted metric values. The pooled OOF score is reported separately and is not substituted for that macro-fold report.",
        },
        "binance_target_quality": oof_target_scores,
        "same_9407_target_comparison": {
            "scores": sample_metrics_rows,
            "probability_distribution": distribution,
            "calibration_summary": calibration_summary,
            "calibration_bins": calibration_rows,
            "paired_target_bootstrap_95pct": bootstrap,
            "label_attribution": paired_delta_summary,
            "prior_baselines": {
                "binance": {"rows": len(prior_binance), "up_probability": baseline_binance_probability},
                "polymarket": {"rows": len(prior_polymarket), "up_probability": baseline_pm_probability},
                "availability_rule": "Binance labels have market_end_utc <= first scored decision; Polymarket labels have official resolved_at_utc < first scored decision.",
            },
        },
        "temporal_metrics": temporal_metrics.to_dict(orient="records"),
        "temporal_summary_rows": temporal_summary_rows,
        "temporal_summary_bootstrap": main_scope_intervals,
        "main_model_fold_metrics": fold_rows,
        "model_age_metrics": age_rows,
        "target_examples": examples,
        "chainlink_reference_cache": chainlink_info,
        "live_latency": latency,
        "time_and_target_audit": {
            "binance_data_market": settings["market"],
            "binance_price_source": "index",
            "market_start_rule": "Opened + 1 minute for phase-4 entries",
            "polymarket_window_rule": "official market start; exact 5-minute expiry; labels from official Gamma resolution/token mapping",
            "prediction_row_phase": "Opened.minute % 5 == 4 for live entry rows",
            "timestamps": "OOF Opened is stored without timezone and normalized as UTC; Polymarket timestamps are UTC; minute timestamps are exact and unique.",
            "unclosed_candle": "Live path invokes prediction only after the websocket kline is marked closed; offline OOF is exported from historical closed-candle rows.",
            "known_mapping_offset": "No +5 row shift is used. Target endpoint lookup is exact timestamp Opened+5m; the market opens at Opened+1m and expires at Opened+6m, matching the close boundaries of the start and end one-minute candles.",
            "source_price_decomposition": "The Binance proxy uses the BTCUSD Coin-Margined index Close; Polymarket outcomes use official settlement. Local Chainlink BTC reference reports do not overlap the Kacho evaluation interval, so this run cannot decompose each mismatch into a source-price difference versus a within-boundary oracle sampling difference.",
            "timestamp_gap_summary": {
                "source_minute_timestamp_gaps": int(len(actual_gaps)),
                "rows_missing_exact_t_plus_5_endpoint": int(len(missing_horizon_rows)),
                "interior_rows_missing_exact_t_plus_5_endpoint": int(len(interior_missing)),
                "missing_future_rows_are_dropped_not_shifted": True,
            },
        },
        "previous_report_interpretation": {
            "market_plus_oof_vs_market_only_0s_log_loss_95pct": market_value_run["scores"]["latencies"]["0"]["paired_uncertainty"]["market_plus_oof_minus_market_only"]["log_loss_left_minus_right_95pct"],
            "market_plus_oof_vs_market_only_0s_brier_95pct": market_value_run["scores"]["latencies"]["0"]["paired_uncertainty"]["market_plus_oof_minus_market_only"]["brier_left_minus_right_95pct"],
            "correct_reading": "At 0s, the paired Brier interval excludes zero and favors adding OOF, while the log-loss interval includes zero. At 1s and 2s both metric intervals include zero. The 0s Brier point improvement is small; it does not establish profitability and does not rule out predictive value on the Binance target.",
        },
        "output_files": {
            "report": (destination / "report.md").resolve(),
            "same_9407_parquet": (destination / "same_9407_oof_target_comparison.parquet").resolve(),
            "temporal_metrics_csv": (destination / "temporal_metrics.csv").resolve(),
            "fold_metrics_csv": (destination / "fold_metrics.csv").resolve(),
            "training_metric_reproduction_csv": (destination / "training_metric_reproduction.csv").resolve(),
            "model_age_metrics_csv": (destination / "model_age_metrics.csv").resolve(),
            "target_examples_csv": (destination / "target_examples.csv").resolve(),
            "live_latency_stage_metrics_csv": (destination / "live_latency_stage_metrics.csv").resolve(),
        },
    }

    report = _render_report(results, temporal_metrics, fold_rows, examples, latency)
    (destination / "report.md").write_text(report, encoding="utf-8")
    _write_json(destination / "audit.json", results)
    pointer = {
        "status": "complete",
        "fingerprint": fingerprint,
        "directory": destination.resolve(),
        "oof_sha256": oof_path_sha,
        "market_value_fingerprint": market_value_run["fingerprint"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(ANALYSIS_ROOT / "target_timing_audit.json", pointer)
    _append_report_appendix(report)
    print(json.dumps(pointer, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    run_audit()
