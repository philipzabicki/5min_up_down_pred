"""Retrospective comparison of the 2026-10-03 BTC model with its predecessor.

This fixed-configuration runner consumes saved OOF and Polymarket artifacts. It
does not retrain the BTC model, contact Polymarket, or submit trades.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from utils.polymarket_market_value import (
    EXECUTION_DELAYS,
    _execution_quotes,
    simulate_fixed_policy,
    walk_forward_models,
)
from utils.polymarket_policy import calibration_metrics


ROOT = Path(__file__).resolve().parent
OLD_RUN_ID = "20261002_041540"
NEW_RUN_ID = "20261003_043549"
COMMON_RUN = ROOT / "data/analysis/polymarket/BTC/runs/8d9688a07f0f7719"
SOURCE_RUN = ROOT / "data/datasets/polymarket/BTC/runs/efa1bf384fc6a2dc"
MODEL_READY_PATH = ROOT / "data/datasets/modeling/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m_model_ready.parquet"
OOF_PATH = ROOT / "data/datasets/modeling/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m_oof_predictions.parquet"
OLD_META_PATH = ROOT / f"data/models/BTC/{OLD_RUN_ID}/lgbm_meta_{OLD_RUN_ID}.json"
NEW_META_PATH = ROOT / f"data/models/BTC/{NEW_RUN_ID}/lgbm_meta_{NEW_RUN_ID}.json"
OLD_MODEL_PATH = ROOT / f"data/models/BTC/{OLD_RUN_ID}/lgbm_{OLD_RUN_ID}.txt"
NEW_MODEL_PATH = ROOT / f"data/models/BTC/{NEW_RUN_ID}/lgbm_{NEW_RUN_ID}.txt"
SELECTOR_PATH = ROOT / "data/analysis/feature_selector/BTC/20261003_015912/recommended_features.json"
VOLUME_PROFILE_TUNING_PATH = ROOT / "data/optuna/volume_profile/BTC/volume_profile_best_binary_logloss_mean_std_20261002_180354.json"
VOLUME_PROFILE_TRIALS_PATH = ROOT / "data/optuna/volume_profile/BTC/volume_profile_trials_binary_logloss_mean_std_20261002_180354.csv"
REACTION_PROFILE_TUNING_PATH = ROOT / "data/optuna/reaction_profile/BTC/reaction_profile_best_binary_logloss_mean_std_20261002_203021.json"
REACTION_PROFILE_TRIALS_PATH = ROOT / "data/optuna/reaction_profile/BTC/reaction_profile_trials_binary_logloss_mean_std_20261002_203021.csv"
LGBM_TUNING_PATH = ROOT / "data/optuna/lgbm/BTC/lgbm_generic_optuna_best_mean_std_20260717_120022.json"
LGBM_TRIALS_PATH = ROOT / "data/optuna/lgbm/BTC/lgbm_generic_optuna_trials_mean_std_20260717_120022.csv"
INDICATOR_FIT_CONFIG_PATH = ROOT / "data/features/indicators_fit/BTC/all/fit_indicators_applied_config.json"
RAW_OHLCV_PATH = ROOT / "data/datasets/raw/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m.csv"
OLD_EVALUATION_PATH = COMMON_RUN / "latency_0s/out_of_fold_probabilities.parquet"
OLD_MARKET_VALUE_PATH = COMMON_RUN / "market_value.json"
COMMON_PATH = COMMON_RUN / "common_latency_0s.parquet"
QUOTE_PATH = SOURCE_RUN / "quotes.parquet"
MARKET_PATH = SOURCE_RUN / "markets.parquet"
REPORT_PATH = ROOT / "docs/btc_new_model_evaluation_20261003.md"
MANIFEST_PATH = ROOT / "docs/btc_new_model_manifest_20261003.json"
OUTPUT_ROOT = ROOT / "data/analysis/polymarket/BTC/new_model_comparison"

INITIAL_BANKROLL_USDC = 100.0
FIXED_STAKE_USDC = 5.0
EXPECTED_PNL_BUFFER_USDC = 0.25
BOOTSTRAP_BLOCK_DAYS = 3
BOOTSTRAP_REPLICATIONS = 2_000
BOOTSTRAP_SEED = 20261003
TARGET_LABEL_AVAILABLE_MINUTES_AFTER_OPEN = 6
MODEL_DECISION_MINUTES_AFTER_OPEN = 1
COMMON_LATENCY_SECONDS = 0


def _json_default(value):
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(values):
    return pd.to_datetime(values, utc=True, errors="coerce")


def purge_training_by_label_availability(
        train_opened, prediction_at, *, label_delay_minutes=TARGET_LABEL_AVAILABLE_MINUTES_AFTER_OPEN
):
    """Keep only training labels observable strictly before prediction_at."""
    opened = _utc(train_opened)
    labels_available_at = opened + pd.Timedelta(minutes=label_delay_minutes)
    cutoff = pd.Timestamp(prediction_at)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    return labels_available_at < cutoff


def align_new_oof_to_markets(common: pd.DataFrame, new_oof: pd.DataFrame):
    """Join one OOF row to one market only by exact UTC Opened timestamps."""
    required_common = {
        "Opened", "condition_id", "market_start_utc", "decision_available_at",
        "timestamp_utc", "target_polymarket_up", "target_binance_proxy_up",
        "p_model_up", "quote_valid", "validation_status",
    }
    required_oof = {"Opened", "oof_pred_proba_up", "target_5m_candle_up"}
    if not required_common.issubset(common.columns):
        raise ValueError(f"Common market rows missing: {sorted(required_common-set(common.columns))}")
    if not required_oof.issubset(new_oof.columns):
        raise ValueError(f"New OOF rows missing: {sorted(required_oof-set(new_oof.columns))}")

    common = common.copy()
    new_oof = new_oof.copy()
    common["Opened"] = _utc(common["Opened"])
    new_oof["Opened"] = _utc(new_oof["Opened"])
    for column in ("market_start_utc", "decision_available_at", "timestamp_utc"):
        common[column] = _utc(common[column])
    if common["Opened"].isna().any() or new_oof["Opened"].isna().any():
        raise ValueError("Opened contains invalid timestamps")
    if common["condition_id"].duplicated().any() or common["Opened"].duplicated().any():
        raise ValueError("Common market rows must be unique by market and Opened")
    if new_oof["Opened"].duplicated().any():
        raise ValueError("New OOF Opened timestamps must be unique")

    joined = common.merge(
        new_oof[["Opened", "oof_pred_proba_up", "target_5m_candle_up"]],
        on="Opened", how="left", validate="one_to_one", indicator="new_oof_join",
    )
    if not joined["market_start_utc"].eq(
            joined["Opened"] + pd.Timedelta(minutes=MODEL_DECISION_MINUTES_AFTER_OPEN)
    ).all():
        raise ValueError("Market start is not the exact decision timestamp for its BTC Opened row")
    if not (joined["timestamp_utc"] >= joined["decision_available_at"]).all():
        raise ValueError("A selected Polymarket quote precedes model availability")
    if not (
            joined["timestamp_utc"]
            <= joined["decision_available_at"] + pd.Timedelta(seconds=2)
    ).all():
        raise ValueError("A selected Polymarket quote exceeds the pinned 2-second tolerance")
    matched = joined["new_oof_join"].eq("both")
    p_new = pd.to_numeric(joined.loc[matched, "oof_pred_proba_up"], errors="coerce")
    if not np.isfinite(p_new.to_numpy(float)).all() or not p_new.between(0, 1).all():
        raise ValueError("Matched new OOF probabilities must be finite values in [0, 1]")
    target_match = np.isclose(
        joined.loc[matched, "target_binance_proxy_up"].to_numpy(float),
        joined.loc[matched, "target_5m_candle_up"].to_numpy(float),
        equal_nan=False,
    )
    if not target_match.all():
        raise ValueError("Stored Binance proxy target disagrees with the new OOF target")
    joined["p_old_model_up"] = pd.to_numeric(joined["p_model_up"], errors="coerce")
    joined["p_new_model_up"] = pd.to_numeric(joined["oof_pred_proba_up"], errors="coerce")
    return joined, {
        "common_rows_before_oof_join": int(len(common)),
        "new_oof_rows_before_join": int(len(new_oof)),
        "exact_timestamp_matches": int(matched.sum()),
        "common_rows_without_new_oof": int((~matched).sum()),
        "new_oof_rows_without_common_market": int(
            len(new_oof) - int(new_oof["Opened"].isin(common["Opened"]).sum())
        ),
        "new_target_matches_binance_proxy": int(target_match.sum()),
    }


def score_binary(y, probability):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    calibration = calibration_metrics(y, p)
    return {
        "rows": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "accuracy_at_0_5": float(accuracy_score(y, p >= 0.5)),
        "calibration_slope": calibration.get("slope"),
        "calibration_intercept": calibration.get("intercept"),
        "calibration_bins": calibration.get("reliability_bins", []),
    }


def paired_block_bootstrap_differences(
        frame, target_column, comparisons, *, replications=BOOTSTRAP_REPLICATIONS,
        block_days=BOOTSTRAP_BLOCK_DAYS, seed=BOOTSTRAP_SEED,
):
    """Use one shared moving-block sample for every model and loss comparison."""
    ordered = frame.sort_values("market_start_utc").copy()
    y = ordered[target_column].to_numpy(dtype=int)
    days = _utc(ordered["market_start_utc"]).dt.floor("D")
    calendar = pd.date_range(days.min(), days.max(), freq="D", tz="UTC")
    day_index = pd.Index(calendar)
    count_by_day = np.bincount(day_index.get_indexer(days), minlength=len(calendar)).astype(float)
    losses = {}
    for model in sorted({name for left, right in comparisons.values() for name in (left, right)}):
        p = np.clip(ordered[model].to_numpy(float), 1e-8, 1 - 1e-8)
        losses[(model, "log_loss")] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        losses[(model, "brier")] = (y - p) ** 2
    daily = {}
    for key, values in losses.items():
        daily[key] = np.bincount(day_index.get_indexer(days), weights=values,
                                 minlength=len(calendar)).astype(float)

    differences = {}
    for label, (left, right) in comparisons.items():
        for metric in ("log_loss", "brier"):
            left_values, right_values = losses[(left, metric)], losses[(right, metric)]
            daily_delta = daily[(left, metric)] - daily[(right, metric)]
            point = float((left_values - right_values).mean())
            rng = np.random.default_rng(seed)
            estimates = np.empty(replications, dtype=float)
            blocks_per_replication = int(math.ceil(len(calendar) / block_days))
            for replication in range(replications):
                starts = rng.integers(
                    0, max(len(calendar) - block_days + 1, 1), size=blocks_per_replication
                )
                selected = np.concatenate([
                    np.arange(start, min(start + block_days, len(calendar)))
                    for start in starts
                ])[:len(calendar)]
                sampled_count = count_by_day[selected].sum()
                estimates[replication] = daily_delta[selected].sum() / sampled_count
            differences[f"{label}_{metric}"] = {
                "left_minus_right": f"{left} - {right}",
                "point_difference": point,
                "ci_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
                "interpretation": "Positive means the left model has higher loss.",
            }
    return {
        "rows": int(len(ordered)),
        "calendar_days": int(len(calendar)),
        "block_days": int(block_days),
        "replications": int(replications),
        "seed": int(seed),
        "same_block_draws_for_all_models_and_metrics": True,
        "differences": differences,
    }


def _input_paths():
    return [
        MODEL_READY_PATH, OOF_PATH, OLD_META_PATH, NEW_META_PATH,
        OLD_MODEL_PATH, NEW_MODEL_PATH, SELECTOR_PATH,
        ROOT / "configs/modeling.json", ROOT / "configs/runtime/active.json",
        COMMON_PATH, OLD_EVALUATION_PATH, OLD_MARKET_VALUE_PATH,
        COMMON_RUN / "inputs.json", QUOTE_PATH, MARKET_PATH,
        ROOT / "data/datasets/modeling/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m_model_ready_metadata.json",
        INDICATOR_FIT_CONFIG_PATH, VOLUME_PROFILE_TUNING_PATH, VOLUME_PROFILE_TRIALS_PATH,
        REACTION_PROFILE_TUNING_PATH, REACTION_PROFILE_TRIALS_PATH,
        LGBM_TUNING_PATH, LGBM_TRIALS_PATH, RAW_OHLCV_PATH,
        ROOT / "utils/polymarket_market_value.py", ROOT / "utils/polymarket_policy.py",
        ROOT / "utils/polymarket_history.py", ROOT / "utils/data.py",
        ROOT / "audit_feature_readiness.py",
        ROOT / "run.py", ROOT / "select_features.py", ROOT / "fit_volume_profile.py",
        ROOT / "fit_indicators.py", ROOT / "features/common_utils.py",
        ROOT / "run_btc_new_model_comparison.py",
    ]


def audit_source_market_exclusions(common):
    markets = pd.read_parquet(
        MARKET_PATH,
        columns=[
            "condition_id", "market_start_utc", "resolved_at_utc", "validation_status"
        ],
    )
    accepted_ids = set(common["condition_id"].astype(str))
    excluded = markets.loc[~markets["condition_id"].astype(str).isin(accepted_ids)].copy()
    if excluded.empty:
        return {
            "source_market_rows": int(len(markets)),
            "common_market_rows": int(len(common)),
            "excluded_market_rows": 0,
            "excluded_reason_counts": {},
        }

    excluded_ids = excluded["condition_id"].astype(str).tolist()
    quotes = pd.read_parquet(
        QUOTE_PATH,
        columns=["condition_id", "timestamp_utc", "quote_valid"],
        filters=[("condition_id", "in", excluded_ids)],
    )
    quotes = quotes.loc[
        quotes["quote_valid"].fillna(False)
        & quotes["condition_id"].astype(str).isin(excluded_ids)
    ]
    quotes = quotes.merge(
        excluded[["condition_id", "market_start_utc"]],
        on="condition_id",
        how="inner",
        validate="many_to_one",
    )
    quotes = quotes.loc[quotes["timestamp_utc"].ge(quotes["market_start_utc"])].copy()
    quotes["quote_delay_s"] = (
        quotes["timestamp_utc"] - quotes["market_start_utc"]
    ).dt.total_seconds()
    first_forward = quotes.sort_values("timestamp_utc").groupby("condition_id").first()
    excluded = excluded.join(first_forward[["quote_delay_s"]], on="condition_id")

    reasons = {}
    missing_resolution = excluded["resolved_at_utc"].isna()
    reasons["missing_resolution_time"] = int(missing_resolution.sum())
    remaining = ~missing_resolution
    over_tolerance = remaining & excluded["quote_delay_s"].gt(2.0)
    reasons["first_valid_quote_after_2s_tolerance"] = int(over_tolerance.sum())
    no_quote = remaining & excluded["quote_delay_s"].isna()
    reasons["no_valid_forward_quote"] = int(no_quote.sum())
    other = remaining & ~over_tolerance & ~no_quote
    if other.any():
        reasons["other_validation_exclusion"] = int(other.sum())
    reasons = {key: value for key, value in reasons.items() if value}
    return {
        "source_market_rows": int(len(markets)),
        "common_market_rows": int(len(common)),
        "excluded_market_rows": int(len(excluded)),
        "excluded_reason_counts": reasons,
        "max_excluded_valid_quote_delay_s": (
            float(excluded["quote_delay_s"].max())
            if excluded["quote_delay_s"].notna().any() else None
        ),
    }


def build_model_fold_audit(new_meta, new_oof):
    """Rebuild OOF row/fold identity and exact label-availability boundaries."""
    ready = pd.read_parquet(MODEL_READY_PATH, columns=["Opened", "target_5m_candle_up"])
    ready["Opened"] = _utc(ready["Opened"])
    if ready["Opened"].duplicated().any():
        raise ValueError("Model-ready Opened timestamps are not unique")
    positions = pd.Index(ready["Opened"]).get_indexer(_utc(new_oof["Opened"]))
    if (positions < 0).any():
        raise ValueError("New OOF timestamps do not all exist in model-ready input")
    stored_target = ready.iloc[positions]["target_5m_candle_up"].to_numpy(float)
    oof_target = new_oof["target_5m_candle_up"].to_numpy(float)
    if not np.isclose(stored_target, oof_target, equal_nan=True).all():
        raise ValueError("New OOF target does not match the model-ready target by exact Opened")

    fold_reports = []
    assigned = np.zeros(len(new_oof), dtype=np.int8)
    for fold in new_meta["walk_forward_folds"]:
        fold_id = int(fold["fold_id"])
        train_start, train_end = int(fold["train_start"]), int(fold["train_end"])
        test_start, test_end = int(fold["test_start"]), int(fold["test_end"])
        if not (0 <= train_start < train_end == test_start < test_end <= len(ready)):
            raise ValueError(f"Invalid saved walk-forward boundaries for fold {fold_id}")
        train_opened = ready.iloc[train_start:train_end]["Opened"]
        test_opened = ready.iloc[test_start:test_end]["Opened"]
        first_prediction_at = test_opened.iloc[0] + pd.Timedelta(
            minutes=MODEL_DECISION_MINUTES_AFTER_OPEN
        )
        train_available = purge_training_by_label_availability(
            train_opened, first_prediction_at
        )
        oof_mask = (positions >= test_start) & (positions < test_end)
        assigned[oof_mask] += 1
        details = new_meta.get("walk_forward_details", {}).get("optuna", [])
        detail = next((item for item in details if int(item["fold_id"]) == fold_id), {})
        fold_reports.append({
            "fold_id": fold_id,
            "train_rows_as_saved": int(train_end - train_start),
            "test_rows_as_saved": int(test_end - test_start),
            "train_opened_start": train_opened.iloc[0],
            "train_opened_end": train_opened.iloc[-1],
            "test_opened_start": test_opened.iloc[0],
            "test_opened_end": test_opened.iloc[-1],
            "first_prediction_at": first_prediction_at,
            "latest_train_label_available_at": train_opened.iloc[-1]
                + pd.Timedelta(minutes=TARGET_LABEL_AVAILABLE_MINUTES_AFTER_OPEN),
            "train_rows_with_label_unavailable_at_first_prediction": int((~train_available).sum()),
            "rows_in_saved_oof": int(oof_mask.sum()),
            "best_iteration_selected_on_this_test_fold": detail.get("best_iteration"),
        })
    if not np.all(assigned == 1):
        raise ValueError("Every new OOF row must map to exactly one saved test fold")
    return fold_reports, positions


def _calibration_columns(frame):
    return {
        "old_btc": "p_old_model_up",
        "new_btc": "oof_raw",
        "market_only": "market_only",
        "new_btc_platt": "oof_platt",
        "market_plus_new_btc": "market_plus_oof",
    }


def _period_text(frame):
    return f"{_utc(frame.market_start_utc).min().isoformat()} – {_utc(frame.market_start_utc).max().isoformat()}"


def _format(value, digits=6):
    return "n/a" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def render_report(payload):
    scores = payload["scores"]
    evaluation = payload["evaluation"]
    quality_rows = []
    model_labels = {
        "old_btc": "Stary BTC (surowy OOF)",
        "new_btc": "Nowy BTC (surowy OOF)",
        "market_only": "MARKET_ONLY (L2 logistic)",
    }
    for target_name, target_col in [
        ("Oficjalny settlement Polymarket", "target_polymarket_up"),
        ("Target BTC Binance (pomocniczo)", "target_binance_proxy_up"),
    ]:
        for key in ("old_btc", "new_btc", "market_only"):
            metric = score_binary(evaluation[target_col], evaluation[key])
            fold_ll = []
            for fold in sorted(evaluation.fold.unique()):
                block = evaluation.loc[evaluation.fold.eq(fold)]
                fold_metric = score_binary(block[target_col], block[key])
                fold_ll.append(f"F{int(fold)} {fold_metric['log_loss']:.6f}")
            quality_rows.append(
                f"| {target_name} | {model_labels[key]} | {metric['rows']:,} | "
                f"{metric['log_loss']:.6f} | {metric['brier']:.6f} | "
                f"{metric['auc']:.4f} | {metric['accuracy_at_0_5']:.4f} | "
                f"{metric['calibration_slope']:.3f} / {metric['calibration_intercept']:.3f} | "
                f"{' · '.join(fold_ll)} |"
            )

    incremental_rows = []
    for key, label in [
        ("market_only", "Skalibrowany MARKET_ONLY"),
        ("new_btc_platt", "Nowy BTC, Platt z wcześniejszych settlementów"),
        ("market_plus_new_btc", "MARKET_ONLY + nowy BTC"),
    ]:
        metric = score_binary(evaluation["target_polymarket_up"], evaluation[key])
        incremental_rows.append(
            f"| {label} | {metric['rows']:,} | {metric['log_loss']:.6f} | "
            f"{metric['brier']:.6f} | {metric['auc']:.4f} | "
            f"{metric['calibration_slope']:.3f} / {metric['calibration_intercept']:.3f} |"
        )
    diff = payload["bootstrap"]["differences"]
    incremental_rows.append(
        f"| Dodanie nowego BTC vs MARKET_ONLY (różnica LL / Brier, 95% CI) | "
        f"{len(evaluation):,} | "
        f"{diff['market_plus_new_minus_market_only_log_loss']['point_difference']:+.6f} "
        f"[{diff['market_plus_new_minus_market_only_log_loss']['ci_95'][0]:+.6f}, "
        f"{diff['market_plus_new_minus_market_only_log_loss']['ci_95'][1]:+.6f}] | "
        f"{diff['market_plus_new_minus_market_only_brier']['point_difference']:+.6f} "
        f"[{diff['market_plus_new_minus_market_only_brier']['ci_95'][0]:+.6f}, "
        f"{diff['market_plus_new_minus_market_only_brier']['ci_95'][1]:+.6f}] | — | "
        f"ujemna różnica sprzyja dodaniu |"
    )

    portfolio_rows = []
    for delay in EXECUTION_DELAYS:
        for strategy in ("market_only", "new_btc_platt", "market_plus_new_btc"):
            result = payload["portfolio"][str(delay)][strategy]
            portfolio_rows.append(
                f"| {strategy} | {delay}s | ${result['ending_available_cash']:.2f} | "
                f"${result['ending_locked_cost']:.2f} | ${result['realized_pnl']:.2f} | "
                f"{result['max_drawdown']:.2%} | ${result['turnover']:.2f} | "
                f"${result['fees']:.2f} | {result['trades']} | "
                f"{result['rejection_reasons'].get('insufficient_cash', 0)} |"
            )

    fold_lines = []
    for fold in payload["model_fold_audit"]:
        fold_lines.append(
            f"- fold {fold['fold_id']}: train {fold['train_opened_start']}–{fold['train_opened_end']}; "
            f"test {fold['test_opened_start']}–{fold['test_opened_end']}; "
            f"label-unavailable rows at first prediction: "
            f"{fold['train_rows_with_label_unavailable_at_first_prediction']}; "
            f"saved best iteration {fold['best_iteration_selected_on_this_test_fold']}."
        )

    new_metric = payload["wide_new_btc"]
    mismatch = payload["target_disagreement"]
    joins = payload["join_counts"]
    source_exclusions = joins["source_market_exclusions"]
    exclusion_labels = {
        "missing_resolution_time": "brak czasu rozstrzygnięcia",
        "first_valid_quote_after_2s_tolerance": "pierwszy poprawny quote po limicie 2 s",
        "no_valid_forward_quote": "brak poprawnego forward quote",
        "other_validation_exclusion": "inna przyczyna walidacji",
    }
    exclusion_text = ", ".join(
        f"{exclusion_labels.get(reason, reason)}: {count}"
        for reason, count in source_exclusions["excluded_reason_counts"].items()
    ) or "brak"
    config = payload["config_audit"]
    training = payload["training_audit"]
    quote = payload["quote_audit"]
    selection = payload["training_selection_audit"]
    live = payload["live_parity"]

    selection_lines = []
    for label, key in (
        ("LGBM Optuna (daty foldów odtworzone z bieżących danych)", "lgbm_optuna"),
        ("selekcja cech", "feature_selector"),
        ("volume profile Optuna", "volume_profile_optuna"),
        ("reaction profile Optuna", "reaction_profile_optuna"),
    ):
        audit = selection[key]
        for overlap in audit["evaluation_validation_overlaps"]:
            selection_lines.append(
                f"- {label}, fold {overlap['fold_id']}: walidacja "
                f"{overlap['validation_start']}–{overlap['validation_end']} "
                f"(train {overlap['train_start']}–{overlap['train_end']}) obejmuje "
                f"{overlap['evaluation_overlap_rows']:,} wierszy ocenianego okresu "
                f"{overlap['evaluation_overlap_start']}–{overlap['evaluation_overlap_end']}."
            )

    indicator_lines = []
    for segment in selection["indicator_fit"]["evaluation_overlapping_segments"]:
        indicator_lines.append(
            f"- segment wskaźników {segment['segment_id']}: kalibracja progów "
            f"{segment['calibration_start']}–{segment['calibration_end']} obejmowała "
            f"{segment['evaluation_rows_in_calibration_prefix']:,} wierszy okresu oceny; "
            f"gap obejmował {segment['evaluation_rows_in_metric_gap']:,} wierszy; "
            f"walidacja metryki zaczynała się {segment['metric_validation_start']} i "
            f"obejmowała {segment['evaluation_rows_in_metric_validation']:,} wierszy oceny."
        )

    largest_live_gap = live["largest_probability_gap_above_tolerance"]
    live_gap_text = (
        "brak"
        if largest_live_gap is None
        else (
            f"{largest_live_gap['Opened']} (|Δp|={largest_live_gap['proba_up_abs_diff']:.8f}, "
            f"największa różnica cechy: {largest_live_gap.get('worst_feature', 'n/a')})"
        )
    )
    return "\n".join([
        "# Ocena nowego modelu BTC — 2026-10-03",
        "",
        "## Zakres i werdykt",
        "",
        f"Porównanie dotyczy {len(evaluation):,} wspólnych, chronologicznych okien BTC 5m "
        f"({payload['evaluation_period']}) na oficjalnym rozstrzygnięciu Polymarket. "
        "To wynik development, nie niezależny test: oceniane etykiety są w foldzie 8 "
        "nowego BTC i uczestniczyły w wyborze jego iteracji; fold walidacyjny selektora "
        "cech i strojenia profili również pokrywa ten okres. Nie trenowałem ponownie "
        "modelu BTC ani nie uruchamiałem szerokiego strojenia.",
        "",
        f"- Względem starego BTC: **{payload['verdict']['new_vs_old']}** "
        "na wspólnych oficjalnych settlementach.",
        f"- Wartość ponad notowania rynku: **{payload['verdict']['incremental_information']}** "
        "dla MARKET_ONLY + nowy BTC względem MARKET_ONLY.",
        f"- Wynik portfela po kosztach: **{payload['verdict']['portfolio']}** "
        "przy stałym $5, buforze EV $0.25 i odtworzonych kwotowaniach.",
        "- Niezależny test: **nie istnieje w tych artefaktach**. Zalecany kolejny krok: "
        "shadow bez zleceń, z niezmienionym modelem i zapisem timestampów snapshotu/wykonania.",
        "",
        "## Identyfikacja modelu i konfiguracji",
        "",
        f"Nowy model: `{NEW_RUN_ID}`, target `{payload['model']['target']}`, "
        f"{payload['model']['new_feature_count']} cech; stary model `{OLD_RUN_ID}`, "
        f"{payload['model']['old_feature_count']} cech. Nowy meta ma pusty wektor "
        "`configured_constraints` i `applied_constraints`; plik modelu zapisuje pusty "
        "`monotone_constraints`. `monotone_constraints_method` i `monotone_penalty` "
        "w hiperparametrach nie aktywują ograniczeń bez wektora.",
        "",
        f"`feature_selection.exclusions_enabled=false`, `excluded_count=0`; artefakt "
        f"zawiera {payload['model']['new_feature_count']} z {payload['model']['feature_source_count']} "
        "dostępnych cech. `modeling.json` zawiera listę wykluczeń, lecz flaga jest "
        "wyłączona. Walidacja konfiguracji wolumenu i reaction profile względem metadanych: "
        f"{config['volume_profile_matches_metadata']} / {config['reaction_profile_matches_metadata']}. "
        "Metadane modelu nie przechowują `config_snapshot` ani `config_path`; porównanie "
        "odtwarza zgodność efektywnych konfiguracji cech, lecz nie dowodzi hash całego "
        "pliku konfiguracji z chwili treningu.",
        "",
        f"Wspólne porównanie MARKET_ONLY odtwarza wcześniejszą definicję: standaryzowana "
        "regresja logistyczna L2 na 17 cechach jednoczesnej książki Kacho (bid/ask obu "
        "tokenów, midy, spready, sumy bid/ask, znormalizowany midpoint, logarytmy rozmiarów "
        "i imbalance rozmiarów). C wybierane jest ze zbioru `{0.01, 0.1, 1, 10}` "
        "na wcześniejszych danych. Wskaźnik `p_market_mid` jest tylko normalizowanym "
        "midpointem, nie ceną zakupu; transakcje używają ask.",
        "",
        f"Targety Binance i oficjalny settlement różnią się w {mismatch['main_disagreements']:,} "
        f"z {mismatch['main_rows']:,} wspólnych okien ({mismatch['main_rate']:.2%}); "
        f"dla szerszego zbioru nowego modelu: {mismatch['wide_disagreements']:,}/"
        f"{mismatch['wide_rows']:,} ({mismatch['wide_rate']:.2%}).",
        "",
        "## 1. Jakość prognoz: stary BTC, nowy BTC i MARKET_ONLY",
        "",
        "Główna część porównuje identyczne okna i ten sam moment obserwacji. Ostatnia "
        "kolumna pokazuje log loss w kolejnych outer foldach.",
        "",
        "| Etykieta | Wariant | Okna | Log loss | Brier | AUC | Accuracy | Kalibracja: slope / intercept | Log loss F0 / F1 / F2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *quality_rows,
        "",
        f"Nowy BTC osobno na szerszym zbiorze OOF z oficjalnym settlementem: n={new_metric['rows']:,}, "
        f"log loss={new_metric['log_loss']:.6f}, Brier={new_metric['brier']:.6f}, "
        f"AUC={new_metric['auc']:.4f}, accuracy={new_metric['accuracy_at_0_5']:.4f}. "
        "To szersza retrospektywna ocena samego modelu; nie jest bezpośrednio porównywana "
        "z późniejszymi modelami drugiego poziomu.",
        "",
        f"Sparowany 3-dniowy moving-block bootstrap, {BOOTSTRAP_REPLICATIONS:,} "
        "replikacji; jeden wspólny zestaw bloków dla wszystkich porównań:",
        "",
        *[
            f"- {name}: Δ {'Brier' if name.endswith('_brier') else 'log loss'}="
            f"{values['point_difference']:+.6f}, "
            f"95% CI [{values['ci_95'][0]:+.6f}, {values['ci_95'][1]:+.6f}]"
            for name, values in payload["bootstrap"]["differences"].items()
        ],
        "",
        "## 2. Informacja ponad MARKET_ONLY",
        "",
        "W MARKET_ONLY + nowy BTC dodano logit prawdopodobieństwa nowego modelu do "
        "tych samych 17 cech książki. Standaryzacja, C i dopasowanie regresji są "
        "wykonywane tylko na wcześniejszych oknach; outer okna porównania są wspólne.",
        "",
        "| Wariant | Okna | Log loss | Brier | AUC | Kalibracja: slope / intercept |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *incremental_rows,
        "",
        "## 3. Chronologiczne portfele startujące z $100",
        "",
        f"Stała stawka ${FIXED_STAKE_USDC:.2f}; zakup tylko gdy dokładny payoff po opłacie "
        f"daje EV powyżej ${EXPECTED_PNL_BUFFER_USDC:.2f}. Gotówka jest blokowana do "
        "`max(resolved_at_utc, market_end_utc)`, potem wpływ staje się dostępny; bez długu "
        "i dopłat. Ask z chwili decyzji zamraża stronę i limit. Przy opóźnieniu 1/2s "
        "wykonanie używa pierwszego zapisanego późniejszego asku nie wyższego od limitu; "
        "spadek ceny daje price improvement, brak dodatkowego slippage ponad limit. "
        "Książka Kacho ma nieznany wiek giełdowy. Opłaty pochodzą z zapisanych metadanych "
        "rynku; historycznych dat zmian opłat nie da się ustalić.",
        "",
        "| Wariant | Opóźnienie | Końcowa gotówka | Kapitał otwarty | PnL | Maks. DD* | Obrót | Opłaty | Transakcje | Pominięte: brak gotówki |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *portfolio_rows,
        "",
        "*Obsunięcie jest liczone od gotówki plus kosztu otwartych pozycji, bez wyceny "
        "rynkowej. Na koniec wszystkie pozycje mają oficjalny settlement i koszt pozycji "
        "otwartych wynosi zero. Test nie modeluje nieznanej trwałości snapshotu ani "
        "rzeczywistego fillu.",
        "",
        "## Jakość OOF, podziały i ograniczenia",
        "",
        f"Nowe OOF: {payload['model']['new_oof_rows']:,} wierszy; zakres "
        f"{payload['model']['new_oof_start']}–{payload['model']['new_oof_end']}. "
        f"Okres rynku {payload['evaluation_period']} znajduje się w foldzie "
        f"{training['market_evaluation_model_fold']} nowego modelu.",
        "",
        *fold_lines,
        "",
        f"Dla wspólnych okien ostatnia etykieta treningowa foldu była dostępna "
        f"{training['latest_training_label_available_before_market_evaluation']} przed "
        "pierwszą decyzją Polymarket z ocenianego zakresu. Na granicy każdego zapisanego "
        "foldu wykryto 5 wierszy, których target t+5m (dostępny w t+6m) nie był jeszcze "
        "dostępny przy pierwszej predykcji t+1m. Żadne z ocenianych okien rynkowych nie "
        "leży w tych pierwszych pięciu minutach foldu; same OOF-y z tych granic nie są "
        "użyte jako niezależne predykcje.",
        "",
        "Najważniejsze ograniczenie: early stopping dla każdego zapisanego OOF foldu "
        "wybierał `best_iteration` na etykietach własnego testu (dla fold 8: "
        f"{training['evaluation_fold_best_iteration']}); końcowe {training['final_model_n_estimators']} "
        "drzew wybrano jako średnią najlepszych iteracji z 10 foldów. Fold 8 pokrywa "
        "oceniany okres, więc te OOF-y nie są niezależne od wyboru iteracji.",
        "",
        "### Okresy strojenia i udział ocenianych danych",
        "",
        f"Zapisane parametry LGBM pochodzą z artefaktu `{LGBM_TUNING_PATH.relative_to(ROOT).as_posix()}` "
        f"(utworzony {selection['lgbm_optuna']['created_utc']}, trial "
        f"{selection['lgbm_optuna']['best_trial']}/{selection['lgbm_optuna']['trials_requested']}); "
        "najlepsze parametry zgadzają się z metadanymi modelu. Zapis studium podaje "
        f"{selection['lgbm_optuna']['saved_rows_after_target_notna']:,} wierszy przed "
        f"filtrem (aktywny: {selection['lgbm_optuna']['saved_input_row_filter_enabled']}), "
        "10 foldów i liczbę wierszy, ale "
        "nie podaje zakresu timestampów ani hasha wejścia; zakres foldów poniżej jest "
        "rekonstrukcją na bieżącym zbiorze o zgodnej liczbie wierszy.",
        *selection_lines,
        "",
        f"Selekcja cech powstała {selection['feature_selector']['created_utc']}: "
        f"{selection['feature_selector']['input_feature_count']} wejściowych cech, "
        f"10 chronologicznych foldów, ranking recency-weighted/permutation; wybrano "
        f"{selection['feature_selector']['recommended_feature_count']}. Optuna volume profile "
        f"(trial {selection['volume_profile_optuna']['best_trial']}/500) i reaction profile "
        f"(trial {selection['reaction_profile_optuna']['best_trial']}/500) miały po 10 foldów "
        "i używały targetu `target_5m_candle_up`; najlepsze konfiguracje zgadzają się "
        "z metadanymi nowego modelu.",
        "Artefakty tych studiów nie zapisują hashy ani zakresów timestampów danych, "
        "dlatego okres foldów został odtworzony z aktualnego model-ready po zapisanych "
        "rozmiarach próbek i algorytmie walk-forward.",
        "",
        f"Dopasowanie wskaźników używa metryki `{selection['indicator_fit']['metric']}` "
        f"na {selection['indicator_fit']['metric_segments_count']} segmentach chronologicznych "
        f"źródła {selection['indicator_fit']['source_start']}–{selection['indicator_fit']['source_end']}; "
        f"w każdym: {selection['indicator_fit']['metric_train_fraction_per_segment']:.0%} "
        f"prefiksu, gap {selection['indicator_fit']['metric_gap_rows']} wierszy i metryka "
        f"na końcowym fragmencie; wymagano {selection['indicator_fit']['minimum_valid_segments']} "
        "poprawnych segmentów.",
        *indicator_lines,
        "",
        "",
        f"Liczebności: wejściowy Kacho miał {source_exclusions['source_market_rows']:,} rynków; "
        f"poprzedni pipeline zachował {source_exclusions['common_market_rows']:,}. "
        f"Pozostałe {source_exclusions['excluded_market_rows']} odrzucono: {exclusion_text}. "
        f"Dokładny join nowego OOF pokrył "
        f"{joins['exact_timestamp_matches']:,}; brak połączeń: "
        f"{joins['common_rows_without_new_oof']:,}; połączenie nie używa pozycji wiersza. "
        f"Próbka główna to {len(evaluation):,} okien: pierwsze {payload['warmup_rows']:,} "
        "służy jako wcześniejsze warm-up/tuning, a trzy outer foldy obejmują resztę. "
        "Kanoniczny OOF został nadpisany nowym treningiem; stare prawdopodobieństwa "
        "zachowano w zapisanym wyniku poprzedniego eksperymentu, którego manifest podaje "
        "oryginalny SHA-256 starego OOF.",
        "",
        f"Źródło decyzji to pierwszy poprawny snapshot Kacho po dostępności OOF; "
        f"timestamp snapshotu jest równy lub późniejszy od decyzji, opóźnienie ma "
        f"medianę {quote['delay_ms_p50']:.0f} ms i maksimum {quote['delay_ms_max']:.0f} ms. "
        "Książka jest późniejszym zapisem historycznym z nieznanym wiekiem feedu. "
        "Prognozy i symulacja wykonania są więc oddzielnymi wynikami.",
        "",
        "## Zgodność offline/live i następny krok",
        "",
        f"`run.py` ładuje model `{payload['live_model_run_id']}` z `active.json`; "
        "ta konfiguracja istniała przed tym zadaniem i nie została przeze mnie zmieniona. "
        f"Pseudo-live replay obejmował {live['audit_start']}–{live['audit_end']}: "
        f"{live['decision_rows']} decyzji, {live['feature_count']} cech i rozgrzewkę "
        f"{live['bootstrap_rows']} świec (wymagane minimum {live['required_stable_window']}). "
        f"OHLCV pochodził wyłącznie z lokalnego CSV; odchylenie od zapisanych OHLCV "
        f"wyniosło {live['live_ohlcv_vs_stored_max_abs_diff']:.1f}. Kolejność cech "
        f"zgadza się z metadanymi modelu (SHA-256 `{live['feature_order_sha256']}`); "
        f"braki/statusy finite różniły się w {live['finite_status_mismatch_values']} "
        f"wartościach. Nie było rozbieżności sygnału ({live['rows_with_signal_mismatch']}) "
        f"ani decyzji biznesowej ({live['rows_with_business_decision_mismatch']}).",
        f"Różnica predykcji max |Δp|={live['max_proba_up_abs_diff']:.8f}, średnia "
        f"{live['mean_proba_up_abs_diff']:.8f}; {live['rows_with_proba_diff_gt_tol']} "
        f"wiersz(e) przekroczyły tolerancję {live['probability_diff_tolerance']:.1e}. "
        f"Największy taki przypadek: {live_gap_text}. Replay wykorzystał "
        "zakotwiczone stany volume/reaction profile z lokalnej historii; nie używał REST.",
        "Nie uruchamiałem aktywnego tradera, zleceń, redeemów ani wiadomości.",
        "",
        "Werdykt: obecne dowody nie uzasadniają wniosku o dodatnim wyniku po kosztach "
        "ani zmiany podejścia modelowego na podstawie tego zbioru. Następny krok: "
        "non-trading shadow na przyszłych oknach, z zamrożonym modelem i polityką. "
        "W tym samym czasie zapisuj wiek/zmianę snapshotu, odbiór, wysłanie/ack/fill "
        "i rzeczywiste opłaty; po zebraniu nowych, nietkniętych etykiet wykonaj jeden "
        "niezależny test bez dostrajania progów.",
        "",
        f"Manifest: [`{MANIFEST_PATH.relative_to(ROOT).as_posix()}`]({MANIFEST_PATH.name}). "
        f"Artefakty wyliczenia: `{payload['output_directory']}`. "
        "Odtworzenie (bez argumentów CLI): `python run_btc_new_model_comparison.py`.",
        "",
    ])


def _effective_config_audit(new_meta):
    from features.reaction_profile_fixed_grid import (
        normalize_config as normalize_reaction_config,
        validate_reaction_profile_model_metadata as validate_reaction_metadata,
    )
    from features.volume_profile_fixed_range import (
        normalize_config as normalize_volume_config,
        validate_volume_profile_model_metadata as validate_volume_metadata,
    )

    current = json.loads((ROOT / "configs/modeling.json").read_text(encoding="utf-8"))["profiles"]["BTC"]
    columns = list(new_meta["feature_columns"])
    validate_volume_metadata(
        new_meta, feature_columns=columns,
        cfg=normalize_volume_config(current.get("volume_profile_fixed_range")),
        source_label=str(NEW_META_PATH),
    )
    validate_reaction_metadata(
        new_meta, feature_columns=columns,
        cfg=normalize_reaction_config(current.get("reaction_profile_fixed_grid")),
        source_label=str(NEW_META_PATH),
    )
    return {
        "volume_profile_matches_metadata": True,
        "reaction_profile_matches_metadata": True,
        "manual_exclusion_flag": bool(current["feature_selection"].get("excluded_feature_names_enabled")),
        "manual_exclusion_names_present_but_disabled": len(current["feature_selection"].get("excluded_feature_names", [])),
        "training_config_snapshot_present": bool(new_meta.get("config_snapshot")),
        "training_config_path_present": bool(new_meta.get("config_path")),
    }


def _walk_forward_validation_overlaps(opened, evaluation_start, evaluation_end, n_splits, ratio):
    row_count = len(opened)
    test_rows = int(math.floor(row_count / (n_splits + 1.0 / ratio)))
    train_rows = int(math.floor(test_rows / ratio))
    opened_index = pd.Index(opened)
    eval_start_idx = int(opened_index.searchsorted(evaluation_start, side="left"))
    eval_end_idx = int(opened_index.searchsorted(evaluation_end, side="right"))
    overlaps = []
    for fold_id in range(n_splits):
        train_start = fold_id * test_rows
        train_end = train_start + train_rows
        valid_start = train_end
        valid_end = valid_start + test_rows
        if valid_end > row_count:
            raise ValueError(f"Selection fold {fold_id} exceeds its filtered dataset")
        overlap_start = max(valid_start, eval_start_idx)
        overlap_end = min(valid_end, eval_end_idx)
        if overlap_end <= overlap_start:
            continue
        overlaps.append({
            "fold_id": fold_id,
            "train_start": opened[train_start],
            "train_end": opened[train_end - 1],
            "validation_start": opened[valid_start],
            "validation_end": opened[valid_end - 1],
            "evaluation_overlap_start": opened[overlap_start],
            "evaluation_overlap_end": opened[overlap_end - 1],
            "evaluation_overlap_rows": int(overlap_end - overlap_start),
        })
    return {
        "rows": int(row_count),
        "n_splits": int(n_splits),
        "test_to_train_ratio": float(ratio),
        "evaluation_validation_overlaps": overlaps,
        "evaluation_in_validation": bool(overlaps),
    }


def build_training_selection_audit(evaluation):
    selector = json.loads(SELECTOR_PATH.read_text(encoding="utf-8"))
    volume_profile = json.loads(VOLUME_PROFILE_TUNING_PATH.read_text(encoding="utf-8"))
    reaction_profile = json.loads(REACTION_PROFILE_TUNING_PATH.read_text(encoding="utf-8"))
    lgbm_tuning = json.loads(LGBM_TUNING_PATH.read_text(encoding="utf-8"))
    new_meta = json.loads(NEW_META_PATH.read_text(encoding="utf-8"))
    indicator_config = json.loads(INDICATOR_FIT_CONFIG_PATH.read_text(encoding="utf-8"))
    ready = pd.read_parquet(
        MODEL_READY_PATH,
        columns=["Opened", "target_5m_candle_up", "target_5m_weight"],
    )
    ready["Opened"] = _utc(ready["Opened"])
    evaluation_start = _utc(evaluation["Opened"]).min()
    evaluation_end = _utc(evaluation["Opened"]).max()

    selector_min_weight = float(selector["row_filter"]["min_weight"])
    filtered = ready.loc[
        ready["target_5m_candle_up"].notna()
        & pd.to_numeric(ready["target_5m_weight"], errors="coerce").ge(selector_min_weight),
        "Opened",
    ].reset_index(drop=True)
    selector_rows = int(selector["row_filter"]["rows_after"])
    volume_rows = int(volume_profile["dataset"]["decision_row_filter"]["rows_after"])
    if len(filtered) != selector_rows or len(filtered) != volume_rows:
        raise ValueError(
            "Current filtered model-ready rows do not match saved selector/volume-profile inputs"
        )

    selector_ratio = float(selector["walk_forward_test_to_train_ratio"])
    selector_cv = _walk_forward_validation_overlaps(
        filtered,
        evaluation_start,
        evaluation_end,
        n_splits=int(selector["topk_n_splits"]),
        ratio=selector_ratio,
    )
    selector_cv["created_utc"] = selector["created_utc"]
    selector_cv["ranking_method"] = selector["ranking_method"]
    selector_cv["input_feature_count"] = int(selector["input_feature_count"])
    selector_cv["recommended_feature_count"] = int(selector["recommended_k"])
    selector_cv["row_filter_min_weight"] = selector_min_weight

    volume_optimization = volume_profile["optimization"]
    volume_cv = _walk_forward_validation_overlaps(
        filtered,
        evaluation_start,
        evaluation_end,
        n_splits=int(volume_optimization["cv_folds"]),
        ratio=float(volume_optimization["walk_forward_test_to_train_ratio"]),
    )
    volume_cv.update({
        "created_utc": volume_profile["created_utc"],
        "best_trial": int(volume_profile["best_trial"]["trial_number"]),
        "trials_requested": int(volume_optimization["n_trials_requested"]),
        "objective": volume_profile["objective"],
        "best_trial_objective_value": float(volume_profile["best_trial"]["objective_value"]),
        "best_trial_feature_count": int(volume_profile["best_trial"]["feature_count"]),
        "fold_recency_weighting": volume_optimization["fold_recency_weighting"],
    })

    reaction_optimization = reaction_profile["optimization"]
    reaction_cv = _walk_forward_validation_overlaps(
        filtered,
        evaluation_start,
        evaluation_end,
        n_splits=int(reaction_optimization["cv_folds"]),
        ratio=float(reaction_optimization["walk_forward_test_to_train_ratio"]),
    )
    reaction_cv.update({
        "created_utc": reaction_profile["created_utc"],
        "best_trial": int(reaction_profile["best_trial"]["trial_number"]),
        "trials_requested": int(reaction_optimization["n_trials_requested"]),
        "objective": reaction_profile["objective"],
        "best_trial_objective_value": float(reaction_profile["best_trial"]["objective_value"]),
        "best_trial_feature_count": int(reaction_profile["best_trial"]["feature_count"]),
        "fold_recency_weighting": reaction_optimization["fold_recency_weighting"],
    })

    lgbm_source = ready.loc[ready["target_5m_candle_up"].notna(), "Opened"].iloc[
        : int(lgbm_tuning["rows_after_target_notna"])
    ].reset_index(drop=True)
    if len(lgbm_source) != int(lgbm_tuning["rows_after_target_notna"]):
        raise ValueError("Current model-ready prefix does not match saved LGBM Optuna row count")
    lgbm_cv = _walk_forward_validation_overlaps(
        lgbm_source,
        evaluation_start,
        evaluation_end,
        n_splits=int(lgbm_tuning["cv_folds"]),
        ratio=float(lgbm_tuning["walk_forward_test_to_train_ratio"]),
    )
    saved_lgbm_params = lgbm_tuning["best_trial"]["params"]
    current_lgbm_params = new_meta["model_hyperparameters"]["cv_optuna_params"]
    lgbm_cv.update({
        "created_utc": lgbm_tuning["created_utc"],
        "study_name": lgbm_tuning["study_name"],
        "best_trial": int(lgbm_tuning["best_trial"]["number"]),
        "trials_requested": int(lgbm_tuning["n_trials_requested"]),
        "saved_rows_after_target_notna": int(lgbm_tuning["rows_after_target_notna"]),
        "best_params_match_new_model_metadata": saved_lgbm_params == current_lgbm_params,
        "input_timestamp_range_recorded": False,
        "saved_input_row_filter_enabled": bool(
            lgbm_tuning.get("decision_row_filter", {}).get("enabled", False)
        ),
        "cv_fold_recency_weighting": lgbm_tuning["fold_recency_weighting"],
    })

    from features.reaction_profile_fixed_grid import normalize_config as normalize_reaction_config
    from features.volume_profile_fixed_range import normalize_config as normalize_volume_config

    volume_cv["best_config_matches_new_model_metadata"] = (
        normalize_volume_config(volume_profile["best_volume_profile_fixed_range"])
        == normalize_volume_config(new_meta["volume_profile_fixed_range"])
    )
    reaction_cv["best_config_matches_new_model_metadata"] = (
        normalize_reaction_config(reaction_profile["best_reaction_profile_fixed_grid"])
        == normalize_reaction_config(new_meta["reaction_profile_fixed_grid"])
    )

    indicator_pair_config = indicator_config["pairs"]["default"]
    interval_cfg = indicator_pair_config["intervals"]["1m"]
    indicator_settings = {**indicator_pair_config, **interval_cfg}
    from utils.data import drop_frozen_ohlc_blocks

    raw_ohlc = pd.read_csv(
        RAW_OHLCV_PATH,
        usecols=["Opened", "Open", "High", "Low", "Close"],
    )
    raw_ohlc, frozen_blocks = drop_frozen_ohlc_blocks(
        raw_ohlc,
        raw_config=interval_cfg.get(
            "drop_frozen_ohlc_blocks",
            indicator_pair_config.get("drop_frozen_ohlc_blocks"),
        ),
    )
    raw_opened = pd.to_datetime(raw_ohlc["Opened"], utc=True).reset_index(drop=True)
    metric_segments = int(indicator_settings["metric_segments_count"])
    train_fraction = float(indicator_settings["metric_train_frac"])
    metric_gap = int(indicator_settings["metric_gap"])
    raw_eval_start = int(raw_opened.searchsorted(evaluation_start, side="left"))
    raw_eval_end = int(raw_opened.searchsorted(evaluation_end, side="right"))
    segment_rows = []
    for segment_id in range(metric_segments):
        segment_start = segment_id * len(raw_opened) // metric_segments
        segment_end = (segment_id + 1) * len(raw_opened) // metric_segments
        fit_end = segment_start + int(train_fraction * (segment_end - segment_start))
        score_start = min(segment_end, fit_end + metric_gap)
        calibration_overlap = max(0, min(fit_end, raw_eval_end) - max(segment_start, raw_eval_start))
        gap_overlap = max(0, min(score_start, raw_eval_end) - max(fit_end, raw_eval_start))
        validation_overlap = max(0, min(segment_end, raw_eval_end) - max(score_start, raw_eval_start))
        if calibration_overlap or gap_overlap or validation_overlap:
            segment_rows.append({
                "segment_id": segment_id,
                "segment_start": raw_opened.iloc[segment_start],
                "segment_end": raw_opened.iloc[segment_end - 1],
                "calibration_start": raw_opened.iloc[segment_start],
                "calibration_end": raw_opened.iloc[fit_end - 1],
                "metric_validation_start": raw_opened.iloc[score_start],
                "metric_validation_end": raw_opened.iloc[segment_end - 1],
                "evaluation_rows_in_calibration_prefix": int(calibration_overlap),
                "evaluation_rows_in_metric_gap": int(gap_overlap),
                "evaluation_rows_in_metric_validation": int(validation_overlap),
            })
    indicator_audit = {
        "source_start": raw_opened.min(),
        "source_end": raw_opened.max(),
        "frozen_ohlc_rows_removed": int(frozen_blocks["rows_removed"]),
        "target_horizons_minutes": indicator_settings["proxy_target_horizonts"],
        "target_price_column": indicator_settings["proxy_target_price_col"],
        "metric": indicator_settings["metric_name"],
        "metric_segments_count": metric_segments,
        "metric_train_fraction_per_segment": train_fraction,
        "metric_gap_rows": metric_gap,
        "minimum_valid_segments": int(indicator_settings["min_valid_segments"]),
        "evaluation_overlapping_segments": segment_rows,
    }
    return {
        "evaluation_opened_start": evaluation_start,
        "evaluation_opened_end": evaluation_end,
        "lgbm_optuna": lgbm_cv,
        "feature_selector": selector_cv,
        "volume_profile_optuna": volume_cv,
        "reaction_profile_optuna": reaction_cv,
        "indicator_fit": indicator_audit,
    }


def run_live_parity_audit():
    import audit_feature_readiness as audit

    rest_source_before = audit.AUDIT_USE_REST_LIVE_OHLCV_SOURCE
    audit.AUDIT_USE_REST_LIVE_OHLCV_SOURCE = False
    try:
        results = audit.run_live_modeling_feature_audit(
            days_back=1,
            bootstrap_candles=21_600,
            max_steps=1_440,
            max_keep=21_600,
            model_meta_path=NEW_META_PATH,
            parquet_path=MODEL_READY_PATH,
            use_anchor_vp_state=True,
            overwrite_anchor_vp_state=False,
            use_anchor_rp_state=True,
            overwrite_anchor_rp_state=False,
        )
    finally:
        audit.AUDIT_USE_REST_LIVE_OHLCV_SOURCE = rest_source_before

    summary = results["summary"].to_dict()
    report = results["live_vs_stored_report"]
    step_summary = results["step_summary_df"]
    feature_columns = list(results["feature_columns"])
    model_feature_columns = json.loads(NEW_META_PATH.read_text(encoding="utf-8"))["feature_columns"]
    if feature_columns != model_feature_columns:
        raise ValueError("Pseudo-live feature order differs from the selected model metadata")

    probability_tolerance = float(audit.PREDICTION_DIFF_TOL)
    above_tolerance = step_summary.loc[
        step_summary["proba_up_abs_diff"].gt(probability_tolerance)
    ]
    largest_gap = None
    if not above_tolerance.empty:
        row = above_tolerance.sort_values("proba_up_abs_diff").iloc[-1]
        largest_gap = {
            key: row[key]
            for key in (
                "Opened", "proba_up_abs_diff", "feature_max_abs_diff",
                "feature_mean_abs_diff", "worst_feature", "live_proba_up",
                "stored_proba_up", "signal_mismatch", "business_decision_mismatch",
            )
            if key in row.index
        }

    source_paths = sorted({
        str(value) for value in results["feature_builder_frame"]["builder_source"].dropna()
        if str(value).strip()
    })
    source_hashes = {}
    for value in source_paths:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            source_hashes[path.relative_to(ROOT).as_posix()] = file_sha256(path)

    live_summary = {
        key: summary.get(key)
        for key in (
            "audit_start", "audit_end", "bootstrap_rows", "audit_rows_total_1m",
            "decision_rows", "required_stable_window", "live_ohlcv_source",
            "live_ohlcv_path", "live_ohlcv_rows", "live_ohlcv_vs_stored_max_abs_diff",
            "volume_profile_anchor_source", "reaction_profile_anchor_source",
            "use_anchor_vp_state", "use_anchor_rp_state", "feature_count",
            "rows_with_live_nonfinite", "rows_with_stored_nonfinite",
            "rows_with_finite_status_mismatch", "rows_with_signal_mismatch",
            "rows_with_business_decision_mismatch", "max_feature_abs_diff",
            "mean_feature_abs_diff", "max_proba_up_abs_diff",
            "mean_proba_up_abs_diff", "rows_with_proba_diff_gt_tol",
        )
    }
    live_summary.update({
        "probability_diff_tolerance": probability_tolerance,
        "feature_order_matches_model_metadata": True,
        "feature_order_sha256": hashlib.sha256("\n".join(feature_columns).encode("utf-8")).hexdigest(),
        "live_nonfinite_values": int(np.asarray(results["live_nonfinite_mask"]).sum()),
        "stored_nonfinite_values": int(np.asarray(results["modeling_nonfinite_mask"]).sum()),
        "finite_status_mismatch_values": int(np.asarray(results["finite_status_mismatch_mask"]).sum()),
        "largest_probability_gap_above_tolerance": largest_gap,
        "feature_source_file_count": len(source_hashes),
        "feature_source_sha256": source_hashes,
        "anchor_vp_state_path": summary.get("anchor_vp_state_path"),
        "anchor_rp_state_path": summary.get("anchor_rp_state_path"),
    })
    return live_summary


def run():
    common = pd.read_parquet(COMMON_PATH)
    new_oof = pd.read_parquet(
        OOF_PATH, columns=["Opened", "target_5m_candle_up", "oof_pred_proba_up"]
    )
    aligned, join_counts = align_new_oof_to_markets(common, new_oof)
    join_counts["source_market_exclusions"] = audit_source_market_exclusions(common)
    if int(join_counts["exact_timestamp_matches"]) != 15_677:
        raise ValueError(f"Unexpected matched common market count: {join_counts}")
    aligned = aligned.loc[aligned["new_oof_join"].eq("both")].copy()
    if not aligned["p_old_model_up"].notna().all():
        raise ValueError("Previous model probabilities are missing from the pinned common frame")
    if not aligned["target_polymarket_up"].eq(aligned["polymarket_outcome_up"]).all():
        raise ValueError("Validated official settlement and scoring target differ")
    # The historical common frame carries the previous model's values. Preserve
    # them separately, then replace the pipeline input with the exact new OOF.
    aligned["p_model_up"] = aligned["p_new_model_up"]

    new_meta = json.loads(NEW_META_PATH.read_text(encoding="utf-8"))
    old_meta = json.loads(OLD_META_PATH.read_text(encoding="utf-8"))
    fold_reports, positions = build_model_fold_audit(new_meta, new_oof)
    position_by_opened = pd.Series(positions, index=_utc(new_oof["Opened"]))
    market_positions = aligned["Opened"].map(position_by_opened)
    if market_positions.isna().any():
        raise ValueError("Common market Opened timestamps are missing from the exact OOF fold map")
    aligned["new_model_fold"] = np.searchsorted(
        np.array([int(fold["test_start"]) for fold in new_meta["walk_forward_folds"]]),
        market_positions.to_numpy(dtype=int), side="right",
    ) - 1
    if aligned["new_model_fold"].nunique() != 1:
        raise ValueError("Polymarket common period is not held by one saved new-model OOF fold")

    live_config = json.loads((ROOT / "configs/runtime/active.json").read_text(encoding="utf-8"))
    active_meta = live_config["assets"]["BTC"]["artifacts"]["model_meta_path"].replace("\\", "/")
    if active_meta != NEW_META_PATH.relative_to(ROOT).as_posix():
        raise ValueError("run.py active BTC metadata does not identify the new model under review")
    config_audit = _effective_config_audit(new_meta)

    input_hashes = {}
    for path in _input_paths():
        if not path.is_file():
            raise FileNotFoundError(f"Required manifest input is missing: {path}")
        input_hashes[path.relative_to(ROOT).as_posix()] = file_sha256(path)

    signature = {
        "old_model_id": OLD_RUN_ID,
        "new_model_id": NEW_RUN_ID,
        "inputs_sha256": input_hashes,
        "decision_contract": {
            "market_join": "exact UTC Opened to market ID and market_start_utc; no row-position join",
            "quote_selection": "existing first-forward Kacho snapshot at 0s OOF latency; max 2s tolerance",
            "quality_target": "official Polymarket outcome",
            "auxiliary_target": "exact saved target_5m_candle_up / Binance proxy",
        },
        "market_model": {
            "family": "StandardScaler + L2 LogisticRegression",
            "features": "17 contemporaneous Kacho book price and top-size features",
            "C_grid": [0.01, 0.1, 1.0, 10.0],
            "outer": "three later chronological 20% blocks after 40% warm-up",
        },
        "bootstrap": {
            "block_days": BOOTSTRAP_BLOCK_DAYS,
            "replications": BOOTSTRAP_REPLICATIONS,
            "seed": BOOTSTRAP_SEED,
        },
        "portfolio": {
            "initial_usdc": INITIAL_BANKROLL_USDC,
            "fixed_stake_usdc": FIXED_STAKE_USDC,
            "expected_pnl_buffer_usdc": EXPECTED_PNL_BUFFER_USDC,
            "execution_delays_s": list(EXECUTION_DELAYS),
            "settlement": "max(official resolution, market end); no additional redemption delay",
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(signature, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()
    output_dir = OUTPUT_ROOT / "runs" / fingerprint[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    signature["fingerprint"] = fingerprint
    signature["output_directory"] = output_dir.relative_to(ROOT).as_posix()
    _write_json(output_dir / "inputs.json", signature)

    model_output_dir = output_dir / "market_models"
    walk_forward_models({COMMON_LATENCY_SECONDS: aligned}, model_output_dir)
    new_predictions = pd.read_parquet(
        model_output_dir / f"latency_{COMMON_LATENCY_SECONDS}s/out_of_fold_probabilities.parquet"
    )
    evaluation = new_predictions.merge(
        aligned[[
            "condition_id", "market_start_utc", "Opened", "p_old_model_up",
            "p_new_model_up", "target_binance_proxy_up", "polymarket_outcome_up",
            "second_layer_computed_at", "p_market_mid", "decision_id",
        ]],
        on=["condition_id", "market_start_utc"], how="left", validate="one_to_one",
        indicator="market_join",
    )
    if not evaluation["market_join"].eq("both").all():
        raise ValueError("Outer predictions did not rejoin to exact market IDs and timestamps")
    if len(evaluation) != 9_407:
        raise ValueError(f"Unexpected common outer evaluation rows: {len(evaluation)}")
    if not evaluation["p_old_model_up"].notna().all():
        raise ValueError("Old model probability missing on shared comparison rows")
    if not np.allclose(evaluation["p_new_model_up"], evaluation["oof_raw"], atol=1e-12, rtol=0):
        raise ValueError("Second-layer run did not use the aligned new BTC OOF values")

    old_saved = pd.read_parquet(
        OLD_EVALUATION_PATH,
        columns=["condition_id", "market_start_utc", "oof_raw", "market_only", "fold"],
    )
    previous = evaluation[["condition_id", "market_start_utc"]].merge(
        old_saved, on=["condition_id", "market_start_utc"], how="left",
        validate="one_to_one", indicator="old_output_join",
    )
    if not previous["old_output_join"].eq("both").all():
        raise ValueError("Prior experiment predictions do not exactly match the common windows")
    if not np.allclose(previous["oof_raw"], evaluation["p_old_model_up"], atol=1e-12, rtol=0):
        raise ValueError("Old raw OOF stored in the previous output differs from common-frame values")
    if not np.array_equal(previous["fold"].to_numpy(), evaluation["fold"].to_numpy()):
        raise ValueError("Prior and current market-model outer partitions differ")
    if not np.allclose(previous["market_only"], evaluation["market_only"], atol=1e-12, rtol=0):
        raise ValueError("Reconstructed MARKET_ONLY probabilities differ from pinned prior results")

    for key, column in _calibration_columns(evaluation).items():
        evaluation[key] = pd.to_numeric(evaluation[column], errors="coerce")
    evaluation["old_btc"] = evaluation["p_old_model_up"].astype(float)
    evaluation["new_btc"] = evaluation["p_new_model_up"].astype(float)
    evaluation["target_polymarket_up"] = evaluation["target_polymarket_up"].astype(int)
    evaluation["target_binance_proxy_up"] = evaluation["target_binance_proxy_up"].astype(int)

    bootstrap = paired_block_bootstrap_differences(
        evaluation,
        "target_polymarket_up",
        {
            "new_minus_old": ("new_btc", "old_btc"),
            "new_minus_market_only": ("new_btc", "market_only"),
            "market_plus_new_minus_market_only": ("market_plus_new_btc", "market_only"),
        },
    )
    wide_new_btc = score_binary(aligned["target_polymarket_up"], aligned["p_new_model_up"])
    main_disagreements = int(
        evaluation["target_polymarket_up"].ne(evaluation["target_binance_proxy_up"]).sum()
    )
    wide_disagreements = int(
        aligned["target_polymarket_up"].ne(aligned["target_binance_proxy_up"]).sum()
    )

    portfolio_frame = evaluation.copy()
    portfolio_frame["p_model_up"] = portfolio_frame["oof_raw"]
    probability_columns = {
        "market_only": "market_only",
        "new_btc_platt": "new_btc_platt",
        "market_plus_new_btc": "market_plus_new_btc",
    }
    portfolio = {}
    for delay in EXECUTION_DELAYS:
        future = None
        if delay:
            future = _execution_quotes(portfolio_frame, QUOTE_PATH, delay).set_index("condition_id")
        portfolio[str(delay)] = {}
        for name, probability_column in probability_columns.items():
            result, ledger, path_frame = simulate_fixed_policy(
                portfolio_frame,
                probability_column,
                execution_delay=delay,
                bankroll=INITIAL_BANKROLL_USDC,
                future_quotes=future,
                min_expected_pnl=EXPECTED_PNL_BUFFER_USDC,
            )
            if not np.isclose(
                    result["final_balance"], INITIAL_BANKROLL_USDC + result["realized_pnl"],
                    atol=1e-8, rtol=0,
            ):
                raise AssertionError("Portfolio final cash does not reconcile to realized PnL")
            result["ending_available_cash"] = float(result["ending_available_cash"])
            result["ending_locked_cost"] = float(result["ending_locked_cost"])
            result["minimum_expected_pnl_buffer_usdc"] = EXPECTED_PNL_BUFFER_USDC
            result["ending_open_positions"] = int(result["ending_open_positions"])
            portfolio[str(delay)][name] = result
            ledger.to_parquet(output_dir / f"portfolio_{delay}s_{name}_trades.parquet", index=False)
            path_frame.to_parquet(output_dir / f"portfolio_{delay}s_{name}_path.parquet", index=False)

    quote_delay_ms = pd.to_numeric(aligned["quote_delay_ms"], errors="coerce")
    fold_id = int(aligned["new_model_fold"].iloc[0])
    target_model_fold = next(item for item in new_meta["walk_forward_details"]["optuna"]
                             if int(item["fold_id"]) == fold_id)
    target_fold_info = next(item for item in new_meta["walk_forward_folds"]
                            if int(item["fold_id"]) == fold_id)
    train_last_opened = _utc(pd.read_parquet(
        MODEL_READY_PATH, columns=["Opened"],
    )["Opened"].iloc[int(target_fold_info["train_end"]) - 1])
    latest_train_label_available = train_last_opened + pd.Timedelta(
        minutes=TARGET_LABEL_AVAILABLE_MINUTES_AFTER_OPEN
    )
    first_eval_decision = _utc(evaluation["decision_available_at"]).min()
    if not latest_train_label_available < first_eval_decision:
        raise ValueError("A new-model training label was unavailable at the first scored market decision")

    fold_summaries = {}
    for fold, block in evaluation.groupby("fold", sort=True):
        fold_summaries[str(int(fold))] = {
            "rows": int(len(block)),
            "start": _utc(block["market_start_utc"]).min(),
            "end": _utc(block["market_start_utc"]).max(),
            "scores": {
                key: score_binary(block["target_polymarket_up"], block[key])
                for key in ("old_btc", "new_btc", "market_only", "market_plus_new_btc")
            },
        }
    evaluation.to_parquet(output_dir / "shared_market_evaluation.parquet", index=False)
    wide_scores = {"new_btc_raw": wide_new_btc}
    quality_scores = {
        "official_settlement": {
            key: score_binary(evaluation["target_polymarket_up"], evaluation[key])
            for key in ("old_btc", "new_btc", "market_only")
        },
        "binance_target_auxiliary": {
            key: score_binary(evaluation["target_binance_proxy_up"], evaluation[key])
            for key in ("old_btc", "new_btc", "market_only")
        },
        "folds": fold_summaries,
    }
    quote_audit = {
        "source": "Kacho first-forward saved book snapshot",
        "rows": int(len(aligned)),
        "delay_ms_min": float(quote_delay_ms.min()),
        "delay_ms_p50": float(quote_delay_ms.median()),
        "delay_ms_p95": float(quote_delay_ms.quantile(0.95)),
        "delay_ms_max": float(quote_delay_ms.max()),
        "quote_timestamp_at_or_after_model_decision": bool(
            aligned["timestamp_utc"].ge(aligned["decision_available_at"]).all()
        ),
        "official_settlement_sources": sorted(aligned["outcome_source"].dropna().astype(str).unique().tolist()),
        "quote_sources": sorted(aligned["source_quote"].dropna().astype(str).unique().tolist()),
    }
    warmup_rows = int(len(aligned) - len(evaluation))
    training_selection_audit = build_training_selection_audit(evaluation)
    training_audit = {
        "market_evaluation_model_fold": fold_id,
        "evaluation_fold_best_iteration": target_model_fold["best_iteration"],
        "final_model_n_estimators": int(new_meta["model_hyperparameters"]["base"]["n_estimators_final"]),
        "final_iteration_is_mean_of_fold_best_iterations": True,
        "evaluation_fold_rows": int(target_fold_info["test_end"] - target_fold_info["test_start"]),
        "latest_training_label_available_before_market_evaluation": latest_train_label_available,
        "first_market_decision": first_eval_decision,
        "days_between_training_label_and_market_evaluation": float(
            (first_eval_decision - latest_train_label_available).total_seconds() / 86400
        ),
        "early_stopping_uses_same_fold_test": True,
        "lgbm_optuna_study_timestamp_range_recorded": False,
        "lgbm_optuna_params_match_saved_trial": training_selection_audit["lgbm_optuna"][
            "best_params_match_new_model_metadata"
        ],
        "selector_and_profile_validation_include_common_period": True,
    }
    target_disagreement = {
        "main_rows": int(len(evaluation)),
        "main_disagreements": main_disagreements,
        "main_rate": main_disagreements / len(evaluation),
        "wide_rows": int(len(aligned)),
        "wide_disagreements": wide_disagreements,
        "wide_rate": wide_disagreements / len(aligned),
    }
    verdict = {
        "new_vs_old": "lower log loss and Brier, but lower accuracy; retrospective development result" if (
            score_binary(evaluation.target_polymarket_up, evaluation.new_btc)["log_loss"]
            < score_binary(evaluation.target_polymarket_up, evaluation.old_btc)["log_loss"]
            and score_binary(evaluation.target_polymarket_up, evaluation.new_btc)["brier"]
            < score_binary(evaluation.target_polymarket_up, evaluation.old_btc)["brier"]
        ) else "did not improve both log loss and Brier",
        "incremental_information": "small lower log loss and Brier; development-only evidence" if (
            score_binary(evaluation.target_polymarket_up, evaluation.market_plus_new_btc)["log_loss"]
            < score_binary(evaluation.target_polymarket_up, evaluation.market_only)["log_loss"]
            and score_binary(evaluation.target_polymarket_up, evaluation.market_plus_new_btc)["brier"]
            < score_binary(evaluation.target_polymarket_up, evaluation.market_only)["brier"]
        ) else "did not improve both log loss and Brier",
        "portfolio": (
            "no robust incremental profitable value: at 1 s MARKET_ONLY ends at "
            f"${portfolio['1']['market_only']['final_balance']:.2f}, "
            f"MARKET_ONLY + new BTC at ${portfolio['1']['market_plus_new_btc']['final_balance']:.2f}, "
            f"and calibrated new BTC alone at ${portfolio['1']['new_btc_platt']['final_balance']:.2f}"
        ),
    }
    old_oof_documented_sha = "51ea42bf9527b470340deab8a2578e61320ade34f4d42bca9646b7d9bc8c15fd"
    model = {
        "old_model_id": OLD_RUN_ID,
        "new_model_id": NEW_RUN_ID,
        "target": new_meta["target_col"],
        "old_feature_count": int(old_meta["feature_count"]),
        "new_feature_count": int(new_meta["feature_count"]),
        "feature_source_count": int(new_meta["feature_selection"]["source_count"]),
        "new_oof_rows": int(len(new_oof)),
        "new_oof_start": _utc(new_oof["Opened"]).min(),
        "new_oof_end": _utc(new_oof["Opened"]).max(),
        "new_oof_sha256": input_hashes[OOF_PATH.relative_to(ROOT).as_posix()],
        "old_oof_original_sha256_recorded_in_previous_report": old_oof_documented_sha,
        "old_oof_original_file_present": False,
        "old_market_probabilities_preserved_in": COMMON_PATH.relative_to(ROOT).as_posix(),
        "active_run.py_model_meta_path": active_meta,
    }
    report_payload = {
        "fingerprint": fingerprint,
        "output_directory": signature["output_directory"],
        "model": model,
        "join_counts": join_counts,
        "wide_new_btc": wide_new_btc,
        "scores": quality_scores,
        "evaluation": evaluation,
        "evaluation_period": _period_text(evaluation),
        "bootstrap": bootstrap,
        "portfolio": portfolio,
        "model_fold_audit": fold_reports,
        "training_audit": training_audit,
        "training_selection_audit": training_selection_audit,
        "config_audit": config_audit,
        "quote_audit": quote_audit,
        "target_disagreement": target_disagreement,
        "verdict": verdict,
        "market_model_C_grid": [0.01, 0.1, 1.0, 10.0],
        "warmup_rows": warmup_rows,
        "live_model_run_id": NEW_RUN_ID,
        "live_parity_summary": "run after this prediction/economics pass",
    }
    # Keep the serialized report payload compact; detailed rows are in the Parquet output.
    report_json = {key: value for key, value in report_payload.items() if key != "evaluation"}
    _write_json(output_dir / "evaluation.json", report_json)
    _write_json(output_dir / "bootstrap.json", bootstrap)
    print(json.dumps({
        "fingerprint": fingerprint,
        "output_directory": str(output_dir),
        "join_counts": join_counts,
        "quality": quality_scores["official_settlement"],
        "bootstrap": bootstrap,
        "portfolio_1s": portfolio["1"],
        "verdict": verdict,
        "quote_audit": quote_audit,
        "input_hash_count": len(input_hashes),
    }, indent=2, ensure_ascii=False, default=_json_default))
    return report_payload


def main():
    payload = run()
    payload["live_parity"] = run_live_parity_audit()
    live = payload["live_parity"]
    payload["live_parity_summary"] = (
        f"{live['decision_rows']} decisions; {live['rows_with_signal_mismatch']} signal "
        f"mismatches; {live['rows_with_proba_diff_gt_tol']} probabilities over tolerance"
    )
    output_dir = ROOT / payload["output_directory"]
    report_json = {key: value for key, value in payload.items() if key != "evaluation"}
    _write_json(output_dir / "evaluation.json", report_json)
    _write_json(output_dir / "live_parity.json", live)
    REPORT_PATH.write_text(render_report(payload), encoding="utf-8")
    manifest = {
        "status": "complete",
        "experiment": "BTC new model comparison vs predecessor and Polymarket",
        "fingerprint": payload["fingerprint"],
        "manifest": str(MANIFEST_PATH.relative_to(ROOT).as_posix()),
        "report": str(REPORT_PATH.relative_to(ROOT).as_posix()),
        "reproduce": "python run_btc_new_model_comparison.py",
        "execution": "offline only; no active trader, orders, redeems or messages",
        "inputs": json.loads((ROOT / payload["output_directory"] / "inputs.json").read_text(encoding="utf-8")),
        "model": payload["model"],
        "join_counts": payload["join_counts"],
        "target_disagreement": payload["target_disagreement"],
        "training_audit": payload["training_audit"],
        "training_selection_audit": payload["training_selection_audit"],
        "config_audit": payload["config_audit"],
        "quote_audit": payload["quote_audit"],
        "live_parity": live,
        "verdict": payload["verdict"],
    }
    _write_json(MANIFEST_PATH, manifest)
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")


if __name__ == "__main__":
    main()
