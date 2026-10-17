"""Fit one May-only scale for saved causal M1 corrections and score June."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

import run_btc_conditional_market_offset_20261018 as offset_source
import run_btc_preopen_research as research
import run_btc_preopen_t59_reassessment as replay


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_m1_correction_shrinkage_20261018.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / "data/analysis/polymarket/BTC/btc_m1_correction_shrinkage_20261018"
REPORT_PATH = ROOT / "reports/btc_preopen/btc_m1_correction_shrinkage_20261018/report.md"
MANIFEST_PATH = OUTPUT_DIR / "frozen_manifest.json"
PREDICTIONS_PATH = OUTPUT_DIR / "development_predictions.parquet"
ECONOMICS_PATH = OUTPUT_DIR / "june_economic_decisions.parquet"
RUN_PATH = OUTPUT_DIR / "run_metadata.json"
PRIOR_INVALID_DIR = OUTPUT_DIR / "prior_invalid_execution"
PRIOR_PREFORMAT_DIR = OUTPUT_DIR / "prior_valid_preformat"
T60_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_research_20261011/book_replay_t60_t59/t60_decision_books_with_source_coverage.parquet"
T59_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_research_20261011/book_replay_t60_t59/t59_ask_ladders_with_source_coverage.parquet"
SOURCE_CONFIG_PATH = ROOT / "configs/research/btc_conditional_market_offset_20261018.json"
SOURCE_REPORT_PATH = ROOT / "reports/btc_preopen/btc_market_basis_vol_interaction_20261017.md"
MAY_START = pd.Timestamp(CONFIG["periods"]["calibration_start_inclusive_utc"])
JUNE_START = pd.Timestamp(CONFIG["periods"]["evaluation_start_inclusive_utc"])
JULY_START = pd.Timestamp(CONFIG["periods"]["settlement_cutoff_exclusive_utc"])
STAKE_USD = float(CONFIG["economics"]["independent_gross_stake_usd"])
MIN_EV_FRACTION = float(CONFIG["economics"]["minimum_net_ev_fraction"])
BOOTSTRAP_REPLICATIONS = int(CONFIG["quality_uncertainty"]["valid_replications"])
BOOTSTRAP_SEED = int(CONFIG["quality_uncertainty"]["seed"])
SNAPSHOT_COLUMNS = (
    "condition_id", "archive_source", "archive_entry_hour_available", "archive_gap_after_fee_update",
    "fee_known", "fee_rate_bps", "fee_collection_mode", "no_future_event_at_entry",
    "no_future_source_event_at_entry", "up_best_ask", "up_ask_age_seconds", "up_quote_source_token_id",
    "up_quote_complemented", "up_ask_levels", "up_book_init_receive_ns", "up_ask_order_ambiguous",
    "up_bbo_ask_comparable", "up_bbo_ask_mismatch", "up_bid_ask_crossed", "archive_gap_after_up_book_init",
    "down_best_ask", "down_ask_age_seconds", "down_quote_source_token_id", "down_quote_complemented",
    "down_ask_levels", "down_book_init_receive_ns", "down_ask_order_ambiguous", "down_bbo_ask_comparable",
    "down_bbo_ask_mismatch", "down_bid_ask_crossed", "archive_gap_after_down_book_init",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _frame_sha256(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    hashed = pd.util.hash_pandas_object(frame.loc[:, columns].reset_index(drop=True), index=False, categorize=True)
    return hashlib.sha256(hashed.to_numpy(dtype=np.uint64).tobytes()).hexdigest()


def _loss(offset: np.ndarray, correction: np.ndarray, outcome: np.ndarray, alpha: float) -> float:
    margin = offset + float(alpha) * correction
    return float(np.mean(np.logaddexp(0.0, margin) - outcome * margin))


def _gradient(offset: np.ndarray, correction: np.ndarray, outcome: np.ndarray, alpha: float) -> float:
    return float(np.mean((expit(offset + float(alpha) * correction) - outcome) * correction))


def _probabilities_from_alpha(p_market: np.ndarray, correction: np.ndarray, alpha: float) -> np.ndarray:
    return expit(offset_source._logit(p_market) + float(alpha) * np.asarray(correction, dtype=np.float64))


def _fit_alpha(offset: np.ndarray, correction: np.ndarray, outcome: np.ndarray) -> dict:
    offset = np.asarray(offset, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    if not len(offset) or len(offset) != len(correction) or len(offset) != len(outcome):
        raise ValueError("Calibration arrays must be non-empty and row-aligned")
    if not np.isfinite(offset).all() or not np.isfinite(correction).all() or not np.isin(outcome, (0.0, 1.0)).all():
        raise ValueError("Calibration inputs contain invalid values")

    loss0 = _loss(offset, correction, outcome, 0.0)
    loss1 = _loss(offset, correction, outcome, 1.0)
    grad0 = _gradient(offset, correction, outcome, 0.0)
    grad1 = _gradient(offset, correction, outcome, 1.0)
    curvature0 = float(np.mean(expit(offset) * (1.0 - expit(offset)) * np.square(correction)))
    if curvature0 <= 0.0:
        alpha, rule = 0.0, "flat_objective_or_zero_correction"
    elif grad0 >= 0.0:
        alpha, rule = 0.0, "lower_boundary_gradient_nonnegative"
    elif grad1 <= 0.0:
        alpha, rule = 1.0, "upper_boundary_gradient_nonpositive"
    else:
        low, high = 0.0, 1.0
        for _ in range(80):
            middle = (low + high) / 2.0
            if _gradient(offset, correction, outcome, middle) < 0.0:
                low = middle
            else:
                high = middle
        alpha, rule = (low + high) / 2.0, "interior_derivative_root_bisection"

    selected_loss = _loss(offset, correction, outcome, alpha)
    if selected_loss > min(loss0, loss1) + 1e-12:
        raise AssertionError("The selected convex objective is worse than an interval endpoint")
    return {
        "alpha": float(alpha),
        "selection_rule": rule,
        "rows": int(len(offset)),
        "loss_alpha_0": loss0,
        "loss_alpha_1": loss1,
        "loss_selected": selected_loss,
        "derivative_alpha_0": grad0,
        "derivative_alpha_1": grad1,
        "derivative_selected": _gradient(offset, correction, outcome, alpha),
        "minimum_not_worse_than_both_endpoints": True,
    }


def _load_predictions() -> tuple[pd.DataFrame, dict]:
    source_path = ROOT / CONFIG["prediction_source"]["development_oof_path"]
    columns = [
        "market_start_utc", "condition_id", "fold_id", "official_outcome_up",
        "label_available_at_utc", "p_market", "p_M1", "causal_prediction_available",
        "prediction_origin", "prediction_source", "market_confirmed", "up_token_id", "down_token_id",
        "order_min_size_shares_current",
    ]
    saved = pd.read_parquet(source_path, columns=columns)
    saved["market_start_utc"] = pd.to_datetime(saved.market_start_utc, utc=True, errors="raise")
    saved["label_available_at_utc"] = pd.to_datetime(saved.label_available_at_utc, utc=True, errors="raise")
    if len(saved) != 8675 or saved.condition_id.duplicated().any():
        raise RuntimeError("The saved development OOF sample no longer matches its frozen row identity")
    if not saved.causal_prediction_available.fillna(False).all():
        raise RuntimeError("A saved M1 row is not marked as causally available")
    if saved.official_outcome_up.isna().any() or not saved.official_outcome_up.isin([0, 1]).all():
        raise RuntimeError("The saved development OOF sample has missing or invalid outcomes")
    if not saved[["p_market", "p_M1"]].apply(lambda column: column.between(0.0, 1.0).all()).all():
        raise RuntimeError("The saved market or M1 probability is outside [0, 1]")

    raw_frames = []
    checkpoint_paths = CONFIG["prediction_source"]["fold_correction_paths"]
    for fold_id, relative_path in checkpoint_paths.items():
        checkpoint_path = ROOT / relative_path
        record_path = checkpoint_path.with_suffix(".json")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("predictions_sha256") != _sha256(checkpoint_path):
            raise RuntimeError(f"M1 correction checkpoint failed its saved hash: {fold_id}")
        if record.get("fit_signature", {}).get("model") != "M1":
            raise RuntimeError(f"Correction checkpoint is not M1: {fold_id}")
        if record.get("fit_signature", {}).get("configuration", {}).get("config_id") != CONFIG["prediction_source"]["selected_m1_config_id"]:
            raise RuntimeError(f"Correction checkpoint is not the selected M1 configuration: {fold_id}")
        raw = pd.read_parquet(checkpoint_path)
        if raw["condition_id"].duplicated().any() or not raw["p_M1"].between(0.0, 1.0).all():
            raise RuntimeError(f"Correction checkpoint has duplicate IDs or invalid probabilities: {fold_id}")
        raw["market_start_utc"] = pd.to_datetime(raw.market_start_utc, utc=True, errors="raise")
        raw["fold_id"] = fold_id
        raw_frames.append(raw[["market_start_utc", "condition_id", "fold_id", "p_market", "raw_correction", "p_M1"]])
    raw_corrections = pd.concat(raw_frames, ignore_index=True)
    if raw_corrections.condition_id.duplicated().any() or len(raw_corrections) != len(saved):
        raise RuntimeError("Selected fold correction checkpoints do not form the complete development OOF sample")
    merged = saved.merge(
        raw_corrections,
        on=["market_start_utc", "condition_id", "fold_id"],
        how="outer",
        validate="one_to_one",
        suffixes=("", "_checkpoint"),
        indicator=True,
    )
    if not merged._merge.eq("both").all():
        raise RuntimeError("Saved M1 probabilities and fold raw corrections have different row coverage")
    for column in ("p_market", "p_M1"):
        if not np.array_equal(merged[column].to_numpy(), merged[f"{column}_checkpoint"].to_numpy()):
            raise RuntimeError(f"Saved M1 fold checkpoint and OOF file disagree on {column}")
    merged = merged.drop(columns=["_merge", "p_market_checkpoint", "p_M1_checkpoint"])
    merged = merged.sort_values(["market_start_utc", "condition_id"], kind="stable").reset_index(drop=True)
    epsilon = float(CONFIG["prediction_source"]["logit_epsilon"])
    if epsilon != float(offset_source.EPSILON):
        raise RuntimeError("The shrinkage config differs from the source experiment's logit clipping")
    offset = offset_source._logit(merged.p_market.to_numpy(dtype=np.float64))
    correction = merged.raw_correction.to_numpy(dtype=np.float64)
    reconstructed_market = _probabilities_from_alpha(merged.p_market.to_numpy(), correction, 0.0)
    reconstructed_m1 = offset_source._apply_offset_correction(merged.p_market.to_numpy(), correction)
    market_error = float(np.max(np.abs(reconstructed_market - merged.p_market.to_numpy())))
    m1_error = float(np.max(np.abs(reconstructed_m1 - merged.p_M1.to_numpy())))
    if not np.allclose(reconstructed_market, merged.p_market.to_numpy(), rtol=0.0, atol=2e-15):
        raise RuntimeError("alpha=0 does not reproduce the saved market probabilities")
    if not np.array_equal(reconstructed_m1, merged.p_M1.to_numpy()):
        raise RuntimeError("alpha=1 does not exactly reproduce the saved M1 probabilities")
    merged["logit_p_market"] = offset
    merged["p_alpha_0"] = reconstructed_market
    merged["p_alpha_1_reconstructed"] = reconstructed_m1
    identity = {
        "development_rows": int(len(merged)),
        "fold_counts": {str(key): int(value) for key, value in merged.groupby("fold_id").size().items()},
        "market_reconstruction_max_abs_error": market_error,
        "m1_reconstruction_max_abs_error": m1_error,
        "raw_correction_used": True,
        "raw_correction_precision_loss": False,
        "causal_prediction_origins": sorted(merged.prediction_origin.dropna().astype(str).unique().tolist()),
    }
    return merged, identity


def _row_quality(frame: pd.DataFrame, probability_column: str) -> dict:
    y = frame.official_outcome_up.to_numpy(dtype=np.float64)
    p = np.clip(frame[probability_column].to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return {
        "rows": int(len(frame)),
        "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p))),
        "brier": float(np.mean(np.square(p - y))),
        "mean_absolute_probability_correction_vs_market": float(np.mean(np.abs(p - frame.p_market.to_numpy(dtype=np.float64)))),
    }


def _paired_block_differences(frame: pd.DataFrame, left_column: str, right_column: str) -> dict:
    ordered = frame.sort_values("market_start_utc", kind="stable").reset_index(drop=True)
    y = ordered.official_outcome_up.to_numpy(dtype=np.float64)
    left = np.clip(ordered[left_column].to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
    right = np.clip(ordered[right_column].to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12)
    ll_diff = -(y * np.log(left) + (1.0 - y) * np.log1p(-left)) + y * np.log(right) + (1.0 - y) * np.log1p(-right)
    brier_diff = np.square(left - y) - np.square(right - y)
    day_values = ordered.market_start_utc.dt.floor("D")
    days = pd.date_range(day_values.min(), day_values.max(), freq="D")
    positions = day_values.map({day: index for index, day in enumerate(days)}).to_numpy(dtype=np.int64)
    counts = np.bincount(positions, minlength=len(days)).astype(np.float64)
    daily_ll = np.bincount(positions, weights=ll_diff, minlength=len(days))
    daily_brier = np.bincount(positions, weights=brier_diff, minlength=len(days))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    block_length = min(3, len(days))
    days_per_draw = len(days)
    ll_draws, brier_draws = [], []
    attempts = 0
    while len(ll_draws) < BOOTSTRAP_REPLICATIONS and attempts < 20000:
        attempts += 1
        block_starts = rng.integers(0, len(days), size=math.ceil(days_per_draw / block_length))
        sampled_days = ((block_starts[:, None] + np.arange(block_length)) % len(days)).reshape(-1)[:days_per_draw]
        denominator = float(counts[sampled_days].sum())
        if denominator <= 0.0:
            continue
        ll_draws.append(float(daily_ll[sampled_days].sum() / denominator))
        brier_draws.append(float(daily_brier[sampled_days].sum() / denominator))
    if len(ll_draws) != BOOTSTRAP_REPLICATIONS:
        raise RuntimeError("Could not obtain the frozen number of valid block-bootstrap replications")
    return {
        "valid_bootstrap_replicates": len(ll_draws),
        "bootstrap_attempts": attempts,
        "seed": BOOTSTRAP_SEED,
        "block_days_utc": block_length,
        "log_loss_difference": float(np.mean(ll_diff)),
        "log_loss_difference_95pct": [float(value) for value in np.quantile(ll_draws, [0.025, 0.975])],
        "brier_difference": float(np.mean(brier_diff)),
        "brier_difference_95pct": [float(value) for value in np.quantile(brier_draws, [0.025, 0.975])],
    }


def _prepare_outputs_and_freeze(frame: pd.DataFrame, alpha_fit: dict, input_identity: dict) -> tuple[pd.DataFrame, dict]:
    alpha = float(alpha_fit["alpha"])
    predictions = frame.copy()
    predictions["p_alpha_selected"] = _probabilities_from_alpha(
        predictions.p_market.to_numpy(), predictions.raw_correction.to_numpy(), alpha,
    )
    predictions = predictions.loc[
        predictions.market_start_utc.ge(MAY_START) & predictions.market_start_utc.lt(JULY_START),
        [
            "market_start_utc", "condition_id", "fold_id", "label_available_at_utc", "official_outcome_up",
            "p_market", "raw_correction", "p_M1", "p_alpha_0", "p_alpha_1_reconstructed",
            "p_alpha_selected", "causal_prediction_available", "prediction_origin", "prediction_source",
            "market_confirmed", "up_token_id", "down_token_id", "order_min_size_shares_current",
        ],
    ].reset_index(drop=True)
    if predictions[["up_token_id", "down_token_id"]].isna().any().any():
        raise RuntimeError("Development prediction rows are missing native UP/DOWN token IDs needed by the execution engine")
    _write_parquet(PREDICTIONS_PATH, predictions)

    code_paths = {
        "runner": Path(__file__),
        "config": CONFIG_PATH,
        "source_offset_helpers": Path(offset_source.__file__),
        "snapshot_map_helper": Path(research.__file__),
        "T59_execution_engine": Path(replay.__file__),
        "source_experiment_config": SOURCE_CONFIG_PATH,
        "source_experiment_report": SOURCE_REPORT_PATH,
    }
    input_paths = {
        "saved_development_oof_predictions": ROOT / CONFIG["prediction_source"]["development_oof_path"],
        "T60_snapshots_both_sides": T60_PATH,
        "T59_snapshots_both_sides": T59_PATH,
    }
    input_paths.update({f"raw_M1_correction_{fold}": ROOT / path for fold, path in CONFIG["prediction_source"]["fold_correction_paths"].items()})
    input_hashes = {name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for name, path in input_paths.items()}
    code_hashes = {name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for name, path in code_paths.items()}
    manifest = {
        "schema_version": 1,
        "experiment_id": CONFIG["experiment_id"],
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "frozen_before_valid_june_evaluation": True,
        "input_hashes": input_hashes,
        "code_hashes": code_hashes,
        "prediction_output": {
            "path": str(PREDICTIONS_PATH.relative_to(ROOT)),
            "sha256": _sha256(PREDICTIONS_PATH),
            "rows": int(len(predictions)),
            "june_rows": int(predictions.market_start_utc.ge(JUNE_START).sum()),
            "sample_sha256": _frame_sha256(predictions, ("market_start_utc", "condition_id", "official_outcome_up", "p_market", "raw_correction", "p_M1", "p_alpha_selected")),
        },
        "sample": {
            "development_rows": input_identity["development_rows"],
            "may_market_rows": int(frame.market_start_utc.ge(MAY_START).mul(frame.market_start_utc.lt(JUNE_START)).sum()),
            "may_calibration_rows": alpha_fit["rows"],
            "june_evaluation_rows": int(predictions.market_start_utc.ge(JUNE_START).sum()),
            "calibration_start_inclusive_utc": MAY_START.isoformat(),
            "calibration_end_exclusive_utc": JUNE_START.isoformat(),
            "calibration_label_available_at_cutoff_utc": CONFIG["periods"]["calibration_label_available_cutoff_utc"],
            "evaluation_start_inclusive_utc": JUNE_START.isoformat(),
            "evaluation_end_exclusive_utc": JULY_START.isoformat(),
            "latest_calibration_label_available_at_utc": frame.loc[
                frame.market_start_utc.ge(MAY_START) & frame.market_start_utc.lt(JUNE_START)
                & frame.label_available_at_utc.le(pd.Timestamp(CONFIG["periods"]["calibration_label_available_cutoff_utc"])),
                "label_available_at_utc",
            ].max().isoformat(),
            "excluded_may_rows_label_unavailable_by_cutoff": int(
                (frame.market_start_utc.ge(MAY_START) & frame.market_start_utc.lt(JUNE_START)
                 & frame.label_available_at_utc.gt(pd.Timestamp(CONFIG["periods"]["calibration_label_available_cutoff_utc"]))).sum()
            ),
            "label_available_at_definition": CONFIG["periods"]["label_available_at_definition"],
            "fold_counts": input_identity["fold_counts"],
        },
        "calibration": {
            **alpha_fit,
            "formula": CONFIG["calibration"]["formula"],
            "range": [0.0, 1.0],
            "objective": "mean(logaddexp(0, offset + alpha*f) - y*(offset + alpha*f)) on all eligible May quality rows",
            "fit_rows_are_unweighted_and_not_EV_or_trade_filtered": True,
            "selected_m1_config": CONFIG["prediction_source"]["selected_m1_config_id"],
            "previous_model_selection_included_may_and_june": True,
            "june_is_not_an_independent_test_of_the_complete_model_selection_and_calibration_procedure": True,
        },
        "reconstruction": input_identity,
        "quality_rules": {
            "metrics": ["binary log-loss", "Brier"],
            "comparisons": ["alpha-selected minus original M1", "alpha-selected minus market"],
            "mean_absolute_probability_correction_vs_market": True,
            "bootstrap": CONFIG["quality_uncertainty"],
        },
        "economic_rules": {
            **CONFIG["economics"],
            "settlement_cutoff_exclusive_utc": CONFIG["periods"]["settlement_cutoff_exclusive_utc"],
            "side_selection": "recompute both T-60 sides using each variant; existing engine selects the greatest positive net EV, ties prefer UP; execute only when selected EV is at least 1% of requested stake",
            "decomposition": "R59 - E60 = (E59 - E60) + (y - p) * net_shares_T59",
        },
        "holdout_used": False,
        "training_features_oof_or_replay_regenerated": False,
    }
    prior_manifest_path = PRIOR_INVALID_DIR / "frozen_manifest.json"
    if prior_manifest_path.is_file():
        manifest["superseded_invalid_execution"] = {
            "manifest_path": str(prior_manifest_path.relative_to(ROOT)),
            "manifest_sha256": _sha256(prior_manifest_path),
            "reason": "The initial economics pass omitted native UP/DOWN token IDs from its market rows, so the engine marked every side unpriceable. Its June quality metrics were unchanged; the economic results were discarded and rerun with the corrected inputs.",
            "alpha_changed": False,
        }
    prior_valid_manifest_path = PRIOR_PREFORMAT_DIR / "frozen_manifest.json"
    if prior_valid_manifest_path.is_file():
        manifest["prior_valid_reporting_pass"] = {
            "manifest_path": prior_valid_manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": _sha256(prior_valid_manifest_path),
            "reason": "Reformatted zero-trade EV/decomposition fields as not applicable; all frozen predictions, alpha, quality results, decision counts, and economic fills are unchanged.",
        }
    identity = dict(manifest)
    identity.pop("frozen_at_utc")
    manifest["freeze_sha256"] = _canonical_sha256(identity)
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        existing_identity = dict(existing)
        existing_identity.pop("frozen_at_utc", None)
        proposed_identity = dict(manifest)
        proposed_identity.pop("frozen_at_utc")
        if existing_identity != proposed_identity:
            raise RuntimeError("A prior manifest exists with a different experiment identity or protocol")
        manifest = existing
    else:
        _write_json(MANIFEST_PATH, manifest)
    return predictions, manifest


def _evaluate_economics(predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    june = predictions.loc[
        predictions.market_start_utc.ge(JUNE_START) & predictions.market_start_utc.lt(JULY_START)
    ].copy()
    if june.empty or not june.label_available_at_utc.le(JULY_START).all():
        raise RuntimeError("The June evaluation rows are empty or include labels after the frozen July cutoff")
    wanted = set(june.condition_id.astype(str).str.lower())
    t60 = pd.read_parquet(T60_PATH, columns=list(SNAPSHOT_COLUMNS))
    t59 = pd.read_parquet(T59_PATH, columns=list(SNAPSHOT_COLUMNS))
    t60 = t60.loc[t60.condition_id.astype(str).str.lower().isin(wanted)].copy()
    t59 = t59.loc[t59.condition_id.astype(str).str.lower().isin(wanted)].copy()
    t60_by_id, t59_by_id = research._snapshot_map(t60), research._snapshot_map(t59)
    markets = research._snapshot_map(june)
    if len(markets) != len(june) or any(market.get("up_token_id") is None or market.get("down_token_id") is None for market in markets.values()):
        raise RuntimeError("The June market rows are missing IDs or native token IDs required for execution pricing")
    variants = {
        "market_alpha_0": "p_market",
        "original_M1_alpha_1": "p_M1",
        "calibrated_M1": "p_alpha_selected",
    }
    summaries: dict[str, dict] = {}
    ledgers: list[dict] = []
    for variant, probability_column in variants.items():
        counters = {
            "t60_priceable_markets": 0,
            "positive_ev_markets": 0,
            "positive_ev_sides": 0,
            "ev_at_least_1pct_decisions": 0,
            "missing_t60_snapshots": 0,
            "qualified_missing_t59_snapshots": 0,
        }
        rows: list[dict] = []
        for item in june.itertuples(index=False):
            cid = str(item.condition_id).lower()
            market = markets[cid]
            t60_snapshot, t59_snapshot = t60_by_id.get(cid), t59_by_id.get(cid)
            row = {
                "condition_id": cid,
                "market_start_utc": item.market_start_utc,
                "model": variant,
                "probability_source": probability_column,
                "p_up_T60": float(getattr(item, probability_column)),
                "official_outcome_up": int(item.official_outcome_up),
                "label_available_at_utc": item.label_available_at_utc,
                "settled_by_cutoff": bool(item.label_available_at_utc <= JULY_START),
                "t60_snapshot_status": "available" if t60_snapshot is not None else "missing_snapshot",
                "t59_snapshot_status": "available" if t59_snapshot is not None else "missing_snapshot",
                "chosen_side": None,
                "chosen_probability": None,
                "chosen_ev_usd_T60": None,
                "chosen_ev_fraction_T60": None,
                "ev_at_least_1pct": False,
                "t59_execution_status": "not_attempted",
            }
            if t60_snapshot is None:
                counters["missing_t60_snapshots"] += 1
                row["t60_valuation_status"] = "missing_snapshot"
                rows.append(row)
                continue

            decision = _evaluate_decision(market, t60_snapshot, float(getattr(item, probability_column)))
            counters["t60_priceable_markets"] += int(decision["priceable_side_count"] > 0)
            counters["positive_ev_markets"] += int(decision["positive_ev_side_count"] > 0)
            counters["positive_ev_sides"] += int(decision["positive_ev_side_count"])
            for side in ("up", "down"):
                row[f"t60_{side}_priceable"] = bool(decision["sides"][side]["priceable"])
                row[f"t60_{side}_ev_usd"] = decision["sides"][side].get("ev_usd")
            side = decision["chosen_side"]
            row["t60_valuation_status"] = "selected_positive_ev_side" if side is not None else "no_positive_ev_side"
            if side is None:
                rows.append(row)
                continue
            selected = decision["sides"][side]
            ev_usd = float(selected["ev_usd"])
            ev_fraction = ev_usd / STAKE_USD
            qualified = ev_fraction + 1e-12 >= MIN_EV_FRACTION
            row.update({
                "chosen_side": side,
                "chosen_probability": float(selected["probability"]),
                "chosen_ev_usd_T60": ev_usd,
                "chosen_ev_fraction_T60": ev_fraction,
                "ev_at_least_1pct": qualified,
            })
            counters["ev_at_least_1pct_decisions"] += int(qualified)
            if not qualified:
                row["t59_execution_status"] = "below_1pct_ev"
                rows.append(row)
                continue
            if t59_snapshot is None:
                counters["qualified_missing_t59_snapshots"] += 1
                row["t59_execution_status"] = "missing_snapshot"
                rows.append(row)
                continue

            execution = replay._price_side(t59_snapshot, market, side, STAKE_USD)
            fill = execution.get("fill") if execution["priceable"] else None
            row["t59_execution_status"] = "filled" if fill is not None else "unpriceable"
            row["t59_execution_reason"] = execution.get("reason")
            if fill is not None:
                y_side = int(item.official_outcome_up) if side == "up" else 1 - int(item.official_outcome_up)
                p_side = float(selected["probability"])
                e60 = float(selected["fill"]["shares"]) * p_side - float(selected["fill"]["cash_debit_usd"])
                e59 = float(fill["shares"]) * p_side - float(fill["cash_debit_usd"])
                settled = bool(item.label_available_at_utc <= JULY_START)
                r59 = float(fill["shares"] if y_side else 0.0) - float(fill["cash_debit_usd"]) if settled else None
                row.update({
                    "t59_gross_usd": float(fill["gross_usd"]),
                    "t59_cash_debit_usd": float(fill["cash_debit_usd"]),
                    "t59_net_shares": float(fill["shares"]),
                    "t59_fee_usd": float(fill.get("fee_usd") or 0.0),
                    "t59_fee_cash_usd": float(fill.get("fee_cash_usd") or 0.0),
                    "t59_fee_shares": float(fill.get("fee_shares") or 0.0),
                    "t59_depth_known": bool(fill.get("depth_known")),
                    "t59_execution_source": fill.get("execution_source"),
                    "t59_execution_assumption": fill.get("execution_assumption"),
                    "t59_fill_status": fill.get("fill_status"),
                    "t59_E_expected_USD": e59,
                    "T60_E_expected_USD": e60,
                    "t59_realized_USD": r59,
                    "execution_component_USD": e59 - e60 if settled else None,
                    "probability_residual_USD": (y_side - p_side) * float(fill["shares"]) if settled else None,
                    "decomposition_error_USD": r59 - e60 - ((e59 - e60) + (y_side - p_side) * float(fill["shares"])) if settled else None,
                })
            rows.append(row)

        ledger = pd.DataFrame(rows)
        filled = ledger.loc[ledger.t59_execution_status.eq("filled")].copy()
        settled = filled.loc[filled.settled_by_cutoff].copy()
        turnover = float(settled.t59_gross_usd.sum()) if len(settled) else 0.0
        pnl = float(settled.t59_realized_USD.sum()) if len(settled) else 0.0
        expected_t60 = float(settled.T60_E_expected_USD.sum()) if len(settled) else None
        expected_t59 = float(settled.t59_E_expected_USD.sum()) if len(settled) else None
        execution_component = float(settled.execution_component_USD.sum()) if len(settled) else None
        probability_residual = float(settled.probability_residual_USD.sum()) if len(settled) else None
        realized_result = pnl if len(settled) else None
        max_decomposition_error = float(settled.decomposition_error_USD.abs().max()) if len(settled) else None
        if max_decomposition_error is not None and max_decomposition_error > 1e-7:
            raise AssertionError(f"EV-to-result decomposition failed for {variant}: {max_decomposition_error}")
        depth = {}
        if len(settled):
            for known, group in settled.groupby("t59_depth_known", dropna=False):
                depth["full_book" if bool(known) else "top_of_book_proxy"] = {
                    "settled_fills": int(len(group)),
                    "turnover_usd": float(group.t59_gross_usd.sum()),
                    "fees_usd": float(group.t59_fee_usd.sum()),
                    "net_pnl_usd": float(group.t59_realized_USD.sum()),
                }
        summaries[variant] = {
            **counters,
            "june_markets": int(len(ledger)),
            "t59_fills": int(len(filled)),
            "settled_fills_by_cutoff": int(len(settled)),
            "unsettled_fills_at_cutoff": int(len(filled) - len(settled)),
            "trade_status": "no_transactions" if len(settled) == 0 else "transactions",
            "turnover_usd": turnover,
            "fees_usd": float(settled.t59_fee_usd.sum()) if len(settled) else 0.0,
            "net_pnl_usd": pnl,
            "net_pnl_over_turnover": pnl / turnover if turnover else None,
            "sum_predicted_ev_T60_on_settled_executions_usd": expected_t60,
            "sum_predicted_ev_T59_on_settled_executions_usd": expected_t59,
            "sum_realized_result_on_same_executions_usd": realized_result,
            "realized_minus_predicted_ev_T60_usd": pnl - expected_t60 if expected_t60 is not None else None,
            "execution_component_E59_minus_E60_usd": execution_component,
            "probability_residual_usd": probability_residual,
            "max_decomposition_error_usd": max_decomposition_error,
            "settled_depth_breakdown": depth,
            "ledger_rows": int(len(ledger)),
        }
        ledgers.append(ledger)

    combined = pd.concat(ledgers, ignore_index=True)
    calibrated_choices = combined.loc[combined.model.eq("calibrated_M1")].set_index("condition_id").chosen_side
    original_choices = combined.loc[combined.model.eq("original_M1_alpha_1")].set_index("condition_id").chosen_side
    changed = calibrated_choices.fillna("none").ne(original_choices.fillna("none"))
    summaries["calibrated_M1"]["side_choices_changed_vs_original_M1"] = int(changed.sum())
    summaries["calibrated_M1"]["side_choices_changed_pct_of_june_rows"] = float(changed.mean())
    return {
        "settlement_cutoff_exclusive_utc": JULY_START.isoformat(),
        "stake_usd": STAKE_USD,
        "minimum_net_ev_fraction": MIN_EV_FRACTION,
        "side_reselected_for_each_probability_variant": True,
        "snapshot_sources": {"T60": str(T60_PATH.relative_to(ROOT)), "T59": str(T59_PATH.relative_to(ROOT))},
        "variants": summaries,
    }, combined


def _evaluate_decision(market: dict, t60_snapshot: dict, p_up: float) -> dict:
    return replay._evaluate_market_sides(
        {**market, "p_candidate_platt": float(p_up)}, t60_snapshot, STAKE_USD,
    )


def _format(value, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _currency(value, digits: int = 4) -> str:
    return "brak transakcji" if value is None or pd.isna(value) else f"${float(value):.{digits}f}"


def _report(manifest: dict, alpha_fit: dict, quality: dict, differences: dict, economics: dict, runtime: dict) -> str:
    alpha = float(alpha_fit["alpha"])
    lines = [
        "# M1 korekta skurczona jednym parametrem — 2026-10-18",
        "",
        "> M1 był wybrany wcześniej z udziałem maja i czerwca. Chronologiczne dopasowanie α w maju nie czyni czerwca niezależnym testem całej procedury.",
        "",
        "## Protokół i zamrożenie",
        "",
        f"Końcowy manifest poprawionego przebiegu zapisano przed ponowną oceną ekonomiczną czerwca: `{manifest['frozen_at_utc']}` (SHA-256 `{manifest['freeze_sha256']}`). Pierwszy manifest z tym samym α został zamrożony przed pierwszym czerwcowym scoringiem.",
        f"Użyto zapisanych foldowych, przyczynowych korekt surowego logitu M1 z konfiguracji `{CONFIG['prediction_source']['selected_m1_config_id']}`. Nie trenowano modelu, nie odtwarzano holdoutu, nie przeliczano cech ani OOF.",
        f"Odtworzenie `alpha=0` dało rynek z maksymalnym błędem {_format(manifest['reconstruction']['market_reconstruction_max_abs_error'], 15)}; `alpha=1` odtworzyło zapisane p_M1 dokładnie (maksymalny błąd {_format(manifest['reconstruction']['m1_reconstruction_max_abs_error'], 15)}). Użyto surowych korekt, bez utraty precyzji przy ich odtwarzaniu.",
        "",
        "Majowe dopasowanie zawiera wszystkie outcome-quality wiersze z market start w maju i etykietą dostępną do 2026-06-01T00:00:00Z; nie filtruje po EV, transakcji ani wycenialności.",
        f"Wiersze maja: {manifest['sample']['may_market_rows']:,}; kwalifikujące się do dopasowania: {manifest['sample']['may_calibration_rows']:,}; wyłączone przez dostępność etykiety: {manifest['sample']['excluded_may_rows_label_unavailable_by_cutoff']:,}. Reguła: `{manifest['sample']['label_available_at_definition']}`.",
        "",
        "## Dopasowanie α w maju (wynik dopasowania)",
        "",
        f"Wybrane α = **{alpha:.12f}** (`{alpha_fit['selection_rule']}`), w ograniczeniu [0, 1]. Minimalizowano nieważony log-loss na {alpha_fit['rows']:,} wierszach.",
        "",
        "| α | Log-loss celu | Pochodna celu |",
        "|---:|---:|---:|",
        f"| 0 (rynek) | {_format(alpha_fit['loss_alpha_0'], 9)} | {_format(alpha_fit['derivative_alpha_0'], 12)} |",
        f"| {alpha:.12f} (wybrane) | {_format(alpha_fit['loss_selected'], 9)} | {_format(alpha_fit['derivative_selected'], 12)} |",
        f"| 1 (M1) | {_format(alpha_fit['loss_alpha_1'], 9)} | {_format(alpha_fit['derivative_alpha_1'], 12)} |",
        "",
        "Maj jest próbką dopasowania, a nie późniejszą walidacją. Bazową konfigurację M1 wybrano wcześniej z użyciem maja i czerwca; czerwiec ocenia zamrożone α warunkowo na tym wybranym M1.",
        "",
        "## Jakość prawdopodobieństw w czerwcu",
        "",
        "Wszystkie trzy warianty liczone są na tych samych czerwcowych OOF wierszach. Ujemna różnica kalibracja minus porównanie oznacza poprawę.",
        "",
        "| Wariant | n | Log-loss | Brier | Śr. bezwzględna korekta p względem rynku |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("market", "original_M1", "calibrated_M1"):
        item = quality[name]
        lines.append(f"| {name} | {item['rows']:,} | {_format(item['log_loss'], 9)} | {_format(item['brier'], 9)} | {_format(item['mean_absolute_probability_correction_vs_market'], 9)} |")
    lines += [
        "",
        "| Sparowana różnica czerwiec | Δ log-loss | 95% CI, blok 3 dni UTC | Δ Brier | 95% CI, blok 3 dni UTC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in (("calibrated_minus_original_M1", "kalibrowany M1 − oryginalny M1"), ("calibrated_minus_market", "kalibrowany M1 − rynek")):
        item = differences[name]
        lines.append(
            f"| {label} | {_format(item['log_loss_difference'], 9)} | [{_format(item['log_loss_difference_95pct'][0], 9)}, {_format(item['log_loss_difference_95pct'][1], 9)}] | "
            f"{_format(item['brier_difference'], 9)} | [{_format(item['brier_difference_95pct'][0], 9)}, {_format(item['brier_difference_95pct'][1], 9)}] |"
        )
    lines += [
        "",
        f"Bootstrap: {BOOTSTRAP_REPLICATIONS:,} poprawnych replikacji, seed {BOOTSTRAP_SEED}; przedziały warunkowe na wybranym M1 i zamrożonym α. Nie obejmują pełnej niepewności wcześniejszego wyboru modelu.",
        "",
        "## Ekonomia czerwcowa, stałe $5",
        "",
        "Decyzja T−60, wykonanie T−59, obie strony wyceniane ponownie dla każdego wariantu, próg EV netto 1% stawki, istniejące opłaty, głębokość, fills i cutoff rozliczeń 2026-07-01T00:00:00Z. Predykcyjne EV i wynik rzeczywisty w dekompozycji odnoszą się do tych samych rozliczonych wykonań EV ≥ 1%.",
        "",
        "| Wariant | Wycenialne rynki | Dodatnie EV (rynki / strony) | Decyzje EV ≥1% | Wypełnienia T−59 | Rozliczone | Obrót | Opłaty | PnL netto | PnL/obrót | Σ EV T−60 na tych wykonaniach |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("market_alpha_0", "original_M1_alpha_1", "calibrated_M1"):
        item = economics["variants"][name]
        pnl_text = "brak transakcji" if item["trade_status"] == "no_transactions" else _currency(item["net_pnl_usd"])
        pnl_turnover = f"{item['net_pnl_over_turnover'] * 100:.4f}%" if item["net_pnl_over_turnover"] is not None else "—"
        lines.append(
            f"| {name} | {item['t60_priceable_markets']:,} | {item['positive_ev_markets']:,} / {item['positive_ev_sides']:,} | {item['ev_at_least_1pct_decisions']:,} | "
            f"{item['t59_fills']:,} | {item['settled_fills_by_cutoff']:,} | ${item['turnover_usd']:.2f} | ${item['fees_usd']:.4f} | "
            f"{pnl_text} | {pnl_turnover} | {_currency(item['sum_predicted_ev_T60_on_settled_executions_usd'])} |"
        )
    lines += [
        "",
        "EV ≥1% — agregaty na wspólnym dla każdej polityki zbiorze jej rozliczonych wykonań:",
        "",
        "| Wariant | Σ E60 | Σ E59 | Σ R59 | R59 − E60 | E59 − E60 | (y−p) × net shares | Maks. błąd tożsamości |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("market_alpha_0", "original_M1_alpha_1", "calibrated_M1"):
        item = economics["variants"][name]
        if item["trade_status"] == "no_transactions":
            lines.append(f"| {name} | brak transakcji | brak transakcji | brak transakcji | brak transakcji | brak transakcji | brak transakcji | — |")
            continue
        lines.append(
            f"| {name} | {_currency(item['sum_predicted_ev_T60_on_settled_executions_usd'])} | {_currency(item['sum_predicted_ev_T59_on_settled_executions_usd'])} | "
            f"{_currency(item['sum_realized_result_on_same_executions_usd'])} | {_currency(item['realized_minus_predicted_ev_T60_usd'])} | "
            f"{_currency(item['execution_component_E59_minus_E60_usd'])} | {_currency(item['probability_residual_usd'])} | {_format(item['max_decomposition_error_usd'], 12)} |"
        )
    original_gap = economics["variants"]["original_M1_alpha_1"]["realized_minus_predicted_ev_T60_usd"]
    lines += [
        "",
        f"Dla α=1 luka między sumą przewidywanego EV T−60 a rzeczywistym wynikiem wyniosła ${original_gap:.4f} w EV≥1%. Dla skalibrowanego M1 nie było rozliczonych wykonań w tej grupie, więc nie ma podstaw do wniosku, że luka ekonomiczna się zmniejszyła.",
    ]
    cal = economics["variants"]["calibrated_M1"]
    prior_note = manifest.get("superseded_invalid_execution")
    if prior_note:
        lines += [
            "",
            "## Uwaga o pierwszym wykonaniu",
            "",
            f"Pierwszy przebieg ekonomiczny nie kwalifikował żadnej strony, ponieważ dziennik predykcji przekazany do silnika nie zawierał token IDs UP/DOWN. Jego manifest, wyniki i artefakty zachowano w `{PRIOR_INVALID_DIR.relative_to(ROOT).as_posix()}`. Jego czerwcowe metryki jakości były poprawne i odpowiadają tym odtworzonym tutaj; wyłącznie ekonomię oznaczono jako nieważną. Poprawiony przebieg użył tego samego α i prawdopodobieństw, a jego manifest z hashami poprawionego kodu zapisano przed ponowną oceną ekonomiczną.",
            "",
        ]
    reporting_note = manifest.get("prior_valid_reporting_pass")
    if reporting_note:
        lines += [
            "",
            "Wartości z pustych grup EV są prezentowane jako „brak transakcji”, a nie jako zerowa rozbieżność EV–wynik. Wcześniejszy poprawny przebieg obliczeniowy, z tą różnicą prezentacji, zachowano w `data/analysis/polymarket/BTC/btc_m1_correction_shrinkage_20261018/prior_valid_preformat`.",
        ]
    lines += [
        "",
        f"Kalibracja zmieniła wybrany kierunek względem M1 w {cal['side_choices_changed_vs_original_M1']:,} z {cal['june_markets']:,} czerwcowych rynków. Źródłem nowej polityki są zapisane snapshoty T−60 i T−59 obu stron; nie użyto wcześniejszej tabeli diagnostycznej zawierającej tylko poprzednio wybraną stronę.",
        "",
        "Depth, opłaty, wynik i obrót w podziale na pełny order book i proxy T−59:",
        "",
        "| Wariant | Głębokość | Rozliczone fills | Obrót | Opłaty | PnL netto |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in ("market_alpha_0", "original_M1_alpha_1", "calibrated_M1"):
        item = economics["variants"][name]
        if not item["settled_depth_breakdown"]:
            lines.append(f"| {name} | brak transakcji | 0 | $0.00 | $0.0000 | $0.0000 |")
        else:
            for depth_name, values in item["settled_depth_breakdown"].items():
                lines.append(f"| {name} | {depth_name} | {values['settled_fills']:,} | ${values['turnover_usd']:.2f} | ${values['fees_usd']:.4f} | ${values['net_pnl_usd']:.4f} |")
    lines += [
        "",
        "## Artefakty i wykonanie",
        "",
        f"- Predykcje wyłącznie z development: `{PREDICTIONS_PATH.relative_to(ROOT).as_posix()}` (SHA-256 `{manifest['prediction_output']['sha256']}`).",
        f"- Manifest przed raportowanym przebiegiem czerwcowym: `{MANIFEST_PATH.relative_to(ROOT).as_posix()}`.",
        f"- Pełny dziennik decyzji i filli czerwcowych: `{ECONOMICS_PATH.relative_to(ROOT).as_posix()}`.",
        f"- Czas całego runnera: {runtime['elapsed_seconds']:.3f} s; szczytowy working set procesu: {runtime['peak_working_set_bytes'] / 1024**3:.3f} GiB.",
        f"- Python {runtime['python_version']}, pandas {runtime['pandas_version']}, NumPy {runtime['numpy_version']}.",
        "",
        "Reprodukcja bez argumentów CLI: `python run_btc_m1_correction_shrinkage_20261018.py`.",
        "",
    ]
    return "\n".join(lines)


def main() -> dict:
    started = time.perf_counter()
    started_utc = pd.Timestamp.now(tz="UTC").isoformat()
    frame, input_identity = _load_predictions()
    may = frame.loc[
        frame.market_start_utc.ge(MAY_START)
        & frame.market_start_utc.lt(JUNE_START)
        & frame.label_available_at_utc.le(pd.Timestamp(CONFIG["periods"]["calibration_label_available_cutoff_utc"])),
    ].copy()
    if may.empty:
        raise RuntimeError("No May development OOF rows have labels available by the frozen calibration cutoff")
    fit = _fit_alpha(
        may.logit_p_market.to_numpy(dtype=np.float64),
        may.raw_correction.to_numpy(dtype=np.float64),
        may.official_outcome_up.to_numpy(dtype=np.float64),
    )
    predictions, manifest = _prepare_outputs_and_freeze(frame, fit, input_identity)
    print(f"[m1-shrinkage] manifest frozen: {manifest['freeze_sha256']}; alpha={fit['alpha']:.12f}", flush=True)

    june = predictions.loc[predictions.market_start_utc.ge(JUNE_START) & predictions.market_start_utc.lt(JULY_START)].copy()
    if not june.label_available_at_utc.le(JULY_START).all():
        raise RuntimeError("A June quality outcome is unavailable by the fixed July cutoff")
    quality = {
        "market": _row_quality(june, "p_market"),
        "original_M1": _row_quality(june, "p_M1"),
        "calibrated_M1": _row_quality(june, "p_alpha_selected"),
    }
    differences = {
        "calibrated_minus_original_M1": _paired_block_differences(june, "p_alpha_selected", "p_M1"),
        "calibrated_minus_market": _paired_block_differences(june, "p_alpha_selected", "p_market"),
    }
    economics, ledgers = _evaluate_economics(predictions)
    _write_parquet(ECONOMICS_PATH, ledgers)
    peak_bytes = offset_source._peak_working_set_bytes()
    if peak_bytes is None:
        raise RuntimeError("Could not measure peak working-set memory")
    runtime = {
        "started_at_utc": started_utc,
        "finished_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_working_set_bytes": int(peak_bytes),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    results = {
        "status": "completed",
        "experiment_id": CONFIG["experiment_id"],
        "frozen_manifest_sha256": manifest["freeze_sha256"],
        "alpha_fit": fit,
        "june_quality": quality,
        "june_paired_differences": differences,
        "june_economics": economics,
        "runtime": runtime,
        "artifacts": {
            "development_predictions": str(PREDICTIONS_PATH.relative_to(ROOT)),
            "june_economic_decisions": str(ECONOMICS_PATH.relative_to(ROOT)),
            "report": str(REPORT_PATH.relative_to(ROOT)),
        },
    }
    _write_json(RUN_PATH, results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(manifest, fit, quality, differences, economics, runtime), encoding="utf-8")
    print(json.dumps({"alpha": fit["alpha"], "quality": quality, "economics": economics["variants"], "runtime": runtime}, indent=2, default=str), flush=True)
    return results


if __name__ == "__main__":
    main()
