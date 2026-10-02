"""Controlled BTC five-minute Binance-target vs Polymarket-target experiment.

This is a retrospective research run. It writes only to a fingerprinted analysis
directory and the experiment report; it never touches the live model or config.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from utils.polymarket_market_value import (
    MODEL_C as MARKET_MODEL_C,
    _execution_quotes,
    market_features,
    paired_block_bootstrap,
    simulate_fixed_policy,
)
from utils.polymarket_policy import calibrate, calibration_metrics, fit_calibrator


ROOT = Path(__file__).resolve().parent
COMMON_RUN = ROOT / "data/analysis/polymarket/BTC/runs/8d9688a07f0f7719"
SOURCE_RUN = ROOT / "data/datasets/polymarket/BTC/runs/efa1bf384fc6a2dc"
MODEL_READY_PATH = ROOT / "data/datasets/modeling/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m_model_ready.parquet"
OOF_PATH = ROOT / "data/datasets/modeling/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m_oof_predictions.parquet"
MAIN_META_PATH = ROOT / "data/models/BTC/20261002_041540/lgbm_meta_20261002_041540.json"
MARKET_VALUE_MANIFEST = COMMON_RUN / "market_value.json"
EXPECTED_COMMON_ROWS = 15_677
EXPECTED_OOF_ROWS = 9_407
OUTER_FRACTIONS = (0.4, 0.6, 0.8, 1.0)
INNER_TRAIN_FRACTION = 0.60
BOOTSTRAP_DAYS = 3
BOOTSTRAP_REPLICATIONS = 2_000
BOOTSTRAP_SEED = 20261003
MODEL_SEED = 37
MODEL_JOBS = 8
TREE_COUNTS = (100, 300)
MODEL_CANDIDATES = (
    {"num_leaves": 7, "min_child_samples": 100, "reg_lambda": 10.0},
    {"num_leaves": 15, "min_child_samples": 100, "reg_lambda": 10.0},
    {"num_leaves": 7, "min_child_samples": 250, "reg_lambda": 25.0},
    {"num_leaves": 15, "min_child_samples": 250, "reg_lambda": 25.0},
)
FIXED_STAKE_USDC = 5.0
INITIAL_PORTFOLIO_USDC = 100.0
SNAPSHOT_LATENCIES = (0, 1, 2)
EXECUTION_DELAYS = (0, 1, 2)
PRACTICAL_LL_HARM = 0.001

LIVE_REPORT_PATH = ROOT / "docs/polymarket_btc_experiment.md"
INPUT_PATHS = (
    ROOT / "run_btc_target_comparison.py",
    MODEL_READY_PATH,
    OOF_PATH,
    MAIN_META_PATH,
    COMMON_RUN / "common_latency_0s.parquet",
    COMMON_RUN / "common_latency_1s.parquet",
    COMMON_RUN / "common_latency_2s.parquet",
    MARKET_VALUE_MANIFEST,
    SOURCE_RUN / "markets.parquet",
    SOURCE_RUN / "quotes.parquet",
    COMMON_RUN / "inputs.json",
    ROOT / "utils/polymarket_market_value.py",
    ROOT / "utils/polymarket_history.py",
    ROOT / "utils/polymarket_policy.py",
    ROOT / "utils/polymarket.py",
    ROOT / "utils/data.py",
)


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
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
                    encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_ids(values) -> str:
    payload = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc(values):
    return pd.to_datetime(values, utc=True, errors="coerce")


def _read_exact_rows(path: Path, timestamps, columns):
    """Read exact timestamp keys from a sorted Parquet source; never nearest-match."""
    keys = pd.DatetimeIndex(_utc(timestamps)).unique()
    naive_keys = [item.tz_convert(None).to_pydatetime() for item in keys]
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(columns=columns, filter=ds.field("Opened").isin(pa.array(naive_keys)))
    frame = table.to_pandas()
    frame["Opened"] = _utc(frame["Opened"])
    if frame["Opened"].duplicated().any():
        raise ValueError(f"Duplicate Opened timestamps in {path}")
    return frame.set_index("Opened").sort_index()


def exact_binance_targets(source: pd.DataFrame, opened_times):
    """Return targets only when both exact source candle timestamps exist."""
    opened = pd.DatetimeIndex(_utc(opened_times))
    future_times = opened + pd.Timedelta(minutes=5)
    current = source.reindex(opened)
    future = source.reindex(future_times)
    current_close = pd.to_numeric(current["Close"], errors="coerce").to_numpy(float)
    future_close = pd.to_numeric(future["Close"], errors="coerce").to_numpy(float)
    exact_endpoint = np.isfinite(current_close) & np.isfinite(future_close)
    targets = np.full(len(opened), np.nan, dtype=float)
    targets[exact_endpoint] = (future_close[exact_endpoint] >= current_close[exact_endpoint]).astype(float)
    available_at = future_times + pd.Timedelta(minutes=1)
    return current, future, targets, exact_endpoint, available_at


def read_exact_oof(opened_times):
    exact = _read_exact_rows(OOF_PATH, opened_times, ["Opened", "oof_pred_proba_up"])
    return exact["oof_pred_proba_up"]


def align_exact_windows(common: pd.DataFrame, model_ready_path=MODEL_READY_PATH,
                        feature_columns=None):
    """Map windows and t+5 labels to exact saved rows and make common exclusions."""
    if feature_columns is None:
        raise ValueError("feature_columns must come from the active model metadata")
    common = common.copy().sort_values("market_start_utc").reset_index(drop=True)
    common["Opened"] = _utc(common["Opened"])
    common["market_start_utc"] = _utc(common["market_start_utc"])
    common["market_end_utc"] = _utc(common["market_end_utc"])
    common["decision_available_at"] = _utc(common["decision_available_at"])
    common["resolved_at_utc"] = _utc(common["resolved_at_utc"])
    common["sample_position"] = np.arange(len(common), dtype=np.int64)

    if not common["condition_id"].is_unique:
        raise ValueError("Common Kacho sample has duplicate condition IDs")
    if not common["Opened"].is_unique:
        raise ValueError("Common Kacho sample has duplicate Opened timestamps")
    if not (common["market_start_utc"] == common["Opened"] + pd.Timedelta(minutes=1)).all():
        raise ValueError("Polymarket market start does not exactly equal Opened + 1 minute")

    schema_columns = pq.ParquetFile(model_ready_path).schema.names
    missing_columns = [name for name in feature_columns if name not in schema_columns]
    if missing_columns:
        raise ValueError(f"Active model-ready feature columns are absent: {missing_columns}")

    feature_times = pd.DatetimeIndex(common["Opened"])
    target_times = feature_times + pd.Timedelta(minutes=5)
    source = _read_exact_rows(
        model_ready_path,
        feature_times.append(target_times),
        ["Opened", "Close", "target_5m_candle_up", "target_5m_weight", *feature_columns],
    )
    current, future, target_binance, exact_endpoint, binance_available = exact_binance_targets(
        source, feature_times)
    feature_values = current[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    feature_finite = np.isfinite(feature_values.to_numpy(dtype=float)).all(axis=1)
    feature_row_exists = feature_values.notna().all(axis=1).to_numpy()
    for column in feature_columns:
        common[column] = feature_values[column].to_numpy()

    saved_target = pd.to_numeric(current["target_5m_candle_up"], errors="coerce").to_numpy(float)
    if not np.array_equal(target_binance[exact_endpoint], saved_target[exact_endpoint]):
        raise ValueError("Saved Binance targets differ from exact Close(t+5) >= Close(t) comparisons")
    existing_proxy = pd.to_numeric(common["target_binance_proxy_up"], errors="coerce").to_numpy(float)
    if not np.array_equal(target_binance[exact_endpoint], existing_proxy[exact_endpoint]):
        raise ValueError("Audited Polymarket-history Binance proxy differs from exact model-ready targets")

    weight = pd.to_numeric(current["target_5m_weight"], errors="coerce").to_numpy(float)
    pm_target = pd.to_numeric(common["target_polymarket_up"], errors="coerce").to_numpy(float)
    official_pm_target = pd.to_numeric(common["polymarket_outcome_up"], errors="coerce").to_numpy(float)
    pm_available = common["resolved_at_utc"].where(
        common["resolved_at_utc"].ge(common["market_end_utc"]), common["market_end_utc"])
    binance_available = pd.Series(binance_available, index=common.index)

    exclusions = np.full(len(common), "", dtype=object)

    def exclude(mask, reason):
        mask = np.asarray(mask, dtype=bool)
        choose = mask & (exclusions == "")
        exclusions[choose] = reason

    exclude(~feature_row_exists, "missing_exact_feature_row_or_value")
    exclude(~feature_finite, "nonfinite_active_feature")
    exclude(~exact_endpoint, "missing_exact_binance_t_plus_5_endpoint")
    exclude(~np.isfinite(target_binance) | ~np.isin(target_binance, [0.0, 1.0]), "invalid_binance_target")
    exclude(~np.isfinite(pm_target) | ~np.isin(pm_target, [0.0, 1.0]), "invalid_polymarket_target")
    exclude(~np.isfinite(official_pm_target) | (pm_target != official_pm_target), "target_differs_from_official_outcome")
    exclude(common["outcome_source"].ne("official_gamma"), "non_official_polymarket_outcome")
    exclude(common["validation_status"].ne("validated"), "unvalidated_market_identity_or_boundary")
    exclude(common["resolved_at_utc"].isna(), "missing_polymarket_label_availability_time")
    exclude(~np.isfinite(weight) | (weight <= 0), "invalid_target_weight")
    exclude(~common["quote_valid"].fillna(False).to_numpy(bool), "invalid_or_missing_kacho_quote")
    sizes = common[["up_bid_size", "down_bid_size", "up_ask_size", "down_ask_size"]]
    size_values = sizes.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    exclude(~np.isfinite(size_values).all(axis=1) | (size_values < 0).any(axis=1), "invalid_book_size")
    prices = common[["up_best_ask", "down_best_ask", "up_best_bid", "down_best_bid"]]
    price_values = prices.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    exclude(~np.isfinite(price_values).all(axis=1), "invalid_book_price")

    common["target_binance_up"] = target_binance
    common["target_polymarket_up"] = pm_target
    common["binance_label_available_at"] = pd.DatetimeIndex(binance_available)
    common["polymarket_label_available_at"] = pm_available
    common["both_labels_available_at"] = pd.concat(
        [common["binance_label_available_at"], common["polymarket_label_available_at"]], axis=1
    ).max(axis=1)
    common["target_5m_weight"] = weight
    common["p_source_oof"] = pd.to_numeric(common["p_model_up"], errors="coerce")
    exact_oof = read_exact_oof(common["Opened"])
    common["p_oof_exact"] = exact_oof.reindex(pd.DatetimeIndex(common["Opened"])).to_numpy(float)
    oof_match = np.isfinite(common["p_source_oof"]) & np.isfinite(common["p_oof_exact"])
    if not oof_match.all() or not np.allclose(
            common.loc[oof_match, "p_source_oof"], common.loc[oof_match, "p_oof_exact"],
            rtol=0.0, atol=1e-12):
        raise ValueError("Kacho OOF does not exactly match the active saved OOF artifact")
    exclude(~np.isfinite(common["p_source_oof"]), "missing_exact_main_oof_probability")
    common["exclusion_reason"] = exclusions
    valid = common.loc[common["exclusion_reason"].eq("")].copy().reset_index(drop=True)
    counts = pd.Series(exclusions).replace("", "included").value_counts().to_dict()
    return valid, counts, source.shape[0]


def make_outer_boundaries(n_rows):
    cuts = [int(math.floor(n_rows * fraction)) for fraction in OUTER_FRACTIONS[:-1]] + [n_rows]
    if len(cuts) != 4 or len(set(cuts)) != 4:
        raise ValueError("Three distinct chronological outer folds are required")
    return cuts


def split_outer_window(data: pd.DataFrame, fold: int):
    """Freeze one outer block by original sample position and prior label availability."""
    cuts = make_outer_boundaries(int(data.sample_position.max()) + 1)
    lo, hi = cuts[fold], cuts[fold + 1]
    test = data.loc[data.sample_position.ge(lo) & data.sample_position.lt(hi)].copy()
    if test.empty:
        raise ValueError(f"No common observations remain in outer test fold {fold}")
    outer_cutoff = test.decision_available_at.min()
    past = data.loc[
        data.sample_position.lt(lo) & (data.both_labels_available_at < outer_cutoff)
    ].copy()
    return past, test, outer_cutoff, lo, hi


def assert_identical_pair_rows(ids_b, ids_p, features_b, features_p, weights_b, weights_p):
    if not np.array_equal(np.asarray(ids_b), np.asarray(ids_p)):
        raise AssertionError("B and P training row IDs differ")
    if list(features_b) != list(features_p):
        raise AssertionError("B and P feature lists or feature order differ")
    if not np.array_equal(np.asarray(weights_b, dtype=float), np.asarray(weights_p, dtype=float)):
        raise AssertionError("B and P observation weights differ")


def assert_meta_oos_rows(frame: pd.DataFrame, prediction_columns=("p_target_B", "p_target_P")):
    """Guard a second layer against in-sample first-stage predictions."""
    required = {
        "sample_position", "base_model_train_last_sample_position",
        "base_prediction_is_oos", "decision_available_at",
        "base_model_train_label_available_at", "both_labels_available_at",
    }
    if not required.issubset(frame.columns):
        raise AssertionError(f"Meta training rows lack provenance columns: {sorted(required-set(frame.columns))}")
    if not frame["base_prediction_is_oos"].fillna(False).all():
        raise AssertionError("Second-layer training includes an in-sample B/P prediction")
    if not (frame["base_model_train_last_sample_position"] < frame["sample_position"]).all():
        raise AssertionError("A B/P prediction was made on a row used to fit its first-stage model")
    if not (frame["base_model_train_label_available_at"] < frame["decision_available_at"]).all():
        raise AssertionError("A B/P first-stage label was unavailable at its prediction time")
    if "outer_cutoff" in frame:
        cutoff = pd.Timestamp(frame["outer_cutoff"].iloc[0])
        if not (frame["both_labels_available_at"] < cutoff).all():
            raise AssertionError("Second-layer model training includes an unavailable official label")
    for column in prediction_columns:
        if column not in frame or not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all():
            raise AssertionError(f"Second-layer input {column} is missing or nonfinite")


def _model_params(candidate, n_estimators):
    return {
        "objective": "binary",
        "n_estimators": int(n_estimators),
        "learning_rate": 0.03,
        "max_depth": 5,
        "num_leaves": int(candidate["num_leaves"]),
        "min_child_samples": int(candidate["min_child_samples"]),
        "reg_lambda": float(candidate["reg_lambda"]),
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "subsample_freq": 1,
        "max_bin": 63,
        "random_state": MODEL_SEED,
        "feature_fraction_seed": MODEL_SEED,
        "bagging_seed": MODEL_SEED,
        "data_random_seed": MODEL_SEED,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": MODEL_JOBS,
        "verbosity": -1,
    }


def _fit_lgbm(x, y, weights, candidate, n_estimators):
    model = lgb.LGBMClassifier(**_model_params(candidate, n_estimators))
    model.fit(x, np.asarray(y, dtype=int), sample_weight=np.asarray(weights, dtype=float))
    return model


def _weighted_log_loss(y, p, weights):
    return float(log_loss(np.asarray(y, dtype=int), np.clip(p, 1e-8, 1 - 1e-8),
                          labels=[0, 1], sample_weight=np.asarray(weights, dtype=float)))


def choose_shared_configuration(past: pd.DataFrame, feature_columns, outer_cutoff):
    """Choose one paired B/P config using only an earlier inner time validation."""
    past = past.sort_values("sample_position").reset_index(drop=True)
    split = int(len(past) * INNER_TRAIN_FRACTION)
    if split < 200 or len(past) - split < 100:
        raise ValueError(f"Insufficient earlier observations for inner validation: {len(past)}")
    validation_start = past.loc[split, "decision_available_at"]
    train = past.iloc[:split].copy()
    validation = past.iloc[split:].copy()
    train = train.loc[train["both_labels_available_at"] < validation_start].copy()
    if len(train) < 200 or len(validation) < 100:
        raise ValueError("Inner split is too small after exact label-availability checks")
    if not (train["both_labels_available_at"] < validation_start).all():
        raise AssertionError("Internal training labels were unavailable at validation start")
    if not (validation["both_labels_available_at"] < outer_cutoff).all():
        raise AssertionError("Inner validation uses an outer-test label")
    x_train = train[list(feature_columns)]
    x_validation = validation[list(feature_columns)]
    assert_identical_pair_rows(
        train.condition_id, train.condition_id, feature_columns, feature_columns,
        train.target_5m_weight, train.target_5m_weight)

    results = []
    max_trees = max(TREE_COUNTS)
    for candidate_id, candidate in enumerate(MODEL_CANDIDATES):
        paired_predictions = {}
        for target_name, target_column in (("B", "target_binance_up"), ("P", "target_polymarket_up")):
            model = _fit_lgbm(x_train, train[target_column], train.target_5m_weight,
                              candidate, max_trees)
            for trees in TREE_COUNTS:
                paired_predictions[(target_name, trees)] = model.predict_proba(
                    x_validation, num_iteration=trees)[:, 1]
        for trees in TREE_COUNTS:
            ll_b = _weighted_log_loss(validation.target_binance_up,
                                      paired_predictions[("B", trees)], validation.target_5m_weight)
            ll_p = _weighted_log_loss(validation.target_polymarket_up,
                                      paired_predictions[("P", trees)], validation.target_5m_weight)
            results.append({
                "candidate_id": candidate_id,
                "candidate": dict(candidate),
                "n_estimators": trees,
                "validation_log_loss_B": ll_b,
                "validation_log_loss_P": ll_p,
                "symmetric_mean_log_loss": (ll_b + ll_p) / 2.0,
            })
    best = min(results, key=lambda row: (row["symmetric_mean_log_loss"], row["candidate_id"], row["n_estimators"]))
    return dict(best), results, dict(train_rows=len(train), validation_rows=len(validation),
        validation_start=str(validation_start), latest_train_label_available=str(train.both_labels_available_at.max()),
        outer_cutoff=str(outer_cutoff), train_ids_sha256=_hash_ids(train.condition_id),
        validation_ids_sha256=_hash_ids(validation.condition_id))


def _score(y, probability):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    result = {
        "rows": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "accuracy_at_0_5": float(accuracy_score(y, p >= 0.5)),
        "probability_mean": float(np.mean(p)),
        "probability_std": float(np.std(p)),
        "probability_min": float(np.min(p)),
        "probability_p05": float(np.quantile(p, 0.05)),
        "probability_p25": float(np.quantile(p, 0.25)),
        "probability_median": float(np.quantile(p, 0.5)),
        "probability_p75": float(np.quantile(p, 0.75)),
        "probability_p95": float(np.quantile(p, 0.95)),
        "probability_max": float(np.max(p)),
    }
    diagnostics = calibration_metrics(y, p)
    result["calibration_slope"] = diagnostics["slope"]
    result["calibration_intercept"] = diagnostics["intercept"]
    result["calibration_bins"] = diagnostics["reliability_bins"]
    return result


def paired_block_bootstrap_target_difference(frame, target_column, left_column, right_column,
                                             replications=BOOTSTRAP_REPLICATIONS,
                                             seed=BOOTSTRAP_SEED):
    """Moving 3-day paired bootstrap of loss(left) - loss(right)."""
    ordered = frame.sort_values("market_start_utc").copy()
    y = ordered[target_column].to_numpy(dtype=int)
    left = np.clip(ordered[left_column].to_numpy(float), 1e-8, 1 - 1e-8)
    right = np.clip(ordered[right_column].to_numpy(float), 1e-8, 1 - 1e-8)
    ll_left = -(y * np.log(left) + (1 - y) * np.log(1 - left))
    ll_right = -(y * np.log(right) + (1 - y) * np.log(1 - right))
    daily = pd.DataFrame({
        "day": _utc(ordered.market_start_utc).dt.floor("D"),
        "ll_delta": ll_left - ll_right,
        "brier_delta": (y - left) ** 2 - (y - right) ** 2,
        "n": 1,
    }).groupby("day").sum()
    days = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
    values = daily.reindex(days, fill_value=0).to_numpy(float)
    point_ll = float(np.mean(ll_left - ll_right))
    point_brier = float(np.mean((y - left) ** 2 - (y - right) ** 2))
    rng = np.random.default_rng(seed)
    estimates = []
    n_blocks = int(math.ceil(len(values) / BOOTSTRAP_DAYS))
    for _ in range(replications):
        starts = rng.integers(0, max(len(values) - BOOTSTRAP_DAYS + 1, 1), size=n_blocks)
        indices = np.concatenate([
            np.arange(start, min(start + BOOTSTRAP_DAYS, len(values))) for start in starts
        ])[:len(values)]
        ll_sum, brier_sum, count = values[indices].sum(axis=0)
        if count:
            estimates.append((ll_sum / count, brier_sum / count))
    estimates = np.asarray(estimates, dtype=float)
    return {
        "left_minus_right": f"{left_column} - {right_column}",
        "target": target_column,
        "rows": int(len(ordered)),
        "calendar_days": int(len(days)),
        "block_days": BOOTSTRAP_DAYS,
        "replications": int(len(estimates)),
        "log_loss_difference": point_ll,
        "log_loss_difference_95pct": np.quantile(estimates[:, 0], [0.025, 0.975]).tolist(),
        "brier_difference": point_brier,
        "brier_difference_95pct": np.quantile(estimates[:, 1], [0.025, 0.975]).tolist(),
        "interpretation": "Positive means the left prediction has higher loss; identical calendar blocks are used for both models.",
    }


def _prior_rate(data, target_column, availability_column, outer_cutoff):
    history = data.loc[data[availability_column] < outer_cutoff, target_column].astype(int)
    if history.empty:
        raise ValueError(f"No earlier history for frequency baseline {target_column}")
    return float(history.mean()), int(len(history))


def train_outer_pair(data: pd.DataFrame, feature_columns, output_dir: Path):
    data = data.sort_values("sample_position").reset_index(drop=True)
    cuts = make_outer_boundaries(len(data) if len(data) == EXPECTED_COMMON_ROWS else int(data.sample_position.max()) + 1)
    prediction_blocks, fold_reports = [], []
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(3):
        past, test, outer_cutoff, lo, hi = split_outer_window(data, fold)
        if test.empty or past.empty:
            raise ValueError(f"Empty test/history in outer fold {fold}")
        if not (past.both_labels_available_at < outer_cutoff).all():
            raise AssertionError("Outer training includes a Binance or Polymarket label unavailable at cutoff")
        if (test.sample_position < lo).any() or (test.sample_position >= hi).any():
            raise AssertionError("Outer test positions escaped their frozen block")

        chosen, candidate_scores, inner = choose_shared_configuration(past, feature_columns, outer_cutoff)
        x_train = past[list(feature_columns)]
        x_test = test[list(feature_columns)]
        ids_b = past.condition_id.to_numpy()
        ids_p = past.condition_id.to_numpy()
        assert_identical_pair_rows(ids_b, ids_p, feature_columns, feature_columns,
                                   past.target_5m_weight, past.target_5m_weight)
        if len(past) != len(np.intersect1d(ids_b, ids_p)):
            raise AssertionError("Paired training IDs are not unique and identical")

        model_b = _fit_lgbm(x_train, past.target_binance_up, past.target_5m_weight,
                            chosen["candidate"], chosen["n_estimators"])
        model_p = _fit_lgbm(x_train, past.target_polymarket_up, past.target_5m_weight,
                            chosen["candidate"], chosen["n_estimators"])
        p_b = model_b.predict_proba(x_test)[:, 1]
        p_p = model_p.predict_proba(x_test)[:, 1]
        if not np.isfinite(p_b).all() or not np.isfinite(p_p).all():
            raise ValueError("Outer predictions contain nonfinite probabilities")

        calibration_rows = past.copy()
        calibration_rows["resolved_at_utc"] = calibration_rows["polymarket_label_available_at"]
        platt = fit_calibrator(calibration_rows, "platt", outer_cutoff)
        p_platt = calibrate(platt, test.p_source_oof)
        p_prior_b, prior_b_rows = _prior_rate(data, "target_binance_up", "binance_label_available_at", outer_cutoff)
        p_prior_p, prior_p_rows = _prior_rate(data, "target_polymarket_up", "polymarket_label_available_at", outer_cutoff)

        block = test[[
            "sample_position", "condition_id", "Opened", "market_start_utc", "market_end_utc",
            "decision_available_at", "binance_label_available_at", "polymarket_label_available_at",
            "both_labels_available_at", "target_binance_up", "target_polymarket_up",
            "target_5m_weight", "p_source_oof",
        ]].copy()
        block["fold_id"] = fold
        block["p_source_platt"] = p_platt
        block["p_target_B"] = p_b
        block["p_target_P"] = p_p
        block["p_prior_B"] = p_prior_b
        block["p_prior_P"] = p_prior_p
        block["prior_B_history_rows"] = prior_b_rows
        block["prior_P_history_rows"] = prior_p_rows
        block["base_prediction_is_oos"] = True
        block["base_model_train_last_sample_position"] = int(past.sample_position.max())
        block["base_model_train_label_available_at"] = past.both_labels_available_at.max()
        block["base_model_train_id_sha256"] = _hash_ids(past.condition_id)
        block["base_prediction_outer_cutoff"] = outer_cutoff
        block["label_agreement"] = block.target_binance_up.eq(block.target_polymarket_up)
        if not (block.base_model_train_last_sample_position < block.sample_position).all():
            raise AssertionError("First-stage predictions are not out of sample")
        prediction_blocks.append(block)

        fold_metadata = {
            "fold_id": fold,
            "sample_position_start": lo,
            "sample_position_end_exclusive": hi,
            "outer_cutoff_utc": outer_cutoff,
            "test_rows": int(len(test)),
            "test_start_utc": test.market_start_utc.min(),
            "test_end_utc": test.market_start_utc.max(),
            "past_rows_before_label_availability_filter": int(data.sample_position.lt(lo).sum()),
            "paired_training_rows": int(len(past)),
            "training_start_utc": past.market_start_utc.min(),
            "training_end_utc": past.market_start_utc.max(),
            "latest_training_label_available_at": past.both_labels_available_at.max(),
            "training_ids_sha256": _hash_ids(past.condition_id),
            "feature_count": len(feature_columns),
            "feature_names_sha256": _hash_ids(feature_columns),
            "observation_weights_sha256": _hash_ids(past.target_5m_weight),
            "seed": MODEL_SEED,
            "selected_shared_configuration": chosen,
            "candidate_validation_scores": candidate_scores,
            "inner_validation": inner,
            "outer_test_used_for_tuning_or_early_stopping": False,
            "early_stopping": "Not used; common tree count selected on earlier internal validation.",
        }
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_b.booster_.save_model(str(fold_dir / "target_B_binance.txt"))
        model_p.booster_.save_model(str(fold_dir / "target_P_polymarket.txt"))
        _write_json(fold_dir / "metadata.json", fold_metadata)
        fold_reports.append(fold_metadata)

    predictions = pd.concat(prediction_blocks, ignore_index=True).sort_values("sample_position").reset_index(drop=True)
    expected_test_rows = int(data.sample_position.ge(cuts[0]).sum())
    if len(predictions) != expected_test_rows:
        raise ValueError(f"Outer folds emitted {len(predictions)} rows, expected {expected_test_rows} after common exclusions")
    return predictions, fold_reports


def build_predictive_metrics(predictions: pd.DataFrame, output_dir: Path):
    models = {
        "source_OOF_raw": "p_source_oof",
        "source_OOF_Platt": "p_source_platt",
        "target_B": "p_target_B",
        "target_P": "p_target_P",
        "prior_B": "p_prior_B",
        "prior_P": "p_prior_P",
    }
    targets = {"Binance": "target_binance_up", "Polymarket": "target_polymarket_up"}
    rows, bins = [], []
    for scope, scope_data in [("overall", predictions), *[(f"fold_{fold}", block)
            for fold, block in predictions.groupby("fold_id", sort=True)]]:
        for model_name, probability_column in models.items():
            for target_name, target_column in targets.items():
                scored = _score(scope_data[target_column], scope_data[probability_column])
                score_row = {"scope": scope, "model": model_name, "target": target_name,
                             **{key: value for key, value in scored.items() if key != "calibration_bins"}}
                rows.append(score_row)
                for calibration_bin in scored["calibration_bins"]:
                    bins.append({"scope": scope, "model": model_name, "target": target_name,
                                 **calibration_bin})
    metrics = pd.DataFrame(rows)
    calibration = pd.DataFrame(bins)
    metrics.to_csv(output_dir / "predictive_metrics.csv", index=False)
    calibration.to_csv(output_dir / "calibration_bins.csv", index=False)

    bootstrap = {}
    for target_name, target_column in targets.items():
        bootstrap[target_name] = {"overall": paired_block_bootstrap_target_difference(
            predictions, target_column, "p_target_P", "p_target_B")}
        bootstrap[target_name]["by_fold"] = {
            str(fold): paired_block_bootstrap_target_difference(block, target_column,
                        "p_target_P", "p_target_B", seed=BOOTSTRAP_SEED + int(fold))
            for fold, block in predictions.groupby("fold_id", sort=True)
        }
    agreement = []
    for is_agree, block in predictions.groupby("label_agreement", sort=True):
        label = "agree" if is_agree else "disagree"
        for model_name, probability_column in models.items():
            for target_name, target_column in targets.items():
                agreement.append({"label_group": label, "rows": len(block), "model": model_name,
                                  "target": target_name, **_score(block[target_column], block[probability_column])})
    agreement_frame = pd.DataFrame(agreement)
    agreement_frame.to_csv(output_dir / "label_agreement_diagnostic.csv", index=False)
    p_vs_b_pm = bootstrap["Polymarket"]["overall"]
    clear_loss = bool(
        p_vs_b_pm["log_loss_difference"] >= PRACTICAL_LL_HARM
        and p_vs_b_pm["log_loss_difference_95pct"][0] > 0.0
        and p_vs_b_pm["brier_difference"] > 0.0
        and p_vs_b_pm["brier_difference_95pct"][0] > 0.0
    )
    return metrics, calibration, {
        "paired_P_minus_B_block_bootstrap": bootstrap,
        "label_agreement_diagnostic": agreement_frame.to_dict(orient="records"),
        "label_agreement_counts": {
            "agree": int(predictions.label_agreement.sum()),
            "disagree": int((~predictions.label_agreement).sum()),
        },
        "predeclared_clear_loss_rule": {
            "log_loss_difference_at_least": PRACTICAL_LL_HARM,
            "log_loss_and_brier_95pct_lower_bounds_above_zero": True,
        },
        "P_clearly_loses_to_B_on_official_target": clear_loss,
        "value_over_market_stage": "skipped_if_P_clearly_loses; otherwise run on the smaller common OOS sample",
    }


def run_value_over_market(predictions: pd.DataFrame, common_frames, output_dir: Path):
    """Second layer trained only on earlier first-stage OOS B/P predictions."""
    variant_definitions = {
        "MARKET_ONLY": ("market_only", None),
        "MARKET_PLUS_B": ("market_plus_oof", "p_target_B"),
        "MARKET_PLUS_P": ("market_plus_oof", "p_target_P"),
        "MARKET_PLUS_SOURCE_OOF": ("market_plus_oof", "p_source_oof"),
    }
    records, fold_reports = [], []
    probabilities = []
    for fold in (1, 2):
        fold_test = predictions.loc[predictions.fold_id.eq(fold)].copy()
        outer_cutoff = fold_test.decision_available_at.min()
        past = predictions.loc[
            predictions.sample_position.lt(int(fold_test.sample_position.min()))
            & predictions.both_labels_available_at.lt(outer_cutoff)
        ].copy()
        if past.empty:
            raise ValueError(f"No strictly earlier first-stage OOS predictions for meta fold {fold}")
        assert_meta_oos_rows(past, ("p_target_B", "p_target_P"))
        past["outer_cutoff"] = outer_cutoff
        assert_meta_oos_rows(past, ("p_target_B", "p_target_P"))

        split = int(len(past) * INNER_TRAIN_FRACTION)
        if split < 100 or len(past) - split < 100:
            raise ValueError("Insufficient first-stage OOS rows for second-layer inner validation")
        selection_start = past.iloc[split].decision_available_at
        train = past.iloc[:split].loc[
            past.iloc[:split].both_labels_available_at.lt(selection_start)
        ].copy()
        selection = past.iloc[split:].copy()
        if train.empty or selection.empty:
            raise ValueError("Second-layer validation became empty after availability filtering")
        if not (train.both_labels_available_at < selection_start).all():
            raise AssertionError("Second-layer training label was unavailable before its validation block")
        if not (selection.both_labels_available_at < outer_cutoff).all():
            raise AssertionError("Second-layer tuning used an outer test label")

        market_frame = common_frames[0].merge(
            predictions[["condition_id", "fold_id", "p_target_B", "p_target_P", "p_source_oof",
                         "base_prediction_is_oos", "base_model_train_last_sample_position",
                         "base_model_train_label_available_at", "sample_position",
                         "both_labels_available_at"]],
            on="condition_id", how="inner", validate="one_to_one", suffixes=("", "_prediction"))
        market_past = market_frame.loc[market_frame.sample_position.lt(int(fold_test.sample_position.min()))
            & market_frame.both_labels_available_at.lt(outer_cutoff)].copy()
        market_test = market_frame.loc[market_frame.fold_id.eq(fold)].copy()
        if set(market_past.condition_id) != set(past.condition_id) or set(market_test.condition_id) != set(fold_test.condition_id):
            raise AssertionError("Second-layer book data and base OOS predictions are not on identical rows")

        # Fit all variants on the exact same OOS rows, quote observations and targets.
        meta_train = market_past.loc[market_past.condition_id.isin(set(train.condition_id))].copy()
        meta_selection = market_past.loc[market_past.condition_id.isin(set(selection.condition_id))].copy()
        if set(meta_train.condition_id) != set(train.condition_id) or set(meta_selection.condition_id) != set(selection.condition_id):
            raise AssertionError("Second-layer train/validation rows differ from the shared chronological split")
        selected_c, validation_scores = {}, {}
        for variant, (matrix_variant, prediction_input) in variant_definitions.items():
            train_variant = meta_train.copy()
            selection_variant = meta_selection.copy()
            past_variant = market_past.copy()
            test_variant = market_test.copy()
            # _matrix computes the optional logit for every variant; market_only
            # ignores it, while the three augmented variants substitute their
            # first-stage out-of-sample probability.
            predictor = prediction_input or "p_source_oof"
            train_variant["p_model_up"] = train_variant[predictor]
            selection_variant["p_model_up"] = selection_variant[predictor]
            past_variant["p_model_up"] = past_variant[predictor]
            test_variant["p_model_up"] = test_variant[predictor]
            scores_for_variant = {}
            for strength in MARKET_MODEL_C:
                from utils.polymarket_market_value import fit_market_model, _matrix
                model = fit_market_model(train_variant, matrix_variant, strength, selection_start)
                p = model.predict_proba(_matrix(selection_variant, matrix_variant))[:, 1]
                scores_for_variant[str(strength)] = float(log_loss(
                    selection_variant.target_polymarket_up.astype(int), p, labels=[0, 1]))
            chosen_strength = min(MARKET_MODEL_C, key=lambda c: scores_for_variant[str(c)])
            from utils.polymarket_market_value import fit_market_model, _matrix
            final_model = fit_market_model(past_variant, matrix_variant, chosen_strength, outer_cutoff)
            predicted = final_model.predict_proba(_matrix(test_variant, matrix_variant))[:, 1]
            selected_c[variant] = float(chosen_strength)
            validation_scores[variant] = scores_for_variant
            out_block = market_test[["condition_id", "market_start_utc", "market_end_utc",
                "decision_available_at", "timestamp_utc", "target_polymarket_up"]].copy()
            out_block["fold_id"] = fold
            out_block[variant] = predicted
            probabilities.append(out_block)
        fold_reports.append({
            "fold_id": fold,
            "outer_cutoff_utc": outer_cutoff,
            "first_stage_oos_meta_history_rows": int(len(past)),
            "meta_training_rows": int(len(train)),
            "meta_selection_rows": int(len(selection)),
            "outer_test_rows": int(len(fold_test)),
            "meta_training_latest_label_available_at": train.both_labels_available_at.max(),
            "meta_selection_start_utc": selection_start,
            "selected_C_by_variant": selected_c,
            "inner_log_loss_by_C": validation_scores,
            "all_first_stage_meta_predictions_are_out_of_sample": True,
            "all_meta_labels_available_before_outer_cutoff": True,
        })

    wide_blocks = []
    for fold in (1, 2):
        fold_probabilities = [block for block in probabilities if int(block.fold_id.iloc[0]) == fold]
        fold_wide = None
        for block in fold_probabilities:
            variant = [name for name in variant_definitions if name in block.columns][0]
            one = block[["condition_id", "market_start_utc", "market_end_utc", "decision_available_at",
                         "timestamp_utc", "target_polymarket_up", "fold_id", variant]].rename(
                             columns={variant: "p_" + variant.lower()})
            fold_wide = one if fold_wide is None else fold_wide.merge(
                one[["condition_id", "p_" + variant.lower()]], on="condition_id", validate="one_to_one")
        if fold_wide is None:
            raise AssertionError(f"Missing second-layer OOS predictions for fold {fold}")
        wide_blocks.append(fold_wide)
    wide = pd.concat(wide_blocks, ignore_index=True).sort_values("market_start_utc").reset_index(drop=True)
    if wide is None or wide.condition_id.duplicated().any():
        raise AssertionError("Second-layer OOS predictions are missing or duplicated")
    prediction_columns = ["p_" + name.lower() for name in variant_definitions]
    if not wide[prediction_columns].notna().all().all():
        raise AssertionError("Second-layer variants do not share the same prediction rows")
    wide.to_parquet(output_dir / "value_over_market_oos_predictions.parquet", index=False)

    score_rows = []
    for fold_label, block in [("overall", wide), *[(f"fold_{fold}", g) for fold, g in wide.groupby("fold_id")]]:
        for variant in variant_definitions:
            probability_column = "p_" + variant.lower()
            score_rows.append({"scope": fold_label, "variant": variant,
                               **_score(block.target_polymarket_up, block[probability_column])})
    score_frame = pd.DataFrame(score_rows)
    score_frame.to_csv(output_dir / "value_over_market_metrics.csv", index=False)
    differences = {}
    for variant in ("MARKET_PLUS_B", "MARKET_PLUS_P", "MARKET_PLUS_SOURCE_OOF"):
        differences[variant + "_minus_MARKET_ONLY"] = paired_block_bootstrap_target_difference(
            wide, "target_polymarket_up", "p_" + variant.lower(), "p_market_only")
    differences["MARKET_PLUS_P_minus_MARKET_PLUS_B"] = paired_block_bootstrap_target_difference(
        wide, "target_polymarket_up", "p_market_plus_p", "p_market_plus_b")
    return {
        "status": "complete",
        "evaluated_rows": int(len(wide)),
        "excluded_initial_outer_fold_rows": int(predictions.loc[predictions.fold_id.eq(0)].shape[0]),
        "reason_for_smaller_sample": "Fold 0 has no earlier first-stage B/P OOS predictions. All four second-layer variants therefore use only folds 1 and 2 on the exact shared OOS meta-training/evaluation flow.",
        "folds": fold_reports,
        "metrics": score_frame.to_dict(orient="records"),
        "paired_block_bootstrap": differences,
        "inner_tuning_budget": {"family": "existing standardized L2 logistic market layer",
            "C": list(MARKET_MODEL_C), "same_rows_and_budget_for_all_variants": True,
            "outer_test_used_for_tuning": False},
    }


def _assert_portfolio_conservation(summary):
    if summary["initial_bankroll"] is None:
        return
    if not np.isclose(summary["final_balance"], summary["initial_bankroll"] + summary["realized_pnl"], atol=1e-8):
        raise AssertionError("Continuous portfolio balance does not reconcile to realized PnL")
    if summary["final_balance"] < -1e-8:
        raise AssertionError("Portfolio ending balance is negative")


def run_economics(predictions: pd.DataFrame, output_dir: Path):
    economic_root = output_dir / "economics"
    economic_root.mkdir(parents=True, exist_ok=True)
    quotes_path = SOURCE_RUN / "quotes.parquet"
    results = {
        "rule": "Existing fixed-$5 positive-EV rule at observed ask, exact per-market fees, order minimum, and observed top-ask size.",
        "primary_snapshot_latency_seconds": 0,
        "snapshot_latency_sensitivity_seconds": [1, 2],
        "execution_delay_sensitivity_seconds": [0, 1, 2],
        "stake_usdc": FIXED_STAKE_USDC,
        "continuous_portfolio_start_usdc": INITIAL_PORTFOLIO_USDC,
        "portfolio_drawdown_basis": "cash + open position cost basis; not mark-to-market",
        "independent_funding_note": "Diagnostics are not a feasible $100 continuous portfolio.",
        "variants": {},
    }
    for snapshot_latency in SNAPSHOT_LATENCIES:
        path = COMMON_RUN / f"common_latency_{snapshot_latency}s.parquet"
        market = pd.read_parquet(path).sort_values("market_start_utc").reset_index(drop=True)
        if not market.condition_id.is_unique:
            raise ValueError("Book frame has duplicate markets")
        if len(market) != EXPECTED_COMMON_ROWS:
            raise ValueError(f"Book frame {snapshot_latency}s has {len(market)} rows, expected common 15,677")
        if not market.condition_id.equals(pd.read_parquet(COMMON_RUN / "common_latency_0s.parquet").sort_values(
                "market_start_utc").reset_index(drop=True).condition_id):
            raise AssertionError("Snapshot-latency frames do not share market IDs")
        data = market.merge(predictions[["condition_id", "fold_id", "p_target_B", "p_target_P"]],
                            on="condition_id", how="inner", validate="one_to_one")
        data = data.loc[data.fold_id.notna()].copy().sort_values("market_start_utc").reset_index(drop=True)
        if len(data) != len(predictions):
            raise ValueError(f"Economic sample at {snapshot_latency}s has {len(data)} rows, expected {len(predictions)} common outer-test rows")
        data["fold"] = data["fold_id"].astype(int)
        for variant, probability_column in (("source_OOF_raw", "p_model_up"),
                                            ("target_B", "p_target_B"),
                                            ("target_P", "p_target_P")):
            output_variant = economic_root / f"snapshot_{snapshot_latency}s" / variant
            output_variant.mkdir(parents=True, exist_ok=True)
            variant_results = {}
            for delay in EXECUTION_DELAYS:
                future_quotes = None
                if delay:
                    future_quotes = _execution_quotes(data, quotes_path, delay).set_index("condition_id")
                frame = data.copy()
                frame["p_model_up"] = pd.to_numeric(frame[probability_column], errors="raise")
                independent, independent_trades, _ = simulate_fixed_policy(
                    frame, "p_model_up", delay, None, future_quotes)
                portfolio, portfolio_trades, path_frame = simulate_fixed_policy(
                    frame, "p_model_up", delay, INITIAL_PORTFOLIO_USDC, future_quotes)
                _assert_portfolio_conservation(portfolio)
                independent_trades.to_parquet(output_variant / f"execution_{delay}s_independent_trades.parquet", index=False)
                portfolio_trades.to_parquet(output_variant / f"execution_{delay}s_portfolio_trades.parquet", index=False)
                path_frame.to_parquet(output_variant / f"execution_{delay}s_portfolio_path.parquet", index=False)
                variant_results[str(delay)] = {"independent_funding": independent, "continuous_100_usdc": portfolio}
            results["variants"].setdefault(variant, {})[str(snapshot_latency)] = variant_results
    _write_json(economic_root / "economics.json", results)
    return results


def _input_manifest(feature_columns):
    paths = {}
    for path in INPUT_PATHS:
        if not path.exists():
            raise FileNotFoundError(path)
        paths[str(path.relative_to(ROOT))] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    feature_selection_path = json.loads(MAIN_META_PATH.read_text(encoding="utf-8"))["feature_selection"]["path"]
    selector_path = ROOT / Path(feature_selection_path)
    if selector_path.exists():
        paths[str(selector_path.relative_to(ROOT))] = {"sha256": file_sha256(selector_path),
                                                       "size_bytes": selector_path.stat().st_size}
    return paths


def render_report(manifest, metric_frame, bootstrap, economics, value_over_market):
    def metric(model, target, scope="overall", field="log_loss"):
        row = metric_frame.loc[(metric_frame.model == model) & (metric_frame.target == target)
                               & (metric_frame.scope == scope)]
        return float(row.iloc[0][field]) if len(row) else float("nan")

    lines = [
        f"## Controlled BTC target comparison — {manifest['fingerprint'][:16]}",
        "",
        f"Run fingerprint: `{manifest['fingerprint']}`. Reproduction: `python run_btc_target_comparison.py`.",
        "",
        f"Retrospective research only. It started from the exact 15,677 validated Kacho BTC 5m windows, used the active 64-feature selection artifact, and scored {manifest['sample']['evaluation_rows']:,} outer test rows (the audited reference count is 9,407). The main BTC model, OOF, modeling configuration, `run.py`, and live artifacts were not changed.",
        "",
        "### Coverage and training controls",
        "",
        f"- Rows: {manifest['sample']['base_rows']:,} validated common windows after additional exact-join exclusions; {manifest['sample']['evaluation_rows']:,} outer test rows; active feature matrix {manifest['sample']['base_rows']:,} × {manifest['sample']['feature_count']} before outer splitting.",
        f"- Feature alignment: exact `Opened`; exact Binance close endpoint `Opened + 5 minutes`; Binance label available at `Opened + 6 minutes`; Polymarket label available at `max(resolved_at_utc, market_end_utc)`. No nearest-match or later feature fill.",
        f"- Common feature/target/quote exclusions after the existing 15,677-row Kacho validity filter: `{json.dumps(manifest['sample']['additional_exclusions'], ensure_ascii=False)}`.",
        "- B/P shared training IDs, ordered 64 features, weights, seed and selected hyperparameters/tree count within every fold. Configuration selection used only earlier inner validation and mean B/P log loss; no outer-test early stopping.",
        "",
        "### Direct predictors on the official Polymarket result",
        "",
        "| Model | Log loss | Brier | AUC | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in ("target_B", "target_P", "source_OOF_raw", "source_OOF_Platt", "prior_B", "prior_P"):
        lines.append(f"| {model} | {metric(model, 'Polymarket'):.6f} | {metric(model, 'Polymarket', field='brier'):.6f} | {metric(model, 'Polymarket', field='auc'):.4f} | {metric(model, 'Polymarket', field='accuracy_at_0_5'):.4f} |")
    official_bootstrap = bootstrap["paired_P_minus_B_block_bootstrap"]["Polymarket"]["overall"]
    lines += [
        "",
        f"P−B on official Polymarket labels: log-loss difference {official_bootstrap['log_loss_difference']:+.6f} (paired 3-day block 95% CI {official_bootstrap['log_loss_difference_95pct'][0]:+.6f} to {official_bootstrap['log_loss_difference_95pct'][1]:+.6f}); Brier difference {official_bootstrap['brier_difference']:+.6f} (95% CI {official_bootstrap['brier_difference_95pct'][0]:+.6f} to {official_bootstrap['brier_difference_95pct'][1]:+.6f}). Positive means P is worse.",
        "",
        "Outer-fold log loss against the official Polymarket target:",
        "",
        "| Fold | B | P | Main OOF | Prior B | Prior P |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in range(3):
        scope = f"fold_{fold}"
        lines.append("| " + str(fold) + " | " + " | ".join(
            f"{metric(model, 'Polymarket', scope):.6f}" for model in
            ("target_B", "target_P", "source_OOF_raw", "prior_B", "prior_P")) + " |")
    lines += [
        "",
        "Fold-level direct scores and calibration bins are in the run artifacts. Label agreement/disagreement is an ex-post diagnostic only.",
        "",
        "### Incremental value over the book",
        "",
    ]
    if value_over_market["status"] == "complete":
        lines.append(f"The second layer used {value_over_market['evaluated_rows']:,} shared rows from folds 1–2. Fold 0 was omitted because no earlier B/P out-of-sample predictions existed to train the layer.")
        lines += ["", "| Variant | Log loss | Brier | AUC |", "| --- | ---: | ---: | ---: |"]
        for row in value_over_market["metrics"]:
            if row["scope"] == "overall":
                lines.append(f"| {row['variant']} | {row['log_loss']:.6f} | {row['brier']:.6f} | {row['auc']:.4f} |")
        lines.append("Paired block intervals are in `value_over_market.json`.")
    else:
        lines.append(f"Skipped under the predeclared stop rule: {value_over_market['reason']}.")
    lines += ["", "### Fixed-$5 economics", "", "The economics use the same observed-book positive-EV entry rule, exact market fee schedule, minimum order size and observed top ask liquidity. Each snapshot latency and execution delay has an independent $5-bet diagnostic and a separate continuous $100 portfolio; no deposits or resets. The 1/2-second cases are sensitivities, not measured live latency.", "", "Primary snapshot: earliest recorded quote (0 s). Delayed execution freezes side and observed ask limit at that quote; later quotes may reject execution but cannot change the decision.", ""]
    lines += [
        "| Model | Exec delay | Independent bets: trades / turnover / fees / PnL / PnL-turnover | $100 portfolio: trades / turnover / fees / PnL / final / max DD |",
        "| --- | ---: | --- | --- |",
    ]
    for variant in ("target_B", "target_P"):
        for delay in EXECUTION_DELAYS:
            result = economics["variants"][variant]["0"][str(delay)]
            independent = result["independent_funding"]
            portfolio = result["continuous_100_usdc"]
            independent_cell = (f"{independent['trades']} / ${independent['turnover']:.2f} / ${independent['fees']:.2f} / "
                                f"${independent['realized_pnl']:.2f} / {independent['pnl_per_turnover']}")
            portfolio_cell = (f"{portfolio['trades']} / ${portfolio['turnover']:.2f} / ${portfolio['fees']:.2f} / "
                              f"${portfolio['realized_pnl']:.2f} / ${portfolio['final_balance']:.2f} / "
                              f"{portfolio['max_drawdown']:.2%}")
            lines.append(f"| {variant} | {delay}s | {independent_cell} | {portfolio_cell} |")
    lines += ["", "Full rejection reasons and per-fold economics are in `economics/economics.json`; drawdown is based on cash plus open-position cost, not mark-to-market.", "", "### Snapshot-age audit", "", "The old session's ~1.5 s field measures time from application `fetched_at` (after the REST market/book snapshot fetch completes) until the cached payload is consumed. It is not the age of the exchange book's last change, a last-price timestamp, or feed delay. Current code schedules prefetch about 1.2 s before a bucket and accepts the cached payload up to 2.5 s old; above that it refetches. Thus 1.5 s is an older application snapshot still within the configured cache limit, while exchange freshness remains unknown. The 2026-06-20 run is the only detailed BTC live session found and used a different model fingerprint; no newer runtime logs are available for comparison.", "", "All results are retrospective and do not authorize live deployment.", ""]
    return "\n".join(lines)


def append_report(section: str, fingerprint: str):
    marker = f"<!-- BTC_TARGET_COMPARISON_{fingerprint} -->"
    current = LIVE_REPORT_PATH.read_text(encoding="utf-8") if LIVE_REPORT_PATH.exists() else ""
    if marker in current:
        return
    LIVE_REPORT_PATH.write_text(current.rstrip() + "\n\n" + marker + "\n" + section + "\n", encoding="utf-8")


def run():
    if not MARKET_VALUE_MANIFEST.exists():
        raise FileNotFoundError(MARKET_VALUE_MANIFEST)
    existing_market_run = json.loads(MARKET_VALUE_MANIFEST.read_text(encoding="utf-8"))
    if existing_market_run.get("coverage", {}).get("common_markets") != EXPECTED_COMMON_ROWS:
        raise ValueError("Current audited market-value manifest no longer reports 15,677 common windows")
    if existing_market_run.get("scores", {}).get("latencies", {}).get("0", {}).get("rows") != EXPECTED_OOF_ROWS:
        raise ValueError("Current audited market-value manifest no longer reports 9,407 evaluation rows")

    main_meta = json.loads(MAIN_META_PATH.read_text(encoding="utf-8"))
    feature_columns = list(main_meta["feature_columns"])
    if len(feature_columns) != 64:
        raise ValueError(f"Expected the active 64-feature list, got {len(feature_columns)}")
    input_hashes = _input_manifest(feature_columns)
    config = {
        "asset": "BTC",
        "target_B": "target_5m_candle_up; exact Close(Opened+5m) >= Close(Opened); available at Opened+6m",
        "target_P": "official Gamma UP=1/DOWN=0; available at max(resolved_at_utc, market_end_utc)",
        "common_market_rows": EXPECTED_COMMON_ROWS,
        "expected_outer_evaluation_rows": EXPECTED_OOF_ROWS,
        "outer_fractions": list(OUTER_FRACTIONS),
        "inner_train_fraction": INNER_TRAIN_FRACTION,
        "features": feature_columns,
        "paired_model_family": "LightGBM binary CPU; 64 active feature-selection artifact columns",
        "paired_model_candidates": [dict(candidate) for candidate in MODEL_CANDIDATES],
        "tree_counts": list(TREE_COUNTS),
        "base_learning_rate": 0.03,
        "base_max_depth": 5,
        "base_colsample_bytree": 0.8,
        "base_subsample": 0.8,
        "weights": "existing target_5m_weight; same rows/weights for B and P",
        "seed": MODEL_SEED,
        "bootstrap": {"block_days": BOOTSTRAP_DAYS, "replications": BOOTSTRAP_REPLICATIONS,
                       "seed": BOOTSTRAP_SEED},
        "economics": {"fixed_stake_usdc": FIXED_STAKE_USDC,
                      "initial_portfolio_usdc": INITIAL_PORTFOLIO_USDC,
                      "snapshot_latencies_seconds": list(SNAPSHOT_LATENCIES),
                      "execution_delays_seconds": list(EXECUTION_DELAYS)},
    }
    fingerprint_payload = json.dumps({"input_hashes": input_hashes, "config": config},
                                     sort_keys=True, separators=(",", ":"), default=_json_default)
    fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
    output_dir = ROOT / "data/analysis/polymarket/BTC/target_comparison/runs" / fingerprint[:16]
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("fingerprint") == fingerprint:
            print(f"Existing complete target-comparison run: {output_dir}")
            return existing
        raise FileExistsError(f"Refusing to overwrite a non-complete prior run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "fingerprint": fingerprint,
        "directory": str(output_dir.relative_to(ROOT)),
        "reproduction_command": "python run_btc_target_comparison.py",
        "input_hashes": input_hashes,
        "configuration": config,
        "expected_source_manifest": {
            "market_value_fingerprint": existing_market_run.get("fingerprint"),
            "common_markets": existing_market_run["coverage"]["common_markets"],
            "evaluated_rows": existing_market_run["scores"]["latencies"]["0"]["rows"],
        },
    }
    _write_json(manifest_path, manifest)

    common = pd.read_parquet(COMMON_RUN / "common_latency_0s.parquet")
    if len(common) != EXPECTED_COMMON_ROWS:
        raise ValueError(f"Common Kacho sample has {len(common)} rows, expected {EXPECTED_COMMON_ROWS}")
    raw_kacho = pd.read_parquet(SOURCE_RUN / "kacho_latency_0s.parquet",
                                columns=["condition_id", "eligible", "exclusion_reason"])
    included_ids = set(common.condition_id)
    upstream_excluded = raw_kacho.loc[~raw_kacho.condition_id.isin(included_ids)]
    upstream_reason_counts = upstream_excluded.exclusion_reason.fillna("unspecified").value_counts().to_dict()
    data, additional_exclusions, feature_source_rows = align_exact_windows(common, MODEL_READY_PATH, feature_columns)
    if data.empty:
        raise ValueError("No common rows remain after exact target, feature and quote checks")
    if not data.sample_position.is_monotonic_increasing:
        raise AssertionError("Common windows are not chronological")
    data.to_parquet(output_dir / "aligned_common_dataset.parquet", index=False)
    matrix_bytes = len(data) * len(feature_columns) * np.dtype(np.float64).itemsize
    feature_report = {
        "model_ready_columns": len(pq.ParquetFile(MODEL_READY_PATH).schema.names),
        "active_model_features": len(feature_columns),
        "feature_names": feature_columns,
        "feature_selection_artifact": main_meta["feature_selection"],
        "common_rows": len(data),
        "feature_matrix_shape": [len(data), len(feature_columns)],
        "matrix_float64_bytes": matrix_bytes,
        "feature_rows_read_at_exact_Openeds_and_t_plus_5_endpoints": feature_source_rows,
        "missing_or_nonfinite_feature_rows": int(data[feature_columns].isna().any(axis=1).sum()),
        "feature_matrix_contains_nonfinite": bool(not np.isfinite(data[feature_columns].to_numpy(float)).all()),
        "upstream_kacho_excluded_rows": int(len(upstream_excluded)),
        "upstream_kacho_exclusion_reasons": upstream_reason_counts,
        "additional_common_exclusions": additional_exclusions,
        "label_agreement_rows": int(data.target_binance_up.eq(data.target_polymarket_up).sum()),
        "label_disagreement_rows": int(data.target_binance_up.ne(data.target_polymarket_up).sum()),
        "binance_up_rate": float(data.target_binance_up.mean()),
        "polymarket_up_rate": float(data.target_polymarket_up.mean()),
        "target_weight_values": data.target_5m_weight.value_counts().to_dict(),
    }
    _write_json(output_dir / "feature_and_coverage.json", feature_report)
    manifest["sample"] = {
        "base_rows": len(data),
        "evaluation_rows": int(data.sample_position.ge(make_outer_boundaries(EXPECTED_COMMON_ROWS)[0]).sum()),
        "feature_count": len(feature_columns),
        "feature_matrix_shape": feature_report["feature_matrix_shape"],
        "feature_matrix_bytes": matrix_bytes,
        "additional_exclusions": additional_exclusions,
        "upstream_kacho_exclusions": upstream_reason_counts,
        "label_disagreement_rows_all_common": feature_report["label_disagreement_rows"],
    }
    _write_json(manifest_path, manifest)

    predictions, folds = train_outer_pair(data, feature_columns, output_dir / "models")
    predictions.to_parquet(output_dir / "outer_test_predictions.parquet", index=False)
    manifest["outer_folds"] = folds
    manifest["prediction_rows"] = len(predictions)
    manifest["sample"]["evaluation_rows"] = int(len(predictions))
    _write_json(manifest_path, manifest)

    metric_frame, calibration_frame, predictive = build_predictive_metrics(predictions, output_dir)
    manifest["predictive_results"] = predictive
    p_clear_loss = predictive["P_clearly_loses_to_B_on_official_target"]
    if p_clear_loss:
        value_over_market = {
            "status": "skipped",
            "reason": "P clearly loses under the predeclared rule: at least +0.001 official-target log loss, with paired block 95% lower bounds above zero for both log loss and Brier.",
        }
    else:
        value_dir = output_dir / "value_over_market"
        value_dir.mkdir(parents=True, exist_ok=True)
        common_frames = {latency: pd.read_parquet(COMMON_RUN / f"common_latency_{latency}s.parquet")
                         for latency in SNAPSHOT_LATENCIES}
        # The existing market-value feature transform and validation budgets are reused.
        value_over_market = run_value_over_market(predictions, common_frames, value_dir)
    _write_json(output_dir / "value_over_market.json", value_over_market)
    manifest["value_over_market"] = value_over_market
    _write_json(manifest_path, manifest)

    economics = run_economics(predictions, output_dir)
    manifest["economics"] = economics
    _write_json(manifest_path, manifest)

    # Snapshot age and live-code findings are attached to this report, with no runtime edits.
    live_audit = {
        "only_detailed_BTC_session_found": "20260620_052109",
        "snapshot_age_field_semantics": "REST order-book/market snapshot application-cache residence from fetched_at (set after _fetch_market_snapshot_with_retry returns) until _resolve_market_snapshot consumes it; not last exchange-book change or a measured feed delay.",
        "current_prefetch_schedule_lead_ms": 1200,
        "current_prefetch_max_age_ms": 2500,
        "stale_prefetch_behavior": "When matching payload age exceeds max age, fetches the market snapshot again; errors also trigger a direct refetch. A missing/future snapshot is not accepted as a fresh quote.",
        "age_1500ms_interpretation": "Within the configured 2.5s application-cache limit; does not prove that the price/book is current or stale at the exchange.",
        "unchanged_but_valid_quote": "Cannot be distinguished because the REST book response has no exchange update timestamp/sequence in the saved telemetry.",
        "feed_delay": "Not measured; no exchange book timestamp or feed receive/update markers.",
        "newer_BTC_log_comparison": "No newer detailed BTC session log was present; the one detailed session used a different runtime model fingerprint.",
        "orders_sent": False,
    }
    _write_json(output_dir / "live_snapshot_age_audit.json", live_audit)
    manifest["live_snapshot_age_audit"] = live_audit
    manifest["status"] = "complete"
    manifest["result_files"] = [str(path.relative_to(output_dir)) for path in sorted(output_dir.rglob("*")) if path.is_file()]
    _write_json(manifest_path, manifest)

    section = render_report(manifest, metric_frame, predictive, economics, value_over_market)
    (output_dir / "report.md").write_text(section, encoding="utf-8")
    append_report(section, fingerprint)
    print(f"BTC target comparison complete: {output_dir}")
    print(f"Fingerprint: {fingerprint}")
    return manifest


if __name__ == "__main__":
    run()
