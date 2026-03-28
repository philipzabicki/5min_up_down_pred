import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from common_config_utils import coerce_path
from create_modeling_dataset import parse_fit_results
from modeling_dataset_utils import (
    load_modeling_dataset_settings,
    resolve_modeling_dataset_output_paths,
)
from project_config import load_runtime_artifact_paths

from features.ADX import get_adx_values
from features.BollingerBands import get_bollinger_bands_values
from features.ChaikinOsc import get_chaikin_oscillator_values
from features.candle_features import (
    RAW_OHLCV_COLS,
    STREAK_FEATURE_PREFIX,
    SUPPORTED_CANDLE_FEATURE_COLS,
    add_candle_derived_features,
)
from features.KeltnerChannel import get_keltner_channel_values
from features.MACD import get_macd_values
from features.session_open_features import SUPPORTED_SESSION_COUNTER_COLS
from features.StochOsc import get_stochastic_oscillator_values

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
FUTURES_REST_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"

OUTPUT_DIR = Path("data/analysis/indicator_stability")
OUTPUT_JSON = OUTPUT_DIR / "summary.json"
OUTPUT_CSV = OUTPUT_DIR / "report.csv"
META_PATH_ENV = "AUDIT_MODEL_META_PATH"
REFERENCE_PATH_ENV = "AUDIT_REFERENCE_PATH"
RUNTIME_ARTIFACT_PATHS = load_runtime_artifact_paths()
RUNTIME_INDICATOR_HISTORY_REQUIREMENTS_PATH = Path(
    RUNTIME_ARTIFACT_PATHS["indicator_history_requirements_path"]
)
RUNTIME_MODEL_META_PATH = Path(RUNTIME_ARTIFACT_PATHS["model_meta_path"])

ANCHORS = 20
MAX_WINDOW = 100_000
ABS_TOL = 1e-6
REL_TOL = 1e-5
SCAN_BACK = 128
LIMIT_FEATURES = 0
OHLCV_MATCH_ABS_TOL = 1e-6
OHLCV_MATCH_REL_TOL = 1e-7

USE_REST_BOOTSTRAP = True
REST_BOOTSTRAP_EXTRA_CANDLES = 100_000
REST_TIMEOUT_SEC = 20

MODELING_DATASET_SETTINGS = load_modeling_dataset_settings()
MODELING_OUTPUT_PATHS = resolve_modeling_dataset_output_paths(MODELING_DATASET_SETTINGS)
OHLCV_LOCAL_PATH = Path(MODELING_DATASET_SETTINGS["raw_data_dir"]) / str(
    MODELING_DATASET_SETTINGS["base_data_file"]
)
FIT_RESULTS_DIR = Path(MODELING_DATASET_SETTINGS["fit_results_dir"])

OHLCV_COLS = list(RAW_OHLCV_COLS)
BASE_FEATURE_COLS = (
    set(OHLCV_COLS)
    | set(SUPPORTED_CANDLE_FEATURE_COLS)
    | set(SUPPORTED_SESSION_COUNTER_COLS)
)
VALUE_BUILDERS = {
    "ADX": get_adx_values,
    "BollingerBands": get_bollinger_bands_values,
    "ChaikinOsc": get_chaikin_oscillator_values,
    "KeltnerChannel": get_keltner_channel_values,
    "MACD": get_macd_values,
    "StochOsc": get_stochastic_oscillator_values,
}


class IndicatorSpec:
    __slots__ = (
        "feature_col",
        "indicator",
        "builder",
        "params",
        "required_candles_estimate",
    )

    def __init__(
        self,
        feature_col,
        indicator,
        builder,
        params,
        required_candles_estimate,
    ):
        self.feature_col = feature_col
        self.indicator = indicator
        self.builder = builder
        self.params = params
        self.required_candles_estimate = required_candles_estimate


def _resolve_env_path(env_name):
    raw = os.environ.get(env_name, "").strip()
    return coerce_path(raw) if raw else None


def resolve_meta_path():
    override = _resolve_env_path(META_PATH_ENV)
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(
                f"{META_PATH_ENV} points to a missing file: {override}"
            )
        return override

    if not RUNTIME_MODEL_META_PATH.exists():
        raise FileNotFoundError(
            "Runtime model metadata path from configs/runtime/active.json is missing: "
            f"{RUNTIME_MODEL_META_PATH}"
        )
    return RUNTIME_MODEL_META_PATH


def resolve_reference_path(meta):
    override = _resolve_env_path(REFERENCE_PATH_ENV)
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(
                f"{REFERENCE_PATH_ENV} points to a missing file: {override}"
            )
        return override

    candidates = []
    meta_data_path_raw = str(meta.get("data_path", "")).strip()
    if meta_data_path_raw:
        candidates.append(coerce_path(meta_data_path_raw))
    candidates.extend(
        [
            MODELING_OUTPUT_PATHS["parquet"],
            MODELING_OUTPUT_PATHS["tail_csv"],
        ]
    )
    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not resolve reference dataset. Searched: {searched}"
    )


def normalize_opened_to_utc_naive(values):
    opened = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(opened, pd.Series):
        return opened.dt.tz_localize(None)
    if isinstance(opened, pd.DatetimeIndex):
        return opened.tz_localize(None)
    return pd.DatetimeIndex(opened).tz_localize(None)


def estimate_required_candles(indicator, params):
    periods = [
        int(v)
        for k, v in params.items()
        if "period" in k.lower() and isinstance(v, (int, np.integer))
    ]
    if not periods:
        return 0

    required = max(periods)
    if indicator == "MACD":
        required = max(
            required,
            int(params.get("slow_period", required))
            + int(params.get("signal_period", 0))
            + 50,
        )
    elif indicator in {"ADX", "BollingerBands", "KeltnerChannel", "StochOsc"}:
        required += 50
    return required


def load_indicator_specs(feature_columns, fit_results_dir):
    fit_configs = parse_fit_results(fit_results_dir)
    fit_by_feature_col = {cfg["feature_col"]: cfg for cfg in fit_configs}

    specs = []
    missing_indicator_features = []
    skipped_non_tuned_features = []

    for col in feature_columns:
        if col in BASE_FEATURE_COLS or col.startswith(STREAK_FEATURE_PREFIX):
            continue

        cfg = fit_by_feature_col.get(col)
        if cfg is None:
            if "_fit_" in str(col):
                missing_indicator_features.append(col)
            else:
                skipped_non_tuned_features.append(col)
            continue

        indicator = str(cfg["indicator"])
        params = cfg["params"]
        if not isinstance(params, dict):
            raise ValueError(
                f"Malformed fit config for feature '{col}' (params is not a dict)."
            )
        builder = VALUE_BUILDERS.get(indicator)
        if builder is None:
            raise ValueError(
                f"Indicator '{indicator}' not supported by VALUE_BUILDERS for feature '{col}'."
            )

        specs.append(
            IndicatorSpec(
                feature_col=col,
                indicator=indicator,
                builder=builder,
                params=params,
                required_candles_estimate=estimate_required_candles(indicator, params),
            )
        )

    if missing_indicator_features:
        preview = ", ".join(missing_indicator_features[:10])
        raise FileNotFoundError(
            "Missing fit configs for model feature columns in fit_results_dir "
            f"{Path(fit_results_dir).resolve()}. Missing_count={len(missing_indicator_features)} "
            f"preview=[{preview}]"
        )

    return specs, skipped_non_tuned_features


def build_feature_window_map(report_df, value_col):
    if value_col not in report_df.columns or report_df.empty:
        return {}

    out = {}
    for row in report_df.loc[:, ["feature_col", value_col]].itertuples(index=False):
        raw_value = getattr(row, value_col)
        if pd.isna(raw_value):
            continue
        out[str(row.feature_col)] = int(raw_value)
    return out


def build_runtime_indicator_history_requirements(summary):
    required_keys = (
        "meta_path",
        "fit_results_dir",
        "unstable_feature_count",
        "global_required_stable_window",
        "required_stable_window_by_feature",
    )
    missing = [key for key in required_keys if key not in summary]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            "Cannot publish runtime indicator history requirements because summary "
            f"is missing keys: {joined}"
        )

    stable_window_by_feature = summary["required_stable_window_by_feature"]
    if not isinstance(stable_window_by_feature, dict) or not stable_window_by_feature:
        raise ValueError(
            "Cannot publish runtime indicator history requirements without a non-empty "
            "required_stable_window_by_feature map."
        )

    return {
        "meta_path": str(summary["meta_path"]),
        "fit_results_dir": str(summary["fit_results_dir"]),
        "analysis_summary_path": str(OUTPUT_JSON),
        "analysis_report_path": str(OUTPUT_CSV),
        "unstable_feature_count": int(summary["unstable_feature_count"]),
        "global_required_stable_window": int(summary["global_required_stable_window"]),
        "required_stable_window_by_feature": {
            str(feature_col): int(window)
            for feature_col, window in stable_window_by_feature.items()
        },
    }


def _read_parquet_tail(path, columns, n_tail):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(
            "Reading parquet tail requires pyarrow. Install pyarrow or use CSV reference."
        ) from exc

    pf = pq.ParquetFile(path)
    remaining = int(n_tail)
    chunks = []
    for rg_idx in range(pf.num_row_groups - 1, -1, -1):
        rg_table = pf.read_row_group(rg_idx, columns=columns)
        rg_rows = rg_table.num_rows
        if rg_rows >= remaining:
            chunks.append(rg_table.slice(rg_rows - remaining, remaining))
            remaining = 0
            break
        chunks.append(rg_table)
        remaining -= rg_rows
        if remaining <= 0:
            break

    if not chunks:
        return pd.DataFrame(columns=columns)
    table = pa.concat_tables(list(reversed(chunks)))
    return table.to_pandas()


def read_reference_tail(reference_path, feature_columns, anchors):
    requested_cols = list(dict.fromkeys(["Opened", *OHLCV_COLS, *feature_columns]))
    derived_needed = {c for c in SUPPORTED_CANDLE_FEATURE_COLS if c in feature_columns}
    suffix = reference_path.suffix.lower()
    if suffix == ".csv":
        header_cols = list(pd.read_csv(reference_path, nrows=0).columns)
        header_set = set(header_cols)

        missing_features = [c for c in feature_columns if c not in header_set]
        unsupported_missing = [c for c in missing_features if c not in derived_needed]
        if unsupported_missing:
            raise ValueError(
                "Reference CSV is missing required feature columns: "
                + ", ".join(unsupported_missing[:10])
            )

        usecols = [c for c in requested_cols if c in header_set]
        missing_derived = [c for c in derived_needed if c not in header_set]
        if missing_derived:
            missing_ohlcv = [c for c in OHLCV_COLS if c not in header_set]
            if missing_ohlcv:
                raise ValueError(
                    "Reference CSV misses candle feature columns and required OHLCV "
                    "columns to rebuild them: " + ", ".join(missing_ohlcv)
                )
            usecols = list(dict.fromkeys(usecols + OHLCV_COLS))

        df = pd.read_csv(reference_path, usecols=usecols, parse_dates=["Opened"])
        if missing_derived:
            df = add_candle_derived_features(df, feature_cols=missing_derived)
            missing_after = [c for c in missing_derived if c not in df.columns]
            if missing_after:
                raise ValueError(
                    "Failed to rebuild candle feature columns for reference CSV: "
                    + ", ".join(missing_after)
                )
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            raise RuntimeError(
                "Reading parquet reference requires pyarrow. Install pyarrow or use CSV reference."
            ) from exc

        pf = pq.ParquetFile(reference_path)
        available_cols = set(pf.schema.names)
        missing_features = [c for c in feature_columns if c not in available_cols]
        unsupported_missing = [c for c in missing_features if c not in derived_needed]
        if unsupported_missing:
            raise ValueError(
                "Reference Parquet is missing required feature columns: "
                + ", ".join(unsupported_missing[:10])
            )

        usecols = [c for c in requested_cols if c in available_cols]
        missing_derived = [c for c in derived_needed if c not in available_cols]
        if missing_derived:
            missing_ohlcv = [c for c in OHLCV_COLS if c not in available_cols]
            if missing_ohlcv:
                raise ValueError(
                    "Reference Parquet misses candle feature columns and required OHLCV "
                    "columns to rebuild them: " + ", ".join(missing_ohlcv[:10])
                )
            usecols = list(dict.fromkeys(usecols + OHLCV_COLS))

        df = _read_parquet_tail(reference_path, usecols, int(max(anchors, 1) * 2))
        if missing_derived:
            df = add_candle_derived_features(df, feature_cols=missing_derived)
            missing_after = [c for c in missing_derived if c not in df.columns]
            if missing_after:
                raise ValueError(
                    "Failed to rebuild candle feature columns for reference Parquet: "
                    + ", ".join(missing_after[:10])
                )
    else:
        raise ValueError(
            f"Unsupported reference file format: {reference_path}. Use CSV or Parquet."
        )

    if df.empty:
        raise RuntimeError(f"Reference dataset is empty: {reference_path}")
    df = df.tail(max(int(anchors), 1)).reset_index(drop=True)
    df["Opened"] = normalize_opened_to_utc_naive(df["Opened"])
    keep_cols = [c for c in requested_cols if c in df.columns]
    return df.loc[:, keep_cols]


def fetch_historical_ohlcv(session, candles):
    all_rows = []
    end_time_ms = None

    while len(all_rows) < candles:
        batch_size = min(1500, candles - len(all_rows))
        params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": batch_size}
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        response = session.get(
            FUTURES_REST_KLINES_URL, params=params, timeout=REST_TIMEOUT_SEC
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            break

        all_rows = data + all_rows
        end_time_ms = int(data[0][0]) - 1
        if len(data) < batch_size:
            break
        time.sleep(0.05)

    if not all_rows:
        raise RuntimeError("REST bootstrap returned no candles.")

    now_ms = int(time.time() * 1000)
    rows = []
    for row in all_rows:
        if int(row[6]) >= now_ms:
            continue
        rows.append(
            {
                "Opened": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                "Open": float(row[1]),
                "High": float(row[2]),
                "Low": float(row[3]),
                "Close": float(row[4]),
                "Volume": float(row[5]),
            }
        )

    if not rows:
        raise RuntimeError("No closed candles found during REST bootstrap.")

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["Opened"])
        .sort_values("Opened")
        .reset_index(drop=True)
    )


def load_local_ohlcv():
    if not OHLCV_LOCAL_PATH.exists():
        raise FileNotFoundError(f"Missing local OHLCV file: {OHLCV_LOCAL_PATH}")
    df = pd.read_csv(
        OHLCV_LOCAL_PATH, usecols=["Opened"] + OHLCV_COLS, parse_dates=["Opened"]
    )
    df = (
        df.drop_duplicates(subset=["Opened"])
        .sort_values("Opened")
        .reset_index(drop=True)
    )
    df["Opened"] = normalize_opened_to_utc_naive(df["Opened"])
    return df


def load_ohlcv_for_audit(required_candles):
    if USE_REST_BOOTSTRAP:
        try:
            with requests.Session() as session:
                df = fetch_historical_ohlcv(
                    session, int(required_candles) + int(REST_BOOTSTRAP_EXTRA_CANDLES)
                )
            df["Opened"] = normalize_opened_to_utc_naive(df["Opened"])
            return df, "futures_rest_bootstrap"
        except Exception as exc:
            print(f"[warn] REST bootstrap failed, fallback to local CSV: {exc}")

    return load_local_ohlcv(), "local_csv"


def compare_anchor_ohlcv(reference_df, ohlcv_df, anchor_positions):
    if any(col not in reference_df.columns for col in OHLCV_COLS):
        return {
            "checked": False,
            "is_match": None,
            "max_abs_diff": np.nan,
            "max_abs_diff_by_col": {},
        }

    ref_ohlcv = reference_df.loc[:, OHLCV_COLS].to_numpy(dtype=np.float64, copy=False)
    audit_ohlcv = (
        ohlcv_df.iloc[anchor_positions]
        .loc[:, OHLCV_COLS]
        .to_numpy(
            dtype=np.float64,
            copy=False,
        )
    )
    abs_diff = np.abs(audit_ohlcv - ref_ohlcv)
    finite_diff = abs_diff[np.isfinite(abs_diff)]
    max_abs_diff = float(np.max(finite_diff)) if finite_diff.size else 0.0
    max_abs_diff_by_col = {}
    for col_idx, col in enumerate(OHLCV_COLS):
        col_diff = abs_diff[:, col_idx]
        col_finite = col_diff[np.isfinite(col_diff)]
        max_abs_diff_by_col[col] = (
            float(np.max(col_finite)) if col_finite.size else float("nan")
        )

    return {
        "checked": True,
        "is_match": bool(
            np.allclose(
                audit_ohlcv,
                ref_ohlcv,
                atol=float(OHLCV_MATCH_ABS_TOL),
                rtol=float(OHLCV_MATCH_REL_TOL),
                equal_nan=True,
            )
        ),
        "max_abs_diff": float(max_abs_diff),
        "max_abs_diff_by_col": max_abs_diff_by_col,
    }


def evaluate_feature_stability(
    spec,
    ohlcv_np,
    anchor_positions,
    reference_values,
    abs_tol,
    rel_tol,
    max_window,
    scan_back,
):
    anchor_positions = np.asarray(anchor_positions, dtype=np.int64)
    reference_values = np.asarray(reference_values, dtype=np.float64)

    finite_mask = np.isfinite(reference_values)
    used_anchor_positions = anchor_positions[finite_mask]
    used_reference_values = reference_values[finite_mask]

    if used_anchor_positions.size == 0:
        return {
            "status": "no_finite_reference",
            "stable_min_window": None,
            "stable_used_anchors": 0,
            "stable_max_abs_error": np.nan,
            "formula_window": int(spec.required_candles_estimate),
            "formula_is_stable": False,
            "formula_max_abs_error": np.nan,
        }

    upper_bound = min(int(max_window), int(used_anchor_positions.min()) + 1)
    # Start from the smallest feasible trailing window. The formula estimate is a
    # diagnostic baseline, not a safe lower bound, because overestimation would
    # artificially inflate the reported minimum live history requirement.
    lower_bound = 2
    if lower_bound > upper_bound:
        return {
            "status": "window_bounds_invalid",
            "stable_min_window": None,
            "stable_used_anchors": int(used_anchor_positions.size),
            "stable_max_abs_error": np.nan,
            "formula_window": int(spec.required_candles_estimate),
            "formula_is_stable": False,
            "formula_max_abs_error": np.nan,
        }

    cache = {}

    def check_window(window_len):
        window_len = int(window_len)
        if window_len in cache:
            return cache[window_len]

        max_abs_err = 0.0
        for pos, ref_val in zip(used_anchor_positions, used_reference_values):
            start = int(pos) - window_len + 1
            if start < 0:
                cache[window_len] = (False, np.inf)
                return cache[window_len]

            window = ohlcv_np[start : int(pos) + 1, :]
            try:
                built = spec.builder(spec.params, window)
            except Exception:
                cache[window_len] = (False, np.inf)
                return cache[window_len]

            if built is None:
                cache[window_len] = (False, np.inf)
                return cache[window_len]

            series = np.asarray(built, dtype=np.float64).reshape(-1)
            if series.shape[0] != window.shape[0]:
                cache[window_len] = (False, np.inf)
                return cache[window_len]
            value = float(series[-1])
            if not np.isfinite(value):
                cache[window_len] = (False, np.inf)
                return cache[window_len]

            abs_err = abs(value - float(ref_val))
            tol = float(abs_tol) + float(rel_tol) * max(abs(value), abs(float(ref_val)))
            if abs_err > tol:
                cache[window_len] = (False, abs_err)
                return cache[window_len]
            if abs_err > max_abs_err:
                max_abs_err = abs_err

        cache[window_len] = (True, max_abs_err)
        return cache[window_len]

    formula_window = max(2, int(spec.required_candles_estimate))
    if formula_window <= upper_bound:
        formula_ok, formula_max_err = check_window(formula_window)
    else:
        formula_ok, formula_max_err = (False, np.inf)

    low = lower_bound
    high = upper_bound
    low_ok, _ = check_window(low)
    if low_ok:
        best = low
    else:
        high_ok, _ = check_window(high)
        if not high_ok:
            return {
                "status": "not_stable_within_max_window",
                "stable_min_window": None,
                "stable_used_anchors": int(used_anchor_positions.size),
                "stable_max_abs_error": np.nan,
                "formula_window": formula_window,
                "formula_is_stable": bool(formula_ok),
                "formula_max_abs_error": float(formula_max_err),
            }

        while low < high:
            mid = (low + high) // 2
            mid_ok, _ = check_window(mid)
            if mid_ok:
                high = mid
            else:
                low = mid + 1
        best = low

    local_lower = max(lower_bound, best - max(int(scan_back), 0))
    for w in range(best - 1, local_lower - 1, -1):
        w_ok, _ = check_window(w)
        if w_ok:
            best = w
        else:
            break

    best_ok, best_err = check_window(best)
    if not best_ok:
        return {
            "status": "search_inconsistent",
            "stable_min_window": None,
            "stable_used_anchors": int(used_anchor_positions.size),
            "stable_max_abs_error": np.nan,
            "formula_window": formula_window,
            "formula_is_stable": bool(formula_ok),
            "formula_max_abs_error": float(formula_max_err),
        }

    return {
        "status": "ok",
        "stable_min_window": int(best),
        "stable_used_anchors": int(used_anchor_positions.size),
        "stable_max_abs_error": float(best_err),
        "formula_window": int(formula_window),
        "formula_is_stable": bool(formula_ok),
        "formula_max_abs_error": float(formula_max_err),
    }


def main():
    started_at = time.time()
    meta_path = resolve_meta_path()
    print(f"[info] using model meta: {meta_path.resolve()}")
    print(f"[info] using fit results dir: {FIT_RESULTS_DIR.resolve()}")

    if not FIT_RESULTS_DIR.exists():
        raise FileNotFoundError(f"Missing fit_results dir: {FIT_RESULTS_DIR}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        raise RuntimeError("No feature_columns in model meta.")

    specs, skipped_non_tuned_features = load_indicator_specs(
        feature_columns, FIT_RESULTS_DIR
    )
    if skipped_non_tuned_features:
        preview = ", ".join(skipped_non_tuned_features[:10])
        print(
            "[info] skipping non-tuned model features in audit: "
            f"count={len(skipped_non_tuned_features)} preview=[{preview}]"
        )
    if not specs:
        raise RuntimeError(
            "No tuned indicator features from fit_results_dir matched feature_columns in model meta."
        )
    if LIMIT_FEATURES and LIMIT_FEATURES > 0:
        specs = specs[: int(LIMIT_FEATURES)]
    audited_feature_columns = [spec.feature_col for spec in specs]

    reference_path = resolve_reference_path(meta)

    max_estimate = max((s.required_candles_estimate for s in specs), default=0)
    required_candles = max(MAX_WINDOW, max_estimate) + ANCHORS + 2

    print(
        f"[info] features={len(specs)} anchors={ANCHORS} max_window={MAX_WINDOW} "
        f"required_candles={required_candles} reference={reference_path}"
    )
    reference_df = read_reference_tail(reference_path, audited_feature_columns, ANCHORS)

    ohlcv_df, ohlcv_source = load_ohlcv_for_audit(required_candles)
    ref_opened = normalize_opened_to_utc_naive(reference_df["Opened"])
    if ref_opened.isna().any():
        raise RuntimeError("Reference Opened contains invalid timestamps.")
    ohlcv_opened = normalize_opened_to_utc_naive(ohlcv_df["Opened"])
    if ohlcv_opened.isna().any():
        raise RuntimeError("OHLCV Opened contains invalid timestamps.")

    reference_df = reference_df.copy()
    reference_df["Opened"] = ref_opened
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df["Opened"] = ohlcv_opened

    ohlcv_index = pd.Index(ohlcv_opened)
    anchor_positions = ohlcv_index.get_indexer(ref_opened)

    if np.any(anchor_positions < 0) and ohlcv_source != "local_csv":
        print(
            "[warn] REST bootstrap does not cover all reference timestamps; "
            "fallback to local OHLCV CSV for alignment."
        )
        ohlcv_df = load_local_ohlcv()
        ohlcv_source = "local_csv_fallback_after_rest_missing_anchors"
        ohlcv_index = pd.Index(ohlcv_df["Opened"])
        anchor_positions = ohlcv_index.get_indexer(ref_opened)

    if np.any(anchor_positions < 0):
        missing_count = int(np.sum(anchor_positions < 0))
        raise RuntimeError(
            f"{missing_count} anchor timestamps from reference are missing in OHLCV source: {ohlcv_source}."
        )

    anchor_ohlcv_match = compare_anchor_ohlcv(reference_df, ohlcv_df, anchor_positions)
    if anchor_ohlcv_match["checked"] and not anchor_ohlcv_match["is_match"]:
        if ohlcv_source != "local_csv":
            print(
                "[warn] anchor OHLCV differs between REST bootstrap and reference dataset; "
                "fallback to local OHLCV CSV."
            )
            ohlcv_df = load_local_ohlcv()
            ohlcv_source = "local_csv_fallback_after_rest_ohlcv_mismatch"
            ohlcv_index = pd.Index(ohlcv_df["Opened"])
            anchor_positions = ohlcv_index.get_indexer(ref_opened)
            if np.any(anchor_positions < 0):
                missing_count = int(np.sum(anchor_positions < 0))
                raise RuntimeError(
                    f"{missing_count} anchor timestamps from reference are missing in fallback OHLCV source: {ohlcv_source}."
                )
            anchor_ohlcv_match = compare_anchor_ohlcv(
                reference_df, ohlcv_df, anchor_positions
            )

        if anchor_ohlcv_match["checked"] and not anchor_ohlcv_match["is_match"]:
            raise RuntimeError(
                "Anchor OHLCV values do not match the reference dataset. "
                "This audit would mix different candle histories and produce invalid stability conclusions."
            )

    ohlcv_np = ohlcv_df[OHLCV_COLS].to_numpy(dtype=np.float64, copy=True)
    print(
        "[info] comparison: reference feature values from reference_path "
        "vs feature values recomputed from truncated OHLCV windows ending on the same timestamps"
    )
    print(f"[info] ohlcv_source={ohlcv_source} ohlcv_rows={len(ohlcv_df)}")
    if anchor_ohlcv_match["checked"]:
        print(
            "[info] anchor_ohlcv_match="
            f"{anchor_ohlcv_match['is_match']} "
            f"max_abs_diff={anchor_ohlcv_match['max_abs_diff']:.12g}"
        )

    rows = []
    for i, spec in enumerate(specs, start=1):
        feature_ref = reference_df[spec.feature_col].to_numpy(dtype=np.float64)
        result = evaluate_feature_stability(
            spec=spec,
            ohlcv_np=ohlcv_np,
            anchor_positions=anchor_positions,
            reference_values=feature_ref,
            abs_tol=ABS_TOL,
            rel_tol=REL_TOL,
            max_window=MAX_WINDOW,
            scan_back=SCAN_BACK,
        )

        stable_min = result["stable_min_window"]
        formula = result["formula_window"]
        rows.append(
            {
                "feature_col": spec.feature_col,
                "indicator": spec.indicator,
                "status": result["status"],
                "required_candles_estimate": int(spec.required_candles_estimate),
                "formula_window_checked": int(formula),
                "formula_is_stable": bool(result["formula_is_stable"]),
                "formula_max_abs_error": float(result["formula_max_abs_error"]),
                "stable_min_window": (
                    int(stable_min) if stable_min is not None else np.nan
                ),
                "stable_minus_estimate": (
                    float(stable_min - spec.required_candles_estimate)
                    if stable_min is not None
                    else np.nan
                ),
                "stable_used_anchors": int(result["stable_used_anchors"]),
                "stable_max_abs_error": float(result["stable_max_abs_error"]),
            }
        )

        if i % 10 == 0 or i == len(specs):
            print(
                f"[progress] {i}/{len(specs)} {spec.feature_col} status={result['status']}"
            )

    report_df = pd.DataFrame(rows)
    stable_mask = report_df["stable_min_window"].notna()
    stable_values = report_df.loc[stable_mask, "stable_min_window"].to_numpy(
        dtype=np.float64
    )
    global_stable_window = int(np.nanmax(stable_values)) if stable_values.size else None
    global_estimate_window = (
        int(report_df["required_candles_estimate"].max()) if len(report_df) else None
    )
    unstable_features = report_df.loc[~stable_mask, "feature_col"].tolist()
    stable_window_by_feature = build_feature_window_map(
        report_df,
        "stable_min_window",
    )
    estimate_window_by_feature = build_feature_window_map(
        report_df,
        "required_candles_estimate",
    )
    formula_window_by_feature = build_feature_window_map(
        report_df,
        "formula_window_checked",
    )

    summary = {
        "meta_path": str(meta_path),
        "fit_results_dir": str(FIT_RESULTS_DIR),
        "ohlcv_source": ohlcv_source,
        "ohlcv_symbol": SYMBOL,
        "ohlcv_interval": INTERVAL,
        "reference_path": str(reference_path),
        "meta_feature_count": len(feature_columns),
        "feature_count_evaluated": len(report_df),
        "skipped_non_tuned_feature_count": len(skipped_non_tuned_features),
        "stable_feature_count": int(stable_mask.sum()),
        "unstable_feature_count": int((~stable_mask).sum()),
        "global_required_stable_window": global_stable_window,
        "global_required_estimate_window": global_estimate_window,
        "required_stable_window_by_feature": stable_window_by_feature,
        "required_estimate_window_by_feature": estimate_window_by_feature,
        "formula_window_by_feature": formula_window_by_feature,
        "abs_tol": float(ABS_TOL),
        "rel_tol": float(REL_TOL),
        "anchors_used": len(reference_df),
        "max_window": int(MAX_WINDOW),
        "search_method": "binary_search_from_smallest_window_with_formula_baseline_and_local_backscan",
        "anchor_ohlcv_abs_tol": float(OHLCV_MATCH_ABS_TOL),
        "anchor_ohlcv_rel_tol": float(OHLCV_MATCH_REL_TOL),
        "anchor_ohlcv_match_checked": bool(anchor_ohlcv_match["checked"]),
        "anchor_ohlcv_is_match": anchor_ohlcv_match["is_match"],
        "anchor_ohlcv_max_abs_diff": float(anchor_ohlcv_match["max_abs_diff"]),
        "anchor_ohlcv_max_abs_diff_by_col": anchor_ohlcv_match["max_abs_diff_by_col"],
        "elapsed_sec": float(time.time() - started_at),
        "unstable_features": unstable_features,
        "comparison": (
            "reference feature value at anchor timestamp vs recomputed value "
            "from trailing OHLCV window ending at the same anchor timestamp"
        ),
    }

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    runtime_requirements = build_runtime_indicator_history_requirements(summary)
    RUNTIME_INDICATOR_HISTORY_REQUIREMENTS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    RUNTIME_INDICATOR_HISTORY_REQUIREMENTS_PATH.write_text(
        json.dumps(runtime_requirements, indent=2),
        encoding="utf-8",
    )

    print(
        f"[done] stable={summary['stable_feature_count']}/{summary['feature_count_evaluated']} "
        f"global_stable_window={summary['global_required_stable_window']} "
        f"global_estimate_window={summary['global_required_estimate_window']}"
    )
    print(f"[done] report_csv={OUTPUT_CSV}")
    print(f"[done] summary_json={OUTPUT_JSON}")
    print(
        "[done] runtime_indicator_history_requirements="
        f"{RUNTIME_INDICATOR_HISTORY_REQUIREMENTS_PATH}"
    )


if __name__ == "__main__":
    main()
