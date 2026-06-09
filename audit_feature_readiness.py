from pathlib import Path

import requests

from features.ADX import get_adx_values
from features.BollingerBands import get_bollinger_bands_values
from features.ChaikinOsc import get_chaikin_oscillator_values
from features.KeltnerChannel import get_keltner_channel_values
from features.MACD import get_macd_values
from features.StochOsc import get_stochastic_oscillator_values
from features.candle_features import (
    RAW_OHLCV_COLS,
    STREAK_FEATURE_PREFIX,
    SUPPORTED_CANDLE_FEATURE_COLS,
)
from features.session_open_features import SUPPORTED_SESSION_OPEN_FEATURE_COLS
from utils.config import coerce_path
from utils.data import (
    load_modeling_dataset_settings,
    resolve_modeling_dataset_output_paths,
)
from utils.project_config import load_runtime_artifact_paths

FUTURES_REST_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"

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
SYMBOL = str(
    MODELING_DATASET_SETTINGS.get("volume_symbol")
    or MODELING_DATASET_SETTINGS.get("symbol")
).upper()
INTERVAL = str(MODELING_DATASET_SETTINGS["interval"])
OUTPUT_DIR = Path("data/analysis/indicator_stability") / str(
    MODELING_DATASET_SETTINGS["active_asset"]
)
OUTPUT_JSON = OUTPUT_DIR / "summary.json"
OUTPUT_CSV = OUTPUT_DIR / "report.csv"
MODELING_OUTPUT_PATHS = resolve_modeling_dataset_output_paths(MODELING_DATASET_SETTINGS)
OHLCV_LOCAL_PATH = Path(MODELING_DATASET_SETTINGS["raw_data_dir"]) / str(
    MODELING_DATASET_SETTINGS["base_data_file"]
)
FIT_RESULTS_DIR = Path(MODELING_DATASET_SETTINGS["fit_results_dir"])

OHLCV_COLS = list(RAW_OHLCV_COLS)
BASE_FEATURE_COLS = (
        set(OHLCV_COLS)
        | set(SUPPORTED_CANDLE_FEATURE_COLS)
        | set(SUPPORTED_SESSION_OPEN_FEATURE_COLS)
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
    validate_volume_profile_feature_columns(
        feature_columns,
        source_label=f"model metadata feature_columns for indicator audit ({fit_results_dir})",
    )
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

            window = ohlcv_np[start: int(pos) + 1, :]
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


def _run_indicator_stability_audit_impl():
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


def _capture_indicator_stability_globals():
    excluded = {
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__annotations__",
        "__builtins__",
        "__file__",
        "__cached__",
        "_capture_indicator_stability_globals",
    }
    return {
        key: value
        for key, value in globals().items()
        if key not in excluded
    }


_INDICATOR_STABILITY_GLOBALS = _capture_indicator_stability_globals()
_INDICATOR_STABILITY_MISSING = object()


def run_indicator_stability_audit():
    previous = {}
    for key, value in _INDICATOR_STABILITY_GLOBALS.items():
        previous[key] = globals().get(key, _INDICATOR_STABILITY_MISSING)
        globals()[key] = value

    try:
        return _INDICATOR_STABILITY_GLOBALS["_run_indicator_stability_audit_impl"]()
    finally:
        for key, value in previous.items():
            if value is _INDICATOR_STABILITY_MISSING:
                globals().pop(key, None)
            else:
                globals()[key] = value


# Live feature parity audit starts below. Its globals intentionally remain the
# default module state; indicator stability globals are restored only while that
# first audit is running.
import copy
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from create_modeling_dataset import (
    add_indicator_values,
    concat_feature_frame,
    parse_fit_results,
)
from features.basis_premium_features import validate_basis_premium_feature_columns
from features.candle_features import (
    RAW_OHLCV_COLS,
    SUPPORTED_CANDLE_FEATURE_COLS,
    add_candle_derived_features,
    add_candle_streak_features,
    resolve_candle_derived_feature_cols,
    resolve_candle_pattern_feature_cols,
    resolve_streak_interval_to_rule,
)
from features.session_open_features import (
    add_session_open_features,
)
from features.realized_volatility import add_realized_volatility_features
from features.volume_profile_fixed_range import (
    FEATURE_VERSION as VP_FEATURE_VERSION,
    AUDIT_ANCHOR_STATE_DIR as VP_AUDIT_ANCHOR_STATE_DIR,
    PSEUDO_LIVE_AUDIT_MODELING_STATE_DIR as VP_PSEUDO_LIVE_AUDIT_MODELING_STATE_DIR,
    PSEUDO_LIVE_AUDIT_RUNTIME_STATE_DIR as VP_PSEUDO_LIVE_AUDIT_RUNTIME_STATE_DIR,
    build_volume_profile_features,
    bootstrap_state_from_history,
    load_state as load_volume_profile_state,
    normalize_config as normalize_volume_profile_config,
    save_state as save_volume_profile_state,
    state_matches_config as volume_profile_state_matches_config,
    validate_volume_profile_feature_columns,
    validate_volume_profile_model_metadata,
)
from run import (
    INDICATOR_HISTORY_REQUIREMENTS_PATH,
    INTERVAL,
    TRADE_POLICY_CONFIG_PATH,
    LIVE_INITIAL_BANKROLL_USDC,
    MODELING_DATASET_SETTINGS,
    MODEL_META_PATH,
    OHLCV_COLS,
    SYMBOL,
    LivePredictor,
    interval_to_timedelta,
    load_indicator_history_requirements,
    load_trade_policy_runtime_config,
    load_indicator_specs,
    load_model_and_meta,
    load_required_stable_window,
    parse_target_bucket_minutes,
)
from utils.data import (
    resolve_modeling_dataset_parquet_path,
    split_feature_subset,
)

INTERVAL_DELTA = interval_to_timedelta(INTERVAL)
DEFAULT_AUDIT_DAYS_BACK = 30
DEFAULT_BOOTSTRAP_CANDLES = 21_600
DEFAULT_MAX_KEEP = DEFAULT_BOOTSTRAP_CANDLES
PREDICTION_DIFF_TOL = 1e-6
REL_DIFF_DENOM_FLOOR = 1e-6
FEATURE_DROP_MAX_ABS_DIFF_TOL = 1e-2
FEATURE_DROP_MEAN_ABS_DIFF_TOL = 1e-3
FEATURE_DROP_MAX_REL_DIFF_TOL = 1e-2

# Runtime settings
AUDIT_DAYS_BACK = DEFAULT_AUDIT_DAYS_BACK
AUDIT_BOOTSTRAP_CANDLES = DEFAULT_BOOTSTRAP_CANDLES
AUDIT_MAX_STEPS = 10_080
AUDIT_MAX_KEEP = AUDIT_BOOTSTRAP_CANDLES
AUDIT_MODEL_META_PATH = MODEL_META_PATH
AUDIT_PARQUET_PATH = None
AUDIT_USE_ANCHOR_VP_STATE = True
AUDIT_OVERWRITE_ANCHOR_VP_STATE = False
AUDIT_ALLOW_UNSTABLE_INDICATOR_SUMMARY = True
AUDIT_OUTPUT_DIR = None
AUDIT_DRILLDOWN_FEATURE = None
AUDIT_TOP_N = 50
AUDIT_PROGRESS_ENABLED = True
AUDIT_PROGRESS_EVERY_STEPS = 60
AUDIT_PROGRESS_MAX_UPDATES = 12
AUDIT_CONSOLE_TOP_N = 5
AUDIT_CONSOLE_LABEL_MAX_LEN = 100
AUDIT_WRITE_DEBUG_CSVS = False
AUDIT_MAX_MEAN_PROBA_ABS_DIFF = None
AUDIT_MAX_MAX_PROBA_ABS_DIFF = None
AUDIT_MAX_SIGNAL_MISMATCH_RATE = None
AUDIT_MAX_BUSINESS_MISMATCH_RATE = None


def _env_optional_float(name, default=None):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


AUDIT_MAX_MEAN_PROBA_ABS_DIFF = _env_optional_float(
    "AUDIT_MAX_MEAN_PROBA_ABS_DIFF",
    AUDIT_MAX_MEAN_PROBA_ABS_DIFF,
)
AUDIT_MAX_MAX_PROBA_ABS_DIFF = _env_optional_float(
    "AUDIT_MAX_MAX_PROBA_ABS_DIFF",
    AUDIT_MAX_MAX_PROBA_ABS_DIFF,
)
AUDIT_MAX_SIGNAL_MISMATCH_RATE = _env_optional_float(
    "AUDIT_MAX_SIGNAL_MISMATCH_RATE",
    AUDIT_MAX_SIGNAL_MISMATCH_RATE,
)
AUDIT_MAX_BUSINESS_MISMATCH_RATE = _env_optional_float(
    "AUDIT_MAX_BUSINESS_MISMATCH_RATE",
    AUDIT_MAX_BUSINESS_MISMATCH_RATE,
)


def _ensure_utc_opened(series):
    opened = pd.to_datetime(series)
    if getattr(opened.dt, "tz", None) is None:
        return opened.dt.tz_localize("UTC")
    return opened.dt.tz_convert("UTC")


def _naive_utc_timestamp(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)


def _optional_path(value):
    if value is None:
        return None
    return Path(value)


def _load_model_feature_importance_frame(meta):
    artifacts = dict(meta.get("artifacts") or {})
    raw_path = str(artifacts.get("final_feature_importance_csv") or "").strip()
    if not raw_path:
        return None

    path = Path(raw_path)
    if not path.exists():
        return None

    frame = pd.read_csv(path)
    required_cols = {"feature", "importance_gain", "importance_split"}
    if not required_cols.issubset(frame.columns):
        return None
    validate_volume_profile_feature_columns(
        frame["feature"].tolist(),
        source_label=f"feature importance artifact {path}",
    )
    validate_basis_premium_feature_columns(
        frame["feature"].tolist(),
        source_label=f"feature importance artifact {path}",
    )

    out = frame.loc[:, ["feature", "importance_gain", "importance_split"]].copy()
    out["importance_gain"] = pd.to_numeric(
        out["importance_gain"], errors="coerce"
    ).fillna(0.0)
    out["importance_split"] = pd.to_numeric(
        out["importance_split"], errors="coerce"
    ).fillna(0.0)
    return out


def resolve_anchor_volume_profile_state_path(anchor_candle_opened):
    stamp = pd.Timestamp(anchor_candle_opened).strftime("%Y%m%d_%H%M")
    return (
            VP_AUDIT_ANCHOR_STATE_DIR
            / f"{SYMBOL}_{INTERVAL}_{VP_FEATURE_VERSION}_audit_anchor_{stamp}"
    )


def _feature_group_map(feature_columns):
    parts = split_feature_subset(feature_columns, source_label="audit feature columns")
    group_by_feature = {}
    for feature in parts["raw_ohlcv_cols"]:
        group_by_feature[feature] = "raw_ohlcv"
    for feature in parts["candle_feature_cols"]:
        group_by_feature[feature] = "candle"
    for feature in parts["streak_feature_cols"]:
        group_by_feature[feature] = "streak"
    for feature in parts["session_feature_cols"]:
        group_by_feature[feature] = "session"
    for feature in parts["realized_volatility_feature_cols"]:
        group_by_feature[feature] = "realized_volatility"
    for feature in parts["basis_premium_feature_cols"]:
        group_by_feature[feature] = "basis_premium"
    for feature in parts["indicator_feature_cols"]:
        group_by_feature[feature] = "indicator"
    for feature in parts["volume_profile_feature_cols"]:
        group_by_feature[feature] = "volume_profile"
    for feature in parts["unclassified_feature_cols"]:
        group_by_feature[feature] = "unclassified"
    return group_by_feature


def _ignored_basis_premium_feature_columns(feature_columns):
    parts = split_feature_subset(feature_columns, source_label="audit feature columns")
    return tuple(parts["basis_premium_feature_cols"])


def _audited_feature_columns(feature_columns):
    ignored = set(_ignored_basis_premium_feature_columns(feature_columns))
    return [feature for feature in feature_columns if feature not in ignored]


def _copy_ignored_basis_premium_features(candidate_df, reference_df, feature_columns):
    ignored_cols = _ignored_basis_premium_feature_columns(feature_columns)
    if not ignored_cols:
        return candidate_df

    missing_reference_cols = [col for col in ignored_cols if col not in reference_df]
    if missing_reference_cols:
        raise ValueError(
            "Stored audit frame is missing ignored basis premium columns: "
            f"{missing_reference_cols[:10]}"
        )

    out = candidate_df.copy()
    for col in ignored_cols:
        out[col] = reference_df[col].to_numpy(dtype=np.float64, copy=False)

    leading_cols = [col for col in out.columns if col not in feature_columns]
    return out.loc[:, [*leading_cols, *feature_columns]].copy()


def _safe_rowwise_mean(values):
    counts = np.isfinite(values).sum(axis=1)
    sums = np.nansum(values, axis=1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def _format_duration(seconds):
    seconds = max(0.0, float(seconds))
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _summary_rate(count, total):
    total = int(total)
    if total <= 0:
        return float("nan")
    return float(count) / float(total)


def _evaluate_guardrail_violations(summary):
    decision_row_count = int(summary.get("decision_row_count", 0) or 0)
    signal_mismatch_rate = float(
        summary.get(
            "signal_mismatch_rate",
            _summary_rate(summary.get("rows_with_signal_mismatch", 0), decision_row_count),
        )
    )
    business_mismatch_rate = float(
        summary.get(
            "business_decision_mismatch_rate",
            _summary_rate(
                summary.get("rows_with_business_decision_mismatch", 0),
                decision_row_count,
            ),
        )
    )

    checks = [
        (
            AUDIT_MAX_MEAN_PROBA_ABS_DIFF,
            "mean_proba_up_abs_diff",
            float(summary.get("mean_proba_up_abs_diff", np.nan)),
        ),
        (
            AUDIT_MAX_MAX_PROBA_ABS_DIFF,
            "max_proba_up_abs_diff",
            float(summary.get("max_proba_up_abs_diff", np.nan)),
        ),
        (
            AUDIT_MAX_SIGNAL_MISMATCH_RATE,
            "signal_mismatch_rate",
            signal_mismatch_rate,
        ),
        (
            AUDIT_MAX_BUSINESS_MISMATCH_RATE,
            "business_decision_mismatch_rate",
            business_mismatch_rate,
        ),
    ]

    violations = []
    for limit, label, value in checks:
        if limit is None or not np.isfinite(float(limit)):
            continue
        if np.isfinite(value) and float(value) > float(limit):
            violations.append(
                f"{label}={float(value):.6f} exceeds limit={float(limit):.6f}"
            )

    return violations


def _truncate_console_label(value, *, max_len=AUDIT_CONSOLE_LABEL_MAX_LEN):
    text = str(value or "").strip()
    if not text:
        return "n/a"
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def _format_console_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        text = value.strip()
        return text or "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric_value):
        return "n/a"
    abs_value = abs(numeric_value)
    if abs_value >= 1000.0 or (0.0 < abs_value < 1e-4):
        return f"{numeric_value:.3e}"
    return f"{numeric_value:.6f}".rstrip("0").rstrip(".")


def _format_console_rate(value):
    if value is None:
        return "n/a"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric_value):
        return "n/a"
    return f"{numeric_value * 100.0:.3f}%"


def _format_console_pct_value(value):
    if value is None:
        return "n/a"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric_value):
        return "n/a"
    return f"{numeric_value:.3f}%"


def _print_console_ranked_rows(
        title,
        frame,
        *,
        label_col,
        metric_specs,
        limit=AUDIT_CONSOLE_TOP_N,
):
    if frame is None or frame.empty:
        return

    print()
    print(f"{title}:")
    display_df = frame.head(int(limit)).reset_index(drop=True)
    for row_idx, row in display_df.iterrows():
        parts = [f"{row_idx + 1}. {_truncate_console_label(row.get(label_col))}"]
        for column_name, display_label, formatter in metric_specs:
            if column_name not in display_df.columns:
                continue
            parts.append(f"{display_label}={formatter(row.get(column_name))}")
        print(f"  {' | '.join(parts)}")


def _print_audit_console_summary(results, written_paths):
    summary = results["live_vs_stored_report"]["summary"]
    reason_summary = results["drift_reason_report"]["summary"]
    violations = _evaluate_guardrail_violations(summary)

    print("Audit summary:")
    print(
        "  window: "
        f"{summary.get('audit_start', 'n/a')} -> {summary.get('audit_end', 'n/a')} | "
        f"bootstrap_rows={_format_console_value(summary.get('bootstrap_rows'))} | "
        "audit_rows_total_1m="
        f"{_format_console_value(summary.get('audit_rows_total_1m'))} | "
        f"decision_rows={_format_console_value(summary.get('decision_row_count'))}"
    )
    print(
        "  prediction drift: "
        f"mean_abs_proba={_format_console_value(summary.get('mean_proba_up_abs_diff'))} | "
        f"max_abs_proba={_format_console_value(summary.get('max_proba_up_abs_diff'))} | "
        "rows_gt_tol="
        f"{_format_console_value(summary.get('rows_with_proba_diff_gt_tol'))}"
    )
    print(
        "  business drift: "
        f"signal={_format_console_value(summary.get('rows_with_signal_mismatch'))} "
        f"({_format_console_rate(summary.get('signal_mismatch_rate'))}) | "
        "decision="
        f"{_format_console_value(summary.get('rows_with_business_decision_mismatch'))} "
        f"({_format_console_rate(summary.get('business_decision_mismatch_rate'))}) | "
        "policy_any="
        f"{_format_console_value(summary.get('rows_with_any_policy_mismatch'))} "
        f"({_format_console_rate(summary.get('any_policy_mismatch_rate'))})"
    )
    ignored_basis_count = int(summary.get("ignored_basis_premium_feature_count", 0) or 0)
    if ignored_basis_count:
        print(
            "  ignored features: "
            f"basis_premium={_format_console_value(ignored_basis_count)}"
        )
    print(
        "  dominant source: "
        f"basis={_format_console_value(reason_summary.get('explanation_basis'))} | "
        f"group={_format_console_value(reason_summary.get('dominant_prediction_impact_group'))} | "
        f"builder={_format_console_value(reason_summary.get('dominant_prediction_impact_builder_name'))} | "
        "feature="
        f"{_truncate_console_label(reason_summary.get('dominant_prediction_impact_feature'))}"
    )
    if violations:
        print("  guardrails: VIOLATED")
        for violation in violations:
            print(f"    - {violation}")
    else:
        print("  guardrails: OK")

    _print_console_ranked_rows(
        "Features to inspect",
        _build_features_to_inspect_df(
            results["live_vs_stored_report"]["feature_summary_df"],
            decision_row_count=summary.get("decision_row_count", 0),
            top_k=min(int(AUDIT_CONSOLE_TOP_N), int(AUDIT_TOP_N)),
        ),
        label_col="feature",
        metric_specs=[
            ("severity", "severity", _format_console_value),
            ("pred_shift_rows", "pred_shift_rows", _format_console_value),
            ("pred_shift_rows_pct", "pred_shift_pct", _format_console_pct_value),
            ("mean_pred_shift", "mean_shift", _format_console_value),
            ("max_pred_shift", "max_shift", _format_console_value),
            (
                "rows_where_prediction_diff_exceeds_tol_explained",
                "pred_diff_rows",
                _format_console_value,
            ),
            (
                "rows_where_up_down_prediction_flips_explained",
                "up_down_flip_rows",
                _format_console_value,
            ),
        ],
    )

    print()
    print("Artifacts:")
    for label, path in written_paths.items():
        print(f"  {label}: {path}")

    return violations


def _is_live_decision_opened(opened, bucket_minutes):
    opened = pd.Timestamp(opened)
    bucket_start = opened.floor(f"{int(bucket_minutes)}min")
    bucket_end = bucket_start + pd.Timedelta(minutes=int(bucket_minutes) - 1)
    return opened == bucket_end


def _safe_colwise_mean(values):
    counts = np.isfinite(values).sum(axis=0)
    sums = np.nansum(values, axis=0)
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def _safe_colwise_rmse(values):
    counts = np.isfinite(values).sum(axis=0)
    sums = np.nansum(np.square(values, dtype=np.float64), axis=0)
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = np.sqrt(sums[valid] / counts[valid])
    return out


def _safe_nanmax_axis1(values):
    valid = np.isfinite(values)
    safe = np.where(valid, values, -np.inf)
    out = safe.max(axis=1)
    out[~valid.any(axis=1)] = np.nan
    return out.astype(np.float64, copy=False)


def _safe_nanmax_axis0(values):
    valid = np.isfinite(values)
    safe = np.where(valid, values, -np.inf)
    out = safe.max(axis=0)
    out[~valid.any(axis=0)] = np.nan
    return out.astype(np.float64, copy=False)


def _safe_relative_diff(
        candidate_values,
        reference_values,
        *,
        denom_floor=REL_DIFF_DENOM_FLOOR,
):
    candidate = np.asarray(candidate_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    out = np.abs(candidate - reference)
    out /= np.maximum(np.abs(reference), float(denom_floor))
    out[~(np.isfinite(candidate) & np.isfinite(reference))] = np.nan
    return out


def _safe_argmax_axis1(values):
    safe = np.where(np.isfinite(values), values, -np.inf)
    return safe.argmax(axis=1)


def _safe_argmax_axis0(values):
    safe = np.where(np.isfinite(values), values, -np.inf)
    return safe.argmax(axis=0)


def _column_values_match(candidate_values, reference_values):
    candidate = np.asarray(candidate_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    same_mask = (candidate == reference) | (np.isnan(candidate) & np.isnan(reference))
    return bool(same_mask.all())


def _fit_indicator_config_map(feature_columns):
    fit_results_dir = Path(MODELING_DATASET_SETTINGS["fit_results_dir"])
    configs = parse_fit_results(fit_results_dir)
    selected = set(feature_columns)
    return {
        cfg["feature_col"]: cfg for cfg in configs if cfg["feature_col"] in selected
    }


def _feature_builder_frame(feature_columns):
    feature_parts = split_feature_subset(
        feature_columns,
        source_label="audit feature columns",
    )
    indicator_config_map = _fit_indicator_config_map(feature_columns)
    candle_pattern_cols = set(
        resolve_candle_pattern_feature_cols(feature_parts["candle_feature_cols"])
    )

    records = []
    for feature in feature_columns:
        if feature in RAW_OHLCV_COLS:
            records.append(
                {
                    "feature": feature,
                    "builder_family": "raw_ohlcv",
                    "builder_name": "raw_ohlcv_passthrough",
                    "builder_source": None,
                }
            )
            continue

        if feature in feature_parts["streak_feature_cols"]:
            records.append(
                {
                    "feature": feature,
                    "builder_family": "streak",
                    "builder_name": "add_candle_streak_features",
                    "builder_source": None,
                }
            )
            continue

        if feature in feature_parts["session_feature_cols"]:
            records.append(
                {
                    "feature": feature,
                    "builder_family": "session",
                    "builder_name": "add_session_open_features",
                    "builder_source": None,
                }
            )
            continue

        if feature in feature_parts["realized_volatility_feature_cols"]:
            records.append(
                {
                    "feature": feature,
                    "builder_family": "realized_volatility",
                    "builder_name": "add_realized_volatility_features",
                    "builder_source": "realized_volatility",
                }
            )
            continue

        if feature in feature_parts["basis_premium_feature_cols"]:
            records.append(
                {
                    "feature": feature,
                    "builder_family": "basis_premium",
                    "builder_name": "add_basis_premium_features",
                    "builder_source": "basis_premium_features",
                }
            )
            continue

        if feature in feature_parts["volume_profile_feature_cols"]:
            records.append(
                {
                    "feature": feature,
                    "builder_family": "volume_profile",
                    "builder_name": "build_volume_profile_features",
                    "builder_source": "volume_profile_fixed_range",
                }
            )
            continue

        if feature in feature_parts["candle_feature_cols"]:
            builder_name = (
                "build_latest_candle_pattern_feature_dict"
                if feature in candle_pattern_cols
                else "build_latest_candle_derived_feature_dict_fast"
            )
            records.append(
                {
                    "feature": feature,
                    "builder_family": "candle",
                    "builder_name": builder_name,
                    "builder_source": None,
                }
            )
            continue

        cfg = indicator_config_map.get(feature)
        if cfg is not None:
            records.append(
                {
                    "feature": feature,
                    "builder_family": "indicator",
                    "builder_name": str(cfg["indicator"]),
                    "builder_source": str(cfg["json_path"]),
                }
            )
            continue

        records.append(
            {
                "feature": feature,
                "builder_family": "unknown",
                "builder_name": "unknown",
                "builder_source": None,
            }
        )

    return pd.DataFrame.from_records(records)


def _build_single_feature_prediction_impact_report(
        *,
        candidate_label,
        reference_label,
        candidate_matrix,
        reference_matrix,
        candidate_pred,
        reference_pred,
        audit_df,
        feature_columns,
        feature_group_by_name,
        feature_builder_frame,
        model,
):
    row_count, feature_count = candidate_matrix.shape
    if feature_count != len(feature_columns):
        raise ValueError("Feature count mismatch for prediction impact audit.")

    builder_meta = (
        feature_builder_frame.set_index("feature")
        .reindex(feature_columns)
        .reset_index(drop=False)
    )
    feature_names = np.asarray(feature_columns, dtype=object)
    feature_groups = np.asarray(
        [feature_group_by_name.get(col, "unknown") for col in feature_columns],
        dtype=object,
    )
    builder_families = (
        builder_meta["builder_family"].fillna("unknown").to_numpy(dtype=object)
    )
    builder_names = (
        builder_meta["builder_name"].fillna("unknown").to_numpy(dtype=object)
    )
    builder_sources = builder_meta["builder_source"].to_numpy(dtype=object)

    base_abs_gap = np.abs(candidate_pred - reference_pred)
    base_signal_mismatch = (candidate_pred >= 0.5) != (reference_pred >= 0.5)
    base_drift_mask = base_abs_gap > PREDICTION_DIFF_TOL

    fixed_pred_matrix = np.empty((row_count, feature_count), dtype=np.float64)
    proba_shift_matrix = np.empty((row_count, feature_count), dtype=np.float64)
    gap_reduction_matrix = np.empty((row_count, feature_count), dtype=np.float64)
    working_matrix = candidate_matrix.copy()

    feature_rows = []
    for feature_idx, feature_name in enumerate(feature_columns):
        candidate_col = candidate_matrix[:, feature_idx]
        reference_col = reference_matrix[:, feature_idx]
        if _column_values_match(candidate_col, reference_col):
            fixed_pred = candidate_pred
        else:
            working_matrix[:, feature_idx] = reference_col
            fixed_pred = np.asarray(
                model.predict(working_matrix), dtype=np.float64
            ).reshape(-1)
            if fixed_pred.shape[0] != row_count:
                raise ValueError(
                    f"Prediction length mismatch for feature impact audit: {fixed_pred.shape[0]} != {row_count}"
                )
            working_matrix[:, feature_idx] = candidate_col

        fixed_pred_matrix[:, feature_idx] = fixed_pred
        proba_shift = candidate_pred - fixed_pred
        fixed_abs_gap = np.abs(fixed_pred - reference_pred)
        gap_reduction = base_abs_gap - fixed_abs_gap
        proba_shift_matrix[:, feature_idx] = proba_shift
        gap_reduction_matrix[:, feature_idx] = gap_reduction

        abs_proba_shift = np.abs(proba_shift)
        proba_shift_mask = abs_proba_shift > PREDICTION_DIFF_TOL
        proba_shift_values = abs_proba_shift[proba_shift_mask]
        fixed_signal_mismatch = (fixed_pred >= 0.5) != (reference_pred >= 0.5)
        fixed_drift_mask = fixed_abs_gap > PREDICTION_DIFF_TOL
        helpful_gap_mask = gap_reduction > PREDICTION_DIFF_TOL
        harmful_gap_mask = gap_reduction < -PREDICTION_DIFF_TOL
        drift_gap_values = gap_reduction[base_drift_mask]
        helpful_gap_values = gap_reduction[helpful_gap_mask]
        if base_drift_mask.any():
            drift_base_gap_values = base_abs_gap[base_drift_mask]
            drift_gap_reduction_ratio = np.divide(
                drift_gap_values,
                drift_base_gap_values,
                out=np.zeros(drift_gap_values.shape[0], dtype=np.float64),
                where=drift_base_gap_values > PREDICTION_DIFF_TOL,
            )
            mean_abs_proba_gap_reduction_on_drift_rows_if_fixed = float(
                np.mean(drift_gap_values)
            )
            max_abs_proba_gap_reduction_on_drift_rows_if_fixed = float(
                np.max(drift_gap_values)
            )
            mean_gap_reduction_ratio_on_drift_rows_if_fixed = float(
                np.mean(drift_gap_reduction_ratio)
            )
            max_gap_reduction_ratio_on_drift_rows_if_fixed = float(
                np.max(drift_gap_reduction_ratio)
            )
        else:
            mean_abs_proba_gap_reduction_on_drift_rows_if_fixed = 0.0
            max_abs_proba_gap_reduction_on_drift_rows_if_fixed = 0.0
            mean_gap_reduction_ratio_on_drift_rows_if_fixed = 0.0
            max_gap_reduction_ratio_on_drift_rows_if_fixed = 0.0

        mean_abs_proba_gap_reduction_on_helped_rows_if_fixed = (
            float(np.mean(helpful_gap_values)) if helpful_gap_mask.any() else 0.0
        )
        best_gap_reduction_idx = int(np.argmax(gap_reduction))

        feature_rows.append(
            {
                "feature": feature_name,
                "group": feature_groups[feature_idx],
                "builder_family": builder_families[feature_idx],
                "builder_name": builder_names[feature_idx],
                "builder_source": builder_sources[feature_idx],
                "rows_pred_shift_gt_tol_if_fixed": int(proba_shift_mask.sum()),
                "mean_abs_proba_shift_on_shift_rows_if_fixed": (
                    float(np.mean(proba_shift_values))
                    if proba_shift_mask.any()
                    else 0.0
                ),
                "mean_abs_proba_shift_if_fixed": float(np.mean(abs_proba_shift)),
                "max_abs_proba_shift_if_fixed": float(np.max(abs_proba_shift)),
                "mean_signed_proba_shift_if_fixed": float(np.mean(proba_shift)),
                "net_abs_proba_gap_reduction_if_fixed": float(np.sum(gap_reduction)),
                "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed": (
                    mean_abs_proba_gap_reduction_on_drift_rows_if_fixed
                ),
                "max_abs_proba_gap_reduction_on_drift_rows_if_fixed": (
                    max_abs_proba_gap_reduction_on_drift_rows_if_fixed
                ),
                "mean_abs_proba_gap_reduction_on_helped_rows_if_fixed": (
                    mean_abs_proba_gap_reduction_on_helped_rows_if_fixed
                ),
                "mean_gap_reduction_ratio_on_drift_rows_if_fixed": (
                    mean_gap_reduction_ratio_on_drift_rows_if_fixed
                ),
                "max_gap_reduction_ratio_on_drift_rows_if_fixed": (
                    max_gap_reduction_ratio_on_drift_rows_if_fixed
                ),
                "mean_abs_proba_gap_reduction_if_fixed": float(np.mean(gap_reduction)),
                "max_abs_proba_gap_reduction_if_fixed": float(np.max(gap_reduction)),
                "rows_abs_proba_gap_reduced_if_fixed": int(helpful_gap_mask.sum()),
                "rows_abs_proba_gap_worsened_if_fixed": int(harmful_gap_mask.sum()),
                "rows_proba_diff_gt_tol_resolved_if_fixed": int(
                    (base_drift_mask & ~fixed_drift_mask).sum()
                ),
                "rows_proba_diff_gt_tol_introduced_if_fixed": int(
                    (~base_drift_mask & fixed_drift_mask).sum()
                ),
                "rows_signal_mismatch_resolved_if_fixed": int(
                    (base_signal_mismatch & ~fixed_signal_mismatch).sum()
                ),
                "rows_signal_mismatch_introduced_if_fixed": int(
                    (~base_signal_mismatch & fixed_signal_mismatch).sum()
                ),
                "worst_gap_reduction_opened": audit_df["Opened"].iloc[
                    best_gap_reduction_idx
                ],
                f"worst_gap_reduction_{candidate_label}_value": float(
                    candidate_col[best_gap_reduction_idx]
                ),
                f"worst_gap_reduction_{reference_label}_value": float(
                    reference_col[best_gap_reduction_idx]
                ),
                f"worst_gap_reduction_{candidate_label}_proba_up": float(
                    candidate_pred[best_gap_reduction_idx]
                ),
                f"worst_gap_reduction_{reference_label}_proba_up": float(
                    reference_pred[best_gap_reduction_idx]
                ),
                "worst_gap_reduction_proba_up_if_fixed": float(
                    fixed_pred[best_gap_reduction_idx]
                ),
            }
        )

    feature_summary_df = (
        pd.DataFrame.from_records(feature_rows)
        .sort_values(
            [
                "rows_signal_mismatch_resolved_if_fixed",
                "rows_proba_diff_gt_tol_resolved_if_fixed",
                "net_abs_proba_gap_reduction_if_fixed",
                "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
                "mean_gap_reduction_ratio_on_drift_rows_if_fixed",
                "rows_abs_proba_gap_reduced_if_fixed",
                "rows_signal_mismatch_introduced_if_fixed",
                "mean_abs_proba_shift_if_fixed",
                "max_abs_proba_shift_if_fixed",
            ],
            ascending=[False, False, False, False, False, False, True, False, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    group_summary_df = (
        feature_summary_df.groupby("group", dropna=False)
        .agg(
            feature_count=("feature", "count"),
            max_mean_abs_proba_shift_if_fixed=("mean_abs_proba_shift_if_fixed", "max"),
            mean_mean_abs_proba_shift_if_fixed=(
                "mean_abs_proba_shift_if_fixed",
                "mean",
            ),
            total_net_abs_proba_gap_reduction_if_fixed=(
                "net_abs_proba_gap_reduction_if_fixed",
                "sum",
            ),
            max_mean_abs_proba_gap_reduction_if_fixed=(
                "mean_abs_proba_gap_reduction_if_fixed",
                "max",
            ),
            mean_mean_abs_proba_gap_reduction_if_fixed=(
                "mean_abs_proba_gap_reduction_if_fixed",
                "mean",
            ),
            max_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed=(
                "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
                "max",
            ),
            mean_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed=(
                "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
                "mean",
            ),
            total_rows_abs_proba_gap_reduced_if_fixed=(
                "rows_abs_proba_gap_reduced_if_fixed",
                "sum",
            ),
            total_rows_abs_proba_gap_worsened_if_fixed=(
                "rows_abs_proba_gap_worsened_if_fixed",
                "sum",
            ),
            total_rows_signal_mismatch_resolved_if_fixed=(
                "rows_signal_mismatch_resolved_if_fixed",
                "sum",
            ),
            total_rows_proba_diff_gt_tol_resolved_if_fixed=(
                "rows_proba_diff_gt_tol_resolved_if_fixed",
                "sum",
            ),
            total_rows_signal_mismatch_introduced_if_fixed=(
                "rows_signal_mismatch_introduced_if_fixed",
                "sum",
            ),
        )
        .sort_values(
            [
                "total_rows_signal_mismatch_resolved_if_fixed",
                "total_rows_proba_diff_gt_tol_resolved_if_fixed",
                "total_net_abs_proba_gap_reduction_if_fixed",
                "max_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
                "max_mean_abs_proba_shift_if_fixed",
            ],
            ascending=[False, False, False, False, False],
            kind="stable",
        )
        .reset_index()
    )

    builder_summary_df = (
        feature_summary_df.groupby(["builder_family", "builder_name"], dropna=False)
        .agg(
            feature_count=("feature", "count"),
            max_mean_abs_proba_shift_if_fixed=("mean_abs_proba_shift_if_fixed", "max"),
            mean_mean_abs_proba_shift_if_fixed=(
                "mean_abs_proba_shift_if_fixed",
                "mean",
            ),
            total_net_abs_proba_gap_reduction_if_fixed=(
                "net_abs_proba_gap_reduction_if_fixed",
                "sum",
            ),
            max_mean_abs_proba_gap_reduction_if_fixed=(
                "mean_abs_proba_gap_reduction_if_fixed",
                "max",
            ),
            mean_mean_abs_proba_gap_reduction_if_fixed=(
                "mean_abs_proba_gap_reduction_if_fixed",
                "mean",
            ),
            max_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed=(
                "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
                "max",
            ),
            mean_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed=(
                "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
                "mean",
            ),
            total_rows_abs_proba_gap_reduced_if_fixed=(
                "rows_abs_proba_gap_reduced_if_fixed",
                "sum",
            ),
            total_rows_abs_proba_gap_worsened_if_fixed=(
                "rows_abs_proba_gap_worsened_if_fixed",
                "sum",
            ),
            total_rows_signal_mismatch_resolved_if_fixed=(
                "rows_signal_mismatch_resolved_if_fixed",
                "sum",
            ),
            total_rows_proba_diff_gt_tol_resolved_if_fixed=(
                "rows_proba_diff_gt_tol_resolved_if_fixed",
                "sum",
            ),
            total_rows_signal_mismatch_introduced_if_fixed=(
                "rows_signal_mismatch_introduced_if_fixed",
                "sum",
            ),
        )
        .sort_values(
            [
                "total_rows_signal_mismatch_resolved_if_fixed",
                "total_rows_proba_diff_gt_tol_resolved_if_fixed",
                "total_net_abs_proba_gap_reduction_if_fixed",
                "max_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
                "max_mean_abs_proba_shift_if_fixed",
            ],
            ascending=[False, False, False, False, False],
            kind="stable",
        )
        .reset_index()
    )

    row_best_idx = _safe_argmax_axis1(gap_reduction_matrix)
    row_best_gap_reduction = gap_reduction_matrix[np.arange(row_count), row_best_idx]
    row_has_helpful_single_feature_fix = row_best_gap_reduction > PREDICTION_DIFF_TOL
    row_summary_df = pd.DataFrame(
        {
            "top_prediction_impact_feature": np.where(
                row_has_helpful_single_feature_fix,
                feature_names[row_best_idx],
                None,
            ),
            "top_prediction_impact_group": np.where(
                row_has_helpful_single_feature_fix,
                feature_groups[row_best_idx],
                None,
            ),
            "top_prediction_impact_builder_family": np.where(
                row_has_helpful_single_feature_fix,
                builder_families[row_best_idx],
                None,
            ),
            "top_prediction_impact_builder_name": np.where(
                row_has_helpful_single_feature_fix,
                builder_names[row_best_idx],
                None,
            ),
            "top_prediction_impact_abs_proba_gap_reduction_if_fixed": np.where(
                row_has_helpful_single_feature_fix,
                row_best_gap_reduction,
                0.0,
            ),
            "top_prediction_impact_abs_proba_shift_if_fixed": np.where(
                row_has_helpful_single_feature_fix,
                np.abs(proba_shift_matrix[np.arange(row_count), row_best_idx]),
                np.nan,
            ),
            "top_prediction_impact_signed_proba_shift_if_fixed": np.where(
                row_has_helpful_single_feature_fix,
                proba_shift_matrix[np.arange(row_count), row_best_idx],
                np.nan,
            ),
            "top_prediction_impact_signal_mismatch_resolved_if_fixed": np.where(
                row_has_helpful_single_feature_fix,
                (
                        base_signal_mismatch
                        & ~(
                        (fixed_pred_matrix[np.arange(row_count), row_best_idx] >= 0.5)
                        != (reference_pred >= 0.5)
                )
                ),
                False,
            ),
            f"top_prediction_impact_{candidate_label}_value": np.where(
                row_has_helpful_single_feature_fix,
                candidate_matrix[np.arange(row_count), row_best_idx],
                np.nan,
            ),
            f"top_prediction_impact_{reference_label}_value": np.where(
                row_has_helpful_single_feature_fix,
                reference_matrix[np.arange(row_count), row_best_idx],
                np.nan,
            ),
            "top_prediction_impact_proba_up_if_fixed": np.where(
                row_has_helpful_single_feature_fix,
                fixed_pred_matrix[np.arange(row_count), row_best_idx],
                np.nan,
            ),
        }
    )

    top_feature = None if feature_summary_df.empty else feature_summary_df.iloc[0]
    summary = pd.Series(
        {
            "rows_with_helpful_single_feature_fix": int(
                row_has_helpful_single_feature_fix.sum()
            ),
            "top_prediction_impact_feature": (
                None if top_feature is None else top_feature["feature"]
            ),
            "top_prediction_impact_group": (
                None if top_feature is None else top_feature["group"]
            ),
            "top_prediction_impact_builder_family": (
                None if top_feature is None else top_feature["builder_family"]
            ),
            "top_prediction_impact_builder_name": (
                None if top_feature is None else top_feature["builder_name"]
            ),
            "max_mean_abs_proba_shift_if_fixed": (
                0.0
                if feature_summary_df.empty
                else float(feature_summary_df["mean_abs_proba_shift_if_fixed"].max())
            ),
            "max_mean_abs_proba_gap_reduction_if_fixed": (
                0.0
                if feature_summary_df.empty
                else float(
                    feature_summary_df["mean_abs_proba_gap_reduction_if_fixed"].max()
                )
            ),
            "max_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed": (
                0.0
                if feature_summary_df.empty
                else float(
                    feature_summary_df[
                        "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed"
                    ].max()
                )
            ),
            "max_net_abs_proba_gap_reduction_if_fixed": (
                0.0
                if feature_summary_df.empty
                else float(
                    feature_summary_df["net_abs_proba_gap_reduction_if_fixed"].max()
                )
            ),
            "max_rows_signal_mismatch_resolved_if_fixed": (
                0
                if feature_summary_df.empty
                else int(
                    feature_summary_df["rows_signal_mismatch_resolved_if_fixed"].max()
                )
            ),
            "max_rows_proba_diff_gt_tol_resolved_if_fixed": (
                0
                if feature_summary_df.empty
                else int(
                    feature_summary_df[
                        "rows_proba_diff_gt_tol_resolved_if_fixed"
                    ].max()
                )
            ),
        }
    )

    return {
        "summary": summary,
        "row_summary_df": row_summary_df,
        "feature_summary_df": feature_summary_df,
        "group_summary_df": group_summary_df,
        "builder_summary_df": builder_summary_df,
        "fixed_pred_matrix": fixed_pred_matrix,
        "proba_shift_matrix": proba_shift_matrix,
        "gap_reduction_matrix": gap_reduction_matrix,
    }


def resolve_recent_history_tail_window(
        *,
        parquet_path,
        audit_end,
        tail_fraction,
):
    tail_fraction = float(tail_fraction)
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError("tail_fraction must be in (0, 1].")

    opened = _load_opened_series(parquet_path)
    opened = pd.to_datetime(opened)
    audit_end = pd.Timestamp(audit_end)
    eligible = opened.loc[opened <= audit_end]
    if eligible.empty:
        raise ValueError("No parquet rows are available at or before audit_end.")

    total_rows = len(eligible)
    keep_rows = int(np.ceil(total_rows * tail_fraction))
    keep_rows = max(1, min(total_rows, keep_rows))
    tail_start = pd.Timestamp(eligible.iloc[total_rows - keep_rows])
    return tail_start, keep_rows, total_rows


def load_modeling_raw_history_frame(
        *,
        parquet_path,
        audit_end,
        history_start=None,
):
    filters = []
    if history_start is not None:
        filters.append(
            (
                "Opened",
                ">=",
                _naive_utc_timestamp(history_start),
            )
        )
    filters.append(
        (
            "Opened",
            "<=",
            _naive_utc_timestamp(audit_end),
        )
    )
    frame = pd.read_parquet(
        parquet_path,
        columns=["Opened", *RAW_OHLCV_COLS],
        filters=filters,
    )
    frame["Opened"] = _ensure_utc_opened(frame["Opened"])
    frame = (
        frame.sort_values("Opened")
        .drop_duplicates(subset=["Opened"])
        .reset_index(drop=True)
    )
    return frame


def build_current_recomputed_feature_history(
        *,
        raw_history_df,
        feature_columns,
):
    feature_parts = split_feature_subset(
        feature_columns,
        source_label="audit feature columns",
    )
    recomputed_feature_columns = _audited_feature_columns(feature_columns)
    feature_frame = raw_history_df.loc[:, ["Opened", *RAW_OHLCV_COLS]].copy()

    configured_rules = resolve_streak_interval_to_rule(
        MODELING_DATASET_SETTINGS.get("candle_streak_intervals", {})
    )
    if feature_parts["streak_intervals"]:
        missing_streak_intervals = [
            label
            for label in feature_parts["streak_intervals"]
            if label not in configured_rules
        ]
        if missing_streak_intervals:
            raise ValueError(
                "Missing configured candle streak intervals for recompute: "
                f"{missing_streak_intervals}"
            )
        feature_frame = add_candle_streak_features(
            feature_frame,
            interval_to_rule={
                label: configured_rules[label]
                for label in feature_parts["streak_intervals"]
            },
        )

    if feature_parts["candle_feature_cols"]:
        feature_frame = add_candle_derived_features(
            feature_frame,
            feature_cols=feature_parts["candle_feature_cols"],
        )

    if feature_parts["session_feature_cols"]:
        feature_frame = add_session_open_features(
            feature_frame,
            feature_cols=feature_parts["session_feature_cols"],
        )

    if feature_parts["realized_volatility_feature_cols"]:
        feature_frame = add_realized_volatility_features(feature_frame)

    vp_cfg = normalize_volume_profile_config(
        MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")
    )
    if feature_parts["volume_profile_feature_cols"]:
        if not vp_cfg["enabled"]:
            raise ValueError(
                "Volume profile features requested but disabled in modeling config."
            )
        vp_features_df, _vp_state = build_volume_profile_features(feature_frame, vp_cfg)
        vp_feature_frame = pd.DataFrame(
            {
                feature_col: vp_features_df[feature_col].to_numpy(
                    dtype=np.float64, copy=False
                )
                for feature_col in feature_parts["volume_profile_feature_cols"]
            },
            index=feature_frame.index,
        )
        feature_frame = concat_feature_frame(
            feature_frame,
            vp_feature_frame,
            context="Current recompute volume profile features",
        )

    if feature_parts["indicator_feature_cols"]:
        indicator_config_map = _fit_indicator_config_map(recomputed_feature_columns)
        indicator_configs = [
            indicator_config_map[feature]
            for feature in feature_parts["indicator_feature_cols"]
            if feature in indicator_config_map
        ]
        missing_indicator_cols = [
            feature
            for feature in feature_parts["indicator_feature_cols"]
            if feature not in indicator_config_map
        ]
        if missing_indicator_cols:
            raise ValueError(
                "Missing indicator configs for current recompute features: "
                f"{missing_indicator_cols[:10]}"
            )
        ohlcv_np = feature_frame[list(OHLCV_COLS)].to_numpy(dtype=np.float64, copy=True)
        feature_frame = add_indicator_values(feature_frame, ohlcv_np, indicator_configs)

    keep_cols = ["Opened", *RAW_OHLCV_COLS]
    keep_cols.extend(col for col in recomputed_feature_columns if col not in keep_cols)
    return feature_frame.loc[:, keep_cols].copy()


def align_feature_frame_to_audit_rows(
        *,
        audit_df,
        feature_frame,
        feature_columns,
):
    aligned = audit_df.loc[:, ["Opened"]].merge(
        feature_frame.loc[:, ["Opened", *feature_columns]],
        on="Opened",
        how="left",
        sort=False,
    )
    if feature_columns and aligned[feature_columns].isna().all(axis=1).any():
        missing_rows = aligned.loc[
            aligned[feature_columns].isna().all(axis=1), "Opened"
        ]
        raise RuntimeError(
            "Current recompute frame is missing rows for audit timestamps. "
            f"First missing={missing_rows.iloc[0]!s}"
        )
    return aligned


def _compare_policy_decisions(
        *,
        predictor,
        candidate_proba,
        reference_proba,
        candidate_label,
        reference_label,
):
    bankroll = float(LIVE_INITIAL_BANKROLL_USDC)
    rows = []
    for candidate_prob, reference_prob in zip(
            candidate_proba, reference_proba, strict=True
    ):
        candidate_decision = predictor.evaluate_policy_decision(
            float(candidate_prob),
            bankroll=bankroll,
        )
        reference_decision = predictor.evaluate_policy_decision(
            float(reference_prob),
            bankroll=bankroll,
        )

        candidate_stake = float(candidate_decision.get("bet_usdc", 0.0) or 0.0)
        reference_stake = float(reference_decision.get("bet_usdc", 0.0) or 0.0)
        candidate_trade = int(candidate_stake > 0.0)
        reference_trade = int(reference_stake > 0.0)
        candidate_side = str(candidate_decision.get("trade_side", "none"))
        reference_side = str(reference_decision.get("trade_side", "none"))
        candidate_reason = str(
            candidate_decision.get(
                "final_reason",
                candidate_decision.get("reason", ""),
            )
        )
        reference_reason = str(
            reference_decision.get(
                "final_reason",
                reference_decision.get("reason", ""),
            )
        )
        rows.append(
            {
                f"{candidate_label}_policy_side": candidate_side,
                f"{reference_label}_policy_side": reference_side,
                f"{candidate_label}_policy_reason": candidate_reason,
                f"{reference_label}_policy_reason": reference_reason,
                f"{candidate_label}_stake_usdc": candidate_stake,
                f"{reference_label}_stake_usdc": reference_stake,
                f"{candidate_label}_trade_flag": candidate_trade,
                f"{reference_label}_trade_flag": reference_trade,
                "policy_side_mismatch": int(candidate_side != reference_side),
                "policy_reason_mismatch": int(candidate_reason != reference_reason),
                "policy_trade_flag_mismatch": int(candidate_trade != reference_trade),
                "policy_stake_abs_diff": abs(candidate_stake - reference_stake),
                "policy_decision_mismatch": int(
                    candidate_side != reference_side
                    or candidate_reason != reference_reason
                    or candidate_trade != reference_trade
                    or abs(candidate_stake - reference_stake) > 1e-9
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_matrix_comparison_report(
        *,
        candidate_label,
        reference_label,
        candidate_matrix,
        reference_matrix,
        audit_df,
        feature_columns,
        feature_group_by_name,
        feature_builder_frame,
        model,
        feature_importance_df=None,
        policy_predictor=None,
):
    feature_names = np.asarray(feature_columns, dtype=object)
    builder_meta = (
        feature_builder_frame.set_index("feature")
        .reindex(feature_columns)
        .reset_index(drop=False)
    )
    builder_groups = (
        builder_meta["builder_family"].fillna("unknown").to_numpy(dtype=object)
    )
    builder_names = (
        builder_meta["builder_name"].fillna("unknown").to_numpy(dtype=object)
    )
    feature_groups = np.asarray(
        [feature_group_by_name.get(col, "unknown") for col in feature_columns],
        dtype=object,
    )
    candidate_nonfinite_mask = ~np.isfinite(candidate_matrix)
    reference_nonfinite_mask = ~np.isfinite(reference_matrix)
    finite_pair_mask = np.isfinite(candidate_matrix) & np.isfinite(reference_matrix)
    finite_status_mismatch_mask = np.logical_xor(
        candidate_nonfinite_mask, reference_nonfinite_mask
    )
    diff_matrix = np.abs(candidate_matrix - reference_matrix)
    diff_matrix[~finite_pair_mask] = np.nan
    rel_diff_matrix = _safe_relative_diff(candidate_matrix, reference_matrix)

    candidate_pred = model.predict(candidate_matrix).astype(np.float64, copy=False)
    reference_pred = model.predict(reference_matrix).astype(np.float64, copy=False)
    pred_abs_diff = np.abs(candidate_pred - reference_pred)
    prediction_impact_report = _build_single_feature_prediction_impact_report(
        candidate_label=candidate_label,
        reference_label=reference_label,
        candidate_matrix=candidate_matrix,
        reference_matrix=reference_matrix,
        candidate_pred=candidate_pred,
        reference_pred=reference_pred,
        audit_df=audit_df,
        feature_columns=feature_columns,
        feature_group_by_name=feature_group_by_name,
        feature_builder_frame=feature_builder_frame,
        model=model,
    )

    row_worst_idx = _safe_argmax_axis1(diff_matrix)
    row_has_finite = np.isfinite(diff_matrix).any(axis=1)
    row_worst_feature = np.where(row_has_finite, feature_names[row_worst_idx], None)
    row_worst_group = np.where(row_has_finite, feature_groups[row_worst_idx], None)
    row_worst_builder_family = np.where(
        row_has_finite, builder_groups[row_worst_idx], None
    )
    row_worst_builder_name = np.where(
        row_has_finite, builder_names[row_worst_idx], None
    )
    row_worst_candidate = np.where(
        row_has_finite,
        candidate_matrix[np.arange(len(audit_df)), row_worst_idx],
        np.nan,
    )
    row_worst_reference = np.where(
        row_has_finite,
        reference_matrix[np.arange(len(audit_df)), row_worst_idx],
        np.nan,
    )

    step_summary_df = pd.DataFrame(
        {
            "Opened": audit_df["Opened"],
            "feature_max_abs_diff": _safe_nanmax_axis1(diff_matrix),
            "feature_mean_abs_diff": _safe_rowwise_mean(diff_matrix),
            "feature_max_rel_diff": _safe_nanmax_axis1(rel_diff_matrix),
            "feature_mean_rel_diff": _safe_rowwise_mean(rel_diff_matrix),
            "worst_feature": row_worst_feature,
            "worst_group": row_worst_group,
            "worst_builder_family": row_worst_builder_family,
            "worst_builder_name": row_worst_builder_name,
            f"worst_feature_{candidate_label}_value": row_worst_candidate,
            f"worst_feature_{reference_label}_value": row_worst_reference,
            f"{candidate_label}_nonfinite_count": candidate_nonfinite_mask.sum(
                axis=1
            ).astype(np.int32),
            f"{reference_label}_nonfinite_count": reference_nonfinite_mask.sum(
                axis=1
            ).astype(np.int32),
            "finite_status_mismatch_count": finite_status_mismatch_mask.sum(
                axis=1
            ).astype(np.int32),
            f"{candidate_label}_proba_up": candidate_pred,
            f"{reference_label}_proba_up": reference_pred,
            "proba_up_abs_diff": pred_abs_diff,
            "signal_mismatch": (
                    (candidate_pred >= 0.5) != (reference_pred >= 0.5)
            ).astype(np.int8),
        }
    )
    step_summary_df = pd.concat(
        [
            step_summary_df.reset_index(drop=True),
            prediction_impact_report["row_summary_df"].reset_index(drop=True),
        ],
        axis=1,
        copy=False,
    )

    if policy_predictor is not None:
        policy_df = _compare_policy_decisions(
            predictor=policy_predictor,
            candidate_proba=candidate_pred,
            reference_proba=reference_pred,
            candidate_label=candidate_label,
            reference_label=reference_label,
        )
        step_summary_df = pd.concat(
            [step_summary_df.reset_index(drop=True), policy_df.reset_index(drop=True)],
            axis=1,
            copy=False,
        )
        step_summary_df["business_decision_mismatch"] = (
                (step_summary_df["signal_mismatch"] > 0)
                | (step_summary_df["policy_side_mismatch"] > 0)
                | (step_summary_df["policy_reason_mismatch"] > 0)
                | (step_summary_df["policy_trade_flag_mismatch"] > 0)
        ).astype(np.int8)
        step_summary_df["stake_only_policy_mismatch"] = (
                (step_summary_df["policy_decision_mismatch"] > 0)
                & (step_summary_df["business_decision_mismatch"] == 0)
        ).astype(np.int8)
    else:
        step_summary_df["business_decision_mismatch"] = step_summary_df[
            "signal_mismatch"
        ].astype(np.int8)
        step_summary_df["stake_only_policy_mismatch"] = np.zeros(
            len(step_summary_df), dtype=np.int8
        )

    col_has_finite = np.isfinite(diff_matrix).any(axis=0)
    col_worst_idx = _safe_argmax_axis0(diff_matrix)
    feature_summary_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "group": [
                feature_group_by_name.get(col, "unknown") for col in feature_columns
            ],
            "max_abs_diff": _safe_nanmax_axis0(diff_matrix),
            "mean_abs_diff": _safe_colwise_mean(diff_matrix),
            "max_rel_diff": _safe_nanmax_axis0(rel_diff_matrix),
            "mean_rel_diff": _safe_colwise_mean(rel_diff_matrix),
            "rmse_abs_diff": _safe_colwise_rmse(diff_matrix),
            "finite_diff_count": np.isfinite(diff_matrix).sum(axis=0).astype(np.int32),
            f"{candidate_label}_nonfinite_count": candidate_nonfinite_mask.sum(
                axis=0
            ).astype(np.int32),
            f"{reference_label}_nonfinite_count": reference_nonfinite_mask.sum(
                axis=0
            ).astype(np.int32),
            "finite_status_mismatch_count": finite_status_mismatch_mask.sum(
                axis=0
            ).astype(np.int32),
            "worst_opened": np.where(
                col_has_finite,
                audit_df["Opened"].iloc[col_worst_idx].to_numpy(),
                pd.NaT,
            ),
            f"worst_{candidate_label}_value": np.where(
                col_has_finite,
                candidate_matrix[col_worst_idx, np.arange(len(feature_columns))],
                np.nan,
            ),
            f"worst_{reference_label}_value": np.where(
                col_has_finite,
                reference_matrix[col_worst_idx, np.arange(len(feature_columns))],
                np.nan,
            ),
        }
    )
    feature_impact_summary_df = prediction_impact_report["feature_summary_df"].drop(
        columns=["group", "builder_family", "builder_name", "builder_source"],
        errors="ignore",
    )
    feature_summary_df = (
        feature_summary_df.merge(
            feature_builder_frame,
            on="feature",
            how="left",
            sort=False,
        )
        .merge(
            feature_impact_summary_df,
            on="feature",
            how="left",
            sort=False,
        )
        .sort_values(
            [
                "rows_signal_mismatch_resolved_if_fixed",
                "rows_proba_diff_gt_tol_resolved_if_fixed",
                "net_abs_proba_gap_reduction_if_fixed",
                "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
                "mean_gap_reduction_ratio_on_drift_rows_if_fixed",
                "mean_abs_proba_shift_if_fixed",
                "finite_status_mismatch_count",
                "max_rel_diff",
                "mean_rel_diff",
                "max_abs_diff",
                "mean_abs_diff",
                "rmse_abs_diff",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            kind="stable",
        )
    )
    if feature_importance_df is not None and not feature_importance_df.empty:
        feature_summary_df = feature_summary_df.merge(
            feature_importance_df,
            on="feature",
            how="left",
            sort=False,
        )
    if "importance_gain" not in feature_summary_df.columns:
        feature_summary_df["importance_gain"] = 0.0
    if "importance_split" not in feature_summary_df.columns:
        feature_summary_df["importance_split"] = 0.0
    feature_summary_df["importance_gain"] = pd.to_numeric(
        feature_summary_df["importance_gain"], errors="coerce"
    ).fillna(0.0)
    feature_summary_df["importance_split"] = pd.to_numeric(
        feature_summary_df["importance_split"], errors="coerce"
    ).fillna(0.0)

    group_summary_df = (
        feature_summary_df.groupby("group", dropna=False)
        .agg(
            feature_count=("feature", "count"),
            max_abs_diff=("max_abs_diff", "max"),
            mean_abs_diff=("mean_abs_diff", "mean"),
            max_rel_diff=("max_rel_diff", "max"),
            mean_rel_diff=("mean_rel_diff", "mean"),
            rmse_abs_diff=("rmse_abs_diff", "mean"),
            total_candidate_nonfinite_count=(
                f"{candidate_label}_nonfinite_count",
                "sum",
            ),
            total_reference_nonfinite_count=(
                f"{reference_label}_nonfinite_count",
                "sum",
            ),
            total_finite_status_mismatch_count=("finite_status_mismatch_count", "sum"),
        )
        .reset_index()
    )
    group_prediction_impact_df = prediction_impact_report["group_summary_df"].drop(
        columns=["feature_count"],
        errors="ignore",
    )
    group_summary_df = group_summary_df.merge(
        group_prediction_impact_df,
        on="group",
        how="left",
        sort=False,
    ).sort_values(
        [
            "total_rows_signal_mismatch_resolved_if_fixed",
            "total_rows_proba_diff_gt_tol_resolved_if_fixed",
            "total_net_abs_proba_gap_reduction_if_fixed",
            "max_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
            "max_mean_abs_proba_shift_if_fixed",
            "max_rel_diff",
            "mean_rel_diff",
            "max_abs_diff",
            "mean_abs_diff",
        ],
        ascending=[False, False, False, False, False, False, False, False, False],
        kind="stable",
    )

    builder_summary_df = (
        feature_summary_df.groupby(["builder_family", "builder_name"], dropna=False)
        .agg(
            feature_count=("feature", "count"),
            max_abs_diff=("max_abs_diff", "max"),
            mean_abs_diff=("mean_abs_diff", "mean"),
            max_rel_diff=("max_rel_diff", "max"),
            mean_rel_diff=("mean_rel_diff", "mean"),
            rmse_abs_diff=("rmse_abs_diff", "mean"),
            total_finite_status_mismatch_count=("finite_status_mismatch_count", "sum"),
        )
        .reset_index()
    )
    builder_prediction_impact_df = prediction_impact_report["builder_summary_df"].drop(
        columns=["feature_count"],
        errors="ignore",
    )
    builder_summary_df = builder_summary_df.merge(
        builder_prediction_impact_df,
        on=["builder_family", "builder_name"],
        how="left",
        sort=False,
    ).sort_values(
        [
            "total_rows_signal_mismatch_resolved_if_fixed",
            "total_rows_proba_diff_gt_tol_resolved_if_fixed",
            "total_net_abs_proba_gap_reduction_if_fixed",
            "max_mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
            "max_mean_abs_proba_shift_if_fixed",
            "max_rel_diff",
            "mean_rel_diff",
            "max_abs_diff",
            "mean_abs_diff",
        ],
        ascending=[False, False, False, False, False, False, False, False, False],
        kind="stable",
    )

    summary_payload = {
        "decision_row_count": len(audit_df),
        "feature_count": len(feature_columns),
        f"rows_with_{candidate_label}_nonfinite": int(
            (step_summary_df[f"{candidate_label}_nonfinite_count"] > 0).sum()
        ),
        f"rows_with_{reference_label}_nonfinite": int(
            (step_summary_df[f"{reference_label}_nonfinite_count"] > 0).sum()
        ),
        "rows_with_finite_status_mismatch": int(
            (step_summary_df["finite_status_mismatch_count"] > 0).sum()
        ),
        "rows_with_signal_mismatch": int(step_summary_df["signal_mismatch"].sum()),
        "signal_mismatch_rate": _summary_rate(
            int(step_summary_df["signal_mismatch"].sum()),
            len(audit_df),
        ),
        "rows_with_business_decision_mismatch": int(
            step_summary_df["business_decision_mismatch"].sum()
        ),
        "business_decision_mismatch_rate": _summary_rate(
            int(step_summary_df["business_decision_mismatch"].sum()),
            len(audit_df),
        ),
        "max_feature_abs_diff": float(step_summary_df["feature_max_abs_diff"].max()),
        "mean_feature_abs_diff": float(step_summary_df["feature_mean_abs_diff"].mean()),
        "max_feature_rel_diff": float(step_summary_df["feature_max_rel_diff"].max()),
        "mean_feature_rel_diff": float(step_summary_df["feature_mean_rel_diff"].mean()),
        "max_proba_up_abs_diff": float(step_summary_df["proba_up_abs_diff"].max()),
        "mean_proba_up_abs_diff": float(step_summary_df["proba_up_abs_diff"].mean()),
        "rows_with_proba_diff_gt_tol": int(
            (step_summary_df["proba_up_abs_diff"] > PREDICTION_DIFF_TOL).sum()
        ),
    }
    summary_payload.update(prediction_impact_report["summary"].to_dict())
    if "policy_decision_mismatch" in step_summary_df.columns:
        summary_payload.update(
            {
                "policy_audit_bankroll_usdc": float(LIVE_INITIAL_BANKROLL_USDC),
                "rows_with_policy_side_mismatch": int(
                    step_summary_df["policy_side_mismatch"].sum()
                ),
                "policy_side_mismatch_rate": _summary_rate(
                    int(step_summary_df["policy_side_mismatch"].sum()),
                    len(audit_df),
                ),
                "rows_with_policy_reason_mismatch": int(
                    step_summary_df["policy_reason_mismatch"].sum()
                ),
                "policy_reason_mismatch_rate": _summary_rate(
                    int(step_summary_df["policy_reason_mismatch"].sum()),
                    len(audit_df),
                ),
                "rows_with_policy_trade_flag_mismatch": int(
                    step_summary_df["policy_trade_flag_mismatch"].sum()
                ),
                "policy_trade_flag_mismatch_rate": _summary_rate(
                    int(step_summary_df["policy_trade_flag_mismatch"].sum()),
                    len(audit_df),
                ),
                "rows_with_stake_only_policy_mismatch": int(
                    step_summary_df["stake_only_policy_mismatch"].sum()
                ),
                "stake_only_policy_mismatch_rate": _summary_rate(
                    int(step_summary_df["stake_only_policy_mismatch"].sum()),
                    len(audit_df),
                ),
                "rows_with_any_policy_mismatch": int(
                    step_summary_df["policy_decision_mismatch"].sum()
                ),
                "any_policy_mismatch_rate": _summary_rate(
                    int(step_summary_df["policy_decision_mismatch"].sum()),
                    len(audit_df),
                ),
                "max_policy_stake_abs_diff": float(
                    step_summary_df["policy_stake_abs_diff"].max()
                ),
                "mean_policy_stake_abs_diff": float(
                    step_summary_df["policy_stake_abs_diff"].mean()
                ),
            }
        )

    return {
        "summary": pd.Series(summary_payload),
        "step_summary_df": step_summary_df,
        "feature_summary_df": feature_summary_df.reset_index(drop=True),
        "group_summary_df": group_summary_df,
        "builder_summary_df": builder_summary_df,
        "audit_df": audit_df,
        "feature_columns": feature_columns,
        "candidate_feature_frame": pd.DataFrame(
            candidate_matrix,
            columns=feature_columns,
            copy=False,
        ),
        "prediction_impact_feature_summary_df": prediction_impact_report[
            "feature_summary_df"
        ],
        "prediction_impact_group_summary_df": prediction_impact_report[
            "group_summary_df"
        ],
        "prediction_impact_builder_summary_df": prediction_impact_report[
            "builder_summary_df"
        ],
        "single_feature_fixed_pred_matrix": prediction_impact_report[
            "fixed_pred_matrix"
        ],
        "single_feature_proba_shift_matrix": prediction_impact_report[
            "proba_shift_matrix"
        ],
        "single_feature_gap_reduction_matrix": prediction_impact_report[
            "gap_reduction_matrix"
        ],
        "diff_matrix": diff_matrix,
        "rel_diff_matrix": rel_diff_matrix,
        "candidate_nonfinite_mask": candidate_nonfinite_mask,
        "reference_nonfinite_mask": reference_nonfinite_mask,
        "finite_status_mismatch_mask": finite_status_mismatch_mask,
        "candidate_label": candidate_label,
        "reference_label": reference_label,
    }


def build_live_drift_reason_report(
        report,
        *,
        top_n=20,
):
    step_summary_df = report["step_summary_df"].copy()
    feature_summary_df = report["feature_summary_df"].copy()
    diff_matrix = np.asarray(report["diff_matrix"], dtype=np.float64)
    rel_diff_matrix = np.asarray(report["rel_diff_matrix"], dtype=np.float64)
    fixed_pred_matrix = report.get("single_feature_fixed_pred_matrix")
    proba_shift_matrix = report.get("single_feature_proba_shift_matrix")
    gap_reduction_matrix = report.get("single_feature_gap_reduction_matrix")
    if fixed_pred_matrix is not None:
        fixed_pred_matrix = np.asarray(fixed_pred_matrix, dtype=np.float64)
    if proba_shift_matrix is not None:
        proba_shift_matrix = np.asarray(proba_shift_matrix, dtype=np.float64)
    if gap_reduction_matrix is not None:
        gap_reduction_matrix = np.asarray(gap_reduction_matrix, dtype=np.float64)
    feature_columns = list(report["feature_columns"])
    feature_names = np.asarray(feature_columns, dtype=object)
    candidate_label = str(report.get("candidate_label", "candidate"))
    reference_label = str(report.get("reference_label", "reference"))
    candidate_proba_col = f"{candidate_label}_proba_up"
    reference_proba_col = f"{reference_label}_proba_up"
    history_shortfall = step_summary_df.get(
        "history_shortfall",
        pd.Series(0, index=step_summary_df.index, dtype=np.int32),
    )

    drift_mask = step_summary_df["proba_up_abs_diff"] > PREDICTION_DIFF_TOL
    business_mask = (
        step_summary_df["business_decision_mismatch"] > 0
        if "business_decision_mismatch" in step_summary_df.columns
        else step_summary_df["signal_mismatch"] > 0
    )
    policy_mask = (
        step_summary_df["policy_decision_mismatch"] > 0
        if "policy_decision_mismatch" in step_summary_df.columns
        else pd.Series(False, index=step_summary_df.index)
    )
    stake_only_mask = (
        step_summary_df["stake_only_policy_mismatch"] > 0
        if "stake_only_policy_mismatch" in step_summary_df.columns
        else pd.Series(False, index=step_summary_df.index)
    )
    explain_mask = business_mask
    explanation_basis = "business_decision_mismatch"
    if not bool(explain_mask.any()) and bool(stake_only_mask.any()):
        explain_mask = stake_only_mask
        explanation_basis = "stake_only_policy_mismatch"
    if not bool(explain_mask.any()) and bool(drift_mask.any()):
        explain_mask = drift_mask
        explanation_basis = "proba_diff_gt_tol"
    if not bool(explain_mask.any()):
        explain_mask = step_summary_df["feature_max_abs_diff"] > 0
        explanation_basis = "feature_diff_fallback"

    dominant_feature_col = (
        "top_prediction_impact_feature"
        if "top_prediction_impact_feature" in step_summary_df.columns
        else "worst_feature"
    )
    dominant_group_col = (
        "top_prediction_impact_group"
        if "top_prediction_impact_group" in step_summary_df.columns
        else "worst_group"
    )
    dominant_builder_family_col = (
        "top_prediction_impact_builder_family"
        if "top_prediction_impact_builder_family" in step_summary_df.columns
        else "worst_builder_family"
    )
    dominant_builder_name_col = (
        "top_prediction_impact_builder_name"
        if "top_prediction_impact_builder_name" in step_summary_df.columns
        else "worst_builder_name"
    )
    row_sort_cols = [
        col
        for col in [
            "top_prediction_impact_abs_proba_gap_reduction_if_fixed",
            "proba_up_abs_diff",
            "feature_max_abs_diff",
            "finite_status_mismatch_count",
        ]
        if col in step_summary_df.columns
    ]
    explain_rows_df = (
        step_summary_df.loc[explain_mask]
        .sort_values(
            row_sort_cols,
            ascending=[False] * len(row_sort_cols),
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if explain_rows_df.empty:
        summary = pd.Series(
            {
                "rows_selected_for_explanation": 0,
                "rows_with_proba_diff_gt_tol": int(drift_mask.sum()),
                "rows_with_business_decision_mismatch": int(business_mask.sum()),
                "rows_with_stake_only_policy_mismatch": int(stake_only_mask.sum()),
                "rows_with_any_policy_mismatch": int(policy_mask.sum()),
                "explanation_basis": explanation_basis,
                "rows_with_history_shortfall": int(history_shortfall.gt(0).sum()),
                "prediction_drift_rows_with_history_shortfall": 0,
                "dominant_prediction_impact_group": None,
                "dominant_prediction_impact_builder_family": None,
                "dominant_prediction_impact_builder_name": None,
                "dominant_prediction_impact_feature": None,
                "dominant_worst_group": None,
                "dominant_worst_builder_family": None,
                "dominant_worst_builder_name": None,
                "dominant_worst_feature": None,
            }
        )
        return {
            "summary": summary,
            "row_summary_df": explain_rows_df,
            "group_summary_df": pd.DataFrame(),
            "builder_summary_df": pd.DataFrame(),
            "feature_summary_df": pd.DataFrame(),
        }

    explain_row_mask = explain_mask.to_numpy(dtype=bool, copy=False)
    if len(explain_rows_df) == len(step_summary_df):
        explain_diff_matrix = diff_matrix
        explain_rel_diff_matrix = rel_diff_matrix
    else:
        explain_diff_matrix = diff_matrix[explain_row_mask, :]
        explain_rel_diff_matrix = rel_diff_matrix[explain_row_mask, :]

    explain_group_summary_df = (
        explain_rows_df.groupby(dominant_group_col, dropna=False)
        .agg(
            row_count=("Opened", "count"),
            max_proba_up_abs_diff=("proba_up_abs_diff", "max"),
            mean_proba_up_abs_diff=("proba_up_abs_diff", "mean"),
            max_top_prediction_impact_abs_proba_gap_reduction_if_fixed=(
                "top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "max",
            ),
            mean_top_prediction_impact_abs_proba_gap_reduction_if_fixed=(
                "top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "mean",
            ),
            max_top_prediction_impact_abs_proba_shift_if_fixed=(
                "top_prediction_impact_abs_proba_shift_if_fixed",
                "max",
            ),
            mean_top_prediction_impact_abs_proba_shift_if_fixed=(
                "top_prediction_impact_abs_proba_shift_if_fixed",
                "mean",
            ),
        )
        .sort_values(
            [
                "row_count",
                "max_top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "mean_top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "max_proba_up_abs_diff",
                "max_top_prediction_impact_abs_proba_shift_if_fixed",
            ],
            ascending=[False, False, False, False, False],
            kind="stable",
        )
        .reset_index()
        .rename(columns={dominant_group_col: "prediction_impact_group"})
    )
    explain_builder_summary_df = (
        explain_rows_df.groupby(
            [dominant_builder_family_col, dominant_builder_name_col],
            dropna=False,
        )
        .agg(
            row_count=("Opened", "count"),
            max_proba_up_abs_diff=("proba_up_abs_diff", "max"),
            mean_proba_up_abs_diff=("proba_up_abs_diff", "mean"),
            max_top_prediction_impact_abs_proba_gap_reduction_if_fixed=(
                "top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "max",
            ),
            mean_top_prediction_impact_abs_proba_gap_reduction_if_fixed=(
                "top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "mean",
            ),
            max_top_prediction_impact_abs_proba_shift_if_fixed=(
                "top_prediction_impact_abs_proba_shift_if_fixed",
                "max",
            ),
            mean_top_prediction_impact_abs_proba_shift_if_fixed=(
                "top_prediction_impact_abs_proba_shift_if_fixed",
                "mean",
            ),
        )
        .sort_values(
            [
                "row_count",
                "max_top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "mean_top_prediction_impact_abs_proba_gap_reduction_if_fixed",
                "max_proba_up_abs_diff",
                "max_top_prediction_impact_abs_proba_shift_if_fixed",
            ],
            ascending=[False, False, False, False, False],
            kind="stable",
        )
        .reset_index()
        .rename(
            columns={
                dominant_builder_family_col: "prediction_impact_builder_family",
                dominant_builder_name_col: "prediction_impact_builder_name",
            }
        )
    )

    if (
            fixed_pred_matrix is not None
            and proba_shift_matrix is not None
            and gap_reduction_matrix is not None
            and candidate_proba_col in step_summary_df.columns
            and reference_proba_col in step_summary_df.columns
    ):
        explain_fixed_pred_matrix = fixed_pred_matrix[explain_row_mask, :]
        explain_abs_proba_shift_matrix = np.abs(proba_shift_matrix[explain_row_mask, :])
        explain_gap_reduction_matrix = gap_reduction_matrix[explain_row_mask, :]
        explain_candidate_pred = step_summary_df.loc[
            explain_mask, candidate_proba_col
        ].to_numpy(
            dtype=np.float64,
            copy=False,
        )
        explain_reference_pred = step_summary_df.loc[
            explain_mask, reference_proba_col
        ].to_numpy(
            dtype=np.float64,
            copy=False,
        )
        explain_signal_mismatch = (
                (explain_candidate_pred >= 0.5) != (explain_reference_pred >= 0.5)
        )[:, None]
        explain_fixed_signal_mismatch = (
                                                explain_fixed_pred_matrix >= 0.5
                                        ) != explain_reference_pred[:, None]

        explain_feature_summary_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "max_abs_proba_gap_reduction_if_fixed_on_explained_rows": _safe_nanmax_axis0(
                        explain_gap_reduction_matrix
                    ),
                    "mean_abs_proba_gap_reduction_if_fixed_on_explained_rows": _safe_colwise_mean(
                        explain_gap_reduction_matrix
                    ),
                    "max_abs_proba_shift_if_fixed_on_explained_rows": _safe_nanmax_axis0(
                        explain_abs_proba_shift_matrix
                    ),
                    "mean_abs_proba_shift_if_fixed_on_explained_rows": _safe_colwise_mean(
                        explain_abs_proba_shift_matrix
                    ),
                    "rows_abs_proba_gap_reduced_if_fixed_on_explained_rows": (
                            explain_gap_reduction_matrix > PREDICTION_DIFF_TOL
                    )
                    .sum(axis=0)
                    .astype(np.int32),
                    "rows_signal_mismatch_resolved_if_fixed_on_explained_rows": (
                            explain_signal_mismatch & ~explain_fixed_signal_mismatch
                    )
                    .sum(axis=0)
                    .astype(np.int32),
                    "max_rel_diff_on_explained_rows": _safe_nanmax_axis0(
                        explain_rel_diff_matrix
                    ),
                    "mean_rel_diff_on_explained_rows": _safe_colwise_mean(
                        explain_rel_diff_matrix
                    ),
                    "max_abs_diff_on_explained_rows": _safe_nanmax_axis0(
                        explain_diff_matrix
                    ),
                    "mean_abs_diff_on_explained_rows": _safe_colwise_mean(
                        explain_diff_matrix
                    ),
                    "rmse_abs_diff_on_explained_rows": _safe_colwise_rmse(
                        explain_diff_matrix
                    ),
                }
            )
            .merge(
                feature_summary_df[
                    ["feature", "group", "builder_family", "builder_name"]
                ],
                on="feature",
                how="left",
                sort=False,
            )
            .sort_values(
                [
                    "rows_signal_mismatch_resolved_if_fixed_on_explained_rows",
                    "mean_abs_proba_gap_reduction_if_fixed_on_explained_rows",
                    "max_abs_proba_gap_reduction_if_fixed_on_explained_rows",
                    "mean_abs_proba_shift_if_fixed_on_explained_rows",
                    "max_abs_proba_shift_if_fixed_on_explained_rows",
                    "max_abs_diff_on_explained_rows",
                ],
                ascending=[False, False, False, False, False, False],
                kind="stable",
            )
            .reset_index(drop=True)
        )
    else:
        feature_max = _safe_nanmax_axis0(explain_diff_matrix)
        feature_mean = _safe_colwise_mean(explain_diff_matrix)
        feature_rmse = _safe_colwise_rmse(explain_diff_matrix)
        feature_rel_max = _safe_nanmax_axis0(explain_rel_diff_matrix)
        feature_rel_mean = _safe_colwise_mean(explain_rel_diff_matrix)
        explain_feature_summary_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "max_rel_diff_on_explained_rows": feature_rel_max,
                    "mean_rel_diff_on_explained_rows": feature_rel_mean,
                    "max_abs_diff_on_explained_rows": feature_max,
                    "mean_abs_diff_on_explained_rows": feature_mean,
                    "rmse_abs_diff_on_explained_rows": feature_rmse,
                }
            )
            .merge(
                feature_summary_df[
                    ["feature", "group", "builder_family", "builder_name"]
                ],
                on="feature",
                how="left",
                sort=False,
            )
            .sort_values(
                [
                    "max_rel_diff_on_explained_rows",
                    "mean_rel_diff_on_explained_rows",
                    "max_abs_diff_on_explained_rows",
                    "mean_abs_diff_on_explained_rows",
                    "rmse_abs_diff_on_explained_rows",
                ],
                ascending=[False, False, False, False, False],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    dominant_group = (
        None
        if explain_group_summary_df.empty
        else explain_group_summary_df.iloc[0]["prediction_impact_group"]
    )
    dominant_builder_family = (
        None
        if explain_builder_summary_df.empty
        else explain_builder_summary_df.iloc[0]["prediction_impact_builder_family"]
    )
    dominant_builder_name = (
        None
        if explain_builder_summary_df.empty
        else explain_builder_summary_df.iloc[0]["prediction_impact_builder_name"]
    )
    dominant_feature = (
        None
        if explain_feature_summary_df.empty
        else explain_feature_summary_df.iloc[0]["feature"]
    )

    summary = pd.Series(
        {
            "rows_selected_for_explanation": int(explain_mask.sum()),
            "rows_with_proba_diff_gt_tol": int(drift_mask.sum()),
            "rows_with_business_decision_mismatch": int(business_mask.sum()),
            "rows_with_stake_only_policy_mismatch": int(stake_only_mask.sum()),
            "rows_with_any_policy_mismatch": int(policy_mask.sum()),
            "explanation_basis": explanation_basis,
            "rows_with_history_shortfall": int(history_shortfall.gt(0).sum()),
            "prediction_drift_rows_with_history_shortfall": int(
                explain_rows_df.get("history_shortfall", pd.Series(dtype=np.int32))
                .gt(0)
                .sum()
            ),
            "dominant_prediction_impact_group": dominant_group,
            "dominant_prediction_impact_builder_family": dominant_builder_family,
            "dominant_prediction_impact_builder_name": dominant_builder_name,
            "dominant_prediction_impact_feature": dominant_feature,
            "dominant_worst_group": dominant_group,
            "dominant_worst_builder_family": dominant_builder_family,
            "dominant_worst_builder_name": dominant_builder_name,
            "dominant_worst_feature": dominant_feature,
        }
    )

    return {
        "summary": summary,
        "row_summary_df": explain_rows_df.head(int(top_n)).copy(),
        "group_summary_df": explain_group_summary_df.head(int(top_n)).copy(),
        "builder_summary_df": explain_builder_summary_df.head(int(top_n)).copy(),
        "feature_summary_df": explain_feature_summary_df.head(int(top_n)).copy(),
    }


class AuditWindow:
    __slots__ = (
        "bootstrap_start",
        "audit_start",
        "audit_end",
        "bootstrap_rows",
        "audit_rows",
        "requested_days_back",
        "max_steps",
    )

    def __init__(
            self,
            bootstrap_start,
            audit_start,
            audit_end,
            bootstrap_rows,
            audit_rows,
            requested_days_back,
            max_steps,
    ):
        self.bootstrap_start = bootstrap_start
        self.audit_start = audit_start
        self.audit_end = audit_end
        self.bootstrap_rows = bootstrap_rows
        self.audit_rows = audit_rows
        self.requested_days_back = requested_days_back
        self.max_steps = max_steps


class PseudoLiveAuditPredictor(LivePredictor):
    def __init__(
            self,
            bootstrap_df,
            *,
            model_meta_path=MODEL_META_PATH,
            max_keep=DEFAULT_MAX_KEEP,
            volume_profile_state=None,
            allow_unstable_indicator_summary=AUDIT_ALLOW_UNSTABLE_INDICATOR_SUMMARY,
    ):
        self.model, meta = load_model_and_meta(model_meta_path)
        self.feature_columns = list(meta.get("feature_columns", []))
        if not self.feature_columns:
            raise ValueError("Missing feature_columns in model metadata.")
        validate_volume_profile_feature_columns(
            self.feature_columns,
            source_label=f"model metadata {model_meta_path}",
        )
        validate_basis_premium_feature_columns(
            self.feature_columns,
            source_label=f"model metadata {model_meta_path}",
        )

        self.candle_feature_columns = [
            col for col in self.feature_columns if col in SUPPORTED_CANDLE_FEATURE_COLS
        ]
        self.candle_derived_feature_columns = tuple(
            resolve_candle_derived_feature_cols(self.candle_feature_columns)
        )
        self.candle_pattern_feature_columns = tuple(
            resolve_candle_pattern_feature_cols(self.candle_feature_columns)
        )
        self.target_col = str(meta.get("target_col", "target_5m_candle_up"))
        self.target_bucket_minutes = parse_target_bucket_minutes(self.target_col)
        self.trade_policy_runtime = load_trade_policy_runtime_config(
            TRADE_POLICY_CONFIG_PATH
        )
        self.live_bankroll_usdc = float(LIVE_INITIAL_BANKROLL_USDC)

        feature_parts = split_feature_subset(
            self.feature_columns,
            source_label=f"model metadata {model_meta_path}",
        )
        if feature_parts["streak_intervals"]:
            self.streak_interval_to_rule = resolve_streak_interval_to_rule(
                feature_parts["streak_intervals"]
            )
        else:
            self.streak_interval_to_rule = {}
        self.session_feature_columns = tuple(feature_parts["session_feature_cols"])
        self.realized_volatility_feature_columns = tuple(
            feature_parts["realized_volatility_feature_cols"]
        )
        self.basis_premium_feature_columns = tuple(
            feature_parts["basis_premium_feature_cols"]
        )
        # The audit copies stored basis-premium features back into the candidate vector.
        self.basis_premium_feature_columns = ()
        self.basis_premium_cfg = {}
        self.basis_premium_interval_to_rule = {}
        self.basis_index_close_col = ""
        self.basis_index_ohlcv_idx = None
        self.basis_futures_close_col = ""
        self.basis_futures_close_np = None
        self.volume_profile_feature_columns = tuple(
            feature_parts["volume_profile_feature_cols"]
        )
        self.volume_profile_cfg = normalize_volume_profile_config(
            MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")
        )
        validate_volume_profile_model_metadata(
            meta,
            feature_columns=self.volume_profile_feature_columns,
            cfg=self.volume_profile_cfg,
            source_label=f"model metadata {model_meta_path}",
        )
        self.volume_profile_enabled = bool(
            self.volume_profile_feature_columns and self.volume_profile_cfg["enabled"]
        )
        self.volume_profile_state_path = (
                VP_PSEUDO_LIVE_AUDIT_RUNTIME_STATE_DIR
                / f"{SYMBOL}_{INTERVAL}_{VP_FEATURE_VERSION}"
        )
        self.volume_profile_modeling_state_path = (
                VP_PSEUDO_LIVE_AUDIT_MODELING_STATE_DIR
                / f"{SYMBOL}_{INTERVAL}_{VP_FEATURE_VERSION}_modeling_end"
        )
        self.volume_profile_state_source_path = None
        self.volume_profile_save_pool = None

        self.indicator_specs = load_indicator_specs(
            self.feature_columns,
            source_label=f"model metadata {model_meta_path}",
        )
        requirements_indicator_specs = self.indicator_specs
        if allow_unstable_indicator_summary:
            requirements_payload = json.loads(
                Path(INDICATOR_HISTORY_REQUIREMENTS_PATH).read_text(encoding="utf-8")
            )
            unstable_feature_cols = {
                str(feature_col).strip()
                for feature_col in requirements_payload.get("unstable_features", [])
                if str(feature_col).strip()
            }
            if not unstable_feature_cols:
                summary_path_raw = requirements_payload.get("analysis_summary_path")
                if summary_path_raw not in (None, ""):
                    summary_path = Path(summary_path_raw)
                    if summary_path.exists():
                        summary_payload = json.loads(
                            summary_path.read_text(encoding="utf-8")
                        )
                        unstable_feature_cols = {
                            str(feature_col).strip()
                            for feature_col in summary_payload.get(
                                "unstable_features", []
                            )
                            if str(feature_col).strip()
                        }
            filtered_specs = [
                spec
                for spec in self.indicator_specs
                if str(spec.feature_col) not in unstable_feature_cols
            ]
            if len(filtered_specs) != len(self.indicator_specs):
                print(
                    "[audit] ignoring unstable features in per-feature history validation "
                    f"count={len(self.indicator_specs) - len(filtered_specs)}"
                )
                requirements_indicator_specs = filtered_specs
        self.indicator_history_requirements = load_indicator_history_requirements(
            INDICATOR_HISTORY_REQUIREMENTS_PATH,
            indicator_specs=requirements_indicator_specs,
            allow_unstable=allow_unstable_indicator_summary,
        )
        self.required_stable_window = int(
            self.indicator_history_requirements["global_required_runtime_window"]
        )
        self.required_stable_window_raw = int(
            self.indicator_history_requirements["global_required_stable_window"]
        )
        self.indicator_stable_window_by_feature = dict(
            self.indicator_history_requirements["stable_window_by_feature"]
        )
        self.indicator_runtime_window_by_feature = dict(
            self.indicator_history_requirements["runtime_window_by_feature"]
        )
        self.bootstrap_candles = len(bootstrap_df)
        self.max_keep = max(int(max_keep), int(self.required_stable_window))

        bootstrap_df = bootstrap_df.copy()
        bootstrap_df["Opened"] = _ensure_utc_opened(bootstrap_df["Opened"])
        bootstrap_df = bootstrap_df.sort_values("Opened").reset_index(drop=True)
        if bootstrap_df.empty:
            raise ValueError("bootstrap_df cannot be empty.")

        self.opened_candles = deque(pd.Timestamp(v) for v in bootstrap_df["Opened"])
        self.ohlcv_np = bootstrap_df[OHLCV_COLS].to_numpy(dtype=np.float64, copy=True)
        self.opened_ns_np = np.fromiter(
            (opened.value for opened in self.opened_candles),
            dtype=np.int64,
            count=len(self.opened_candles),
        )
        self.candle_open_close = {
            opened: (float(self.ohlcv_np[i, 0]), float(self.ohlcv_np[i, 3]))
            for i, opened in enumerate(self.opened_candles)
        }
        self.records = []
        self.predicted_buckets = set()
        self.local_tz = None
        self.last_indicator_nan_cols = []
        self.last_processed_closed_opened = (
            self.opened_candles[-1] if self.opened_candles else None
        )
        self.predictions_path = None
        self.realized_volatility_state = None
        self.latest_realized_volatility_values = {}
        self._initialize_realized_volatility_state()

        self.volume_profile_state = None
        if self.volume_profile_enabled:
            if volume_profile_state is not None:
                self.volume_profile_state = copy.deepcopy(volume_profile_state)
            else:
                self.volume_profile_state = bootstrap_state_from_history(
                    bootstrap_df.loc[:, ["Opened", "High", "Low", "Volume"]],
                    self.volume_profile_cfg,
                )

    def _save_runtime_volume_profile_state(self, log=False, context="state"):
        return None

    def evaluate_policy_decision(
            self,
            prob_up_raw,
            *,
            bankroll=None,
    ):
        prev_bankroll = float(self.live_bankroll_usdc)
        try:
            if bankroll is not None:
                self.live_bankroll_usdc = float(bankroll)
            return dict(self._build_policy_intent(proba_up=float(prob_up_raw)))
        finally:
            self.live_bankroll_usdc = prev_bankroll

    def build_feature_snapshot(self, volume_profile_values=None):
        vector = self._build_feature_vector(volume_profile_values=volume_profile_values)
        nonfinite_feature_indices = tuple(
            int(idx) for idx in np.flatnonzero(~np.isfinite(vector[0, :]))
        )
        return {
            "vector": vector,
            "nonfinite_feature_indices": nonfinite_feature_indices,
            "indicator_nan_cols": tuple(self.last_indicator_nan_cols),
        }


def _load_opened_series(parquet_path):
    opened = pd.read_parquet(parquet_path, columns=["Opened"])["Opened"]
    return pd.to_datetime(opened)


def resolve_audit_window(
        *,
        parquet_path,
        days_back=DEFAULT_AUDIT_DAYS_BACK,
        bootstrap_candles=DEFAULT_BOOTSTRAP_CANDLES,
        max_steps=None,
):
    opened = _load_opened_series(parquet_path)
    if opened.empty:
        raise ValueError(f"Modeling parquet has no rows: {parquet_path}")

    audit_end = pd.Timestamp(opened.iloc[-1])
    requested_start = audit_end - pd.Timedelta(days=int(days_back))
    start_idx = int(opened.searchsorted(requested_start, side="left"))
    if start_idx >= len(opened):
        raise ValueError("Resolved audit_start is beyond the dataset end.")
    if start_idx < int(bootstrap_candles):
        raise ValueError(
            "Not enough history before the requested audit start. "
            f"Need bootstrap_candles={bootstrap_candles}, got only {start_idx} rows."
        )

    bootstrap_start_idx = start_idx - int(bootstrap_candles)
    if max_steps is None:
        end_idx = len(opened) - 1
    else:
        end_idx = min(len(opened) - 1, start_idx + int(max_steps) - 1)

    if end_idx < start_idx:
        raise ValueError("Audit window resolved to zero rows.")

    return AuditWindow(
        bootstrap_start=pd.Timestamp(opened.iloc[bootstrap_start_idx]),
        audit_start=pd.Timestamp(opened.iloc[start_idx]),
        audit_end=pd.Timestamp(opened.iloc[end_idx]),
        bootstrap_rows=int(bootstrap_candles),
        audit_rows=int(end_idx - start_idx + 1),
        requested_days_back=int(days_back),
        max_steps=None if max_steps is None else int(max_steps),
    )


def load_modeling_audit_frame(
        *,
        parquet_path,
        feature_columns,
        audit_window,
):
    columns = ["Opened", *RAW_OHLCV_COLS, *feature_columns]
    frame = pd.read_parquet(
        parquet_path,
        columns=columns,
        filters=[
            (
                "Opened",
                ">=",
                audit_window.bootstrap_start.to_pydatetime().replace(tzinfo=None),
            ),
            (
                "Opened",
                "<=",
                audit_window.audit_end.to_pydatetime().replace(tzinfo=None),
            ),
        ],
    )
    frame["Opened"] = _ensure_utc_opened(frame["Opened"])
    frame = (
        frame.sort_values("Opened")
        .drop_duplicates(subset=["Opened"])
        .reset_index(drop=True)
    )
    return frame


def load_anchor_volume_profile_history(
        *,
        parquet_path,
        anchor_candle_opened,
):
    anchor_candle_opened = pd.Timestamp(anchor_candle_opened)
    frame = pd.read_parquet(
        parquet_path,
        columns=["Opened", "High", "Low", "Volume"],
        filters=[
            (
                "Opened",
                "<=",
                _naive_utc_timestamp(anchor_candle_opened),
            ),
        ],
    )
    frame["Opened"] = _ensure_utc_opened(frame["Opened"])
    frame = (
        frame.sort_values("Opened")
        .drop_duplicates(subset=["Opened"])
        .reset_index(drop=True)
    )
    return frame


def build_or_load_anchor_volume_profile_state(
        *,
        parquet_path,
        anchor_candle_opened,
        overwrite=False,
):
    state_path = resolve_anchor_volume_profile_state_path(anchor_candle_opened)
    vp_cfg = normalize_volume_profile_config(
        MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")
    )
    if (
            not overwrite
            and state_path.with_suffix(".npz").exists()
            and state_path.with_suffix(".json").exists()
    ):
        try:
            state = load_volume_profile_state(state_path)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(
                "[audit] existing anchor vp state is unreadable; rebuilding "
                f"path={state_path.with_suffix('.npz')} error={exc}"
            )
        else:
            if volume_profile_state_matches_config(state, vp_cfg):
                return state, state_path
            print(
                "[audit] existing anchor vp state config mismatch; rebuilding "
                f"path={state_path.with_suffix('.npz')}"
            )

    history_df = load_anchor_volume_profile_history(
        parquet_path=parquet_path,
        anchor_candle_opened=anchor_candle_opened,
    )
    state = bootstrap_state_from_history(
        history_df.loc[:, ["Opened", "High", "Low", "Volume"]],
        vp_cfg,
    )
    save_volume_profile_state(state, state_path)
    return load_volume_profile_state(state_path), state_path


def run_stored_modeling_vs_current_recompute_audit(
        *,
        days_back=DEFAULT_AUDIT_DAYS_BACK,
        max_steps=None,
        history_tail_fraction=1.0,
        model_meta_path=MODEL_META_PATH,
        parquet_path=None,
):
    model_meta_path = Path(model_meta_path)
    parquet_path = (
            _optional_path(parquet_path) or resolve_modeling_dataset_parquet_path()
    )

    model, meta = load_model_and_meta(model_meta_path)
    feature_importance_df = _load_model_feature_importance_frame(meta)
    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        raise ValueError("Missing feature_columns in model metadata.")
    validate_volume_profile_feature_columns(
        feature_columns,
        source_label=f"model metadata {model_meta_path}",
    )
    validate_basis_premium_feature_columns(
        feature_columns,
        source_label=f"model metadata {model_meta_path}",
    )
    ignored_basis_feature_columns = _ignored_basis_premium_feature_columns(
        feature_columns
    )
    recomputed_feature_columns = _audited_feature_columns(feature_columns)
    validate_volume_profile_model_metadata(
        meta,
        feature_columns=feature_columns,
        cfg=normalize_volume_profile_config(
            MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")
        ),
        source_label=f"model metadata {model_meta_path}",
    )

    audit_window = resolve_audit_window(
        parquet_path=parquet_path,
        days_back=days_back,
        bootstrap_candles=0,
        max_steps=max_steps,
    )
    stored_audit_df = load_modeling_audit_frame(
        parquet_path=parquet_path,
        feature_columns=feature_columns,
        audit_window=audit_window,
    )
    history_start, history_rows, history_total_rows = (
        resolve_recent_history_tail_window(
            parquet_path=parquet_path,
            audit_end=audit_window.audit_end,
            tail_fraction=history_tail_fraction,
        )
    )
    if audit_window.audit_start < history_start:
        raise ValueError(
            "stored_vs_current_recompute history tail is too short for the requested audit window. "
            f"audit_start={audit_window.audit_start.isoformat()} "
            f"history_start={history_start.isoformat()} "
            f"tail_fraction={float(history_tail_fraction):.6f}"
        )
    raw_history_df = load_modeling_raw_history_frame(
        parquet_path=parquet_path,
        audit_end=audit_window.audit_end,
        history_start=history_start,
    )
    recomputed_history_df = build_current_recomputed_feature_history(
        raw_history_df=raw_history_df,
        feature_columns=recomputed_feature_columns,
    )
    recomputed_audit_df = align_feature_frame_to_audit_rows(
        audit_df=stored_audit_df,
        feature_frame=recomputed_history_df,
        feature_columns=recomputed_feature_columns,
    )
    recomputed_audit_df = _copy_ignored_basis_premium_features(
        recomputed_audit_df,
        stored_audit_df,
        feature_columns,
    )

    feature_group_by_name = _feature_group_map(feature_columns)
    feature_builder_frame = _feature_builder_frame(feature_columns)
    policy_predictor = PseudoLiveAuditPredictor(
        raw_history_df.tail(
            max(1, min(len(raw_history_df), DEFAULT_BOOTSTRAP_CANDLES))
        ).copy(),
        model_meta_path=model_meta_path,
        max_keep=max(
            DEFAULT_MAX_KEEP, len(raw_history_df.tail(DEFAULT_BOOTSTRAP_CANDLES))
        ),
        volume_profile_state=None,
    )
    report = build_matrix_comparison_report(
        candidate_label="recomputed",
        reference_label="stored",
        candidate_matrix=recomputed_audit_df[feature_columns].to_numpy(
            dtype=np.float64, copy=True
        ),
        reference_matrix=stored_audit_df[feature_columns].to_numpy(
            dtype=np.float64, copy=True
        ),
        audit_df=stored_audit_df,
        feature_columns=feature_columns,
        feature_group_by_name=feature_group_by_name,
        feature_builder_frame=feature_builder_frame,
        model=model,
        feature_importance_df=feature_importance_df,
        policy_predictor=policy_predictor,
    )
    report["summary"] = pd.concat(
        [
            pd.Series(
                {
                    "parquet_path": str(parquet_path),
                    "audit_start": audit_window.audit_start.isoformat(),
                    "audit_end": audit_window.audit_end.isoformat(),
                    "audit_rows": audit_window.audit_rows,
                    "days_back": audit_window.requested_days_back,
                    "max_steps": audit_window.max_steps,
                    "history_tail_fraction": float(history_tail_fraction),
                    "history_start": history_start.isoformat(),
                    "history_rows": int(history_rows),
                    "history_total_rows": int(history_total_rows),
                    "ignored_basis_premium_feature_count": len(
                        ignored_basis_feature_columns
                    ),
                    "ignored_basis_premium_features": ", ".join(
                        ignored_basis_feature_columns
                    ),
                    "report_type": "stored_modeling_vs_current_recompute",
                }
            ),
            report["summary"],
        ]
    )
    report["audit_window"] = audit_window
    report["audit_df"] = stored_audit_df
    report["stored_audit_df"] = stored_audit_df
    report["recomputed_audit_df"] = recomputed_audit_df
    report["candidate_feature_frame"] = recomputed_audit_df.loc[
        :, feature_columns
    ].copy()
    report["feature_columns"] = feature_columns
    return report


def run_live_modeling_feature_audit(
        *,
        days_back=DEFAULT_AUDIT_DAYS_BACK,
        bootstrap_candles=DEFAULT_BOOTSTRAP_CANDLES,
        max_steps=None,
        max_keep=DEFAULT_MAX_KEEP,
        model_meta_path=MODEL_META_PATH,
        parquet_path=None,
        use_anchor_vp_state=True,
        overwrite_anchor_vp_state=False,
):
    model_meta_path = Path(model_meta_path)
    parquet_path = (
            _optional_path(parquet_path) or resolve_modeling_dataset_parquet_path()
    )
    progress_enabled = bool(AUDIT_PROGRESS_ENABLED)
    progress_every = max(1, int(AUDIT_PROGRESS_EVERY_STEPS))

    if progress_enabled:
        print(
            "[audit] starting live_vs_stored "
            f"days_back={int(days_back)} "
            f"bootstrap_candles={int(bootstrap_candles)} "
            f"max_steps={max_steps if max_steps is not None else 'all'} "
            f"max_keep={int(max_keep)}"
        )
        if AUDIT_ALLOW_UNSTABLE_INDICATOR_SUMMARY:
            print(
                "[audit] indicator history requirements artifact contains unstable features; "
                "continuing because audit override is enabled"
            )
        print(
            "[audit] inputs " f"model_meta={model_meta_path} " f"parquet={parquet_path}"
        )

    model, meta = load_model_and_meta(model_meta_path)
    feature_importance_df = _load_model_feature_importance_frame(meta)
    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        raise ValueError("Missing feature_columns in model metadata.")
    validate_volume_profile_feature_columns(
        feature_columns,
        source_label=f"model metadata {model_meta_path}",
    )
    validate_basis_premium_feature_columns(
        feature_columns,
        source_label=f"model metadata {model_meta_path}",
    )
    ignored_basis_feature_columns = _ignored_basis_premium_feature_columns(
        feature_columns
    )
    validate_volume_profile_model_metadata(
        meta,
        feature_columns=feature_columns,
        cfg=normalize_volume_profile_config(
            MODELING_DATASET_SETTINGS.get("volume_profile_fixed_range")
        ),
        source_label=f"model metadata {model_meta_path}",
    )

    required_stable_window = load_required_stable_window(
        INDICATOR_HISTORY_REQUIREMENTS_PATH,
        allow_unstable=AUDIT_ALLOW_UNSTABLE_INDICATOR_SUMMARY,
    )
    resolved_bootstrap_candles = max(
        int(bootstrap_candles), int(required_stable_window)
    )
    resolved_max_keep = max(int(max_keep), int(required_stable_window))
    if progress_enabled and (
            resolved_bootstrap_candles != int(bootstrap_candles)
            or resolved_max_keep != int(max_keep)
    ):
        print(
            "[audit] adjusted_history_windows "
            f"required_stable_window={int(required_stable_window)} "
            f"bootstrap_candles={int(bootstrap_candles)}->{resolved_bootstrap_candles} "
            f"max_keep={int(max_keep)}->{resolved_max_keep}"
        )
    bootstrap_candles = resolved_bootstrap_candles
    max_keep = resolved_max_keep

    audit_window = resolve_audit_window(
        parquet_path=parquet_path,
        days_back=days_back,
        bootstrap_candles=bootstrap_candles,
        max_steps=max_steps,
    )
    if progress_enabled:
        print(
            "[audit] window "
            f"bootstrap_start={audit_window.bootstrap_start.isoformat()} "
            f"audit_start={audit_window.audit_start.isoformat()} "
            f"audit_end={audit_window.audit_end.isoformat()} "
            f"bootstrap_rows={int(audit_window.bootstrap_rows)} "
            f"audit_rows={int(audit_window.audit_rows)}"
        )
    modeling_frame = load_modeling_audit_frame(
        parquet_path=parquet_path,
        feature_columns=feature_columns,
        audit_window=audit_window,
    )

    if len(modeling_frame) != audit_window.bootstrap_rows + audit_window.audit_rows:
        raise RuntimeError(
            "Unexpected audit frame length. "
            f"Expected {audit_window.bootstrap_rows + audit_window.audit_rows}, "
            f"got {len(modeling_frame)}."
        )

    bootstrap_df = modeling_frame.iloc[: audit_window.bootstrap_rows].copy()
    audit_df = (
        modeling_frame.iloc[audit_window.bootstrap_rows:].copy().reset_index(drop=True)
    )
    if audit_df.empty:
        raise RuntimeError("Audit dataframe is empty after splitting bootstrap rows.")

    anchor_candle_opened = pd.Timestamp(bootstrap_df["Opened"].iloc[-1])
    anchor_vp_state = None
    anchor_vp_state_path = None
    if use_anchor_vp_state:
        if progress_enabled:
            print(
                "[audit] loading anchor vp state "
                f"anchor_opened={anchor_candle_opened.isoformat()} "
                f"overwrite={bool(overwrite_anchor_vp_state)}"
            )
        anchor_vp_state, anchor_vp_state_path = (
            build_or_load_anchor_volume_profile_state(
                parquet_path=parquet_path,
                anchor_candle_opened=anchor_candle_opened,
                overwrite=overwrite_anchor_vp_state,
            )
        )
        if progress_enabled:
            print(
                "[audit] anchor vp state ready "
                f"path={anchor_vp_state_path.with_suffix('.npz')}"
            )

    predictor = PseudoLiveAuditPredictor(
        bootstrap_df,
        model_meta_path=model_meta_path,
        max_keep=max_keep,
        volume_profile_state=anchor_vp_state,
    )
    predictor.model = model
    if progress_enabled:
        print(
            "[audit] predictor ready "
            f"required_stable_window={int(predictor.required_stable_window)} "
            f"bootstrap_rows_loaded={int(predictor.bootstrap_candles)} "
            f"initial_buffer_rows={len(predictor.opened_candles)}"
        )

    feature_group_by_name = _feature_group_map(feature_columns)
    feature_builder_frame = _feature_builder_frame(feature_columns)
    indicator_nan_count_rows = []
    history_rows_used_rows = []
    history_shortfall_rows = []
    decision_row_indices = []
    live_vector_rows = []
    live_nonfinite_rows = []

    ignored_basis_feature_indices = [
        feature_columns.index(feature)
        for feature in ignored_basis_feature_columns
        if feature in feature_columns
    ]
    ohlcv_matrix = audit_df[list(RAW_OHLCV_COLS)].to_numpy(dtype=np.float64, copy=False)
    opened_values = audit_df["Opened"].to_list()
    loop_started_at = time.perf_counter()
    total_steps = len(opened_values)
    decision_steps = 0
    progress_interval = max(1, int(progress_every))
    if total_steps > 0:
        progress_interval = max(
            progress_interval,
            int(np.ceil(total_steps / float(AUDIT_PROGRESS_MAX_UPDATES))),
        )

    for row_idx, opened in enumerate(opened_values):
        predictor._append_new_candle(
            pd.Timestamp(opened),
            tuple(float(v) for v in ohlcv_matrix[row_idx, :]),
        )
        volume_profile_values = (
            predictor._prepare_volume_profile_features_for_latest_candle(opened)
        )
        is_decision_step = _is_live_decision_opened(
            opened, predictor.target_bucket_minutes
        )
        if is_decision_step:
            snapshot = predictor.build_feature_snapshot(volume_profile_values)
            live_vector = snapshot["vector"][0, :].astype(np.float64, copy=True)
            if ignored_basis_feature_indices:
                live_vector[ignored_basis_feature_indices] = audit_df.loc[
                    row_idx, list(ignored_basis_feature_columns)
                ].to_numpy(dtype=np.float64, copy=False)
            live_vector_rows.append(live_vector)
            nonfinite_row = ~np.isfinite(live_vector)
            live_nonfinite_rows.append(nonfinite_row)
            indicator_nan_count_rows.append(len(snapshot["indicator_nan_cols"]))
            current_history_rows = len(predictor.opened_candles)
            history_rows_used_rows.append(current_history_rows)
            history_shortfall_rows.append(
                max(0, int(predictor.required_stable_window) - current_history_rows)
            )
            decision_row_indices.append(row_idx)
            decision_steps += 1
        completed_steps = row_idx + 1
        if progress_enabled and (
                completed_steps == 1
                or completed_steps == total_steps
                or completed_steps % progress_interval == 0
        ):
            elapsed_sec = time.perf_counter() - loop_started_at
            steps_per_sec = (
                completed_steps / elapsed_sec if elapsed_sec > 0 else float("nan")
            )
            remaining_steps = total_steps - completed_steps
            eta_sec = (
                remaining_steps / steps_per_sec
                if np.isfinite(steps_per_sec) and steps_per_sec > 0
                else float("nan")
            )
            print(
                "[audit] progress "
                f"{completed_steps}/{total_steps} "
                f"({(completed_steps / total_steps) * 100.0:.1f}%) "
                f"opened={pd.Timestamp(opened).isoformat()} "
                f"buffer_rows={len(predictor.opened_candles)} "
                f"decision_rows={decision_steps} "
                f"elapsed={_format_duration(elapsed_sec)} "
                f"eta={_format_duration(eta_sec) if np.isfinite(eta_sec) else 'n/a'}"
            )

    decision_audit_df = (
        audit_df.iloc[decision_row_indices].copy().reset_index(drop=True)
    )
    if decision_audit_df.empty:
        raise RuntimeError("No live decision rows were selected for the audit window.")
    modeling_matrix = decision_audit_df[feature_columns].to_numpy(
        dtype=np.float64, copy=True
    )
    live_matrix = np.vstack(live_vector_rows).astype(np.float64, copy=False)
    live_nonfinite_mask = np.vstack(live_nonfinite_rows).astype(bool, copy=False)
    indicator_nan_count = np.asarray(indicator_nan_count_rows, dtype=np.int32)
    history_rows_used = np.asarray(history_rows_used_rows, dtype=np.int32)
    history_shortfall = np.asarray(history_shortfall_rows, dtype=np.int32)

    if progress_enabled:
        print("[audit] building comparison report")
    live_report = build_matrix_comparison_report(
        candidate_label="live",
        reference_label="stored",
        candidate_matrix=live_matrix,
        reference_matrix=modeling_matrix,
        audit_df=decision_audit_df,
        feature_columns=feature_columns,
        feature_group_by_name=feature_group_by_name,
        feature_builder_frame=feature_builder_frame,
        model=model,
        feature_importance_df=feature_importance_df,
        policy_predictor=predictor,
    )
    live_report["step_summary_df"]["indicator_nan_count"] = indicator_nan_count
    live_report["step_summary_df"]["history_rows_used"] = history_rows_used
    live_report["step_summary_df"]["history_shortfall"] = history_shortfall
    live_report["summary"] = pd.concat(
        [
            pd.Series(
                {
                    "parquet_path": str(parquet_path),
                    "audit_start": audit_window.audit_start.isoformat(),
                    "audit_end": audit_window.audit_end.isoformat(),
                    "bootstrap_rows": audit_window.bootstrap_rows,
                    "audit_rows_total_1m": audit_window.audit_rows,
                    "decision_rows": len(decision_audit_df),
                    "days_back": audit_window.requested_days_back,
                    "max_steps": audit_window.max_steps,
                    "max_keep": int(max_keep),
                    "required_stable_window": int(predictor.required_stable_window),
                    "ignored_basis_premium_feature_count": len(
                        ignored_basis_feature_columns
                    ),
                    "ignored_basis_premium_features": ", ".join(
                        ignored_basis_feature_columns
                    ),
                    "use_anchor_vp_state": bool(use_anchor_vp_state),
                    "anchor_vp_state_path": (
                        None
                        if anchor_vp_state_path is None
                        else str(anchor_vp_state_path.with_suffix(".npz"))
                    ),
                    "report_type": "pseudo_live_vs_stored_modeling_decision_only",
                }
            ),
            live_report["summary"],
        ]
    )

    live_feature_frame = pd.DataFrame(
        live_matrix,
        columns=feature_columns,
        index=decision_audit_df.index,
    )
    live_report["audit_window"] = audit_window
    live_report["audit_df"] = decision_audit_df
    live_report["live_feature_frame"] = live_feature_frame
    live_report["candidate_feature_frame"] = live_feature_frame
    live_report["feature_columns"] = feature_columns
    if progress_enabled:
        print("[audit] building drift reason report")
    live_report["drift_reason_report"] = build_live_drift_reason_report(
        live_report,
        top_n=AUDIT_TOP_N,
    )

    return {
        "summary": live_report["summary"],
        "audit_window": audit_window,
        "modeling_frame": modeling_frame,
        "audit_df": decision_audit_df,
        "live_feature_frame": live_feature_frame,
        "step_summary_df": live_report["step_summary_df"],
        "feature_summary_df": live_report["feature_summary_df"],
        "group_summary_df": live_report["group_summary_df"],
        "builder_summary_df": live_report["builder_summary_df"],
        "live_nonfinite_mask": live_report["candidate_nonfinite_mask"],
        "modeling_nonfinite_mask": live_report["reference_nonfinite_mask"],
        "finite_status_mismatch_mask": live_report["finite_status_mismatch_mask"],
        "feature_columns": feature_columns,
        "feature_builder_frame": feature_builder_frame,
        "live_vs_stored_report": live_report,
        "drift_reason_report": live_report["drift_reason_report"],
    }


def feature_drilldown(
        results,
        feature_name,
        *,
        report_key=None,
        top_n=20,
):
    report = results if report_key is None else results[report_key]
    feature_columns = list(report["feature_columns"])
    if feature_name not in feature_columns:
        raise KeyError(f"Unknown feature_name={feature_name!r}")

    feature_idx = feature_columns.index(feature_name)
    audit_df = report["audit_df"]
    candidate_feature_frame = report.get(
        "candidate_feature_frame", report.get("live_feature_frame")
    )
    step_summary_df = report["step_summary_df"]
    candidate_nonfinite_mask = report.get(
        "candidate_nonfinite_mask",
        report.get("live_nonfinite_mask"),
    )
    reference_nonfinite_mask = report.get(
        "reference_nonfinite_mask",
        report.get("modeling_nonfinite_mask"),
    )
    candidate_series = candidate_feature_frame[feature_name].to_numpy(
        dtype=np.float64, copy=False
    )
    reference_series = audit_df[feature_name].to_numpy(dtype=np.float64, copy=False)
    abs_diff = np.abs(candidate_series - reference_series)
    abs_diff[~(np.isfinite(candidate_series) & np.isfinite(reference_series))] = np.nan
    rel_diff = _safe_relative_diff(candidate_series, reference_series)
    candidate_label = str(report.get("candidate_label", "candidate"))
    reference_label = str(report.get("reference_label", "reference"))
    candidate_proba_col = f"{candidate_label}_proba_up"
    reference_proba_col = f"{reference_label}_proba_up"

    drilldown_df = pd.DataFrame(
        {
            "Opened": audit_df["Opened"],
            "Open": audit_df["Open"],
            "High": audit_df["High"],
            "Low": audit_df["Low"],
            "Close": audit_df["Close"],
            "Volume": audit_df["Volume"],
            "stored_value": reference_series,
            "candidate_value": candidate_series,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "candidate_nonfinite": candidate_nonfinite_mask[:, feature_idx],
            "stored_nonfinite": reference_nonfinite_mask[:, feature_idx],
            "finite_status_mismatch": report["finite_status_mismatch_mask"][
                :, feature_idx
            ],
            "proba_up_abs_diff": step_summary_df["proba_up_abs_diff"],
        }
    )
    proba_cols = [col for col in step_summary_df.columns if col.endswith("_proba_up")]
    for col in proba_cols:
        drilldown_df[col] = step_summary_df[col]
    fixed_pred_matrix = report.get("single_feature_fixed_pred_matrix")
    proba_shift_matrix = report.get("single_feature_proba_shift_matrix")
    gap_reduction_matrix = report.get("single_feature_gap_reduction_matrix")
    if (
            fixed_pred_matrix is not None
            and proba_shift_matrix is not None
            and gap_reduction_matrix is not None
            and candidate_proba_col in step_summary_df.columns
            and reference_proba_col in step_summary_df.columns
    ):
        fixed_pred_series = np.asarray(fixed_pred_matrix, dtype=np.float64)[
            :, feature_idx
        ]
        proba_shift_series = np.asarray(proba_shift_matrix, dtype=np.float64)[
            :, feature_idx
        ]
        gap_reduction_series = np.asarray(gap_reduction_matrix, dtype=np.float64)[
            :, feature_idx
        ]
        reference_pred = step_summary_df[reference_proba_col].to_numpy(
            dtype=np.float64, copy=False
        )
        candidate_pred = step_summary_df[candidate_proba_col].to_numpy(
            dtype=np.float64, copy=False
        )
        base_signal_mismatch = (candidate_pred >= 0.5) != (reference_pred >= 0.5)
        fixed_signal_mismatch = (fixed_pred_series >= 0.5) != (reference_pred >= 0.5)
        drilldown_df["proba_up_if_feature_fixed"] = fixed_pred_series
        drilldown_df["proba_up_shift_if_feature_fixed"] = proba_shift_series
        drilldown_df["abs_proba_gap_reduction_if_feature_fixed"] = gap_reduction_series
        drilldown_df["signal_mismatch_resolved_if_feature_fixed"] = (
                base_signal_mismatch & ~fixed_signal_mismatch
        ).astype(np.int8)
        drilldown_df["signal_mismatch_introduced_if_feature_fixed"] = (
                ~base_signal_mismatch & fixed_signal_mismatch
        ).astype(np.int8)

    sort_spec = [
        ("abs_proba_gap_reduction_if_feature_fixed", False),
        ("proba_up_abs_diff", False),
        ("abs_diff", False),
    ]
    sort_cols = [col for col, _ascending in sort_spec if col in drilldown_df.columns]
    sort_ascending = [
        ascending for col, ascending in sort_spec if col in drilldown_df.columns
    ]
    return drilldown_df.sort_values(
        sort_cols,
        ascending=sort_ascending,
        na_position="last",
        kind="stable",
    ).head(int(top_n))


def _build_feature_prediction_impact_export_df(
        feature_summary_df,
        *,
        top_k=None,
        only_impactful=False,
):
    feature_summary_df = feature_summary_df.copy()
    if (
            "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed"
            not in feature_summary_df.columns
    ):
        feature_summary_df["mean_abs_proba_gap_reduction_on_drift_rows_if_fixed"] = (
            feature_summary_df.get("mean_abs_proba_gap_reduction_if_fixed", 0.0)
        )
    if (
            "mean_abs_proba_gap_reduction_on_helped_rows_if_fixed"
            not in feature_summary_df.columns
    ):
        feature_summary_df["mean_abs_proba_gap_reduction_on_helped_rows_if_fixed"] = (
            feature_summary_df.get("mean_abs_proba_gap_reduction_if_fixed", 0.0)
        )
    if "mean_gap_reduction_ratio_on_drift_rows_if_fixed" not in feature_summary_df.columns:
        feature_summary_df["mean_gap_reduction_ratio_on_drift_rows_if_fixed"] = 0.0
    if "max_gap_reduction_ratio_on_drift_rows_if_fixed" not in feature_summary_df.columns:
        feature_summary_df["max_gap_reduction_ratio_on_drift_rows_if_fixed"] = 0.0

    slim_columns = [
        "rank",
        "feature",
        "group",
        "builder",
        "net_pred_gap_reduction",
        "mean_pred_gap_reduction_on_drift_rows",
        "mean_pred_gap_reduction_on_helped_rows",
        "mean_gap_reduction_ratio_on_drift_rows",
        "max_gap_reduction_ratio_on_drift_rows",
        "mean_pred_shift",
        "max_pred_shift",
        "signal_flips_resolved",
        "proba_drift_rows_resolved",
        "rows_helped",
        "rows_hurt",
    ]
    if set(slim_columns).issubset(feature_summary_df.columns):
        export_df = (
            feature_summary_df.loc[:, slim_columns]
            .sort_values(["rank"], ascending=[True], kind="stable")
            .reset_index(drop=True)
        )
        if only_impactful and not export_df.empty:
            impact_mask = (
                    export_df["rows_helped"].gt(0)
                    | export_df["signal_flips_resolved"].gt(0)
                    | export_df["proba_drift_rows_resolved"].gt(0)
                    | export_df["net_pred_gap_reduction"].gt(PREDICTION_DIFF_TOL)
            )
            export_df = export_df.loc[impact_mask].reset_index(drop=True)
            export_df["rank"] = np.arange(1, len(export_df) + 1, dtype=np.int32)
            export_df = export_df.loc[:, slim_columns]
        if top_k is not None:
            export_df = export_df.head(int(top_k)).reset_index(drop=True)
        return export_df

    export_columns = [
        "feature",
        "group",
        "builder_name",
        "net_abs_proba_gap_reduction_if_fixed",
        "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed",
        "mean_abs_proba_gap_reduction_on_helped_rows_if_fixed",
        "mean_gap_reduction_ratio_on_drift_rows_if_fixed",
        "max_gap_reduction_ratio_on_drift_rows_if_fixed",
        "mean_abs_proba_shift_if_fixed",
        "max_abs_proba_shift_if_fixed",
        "rows_signal_mismatch_resolved_if_fixed",
        "rows_proba_diff_gt_tol_resolved_if_fixed",
        "rows_abs_proba_gap_reduced_if_fixed",
        "rows_abs_proba_gap_worsened_if_fixed",
    ]
    rename_map = {
        "builder_name": "builder",
        "net_abs_proba_gap_reduction_if_fixed": "net_pred_gap_reduction",
        "mean_abs_proba_gap_reduction_on_drift_rows_if_fixed": "mean_pred_gap_reduction_on_drift_rows",
        "mean_abs_proba_gap_reduction_on_helped_rows_if_fixed": "mean_pred_gap_reduction_on_helped_rows",
        "mean_gap_reduction_ratio_on_drift_rows_if_fixed": "mean_gap_reduction_ratio_on_drift_rows",
        "max_gap_reduction_ratio_on_drift_rows_if_fixed": "max_gap_reduction_ratio_on_drift_rows",
        "mean_abs_proba_shift_if_fixed": "mean_pred_shift",
        "max_abs_proba_shift_if_fixed": "max_pred_shift",
        "rows_signal_mismatch_resolved_if_fixed": "signal_flips_resolved",
        "rows_proba_diff_gt_tol_resolved_if_fixed": "proba_drift_rows_resolved",
        "rows_abs_proba_gap_reduced_if_fixed": "rows_helped",
        "rows_abs_proba_gap_worsened_if_fixed": "rows_hurt",
    }
    if feature_summary_df.empty:
        return pd.DataFrame(columns=slim_columns)

    sort_spec = [
        ("rows_signal_mismatch_resolved_if_fixed", False),
        ("rows_proba_diff_gt_tol_resolved_if_fixed", False),
        ("net_abs_proba_gap_reduction_if_fixed", False),
        ("mean_abs_proba_gap_reduction_on_drift_rows_if_fixed", False),
        ("mean_gap_reduction_ratio_on_drift_rows_if_fixed", False),
        ("mean_abs_proba_shift_if_fixed", False),
        ("max_abs_proba_shift_if_fixed", False),
        ("rows_abs_proba_gap_reduced_if_fixed", False),
        ("feature", True),
    ]
    sort_cols = [
        col for col, _ascending in sort_spec if col in feature_summary_df.columns
    ]
    sort_ascending = [
        ascending for col, ascending in sort_spec if col in feature_summary_df.columns
    ]
    export_df = (
        feature_summary_df.sort_values(
            sort_cols,
            ascending=sort_ascending,
            na_position="last",
            kind="stable",
        )
        .loc[:, export_columns]
        .rename(columns=rename_map)
        .reset_index(drop=True)
    )
    export_df.insert(0, "rank", np.arange(1, len(export_df) + 1, dtype=np.int32))
    if only_impactful and not export_df.empty:
        impact_mask = (
                export_df["rows_helped"].gt(0)
                | export_df["signal_flips_resolved"].gt(0)
                | export_df["proba_drift_rows_resolved"].gt(0)
                | export_df["net_pred_gap_reduction"].gt(PREDICTION_DIFF_TOL)
        )
        export_df = export_df.loc[impact_mask].reset_index(drop=True)
        export_df["rank"] = np.arange(1, len(export_df) + 1, dtype=np.int32)
        export_df = export_df.loc[:, slim_columns]
    if top_k is not None:
        export_df = export_df.head(int(top_k)).reset_index(drop=True)
    return export_df


FEATURES_TO_INSPECT_COLUMNS = [
    "rank",
    "severity",
    "feature",
    "pred_shift_rows",
    "pred_shift_rows_pct",
    "mean_pred_shift",
    "max_pred_shift",
    "rows_where_prediction_diff_exceeds_tol_explained",
    "rows_where_up_down_prediction_flips_explained",
    "max_feature_abs_diff",
    "mean_feature_abs_diff",
    "importance_gain",
]


def _build_features_to_inspect_df(
        feature_summary_df,
        *,
        decision_row_count,
        top_k=None,
):
    if feature_summary_df is None or feature_summary_df.empty:
        return pd.DataFrame(columns=FEATURES_TO_INSPECT_COLUMNS)

    export_df = feature_summary_df.copy()
    if "mean_abs_proba_shift_on_shift_rows_if_fixed" not in export_df.columns:
        export_df["mean_abs_proba_shift_on_shift_rows_if_fixed"] = export_df.get(
            "mean_abs_proba_shift_if_fixed",
            0.0,
        )

    numeric_defaults = {
        "rows_pred_shift_gt_tol_if_fixed": 0,
        "mean_abs_proba_shift_on_shift_rows_if_fixed": 0.0,
        "max_abs_proba_shift_if_fixed": 0.0,
        "rows_proba_diff_gt_tol_resolved_if_fixed": 0,
        "rows_signal_mismatch_resolved_if_fixed": 0,
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "importance_gain": 0.0,
    }
    for column_name, default_value in numeric_defaults.items():
        if column_name not in export_df.columns:
            export_df[column_name] = default_value
        export_df[column_name] = pd.to_numeric(
            export_df[column_name],
            errors="coerce",
        ).fillna(default_value)

    if "feature" not in export_df.columns:
        export_df["feature"] = ""

    pred_shift_rows = export_df["rows_pred_shift_gt_tol_if_fixed"].astype(np.int64)
    pred_diff_rows_explained = export_df[
        "rows_proba_diff_gt_tol_resolved_if_fixed"
    ].astype(np.int64)
    up_down_flip_rows_explained = export_df[
        "rows_signal_mismatch_resolved_if_fixed"
    ].astype(np.int64)
    max_pred_shift = export_df["max_abs_proba_shift_if_fixed"].astype(float)

    inspect_mask = (
            pred_shift_rows.gt(0)
            | pred_diff_rows_explained.gt(0)
            | up_down_flip_rows_explained.gt(0)
            | max_pred_shift.gt(PREDICTION_DIFF_TOL)
    )
    if not bool(inspect_mask.any()):
        return pd.DataFrame(columns=FEATURES_TO_INSPECT_COLUMNS)

    decision_rows = max(int(decision_row_count or 0), 0)
    selected = pd.DataFrame(
        {
            "severity": np.select(
                [
                    up_down_flip_rows_explained.gt(0),
                    pred_diff_rows_explained.gt(0),
                    pred_shift_rows.gt(0),
                ],
                ["critical", "high", "medium"],
                default="low",
            ),
            "feature": export_df["feature"].astype(str),
            "pred_shift_rows": pred_shift_rows,
            "pred_shift_rows_pct": (
                np.where(
                    decision_rows > 0,
                    pred_shift_rows.astype(float) / float(decision_rows) * 100.0,
                    0.0,
                )
            ),
            "mean_pred_shift": export_df[
                "mean_abs_proba_shift_on_shift_rows_if_fixed"
            ].astype(float),
            "max_pred_shift": max_pred_shift,
            "rows_where_prediction_diff_exceeds_tol_explained": (
                pred_diff_rows_explained
            ),
            "rows_where_up_down_prediction_flips_explained": (
                up_down_flip_rows_explained
            ),
            "max_feature_abs_diff": export_df["max_abs_diff"].astype(float),
            "mean_feature_abs_diff": export_df["mean_abs_diff"].astype(float),
            "importance_gain": export_df["importance_gain"].astype(float),
        }
    ).loc[inspect_mask].reset_index(drop=True)

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    selected["_severity_order"] = selected["severity"].map(severity_order).fillna(9)
    selected = selected.sort_values(
        [
            "_severity_order",
            "rows_where_up_down_prediction_flips_explained",
            "rows_where_prediction_diff_exceeds_tol_explained",
            "pred_shift_rows",
            "max_pred_shift",
            "mean_pred_shift",
            "importance_gain",
            "feature",
        ],
        ascending=[True, False, False, False, False, False, False, True],
        kind="stable",
    ).drop(columns=["_severity_order"]).reset_index(drop=True)
    selected.insert(0, "rank", np.arange(1, len(selected) + 1, dtype=np.int32))
    selected = selected.loc[:, FEATURES_TO_INSPECT_COLUMNS]
    if top_k is not None:
        selected = selected.head(int(top_k)).reset_index(drop=True)
    return selected


ROWS_TO_INSPECT_COLUMNS = [
    "Opened",
    "live_proba_up",
    "stored_proba_up",
    "pred_abs_diff",
    "up_down_prediction_flipped",
    "business_decision_changed",
    "policy_decision_changed",
    "top_feature",
    "top_feature_pred_shift",
    "top_feature_live_value",
    "top_feature_stored_value",
    "max_feature_abs_diff",
    "mean_feature_abs_diff",
]


def _build_rows_to_inspect_df(step_summary_df, *, top_k=None):
    if step_summary_df is None or step_summary_df.empty:
        return pd.DataFrame(columns=ROWS_TO_INSPECT_COLUMNS)

    frame = step_summary_df.copy()
    pred_diff = pd.to_numeric(
        frame.get("proba_up_abs_diff", 0.0),
        errors="coerce",
    ).fillna(0.0)
    up_down_flipped = pd.to_numeric(
        frame.get("signal_mismatch", 0),
        errors="coerce",
    ).fillna(0).astype(bool)
    business_changed = pd.to_numeric(
        frame.get("business_decision_mismatch", 0),
        errors="coerce",
    ).fillna(0).astype(bool)
    policy_changed = pd.to_numeric(
        frame.get("policy_decision_mismatch", 0),
        errors="coerce",
    ).fillna(0).astype(bool)
    top_feature = frame.get("top_prediction_impact_feature", "")
    top_feature = pd.Series(top_feature, index=frame.index).fillna("").astype(str)
    top_feature_shift = pd.to_numeric(
        frame.get("top_prediction_impact_abs_proba_shift_if_fixed", 0.0),
        errors="coerce",
    ).fillna(0.0)

    inspect_mask = (
            pred_diff.gt(PREDICTION_DIFF_TOL)
            | up_down_flipped
            | business_changed
            | policy_changed
            | top_feature.ne("")
            | top_feature_shift.gt(PREDICTION_DIFF_TOL)
    )
    if not bool(inspect_mask.any()):
        return pd.DataFrame(columns=ROWS_TO_INSPECT_COLUMNS)

    out = pd.DataFrame(
        {
            "Opened": frame.get("Opened"),
            "live_proba_up": pd.to_numeric(
                frame.get("live_proba_up", np.nan),
                errors="coerce",
            ),
            "stored_proba_up": pd.to_numeric(
                frame.get("stored_proba_up", np.nan),
                errors="coerce",
            ),
            "pred_abs_diff": pred_diff,
            "up_down_prediction_flipped": up_down_flipped,
            "business_decision_changed": business_changed,
            "policy_decision_changed": policy_changed,
            "top_feature": top_feature,
            "top_feature_pred_shift": top_feature_shift,
            "top_feature_live_value": pd.to_numeric(
                frame.get("top_prediction_impact_live_value", np.nan),
                errors="coerce",
            ),
            "top_feature_stored_value": pd.to_numeric(
                frame.get("top_prediction_impact_stored_value", np.nan),
                errors="coerce",
            ),
            "max_feature_abs_diff": pd.to_numeric(
                frame.get("feature_max_abs_diff", 0.0),
                errors="coerce",
            ).fillna(0.0),
            "mean_feature_abs_diff": pd.to_numeric(
                frame.get("feature_mean_abs_diff", 0.0),
                errors="coerce",
            ).fillna(0.0),
        }
    ).loc[inspect_mask].reset_index(drop=True)

    out = out.sort_values(
        [
            "up_down_prediction_flipped",
            "business_decision_changed",
            "policy_decision_changed",
            "pred_abs_diff",
            "top_feature_pred_shift",
            "max_feature_abs_diff",
            "Opened",
        ],
        ascending=[False, False, False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    out = out.loc[:, ROWS_TO_INSPECT_COLUMNS]
    if top_k is not None:
        out = out.head(int(top_k)).reset_index(drop=True)
    return out


def _summary_int(summary, key, default=0):
    value = summary.get(key, default)
    if pd.isna(value):
        return int(default)
    return int(value)


def _summary_float(summary, key, default=0.0):
    value = summary.get(key, default)
    if pd.isna(value):
        return float(default)
    return float(value)


def _build_live_feature_parity_summary_payload(
        report,
        drift_reason_report,
        features_to_inspect_df,
):
    live_summary = report["summary"]
    reason_summary = drift_reason_report["summary"]
    guardrail_violations = _evaluate_guardrail_violations(live_summary)

    proba_drift_rows = _summary_int(live_summary, "rows_with_proba_diff_gt_tol")
    signal_mismatch_rows = _summary_int(live_summary, "rows_with_signal_mismatch")
    business_mismatch_rows = _summary_int(
        live_summary,
        "rows_with_business_decision_mismatch",
    )
    policy_mismatch_rows = _summary_int(live_summary, "rows_with_any_policy_mismatch")
    features_to_inspect_count = int(len(features_to_inspect_df))

    if guardrail_violations:
        verdict = "fail"
    elif (
            signal_mismatch_rows > 0
            or business_mismatch_rows > 0
            or policy_mismatch_rows > 0
            or proba_drift_rows > 0
            or features_to_inspect_count > 0
    ):
        verdict = "inspect"
    else:
        verdict = "ok"

    severity_counts = (
        features_to_inspect_df["severity"]
        .value_counts()
        .reindex(["critical", "high", "medium", "low"], fill_value=0)
        .astype(int)
        .to_dict()
        if not features_to_inspect_df.empty
        else {"critical": 0, "high": 0, "medium": 0, "low": 0}
    )

    payload = {
        "verdict": verdict,
        "proba_diff_tol": PREDICTION_DIFF_TOL,
        "severity": {
            "critical": "feature explains at least one up/down prediction flip",
            "high": "feature explains at least one row where live/stored prediction difference exceeded proba_diff_tol",
            "medium": "feature changes prediction by more than proba_diff_tol",
        },
        "audit_start": live_summary.get("audit_start"),
        "audit_end": live_summary.get("audit_end"),
        "bootstrap_rows": _summary_int(live_summary, "bootstrap_rows"),
        "audit_rows_total_1m": _summary_int(live_summary, "audit_rows_total_1m"),
        "decision_rows": _summary_int(live_summary, "decision_row_count"),
        "feature_count": _summary_int(live_summary, "feature_count"),
        "features_to_inspect": features_to_inspect_count,
        "features_to_inspect_by_severity": severity_counts,
        "proba_drift_rows": proba_drift_rows,
        "max_proba_up_abs_diff": _summary_float(
            live_summary,
            "max_proba_up_abs_diff",
        ),
        "mean_proba_up_abs_diff": _summary_float(
            live_summary,
            "mean_proba_up_abs_diff",
        ),
        "signal_mismatch_rows": signal_mismatch_rows,
        "business_decision_mismatch_rows": business_mismatch_rows,
        "policy_mismatch_rows": policy_mismatch_rows,
        "explanation_basis": reason_summary.get("explanation_basis"),
        "top_features": json.loads(
            features_to_inspect_df.head(10).to_json(orient="records")
        ),
    }
    if guardrail_violations:
        payload["guardrail_violations"] = guardrail_violations
    return payload


def _build_feature_drop_candidates_df(
        feature_summary_df,
        *,
        top_k=None,
):
    if feature_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "feature",
                "group",
                "builder",
                "drop_candidate_reason",
                "importance_gain",
                "importance_split",
                "max_abs_diff",
                "mean_abs_diff",
                "rmse_abs_diff",
                "max_rel_diff",
                "mean_rel_diff",
                "finite_status_mismatch_count",
                "mean_pred_shift",
                "max_pred_shift",
                "signal_flips_resolved",
                "proba_drift_rows_resolved",
            ]
        )

    export_df = feature_summary_df.copy()
    if "builder_name" in export_df.columns:
        export_df["builder"] = export_df["builder_name"]
    elif "builder" not in export_df.columns:
        export_df["builder"] = "unknown"

    if "importance_gain" not in export_df.columns:
        export_df["importance_gain"] = 0.0
    if "importance_split" not in export_df.columns:
        export_df["importance_split"] = 0.0
    if "mean_abs_proba_shift_if_fixed" not in export_df.columns:
        export_df["mean_abs_proba_shift_if_fixed"] = 0.0
    if "max_abs_proba_shift_if_fixed" not in export_df.columns:
        export_df["max_abs_proba_shift_if_fixed"] = 0.0
    if "rows_signal_mismatch_resolved_if_fixed" not in export_df.columns:
        export_df["rows_signal_mismatch_resolved_if_fixed"] = 0
    if "rows_proba_diff_gt_tol_resolved_if_fixed" not in export_df.columns:
        export_df["rows_proba_diff_gt_tol_resolved_if_fixed"] = 0

    export_df["flag_finite_status_mismatch"] = (
            export_df["finite_status_mismatch_count"] > 0
    )
    export_df["flag_max_abs_diff"] = export_df["max_abs_diff"].abs().gt(
        FEATURE_DROP_MAX_ABS_DIFF_TOL
    )
    export_df["flag_mean_abs_diff"] = export_df["mean_abs_diff"].abs().gt(
        FEATURE_DROP_MEAN_ABS_DIFF_TOL
    )
    export_df["flag_max_rel_diff"] = export_df["max_rel_diff"].gt(
        FEATURE_DROP_MAX_REL_DIFF_TOL
    )
    export_df["flag_prediction_impact"] = (
            export_df["max_abs_proba_shift_if_fixed"].gt(PREDICTION_DIFF_TOL)
            | export_df["rows_signal_mismatch_resolved_if_fixed"].gt(0)
            | export_df["rows_proba_diff_gt_tol_resolved_if_fixed"].gt(0)
    )
    export_df["flag_abs_diff_material"] = (
            export_df["flag_max_abs_diff"] | export_df["flag_mean_abs_diff"]
    )
    export_df["drop_candidate"] = (
            export_df["flag_finite_status_mismatch"]
            | export_df["flag_max_rel_diff"]
            | (
                    export_df["flag_abs_diff_material"]
                    & export_df["flag_prediction_impact"]
            )
    )

    reason_cols = [
        ("flag_finite_status_mismatch", "finite_status_mismatch"),
        (
            "flag_max_abs_diff",
            f"max_abs_diff>{FEATURE_DROP_MAX_ABS_DIFF_TOL:g}",
        ),
        (
            "flag_mean_abs_diff",
            f"mean_abs_diff>{FEATURE_DROP_MEAN_ABS_DIFF_TOL:g}",
        ),
        (
            "flag_max_rel_diff",
            f"max_rel_diff>{FEATURE_DROP_MAX_REL_DIFF_TOL:g}",
        ),
    ]
    reasons = []
    for row in export_df.itertuples(index=False):
        parts = [label for flag_name, label in reason_cols if getattr(row, flag_name)]
        reasons.append("; ".join(parts))
    export_df["drop_candidate_reason"] = reasons

    export_df = export_df.loc[export_df["drop_candidate"]].copy()
    export_df = export_df.sort_values(
        [
            "finite_status_mismatch_count",
            "max_rel_diff",
            "mean_rel_diff",
            "max_abs_diff",
            "mean_abs_diff",
            "importance_gain",
            "importance_split",
            "max_abs_proba_shift_if_fixed",
            "mean_abs_proba_shift_if_fixed",
            "feature",
        ],
        ascending=[False, False, False, False, False, False, False, False, False, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    export_df = export_df.loc[
        :,
        [
            "feature",
            "group",
            "builder",
            "drop_candidate_reason",
            "importance_gain",
            "importance_split",
            "max_abs_diff",
            "mean_abs_diff",
            "rmse_abs_diff",
            "max_rel_diff",
            "mean_rel_diff",
            "finite_status_mismatch_count",
            "mean_abs_proba_shift_if_fixed",
            "max_abs_proba_shift_if_fixed",
            "rows_signal_mismatch_resolved_if_fixed",
            "rows_proba_diff_gt_tol_resolved_if_fixed",
        ],
    ].rename(
        columns={
            "mean_abs_proba_shift_if_fixed": "mean_pred_shift",
            "max_abs_proba_shift_if_fixed": "max_pred_shift",
            "rows_signal_mismatch_resolved_if_fixed": "signal_flips_resolved",
            "rows_proba_diff_gt_tol_resolved_if_fixed": "proba_drift_rows_resolved",
        }
    )
    export_df.insert(0, "rank", np.arange(1, len(export_df) + 1, dtype=np.int32))
    if top_k is not None:
        export_df = export_df.head(int(top_k)).reset_index(drop=True)
    return export_df


def save_audit_outputs(
        results,
        *,
        output_dir,
        drilldown_feature_name=None,
        top_n=50,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = {}
    report = results["live_vs_stored_report"]
    drift_reason_report = results["drift_reason_report"]

    written_paths["summary_json"] = output_dir / "live_vs_stored_summary.json"
    written_paths["features_to_inspect_csv"] = output_dir / "features_to_inspect.csv"
    written_paths["rows_to_inspect_csv"] = output_dir / "rows_to_inspect.csv"
    features_to_inspect_df = _build_features_to_inspect_df(
        report["feature_summary_df"],
        decision_row_count=report["summary"].get("decision_row_count", 0),
    )
    rows_to_inspect_df = _build_rows_to_inspect_df(report["step_summary_df"])

    written_paths["summary_json"].write_text(
        json.dumps(
            _build_live_feature_parity_summary_payload(
                report,
                drift_reason_report,
                features_to_inspect_df,
            ),
            indent=2,
            ensure_ascii=True,
            default=str,
        ),
        encoding="utf-8",
    )
    features_to_inspect_df.to_csv(
        written_paths["features_to_inspect_csv"],
        index=False,
    )
    rows_to_inspect_df.to_csv(
        written_paths["rows_to_inspect_csv"],
        index=False,
    )

    if AUDIT_WRITE_DEBUG_CSVS:
        written_paths["debug_decision_rows_csv"] = (
                output_dir / "debug_live_vs_stored_decision_rows.csv"
        )
        written_paths["debug_feature_prediction_impacts_csv"] = (
                output_dir / "debug_live_vs_stored_feature_prediction_impacts.csv"
        )
        written_paths["debug_feature_prediction_impacts_full_csv"] = (
                output_dir / "debug_live_vs_stored_feature_prediction_impacts_full.csv"
        )
        written_paths["debug_feature_drop_candidates_csv"] = (
                output_dir / "debug_live_vs_stored_feature_drop_candidates.csv"
        )
        written_paths["debug_builder_prediction_impacts_csv"] = (
                output_dir / "debug_live_vs_stored_builder_prediction_impacts.csv"
        )
        written_paths["debug_group_prediction_impacts_csv"] = (
                output_dir / "debug_live_vs_stored_group_prediction_impacts.csv"
        )

        decision_sort_spec = [
            ("top_prediction_impact_abs_proba_gap_reduction_if_fixed", False),
            ("proba_up_abs_diff", False),
            ("top_prediction_impact_abs_proba_shift_if_fixed", False),
            ("feature_mean_rel_diff", False),
            ("feature_mean_abs_diff", False),
            ("Opened", True),
        ]
        decision_sort_cols = [
            col
            for col, _ascending in decision_sort_spec
            if col in report["step_summary_df"]
        ]
        decision_sort_ascending = [
            ascending
            for col, ascending in decision_sort_spec
            if col in report["step_summary_df"]
        ]
        decision_rows_export_df = report["step_summary_df"].sort_values(
            decision_sort_cols,
            ascending=decision_sort_ascending,
            kind="stable",
            na_position="last",
        )
        feature_export_df = _build_feature_prediction_impact_export_df(
            report["feature_summary_df"],
            only_impactful=True,
        )
        drop_candidate_df = _build_feature_drop_candidates_df(
            report["feature_summary_df"]
        )

        decision_rows_export_df.to_csv(
            written_paths["debug_decision_rows_csv"],
            index=False,
        )
        feature_export_df.to_csv(
            written_paths["debug_feature_prediction_impacts_csv"],
            index=False,
        )
        report["feature_summary_df"].to_csv(
            written_paths["debug_feature_prediction_impacts_full_csv"],
            index=False,
        )
        drop_candidate_df.to_csv(
            written_paths["debug_feature_drop_candidates_csv"],
            index=False,
        )
        report["builder_summary_df"].to_csv(
            written_paths["debug_builder_prediction_impacts_csv"],
            index=False,
        )
        report["group_summary_df"].to_csv(
            written_paths["debug_group_prediction_impacts_csv"],
            index=False,
        )

    if drilldown_feature_name:
        drilldown_path = (
                output_dir / f"live_vs_stored_drilldown_{drilldown_feature_name}.csv"
        )
        feature_drilldown(
            results,
            drilldown_feature_name,
            report_key="live_vs_stored_report",
            top_n=top_n,
        ).to_csv(drilldown_path, index=False)
        written_paths["drilldown_csv"] = drilldown_path

    return written_paths


def _default_output_dir():
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    return (
            Path("data/analysis/live_feature_parity")
            / str(MODELING_DATASET_SETTINGS["active_asset"])
            / stamp
    )


def run_live_feature_parity_audit():
    started_at = time.perf_counter()
    results = run_live_modeling_feature_audit(
        days_back=AUDIT_DAYS_BACK,
        bootstrap_candles=AUDIT_BOOTSTRAP_CANDLES,
        max_steps=AUDIT_MAX_STEPS,
        max_keep=AUDIT_MAX_KEEP,
        model_meta_path=AUDIT_MODEL_META_PATH,
        parquet_path=AUDIT_PARQUET_PATH,
        use_anchor_vp_state=AUDIT_USE_ANCHOR_VP_STATE,
        overwrite_anchor_vp_state=AUDIT_OVERWRITE_ANCHOR_VP_STATE,
    )
    output_dir = _optional_path(AUDIT_OUTPUT_DIR) or _default_output_dir()
    if AUDIT_PROGRESS_ENABLED:
        print(f"[audit] saving outputs -> {output_dir}")
    written_paths = save_audit_outputs(
        results,
        output_dir=output_dir,
        drilldown_feature_name=AUDIT_DRILLDOWN_FEATURE,
        top_n=AUDIT_TOP_N,
    )
    if AUDIT_PROGRESS_ENABLED:
        print(
            "[audit] finished "
            f"elapsed={_format_duration(time.perf_counter() - started_at)}"
        )

    violations = _print_audit_console_summary(results, written_paths)
    if violations:
        raise RuntimeError(
            "Live-vs-stored audit exceeded configured drift guardrails. "
            "See console summary above."
        )


__all__ = [
    "AuditWindow",
    "DEFAULT_AUDIT_DAYS_BACK",
    "DEFAULT_BOOTSTRAP_CANDLES",
    "DEFAULT_MAX_KEEP",
    "IndicatorSpec",
    "PseudoLiveAuditPredictor",
    "VALUE_BUILDERS",
    "build_or_load_anchor_volume_profile_state",
    "build_live_drift_reason_report",
    "estimate_required_candles",
    "evaluate_feature_stability",
    "feature_drilldown",
    "load_modeling_audit_frame",
    "normalize_opened_to_utc_naive",
    "resolve_audit_window",
    "resolve_anchor_volume_profile_state_path",
    "run_feature_readiness_audit",
    "run_indicator_stability_audit",
    "run_live_feature_parity_audit",
    "run_live_modeling_feature_audit",
    "save_audit_outputs",
]


def run_feature_readiness_audit():
    print("[audit] running indicator stability audit")
    run_indicator_stability_audit()
    print("[audit] running live feature parity audit")
    run_live_feature_parity_audit()


def main():
    run_feature_readiness_audit()


if __name__ == "__main__":
    main()
