from pathlib import Path

from common_config_utils import (
    coerce_path,
    load_json_object,
    path_to_portable_str,
    require_positive_int,
    require_text,
)

CONFIGS_DIR = Path("configs")
DATA_DIR = Path("data")
RAW_DATASETS_DIR = DATA_DIR / "raw_datasets"
MODELING_DATASETS_DIR = DATA_DIR / "modeling_datasets"
DATASETS_CONFIG_PATH = CONFIGS_DIR / "datasets.json"
MODELING_CONFIG_PATH = CONFIGS_DIR / "modeling.json"
INDICATOR_FIT_CONFIG_PATH = CONFIGS_DIR / "indicator_fit.json"
LIVE_CONFIG_PATH = CONFIGS_DIR / "live.json"
ACTIVE_CONFIG_PATH = CONFIGS_DIR / "active.json"

RUNTIME_DIR = CONFIGS_DIR / "runtime"
RUNTIME_ACTIVE_PATH = RUNTIME_DIR / "active.json"


def _require_bool_or_default(payload, key, *, default, source_label):
    if key not in payload:
        return bool(default)
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(
            f"{source_label}.{key} must be a JSON boolean, got: {value!r}"
        )
    return value


def _normalize_train_lgbm_config(raw_config, *, profile_name):
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError(
            f"Modeling profile '{profile_name}' must define 'train_lgbm' as a JSON object."
        )
    return {
        "train_default_model": _require_bool_or_default(
            raw_config,
            "train_default_model",
            default=True,
            source_label="modeling.train_lgbm",
        ),
        "save_oof_predictions": _require_bool_or_default(
            raw_config,
            "save_oof_predictions",
            default=True,
            source_label="modeling.train_lgbm",
        ),
    }


def _normalize_candle_streak_intervals(raw_config, *, profile_name):
    if not isinstance(raw_config, dict) or not raw_config:
        raise ValueError(
            f"Modeling profile '{profile_name}' must define non-empty "
            "'candle_streak_intervals' as a JSON object."
        )

    normalized = {}
    for raw_interval, raw_lag_count in raw_config.items():
        interval = str(raw_interval).strip()
        if not interval:
            raise ValueError(
                f"Modeling profile '{profile_name}' contains an empty "
                "'candle_streak_intervals' key."
            )
        if interval in normalized:
            raise ValueError(
                f"Modeling profile '{profile_name}' defines duplicate candle interval "
                f"after normalization: {interval!r}."
            )
        if isinstance(raw_lag_count, bool) or not isinstance(raw_lag_count, int):
            raise ValueError(
                f"Modeling profile '{profile_name}' must define integer lag counts "
                f"for candle_streak_intervals[{interval!r}], got {raw_lag_count!r}."
            )
        if raw_lag_count < 0:
            raise ValueError(
                f"Modeling profile '{profile_name}' must define non-negative lag "
                f"counts for candle_streak_intervals[{interval!r}], got {raw_lag_count}."
            )
        normalized[interval] = int(raw_lag_count)

    if not normalized:
        raise ValueError(
            f"Modeling profile '{profile_name}' contains no valid candle intervals."
        )
    return normalized


def _require_text_with_fallback(payload, key, fallback_key=None, *, source_label=None):
    if key in payload:
        return require_text(payload, key)
    if fallback_key and fallback_key in payload:
        return require_text(payload, fallback_key)
    label = source_label or key
    if fallback_key:
        raise ValueError(
            f"Missing required config key: {key} (fallback: {fallback_key}) in {label}"
        )
    raise ValueError(f"Missing required config key: {label}")


def _load_profiles(config_path):
    payload = load_json_object(config_path)
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(
            f"Missing or invalid 'profiles' object in config: {Path(config_path)}"
        )
    return payload, profiles


def _load_named_profile(config_path, profile_name):
    _, profiles = _load_profiles(config_path)
    if profile_name not in profiles:
        available = ", ".join(sorted(str(key) for key in profiles))
        raise ValueError(
            f"Profile '{profile_name}' not found in {Path(config_path)}. "
            f"Available: {available}"
        )
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise ValueError(
            f"Profile '{profile_name}' in {Path(config_path)} must be a JSON object."
        )
    return dict(profile)


def load_active_profile_names(config_path=ACTIVE_CONFIG_PATH):
    payload = load_json_object(config_path)
    return {
        "dataset_profile": require_text(payload, "dataset_profile"),
        "modeling_profile": require_text(payload, "modeling_profile"),
        "indicator_fit_profile": require_text(payload, "indicator_fit_profile"),
        "live_profile": require_text(payload, "live_profile"),
    }


def load_dataset_profile(profile_name=None, *, active_config_path=ACTIVE_CONFIG_PATH):
    if profile_name is None:
        profile_name = load_active_profile_names(active_config_path)["dataset_profile"]
    profile = _load_named_profile(DATASETS_CONFIG_PATH, profile_name)
    required_keys = (
        "symbol",
        "interval",
        "market",
        "source",
        "price_source",
        "volume_source",
        "volume_symbol",
        "volume_market",
        "base_data_file",
    )
    for key in required_keys:
        require_text(profile, key)
    profile["raw_data_dir"] = _require_text_with_fallback(
        profile,
        "raw_data_dir",
        "data_dir",
        source_label=f"dataset profile '{profile_name}'",
    )
    intervals = profile.get("intervals")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError(
            f"Dataset profile '{profile_name}' must define non-empty 'intervals'."
        )
    profile["intervals"] = [
        str(value).strip() for value in intervals if str(value).strip()
    ]
    if not profile["intervals"]:
        raise ValueError(
            f"Dataset profile '{profile_name}' contains no valid interval values."
        )
    profile["quiet"] = bool(profile.get("quiet", False))
    profile["start_date"] = str(profile.get("start_date", "") or "").strip()
    profile["end_date"] = str(profile.get("end_date", "") or "").strip()
    profile["raw_ohlcv_repair"] = profile.get("raw_ohlcv_repair")
    profile["data_dir"] = profile["raw_data_dir"]
    return profile


def load_modeling_profile(profile_name=None, *, active_config_path=ACTIVE_CONFIG_PATH):
    if profile_name is None:
        profile_name = load_active_profile_names(active_config_path)["modeling_profile"]
    profile = _load_named_profile(MODELING_CONFIG_PATH, profile_name)
    feature_selection = profile.get("feature_selection")
    if not isinstance(feature_selection, dict):
        raise ValueError(
            f"Modeling profile '{profile_name}' must define 'feature_selection'."
        )
    selection_mode = str(feature_selection.get("mode", "")).strip().lower()
    if selection_mode not in {"artifact", "none"}:
        raise ValueError(
            f"Modeling profile '{profile_name}' has unsupported feature_selection.mode="
            f"{feature_selection.get('mode')!r}. Expected 'artifact' or 'none'."
        )
    if selection_mode == "artifact":
        require_text(feature_selection, "artifact_path")
    profile["output_dir"] = str(
        profile.get("output_dir") or path_to_portable_str(MODELING_DATASETS_DIR)
    ).strip()
    if not profile["output_dir"]:
        raise ValueError(
            f"Modeling profile '{profile_name}' must define non-empty 'output_dir'."
        )
    require_text(profile, "output_suffix")
    require_text(profile, "fit_results_dir")
    require_positive_int(profile, "preview_rows")
    profile["candle_streak_intervals"] = _normalize_candle_streak_intervals(
        profile.get("candle_streak_intervals"),
        profile_name=profile_name,
    )
    profile["feature_selection"] = dict(feature_selection)
    profile["train_lgbm"] = _normalize_train_lgbm_config(
        profile.get("train_lgbm"),
        profile_name=profile_name,
    )
    return profile


def load_indicator_fit_profile(
    profile_name=None, *, active_config_path=ACTIVE_CONFIG_PATH
):
    if profile_name is None:
        profile_name = load_active_profile_names(active_config_path)[
            "indicator_fit_profile"
        ]
    profile = _load_named_profile(INDICATOR_FIT_CONFIG_PATH, profile_name)
    indicators = profile.get("indicators")
    if not isinstance(indicators, list) or not indicators:
        raise ValueError(
            f"Indicator-fit profile '{profile_name}' must define non-empty 'indicators'."
        )
    require_text(profile, "proxy_target_price_col")
    horizons = profile.get("proxy_target_horizonts")
    if not isinstance(horizons, list) or not horizons:
        raise ValueError(
            f"Indicator-fit profile '{profile_name}' must define non-empty "
            "'proxy_target_horizonts'."
        )
    metric = profile.get("metric")
    if not isinstance(metric, dict):
        raise ValueError(
            f"Indicator-fit profile '{profile_name}' must define 'metric'."
        )
    for key in (
        "name",
        "segments_count",
        "train_frac",
        "gap",
        "q_ext",
        "q_mid",
        "stat",
        "clip_q",
        "min_bucket_size",
        "min_valid_segments",
    ):
        if key not in metric:
            raise ValueError(
                f"Indicator-fit profile '{profile_name}' missing metric.{key}."
            )
    profile["indicators"] = [
        str(value).strip() for value in indicators if str(value).strip()
    ]
    if not profile["indicators"]:
        raise ValueError(
            f"Indicator-fit profile '{profile_name}' contains no valid indicators."
        )
    profile["proxy_target_mode"] = str(
        profile.get("proxy_target_mode", "ahead_ret")
    ).strip().lower()
    if profile["proxy_target_mode"] not in {"ahead_ret", "candle_up"}:
        raise ValueError(
            f"Indicator-fit profile '{profile_name}' has unsupported "
            f"proxy_target_mode={profile['proxy_target_mode']!r}. "
            "Expected 'ahead_ret' or 'candle_up'."
        )
    profile["proxy_target_time_col"] = str(
        profile.get("proxy_target_time_col", "Opened")
    ).strip()
    if not profile["proxy_target_time_col"]:
        raise ValueError(
            f"Indicator-fit profile '{profile_name}' must define non-empty "
            "'proxy_target_time_col'."
        )
    return profile


def load_live_profile(profile_name=None, *, active_config_path=ACTIVE_CONFIG_PATH):
    if profile_name is None:
        profile_name = load_active_profile_names(active_config_path)["live_profile"]
    profile = _load_named_profile(LIVE_CONFIG_PATH, profile_name)
    dataset = load_dataset_profile(active_config_path=active_config_path)
    for key in (
        "settlement_source",
        "settlement_ticker",
        "default_price_source",
        "default_volume_source",
    ):
        profile.pop(key, None)
    required_text_keys = (
        "polymarket_gamma_host",
        "polymarket_series_slug",
        "polymarket_market_slug_prefix",
    )
    for key in required_text_keys:
        require_text(profile, key)
    live_symbol = str(profile.get("symbol", "") or "").strip().upper()
    if live_symbol and live_symbol != str(dataset["symbol"]).strip().upper():
        raise ValueError(
            f"Live profile '{profile_name}' symbol={live_symbol!r} does not match "
            f"dataset symbol={dataset['symbol']!r}. Live market data must match the "
            "active modeling dataset."
        )
    live_interval = str(profile.get("interval", "") or "").strip()
    if live_interval and live_interval != str(dataset["interval"]).strip():
        raise ValueError(
            f"Live profile '{profile_name}' interval={live_interval!r} does not match "
            f"dataset interval={dataset['interval']!r}. Live market data must match "
            "the active modeling dataset."
        )
    profile["symbol"] = dataset["symbol"]
    profile["interval"] = dataset["interval"]
    if "polymarket_market_slug_override" not in profile:
        profile["polymarket_market_slug_override"] = ""
    profile.setdefault("polymarket_clob_host", "https://clob.polymarket.com")
    profile.setdefault("polymarket_data_api_host", "https://data-api.polymarket.com")
    profile.setdefault("polymarket_relayer_host", "https://relayer-v2.polymarket.com")
    profile.setdefault("polymarket_paper_mode", True)
    profile.setdefault("polymarket_disable_order_submission", False)
    profile.setdefault("polymarket_signature_type", 2)
    profile.setdefault("polymarket_chain_id", 137)
    profile.setdefault("polymarket_max_exposure_usdc", float("inf"))
    profile.setdefault("polymarket_max_bankroll_usdc", float("inf"))
    profile.setdefault("polymarket_start_bankroll_usdc", 100.0)
    profile.setdefault("polymarket_no_trade_last_seconds", 20)
    profile.setdefault("polymarket_clob_http_timeout_sec", profile["polymarket_market_request_timeout_sec"])
    profile.setdefault("polymarket_market_lookup_max_wait_ms", 2500)
    profile.setdefault("polymarket_market_lookup_retry_ms", 100)
    profile.setdefault("polymarket_market_lookup_prefetch_lead_ms", 1200)
    profile.setdefault("polymarket_market_lookup_prefetch_max_age_ms", 2500)
    profile.setdefault("polymarket_execution_mode", "fok")
    profile.setdefault("polymarket_order_price_cap", 0.56)
    profile.setdefault("polymarket_import_untracked_open_positions", False)
    profile.setdefault("polymarket_enable_exit_orders", True)
    profile.setdefault("polymarket_exit_min_profit_usdc", 0.15)
    profile.setdefault("polymarket_exit_min_roi", 0.01)
    profile.setdefault("polymarket_exit_min_seconds_to_close", 45)
    profile.setdefault("polymarket_exit_redeem_profit_tolerance", 0.01)
    profile.setdefault("polymarket_redeem_resolved_positions", True)
    return profile


def load_modeling_settings(*, active_config_path=ACTIVE_CONFIG_PATH):
    dataset = load_dataset_profile(active_config_path=active_config_path)
    modeling = load_modeling_profile(active_config_path=active_config_path)
    feature_selection = modeling["feature_selection"]
    selection_mode = str(feature_selection["mode"]).strip().lower()
    feature_subset_path = None
    feature_subset_list_key = None
    if selection_mode == "artifact":
        feature_subset_path = coerce_path(
            require_text(feature_selection, "artifact_path")
        )
        raw_list_key = str(feature_selection.get("artifact_list_key", "") or "").strip()
        feature_subset_list_key = raw_list_key or None
    excluded_feature_names = feature_selection.get("excluded_feature_names", [])
    if not isinstance(excluded_feature_names, list):
        raise ValueError(
            "modeling.feature_selection.excluded_feature_names must be a JSON array."
        )
    return {
        "raw_data_dir": coerce_path(require_text(dataset, "raw_data_dir")),
        "data_dir": coerce_path(require_text(dataset, "raw_data_dir")),
        "base_data_file": require_text(dataset, "base_data_file"),
        "modeling_output_dir": coerce_path(require_text(modeling, "output_dir")),
        "output_suffix": require_text(modeling, "output_suffix"),
        "fit_results_dir": coerce_path(require_text(modeling, "fit_results_dir")),
        "preview_rows": require_positive_int(modeling, "preview_rows"),
        "candle_streak_intervals": dict(modeling["candle_streak_intervals"]),
        "feature_subset_path": feature_subset_path,
        "feature_subset_list_key": feature_subset_list_key,
        "excluded_feature_names": tuple(str(value) for value in excluded_feature_names),
        "float_precision": require_text(modeling, "float_precision"),
        "volume_profile_fixed_range": modeling.get("volume_profile_fixed_range"),
        "drop_frozen_ohlc_blocks": modeling.get("drop_frozen_ohlc_blocks"),
        "train_lgbm": dict(modeling["train_lgbm"]),
    }


def load_fetch_settings(*, active_config_path=ACTIVE_CONFIG_PATH):
    dataset = load_dataset_profile(active_config_path=active_config_path)
    return {
        "symbol": dataset["symbol"],
        "market": dataset["market"],
        "source": dataset["source"],
        "price_source": dataset["price_source"],
        "volume_source": dataset["volume_source"],
        "volume_symbol": dataset["volume_symbol"],
        "volume_market": dataset["volume_market"],
        "intervals": list(dataset["intervals"]),
        "raw_ohlcv_repair": dataset.get("raw_ohlcv_repair"),
        "start_date": dataset["start_date"],
        "end_date": dataset["end_date"],
        "quiet": bool(dataset["quiet"]),
    }


def build_indicator_fit_legacy_config(*, active_config_path=ACTIVE_CONFIG_PATH):
    active = load_active_profile_names(active_config_path)
    dataset = load_dataset_profile(active_config_path=active_config_path)
    fit = load_indicator_fit_profile(active_config_path=active_config_path)
    metric = dict(fit["metric"])
    return {
        "pairs": {
            active["indicator_fit_profile"]: {
                "proxy_target_horizonts": list(fit["proxy_target_horizonts"]),
                "proxy_target_price_col": str(fit["proxy_target_price_col"]),
                "proxy_target_mode": str(fit["proxy_target_mode"]),
                "proxy_target_time_col": str(fit["proxy_target_time_col"]),
                "metric_name": str(metric["name"]),
                "metric_segments_count": int(metric["segments_count"]),
                "metric_train_frac": float(metric["train_frac"]),
                "metric_gap": int(metric["gap"]),
                "q_ext": metric["q_ext"],
                "q_mid": metric["q_mid"],
                "stat": str(metric["stat"]),
                "clip_q": float(metric["clip_q"]),
                "min_bucket_size": int(metric["min_bucket_size"]),
                "min_valid_segments": int(metric["min_valid_segments"]),
                "base_pop_size": int(fit["base_pop_size"]),
                "drop_frozen_ohlc_blocks": fit.get("drop_frozen_ohlc_blocks"),
                "intervals": {
                    dataset["interval"]: {
                        "data_path": path_to_portable_str(dataset["raw_data_dir"]),
                        "data_file": dataset["base_data_file"],
                        "indicators": list(fit["indicators"]),
                    }
                },
            }
        }
    }


def load_runtime_artifact_paths(runtime_manifest_path=RUNTIME_ACTIVE_PATH):
    payload = load_json_object(runtime_manifest_path)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(
            f"Missing or invalid 'artifacts' object in runtime manifest: {runtime_manifest_path}"
        )
    trade_policy_runtime_config_path = coerce_path(
        require_text(artifacts, "trade_policy_runtime_config_path")
    )
    return {
        "model_meta_path": coerce_path(require_text(artifacts, "model_meta_path")),
        "trade_policy_runtime_config_path": trade_policy_runtime_config_path,
        "indicator_history_requirements_path": coerce_path(
            require_text(artifacts, "indicator_history_requirements_path")
        ),
    }
