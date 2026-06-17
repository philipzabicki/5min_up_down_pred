from pathlib import Path

from utils.config import (
    coerce_path,
    load_json_object,
    path_to_portable_str,
    require_positive_int,
    require_text,
)

CONFIGS_DIR = Path("configs")
DATA_DIR = Path("data")
DATASETS_DIR = DATA_DIR / "datasets"
RAW_DATASETS_DIR = DATASETS_DIR / "raw"
MODELING_DATASETS_DIR = DATASETS_DIR / "modeling"
DATASETS_CONFIG_PATH = CONFIGS_DIR / "datasets.json"
MODELING_CONFIG_PATH = CONFIGS_DIR / "modeling.json"
INDICATOR_FIT_CONFIG_PATH = CONFIGS_DIR / "indicator_fit.json"
LIVE_CONFIG_PATH = CONFIGS_DIR / "live.json"
ACTIVE_CONFIG_PATH = CONFIGS_DIR / "active.json"

RUNTIME_DIR = CONFIGS_DIR / "runtime"
RUNTIME_ACTIVE_PATH = RUNTIME_DIR / "active.json"

DEFAULT_WALK_FORWARD_TEST_TO_TRAIN_RATIO = 0.1
ASSET_PLACEHOLDER = "{asset}"


def normalize_asset_name(value, *, source_label="active_asset"):
    asset = str(value).strip().upper()
    if not asset:
        raise ValueError(f"Config key '{source_label}' cannot be empty")
    if any(ch in asset for ch in "\\/{} "):
        raise ValueError(
            f"Config key '{source_label}' must be a compact asset code, got: {value!r}"
        )
    return asset


def format_asset_text(value, asset):
    return str(value).replace(ASSET_PLACEHOLDER, normalize_asset_name(asset))


def active_asset_path(path_template, *, active_config_path=ACTIVE_CONFIG_PATH):
    return coerce_path(
        format_asset_text(path_template, load_active_asset(active_config_path))
    )


def _require_bool_or_default(payload, key, *, default, source_label):
    if key not in payload:
        return bool(default)
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(
            f"{source_label}.{key} must be a JSON boolean, got: {value!r}"
        )
    return value


def _normalize_lgbm_monotone_constraints(raw_config, *, source_label):
    if raw_config is None:
        return {}
    if not isinstance(raw_config, dict):
        raise ValueError(
            f"{source_label}.monotone_constraints must be a JSON object mapping "
            "feature names to -1, 0, or 1."
        )

    normalized = {}
    for raw_feature_name, raw_direction in raw_config.items():
        feature_name = str(raw_feature_name).strip()
        if not feature_name:
            raise ValueError(
                f"{source_label}.monotone_constraints contains an empty feature name."
            )
        if isinstance(raw_direction, bool) or not isinstance(raw_direction, int):
            raise ValueError(
                f"{source_label}.monotone_constraints[{feature_name!r}] must be "
                f"-1, 0, or 1, got: {raw_direction!r}"
            )
        if raw_direction not in (-1, 0, 1):
            raise ValueError(
                f"{source_label}.monotone_constraints[{feature_name!r}] must be "
                f"-1, 0, or 1, got: {raw_direction!r}"
            )
        normalized[feature_name] = int(raw_direction)
    return normalized


def _normalize_train_lgbm_config(raw_config, *, profile_name):
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError(
            f"Modeling profile '{profile_name}' must define 'train_lgbm' as a JSON object."
        )
    source_label = "modeling.train_lgbm"
    raw_walk_forward_ratio = raw_config.get(
        "walk_forward_test_to_train_ratio",
        DEFAULT_WALK_FORWARD_TEST_TO_TRAIN_RATIO,
    )
    walk_forward_ratio = float(raw_walk_forward_ratio)
    if not (0.0 < walk_forward_ratio < 1.0):
        raise ValueError(
            f"{source_label}.walk_forward_test_to_train_ratio must be in (0, 1), "
            f"got: {raw_walk_forward_ratio!r}"
        )
    return {
        "train_default_model": _require_bool_or_default(
            raw_config,
            "train_default_model",
            default=True,
            source_label=source_label,
        ),
        "save_oof_predictions": _require_bool_or_default(
            raw_config,
            "save_oof_predictions",
            default=True,
            source_label=source_label,
        ),
        "monotone_constraints": _normalize_lgbm_monotone_constraints(
            raw_config.get("monotone_constraints"),
            source_label=source_label,
        ),
        "walk_forward_test_to_train_ratio": walk_forward_ratio,
    }


def _normalize_indicator_quantile_pairs(raw_pairs, *, profile_name):
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError(
            f"Indicator-fit profile '{profile_name}' metric.quantile_pairs must be "
            "a non-empty JSON array."
        )

    out = []
    for index, raw_pair in enumerate(raw_pairs):
        if not isinstance(raw_pair, dict):
            raise ValueError(
                f"Indicator-fit profile '{profile_name}' metric.quantile_pairs[{index}] "
                "must be a JSON object."
            )
        if "q_ext" not in raw_pair or "q_mid" not in raw_pair:
            raise ValueError(
                f"Indicator-fit profile '{profile_name}' metric.quantile_pairs[{index}] "
                "must define q_ext and q_mid."
            )

        q_ext = float(raw_pair["q_ext"])
        q_mid = float(raw_pair["q_mid"])
        if not (0.0 < q_ext < 0.5):
            raise ValueError(
                f"Indicator-fit profile '{profile_name}' metric.quantile_pairs[{index}]."
                f"q_ext must satisfy 0 < value < 0.5, got: {q_ext}"
            )
        if not (0.0 < q_mid < 0.5):
            raise ValueError(
                f"Indicator-fit profile '{profile_name}' metric.quantile_pairs[{index}]."
                f"q_mid must satisfy 0 < value < 0.5, got: {q_mid}"
            )
        if not (0.5 - q_mid > q_ext):
            raise ValueError(
                f"Indicator-fit profile '{profile_name}' metric.quantile_pairs[{index}] "
                "is invalid: require 0.5 - q_mid > q_ext so mid band does not "
                f"overlap extremes. Got q_ext={q_ext}, q_mid={q_mid}"
            )
        if not any(
                abs(q_ext - prev["q_ext"]) <= 1e-12
                and abs(q_mid - prev["q_mid"]) <= 1e-12
                for prev in out
        ):
            out.append({"q_ext": q_ext, "q_mid": q_mid})
    return out


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


def load_active_asset(config_path=ACTIVE_CONFIG_PATH):
    payload = load_json_object(config_path)
    if "active_asset" in payload:
        return normalize_asset_name(require_text(payload, "active_asset"))
    if "dataset_profile" in payload:
        # Backward-compatible fallback for older local configs. New configs should
        # use active_asset explicitly and name dataset profiles by asset code.
        return normalize_asset_name(str(require_text(payload, "dataset_profile")).split("_", 1)[0])
    raise ValueError(f"Missing required config key: active_asset in {Path(config_path)}")


def load_active_profile_names(config_path=ACTIVE_CONFIG_PATH):
    payload = load_json_object(config_path)
    active_asset = load_active_asset(config_path)
    modeling_profile = (
        require_text(payload, "modeling_profile")
        if "modeling_profile" in payload
        else active_asset
    )
    return {
        "active_asset": active_asset,
        "dataset_profile": str(payload.get("dataset_profile") or active_asset).strip(),
        "modeling_profile": modeling_profile,
        "indicator_fit_profile": require_text(payload, "indicator_fit_profile"),
        "live_profile": require_text(payload, "live_profile"),
    }


def load_dataset_profile(profile_name=None, *, active_config_path=ACTIVE_CONFIG_PATH):
    active = load_active_profile_names(active_config_path)
    active_asset = active["active_asset"]
    if profile_name is None:
        profile_name = active["dataset_profile"]
    profile = _load_named_profile(DATASETS_CONFIG_PATH, profile_name)
    profile_asset = normalize_asset_name(profile.get("asset", active_asset))
    if profile_name == active["dataset_profile"] and profile_asset != active_asset:
        raise ValueError(
            f"Active asset {active_asset!r} does not match dataset profile "
            f"{profile_name!r} asset={profile_asset!r}."
        )
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
    profile["raw_data_dir"] = format_asset_text(profile["raw_data_dir"], profile_asset)
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
    profile["asset"] = profile_asset
    profile["profile_name"] = str(profile_name)
    return profile


def load_modeling_profile(profile_name=None, *, active_config_path=ACTIVE_CONFIG_PATH):
    active = load_active_profile_names(active_config_path)
    active_asset = active["active_asset"]
    if profile_name is None:
        profile_name = active["modeling_profile"]
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
    profile["output_dir"] = format_asset_text(profile["output_dir"], active_asset)
    if not profile["output_dir"]:
        raise ValueError(
            f"Modeling profile '{profile_name}' must define non-empty 'output_dir'."
        )
    require_text(profile, "output_suffix")
    profile["fit_results_dir"] = format_asset_text(
        require_text(profile, "fit_results_dir"),
        active_asset,
    )
    require_positive_int(profile, "preview_rows")
    profile["candle_streak_intervals"] = _normalize_candle_streak_intervals(
        profile.get("candle_streak_intervals"),
        profile_name=profile_name,
    )
    profile["feature_intervals"] = dict(profile.get("feature_intervals") or {})
    profile["basis_premium_features"] = dict(
        profile.get("basis_premium_features") or {}
    )
    feature_selection = dict(feature_selection)
    if str(feature_selection.get("artifact_path", "") or "").strip():
        feature_selection["artifact_path"] = format_asset_text(
            feature_selection["artifact_path"],
            active_asset,
        )
    profile["feature_selection"] = feature_selection
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
            "quantile_pairs",
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
    metric["quantile_pairs"] = _normalize_indicator_quantile_pairs(
        metric["quantile_pairs"],
        profile_name=profile_name,
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


def load_live_profile(
        profile_name=None,
        *,
        active_config_path=ACTIVE_CONFIG_PATH,
        dataset_profile_name=None,
):
    if profile_name is None:
        profile_name = load_active_profile_names(active_config_path)["live_profile"]
    profile = _load_named_profile(LIVE_CONFIG_PATH, profile_name)
    dataset = load_dataset_profile(
        dataset_profile_name,
        active_config_path=active_config_path,
    )
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
    active = load_active_profile_names(active_config_path)
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
        "active_asset": active["active_asset"],
        "symbol": dataset["symbol"],
        "interval": dataset["interval"],
        "market": dataset["market"],
        "volume_symbol": dataset["volume_symbol"],
        "volume_market": dataset["volume_market"],
        "raw_data_dir": coerce_path(require_text(dataset, "raw_data_dir")),
        "data_dir": coerce_path(require_text(dataset, "raw_data_dir")),
        "base_data_file": require_text(dataset, "base_data_file"),
        "modeling_output_dir": coerce_path(require_text(modeling, "output_dir")),
        "output_suffix": require_text(modeling, "output_suffix"),
        "fit_results_dir": coerce_path(require_text(modeling, "fit_results_dir")),
        "preview_rows": require_positive_int(modeling, "preview_rows"),
        "candle_streak_intervals": dict(modeling["candle_streak_intervals"]),
        "feature_intervals": dict(modeling.get("feature_intervals") or {}),
        "basis_premium_features": dict(
            modeling.get("basis_premium_features") or {}
        ),
        "feature_subset_path": feature_subset_path,
        "feature_subset_list_key": feature_subset_list_key,
        "excluded_feature_names": tuple(str(value) for value in excluded_feature_names),
        "float_precision": require_text(modeling, "float_precision"),
        "volume_profile_fixed_range": modeling.get("volume_profile_fixed_range"),
        "drop_frozen_ohlc_blocks": modeling.get("drop_frozen_ohlc_blocks"),
        "train_lgbm": dict(modeling["train_lgbm"]),
    }


def load_fetch_settings(*, active_config_path=ACTIVE_CONFIG_PATH):
    active = load_active_profile_names(active_config_path)
    dataset = load_dataset_profile(active_config_path=active_config_path)
    return {
        "active_asset": active["active_asset"],
        "raw_data_dir": dataset["raw_data_dir"],
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


def build_indicator_fit_config(*, active_config_path=ACTIVE_CONFIG_PATH):
    active = load_active_profile_names(active_config_path)
    dataset = load_dataset_profile(active_config_path=active_config_path)
    fit = load_indicator_fit_profile(active_config_path=active_config_path)
    metric = dict(fit["metric"])
    metric_recency_weighting = dict(metric.get("recency_weighting") or {})
    return {
        "pairs": {
            f"{active['active_asset']}_{active['indicator_fit_profile']}": {
                "proxy_target_horizonts": list(fit["proxy_target_horizonts"]),
                "proxy_target_price_col": str(fit["proxy_target_price_col"]),
                "proxy_target_mode": str(fit["proxy_target_mode"]),
                "proxy_target_time_col": str(fit["proxy_target_time_col"]),
                "metric_name": str(metric["name"]),
                "metric_segments_count": int(metric["segments_count"]),
                "metric_train_frac": float(metric["train_frac"]),
                "metric_gap": int(metric["gap"]),
                "quantile_pairs": list(metric["quantile_pairs"]),
                "stat": str(metric["stat"]),
                "clip_q": float(metric["clip_q"]),
                "min_bucket_size": int(metric["min_bucket_size"]),
                "min_valid_segments": int(metric["min_valid_segments"]),
                "metric_recency_weighting_enabled": bool(
                    metric_recency_weighting.get("enabled", False)
                ),
                "metric_recency_weighting_mode": str(
                    metric_recency_weighting.get("mode", "linear")
                ),
                "metric_recency_weight_min": float(
                    metric_recency_weighting.get("min_weight", 1.0)
                ),
                "metric_recency_weight_max": float(
                    metric_recency_weighting.get("max_weight", 1.5)
                ),
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


def _format_runtime_path(value, asset, *, source_label):
    text = str(value).strip()
    if ASSET_PLACEHOLDER in text:
        if asset is None:
            raise ValueError(
                f"{source_label} uses {ASSET_PLACEHOLDER!r} but no runtime asset "
                "was selected. Pass asset=... or define the path under "
                "assets.<asset> in the runtime manifest."
            )
        text = format_asset_text(text, asset)
    elif asset is not None:
        text = format_asset_text(text, asset)
    return coerce_path(text)


def _normalize_runtime_artifacts(artifacts, asset, *, source_label):
    if not isinstance(artifacts, dict):
        raise ValueError(
            f"Missing or invalid 'artifacts' object in runtime manifest: {source_label}"
        )
    if "trade_policy_runtime_config_path" in artifacts:
        raise ValueError(
            "Runtime manifest uses deprecated artifacts.trade_policy_runtime_config_path. "
            "Use artifacts.trade_policy_path as the single active trade policy path."
        )
    indicator_requirements_path = str(
        artifacts.get("indicator_history_requirements_path") or ""
    ).strip()
    if not indicator_requirements_path:
        if asset is None:
            raise ValueError(
                "Missing required config key: indicator_history_requirements_path"
            )
        indicator_requirements_path = (
            f"data/runtime/{normalize_asset_name(asset)}/indicator_history_requirements.json"
        )
    return {
        "model_meta_path": _format_runtime_path(
            require_text(artifacts, "model_meta_path"),
            asset,
            source_label=f"{source_label}.artifacts.model_meta_path",
        ),
        "trade_policy_path": _format_runtime_path(
            require_text(artifacts, "trade_policy_path"),
            asset,
            source_label=f"{source_label}.artifacts.trade_policy_path",
        ),
        "indicator_history_requirements_path": _format_runtime_path(
            indicator_requirements_path,
            asset,
            source_label=(
                f"{source_label}.artifacts.indicator_history_requirements_path"
            ),
        ),
    }


def _runtime_asset_enabled(entry, *, source_label):
    if "enabled" not in entry:
        return True
    enabled = entry["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError(
            f"{source_label}.enabled must be a JSON boolean, got: {enabled!r}"
        )
    return bool(enabled)


def _load_runtime_asset_entries(payload, *, runtime_manifest_path):
    if "trade_policy_presets" in payload:
        raise ValueError(
            "Runtime manifest must not define trade_policy_presets. "
            "Only artifacts.trade_policy_path is active for live runtime."
        )
    assets = payload.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError(
            f"Runtime manifest {runtime_manifest_path} must define non-empty "
            "'assets' object."
        )

    normalized_assets = {}
    for raw_asset, raw_entry in assets.items():
        runtime_asset = normalize_asset_name(
            raw_asset,
            source_label=f"{runtime_manifest_path}.assets key",
        )
        if runtime_asset in normalized_assets:
            raise ValueError(
                f"Runtime manifest {runtime_manifest_path} defines duplicate asset "
                f"after normalization: {runtime_asset!r}."
            )
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"Runtime manifest asset {runtime_asset!r} must be a JSON object."
            )
        entry = dict(raw_entry)
        entry_asset = entry.get("asset")
        if entry_asset is not None:
            entry_asset = normalize_asset_name(
                entry_asset,
                source_label=f"assets.{runtime_asset}.asset",
            )
            if entry_asset != runtime_asset:
                raise ValueError(
                    f"Runtime manifest asset key {runtime_asset!r} does not match "
                    f"entry asset={entry_asset!r}."
                )
        normalized_assets[runtime_asset] = entry
    return normalized_assets


def _build_runtime_asset_settings(asset, entry, *, source_label):
    enabled = _runtime_asset_enabled(entry, source_label=source_label)
    dataset_profile = str(entry.get("dataset_profile") or asset or "").strip()
    live_profile = str(entry.get("live_profile") or "").strip() or None
    if not enabled:
        return {
            "asset": asset,
            "enabled": False,
            "dataset_profile": dataset_profile,
            "live_profile": live_profile,
            "artifacts": None,
        }
    if not dataset_profile:
        raise ValueError(f"Runtime manifest {source_label} must define dataset_profile.")
    artifacts = _normalize_runtime_artifacts(
        entry.get("artifacts"),
        asset,
        source_label=source_label,
    )
    return {
        "asset": asset,
        "enabled": enabled,
        "dataset_profile": dataset_profile,
        "live_profile": live_profile,
        "artifacts": artifacts,
    }


def _load_runtime_asset_settings_map(runtime_manifest_path):
    payload = load_json_object(runtime_manifest_path)
    entries = _load_runtime_asset_entries(
        payload,
        runtime_manifest_path=runtime_manifest_path,
    )
    return {
        runtime_asset: _build_runtime_asset_settings(
            runtime_asset,
            entry,
            source_label=f"{runtime_manifest_path}.assets.{runtime_asset}",
        )
        for runtime_asset, entry in entries.items()
    }


def load_enabled_runtime_asset_settings(
        *,
        runtime_manifest_path=RUNTIME_ACTIVE_PATH,
):
    settings = _load_runtime_asset_settings_map(runtime_manifest_path)
    enabled = {
        runtime_asset: runtime_settings
        for runtime_asset, runtime_settings in settings.items()
        if runtime_settings["enabled"]
    }
    if not enabled:
        raise ValueError(
            f"Runtime manifest {runtime_manifest_path} has no enabled assets."
        )
    return enabled


def load_runtime_asset_settings(
        asset=None,
        *,
        runtime_manifest_path=RUNTIME_ACTIVE_PATH,
):
    settings = _load_runtime_asset_settings_map(runtime_manifest_path)
    if asset is None:
        enabled = {
            runtime_asset: runtime_settings
            for runtime_asset, runtime_settings in settings.items()
            if runtime_settings["enabled"]
        }
        if len(enabled) == 1:
            return next(iter(enabled.values()))
        if not enabled:
            raise ValueError(
                f"Runtime manifest {runtime_manifest_path} has no enabled assets."
            )
        available = ", ".join(sorted(enabled))
        raise ValueError(
            f"Runtime manifest {runtime_manifest_path} defines multiple enabled "
            f"assets: {available}. Use load_enabled_runtime_asset_settings() to "
            "run all enabled assets, or pass asset=... for one runtime process."
        )

    runtime_asset = normalize_asset_name(asset, source_label="runtime asset")
    if runtime_asset not in settings:
        available = ", ".join(sorted(settings))
        raise ValueError(
            f"Runtime asset {runtime_asset!r} not found in {runtime_manifest_path}. "
            f"Available: {available}"
        )
    runtime_settings = settings[runtime_asset]
    if not runtime_settings["enabled"]:
        raise ValueError(
            f"Runtime asset {runtime_asset!r} is disabled in {runtime_manifest_path}."
        )
    return runtime_settings


def load_runtime_artifact_paths(
        runtime_manifest_path=RUNTIME_ACTIVE_PATH,
        *,
        asset=None,
):
    return load_runtime_asset_settings(
        asset,
        runtime_manifest_path=runtime_manifest_path,
    )["artifacts"]
