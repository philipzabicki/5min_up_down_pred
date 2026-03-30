import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from data_quality_filters import drop_frozen_ohlc_blocks
from modeling_dataset_utils import (
    MODELING_DATASET_CONFIG_FILE,
    load_excluded_feature_names_from_settings,
    load_feature_subset_from_settings,
    load_modeling_dataset_settings,
    resolve_raw_dataset_input_path,
    resolve_modeling_float_dtype,
    resolve_modeling_float_dtype_name,
    resolve_modeling_dataset_output_paths,
    split_feature_subset,
)
from target_weights import (
    TARGET_WEIGHT_COL,
    TARGET_WEIGHT_DECISION_VALUE,
    TARGET_WEIGHT_OTHER_VALUE,
    add_target_weights,
    compute_binary_close_target_from_opened,
    summarize_target_weights,
)

from features.ADX import get_adx_values
from features.BollingerBands import get_bollinger_bands_values
from features.ChaikinOsc import get_chaikin_oscillator_values
from features.candle_features import (
    add_candle_derived_features,
    add_candle_streak_features,
    resolve_streak_interval_to_rule,
)
from features.KeltnerChannel import get_keltner_channel_values
from features.MACD import get_macd_values
from features.realized_volatility import (
    REALIZED_VOLATILITY_FEATURE_COLUMNS,
    add_realized_volatility_features,
)
from features.session_open_features import add_session_counter_features
from features.StochOsc import get_stochastic_oscillator_values
from features.volume_profile_fixed_range import (
    FEATURE_VERSION as VP_FEATURE_VERSION,
    MODELING_STATE_DIR as VP_MODELING_STATE_DIR,
    build_volume_profile_features,
    get_feature_columns as get_volume_profile_feature_columns,
    normalize_config as normalize_volume_profile_config,
    save_state as save_volume_profile_state,
)

TARGET_TIME_COL = "Opened"
TARGET_PRICE_COL = "Close"
TARGET_COL = "target_5m_candle_up"
TARGET_HORIZON_MINUTES = 5
PSEUDO_TARGET_RE = re.compile(r"^target_\d+m_ahead_ret$")
PARAM_NAME_PART_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")

FIT_RESULT_BASE_RE = re.compile(
    r"^(?P<indicator>ADX|BollingerBands|ChaikinOsc|KeltnerChannel|MACD|StochOsc)"
    r"_target_(?P<horizon>\d+m)_ahead_ret_pop(?P<pop>\d+)(?:_.*)?$"
)
BASE_DATA_FILE_SYMBOL_INTERVAL_RE = re.compile(
    r"^(?P<symbol>[A-Za-z0-9_]+?)(?P<interval>\d+[mhdwM])$"
)

VALUE_BUILDERS = {
    "ADX": get_adx_values,
    "BollingerBands": get_bollinger_bands_values,
    "ChaikinOsc": get_chaikin_oscillator_values,
    "KeltnerChannel": get_keltner_channel_values,
    "MACD": get_macd_values,
    "StochOsc": get_stochastic_oscillator_values,
}


def infer_ohlcv_columns(df):
    cols = list(df.columns)
    lower = [c.lower() for c in cols]
    required = ["open", "high", "low", "close", "volume"]
    if all(req in lower for req in required):
        return [cols[lower.index(req)] for req in required]
    if len(cols) >= 6:
        return cols[1:6]
    raise ValueError("Cannot infer OHLCV columns automatically.")


def require_columns(df, required):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def concat_feature_frame(df, feature_frame, context):
    if feature_frame is None or feature_frame.empty:
        return df

    duplicate_cols = [col for col in feature_frame.columns if col in df.columns]
    if duplicate_cols:
        preview = ", ".join(duplicate_cols[:10])
        raise ValueError(
            f"{context} would create duplicate columns. "
            f"Duplicate_count={len(duplicate_cols)} preview=[{preview}]"
        )

    if len(feature_frame) != len(df):
        raise ValueError(
            f"{context} length mismatch: {len(feature_frame)} != {len(df)}"
        )

    feature_frame = feature_frame.copy()
    feature_frame.index = df.index
    return pd.concat([df, feature_frame], axis=1, copy=False)


def build_target(df):
    out = df.copy()
    out[TARGET_TIME_COL] = pd.to_datetime(out[TARGET_TIME_COL], errors="raise")
    out = out.sort_values(TARGET_TIME_COL).reset_index(drop=True)

    out[TARGET_COL] = compute_binary_close_target_from_opened(
        opened_values=out[TARGET_TIME_COL],
        close_values=out[TARGET_PRICE_COL],
        horizon_minutes=TARGET_HORIZON_MINUTES,
    )
    out = add_target_weights(
        out, opened_col=TARGET_TIME_COL, weight_col=TARGET_WEIGHT_COL
    )
    return out


def drop_pseudo_targets(df, target_col):
    pseudo_targets = [
        col for col in df.columns if PSEUDO_TARGET_RE.match(col) and col != target_col
    ]
    if pseudo_targets:
        df = df.drop(columns=pseudo_targets)
    return df, pseudo_targets


def _normalize_float_token(text_or_number):
    return f"{float(text_or_number):.6f}".rstrip("0").rstrip(".")


def _normalize_param_value_token(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _normalize_float_token(value)
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _param_name_code(param_name):
    parts = PARAM_NAME_PART_RE.findall(str(param_name))
    if not parts:
        return re.sub(r"[^a-z0-9]+", "", str(param_name).lower())
    return "".join(part.lower()[:3] for part in parts)


def _param_suffix_for_feature(params):
    return "_".join(
        f"{_param_name_code(name)}{_normalize_param_value_token(value)}"
        for name, value in sorted(params.items())
    )


def _params_signature(params):
    return tuple(
        (str(name), _normalize_param_value_token(value))
        for name, value in sorted(params.items())
    )


def _stat_code(metric_stat):
    stat_txt = str(metric_stat).strip().lower()
    if stat_txt in {"mean_clip", "mc"}:
        return "mc"
    if stat_txt in {"median", "md"}:
        return "md"
    raise ValueError(
        f"Unsupported metric.stat='{metric_stat}'. Expected mean_clip/median/mc/md."
    )


def _extract_fit_params(payload, json_path):
    if isinstance(payload.get("params"), dict):
        return payload["params"]

    best = payload.get("best")
    if isinstance(best, dict) and isinstance(best.get("params"), dict):
        return best["params"]

    raise ValueError(
        f"Malformed fit config (missing params or best.params dict): {json_path}"
    )


def _metric_suffix_for_feature(payload, json_path):
    current_metric_name = "extremes_vs_mid_ir_oof"
    metric = payload.get("metric")
    if not isinstance(metric, dict):
        return None

    metric_name = str(metric.get("name", "")).strip()
    if metric_name != current_metric_name:
        return None

    required = ("q_ext", "q_mid", "train_frac", "stat", "segments_count")
    missing = [k for k in required if k not in metric]
    if missing:
        return None

    return (
        f"qe{_normalize_float_token(metric['q_ext'])}"
        f"_qm{_normalize_float_token(metric['q_mid'])}"
        f"_tf{_normalize_float_token(metric['train_frac'])}"
        f"_st{_stat_code(metric['stat'])}"
        f"_sg{int(metric['segments_count'])}"
    )


def _build_feature_col(indicator, horizon, pop, params, metric_suffix):
    base = (
        f"{indicator}_fit_{horizon}_pop{int(pop)}_{_param_suffix_for_feature(params)}"
    )
    if not metric_suffix:
        return base
    return f"{base}_{metric_suffix}"


def parse_fit_results(fit_dir):
    fit_dir = Path(fit_dir)
    if not fit_dir.exists():
        raise FileNotFoundError(f"Missing fit configs dir: {fit_dir}")

    configs = []
    unmatched = []
    seen_feature_cols = {}

    for json_path in sorted(fit_dir.rglob("*.json")):
        if json_path.name in {
            "fit_indicators_config.json",
            "fit_indicators_applied_config.json",
        }:
            continue

        match = FIT_RESULT_BASE_RE.match(json_path.stem)
        if not match:
            unmatched.append(str(json_path.relative_to(fit_dir)))
            continue

        indicator = match.group("indicator")
        if indicator not in VALUE_BUILDERS:
            raise ValueError(f"Unsupported indicator '{indicator}' in {json_path.name}")

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        params = _extract_fit_params(payload, json_path=json_path)
        metric_suffix = _metric_suffix_for_feature(payload, json_path=json_path)
        params_signature = _params_signature(params)

        horizon = match.group("horizon")
        pop_size = int(match.group("pop"))
        feature_col = _build_feature_col(
            indicator=indicator,
            horizon=horizon,
            pop=pop_size,
            params=params,
            metric_suffix=metric_suffix,
        )
        if feature_col in seen_feature_cols:
            if seen_feature_cols[feature_col] == params_signature:
                continue
            raise ValueError(
                f"Duplicate output feature column '{feature_col}' from fit dir {fit_dir}"
            )
        seen_feature_cols[feature_col] = params_signature

        configs.append(
            {
                "horizon": horizon,
                "horizon_minutes": int(horizon[:-1]),
                "indicator": indicator,
                "pop_size": pop_size,
                "feature_col": feature_col,
                "params": params,
                "json_path": json_path,
            }
        )

    if not configs:
        sample = ", ".join(unmatched[:5]) if unmatched else "none"
        raise FileNotFoundError(
            "No fit config files matched required naming scheme in "
            f"{fit_dir}. First unmatched: {sample}"
        )

    if unmatched:
        print(
            f"warning: ignored {len(unmatched)} non-matching files in {fit_dir}; "
            f"first={unmatched[0]}"
        )

    return sorted(
        configs,
        key=lambda item: (
            int(item["horizon_minutes"]),
            str(item["indicator"]),
            str(item["feature_col"]),
            str(item["json_path"]),
        ),
    )


def resolve_volume_profile_modeling_state_path(base_data_file):
    stem = Path(base_data_file).stem
    match = BASE_DATA_FILE_SYMBOL_INTERVAL_RE.match(stem)
    if not match:
        raise ValueError(
            "Could not derive symbol/interval for volume profile modeling state from "
            f"base_data_file={base_data_file!r}. Expected e.g. BTCUSDT1m.csv"
        )
    symbol = match.group("symbol")
    interval = match.group("interval")
    return (
        VP_MODELING_STATE_DIR / f"{symbol}_{interval}_{VP_FEATURE_VERSION}_modeling_end"
    )


def add_indicator_values(df, ohlcv_np, configs, float_dtype=np.float64):
    feature_values = {}
    for cfg in configs:
        print(f"processing fit config: {cfg['json_path'].name} -> feature_col={cfg['feature_col']}")
        indicator = cfg["indicator"]
        horizon = cfg["horizon"]
        feature_col = cfg["feature_col"]
        params = cfg["params"]

        builder = VALUE_BUILDERS.get(indicator)
        if builder is None:
            raise ValueError(f"Indicator {indicator} not supported by VALUE_BUILDERS")

        values = np.asarray(builder(params, ohlcv_np)).reshape(-1)
        if len(values) != len(df):
            raise ValueError(
                f"Length mismatch for {indicator} ({horizon}): {len(values)} != {len(df)}"
            )
        if feature_col in df.columns:
            raise ValueError(f"Column {feature_col} already exists in dataframe")
        if feature_col in feature_values:
            raise ValueError(
                f"Duplicate indicator feature column requested: {feature_col}"
            )
        feature_values[feature_col] = values.astype(float_dtype, copy=False)

    feature_frame = pd.DataFrame(feature_values, index=df.index)
    return concat_feature_frame(df, feature_frame, context="Indicator features")


def build_dataset_from_settings(settings):
    fit_results_dir = Path(settings["fit_results_dir"])
    base_data_file = str(settings["base_data_file"])
    streak_intervals = list(settings["candle_streak_intervals"])
    feature_subset = load_feature_subset_from_settings(settings)
    excluded_features = load_excluded_feature_names_from_settings(settings)
    excluded_feature_names = (
        tuple(excluded_features["features"]) if excluded_features else tuple()
    )
    excluded_feature_set = set(excluded_feature_names)
    feature_subset_parts = (
        split_feature_subset(feature_subset["features"]) if feature_subset else None
    )
    vp_cfg = settings.get("volume_profile_fixed_range")
    vp_normalized_cfg = normalize_volume_profile_config(vp_cfg)
    vp_enabled = bool(vp_normalized_cfg["enabled"])
    vp_feature_cols = tuple(vp_normalized_cfg["feature_columns"])
    float_dtype = resolve_modeling_float_dtype(settings)
    float_dtype_name = resolve_modeling_float_dtype_name(settings)
    drop_frozen_ohlc_blocks_cfg = settings.get("drop_frozen_ohlc_blocks")
    if feature_subset_parts and feature_subset_parts["unclassified_feature_cols"]:
        preview = ", ".join(feature_subset_parts["unclassified_feature_cols"][:10])
        raise ValueError(
            "Feature subset contains unsupported feature names. "
            f"Missing_count={len(feature_subset_parts['unclassified_feature_cols'])} "
            f"preview=[{preview}]"
        )
    if feature_subset_parts and feature_subset_parts["volume_profile_feature_cols"]:
        if not vp_enabled:
            raise ValueError(
                "Feature subset references volume profile features but "
                "volume_profile_fixed_range.enabled is false."
            )
        missing_vp_features = [
            col
            for col in feature_subset_parts["volume_profile_feature_cols"]
            if col not in set(vp_feature_cols)
        ]
        if missing_vp_features:
            preview = ", ".join(missing_vp_features[:10])
            raise ValueError(
                "Feature subset references unsupported volume profile features. "
                f"Missing_count={len(missing_vp_features)} preview=[{preview}]"
            )

    configured_streak_interval_to_rule = resolve_streak_interval_to_rule(
        streak_intervals
    )
    if feature_subset_parts:
        requested_streak_intervals = feature_subset_parts["streak_intervals"]
        unsupported_streak_intervals = [
            label
            for label in requested_streak_intervals
            if label not in configured_streak_interval_to_rule
        ]
        if unsupported_streak_intervals:
            raise ValueError(
                "Feature subset requires candle streak intervals that are missing in "
                f"{MODELING_DATASET_CONFIG_FILE}: {unsupported_streak_intervals}"
            )
        streak_interval_to_rule = {
            label: configured_streak_interval_to_rule[label]
            for label in requested_streak_intervals
        }
    else:
        streak_interval_to_rule = configured_streak_interval_to_rule

    input_file = resolve_raw_dataset_input_path(settings)
    if not input_file.exists():
        raise FileNotFoundError(f"Missing base dataset: {input_file}")

    configs = parse_fit_results(fit_results_dir)
    if feature_subset_parts:
        selected_indicator_feature_cols = set(
            feature_subset_parts["indicator_feature_cols"]
        )
        available_indicator_feature_cols = {cfg["feature_col"] for cfg in configs}
        missing_indicator_features = [
            col
            for col in feature_subset_parts["indicator_feature_cols"]
            if col not in available_indicator_feature_cols
        ]
        if missing_indicator_features:
            preview = ", ".join(missing_indicator_features[:10])
            raise FileNotFoundError(
                "Feature subset references indicator features that do not exist in "
                f"{fit_results_dir}. Missing_count={len(missing_indicator_features)} "
                f"preview=[{preview}]"
            )
        configs = [
            cfg
            for cfg in configs
            if cfg["feature_col"] in selected_indicator_feature_cols
        ]
    elif excluded_feature_set:
        configs = [
            cfg for cfg in configs if cfg["feature_col"] not in excluded_feature_set
        ]

    print(f"loading base dataset: {input_file}")
    if feature_subset:
        source_count = int(feature_subset.get("source_count", feature_subset["count"]))
        print(
            "feature subset active: "
            f"path={feature_subset['path']} count={feature_subset['count']} "
            f"source_count={source_count} format={feature_subset['format']} "
            f"excluded_from_subset={feature_subset.get('excluded_from_subset_count', 0)}"
        )
    if excluded_features:
        preview = ", ".join(excluded_feature_names[:5])
        print(
            "feature exclusions active: "
            f"count={excluded_features['count']} preview=[{preview}]"
        )
    print(f"float precision mode: {float_dtype_name}")
    df = pd.read_csv(input_file)
    df, drop_frozen_summary = drop_frozen_ohlc_blocks(
        df,
        raw_config=drop_frozen_ohlc_blocks_cfg,
    )
    if drop_frozen_summary["enabled"]:
        print(
            "drop frozen OHLC blocks: "
            f"min_block_len={drop_frozen_summary['min_block_len']} "
            f"removed_rows={drop_frozen_summary['rows_removed']} "
            f"removed_blocks={drop_frozen_summary['blocks_removed']} "
            f"largest_block_len={drop_frozen_summary['largest_block_len']} "
            f"rows_after={drop_frozen_summary['rows_after']}"
        )
    if streak_interval_to_rule:
        print(
            "adding candle streak features for intervals: "
            + ", ".join(streak_interval_to_rule.keys())
        )
        df = add_candle_streak_features(df, interval_to_rule=streak_interval_to_rule)
    else:
        print("skipping candle streak features (none requested)")
    selected_candle_feature_cols = (
        feature_subset_parts["candle_feature_cols"] if feature_subset_parts else None
    )
    if selected_candle_feature_cols is None or selected_candle_feature_cols:
        print("adding candle derived/pattern features")
        df = add_candle_derived_features(df, feature_cols=selected_candle_feature_cols)
    else:
        print("skipping candle derived/pattern features (none requested)")
    selected_session_feature_cols = (
        feature_subset_parts["session_feature_cols"] if feature_subset_parts else None
    )
    selected_realized_volatility_feature_cols = (
        feature_subset_parts["realized_volatility_feature_cols"]
        if feature_subset_parts
        else None
    )
    if selected_session_feature_cols is None or selected_session_feature_cols:
        print("adding global session counter features")
        df = add_session_counter_features(
            df,
            feature_cols=selected_session_feature_cols,
        )
    else:
        print("skipping global session counter features (none requested)")
    should_add_realized_volatility = (
        selected_realized_volatility_feature_cols is None
        or bool(selected_realized_volatility_feature_cols)
    )
    if should_add_realized_volatility:
        kept_realized_volatility_cols = [
            col
            for col in REALIZED_VOLATILITY_FEATURE_COLUMNS
            if col not in excluded_feature_set
        ]
        if kept_realized_volatility_cols or feature_subset_parts is not None:
            print("adding realized volatility features")
            df = add_realized_volatility_features(df)
        else:
            print("skipping realized volatility features (all excluded)")
    else:
        print("skipping realized volatility features (none requested)")
    if vp_enabled:
        expected_vp_cols = get_volume_profile_feature_columns(vp_normalized_cfg)
        print(
            "adding fixed-range volume profile features "
            f"({len(expected_vp_cols)} cols)"
        )
        vp_features_df, vp_state = build_volume_profile_features(df, vp_normalized_cfg)
        vp_feature_frame = pd.DataFrame(
            {
                feature_col: vp_features_df[feature_col].to_numpy(
                    dtype=float_dtype, copy=False
                )
                for feature_col in expected_vp_cols
            },
            index=df.index,
        )
        df = concat_feature_frame(
            df,
            vp_feature_frame,
            context="Volume profile features",
        )
        vp_state_path = resolve_volume_profile_modeling_state_path(base_data_file)
        saved_paths = save_volume_profile_state(vp_state, vp_state_path)
        print(f"[vp] saved modeling-end state -> {saved_paths['npz']}")
    else:
        print("skipping fixed-range volume profile features (disabled)")
    ohlcv_cols = infer_ohlcv_columns(df)
    ohlcv_np = df[ohlcv_cols].to_numpy(dtype=np.float64, copy=True)
    print(f"adding {len(configs)} indicator configs from {fit_results_dir}")
    df = add_indicator_values(df, ohlcv_np, configs, float_dtype=float_dtype)

    require_columns(df, [TARGET_TIME_COL, TARGET_PRICE_COL])
    df = build_target(df)
    df, dropped_pseudo_targets = drop_pseudo_targets(df, TARGET_COL)
    protected_cols = {
        TARGET_TIME_COL,
        TARGET_COL,
        TARGET_WEIGHT_COL,
        *ohlcv_cols,
    }
    excluded_present_cols = [col for col in excluded_feature_names if col in df.columns]
    excluded_droppable_cols = [
        col for col in excluded_present_cols if col not in protected_cols
    ]
    excluded_protected_cols = [
        col for col in excluded_present_cols if col in protected_cols
    ]
    excluded_missing_cols = [
        col for col in excluded_feature_names if col not in df.columns
    ]
    if feature_subset:
        missing_selected_features = [
            col for col in feature_subset["features"] if col not in df.columns
        ]
        if missing_selected_features:
            preview = ", ".join(missing_selected_features[:10])
            raise ValueError(
                "Generated dataset is missing requested subset features. "
                f"Missing_count={len(missing_selected_features)} preview=[{preview}]"
            )
        keep_cols = [TARGET_TIME_COL]
        keep_cols.extend(col for col in ohlcv_cols if col not in keep_cols)
        keep_cols.extend(
            col for col in feature_subset["features"] if col not in keep_cols
        )
        if vp_enabled:
            keep_cols.extend(col for col in vp_feature_cols if col not in keep_cols)
        keep_cols.extend(
            col
            for col in (TARGET_COL, TARGET_WEIGHT_COL)
            if col in df.columns and col not in keep_cols
        )
        dropped_unselected_cols = [
            col for col in df.columns if col not in set(keep_cols)
        ]
        df = df.loc[:, keep_cols].copy()
        print(
            "subset pruning: "
            f"kept_feature_cols={feature_subset['count']} "
            f"dropped_other_cols={len(dropped_unselected_cols)}"
        )
    elif excluded_droppable_cols:
        df = df.drop(columns=excluded_droppable_cols)
        print(
            "feature exclusions applied: "
            f"dropped_feature_cols={len(excluded_droppable_cols)} "
            f"missing_requested={len(excluded_missing_cols)} "
            f"protected_kept={len(excluded_protected_cols)}"
        )
    elif excluded_features:
        print(
            "feature exclusions applied: "
            f"dropped_feature_cols=0 missing_requested={len(excluded_missing_cols)} "
            f"protected_kept={len(excluded_protected_cols)}"
        )
    float_cols = [col for col in df.columns if pd.api.types.is_float_dtype(df[col])]
    if float_cols:
        df = df.astype({col: float_dtype for col in float_cols}, copy=False)

    output_paths = resolve_modeling_dataset_output_paths(settings)
    output_parquet = output_paths["parquet"]
    output_head_csv = output_paths["head_csv"]
    output_tail_csv = output_paths["tail_csv"]
    preview_rows = int(settings["preview_rows"])

    df.to_parquet(output_parquet, index=False)
    df.head(preview_rows).to_csv(output_head_csv, index=False)
    df.tail(preview_rows).to_csv(output_tail_csv, index=False)

    class_counts = df[TARGET_COL].value_counts(dropna=False).sort_index().to_dict()
    weight_summary = summarize_target_weights(df[TARGET_WEIGHT_COL].to_numpy())
    print(f"target column added: {TARGET_COL} (h={TARGET_HORIZON_MINUTES}m)")
    print(
        f"target weight column added: {TARGET_WEIGHT_COL} "
        f"(minute%5==4 -> {TARGET_WEIGHT_DECISION_VALUE}, else {TARGET_WEIGHT_OTHER_VALUE})"
    )
    print(f"dropped pseudo-target columns: {dropped_pseudo_targets}")
    print(f"target class counts: {class_counts}")
    print(f"target weight summary: {weight_summary}")
    print(f"saved modeling dataset (parquet) -> {output_parquet}")
    print(f"saved preview head({preview_rows}) -> {output_head_csv}")
    print(f"saved preview tail({preview_rows}) -> {output_tail_csv}")
    return output_parquet


def main():
    settings = load_modeling_dataset_settings()
    print(
        "modeling dataset settings: "
        f"config={MODELING_DATASET_CONFIG_FILE} | "
        f"raw_data_dir={settings['raw_data_dir']} | "
        f"modeling_output_dir={settings['modeling_output_dir']} | "
        f"base_data_file={settings['base_data_file']} | "
        f"fit_results_dir={settings['fit_results_dir']} | "
        f"candle_streak_intervals={settings['candle_streak_intervals']} | "
        f"drop_frozen_ohlc_blocks={settings['drop_frozen_ohlc_blocks']} | "
        f"output_suffix={settings['output_suffix']} | "
        f"preview_rows={settings['preview_rows']} | "
        f"feature_subset_path={settings['feature_subset_path']} | "
        f"excluded_feature_names_count={len(settings['excluded_feature_names'])} | "
        f"float_precision={resolve_modeling_float_dtype_name(settings)}"
    )
    output_path = build_dataset_from_settings(settings)
    print(f"Generated dataset: {output_path}")


if __name__ == "__main__":
    main()
