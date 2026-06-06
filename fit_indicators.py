import hashlib
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableMating,
    MixedVariableSampling,
)
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.variable import Binary, Choice, Integer, Real
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.bitflip import BFM
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

from utils.config import coerce_path
from utils.data import compute_binary_close_target_from_opened
from utils.data import drop_frozen_ohlc_blocks

try:
    from pymoo.core.problem import StarmapParallelization
except ImportError:
    from pymoo.parallelization.starmap import StarmapParallelization

from features.ADX import ADXFitting, adx_initializer
from features.BollingerBands import BollingerBandsFitting, bollinger_bands_initializer
from features.ChaikinOsc import ChaikinOscillatorFitting, adl_initializer
from features.common_utils import (
    compute_indicator_pop_sizes,
    normalize_indicators,
    resolve_base_pop_size,
)
from utils.project_config import (
    ACTIVE_CONFIG_PATH,
    INDICATOR_FIT_CONFIG_PATH,
    active_asset_path,
    build_indicator_fit_config,
    load_active_asset,
)
from features.KeltnerChannel import KeltnerChannelFitting, keltner_channel_initializer
from features.MACD import MACDFitting, macd_initializer
from features.StochOsc import (
    StochasticOscillatorFitting,
    stochastic_oscillator_initializer,
)

CONFIG_FILE = str(INDICATOR_FIT_CONFIG_PATH)
CPU_CORES_COUNT = 17
FIT_RESULTS_ROOT = active_asset_path("data/features/indicators_fit/{asset}/tuning")

DEFAULT_METRIC_NAME = "extremes_vs_mid_ir_oof"
DEFAULT_METRIC_SEGMENTS_COUNT = 12
DEFAULT_METRIC_TRAIN_FRAC = 0.80
DEFAULT_METRIC_GAP = 1500
DEFAULT_METRIC_Q_EXT = 0.10
DEFAULT_METRIC_Q_MID = 0.10
DEFAULT_METRIC_STAT = "mean_clip"
DEFAULT_METRIC_CLIP_Q = 0.01
DEFAULT_METRIC_MIN_BUCKET_SIZE = 50
DEFAULT_METRIC_MIN_VALID_SEGMENTS = 2
DEFAULT_METRIC_RECENCY_WEIGHTING_ENABLED = False
DEFAULT_METRIC_RECENCY_WEIGHTING_MODE = "linear"
DEFAULT_METRIC_RECENCY_WEIGHT_MIN = 1.0
DEFAULT_METRIC_RECENCY_WEIGHT_MAX = 1.5
DEFAULT_PROXY_TARGET_MODE = "ahead_ret"
DEFAULT_PROXY_TARGET_TIME_COL = "Opened"
SUPPORTED_PROXY_TARGET_MODES = frozenset({"ahead_ret", "candle_up"})

SELECTION_METRIC_NAME = DEFAULT_METRIC_NAME

PROBLEM_MAP = {
    "ADX": ADXFitting,
    "ChaikinOsc": ChaikinOscillatorFitting,
    "KeltnerChannel": KeltnerChannelFitting,
    "MACD": MACDFitting,
    "StochOsc": StochasticOscillatorFitting,
    "BollingerBands": BollingerBandsFitting,
}
INITIALIZER_MAP = {
    "ADX": adx_initializer,
    "ChaikinOsc": adl_initializer,
    "KeltnerChannel": keltner_channel_initializer,
    "MACD": macd_initializer,
    "StochOsc": stochastic_oscillator_initializer,
    "BollingerBands": bollinger_bands_initializer,
}


def df_size_mib(df):
    return df.memory_usage(index=True, deep=True).sum() / 1024 ** 2


def to_serializable_params(params):
    clean = {}
    for k, v in params.items():
        if isinstance(v, np.integer):
            clean[k] = int(v)
        elif isinstance(v, np.floating):
            clean[k] = float(v)
        else:
            clean[k] = v
    return clean


def _infer_ohlcv_cols(df):
    cols = list(df.columns)
    low = [c.lower() for c in cols]
    need = ["open", "high", "low", "close", "volume"]
    if all(n in low for n in need):
        return [cols[low.index(n)] for n in need]

    if len(cols) >= 6:
        return cols[1:6]

    raise ValueError(
        "Cannot infer OHLCV columns. Provide them explicitly in code/config."
    )


def _normalize_horizons(raw_horizons):
    horizons = sorted({int(h) for h in raw_horizons})
    if not horizons:
        raise ValueError("proxy_target_horizonts cannot be empty.")
    if any(h <= 0 for h in horizons):
        raise ValueError(f"All horizons must be positive. Got: {horizons}")
    return horizons


def _resolve_proxy_target_horizonts(interval_cfg, pair_cfg):
    raw_horizons = interval_cfg.get("proxy_target_horizonts")
    if raw_horizons is None:
        raw_horizons = pair_cfg.get("proxy_target_horizonts")
    if raw_horizons is None:
        raise ValueError(
            "Missing proxy_target_horizonts in fit_indicators_config "
            "(pair/interval level)."
        )
    return _normalize_horizons(raw_horizons)


def _resolve_proxy_target_price_col(interval_cfg, pair_cfg):
    return str(
        interval_cfg.get(
            "proxy_target_price_col",
            pair_cfg.get("proxy_target_price_col", "Close"),
        )
    )


def _resolve_proxy_target_mode(interval_cfg, pair_cfg):
    raw_mode = interval_cfg.get(
        "proxy_target_mode",
        pair_cfg.get("proxy_target_mode", DEFAULT_PROXY_TARGET_MODE),
    )
    mode = str(raw_mode).strip().lower()
    if mode not in SUPPORTED_PROXY_TARGET_MODES:
        raise ValueError(
            "proxy_target_mode must be one of "
            f"{sorted(SUPPORTED_PROXY_TARGET_MODES)}, got: {raw_mode!r}"
        )
    return mode


def _resolve_proxy_target_time_col(interval_cfg, pair_cfg):
    return str(
        interval_cfg.get(
            "proxy_target_time_col",
            pair_cfg.get("proxy_target_time_col", DEFAULT_PROXY_TARGET_TIME_COL),
        )
    )


def _resolve_existing_column_name(df, requested_col, *, field_name="column"):
    requested = str(requested_col)
    if requested in df.columns:
        return requested

    requested_l = requested.lower()
    for col in df.columns:
        if str(col).lower() == requested_l:
            return str(col)
    raise ValueError(
        f"{field_name}='{requested_col}' not found in dataframe."
    )


def _build_proxy_target_col_name(horizon_minutes, target_mode):
    horizon = int(horizon_minutes)
    if target_mode == "ahead_ret":
        return f"target_{horizon}m_ahead_ret"
    if target_mode == "candle_up":
        return f"target_{horizon}m_candle_up"
    raise ValueError(f"Unsupported proxy_target_mode: {target_mode}")


def _build_proxy_target_np_ahead_ret(price_np, horizon_minutes):
    horizon = int(horizon_minutes)
    if horizon <= 0:
        raise ValueError(f"horizon_minutes must be > 0, got: {horizon_minutes}")

    prices = np.asarray(price_np, dtype=np.float64).reshape(-1)
    safe_base = np.where(
        np.isfinite(prices) & (np.abs(prices) > 1e-12),
        prices,
        np.nan,
    )
    future = np.roll(safe_base, -horizon)
    future[-horizon:] = np.nan
    return (future / safe_base) - 1.0


def _build_proxy_target_np_candle_up(opened_values, close_values, horizon_minutes):
    return compute_binary_close_target_from_opened(
        opened_values=opened_values,
        close_values=close_values,
        horizon_minutes=int(horizon_minutes),
    )


def _format_float_token(value):
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _normalize_prob_param_values(raw_value, field_name):
    if isinstance(raw_value, (list, tuple, set)):
        values = list(raw_value)
    else:
        values = [raw_value]
    if not values:
        raise ValueError(f"{field_name} list cannot be empty.")

    out = []
    for value in values:
        val = float(value)
        if not (0.0 < val < 0.5):
            raise ValueError(f"{field_name} must satisfy 0 < value < 0.5, got: {val}")
        if not any(abs(val - prev) <= 1e-12 for prev in out):
            out.append(float(val))
    return out


def _coerce_bool_param(raw_value, field_name):
    if isinstance(raw_value, bool):
        return bool(raw_value)
    raise ValueError(f"{field_name} must be a boolean, got: {raw_value!r}")


def _resolve_metric_configs(interval_cfg, pair_cfg):
    metric_name = str(
        interval_cfg.get(
            "metric_name", pair_cfg.get("metric_name", DEFAULT_METRIC_NAME)
        )
    )
    if metric_name != SELECTION_METRIC_NAME:
        raise ValueError(
            f"metric_name must be '{SELECTION_METRIC_NAME}', got: '{metric_name}'"
        )

    segments_count = int(
        interval_cfg.get(
            "metric_segments_count",
            pair_cfg.get("metric_segments_count", DEFAULT_METRIC_SEGMENTS_COUNT),
        )
    )
    if segments_count < 1:
        raise ValueError(f"metric_segments_count must be >= 1, got: {segments_count}")

    train_frac = float(
        interval_cfg.get(
            "metric_train_frac",
            pair_cfg.get("metric_train_frac", DEFAULT_METRIC_TRAIN_FRAC),
        )
    )
    if not (0.0 < train_frac < 1.0):
        raise ValueError(
            f"metric_train_frac must satisfy 0 < frac < 1, got: {train_frac}"
        )

    gap = int(
        interval_cfg.get("metric_gap", pair_cfg.get("metric_gap", DEFAULT_METRIC_GAP))
    )
    if gap < 0:
        raise ValueError(f"metric_gap must be >= 0, got: {gap}")
    q_ext_values = _normalize_prob_param_values(
        interval_cfg.get("q_ext", pair_cfg.get("q_ext", DEFAULT_METRIC_Q_EXT)),
        "q_ext",
    )
    q_mid_values = _normalize_prob_param_values(
        interval_cfg.get("q_mid", pair_cfg.get("q_mid", DEFAULT_METRIC_Q_MID)),
        "q_mid",
    )

    stat = (
        str(interval_cfg.get("stat", pair_cfg.get("stat", DEFAULT_METRIC_STAT)))
        .strip()
        .lower()
    )
    if stat not in {"mean_clip", "median"}:
        raise ValueError(f"stat must be 'mean_clip' or 'median', got: {stat}")

    clip_q = float(
        interval_cfg.get("clip_q", pair_cfg.get("clip_q", DEFAULT_METRIC_CLIP_Q))
    )
    if not (0.0 <= clip_q < 0.5):
        raise ValueError(f"clip_q must satisfy 0 <= clip_q < 0.5, got: {clip_q}")

    min_bucket_size = int(
        interval_cfg.get(
            "min_bucket_size",
            pair_cfg.get("min_bucket_size", DEFAULT_METRIC_MIN_BUCKET_SIZE),
        )
    )
    if min_bucket_size < 1:
        raise ValueError(f"min_bucket_size must be >= 1, got: {min_bucket_size}")

    min_valid_segments = int(
        interval_cfg.get(
            "min_valid_segments",
            pair_cfg.get("min_valid_segments", DEFAULT_METRIC_MIN_VALID_SEGMENTS),
        )
    )
    if min_valid_segments < 1:
        raise ValueError(f"min_valid_segments must be >= 1, got: {min_valid_segments}")

    recency_weighting_enabled = _coerce_bool_param(
        interval_cfg.get(
            "metric_recency_weighting_enabled",
            pair_cfg.get(
                "metric_recency_weighting_enabled",
                DEFAULT_METRIC_RECENCY_WEIGHTING_ENABLED,
            ),
        ),
        "metric_recency_weighting_enabled",
    )
    recency_weighting_mode = str(
        interval_cfg.get(
            "metric_recency_weighting_mode",
            pair_cfg.get(
                "metric_recency_weighting_mode",
                DEFAULT_METRIC_RECENCY_WEIGHTING_MODE,
            ),
        )
    ).strip().lower()
    if recency_weighting_mode != "linear":
        raise ValueError(
            "metric_recency_weighting_mode must be 'linear', "
            f"got: {recency_weighting_mode!r}"
        )
    recency_weight_min = float(
        interval_cfg.get(
            "metric_recency_weight_min",
            pair_cfg.get(
                "metric_recency_weight_min",
                DEFAULT_METRIC_RECENCY_WEIGHT_MIN,
            ),
        )
    )
    recency_weight_max = float(
        interval_cfg.get(
            "metric_recency_weight_max",
            pair_cfg.get(
                "metric_recency_weight_max",
                DEFAULT_METRIC_RECENCY_WEIGHT_MAX,
            ),
        )
    )
    if recency_weight_min <= 0.0 or recency_weight_max <= 0.0:
        raise ValueError(
            "metric recency weights must be strictly positive: "
            f"min={recency_weight_min}, max={recency_weight_max}"
        )
    if recency_weight_max < recency_weight_min:
        raise ValueError(
            "metric_recency_weight_max must be >= metric_recency_weight_min: "
            f"min={recency_weight_min}, max={recency_weight_max}"
        )

    common = {
        "name": metric_name,
        "segments_count": int(segments_count),
        "train_frac": float(train_frac),
        "gap": int(gap),
        "stat": stat,
        "clip_q": float(clip_q),
        "min_bucket_size": int(min_bucket_size),
        "min_valid_segments": int(min_valid_segments),
        "recency_weighting_enabled": bool(recency_weighting_enabled),
        "recency_weighting_mode": recency_weighting_mode,
        "recency_weight_min": float(recency_weight_min),
        "recency_weight_max": float(recency_weight_max),
    }

    out = []
    for q_ext in q_ext_values:
        for q_mid in q_mid_values:
            if not (0.5 - q_mid > q_ext):
                raise ValueError(
                    "Invalid q_ext/q_mid: require 0.5 - q_mid > q_ext so mid band does not "
                    "overlap extremes. "
                    f"Got q_ext={q_ext}, q_mid={q_mid}"
                )
            metric = dict(common)
            metric["q_ext"] = float(q_ext)
            metric["q_mid"] = float(q_mid)
            out.append(metric)
    return out


def _stat_code_for_filename(stat):
    stat_norm = str(stat).strip().lower()
    if stat_norm == "mean_clip":
        return "mc"
    if stat_norm == "median":
        return "md"
    raise ValueError(f"Unsupported stat for filename: {stat}")


def _metric_filename_suffix(metric_config):
    suffix = (
        f"qe{_format_float_token(metric_config['q_ext'])}"
        f"_qm{_format_float_token(metric_config['q_mid'])}"
        f"_tf{_format_float_token(metric_config['train_frac'])}"
        f"_st{_stat_code_for_filename(metric_config['stat'])}"
        f"_sg{int(metric_config['segments_count'])}"
    )
    if bool(metric_config.get("recency_weighting_enabled")) and not np.isclose(
            float(metric_config["recency_weight_min"]),
            float(metric_config["recency_weight_max"]),
    ):
        suffix += (
            f"_rwlin{_format_float_token(metric_config['recency_weight_min'])}"
            f"-{_format_float_token(metric_config['recency_weight_max'])}"
        )
    return suffix


def _fit_config_hash(config_payload):
    normalized = json.dumps(
        config_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:16]


def _fit_result_filename(ind_name, target_col, filename_pop_size, metric_config):
    metric_suffix = _metric_filename_suffix(metric_config)
    return f"{ind_name}_{target_col}_pop{filename_pop_size}_{metric_suffix}.json"


ALGORITHM = MixedVariableGA
TERMINATION = DefaultSingleObjectiveTermination(
    xtol=1e-6, ftol=1e-6, period=10, n_max_gen=50
)

MATING = MixedVariableMating(
    crossover={
        Binary: UX(),
        Real: SBX(eta=3),
        Integer: SBX(eta=3, vtype=float, repair=RoundingRepair()),
        Choice: UX(),
    },
    mutation={
        Binary: BFM(),
        Real: PM(eta=7),
        Integer: PM(eta=7, vtype=float, repair=RoundingRepair()),
        Choice: ChoiceRandomMutation(),
    },
    eliminate_duplicates=MixedVariableDuplicateElimination(),
)


def run_indicator_ga(
        ind_name,
        ohlcv_np,
        target_np,
        pop_size,
        interval,
        metric_config,
):
    with Pool(
            CPU_CORES_COUNT,
            initializer=INITIALIZER_MAP[ind_name],
            initargs=(
                    ohlcv_np,
                    target_np,
                    metric_config["segments_count"],
                    metric_config["train_frac"],
                    metric_config["gap"],
                    metric_config["q_ext"],
                    metric_config["q_mid"],
                    metric_config["stat"],
                    metric_config["clip_q"],
                    metric_config["min_bucket_size"],
                    metric_config["min_valid_segments"],
                    metric_config["recency_weighting_enabled"],
                    metric_config["recency_weighting_mode"],
                    metric_config["recency_weight_min"],
                    metric_config["recency_weight_max"],
            ),
    ) as pool:
        runner = StarmapParallelization(pool.starmap)

        print(f"[{interval} {ind_name}] GA (pop={pop_size})")

        problem = PROBLEM_MAP[ind_name](elementwise_runner=runner)
        algorithm = MixedVariableGA(
            pop_size=pop_size,
            sampling=MixedVariableSampling(),
            mating=MATING,
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        )

        res = minimize(
            problem,
            algorithm,
            save_history=False,
            termination=TERMINATION,
            verbose=True,
        )

        score = float(-res.F[0])
        best = {
            "params": to_serializable_params(res.X),
            "score": score,
            "abs_corr": score,
        }
        return best


def main():
    cfg = build_indicator_fit_config()
    active_asset = load_active_asset()
    config_hash = _fit_config_hash(cfg)
    results_dir = FIT_RESULTS_ROOT / config_hash
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "fit_indicators_applied_config.json").write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8",
    )
    print(
        "fit results dir: "
        f"{results_dir} (config_hash={config_hash}, "
        f"active_asset={active_asset}, "
        f"indicator_fit_config={INDICATOR_FIT_CONFIG_PATH}, "
        f"active_profiles={ACTIVE_CONFIG_PATH})"
    )

    for pair, pair_cfg in cfg["pairs"].items():
        for interval, interval_cfg in pair_cfg["intervals"].items():
            data_path = coerce_path(interval_cfg["data_path"])
            data_file = interval_cfg.get("data_file", "dataset.csv")
            proxy_target_horizonts = _resolve_proxy_target_horizonts(
                interval_cfg, pair_cfg
            )
            proxy_target_price_col_raw = _resolve_proxy_target_price_col(
                interval_cfg, pair_cfg
            )
            proxy_target_mode = _resolve_proxy_target_mode(interval_cfg, pair_cfg)
            proxy_target_time_col_raw = _resolve_proxy_target_time_col(
                interval_cfg, pair_cfg
            )
            metric_configs = _resolve_metric_configs(interval_cfg, pair_cfg)
            indicators_cfg = interval_cfg["indicators"]
            indicator_names = normalize_indicators(indicators_cfg)
            base_pop_size = resolve_base_pop_size(
                interval_cfg, pair_cfg, indicators_cfg
            )
            indicator_pop_sizes = compute_indicator_pop_sizes(
                indicator_names=indicator_names,
                problem_map=PROBLEM_MAP,
                base_pop_size=base_pop_size,
            )

            df = pd.read_csv(data_path / data_file)
            df, drop_frozen_summary = drop_frozen_ohlc_blocks(
                df,
                raw_config=interval_cfg.get(
                    "drop_frozen_ohlc_blocks",
                    pair_cfg.get("drop_frozen_ohlc_blocks"),
                ),
            )
            ohlcv_cols = _infer_ohlcv_cols(df)
            ohlcv_np = df[ohlcv_cols].to_numpy(dtype=np.float64, copy=False)
            proxy_target_price_col = _resolve_existing_column_name(
                df,
                proxy_target_price_col_raw,
                field_name="proxy_target_price_col",
            )
            proxy_price_np = pd.to_numeric(
                df[proxy_target_price_col],
                errors="coerce",
            ).to_numpy(dtype=np.float64, copy=False)
            proxy_target_time_col = None
            if proxy_target_mode == "candle_up":
                proxy_target_time_col = _resolve_existing_column_name(
                    df,
                    proxy_target_time_col_raw,
                    field_name="proxy_target_time_col",
                )

            print(
                f"{pair} | {interval} df ~{df_size_mib(df):.2f} MiB | "
                f"ohlcv cols: {ohlcv_cols} | "
                f"proxy_target_horizonts: {proxy_target_horizonts} | "
                f"proxy_target_mode: {proxy_target_mode} | "
                f"proxy_target_price_col: {proxy_target_price_col} | "
                f"proxy_target_time_col: {proxy_target_time_col or '-'} | "
                f"metric_variants={len(metric_configs)} | "
                f"base_pop_size: {base_pop_size}"
            )
            if drop_frozen_summary["enabled"]:
                print(
                    f"{pair} | {interval} drop_frozen_ohlc_blocks "
                    f"min_block_len={drop_frozen_summary['min_block_len']} "
                    f"removed_rows={drop_frozen_summary['rows_removed']} "
                    f"removed_blocks={drop_frozen_summary['blocks_removed']} "
                    f"largest_block_len={drop_frozen_summary['largest_block_len']} "
                    f"rows_after={drop_frozen_summary['rows_after']}"
                )
            for metric_idx, metric_cfg in enumerate(metric_configs, start=1):
                print(f"{pair} | {interval} metric[{metric_idx}] -> {metric_cfg}")
            pop_sizes_text = ", ".join(
                f"{name}={indicator_pop_sizes[name]}"
                for name in sorted(indicator_pop_sizes)
            )
            print(f"{pair} | {interval} computed pop sizes -> {pop_sizes_text}")

            for horizon_minutes in proxy_target_horizonts:
                target_col = _build_proxy_target_col_name(
                    horizon_minutes=int(horizon_minutes),
                    target_mode=proxy_target_mode,
                )
                if proxy_target_mode == "ahead_ret":
                    target_np = _build_proxy_target_np_ahead_ret(
                        price_np=proxy_price_np,
                        horizon_minutes=int(horizon_minutes),
                    )
                else:
                    target_np = _build_proxy_target_np_candle_up(
                        opened_values=df[proxy_target_time_col],
                        close_values=df[proxy_target_price_col],
                        horizon_minutes=int(horizon_minutes),
                    )
                finite_target_count = int(np.isfinite(target_np).sum())
                if finite_target_count < 3:
                    raise ValueError(
                        f"Too few finite proxy target samples for horizon "
                        f"{horizon_minutes}m in {data_path / data_file}: "
                        f"{finite_target_count}"
                    )
                print(
                    f"{pair} | {interval} proxy target {target_col} "
                    f"(finite={finite_target_count}/{len(target_np)})"
                )

                for ind_name in indicator_names:
                    for metric_config in metric_configs:
                        pop_size = int(indicator_pop_sizes[ind_name])
                        filename_pop_size = int(base_pop_size)

                        out_json = results_dir / _fit_result_filename(
                            ind_name,
                            target_col,
                            filename_pop_size,
                            metric_config=metric_config,
                        )

                        if out_json.exists():
                            print(
                                f"[{interval} {ind_name} {target_col}] {out_json.name} exists - skipping."
                            )
                            continue

                        best = run_indicator_ga(
                            ind_name=ind_name,
                            ohlcv_np=ohlcv_np,
                            target_np=target_np,
                            pop_size=pop_size,
                            interval=f"{interval}|{target_col}|{_metric_filename_suffix(metric_config)}",
                            metric_config=metric_config,
                        )

                        payload = {
                            "best": best,
                            "population": {
                                "base_pop_size": filename_pop_size,
                                "adjusted_pop_size": pop_size,
                            },
                            "proxy_target": {
                                "mode": proxy_target_mode,
                                "horizon_minutes": int(horizon_minutes),
                                "price_col": proxy_target_price_col,
                                "time_col": proxy_target_time_col,
                            },
                            "metric": dict(metric_config),
                            "fit_config_hash": config_hash,
                        }
                        out_json.write_text(
                            json.dumps(payload, indent=2), encoding="utf-8"
                        )
                        print(
                            f"[{interval} {ind_name} {target_col}] saved -> {out_json}"
                        )


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total execution time: {time.time() - t0:.2f}s")
