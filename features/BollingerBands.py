from numpy import frombuffer, sqrt, float64, where
from functools import lru_cache
from pymoo.core.variable import Integer, Real, Choice
from pymoo.core.problem import ElementwiseProblem

from .ta_tools import (
    MA_FUNCS,
    apply_ma,
    bollinger_channel_signal,
    get_1d_ma,
    precompute_ohlcv_sources,
)
from .common_utils import (
    NAN_PENALTY,
    NAN_RATIO_THRESHOLD,
    extremes_vs_mid_ir_oof,
    log_nan_debug,
    score_nan_ratio,
    score_nan_stats,
)

OHLCV_BYTES = None
OHLCV_SHAPE = None
SOURCE_CACHE = None
VOLUME_CACHE = None
TARGET_BYTES = None
TARGET_SHAPE = None
METRIC_SEGMENTS_COUNT = 12
METRIC_TRAIN_FRAC = 0.80
METRIC_GAP = 1500
METRIC_Q_EXT = 0.10
METRIC_Q_MID = 0.10
METRIC_STAT = "mean_clip"
METRIC_CLIP_Q = 0.01
METRIC_MIN_BUCKET_SIZE = 50
METRIC_MIN_VALID_SEGMENTS = 2
DEBUG = False


class BollingerBandsFitting(ElementwiseProblem):
    def __init__(self, *args, **kwargs):
        all_ma_options = list(MA_FUNCS.keys())
        no_vol_ma_options = [k for k in MA_FUNCS.keys() if "VWMA" not in k]
        price_sources = [
            "open",
            "high",
            "low",
            "close",
            "hl2",
            "hlc3",
            "ohlc4",
            "hlcc4",
        ]

        bands_variables = {
            "ma_type": Choice(options=all_ma_options),
            "ma_period": Integer(bounds=(2, 2000)),
            "ma_source": Choice(options=price_sources),
            "std_ma_type": Choice(options=no_vol_ma_options),
            "std_ma_period": Integer(bounds=(2, 2000)),
            "std_source": Choice(options=price_sources),
            # "std_multi": Real(bounds=(0.001, 15.000)), # PROJ ADJUSTED
        }
        super().__init__(*args, vars=bands_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        bb_z = custom_bollinger_bands(
            ohlcv=None,
            ma_type=X["ma_type"],
            ma_period=X["ma_period"],
            ma_source=X["ma_source"],
            std_ma_type=X["std_ma_type"],
            std_ma_period=X["std_ma_period"],
            std_source=X["std_source"],
            # std_multi=X["std_multi"], # PROJ ADJUSTED
        )
        if DEBUG:
            nan_ratio, nan_count, total_count = score_nan_stats(bb_z)
            log_nan_debug("BollingerBands", dict(X), nan_ratio, nan_count, total_count)
        else:
            nan_ratio = score_nan_ratio(bb_z)
        if nan_ratio > NAN_RATIO_THRESHOLD:
            out["F"] = NAN_PENALTY + nan_ratio
            return
        target = frombuffer(TARGET_BYTES, dtype=float64).reshape(TARGET_SHAPE)
        out["F"] = -extremes_vs_mid_ir_oof(
            bb_z,
            target,
            segments_count=METRIC_SEGMENTS_COUNT,
            train_frac=METRIC_TRAIN_FRAC,
            gap=METRIC_GAP,
            q_ext=METRIC_Q_EXT,
            q_mid=METRIC_Q_MID,
            stat=METRIC_STAT,
            clip_q=METRIC_CLIP_Q,
            min_bucket_size=METRIC_MIN_BUCKET_SIZE,
            min_valid_segments=METRIC_MIN_VALID_SEGMENTS,
        )


def get_bollinger_bands_values(params, ohlcv_np):
    return custom_bollinger_bands(
        ohlcv=ohlcv_np,
        ma_type=params["ma_type"],
        ma_period=params["ma_period"],
        ma_source=params["ma_source"],
        std_ma_type=params["std_ma_type"],
        std_ma_period=params["std_ma_period"],
        std_source=params["std_source"],
        # std_multi=params["std_multi"], # PROJ ADJUSTED
    )


def get_bollinger_bands_features(params, ohlcv_np):
    bb_z = get_bollinger_bands_values(params, ohlcv_np)
    base_name = (
        f"bb_ma{params['ma_type']}{params['ma_period']}_"
        f"std{params['std_ma_type']}{params['std_ma_period']}_"
        f"mul{params['std_multi']}_"
        f"masrc{params['ma_source']}_"
        f"stdsrc{params['std_source']}"
    )
    return {f"{base_name}_bb_z": bb_z}


def bollinger_bands_initializer(
    ohlcv,
    target,
    metric_segments_count=12,
    metric_train_frac=0.80,
    metric_gap=1500,
    q_ext=0.10,
    q_mid=0.10,
    stat="mean_clip",
    clip_q=0.01,
    min_bucket_size=50,
    min_valid_segments=2,
):
    global OHLCV_BYTES, OHLCV_SHAPE, SOURCE_CACHE, VOLUME_CACHE
    global TARGET_BYTES, TARGET_SHAPE
    global METRIC_SEGMENTS_COUNT, METRIC_TRAIN_FRAC, METRIC_GAP
    global METRIC_Q_EXT, METRIC_Q_MID, METRIC_STAT, METRIC_CLIP_Q
    global METRIC_MIN_BUCKET_SIZE, METRIC_MIN_VALID_SEGMENTS
    # Reset memoized arrays to avoid stale cache hits after re-initialization
    # in the same process.
    bb_ma_cache.cache_clear()
    bb_std_cache.cache_clear()
    OHLCV_BYTES = ohlcv.tobytes()
    OHLCV_SHAPE = ohlcv.shape
    SOURCE_CACHE = precompute_ohlcv_sources(ohlcv)
    VOLUME_CACHE = ohlcv[:, 4]
    t = target.astype(float64, copy=False)
    TARGET_BYTES = t.tobytes()
    TARGET_SHAPE = t.shape
    METRIC_SEGMENTS_COUNT = max(1, int(metric_segments_count))
    METRIC_TRAIN_FRAC = float(metric_train_frac)
    METRIC_GAP = max(0, int(metric_gap))
    METRIC_Q_EXT = float(q_ext)
    METRIC_Q_MID = float(q_mid)
    METRIC_STAT = str(stat)
    METRIC_CLIP_Q = float(clip_q)
    METRIC_MIN_BUCKET_SIZE = max(1, int(min_bucket_size))
    METRIC_MIN_VALID_SEGMENTS = max(1, int(min_valid_segments))


@lru_cache(maxsize=43)
def bb_ma_cache(ma_type, ma_period, ma_source):
    return apply_ma(SOURCE_CACHE[ma_source], ma_type, ma_period, VOLUME_CACHE)


@lru_cache(maxsize=43)
def bb_std_cache(ma_type, ma_period, ma_source, std_ma_type, std_ma_period, std_source):
    center_ma = bb_ma_cache(ma_type, ma_period, ma_source)
    src = SOURCE_CACHE[std_source]
    diff2 = (src - center_ma) ** 2
    var_smooth = get_1d_ma(diff2, std_ma_type, std_ma_period)
    var_nonneg = where(var_smooth >= 0.0, var_smooth, float("nan"))
    return sqrt(var_nonneg)


def custom_bollinger_bands(
    ohlcv,
    ma_type,
    ma_period,
    ma_source,
    std_ma_type,
    std_ma_period,
    std_source,
    std_multi=1.0,
):
    if ohlcv is None:
        center_ma = bb_ma_cache(ma_type, ma_period, ma_source)
        std = bb_std_cache(
            ma_type, ma_period, ma_source, std_ma_type, std_ma_period, std_source
        )
        return bollinger_channel_signal(
            np_close=SOURCE_CACHE["close"],
            np_xMA=center_ma,
            np_STD=std,
            std_multi=std_multi,
        )

    local_sources = precompute_ohlcv_sources(ohlcv)
    local_volume = ohlcv[:, 4]
    center_ma = apply_ma(local_sources[ma_source], ma_type, ma_period, local_volume)
    src = local_sources[std_source]

    diff2 = (src - center_ma) ** 2
    var_smooth = get_1d_ma(diff2, std_ma_type, std_ma_period)
    var_nonneg = where(var_smooth >= 0.0, var_smooth, float("nan"))
    std = sqrt(var_nonneg)
    return bollinger_channel_signal(
        np_close=ohlcv[:, 3], np_xMA=center_ma, np_STD=std, std_multi=std_multi
    )
