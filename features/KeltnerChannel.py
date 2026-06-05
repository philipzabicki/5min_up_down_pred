from numpy import frombuffer, float64
from functools import lru_cache
from talib import TRANGE
from pymoo.core.variable import Integer, Real, Choice
from pymoo.core.problem import ElementwiseProblem

from .ta_tools import (
    MA_FUNCS,
    apply_ma,
    get_1d_ma,
    keltner_channel_signal,
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
TRANGE_BYTES = None
TRANGE_SHAPE = None
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
METRIC_RECENCY_WEIGHTING_ENABLED = False
METRIC_RECENCY_WEIGHTING_MODE = "linear"
METRIC_RECENCY_WEIGHT_MIN = 1.0
METRIC_RECENCY_WEIGHT_MAX = 1.5
DEBUG = False


class KeltnerChannelFitting(ElementwiseProblem):
    def __init__(self, *args, **kwargs):
        all_ma_options = list(MA_FUNCS.keys())
        no_vol_ma_options = [k for k in MA_FUNCS.keys() if "VWMA" not in k]
        bands_variables = {
            "ma_type": Choice(options=all_ma_options),
            "atr_ma_type": Choice(options=no_vol_ma_options),
            "ma_period": Integer(bounds=(2, 2000)),
            "atr_period": Integer(bounds=(2, 2000)),
            # "atr_multi": Real(bounds=(0.001, 15.000)), # PROJ ADJUSTED
            "source": Choice(
                options=[
                    "open",
                    "high",
                    "low",
                    "close",
                    "hl2",
                    "hlc3",
                    "ohlc4",
                    "hlcc4",
                ]
            ),
        }
        super().__init__(*args, vars=bands_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        kc_z = custom_keltner_channel(
            ohlcv=None,
            ma_type=X["ma_type"],
            ma_period=X["ma_period"],
            atr_ma_type=X["atr_ma_type"],
            atr_period=X["atr_period"],
            # atr_multi=X["atr_multi"], # PROJ ADJUSTED
            source=X["source"],
        )
        if DEBUG:
            nan_ratio, nan_count, total_count = score_nan_stats(kc_z)
            log_nan_debug("KeltnerChannel", dict(X), nan_ratio, nan_count, total_count)
        else:
            nan_ratio = score_nan_ratio(kc_z)
        if nan_ratio > NAN_RATIO_THRESHOLD:
            out["F"] = NAN_PENALTY + nan_ratio
            return
        target = frombuffer(TARGET_BYTES, dtype=float64).reshape(TARGET_SHAPE)
        out["F"] = -extremes_vs_mid_ir_oof(
            kc_z,
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
            recency_weighting_enabled=METRIC_RECENCY_WEIGHTING_ENABLED,
            recency_weighting_mode=METRIC_RECENCY_WEIGHTING_MODE,
            recency_weight_min=METRIC_RECENCY_WEIGHT_MIN,
            recency_weight_max=METRIC_RECENCY_WEIGHT_MAX,
        )


def get_keltner_channel_values(params, ohlcv_np):
    return custom_keltner_channel(
        ohlcv=ohlcv_np,
        ma_type=params["ma_type"],
        ma_period=params["ma_period"],
        atr_ma_type=params["atr_ma_type"],
        atr_period=params["atr_period"],
        # atr_multi=params["atr_multi"], # PROJ ADJUSTED
        source=params["source"],
    )


def get_keltner_channel_features(params, ohlcv_np):
    kc_z = get_keltner_channel_values(params, ohlcv_np)
    base_name = (
        f"kc_ma{params['ma_type']}{params['ma_period']}_"
        f"atr{params['atr_ma_type']}{params['atr_period']}_"
        f"mul{params['atr_multi']}_src{params['source']}"
    )
    return {f"{base_name}_kc_z": kc_z}


def keltner_channel_initializer(
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
    recency_weighting_enabled=False,
    recency_weighting_mode="linear",
    recency_weight_min=1.0,
    recency_weight_max=1.5,
):
    global TRANGE_BYTES, TRANGE_SHAPE, OHLCV_BYTES, OHLCV_SHAPE
    global SOURCE_CACHE, VOLUME_CACHE, TARGET_BYTES, TARGET_SHAPE
    global METRIC_SEGMENTS_COUNT, METRIC_TRAIN_FRAC, METRIC_GAP
    global METRIC_Q_EXT, METRIC_Q_MID, METRIC_STAT, METRIC_CLIP_Q
    global METRIC_MIN_BUCKET_SIZE, METRIC_MIN_VALID_SEGMENTS
    global METRIC_RECENCY_WEIGHTING_ENABLED, METRIC_RECENCY_WEIGHTING_MODE
    global METRIC_RECENCY_WEIGHT_MIN, METRIC_RECENCY_WEIGHT_MAX
    # Reset memoized arrays to avoid stale cache hits after re-initialization
    # in the same process.
    get_ma_from_source_cache.cache_clear()
    custom_atr_cache.cache_clear()
    tr = TRANGE(*ohlcv[:, 1:4].T)
    TRANGE_BYTES = tr.tobytes()
    TRANGE_SHAPE = tr.shape
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
    METRIC_RECENCY_WEIGHTING_ENABLED = bool(recency_weighting_enabled)
    METRIC_RECENCY_WEIGHTING_MODE = str(recency_weighting_mode)
    METRIC_RECENCY_WEIGHT_MIN = float(recency_weight_min)
    METRIC_RECENCY_WEIGHT_MAX = float(recency_weight_max)


@lru_cache(maxsize=24)
def get_ma_from_source_cache(ma_type, ma_period, source):
    return apply_ma(SOURCE_CACHE[source], ma_type, ma_period, VOLUME_CACHE)


@lru_cache(maxsize=24)
def custom_atr_cache(atr_ma_type, atr_period):
    true_range = frombuffer(TRANGE_BYTES).reshape(TRANGE_SHAPE)
    return get_1d_ma(true_range, atr_ma_type, atr_period)


def custom_keltner_channel(
    ohlcv,
    ma_type,
    ma_period,
    atr_ma_type,
    atr_period,
    source,
    atr_multi=1.0,
):
    if ohlcv is None:
        return keltner_channel_signal(
            np_close=SOURCE_CACHE["close"],
            np_xMA=get_ma_from_source_cache(ma_type, ma_period, source),
            np_ATR=custom_atr_cache(atr_ma_type, atr_period),
            atr_multi=atr_multi,
        )

    local_sources = precompute_ohlcv_sources(ohlcv)
    local_volume = ohlcv[:, 4]
    return keltner_channel_signal(
        np_close=ohlcv[:, 3],
        np_xMA=apply_ma(local_sources[source], ma_type, ma_period, local_volume),
        np_ATR=get_1d_ma(TRANGE(*ohlcv[:, 1:4].T), atr_ma_type, atr_period),
        atr_multi=atr_multi,
    )
