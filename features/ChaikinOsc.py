from numpy import array, frombuffer, float64
from talib import AD
from functools import lru_cache
from pymoo.core.variable import Integer, Choice
from pymoo.core.problem import ElementwiseProblem

from .ta_tools import MA_FUNCS, get_1d_ma
from .common_utils import (
    NAN_PENALTY,
    NAN_RATIO_THRESHOLD,
    extremes_vs_mid_ir_oof,
    log_nan_debug,
    score_nan_ratio,
    score_nan_stats,
)

ADL_BYTES = None
ADL_SHAPE = None
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


class ChaikinOscillatorFitting(ElementwiseProblem):
    def __init__(self, *args, **kwargs):
        no_vol_ma_options = [k for k in MA_FUNCS.keys() if "VWMA" not in k]
        chaikin_variables = {
            "fast_period": Integer(bounds=(2, 1000)),
            "slow_period": Integer(bounds=(2, 2000)),
            "fast_ma_type": Choice(options=no_vol_ma_options),
            "slow_ma_type": Choice(options=no_vol_ma_options),
        }
        super().__init__(*args, vars=chaikin_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        osc = custom_chaikin_oscillator(
            ohlcv=None,
            fast_ma_type=X["fast_ma_type"],
            fast_period=X["fast_period"],
            slow_ma_type=X["slow_ma_type"],
            slow_period=X["slow_period"],
        )
        if DEBUG:
            nan_ratio, nan_count, total_count = score_nan_stats(osc)
            log_nan_debug("ChaikinOsc", dict(X), nan_ratio, nan_count, total_count)
        else:
            nan_ratio = score_nan_ratio(osc)
        if nan_ratio > NAN_RATIO_THRESHOLD:
            out["F"] = NAN_PENALTY + nan_ratio
            return
        target = frombuffer(TARGET_BYTES, dtype=float64).reshape(TARGET_SHAPE)
        out["F"] = -extremes_vs_mid_ir_oof(
            osc,
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


def get_chaikin_oscillator_values(params, ohlcv_np):
    return custom_chaikin_oscillator(
        ohlcv_np,
        fast_ma_type=params["fast_ma_type"],
        fast_period=params["fast_period"],
        slow_ma_type=params["slow_ma_type"],
        slow_period=params["slow_period"],
    )


def get_chaikin_oscillator_features(params, ohlcv_np):
    ch = get_chaikin_oscillator_values(params, ohlcv_np)
    base_name = (
        f"chosc_f{params['fast_period']}_s{params['slow_period']}_"
        f"maF{params['fast_ma_type']}_maS{params['slow_ma_type']}"
    )
    from numpy import concatenate, zeros

    prev = concatenate([zeros(1, dtype=ch.dtype), ch[:-1]])
    return {f"{base_name}_chosc": ch, f"{base_name}_chosc_diff": ch - prev}


def adl_initializer(
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
    global ADL_BYTES, ADL_SHAPE, TARGET_BYTES, TARGET_SHAPE
    global METRIC_SEGMENTS_COUNT, METRIC_TRAIN_FRAC, METRIC_GAP
    global METRIC_Q_EXT, METRIC_Q_MID, METRIC_STAT, METRIC_CLIP_Q
    global METRIC_MIN_BUCKET_SIZE, METRIC_MIN_VALID_SEGMENTS
    global METRIC_RECENCY_WEIGHTING_ENABLED, METRIC_RECENCY_WEIGHTING_MODE
    global METRIC_RECENCY_WEIGHT_MIN, METRIC_RECENCY_WEIGHT_MAX
    # Reset memoized arrays to avoid stale cache hits after re-initialization
    # in the same process.
    get_adl_ma_cache.cache_clear()
    adl = AD(*ohlcv[:, 1:5].T)
    ADL_BYTES = adl.tobytes()
    ADL_SHAPE = adl.shape
    t = array(target, dtype=float64, copy=False)
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


@lru_cache(maxsize=64)
def get_adl_ma_cache(ma_type, period):
    adl = frombuffer(ADL_BYTES).reshape(ADL_SHAPE)
    return get_1d_ma(adl, ma_type, period)


def custom_chaikin_oscillator(
    ohlcv, fast_period, slow_period, fast_ma_type, slow_ma_type
):
    if ohlcv is None:
        return get_adl_ma_cache(fast_ma_type, fast_period) - get_adl_ma_cache(
            slow_ma_type, slow_period
        )
    adl = AD(*ohlcv[:, 1:5].T)
    return get_1d_ma(adl, fast_ma_type, fast_period) - get_1d_ma(
        adl, slow_ma_type, slow_period
    )
