from numpy import frombuffer, float64
from functools import lru_cache
from talib import STOCHF
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

OHLCV_BYTES = None
OHLCV_SHAPE = None
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


class StochasticOscillatorFitting(ElementwiseProblem):
    def __init__(self, *args, **kwargs):
        no_vol_ma_options = [k for k in MA_FUNCS.keys() if "VWMA" not in k]
        stoch_variables = {
            "fastK_period": Integer(bounds=(2, 1000)),
            "slowK_period": Integer(bounds=(1, 2000)),
            "slowD_period": Integer(bounds=(2, 2000)),
            "slowK_ma_type": Choice(options=no_vol_ma_options),
            "slowD_ma_type": Choice(options=no_vol_ma_options),
        }
        super().__init__(*args, vars=stoch_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        slowK, slowD = custom_stochastic_oscillator(
            ohlcv=None,
            fastK_period=X["fastK_period"],
            slowK_period=X["slowK_period"],
            slowD_period=X["slowD_period"],
            slowK_ma_type=X["slowK_ma_type"],
            slowD_ma_type=X["slowD_ma_type"],
        )
        kd_spread = slowK - slowD
        if DEBUG:
            nan_ratio, nan_count, total_count = score_nan_stats(kd_spread)
            log_nan_debug("StochOsc", dict(X), nan_ratio, nan_count, total_count)
        else:
            nan_ratio = score_nan_ratio(kd_spread)
        if nan_ratio > NAN_RATIO_THRESHOLD:
            out["F"] = NAN_PENALTY + nan_ratio
            return
        target = frombuffer(TARGET_BYTES, dtype=float64).reshape(TARGET_SHAPE)
        out["F"] = -extremes_vs_mid_ir_oof(
            kd_spread,
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


def get_stochastic_oscillator_values(params, ohlcv_np):
    slowK, slowD = custom_stochastic_oscillator(
        ohlcv_np,
        fastK_period=params["fastK_period"],
        slowK_period=params["slowK_period"],
        slowD_period=params["slowD_period"],
        slowK_ma_type=params["slowK_ma_type"],
        slowD_ma_type=params["slowD_ma_type"],
    )
    return slowK - slowD


def get_stochastic_oscillator_features(params, ohlcv_np):
    kd = get_stochastic_oscillator_values(params, ohlcv_np)
    base_name = (
        f"stoch_fK{params['fastK_period']}_sK{params['slowK_period']}_sD{params['slowD_period']}_"
        f"maK{params['slowK_ma_type']}_maD{params['slowD_ma_type']}"
    )
    return {f"{base_name}_kd_spread": kd}


def stochastic_oscillator_initializer(
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
    global OHLCV_BYTES, OHLCV_SHAPE, TARGET_BYTES, TARGET_SHAPE
    global METRIC_SEGMENTS_COUNT, METRIC_TRAIN_FRAC, METRIC_GAP
    global METRIC_Q_EXT, METRIC_Q_MID, METRIC_STAT, METRIC_CLIP_Q
    global METRIC_MIN_BUCKET_SIZE, METRIC_MIN_VALID_SEGMENTS
    global METRIC_RECENCY_WEIGHTING_ENABLED, METRIC_RECENCY_WEIGHTING_MODE
    global METRIC_RECENCY_WEIGHT_MIN, METRIC_RECENCY_WEIGHT_MAX
    # Reset memoized arrays to avoid stale cache hits after re-initialization
    # in the same process.
    stochf_cache.cache_clear()
    OHLCV_BYTES = ohlcv.tobytes()
    OHLCV_SHAPE = ohlcv.shape
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


@lru_cache(maxsize=64)
def stochf_cache(fastk_period):
    ohlcv_array = frombuffer(OHLCV_BYTES).reshape(OHLCV_SHAPE)
    fastK, _ = STOCHF(
        *ohlcv_array[:, 1:4].T,
        fastk_period=fastk_period,
        fastd_period=1,
        fastd_matype=0,
    )
    return fastK


def custom_stochastic_oscillator(
    ohlcv, fastK_period, slowK_period, slowD_period, slowK_ma_type, slowD_ma_type
):
    if ohlcv is None:
        fastK = stochf_cache(fastK_period)
    else:
        fastK, _ = STOCHF(
            *ohlcv[:, 1:4].T, fastk_period=fastK_period, fastd_period=1, fastd_matype=0
        )

    slowK = (
        fastK if slowK_period == 1 else get_1d_ma(fastK, slowK_ma_type, slowK_period)
    )
    return slowK, get_1d_ma(slowK, slowD_ma_type, slowD_period)
