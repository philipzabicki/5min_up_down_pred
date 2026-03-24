from numpy import frombuffer, float64
from functools import lru_cache
from pymoo.core.variable import Integer, Choice
from pymoo.core.problem import ElementwiseProblem

from .ta_tools import (
    MA_FUNCS,
    apply_ma,
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


class MACDFitting(ElementwiseProblem):
    def __init__(self, *args, **kwargs):
        all_ma_options = list(MA_FUNCS.keys())
        no_vol_ma_options = [k for k in MA_FUNCS.keys() if "VWMA" not in k]
        sources = ["open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4"]
        macd_variables = {
            "fast_source": Choice(options=sources),
            "slow_source": Choice(options=sources),
            "fast_period": Integer(bounds=(2, 1000)),
            "slow_period": Integer(bounds=(2, 2000)),
            "signal_period": Integer(bounds=(2, 2000)),
            "fast_ma_type": Choice(options=all_ma_options),
            "slow_ma_type": Choice(options=all_ma_options),
            "signal_ma_type": Choice(options=no_vol_ma_options),
        }
        super().__init__(*args, vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        macd, sig = custom_macd(
            ohlcv=None,
            fast_source=X["fast_source"],
            slow_source=X["slow_source"],
            fast_ma_type=X["fast_ma_type"],
            fast_period=X["fast_period"],
            slow_ma_type=X["slow_ma_type"],
            slow_period=X["slow_period"],
            signal_ma_type=X["signal_ma_type"],
            signal_period=X["signal_period"],
        )
        hist = macd - sig
        if DEBUG:
            nan_ratio, nan_count, total_count = score_nan_stats(hist)
            log_nan_debug("MACD", dict(X), nan_ratio, nan_count, total_count)
        else:
            nan_ratio = score_nan_ratio(hist)
        if nan_ratio > NAN_RATIO_THRESHOLD:
            out["F"] = NAN_PENALTY + nan_ratio
            return
        target = frombuffer(TARGET_BYTES, dtype=float64).reshape(TARGET_SHAPE)
        out["F"] = -extremes_vs_mid_ir_oof(
            hist,
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


def get_macd_values(params, ohlcv_np):
    macd, sig = custom_macd(
        ohlcv_np,
        fast_source=params["fast_source"],
        slow_source=params["slow_source"],
        fast_ma_type=params["fast_ma_type"],
        fast_period=params["fast_period"],
        slow_ma_type=params["slow_ma_type"],
        slow_period=params["slow_period"],
        signal_ma_type=params["signal_ma_type"],
        signal_period=params["signal_period"],
    )
    return macd - sig


def get_macd_features(params, ohlcv_np):
    hist = get_macd_values(params, ohlcv_np)
    base_name = (
        f"macd_f{params['fast_period']}_s{params['slow_period']}_sig{params['signal_period']}_"
        f"srcF{params['fast_source']}_srcS{params['slow_source']}_"
        f"maF{params['fast_ma_type']}_maS{params['slow_ma_type']}_maSig{params['signal_ma_type']}"
    )
    return {f"{base_name}_macd_hist": hist}


def macd_initializer(
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
    get_ma_from_source_cache.cache_clear()
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


@lru_cache(maxsize=64)
def get_ma_from_source_cache(ma_type, ma_period, source):
    return apply_ma(SOURCE_CACHE[source], ma_type, ma_period, VOLUME_CACHE)


def custom_macd(
    ohlcv,
    fast_source,
    slow_source,
    fast_period,
    slow_period,
    signal_period,
    fast_ma_type,
    slow_ma_type,
    signal_ma_type,
):
    if ohlcv is None:
        macd = get_ma_from_source_cache(
            fast_ma_type, fast_period, fast_source
        ) - get_ma_from_source_cache(slow_ma_type, slow_period, slow_source)
        return macd, get_1d_ma(macd, signal_ma_type, signal_period)

    local_sources = precompute_ohlcv_sources(ohlcv)
    local_volume = ohlcv[:, 4]
    macd = apply_ma(
        local_sources[fast_source], fast_ma_type, fast_period, local_volume
    ) - apply_ma(local_sources[slow_source], slow_ma_type, slow_period, local_volume)
    return macd, get_1d_ma(macd, signal_ma_type, signal_period)
