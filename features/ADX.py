from numpy import (
    roll,
    where,
    divide,
    maximum,
    clip,
    zeros_like,
    array,
    frombuffer,
    concatenate,
    zeros,
)
from numpy import float64
from talib import TRANGE
from functools import lru_cache
from pymoo.core.variable import Real, Integer, Choice
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

TRANGE_BYTES = None
TRANGE_SHAPE = None
UP_MOVE_BYTES = None
UP_MOVE_SHAPE = None
DOWN_MOVE_BYTES = None
DOWN_MOVE_SHAPE = None

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


class ADXFitting(ElementwiseProblem):
    def __init__(self, *args, **kwargs):
        no_vol_ma_options = [k for k in MA_FUNCS.keys() if "VWMA" not in k]
        adx_variables = {
            "atr_period": Integer(bounds=(2, 2000)),
            "posDM_period": Integer(bounds=(2, 2000)),
            "negDM_period": Integer(bounds=(2, 2000)),
            "adx_period": Integer(bounds=(2, 2000)),
            "ma_type_atr": Choice(options=no_vol_ma_options),
            "ma_type_posDM": Choice(options=no_vol_ma_options),
            "ma_type_negDM": Choice(options=no_vol_ma_options),
            "ma_type_adx": Choice(options=no_vol_ma_options),
            # "adx_threshold": Real(bounds=(0, 100)), # PROJ ADJUSTED
        }
        super().__init__(*args, vars=adx_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        adx, _, _ = custom_adx(
            None,
            atr_period=X["atr_period"],
            posDM_period=X["posDM_period"],
            negDM_period=X["negDM_period"],
            adx_period=X["adx_period"],
            ma_type_atr=X["ma_type_atr"],
            ma_type_posDM=X["ma_type_posDM"],
            ma_type_negDM=X["ma_type_negDM"],
            ma_type_adx=X["ma_type_adx"],
        )
        if DEBUG:
            # score = adx - float(X["adx_threshold"]) # PROJ ADJUSTED
            score = adx
            nan_ratio, nan_count, total_count = score_nan_stats(score)
            log_nan_debug("ADX", dict(X), nan_ratio, nan_count, total_count)
        else:
            nan_ratio = score_nan_ratio(adx)
        if nan_ratio > NAN_RATIO_THRESHOLD:
            out["F"] = NAN_PENALTY + nan_ratio
            return
        target = frombuffer(TARGET_BYTES, dtype=float64).reshape(TARGET_SHAPE)
        # The metric depends on bucket assignment by x, so subtracting
        # a scalar threshold does not change the objective value.
        out["F"] = -extremes_vs_mid_ir_oof(
            adx,
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


def get_adx_values(params, ohlcv_np):
    adx, _, _ = custom_adx(
        ohlcv_np,
        atr_period=params["atr_period"],
        posDM_period=params["posDM_period"],
        negDM_period=params["negDM_period"],
        adx_period=params["adx_period"],
        ma_type_atr=params["ma_type_atr"],
        ma_type_posDM=params["ma_type_posDM"],
        ma_type_negDM=params["ma_type_negDM"],
        ma_type_adx=params["ma_type_adx"],
    )
    return adx
    # return adx - float(params["adx_threshold"]) # PROJ ADJUSTED


def get_adx_features(params, ohlcv_np):
    adx, plus_DI, minus_DI = custom_adx(
        ohlcv_np,
        atr_period=params["atr_period"],
        posDM_period=params["posDM_period"],
        negDM_period=params["negDM_period"],
        adx_period=params["adx_period"],
        ma_type_atr=params["ma_type_atr"],
        ma_type_posDM=params["ma_type_posDM"],
        ma_type_negDM=params["ma_type_negDM"],
        ma_type_adx=params["ma_type_adx"],
    )

    base_name = (
        f"adx_atr{params['atr_period']}_pDM{params['posDM_period']}_"
        f"nDM{params['negDM_period']}_adx{params['adx_period']}_"
        f"maATR{params['ma_type_atr']}_mapDM{params['ma_type_posDM']}_"
        f"manDM{params['ma_type_negDM']}_maADX{params['ma_type_adx']}"
    )
    thr = float(params["adx_threshold"])

    di_spread = plus_DI - minus_DI
    di_spread_prev = concatenate([zeros(1, dtype=di_spread.dtype), di_spread[:-1]])
    delta_plus = concatenate(
        [zeros(1, dtype=plus_DI.dtype), plus_DI[1:] - plus_DI[:-1]]
    )
    delta_minus = concatenate(
        [zeros(1, dtype=minus_DI.dtype), minus_DI[1:] - minus_DI[:-1]]
    )

    return {
        f"{base_name}_adx_minus_thr": adx - thr,
        f"{base_name}_di_spread": di_spread,
        f"{base_name}_di_spread_prev": di_spread_prev,
        f"{base_name}_delta_plus_di": delta_plus,
        f"{base_name}_delta_minus_di": delta_minus,
    }


@lru_cache(maxsize=16)
def custom_atr_cache(atr_ma_type, atr_period):
    true_range = frombuffer(TRANGE_BYTES).reshape(TRANGE_SHAPE)
    return get_1d_ma(true_range, atr_ma_type, atr_period)


@lru_cache(maxsize=16)
def custom_pos_dm_cache(ma_type_posDM, posDM_period):
    up_move = frombuffer(UP_MOVE_BYTES).reshape(UP_MOVE_SHAPE)
    return get_1d_ma(up_move, ma_type_posDM, posDM_period)


@lru_cache(maxsize=16)
def custom_neg_dm_cache(ma_type_negDM, negDM_period):
    down_move = frombuffer(DOWN_MOVE_BYTES).reshape(DOWN_MOVE_SHAPE)
    return get_1d_ma(down_move, ma_type_negDM, negDM_period)


def adx_initializer(
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
    global TRANGE_BYTES, TRANGE_SHAPE
    global UP_MOVE_BYTES, UP_MOVE_SHAPE
    global DOWN_MOVE_BYTES, DOWN_MOVE_SHAPE
    global TARGET_BYTES, TARGET_SHAPE
    global METRIC_SEGMENTS_COUNT, METRIC_TRAIN_FRAC, METRIC_GAP
    global METRIC_Q_EXT, METRIC_Q_MID, METRIC_STAT, METRIC_CLIP_Q
    global METRIC_MIN_BUCKET_SIZE, METRIC_MIN_VALID_SEGMENTS

    # Reset memoized arrays to avoid stale cache hits after re-initialization
    # in the same process.
    custom_atr_cache.cache_clear()
    custom_pos_dm_cache.cache_clear()
    custom_neg_dm_cache.cache_clear()

    true_range = TRANGE(*ohlcv[:, 1:4].T)
    TRANGE_BYTES = true_range.tobytes()
    TRANGE_SHAPE = true_range.shape

    prev_high = roll(ohlcv[:, 1], 1)
    prev_high[0] = ohlcv[0, 1]
    prev_low = roll(ohlcv[:, 2], 1)
    prev_low[0] = ohlcv[0, 2]

    high_diff = ohlcv[:, 1] - prev_high
    low_diff = prev_low - ohlcv[:, 2]
    up_move = where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    down_move = where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    UP_MOVE_BYTES = up_move.tobytes()
    UP_MOVE_SHAPE = up_move.shape
    DOWN_MOVE_BYTES = down_move.tobytes()
    DOWN_MOVE_SHAPE = down_move.shape

    target = array(target, dtype=float64, copy=False)
    TARGET_BYTES = target.tobytes()
    TARGET_SHAPE = target.shape
    METRIC_SEGMENTS_COUNT = max(1, int(metric_segments_count))
    METRIC_TRAIN_FRAC = float(metric_train_frac)
    METRIC_GAP = max(0, int(metric_gap))
    METRIC_Q_EXT = float(q_ext)
    METRIC_Q_MID = float(q_mid)
    METRIC_STAT = str(stat)
    METRIC_CLIP_Q = float(clip_q)
    METRIC_MIN_BUCKET_SIZE = max(1, int(min_bucket_size))
    METRIC_MIN_VALID_SEGMENTS = max(1, int(min_valid_segments))


def custom_adx(
    ohlcv,
    atr_period,
    posDM_period,
    negDM_period,
    adx_period,
    ma_type_atr,
    ma_type_posDM,
    ma_type_negDM,
    ma_type_adx,
):
    if ohlcv is None:
        TR_smooth = custom_atr_cache(ma_type_atr, atr_period)
        pos_DM_smooth = custom_pos_dm_cache(ma_type_posDM, posDM_period)
        neg_DM_smooth = custom_neg_dm_cache(ma_type_negDM, negDM_period)
    else:
        true_range = TRANGE(*ohlcv[:, 1:4].T)
        TR_smooth = get_1d_ma(true_range, ma_type_atr, atr_period)

        prev_high = roll(ohlcv[:, 1], 1)
        prev_high[0] = ohlcv[0, 1]
        prev_low = roll(ohlcv[:, 2], 1)
        prev_low[0] = ohlcv[0, 2]

        high_diff = ohlcv[:, 1] - prev_high
        low_diff = prev_low - ohlcv[:, 2]

        up_move = where((high_diff > low_diff) & (high_diff > 0.0), high_diff, 0.0)
        down_move = where((low_diff > high_diff) & (low_diff > 0.0), low_diff, 0.0)

        pos_DM_smooth = get_1d_ma(up_move, ma_type_posDM, posDM_period)
        neg_DM_smooth = get_1d_ma(down_move, ma_type_negDM, negDM_period)

    pos_DM_smooth = maximum(pos_DM_smooth, 0.0)
    neg_DM_smooth = maximum(neg_DM_smooth, 0.0)
    tiny = 1e-12

    plus = divide(
        pos_DM_smooth, TR_smooth, out=zeros_like(TR_smooth), where=abs(TR_smooth) > tiny
    )
    minus = divide(
        neg_DM_smooth, TR_smooth, out=zeros_like(TR_smooth), where=abs(TR_smooth) > tiny
    )
    plus_DI = 100.0 * plus
    minus_DI = 100.0 * minus

    denom = abs(plus_DI) + abs(minus_DI)
    DX = divide(
        abs(plus_DI - minus_DI), denom, out=zeros_like(denom), where=denom > tiny
    )
    DX = clip(DX, 0.0, 1.0)
    DX_raw = 100.0 * DX

    return get_1d_ma(DX_raw, ma_type_adx, adx_period), plus_DI, minus_DI
