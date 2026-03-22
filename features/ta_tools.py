from numpy import int64, empty, where
from numba import jit
import talib
from tindicators import ti

from .ta_custom_ma import (
    ALMA,
    HammingMA,
    LWMA,
    MGD,
    GMA_or_SMA,
    FBA,
    NadarayWatsonMA,
    VWMA,
    RMA,
    HullMA,
    LINEARREG_fast,
    SWMA_fast,
    SWMA_INV_fast,
)


MA_FUNCS = {
    # TA-Lib Standard
    "SMA": lambda s, period: talib.SMA(s, timeperiod=period),
    "EMA": lambda s, period: talib.EMA(s, timeperiod=period),
    "WMA": lambda s, period: talib.WMA(s, timeperiod=period),
    "KAMA": lambda s, period: talib.KAMA(s, timeperiod=period),
    "TRIMA": lambda s, period: talib.TRIMA(s, timeperiod=period),
    "DEMA": lambda s, period: talib.DEMA(s, timeperiod=period),
    "TEMA": lambda s, period: talib.TEMA(s, timeperiod=period),
    "T3": lambda s, period: talib.T3(s, timeperiod=period),
    "MAMA": lambda s, _: talib.MAMA(s)[0],
    # Pandas-TA & External Libs (ti)
    "EHMA": lambda s, period: ti.ehma(s, period),
    "LMA": lambda s, period: ti.lma(s, period),
    "SHMMA": lambda s, period: ti.shmma(s, period),
    "AHMA": lambda s, period: ti.ahma(s, period),
    # Custom Functions
    "LINREG": lambda s, period: LINEARREG_fast(s, period),
    "SWMA": lambda s, period: SWMA_fast(s, period),
    "SWMA_INV": lambda s, period: SWMA_INV_fast(s, period),
    "HMA": lambda s, period: HullMA(s, max(period, 4)),
    "RMA": lambda s, period: RMA(s, timeperiod=period),
    "ALMA": lambda s, period: ALMA(s, timeperiod=period),
    "HAMMING": lambda s, period: HammingMA(s, period),
    "LWMA": lambda s, period: LWMA(s, period),
    "MGD": lambda s, period: MGD(s, period),
    "GMA": lambda s, period: GMA_or_SMA(s, period), # SMA for s<=0
    "FBA": lambda s, period: FBA(s, period),
    # Nadaraya-Watson (Kernel variations)
    "NWMA_GAUSS": lambda s, period: NadarayWatsonMA(s, period, kernel=0),
    "NWMA_EPAN": lambda s, period: NadarayWatsonMA(s, period, kernel=1),
    "NWMA_UNIF": lambda s, period: NadarayWatsonMA(s, period, kernel=2),
    "NWMA_TRIA": lambda s, period: NadarayWatsonMA(s, period, kernel=3),
    "NWMA_BIW": lambda s, period: NadarayWatsonMA(s, period, kernel=4),
    "NWMA_COS": lambda s, period: NadarayWatsonMA(s, period, kernel=5),
    # Volume Weighted
    "VWMA_PTA": lambda s, period, v: VWMA(s, v, timeperiod=period), # legacy name, same as "VWMA"
    "VWMA": lambda s, period, v: VWMA(s, v, timeperiod=period),
}


# Other
@jit(nopython=True, nogil=True, cache=True)
def extract_segments_indices(signal):
    n = signal.shape[0]
    segments_out = empty((n // 2, 2), dtype=int64)
    seg_count = 0
    i = 0
    while i < n:
        if signal[i] == 1:
            start = i
            i += 1
            while i < n and signal[i] != -1:
                i += 1
            if i < n:  # Found -1
                segments_out[seg_count, 0] = start
                segments_out[seg_count, 1] = i
                seg_count += 1
        else:
            i += 1
    return segments_out[:seg_count]


# Main functions
def get_1d_ma(close, ma_type, ma_period):
    # print(f'Getting MA type {ma_type} period {ma_period} close type {type(close)} close shape {close.shape}')
    return MA_FUNCS[ma_type](close, int(ma_period))


def get_source_from_ohlcv(np_df, source="close"):
    if source == "close":
        return np_df[:, 3]
    if source == "open":
        return np_df[:, 0]
    if source == "high":
        return np_df[:, 1]
    if source == "low":
        return np_df[:, 2]
    if source == "hl2":
        return (np_df[:, 1] + np_df[:, 2]) * 0.5
    if source == "hlc3":
        return (np_df[:, 1] + np_df[:, 2] + np_df[:, 3]) / 3.0
    if source == "ohlc4":
        return (np_df[:, 0] + np_df[:, 1] + np_df[:, 2] + np_df[:, 3]) / 4.0
    if source == "hlcc4":
        return (np_df[:, 1] + np_df[:, 2] + 2.0 * np_df[:, 3]) / 4.0
    if source == "vwap":
        if np_df.shape[1] <= 5:
            raise ValueError("Source 'vwap' requires OHLCV array with column index 5.")
        return np_df[:, 5]
    raise ValueError(f"Unknown source column: {source}")


def precompute_ohlcv_sources(np_df):
    out = {
        "open": np_df[:, 0],
        "high": np_df[:, 1],
        "low": np_df[:, 2],
        "close": np_df[:, 3],
    }
    out["hl2"] = (out["high"] + out["low"]) * 0.5
    out["hlc3"] = (out["high"] + out["low"] + out["close"]) / 3.0
    out["ohlc4"] = (out["open"] + out["high"] + out["low"] + out["close"]) / 4.0
    out["hlcc4"] = (out["high"] + out["low"] + 2.0 * out["close"]) / 4.0
    if np_df.shape[1] > 5:
        out["vwap"] = np_df[:, 5]
    return out


def apply_ma(source_arr, ma_type, ma_period, volume_arr=None):
    period = int(ma_period)
    if ma_type not in ("VWMA_PTA", "VWMA"):
        return MA_FUNCS[ma_type](source_arr, period)
    if volume_arr is None:
        raise ValueError(f"MA type {ma_type} requires volume array.")
    return MA_FUNCS[ma_type](source_arr, period, volume_arr)


def get_ma_from_source(np_df, ma_type, ma_period, source="close"):
    source_arr = get_source_from_ohlcv(np_df, source)
    return apply_ma(source_arr, ma_type, ma_period, np_df[:, 4])


def get_std_from_source(np_df, std_period, source="close"):
    s = get_source_from_ohlcv(np_df, source)
    return talib.STDDEV(s, timeperiod=int(std_period), nbdev=1.0)


## ADX signal functions
@jit(nopython=True, nogil=True, cache=True)
def ADX_trend_signal(adx_col, minus_di, plus_di, adx_threshold):
    return [
        (
            1
            if (adx >= adx_threshold) and (pDI > mDI)
            else -1 if (adx >= adx_threshold) and (pDI < mDI) else 0
        )
        for adx, mDI, pDI in zip(adx_col, minus_di, plus_di)
    ]


@jit(nopython=True, nogil=True, cache=True)
def ADX_DIs_cross_above_threshold(adx_col, minus_di, plus_di, adx_threshold):
    return [0] + [
        (
            0
            if adx < adx_threshold
            else (
                1
                if (cur_pDI > cur_mDI) and (prev_pDI < prev_mDI)
                else -1 if (cur_pDI < cur_mDI) and (prev_pDI > prev_mDI) else 0
            )
        )
        for cur_pDI, cur_mDI, adx, prev_pDI, prev_mDI in zip(
            plus_di[1:], minus_di[1:], adx_col[1:], plus_di[:-1], minus_di[:-1]
        )
    ]


@jit(nopython=True, nogil=True, cache=True)
def ADX_DIs_approaching_cross_above_threshold(
    adx_col, minus_di, plus_di, adx_threshold
):
    return [0] + [
        (
            0
            if adx < adx_threshold
            else (
                1
                if (cur_pDI > prev_pDI) and (cur_mDI < prev_mDI) and (cur_pDI < cur_mDI)
                else (
                    -1
                    if (cur_pDI < prev_pDI)
                    and (cur_mDI > prev_mDI)
                    and (cur_pDI > cur_mDI)
                    else 0
                )
            )
        )
        for cur_pDI, cur_mDI, adx, prev_pDI, prev_mDI in zip(
            plus_di[1:], minus_di[1:], adx_col[1:], plus_di[:-1], minus_di[:-1]
        )
    ]


## Chaikin Oscillator signal functions
@jit(nopython=True, nogil=True, cache=True)
def ChaikinOscillator_signal(
    chaikin_oscillator,
):
    return [0] + [
        1 if cur_chosc > 0 > prev_chosc else -1 if cur_chosc < 0 < prev_chosc else 0
        for cur_chosc, prev_chosc in zip(
            chaikin_oscillator[1:], chaikin_oscillator[:-1]
        )
    ]


## Keltner Channel signal function
# @jit(nopython=True, nogil=True, cache=True)
# def keltner_channel_signal(np_close, np_xMA, np_ATR, atr_multi=1.0):
#     return where(np_ATR != 0, (np_xMA - np_close) / (np_ATR * atr_multi), 0)

## PROJ ADJUSTED Keltner Channel signal function (skipping atr_multi)
@jit(nopython=True, nogil=True, cache=True)
def keltner_channel_signal(np_close, np_xMA, np_ATR, atr_multi):
    numerator = np_xMA - np_close
    return where(
        np_ATR == 0,
        where(numerator == 0, 0.0, float("nan")),
        numerator / np_ATR,
    )

## Bollinger Bands signal function
# @jit(nopython=True, nogil=True, cache=True)
# def bollinger_channel_signal(np_close, np_xMA, np_STD, std_multi=1.0):
#     return where(np_STD != 0, (np_xMA - np_close) / (np_STD * std_multi), 0)

## PROJ ADJUSTED Bollinger Bands signal function (skipping std_multi)
@jit(nopython=True, nogil=True, cache=True)
def bollinger_channel_signal(np_close, np_xMA, np_STD, std_multi):
    numerator = np_xMA - np_close
    return where(
        np_STD == 0,
        where(numerator == 0, 0.0, float("nan")),
        numerator / np_STD,
    )


## MACD signal functions
@jit(nopython=True, nogil=True, cache=True)
def MACD_lines_cross_with_zero(macd_col, signal_col):
    return [0] + [
        (
            1
            if (cur_sig < 0)
            and (cur_macd < 0)
            and (cur_macd > cur_sig)
            and (prev_macd < prev_sig)
            else (
                -1
                if (cur_sig > 0)
                and (cur_macd > 0)
                and (cur_macd < cur_sig)
                and (prev_macd > prev_sig)
                else 0
            )
        )
        for cur_sig, cur_macd, prev_sig, prev_macd in zip(
            signal_col[1:], macd_col[1:], signal_col[:-1], macd_col[:-1]
        )
    ]


@jit(nopython=True, nogil=True, cache=True)
def MACD_lines_cross(macd_col, signal_col):
    return [0] + [
        (
            1
            if (cur_macd > cur_sig) and (prev_macd < prev_sig)
            else -1 if (cur_macd < cur_sig) and (prev_macd > prev_sig) else 0
        )
        for cur_sig, cur_macd, prev_sig, prev_macd in zip(
            signal_col[1:], macd_col[1:], signal_col[:-1], macd_col[:-1]
        )
    ]


@jit(nopython=True, nogil=True, cache=True)
def MACD_lines_approaching_cross_with_zero(macd_col, signal_col):
    return [0] + [
        (
            1
            if (cur_sig < 0)
            and (cur_macd < 0)
            and (cur_macd > prev_macd)
            and (cur_sig < prev_sig)
            else (
                -1
                if (cur_sig > 0)
                and (cur_macd > 0)
                and (cur_macd < prev_macd)
                and (cur_sig > prev_sig)
                else 0
            )
        )
        for cur_sig, cur_macd, prev_sig, prev_macd in zip(
            signal_col[1:], macd_col[1:], signal_col[:-1], macd_col[:-1]
        )
    ]


@jit(nopython=True, nogil=True, cache=True)
def MACD_lines_approaching_cross(macd_col, signal_col):
    return [0] + [
        (
            1
            if (cur_macd > prev_macd) and (cur_sig < prev_sig)
            else -1 if (cur_macd < prev_macd) and (cur_sig > prev_sig) else 0
        )
        for cur_sig, cur_macd, prev_sig, prev_macd in zip(
            signal_col[1:], macd_col[1:], signal_col[:-1], macd_col[:-1]
        )
    ]


@jit(nopython=True, nogil=True, cache=True)
def MACD_signal_line_zero_cross(_, signal_col):
    return [0] + [
        1 if cur_sig > 0 > prev_sig else -1 if cur_sig < 0 < prev_sig else 0
        for cur_sig, prev_sig in zip(signal_col[1:], signal_col[:-1])
    ]


@jit(nopython=True, nogil=True, cache=True)
def MACD_line_zero_cross(macd_col, _):
    return [0] + [
        1 if cur_macd > 0 > prev_macd else -1 if cur_macd < 0 < prev_macd else 0
        for cur_macd, prev_macd in zip(macd_col[1:], macd_col[:-1])
    ]


@jit(nopython=True, nogil=True, cache=True)
def MACD_histogram_reversal(macd_col, signal_col):
    macdhist_col = macd_col - signal_col
    return [0] * 3 + [
        (
            1
            if cur_macd > prev_macd and prev_macd < preprev_macd < prepreprev_macd
            else (
                -1
                if cur_macd < prev_macd and prev_macd > preprev_macd > prepreprev_macd
                else 0
            )
        )
        for cur_macd, prev_macd, preprev_macd, prepreprev_macd in zip(
            macdhist_col[3:], macdhist_col[2:-1], macdhist_col[1:-2], macdhist_col[:-3]
        )
    ]


## Stochastic Oscillator signal functions
@jit(nopython=True, nogil=True, cache=True)
def k_int_cross(
    k_line,
    d_line=None,
    oversold_threshold=20.0,
    overbought_threshold=80.0,
):
    return [0] + [
        (
            1
            if (prev > oversold_threshold > curr)
            else -1 if (prev < overbought_threshold < curr) else 0
        )
        for curr, prev in zip(k_line[1:], k_line[:-1])
    ]


# 2. k_ext_cross: k-line external threshold cross
@jit(nopython=True, nogil=True, cache=True)
def k_ext_cross(
    k_line,
    d_line=None,
    oversold_threshold=20.0,
    overbought_threshold=80.0,
):
    return [0] + [
        (
            1
            if (prev < oversold_threshold < curr)
            else -1 if (prev > overbought_threshold > curr) else 0
        )
        for curr, prev in zip(k_line[1:], k_line[:-1])
    ]


# 3. d_int_cross: d-line internal threshold cross
@jit(nopython=True, nogil=True, cache=True)
def d_int_cross(
    k_line,
    d_line=None,
    oversold_threshold=20.0,
    overbought_threshold=80.0,
):
    return [0] + [
        (
            1
            if (prev > oversold_threshold > curr)
            else -1 if (prev < overbought_threshold < curr) else 0
        )
        for curr, prev in zip(d_line[1:], d_line[:-1])
    ]


# 4. d_ext_cross: d-line external threshold cross
@jit(nopython=True, nogil=True, cache=True)
def d_ext_cross(
    k_line,
    d_line=None,
    oversold_threshold=20.0,
    overbought_threshold=80.0,
):
    return [0] + [
        (
            1
            if (prev < oversold_threshold < curr)
            else -1 if (prev > overbought_threshold > curr) else 0
        )
        for curr, prev in zip(d_line[1:], d_line[:-1])
    ]


# 5. k_cross_int_os_ext_ob:
#    Buy signal when k-line crosses oversold internally; sell signal when k-line crosses overbought externally.
@jit(nopython=True, nogil=True, cache=True)
def k_cross_int_os_ext_ob(
    k_line,
    d_line=None,
    oversold_threshold=20.0,
    overbought_threshold=80.0,
):
    return [0] + [
        (
            1
            if (prev > oversold_threshold > curr)
            else -1 if (prev > overbought_threshold > curr) else 0
        )
        for curr, prev in zip(k_line[1:], k_line[:-1])
    ]


# 6. k_cross_ext_os_int_ob:
#    Buy signal when k-line crosses oversold externally; sell signal when k-line crosses overbought internally.
@jit(nopython=True, nogil=True, cache=True)
def k_cross_ext_os_int_ob(
    k_line,
    d_line=None,
    oversold_threshold=20.0,
    overbought_threshold=80.0,
):
    return [0] + [
        (
            1
            if (prev < oversold_threshold < curr)
            else -1 if (prev < overbought_threshold < curr) else 0
        )
        for curr, prev in zip(k_line[1:], k_line[:-1])
    ]


# 7. d_cross_int_os_ext_ob:
#    Buy signal when d-line crosses oversold internally; sell signal when d-line crosses overbought externally.
@jit(nopython=True, nogil=True, cache=True)
def d_cross_int_os_ext_ob(
    k_line,
    d_line=None,
    oversold_threshold=20.0,
    overbought_threshold=80.0,
):
    return [0] + [
        (
            1
            if (prev > oversold_threshold > curr)
            else -1 if (prev > overbought_threshold > curr) else 0
        )
        for curr, prev in zip(d_line[1:], d_line[:-1])
    ]


# 8. d_cross_ext_os_int_ob:
#    Buy signal when d-line crosses oversold externally; sell signal when d-line crosses overbought internally.
@jit(nopython=True, nogil=True, cache=True)
def d_cross_ext_os_int_ob(
    k_line,
    d_line=None,
    oversold_threshold=20.0,
    overbought_threshold=80.0,
):
    return [0] + [
        (
            1
            if (prev < oversold_threshold < curr)
            else -1 if (prev < overbought_threshold < curr) else 0
        )
        for curr, prev in zip(d_line[1:], d_line[:-1])
    ]


# 9. kd_cross: Generic crossing between k and d lines (no threshold limits)
@jit(nopython=True, nogil=True, cache=True)
def kd_cross(
    k_line,
    d_line=None,
    oversold_threshold=20.0,
    overbought_threshold=80.0,
):
    return [0] + [
        (
            1
            if (prev_k < prev_d and curr_k > curr_d)
            else -1 if (prev_k > prev_d and curr_k < curr_d) else 0
        )
        for curr_k, curr_d, prev_k, prev_d in zip(
            k_line[1:], d_line[1:], k_line[:-1], d_line[:-1]
        )
    ]


# 10. kd_cross_inside: k and d cross when both are within the threshold range (between oversold and overbought)
@jit(nopython=True, nogil=True, cache=True)
def kd_cross_inside(
    k_line,
    d_line=None,
    oversold_threshold=20.0,
    overbought_threshold=80.0,
):
    return [0] + [
        (
            1
            if (
                prev_k < prev_d
                and curr_k > curr_d
                and (oversold_threshold < curr_k < overbought_threshold)
                and (oversold_threshold < curr_d < overbought_threshold)
            )
            else (
                -1
                if (
                    prev_k > prev_d
                    and curr_k < curr_d
                    and (oversold_threshold < curr_k < overbought_threshold)
                    and (oversold_threshold < curr_d < overbought_threshold)
                )
                else 0
            )
        )
        for curr_k, curr_d, prev_k, prev_d in zip(
            k_line[1:], d_line[1:], k_line[:-1], d_line[:-1]
        )
    ]


# 11. kd_cross_outside: k and d cross when both are outside the threshold range on the same side
@jit(nopython=True, nogil=True, cache=True)
def kd_cross_outside(
    k_line,
    d_line=None,
    oversold_threshold=20.0,
    overbought_threshold=80.0,
):
    return [0] + [
        (
            1
            if (
                prev_k < prev_d
                and curr_k > curr_d
                and (curr_k < oversold_threshold and curr_d < oversold_threshold)
            )
            else (
                -1
                if (
                    prev_k > prev_d
                    and curr_k < curr_d
                    and (
                        curr_k > overbought_threshold and curr_d > overbought_threshold
                    )
                )
                else 0
            )
        )
        for curr_k, curr_d, prev_k, prev_d in zip(
            k_line[1:], d_line[1:], k_line[:-1], d_line[:-1]
        )
    ]
