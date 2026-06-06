import numpy as np
import talib
from numba import jit


### Helper Functions ###
@jit(nopython=True, nogil=True, cache=True)
def fib_to(n, normalization=True):
    fibs = np.empty(n)
    fibs[0], fibs[1] = 1, 2
    for i in range(2, n):
        fibs[i] = fibs[i - 1] + fibs[i - 2]
    if normalization:
        return (fibs - np.min(fibs)) / (np.max(fibs) - np.min(fibs))
    else:
        return fibs


# def HullMA(
#         close: list | np.ndarray, timeperiod: int
# ) -> ndarray[Any, dtype[floating[_64Bit]]]:
#     return talib.WMA(
#         np.nan_to_num(talib.WMA(close, timeperiod // 2) * 2)
#         - np.nan_to_num(talib.WMA(close, timeperiod)),
#         int(np.sqrt(timeperiod)),
#     )


def HullMA(close, timeperiod):
    # Obliczamy WMA dla połowy i pełnego okresu
    # Nie używamy nan_to_num - chcemy, aby NaNy się propagowały
    wma_half = talib.WMA(close, timeperiod // 2)
    wma_full = talib.WMA(close, timeperiod)

    # Obliczamy "surowe" HMA
    # Tam gdzie wma_full jest NaN, wynik też będzie NaN (poprawnie)
    raw_hma = (2 * wma_half) - wma_full

    # Ostateczne wygładzanie
    return talib.WMA(raw_hma, int(np.sqrt(timeperiod)))


@jit(nopython=True, nogil=True, cache=True)
def LINEARREG_fast(close, timeperiod):
    close = np.ascontiguousarray(close)
    n = close.shape[0]

    out = np.empty_like(close)
    out[:] = np.nan

    if timeperiod < 2 or timeperiod > n:
        return out

    L = timeperiod
    sum_x = 0.5 * L * (L - 1)
    sum_x2 = L * (L - 1) * (2 * L - 1) / 6.0
    divisor = L * sum_x2 - sum_x * sum_x
    endpoint_shift = 0.5 * (L - 1)

    sum_y = 0.0
    sum_xy = 0.0

    for k in range(L):
        y = close[k]
        sum_y += y
        sum_xy += k * y

    m = (L * sum_xy - sum_x * sum_y) / divisor
    out[L - 1] = (sum_y / L) + m * endpoint_shift

    for i in range(L, n):
        old = close[i - L]
        new = close[i]

        prev_sum_y = sum_y
        sum_y = prev_sum_y - old + new
        sum_xy = sum_xy - (prev_sum_y - old) + (L - 1) * new

        m = (L * sum_xy - sum_x * sum_y) / divisor
        out[i] = (sum_y / L) + m * endpoint_shift

    return out


def _triangle_weights(period, dtype=np.float64, inverted=False):
    if period <= 0:
        raise ValueError("period must be > 0")

    half = period // 2

    if period % 2 == 1:
        raw = np.concatenate(
            (
                np.arange(1, half + 2, dtype=np.float64),
                np.arange(half, 0, -1, dtype=np.float64),
            )
        )
    else:
        raw = np.concatenate(
            (
                np.arange(1, half + 1, dtype=np.float64),
                np.arange(half, 0, -1, dtype=np.float64),
            )
        )

    if inverted:
        raw = raw.max() + 1.0 - raw

    raw /= raw.sum()
    return raw.astype(dtype, copy=False)


def SWMA_fast(close, period):
    close = np.ascontiguousarray(close)
    n = close.shape[0]

    out = np.empty_like(close)
    out[:] = np.nan

    if period <= 0 or period > n:
        return out

    w = _triangle_weights(period, dtype=close.dtype, inverted=False)

    # np.convolve odwraca drugi wektor
    valid = np.convolve(close, w[::-1], mode="valid")
    out[period - 1:] = valid
    return out


def SWMA_INV_fast(close, period):
    close = np.ascontiguousarray(close)
    n = close.shape[0]

    out = np.empty_like(close)
    out[:] = np.nan

    if period <= 0 or period > n:
        return out

    w = _triangle_weights(period, dtype=close.dtype, inverted=True)

    # np.convolve odwraca drugi wektor
    valid = np.convolve(close, w[::-1], mode="valid")
    out[period - 1:] = valid
    return out


@jit(nopython=True, nogil=True, cache=True)
def RMA(close, timeperiod):
    close = np.ascontiguousarray(close)
    n = close.shape[0]

    out = np.empty_like(close)
    out[:] = np.nan

    if timeperiod <= 0 or timeperiod > n:
        return out

    alpha = 1.0 / timeperiod

    sma = 0.0
    for i in range(timeperiod):
        sma += close[i]
    sma /= timeperiod

    out[timeperiod - 1] = sma

    for i in range(timeperiod, n):
        out[i] = (alpha * close[i]) + ((1.0 - alpha) * out[i - 1])

    return out


@jit(nopython=True, nogil=True, cache=True)
def LSMA(close, timeperiod):
    close = np.ascontiguousarray(close)
    lsma = np.empty_like(close)
    A = np.column_stack((np.arange(timeperiod), np.ones(timeperiod)))
    AT = np.ascontiguousarray(A.T)
    ATA_inv = np.linalg.inv(np.dot(AT, A))
    for i in range(timeperiod - 1, len(close)):
        m, c = np.dot(ATA_inv, np.dot(AT, close[i - timeperiod + 1: i + 1]))
        lsma[i] = m * (timeperiod - 1) + c
    return lsma


def ALMA(close, timeperiod, offset=0.85, sigma=6):
    close = np.ascontiguousarray(close)
    n = close.shape[0]

    out = np.empty_like(close)
    out[:] = np.nan

    if timeperiod <= 0 or timeperiod > n:
        return out

    m = offset * (timeperiod - 1)
    s = timeperiod / sigma
    denom = 2.0 * s * s

    wtd = np.array(
        [np.exp(-((i - m) ** 2) / denom) for i in range(timeperiod)],
        dtype=close.dtype,
    )
    wtd /= wtd.sum()

    # convolve odwraca kernel, więc dla ALMA trzeba odwrócić wagi ręcznie
    alma_valid = np.convolve(close, wtd[::-1], "valid")
    out[timeperiod - 1:] = alma_valid
    return out


# Geometric MA cannot be computed for non-positive values, so we fall back to SMA in that case.
# Which is closest to GMA in terms of smoothing properties, but can handle any values.
def GMA_or_SMA(close, period):
    close = np.ascontiguousarray(close)

    if np.min(close) <= 0:
        return talib.SMA(close, timeperiod=period)

    return GMA(close, period)


@jit(nopython=True, nogil=True, cache=True)
def GMA(close, period):
    close_abs = np.abs(close).astype(np.float64)
    n = close_abs.shape[0]

    out_log = np.empty_like(close_abs)
    out_log[:] = np.nan

    if period <= 0 or period > n:
        return np.exp(out_log)

    eps = 1e-12
    for i in range(n):
        if close_abs[i] <= eps:
            close_abs[i] = eps

    log_close = np.log(close_abs)
    inv_period = 1.0 / period

    window_sum = 0.0
    for i in range(period):
        window_sum += log_close[i]

    out_log[period - 1] = window_sum * inv_period

    for i in range(period, n):
        window_sum -= log_close[i - period]
        window_sum += log_close[i]
        out_log[i] = window_sum * inv_period

    return np.exp(out_log)


@jit(nopython=True, nogil=True, cache=True)
def VWMAv1(close, volume, timeperiod):
    close = np.ascontiguousarray(close)
    volume = np.ascontiguousarray(volume)
    vwma = np.array(
        [
            np.sum(close[i - timeperiod: i] * volume[i - timeperiod: i])
            / np.sum(volume[i - timeperiod: i])
            for i in range(timeperiod, len(close) + 1)
        ]
    )
    return np.concatenate((np.zeros(timeperiod - 1), vwma))


@jit(nopython=True, nogil=True, cache=True)
def VWMA(close, volume, timeperiod):
    close = np.ascontiguousarray(close)
    volume = np.ascontiguousarray(volume)
    n = close.shape[0]

    out = np.empty_like(close)
    out[:] = np.nan

    if timeperiod <= 0 or timeperiod > n:
        return out

    tiny = 1e-12
    window_sum_volume = 0.0
    window_sum_cxv = 0.0
    for i in range(timeperiod):
        window_sum_volume += volume[i]
        window_sum_cxv += close[i] * volume[i]

    if window_sum_volume > tiny:
        out[timeperiod - 1] = window_sum_cxv / window_sum_volume

    for i in range(timeperiod, n):
        window_sum_volume -= volume[i - timeperiod]
        window_sum_volume += volume[i]
        window_sum_cxv -= close[i - timeperiod] * volume[i - timeperiod]
        window_sum_cxv += close[i] * volume[i]

        if window_sum_volume > tiny:
            out[i] = window_sum_cxv / window_sum_volume
        else:
            out[i] = np.nan

    return out


def HammingMA(close, timeperiod):
    close = np.ascontiguousarray(close)
    n = close.shape[0]

    out = np.empty_like(close)
    out[:] = np.nan

    if timeperiod <= 0 or timeperiod > n:
        return out

    w = np.hamming(timeperiod).astype(close.dtype)
    hma_valid = np.convolve(close, w, mode="valid") / w.sum()
    out[timeperiod - 1:] = hma_valid
    return out


@jit(nopython=True, nogil=True, cache=True)
def NadarayWatsonMA(close, timeperiod, kernel=0):
    close = np.ascontiguousarray(close)
    n = close.shape[0]

    out = np.empty_like(close)
    out[:] = np.nan

    if timeperiod <= 0 or timeperiod > n:
        return out

    if timeperiod == 1:
        out[0] = close[0]
        return out

    # oldest -> newest, więc newest ma lag 0 i największą wagę
    lags = np.arange(timeperiod - 1, -1, -1, dtype=np.float64)
    u = (
            lags / timeperiod
    )  # zostawiam obecną skalę, żeby nie rozjechać charakteru wygładzania

    if kernel == 0:
        weights = np.exp(-0.5 * u ** 2) / np.sqrt(2.0 * np.pi)
    elif kernel == 1:
        weights = np.where(u <= 1.0, 0.75 * (1.0 - u ** 2), 0.0)
    elif kernel == 2:
        weights = np.where(u <= 1.0, 0.5, 0.0)
    elif kernel == 3:
        weights = np.where(u <= 1.0, 1.0 - u, 0.0)
    elif kernel == 4:
        weights = np.where(u <= 1.0, (15.0 / 16.0) * (1.0 - u ** 2) ** 2, 0.0)
    elif kernel == 5:
        weights = np.where(u <= 1.0, (np.pi / 4.0) * np.cos((np.pi / 2.0) * u), 0.0)
    else:
        return out

    weights = np.ascontiguousarray(weights)
    weights_sum = weights.sum()
    if weights_sum == 0.0:
        return out

    for i in range(timeperiod - 1, n):
        out[i] = (weights @ close[i - timeperiod + 1: i + 1]) / weights_sum

    return out


@jit(nopython=True, nogil=True, cache=True)
def LWMA(close, period):
    close = np.ascontiguousarray(close)
    n = close.shape[0]

    out = np.empty_like(close)
    out[:] = np.nan

    if period <= 0 or period > n:
        return out

    weights = np.ascontiguousarray(np.arange(1, period + 1, dtype=close.dtype))
    weights_sum = weights.sum()

    for i in range(period - 1, n):
        out[i] = np.dot(close[i - period + 1: i + 1], weights) / weights_sum

    return out


@jit(nopython=True, nogil=True, cache=True)
def MGD(close, period):
    close = np.ascontiguousarray(close)
    n = close.shape[0]

    md = np.empty_like(close)
    md[0] = close[0]
    eps = 1e-12

    for i in range(1, n):
        prev = md[i - 1]
        denom = prev if prev != 0.0 else 1.0

        ratio = close[i] / denom
        scale = period * np.power(ratio, 4)

        if not np.isfinite(scale) or abs(scale) < eps:
            md[i] = prev
        else:
            md[i] = prev + (close[i] - prev) / scale

    # warmup jako NaN (żeby nie udawał danych)
    if period > 1 and period - 1 < n:
        for i in range(period - 1):
            md[i] = np.nan

    return md


### It behaves differently depending on close len
# @feature_timeit
@jit(nopython=True, nogil=True, cache=True)
def VIDYA(close, k, period):
    alpha = 2 / (period + 1)
    # k = talib.CMO(close, period)
    k = np.abs(k) / 100
    VIDYA = np.empty_like(close)
    VIDYA[period - 1] = close[period - 1]
    for i in range(period, len(close)):
        VIDYA[i] = alpha * k[i] * close[i] + (1 - alpha * k[i]) * VIDYA[i - 1]
    return VIDYA


# @feature_timeit
def FBA(close, period):
    fibs = []
    a, b = 1, 2
    while b <= period:
        fibs.append(b)
        a, b = b, a + b
    moving_averages = np.array([talib.EMA(close, i) for i in fibs]) / 100
    return (np.sum(moving_averages, axis=0) / len(fibs)) * 100


# @jit(nopython=True, nogil=True, cache=True)
def CWMA(close, weights, period):
    cwma = np.zeros_like(close)
    window_weight_sum = np.sum(weights)
    window_prod_sum = np.sum(close[:period] * weights)
    cwma[period - 1] = window_prod_sum / window_weight_sum
    for i in range(period, len(close)):
        # print(f'window_weight_sum {window_weight_sum}')
        # print(f'window_prod_sum {window_prod_sum}')
        window_prod_sum = np.sum(close[i - period: i] * weights)
        cwma[i] = window_prod_sum / window_weight_sum
        # print(f'cwma {cwma[i]}')
    return cwma


# @jit(nopython=True, nogil=True, cache=True)
def FWMA(close, period):
    print(f"fibs {fib_to(period + 1, normalization=True)[1:]}")
    return CWMA(close, fib_to(period + 1, normalization=True)[1:], period)
