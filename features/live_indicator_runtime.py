import numpy as np
from talib import AD, EMA, STOCHF, TRANGE

from .ta_tools import apply_ma, get_1d_ma, precompute_ohlcv_sources


class _SharpModifiedMovingAverageState:
    __slots__ = (
        "period",
        "count",
        "total",
        "weighted_total",
        "window_values",
        "value",
    )

    def __init__(
            self,
            period,
            *,
            count=0,
            total=0.0,
            weighted_total=0.0,
            window_values=(),
            value=float("nan"),
    ):
        self.period = int(period)
        self.count = int(count)
        self.total = float(total)
        self.weighted_total = float(weighted_total)
        self.window_values = list(window_values)
        self.value = float(value)

    def update(self, value):
        value = float(value)
        period = self.period
        index = self.count
        if index < period - 1:
            self.total += value
            self.weighted_total += (-period + 2 * index + 1.0) / 2.0 * value
            self.window_values.append(value)
            self.count += 1
            return self.value

        self.total += value
        self.weighted_total += (period - 1.0) / 2.0 * value
        self.value = self.total / period + (6.0 * self.weighted_total) / (
            (period + 1) * period
        )
        self.weighted_total -= self.total
        oldest = self.window_values.pop(0)
        self.weighted_total += (period + 1.0) / 2.0 * oldest
        self.total -= oldest
        self.window_values.append(value)
        self.count += 1
        return self.value

    def to_dict(self):
        return {
            "period": self.period,
            "count": self.count,
            "total": self.total,
            "weighted_total": self.weighted_total,
            "window_values": list(self.window_values),
            "value": self.value,
        }


class ChaikinOscillatorRuntimeState:
    """Continue the fitted Chaikin EMA minus SHMMA from its training history."""

    __slots__ = (
        "fast_period",
        "slow_period",
        "adl_value",
        "fast_ema_value",
        "fast_ema_alpha",
        "shmma_state",
        "value",
    )

    def __init__(self, fast_period, slow_period, adl_value, fast_ema_value, shmma_state):
        self.fast_period = int(fast_period)
        self.slow_period = int(slow_period)
        self.adl_value = float(adl_value)
        self.fast_ema_value = float(fast_ema_value)
        self.fast_ema_alpha = 2.0 / (self.fast_period + 1.0)
        self.shmma_state = shmma_state
        self.value = self.fast_ema_value - self.shmma_state.value

    @classmethod
    def from_history(cls, ohlcv, *, fast_period, slow_period):
        values = np.asarray(ohlcv, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 5:
            raise ValueError("Chaikin history must be an OHLCV matrix.")
        adl = AD(*values[:, 1:5].T)
        fast_ema = EMA(adl, timeperiod=int(fast_period))
        shmma_state = _SharpModifiedMovingAverageState(slow_period)
        for value in adl:
            shmma_state.update(float(value))
        if not np.isfinite(fast_ema[-1]) or not np.isfinite(shmma_state.value):
            raise ValueError("Chaikin history is too short to initialize its fitted state.")
        return cls(
            fast_period,
            slow_period,
            adl[-1],
            fast_ema[-1],
            shmma_state,
        )

    @classmethod
    def from_dict(cls, payload):
        shmma = payload["shmma_state"]
        return cls(
            payload["fast_period"],
            payload["slow_period"],
            payload["adl_value"],
            payload["fast_ema_value"],
            _SharpModifiedMovingAverageState(
                shmma["period"],
                count=shmma["count"],
                total=shmma["total"],
                weighted_total=shmma["weighted_total"],
                window_values=shmma["window_values"],
                value=shmma["value"],
            ),
        )

    def to_dict(self):
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "adl_value": self.adl_value,
            "fast_ema_value": self.fast_ema_value,
            "shmma_state": self.shmma_state.to_dict(),
        }

    def update(self, ohlcv_row):
        row = np.asarray(ohlcv_row, dtype=np.float64).reshape(-1)
        increment = AD(
            np.asarray([row[1]], dtype=np.float64),
            np.asarray([row[2]], dtype=np.float64),
            np.asarray([row[3]], dtype=np.float64),
            np.asarray([row[4]], dtype=np.float64),
        )[0]
        self.adl_value += float(increment)
        self.fast_ema_value = (
            (self.adl_value - self.fast_ema_value) * self.fast_ema_alpha
        ) + self.fast_ema_value
        slow_value = self.shmma_state.update(self.adl_value)
        self.value = self.fast_ema_value - slow_value
        return self.value


class IndicatorFullHistoryScratch:
    __slots__ = ("ohlcv", "sources", "volume")

    def __init__(self, ohlcv):
        self.ohlcv = ohlcv
        self.sources = precompute_ohlcv_sources(ohlcv)
        self.volume = ohlcv[:, 4]


class IndicatorWindowScratch:
    __slots__ = (
        "full_scratch",
        "window_len",
        "ohlcv",
        "volume",
        "_source_cache",
        "_trange",
        "_adl",
        "_fastk_by_period",
        "_up_move",
        "_down_move",
    )

    def __init__(self, full_scratch, window_len):
        self.full_scratch = full_scratch
        self.window_len = min(int(window_len), int(full_scratch.ohlcv.shape[0]))
        if self.window_len <= 0:
            self.window_len = int(full_scratch.ohlcv.shape[0])
        self.ohlcv = full_scratch.ohlcv[-self.window_len:, :]
        self.volume = full_scratch.volume[-self.window_len:]
        self._source_cache = {}
        self._trange = None
        self._adl = None
        self._fastk_by_period = {}
        self._up_move = None
        self._down_move = None

    def get_source(self, source):
        cached = self._source_cache.get(source)
        if cached is not None:
            return cached

        base = self.full_scratch.sources[source]
        if self.window_len >= int(base.shape[0]):
            cached = base
        else:
            cached = base[-self.window_len:]
        self._source_cache[source] = cached
        return cached

    def get_trange(self):
        if self._trange is None:
            self._trange = TRANGE(*self.ohlcv[:, 1:4].T)
        return self._trange

    def get_adl(self):
        if self._adl is None:
            self._adl = AD(*self.ohlcv[:, 1:5].T)
        return self._adl

    def get_fastk(self, fastk_period):
        period = int(fastk_period)
        cached = self._fastk_by_period.get(period)
        if cached is not None:
            return cached
        cached, _ = STOCHF(
            *self.ohlcv[:, 1:4].T,
            fastk_period=period,
            fastd_period=1,
            fastd_matype=0,
        )
        self._fastk_by_period[period] = cached
        return cached

    def get_up_down_moves(self):
        if self._up_move is not None and self._down_move is not None:
            return self._up_move, self._down_move

        prev_high = np.roll(self.ohlcv[:, 1], 1)
        prev_high[0] = self.ohlcv[0, 1]
        prev_low = np.roll(self.ohlcv[:, 2], 1)
        prev_low[0] = self.ohlcv[0, 2]

        high_diff = self.ohlcv[:, 1] - prev_high
        low_diff = prev_low - self.ohlcv[:, 2]
        self._up_move = np.where(
            (high_diff > low_diff) & (high_diff > 0.0), high_diff, 0.0
        )
        self._down_move = np.where(
            (low_diff > high_diff) & (low_diff > 0.0), low_diff, 0.0
        )
        return self._up_move, self._down_move


def _last_float(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[-1])


def _channel_ratio_last(numerator, denominator):
    if not np.isfinite(denominator):
        return float("nan")
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("nan")
    return float(numerator / denominator)


def _ma_from_source(scratch, source, ma_type, period):
    return apply_ma(scratch.get_source(source), ma_type, period, scratch.volume)


def get_adx_latest_value_live(params, scratch):
    tr_smooth = get_1d_ma(
        scratch.get_trange(), params["ma_type_atr"], params["atr_period"]
    )
    up_move, down_move = scratch.get_up_down_moves()
    pos_dm_smooth = get_1d_ma(up_move, params["ma_type_posDM"], params["posDM_period"])
    neg_dm_smooth = get_1d_ma(
        down_move, params["ma_type_negDM"], params["negDM_period"]
    )

    pos_dm_smooth = np.maximum(pos_dm_smooth, 0.0)
    neg_dm_smooth = np.maximum(neg_dm_smooth, 0.0)
    tiny = 1e-12

    plus = np.divide(
        pos_dm_smooth,
        tr_smooth,
        out=np.zeros_like(tr_smooth),
        where=np.abs(tr_smooth) > tiny,
    )
    minus = np.divide(
        neg_dm_smooth,
        tr_smooth,
        out=np.zeros_like(tr_smooth),
        where=np.abs(tr_smooth) > tiny,
    )
    plus_di = 100.0 * plus
    minus_di = 100.0 * minus

    denom = np.abs(plus_di) + np.abs(minus_di)
    dx = np.divide(
        np.abs(plus_di - minus_di),
        denom,
        out=np.zeros_like(denom),
        where=denom > tiny,
    )
    dx_raw = 100.0 * np.clip(dx, 0.0, 1.0)
    adx = get_1d_ma(dx_raw, params["ma_type_adx"], params["adx_period"])
    return _last_float(adx)


def get_bollinger_bands_latest_value_live(params, scratch):
    center_ma = _ma_from_source(
        scratch, params["ma_source"], params["ma_type"], params["ma_period"]
    )
    src = scratch.get_source(params["std_source"])
    diff2 = (src - center_ma) ** 2
    var_smooth = get_1d_ma(diff2, params["std_ma_type"], params["std_ma_period"])
    var_last = _last_float(var_smooth)
    if not np.isfinite(var_last) or var_last < 0.0:
        return float("nan")
    std_last = float(np.sqrt(var_last))
    numerator = _last_float(center_ma) - _last_float(scratch.get_source("close"))
    return _channel_ratio_last(numerator, std_last)


def get_chaikin_oscillator_latest_value_live(params, scratch):
    adl = scratch.get_adl()
    fast = get_1d_ma(adl, params["fast_ma_type"], params["fast_period"])
    slow = get_1d_ma(adl, params["slow_ma_type"], params["slow_period"])
    return _last_float(fast) - _last_float(slow)


def get_keltner_channel_latest_value_live(params, scratch):
    center_ma = _ma_from_source(
        scratch, params["source"], params["ma_type"], params["ma_period"]
    )
    atr = get_1d_ma(scratch.get_trange(), params["atr_ma_type"], params["atr_period"])
    numerator = _last_float(center_ma) - _last_float(scratch.get_source("close"))
    return _channel_ratio_last(numerator, _last_float(atr))


def get_macd_latest_value_live(params, scratch):
    fast = _ma_from_source(
        scratch,
        params["fast_source"],
        params["fast_ma_type"],
        params["fast_period"],
    )
    slow = _ma_from_source(
        scratch,
        params["slow_source"],
        params["slow_ma_type"],
        params["slow_period"],
    )
    macd = fast - slow
    signal = get_1d_ma(macd, params["signal_ma_type"], params["signal_period"])
    return _last_float(macd) - _last_float(signal)


def get_stochastic_oscillator_latest_value_live(params, scratch):
    fastk = scratch.get_fastk(params["fastK_period"])
    if int(params["slowK_period"]) == 1:
        slowk = fastk
    else:
        slowk = get_1d_ma(fastk, params["slowK_ma_type"], params["slowK_period"])
    slowd = get_1d_ma(slowk, params["slowD_ma_type"], params["slowD_period"])
    return _last_float(slowk) - _last_float(slowd)


LATEST_VALUE_BUILDERS = {
    "ADX": get_adx_latest_value_live,
    "BollingerBands": get_bollinger_bands_latest_value_live,
    "ChaikinOsc": get_chaikin_oscillator_latest_value_live,
    "KeltnerChannel": get_keltner_channel_latest_value_live,
    "MACD": get_macd_latest_value_live,
    "StochOsc": get_stochastic_oscillator_latest_value_live,
}
