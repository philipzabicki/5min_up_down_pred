import numpy as np
import pandas as pd
from numba import jit

REALIZED_VOL_WINDOWS_MINUTES = (1, 5, 15, 60, 240)
REALIZED_VOL_EPS = 1e-12

_WINDOW_LABELS = ("1m", "5m", "15m", "1h", "4h")
_WINDOWS_ARRAY = np.asarray(REALIZED_VOL_WINDOWS_MINUTES, dtype=np.int64)
_MAX_WINDOW = int(REALIZED_VOL_WINDOWS_MINUTES[-1])
_CE_WINDOW_PAIRS = ((1, 5), (5, 15), (15, 60), (60, 240))


def _rv_feature_col(label):
    return f"realized_volatility_{label}"


def _rv_up_feature_col(label):
    return f"realized_volatility_up_{label}"


def _rv_down_feature_col(label):
    return f"realized_volatility_down_{label}"


def _rv_ce_feature_col(short_label, long_label):
    return f"realized_volatility_compression_expansion_{short_label}_{long_label}"


RV_FEATURE_COLUMNS = tuple(_rv_feature_col(label) for label in _WINDOW_LABELS)
RV_UP_FEATURE_COLUMNS = tuple(_rv_up_feature_col(label) for label in _WINDOW_LABELS)
RV_DOWN_FEATURE_COLUMNS = tuple(
    _rv_down_feature_col(label) for label in _WINDOW_LABELS
)
VOV_FEATURE_COLUMNS = tuple(f"vov_{label}" for label in _WINDOW_LABELS)
RV_CE_FEATURE_COLUMNS = tuple(
    _rv_ce_feature_col(
        _WINDOW_LABELS[REALIZED_VOL_WINDOWS_MINUTES.index(short_w)],
        _WINDOW_LABELS[REALIZED_VOL_WINDOWS_MINUTES.index(long_w)],
    )
    for short_w, long_w in _CE_WINDOW_PAIRS
)
REALIZED_VOLATILITY_FEATURE_COLUMNS = (
    RV_FEATURE_COLUMNS
    + RV_UP_FEATURE_COLUMNS
    + RV_DOWN_FEATURE_COLUMNS
    + VOV_FEATURE_COLUMNS
    + RV_CE_FEATURE_COLUMNS
)
_ALL_FEATURE_COLUMNS_SET = set(REALIZED_VOLATILITY_FEATURE_COLUMNS)


def _dedupe_ordered(values):
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return tuple(out)


def is_realized_volatility_feature(feature_name):
    return str(feature_name).strip() in _ALL_FEATURE_COLUMNS_SET


def resolve_realized_volatility_feature_cols(feature_cols=None):
    if feature_cols is None:
        return REALIZED_VOLATILITY_FEATURE_COLUMNS

    requested = [str(col).strip() for col in feature_cols]
    if not requested:
        raise ValueError("feature_cols cannot be empty.")

    requested = _dedupe_ordered(requested)
    unsupported = [col for col in requested if col not in _ALL_FEATURE_COLUMNS_SET]
    if unsupported:
        supported = ", ".join(REALIZED_VOLATILITY_FEATURE_COLUMNS)
        raise ValueError(
            "Unsupported realized volatility feature columns: "
            + ", ".join(unsupported)
            + f". Supported: {supported}"
        )
    return requested


def _empty_feature_arrays(length):
    return {
        feature_col: np.full(int(length), np.nan, dtype=np.float64)
        for feature_col in REALIZED_VOLATILITY_FEATURE_COLUMNS
    }


def _empty_feature_dict():
    return {feature_col: float("nan") for feature_col in REALIZED_VOLATILITY_FEATURE_COLUMNS}


def _validate_close_array(close):
    close_arr = np.asarray(close, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(close_arr)):
        raise ValueError("close must contain only finite values.")
    if np.any(close_arr <= 0.0):
        raise ValueError("close must contain only positive values.")
    return np.ascontiguousarray(close_arr)


@jit(nopython=True, nogil=True, cache=True)
def _window_moments_from_buffer(return_buffer, returns_seen, window):
    start = returns_seen - window
    sum_r2 = 0.0
    sum_up2 = 0.0
    sum_down2 = 0.0
    sum_r4 = 0.0

    for offset in range(window):
        pos = (start + offset) % _MAX_WINDOW
        ret = return_buffer[pos]
        ret2 = ret * ret
        sum_r2 += ret2
        if ret > 0.0:
            sum_up2 += ret2
        elif ret < 0.0:
            sum_down2 += ret2
        sum_r4 += ret2 * ret2

    return sum_r2, sum_up2, sum_down2, sum_r4


@jit(nopython=True, nogil=True, cache=True)
def _compute_realized_volatility_core(close, windows, eps):
    rows = close.shape[0]
    window_count = windows.shape[0]
    rv = np.full((rows, window_count), np.nan, dtype=np.float64)
    rv_up = np.full((rows, window_count), np.nan, dtype=np.float64)
    rv_down = np.full((rows, window_count), np.nan, dtype=np.float64)
    vov = np.full((rows, window_count), np.nan, dtype=np.float64)

    return_buffer = np.zeros(_MAX_WINDOW, dtype=np.float64)
    returns_seen = 0

    for row_idx in range(1, rows):
        prev_close = close[row_idx - 1]
        curr_close = close[row_idx]
        ret = np.log(curr_close / prev_close)

        buffer_pos = returns_seen % _MAX_WINDOW
        next_returns_seen = returns_seen + 1
        return_buffer[buffer_pos] = ret

        for window_idx in range(window_count):
            window = int(windows[window_idx])
            if next_returns_seen >= window:
                sum_r2, sum_up2, sum_down2, sum_r4 = _window_moments_from_buffer(
                    return_buffer,
                    next_returns_seen,
                    window,
                )
                inv_window = 1.0 / float(window)
                mean_r2 = sum_r2 * inv_window
                if mean_r2 < 0.0:
                    mean_r2 = 0.0
                mean_up2 = sum_up2 * inv_window
                if mean_up2 < 0.0:
                    mean_up2 = 0.0
                mean_down2 = sum_down2 * inv_window
                if mean_down2 < 0.0:
                    mean_down2 = 0.0
                mean_r4 = sum_r4 * inv_window
                var_r2 = mean_r4 - (mean_r2 * mean_r2)
                if var_r2 < 0.0:
                    var_r2 = 0.0

                rv[row_idx, window_idx] = np.sqrt(mean_r2)
                rv_up[row_idx, window_idx] = np.sqrt(mean_up2)
                rv_down[row_idx, window_idx] = np.sqrt(mean_down2)
                vov[row_idx, window_idx] = np.sqrt(var_r2) / (mean_r2 + eps)

        returns_seen = next_returns_seen

    return rv, rv_up, rv_down, vov


def compute_realized_volatility_feature_arrays(close):
    close_arr = _validate_close_array(close)
    rows = int(close_arr.shape[0])
    if rows == 0:
        return _empty_feature_arrays(0)

    rv, rv_up, rv_down, vov = _compute_realized_volatility_core(
        close_arr,
        _WINDOWS_ARRAY,
        float(REALIZED_VOL_EPS),
    )

    feature_arrays = {}
    for window_idx, label in enumerate(_WINDOW_LABELS):
        feature_arrays[_rv_feature_col(label)] = rv[:, window_idx]
        feature_arrays[_rv_up_feature_col(label)] = rv_up[:, window_idx]
        feature_arrays[_rv_down_feature_col(label)] = rv_down[:, window_idx]
        feature_arrays[f"vov_{label}"] = vov[:, window_idx]

    for short_window, long_window in _CE_WINDOW_PAIRS:
        short_label = _WINDOW_LABELS[REALIZED_VOL_WINDOWS_MINUTES.index(short_window)]
        long_label = _WINDOW_LABELS[REALIZED_VOL_WINDOWS_MINUTES.index(long_window)]
        short_values = feature_arrays[_rv_feature_col(short_label)]
        long_values = feature_arrays[_rv_feature_col(long_label)]
        ce_values = np.full(rows, np.nan, dtype=np.float64)
        valid_mask = np.isfinite(short_values) & np.isfinite(long_values)
        ce_values[valid_mask] = np.log(
            (short_values[valid_mask] + REALIZED_VOL_EPS)
            / (long_values[valid_mask] + REALIZED_VOL_EPS)
        )
        feature_arrays[_rv_ce_feature_col(short_label, long_label)] = ce_values

    return feature_arrays


def add_realized_volatility_features(
    df: pd.DataFrame,
    close_col: str = "Close",
    float_dtype=np.float64,
) -> pd.DataFrame:
    if close_col not in df.columns:
        raise ValueError(f"Missing required close column: {close_col}")

    feature_arrays = compute_realized_volatility_feature_arrays(
        df[close_col].to_numpy(dtype=np.float64, copy=False)
    )
    feature_frame = pd.DataFrame(
        {
            feature_col: np.asarray(values, dtype=float_dtype)
            for feature_col, values in feature_arrays.items()
        },
        index=df.index,
    )
    duplicate_cols = [col for col in feature_frame.columns if col in df.columns]
    if duplicate_cols:
        base_df = df.drop(columns=duplicate_cols)
    else:
        base_df = df
    return pd.concat([base_df, feature_frame], axis=1, copy=False)


class RealizedVolatilityRuntimeState:
    __slots__ = (
        "prev_close",
        "returns_seen",
        "return_buffer",
        "latest_values",
    )

    def __init__(self):
        self.prev_close = None
        self.returns_seen = 0
        self.return_buffer = np.zeros(_MAX_WINDOW, dtype=np.float64)
        self.latest_values = _empty_feature_dict()

    @classmethod
    def from_close_history(cls, close_values):
        state = cls()
        close_arr = _validate_close_array(close_values)
        for close_value in close_arr:
            state.update(float(close_value))
        return state

    def update(self, close: float) -> dict[str, float]:
        close_value = float(close)
        if not np.isfinite(close_value):
            raise ValueError("close must be finite.")
        if close_value <= 0.0:
            raise ValueError("close must be positive.")

        if self.prev_close is None:
            self.prev_close = close_value
            self.latest_values = _empty_feature_dict()
            return dict(self.latest_values)

        ret = float(np.log(close_value / self.prev_close))

        buffer_pos = self.returns_seen % _MAX_WINDOW
        next_returns_seen = self.returns_seen + 1
        self.return_buffer[buffer_pos] = ret
        rv_values = np.full(len(REALIZED_VOL_WINDOWS_MINUTES), np.nan, dtype=np.float64)
        rv_up_values = np.full(
            len(REALIZED_VOL_WINDOWS_MINUTES), np.nan, dtype=np.float64
        )
        rv_down_values = np.full(
            len(REALIZED_VOL_WINDOWS_MINUTES), np.nan, dtype=np.float64
        )
        vov_values = np.full(len(REALIZED_VOL_WINDOWS_MINUTES), np.nan, dtype=np.float64)

        for window_idx, window in enumerate(REALIZED_VOL_WINDOWS_MINUTES):
            if next_returns_seen >= window:
                sum_r2, sum_up2, sum_down2, sum_r4 = _window_moments_from_buffer(
                    self.return_buffer,
                    next_returns_seen,
                    int(window),
                )
                inv_window = 1.0 / float(window)
                mean_r2 = sum_r2 * inv_window
                if mean_r2 < 0.0:
                    mean_r2 = 0.0
                mean_up2 = sum_up2 * inv_window
                if mean_up2 < 0.0:
                    mean_up2 = 0.0
                mean_down2 = sum_down2 * inv_window
                if mean_down2 < 0.0:
                    mean_down2 = 0.0
                mean_r4 = sum_r4 * inv_window
                var_r2 = mean_r4 - (mean_r2 * mean_r2)
                if var_r2 < 0.0:
                    var_r2 = 0.0
                rv_values[window_idx] = np.sqrt(mean_r2)
                rv_up_values[window_idx] = np.sqrt(mean_up2)
                rv_down_values[window_idx] = np.sqrt(mean_down2)
                vov_values[window_idx] = np.sqrt(var_r2) / (mean_r2 + REALIZED_VOL_EPS)

        self.returns_seen = next_returns_seen
        self.prev_close = close_value

        values = {}
        for window_idx, label in enumerate(_WINDOW_LABELS):
            values[_rv_feature_col(label)] = float(rv_values[window_idx])
            values[_rv_up_feature_col(label)] = float(rv_up_values[window_idx])
            values[_rv_down_feature_col(label)] = float(rv_down_values[window_idx])
            values[f"vov_{label}"] = float(vov_values[window_idx])

        for short_window, long_window in _CE_WINDOW_PAIRS:
            short_label = _WINDOW_LABELS[REALIZED_VOL_WINDOWS_MINUTES.index(short_window)]
            long_label = _WINDOW_LABELS[REALIZED_VOL_WINDOWS_MINUTES.index(long_window)]
            short_value = values[_rv_feature_col(short_label)]
            long_value = values[_rv_feature_col(long_label)]
            if np.isfinite(short_value) and np.isfinite(long_value):
                values[_rv_ce_feature_col(short_label, long_label)] = float(
                    np.log(
                        (short_value + REALIZED_VOL_EPS)
                        / (long_value + REALIZED_VOL_EPS)
                    )
                )
            else:
                values[_rv_ce_feature_col(short_label, long_label)] = float("nan")

        self.latest_values = values
        return dict(values)
