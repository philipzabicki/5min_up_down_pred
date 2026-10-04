"""Versioned full-history indicator semantics for entry-state collection."""
from __future__ import annotations

import numpy as np

from .ADX import get_adx_values


def get_adx_latest_value_full_history(params, ohlcv):
    """Return batch ADX semantics over every candle available at this time."""
    values = get_adx_values(params, np.asarray(ohlcv, dtype=np.float64))
    return float(values[-1]) if values.size else float("nan")
