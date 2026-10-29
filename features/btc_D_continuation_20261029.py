"""State helpers for the frozen E0041 development generator continuation."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


TAIL_ROWS = 12_001
MINUTE_NS = 60_000_000_000
ADX_TEMA_PERIOD = 640


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encode_value(value):
    if isinstance(value, pd.DataFrame):
        encoded_columns = {}
        dtypes = {}
        for column in value.columns:
            series = value[column]
            dtypes[str(column)] = str(series.dtype)
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                encoded_columns[str(column)] = {
                    "__datetime_series__": True,
                    "dtype": str(series.dtype),
                    "values": series.dt.as_unit("ns").astype("int64").to_numpy(dtype=np.int64).tolist(),
                }
            else:
                encoded_columns[str(column)] = _encode_value(series.to_numpy())
        return {
            "__dataframe__": True,
            "columns": list(value.columns),
            "dtypes": dtypes,
            "values": encoded_columns,
        }
    if isinstance(value, (pd.Timestamp,)):
        return {"__timestamp__": value.isoformat()}
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "M":
            return {
                "__ndarray__": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "datetime_ns": value.astype("datetime64[ns]").astype(np.int64).tolist(),
            }
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": _encode_value(value.tolist()),
        }
    if isinstance(value, dict):
        return {str(key): _encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        item = float(value)
        if np.isnan(item):
            return {"__float__": "nan"}
        if np.isposinf(item):
            return {"__float__": "+inf"}
        if np.isneginf(item):
            return {"__float__": "-inf"}
        return item
    if isinstance(value, float) and not np.isfinite(value):
        return {"__float__": "nan" if np.isnan(value) else "+inf" if value > 0 else "-inf"}
    return value


def _decode_value(value):
    if isinstance(value, dict) and value.get("__dataframe__") is True:
        decoded = {}
        for key, item in value["values"].items():
            if isinstance(item, dict) and item.get("__datetime_series__") is True:
                timestamps = pd.to_datetime(item["values"], unit="ns", utc="UTC" in item["dtype"])
                decoded[key] = pd.Series(timestamps).astype(item["dtype"])
            else:
                decoded[key] = _decode_value(item)
        return pd.DataFrame(decoded, columns=value["columns"])
    if isinstance(value, dict) and "__timestamp__" in value:
        return pd.Timestamp(value["__timestamp__"])
    if isinstance(value, dict) and value.get("__ndarray__") is True:
        if "datetime_ns" in value:
            return np.asarray(value["datetime_ns"], dtype=np.int64).astype("datetime64[ns]").astype(np.dtype(value["dtype"])).reshape(value["shape"])
        return np.asarray(_decode_value(value["values"]), dtype=np.dtype(value["dtype"])).reshape(value["shape"])
    if isinstance(value, dict) and "__float__" in value:
        return {"nan": np.nan, "+inf": np.inf, "-inf": -np.inf}[value["__float__"]]
    if isinstance(value, dict):
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def atomic_checkpoint(path: Path, payload: dict) -> str:
    """Persist a self-authenticating JSON checkpoint with an atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _encode_value(dict(payload))
    body.pop("integrity_sha256", None)
    body["integrity_sha256"] = canonical_sha256(body)
    data = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)
    return file_sha256(path)


def load_checkpoint(path: Path, *, identity_sha256: str) -> dict:
    path = Path(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    observed_integrity = body.pop("integrity_sha256", None)
    if observed_integrity != canonical_sha256(body):
        raise RuntimeError("continuation checkpoint integrity hash mismatch")
    if body.get("identity_sha256") != identity_sha256:
        raise RuntimeError("continuation checkpoint identity mismatch")
    return _decode_value(body)


def state_digest(state: dict) -> str:
    return canonical_sha256(_encode_value(state))


def initialize_streak_states(prefix: pd.DataFrame, intervals: tuple[str, ...]) -> dict:
    """Seed unbounded candle streak counters and in-flight bucket aggregates."""
    from features.candle_features import add_candle_streak_features, resolve_streak_interval_to_rule

    if prefix.empty:
        raise ValueError("a streak checkpoint requires a non-empty raw prefix")
    rules = resolve_streak_interval_to_rule(intervals)
    measured = add_candle_streak_features(prefix, interval_to_rule=rules)
    opened = pd.DatetimeIndex(pd.to_datetime(prefix["Opened"], utc=True, errors="raise")).as_unit("ns")
    last_ns = int(opened.asi8[-1])
    result = {"last_opened_ns": last_ns, "intervals": {}}
    for label, rule in rules.items():
        bucket_minutes = int(pd.Timedelta(rule).value // MINUTE_NS)
        bucket_ns = bucket_minutes * MINUTE_NS
        bucket_start = (last_ns // bucket_ns) * bucket_ns
        in_bucket = np.flatnonzero(opened.asi8 >= bucket_start)
        current = prefix.iloc[in_bucket]
        expected_open = float(pd.to_numeric(current["Open"], errors="coerce").iloc[0])
        state = {
            "minutes": bucket_minutes,
            "bucket_start_ns": bucket_start,
            "bucket_count": int(len(current)),
            "bucket_open": float(current["Open"].iloc[0]),
            "bucket_close": float(current["Close"].iloc[-1]),
            "bucket_last_opened_ns": int(opened.asi8[-1]),
            "last_value": int(measured[f"candle_streak_{label}"].iloc[-1]),
            "emitted": bool(
                len(current) == bucket_minutes
                and last_ns == bucket_start + (bucket_minutes - 1) * MINUTE_NS
            ),
        }
        if not np.isfinite(expected_open):
            raise ValueError("streak prefix contains a non-finite in-flight candle open")
        result["intervals"][label] = state
    return result


def _finish_streak_bucket(state: dict) -> None:
    if state["emitted"]:
        return
    complete = (
        int(state["bucket_count"]) == int(state["minutes"])
        and int(state["bucket_last_opened_ns"])
        == int(state["bucket_start_ns"]) + (int(state["minutes"]) - 1) * MINUTE_NS
    )
    if not complete:
        return
    diff = float(state["bucket_close"]) - float(state["bucket_open"])
    sign = int(np.sign(diff))
    previous = int(state["last_value"])
    previous_sign = int(np.sign(previous))
    state["last_value"] = previous + sign if sign and previous_sign == sign else sign
    state["emitted"] = True


def advance_streak_states(state: dict, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Advance streak state in timestamp order and emit the causal row values."""
    if rows.empty:
        return {f"candle_streak_{key}": np.empty(0, dtype=np.int32) for key in state["intervals"]}
    opened = pd.DatetimeIndex(pd.to_datetime(rows["Opened"], utc=True, errors="raise")).as_unit("ns")
    if opened.has_duplicates or not opened.is_monotonic_increasing:
        raise ValueError("streak continuation timestamps must be unique and ordered")
    if state.get("last_opened_ns") is not None and int(opened.asi8[0]) <= int(state["last_opened_ns"]):
        raise ValueError("streak continuation overlaps or reorders its checkpoint")
    outputs = {
        f"candle_streak_{key}": np.empty(len(rows), dtype=np.int32)
        for key in state["intervals"]
    }
    opens = rows["Open"].to_numpy(dtype=np.float64, copy=False)
    closes = rows["Close"].to_numpy(dtype=np.float64, copy=False)
    for row_index, timestamp_ns in enumerate(opened.asi8):
        for label, item in state["intervals"].items():
            bucket_ns = int(item["minutes"]) * MINUTE_NS
            bucket_start = (int(timestamp_ns) // bucket_ns) * bucket_ns
            if bucket_start != int(item["bucket_start_ns"]):
                _finish_streak_bucket(item)
                item.update({
                    "bucket_start_ns": bucket_start,
                    "bucket_count": 0,
                    "bucket_open": float("nan"),
                    "bucket_close": float("nan"),
                    "bucket_last_opened_ns": None,
                    "emitted": False,
                })
            if int(item["bucket_count"]) == 0:
                item["bucket_open"] = float(opens[row_index])
            item["bucket_count"] = int(item["bucket_count"]) + 1
            item["bucket_close"] = float(closes[row_index])
            item["bucket_last_opened_ns"] = int(timestamp_ns)
            if (
                int(item["bucket_count"]) == int(item["minutes"])
                and int(timestamp_ns) == bucket_start + (int(item["minutes"]) - 1) * MINUTE_NS
            ):
                _finish_streak_bucket(item)
            outputs[f"candle_streak_{label}"][row_index] = int(item["last_value"])
        state["last_opened_ns"] = int(timestamp_ns)
    return outputs


def seed_adx_tema_state(prefix: pd.DataFrame, params: dict) -> dict:
    """Recover the three saved TA-Lib EMA stages from the causal raw prefix."""
    import talib
    from features.ADX import custom_adx

    ohlcv = prefix[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=np.float64, copy=False)
    _, plus_di, minus_di = custom_adx(
        ohlcv,
        atr_period=int(params["atr_period"]),
        posDM_period=int(params["posDM_period"]),
        negDM_period=int(params["negDM_period"]),
        adx_period=int(params["adx_period"]),
        ma_type_atr=str(params["ma_type_atr"]),
        ma_type_posDM=str(params["ma_type_posDM"]),
        ma_type_negDM=str(params["ma_type_negDM"]),
        ma_type_adx=str(params["ma_type_adx"]),
    )
    denominator = np.abs(plus_di) + np.abs(minus_di)
    dx = np.divide(
        100.0 * np.abs(plus_di - minus_di), denominator,
        out=np.zeros_like(denominator), where=denominator > 1e-12,
    )
    dx = np.clip(dx, 0.0, 100.0)
    period = int(params["adx_period"])
    if str(params["ma_type_adx"]) != "TEMA":
        raise RuntimeError("frozen E0041 ADX continuation expects its saved TEMA implementation")
    ema1 = talib.EMA(dx, timeperiod=period)
    ema2 = talib.EMA(ema1, timeperiod=period)
    ema3 = talib.EMA(ema2, timeperiod=period)
    finite = [values[np.isfinite(values)] for values in (ema1, ema2, ema3)]
    if any(values.size == 0 for values in finite):
        raise RuntimeError("ADX TEMA warmup is not complete at the checkpoint")
    return {
        "period": period,
        "seen": int(len(prefix)),
        "ema1": float(finite[0][-1]),
        "ema2": float(finite[1][-1]),
        "ema3": float(finite[2][-1]),
        "alpha": 2.0 / (period + 1.0),
    }


def advance_adx_tema(state: dict, plus_di, minus_di) -> np.ndarray:
    plus_di = np.asarray(plus_di, dtype=np.float64)
    minus_di = np.asarray(minus_di, dtype=np.float64)
    if plus_di.shape != minus_di.shape:
        raise ValueError("ADX directional-index arrays have different shapes")
    denominator = np.abs(plus_di) + np.abs(minus_di)
    dx = np.divide(
        100.0 * np.abs(plus_di - minus_di), denominator,
        out=np.zeros_like(denominator), where=denominator > 1e-12,
    )
    dx = np.clip(dx, 0.0, 100.0)
    out = np.full(len(dx), np.nan, dtype=np.float64)
    alpha = float(state["alpha"])
    for row, value in enumerate(dx):
        if not (np.isfinite(value) and np.isfinite(state["ema1"]) and np.isfinite(state["ema2"]) and np.isfinite(state["ema3"])):
            continue
        state["ema1"] = (float(value) - float(state["ema1"])) * alpha + float(state["ema1"])
        state["ema2"] = (float(state["ema1"]) - float(state["ema2"])) * alpha + float(state["ema2"])
        state["ema3"] = (float(state["ema2"]) - float(state["ema3"])) * alpha + float(state["ema3"])
        state["seen"] = int(state["seen"]) + 1
        out[row] = 3.0 * float(state["ema1"]) - 3.0 * float(state["ema2"]) + float(state["ema3"])
    return out


def seed_chaikin_adl(prefix: pd.DataFrame) -> tuple[dict, np.ndarray]:
    import talib

    values = talib.AD(
        prefix["High"].to_numpy(dtype=np.float64, copy=False),
        prefix["Low"].to_numpy(dtype=np.float64, copy=False),
        prefix["Close"].to_numpy(dtype=np.float64, copy=False),
        prefix["Volume"].to_numpy(dtype=np.float64, copy=False),
    )
    if not len(values):
        raise ValueError("Chaikin ADL checkpoint prefix is empty")
    return {"last_adl": float(values[-1]), "seen": int(len(values))}, values


def advance_chaikin_adl(state: dict, rows: pd.DataFrame) -> np.ndarray:
    high = rows["High"].to_numpy(dtype=np.float64, copy=False)
    low = rows["Low"].to_numpy(dtype=np.float64, copy=False)
    close = rows["Close"].to_numpy(dtype=np.float64, copy=False)
    volume = rows["Volume"].to_numpy(dtype=np.float64, copy=False)
    spread = high - low
    multiplier = np.divide(
        (close - low) - (high - close), spread,
        out=np.zeros_like(spread), where=spread != 0.0,
    )
    out = np.empty(len(rows), dtype=np.float64)
    previous = float(state["last_adl"])
    for index, value in enumerate(multiplier * volume):
        previous += float(value)
        out[index] = previous
    state["last_adl"] = previous
    state["seen"] = int(state["seen"]) + len(rows)
    return out


def compare_feature_arrays(actual, expected, *, atol: float, rtol: float) -> dict:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return {"status": "FAIL_SHAPE", "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    actual = actual.astype(np.float64, copy=False)
    expected = expected.astype(np.float64, copy=False)
    nan_mismatch = np.isnan(actual) != np.isnan(expected)
    posinf_mismatch = np.isposinf(actual) != np.isposinf(expected)
    neginf_mismatch = np.isneginf(actual) != np.isneginf(expected)
    finite = np.isfinite(actual) & np.isfinite(expected)
    difference = np.zeros(actual.shape, dtype=np.float64)
    difference[finite] = np.abs(actual[finite] - expected[finite])
    allowed = atol + rtol * np.abs(expected[finite])
    numeric_mismatch = np.zeros(actual.shape, dtype=bool)
    numeric_mismatch[finite] = difference[finite] > allowed
    mismatches = int(nan_mismatch.sum() + posinf_mismatch.sum() + neginf_mismatch.sum() + numeric_mismatch.sum())
    finite_errors = difference[finite]
    status = "PASS" if mismatches == 0 else "FAIL"
    return {
        "status": status,
        "rows": int(actual.shape[0]) if actual.ndim else 1,
        "numeric_mismatch_rows": int(numeric_mismatch.sum()),
        "nan_mask_mismatch_rows": int(nan_mismatch.sum()),
        "posinf_mask_mismatch_rows": int(posinf_mismatch.sum()),
        "neginf_mask_mismatch_rows": int(neginf_mismatch.sum()),
        "max_abs_error": float(finite_errors.max()) if finite_errors.size else 0.0,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def historical_flow_join_cast(
    target_opened,
    source_opened,
    source_values,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror E0041's exact-time search and float32 cast at the Arrow join site."""
    target = pd.DatetimeIndex(pd.to_datetime(target_opened, utc=True, errors="raise")).as_unit("ns")
    source = pd.DatetimeIndex(pd.to_datetime(source_opened, utc=True, errors="raise")).as_unit("ns")
    if target.has_duplicates or not target.is_monotonic_increasing:
        raise ValueError("historical FLOW join target keys must be unique and ordered")
    if source.has_duplicates or not source.is_monotonic_increasing:
        raise ValueError("historical FLOW join source keys must be unique and ordered")
    values = np.asarray(source_values)
    if values.ndim != 2 or values.shape[0] != len(source):
        raise ValueError("historical FLOW values do not align with their source keys")
    indexer = np.searchsorted(source.asi8, target.asi8)
    matched = indexer < len(source)
    valid_rows = np.flatnonzero(matched)
    matched[valid_rows] = source.asi8[indexer[valid_rows]] == target.asi8[valid_rows]
    aligned = np.full((len(target), values.shape[1]), np.nan, dtype=np.float32)
    aligned[matched] = np.asarray(values[indexer[matched]], dtype=np.float32)
    return aligned, matched
