import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fit_indicators as fi
from common_config_utils import coerce_path
from features import ta_tools
from project_config import INDICATOR_FIT_CONFIG_PATH, build_indicator_fit_legacy_config

PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# CONFIG (edit here; no CLI parser)
# -----------------------------------------------------------------------------
FIT_CONFIG_PATH = INDICATOR_FIT_CONFIG_PATH
PAIR = None
INTERVAL = None

# None -> full dataset
ROWS_LIMIT = None

# Deterministic MA sweep across the search bounds used by indicators such as
# Integer(bounds=(2, 2000)).
MA_PERIOD_MIN = 2
MA_PERIOD_MAX = 2000
MA_PERIOD_STEP = 100
MA_WARMUP_REPEATS = 1
MA_REPEATS = 3
MA_SLOWEST_CASES_TOP_N = 10

SUMMARY_TOP_N = 20
DETAIL_TOP_N = 30
PRINT_DETAILS = False
JSON_OUT = None


class DatasetContext:
    __slots__ = ("pair", "interval", "data_path", "data_file")

    def __init__(self, pair, interval, data_path, data_file):
        self.pair = pair
        self.interval = interval
        self.data_path = data_path
        self.data_file = data_file

    def to_dict(self):
        return {
            "pair": self.pair,
            "interval": self.interval,
            "data_path": self.data_path,
            "data_file": self.data_file,
        }


def _serialize_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    return value


def default_json_report_path():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path("data") / "analysis" / "profile_ma_funcs" / f"profile_ma_funcs_{ts}.json"
    )


def _pick_single(values, requested, field_name):
    if requested:
        if requested not in values:
            raise ValueError(
                f"{field_name}='{requested}' not found. Available: {list(values)}"
            )
        return requested
    if not values:
        raise ValueError(f"No available values for {field_name}.")
    if len(values) > 1:
        raise ValueError(
            f"Multiple {field_name} candidates found: {list(values)}. "
            f"Set {field_name.upper()} constant at top of this file to disambiguate."
        )
    return str(values[0])


def resolve_dataset_context(
    config_path,
    pair,
    interval,
):
    if Path(config_path) != INDICATOR_FIT_CONFIG_PATH:
        raise ValueError(
            "Custom fit config path overrides are no longer supported. "
            f"Expected: {INDICATOR_FIT_CONFIG_PATH}"
        )
    cfg = build_indicator_fit_legacy_config()

    all_pairs = list(cfg.get("pairs", {}).keys())
    selected_pair = _pick_single(all_pairs, pair, "pair")
    pair_cfg = cfg["pairs"][selected_pair]

    intervals_cfg = pair_cfg.get("intervals", {})
    selected_interval = _pick_single(list(intervals_cfg.keys()), interval, "interval")
    interval_cfg = intervals_cfg[selected_interval]

    data_path = coerce_path(interval_cfg["data_path"])
    if not data_path.is_absolute():
        data_path = (PROJECT_ROOT / data_path).resolve()
    data_file = str(interval_cfg.get("data_file", "dataset.csv"))
    return DatasetContext(
        pair=selected_pair,
        interval=selected_interval,
        data_path=data_path,
        data_file=data_file,
    )


def load_ohlcv_array(
    context,
    rows,
):
    csv_path = context.data_path / context.data_file
    t0 = time.perf_counter()
    df = pd.read_csv(csv_path)
    csv_sec = time.perf_counter() - t0

    if rows is not None and rows > 0 and len(df) > rows:
        df = df.tail(int(rows)).copy()

    t1 = time.perf_counter()
    ohlcv_cols = fi._infer_ohlcv_cols(df)
    ohlcv_np = df[ohlcv_cols].to_numpy(dtype=np.float64, copy=False)
    arrays_sec = time.perf_counter() - t1

    stage_times = {
        "csv_load_sec": float(csv_sec),
        "array_prepare_sec": float(arrays_sec),
    }
    return ohlcv_np, stage_times, len(df)


def _build_ma_period_grid(
    *,
    period_min,
    period_max,
    period_step,
    max_allowed,
):
    lo = max(2, int(period_min))
    hi = min(int(period_max), int(max_allowed))
    if hi < lo:
        raise ValueError(
            f"Invalid MA period range after clipping: lo={lo}, hi={hi}, "
            f"max_allowed={max_allowed}"
        )

    step = max(1, int(period_step))
    periods = list(range(lo, hi + 1, step))
    if periods[-1] != hi:
        periods.append(hi)
    return periods


def _finite_float_values(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def benchmark_ma_funcs_direct(
    ohlcv_np,
    *,
    period_min,
    period_max,
    period_step,
    warmup_repeats,
    repeats,
    slowest_cases_top_n,
):
    if ohlcv_np.ndim != 2 or ohlcv_np.shape[1] < 5:
        raise ValueError("ohlcv_np must have shape (n, >=5).")

    close = np.asarray(ohlcv_np[:, 3], dtype=np.float64)
    volume = np.asarray(ohlcv_np[:, 4], dtype=np.float64)
    n = int(close.shape[0])
    if n < 3:
        raise ValueError("Need at least 3 rows to benchmark MA_FUNCS.")

    periods = _build_ma_period_grid(
        period_min=int(period_min),
        period_max=int(period_max),
        period_step=int(period_step),
        max_allowed=n - 1,
    )
    warmups = int(max(0, int(warmup_repeats)))
    reps = int(max(1, int(repeats)))
    top_n = int(max(1, int(slowest_cases_top_n)))
    rows = []
    global_case_rows = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for ma_name, func in ta_tools.MA_FUNCS.items():
            use_volume = ma_name in {"VWMA_PTA", "VWMA"}
            total_sec = 0.0
            total_calls = 0
            period_avg_ms_values = []
            finite_ratio_values = []
            ma_case_rows = []

            for period in periods:
                args = (
                    (close, int(period), volume) if use_volume else (close, int(period))
                )

                sample_out = None
                error_text = None
                measured_total_sec = 0.0
                measured_ok = 0

                for _ in range(warmups):
                    try:
                        sample_out = func(*args)
                    except Exception as exc:
                        error_text = f"{type(exc).__name__}: {exc}"
                        break

                if error_text is None:
                    for _ in range(reps):
                        try:
                            tr = time.perf_counter()
                            sample_out = func(*args)
                            measured_total_sec += time.perf_counter() - tr
                            measured_ok += 1
                        except Exception as exc:
                            error_text = f"{type(exc).__name__}: {exc}"
                            break

                if sample_out is not None:
                    try:
                        out_arr = np.asarray(sample_out, dtype=np.float64).reshape(-1)
                        finite_ratio = float(
                            np.isfinite(out_arr).sum() / max(1, out_arr.size)
                        )
                    except Exception:
                        finite_ratio = float("nan")
                else:
                    finite_ratio = float("nan")

                avg_call_ms = (
                    (measured_total_sec / measured_ok) * 1000.0
                    if measured_ok > 0
                    else float("nan")
                )
                case_row = {
                    "ma_type": ma_name,
                    "period": int(period),
                    "avg_call_ms": float(avg_call_ms),
                    "total_sec": float(measured_total_sec),
                    "measured_calls": int(measured_ok),
                    "finite_ratio": float(finite_ratio),
                    "error": error_text,
                }
                ma_case_rows.append(case_row)
                global_case_rows.append(case_row)

                if measured_ok > 0:
                    total_sec += float(measured_total_sec)
                    total_calls += int(measured_ok)
                    period_avg_ms_values.append(float(avg_call_ms))
                if np.isfinite(finite_ratio):
                    finite_ratio_values.append(float(finite_ratio))

            period_avg_arr = _finite_float_values(period_avg_ms_values)
            finite_ratio_arr = _finite_float_values(finite_ratio_values)
            slowest_cases = sorted(
                ma_case_rows,
                key=lambda row: (
                    1 if not np.isfinite(float(row["avg_call_ms"])) else 0,
                    (
                        -float(row["avg_call_ms"])
                        if np.isfinite(float(row["avg_call_ms"]))
                        else float("-inf")
                    ),
                ),
            )[:top_n]
            unique_errors = sorted(
                {str(row["error"]) for row in ma_case_rows if row.get("error")}
            )

            rows.append(
                {
                    "ma_type": ma_name,
                    "uses_volume": bool(use_volume),
                    "period_count": len(periods),
                    "period_min": int(periods[0]),
                    "period_max": int(periods[-1]),
                    "warmup_repeats": int(warmups),
                    "repeat_count_requested": int(reps),
                    "measured_calls": int(total_calls),
                    "total_sec": float(total_sec),
                    "avg_call_ms": (
                        float((total_sec / total_calls) * 1000.0)
                        if total_calls > 0
                        else float("nan")
                    ),
                    "period_avg_ms_mean": (
                        float(np.mean(period_avg_arr))
                        if period_avg_arr.size
                        else float("nan")
                    ),
                    "period_avg_ms_median": (
                        float(np.median(period_avg_arr))
                        if period_avg_arr.size
                        else float("nan")
                    ),
                    "period_avg_ms_p95": (
                        float(np.percentile(period_avg_arr, 95.0))
                        if period_avg_arr.size
                        else float("nan")
                    ),
                    "slowest_period": (
                        int(slowest_cases[0]["period"])
                        if slowest_cases
                        and np.isfinite(float(slowest_cases[0]["avg_call_ms"]))
                        else None
                    ),
                    "slowest_period_avg_ms": (
                        float(slowest_cases[0]["avg_call_ms"])
                        if slowest_cases
                        else float("nan")
                    ),
                    "periods_failed": int(
                        sum(1 for row in ma_case_rows if row.get("error"))
                    ),
                    "finite_ratio_mean": (
                        float(np.mean(finite_ratio_arr))
                        if finite_ratio_arr.size
                        else float("nan")
                    ),
                    "errors": unique_errors,
                    "slowest_cases": slowest_cases,
                }
            )

    rows.sort(
        key=lambda row: (
            1 if not np.isfinite(float(row["total_sec"])) else 0,
            (
                -float(row["total_sec"])
                if np.isfinite(float(row["total_sec"]))
                else float("-inf")
            ),
        )
    )
    global_slowest_cases = sorted(
        global_case_rows,
        key=lambda row: (
            1 if not np.isfinite(float(row["avg_call_ms"])) else 0,
            (
                -float(row["avg_call_ms"])
                if np.isfinite(float(row["avg_call_ms"]))
                else float("-inf")
            ),
        ),
    )[:top_n]
    return {
        "period_min_requested": int(period_min),
        "period_max_requested": int(period_max),
        "period_step": int(max(1, int(period_step))),
        "period_min_used": int(periods[0]),
        "period_max_used": int(periods[-1]),
        "period_count": len(periods),
        "warmup_repeats": int(warmups),
        "repeat_count_requested": int(reps),
        "slowest_cases": global_slowest_cases,
        "rows": rows,
    }


def main():
    context = resolve_dataset_context(
        config_path=FIT_CONFIG_PATH,
        pair=PAIR,
        interval=INTERVAL,
    )
    ohlcv_np, stage_times, rows_used = load_ohlcv_array(
        context=context,
        rows=ROWS_LIMIT,
    )

    ma_benchmark = benchmark_ma_funcs_direct(
        ohlcv_np=ohlcv_np,
        period_min=MA_PERIOD_MIN,
        period_max=MA_PERIOD_MAX,
        period_step=MA_PERIOD_STEP,
        warmup_repeats=MA_WARMUP_REPEATS,
        repeats=MA_REPEATS,
        slowest_cases_top_n=MA_SLOWEST_CASES_TOP_N,
    )
    report = {
        "context": _serialize_value(context.to_dict()),
        "run_params": {
            "rows_used": int(rows_used),
            "rows_limit": None if ROWS_LIMIT is None else int(ROWS_LIMIT),
            "ma_period_min": int(MA_PERIOD_MIN),
            "ma_period_max": int(MA_PERIOD_MAX),
            "ma_period_step": int(MA_PERIOD_STEP),
            "ma_warmup_repeats": int(MA_WARMUP_REPEATS),
            "ma_repeats": int(MA_REPEATS),
            "ma_slowest_cases_top_n": int(MA_SLOWEST_CASES_TOP_N),
        },
        "stages": stage_times,
        "ma_funcs_benchmark": ma_benchmark,
    }

    json_out = JSON_OUT if JSON_OUT is not None else default_json_report_path()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(_serialize_value(report), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    rows = ma_benchmark.get("rows", [])
    summary_top_n = max(1, int(SUMMARY_TOP_N))
    print("=== profile_ma_funcs summary ===")
    print(
        f"saved_report={json_out} | rows={rows_used} | pair={context.pair} | "
        f"interval={context.interval} | data={context.data_path / context.data_file}"
    )
    print(
        f"stages: csv_load={stage_times.get('csv_load_sec', 0.0):.3f}s, "
        f"array_prepare={stage_times.get('array_prepare_sec', 0.0):.3f}s"
    )
    print(
        "ma_sweep: "
        f"periods={ma_benchmark.get('period_min_used')}..{ma_benchmark.get('period_max_used')} "
        f"step={ma_benchmark.get('period_step')} "
        f"count={ma_benchmark.get('period_count')} "
        f"warmup={ma_benchmark.get('warmup_repeats')} "
        f"repeats={ma_benchmark.get('repeat_count_requested')}"
    )

    if rows:
        print("ma_funcs_total_sec(top):")
        for row in rows[:summary_top_n]:
            total_sec = row.get("total_sec")
            avg_ms = row.get("avg_call_ms")
            slowest_period = row.get("slowest_period")
            total_txt = f"{total_sec:.4f}" if np.isfinite(total_sec) else "nan"
            avg_txt = f"{avg_ms:.4f}" if np.isfinite(avg_ms) else "nan"
            slowest_txt = str(slowest_period) if slowest_period is not None else "n/a"
            print(
                f"  {row['ma_type']:16s} total={total_txt:>10s}s  "
                f"avg_call={avg_txt:>10s} ms  slowest_period={slowest_txt}"
            )

    if PRINT_DETAILS and rows:
        print("\n=== detailed profile output enabled ===")
        print("\n[detail] standalone MA_FUNCS sweep aggregate")
        for row in rows[: int(max(1, DETAIL_TOP_N))]:
            total_sec = row.get("total_sec")
            avg_ms = row.get("avg_call_ms")
            median_ms = row.get("period_avg_ms_median")
            p95_ms = row.get("period_avg_ms_p95")
            total_txt = f"{total_sec:.4f}" if np.isfinite(total_sec) else "nan"
            avg_txt = f"{avg_ms:.4f}" if np.isfinite(avg_ms) else "nan"
            median_txt = f"{median_ms:.4f}" if np.isfinite(median_ms) else "nan"
            p95_txt = f"{p95_ms:.4f}" if np.isfinite(p95_ms) else "nan"
            print(
                f"{row['ma_type']:16s} total={total_txt:>10s}s  "
                f"avg={avg_txt:>10s}ms  median={median_txt:>10s}ms  "
                f"p95={p95_txt:>10s}ms  periods={row['period_count']:4d}  "
                f"calls={row['measured_calls']:6d}"
            )
            for case in row.get("slowest_cases", [])[:3]:
                case_ms = case.get("avg_call_ms")
                case_ms_txt = f"{case_ms:.4f}" if np.isfinite(case_ms) else "nan"
                err_txt = f" error={case['error']}" if case.get("error") else ""
                print(
                    f"  period={int(case['period']):4d}  "
                    f"avg_call={case_ms_txt:>10s}ms  "
                    f"calls={int(case['measured_calls']):3d}{err_txt}"
                )

        global_slowest_cases = ma_benchmark.get("slowest_cases", [])
        if global_slowest_cases:
            print("\n[detail] standalone MA_FUNCS slowest ma+period cases")
            for case in global_slowest_cases[: int(max(1, DETAIL_TOP_N))]:
                case_ms = case.get("avg_call_ms")
                case_ms_txt = f"{case_ms:.4f}" if np.isfinite(case_ms) else "nan"
                err_txt = f" error={case['error']}" if case.get("error") else ""
                print(
                    f"{case['ma_type']:16s} period={int(case['period']):4d}  "
                    f"avg_call={case_ms_txt:>10s}ms  "
                    f"calls={int(case['measured_calls']):3d}{err_txt}"
                )


if __name__ == "__main__":
    main()
