from __future__ import annotations

import json
import inspect
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from e0037_trace_control import ROOT, process_measurements

sys.path.insert(0, str(ROOT))


CONTROL_ROOT = Path(__file__).resolve().parent
RUN_ROOT = CONTROL_ROOT / "runs" / "resume_cursor2340"
MARKET_START = pd.Timestamp("2026-10-05T12:00:00Z")
SLOT_PATH = ROOT / "data/analysis/polymarket/BTC/btc_D_historical_inputs_20261030_v4/slot_coverage.parquet"


def _resource_snapshot(prep) -> dict:
    return {
        "available_ram_bytes": int(prep.available_ram_bytes()),
        "D_free_bytes": int(shutil.disk_usage(ROOT).free),
    }


def _market_row() -> pd.DataFrame:
    dataset = ds.dataset(SLOT_PATH, format="parquet")
    timestamp_type = dataset.schema.field("market_start_utc").type
    start_scalar = pa.scalar(MARKET_START.to_pydatetime(), type=timestamp_type)
    columns = [
        "market_start_utc", "condition_id", "up_token_id", "down_token_id", "market_identity_status",
    ]
    frame = dataset.to_table(columns=columns, filter=ds.field("market_start_utc") == start_scalar).to_pandas()
    if len(frame) != 1 or frame.iloc[0]["market_identity_status"] != "CONFIRMED":
        raise RuntimeError("fixed cursor-control market is not one confirmed technical identity")
    return frame


def _install_phase_timers(prep, continuation) -> tuple[dict, object, object]:
    import pyarrow.parquet as parquet_module

    timers = {
        "parquet_open_metadata_seconds": 0.0,
        "batch_read_decompression_seconds": 0.0,
        "event_parse_filter_seconds": 0.0,
        "receive_group_state_update_seconds": 0.0,
        "snapshot_construction_seconds": 0.0,
        "checkpoint_load_seconds": 0.0,
        "checkpoint_save_seconds": 0.0,
        "post_loop_finalization_seconds": 0.0,
        "post_loop_finalization_measured": False,
    }
    original_parquet_file = parquet_module.ParquetFile

    class TimedParquetFile:
        def __init__(self, *args, **kwargs):
            started = time.perf_counter()
            self._inner = original_parquet_file(*args, **kwargs)
            timers["parquet_open_metadata_seconds"] += time.perf_counter() - started

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def iter_batches(self, *args, **kwargs):
            iterator = iter(self._inner.iter_batches(*args, **kwargs))
            while True:
                started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                timers["batch_read_decompression_seconds"] += time.perf_counter() - started
                yield batch

    parquet_module.ParquetFile = TimedParquetFile
    original_as_event_frame = prep._as_event_frame
    original_apply_receive_group = prep._apply_native_receive_group
    original_snapshot = prep._direct_book_snapshot
    original_load = continuation.load_checkpoint
    original_save = continuation.atomic_checkpoint

    def timed_parse(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_as_event_frame(*args, **kwargs)
        finally:
            timers["event_parse_filter_seconds"] += time.perf_counter() - started

    def timed_state_update(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_apply_receive_group(*args, **kwargs)
        finally:
            timers["receive_group_state_update_seconds"] += time.perf_counter() - started

    def timed_snapshot(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_snapshot(*args, **kwargs)
        finally:
            timers["snapshot_construction_seconds"] += time.perf_counter() - started

    def timed_load(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_load(*args, **kwargs)
        finally:
            timers["checkpoint_load_seconds"] += time.perf_counter() - started

    def timed_save(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_save(*args, **kwargs)
        finally:
            timers["checkpoint_save_seconds"] += time.perf_counter() - started

    prep._as_event_frame = timed_parse
    prep._apply_native_receive_group = timed_state_update
    prep._direct_book_snapshot = timed_snapshot
    continuation.load_checkpoint = timed_load
    continuation.atomic_checkpoint = timed_save

    source_lines, first_line = inspect.getsourcelines(prep.replay_native_books)
    finalization_line = next(
        (first_line + index for index, line in enumerate(source_lines)
         if line.strip() == "capture(int((end_hour + pd.Timedelta(hours=1)).value), inclusive=False)"),
        None,
    )
    replay_code = prep.replay_native_books.__code__
    finalization_started = None

    def trace(frame, event, _arg):
        nonlocal finalization_started
        if frame.f_code is replay_code:
            if event == "line" and frame.f_lineno == finalization_line and finalization_started is None:
                finalization_started = time.perf_counter()
            elif event == "return" and finalization_started is not None:
                timers["post_loop_finalization_seconds"] += time.perf_counter() - finalization_started
                timers["post_loop_finalization_measured"] = True
        return trace

    return timers, trace, original_parquet_file


def run_control(mode: str) -> int:
    if mode not in {"continuous", "interrupt", "resume", "profiled_continuous", "profiled_resume"}:
        raise ValueError("unknown fixed resume-control mode")
    run_dir = RUN_ROOT / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path, result_path = run_dir / "run.log", run_dir / "run.json"
    exit_path = run_dir / "exit_code.txt"
    t0, cpu0 = time.perf_counter(), time.process_time()
    read0, _ = process_measurements()
    prep = None
    record = {
        "mode": mode,
        "process_fresh": True,
        "market_start_utc": MARKET_START.isoformat().replace("+00:00", "Z"),
    }
    try:
        from features import btc_D_historical_input_preparation_20261030 as prep
        from features import btc_D_continuation_20261029 as continuation

        protocol, protocol_sha = prep.load_protocol()
        prep.RUNTIME = (
            RUN_ROOT / "interrupt" if mode in {"interrupt", "resume"}
            else run_dir / "runtime" if mode == "profiled_resume"
            else run_dir
        )
        record["runtime_path"] = str(prep.RUNTIME.relative_to(ROOT))
        record["resource_before"] = _resource_snapshot(prep)
        market_rows = _market_row()
        profile_timers = None
        if mode == "profiled_resume":
            source_cursor_checkpoint = RUN_ROOT / "interrupt/checkpoints/cursor24_checkpoint.json"
            destination_checkpoint = prep.RUNTIME / "checkpoints" / "native_book_replay.json"
            destination_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_cursor_checkpoint, destination_checkpoint)
            record["copied_authenticated_cursor24_checkpoint"] = str(source_cursor_checkpoint.relative_to(ROOT))
        if mode in {"profiled_continuous", "profiled_resume"}:
            profile_timers, trace, original_parquet_file = _install_phase_timers(prep, continuation)
            record["phase_timing_instrumentation"] = "pyarrow batch next, event parse/filter, receive-group update, snapshot, checkpoint read/write, and replay post-loop trace"
        _IO_AUDIT = {key: 0 for key in prep._IO_AUDIT}
        prep._IO_AUDIT.clear()
        prep._IO_AUDIT.update(_IO_AUDIT)
        original_atomic = continuation.atomic_checkpoint
        cursor_checkpoint = prep.RUNTIME / "checkpoints" / "cursor24_checkpoint.json"
        checkpoint_capture = {"captured": False}

        if mode in {"continuous", "interrupt"}:
            def atomic_with_cursor_capture(path: Path, payload: dict) -> str:
                digest = original_atomic(path, payload)
                if int(payload.get("hour_cursor", -1)) == 24:
                    shutil.copyfile(path, cursor_checkpoint)
                    checkpoint_capture["captured"] = True
                    record["cursor24_checkpoint_bytes"] = cursor_checkpoint.stat().st_size
                    record["cursor24_checkpoint_sha256"] = continuation.file_sha256(cursor_checkpoint)
                    if mode == "interrupt":
                        record.update({
                            "status": "INTENTIONALLY_INTERRUPTED_AFTER_PRODUCTION_ATOMIC_SAVE",
                            "checkpoint_hour_cursor": 24,
                            "measurement": {
                                "wall_seconds": time.perf_counter() - t0,
                                "cpu_seconds": time.process_time() - cpu0,
                                "process_read_transfer_bytes": process_measurements()[0] - read0,
                                "peak_working_set_bytes": process_measurements()[1],
                                "resource_after": _resource_snapshot(prep),
                            },
                            "io_audit": dict(prep._IO_AUDIT),
                            "exit_code": 86,
                        })
                        result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                        log_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                        exit_path.write_text("86\n", encoding="utf-8")
                        raise SystemExit(86)
                return digest
            continuation.atomic_checkpoint = atomic_with_cursor_capture

        replay_started = time.perf_counter()
        if profile_timers is not None:
            prior_trace = sys.gettrace()
            sys.settrace(trace)
            try:
                t60, t59, quality, replay_manifest = prep.replay_native_books(protocol, market_rows, protocol_sha)
            finally:
                sys.settrace(prior_trace)
                import pyarrow.parquet as parquet_module
                parquet_module.ParquetFile = original_parquet_file
        else:
            t60, t59, quality, replay_manifest = prep.replay_native_books(protocol, market_rows, protocol_sha)
        replay_seconds = time.perf_counter() - replay_started
        output_dir = CONTROL_ROOT / "runs" / "resume_cursor2340" / mode / "products"
        output_dir.mkdir(parents=True, exist_ok=True)
        t60.to_pickle(output_dir / "t60.pkl")
        t59.to_pickle(output_dir / "t59.pkl")
        quality.to_pickle(output_dir / "source_quality.pkl")
        record.update({
            "status": "COMPLETE",
            "protocol_sha256": protocol_sha,
            "market_identity_status": str(market_rows.iloc[0]["market_identity_status"]),
            "condition_id": str(market_rows.iloc[0]["condition_id"]),
            "t60_rows": len(t60),
            "t59_rows": len(t59),
            "source_quality_rows": len(quality),
            "cursor24_checkpoint_captured": checkpoint_capture["captured"],
            "replay_manifest": replay_manifest,
            "output_bytes": sum(path.stat().st_size for path in output_dir.glob("*.pkl")),
            "final_checkpoint_bytes": (prep.RUNTIME / "checkpoints" / "native_book_replay.json").stat().st_size,
            "io_audit": dict(prep._IO_AUDIT),
            "replay_wall_seconds": replay_seconds,
            "phase_timings": profile_timers,
            "measurement": {
                "wall_seconds": time.perf_counter() - t0,
                "cpu_seconds": time.process_time() - cpu0,
                "process_read_transfer_bytes": process_measurements()[0] - read0,
                "peak_working_set_bytes": process_measurements()[1],
                "resource_after": _resource_snapshot(prep),
            },
            "exit_code": 0,
        })
        result_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        log_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        exit_path.write_text("0\n", encoding="utf-8")
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        record.update({
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "io_audit": dict(prep._IO_AUDIT) if prep is not None else {},
            "measurement": {
                "wall_seconds": time.perf_counter() - t0,
                "cpu_seconds": time.process_time() - cpu0,
                "process_read_transfer_bytes": process_measurements()[0] - read0,
                "peak_working_set_bytes": process_measurements()[1],
                "resource_after": _resource_snapshot(prep) if prep is not None else None,
            },
            "exit_code": 1,
        })
        result_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        log_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        exit_path.write_text("1\n", encoding="utf-8")
        raise
