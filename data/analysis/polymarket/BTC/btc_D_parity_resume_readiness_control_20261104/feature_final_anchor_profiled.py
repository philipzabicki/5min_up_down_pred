from __future__ import annotations

import json
import hashlib
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from e0037_trace_control import ROOT, process_measurements

sys.path.insert(0, str(ROOT))


CONTROL = Path(__file__).resolve().parent
RUN_DIR = CONTROL / "runs" / "feature_final_anchor_profiled"
RUNTIME = RUN_DIR / "runtime"
SOURCE_CHECKPOINT = ROOT / "data/analysis/polymarket/BTC/btc_D_historical_inputs_20261030_v4/checkpoints/feature_part_000002_state.json"
ANCHOR = pd.Timestamp("2026-10-05T11:58:00Z")


def _resource_snapshot(prep) -> dict:
    return {"available_ram_bytes": int(prep.available_ram_bytes()), "D_free_bytes": int(shutil.disk_usage(ROOT).free)}


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    t0, cpu0 = time.perf_counter(), time.process_time()
    read0, _ = process_measurements()
    prep = None
    record = {"status": "RUNNING", "anchor_opened_utc": ANCHOR.isoformat().replace("+00:00", "Z")}
    timers = {name: 0.0 for name in (
        "feature_checkpoint_load_seconds", "runtime_state_restore_seconds", "hybrid_read_seconds",
        "base64_build_seconds", "streak_state_seconds", "profile_state_seconds",
        "recursive_indicator_seconds", "flow_source_verify_read_seconds", "flow_compute_seconds",
        "dynamic_feature_seconds", "feature_compose_seconds", "dataframe_parquet_write_seconds",
        "atomic_checkpoint_save_seconds",
    )}
    try:
        from features import btc_D_historical_input_preparation_20261030 as prep
        from features import btc_D_continuation_20261029 as continuation
        import run_btc_D_continuation_parity_closure_20261029 as e47
        import features.btc_D_input_generator_20261029 as e45

        protocol, protocol_sha = prep.load_protocol()
        prep.RUNTIME = RUNTIME
        resource_before = _resource_snapshot(prep)
        if resource_before["available_ram_bytes"] < 16 * 1024**3:
            raise RuntimeError("profiled final feature control stopped: available physical RAM below 16 GiB")

        checkpoint_read_started = time.perf_counter()
        header = json.loads(SOURCE_CHECKPOINT.read_text(encoding="utf-8"))
        checkpoint = continuation.load_checkpoint(SOURCE_CHECKPOINT, identity_sha256=header["identity_sha256"])
        timers["feature_checkpoint_load_seconds"] += time.perf_counter() - checkpoint_read_started
        if checkpoint.get("protocol_sha256") != protocol_sha:
            raise RuntimeError("saved E0048 part-2 feature checkpoint uses a different frozen protocol")
        checkpoint_copy = RUNTIME / "checkpoints" / prep.CHECKPOINT_NAME
        checkpoint_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_CHECKPOINT, checkpoint_copy)
        start_opened = pd.Timestamp(checkpoint["next_expected_opened_utc"])
        delta = (ANCHOR - start_opened) / pd.Timedelta(minutes=1)
        if delta < 0 or not float(delta).is_integer():
            raise RuntimeError("final technical anchor is before or off the saved feature cursor")
        count = int(delta) + 1

        wrapped = []
        def instrument(owner, name, timer_name):
            original = getattr(owner, name)
            def measured(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    timers[timer_name] += time.perf_counter() - started
            setattr(owner, name, measured)
            wrapped.append((owner, name, original))

        instrument(prep, "_load_runtime_state", "runtime_state_restore_seconds")
        instrument(prep, "read_hybrid_range", "hybrid_read_seconds")
        instrument(prep, "_build_base64_chunk", "base64_build_seconds")
        instrument(continuation, "advance_streak_states", "streak_state_seconds")
        instrument(e47, "_profile_features_for_rows", "profile_state_seconds")
        instrument(e47, "_recursive_indicator_overrides", "recursive_indicator_seconds")
        instrument(prep, "read_flow_range", "flow_source_verify_read_seconds")
        instrument(e47, "_compute_flow_block", "flow_compute_seconds")
        instrument(e45, "generate_dynamic_block", "dynamic_feature_seconds")
        instrument(prep, "_compose_feature_rows", "feature_compose_seconds")
        instrument(continuation, "load_checkpoint", "feature_checkpoint_load_seconds")
        instrument(continuation, "atomic_checkpoint", "atomic_checkpoint_save_seconds")
        original_to_parquet = pd.DataFrame.to_parquet
        def timed_to_parquet(frame, *args, **kwargs):
            started = time.perf_counter()
            try:
                return original_to_parquet(frame, *args, **kwargs)
            finally:
                timers["dataframe_parquet_write_seconds"] += time.perf_counter() - started
        pd.DataFrame.to_parquet = timed_to_parquet

        try:
            generation_started = time.perf_counter()
            features, _next_checkpoint = prep._materialize_feature_chunk(
                protocol, protocol_sha, header["identity_sha256"], count,
                output_name="prefix_final_anchor_profiled", checkpoint_path=checkpoint_copy,
            )
            generation_seconds = time.perf_counter() - generation_started
        finally:
            pd.DataFrame.to_parquet = original_to_parquet
            for owner, name, original in reversed(wrapped):
                setattr(owner, name, original)

        if pd.Timestamp(features["Opened"].iloc[-1]) != ANCHOR:
            raise RuntimeError("profiled feature prefix did not stop at the preregistered anchor")
        feature_names = [column for column in features.columns if column not in {"Opened", "flow_source_join_matched"}]
        if len(feature_names) != 102:
            raise RuntimeError("profiled feature prefix did not emit exactly 102 features")
        row = features.iloc[-1:][["Opened", *feature_names]].copy()
        output = RUN_DIR / "generated_final_anchor.parquet"
        row.to_parquet(output, index=False, compression="zstd")
        read1, peak = process_measurements()
        result = {
            "status": "GENERATED",
            "protocol_sha256": protocol_sha,
            "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
            "source_checkpoint_identity_sha256": header["identity_sha256"],
            "source_row_index": checkpoint["source_row_index"],
            "source_byte_offset": checkpoint["source_byte_offset"],
            "start_opened_utc": start_opened.isoformat().replace("+00:00", "Z"),
            "anchor_opened_utc": ANCHOR.isoformat().replace("+00:00", "Z"),
            "rows_generated": count,
            "ordered_feature_count": len(feature_names),
            "ordered_feature_names_sha256": hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest(),
            "timers_seconds": timers,
            "generation_seconds": generation_seconds,
            "io_audit": dict(prep._IO_AUDIT),
            "raw_source_audit": json.loads((RUNTIME / "source_audit" / "prefix_final_anchor_profiled.json").read_text(encoding="utf-8")),
            "measurement": {
                "wall_seconds": time.perf_counter() - t0,
                "cpu_seconds": time.process_time() - cpu0,
                "process_read_transfer_bytes": read1 - read0,
                "peak_working_set_bytes": peak,
                "resource_before": resource_before,
                "resource_after": _resource_snapshot(prep),
                "generated_chunk_bytes": (RUNTIME / "minute_features" / "prefix_final_anchor_profiled.parquet").stat().st_size,
                "generated_anchor_bytes": output.stat().st_size,
                "runtime_bytes": sum(path.stat().st_size for path in RUNTIME.rglob("*") if path.is_file()),
            },
            "exit_code": 0,
        }
        (RUN_DIR / "run.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        (RUN_DIR / "run.log").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        (RUN_DIR / "exit_code.txt").write_text("0\n", encoding="utf-8")
        return 0
    except Exception as exc:
        record.update({
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "timers_seconds": timers,
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
        (RUN_DIR / "run.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        (RUN_DIR / "run.log").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        (RUN_DIR / "exit_code.txt").write_text("1\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
