from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from e0037_trace_control import ROOT, process_measurements

sys.path.insert(0, str(ROOT))
from features.btc_D_continuation_20261029 import atomic_checkpoint, load_checkpoint, state_digest


CONTROL = Path(__file__).resolve().parent
RUNS = CONTROL / "runs" / "resume_cursor2340"
SOURCE = RUNS / "interrupt/checkpoints/native_book_replay.json"
OUTPUT_DIR = CONTROL / "runs" / "atomic_checkpoint_retry"
OUTPUT = OUTPUT_DIR / "result.json"
LOG = OUTPUT_DIR / "run.log"
EXIT = OUTPUT_DIR / "exit_code.txt"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0, cpu0 = time.perf_counter(), time.process_time()
    read0, _ = process_measurements()
    source_header = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload = load_checkpoint(SOURCE, identity_sha256=source_header["identity_sha256"])
    test_path = OUTPUT_DIR / "atomic_retry_checkpoint.json"
    first_hash = atomic_checkpoint(test_path, payload)
    first_size = test_path.stat().st_size
    second_hash = atomic_checkpoint(test_path, payload)
    second_size = test_path.stat().st_size
    loaded = load_checkpoint(test_path, identity_sha256=source_header["identity_sha256"])
    if first_hash != second_hash or first_size != second_size or state_digest(payload) != state_digest(loaded):
        raise AssertionError("repeated production atomic checkpoint save was not idempotent")
    if list(OUTPUT_DIR.glob("atomic_retry_checkpoint.json.*.tmp")):
        raise AssertionError("atomic checkpoint temporary file remained after replace")
    read1, peak = process_measurements()
    result = {
        "status": "PASS",
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_identity_sha256": source_header["identity_sha256"],
        "first_save_sha256": first_hash,
        "second_save_sha256": second_hash,
        "first_save_bytes": first_size,
        "second_save_bytes": second_size,
        "exact_loaded_payload_equality": True,
        "temporary_files_remaining": 0,
        "measurement": {
            "wall_seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "process_read_transfer_bytes": read1 - read0,
            "peak_working_set_bytes": peak,
            "checkpoint_bytes_written": first_size + second_size,
        },
        "exit_code": 0,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    LOG.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    EXIT.write_text("0\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LOG.write_text(f"{type(exc).__name__}: {exc}\nexit_code=1\n", encoding="utf-8")
        EXIT.write_text("1\n", encoding="utf-8")
        raise
